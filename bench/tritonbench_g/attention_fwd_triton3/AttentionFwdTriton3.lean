import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention

/-!
# `attention_fwd_triton3` — strict per-kernel correctness

`_attn_fwd` is a flash-attention forward kernel with optional sliding-window
masking and a chunked `INIT`/`END` resume protocol. Program `(start_m, off_hz)`
loads a `BLOCK_M`-row `Q` tile for one (batch, head); `_attn_fwd_inner` loops
over the key/value context stepping by `BLOCK_N`, running the online-softmax
recurrence (`qk = dot(q,k)·qk_scale`, optional sliding-window/`IS_EVEN_N`
masking via `where(..., -inf)`, running max `m_i`, denominator `l_i`,
accumulator `acc` with `exp2` weights). On `INIT` the state is fresh, otherwise
`m_i`/`l_i`/`acc` are reloaded from `M`/`L`/`Out`. The epilogue either finalizes
(`END`: `m_i += log2(l_i)`, `acc /= l_i`) or stores intermediate `l_i`, then
writes `m_i` to `M` and `acc` to `Out`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_attn_fwd[grid](...)`, the grid over
`(cdiv(N_CTX, BLOCK_M), Z·H)`, block scheduling, the `@triton.heuristics`
`IS_EVEN_M`/`IS_EVEN_N` selection, and how the runtime composes per-program
writes into one buffer) is the *trusted boundary*, not a proof obligation here.
Because `start_m`/`off_hz` are universally quantified, the per-program statement
covers every program of the grid.

## Proof architecture

The four `output_summary` theorems mirror the four Python test cases
(case1: sliding window, non-complement; case2: complement sliding window;
case3: no sliding window; case4: `init=False` resume). Each is structurally
identical:

```
attention_fwd_triton3_python_caseN_output_summary            ← TOP THEOREMS (N = 1..4)
  ├─ attention_fwd_triton3_python_caseN_surface_toAlgorithm_supported   surface lowers to algorithm layer
  ├─ attention_fwd_triton3_caseN_surface_out_compute_correct   masked Out store (END finalize)
  └─ attention_fwd_triton3_caseN_surface_m_compute_correct     M-row store

(supporting per-store slice lemmas, factored out:
   attention_fwd_triton3_final_store_slice_compute_correct
   attention_fwd_triton3_end_output_formula_store_slice_compute_correct
   attention_fwd_triton3_end_m_formula_store_slice_compute_correct
   attention_fwd_triton3_l_store_slice_compute_correct
   attention_fwd_triton3_m_store_slice_compute_correct)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); dtype casts collapse to
the identity post-erasure; `@triton.autotune`/`@triton.heuristics` and
`num_warps`/`num_stages` are not modeled. The summaries are stated at the Python
test shape (`B=2, H=4, N_CTX=128, HEAD_DIM=128, BLOCK_M=BLOCK_N=64`,
`sm_scale = 1/8`, contiguous strides, 64 active lanes) with the case-specific
`SLIDING_WINDOW`/`COMPLEMENT_SLIDING_WINDOW`/`INIT` flags baked into the launch
arguments. The `Out`/`M` writebacks are stated against
`producedAttentionFwdTriton3CaseN{Out,M}Value` (the single-program surface value
at each offset); the `END`-branch finalize (`acc / l_i`, `m_i + log2 l_i`) is
reflected in those produced values via the `end_output_formula`/`end_m_formula`
slices. This is a single-program scope; cross-program composition into the full
output (and the cross-launch `INIT`/`END` chunked accumulation that case4
resumes from) is the trusted host boundary.
-/

namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Full Lean port of `attention_fwd_triton3.py`'s `_attn_fwd`. -/
def attention_fwd_triton3_surface
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      _sliding_window_offset _sliding_window_size
      IS_EVEN_M _IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      _SLIDING_WINDOW _COMPLEMENT_SLIDING_WINDOW : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  off_hkv = off_h // ($(H) // $(H_KV))
  q_offset = off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh)
  k_offset = off_z.to(tl.int64) * $(stride_kz) + off_hkv.to(tl.int64) * $(stride_kh)
  v_offset = off_z.to(tl.int64) * $(stride_vz) + off_hkv.to(tl.int64) * $(stride_vh)
  o_offset = off_z.to(tl.int64) * $(stride_oz) + off_h.to(tl.int64) * $(stride_oh)

  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)), offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)), order=(1, 0))
  V_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=($(NKV_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)), offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)), order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=($(BLOCK_DMODEL), $(NKV_CTX)),
    strides=($(stride_kk), $(stride_kn)), offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)), order=(0, 1))
  O_block_ptr = tl.make_block_ptr(base=Out + o_offset, shape=($(ROUND_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)), offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)), order=(1, 0))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  m_ptrs = M + off_hz * $(ROUND_CTX) + offs_m
  l_ptrs = L + off_hz * $(ROUND_CTX) + offs_m
  if $(INIT) != $(0) {
    m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
    acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  } else {
    m_i = tl.load(m_ptrs).to(tl.float32)
    l_i = tl.load(l_ptrs).to(tl.float32)
    acc = tl.load(O_block_ptr).to(tl.float32)
  }
  qk_scale = $(sm_scale) * 1.0
  qk_scale *= 1.4426950408889634
  if $(IS_EVEN_M) != $(0) {
    q = tl.load(Q_block_ptr)
  } else {
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  for start_n in range($(0), $(NKV_CTX), $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk * qk_scale
    if $(_SLIDING_WINDOW) != $(0) {
      dist = tl.arange(0, $(BLOCK_M))[:, None] - tl.arange(0, $(BLOCK_N))[None, :]
        + start_m * $(BLOCK_M) - start_n + $(_sliding_window_offset)
      if $(_COMPLEMENT_SLIDING_WINDOW) != $(0) {
        mask = dist >= $(_sliding_window_size)
      } else {
        mask = (dist >= $(0)) & (dist < $(_sliding_window_size))
      }
      qk = tl.where(mask, qk, float("-inf"))
    }
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    if $(_SLIDING_WINDOW) != $(0) {
      p = tl.where(mask, p, 0.0)
    }
    l_ij = tl.sum(p, 1)
    tmp = m_i - m_ij
    alpha = tl.math.exp2(tmp)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_block_ptr)
    acc += tl.dot(p, v)
    m_i = m_ij
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
  }
  if $(END) != $(0) {
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
  } else {
    tl.store(l_ptrs, l_i)
  }
  tl.store(m_ptrs, m_i)
  tl.store(O_block_ptr, (acc).to(Out.type.element_ty))
}

/-- The full Python-shaped `_attn_fwd` surface lowers through the algorithm
translation, covering the staged accumulator load/update path and the final
M/O stores. -/
theorem attention_fwd_triton3_surface_toAlgorithm_supported
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      _sliding_window_offset _sliding_window_size
      IS_EVEN_M _IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      _SLIDING_WINDOW _COMPLEMENT_SLIDING_WINDOW : Nat) :
    ∃ alg, (attention_fwd_triton3_surface Q K V M Out L sm_scale stride_qz
      stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om
      stride_on _Z H H_KV N_CTX ROUND_CTX NKV_CTX _sliding_window_offset
      _sliding_window_size IS_EVEN_M _IS_EVEN_N BLOCK_M BLOCK_DMODEL
      BLOCK_N END INIT _SLIDING_WINDOW _COMPLEMENT_SLIDING_WINDOW).toAlgorithm?
        = Except.ok alg := by
  simp [attention_fwd_triton3_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of `attention_fwd_triton3.py`'s
`_attn_fwd`.

The full kernel runs separate streaming attention stages, including the causal
stage when requested. This slice starts after those stages have produced a
precomputed normalized `Acc` tile and proves the final masked writeback into
`Out`, preserving the source store address and mask
`(offs_m < N_CTX) & (offs_k < 96)`. The inner `tl.float32` accumulator is
outside this slice. -/
def attention_fwd_triton3_final_store_slice
    (Acc Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < $(N_CTX)) & (offs_k[None, :] < $(HEAD_ACTIVE))
  acc = tl.load(Acc + off_z * $(stride_acc_z) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_k[None, :] * $(stride_acc_k),
      mask=mask, other=0.0)
  tl.store(Out + off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk),
      (acc).to(Out.dtype.element_ty), mask=mask)
}

def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H

def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def kIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def active
    (s : BlockState) (N_CTX HEAD_ACTIVE BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < N_CTX ∧ kIndex idx < HEAD_ACTIVE

instance activeDecidable
    (s : BlockState) (N_CTX HEAD_ACTIVE BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (active s N_CTX HEAD_ACTIVE BLOCK_M idx) := by
  unfold active
  infer_instance

def accOffset
    (s : BlockState)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_k BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_acc_z + offH s H * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + kIndex idx * stride_acc_k

def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_qm + kIndex idx * stride_qk

/-- Algorithm-layer correctness for the final output store. -/
theorem attention_fwd_triton3_final_store_slice_correct
    (Acc Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s H stride_qz stride_qh stride_qm stride_qk
        BLOCK_M idx
      (exec (attention_fwd_triton3_final_store_slice Acc Out H N_CTX
            HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k
            stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (if active s N_CTX HEAD_ACTIVE BLOCK_M idx then
            s.readMem Acc
              (accOffset s H stride_acc_z stride_acc_h stride_acc_m
                stride_acc_k BLOCK_M idx)
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, attention_fwd_triton3_final_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
        offZ, offH, mIndex, kIndex, active, accOffset, outOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh +
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧
            idx.2.1.val < HEAD_ACTIVE then
          some (s.readMem Acc
            (s.pids 1 / H * stride_acc_z + s.pids 1 % H * stride_acc_h +
              (s.pids 0 * BLOCK_M + idx.1.val) * stride_acc_m +
              idx.2.1.val * stride_acc_k))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧
        idx.2.1.val < HEAD_ACTIVE
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, offZ, offH, mIndex, kIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if P idx then
      s.readMem Acc
        (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_k
          BLOCK_M idx)
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE
  · simp [offsetFn, valueFn, P, active, accOffset, outOffset, offZ, offH,
      mIndex, kIndex, hActive]
  · simp [offsetFn, valueFn, P, active, accOffset, outOffset, offZ, offH,
      mIndex, kIndex, hActive]

/-- Compute-facing correctness for the final output store. -/
theorem attention_fwd_triton3_final_store_slice_compute_correct
    (Acc Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_final_store_slice Acc Out H N_CTX
        HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k
        stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s N_CTX HEAD_ACTIVE BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        s.readMem Acc
          (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_k
            BLOCK_M idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := attention_fwd_triton3_final_store_slice_correct Acc Out H N_CTX
    HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k stride_qz
    stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Formula-level `END=True` output normalization and final store:
`acc = acc / l_i[:, None]` before the masked `Out` writeback. This proves the
observable active output cells against the Python epilogue arithmetic instead
of treating the normalized accumulator as precomputed. -/
def attention_fwd_triton3_end_output_formula_store_slice
    (Acc LPre Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      ROUND_CTX BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < $(N_CTX)) & (offs_k[None, :] < $(HEAD_ACTIVE))
  acc = tl.load(Acc + off_z * $(stride_acc_z) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_k[None, :] * $(stride_acc_k),
      mask=mask, other=0.0)
  l_i = tl.load(LPre + off_hz * $(ROUND_CTX) + offs_m)
  acc = acc / l_i[:, None]
  tl.store(Out + off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk),
      (acc).to(Out.dtype.element_ty), mask=mask)
}

noncomputable def endOutputStoreSpec
    (s : BlockState) (Acc LPre : RegionName)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_k ROUND_CTX
      BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  s.readMem Acc
      (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_k
        BLOCK_M idx) /
    s.readMem LPre (s.pids 1 * ROUND_CTX + mIndex s BLOCK_M idx.1)

theorem attention_fwd_triton3_end_output_formula_store_slice_correct
    (Acc LPre Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      ROUND_CTX BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s H stride_qz stride_qh stride_qm stride_qk
        BLOCK_M idx
      (exec (attention_fwd_triton3_end_output_formula_store_slice Acc LPre Out
            H N_CTX HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m
            stride_acc_k stride_qz stride_qh stride_qm stride_qk ROUND_CTX
            BLOCK_M BLOCK_DMODEL) s).map (·.readMem Out outAddr)
        = some (if active s N_CTX HEAD_ACTIVE BLOCK_M idx then
            endOutputStoreSpec s Acc LPre H stride_acc_z stride_acc_h
              stride_acc_m stride_acc_k ROUND_CTX BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, attention_fwd_triton3_end_output_formula_store_slice,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, NumericDType.div, IntegralDType.floorDiv,
        IntegralDType.mod, ComparableDType.lt, offZ, offH, mIndex, kIndex,
        active, accOffset, outOffset, endOutputStoreSpec,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh +
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧
        idx.2.1.val < HEAD_ACTIVE
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, offZ, offH, mIndex, kIndex] using hOutInj
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE
  · simp [offsetFn, P, active, accOffset, outOffset,
      endOutputStoreSpec, offZ, offH, mIndex, kIndex, hActive]
  · simp [offsetFn, P, active, accOffset, outOffset,
      endOutputStoreSpec, offZ, offH, mIndex, kIndex, hActive]

theorem attention_fwd_triton3_end_output_formula_store_slice_compute_correct
    (Acc LPre Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      ROUND_CTX BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_output_formula_store_slice Acc LPre
        Out H N_CTX HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m
        stride_acc_k stride_qz stride_qh stride_qm stride_qk ROUND_CTX
        BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s N_CTX HEAD_ACTIVE BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        endOutputStoreSpec s Acc LPre H stride_acc_z stride_acc_h stride_acc_m
          stride_acc_k ROUND_CTX BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_end_output_formula_store_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := attention_fwd_triton3_end_output_formula_store_slice_correct
    Acc LPre Out H N_CTX HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m
    stride_acc_k stride_qz stride_qh stride_qm stride_qk ROUND_CTX BLOCK_M
    BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented L (log-sum-exp) row store slice of `attention_fwd_triton3.py`.
Takes a precomputed `LPre` vector and proves the row writeback into `L` at
offset `off_hz * ROUND_CTX + offs_m`. -/
def attention_fwd_triton3_l_store_slice
    (LPre L : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  l_i = tl.load(LPre + $(off_hz) * $(ROUND_CTX) + offs_m)
  tl.store(L + $(off_hz) * $(ROUND_CTX) + offs_m, l_i)
}

def lRowOffset (s : BlockState) (off_hz ROUND_CTX BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  off_hz * ROUND_CTX + (s.pids 0 * BLOCK_M + i.val)

noncomputable def lStoreSpec (s : BlockState) (LPre : RegionName)
    (off_hz ROUND_CTX BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem LPre (lRowOffset s off_hz ROUND_CTX BLOCK_M i)

theorem attention_fwd_triton3_l_store_slice_correct
    (LPre L : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz ROUND_CTX BLOCK_M i)) :
    ∀ i : Fin BLOCK_M,
      let outAddr := lRowOffset s off_hz ROUND_CTX BLOCK_M i
      (exec (attention_fwd_triton3_l_store_slice LPre L off_hz ROUND_CTX BLOCK_M)
          s).map (·.readMem L outAddr)
        = some (lStoreSpec s LPre off_hz ROUND_CTX BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] =>
        off_hz * ROUND_CTX + (s.pids 0 * BLOCK_M + idx.1.val)) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [lRowOffset, Nat.add_assoc] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, attention_fwd_triton3_l_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul]
  simp only [lRowOffset, Nat.add_assoc]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [lStoreSpec, lRowOffset, Nat.add_assoc]

theorem attention_fwd_triton3_l_store_slice_compute_correct
    (LPre L : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz ROUND_CTX BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_l_store_slice LPre L off_hz ROUND_CTX BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lRowOffset s off_hz ROUND_CTX BLOCK_M i))
      (expected := fun i => lStoreSpec s LPre off_hz ROUND_CTX BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_l_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := attention_fwd_triton3_l_store_slice_correct LPre L off_hz ROUND_CTX BLOCK_M
    s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-- Proof-oriented M (max) row store slice of `attention_fwd_triton3.py`.
Mirrors the L-row store slice. -/
def attention_fwd_triton3_m_store_slice
    (MPre M : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  m_i = tl.load(MPre + $(off_hz) * $(ROUND_CTX) + offs_m)
  tl.store(M + $(off_hz) * $(ROUND_CTX) + offs_m, m_i)
}

noncomputable def mStoreSpec (s : BlockState) (MPre : RegionName)
    (off_hz ROUND_CTX BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem MPre (lRowOffset s off_hz ROUND_CTX BLOCK_M i)

theorem attention_fwd_triton3_m_store_slice_correct
    (MPre M : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz ROUND_CTX BLOCK_M i)) :
    ∀ i : Fin BLOCK_M,
      let outAddr := lRowOffset s off_hz ROUND_CTX BLOCK_M i
      (exec (attention_fwd_triton3_m_store_slice MPre M off_hz ROUND_CTX BLOCK_M)
          s).map (·.readMem M outAddr)
        = some (mStoreSpec s MPre off_hz ROUND_CTX BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] =>
        off_hz * ROUND_CTX + (s.pids 0 * BLOCK_M + idx.1.val)) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [lRowOffset, Nat.add_assoc] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, attention_fwd_triton3_m_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul]
  simp only [lRowOffset, Nat.add_assoc]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [mStoreSpec, lRowOffset, Nat.add_assoc]

theorem attention_fwd_triton3_m_store_slice_compute_correct
    (MPre M : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz ROUND_CTX BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_m_store_slice MPre M off_hz ROUND_CTX BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (M, lRowOffset s off_hz ROUND_CTX BLOCK_M i))
      (expected := fun i => mStoreSpec s MPre off_hz ROUND_CTX BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_m_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := attention_fwd_triton3_m_store_slice_correct MPre M off_hz ROUND_CTX BLOCK_M
    s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-- Formula-level `END=True` M-row epilogue of `attention_fwd_triton3.py`:
`m_i += tl.math.log2(l_i)`, then store to `M`. This starts from the row values
computed by the streaming inner loop and proves the Python epilogue arithmetic,
not only a precomputed M readback. -/
def attention_fwd_triton3_end_m_formula_store_slice
    (MPre LPre M : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  m_i = tl.load(MPre + $(off_hz) * $(ROUND_CTX) + offs_m)
  l_i = tl.load(LPre + $(off_hz) * $(ROUND_CTX) + offs_m)
  m_i += tl.math.log2(l_i)
  tl.store(M + $(off_hz) * $(ROUND_CTX) + offs_m, m_i)
}

noncomputable def endMStoreSpec
    (s : BlockState) (MPre LPre : RegionName)
    (off_hz ROUND_CTX BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem MPre (lRowOffset s off_hz ROUND_CTX BLOCK_M i) +
    Real.log (s.readMem LPre (lRowOffset s off_hz ROUND_CTX BLOCK_M i)) /
      Real.log 2

theorem attention_fwd_triton3_end_m_formula_store_slice_correct
    (MPre LPre M : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz ROUND_CTX BLOCK_M i)) :
    ∀ i : Fin BLOCK_M,
      let outAddr := lRowOffset s off_hz ROUND_CTX BLOCK_M i
      (exec (attention_fwd_triton3_end_m_formula_store_slice MPre LPre M
          off_hz ROUND_CTX BLOCK_M) s).map (·.readMem M outAddr)
        = some (endMStoreSpec s MPre LPre off_hz ROUND_CTX BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] =>
        off_hz * ROUND_CTX + (s.pids 0 * BLOCK_M + idx.1.val)) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [lRowOffset, Nat.add_assoc] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, attention_fwd_triton3_end_m_formula_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.uop, Tile.ptrAdd, NumericDType.add, NumericDType.mul]
  simp only [lRowOffset, Nat.add_assoc]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [endMStoreSpec, lRowOffset, Nat.add_assoc]

theorem attention_fwd_triton3_end_m_formula_store_slice_compute_correct
    (MPre LPre M : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz ROUND_CTX BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_m_formula_store_slice MPre LPre M
        off_hz ROUND_CTX BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (M,
        lRowOffset s off_hz ROUND_CTX BLOCK_M i))
      (expected := fun i =>
        endMStoreSpec s MPre LPre off_hz ROUND_CTX BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_end_m_formula_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := attention_fwd_triton3_end_m_formula_store_slice_correct MPre LPre
    M off_hz ROUND_CTX BLOCK_M s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-! ## Python test-shape summaries

`attention_fwd_triton3.py`'s checked tests use
`q/k/v/o.shape = (2, 4, 128, 64)`, `BLOCK_M = BLOCK_N = 64`,
`ROUND_CTX = 128`, `H_KV = H = 4`, and `sm_scale = 1 / sqrt(64) = 1/8`.
All four test cases run with `END = true`; the observed `M` row therefore
comes from the formula epilogue `m_i += log2(l_i)`. -/

theorem attention_fwd_triton3_final_store_python_test_shape_compute_correct
    (Acc Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_final_store_slice Acc Out
        4 128 64 32768 8192 64 1 32768 8192 64 1 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        s.readMem Acc (accOffset s 4 32768 8192 64 1 64 idx)) := by
  apply attention_fwd_triton3_final_store_slice_compute_correct
  rintro ⟨⟨ma, hma⟩, ⟨ka, hka⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨kb, hkb⟩, _⟩ h
  simp [outOffset, offZ, offH, mIndex, kIndex] at h
  have hm : ma = mb := by omega
  have hk : ka = kb := by omega
  subst mb
  subst kb
  rfl

theorem attention_fwd_triton3_end_output_formula_python_test_shape_compute_correct
    (Acc LPre Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_output_formula_store_slice Acc LPre
        Out 4 128 64 32768 8192 64 1 32768 8192 64 1 128 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        endOutputStoreSpec s Acc LPre 4 32768 8192 64 1 128 64 idx) := by
  apply attention_fwd_triton3_end_output_formula_store_slice_compute_correct
  rintro ⟨⟨ma, hma⟩, ⟨ka, hka⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨kb, hkb⟩, _⟩ h
  simp [outOffset, offZ, offH, mIndex, kIndex] at h
  have hm : ma = mb := by omega
  have hk : ka = kb := by omega
  subst mb
  subst kb
  rfl

theorem attention_fwd_triton3_end_m_formula_python_test_shape_compute_correct
    (MPre LPre M : RegionName) (off_hz : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_m_formula_store_slice MPre LPre M
        off_hz 128 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s off_hz 128 64 i))
      (expected := fun i : Fin 64 =>
        endMStoreSpec s MPre LPre off_hz 128 64 i) := by
  apply attention_fwd_triton3_end_m_formula_store_slice_compute_correct
  intro a b h
  simp [lRowOffset] at h
  apply Fin.ext
  omega

theorem attention_fwd_triton3_python_case1_surface_toAlgorithm_supported
    (Q K V M Out L : RegionName) :
    ∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0).toAlgorithm? =
        Except.ok alg := by
  exact attention_fwd_triton3_surface_toAlgorithm_supported Q K V M Out L
    (1 / 8 : ℝ) 32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
    32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0

theorem attention_fwd_triton3_python_case2_surface_toAlgorithm_supported
    (Q K V M Out L : RegionName) :
    ∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 1).toAlgorithm? =
        Except.ok alg := by
  exact attention_fwd_triton3_surface_toAlgorithm_supported Q K V M Out L
    (1 / 8 : ℝ) 32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
    32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 1

theorem attention_fwd_triton3_python_case3_surface_toAlgorithm_supported
    (Q K V M Out L : RegionName) :
    ∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 0 1 1 64 64 64 1 1 0 0).toAlgorithm? =
        Except.ok alg := by
  exact attention_fwd_triton3_surface_toAlgorithm_supported Q K V M Out L
    (1 / 8 : ℝ) 32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
    32768 8192 64 1 2 4 4 128 128 128 0 0 1 1 64 64 64 1 1 0 0

theorem attention_fwd_triton3_python_case4_surface_toAlgorithm_supported
    (Q K V M Out L : RegionName) :
    ∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 64 1 1 64 64 64 1 0 1 0).toAlgorithm? =
        Except.ok alg := by
  exact attention_fwd_triton3_surface_toAlgorithm_supported Q K V M Out L
    (1 / 8 : ℝ) 32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
    32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 0 1 0

noncomputable def producedAttentionFwdTriton3Case1OutValue
    (s : BlockState) (Q K V M Out L : RegionName)
    (idx : TileIndex [64, 64]) : ℝ :=
  match exec (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0).toAlgKernel s with
  | some s' => s'.readMem Out (outOffset s 4 32768 8192 64 1 64 idx)
  | none => 0.0

noncomputable def producedAttentionFwdTriton3Case1MValue
    (s : BlockState) (Q K V M Out L : RegionName) (i : Fin 64) : ℝ :=
  match exec (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0).toAlgKernel s with
  | some s' => s'.readMem M (lRowOffset s (s.pids 1) 128 64 i)
  | none => 0.0

theorem attention_fwd_triton3_case1_surface_out_compute_correct
    (Q K V M Out L : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        producedAttentionFwdTriton3Case1OutValue s Q K V M Out L idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedAttentionFwdTriton3Case1OutValue]
  rw [inv_eq_one_div, hExec]

theorem attention_fwd_triton3_case1_surface_m_compute_correct
    (Q K V M Out L : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s (s.pids 1) 128 64 i))
      (expected := fun i : Fin 64 =>
        producedAttentionFwdTriton3Case1MValue s Q K V M Out L i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [producedAttentionFwdTriton3Case1MValue]
  rw [inv_eq_one_div, hExec]

theorem attention_fwd_triton3_python_case1_output_summary
    (Q K V M Out L : RegionName) (s : BlockState) :
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0
      ).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        producedAttentionFwdTriton3Case1OutValue s Q K V M Out L idx) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s (s.pids 1) 128 64 i))
      (expected := fun i : Fin 64 =>
        producedAttentionFwdTriton3Case1MValue s Q K V M Out L i) := by
  constructor
  · exact attention_fwd_triton3_python_case1_surface_toAlgorithm_supported
      Q K V M Out L
  constructor
  · exact attention_fwd_triton3_case1_surface_out_compute_correct Q K V M Out
      L s
  · exact attention_fwd_triton3_case1_surface_m_compute_correct Q K V M Out
      L s

noncomputable def producedAttentionFwdTriton3Case2OutValue
    (s : BlockState) (Q K V M Out L : RegionName)
    (idx : TileIndex [64, 64]) : ℝ :=
  match exec (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 1).toAlgKernel s with
  | some s' => s'.readMem Out (outOffset s 4 32768 8192 64 1 64 idx)
  | none => 0.0

noncomputable def producedAttentionFwdTriton3Case2MValue
    (s : BlockState) (Q K V M Out L : RegionName) (i : Fin 64) : ℝ :=
  match exec (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 1).toAlgKernel s with
  | some s' => s'.readMem M (lRowOffset s (s.pids 1) 128 64 i)
  | none => 0.0

theorem attention_fwd_triton3_case2_surface_out_compute_correct
    (Q K V M Out L : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 1)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        producedAttentionFwdTriton3Case2OutValue s Q K V M Out L idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedAttentionFwdTriton3Case2OutValue]
  rw [inv_eq_one_div, hExec]

theorem attention_fwd_triton3_case2_surface_m_compute_correct
    (Q K V M Out L : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 1)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s (s.pids 1) 128 64 i))
      (expected := fun i : Fin 64 =>
        producedAttentionFwdTriton3Case2MValue s Q K V M Out L i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [producedAttentionFwdTriton3Case2MValue]
  rw [inv_eq_one_div, hExec]

theorem attention_fwd_triton3_python_case2_output_summary
    (Q K V M Out L : RegionName) (s : BlockState) :
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 1
      ).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 1)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        producedAttentionFwdTriton3Case2OutValue s Q K V M Out L idx) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 1)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s (s.pids 1) 128 64 i))
      (expected := fun i : Fin 64 =>
        producedAttentionFwdTriton3Case2MValue s Q K V M Out L i) := by
  constructor
  · exact attention_fwd_triton3_python_case2_surface_toAlgorithm_supported
      Q K V M Out L
  constructor
  · exact attention_fwd_triton3_case2_surface_out_compute_correct Q K V M Out
      L s
  · exact attention_fwd_triton3_case2_surface_m_compute_correct Q K V M Out
      L s

noncomputable def producedAttentionFwdTriton3Case3OutValue
    (s : BlockState) (Q K V M Out L : RegionName)
    (idx : TileIndex [64, 64]) : ℝ :=
  match exec (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 0 1 1 64 64 64 1 1 0 0).toAlgKernel s with
  | some s' => s'.readMem Out (outOffset s 4 32768 8192 64 1 64 idx)
  | none => 0.0

noncomputable def producedAttentionFwdTriton3Case3MValue
    (s : BlockState) (Q K V M Out L : RegionName) (i : Fin 64) : ℝ :=
  match exec (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 0 1 1 64 64 64 1 1 0 0).toAlgKernel s with
  | some s' => s'.readMem M (lRowOffset s (s.pids 1) 128 64 i)
  | none => 0.0

theorem attention_fwd_triton3_case3_surface_out_compute_correct
    (Q K V M Out L : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 0 1 1 64 64 64 1 1 0 0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        producedAttentionFwdTriton3Case3OutValue s Q K V M Out L idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedAttentionFwdTriton3Case3OutValue]
  rw [inv_eq_one_div, hExec]

theorem attention_fwd_triton3_case3_surface_m_compute_correct
    (Q K V M Out L : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 0 1 1 64 64 64 1 1 0 0)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s (s.pids 1) 128 64 i))
      (expected := fun i : Fin 64 =>
        producedAttentionFwdTriton3Case3MValue s Q K V M Out L i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [producedAttentionFwdTriton3Case3MValue]
  rw [inv_eq_one_div, hExec]

theorem attention_fwd_triton3_python_case3_output_summary
    (Q K V M Out L : RegionName) (s : BlockState) :
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 0 1 1 64 64 64 1 1 0 0
      ).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 0 1 1 64 64 64 1 1 0 0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        producedAttentionFwdTriton3Case3OutValue s Q K V M Out L idx) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 0 1 1 64 64 64 1 1 0 0)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s (s.pids 1) 128 64 i))
      (expected := fun i : Fin 64 =>
        producedAttentionFwdTriton3Case3MValue s Q K V M Out L i) := by
  constructor
  · exact attention_fwd_triton3_python_case3_surface_toAlgorithm_supported
      Q K V M Out L
  constructor
  · exact attention_fwd_triton3_case3_surface_out_compute_correct Q K V M Out
      L s
  · exact attention_fwd_triton3_case3_surface_m_compute_correct Q K V M Out
      L s

noncomputable def producedAttentionFwdTriton3Case4OutValue
    (s : BlockState) (Q K V M Out L : RegionName)
    (idx : TileIndex [64, 64]) : ℝ :=
  match exec (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 64 1 1 64 64 64 1 0 1 0).toAlgKernel s with
  | some s' => s'.readMem Out (outOffset s 4 32768 8192 64 1 64 idx)
  | none => 0.0

noncomputable def producedAttentionFwdTriton3Case4MValue
    (s : BlockState) (Q K V M Out L : RegionName) (i : Fin 64) : ℝ :=
  match exec (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 64 1 1 64 64 64 1 0 1 0).toAlgKernel s with
  | some s' => s'.readMem M (lRowOffset s (s.pids 1) 128 64 i)
  | none => 0.0

theorem attention_fwd_triton3_case4_surface_out_compute_correct
    (Q K V M Out L : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 64 1 1 64 64 64 1 0 1 0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        producedAttentionFwdTriton3Case4OutValue s Q K V M Out L idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedAttentionFwdTriton3Case4OutValue]
  rw [inv_eq_one_div, hExec]

theorem attention_fwd_triton3_case4_surface_m_compute_correct
    (Q K V M Out L : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 64 1 1 64 64 64 1 0 1 0)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s (s.pids 1) 128 64 i))
      (expected := fun i : Fin 64 =>
        producedAttentionFwdTriton3Case4MValue s Q K V M Out L i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [producedAttentionFwdTriton3Case4MValue]
  rw [inv_eq_one_div, hExec]

theorem attention_fwd_triton3_python_case4_output_summary
    (Q K V M Out L : RegionName) (s : BlockState) :
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 0 1 0
      ).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 64 1 1 64 64 64 1 0 1 0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        producedAttentionFwdTriton3Case4OutValue s Q K V M Out L idx) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 64 1 1 64 64 64 1 0 1 0)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s (s.pids 1) 128 64 i))
      (expected := fun i : Fin 64 =>
        producedAttentionFwdTriton3Case4MValue s Q K V M Out L i) := by
  constructor
  · exact attention_fwd_triton3_python_case4_surface_toAlgorithm_supported
      Q K V M Out L
  constructor
  · exact attention_fwd_triton3_case4_surface_out_compute_correct Q K V M Out
      L s
  · exact attention_fwd_triton3_case4_surface_m_compute_correct Q K V M Out
      L s

/-- Strengthened checked-shape END epilogue summary: in addition to the
existing case summaries, expose the compute-correct `acc / l_i[:, None]`
producer for the final `Out` writeback used when `END=True`. -/
theorem attention_fwd_triton3_python_end_output_formula_summary
    (Acc LPre Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_output_formula_store_slice Acc LPre
        Out 4 128 64 32768 8192 64 1 32768 8192 64 1 128 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        endOutputStoreSpec s Acc LPre 4 32768 8192 64 1 128 64 idx) := by
  exact attention_fwd_triton3_end_output_formula_python_test_shape_compute_correct
    Acc LPre Out s

/-- Combined checked-shape summary for `test_forward` in `attention_fwd_triton3.py`.

This pins all four Python branch launches (`sliding_window`, complement window,
plain full-window, and `INIT=False`) and exposes the END epilogue arithmetic
that mutates the observable `Out` and `M` tensors. -/
theorem attention_fwd_triton3_python_test_shape_complete_summary
    (Q K V M Out L Acc MPre LPre : RegionName) (off_hz : Nat)
    (s : BlockState) :
    ((∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0
      ).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 1
      ).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 0 1 1 64 64 64 1 1 0 0
      ).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 0 1 0
      ).toAlgorithm? = Except.ok alg)) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_output_formula_store_slice Acc LPre
        Out 4 128 64 32768 8192 64 1 32768 8192 64 1 128 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        endOutputStoreSpec s Acc LPre 4 32768 8192 64 1 128 64 idx) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_m_formula_store_slice MPre LPre M
        off_hz 128 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s off_hz 128 64 i))
      (expected := fun i : Fin 64 =>
        endMStoreSpec s MPre LPre off_hz 128 64 i) := by
  constructor
  · constructor
    · exact attention_fwd_triton3_python_case1_surface_toAlgorithm_supported
        Q K V M Out L
    constructor
    · exact attention_fwd_triton3_python_case2_surface_toAlgorithm_supported
        Q K V M Out L
    constructor
    · exact attention_fwd_triton3_python_case3_surface_toAlgorithm_supported
        Q K V M Out L
    · exact attention_fwd_triton3_python_case4_surface_toAlgorithm_supported
        Q K V M Out L
  constructor
  · exact attention_fwd_triton3_python_end_output_formula_summary Acc LPre Out s
  · exact attention_fwd_triton3_end_m_formula_python_test_shape_compute_correct
      MPre LPre M off_hz s

/-! ## Genuine closed-form attention spec for cases 1/2/3

The `producedAttentionFwdTriton3CaseN{Out,M}Value` definitions above are the
single-program *surface* values (`exec … |> readMem`). The genuine, non-self-
referential closed form for the `END=True` normalized output `acc / l_i` is the
predicate-masked base-2 per-key-scale attention `attentionRealBase2PerKeyScalePred`
(STAGE 1, `VeriTile/Triton/Math/Attention.lean`), instantiated at the kernel's
block-ptr tile layout and per-case sliding-window `keep` predicate:

* case 1 (`SLIDING_WINDOW`, non-complement) → `slidingWindowKeep qStart 0 64`,
* case 2 (`COMPLEMENT_SLIDING_WINDOW`)       → `complementSlidingWindowKeep qStart 0 64`,
* case 3 (no window)                          → `noWindowKeep` (= unmasked softmax).

These definitions and their streaming bridges are sorry-free. Linking the
kernel's `producedAttentionFwdTriton3CaseN OutValue` to these closed forms is the
remaining exec-side realization (per-statement block-ptr/where/exp2 recipes +
`attnInvariant`/`preLoop`/`attn_step`/`postLoop` over `forRangeDyn_inv`, as in
`VeriTile/Examples/AttentionForwardClosedForm.lean`). -/

open VeriTile.Triton (attentionRealBase2PerKeyScalePred attnKeyListPred osStep
  slidingWindowKeep complementSlidingWindowKeep noWindowKeep pow2)

/-- Base address of the `(off_z, off_h)` plane for the Python test shape
(strides `qz = 32768`, `qh = 8192`, `H = 4`). Q and Out share this plane; K and V
share it too (`off_hkv = off_h` since `H_KV = H = 4`). -/
def baseOffset3 (s : BlockState) : Nat :=
  s.pids 1 / 4 * 32768 + s.pids 1 % 4 * 8192

/-- Query tile at the block-ptr layout: row `i` = global query `start_m·64 + i`,
head lane `e`, contiguous head dim (`stride_qm = 64`, `stride_qk = 1`). -/
noncomputable def qTile3 (s : BlockState) (Q : RegionName) :
    TileIndex [64, 64] → ℝ :=
  fun (i, e, _) => s.readMem Q (baseOffset3 s + (s.pids 0 * 64 + i.val) * 64 + e.val)

/-- Key tile at the block-ptr layout: column `j` (global key), head lane `e`. -/
noncomputable def kTile3 (s : BlockState) (K : RegionName) :
    TileIndex [128, 64] → ℝ :=
  fun (j, e, _) => s.readMem K (baseOffset3 s + j.val * 64 + e.val)

/-- Value tile at the block-ptr layout: row `j` (global key), head lane `d`. -/
noncomputable def vTile3 (s : BlockState) (V : RegionName) :
    TileIndex [128, 64] → ℝ :=
  fun (j, d, _) => s.readMem V (baseOffset3 s + j.val * 64 + d.val)

/-- The kernel's scalar score scale `qk_scale = sm_scale · log2e = (1/8)·log2e`,
applied uniformly to every key. The `exp2(qk·qk_scale)` softmax weight is exactly
`pow2 (keyScale3 · raw)`. -/
noncomputable def keyScale3 : Fin 128 → ℝ :=
  fun _ => (1 / 8 : ℝ) * 1.4426950408889634

/-- Global query row for output tile-row `i` in this program. -/
def qStart3 (s : BlockState) : Nat := s.pids 0 * 64

/-- **Genuine closed form, case 1** (sliding window, non-complement): the
normalized `END` output is predicate-masked base-2 attention with the
`slidingWindowKeep qStart 0 64` mask. -/
noncomputable def attentionFwdTriton3Case1OutSpec
    (s : BlockState) (Q K V : RegionName) (idx : TileIndex [64, 64]) : ℝ :=
  attentionRealBase2PerKeyScalePred (qTile3 s Q) (kTile3 s K) (vTile3 s V)
    keyScale3 (fun i j => slidingWindowKeep (qStart3 s) 0 64 i j) idx

/-- **Genuine closed form, case 2** (complement sliding window). -/
noncomputable def attentionFwdTriton3Case2OutSpec
    (s : BlockState) (Q K V : RegionName) (idx : TileIndex [64, 64]) : ℝ :=
  attentionRealBase2PerKeyScalePred (qTile3 s Q) (kTile3 s K) (vTile3 s V)
    keyScale3 (fun i j => complementSlidingWindowKeep (qStart3 s) 0 64 i j) idx

/-- **Genuine closed form, case 3** (no sliding window) — plain base-2 softmax. -/
noncomputable def attentionFwdTriton3Case3OutSpec
    (s : BlockState) (Q K V : RegionName) (idx : TileIndex [64, 64]) : ℝ :=
  attentionRealBase2PerKeyScalePred (qTile3 s Q) (kTile3 s K) (vTile3 s V)
    keyScale3 (fun i j => noWindowKeep i j) idx

/-- Streaming bridge, case 1: the closed form equals the `osStep` online-softmax
fold over the masked key list — the form the exec loop realizes. -/
theorem attentionFwdTriton3Case1OutSpec_eq_streaming
    (s : BlockState) (Q K V : RegionName) (i d : Fin 64) :
    attentionFwdTriton3Case1OutSpec s Q K V (i, d, PUnit.unit)
      = (let st := (attnKeyListPred (qTile3 s Q) (kTile3 s K) (vTile3 s V)
            keyScale3 (fun i j => slidingWindowKeep (qStart3 s) 0 64 i j) i d).foldl
              osStep (0, 0, 0)
         st.2.2 / st.2.1) := by
  simpa [attentionFwdTriton3Case1OutSpec] using
    VeriTile.Triton.attentionRealBase2PerKeyScalePred_eq_streaming
      (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3
      (fun i j => slidingWindowKeep (qStart3 s) 0 64 i j) i d

/-- Streaming bridge, case 2. -/
theorem attentionFwdTriton3Case2OutSpec_eq_streaming
    (s : BlockState) (Q K V : RegionName) (i d : Fin 64) :
    attentionFwdTriton3Case2OutSpec s Q K V (i, d, PUnit.unit)
      = (let st := (attnKeyListPred (qTile3 s Q) (kTile3 s K) (vTile3 s V)
            keyScale3 (fun i j => complementSlidingWindowKeep (qStart3 s) 0 64 i j) i d).foldl
              osStep (0, 0, 0)
         st.2.2 / st.2.1) := by
  simpa [attentionFwdTriton3Case2OutSpec] using
    VeriTile.Triton.attentionRealBase2PerKeyScalePred_eq_streaming
      (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3
      (fun i j => complementSlidingWindowKeep (qStart3 s) 0 64 i j) i d

/-- Streaming bridge, case 3 — also equals the plain (unmasked) base-2 softmax. -/
theorem attentionFwdTriton3Case3OutSpec_eq_streaming
    (s : BlockState) (Q K V : RegionName) (i d : Fin 64) :
    attentionFwdTriton3Case3OutSpec s Q K V (i, d, PUnit.unit)
      = (let st := (attnKeyListPred (qTile3 s Q) (kTile3 s K) (vTile3 s V)
            keyScale3 (fun i j => noWindowKeep i j) i d).foldl osStep (0, 0, 0)
         st.2.2 / st.2.1) := by
  simpa [attentionFwdTriton3Case3OutSpec] using
    VeriTile.Triton.attentionRealBase2PerKeyScalePred_eq_streaming
      (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3
      (fun i j => noWindowKeep i j) i d

/-- Case 3's masked closed form is the plain unmasked base-2 per-key-scale
softmax (`attentionRealBase2PerKeyScale`). -/
theorem attentionFwdTriton3Case3OutSpec_eq_unmasked
    (s : BlockState) (Q K V : RegionName) (idx : TileIndex [64, 64]) :
    attentionFwdTriton3Case3OutSpec s Q K V idx
      = VeriTile.Triton.attentionRealBase2PerKeyScale
          (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3 idx := by
  simpa [attentionFwdTriton3Case3OutSpec] using
    VeriTile.Triton.attentionRealBase2PerKeyScalePred_noWindow_eq
      (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3 idx

/-! ## Forward-loop per-statement op-eval recipes (RECIPE LAYER)

The streaming inner loop of `attention_fwd_triton3.py`'s `_attn_fwd_inner` is
expressed as a `forRangeDyn` body whose statement order (extracted from the
lowered surface AST at the Python test shape, `BLOCK_M=BLOCK_N=BLOCK_DMODEL=64`,
`IS_EVEN_N=1` so the boundary `tl.where` branches are dead) is:

```
 1  k    = tl.load(K_block_ptr)                              -- block-ptr load (col offset)
 2  qk   = tl.zeros([64,64])                                 -- neutral dot seed
 3  qk  += tl.dot(q, k)                                      -- q·k
 4  qk   = qk * qk_scale                                     -- scalar scale (= keyScale3)
 5  dist = arange[:,None] - arange[None,:] + start_m·64
                                        - start_n + 0        -- sliding-window distance (nat)
 6  mask = (dist >= 0) & (dist < 64)                         -- case-1 sliding-window keep
 7  qk   = tl.where(mask, qk, -inf)                          -- non-kept lanes → -inf
 8  m_ij = tl.maximum(m_i, tl.max(qk, 1))                    -- running max (reduceMax)
 9  qk   = qk - m_ij[:,None]                                 -- max-shift
10  p    = tl.math.exp2(qk)                                  -- base-2 softmax weights
11  p    = tl.where(mask, p, 0)                              -- zero non-kept lanes
12  l_ij = tl.sum(p, 1)                                      -- denominator increment (reduceSum)
13  tmp  = m_i - m_ij                                        -- log-domain correction
14  alpha = tl.math.exp2(tmp)                                -- rescale factor
15  l_i  = l_i * alpha + l_ij                                -- denominator carry
16  acc  = acc * alpha[:,None]                               -- accumulator rescale
17  v    = tl.load(V_block_ptr)                              -- block-ptr load (row offset)
18  acc += tl.dot(p, v)                                      -- numerator accumulation
19  m_i  = m_ij                                              -- max carry
20  V_block_ptr = tl.advance(V_block_ptr, [64, 0])           -- V steps down the key axis
21  K_block_ptr = tl.advance(K_block_ptr, [0, 64])           -- K steps along the key axis
```

Case 2 replaces line 6's `mask` with `dist >= 64`
(`complementSlidingWindowKeep`); case 3 drops lines 6/7/11 entirely
(`noWindowKeep`). Each recipe below is a standalone `evalOp` reduction with
abstract register-readback hypotheses over a symbolic `BlockState`, so the
eventual step lemma threads them through `stepStmts.cons_some` without reducing a
nested `setReg` literal. ASSEMBLY (invariant / `attn_step` / pre+post-loop) is
the NEXT stage and is intentionally NOT attempted here. -/

/-- `evalOp` helper for `tl.math.exp2` (`Op.exp2`) → `Tile.uop realExp2`. -/
theorem aft3_evalOp_exp2 {shape : TileShape} (a : Op .real shape) (s : BlockState) :
    evalOp (.exp2 a) s = (do let va ← evalOp a s; some (Tile.uop WithBot.realExp2 va)) := by
  simp [evalOp]

/-- `evalOp` helper for the `>=` predicate (`Op.ge`), which has no `@[simp]` form. -/
theorem aft3_evalOp_ge {dtype a b shape} (h : ComparableDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L1: `k = tl.load(K_block_ptr)`** (no boundary check). K's block ptr has
`order=(0,1)`: parent `(BLOCK_DMODEL, NKV_CTX)`, strides `(stride_kk, stride_kn)`,
offsets `[0, colOff]` (column advances by `BLOCK_N` per block). Each lane reads
`readMem` at `base + e·strideT + (colOff + j)·strideS`. With `K`'s
`strides=(1, 64)` this is head-lane `e`, key `colOff + j` — exactly a `kTile3`
read after `colOff` column advances. -/
theorem aft3_load_k_eval
    (region : RegionName) (base rows cols BT BS strideT strideS colOff : Nat)
    (ptrOp : Op .blockPtr [BT, BS]) (s : BlockState)
    (hp : evalOp ptrOp s = some
      ⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [0, colOff] }⟩) :
    evalOp (.load .real (.blockPtr ptrOp []) .none) s
      = some ⟨fun idx : TileIndex [BT, BS] =>
          some (s.readMem region
            (base + idx.1.val * strideT + (colOff + idx.2.1.val) * strideS))⟩ := by
  simp only [evalOp, hp, Option.bind]
  refine congrArg some ?_
  ext idx
  simp only [TileShape.indexToList, BlockPtr.inBounds, List.all_nil,
    BlockPtr.address_2d_zero_row_offset, BlockState.readMemValue_real, if_true]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L17: `v = tl.load(V_block_ptr)`** (no boundary check). V's block ptr has
`order=(1,0)`: parent `(NKV_CTX, BLOCK_DMODEL)`, strides `(stride_vk, stride_vn)`,
offsets `[rowOff, 0]` (row advances by `BLOCK_N` per block). Each lane reads
`base + (rowOff + j)·strideT + d·strideS`. With `V`'s `strides=(64, 1)` this is
key `rowOff + j`, head-lane `d` — exactly a `vTile3` read after `rowOff` row
advances. -/
theorem aft3_load_v_eval
    (region : RegionName) (base rows cols BT BS strideT strideS rowOff : Nat)
    (ptrOp : Op .blockPtr [BT, BS]) (s : BlockState)
    (hp : evalOp ptrOp s = some
      ⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [rowOff, 0] }⟩) :
    evalOp (.load .real (.blockPtr ptrOp []) .none) s
      = some ⟨fun idx : TileIndex [BT, BS] =>
          some (s.readMem region
            (base + (rowOff + idx.1.val) * strideT + idx.2.1.val * strideS))⟩ := by
  simp only [evalOp, hp, Option.bind]
  refine congrArg some ?_
  ext idx
  have haddr : BlockPtr.address
      { region := region, baseOffset := base, parentShape := [rows, cols],
        blockShape := [BT, BS], strides := [strideT, strideS],
        offsets := [rowOff, 0] }
      [idx.1.val, idx.2.1.val]
      = base + (rowOff + idx.1.val) * strideT + idx.2.1.val * strideS := by
    show base + ((rowOff + idx.1.val) * strideT + (0 + idx.2.1.val) * strideS) = _
    rw [Nat.zero_add, Nat.add_assoc]
  simp only [TileShape.indexToList, BlockPtr.inBounds, List.all_nil,
    haddr, BlockState.readMemValue_real, if_true]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L21: `K_block_ptr = tl.advance(K_block_ptr, [0, BLOCK_N])`** — advance the
column offset of a `[0, colOff]` block pointer by `BLOCK_N`. -/
theorem aft3_advance_k_eval (s : BlockState) (region : RegionName)
    (base rows cols BT BS strideT strideS colOff d : Nat) (name : RegName)
    (hkp : s.regs .blockPtr [BT, BS] name = some
      (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [0, colOff] }⟩)) :
    evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BT, BS] name) [0, d]) s
      = some (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [0, colOff + d] }⟩) := by
  simp only [evalOp, evalOp_ref, hkp, Option.bind]
  refine congrArg some ?_
  ext i
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L20: `V_block_ptr = tl.advance(V_block_ptr, [BLOCK_N, 0])`** — advance the
row offset of a `[rowOff, 0]` block pointer by `BLOCK_N`. -/
theorem aft3_advance_v_eval (s : BlockState) (region : RegionName)
    (base rows cols BT BS strideT strideS rowOff d : Nat) (name : RegName)
    (hkp : s.regs .blockPtr [BT, BS] name = some
      (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [rowOff, 0] }⟩)) :
    evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BT, BS] name) [d, 0]) s
      = some (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [rowOff + d, 0] }⟩) := by
  simp only [evalOp, evalOp_ref, hkp, Option.bind]
  refine congrArg some ?_
  ext i
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L2: `qk = tl.zeros([BLOCK_M, BLOCK_N])`** — the all-`0` neutral dot seed. -/
theorem aft3_qkzeros_eval (s : BlockState) (BM BN : Nat) :
    evalOp (Op.full [BM, BN] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L3: `qk += tl.dot(q, k)`** — adds the `q·k` dot to the zero-seeded `qk`
tile. Both `q` and `k` are `.real` (K is loaded transposed by the block-ptr
`order=(0,1)`, so no `tl.trans` appears in the AST). -/
theorem aft3_qk_dot_eval (s : BlockState) (BM BN BD : Nat)
    (qktile : Tile .real [BM, BN]) (qtile : Tile .real [BM, BD]) (ktile : Tile .real [BD, BN])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hq : s.regs .real [BM, BD] "q" = some qtile)
    (hk : s.regs .real [BD, BN] "k" = some ktile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "qk")
        (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k"))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          qktile (Tile.dot [] qtile ktile)) := by
  have hqr : evalOp (Op.ref .real [BM, BD] "q") s = some qtile := by rw [evalOp_ref, hq]
  have hkr : evalOp (Op.ref .real [BD, BN] "k") s = some ktile := by rw [evalOp_ref, hk]
  have hdot : @evalOp TileDType.real [BM, BN]
      (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k")) s
      = some (Tile.dot [] qtile ktile) := by
    erw [evalOp_dot [] (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k"), hqr, hkr]; rfl
  rw [evalOp_add]; simp only [evalOp_ref, hqk, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L4: `qk = qk * qk_scale`** — scale by the scalar `qk_scale = keyScale3`
(`(1/8)·log2e`), broadcast on the right. Each lane's `exp2(qk·qk_scale)` weight is
exactly `pow2 (keyScale3 · raw)`. -/
theorem aft3_qk_scale_eval (s : BlockState) (BM BN : Nat) (sc : ℝ)
    (qktile : Tile .real [BM, BN]) (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, BN] "qk") (Op.const sc)) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qktile
          (Tile.scalar (some sc : WithBot ℝ))) := by
  rw [evalOp_mul]; simp [evalOp_ref, evalOp_const, hqk]

/-- `evalOp` helper for `Op.boolAnd` (no `@[simp]` form). -/
theorem aft3_evalOp_boolAnd {a b shape} (bc : Broadcast a b shape)
    (x : Op .bool a) (y : Op .bool b) (s : BlockState) :
    evalOp (.boolAnd bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s;
      some (Tile.bop (fun p q : Bool => p && q) bc vx vy)) := by
  simp [evalOp]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L5: `dist = arange[:,None] - arange[None,:] + start_m·BLOCK_M
- start_n + offset`** (here `offset = 0`). With `start_m = SM`, `start_n = SN`,
each cell `(i, j)` is the truncated-nat distance
`((i - j) + SM·64 - SN) + 0` — the per-lane sliding-window position. -/
theorem aft3_dist_eval (s : BlockState) (SM SN : Nat)
    (hsm : s.regs .nat [] "start_m" = some (Tile.scalar SM))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOp (Op.add .nat Broadcast.scalarR
        (Op.sub .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarR
            (Op.sub .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.expandDim ⟨1, by simp⟩ (Op.arange 64))
              (Op.expandDim ⟨0, by simp⟩ (Op.arange 64)))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 64)))
          (Op.ref .nat [] "start_n"))
        (Op.constNat 0)) s
      = some ⟨fun idx : TileIndex [64, 64] =>
          ((idx.1.val - idx.2.1.val) + SM * 64 - SN) + 0⟩ := by
  simp only [evalOp.eq_def, hsm, hsn, Option.bind_eq_bind, Option.bind_some, Option.bind]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.expandDim_data, Tile.vec, Tile.scalar,
    Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
    Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
    Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
    Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, NumericDType.sub]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L6 (case 1, sliding window non-complement): `mask = (dist >= 0) &
(dist < SLIDING_WINDOW_SIZE)`** (`= 64`). Cell `(i, j)` is kept iff the distance
is in `[0, 64)` — exactly `slidingWindowKeep`'s in-window condition. -/
theorem aft3_mask_case1_eval (s : BlockState) (disttile : Tile .nat [64, 64])
    (hd : s.regs .nat [64, 64] "dist" = some disttile) :
    evalOp (Op.boolAnd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ge ComparableDType.nat Broadcast.scalarR (Op.ref .nat [64, 64] "dist")
          (Op.constNat 0))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [64, 64] "dist")
          (Op.constNat 64))) s
      = some ⟨fun idx : TileIndex [64, 64] =>
          ComparableDType.nat.ge (disttile.data idx) 0
            && ComparableDType.nat.lt (disttile.data idx) 64⟩ := by
  rw [aft3_evalOp_boolAnd, aft3_evalOp_ge, evalOp_lt]
  simp only [evalOp_ref, evalOp_constNat, hd, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.cop_data, Tile.scalar,
    Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
    Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
    Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L6 (case 2, complement sliding window): `mask = (dist >= SLIDING_WINDOW_SIZE)`**
(`= 64`). Cell `(i, j)` is kept iff the distance is `≥ 64` — exactly
`complementSlidingWindowKeep`'s condition. -/
theorem aft3_mask_case2_eval (s : BlockState) (disttile : Tile .nat [64, 64])
    (hd : s.regs .nat [64, 64] "dist" = some disttile) :
    evalOp (Op.ge ComparableDType.nat Broadcast.scalarR (Op.ref .nat [64, 64] "dist")
        (Op.constNat 64)) s
      = some ⟨fun idx : TileIndex [64, 64] =>
          ComparableDType.nat.ge (disttile.data idx) 64⟩ := by
  rw [aft3_evalOp_ge]
  simp only [evalOp_ref, evalOp_constNat, hd, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.cop_data, Tile.scalar,
    Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
    Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L7: `qk = tl.where(mask, qk, -inf)`** — keep `qk` on lanes the case
predicate selects, set non-kept lanes to `⊥` (`-inf`) so they vanish under
`exp2`. Works for any case `mask` tile (`slidingWindowKeep` /
`complementSlidingWindowKeep`). -/
theorem aft3_where_eval (s : BlockState) (masktile : Tile .bool [64, 64])
    (qktile : Tile .real [64, 64])
    (hmask : s.regs .bool [64, 64] "mask" = some masktile)
    (hqk : s.regs .real [64, 64] "qk" = some qktile) :
    evalOp (Op.where (Op.ref .bool [64, 64] "mask")
        (Op.ref .real [64, 64] "qk") (Op.broadcast Op.negInf [64, 64])) s
      = some ⟨fun idx : TileIndex [64, 64] =>
          if masktile.data idx then qktile.data idx else (⊥ : WithBot ℝ)⟩ := by
  have hbcast : @evalOp TileDType.real [64, 64] (Op.broadcast Op.negInf [64, 64]) s
      = some (⟨fun _ : TileIndex [64, 64] => (⊥ : WithBot ℝ)⟩ : Tile .real [64, 64]) := by
    simp only [evalOp, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]; rfl
  rw [evalOp_where]
  simp only [evalOp_ref, hmask, hqk, hbcast, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.select_data, Tile.scalar]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L8: `m_ij = tl.maximum(m_i, tl.max(qk, 1))`** — lowered to
`where(m_i > reduceMax(qk,1), m_i, reduceMax(qk,1))`, i.e. the running max keeps
`m_i` when it already dominates the new block row-max, else takes the row-max.
`reduceMax`'s `eraseAxis` result-shape blocks `rw`, so the reduced row is proven
then defeq-coerced to `[64]`. -/
theorem aft3_mij_eval (s : BlockState)
    (mtile : Tile .real [64]) (qktile : Tile .real [64, 64]) (rmaxT : Tile .real [64])
    (hmi : s.regs .real [64] "m_i" = some mtile)
    (hqk : s.regs .real [64, 64] "qk" = some qktile)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [64, 64].length) qktile = some rmaxT) :
    evalOp (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [64] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [64, 64].length) Bool.false (Op.ref .real [64, 64] "qk")))
        (Op.ref .real [64] "m_i")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [64, 64].length) Bool.false (Op.ref .real [64, 64] "qk"))) s
      = some (Tile.select
          (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT)
          mtile rmaxT) := by
  have hrmaxN : evalOp (Op.reduceMax (⟨1, by simp⟩ : Fin [64, 64].length) Bool.false
      (Op.ref .real [64, 64] "qk")) s = some rmaxT := by
    rw [evalOp_reduceMax]; simp only [evalOp_ref, hqk]; exact hrm
  have hrmax : @evalOp TileDType.real [64]
      (Op.reduceMax (⟨1, by simp⟩ : Fin [64, 64].length) Bool.false
        (Op.ref .real [64, 64] "qk")) s = some rmaxT := hrmaxN
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hmi, hrmax, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L9: `qk = qk - m_ij[:, None]`** — the max-shift before `exp2`. -/
theorem aft3_qk_sub_eval (s : BlockState) (hax : 1 < [64].length.succ)
    (qktile : Tile .real [64, 64]) (mc : Tile .real [64])
    (hqk : s.regs .real [64, 64] "qk" = some qktile)
    (hmij : s.regs .real [64] "m_ij" = some mc) :
    evalOp (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [64, 64] "qk") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [64] "m_ij"))) s
      = some (Tile.bop NumericDType.real.sub
          (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          qktile (Tile.expandDim ⟨1, hax⟩ mc)) := by
  have hexp : @evalOp TileDType.real [64, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [64] "m_ij")) s
      = some (Tile.expandDim ⟨1, hax⟩ mc) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hmij
  rw [evalOp_sub]
  simp only [evalOp_ref, hqk, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L10: `p = tl.math.exp2(qk)`** — the base-2 softmax weights (`⊥`/`-inf`
lanes collapse to `0`). -/
theorem aft3_p_eval (s : BlockState) (qktile : Tile .real [64, 64])
    (hqk : s.regs .real [64, 64] "qk" = some qktile) :
    evalOp (Op.exp2 (Op.ref .real [64, 64] "qk")) s
      = some (Tile.uop WithBot.realExp2 qktile) := by
  rw [aft3_evalOp_exp2]; simp only [evalOp_ref, hqk, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L11: `p = tl.where(mask, p, 0)`** — zero the non-kept lanes (redundant with
L7's `-inf`+`exp2`, but kept for the complement/case mask). -/
theorem aft3_p_mask_eval (s : BlockState) (masktile : Tile .bool [64, 64])
    (ptile : Tile .real [64, 64])
    (hmask : s.regs .bool [64, 64] "mask" = some masktile)
    (hp : s.regs .real [64, 64] "p" = some ptile) :
    evalOp (Op.where (Op.ref .bool [64, 64] "mask")
        (Op.ref .real [64, 64] "p") (Op.broadcast (Op.const 0.0) [64, 64])) s
      = some ⟨fun idx : TileIndex [64, 64] =>
          if masktile.data idx then ptile.data idx else (some (0.0 : ℝ) : WithBot ℝ)⟩ := by
  have hbcast : @evalOp TileDType.real [64, 64] (Op.broadcast (Op.const 0.0) [64, 64]) s
      = some (⟨fun _ : TileIndex [64, 64] => (some (0.0 : ℝ) : WithBot ℝ)⟩ : Tile .real [64, 64]) := by
    simp only [evalOp, evalOp_const, Option.bind_eq_bind, Option.bind_some]; rfl
  rw [evalOp_where]
  simp only [evalOp_ref, hmask, hp, hbcast, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.select_data, Tile.scalar]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L12: `l_ij = tl.sum(p, 1)`** — the per-row denominator increment. -/
theorem aft3_lij_eval (s : BlockState) (ptile : Tile .real [64, 64])
    (hp : s.regs .real [64, 64] "p" = some ptile) :
    evalOp (Op.reduceSum (⟨1, by simp⟩ : Fin [64, 64].length) Bool.false
        (Op.ref .real [64, 64] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [64, 64].length) ptile) := by
  rw [evalOp_reduceSum]; simp only [evalOp_ref, hp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L13: `tmp = m_i - m_ij`** — the log-domain correction exponent. -/
theorem aft3_tmp_eval (s : BlockState) (mi mij : Tile .real [64])
    (hmi : s.regs .real [64] "m_i" = some mi)
    (hmij : s.regs .real [64] "m_ij" = some mij) :
    evalOp (Op.sub .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [64] "m_i") (Op.ref .real [64] "m_ij")) s
      = some (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mi mij) := by
  rw [evalOp_sub]; simp only [evalOp_ref, hmi, hmij, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L14: `alpha = tl.math.exp2(tmp)`** — the running rescale factor
`exp2(m_i − m_ij)` (`= osStep`'s `α`; on a fresh row `m_i = −∞`, `tmp = ⊥` and
`exp2 ⊥ = 0`). -/
theorem aft3_alpha_eval (s : BlockState) (tmptile : Tile .real [64])
    (htmp : s.regs .real [64] "tmp" = some tmptile) :
    evalOp (Op.exp2 (Op.ref .real [64] "tmp")) s
      = some (Tile.uop WithBot.realExp2 tmptile) := by
  rw [aft3_evalOp_exp2]; simp only [evalOp_ref, htmp, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L15: `l_i = l_i * alpha + l_ij`** — the denominator carry
(`l_i ↦ l_i·α + Σp`). -/
theorem aft3_li_eval (s : BlockState) (li alpha lij : Tile .real [64])
    (hli : s.regs .real [64] "l_i" = some li)
    (halpha : s.regs .real [64] "alpha" = some alpha)
    (hlij : s.regs .real [64] "l_ij" = some lij) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [64] "l_i") (Op.ref .real [64] "alpha"))
        (Op.ref .real [64] "l_ij")) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
          (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) li alpha) lij) := by
  rw [evalOp_add, evalOp_mul]
  simp only [evalOp_ref, hli, halpha, hlij, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L16: `acc = acc * alpha[:, None]`** — rescale the output accumulator by the
per-row `α` before adding the new block's value contribution. -/
theorem aft3_acc_rescale_eval (s : BlockState) (hax : 1 < [64].length.succ)
    (acctile : Tile .real [64, 64]) (alpha : Tile .real [64])
    (hacc : s.regs .real [64, 64] "acc" = some acctile)
    (halpha : s.regs .real [64] "alpha" = some alpha) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [64, 64] "acc") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [64] "alpha"))) s
      = some (Tile.bop NumericDType.real.mul
          (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          acctile (Tile.expandDim ⟨1, hax⟩ alpha)) := by
  have hexp : @evalOp TileDType.real [64, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [64] "alpha")) s
      = some (Tile.expandDim ⟨1, hax⟩ alpha) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ halpha
  rw [evalOp_mul]
  simp only [evalOp_ref, hacc, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L18: `acc += tl.dot(p, v)`** — numerator accumulation. Both `p` and `v` are
`.real` (no fp16 round-trip in this transcription). -/
theorem aft3_acc_eval (s : BlockState) (BM BN BD : Nat)
    (acctile : Tile .real [BM, BD]) (ptile : Tile .real [BM, BN]) (vtile : Tile .real [BN, BD])
    (hacc : s.regs .real [BM, BD] "acc" = some acctile)
    (hp : s.regs .real [BM, BN] "p" = some ptile)
    (hv : s.regs .real [BN, BD] "v" = some vtile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BD] "acc")
        (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, BD] "v"))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acctile
          (Tile.dot [] ptile vtile)) := by
  have hpr : evalOp (Op.ref .real [BM, BN] "p") s = some ptile := by rw [evalOp_ref, hp]
  have hvr : evalOp (Op.ref .real [BN, BD] "v") s = some vtile := by rw [evalOp_ref, hv]
  have hdot : @evalOp TileDType.real [BM, BD]
      (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, BD] "v")) s
      = some (Tile.dot [] ptile vtile) := by
    erw [evalOp_dot [] (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, BD] "v"), hpr, hvr]; rfl
  rw [evalOp_add]; simp only [evalOp_ref, hacc, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L19: `m_i = m_ij`** — the running-max carry (a bare register read of the
new `m_ij`). -/
theorem aft3_mi_carry_eval (s : BlockState) (mij : Tile .real [64])
    (hmij : s.regs .real [64] "m_ij" = some mij) :
    evalOp (Op.ref .real [64] "m_ij") s = some mij := by
  rw [evalOp_ref, hmij]


/-! ## Exec-assembly foundation: body decomposition

The lowered Python-test-shape (case 1) `attention_fwd_triton3_surface` algorithm
body splits as `aft3PreLoop ++ Stmt.forRange "start_n" 0 128 64 aft3LoopBody ::
aft3PostLoop`. The preLoop sets up offsets / block pointers / `m_ptrs`/`l_ptrs`,
runs the constexpr `INIT` branch (`m_i = -inf`, `l_i = 1`, `acc = 0`), the
`qk_scale` scalar, and the constexpr `IS_EVEN_M` `q` load. The loop body is the
streaming online-softmax block recurrence (with the constexpr sliding-window
`ifThen`/`ifThenElse`). The postLoop runs the constexpr `END` finalize
(`m_i += log2 l_i`, `acc /= l_i`) then stores `m_i`/`acc`. -/

noncomputable def aft3PreLoop (Q K V M Out L : RegionName) : List Stmt :=
  [ Stmt.assign TileDType.nat [] "start_m" (Op.programId 0),
    Stmt.assign TileDType.nat [] "off_hz" (Op.programId 1),
    Stmt.assign TileDType.nat [] "off_z" (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat 4)),
    Stmt.assign TileDType.nat [] "off_h" (Op.mod IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat 4)),
    Stmt.assign TileDType.nat [] "off_hkv" (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h") (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat 4) (Op.constNat 4))),
    Stmt.assign TileDType.nat [] "q_offset" (Op.add NumericDType.nat Broadcast.nil (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat 32768)) (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h") (Op.constNat 8192))),
    Stmt.assign TileDType.nat [] "k_offset" (Op.add NumericDType.nat Broadcast.nil (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat 32768)) (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hkv") (Op.constNat 8192))),
    Stmt.assign TileDType.nat [] "v_offset" (Op.add NumericDType.nat Broadcast.nil (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat 32768)) (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hkv") (Op.constNat 8192))),
    Stmt.assign TileDType.nat [] "o_offset" (Op.add NumericDType.nat Broadcast.nil (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat 32768)) (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h") (Op.constNat 8192))),
    Stmt.assign TileDType.blockPtr [64, 64] "Q_block_ptr" (Op.makeBlockPtrDynOffsets Q (Op.ref TileDType.nat [] "q_offset") [128, 64] [64, 64] [64, 1] [Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_m") (Op.constNat 64), Op.constNat 0]),
    Stmt.assign TileDType.blockPtr [64, 64] "V_block_ptr" (Op.makeBlockPtrDyn V (Op.ref TileDType.nat [] "v_offset") [128, 64] [64, 64] [64, 1] [0, 0]),
    Stmt.assign TileDType.blockPtr [64, 64] "K_block_ptr" (Op.makeBlockPtrDyn K (Op.ref TileDType.nat [] "k_offset") [64, 128] [64, 64] [1, 64] [0, 0]),
    Stmt.assign TileDType.blockPtr [64, 64] "O_block_ptr" (Op.makeBlockPtrDynOffsets Out (Op.ref TileDType.nat [] "o_offset") [128, 64] [64, 64] [64, 1] [Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_m") (Op.constNat 64), Op.constNat 0]),
    Stmt.assign TileDType.nat [64] "offs_m" (Op.add NumericDType.nat Broadcast.scalarL (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_m") (Op.constNat 64)) (Op.arange 64)),
    Stmt.assign TileDType.ptr [64] "m_ptrs" (Op.ptrAdd Broadcast.scalarL (Op.ptrBase M) (Op.add NumericDType.nat Broadcast.scalarL (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat 128)) (Op.ref TileDType.nat [64] "offs_m"))),
    Stmt.assign TileDType.ptr [64] "l_ptrs" (Op.ptrAdd Broadcast.scalarL (Op.ptrBase L) (Op.add NumericDType.nat Broadcast.scalarL (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat 128)) (Op.ref TileDType.nat [64] "offs_m"))),
    Stmt.ifThenElse (Op.ne ComparableDType.nat Broadcast.nil (Op.constNat 1) (Op.constNat 0)) [Stmt.assign TileDType.real [64] "m_i" ((Op.add NumericDType.real Broadcast.scalarR (Op.full [64] (Op.const 0)) Op.negInf)), Stmt.assign TileDType.real [64] "l_i" ((Op.add NumericDType.real Broadcast.scalarR (Op.full [64] (Op.const 0)) (Op.const 1.0))), Stmt.assign TileDType.real [64, 64] "acc" (Op.full [64, 64] (Op.const 0))] [Stmt.assign TileDType.real [64] "m_i" (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [64] "m_ptrs")) MaskOpt.none), Stmt.assign TileDType.real [64] "l_i" (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [64] "l_ptrs")) MaskOpt.none), Stmt.assign TileDType.real [64, 64] "acc" (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "O_block_ptr") []) MaskOpt.none)],
    Stmt.assign TileDType.real [] "qk_scale" (Op.mul NumericDType.real Broadcast.nil (Op.const (1 / 8)) (Op.const 1.0)),
    Stmt.assign TileDType.real [] "qk_scale" (Op.mul NumericDType.real Broadcast.nil (Op.ref TileDType.real [] "qk_scale") (Op.const 1.4426950408889634)),
    Stmt.ifThenElse (Op.ne ComparableDType.nat Broadcast.nil (Op.constNat 1) (Op.constNat 0)) [Stmt.assign TileDType.real [64, 64] "q" (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "Q_block_ptr") []) MaskOpt.none)] [Stmt.assign TileDType.real [64, 64] "q" (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "Q_block_ptr") [0, 1]) MaskOpt.none)] ]

def aft3LoopBody : List Stmt :=
  [ Stmt.assign TileDType.real [64, 64] "k" (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "K_block_ptr") []) MaskOpt.none),
    Stmt.assign TileDType.real [64, 64] "qk" (Op.full [64, 64] (Op.const 0)),
    Stmt.assign TileDType.real [64, 64] "qk" ((Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref TileDType.real [64, 64] "qk") ((Op.dot (batch := []) (Op.ref TileDType.real [64, 64] "q") (Op.ref TileDType.real [64, 64] "k"))))),
    Stmt.assign TileDType.real [64, 64] "qk" ((Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [64, 64] "qk") (Op.ref TileDType.real [] "qk_scale"))),
    Stmt.ifThen (Op.ne ComparableDType.nat Broadcast.nil (Op.constNat 1) (Op.constNat 0)) [Stmt.assign TileDType.nat [64, 64] "dist" (Op.add NumericDType.nat Broadcast.scalarR (Op.sub NumericDType.nat Broadcast.scalarR (Op.add NumericDType.nat Broadcast.scalarR (Op.sub NumericDType.nat Broadcast.nil.consL.consR (Op.expandDim ⟨1, by decide⟩ (Op.arange 64)) (Op.expandDim ⟨0, by decide⟩ (Op.arange 64))) (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_m") (Op.constNat 64))) (Op.ref TileDType.nat [] "start_n")) (Op.constNat 0)), Stmt.ifThenElse (Op.ne ComparableDType.nat Broadcast.nil (Op.constNat 0) (Op.constNat 0)) [Stmt.assign TileDType.bool [64, 64] "mask" (Op.ge ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [64, 64] "dist") (Op.constNat 64))] [Stmt.assign TileDType.bool [64, 64] "mask" (Op.boolAnd Broadcast.nil.consSame.consSame (Op.ge ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [64, 64] "dist") (Op.constNat 0)) (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [64, 64] "dist") (Op.constNat 64)))], Stmt.assign TileDType.real [64, 64] "qk" ((Op.ref TileDType.bool [64, 64] "mask").where (Op.ref TileDType.real [64, 64] "qk") (Op.negInf.broadcast [64, 64]))],
    Stmt.assign TileDType.real [64] "m_ij" (((Op.gt ComparableDType.real Broadcast.nil.consSame (Op.ref TileDType.real [64] "m_i") (Op.reduceMax ⟨1, by decide⟩ Bool.false (Op.ref TileDType.real [64, 64] "qk"))).where (Op.ref TileDType.real [64] "m_i") (Op.reduceMax ⟨1, by decide⟩ Bool.false (Op.ref TileDType.real [64, 64] "qk")))),
    Stmt.assign TileDType.real [64, 64] "qk" (Op.sub NumericDType.real Broadcast.nil.consR.consSame (Op.ref TileDType.real [64, 64] "qk") (Op.expandDim ⟨1, by decide⟩ (Op.ref TileDType.real [64] "m_ij"))),
    Stmt.assign TileDType.real [64, 64] "p" (Op.ref TileDType.real [64, 64] "qk").exp2,
    Stmt.ifThen (Op.ne ComparableDType.nat Broadcast.nil (Op.constNat 1) (Op.constNat 0)) [Stmt.assign TileDType.real [64, 64] "p" ((Op.ref TileDType.bool [64, 64] "mask").where (Op.ref TileDType.real [64, 64] "p") ((Op.const 0.0).broadcast [64, 64]))],
    Stmt.assign TileDType.real [64] "l_ij" (Op.reduceSum ⟨1, by decide⟩ Bool.false (Op.ref TileDType.real [64, 64] "p")),
    Stmt.assign TileDType.real [64] "tmp" ((Op.sub NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [64] "m_i") (Op.ref TileDType.real [64] "m_ij"))),
    Stmt.assign TileDType.real [64] "alpha" (Op.ref TileDType.real [64] "tmp").exp2,
    Stmt.assign TileDType.real [64] "l_i" ((Op.add NumericDType.real Broadcast.nil.consSame (Op.mul NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [64] "l_i") (Op.ref TileDType.real [64] "alpha")) (Op.ref TileDType.real [64] "l_ij"))),
    Stmt.assign TileDType.real [64, 64] "acc" ((Op.mul NumericDType.real Broadcast.nil.consR.consSame (Op.ref TileDType.real [64, 64] "acc") (Op.expandDim ⟨1, by decide⟩ (Op.ref TileDType.real [64] "alpha")))),
    Stmt.assign TileDType.real [64, 64] "v" (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "V_block_ptr") []) MaskOpt.none),
    Stmt.assign TileDType.real [64, 64] "acc" ((Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref TileDType.real [64, 64] "acc") ((Op.dot (batch := []) (Op.ref TileDType.real [64, 64] "p") (Op.ref TileDType.real [64, 64] "v"))))),
    Stmt.assign TileDType.real [64] "m_i" ((Op.ref TileDType.real [64] "m_ij")),
    Stmt.assign TileDType.blockPtr [64, 64] "V_block_ptr" ((Op.ref TileDType.blockPtr [64, 64] "V_block_ptr").advanceBlockPtr [64, 0]),
    Stmt.assign TileDType.blockPtr [64, 64] "K_block_ptr" ((Op.ref TileDType.blockPtr [64, 64] "K_block_ptr").advanceBlockPtr [0, 64]) ]

def aft3PostLoop (M Out L : RegionName) : List Stmt :=
  [ Stmt.ifThenElse (Op.ne ComparableDType.nat Broadcast.nil (Op.constNat 1) (Op.constNat 0)) [Stmt.assign TileDType.real [64] "m_i" ((Op.add NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [64] "m_i") (Op.ref TileDType.real [64] "l_i").log2)), Stmt.assign TileDType.real [64, 64] "acc" ((Op.div NumericDType.real Broadcast.nil.consR.consSame (Op.ref TileDType.real [64, 64] "acc") (Op.expandDim ⟨1, by decide⟩ (Op.ref TileDType.real [64] "l_i"))))] [Stmt.store TileDType.real [64] (MemAccess.ptr (Op.ref TileDType.ptr [64] "l_ptrs")) ((Op.ref TileDType.real [64] "l_i")) MaskOpt.none],
    Stmt.store TileDType.real [64] (MemAccess.ptr (Op.ref TileDType.ptr [64] "m_ptrs")) ((Op.ref TileDType.real [64] "m_i")) MaskOpt.none,
    Stmt.store TileDType.real [64, 64] (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "O_block_ptr") []) ((Op.ref TileDType.real [64, 64] "acc")) MaskOpt.none ]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Body split (case 1).** The lowered algorithm body of the case-1 surface is
exactly `aft3PreLoop ++ Stmt.forRange "start_n" 0 128 64 aft3LoopBody ::
aft3PostLoop`. -/
theorem aft3_body_split (Q K V M Out L : RegionName) :
    (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0).toAlgKernel.body
      = aft3PreLoop Q K V M Out L
        ++ Stmt.forRange "start_n" 0 128 64 aft3LoopBody :: aft3PostLoop M Out L := by
  rfl


end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
