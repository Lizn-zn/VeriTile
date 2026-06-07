import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention
import VeriTile.Triton.LoopInvariant

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

/-- **Faithful nat-truncated sliding-window distance.** The kernel computes
`dist = (i − jL : ℕ) + start_m·64 − start_n` with **natural** subtraction (negative
results truncate to `0`), where `start_m = SM`, the block-local key is
`jL = j mod 64`, and `start_n = (j / 64)·64`. For a global key `j` and row `i`
this is `(i − j%64 : ℕ) + SM·64 − (j/64)·64`. This is the exact quantity the
lowered surface evaluates per cell (see `aft3MaskCell1`/`aft3MaskCell2`), NOT the
signed-ℤ distance of `slidingWindowKeep`. -/
def natDist3 (SM : Nat) (i : Fin 64) (j : Fin 128) : Nat :=
  (i.val - j.val % 64) + SM * 64 - (j.val / 64) * 64

/-- **Faithful case-1 keep predicate** (sliding window, non-complement): the
kernel's `mask = (dist ≥ 0) & (dist < 64)`. The `dist ≥ 0` clause is vacuous on
ℕ, so this is exactly `natDist3 SM i j < 64` — matching `aft3MaskCell1` cell for
cell (true by construction). -/
def natSlidingWindowKeep (SM : Nat) (i : Fin 64) (j : Fin 128) : Prop :=
  natDist3 SM i j < 64

instance natSlidingWindowKeepDecidable (SM : Nat) (i : Fin 64) (j : Fin 128) :
    Decidable (natSlidingWindowKeep SM i j) := by
  unfold natSlidingWindowKeep; infer_instance

/-- **Faithful case-2 keep predicate** (complement sliding window): the kernel's
`mask = dist ≥ 64`, i.e. `64 ≤ natDist3 SM i j` — matching `aft3MaskCell2`. -/
def natComplementSlidingWindowKeep (SM : Nat) (i : Fin 64) (j : Fin 128) : Prop :=
  64 ≤ natDist3 SM i j

instance natComplementSlidingWindowKeepDecidable (SM : Nat) (i : Fin 64) (j : Fin 128) :
    Decidable (natComplementSlidingWindowKeep SM i j) := by
  unfold natComplementSlidingWindowKeep; infer_instance

/-- **Genuine closed form, case 1** (sliding window, non-complement): the
normalized `END` output is predicate-masked base-2 attention with the *faithful*
nat-truncated `natSlidingWindowKeep` mask (the exact condition the kernel computes,
not the signed-ℤ `slidingWindowKeep`). -/
noncomputable def attentionFwdTriton3Case1OutSpec
    (s : BlockState) (Q K V : RegionName) (idx : TileIndex [64, 64]) : ℝ :=
  attentionRealBase2PerKeyScalePred (qTile3 s Q) (kTile3 s K) (vTile3 s V)
    keyScale3 (fun i j => natSlidingWindowKeep (s.pids 0) i j) idx

/-- **Genuine closed form, case 2** (complement sliding window) — faithful
nat-truncated `natComplementSlidingWindowKeep` mask. -/
noncomputable def attentionFwdTriton3Case2OutSpec
    (s : BlockState) (Q K V : RegionName) (idx : TileIndex [64, 64]) : ℝ :=
  attentionRealBase2PerKeyScalePred (qTile3 s Q) (kTile3 s K) (vTile3 s V)
    keyScale3 (fun i j => natComplementSlidingWindowKeep (s.pids 0) i j) idx

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
            keyScale3 (fun i j => natSlidingWindowKeep (s.pids 0) i j) i d).foldl
              osStep (0, 0, 0)
         st.2.2 / st.2.1) := by
  simpa [attentionFwdTriton3Case1OutSpec] using
    VeriTile.Triton.attentionRealBase2PerKeyScalePred_eq_streaming
      (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3
      (fun i j => natSlidingWindowKeep (s.pids 0) i j) i d

/-- Streaming bridge, case 2. -/
theorem attentionFwdTriton3Case2OutSpec_eq_streaming
    (s : BlockState) (Q K V : RegionName) (i d : Fin 64) :
    attentionFwdTriton3Case2OutSpec s Q K V (i, d, PUnit.unit)
      = (let st := (attnKeyListPred (qTile3 s Q) (kTile3 s K) (vTile3 s V)
            keyScale3 (fun i j => natComplementSlidingWindowKeep (s.pids 0) i j) i d).foldl
              osStep (0, 0, 0)
         st.2.2 / st.2.1) := by
  simpa [attentionFwdTriton3Case2OutSpec] using
    VeriTile.Triton.attentionRealBase2PerKeyScalePred_eq_streaming
      (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3
      (fun i j => natComplementSlidingWindowKeep (s.pids 0) i j) i d

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

/-- **L4 (register variant): `qk = qk * qk_scale`** with `qk_scale` read from the
`qk_scale` register (= `keyScale3`). -/
theorem aft3_qk_scale_ref_eval (s : BlockState) (BM BN : Nat) (sc : ℝ)
    (qktile : Tile .real [BM, BN]) (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hqs : s.regs .real [] "qk_scale" = some (Tile.scalar (some sc))) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, BN] "qk") (Op.ref .real [] "qk_scale")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qktile
          (Tile.scalar (some sc : WithBot ℝ))) := by
  rw [evalOp_mul]; simp only [evalOp_ref, hqk, hqs, Option.bind_eq_bind, Option.bind_some]

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


/-! ## ⊥-seed online-softmax recurrence (deliverable 3, generic core)

The kernel seeds its running `max` register at `⊥` (`tl.zeros − inf`) and
`l_i`/`acc` at `0`. `aft3OsStepBot` is the faithful per-key ⊥-seeded update;
`aft3StateBot` folds it over a key list. These generic lemmas (over an arbitrary
`List (ℝ × ℝ)`) mirror the flash-attn `osStepBot` core and are reused by the
predicate-windowed layer below. -/

open VeriTile.Triton (osStep pow2)

/-- ⊥-seeded per-key online-softmax update (the kernel's register recurrence:
`max` carried as `WithBot ℝ`, `l`/`acc` as `ℝ`). -/
noncomputable def aft3OsStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let s := sv.1; let v := sv.2
  let m' := m ⊔ ((s : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
  let p := pow2 (s - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)

/-- The running `max` component of an `aft3OsStepBot` fold is the `⊔`-fold of the
coerced scores. -/
theorem aft3StateBot_fst
    (xs : List (ℝ × ℝ)) (m₀ : WithBot ℝ) (l₀ acc₀ : ℝ) :
    (xs.foldl aft3OsStepBot (m₀, l₀, acc₀)).1
      = (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldl (· ⊔ ·) m₀ := by
  induction xs generalizing m₀ l₀ acc₀ with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl_cons, List.map_cons]; rw [ih]; rfl

/-- **⊥-seeded consistency.** Folding `aft3OsStepBot` from a start anchored to the
true (max-free) batch denominator `L` / accumulator `T` keeps that invariant. -/
theorem aft3OsStepBot_foldl_consistent (xs : List (ℝ × ℝ)) (m : WithBot ℝ) (l acc T L : ℝ)
    (hl : l = (m.elim 0 (fun r => pow2 (-r))) * L)
    (hacc : acc = (m.elim 0 (fun r => pow2 (-r))) * T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0) :
    let st := xs.foldl aft3OsStepBot (m, l, acc)
    st.2.1 = (st.1.elim 0 (fun r => pow2 (-r))) * (L + (xs.map (fun p => pow2 p.1)).sum) ∧
    st.2.2 = (st.1.elim 0 (fun r => pow2 (-r))) * (T + (xs.map (fun p => pow2 p.1 * p.2)).sum) := by
  induction xs generalizing m l acc T L with
  | nil => simp [hl, hacc]
  | cons x xs ih =>
    obtain ⟨s, v⟩ := x
    set m' : WithBot ℝ := m ⊔ ((s : ℝ) : WithBot ℝ) with hm'
    have hm'r : ∃ r : ℝ, m' = (r : WithBot ℝ) := by
      cases m with
      | bot => exact ⟨s, by rw [hm']; rfl⟩
      | coe a => exact ⟨max a s, by rw [hm']; rw [← WithBot.coe_max]⟩
    obtain ⟨mr, hmr⟩ := hm'r
    have hκm' : m'.elim 0 (fun r => pow2 (-r)) = pow2 (-mr) := by rw [hmr]; rfl
    have hunbot : m'.unbotD 0 = mr := by rw [hmr]; rfl
    have hp : pow2 (s - m'.unbotD 0) = pow2 (-mr) * pow2 s := by
      rw [hunbot, ← pow2_add]; ring_nf
    have hl' : l * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
        + pow2 (s - m'.unbotD 0) = pow2 (-mr) * (L + pow2 s) := by
      cases m with
      | bot =>
        rw [hmL rfl]
        have hz : (WithBot.realExp2 (WithBot.realSub (⊥ : WithBot ℝ) m')).unbotD 0 = 0 := by
          rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
        rw [hz, mul_zero, zero_add, hp]; ring
      | coe a =>
        have hm'a : m' = ((max a s : ℝ) : WithBot ℝ) := by rw [hm']; rw [← WithBot.coe_max]
        have hmra : mr = max a s := by rw [hm'a] at hmr; exact (WithBot.coe_inj.mp hmr.symm)
        have hαa : (pow2 (-a)) * (WithBot.realExp2 (WithBot.realSub (↑a) m')).unbotD 0
            = pow2 (-mr) := by
          rw [hm'a, WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe]
          rw [show Real.exp ((a - max a s) * Real.log 2) = pow2 (a - max a s) from by
            simp [pow2, mul_comm], ← pow2_add, hmra]; ring_nf
        rw [hl, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [mul_right_comm, hαa, hp]; ring
    have hacc' : acc * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
        + pow2 (s - m'.unbotD 0) * v = pow2 (-mr) * (T + pow2 s * v) := by
      cases m with
      | bot =>
        rw [hmT rfl]
        have hz : (WithBot.realExp2 (WithBot.realSub (⊥ : WithBot ℝ) m')).unbotD 0 = 0 := by
          rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
        rw [hz, mul_zero, zero_add, hp]; ring
      | coe a =>
        have hm'a : m' = ((max a s : ℝ) : WithBot ℝ) := by rw [hm']; rw [← WithBot.coe_max]
        have hmra : mr = max a s := by rw [hm'a] at hmr; exact (WithBot.coe_inj.mp hmr.symm)
        have hαa : (pow2 (-a)) * (WithBot.realExp2 (WithBot.realSub (↑a) m')).unbotD 0
            = pow2 (-mr) := by
          rw [hm'a, WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe]
          rw [show Real.exp ((a - max a s) * Real.log 2) = pow2 (a - max a s) from by
            simp [pow2, mul_comm], ← pow2_add, hmra]; ring_nf
        rw [hacc, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [mul_right_comm, hαa, hp]; ring
    have step := ih m'
      (l * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0 + pow2 (s - m'.unbotD 0))
      (acc * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0 + pow2 (s - m'.unbotD 0) * v)
      (T + pow2 s * v) (L + pow2 s) (by rw [hl', hκm']) (by rw [hacc', hκm'])
      (by rw [hmr]; simp) (by rw [hmr]; simp)
    simpa [List.foldl_cons, aft3OsStepBot, hm', List.map_cons, add_assoc] using step

/-- The `WithBot ⊔`-fold is seed/direction-agnostic. -/
theorem aft3_foldl_sup_bot_eq_foldr (L : List (WithBot ℝ)) :
    L.foldl (· ⊔ ·) (⊥ : WithBot ℝ) = L.foldr (· ⊔ ·) (⊥ : WithBot ℝ) := by
  have gen : ∀ (m : WithBot ℝ), L.foldl (· ⊔ ·) m = m ⊔ L.foldr (· ⊔ ·) ⊥ := by
    induction L with
    | nil => intro m; simp
    | cons a t ih =>
      intro m
      simp only [List.foldl_cons, List.foldr_cons, ih]
      rw [max_assoc]
  rw [gen ⊥, bot_sup_eq]

/-- The `.1` channel of an `aft3OsStepBot` block fold: `m ⊔ blockSup`. -/
theorem aft3OsStepBot_block_fst (m : WithBot ℝ) (l acc : ℝ) (block : List (ℝ × ℝ)) :
    (block.foldl aft3OsStepBot (m, l, acc)).1
      = m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  rw [aft3StateBot_fst]
  induction block generalizing m with
  | nil => simp
  | cons a t ih =>
    simp only [List.map_cons, List.foldl_cons, List.foldr_cons]
    rw [ih]
    rw [show (m ⊔ ((a.1 : ℝ) : WithBot ℝ)) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) t))
          = m ⊔ (((a.1 : ℝ) : WithBot ℝ) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) t))) from by
      rw [sup_assoc]]

/-- **The block-at-once update equals the key-by-key `aft3OsStepBot` fold.** -/
theorem aft3OsStepBot_block_eq (m : WithBot ℝ) (l acc T L : ℝ) (block : List (ℝ × ℝ))
    (hl : l = (m.elim 0 (fun r => pow2 (-r))) * L)
    (hacc : acc = (m.elim 0 (fun r => pow2 (-r))) * T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0) :
    let M' := m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
    (M',
     l * (WithBot.realExp2 (WithBot.realSub m M')).unbotD 0
       + (block.map (fun p => pow2 (p.1 - M'.unbotD 0))).sum,
     acc * (WithBot.realExp2 (WithBot.realSub m M')).unbotD 0
       + (block.map (fun p => pow2 (p.1 - M'.unbotD 0) * p.2)).sum)
      = block.foldl aft3OsStepBot (m, l, acc) := by
  intro M'
  have hfst : (block.foldl aft3OsStepBot (m, l, acc)).1 = M' := by
    rw [aft3OsStepBot_block_fst]
  obtain ⟨hfold_l, hfold_acc⟩ := aft3OsStepBot_foldl_consistent block m l acc T L hl hacc hmL hmT
  rw [hfst] at hfold_l hfold_acc
  have hM'eq : M' = m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := rfl
  cases hM' : M' with
  | bot =>
    have hempty : block = [] := by
      rcases block with _ | ⟨a, t⟩
      · rfl
      · exfalso
        have : ((a.1 : ℝ) : WithBot ℝ) ≤ M' := by
          rw [hM'eq]
          exact le_sup_of_le_right (by simp only [List.map_cons, List.foldr_cons]; exact le_sup_left)
        rw [hM'] at this
        exact absurd (le_bot_iff.mp this) (WithBot.coe_ne_bot)
    have hm0 : m = ⊥ := by
      rw [hM'eq, hempty] at hM'
      simpa only [List.map_nil, List.foldr_nil, sup_bot_eq] using hM'
    have hl0 : l = 0 := by rw [hl, hm0]; simp [hmL hm0]
    have hacc0 : acc = 0 := by rw [hacc, hm0]; simp [hmT hm0]
    subst hempty
    rw [hl0, hacc0]
    simp only [List.foldl_nil, List.map_nil, List.sum_nil, add_zero, mul_zero, zero_mul]
    rw [hm0]
  | coe Mr =>
    rw [hM'] at hfst hfold_l hfold_acc
    have hlα : l * (WithBot.realExp2 (WithBot.realSub m (↑Mr : WithBot ℝ))).unbotD 0 = pow2 (-Mr) * L := by
      cases hm : m with
      | bot =>
        rw [hl, hm, show ((⊥ : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = 0 from rfl,
          zero_mul, hmL hm]; ring
      | coe a =>
        rw [hl, hm, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe,
          show Real.exp ((a - Mr) * Real.log 2) = pow2 (a - Mr) from by simp [pow2, mul_comm]]
        rw [mul_right_comm, ← pow2_add]; ring_nf
    have haccα : acc * (WithBot.realExp2 (WithBot.realSub m (↑Mr : WithBot ℝ))).unbotD 0 = pow2 (-Mr) * T := by
      cases hm : m with
      | bot =>
        rw [hacc, hm, show ((⊥ : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = 0 from rfl,
          zero_mul, hmT hm]; ring
      | coe a =>
        rw [hacc, hm, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe,
          show Real.exp ((a - Mr) * Real.log 2) = pow2 (a - Mr) from by simp [pow2, mul_comm]]
        rw [mul_right_comm, ← pow2_add]; ring_nf
    have hsumL : (block.map (fun p => pow2 (p.1 - (↑Mr : WithBot ℝ).unbotD 0))).sum
        = pow2 (-Mr) * (block.map (fun p => pow2 p.1)).sum := by
      have := VeriTile.Triton.sum_map_pow2_sub ((↑Mr : WithBot ℝ).unbotD 0) block (fun _ => 1)
      simp only [mul_one] at this
      rw [this, WithBot.unbotD_coe]
    have hsumT : (block.map (fun p => pow2 (p.1 - (↑Mr : WithBot ℝ).unbotD 0) * p.2)).sum
        = pow2 (-Mr) * (block.map (fun p => pow2 p.1 * p.2)).sum := by
      rw [VeriTile.Triton.sum_map_pow2_sub ((↑Mr : WithBot ℝ).unbotD 0) block (fun p => p.2), WithBot.unbotD_coe]
    refine Prod.ext hfst.symm (Prod.ext ?_ ?_)
    · rw [hfold_l, hlα, hsumL, show ((↑Mr : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-Mr) from rfl]; ring
    · rw [hfold_acc, haccα, hsumT, show ((↑Mr : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-Mr) from rfl]; ring

/-- Generic threshold-split for a `.val`-ascending `filterMap`: the `j.val < hi₂`
window splits into the `j.val < t` prefix and the `t ≤ j.val < hi₂` block
(`t ≤ hi₂`). (Ported from the flash-attn private helper.) -/
private theorem aft3_filterMap_window_split {n : Nat} (l : List (Fin n))
    (hsorted : l.Pairwise (fun a b => a.val < b.val))
    (t hi₂ : Nat) (Q : Fin n → Prop) [DecidablePred Q]
    (g : Fin n → ℝ × ℝ) (hle : t ≤ hi₂) :
    l.filterMap (fun j => if Q j ∧ j.val < hi₂ then some (g j) else none)
      = l.filterMap (fun j => if Q j ∧ j.val < t then some (g j) else none)
        ++ l.filterMap (fun j => if t ≤ j.val ∧ j.val < hi₂ ∧ Q j then some (g j) else none) := by
  induction l with
  | nil => simp
  | cons a tl ih =>
    have htl : tl.Pairwise (fun x y => x.val < y.val) := (List.pairwise_cons.mp hsorted).2
    have hahead : ∀ b ∈ tl, a.val < b.val := (List.pairwise_cons.mp hsorted).1
    rw [List.filterMap_cons, List.filterMap_cons, List.filterMap_cons]
    by_cases hlt : a.val < t
    · rw [ih htl]
      have hnb : ¬ (t ≤ a.val ∧ a.val < hi₂ ∧ Q a) := fun h => (Nat.not_le.mpr hlt) h.1
      rw [if_neg hnb]
      by_cases hQ : Q a
      · have h2 : a.val < hi₂ := lt_of_lt_of_le hlt hle
        rw [if_pos (And.intro hQ h2 : Q a ∧ a.val < hi₂),
          if_pos (And.intro hQ hlt : Q a ∧ a.val < t)]
        rfl
      · rw [if_neg (fun h : Q a ∧ a.val < hi₂ => hQ h.1),
          if_neg (fun h : Q a ∧ a.val < t => hQ h.1)]
    · have hge : t ≤ a.val := Nat.not_lt.mp hlt
      have htail_prefix : tl.filterMap (fun j => if Q j ∧ j.val < t then some (g j) else none) = [] := by
        apply List.filterMap_eq_nil_iff.mpr
        intro b hb
        have hab : a.val < b.val := hahead b hb
        have hbt : ¬ (b.val < t) := by omega
        simp [hbt]
      rw [ih htl, htail_prefix]
      have hnp : ¬ (Q a ∧ a.val < t) := fun h => hlt h.2
      rw [if_neg hnp]
      by_cases hQ : Q a
      · by_cases h2 : a.val < hi₂
        · rw [if_pos (And.intro hQ h2 : Q a ∧ a.val < hi₂),
            if_pos (And.intro hge (And.intro h2 hQ) : t ≤ a.val ∧ a.val < hi₂ ∧ Q a)]
          rfl
        · rw [if_neg (fun h : Q a ∧ a.val < hi₂ => h2 h.2),
            if_neg (fun h : t ≤ a.val ∧ a.val < hi₂ ∧ Q a => h2 h.2.1)]
      · rw [if_neg (fun h : Q a ∧ a.val < hi₂ => hQ h.1),
          if_neg (fun h : t ≤ a.val ∧ a.val < hi₂ ∧ Q a => hQ h.2.2)]

/-! ## Predicate-windowed ⊥-seeded recurrence (deliverable 3, windowed layer)

`aft3KeysUpto` is the per-row key prefix the kernel has streamed after window
`[0, hi)`: keys `j < hi` passing the case `keep` predicate, scored
`keyScale3 j · (Q row i · K row j)`, valued `V (j, d)`. `aft3RunningMax` /
`aft3StateBot` are the ⊥-seeded running max / `(max, denom, acc)` over that
prefix. At the full window `hi = 128` this is exactly `attnKeyListPred`, so the
final ratio reads off `attentionFwdTriton3CaseNOutSpec` via the streaming
bridges. -/

/-- Per-row windowed key prefix `[0, hi)` filtered by `keep i j`. -/
noncomputable def aft3KeysUpto
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin 64) (d : Fin 64) :
    List (ℝ × ℝ) :=
  (List.finRange 128).filterMap (fun j : Fin 128 =>
    if j.val < hi ∧ keep i j then
      some (keyScale j * Finset.univ.sum (fun e : Fin 64 =>
              qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)),
            vT (j, d, PUnit.unit))
    else none)

/-- ⊥-seeded running max over the windowed prefix (the `WithBot ⊔`-fold). -/
noncomputable def aft3RunningMax
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin 64) (d : Fin 64) :
    WithBot ℝ :=
  ((aft3KeysUpto qT kT vT keyScale keep hi i d).map
    (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥

/-- ⊥-seeded running `(max, denom, acc)` after the windowed prefix `[0, hi)`. -/
noncomputable def aft3StateBot
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin 64) (d : Fin 64) :
    WithBot ℝ × ℝ × ℝ :=
  (aft3KeysUpto qT kT vT keyScale keep hi i d).foldl aft3OsStepBot (⊥, 0, 0)

/-- Block `c` per-row key list: kept keys with `c·64 ≤ j < (c+1)·64`. -/
noncomputable def aft3Block
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (c : Nat) (i : Fin 64) (d : Fin 64) :
    List (ℝ × ℝ) :=
  (List.finRange 128).filterMap (fun j : Fin 128 =>
    if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ keep i j then
      some (keyScale j * Finset.univ.sum (fun e : Fin 64 =>
              qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)),
            vT (j, d, PUnit.unit))
    else none)

/-- The running max of `aft3StateBot` is `aft3RunningMax`. -/
theorem aft3StateBot_fst_eq_runningMax
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin 64) (d : Fin 64) :
    (aft3StateBot qT kT vT keyScale keep hi i d).1
      = aft3RunningMax qT kT vT keyScale keep hi i d := by
  rw [aft3StateBot, aft3StateBot_fst, aft3RunningMax, aft3_foldl_sup_bot_eq_foldr]

/-- ⊥-seeded state at the empty window is `(⊥, 0, 0)` — the kernel's preLoop
init. The base case making `attnInvariant … 0` satisfiable. -/
theorem aft3StateBot_zero
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (i : Fin 64) (d : Fin 64) :
    aft3StateBot qT kT vT keyScale keep 0 i d = (⊥, 0, 0) := by
  unfold aft3StateBot aft3KeysUpto
  rw [show (List.finRange 128).filterMap
        (fun j : Fin 128 => if j.val < 0 ∧ keep i j
          then some (keyScale j * Finset.univ.sum (fun e : Fin 64 =>
                qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)), vT (j, d, PUnit.unit))
          else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-- ⊥-seeded running max at the empty window is `⊥`. -/
theorem aft3RunningMax_zero
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (i : Fin 64) (d : Fin 64) :
    aft3RunningMax qT kT vT keyScale keep 0 i d = ⊥ := by
  unfold aft3RunningMax aft3KeysUpto
  rw [show (List.finRange 128).filterMap
        (fun j : Fin 128 => if j.val < 0 ∧ keep i j
          then some (keyScale j * Finset.univ.sum (fun e : Fin 64 =>
                qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)), vT (j, d, PUnit.unit))
          else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-- **Window split**: keys through `c+1` blocks = keys through `c` blocks ++ block `c`. -/
theorem aft3KeysUpto_succ
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (c : Nat) (i : Fin 64) (d : Fin 64) :
    aft3KeysUpto qT kT vT keyScale keep ((c + 1) * 64) i d
      = aft3KeysUpto qT kT vT keyScale keep (c * 64) i d
        ++ aft3Block qT kT vT keyScale keep c i d := by
  unfold aft3KeysUpto aft3Block
  rw [show (List.finRange 128).filterMap
        (fun j : Fin 128 => if j.val < (c + 1) * 64 ∧ keep i j then
          some (keyScale j * Finset.univ.sum (fun e : Fin 64 =>
              qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)), vT (j, d, PUnit.unit)) else none)
      = (List.finRange 128).filterMap
        (fun j : Fin 128 => if keep i j ∧ j.val < (c + 1) * 64 then
          some (keyScale j * Finset.univ.sum (fun e : Fin 64 =>
              qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)), vT (j, d, PUnit.unit)) else none)
      from List.filterMap_congr (fun j _ => by simp only [and_comm])]
  rw [aft3_filterMap_window_split (List.finRange 128) (List.pairwise_lt_finRange 128)
    (c * 64) ((c + 1) * 64) (fun j => keep i j)
    (fun j => (keyScale j * Finset.univ.sum (fun e : Fin 64 =>
        qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)), vT (j, d, PUnit.unit)))
    (by nlinarith [Nat.zero_le 64])]
  refine congrArg₂ (· ++ ·) ?_ ?_
  · apply List.filterMap_congr; intro j _; simp only [and_comm]
  · apply List.filterMap_congr; intro j _
    by_cases h1 : c * 64 ≤ j.val <;> by_cases h2 : j.val < (c + 1) * 64 <;>
      by_cases h3 : keep i j <;> simp [h1, h2, h3, and_assoc]

/-- **One-block ⊥-seeded advance**: `aft3StateBot` after `c+1` blocks is the
`aft3OsStepBot` fold of block `c`'s keys onto `aft3StateBot` after `c` blocks. -/
theorem aft3StateBot_succ
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (c : Nat) (i : Fin 64) (d : Fin 64) :
    aft3StateBot qT kT vT keyScale keep ((c + 1) * 64) i d
      = (aft3Block qT kT vT keyScale keep c i d).foldl aft3OsStepBot
          (aft3StateBot qT kT vT keyScale keep (c * 64) i d) := by
  unfold aft3StateBot
  rw [aft3KeysUpto_succ, List.foldl_append]

/-- At the full window `hi = 128`, the windowed prefix is the full
`attnKeyListPred`. -/
theorem aft3KeysUpto_full
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (i : Fin 64) (d : Fin 64) :
    aft3KeysUpto qT kT vT keyScale keep 128 i d
      = attnKeyListPred qT kT vT keyScale keep i d := by
  unfold aft3KeysUpto VeriTile.Triton.attnKeyListPred
  apply List.filterMap_congr
  intro j _
  simp only [j.isLt, true_and]

/-- ⊥-seeded denominator = `κ(aft3RunningMax)·Σpow2 score`. -/
theorem aft3StateBot_snd_fst
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin 64) (d : Fin 64) :
    (aft3StateBot qT kT vT keyScale keep hi i d).2.1
      = ((aft3RunningMax qT kT vT keyScale keep hi i d).elim 0 (fun r => pow2 (-r)))
        * ((aft3KeysUpto qT kT vT keyScale keep hi i d).map (fun p => pow2 p.1)).sum := by
  have h := (aft3OsStepBot_foldl_consistent (aft3KeysUpto qT kT vT keyScale keep hi i d)
    ⊥ 0 0 0 0 (by simp) (by simp) (fun _ => rfl) (fun _ => rfl)).1
  rw [aft3StateBot]
  rw [show (List.foldl aft3OsStepBot (⊥, 0, 0) (aft3KeysUpto qT kT vT keyScale keep hi i d)).2.1
        = _ from h]
  rw [show (List.foldl aft3OsStepBot (⊥, 0, 0) (aft3KeysUpto qT kT vT keyScale keep hi i d)).1
        = aft3RunningMax qT kT vT keyScale keep hi i d from by
    rw [aft3StateBot_fst, aft3RunningMax, aft3_foldl_sup_bot_eq_foldr]]
  rw [zero_add]

/-- ⊥-seeded accumulator = `κ(aft3RunningMax)·Σpow2 score·v`. -/
theorem aft3StateBot_snd_snd
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin 64) (d : Fin 64) :
    (aft3StateBot qT kT vT keyScale keep hi i d).2.2
      = ((aft3RunningMax qT kT vT keyScale keep hi i d).elim 0 (fun r => pow2 (-r)))
        * ((aft3KeysUpto qT kT vT keyScale keep hi i d).map (fun p => pow2 p.1 * p.2)).sum := by
  have h := (aft3OsStepBot_foldl_consistent (aft3KeysUpto qT kT vT keyScale keep hi i d)
    ⊥ 0 0 0 0 (by simp) (by simp) (fun _ => rfl) (fun _ => rfl)).2
  rw [aft3StateBot]
  rw [show (List.foldl aft3OsStepBot (⊥, 0, 0) (aft3KeysUpto qT kT vT keyScale keep hi i d)).2.2
        = _ from h]
  rw [show (List.foldl aft3OsStepBot (⊥, 0, 0) (aft3KeysUpto qT kT vT keyScale keep hi i d)).1
        = aft3RunningMax qT kT vT keyScale keep hi i d from by
    rw [aft3StateBot_fst, aft3RunningMax, aft3_foldl_sup_bot_eq_foldr]]
  rw [zero_add]

/-- The ratio `acc/denom` of the ⊥-seeded state equals that of the real-0-seeded
`osStep` fold — the seed cancels (`pow2` never zero) when the window is nonempty. -/
theorem aft3StateBot_ratio_eq
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin 64) (d : Fin 64)
    (hne : aft3RunningMax qT kT vT keyScale keep hi i d ≠ ⊥) :
    (aft3StateBot qT kT vT keyScale keep hi i d).2.2
        / (aft3StateBot qT kT vT keyScale keep hi i d).2.1
      = (let st := (aft3KeysUpto qT kT vT keyScale keep hi i d).foldl osStep (0, 0, 0)
         st.2.2 / st.2.1) := by
  rw [aft3StateBot_snd_fst, aft3StateBot_snd_snd]
  simp only
  have hcL := (VeriTile.Triton.osStep_foldl_consistent
    (aft3KeysUpto qT kT vT keyScale keep hi i d) 0 0 0 0 0 (by simp) (by simp)).1
  have hcT := (VeriTile.Triton.osStep_foldl_consistent
    (aft3KeysUpto qT kT vT keyScale keep hi i d) 0 0 0 0 0 (by simp) (by simp)).2
  rw [show (List.foldl osStep (0, 0, 0) (aft3KeysUpto qT kT vT keyScale keep hi i d)).2.1
        = _ from hcL,
      show (List.foldl osStep (0, 0, 0) (aft3KeysUpto qT kT vT keyScale keep hi i d)).2.2
        = _ from hcT]
  cases hM : aft3RunningMax qT kT vT keyScale keep hi i d with
  | bot => exact absurd hM hne
  | coe r =>
    rw [show ((↑r : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-r) from rfl]
    simp only [zero_add]
    rw [mul_div_mul_left _ _ (ne_of_gt (pow2_pos _)),
        mul_div_mul_left _ _ (ne_of_gt (pow2_pos _))]

/-- **Full-window ⊥-seeded state reads off the case-1 closed form.** -/
theorem aft3StateBot_full_eq_spec_case1
    (s : BlockState) (Q K V : RegionName) (i d : Fin 64)
    (hne : aft3RunningMax (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3
        (fun i j => natSlidingWindowKeep (s.pids 0) i j) 128 i d ≠ ⊥) :
    (let st := aft3StateBot (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3
        (fun i j => natSlidingWindowKeep (s.pids 0) i j) 128 i d
     st.2.2 / st.2.1)
      = attentionFwdTriton3Case1OutSpec s Q K V (i, d, PUnit.unit) := by
  simp only
  rw [aft3StateBot_ratio_eq _ _ _ _ _ _ _ _ hne]
  rw [aft3KeysUpto_full]
  exact (attentionFwdTriton3Case1OutSpec_eq_streaming s Q K V i d).symm

/-- **Full-window ⊥-seeded state reads off the case-2 closed form.** -/
theorem aft3StateBot_full_eq_spec_case2
    (s : BlockState) (Q K V : RegionName) (i d : Fin 64)
    (hne : aft3RunningMax (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3
        (fun i j => natComplementSlidingWindowKeep (s.pids 0) i j) 128 i d ≠ ⊥) :
    (let st := aft3StateBot (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3
        (fun i j => natComplementSlidingWindowKeep (s.pids 0) i j) 128 i d
     st.2.2 / st.2.1)
      = attentionFwdTriton3Case2OutSpec s Q K V (i, d, PUnit.unit) := by
  simp only
  rw [aft3StateBot_ratio_eq _ _ _ _ _ _ _ _ hne]
  rw [aft3KeysUpto_full]
  exact (attentionFwdTriton3Case2OutSpec_eq_streaming s Q K V i d).symm

/-- **Full-window ⊥-seeded state reads off the case-3 closed form.** -/
theorem aft3StateBot_full_eq_spec_case3
    (s : BlockState) (Q K V : RegionName) (i d : Fin 64)
    (hne : aft3RunningMax (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3
        (fun i j => noWindowKeep i j) 128 i d ≠ ⊥) :
    (let st := aft3StateBot (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3
        (fun i j => noWindowKeep i j) 128 i d
     st.2.2 / st.2.1)
      = attentionFwdTriton3Case3OutSpec s Q K V (i, d, PUnit.unit) := by
  simp only
  rw [aft3StateBot_ratio_eq _ _ _ _ _ _ _ _ hne]
  rw [aft3KeysUpto_full]
  exact (attentionFwdTriton3Case3OutSpec_eq_streaming s Q K V i d).symm


/-! ### `l_i = 1` seed reconciliation

The kernel seeds `l_i = tl.zeros + 1.0` (not `0`). On the first streamed key the
running max transitions from `⊥`, forcing `α = (realExp2 (realSub ⊥ m')).unbotD 0
= 0`, which annihilates the `1` carry. Hence the seed-`1` fold and the seed-`0`
fold agree on every nonempty key prefix. `aft3StateBot1` is the faithful seed-`1`
state; it equals `aft3StateBot` on nonempty windows. -/

/-- **⊥-seed independence of the `l`/`acc` carries.** From a `⊥`-max start the
first key resets the carries, so the `aft3OsStepBot` fold over a nonempty list is
independent of the initial `l`/`acc`. -/
theorem aft3OsStepBot_bot_seed_indep (xs : List (ℝ × ℝ)) (hne : xs ≠ [])
    (l acc l' acc' : ℝ) :
    xs.foldl aft3OsStepBot (⊥, l, acc) = xs.foldl aft3OsStepBot (⊥, l', acc') := by
  obtain ⟨x, t, rfl⟩ := List.exists_cons_of_ne_nil hne
  obtain ⟨s, v⟩ := x
  have hstep : ∀ L A : ℝ, aft3OsStepBot (⊥, L, A) (s, v)
      = (((s : ℝ) : WithBot ℝ), pow2 (s - s), pow2 (s - s) * v) := by
    intro L A
    simp only [aft3OsStepBot, bot_sup_eq]
    have hα : (WithBot.realExp2 (WithBot.realSub (⊥ : WithBot ℝ) ((s : ℝ) : WithBot ℝ))).unbotD 0 = 0 := by
      rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
    have hub : (((s : ℝ) : WithBot ℝ)).unbotD 0 = s := by rfl
    rw [hα, hub]
    simp
  simp only [List.foldl_cons, hstep]

/-- ⊥-seeded running state from the kernel's `l_i = 1` seed. -/
noncomputable def aft3StateBot1
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin 64) (d : Fin 64) :
    WithBot ℝ × ℝ × ℝ :=
  (aft3KeysUpto qT kT vT keyScale keep hi i d).foldl aft3OsStepBot (⊥, 1, 0)

/-- The faithful seed-`1` state equals the seed-`0` state whenever the window is
nonempty (`aft3RunningMax ≠ ⊥`). -/
theorem aft3StateBot1_eq_aft3StateBot
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin 64) (d : Fin 64)
    (hne : aft3RunningMax qT kT vT keyScale keep hi i d ≠ ⊥) :
    aft3StateBot1 qT kT vT keyScale keep hi i d
      = aft3StateBot qT kT vT keyScale keep hi i d := by
  have hxs : aft3KeysUpto qT kT vT keyScale keep hi i d ≠ [] := by
    intro h
    apply hne
    unfold aft3RunningMax
    rw [h]; rfl
  unfold aft3StateBot1 aft3StateBot
  exact aft3OsStepBot_bot_seed_indep _ hxs 1 0 0 0


/-! ## Loop invariant (deliverable 4)

`attnInvariant` binds the kernel's live registers after `c` key blocks (counter
`i = c·64`, window `[0, i)`): the streaming state `m_i`/`l_i`/`acc` equals the
⊥-seeded `aft3RunningMax`/`aft3StateBot` over the first `i` kept keys (per the
case `keep` predicate), and the setup registers (`q`, `qk_scale`, the `K`/`V`
block pointers advanced by `i`, `start_m`, `off_hz`, the `m_ptrs`/`l_ptrs`
vectors, the `O_block_ptr`) carry their loop-entry values. Memory / undef are
preserved. Strides specialized to the Python case layout. -/

noncomputable def attnInvariant
    (Q K V M Out L : RegionName) (s0 : BlockState)
    (keep : Fin 64 → Fin 128 → Prop) [∀ i j, Decidable (keep i j)]
    (i : Nat) (s : BlockState) : Prop :=
  let base := baseOffset3 s0
  let qT := qTile3 s0 Q
  let kT := kTile3 s0 K
  let vT := vTile3 s0 V
  s.pids = s0.pids ∧ i % 64 = 0 ∧ i ≤ 128 ∧
  (s.regs .real [64] "m_i" = some ⟨fun r : TileIndex [64] =>
      aft3RunningMax qT kT vT keyScale3 keep i r.1 ⟨0, by norm_num⟩⟩) ∧
  (s.regs .real [64] "l_i" = some ⟨fun r : TileIndex [64] =>
      ((aft3StateBot1 qT kT vT keyScale3 keep i r.1 ⟨0, by norm_num⟩).2.1 : ℝ)⟩) ∧
  (s.regs .real [64, 64] "acc" = some ⟨fun idx : TileIndex [64, 64] =>
      ((aft3StateBot1 qT kT vT keyScale3 keep i idx.1 idx.2.1).2.2 : ℝ)⟩) ∧
  (s.regs .real [64, 64] "q" = some ⟨fun idx : TileIndex [64, 64] =>
      some (qT idx)⟩) ∧
  (s.regs .real [] "qk_scale" = some (Tile.scalar (some ((1 / 8 : ℝ) * 1.4426950408889634)))) ∧
  (s.regs .nat [] "start_m" = some (Tile.scalar (s0.pids 0))) ∧
  (s.regs .nat [] "off_hz" = some (Tile.scalar (s0.pids 1))) ∧
  (s.regs .blockPtr [64, 64] "K_block_ptr" = some
    (⟨fun _ : TileIndex [64, 64] =>
      { region := K, baseOffset := s0.pids 1 / 4 * 32768 + s0.pids 1 % 4 * 8192,
        parentShape := [64, 128], blockShape := [64, 64], strides := [1, 64],
        offsets := [0, i] }⟩)) ∧
  (s.regs .blockPtr [64, 64] "V_block_ptr" = some
    (⟨fun _ : TileIndex [64, 64] =>
      { region := V, baseOffset := s0.pids 1 / 4 * 32768 + s0.pids 1 % 4 * 8192,
        parentShape := [128, 64], blockShape := [64, 64], strides := [64, 1],
        offsets := [i, 0] }⟩)) ∧
  (s.regs .ptr [64] "m_ptrs" = some
    (Tile.ptrAdd Broadcast.scalarL (Tile.scalar (M.cast, (0 : Nat)))
      (Tile.bop NumericDType.nat.add Broadcast.scalarL (Tile.scalar (s0.pids 1 * 128))
        (Tile.vec (fun r : Fin 64 => s0.pids 0 * 64 + r.val))))) ∧
  (s.regs .ptr [64] "l_ptrs" = some
    (Tile.ptrAdd Broadcast.scalarL (Tile.scalar (L.cast, (0 : Nat)))
      (Tile.bop NumericDType.nat.add Broadcast.scalarL (Tile.scalar (s0.pids 1 * 128))
        (Tile.vec (fun r : Fin 64 => s0.pids 0 * 64 + r.val))))) ∧
  (s.regs .blockPtr [64, 64] "O_block_ptr" = some
    (⟨fun _ : TileIndex [64, 64] =>
      { region := Out, baseOffset := s0.pids 1 / 4 * 32768 + s0.pids 1 % 4 * 8192,
        parentShape := [128, 64], blockShape := [64, 64], strides := [64, 1],
        offsets := [s0.pids 0 * 64, 0] }⟩)) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-! ### preLoop evaluation (deliverable 4) -/

/-- `makeBlockPtrDyn` evaluation recipe. -/
theorem aft3_makeBlockPtrDyn_eval (region : RegionName) (baseOffset : Op .nat [])
    (parentShape : List Nat) (blockShape : TileShape)
    (strides offsets : List Nat) (s : BlockState) (base : Nat)
    (hb : evalOp baseOffset s = some (Tile.scalar base)) :
    evalOp (.makeBlockPtrDyn region baseOffset parentShape blockShape strides offsets) s
      = some (⟨fun _ : TileIndex blockShape =>
          { region := region, baseOffset := base, parentShape := parentShape,
            blockShape := blockShape, strides := strides, offsets := offsets }⟩) := by
  simp only [evalOp, hb, Option.bind]
  rfl

/-- `makeBlockPtrDynOffsets` with dynamic row offset, literal `0` column. -/
theorem aft3_makeBlockPtr_rowcol_eval (region : RegionName) (baseOffset : Op .nat [])
    (parentShape : List Nat) (blockShape : TileShape) (strides : List Nat)
    (rowOp : Op .nat []) (s : BlockState) (base rowOff : Nat)
    (hb : evalOp baseOffset s = some (Tile.scalar base))
    (hr : evalOp rowOp s = some (Tile.scalar rowOff)) :
    evalOp (.makeBlockPtrDynOffsets region baseOffset parentShape blockShape strides
        [rowOp, Op.constNat 0]) s
      = some (⟨fun _ : TileIndex blockShape =>
          { region := region, baseOffset := base, parentShape := parentShape,
            blockShape := blockShape, strides := strides, offsets := [rowOff, 0] }⟩) := by
  simp only [evalOp, hb, hr, Option.bind, List.mapM, List.mapM.loop, evalOp_constNat, Tile.scalar]
  rfl

/-- `Stmt.ifThenElse` with a constexpr-true condition runs the then-branch. -/
theorem aft3_ifThenElse_true {cond : Op .bool []} {t e : List Stmt} {s : BlockState}
    (hc : evalOp cond s = some (Tile.scalar Bool.true)) :
    stepStmt (.ifThenElse cond t e) s = stepStmts t s := by
  simp only [stepStmt, hc, Option.bind, Tile.scalar]
  rfl

/-- Eval helper for `floorDiv` (no `@[simp]` form). -/
theorem aft3_evalOp_floorDiv {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.floorDiv h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.floorDiv bc vx vy)) := by
  simp [evalOp]

/-- Eval helper for `mod` (no `@[simp]` form). -/
theorem aft3_evalOp_mod {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.mod h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.mod bc vx vy)) := by
  simp [evalOp]

/-- Eval helper for `ptrAdd` (no `@[simp]` form). -/
theorem aft3_evalOp_ptrAdd {a b shape} (bc : Broadcast a b shape)
    (ptr : Op .ptr a) (off : Op .nat b) (s : BlockState) :
    evalOp (.ptrAdd bc ptr off) s = (do
      let ptrs ← evalOp ptr s; let offs ← evalOp off s;
      some (Tile.ptrAdd bc ptrs offs)) := by simp [evalOp]

/-- Eval helper for `ptrBase` (the base region pointer). -/
theorem aft3_evalOp_ptrBase (region : RegionName) (s : BlockState) :
    evalOp (.ptrBase region) s = some (Tile.scalar (region.cast, 0)) := by simp [evalOp]

/-- `Stmt.ifThen` with a constexpr-true condition runs the body. -/
theorem aft3_ifThen_true {cond : Op .bool []} {body : List Stmt} {s : BlockState}
    (hc : evalOp cond s = some (Tile.scalar Bool.true)) :
    stepStmt (.ifThen cond body) s = stepStmts body s := by
  simp only [stepStmt, hc, Option.bind, Tile.scalar]
  rfl

/-- `Stmt.ifThenElse` with a constexpr-false condition runs the else-branch. -/
theorem aft3_ifThenElse_false {cond : Op .bool []} {t e : List Stmt} {s : BlockState}
    (hc : evalOp cond s = some (Tile.scalar Bool.false)) :
    stepStmt (.ifThenElse cond t e) s = stepStmts e s := by
  simp only [stepStmt, hc, Option.bind, Tile.scalar]
  rfl

/-- The constexpr `INIT != 0` / `IS_EVEN_M != 0` condition (`Op.ne 1 0`) is true. -/
theorem aft3_ne_one_zero_true (s : BlockState) :
    evalOp (Op.ne ComparableDType.nat Broadcast.nil (Op.constNat 1) (Op.constNat 0)) s
      = some (Tile.scalar Bool.true) := by
  unfold evalOp
  simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- The constexpr `COMPLEMENT != 0` condition (`Op.ne 0 0`) is false. -/
theorem aft3_ne_zero_zero_false (s : BlockState) :
    evalOp (Op.ne ComparableDType.nat Broadcast.nil (Op.constNat 0) (Op.constNat 0)) s
      = some (Tile.scalar Bool.false) := by
  unfold evalOp
  simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `Stmt.ifThen` with a constexpr-false condition is a no-op. -/
theorem aft3_ifThen_false {cond : Op .bool []} {body : List Stmt} {s : BlockState}
    (hc : evalOp cond s = some (Tile.scalar Bool.false)) :
    stepStmt (.ifThen cond body) s = some s := by
  simp only [stepStmt, hc, Option.bind, Tile.scalar]
  rfl

/-! ### preLoop execution (deliverable 4, exec side)

Steps the 20 lowered `aft3PreLoop` statements from a clean state to a loop-entry
state satisfying `attnInvariant … 0`. The scalar / pointer / index prefix is
discharged by `simp` (the `@[simp] evalOp_*`/`setReg` lemmas thread register
lookups through the `setReg` chain); the `q` block-pointer load is then threaded
through `aft3_load_v_eval` with the just-set `Q_block_ptr` readback (abstract,
never whnf-ing the literal state). -/

/-- The prefix of `aft3PreLoop` before the `INIT` branch (stmts 0–15: the 9
scalar offsets, 4 block pointers, `offs_m`, `m_ptrs`, `l_ptrs`). -/
noncomputable def aft3PreLoopScalars (Q K V M Out L : RegionName) : List Stmt :=
  (aft3PreLoop Q K V M Out L).take 16

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- The scalar/pointer/index prefix steps to a state exposing the register
readbacks the `INIT`/`qk_scale`/`q`-load chunks need. -/
theorem aft3PreLoopScalars_eval (Q K V M Out L : RegionName) (s : BlockState) :
    ∃ s11, stepStmts (aft3PreLoopScalars Q K V M Out L) s = some s11
      ∧ s11.pids = s.pids ∧ s11.mem = s.mem ∧ s11.undef = s.undef
      ∧ s11.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s11.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 1))
      ∧ s11.regs .blockPtr [64, 64] "Q_block_ptr" = some
          (⟨fun _ : TileIndex [64, 64] =>
            { region := Q, baseOffset := s.pids 1 / 4 * 32768 + s.pids 1 % 4 * 8192,
              parentShape := [128, 64], blockShape := [64, 64], strides := [64, 1],
              offsets := [s.pids 0 * 64, 0] }⟩)
      ∧ s11.regs .blockPtr [64, 64] "K_block_ptr" = some
          (⟨fun _ : TileIndex [64, 64] =>
            { region := K, baseOffset := s.pids 1 / 4 * 32768 + s.pids 1 % 4 * 8192,
              parentShape := [64, 128], blockShape := [64, 64], strides := [1, 64],
              offsets := [0, 0] }⟩)
      ∧ s11.regs .blockPtr [64, 64] "V_block_ptr" = some
          (⟨fun _ : TileIndex [64, 64] =>
            { region := V, baseOffset := s.pids 1 / 4 * 32768 + s.pids 1 % 4 * 8192,
              parentShape := [128, 64], blockShape := [64, 64], strides := [64, 1],
              offsets := [0, 0] }⟩)
      ∧ s11.regs .blockPtr [64, 64] "O_block_ptr" = some
          (⟨fun _ : TileIndex [64, 64] =>
            { region := Out, baseOffset := s.pids 1 / 4 * 32768 + s.pids 1 % 4 * 8192,
              parentShape := [128, 64], blockShape := [64, 64], strides := [64, 1],
              offsets := [s.pids 0 * 64, 0] }⟩)
      ∧ s11.regs .ptr [64] "m_ptrs" = some
          (Tile.ptrAdd Broadcast.scalarL (Tile.scalar (M.cast, (0 : Nat)))
            (Tile.bop NumericDType.nat.add Broadcast.scalarL (Tile.scalar (s.pids 1 * 128))
              (Tile.vec (fun r : Fin 64 => s.pids 0 * 64 + r.val))))
      ∧ s11.regs .ptr [64] "l_ptrs" = some
          (Tile.ptrAdd Broadcast.scalarL (Tile.scalar (L.cast, (0 : Nat)))
            (Tile.bop NumericDType.nat.add Broadcast.scalarL (Tile.scalar (s.pids 1 * 128))
              (Tile.vec (fun r : Fin 64 => s.pids 0 * 64 + r.val)))) := by
  unfold aft3PreLoopScalars aft3PreLoop
  simp only [List.take_succ_cons, List.take_zero, List.append_nil]
  -- stmt 0: start_m = programId 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  -- stmt 1: off_hz = programId 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  -- stmt 2: off_z = off_hz / 4
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat 4)) _
        = some (Tile.scalar (s.pids 1 / 4)) from by
      rw [aft3_evalOp_floorDiv]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]; rfl))]
  -- stmt 3: off_h = off_hz % 4
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat 4)) _
        = some (Tile.scalar (s.pids 1 % 4)) from by
      rw [aft3_evalOp_mod]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]; rfl))]
  -- stmt 4: off_hkv = off_h / (4 / 4)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h") (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat 4) (Op.constNat 4))) _
        = some (Tile.scalar (s.pids 1 % 4 / (4 / 4))) from by
      rw [aft3_evalOp_floorDiv, aft3_evalOp_floorDiv]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]; rfl))]
  -- stmt 5: q_offset = off_z*32768 + off_h*8192
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat 32768)) (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h") (Op.constNat 8192))) _
        = some (Tile.scalar (s.pids 1 / 4 * 32768 + s.pids 1 % 4 * 8192)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]; rfl))]
  -- stmt 6: k_offset = off_z*32768 + off_hkv*8192
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat 32768)) (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hkv") (Op.constNat 8192))) _
        = some (Tile.scalar (s.pids 1 / 4 * 32768 + s.pids 1 % 4 / (4 / 4) * 8192)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]; rfl))]
  -- stmt 7: v_offset = same as k_offset
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat 32768)) (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hkv") (Op.constNat 8192))) _
        = some (Tile.scalar (s.pids 1 / 4 * 32768 + s.pids 1 % 4 / (4 / 4) * 8192)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]; rfl))]
  -- stmt 8: o_offset = off_z*32768 + off_h*8192  (= q_offset)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat 32768)) (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h") (Op.constNat 8192))) _
        = some (Tile.scalar (s.pids 1 / 4 * 32768 + s.pids 1 % 4 * 8192)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]; rfl))]
  -- stmt 9: Q_block_ptr = makeBlockPtrDynOffsets Q q_offset [start_m*64, 0]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_makeBlockPtr_rowcol_eval Q (Op.ref TileDType.nat [] "q_offset") [128, 64] [64, 64] [64, 1]
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_m") (Op.constNat 64)) _
      (s.pids 1 / 4 * 32768 + s.pids 1 % 4 * 8192) (s.pids 0 * 64)
      (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same])
      (by rw [evalOp_mul]; simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_ne_name,
            ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_same, BlockState.setReg_pids,
            Option.bind_eq_bind, Option.bind_some]; rfl)))]
  -- stmt 10: V_block_ptr = makeBlockPtrDyn V v_offset [0,0]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_makeBlockPtrDyn_eval V (Op.ref TileDType.nat [] "v_offset") [128, 64] [64, 64] [64, 1] [0, 0] _
      (s.pids 1 / 4 * 32768 + s.pids 1 % 4 / (4 / 4) * 8192)
      (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  -- stmt 11: K_block_ptr = makeBlockPtrDyn K k_offset [0,0]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_makeBlockPtrDyn_eval K (Op.ref TileDType.nat [] "k_offset") [64, 128] [64, 64] [1, 64] [0, 0] _
      (s.pids 1 / 4 * 32768 + s.pids 1 % 4 / (4 / 4) * 8192)
      (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  -- stmt 12: O_block_ptr = makeBlockPtrDynOffsets Out o_offset [start_m*64, 0]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_makeBlockPtr_rowcol_eval Out (Op.ref TileDType.nat [] "o_offset") [128, 64] [64, 64] [64, 1]
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_m") (Op.constNat 64)) _
      (s.pids 1 / 4 * 32768 + s.pids 1 % 4 * 8192) (s.pids 0 * 64)
      (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same])
      (by rw [evalOp_mul]; simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_ne_name,
            ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_same, BlockState.setReg_pids,
            Option.bind_eq_bind, Option.bind_some]; rfl)))]
  -- stmt 13: offs_m = start_m*64 + arange 64
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.scalarL (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_m") (Op.constNat 64)) (Op.arange 64)) _
        = some (Tile.vec (fun r : Fin 64 => s.pids 0 * 64 + r.val)) from by
      rw [evalOp_add, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, evalOp_arange, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 14: m_ptrs = M + (off_hz*128 + offs_m)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase M) (Op.add NumericDType.nat Broadcast.scalarL (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat 128)) (Op.ref TileDType.nat [64] "offs_m"))) _
        = some (Tile.ptrAdd Broadcast.scalarL (Tile.scalar (M.cast, (0 : Nat)))
            (Tile.bop NumericDType.nat.add Broadcast.scalarL (Tile.scalar (s.pids 1 * 128))
              (Tile.vec (fun r : Fin 64 => s.pids 0 * 64 + r.val)))) from by
      rw [aft3_evalOp_ptrAdd, evalOp_add, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, aft3_evalOp_ptrBase, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]; rfl))]
  -- stmt 15: l_ptrs = L + (off_hz*128 + offs_m)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase L) (Op.add NumericDType.nat Broadcast.scalarL (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat 128)) (Op.ref TileDType.nat [64] "offs_m"))) _
        = some (Tile.ptrAdd Broadcast.scalarL (Tile.scalar (L.cast, (0 : Nat)))
            (Tile.bop NumericDType.nat.add Broadcast.scalarL (Tile.scalar (s.pids 1 * 128))
              (Tile.vec (fun r : Fin 64 => s.pids 0 * 64 + r.val)))) from by
      rw [aft3_evalOp_ptrAdd, evalOp_add, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, aft3_evalOp_ptrBase, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]; rfl))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_pids]
  · funext rg o; simp only [BlockState.setReg_mem]
  · funext rg o; simp only [BlockState.setReg_undef]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, Nat.reduceDiv, Nat.div_one]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, Nat.reduceDiv, Nat.div_one]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]

/-- The 3 `INIT` then-branch statements step `m_i = ⊥`, `l_i = 1`, `acc = 0`. -/
theorem aft3_init_steps (s : BlockState) :
    stepStmts [Stmt.assign TileDType.real [64] "m_i" ((Op.add NumericDType.real Broadcast.scalarR (Op.full [64] (Op.const 0)) Op.negInf)),
        Stmt.assign TileDType.real [64] "l_i" ((Op.add NumericDType.real Broadcast.scalarR (Op.full [64] (Op.const 0)) (Op.const 1.0))),
        Stmt.assign TileDType.real [64, 64] "acc" (Op.full [64, 64] (Op.const 0))] s
      = some (((s.setReg "m_i" .real [64] ⟨fun _ : TileIndex [64] => (⊥ : WithBot ℝ)⟩).setReg
          "l_i" .real [64] ⟨fun _ : TileIndex [64] => some (1 : ℝ)⟩).setReg
          "acc" .real [64, 64] ⟨fun _ : TileIndex [64, 64] => some (0 : ℝ)⟩) := by
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.real Broadcast.scalarR (Op.full [64] (Op.const 0)) Op.negInf) s
        = some (⟨fun _ : TileIndex [64] => (⊥ : WithBot ℝ)⟩ : Tile .real [64]) from by
      rw [evalOp_add]
      simp only [evalOp_full, evalOp_const, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.real Broadcast.scalarR (Op.full [64] (Op.const 0)) (Op.const 1.0)) _
        = some (⟨fun _ : TileIndex [64] => some (1 : ℝ)⟩ : Tile .real [64]) from by
      rw [evalOp_add]
      simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map]
      refine congrArg some ?_; norm_num))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [64, 64] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [64, 64] => some (0 : ℝ)⟩ : Tile .real [64, 64]) from by
      simp [evalOp_full, evalOp_const]))]
  rw [stepStmts.nil]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **PreLoop execution.** The 20 deterministic preLoop statements step a clean
state (`undef = 0`) to the loop-entry state, establishing `attnInvariant … 0`
for any case `keep` predicate: `m_i = ⊥`, `l_i = 1`, `acc = 0` (the constexpr
`INIT` then-branch), `q = qTile3`, `qk_scale`, the index vectors, the four block
pointers (`K`/`V` at offsets `0`), and `start_m`/`off_hz`. -/
theorem aft3PreLoop_eval
    (Q K V M Out L : RegionName) (s : BlockState)
    (keep : Fin 64 → Fin 128 → Prop) [∀ i j, Decidable (keep i j)]
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts (aft3PreLoop Q K V M Out L) s = some s'
      ∧ attnInvariant Q K V M Out L s keep 0 s' := by
  obtain ⟨s11, h11, hpids, hmem, huf, hstart, hoffhz, hQp, hKp, hVp, hOp, hMptr, hLptr⟩ :=
    aft3PreLoopScalars_eval Q K V M Out L s
  -- split body into prefix ++ [INIT, qk1, qk2, qload]
  rw [show aft3PreLoop Q K V M Out L
      = aft3PreLoopScalars Q K V M Out L ++ (aft3PreLoop Q K V M Out L).drop 16 from by
    rw [aft3PreLoopScalars]; rw [List.take_append_drop]]
  rw [stepStmts.append_some h11]
  -- drop 16 = [INIT, qk_scale1, qk_scale2, q-ifThenElse]
  rw [show (aft3PreLoop Q K V M Out L).drop 16
      = [ Stmt.ifThenElse (Op.ne ComparableDType.nat Broadcast.nil (Op.constNat 1) (Op.constNat 0)) [Stmt.assign TileDType.real [64] "m_i" ((Op.add NumericDType.real Broadcast.scalarR (Op.full [64] (Op.const 0)) Op.negInf)), Stmt.assign TileDType.real [64] "l_i" ((Op.add NumericDType.real Broadcast.scalarR (Op.full [64] (Op.const 0)) (Op.const 1.0))), Stmt.assign TileDType.real [64, 64] "acc" (Op.full [64, 64] (Op.const 0))] [Stmt.assign TileDType.real [64] "m_i" (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [64] "m_ptrs")) MaskOpt.none), Stmt.assign TileDType.real [64] "l_i" (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [64] "l_ptrs")) MaskOpt.none), Stmt.assign TileDType.real [64, 64] "acc" (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "O_block_ptr") []) MaskOpt.none)],
          Stmt.assign TileDType.real [] "qk_scale" (Op.mul NumericDType.real Broadcast.nil (Op.const (1 / 8)) (Op.const 1.0)),
          Stmt.assign TileDType.real [] "qk_scale" (Op.mul NumericDType.real Broadcast.nil (Op.ref TileDType.real [] "qk_scale") (Op.const 1.4426950408889634)),
          Stmt.ifThenElse (Op.ne ComparableDType.nat Broadcast.nil (Op.constNat 1) (Op.constNat 0)) [Stmt.assign TileDType.real [64, 64] "q" (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "Q_block_ptr") []) MaskOpt.none)] [Stmt.assign TileDType.real [64, 64] "q" (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "Q_block_ptr") [0, 1]) MaskOpt.none)] ] from by rfl]
  -- INIT ifThenElse (constexpr true)
  rw [stepStmts.cons_some
    (show stepStmt _ s11 = some _ from by
      rw [aft3_ifThenElse_true (aft3_ne_one_zero_true s11), aft3_init_steps s11])]
  -- qk_scale = (1/8)*1.0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul NumericDType.real Broadcast.nil (Op.const (1 / 8)) (Op.const 1.0)) _
        = some (Tile.scalar (some ((1 / 8 : ℝ) * 1.0))) from by
      rw [evalOp_mul]
      simp only [evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (congrArg Tile.scalar ?_)
      simp only [NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]))]
  -- qk_scale = qk_scale * 1.4426950408889634
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul NumericDType.real Broadcast.nil (Op.ref TileDType.real [] "qk_scale") (Op.const 1.4426950408889634)) _
        = some (Tile.scalar (some ((1 / 8 : ℝ) * 1.4426950408889634))) from by
      rw [evalOp_mul]
      simp only [evalOp_ref, evalOp_const, BlockState.setReg_same, Option.bind_eq_bind,
        Option.bind_some]
      refine congrArg some (congrArg Tile.scalar ?_)
      simp only [NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]
      refine congrArg some ?_; norm_num))]
  -- q ifThenElse (constexpr true): q = load Q_block_ptr
  rw [stepStmts.cons_some
    (show stepStmt _ _ = some _ from by
      rw [aft3_ifThenElse_true (aft3_ne_one_zero_true _)]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (aft3_load_v_eval Q (s.pids 1 / 4 * 32768 + s.pids 1 % 4 * 8192) 128 64 64 64 64 1 (s.pids 0 * 64)
          (Op.ref TileDType.blockPtr [64, 64] "Q_block_ptr") _
          (by rw [evalOp_ref]
              simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
                BlockState.setReg_same]
              exact hQp)))]
      rw [stepStmts.nil])]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  -- now establish attnInvariant ... 0
  simp only [attnInvariant]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- pids
    simp only [BlockState.setReg_pids]; exact hpids
  · -- 0 % 64 = 0
    trivial
  · -- 0 ≤ 128
    norm_num
  · -- m_i = aft3RunningMax ... 0 = ⊥
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    refine congrArg some ?_
    refine Tile.ext (fun r => ?_)
    simp only [aft3RunningMax_zero]
  · -- l_i = aft3StateBot1 ... 0 .2.1 = 1
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    refine congrArg some ?_
    refine Tile.ext (fun r => ?_)
    have hz : aft3StateBot1 (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3 keep 0 r.1 ⟨0, by norm_num⟩
          = (⊥, 1, 0) := by
      unfold aft3StateBot1 aft3KeysUpto
      rw [show (List.finRange 128).filterMap
            (fun j : Fin 128 => if j.val < 0 ∧ keep r.1 j
              then some (keyScale3 j * Finset.univ.sum (fun e : Fin 64 =>
                    qTile3 s Q (r.1, e, PUnit.unit) * kTile3 s K (j, e, PUnit.unit)), vTile3 s V (j, ⟨0, by norm_num⟩, PUnit.unit))
              else none) = [] from by
        apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
      rfl
    simp only [hz]; rfl
  · -- acc = aft3StateBot1 ... 0 .2.2 = 0
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    refine congrArg some ?_
    refine Tile.ext (fun idx => ?_)
    have hz : aft3StateBot1 (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3 keep 0 idx.1 idx.2.1
          = (⊥, 1, 0) := by
      unfold aft3StateBot1 aft3KeysUpto
      rw [show (List.finRange 128).filterMap
            (fun j : Fin 128 => if j.val < 0 ∧ keep idx.1 j
              then some (keyScale3 j * Finset.univ.sum (fun e : Fin 64 =>
                    qTile3 s Q (idx.1, e, PUnit.unit) * kTile3 s K (j, e, PUnit.unit)), vTile3 s V (j, idx.2.1, PUnit.unit))
              else none) = [] from by
        apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
      rfl
    simp only [hz]; rfl
  · -- q = qTile3
    simp only [BlockState.setReg_same]
    refine congrArg some ?_
    refine Tile.ext (fun idx => ?_)
    simp only [qTile3, baseOffset3, BlockState.setReg_readMem]
    refine congrArg some ?_
    rw [show s11.readMem Q (s.pids 1 / 4 * 32768 + s.pids 1 % 4 * 8192 + (s.pids 0 * 64 + idx.1.val) * 64 + idx.2.1.val * 1)
          = s.readMem Q (s.pids 1 / 4 * 32768 + s.pids 1 % 4 * 8192 + (s.pids 0 * 64 + idx.1.val) * 64 + idx.2.1.val) from by
      unfold BlockState.readMem; rw [hmem]; ring_nf]
  · -- qk_scale
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · -- start_m
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    rw [hstart]
  · -- off_hz
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    rw [hoffhz]
  · -- K_block_ptr (offsets [0, 0])
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    rw [hKp]
  · -- V_block_ptr (offsets [0, 0])
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    rw [hVp]
  · -- m_ptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    rw [hMptr]
  · -- l_ptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    rw [hLptr]
  · -- O_block_ptr
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    rw [hOp]
  · -- undef = 0
    intro rg o
    simp only [BlockState.setReg_undef]
    rw [huf]; exact hundef rg o
  · -- mem = s.mem
    funext rg o
    simp only [BlockState.setReg_mem]
    rw [show s11.mem rg o = s.mem rg o from by rw [hmem]]

/-! ### loopBody execution (deliverable 4, exec side)

Steps the 19 lowered `aft3LoopBody` statements (case 1: `SLIDING_WINDOW=1`,
`COMPLEMENT=0`) through the banked op-eval recipes, threading the sliding-window
`ifThen`/`ifThenElse` constexpr branches into the symbolic masked `qk`/`p` tiles.
-/

/-- The case-1 sliding-window mask cell for `(i, j)` at `start_m = SM`,
`start_n = SN`: the truncated-nat distance `((i - j) + SM·64 - SN) + 0` lies in
`[0, 64)`. -/
noncomputable def aft3MaskCell1 (SM SN : Nat) (idx : TileIndex [64, 64]) : Bool :=
  ComparableDType.nat.ge (((idx.1.val - idx.2.1.val) + SM * 64 - SN) + 0) 0
    && ComparableDType.nat.lt (((idx.1.val - idx.2.1.val) + SM * 64 - SN) + 0) 64

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Sliding-window block (case 1).** The `ifThen` body `[dist, ifThenElse mask, qk =
where]` steps from a state with `start_m=SM`/`start_n=SN`/`qk=qktile` to one where
`dist`/`mask` are set and `qk` is the masked tile (`⊥` on out-of-window lanes). -/
theorem aft3_window1_steps (s : BlockState) (SM SN : Nat) (qktile : Tile .real [64, 64])
    (hsm : s.regs .nat [] "start_m" = some (Tile.scalar SM))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hqk : s.regs .real [64, 64] "qk" = some qktile) :
    stepStmts [ Stmt.assign TileDType.nat [64, 64] "dist" (Op.add NumericDType.nat Broadcast.scalarR (Op.sub NumericDType.nat Broadcast.scalarR (Op.add NumericDType.nat Broadcast.scalarR (Op.sub NumericDType.nat Broadcast.nil.consL.consR (Op.expandDim ⟨1, by decide⟩ (Op.arange 64)) (Op.expandDim ⟨0, by decide⟩ (Op.arange 64))) (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_m") (Op.constNat 64))) (Op.ref TileDType.nat [] "start_n")) (Op.constNat 0)),
        Stmt.ifThenElse (Op.ne ComparableDType.nat Broadcast.nil (Op.constNat 0) (Op.constNat 0)) [Stmt.assign TileDType.bool [64, 64] "mask" (Op.ge ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [64, 64] "dist") (Op.constNat 64))] [Stmt.assign TileDType.bool [64, 64] "mask" (Op.boolAnd Broadcast.nil.consSame.consSame (Op.ge ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [64, 64] "dist") (Op.constNat 0)) (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [64, 64] "dist") (Op.constNat 64)))],
        Stmt.assign TileDType.real [64, 64] "qk" ((Op.ref TileDType.bool [64, 64] "mask").where (Op.ref TileDType.real [64, 64] "qk") (Op.negInf.broadcast [64, 64]))] s
      = some ((((s.setReg "dist" .nat [64, 64] ⟨fun idx : TileIndex [64, 64] => ((idx.1.val - idx.2.1.val) + SM * 64 - SN) + 0⟩).setReg
          "mask" .bool [64, 64] ⟨fun idx : TileIndex [64, 64] => aft3MaskCell1 SM SN idx⟩).setReg
          "qk" .real [64, 64] ⟨fun idx : TileIndex [64, 64] =>
            if aft3MaskCell1 SM SN idx then qktile.data idx else (⊥ : WithBot ℝ)⟩)) := by
  set disttile : Tile .nat [64, 64] :=
    ⟨fun idx : TileIndex [64, 64] => ((idx.1.val - idx.2.1.val) + SM * 64 - SN) + 0⟩ with hdist
  set masktile : Tile .bool [64, 64] :=
    ⟨fun idx : TileIndex [64, 64] =>
      ComparableDType.nat.ge (disttile.data idx) 0 && ComparableDType.nat.lt (disttile.data idx) 64⟩ with hmtile
  have hmaskeq : masktile = ⟨fun idx : TileIndex [64, 64] => aft3MaskCell1 SM SN idx⟩ := by
    refine Tile.ext (fun idx => ?_); simp only [hmtile, aft3MaskCell1, hdist]
  -- stmt: dist
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (aft3_dist_eval s SM SN hsm hsn))]
  -- stmt: mask ifThenElse (COMPLEMENT false → else branch = case1 mask)
  rw [stepStmts.cons_some
    (show stepStmt _ (s.setReg "dist" .nat [64, 64] disttile) = some _ from by
      rw [aft3_ifThenElse_false (aft3_ne_zero_zero_false _)]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (aft3_mask_case1_eval (s.setReg "dist" .nat [64, 64] disttile) disttile
          (by rw [BlockState.setReg_same])))]
      rw [stepStmts.nil])]
  -- stmt: qk = where mask qk -inf
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_where_eval _ masktile qktile
      (by rw [BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same]; exact hqk)))]
  rw [stepStmts.nil]
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **p-mask block (case 1).** The `ifThen` body `[p = where mask p 0]` zeros the
out-of-window lanes of `p`. -/
theorem aft3_pmask1_steps (s : BlockState) (masktile : Tile .bool [64, 64])
    (ptile : Tile .real [64, 64])
    (hmask : s.regs .bool [64, 64] "mask" = some masktile)
    (hp : s.regs .real [64, 64] "p" = some ptile) :
    stepStmts [ Stmt.assign TileDType.real [64, 64] "p" ((Op.ref TileDType.bool [64, 64] "mask").where (Op.ref TileDType.real [64, 64] "p") ((Op.const 0.0).broadcast [64, 64]))] s
      = some (s.setReg "p" .real [64, 64] ⟨fun idx : TileIndex [64, 64] =>
          if masktile.data idx then ptile.data idx else (some (0.0 : ℝ) : WithBot ℝ)⟩) := by
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (aft3_p_mask_eval s masktile ptile hmask hp))]
  rw [stepStmts.nil]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Loop-body execution chain (case 1).** The 19 lowered `aft3LoopBody`
statements step the iteration-entry state `sin` (with `start_n = SN`, the
invariant's register readbacks `m_i`/`l_i`/`acc`/`q`/`qk_scale`/`start_m`, and
the `K`/`V` block pointers at `[0, kcol]`/`[vrow, 0]`) to a final state `sF`,
exposing the symbolic `m_i`/`l_i`/`acc` register values (the masked online-softmax
block recurrence over `aft3MaskCell1`) plus the advanced block pointers. The
sliding-window `ifThen`/`ifThenElse` constexpr branches fold into the masked
`qk`/`p` tiles via `aft3_window1_steps`/`aft3_pmask1_steps`. -/
theorem aft3LoopBody_steps (sin : BlockState) (SM SN : Nat)
    (Kreg Vreg : RegionName) (kbase vbase kcol vrow : Nat)
    (qtile : Tile .real [64, 64]) (mtile ltile : Tile .real [64]) (acctile : Tile .real [64, 64])
    (ktile vtile : Tile .real [64, 64]) (sc : ℝ)
    (hsm : sin.regs .nat [] "start_m" = some (Tile.scalar SM))
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hq : sin.regs .real [64, 64] "q" = some qtile)
    (hqs : sin.regs .real [] "qk_scale" = some (Tile.scalar (some sc)))
    (hmi : sin.regs .real [64] "m_i" = some mtile)
    (hli : sin.regs .real [64] "l_i" = some ltile)
    (hacc : sin.regs .real [64, 64] "acc" = some acctile)
    (hKp : sin.regs .blockPtr [64, 64] "K_block_ptr" = some
      (⟨fun _ : TileIndex [64, 64] =>
        { region := Kreg, baseOffset := kbase, parentShape := [64, 128],
          blockShape := [64, 64], strides := [1, 64], offsets := [0, kcol] }⟩))
    (hVp : sin.regs .blockPtr [64, 64] "V_block_ptr" = some
      (⟨fun _ : TileIndex [64, 64] =>
        { region := Vreg, baseOffset := vbase, parentShape := [128, 64],
          blockShape := [64, 64], strides := [64, 1], offsets := [vrow, 0] }⟩))
    (hkload : ∀ idx : TileIndex [64, 64],
      ktile.data idx = some (sin.readMem Kreg (kbase + idx.1.val * 1 + (kcol + idx.2.1.val) * 64)))
    (hvload : ∀ idx : TileIndex [64, 64],
      vtile.data idx = some (sin.readMem Vreg (vbase + (vrow + idx.1.val) * 64 + idx.2.1.val * 1)))
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ sF, stepStmts aft3LoopBody sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = 0)
      ∧ sF.regs .real [64, 64] "q" = some qtile
      ∧ sF.regs .real [] "qk_scale" = some (Tile.scalar (some sc))
      ∧ sF.regs .nat [] "start_m" = some (Tile.scalar SM)
      ∧ sF.regs .blockPtr [64, 64] "K_block_ptr" = some
          (⟨fun _ : TileIndex [64, 64] =>
            { region := Kreg, baseOffset := kbase, parentShape := [64, 128],
              blockShape := [64, 64], strides := [1, 64], offsets := [0, kcol + 64] }⟩)
      ∧ sF.regs .blockPtr [64, 64] "V_block_ptr" = some
          (⟨fun _ : TileIndex [64, 64] =>
            { region := Vreg, baseOffset := vbase, parentShape := [128, 64],
              blockShape := [64, 64], strides := [64, 1], offsets := [vrow + 64, 0] }⟩)
      ∧ sF.regs .nat [] "off_hz" = sin.regs .nat [] "off_hz"
      ∧ sF.regs .ptr [64] "m_ptrs" = sin.regs .ptr [64] "m_ptrs"
      ∧ sF.regs .ptr [64] "l_ptrs" = sin.regs .ptr [64] "l_ptrs"
      ∧ sF.regs .blockPtr [64, 64] "O_block_ptr" = sin.regs .blockPtr [64, 64] "O_block_ptr"
      ∧ ∃ (qkT : Tile .real [64, 64]) (rmaxT mijT alphaT lijT : Tile .real [64])
            (pT pmT : Tile .real [64, 64]) (acc1T : Tile .real [64, 64]),
          (qkT = ⟨fun idx : TileIndex [64, 64] =>
            if aft3MaskCell1 SM SN idx
            then (Tile.bop NumericDType.real.mul Broadcast.scalarR
                    (Tile.bop NumericDType.real.add
                      (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                      (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
                    (Tile.scalar (some sc))).data idx
            else (⊥ : WithBot ℝ)⟩)
          ∧ Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [64, 64].length) qkT = some rmaxT
          ∧ mijT = Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT
          ∧ alphaT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT)
          ∧ pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mijT))
          ∧ pmT = ⟨fun idx : TileIndex [64, 64] =>
              if aft3MaskCell1 SM SN idx then pT.data idx else (some (0.0 : ℝ) : WithBot ℝ)⟩
          ∧ lijT = Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [64, 64].length) pmT
          ∧ acc1T = Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile (Tile.expandDim ⟨1, by simp⟩ alphaT)
          ∧ sF.regs .real [64] "m_i" = some mijT
          ∧ sF.regs .real [64] "l_i" = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
              (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile alphaT) lijT)
          ∧ sF.regs .real [64, 64] "acc" = some (Tile.bop NumericDType.real.add
              (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acc1T (Tile.dot [] pmT vtile)) := by
  set qkSeedT : Tile .real [64, 64] :=
    Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
      (Tile.scalar (some sc)) with hqkSeed
  set maskT : Tile .bool [64, 64] := ⟨fun idx : TileIndex [64, 64] => aft3MaskCell1 SM SN idx⟩ with hmaskT
  set qkT : Tile .real [64, 64] :=
    ⟨fun idx : TileIndex [64, 64] =>
      if maskT.data idx then qkSeedT.data idx else (⊥ : WithBot ℝ)⟩ with hqkT
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [64, 64].length) qkT = some t :=
    ⟨_, by unfold Tile.reduceMaxDrop; rw [dif_pos (show 0 < TileShape.axisDim [64, 64] (⟨1, by simp⟩ : Fin [64, 64].length) from by decide)]⟩
  set mijT : Tile .real [64] := Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT with hmij
  set alphaT : Tile .real [64] := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT) with halpha
  set pT : Tile .real [64, 64] := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mijT)) with hpT
  set pmT : Tile .real [64, 64] := ⟨fun idx : TileIndex [64, 64] => if maskT.data idx then pT.data idx else (some (0.0 : ℝ) : WithBot ℝ)⟩ with hpmT
  set lijT : Tile .real [64] := Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [64, 64].length) pmT with hlij
  set acc1T : Tile .real [64, 64] := Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile (Tile.expandDim ⟨1, by simp⟩ alphaT) with hacc1
  unfold aft3LoopBody
  -- L1: k = load K_block_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "K_block_ptr") []) MaskOpt.none) sin
        = some ktile from by
      rw [aft3_load_k_eval Kreg kbase 64 128 64 64 1 64 kcol (Op.ref TileDType.blockPtr [64, 64] "K_block_ptr") sin
        (by rw [evalOp_ref]; exact hKp)]
      refine congrArg some ?_; refine Tile.ext (fun idx => ?_); rw [hkload idx]))]
  -- L2: qk = zeros
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (aft3_qkzeros_eval _ 64 64))]
  -- L3: qk += dot q k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_qk_dot_eval _ 64 64 64 (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) qtile ktile
      (by rw [BlockState.setReg_same] <;> try rfl)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same]; exact hq)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)))]
  -- L4: qk *= qk_scale  (qk_scale read from register)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_qk_scale_ref_eval _ 64 64 sc
      (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
      (by rw [BlockState.setReg_same] <;> try rfl)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hqs)))]
  -- L5: window ifThen (SLIDING_WINDOW true) → dist/mask/qk masked = qkT
  rw [stepStmts.cons_some
    (show stepStmt _ _ = some _ from
      (aft3_ifThen_true (aft3_ne_one_zero_true _)).trans
        (aft3_window1_steps _ SM SN qkSeedT
          (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hsm)
          (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hsn)
          (by rw [BlockState.setReg_same])))]
  -- L6: m_ij = maximum(m_i, max(qk,1))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_mij_eval _ mtile qkT rmaxT
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hmi)
      (by rw [BlockState.setReg_same] <;> try rfl)
      hrm))]
  -- L7: qk -= m_ij[:, None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_qk_sub_eval _ (by simp) qkT mijT
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)
      (by rw [BlockState.setReg_same])))]
  -- L8: p = exp2 qk
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_p_eval _ (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mijT))
      (by rw [BlockState.setReg_same])))]
  -- L9: pmask ifThen (SLIDING_WINDOW true) → p masked = pmT
  rw [stepStmts.cons_some
    (show stepStmt _ _ = some _ from
      (aft3_ifThen_true (aft3_ne_one_zero_true _)).trans
        (aft3_pmask1_steps _ maskT pT
          (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
                BlockState.setReg_same] <;> try rfl)
          (by rw [BlockState.setReg_same])))]
  -- L10: l_ij = sum p 1
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .real [64] "l_ij"
    (Op.reduceSum (⟨1, by simp⟩ : Fin [64, 64].length) Bool.false (Op.ref .real [64, 64] "p")) _ lijT
    (aft3_lij_eval _ pmT (by rw [BlockState.setReg_same])))]
  -- L11: tmp = m_i - m_ij
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_tmp_eval _ mtile mijT
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hmi)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)))]
  -- L12: alpha = exp2 tmp
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_alpha_eval _ (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT)
      (by rw [BlockState.setReg_same])))]
  -- L13: l_i = l_i * alpha + l_ij
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_li_eval _ ltile alphaT lijT
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hli)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)))]
  -- L14: acc = acc * alpha[:, None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_acc_rescale_eval _ (by simp) acctile alphaT
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hacc)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)))]
  -- L15: v = load V_block_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "V_block_ptr") []) MaskOpt.none) _
        = some vtile from by
      rw [aft3_load_v_eval Vreg vbase 128 64 64 64 64 1 vrow (Op.ref TileDType.blockPtr [64, 64] "V_block_ptr") _
        (by rw [evalOp_ref]
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
            exact hVp)]
      refine congrArg some ?_; refine Tile.ext (fun idx => ?_)
      simp only [BlockState.setReg_readMem]; rw [hvload idx]))]
  -- L16: acc += dot p v
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_acc_eval _ 64 64 64 acc1T pmT vtile
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)
      (by rw [BlockState.setReg_same])))]
  -- L17: m_i = m_ij
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_mi_carry_eval _ mijT
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)))]
  -- L18: V_block_ptr = advance [64, 0]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_advance_v_eval _ Vreg vbase 128 64 64 64 64 1 vrow 64 "V_block_ptr"
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hVp)))]
  -- L19: K_block_ptr = advance [0, 64]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_advance_k_eval _ Kreg kbase 64 128 64 64 1 64 kcol 64 "K_block_ptr"
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hKp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, qkT, rmaxT, mijT, alphaT, lijT, pT, pmT, acc1T,
    rfl, hrm, rfl, rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_pids]
  · funext rg o; simp only [BlockState.setReg_mem]
  · intro rg o; simp only [BlockState.setReg_undef]; exact hundef rg o
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]; exact hq
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]; exact hqs
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]; exact hsm
  · simp only [BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]

/-! ## Case 3 (no sliding window): genuine unmasked closed form

Case 3 (`_SLIDING_WINDOW = 0`) lowers the two sliding-window `ifThen` guards to the
constexpr-false condition `Op.ne (constNat 0) (constNat 0)`, so the `dist`/`mask`/
`where` block and the `p`-mask are dead and skipped. The streaming loop then runs
the *unmasked* online-softmax recurrence over every key — i.e. the `noWindowKeep`
predicate — which equals the plain base-2 per-key-scale softmax
`attentionFwdTriton3Case3OutSpec`.

(Cases 1/2 are NOT closeable against `slidingWindowKeep`/
`complementSlidingWindowKeep`: the kernel models the Triton `dist` with **natural**
subtraction `Op.sub NumericDType.nat`, which truncates negative distances to `0`,
whereas the math specs use signed ℤ arithmetic. The two masks differ on a majority
of cells, so those genuine-spec theorems are false — a modeling boundary, not a
provability gap.) -/

/-- Case-3 lowered loop body: identical to `aft3LoopBody` but with the two
sliding-window `ifThen` conditions constexpr-false (`Op.ne 0 0`) and the
dead mask comparisons using `size = 0`. -/

def aft3LoopBody3 : List Stmt :=
  [ Stmt.assign TileDType.real [64, 64] "k" (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "K_block_ptr") []) MaskOpt.none),
    Stmt.assign TileDType.real [64, 64] "qk" (Op.full [64, 64] (Op.const 0)),
    Stmt.assign TileDType.real [64, 64] "qk" ((Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref TileDType.real [64, 64] "qk") ((Op.dot (batch := []) (Op.ref TileDType.real [64, 64] "q") (Op.ref TileDType.real [64, 64] "k"))))),
    Stmt.assign TileDType.real [64, 64] "qk" ((Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [64, 64] "qk") (Op.ref TileDType.real [] "qk_scale"))),
    Stmt.ifThen (Op.ne ComparableDType.nat Broadcast.nil (Op.constNat 0) (Op.constNat 0)) [Stmt.assign TileDType.nat [64, 64] "dist" (Op.add NumericDType.nat Broadcast.scalarR (Op.sub NumericDType.nat Broadcast.scalarR (Op.add NumericDType.nat Broadcast.scalarR (Op.sub NumericDType.nat Broadcast.nil.consL.consR (Op.expandDim ⟨1, by decide⟩ (Op.arange 64)) (Op.expandDim ⟨0, by decide⟩ (Op.arange 64))) (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_m") (Op.constNat 64))) (Op.ref TileDType.nat [] "start_n")) (Op.constNat 0)), Stmt.ifThenElse (Op.ne ComparableDType.nat Broadcast.nil (Op.constNat 0) (Op.constNat 0)) [Stmt.assign TileDType.bool [64, 64] "mask" (Op.ge ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [64, 64] "dist") (Op.constNat 0))] [Stmt.assign TileDType.bool [64, 64] "mask" (Op.boolAnd Broadcast.nil.consSame.consSame (Op.ge ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [64, 64] "dist") (Op.constNat 0)) (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [64, 64] "dist") (Op.constNat 0)))], Stmt.assign TileDType.real [64, 64] "qk" ((Op.ref TileDType.bool [64, 64] "mask").where (Op.ref TileDType.real [64, 64] "qk") (Op.negInf.broadcast [64, 64]))],
    Stmt.assign TileDType.real [64] "m_ij" (((Op.gt ComparableDType.real Broadcast.nil.consSame (Op.ref TileDType.real [64] "m_i") (Op.reduceMax ⟨1, by decide⟩ Bool.false (Op.ref TileDType.real [64, 64] "qk"))).where (Op.ref TileDType.real [64] "m_i") (Op.reduceMax ⟨1, by decide⟩ Bool.false (Op.ref TileDType.real [64, 64] "qk")))),
    Stmt.assign TileDType.real [64, 64] "qk" (Op.sub NumericDType.real Broadcast.nil.consR.consSame (Op.ref TileDType.real [64, 64] "qk") (Op.expandDim ⟨1, by decide⟩ (Op.ref TileDType.real [64] "m_ij"))),
    Stmt.assign TileDType.real [64, 64] "p" (Op.ref TileDType.real [64, 64] "qk").exp2,
    Stmt.ifThen (Op.ne ComparableDType.nat Broadcast.nil (Op.constNat 0) (Op.constNat 0)) [Stmt.assign TileDType.real [64, 64] "p" ((Op.ref TileDType.bool [64, 64] "mask").where (Op.ref TileDType.real [64, 64] "p") ((Op.const 0.0).broadcast [64, 64]))],
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


set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Body split (case 3).** The lowered algorithm body of the case-3 surface is
`aft3PreLoop ++ forRange aft3LoopBody3 :: aft3PostLoop`. -/
theorem aft3_body_split_case3 (Q K V M Out L : RegionName) :
    (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 0 1 1 64 64 64 1 1 0 0).toAlgKernel.body
      = aft3PreLoop Q K V M Out L
        ++ Stmt.forRange "start_n" 0 128 64 aft3LoopBody3 :: aft3PostLoop M Out L := by
  rfl

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Loop-body execution chain (case 3, no window).** The 19 lowered
`aft3LoopBody3` statements step the iteration-entry state `sin` to a final state
`sF`, exposing the symbolic `m_i`/`l_i`/`acc` registers — the *unmasked* online-
softmax block recurrence (every key kept). The two sliding-window `ifThen` guards
are constexpr-false (`aft3_ifThen_false`) so the `dist`/`mask`/`where` block and
the `p`-mask are skipped: `qk` stays the raw `q·k·scale` tile and `p` stays
`exp2(qk - m_ij)`. -/
theorem aft3LoopBody3_steps (sin : BlockState) (SM SN ohz : Nat)
    (Kreg Vreg : RegionName) (kbase vbase kcol vrow : Nat)
    (qtile : Tile .real [64, 64]) (mtile ltile : Tile .real [64]) (acctile : Tile .real [64, 64])
    (ktile vtile : Tile .real [64, 64]) (sc : ℝ)
    (mptrs lptrs : Tile .ptr [64]) (oblk : Tile .blockPtr [64, 64])
    (hmptrs : sin.regs .ptr [64] "m_ptrs" = some mptrs)
    (hlptrs : sin.regs .ptr [64] "l_ptrs" = some lptrs)
    (hoblk : sin.regs .blockPtr [64, 64] "O_block_ptr" = some oblk)
    (hsm : sin.regs .nat [] "start_m" = some (Tile.scalar SM))
    (hoff : sin.regs .nat [] "off_hz" = some (Tile.scalar ohz))
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hq : sin.regs .real [64, 64] "q" = some qtile)
    (hqs : sin.regs .real [] "qk_scale" = some (Tile.scalar (some sc)))
    (hmi : sin.regs .real [64] "m_i" = some mtile)
    (hli : sin.regs .real [64] "l_i" = some ltile)
    (hacc : sin.regs .real [64, 64] "acc" = some acctile)
    (hKp : sin.regs .blockPtr [64, 64] "K_block_ptr" = some
      (⟨fun _ : TileIndex [64, 64] =>
        { region := Kreg, baseOffset := kbase, parentShape := [64, 128],
          blockShape := [64, 64], strides := [1, 64], offsets := [0, kcol] }⟩))
    (hVp : sin.regs .blockPtr [64, 64] "V_block_ptr" = some
      (⟨fun _ : TileIndex [64, 64] =>
        { region := Vreg, baseOffset := vbase, parentShape := [128, 64],
          blockShape := [64, 64], strides := [64, 1], offsets := [vrow, 0] }⟩))
    (hkload : ∀ idx : TileIndex [64, 64],
      ktile.data idx = some (sin.readMem Kreg (kbase + idx.1.val * 1 + (kcol + idx.2.1.val) * 64)))
    (hvload : ∀ idx : TileIndex [64, 64],
      vtile.data idx = some (sin.readMem Vreg (vbase + (vrow + idx.1.val) * 64 + idx.2.1.val * 1)))
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ sF, stepStmts aft3LoopBody3 sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = 0)
      ∧ sF.regs .real [64, 64] "q" = some qtile
      ∧ sF.regs .real [] "qk_scale" = some (Tile.scalar (some sc))
      ∧ sF.regs .nat [] "start_m" = some (Tile.scalar SM)
      ∧ sF.regs .nat [] "off_hz" = some (Tile.scalar ohz)
      ∧ sF.regs .blockPtr [64, 64] "K_block_ptr" = some
          (⟨fun _ : TileIndex [64, 64] =>
            { region := Kreg, baseOffset := kbase, parentShape := [64, 128],
              blockShape := [64, 64], strides := [1, 64], offsets := [0, kcol + 64] }⟩)
      ∧ sF.regs .blockPtr [64, 64] "V_block_ptr" = some
          (⟨fun _ : TileIndex [64, 64] =>
            { region := Vreg, baseOffset := vbase, parentShape := [128, 64],
              blockShape := [64, 64], strides := [64, 1], offsets := [vrow + 64, 0] }⟩)
      ∧ sF.regs .ptr [64] "m_ptrs" = some mptrs
      ∧ sF.regs .ptr [64] "l_ptrs" = some lptrs
      ∧ sF.regs .blockPtr [64, 64] "O_block_ptr" = some oblk
      ∧ ∃ (qkT : Tile .real [64, 64]) (rmaxT mijT alphaT lijT : Tile .real [64])
            (pT acc1T : Tile .real [64, 64]),
          (qkT = Tile.bop NumericDType.real.mul Broadcast.scalarR
                    (Tile.bop NumericDType.real.add
                      (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                      (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
                    (Tile.scalar (some sc)))
          ∧ Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [64, 64].length) qkT = some rmaxT
          ∧ mijT = Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT
          ∧ alphaT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT)
          ∧ pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mijT))
          ∧ lijT = Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [64, 64].length) pT
          ∧ acc1T = Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile (Tile.expandDim ⟨1, by simp⟩ alphaT)
          ∧ sF.regs .real [64] "m_i" = some mijT
          ∧ sF.regs .real [64] "l_i" = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
              (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile alphaT) lijT)
          ∧ sF.regs .real [64, 64] "acc" = some (Tile.bop NumericDType.real.add
              (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acc1T (Tile.dot [] pT vtile)) := by
  set qkT : Tile .real [64, 64] :=
    Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
      (Tile.scalar (some sc)) with hqkT
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [64, 64].length) qkT = some t :=
    ⟨_, by unfold Tile.reduceMaxDrop; rw [dif_pos (show 0 < TileShape.axisDim [64, 64] (⟨1, by simp⟩ : Fin [64, 64].length) from by decide)]⟩
  set mijT : Tile .real [64] := Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT with hmij
  set alphaT : Tile .real [64] := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT) with halpha
  set pT : Tile .real [64, 64] := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mijT)) with hpT
  set lijT : Tile .real [64] := Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [64, 64].length) pT with hlij
  set acc1T : Tile .real [64, 64] := Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile (Tile.expandDim ⟨1, by simp⟩ alphaT) with hacc1
  unfold aft3LoopBody3
  -- L1: k = load K_block_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "K_block_ptr") []) MaskOpt.none) sin
        = some ktile from by
      rw [aft3_load_k_eval Kreg kbase 64 128 64 64 1 64 kcol (Op.ref TileDType.blockPtr [64, 64] "K_block_ptr") sin
        (by rw [evalOp_ref]; exact hKp)]
      refine congrArg some ?_; refine Tile.ext (fun idx => ?_); rw [hkload idx]))]
  -- L2: qk = zeros
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (aft3_qkzeros_eval _ 64 64))]
  -- L3: qk += dot q k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_qk_dot_eval _ 64 64 64 (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) qtile ktile
      (by rw [BlockState.setReg_same] <;> try rfl)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same]; exact hq)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)))]
  -- L4: qk *= qk_scale
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_qk_scale_ref_eval _ 64 64 sc
      (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
      (by rw [BlockState.setReg_same] <;> try rfl)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hqs)))]
  -- L5: window ifThen (SLIDING_WINDOW false) → skipped, qk stays = qkT
  rw [stepStmts.cons_some (aft3_ifThen_false (aft3_ne_zero_zero_false _))]
  -- L6: m_ij = maximum(m_i, max(qk,1))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_mij_eval _ mtile qkT rmaxT
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hmi)
      (by rw [BlockState.setReg_same] <;> try rfl)
      hrm))]
  -- L7: qk -= m_ij[:, None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_qk_sub_eval _ (by simp) qkT mijT
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)
      (by rw [BlockState.setReg_same])))]
  -- L8: p = exp2 qk
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_p_eval _ (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mijT))
      (by rw [BlockState.setReg_same])))]
  -- L9: pmask ifThen (SLIDING_WINDOW false) → skipped, p stays = pT
  rw [stepStmts.cons_some (aft3_ifThen_false (aft3_ne_zero_zero_false _))]
  -- L10: l_ij = sum p 1
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .real [64] "l_ij"
    (Op.reduceSum (⟨1, by simp⟩ : Fin [64, 64].length) Bool.false (Op.ref .real [64, 64] "p")) _ lijT
    (aft3_lij_eval _ pT (by rw [BlockState.setReg_same])))]
  -- L11: tmp = m_i - m_ij
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_tmp_eval _ mtile mijT
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hmi)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)))]
  -- L12: alpha = exp2 tmp
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_alpha_eval _ (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT)
      (by rw [BlockState.setReg_same])))]
  -- L13: l_i = l_i * alpha + l_ij
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_li_eval _ ltile alphaT lijT
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hli)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)))]
  -- L14: acc = acc * alpha[:, None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_acc_rescale_eval _ (by simp) acctile alphaT
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hacc)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)))]
  -- L15: v = load V_block_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "V_block_ptr") []) MaskOpt.none) _
        = some vtile from by
      rw [aft3_load_v_eval Vreg vbase 128 64 64 64 64 1 vrow (Op.ref TileDType.blockPtr [64, 64] "V_block_ptr") _
        (by rw [evalOp_ref]
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
            exact hVp)]
      refine congrArg some ?_; refine Tile.ext (fun idx => ?_)
      simp only [BlockState.setReg_readMem]; rw [hvload idx]))]
  -- L16: acc += dot p v
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_acc_eval _ 64 64 64 acc1T pT vtile
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)
      (by rw [BlockState.setReg_same])))]
  -- L17: m_i = m_ij
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_mi_carry_eval _ mijT
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same] <;> try rfl)))]
  -- L18: V_block_ptr = advance [64, 0]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_advance_v_eval _ Vreg vbase 128 64 64 64 64 1 vrow 64 "V_block_ptr"
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hVp)))]
  -- L19: K_block_ptr = advance [0, 64]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft3_advance_k_eval _ Kreg kbase 64 128 64 64 1 64 kcol 64 "K_block_ptr"
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]; exact hKp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, qkT, rmaxT, mijT, alphaT, lijT, pT, acc1T,
    rfl, hrm, rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_pids]
  · funext rg o; simp only [BlockState.setReg_mem]
  · intro rg o; simp only [BlockState.setReg_undef]; exact hundef rg o
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]; exact hq
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]; exact hqs
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]; exact hsm
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]; exact hoff
  · simp only [BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]; exact hmptrs
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]; exact hlptrs
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]; exact hoblk
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]

/-! ### attn_step bridges (deliverable 4): block sums / running max -/

/-- Map-and-sum of a `filterMap` over `finRange` is a masked `Finset.sum`. -/
theorem aft3_filterMap_finRange_sum {α : Type*} (n : Nat)
    (p : Fin n → Prop) [DecidablePred p] (g : Fin n → α) (h : α → ℝ) :
    (((List.finRange n).filterMap (fun j => if p j then some (g j) else none)).map h).sum
      = ∑ j : Fin n, if p j then h (g j) else 0 := by
  rw [List.map_filterMap]
  rw [show (fun j : Fin n => Option.map h (if p j then some (g j) else none))
        = (fun j : Fin n => if p j then some (h (g j)) else none) from by
    funext j; by_cases hj : p j <;> simp [hj]]
  rw [show (((List.finRange n).filterMap (fun j => if p j then some (h (g j)) else none))).sum
        = ((List.finRange n).map (fun j => if p j then h (g j) else 0)).sum from by
    induction (List.finRange n) with
    | nil => simp
    | cons a t ih => by_cases ha : p a <;> simp [ha, ih]]
  rw [← List.sum_ofFn]; congr 1; rw [List.ofFn_eq_map]

/-- The map-and-sum of `aft3Block` (case `keep`) equals a `Fin 64`-masked
`Finset.sum`, reindexing block `c`'s window onto lanes `jL` (global key
`c·64 + jL`). -/
theorem aft3Block_map_sum
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (c : Nat) (i : Fin 64) (d : Fin 64)
    (hwin : (c + 1) * 64 ≤ 128) (h : ℝ × ℝ → ℝ) :
    ((aft3Block qT kT vT keyScale keep c i d).map h).sum
      = ∑ jL : Fin 64,
          (if keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ then
            h (keyScale ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ *
                Finset.univ.sum (fun e : Fin 64 =>
                  qT (i, e, PUnit.unit) * kT (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit)),
                vT (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit))
           else 0) := by
  rw [aft3Block, aft3_filterMap_finRange_sum 128
    (fun j => c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ keep i j)
    (fun j => (keyScale j * Finset.univ.sum (fun e : Fin 64 =>
        qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)), vT (j, d, PUnit.unit))) h]
  rw [show (∑ j : Fin 128, if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ keep i j
            then h (keyScale j * Finset.univ.sum (fun e : Fin 64 =>
                  qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)), vT (j, d, PUnit.unit)) else 0)
        = ∑ j ∈ Finset.univ.filter (fun j : Fin 128 => c * 64 ≤ j.val ∧ j.val < (c + 1) * 64),
            (if keep i j then h (keyScale j * Finset.univ.sum (fun e : Fin 64 =>
                qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)), vT (j, d, PUnit.unit)) else 0) from by
    rw [Finset.sum_filter]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    by_cases hwj : c * 64 ≤ j.val ∧ j.val < (c + 1) * 64
    · by_cases hcj : keep i j
      · rw [if_pos ⟨hwj.1, hwj.2, hcj⟩, if_pos hwj, if_pos hcj]
      · rw [if_neg (fun hh => hcj hh.2.2), if_pos hwj, if_neg hcj]
    · rw [if_neg (fun hh => hwj ⟨hh.1, hh.2.1⟩), if_neg hwj]]
  symm
  refine Finset.sum_bij
    (i := fun jL _ => (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ : Fin 128)) ?_ ?_ ?_ ?_
  · intro jL _
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    have := jL.isLt; omega
  · intro a _ b _ hab
    apply Fin.ext
    have : c * 64 + a.val = c * 64 + b.val := by simpa using congrArg Fin.val hab
    omega
  · intro j hj
    have hj2 : c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 := (Finset.mem_filter.mp hj).2
    exact ⟨⟨j.val - c * 64, by omega⟩, Finset.mem_univ _, by apply Fin.ext; simp only; omega⟩
  · intro jL _; rfl

/-- A real `foldl max` over a list, coerced to `WithBot`, is `max` of the seed
with the `WithBot` `foldr` of the coerced list. -/
theorem aft3_foldl_max_coe (m0 : ℝ) (L : List ℝ) :
    ((L.foldl (fun a x => max a x) m0 : ℝ) : WithBot ℝ)
      = max ((m0 : ℝ) : WithBot ℝ) ((L.map (fun x => ((x : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥) := by
  induction L generalizing m0 with
  | nil => simp
  | cons a t ih =>
    simp only [List.foldl_cons, List.map_cons, List.foldr_cons, ih]
    rw [WithBot.coe_max, ← max_assoc]

/-- **`aft3RunningMax` one-block advance.** -/
theorem aft3RunningMax_succ
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (c : Nat) (i : Fin 64) (d : Fin 64) :
    aft3RunningMax qT kT vT keyScale keep ((c + 1) * 64) i d
      = aft3RunningMax qT kT vT keyScale keep (c * 64) i d
        ⊔ ((aft3Block qT kT vT keyScale keep c i d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  unfold aft3RunningMax
  rw [aft3KeysUpto_succ, List.map_append, List.foldr_append]
  generalize (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) (aft3Block qT kT vT keyScale keep c i d)).foldr (· ⊔ ·) ⊥ = B
  induction (aft3KeysUpto qT kT vT keyScale keep (c * 64) i d).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) with
  | nil => simp
  | cons a t ih => simp only [List.foldr_cons, ih, sup_assoc]

/-- At `c·64`, either the seed-`1` state equals the seed-`0` state (nonempty window)
or the running max is `⊥` (empty window, `c = 0`). The decisive split for the step. -/
theorem aft3StateBot1_eq_or_bot
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i d : Fin 64) :
    aft3StateBot1 qT kT vT keyScale keep hi i d = aft3StateBot qT kT vT keyScale keep hi i d
      ∨ aft3RunningMax qT kT vT keyScale keep hi i d = ⊥ := by
  by_cases h : aft3RunningMax qT kT vT keyScale keep hi i d = ⊥
  · exact Or.inr h
  · exact Or.inl (aft3StateBot1_eq_aft3StateBot qT kT vT keyScale keep hi i d h)

/-- `aft3RunningMax` is independent of the channel index `d` (the score `.1` only
involves `qT`/`kT`/`keyScale`, never `vT`/`d`). -/
theorem aft3RunningMax_eq
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin 64) (d d' : Fin 64) :
    aft3RunningMax qT kT vT keyScale keep hi i d
      = aft3RunningMax qT kT vT keyScale keep hi i d' := by
  unfold aft3RunningMax aft3KeysUpto
  congr 1
  rw [List.map_filterMap, List.map_filterMap]
  apply List.filterMap_congr
  intro j _
  by_cases hj : j.val < hi ∧ keep i j <;> simp [hj]

/-- The ⊥-seeded denominator (`aft3StateBot.2.1`) is independent of the channel
index `d` — it depends on `d` only through the (d-independent) running max and the
score-only key list. -/
theorem aft3StateBot_snd_fst_indep
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin 64) (d d' : Fin 64) :
    (aft3StateBot qT kT vT keyScale keep hi i d).2.1
      = (aft3StateBot qT kT vT keyScale keep hi i d').2.1 := by
  rw [aft3StateBot_snd_fst, aft3StateBot_snd_fst,
    aft3RunningMax_eq qT kT vT keyScale keep hi i d d']
  congr 2
  unfold aft3KeysUpto
  rw [List.map_filterMap, List.map_filterMap]
  apply List.filterMap_congr
  intro j _
  by_cases hj : j.val < hi ∧ keep i j <;> simp [hj]

/-! ### Case 3 per-channel register bridges (no-window)

These specialize the flash-attn bridge layer to the `noWindowKeep` predicate (every
key kept). Because there is no mask, the kernel's `qk` cell is the raw score and the
`reduceMax`/`reduceSum`/`dot` reductions equal the `aft3Block` list sums directly. -/

/-- Any member of a `WithBot ℝ` list is `≤` its `foldr (⊔) ⊥`. -/
theorem aft3_mem_le_foldr_sup (a : WithBot ℝ) :
    ∀ (L : List (WithBot ℝ)), a ∈ L → a ≤ L.foldr (· ⊔ ·) ⊥ := by
  intro L
  induction L with
  | nil => intro h; simp at h
  | cons x t ih =>
    intro h
    simp only [List.foldr_cons]
    rcases List.mem_cons.mp h with h | h
    · rw [h]; exact le_sup_left
    · exact le_trans (ih h) le_sup_right

/-- `aft3RunningMax` over a nonempty window `[0, hi)` (`hi > 0`) for `noWindowKeep`
is `≠ ⊥`: key 0 is always kept, so the score list is nonempty. -/
theorem aft3RunningMax_noWindow_ne_bot (s : BlockState) (Q K V : RegionName) (hi : Nat)
    (hhi : 0 < hi) (i d : Fin 64) :
    aft3RunningMax (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3
        (fun i j => noWindowKeep i j) hi i d ≠ ⊥ := by
  unfold aft3RunningMax aft3KeysUpto
  set sc := keyScale3 (⟨0, by norm_num⟩ : Fin 128) *
      Finset.univ.sum (fun e : Fin 64 => qTile3 s Q (i, e, PUnit.unit) *
        kTile3 s K (⟨0, by norm_num⟩, e, PUnit.unit)) with hsc
  set L := ((List.finRange 128).filterMap (fun j : Fin 128 =>
      if j.val < hi ∧ noWindowKeep i j then
        some (keyScale3 j * Finset.univ.sum (fun e : Fin 64 =>
                qTile3 s Q (i, e, PUnit.unit) * kTile3 s K (j, e, PUnit.unit)),
              vTile3 s V (j, d, PUnit.unit))
      else none)).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) with hL
  have hmemL : ((sc : ℝ) : WithBot ℝ) ∈ L := by
    rw [hL, List.mem_map]
    refine ⟨(sc, vTile3 s V (⟨0, by norm_num⟩, d, PUnit.unit)), ?_, rfl⟩
    rw [List.mem_filterMap]
    refine ⟨⟨0, by norm_num⟩, List.mem_finRange _, ?_⟩
    rw [if_pos ⟨hhi, by trivial⟩]
  have hle : ((sc : ℝ) : WithBot ℝ) ≤ L.foldr (· ⊔ ·) ⊥ := aft3_mem_le_foldr_sup _ L hmemL
  intro hbot
  exact absurd (le_bot_iff.mp (hbot ▸ hle)) WithBot.coe_ne_bot

/-- If the ⊥-seeded running max over `[0, hi)` is `⊥`, the key list is empty, hence
its `pow2`-score (resp. `·v`) sum is `0`. -/
theorem aft3KeysUpto_sum_zero_of_bot
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin 64) (d : Fin 64)
    (hbot : aft3RunningMax qT kT vT keyScale keep hi i d = ⊥) (h : ℝ × ℝ → ℝ) :
    ((aft3KeysUpto qT kT vT keyScale keep hi i d).map h).sum = 0 := by
  rw [show aft3KeysUpto qT kT vT keyScale keep hi i d = [] from ?_, List.map_nil, List.sum_nil]
  by_contra hne
  obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ hne
  have hmem : ((p.1 : ℝ) : WithBot ℝ) ∈
      (aft3KeysUpto qT kT vT keyScale keep hi i d).map (fun q => ((q.1 : ℝ) : WithBot ℝ)) :=
    List.mem_map_of_mem hp
  have := aft3_mem_le_foldr_sup _ _ hmem
  rw [← aft3RunningMax, hbot] at this
  exact absurd (le_bot_iff.mp this) WithBot.coe_ne_bot

/-- The ⊥-seeded denominator after `c` blocks: `l = κ(M_c)·L_c`. -/
theorem aft3_denom_anchor
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin 64) (d : Fin 64) :
    (aft3StateBot qT kT vT keyScale keep hi i d).2.1
      = ((aft3StateBot qT kT vT keyScale keep hi i d).1.elim 0 (fun r => pow2 (-r)))
        * (0 + ((aft3KeysUpto qT kT vT keyScale keep hi i d).map (fun p => pow2 p.1)).sum) := by
  rw [aft3StateBot_snd_fst, aft3StateBot_fst_eq_runningMax, zero_add]

/-- The ⊥-seeded accumulator after `c` blocks: `acc = κ(M_c)·T_c`. -/
theorem aft3_acc_anchor
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin 64) (d : Fin 64) :
    (aft3StateBot qT kT vT keyScale keep hi i d).2.2
      = ((aft3StateBot qT kT vT keyScale keep hi i d).1.elim 0 (fun r => pow2 (-r)))
        * (0 + ((aft3KeysUpto qT kT vT keyScale keep hi i d).map (fun p => pow2 p.1 * p.2)).sum) := by
  rw [aft3StateBot_snd_snd, aft3StateBot_fst_eq_runningMax, zero_add]

/-- Canonical axis-1 index of `[64, 64]`. -/
abbrev aft3Ax1 : Fin [64, 64].length := ⟨1, by simp⟩

/-- **The `q·k` score cell (case 3).** With `q` loaded as `some (qT idx)` and the K
block reading `kT` at global key `c·64 + jL`, the case-3 `qk` tile cell `(i, jL)`
(= `(0 + dot q k)·sc`) is `some (sc · Σ_e qT(i,e)·kT(c·64+jL, e))`. -/
theorem aft3_score_cell (s0 : BlockState) (Q K : RegionName) (sc : ℝ) (c : Nat)
    (i jL : Fin 64) (hjL : c * 64 + jL.val < 128)
    (qtile : Tile .real [64, 64]) (ktile : Tile .real [64, 64])
    (hq : qtile = ⟨fun idx : TileIndex [64, 64] => some (qTile3 s0 Q idx)⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (baseOffset3 s0 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64))) :
    (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
        (Tile.scalar (some sc))).data (i, jL, PUnit.unit)
      = some (sc * Finset.univ.sum (fun e : Fin 64 =>
          qTile3 s0 Q (i, e, PUnit.unit) * kTile3 s0 K (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit))) := by
  have hdot : (Tile.dot [] qtile ktile).data (i, jL, PUnit.unit)
      = some (Finset.univ.sum (fun e : Fin 64 =>
          qTile3 s0 Q (i, e, PUnit.unit) * kTile3 s0 K (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit))) := by
    rw [Tile.dot_nil_data]
    rw [show (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
          (fun e => Option.map₂ (· * ·) (qtile.data (i, e, PUnit.unit)) (ktile.data (e, jL, PUnit.unit))))
        = @Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
          (fun e => (some (qTile3 s0 Q (i, e, PUnit.unit)
              * kTile3 s0 K (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit)) : WithBot ℝ))
        from Finset.sum_congr rfl (fun e _ => by
          simp only [hq, hk (e, jL, PUnit.unit), Option.map₂, Option.bind, Option.map]
          refine congrArg some ?_
          rw [show kTile3 s0 K (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit)
                = s0.readMem K (baseOffset3 s0 + e.val * 1 + (c * 64 + jL.val) * 64) from by
            simp only [kTile3]; congr 1; ring])]
    rw [WithBot.sum_someTerm_eq_some]
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Broadcast.scalarR,
    Tile.scalar_data, NumericDType.mul, NumericDType.add, hdot]
  show some _ = _
  refine congrArg some ?_
  simp only [WithBot.realMul, WithBot.realAdd, Option.map₂, Option.bind, Option.map, zero_add]
  ring

/-- **`reduceMax` row (case 3).** -/
theorem aft3_reduceMaxDrop_row (qk : Tile .real [64, 64]) (rmaxT : Tile .real [64])
    (hrm : Tile.reduceMaxDrop aft3Ax1 qk = some rmaxT)
    (i : Fin 64) (g : Fin 64 → WithBot ℝ)
    (hqk : ∀ jL : Fin 64, qk.data (i, jL, PUnit.unit) = g jL) :
    rmaxT.data (i, PUnit.unit) = Finset.univ.sup g := by
  unfold Tile.reduceMaxDrop at hrm
  rw [dif_pos (show 0 < TileShape.axisDim [64, 64] (⟨1, by simp⟩ : Fin [64, 64].length) from by decide)] at hrm
  rw [← Option.some.inj hrm]
  simp only [Finset.sup'_eq_sup]
  exact Finset.sup_congr rfl (fun jL _ => hqk jL)

/-- `filterMap`-then-coe `foldr ⊔ ⊥` over `finRange n` equals the masked `Finset.sup`. -/
theorem aft3_filterMap_foldr_sup (n : Nat) (P : Fin n → Prop) [DecidablePred P]
    (sc : Fin n → ℝ) :
    (((List.finRange n).filterMap (fun j => if P j then some (sc j) else none)).map
        (fun x => ((x : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = Finset.univ.sup (fun j : Fin n => if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥) := by
  rw [show (((List.finRange n).filterMap (fun j => if P j then some (sc j) else none)).map
        (fun x => ((x : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = (List.finRange n).foldr (fun j a => (if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥) ⊔ a) ⊥ from by
    induction (List.finRange n) with
    | nil => simp
    | cons a t ih => by_cases ha : P a <;> simp [ha, ih]]
  apply le_antisymm
  · induction (List.finRange n) with
    | nil => simp
    | cons a t ih =>
      simp only [List.foldr_cons]
      exact sup_le (Finset.le_sup (f := fun j : Fin n => if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥)
        (Finset.mem_univ a)) ih
  · apply Finset.sup_le
    intro j _
    have key : ∀ (l : List (Fin n)), j ∈ l →
        (if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥)
          ≤ l.foldr (fun j a => (if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥) ⊔ a) ⊥ := by
      intro l hl
      induction l with
      | nil => simp at hl
      | cons a t ih =>
        simp only [List.foldr_cons]
        rcases List.mem_cons.mp hl with h | h
        · subst h; exact le_sup_left
        · exact le_trans (ih h) le_sup_right
    exact key _ (List.mem_finRange j)

/-- **Block sup (case 3).** The kernel's `tl.max(qk, 1)` over block `c` (every cell
the raw score `some (sc · Σ q·k)`) equals the `aft3Block`-windowed `foldr ⊔ ⊥`. -/
theorem aft3Block_noWindow_blockSup (s0 : BlockState) (Q K V : RegionName) (sc : ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 64)
    (hsc : sc = keyScale3 (⟨0, by norm_num⟩ : Fin 128)) :
    Finset.univ.sup (fun jL : Fin 64 =>
        ((sc * Finset.univ.sum (fun e : Fin 64 =>
            qTile3 s0 Q (i, e, PUnit.unit) *
              kTile3 s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ))
      = ((aft3Block (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
          (fun i j => noWindowKeep i j) c i d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  rw [show (aft3Block (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
        (fun i j => noWindowKeep i j) c i d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
      = ((List.finRange 128).filterMap (fun j : Fin 128 =>
          if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64
          then some (keyScale3 j * Finset.univ.sum (fun e : Fin 64 =>
                qTile3 s0 Q (i, e, PUnit.unit) * kTile3 s0 K (j, e, PUnit.unit))) else none)).map
            (fun x => ((x : ℝ) : WithBot ℝ)) from by
    unfold aft3Block
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr
    intro j _
    by_cases hj : c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 <;>
      simp [hj, noWindowKeep]]
  rw [aft3_filterMap_foldr_sup 128
    (fun j => c * 64 ≤ j.val ∧ j.val < (c + 1) * 64)
    (fun j => keyScale3 j * Finset.univ.sum (fun e : Fin 64 =>
        qTile3 s0 Q (i, e, PUnit.unit) * kTile3 s0 K (j, e, PUnit.unit)))]
  symm
  apply le_antisymm
  · apply Finset.sup_le; intro j _
    by_cases hj : c * 64 ≤ j.val ∧ j.val < (c + 1) * 64
    · rw [if_pos hj]
      have hjL : j.val - c * 64 < 64 := by omega
      refine le_trans ?_ (Finset.le_sup (Finset.mem_univ (⟨j.val - c * 64, hjL⟩ : Fin 64)))
      simp only
      have hfin : (⟨c * 64 + (j.val - c * 64), by omega⟩ : Fin 128) = j := by apply Fin.ext; simp; omega
      apply le_of_eq
      rw [hsc, keyScale3, keyScale3]
      congr 1; rw [hfin]
    · rw [if_neg hj]; exact bot_le
  · apply Finset.sup_le; intro jL _
    have hb : c * 64 + jL.val < 128 := by have := jL.isLt; omega
    refine le_trans ?_ (Finset.le_sup (Finset.mem_univ (⟨c * 64 + jL.val, hb⟩ : Fin 128)))
    simp only
    rw [if_pos (by have := jL.isLt; exact ⟨by omega, by omega⟩)]
    apply le_of_eq
    rw [hsc, keyScale3, keyScale3]

/-- **`m_ij = aft3RunningMax((c+1)·64)` (case 3).** -/
theorem aft3_mij_reg_eq (s0 : BlockState) (Q K V : RegionName) (sc : ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 64)
    (hsc : sc = keyScale3 (⟨0, by norm_num⟩ : Fin 128))
    (qtile ktile : Tile .real [64, 64]) (mtile rmaxT : Tile .real [64])
    (qkT : Tile .real [64, 64])
    (hq : qtile = ⟨fun idx : TileIndex [64, 64] => some (qTile3 s0 Q idx)⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (baseOffset3 s0 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (hqkT : qkT = Tile.bop NumericDType.real.mul Broadcast.scalarR
              (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
              (Tile.scalar (some sc)))
    (hmtile : mtile.data (i, PUnit.unit)
        = aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
            (fun i j => noWindowKeep i j) (c * 64) i ⟨0, by norm_num⟩)
    (hrmax : Tile.reduceMaxDrop aft3Ax1 qkT = some rmaxT) :
    (Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT).data (i, PUnit.unit)
      = aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
          (fun i j => noWindowKeep i j) ((c + 1) * 64) i ⟨0, by norm_num⟩ := by
  have hrmaxcell : rmaxT.data (i, PUnit.unit)
      = ((aft3Block (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
          (fun i j => noWindowKeep i j) c i ⟨0, by norm_num⟩).map
          (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    rw [aft3_reduceMaxDrop_row qkT rmaxT hrmax i
      (fun jL => ((sc * Finset.univ.sum (fun e : Fin 64 =>
          qTile3 s0 Q (i, e, PUnit.unit) *
            kTile3 s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ))
      (fun jL => by rw [hqkT]; exact aft3_score_cell s0 Q K sc c i jL (by have := jL.isLt; omega) qtile ktile hq hk)]
    exact aft3Block_noWindow_blockSup s0 Q K V sc c hc1 i ⟨0, by norm_num⟩ hsc
  rw [aft3RunningMax_succ (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
      (fun i j => noWindowKeep i j) c i ⟨0, by norm_num⟩]
  rw [Tile.select_data, Tile.cop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hmtile, hrmaxcell]
  set M := aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
      (fun i j => noWindowKeep i j) (c * 64) i ⟨0, by norm_num⟩
  set S := ((aft3Block (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
      (fun i j => noWindowKeep i j) c i ⟨0, by norm_num⟩).map
      (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
  by_cases h : M ≤ S
  · rw [if_neg (by simp [not_lt.mpr h]), max_eq_right h]
  · rw [if_pos (by simpa using not_le.mp h), max_eq_left (le_of_lt (not_le.mp h))]

/-- **`Σ_jL p[i,jL] = aft3Block` pow2-score sum (case 3).** -/
theorem aft3_nume_row_sum (s0 : BlockState) (Q K V : RegionName) (sc : ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 64)
    (hsc : sc = keyScale3 (⟨0, by norm_num⟩ : Fin 128))
    (qtile ktile : Tile .real [64, 64]) (mijT : Tile .real [64]) (pT : Tile .real [64, 64])
    (qkT : Tile .real [64, 64])
    (hq : qtile = ⟨fun idx : TileIndex [64, 64] => some (qTile3 s0 Q idx)⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (baseOffset3 s0 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (hqkT : qkT = Tile.bop NumericDType.real.mul Broadcast.scalarR
              (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
              (Tile.scalar (some sc)))
    (Mr : ℝ) (hmij : mijT.data (i, PUnit.unit) = (Mr : WithBot ℝ))
    (hpT : pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
        (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mijT))) :
    (Tile.reduceSumDrop aft3Ax1 pT).data (i, PUnit.unit)
      = some ((aft3Block (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
          (fun i j => noWindowKeep i j) c i d).map (fun p => pow2 (p.1 - Mr))).sum := by
  rw [Tile.reduceSumDrop_data]
  have hcell : ∀ jL : Fin 64,
      pT.data (TileShape.insertAxisIndex [64, 64] (⟨1, by simp⟩ : Fin [64, 64].length) (i, PUnit.unit) jL)
        = some (pow2 ((sc * Finset.univ.sum (fun e : Fin 64 =>
              qTile3 s0 Q (i, e, PUnit.unit) *
                kTile3 s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit))) - Mr)) := by
    intro jL
    rw [show (TileShape.insertAxisIndex [64, 64] (⟨1, by simp⟩ : Fin [64, 64].length) (i, PUnit.unit) jL) = (i, jL, PUnit.unit) from rfl]
    rw [hpT]
    show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
      TileShape.dropInsertedIndex, NumericDType.sub, hmij]
    rw [hqkT, aft3_score_cell s0 Q K sc c i jL (by have := jL.isLt; omega) qtile ktile hq hk]
    simp only [WithBot.realSub, Option.map₂, Option.bind, Option.map, WithBot.realExp2_some]
    refine congrArg some ?_; simp only [pow2]; ring_nf
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [aft3Block_map_sum (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
      (fun i j => noWindowKeep i j) c i d hc1 (fun p => pow2 (p.1 - Mr))]
  refine Finset.sum_congr rfl (fun jL _ => ?_)
  rw [if_pos (by trivial)]
  rw [hsc, keyScale3, keyScale3]

/-- **`Σ_jL p[i,jL]·v[jL,d] = aft3Block` pow2-score·v sum (case 3).** -/
theorem aft3_acc_dot_block (s0 : BlockState) (Q K V : RegionName) (sc : ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 64)
    (hsc : sc = keyScale3 (⟨0, by norm_num⟩ : Fin 128))
    (qtile ktile vtile : Tile .real [64, 64]) (mijT : Tile .real [64]) (pT : Tile .real [64, 64])
    (qkT : Tile .real [64, 64])
    (hq : qtile = ⟨fun idx : TileIndex [64, 64] => some (qTile3 s0 Q idx)⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (baseOffset3 s0 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (hv : ∀ idx : TileIndex [64, 64],
        vtile.data idx = some (s0.readMem V (baseOffset3 s0 + (c * 64 + idx.1.val) * 64 + idx.2.1.val * 1)))
    (hqkT : qkT = Tile.bop NumericDType.real.mul Broadcast.scalarR
              (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
              (Tile.scalar (some sc)))
    (Mr : ℝ) (hmij : mijT.data (i, PUnit.unit) = (Mr : WithBot ℝ))
    (hpT : pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
        (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mijT))) :
    (Tile.dot [] pT vtile).data (i, d, PUnit.unit)
      = some ((aft3Block (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
          (fun i j => noWindowKeep i j) c i d).map (fun p => pow2 (p.1 - Mr) * p.2)).sum := by
  rw [Tile.dot_nil_data]
  have hcell : ∀ jL : Fin 64,
      Option.map₂ (· * ·) (pT.data (i, jL, PUnit.unit)) (vtile.data (jL, d, PUnit.unit))
        = some (pow2 ((sc * Finset.univ.sum (fun e : Fin 64 =>
              qTile3 s0 Q (i, e, PUnit.unit) *
                kTile3 s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit))) - Mr)
              * vTile3 s0 V (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit)) := by
    intro jL
    have hpcell : pT.data (i, jL, PUnit.unit)
        = some (pow2 ((sc * Finset.univ.sum (fun e : Fin 64 =>
            qTile3 s0 Q (i, e, PUnit.unit) *
              kTile3 s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit))) - Mr)) := by
      rw [hpT]
      show WithBot.realExp2 _ = _
      simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
        TileShape.dropInsertedIndex, NumericDType.sub, hmij]
      rw [hqkT, aft3_score_cell s0 Q K sc c i jL (by have := jL.isLt; omega) qtile ktile hq hk]
      simp only [WithBot.realSub, Option.map₂, Option.bind, Option.map, WithBot.realExp2_some]
      refine congrArg some ?_; simp only [pow2]; ring_nf
    rw [hpcell, hv (jL, d, PUnit.unit)]
    rw [show s0.readMem V (baseOffset3 s0 + (c * 64 + jL.val) * 64 + d.val * 1)
          = vTile3 s0 V (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit) from by
      simp only [vTile3]; congr 1; ring]
    simp only [Option.map₂, Option.bind, Option.map]
  rw [show (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·) (pT.data (i, k, PUnit.unit)) (vtile.data (k, d, PUnit.unit))))
      = @Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ (fun jL =>
          (some (pow2 ((sc * Finset.univ.sum (fun e : Fin 64 =>
                qTile3 s0 Q (i, e, PUnit.unit) *
                  kTile3 s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit))) - Mr)
              * vTile3 s0 V (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit)) : WithBot ℝ))
      from Finset.sum_congr rfl (fun jL _ => hcell jL)]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [aft3Block_map_sum (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
      (fun i j => noWindowKeep i j) c i d hc1 (fun p => pow2 (p.1 - Mr) * p.2)]
  refine Finset.sum_congr rfl (fun jL _ => ?_)
  rw [if_pos (by trivial)]
  rw [hsc, keyScale3, keyScale3]

set_option maxHeartbeats 1000000 in
/-- **`l_i' = aft3StateBot((c+1)·64).2.1` (case 3).** The kernel's `l_i·α + Σp`
register equals the ⊥-seeded denominator after `c+1` blocks. -/
theorem aft3_denom_reg_eq (s0 : BlockState) (Q K V : RegionName) (sc : ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 64)
    (hsc : sc = keyScale3 (⟨0, by norm_num⟩ : Fin 128))
    (qtile ktile : Tile .real [64, 64]) (qkT : Tile .real [64, 64])
    (ltile mtile mijT alphaT : Tile .real [64]) (pT : Tile .real [64, 64])
    (hq : qtile = ⟨fun idx : TileIndex [64, 64] => some (qTile3 s0 Q idx)⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (baseOffset3 s0 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (hqkT : qkT = Tile.bop NumericDType.real.mul Broadcast.scalarR
              (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
              (Tile.scalar (some sc)))
    (hltile : ltile.data (i, PUnit.unit) = some
        ((aft3StateBot1 (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
            (fun i j => noWindowKeep i j) (c * 64) i ⟨0, by norm_num⟩).2.1))
    (hmtile : mtile.data (i, PUnit.unit)
        = aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
            (fun i j => noWindowKeep i j) (c * 64) i ⟨0, by norm_num⟩)
    (hmij : mijT.data (i, PUnit.unit)
        = aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
            (fun i j => noWindowKeep i j) ((c + 1) * 64) i ⟨0, by norm_num⟩)
    (halpha : alphaT = Tile.uop WithBot.realExp2
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
    (hpT : pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
        (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mijT))) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
        (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile alphaT)
        (Tile.reduceSumDrop aft3Ax1 pT)).data (i, PUnit.unit)
      = some ((aft3StateBot (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
          (fun i j => noWindowKeep i j) ((c + 1) * 64) i ⟨0, by norm_num⟩).2.1) := by
  set qT := qTile3 s0 Q
  set kT := kTile3 s0 K
  set vT := vTile3 s0 V
  set kp : Fin 64 → Fin 128 → Prop := fun i j => noWindowKeep i j
  set m := (aft3StateBot qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩).1 with hm_def
  set Mc := aft3RunningMax qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩ with hMc
  set Mc1 := aft3RunningMax qT kT vT keyScale3 kp ((c + 1) * 64) i ⟨0, by norm_num⟩ with hMc1
  have hmMc : m = Mc := by rw [hm_def, hMc, aft3StateBot_fst_eq_runningMax]
  have hne : Mc1 ≠ ⊥ := by
    rw [hMc1]
    exact aft3RunningMax_noWindow_ne_bot s0 Q K V ((c + 1) * 64) (by positivity) i ⟨0, by norm_num⟩
  obtain ⟨Mr, hMr⟩ : ∃ Mr : ℝ, Mc1 = (Mr : WithBot ℝ) := by
    cases hh : Mc1 with
    | bot => exact absurd hh hne
    | coe x => exact ⟨x, rfl⟩
  have hMsucc : Mc1 = m ⊔ ((aft3Block qT kT vT keyScale3 kp c i ⟨0, by norm_num⟩).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    have h1 : Mc1 = (aft3StateBot qT kT vT keyScale3 kp ((c + 1) * 64) i ⟨0, by norm_num⟩).1 := by
      rw [hMc1, aft3StateBot_fst_eq_runningMax]
    rw [h1, aft3StateBot_succ, aft3OsStepBot_block_fst m
        ((aft3StateBot qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩).2.1)
        ((aft3StateBot qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩).2.2)]
  have halphaVal : alphaT.data (i, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [halpha]; show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmtile, hmij,
      NumericDType.sub, ← hMc, ← hMc1, hmMc]
  have hsum := aft3_nume_row_sum s0 Q K V sc c hc1 i ⟨0, by norm_num⟩ hsc qtile ktile mijT pT qkT
    hq hk hqkT Mr (by rw [hmij]; exact hMr) hpT
  have hblockEq := aft3OsStepBot_block_eq m
    ((aft3StateBot qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩).2.1)
    ((aft3StateBot qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩).2.2)
    ((aft3KeysUpto qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩).map (fun p => pow2 p.1 * p.2)).sum
    ((aft3KeysUpto qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩).map (fun p => pow2 p.1)).sum
    (aft3Block qT kT vT keyScale3 kp c i ⟨0, by norm_num⟩)
    (by rw [aft3_denom_anchor, zero_add, hm_def])
    (by rw [aft3_acc_anchor, zero_add, hm_def])
    (fun hbot => aft3KeysUpto_sum_zero_of_bot qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩
      (by rw [← aft3StateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => aft3KeysUpto_sum_zero_of_bot qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩
      (by rw [← aft3StateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  rw [show (aft3StateBot qT kT vT keyScale3 kp ((c + 1) * 64) i ⟨0, by norm_num⟩).2.1
        = (Mc1, (aft3StateBot qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩).2.1
              * (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
              + ((aft3Block qT kT vT keyScale3 kp c i ⟨0, by norm_num⟩).map (fun p => pow2 (p.1 - Mc1.unbotD 0))).sum,
            _).2.1 from by
    rw [aft3StateBot_succ]; rw [← hblockEq]]
  set α : ℝ := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp2 (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  -- seed-1 vs seed-0 denominator cancel against α
  have hl1 : (aft3StateBot1 qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩).2.1 * α
      = (aft3StateBot qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩).2.1 * α := by
    rcases aft3StateBot1_eq_or_bot qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩ with heq | hbot
    · rw [heq]
    · have hmbot : m = ⊥ := by rw [hm_def, aft3StateBot_fst_eq_runningMax]; exact hbot
      have hα0 : α = 0 := by rw [hαdef, hmbot, WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
      rw [hα0, mul_zero, mul_zero]
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex]
  erw [hsum]
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    hltile, halphaVal, hαsome]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [show (aft3StateBot1 qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩).2.1 * α
        = (aft3StateBot qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩).2.1 * α from hl1]
  rw [hMr, WithBot.unbotD_coe]

set_option maxHeartbeats 1000000 in
/-- **`acc' = aft3StateBot((c+1)·64).2.2` (case 3, per channel `d`).** -/
theorem aft3_acc_reg_eq (s0 : BlockState) (Q K V : RegionName) (sc : ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 64)
    (hsc : sc = keyScale3 (⟨0, by norm_num⟩ : Fin 128))
    (qtile ktile vtile : Tile .real [64, 64]) (qkT : Tile .real [64, 64])
    (acctile acc1T pT : Tile .real [64, 64]) (mtile mijT alphaT : Tile .real [64])
    (hq : qtile = ⟨fun idx : TileIndex [64, 64] => some (qTile3 s0 Q idx)⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (baseOffset3 s0 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (hv : ∀ idx : TileIndex [64, 64],
        vtile.data idx = some (s0.readMem V (baseOffset3 s0 + (c * 64 + idx.1.val) * 64 + idx.2.1.val * 1)))
    (hqkT : qkT = Tile.bop NumericDType.real.mul Broadcast.scalarR
              (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
              (Tile.scalar (some sc)))
    (hacctile : acctile.data (i, d, PUnit.unit) = some
        ((aft3StateBot1 (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
            (fun i j => noWindowKeep i j) (c * 64) i d).2.2))
    (hmtile : mtile.data (i, PUnit.unit)
        = aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
            (fun i j => noWindowKeep i j) (c * 64) i ⟨0, by norm_num⟩)
    (hmij : mijT.data (i, PUnit.unit)
        = aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
            (fun i j => noWindowKeep i j) ((c + 1) * 64) i ⟨0, by norm_num⟩)
    (halpha : alphaT = Tile.uop WithBot.realExp2
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
    (hacc1 : acc1T = Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
    (hpT : pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
        (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mijT))) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        acc1T (Tile.dot [] pT vtile)).data (i, d, PUnit.unit)
      = some ((aft3StateBot (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
          (fun i j => noWindowKeep i j) ((c + 1) * 64) i d).2.2) := by
  set qT := qTile3 s0 Q
  set kT := kTile3 s0 K
  set vT := vTile3 s0 V
  set kp : Fin 64 → Fin 128 → Prop := fun i j => noWindowKeep i j
  set m := (aft3StateBot qT kT vT keyScale3 kp (c * 64) i d).1 with hm_def
  set Mc := aft3RunningMax qT kT vT keyScale3 kp (c * 64) i ⟨0, by norm_num⟩ with hMc
  set Mc1 := aft3RunningMax qT kT vT keyScale3 kp ((c + 1) * 64) i ⟨0, by norm_num⟩ with hMc1
  have hmMc : m = Mc := by
    rw [hm_def, hMc, aft3StateBot_fst_eq_runningMax,
      aft3RunningMax_eq qT kT vT keyScale3 kp (c * 64) i d ⟨0, by norm_num⟩]
  have hne : Mc1 ≠ ⊥ := by
    rw [hMc1]
    exact aft3RunningMax_noWindow_ne_bot s0 Q K V ((c + 1) * 64) (by positivity) i ⟨0, by norm_num⟩
  obtain ⟨Mr, hMr⟩ : ∃ Mr : ℝ, Mc1 = (Mr : WithBot ℝ) := by
    cases hh : Mc1 with
    | bot => exact absurd hh hne
    | coe x => exact ⟨x, rfl⟩
  have hMsucc : Mc1 = m ⊔ ((aft3Block qT kT vT keyScale3 kp c i d).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    have h1 : Mc1 = (aft3StateBot qT kT vT keyScale3 kp ((c + 1) * 64) i d).1 := by
      rw [hMc1, aft3StateBot_fst_eq_runningMax,
        aft3RunningMax_eq qT kT vT keyScale3 kp ((c + 1) * 64) i ⟨0, by norm_num⟩ d]
    rw [h1, aft3StateBot_succ, aft3OsStepBot_block_fst m
        ((aft3StateBot qT kT vT keyScale3 kp (c * 64) i d).2.1)
        ((aft3StateBot qT kT vT keyScale3 kp (c * 64) i d).2.2)]
  have halphaVal : alphaT.data (i, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [halpha]; show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmtile, hmij,
      NumericDType.sub, ← hMc, ← hMc1, hmMc]
  have hdot := aft3_acc_dot_block s0 Q K V sc c hc1 i d hsc qtile ktile vtile mijT pT qkT
    hq hk hv hqkT Mr (by rw [hmij]; exact hMr) hpT
  have hblockEq := aft3OsStepBot_block_eq m
    ((aft3StateBot qT kT vT keyScale3 kp (c * 64) i d).2.1)
    ((aft3StateBot qT kT vT keyScale3 kp (c * 64) i d).2.2)
    ((aft3KeysUpto qT kT vT keyScale3 kp (c * 64) i d).map (fun p => pow2 p.1 * p.2)).sum
    ((aft3KeysUpto qT kT vT keyScale3 kp (c * 64) i d).map (fun p => pow2 p.1)).sum
    (aft3Block qT kT vT keyScale3 kp c i d)
    (by rw [aft3_denom_anchor, zero_add, hm_def])
    (by rw [aft3_acc_anchor, zero_add, hm_def])
    (fun hbot => aft3KeysUpto_sum_zero_of_bot qT kT vT keyScale3 kp (c * 64) i d
      (by rw [← aft3StateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => aft3KeysUpto_sum_zero_of_bot qT kT vT keyScale3 kp (c * 64) i d
      (by rw [← aft3StateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  rw [show (aft3StateBot qT kT vT keyScale3 kp ((c + 1) * 64) i d).2.2
        = (Mc1, _,
            (aft3StateBot qT kT vT keyScale3 kp (c * 64) i d).2.2
              * (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
              + ((aft3Block qT kT vT keyScale3 kp c i d).map (fun p => pow2 (p.1 - Mc1.unbotD 0) * p.2)).sum).2.2
        from by rw [aft3StateBot_succ]; rw [← hblockEq]]
  set α : ℝ := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp2 (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  have hacc1cancel : (aft3StateBot1 qT kT vT keyScale3 kp (c * 64) i d).2.2 * α
      = (aft3StateBot qT kT vT keyScale3 kp (c * 64) i d).2.2 * α := by
    rcases aft3StateBot1_eq_or_bot qT kT vT keyScale3 kp (c * 64) i d with heq | hbot
    · rw [heq]
    · have hmbot : m = ⊥ := by rw [hm_def, aft3StateBot_fst_eq_runningMax]; exact hbot
      have hα0 : α = 0 := by rw [hαdef, hmbot, WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
      rw [hα0, mul_zero, mul_zero]
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex]
  erw [hdot]
  rw [hacc1, Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, hacctile, halphaVal, hαsome]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [show (aft3StateBot1 qT kT vT keyScale3 kp (c * 64) i d).2.2 * α
        = (aft3StateBot qT kT vT keyScale3 kp (c * 64) i d).2.2 * α from hacc1cancel]
  rw [hMr, WithBot.unbotD_coe]

/-! ### Masked per-channel register bridges (cases 1/2, sliding window)

These mirror the case-3 (`noWindowKeep`) bridges, but the kernel's `qk` cell is
the *masked* tile `if mc jL then some score else ⊥` (out-of-window lanes set to
`-inf` by the `tl.where`). The reconciliation hypothesis `hmc` ties the lowered
nat-mask `mc` (= `aft3MaskCell1`/`aft3MaskCell2` on lane `jL`) to the faithful
`keep` predicate at the global key `c·64 + jL`. The empty-block case (`mc` all
false) is now genuinely reachable, so — unlike case 3 — no `ne_bot` assumption is
made: the ⊥-carry (`m_ij = ⊥`, `α = 0`) is handled by the seed-cancellation in
`aft3OsStepBot`. -/

/-- **The masked `q·k` score cell (cases 1/2).** On a kept lane the cell is the
score `some (sc · Σ q·k)`; on a masked lane it is `⊥`. -/
theorem aft3_score_cell_masked (s0 : BlockState) (Q K : RegionName) (sc : ℝ) (c : Nat)
    (i jL : Fin 64) (hjL : c * 64 + jL.val < 128) (mc : Bool)
    (qtile : Tile .real [64, 64]) (ktile : Tile .real [64, 64])
    (hq : qtile = ⟨fun idx : TileIndex [64, 64] => some (qTile3 s0 Q idx)⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (baseOffset3 s0 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64))) :
    (if mc then
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
          (Tile.scalar (some sc))).data (i, jL, PUnit.unit)
      else (⊥ : WithBot ℝ))
      = if mc then some (sc * Finset.univ.sum (fun e : Fin 64 =>
          qTile3 s0 Q (i, e, PUnit.unit) * kTile3 s0 K (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit)))
        else (⊥ : WithBot ℝ) := by
  match mc with
  | Bool.false => rfl
  | Bool.true => exact aft3_score_cell s0 Q K sc c i jL hjL qtile ktile hq hk

/-- **Masked block sup (cases 1/2).** The kernel's `tl.max(qk, 1)` over block `c`
(cell = `if mc jL then some score else ⊥`) equals the `aft3Block`-windowed
`foldr ⊔ ⊥` for the `keep` predicate, given the cell-mask reconciliation. -/
theorem aft3Block_masked_blockSup (s0 : BlockState) (Q K V : RegionName) (sc : ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 64)
    (hsc : sc = keyScale3 (⟨0, by norm_num⟩ : Fin 128))
    (keep : Fin 64 → Fin 128 → Prop) [∀ i j, Decidable (keep i j)]
    (mc : Fin 64 → Bool)
    (hmc : ∀ jL : Fin 64, mc jL = decide (keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩)) :
    Finset.univ.sup (fun jL : Fin 64 =>
        if mc jL then
          ((sc * Finset.univ.sum (fun e : Fin 64 =>
              qTile3 s0 Q (i, e, PUnit.unit) *
                kTile3 s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
        else (⊥ : WithBot ℝ))
      = ((aft3Block (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
          keep c i d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  -- replace the Bool cell-mask by the faithful `keep` predicate at the global key
  rw [show (fun jL : Fin 64 =>
        if mc jL then
          ((sc * Finset.univ.sum (fun e : Fin 64 =>
              qTile3 s0 Q (i, e, PUnit.unit) *
                kTile3 s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
        else (⊥ : WithBot ℝ))
      = (fun jL : Fin 64 =>
          if keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ then
            ((sc * Finset.univ.sum (fun e : Fin 64 =>
                qTile3 s0 Q (i, e, PUnit.unit) *
                  kTile3 s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
          else (⊥ : WithBot ℝ)) from by
    funext jL
    rw [hmc jL]
    by_cases hk : keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩
    · rw [if_pos (by exact decide_eq_true hk), if_pos hk]
    · rw [if_neg (by simp [hk]), if_neg hk]]
  rw [show (aft3Block (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
        keep c i d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
      = ((List.finRange 128).filterMap (fun j : Fin 128 =>
          if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ keep i j
          then some (keyScale3 j * Finset.univ.sum (fun e : Fin 64 =>
                qTile3 s0 Q (i, e, PUnit.unit) * kTile3 s0 K (j, e, PUnit.unit))) else none)).map
            (fun x => ((x : ℝ) : WithBot ℝ)) from by
    unfold aft3Block
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr
    intro j _
    by_cases hj : c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ keep i j <;> simp [hj]]
  rw [aft3_filterMap_foldr_sup 128
    (fun j => c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ keep i j)
    (fun j => keyScale3 j * Finset.univ.sum (fun e : Fin 64 =>
        qTile3 s0 Q (i, e, PUnit.unit) * kTile3 s0 K (j, e, PUnit.unit)))]
  symm
  apply le_antisymm
  · apply Finset.sup_le; intro j _
    by_cases hj : c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ keep i j
    · rw [if_pos hj]
      have hjL : j.val - c * 64 < 64 := by omega
      refine le_trans ?_ (Finset.le_sup (Finset.mem_univ (⟨j.val - c * 64, hjL⟩ : Fin 64)))
      simp only
      have hfin : (⟨c * 64 + (j.val - c * 64), by omega⟩ : Fin 128) = j := by
        apply Fin.ext; simp; omega
      rw [if_pos (show keep i ⟨c * 64 + (j.val - c * 64), by omega⟩ from by rw [hfin]; exact hj.2.2)]
      apply le_of_eq
      rw [hsc, keyScale3, keyScale3]
      congr 1; rw [hfin]
    · rw [if_neg hj]; exact bot_le
  · apply Finset.sup_le; intro jL _
    have hb : c * 64 + jL.val < 128 := by have := jL.isLt; omega
    by_cases hkeep : keep i ⟨c * 64 + jL.val, hb⟩
    · rw [if_pos hkeep]
      refine le_trans ?_ (Finset.le_sup (Finset.mem_univ (⟨c * 64 + jL.val, hb⟩ : Fin 128)))
      simp only
      rw [if_pos (by have := jL.isLt; exact ⟨by omega, by omega, hkeep⟩)]
      apply le_of_eq
      rw [hsc, keyScale3, keyScale3]
    · rw [if_neg hkeep]; exact bot_le

/-- **`m_ij = aft3RunningMax((c+1)·64)` (cases 1/2, masked).** Mirror of
`aft3_mij_reg_eq` with the masked qk cell. -/
theorem aft3_mij_reg_eq_masked (s0 : BlockState) (Q K V : RegionName) (sc : ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 64)
    (hsc : sc = keyScale3 (⟨0, by norm_num⟩ : Fin 128))
    (keep : Fin 64 → Fin 128 → Prop) [∀ i j, Decidable (keep i j)]
    (mc : Fin 64 → Bool)
    (hmc : ∀ jL : Fin 64, mc jL = decide (keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩))
    (qtile ktile : Tile .real [64, 64]) (mtile rmaxT : Tile .real [64])
    (qkT : Tile .real [64, 64])
    (hq : qtile = ⟨fun idx : TileIndex [64, 64] => some (qTile3 s0 Q idx)⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (baseOffset3 s0 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (hqkT : ∀ jL : Fin 64, qkT.data (i, jL, PUnit.unit) =
        if mc jL then
          (Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
            (Tile.scalar (some sc))).data (i, jL, PUnit.unit)
        else (⊥ : WithBot ℝ))
    (hmtile : mtile.data (i, PUnit.unit)
        = aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
            keep (c * 64) i ⟨0, by norm_num⟩)
    (hrmax : Tile.reduceMaxDrop aft3Ax1 qkT = some rmaxT) :
    (Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT).data (i, PUnit.unit)
      = aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
          keep ((c + 1) * 64) i ⟨0, by norm_num⟩ := by
  have hrmaxcell : rmaxT.data (i, PUnit.unit)
      = ((aft3Block (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
          keep c i ⟨0, by norm_num⟩).map
          (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    rw [aft3_reduceMaxDrop_row qkT rmaxT hrmax i
      (fun jL => if mc jL then
          ((sc * Finset.univ.sum (fun e : Fin 64 =>
              qTile3 s0 Q (i, e, PUnit.unit) *
                kTile3 s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
          else (⊥ : WithBot ℝ))
      (fun jL => by
        rw [hqkT jL]
        exact aft3_score_cell_masked s0 Q K sc c i jL (by have := jL.isLt; omega) (mc jL) qtile ktile hq hk)]
    exact aft3Block_masked_blockSup s0 Q K V sc c hc1 i ⟨0, by norm_num⟩ hsc keep mc hmc
  rw [aft3RunningMax_succ (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
      keep c i ⟨0, by norm_num⟩]
  rw [Tile.select_data, Tile.cop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hmtile, hrmaxcell]
  set M := aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
      keep (c * 64) i ⟨0, by norm_num⟩
  set S := ((aft3Block (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
      keep c i ⟨0, by norm_num⟩).map
      (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
  by_cases h : M ≤ S
  · rw [if_neg (by simp [not_lt.mpr h]), max_eq_right h]
  · rw [if_pos (by simpa using not_le.mp h), max_eq_left (le_of_lt (not_le.mp h))]

/-- **Masked `pmT` cell (cases 1/2).** The kernel's masked `p` tile cell on lane
`jL` is `some (exp2(score − Mr))` when kept, `some 0` when masked — given `mij`
cell `= some Mr`. -/
theorem aft3_pmT_cell_masked (s0 : BlockState) (Q K : RegionName) (sc : ℝ) (c : Nat)
    (i jL : Fin 64) (hjL : c * 64 + jL.val < 128) (mcj : Bool)
    (qtile ktile : Tile .real [64, 64]) (mijT : Tile .real [64]) (Mc1 : WithBot ℝ)
    (qkT pT : Tile .real [64, 64])
    (hq : qtile = ⟨fun idx : TileIndex [64, 64] => some (qTile3 s0 Q idx)⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (baseOffset3 s0 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (hqkTcell : qkT.data (i, jL, PUnit.unit)
        = if mcj then
            (Tile.bop NumericDType.real.mul Broadcast.scalarR
              (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
              (Tile.scalar (some sc))).data (i, jL, PUnit.unit)
          else (⊥ : WithBot ℝ))
    (hmij : mijT.data (i, PUnit.unit) = Mc1)
    (hkeptbot : mcj = Bool.true → Mc1 ≠ ⊥)
    (hpT : pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
        (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mijT))) :
    (if mcj then pT.data (i, jL, PUnit.unit) else (some (0.0 : ℝ) : WithBot ℝ))
      = some (if mcj then
          pow2 ((sc * Finset.univ.sum (fun e : Fin 64 =>
            qTile3 s0 Q (i, e, PUnit.unit) *
              kTile3 s0 K (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit))) - Mc1.unbotD 0)
          else 0) := by
  match mcj with
  | Bool.false => show (some (0.0 : ℝ) : WithBot ℝ) = some (0 : ℝ); norm_num
  | Bool.true =>
       obtain ⟨Mr, hMr⟩ : ∃ Mr : ℝ, Mc1 = (Mr : WithBot ℝ) := by
         cases hh : Mc1 with
         | coe x => exact ⟨x, rfl⟩
         | bot => exact absurd hh (hkeptbot rfl)
       show pT.data (i, jL, PUnit.unit) = some (pow2 _ )
       rw [hpT]
       show WithBot.realExp2 _ = _
       simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
         TileShape.dropInsertedIndex, NumericDType.sub, hmij, hMr, WithBot.unbotD_coe]
       rw [show qkT.data (i, jL, PUnit.unit) =
             (Tile.bop NumericDType.real.mul Broadcast.scalarR
              (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
              (Tile.scalar (some sc))).data (i, jL, PUnit.unit) from hqkTcell]
       rw [aft3_score_cell s0 Q K sc c i jL hjL qtile ktile hq hk]
       simp only [WithBot.realSub, Option.map₂, Option.bind, Option.map, WithBot.realExp2_some]
       refine congrArg some ?_; simp only [pow2]; ring_nf

set_option maxHeartbeats 1600000 in
/-- **`Σ_jL pm[i,jL] = aft3Block` pow2-score sum (cases 1/2, masked).** The masked
`p` reduceSum equals the windowed `aft3Block` pow2-score sum. -/
theorem aft3_nume_row_sum_masked (s0 : BlockState) (Q K V : RegionName) (sc : ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 64)
    (hsc : sc = keyScale3 (⟨0, by norm_num⟩ : Fin 128))
    (keep : Fin 64 → Fin 128 → Prop) [∀ i j, Decidable (keep i j)]
    (mc : Fin 64 → Bool)
    (hmc : ∀ jL : Fin 64, mc jL = decide (keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩))
    (qtile ktile : Tile .real [64, 64]) (mijT : Tile .real [64]) (pT pmT : Tile .real [64, 64])
    (qkT : Tile .real [64, 64])
    (hq : qtile = ⟨fun idx : TileIndex [64, 64] => some (qTile3 s0 Q idx)⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (baseOffset3 s0 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (hqkT : ∀ jL : Fin 64, qkT.data (i, jL, PUnit.unit) =
        if mc jL then
          (Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
            (Tile.scalar (some sc))).data (i, jL, PUnit.unit)
        else (⊥ : WithBot ℝ))
    (hmij : mijT.data (i, PUnit.unit)
        = aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3 keep ((c + 1) * 64) i ⟨0, by norm_num⟩)
    (hpT : pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
        (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mijT)))
    (hpmT : ∀ jL : Fin 64, pmT.data (i, jL, PUnit.unit) =
        if mc jL then pT.data (i, jL, PUnit.unit) else (some (0.0 : ℝ) : WithBot ℝ)) :
    (Tile.reduceSumDrop aft3Ax1 pmT).data (i, PUnit.unit)
      = some ((aft3Block (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
          keep c i d).map (fun p =>
            pow2 (p.1 - (aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
              keep ((c + 1) * 64) i ⟨0, by norm_num⟩).unbotD 0))).sum := by
  set Mc1 := aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3 keep ((c + 1) * 64) i ⟨0, by norm_num⟩ with hMc1
  -- a kept lane forces `Mc1 ≠ ⊥` (its score is in the windowed prefix)
  have hkeptbot : ∀ jL : Fin 64, keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ → Mc1 ≠ ⊥ := by
    intro jL hkp
    rw [hMc1, aft3RunningMax_eq (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3 keep
      ((c + 1) * 64) i ⟨0, by norm_num⟩ d]
    unfold aft3RunningMax aft3KeysUpto
    intro hbot
    set sc' := keyScale3 (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ : Fin 128) *
        Finset.univ.sum (fun e : Fin 64 => qTile3 s0 Q (i, e, PUnit.unit) *
          kTile3 s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit)) with hsc'
    have hmem : ((sc' : ℝ) : WithBot ℝ) ∈
        ((List.finRange 128).filterMap (fun j : Fin 128 =>
          if j.val < (c + 1) * 64 ∧ keep i j then
            some (keyScale3 j * Finset.univ.sum (fun e : Fin 64 =>
                    qTile3 s0 Q (i, e, PUnit.unit) * kTile3 s0 K (j, e, PUnit.unit)),
                  vTile3 s0 V (j, d, PUnit.unit))
          else none)).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) := by
      rw [List.mem_map]
      refine ⟨(sc', vTile3 s0 V (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit)), ?_, rfl⟩
      rw [List.mem_filterMap]
      refine ⟨⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, List.mem_finRange _, ?_⟩
      have hlt : (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ : Fin 128).val < (c + 1) * 64 := by
        show c * 64 + jL.val < (c + 1) * 64; have := jL.isLt; omega
      rw [if_pos ⟨hlt, hkp⟩]
    have hle := aft3_mem_le_foldr_sup _ _ hmem
    exact absurd (le_bot_iff.mp (hbot ▸ hle)) WithBot.coe_ne_bot
  rw [Tile.reduceSumDrop_data]
  have hcell : ∀ jL : Fin 64,
      pmT.data (TileShape.insertAxisIndex [64, 64] (⟨1, by simp⟩ : Fin [64, 64].length) (i, PUnit.unit) jL)
        = some (if keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ then
            pow2 ((sc * Finset.univ.sum (fun e : Fin 64 =>
              qTile3 s0 Q (i, e, PUnit.unit) *
                kTile3 s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit))) - Mc1.unbotD 0)
            else 0) := by
    intro jL
    rw [show (TileShape.insertAxisIndex [64, 64] (⟨1, by simp⟩ : Fin [64, 64].length) (i, PUnit.unit) jL) = (i, jL, PUnit.unit) from rfl]
    rw [hpmT jL]
    rw [aft3_pmT_cell_masked s0 Q K sc c i jL (by have := jL.isLt; omega) (mc jL)
      qtile ktile mijT Mc1 qkT pT hq hk (hqkT jL) (by rw [hmij]) ?_ hpT]
    · rw [hmc jL]
      by_cases hkp : keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩
      · rw [if_pos hkp, if_pos (decide_eq_true hkp)]
      · rw [if_neg hkp, if_neg (by simp [hkp])]
    · intro hmcj
      exact hkeptbot jL (by have := (hmc jL).symm.trans hmcj; exact of_decide_eq_true this)
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [aft3Block_map_sum (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
      keep c i d hc1 (fun p => pow2 (p.1 - Mc1.unbotD 0))]
  refine Finset.sum_congr rfl (fun jL _ => ?_)
  have hjLlt : jL.val < 64 := jL.isLt
  have hb : c * 64 + jL.val < 128 := by omega
  by_cases hkp : keep i ⟨c * 64 + jL.val, hb⟩
  · rw [if_pos hkp, if_pos hkp, hsc, keyScale3, keyScale3]
  · rw [if_neg hkp, if_neg hkp]

set_option maxHeartbeats 1600000 in
/-- **`Σ_jL pm[i,jL]·v[jL,d] = aft3Block` pow2-score·v sum (cases 1/2, masked).** -/
theorem aft3_acc_dot_block_masked (s0 : BlockState) (Q K V : RegionName) (sc : ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 64)
    (hsc : sc = keyScale3 (⟨0, by norm_num⟩ : Fin 128))
    (keep : Fin 64 → Fin 128 → Prop) [∀ i j, Decidable (keep i j)]
    (mc : Fin 64 → Bool)
    (hmc : ∀ jL : Fin 64, mc jL = decide (keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩))
    (qtile ktile vtile : Tile .real [64, 64]) (mijT : Tile .real [64]) (pT pmT : Tile .real [64, 64])
    (qkT : Tile .real [64, 64])
    (hq : qtile = ⟨fun idx : TileIndex [64, 64] => some (qTile3 s0 Q idx)⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (baseOffset3 s0 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (hv : ∀ idx : TileIndex [64, 64],
        vtile.data idx = some (s0.readMem V (baseOffset3 s0 + (c * 64 + idx.1.val) * 64 + idx.2.1.val * 1)))
    (hqkT : ∀ jL : Fin 64, qkT.data (i, jL, PUnit.unit) =
        if mc jL then
          (Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
            (Tile.scalar (some sc))).data (i, jL, PUnit.unit)
        else (⊥ : WithBot ℝ))
    (hmij : mijT.data (i, PUnit.unit)
        = aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3 keep ((c + 1) * 64) i ⟨0, by norm_num⟩)
    (hpT : pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
        (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mijT)))
    (hpmT : ∀ jL : Fin 64, pmT.data (i, jL, PUnit.unit) =
        if mc jL then pT.data (i, jL, PUnit.unit) else (some (0.0 : ℝ) : WithBot ℝ)) :
    (Tile.dot [] pmT vtile).data (i, d, PUnit.unit)
      = some ((aft3Block (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
          keep c i d).map (fun p =>
            pow2 (p.1 - (aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
              keep ((c + 1) * 64) i ⟨0, by norm_num⟩).unbotD 0) * p.2)).sum := by
  set Mc1 := aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3 keep ((c + 1) * 64) i ⟨0, by norm_num⟩ with hMc1
  have hkeptbot : ∀ jL : Fin 64, keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ → Mc1 ≠ ⊥ := by
    intro jL hkp
    rw [hMc1, aft3RunningMax_eq (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3 keep
      ((c + 1) * 64) i ⟨0, by norm_num⟩ d]
    unfold aft3RunningMax aft3KeysUpto
    intro hbot
    set sc' := keyScale3 (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ : Fin 128) *
        Finset.univ.sum (fun e : Fin 64 => qTile3 s0 Q (i, e, PUnit.unit) *
          kTile3 s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit)) with hsc'
    have hmem : ((sc' : ℝ) : WithBot ℝ) ∈
        ((List.finRange 128).filterMap (fun j : Fin 128 =>
          if j.val < (c + 1) * 64 ∧ keep i j then
            some (keyScale3 j * Finset.univ.sum (fun e : Fin 64 =>
                    qTile3 s0 Q (i, e, PUnit.unit) * kTile3 s0 K (j, e, PUnit.unit)),
                  vTile3 s0 V (j, d, PUnit.unit))
          else none)).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) := by
      rw [List.mem_map]
      refine ⟨(sc', vTile3 s0 V (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit)), ?_, rfl⟩
      rw [List.mem_filterMap]
      refine ⟨⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, List.mem_finRange _, ?_⟩
      have hlt : (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ : Fin 128).val < (c + 1) * 64 := by
        show c * 64 + jL.val < (c + 1) * 64; have := jL.isLt; omega
      rw [if_pos ⟨hlt, hkp⟩]
    have hle := aft3_mem_le_foldr_sup _ _ hmem
    exact absurd (le_bot_iff.mp (hbot ▸ hle)) WithBot.coe_ne_bot
  rw [Tile.dot_nil_data]
  have hcell : ∀ jL : Fin 64,
      Option.map₂ (· * ·) (pmT.data (i, jL, PUnit.unit)) (vtile.data (jL, d, PUnit.unit))
        = some (if keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ then
            pow2 ((sc * Finset.univ.sum (fun e : Fin 64 =>
              qTile3 s0 Q (i, e, PUnit.unit) *
                kTile3 s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit))) - Mc1.unbotD 0)
              * vTile3 s0 V (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit)
            else 0) := by
    intro jL
    have hpmcell : pmT.data (i, jL, PUnit.unit)
        = some (if keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ then
            pow2 ((sc * Finset.univ.sum (fun e : Fin 64 =>
              qTile3 s0 Q (i, e, PUnit.unit) *
                kTile3 s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit))) - Mc1.unbotD 0)
            else 0) := by
      rw [hpmT jL]
      rw [aft3_pmT_cell_masked s0 Q K sc c i jL (by have := jL.isLt; omega) (mc jL)
        qtile ktile mijT Mc1 qkT pT hq hk (hqkT jL) (by rw [hmij]) ?_ hpT]
      · rw [hmc jL]
        by_cases hkp : keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩
        · rw [if_pos hkp, if_pos (decide_eq_true hkp)]
        · rw [if_neg hkp, if_neg (by simp [hkp])]
      · intro hmcj
        exact hkeptbot jL (by have := (hmc jL).symm.trans hmcj; exact of_decide_eq_true this)
    rw [hpmcell, hv (jL, d, PUnit.unit)]
    rw [show s0.readMem V (baseOffset3 s0 + (c * 64 + jL.val) * 64 + d.val * 1)
          = vTile3 s0 V (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit) from by
      simp only [vTile3]; congr 1; ring]
    by_cases hkp : keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩
    · rw [if_pos hkp, if_pos hkp]; rfl
    · rw [if_neg hkp, if_neg hkp]
      simp only [Option.map₂, Option.bind, Option.map]; rw [zero_mul]
  rw [show (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·) (pmT.data (i, k, PUnit.unit)) (vtile.data (k, d, PUnit.unit))))
      = @Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ (fun jL =>
          (some (if keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ then
              pow2 ((sc * Finset.univ.sum (fun e : Fin 64 =>
                qTile3 s0 Q (i, e, PUnit.unit) *
                  kTile3 s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit))) - Mc1.unbotD 0)
                * vTile3 s0 V (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit)
              else 0) : WithBot ℝ))
      from Finset.sum_congr rfl (fun jL _ => hcell jL)]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [aft3Block_map_sum (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
      keep c i d hc1 (fun p => pow2 (p.1 - Mc1.unbotD 0) * p.2)]
  refine Finset.sum_congr rfl (fun jL _ => ?_)
  by_cases hkp : keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩
  · rw [if_pos hkp, if_pos hkp, hsc, keyScale3, keyScale3]
  · rw [if_neg hkp, if_neg hkp]

/-! ### Faithful kernel running state for cases 1/2 (`aft3StateBotK`)

The kernel seeds `l_i = 1`, `acc = 0` at preLoop (window `[0,0)`). But — unlike the
filtered model `aft3StateBot1` — on a **fully-masked** block the kernel *executes*
the block with all lanes zeroed (`α = exp2(⊥ − ⊥) = 0`, `l_ij = 0`), so the seed-1
is annihilated: after any masked block from a `⊥` running-max the register becomes
`0`. Thus the faithful tracked value is the kernel seed at window `0` and the
seed-`0` ⊥-state `aft3StateBot` for every later window: -/
noncomputable def aft3StateBotK
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin 64) (d : Fin 64) :
    WithBot ℝ × ℝ × ℝ :=
  if hi = 0 then (⊥, 1, 0)
  else aft3StateBot qT kT vT keyScale keep hi i d

/-- The running max of `aft3StateBotK` is `aft3RunningMax` (the seed-`1` and
seed-`0` states share the `⊥` max at window `0`). -/
theorem aft3StateBotK_fst
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (hi : Nat) (i : Fin 64) (d : Fin 64) :
    (aft3StateBotK qT kT vT keyScale keep hi i d).1
      = aft3RunningMax qT kT vT keyScale keep hi i d := by
  unfold aft3StateBotK
  split
  · rename_i h; subst h; rw [aft3RunningMax_zero]
  · rw [aft3StateBot_fst_eq_runningMax]

/-- **The seed cancellation.** At every window `c·64`, the kernel state's `l`/`acc`
agree with `aft3StateBot` after multiplying by the next-block rescale `α`
(`= exp2(M_c − M_{c+1})`). At `c = 0` the running max is `⊥`, so `α = 0` kills the
seed-1; for `c > 0` the two states are definitionally equal. -/
theorem aft3StateBotK_cancel
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (c : Nat) (i : Fin 64) (d : Fin 64) (Mc1 : WithBot ℝ) :
    let m := (aft3StateBot qT kT vT keyScale keep (c * 64) i d).1
    let α := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
    (aft3StateBotK qT kT vT keyScale keep (c * 64) i d).2.1 * α
        = (aft3StateBot qT kT vT keyScale keep (c * 64) i d).2.1 * α
      ∧ (aft3StateBotK qT kT vT keyScale keep (c * 64) i d).2.2 * α
        = (aft3StateBot qT kT vT keyScale keep (c * 64) i d).2.2 * α := by
  intro m α
  unfold aft3StateBotK
  by_cases hc0 : c = 0
  · subst hc0
    simp only [Nat.zero_mul, if_pos rfl]
    have hmbot : m = ⊥ := by
      show (aft3StateBot qT kT vT keyScale keep (0 * 64) i d).1 = ⊥
      rw [aft3StateBot_fst_eq_runningMax, Nat.zero_mul, aft3RunningMax_zero]
    have hα0 : α = 0 := by
      show (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 = 0
      rw [hmbot, WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
    rw [hα0]; simp
  · simp only [Nat.mul_eq_zero, hc0, false_or, OfNat.ofNat_ne_zero, or_self, if_neg]
    exact ⟨rfl, rfl⟩

/-- At window `0` the kernel state `aft3StateBotK` is the seed `(⊥, 1, 0)` =
`aft3StateBot1` — the preLoop init, making `attnInvariantK … 0` follow from the
banked `attnInvariant … 0`. -/
theorem aft3StateBotK_zero
    (qT : TileIndex [64, 64] → ℝ) (kT vT : TileIndex [128, 64] → ℝ)
    (keyScale : Fin 128 → ℝ) (keep : Fin 64 → Fin 128 → Prop)
    [∀ i j, Decidable (keep i j)] (i : Fin 64) (d : Fin 64) :
    aft3StateBotK qT kT vT keyScale keep 0 i d
      = aft3StateBot1 qT kT vT keyScale keep 0 i d := by
  unfold aft3StateBotK aft3StateBot1 aft3KeysUpto
  rw [if_pos rfl]
  rw [show (List.finRange 128).filterMap
        (fun j : Fin 128 => if j.val < 0 ∧ keep i j
          then some (keyScale j * Finset.univ.sum (fun e : Fin 64 =>
                qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)), vT (j, d, PUnit.unit))
          else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-- **Masked loop invariant (cases 1/2).** Identical to `attnInvariant` but the
denominator/accumulator carry the faithful kernel state `aft3StateBotK` (seed-1
at window 0, then the seed-0 ⊥-state). -/
noncomputable def attnInvariantK
    (Q K V M Out L : RegionName) (s0 : BlockState)
    (keep : Fin 64 → Fin 128 → Prop) [∀ i j, Decidable (keep i j)]
    (i : Nat) (s : BlockState) : Prop :=
  let qT := qTile3 s0 Q
  let kT := kTile3 s0 K
  let vT := vTile3 s0 V
  s.pids = s0.pids ∧ i % 64 = 0 ∧ i ≤ 128 ∧
  (s.regs .real [64] "m_i" = some ⟨fun r : TileIndex [64] =>
      aft3RunningMax qT kT vT keyScale3 keep i r.1 ⟨0, by norm_num⟩⟩) ∧
  (s.regs .real [64] "l_i" = some ⟨fun r : TileIndex [64] =>
      ((aft3StateBotK qT kT vT keyScale3 keep i r.1 ⟨0, by norm_num⟩).2.1 : ℝ)⟩) ∧
  (s.regs .real [64, 64] "acc" = some ⟨fun idx : TileIndex [64, 64] =>
      ((aft3StateBotK qT kT vT keyScale3 keep i idx.1 idx.2.1).2.2 : ℝ)⟩) ∧
  (s.regs .real [64, 64] "q" = some ⟨fun idx : TileIndex [64, 64] =>
      some (qT idx)⟩) ∧
  (s.regs .real [] "qk_scale" = some (Tile.scalar (some ((1 / 8 : ℝ) * 1.4426950408889634)))) ∧
  (s.regs .nat [] "start_m" = some (Tile.scalar (s0.pids 0))) ∧
  (s.regs .nat [] "off_hz" = some (Tile.scalar (s0.pids 1))) ∧
  (s.regs .blockPtr [64, 64] "K_block_ptr" = some
    (⟨fun _ : TileIndex [64, 64] =>
      { region := K, baseOffset := s0.pids 1 / 4 * 32768 + s0.pids 1 % 4 * 8192,
        parentShape := [64, 128], blockShape := [64, 64], strides := [1, 64],
        offsets := [0, i] }⟩)) ∧
  (s.regs .blockPtr [64, 64] "V_block_ptr" = some
    (⟨fun _ : TileIndex [64, 64] =>
      { region := V, baseOffset := s0.pids 1 / 4 * 32768 + s0.pids 1 % 4 * 8192,
        parentShape := [128, 64], blockShape := [64, 64], strides := [64, 1],
        offsets := [i, 0] }⟩)) ∧
  (s.regs .ptr [64] "m_ptrs" = some
    (Tile.ptrAdd Broadcast.scalarL (Tile.scalar (M.cast, (0 : Nat)))
      (Tile.bop NumericDType.nat.add Broadcast.scalarL (Tile.scalar (s0.pids 1 * 128))
        (Tile.vec (fun r : Fin 64 => s0.pids 0 * 64 + r.val))))) ∧
  (s.regs .ptr [64] "l_ptrs" = some
    (Tile.ptrAdd Broadcast.scalarL (Tile.scalar (L.cast, (0 : Nat)))
      (Tile.bop NumericDType.nat.add Broadcast.scalarL (Tile.scalar (s0.pids 1 * 128))
        (Tile.vec (fun r : Fin 64 => s0.pids 0 * 64 + r.val))))) ∧
  (s.regs .blockPtr [64, 64] "O_block_ptr" = some
    (⟨fun _ : TileIndex [64, 64] =>
      { region := Out, baseOffset := s0.pids 1 / 4 * 32768 + s0.pids 1 % 4 * 8192,
        parentShape := [128, 64], blockShape := [64, 64], strides := [64, 1],
        offsets := [s0.pids 0 * 64, 0] }⟩)) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-- The base case: the banked `attnInvariant … 0` (which carries `aft3StateBot1`)
yields `attnInvariantK … 0` (which carries `aft3StateBotK`), since both equal the
seed `(⊥,1,0)` at window 0. -/
theorem attnInvariant_zero_to_K
    (Q K V M Out L : RegionName) (s0 : BlockState)
    (keep : Fin 64 → Fin 128 → Prop) [∀ i j, Decidable (keep i j)]
    (s : BlockState) (h : attnInvariant Q K V M Out L s0 keep 0 s) :
    attnInvariantK Q K V M Out L s0 keep 0 s := by
  obtain ⟨hpids, hmod, hile, hmi, hli, hacc, hq, hqs, hsm, hoff, hKp, hVp, hMptr, hLptr, hOp, hundef, hmem⟩ := h
  refine ⟨hpids, hmod, hile, hmi, ?_, ?_, hq, hqs, hsm, hoff, hKp, hVp, hMptr, hLptr, hOp, hundef, hmem⟩
  · rw [hli]; simp only [aft3StateBotK_zero]
  · rw [hacc]; simp only [aft3StateBotK_zero]

/-- **Case-1 mask reconciliation (TRUE by construction).** At loop counter `SN =
c·64`, the lowered nat-mask cell `aft3MaskCell1 SM (c·64) (i, jL)` equals the
faithful `natSlidingWindowKeep SM i ⟨c·64 + jL⟩`. The `dist ≥ 0` conjunct is
vacuous on ℕ. -/
theorem aft3MaskCell1_eq_keep (SM c : Nat) (i jL : Fin 64)
    (hb : c * 64 + jL.val < 128) :
    aft3MaskCell1 SM (c * 64) (i, jL, PUnit.unit)
      = decide (natSlidingWindowKeep SM i ⟨c * 64 + jL.val, hb⟩) := by
  unfold aft3MaskCell1 natSlidingWindowKeep natDist3 ComparableDType.ge ComparableDType.lt
  have hjL : jL.val < 64 := jL.isLt
  have hmod : ((⟨c * 64 + jL.val, hb⟩ : Fin 128).val) % 64 = jL.val := by show (c * 64 + jL.val) % 64 = jL.val; omega
  have hdiv : ((⟨c * 64 + jL.val, hb⟩ : Fin 128).val) / 64 = c := by show (c * 64 + jL.val) / 64 = c; omega
  simp only [hmod, hdiv, Nat.zero_le, decide_true, Bool.true_and, Nat.add_zero]

set_option maxHeartbeats 1000000 in
/-- **`l_i' = aft3StateBot((c+1)·64).2.1` (cases 1/2, masked).** From the kernel
seed state `aft3StateBotK(c·64)` the masked `l_i·α + Σpm` register equals the
seed-`0` denominator after `c+1` blocks. -/
theorem aft3_denom_reg_eq_masked (s0 : BlockState) (Q K V : RegionName) (sc : ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 64)
    (hsc : sc = keyScale3 (⟨0, by norm_num⟩ : Fin 128))
    (keep : Fin 64 → Fin 128 → Prop) [∀ i j, Decidable (keep i j)]
    (mc : Fin 64 → Bool)
    (hmc : ∀ jL : Fin 64, mc jL = decide (keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩))
    (qtile ktile : Tile .real [64, 64]) (qkT : Tile .real [64, 64])
    (ltile mtile mijT alphaT : Tile .real [64]) (pT pmT : Tile .real [64, 64])
    (hq : qtile = ⟨fun idx : TileIndex [64, 64] => some (qTile3 s0 Q idx)⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (baseOffset3 s0 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (hqkT : ∀ jL : Fin 64, qkT.data (i, jL, PUnit.unit) =
        if mc jL then
          (Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
            (Tile.scalar (some sc))).data (i, jL, PUnit.unit)
        else (⊥ : WithBot ℝ))
    (hltile : ltile.data (i, PUnit.unit) = some
        ((aft3StateBotK (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
            keep (c * 64) i ⟨0, by norm_num⟩).2.1))
    (hmtile : mtile.data (i, PUnit.unit)
        = aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
            keep (c * 64) i ⟨0, by norm_num⟩)
    (hmij : mijT.data (i, PUnit.unit)
        = aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
            keep ((c + 1) * 64) i ⟨0, by norm_num⟩)
    (halpha : alphaT = Tile.uop WithBot.realExp2
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
    (hpT : pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
        (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mijT)))
    (hpmT : ∀ jL : Fin 64, pmT.data (i, jL, PUnit.unit) =
        if mc jL then pT.data (i, jL, PUnit.unit) else (some (0.0 : ℝ) : WithBot ℝ)) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
        (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile alphaT)
        (Tile.reduceSumDrop aft3Ax1 pmT)).data (i, PUnit.unit)
      = some ((aft3StateBot (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
          keep ((c + 1) * 64) i ⟨0, by norm_num⟩).2.1) := by
  set qT := qTile3 s0 Q
  set kT := kTile3 s0 K
  set vT := vTile3 s0 V
  set m := (aft3StateBot qT kT vT keyScale3 keep (c * 64) i ⟨0, by norm_num⟩).1 with hm_def
  set Mc := aft3RunningMax qT kT vT keyScale3 keep (c * 64) i ⟨0, by norm_num⟩ with hMc
  set Mc1 := aft3RunningMax qT kT vT keyScale3 keep ((c + 1) * 64) i ⟨0, by norm_num⟩ with hMc1
  have hmMc : m = Mc := by rw [hm_def, hMc, aft3StateBot_fst_eq_runningMax]
  have hMsucc : Mc1 = m ⊔ ((aft3Block qT kT vT keyScale3 keep c i ⟨0, by norm_num⟩).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    have h1 : Mc1 = (aft3StateBot qT kT vT keyScale3 keep ((c + 1) * 64) i ⟨0, by norm_num⟩).1 := by
      rw [hMc1, aft3StateBot_fst_eq_runningMax]
    rw [h1, aft3StateBot_succ, aft3OsStepBot_block_fst m
        ((aft3StateBot qT kT vT keyScale3 keep (c * 64) i ⟨0, by norm_num⟩).2.1)
        ((aft3StateBot qT kT vT keyScale3 keep (c * 64) i ⟨0, by norm_num⟩).2.2)]
  have halphaVal : alphaT.data (i, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [halpha]; show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmtile, hmij,
      NumericDType.sub, ← hMc, ← hMc1, hmMc]
  have hsum := aft3_nume_row_sum_masked s0 Q K V sc c hc1 i ⟨0, by norm_num⟩ hsc keep mc hmc
    qtile ktile mijT pT pmT qkT hq hk hqkT hmij hpT hpmT
  have hblockEq := aft3OsStepBot_block_eq m
    ((aft3StateBot qT kT vT keyScale3 keep (c * 64) i ⟨0, by norm_num⟩).2.1)
    ((aft3StateBot qT kT vT keyScale3 keep (c * 64) i ⟨0, by norm_num⟩).2.2)
    ((aft3KeysUpto qT kT vT keyScale3 keep (c * 64) i ⟨0, by norm_num⟩).map (fun p => pow2 p.1 * p.2)).sum
    ((aft3KeysUpto qT kT vT keyScale3 keep (c * 64) i ⟨0, by norm_num⟩).map (fun p => pow2 p.1)).sum
    (aft3Block qT kT vT keyScale3 keep c i ⟨0, by norm_num⟩)
    (by rw [aft3_denom_anchor, zero_add, hm_def])
    (by rw [aft3_acc_anchor, zero_add, hm_def])
    (fun hbot => aft3KeysUpto_sum_zero_of_bot qT kT vT keyScale3 keep (c * 64) i ⟨0, by norm_num⟩
      (by rw [← aft3StateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => aft3KeysUpto_sum_zero_of_bot qT kT vT keyScale3 keep (c * 64) i ⟨0, by norm_num⟩
      (by rw [← aft3StateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  rw [show (aft3StateBot qT kT vT keyScale3 keep ((c + 1) * 64) i ⟨0, by norm_num⟩).2.1
        = (Mc1, (aft3StateBot qT kT vT keyScale3 keep (c * 64) i ⟨0, by norm_num⟩).2.1
              * (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
              + ((aft3Block qT kT vT keyScale3 keep c i ⟨0, by norm_num⟩).map (fun p => pow2 (p.1 - Mc1.unbotD 0))).sum,
            _).2.1 from by
    rw [aft3StateBot_succ]; rw [← hblockEq]]
  set α : ℝ := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp2 (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  have hcancel := (aft3StateBotK_cancel qT kT vT keyScale3 keep c i ⟨0, by norm_num⟩ Mc1).1
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex]
  erw [hsum]
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    hltile, halphaVal, hαsome]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [show (aft3StateBotK qT kT vT keyScale3 keep (c * 64) i ⟨0, by norm_num⟩).2.1 * α
        = (aft3StateBot qT kT vT keyScale3 keep (c * 64) i ⟨0, by norm_num⟩).2.1 * α from hcancel]

set_option maxHeartbeats 1000000 in
/-- **`acc' = aft3StateBot((c+1)·64).2.2` (cases 1/2, masked).** -/
theorem aft3_acc_reg_eq_masked (s0 : BlockState) (Q K V : RegionName) (sc : ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 64)
    (hsc : sc = keyScale3 (⟨0, by norm_num⟩ : Fin 128))
    (keep : Fin 64 → Fin 128 → Prop) [∀ i j, Decidable (keep i j)]
    (mc : Fin 64 → Bool)
    (hmc : ∀ jL : Fin 64, mc jL = decide (keep i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩))
    (qtile ktile vtile : Tile .real [64, 64]) (qkT : Tile .real [64, 64])
    (acctile acc1T pT pmT : Tile .real [64, 64]) (mtile mijT alphaT : Tile .real [64])
    (hq : qtile = ⟨fun idx : TileIndex [64, 64] => some (qTile3 s0 Q idx)⟩)
    (hk : ∀ idx : TileIndex [64, 64],
        ktile.data idx = some (s0.readMem K (baseOffset3 s0 + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)))
    (hv : ∀ idx : TileIndex [64, 64],
        vtile.data idx = some (s0.readMem V (baseOffset3 s0 + (c * 64 + idx.1.val) * 64 + idx.2.1.val * 1)))
    (hqkT : ∀ jL : Fin 64, qkT.data (i, jL, PUnit.unit) =
        if mc jL then
          (Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [64, 64]) (Tile.dot [] qtile ktile))
            (Tile.scalar (some sc))).data (i, jL, PUnit.unit)
        else (⊥ : WithBot ℝ))
    (hacctile : acctile.data (i, d, PUnit.unit) = some
        ((aft3StateBotK (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
            keep (c * 64) i d).2.2))
    (hmtile : mtile.data (i, PUnit.unit)
        = aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
            keep (c * 64) i ⟨0, by norm_num⟩)
    (hmij : mijT.data (i, PUnit.unit)
        = aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
            keep ((c + 1) * 64) i ⟨0, by norm_num⟩)
    (halpha : alphaT = Tile.uop WithBot.realExp2
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
    (hacc1 : acc1T = Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
    (hpT : pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
        (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mijT)))
    (hpmT : ∀ jL : Fin 64, pmT.data (i, jL, PUnit.unit) =
        if mc jL then pT.data (i, jL, PUnit.unit) else (some (0.0 : ℝ) : WithBot ℝ)) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        acc1T (Tile.dot [] pmT vtile)).data (i, d, PUnit.unit)
      = some ((aft3StateBot (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
          keep ((c + 1) * 64) i d).2.2) := by
  set qT := qTile3 s0 Q
  set kT := kTile3 s0 K
  set vT := vTile3 s0 V
  set m := (aft3StateBot qT kT vT keyScale3 keep (c * 64) i d).1 with hm_def
  set Mc := aft3RunningMax qT kT vT keyScale3 keep (c * 64) i ⟨0, by norm_num⟩ with hMc
  set Mc1 := aft3RunningMax qT kT vT keyScale3 keep ((c + 1) * 64) i ⟨0, by norm_num⟩ with hMc1
  have hmMc : m = Mc := by
    rw [hm_def, hMc, aft3StateBot_fst_eq_runningMax,
      aft3RunningMax_eq qT kT vT keyScale3 keep (c * 64) i d ⟨0, by norm_num⟩]
  have hMsucc : Mc1 = m ⊔ ((aft3Block qT kT vT keyScale3 keep c i d).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    have h1 : Mc1 = (aft3StateBot qT kT vT keyScale3 keep ((c + 1) * 64) i d).1 := by
      rw [hMc1, aft3StateBot_fst_eq_runningMax,
        aft3RunningMax_eq qT kT vT keyScale3 keep ((c + 1) * 64) i ⟨0, by norm_num⟩ d]
    rw [h1, aft3StateBot_succ, aft3OsStepBot_block_fst m
        ((aft3StateBot qT kT vT keyScale3 keep (c * 64) i d).2.1)
        ((aft3StateBot qT kT vT keyScale3 keep (c * 64) i d).2.2)]
  have halphaVal : alphaT.data (i, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [halpha]; show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmtile, hmij,
      NumericDType.sub, ← hMc, ← hMc1, hmMc]
  have hdot := aft3_acc_dot_block_masked s0 Q K V sc c hc1 i d hsc keep mc hmc
    qtile ktile vtile mijT pT pmT qkT hq hk hv hqkT hmij hpT hpmT
  have hblockEq := aft3OsStepBot_block_eq m
    ((aft3StateBot qT kT vT keyScale3 keep (c * 64) i d).2.1)
    ((aft3StateBot qT kT vT keyScale3 keep (c * 64) i d).2.2)
    ((aft3KeysUpto qT kT vT keyScale3 keep (c * 64) i d).map (fun p => pow2 p.1 * p.2)).sum
    ((aft3KeysUpto qT kT vT keyScale3 keep (c * 64) i d).map (fun p => pow2 p.1)).sum
    (aft3Block qT kT vT keyScale3 keep c i d)
    (by rw [aft3_denom_anchor, zero_add, hm_def])
    (by rw [aft3_acc_anchor, zero_add, hm_def])
    (fun hbot => aft3KeysUpto_sum_zero_of_bot qT kT vT keyScale3 keep (c * 64) i d
      (by rw [← aft3StateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => aft3KeysUpto_sum_zero_of_bot qT kT vT keyScale3 keep (c * 64) i d
      (by rw [← aft3StateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  rw [show (aft3StateBot qT kT vT keyScale3 keep ((c + 1) * 64) i d).2.2
        = (Mc1, _,
            (aft3StateBot qT kT vT keyScale3 keep (c * 64) i d).2.2
              * (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
              + ((aft3Block qT kT vT keyScale3 keep c i d).map (fun p => pow2 (p.1 - Mc1.unbotD 0) * p.2)).sum).2.2
        from by rw [aft3StateBot_succ]; rw [← hblockEq]]
  set α : ℝ := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp2 (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  have hcancel := (aft3StateBotK_cancel qT kT vT keyScale3 keep c i d Mc1).2
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex]
  erw [hdot]
  rw [hacc1, Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, hacctile, halphaVal, hαsome]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [show (aft3StateBotK qT kT vT keyScale3 keep (c * 64) i d).2.2 * α
        = (aft3StateBot qT kT vT keyScale3 keep (c * 64) i d).2.2 * α from hcancel]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Step lemma (case 1, sliding window).** The masked loop body `aft3LoopBody`
advances `attnInvariantK … natSlidingWindowKeep` by one key block (`i → i + 64`),
via `aft3LoopBody_steps` + the masked `mij`/`denom`/`acc` register bridges. The
mask reconciliation `aft3MaskCell1 ↔ natSlidingWindowKeep` is TRUE by
construction (`aft3MaskCell1_eq_keep`). -/
theorem aft3_attn_step1 (Q K V M Out L : RegionName) (s0 : BlockState)
    (i : Nat) (s : BlockState) (hilt : i < 128) (himod : i % 64 = 0)
    (hinv : attnInvariantK Q K V M Out L s0
      (fun i j => natSlidingWindowKeep (s0.pids 0) i j) i s) :
    ∃ s', stepStmts aft3LoopBody (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ attnInvariantK Q K V M Out L s0
          (fun i j => natSlidingWindowKeep (s0.pids 0) i j) (i + 64) s' := by
  obtain ⟨hpids, hmod, hile, hmi, hli, hacc, hq, hqs, hsm, hoff, hKp, hVp, hMptr, hLptr, hOp, hundef, hmem⟩ := hinv
  set c := i / 64 with hc_def
  have hi : i = c * 64 := by omega
  have hc1 : (c + 1) * 64 ≤ 128 := by omega
  set base := baseOffset3 s0 with hbase
  set sc0 : ℝ := (1 / 8 : ℝ) * 1.4426950408889634 with hsc0
  set qT := qTile3 s0 Q with hqT
  set kT := kTile3 s0 K with hkT
  set vT := vTile3 s0 V with hvT
  set kp : Fin 64 → Fin 128 → Prop := fun i j => natSlidingWindowKeep (s0.pids 0) i j with hkp
  have hbaseEq : (s0.pids 1 / 4 * 32768 + s0.pids 1 % 4 * 8192 : Nat) = base := by
    simp [hbase, baseOffset3]
  -- run the masked loop body chain
  obtain ⟨sF, hchain, hpidsF, hmemF, hundefF, hqF, hqsF, hsmF, hKpF, hVpF, hoffF, hMptrF, hLptrF, hOpF,
      qkT, rmaxT, mijT, alphaT, lijT, pT, pmT, acc1T,
      hqkData, hrm, hmijd, halphad, hpTd, hpmTd, hlijd, hacc1d, hmiF, hliF, haccF⟩ :=
    aft3LoopBody_steps (s.setReg "start_n" .nat [] (Tile.scalar i)) (s0.pids 0) i
      K V base base i i
      (⟨fun idx : TileIndex [64, 64] => some (qT idx)⟩)
      (⟨fun r : TileIndex [64] => aft3RunningMax qT kT vT keyScale3 kp i r.1 ⟨0, by norm_num⟩⟩)
      (⟨fun r : TileIndex [64] => ((aft3StateBotK qT kT vT keyScale3 kp i r.1 ⟨0, by norm_num⟩).2.1 : ℝ)⟩)
      (⟨fun idx : TileIndex [64, 64] => ((aft3StateBotK qT kT vT keyScale3 kp i idx.1 idx.2.1).2.2 : ℝ)⟩)
      (⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem K (base + idx.1.val * 1 + (i + idx.2.1.val) * 64))⟩)
      (⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem V (base + (i + idx.1.val) * 64 + idx.2.1.val * 1))⟩)
      sc0
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsm)
      (by simp [BlockState.setReg_same])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hqs)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmi)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hli)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hacc)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; rw [hKp, hbaseEq])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; rw [hVp, hbaseEq])
      (fun idx => rfl)
      (fun idx => rfl)
      (by intro rg o; rw [BlockState.setReg_undef]; exact hundef rg o)
  refine ⟨sF, hchain, ?_⟩
  have hrmemK : ∀ idx : TileIndex [64, 64],
      (⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem K (base + idx.1.val * 1 + (i + idx.2.1.val) * 64))⟩ : Tile .real [64, 64]).data idx
        = some (s0.readMem K (base + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)) := by
    intro idx
    simp only [BlockState.setReg_readMem]
    unfold BlockState.readMem; rw [hmem, hi]
  have hrmemV : ∀ idx : TileIndex [64, 64],
      (⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem V (base + (i + idx.1.val) * 64 + idx.2.1.val * 1))⟩ : Tile .real [64, 64]).data idx
        = some (s0.readMem V (base + (c * 64 + idx.1.val) * 64 + idx.2.1.val * 1)) := by
    intro idx
    simp only [BlockState.setReg_readMem]
    unfold BlockState.readMem; rw [hmem, hi]
  set qtileF : Tile .real [64, 64] := ⟨fun idx : TileIndex [64, 64] => some (qT idx)⟩ with hqtileF
  set ktileF : Tile .real [64, 64] :=
    ⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem K (base + idx.1.val * 1 + (i + idx.2.1.val) * 64))⟩ with hktileF
  set vtileF : Tile .real [64, 64] :=
    ⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem V (base + (i + idx.1.val) * 64 + idx.2.1.val * 1))⟩ with hvtileF
  have hsckey : sc0 = keyScale3 (⟨0, by norm_num⟩ : Fin 128) := by rw [hsc0, keyScale3]
  -- the cell-mask: aft3MaskCell1 (pids 0) i, reconciled with the keep predicate at c
  set mc : Fin 64 → Bool := fun jL => aft3MaskCell1 (s0.pids 0) i (⟨0, by norm_num⟩, jL, PUnit.unit) with hmcdef
  have hmc_row : ∀ r jL : Fin 64, aft3MaskCell1 (s0.pids 0) i (r, jL, PUnit.unit)
      = decide (kp r ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩) := by
    intro r jL
    have heq : aft3MaskCell1 (s0.pids 0) i (r, jL, PUnit.unit)
        = aft3MaskCell1 (s0.pids 0) (c * 64) (r, jL, PUnit.unit) := by rw [hi]
    rw [heq, aft3MaskCell1_eq_keep (s0.pids 0) c r jL (by have := jL.isLt; omega)]
  have hmc : ∀ jL : Fin 64, mc jL = decide (kp ⟨0, by norm_num⟩ ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩) := by
    intro jL
    rw [hmcdef]; simp only
    exact hmc_row ⟨0, by norm_num⟩ jL
  -- m_i cell readback
  have hmicell : ∀ r : Fin 64,
      (⟨fun r : TileIndex [64] => aft3RunningMax qT kT vT keyScale3 kp i r.1 ⟨0, by norm_num⟩⟩ : Tile .real [64]).data (r, PUnit.unit)
        = aft3RunningMax qT kT vT keyScale3 kp (c * 64) r ⟨0, by norm_num⟩ := by
    intro r; simp only; rw [hi]
  -- mijT cell = aft3RunningMax((c+1)*64)
  have hmijcell : ∀ r : Fin 64, mijT.data (r, PUnit.unit)
      = aft3RunningMax qT kT vT keyScale3 kp ((c + 1) * 64) r ⟨0, by norm_num⟩ := by
    intro r
    rw [hmijd]
    refine aft3_mij_reg_eq_masked s0 Q K V sc0 c hc1 r ⟨0, by norm_num⟩ hsckey kp
      (fun jL => aft3MaskCell1 (s0.pids 0) i (r, jL, PUnit.unit))
      (fun jL => hmc_row r jL)
      qtileF ktileF
      (⟨fun r : TileIndex [64] => aft3RunningMax qT kT vT keyScale3 kp i r.1 ⟨0, by norm_num⟩⟩) rmaxT qkT
      hqtileF hrmemK ?_ ?_ ?_
    · intro jL; rw [hqkData]
    · rw [hmicell r]
    · exact hrm
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF, BlockState.setReg_pids, hpids]
  · omega
  · omega
  · rw [hmiF]; refine congrArg some ?_; ext r
    rw [hmijcell r.1, show ((c + 1) * 64 : Nat) = i + 64 from by omega]
  · rw [hliF]; refine congrArg some ?_; refine Tile.ext (fun r => ?_)
    obtain ⟨r, ⟨⟩⟩ := r
    have hbr := aft3_denom_reg_eq_masked s0 Q K V sc0 c hc1 r ⟨0, by norm_num⟩ hsckey kp
      (fun jL => aft3MaskCell1 (s0.pids 0) i (r, jL, PUnit.unit)) (fun jL => hmc_row r jL)
      qtileF ktileF qkT
      (⟨fun r : TileIndex [64] => ((aft3StateBotK qT kT vT keyScale3 kp i r.1 ⟨0, by norm_num⟩).2.1 : ℝ)⟩)
      (⟨fun r : TileIndex [64] => aft3RunningMax qT kT vT keyScale3 kp i r.1 ⟨0, by norm_num⟩⟩)
      mijT alphaT pT pmT hqtileF hrmemK
      (fun jL => by rw [hqkData])
      (by simp only [hi]; rfl) (by simp only [hi]; rfl)
      (by rw [hmijcell r]) halphad hpTd
      (fun jL => by rw [hpmTd])
    show (Tile.bop NumericDType.real.add _ _ _).data _ = _
    rw [hlijd, hbr]
    show _ = ((aft3StateBotK qT kT vT keyScale3 kp (i + 64) r ⟨0, by norm_num⟩).2.1 : WithBot ℝ)
    rw [show ((i + 64) : Nat) = (c + 1) * 64 from by omega,
      show aft3StateBotK qT kT vT keyScale3 kp ((c + 1) * 64) r ⟨0, by norm_num⟩
        = aft3StateBot qT kT vT keyScale3 kp ((c + 1) * 64) r ⟨0, by norm_num⟩ from by
        unfold aft3StateBotK; rw [if_neg (by omega)]]
    rw [← hqT, ← hkT, ← hvT]; rfl
  · rw [haccF]; refine congrArg some ?_; refine Tile.ext (fun idx => ?_)
    obtain ⟨ir, id, ⟨⟩⟩ := idx
    have hbr := aft3_acc_reg_eq_masked s0 Q K V sc0 c hc1 ir id hsckey kp
      (fun jL => aft3MaskCell1 (s0.pids 0) i (ir, jL, PUnit.unit)) (fun jL => hmc_row ir jL)
      qtileF ktileF vtileF qkT
      (⟨fun idx : TileIndex [64, 64] => ((aft3StateBotK qT kT vT keyScale3 kp i idx.1 idx.2.1).2.2 : ℝ)⟩)
      acc1T pT pmT
      (⟨fun r : TileIndex [64] => aft3RunningMax qT kT vT keyScale3 kp i r.1 ⟨0, by norm_num⟩⟩)
      mijT alphaT hqtileF hrmemK hrmemV
      (fun jL => by rw [hqkData])
      (by simp only [hi]; rfl) (by simp only [hi]; rfl)
      (by rw [hmijcell ir]) halphad hacc1d hpTd
      (fun jL => by rw [hpmTd])
    show (Tile.bop NumericDType.real.add _ _ _).data _ = _
    rw [hbr]
    show _ = ((aft3StateBotK qT kT vT keyScale3 kp (i + 64) ir id).2.2 : WithBot ℝ)
    rw [show ((i + 64) : Nat) = (c + 1) * 64 from by omega,
      show aft3StateBotK qT kT vT keyScale3 kp ((c + 1) * 64) ir id
        = aft3StateBot qT kT vT keyScale3 kp ((c + 1) * 64) ir id from by
        unfold aft3StateBotK; rw [if_neg (by omega)]]
    rw [← hqT, ← hkT, ← hvT]; rfl
  · rw [hqF]
  · rw [hqsF]
  · rw [hsmF]
  · rw [hoffF, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoff
  · rw [hKpF, ← hbaseEq]
  · rw [hVpF, ← hbaseEq]
  · rw [hMptrF, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hMptr
  · rw [hLptrF, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hLptr
  · rw [hOpF, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hOp
  · exact hundefF
  · rw [hmemF]; funext rg o; rw [BlockState.setReg_mem]; exact congrFun (congrFun hmem rg) o

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Step lemma (case 3, no window).** The loop body `aft3LoopBody3` advances
`attnInvariant … noWindowKeep` by one key block (`i → i + 64`), via
`aft3LoopBody3_steps` + the `mij`/`denom`/`acc` register bridges (all reading off
`aft3RunningMax`/`aft3StateBot1` at `(c+1)·64`). -/
theorem aft3_attn_step3 (Q K V M Out L : RegionName) (s0 : BlockState)
    (i : Nat) (s : BlockState) (hilt : i < 128) (himod : i % 64 = 0)
    (hinv : attnInvariant Q K V M Out L s0 (fun i j => noWindowKeep i j) i s) :
    ∃ s', stepStmts aft3LoopBody3 (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ attnInvariant Q K V M Out L s0 (fun i j => noWindowKeep i j) (i + 64) s' := by
  simp only [attnInvariant] at hinv
  obtain ⟨hpids, hmod, hile, hmi, hli, hacc, hq, hqs, hsm, hoff, hKp, hVp, hMptr, hLptr, hOp, hundef, hmem⟩ := hinv
  set c := i / 64 with hc_def
  have hi : i = c * 64 := by omega
  have hc1 : (c + 1) * 64 ≤ 128 := by omega
  set base := baseOffset3 s0 with hbase
  set sc0 : ℝ := (1 / 8 : ℝ) * 1.4426950408889634 with hsc0
  set qT := qTile3 s0 Q with hqT
  set kT := kTile3 s0 K with hkT
  set vT := vTile3 s0 V with hvT
  set kp : Fin 64 → Fin 128 → Prop := fun i j => noWindowKeep i j with hkp
  have hbaseEq : (s0.pids 1 / 4 * 32768 + s0.pids 1 % 4 * 8192 : Nat) = base := by
    simp [hbase, baseOffset3]
  -- run the loop body chain
  obtain ⟨sF, hchain, hpidsF, hmemF, hundefF, hqF, hqsF, hsmF, hoffF, hKpF, hVpF,
      hMptrF, hLptrF, hOpF,
      qkT, rmaxT, mijT, alphaT, lijT, pT, acc1T,
      hqkData, hrm, hmijd, halphad, hpTd, hlijd, hacc1d, hmiF, hliF, haccF⟩ :=
    aft3LoopBody3_steps (s.setReg "start_n" .nat [] (Tile.scalar i)) (s0.pids 0) i (s0.pids 1)
      K V base base i i
      (⟨fun idx : TileIndex [64, 64] => some (qT idx)⟩)
      (⟨fun r : TileIndex [64] => aft3RunningMax qT kT vT keyScale3 kp i r.1 ⟨0, by norm_num⟩⟩)
      (⟨fun r : TileIndex [64] => ((aft3StateBot1 qT kT vT keyScale3 kp i r.1 ⟨0, by norm_num⟩).2.1 : ℝ)⟩)
      (⟨fun idx : TileIndex [64, 64] => ((aft3StateBot1 qT kT vT keyScale3 kp i idx.1 idx.2.1).2.2 : ℝ)⟩)
      (⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem K (base + idx.1.val * 1 + (i + idx.2.1.val) * 64))⟩)
      (⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem V (base + (i + idx.1.val) * 64 + idx.2.1.val * 1))⟩)
      sc0 _ _ _
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hMptr)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hLptr)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hOp)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsm)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoff)
      (by simp [BlockState.setReg_same])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hqs)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmi)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hli)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hacc)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; rw [hKp, hbaseEq])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; rw [hVp, hbaseEq])
      (fun idx => rfl)
      (fun idx => rfl)
      (by intro rg o; rw [BlockState.setReg_undef]; exact hundef rg o)
  refine ⟨sF, hchain, ?_⟩
  -- k/v load tiles read s0's memory (s.mem = s0.mem)
  have hrmemK : ∀ idx : TileIndex [64, 64],
      (⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem K (base + idx.1.val * 1 + (i + idx.2.1.val) * 64))⟩ : Tile .real [64, 64]).data idx
        = some (s0.readMem K (base + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 64)) := by
    intro idx
    simp only [BlockState.setReg_readMem]
    unfold BlockState.readMem; rw [hmem, hi]
  have hrmemV : ∀ idx : TileIndex [64, 64],
      (⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem V (base + (i + idx.1.val) * 64 + idx.2.1.val * 1))⟩ : Tile .real [64, 64]).data idx
        = some (s0.readMem V (base + (c * 64 + idx.1.val) * 64 + idx.2.1.val * 1)) := by
    intro idx
    simp only [BlockState.setReg_readMem]
    unfold BlockState.readMem; rw [hmem, hi]
  set qtileF : Tile .real [64, 64] := ⟨fun idx : TileIndex [64, 64] => some (qT idx)⟩ with hqtileF
  set ktileF : Tile .real [64, 64] :=
    ⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem K (base + idx.1.val * 1 + (i + idx.2.1.val) * 64))⟩ with hktileF
  set vtileF : Tile .real [64, 64] :=
    ⟨fun idx : TileIndex [64, 64] => some ((s.setReg "start_n" .nat [] (Tile.scalar i)).readMem V (base + (i + idx.1.val) * 64 + idx.2.1.val * 1))⟩ with hvtileF
  -- hsc0 = keyScale3
  have hsckey : sc0 = keyScale3 (⟨0, by norm_num⟩ : Fin 128) := by rw [hsc0, keyScale3]
  -- qkT data per cell (score), and the m_i tile cell readback
  have hmicell : ∀ r : Fin 64,
      (⟨fun r : TileIndex [64] => aft3RunningMax qT kT vT keyScale3 kp i r.1 ⟨0, by norm_num⟩⟩ : Tile .real [64]).data (r, PUnit.unit)
        = aft3RunningMax qT kT vT keyScale3 kp (c * 64) r ⟨0, by norm_num⟩ := by
    intro r; simp only; rw [hi]
  -- mijT cell = aft3RunningMax((c+1)*64)
  have hmijcell : ∀ r : Fin 64, mijT.data (r, PUnit.unit)
      = aft3RunningMax qT kT vT keyScale3 kp ((c + 1) * 64) r ⟨0, by norm_num⟩ := by
    intro r
    rw [hmijd]
    refine aft3_mij_reg_eq s0 Q K V sc0 c hc1 r ⟨0, by norm_num⟩ hsckey qtileF ktileF
      (⟨fun r : TileIndex [64] => aft3RunningMax qT kT vT keyScale3 kp i r.1 ⟨0, by norm_num⟩⟩) rmaxT qkT
      hqtileF hrmemK ?_ ?_ ?_
    · rw [hqkData]
    · rw [hmicell r]
    · exact hrm
  -- pT cell = exp2(qk - mij)
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF, BlockState.setReg_pids, hpids]
  · omega
  · omega
  · -- m_i = aft3RunningMax(i+64)
    rw [hmiF]; refine congrArg some ?_; ext r
    rw [hmijcell r.1, show ((c + 1) * 64 : Nat) = i + 64 from by omega]
  · -- l_i = aft3StateBot1(i+64).2.1
    rw [hliF]; refine congrArg some ?_; refine Tile.ext (fun r => ?_)
    obtain ⟨r, ⟨⟩⟩ := r
    have hbr := aft3_denom_reg_eq s0 Q K V sc0 c hc1 r ⟨0, by norm_num⟩ hsckey qtileF ktileF qkT
      (⟨fun r : TileIndex [64] => ((aft3StateBot1 qT kT vT keyScale3 kp i r.1 ⟨0, by norm_num⟩).2.1 : ℝ)⟩)
      (⟨fun r : TileIndex [64] => aft3RunningMax qT kT vT keyScale3 kp i r.1 ⟨0, by norm_num⟩⟩)
      mijT alphaT pT hqtileF hrmemK (by rw [hqkData]) (by simp only [hi]; rfl) (by simp only [hi]; rfl)
      (by rw [hmijcell r]) halphad hpTd
    have hne : aft3RunningMax qT kT vT keyScale3 kp ((c + 1) * 64) r ⟨0, by norm_num⟩ ≠ ⊥ :=
      aft3RunningMax_noWindow_ne_bot s0 Q K V ((c + 1) * 64) (by positivity) r ⟨0, by norm_num⟩
    show (Tile.bop NumericDType.real.add _ _ _).data _ = _
    rw [hlijd, hbr]
    show _ = ((aft3StateBot1 qT kT vT keyScale3 kp (i + 64) r ⟨0, by norm_num⟩).2.1 : WithBot ℝ)
    rw [show ((i + 64) : Nat) = (c + 1) * 64 from by omega,
      aft3StateBot1_eq_aft3StateBot qT kT vT keyScale3 kp ((c + 1) * 64) r ⟨0, by norm_num⟩ hne]
    rfl
  · -- acc = aft3StateBot1(i+64).2.2
    rw [haccF]; refine congrArg some ?_; refine Tile.ext (fun idx => ?_)
    obtain ⟨ir, id, ⟨⟩⟩ := idx
    have hbr := aft3_acc_reg_eq s0 Q K V sc0 c hc1 ir id hsckey qtileF ktileF vtileF qkT
      (⟨fun idx : TileIndex [64, 64] => ((aft3StateBot1 qT kT vT keyScale3 kp i idx.1 idx.2.1).2.2 : ℝ)⟩)
      acc1T pT
      (⟨fun r : TileIndex [64] => aft3RunningMax qT kT vT keyScale3 kp i r.1 ⟨0, by norm_num⟩⟩)
      mijT alphaT hqtileF hrmemK hrmemV (by rw [hqkData]) (by simp only [hi]; rfl) (by simp only [hi]; rfl)
      (by rw [hmijcell ir]) halphad hacc1d hpTd
    have hne : aft3RunningMax qT kT vT keyScale3 kp ((c + 1) * 64) ir id ≠ ⊥ :=
      aft3RunningMax_noWindow_ne_bot s0 Q K V ((c + 1) * 64) (by positivity) ir id
    show (Tile.bop NumericDType.real.add _ _ _).data _ = _
    rw [hbr]
    show _ = ((aft3StateBot1 qT kT vT keyScale3 kp (i + 64) ir id).2.2 : WithBot ℝ)
    rw [show ((i + 64) : Nat) = (c + 1) * 64 from by omega,
      aft3StateBot1_eq_aft3StateBot qT kT vT keyScale3 kp ((c + 1) * 64) ir id hne]
    rfl
  · rw [hqF]
  · rw [hqsF]
  · rw [hsmF]
  · rw [hoffF]
  · -- K_block_ptr advanced: kcol = i, base = baseOffset3 s0
    rw [hKpF, ← hbaseEq]
  · rw [hVpF, ← hbaseEq]
  · rw [hMptrF]
  · rw [hLptrF]
  · rw [hOpF]
  · exact hundefF
  · rw [hmemF]; funext rg o; rw [BlockState.setReg_mem]; exact congrFun (congrFun hmem rg) o

/-! ### postLoop evaluation (case 3, no window) -/

/-- A `foldl` of `writeMem`s into region `R'` preserves `readMem R o` whenever the
read region `R` differs from the written region `R'`. -/
theorem aft3_foldl_writeMem_readMem_other_region {α : Type} (R R' : RegionName)
    (hRR' : R ≠ R') (offsetFn : α → Nat) (valueFn : α → ℝ) (o : Nat) (l : List α)
    (s : BlockState) :
    ((l.foldl (fun acc k => acc.writeMem R' (offsetFn k) (valueFn k)) s).readMem R o)
      = s.readMem R o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
    rw [List.foldl_cons, ih]
    exact BlockState.writeMem_readMem_of_ne_region s R' (offsetFn hd) (valueFn hd) R o hRR'

/-- Injectivity of the case-3 `O` block-pointer store addresses (Python shape). -/
theorem aft3_O_blockptr_offset_injective (base p0 : Nat) :
    Function.Injective
      (fun idx : TileIndex [64, 64] => base + (p0 * 64 + idx.1.val) * 64 + idx.2.1.val * 1) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp only at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb; subst db; rfl

/-- Injectivity of the case-3 `M` row store addresses (Python shape). -/
theorem aft3_M_ptr_offset_injective (p0 p1 : Nat) :
    Function.Injective
      (fun idx : TileIndex [64] => p1 * 128 + (p0 * 64 + idx.1.val)) := by
  rintro ⟨⟨a, ha⟩, _⟩ ⟨⟨b, hb⟩, _⟩ h
  simp only at h
  have : a = b := by omega
  subst b; rfl

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **PostLoop execution + genuine readbacks (case 3, no window).** Stepping the
3 post-loop statements (END finalize + M-row store + O block-ptr store) from a
loop-end state satisfying `attnInvariant … noWindowKeep 128`, the kernel's `O`
store holds the genuine closed-form attention ratio
(`attentionFwdTriton3Case3OutSpec`) at every injective output lane, and the `M`
store holds the genuine finalize value `m_i + log2 l_i` (running max + log2 of the
denominator) at every injective row lane. -/
theorem aft3PostLoop_eval
    (Q K V M Out L : RegionName) (s0 : BlockState) (s : BlockState)
    (hMO : M ≠ Out)
    (hinv : attnInvariant Q K V M Out L s0 (fun i j => noWindowKeep i j) 128 s) :
    ∃ sP, stepStmts (aft3PostLoop M Out L) s = some sP
      ∧ (∀ idx : TileIndex [64, 64],
          sP.readMem Out (outOffset s0 4 32768 8192 64 1 64 idx)
            = attentionFwdTriton3Case3OutSpec s0 Q K V idx)
      ∧ (∀ i : Fin 64,
          sP.readMem M (lRowOffset s0 (s0.pids 1) 128 64 i)
            = (aft3RunningMax (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
                  (fun i j => noWindowKeep i j) 128 i ⟨0, by norm_num⟩).unbotD 0
              + Real.log
                ((aft3StateBot1 (qTile3 s0 Q) (kTile3 s0 K) (vTile3 s0 V) keyScale3
                    (fun i j => noWindowKeep i j) 128 i ⟨0, by norm_num⟩).2.1) / Real.log 2) := by
  simp only [attnInvariant] at hinv
  obtain ⟨hpids, _, _, hmi, hli, hacc, hq, hqs, hsm, hoff, hKp, hVp, hMptr, hLptr, hOp, hundef, hmem⟩ :=
    hinv
  set qT := qTile3 s0 Q with hqT
  set kT := kTile3 s0 K with hkT
  set vT := vTile3 s0 V with hvT
  set kp : Fin 64 → Fin 128 → Prop := fun i j => noWindowKeep i j with hkp
  -- register tiles at loop end
  set miTile : Tile .real [64] :=
    ⟨fun r : TileIndex [64] => aft3RunningMax qT kT vT keyScale3 kp 128 r.1 ⟨0, by norm_num⟩⟩
    with hmiTile
  set liTile : Tile .real [64] :=
    ⟨fun r : TileIndex [64] => ((aft3StateBot1 qT kT vT keyScale3 kp 128 r.1 ⟨0, by norm_num⟩).2.1 : ℝ)⟩
    with hliTile
  set accTile : Tile .real [64, 64] :=
    ⟨fun idx : TileIndex [64, 64] => ((aft3StateBot1 qT kT vT keyScale3 kp 128 idx.1 idx.2.1).2.2 : ℝ)⟩
    with haccTile
  unfold aft3PostLoop
  -- stmt 0: END ifThenElse (constexpr true): m_i += log2 l_i ; acc = acc / l_i[:, None]
  -- finalized m_i tile
  set miFin : Tile .real [64] :=
    Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil) miTile
      (Tile.uop WithBot.realLog2 liTile) with hmiFin
  set accFin : Tile .real [64, 64] :=
    Tile.bop NumericDType.real.div (Broadcast.consSame (Broadcast.consR Broadcast.nil)) accTile
      (Tile.expandDim ⟨1, by simp⟩ liTile) with haccFin
  rw [stepStmts.cons_some
    (show stepStmt _ s = some _ from by
      rw [aft3_ifThenElse_true (aft3_ne_one_zero_true s)]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.add NumericDType.real (Broadcast.consSame Broadcast.nil)
            (Op.ref TileDType.real [64] "m_i") (Op.ref TileDType.real [64] "l_i").log2) s
            = some miFin from by
          rw [evalOp_add]
          simp only [evalOp, evalOp.eq_def, evalOp_ref, hmi, hli, Option.bind_eq_bind,
            Option.bind_some]
          rfl))]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.div NumericDType.real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.ref TileDType.real [64, 64] "acc")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.real [64] "l_i"))) _
            = some accFin from by
          have hexp : @evalOp TileDType.real [64, 1]
                (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.real [64] "l_i"))
                (s.setReg "m_i" .real [64] miFin)
              = some (Tile.expandDim ⟨1, by simp⟩ liTile) :=
            evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hli)
          rw [evalOp_div]
          simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true, BlockState.setReg_same, hexp, hacc, Option.bind_eq_bind,
            Option.bind_some]
          rfl))]
      rw [stepStmts.nil])]
  -- state after the END finalize
  set s2 := (s.setReg "m_i" .real [64] miFin).setReg "acc" .real [64, 64] accFin with hs2
  -- m_ptrs / l_ptrs readbacks in s2
  have hMptr2 : s2.regs .ptr [64] "m_ptrs" = some
      (Tile.ptrAdd Broadcast.scalarL (Tile.scalar (M.cast, (0 : Nat)))
        (Tile.bop NumericDType.nat.add Broadcast.scalarL (Tile.scalar (s0.pids 1 * 128))
          (Tile.vec (fun r : Fin 64 => s0.pids 0 * 64 + r.val)))) := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hMptr
  have hmi2 : s2.regs .real [64] "m_i" = some miFin := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_same]
  -- M store address function and value function
  set mOffFn : TileIndex [64] → Nat :=
    fun r => s0.pids 1 * 128 + (s0.pids 0 * 64 + r.1.val) with hmOffFn
  set mValTile : Tile .real [64] := miFin with hmValTile
  -- stmt 1: store M via m_ptrs (m_i = miFin)
  have hmptrEval : evalOp (Op.ref TileDType.ptr [64] "m_ptrs") s2
      = some (⟨fun r : TileIndex [64] => (M.cast, mOffFn r)⟩ : Tile .ptr [64]) := by
    rw [evalOp_ref, hMptr2]
    refine congrArg some ?_; ext r
    · rfl
    · simp only [Tile.ptrAdd_data, Tile.scalar_data, Tile.bop_data, Tile.vec_data,
        Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL, Broadcast.leftIndex_nil,
        Broadcast.rightIndex_nil, NumericDType.add, Nat.zero_add, hmOffFn]
  have hstore1 : stepStmt (Stmt.store TileDType.real [64] (MemAccess.ptr (Op.ref TileDType.ptr [64] "m_ptrs"))
      (Op.ref TileDType.real [64] "m_i") MaskOpt.none) s2
      = some ((TileShape.allIndices [64]).foldl
          (fun acc r => acc.writeMemTyped .real M (mOffFn r) (miFin.data r)) s2) := by
    simp only [stepStmt, evalOp_ref, hmi2, hmptrEval, Option.bind_eq_bind, Option.bind_some,
      Option.map_some, if_true, Region.cast_id, mValTile]
  rw [stepStmts.cons_some hstore1]
  set s3 := (TileShape.allIndices [64]).foldl
      (fun acc r => acc.writeMemTyped .real M (mOffFn r) (miFin.data r)) s2 with hs3
  -- O_block_ptr / acc readbacks in s3 (stores only touch memory)
  have hOp3 : s3.regs .blockPtr [64, 64] "O_block_ptr" = some
      (⟨fun _ : TileIndex [64, 64] =>
        { region := Out, baseOffset := s0.pids 1 / 4 * 32768 + s0.pids 1 % 4 * 8192,
          parentShape := [128, 64], blockShape := [64, 64], strides := [64, 1],
          offsets := [s0.pids 0 * 64, 0] }⟩) := by
    rw [hs3]
    simp only [BlockState.foldl_writeMemTyped_regs]
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hOp
  have hacc3 : s3.regs .real [64, 64] "acc" = some accFin := by
    rw [hs3]
    simp only [BlockState.foldl_writeMemTyped_regs]
    rw [hs2, BlockState.setReg_same]
  -- O store address / value functions
  set oOffFn : TileIndex [64, 64] → Nat :=
    fun idx => s0.pids 1 / 4 * 32768 + s0.pids 1 % 4 * 8192 + (s0.pids 0 * 64 + idx.1.val) * 64 + idx.2.1.val * 1
    with hoOffFn
  set oValFn : TileIndex [64, 64] → WithBot ℝ := fun idx => accFin.data idx with hoValFn
  -- stmt 2: store O via O_block_ptr (acc = accFin)
  have hOpref : @evalOp TileDType.blockPtr [64, 64] (Op.ref .blockPtr [64, 64] "O_block_ptr") s3
      = some (⟨fun _ : TileIndex [64, 64] =>
          { region := Out, baseOffset := s0.pids 1 / 4 * 32768 + s0.pids 1 % 4 * 8192,
            parentShape := [128, 64], blockShape := [64, 64], strides := [64, 1],
            offsets := [s0.pids 0 * 64, 0] }⟩
          : Tile .blockPtr [64, 64]) := by rw [evalOp_ref]; exact hOp3
  have hstore2 : stepStmt (Stmt.store TileDType.real [64, 64]
      (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "O_block_ptr") [])
      (Op.ref TileDType.real [64, 64] "acc") MaskOpt.none) s3
      = some ((TileShape.allIndices [64, 64]).foldl
          (fun acc idx => acc.writeMemTyped .real Out (oOffFn idx) (oValFn idx)) s3) := by
    simp only [stepStmt, evalOp_ref, hacc3, hOp3, Option.bind_eq_bind, Option.bind_some,
      Option.map_some]
    refine congrArg some ?_
    refine List.foldl_ext _ _ s3 ?_
    intro acc idx _
    simp only [TileShape.indexToList, BlockPtr.inBounds, List.all_nil, Bool.and_true,
      Bool.true_and, if_true]
    have haddr : BlockPtr.address
        { region := Out, baseOffset := s0.pids 1 / 4 * 32768 + s0.pids 1 % 4 * 8192,
          parentShape := [128, 64], blockShape := [64, 64], strides := [64, 1],
          offsets := [s0.pids 0 * 64, 0] }
        [idx.1.val, idx.2.1.val]
        = oOffFn idx := by
      show s0.pids 1 / 4 * 32768 + s0.pids 1 % 4 * 8192 +
          ((s0.pids 0 * 64 + idx.1.val) * 64 + (0 + idx.2.1.val) * 1) = _
      rw [Nat.zero_add, hoOffFn]; ring
    rw [haddr]
  rw [stepStmts.cons_some hstore2, stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_⟩
  · -- O readback: genuine attention ratio
    intro idx
    have hinjO : Function.Injective oOffFn := by
      rw [hoOffFn]; exact aft3_O_blockptr_offset_injective _ (s0.pids 0)
    -- outOffset = oOffFn
    have houtOff : outOffset s0 4 32768 8192 64 1 64 idx = oOffFn idx := by
      simp only [outOffset, offZ, offH, mIndex, kIndex, hoOffFn]
    rw [houtOff]
    -- O store readback (writeMemTyped real → writeMem with storeValue)
    simp only [BlockState.writeMemTyped_real]
    rw [BlockState.scatter_readback_nd (region := Out) s3 oOffFn
      (fun idx => FloatDType.real.storeValue (oValFn idx)) hinjO idx]
    -- decode: accFin / liFin = StateBot ratio = spec
    simp only [FloatDType.real_storeValue, hoValFn, haccFin]
    obtain ⟨ir, id, ⟨⟩⟩ := idx
    have hne : aft3RunningMax qT kT vT keyScale3 kp 128 ir id ≠ ⊥ :=
      aft3RunningMax_noWindow_ne_bot s0 Q K V 128 (by norm_num) ir id
    have hne' : aft3RunningMax qT kT vT keyScale3 kp 128 ir ⟨0, by norm_num⟩ ≠ ⊥ :=
      aft3RunningMax_noWindow_ne_bot s0 Q K V 128 (by norm_num) ir ⟨0, by norm_num⟩
    -- accTile / liTile cell: (accFin.data).unbotD 0 = accTile/liTile real ratio
    simp only [Tile.bop_data, Tile.expandDim_data, TileShape.dropInsertedIndex,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div, WithBot.realDiv,
      Option.map₂, Option.bind, Option.map, haccTile, hliTile, WithBot.unbotD_coe]
    rw [aft3StateBot1_eq_aft3StateBot qT kT vT keyScale3 kp 128 ir id hne,
      aft3StateBot1_eq_aft3StateBot qT kT vT keyScale3 kp 128 ir ⟨0, by norm_num⟩ hne']
    rw [show ((aft3StateBot qT kT vT keyScale3 kp 128 ir ⟨0, by norm_num⟩).2.1 : ℝ)
          = (aft3StateBot qT kT vT keyScale3 kp 128 ir id).2.1 from
        aft3StateBot_snd_fst_indep qT kT vT keyScale3 kp 128 ir ⟨0, by norm_num⟩ id]
    rw [← aft3StateBot_full_eq_spec_case3 s0 Q K V ir id hne]
    rfl
  · -- M readback: genuine finalize value (running max + log2 denom)
    intro i
    have hinjM : Function.Injective mOffFn := by
      rw [hmOffFn]; exact aft3_M_ptr_offset_injective (s0.pids 0) (s0.pids 1)
    -- the O store (region Out) and O_block_ptr setReg preserve readMem M
    rw [show (lRowOffset s0 (s0.pids 1) 128 64 i) = mOffFn (i, PUnit.unit) from by
      simp only [lRowOffset, hmOffFn]]
    rw [show ((TileShape.allIndices [64, 64]).foldl
            (fun acc idx => acc.writeMemTyped .real Out (oOffFn idx) (oValFn idx)) s3).readMem M
              (mOffFn (i, PUnit.unit))
          = s3.readMem M (mOffFn (i, PUnit.unit)) from by
      simp only [BlockState.writeMemTyped_real]
      exact aft3_foldl_writeMem_readMem_other_region M Out hMO oOffFn
        (fun idx => FloatDType.real.storeValue (oValFn idx)) (mOffFn (i, PUnit.unit))
        (TileShape.allIndices [64, 64]) s3]
    -- s3 is the M scatter
    rw [hs3]
    simp only [BlockState.writeMemTyped_real]
    rw [BlockState.scatter_readback_nd (region := M) s2 mOffFn
      (fun r => FloatDType.real.storeValue (miFin.data r)) hinjM (i, PUnit.unit)]
    -- decode miFin cell = running max + log2 denom
    simp only [FloatDType.real_storeValue, miFin, Tile.bop_data, Tile.uop_data,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, hmiTile, hliTile]
    have hne : aft3RunningMax qT kT vT keyScale3 kp 128 i ⟨0, by norm_num⟩ ≠ ⊥ :=
      aft3RunningMax_noWindow_ne_bot s0 Q K V 128 (by norm_num) i ⟨0, by norm_num⟩
    obtain ⟨mr, hmr⟩ := WithBot.ne_bot_iff_exists.mp hne
    rw [← hmr]
    rw [show ((aft3StateBot1 qT kT vT keyScale3 kp 128 i ⟨0, by norm_num⟩).2.1 : WithBot ℝ)
          = some ((aft3StateBot1 qT kT vT keyScale3 kp 128 i ⟨0, by norm_num⟩).2.1) from rfl]
    rw [WithBot.realLog2_some]
    simp only [WithBot.realAdd, Option.map₂, Option.bind, Option.map, WithBot.unbotD_coe]
    rfl

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Full kernel execution (case 3, no window).** The lowered case-3 surface body
steps a clean state (`undef = 0`) through preLoop + the `forRange` streaming loop
(via `forRange_inv` with `attnInvariant noWindowKeep` as the loop invariant,
advanced by `aft3_attn_step3`) + postLoop, leaving the `O` and `M` writebacks at
their genuine closed forms. -/
theorem aft3_attn_exec
    (Q K V M Out L : RegionName) (s : BlockState)
    (hMO : M ≠ Out) (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sF, stepStmts (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
        32768 8192 64 1 2 4 4 128 128 128 0 0 1 1 64 64 64 1 1 0 0).toAlgKernel.body s = some sF
      ∧ (∀ idx : TileIndex [64, 64],
          sF.readMem Out (outOffset s 4 32768 8192 64 1 64 idx)
            = attentionFwdTriton3Case3OutSpec s Q K V idx)
      ∧ (∀ i : Fin 64,
          sF.readMem M (lRowOffset s (s.pids 1) 128 64 i)
            = (aft3RunningMax (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3
                  (fun i j => noWindowKeep i j) 128 i ⟨0, by norm_num⟩).unbotD 0
              + Real.log
                ((aft3StateBot1 (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3
                    (fun i j => noWindowKeep i j) 128 i ⟨0, by norm_num⟩).2.1) / Real.log 2) := by
  rw [aft3_body_split_case3]
  -- preLoop
  obtain ⟨sp, hpre, hinv0⟩ :=
    aft3PreLoop_eval Q K V M Out L s (fun i j => noWindowKeep i j) hundef
  rw [stepStmts.append_some hpre]
  -- the forRange streaming loop via forRange_inv with P = attnInvariant
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    forRange_inv (idx := "start_n") (start := 0) (stop := 128) (step := 64)
      (body := aft3LoopBody3)
      (P := fun i st => attnInvariant Q K V M Out L s (fun i j => noWindowKeep i j) i st)
      (s_init := sp)
      (by norm_num)
      hinv0
      (fun i st hi hP => aft3_attn_step3 Q K V M Out L s i st hi
        (by obtain ⟨_, hmod, _, _⟩ := hP; exact hmod) hP)
  rw [stepStmts.cons_some hloop]
  -- at loop exit, counter `final` is the first multiple of 64 ≥ 128, i.e. 128
  have hfinal : final = 128 := by
    obtain ⟨_, hmod, hle, _⟩ := hinvL
    omega
  subst hfinal
  -- postLoop
  obtain ⟨sF, hpost, hO, hM⟩ :=
    aft3PostLoop_eval Q K V M Out L s sL hMO hinvL
  refine ⟨sF, hpost, hO, hM⟩

/-! ## Case 3 genuine closed-form output summary

The case-3 `Out`/`M` writebacks at the Python test shape equal their genuine
closed forms (the base-2 per-key-scale softmax `attentionFwdTriton3Case3OutSpec`
for `Out`, the running-max + log2-denominator finalize for `M`) — no longer the
self-referential "executed kernel output" carrier. This requires a clean
(`undef = 0`) input and `M ≠ Out` (so the `O` store does not clobber the `M`
row), both genuine preconditions of single-program correctness. -/

/-- Genuine `M`-row spec (case 3): `m_i + log2 l_i` finalize at full window. -/
noncomputable def attentionFwdTriton3Case3MSpec
    (s : BlockState) (Q K V : RegionName) (i : Fin 64) : ℝ :=
  (aft3RunningMax (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3
      (fun i j => noWindowKeep i j) 128 i ⟨0, by norm_num⟩).unbotD 0
    + Real.log
      ((aft3StateBot1 (qTile3 s Q) (kTile3 s K) (vTile3 s V) keyScale3
          (fun i j => noWindowKeep i j) 128 i ⟨0, by norm_num⟩).2.1) / Real.log 2

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- Genuine case-3 `Out`-store correctness: the masked `Out` writeback realizes the
closed-form attention ratio at every active lane. -/
theorem attention_fwd_triton3_case3_surface_out_compute_correct
    (Q K V M Out L : RegionName) (s : BlockState)
    (hMO : M ≠ Out) (hundef : ∀ rg o, s.undef rg o = 0) :
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
        attentionFwdTriton3Case3OutSpec s Q K V idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  obtain ⟨sF, hstep, hO, _⟩ := aft3_attn_exec Q K V M Out L s hMO hundef
  rw [exec] at hExec
  rw [hstep] at hExec
  obtain rfl : sF = s' := Option.some.inj hExec
  exact hO idx

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- Genuine case-3 `M`-store correctness: the `M`-row writeback realizes the
`m_i + log2 l_i` finalize at every row. -/
theorem attention_fwd_triton3_case3_surface_m_compute_correct
    (Q K V M Out L : RegionName) (s : BlockState)
    (hMO : M ≠ Out) (hundef : ∀ rg o, s.undef rg o = 0) :
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
        attentionFwdTriton3Case3MSpec s Q K V i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  obtain ⟨sF, hstep, _, hM⟩ := aft3_attn_exec Q K V M Out L s hMO hundef
  rw [exec] at hExec
  rw [hstep] at hExec
  obtain rfl : sF = s' := Option.some.inj hExec
  exact hM i

/-- **Case 3 genuine output summary.** The case-3 surface lowers to the algorithm
layer, and its `Out`/`M` writebacks realize the genuine closed forms
(`attentionFwdTriton3Case3OutSpec` / `attentionFwdTriton3Case3MSpec`) on clean
(`undef = 0`) input with `M ≠ Out`. -/
theorem attention_fwd_triton3_python_case3_output_summary
    (Q K V M Out L : RegionName) (s : BlockState)
    (hMO : M ≠ Out) (hundef : ∀ rg o, s.undef rg o = 0) :
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
        attentionFwdTriton3Case3OutSpec s Q K V idx) ∧
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
        attentionFwdTriton3Case3MSpec s Q K V i) := by
  refine ⟨?_, ?_, ?_⟩
  · exact attention_fwd_triton3_python_case3_surface_toAlgorithm_supported
      Q K V M Out L
  · exact attention_fwd_triton3_case3_surface_out_compute_correct Q K V M Out L s hMO hundef
  · exact attention_fwd_triton3_case3_surface_m_compute_correct Q K V M Out L s hMO hundef

end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
