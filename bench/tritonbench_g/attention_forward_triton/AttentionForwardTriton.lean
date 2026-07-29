import VeriTile.Triton
import VeriTile.Examples.AttentionForwardClosedForm

/-!
# `attention_forward_triton` — strict per-kernel correctness

`_attn_fwd` is a flash-attention forward kernel: program `(start_m, off_hz)`
loads a `BLOCK_M`-row tile of `Q` for one (batch, head), then over the key/value
context (`_attn_fwd_inner`, stepping by `BLOCK_N`) runs the online-softmax
recurrence — block scores `qk = q·k · q_scale · k_scale`, running max `m_i`,
rescaled denominator `l_i`, and accumulator `acc` updated with `exp2(qk - m_ij)`
weights — and finally stores `acc / l_i` to `Out`, masked to the first 96 head
lanes.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_attn_fwd[grid](...)`, the grid over
`(cdiv(N_CTX, BLOCK_M), Z·H)`, block scheduling, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because `start_m`/`off_hz` are universally quantified, the
per-program statement covers every program of the grid.

## Proof architecture

```
attention_forward_triton_closed_form_correct                  ← TOP THEOREM (genuine closed form)
  expected = attentionRealBase2PerKeyScale (qTile) (kTile) (vTile) (keyScale)
  └─ (online-softmax recurrence == batch base-2 softmax, Math/Attention.lean)
     (the surface lowers to the algorithm layer inline via `rfl`, so the standalone
      `attention_forward_triton_surface_toAlgorithm_supported` lemma is not invoked here)

attention_forward_triton_final_store_slice_compute_correct    ← ComputeCorrect over the masked Out store
  └─ attention_forward_triton_final_store_slice_correct        algorithm-layer readback per lane
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `float16`/`float32`
casts collapse to the identity post-erasure; the launch-time
`num_warps`/`num_stages` are not modeled. Cross-program composition into the
full `[Z,H,N_CTX,HEAD_DIM]` output is the trusted host boundary.

## Top theorem: closed-form value (NOT self-referential)

`attention_forward_triton_closed_form_correct` is a **genuine closed-form value
claim**: every active output lane of `Out` equals
`VeriTile.Triton.attentionRealBase2PerKeyScale` of the loaded Q/K/V tiles under
the per-block key scale — i.e. the base-2, per-key-scaled attention output, not
the kernel's own executed value. It is **general**: arbitrary batch/head strides,
head count `H`, block sizes, KV-block count (`N_CTX = BLOCK_N · numKVBlocks`),
head/active dimensions, and arbitrary `q_scale`/`k_scale`. The only layout
assumptions are the contiguity contracts the kernel relies on
(`stride_qm = stride_kn = HEAD_DIM`, head stride `1`). The Python test case
(`B=2, H=4, N_CTX=128, HEAD_DIM=128, BLOCK_M=128, BLOCK_N=64, HEAD_ACTIVE=96`) is
the special case.

The mathematical heart — online-softmax recurrence == batch base-2 softmax —
is proved sorry-free in `VeriTile/Triton/Math/Attention.lean`
(`attentionRealBase2PerKeyScale_eq_streaming`, `osBlockStep_foldl_eq_batch`), and
the full `exec`-side loop unfolding (Phase 3) is complete in
`VeriTile/Examples/AttentionForwardClosedForm.lean`. The top theorem here is now
**proved sorry-free** by bridging to that result (this kernel is tracked as
`attention-forward-online-softmax-recurrence`, #162).
-/

namespace VeriTile.Bench.TritonBenchG.AttentionForwardTriton

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Full Lean port of `attention_forward_triton.py`'s `_attn_fwd`. -/
def attention_forward_triton_helper_call_surface
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX _HEAD_DIM BLOCK_M BLOCK_N _STAGE : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  qvk_offset = off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh)
  vk_offset = qvk_offset // $(stride_qm)
  q_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_M))
  k_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_N))

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_k = tl.arange(0, 128)
  Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  Q_scale_ptr = Q_scale + q_scale_offset + start_m
  K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * $(stride_kn)
  K_scale_ptr = K_scale + k_scale_offset
  V_ptrs = V + qvk_offset + offs_n[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  O_block_ptr = Out + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
  acc = tl.zeros([$(BLOCK_M), 128], dtype=tl.float32)
  q = tl.load(Q_ptrs, mask=((offs_m[:, None] < $(N_CTX)) & ((tl.arange(0, 128) < 96)[None, :])))
  q_scale = tl.load(Q_scale_ptr)
  acc, l_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs,
    start_m, $(BLOCK_M), $(_HEAD_DIM), $(BLOCK_N), $(4 - _STAGE), offs_m, offs_n, $(N_CTX))
  acc = acc / l_i[:, None]
  tl.store(O_block_ptr, (acc).to(Out.type.element_ty), mask=((offs_m[:, None] < $(N_CTX)) & ((tl.arange(0, 128) < 96)[None, :])))
}

/-- The helper-call shaped attention-forward surface lowers to the algorithm
layer. -/
theorem attention_forward_triton_helper_call_surface_toAlgorithm_supported
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N STAGE : Nat) :
    ∃ alg, (attention_forward_triton_helper_call_surface Q K V Q_scale K_scale
      Out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh
      stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz
      stride_oh stride_om stride_on Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N
      STAGE).toAlgorithm? = Except.ok alg := by
  simp [attention_forward_triton_helper_call_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Full Lean port of `attention_forward_triton.py`'s `_attn_fwd`.

The upstream kernel calls a separate `@triton.jit` helper `_attn_fwd_inner` to
run the K/V streaming-softmax loop. The DSL has no function-call surface, so the
helper body is inlined verbatim into the outer kernel; semantically the two
forms are identical for this fixed-stage path.

The literal `128` and `96` in the upstream kernel correspond to the
`BLOCK_DMODEL` / `HEAD_ACTIVE` parameters threaded through the bundled tests
(`head_dim = 128`, with the inner dot using only the first 96 lanes of the head
dimension). They appear here as explicit Lean parameters. -/
def attention_forward_triton_surface
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE _STAGE : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  qvk_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  vk_offset = qvk_offset // $(stride_qm)
  q_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_M))
  k_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_N))

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  Q_scale_ptr = Q_scale + q_scale_offset + start_m
  K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * $(stride_kn)
  K_scale_ptr = K_scale + k_scale_offset
  V_ptrs = V + qvk_offset + offs_n[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  O_block_ptr = Out + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(Q_ptrs,
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
  q_scale = tl.load(Q_scale_ptr)
  for start_n in range(0, $(N_CTX), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs, mask=k_mask)
    k_scale = tl.load(K_scale_ptr)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, v, out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr += $(1)
    V_ptrs += $(BLOCK_N) * $(HEAD_DIM)
  }
  acc = acc / l_i[:, None]
  tl.store(O_block_ptr, (acc).to(Out.type.element_ty),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}

/-- The full inlined attention-forward surface lowers to the algorithm layer. -/
theorem attention_forward_triton_surface_toAlgorithm_supported
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat) :
    ∃ alg, (attention_forward_triton_surface Q K V Q_scale K_scale Out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn
      stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh
      stride_om stride_on Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE STAGE).toAlgorithm? = Except.ok alg := by
  simp [attention_forward_triton_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of `attention_forward_triton.py`'s
`_attn_fwd`.

The full kernel computes a tiled attention accumulator with Q/K/V block loads,
quantization scales, and a streaming softmax reduction. This slice starts after
`acc = acc / l_i[:, None]` with a precomputed `Acc` tile and proves the final
masked writeback into `Out`. It preserves the source program-id decomposition
and the source store mask `(offs_m < N_CTX) & (offs_k < 96)`. The source forms
`O_block_ptr` with the Q strides, so this slice names those as the output store
strides. The inner `tl.float32` accumulator and `p.to(tl.float16)` dot-input
cast are outside this slice. -/
def attention_forward_triton_final_store_slice
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

/-! ## Closed-form spec inputs (loaded tiles, per-key scale)

The genuine `expected` for the top theorem is `attentionRealBase2PerKeyScale` of
the loaded Q/K/V tiles under the per-block key scale, mirroring
`VeriTile/Examples/AttentionForwardClosedForm.lean`. Under the contiguity
contracts `stride_qm = stride_kn = HEAD_DIM`, head stride `1`, every loaded
element sits at `base + row · HEAD_DIM + col`; masked-off head lanes load `0`,
so summing over the `HEAD_ACTIVE` active lanes is the full contraction. -/

/-- Batch/head base offset `off_z · stride_qz + off_h · stride_qh`. -/
def baseOffset (s : BlockState) (H stride_qz stride_qh : Nat) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh

noncomputable def qTile (s : BlockState) (Q : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE : Nat) :
    TileIndex [BLOCK_M, HEAD_ACTIVE] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q (baseOffset s H stride_qz stride_qh + mIndex s BLOCK_M i * HEAD_DIM + e.val)

noncomputable def kTile (s : BlockState) (K : RegionName)
    (H stride_qz stride_qh HEAD_DIM S HEAD_ACTIVE : Nat) :
    TileIndex [S, HEAD_ACTIVE] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (baseOffset s H stride_qz stride_qh + j.val * HEAD_DIM + e.val)

noncomputable def vTile (s : BlockState) (V : RegionName)
    (H stride_qz stride_qh HEAD_DIM S HEAD_ACTIVE : Nat) :
    TileIndex [S, HEAD_ACTIVE] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (baseOffset s H stride_qz stride_qh + j.val * HEAD_DIM + d.val)

/-- Per-key scale `q_scale · k_scale[block(j)]`, `block(j) = j / BLOCK_N`.
`q_scale` is read at `off_hz · cdiv(N_CTX, BLOCK_M) + pid₀`; `k_scale[b]` at
`off_hz · cdiv(N_CTX, BLOCK_N) + b`. -/
noncomputable def keyScale (s : BlockState) (Q_scale K_scale : RegionName)
    (N_CTX BLOCK_M BLOCK_N S : Nat) :
    Fin S → ℝ :=
  fun j =>
    s.readMem Q_scale (s.pids 1 * cdiv N_CTX BLOCK_M + s.pids 0) *
      s.readMem K_scale (s.pids 1 * cdiv N_CTX BLOCK_N + j.val / BLOCK_N)

/-- Algorithm-layer correctness for the attention-forward final output store. -/
theorem attention_forward_triton_final_store_slice_correct
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
      (exec (attention_forward_triton_final_store_slice Acc Out H N_CTX
            HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k
            stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (if active s N_CTX HEAD_ACTIVE BLOCK_M idx then
            s.readMem Acc
              (accOffset s H stride_acc_z stride_acc_h stride_acc_m
                stride_acc_k BLOCK_M idx)
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, attention_forward_triton_final_store_slice, stepStmts, stepStmt,
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

/-- Compute-facing correctness for the attention-forward final output store. -/
theorem attention_forward_triton_final_store_slice_compute_correct
    (Acc Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_forward_triton_final_store_slice Acc Out H N_CTX
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
  · simp [attention_forward_triton_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := attention_forward_triton_final_store_slice_correct Acc Out H N_CTX
    HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k stride_qz
    stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- **Closed-form correctness for `attention_forward_triton` (general statement).**

For arbitrary batch/head strides, head count, block sizes, KV-block count,
head/active dimensions and arbitrary `q_scale`/`k_scale`, every active output
lane of `Out` (`mIndex < N_CTX ∧ head < HEAD_ACTIVE`) equals
`attentionRealBase2PerKeyScale` of the loaded Q/K/V tiles under the per-block key
scale — the genuine base-2, per-key-scaled attention output, NOT the kernel's own
executed value. Inactive lanes are unconstrained (masked out by the write map).

Layout contracts: `N_CTX = BLOCK_N · numKVBlocks`, `stride_qm = stride_kn =
HEAD_DIM` and head stride `1` (so the per-block pointer advance composes into a
per-key address), `0 < BLOCK_N`, `HEAD_ACTIVE ≤ BLOCK_DMODEL`. The Python test
case (`B=2, H=4, N_CTX=128, HEAD_DIM=128, BLOCK_M=128, BLOCK_N=64,
HEAD_ACTIVE=96`, `q_scale = k_scale = 1`) is the special case.

**Proven sorry-free**: bridges (via `realizes_writeIf_iff` +
`computeCorrect_of_toAlgKernel`) to
`VeriTile.Examples.AttentionForwardClosedForm.attention_forward_triton_closed_form_correct`,
whose full `exec`-side loop unfolding (preLoop + per-block step + postLoop) and
math core (`Math/Attention.lean`) are both complete. Extra preconditions:
`HEAD_ACTIVE ≤ HEAD_DIM` (store-offset injectivity), clean initial `undef`.
Tracked as `attention-forward-online-softmax-recurrence`, #162. -/
specification attention_forward_triton_closed_form_correct
    (Q K V Q_scale K_scale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks
      HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
    (hBN : 0 < BLOCK_N) (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL)
    (hHD : HEAD_ACTIVE ≤ HEAD_DIM) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_forward_triton_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s (BLOCK_N * numKVBlocks) HEAD_ACTIVE BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        if h : idx.2.1.val < HEAD_ACTIVE then
          attentionRealBase2PerKeyScale
            (qTile s Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE)
            (kTile s K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (vTile s V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (keyScale s Q_scale K_scale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N
              (BLOCK_N * numKVBlocks))
            (idx.1, ⟨idx.2.1.val, h⟩, PUnit.unit)
        else (0 : ℝ)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst hs0
  intro idx hActive
  obtain ⟨hm, hk⟩ := hActive
  have hmain := VeriTile.Examples.AttentionForwardClosedForm.attention_forward_triton_closed_form_correct
    Q K V Q_scale K_scale Out s0 stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM
    BLOCK_DMODEL HEAD_ACTIVE STAGE hBN hActiveLe hHD hundef idx ⟨hm, hk⟩
  have hExec2 : exec (VeriTile.Examples.AttentionForwardClosedForm.attention_forward_triton_surface
      Q K V Q_scale K_scale Out stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE) s0
      = some s' := hExec
  rw [hExec2] at hmain
  simp only [ComputeCorrect.OutputReadable.read_real]
  rw [dif_pos (show idx.2.1.val < HEAD_ACTIVE from hk)]
  exact hmain

/-! # ══════════ The `⊨[R]` io headline — `StreamMasked3DKernelIO₅` ══════════

The five-input skin face of `_attn_fwd`: three streamed 2-D tile channels
(`Q` static, `K` transposed / `V` advanced `BLOCK_N` rows per step) plus two
**scalar-width** quantization-scale channels (`Q_scale` static, `K_scale`
advancing one slot per step) folding into the single masked terminal `Out`
store. Everything below is purely additive; the exact surfaces above are
untouched.

Honest boundaries specific to this kernel:

* **In-loop fp16 round-trip.** Loop-body statement 14 is
  `p = (p).to(tl.float16)` — an *in-loop* rounding event, outside the skin's
  single-boundary-round shape. The headline therefore pins
  `R.round .fp16 = id`, exactly the file's declared fp16 modeling boundary
  (the exact stack above already treats this cast as the identity). Under that
  hypothesis every rounding site in the kernel is the identity and the lowered
  body is cast-free under `execR R`, so the exact
  `preLoop → attn_step → attn_postLoop` stack of
  `VeriTile/Examples/AttentionForwardClosedForm.lean` is reused unchanged.
* **`.real` terminal grid.** The lowered store is `.real`-typed — the
  surface's `.to(Out.type.element_ty)` cast **erases at translation**
  (`attnStoreStmt` is a bare `Stmt.store .real … (Op.ref .real … "acc")`), the
  same boundary the exact stack draws — so `outDType := .real` and the
  terminal cells carry the exact fold values at every `R`. (The host buffer's
  float16-ness lives inside that erased cast, not in the algorithm-layer
  store.)
* **No causal sentinel.** This kernel is the `STAGE = 1` specialization of the
  `attn_fwd_triton` family: a single non-causal inner pass (`lo, hi = 0,
  N_CTX`). There is no `-1e6` `tl.where` sentinel, so the sibling ports'
  score-bound side condition is simply absent here. -/

section IOFace

open scoped VeriTile.Triton.StreamMasked3DKernelIO₅

open VeriTile.Examples.AttentionForwardClosedForm
  (attnLoopBody attnAccAssign attnStoreStmt attnInvariant preLoop preLoop_scalars
   attn_step attn_postLoop attention_forward_triton_exec_reduction attn_body_split
   qTileMasked qTileMasked_active
   qmask_eval kmask_eval vmask_eval load_ptr_none_real load_ptr_mask_real
   qk_op_eval mij_op_eval qk2_op_eval p_op_eval lij_op_eval alpha_op_eval
   li_op_eval acc1_op_eval pfp16_op_eval acc2_op_eval kptr_adv_eval ksptr_adv_eval)

/-! ### Cast-free collapse under `R.round .fp16 = id` -/

/-- The `R`-cast on the `.real` channel never rounds. -/
private theorem attnIO_Rcast_real_real (R : RoundingModel) :
    R.cast .real .real = FloatDType.cast .real .real := by
  funext x
  unfold RoundingModel.cast FloatDType.cast
  rw [show R.roundW .real = id from by
    funext y; cases y <;> simp [RoundingModel.roundW]]
  rfl

/-- With `R.round .fp16 = id`, the `R`-cast into the fp16 grid is the exact
cast (the in-loop rounding-event site collapses). -/
private theorem attnIO_Rcast_real_fp16 (R : RoundingModel) (hfp16 : R.round .fp16 = id) :
    R.cast .real .fp16 = FloatDType.cast .real .fp16 := by
  funext x
  unfold RoundingModel.cast FloatDType.cast
  rw [show R.roundW .fp16 = id from by
    funext y; cases y <;> simp [RoundingModel.roundW, hfp16]]
  rfl

/-- The `R`-cast out of the fp16 grid lands on the `.real` channel, which
never rounds. -/
private theorem attnIO_Rcast_fp16_real (R : RoundingModel) :
    R.cast .fp16 .real = FloatDType.cast .fp16 .real := by
  funext x
  unfold RoundingModel.cast FloatDType.cast
  rw [show R.roundW .real = id from by
    funext y; cases y <;> simp [RoundingModel.roundW]]
  rfl

/-- `.real` stores never round: `writeMemTypedR` delegates to the exact write. -/
private theorem attnIO_wmtR_real (R : RoundingModel) (s : BlockState)
    (region : RegionName) (o : Nat) (v : TileCarrier .real) :
    s.writeMemTypedR R .real region o v = s.writeMemTyped .real region o v := rfl

/-! ### Statement transcriptions (the compiled body, verbatim)

`(surface …).toAlgKernel.body` is a literal 25-element list: the 22-statement
prologue, the KV `forRange`, `acc /= l_i` and the masked terminal store. The
prologue is split head (0–10, the scalar offsets + index vectors) / tail
(11–19, the six pointer seeds + the three accumulators) / loads (20–21) so the
safety walk can reuse the exact stack's `preLoop_scalars` head walk. -/

/-- Prologue statements 0–10: the scalar offsets and the three index vectors. -/
def attnIOPreHead (stride_qz stride_qh H HEAD_DIM NC BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) :
    List Stmt :=
  [ Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [] "off_hz" (Op.programId 1),
    Stmt.assign .nat [] "off_z"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)),
    Stmt.assign .nat [] "off_h"
      (Op.mod .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)),
    Stmt.assign .nat [] "qvk_offset"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat stride_qz))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat stride_qh))),
    Stmt.assign .nat [] "vk_offset"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "qvk_offset") (Op.constNat HEAD_DIM)),
    Stmt.assign .nat [] "q_scale_offset"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (Op.constNat NC) (Op.constNat BLOCK_M)) (Op.constNat 1))
          (Op.constNat BLOCK_M))),
    Stmt.assign .nat [] "k_scale_offset"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (Op.constNat NC) (Op.constNat BLOCK_N)) (Op.constNat 1))
          (Op.constNat BLOCK_N))),
    Stmt.assign .nat [BLOCK_M] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M))
        (Op.arange BLOCK_M)),
    Stmt.assign .nat [BLOCK_N] "offs_n" (Op.arange BLOCK_N),
    Stmt.assign .nat [BLOCK_DMODEL] "offs_k" (Op.arange BLOCK_DMODEL) ]

/-- Prologue statements 11–19: the six pointer seeds and `m_i`/`l_i`/`acc`. -/
def attnIOPreTail9 (Q K V QScale KScale Out : RegionName)
    (HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) : List Stmt :=
  [ Stmt.assign .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [] "Q_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase QScale)
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "q_scale_offset") (Op.ref .nat [] "start_m"))),
    Stmt.assign .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM)))),
    Stmt.assign .ptr [] "K_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase KScale) (Op.ref .nat [] "k_scale_offset")),
    Stmt.assign .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .real [BLOCK_M] "m_i"
      (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf),
    Stmt.assign .real [BLOCK_M] "l_i"
      (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) (Op.const 1.0)),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc" (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)) ]

/-- Prologue statements 20–21: the masked `q` load and the `q_scale` slot load. -/
def attnIOPreLoads (NC BLOCK_M BLOCK_DMODEL HEAD_ACTIVE : Nat) : List Stmt :=
  [ Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "q"
      (Op.load .real (.ptr (.ref .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"))
        (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat NC))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL)
              (Op.constNat HEAD_ACTIVE)))))),
    Stmt.assign .real [] "q_scale"
      (Op.load .real (.ptr (.ref .ptr [] "Q_scale_ptr")) .none) ]

/-- The 20 load-free prologue statements (head ++ tail). -/
def attnIOPre20 (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM NC BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) : List Stmt :=
  attnIOPreHead stride_qz stride_qh H HEAD_DIM NC BLOCK_M BLOCK_N BLOCK_DMODEL
    ++ attnIOPreTail9 Q K V QScale KScale Out HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL

/-- The whole 22-statement prologue. -/
def attnIOPre22 (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM NC BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) :
    List Stmt :=
  attnIOPre20 Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM NC BLOCK_M BLOCK_N BLOCK_DMODEL
    ++ attnIOPreLoads NC BLOCK_M BLOCK_DMODEL HEAD_ACTIVE

/-- The prologue transcription is the compiled body's first 22 statements. -/
theorem attnIO_pre22_check (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat) :
    (attention_forward_triton_surface Q K V QScale KScale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE).toAlgKernel.body.take 22
      = attnIOPre22 Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM
          (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE := rfl

/-- Fully explicit body split: prologue ++ [KV loop, `acc /= l_i`, store]. -/
theorem attnIO_body_split (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat) :
    (attention_forward_triton_surface Q K V QScale KScale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE).toAlgKernel.body
      = attnIOPre22 Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM
          (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE
        ++ (Stmt.forRange "start_n" 0 (BLOCK_N * numKVBlocks) BLOCK_N
              (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
            :: attnAccAssign BLOCK_M BLOCK_DMODEL
            :: attnStoreStmt BLOCK_M BLOCK_DMODEL BLOCK_N numKVBlocks HEAD_ACTIVE :: []) := rfl

/-! ### The cast-free run bridge

Under `hfp16` every rounding site in the lowered kernel is the identity, so
`execR R` and the exact `exec` coincide statement by statement. -/

set_option maxHeartbeats 4000000 in
/-- Every prologue statement is cast-free (`.nat`/`.real`/`.ptr` register
arithmetic and the two `.real` loads — no rounding site in sight). -/
private theorem attnIO_pre22_stmt_castFree (R : RoundingModel)
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM NC BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) :
    ∀ st ∈ attnIOPre22 Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM NC
        BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [attnIOPre22, attnIOPre20, attnIOPreHead, attnIOPreTail9, attnIOPreLoads,
    List.cons_append, List.nil_append, List.mem_append, List.mem_cons, List.not_mem_nil,
    or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]

set_option maxHeartbeats 4000000 in
/-- Every loop-body statement is cast-free **given `R.round .fp16 = id`**: the
three `castFloat` sites (the fp32 wrapper in statement 4, statement 14's
`p = (p).to(tl.float16)`, and the `fp16 → real` re-widening inside statement
15's dot) collapse via the three `attnIO_Rcast_*` lemmas. -/
private theorem attnIO_loopBody_stmt_castFree (R : RoundingModel)
    (hfp16 : R.round .fp16 = id)
    (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks : Nat) :
    ∀ st ∈ attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [attnLoopBody, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def,
      attnIO_Rcast_real_real R, attnIO_Rcast_real_fp16 R hfp16, attnIO_Rcast_fp16_real R]

set_option maxHeartbeats 4000000 in
/-- Both postLoop statements are cast-free: the `acc /= l_i` divide is
register arithmetic and the masked terminal store is `.real`-typed
(`attnIO_wmtR_real`). -/
private theorem attnIO_postLoop_stmt_castFree (R : RoundingModel)
    (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat) :
    ∀ st ∈ [attnAccAssign BLOCK_M BLOCK_DMODEL,
            attnStoreStmt BLOCK_M BLOCK_DMODEL BLOCK_N numKVBlocks HEAD_ACTIVE],
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl <;>
    simp only [attnAccAssign, attnStoreStmt, stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def,
      attnIO_wmtR_real R]

/-- Per-statement cast-free collapse lifts to statement lists (walks the
actual successor chain; a failing step collapses on both sides). -/
private theorem attnIO_stepStmtsR_castFree_of_stmts (R : RoundingModel) :
    ∀ (l : List Stmt), (∀ st ∈ l, ∀ u, stepStmtR R st u = stepStmt st u) →
      ∀ s, stepStmtsR R l s = stepStmts l s
  | [], _, s => by simp only [stepStmtsR, stepStmts]
  | st :: rest, h, s => by
      simp only [stepStmtsR, stepStmts, h st List.mem_cons_self s]
      cases stepStmt st s with
      | none => rfl
      | some s' =>
          exact attnIO_stepStmtsR_castFree_of_stmts R rest
            (fun st' h' u => h st' (List.mem_cons_of_mem _ h') u) s'

/-- The prologue collapses onto the exact stepper. -/
private theorem attnIO_pre22_castFree (R : RoundingModel)
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM NC BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat)
    (t : BlockState) :
    stepStmtsR R (attnIOPre22 Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM NC
        BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) t
      = stepStmts (attnIOPre22 Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM NC
        BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) t :=
  attnIO_stepStmtsR_castFree_of_stmts R _
    (attnIO_pre22_stmt_castFree R Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM NC
      BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) t

/-- The loop body collapses onto the exact stepper (under `hfp16`). -/
private theorem attnIO_loopBody_castFree (R : RoundingModel) (hfp16 : R.round .fp16 = id)
    (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks : Nat) (t : BlockState) :
    stepStmtsR R (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks) t
      = stepStmts (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks) t :=
  attnIO_stepStmtsR_castFree_of_stmts R _
    (attnIO_loopBody_stmt_castFree R hfp16 BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM
      numKVBlocks) t

/-- The static streaming `forRange` statement is cast-free given
`R.round .fp16 = id` (static bounds, cast-free body through
`stepForRangeAuxR_castFree`). -/
private theorem attnIO_loopStmt_castFree (R : RoundingModel) (hfp16 : R.round .fp16 = id)
    (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks : Nat) :
    ∀ u, stepStmtR R
        (Stmt.forRange "start_n" 0 (BLOCK_N * numKVBlocks) BLOCK_N
          (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)) u
      = stepStmt
        (Stmt.forRange "start_n" 0 (BLOCK_N * numKVBlocks) BLOCK_N
          (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)) u := by
  intro u
  rw [stepStmtR_forRange,
    stepForRangeAuxR_castFree R _
      (attnIO_loopBody_castFree R hfp16 BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM
        numKVBlocks) "start_n",
    ← stepForRangeAux.forRange_unfold]

/-- The postLoop collapses onto the exact stepper. -/
private theorem attnIO_postLoop_castFree (R : RoundingModel)
    (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat) (t : BlockState) :
    stepStmtsR R [attnAccAssign BLOCK_M BLOCK_DMODEL,
        attnStoreStmt BLOCK_M BLOCK_DMODEL BLOCK_N numKVBlocks HEAD_ACTIVE] t
      = stepStmts [attnAccAssign BLOCK_M BLOCK_DMODEL,
        attnStoreStmt BLOCK_M BLOCK_DMODEL BLOCK_N numKVBlocks HEAD_ACTIVE] t :=
  attnIO_stepStmtsR_castFree_of_stmts R _
    (attnIO_postLoop_stmt_castFree R BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks) t

set_option maxHeartbeats 4000000 in
/-- **The cast-free run bridge**: under `hfp16` the rounded interpreter agrees
with the exact one on the whole kernel, so every exact-stack `exec` fact
transports verbatim to `execR R`. -/
private theorem attnIO_execR_eq_exec (R : RoundingModel) (hfp16 : R.round .fp16 = id)
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
    (s : BlockState) :
    execR R (attention_forward_triton_surface Q K V QScale KScale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE).toAlgKernel s
      = exec (attention_forward_triton_surface Q K V QScale KScale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE).toAlgKernel s := by
  unfold execR exec
  rw [attnIO_body_split]
  refine attnIO_stepStmtsR_castFree_of_stmts R _ ?_ s
  intro st hst u
  rcases List.mem_append.mp hst with hpre | htail
  · exact attnIO_pre22_stmt_castFree R Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM
      (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE st hpre u
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at htail
    rcases htail with rfl | rfl | rfl
    · exact attnIO_loopStmt_castFree R hfp16 BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM
        numKVBlocks u
    · exact attnIO_postLoop_stmt_castFree R BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks
        _ (by simp) u
    · exact attnIO_postLoop_stmt_castFree R BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks
        _ (by simp) u

set_option maxHeartbeats 4000000 in
/-- The `attention_forward_triton` surface sits inside the flat-memory bridge's
covered fragment (plain `ptrAdd` walks only — no `ptrSub`, no atomics, no block
pointers). -/
private theorem attnIO_flattenOk (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat) :
    ((attention_forward_triton_surface Q K V QScale KScale Out
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE STAGE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [attnIO_body_split]
  simp [attnIOPre22, attnIOPre20, attnIOPreHead, attnIOPreTail9, attnIOPreLoads,
    attnLoopBody, attnAccAssign, attnStoreStmt,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  simp [Op.FlattenOk.eq_def]

/-! ### IO signature, stream tiles, and the closed-form spec `f`

Window transcription (strides pinned to the exec stack's contiguous layout
`stride_qm = stride_kn = stride_vk = stride_om = HEAD_DIM`, unit fastest
stride; shared plane base `p₁/H·stride_qz + p₁%H·stride_qh`; the context is
`N_CTX = BLOCK_N · numKVBlocks`, so the trip count is `T = numKVBlocks`):

* `read1` (`Q`, static — the window ignores `t`): lane `j = (i, e)` row-major
  over `[BLOCK_M, BLOCK_DMODEL]` reads `base + (p₀·BM + i)·HEAD_DIM + e`,
  masked `p₀·BM + i < N_CTX ∧ e < HEAD_ACTIVE` (the kernel's `q` load mask).
* `read2` (`K`, transposed, advanced `BLOCK_N` keys per step): lane
  `j = (e, r)` row-major over `[BLOCK_DMODEL, BLOCK_N]` reads
  `base + e + (t·BN + r)·HEAD_DIM`, masked `r < N_CTX ⊖ t·BN ∧
  e < HEAD_ACTIVE` (`⊖` = the kernel's ℕ-truncated `N_CTX - start_n`,
  transcribed verbatim).
* `read3` (`V`, advanced `BLOCK_N` rows per step): lane `j = (r, d)` reads
  `base + (t·BN + r)·HEAD_DIM + d`, masked `r < N_CTX ⊖ t·BN ∧
  d < HEAD_ACTIVE`.
* `read4` (`Q_scale`, **scalar**, static): the single lane reads
  `p₁·⌈N_CTX/BM⌉ + p₀`, unmasked.
* `read5` (`K_scale`, **scalar**, one slot per step): the single lane reads
  `p₁·⌈N_CTX/BN⌉ + t`, unmasked.
* `write` (`Out`): lane `(i, e)` writes `base + (p₀·BM + i)·HEAD_DIM + e`
  under the genuine store mask `p₀·BM + i < N_CTX ∧ e < HEAD_ACTIVE`.
* `outDType := .real`: the lowered terminal store is `.real`-typed (the
  surface's `.to(Out.type.element_ty)` erases at translation — see
  `attnStoreStmt`), so the terminal grid is exact. -/

/-- **Streaming IO signature** of `_attn_fwd` on the five-stream single-store
attention fold skin (`T = numKVBlocks` under `N_CTX = BLOCK_N · numKVBlocks`;
the trip count is pid-free, so there is no launch-legality `pre` analog to
carry). -/
def attnFwdIO (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat) : StreamMasked3DKernelIO₅ where
  kernel := attention_forward_triton_surface Q K V QScale KScale Out
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
  inp1 := Q
  inp2 := K
  inp3 := V
  inp4 := QScale
  inp5 := KScale
  out := Out
  T := numKVBlocks
  B1 := BLOCK_M * BLOCK_DMODEL
  B2 := BLOCK_DMODEL * BLOCK_N
  B3 := BLOCK_N * BLOCK_DMODEL
  B4 := 1
  B5 := 1
  C := BLOCK_M * BLOCK_DMODEL
  outDType := .real
  read1 := fun p₀ p₁ _ _ j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (p₀ * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  read2 := fun _ p₁ _ t j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD_DIM
  read3 := fun _ p₁ _ t j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (t.val * BLOCK_N + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  read4 := fun p₀ p₁ _ _ _ =>
    p₁ * ((BLOCK_N * numKVBlocks + BLOCK_M - 1) / BLOCK_M) + p₀
  read5 := fun _ p₁ _ t _ =>
    p₁ * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + t.val
  write := fun p₀ p₁ _ j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (p₀ * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  mask1 := fun p₀ _ _ _ j =>
    p₀ * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
      ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE
  mask2 := fun _ _ _ t j =>
    j.val % BLOCK_N < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE
  mask3 := fun _ _ _ t j =>
    j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks - t.val * BLOCK_N
      ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE
  mask4 := fun _ _ _ _ _ => True
  mask5 := fun _ _ _ _ _ => True
  writeMask := fun p₀ _ _ j =>
    p₀ * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
      ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE

/-- The **row-masked query tile** read off the static first stream (the window
ignores `t`, so the step-`0` slice carries the whole tile; rows whose global
index is `≥ N_CTX` are `0`, mirroring the kernel's `q` row mask). -/
noncomputable def attnIOqTm (p₀ NC BLOCK_M BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (xs : Fin T → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ) :
    TileIndex [BLOCK_M, HEAD_ACTIVE] → ℝ :=
  fun idx =>
    if h : p₀ * BLOCK_M + idx.1.val < NC ∧ 0 < T ∧ idx.2.1.val < BLOCK_DMODEL then
      xs ⟨0, h.2.1⟩ (Lane2D.encode (idx.1, ⟨idx.2.1.val, h.2.2⟩, PUnit.unit))
    else 0

/-- The **global key tile** read off the transposed `K` stream: global key `jg`
lives in step `jg / BLOCK_N` at block-local column `jg % BLOCK_N`. -/
noncomputable def attnIOkTm (BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (ys : Fin T → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ) :
    TileIndex [BLOCK_N * T, HEAD_ACTIVE] → ℝ :=
  fun idx =>
    if h : idx.2.1.val < BLOCK_DMODEL ∧ idx.1.val / BLOCK_N < T ∧ 0 < BLOCK_N then
      ys ⟨idx.1.val / BLOCK_N, h.2.1⟩
        (Lane2D.encode (⟨idx.2.1.val, h.1⟩, ⟨idx.1.val % BLOCK_N, Nat.mod_lt _ h.2.2⟩, PUnit.unit))
    else 0

/-- The **global value tile** read off the `V` stream (row-major per-step
tiles). -/
noncomputable def attnIOvTm (BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (zs : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ) :
    TileIndex [BLOCK_N * T, HEAD_ACTIVE] → ℝ :=
  fun idx =>
    if h : idx.2.1.val < BLOCK_DMODEL ∧ idx.1.val / BLOCK_N < T ∧ 0 < BLOCK_N then
      zs ⟨idx.1.val / BLOCK_N, h.2.1⟩
        (Lane2D.encode (⟨idx.1.val % BLOCK_N, Nat.mod_lt _ h.2.2⟩, ⟨idx.2.1.val, h.1⟩, PUnit.unit))
    else 0

/-- The per-key score-scale carrier `q_scale · k_scale` read off the two
scalar streams: the static `Q_scale` slot (step `0`) times key `j`'s
`K_scale` slot (step `j / BLOCK_N`). -/
noncomputable def attnIOkeyScale (BLOCK_N T : Nat) (ws vs : Fin T → Fin 1 → ℝ) :
    Fin (BLOCK_N * T) → ℝ :=
  fun j =>
    (if h : 0 < T then ws ⟨0, h⟩ 0 else 0)
      * (if h : j.val / BLOCK_N < T then vs ⟨j.val / BLOCK_N, h⟩ 0 else 0)

/-- **The streamed closed-form spec `f`**: base-2 per-key-scale attention over
the tiles assembled from the five streams (`attentionRealBase2PerKeyScale`
restated tile-parametrically; the contraction axis is the `HEAD_ACTIVE`
active head lanes, exactly as in the exact stack). -/
noncomputable def attnFwdIOOutSpec
    (p₀ BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (xs : Fin T → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ)
    (ys : Fin T → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ)
    (zs : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (ws vs : Fin T → Fin 1 → ℝ)
    (idx : TileIndex [BLOCK_M, HEAD_ACTIVE]) : ℝ :=
  attentionRealBase2PerKeyScale
    (attnIOqTm p₀ (BLOCK_N * T) BLOCK_M BLOCK_DMODEL HEAD_ACTIVE T xs)
    (attnIOkTm BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T ys)
    (attnIOvTm BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T zs)
    (attnIOkeyScale BLOCK_N T ws vs) idx

/-! ### Stream-pin tile bridges

Under the skin's input pins the kernel-side tiles/scales of the exact stack
(`VeriTile/Examples/AttentionForwardClosedForm.lean`) coincide with the
stream-built ones. The exact stack's names are opened under an `ex`-prefix to
keep them apart from this file's own layout vocabulary. -/

open VeriTile.Examples.AttentionForwardClosedForm renaming
  qTile → exQTile, kTile → exKTile, vTile → exVTile, keyScale → exKeyScale,
  baseOffset → exBaseOffset, mIndex → exMIndex, outOffset → exOutOffset, cdiv → exCdiv

private theorem attnIO_qTm_eq (s₀ : BlockState) (Q : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (hT : 0 < T) (hAD : HEAD_ACTIVE ≤ BLOCK_DMODEL)
    (xs : Fin T → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ)
    (hx : ∀ (t : Fin T) (j : Fin (BLOCK_M * BLOCK_DMODEL)),
      s₀.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * T
        ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      s₀.readMem Q (s₀.pids 1 / H * stride_qz + s₀.pids 1 % H * stride_qh
          + (s₀.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL)
        = xs t j) :
    qTileMasked s₀ Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE (BLOCK_N * T)
      = attnIOqTm (s₀.pids 0) (BLOCK_N * T) BLOCK_M BLOCK_DMODEL HEAD_ACTIVE T xs := by
  funext idx
  obtain ⟨i, e, ⟨⟩⟩ := idx
  have heBD : e.val < BLOCK_DMODEL := lt_of_lt_of_le e.isLt hAD
  simp only [qTileMasked, attnIOqTm, exMIndex, exBaseOffset]
  by_cases hm : s₀.pids 0 * BLOCK_M + i.val < BLOCK_N * T
  · rw [if_pos hm, dif_pos (⟨hm, hT, heBD⟩ :
      s₀.pids 0 * BLOCK_M + i.val < BLOCK_N * T ∧ 0 < T ∧ e.val < BLOCK_DMODEL)]
    rw [← hx ⟨0, hT⟩ (Lane2D.encode (i, ⟨e.val, heBD⟩, PUnit.unit)) (by
      simp only [Lane2D.encode_div, Lane2D.encode_mod]
      exact ⟨hm, e.isLt⟩)]
    simp only [Lane2D.encode_div, Lane2D.encode_mod]
  · rw [if_neg hm, dif_neg (fun hc => hm hc.1)]

private theorem attnIO_kTm_eq (s₀ : BlockState) (K : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (hBN : 0 < BLOCK_N) (hAD : HEAD_ACTIVE ≤ BLOCK_DMODEL)
    (ys : Fin T → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ)
    (hy : ∀ (t : Fin T) (j : Fin (BLOCK_DMODEL * BLOCK_N)),
      j.val % BLOCK_N < BLOCK_N * T - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE →
      s₀.readMem K (s₀.pids 1 / H * stride_qz + s₀.pids 1 % H * stride_qh
          + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD_DIM)
        = ys t j) :
    exKTile s₀ K H stride_qz stride_qh HEAD_DIM (BLOCK_N * T) HEAD_ACTIVE
      = attnIOkTm BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T ys := by
  funext idx
  obtain ⟨jg, e, ⟨⟩⟩ := idx
  have heBD : e.val < BLOCK_DMODEL := lt_of_lt_of_le e.isLt hAD
  have hjT : jg.val / BLOCK_N < T :=
    (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [Nat.mul_comm T BLOCK_N]; exact jg.isLt)
  have hsplit : jg.val / BLOCK_N * BLOCK_N + jg.val % BLOCK_N = jg.val := by
    conv_rhs => rw [← Nat.div_add_mod jg.val BLOCK_N]
    rw [Nat.mul_comm]
  simp only [exKTile, attnIOkTm, exBaseOffset]
  rw [dif_pos (⟨heBD, hjT, hBN⟩ :
    e.val < BLOCK_DMODEL ∧ jg.val / BLOCK_N < T ∧ 0 < BLOCK_N)]
  rw [← hy ⟨jg.val / BLOCK_N, hjT⟩
      (Lane2D.encode (⟨e.val, heBD⟩, ⟨jg.val % BLOCK_N, Nat.mod_lt _ hBN⟩, PUnit.unit))
      (by
        simp only [Lane2D.encode_div, Lane2D.encode_mod]
        refine ⟨?_, e.isLt⟩
        have := jg.isLt
        omega)]
  simp only [Lane2D.encode_div, Lane2D.encode_mod]
  refine congrArg (s₀.readMem K) ?_
  rw [hsplit]
  ring

private theorem attnIO_vTm_eq (s₀ : BlockState) (V : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (hBN : 0 < BLOCK_N) (hAD : HEAD_ACTIVE ≤ BLOCK_DMODEL)
    (zs : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (hz : ∀ (t : Fin T) (j : Fin (BLOCK_N * BLOCK_DMODEL)),
      j.val / BLOCK_DMODEL < BLOCK_N * T - t.val * BLOCK_N ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      s₀.readMem V (s₀.pids 1 / H * stride_qz + s₀.pids 1 % H * stride_qh
          + (t.val * BLOCK_N + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL)
        = zs t j) :
    exVTile s₀ V H stride_qz stride_qh HEAD_DIM (BLOCK_N * T) HEAD_ACTIVE
      = attnIOvTm BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T zs := by
  funext idx
  obtain ⟨jg, d, ⟨⟩⟩ := idx
  have hdBD : d.val < BLOCK_DMODEL := lt_of_lt_of_le d.isLt hAD
  have hjT : jg.val / BLOCK_N < T :=
    (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [Nat.mul_comm T BLOCK_N]; exact jg.isLt)
  have hsplit : jg.val / BLOCK_N * BLOCK_N + jg.val % BLOCK_N = jg.val := by
    conv_rhs => rw [← Nat.div_add_mod jg.val BLOCK_N]
    rw [Nat.mul_comm]
  simp only [exVTile, attnIOvTm, exBaseOffset]
  rw [dif_pos (⟨hdBD, hjT, hBN⟩ :
    d.val < BLOCK_DMODEL ∧ jg.val / BLOCK_N < T ∧ 0 < BLOCK_N)]
  rw [← hz ⟨jg.val / BLOCK_N, hjT⟩
      (Lane2D.encode (⟨jg.val % BLOCK_N, Nat.mod_lt _ hBN⟩, ⟨d.val, hdBD⟩, PUnit.unit))
      (by
        simp only [Lane2D.encode_div, Lane2D.encode_mod]
        refine ⟨?_, d.isLt⟩
        have := jg.isLt
        omega)]
  simp only [Lane2D.encode_div, Lane2D.encode_mod]
  refine congrArg (s₀.readMem V) ?_
  rw [hsplit]

private theorem attnIO_keyScale_eq (s₀ : BlockState) (QScale KScale : RegionName)
    (BLOCK_M BLOCK_N T : Nat) (hT : 0 < T) (hBN : 0 < BLOCK_N)
    (ws vs : Fin T → Fin 1 → ℝ)
    (hw : ∀ (t : Fin T) (j : Fin 1),
      s₀.readMem QScale (s₀.pids 1 * ((BLOCK_N * T + BLOCK_M - 1) / BLOCK_M) + s₀.pids 0)
        = ws t j)
    (hv : ∀ (t : Fin T) (j : Fin 1),
      s₀.readMem KScale (s₀.pids 1 * ((BLOCK_N * T + BLOCK_N - 1) / BLOCK_N) + t.val)
        = vs t j) :
    exKeyScale s₀ QScale KScale (BLOCK_N * T) BLOCK_M BLOCK_N (BLOCK_N * T)
      = attnIOkeyScale BLOCK_N T ws vs := by
  funext j
  have hjT : j.val / BLOCK_N < T :=
    (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [Nat.mul_comm T BLOCK_N]; exact j.isLt)
  simp only [exKeyScale, attnIOkeyScale, exCdiv]
  rw [dif_pos hT, dif_pos hjT, ← hw ⟨0, hT⟩ 0, ← hv ⟨j.val / BLOCK_N, hjT⟩ 0]

/-- The closed form only consumes row `i` of the query tile, so two tiles that
agree on that row give the same value (this is what lets the row-masked
stream tile stand in for the exact stack's unmasked `qTile` on active rows). -/
private theorem attnIO_spec_row_congr {M S D : Nat}
    (Q Q' : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ) (kS : Fin S → ℝ)
    (i : Fin M) (d : Fin D)
    (hrow : ∀ e : Fin D, Q (i, e, PUnit.unit) = Q' (i, e, PUnit.unit)) :
    attentionRealBase2PerKeyScale Q K V kS (i, d, PUnit.unit)
      = attentionRealBase2PerKeyScale Q' K V kS (i, d, PUnit.unit) := by
  have hraw : ∀ j : Fin S,
      Finset.univ.sum (fun e : Fin D => Q (i, e, PUnit.unit) * K (j, e, PUnit.unit))
        = Finset.univ.sum (fun e : Fin D => Q' (i, e, PUnit.unit) * K (j, e, PUnit.unit)) :=
    fun j => Finset.sum_congr rfl (fun e _ => by rw [hrow e])
  simp only [attentionRealBase2PerKeyScale, hraw]

/-! ### The safety walk (weak invariant)

The skin's `hts` obligation quantifies over **arbitrary** launch states (no
clean-`undef` pin), so the exact stack's `attnInvariant` is unavailable there.
The safety walk instead runs on the *shape* half: exact pins for the
loop-carried pointer/index registers (whose addresses are the bound
obligations) plus bare existence for the value registers. -/

/-- General `Q_ptrs` / `O_block_ptr` tile (constant across the KV loop): cell
`(i, e)` → `base + (pid₀·BLOCK_M + i)·HEAD_DIM + e`. -/
private def attnIOrowPtrs (s0 : BlockState) (rg : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL : Nat) :
    Tile .ptr [BLOCK_M, BLOCK_DMODEL] :=
  ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
    (rg.cast, exBaseOffset s0 H stride_qz stride_qh
      + (s0.pids 0 * BLOCK_M + idx.1.val) * HEAD_DIM + idx.2.1.val)⟩

/-- General `K_ptrs` after `c` key blocks (transposed tile): cell `(e, jL)` →
`base + e + (c·BLOCK_N + jL)·HEAD_DIM`. -/
private def attnIOkPtrs (s0 : BlockState) (K : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL c : Nat) :
    Tile .ptr [BLOCK_DMODEL, BLOCK_N] :=
  ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
    (K.cast, exBaseOffset s0 H stride_qz stride_qh
      + idx.1.val + (c * BLOCK_N + idx.2.1.val) * HEAD_DIM)⟩

/-- General `V_ptrs` after `c` key blocks: cell `(jL, d)` →
`base + (c·BLOCK_N + jL)·HEAD_DIM + d`. -/
private def attnIOvPtrs (s0 : BlockState) (V : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL c : Nat) :
    Tile .ptr [BLOCK_N, BLOCK_DMODEL] :=
  ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
    (V.cast, exBaseOffset s0 H stride_qz stride_qh
      + (c * BLOCK_N + idx.1.val) * HEAD_DIM + idx.2.1.val)⟩

/-- A scalar-width pointer (`Q_scale_ptr` / `K_scale_ptr`). -/
private def attnIOsclPtr (rg : RegionName) (o : Nat) : Tile .ptr [] :=
  ⟨fun _ : TileIndex [] => (rg.cast, o)⟩

private theorem attnIOkPtrs_succ (s0 : BlockState) (K : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL c : Nat) :
    Tile.ptrAdd Broadcast.scalarR
        (attnIOkPtrs s0 K H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL c)
        (Tile.scalar (BLOCK_N * HEAD_DIM))
      = attnIOkPtrs s0 K H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c + 1) := by
  ext idx
  · rfl
  · simp only [attnIOkPtrs, Tile.ptrAdd_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR]
    ring

private theorem attnIOvPtrs_succ (s0 : BlockState) (V : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL c : Nat) :
    Tile.ptrAdd Broadcast.scalarR
        (attnIOvPtrs s0 V H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL c)
        (Tile.scalar (BLOCK_N * HEAD_DIM))
      = attnIOvPtrs s0 V H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c + 1) := by
  ext idx
  · rfl
  · simp only [attnIOvPtrs, Tile.ptrAdd_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR]
    ring

private theorem attnIOsclPtr_succ (rg : RegionName) (o : Nat) :
    Tile.ptrAdd Broadcast.nil (attnIOsclPtr rg o) (Tile.scalar 1)
      = attnIOsclPtr rg (o + 1) := by
  ext idx
  · rfl
  · simp only [attnIOsclPtr, Tile.ptrAdd_data, Tile.scalar,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]

/-- Register readback survives a `setReg` under a different name. -/
private theorem attnIO_regs_chain {d d' : TileDType} {sh sh' : TileShape}
    {n n' : RegName} {s : BlockState} {v : Tile d sh} {w : Tile d' sh'}
    (hne : n ≠ n') (h : s.regs d sh n = some v) :
    (s.setReg n' d' sh' w).regs d sh n = some v := by
  simp only [BlockState.setReg_ne_name, ne_eq, hne, not_false_eq_true, h]

/-- **Masked pointer load, `undef`-generic** (the weak-walk counterpart of the
exact stack's `load_ptr_mask_real`, which pins `undef = 0`): the load always
succeeds; masked-off lanes carry the state's `undef` value. -/
private theorem attnIO_load_mask_realW {shape : TileShape}
    (ptrOp : Op .ptr shape) (maskOp : Op .bool shape) (s : BlockState)
    (ptrs : Tile .ptr shape) (masks : Tile .bool shape)
    (hp : evalOp ptrOp s = some ptrs) (hm : evalOp maskOp s = some masks) :
    evalOp (Op.load .real (.ptr ptrOp) (.mask maskOp)) s
      = some ⟨fun i => if masks.data i then
          some (s.readMem (ptrs.data i).1 (ptrs.data i).2)
        else some (s.undef (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, hp, hm, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue_real]
  cases hmi : masks.data i <;> simp [hmi]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- The prologue head is the compiled body's first 11 statements (so the exact
stack's `undef`-free `preLoop_scalars` walk drives the weak head walk). -/
private theorem attnIO_preHead_check (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat) :
    (attention_forward_triton_surface Q K V QScale KScale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE).toAlgKernel.body.take 11
      = attnIOPreHead stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks)
          BLOCK_M BLOCK_N BLOCK_DMODEL := rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Weak 9-statement tail walk** (prologue statements 11–19: the six pointer
seeds and the three accumulator inits), from an arbitrary launch state. -/
private theorem attnIO_preTail9W
    (s1 : BlockState) (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM NC BLOCK_M BLOCK_N BLOCK_DMODEL : Nat)
    (hstartm : s1.regs .nat [] "start_m" = some (Tile.scalar (s1.pids 0)))
    (hqvk : s1.regs .nat [] "qvk_offset"
      = some (Tile.scalar (exBaseOffset s1 H stride_qz stride_qh)))
    (hqso : s1.regs .nat [] "q_scale_offset"
      = some (Tile.scalar (s1.pids 1 * ((NC + BLOCK_M - 1) / BLOCK_M))))
    (hkso : s1.regs .nat [] "k_scale_offset"
      = some (Tile.scalar (s1.pids 1 * ((NC + BLOCK_N - 1) / BLOCK_N))))
    (hoffsm : s1.regs .nat [BLOCK_M] "offs_m"
      = some (Tile.vec (fun r : Fin BLOCK_M => s1.pids 0 * BLOCK_M + r.val)))
    (hoffsn : s1.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hoffsk : s1.regs .nat [BLOCK_DMODEL] "offs_k"
      = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))) :
    ∃ s20, stepStmts (attnIOPreTail9 Q K V QScale KScale Out HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL) s1
        = some s20
      ∧ s20.pids = s1.pids ∧ s20.mem = s1.mem ∧ s20.undef = s1.undef
      ∧ s20.regs .nat [BLOCK_M] "offs_m"
          = some (Tile.vec (fun r : Fin BLOCK_M => s1.pids 0 * BLOCK_M + r.val))
      ∧ s20.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
      ∧ (∃ mT : Tile .real [BLOCK_M], s20.regs .real [BLOCK_M] "m_i" = some mT)
      ∧ (∃ lT : Tile .real [BLOCK_M], s20.regs .real [BLOCK_M] "l_i" = some lT)
      ∧ (∃ aT : Tile .real [BLOCK_M, BLOCK_DMODEL],
          s20.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some aT)
      ∧ s20.regs .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"
          = some (attnIOrowPtrs s1 Q H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL)
      ∧ s20.regs .ptr [] "Q_scale_ptr"
          = some (attnIOsclPtr QScale (s1.pids 1 * ((NC + BLOCK_M - 1) / BLOCK_M) + s1.pids 0))
      ∧ s20.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs"
          = some (attnIOkPtrs s1 K H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL 0)
      ∧ s20.regs .ptr [] "K_scale_ptr"
          = some (attnIOsclPtr KScale (s1.pids 1 * ((NC + BLOCK_N - 1) / BLOCK_N) + 0))
      ∧ s20.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
          = some (attnIOvPtrs s1 V H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL 0)
      ∧ s20.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
          = some (attnIOrowPtrs s1 Out H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL) := by
  unfold attnIOPreTail9
  -- stmt 11: Q_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))) s1
        = some (attnIOrowPtrs s1 Q H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL) from by
      rw [evalOp_ptrAdd, evalOp_ptrBase, evalOp_add, evalOp_add, evalOp_mul, evalOp_mul]
      erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hoffsm,
        evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hoffsk]
      rw [evalOp_ref, hqvk]
      simp only [evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
      refine congrArg some ?_
      ext idx
      · rfl
      · simp only [attnIOrowPtrs, Tile.ptrAdd_data, Tile.scalar, Tile.bop_data,
          Tile.expandDim_data, Tile.vec_data, Broadcast.leftIndex_scalarL,
          Broadcast.rightIndex_scalarL, Broadcast.leftIndex_scalarR,
          Broadcast.rightIndex_scalarR, Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
          Broadcast.leftIndex_consR, Broadcast.rightIndex_consR, Broadcast.leftIndex_nil,
          Broadcast.rightIndex_nil, TileShape.dropInsertedIndex, NumericDType.add,
          NumericDType.mul, NumericDType.nat_add, NumericDType.nat_mul, exBaseOffset]
        ring_nf))]
  -- stmt 12: Q_scale_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase QScale)
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "q_scale_offset") (Op.ref .nat [] "start_m"))) _
        = some (attnIOsclPtr QScale (s1.pids 1 * ((NC + BLOCK_M - 1) / BLOCK_M) + s1.pids 0)) from by
      rw [evalOp_ptrAdd, evalOp_ptrBase, evalOp_add]
      simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, hqso, hstartm, Option.bind_some, Option.bind_eq_bind]
      refine congrArg some ?_
      ext idx
      · rfl
      · simp only [attnIOsclPtr, Tile.ptrAdd_data, Tile.scalar, Tile.bop_data,
          Broadcast.leftIndex_nil, Broadcast.rightIndex_nil, NumericDType.add,
          NumericDType.nat_add, Nat.zero_add]))]
  -- stmt 13: K_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM)))) _
        = some (attnIOkPtrs s1 K H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL 0) from by
      rw [evalOp_ptrAdd, evalOp_ptrBase, evalOp_add, evalOp_add, evalOp_mul]
      erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _
            (attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide) hoffsk)),
        evalOp_expandDim_ref_of_regs _ _ _ _ _ _
            (attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide) hoffsn))]
      rw [evalOp_ref, attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide) hqvk)]
      simp only [evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
      refine congrArg some ?_
      ext idx
      · rfl
      · simp only [attnIOkPtrs, Tile.ptrAdd_data, Tile.scalar, Tile.bop_data,
          Tile.expandDim_data, Tile.vec_data, Broadcast.leftIndex_scalarL,
          Broadcast.rightIndex_scalarL, Broadcast.leftIndex_scalarR,
          Broadcast.rightIndex_scalarR, Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
          Broadcast.leftIndex_consR, Broadcast.rightIndex_consR, Broadcast.leftIndex_nil,
          Broadcast.rightIndex_nil, TileShape.dropInsertedIndex, NumericDType.add,
          NumericDType.mul, NumericDType.nat_add, NumericDType.nat_mul, exBaseOffset]
        ring_nf))]
  -- stmt 14: K_scale_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase KScale) (Op.ref .nat [] "k_scale_offset")) _
        = some (attnIOsclPtr KScale (s1.pids 1 * ((NC + BLOCK_N - 1) / BLOCK_N) + 0)) from by
      rw [evalOp_ptrAdd, evalOp_ptrBase, evalOp_ref]
      simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hkso,
        Option.bind_some, Option.bind_eq_bind]
      refine congrArg some ?_
      ext idx
      · rfl
      · simp only [attnIOsclPtr, Tile.ptrAdd_data, Tile.scalar,
          Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
        omega))]
  -- stmt 15: V_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))) _
        = some (attnIOvPtrs s1 V H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL 0) from by
      rw [evalOp_ptrAdd, evalOp_ptrBase, evalOp_add, evalOp_add, evalOp_mul, evalOp_mul]
      erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _
            (attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide)
              (attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide) hoffsn)))),
        evalOp_expandDim_ref_of_regs _ _ _ _ _ _
            (attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide)
              (attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide) hoffsk))))]
      rw [evalOp_ref, attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide)
            (attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide) hqvk)))]
      simp only [evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
      refine congrArg some ?_
      ext idx
      · rfl
      · simp only [attnIOvPtrs, Tile.ptrAdd_data, Tile.scalar, Tile.bop_data,
          Tile.expandDim_data, Tile.vec_data, Broadcast.leftIndex_scalarL,
          Broadcast.rightIndex_scalarL, Broadcast.leftIndex_scalarR,
          Broadcast.rightIndex_scalarR, Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
          Broadcast.leftIndex_consR, Broadcast.rightIndex_consR, Broadcast.leftIndex_nil,
          Broadcast.rightIndex_nil, TileShape.dropInsertedIndex, NumericDType.add,
          NumericDType.mul, NumericDType.nat_add, NumericDType.nat_mul, exBaseOffset]
        ring_nf))]
  -- stmt 16: O_block_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))) _
        = some (attnIOrowPtrs s1 Out H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL) from by
      rw [evalOp_ptrAdd, evalOp_ptrBase, evalOp_add, evalOp_add, evalOp_mul, evalOp_mul]
      erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _
            (attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide)
              (attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide)
                (attnIO_regs_chain (by decide) hoffsm))))),
        evalOp_expandDim_ref_of_regs _ _ _ _ _ _
            (attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide)
              (attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide)
                (attnIO_regs_chain (by decide) hoffsk)))))]
      rw [evalOp_ref, attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide)
            (attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide)
              (attnIO_regs_chain (by decide) hqvk))))]
      simp only [evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
      refine congrArg some ?_
      ext idx
      · rfl
      · simp only [attnIOrowPtrs, Tile.ptrAdd_data, Tile.scalar, Tile.bop_data,
          Tile.expandDim_data, Tile.vec_data, Broadcast.leftIndex_scalarL,
          Broadcast.rightIndex_scalarL, Broadcast.leftIndex_scalarR,
          Broadcast.rightIndex_scalarR, Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
          Broadcast.leftIndex_consR, Broadcast.rightIndex_consR, Broadcast.leftIndex_nil,
          Broadcast.rightIndex_nil, TileShape.dropInsertedIndex, NumericDType.add,
          NumericDType.mul, NumericDType.nat_add, NumericDType.nat_mul, exBaseOffset]
        ring_nf))]
  -- stmt 17: m_i = full ⊥
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf) _
        = some (⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩ : Tile .real [BLOCK_M]) from by
      rw [evalOp_add]
      simp only [evalOp_full, evalOp_const, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map]
      rfl))]
  -- stmt 18: l_i = full 1.0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) (Op.const 1.0)) _
        = some (⟨fun _ : TileIndex [BLOCK_M] => (some (1 : ℝ) : WithBot ℝ)⟩ : Tile .real [BLOCK_M]) from by
      rw [evalOp_add]
      simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map]
      norm_num))]
  -- stmt 19: acc = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => (some (0 : ℝ) : WithBot ℝ)⟩
            : Tile .real [BLOCK_M, BLOCK_DMODEL]) from by
      simp only [evalOp_full, evalOp_const]
      rfl))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · rfl
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids, hoffsm]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, hoffsn]
  · exact ⟨⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩,
      by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, BlockState.setReg_same]⟩
  · exact ⟨⟨fun _ : TileIndex [BLOCK_M] => (some (1 : ℝ) : WithBot ℝ)⟩,
      by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, BlockState.setReg_same]⟩
  · exact ⟨⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => (some (0 : ℝ) : WithBot ℝ)⟩,
      by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, BlockState.setReg_same]⟩
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

/-- A scalar-width unmasked load lands in `Tile.scalar (some …)` normal form
(the shape the streaming recipes consume for `q_scale`). -/
private theorem attnIO_scalar_load_form (u : BlockState) (name : RegName) (ptrs : Tile .ptr [])
    (hp : evalOp (Op.ref .ptr [] name) u = some ptrs) :
    evalOp (Op.load .real (.ptr (Op.ref .ptr [] name)) .none) u
      = some (Tile.scalar
          (some (u.readMem (ptrs.data PUnit.unit).1 (ptrs.data PUnit.unit).2))) := by
  rw [load_ptr_none_real (Op.ref .ptr [] name) u ptrs hp]
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Weak 20-statement prologue-prefix walk** from an arbitrary launch state:
pins the pointer/index registers the two terminal loads and the loop safety
walk consume. The 11-statement head is the exact stack's `undef`-free
`preLoop_scalars`. -/
private theorem attnIO_pre20W (s : BlockState) (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat) :
    ∃ s20, stepStmts (attnIOPre20 Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM
        (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N BLOCK_DMODEL) s = some s20
      ∧ s20.pids = s.pids ∧ s20.mem = s.mem ∧ s20.undef = s.undef
      ∧ s20.regs .nat [BLOCK_M] "offs_m"
          = some (Tile.vec (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val))
      ∧ s20.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
      ∧ (∃ mT : Tile .real [BLOCK_M], s20.regs .real [BLOCK_M] "m_i" = some mT)
      ∧ (∃ lT : Tile .real [BLOCK_M], s20.regs .real [BLOCK_M] "l_i" = some lT)
      ∧ (∃ aT : Tile .real [BLOCK_M, BLOCK_DMODEL],
          s20.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some aT)
      ∧ s20.regs .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"
          = some (attnIOrowPtrs s Q H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL)
      ∧ s20.regs .ptr [] "Q_scale_ptr"
          = some (attnIOsclPtr QScale
              (s.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_M - 1) / BLOCK_M) + s.pids 0))
      ∧ s20.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs"
          = some (attnIOkPtrs s K H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL 0)
      ∧ s20.regs .ptr [] "K_scale_ptr"
          = some (attnIOsclPtr KScale
              (s.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + 0))
      ∧ s20.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
          = some (attnIOvPtrs s V H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL 0)
      ∧ s20.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
          = some (attnIOrowPtrs s Out H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL) := by
  obtain ⟨s11, h11, hpids, hstartm, hqvk, hqso, hkso, hoffsm, hoffsn, hoffsk, hundef, hmem⟩ :=
    preLoop_scalars Q K V QScale KScale Out s stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks
      HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE
  have h11' : stepStmts (attnIOPreHead stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks)
      BLOCK_M BLOCK_N BLOCK_DMODEL) s = some s11 := h11
  unfold attnIOPre20
  rw [stepStmts.append_some h11']
  have hstartm' : s11.regs .nat [] "start_m" = some (Tile.scalar (s11.pids 0)) := by
    rw [hpids]; exact hstartm
  have hqvk' : s11.regs .nat [] "qvk_offset"
      = some (Tile.scalar (exBaseOffset s11 H stride_qz stride_qh)) := by
    simp only [exBaseOffset, hpids]; exact hqvk
  have hqso' : s11.regs .nat [] "q_scale_offset"
      = some (Tile.scalar (s11.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_M - 1) / BLOCK_M))) := by
    rw [hpids]; simpa only [exCdiv] using hqso
  have hkso' : s11.regs .nat [] "k_scale_offset"
      = some (Tile.scalar (s11.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N))) := by
    rw [hpids]; simpa only [exCdiv] using hkso
  have hoffsm' : s11.regs .nat [BLOCK_M] "offs_m"
      = some (Tile.vec (fun r : Fin BLOCK_M => s11.pids 0 * BLOCK_M + r.val)) := by
    rw [hpids]; exact hoffsm
  obtain ⟨s20, hTail, hpids20, hmem20, hundef20, hoffsm20, hoffsn20, hmi20, hli20, hacc20,
    hqp20, hqsp20, hkp20, hksp20, hvp20, hop20⟩ :=
    attnIO_preTail9W s11 Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM
      (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N BLOCK_DMODEL
      hstartm' hqvk' hqso' hkso' hoffsm' hoffsn hoffsk
  refine ⟨s20, hTail, ?_, ?_, ?_, ?_, hoffsn20, hmi20, hli20, hacc20, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpids20, hpids]
  · rw [hmem20, hmem]
  · rw [hundef20, hundef]
  · rw [hoffsm20, hpids]
  · rw [hqp20]; simp only [attnIOrowPtrs, exBaseOffset, hpids]
  · rw [hqsp20, hpids]
  · rw [hkp20]; simp only [attnIOkPtrs, exBaseOffset, hpids]
  · rw [hksp20, hpids]
  · rw [hvp20]; simp only [attnIOvPtrs, exBaseOffset, hpids]
  · rw [hop20]; simp only [attnIOrowPtrs, exBaseOffset, hpids]

/-- Safety-walk loop invariant: counter alignment/bound, exact pins for the
index vectors, the three streamed pointers and the output pointer (whose
addresses are the bound obligations), and bare existence for the value
registers. -/
private def attnIOSafeInv (K KScale V Out : RegionName) (s0 : BlockState)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks : Nat)
    (c : Nat) (s : BlockState) : Prop :=
  c % BLOCK_N = 0 ∧ c ≤ BLOCK_N * numKVBlocks ∧
  s.regs .nat [BLOCK_M] "offs_m"
      = some (Tile.vec (fun r : Fin BLOCK_M => s0.pids 0 * BLOCK_M + r.val)) ∧
  s.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) ∧
  (∃ mT : Tile .real [BLOCK_M], s.regs .real [BLOCK_M] "m_i" = some mT) ∧
  (∃ lT : Tile .real [BLOCK_M], s.regs .real [BLOCK_M] "l_i" = some lT) ∧
  (∃ aT : Tile .real [BLOCK_M, BLOCK_DMODEL],
    s.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some aT) ∧
  (∃ qT : Tile .real [BLOCK_M, BLOCK_DMODEL],
    s.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some qT) ∧
  (∃ qsv : ℝ, s.regs .real [] "q_scale" = some (Tile.scalar (some qsv))) ∧
  s.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs"
      = some (attnIOkPtrs s0 K H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c / BLOCK_N)) ∧
  s.regs .ptr [] "K_scale_ptr"
      = some (attnIOsclPtr KScale
          (s0.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + c / BLOCK_N)) ∧
  s.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
      = some (attnIOvPtrs s0 V H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c / BLOCK_N)) ∧
  s.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
      = some (attnIOrowPtrs s0 Out H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL)

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Weak prologue**: from an arbitrary launch state the full 22-statement
prologue steps to a state satisfying `attnIOSafeInv … 0`. -/
private theorem attnIO_pre22W (s : BlockState) (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat) :
    ∃ sp, stepStmts (attnIOPre22 Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM
        (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) s = some sp
      ∧ attnIOSafeInv K KScale V Out s stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
          BLOCK_DMODEL numKVBlocks 0 sp := by
  unfold attnIOPre22 attnIOPreLoads
  obtain ⟨s20, h20, hpids20, hmem20, hundef20, hoffsm, hoffsn, hmi, hli, hacc,
    hqp, hqsp, hkp, hksp, hvp, hop⟩ :=
    attnIO_pre20W s Q K V QScale KScale Out stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks
      HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE
  rw [stepStmts.append_some h20]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (attnIO_load_mask_realW (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs") _ _ _ _
      (by rw [evalOp_ref]; exact hqp)
      (qmask_eval s20 BLOCK_M BLOCK_DMODEL (BLOCK_N * numKVBlocks) HEAD_ACTIVE
        (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val) hoffsm)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (attnIO_scalar_load_form _ "Q_scale_ptr" _
      (by rw [evalOp_ref]; exact attnIO_regs_chain (by decide) hqsp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, Nat.zero_mod _, Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide) hoffsm)
  · exact attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide) hoffsn)
  · exact ⟨_, attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide) hmi.choose_spec)⟩
  · exact ⟨_, attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide) hli.choose_spec)⟩
  · exact ⟨_, attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide) hacc.choose_spec)⟩
  · exact ⟨_, attnIO_regs_chain (by decide) (BlockState.setReg_same _ _ _ _ _)⟩
  · exact ⟨_, BlockState.setReg_same _ _ _ _ _⟩
  · rw [attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide) hkp), Nat.zero_div]
  · rw [attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide) hksp), Nat.zero_div]
  · rw [attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide) hvp), Nat.zero_div]
  · exact attnIO_regs_chain (by decide) (attnIO_regs_chain (by decide) hop)

set_option maxHeartbeats 4000000 in
/-- The 20 load-free prologue statements are trace-safe at every state. -/
private theorem attnIO_pre20_stmt_safe (R : RoundingModel) (bounds : RegionBounds)
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM NC BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) :
    ∀ st ∈ attnIOPre20 Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM NC
        BLOCK_M BLOCK_N BLOCK_DMODEL,
      ∀ u : BlockState, Stmt.TraceSafeR R bounds st u := by
  intro st hst u
  simp only [attnIOPre20, attnIOPreHead, attnIOPreTail9, List.cons_append, List.nil_append,
    List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]

/-- `evalOpR` value of the shared `q`-load / terminal-store mask
`(offs_m[:,None] < N_CTX) & (arange(BLOCK_DMODEL) < HEAD_ACTIVE)[None,:]` under
an `offs_m` pin (the op is cast-free, so `R` is inert). -/
private theorem attnIO_qmask_evalR (R : RoundingModel) (u : BlockState)
    (BLOCK_M BLOCK_DMODEL NC HEAD_ACTIVE : Nat) (g : Fin BLOCK_M → Nat)
    (hoffsm : u.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec g)) :
    evalOpR R (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat NC))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL)
            (Op.constNat HEAD_ACTIVE)))) u
      = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (decide (g idx.1 < NC) && decide (idx.2.1.val < HEAD_ACTIVE))⟩ := by
  rw [show evalOpR R (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat NC))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL)
            (Op.constNat HEAD_ACTIVE)))) u
      = evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat NC))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL)
            (Op.constNat HEAD_ACTIVE)))) u
      from by simp only [evalOpR.eq_def, evalOp.eq_def]]
  exact qmask_eval u BLOCK_M BLOCK_DMODEL NC HEAD_ACTIVE g hoffsm

/-- `evalOpR` value of the in-loop `k_mask` op under `offs_n`/`start_n` pins
(via the exact `kmask_eval` recipe — the op is cast-free). -/
private theorem attnIO_kmask_evalR (R : RoundingModel) (u : BlockState)
    (BLOCK_DMODEL BLOCK_N NC SN HEAD_ACTIVE : Nat)
    (hoffsn : u.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOpR R (Op.boolAnd (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat NC) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))) u
      = some ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
          (decide (idx.2.1.val < NC - SN) && decide (idx.1.val < HEAD_ACTIVE))⟩ := by
  rw [show evalOpR R (Op.boolAnd (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat NC) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))) u
      = evalOp (Op.boolAnd (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat NC) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))) u
      from by simp only [evalOpR.eq_def, evalOp.eq_def]]
  exact kmask_eval u BLOCK_DMODEL BLOCK_N NC SN HEAD_ACTIVE hoffsn hsn

/-- `evalOpR` value of the in-loop `v`-load mask under `offs_n`/`start_n` pins. -/
private theorem attnIO_vmask_evalR (R : RoundingModel) (u : BlockState)
    (BLOCK_N BLOCK_DMODEL NC SN HEAD_ACTIVE : Nat)
    (hoffsn : u.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOpR R (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat NC) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))) u
      = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
          (decide (idx.1.val < NC - SN) && decide (idx.2.1.val < HEAD_ACTIVE))⟩ := by
  rw [show evalOpR R (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat NC) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))) u
      = evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat NC) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))) u
      from by simp only [evalOpR.eq_def, evalOp.eq_def]]
  exact vmask_eval u BLOCK_N BLOCK_DMODEL NC SN HEAD_ACTIVE hoffsn hsn

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Weak loop-body step**: from the shape pins alone the 19-statement body
steps successfully, advancing the three streamed pointers one block and
preserving every other pin (the `undef`-generic counterpart of the exact
stack's `attn_loopBody_steps`). -/
private theorem attnIO_attn_stepW (K KScale V Out : RegionName) (s0 : BlockState)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (hBN : 0 < BLOCK_N)
    (c : Nat) (s : BlockState) (hc : c < BLOCK_N * numKVBlocks)
    (hP : attnIOSafeInv K KScale V Out s0 stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
      BLOCK_DMODEL numKVBlocks c s) :
    ∃ s', stepStmts (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
        (s.setReg "start_n" .nat [] (Tile.scalar c)) = some s'
      ∧ attnIOSafeInv K KScale V Out s0 stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
          BLOCK_DMODEL numKVBlocks (c + BLOCK_N) s' := by
  obtain ⟨hmod, hle, hoffsm, hoffsn, ⟨mT, hmi⟩, ⟨lT, hli⟩, ⟨aT, hacc⟩, ⟨qT, hq⟩, ⟨qsv, hqs⟩,
    hKp, hKsp, hVp, hOp⟩ := hP
  have hax : 1 < [BLOCK_M].length.succ := by simp
  have hcBN : c / BLOCK_N * BLOCK_N = c := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)
  have hcdivlt : c / BLOCK_N < numKVBlocks := by
    have h1 : c / BLOCK_N * BLOCK_N < numKVBlocks * BLOCK_N := by
      rw [hcBN, Nat.mul_comm numKVBlocks BLOCK_N]
      exact hc
    exact lt_of_mul_lt_mul_right h1 (Nat.zero_le BLOCK_N)
  have hdiv : (c + BLOCK_N) / BLOCK_N = c / BLOCK_N + 1 := by
    conv_lhs => rw [show c = c / BLOCK_N * BLOCK_N from hcBN.symm]
    rw [Nat.add_div_right _ hBN, Nat.mul_div_cancel _ hBN]
  have hle' : c + BLOCK_N ≤ BLOCK_N * numKVBlocks := by
    have h1 : (c / BLOCK_N + 1) * BLOCK_N ≤ numKVBlocks * BLOCK_N :=
      Nat.mul_le_mul_right BLOCK_N hcdivlt
    calc c + BLOCK_N = c / BLOCK_N * BLOCK_N + BLOCK_N := by rw [hcBN]
      _ = (c / BLOCK_N + 1) * BLOCK_N := by ring
      _ ≤ numKVBlocks * BLOCK_N := h1
      _ = BLOCK_N * numKVBlocks := Nat.mul_comm _ _
  set u0 : BlockState := s.setReg "start_n" .nat [] (Tile.scalar c) with hu0
  set Kptr := attnIOkPtrs s0 K H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c / BLOCK_N)
    with hKptr
  set Ksptr := attnIOsclPtr KScale
    (s0.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + c / BLOCK_N) with hKsptr
  set Vptr := attnIOvPtrs s0 V H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c / BLOCK_N)
    with hVptr
  set kmaskT : Tile .bool [BLOCK_DMODEL, BLOCK_N] :=
    ⟨fun idx => (decide (idx.2.1.val < BLOCK_N * numKVBlocks - c)
      && decide (idx.1.val < HEAD_ACTIVE))⟩ with hkm
  set kloadT : Tile .real [BLOCK_DMODEL, BLOCK_N] :=
    ⟨fun i => if kmaskT.data i then some (u0.readMem (Kptr.data i).1 (Kptr.data i).2)
      else some (u0.undef (Kptr.data i).1 (Kptr.data i).2)⟩ with hkl
  set ksv : ℝ := u0.readMem (Ksptr.data PUnit.unit).1 (Ksptr.data PUnit.unit).2 with hksv
  set qkT : Tile .real [BLOCK_M, BLOCK_N] := Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (⟨fun i => FloatDType.real.cast FloatDType.real ((Tile.dot [] qT kloadT).data i)⟩
          : Tile .real [BLOCK_M, BLOCK_N])
        (Tile.scalar (some qsv : WithBot ℝ))) (Tile.scalar (some ksv : WithBot ℝ)) with hqk
  obtain ⟨rmaxT, hrm⟩ :
      ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) qkT = some t :=
    ⟨_, by
      unfold Tile.reduceMaxDrop
      rw [dif_pos (show 0 < TileShape.axisDim [BLOCK_M, BLOCK_N]
        (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) from hBN)]⟩
  set mijT : Tile .real [BLOCK_M] :=
    Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mT rmaxT)
      mT rmaxT with hmij
  set qk2T : Tile .real [BLOCK_M, BLOCK_N] :=
    Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT
      (Tile.expandDim ⟨1, hax⟩ mijT) with hqk2
  set pT : Tile .real [BLOCK_M, BLOCK_N] := Tile.uop WithBot.realExp2 qk2T with hpT
  set lijT : Tile .real [BLOCK_M] :=
    Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) pT with hlij
  set alphaT : Tile .real [BLOCK_M] := Tile.uop WithBot.realExp2
    (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mT mijT) with hal
  set acc1T : Tile .real [BLOCK_M, BLOCK_DMODEL] :=
    Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) aT
      (Tile.expandDim ⟨1, hax⟩ alphaT) with hacc1
  set vmaskT : Tile .bool [BLOCK_N, BLOCK_DMODEL] :=
    ⟨fun idx => (decide (idx.1.val < BLOCK_N * numKVBlocks - c)
      && decide (idx.2.1.val < HEAD_ACTIVE))⟩ with hvm
  set vloadT : Tile .real [BLOCK_N, BLOCK_DMODEL] :=
    ⟨fun i => if vmaskT.data i then some (u0.readMem (Vptr.data i).1 (Vptr.data i).2)
      else some (u0.undef (Vptr.data i).1 (Vptr.data i).2)⟩ with hvl
  unfold attnLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_n") u0 = some (Tile.scalar c) from by
      rw [evalOp_ref]; simp [hu0]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (kmask_eval _ BLOCK_DMODEL BLOCK_N (BLOCK_N * numKVBlocks) c HEAD_ACTIVE
      (by simp [hu0, hoffsn]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (attnIO_load_mask_realW (Op.ref .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs") _ _ Kptr kmaskT
      (by rw [evalOp_ref]; simp [hu0, hKp]) (by rw [evalOp_ref]; simp [hkm])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (attnIO_scalar_load_form _ "K_scale_ptr" Ksptr (by rw [evalOp_ref]; simp [hu0, hKsp])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (qk_op_eval _ BLOCK_M BLOCK_N BLOCK_DMODEL qT kloadT qsv ksv
      (by simp [hu0, hq]) (by simp [hkl]) (by simp [hu0, hqs]) (by simp [hksv])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mij_op_eval _ BLOCK_M BLOCK_N mT qkT rmaxT (by simp [hu0, hmi]) (by simp [hqk]) hrm))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (qk2_op_eval _ BLOCK_M BLOCK_N hax qkT mijT (by simp [hqk]) (by simp [hmij])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (p_op_eval _ BLOCK_M BLOCK_N qk2T (by simp [hqk2])))]
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .real [BLOCK_M] "l_ij"
    (Op.reduceSum (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false
      (Op.ref .real [BLOCK_M, BLOCK_N] "p")) _ lijT
    (lij_op_eval _ BLOCK_M BLOCK_N pT (by simp [hpT])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (alpha_op_eval _ BLOCK_M mT mijT (by simp [hu0, hmi]) (by simp [hmij])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (li_op_eval _ BLOCK_M lT alphaT lijT (by simp [hu0, hli]) (by simp [hal]) (by simp [hlij])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (acc1_op_eval _ BLOCK_M BLOCK_DMODEL hax aT alphaT (by simp [hu0, hacc]) (by simp [hal])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (attnIO_load_mask_realW (Op.ref .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs") _ _ Vptr vmaskT
      (by rw [evalOp_ref]; simp [hu0, hVp])
      (vmask_eval _ BLOCK_N BLOCK_DMODEL (BLOCK_N * numKVBlocks) c HEAD_ACTIVE
        (by simp [hu0, hoffsn]) (by simp [hu0]))))]
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some TileDType.fp16 [BLOCK_M, BLOCK_N] "p"
    (Op.castFloat .real .fp16 (Op.ref .real [BLOCK_M, BLOCK_N] "p")) _
    (⟨fun i => FloatDType.real.cast FloatDType.fp16 (pT.data i)⟩ : Tile .fp16 [BLOCK_M, BLOCK_N])
    (pfp16_op_eval _ BLOCK_M BLOCK_N pT (by simp [hpT])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (acc2_op_eval _ BLOCK_M BLOCK_N BLOCK_DMODEL acc1T pT vloadT
      (by simp [hacc1]) (by simp [hpT]) (by simp [hvl])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BLOCK_M] "m_ij") _ = some mijT from by
      rw [evalOp_ref]; simp [hmij]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (kptr_adv_eval _ BLOCK_DMODEL BLOCK_N BLOCK_N HEAD_DIM Kptr "K_ptrs" (by simp [hu0, hKp])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ksptr_adv_eval _ Ksptr "K_scale_ptr" (by simp [hu0, hKsp])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (kptr_adv_eval _ BLOCK_N BLOCK_DMODEL BLOCK_N HEAD_DIM Vptr "V_ptrs" (by simp [hu0, hVp])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, hle', ?_, ?_, ⟨mijT, ?_⟩,
    ⟨Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
      (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) lT alphaT) lijT, ?_⟩,
    ⟨Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      acc1T (Tile.dot [] pT vloadT), ?_⟩,
    ⟨qT, ?_⟩, ⟨qsv, ?_⟩, ?_, ?_, ?_, ?_⟩
  · rw [Nat.add_mod_right]; exact hmod
  · simp [hu0, hoffsm]
  · simp [hu0, hoffsn]
  · simp
  · simp
  · simp
  · simp [hu0, hq]
  · simp [hu0, hqs]
  · simp only [hdiv]
    rw [← attnIOkPtrs_succ s0 K H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c / BLOCK_N)]
    simp [hKptr]
  · simp only [hdiv, ← Nat.add_assoc]
    rw [← attnIOsclPtr_succ KScale
      (s0.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + c / BLOCK_N)]
    simp [hKsptr]
  · simp only [hdiv]
    rw [← attnIOvPtrs_succ s0 V H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c / BLOCK_N)]
    simp [hVptr]
  · simp [hu0, hOp]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Weak loop-body safety**: from an `attnIOSafeInv` state every statement of
the streamed body is trace-safe — the `k`/`k_scale`/`v` loads' active lanes are
bounded by the skin's `read2`/`read5`/`read3` window bounds. -/
private theorem attnIO_bodySafeW (R : RoundingModel) (bounds : RegionBounds)
    (K KScale V Out : RegionName) (s0 : BlockState)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (hBN : 0 < BLOCK_N)
    (c : Nat) (s : BlockState) (hc : c < BLOCK_N * numKVBlocks)
    (hP : attnIOSafeInv K KScale V Out s0 stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
      BLOCK_DMODEL numKVBlocks c s)
    (hbK : ∀ (t : Fin numKVBlocks) (j : Fin (BLOCK_DMODEL * BLOCK_N)),
      j.val % BLOCK_N < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE →
      s0.pids 1 / H * stride_qz + s0.pids 1 % H * stride_qh
        + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD_DIM < bounds K)
    (hbKS : ∀ (t : Fin numKVBlocks) (j : Fin 1),
      s0.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + t.val < bounds KScale)
    (hbV : ∀ (t : Fin numKVBlocks) (j : Fin (BLOCK_N * BLOCK_DMODEL)),
      j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks - t.val * BLOCK_N
        ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      s0.pids 1 / H * stride_qz + s0.pids 1 % H * stride_qh
        + (t.val * BLOCK_N + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL < bounds V) :
    Stmt.TraceSafeListR R bounds
      (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
      (s.setReg "start_n" .nat [] (Tile.scalar c)) := by
  obtain ⟨hmod, hle, hoffsm, hoffsn, ⟨mT, hmi⟩, ⟨lT, hli⟩, ⟨aT, hacc⟩, ⟨qT, hq⟩, ⟨qsv, hqs⟩,
    hKp, hKsp, hVp, hOp⟩ := hP
  have hcBN : c / BLOCK_N * BLOCK_N = c := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)
  have hcdivlt : c / BLOCK_N < numKVBlocks := by
    have h1 : c / BLOCK_N * BLOCK_N < numKVBlocks * BLOCK_N := by
      rw [hcBN, Nat.mul_comm numKVBlocks BLOCK_N]
      exact hc
    exact lt_of_mul_lt_mul_right h1 (Nat.zero_le BLOCK_N)
  unfold attnLoopBody
  -- (0) start_n = start_n (identity)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s1 h1 => ?_)
  obtain ⟨v0, hv0, rfl⟩ := stepStmtR_assign_inv h1
  rw [evalOpR_ref, BlockState.setReg_same] at hv0
  obtain rfl := Option.some.inj hv0
  -- (1) k_mask
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s2 h2 => ?_)
  obtain ⟨vkm, hvkm, rfl⟩ := stepStmtR_assign_inv h2
  rw [attnIO_kmask_evalR R _ BLOCK_DMODEL BLOCK_N (BLOCK_N * numKVBlocks) c HEAD_ACTIVE
    (by simp [hoffsn]) (BlockState.setReg_same _ _ _ _ _)] at hvkm
  obtain rfl := Option.some.inj hvkm
  -- (2) k = tl.load(K_ptrs, mask=k_mask): the read2 window bound
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s3 h3 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs hptrs idx hact
    rw [evalOpR_ref] at hptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] at hptrs
    rw [hKp] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hmasks, hidx⟩ := hact
    rw [evalOpR_ref, BlockState.setReg_same] at hmasks
    obtain rfl := Option.some.inj hmasks
    have hP2 : idx.2.1.val < BLOCK_N * numKVBlocks - c ∧ idx.1.val < HEAD_ACTIVE := by
      simpa using hidx
    have hbound := hbK ⟨c / BLOCK_N, hcdivlt⟩ (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) (by
      simp only [Lane2D.encode_div, Lane2D.encode_mod]
      rw [hcBN]
      exact ⟨hP2.1, hP2.2⟩)
    simpa [attnIOkPtrs, exBaseOffset, Lane2D.encode_div, Lane2D.encode_mod] using hbound
  obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv h3
  -- (3) k_scale = tl.load(K_scale_ptr): the read5 slot bound
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s4 h4 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs hptrs i _
    rw [evalOpR_ref] at hptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] at hptrs
    rw [hKsp] at hptrs
    obtain rfl := Option.some.inj hptrs
    have hbound := hbKS ⟨c / BLOCK_N, hcdivlt⟩ 0
    simpa [attnIOsclPtr] using hbound
  obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv h4
  -- (4)–(11): register-only assigns
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s5 h5 => ?_)
  obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv h5
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s6 h6 => ?_)
  obtain ⟨v6, -, rfl⟩ := stepStmtR_assign_inv h6
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s7 h7 => ?_)
  obtain ⟨v7, -, rfl⟩ := stepStmtR_assign_inv h7
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s8 h8 => ?_)
  obtain ⟨v8, -, rfl⟩ := stepStmtR_assign_inv h8
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s9 h9 => ?_)
  obtain ⟨v9, -, rfl⟩ := stepStmtR_assign_inv h9
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s10 h10 => ?_)
  obtain ⟨v10, -, rfl⟩ := stepStmtR_assign_inv h10
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s11 h11 => ?_)
  obtain ⟨v11, -, rfl⟩ := stepStmtR_assign_inv h11
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s12 h12 => ?_)
  obtain ⟨v12, -, rfl⟩ := stepStmtR_assign_inv h12
  -- (12) v = tl.load(V_ptrs, mask=…): the read3 window bound
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s13 h13 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, by simp [Op.SafeAtR.eq_def], ?_⟩
    intro ptrs hptrs idx hact
    rw [evalOpR_ref] at hptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] at hptrs
    rw [hVp] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hmasks, hidx⟩ := hact
    rw [attnIO_vmask_evalR R _ BLOCK_N BLOCK_DMODEL (BLOCK_N * numKVBlocks) c HEAD_ACTIVE
      (by simp [hoffsn])
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same])] at hmasks
    obtain rfl := Option.some.inj hmasks
    have hP2 : idx.1.val < BLOCK_N * numKVBlocks - c ∧ idx.2.1.val < HEAD_ACTIVE := by
      simpa using hidx
    have hbound := hbV ⟨c / BLOCK_N, hcdivlt⟩ (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) (by
      simp only [Lane2D.encode_div, Lane2D.encode_mod]
      rw [hcBN]
      exact ⟨hP2.1, hP2.2⟩)
    simpa [attnIOvPtrs, exBaseOffset, Lane2D.encode_div, Lane2D.encode_mod] using hbound
  obtain ⟨v13, -, rfl⟩ := stepStmtR_assign_inv h13
  -- (13)–(18): register-only assigns
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s14 h14 => ?_)
  obtain ⟨v14, -, rfl⟩ := stepStmtR_assign_inv h14
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s15 h15 => ?_)
  obtain ⟨v15, -, rfl⟩ := stepStmtR_assign_inv h15
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s16 h16 => ?_)
  obtain ⟨v16, -, rfl⟩ := stepStmtR_assign_inv h16
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s17 h17 => ?_)
  obtain ⟨v17, -, rfl⟩ := stepStmtR_assign_inv h17
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s18 h18 => ?_)
  obtain ⟨v18, -, rfl⟩ := stepStmtR_assign_inv h18
  exact Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (fun _ _ => Stmt.TraceSafeListR.nil_intro)

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Weak postLoop safety**: the `acc /= l_i` divide is register-only and the
masked terminal store's active lanes are bounded by the skin's `write` window
bound. -/
private theorem attnIO_postSafeW (R : RoundingModel) (bounds : RegionBounds)
    (K KScale V Out : RegionName) (s0 : BlockState)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (c : Nat) (s : BlockState)
    (hP : attnIOSafeInv K KScale V Out s0 stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
      BLOCK_DMODEL numKVBlocks c s)
    (hbO : ∀ j : Fin (BLOCK_M * BLOCK_DMODEL),
      s0.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
        ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      s0.pids 1 / H * stride_qz + s0.pids 1 % H * stride_qh
        + (s0.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
        < bounds Out) :
    Stmt.TraceSafeListR R bounds
      [attnAccAssign BLOCK_M BLOCK_DMODEL,
       attnStoreStmt BLOCK_M BLOCK_DMODEL BLOCK_N numKVBlocks HEAD_ACTIVE] s := by
  obtain ⟨hmod, hle, hoffsm, hoffsn, ⟨mT, hmi⟩, ⟨lT, hli⟩, ⟨aT, hacc⟩, ⟨qT, hq⟩, ⟨qsv, hqs⟩,
    hKp, hKsp, hVp, hOp⟩ := hP
  -- (1) acc /= l_i[:, None]: register-only
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [attnAccAssign, Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s1 h1 => ?_)
  rw [attnAccAssign] at h1
  obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv h1
  -- (2) the masked terminal store: the write window bound
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  simp only [attnStoreStmt, Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR,
    Op.SafeAtR.eq_def, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafeR]
  refine ⟨trivial, trivial, by simp [Op.SafeAtR.eq_def], ?_⟩
  intro ptrs hptrs idx hact
  rw [evalOpR_ref] at hptrs
  simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] at hptrs
  rw [hOp] at hptrs
  obtain rfl := Option.some.inj hptrs
  obtain ⟨masks, hmasks, hidx⟩ := hact
  rw [attnIO_qmask_evalR R _ BLOCK_M BLOCK_DMODEL (BLOCK_N * numKVBlocks) HEAD_ACTIVE
    (fun r : Fin BLOCK_M => s0.pids 0 * BLOCK_M + r.val)
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true];
        exact hoffsm)] at hmasks
  obtain rfl := Option.some.inj hmasks
  have hP2 : s0.pids 0 * BLOCK_M + idx.1.val < BLOCK_N * numKVBlocks
      ∧ idx.2.1.val < HEAD_ACTIVE := by simpa using hidx
  have hbound := hbO (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) (by
    simp only [Lane2D.encode_div, Lane2D.encode_mod]
    exact hP2)
  simpa [attnIOrowPtrs, exBaseOffset, Lane2D.encode_div, Lane2D.encode_mod] using hbound

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The `TraceSafeR` walk for the whole kernel**: the 20 load-free prologue
assigns are safe at every state, the `q`/`q_scale` loads are bounded by the
`read1`/`read4` windows at the walked prefix state, the KV loop runs
`Stmt.forRangeTraceSafeR_inv` over `attnIOSafeInv`, and the terminal store is
bounded by the `write` window. -/
private theorem attnIO_traceSafeR (R : RoundingModel) (hfp16 : R.round .fp16 = id)
    (bounds : RegionBounds) (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks : Nat)
    (s : BlockState) (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (hbQ : ∀ (t : Fin numKVBlocks) (j : Fin (BLOCK_M * BLOCK_DMODEL)),
      s.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
        ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh
        + (s.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL < bounds Q)
    (hbK : ∀ (t : Fin numKVBlocks) (j : Fin (BLOCK_DMODEL * BLOCK_N)),
      j.val % BLOCK_N < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE →
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh
        + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD_DIM < bounds K)
    (hbV : ∀ (t : Fin numKVBlocks) (j : Fin (BLOCK_N * BLOCK_DMODEL)),
      j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks - t.val * BLOCK_N
        ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh
        + (t.val * BLOCK_N + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL < bounds V)
    (hbQS : ∀ (t : Fin numKVBlocks) (j : Fin 1),
      s.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_M - 1) / BLOCK_M) + s.pids 0 < bounds QScale)
    (hbKS : ∀ (t : Fin numKVBlocks) (j : Fin 1),
      s.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + t.val < bounds KScale)
    (hbO : ∀ j : Fin (BLOCK_M * BLOCK_DMODEL),
      s.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
        ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh
        + (s.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
        < bounds Out) :
    ((attention_forward_triton_surface Q K V QScale KScale Out
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE STAGE).toAlgKernel).TraceSafeR R bounds s := by
  unfold Kernel.TraceSafeR
  rw [attnIO_body_split]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- the prologue: 20 register-only assigns then the two bounded loads
    unfold attnIOPre22
    refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
    · exact Stmt.TraceSafeListR.of_forall _ _
        (attnIO_pre20_stmt_safe R bounds Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM
          (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N BLOCK_DMODEL)
    · intro s2 hs2
      obtain ⟨s20, h20, hpids20, hmem20, hundef20, hoffsm, hoffsn, hmi, hli, hacc,
        hqp, hqsp, hkp, hksp, hvp, hop⟩ :=
        attnIO_pre20W s Q K V QScale KScale Out stride_qz stride_qh Z H BLOCK_M BLOCK_N
          numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE
      rw [attnIO_stepStmtsR_castFree_of_stmts R _
          (fun st hst u => attnIO_pre22_stmt_castFree R Q K V QScale KScale Out stride_qz
            stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N BLOCK_DMODEL
            HEAD_ACTIVE st (by
              unfold attnIOPre22
              exact List.mem_append_left _ hst) u) s, h20] at hs2
      obtain rfl := Option.some.inj hs2
      unfold attnIOPreLoads
      -- the q load: read1 window bound
      refine Stmt.TraceSafeListR.cons_intro ?_ (fun s21 h21 => ?_)
      · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
          MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
        refine ⟨trivial, by simp [Op.SafeAtR.eq_def], ?_⟩
        intro ptrs hptrs idx hact
        rw [evalOpR_ref] at hptrs
        rw [hqp] at hptrs
        obtain rfl := Option.some.inj hptrs
        obtain ⟨masks, hmasks, hidx⟩ := hact
        rw [attnIO_qmask_evalR R _ BLOCK_M BLOCK_DMODEL (BLOCK_N * numKVBlocks) HEAD_ACTIVE
          (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val) hoffsm] at hmasks
        obtain rfl := Option.some.inj hmasks
        have hP2 : s.pids 0 * BLOCK_M + idx.1.val < BLOCK_N * numKVBlocks
            ∧ idx.2.1.val < HEAD_ACTIVE := by simpa using hidx
        have hbound := hbQ ⟨0, hnum⟩ (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) (by
          simp only [Lane2D.encode_div, Lane2D.encode_mod]
          exact hP2)
        simpa [attnIOrowPtrs, exBaseOffset, Lane2D.encode_div, Lane2D.encode_mod] using hbound
      obtain ⟨vq, -, rfl⟩ := stepStmtR_assign_inv h21
      -- the q_scale load: read4 slot bound
      refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
      refine ⟨trivial, trivial, ?_⟩
      intro ptrs hptrs i _
      rw [evalOpR_ref] at hptrs
      rw [attnIO_regs_chain (by decide) hqsp] at hptrs
      obtain rfl := Option.some.inj hptrs
      have hbound := hbQS ⟨0, hnum⟩ 0
      simpa [attnIOsclPtr] using hbound
  · -- after the prologue: the KV loop, then the postLoop
    intro sp hsp
    obtain ⟨spW, hpreW, hinv0⟩ :=
      attnIO_pre22W s Q K V QScale KScale Out stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks
        HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE
    rw [attnIO_pre22_castFree R Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM
        (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE s, hpreW] at hsp
    obtain rfl := Option.some.inj hsp
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun s3 hs3 => ?_)
    · -- the KV loop is trace-safe (invariant principle over `attnIOSafeInv`)
      show Stmt.TraceSafeR R bounds _ spW
      simp only [Stmt.TraceSafeR]
      refine Stmt.forRangeTraceSafeR_inv R bounds "start_n" (BLOCK_N * numKVBlocks) BLOCK_N
        (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
        (attnIOSafeInv K KScale V Out s stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
          BLOCK_DMODEL numKVBlocks)
        ?_ 0 spW hinv0
      intro c st hc hPc
      refine ⟨attnIO_bodySafeW R bounds K KScale V Out s stride_qz stride_qh H HEAD_DIM
        BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks hBN c st hc hPc hbK hbKS hbV, ?_⟩
      obtain ⟨st', hstep, hPc'⟩ := attnIO_attn_stepW K KScale V Out s stride_qz stride_qh H
        HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks hBN c st hc hPc
      exact ⟨st', by
        rw [attnIO_loopBody_castFree R hfp16 BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM
          numKVBlocks]
        exact hstep, hPc'⟩
    · -- identify the post-loop state and finish on the terminal store
      obtain ⟨final, sfin, hLoop, hfinal, hPfin⟩ :=
        forRange_inv (idx := "start_n") (start := 0) (stop := BLOCK_N * numKVBlocks)
          (step := BLOCK_N)
          (body := attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
          (P := attnIOSafeInv K KScale V Out s stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
            BLOCK_DMODEL numKVBlocks)
          (s_init := spW) hBN.ne' hinv0
          (fun c st hc hPc => attnIO_attn_stepW K KScale V Out s stride_qz stride_qh H HEAD_DIM
            BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks hBN c st hc hPc)
      rw [show stepStmtR R (Stmt.forRange "start_n" 0 (BLOCK_N * numKVBlocks) BLOCK_N
            (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)) spW
          = some sfin from by
        rw [attnIO_loopStmt_castFree R hfp16 BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM
          numKVBlocks spW]
        exact hLoop] at hs3
      obtain rfl := Option.some.inj hs3
      exact attnIO_postSafeW R bounds K KScale V Out s stride_qz stride_qh H HEAD_DIM
        BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks final sfin hPfin hbO

/-! ### The postLoop run and its frame

`attn_postLoop` delivers the readback value at each active lane; the skin
additionally needs termination and the per-cell frame (every cell outside the
masked `Out` window is untouched). Both come from stepping the two postLoop
statements explicitly. -/

/-- A `P`-masked single-region `writeMem` scatter preserves every cell no
active lane hits. -/
private theorem attnIO_foldl_writeMem_frame_masked {α : Type} (region : RegionName)
    (offFn : α → Nat) (valFn : α → ℝ) (P : α → Prop) [DecidablePred P] :
    ∀ (l : List α) (s : BlockState) (r : RegionName) (o : Nat),
      (r = region → ∀ k ∈ l, P k → offFn k ≠ o) →
      ((l.foldl (fun acc k => if P k then acc.writeMem region (offFn k) (valFn k) else acc) s).mem r o
        = s.mem r o)
  | [], _, _, _, _ => rfl
  | k :: rest, s, r, o, h => by
      rw [List.foldl_cons]
      by_cases hPk : P k
      · rw [if_pos hPk,
          attnIO_foldl_writeMem_frame_masked region offFn valFn P rest _ r o
            (fun hr k' hk' hP' => h hr k' (List.mem_cons_of_mem _ hk') hP'),
          BlockState.writeMem_mem]
        rw [if_neg (fun hro => h hro.1 k List.mem_cons_self hPk hro.2.symm)]
      · rw [if_neg hPk]
        exact attnIO_foldl_writeMem_frame_masked region offFn valFn P rest s r o
          (fun hr k' hk' hP' => h hr k' (List.mem_cons_of_mem _ hk') hP')

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **PostLoop run + frame**: from the loop-exit register pins the `acc /= l_i`
rescale and the masked terminal store run to completion, and every cell outside
the write-active `Out` window is untouched. -/
private theorem attnIO_postLoop_runW (Out : RegionName) (s0 : BlockState)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (st : BlockState)
    (hoffsm : st.regs .nat [BLOCK_M] "offs_m"
      = some (Tile.vec (fun r : Fin BLOCK_M => s0.pids 0 * BLOCK_M + r.val)))
    (hOp : st.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
      = some (attnIOrowPtrs s0 Out H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL))
    (lT : Tile .real [BLOCK_M]) (hli : st.regs .real [BLOCK_M] "l_i" = some lT)
    (aT : Tile .real [BLOCK_M, BLOCK_DMODEL])
    (hacc : st.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some aT) :
    ∃ sfin, stepStmts [attnAccAssign BLOCK_M BLOCK_DMODEL,
        attnStoreStmt BLOCK_M BLOCK_DMODEL BLOCK_N numKVBlocks HEAD_ACTIVE] st = some sfin
      ∧ ∀ r o,
          (r = Out → ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
            (s0.pids 0 * BLOCK_M + idx.1.val < BLOCK_N * numKVBlocks
              ∧ idx.2.1.val < HEAD_ACTIVE) →
            o ≠ exBaseOffset s0 H stride_qz stride_qh
              + (s0.pids 0 * BLOCK_M + idx.1.val) * HEAD_DIM + idx.2.1.val) →
          sfin.mem r o = st.mem r o := by
  have hax : 1 < [BLOCK_M].length.succ := by simp
  set acc' : Tile .real [BLOCK_M, BLOCK_DMODEL] :=
    Tile.bop NumericDType.real.div (Broadcast.consSame (Broadcast.consR Broadcast.nil)) aT
      (Tile.expandDim ⟨1, hax⟩ lT) with hacc'
  have hexpN : evalOp (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "l_i")) st
      = some (Tile.expandDim ⟨1, hax⟩ lT) := by rw [evalOp_expandDim]; simp [hli]
  have hexp2 : @evalOp TileDType.real [BLOCK_M, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "l_i")) st
      = some (Tile.expandDim ⟨1, hax⟩ lT) := hexpN
  have hAccEval : evalOp (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "l_i"))) st = some acc' := by
    rw [evalOp_div]
    simp only [evalOp_ref, hacc, hexp2, Option.bind_eq_bind, Option.bind_some]
    rfl
  set st1 := st.setReg "acc" .real [BLOCK_M, BLOCK_DMODEL] acc' with hst1
  set oOffFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx => exBaseOffset s0 H stride_qz stride_qh
      + (s0.pids 0 * BLOCK_M + idx.1.val) * HEAD_DIM + idx.2.1.val with hoOffFn
  set P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx => s0.pids 0 * BLOCK_M + idx.1.val < BLOCK_N * numKVBlocks
      ∧ idx.2.1.val < HEAD_ACTIVE with hPdef
  have hacc2 : st1.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some acc' := by
    rw [hst1]; simp only [BlockState.setReg_same]
  have hopEval : evalOp (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr") st1
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out.cast, oOffFn idx)⟩
          : Tile .ptr [BLOCK_M, BLOCK_DMODEL]) := by
    rw [evalOp_ref, hst1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hOp]
    rfl
  have hmaskEval : evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
      (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
        (Op.constNat (BLOCK_N * numKVBlocks)))
      (Op.expandDim ⟨0, by simp⟩
        (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))) st1
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => decide (P idx)⟩
          : Tile .bool [BLOCK_M, BLOCK_DMODEL]) := by
    rw [qmask_eval st1 BLOCK_M BLOCK_DMODEL (BLOCK_N * numKVBlocks) HEAD_ACTIVE
      (fun r : Fin BLOCK_M => s0.pids 0 * BLOCK_M + r.val)
      (by rw [hst1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffsm)]
    refine congrArg some ?_
    ext idx
    simp [hPdef]
  have hstore : stepStmt (attnStoreStmt BLOCK_M BLOCK_DMODEL BLOCK_N numKVBlocks HEAD_ACTIVE) st1
      = some ((TileShape.allIndices [BLOCK_M, BLOCK_DMODEL]).foldl
          (fun acc idx => if P idx then acc.writeMem Out (oOffFn idx) ((acc'.data idx).unbotD 0)
            else acc) st1) := by
    simp only [attnStoreStmt, stepStmt, evalOp_ref, hacc2, hopEval, hmaskEval,
      Option.bind_eq_bind, Option.bind_some, Option.map_some, decide_eq_true_eq]
    refine congrArg some ?_
    refine List.foldl_ext _ _ st1 ?_
    intro acc idx _
    by_cases hk : P idx
    · simp only [if_pos hk, Region.cast_id, BlockState.writeMemTyped_real,
        FloatDType.real_storeValue]
    · simp only [if_neg hk]
  rw [attnAccAssign, stepStmts.cons_some (stepStmt_assign_eq_some hAccEval),
    stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro r o hguard
  rw [attnIO_foldl_writeMem_frame_masked Out oOffFn (fun idx => (acc'.data idx).unbotD 0) P
    (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL]) st1 r o
    (fun hr k _ hPk heq => hguard hr k hPk heq.symm)]
  rw [hst1]
  simp only [BlockState.setReg_mem]

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The `⊨[R]` io headline — on the `StreamMasked3DKernelIO₅` skin.** On its
five-stream single-store signature (`Q`/`K`/`V` tile channels plus the two
scalar-width `Q_scale`/`K_scale` channels), `_attn_fwd` implements the genuine
closed form: output lane `j = (i, e)` of `Out` holds
`attnFwdIOOutSpec` — base-2 per-key-scale attention over the tiles assembled
from the five streams, with `keyScale j = Q_scale · K_scale[j / BLOCK_N]` — for
every rounding model that is trivial on the fp16 grid.

**Hypothesis provenance**:
* `hfp16` pins `R.round .fp16 = id` — loop-body statement 14
  (`p = (p).to(tl.float16)`, and the `out_dtype=tl.float16` dot it feeds) is an
  *in-loop* rounding event outside the skin's single-boundary-round shape;
  this is the file's declared fp16 modeling boundary (the exact stack above
  already treats this cast as the identity), now explicit as a hypothesis.
* `hBD`/`hBN`/`hnum` positivity shape the KV walk (`N_CTX = BLOCK_N ·
  numKVBlocks > 0`) — inherited from the exact headline
  `attention_forward_triton_closed_form_correct`.
* `hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL` — the head mask lives inside the
  block width; it is what makes the stream lane `i·BLOCK_DMODEL + e` exist for
  every head-active `e`.
* `hHD : HEAD_ACTIVE ≤ HEAD_DIM` — contiguous-row `Out` window injectivity
  (host launches use `HEAD_DIM = BLOCK_DMODEL`), inherited verbatim.

The exact headline's `hundef` is **not** a hypothesis here — the skin's Hoare
triple carries the `undef` pin itself. The output grid is the `.real` default:
the store-side `.to(Out.type.element_ty)` cast **erases at translation**
(`attnStoreStmt` is a bare `Stmt.store .real …`), so at every such `R` the
terminal cells carry the exact fold values. -/
specification attention_forward_triton_io_correctness (R : RoundingModel)
    (hfp16 : R.round .fp16 = id)
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat)
    (hBD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL) (hHD : HEAD_ACTIVE ≤ HEAD_DIM) :
    attnFwdIO Q K V QScale KScale Out stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N
        BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks ⊨[R]
      fun p₀ _ _ xs ys zs ws vs j =>
        if h : j.val % BLOCK_DMODEL < HEAD_ACTIVE then
          attnFwdIOOutSpec p₀ BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks xs ys zs ws vs
            ((Lane2D.decode j).1, ⟨j.val % BLOCK_DMODEL, h⟩, PUnit.unit)
        else 0 := by
  refine StreamMasked3DKernelIO₅.ImplementsR.intro _ ?_ ?_ ?_
  · exact attnIO_flattenOk Q K V QScale KScale Out stride_qz stride_qh Z H BLOCK_M BLOCK_N
      numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE
  · -- the trace-safety walk
    intro bounds st xs ys zs ws vs _hx _hy _hz _hw _hv hbr1 hbr2 hbr3 hbr4 hbr5 hbw
    simp only [attnFwdIO] at hbr1 hbr2 hbr3 hbr4 hbr5 hbw ⊢
    exact attnIO_traceSafeR R hfp16 bounds Q K V QScale KScale Out stride_qz stride_qh Z H
      HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks st hBN hnum
      (fun t j hm => hbr1 t j hm) (fun t j hm => hbr2 t j hm) (fun t j hm => hbr3 t j hm)
      (fun t j => hbr4 t j trivial) (fun t j => hbr5 t j trivial) (fun j hm => hbw j hm)
  · -- the rounded Hoare triple: the exact invariant stack + the cast-free collapse
    intro s₀ xs ys zs ws vs hu hx hy hz hw hv
    simp only [attnFwdIO] at hx hy hz hw hv ⊢
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hu]
    have hnB : 1 ≤ numKVBlocks := hnum
    -- the stream-pin tile bridges
    have hqTeq := attnIO_qTm_eq s₀ Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE numKVBlocks hnum hActiveLe xs (fun t j hm => hx t j hm)
    have hkTeq := attnIO_kTm_eq s₀ K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE numKVBlocks hBN hActiveLe ys (fun t j hm => hy t j hm)
    have hvTeq := attnIO_vTm_eq s₀ V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE numKVBlocks hBN hActiveLe zs (fun t j hm => hz t j hm)
    have hksEq := attnIO_keyScale_eq s₀ QScale KScale BLOCK_M BLOCK_N numKVBlocks hnum hBN ws vs
      (fun t j => hw t j trivial) (fun t j => hv t j trivial)
    -- the exact run: preLoop invariant, the KV `forRange`, postLoop (+ frame)
    obtain ⟨sF, hexec, hval, hframe⟩ :=
      attention_forward_triton_exec_reduction Q K V QScale KScale Out s₀
        stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE
        STAGE hBN
        (attnInvariant Q K V QScale KScale Out s₀ stride_qz stride_qh H BLOCK_M BLOCK_N
          numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE)
        (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
        (attnAccAssign BLOCK_M BLOCK_DMODEL)
        (attnStoreStmt BLOCK_M BLOCK_DMODEL BLOCK_N numKVBlocks HEAD_ACTIVE)
        (attn_body_split Q K V QScale KScale Out stride_qz stride_qh Z H BLOCK_M BLOCK_N
          numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE)
        (preLoop Q K V QScale KScale Out s₀ stride_qz stride_qh H BLOCK_M BLOCK_N numKVBlocks
          HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE hundef')
        (fun i st hinv => by
          simp only [attnInvariant] at hinv
          obtain ⟨_, hieq, hcle, _⟩ := hinv
          calc i = i / BLOCK_N * BLOCK_N := hieq
            _ ≤ numKVBlocks * BLOCK_N := Nat.mul_le_mul_right _ hcle
            _ = BLOCK_N * numKVBlocks := Nat.mul_comm _ _)
        (fun i st hlt hinv => attn_step Q K V QScale KScale Out s₀ stride_qz stride_qh H
          BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE hBN hActiveLe i st hlt
          hinv)
        (post := fun sfin =>
          (∀ (idx : TileIndex [BLOCK_M, BLOCK_DMODEL])
              (hact : exMIndex s₀ BLOCK_M idx.1 < BLOCK_N * numKVBlocks
                ∧ idx.2.1.val < HEAD_ACTIVE),
            sfin.readMem Out
                (exOutOffset s₀ H stride_qz stride_qh HEAD_DIM 1 BLOCK_M BLOCK_DMODEL idx)
              = attentionRealBase2PerKeyScale
                  (exQTile s₀ Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE)
                  (exKTile s₀ K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
                  (exVTile s₀ V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
                  (exKeyScale s₀ QScale KScale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N
                    (BLOCK_N * numKVBlocks))
                  (idx.1, ⟨idx.2.1.val, hact.2⟩, PUnit.unit))
          ∧ (∀ r o,
              (r = Out → ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
                (s₀.pids 0 * BLOCK_M + idx.1.val < BLOCK_N * numKVBlocks
                  ∧ idx.2.1.val < HEAD_ACTIVE) →
                o ≠ exBaseOffset s₀ H stride_qz stride_qh
                  + (s₀.pids 0 * BLOCK_M + idx.1.val) * HEAD_DIM + idx.2.1.val) →
              sfin.mem r o = s₀.mem r o))
        (fun st hinv => by
          have hinv' := hinv
          simp only [attnInvariant] at hinv'
          obtain ⟨-, -, -, -, hli, hacc, -, -, -, hoffsm, -, hOp, -, -, -, -, hmemst⟩ := hinv'
          obtain ⟨sfin, hstepFin, hfr⟩ :=
            attnIO_postLoop_runW Out s₀ stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
              BLOCK_DMODEL HEAD_ACTIVE numKVBlocks st
              (by simpa only [exMIndex] using hoffsm)
              (by simpa only [attnIOrowPtrs, exMIndex] using hOp)
              _ hli _ hacc
          refine ⟨sfin, hstepFin, ?_, ?_⟩
          · intro idx hact
            obtain ⟨sfin', hstep', hvalue⟩ :=
              attn_postLoop Q K V QScale KScale Out s₀ stride_qz stride_qh H BLOCK_M BLOCK_N
                numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE hBN hnB hHD idx hact st hinv
            rw [hstepFin] at hstep'
            obtain rfl := Option.some.inj hstep'
            exact hvalue
          · intro r o hguard
            rw [hfr r o hguard, hmemst])
    refine ⟨sF, ?_, ?_, ?_⟩
    · -- termination under `execR R` (everything cast-free under `hfp16`)
      show execR R _ s₀ = some sF
      rw [attnIO_execR_eq_exec R hfp16 Q K V QScale KScale Out stride_qz stride_qh Z H BLOCK_M
        BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE s₀]
      exact hexec
    · -- `Out` readback: the streamed closed form on every write-active lane
      intro j hj
      have hjBD : j.val % BLOCK_DMODEL < HEAD_ACTIVE := hj.2
      have hact : exMIndex s₀ BLOCK_M (Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).1
            < BLOCK_N * numKVBlocks
          ∧ (Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).2.1.val < HEAD_ACTIVE := by
        simp only [exMIndex, Lane2D.decode_row, Lane2D.decode_col]
        exact hj
      have hOj := hval (Lane2D.decode j) hact
      have hspec : attentionRealBase2PerKeyScale
            (exQTile s₀ Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE)
            (exKTile s₀ K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (exVTile s₀ V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (exKeyScale s₀ QScale KScale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N
              (BLOCK_N * numKVBlocks))
            ((Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).1,
              ⟨(Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).2.1.val, hact.2⟩, PUnit.unit)
          = attnFwdIOOutSpec (s₀.pids 0) BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks
              xs ys zs ws vs
              ((Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).1,
                ⟨j.val % BLOCK_DMODEL, hjBD⟩, PUnit.unit) := by
        unfold attnFwdIOOutSpec
        rw [← hqTeq, ← hkTeq, ← hvTeq, ← hksEq]
        exact attnIO_spec_row_congr _ _ _ _ _ _ _ (fun e' =>
          (qTileMasked_active s₀ Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE
            (BLOCK_N * numKVBlocks)
            ((Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).1, e', PUnit.unit) hact.1).symm)
      rw [show s₀.pids 1 / H * stride_qz + s₀.pids 1 % H * stride_qh
              + (s₀.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
            = exOutOffset s₀ H stride_qz stride_qh HEAD_DIM 1 BLOCK_M BLOCK_DMODEL
                (Lane2D.decode j) from by
        simp only [exOutOffset, exBaseOffset, exMIndex, Lane2D.decode_row, Lane2D.decode_col,
          Nat.mul_one]]
      rw [BlockState.readMemAs_real, hOj, dif_pos hjBD, hspec]
      simp [FloatDType.ofReal]
    · -- the frame: cells outside the write-active `Out` window are untouched
      intro r' o' hcond
      refine hframe r' o' ?_
      intro hr idx hPidx
      rcases hcond with hne | hguard
      · exact absurd hr hne
      · intro heq
        refine hguard (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) ?_ (heq.trans ?_)
        · simp only [Lane2D.encode_div, Lane2D.encode_mod]
          exact hPidx
        · simp only [exBaseOffset, Lane2D.encode_div, Lane2D.encode_mod]

end IOFace

end VeriTile.Bench.TritonBenchG.AttentionForwardTriton
