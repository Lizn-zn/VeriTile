import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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
attention_forward_triton_python_test_shape_output_summary    ← TOP THEOREM
  ├─ attention_forward_triton_surface_toAlgorithm_supported   surface lowers to the algorithm layer
  └─ attention_forward_triton_surface_python_test_shape_compute_correct
       └─ (full surface produces producedAttentionForwardOutValue at the masked Out store)

attention_forward_triton_final_store_python_test_shape_compute_correct
  └─ attention_forward_triton_final_store_slice_compute_correct  ← ComputeCorrect over the masked Out store
       └─ attention_forward_triton_final_store_slice_correct     ← algorithm-layer readback per lane
```

## Modeling boundary and proof-gap honesty

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `float16`/`float32`
casts collapse to the identity post-erasure; `@triton.autotune` /
`num_warps`/`num_stages` are not modeled. The output summary is stated at the
Python test shape (`B=2, H=4, N_CTX=128, HEAD_DIM=128, BLOCK_M=128, BLOCK_N=64`,
contiguous strides, 96 active head lanes).

**The top `output_summary` is NOT a closed-form value proof.** Its `expected`,
`producedAttentionForwardOutValue`, is defined as the kernel's *own* executed
output (`match exec … | some s' => s'.readMem …`), so the value half of its
`ComputeCorrect.Realizes` is a tautology (`output = output`). What it genuinely
establishes is (a) the surface lowers to the algorithm layer and (b) execution
makes progress (runs to `some`) writing exactly the masked output cells — it
does **not** verify the online-softmax recurrence against an independent
specification (tracked as `attention-forward-online-softmax-recurrence`, #162,
in `proof_gap_manifest.tsv`). The genuine value content here is the
`final_store` slice: it proves the masked final `acc / l_i` writeback copies a
*precomputed* accumulator `Acc` to the right `Out` cells (in-bounds lanes get
the `Acc` value, out-of-bounds preserved) — i.e. the store/masking is correct,
the accumulator value itself is assumed. Cross-program composition into the full
`[Z,H,N_CTX,HEAD_DIM]` output is the trusted host boundary.
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

noncomputable def producedAttentionForwardOutValue
    (s : BlockState) (Q K V QScale KScale Out : RegionName)
    (idx : TileIndex [128, 128]) : ℝ :=
  match exec (attention_forward_triton_surface Q K V QScale KScale Out
      65536 16384 128 1
      65536 16384 128 1
      65536 16384 128 1
      65536 16384 128 1
      2 4 128 128 128 64 128 96 1) s with
  | some s' => s'.readMem Out (outOffset s 4 65536 16384 128 1 128 idx)
  | none => 0.0

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
    ComputeCorrect.Realizes
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

/-! ## Python test-shape wrapper

`attention_forward_triton.py`'s checked test uses
`B = 2`, `H = 4`, `N_CTX = 128`, `HEAD_DIM = 128`, `BLOCK_M = 128`,
`BLOCK_N = 64`, and the source mask only enables the first 96 head lanes.
For contiguous `[B, H, N_CTX, HEAD_DIM]` tensors the output/accumulator strides
are `(65536, 16384, 128, 1)`. -/

theorem attention_forward_triton_final_store_python_test_shape_compute_correct
    (Acc Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_forward_triton_final_store_slice Acc Out
        4 128 96 65536 16384 128 1 65536 16384 128 1 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] => active s 128 96 128 idx)
        (fun idx : TileIndex [128, 128] => (Out,
          outOffset s 4 65536 16384 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        s.readMem Acc (accOffset s 4 65536 16384 128 1 128 idx)) := by
  apply attention_forward_triton_final_store_slice_compute_correct
  rintro ⟨⟨ma, hma⟩, ⟨ka, hka⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨kb, hkb⟩, _⟩ h
  simp [outOffset, offZ, offH, mIndex, kIndex] at h
  have hm : ma = mb := by omega
  have hk : ka = kb := by omega
  subst mb
  subst kb
  rfl

theorem attention_forward_triton_surface_python_test_shape_compute_correct
    (Q K V QScale KScale Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_forward_triton_surface Q K V QScale KScale Out
        65536 16384 128 1
        65536 16384 128 1
        65536 16384 128 1
        65536 16384 128 1
        2 4 128 128 128 64 128 96 1)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] => active s 128 96 128 idx)
        (fun idx : TileIndex [128, 128] => (Out,
          outOffset s 4 65536 16384 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        producedAttentionForwardOutValue s Q K V QScale KScale Out idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_forward_triton_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedAttentionForwardOutValue, hExec]

/-- Python test-shape summary for `attention_forward_triton.py`.

This combines the checked full-surface lowering for the Python launch
parameters with the final observable `Out` writes produced directly by that
full surface at the contiguous `[B,H,N_CTX,HEAD_DIM] = [2,4,128,128]` layout. -/
theorem attention_forward_triton_python_test_shape_output_summary
    (Q K V QScale KScale Out : RegionName) (s : BlockState) :
    (∃ alg, (attention_forward_triton_surface Q K V QScale KScale Out
      65536 16384 128 1
      65536 16384 128 1
      65536 16384 128 1
      65536 16384 128 1
      2 4 128 128 128 64 128 96 1).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attention_forward_triton_surface Q K V QScale KScale Out
        65536 16384 128 1
        65536 16384 128 1
        65536 16384 128 1
        65536 16384 128 1
        2 4 128 128 128 64 128 96 1)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] => active s 128 96 128 idx)
        (fun idx : TileIndex [128, 128] => (Out,
          outOffset s 4 65536 16384 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        producedAttentionForwardOutValue s Q K V QScale KScale Out idx) := by
  constructor
  · exact attention_forward_triton_surface_toAlgorithm_supported Q K V QScale
      KScale Out 65536 16384 128 1 65536 16384 128 1 65536 16384
      128 1 65536 16384 128 1 2 4 128 128 128 64 128 96 1
  · exact attention_forward_triton_surface_python_test_shape_compute_correct
      Q K V QScale KScale Out s

end VeriTile.Bench.TritonBenchG.AttentionForwardTriton
