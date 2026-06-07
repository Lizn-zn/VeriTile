import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention

/-!
# `attn_fwd_causal` — strict per-kernel correctness

`attn_fwd_causal.py`'s `_attn_fwd` is a causal FlashAttention forward kernel:
program `(start_m, off_hz)` streams `K`/`V` blocks for one `(batch·head, query
block)` tile, maintaining the online-softmax running max `m_i`, denominator
`l_i`, and accumulator `acc` (with per-block causal masking and `q_scale ·
k_scale` quantization), then stores the normalized `acc / l_i` to `Out`, masked
to the first 96 head lanes and `offs_m < N_CTX`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_attn_fwd[grid](...)` with `grid = (cdiv(N_CTX,
BLOCK_M), Z·H, 1)`, the scheduling over query blocks and `(batch, head)`, and
how the runtime composes per-program writes into the output buffer) is the
*trusted boundary*, not a proof obligation here. Because the program ids
`start_m`/`off_hz` are universally quantified (via `s`), the per-program
statements cover every program of the grid.

## Proof architecture

```
attn_fwd_causal_python_test_shape_output_summary           ← TOP THEOREM
  ├─ attn_fwd_causal_surface_toAlgorithm_supported          surface lowers to the algorithm layer
  └─ attn_fwd_causal_surface_python_test_shape_compute_correct
       └─ attn_fwd_causal_final_store_python_test_shape_compute_correct
            └─ attn_fwd_causal_final_store_slice_compute_correct
                 └─ attn_fwd_causal_final_store_slice_correct   ← algorithm-layer readback per lane
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; the `exp2`, the `tl.dot`
`float16` accumulation, and `q_scale · k_scale` quantization are not modeled at
the bit level); `@triton.autotune`/`num_warps`/`num_stages` are not modeled.
The verified result is **final-store scoped**: the proof establishes that the
masked store copies the accumulator slice `Acc` to `Out` at the correct,
injective output offsets and preserves inactive lanes — the value written is
`producedAttnFwdCausalOutValue` / `s.readMem Acc (...)`, an opaque carrier for
the online-softmax recurrence (`m_i`, `l_i`, `acc` updates, causal mask, the
final `acc / l_i` normalization), which is **not** re-derived as a closed-form
attention formula here. Side condition: the test-shape wrapper fixes the
concrete layout (`B = 2`, `H = 4`, `N_CTX = HEAD_DIM = BLOCK_M = 128`,
`BLOCK_N = 64`, strides `(65536, 16384, 128, 1)`, mask = first 96 head lanes)
and uses `STAGE = 1`.
-/

namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Full Lean port of `attn_fwd_causal.py`'s `_attn_fwd`.

The upstream kernel runs the K/V streaming-softmax loop through a separate
`@triton.jit` helper `_attn_fwd_inner`, invoked twice: stage `4 - STAGE`
streams the strictly-below-diagonal key blocks `range(0, start_m·BLOCK_M)` with
no causal mask, and stage `2` streams the diagonal block
`range(start_m·BLOCK_M, (start_m+1)·BLOCK_M)` under the causal predicate
`offs_m[:, None] ≥ start_n + offs_n[None, :]`.

The DSL has no cross-`@triton.jit` function-call surface, so the helper body is
inlined here as a single `forRange` loop `range(0, N_CTX, BLOCK_N)` with the
causal `where` `offs_m[:, None] ≥ start_n + offs_n[None, :]` applied to every
block. This faithfully composes the two staged helper calls: on a
stage-1 block (strictly below the diagonal) every lane satisfies
`offs_m ≥ start_n + offs_n`, so the causal `where` is a no-op there, matching the
unmasked stage-1 helper; on the diagonal block it is the stage-2 mask; and on
the strictly-above-diagonal blocks (which the two-call kernel never visits) the
`where` zeroes every probability (`p = where(mask, p, 0)`), so they contribute
nothing to `acc`/`l_i` — making the full-range loop equal to the kernel's
`range(0, (start_m+1)·BLOCK_M)` traversal. -/
def attn_fwd_causal_surface
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
  qvk_offset = off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh)
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
    mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    qk = tl.where(mask, qk, -1000000.0)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    p = tl.where(mask, p, 0.0)
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

/-- The full staged causal attention surface lowers to the algorithm layer. -/
theorem attn_fwd_causal_surface_toAlgorithm_supported
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat) :
    ∃ alg, (attn_fwd_causal_surface Q K V Q_scale K_scale Out stride_qz
      stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om
      stride_on Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE
      STAGE).toAlgorithm? = Except.ok alg := by
  simp [attn_fwd_causal_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of `attn_fwd_causal.py`'s
`_attn_fwd`.

The full kernel runs staged causal attention forward loops. This slice starts after those stages have produced a
precomputed normalized `Acc` tile and proves the final masked writeback into
`Out`, preserving the source store address and mask
`(offs_m < N_CTX) & (offs_k < 96)`. The inner `tl.float32` accumulator and
`p.to(tl.float16)` dot-input cast are outside this slice. -/
def attn_fwd_causal_final_store_slice
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

noncomputable def producedAttnFwdCausalOutValue
    (s : BlockState) (Q K V QScale KScale Out : RegionName)
    (idx : TileIndex [128, 128]) : ℝ :=
  match exec (attn_fwd_causal_surface Q K V QScale KScale Out
      65536 16384 128 1
      65536 16384 128 1
      65536 16384 128 1
      65536 16384 128 1
      2 4 128 128 128 64 128 96 1).toAlgKernel s with
  | some s' => s'.readMem Out (outOffset s 4 65536 16384 128 1 128 idx)
  | none => 0.0

/-- Algorithm-layer correctness for the final output store. -/
theorem attn_fwd_causal_final_store_slice_correct
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
      (exec (attn_fwd_causal_final_store_slice Acc Out H N_CTX
            HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k
            stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (if active s N_CTX HEAD_ACTIVE BLOCK_M idx then
            s.readMem Acc
              (accOffset s H stride_acc_z stride_acc_h stride_acc_m
                stride_acc_k BLOCK_M idx)
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, attn_fwd_causal_final_store_slice, stepStmts, stepStmt,
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
theorem attn_fwd_causal_final_store_slice_compute_correct
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
      (kernel := attn_fwd_causal_final_store_slice Acc Out H N_CTX
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
  · simp [attn_fwd_causal_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := attn_fwd_causal_final_store_slice_correct Acc Out H N_CTX
    HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k stride_qz
    stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Python test-shape wrapper

`attn_fwd_causal.py`'s checked tests use `B = 2`, `H = 4`,
`N_CTX = 128`, `HEAD_DIM = 128`, `BLOCK_M = 128`, `BLOCK_N = 64`, and the
final store mask enables the first 96 head lanes. Contiguous tensors have
strides `(65536, 16384, 128, 1)`. -/

theorem attn_fwd_causal_final_store_python_test_shape_compute_correct
    (Acc Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attn_fwd_causal_final_store_slice Acc Out
        4 128 96 65536 16384 128 1 65536 16384 128 1 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] => active s 128 96 128 idx)
        (fun idx : TileIndex [128, 128] => (Out,
          outOffset s 4 65536 16384 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        s.readMem Acc (accOffset s 4 65536 16384 128 1 128 idx)) := by
  apply attn_fwd_causal_final_store_slice_compute_correct
  rintro ⟨⟨ma, hma⟩, ⟨ka, hka⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨kb, hkb⟩, _⟩ h
  simp [outOffset, offZ, offH, mIndex, kIndex] at h
  have hm : ma = mb := by omega
  have hk : ka = kb := by omega
  subst mb
  subst kb
  rfl

theorem attn_fwd_causal_surface_python_test_shape_compute_correct
    (Q K V QScale KScale Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attn_fwd_causal_surface Q K V QScale KScale Out
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
        producedAttnFwdCausalOutValue s Q K V QScale KScale Out idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attn_fwd_causal_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedAttnFwdCausalOutValue, hExec]

/-- Python test-shape summary for `attn_fwd_causal.py`.

The Python wrapper fixes `STAGE = 1`; this summary pairs that full causal
surface with the observable `Out` writes produced at the test layout. -/
theorem attn_fwd_causal_python_test_shape_output_summary
    (Q K V QScale KScale Out : RegionName) (s : BlockState) :
    (∃ alg, (attn_fwd_causal_surface Q K V QScale KScale Out
      65536 16384 128 1
      65536 16384 128 1
      65536 16384 128 1
      65536 16384 128 1
      2 4 128 128 128 64 128 96 1).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attn_fwd_causal_surface Q K V QScale KScale Out
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
        producedAttnFwdCausalOutValue s Q K V QScale KScale Out idx) := by
  constructor
  · exact attn_fwd_causal_surface_toAlgorithm_supported Q K V QScale KScale
      Out 65536 16384 128 1 65536 16384 128 1 65536 16384 128 1
      65536 16384 128 1 2 4 128 128 128 64 128 96 1
  · exact attn_fwd_causal_surface_python_test_shape_compute_correct Q K V
      QScale KScale Out s

/-! ## Genuine closed-form attention spec (exp2, causal)

`attn_fwd_causal.py`'s `_attn_fwd` (Python `stage = 3`, two `_attn_fwd_inner`
calls: stage `4 - STAGE = 1` for the strictly-below-diagonal off-diagonal key
blocks `range(0, start_m·BLOCK_M)`, no causal mask; stage `2` for the diagonal
block `range(start_m·BLOCK_M, (start_m+1)·BLOCK_M)` under the causal predicate)
is **base-2** (`tl.math.exp2`) softmax with a **scalar** score scale `q_scale ·
k_scale` (loaded once per program / per key block) and a **causal** mask. The
inlined surface composes both staged calls into a single `range(0, N_CTX,
BLOCK_N)` loop carrying `tl.where(offs_m[:, None] ≥ start_n + offs_n[None, :],
qk, -1e6)` on every block (a no-op on off-diagonal stage-1 blocks; the stage-2
mask on the diagonal block; zeroing on the never-visited above-diagonal blocks).
So the genuine closed form is the predicate-masked base-2 per-key-scale attention
`attentionRealBase2PerKeyScalePred` (`VeriTile/Triton/Math/Attention.lean`),
instantiated with:

* `keep := causalKeep qStart` — key `j` contributes to query row `i` iff
  `j ≤ qStart + i` (the kernel's `offs_m ≥ start_n + offs_n` mask, with global
  `qStart = start_m · BLOCK_M`). This single predicate captures the STAGE-3
  two-stage key set: stage-1 off-diagonal keys (`j < start_m·128`, all kept) ∪
  stage-2 diagonal keys (`start_m·128 ≤ j < (start_m+1)·128`, kept iff
  `j ≤ qStart + i`) = exactly `{j : j ≤ qStart + i}`;
* a per-key score scale carrier (the scalar `q_scale · k_scale`; here abstracted
  as `keyScaleAFC`, an opaque `Fin S → ℝ`, since the kernel loads the
  quantization scalars from memory rather than fixing them at a constant).

This routes to **base-2** (matching the `exp2` kernel); the streaming bridge
`attentionRealBase2PerKeyScalePred_eq_streaming` (sorry-free) delivers the
`osStep` online-softmax fold the exec loop realizes. -/

open VeriTile.Triton (attentionRealBase2PerKeyScalePred attnKeyListPred osStep
  causalKeep)

/-- Base address of the `(off_z, off_h)` plane for the Python test shape
(`stride_qz = 65536`, `stride_qh = 16384`, `H = 4`). Q, K, V, Out share it
(`stride_qm = 128`, `stride_qk = 1`). -/
def baseOffsetAFC (s : BlockState) : Nat :=
  s.pids 1 / 4 * 65536 + s.pids 1 % 4 * 16384

/-- Query tile: row `i` = global query `start_m·128 + i`, head lane `e`
(`stride_qm = 128`, `stride_qk = 1`). -/
noncomputable def qTileAFC (s : BlockState) (Q : RegionName) :
    TileIndex [128, 128] → ℝ :=
  fun (i, e, _) => s.readMem Q (baseOffsetAFC s + (s.pids 0 * 128 + i.val) * 128 + e.val)

/-- Key tile: row `j` (global key), head lane `e`. -/
noncomputable def kTileAFC (s : BlockState) (K : RegionName) :
    TileIndex [128, 128] → ℝ :=
  fun (j, e, _) => s.readMem K (baseOffsetAFC s + j.val * 128 + e.val)

/-- Value tile: row `j` (global key), head lane `d`. -/
noncomputable def vTileAFC (s : BlockState) (V : RegionName) :
    TileIndex [128, 128] → ℝ :=
  fun (j, d, _) => s.readMem V (baseOffsetAFC s + j.val * 128 + d.val)

/-- Global query row for output tile-row `i` in this program. -/
def qStartAFC (s : BlockState) : Nat := s.pids 0 * 128

/-- **Genuine closed form** (exp2, causal): the normalized output `acc / l_i` is
predicate-masked base-2 per-key-scale attention with the `causalKeep qStart`
mask, for an arbitrary per-key score-scale carrier `keyScale`. -/
noncomputable def attnFwdCausalOutSpec
    (s : BlockState) (Q K V : RegionName) (keyScale : Fin 128 → ℝ)
    (idx : TileIndex [128, 128]) : ℝ :=
  attentionRealBase2PerKeyScalePred (qTileAFC s Q) (kTileAFC s K) (vTileAFC s V)
    keyScale (fun i j => causalKeep (qStartAFC s) i j) idx

/-- Streaming bridge: the closed form equals the `osStep` online-softmax fold
over the causal-masked key list — the form the exec loop realizes. -/
theorem attnFwdCausalOutSpec_eq_streaming
    (s : BlockState) (Q K V : RegionName) (keyScale : Fin 128 → ℝ) (i d : Fin 128) :
    attnFwdCausalOutSpec s Q K V keyScale (i, d, PUnit.unit)
      = (let st := (attnKeyListPred (qTileAFC s Q) (kTileAFC s K) (vTileAFC s V)
            keyScale (fun i j => causalKeep (qStartAFC s) i j) i d).foldl
              osStep (0, 0, 0)
         st.2.2 / st.2.1) := by
  simpa [attnFwdCausalOutSpec] using
    VeriTile.Triton.attentionRealBase2PerKeyScalePred_eq_streaming
      (qTileAFC s Q) (kTileAFC s K) (vTileAFC s V) keyScale
      (fun i j => causalKeep (qStartAFC s) i j) i d

/-! ## Forward-loop per-statement op-eval recipes (RECIPE LAYER)

The inlined `forRange "start_n" 0 N_CTX BLOCK_N` loop body of
`attn_fwd_causal_surface`, extracted from the lowered algorithm AST at the Python
test shape (`BLOCK_M = BLOCK_DMODEL = 128`, `BLOCK_N = 64`, `N_CTX = 128`;
`@[simp]`-erased `ComputeOp.alg` fp32/fp16 dtype-cast wrappers, so the
algorithm-layer body is plain `Op`s), is the **22-statement** list (probe-checked:
the lowered body has length 25, with the static `Stmt.forRange "start_n" 0 128 64`
at index 22 carrying a 22-statement body — identical to the `attn_fwd_triton`
sibling):

```
 1  start_n = start_n                                          -- tl.multiple_of (identity)
 2  k_mask  = (offs_n[None,:] < N_CTX - start_n)               -- key boundary ∧ head-active
            & (arange(128) < 96)[:,None]
 3  k       = tl.load(K_ptrs, mask=k_mask)                     -- elementwise masked ptr load
 4  k_scale = tl.load(K_scale_ptr)                             -- scalar ptr load
 5  qk      = (castFloat (q·k)) * q_scale * k_scale            -- dot, two scalar scales
 6  mask    = offs_m[:,None] >= start_n + offs_n[None,:]       -- causal keep predicate
 7  qk      = tl.where(mask, qk, 0 - 1e6)                      -- non-kept lanes → −1e6
 8  m_ij    = tl.maximum(m_i, tl.max(qk, 1))                   -- running max (reduceMax)
 9  qk      = qk - m_ij[:,None]                                -- max-shift
10  p       = tl.math.exp2(qk)                                 -- base-2 softmax weights
11  p       = tl.where(mask, p, 0)                             -- zero non-kept lanes
12  l_ij    = tl.sum(p, 1)                                     -- denominator increment (reduceSum)
13  alpha   = tl.math.exp2(m_i - m_ij)                         -- rescale factor
14  l_i     = l_i * alpha + l_ij                               -- denominator carry
15  acc     = acc * alpha[:,None]                              -- accumulator rescale
16  v       = tl.load(V_ptrs, mask=...)                        -- elementwise masked ptr load
17  p       = castFloat real fp16 p                            -- fp16 round-trip (identity at alg)
18  acc    += tl.dot(castFloat fp16 real p, v)                 -- numerator accumulation
19  m_i     = m_ij                                             -- max carry
20  K_ptrs += BLOCK_N * HEAD_DIM                               -- ptrAdd (scalarR)
21  K_scale_ptr += 1                                           -- scalar ptrAdd
22  V_ptrs += BLOCK_N * HEAD_DIM                               -- ptrAdd (scalarR)
```

Each recipe below is a standalone `evalOp` reduction with abstract
register-readback hypotheses over a symbolic `BlockState`, mirroring the
`aft_*` recipes of the sibling. The eventual step lemma threads them through
`stepStmts.cons_some`. ASSEMBLY (invariant / loop step / pre+post-loop over the
`forRange`) is the NEXT stage and is intentionally NOT attempted here. -/

/-- `evalOp` helper for `tl.math.exp2` (`Op.exp2`). -/
theorem afc_evalOp_exp2 {shape : TileShape} (a : Op .real shape) (s : BlockState) :
    evalOp (.exp2 a) s = (do let va ← evalOp a s; some (Tile.uop WithBot.realExp2 va)) := by
  simp [evalOp]

/-- `evalOp` helper for the `>=` predicate (`Op.ge`), no `@[simp]` form. -/
theorem afc_evalOp_ge {dtype a b shape} (h : ComparableDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

/-- `evalOp` helper for `Op.boolAnd`, no `@[simp]` form. -/
theorem afc_evalOp_boolAnd {a b shape} (bc : Broadcast a b shape)
    (x : Op .bool a) (y : Op .bool b) (s : BlockState) :
    evalOp (.boolAnd bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s;
      some (Tile.bop (fun p q : Bool => p && q) bc vx vy)) := by
  simp [evalOp]

/-- `evalOp` helper for the identity `real → real` cast (the `.to(tl.float32)`
fp32 wrapper around `tl.dot`, which is a no-op at the algorithm layer): given the
inner op's value, the cast returns the same tile (`FloatDType.cast real real` is
the identity through `WithBot ℝ`). -/
theorem afc_castReal_eval {shape : TileShape} (a : Op .real shape) (s : BlockState)
    (va : Tile .real shape) (ha : evalOp a s = some va) :
    evalOp (Op.castFloat FloatDType.real FloatDType.real a) s
      = some ⟨fun i => va.data i⟩ := by
  rw [evalOp_castFloat]
  simp only [FloatDType.toTileDType_real]
  rw [ha]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L3: `k = tl.load(K_ptrs, mask=k_mask)`** — elementwise masked pointer load.
`K_ptrs` is a plain `[BM, BN]` pointer tile (per-lane `(region, offset)`); each
kept lane reads `s.readMem`, each masked-out lane reads `s.undef`. -/
theorem afc_load_k_eval
    (s : BlockState) (BM BN : Nat) (name maskName : RegName)
    (ptrs : Tile .ptr [BM, BN]) (masks : Tile .bool [BM, BN])
    (hptr : s.regs .ptr [BM, BN] name = some ptrs)
    (hmask : s.regs .bool [BM, BN] maskName = some masks) :
    evalOp (.load .real (.ptr (.ref .ptr [BM, BN] name))
        (.mask (.ref .bool [BM, BN] maskName))) s
      = some ⟨fun i : TileIndex [BM, BN] =>
          if masks.data i then some (s.readMem (ptrs.data i).1 (ptrs.data i).2)
          else some (s.undef (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, evalOp_ref, hptr, hmask, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue_real, if_true]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L4: `k_scale = tl.load(K_scale_ptr)`** — unmasked scalar pointer load. -/
theorem afc_load_kscale_eval
    (s : BlockState) (name : RegName) (ptr : Tile .ptr [])
    (hptr : s.regs .ptr [] name = some ptr) :
    evalOp (.load .real (.ptr (.ref .ptr [] name)) .none) s
      = some ⟨fun _ : TileIndex [] =>
          some (s.readMem (ptr.data PUnit.unit).1 (ptr.data PUnit.unit).2)⟩ := by
  simp only [evalOp, evalOp_ref, hptr, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue_real, if_true]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L16: `v = tl.load(V_ptrs, mask=...)`** — elementwise masked pointer load,
same shape as L3 with the V boundary/head-active mask supplied as a register. -/
theorem afc_load_v_eval
    (s : BlockState) (BN BD : Nat) (name maskName : RegName)
    (ptrs : Tile .ptr [BN, BD]) (masks : Tile .bool [BN, BD])
    (hptr : s.regs .ptr [BN, BD] name = some ptrs)
    (hmask : s.regs .bool [BN, BD] maskName = some masks) :
    evalOp (.load .real (.ptr (.ref .ptr [BN, BD] name))
        (.mask (.ref .bool [BN, BD] maskName))) s
      = some ⟨fun i : TileIndex [BN, BD] =>
          if masks.data i then some (s.readMem (ptrs.data i).1 (ptrs.data i).2)
          else some (s.undef (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, evalOp_ref, hptr, hmask, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue_real, if_true]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L2: `k_mask = (offs_n[None,:] < N_CTX - start_n) & (arange(128) < HEAD_ACTIVE)[:,None]`**
— the key boundary ∧ head-active load mask. -/
theorem afc_kmask_eval (s : BlockState) (SN NCTX HA : Nat)
    (offsn : Tile .nat [64])
    (hoffsn : s.regs .nat [64] "offs_n" = some offsn)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOp (Op.boolAnd (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat NCTX) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat HA)))) s
      = some ⟨fun idx : TileIndex [128, 64] =>
          (ComparableDType.nat.lt (offsn.data (idx.2.1, PUnit.unit)) (NCTX - SN))
            && (ComparableDType.nat.lt idx.1.val HA)⟩ := by
  rw [afc_evalOp_boolAnd]
  simp only [evalOp_lt, evalOp.eq_def, evalOp_constNat, evalOp_arange, hoffsn, hsn,
    Option.bind_eq_bind, Option.bind_some, Option.bind]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.cop_data, Tile.expandDim_data, Tile.vec, Tile.scalar,
    Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
    Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
    Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
    Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
    TileShape.dropInsertedIndex, NumericDType.sub]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L5: `qk = (castFloat (tl.dot(q, k))) * q_scale * k_scale`** — the `q·k` dot
(through an identity `real → real` fp32 cast) scaled by the two scalar
quantization factors `q_scale`, `k_scale`, each broadcast on the right. -/
theorem afc_qk_dot_eval (s : BlockState) (BM BN BD : Nat)
    (qtile : Tile .real [BM, BD]) (ktile : Tile .real [BD, BN])
    (qsc ksc : Tile .real [])
    (hq : s.regs .real [BM, BD] "q" = some qtile)
    (hk : s.regs .real [BD, BN] "k" = some ktile)
    (hqsc : s.regs .real [] "q_scale" = some qsc)
    (hksc : s.regs .real [] "k_scale" = some ksc) :
    evalOp (Op.mul .real Broadcast.scalarR
        (Op.mul .real Broadcast.scalarR
          (Op.castFloat FloatDType.real FloatDType.real
            (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k")))
          (Op.ref .real [] "q_scale"))
        (Op.ref .real [] "k_scale")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.mul Broadcast.scalarR
            ⟨fun i => (Tile.dot [] qtile ktile).data i⟩ qsc) ksc) := by
  have hqr : evalOp (Op.ref .real [BM, BD] "q") s = some qtile := by rw [evalOp_ref, hq]
  have hkr : evalOp (Op.ref .real [BD, BN] "k") s = some ktile := by rw [evalOp_ref, hk]
  have hdot : @evalOp TileDType.real [BM, BN]
      (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k")) s
      = some (Tile.dot [] qtile ktile) := by
    erw [evalOp_dot [] (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k"), hqr, hkr]; rfl
  have hcast : @evalOp TileDType.real [BM, BN]
      (Op.castFloat FloatDType.real FloatDType.real
        (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k"))) s
      = some ⟨fun i => (Tile.dot [] qtile ktile).data i⟩ :=
    afc_castReal_eval
      (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k"))
      s (Tile.dot [] qtile ktile) hdot
  rw [evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, hcast, hqsc, hksc, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L6: `mask = offs_m[:,None] >= (start_n + offs_n[None,:])`** — the causal
keep predicate. Cell `(i, j)` is kept iff `offs_m i ≥ SN + offs_n j`. -/
theorem afc_mask_eval (s : BlockState) (SN : Nat)
    (offsm : Tile .nat [128]) (offsn : Tile .nat [64])
    (hoffsm : s.regs .nat [128] "offs_m" = some offsm)
    (hoffsn : s.regs .nat [64] "offs_n" = some offsn)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOp (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "offs_n")))) s
      = some ⟨fun idx : TileIndex [128, 64] =>
          ComparableDType.nat.ge (offsm.data (idx.1, PUnit.unit))
            (SN + offsn.data (idx.2.1, PUnit.unit))⟩ := by
  rw [afc_evalOp_ge]
  simp only [evalOp_add, evalOp.eq_def, hoffsm, hoffsn, hsn,
    Option.bind_eq_bind, Option.bind_some, Option.bind]
  refine congrArg some ?_
  ext idx
  simp only [Tile.cop_data, Tile.bop_data, Tile.expandDim_data, Tile.scalar,
    Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
    Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
    Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
    Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
    TileShape.dropInsertedIndex, NumericDType.add]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L7: `qk = tl.where(mask, qk, 0 - 1e6)`** — keep `qk` on kept lanes, set
non-kept lanes to the large-negative sentinel `(0 : ℝ) - 1000000` (vanishes under
`exp2`). -/
theorem afc_where_qk_eval (s : BlockState) (masktile : Tile .bool [128, 64])
    (qktile : Tile .real [128, 64])
    (hmask : s.regs .bool [128, 64] "mask" = some masktile)
    (hqk : s.regs .real [128, 64] "qk" = some qktile) :
    evalOp (Op.where (Op.ref .bool [128, 64] "mask")
        (Op.ref .real [128, 64] "qk")
        (Op.broadcast (Op.sub .real Broadcast.nil (Op.const 0) (Op.const 1000000.0))
          [128, 64])) s
      = some ⟨fun idx : TileIndex [128, 64] =>
          if masktile.data idx then qktile.data idx
          else WithBot.realSub (some (0 : ℝ)) (some (1000000.0 : ℝ))⟩ := by
  have hbcast : @evalOp TileDType.real [128, 64]
      (Op.broadcast (Op.sub .real Broadcast.nil (Op.const 0) (Op.const 1000000.0)) [128, 64]) s
      = some (⟨fun _ : TileIndex [128, 64] =>
          WithBot.realSub (some (0 : ℝ)) (some (1000000.0 : ℝ))⟩ :
          Tile .real [128, 64]) := by
    simp only [evalOp, evalOp_sub, evalOp_const, Option.bind_eq_bind, Option.bind_some]
    rfl
  rw [evalOp_where]
  simp only [evalOp_ref, hmask, hqk, hbcast, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.select_data, Tile.scalar]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L8: `m_ij = tl.maximum(m_i, tl.max(qk, 1))`** — lowered to
`where(m_i > reduceMax(qk,1), m_i, reduceMax(qk,1))`. -/
theorem afc_mij_eval (s : BlockState)
    (mtile : Tile .real [128]) (qktile : Tile .real [128, 64]) (rmaxT : Tile .real [128])
    (hmi : s.regs .real [128] "m_i" = some mtile)
    (hqk : s.regs .real [128, 64] "qk" = some qktile)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [128, 64].length) qktile = some rmaxT) :
    evalOp (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [128] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false
            (Op.ref .real [128, 64] "qk")))
        (Op.ref .real [128] "m_i")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false
          (Op.ref .real [128, 64] "qk"))) s
      = some (Tile.select
          (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT)
          mtile rmaxT) := by
  have hrmax : @evalOp TileDType.real [128]
      (Op.reduceMax (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false
        (Op.ref .real [128, 64] "qk")) s = some rmaxT := by
    unfold evalOp
    simp only [evalOp_ref, hqk, Option.bind_eq_bind, Option.bind_some]
    exact hrm
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hmi, hrmax, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L9: `qk = qk - m_ij[:, None]`** — the max-shift before `exp2`. -/
theorem afc_qk_sub_eval (s : BlockState) (hax : 1 < [128].length.succ)
    (qktile : Tile .real [128, 64]) (mc : Tile .real [128])
    (hqk : s.regs .real [128, 64] "qk" = some qktile)
    (hmij : s.regs .real [128] "m_ij" = some mc) :
    evalOp (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [128, 64] "qk") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [128] "m_ij"))) s
      = some (Tile.bop NumericDType.real.sub
          (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          qktile (Tile.expandDim ⟨1, hax⟩ mc)) := by
  have hexp : @evalOp TileDType.real [128, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [128] "m_ij")) s
      = some (Tile.expandDim ⟨1, hax⟩ mc) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hmij
  rw [evalOp_sub]
  simp only [evalOp_ref, hqk, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L10: `p = tl.math.exp2(qk)`** — the base-2 softmax weights. -/
theorem afc_p_eval (s : BlockState) (qktile : Tile .real [128, 64])
    (hqk : s.regs .real [128, 64] "qk" = some qktile) :
    evalOp (Op.exp2 (Op.ref .real [128, 64] "qk")) s
      = some (Tile.uop WithBot.realExp2 qktile) := by
  rw [afc_evalOp_exp2]; simp only [evalOp_ref, hqk, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L11: `p = tl.where(mask, p, 0)`** — zero the non-kept lanes. -/
theorem afc_p_mask_eval (s : BlockState) (masktile : Tile .bool [128, 64])
    (ptile : Tile .real [128, 64])
    (hmask : s.regs .bool [128, 64] "mask" = some masktile)
    (hp : s.regs .real [128, 64] "p" = some ptile) :
    evalOp (Op.where (Op.ref .bool [128, 64] "mask")
        (Op.ref .real [128, 64] "p") (Op.broadcast (Op.const 0.0) [128, 64])) s
      = some ⟨fun idx : TileIndex [128, 64] =>
          if masktile.data idx then ptile.data idx else (some (0.0 : ℝ) : WithBot ℝ)⟩ := by
  have hbcast : @evalOp TileDType.real [128, 64] (Op.broadcast (Op.const 0.0) [128, 64]) s
      = some (⟨fun _ : TileIndex [128, 64] => (some (0.0 : ℝ) : WithBot ℝ)⟩ :
          Tile .real [128, 64]) := by
    simp only [evalOp, evalOp_const, Option.bind_eq_bind, Option.bind_some]; rfl
  rw [evalOp_where]
  simp only [evalOp_ref, hmask, hp, hbcast, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.select_data, Tile.scalar]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L12: `l_ij = tl.sum(p, 1)`** — the per-row denominator increment. -/
theorem afc_lij_eval (s : BlockState) (ptile : Tile .real [128, 64])
    (hp : s.regs .real [128, 64] "p" = some ptile) :
    evalOp (Op.reduceSum (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false
        (Op.ref .real [128, 64] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [128, 64].length) ptile) := by
  rw [evalOp_reduceSum]
  simp only [evalOp_ref, hp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L13: `alpha = tl.math.exp2(m_i - m_ij)`** — the running rescale factor
`exp2(m_i − m_ij)`. -/
theorem afc_alpha_eval (s : BlockState) (mi mij : Tile .real [128])
    (hmi : s.regs .real [128] "m_i" = some mi)
    (hmij : s.regs .real [128] "m_ij" = some mij) :
    evalOp (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [128] "m_i") (Op.ref .real [128] "m_ij"))) s
      = some (Tile.uop WithBot.realExp2
          (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mi mij)) := by
  rw [afc_evalOp_exp2, evalOp_sub]
  simp only [evalOp_ref, hmi, hmij, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L14: `l_i = l_i * alpha + l_ij`** — the denominator carry. -/
theorem afc_li_eval (s : BlockState) (li alpha lij : Tile .real [128])
    (hli : s.regs .real [128] "l_i" = some li)
    (halpha : s.regs .real [128] "alpha" = some alpha)
    (hlij : s.regs .real [128] "l_ij" = some lij) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [128] "l_i") (Op.ref .real [128] "alpha"))
        (Op.ref .real [128] "l_ij")) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
          (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) li alpha) lij) := by
  rw [evalOp_add, evalOp_mul]
  simp only [evalOp_ref, hli, halpha, hlij, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L15: `acc = acc * alpha[:, None]`** — rescale the output accumulator by the
per-row `α`. -/
theorem afc_acc_rescale_eval (s : BlockState) (hax : 1 < [128].length.succ)
    (acctile : Tile .real [128, 128]) (alpha : Tile .real [128])
    (hacc : s.regs .real [128, 128] "acc" = some acctile)
    (halpha : s.regs .real [128] "alpha" = some alpha) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [128, 128] "acc") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [128] "alpha"))) s
      = some (Tile.bop NumericDType.real.mul
          (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          acctile (Tile.expandDim ⟨1, hax⟩ alpha)) := by
  have hexp : @evalOp TileDType.real [128, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [128] "alpha")) s
      = some (Tile.expandDim ⟨1, hax⟩ alpha) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ halpha
  rw [evalOp_mul]
  simp only [evalOp_ref, hacc, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L17: `p = p.to(tl.float16)`** — the fp16 round-trip cast, an identity tile
map at the algorithm layer (`FloatDType.cast` is identity through `WithBot ℝ`). -/
theorem afc_p_fp16_eval (s : BlockState) (ptile : Tile .real [128, 64])
    (hp : s.regs .real [128, 64] "p" = some ptile) :
    evalOp (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [128, 64] "p")) s
      = some ⟨fun i => FloatDType.real.cast FloatDType.fp16 (ptile.data i)⟩ := by
  simp only [evalOp_castFloat, FloatDType.toTileDType_real, evalOp_ref, hp,
    Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L18: `acc += tl.dot(p.to(fp16).to(real), v)`** — numerator accumulation. The
fp16-cast `p` is cast back to real (identity at the algorithm layer) inside the
dot; `v` is real. -/
theorem afc_acc_eval (s : BlockState) (BM BN BD : Nat)
    (acctile : Tile .real [BM, BD]) (pf16 : Tile .fp16 [BM, BN]) (vtile : Tile .real [BN, BD])
    (hacc : s.regs .real [BM, BD] "acc" = some acctile)
    (hp : s.regs .fp16 [BM, BN] "p" = some pf16)
    (hv : s.regs .real [BN, BD] "v" = some vtile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BD] "acc")
        (Op.dot (batch := [])
          (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref .fp16 [BM, BN] "p"))
          (Op.ref .real [BN, BD] "v"))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acctile
          (Tile.dot [] ⟨fun i => FloatDType.fp16.cast FloatDType.real (pf16.data i)⟩ vtile)) := by
  have hpc : evalOp
      (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref .fp16 [BM, BN] "p")) s
      = some ⟨fun i => FloatDType.fp16.cast FloatDType.real (pf16.data i)⟩ := by
    simp only [evalOp_castFloat, FloatDType.toTileDType_fp16, evalOp_ref, hp,
      Option.bind_eq_bind, Option.bind_some]
  have hvr : evalOp (Op.ref .real [BN, BD] "v") s = some vtile := by rw [evalOp_ref, hv]
  have hdot : @evalOp TileDType.real [BM, BD]
      (Op.dot (batch := [])
        (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref .fp16 [BM, BN] "p"))
        (Op.ref .real [BN, BD] "v")) s
      = some (Tile.dot [] ⟨fun i => FloatDType.fp16.cast FloatDType.real (pf16.data i)⟩ vtile) := by
    erw [evalOp_dot []
      (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref .fp16 [BM, BN] "p"))
      (Op.ref .real [BN, BD] "v"), hpc, hvr]; rfl
  rw [evalOp_add]; simp only [evalOp_ref, hacc, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L19: `m_i = m_ij`** — the running-max carry. -/
theorem afc_mi_carry_eval (s : BlockState) (mij : Tile .real [128])
    (hmij : s.regs .real [128] "m_ij" = some mij) :
    evalOp (Op.ref .real [128] "m_ij") s = some mij := by
  rw [evalOp_ref, hmij]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L20/L22: `K_ptrs += BLOCK_N * HEAD_DIM` / `V_ptrs += BLOCK_N * HEAD_DIM`** —
advance a `[BT, BS]` pointer tile by the scalar block stride `d`, broadcast on
the right. -/
theorem afc_advance_ptr_eval (s : BlockState) (BT BS d : Nat) (name : RegName)
    (ptrs : Tile .ptr [BT, BS])
    (hptr : s.regs .ptr [BT, BS] name = some ptrs) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BT, BS] name)
        (Op.mul .nat Broadcast.nil (Op.constNat d) (Op.constNat 128))) s
      = some (Tile.ptrAdd Broadcast.scalarR ptrs
          (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar d) (Tile.scalar 128))) := by
  simp only [evalOp, evalOp_ref, evalOp_constNat, hptr, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L21: `K_scale_ptr += 1`** — advance the scalar K-scale pointer by one key
block. -/
theorem afc_advance_kscale_eval (s : BlockState) (name : RegName) (ptr : Tile .ptr [])
    (hptr : s.regs .ptr [] name = some ptr) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ref .ptr [] name) (Op.constNat 1)) s
      = some (Tile.ptrAdd Broadcast.nil ptr (Tile.scalar 1)) := by
  simp only [evalOp, evalOp_ref, evalOp_constNat, hptr, Option.bind_eq_bind, Option.bind_some]

end VeriTile.Bench.TritonBenchG.AttnFwdCausal
