import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention
import VeriTile.Triton.LoopInvariant

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
  └─ attn_fwd_causal_surface_genuine_compute_correct
       └─ afc_attn_exec                                     full body: preLoop + forRange loop + postLoop
            ├─ afcPreLoop_invariant                         preLoop ⇒ afcInvariant base case
            ├─ forRange_inv (afc_attn_step)                 streaming online-softmax loop
            └─ afcPostLoop_eval                             acc /= l_i + masked store ⇒ attnFwdCausalOutSpec
                 └─ afcStateBot_full_eq_spec                ⊥-seed fold = genuine closed form
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; the `exp2`, the `tl.dot`
`float16` accumulation, and `q_scale · k_scale` quantization are not modeled at
the bit level); `@triton.autotune`/`num_warps`/`num_stages` are not modeled.
The verified result is the **genuine closed form**: the proof models the full
online-softmax K/V streaming loop (per-block `m_i`/`l_i`/`acc` updates, causal
mask, the final `acc / l_i` normalization) and establishes that the masked store
writes `attnFwdCausalOutSpec` — predicate-masked base-2 (`exp2`) per-key-scale
attention with the `causalKeep qStart` mask — to `Out` at every active output
lane, preserving inactive lanes. Side conditions: clean input (`undef = 0`) and
the sentinel score bound `afcScoreBound`. The test-shape wrapper fixes the
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

/-- **Masked query tile** — the tile the kernel actually loads: `qTileAFC` but
zeroed on the head-inactive lanes `e ≥ 96` and on out-of-range query rows
`qStart + i ≥ N_CTX = 128` (the `q = tl.load(..., mask=((offs_m < N_CTX) &
(arange < HEAD_ACTIVE)))` load mask, with `HEAD_ACTIVE = 96`). The query-row
boundary clause is vacuous on the active output lanes (`offs_m < 128`), so this
matches `qTileAFC` head-masked there; it only zeroes the masked-off rows the
final store discards. -/
noncomputable def qTileAFCm (s : BlockState) (Q : RegionName) :
    TileIndex [128, 128] → ℝ :=
  fun (i, e, u) =>
    if qStartAFC s + i.val < 128 ∧ e.val < 96 then qTileAFC s Q (i, e, u) else 0

/-- **Masked value tile** — the tile the kernel actually loads: `vTileAFC` but
zeroed on the head-inactive lanes `d ≥ 96` (the `v = tl.load(..., mask=(... &
(arange < HEAD_ACTIVE)))` head-active mask). -/
noncomputable def vTileAFCm (s : BlockState) (V : RegionName) :
    TileIndex [128, 128] → ℝ :=
  fun (j, d, u) => if d.val < 96 then vTileAFC s V (j, d, u) else 0

/-- **Genuine closed form** (exp2, causal): the normalized output `acc / l_i` is
predicate-masked base-2 per-key-scale attention with the `causalKeep qStart`
mask, for an arbitrary per-key score-scale carrier `keyScale`. Uses the kernel's
actually-loaded **masked** q/v tiles (head lanes `≥ 96` zeroed). -/
noncomputable def attnFwdCausalOutSpec
    (s : BlockState) (Q K V : RegionName) (keyScale : Fin 128 → ℝ)
    (idx : TileIndex [128, 128]) : ℝ :=
  attentionRealBase2PerKeyScalePred (qTileAFCm s Q) (kTileAFC s K) (vTileAFCm s V)
    keyScale (fun i j => causalKeep (qStartAFC s) i j) idx

/-- Streaming bridge: the closed form equals the `osStep` online-softmax fold
over the causal-masked key list — the form the exec loop realizes. -/
theorem attnFwdCausalOutSpec_eq_streaming
    (s : BlockState) (Q K V : RegionName) (keyScale : Fin 128 → ℝ) (i d : Fin 128) :
    attnFwdCausalOutSpec s Q K V keyScale (i, d, PUnit.unit)
      = (let st := (attnKeyListPred (qTileAFCm s Q) (kTileAFC s K) (vTileAFCm s V)
            keyScale (fun i j => causalKeep (qStartAFC s) i j) i d).foldl
              osStep (0, 0, 0)
         st.2.2 / st.2.1) := by
  simpa [attnFwdCausalOutSpec] using
    VeriTile.Triton.attentionRealBase2PerKeyScalePred_eq_streaming
      (qTileAFCm s Q) (kTileAFC s K) (vTileAFCm s V) keyScale
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

/-! # FOUNDATION: exec-assembly bank (ported from attn_fwd_triton sibling)

PTR-bind kit, ⊥-seed online-softmax math, body_split, preLoop AST + invariant +
preLoop_eval. Structurally identical to `attn_fwd_triton`; renamed `aft`→`afc`.
Step/attn_step/postLoop/top theorems are the NEXT stage (not in this bank). -/

namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal

open VeriTile.Triton

set_option linter.unusedSimpArgs false

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- `evalOp (Op.ptrBase R)` — the base pointer tile `(R, 0)` at the empty shape. -/
theorem afc_evalOp_ptrBase {d : TileDType} (region : Region d) (s : BlockState) :
    evalOp (Op.ptrBase region) s = some (Tile.scalar (Region.cast region, 0)) := by
  simp only [evalOp]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Bind-aware `ptrAdd` reducer.** Given the operand evaluations
`evalOp ptr s = some ptrs` and `evalOp off s = some offs`, fire the `ptrAdd`
reduction inside the `Option.bind` do-block. Threads each `expandDim`-bearing
offset (pre-proved via `evalOp_expandDim_ref_of_regs` / `evalOp_add` / `evalOp_mul`)
into the elementwise `Tile.ptrAdd`. -/
theorem afc_evalOp_ptrAdd_of {a b shape : TileShape}
    (bc : Broadcast a b shape) (ptr : Op .ptr a) (off : Op .nat b) (s : BlockState)
    (ptrs : Tile .ptr a) (offs : Tile .nat b)
    (hptr : evalOp ptr s = some ptrs) (hoff : evalOp off s = some offs) :
    evalOp (Op.ptrAdd bc ptr off) s = some (Tile.ptrAdd bc ptrs offs) := by
  simp only [evalOp, hptr, hoff, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Bind-aware masked `.ptr` load reducer.** Given the pointer-tile and mask-tile
evaluations, fire the masked `tl.load` over an arbitrary evaluated `.ptr` op (not
just a `ref`): kept lanes read `s.readMem`, masked-out lanes read `s.undef`.
Generalizes `afc_load_k_eval`/`afc_load_v_eval` to inline (non-`ref`) ptr/mask ops,
unblocking the preLoop `q` load (inline `boolAnd` mask). -/
theorem afc_evalOp_load_ptr_mask_of {shape : TileShape}
    (ptrOp : Op .ptr shape) (maskOp : Op .bool shape) (s : BlockState)
    (ptrs : Tile .ptr shape) (masks : Tile .bool shape)
    (hptr : evalOp ptrOp s = some ptrs) (hmask : evalOp maskOp s = some masks) :
    evalOp (.load .real (.ptr ptrOp) (.mask maskOp)) s
      = some ⟨fun i : TileIndex shape =>
          if masks.data i then some (s.readMem (ptrs.data i).1 (ptrs.data i).2)
          else some (s.undef (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, hptr, hmask, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue_real, if_true]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Bind-aware unmasked `.ptr` load reducer.** Given the pointer-tile evaluation,
fire the unmasked `tl.load` over an arbitrary evaluated `.ptr` op: every lane reads
`s.readMem`. Unblocks the preLoop scalar `q_scale` load (over `Q_scale_ptr`). -/
theorem afc_evalOp_load_ptr_none_of {shape : TileShape}
    (ptrOp : Op .ptr shape) (s : BlockState) (ptrs : Tile .ptr shape)
    (hptr : evalOp ptrOp s = some ptrs) :
    evalOp (.load .real (.ptr ptrOp) .none) s
      = some ⟨fun i : TileIndex shape =>
          some (s.readMem (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, hptr, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue_real, if_true]

/-- The `reduceMaxDrop` over axis 1 of a `[128, 64]` real tile always succeeds
(axis dim `64 > 0`); the explicit `some`-value form lets the `m_ij` recipe's
`hrm` hypothesis be discharged for an inferred `qk` tile inside the loop-body chain. -/
theorem afc_reduceMaxDrop1_some (x : Tile .real [128, 64]) :
    Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [128, 64].length) x
      = some ⟨fun outIdx =>
          (Finset.univ : Finset (Fin (TileShape.axisDim [128, 64] (⟨1, by simp⟩ : Fin [128, 64].length)))).sup'
            (by exact ⟨⟨0, by decide⟩, Finset.mem_univ _⟩)
            (fun k => x.data (TileShape.insertAxisIndex [128, 64] (⟨1, by simp⟩ : Fin [128, 64].length) outIdx k))⟩ := by
  unfold Tile.reduceMaxDrop
  rw [dif_pos (by decide : 0 < TileShape.axisDim [128, 64] (⟨1, by simp⟩ : Fin [128, 64].length))]

/-! ## FOUNDATION Part 3 — ⊥-seeded online-softmax recurrence (the loop-invariant math)

The kernel seeds the running max `m_i` at `-inf` (`tl.zeros − float("inf")`, i.e.
`⊥`), the denominator `l_i` at `0` (the surface adds `+ 1.0` but the genuine
spec uses the ⊥-seeded `0` recurrence — see the modeling note) and the
accumulator `acc` at `0`, then streams `BLOCK_N = 64`-key blocks. Mirroring the
flash-attn foundation (`flashStateBot`/`flashRunningMax`/`osStepBot`), we model
the running register state as a `WithBot ℝ × ℝ × ℝ` fold of `osStepBot` over the
causally-filtered key prefix. All defs/lemmas here are pure math over the AFC
tile-functions (`qTileAFC`/`kTileAFC`/`vTileAFC`), independent of the exec layer.

`osStepBot` and the consistency/block lemmas are ported from the flash-attn
foundation (`origin/fix/flash-attn-full`, #303); the spec bridge
`afcStateBot_full_eq_spec` reconnects to `attnFwdCausalOutSpec` via the
`attnFwdCausalOutSpec_eq_streaming` online-softmax fold (the running max cancels
in the `acc / l` ratio, so the ⊥-seed and the neutral `(0,0,0)` seed agree). -/

open VeriTile.Triton (osStep pow2 pow2_add pow2_pos sum_map_pow2_sub)

/-- The `(score, value)` pair the kernel streams for output `(i, d)` at global
key `j`: score `keyScale j · (q row i · k row j)`, value `V[j, d]`. The per-key
score scale `keyScale j` carries the scalar `q_scale · k_scale` quantization. -/
noncomputable def afcKV
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (i : Fin 128) (d : Fin 128) (j : Fin 128) : ℝ × ℝ :=
  (keyScale j * Finset.univ.sum (fun e : Fin 128 => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)),
   vT (j, d, PUnit.unit))

/-- Causal per-row key list over the window `[0, hi)`: keys `j < hi` with
`j ≤ qStart + i` (the `offs_m ≥ start_n + offs_n` causal mask), in index order.
After `c` blocks `hi = c · 64`, this is the prefix the kernel has streamed. -/
noncomputable def afcKeysUpto
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128) : List (ℝ × ℝ) :=
  (List.finRange 128).filterMap (fun j : Fin 128 =>
    if j.val < hi ∧ j.val ≤ qStart + i.val then
      some (afcKV qT kT vT keyScale i d j)
    else none)

/-- Block-`c` per-row key list: keys with `c·64 ≤ j < (c+1)·64` passing the
causal filter — the keys the loop's `c`-th iteration streams. -/
noncomputable def afcBlock
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart c : Nat) (i : Fin 128) (d : Fin 128) : List (ℝ × ℝ) :=
  (List.finRange 128).filterMap (fun j : Fin 128 =>
    if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val then
      some (afcKV qT kT vT keyScale i d j)
    else none)

/-- **⊥-seeded running max** of the streamed key prefix `[0, hi)`: the value the
kernel carries in `m_i` (`tl.zeros − inf` seeds at `⊥`). The `WithBot ⊔`-fold of
the coerced per-key scores; `⊥` on the empty / `hi = 0` window. -/
noncomputable def afcRunningMax
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128) : WithBot ℝ :=
  ((afcKeysUpto qT kT vT keyScale qStart hi i d).map
    (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥

/-- One ⊥-seeded online-softmax step: like `osStep`, but the running max lives in
`WithBot ℝ` (seeded `⊥`), so `α = realExp2(m ⊖ m')` is `0` on the first block —
faithful to the kernel's `m_i` register (`tl.zeros − inf`) and `l_i`/`acc`
(seeded `0`). Ported from the flash-attn foundation. -/
noncomputable def osStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let s := sv.1; let v := sv.2
  let m' := m ⊔ ((s : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
  let p := pow2 (s - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)

/-- `afcStateBot` — the ⊥-seeded running `(max, denom, acc)` after streaming the
window `[0, hi)`. Faithful to the kernel's register recurrence (`m_i` seeded `⊥`,
`l_i`/`acc` seeded `0`). -/
noncomputable def afcStateBot
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128) : WithBot ℝ × ℝ × ℝ :=
  (afcKeysUpto qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 0, 0)

/-- The running `max` component of an `osStepBot` fold is the `WithBot ⊔`-fold. -/
theorem afcStateBot_fst
    (xs : List (ℝ × ℝ)) (m₀ : WithBot ℝ) (l₀ acc₀ : ℝ) :
    (xs.foldl osStepBot (m₀, l₀, acc₀)).1
      = (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldl (· ⊔ ·) m₀ := by
  induction xs generalizing m₀ l₀ acc₀ with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl_cons, List.map_cons]; rw [ih]; rfl

/-- The `WithBot ⊔`-fold is seed/direction-agnostic. -/
theorem afc_foldl_sup_bot_eq_foldr (L : List (WithBot ℝ)) :
    L.foldl (· ⊔ ·) (⊥ : WithBot ℝ) = L.foldr (· ⊔ ·) (⊥ : WithBot ℝ) := by
  have gen : ∀ (m : WithBot ℝ), L.foldl (· ⊔ ·) m = m ⊔ L.foldr (· ⊔ ·) ⊥ := by
    induction L with
    | nil => intro m; simp
    | cons a t ih => intro m; simp only [List.foldl_cons, List.foldr_cons, ih]; rw [max_assoc]
  rw [gen ⊥, bot_sup_eq]

/-- The ⊥-seeded running `max` of `afcStateBot` is exactly `afcRunningMax`. -/
theorem afcStateBot_fst_eq_runningMax
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128) :
    (afcStateBot qT kT vT keyScale qStart hi i d).1
      = afcRunningMax qT kT vT keyScale qStart hi i d := by
  rw [afcStateBot, afcStateBot_fst, afcRunningMax, afc_foldl_sup_bot_eq_foldr]

/-- **⊥-seeded consistency.** Folding `osStepBot` from a start `(m, l, acc)` whose
`l`/`acc` are anchored to the true (max-free) batch denominator `L` / accumulator
`T` via the `⊥`-aware factor `κ` (`κ ⊥ = 0`, `κ (some r) = pow2(−r)`) keeps that
invariant. Ported from the flash-attn foundation. -/
theorem osStepBot_foldl_consistent (xs : List (ℝ × ℝ)) (m : WithBot ℝ) (l acc T L : ℝ)
    (hl : l = (m.elim 0 (fun r => pow2 (-r))) * L)
    (hacc : acc = (m.elim 0 (fun r => pow2 (-r))) * T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0) :
    let st := xs.foldl osStepBot (m, l, acc)
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
    simpa [List.foldl_cons, osStepBot, hm', List.map_cons, add_assoc] using step

/-- The ⊥-seeded denominator equals `κ(afcRunningMax)·Σpow2 score`. -/
theorem afcStateBot_snd_fst
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128) :
    (afcStateBot qT kT vT keyScale qStart hi i d).2.1
      = ((afcRunningMax qT kT vT keyScale qStart hi i d).elim 0 (fun r => pow2 (-r)))
        * (0 + ((afcKeysUpto qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1)).sum) := by
  rw [afcStateBot]
  rw [(osStepBot_foldl_consistent (afcKeysUpto qT kT vT keyScale qStart hi i d) ⊥ 0 0 0 0
    (by simp) (by simp) (by simp) (by simp)).1]
  rw [show ((afcKeysUpto qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 0, 0)).1
        = afcRunningMax qT kT vT keyScale qStart hi i d from by
    rw [afcStateBot_fst, afcRunningMax, afc_foldl_sup_bot_eq_foldr]]

/-- The ⊥-seeded accumulator equals `κ(afcRunningMax)·Σpow2 score·v`. -/
theorem afcStateBot_snd_snd
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128) :
    (afcStateBot qT kT vT keyScale qStart hi i d).2.2
      = ((afcRunningMax qT kT vT keyScale qStart hi i d).elim 0 (fun r => pow2 (-r)))
        * (0 + ((afcKeysUpto qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1 * p.2)).sum) := by
  rw [afcStateBot]
  rw [(osStepBot_foldl_consistent (afcKeysUpto qT kT vT keyScale qStart hi i d) ⊥ 0 0 0 0
    (by simp) (by simp) (by simp) (by simp)).2]
  rw [show ((afcKeysUpto qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 0, 0)).1
        = afcRunningMax qT kT vT keyScale qStart hi i d from by
    rw [afcStateBot_fst, afcRunningMax, afc_foldl_sup_bot_eq_foldr]]

/-- The ⊥-seeded `acc / denom` ratio is the running-max-free batch ratio (the max
factor cancels). Valid whenever the streamed window is nonempty (`afcRunningMax ≠ ⊥`). -/
theorem afcStateBot_ratio_eq
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128)
    (hne : afcRunningMax qT kT vT keyScale qStart hi i d ≠ ⊥) :
    (afcStateBot qT kT vT keyScale qStart hi i d).2.2
        / (afcStateBot qT kT vT keyScale qStart hi i d).2.1
      = ((afcKeysUpto qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1 * p.2)).sum
        / ((afcKeysUpto qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1)).sum := by
  rw [afcStateBot_snd_fst, afcStateBot_snd_snd, zero_add, zero_add]
  cases hM : afcRunningMax qT kT vT keyScale qStart hi i d with
  | bot => exact absurd hM hne
  | coe r =>
    have hκ : ((r : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-r) := rfl
    rw [hκ]
    have hpos : pow2 (-r) ≠ 0 := ne_of_gt (pow2_pos _)
    rw [mul_div_mul_left _ _ hpos]

/-- The ⊥-seeded state at the empty / `hi = 0` window is `(⊥, 0, 0)` — the kernel's
preLoop init (`m_i = -inf`, `l_i`/`acc` ⊥-seeded `0`). -/
theorem afcStateBot_zero
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart : Nat) (i : Fin 128) (d : Fin 128) :
    afcStateBot qT kT vT keyScale qStart 0 i d = (⊥, 0, 0) := by
  unfold afcStateBot afcKeysUpto
  rw [show (List.finRange 128).filterMap
        (fun j : Fin 128 => if j.val < 0 ∧ j.val ≤ qStart + i.val
          then some (afcKV qT kT vT keyScale i d j) else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-- The ⊥-seeded running max at the empty / `hi = 0` window is `⊥`. -/
theorem afcRunningMax_zero
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart : Nat) (i : Fin 128) (d : Fin 128) :
    afcRunningMax qT kT vT keyScale qStart 0 i d = ⊥ := by
  unfold afcRunningMax afcKeysUpto
  rw [show (List.finRange 128).filterMap
        (fun j : Fin 128 => if j.val < 0 ∧ j.val ≤ qStart + i.val
          then some (afcKV qT kT vT keyScale i d j) else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-- Generic threshold-split for a `.val`-ascending list. Ported from flash-attn. -/
private theorem afc_filterMap_window_split {n : Nat} (l : List (Fin n))
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

/-- **Window split** (`hi = c·64`): keys streamed through `c+1` blocks = keys
through `c` blocks ++ block `c`. -/
theorem afcKeysUpto_succ
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart c : Nat) (i : Fin 128) (d : Fin 128) :
    afcKeysUpto qT kT vT keyScale qStart ((c + 1) * 64) i d
      = afcKeysUpto qT kT vT keyScale qStart (c * 64) i d
        ++ afcBlock qT kT vT keyScale qStart c i d := by
  unfold afcKeysUpto afcBlock
  rw [show (List.finRange 128).filterMap
        (fun j : Fin 128 => if j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val
          then some (afcKV qT kT vT keyScale i d j) else none)
      = (List.finRange 128).filterMap
        (fun j : Fin 128 => if j.val ≤ qStart + i.val ∧ j.val < (c + 1) * 64
          then some (afcKV qT kT vT keyScale i d j) else none)
      from List.filterMap_congr (fun j _ => by simp only [and_comm])]
  rw [afc_filterMap_window_split (List.finRange 128) (List.pairwise_lt_finRange 128)
    (c * 64) ((c + 1) * 64) (fun j => j.val ≤ qStart + i.val)
    (fun j => afcKV qT kT vT keyScale i d j) (by nlinarith [Nat.zero_le (64 : Nat)])]
  refine congrArg₂ (· ++ ·) ?_ ?_
  · apply List.filterMap_congr; intro j _; simp only [and_comm]
  · apply List.filterMap_congr; intro j _
    by_cases h1 : c * 64 ≤ j.val <;> by_cases h2 : j.val < (c + 1) * 64 <;>
      by_cases h3 : j.val ≤ qStart + i.val <;> simp [h1, h2, h3, and_assoc]

/-- **One-block advance**: `afcStateBot` after `c+1` blocks is `osStepBot`-folded
over block `c`'s keys from `afcStateBot` after `c` blocks. -/
theorem afcStateBot_succ
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart c : Nat) (i : Fin 128) (d : Fin 128) :
    afcStateBot qT kT vT keyScale qStart ((c + 1) * 64) i d
      = (afcBlock qT kT vT keyScale qStart c i d).foldl osStepBot
          (afcStateBot qT kT vT keyScale qStart (c * 64) i d) := by
  unfold afcStateBot
  rw [afcKeysUpto_succ, List.foldl_append]

/-- The running max one-block advance: `afcRunningMax((c+1)·64) = afcRunningMax(c·64) ⊔ blockSup`. -/
theorem afcRunningMax_succ
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart c : Nat) (i : Fin 128) (d : Fin 128) :
    afcRunningMax qT kT vT keyScale qStart ((c + 1) * 64) i d
      = afcRunningMax qT kT vT keyScale qStart (c * 64) i d
        ⊔ ((afcBlock qT kT vT keyScale qStart c i d).map
            (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  unfold afcRunningMax
  rw [afcKeysUpto_succ, List.map_append]
  induction (afcKeysUpto qT kT vT keyScale qStart (c * 64) i d) with
  | nil => simp
  | cons a t ih => simp only [List.map_cons, List.foldr_cons, List.cons_append, ih, max_assoc]

/-- **The block-at-once update equals the key-by-key `osStepBot` fold.** Ported
from the flash-attn foundation (`osStepBot_block_eq`). For a block with max
`M' = m ⊔ blockSup` and a state `(m, l, acc)` anchored via `l = κ(m)·L`,
`acc = κ(m)·T`, the kernel's one-shot rescale-and-add lands on
`block.foldl osStepBot (m, l, acc)`. -/
theorem osStepBot_block_eq (m : WithBot ℝ) (l acc T L : ℝ) (block : List (ℝ × ℝ))
    (hl : l = (m.elim 0 (fun r => pow2 (-r))) * L)
    (hacc : acc = (m.elim 0 (fun r => pow2 (-r))) * T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0) :
    let M' := m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
    (M',
     l * (WithBot.realExp2 (WithBot.realSub m M')).unbotD 0
       + (block.map (fun p => pow2 (p.1 - M'.unbotD 0))).sum,
     acc * (WithBot.realExp2 (WithBot.realSub m M')).unbotD 0
       + (block.map (fun p => pow2 (p.1 - M'.unbotD 0) * p.2)).sum)
      = block.foldl osStepBot (m, l, acc) := by
  intro M'
  have hfst : (block.foldl osStepBot (m, l, acc)).1 = M' := by
    rw [afcStateBot_fst]
    have gen : ∀ (m0 : WithBot ℝ) (L0 : List (WithBot ℝ)),
        L0.foldl (· ⊔ ·) m0 = m0 ⊔ L0.foldr (· ⊔ ·) ⊥ := by
      intro m0 L0
      induction L0 generalizing m0 with
      | nil => simp
      | cons a t ih => simp only [List.foldl_cons, List.foldr_cons, ih]; rw [max_assoc]
    rw [gen]
  obtain ⟨hfold_l, hfold_acc⟩ := osStepBot_foldl_consistent block m l acc T L hl hacc hmL hmT
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
      have := sum_map_pow2_sub ((↑Mr : WithBot ℝ).unbotD 0) block (fun _ => 1)
      simp only [mul_one] at this
      rw [this, WithBot.unbotD_coe]
    have hsumT : (block.map (fun p => pow2 (p.1 - (↑Mr : WithBot ℝ).unbotD 0) * p.2)).sum
        = pow2 (-Mr) * (block.map (fun p => pow2 p.1 * p.2)).sum := by
      rw [sum_map_pow2_sub ((↑Mr : WithBot ℝ).unbotD 0) block (fun p => p.2), WithBot.unbotD_coe]
    refine Prod.ext hfst.symm (Prod.ext ?_ ?_)
    · rw [hfold_l, hlα, hsumL, show ((↑Mr : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-Mr) from rfl]; ring
    · rw [hfold_acc, haccα, hsumT, show ((↑Mr : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-Mr) from rfl]; ring

/-- At the full window `hi = 128`, the causal ⊥-seeded key list is exactly the
predicate-filtered `attnKeyListPred` with `causalKeep qStart` — the list the
streaming spec `attnFwdCausalOutSpec_eq_streaming` folds. -/
theorem afcKeysUpto_full_eq_pred
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart : Nat) (i : Fin 128) (d : Fin 128) :
    afcKeysUpto qT kT vT keyScale qStart 128 i d
      = attnKeyListPred qT kT vT keyScale (fun a b => causalKeep qStart a b) i d := by
  unfold afcKeysUpto attnKeyListPred afcKV
  apply List.filterMap_congr
  intro j _
  have hjlt : j.val < 128 := j.isLt
  have hiff : causalKeep qStart i j ↔ j.val ≤ qStart + i.val := by
    unfold causalKeep; omega
  by_cases hc : causalKeep qStart i j
  · rw [if_pos ⟨hjlt, hiff.mp hc⟩, if_pos hc]
  · rw [if_neg (fun hh => hc (hiff.mpr hh.2)), if_neg hc]

/-- **The ⊥-seeded full-window state reads off the genuine closed-form spec.**
`afcStateBot(128).acc / afcStateBot(128).denom = attnFwdCausalOutSpec`. -/
theorem afcStateBot_full_eq_spec
    (s : BlockState) (Q K V : RegionName) (keyScale : Fin 128 → ℝ) (i d : Fin 128)
    (hne : afcRunningMax (qTileAFCm s Q) (kTileAFC s K) (vTileAFCm s V) keyScale
      (qStartAFC s) 128 i d ≠ ⊥) :
    (let st := afcStateBot (qTileAFCm s Q) (kTileAFC s K) (vTileAFCm s V) keyScale
        (qStartAFC s) 128 i d
     st.2.2 / st.2.1)
      = attnFwdCausalOutSpec s Q K V keyScale (i, d, PUnit.unit) := by
  simp only
  rw [afcStateBot_ratio_eq _ _ _ _ _ _ _ _ hne]
  rw [afcKeysUpto_full_eq_pred]
  rw [attnFwdCausalOutSpec_eq_streaming]
  rw [VeriTile.Triton.osStep_foldl_eq_batch]

/-! ### `l_i = 1` seed reconciliation

The kernel seeds `l_i = tl.zeros + 1.0` (not `0`). On the first streamed key the
running max transitions from `⊥`, forcing `α = (realExp2 (realSub ⊥ m')).unbotD 0
= 0`, which annihilates the `1` carry. Hence the seed-`1` fold and the seed-`0`
fold agree on every nonempty key prefix. `afcStateBot1` is the faithful seed-`1`
state; it equals `afcStateBot` on nonempty windows. Ported from the triton3
foundation (`aft3StateBot1`). -/

/-- **⊥-seed independence of the `l`/`acc` carries.** From a `⊥`-max start the
first key resets the carries, so the `osStepBot` fold over a nonempty list is
independent of the initial `l`/`acc`. -/
theorem osStepBot_bot_seed_indep (xs : List (ℝ × ℝ)) (hne : xs ≠ [])
    (l acc l' acc' : ℝ) :
    xs.foldl osStepBot (⊥, l, acc) = xs.foldl osStepBot (⊥, l', acc') := by
  obtain ⟨x, t, rfl⟩ := List.exists_cons_of_ne_nil hne
  obtain ⟨s, v⟩ := x
  have hstep : ∀ L A : ℝ, osStepBot (⊥, L, A) (s, v)
      = (((s : ℝ) : WithBot ℝ), pow2 (s - s), pow2 (s - s) * v) := by
    intro L A
    simp only [osStepBot, bot_sup_eq]
    have hα : (WithBot.realExp2 (WithBot.realSub (⊥ : WithBot ℝ) ((s : ℝ) : WithBot ℝ))).unbotD 0 = 0 := by
      rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
    have hub : (((s : ℝ) : WithBot ℝ)).unbotD 0 = s := by rfl
    rw [hα, hub]
    simp
  simp only [List.foldl_cons, hstep]

/-- ⊥-seeded running state from the kernel's `l_i = 1` seed. -/
noncomputable def afcStateBot1
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128) : WithBot ℝ × ℝ × ℝ :=
  (afcKeysUpto qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 1, 0)

/-- The faithful seed-`1` state equals the seed-`0` state whenever the window is
nonempty (`afcRunningMax ≠ ⊥`). -/
theorem afcStateBot1_eq_afcStateBot
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128)
    (hne : afcRunningMax qT kT vT keyScale qStart hi i d ≠ ⊥) :
    afcStateBot1 qT kT vT keyScale qStart hi i d
      = afcStateBot qT kT vT keyScale qStart hi i d := by
  have hxs : afcKeysUpto qT kT vT keyScale qStart hi i d ≠ [] := by
    intro h
    apply hne
    unfold afcRunningMax
    rw [h]; rfl
  unfold afcStateBot1 afcStateBot
  exact osStepBot_bot_seed_indep _ hxs 1 0 0 0

/-- The seed-`1` running `max` is `afcRunningMax` (the `l`-seed does not affect the
max channel). -/
theorem afcStateBot1_fst_eq_runningMax
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128) :
    (afcStateBot1 qT kT vT keyScale qStart hi i d).1
      = afcRunningMax qT kT vT keyScale qStart hi i d := by
  rw [afcStateBot1, afcStateBot_fst, afcRunningMax, afc_foldl_sup_bot_eq_foldr]

/-- The seed-`1` state at the empty / `hi = 0` window is `(⊥, 1, 0)` — the kernel's
preLoop init (`m_i = -inf`, `l_i = 1`, `acc = 0`). -/
theorem afcStateBot1_zero
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart : Nat) (i : Fin 128) (d : Fin 128) :
    afcStateBot1 qT kT vT keyScale qStart 0 i d = (⊥, 1, 0) := by
  unfold afcStateBot1 afcKeysUpto
  rw [show (List.finRange 128).filterMap
        (fun j : Fin 128 => if j.val < 0 ∧ j.val ≤ qStart + i.val
          then some (afcKV qT kT vT keyScale i d j) else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-- **One-block advance for the seed-`1` state**: `afcStateBot1` after `c+1` blocks
is `osStepBot`-folded over block `c`'s keys from `afcStateBot1` after `c` blocks. -/
theorem afcStateBot1_succ
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart c : Nat) (i : Fin 128) (d : Fin 128) :
    afcStateBot1 qT kT vT keyScale qStart ((c + 1) * 64) i d
      = (afcBlock qT kT vT keyScale qStart c i d).foldl osStepBot
          (afcStateBot1 qT kT vT keyScale qStart (c * 64) i d) := by
  unfold afcStateBot1
  rw [afcKeysUpto_succ, List.foldl_append]

/-! ## FOUNDATION Part 1 — `afcBody_split` (preLoop ++ forRange afcLoopBody :: postLoop)

The lowered algorithm body of `attn_fwd_causal_surface` at the Python test shape is
a 25-statement list: 22 preLoop statements (`afcPreLoop`), then the static
`Stmt.forRange "start_n" 0 128 64 afcLoopBody` (loop body = 22 statements), then 2
postLoop statements (`acc = acc / l_i[:, None]` and the masked `tl.store`). This is
a **static** `forRange` (range bounds `0..128 step 64`, NOT a `forRangeDyn`), so the
`forRange_inv` master invariant principle drives the loop. Mirrors `flash_body_split`.

The three pieces are transcribed concretely (the per-statement op-eval recipes above
encode the exact `Op`/`Broadcast`/dtype terms); `afcBody_split` is checked by `rfl`. -/

namespace AfcFoundation

open VeriTile.Triton

/-- The 22 lowered loop-body statements (statements 0–21 of the `forRange` body),
matching the recipe op-eval lemmas `afc_*`. -/
def afcLoopBody : List Stmt :=
  [ -- 0: start_n = tl.multiple_of(start_n, BLOCK_N)  (identity)
    Stmt.assign .nat [] "start_n" (Op.ref .nat [] "start_n"),
    -- 1: k_mask
    Stmt.assign .bool [128, 64] "k_mask"
      (Op.boolAnd (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat 128) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat 96)))),
    -- 2: k = tl.load(K_ptrs, mask=k_mask)
    Stmt.assign .real [128, 64] "k"
      (Op.load .real (.ptr (.ref .ptr [128, 64] "K_ptrs")) (.mask (.ref .bool [128, 64] "k_mask"))),
    -- 3: k_scale = tl.load(K_scale_ptr)
    Stmt.assign .real [] "k_scale"
      (Op.load .real (.ptr (.ref .ptr [] "K_scale_ptr")) .none),
    -- 4: qk = castFloat(q·k) * q_scale * k_scale
    Stmt.assign .real [128, 64] "qk"
      (Op.mul .real Broadcast.scalarR
        (Op.mul .real Broadcast.scalarR
          (Op.castFloat FloatDType.real FloatDType.real
            (Op.dot (batch := []) (Op.ref .real [128, 128] "q") (Op.ref .real [128, 64] "k")))
          (Op.ref .real [] "q_scale"))
        (Op.ref .real [] "k_scale")),
    -- 5: mask = offs_m[:,None] >= start_n + offs_n[None,:]
    Stmt.assign .bool [128, 64] "mask"
      (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "offs_n")))),
    -- 6: qk = tl.where(mask, qk, -1000000.0)  (unary minus → sub (const 0.0) (const 1e6))
    Stmt.assign .real [128, 64] "qk"
      (Op.where (Op.ref .bool [128, 64] "mask")
        (Op.ref .real [128, 64] "qk")
        (Op.broadcast (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1000000.0)) [128, 64])),
    -- 7: m_ij = maximum(m_i, max(qk,1))
    Stmt.assign .real [128] "m_ij"
      (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [128] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false
            (Op.ref .real [128, 64] "qk")))
        (Op.ref .real [128] "m_i")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false
          (Op.ref .real [128, 64] "qk"))),
    -- 8: qk = qk - m_ij[:, None]
    Stmt.assign .real [128, 64] "qk"
      (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [128, 64] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "m_ij"))),
    -- 9: p = exp2(qk)
    Stmt.assign .real [128, 64] "p" (Op.exp2 (Op.ref .real [128, 64] "qk")),
    -- 10: p = tl.where(mask, p, 0)
    Stmt.assign .real [128, 64] "p"
      (Op.where (Op.ref .bool [128, 64] "mask")
        (Op.ref .real [128, 64] "p") (Op.broadcast (Op.const 0.0) [128, 64])),
    -- 11: l_ij = sum(p, 1)
    Stmt.assign .real [128] "l_ij"
      (Op.reduceSum (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false (Op.ref .real [128, 64] "p")),
    -- 12: alpha = exp2(m_i - m_ij)
    Stmt.assign .real [128] "alpha"
      (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [128] "m_i") (Op.ref .real [128] "m_ij"))),
    -- 13: l_i = l_i * alpha + l_ij
    Stmt.assign .real [128] "l_i"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [128] "l_i") (Op.ref .real [128] "alpha"))
        (Op.ref .real [128] "l_ij")),
    -- 14: acc = acc * alpha[:, None]
    Stmt.assign .real [128, 128] "acc"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [128, 128] "acc") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "alpha"))),
    -- 15: v = tl.load(V_ptrs, mask=...)  (v is [64,128]: rows=keys, cols=head)
    Stmt.assign .real [64, 128] "v"
      (Op.load .real (.ptr (.ref .ptr [64, 128] "V_ptrs"))
        (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [64] "offs_n"))
            (Op.sub .nat Broadcast.nil (Op.constNat 128) (Op.ref .nat [] "start_n")))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat 96)))))),
    -- 16: p = p.to(fp16)
    Stmt.assign .fp16 [128, 64] "p"
      (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [128, 64] "p")),
    -- 17: acc += dot(p.to(real), v)
    Stmt.assign .real [128, 128] "acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [128, 128] "acc")
        (Op.dot (batch := [])
          (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref .fp16 [128, 64] "p"))
          (Op.ref .real [64, 128] "v"))),
    -- 18: m_i = m_ij
    Stmt.assign .real [128] "m_i" (Op.ref .real [128] "m_ij"),
    -- 19: K_ptrs += BLOCK_N * HEAD_DIM
    Stmt.assign .ptr [128, 64] "K_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [128, 64] "K_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat 64) (Op.constNat 128))),
    -- 20: K_scale_ptr += 1
    Stmt.assign .ptr [] "K_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ref .ptr [] "K_scale_ptr") (Op.constNat 1)),
    -- 21: V_ptrs += BLOCK_N * HEAD_DIM
    Stmt.assign .ptr [64, 128] "V_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [64, 128] "V_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat 64) (Op.constNat 128))) ]

end AfcFoundation

namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal

open VeriTile.Triton

set_option maxRecDepth 8000 in
/-- The lowered `forRange` loop body of the Python-shape AFC kernel is exactly
`afcLoopBody` (22 statements). Checked by `rfl`. -/
theorem afcLoopBody_check
    (Q K V QScale KScale Out : RegionName) :
    (match ((attn_fwd_causal_surface Q K V QScale KScale Out
        65536 16384 128 1 65536 16384 128 1 65536 16384 128 1 65536 16384 128 1
        2 4 128 128 128 64 128 96 1).toAlgKernel.body)[22]? with
      | some (Stmt.forRange _ _ _ _ body) => body
      | _ => [])
      = AfcFoundation.afcLoopBody :=
  rfl

set_option maxRecDepth 8000 in
/-- **`afcBody_split`** — the lowered AFC body decomposes as
`take 22 ++ (forRange "start_n" 0 128 64 afcLoopBody :: drop 23)`. The static
`forRange` (NOT `forRangeDyn`) sits at index 22; the 2 postLoop statements follow.
Pure structural identity, checked by `rfl`. -/
theorem afcBody_split
    (Q K V QScale KScale Out : RegionName) :
    (attn_fwd_causal_surface Q K V QScale KScale Out
        65536 16384 128 1 65536 16384 128 1 65536 16384 128 1 65536 16384 128 1
        2 4 128 128 128 64 128 96 1).toAlgKernel.body
      = (attn_fwd_causal_surface Q K V QScale KScale Out
          65536 16384 128 1 65536 16384 128 1 65536 16384 128 1 65536 16384 128 1
          2 4 128 128 128 64 128 96 1).toAlgKernel.body.take 22
        ++ (Stmt.forRange "start_n" 0 128 64 AfcFoundation.afcLoopBody
            :: (attn_fwd_causal_surface Q K V QScale KScale Out
                65536 16384 128 1 65536 16384 128 1 65536 16384 128 1 65536 16384 128 1
                2 4 128 128 128 64 128 96 1).toAlgKernel.body.drop 23) :=
  rfl

/-! ## FOUNDATION Part 4 — `afcPreLoop` AST + `afcInvariant` + `afcPreLoop_eval`

`afcPreLoop` is the 22-statement deterministic prefix (`= body.take 22`). The loop
invariant `afcInvariant` binds the running registers after `c` blocks (counter
`i = c·64`): `m_i`/`l_i`/`acc` to the three components of the ⊥-seeded `afcStateBot`
(running max = `afcRunningMax`), together with the static index vectors, the loaded
`q`/`q_scale`, and the four streamed pointer tiles (`K_ptrs`/`K_scale_ptr`/`V_ptrs`
advanced by `i`, `O_block_ptr`/`Q_ptrs`/`Q_scale_ptr` fixed). `afcPreLoop_eval`
steps the prefix to a state satisfying `afcInvariant … 0`. -/

namespace AfcFoundation

open VeriTile.Triton

/-- The 22 lowered preLoop statements of the Python-shape AFC kernel (`= body.take 22`). -/
def afcPreLoop (Q K V QScale KScale Out : RegionName) : List Stmt :=
  [ Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [] "off_hz" (Op.programId 1),
    Stmt.assign .nat [] "off_z"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 4)),
    Stmt.assign .nat [] "off_h"
      (Op.mod .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 4)),
    Stmt.assign .nat [] "qvk_offset"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat 65536))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat 16384))),
    Stmt.assign .nat [] "vk_offset"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "qvk_offset") (Op.constNat 128)),
    Stmt.assign .nat [] "q_scale_offset"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat 128) (Op.constNat 128)) (Op.constNat 1))
          (Op.constNat 128))),
    Stmt.assign .nat [] "k_scale_offset"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat 128) (Op.constNat 64)) (Op.constNat 1))
          (Op.constNat 64))),
    Stmt.assign .nat [128] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128)) (Op.arange 128)),
    Stmt.assign .nat [64] "offs_n" (Op.arange 64),
    Stmt.assign .nat [128] "offs_k" (Op.arange 128),
    Stmt.assign .ptr [128, 128] "Q_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) (Op.constNat 128)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [] "Q_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase QScale)
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "q_scale_offset") (Op.ref .nat [] "start_m"))),
    Stmt.assign .ptr [128, 64] "K_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_k")))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "offs_n")) (Op.constNat 128)))),
    Stmt.assign .ptr [] "K_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase KScale) (Op.ref .nat [] "k_scale_offset")),
    Stmt.assign .ptr [64, 128] "V_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [64] "offs_n")) (Op.constNat 128)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [128, 128] "O_block_ptr"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) (Op.constNat 128)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .real [128] "m_i"
      (Op.add .real Broadcast.scalarR (Op.full [128] (Op.const 0)) Op.negInf),
    Stmt.assign .real [128] "l_i"
      (Op.add .real Broadcast.scalarR (Op.full [128] (Op.const 0)) (Op.const 1.0)),
    Stmt.assign .real [128, 128] "acc" (Op.full [128, 128] (Op.const 0)),
    Stmt.assign .real [128, 128] "q"
      (Op.load .real (.ptr (.ref .ptr [128, 128] "Q_ptrs"))
        (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) (Op.constNat 128))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat 96)))))),
    Stmt.assign .real [] "q_scale"
      (Op.load .real (.ptr (.ref .ptr [] "Q_scale_ptr")) .none) ]

/-- **PreLoop head** — statements 0–10 of `afcPreLoop` (the scalar index/offset
setup: program ids, `off_z`/`off_h`, `qvk_offset`/`vk_offset`, the two scale
offsets, and the three index vectors `offs_m`/`offs_n`/`offs_k`). No pointer or
load statements; resolves entirely by scalar `setReg`-peeling. -/
def afcPreLoopHead : List Stmt :=
  [ Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [] "off_hz" (Op.programId 1),
    Stmt.assign .nat [] "off_z"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 4)),
    Stmt.assign .nat [] "off_h"
      (Op.mod .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 4)),
    Stmt.assign .nat [] "qvk_offset"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat 65536))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat 16384))),
    Stmt.assign .nat [] "vk_offset"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "qvk_offset") (Op.constNat 128)),
    Stmt.assign .nat [] "q_scale_offset"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat 128) (Op.constNat 128)) (Op.constNat 1))
          (Op.constNat 128))),
    Stmt.assign .nat [] "k_scale_offset"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat 128) (Op.constNat 64)) (Op.constNat 1))
          (Op.constNat 64))),
    Stmt.assign .nat [128] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128)) (Op.arange 128)),
    Stmt.assign .nat [64] "offs_n" (Op.arange 64),
    Stmt.assign .nat [128] "offs_k" (Op.arange 128) ]

/-- **PreLoop tail** — statements 11–21 of `afcPreLoop` (the four streamed pointer
tiles `Q_ptrs`/`Q_scale_ptr`/`K_ptrs`/`K_scale_ptr`/`V_ptrs`/`O_block_ptr`, the
running-state seeds `m_i`/`l_i`/`acc`, and the masked `q` / scalar `q_scale`
loads). Steps from a state where the head registers are already bound. -/
def afcPreLoopTail (Q K V QScale KScale Out : RegionName) : List Stmt :=
  [ Stmt.assign .ptr [128, 128] "Q_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) (Op.constNat 128)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [] "Q_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase QScale)
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "q_scale_offset") (Op.ref .nat [] "start_m"))),
    Stmt.assign .ptr [128, 64] "K_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_k")))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "offs_n")) (Op.constNat 128)))),
    Stmt.assign .ptr [] "K_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase KScale) (Op.ref .nat [] "k_scale_offset")),
    Stmt.assign .ptr [64, 128] "V_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [64] "offs_n")) (Op.constNat 128)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [128, 128] "O_block_ptr"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) (Op.constNat 128)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .real [128] "m_i"
      (Op.add .real Broadcast.scalarR (Op.full [128] (Op.const 0)) Op.negInf),
    Stmt.assign .real [128] "l_i"
      (Op.add .real Broadcast.scalarR (Op.full [128] (Op.const 0)) (Op.const 1.0)),
    Stmt.assign .real [128, 128] "acc" (Op.full [128, 128] (Op.const 0)),
    Stmt.assign .real [128, 128] "q"
      (Op.load .real (.ptr (.ref .ptr [128, 128] "Q_ptrs"))
        (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) (Op.constNat 128))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat 96)))))),
    Stmt.assign .real [] "q_scale"
      (Op.load .real (.ptr (.ref .ptr [] "Q_scale_ptr")) .none) ]

/-- `afcPreLoop` decomposes as `afcPreLoopHead ++ afcPreLoopTail`. Checked by `rfl`. -/
theorem afcPreLoop_eq_head_tail (Q K V QScale KScale Out : RegionName) :
    afcPreLoop Q K V QScale KScale Out
      = afcPreLoopHead ++ afcPreLoopTail Q K V QScale KScale Out :=
  rfl

end AfcFoundation

namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal

open VeriTile.Triton

set_option maxRecDepth 8000 in
/-- `body.take 22 = afcPreLoop`. Checked by `rfl`. -/
theorem afcPreLoop_check (Q K V QScale KScale Out : RegionName) :
    (attn_fwd_causal_surface Q K V QScale KScale Out
        65536 16384 128 1 65536 16384 128 1 65536 16384 128 1 65536 16384 128 1
        2 4 128 128 128 64 128 96 1).toAlgKernel.body.take 22
      = AfcFoundation.afcPreLoop Q K V QScale KScale Out :=
  rfl

/-! ### Streamed-pointer closed cell-forms

The loop streams `K_ptrs`/`K_scale_ptr`/`V_ptrs` forward by `64·128 = 8192` (resp.
`1`) per block. After `c` blocks the closed cell-forms below hold; the preLoop
base is the `c = 0` instance and the loop-body advance maps `c → c+1`. -/

/-- `K_ptrs` after `c` blocks: transposed key tile `[BD, BN]`, cell `(e, j)`
addresses `K[baseOffset + e + (c·64 + j)·128]`. -/
noncomputable def kPtrsAFC (s0 : BlockState) (K : RegionName) (c : Nat) :
    Tile .ptr [128, 64] :=
  ⟨fun idx : TileIndex [128, 64] => (K.cast, baseOffsetAFC s0 + idx.1.val + (c * 64 + idx.2.1.val) * 128)⟩

/-- `K_scale_ptr` after `c` blocks: scalar pointer addressing `KScale[k_scale_offset + c]`. -/
noncomputable def kScalePtrAFC (s0 : BlockState) (KScale : RegionName) (c : Nat) :
    Tile .ptr [] :=
  ⟨fun _ : TileIndex [] => (KScale.cast, s0.pids 1 * ((128 + 64 - 1) / 64) + c)⟩

/-- `V_ptrs` after `c` blocks: value tile `[BN, BD]`, cell `(j, d)` addresses
`V[baseOffset + (c·64 + j)·128 + d]`. -/
noncomputable def vPtrsAFC (s0 : BlockState) (V : RegionName) (c : Nat) :
    Tile .ptr [64, 128] :=
  ⟨fun idx : TileIndex [64, 128] => (V.cast, baseOffsetAFC s0 + (c * 64 + idx.1.val) * 128 + idx.2.1.val)⟩

/-- `O_block_ptr` (constant through the loop): output tile `[BM, BD]`, cell `(i, e)`
addresses `Out[baseOffset + (qStart + i)·128 + e]` (the preLoop-tail stmt 16
`ptrAdd (ptrBase Out) (qvk + offs_m[:,None]·128 + offs_k[None,:]·1)`). Mirrors the
`outOffset` cell-form at the Python test shape. -/
noncomputable def oBlockPtrAFC (s0 : BlockState) (Out : RegionName) :
    Tile .ptr [128, 128] :=
  ⟨fun idx : TileIndex [128, 128] => (Out.cast, baseOffsetAFC s0 + (s0.pids 0 * 128 + idx.1.val) * 128 + idx.2.1.val)⟩

/-- One-block advance of `K_ptrs`: the loop-body `ptrAdd … (64·128)` maps
`kPtrsAFC c → kPtrsAFC (c+1)`. -/
theorem kPtrsAFC_succ (s0 : BlockState) (K : RegionName) (c : Nat) :
    Tile.ptrAdd Broadcast.scalarR (kPtrsAFC s0 K c)
        (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar 64) (Tile.scalar 128))
      = kPtrsAFC s0 K (c + 1) := by
  ext idx
  · rfl
  · simp only [kPtrsAFC, Tile.ptrAdd_data, Tile.bop_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR, Broadcast.leftIndex_nil,
      Broadcast.rightIndex_nil, NumericDType.nat_mul]
    ring

/-- One-block advance of `K_scale_ptr`: the loop-body `ptrAdd … 1` maps
`kScalePtrAFC c → kScalePtrAFC (c+1)`. -/
theorem kScalePtrAFC_succ (s0 : BlockState) (KScale : RegionName) (c : Nat) :
    Tile.ptrAdd Broadcast.nil (kScalePtrAFC s0 KScale c) (Tile.scalar 1)
      = kScalePtrAFC s0 KScale (c + 1) := by
  ext idx
  · rfl
  · simp only [kScalePtrAFC, Tile.ptrAdd_data, Tile.scalar,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    omega

/-- One-block advance of `V_ptrs`: the loop-body `ptrAdd … (64·128)` maps
`vPtrsAFC c → vPtrsAFC (c+1)`. -/
theorem vPtrsAFC_succ (s0 : BlockState) (V : RegionName) (c : Nat) :
    Tile.ptrAdd Broadcast.scalarR (vPtrsAFC s0 V c)
        (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar 64) (Tile.scalar 128))
      = vPtrsAFC s0 V (c + 1) := by
  ext idx
  · rfl
  · simp only [vPtrsAFC, Tile.ptrAdd_data, Tile.bop_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR, Broadcast.leftIndex_nil,
      Broadcast.rightIndex_nil, NumericDType.nat_mul]
    ring

/-- **Sentinel boundedness side-condition.** For a faithful bounded-input kernel,
every causally-kept key's coerced score exceeds the `-1e6` masking sentinel — i.e.
the streamed running max is never `⊥` once a key is kept, and the kept scores stay
above `-1e6`. Captured as: at the full window, the running max is `> -1e6`
(equivalently the masked-block `max(m_i, -1e6)` agrees with `afcRunningMax`). This
is a legitimate precondition for bounded `Q`/`K` (analogous to #316's `undef = 0` /
`M ≠ Out` preconditions). -/
def afcScoreBound
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ) (qStart : Nat) : Prop :=
  ∀ (j : Fin 128) (i d : Fin 128),
    keyScale j * Finset.univ.sum (fun e : Fin 128 => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit))
      > -1000000.0

/-- **Loop invariant** for the AFC streaming loop (counter `i = c·64`, window
`hi_c = i`). Binds the running-state registers after `c` blocks to the seed-`1`
⊥-seeded `afcStateBot1`/`afcRunningMax` over the first `i` keys, per output row `r`
(channel `d` for `acc`), keyed by the per-key score scale `keyScale`. Also binds
the static index vectors (`offs_m`/`offs_n`/`offs_k`), the loaded `q`/`q_scale`,
the program ids, and the three streamed pointers (`K_ptrs`/`K_scale_ptr`/`V_ptrs`
advanced by `c = i/64`), and preserves `undef`/`mem`. -/
noncomputable def afcInvariant
    (Q K V QScale KScale Out : RegionName) (s0 : BlockState)
    (keyScale : Fin 128 → ℝ) (i : Nat) (s : BlockState) : Prop :=
  let qStart := qStartAFC s0
  let qT := qTileAFCm s0 Q
  let kT := kTileAFC s0 K
  let vT := vTileAFCm s0 V
  s.pids = s0.pids ∧ i % 64 = 0 ∧ i ≤ 128 ∧
  (s.regs .real [128] "m_i" = some ⟨fun r : TileIndex [128] =>
      afcRunningMax qT kT vT keyScale qStart i r.1 ⟨0, by norm_num⟩⟩) ∧
  (s.regs .real [128] "l_i" = some ⟨fun r : TileIndex [128] =>
      ((afcStateBot1 qT kT vT keyScale qStart i r.1 ⟨0, by norm_num⟩).2.1 : ℝ)⟩) ∧
  (s.regs .real [128, 128] "acc" = some ⟨fun idx : TileIndex [128, 128] =>
      ((afcStateBot1 qT kT vT keyScale qStart i idx.1 idx.2.1).2.2 : ℝ)⟩) ∧
  (s.regs .nat [128] "offs_m" = some (Tile.vec (fun r : Fin 128 => qStart + r.val))) ∧
  (s.regs .nat [64] "offs_n" = some (Tile.vec (fun j : Fin 64 => j.val))) ∧
  (s.regs .real [128, 128] "q" = some ⟨fun idx : TileIndex [128, 128] =>
      if qStart + idx.1.val < 128 ∧ idx.2.1.val < 96 then some (qTileAFC s0 Q idx) else some (0.0 : ℝ)⟩) ∧
  (s.regs .real [] "q_scale" = some (Tile.scalar
      (some (s0.readMem QScale (s0.pids 1 * ((128 + 128 - 1) / 128) + s0.pids 0))))) ∧
  (s.regs .ptr [128, 64] "K_ptrs" = some (kPtrsAFC s0 K (i / 64))) ∧
  (s.regs .ptr [] "K_scale_ptr" = some (kScalePtrAFC s0 KScale (i / 64))) ∧
  (s.regs .ptr [64, 128] "V_ptrs" = some (vPtrsAFC s0 V (i / 64))) ∧
  (s.regs .ptr [128, 128] "O_block_ptr" = some (oBlockPtrAFC s0 Out)) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.AfcInvariantBase

open VeriTile.Triton VeriTile.Bench.TritonBenchG.AttnFwdCausal

/-- The running-state bindings of `afcInvariant … 0` are the ⊥-seed inits
(`m_i = ⊥`, `l_i = 0`, `acc = 0`) — the base case the preLoop establishes. Pure
math (reads off `afcRunningMax_zero`/`afcStateBot_zero`); the exec preLoop step
supplies the register equalities, this supplies the value normalization. -/
theorem afcInvariant_running_zero
    (Q K V : RegionName) (s0 : BlockState) (keyScale : Fin 128 → ℝ) :
    (⟨fun r : TileIndex [128] =>
        afcRunningMax (qTileAFCm s0 Q) (kTileAFC s0 K) (vTileAFCm s0 V) keyScale
          (qStartAFC s0) 0 r.1 ⟨0, by norm_num⟩⟩ : Tile .real [128])
        = ⟨fun _ : TileIndex [128] => (⊥ : WithBot ℝ)⟩
      ∧ (⟨fun r : TileIndex [128] =>
        ((afcStateBot1 (qTileAFCm s0 Q) (kTileAFC s0 K) (vTileAFCm s0 V) keyScale
          (qStartAFC s0) 0 r.1 ⟨0, by norm_num⟩).2.1 : ℝ)⟩ : Tile .real [128])
        = ⟨fun _ : TileIndex [128] => (some (1 : ℝ) : WithBot ℝ)⟩
      ∧ (⟨fun idx : TileIndex [128, 128] =>
        ((afcStateBot1 (qTileAFCm s0 Q) (kTileAFC s0 K) (vTileAFCm s0 V) keyScale
          (qStartAFC s0) 0 idx.1 idx.2.1).2.2 : ℝ)⟩ : Tile .real [128, 128])
        = ⟨fun _ : TileIndex [128, 128] => (some (0 : ℝ) : WithBot ℝ)⟩ := by
  refine ⟨?_, ?_, ?_⟩
  · ext r; simp only [afcRunningMax_zero]
  · ext r; simp only [afcStateBot1_zero]; rfl
  · ext idx; simp only [afcStateBot1_zero]; rfl

end VeriTile.Bench.TritonBenchG.AttnFwdCausal.AfcInvariantBase

/-- `qvk_offset` value for the AFC Python test shape: `(pid1/4)·65536 + (pid1%4)·16384`. -/
def qvkOffAFC (s : BlockState) : Nat :=
  s.pids 1 / 4 * 65536 + s.pids 1 % 4 * 16384

theorem qvkOffAFC_eq_baseOffset (s : BlockState) : qvkOffAFC s = baseOffsetAFC s := by
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **PreLoop head execution.** The 11 scalar/index statements (`afcPreLoopHead`)
step a clean state `s` to a state `s1` that binds `start_m`/`off_hz` (program
ids), the scalar offsets `qvk_offset`/`q_scale_offset`, and the three index
vectors `offs_m`/`offs_n`/`offs_k`, and preserves `pids`/`mem`/`undef`. These are
exactly the readbacks the tail (`afcPreLoopTail`) consumes. -/
theorem afcPreLoopHead_eval
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s1, stepStmts AfcFoundation.afcPreLoopHead s = some s1
      ∧ s1.pids = s.pids ∧ s1.mem = s.mem ∧ (∀ rg o, s1.undef rg o = 0)
      ∧ s1.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s1.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 1))
      ∧ s1.regs .nat [] "qvk_offset" = some (Tile.scalar (qvkOffAFC s))
      ∧ s1.regs .nat [] "q_scale_offset"
          = some (Tile.scalar (s.pids 1 * ((128 + 128 - 1) / 128)))
      ∧ s1.regs .nat [] "k_scale_offset"
          = some (Tile.scalar (s.pids 1 * ((128 + 64 - 1) / 64)))
      ∧ s1.regs .nat [128] "offs_m" = some (Tile.vec (fun r : Fin 128 => s.pids 0 * 128 + r.val))
      ∧ s1.regs .nat [64] "offs_n" = some (Tile.vec (fun j : Fin 64 => j.val))
      ∧ s1.regs .nat [128] "offs_k" = some (Tile.vec (fun e : Fin 128 => e.val)) := by
  unfold AfcFoundation.afcPreLoopHead
  -- stmt 0: start_m = programId 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  -- stmt 1: off_hz = programId 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  -- stmt 2: off_z = off_hz / 4
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 4)) _
        = some (Tile.scalar (s.pids 1 / 4)) from by
      rw [evalOp]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 3: off_h = off_hz % 4
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mod .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 4)) _
        = some (Tile.scalar (s.pids 1 % 4)) from by
      rw [evalOp]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 4: qvk_offset = off_z*65536 + off_h*16384
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat 65536))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat 16384))) _
        = some (Tile.scalar (qvkOffAFC s)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 5: vk_offset = qvk_offset / 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "qvk_offset") (Op.constNat 128)) _
        = some (Tile.scalar (qvkOffAFC s / 128)) from by
      rw [evalOp]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 6: q_scale_offset = off_hz * ((128+128-1)/128)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat 128) (Op.constNat 128)) (Op.constNat 1))
          (Op.constNat 128))) _
        = some (Tile.scalar (s.pids 1 * ((128 + 128 - 1) / 128))) from by
      rw [evalOp_mul, evalOp_div, evalOp_sub, evalOp_add]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 7: k_scale_offset = off_hz * ((128+64-1)/64)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat 128) (Op.constNat 64)) (Op.constNat 1))
          (Op.constNat 64))) _
        = some (Tile.scalar (s.pids 1 * ((128 + 64 - 1) / 64))) from by
      rw [evalOp_mul, evalOp_div, evalOp_sub, evalOp_add]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 8: offs_m = start_m*128 + arange 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128)) (Op.arange 128)) _
        = some (Tile.vec (fun r : Fin 128 => s.pids 0 * 128 + r.val)) from by
      rw [evalOp_add, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, evalOp_arange, BlockState.setReg_same,
        BlockState.setReg_pids, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 9: offs_n = arange 64
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange 64) _ = some (Tile.vec (fun j : Fin 64 => j.val)) from
      evalOp_arange 64 _))]
  -- stmt 10: offs_k = arange 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange 128) _ = some (Tile.vec (fun e : Fin 128 => e.val)) from
      evalOp_arange 128 _))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rfl  -- pids
  · rfl  -- mem
  · intro rg o; exact hundef rg o  -- undef
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]  -- start_m
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]  -- off_hz
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]  -- qvk_offset
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]  -- q_scale_offset
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]  -- k_scale_offset
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]  -- offs_m
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]  -- offs_n
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]  -- offs_k

/-- Lift a base-state register readback through a `setReg` to a different name —
used to thread the `s1` head readbacks (`offs_*`/`qvk_offset`) through the
pointer-register `setReg`s accumulated by the earlier tail statements, so the
`expandDim`/`ref` offset rewrites fire on the wrapped state. -/
theorem regs_setReg_chain {d d' : TileDType} {sh sh' : TileShape}
    {n n' : RegName} {s : BlockState} {v : Tile d sh} {w : Tile d' sh'}
    (hne : n ≠ n') (h : s.regs d sh n = some v) :
    (s.setReg n' d' sh' w).regs d sh n = some v := by
  simp only [BlockState.setReg_ne_name, ne_eq, hne, not_false_eq_true, h]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **PreLoop tail execution.** The 11 pointer/seed/load statements
(`afcPreLoopTail`) step an abstract head-exit state `s1` (with the head readbacks
supplied as hypotheses) to the loop-entry state `s0`, exposing the seeds
`m_i`/`l_i`/`acc`, the index vectors, the program-id scalars, and the loaded
`q_scale`, preserving `pids`/`mem`/`undef`. Operating on an abstract `s1` (rather
than a 22-deep `setReg` literal) lets the `expandDim`/`ref` offset proofs fire on
clean `ne_name`/`same` readbacks. -/
theorem afcPreLoopTail_eval
    (s1 : BlockState) (Q K V QScale KScale Out : RegionName)
    (hundef : ∀ rg o, s1.undef rg o = 0)
    (hstartm : s1.regs .nat [] "start_m" = some (Tile.scalar (s1.pids 0)))
    (hoffhz : s1.regs .nat [] "off_hz" = some (Tile.scalar (s1.pids 1)))
    (hqvk : s1.regs .nat [] "qvk_offset" = some (Tile.scalar (qvkOffAFC s1)))
    (hqso : s1.regs .nat [] "q_scale_offset"
        = some (Tile.scalar (s1.pids 1 * ((128 + 128 - 1) / 128))))
    (hkso : s1.regs .nat [] "k_scale_offset"
        = some (Tile.scalar (s1.pids 1 * ((128 + 64 - 1) / 64))))
    (hoffsm : s1.regs .nat [128] "offs_m"
        = some (Tile.vec (fun r : Fin 128 => s1.pids 0 * 128 + r.val)))
    (hoffsn : s1.regs .nat [64] "offs_n" = some (Tile.vec (fun j : Fin 64 => j.val)))
    (hoffsk : s1.regs .nat [128] "offs_k" = some (Tile.vec (fun e : Fin 128 => e.val))) :
    ∃ s0, stepStmts (AfcFoundation.afcPreLoopTail Q K V QScale KScale Out) s1 = some s0
      ∧ s0.pids = s1.pids ∧ s0.mem = s1.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "start_m" = some (Tile.scalar (s1.pids 0))
      ∧ s0.regs .nat [] "off_hz" = some (Tile.scalar (s1.pids 1))
      ∧ s0.regs .real [128] "m_i" = some ⟨fun _ : TileIndex [128] => (⊥ : WithBot ℝ)⟩
      ∧ s0.regs .real [128] "l_i" = some ⟨fun _ : TileIndex [128] => (some (1 : ℝ) : WithBot ℝ)⟩
      ∧ s0.regs .real [128, 128] "acc" = some ⟨fun _ : TileIndex [128, 128] => (some (0 : ℝ) : WithBot ℝ)⟩
      ∧ s0.regs .nat [128] "offs_m" = some (Tile.vec (fun r : Fin 128 => s1.pids 0 * 128 + r.val))
      ∧ s0.regs .nat [64] "offs_n" = some (Tile.vec (fun j : Fin 64 => j.val))
      ∧ s0.regs .nat [128] "offs_k" = some (Tile.vec (fun e : Fin 128 => e.val))
      ∧ s0.regs .real [] "q_scale" = some ⟨fun _ : TileIndex [] =>
          some (s1.readMem QScale ((s1.pids 1 * ((128 + 128 - 1) / 128) + s1.pids 0)))⟩
      ∧ s0.regs .ptr [128, 64] "K_ptrs" = some (kPtrsAFC s1 K 0)
      ∧ s0.regs .ptr [] "K_scale_ptr" = some (kScalePtrAFC s1 KScale 0)
      ∧ s0.regs .ptr [64, 128] "V_ptrs" = some (vPtrsAFC s1 V 0)
      ∧ s0.regs .ptr [128, 128] "O_block_ptr" = some (oBlockPtrAFC s1 Out)
      ∧ s0.regs .real [128, 128] "q" = some ⟨fun idx : TileIndex [128, 128] =>
          if s1.pids 0 * 128 + idx.1.val < 128 ∧ idx.2.1.val < 96
          then some (qTileAFC s1 Q idx) else some (0.0 : ℝ)⟩ := by
  unfold AfcFoundation.afcPreLoopTail
  -- stmt 11: Q_ptrs = ptrAdd (ptrBase Q) (qvk + offs_m[:,None]*128 + offs_k[None,:]*1)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (afc_evalOp_ptrAdd_of Broadcast.scalarL (Op.ptrBase Q) _ _ _ _
      (afc_evalOp_ptrBase Q _)
      (show evalOp (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) (Op.constNat 128)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_k")) (Op.constNat 1))) _
          = some _ from by
        rw [evalOp_add, evalOp_add, evalOp_mul, evalOp_mul]
        erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hoffsm,
          evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hoffsk]
        rw [evalOp_ref, hqvk]
        simp only [evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
        rfl)))]
  -- stmt 12: Q_scale_ptr = ptrAdd (ptrBase QScale) (q_scale_offset + start_m)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (afc_evalOp_ptrAdd_of Broadcast.nil (Op.ptrBase QScale) _ _ _ _
      (afc_evalOp_ptrBase QScale _)
      (show evalOp (Op.add .nat Broadcast.nil (Op.ref .nat [] "q_scale_offset") (Op.ref .nat [] "start_m")) _
          = some _ from by
        simp only [evalOp_add, evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, hqso, hstartm, Option.bind_some, Option.bind_eq_bind]
        rfl)))]
  -- stmt 13: K_ptrs = ptrAdd (ptrBase K) (qvk + offs_k[:,None] + offs_n[None,:]*128)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (afc_evalOp_ptrAdd_of Broadcast.scalarL (Op.ptrBase K) _ _ _ _
      (afc_evalOp_ptrBase K _)
      (show evalOp (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_k")))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "offs_n")) (Op.constNat 128))) _
          = some _ from by
        rw [evalOp_add, evalOp_add, evalOp_mul]
        erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hoffsk)),
            evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hoffsn))]
        rw [evalOp_ref, regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hqvk)]
        simp only [evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
        rfl)))]
  -- stmt 14: K_scale_ptr = ptrAdd (ptrBase KScale) k_scale_offset
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (afc_evalOp_ptrAdd_of Broadcast.nil (Op.ptrBase KScale) _ _ _ _
      (afc_evalOp_ptrBase KScale _)
      (show evalOp (Op.ref .nat [] "k_scale_offset") _ = some _ from by
        simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true]
        rw [hkso])))]
  -- stmt 15: V_ptrs = ptrAdd (ptrBase V) (qvk + offs_n[:,None]*128 + offs_k[None,:]*1)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (afc_evalOp_ptrAdd_of Broadcast.scalarL (Op.ptrBase V) _ _ _ _
      (afc_evalOp_ptrBase V _)
      (show evalOp (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [64] "offs_n")) (Op.constNat 128)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_k")) (Op.constNat 1))) _
          = some _ from by
        rw [evalOp_add, evalOp_add, evalOp_mul, evalOp_mul]
        erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                (regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hoffsn)))),
            evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                (regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hoffsk))))]
        rw [evalOp_ref, regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hqvk)))]
        simp only [evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
        rfl)))]
  -- stmt 16: O_block_ptr = ptrAdd (ptrBase Out) (qvk + offs_m[:,None]*128 + offs_k[None,:]*1)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (afc_evalOp_ptrAdd_of Broadcast.scalarL (Op.ptrBase Out) _ _ _ _
      (afc_evalOp_ptrBase Out _)
      (show evalOp (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) (Op.constNat 128)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_k")) (Op.constNat 1))) _
          = some _ from by
        rw [evalOp_add, evalOp_add, evalOp_mul, evalOp_mul]
        erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                  (regs_setReg_chain (by decide) hoffsm))))),
            evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                  (regs_setReg_chain (by decide) hoffsk)))))]
        rw [evalOp_ref, regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                (regs_setReg_chain (by decide) hqvk))))]
        simp only [evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
        rfl)))]
  -- stmt 17: m_i = full 0 + (-inf) = full ⊥
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [128] (Op.const 0)) Op.negInf) _
        = some (⟨fun _ : TileIndex [128] => (⊥ : WithBot ℝ)⟩ : Tile .real [128]) from by
      rw [evalOp_add]
      simp only [evalOp_full, evalOp_const, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map]
      rfl))]
  -- stmt 18: l_i = full 0 + 1.0 = full 1.0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [128] (Op.const 0)) (Op.const 1.0)) _
        = some (⟨fun _ : TileIndex [128] => (some (1 : ℝ) : WithBot ℝ)⟩ : Tile .real [128]) from by
      rw [evalOp_add]
      simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map]
      norm_num))]
  -- stmt 19: acc = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [128, 128] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [128, 128] => (some (0 : ℝ) : WithBot ℝ)⟩ : Tile .real [128, 128]) from by
      simp only [evalOp_full, evalOp_const]
      rfl))]
  -- stmt 20: q = load Q_ptrs (masked)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (afc_evalOp_load_ptr_mask_of (Op.ref .ptr [128, 128] "Q_ptrs") _ _ _ _
      (by rw [evalOp_ref]; rfl)
      (show evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) (Op.constNat 128))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat 96)))) _
          = some _ from by
        rw [afc_evalOp_boolAnd, evalOp_lt]
        erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _
            (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                  (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                    (regs_setReg_chain (by decide) hoffsm))))))))),
          evalOp_expandDim]
        simp only [evalOp_lt, evalOp_arange, evalOp_constNat,
          Option.bind_some, Option.bind_eq_bind]
        rfl)))]
  -- stmt 21: q_scale = load Q_scale_ptr (unmasked scalar)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (afc_evalOp_load_ptr_none_of (Op.ref .ptr [] "Q_scale_ptr") _ _
      (by rw [evalOp_ref]; rfl)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rfl  -- pids
  · rfl  -- mem
  · intro rg o; exact hundef rg o  -- undef
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids, hstartm]  -- start_m
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids, hoffhz]  -- off_hz
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]  -- m_i
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]  -- l_i
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]  -- acc
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids, hoffsm]  -- offs_m
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, hoffsn]  -- offs_n
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, hoffsk]  -- offs_k
  · -- q_scale
    simp only [BlockState.setReg_same, BlockState.setReg_pids]
    refine congrArg some ?_
    ext _
    simp only [BlockState.readMem, BlockState.setReg_mem, castTile_self,
      Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
      Tile.bop_data, Region.cast, NumericDType.nat_add, Nat.zero_add]
  · -- K_ptrs = kPtrsAFC s1 K 0
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    · simp only [kPtrsAFC, Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex_scalarL,
        Broadcast.rightIndex_scalarL, Region.cast]
    · simp only [kPtrsAFC, Tile.ptrAdd_data, Tile.scalar, Tile.bop_data, Tile.expandDim_data,
        Tile.vec_data, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
        Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, NumericDType.nat_add,
        NumericDType.nat_mul, qvkOffAFC, baseOffsetAFC]
      ring_nf
  · -- K_scale_ptr = kScalePtrAFC s1 KScale 0
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    · simp only [kScalePtrAFC, Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex_nil,
        Broadcast.rightIndex_nil, Region.cast]
    · simp only [kScalePtrAFC, Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex_nil,
        Broadcast.rightIndex_nil, Nat.zero_add, Nat.add_zero]
  · -- V_ptrs = vPtrsAFC s1 V 0
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    · simp only [vPtrsAFC, Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex_scalarL,
        Broadcast.rightIndex_scalarL, Region.cast]
    · simp only [vPtrsAFC, Tile.ptrAdd_data, Tile.scalar, Tile.bop_data, Tile.expandDim_data,
        Tile.vec_data, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
        Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, NumericDType.nat_add,
        NumericDType.nat_mul, qvkOffAFC, baseOffsetAFC]
      ring_nf
  · -- O_block_ptr = oBlockPtrAFC s1 Out
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    · simp only [oBlockPtrAFC, Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex_scalarL,
        Broadcast.rightIndex_scalarL, Region.cast]
    · simp only [oBlockPtrAFC, Tile.ptrAdd_data, Tile.scalar, Tile.bop_data, Tile.expandDim_data,
        Tile.vec_data, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
        Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, NumericDType.nat_add,
        NumericDType.nat_mul, qvkOffAFC, baseOffsetAFC]
      ring_nf
  · -- q = masked load = invariant q form
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
    refine congrArg some ?_
    ext idx
    simp only [Tile.bop_data, Tile.cop_data, Tile.expandDim_data, Tile.vec_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
      Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
      TileShape.dropInsertedIndex]
    rw [show (ComparableDType.nat.lt (s1.pids 0 * 128 + idx.1.val) 128
          && ComparableDType.nat.lt idx.2.1.val 96)
        = decide (s1.pids 0 * 128 + idx.1.val < 128 ∧ idx.2.1.val < 96) from by
      rw [Bool.eq_iff_iff]; simp only [Bool.and_eq_true, ComparableDType.nat_lt_eq_true,
        decide_eq_true_eq]]
    by_cases hk : s1.pids 0 * 128 + idx.1.val < 128 ∧ idx.2.1.val < 96
    · rw [if_pos (by simp only [decide_eq_true_eq]; exact hk), if_pos hk]
      refine congrArg some ?_
      simp only [qTileAFC, BlockState.readMem, BlockState.setReg_mem, castTile_self,
        Tile.ptrAdd_data, Tile.scalar, Tile.bop_data,
        Tile.expandDim_data, Tile.vec_data, Broadcast.leftIndex_scalarL,
        Broadcast.rightIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul, NumericDType.nat_add, NumericDType.nat_mul,
        Region.cast, qvkOffAFC, baseOffsetAFC]
      ring_nf
    · rw [if_neg (by simp only [decide_eq_true_eq]; exact hk), if_neg hk]
      simp only [BlockState.setReg_undef, hundef]
      norm_num

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **PreLoop execution.** The 22 deterministic preLoop statements step a clean
state `s` (with `s.undef ≡ 0`) to the loop-entry state `s0`, exposing every
register readback the loop body / invariant base case needs: the running
`m_i`/`l_i`/`acc` registers carry the kernel seeds (`full ⊥`, `full 1.0`,
`full 0`), the index vectors (`offs_m`/`offs_n`/`offs_k`), the four streamed
pointer tiles (`K_ptrs`/`K_scale_ptr`/`V_ptrs`/`Q_ptrs`), the loaded masked `q`
tile and scalar `q_scale`, the program ids and `start_m`/`off_hz` scalars, and
preserves `pids`/`mem`/`undef`.

Proved by composing `afcPreLoopHead_eval` (statements 0–10) and
`afcPreLoopTail_eval` (statements 11–21) through `stepStmts.append_some`, keyed on
`afcPreLoop_eq_head_tail` — avoiding the heartbeat timeout the monolithic
22-statement chain hit at the final `rfl` over the deep `setReg` literal.

Modeling note on `l_i`: the kernel seeds `l_i = tl.zeros + 1.0 = 1.0`, faithfully
exposed here as `full 1.0`. The ⊥-seed foundation (`afcStateBot`, anchored at
`l = 0`) absorbs this `+1.0` on the first block — `α = realExp2(⊥ − m₀) = 0`, so
`l_i ← 1.0·0 + l_ij = l_ij`, matching `afcStateBot` from block 1 on — so the
running-state invariant `afcInvariant` (which binds `l_i` to `afcStateBot`) is
established only after the first loop iteration, not at the raw seed; this lemma
exposes the genuine seed register, deferring the invariant rebind to the step lemma. -/
theorem afcPreLoop_eval
    (s : BlockState) (Q K V QScale KScale Out : RegionName)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (AfcFoundation.afcPreLoop Q K V QScale KScale Out) s = some s0
      ∧ s0.pids = s.pids ∧ s0.mem = s.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 1))
      ∧ s0.regs .real [128] "m_i" = some ⟨fun _ : TileIndex [128] => (⊥ : WithBot ℝ)⟩
      ∧ s0.regs .real [128] "l_i" = some ⟨fun _ : TileIndex [128] => (some (1 : ℝ) : WithBot ℝ)⟩
      ∧ s0.regs .real [128, 128] "acc" = some ⟨fun _ : TileIndex [128, 128] => (some (0 : ℝ) : WithBot ℝ)⟩
      ∧ s0.regs .nat [128] "offs_m" = some (Tile.vec (fun r : Fin 128 => s.pids 0 * 128 + r.val))
      ∧ s0.regs .nat [64] "offs_n" = some (Tile.vec (fun j : Fin 64 => j.val))
      ∧ s0.regs .nat [128] "offs_k" = some (Tile.vec (fun e : Fin 128 => e.val))
      ∧ s0.regs .real [] "q_scale" = some ⟨fun _ : TileIndex [] =>
          some (s.readMem QScale ((s.pids 1 * ((128 + 128 - 1) / 128) + s.pids 0)))⟩
      ∧ s0.regs .ptr [128, 64] "K_ptrs" = some (kPtrsAFC s K 0)
      ∧ s0.regs .ptr [] "K_scale_ptr" = some (kScalePtrAFC s KScale 0)
      ∧ s0.regs .ptr [64, 128] "V_ptrs" = some (vPtrsAFC s V 0)
      ∧ s0.regs .ptr [128, 128] "O_block_ptr" = some (oBlockPtrAFC s Out)
      ∧ s0.regs .real [128, 128] "q" = some ⟨fun idx : TileIndex [128, 128] =>
          if s.pids 0 * 128 + idx.1.val < 128 ∧ idx.2.1.val < 96
          then some (qTileAFC s Q idx) else some (0.0 : ℝ)⟩ := by
  rw [AfcFoundation.afcPreLoop_eq_head_tail]
  obtain ⟨s1, hHead, hpids1, hmem1, hundef1, hstartm1, hoffhz1, hqvk1, hqso1, hkso1,
    hoffsm1, hoffsn1, hoffsk1⟩ := afcPreLoopHead_eval s hundef
  rw [stepStmts.append_some hHead]
  have hstartm1' : s1.regs .nat [] "start_m" = some (Tile.scalar (s1.pids 0)) := by
    rw [hpids1]; exact hstartm1
  have hoffhz1' : s1.regs .nat [] "off_hz" = some (Tile.scalar (s1.pids 1)) := by
    rw [hpids1]; exact hoffhz1
  have hqvk1' : s1.regs .nat [] "qvk_offset" = some (Tile.scalar (qvkOffAFC s1)) := by
    rw [show qvkOffAFC s1 = qvkOffAFC s from by simp only [qvkOffAFC, hpids1]]; exact hqvk1
  have hqso1' : s1.regs .nat [] "q_scale_offset"
      = some (Tile.scalar (s1.pids 1 * ((128 + 128 - 1) / 128))) := by
    rw [hpids1]; exact hqso1
  have hkso1' : s1.regs .nat [] "k_scale_offset"
      = some (Tile.scalar (s1.pids 1 * ((128 + 64 - 1) / 64))) := by
    rw [hpids1]; exact hkso1
  have hoffsm1' : s1.regs .nat [128] "offs_m"
      = some (Tile.vec (fun r : Fin 128 => s1.pids 0 * 128 + r.val)) := by
    rw [hpids1]; exact hoffsm1
  obtain ⟨s0, hTail, hpids0, hmem0, hundef0, hstartm0, hoffhz0, hmi0, hli0, hacc0,
    hoffsm0, hoffsn0, hoffsk0, hqscale0, hkp0, hksp0, hvp0, hop0, hq0⟩ :=
    afcPreLoopTail_eval s1 Q K V QScale KScale Out hundef1
      hstartm1' hoffhz1' hqvk1' hqso1' hkso1' hoffsm1' hoffsn1 hoffsk1
  -- base/pid-derived closed forms agree between s1 and s (pids/mem coincide)
  have hbase : baseOffsetAFC s1 = baseOffsetAFC s := by simp only [baseOffsetAFC, hpids1]
  have hkpEq : kPtrsAFC s1 K 0 = kPtrsAFC s K 0 := by
    simp only [kPtrsAFC, hbase]
  have hkspEq : kScalePtrAFC s1 KScale 0 = kScalePtrAFC s KScale 0 := by
    simp only [kScalePtrAFC, hpids1]
  have hvpEq : vPtrsAFC s1 V 0 = vPtrsAFC s V 0 := by
    simp only [vPtrsAFC, hbase]
  have hopEq : oBlockPtrAFC s1 Out = oBlockPtrAFC s Out := by
    simp only [oBlockPtrAFC, hbase, hpids1]
  refine ⟨s0, hTail, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpids0, hpids1]
  · rw [hmem0, hmem1]
  · exact hundef0
  · rw [hstartm0, hpids1]
  · rw [hoffhz0, hpids1]
  · exact hmi0
  · exact hli0
  · exact hacc0
  · rw [hoffsm0, hpids1]
  · exact hoffsn0
  · exact hoffsk0
  · rw [hqscale0]
    refine congrArg some ?_
    ext idx
    simp only [BlockState.readMem, hmem1, hpids1]
  · rw [hkp0, hkpEq]
  · rw [hksp0, hkspEq]
  · rw [hvp0, hvpEq]
  · rw [hop0, hopEq]
  · rw [hq0]
    refine congrArg some ?_
    ext idx
    simp only [hpids1, qTileAFC, BlockState.readMem, hmem1, hbase]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **PreLoop ⇒ invariant base case.** The 22 deterministic preLoop statements step
a clean state `s` (with `s.undef ≡ 0`) to a loop-entry state `s0` satisfying the
strengthened `afcInvariant … 0`: the running `m_i`/`l_i`/`acc` carry the ⊥-seed
inits (`afcRunningMax 0 = ⊥`, `afcStateBot1 0 = (⊥,1,0)`), the index vectors and
program-id scalars are bound, the masked `q` / scalar `q_scale` are loaded, and the
three streamed pointers carry their `c = 0` cell-forms (`kPtrsAFC`/`kScalePtrAFC`/
`vPtrsAFC`). This is the base case `forRange_inv` consumes. -/
theorem afcPreLoop_invariant
    (s : BlockState) (Q K V QScale KScale Out : RegionName)
    (keyScale : Fin 128 → ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (AfcFoundation.afcPreLoop Q K V QScale KScale Out) s = some s0
      ∧ afcInvariant Q K V QScale KScale Out s keyScale 0 s0 := by
  obtain ⟨s0, hstep, hpids, hmem, hundef0, hstartm, hoffhz, hmi, hli, hacc,
    hoffsm, hoffsn, hoffsk, hqscale, hkp, hksp, hvp, hop, hq⟩ :=
    afcPreLoop_eval s Q K V QScale KScale Out hundef
  obtain ⟨hzm, hzl, hza⟩ :=
    VeriTile.Bench.TritonBenchG.AttnFwdCausal.AfcInvariantBase.afcInvariant_running_zero
      Q K V s keyScale
  refine ⟨s0, hstep, ?_⟩
  simp only [afcInvariant, qStartAFC]
  refine ⟨hpids, by norm_num, by norm_num, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, fun rg o => hundef0 rg o, hmem⟩
  · -- m_i = afcRunningMax ... 0 = ⊥
    rw [hmi]; exact congrArg some hzm.symm
  · -- l_i = afcStateBot1 ... 0 .2.1 = 1
    rw [hli]; exact congrArg some hzl.symm
  · -- acc = afcStateBot1 ... 0 .2.2 = 0
    rw [hacc]; exact congrArg some hza.symm
  · -- offs_m = vec (qStart + r)
    rw [hoffsm]
  · -- offs_n
    exact hoffsn
  · -- q
    rw [hq]; rfl
  · -- q_scale
    rw [hqscale]; rfl
  · -- K_ptrs (i/64 = 0)
    rw [hkp]
  · -- K_scale_ptr
    rw [hksp]
  · -- V_ptrs
    rw [hvp]
  · -- O_block_ptr
    rw [hop]

/-! ## FOUNDATION Part 5 — `afcLoopBody_steps` (loop-body execution chain)

The 22 lowered `afcLoopBody` statements (statements 0–21) step the iteration-entry
state `sin` (with `start_n = SN` and the invariant's register readbacks
`offs_m`/`offs_n`/`m_i`/`l_i`/`acc`/`q`/`q_scale`/`k_scale`/`K_ptrs`/`K_scale_ptr`/
`V_ptrs`) to a final state `sF`, exposing the symbolic `m_i`/`l_i`/`acc` register
values (the kernel's per-block tile arithmetic over the masked `qk`) plus the
advanced pointers. Threaded through `stepStmts.cons_some` via the banked `afc_*`
op-eval recipes. Split into a head (0–10) and a tail (11–21) to dodge the
heartbeat ceiling, mirroring the preLoop split. -/

namespace AfcFoundation

open VeriTile.Triton

/-- **Loop-body head** — statements 0–10 of `afcLoopBody` (`start_n` identity,
`k_mask`, the `k`/`k_scale` loads, the `qk` dot+scale, the causal `mask`, the
`-1e6` sentinel `where`, `m_ij`, the max-shift, `p = exp2`, `p` zero-mask). -/
def afcLoopBodyHead : List Stmt :=
  (List.take 11 AfcFoundation.afcLoopBody)

/-- **Loop-body tail** — statements 11–21 of `afcLoopBody` (`l_ij`, `alpha`,
`l_i`, `acc` rescale, the `v` load, the `p` fp16 cast, the `acc` accumulate,
the `m_i` carry, and the three pointer advances). -/
def afcLoopBodyTail : List Stmt :=
  (List.drop 11 AfcFoundation.afcLoopBody)

theorem afcLoopBody_eq_head_tail :
    AfcFoundation.afcLoopBody = afcLoopBodyHead ++ afcLoopBodyTail := by
  rw [afcLoopBodyHead, afcLoopBodyTail, List.take_append_drop]

end AfcFoundation

namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal

open VeriTile.Triton

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Loop-body head execution.** The 11 head statements (`afcLoopBodyHead`,
statements 0–10) step the iteration-entry state `sin` (with `start_n`/`offs_m`/
`offs_n`/`m_i`/`K_ptrs`/`K_scale_ptr`/`q`/`q_scale` set) to a state `s1`, exposing
the causal `mask` tile, the running max `m_ij`, and the zero-masked `p` tile, while
preserving the carried `l_i`/`acc`/`V_ptrs` and the index/pointer registers the tail
consumes. The masked `qk` and its post-shift `exp2` are threaded through the banked
`afc_*` recipes. -/
theorem afcLoopBodyHead_steps
    (sin : BlockState) (SN : Nat)
    (offsm : Tile .nat [128]) (offsn : Tile .nat [64])
    (kptrs : Tile .ptr [128, 64]) (ksptr : Tile .ptr [])
    (mtile : Tile .real [128]) (qtile : Tile .real [128, 128]) (qsc : Tile .real [])
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoffsm : sin.regs .nat [128] "offs_m" = some offsm)
    (hoffsn : sin.regs .nat [64] "offs_n" = some offsn)
    (hmi : sin.regs .real [128] "m_i" = some mtile)
    (hkp : sin.regs .ptr [128, 64] "K_ptrs" = some kptrs)
    (hksp : sin.regs .ptr [] "K_scale_ptr" = some ksptr)
    (hq : sin.regs .real [128, 128] "q" = some qtile)
    (hqsc : sin.regs .real [] "q_scale" = some qsc) :
    ∃ s1, stepStmts AfcFoundation.afcLoopBodyHead sin = some s1
      ∧ s1.pids = sin.pids ∧ s1.mem = sin.mem ∧ (∀ rg o, s1.undef rg o = sin.undef rg o)
      ∧ ∃ (kmaskT : Tile .bool [128, 64]) (ktile : Tile .real [128, 64])
          (kscT : Tile .real []) (qkdotT : Tile .real [128, 64])
          (maskT : Tile .bool [128, 64]) (qkSentT : Tile .real [128, 64])
          (rmaxT mijT : Tile .real [128]) (qkShiftT pExpT pT : Tile .real [128, 64]),
        -- recipe-produced symbolic tiles
        (kmaskT = ⟨fun idx : TileIndex [128, 64] =>
            (ComparableDType.nat.lt (offsn.data (idx.2.1, PUnit.unit)) (128 - SN))
              && (ComparableDType.nat.lt idx.1.val 96)⟩)
        ∧ (ktile = ⟨fun i : TileIndex [128, 64] =>
            if kmaskT.data i then some (sin.readMem (kptrs.data i).1 (kptrs.data i).2)
            else some (sin.undef (kptrs.data i).1 (kptrs.data i).2)⟩)
        ∧ (kscT = ⟨fun _ : TileIndex [] =>
            some (sin.readMem (ksptr.data PUnit.unit).1 (ksptr.data PUnit.unit).2)⟩)
        ∧ (qkdotT = Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.bop NumericDType.real.mul Broadcast.scalarR
              ⟨fun i => (Tile.dot [] qtile ktile).data i⟩ qsc) kscT)
        ∧ (maskT = ⟨fun idx : TileIndex [128, 64] =>
            ComparableDType.nat.ge (offsm.data (idx.1, PUnit.unit))
              (SN + offsn.data (idx.2.1, PUnit.unit))⟩)
        ∧ (qkSentT = ⟨fun idx : TileIndex [128, 64] =>
            if maskT.data idx then qkdotT.data idx
            else WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ))⟩)
        ∧ (Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [128, 64].length) qkSentT = some rmaxT)
        ∧ (mijT = Tile.select
            (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT)
            mtile rmaxT)
        ∧ (qkShiftT = Tile.bop NumericDType.real.sub
            (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))
        ∧ (pExpT = Tile.uop WithBot.realExp2 qkShiftT)
        ∧ (pT = ⟨fun idx : TileIndex [128, 64] =>
            if maskT.data idx then pExpT.data idx else (some (0.0 : ℝ) : WithBot ℝ)⟩)
        -- register readbacks at s1
        ∧ s1.regs .real [128] "m_i" = some mtile
        ∧ s1.regs .real [128] "m_ij" = some mijT
        ∧ s1.regs .bool [128, 64] "mask" = some maskT
        ∧ s1.regs .real [128, 64] "p" = some pT
        ∧ s1.regs .real [128] "l_i" = sin.regs .real [128] "l_i"
        ∧ s1.regs .real [128, 128] "acc" = sin.regs .real [128, 128] "acc"
        ∧ s1.regs .real [64, 128] "v" = sin.regs .real [64, 128] "v"
        ∧ s1.regs .ptr [64, 128] "V_ptrs" = sin.regs .ptr [64, 128] "V_ptrs"
        ∧ s1.regs .ptr [128, 64] "K_ptrs" = some kptrs
        ∧ s1.regs .ptr [] "K_scale_ptr" = some ksptr
        ∧ s1.regs .nat [128] "offs_m" = some offsm
        ∧ s1.regs .nat [64] "offs_n" = some offsn
        ∧ s1.regs .nat [] "start_n" = some (Tile.scalar SN)
        ∧ s1.regs .real [128, 128] "q" = some qtile
        ∧ s1.regs .real [] "q_scale" = some qsc := by
  -- the symbolic tiles
  set kmaskT : Tile .bool [128, 64] := ⟨fun idx : TileIndex [128, 64] =>
      (ComparableDType.nat.lt (offsn.data (idx.2.1, PUnit.unit)) (128 - SN))
        && (ComparableDType.nat.lt idx.1.val 96)⟩ with hkmaskT
  set ktile : Tile .real [128, 64] := ⟨fun i : TileIndex [128, 64] =>
      if kmaskT.data i then some (sin.readMem (kptrs.data i).1 (kptrs.data i).2)
      else some (sin.undef (kptrs.data i).1 (kptrs.data i).2)⟩ with hktile
  set kscT : Tile .real [] := ⟨fun _ : TileIndex [] =>
      some (sin.readMem (ksptr.data PUnit.unit).1 (ksptr.data PUnit.unit).2)⟩ with hkscT
  set qkdotT : Tile .real [128, 64] := Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        ⟨fun i => (Tile.dot [] qtile ktile).data i⟩ qsc) kscT with hqkdotT
  set maskT : Tile .bool [128, 64] := ⟨fun idx : TileIndex [128, 64] =>
      ComparableDType.nat.ge (offsm.data (idx.1, PUnit.unit))
        (SN + offsn.data (idx.2.1, PUnit.unit))⟩ with hmaskT
  set qkSentT : Tile .real [128, 64] := ⟨fun idx : TileIndex [128, 64] =>
      if maskT.data idx then qkdotT.data idx
      else WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ))⟩ with hqkSentT
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [128, 64].length) qkSentT = some t :=
    ⟨_, afc_reduceMaxDrop1_some qkSentT⟩
  set mijT : Tile .real [128] := Tile.select
      (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT)
      mtile rmaxT with hmijT
  set qkShiftT : Tile .real [128, 64] := Tile.bop NumericDType.real.sub
      (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT) with hqkShiftT
  set pExpT : Tile .real [128, 64] := Tile.uop WithBot.realExp2 qkShiftT with hpExpT
  set pT : Tile .real [128, 64] := ⟨fun idx : TileIndex [128, 64] =>
      if maskT.data idx then pExpT.data idx else (some (0.0 : ℝ) : WithBot ℝ)⟩ with hpT
  unfold AfcFoundation.afcLoopBodyHead AfcFoundation.afcLoopBody
  simp only [List.take_succ_cons, List.take_zero]
  -- stmt 0: start_n = start_n (identity)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_n") sin = some (Tile.scalar SN) from by rw [evalOp_ref, hsn]))]
  -- stmt 1: k_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some kmaskT from by
      rw [afc_kmask_eval _ SN 128 96 offsn
        (by simp [BlockState.setReg_ne_name, hoffsn])
        (by rw [BlockState.setReg_same])]))]
  -- stmt 2: k = load K_ptrs (masked)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some ktile from by
      rw [afc_load_k_eval _ 128 64 "K_ptrs" "k_mask" kptrs kmaskT
        (by simp [BlockState.setReg_ne_name, hkp]) (by rw [BlockState.setReg_same])]
      rfl))]
  -- stmt 3: k_scale = load K_scale_ptr (scalar)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some kscT from by
      rw [afc_load_kscale_eval _ "K_scale_ptr" ksptr
        (by simp [BlockState.setReg_ne_name, hksp])]
      rfl))]
  -- stmt 4: qk = castFloat(q·k) * q_scale * k_scale
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some qkdotT from by
      rw [afc_qk_dot_eval _ 128 64 128 qtile ktile qsc kscT
        (by simp [BlockState.setReg_ne_name, hq]) (by simp [BlockState.setReg_ne_name])
        (by simp [BlockState.setReg_ne_name, hqsc]) (by rw [BlockState.setReg_same])]))]
  -- stmt 5: mask = offs_m[:,None] >= start_n + offs_n[None,:]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some maskT from by
      rw [afc_mask_eval _ SN offsm offsn
        (by simp [BlockState.setReg_ne_name, hoffsm]) (by simp [BlockState.setReg_ne_name, hoffsn])
        (by simp [BlockState.setReg_ne_name, hsn])]))]
  -- stmt 6: qk = where(mask, qk, -1e6)  (body uses `Op.const 0.0`, inlined here)
  have hbcast6 : ∀ t : BlockState, @evalOp TileDType.real [128, 64]
      (Op.broadcast (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1000000.0)) [128, 64]) t
      = some (⟨fun _ : TileIndex [128, 64] =>
          WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ))⟩ : Tile .real [128, 64]) := by
    intro t
    simp only [evalOp, evalOp_sub, evalOp_const, Option.bind_eq_bind, Option.bind_some]
    rfl
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.where (Op.ref .bool [128, 64] "mask")
        (Op.ref .real [128, 64] "qk")
        (Op.broadcast (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1000000.0)) [128, 64])) _
        = some qkSentT from by
      rw [evalOp_where]
      simp only [evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        hbcast6, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_
      ext idx
      simp only [hqkSentT, Tile.select_data, Tile.scalar]))]
  -- stmt 7: m_ij = maximum(m_i, max(qk,1))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some mijT from by
      rw [afc_mij_eval _ mtile qkSentT rmaxT
        (by simp [BlockState.setReg_ne_name, hmi]) (by rw [BlockState.setReg_same]) hrm]))]
  -- stmt 8: qk = qk - m_ij[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some qkShiftT from by
      rw [afc_qk_sub_eval _ (by simp) qkSentT mijT
        (by simp [BlockState.setReg_ne_name]) (by rw [BlockState.setReg_same])]))]
  -- stmt 9: p = exp2(qk)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some pExpT from by
      rw [afc_p_eval _ qkShiftT (by rw [BlockState.setReg_same])]))]
  -- stmt 10: p = where(mask, p, 0)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some pT from by
      rw [afc_p_mask_eval _ maskT pExpT
        (by simp [BlockState.setReg_ne_name]) (by rw [BlockState.setReg_same])]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, kmaskT, ktile, kscT, qkdotT, maskT, qkSentT, rmaxT, mijT,
    qkShiftT, pExpT, pT, rfl, rfl, rfl, rfl, rfl, rfl, hrm, rfl, rfl, rfl, rfl,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_pids]
  · funext rg o; simp [BlockState.setReg_mem]
  · intro rg o; simp [BlockState.setReg_undef]
  · simp [BlockState.setReg_ne_name, hmi]  -- m_i
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]  -- m_ij
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]  -- mask
  · simp [BlockState.setReg_same]  -- p
  · simp [BlockState.setReg_ne_name]  -- l_i
  · simp [BlockState.setReg_ne_name]  -- acc
  · simp [BlockState.setReg_ne_name]  -- v
  · simp [BlockState.setReg_ne_name]  -- V_ptrs
  · simp [BlockState.setReg_ne_name, hkp]  -- K_ptrs
  · simp [BlockState.setReg_ne_name, hksp]  -- K_scale_ptr
  · simp [BlockState.setReg_ne_name, hoffsm]  -- offs_m
  · simp [BlockState.setReg_ne_name, hoffsn]  -- offs_n
  · simp [BlockState.setReg_ne_name, hsn]  -- start_n
  · simp [BlockState.setReg_ne_name, hq]  -- q
  · simp [BlockState.setReg_ne_name, hqsc]  -- q_scale

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Loop-body tail execution.** The 11 tail statements (`afcLoopBodyTail`,
statements 11–21) step a head-exit state `s1` (with the head's `m_i`/`m_ij`/`mask`/
`p`/`l_i`/`acc`/`offs_n`/`start_n`/`V_ptrs`/`K_ptrs`/`K_scale_ptr` readbacks
supplied) to the final state `sF`, exposing the symbolic `m_i` (carry `m_ij`),
`l_i` (`= l_i·α + l_ij`), `acc` (`= acc·α + dot(p, v)`), and the advanced
`K_ptrs`/`K_scale_ptr`/`V_ptrs`. Threaded through the banked `afc_*` recipes; the
inline `v`-load mask is evaluated directly. -/
theorem afcLoopBodyTail_steps
    (s1 : BlockState) (SN : Nat)
    (offsn : Tile .nat [64])
    (kptrs : Tile .ptr [128, 64]) (ksptr : Tile .ptr []) (vptrs : Tile .ptr [64, 128])
    (mtile mijT : Tile .real [128]) (maskT : Tile .bool [128, 64])
    (pT : Tile .real [128, 64]) (litile : Tile .real [128]) (acctile : Tile .real [128, 128])
    (hsn : s1.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoffsn : s1.regs .nat [64] "offs_n" = some offsn)
    (hmi : s1.regs .real [128] "m_i" = some mtile)
    (hmij : s1.regs .real [128] "m_ij" = some mijT)
    (hmask : s1.regs .bool [128, 64] "mask" = some maskT)
    (hp : s1.regs .real [128, 64] "p" = some pT)
    (hli : s1.regs .real [128] "l_i" = some litile)
    (hacc : s1.regs .real [128, 128] "acc" = some acctile)
    (hvp : s1.regs .ptr [64, 128] "V_ptrs" = some vptrs)
    (hkp : s1.regs .ptr [128, 64] "K_ptrs" = some kptrs)
    (hksp : s1.regs .ptr [] "K_scale_ptr" = some ksptr) :
    ∃ sF, stepStmts AfcFoundation.afcLoopBodyTail s1 = some sF
      ∧ sF.pids = s1.pids ∧ sF.mem = s1.mem ∧ (∀ rg o, sF.undef rg o = s1.undef rg o)
      ∧ sF.regs .nat [128] "offs_m" = s1.regs .nat [128] "offs_m"
      ∧ sF.regs .nat [64] "offs_n" = s1.regs .nat [64] "offs_n"
      ∧ sF.regs .real [128, 128] "q" = s1.regs .real [128, 128] "q"
      ∧ sF.regs .real [] "q_scale" = s1.regs .real [] "q_scale"
      ∧ ∃ (lijT alphaT : Tile .real [128]) (vmaskT : Tile .bool [64, 128])
          (vtile : Tile .real [64, 128]) (pf16 : Tile .fp16 [128, 64]),
        (lijT = Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [128, 64].length) pT)
        ∧ (alphaT = Tile.uop WithBot.realExp2
            (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
        ∧ (vmaskT = ⟨fun idx : TileIndex [64, 128] =>
            (ComparableDType.nat.lt (offsn.data (idx.1, PUnit.unit)) (128 - SN))
              && (ComparableDType.nat.lt idx.2.1.val 96)⟩)
        ∧ (vtile = ⟨fun i : TileIndex [64, 128] =>
            if vmaskT.data i then some (s1.readMem (vptrs.data i).1 (vptrs.data i).2)
            else some (s1.undef (vptrs.data i).1 (vptrs.data i).2)⟩)
        ∧ (pf16 = ⟨fun i => FloatDType.real.cast FloatDType.fp16 (pT.data i)⟩)
        ∧ sF.regs .real [128] "m_i" = some mijT
        ∧ sF.regs .real [128] "l_i" = some (Tile.bop NumericDType.real.add
            (Broadcast.consSame Broadcast.nil)
            (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) litile alphaT) lijT)
        ∧ sF.regs .real [128, 128] "acc" = some (Tile.bop NumericDType.real.add
            (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
            (Tile.dot [] ⟨fun i => FloatDType.fp16.cast FloatDType.real (pf16.data i)⟩ vtile))
        ∧ sF.regs .ptr [128, 64] "K_ptrs" = some
            (Tile.ptrAdd Broadcast.scalarR kptrs
              (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar 64) (Tile.scalar 128)))
        ∧ sF.regs .ptr [] "K_scale_ptr" = some
            (Tile.ptrAdd Broadcast.nil ksptr (Tile.scalar 1))
        ∧ sF.regs .ptr [64, 128] "V_ptrs" = some
            (Tile.ptrAdd Broadcast.scalarR vptrs
              (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar 64) (Tile.scalar 128))) := by
  set lijT : Tile .real [128] := Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [128, 64].length) pT with hlijT
  set alphaT : Tile .real [128] := Tile.uop WithBot.realExp2
      (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT) with halphaT
  set vmaskT : Tile .bool [64, 128] := ⟨fun idx : TileIndex [64, 128] =>
      (ComparableDType.nat.lt (offsn.data (idx.1, PUnit.unit)) (128 - SN))
        && (ComparableDType.nat.lt idx.2.1.val 96)⟩ with hvmaskT
  set vtile : Tile .real [64, 128] := ⟨fun i : TileIndex [64, 128] =>
      if vmaskT.data i then some (s1.readMem (vptrs.data i).1 (vptrs.data i).2)
      else some (s1.undef (vptrs.data i).1 (vptrs.data i).2)⟩ with hvtile
  set pf16 : Tile .fp16 [128, 64] := ⟨fun i => FloatDType.real.cast FloatDType.fp16 (pT.data i)⟩ with hpf16
  unfold AfcFoundation.afcLoopBodyTail AfcFoundation.afcLoopBody
  simp only [List.drop_succ_cons, List.drop_zero]
  -- stmt 11: l_ij = sum(p, 1)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some lijT from afc_lij_eval _ pT hp))]
  -- stmt 12: alpha = exp2(m_i - m_ij)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some alphaT from by
      rw [afc_alpha_eval _ mtile mijT
        (by simp [BlockState.setReg_ne_name, hmi]) (by simp [BlockState.setReg_ne_name, hmij])]))]
  -- stmt 13: l_i = l_i * alpha + l_ij
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      afc_li_eval _ litile alphaT lijT
        (by simp [BlockState.setReg_ne_name, hli]) (by rw [BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  -- stmt 14: acc = acc * alpha[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      afc_acc_rescale_eval _ (by simp) acctile alphaT
        (by simp [BlockState.setReg_ne_name, hacc]) (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  -- stmt 15: v = load V_ptrs (inline boolAnd mask)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some vtile from by
      rw [afc_evalOp_load_ptr_mask_of (Op.ref .ptr [64, 128] "V_ptrs") _ _ vptrs vmaskT
        (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, hvp])
        (show evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [64] "offs_n"))
              (Op.sub .nat Broadcast.nil (Op.constNat 128) (Op.ref .nat [] "start_n")))
            (Op.expandDim ⟨0, by simp⟩
              (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat 96)))) _
            = some vmaskT from by
          rw [afc_evalOp_boolAnd, evalOp_lt]
          erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                (regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hoffsn)))),
            evalOp_expandDim]
          simp only [evalOp_lt, evalOp_sub, evalOp_arange, evalOp_constNat, evalOp_ref,
            BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same, hsn, Option.bind_some, Option.bind_eq_bind]
          refine congrArg some ?_
          ext idx
          simp only [hvmaskT, Tile.bop_data, Tile.cop_data, Tile.expandDim_data, Tile.vec,
            Tile.scalar, Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
            Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
            Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
            Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
            TileShape.dropInsertedIndex, NumericDType.sub])]
      rfl))]
  -- stmt 16: p = p.to(fp16)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some pf16 from
      afc_p_fp16_eval _ pT (by simp [BlockState.setReg_ne_name, hp])))]
  -- stmt 17: acc += dot(p.to(real), v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      afc_acc_eval _ 128 64 128
        (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
        pf16 vtile
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
        (by rw [BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  -- stmt 18: m_i = m_ij
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some mijT from
      afc_mi_carry_eval _ mijT (by simp [BlockState.setReg_ne_name, hmij])))]
  -- stmt 19: K_ptrs += 64 * 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      afc_advance_ptr_eval _ 128 64 64 "K_ptrs" kptrs
        (by simp [BlockState.setReg_ne_name, hkp])))]
  -- stmt 20: K_scale_ptr += 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      afc_advance_kscale_eval _ "K_scale_ptr" ksptr
        (by simp [BlockState.setReg_ne_name, hksp])))]
  -- stmt 21: V_ptrs += 64 * 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      afc_advance_ptr_eval _ 64 128 64 "V_ptrs" vptrs
        (by simp [BlockState.setReg_ne_name, hvp])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, lijT, alphaT, vmaskT, vtile, pf16,
    rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_pids]
  · funext rg o; simp [BlockState.setReg_mem]
  · intro rg o; simp [BlockState.setReg_undef]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]  -- offs_m
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]  -- offs_n
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]  -- q
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]  -- q_scale
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]  -- m_i
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]  -- l_i
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]  -- acc
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]  -- K_ptrs
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]  -- K_scale_ptr
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]  -- V_ptrs

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Loop-body execution chain.** The 22 lowered `afcLoopBody` statements step the
iteration-entry state `sin` (with `start_n = SN` and the invariant's register
readbacks `offs_m`/`offs_n`/`m_i`/`l_i`/`acc`/`q`/`q_scale`/`K_ptrs`/`K_scale_ptr`/
`V_ptrs`) to a final state `sF`, exposing the symbolic `m_i`/`l_i`/`acc` register
values — the kernel's per-block online-softmax tile arithmetic over the causal
masked `qk` (max `m_ij`, shifted `exp2`, zero-masked `p`, rescaled `l_i`/`acc`) —
plus the advanced `K_ptrs`/`K_scale_ptr`/`V_ptrs`. Composes `afcLoopBodyHead_steps`
(0–10) and `afcLoopBodyTail_steps` (11–21) through `stepStmts.append_some`. -/
theorem afcLoopBody_steps
    (sin : BlockState) (SN : Nat)
    (offsm : Tile .nat [128]) (offsn : Tile .nat [64])
    (kptrs : Tile .ptr [128, 64]) (ksptr : Tile .ptr []) (vptrs : Tile .ptr [64, 128])
    (mtile : Tile .real [128]) (qtile : Tile .real [128, 128]) (qsc : Tile .real [])
    (litile : Tile .real [128]) (acctile : Tile .real [128, 128])
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoffsm : sin.regs .nat [128] "offs_m" = some offsm)
    (hoffsn : sin.regs .nat [64] "offs_n" = some offsn)
    (hmi : sin.regs .real [128] "m_i" = some mtile)
    (hli : sin.regs .real [128] "l_i" = some litile)
    (hacc : sin.regs .real [128, 128] "acc" = some acctile)
    (hq : sin.regs .real [128, 128] "q" = some qtile)
    (hqsc : sin.regs .real [] "q_scale" = some qsc)
    (hkp : sin.regs .ptr [128, 64] "K_ptrs" = some kptrs)
    (hksp : sin.regs .ptr [] "K_scale_ptr" = some ksptr)
    (hvp : sin.regs .ptr [64, 128] "V_ptrs" = some vptrs) :
    ∃ sF, stepStmts AfcFoundation.afcLoopBody sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = sin.undef rg o)
      ∧ ∃ (kmaskT : Tile .bool [128, 64]) (ktile : Tile .real [128, 64])
          (kscT : Tile .real []) (qkdotT : Tile .real [128, 64])
          (maskT : Tile .bool [128, 64]) (qkSentT : Tile .real [128, 64])
          (rmaxT mijT : Tile .real [128]) (pT : Tile .real [128, 64])
          (lijT alphaT : Tile .real [128]) (vmaskT : Tile .bool [64, 128])
          (vtile : Tile .real [64, 128]) (pf16 : Tile .fp16 [128, 64]),
        -- the masked score / running-max / softmax-weight tiles
        (kmaskT = ⟨fun idx : TileIndex [128, 64] =>
            (ComparableDType.nat.lt (offsn.data (idx.2.1, PUnit.unit)) (128 - SN))
              && (ComparableDType.nat.lt idx.1.val 96)⟩)
        ∧ (ktile = ⟨fun i : TileIndex [128, 64] =>
            if kmaskT.data i then some (sin.readMem (kptrs.data i).1 (kptrs.data i).2)
            else some (sin.undef (kptrs.data i).1 (kptrs.data i).2)⟩)
        ∧ (kscT = ⟨fun _ : TileIndex [] =>
            some (sin.readMem (ksptr.data PUnit.unit).1 (ksptr.data PUnit.unit).2)⟩)
        ∧ (qkdotT = Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.bop NumericDType.real.mul Broadcast.scalarR
              ⟨fun i => (Tile.dot [] qtile ktile).data i⟩ qsc) kscT)
        ∧ (maskT = ⟨fun idx : TileIndex [128, 64] =>
            ComparableDType.nat.ge (offsm.data (idx.1, PUnit.unit))
              (SN + offsn.data (idx.2.1, PUnit.unit))⟩)
        ∧ (qkSentT = ⟨fun idx : TileIndex [128, 64] =>
            if maskT.data idx then qkdotT.data idx
            else WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ))⟩)
        ∧ (Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [128, 64].length) qkSentT = some rmaxT)
        ∧ (mijT = Tile.select
            (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT)
            mtile rmaxT)
        ∧ (pT = ⟨fun idx : TileIndex [128, 64] =>
            if maskT.data idx then
              (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
                (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data idx
            else (some (0.0 : ℝ) : WithBot ℝ)⟩)
        ∧ (lijT = Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [128, 64].length) pT)
        ∧ (alphaT = Tile.uop WithBot.realExp2
            (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
        ∧ (vmaskT = ⟨fun idx : TileIndex [64, 128] =>
            (ComparableDType.nat.lt (offsn.data (idx.1, PUnit.unit)) (128 - SN))
              && (ComparableDType.nat.lt idx.2.1.val 96)⟩)
        ∧ (vtile = ⟨fun i : TileIndex [64, 128] =>
            if vmaskT.data i then some (sin.readMem (vptrs.data i).1 (vptrs.data i).2)
            else some (sin.undef (vptrs.data i).1 (vptrs.data i).2)⟩)
        ∧ (pf16 = ⟨fun i => FloatDType.real.cast FloatDType.fp16 (pT.data i)⟩)
        -- final register readbacks
        ∧ sF.regs .real [128] "m_i" = some mijT
        ∧ sF.regs .real [128] "l_i" = some (Tile.bop NumericDType.real.add
            (Broadcast.consSame Broadcast.nil)
            (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) litile alphaT) lijT)
        ∧ sF.regs .real [128, 128] "acc" = some (Tile.bop NumericDType.real.add
            (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
            (Tile.dot [] ⟨fun i => FloatDType.fp16.cast FloatDType.real (pf16.data i)⟩ vtile))
        ∧ sF.regs .nat [128] "offs_m" = some offsm
        ∧ sF.regs .nat [64] "offs_n" = some offsn
        ∧ sF.regs .real [128, 128] "q" = some qtile
        ∧ sF.regs .real [] "q_scale" = some qsc
        ∧ sF.regs .ptr [128, 64] "K_ptrs" = some
            (Tile.ptrAdd Broadcast.scalarR kptrs
              (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar 64) (Tile.scalar 128)))
        ∧ sF.regs .ptr [] "K_scale_ptr" = some
            (Tile.ptrAdd Broadcast.nil ksptr (Tile.scalar 1))
        ∧ sF.regs .ptr [64, 128] "V_ptrs" = some
            (Tile.ptrAdd Broadcast.scalarR vptrs
              (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar 64) (Tile.scalar 128))) := by
  rw [AfcFoundation.afcLoopBody_eq_head_tail]
  -- head
  obtain ⟨s1, hHead, hpids1, hmem1, hundef1, kmaskT, ktile, kscT, qkdotT, maskT, qkSentT,
    rmaxT, mijT, qkShiftT, pExpT, pT, hkmaskT, hktile, hkscT, hqkdotT, hmaskT, hqkSentT,
    hrm, hmijT, hqkShiftT, hpExpT, hpT, hs1mi, hs1mij, hs1mask, hs1p, hs1li, hs1acc,
    hs1v, hs1vp, hs1kp, hs1ksp, hs1offsm, hs1offsn, hs1sn, hs1q, hs1qsc⟩ :=
    afcLoopBodyHead_steps sin SN offsm offsn kptrs ksptr mtile qtile qsc
      hsn hoffsm hoffsn hmi hkp hksp hq hqsc
  rw [stepStmts.append_some hHead]
  -- tail (consuming s1's readbacks)
  have hs1li' : s1.regs .real [128] "l_i" = some litile := by rw [hs1li]; exact hli
  have hs1acc' : s1.regs .real [128, 128] "acc" = some acctile := by rw [hs1acc]; exact hacc
  have hs1vp' : s1.regs .ptr [64, 128] "V_ptrs" = some vptrs := by rw [hs1vp]; exact hvp
  obtain ⟨sF, hTail, hpidsF, hmemF, hundefF, hFoffsm, hFoffsn, hFq, hFqsc,
    lijT, alphaT, vmaskT, vtile, pf16,
    hlijT, halphaT, hvmaskT, hvtile, hpf16, hFmi, hFli, hFacc, hFkp, hFksp, hFvp⟩ :=
    afcLoopBodyTail_steps s1 SN offsn kptrs ksptr vptrs mtile mijT maskT pT litile acctile
      hs1sn hs1offsn hs1mi hs1mij hs1mask hs1p hs1li' hs1acc' hs1vp' hs1kp hs1ksp
  -- the head's `p` register is the zero-masked exp2 tile; rewrite the symbolic forms
  have hpT' : pT = ⟨fun idx : TileIndex [128, 64] =>
      if maskT.data idx then
        (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
          (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data idx
      else (some (0.0 : ℝ) : WithBot ℝ)⟩ := by
    rw [hpT, hpExpT, hqkShiftT]
  -- the head's `kscT` reads sin.mem; tail's vtile reads s1.mem = sin.mem; align readbacks
  have hsinmem : s1.mem = sin.mem := hmem1
  have hsinundef : ∀ rg o, s1.undef rg o = sin.undef rg o := hundef1
  refine ⟨sF, hTail, ?_, ?_, ?_, kmaskT, ktile, kscT, qkdotT, maskT, qkSentT, rmaxT, mijT,
    pT, lijT, alphaT, vmaskT, vtile, pf16,
    hkmaskT, hktile, hkscT, hqkdotT, hmaskT, hqkSentT, hrm, hmijT, hpT', hlijT, halphaT,
    ?_, ?_, hpf16, hFmi, hFli, hFacc, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF, hpids1]
  · rw [hmemF, hmem1]
  · intro rg o; rw [hundefF, hundef1]
  · -- vmaskT (head offsn already matches sin's offsn)
    rw [hvmaskT]
  · -- vtile reads s1.mem/undef = sin.mem/undef
    rw [hvtile]
    refine congrArg _ ?_
    ext i
    rw [show s1.readMem (vptrs.data i).1 (vptrs.data i).2
          = sin.readMem (vptrs.data i).1 (vptrs.data i).2 from by
      unfold BlockState.readMem; rw [hsinmem],
      hsinundef]
  · rw [hFoffsm, hs1offsm]  -- offs_m
  · rw [hFoffsn, hs1offsn]  -- offs_n
  · rw [hFq, hs1q]  -- q
  · rw [hFqsc, hs1qsc]  -- q_scale
  · exact hFkp
  · exact hFksp
  · exact hFvp

/-! ## FOUNDATION Part 5 — `afcPostLoop` AST + check

The 2 lowered postLoop statements (`= body.drop 23`): `acc = acc / l_i[:, None]`
(per-row denominator normalization) and the masked `tl.store` of `acc` to
`O_block_ptr` (mask `(offs_m < 128) & (offs_k < 96)`). The store value is a plain
`Op.ref` (no cast: `Out` is real). Checked by `rfl`. -/

/-- The 2 lowered postLoop statements of the Python-shape AFC kernel
(`= body.drop 23`). -/
def afcPostLoop (Out : RegionName) : List Stmt :=
  [ -- 23: acc = acc / l_i[:, None]
    Stmt.assign .real [128, 128] "acc"
      (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [128, 128] "acc") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "l_i"))),
    -- 24: tl.store(O_block_ptr, acc, mask=(offs_m < 128) & (offs_k < 96))
    Stmt.store .real [128, 128] (.ptr (.ref .ptr [128, 128] "O_block_ptr"))
      (Op.ref .real [128, 128] "acc")
      (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) (Op.constNat 128))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat 96))))) ]

set_option maxRecDepth 8000 in
/-- The 2 lowered postLoop statements (`body.drop 23`) of the Python-shape AFC
kernel are exactly `afcPostLoop`. Checked by `rfl`. -/
theorem afcPostLoop_check (Q K V QScale KScale Out : RegionName) :
    (attn_fwd_causal_surface Q K V QScale KScale Out
        65536 16384 128 1 65536 16384 128 1 65536 16384 128 1 65536 16384 128 1
        2 4 128 128 128 64 128 96 1).toAlgKernel.body.drop 23
      = afcPostLoop Out :=
  rfl

/-! ## Masked-block bridge layer (ported aft3 → afc; [128,64], -1e6 sentinel, HEAD_ACTIVE) -/

/-- Canonical axis-1 index of `[128, 64]`. -/
abbrev afcAx1 : Fin [128, 64].length := ⟨1, by simp⟩

/-- `filterMap`-then-map-and-sum over `finRange n` equals the masked `Finset.sum`. -/
theorem afc_filterMap_finRange_sum {α : Type*} (n : Nat)
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

/-- Any member of a `WithBot ℝ` list is `≤` its `foldr (⊔) ⊥`. -/
theorem afc_mem_le_foldr_sup (a : WithBot ℝ) :
    ∀ (L : List (WithBot ℝ)), a ∈ L → a ≤ L.foldr (· ⊔ ·) ⊥ := by
  intro L
  induction L with
  | nil => intro h; simp at h
  | cons x t ih =>
    intro h
    rcases List.mem_cons.mp h with rfl | h
    · exact le_sup_left
    · exact le_trans (ih h) le_sup_right

/-- `filterMap`-then-coe `foldr ⊔ ⊥` over `finRange n` equals the masked `Finset.sup`. -/
theorem afc_filterMap_foldr_sup (n : Nat) (P : Fin n → Prop) [DecidablePred P]
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
/-- The map-and-sum of `afcBlock` equals a `Fin 64`-masked `Finset.sum`, reindexing
block `c`'s window onto lanes `jL` (global key `c·64 + jL`). -/
theorem afcBlock_map_sum
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart c : Nat) (i : Fin 128) (d : Fin 128)
    (hwin : (c + 1) * 64 ≤ 128) (h : ℝ × ℝ → ℝ) :
    ((afcBlock qT kT vT keyScale qStart c i d).map h).sum
      = ∑ jL : Fin 64,
          (if (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ : Fin 128).val ≤ qStart + i.val then
            h (afcKV qT kT vT keyScale i d ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩)
           else 0) := by
  rw [afcBlock, afc_filterMap_finRange_sum 128
    (fun j => c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val)
    (fun j => afcKV qT kT vT keyScale i d j) h]
  rw [show (∑ j : Fin 128, if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val
            then h (afcKV qT kT vT keyScale i d j) else 0)
        = ∑ j ∈ Finset.univ.filter (fun j : Fin 128 => c * 64 ≤ j.val ∧ j.val < (c + 1) * 64),
            (if j.val ≤ qStart + i.val then h (afcKV qT kT vT keyScale i d j) else 0) from by
    rw [Finset.sum_filter]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    by_cases hwj : c * 64 ≤ j.val ∧ j.val < (c + 1) * 64
    · by_cases hcj : j.val ≤ qStart + i.val
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

/-- `afcRunningMax` is independent of the channel index `d`. -/
theorem afcRunningMax_eq
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d d' : Fin 128) :
    afcRunningMax qT kT vT keyScale qStart hi i d
      = afcRunningMax qT kT vT keyScale qStart hi i d' := by
  unfold afcRunningMax afcKeysUpto
  congr 1
  rw [List.map_filterMap, List.map_filterMap]
  apply List.filterMap_congr
  intro j _
  by_cases hj : j.val < hi ∧ j.val ≤ qStart + i.val <;> simp [afcKV, hj]
/-- **`afcRunningMax` over a nonempty causal window ≠ ⊥.** For `hi ≥ 1`, key `j = 0`
is always kept (causal `0 ≤ qStart + i`), so the running max is not `⊥`. -/
theorem afcRunningMax_ne_bot
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (hhi : 1 ≤ hi) (i d : Fin 128) :
    afcRunningMax qT kT vT keyScale qStart hi i d ≠ ⊥ := by
  unfold afcRunningMax afcKeysUpto
  set sc0 : ℝ := keyScale (⟨0, by norm_num⟩ : Fin 128) *
      Finset.univ.sum (fun e : Fin 128 => qT (i, e, PUnit.unit) *
        kT (⟨0, by norm_num⟩, e, PUnit.unit)) with hsc0
  have hmem : ((sc0 : ℝ) : WithBot ℝ) ∈
      ((List.finRange 128).filterMap (fun j : Fin 128 =>
        if j.val < hi ∧ j.val ≤ qStart + i.val
        then some (afcKV qT kT vT keyScale i d j) else none)).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ)) := by
    rw [List.mem_map]
    refine ⟨afcKV qT kT vT keyScale i d ⟨0, by norm_num⟩, ?_, rfl⟩
    rw [List.mem_filterMap]
    refine ⟨⟨0, by norm_num⟩, List.mem_finRange _, ?_⟩
    rw [if_pos ⟨show (0:Nat) < hi from by omega, show (0:Nat) ≤ qStart + i.val from Nat.zero_le _⟩]
  have hle := afc_mem_le_foldr_sup _ _ hmem
  intro hbot
  exact absurd (le_bot_iff.mp (hbot ▸ hle)) WithBot.coe_ne_bot


/-- **The `q·k` score cell (AFC, HEAD_ACTIVE).** With the invariant's masked `q`
(zero on head lanes `e ≥ 96`) and the loaded `k` block (`kmaskT(e,jL) =
(jL < 128 - SN) ∧ (e < 96)`, masked-out lanes reading `undef = 0`), the kernel's
`qk` dot cell `(i, jL)` equals
`(Σ_{e<128} qm(i,e)·kread e jL) · qsc · ksc`, where `qm` zeros `e ≥ 96`. -/
theorem afc_score_cell (s0 : BlockState) (qStart SN : Nat) (jL : Fin 64)
    (i : Fin 128) (qsc ksc : ℝ)
    (qm : TileIndex [128, 128] → ℝ) (kread : Fin 128 → Fin 64 → ℝ)
    (qtile : Tile .real [128, 128]) (ktile : Tile .real [128, 64])
    (kscT : Tile .real [])
    (hjLwin : jL.val < 128 - SN)
    (hqtile : ∀ e : Fin 128, qtile.data (i, e, PUnit.unit)
        = some (if e.val < 96 then qm (i, e, PUnit.unit) else 0))
    (hktile : ∀ e : Fin 128, ktile.data (e, jL, PUnit.unit)
        = some (if jL.val < 128 - SN ∧ e.val < 96 then kread e jL else 0))
    (hkscT : kscT.data PUnit.unit = some ksc) :
    (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          ⟨fun ix => (Tile.dot [] qtile ktile).data ix⟩
          (Tile.scalar (some qsc))) kscT).data (i, jL, PUnit.unit)
      = some ((Finset.univ.sum (fun e : Fin 128 =>
          (if e.val < 96 then qm (i, e, PUnit.unit) else 0) * kread e jL)) * qsc * ksc) := by
  have hdot : (Tile.dot [] qtile ktile).data (i, jL, PUnit.unit)
      = some (Finset.univ.sum (fun e : Fin 128 =>
          (if e.val < 96 then qm (i, e, PUnit.unit) else 0) * kread e jL)) := by
    rw [Tile.dot_nil_data]
    rw [show (@Finset.sum (Fin 128) (WithBot ℝ) _ Finset.univ
          (fun e => Option.map₂ (· * ·) (qtile.data (i, e, PUnit.unit)) (ktile.data (e, jL, PUnit.unit))))
        = @Finset.sum (Fin 128) (WithBot ℝ) _ Finset.univ
          (fun e => (some ((if e.val < 96 then qm (i, e, PUnit.unit) else 0) * kread e jL) : WithBot ℝ))
        from Finset.sum_congr rfl (fun e _ => by
          rw [hqtile e, hktile e]
          simp only [Option.map₂, Option.bind, Option.map]
          refine congrArg some ?_
          by_cases he : e.val < 96
          · rw [if_pos he, if_pos ⟨hjLwin, he⟩]
          · rw [if_neg he, if_neg (by simp [he])]; ring)]
    rw [WithBot.sum_someTerm_eq_some]
  simp only [Tile.bop_data, Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
    Tile.scalar, NumericDType.mul, hdot, hkscT]
  simp only [WithBot.realMul, Option.map₂, Option.bind, Option.map]


/-- **`reduceMax` row (AFC, [128,64]).** -/
theorem afc_reduceMaxDrop_row (qk : Tile .real [128, 64]) (rmaxT : Tile .real [128])
    (hrm : Tile.reduceMaxDrop afcAx1 qk = some rmaxT)
    (i : Fin 128) (g : Fin 64 → WithBot ℝ)
    (hqk : ∀ jL : Fin 64, qk.data (TileShape.insertAxisIndex [128, 64] afcAx1 (i, PUnit.unit) jL) = g jL) :
    rmaxT.data (i, PUnit.unit) = Finset.univ.sup g := by
  unfold Tile.reduceMaxDrop at hrm
  rw [dif_pos (show 0 < TileShape.axisDim [128, 64] afcAx1 from by decide)] at hrm
  rw [← Option.some.inj hrm]
  simp only [Finset.sup'_eq_sup]
  exact Finset.sup_congr rfl (fun jL _ => hqk jL)
/-- **Block sup (AFC).** -/
theorem afcBlock_blockSup
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart c : Nat) (i d : Fin 128) (hc1 : (c + 1) * 64 ≤ 128) :
    ((afcBlock qT kT vT keyScale qStart c i d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = Finset.univ.sup (fun jL : Fin 64 =>
          if (c * 64 + jL.val) ≤ qStart + i.val then
            (((afcKV qT kT vT keyScale i d ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 : ℝ) : WithBot ℝ)
          else (⊥ : WithBot ℝ)) := by
  rw [show (afcBlock qT kT vT keyScale qStart c i d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
      = ((List.finRange 128).filterMap (fun j : Fin 128 =>
          if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val
          then some ((afcKV qT kT vT keyScale i d j).1) else none)).map
            (fun x => ((x : ℝ) : WithBot ℝ)) from by
    unfold afcBlock
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr
    intro j _
    by_cases hj : c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val <;> simp [hj]]
  rw [afc_filterMap_foldr_sup 128
    (fun j => c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val)
    (fun j => (afcKV qT kT vT keyScale i d j).1)]
  apply le_antisymm
  · apply Finset.sup_le; intro j _
    by_cases hj : c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val
    · rw [if_pos hj]
      have hjL : j.val - c * 64 < 64 := by omega
      have hfin : (⟨c * 64 + (j.val - c * 64), by omega⟩ : Fin 128) = j := by apply Fin.ext; simp only; omega
      refine le_trans ?_ (Finset.le_sup (Finset.mem_univ (⟨j.val - c * 64, hjL⟩ : Fin 64)))
      simp only
      rw [if_pos (show c * 64 + (j.val - c * 64) ≤ qStart + i.val from by have := hj.2.2; omega)]
      apply le_of_eq
      rw [hfin]
    · rw [if_neg hj]; exact bot_le
  · apply Finset.sup_le; intro jL _
    have hb : c * 64 + jL.val < 128 := by have := jL.isLt; omega
    by_cases hkeep : c * 64 + jL.val ≤ qStart + i.val
    · rw [if_pos hkeep]
      refine le_trans ?_ (Finset.le_sup (Finset.mem_univ (⟨c * 64 + jL.val, hb⟩ : Fin 128)))
      simp only
      rw [if_pos (by have := jL.isLt; exact ⟨by omega, by omega, hkeep⟩)]
    · rw [if_neg hkeep]; exact bot_le
/-- Helper: `WithBot.realSub (some 0) (some 1e6) = some (-1000000)`. -/
theorem afc_sentinel_eq : WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ)) = some (-1000000.0 : ℝ) := by
  simp only [WithBot.realSub, Option.map₂, Option.bind, Option.map]; norm_num

/-- Under `afcScoreBound`, every causally-kept key's coerced score exceeds the
`-1e6` sentinel; hence the running max over a nonempty window does too. -/
theorem afcRunningMax_gt_sentinel
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ) (qStart hi : Nat)
    (hhi : 1 ≤ hi) (i d : Fin 128)
    (hsb : afcScoreBound qT kT vT keyScale qStart) :
    afcRunningMax qT kT vT keyScale qStart hi i d
      > some (-1000000.0 : ℝ) := by
  unfold afcRunningMax afcKeysUpto
  -- every coerced score in the list is > some(-1e6); the foldr ⊔ ⊥ inherits it via the nonempty key 0
  have hkey0 : ((afcKV qT kT vT keyScale i d ⟨0, by norm_num⟩).1 : ℝ) > -1000000.0 := by
    have := hsb ⟨0, by norm_num⟩ i d
    simpa [afcKV] using this
  -- key 0 is in the list
  have hmem : ((((afcKV qT kT vT keyScale i d ⟨0, by norm_num⟩).1 : ℝ)) : WithBot ℝ) ∈
      ((List.finRange 128).filterMap (fun j : Fin 128 =>
        if j.val < hi ∧ j.val ≤ qStart + i.val
        then some (afcKV qT kT vT keyScale i d j) else none)).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ)) := by
    rw [List.mem_map]
    refine ⟨afcKV qT kT vT keyScale i d ⟨0, by norm_num⟩, ?_, rfl⟩
    rw [List.mem_filterMap]
    refine ⟨⟨0, by norm_num⟩, List.mem_finRange _, ?_⟩
    rw [if_pos ⟨show (0:Nat) < hi from by omega, Nat.zero_le _⟩]
  -- and every element of the list is ≤ the foldr; combine with: foldr ≥ key0 > -1e6
  have hle := afc_mem_le_foldr_sup _ _ hmem
  refine lt_of_lt_of_le ?_ hle
  exact (WithBot.coe_lt_coe).mpr hkey0
set_option maxHeartbeats 1600000 in
/-- **`m_ij = afcRunningMax((c+1)·64)` (AFC, masked with -1e6 sentinel).** -/
theorem afc_mij_reg_eq_masked
    (qT kT vT : TileIndex [128, 128] → ℝ) (qStart : Nat)
    (keyScale : Fin 128 → ℝ) (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i : Fin 128)
    (hsb : afcScoreBound qT kT vT keyScale qStart)
    (qkSentT : Tile .real [128, 64]) (mtile rmaxT mijT : Tile .real [128])
    (hsent : ∀ jL : Fin 64, qkSentT.data (TileShape.insertAxisIndex [128, 64] afcAx1 (i, PUnit.unit) jL)
        = if qStart + i.val ≥ c * 64 + jL.val then
            (((afcKV qT kT vT keyScale i ⟨0, by norm_num⟩
                ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hrmax : Tile.reduceMaxDrop afcAx1 qkSentT = some rmaxT)
    (hmtile : mtile.data (i, PUnit.unit)
        = afcRunningMax qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩)
    (hmij : mijT = Tile.select
        (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT) :
    mijT.data (i, PUnit.unit)
      = afcRunningMax qT kT vT keyScale qStart ((c + 1) * 64) i ⟨0, by norm_num⟩ := by
  have hrmaxcell : rmaxT.data (i, PUnit.unit)
      = Finset.univ.sup (fun jL : Fin 64 =>
          if qStart + i.val ≥ c * 64 + jL.val then
            (((afcKV qT kT vT keyScale i ⟨0, by norm_num⟩ ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ)) :=
    afc_reduceMaxDrop_row qkSentT rmaxT hrmax i _ hsent
  rw [afcRunningMax_succ qT kT vT keyScale qStart c i ⟨0, by norm_num⟩]
  rw [afcBlock_blockSup qT kT vT keyScale qStart c i ⟨0, by norm_num⟩ hc1]
  set BSk : WithBot ℝ := Finset.univ.sup (fun jL : Fin 64 =>
      if (c * 64 + jL.val) ≤ qStart + i.val then
        (((afcKV qT kT vT keyScale i ⟨0, by norm_num⟩ ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 : ℝ) : WithBot ℝ)
      else (⊥ : WithBot ℝ)) with hBSk
  set M := afcRunningMax qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩ with hMdef
  -- BSk ≤ rmaxT ≤ BSk ⊔ some(-1e6)
  have hBSk_le : BSk ≤ rmaxT.data (i, PUnit.unit) := by
    rw [hrmaxcell, hBSk]
    apply Finset.sup_le; intro jL _
    split
    · rename_i hk
      refine Finset.le_sup_of_le (Finset.mem_univ jL) ?_
      rw [if_pos (show qStart + i.val ≥ c * 64 + jL.val from by omega)]
    · exact bot_le
  have hr_le : rmaxT.data (i, PUnit.unit) ≤ BSk ⊔ ((-1000000.0 : ℝ) : WithBot ℝ) := by
    rw [hrmaxcell, hBSk]
    apply Finset.sup_le; intro jL _
    split
    · rename_i hk
      refine le_sup_of_le_left (Finset.le_sup_of_le (Finset.mem_univ jL) ?_)
      rw [if_pos (show c * 64 + jL.val ≤ qStart + i.val from by omega)]
    · exact le_sup_right
  -- M ⊔ BSk ≥ some(-1e6): the total window [0,(c+1)·64) is nonempty (key 0 kept)
  have hMBSk_sentinel : ((-1000000.0 : ℝ) : WithBot ℝ) ≤ M ⊔ BSk := by
    by_cases hc0 : c = 0
    · -- key 0 lives in block 0 = BSk
      subst hc0
      refine le_sup_of_le_right ?_
      rw [hBSk]
      refine Finset.le_sup_of_le (Finset.mem_univ (0 : Fin 64)) ?_
      rw [if_pos (show 0 * 64 + (0 : Fin 64).val ≤ qStart + i.val from by simp)]
      refine le_of_lt ?_
      rw [WithBot.coe_lt_coe]
      have hbnd := hsb ⟨0 * 64 + (0 : Fin 64).val, by simp⟩ i ⟨0, by norm_num⟩
      simpa [afcKV] using hbnd
    · -- key 0 lives in the prefix window c·64 ≥ 64, so M > some(-1e6)
      refine le_sup_of_le_left (le_of_lt ?_)
      rw [hMdef]
      exact afcRunningMax_gt_sentinel qT kT vT keyScale qStart (c * 64) (by omega) i ⟨0, by norm_num⟩ hsb
  have hdom : M ⊔ (BSk ⊔ ((-1000000.0 : ℝ) : WithBot ℝ)) = M ⊔ BSk := by
    rw [← sup_assoc, sup_eq_left.mpr hMBSk_sentinel]
  -- mijT = max(M, rmaxT) and squeeze
  rw [hmij, Tile.select_data, Tile.cop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hmtile]
  -- goal: (if decide (M > rmaxT.data) then M else rmaxT.data) = M ⊔ BSk
  have hsqueeze : M ⊔ rmaxT.data (i, PUnit.unit) = M ⊔ BSk := by
    apply le_antisymm
    · exact sup_le_sup_left (le_trans hr_le (le_of_eq rfl)) M |>.trans (le_of_eq hdom) |>.trans (le_refl _)
    · exact sup_le_sup_left hBSk_le M
  by_cases hcmp : M > rmaxT.data (i, PUnit.unit)
  · rw [if_pos (by simpa using hcmp)]
    rw [← hsqueeze, max_eq_left (le_of_lt hcmp)]
  · rw [if_neg (by simpa using hcmp)]
    rw [← hsqueeze, max_eq_right (not_lt.mp hcmp)]


set_option maxHeartbeats 1600000 in
/-- **Masked `pT` cell (AFC).** The kernel's masked `p` tile cell on lane `jL` is
`some (pow2(score − Mc1))` when causally kept, `some 0` when masked — given the
`mij` cell `= Mc1` (`= afcRunningMax((c+1)·64) ≠ ⊥`). -/
theorem afc_pmT_cell_masked
    (qT kT vT : TileIndex [128, 128] → ℝ) (qStart : Nat) (keyScale : Fin 128 → ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i : Fin 128) (jL : Fin 64) (Mc1 : WithBot ℝ)
    (qkSentT : Tile .real [128, 64]) (mijT : Tile .real [128]) (pT : Tile .real [128, 64])
    (kept : Bool)
    (_hkept : kept = decide (qStart + i.val ≥ c * 64 + jL.val))
    (hsent : qkSentT.data (i, jL, PUnit.unit)
        = if kept then
            (((afcKV qT kT vT keyScale i ⟨0, by norm_num⟩
                ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hmij : mijT.data (i, PUnit.unit) = Mc1)
    (hkeptbot : kept = Bool.true → Mc1 ≠ ⊥)
    (hpT : pT.data (i, jL, PUnit.unit)
        = if kept then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ)) :
    pT.data (i, jL, PUnit.unit)
      = some (if kept then
          pow2 ((afcKV qT kT vT keyScale i ⟨0, by norm_num⟩
            ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 - Mc1.unbotD 0)
          else 0) := by
  rw [hpT]
  by_cases hk : kept = Bool.true
  · rw [if_pos hk]
    obtain ⟨Mr, hMr⟩ : ∃ Mr : ℝ, Mc1 = (Mr : WithBot ℝ) := by
      cases hh : Mc1 with
      | coe x => exact ⟨x, rfl⟩
      | bot => exact absurd hh (hkeptbot hk)
    simp only [Tile.uop_data, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.expandDim_data, TileShape.dropInsertedIndex, NumericDType.sub, hmij, hMr,
      WithBot.unbotD_coe]
    rw [show qkSentT.data (i, jL, PUnit.unit) =
          (((afcKV qT kT vT keyScale i ⟨0, by norm_num⟩
              ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 : ℝ) : WithBot ℝ) from by
      rw [hsent, if_pos hk]]
    rw [if_pos hk]
    simp only [WithBot.realSub, Option.map₂, Option.bind, Option.map, WithBot.realExp2_some]
    refine congrArg some ?_
    simp only [pow2]; ring_nf
  · rw [if_neg hk, if_neg hk]
    norm_num
set_option maxHeartbeats 1600000 in
/-- **`Σ_jL pT[i,jL] = afcBlock` pow2-score sum (AFC, masked).** The kernel's masked
`p` reduceSum equals the windowed `afcBlock` pow2-score sum (shifted by `Mc1 =
afcRunningMax((c+1)·64)`). -/
theorem afc_nume_row_sum_masked
    (qT kT vT : TileIndex [128, 128] → ℝ) (qStart : Nat) (keyScale : Fin 128 → ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 128)
    (qkSentT : Tile .real [128, 64]) (mijT : Tile .real [128]) (pT : Tile .real [128, 64])
    (hsent : ∀ jL : Fin 64, qkSentT.data (i, jL, PUnit.unit)
        = if qStart + i.val ≥ c * 64 + jL.val then
            (((afcKV qT kT vT keyScale i ⟨0, by norm_num⟩
                ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hmij : mijT.data (i, PUnit.unit)
        = afcRunningMax qT kT vT keyScale qStart ((c + 1) * 64) i ⟨0, by norm_num⟩)
    (hpT : ∀ jL : Fin 64, pT.data (i, jL, PUnit.unit)
        = if (qStart + i.val ≥ c * 64 + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ)) :
    (Tile.reduceSumDrop afcAx1 pT).data (i, PUnit.unit)
      = some ((afcBlock qT kT vT keyScale qStart c i d).map
          (fun p => pow2 (p.1 - (afcRunningMax qT kT vT keyScale
            qStart ((c + 1) * 64) i ⟨0, by norm_num⟩).unbotD 0))).sum := by
  set Mc1 := afcRunningMax qT kT vT keyScale qStart ((c + 1) * 64) i ⟨0, by norm_num⟩ with hMc1
  have hMc1bot : Mc1 ≠ ⊥ := by
    rw [hMc1]; exact afcRunningMax_ne_bot qT kT vT keyScale qStart ((c + 1) * 64) (by omega) i ⟨0, by norm_num⟩
  rw [Tile.reduceSumDrop_data]
  have hcell : ∀ jL : Fin 64,
      pT.data (TileShape.insertAxisIndex [128, 64] afcAx1 (i, PUnit.unit) jL)
        = some (if decide (qStart + i.val ≥ c * 64 + jL.val) then
            pow2 ((afcKV qT kT vT keyScale i ⟨0, by norm_num⟩ ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 - Mc1.unbotD 0)
            else 0) := by
    intro jL
    rw [show (TileShape.insertAxisIndex [128, 64] afcAx1 (i, PUnit.unit) jL) = (i, jL, PUnit.unit) from rfl]
    have hsentR : qkSentT.data (i, jL, PUnit.unit)
        = if decide (qStart + i.val ≥ c * 64 + jL.val) then
            (((afcKV qT kT vT keyScale i ⟨0, by norm_num⟩ ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ) := by
      rw [hsent jL]; by_cases h : qStart + i.val ≥ c * 64 + jL.val <;> simp [h]
    have hpTR : pT.data (i, jL, PUnit.unit)
        = if decide (qStart + i.val ≥ c * 64 + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ) := by
      rw [hpT jL]; by_cases h : qStart + i.val ≥ c * 64 + jL.val
      · rw [if_pos h, if_pos (decide_eq_true h)]
      · rw [if_neg h, if_neg (by simp [h])]
    exact afc_pmT_cell_masked qT kT vT qStart keyScale c hc1 i jL Mc1 qkSentT mijT pT
      (decide (qStart + i.val ≥ c * 64 + jL.val)) (by rfl) hsentR
      (by rw [hmij]) (fun _ => hMc1bot) hpTR
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [afcBlock_map_sum qT kT vT keyScale qStart c i d hc1
      (fun p => pow2 (p.1 - Mc1.unbotD 0))]
  refine Finset.sum_congr rfl (fun jL _ => ?_)
  simp only [decide_eq_true_eq]
  by_cases hkp : (c * 64 + jL.val) ≤ qStart + i.val
  · rw [if_pos (show qStart + i.val ≥ c * 64 + jL.val from by omega),
        if_pos (show c * 64 + jL.val ≤ qStart + i.val from hkp)]
    rfl
  · rw [if_neg (show ¬ qStart + i.val ≥ c * 64 + jL.val from by omega),
        if_neg (show ¬ c * 64 + jL.val ≤ qStart + i.val from hkp)]



set_option maxHeartbeats 1600000 in
/-- **`Σ_jL pT[i,jL]·v[jL,d] = afcBlock` pow2-score·value sum (AFC, masked).** The
kernel's `dot(p, v)` numerator increment equals the windowed `afcBlock`
pow2-score·value sum. `vval jL = vTileAFC-value at global key c·64+jL, channel d`
(carried by the masked v load: for `d < 96` it is `vTileAFC(c·64+jL, d)`). -/
theorem afc_acc_dot_block_masked
    (qT kT vT : TileIndex [128, 128] → ℝ) (qStart : Nat) (keyScale : Fin 128 → ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 128)
    (qkSentT : Tile .real [128, 64]) (mijT : Tile .real [128]) (pT : Tile .real [128, 64])
    (vtile : Tile .real [64, 128]) (vval : Fin 64 → ℝ)
    (hsent : ∀ jL : Fin 64, qkSentT.data (i, jL, PUnit.unit)
        = if qStart + i.val ≥ c * 64 + jL.val then
            (((afcKV qT kT vT keyScale i ⟨0, by norm_num⟩
                ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hmij : mijT.data (i, PUnit.unit)
        = afcRunningMax qT kT vT keyScale qStart ((c + 1) * 64) i ⟨0, by norm_num⟩)
    (hpT : ∀ jL : Fin 64, pT.data (i, jL, PUnit.unit)
        = if (qStart + i.val ≥ c * 64 + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ))
    (hv : ∀ jL : Fin 64, vtile.data (jL, d, PUnit.unit) = some (vval jL))
    (hvval : ∀ jL : Fin 64, vval jL
        = (afcKV qT kT vT keyScale i d ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).2) :
    (Tile.dot [] pT vtile).data (i, d, PUnit.unit)
      = some ((afcBlock qT kT vT keyScale qStart c i d).map
          (fun p => pow2 (p.1 - (afcRunningMax qT kT vT keyScale
            qStart ((c + 1) * 64) i ⟨0, by norm_num⟩).unbotD 0) * p.2)).sum := by
  set Mc1 := afcRunningMax qT kT vT keyScale qStart ((c + 1) * 64) i ⟨0, by norm_num⟩ with hMc1
  have hMc1bot : Mc1 ≠ ⊥ := by
    rw [hMc1]; exact afcRunningMax_ne_bot qT kT vT keyScale qStart ((c + 1) * 64) (by omega) i ⟨0, by norm_num⟩
  -- the masked pT cell value
  have hpcell : ∀ jL : Fin 64, pT.data (i, jL, PUnit.unit)
      = some (if decide (qStart + i.val ≥ c * 64 + jL.val) then
          pow2 ((afcKV qT kT vT keyScale i ⟨0, by norm_num⟩ ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 - Mc1.unbotD 0)
          else 0) := by
    intro jL
    have hsentR : qkSentT.data (i, jL, PUnit.unit)
        = if decide (qStart + i.val ≥ c * 64 + jL.val) then
            (((afcKV qT kT vT keyScale i ⟨0, by norm_num⟩ ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ) := by
      rw [hsent jL]; by_cases h : qStart + i.val ≥ c * 64 + jL.val <;> simp [h]
    have hpTR : pT.data (i, jL, PUnit.unit)
        = if decide (qStart + i.val ≥ c * 64 + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ) := by
      rw [hpT jL]; by_cases h : qStart + i.val ≥ c * 64 + jL.val
      · rw [if_pos h, if_pos (decide_eq_true h)]
      · rw [if_neg h, if_neg (by simp [h])]
    have := afc_pmT_cell_masked qT kT vT qStart keyScale c hc1 i jL Mc1 qkSentT mijT pT
      (decide (qStart + i.val ≥ c * 64 + jL.val)) (by rfl) hsentR (by rw [hmij]) (fun _ => hMc1bot) hpTR
    rw [this]
  rw [Tile.dot_nil_data]
  have hterm : ∀ jL : Fin 64,
      Option.map₂ (· * ·) (pT.data (i, jL, PUnit.unit)) (vtile.data (jL, d, PUnit.unit))
        = some (if (qStart + i.val ≥ c * 64 + jL.val) then
            pow2 ((afcKV qT kT vT keyScale i ⟨0, by norm_num⟩ ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 - Mc1.unbotD 0)
              * vval jL
            else 0) := by
    intro jL
    rw [hpcell jL, hv jL]
    simp only [Option.map₂, Option.bind, Option.map, decide_eq_true_eq]
    refine congrArg some ?_
    by_cases h : qStart + i.val ≥ c * 64 + jL.val
    · rw [if_pos h, if_pos h]
    · rw [if_neg h, if_neg h]; ring
  rw [show (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
        (fun jL => Option.map₂ (· * ·) (pT.data (i, jL, PUnit.unit)) (vtile.data (jL, d, PUnit.unit))))
      = @Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ (fun jL =>
          (some (if (qStart + i.val ≥ c * 64 + jL.val) then
              pow2 ((afcKV qT kT vT keyScale i ⟨0, by norm_num⟩ ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 - Mc1.unbotD 0)
                * vval jL else 0) : WithBot ℝ))
      from Finset.sum_congr rfl (fun jL _ => hterm jL)]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [afcBlock_map_sum qT kT vT keyScale qStart c i d hc1
      (fun p => pow2 (p.1 - Mc1.unbotD 0) * p.2)]
  refine Finset.sum_congr rfl (fun jL _ => ?_)
  by_cases hkp : (c * 64 + jL.val) ≤ qStart + i.val
  · rw [if_pos (show qStart + i.val ≥ c * 64 + jL.val from by omega),
        if_pos (show c * 64 + jL.val ≤ qStart + i.val from hkp)]
    rw [hvval jL,
      show (afcKV qT kT vT keyScale i ⟨0, by norm_num⟩ ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1
          = (afcKV qT kT vT keyScale i d ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 from by
        simp only [afcKV]]
  · rw [if_neg (show ¬ qStart + i.val ≥ c * 64 + jL.val from by omega),
        if_neg (show ¬ c * 64 + jL.val ≤ qStart + i.val from hkp)]


/-- **Seed-1 vs seed-0 carry cancellation (AFC).** At the block-`c` transition the
rescale factor `α = exp2(m ⊖ Mc1)` (with `m = afcRunningMax(c·64)`) annihilates the
difference between the seed-`1` state `afcStateBot1(c·64)` and the seed-`0` state
`afcStateBot(c·64)`: when `c = 0` the running max is `⊥` so `α = 0`; when `c ≥ 1`
the two states coincide. -/
theorem afcStateBot1_cancel
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart c : Nat) (i d : Fin 128) (Mc1 : WithBot ℝ) :
    let m := (afcStateBot qT kT vT keyScale qStart (c * 64) i d).1
    let α := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
    (afcStateBot1 qT kT vT keyScale qStart (c * 64) i d).2.1 * α
        = (afcStateBot qT kT vT keyScale qStart (c * 64) i d).2.1 * α
      ∧ (afcStateBot1 qT kT vT keyScale qStart (c * 64) i d).2.2 * α
        = (afcStateBot qT kT vT keyScale qStart (c * 64) i d).2.2 * α := by
  intro m α
  by_cases hc0 : c = 0
  · subst hc0
    have hmbot : m = ⊥ := by
      show (afcStateBot qT kT vT keyScale qStart (0 * 64) i d).1 = ⊥
      rw [afcStateBot_fst_eq_runningMax, Nat.zero_mul, afcRunningMax_zero]
    have hα0 : α = 0 := by
      show (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 = 0
      rw [hmbot, WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
    rw [hα0]; simp
  · have hne : afcRunningMax qT kT vT keyScale qStart (c * 64) i d ≠ ⊥ :=
      afcRunningMax_ne_bot qT kT vT keyScale qStart (c * 64) (by omega) i d
    rw [afcStateBot1_eq_afcStateBot qT kT vT keyScale qStart (c * 64) i d hne]
    exact ⟨rfl, rfl⟩
/-- If the ⊥-seeded running max over `[0, hi)` is `⊥`, the key list is empty, so its
`pow2`-score (resp. `·v`) sum is `0`. -/
theorem afcKeysUpto_sum_zero_of_bot
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i d : Fin 128)
    (hbot : afcRunningMax qT kT vT keyScale qStart hi i d = ⊥) (h : ℝ × ℝ → ℝ) :
    ((afcKeysUpto qT kT vT keyScale qStart hi i d).map h).sum = 0 := by
  rw [show afcKeysUpto qT kT vT keyScale qStart hi i d = [] from ?_, List.map_nil, List.sum_nil]
  by_contra hne
  obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ hne
  have hmem : ((p.1 : ℝ) : WithBot ℝ) ∈
      (afcKeysUpto qT kT vT keyScale qStart hi i d).map (fun q => ((q.1 : ℝ) : WithBot ℝ)) :=
    List.mem_map_of_mem hp
  have := afc_mem_le_foldr_sup _ _ hmem
  rw [← afcRunningMax, hbot] at this
  exact absurd (le_bot_iff.mp this) WithBot.coe_ne_bot

/-- ⊥-seeded denominator anchor: `l = κ(M_c)·L_c`. -/
theorem afc_denom_anchor
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i d : Fin 128) :
    (afcStateBot qT kT vT keyScale qStart hi i d).2.1
      = ((afcStateBot qT kT vT keyScale qStart hi i d).1.elim 0 (fun r => pow2 (-r)))
        * (0 + ((afcKeysUpto qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1)).sum) := by
  rw [afcStateBot_snd_fst, afcStateBot_fst_eq_runningMax]

/-- ⊥-seeded accumulator anchor: `acc = κ(M_c)·T_c`. -/
theorem afc_acc_anchor
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i d : Fin 128) :
    (afcStateBot qT kT vT keyScale qStart hi i d).2.2
      = ((afcStateBot qT kT vT keyScale qStart hi i d).1.elim 0 (fun r => pow2 (-r)))
        * (0 + ((afcKeysUpto qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1 * p.2)).sum) := by
  rw [afcStateBot_snd_snd, afcStateBot_fst_eq_runningMax]
set_option maxHeartbeats 1600000 in
/-- **`l_i' = afcStateBot((c+1)·64).2.1` (AFC, masked).** The kernel's denominator
carry `l_i·α + l_ij` (with `l_i = afcStateBot1(c·64).2.1`) lands on the seed-0
⊥-state denominator after `c+1` blocks (= seed-1 on the nonempty `(c+1)·64` window). -/
theorem afc_denom_reg_eq_masked
    (qT kT vT : TileIndex [128, 128] → ℝ) (qStart : Nat) (keyScale : Fin 128 → ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i : Fin 128)
    (qkSentT : Tile .real [128, 64]) (mtile mijT alphaT litile lijT : Tile .real [128])
    (pT : Tile .real [128, 64])
    (hsent : ∀ jL : Fin 64, qkSentT.data (i, jL, PUnit.unit)
        = if qStart + i.val ≥ c * 64 + jL.val then
            (((afcKV qT kT vT keyScale i ⟨0, by norm_num⟩
                ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hlitile : litile.data (i, PUnit.unit) = some
        ((afcStateBot1 qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩).2.1))
    (hmtile : mtile.data (i, PUnit.unit)
        = afcRunningMax qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩)
    (hmij : mijT.data (i, PUnit.unit)
        = afcRunningMax qT kT vT keyScale qStart ((c + 1) * 64) i ⟨0, by norm_num⟩)
    (halpha : alphaT = Tile.uop WithBot.realExp2
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
    (hlijT : lijT = Tile.reduceSumDrop afcAx1 pT)
    (hpT : ∀ jL : Fin 64, pT.data (i, jL, PUnit.unit)
        = if (qStart + i.val ≥ c * 64 + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ)) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
        (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) litile alphaT) lijT).data (i, PUnit.unit)
      = some ((afcStateBot qT kT vT keyScale qStart ((c + 1) * 64) i ⟨0, by norm_num⟩).2.1) := by
  set m := (afcStateBot qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩).1 with hm_def
  set Mc := afcRunningMax qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩ with hMc
  set Mc1 := afcRunningMax qT kT vT keyScale qStart ((c + 1) * 64) i ⟨0, by norm_num⟩ with hMc1
  have hmMc : m = Mc := by rw [hm_def, hMc, afcStateBot_fst_eq_runningMax]
  have hMsucc : Mc1 = m ⊔ ((afcBlock qT kT vT keyScale qStart c i ⟨0, by norm_num⟩).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    rw [hMc1, afcRunningMax_succ, hmMc, ← hMc]
  have halphaVal : alphaT.data (i, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [halpha]; show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmtile, hmij,
      NumericDType.sub, ← hMc, ← hMc1, hmMc]
  have hsum := afc_nume_row_sum_masked qT kT vT qStart keyScale c hc1 i ⟨0, by norm_num⟩ qkSentT mijT pT hsent hmij hpT
  have hblockEq := osStepBot_block_eq m
    ((afcStateBot qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩).2.1)
    ((afcStateBot qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩).2.2)
    ((afcKeysUpto qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩).map (fun p => pow2 p.1 * p.2)).sum
    ((afcKeysUpto qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩).map (fun p => pow2 p.1)).sum
    (afcBlock qT kT vT keyScale qStart c i ⟨0, by norm_num⟩)
    (by rw [afc_denom_anchor, zero_add, hm_def])
    (by rw [afc_acc_anchor, zero_add, hm_def])
    (fun hbot => afcKeysUpto_sum_zero_of_bot qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩
      (by rw [← afcStateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => afcKeysUpto_sum_zero_of_bot qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩
      (by rw [← afcStateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  rw [show (afcStateBot qT kT vT keyScale qStart ((c + 1) * 64) i ⟨0, by norm_num⟩).2.1
        = (Mc1, (afcStateBot qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩).2.1
              * (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
              + ((afcBlock qT kT vT keyScale qStart c i ⟨0, by norm_num⟩).map (fun p => pow2 (p.1 - Mc1.unbotD 0))).sum,
            _).2.1 from by
    rw [afcStateBot_succ]; rw [← hblockEq]]
  set α : ℝ := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp2 (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  have hcancel := (afcStateBot1_cancel qT kT vT keyScale qStart c i ⟨0, by norm_num⟩ Mc1).1
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex]
  rw [hlijT]
  erw [hsum]
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    hlitile, halphaVal, hαsome]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [show (afcStateBot1 qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩).2.1 * α
        = (afcStateBot qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩).2.1 * α from by
    have := hcancel; simp only [← hm_def, ← hαdef] at this ⊢; exact this]


set_option maxHeartbeats 1600000 in
/-- **`acc' = afcStateBot((c+1)·64).2.2` (AFC, masked).** The kernel's accumulator
carry `acc·α + dot(p, v)` lands on the seed-0 ⊥-state accumulator after `c+1`
blocks. The masked v-load value at channel `d` is carried via `vval`/`hvval`
(matching `afcKV`'s value component). -/
theorem afc_acc_reg_eq_masked
    (qT kT vT : TileIndex [128, 128] → ℝ) (qStart : Nat) (keyScale : Fin 128 → ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 128)
    (qkSentT : Tile .real [128, 64]) (mtile mijT alphaT : Tile .real [128])
    (acctile acc1T : Tile .real [128, 128]) (pT : Tile .real [128, 64]) (vtile : Tile .real [64, 128]) (vval : Fin 64 → ℝ)
    (hsent : ∀ jL : Fin 64, qkSentT.data (i, jL, PUnit.unit)
        = if qStart + i.val ≥ c * 64 + jL.val then
            (((afcKV qT kT vT keyScale i ⟨0, by norm_num⟩
                ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hacctile : acctile.data (i, d, PUnit.unit) = some
        ((afcStateBot1 qT kT vT keyScale qStart (c * 64) i d).2.2))
    (hmtile : mtile.data (i, PUnit.unit)
        = afcRunningMax qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩)
    (hmij : mijT.data (i, PUnit.unit)
        = afcRunningMax qT kT vT keyScale qStart ((c + 1) * 64) i ⟨0, by norm_num⟩)
    (halpha : alphaT = Tile.uop WithBot.realExp2
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
    (hacc1 : acc1T = Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
    (hpT : ∀ jL : Fin 64, pT.data (i, jL, PUnit.unit)
        = if (qStart + i.val ≥ c * 64 + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ))
    (hv : ∀ jL : Fin 64, vtile.data (jL, d, PUnit.unit) = some (vval jL))
    (hvval : ∀ jL : Fin 64, vval jL
        = (afcKV qT kT vT keyScale i d ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).2) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        acc1T (Tile.dot [] pT vtile)).data (i, d, PUnit.unit)
      = some ((afcStateBot qT kT vT keyScale qStart ((c + 1) * 64) i d).2.2) := by
  set m := (afcStateBot qT kT vT keyScale qStart (c * 64) i d).1 with hm_def
  set Mc := afcRunningMax qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩ with hMc
  set Mc1 := afcRunningMax qT kT vT keyScale qStart ((c + 1) * 64) i ⟨0, by norm_num⟩ with hMc1
  have hmMc : m = Mc := by
    rw [hm_def, hMc, afcStateBot_fst_eq_runningMax,
      afcRunningMax_eq qT kT vT keyScale qStart (c * 64) i d ⟨0, by norm_num⟩]
  have hmd : m = afcRunningMax qT kT vT keyScale qStart (c * 64) i d := by
    rw [hm_def, afcStateBot_fst_eq_runningMax]
  have hMsucc : Mc1 = m ⊔ ((afcBlock qT kT vT keyScale qStart c i d).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    rw [hMc1, afcRunningMax_eq qT kT vT keyScale qStart ((c + 1) * 64) i ⟨0, by norm_num⟩ d,
      afcRunningMax_succ, hmd]
  have halphaVal : alphaT.data (i, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [halpha]; show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmtile, hmij,
      NumericDType.sub, ← hMc, ← hMc1, hmMc]
  have hdot := afc_acc_dot_block_masked qT kT vT qStart keyScale c hc1 i d qkSentT mijT pT vtile vval
    hsent hmij hpT hv hvval
  have hblockEq := osStepBot_block_eq m
    ((afcStateBot qT kT vT keyScale qStart (c * 64) i d).2.1)
    ((afcStateBot qT kT vT keyScale qStart (c * 64) i d).2.2)
    ((afcKeysUpto qT kT vT keyScale qStart (c * 64) i d).map (fun p => pow2 p.1 * p.2)).sum
    ((afcKeysUpto qT kT vT keyScale qStart (c * 64) i d).map (fun p => pow2 p.1)).sum
    (afcBlock qT kT vT keyScale qStart c i d)
    (by rw [afc_denom_anchor, zero_add, hm_def])
    (by rw [afc_acc_anchor, zero_add, hm_def])
    (fun hbot => afcKeysUpto_sum_zero_of_bot qT kT vT keyScale qStart (c * 64) i d
      (by rw [← afcStateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => afcKeysUpto_sum_zero_of_bot qT kT vT keyScale qStart (c * 64) i d
      (by rw [← afcStateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  rw [show (afcStateBot qT kT vT keyScale qStart ((c + 1) * 64) i d).2.2
        = (Mc1, _,
            (afcStateBot qT kT vT keyScale qStart (c * 64) i d).2.2
              * (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
              + ((afcBlock qT kT vT keyScale qStart c i d).map (fun p => pow2 (p.1 - Mc1.unbotD 0) * p.2)).sum).2.2
        from by rw [afcStateBot_succ]; rw [← hblockEq]]
  set α : ℝ := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp2 (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  have hcancel := (afcStateBot1_cancel qT kT vT keyScale qStart c i d Mc1).2
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex]
  erw [hdot]
  rw [hacc1, Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, hacctile, halphaVal, hαsome]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [show (afcStateBot1 qT kT vT keyScale qStart (c * 64) i d).2.2 * α
        = (afcStateBot qT kT vT keyScale qStart (c * 64) i d).2.2 * α from by
    have := hcancel; simp only [← hm_def, ← hαdef] at this ⊢; exact this]


/-! ## Step lemma + postLoop + top theorem (assembly) -/

/-- **Generic register-frame lemma.** If every statement in `body`, when stepped,
preserves the register `(d, sh, name)`, then so does the whole `stepStmts` run. -/
theorem stepStmts_regs_preserved {body : List Stmt} {s s' : BlockState}
    {d : TileDType} {sh : TileShape} {name : RegName}
    (hpre : ∀ st s1 s2, st ∈ body → stepStmt st s1 = some s2 → s2.regs d sh name = s1.regs d sh name)
    (h : stepStmts body s = some s') :
    s'.regs d sh name = s.regs d sh name := by
  induction body generalizing s with
  | nil => simp only [stepStmts] at h; rw [← Option.some.inj h]
  | cons st rest ih =>
    unfold stepStmts at h
    cases hst : stepStmt st s with
    | none => rw [hst] at h; simp at h
    | some mid =>
      rw [hst] at h
      have hmid := hpre st s mid (by simp) hst
      rw [ih (fun st' s1 s2 hmem hstep => hpre st' s1 s2 (by simp [hmem]) hstep) h, hmid]

/-- An `assign` to register `name'` preserves any other register `(d, sh, name)`
(`name ≠ name'`), whatever the assigned value. -/
theorem stepStmt_assign_regs_ne {d d' : TileDType} {sh sh' : TileShape}
    {name name' : RegName} {e : Op d' sh'} {s s' : BlockState}
    (hne : name ≠ name') (h : stepStmt (.assign d' sh' name' e) s = some s') :
    s'.regs d sh name = s.regs d sh name := by
  cases hev : evalOp e s with
  | none => rw [show stepStmt (.assign d' sh' name' e) s = none from by simp [stepStmt, hev]] at h
            exact absurd h.symm (Option.some_ne_none s')
  | some v =>
      rw [stepStmt_assign_eq_some hev] at h
      rw [← Option.some.inj h, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- The loop body `afcLoopBody` does not assign `O_block_ptr`, so it is preserved
through the streaming loop iteration. -/
theorem afcLoopBody_preserves_OblockPtr {s s' : BlockState}
    (h : stepStmts AfcFoundation.afcLoopBody s = some s') :
    s'.regs .ptr [128, 128] "O_block_ptr" = s.regs .ptr [128, 128] "O_block_ptr" := by
  refine stepStmts_regs_preserved (fun st s1 s2 hmem hstep => ?_) h
  fin_cases hmem <;> exact stepStmt_assign_regs_ne (by decide) hstep

/-- **The kernel's per-key score scale carrier** `qk_scale = q_scale · k_scale`,
loaded once per program (`q_scale = QScale[off_hz·1 + start_m]`) and once per key
block (`k_scale = KScale[off_hz·2 + start_n/BLOCK_N]`). Key `j` lives in block
`j / 64`, so its scale is `q_scale · KScale[off_hz·2 + j/64]`. -/
noncomputable def keyScaleAFC (s : BlockState) (QScale KScale : RegionName) :
    Fin 128 → ℝ :=
  fun j => s.readMem QScale (s.pids 1 * 1 + s.pids 0)
            * s.readMem KScale (s.pids 1 * 2 + j.val / 64)

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Step lemma (AFC, causal, masked).** -/
theorem afc_attn_step (Q K V QScale KScale Out : RegionName) (s0 : BlockState)
    (i : Nat) (s : BlockState) (hilt : i < 128) (himod : i % 64 = 0)
    (hsb : afcScoreBound (qTileAFCm s0 Q) (kTileAFC s0 K) (vTileAFCm s0 V)
      (keyScaleAFC s0 QScale KScale) (qStartAFC s0))
    (hinv : afcInvariant Q K V QScale KScale Out s0 (keyScaleAFC s0 QScale KScale) i s) :
    ∃ s', stepStmts AfcFoundation.afcLoopBody (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ afcInvariant Q K V QScale KScale Out s0 (keyScaleAFC s0 QScale KScale) (i + 64) s' := by
  set keyScale := keyScaleAFC s0 QScale KScale with hkeyScale
  set qStart := qStartAFC s0 with hqStart
  set qT := qTileAFCm s0 Q with hqT
  set kT := kTileAFC s0 K with hkT
  set vT := vTileAFCm s0 V with hvT
  simp only [afcInvariant] at hinv
  obtain ⟨hpids, hmod, hile, hmi, hli, hacc, hoffsm, hoffsn,
    hq, hqs, hKp, hKsp, hVp, hOp, hundef, hmem⟩ := hinv
  set c := i / 64 with hc_def
  have hi : i = c * 64 := by omega
  have hc1 : (c + 1) * 64 ≤ 128 := by omega
  set sin := s.setReg "start_n" .nat [] (Tile.scalar i) with hsin
  have hpids' : sin.pids = s0.pids := by rw [hsin]; simpa using hpids
  have hmem' : sin.mem = s0.mem := by
    rw [hsin]; funext rg o; rw [BlockState.setReg_mem]; exact congrFun (congrFun hmem rg) o
  have hundef' : ∀ rg o, sin.undef rg o = 0 := by
    intro rg o; rw [hsin, BlockState.setReg_undef]; exact hundef rg o
  obtain ⟨sF, hchain, hpidsF, hmemF, hundefF,
      kmaskT, ktile, kscT, qkdotT, maskT, qkSentT, rmaxT, mijT, pT, lijT, alphaT,
      vmaskT, vtile, pf16,
      hkmaskT, hktile, hkscT, hqkdotT, hmaskT, hqkSentT, hrm, hmijT, hpT, hlijT, halphaT,
      hvmaskT, hvtile, hpf16, hFmi, hFli, hFacc, hFoffsm, hFoffsn, hFq, hFqsc, hFKp, hFKsp, hFVp⟩ :=
    afcLoopBody_steps sin i
      (Tile.vec (fun r : Fin 128 => qStart + r.val)) (Tile.vec (fun j : Fin 64 => j.val))
      (kPtrsAFC s0 K c) (kScalePtrAFC s0 KScale c) (vPtrsAFC s0 V c)
      ⟨fun r : TileIndex [128] => afcRunningMax qT kT vT keyScale qStart i r.1 ⟨0, by norm_num⟩⟩
      ⟨fun idx : TileIndex [128, 128] =>
        if qStart + idx.1.val < 128 ∧ idx.2.1.val < 96 then some (qTileAFC s0 Q idx) else some (0.0 : ℝ)⟩
      (Tile.scalar (some (s0.readMem QScale (s0.pids 1 * ((128 + 128 - 1) / 128) + s0.pids 0))))
      ⟨fun r : TileIndex [128] => ((afcStateBot1 qT kT vT keyScale qStart i r.1 ⟨0, by norm_num⟩).2.1 : ℝ)⟩
      ⟨fun idx : TileIndex [128, 128] => ((afcStateBot1 qT kT vT keyScale qStart i idx.1 idx.2.1).2.2 : ℝ)⟩
      (by rw [hsin]; simpa using rfl)
      (by rw [hsin]; simpa using hoffsm)
      (by rw [hsin]; simpa using hoffsn)
      (by rw [hsin]; simpa using hmi)
      (by rw [hsin]; simpa using hli)
      (by rw [hsin]; simpa using hacc)
      (by rw [hsin]; simpa using hq)
      (by rw [hsin]; simpa using hqs)
      (by rw [hsin, hc_def]; simpa using hKp)
      (by rw [hsin, hc_def]; simpa using hKsp)
      (by rw [hsin, hc_def]; simpa using hVp)
  refine ⟨sF, hchain, ?_⟩
  -- scalar scales
  set qsc : ℝ := s0.readMem QScale (s0.pids 1 * ((128 + 128 - 1) / 128) + s0.pids 0) with hqscv
  set ksc : ℝ := s0.readMem KScale (s0.pids 1 * ((128 + 64 - 1) / 64) + c) with hkscv
  -- k_scale cell
  have hkscData : kscT.data PUnit.unit = some ksc := by
    rw [hkscT]; simp only [kScalePtrAFC, Region.cast, hkscv]
    refine congrArg some ?_
    unfold BlockState.readMem; rw [hmem']
  -- keyScale at a block-c key
  have hkeyBlock : ∀ jL : Fin 64,
      keyScale (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ : Fin 128) = qsc * ksc := by
    intro jL
    have hdiv : (c * 64 + jL.val) / 64 = c := by
      have := jL.isLt; omega
    simp only [hkeyScale, keyScaleAFC, hqscv, hkscv, hdiv]
  -- the q register cell in afc_score_cell `hqtile` form (qm = qT = qTileAFCm)
  have hqcell : ∀ (ir : Fin 128) (e : Fin 128),
      (⟨fun idx : TileIndex [128, 128] =>
        if qStart + idx.1.val < 128 ∧ idx.2.1.val < 96 then some (qTileAFC s0 Q idx) else some (0.0 : ℝ)⟩
          : Tile .real [128, 128]).data (ir, e, PUnit.unit)
        = some (if e.val < 96 then qT (ir, e, PUnit.unit) else 0) := by
    intro ir e
    show (if qStart + ir.val < 128 ∧ e.val < 96 then some (qTileAFC s0 Q (ir, e, PUnit.unit)) else some (0.0 : ℝ))
        = some (if e.val < 96 then qT (ir, e, PUnit.unit) else 0)
    rw [hqT]; simp only [qTileAFCm, hqStart]
    by_cases he : e.val < 96
    · by_cases hb : qStartAFC s0 + ir.val < 128
      · rw [if_pos ⟨hb, he⟩, if_pos he, if_pos ⟨hb, he⟩]
      · rw [if_neg (fun h => hb h.1), if_pos he, if_neg (fun h => hb h.1)]; norm_num
    · rw [if_neg (fun h => he h.2), if_neg he]; norm_num
  -- the k load cell in afc_score_cell `hktile` form (kread = kT global key)
  have hkcell : ∀ (e : Fin 128) (jL : Fin 64),
      ktile.data (e, jL, PUnit.unit)
        = some (if jL.val < 128 - i ∧ e.val < 96 then
            kT (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit) else 0) := by
    intro e jL
    rw [hktile]; simp only [hkmaskT, kPtrsAFC, Region.cast, Tile.vec]
    rw [show (ComparableDType.nat.lt jL.val (128 - i) && ComparableDType.nat.lt e.val 96)
          = decide (jL.val < 128 - i ∧ e.val < 96) from by
      rw [Bool.eq_iff_iff]; simp only [Bool.and_eq_true, ComparableDType.nat_lt_eq_true,
        decide_eq_true_eq]]
    by_cases hb : jL.val < 128 - i ∧ e.val < 96
    · rw [if_pos (by simp only [decide_eq_true_eq]; exact hb), if_pos hb]
      refine congrArg some ?_
      rw [hkT, kTileAFC]
      rw [show baseOffsetAFC s0 + e.val + (c * 64 + jL.val) * 128
            = baseOffsetAFC s0 + (c * 64 + jL.val) * 128 + e.val from by omega]
      show sin.readMem _ _ = s0.readMem K _
      unfold BlockState.readMem; rw [hmem']
    · rw [if_neg (by simp only [decide_eq_true_eq]; exact hb), if_neg hb]
      refine congrArg some ?_
      exact hundef' _ _
  -- masked q tile is already 0 on e ≥ 96
  have hqTmask : ∀ (ir e : Fin 128), (if e.val < 96 then qT (ir, e, PUnit.unit) else 0) = qT (ir, e, PUnit.unit) := by
    intro ir e
    by_cases he : e.val < 96
    · rw [if_pos he]
    · rw [if_neg he, hqT]; simp only [qTileAFCm]; rw [if_neg (fun h => he h.2)]
  -- jL window: c ≤ 1 so jL.val < 128 - i always
  have hjLwin : ∀ jL : Fin 64, jL.val < 128 - i := by
    intro jL; have := jL.isLt; omega
  -- the qk dot cell equals the afcKV score (per kept key)
  have hqkcell : ∀ (ir : Fin 128) (d : Fin 128) (jL : Fin 64),
      qkdotT.data (ir, jL, PUnit.unit)
        = some ((afcKV qT kT vT keyScale ir d ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1) := by
    intro ir d jL
    rw [hqkdotT]
    have hsc := afc_score_cell s0 qStart i jL ir qsc ksc qT
      (fun e jL => kT (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit))
      ⟨fun idx : TileIndex [128, 128] =>
        if qStart + idx.1.val < 128 ∧ idx.2.1.val < 96 then some (qTileAFC s0 Q idx) else some (0.0 : ℝ)⟩
      ktile kscT (hjLwin jL)
      (fun e => hqcell ir e) (fun e => hkcell e jL) hkscData
    rw [hsc]
    refine congrArg some ?_
    simp only [afcKV, hkeyBlock jL]
    rw [show (Finset.univ.sum (fun e : Fin 128 => (if e.val < 96 then qT (ir, e, PUnit.unit) else 0)
            * kT (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit)))
          = Finset.univ.sum (fun e : Fin 128 => qT (ir, e, PUnit.unit)
            * kT (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit))
        from Finset.sum_congr rfl (fun e _ => by rw [hqTmask ir e])]
    ring
  -- the causal mask cell
  have hmaskcell : ∀ (ir : Fin 128) (jL : Fin 64),
      maskT.data (ir, jL, PUnit.unit) = decide (qStart + ir.val ≥ c * 64 + jL.val) := by
    intro ir jL
    rw [hmaskT]; simp only [Tile.vec]
    rw [show (ComparableDType.nat.ge (qStart + ir.val) (i + jL.val))
          = decide (qStart + ir.val ≥ c * 64 + jL.val) from by
      rw [Bool.eq_iff_iff]; simp only [ComparableDType.nat_ge_eq_true, decide_eq_true_eq]
      omega]
  -- the sentinel cell: kept → afcKV score, else −1e6
  have hsentcell : ∀ (ir : Fin 128) (d : Fin 128) (jL : Fin 64),
      qkSentT.data (ir, jL, PUnit.unit)
        = if qStart + ir.val ≥ c * 64 + jL.val then
            (((afcKV qT kT vT keyScale ir d ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ) := by
    intro ir d jL
    rw [hqkSentT]; simp only [hmaskcell ir jL, decide_eq_true_eq]
    by_cases hk : qStart + ir.val ≥ c * 64 + jL.val
    · rw [if_pos hk, if_pos hk]
      exact hqkcell ir d jL
    · rw [if_neg hk, if_neg hk]
      exact afc_sentinel_eq
  -- assemble the new invariant
  simp only [afcInvariant, ← hqStart, ← hqT, ← hkT, ← hvT]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- pids
    rw [hpidsF, hpids']
  · -- (i + 64) % 64 = 0
    omega
  · -- i + 64 ≤ 128
    omega
  · -- m_i = afcRunningMax (i+64)
    rw [hFmi]; refine congrArg some ?_; refine Tile.ext (fun r => ?_)
    obtain ⟨ir, ⟨⟩⟩ := r
    have hbr := afc_mij_reg_eq_masked qT kT vT qStart keyScale c hc1 ir hsb
      qkSentT ⟨fun r : TileIndex [128] =>
        afcRunningMax qT kT vT keyScale qStart i r.1 ⟨0, by norm_num⟩⟩ rmaxT mijT
      (fun jL => by
        rw [show (TileShape.insertAxisIndex [128, 64] afcAx1 (ir, PUnit.unit) jL)
              = (ir, jL, PUnit.unit) from rfl]
        rw [hsentcell ir ⟨0, by norm_num⟩ jL])
      hrm (by simp only [hi]) hmijT
    rw [hbr, show ((c + 1) * 64 : Nat) = i + 64 from by omega]
  · -- l_i = afcStateBot1 (i+64) .2.1
    rw [hFli]; refine congrArg some ?_; refine Tile.ext (fun r => ?_)
    obtain ⟨ir, ⟨⟩⟩ := r
    have hmijcell : mijT.data (ir, PUnit.unit)
        = afcRunningMax qT kT vT keyScale qStart ((c + 1) * 64) ir ⟨0, by norm_num⟩ := by
      refine afc_mij_reg_eq_masked qT kT vT qStart keyScale c hc1 ir hsb
        qkSentT ⟨fun r : TileIndex [128] =>
          afcRunningMax qT kT vT keyScale qStart i r.1 ⟨0, by norm_num⟩⟩ rmaxT mijT
        (fun jL => by
          rw [show (TileShape.insertAxisIndex [128, 64] afcAx1 (ir, PUnit.unit) jL)
                = (ir, jL, PUnit.unit) from rfl]
          rw [hsentcell ir ⟨0, by norm_num⟩ jL])
        hrm (by simp only [hi]) hmijT
    have hbr := afc_denom_reg_eq_masked qT kT vT qStart keyScale c hc1 ir
      qkSentT ⟨fun r : TileIndex [128] => afcRunningMax qT kT vT keyScale qStart i r.1 ⟨0, by norm_num⟩⟩
      mijT alphaT ⟨fun r : TileIndex [128] => ((afcStateBot1 qT kT vT keyScale qStart i r.1 ⟨0, by norm_num⟩).2.1 : ℝ)⟩
      lijT pT
      (fun jL => by rw [hsentcell ir ⟨0, by norm_num⟩ jL])
      (by simp only [hi]; rfl) (by simp only [hi]) hmijcell
      halphaT hlijT
      (fun jL => by
        rw [hpT]; simp only [hmaskcell ir jL, decide_eq_true_eq])
    have hne : afcRunningMax qT kT vT keyScale qStart ((c + 1) * 64) ir ⟨0, by norm_num⟩ ≠ ⊥ :=
      afcRunningMax_ne_bot qT kT vT keyScale qStart ((c + 1) * 64) (by omega) ir ⟨0, by norm_num⟩
    show (Tile.bop NumericDType.real.add _ _ _).data _ = _
    rw [hbr, show ((i + 64) : Nat) = (c + 1) * 64 from by omega]
    exact congrArg (fun st : WithBot ℝ × ℝ × ℝ => (st.2.1 : WithBot ℝ))
      (afcStateBot1_eq_afcStateBot qT kT vT keyScale qStart ((c + 1) * 64) ir ⟨0, by norm_num⟩ hne).symm
  · -- acc = afcStateBot1 (i+64) .2.2
    rw [hFacc, hpf16]
    -- fp16 round-trip is identity at the algorithm layer
    simp only [FloatDType.cast, FloatDType.real_toWithBot, FloatDType.fp16_ofWithBot,
      FloatDType.fp16_toWithBot, FloatDType.real_ofWithBot]
    refine congrArg some ?_; refine Tile.ext (fun idx => ?_)
    obtain ⟨ir, id, ⟨⟩⟩ := idx
    have hmijcell : mijT.data (ir, PUnit.unit)
        = afcRunningMax qT kT vT keyScale qStart ((c + 1) * 64) ir ⟨0, by norm_num⟩ := by
      refine afc_mij_reg_eq_masked qT kT vT qStart keyScale c hc1 ir hsb
        qkSentT ⟨fun r : TileIndex [128] =>
          afcRunningMax qT kT vT keyScale qStart i r.1 ⟨0, by norm_num⟩⟩ rmaxT mijT
        (fun jL => by
          rw [show (TileShape.insertAxisIndex [128, 64] afcAx1 (ir, PUnit.unit) jL)
                = (ir, jL, PUnit.unit) from rfl]
          rw [hsentcell ir ⟨0, by norm_num⟩ jL])
        hrm (by simp only [hi]) hmijT
    -- masked v-load cell = vTileAFCm value = afcKV.2
    have hvload : ∀ jL : Fin 64,
        vtile.data (jL, id, PUnit.unit)
          = some ((afcKV qT kT vT keyScale ir id ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).2) := by
      intro jL
      rw [hvtile]; simp only [hvmaskT, Tile.vec, vPtrsAFC, Region.cast]
      rw [show (ComparableDType.nat.lt jL.val (128 - i) && ComparableDType.nat.lt id.val 96)
            = decide (jL.val < 128 - i ∧ id.val < 96) from by
        rw [Bool.eq_iff_iff]; simp only [Bool.and_eq_true, ComparableDType.nat_lt_eq_true,
          decide_eq_true_eq]]
      simp only [afcKV, hvT, vTileAFCm]
      by_cases hid : id.val < 96
      · rw [if_pos (by simp only [decide_eq_true_eq]; exact ⟨hjLwin jL, hid⟩), if_pos hid]
        refine congrArg some ?_
        show sin.readMem _ (baseOffsetAFC s0 + (c * 64 + jL.val) * 128 + id.val) = _
        rw [vTileAFC]
        unfold BlockState.readMem; rw [hmem']
      · rw [if_neg (by simp only [decide_eq_true_eq, not_and]; intro _ h; exact hid h), if_neg hid]
        exact congrArg some (hundef' _ _)
    have hbr := afc_acc_reg_eq_masked qT kT vT qStart keyScale c hc1 ir id
      qkSentT ⟨fun r : TileIndex [128] => afcRunningMax qT kT vT keyScale qStart i r.1 ⟨0, by norm_num⟩⟩
      mijT alphaT
      ⟨fun idx : TileIndex [128, 128] => ((afcStateBot1 qT kT vT keyScale qStart i idx.1 idx.2.1).2.2 : ℝ)⟩
      (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        ⟨fun idx : TileIndex [128, 128] => ((afcStateBot1 qT kT vT keyScale qStart i idx.1 idx.2.1).2.2 : ℝ)⟩
        (Tile.expandDim ⟨1, by simp⟩ alphaT))
      pT vtile (fun jL => (afcKV qT kT vT keyScale ir id ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩).2)
      (fun jL => by rw [hsentcell ir ⟨0, by norm_num⟩ jL])
      (by simp only [hi]; rfl) (by simp only [hi]) hmijcell halphaT rfl
      (fun jL => by
        rw [hpT]; simp only [hmaskcell ir jL, decide_eq_true_eq])
      hvload (fun jL => rfl)
    have hne : afcRunningMax qT kT vT keyScale qStart ((c + 1) * 64) ir id ≠ ⊥ := by
      rw [afcRunningMax_eq qT kT vT keyScale qStart ((c + 1) * 64) ir id ⟨0, by norm_num⟩]
      exact afcRunningMax_ne_bot qT kT vT keyScale qStart ((c + 1) * 64) (by omega) ir ⟨0, by norm_num⟩
    show (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) _
      (Tile.dot [] pT vtile)).data _ = _
    rw [hbr, show ((i + 64) : Nat) = (c + 1) * 64 from by omega]
    exact congrArg (fun st : WithBot ℝ × ℝ × ℝ => (st.2.2 : WithBot ℝ))
      (afcStateBot1_eq_afcStateBot qT kT vT keyScale qStart ((c + 1) * 64) ir id hne).symm
  · -- offs_m
    rw [hFoffsm]
  · -- offs_n
    rw [hFoffsn]
  · -- q
    rw [hFq]
  · -- q_scale
    rw [hFqsc]
  · -- K_ptrs (i/64 + 1 = (i+64)/64)
    rw [hFKp, kPtrsAFC_succ, show (i + 64) / 64 = c + 1 from by omega]
  · -- K_scale_ptr
    rw [hFKsp, kScalePtrAFC_succ, show (i + 64) / 64 = c + 1 from by omega]
  · -- V_ptrs
    rw [hFVp, vPtrsAFC_succ, show (i + 64) / 64 = c + 1 from by omega]
  · -- O_block_ptr (loop body does not assign it)
    rw [afcLoopBody_preserves_OblockPtr hchain, hsin,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hOp
  · -- undef
    intro rg o; rw [hundefF, hundef']
  · -- mem
    rw [hmemF, hmem']

/-- The running denominator `afcStateBot.2.1` is independent of the channel `d`
(it sums `pow2(score)` over the kept keys, and the score `afcKV.1` does not depend
on `d`). -/
theorem afcStateBot_snd_fst_indep
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d d' : Fin 128) :
    (afcStateBot qT kT vT keyScale qStart hi i d).2.1
      = (afcStateBot qT kT vT keyScale qStart hi i d').2.1 := by
  rw [afcStateBot_snd_fst, afcStateBot_snd_fst,
    afcRunningMax_eq qT kT vT keyScale qStart hi i d d']
  congr 2
  unfold afcKeysUpto
  rw [List.map_filterMap, List.map_filterMap]
  refine congrArg List.sum (List.filterMap_congr ?_)
  intro j _
  by_cases hj : j.val < hi ∧ j.val ≤ qStart + i.val <;> simp [afcKV, hj]

/-- Injectivity of the AFC `O_block_ptr` store addresses (Python shape): cell
`(i, e)` maps to `base + (p0·128 + i)·128 + e`, injective in `(i, e)`. -/
theorem afc_oBlockPtr_offset_injective (base p0 : Nat) :
    Function.Injective
      (fun idx : TileIndex [128, 128] => base + (p0 * 128 + idx.1.val) * 128 + idx.2.1.val) := by
  rintro ⟨⟨ma, hma⟩, ⟨ea, hea⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨eb, heb⟩, _⟩ h
  simp only at h
  have hm : ma = mb := by omega
  have he : ea = eb := by omega
  subst mb; subst eb; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **postLoop evaluation (AFC, causal).** From a loop-end state satisfying
`afcInvariant … 128`, the 2 postLoop statements (`acc = acc / l_i[:, None]` and the
masked `tl.store(O_block_ptr, acc, mask)`) write the genuine closed form
`attnFwdCausalOutSpec` to `Out` at every active output lane (`offs_m < 128 ∧
offs_k < 96`), and preserve `Out` on the inactive lanes. Mirrors
`aft3PostLoop_eval`; the masked elementwise pointer store readback goes through
`scatter_readback_prop_masked_nd`. -/
theorem afcPostLoop_eval
    (Q K V QScale KScale Out : RegionName) (s0 : BlockState) (s : BlockState)
    (keyScale : Fin 128 → ℝ)
    (hinv : afcInvariant Q K V QScale KScale Out s0 keyScale 128 s) :
    ∃ sP, stepStmts (afcPostLoop Out) s = some sP
      ∧ ∀ idx : TileIndex [128, 128],
          sP.readMem Out (outOffset s0 4 65536 16384 128 1 128 idx)
            = if active s0 128 96 128 idx then
                attnFwdCausalOutSpec s0 Q K V keyScale idx
              else s.readMem Out (outOffset s0 4 65536 16384 128 1 128 idx) := by
  simp only [afcInvariant] at hinv
  obtain ⟨hpids, _, _, _hmi, hli, hacc, hoffsm, _hoffsn,
    _hq, _hqs, _hKp, _hKsp, _hVp, hOp, hundef, hmem⟩ := hinv
  set qStart := qStartAFC s0 with hqStart
  set qT := qTileAFCm s0 Q with hqT
  set kT := kTileAFC s0 K with hkT
  set vT := vTileAFCm s0 V with hvT
  -- register tiles at loop end (window = 128)
  set liTile : Tile .real [128] :=
    ⟨fun r : TileIndex [128] => ((afcStateBot1 qT kT vT keyScale qStart 128 r.1 ⟨0, by norm_num⟩).2.1 : ℝ)⟩
    with hliTile
  set accTile : Tile .real [128, 128] :=
    ⟨fun idx : TileIndex [128, 128] => ((afcStateBot1 qT kT vT keyScale qStart 128 idx.1 idx.2.1).2.2 : ℝ)⟩
    with haccTile
  -- normalized acc tile
  set accFin : Tile .real [128, 128] :=
    Tile.bop NumericDType.real.div (Broadcast.consSame (Broadcast.consR Broadcast.nil)) accTile
      (Tile.expandDim ⟨1, by simp⟩ liTile) with haccFin
  unfold afcPostLoop
  -- stmt 23: acc = acc / l_i[:, None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [128, 128] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "l_i"))) s
        = some accFin from by
      have hexp : @evalOp TileDType.real [128, 1]
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.real [128] "l_i")) s
          = some (Tile.expandDim ⟨1, by simp⟩ liTile) :=
        evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hli
      rw [evalOp_div]
      simp only [evalOp_ref, hexp, hacc, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  set s2 := s.setReg "acc" .real [128, 128] accFin with hs2
  -- readbacks in s2 for the masked store
  have hOp2 : s2.regs .ptr [128, 128] "O_block_ptr" = some (oBlockPtrAFC s0 Out) := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hOp
  have hacc2 : s2.regs .real [128, 128] "acc" = some accFin := by
    rw [hs2, BlockState.setReg_same]
  have hoffsm2 : s2.regs .nat [128] "offs_m" = some (Tile.vec (fun r : Fin 128 => qStart + r.val)) := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffsm
  -- store offset / value / mask functions
  set oOffFn : TileIndex [128, 128] → Nat :=
    fun idx => baseOffsetAFC s0 + (s0.pids 0 * 128 + idx.1.val) * 128 + idx.2.1.val with hoOffFn
  set P : TileIndex [128, 128] → Prop :=
    fun idx => qStart + idx.1.val < 128 ∧ idx.2.1.val < 96 with hP
  -- the O_block_ptr ptr tile readback (per-lane (Out, oOffFn))
  have hopEval : evalOp (Op.ref .ptr [128, 128] "O_block_ptr") s2
      = some (⟨fun idx : TileIndex [128, 128] => (Out.cast, oOffFn idx)⟩ : Tile .ptr [128, 128]) := by
    rw [evalOp_ref, hOp2]
    refine congrArg some ?_; ext idx
    · rfl
    · simp only [oBlockPtrAFC, hoOffFn]
  -- the store mask readback (offs_m[:,None] < 128 ∧ arange < 96)
  have hmaskEval : evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) (Op.constNat 128))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat 96)))) s2
      = some (⟨fun idx : TileIndex [128, 128] => decide (P idx)⟩ : Tile .bool [128, 128]) := by
    rw [afc_evalOp_boolAnd, evalOp_lt]
    erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hoffsm2, evalOp_expandDim]
    simp only [evalOp_lt, evalOp_arange, evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
    refine congrArg some ?_; ext idx
    simp only [Tile.bop_data, Tile.cop_data, Tile.expandDim_data, Tile.vec_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
      Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil, TileShape.dropInsertedIndex]
    rw [show (ComparableDType.nat.lt (qStart + idx.1.val) 128 && ComparableDType.nat.lt idx.2.1.val 96)
          = decide (P idx) from by
      rw [Bool.eq_iff_iff]; simp only [hP, Bool.and_eq_true, ComparableDType.nat_lt_eq_true,
        decide_eq_true_eq]]
  -- stmt 24: masked store of acc to O_block_ptr
  have hstore : stepStmt (Stmt.store .real [128, 128] (.ptr (.ref .ptr [128, 128] "O_block_ptr"))
      (Op.ref .real [128, 128] "acc")
      (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) (Op.constNat 128))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat 96)))))) s2
      = some ((TileShape.allIndices [128, 128]).foldl
          (fun acc idx => if P idx then acc.writeMem Out (oOffFn idx) ((accFin.data idx).unbotD 0) else acc) s2) := by
    simp only [stepStmt, evalOp_ref, hacc2, hopEval, hmaskEval, Option.bind_eq_bind, Option.bind_some,
      Option.map_some, decide_eq_true_eq]
    refine congrArg some ?_
    refine List.foldl_ext _ _ s2 ?_
    intro acc idx _
    by_cases hk : P idx
    · simp only [if_pos hk, Region.cast_id, BlockState.writeMemTyped_real, FloatDType.real_storeValue]
    · simp only [if_neg hk]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  have hinjO : Function.Injective oOffFn := by
    rw [hoOffFn]; exact afc_oBlockPtr_offset_injective _ (s0.pids 0)
  have houtOff : outOffset s0 4 65536 16384 128 1 128 idx = oOffFn idx := by
    simp only [outOffset, offZ, offH, mIndex, kIndex, hoOffFn, baseOffsetAFC, Nat.mul_one]
  rw [houtOff]
  rw [BlockState.scatter_readback_prop_masked_nd _ oOffFn
    (fun idx => (accFin.data idx).unbotD 0) P hinjO idx]
  have hactiveP : active s0 128 96 128 idx ↔ P idx := by
    simp only [active, mIndex, kIndex, hP, hqStart, qStartAFC]
  by_cases hk : P idx
  · rw [if_pos hk, if_pos (hactiveP.mpr hk)]
    -- decode: accFin cell / liFin = StateBot ratio = spec
    obtain ⟨ir, id, ⟨⟩⟩ := idx
    have hne : afcRunningMax qT kT vT keyScale qStart 128 ir id ≠ ⊥ :=
      afcRunningMax_ne_bot qT kT vT keyScale qStart 128 (by norm_num) ir id
    have hne' : afcRunningMax qT kT vT keyScale qStart 128 ir ⟨0, by norm_num⟩ ≠ ⊥ :=
      afcRunningMax_ne_bot qT kT vT keyScale qStart 128 (by norm_num) ir ⟨0, by norm_num⟩
    simp only [haccFin, Tile.bop_data, Tile.expandDim_data, TileShape.dropInsertedIndex,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div, WithBot.realDiv,
      Option.map₂, Option.bind, Option.map, haccTile, hliTile, WithBot.unbotD_coe]
    rw [afcStateBot1_eq_afcStateBot qT kT vT keyScale qStart 128 ir id hne,
      afcStateBot1_eq_afcStateBot qT kT vT keyScale qStart 128 ir ⟨0, by norm_num⟩ hne']
    rw [afcStateBot_snd_fst_indep qT kT vT keyScale qStart 128 ir ⟨0, by norm_num⟩ id]
    rw [← afcStateBot_full_eq_spec s0 Q K V keyScale ir id (by rw [← hqStart, ← hqT, ← hkT, ← hvT]; exact hne)]
    simp only [hqStart, hqT, hkT, hvT, WithBot.unbotD_some]
  · rw [if_neg hk, if_neg (fun h => hk (hactiveP.mp h)), hs2]
    simp only [BlockState.readMem, BlockState.setReg_mem]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Full kernel execution (AFC, causal).** The lowered AFC surface body steps a
clean state (`undef = 0`, score-bounded) through preLoop + the `forRange` streaming
loop (via `forRange_inv` with `afcInvariant` as the loop invariant, advanced by
`afc_attn_step`) + postLoop, leaving the `Out` writeback at every active lane equal
to the genuine causal closed form `attnFwdCausalOutSpec`. Mirrors `aft3_attn_exec`. -/
theorem afc_attn_exec
    (Q K V QScale KScale Out : RegionName) (s : BlockState)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hsb : afcScoreBound (qTileAFCm s Q) (kTileAFC s K) (vTileAFCm s V)
      (keyScaleAFC s QScale KScale) (qStartAFC s)) :
    ∃ sF, stepStmts (attn_fwd_causal_surface Q K V QScale KScale Out
        65536 16384 128 1 65536 16384 128 1 65536 16384 128 1 65536 16384 128 1
        2 4 128 128 128 64 128 96 1).toAlgKernel.body s = some sF
      ∧ ∀ idx : TileIndex [128, 128],
          active s 128 96 128 idx →
            sF.readMem Out (outOffset s 4 65536 16384 128 1 128 idx)
              = attnFwdCausalOutSpec s Q K V (keyScaleAFC s QScale KScale) idx := by
  set keyScale := keyScaleAFC s QScale KScale with hkeyScale
  rw [afcBody_split, afcPreLoop_check, afcPostLoop_check]
  -- preLoop ⇒ invariant base case
  obtain ⟨sp, hpre, hinv0⟩ :=
    afcPreLoop_invariant s Q K V QScale KScale Out keyScale hundef
  rw [stepStmts.append_some hpre]
  -- the forRange streaming loop via forRange_inv with P = afcInvariant
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    forRange_inv (idx := "start_n") (start := 0) (stop := 128) (step := 64)
      (body := AfcFoundation.afcLoopBody)
      (P := fun i st => afcInvariant Q K V QScale KScale Out s keyScale i st)
      (s_init := sp)
      (by norm_num)
      hinv0
      (fun i st hi hP =>
        afc_attn_step Q K V QScale KScale Out s i st hi
          (by simp only [afcInvariant] at hP; exact hP.2.1) hsb hP)
  rw [stepStmts.cons_some hloop]
  -- at loop exit the counter is 128
  have hfinal : final = 128 := by
    obtain ⟨_, hmod, hle, _⟩ := hinvL
    omega
  subst hfinal
  -- postLoop
  obtain ⟨sF, hpost, hO⟩ := afcPostLoop_eval Q K V QScale KScale Out s sL keyScale hinvL
  refine ⟨sF, hpost, ?_⟩
  intro idx hact
  rw [hO idx, if_pos hact]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- Genuine AFC `Out`-store correctness: the masked `Out` writeback realizes the
closed-form causal attention ratio `attnFwdCausalOutSpec` at every active lane, on
clean (`undef = 0`) score-bounded input. -/
theorem attn_fwd_causal_surface_genuine_compute_correct
    (Q K V QScale KScale Out : RegionName) (s : BlockState)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hsb : afcScoreBound (qTileAFCm s Q) (kTileAFC s K) (vTileAFCm s V)
      (keyScaleAFC s QScale KScale) (qStartAFC s)) :
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
        attnFwdCausalOutSpec s Q K V (keyScaleAFC s QScale KScale) idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attn_fwd_causal_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  obtain ⟨sF, hstep, hO⟩ := afc_attn_exec Q K V QScale KScale Out s hundef hsb
  rw [exec] at hExec
  rw [hstep] at hExec
  obtain rfl : sF = s' := Option.some.inj hExec
  exact hO idx hActive

/-- **Python test-shape genuine output summary for `attn_fwd_causal.py`.**

The Python wrapper fixes `STAGE = 1`; this summary establishes that the full
causal surface lowers to the algorithm layer, and that its masked `Out` writeback
realizes the **genuine closed-form causal attention** `attnFwdCausalOutSpec`
(base-2 / `exp2`, per-key score scale, causal mask) at every active output lane —
no longer a self-referential "executed kernel output" carrier. Side conditions:
clean input (`undef = 0`) and the sentinel score bound `afcScoreBound` (every
causally-kept key's coerced score exceeds the `-1e6` masking sentinel), both
genuine preconditions of single-program correctness. -/
theorem attn_fwd_causal_python_test_shape_output_summary
    (Q K V QScale KScale Out : RegionName) (s : BlockState)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hsb : afcScoreBound (qTileAFCm s Q) (kTileAFC s K) (vTileAFCm s V)
      (keyScaleAFC s QScale KScale) (qStartAFC s)) :
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
        attnFwdCausalOutSpec s Q K V (keyScaleAFC s QScale KScale) idx) := by
  refine ⟨?_, ?_⟩
  · exact attn_fwd_causal_surface_toAlgorithm_supported Q K V QScale KScale
      Out 65536 16384 128 1 65536 16384 128 1 65536 16384 128 1
      65536 16384 128 1 2 4 128 128 128 64 128 96 1
  · exact attn_fwd_causal_surface_genuine_compute_correct Q K V QScale KScale Out s hundef hsb

/-! ## General (dimension-parameterized) genuine causal closed-form correctness

This section removes the Python test-shape pin (`B = 2`, `H = 4`,
`N_CTX = HEAD_DIM = BLOCK_M = BLOCK_DMODEL = 128`, `BLOCK_N = 64`,
`HEAD_ACTIVE = 96`, contiguous strides `(65536, 16384, 128, 1)`) and verifies the
genuine causal closed form `attentionRealBase2PerKeyScalePred ... (causalKeep)` at
the **dimension-parameterized** contiguous layout. -/

namespace AfcFoundation

open VeriTile.Triton

/-- General loop body (symbolic `BLOCK_M`/`BLOCK_N`/`BLOCK_DMODEL`/`HEAD_DIM`/
`N_CTX`/`HEAD_ACTIVE`). Mirrors `afcLoopBody` with the test-shape numerals replaced
by the corresponding dimension parameters. -/
def afcLoopBodyG (N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) : List Stmt :=
  [ -- 0: start_n = tl.multiple_of(start_n, BLOCK_N)  (identity)
    Stmt.assign .nat [] "start_n" (Op.ref .nat [] "start_n"),
    -- 1: k_mask
    Stmt.assign .bool [BLOCK_DMODEL, BLOCK_N] "k_mask"
      (Op.boolAnd (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat N_CTX) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))),
    -- 2: k = tl.load(K_ptrs, mask=k_mask)
    Stmt.assign .real [BLOCK_DMODEL, BLOCK_N] "k"
      (Op.load .real (.ptr (.ref .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs")) (.mask (.ref .bool [BLOCK_DMODEL, BLOCK_N] "k_mask"))),
    -- 3: k_scale = tl.load(K_scale_ptr)
    Stmt.assign .real [] "k_scale"
      (Op.load .real (.ptr (.ref .ptr [] "K_scale_ptr")) .none),
    -- 4: qk = castFloat(q·k) * q_scale * k_scale
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.mul .real Broadcast.scalarR
        (Op.mul .real Broadcast.scalarR
          (Op.castFloat FloatDType.real FloatDType.real
            (Op.dot (batch := []) (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "q") (Op.ref .real [BLOCK_DMODEL, BLOCK_N] "k")))
          (Op.ref .real [] "q_scale"))
        (Op.ref .real [] "k_scale")),
    -- 5: mask = offs_m[:,None] >= start_n + offs_n[None,:]
    Stmt.assign .bool [BLOCK_M, BLOCK_N] "mask"
      (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")))),
    -- 6: qk = tl.where(mask, qk, -1000000.0)
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.where (Op.ref .bool [BLOCK_M, BLOCK_N] "mask")
        (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
        (Op.broadcast (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1000000.0)) [BLOCK_M, BLOCK_N])),
    -- 7: m_ij = maximum(m_i, max(qk,1))
    Stmt.assign .real [BLOCK_M] "m_ij"
      (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BLOCK_M] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false
            (Op.ref .real [BLOCK_M, BLOCK_N] "qk")))
        (Op.ref .real [BLOCK_M] "m_i")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false
          (Op.ref .real [BLOCK_M, BLOCK_N] "qk"))),
    -- 8: qk = qk - m_ij[:, None]
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_N] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "m_ij"))),
    -- 9: p = exp2(qk)
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p" (Op.exp2 (Op.ref .real [BLOCK_M, BLOCK_N] "qk")),
    -- 10: p = tl.where(mask, p, 0)
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p"
      (Op.where (Op.ref .bool [BLOCK_M, BLOCK_N] "mask")
        (Op.ref .real [BLOCK_M, BLOCK_N] "p") (Op.broadcast (Op.const 0.0) [BLOCK_M, BLOCK_N])),
    -- 11: l_ij = sum(p, 1)
    Stmt.assign .real [BLOCK_M] "l_ij"
      (Op.reduceSum (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false (Op.ref .real [BLOCK_M, BLOCK_N] "p")),
    -- 12: alpha = exp2(m_i - m_ij)
    Stmt.assign .real [BLOCK_M] "alpha"
      (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BLOCK_M] "m_i") (Op.ref .real [BLOCK_M] "m_ij"))),
    -- 13: l_i = l_i * alpha + l_ij
    Stmt.assign .real [BLOCK_M] "l_i"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BLOCK_M] "l_i") (Op.ref .real [BLOCK_M] "alpha"))
        (Op.ref .real [BLOCK_M] "l_ij")),
    -- 14: acc = acc * alpha[:, None]
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "alpha"))),
    -- 15: v = tl.load(V_ptrs, mask=...)
    Stmt.assign .real [BLOCK_N, BLOCK_DMODEL] "v"
      (Op.load .real (.ptr (.ref .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"))
        (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
            (Op.sub .nat Broadcast.nil (Op.constNat N_CTX) (Op.ref .nat [] "start_n")))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))))),
    -- 16: p = p.to(fp16)
    Stmt.assign .fp16 [BLOCK_M, BLOCK_N] "p"
      (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BLOCK_M, BLOCK_N] "p")),
    -- 17: acc += dot(p.to(real), v)
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
        (Op.dot (batch := [])
          (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref .fp16 [BLOCK_M, BLOCK_N] "p"))
          (Op.ref .real [BLOCK_N, BLOCK_DMODEL] "v"))),
    -- 18: m_i = m_ij
    Stmt.assign .real [BLOCK_M] "m_i" (Op.ref .real [BLOCK_M] "m_ij"),
    -- 19: K_ptrs += BLOCK_N * HEAD_DIM
    Stmt.assign .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BLOCK_N) (Op.constNat HEAD_DIM))),
    -- 20: K_scale_ptr += 1
    Stmt.assign .ptr [] "K_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ref .ptr [] "K_scale_ptr") (Op.constNat 1)),
    -- 21: V_ptrs += BLOCK_N * HEAD_DIM
    Stmt.assign .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BLOCK_N) (Op.constNat HEAD_DIM))) ]

end AfcFoundation

section General

open VeriTile.Triton

/-! ### General spec layer (genuine causal closed form over symbolic dims) -/

/-- General per-plane base offset: `off_z·stride_qz + off_h·stride_qh` with
`off_z = pids1 / H`, `off_h = pids1 % H`. -/
def baseOffsetAFCG (s : BlockState) (stride_qz stride_qh H : Nat) : Nat :=
  s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh

/-- General query tile: row `i` = global query `pids0·BLOCK_M + i`, head lane `e`
(`stride_qm = HEAD_DIM`, `stride_qk = 1`). -/
noncomputable def qTileAFCG (s : BlockState) (Q : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL : Nat) :
    TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
  fun (i, e, _) => s.readMem Q (baseOffsetAFCG s stride_qz stride_qh H
    + (s.pids 0 * BLOCK_M + i.val) * HEAD_DIM + e.val)

/-- General key tile: row `j` (global key), head lane `e` (`K[base + j·HEAD_DIM + e]`). -/
noncomputable def kTileAFCG (s : BlockState) (K : RegionName)
    (stride_qz stride_qh H HEAD_DIM SEQ BLOCK_DMODEL : Nat) :
    TileIndex [SEQ, BLOCK_DMODEL] → ℝ :=
  fun (j, e, _) => s.readMem K (baseOffsetAFCG s stride_qz stride_qh H + j.val * HEAD_DIM + e.val)

/-- General value tile: row `j` (global key), head lane `d` (`V[base + j·HEAD_DIM + d]`). -/
noncomputable def vTileAFCG (s : BlockState) (V : RegionName)
    (stride_qz stride_qh H HEAD_DIM SEQ BLOCK_DMODEL : Nat) :
    TileIndex [SEQ, BLOCK_DMODEL] → ℝ :=
  fun (j, d, _) => s.readMem V (baseOffsetAFCG s stride_qz stride_qh H + j.val * HEAD_DIM + d.val)

/-- General global query row for output tile-row `i`. -/
def qStartAFCG (s : BlockState) (BLOCK_M : Nat) : Nat := s.pids 0 * BLOCK_M

/-- General masked query tile: head-active (`e < HEAD_ACTIVE`) and query-row
boundary (`qStart + i < N_CTX`) masking, mirroring the kernel's `q` load mask. -/
noncomputable def qTileAFCmG (s : BlockState) (Q : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE : Nat) :
    TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
  fun (i, e, u) =>
    if qStartAFCG s BLOCK_M + i.val < N_CTX ∧ e.val < HEAD_ACTIVE then
      qTileAFCG s Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL (i, e, u) else 0

/-- General masked value tile: head-active (`d < HEAD_ACTIVE`) masking. -/
noncomputable def vTileAFCmG (s : BlockState) (V : RegionName)
    (stride_qz stride_qh H HEAD_DIM SEQ BLOCK_DMODEL HEAD_ACTIVE : Nat) :
    TileIndex [SEQ, BLOCK_DMODEL] → ℝ :=
  fun (j, d, u) =>
    if d.val < HEAD_ACTIVE then
      vTileAFCG s V stride_qz stride_qh H HEAD_DIM SEQ BLOCK_DMODEL (j, d, u) else 0

/-- **General genuine closed form** (exp2, causal): predicate-masked base-2
per-key-scale attention with the `causalKeep qStart` mask, over the kernel's
actually-loaded masked q/v tiles. -/
noncomputable def attnFwdCausalOutSpecG
    (s : BlockState) (Q K V : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (keyScale : Fin (BLOCK_N * numKVBlocks) → ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  attentionRealBase2PerKeyScalePred
    (qTileAFCmG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
    (kTileAFCG s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
    (vTileAFCmG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
    keyScale (fun i j => causalKeep (qStartAFCG s BLOCK_M) i j) idx

/-- General streaming bridge: the closed form equals the `osStep` online-softmax
fold over the causal-masked key list. -/
theorem attnFwdCausalOutSpecG_eq_streaming
    (s : BlockState) (Q K V : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (keyScale : Fin (BLOCK_N * numKVBlocks) → ℝ)
    (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    attnFwdCausalOutSpecG s Q K V stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N
        BLOCK_DMODEL HEAD_ACTIVE numKVBlocks keyScale (i, d, PUnit.unit)
      = (let st := (attnKeyListPred
            (qTileAFCmG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
            (kTileAFCG s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
            (vTileAFCmG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
            keyScale (fun i j => causalKeep (qStartAFCG s BLOCK_M) i j) i d).foldl
              osStep (0, 0, 0)
         st.2.2 / st.2.1) := by
  simpa [attnFwdCausalOutSpecG] using
    VeriTile.Triton.attentionRealBase2PerKeyScalePred_eq_streaming
      (qTileAFCmG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
      (kTileAFCG s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
      (vTileAFCmG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
      keyScale (fun i j => causalKeep (qStartAFCG s BLOCK_M) i j) i d

/-! ### General ⊥-seed online-softmax foundation math

Mirrors the pinned `afcKV`/`afcKeysUpto`/`afcBlock`/`afcRunningMax`/`afcStateBot`/
`afcStateBot1` over symbolic `BLOCK_M`(query rows)/`BLOCK_DMODEL`(channels)/
`SEQ`(keys)/`BLOCK_N`(block stride). Reuses the dim-agnostic generic core
(`osStepBot`, `osStepBot_foldl_consistent`, `osStepBot_block_eq`,
`osStepBot_bot_seed_indep`, `afc_filterMap_window_split`, `afc_filterMap_foldr_sup`,
`afc_mem_le_foldr_sup`, `afc_filterMap_finRange_sum`, `afc_foldl_sup_bot_eq_foldr`). -/

variable {BLOCK_M BLOCK_DMODEL SEQ : Nat}

/-- General `(score, value)` pair the kernel streams for output `(i, d)` at key `j`. -/
noncomputable def afcKVG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) (j : Fin SEQ) : ℝ × ℝ :=
  (keyScale j * Finset.univ.sum (fun e : Fin BLOCK_DMODEL => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)),
   vT (j, d, PUnit.unit))

/-- General causal per-row key list over `[0, hi)`: keys `j < hi` with `j ≤ qStart + i`. -/
noncomputable def afcKeysUptoG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) : List (ℝ × ℝ) :=
  (List.finRange SEQ).filterMap (fun j : Fin SEQ =>
    if j.val < hi ∧ j.val ≤ qStart + i.val then
      some (afcKVG qT kT vT keyScale i d j)
    else none)

/-- General block-`c` per-row key list (`c·BLOCK_N ≤ j < (c+1)·BLOCK_N`, causal). -/
noncomputable def afcBlockG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) : List (ℝ × ℝ) :=
  (List.finRange SEQ).filterMap (fun j : Fin SEQ =>
    if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val then
      some (afcKVG qT kT vT keyScale i d j)
    else none)

/-- General ⊥-seeded running max of the streamed key prefix `[0, hi)`. -/
noncomputable def afcRunningMaxG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) : WithBot ℝ :=
  ((afcKeysUptoG qT kT vT keyScale qStart hi i d).map
    (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥

/-- General ⊥-seeded running `(max, denom, acc)` after streaming `[0, hi)` (seed-`0`). -/
noncomputable def afcStateBotG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) : WithBot ℝ × ℝ × ℝ :=
  (afcKeysUptoG qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 0, 0)

/-- General ⊥-seeded running state from the kernel's `l_i = 1` seed. -/
noncomputable def afcStateBot1G
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) : WithBot ℝ × ℝ × ℝ :=
  (afcKeysUptoG qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 1, 0)

variable (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
  (keyScale : Fin SEQ → ℝ)

/-- General: ⊥-seeded running max of `afcStateBotG` is `afcRunningMaxG`. -/
theorem afcStateBotG_fst_eq_runningMax (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    (afcStateBotG qT kT vT keyScale qStart hi i d).1
      = afcRunningMaxG qT kT vT keyScale qStart hi i d := by
  rw [afcStateBotG, afcStateBot_fst, afcRunningMaxG, afc_foldl_sup_bot_eq_foldr]

/-- General ⊥-seeded denominator. -/
theorem afcStateBotG_snd_fst (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    (afcStateBotG qT kT vT keyScale qStart hi i d).2.1
      = ((afcRunningMaxG qT kT vT keyScale qStart hi i d).elim 0 (fun r => pow2 (-r)))
        * (0 + ((afcKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1)).sum) := by
  rw [afcStateBotG]
  rw [(osStepBot_foldl_consistent (afcKeysUptoG qT kT vT keyScale qStart hi i d) ⊥ 0 0 0 0
    (by simp) (by simp) (by simp) (by simp)).1]
  rw [show ((afcKeysUptoG qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 0, 0)).1
        = afcRunningMaxG qT kT vT keyScale qStart hi i d from by
    rw [afcStateBot_fst, afcRunningMaxG, afc_foldl_sup_bot_eq_foldr]]

/-- General ⊥-seeded accumulator. -/
theorem afcStateBotG_snd_snd (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    (afcStateBotG qT kT vT keyScale qStart hi i d).2.2
      = ((afcRunningMaxG qT kT vT keyScale qStart hi i d).elim 0 (fun r => pow2 (-r)))
        * (0 + ((afcKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1 * p.2)).sum) := by
  rw [afcStateBotG]
  rw [(osStepBot_foldl_consistent (afcKeysUptoG qT kT vT keyScale qStart hi i d) ⊥ 0 0 0 0
    (by simp) (by simp) (by simp) (by simp)).2]
  rw [show ((afcKeysUptoG qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 0, 0)).1
        = afcRunningMaxG qT kT vT keyScale qStart hi i d from by
    rw [afcStateBot_fst, afcRunningMaxG, afc_foldl_sup_bot_eq_foldr]]

/-- General ⊥-seeded ratio (max factor cancels) on a nonempty window. -/
theorem afcStateBotG_ratio_eq (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hne : afcRunningMaxG qT kT vT keyScale qStart hi i d ≠ ⊥) :
    (afcStateBotG qT kT vT keyScale qStart hi i d).2.2
        / (afcStateBotG qT kT vT keyScale qStart hi i d).2.1
      = ((afcKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1 * p.2)).sum
        / ((afcKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1)).sum := by
  rw [afcStateBotG_snd_fst, afcStateBotG_snd_snd, zero_add, zero_add]
  cases hM : afcRunningMaxG qT kT vT keyScale qStart hi i d with
  | bot => exact absurd hM hne
  | coe r =>
    have hκ : ((r : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-r) := rfl
    rw [hκ]
    have hpos : pow2 (-r) ≠ 0 := ne_of_gt (pow2_pos _)
    rw [mul_div_mul_left _ _ hpos]

/-- General ⊥-seeded state at the empty window is `(⊥, 0, 0)`. -/
theorem afcStateBotG_zero (qStart : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    afcStateBotG qT kT vT keyScale qStart 0 i d = (⊥, 0, 0) := by
  unfold afcStateBotG afcKeysUptoG
  rw [show (List.finRange SEQ).filterMap
        (fun j : Fin SEQ => if j.val < 0 ∧ j.val ≤ qStart + i.val
          then some (afcKVG qT kT vT keyScale i d j) else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-- General ⊥-seeded running max at the empty window is `⊥`. -/
theorem afcRunningMaxG_zero (qStart : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    afcRunningMaxG qT kT vT keyScale qStart 0 i d = ⊥ := by
  unfold afcRunningMaxG afcKeysUptoG
  rw [show (List.finRange SEQ).filterMap
        (fun j : Fin SEQ => if j.val < 0 ∧ j.val ≤ qStart + i.val
          then some (afcKVG qT kT vT keyScale i d j) else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-- General window split (`hi = c·BLOCK_N`). -/
theorem afcKeysUptoG_succ (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    afcKeysUptoG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i d
      = afcKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i d
        ++ afcBlockG qT kT vT keyScale qStart BLOCK_N c i d := by
  unfold afcKeysUptoG afcBlockG
  rw [show (List.finRange SEQ).filterMap
        (fun j : Fin SEQ => if j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val
          then some (afcKVG qT kT vT keyScale i d j) else none)
      = (List.finRange SEQ).filterMap
        (fun j : Fin SEQ => if j.val ≤ qStart + i.val ∧ j.val < (c + 1) * BLOCK_N
          then some (afcKVG qT kT vT keyScale i d j) else none)
      from List.filterMap_congr (fun j _ => by simp only [and_comm])]
  rw [afc_filterMap_window_split (List.finRange SEQ) (List.pairwise_lt_finRange SEQ)
    (c * BLOCK_N) ((c + 1) * BLOCK_N) (fun j => j.val ≤ qStart + i.val)
    (fun j => afcKVG qT kT vT keyScale i d j) (by nlinarith [Nat.zero_le BLOCK_N])]
  refine congrArg₂ (· ++ ·) ?_ ?_
  · apply List.filterMap_congr; intro j _; simp only [and_comm]
  · apply List.filterMap_congr; intro j _
    by_cases h1 : c * BLOCK_N ≤ j.val <;> by_cases h2 : j.val < (c + 1) * BLOCK_N <;>
      by_cases h3 : j.val ≤ qStart + i.val <;> simp [h1, h2, h3, and_assoc]

/-- General one-block advance of `afcStateBotG`. -/
theorem afcStateBotG_succ (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    afcStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i d
      = (afcBlockG qT kT vT keyScale qStart BLOCK_N c i d).foldl osStepBot
          (afcStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d) := by
  unfold afcStateBotG
  rw [afcKeysUptoG_succ, List.foldl_append]

/-- General running max one-block advance. -/
theorem afcRunningMaxG_succ (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    afcRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i d
      = afcRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i d
        ⊔ ((afcBlockG qT kT vT keyScale qStart BLOCK_N c i d).map
            (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  unfold afcRunningMaxG
  rw [afcKeysUptoG_succ, List.map_append]
  induction (afcKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i d) with
  | nil => simp
  | cons a t ih => simp only [List.map_cons, List.foldr_cons, List.cons_append, ih, max_assoc]

/-- General `afcRunningMaxG` is channel-independent. -/
theorem afcRunningMaxG_eq (qStart hi : Nat) (i : Fin BLOCK_M) (d d' : Fin BLOCK_DMODEL) :
    afcRunningMaxG qT kT vT keyScale qStart hi i d
      = afcRunningMaxG qT kT vT keyScale qStart hi i d' := by
  unfold afcRunningMaxG afcKeysUptoG
  congr 1
  rw [List.map_filterMap, List.map_filterMap]
  apply List.filterMap_congr
  intro j _
  by_cases hj : j.val < hi ∧ j.val ≤ qStart + i.val <;> simp [afcKVG, hj]

/-- General `afcRunningMaxG` over a nonempty causal window ≠ ⊥ (key 0 always kept). -/
theorem afcRunningMaxG_ne_bot (qStart hi : Nat) (hhi : 1 ≤ hi) (hSEQ : 0 < SEQ)
    (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    afcRunningMaxG qT kT vT keyScale qStart hi i d ≠ ⊥ := by
  unfold afcRunningMaxG afcKeysUptoG
  set sc0 : ℝ := keyScale (⟨0, hSEQ⟩ : Fin SEQ) *
      Finset.univ.sum (fun e : Fin BLOCK_DMODEL => qT (i, e, PUnit.unit) *
        kT (⟨0, hSEQ⟩, e, PUnit.unit)) with hsc0
  have hmem : ((sc0 : ℝ) : WithBot ℝ) ∈
      ((List.finRange SEQ).filterMap (fun j : Fin SEQ =>
        if j.val < hi ∧ j.val ≤ qStart + i.val
        then some (afcKVG qT kT vT keyScale i d j) else none)).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ)) := by
    rw [List.mem_map]
    refine ⟨afcKVG qT kT vT keyScale i d ⟨0, hSEQ⟩, ?_, rfl⟩
    rw [List.mem_filterMap]
    refine ⟨⟨0, hSEQ⟩, List.mem_finRange _, ?_⟩
    rw [if_pos ⟨show (0:Nat) < hi from by omega, Nat.zero_le _⟩]
  have hle := afc_mem_le_foldr_sup _ _ hmem
  intro hbot
  exact absurd (le_bot_iff.mp (hbot ▸ hle)) WithBot.coe_ne_bot

/-- General seed-1 = seed-0 on nonempty windows. -/
theorem afcStateBot1G_eq_afcStateBotG (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hne : afcRunningMaxG qT kT vT keyScale qStart hi i d ≠ ⊥) :
    afcStateBot1G qT kT vT keyScale qStart hi i d
      = afcStateBotG qT kT vT keyScale qStart hi i d := by
  have hxs : afcKeysUptoG qT kT vT keyScale qStart hi i d ≠ [] := by
    intro h; apply hne; unfold afcRunningMaxG; rw [h]; rfl
  unfold afcStateBot1G afcStateBotG
  exact osStepBot_bot_seed_indep _ hxs 1 0 0 0

/-- General seed-1 state at the empty window is `(⊥, 1, 0)`. -/
theorem afcStateBot1G_zero (qStart : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    afcStateBot1G qT kT vT keyScale qStart 0 i d = (⊥, 1, 0) := by
  unfold afcStateBot1G afcKeysUptoG
  rw [show (List.finRange SEQ).filterMap
        (fun j : Fin SEQ => if j.val < 0 ∧ j.val ≤ qStart + i.val
          then some (afcKVG qT kT vT keyScale i d j) else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-- General seed-1 one-block advance. -/
theorem afcStateBot1G_succ (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    afcStateBot1G qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i d
      = (afcBlockG qT kT vT keyScale qStart BLOCK_N c i d).foldl osStepBot
          (afcStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i d) := by
  unfold afcStateBot1G
  rw [afcKeysUptoG_succ, List.foldl_append]

/-- General seed-1 running max is `afcRunningMaxG`. -/
theorem afcStateBot1G_fst_eq_runningMax (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    (afcStateBot1G qT kT vT keyScale qStart hi i d).1
      = afcRunningMaxG qT kT vT keyScale qStart hi i d := by
  rw [afcStateBot1G, afcStateBot_fst, afcRunningMaxG, afc_foldl_sup_bot_eq_foldr]

/-- General denominator channel-independence. -/
theorem afcStateBotG_snd_fst_indep (qStart hi : Nat) (i : Fin BLOCK_M) (d d' : Fin BLOCK_DMODEL) :
    (afcStateBotG qT kT vT keyScale qStart hi i d).2.1
      = (afcStateBotG qT kT vT keyScale qStart hi i d').2.1 := by
  rw [afcStateBotG_snd_fst, afcStateBotG_snd_fst,
    afcRunningMaxG_eq qT kT vT keyScale qStart hi i d d']
  congr 2
  unfold afcKeysUptoG
  rw [List.map_filterMap, List.map_filterMap]
  refine congrArg List.sum (List.filterMap_congr ?_)
  intro j _
  by_cases hj : j.val < hi ∧ j.val ≤ qStart + i.val <;> simp [afcKVG, hj]

/-- General full-window bridge: at `hi = SEQ`, the causal ⊥-seeded key list equals
the predicate-filtered `attnKeyListPred` with `causalKeep qStart`. -/
theorem afcKeysUptoG_full_eq_pred (qStart : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    afcKeysUptoG qT kT vT keyScale qStart SEQ i d
      = attnKeyListPred qT kT vT keyScale (fun a b => causalKeep qStart a b) i d := by
  unfold afcKeysUptoG attnKeyListPred afcKVG
  apply List.filterMap_congr
  intro j _
  have hjlt : j.val < SEQ := j.isLt
  have hiff : causalKeep qStart i j ↔ j.val ≤ qStart + i.val := by
    unfold causalKeep; omega
  by_cases hc : causalKeep qStart i j
  · rw [if_pos ⟨hjlt, hiff.mp hc⟩, if_pos hc]
  · rw [if_neg (fun hh => hc (hiff.mpr hh.2)), if_neg hc]

/-- General sentinel boundedness side-condition. -/
def afcScoreBoundG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart : Nat) : Prop :=
  ∀ (j : Fin SEQ) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL),
    keyScale j * Finset.univ.sum (fun e : Fin BLOCK_DMODEL => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit))
      > -1000000.0

/-- General: under `afcScoreBoundG`, running max over a nonempty causal window
exceeds the `-1e6` sentinel. -/
theorem afcRunningMaxG_gt_sentinel (qStart hi : Nat) (hhi : 1 ≤ hi) (hSEQ : 0 < SEQ)
    (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hsb : afcScoreBoundG qT kT vT keyScale qStart) :
    afcRunningMaxG qT kT vT keyScale qStart hi i d > some (-1000000.0 : ℝ) := by
  unfold afcRunningMaxG afcKeysUptoG
  have hkey0 : ((afcKVG qT kT vT keyScale i d ⟨0, hSEQ⟩).1 : ℝ) > -1000000.0 := by
    have := hsb ⟨0, hSEQ⟩ i d
    simpa [afcKVG] using this
  have hmem : ((((afcKVG qT kT vT keyScale i d ⟨0, hSEQ⟩).1 : ℝ)) : WithBot ℝ) ∈
      ((List.finRange SEQ).filterMap (fun j : Fin SEQ =>
        if j.val < hi ∧ j.val ≤ qStart + i.val
        then some (afcKVG qT kT vT keyScale i d j) else none)).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ)) := by
    rw [List.mem_map]
    refine ⟨afcKVG qT kT vT keyScale i d ⟨0, hSEQ⟩, ?_, rfl⟩
    rw [List.mem_filterMap]
    refine ⟨⟨0, hSEQ⟩, List.mem_finRange _, ?_⟩
    rw [if_pos ⟨show (0:Nat) < hi from by omega, Nat.zero_le _⟩]
  have hle := afc_mem_le_foldr_sup _ _ hmem
  refine lt_of_lt_of_le ?_ hle
  exact (WithBot.coe_lt_coe).mpr hkey0

/-- General `afcBlockG` map-and-sum: reindex block `c`'s causal window onto
`Fin BLOCK_N` lanes (global key `c·BLOCK_N + jL`). -/
theorem afcBlockG_map_sum (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hwin : (c + 1) * BLOCK_N ≤ SEQ) (h : ℝ × ℝ → ℝ) :
    ((afcBlockG qT kT vT keyScale qStart BLOCK_N c i d).map h).sum
      = ∑ jL : Fin BLOCK_N,
          (if (c * BLOCK_N + jL.val) ≤ qStart + i.val then
            h (afcKVG qT kT vT keyScale i d ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩)
           else 0) := by
  rw [afcBlockG, afc_filterMap_finRange_sum SEQ
    (fun j => c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val)
    (fun j => afcKVG qT kT vT keyScale i d j) h]
  rw [show (∑ j : Fin SEQ, if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val
            then h (afcKVG qT kT vT keyScale i d j) else 0)
        = ∑ j ∈ Finset.univ.filter (fun j : Fin SEQ => c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N),
            (if j.val ≤ qStart + i.val then h (afcKVG qT kT vT keyScale i d j) else 0) from by
    rw [Finset.sum_filter]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    by_cases hwj : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N
    · by_cases hcj : j.val ≤ qStart + i.val
      · rw [if_pos ⟨hwj.1, hwj.2, hcj⟩, if_pos hwj, if_pos hcj]
      · rw [if_neg (fun hh => hcj hh.2.2), if_pos hwj, if_neg hcj]
    · rw [if_neg (fun hh => hwj ⟨hh.1, hh.2.1⟩), if_neg hwj]]
  symm
  have hexp : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := by ring
  refine Finset.sum_bij
    (i := fun jL _ => (⟨c * BLOCK_N + jL.val, by have := jL.isLt; omega⟩ : Fin SEQ)) ?_ ?_ ?_ ?_
  · intro jL _
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    have := jL.isLt; omega
  · intro a _ b _ hab
    apply Fin.ext
    have : c * BLOCK_N + a.val = c * BLOCK_N + b.val := by simpa using congrArg Fin.val hab
    omega
  · intro j hj
    have hj2 : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N := (Finset.mem_filter.mp hj).2
    exact ⟨⟨j.val - c * BLOCK_N, by omega⟩, Finset.mem_univ _, by apply Fin.ext; simp only; omega⟩
  · intro jL _; rfl

/-- General `afcBlockG` running-sup bridge. -/
theorem afcBlockG_blockSup (qStart BLOCK_N c : Nat) (i d : Fin BLOCK_M) (d2 : Fin BLOCK_DMODEL)
    (hwin : (c + 1) * BLOCK_N ≤ SEQ) :
    ((afcBlockG qT kT vT keyScale qStart BLOCK_N c i d2).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = Finset.univ.sup (fun jL : Fin BLOCK_N =>
          if (c * BLOCK_N + jL.val) ≤ qStart + i.val then
            (((afcKVG qT kT vT keyScale i d2 ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else (⊥ : WithBot ℝ)) := by
  rw [show (afcBlockG qT kT vT keyScale qStart BLOCK_N c i d2).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
      = ((List.finRange SEQ).filterMap (fun j : Fin SEQ =>
          if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val
          then some ((afcKVG qT kT vT keyScale i d2 j).1) else none)).map
            (fun x => ((x : ℝ) : WithBot ℝ)) from by
    unfold afcBlockG
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr
    intro j _
    by_cases hj : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val <;> simp [hj]]
  rw [afc_filterMap_foldr_sup SEQ
    (fun j => c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val)
    (fun j => (afcKVG qT kT vT keyScale i d2 j).1)]
  have hexp : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := by ring
  apply le_antisymm
  · apply Finset.sup_le; intro j _
    by_cases hj : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val
    · rw [if_pos hj]
      have hjL : j.val - c * BLOCK_N < BLOCK_N := by omega
      have hfin : (⟨c * BLOCK_N + (j.val - c * BLOCK_N), by omega⟩ : Fin SEQ) = j := by apply Fin.ext; simp only; omega
      refine le_trans ?_ (Finset.le_sup (Finset.mem_univ (⟨j.val - c * BLOCK_N, hjL⟩ : Fin BLOCK_N)))
      simp only
      rw [if_pos (show c * BLOCK_N + (j.val - c * BLOCK_N) ≤ qStart + i.val from by have := hj.2.2; omega)]
      apply le_of_eq
      rw [hfin]
    · rw [if_neg hj]; exact bot_le
  · apply Finset.sup_le; intro jL _
    have hb : c * BLOCK_N + jL.val < SEQ := by have := jL.isLt; omega
    by_cases hkeep : c * BLOCK_N + jL.val ≤ qStart + i.val
    · rw [if_pos hkeep]
      refine le_trans ?_ (Finset.le_sup (Finset.mem_univ (⟨c * BLOCK_N + jL.val, hb⟩ : Fin SEQ)))
      simp only
      rw [if_pos (by have := jL.isLt; exact ⟨by omega, by omega, hkeep⟩)]
    · rw [if_neg hkeep]; exact bot_le

/-! ### General masked-block bridge layer ([BLOCK_M,BLOCK_N], -1e6 sentinel, HEAD_ACTIVE) -/

/-- General canonical axis-1 index of `[BLOCK_M, BLOCK_N]`. -/
abbrev afcAx1G (BLOCK_M BLOCK_N : Nat) : Fin [BLOCK_M, BLOCK_N].length := ⟨1, by simp⟩

/-- General `reduceMax` row. -/
theorem afc_reduceMaxDrop_rowG (BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (qk : Tile .real [BLOCK_M, BLOCK_N]) (rmaxT : Tile .real [BLOCK_M])
    (hrm : Tile.reduceMaxDrop (afcAx1G BLOCK_M BLOCK_N) qk = some rmaxT)
    (i : Fin BLOCK_M) (g : Fin BLOCK_N → WithBot ℝ)
    (hqk : ∀ jL : Fin BLOCK_N,
        qk.data (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (afcAx1G BLOCK_M BLOCK_N) (i, PUnit.unit) jL) = g jL) :
    rmaxT.data (i, PUnit.unit) = Finset.univ.sup g := by
  unfold Tile.reduceMaxDrop at hrm
  rw [dif_pos (show 0 < TileShape.axisDim [BLOCK_M, BLOCK_N] (afcAx1G BLOCK_M BLOCK_N) from hBN)] at hrm
  rw [← Option.some.inj hrm]
  simp only [Finset.sup'_eq_sup]
  exact Finset.sup_congr rfl (fun jL _ => hqk jL)

/-- General `q·k` score cell (HEAD_ACTIVE). -/
theorem afc_score_cellG (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat)
    (qStart SN : Nat) (jL : Fin BLOCK_N)
    (i : Fin BLOCK_M) (qsc ksc : ℝ)
    (qm : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kread : Fin BLOCK_DMODEL → Fin BLOCK_N → ℝ)
    (qtile : Tile .real [BLOCK_M, BLOCK_DMODEL]) (ktile : Tile .real [BLOCK_DMODEL, BLOCK_N])
    (kscT : Tile .real [])
    (hjLwin : jL.val < SN)
    (hqtile : ∀ e : Fin BLOCK_DMODEL, qtile.data (i, e, PUnit.unit)
        = some (if e.val < HEAD_ACTIVE then qm (i, e, PUnit.unit) else 0))
    (hktile : ∀ e : Fin BLOCK_DMODEL, ktile.data (e, jL, PUnit.unit)
        = some (if jL.val < SN ∧ e.val < HEAD_ACTIVE then kread e jL else 0))
    (hkscT : kscT.data PUnit.unit = some ksc) :
    (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          ⟨fun ix => (Tile.dot [] qtile ktile).data ix⟩
          (Tile.scalar (some qsc))) kscT).data (i, jL, PUnit.unit)
      = some ((Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
          (if e.val < HEAD_ACTIVE then qm (i, e, PUnit.unit) else 0) * kread e jL)) * qsc * ksc) := by
  have hdot : (Tile.dot [] qtile ktile).data (i, jL, PUnit.unit)
      = some (Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
          (if e.val < HEAD_ACTIVE then qm (i, e, PUnit.unit) else 0) * kread e jL)) := by
    rw [Tile.dot_nil_data]
    rw [show (@Finset.sum (Fin BLOCK_DMODEL) (WithBot ℝ) _ Finset.univ
          (fun e => Option.map₂ (· * ·) (qtile.data (i, e, PUnit.unit)) (ktile.data (e, jL, PUnit.unit))))
        = @Finset.sum (Fin BLOCK_DMODEL) (WithBot ℝ) _ Finset.univ
          (fun e => (some ((if e.val < HEAD_ACTIVE then qm (i, e, PUnit.unit) else 0) * kread e jL) : WithBot ℝ))
        from Finset.sum_congr rfl (fun e _ => by
          rw [hqtile e, hktile e]
          simp only [Option.map₂, Option.bind, Option.map]
          refine congrArg some ?_
          by_cases he : e.val < HEAD_ACTIVE
          · rw [if_pos he, if_pos ⟨hjLwin, he⟩]
          · rw [if_neg he, if_neg (by simp [he])]; ring)]
    rw [WithBot.sum_someTerm_eq_some]
  simp only [Tile.bop_data, Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
    Tile.scalar, NumericDType.mul, hdot, hkscT]
  simp only [WithBot.realMul, Option.map₂, Option.bind, Option.map]

set_option maxHeartbeats 1600000 in
/-- General `m_ij = afcRunningMaxG((c+1)·BLOCK_N)` (masked with -1e6 sentinel). -/
theorem afc_mij_reg_eq_maskedG (BLOCK_N : Nat) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M)
    (hsb : afcScoreBoundG qT kT vT keyScale qStart)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mtile rmaxT mijT : Tile .real [BLOCK_M])
    (hsent : ∀ jL : Fin BLOCK_N, qkSentT.data (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (afcAx1G BLOCK_M BLOCK_N) (i, PUnit.unit) jL)
        = if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((afcKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hrmax : Tile.reduceMaxDrop (afcAx1G BLOCK_M BLOCK_N) qkSentT = some rmaxT)
    (hmtile : mtile.data (i, PUnit.unit)
        = afcRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩)
    (hmij : mijT = Tile.select
        (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT) :
    mijT.data (i, PUnit.unit)
      = afcRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ := by
  have hSEQ : 0 < SEQ := by
    have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega
  have hrmaxcell : rmaxT.data (i, PUnit.unit)
      = Finset.univ.sup (fun jL : Fin BLOCK_N =>
          if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((afcKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ)) :=
    afc_reduceMaxDrop_rowG BLOCK_M BLOCK_N hBN qkSentT rmaxT hrmax i _ hsent
  rw [afcRunningMaxG_succ qT kT vT keyScale qStart BLOCK_N c i ⟨0, hBD⟩]
  rw [afcBlockG_blockSup qT kT vT keyScale qStart BLOCK_N c i i ⟨0, hBD⟩ hc1]
  set BSk : WithBot ℝ := Finset.univ.sup (fun jL : Fin BLOCK_N =>
      if (c * BLOCK_N + jL.val) ≤ qStart + i.val then
        (((afcKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
      else (⊥ : WithBot ℝ)) with hBSk
  set M := afcRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩ with hMdef
  have hBSk_le : BSk ≤ rmaxT.data (i, PUnit.unit) := by
    rw [hrmaxcell, hBSk]
    apply Finset.sup_le; intro jL _
    split
    · rename_i hk
      refine Finset.le_sup_of_le (Finset.mem_univ jL) ?_
      rw [if_pos (show qStart + i.val ≥ c * BLOCK_N + jL.val from by omega)]
    · exact bot_le
  have hr_le : rmaxT.data (i, PUnit.unit) ≤ BSk ⊔ ((-1000000.0 : ℝ) : WithBot ℝ) := by
    rw [hrmaxcell, hBSk]
    apply Finset.sup_le; intro jL _
    split
    · rename_i hk
      refine le_sup_of_le_left (Finset.le_sup_of_le (Finset.mem_univ jL) ?_)
      rw [if_pos (show c * BLOCK_N + jL.val ≤ qStart + i.val from by omega)]
    · exact le_sup_right
  have hMBSk_sentinel : ((-1000000.0 : ℝ) : WithBot ℝ) ≤ M ⊔ BSk := by
    by_cases hc0 : c = 0
    · subst hc0
      refine le_sup_of_le_right ?_
      rw [hBSk]
      refine Finset.le_sup_of_le (Finset.mem_univ (⟨0, hBN⟩ : Fin BLOCK_N)) ?_
      rw [if_pos (show 0 * BLOCK_N + (⟨0, hBN⟩ : Fin BLOCK_N).val ≤ qStart + i.val from by simp)]
      refine le_of_lt ?_
      rw [WithBot.coe_lt_coe]
      have hbnd := hsb ⟨0 * BLOCK_N + (⟨0, hBN⟩ : Fin BLOCK_N).val, by simp only [Nat.zero_mul, Nat.add_zero]; exact hSEQ⟩ i ⟨0, hBD⟩
      simpa [afcKVG] using hbnd
    · refine le_sup_of_le_left (le_of_lt ?_)
      rw [hMdef]
      exact afcRunningMaxG_gt_sentinel qT kT vT keyScale qStart (c * BLOCK_N) (by
        have : 1 ≤ c := by omega
        calc 1 ≤ c := this
          _ ≤ c * BLOCK_N := Nat.le_mul_of_pos_right c hBN) hSEQ i ⟨0, hBD⟩ hsb
  have hdom : M ⊔ (BSk ⊔ ((-1000000.0 : ℝ) : WithBot ℝ)) = M ⊔ BSk := by
    rw [← sup_assoc, sup_eq_left.mpr hMBSk_sentinel]
  rw [hmij, Tile.select_data, Tile.cop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hmtile]
  have hsqueeze : M ⊔ rmaxT.data (i, PUnit.unit) = M ⊔ BSk := by
    apply le_antisymm
    · exact sup_le_sup_left (le_trans hr_le (le_of_eq rfl)) M |>.trans (le_of_eq hdom) |>.trans (le_refl _)
    · exact sup_le_sup_left hBSk_le M
  by_cases hcmp : M > rmaxT.data (i, PUnit.unit)
  · rw [if_pos (by simpa using hcmp)]
    rw [← hsqueeze, max_eq_left (le_of_lt hcmp)]
  · rw [if_neg (by simpa using hcmp)]
    rw [← hsqueeze, max_eq_right (not_lt.mp hcmp)]

set_option maxHeartbeats 1600000 in
/-- General masked `pT` cell. -/
theorem afc_pmT_cell_maskedG (BLOCK_N : Nat) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M) (jL : Fin BLOCK_N) (Mc1 : WithBot ℝ)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mijT : Tile .real [BLOCK_M]) (pT : Tile .real [BLOCK_M, BLOCK_N])
    (kept : Bool)
    (_hkept : kept = decide (qStart + i.val ≥ c * BLOCK_N + jL.val))
    (hsent : qkSentT.data (i, jL, PUnit.unit)
        = if kept then
            (((afcKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hmij : mijT.data (i, PUnit.unit) = Mc1)
    (hkeptbot : kept = Bool.true → Mc1 ≠ ⊥)
    (hpT : pT.data (i, jL, PUnit.unit)
        = if kept then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ)) :
    pT.data (i, jL, PUnit.unit)
      = some (if kept then
          pow2 ((afcKVG qT kT vT keyScale i ⟨0, hBD⟩
            ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 - Mc1.unbotD 0)
          else 0) := by
  rw [hpT]
  by_cases hk : kept = Bool.true
  · rw [if_pos hk]
    obtain ⟨Mr, hMr⟩ : ∃ Mr : ℝ, Mc1 = (Mr : WithBot ℝ) := by
      cases hh : Mc1 with
      | coe x => exact ⟨x, rfl⟩
      | bot => exact absurd hh (hkeptbot hk)
    simp only [Tile.uop_data, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.expandDim_data, TileShape.dropInsertedIndex, NumericDType.sub, hmij, hMr,
      WithBot.unbotD_coe]
    rw [show qkSentT.data (i, jL, PUnit.unit) =
          (((afcKVG qT kT vT keyScale i ⟨0, hBD⟩
              ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ) from by
      rw [hsent, if_pos hk]]
    rw [if_pos hk]
    simp only [WithBot.realSub, Option.map₂, Option.bind, Option.map, WithBot.realExp2_some]
    refine congrArg some ?_
    simp only [pow2]; ring_nf
  · rw [if_neg hk, if_neg hk]
    norm_num

set_option maxHeartbeats 1600000 in
/-- General masked `p` reduceSum row. -/
theorem afc_nume_row_sum_maskedG (BLOCK_N : Nat) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hsb : afcScoreBoundG qT kT vT keyScale qStart)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mijT : Tile .real [BLOCK_M]) (pT : Tile .real [BLOCK_M, BLOCK_N])
    (hsent : ∀ jL : Fin BLOCK_N, qkSentT.data (i, jL, PUnit.unit)
        = if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((afcKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hmij : mijT.data (i, PUnit.unit)
        = afcRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩)
    (hpT : ∀ jL : Fin BLOCK_N, pT.data (i, jL, PUnit.unit)
        = if (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ)) :
    (Tile.reduceSumDrop (afcAx1G BLOCK_M BLOCK_N) pT).data (i, PUnit.unit)
      = some ((afcBlockG qT kT vT keyScale qStart BLOCK_N c i d).map
          (fun p => pow2 (p.1 - (afcRunningMaxG qT kT vT keyScale
            qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩).unbotD 0))).sum := by
  have hc1' : 1 ≤ (c + 1) * BLOCK_N := by
    have : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega
  set Mc1 := afcRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ with hMc1
  have hMc1bot : Mc1 ≠ ⊥ := by
    rw [hMc1]; exact afcRunningMaxG_ne_bot qT kT vT keyScale qStart ((c + 1) * BLOCK_N) hc1' (by omega) i ⟨0, hBD⟩
  rw [Tile.reduceSumDrop_data]
  have hcell : ∀ jL : Fin BLOCK_N,
      pT.data (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (afcAx1G BLOCK_M BLOCK_N) (i, PUnit.unit) jL)
        = some (if decide (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            pow2 ((afcKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 - Mc1.unbotD 0)
            else 0) := by
    intro jL
    rw [show (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (afcAx1G BLOCK_M BLOCK_N) (i, PUnit.unit) jL) = (i, jL, PUnit.unit) from rfl]
    have hsentR : qkSentT.data (i, jL, PUnit.unit)
        = if decide (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (((afcKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ) := by
      rw [hsent jL]; by_cases h : qStart + i.val ≥ c * BLOCK_N + jL.val <;> simp [h]
    have hpTR : pT.data (i, jL, PUnit.unit)
        = if decide (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ) := by
      rw [hpT jL]; by_cases h : qStart + i.val ≥ c * BLOCK_N + jL.val
      · rw [if_pos h, if_pos (decide_eq_true h)]
      · rw [if_neg h, if_neg (by simp [h])]
    exact afc_pmT_cell_maskedG qT kT vT keyScale BLOCK_N hBD qStart c hc1 i jL Mc1 qkSentT mijT pT
      (decide (qStart + i.val ≥ c * BLOCK_N + jL.val)) (by rfl) hsentR
      (by rw [hmij]) (fun _ => hMc1bot) hpTR
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [afcBlockG_map_sum qT kT vT keyScale qStart BLOCK_N c i d hc1
      (fun p => pow2 (p.1 - Mc1.unbotD 0))]
  refine Finset.sum_congr rfl (fun jL _ => ?_)
  simp only [decide_eq_true_eq]
  by_cases hkp : (c * BLOCK_N + jL.val) ≤ qStart + i.val
  · rw [if_pos (show qStart + i.val ≥ c * BLOCK_N + jL.val from by omega),
        if_pos (show c * BLOCK_N + jL.val ≤ qStart + i.val from hkp)]
    rfl
  · rw [if_neg (show ¬ qStart + i.val ≥ c * BLOCK_N + jL.val from by omega),
        if_neg (show ¬ c * BLOCK_N + jL.val ≤ qStart + i.val from hkp)]

set_option maxHeartbeats 1600000 in
/-- General masked `dot(p, v)` numerator row. -/
theorem afc_acc_dot_block_maskedG (BLOCK_N : Nat) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mijT : Tile .real [BLOCK_M]) (pT : Tile .real [BLOCK_M, BLOCK_N])
    (vtile : Tile .real [BLOCK_N, BLOCK_DMODEL]) (vval : Fin BLOCK_N → ℝ)
    (hMc1bot : afcRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ ≠ ⊥)
    (hsent : ∀ jL : Fin BLOCK_N, qkSentT.data (i, jL, PUnit.unit)
        = if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((afcKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hmij : mijT.data (i, PUnit.unit)
        = afcRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩)
    (hpT : ∀ jL : Fin BLOCK_N, pT.data (i, jL, PUnit.unit)
        = if (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ))
    (hv : ∀ jL : Fin BLOCK_N, vtile.data (jL, d, PUnit.unit) = some (vval jL))
    (hvval : ∀ jL : Fin BLOCK_N, vval jL
        = (afcKVG qT kT vT keyScale i d ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).2) :
    (Tile.dot [] pT vtile).data (i, d, PUnit.unit)
      = some ((afcBlockG qT kT vT keyScale qStart BLOCK_N c i d).map
          (fun p => pow2 (p.1 - (afcRunningMaxG qT kT vT keyScale
            qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩).unbotD 0) * p.2)).sum := by
  set Mc1 := afcRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ with hMc1
  have hpcell : ∀ jL : Fin BLOCK_N, pT.data (i, jL, PUnit.unit)
      = some (if decide (qStart + i.val ≥ c * BLOCK_N + jL.val) then
          pow2 ((afcKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 - Mc1.unbotD 0)
          else 0) := by
    intro jL
    have hsentR : qkSentT.data (i, jL, PUnit.unit)
        = if decide (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (((afcKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ) := by
      rw [hsent jL]; by_cases h : qStart + i.val ≥ c * BLOCK_N + jL.val <;> simp [h]
    have hpTR : pT.data (i, jL, PUnit.unit)
        = if decide (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ) := by
      rw [hpT jL]; by_cases h : qStart + i.val ≥ c * BLOCK_N + jL.val
      · rw [if_pos h, if_pos (decide_eq_true h)]
      · rw [if_neg h, if_neg (by simp [h])]
    have := afc_pmT_cell_maskedG qT kT vT keyScale BLOCK_N hBD qStart c hc1 i jL Mc1 qkSentT mijT pT
      (decide (qStart + i.val ≥ c * BLOCK_N + jL.val)) (by rfl) hsentR (by rw [hmij]) (fun _ => hMc1bot) hpTR
    rw [this]
  rw [Tile.dot_nil_data]
  have hterm : ∀ jL : Fin BLOCK_N,
      Option.map₂ (· * ·) (pT.data (i, jL, PUnit.unit)) (vtile.data (jL, d, PUnit.unit))
        = some (if (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            pow2 ((afcKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 - Mc1.unbotD 0)
              * vval jL
            else 0) := by
    intro jL
    rw [hpcell jL, hv jL]
    simp only [Option.map₂, Option.bind, Option.map, decide_eq_true_eq]
    refine congrArg some ?_
    by_cases h : qStart + i.val ≥ c * BLOCK_N + jL.val
    · rw [if_pos h, if_pos h]
    · rw [if_neg h, if_neg h]; ring
  rw [show (@Finset.sum (Fin BLOCK_N) (WithBot ℝ) _ Finset.univ
        (fun jL => Option.map₂ (· * ·) (pT.data (i, jL, PUnit.unit)) (vtile.data (jL, d, PUnit.unit))))
      = @Finset.sum (Fin BLOCK_N) (WithBot ℝ) _ Finset.univ (fun jL =>
          (some (if (qStart + i.val ≥ c * BLOCK_N + jL.val) then
              pow2 ((afcKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 - Mc1.unbotD 0)
                * vval jL else 0) : WithBot ℝ))
      from Finset.sum_congr rfl (fun jL _ => hterm jL)]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [afcBlockG_map_sum qT kT vT keyScale qStart BLOCK_N c i d hc1
      (fun p => pow2 (p.1 - Mc1.unbotD 0) * p.2)]
  refine Finset.sum_congr rfl (fun jL _ => ?_)
  by_cases hkp : (c * BLOCK_N + jL.val) ≤ qStart + i.val
  · rw [if_pos (show qStart + i.val ≥ c * BLOCK_N + jL.val from by omega),
        if_pos (show c * BLOCK_N + jL.val ≤ qStart + i.val from hkp)]
    rw [hvval jL,
      show (afcKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1
          = (afcKVG qT kT vT keyScale i d ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 from by
        simp only [afcKVG]]
  · rw [if_neg (show ¬ qStart + i.val ≥ c * BLOCK_N + jL.val from by omega),
        if_neg (show ¬ c * BLOCK_N + jL.val ≤ qStart + i.val from hkp)]

/-- General seed-1 vs seed-0 carry cancellation. -/
theorem afcStateBot1G_cancel (qStart BLOCK_N c : Nat) (hBN : 0 < BLOCK_N) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) (Mc1 : WithBot ℝ)
    (hSEQ : 0 < SEQ) :
    let m := (afcStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).1
    let α := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
    (afcStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i d).2.1 * α
        = (afcStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.1 * α
      ∧ (afcStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2 * α
        = (afcStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2 * α := by
  intro m α
  by_cases hc0 : c = 0
  · subst hc0
    have hmbot : m = ⊥ := by
      show (afcStateBotG qT kT vT keyScale qStart (0 * BLOCK_N) i d).1 = ⊥
      rw [afcStateBotG_fst_eq_runningMax, Nat.zero_mul, afcRunningMaxG_zero]
    have hα0 : α = 0 := by
      show (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 = 0
      rw [hmbot, WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
    rw [hα0]; simp
  · have hne : afcRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i d ≠ ⊥ :=
      afcRunningMaxG_ne_bot qT kT vT keyScale qStart (c * BLOCK_N)
        (by calc 1 ≤ c := by omega
              _ ≤ c * BLOCK_N := Nat.le_mul_of_pos_right c hBN) hSEQ i d
    rw [afcStateBot1G_eq_afcStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d hne]
    exact ⟨rfl, rfl⟩

/-- General empty-window sum-zero. -/
theorem afcKeysUptoG_sum_zero_of_bot (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hbot : afcRunningMaxG qT kT vT keyScale qStart hi i d = ⊥) (h : ℝ × ℝ → ℝ) :
    ((afcKeysUptoG qT kT vT keyScale qStart hi i d).map h).sum = 0 := by
  rw [show afcKeysUptoG qT kT vT keyScale qStart hi i d = [] from ?_, List.map_nil, List.sum_nil]
  by_contra hne
  obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ hne
  have hmem : ((p.1 : ℝ) : WithBot ℝ) ∈
      (afcKeysUptoG qT kT vT keyScale qStart hi i d).map (fun q => ((q.1 : ℝ) : WithBot ℝ)) :=
    List.mem_map_of_mem hp
  have := afc_mem_le_foldr_sup _ _ hmem
  rw [← afcRunningMaxG, hbot] at this
  exact absurd (le_bot_iff.mp this) WithBot.coe_ne_bot

/-- General denom anchor. -/
theorem afc_denom_anchorG (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    (afcStateBotG qT kT vT keyScale qStart hi i d).2.1
      = ((afcStateBotG qT kT vT keyScale qStart hi i d).1.elim 0 (fun r => pow2 (-r)))
        * (0 + ((afcKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1)).sum) := by
  rw [afcStateBotG_snd_fst, afcStateBotG_fst_eq_runningMax]

/-- General acc anchor. -/
theorem afc_acc_anchorG (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    (afcStateBotG qT kT vT keyScale qStart hi i d).2.2
      = ((afcStateBotG qT kT vT keyScale qStart hi i d).1.elim 0 (fun r => pow2 (-r)))
        * (0 + ((afcKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1 * p.2)).sum) := by
  rw [afcStateBotG_snd_snd, afcStateBotG_fst_eq_runningMax]

set_option maxHeartbeats 1600000 in
/-- General `l_i' = afcStateBotG((c+1)·BLOCK_N).2.1` (masked). -/
theorem afc_denom_reg_eq_maskedG (BLOCK_N : Nat) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M)
    (hsb : afcScoreBoundG qT kT vT keyScale qStart)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mtile mijT alphaT litile lijT : Tile .real [BLOCK_M])
    (pT : Tile .real [BLOCK_M, BLOCK_N])
    (hsent : ∀ jL : Fin BLOCK_N, qkSentT.data (i, jL, PUnit.unit)
        = if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((afcKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hlitile : litile.data (i, PUnit.unit) = some
        ((afcStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.1))
    (hmtile : mtile.data (i, PUnit.unit)
        = afcRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩)
    (hmij : mijT.data (i, PUnit.unit)
        = afcRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩)
    (halpha : alphaT = Tile.uop WithBot.realExp2
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
    (hlijT : lijT = Tile.reduceSumDrop (afcAx1G BLOCK_M BLOCK_N) pT)
    (hpT : ∀ jL : Fin BLOCK_N, pT.data (i, jL, PUnit.unit)
        = if (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ)) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
        (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) litile alphaT) lijT).data (i, PUnit.unit)
      = some ((afcStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩).2.1) := by
  have hSEQ : 0 < SEQ := by
    have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega
  set m := (afcStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).1 with hm_def
  set Mc := afcRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩ with hMc
  set Mc1 := afcRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ with hMc1
  have hmMc : m = Mc := by rw [hm_def, hMc, afcStateBotG_fst_eq_runningMax]
  have hMsucc : Mc1 = m ⊔ ((afcBlockG qT kT vT keyScale qStart BLOCK_N c i ⟨0, hBD⟩).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    rw [hMc1, afcRunningMaxG_succ, hmMc, ← hMc]
  have halphaVal : alphaT.data (i, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [halpha]; show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmtile, hmij,
      NumericDType.sub, ← hMc, ← hMc1, hmMc]
  have hsum := afc_nume_row_sum_maskedG qT kT vT keyScale BLOCK_N hBN hBM hBD qStart c hc1 i ⟨0, hBD⟩ hsb qkSentT mijT pT hsent hmij hpT
  have hblockEq := osStepBot_block_eq m
    ((afcStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.1)
    ((afcStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.2)
    ((afcKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).map (fun p => pow2 p.1 * p.2)).sum
    ((afcKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).map (fun p => pow2 p.1)).sum
    (afcBlockG qT kT vT keyScale qStart BLOCK_N c i ⟨0, hBD⟩)
    (by rw [afc_denom_anchorG, zero_add, hm_def])
    (by rw [afc_acc_anchorG, zero_add, hm_def])
    (fun hbot => afcKeysUptoG_sum_zero_of_bot qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩
      (by rw [← afcStateBotG_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => afcKeysUptoG_sum_zero_of_bot qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩
      (by rw [← afcStateBotG_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  rw [show (afcStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩).2.1
        = (Mc1, (afcStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.1
              * (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
              + ((afcBlockG qT kT vT keyScale qStart BLOCK_N c i ⟨0, hBD⟩).map (fun p => pow2 (p.1 - Mc1.unbotD 0))).sum,
            _).2.1 from by
    rw [afcStateBotG_succ]; rw [← hblockEq]]
  set α : ℝ := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp2 (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  have hcancel := (afcStateBot1G_cancel qT kT vT keyScale qStart BLOCK_N c hBN i ⟨0, hBD⟩ Mc1 hSEQ).1
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex]
  rw [hlijT]
  erw [hsum]
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    hlitile, halphaVal, hαsome]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [show (afcStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.1 * α
        = (afcStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.1 * α from by
    have := hcancel; simp only [← hm_def, ← hαdef] at this ⊢; exact this]

set_option maxHeartbeats 1600000 in
/-- General `acc' = afcStateBotG((c+1)·BLOCK_N).2.2` (masked). -/
theorem afc_acc_reg_eq_maskedG (BLOCK_N : Nat) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hsb : afcScoreBoundG qT kT vT keyScale qStart)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mtile mijT alphaT : Tile .real [BLOCK_M])
    (acctile acc1T : Tile .real [BLOCK_M, BLOCK_DMODEL]) (pT : Tile .real [BLOCK_M, BLOCK_N])
    (vtile : Tile .real [BLOCK_N, BLOCK_DMODEL]) (vval : Fin BLOCK_N → ℝ)
    (hsent : ∀ jL : Fin BLOCK_N, qkSentT.data (i, jL, PUnit.unit)
        = if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((afcKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hacctile : acctile.data (i, d, PUnit.unit) = some
        ((afcStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2))
    (hmtile : mtile.data (i, PUnit.unit)
        = afcRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩)
    (hmij : mijT.data (i, PUnit.unit)
        = afcRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩)
    (halpha : alphaT = Tile.uop WithBot.realExp2
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
    (hacc1 : acc1T = Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
    (hpT : ∀ jL : Fin BLOCK_N, pT.data (i, jL, PUnit.unit)
        = if (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ))
    (hv : ∀ jL : Fin BLOCK_N, vtile.data (jL, d, PUnit.unit) = some (vval jL))
    (hvval : ∀ jL : Fin BLOCK_N, vval jL
        = (afcKVG qT kT vT keyScale i d ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).2) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        acc1T (Tile.dot [] pT vtile)).data (i, d, PUnit.unit)
      = some ((afcStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i d).2.2) := by
  have hSEQ : 0 < SEQ := by
    have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega
  have hc1' : 1 ≤ (c + 1) * BLOCK_N := by
    have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega
  set m := (afcStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).1 with hm_def
  set Mc := afcRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩ with hMc
  set Mc1 := afcRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ with hMc1
  have hmMc : m = Mc := by
    rw [hm_def, hMc, afcStateBotG_fst_eq_runningMax,
      afcRunningMaxG_eq qT kT vT keyScale qStart (c * BLOCK_N) i d ⟨0, hBD⟩]
  have hmd : m = afcRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i d := by
    rw [hm_def, afcStateBotG_fst_eq_runningMax]
  have hMsucc : Mc1 = m ⊔ ((afcBlockG qT kT vT keyScale qStart BLOCK_N c i d).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    rw [hMc1, afcRunningMaxG_eq qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ d,
      afcRunningMaxG_succ, hmd]
  have halphaVal : alphaT.data (i, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [halpha]; show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmtile, hmij,
      NumericDType.sub, ← hMc, ← hMc1, hmMc]
  have hMc1bot : Mc1 ≠ ⊥ := by
    rw [hMc1]; exact afcRunningMaxG_ne_bot qT kT vT keyScale qStart ((c + 1) * BLOCK_N) hc1' hSEQ i ⟨0, hBD⟩
  have hdot := afc_acc_dot_block_maskedG qT kT vT keyScale BLOCK_N hBM hBD qStart c hc1 i d qkSentT mijT pT vtile vval
    (by rw [← hMc1]; exact hMc1bot) hsent hmij hpT hv hvval
  have hblockEq := osStepBot_block_eq m
    ((afcStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.1)
    ((afcStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2)
    ((afcKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i d).map (fun p => pow2 p.1 * p.2)).sum
    ((afcKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i d).map (fun p => pow2 p.1)).sum
    (afcBlockG qT kT vT keyScale qStart BLOCK_N c i d)
    (by rw [afc_denom_anchorG, zero_add, hm_def])
    (by rw [afc_acc_anchorG, zero_add, hm_def])
    (fun hbot => afcKeysUptoG_sum_zero_of_bot qT kT vT keyScale qStart (c * BLOCK_N) i d
      (by rw [← afcStateBotG_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => afcKeysUptoG_sum_zero_of_bot qT kT vT keyScale qStart (c * BLOCK_N) i d
      (by rw [← afcStateBotG_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  rw [show (afcStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i d).2.2
        = (Mc1, _,
            (afcStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2
              * (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
              + ((afcBlockG qT kT vT keyScale qStart BLOCK_N c i d).map (fun p => pow2 (p.1 - Mc1.unbotD 0) * p.2)).sum).2.2
        from by rw [afcStateBotG_succ]; rw [← hblockEq]]
  set α : ℝ := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp2 (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  have hcancel := (afcStateBot1G_cancel qT kT vT keyScale qStart BLOCK_N c hBN i d Mc1 hSEQ).2
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex]
  erw [hdot]
  rw [hacc1, Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, hacctile, halphaVal, hαsome]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [show (afcStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2 * α
        = (afcStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2 * α from by
    have := hcancel; simp only [← hm_def, ← hαdef] at this ⊢; exact this]

/-- **General ⊥-seeded full-window state reads off the genuine closed-form spec.** -/
theorem afcStateBotG_full_eq_spec
    (s : BlockState) (Q K V : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (keyScale : Fin (BLOCK_N * numKVBlocks) → ℝ)
    (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hne : afcRunningMaxG
        (qTileAFCmG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
        (kTileAFCG s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
        (vTileAFCmG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
        keyScale (qStartAFCG s BLOCK_M) (BLOCK_N * numKVBlocks) i d ≠ ⊥) :
    (let st := afcStateBotG
        (qTileAFCmG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
        (kTileAFCG s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
        (vTileAFCmG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
        keyScale (qStartAFCG s BLOCK_M) (BLOCK_N * numKVBlocks) i d
     st.2.2 / st.2.1)
      = attnFwdCausalOutSpecG s Q K V stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N
          BLOCK_DMODEL HEAD_ACTIVE numKVBlocks keyScale (i, d, PUnit.unit) := by
  simp only
  rw [afcStateBotG_ratio_eq _ _ _ _ _ _ _ _ hne]
  rw [afcKeysUptoG_full_eq_pred]
  rw [attnFwdCausalOutSpecG_eq_streaming]
  rw [VeriTile.Triton.osStep_foldl_eq_batch]

set_option maxRecDepth 8000 in
/-- **General body split** — the lowered general AFC body decomposes as
`take 22 ++ (forRange "start_n" 0 N_CTX BLOCK_N afcLoopBodyG :: drop 23)`. The
static `forRange` (NOT `forRangeDyn`) sits at index 22 regardless of dimension
values (the AST structure is dim-value-independent), checked by `rfl`. -/
theorem afc_body_splitG
    (Q K V QScale KScale Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat) :
    (attn_fwd_causal_surface Q K V QScale KScale Out
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.body
      = (attn_fwd_causal_surface Q K V QScale KScale Out
          sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
          Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.body.take 22
        ++ (Stmt.forRange "start_n" 0 N_CTX BLOCK_N
              (AfcFoundation.afcLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE)
            :: (attn_fwd_causal_surface Q K V QScale KScale Out
                sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
                Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.body.drop 23) :=
  rfl

end General

end VeriTile.Bench.TritonBenchG.AttnFwdCausal
