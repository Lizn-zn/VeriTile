import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention
import VeriTile.Triton.Kernel

/-!
# `attn_fwd_triton` — strict per-kernel correctness

`attn_fwd_triton.py`'s `_attn_fwd` is a (full, `STAGE = 3`) FlashAttention
forward kernel: program `(start_m, off_hz)` streams `K`/`V` blocks for one
`(batch·head, query block)` tile, maintaining the online-softmax running max
`m_i`, denominator `l_i`, and accumulator `acc` (with `q_scale · k_scale`
quantization), then stores the normalized `acc / l_i` to `Out`, masked to the
first 96 head lanes and `offs_m < N_CTX`.

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
attn_fwd_triton_python_test_shape_output_summary           ← TOP THEOREM
  ├─ attn_fwd_triton_surface_toAlgorithm_supported          surface lowers to the algorithm layer
  └─ attn_fwd_triton_surface_python_test_shape_compute_correct
       └─ attn_fwd_triton_final_store_python_test_shape_compute_correct
            └─ attn_fwd_triton_final_store_slice_compute_correct
                 └─ attn_fwd_triton_final_store_slice_correct   ← algorithm-layer readback per lane
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; the `exp2`, the `tl.dot`
`float16` accumulation, and `q_scale · k_scale` quantization are not modeled at
the bit level); `@triton.autotune`/`num_warps`/`num_stages` are not modeled.
The verified result is the **genuine closed form**: the full surface (preLoop +
the inlined `forRange` streaming loop + postLoop) is executed and its `Out` store
is proven equal to `attnFwdTritonOutSpec` — masked base-2 (`exp2`) per-key-scale
**causal** softmax (`keyScale = aftKeyScale = q_scale · k_scale[block]`,
`keep = causalKeep qStart`, head-active `q`/`k`/`v` masks at `e/d ≥ 96`) — at
every active output lane (`attn_fwd_triton_python_test_shape_output_summary`). The
inner online-softmax recurrence (`m_i`/`l_i`/`acc` updates, the final `acc / l_i`
normalization) is reconciled to this closed form via the ⊥-seeded `aftStateBot`
streaming fold and `aftStateBot_full_eq_spec`; the loop is driven by `forRange_inv`
+ `aft_attn_step`. Two genuine side conditions are explicit: `aftScoreBound` (the
`-1e6` `tl.where` mask sentinel is inert — every kept score `≥ -1e6`) and the
clean `undef ≡ 0` initial state. Side condition: the test-shape wrapper fixes the
concrete layout (`B = 2`, `H = 4`, `N_CTX = HEAD_DIM = BLOCK_M = 128`,
`BLOCK_N = 64`, strides `(65536, 16384, 128, 1)`, mask = first 96 head lanes) and
uses `STAGE = 3`.
-/

namespace VeriTile.Bench.TritonBenchG.AttnFwdTriton

open VeriTile.Triton

set_option linter.unusedSimpArgs false

section Correct

/-- Full Lean port of `attn_fwd_triton.py`'s `_attn_fwd` (`STAGE = 3`).

The upstream kernel runs the K/V streaming-softmax loop through a separate
`@triton.jit` helper `_attn_fwd_inner`, invoked twice: stage `4 - STAGE = 1`
streams the strictly-below-diagonal key blocks `range(0, start_m·BLOCK_M)` with
no causal mask, and stage `2` streams the diagonal block
`range(start_m·BLOCK_M, (start_m+1)·BLOCK_M)` under the causal predicate
`offs_m[:, None] ≥ start_n + offs_n[None, :]`.

The DSL has no cross-`@triton.jit` function-call surface, so the helper body is
inlined here as a single `forRange` loop `range(0, N_CTX, BLOCK_N)` with the
causal `where` `offs_m[:, None] ≥ start_n + offs_n[None, :]` applied to every
block. This faithfully composes the two staged helper calls of `STAGE = 3`: on a
stage-1 block (strictly below the diagonal) every lane satisfies
`offs_m ≥ start_n + offs_n`, so the causal `where` is a no-op there, matching the
unmasked stage-1 helper; on the diagonal block it is the stage-2 mask; and on
the strictly-above-diagonal blocks (which the two-call kernel never visits) the
`where` zeroes every probability (`p = where(mask, p, 0)`), so they contribute
nothing to `acc`/`l_i` — making the full-range loop equal to the kernel's
`range(0, (start_m+1)·BLOCK_M)` traversal. -/
def attn_fwd_triton_surface
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

/-- The full staged `attn_fwd_triton` surface lowers to the algorithm layer. -/
theorem attn_fwd_triton_surface_toAlgorithm_supported
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat) :
    ∃ alg, (attn_fwd_triton_surface Q K V Q_scale K_scale Out stride_qz
      stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om
      stride_on Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE
      STAGE).toAlgorithm? = Except.ok alg := by
  simp [attn_fwd_triton_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of `attn_fwd_triton.py`'s
`_attn_fwd`.

The full kernel runs staged attention forward loops. This slice starts after those stages have produced a
precomputed normalized `Acc` tile and proves the final masked writeback into
`Out`, preserving the source store address and mask
`(offs_m < N_CTX) & (offs_k < 96)`. The inner `tl.float32` accumulator and
`p.to(tl.float16)` dot-input cast are outside this slice. -/
def attn_fwd_triton_final_store_slice
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
theorem attn_fwd_triton_final_store_slice_correct
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
      (exec (attn_fwd_triton_final_store_slice Acc Out H N_CTX
            HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k
            stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (if active s N_CTX HEAD_ACTIVE BLOCK_M idx then
            s.readMem Acc
              (accOffset s H stride_acc_z stride_acc_h stride_acc_m
                stride_acc_k BLOCK_M idx)
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, attn_fwd_triton_final_store_slice, stepStmts, stepStmt,
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
theorem attn_fwd_triton_final_store_slice_compute_correct
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
      (kernel := attn_fwd_triton_final_store_slice Acc Out H N_CTX
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
  · simp [attn_fwd_triton_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := attn_fwd_triton_final_store_slice_correct Acc Out H N_CTX
    HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k stride_qz
    stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Python test-shape wrapper

`attn_fwd_triton.py`'s checked tests use `B = 2`, `H = 4`, `N_CTX = 128`,
`HEAD_DIM = 128`, `BLOCK_M = 128`, `BLOCK_N = 64`, and the source mask enables
the first 96 head lanes. Contiguous `[B, H, N_CTX, HEAD_DIM]` tensors have
strides `(65536, 16384, 128, 1)`. -/

end Correct

section TestShape

theorem attn_fwd_triton_final_store_python_test_shape_compute_correct
    (Acc Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attn_fwd_triton_final_store_slice Acc Out
        4 128 96 65536 16384 128 1 65536 16384 128 1 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] => active s 128 96 128 idx)
        (fun idx : TileIndex [128, 128] => (Out,
          outOffset s 4 65536 16384 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        s.readMem Acc (accOffset s 4 65536 16384 128 1 128 idx)) := by
  apply attn_fwd_triton_final_store_slice_compute_correct
  rintro ⟨⟨ma, hma⟩, ⟨ka, hka⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨kb, hkb⟩, _⟩ h
  simp [outOffset, offZ, offH, mIndex, kIndex] at h
  have hm : ma = mb := by omega
  have hk : ka = kb := by omega
  subst mb
  subst kb
  rfl

end TestShape

section Correct

/-! ## Genuine closed-form attention spec (exp2, causal)

`attn_fwd_triton.py`'s `_attn_fwd` (`STAGE = 3`) is **base-2** (`tl.math.exp2`)
softmax with a **scalar** score scale `q_scale · k_scale` (loaded once per
program / per key block) and a **causal** mask
`tl.where(offs_m[:, None] ≥ start_n + offs_n[None, :], qk, -inf)`. So the genuine
closed form is the predicate-masked base-2 per-key-scale attention
`attentionRealBase2PerKeyScalePred` (`VeriTile/Triton/Math/Attention.lean`),
instantiated with:

* `keep := causalKeep qStart` — key `j` contributes to query row `i` iff
  `j ≤ qStart + i` (the kernel's `offs_m ≥ start_n + offs_n` mask, with global
  `qStart = start_m · BLOCK_M`);
* a per-key score scale carrier (the scalar `q_scale · k_scale`; here abstracted
  as `keyScaleAFT`, an opaque `Fin S → ℝ`, since the kernel loads the quantization
  scalars from memory rather than fixing them at a constant like triton3).

This routes to **base-2** (NOT FlashAttention1's natural-exp
`attentionRealCausalBlock`), matching the `exp2` kernel; the streaming bridge
`attentionRealBase2PerKeyScalePred_eq_streaming` (sorry-free) delivers the
`osStep` online-softmax fold the exec loop realizes. -/

open VeriTile.Triton (attentionRealBase2PerKeyScalePred attnKeyListPred osStep
  causalKeep)

/-- Base address of the `(off_z, off_h)` plane for the Python test shape
(`stride_qz = 65536`, `stride_qh = 16384`, `H = 4`). Q, K, V, Out share it
(`stride_qm = 128`, `stride_qk = 1`). -/
def baseOffsetAFT (s : BlockState) : Nat :=
  s.pids 1 / 4 * 65536 + s.pids 1 % 4 * 16384

/-- Query tile: row `i` = global query `start_m·128 + i`, head lane `e`
(`stride_qm = 128`, `stride_qk = 1`). -/
noncomputable def qTileAFT (s : BlockState) (Q : RegionName) :
    TileIndex [128, 128] → ℝ :=
  fun (i, e, _) => s.readMem Q (baseOffsetAFT s + (s.pids 0 * 128 + i.val) * 128 + e.val)

/-- **Masked query tile** (HEAD_ACTIVE faithful): the kernel loads `q` through the
`(offs_m < 128) & (arange < 96)` mask, so the genuine score only sees head lanes
`e < 96` (and only rows `qStart + i < 128`). This is the q the kernel's `tl.dot`
actually contracts — `qMaskedAFT i e = if (qStart + i < 128 ∧ e < 96) then
qTileAFT i e else 0`. The closed-form spec is stated over this masked q (faithful
to `qLoadedAFT`), so the score `Σ_{e<128} qMasked·k = Σ_{e<96} qTile·k`. -/
noncomputable def qMaskedAFT (s : BlockState) (Q : RegionName) :
    TileIndex [128, 128] → ℝ :=
  fun (i, e, _) =>
    if (s.pids 0 * 128 + i.val < 128 ∧ e.val < 96) then
      s.readMem Q (baseOffsetAFT s + (s.pids 0 * 128 + i.val) * 128 + e.val)
    else 0

/-- Key tile: row `j` (global key), head lane `e`. -/
noncomputable def kTileAFT (s : BlockState) (K : RegionName) :
    TileIndex [128, 128] → ℝ :=
  fun (j, e, _) => s.readMem K (baseOffsetAFT s + j.val * 128 + e.val)

/-- Value tile: row `j` (global key), head lane `d`. -/
noncomputable def vTileAFT (s : BlockState) (V : RegionName) :
    TileIndex [128, 128] → ℝ :=
  fun (j, d, _) => s.readMem V (baseOffsetAFT s + j.val * 128 + d.val)

/-- **Masked value tile** (HEAD_ACTIVE faithful) — the tile the kernel actually
loads: `vTileAFT` but `0` for output channels `d ≥ 96` (the `arange < 96` head-active
mask on the `v` load zeroes those lanes, so `acc[·, d≥96] = 0`). The closed-form
spec accumulates this masked value; the final store masks `d ≥ 96` anyway, and
`vMaskedAFT = vTileAFT` at `d < 96` where it matters. -/
noncomputable def vMaskedAFT (s : BlockState) (V : RegionName) :
    TileIndex [128, 128] → ℝ :=
  fun (j, d, u) => if d.val < 96 then vTileAFT s V (j, d, u) else 0

/-- Global query row for output tile-row `i` in this program. -/
def qStartAFT (s : BlockState) : Nat := s.pids 0 * 128

/-- **Genuine closed form** (exp2, causal): the normalized output `acc / l_i` is
predicate-masked base-2 per-key-scale attention with the `causalKeep qStart`
mask, for an arbitrary per-key score-scale carrier `keyScale`. -/
noncomputable def attnFwdTritonOutSpec
    (s : BlockState) (Q K V : RegionName) (keyScale : Fin 128 → ℝ)
    (idx : TileIndex [128, 128]) : ℝ :=
  attentionRealBase2PerKeyScalePred (qMaskedAFT s Q) (kTileAFT s K) (vMaskedAFT s V)
    keyScale (fun i j => causalKeep (qStartAFT s) i j) idx

/-- Streaming bridge: the closed form equals the `osStep` online-softmax fold
over the causal-masked key list — the form the exec loop realizes. -/
theorem attnFwdTritonOutSpec_eq_streaming
    (s : BlockState) (Q K V : RegionName) (keyScale : Fin 128 → ℝ) (i d : Fin 128) :
    attnFwdTritonOutSpec s Q K V keyScale (i, d, PUnit.unit)
      = (let st := (attnKeyListPred (qMaskedAFT s Q) (kTileAFT s K) (vMaskedAFT s V)
            keyScale (fun i j => causalKeep (qStartAFT s) i j) i d).foldl
              osStep (0, 0, 0)
         st.2.2 / st.2.1) := by
  simpa [attnFwdTritonOutSpec] using
    VeriTile.Triton.attentionRealBase2PerKeyScalePred_eq_streaming
      (qMaskedAFT s Q) (kTileAFT s K) (vMaskedAFT s V) keyScale
      (fun i j => causalKeep (qStartAFT s) i j) i d

/-! ## Forward-loop per-statement op-eval recipes (RECIPE LAYER)

The inlined `forRange "start_n" 0 N_CTX BLOCK_N` loop body of
`attn_fwd_triton_surface`, extracted from the lowered algorithm AST at the Python
test shape (`BLOCK_M = BLOCK_DMODEL = 128`, `BLOCK_N = 64`, `N_CTX = 128`;
`@[simp]`-erased `ComputeOp.alg` fp32/fp16 dtype-cast wrappers, so the
algorithm-layer body is plain `Op`s), is:

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
`aft3_*` / `flash_*` recipes. The eventual step lemma threads them through
`stepStmts.cons_some`. ASSEMBLY (invariant / `attn_step` / pre+post-loop over the
`forRange`) is the NEXT stage and is intentionally NOT attempted here. -/

/-- `evalOp` helper for `tl.math.exp2` (`Op.exp2`). -/
theorem aft_evalOp_exp2 {shape : TileShape} (a : Op .real shape) (s : BlockState) :
    evalOp (.exp2 a) s = (do let va ← evalOp a s; some (Tile.uop WithBot.realExp2 va)) := by
  simp [evalOp]

/-- `evalOp` helper for the `>=` predicate (`Op.ge`), no `@[simp]` form. -/
theorem aft_evalOp_ge {dtype a b shape} (h : ComparableDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

/-- `evalOp` helper for `Op.boolAnd`, no `@[simp]` form. -/
theorem aft_evalOp_boolAnd {a b shape} (bc : Broadcast a b shape)
    (x : Op .bool a) (y : Op .bool b) (s : BlockState) :
    evalOp (.boolAnd bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s;
      some (Tile.bop (fun p q : Bool => p && q) bc vx vy)) := by
  simp [evalOp]

/-- `evalOp` helper for the identity `real → real` cast (the `.to(tl.float32)`
fp32 wrapper around `tl.dot`, which is a no-op at the algorithm layer): given the
inner op's value, the cast returns the same tile (`FloatDType.cast real real` is
the identity through `WithBot ℝ`). -/
theorem aft_castReal_eval {shape : TileShape} (a : Op .real shape) (s : BlockState)
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
theorem aft_load_k_eval
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
theorem aft_load_kscale_eval
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
theorem aft_load_v_eval
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
theorem aft_kmask_eval (s : BlockState) (SN NCTX HA : Nat)
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
  rw [aft_evalOp_boolAnd]
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
theorem aft_qk_dot_eval (s : BlockState) (BM BN BD : Nat)
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
    aft_castReal_eval
      (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k"))
      s (Tile.dot [] qtile ktile) hdot
  rw [evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, hcast, hqsc, hksc, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L6: `mask = offs_m[:,None] >= (start_n + offs_n[None,:])`** — the causal
keep predicate. Cell `(i, j)` is kept iff `offs_m i ≥ SN + offs_n j`. -/
theorem aft_mask_eval (s : BlockState) (SN : Nat)
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
  rw [aft_evalOp_ge]
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
theorem aft_where_qk_eval (s : BlockState) (masktile : Tile .bool [128, 64])
    (qktile : Tile .real [128, 64])
    (hmask : s.regs .bool [128, 64] "mask" = some masktile)
    (hqk : s.regs .real [128, 64] "qk" = some qktile) :
    evalOp (Op.where (Op.ref .bool [128, 64] "mask")
        (Op.ref .real [128, 64] "qk")
        (Op.broadcast (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1000000.0))
          [128, 64])) s
      = some ⟨fun idx : TileIndex [128, 64] =>
          if masktile.data idx then qktile.data idx
          else WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ))⟩ := by
  have hbcast : @evalOp TileDType.real [128, 64]
      (Op.broadcast (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1000000.0)) [128, 64]) s
      = some (⟨fun _ : TileIndex [128, 64] =>
          WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ))⟩ :
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
theorem aft_mij_eval (s : BlockState)
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
theorem aft_qk_sub_eval (s : BlockState) (hax : 1 < [128].length.succ)
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
theorem aft_p_eval (s : BlockState) (qktile : Tile .real [128, 64])
    (hqk : s.regs .real [128, 64] "qk" = some qktile) :
    evalOp (Op.exp2 (Op.ref .real [128, 64] "qk")) s
      = some (Tile.uop WithBot.realExp2 qktile) := by
  rw [aft_evalOp_exp2]; simp only [evalOp_ref, hqk, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L11: `p = tl.where(mask, p, 0)`** — zero the non-kept lanes. -/
theorem aft_p_mask_eval (s : BlockState) (masktile : Tile .bool [128, 64])
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
theorem aft_lij_eval (s : BlockState) (ptile : Tile .real [128, 64])
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
theorem aft_alpha_eval (s : BlockState) (mi mij : Tile .real [128])
    (hmi : s.regs .real [128] "m_i" = some mi)
    (hmij : s.regs .real [128] "m_ij" = some mij) :
    evalOp (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [128] "m_i") (Op.ref .real [128] "m_ij"))) s
      = some (Tile.uop WithBot.realExp2
          (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mi mij)) := by
  rw [aft_evalOp_exp2, evalOp_sub]
  simp only [evalOp_ref, hmi, hmij, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L14: `l_i = l_i * alpha + l_ij`** — the denominator carry. -/
theorem aft_li_eval (s : BlockState) (li alpha lij : Tile .real [128])
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
theorem aft_acc_rescale_eval (s : BlockState) (hax : 1 < [128].length.succ)
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
theorem aft_p_fp16_eval (s : BlockState) (ptile : Tile .real [128, 64])
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
theorem aft_acc_eval (s : BlockState) (BM BN BD : Nat)
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
theorem aft_mi_carry_eval (s : BlockState) (mij : Tile .real [128])
    (hmij : s.regs .real [128] "m_ij" = some mij) :
    evalOp (Op.ref .real [128] "m_ij") s = some mij := by
  rw [evalOp_ref, hmij]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L20/L22: `K_ptrs += BLOCK_N * HEAD_DIM` / `V_ptrs += BLOCK_N * HEAD_DIM`** —
advance a `[BT, BS]` pointer tile by the scalar block stride `d`, broadcast on
the right. -/
theorem aft_advance_ptr_eval (s : BlockState) (BT BS d : Nat) (name : RegName)
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
theorem aft_advance_kscale_eval (s : BlockState) (name : RegName) (ptr : Tile .ptr [])
    (hptr : s.regs .ptr [] name = some ptr) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ref .ptr [] name) (Op.constNat 1)) s
      = some (Tile.ptrAdd Broadcast.nil ptr (Tile.scalar 1)) := by
  simp only [evalOp, evalOp_ref, evalOp_constNat, hptr, Option.bind_eq_bind, Option.bind_some]

/-! ## PTR-BIND lemma kit (bind-aware `.ptr` stepping)

The preLoop `.ptr` constructions (`Q_ptrs`/`K_ptrs`/`V_ptrs`/`O_block_ptr` =
`ptrAdd (ptrBase R) <add-with-expandDim>`) and the masked `tl.load`s do not reduce
under `simp`/`simp only` inside the `stepStmt`/`evalOp` `Option.bind` do-block:
`evalOp_expandDim_ref_of_regs` fires only on a bare `rw`, not inside the bind chain.
These reducers take **pre-evaluated** operand values (`evalOp ptr s = some …`,
`evalOp off s = some …`) and fire the `ptrAdd`/`load` reduction with the operands
already discharged — so each `expandDim`/`add`/`mul` sub-operand is proved as its own
`have hexp := evalOp_expandDim_ref_of_regs …` term and threaded in. Keyed to fire
inside the do-block for `aftPreLoop_eval` and the `aft_load_*` threading. -/

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- `evalOp (Op.ptrBase R)` — the base pointer tile `(R, 0)` at the empty shape. -/
theorem aft_evalOp_ptrBase {d : TileDType} (region : Region d) (s : BlockState) :
    evalOp (Op.ptrBase region) s = some (Tile.scalar (Region.cast region, 0)) := by
  simp only [evalOp]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Bind-aware `ptrAdd` reducer.** Given the operand evaluations
`evalOp ptr s = some ptrs` and `evalOp off s = some offs`, fire the `ptrAdd`
reduction inside the `Option.bind` do-block. Threads each `expandDim`-bearing
offset (pre-proved via `evalOp_expandDim_ref_of_regs` / `evalOp_add` / `evalOp_mul`)
into the elementwise `Tile.ptrAdd`. -/
theorem aft_evalOp_ptrAdd_of {a b shape : TileShape}
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
Generalizes `aft_load_k_eval`/`aft_load_v_eval` to inline (non-`ref`) ptr/mask ops,
unblocking the preLoop `q` load (inline `boolAnd` mask). -/
theorem aft_evalOp_load_ptr_mask_of {shape : TileShape}
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
theorem aft_evalOp_load_ptr_none_of {shape : TileShape}
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
theorem aft_reduceMaxDrop1_some (x : Tile .real [128, 64]) :
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
causally-filtered key prefix. All defs/lemmas here are pure math over the AFT
tile-functions (`qTileAFT`/`kTileAFT`/`vTileAFT`), independent of the exec layer.

`osStepBot` and the consistency/block lemmas are ported from the flash-attn
foundation (`origin/fix/flash-attn-full`, #303); the spec bridge
`aftStateBot_full_eq_spec` reconnects to `attnFwdTritonOutSpec` via the
`attnFwdTritonOutSpec_eq_streaming` online-softmax fold (the running max cancels
in the `acc / l` ratio, so the ⊥-seed and the neutral `(0,0,0)` seed agree). -/

open VeriTile.Triton (osStep pow2 pow2_add pow2_pos sum_map_pow2_sub)

/-- The `(score, value)` pair the kernel streams for output `(i, d)` at global
key `j`: score `keyScale j · (q row i · k row j)`, value `V[j, d]`. The per-key
score scale `keyScale j` carries the scalar `q_scale · k_scale` quantization. -/
noncomputable def aftKV
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (i : Fin 128) (d : Fin 128) (j : Fin 128) : ℝ × ℝ :=
  (keyScale j * Finset.univ.sum (fun e : Fin 128 => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)),
   vT (j, d, PUnit.unit))

/-- Causal per-row key list over the window `[0, hi)`: keys `j < hi` with
`j ≤ qStart + i` (the `offs_m ≥ start_n + offs_n` causal mask), in index order.
After `c` blocks `hi = c · 64`, this is the prefix the kernel has streamed. -/
noncomputable def aftKeysUpto
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128) : List (ℝ × ℝ) :=
  (List.finRange 128).filterMap (fun j : Fin 128 =>
    if j.val < hi ∧ j.val ≤ qStart + i.val then
      some (aftKV qT kT vT keyScale i d j)
    else none)

/-- Block-`c` per-row key list: keys with `c·64 ≤ j < (c+1)·64` passing the
causal filter — the keys the loop's `c`-th iteration streams. -/
noncomputable def aftBlock
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart c : Nat) (i : Fin 128) (d : Fin 128) : List (ℝ × ℝ) :=
  (List.finRange 128).filterMap (fun j : Fin 128 =>
    if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val then
      some (aftKV qT kT vT keyScale i d j)
    else none)

/-- **⊥-seeded running max** of the streamed key prefix `[0, hi)`: the value the
kernel carries in `m_i` (`tl.zeros − inf` seeds at `⊥`). The `WithBot ⊔`-fold of
the coerced per-key scores; `⊥` on the empty / `hi = 0` window. -/
noncomputable def aftRunningMax
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128) : WithBot ℝ :=
  ((aftKeysUpto qT kT vT keyScale qStart hi i d).map
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

/-- `aftStateBot` — the ⊥-seeded running `(max, denom, acc)` after streaming the
window `[0, hi)`. Faithful to the kernel's register recurrence (`m_i` seeded `⊥`,
`l_i`/`acc` seeded `0`). -/
noncomputable def aftStateBot
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128) : WithBot ℝ × ℝ × ℝ :=
  (aftKeysUpto qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 0, 0)

/-- The running `max` component of an `osStepBot` fold is the `WithBot ⊔`-fold. -/
theorem aftStateBot_fst
    (xs : List (ℝ × ℝ)) (m₀ : WithBot ℝ) (l₀ acc₀ : ℝ) :
    (xs.foldl osStepBot (m₀, l₀, acc₀)).1
      = (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldl (· ⊔ ·) m₀ := by
  induction xs generalizing m₀ l₀ acc₀ with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl_cons, List.map_cons]; rw [ih]; rfl

/-- The `WithBot ⊔`-fold is seed/direction-agnostic. -/
theorem aft_foldl_sup_bot_eq_foldr (L : List (WithBot ℝ)) :
    L.foldl (· ⊔ ·) (⊥ : WithBot ℝ) = L.foldr (· ⊔ ·) (⊥ : WithBot ℝ) := by
  have gen : ∀ (m : WithBot ℝ), L.foldl (· ⊔ ·) m = m ⊔ L.foldr (· ⊔ ·) ⊥ := by
    induction L with
    | nil => intro m; simp
    | cons a t ih => intro m; simp only [List.foldl_cons, List.foldr_cons, ih]; rw [max_assoc]
  rw [gen ⊥, bot_sup_eq]

/-- The ⊥-seeded running `max` of `aftStateBot` is exactly `aftRunningMax`. -/
theorem aftStateBot_fst_eq_runningMax
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128) :
    (aftStateBot qT kT vT keyScale qStart hi i d).1
      = aftRunningMax qT kT vT keyScale qStart hi i d := by
  rw [aftStateBot, aftStateBot_fst, aftRunningMax, aft_foldl_sup_bot_eq_foldr]

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

/-- The ⊥-seeded denominator equals `κ(aftRunningMax)·Σpow2 score`. -/
theorem aftStateBot_snd_fst
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128) :
    (aftStateBot qT kT vT keyScale qStart hi i d).2.1
      = ((aftRunningMax qT kT vT keyScale qStart hi i d).elim 0 (fun r => pow2 (-r)))
        * (0 + ((aftKeysUpto qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1)).sum) := by
  rw [aftStateBot]
  rw [(osStepBot_foldl_consistent (aftKeysUpto qT kT vT keyScale qStart hi i d) ⊥ 0 0 0 0
    (by simp) (by simp) (by simp) (by simp)).1]
  rw [show ((aftKeysUpto qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 0, 0)).1
        = aftRunningMax qT kT vT keyScale qStart hi i d from by
    rw [aftStateBot_fst, aftRunningMax, aft_foldl_sup_bot_eq_foldr]]

/-- The ⊥-seeded accumulator equals `κ(aftRunningMax)·Σpow2 score·v`. -/
theorem aftStateBot_snd_snd
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128) :
    (aftStateBot qT kT vT keyScale qStart hi i d).2.2
      = ((aftRunningMax qT kT vT keyScale qStart hi i d).elim 0 (fun r => pow2 (-r)))
        * (0 + ((aftKeysUpto qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1 * p.2)).sum) := by
  rw [aftStateBot]
  rw [(osStepBot_foldl_consistent (aftKeysUpto qT kT vT keyScale qStart hi i d) ⊥ 0 0 0 0
    (by simp) (by simp) (by simp) (by simp)).2]
  rw [show ((aftKeysUpto qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 0, 0)).1
        = aftRunningMax qT kT vT keyScale qStart hi i d from by
    rw [aftStateBot_fst, aftRunningMax, aft_foldl_sup_bot_eq_foldr]]

/-- The ⊥-seeded `acc / denom` ratio is the running-max-free batch ratio (the max
factor cancels). Valid whenever the streamed window is nonempty (`aftRunningMax ≠ ⊥`). -/
theorem aftStateBot_ratio_eq
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128)
    (hne : aftRunningMax qT kT vT keyScale qStart hi i d ≠ ⊥) :
    (aftStateBot qT kT vT keyScale qStart hi i d).2.2
        / (aftStateBot qT kT vT keyScale qStart hi i d).2.1
      = ((aftKeysUpto qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1 * p.2)).sum
        / ((aftKeysUpto qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1)).sum := by
  rw [aftStateBot_snd_fst, aftStateBot_snd_snd, zero_add, zero_add]
  cases hM : aftRunningMax qT kT vT keyScale qStart hi i d with
  | bot => exact absurd hM hne
  | coe r =>
    have hκ : ((r : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-r) := rfl
    rw [hκ]
    have hpos : pow2 (-r) ≠ 0 := ne_of_gt (pow2_pos _)
    rw [mul_div_mul_left _ _ hpos]

/-- The ⊥-seeded running max at the empty / `hi = 0` window is `⊥`. -/
theorem aftRunningMax_zero
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart : Nat) (i : Fin 128) (d : Fin 128) :
    aftRunningMax qT kT vT keyScale qStart 0 i d = ⊥ := by
  unfold aftRunningMax aftKeysUpto
  rw [show (List.finRange 128).filterMap
        (fun j : Fin 128 => if j.val < 0 ∧ j.val ≤ qStart + i.val
          then some (aftKV qT kT vT keyScale i d j) else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-- Generic threshold-split for a `.val`-ascending list. Ported from flash-attn. -/
private theorem aft_filterMap_window_split {n : Nat} (l : List (Fin n))
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
theorem aftKeysUpto_succ
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart c : Nat) (i : Fin 128) (d : Fin 128) :
    aftKeysUpto qT kT vT keyScale qStart ((c + 1) * 64) i d
      = aftKeysUpto qT kT vT keyScale qStart (c * 64) i d
        ++ aftBlock qT kT vT keyScale qStart c i d := by
  unfold aftKeysUpto aftBlock
  rw [show (List.finRange 128).filterMap
        (fun j : Fin 128 => if j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val
          then some (aftKV qT kT vT keyScale i d j) else none)
      = (List.finRange 128).filterMap
        (fun j : Fin 128 => if j.val ≤ qStart + i.val ∧ j.val < (c + 1) * 64
          then some (aftKV qT kT vT keyScale i d j) else none)
      from List.filterMap_congr (fun j _ => by simp only [and_comm])]
  rw [aft_filterMap_window_split (List.finRange 128) (List.pairwise_lt_finRange 128)
    (c * 64) ((c + 1) * 64) (fun j => j.val ≤ qStart + i.val)
    (fun j => aftKV qT kT vT keyScale i d j) (by nlinarith [Nat.zero_le (64 : Nat)])]
  refine congrArg₂ (· ++ ·) ?_ ?_
  · apply List.filterMap_congr; intro j _; simp only [and_comm]
  · apply List.filterMap_congr; intro j _
    by_cases h1 : c * 64 ≤ j.val <;> by_cases h2 : j.val < (c + 1) * 64 <;>
      by_cases h3 : j.val ≤ qStart + i.val <;> simp [h1, h2, h3, and_assoc]

/-- **One-block advance**: `aftStateBot` after `c+1` blocks is `osStepBot`-folded
over block `c`'s keys from `aftStateBot` after `c` blocks. -/
theorem aftStateBot_succ
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart c : Nat) (i : Fin 128) (d : Fin 128) :
    aftStateBot qT kT vT keyScale qStart ((c + 1) * 64) i d
      = (aftBlock qT kT vT keyScale qStart c i d).foldl osStepBot
          (aftStateBot qT kT vT keyScale qStart (c * 64) i d) := by
  unfold aftStateBot
  rw [aftKeysUpto_succ, List.foldl_append]

/-- The running max one-block advance: `aftRunningMax((c+1)·64) = aftRunningMax(c·64) ⊔ blockSup`. -/
theorem aftRunningMax_succ
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart c : Nat) (i : Fin 128) (d : Fin 128) :
    aftRunningMax qT kT vT keyScale qStart ((c + 1) * 64) i d
      = aftRunningMax qT kT vT keyScale qStart (c * 64) i d
        ⊔ ((aftBlock qT kT vT keyScale qStart c i d).map
            (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  unfold aftRunningMax
  rw [aftKeysUpto_succ, List.map_append]
  induction (aftKeysUpto qT kT vT keyScale qStart (c * 64) i d) with
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
    rw [aftStateBot_fst]
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
streaming spec `attnFwdTritonOutSpec_eq_streaming` folds. -/
theorem aftKeysUpto_full_eq_pred
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart : Nat) (i : Fin 128) (d : Fin 128) :
    aftKeysUpto qT kT vT keyScale qStart 128 i d
      = attnKeyListPred qT kT vT keyScale (fun a b => causalKeep qStart a b) i d := by
  unfold aftKeysUpto attnKeyListPred aftKV
  apply List.filterMap_congr
  intro j _
  have hjlt : j.val < 128 := j.isLt
  have hiff : causalKeep qStart i j ↔ j.val ≤ qStart + i.val := by
    unfold causalKeep; omega
  by_cases hc : causalKeep qStart i j
  · rw [if_pos ⟨hjlt, hiff.mp hc⟩, if_pos hc]
  · rw [if_neg (fun hh => hc (hiff.mpr hh.2)), if_neg hc]

/-- **The ⊥-seeded full-window state reads off the genuine closed-form spec.**
`aftStateBot(128).acc / aftStateBot(128).denom = attnFwdTritonOutSpec`. -/
theorem aftStateBot_full_eq_spec
    (s : BlockState) (Q K V : RegionName) (keyScale : Fin 128 → ℝ) (i d : Fin 128)
    (hne : aftRunningMax (qMaskedAFT s Q) (kTileAFT s K) (vMaskedAFT s V) keyScale
      (qStartAFT s) 128 i d ≠ ⊥) :
    (let st := aftStateBot (qMaskedAFT s Q) (kTileAFT s K) (vMaskedAFT s V) keyScale
        (qStartAFT s) 128 i d
     st.2.2 / st.2.1)
      = attnFwdTritonOutSpec s Q K V keyScale (i, d, PUnit.unit) := by
  simp only
  rw [aftStateBot_ratio_eq _ _ _ _ _ _ _ _ hne]
  rw [aftKeysUpto_full_eq_pred]
  rw [attnFwdTritonOutSpec_eq_streaming]
  rw [VeriTile.Triton.osStep_foldl_eq_batch]

/-! ## StateBot1 / StateBotK — kernel `l_i = 1` seed reconciliation

The kernel seeds `l_i = 1` (`tl.zeros + 1.0`) at preLoop (window `[0,0)`). On a
fully-masked block the kernel still *executes* the block (the masked `α =
exp2(⊥ − ⊥) = 0` annihilates the seed-`1`), so the faithful running state is the
seed-`1` `(⊥,1,0)` at window `0` and the seed-`0` ⊥-state for every later window.
`aftStateBotK` carries this; `aftStateBot1` (a pure `(⊥,1,0)`-seed fold) bridges. -/

/-- ⊥-seeded online-softmax fold from the kernel's `l_i = 1` seed. -/
noncomputable def aftStateBot1
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128) : WithBot ℝ × ℝ × ℝ :=
  (aftKeysUpto qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 1, 0)

/-- From a `⊥` running-max seed, the first `osStepBot` step wipes the `l`/`acc`
seed (`α = exp2(⊥ − s) = 0`), so a nonempty fold is seed-independent. -/
theorem osStepBot_bot_seed_indep (xs : List (ℝ × ℝ)) (hne : xs ≠ [])
    (l acc l' acc' : ℝ) :
    xs.foldl osStepBot (⊥, l, acc) = xs.foldl osStepBot (⊥, l', acc') := by
  obtain ⟨x, t, rfl⟩ := List.exists_cons_of_ne_nil hne
  obtain ⟨sv, v⟩ := x
  have hstep : ∀ L A : ℝ, osStepBot (⊥, L, A) (sv, v)
      = (((sv : ℝ) : WithBot ℝ), pow2 (sv - sv), pow2 (sv - sv) * v) := by
    intro L A
    simp only [osStepBot, bot_sup_eq]
    have hα : (WithBot.realExp2 (WithBot.realSub (⊥ : WithBot ℝ) ((sv : ℝ) : WithBot ℝ))).unbotD 0 = 0 := by
      rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
    have hub : (((sv : ℝ) : WithBot ℝ)).unbotD 0 = sv := by rfl
    rw [hα, hub]; simp
  simp only [List.foldl_cons, hstep]

/-- **Faithful kernel running state** (`l_i = 1` seed): `(⊥,1,0)` at window `0`,
the seed-`0` ⊥-state for later windows. -/
noncomputable def aftStateBotK
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d : Fin 128) : WithBot ℝ × ℝ × ℝ :=
  if hi = 0 then (⊥, 1, 0)
  else aftStateBot qT kT vT keyScale qStart hi i d

/-- **Seed cancellation.** From the kernel state `aftStateBotK(c·64)`, multiplying
`l`/`acc` by the next-block rescale `α = exp2(M_c − Mc1)` gives the same result as
from `aftStateBot(c·64)`: at `c = 0` the running max is `⊥` so `α = 0` kills the
seed-`1`; for `c > 0` the two states are definitionally equal. -/
theorem aftStateBotK_cancel
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart c : Nat) (i : Fin 128) (d : Fin 128) (Mc1 : WithBot ℝ) :
    let m := (aftStateBot qT kT vT keyScale qStart (c * 64) i d).1
    let α := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
    (aftStateBotK qT kT vT keyScale qStart (c * 64) i d).2.1 * α
        = (aftStateBot qT kT vT keyScale qStart (c * 64) i d).2.1 * α
      ∧ (aftStateBotK qT kT vT keyScale qStart (c * 64) i d).2.2 * α
        = (aftStateBot qT kT vT keyScale qStart (c * 64) i d).2.2 * α := by
  intro m α
  unfold aftStateBotK
  by_cases hc0 : c = 0
  · subst hc0
    simp only [Nat.zero_mul, if_pos rfl]
    have hmbot : m = ⊥ := by
      show (aftStateBot qT kT vT keyScale qStart (0 * 64) i d).1 = ⊥
      rw [aftStateBot_fst_eq_runningMax, Nat.zero_mul, aftRunningMax_zero]
    have hα0 : α = 0 := by
      show (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 = 0
      rw [hmbot, WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
    rw [hα0]; simp
  · simp only [Nat.mul_eq_zero, hc0, false_or, OfNat.ofNat_ne_zero, or_self, if_neg]
    exact ⟨rfl, rfl⟩

end Correct

/-! ## FOUNDATION Part 1 — `aftBody_split` (preLoop ++ forRange aftLoopBody :: postLoop)

The lowered algorithm body of `attn_fwd_triton_surface` at the Python test shape is
a 25-statement list: 22 preLoop statements (`aftPreLoop`), then the static
`Stmt.forRange "start_n" 0 128 64 aftLoopBody` (loop body = 22 statements), then 2
postLoop statements (`acc = acc / l_i[:, None]` and the masked `tl.store`). This is
a **static** `forRange` (range bounds `0..128 step 64`, NOT a `forRangeDyn`), so the
`forRange_inv` master invariant principle drives the loop. Mirrors `flash_body_split`.

The three pieces are transcribed concretely (the per-statement op-eval recipes above
encode the exact `Op`/`Broadcast`/dtype terms); `aftBody_split` is checked by `rfl`. -/

namespace AftFoundation

open VeriTile.Triton

/-- The 22 lowered loop-body statements (statements 0–21 of the `forRange` body),
matching the recipe op-eval lemmas `aft_*`. -/
def aftLoopBody : List Stmt :=
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

/-- The 2 lowered postLoop statements (`body.drop 23`): the `acc = acc / l_i[:,None]`
finalize and the masked `tl.store(O_block_ptr, acc, mask=(offs_m<128)&(offs_k<96))`. -/
def aftPostLoop (Out : RegionName) : List Stmt :=
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

end AftFoundation

namespace VeriTile.Bench.TritonBenchG.AttnFwdTriton

open VeriTile.Triton

set_option maxRecDepth 8000 in
/-- **`aftBody_split`** — the lowered AFT body decomposes as
`take 22 ++ (forRange "start_n" 0 128 64 aftLoopBody :: drop 23)`. The static
`forRange` (NOT `forRangeDyn`) sits at index 22; the 2 postLoop statements follow.
Pure structural identity, checked by `rfl`. -/
theorem aftBody_split
    (Q K V QScale KScale Out : RegionName) :
    (attn_fwd_triton_surface Q K V QScale KScale Out
        65536 16384 128 1 65536 16384 128 1 65536 16384 128 1 65536 16384 128 1
        2 4 128 128 128 64 128 96 3).toAlgKernel.body
      = (attn_fwd_triton_surface Q K V QScale KScale Out
          65536 16384 128 1 65536 16384 128 1 65536 16384 128 1 65536 16384 128 1
          2 4 128 128 128 64 128 96 3).toAlgKernel.body.take 22
        ++ (Stmt.forRange "start_n" 0 128 64 AftFoundation.aftLoopBody
            :: (attn_fwd_triton_surface Q K V QScale KScale Out
                65536 16384 128 1 65536 16384 128 1 65536 16384 128 1 65536 16384 128 1
                2 4 128 128 128 64 128 96 3).toAlgKernel.body.drop 23) :=
  rfl

/-! ## FOUNDATION Part 4 — `aftPreLoop` AST + `aftInvariant` + `aftPreLoop_eval`

`aftPreLoop` is the 22-statement deterministic prefix (`= body.take 22`). The loop
invariant `aftInvariant` binds the running registers after `c` blocks (counter
`i = c·64`): `m_i`/`l_i`/`acc` to the three components of the ⊥-seeded `aftStateBot`
(running max = `aftRunningMax`), together with the static index vectors, the loaded
`q`/`q_scale`, and the four streamed pointer tiles (`K_ptrs`/`K_scale_ptr`/`V_ptrs`
advanced by `i`, `O_block_ptr`/`Q_ptrs`/`Q_scale_ptr` fixed). `aftPreLoop_eval`
steps the prefix to a state satisfying `aftInvariant … 0`. -/

namespace AftFoundation

open VeriTile.Triton

/-- The 22 lowered preLoop statements of the Python-shape AFT kernel (`= body.take 22`). -/
def aftPreLoop (Q K V QScale KScale Out : RegionName) : List Stmt :=
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

/-- **PreLoop head** — statements 0–10 of `aftPreLoop` (the scalar index/offset
setup: program ids, `off_z`/`off_h`, `qvk_offset`/`vk_offset`, the two scale
offsets, and the three index vectors `offs_m`/`offs_n`/`offs_k`). No pointer or
load statements; resolves entirely by scalar `setReg`-peeling. -/
def aftPreLoopHead : List Stmt :=
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

/-- **PreLoop tail** — statements 11–21 of `aftPreLoop` (the four streamed pointer
tiles `Q_ptrs`/`Q_scale_ptr`/`K_ptrs`/`K_scale_ptr`/`V_ptrs`/`O_block_ptr`, the
running-state seeds `m_i`/`l_i`/`acc`, and the masked `q` / scalar `q_scale`
loads). Steps from a state where the head registers are already bound. -/
def aftPreLoopTail (Q K V QScale KScale Out : RegionName) : List Stmt :=
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

/-- `aftPreLoop` decomposes as `aftPreLoopHead ++ aftPreLoopTail`. Checked by `rfl`. -/
theorem aftPreLoop_eq_head_tail (Q K V QScale KScale Out : RegionName) :
    aftPreLoop Q K V QScale KScale Out
      = aftPreLoopHead ++ aftPreLoopTail Q K V QScale KScale Out :=
  rfl

end AftFoundation

namespace VeriTile.Bench.TritonBenchG.AttnFwdTriton

open VeriTile.Triton

/-- The loaded (masked) query tile the preLoop binds to `q`: lane `(i, e)` reads
`Q[baseOffset + (qStart + i)·128 + e]` when `qStart + i < 128 ∧ e < 96`, else the
zero `undef` cell. Carried by the invariant so the step lemma threads it through
the per-block `q·k` dot. -/
noncomputable def qLoadedAFT (s0 : BlockState) (Q : RegionName) :
    Tile .real [128, 128] :=
  ⟨fun idx : TileIndex [128, 128] =>
    if (ComparableDType.nat.lt (s0.pids 0 * 128 + idx.1.val) 128
        && ComparableDType.nat.lt idx.2.1.val 96) then
      some (s0.readMem Q (baseOffsetAFT s0 + (s0.pids 0 * 128 + idx.1.val) * 128 + idx.2.1.val))
    else some (0 : ℝ)⟩

/-- The scalar `q_scale` the preLoop binds: `QScale[q_scale_offset + start_m]`. -/
noncomputable def qScaleAFT (s0 : BlockState) (QScale : RegionName) :
    Tile .real [] :=
  ⟨fun _ : TileIndex [] =>
    some (s0.readMem QScale (s0.pids 1 * ((128 + 128 - 1) / 128) + s0.pids 0))⟩

/-- `K_ptrs` after `c = i/64` blocks: lane `(e, j)` (head-channel `e`, block-key
`j`) addresses `K` at `baseOffset + e + j·128 + i·128`. The `+ i·128` accumulates
the `c` `BLOCK_N·HEAD_DIM = 64·128`-element pointer advances; at global key
`i + j` the cell is `K[baseOffset + (i + j)·128 + e] = kTileAFT (i+j) e`. -/
noncomputable def kPtrsAFT (s0 : BlockState) (K : RegionName) (i : Nat) :
    Tile .ptr [128, 64] :=
  ⟨fun idx : TileIndex [128, 64] =>
    (Region.cast K, baseOffsetAFT s0 + idx.1.val + idx.2.1.val * 128 + i * 128)⟩

/-- `V_ptrs` after `c = i/64` blocks: lane `(j, e)` (block-key `j`, head-channel
`e`) addresses `V` at `baseOffset + j·128 + e + i·128`; at global key `i + j` the
cell is `V[baseOffset + (i + j)·128 + e] = vTileAFT (i+j) e`. -/
noncomputable def vPtrsAFT (s0 : BlockState) (V : RegionName) (i : Nat) :
    Tile .ptr [64, 128] :=
  ⟨fun idx : TileIndex [64, 128] =>
    (Region.cast V, baseOffsetAFT s0 + idx.1.val * 128 + idx.2.1.val + i * 128)⟩

/-- `K_scale_ptr` after `c = i/64` blocks: scalar pointer to `KScale` at
`k_scale_offset + c`. -/
noncomputable def kScalePtrAFT (s0 : BlockState) (KScale : RegionName) (i : Nat) :
    Tile .ptr [] :=
  ⟨fun _ : TileIndex [] =>
    (Region.cast KScale, s0.pids 1 * ((128 + 64 - 1) / 64) + i / 64)⟩

/-- One-block advance of `K_ptrs`: the loop-body `ptrAdd … (64·128)` maps
`kPtrsAFT s0 K i` to `kPtrsAFT s0 K (i + 64)`. -/
theorem kPtrsAFT_succ (s0 : BlockState) (K : RegionName) (i : Nat) :
    Tile.ptrAdd Broadcast.scalarR (kPtrsAFT s0 K i)
        (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar 64) (Tile.scalar 128))
      = kPtrsAFT s0 K (i + 64) := by
  ext idx
  · rfl
  · simp only [kPtrsAFT, Tile.ptrAdd_data, Tile.bop_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR, Broadcast.leftIndex_nil,
      Broadcast.rightIndex_nil, NumericDType.nat_mul]
    ring

/-- One-block advance of `V_ptrs`: the loop-body `ptrAdd … (64·128)` maps
`vPtrsAFT s0 V i` to `vPtrsAFT s0 V (i + 64)`. -/
theorem vPtrsAFT_succ (s0 : BlockState) (V : RegionName) (i : Nat) :
    Tile.ptrAdd Broadcast.scalarR (vPtrsAFT s0 V i)
        (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar 64) (Tile.scalar 128))
      = vPtrsAFT s0 V (i + 64) := by
  ext idx
  · rfl
  · simp only [vPtrsAFT, Tile.ptrAdd_data, Tile.bop_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR, Broadcast.leftIndex_nil,
      Broadcast.rightIndex_nil, NumericDType.nat_mul]
    ring

/-- One-block advance of `K_scale_ptr`: the loop-body `ptrAdd … 1` maps
`kScalePtrAFT s0 KScale i` to `kScalePtrAFT s0 KScale (i + 64)` (when `i % 64 = 0`). -/
theorem kScalePtrAFT_succ (s0 : BlockState) (KScale : RegionName) (i : Nat) (hi : i % 64 = 0) :
    Tile.ptrAdd Broadcast.nil (kScalePtrAFT s0 KScale i) (Tile.scalar 1)
      = kScalePtrAFT s0 KScale (i + 64) := by
  ext idx
  · rfl
  · simp only [kScalePtrAFT, Tile.ptrAdd_data, Tile.scalar,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil, NumericDType.nat_add]
    have : (i + 64) / 64 = i / 64 + 1 := by omega
    rw [this]; omega

/-- **Loop invariant** for the AFT streaming loop (counter `i = c·64`, window
`hi_c = i`). Binds the running-state registers after `c` blocks to the ⊥-seeded
`aftStateBot`/`aftRunningMax` over the first `i` keys, per output row `r` (channel
`d` for `acc`), keyed by the per-key score scale `keyScale`. Also binds the static
index vectors (`offs_m`/`offs_n`/`offs_k`), the loaded `q`/`q_scale`, the program
ids, and the three streamed pointer tiles (`K_ptrs`/`K_scale_ptr`/`V_ptrs`
advanced by `i = c·64` with exact addresses — `kPtrsAFT`/`kScalePtrAFT`/
`vPtrsAFT`), and preserves `undef`/`mem`. -/
noncomputable def aftInvariant
    (Q K V QScale KScale Out : RegionName) (s0 : BlockState)
    (keyScale : Fin 128 → ℝ) (i : Nat) (s : BlockState) : Prop :=
  let qStart := qStartAFT s0
  let qT := qMaskedAFT s0 Q
  let kT := kTileAFT s0 K
  let vT := vMaskedAFT s0 V
  s.pids = s0.pids ∧ i % 64 = 0 ∧ i ≤ 128 ∧
  (s.regs .real [128] "m_i" = some ⟨fun r : TileIndex [128] =>
      aftRunningMax qT kT vT keyScale qStart i r.1 ⟨0, by norm_num⟩⟩) ∧
  (s.regs .real [128] "l_i" = some ⟨fun r : TileIndex [128] =>
      ((aftStateBotK qT kT vT keyScale qStart i r.1 ⟨0, by norm_num⟩).2.1 : ℝ)⟩) ∧
  (s.regs .real [128, 128] "acc" = some ⟨fun idx : TileIndex [128, 128] =>
      ((aftStateBotK qT kT vT keyScale qStart i idx.1 idx.2.1).2.2 : ℝ)⟩) ∧
  (s.regs .nat [128] "offs_m" = some (Tile.vec (fun r : Fin 128 => qStart + r.val))) ∧
  (s.regs .nat [64] "offs_n" = some (Tile.vec (fun j : Fin 64 => j.val))) ∧
  (s.regs .nat [128] "offs_k" = some (Tile.vec (fun e : Fin 128 => e.val))) ∧
  (s.regs .nat [] "start_m" = some (Tile.scalar (s0.pids 0))) ∧
  (s.regs .nat [] "off_hz" = some (Tile.scalar (s0.pids 1))) ∧
  (s.regs .real [128, 128] "q" = some (qLoadedAFT s0 Q)) ∧
  (s.regs .real [] "q_scale" = some (qScaleAFT s0 QScale)) ∧
  (s.regs .ptr [128, 64] "K_ptrs" = some (kPtrsAFT s0 K i)) ∧
  (s.regs .ptr [] "K_scale_ptr" = some (kScalePtrAFT s0 KScale i)) ∧
  (s.regs .ptr [64, 128] "V_ptrs" = some (vPtrsAFT s0 V i)) ∧
  (s.regs .ptr [128, 128] "O_block_ptr" = some ⟨fun idx : TileIndex [128, 128] =>
      (Region.cast Out, baseOffsetAFT s0 + (qStartAFT s0 + idx.1.val) * 128 + idx.2.1.val)⟩) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-- `qvk_offset` value for the AFT Python test shape: `(pid1/4)·65536 + (pid1%4)·16384`. -/
def qvkOffAFT (s : BlockState) : Nat :=
  s.pids 1 / 4 * 65536 + s.pids 1 % 4 * 16384

theorem qvkOffAFT_eq_baseOffset (s : BlockState) : qvkOffAFT s = baseOffsetAFT s := by
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **PreLoop head execution.** The 11 scalar/index statements (`aftPreLoopHead`)
step a clean state `s` to a state `s1` that binds `start_m`/`off_hz` (program
ids), the scalar offsets `qvk_offset`/`q_scale_offset`, and the three index
vectors `offs_m`/`offs_n`/`offs_k`, and preserves `pids`/`mem`/`undef`. These are
exactly the readbacks the tail (`aftPreLoopTail`) consumes. -/
theorem aftPreLoopHead_eval
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s1, stepStmts AftFoundation.aftPreLoopHead s = some s1
      ∧ s1.pids = s.pids ∧ s1.mem = s.mem ∧ (∀ rg o, s1.undef rg o = 0)
      ∧ s1.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s1.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 1))
      ∧ s1.regs .nat [] "qvk_offset" = some (Tile.scalar (qvkOffAFT s))
      ∧ s1.regs .nat [] "q_scale_offset"
          = some (Tile.scalar (s.pids 1 * ((128 + 128 - 1) / 128)))
      ∧ s1.regs .nat [] "k_scale_offset"
          = some (Tile.scalar (s.pids 1 * ((128 + 64 - 1) / 64)))
      ∧ s1.regs .nat [128] "offs_m" = some (Tile.vec (fun r : Fin 128 => s.pids 0 * 128 + r.val))
      ∧ s1.regs .nat [64] "offs_n" = some (Tile.vec (fun j : Fin 64 => j.val))
      ∧ s1.regs .nat [128] "offs_k" = some (Tile.vec (fun e : Fin 128 => e.val)) := by
  unfold AftFoundation.aftPreLoopHead
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
        = some (Tile.scalar (qvkOffAFT s)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 5: vk_offset = qvk_offset / 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "qvk_offset") (Op.constNat 128)) _
        = some (Tile.scalar (qvkOffAFT s / 128)) from by
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

/-- Predicate: statement `st` is an assign that does not target register `name`
(non-assign statements are excluded — `aftLoopBody` is all assigns). -/
def stmtNotAssign (name : RegName) : Stmt → Prop
  | .assign _ _ n' _ => n' ≠ name
  | _ => False

/-- A single `stepStmt` of an assign that does not target `name` preserves the
`(d, sh, name)` register. -/
theorem stepStmt_regs_frame {d : TileDType} {sh : TileShape} {name : RegName}
    {st : Stmt} {s s' : BlockState} (hna : stmtNotAssign name st)
    (h : stepStmt st s = some s') :
    s'.regs d sh name = s.regs d sh name := by
  cases st with
  | assign d' sh' n' e =>
    have hne : n' ≠ name := hna
    cases hv : evalOp e s with
    | none =>
      rw [show stepStmt (.assign d' sh' n' e) s = none from by
        simp only [stepStmt, hv]; rfl] at h
      exact absurd h (by simp)
    | some v =>
      rw [show stepStmt (.assign d' sh' n' e) s = some (s.setReg n' d' sh' v) from by
        simp only [stepStmt, hv]; rfl] at h
      rw [← Option.some.inj h]
      exact BlockState.setReg_ne_name s n' name d' d sh' sh v (Ne.symm hne)
  | _ => exact absurd hna (by simp [stmtNotAssign])

/-- Every statement of `aftLoopBody` leaves any register distinct from the body's
assigned names unassigned. The name-distinctness side condition is `by decide`. -/
theorem aftLoopBody_stmtNotAssign (name : RegName)
    (hne : name ≠ "start_n" ∧ name ≠ "k_mask" ∧ name ≠ "k" ∧ name ≠ "k_scale" ∧
      name ≠ "qk" ∧ name ≠ "mask" ∧ name ≠ "m_ij" ∧ name ≠ "p" ∧ name ≠ "l_ij" ∧
      name ≠ "alpha" ∧ name ≠ "l_i" ∧ name ≠ "acc" ∧ name ≠ "v" ∧ name ≠ "m_i" ∧
      name ≠ "K_ptrs" ∧ name ≠ "K_scale_ptr" ∧ name ≠ "V_ptrs") :
    ∀ st ∈ AftFoundation.aftLoopBody, stmtNotAssign name st := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := hne
  intro st hst
  simp only [AftFoundation.aftLoopBody, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h <;> rw [h] <;>
    simp only [stmtNotAssign] <;>
    first
      | exact h1.symm | exact h2.symm | exact h3.symm | exact h4.symm | exact h5.symm
      | exact h6.symm | exact h7.symm | exact h8.symm | exact h9.symm | exact h10.symm
      | exact h11.symm | exact h12.symm | exact h13.symm | exact h14.symm | exact h15.symm
      | exact h16.symm | exact h17.symm

/-- **Register frame for `stepStmts`.** If every statement in `body` leaves
`name` unassigned and the body steps successfully, the `(d, sh, name)` register is
unchanged. Used to carry the loop-invariant registers the body never touches
(`offs_k`/`start_m`/`off_hz`/`O_block_ptr`) through `aftLoopBody`. -/
theorem stepStmts_regs_frame {d : TileDType} {sh : TileShape} {name : RegName}
    {body : List Stmt} {s s' : BlockState}
    (hna : ∀ st ∈ body, stmtNotAssign name st)
    (h : stepStmts body s = some s') :
    s'.regs d sh name = s.regs d sh name := by
  induction body generalizing s with
  | nil => rw [stepStmts] at h; rw [← Option.some.inj h]
  | cons st rest ih =>
    rw [stepStmts] at h
    cases hst : stepStmt st s with
    | none => rw [hst] at h; exact absurd h (by simp)
    | some mid =>
      rw [hst] at h
      rw [ih (fun st' hst' => hna st' (List.mem_cons_of_mem _ hst')) h]
      exact stepStmt_regs_frame (hna st (List.mem_cons_self ..)) hst

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 100000 in
/-- **PreLoop tail execution.** The 11 pointer/seed/load statements
(`aftPreLoopTail`) step an abstract head-exit state `s1` (with the head readbacks
supplied as hypotheses) to the loop-entry state `s0`, exposing the seeds
`m_i`/`l_i`/`acc`, the index vectors, the program-id scalars, and the loaded
`q_scale`, preserving `pids`/`mem`/`undef`. Operating on an abstract `s1` (rather
than a 22-deep `setReg` literal) lets the `expandDim`/`ref` offset proofs fire on
clean `ne_name`/`same` readbacks. -/
theorem aftPreLoopTail_eval
    (s1 : BlockState) (Q K V QScale KScale Out : RegionName)
    (hundef : ∀ rg o, s1.undef rg o = 0)
    (hstartm : s1.regs .nat [] "start_m" = some (Tile.scalar (s1.pids 0)))
    (hoffhz : s1.regs .nat [] "off_hz" = some (Tile.scalar (s1.pids 1)))
    (hqvk : s1.regs .nat [] "qvk_offset" = some (Tile.scalar (qvkOffAFT s1)))
    (hqso : s1.regs .nat [] "q_scale_offset"
        = some (Tile.scalar (s1.pids 1 * ((128 + 128 - 1) / 128))))
    (hkso : s1.regs .nat [] "k_scale_offset"
        = some (Tile.scalar (s1.pids 1 * ((128 + 64 - 1) / 64))))
    (hoffsm : s1.regs .nat [128] "offs_m"
        = some (Tile.vec (fun r : Fin 128 => s1.pids 0 * 128 + r.val)))
    (hoffsn : s1.regs .nat [64] "offs_n" = some (Tile.vec (fun j : Fin 64 => j.val)))
    (hoffsk : s1.regs .nat [128] "offs_k" = some (Tile.vec (fun e : Fin 128 => e.val))) :
    ∃ s0, stepStmts (AftFoundation.aftPreLoopTail Q K V QScale KScale Out) s1 = some s0
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
      ∧ s0.regs .ptr [128, 64] "K_ptrs" = some ⟨fun idx : TileIndex [128, 64] =>
          (Region.cast K, qvkOffAFT s1 + idx.1.val + idx.2.1.val * 128)⟩
      ∧ s0.regs .ptr [] "K_scale_ptr" = some ⟨fun _ : TileIndex [] =>
          (Region.cast KScale, s1.pids 1 * ((128 + 64 - 1) / 64))⟩
      ∧ s0.regs .ptr [64, 128] "V_ptrs" = some ⟨fun idx : TileIndex [64, 128] =>
          (Region.cast V, qvkOffAFT s1 + idx.1.val * 128 + idx.2.1.val)⟩
      ∧ s0.regs .ptr [128, 128] "O_block_ptr" = some ⟨fun idx : TileIndex [128, 128] =>
          (Region.cast Out, qvkOffAFT s1 + (s1.pids 0 * 128 + idx.1.val) * 128 + idx.2.1.val)⟩
      ∧ s0.regs .real [128, 128] "q" = some ⟨fun idx : TileIndex [128, 128] =>
          if (ComparableDType.nat.lt (s1.pids 0 * 128 + idx.1.val) 128
              && ComparableDType.nat.lt idx.2.1.val 96) then
            some (s1.readMem Q (qvkOffAFT s1 + (s1.pids 0 * 128 + idx.1.val) * 128 + idx.2.1.val))
          else some (0 : ℝ)⟩
 := by
  unfold AftFoundation.aftPreLoopTail
  -- stmt 11: Q_ptrs = ptrAdd (ptrBase Q) (qvk + offs_m[:,None]*128 + offs_k[None,:]*1)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft_evalOp_ptrAdd_of Broadcast.scalarL (Op.ptrBase Q) _ _ _ _
      (aft_evalOp_ptrBase Q _)
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
    (aft_evalOp_ptrAdd_of Broadcast.nil (Op.ptrBase QScale) _ _ _ _
      (aft_evalOp_ptrBase QScale _)
      (show evalOp (Op.add .nat Broadcast.nil (Op.ref .nat [] "q_scale_offset") (Op.ref .nat [] "start_m")) _
          = some _ from by
        simp only [evalOp_add, evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, hqso, hstartm, Option.bind_some, Option.bind_eq_bind]
        rfl)))]
  -- stmt 13: K_ptrs = ptrAdd (ptrBase K) (qvk + offs_k[:,None] + offs_n[None,:]*128)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft_evalOp_ptrAdd_of Broadcast.scalarL (Op.ptrBase K) _ _ _ _
      (aft_evalOp_ptrBase K _)
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
    (aft_evalOp_ptrAdd_of Broadcast.nil (Op.ptrBase KScale) _ _ _ _
      (aft_evalOp_ptrBase KScale _)
      (show evalOp (Op.ref .nat [] "k_scale_offset") _ = some _ from by
        simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true]
        rw [hkso])))]
  -- stmt 15: V_ptrs = ptrAdd (ptrBase V) (qvk + offs_n[:,None]*128 + offs_k[None,:]*1)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft_evalOp_ptrAdd_of Broadcast.scalarL (Op.ptrBase V) _ _ _ _
      (aft_evalOp_ptrBase V _)
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
    (aft_evalOp_ptrAdd_of Broadcast.scalarL (Op.ptrBase Out) _ _ _ _
      (aft_evalOp_ptrBase Out _)
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
    (aft_evalOp_load_ptr_mask_of (Op.ref .ptr [128, 128] "Q_ptrs") _ _ _ _
      (by rw [evalOp_ref]; rfl)
      (show evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) (Op.constNat 128))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat 96)))) _
          = some _ from by
        rw [aft_evalOp_boolAnd, evalOp_lt]
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
    (aft_evalOp_load_ptr_none_of (Op.ref .ptr [] "Q_scale_ptr") _ _
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
  · -- K_ptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [Tile.ptrAdd_data, Tile.scalar_data, Tile.bop_data, Tile.expandDim_data,
      Tile.vec, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
      Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      TileShape.dropInsertedIndex, NumericDType.nat_add, NumericDType.nat_mul, Prod.mk.injEq,
      Nat.zero_add, Nat.mul_one, and_self]
  · -- K_scale_ptr
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
    refine congrArg some (Tile.ext (fun _ => ?_))
    simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex_nil,
      Broadcast.rightIndex_nil, NumericDType.nat_add, Nat.zero_add]
  · -- V_ptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [Tile.ptrAdd_data, Tile.scalar_data, Tile.bop_data, Tile.expandDim_data,
      Tile.vec, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
      Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      TileShape.dropInsertedIndex, NumericDType.nat_add, NumericDType.nat_mul, Prod.mk.injEq,
      Nat.zero_add, Nat.mul_one, and_self]
  · -- O_block_ptr
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [Tile.ptrAdd_data, Tile.scalar_data, Tile.bop_data, Tile.expandDim_data,
      Tile.vec, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
      Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      TileShape.dropInsertedIndex, NumericDType.nat_add, NumericDType.nat_mul, Prod.mk.injEq,
      Nat.zero_add, Nat.mul_one, and_self]
  · -- q (masked load)
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
    refine congrArg some (Tile.ext (fun idx => ?_))
    obtain ⟨r, e, ⟨⟩⟩ := idx
    simp only [Tile.bop_data, Tile.cop_data, Tile.expandDim_data, Tile.vec, Tile.scalar_data,
      Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
      Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR, TileShape.dropInsertedIndex]
    by_cases hk : (ComparableDType.nat.lt (s1.pids 0 * 128 + r.val) 128)
        && (ComparableDType.nat.lt e.val 96)
    · rw [if_pos hk, if_pos hk]
      simp only [BlockState.readMem, BlockState.setReg_mem, castTile_self,
        Tile.ptrAdd_data, Tile.scalar_data, Tile.bop_data, Tile.expandDim_data, Tile.vec,
        Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
        TileShape.dropInsertedIndex, Region.cast, NumericDType.nat_add, NumericDType.nat_mul,
        Nat.zero_add, Nat.mul_one]
    · rw [if_neg hk, if_neg hk]
      simp only [BlockState.setReg_undef, hundef]

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

Proved by composing `aftPreLoopHead_eval` (statements 0–10) and
`aftPreLoopTail_eval` (statements 11–21) through `stepStmts.append_some`, keyed on
`aftPreLoop_eq_head_tail` — avoiding the heartbeat timeout the monolithic
22-statement chain hit at the final `rfl` over the deep `setReg` literal.

Modeling note on `l_i`: the kernel seeds `l_i = tl.zeros + 1.0 = 1.0`, faithfully
exposed here as `full 1.0`. The ⊥-seed foundation (`aftStateBot`, anchored at
`l = 0`) absorbs this `+1.0` on the first block — `α = realExp2(⊥ − m₀) = 0`, so
`l_i ← 1.0·0 + l_ij = l_ij`, matching `aftStateBot` from block 1 on — so the
running-state invariant `aftInvariant` (which binds `l_i` to `aftStateBot`) is
established only after the first loop iteration, not at the raw seed; this lemma
exposes the genuine seed register, deferring the invariant rebind to the step lemma. -/
theorem aftPreLoop_eval
    (s : BlockState) (Q K V QScale KScale Out : RegionName)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (AftFoundation.aftPreLoop Q K V QScale KScale Out) s = some s0
      ∧ s0.pids = s.pids ∧ s0.mem = s.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 1))
      ∧ s0.regs .real [128] "m_i" = some ⟨fun _ : TileIndex [128] => (⊥ : WithBot ℝ)⟩
      ∧ s0.regs .real [128] "l_i" = some ⟨fun _ : TileIndex [128] => (some (1 : ℝ) : WithBot ℝ)⟩
      ∧ s0.regs .real [128, 128] "acc" = some ⟨fun _ : TileIndex [128, 128] => (some (0 : ℝ) : WithBot ℝ)⟩
      ∧ s0.regs .nat [128] "offs_m" = some (Tile.vec (fun r : Fin 128 => s.pids 0 * 128 + r.val))
      ∧ s0.regs .nat [64] "offs_n" = some (Tile.vec (fun j : Fin 64 => j.val))
      ∧ s0.regs .nat [128] "offs_k" = some (Tile.vec (fun e : Fin 128 => e.val))
      ∧ s0.regs .real [] "q_scale" = some (qScaleAFT s QScale)
      ∧ s0.regs .ptr [128, 64] "K_ptrs" = some (kPtrsAFT s K 0)
      ∧ s0.regs .ptr [] "K_scale_ptr" = some (kScalePtrAFT s KScale 0)
      ∧ s0.regs .ptr [64, 128] "V_ptrs" = some (vPtrsAFT s V 0)
      ∧ s0.regs .real [128, 128] "q" = some (qLoadedAFT s Q)
      ∧ s0.regs .ptr [128, 128] "O_block_ptr" = some ⟨fun idx : TileIndex [128, 128] =>
          (Region.cast Out, baseOffsetAFT s + (s.pids 0 * 128 + idx.1.val) * 128 + idx.2.1.val)⟩ := by
  rw [AftFoundation.aftPreLoop_eq_head_tail]
  obtain ⟨s1, hHead, hpids1, hmem1, hundef1, hstartm1, hoffhz1, hqvk1, hqso1, hkso1,
    hoffsm1, hoffsn1, hoffsk1⟩ := aftPreLoopHead_eval s hundef
  rw [stepStmts.append_some hHead]
  have hstartm1' : s1.regs .nat [] "start_m" = some (Tile.scalar (s1.pids 0)) := by
    rw [hpids1]; exact hstartm1
  have hoffhz1' : s1.regs .nat [] "off_hz" = some (Tile.scalar (s1.pids 1)) := by
    rw [hpids1]; exact hoffhz1
  have hqvk1' : s1.regs .nat [] "qvk_offset" = some (Tile.scalar (qvkOffAFT s1)) := by
    rw [show qvkOffAFT s1 = qvkOffAFT s from by simp only [qvkOffAFT, hpids1]]; exact hqvk1
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
    hoffsm0, hoffsn0, hoffsk0, hqscale0, hKp0, hKsp0, hVp0, hOp0, hq0⟩ :=
    aftPreLoopTail_eval s1 Q K V QScale KScale Out hundef1
      hstartm1' hoffhz1' hqvk1' hqso1' hkso1' hoffsm1' hoffsn1 hoffsk1
  have hbase1 : qvkOffAFT s1 = baseOffsetAFT s := by simp only [qvkOffAFT, baseOffsetAFT, hpids1]
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
    simp only [qScaleAFT]
    refine congrArg some ?_
    ext idx
    simp only [BlockState.readMem, hmem1, hpids1]
  · rw [hKp0]; refine congrArg some (Tile.ext (fun idx => Prod.ext rfl ?_))
    simp only [kPtrsAFT, hbase1, Nat.zero_mul, Nat.add_zero]
  · rw [hKsp0]; refine congrArg some (Tile.ext (fun _ => Prod.ext rfl ?_))
    simp only [kScalePtrAFT, hpids1, Nat.zero_div, Nat.add_zero]
  · rw [hVp0]; refine congrArg some (Tile.ext (fun idx => Prod.ext rfl ?_))
    simp only [vPtrsAFT, hbase1, Nat.zero_mul, Nat.add_zero]
  · -- q
    rw [hq0]; refine congrArg some (Tile.ext (fun idx => ?_))
    obtain ⟨r, e, ⟨⟩⟩ := idx
    simp only [qLoadedAFT, hpids1, hbase1, BlockState.readMem, hmem1]
  · -- O_block_ptr
    rw [hOp0]; refine congrArg some (Tile.ext (fun idx => Prod.ext rfl ?_))
    simp only [hbase1, hpids1, Nat.zero_mul, Nat.add_zero]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Score cell (kernel-faithful).** The raw `qk` cell the loop body produces at
output row `i`, block-key `jL` (block `c`), `qkRawT = ((q · k)·q_scale)·k_scale`,
equals `q_scale · k_scale · Σ_e (qLoadedAFT i e) · kTileAFT (c·64+jL) e`. The
loaded `q` carries the head-active mask (`q[e≥96] = 0`), so the `HEAD_DIM = 128`
contraction is genuinely the `HEAD_ACTIVE = 96` dot. `ktile` is the loop's masked
`K_ptrs` load; here the kept-lane address `base + e + (c·64+jL)·128` is supplied as
a hypothesis (`hk`), reading `kTileAFT (c·64+jL) e`. -/
theorem aft_score_cell (s0 : BlockState) (Q K : RegionName) (qsc ksc : ℝ) (c : Nat)
    (i : Fin 128) (jL : Fin 64) (hjL : c * 64 + jL.val < 128)
    (qtile : Tile .real [128, 128]) (ktile : Tile .real [128, 64])
    (hq : qtile = qLoadedAFT s0 Q)
    (hk : ∀ idx : TileIndex [128, 64],
        ktile.data idx = some (s0.readMem K
          (baseOffsetAFT s0 + idx.1.val + (c * 64 + idx.2.1.val) * 128))) :
    (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (⟨fun i => (Tile.dot [] qtile ktile).data i⟩ : Tile .real [128, 64])
          (Tile.scalar (some qsc)))
        (Tile.scalar (some ksc))).data (i, jL, PUnit.unit)
      = some (qsc * ksc * Finset.univ.sum (fun e : Fin 128 =>
          ((qLoadedAFT s0 Q).data (i, e, PUnit.unit)).unbotD 0
            * kTileAFT s0 K (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit))) := by
  have hdot : (Tile.dot [] qtile ktile).data (i, jL, PUnit.unit)
      = some (Finset.univ.sum (fun e : Fin 128 =>
          ((qLoadedAFT s0 Q).data (i, e, PUnit.unit)).unbotD 0
            * kTileAFT s0 K (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit))) := by
    rw [Tile.dot_nil_data]
    rw [show (@Finset.sum (Fin 128) (WithBot ℝ) _ Finset.univ
          (fun e => Option.map₂ (· * ·) (qtile.data (i, e, PUnit.unit)) (ktile.data (e, jL, PUnit.unit))))
        = @Finset.sum (Fin 128) (WithBot ℝ) _ Finset.univ
          (fun e => (some (((qLoadedAFT s0 Q).data (i, e, PUnit.unit)).unbotD 0
              * kTileAFT s0 K (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit)) : WithBot ℝ))
        from Finset.sum_congr rfl (fun e _ => by
          rw [hq, hk (e, jL, PUnit.unit)]
          -- qLoadedAFT cell is `some _`, so unbotD recovers it
          rcases hqe : (qLoadedAFT s0 Q).data (i, e, PUnit.unit) with _ | qv
          · -- impossible: qLoadedAFT is always `some`
            exfalso; simp only [qLoadedAFT] at hqe; split at hqe <;> exact absurd hqe (by simp)
          · simp only [Option.map₂, Option.bind, Option.map]
            refine congrArg some ?_
            rw [show WithBot.unbotD 0 (some qv) = qv from rfl,
              show kTileAFT s0 K (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit)
                  = s0.readMem K (baseOffsetAFT s0 + e.val + (c * 64 + jL.val) * 128) from by
              simp only [kTileAFT]; congr 1; ring] )]
    rw [WithBot.sum_someTerm_eq_some]
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Broadcast.scalarR,
    Tile.scalar_data, NumericDType.mul, hdot]
  refine congrArg some ?_
  simp only [WithBot.realMul, Option.map₂, Option.bind, Option.map]
  ring


/-! ## FOUNDATION Part 5 — loop-body head/tail split + `aftLoopBody_steps`

The 22-statement `aftLoopBody` is split into a **head** (statements 0–10: the
`start_n`/`k_mask`/`k`/`k_scale`/`qk`-dot/`mask`/`where qk`/`m_ij`/`qk`-sub/`p`-exp2/
`p`-where chain that builds the masked block weights) and a **tail** (statements
11–21: `l_ij`/`alpha`/`l_i`/`acc`-rescale/`v`-load/`p`-fp16/`acc`-dot/`m_i`-carry +
the three pointer advances). Each half steps under `set_option maxHeartbeats
1600000` to dodge the 4M-heartbeat ceiling, composed via `stepStmts.append_some`.
Mirrors flash's `flashLoopBody_steps`. -/

open VeriTile.Triton

/-- **Loop-body head** — statements 0–10 of `aftLoopBody` (`= aftLoopBody.take 11`). -/
def aftLoopBodyHead : List Stmt := List.take 11 AftFoundation.aftLoopBody

/-- **Loop-body tail** — statements 11–21 of `aftLoopBody` (`= aftLoopBody.drop 11`). -/
def aftLoopBodyTail : List Stmt := List.drop 11 AftFoundation.aftLoopBody

/-- `aftLoopBody = aftLoopBodyHead ++ aftLoopBodyTail`. Checked by `rfl`. -/
theorem aftLoopBody_eq_head_tail :
    AftFoundation.aftLoopBody = aftLoopBodyHead ++ aftLoopBodyTail := rfl

/-- Explicit cons-form of `aftLoopBodyHead` (the 11 head statements), so the step
chain rewrites fire on a literal list rather than a `List.take` thunk. -/
theorem aftLoopBodyHead_eq :
    aftLoopBodyHead =
    [ Stmt.assign .nat [] "start_n" (Op.ref .nat [] "start_n"),
      Stmt.assign .bool [128, 64] "k_mask"
        (Op.boolAnd (Broadcast.consL (Broadcast.consR Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "offs_n"))
            (Op.sub .nat Broadcast.nil (Op.constNat 128) (Op.ref .nat [] "start_n")))
          (Op.expandDim ⟨1, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat 96)))),
      Stmt.assign .real [128, 64] "k"
        (Op.load .real (.ptr (.ref .ptr [128, 64] "K_ptrs")) (.mask (.ref .bool [128, 64] "k_mask"))),
      Stmt.assign .real [] "k_scale"
        (Op.load .real (.ptr (.ref .ptr [] "K_scale_ptr")) .none),
      Stmt.assign .real [128, 64] "qk"
        (Op.mul .real Broadcast.scalarR
          (Op.mul .real Broadcast.scalarR
            (Op.castFloat FloatDType.real FloatDType.real
              (Op.dot (batch := []) (Op.ref .real [128, 128] "q") (Op.ref .real [128, 64] "k")))
            (Op.ref .real [] "q_scale"))
          (Op.ref .real [] "k_scale")),
      Stmt.assign .bool [128, 64] "mask"
        (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "offs_n")))),
      Stmt.assign .real [128, 64] "qk"
        (Op.where (Op.ref .bool [128, 64] "mask")
          (Op.ref .real [128, 64] "qk")
          (Op.broadcast (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1000000.0)) [128, 64])),
      Stmt.assign .real [128] "m_ij"
        (Op.where
          (Op.gt .real (Broadcast.consSame Broadcast.nil)
            (Op.ref .real [128] "m_i")
            (Op.reduceMax (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false
              (Op.ref .real [128, 64] "qk")))
          (Op.ref .real [128] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false
            (Op.ref .real [128, 64] "qk"))),
      Stmt.assign .real [128, 64] "qk"
        (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .real [128, 64] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "m_ij"))),
      Stmt.assign .real [128, 64] "p" (Op.exp2 (Op.ref .real [128, 64] "qk")),
      Stmt.assign .real [128, 64] "p"
        (Op.where (Op.ref .bool [128, 64] "mask")
          (Op.ref .real [128, 64] "p") (Op.broadcast (Op.const 0.0) [128, 64])) ] := rfl

/-- Explicit cons-form of `aftLoopBodyTail` (the 11 tail statements). -/
theorem aftLoopBodyTail_eq :
    aftLoopBodyTail =
    [ Stmt.assign .real [128] "l_ij"
        (Op.reduceSum (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false (Op.ref .real [128, 64] "p")),
      Stmt.assign .real [128] "alpha"
        (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [128] "m_i") (Op.ref .real [128] "m_ij"))),
      Stmt.assign .real [128] "l_i"
        (Op.add .real (Broadcast.consSame Broadcast.nil)
          (Op.mul .real (Broadcast.consSame Broadcast.nil)
            (Op.ref .real [128] "l_i") (Op.ref .real [128] "alpha"))
          (Op.ref .real [128] "l_ij")),
      Stmt.assign .real [128, 128] "acc"
        (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .real [128, 128] "acc") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "alpha"))),
      Stmt.assign .real [64, 128] "v"
        (Op.load .real (.ptr (.ref .ptr [64, 128] "V_ptrs"))
          (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [64] "offs_n"))
              (Op.sub .nat Broadcast.nil (Op.constNat 128) (Op.ref .nat [] "start_n")))
            (Op.expandDim ⟨0, by simp⟩
              (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat 96)))))),
      Stmt.assign .fp16 [128, 64] "p"
        (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [128, 64] "p")),
      Stmt.assign .real [128, 128] "acc"
        (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [128, 128] "acc")
          (Op.dot (batch := [])
            (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref .fp16 [128, 64] "p"))
            (Op.ref .real [64, 128] "v"))),
      Stmt.assign .real [128] "m_i" (Op.ref .real [128] "m_ij"),
      Stmt.assign .ptr [128, 64] "K_ptrs"
        (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [128, 64] "K_ptrs")
          (Op.mul .nat Broadcast.nil (Op.constNat 64) (Op.constNat 128))),
      Stmt.assign .ptr [] "K_scale_ptr"
        (Op.ptrAdd Broadcast.nil (Op.ref .ptr [] "K_scale_ptr") (Op.constNat 1)),
      Stmt.assign .ptr [64, 128] "V_ptrs"
        (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [64, 128] "V_ptrs")
          (Op.mul .nat Broadcast.nil (Op.constNat 64) (Op.constNat 128))) ] := rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Loop-body head execution chain.** The 11 head statements step a state `sin`
(with `start_n = SN`, the index vectors `offs_m`/`offs_n`, the loaded `q`/`q_scale`,
the running `m_i`, and the streamed `K_ptrs`/`K_scale_ptr` pointer tiles) to a state
`sH`, exposing the masked block weights `p` (zeroed outside the causal mask), the
causal `mask`, the new running max `m_ij`, plus the preserved tail-consumed registers
`m_i`/`l_i`/`acc`/`offs_n`/`offs_m`/`V_ptrs`/`K_ptrs`/`K_scale_ptr`. Threaded through
`stepStmts.cons_some` via the banked `aft_*` recipes. -/
theorem aftLoopBodyHead_steps
    (sin : BlockState) (SN : Nat)
    (offsm : Tile .nat [128]) (offsn : Tile .nat [64])
    (qtile : Tile .real [128, 128]) (qsc ksc : Tile .real [])
    (ktile : Tile .real [128, 64]) (kmaskT : Tile .bool [128, 64])
    (mtile litile : Tile .real [128]) (acctile : Tile .real [128, 128])
    (Kptrs : Tile .ptr [128, 64]) (Ksp : Tile .ptr []) (Vptrs : Tile .ptr [64, 128])
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoffsm : sin.regs .nat [128] "offs_m" = some offsm)
    (hoffsn : sin.regs .nat [64] "offs_n" = some offsn)
    (hq : sin.regs .real [128, 128] "q" = some qtile)
    (hqsc : sin.regs .real [] "q_scale" = some qsc)
    (hmi : sin.regs .real [128] "m_i" = some mtile)
    (hli : sin.regs .real [128] "l_i" = some litile)
    (hacc : sin.regs .real [128, 128] "acc" = some acctile)
    (hKp : sin.regs .ptr [128, 64] "K_ptrs" = some Kptrs)
    (hKsp : sin.regs .ptr [] "K_scale_ptr" = some Ksp)
    (hVp : sin.regs .ptr [64, 128] "V_ptrs" = some Vptrs)
    (hkmask : ∀ idx : TileIndex [128, 64],
      kmaskT.data idx = ((ComparableDType.nat.lt (offsn.data (idx.2.1, PUnit.unit)) (128 - SN))
        && (ComparableDType.nat.lt idx.1.val 96)))
    (hkload : ∀ idx : TileIndex [128, 64],
      ktile.data idx = (if kmaskT.data idx then some (sin.readMem (Kptrs.data idx).1 (Kptrs.data idx).2)
        else some (sin.undef (Kptrs.data idx).1 (Kptrs.data idx).2)))
    (hkscval : ksc = ⟨fun _ : TileIndex [] => some (sin.readMem (Ksp.data PUnit.unit).1 (Ksp.data PUnit.unit).2)⟩)
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ sH, stepStmts aftLoopBodyHead sin = some sH
      ∧ sH.pids = sin.pids ∧ sH.mem = sin.mem ∧ (∀ rg o, sH.undef rg o = 0)
      ∧ sH.regs .nat [] "start_n" = some (Tile.scalar SN)
      ∧ sH.regs .nat [128] "offs_m" = some offsm
      ∧ sH.regs .nat [64] "offs_n" = some offsn
      ∧ sH.regs .real [128, 128] "q" = some qtile
      ∧ sH.regs .real [] "q_scale" = some qsc
      ∧ sH.regs .real [128] "m_i" = some mtile
      ∧ sH.regs .real [128] "l_i" = some litile
      ∧ sH.regs .real [128, 128] "acc" = some acctile
      ∧ sH.regs .ptr [128, 64] "K_ptrs" = some Kptrs
      ∧ sH.regs .ptr [] "K_scale_ptr" = some Ksp
      ∧ sH.regs .ptr [64, 128] "V_ptrs" = some Vptrs
      ∧ ∃ (maskT : Tile .bool [128, 64]) (qkRawT qk6T : Tile .real [128, 64])
            (rmaxT mijT : Tile .real [128]) (pT : Tile .real [128, 64]),
          (∀ idx : TileIndex [128, 64],
            maskT.data idx = ComparableDType.nat.ge (offsm.data (idx.1, PUnit.unit))
              (SN + offsn.data (idx.2.1, PUnit.unit)))
          ∧ qkRawT = Tile.bop NumericDType.real.mul Broadcast.scalarR
              (Tile.bop NumericDType.real.mul Broadcast.scalarR
                ⟨fun i => (Tile.dot [] qtile ktile).data i⟩ qsc) ksc
          ∧ (∀ idx : TileIndex [128, 64],
            qk6T.data idx = if maskT.data idx then qkRawT.data idx
              else WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ)))
          ∧ Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [128, 64].length) qk6T = some rmaxT
          ∧ mijT = Tile.select
              (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT
          ∧ (∀ idx : TileIndex [128, 64],
            pT.data idx = if maskT.data idx
              then WithBot.realExp2 (WithBot.realSub (qk6T.data idx) (mijT.data (idx.1, PUnit.unit)))
              else (some (0.0 : ℝ) : WithBot ℝ))
          ∧ sH.regs .bool [128, 64] "mask" = some maskT
          ∧ sH.regs .real [128] "m_ij" = some mijT
          ∧ sH.regs .real [128, 64] "p" = some pT := by
  -- the qk-raw tile (q·k * q_scale * k_scale)
  set qkRawT : Tile .real [128, 64] := Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        ⟨fun i => (Tile.dot [] qtile ktile).data i⟩ qsc) ksc with hqkRaw
  -- the causal mask tile
  set maskT : Tile .bool [128, 64] := ⟨fun idx : TileIndex [128, 64] =>
      ComparableDType.nat.ge (offsm.data (idx.1, PUnit.unit)) (SN + offsn.data (idx.2.1, PUnit.unit))⟩ with hmaskT
  -- the qk tile after the where-mask (stmt 6)
  set qk6T : Tile .real [128, 64] := ⟨fun idx : TileIndex [128, 64] =>
      if maskT.data idx then qkRawT.data idx
      else WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ))⟩ with hqk6T
  -- reduceMax exists
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [128, 64].length) qk6T = some t :=
    ⟨_, aft_reduceMaxDrop1_some qk6T⟩
  set mijT : Tile .real [128] := Tile.select
      (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT with hmij
  -- qk after the max-shift (stmt 8) and p after exp2 + where-zero (stmts 9, 10)
  set qk8T : Tile .real [128, 64] := Tile.bop NumericDType.real.sub
      (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qk6T (Tile.expandDim ⟨1, by simp⟩ mijT) with hqk8
  set p9T : Tile .real [128, 64] := Tile.uop WithBot.realExp2 qk8T with hp9
  set pT : Tile .real [128, 64] := ⟨fun idx : TileIndex [128, 64] =>
      if maskT.data idx then p9T.data idx else (some (0.0 : ℝ) : WithBot ℝ)⟩ with hpT
  rw [aftLoopBodyHead_eq]
  -- stmt 0: start_n = start_n (identity)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_n") sin = some (Tile.scalar SN) from by rw [evalOp_ref, hsn]))]
  -- stmt 1: k_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.boolAnd (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat 128) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat 96)))) _
        = some kmaskT from by
      rw [aft_kmask_eval _ SN 128 96 offsn
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffsn)
        (by rw [BlockState.setReg_same])]
      refine congrArg some ?_; ext idx; rw [hkmask idx]))]
  -- stmt 2: k = load(K_ptrs, k_mask)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (.ptr (.ref .ptr [128, 64] "K_ptrs")) (.mask (.ref .bool [128, 64] "k_mask"))) _
        = some ktile from by
      rw [aft_load_k_eval _ 128 64 "K_ptrs" "k_mask" Kptrs kmaskT
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hKp)
        (by rw [BlockState.setReg_same])]
      refine congrArg some ?_; ext idx
      simp only [BlockState.setReg_readMem, BlockState.setReg_undef]
      rw [hkload idx]; rfl))]
  -- stmt 3: k_scale = load(K_scale_ptr)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (.ptr (.ref .ptr [] "K_scale_ptr")) .none) _ = some ksc from by
      rw [aft_load_kscale_eval _ "K_scale_ptr" Ksp
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hKsp)]
      rw [hkscval]
      refine congrArg some ?_; ext idx
      simp only [BlockState.setReg_readMem]))]
  -- stmt 4: qk = castFloat(q·k) * q_scale * k_scale
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .real Broadcast.scalarR
        (Op.mul .real Broadcast.scalarR
          (Op.castFloat FloatDType.real FloatDType.real
            (Op.dot (batch := []) (Op.ref .real [128, 128] "q") (Op.ref .real [128, 64] "k")))
          (Op.ref .real [] "q_scale"))
        (Op.ref .real [] "k_scale")) _ = some qkRawT from by
      rw [aft_qk_dot_eval _ 128 64 128 qtile ktile qsc ksc
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq)
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact rfl)
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hqsc)
        (by rw [BlockState.setReg_same])]))]
  -- stmt 5: mask = offs_m[:,None] >= start_n + offs_n[None,:]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "offs_n")))) _ = some maskT from by
      rw [aft_mask_eval _ SN offsm offsn
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffsm)
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffsn)
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_same])]))]
  -- stmt 6: qk = where(mask, qk, -1e6)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.where (Op.ref .bool [128, 64] "mask")
        (Op.ref .real [128, 64] "qk")
        (Op.broadcast (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1000000.0)) [128, 64])) _
        = some qk6T from by
      rw [aft_where_qk_eval _ maskT qkRawT
        (by rw [BlockState.setReg_same])
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_same])]))]
  -- stmt 7: m_ij = maximum(m_i, max(qk,1))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft_mij_eval _ mtile qk6T rmaxT
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmi)
      (by rw [BlockState.setReg_same])
      hrm))]
  -- stmt 8: qk = qk - m_ij[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft_qk_sub_eval _ (by simp) qk6T mijT
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_same])
      (by rw [BlockState.setReg_same])))]
  -- stmt 9: p = exp2(qk)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft_p_eval _ qk8T
      (by rw [BlockState.setReg_same])))]
  -- stmt 10: p = where(mask, p, 0)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft_p_mask_eval _ maskT p9T
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact rfl)
      (by rw [BlockState.setReg_same])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    maskT, qkRawT, qk6T, rmaxT, mijT, pT,
    (fun idx => rfl), rfl, (fun idx => rfl), hrm, rfl, ?_, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_pids]
  · funext rg o; simp [BlockState.setReg_mem]
  · intro rg o; simp [BlockState.setReg_undef, hundef]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, hsn]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hoffsm]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hoffsn]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hq]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hqsc]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hmi]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hli]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hacc]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hKp]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hKsp]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hVp]
  -- p cell readback
  · intro idx
    simp only [hpT, hp9, hqk8, Tile.uop_data, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.expandDim_data, TileShape.dropInsertedIndex, NumericDType.sub]
  -- mask reg
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_same]
  -- m_ij reg
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_same]
  -- p reg
  · rw [BlockState.setReg_same]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Loop-body tail execution chain.** The 11 tail statements step the head-exit
state `sH` (with the masked weights `p`, the causal `mask`, the new running max
`m_ij`, the old `m_i`/`l_i`/`acc`, the index vector `offs_n`, `start_n`, and the
streamed pointer tiles) to the final loop-body state `sF`, exposing the updated
running-state registers `m_i`/`l_i`/`acc` (the kernel's per-block rescale-and-add)
and the advanced `K_ptrs`/`K_scale_ptr`/`V_ptrs`. Threaded through
`stepStmts.cons_some` via the banked `aft_*` recipes. -/
theorem aftLoopBodyTail_steps
    (sH : BlockState) (SN : Nat)
    (offsn : Tile .nat [64])
    (ptile : Tile .real [128, 64]) (vtile : Tile .real [64, 128])
    (vmaskT : Tile .bool [64, 128])
    (mtile mijtile litile lijtile : Tile .real [128]) (acctile : Tile .real [128, 128])
    (Kptrs : Tile .ptr [128, 64]) (Ksp : Tile .ptr []) (Vptrs : Tile .ptr [64, 128])
    (hsn : sH.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoffsn : sH.regs .nat [64] "offs_n" = some offsn)
    (hp : sH.regs .real [128, 64] "p" = some ptile)
    (hmi : sH.regs .real [128] "m_i" = some mtile)
    (hmij : sH.regs .real [128] "m_ij" = some mijtile)
    (hli : sH.regs .real [128] "l_i" = some litile)
    (hacc : sH.regs .real [128, 128] "acc" = some acctile)
    (hKp : sH.regs .ptr [128, 64] "K_ptrs" = some Kptrs)
    (hKsp : sH.regs .ptr [] "K_scale_ptr" = some Ksp)
    (hVp : sH.regs .ptr [64, 128] "V_ptrs" = some Vptrs)
    (hvmask : ∀ idx : TileIndex [64, 128],
      vmaskT.data idx = ((ComparableDType.nat.lt (offsn.data (idx.1, PUnit.unit)) (128 - SN))
        && (ComparableDType.nat.lt idx.2.1.val 96)))
    (hvload : ∀ idx : TileIndex [64, 128],
      vtile.data idx = (if vmaskT.data idx then some (sH.readMem (Vptrs.data idx).1 (Vptrs.data idx).2)
        else some (sH.undef (Vptrs.data idx).1 (Vptrs.data idx).2)))
    (hundef : ∀ rg o, sH.undef rg o = 0) :
    ∃ sF, stepStmts aftLoopBodyTail sH = some sF
      ∧ sF.pids = sH.pids ∧ sF.mem = sH.mem ∧ (∀ rg o, sF.undef rg o = 0)
      ∧ ∃ (lijT alphaT : Tile .real [128]),
          lijT = Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [128, 64].length) ptile
          ∧ alphaT = Tile.uop WithBot.realExp2
              (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijtile)
          ∧ sF.regs .real [128] "m_i" = some mijtile
          ∧ sF.regs .real [128] "l_i" = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
              (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) litile alphaT) lijT)
          ∧ sF.regs .real [128, 128] "acc" = some (Tile.bop NumericDType.real.add
              (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
              (Tile.dot [] ⟨fun i => FloatDType.fp16.cast FloatDType.real
                (FloatDType.real.cast FloatDType.fp16 (ptile.data i))⟩ vtile))
          ∧ sF.regs .ptr [128, 64] "K_ptrs" = some (Tile.ptrAdd Broadcast.scalarR Kptrs
              (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar 64) (Tile.scalar 128)))
          ∧ sF.regs .ptr [] "K_scale_ptr" = some (Tile.ptrAdd Broadcast.nil Ksp (Tile.scalar 1))
          ∧ sF.regs .ptr [64, 128] "V_ptrs" = some (Tile.ptrAdd Broadcast.scalarR Vptrs
              (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar 64) (Tile.scalar 128))) := by
  set lijT : Tile .real [128] :=
    (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [128, 64].length) ptile : Tile .real [128]) with hlijT
  set alphaT : Tile .real [128] := Tile.uop WithBot.realExp2
      (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijtile) with halphaT
  -- the fp16-cast p tile
  set pf16T : Tile .fp16 [128, 64] := ⟨fun i => FloatDType.real.cast FloatDType.fp16 (ptile.data i)⟩ with hpf16T
  rw [aftLoopBodyTail_eq]
  -- stmt 11: l_ij = sum(p, 1)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.reduceSum (⟨1, by simp⟩ : Fin [128, 64].length) Bool.false
        (Op.ref .real [128, 64] "p")) sH = some lijT from aft_lij_eval sH ptile hp))]
  -- stmt 12: alpha = exp2(m_i - m_ij)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft_alpha_eval _ mtile mijtile
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmi)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmij)))]
  -- stmt 13: l_i = l_i * alpha + l_ij
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft_li_eval _ litile alphaT lijT
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hli)
      (by rw [BlockState.setReg_same])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)];
          exact BlockState.setReg_same _ _ _ _ _)))]
  -- stmt 14: acc = acc * alpha[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft_acc_rescale_eval _ (by simp) acctile alphaT
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hacc)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)];
          exact BlockState.setReg_same _ _ _ _ _)))]
  -- stmt 15: v = load(V_ptrs, mask)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (.ptr (.ref .ptr [64, 128] "V_ptrs"))
        (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [64] "offs_n"))
            (Op.sub .nat Broadcast.nil (Op.constNat 128) (Op.ref .nat [] "start_n")))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat 96)))))) _
        = some vtile from by
      rw [aft_evalOp_load_ptr_mask_of (Op.ref .ptr [64, 128] "V_ptrs") _ _ Vptrs vmaskT
        (by rw [evalOp_ref, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
              BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hVp)
        (show evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [64] "offs_n"))
              (Op.sub .nat Broadcast.nil (Op.constNat 128) (Op.ref .nat [] "start_n")))
            (Op.expandDim ⟨0, by simp⟩
              (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat 96)))) _
            = some vmaskT from by
          rw [aft_evalOp_boolAnd, evalOp_lt]
          erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
                    BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
                    BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
                    BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffsn),
            evalOp_expandDim]
          simp only [evalOp_lt, evalOp_arange, evalOp_constNat, evalOp_sub, evalOp_ref,
            BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            Option.bind_some, Option.bind_eq_bind, hsn]
          refine congrArg some ?_; ext idx
          rw [hvmask idx]
          simp only [Tile.bop_data, Tile.cop_data, Tile.expandDim_data, Tile.vec, Tile.scalar,
            Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
            Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
            Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
            Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
            TileShape.dropInsertedIndex, NumericDType.sub])]
      refine congrArg some ?_; ext idx
      simp only [BlockState.setReg_readMem, BlockState.setReg_undef]
      rw [hvload idx]; rfl))]
  -- stmt 16: p = p.to(fp16)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft_p_fp16_eval _ ptile
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hp)))]
  -- stmt 17: acc += dot(p.to(real), v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft_acc_eval _ 128 64 128
      (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
      pf16T vtile
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)];
          exact BlockState.setReg_same _ _ _ _ _)
      (by exact BlockState.setReg_same _ _ _ _ _)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)];
          exact BlockState.setReg_same _ _ _ _ _)))]
  -- stmt 18: m_i = m_ij
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft_mi_carry_eval _ mijtile
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmij)))]
  -- stmt 19: K_ptrs += 64*128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft_advance_ptr_eval _ 128 64 64 "K_ptrs" Kptrs
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hKp)))]
  -- stmt 20: K_scale_ptr += 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft_advance_kscale_eval _ "K_scale_ptr" Ksp
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hKsp)))]
  -- stmt 21: V_ptrs += 64*128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft_advance_ptr_eval _ 64 128 64 "V_ptrs" Vptrs
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hVp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, lijT, alphaT, rfl, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_pids]
  · funext rg o; simp [BlockState.setReg_mem]
  · intro rg o; simp [BlockState.setReg_undef, hundef]
  -- m_i = m_ij
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_same]
  -- l_i
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_same]
  -- acc
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_same]
  -- K_ptrs
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_same]
  -- K_scale_ptr
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_same]
  -- V_ptrs
  · rw [BlockState.setReg_same]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Loop-body execution chain** (`aftLoopBody`, head ++ tail). Steps the
iteration-entry state `sin` to the final state `sF`, exposing the kernel's per-block
online-softmax update: the new running max `m_ij` (= `m_i ⊔ blockMax`), the rescaled
denominator `l_i` and accumulator `acc`, plus the advanced `K_ptrs`/`K_scale_ptr`/
`V_ptrs`. The masked block weights `p` (cell `if causal then exp2(qk−m_ij) else 0`),
the score tile `qkRaw` (= `q·k·q_scale·k_scale`), and the value tile `v` are exposed
symbolically for the math bridge. Composes `aftLoopBodyHead_steps` and
`aftLoopBodyTail_steps` via `stepStmts.append_some`. -/
theorem aftLoopBody_steps
    (sin : BlockState) (SN : Nat)
    (offsm : Tile .nat [128]) (offsn : Tile .nat [64])
    (qtile : Tile .real [128, 128]) (qsc ksc : Tile .real [])
    (ktile : Tile .real [128, 64]) (kmaskT : Tile .bool [128, 64])
    (vtile : Tile .real [64, 128]) (vmaskT : Tile .bool [64, 128])
    (mtile litile : Tile .real [128]) (acctile : Tile .real [128, 128])
    (Kptrs : Tile .ptr [128, 64]) (Ksp : Tile .ptr []) (Vptrs : Tile .ptr [64, 128])
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoffsm : sin.regs .nat [128] "offs_m" = some offsm)
    (hoffsn : sin.regs .nat [64] "offs_n" = some offsn)
    (hq : sin.regs .real [128, 128] "q" = some qtile)
    (hqsc : sin.regs .real [] "q_scale" = some qsc)
    (hmi : sin.regs .real [128] "m_i" = some mtile)
    (hli : sin.regs .real [128] "l_i" = some litile)
    (hacc : sin.regs .real [128, 128] "acc" = some acctile)
    (hKp : sin.regs .ptr [128, 64] "K_ptrs" = some Kptrs)
    (hKsp : sin.regs .ptr [] "K_scale_ptr" = some Ksp)
    (hVp : sin.regs .ptr [64, 128] "V_ptrs" = some Vptrs)
    (hkmask : ∀ idx : TileIndex [128, 64],
      kmaskT.data idx = ((ComparableDType.nat.lt (offsn.data (idx.2.1, PUnit.unit)) (128 - SN))
        && (ComparableDType.nat.lt idx.1.val 96)))
    (hkload : ∀ idx : TileIndex [128, 64],
      ktile.data idx = (if kmaskT.data idx then some (sin.readMem (Kptrs.data idx).1 (Kptrs.data idx).2)
        else some (sin.undef (Kptrs.data idx).1 (Kptrs.data idx).2)))
    (hkscval : ksc = ⟨fun _ : TileIndex [] => some (sin.readMem (Ksp.data PUnit.unit).1 (Ksp.data PUnit.unit).2)⟩)
    (hvmask : ∀ idx : TileIndex [64, 128],
      vmaskT.data idx = ((ComparableDType.nat.lt (offsn.data (idx.1, PUnit.unit)) (128 - SN))
        && (ComparableDType.nat.lt idx.2.1.val 96)))
    (hvload : ∀ idx : TileIndex [64, 128],
      vtile.data idx = (if vmaskT.data idx then some (sin.readMem (Vptrs.data idx).1 (Vptrs.data idx).2)
        else some (sin.undef (Vptrs.data idx).1 (Vptrs.data idx).2)))
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ sF, stepStmts AftFoundation.aftLoopBody sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = 0)
      ∧ ∃ (maskT : Tile .bool [128, 64]) (qkRawT qk6T : Tile .real [128, 64])
            (rmaxT mijT : Tile .real [128]) (pT : Tile .real [128, 64])
            (lijT alphaT : Tile .real [128]),
          (∀ idx : TileIndex [128, 64],
            maskT.data idx = ComparableDType.nat.ge (offsm.data (idx.1, PUnit.unit))
              (SN + offsn.data (idx.2.1, PUnit.unit)))
          ∧ qkRawT = Tile.bop NumericDType.real.mul Broadcast.scalarR
              (Tile.bop NumericDType.real.mul Broadcast.scalarR
                ⟨fun i => (Tile.dot [] qtile ktile).data i⟩ qsc) ksc
          ∧ (∀ idx : TileIndex [128, 64],
            qk6T.data idx = if maskT.data idx then qkRawT.data idx
              else WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ)))
          ∧ Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [128, 64].length) qk6T = some rmaxT
          ∧ mijT = Tile.select
              (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT
          ∧ (∀ idx : TileIndex [128, 64],
            pT.data idx = if maskT.data idx
              then WithBot.realExp2 (WithBot.realSub (qk6T.data idx) (mijT.data (idx.1, PUnit.unit)))
              else (some (0.0 : ℝ) : WithBot ℝ))
          ∧ lijT = Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [128, 64].length) pT
          ∧ alphaT = Tile.uop WithBot.realExp2
              (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT)
          ∧ sF.regs .real [128] "m_i" = some mijT
          ∧ sF.regs .real [128] "l_i" = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
              (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) litile alphaT) lijT)
          ∧ sF.regs .real [128, 128] "acc" = some (Tile.bop NumericDType.real.add
              (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
              (Tile.dot [] ⟨fun i => FloatDType.fp16.cast FloatDType.real
                (FloatDType.real.cast FloatDType.fp16 (pT.data i))⟩ vtile))
          ∧ sF.regs .ptr [128, 64] "K_ptrs" = some (Tile.ptrAdd Broadcast.scalarR Kptrs
              (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar 64) (Tile.scalar 128)))
          ∧ sF.regs .ptr [] "K_scale_ptr" = some (Tile.ptrAdd Broadcast.nil Ksp (Tile.scalar 1))
          ∧ sF.regs .ptr [64, 128] "V_ptrs" = some (Tile.ptrAdd Broadcast.scalarR Vptrs
              (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar 64) (Tile.scalar 128))) := by
  rw [aftLoopBody_eq_head_tail]
  -- run the head
  obtain ⟨sH, hHead, hpidsH, hmemH, hundefH, hsnH, hoffsmH, hoffsnH, hqH, hqscH, hmiH, hliH,
      haccH, hKpH, hKspH, hVpH,
      maskT, qkRawT, qk6T, rmaxT, mijT, pT,
      hmaskd, hqkRawd, hqk6d, hrm, hmijd, hpd, hmaskReg, hmijReg, hpReg⟩ :=
    aftLoopBodyHead_steps sin SN offsm offsn qtile qsc ksc ktile kmaskT mtile litile acctile
      Kptrs Ksp Vptrs hsn hoffsm hoffsn hq hqsc hmi hli hacc hKp hKsp hVp hkmask hkload hkscval hundef
  rw [stepStmts.append_some hHead]
  -- the v-load on sH reads sin's mem/undef (head preserves them)
  have hvloadH : ∀ idx : TileIndex [64, 128],
      vtile.data idx = (if vmaskT.data idx then some (sH.readMem (Vptrs.data idx).1 (Vptrs.data idx).2)
        else some (sH.undef (Vptrs.data idx).1 (Vptrs.data idx).2)) := by
    intro idx
    rw [hvload idx]
    cases hm : vmaskT.data idx
    · simp only [hm, Bool.false_eq_true, if_false, hundefH, hundef]
    · simp only [hm, if_true, BlockState.readMem, hmemH]
  have hoffsnH' : sH.regs .nat [64] "offs_n" = some offsn := hoffsnH
  -- run the tail
  obtain ⟨sF, hTail, hpidsF, hmemF, hundefF, lijT, alphaT, hlijd, halphad,
      hmiF, hliF, haccF, hKpF, hKspF, hVpF⟩ :=
    aftLoopBodyTail_steps sH SN offsn pT vtile vmaskT mtile mijT litile
      (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [128, 64].length) pT) acctile
      Kptrs Ksp Vptrs hsnH hoffsnH' hpReg hmiH hmijReg hliH haccH hKpH hKspH hVpH
      hvmask hvloadH hundefH
  refine ⟨sF, hTail, ?_, ?_, ?_,
    maskT, qkRawT, qk6T, rmaxT, mijT, pT, lijT, alphaT,
    hmaskd, hqkRawd, hqk6d, hrm, hmijd, hpd, hlijd, halphad, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF, hpidsH]
  · rw [hmemF, hmemH]
  · exact hundefF
  · exact hmiF
  · rw [hliF, halphad]
  · rw [haccF, halphad]
  · exact hKpF
  · exact hKspF
  · exact hVpF

/-! ## FINAL Part 1 — masked block bridges (single-pT) + StateBot1 + aftScoreBound

The kernel masks `qk` with the **real `-1e6` sentinel** (`tl.where(mask, qk, -1e6)`),
NOT a true `-inf`/`⊥`. So the per-block `reduceMax` carries a `-1e6` floor on
masked lanes. The running-max bridge `aft_mij_reg_eq` therefore needs an
`aftScoreBound` precondition (every kept score `≥ -1e6`), which keeps the floor
from ever raising the row max above the genuine `aftRunningMax`. The causal mask
keeps key `0` for every row, so the running max is `≠ ⊥` from block `0` on.

Unlike triton3 (separate `pmT`), aft folds the mask directly into `pT`
(`pT = if mask then exp2 else 0`), so the denominator/accumulator bridges run
against that single masked tile (`l_ij = Σ pT`, `acc += Σ pT·v`). -/

open VeriTile.Triton

/-- `((qLoadedAFT s0 Q).data (i,e)).unbotD 0 = qMaskedAFT s0 Q (i,e)` — the loaded
masked q cell decodes to the masked-q spec carrier. -/
theorem qLoaded_unbotD_eq_qMasked (s0 : BlockState) (Q : RegionName)
    (i e : Fin 128) :
    ((qLoadedAFT s0 Q).data (i, e, PUnit.unit)).unbotD 0
      = qMaskedAFT s0 Q (i, e, PUnit.unit) := by
  simp only [qLoadedAFT, qMaskedAFT]
  by_cases hc : (ComparableDType.nat.lt (s0.pids 0 * 128 + i.val) 128
      && ComparableDType.nat.lt e.val 96)
  · rw [if_pos hc]
    have hc' : s0.pids 0 * 128 + i.val < 128 ∧ e.val < 96 := by
      simp only [ComparableDType.lt, Bool.and_eq_true, decide_eq_true_eq] at hc; exact hc
    rw [if_pos hc']; rfl
  · rw [if_neg hc]
    have hc' : ¬ (s0.pids 0 * 128 + i.val < 128 ∧ e.val < 96) := by
      simp only [ComparableDType.lt, Bool.and_eq_true, decide_eq_true_eq] at hc; exact hc
    rw [if_neg hc']; rfl

/-- **K head-mask irrelevance.** The kernel loads `k` through the `arange < 96`
head-active mask (so `ktile = 0` at lanes `e ≥ 96`), but the dot with the loaded
masked-`q` (`qLoadedAFT`, also `0` at `e ≥ 96`) is unchanged by replacing the
masked `ktile` with the *full* (unmasked) `k` readback: the `e ≥ 96` dot terms
vanish because the `q` factor is `0` there. This lets the score bridges
(`aft_score_cell`/`_masked`) take the unconditional full-`k` readback `hk` even
though the loop only supplies the masked `ktile`. -/
theorem aft_dot_kmask_irrel (s0 : BlockState) (Q : RegionName)
    (i : Fin 128) (jL : Fin 64)
    (ktile kfull : Tile .real [128, 64])
    (hagree : ∀ e : Fin 128, e.val < 96 → ktile.data (e, jL, PUnit.unit) = kfull.data (e, jL, PUnit.unit))
    (hksome : ∀ e : Fin 128, 96 ≤ e.val → (∃ v : ℝ, ktile.data (e, jL, PUnit.unit) = some v))
    (hfsome : ∀ e : Fin 128, 96 ≤ e.val → (∃ w : ℝ, kfull.data (e, jL, PUnit.unit) = some w)) :
    (Tile.dot [] (qLoadedAFT s0 Q) ktile).data (i, jL, PUnit.unit)
      = (Tile.dot [] (qLoadedAFT s0 Q) kfull).data (i, jL, PUnit.unit) := by
  rw [Tile.dot_nil_data, Tile.dot_nil_data]
  refine Finset.sum_congr rfl (fun e _ => ?_)
  by_cases he : e.val < 96
  · rw [hagree e he]
  · -- qLoadedAFT cell is `some 0` at e ≥ 96, so both products are `some 0`
    have hge : 96 ≤ e.val := by omega
    have hq0 : (qLoadedAFT s0 Q).data (i, e, PUnit.unit) = some (0 : ℝ) := by
      simp only [qLoadedAFT]
      rw [if_neg (by
        simp only [ComparableDType.lt, Bool.and_eq_true, decide_eq_true_eq]
        rintro ⟨_, h⟩; exact he h)]
    obtain ⟨v, hv⟩ := hksome e hge
    obtain ⟨w, hw⟩ := hfsome e hge
    rw [hq0, hv, hw]
    simp only [Option.map₂, Option.bind, Option.map, zero_mul]

/-- **Score cell (masked-q spec carrier).** The raw `qk` cell at row `i`,
block-key `jL` (block `c`) is `some (qsc·ksc·Σ_e qMaskedAFT(i,e)·kTileAFT(c·64+jL,e))`. -/
theorem aft_score_cell_masked (s0 : BlockState) (Q K : RegionName) (qsc ksc : ℝ) (c : Nat)
    (i : Fin 128) (jL : Fin 64) (hjL : c * 64 + jL.val < 128)
    (qtile : Tile .real [128, 128]) (ktile : Tile .real [128, 64])
    (hq : qtile = qLoadedAFT s0 Q)
    (hk : ∀ idx : TileIndex [128, 64],
        ktile.data idx = some (s0.readMem K
          (baseOffsetAFT s0 + idx.1.val + (c * 64 + idx.2.1.val) * 128))) :
    (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (⟨fun i => (Tile.dot [] qtile ktile).data i⟩ : Tile .real [128, 64])
          (Tile.scalar (some qsc)))
        (Tile.scalar (some ksc))).data (i, jL, PUnit.unit)
      = some (qsc * ksc * Finset.univ.sum (fun e : Fin 128 =>
          qMaskedAFT s0 Q (i, e, PUnit.unit)
            * kTileAFT s0 K (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit))) := by
  rw [aft_score_cell s0 Q K qsc ksc c i jL hjL qtile ktile hq hk]
  refine congrArg some ?_
  rw [show (fun e : Fin 128 => ((qLoadedAFT s0 Q).data (i, e, PUnit.unit)).unbotD 0
          * kTileAFT s0 K (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit))
        = (fun e : Fin 128 => qMaskedAFT s0 Q (i, e, PUnit.unit)
          * kTileAFT s0 K (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit))
      from by funext e; rw [qLoaded_unbotD_eq_qMasked]]

/-- The per-key causal score `keyScale j · (qMasked row i · k row j)` — `aftKV`'s
`.1` for the masked-q tiles. The program's per-key `keyScale` carries `q_scale`
times the per-block `k_scale` (`keyScale j = q_scale · k_scale[j/64]`), so it
varies across key blocks. -/
noncomputable def aftBlockScore (s0 : BlockState) (Q K : RegionName) (keyScale : Fin 128 → ℝ)
    (i : Fin 128) (j : Fin 128) : ℝ :=
  keyScale j * Finset.univ.sum (fun e : Fin 128 =>
    qMaskedAFT s0 Q (i, e, PUnit.unit) * kTileAFT s0 K (j, e, PUnit.unit))

/-- **Score bound precondition.** The kernel's `-1e6` mask sentinel never raises a
row's running max above the genuine `aftRunningMax`, provided every kept score is
`≥ -1e6` (`= 0 - 1000000`). With the causal mask (key `0` always kept) this makes
the floor inert from block `0` on. -/
def aftScoreBound (s0 : BlockState) (Q K : RegionName) (keyScale : Fin 128 → ℝ) : Prop :=
  ∀ (i j : Fin 128), j.val ≤ qStartAFT s0 + i.val →
    (0.0 : ℝ) - (1000000.0 : ℝ) ≤ aftBlockScore s0 Q K keyScale i j

/-- Any member of a `WithBot ℝ` list is `≤` its `foldr (⊔) ⊥`. -/
theorem aft_mem_le_foldr_sup (a : WithBot ℝ) :
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

/-- `aftRunningMax` over the causal window equals the windowed `foldr ⊔ ⊥` of the
coerced block scores; spelled directly via `aftBlock`. -/
theorem aftBlock_blockSup (s0 : BlockState) (Q K V : RegionName) (keyScale : Fin 128 → ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 128) :
    Finset.univ.sup (fun jL : Fin 64 =>
        if (c * 64 + jL.val ≤ qStartAFT s0 + i.val) then
          ((aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ : ℝ) : WithBot ℝ)
        else (⊥ : WithBot ℝ))
      = ((aftBlock (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
          (qStartAFT s0) c i d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  rw [show (aftBlock (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
        (qStartAFT s0) c i d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
      = ((List.finRange 128).filterMap (fun j : Fin 128 =>
          if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStartAFT s0 + i.val
          then some (aftBlockScore s0 Q K keyScale i j) else none)).map
            (fun x => ((x : ℝ) : WithBot ℝ)) from by
    unfold aftBlock
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr
    intro j _
    by_cases hj : c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStartAFT s0 + i.val
    · simp only [hj, if_true]; rfl
    · simp [hj]]
  rw [show (((List.finRange 128).filterMap (fun j : Fin 128 =>
          if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStartAFT s0 + i.val
          then some (aftBlockScore s0 Q K keyScale i j) else none)).map
            (fun x => ((x : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = Finset.univ.sup (fun j : Fin 128 =>
          if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStartAFT s0 + i.val
          then ((aftBlockScore s0 Q K keyScale i j : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ)) from by
    rw [show (((List.finRange 128).filterMap (fun j : Fin 128 =>
            if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStartAFT s0 + i.val
            then some (aftBlockScore s0 Q K keyScale i j) else none)).map
              (fun x => ((x : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
        = (List.finRange 128).foldr (fun j a =>
            (if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStartAFT s0 + i.val
              then ((aftBlockScore s0 Q K keyScale i j : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ)) ⊔ a) ⊥
        from by
      induction (List.finRange 128) with
      | nil => simp
      | cons a t ih =>
        by_cases ha : c * 64 ≤ a.val ∧ a.val < (c + 1) * 64 ∧ a.val ≤ qStartAFT s0 + i.val <;>
          simp [ha, ih]]
    apply le_antisymm
    · induction (List.finRange 128) with
      | nil => simp
      | cons a t ih =>
        simp only [List.foldr_cons]
        exact sup_le (Finset.le_sup (f := fun j : Fin 128 =>
          if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStartAFT s0 + i.val
          then ((aftBlockScore s0 Q K keyScale i j : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))
          (Finset.mem_univ a)) ih
    · apply Finset.sup_le; intro j _
      have key : ∀ (l : List (Fin 128)), j ∈ l →
          (if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStartAFT s0 + i.val
            then ((aftBlockScore s0 Q K keyScale i j : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))
            ≤ l.foldr (fun j a =>
              (if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStartAFT s0 + i.val
                then ((aftBlockScore s0 Q K keyScale i j : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ)) ⊔ a) ⊥ := by
        intro l hl
        induction l with
        | nil => simp at hl
        | cons a t ih =>
          simp only [List.foldr_cons]
          rcases List.mem_cons.mp hl with h | h
          · subst h; exact le_sup_left
          · exact le_trans (ih h) le_sup_right
      exact key _ (List.mem_finRange j)]
  -- relate the [128]-indexed sup to the [64]-indexed sup
  symm
  apply le_antisymm
  · apply Finset.sup_le; intro j _
    by_cases hj : c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStartAFT s0 + i.val
    · rw [if_pos hj]
      have hjL : j.val - c * 64 < 64 := by omega
      refine le_trans ?_ (Finset.le_sup (Finset.mem_univ (⟨j.val - c * 64, hjL⟩ : Fin 64)))
      simp only
      have hfin : c * 64 + (j.val - c * 64) = j.val := by omega
      rw [if_pos (show c * 64 + (⟨j.val - c * 64, hjL⟩ : Fin 64).val ≤ qStartAFT s0 + i.val from by
        simp only; rw [hfin]; exact hj.2.2)]
      apply le_of_eq
      congr 2
      apply Fin.ext; simp only; rw [hfin]
    · rw [if_neg hj]; exact bot_le
  · apply Finset.sup_le; intro jL _
    have hb : c * 64 + jL.val < 128 := by have := jL.isLt; omega
    by_cases hkeep : c * 64 + jL.val ≤ qStartAFT s0 + i.val
    · rw [if_pos hkeep]
      refine le_trans ?_ (Finset.le_sup (Finset.mem_univ (⟨c * 64 + jL.val, hb⟩ : Fin 128)))
      simp only
      rw [if_pos (by have := jL.isLt; exact ⟨by omega, by omega, hkeep⟩)]
    · rw [if_neg hkeep]; exact bot_le

/-- Canonical axis-1 index of `[128, 64]`. -/
abbrev aftAx1 : Fin [128, 64].length := ⟨1, by simp⟩

/-- **`reduceMax` row.** The `tl.max(qk, 1)` cell at row `i` equals the `Finset.sup`
over the block lanes of the row cells. -/
theorem aft_reduceMaxDrop_row (qk : Tile .real [128, 64]) (rmaxT : Tile .real [128])
    (hrm : Tile.reduceMaxDrop aftAx1 qk = some rmaxT)
    (i : Fin 128) (g : Fin 64 → WithBot ℝ)
    (hqk : ∀ jL : Fin 64, qk.data (i, jL, PUnit.unit) = g jL) :
    rmaxT.data (i, PUnit.unit) = Finset.univ.sup g := by
  unfold Tile.reduceMaxDrop at hrm
  rw [dif_pos (show 0 < TileShape.axisDim [128, 64] aftAx1 from by decide)] at hrm
  rw [← Option.some.inj hrm]
  simp only [Finset.sup'_eq_sup]
  exact Finset.sup_congr rfl (fun jL _ => hqk jL)

/-- **`m_ij = aftRunningMax((c+1)·64)` (single-pT, `-1e6` floor).** The kernel
running max after block `c` equals the genuine `aftRunningMax`. The `-1e6`
`where`-sentinel floor is inert because (a) `aftScoreBound` keeps every kept score
`≥ -1e6` and (b) the causal mask keeps key `0`, so the running max `≥` some kept
score `≥ -1e6` from block `0` on. -/
theorem aft_mij_reg_eq (s0 : BlockState) (Q K V : RegionName) (keyScale : Fin 128 → ℝ) (qsc ksc : ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i : Fin 128)
    (hks : ∀ jL : Fin 64, keyScale ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ = qsc * ksc)
    (hbound : aftScoreBound s0 Q K keyScale)
    (qtile : Tile .real [128, 128]) (ktile : Tile .real [128, 64]) (qkRawT : Tile .real [128, 64])
    (maskT : Tile .bool [128, 64]) (mtile rmaxT qk6T : Tile .real [128])
    (qk6 : Tile .real [128, 64])
    (hq : qtile = qLoadedAFT s0 Q)
    (hk : ∀ idx : TileIndex [128, 64],
        ktile.data idx = some (s0.readMem K
          (baseOffsetAFT s0 + idx.1.val + (c * 64 + idx.2.1.val) * 128)))
    (hqkRaw : qkRawT = Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (⟨fun i => (Tile.dot [] qtile ktile).data i⟩ : Tile .real [128, 64])
          (Tile.scalar (some qsc))) (Tile.scalar (some ksc)))
    (hmask : ∀ jL : Fin 64, maskT.data (i, jL, PUnit.unit)
        = ComparableDType.nat.ge (qStartAFT s0 + i.val) (c * 64 + jL.val))
    (hqk6 : ∀ jL : Fin 64, qk6.data (i, jL, PUnit.unit)
        = if maskT.data (i, jL, PUnit.unit) then qkRawT.data (i, jL, PUnit.unit)
          else WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ)))
    (hmtile : mtile.data (i, PUnit.unit)
        = aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
            (qStartAFT s0) (c * 64) i ⟨0, by norm_num⟩)
    (hrmax : Tile.reduceMaxDrop aftAx1 qk6 = some rmaxT) :
    (Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT)
        mtile rmaxT).data (i, PUnit.unit)
      = aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
          (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩ := by
  set qT := qMaskedAFT s0 Q
  set kT := kTileAFT s0 K
  set vT := vMaskedAFT s0 V
  set qStart := qStartAFT s0
  -- the floor value as WithBot
  set floor : WithBot ℝ := WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ)) with hfloor
  have hfloorR : floor = ((0.0 - 1000000.0 : ℝ) : WithBot ℝ) := by
    rw [hfloor]; rfl
  -- per-lane cell value
  have hcell : ∀ jL : Fin 64, qk6.data (i, jL, PUnit.unit)
      = if (c * 64 + jL.val ≤ qStart + i.val) then
          ((aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ : ℝ) : WithBot ℝ)
        else floor := by
    intro jL
    rw [hqk6 jL, hmask jL]
    by_cases hkp : c * 64 + jL.val ≤ qStart + i.val
    · have hb : ComparableDType.nat.ge (qStart + i.val) (c * 64 + jL.val) = Bool.true := by
        rw [ComparableDType.nat_ge_eq_true]; omega
      rw [hb, if_pos rfl, if_pos hkp, hqkRaw]
      have := aft_score_cell_masked s0 Q K qsc ksc c i jL (by have := jL.isLt; omega) qtile ktile hq hk
      rw [this]
      rw [show ((aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ : ℝ) : WithBot ℝ)
            = some (qsc * ksc * Finset.univ.sum (fun e : Fin 128 =>
                qMaskedAFT s0 Q (i, e, PUnit.unit) * kTileAFT s0 K (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit)))
          from by rw [aftBlockScore, hks jL]; rfl]
    · have hb : ComparableDType.nat.ge (qStart + i.val) (c * 64 + jL.val) = Bool.false := by
        rw [← Bool.not_eq_true, ComparableDType.nat_ge_eq_true]; omega
      rw [hb, if_neg (by simp), if_neg hkp]
  -- rmax cell = sup over jL of qk6 cells
  have hrmaxcell : rmaxT.data (i, PUnit.unit) = Finset.univ.sup (fun jL : Fin 64 =>
      if (c * 64 + jL.val ≤ qStart + i.val) then
        ((aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ : ℝ) : WithBot ℝ)
      else floor) :=
    aft_reduceMaxDrop_row qk6 rmaxT hrmax i _ hcell
  -- aftRunningMax((c+1)·64) = aftRunningMax(c·64) ⊔ blockSup
  rw [aftRunningMax_succ qT kT vT keyScale qStart c i ⟨0, by norm_num⟩]
  rw [← aftBlock_blockSup s0 Q K V keyScale c hc1 i ⟨0, by norm_num⟩]
  -- the select is max(mtile, rmax)
  rw [Tile.select_data, Tile.cop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hmtile, hrmaxcell]
  set M := aftRunningMax qT kT vT keyScale qStart (c * 64) i ⟨0, by norm_num⟩ with hM
  set S := Finset.univ.sup (fun jL : Fin 64 =>
      if (c * 64 + jL.val ≤ qStart + i.val) then
        ((aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ : ℝ) : WithBot ℝ)
      else floor) with hS
  set BS := Finset.univ.sup (fun jL : Fin 64 =>
      if (c * 64 + jL.val ≤ qStart + i.val) then
        ((aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ : ℝ) : WithBot ℝ)
      else (⊥ : WithBot ℝ)) with hBS
  -- `BS ≤ S ≤ BS ⊔ floor` (each S-term is a BS-term or `floor`)
  have hBSleS : BS ≤ S := by
    rw [hBS, hS]; apply Finset.sup_mono_fun; intro jL _
    by_cases hkp : c * 64 + jL.val ≤ qStart + i.val
    · rw [if_pos hkp, if_pos hkp]
    · rw [if_neg hkp, if_neg hkp]; exact bot_le
  have hSleBSfloor : S ≤ BS ⊔ floor := by
    rw [hS]; apply Finset.sup_le; intro jL _
    by_cases hkp : c * 64 + jL.val ≤ qStart + i.val
    · rw [if_pos hkp]
      refine le_sup_of_le_left ?_
      rw [hBS]
      have hle := Finset.le_sup (f := fun jL : Fin 64 =>
        if (c * 64 + jL.val ≤ qStart + i.val) then
          ((aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ : ℝ) : WithBot ℝ)
        else (⊥ : WithBot ℝ)) (Finset.mem_univ jL)
      simp only [if_pos hkp] at hle
      exact hle
    · rw [if_neg hkp]; exact le_sup_right
  -- `floor ≤ aftRunningMax((c+1)·64) = M ⊔ BS` (key 0 is kept, score ≥ -1e6)
  have hfloorLe : floor ≤ M ⊔ BS := by
    rw [hM, hBS]
    rw [aftBlock_blockSup s0 Q K V keyScale c hc1 i ⟨0, by norm_num⟩]
    rw [← aftRunningMax_succ qT kT vT keyScale qStart c i ⟨0, by norm_num⟩]
    -- the running max over [0,(c+1)·64) contains key 0's score ≥ -1e6
    have hmem : (((aftBlockScore s0 Q K keyScale i ⟨0, by norm_num⟩ : ℝ) : WithBot ℝ))
        ≤ aftRunningMax qT kT vT keyScale qStart ((c + 1) * 64) i ⟨0, by norm_num⟩ := by
      unfold aftRunningMax aftKeysUpto
      apply aft_mem_le_foldr_sup
      rw [List.mem_map]
      refine ⟨aftKV qT kT vT keyScale i ⟨0, by norm_num⟩ ⟨0, by norm_num⟩, ?_, ?_⟩
      · rw [List.mem_filterMap]
        refine ⟨⟨0, by norm_num⟩, List.mem_finRange _, ?_⟩
        rw [if_pos (show (⟨0, by norm_num⟩ : Fin 128).val < (c + 1) * 64
            ∧ (⟨0, by norm_num⟩ : Fin 128).val ≤ qStart + i.val from
          ⟨by simp only; omega, by simp⟩)]
      · simp only [aftKV, aftBlockScore, qT, kT]
    refine le_trans ?_ hmem
    rw [hfloorR]
    apply WithBot.coe_le_coe.mpr
    have := hbound i ⟨0, by norm_num⟩ (by simp)
    simpa [aftBlockScore] using this
  have hkey : M ⊔ S = M ⊔ BS := by
    apply le_antisymm
    · calc M ⊔ S ≤ M ⊔ (BS ⊔ floor) := sup_le_sup_left hSleBSfloor M
        _ = (M ⊔ BS) ⊔ floor := by rw [sup_assoc]
        _ = M ⊔ BS := by rw [sup_eq_left.mpr hfloorLe]
    · exact sup_le_sup_left hBSleS M
  -- the select `if M > S then M else S` is `max M S`; rewrite via hkey to `max M BS`
  rw [show (if decide (M > S) = Bool.true then M else S) = M ⊔ S from by
    by_cases h : M ≤ S
    · rw [if_neg (by simp [not_lt.mpr h]), max_eq_right h]
    · rw [if_pos (by simpa using not_le.mp h), max_eq_left (le_of_lt (not_le.mp h))]]
  rw [hkey]

set_option maxRecDepth 8000 in
/-- `aftBlock`'s `h`-image sum collapses to a `Fin 64` masked `Finset.sum` over the
block lanes (each a kept score/value pair). -/
theorem aftBlock_map_sum
    (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart c : Nat) (i d : Fin 128) (hwin : (c + 1) * 64 ≤ 128) (h : ℝ × ℝ → ℝ) :
    ((aftBlock qT kT vT keyScale qStart c i d).map h).sum
      = ∑ jL : Fin 64,
          (if c * 64 + jL.val ≤ qStart + i.val then
            h (aftKV qT kT vT keyScale i d ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩)
           else 0) := by
  rw [aftBlock]
  rw [show ((((List.finRange 128).filterMap (fun j : Fin 128 =>
          if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val
          then some (aftKV qT kT vT keyScale i d j) else none)).map h)).sum
        = ∑ j : Fin 128, if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val
            then h (aftKV qT kT vT keyScale i d j) else 0 from by
    rw [List.map_filterMap]
    rw [show (fun j : Fin 128 => Option.map h (if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64
            ∧ j.val ≤ qStart + i.val then some (aftKV qT kT vT keyScale i d j) else none))
          = (fun j : Fin 128 => if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val
            then some (h (aftKV qT kT vT keyScale i d j)) else none) from by
      funext j; by_cases hj : c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val <;> simp [hj]]
    rw [show (((List.finRange 128).filterMap (fun j : Fin 128 =>
            if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val
            then some (h (aftKV qT kT vT keyScale i d j)) else none))).sum
          = ((List.finRange 128).map (fun j : Fin 128 =>
              if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val
              then h (aftKV qT kT vT keyScale i d j) else 0)).sum from by
      induction (List.finRange 128) with
      | nil => simp
      | cons a t ih =>
        by_cases ha : c * 64 ≤ a.val ∧ a.val < (c + 1) * 64 ∧ a.val ≤ qStart + i.val <;>
          simp [ha, ih]]
    rw [← List.sum_ofFn, List.ofFn_eq_map]]
  rw [show (∑ j : Fin 128, if c * 64 ≤ j.val ∧ j.val < (c + 1) * 64 ∧ j.val ≤ qStart + i.val
            then h (aftKV qT kT vT keyScale i d j) else 0)
        = ∑ j ∈ Finset.univ.filter (fun j : Fin 128 => c * 64 ≤ j.val ∧ j.val < (c + 1) * 64),
            (if j.val ≤ qStart + i.val then h (aftKV qT kT vT keyScale i d j) else 0) from by
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

/-- A kept lane forces the running max `≠ ⊥` (its score is in the windowed prefix). -/
theorem aftRunningMax_succ_ne_bot_of_kept (s0 : BlockState) (Q K V : RegionName) (keyScale : Fin 128 → ℝ)
    (c : Nat) (i d : Fin 128) (jL : Fin 64) (hb : c * 64 + jL.val < 128)
    (hkp : c * 64 + jL.val ≤ qStartAFT s0 + i.val) :
    aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
        (qStartAFT s0) ((c + 1) * 64) i d ≠ ⊥ := by
  set p := aftKV (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
      i d ⟨c * 64 + jL.val, hb⟩ with hp
  have hmem : ((p.1 : ℝ) : WithBot ℝ) ∈
      ((aftKeysUpto (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
        (qStartAFT s0) ((c + 1) * 64) i d).map (fun q => ((q.1 : ℝ) : WithBot ℝ))) := by
    rw [List.mem_map]
    refine ⟨p, ?_, rfl⟩
    unfold aftKeysUpto
    rw [List.mem_filterMap]
    refine ⟨⟨c * 64 + jL.val, hb⟩, List.mem_finRange _, ?_⟩
    rw [if_pos (show (⟨c * 64 + jL.val, hb⟩ : Fin 128).val < (c + 1) * 64
        ∧ (⟨c * 64 + jL.val, hb⟩ : Fin 128).val ≤ qStartAFT s0 + i.val from
      ⟨by have := jL.isLt; simp only; omega, hkp⟩)]
  have hle := aft_mem_le_foldr_sup _ _ hmem
  rw [← aftRunningMax] at hle
  intro hbot
  exact absurd (le_bot_iff.mp (hbot ▸ hle)) WithBot.coe_ne_bot

set_option maxHeartbeats 1600000 in
/-- **Masked `pT` cell** (single-pT). The kernel's masked `p` cell on lane `jL` is
`some (exp2(score − Mr))` when causally kept, `some 0` when masked — given `mij`
cell `= aftRunningMax((c+1)·64) = some Mr`. -/
theorem aft_pT_cell (s0 : BlockState) (Q K V : RegionName) (keyScale : Fin 128 → ℝ) (qsc ksc : ℝ) (c : Nat)
    (i : Fin 128) (jL : Fin 64) (hjL : c * 64 + jL.val < 128)
    (hks : keyScale ⟨c * 64 + jL.val, hjL⟩ = qsc * ksc)
    (qtile : Tile .real [128, 128]) (ktile : Tile .real [128, 64]) (qkRawT qk6T : Tile .real [128, 64])
    (maskT : Tile .bool [128, 64]) (mijT : Tile .real [128]) (pT : Tile .real [128, 64])
    (hq : qtile = qLoadedAFT s0 Q)
    (hk : ∀ idx : TileIndex [128, 64],
        ktile.data idx = some (s0.readMem K
          (baseOffsetAFT s0 + idx.1.val + (c * 64 + idx.2.1.val) * 128)))
    (hqkRaw : qkRawT = Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (⟨fun i => (Tile.dot [] qtile ktile).data i⟩ : Tile .real [128, 64])
          (Tile.scalar (some qsc))) (Tile.scalar (some ksc)))
    (hmask : maskT.data (i, jL, PUnit.unit)
        = ComparableDType.nat.ge (qStartAFT s0 + i.val) (c * 64 + jL.val))
    (hqk6 : qk6T.data (i, jL, PUnit.unit)
        = if maskT.data (i, jL, PUnit.unit) then qkRawT.data (i, jL, PUnit.unit)
          else WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ)))
    (hmij : mijT.data (i, PUnit.unit)
        = aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
            (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩)
    (hpT : pT.data (i, jL, PUnit.unit)
        = if maskT.data (i, jL, PUnit.unit)
          then WithBot.realExp2 (WithBot.realSub (qk6T.data (i, jL, PUnit.unit))
            (mijT.data (i, PUnit.unit)))
          else (some (0.0 : ℝ) : WithBot ℝ))
    (hMij_kept : c * 64 + jL.val ≤ qStartAFT s0 + i.val →
        aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
            (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩ ≠ ⊥) :
    pT.data (i, jL, PUnit.unit)
      = some (if c * 64 + jL.val ≤ qStartAFT s0 + i.val then
          pow2 (aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, hjL⟩
            - (aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
                (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩).unbotD 0)
          else 0) := by
  rw [hpT, hmask]
  by_cases hkp : c * 64 + jL.val ≤ qStartAFT s0 + i.val
  · have hge : ComparableDType.nat.ge (qStartAFT s0 + i.val) (c * 64 + jL.val) = Bool.true := by
      rw [ComparableDType.nat_ge_eq_true]; omega
    rw [hge, if_pos rfl, if_pos hkp]
    obtain ⟨Mr, hMr⟩ : ∃ Mr : ℝ, aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V)
        keyScale (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩ = (Mr : WithBot ℝ) := by
      cases hh : aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V)
          keyScale (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩ with
      | coe x => exact ⟨x, rfl⟩
      | bot => exact absurd hh (hMij_kept hkp)
    rw [hqk6, hmask, hge, if_pos rfl, hqkRaw]
    rw [aft_score_cell_masked s0 Q K qsc ksc c i jL hjL qtile ktile hq hk]
    rw [hmij, hMr, WithBot.unbotD_coe]
    rw [show (qsc * ksc * Finset.univ.sum (fun e : Fin 128 =>
            qMaskedAFT s0 Q (i, e, PUnit.unit) * kTileAFT s0 K (⟨c * 64 + jL.val, hjL⟩, e, PUnit.unit)))
          = aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, hjL⟩ from by
        simp only [aftBlockScore, hks]]
    show WithBot.realExp2 (WithBot.realSub
        ((aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, hjL⟩ : ℝ) : WithBot ℝ) ((Mr : ℝ) : WithBot ℝ)) = _
    rw [WithBot.realSub_coe_coe, WithBot.realExp2_coe]
    refine congrArg some ?_
    simp only [pow2]; ring_nf
  · have hge : ComparableDType.nat.ge (qStartAFT s0 + i.val) (c * 64 + jL.val) = Bool.false := by
      rw [← Bool.not_eq_true, ComparableDType.nat_ge_eq_true]; omega
    rw [hge, if_neg (by simp), if_neg hkp]; norm_num

/-- `aftRunningMax` is independent of the channel `d` (the score `.1` involves
only `qT`/`kT`/`keyScale`). -/
theorem aftRunningMax_eq (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d d' : Fin 128) :
    aftRunningMax qT kT vT keyScale qStart hi i d
      = aftRunningMax qT kT vT keyScale qStart hi i d' := by
  unfold aftRunningMax aftKeysUpto
  rw [List.map_filterMap, List.map_filterMap]
  rw [List.filterMap_congr (l := List.finRange 128)
    (f := fun j : Fin 128 => Option.map (fun p => ((p.1 : ℝ) : WithBot ℝ))
      (if j.val < hi ∧ j.val ≤ qStart + i.val then some (aftKV qT kT vT keyScale i d j) else none))
    (g := fun j : Fin 128 => Option.map (fun p => ((p.1 : ℝ) : WithBot ℝ))
      (if j.val < hi ∧ j.val ≤ qStart + i.val then some (aftKV qT kT vT keyScale i d' j) else none))
    (fun j _ => by by_cases hj : j.val < hi ∧ j.val ≤ qStart + i.val <;> simp [hj, aftKV])]

/-- The ⊥-seeded denominator (`aftStateBot.2.1`) is independent of the channel `d`. -/
theorem aftStateBot_snd_fst_indep (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i : Fin 128) (d d' : Fin 128) :
    (aftStateBot qT kT vT keyScale qStart hi i d).2.1
      = (aftStateBot qT kT vT keyScale qStart hi i d').2.1 := by
  rw [aftStateBot_snd_fst, aftStateBot_snd_fst, aftRunningMax_eq qT kT vT keyScale qStart hi i d d']
  congr 2
  unfold aftKeysUpto
  rw [List.map_filterMap, List.map_filterMap]
  rw [List.filterMap_congr (l := List.finRange 128)
    (f := fun j : Fin 128 => Option.map (fun p => pow2 p.1)
      (if j.val < hi ∧ j.val ≤ qStart + i.val then some (aftKV qT kT vT keyScale i d j) else none))
    (g := fun j : Fin 128 => Option.map (fun p => pow2 p.1)
      (if j.val < hi ∧ j.val ≤ qStart + i.val then some (aftKV qT kT vT keyScale i d' j) else none))
    (fun j _ => by by_cases hj : j.val < hi ∧ j.val ≤ qStart + i.val <;> simp [hj, aftKV])]

/-- **`Σ_jL pT[i,jL]` cell sum (single-pT).** Given the `pT` cell decoded form
(per `aft_pT_cell`), the `aftBlock` pow2-score sum over kept lanes. The shared
`pT`-cell hypothesis `hpc` is supplied by the caller (from `aft_pT_cell`). -/
theorem aft_nume_row_sum (s0 : BlockState) (Q K V : RegionName) (keyScale : Fin 128 → ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 128) (pT : Tile .real [128, 64])
    (hpc : ∀ jL : Fin 64, pT.data (i, jL, PUnit.unit)
        = some (if c * 64 + jL.val ≤ qStartAFT s0 + i.val then
            pow2 (aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩
              - (aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
                  (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩).unbotD 0)
            else 0)) :
    (Tile.reduceSumDrop aftAx1 pT).data (i, PUnit.unit)
      = some ((aftBlock (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
          (qStartAFT s0) c i d).map (fun p =>
            pow2 (p.1 - (aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V)
              keyScale (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩).unbotD 0))).sum := by
  set Mc1 := aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
      (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩ with hMc1
  rw [Tile.reduceSumDrop_data]
  have hcell : ∀ jL : Fin 64,
      pT.data (TileShape.insertAxisIndex [128, 64] aftAx1 (i, PUnit.unit) jL)
        = some (if c * 64 + jL.val ≤ qStartAFT s0 + i.val then
            pow2 (aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ - Mc1.unbotD 0)
            else 0) := by
    intro jL
    rw [show TileShape.insertAxisIndex [128, 64] aftAx1 (i, PUnit.unit) jL = (i, jL, PUnit.unit) from rfl]
    exact hpc jL
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [aftBlock_map_sum (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
      (qStartAFT s0) c i d hc1 (fun p => pow2 (p.1 - Mc1.unbotD 0))]
  refine Finset.sum_congr rfl (fun jL _ => ?_)
  by_cases hkp : c * 64 + jL.val ≤ qStartAFT s0 + i.val
  · rw [if_pos hkp, if_pos hkp]; rfl
  · rw [if_neg hkp, if_neg hkp]

/-- **`Σ_jL pT[i,jL]·v[jL,d]` cell sum (single-pT).** -/
theorem aft_acc_dot_block (s0 : BlockState) (Q K V : RegionName) (keyScale : Fin 128 → ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 128)
    (pT : Tile .real [128, 64]) (vtile : Tile .real [64, 128])
    (hv : ∀ idx : TileIndex [64, 128],
        vtile.data idx = some (vMaskedAFT s0 V
          (⟨c * 64 + idx.1.val, by have := idx.1.isLt; omega⟩, idx.2.1, PUnit.unit)))
    (hpc : ∀ jL : Fin 64, pT.data (i, jL, PUnit.unit)
        = some (if c * 64 + jL.val ≤ qStartAFT s0 + i.val then
            pow2 (aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩
              - (aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
                  (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩).unbotD 0)
            else 0)) :
    (Tile.dot [] pT vtile).data (i, d, PUnit.unit)
      = some ((aftBlock (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
          (qStartAFT s0) c i d).map (fun p =>
            pow2 (p.1 - (aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V)
              keyScale (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩).unbotD 0) * p.2)).sum := by
  set Mc1 := aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
      (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩ with hMc1
  rw [Tile.dot_nil_data]
  have hcell : ∀ jL : Fin 64,
      Option.map₂ (· * ·) (pT.data (i, jL, PUnit.unit)) (vtile.data (jL, d, PUnit.unit))
        = some (if c * 64 + jL.val ≤ qStartAFT s0 + i.val then
            pow2 (aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ - Mc1.unbotD 0)
              * vMaskedAFT s0 V (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit)
            else 0) := by
    intro jL
    rw [hpc jL, hv (jL, d, PUnit.unit)]
    by_cases hkp : c * 64 + jL.val ≤ qStartAFT s0 + i.val
    · rw [if_pos hkp, if_pos hkp]; rfl
    · rw [if_neg hkp, if_neg hkp]
      simp only [Option.map₂, Option.bind, Option.map]; rw [zero_mul]
  rw [show (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·) (pT.data (i, k, PUnit.unit)) (vtile.data (k, d, PUnit.unit))))
      = @Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ (fun jL =>
          (some (if c * 64 + jL.val ≤ qStartAFT s0 + i.val then
              pow2 (aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ - Mc1.unbotD 0)
                * vMaskedAFT s0 V (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit)
              else 0) : WithBot ℝ))
      from Finset.sum_congr rfl (fun jL _ => hcell jL)]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [aftBlock_map_sum (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
      (qStartAFT s0) c i d hc1 (fun p => pow2 (p.1 - Mc1.unbotD 0) * p.2)]
  refine Finset.sum_congr rfl (fun jL _ => ?_)
  by_cases hkp : c * 64 + jL.val ≤ qStartAFT s0 + i.val
  · rw [if_pos hkp, if_pos hkp]; rfl
  · rw [if_neg hkp, if_neg hkp]

/-- If the ⊥-seeded running max over `[0, hi)` is `⊥`, the key list is empty, so its
`h`-image sum is `0`. -/
theorem aftKeysUpto_sum_zero_of_bot (qT kT vT : TileIndex [128, 128] → ℝ) (keyScale : Fin 128 → ℝ)
    (qStart hi : Nat) (i d : Fin 128)
    (hbot : aftRunningMax qT kT vT keyScale qStart hi i d = ⊥) (h : ℝ × ℝ → ℝ) :
    ((aftKeysUpto qT kT vT keyScale qStart hi i d).map h).sum = 0 := by
  rw [show aftKeysUpto qT kT vT keyScale qStart hi i d = [] from ?_, List.map_nil, List.sum_nil]
  by_contra hne
  obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ hne
  have hmem : ((p.1 : ℝ) : WithBot ℝ) ∈
      (aftKeysUpto qT kT vT keyScale qStart hi i d).map (fun q => ((q.1 : ℝ) : WithBot ℝ)) :=
    List.mem_map_of_mem hp
  have := aft_mem_le_foldr_sup _ _ hmem
  rw [← aftRunningMax, hbot] at this
  exact absurd (le_bot_iff.mp this) WithBot.coe_ne_bot

/-- The `.1` of a `block.foldl osStepBot (m,l,acc)` is `m ⊔ blockSup`. -/
theorem osStepBot_block_fst (m : WithBot ℝ) (l acc : ℝ) (block : List (ℝ × ℝ)) :
    (block.foldl osStepBot (m, l, acc)).1
      = m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  rw [aftStateBot_fst]
  generalize (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))) = L
  induction L generalizing m with
  | nil => simp
  | cons a t ih => simp only [List.foldl_cons, List.foldr_cons, ih]; rw [max_assoc]

set_option maxHeartbeats 1000000 in
/-- **`l_i' = aftStateBot((c+1)·64).2.1` (single-pT).** From the kernel seed state
`aftStateBotK(c·64)` the masked `l_i·α + Σ pT` register equals the seed-`0`
denominator after `c+1` blocks. -/
theorem aft_denom_reg_eq (s0 : BlockState) (Q K V : RegionName) (keyScale : Fin 128 → ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i : Fin 128)
    (ltile mtile mijT alphaT : Tile .real [128]) (pT : Tile .real [128, 64])
    (hltile : ltile.data (i, PUnit.unit) = some
        ((aftStateBotK (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
            (qStartAFT s0) (c * 64) i ⟨0, by norm_num⟩).2.1))
    (hmtile : mtile.data (i, PUnit.unit)
        = aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
            (qStartAFT s0) (c * 64) i ⟨0, by norm_num⟩)
    (hmij : mijT.data (i, PUnit.unit)
        = aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
            (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩)
    (halpha : alphaT = Tile.uop WithBot.realExp2
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
    (hpc : ∀ jL : Fin 64, pT.data (i, jL, PUnit.unit)
        = some (if c * 64 + jL.val ≤ qStartAFT s0 + i.val then
            pow2 (aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩
              - (aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
                  (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩).unbotD 0)
            else 0)) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
        (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile alphaT)
        (Tile.reduceSumDrop aftAx1 pT)).data (i, PUnit.unit)
      = some ((aftStateBot (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
          (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩).2.1) := by
  set qT := qMaskedAFT s0 Q; set kT := kTileAFT s0 K; set vT := vMaskedAFT s0 V
  set kc := keyScale; set qStart := qStartAFT s0
  set m := (aftStateBot qT kT vT kc qStart (c * 64) i ⟨0, by norm_num⟩).1 with hm_def
  set Mc := aftRunningMax qT kT vT kc qStart (c * 64) i ⟨0, by norm_num⟩ with hMc
  set Mc1 := aftRunningMax qT kT vT kc qStart ((c + 1) * 64) i ⟨0, by norm_num⟩ with hMc1
  have hmMc : m = Mc := by rw [hm_def, hMc, aftStateBot_fst_eq_runningMax]
  have hMsucc : Mc1 = m ⊔ ((aftBlock qT kT vT kc qStart c i ⟨0, by norm_num⟩).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    have h1 : Mc1 = (aftStateBot qT kT vT kc qStart ((c + 1) * 64) i ⟨0, by norm_num⟩).1 := by
      rw [hMc1, aftStateBot_fst_eq_runningMax]
    rw [h1, aftStateBot_succ, osStepBot_block_fst m
        ((aftStateBot qT kT vT kc qStart (c * 64) i ⟨0, by norm_num⟩).2.1)
        ((aftStateBot qT kT vT kc qStart (c * 64) i ⟨0, by norm_num⟩).2.2)]
  have halphaVal : alphaT.data (i, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [halpha]; show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Tile.uop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmtile, hmij,
      NumericDType.sub, ← hMc, ← hMc1, hmMc]
  have hsum := aft_nume_row_sum s0 Q K V keyScale c hc1 i ⟨0, by norm_num⟩ pT hpc
  have hblockEq := osStepBot_block_eq m
    ((aftStateBot qT kT vT kc qStart (c * 64) i ⟨0, by norm_num⟩).2.1)
    ((aftStateBot qT kT vT kc qStart (c * 64) i ⟨0, by norm_num⟩).2.2)
    ((aftKeysUpto qT kT vT kc qStart (c * 64) i ⟨0, by norm_num⟩).map (fun p => pow2 p.1 * p.2)).sum
    ((aftKeysUpto qT kT vT kc qStart (c * 64) i ⟨0, by norm_num⟩).map (fun p => pow2 p.1)).sum
    (aftBlock qT kT vT kc qStart c i ⟨0, by norm_num⟩)
    (by rw [aftStateBot_snd_fst, zero_add, hm_def, aftStateBot_fst_eq_runningMax])
    (by rw [aftStateBot_snd_snd, zero_add, hm_def, aftStateBot_fst_eq_runningMax])
    (fun hbot => aftKeysUpto_sum_zero_of_bot qT kT vT kc qStart (c * 64) i ⟨0, by norm_num⟩
      (by rw [← aftStateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => aftKeysUpto_sum_zero_of_bot qT kT vT kc qStart (c * 64) i ⟨0, by norm_num⟩
      (by rw [← aftStateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  rw [show (aftStateBot qT kT vT kc qStart ((c + 1) * 64) i ⟨0, by norm_num⟩).2.1
        = (Mc1, (aftStateBot qT kT vT kc qStart (c * 64) i ⟨0, by norm_num⟩).2.1
              * (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
              + ((aftBlock qT kT vT kc qStart c i ⟨0, by norm_num⟩).map (fun p => pow2 (p.1 - Mc1.unbotD 0))).sum,
            _).2.1 from by
    rw [aftStateBot_succ]; rw [← hblockEq]]
  set α : ℝ := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp2 (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  have hcancel := (aftStateBotK_cancel qT kT vT kc qStart c i ⟨0, by norm_num⟩ Mc1).1
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex]
  erw [hsum]
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    hltile, halphaVal, hαsome]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [show (aftStateBotK qT kT vT kc qStart (c * 64) i ⟨0, by norm_num⟩).2.1 * α
        = (aftStateBot qT kT vT kc qStart (c * 64) i ⟨0, by norm_num⟩).2.1 * α from hcancel]

set_option maxHeartbeats 1000000 in
/-- **`acc' = aftStateBot((c+1)·64).2.2` (single-pT).** -/
theorem aft_acc_reg_eq (s0 : BlockState) (Q K V : RegionName) (keyScale : Fin 128 → ℝ)
    (c : Nat) (hc1 : (c + 1) * 64 ≤ 128) (i d : Fin 128)
    (acctile acc1T : Tile .real [128, 128]) (pT : Tile .real [128, 64]) (vtile : Tile .real [64, 128])
    (mtile mijT alphaT : Tile .real [128])
    (hv : ∀ idx : TileIndex [64, 128],
        vtile.data idx = some (vMaskedAFT s0 V
          (⟨c * 64 + idx.1.val, by have := idx.1.isLt; omega⟩, idx.2.1, PUnit.unit)))
    (hacctile : acctile.data (i, d, PUnit.unit) = some
        ((aftStateBotK (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
            (qStartAFT s0) (c * 64) i d).2.2))
    (hmtile : mtile.data (i, PUnit.unit)
        = aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
            (qStartAFT s0) (c * 64) i ⟨0, by norm_num⟩)
    (hmij : mijT.data (i, PUnit.unit)
        = aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
            (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩)
    (halpha : alphaT = Tile.uop WithBot.realExp2
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
    (hacc1 : acc1T = Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
    (hpc : ∀ jL : Fin 64, pT.data (i, jL, PUnit.unit)
        = some (if c * 64 + jL.val ≤ qStartAFT s0 + i.val then
            pow2 (aftBlockScore s0 Q K keyScale i ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩
              - (aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
                  (qStartAFT s0) ((c + 1) * 64) i ⟨0, by norm_num⟩).unbotD 0)
            else 0)) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        acc1T (Tile.dot [] pT vtile)).data (i, d, PUnit.unit)
      = some ((aftStateBot (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale
          (qStartAFT s0) ((c + 1) * 64) i d).2.2) := by
  set qT := qMaskedAFT s0 Q; set kT := kTileAFT s0 K; set vT := vMaskedAFT s0 V
  set kc := keyScale; set qStart := qStartAFT s0
  set m := (aftStateBot qT kT vT kc qStart (c * 64) i d).1 with hm_def
  set Mc := aftRunningMax qT kT vT kc qStart (c * 64) i ⟨0, by norm_num⟩ with hMc
  set Mc1 := aftRunningMax qT kT vT kc qStart ((c + 1) * 64) i ⟨0, by norm_num⟩ with hMc1
  have hmMc : m = Mc := by
    rw [hm_def, hMc, aftStateBot_fst_eq_runningMax,
      aftRunningMax_eq qT kT vT kc qStart (c * 64) i d ⟨0, by norm_num⟩]
  have hMsucc : Mc1 = m ⊔ ((aftBlock qT kT vT kc qStart c i d).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    have h1 : Mc1 = (aftStateBot qT kT vT kc qStart ((c + 1) * 64) i d).1 := by
      rw [hMc1, aftStateBot_fst_eq_runningMax,
        aftRunningMax_eq qT kT vT kc qStart ((c + 1) * 64) i ⟨0, by norm_num⟩ d]
    rw [h1, aftStateBot_succ, osStepBot_block_fst m
        ((aftStateBot qT kT vT kc qStart (c * 64) i d).2.1)
        ((aftStateBot qT kT vT kc qStart (c * 64) i d).2.2)]
  have halphaVal : alphaT.data (i, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [halpha]; show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Tile.uop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmtile, hmij,
      NumericDType.sub, ← hMc, ← hMc1, hmMc]
  have hdot := aft_acc_dot_block s0 Q K V keyScale c hc1 i d pT vtile hv hpc
  have hblockEq := osStepBot_block_eq m
    ((aftStateBot qT kT vT kc qStart (c * 64) i d).2.1)
    ((aftStateBot qT kT vT kc qStart (c * 64) i d).2.2)
    ((aftKeysUpto qT kT vT kc qStart (c * 64) i d).map (fun p => pow2 p.1 * p.2)).sum
    ((aftKeysUpto qT kT vT kc qStart (c * 64) i d).map (fun p => pow2 p.1)).sum
    (aftBlock qT kT vT kc qStart c i d)
    (by rw [aftStateBot_snd_fst, zero_add, hm_def, aftStateBot_fst_eq_runningMax])
    (by rw [aftStateBot_snd_snd, zero_add, hm_def, aftStateBot_fst_eq_runningMax])
    (fun hbot => aftKeysUpto_sum_zero_of_bot qT kT vT kc qStart (c * 64) i d
      (by rw [← aftStateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => aftKeysUpto_sum_zero_of_bot qT kT vT kc qStart (c * 64) i d
      (by rw [← aftStateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  rw [show (aftStateBot qT kT vT kc qStart ((c + 1) * 64) i d).2.2
        = (Mc1, _,
            (aftStateBot qT kT vT kc qStart (c * 64) i d).2.2
              * (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
              + ((aftBlock qT kT vT kc qStart c i d).map (fun p => pow2 (p.1 - Mc1.unbotD 0) * p.2)).sum).2.2
        from by rw [aftStateBot_succ]; rw [← hblockEq]]
  set α : ℝ := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp2 (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  have hcancel := (aftStateBotK_cancel qT kT vT kc qStart c i d Mc1).2
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex]
  erw [hdot]
  rw [hacc1, Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, hacctile, halphaVal, hαsome]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [show (aftStateBotK qT kT vT kc qStart (c * 64) i d).2.2 * α
        = (aftStateBot qT kT vT kc qStart (c * 64) i d).2.2 * α from hcancel]

/-! ## FINAL Part 2 (foundation) — genuine per-key score scale `aftKeyScale`

`aftKeyScale s0 QScale KScale j = q_scale · k_scale[j/64]` reads the program's
`q_scale` (`QScale[q_scale_offset + start_m]`) and the per-block `k_scale`
(`KScale[k_scale_offset + j/64]`). The remaining `aft_attn_step` threads
`aftLoopBody_steps` + the masked bridges against this scale via the block-`c`
agreement `aftKeyScale_block`. -/

/-- The genuine per-key score scale: `q_scale · k_scale[j/64]` for the program. -/
noncomputable def aftKeyScale (s0 : BlockState) (QScale KScale : RegionName) :
    Fin 128 → ℝ :=
  fun j => (s0.readMem QScale (s0.pids 1 * ((128 + 128 - 1) / 128) + s0.pids 0))
    * (s0.readMem KScale (s0.pids 1 * ((128 + 64 - 1) / 64) + j.val / 64))

/-- The block-`c` agreement: on block `c`'s keys, `aftKeyScale` equals the loaded
`q_scale · k_scale[c]`. -/
theorem aftKeyScale_block (s0 : BlockState) (QScale KScale : RegionName) (c : Nat)
    (jL : Fin 64) (hb : c * 64 + jL.val < 128) :
    aftKeyScale s0 QScale KScale ⟨c * 64 + jL.val, hb⟩
      = (s0.readMem QScale (s0.pids 1 * ((128 + 128 - 1) / 128) + s0.pids 0))
        * (s0.readMem KScale (s0.pids 1 * ((128 + 64 - 1) / 64) + c)) := by
  unfold aftKeyScale
  have hdiv : (c * 64 + jL.val) / 64 = c := by have := jL.isLt; omega
  simp only [hdiv]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **AFT streaming block step.** `aftInvariant i → aftInvariant (i+64)` over
`aftLoopBody`, given the score bound (`-1e6` floor inert). -/
theorem aft_attn_step (Q K V QScale KScale Out : RegionName) (s0 : BlockState)
    (i : Nat) (s : BlockState) (hilt : i < 128) (himod : i % 64 = 0)
    (hsb : aftScoreBound s0 Q K (aftKeyScale s0 QScale KScale))
    (hinv : aftInvariant Q K V QScale KScale Out s0 (aftKeyScale s0 QScale KScale) i s) :
    ∃ s', stepStmts AftFoundation.aftLoopBody (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ aftInvariant Q K V QScale KScale Out s0 (aftKeyScale s0 QScale KScale) (i + 64) s' := by
  set keyScale := aftKeyScale s0 QScale KScale with hkeyScale
  set qStart := qStartAFT s0 with hqStart
  simp only [aftInvariant] at hinv
  obtain ⟨hpids, hmod, hile, hmi, hli, hacc, hoffsm, hoffsn, hoffsk, hstartm, hoffhz,
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
  -- scalar scales
  set qsc : ℝ := s0.readMem QScale (s0.pids 1 * ((128 + 128 - 1) / 128) + s0.pids 0) with hqscv
  set ksc : ℝ := s0.readMem KScale (s0.pids 1 * ((128 + 64 - 1) / 64) + c) with hkscv
  -- the supplied k_mask / k tiles (set-shielded)
  set kmaskTile : Tile .bool [128, 64] :=
    ⟨fun idx : TileIndex [128, 64] =>
      (ComparableDType.nat.lt ((Tile.vec (fun j : Fin 64 => j.val)).data (idx.2.1, PUnit.unit)) (128 - i))
        && (ComparableDType.nat.lt idx.1.val 96)⟩ with hkmaskTile
  set ktileB : Tile .real [128, 64] :=
    ⟨fun idx : TileIndex [128, 64] =>
      if kmaskTile.data idx then some (sin.readMem ((kPtrsAFT s0 K i).data idx).1 ((kPtrsAFT s0 K i).data idx).2)
      else some (sin.undef ((kPtrsAFT s0 K i).data idx).1 ((kPtrsAFT s0 K i).data idx).2)⟩ with hktileB
  set kscT : Tile .real [] :=
    ⟨fun _ : TileIndex [] => some (sin.readMem ((kScalePtrAFT s0 KScale i).data PUnit.unit).1
      ((kScalePtrAFT s0 KScale i).data PUnit.unit).2)⟩ with hkscT
  set vmaskTile : Tile .bool [64, 128] :=
    ⟨fun idx : TileIndex [64, 128] =>
      (ComparableDType.nat.lt ((Tile.vec (fun j : Fin 64 => j.val)).data (idx.1, PUnit.unit)) (128 - i))
        && (ComparableDType.nat.lt idx.2.1.val 96)⟩ with hvmaskTile
  set vtileB : Tile .real [64, 128] :=
    ⟨fun idx : TileIndex [64, 128] =>
      if vmaskTile.data idx then some (sin.readMem ((vPtrsAFT s0 V i).data idx).1 ((vPtrsAFT s0 V i).data idx).2)
      else some (sin.undef ((vPtrsAFT s0 V i).data idx).1 ((vPtrsAFT s0 V i).data idx).2)⟩ with hvtileB
  obtain ⟨sF, hchain, hpidsF, hmemF, hundefF,
      maskT, qkRawT, qk6T, rmaxT, mijT, pT, lijT, alphaT,
      hmaskd, hqkRawd, hqk6d, hrm, hmijd, hpd, hlijd, halphad, hFmi, hFli, hFacc, hFKp, hFKsp, hFVp⟩ :=
    aftLoopBody_steps sin i
      (Tile.vec (fun r : Fin 128 => qStart + r.val)) (Tile.vec (fun j : Fin 64 => j.val))
      (qLoadedAFT s0 Q) (qScaleAFT s0 QScale) kscT
      ktileB kmaskTile vtileB vmaskTile
      ⟨fun r : TileIndex [128] => aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart i r.1 ⟨0, by norm_num⟩⟩
      ⟨fun r : TileIndex [128] => ((aftStateBotK (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart i r.1 ⟨0, by norm_num⟩).2.1 : ℝ)⟩
      ⟨fun idx : TileIndex [128, 128] => ((aftStateBotK (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart i idx.1 idx.2.1).2.2 : ℝ)⟩
      (kPtrsAFT s0 K i) (kScalePtrAFT s0 KScale i) (vPtrsAFT s0 V i)
      (by rw [hsin]; simpa using BlockState.setReg_same _ _ _ _ _)
      (by rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffsm)
      (by rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffsn)
      (by rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq)
      (by rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hqs)
      (by rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmi)
      (by rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hli)
      (by rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hacc)
      (by rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hKp)
      (by rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hKsp)
      (by rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hVp)
      (fun idx => rfl)
      (fun idx => rfl)
      rfl
      (fun idx => rfl)
      (fun idx => rfl)
      hundef'
  refine ⟨sF, hchain, ?_⟩
  -- k_scale cell value
  have hkscData : kscT.data PUnit.unit = some ksc := by
    rw [hkscT]; simp only [kScalePtrAFT, Region.cast, hkscv]
    refine congrArg some ?_
    show sin.readMem KScale (s0.pids 1 * ((128 + 64 - 1) / 64) + i / 64) = s0.readMem KScale _
    rw [show i / 64 = c from by rw [hc_def]]
    unfold BlockState.readMem; rw [hmem']
  -- keyScale at a block-c key = qsc * ksc
  have hkeyBlock : ∀ jL : Fin 64,
      keyScale (⟨c * 64 + jL.val, by have := jL.isLt; omega⟩ : Fin 128) = qsc * ksc := by
    intro jL
    rw [hkeyScale, aftKeyScale_block s0 QScale KScale c jL (by have := jL.isLt; omega), ← hqscv, ← hkscv]
  -- full-k readback tile (the bridges want this); equals masked dot via aft_dot_kmask_irrel
  set kfullT : Tile .real [128, 64] :=
    ⟨fun idx : TileIndex [128, 64] => some (s0.readMem K
      (baseOffsetAFT s0 + idx.1.val + (c * 64 + idx.2.1.val) * 128))⟩ with hkfullT
  have hkfull : ∀ idx : TileIndex [128, 64],
      kfullT.data idx = some (s0.readMem K (baseOffsetAFT s0 + idx.1.val + (c * 64 + idx.2.1.val) * 128)) :=
    fun idx => rfl
  -- masked ktile agrees with full at e < 96
  have hkagree : ∀ (jL : Fin 64) (e : Fin 128), e.val < 96 →
      ktileB.data (e, jL, PUnit.unit) = kfullT.data (e, jL, PUnit.unit) := by
    intro jL e he
    rw [hktileB, hkfullT]
    have hmaskTrue : kmaskTile.data (e, jL, PUnit.unit) = Bool.true := by
      rw [hkmaskTile]; simp only [Tile.vec_data]
      rw [Bool.and_eq_true]; constructor
      · rw [ComparableDType.nat_lt_eq_true]; have := jL.isLt; omega
      · rw [ComparableDType.nat_lt_eq_true]; exact he
    simp only [hmaskTrue, if_true, kPtrsAFT, Region.cast]
    refine congrArg some ?_
    show sin.readMem K _ = s0.readMem K _
    rw [show baseOffsetAFT s0 + e.val + jL.val * 128 + i * 128
          = baseOffsetAFT s0 + e.val + (c * 64 + jL.val) * 128 from by rw [hi]; ring]
    unfold BlockState.readMem; rw [hmem']
  have hksome : ∀ (jL : Fin 64) (e : Fin 128), 96 ≤ e.val →
      ∃ v : ℝ, ktileB.data (e, jL, PUnit.unit) = some v := by
    intro jL e he
    rw [hktileB]; cases h : kmaskTile.data (e, jL, PUnit.unit) <;> simp [h]
  have hfsome : ∀ (jL : Fin 64) (e : Fin 128), 96 ≤ e.val →
      ∃ w : ℝ, kfullT.data (e, jL, PUnit.unit) = some w := fun jL e _ => ⟨_, rfl⟩
  -- the dot tiles agree (masked k vs full k), so loop qkRawT = full qkRawT
  have hdoteq : (⟨fun ix => (Tile.dot [] (qLoadedAFT s0 Q) ktileB).data ix⟩ : Tile .real [128, 64])
      = ⟨fun ix => (Tile.dot [] (qLoadedAFT s0 Q) kfullT).data ix⟩ := by
    refine Tile.ext (fun ix => ?_)
    obtain ⟨ir, jL, ⟨⟩⟩ := ix
    exact aft_dot_kmask_irrel s0 Q ir jL ktileB kfullT (hkagree jL) (hksome jL) (hfsome jL)
  set qkRawFull : Tile .real [128, 64] :=
    Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (⟨fun ix => (Tile.dot [] (qLoadedAFT s0 Q) kfullT).data ix⟩ : Tile .real [128, 64])
        (Tile.scalar (some qsc))) (Tile.scalar (some ksc)) with hqkRawFull
  have hqScaleEq : qScaleAFT s0 QScale = Tile.scalar (some qsc) := by
    refine Tile.ext (fun u => ?_); obtain ⟨⟩ := u
    show some (s0.readMem QScale _) = some qsc
    rw [← hqscv]
  have hkscEq : kscT = Tile.scalar (some ksc) := by
    refine Tile.ext (fun u => ?_); obtain ⟨⟩ := u
    show kscT.data PUnit.unit = (Tile.scalar (some ksc)).data PUnit.unit
    rw [hkscData]; rfl
  have hqkRawEq : qkRawT = qkRawFull := by
    rw [hqkRawd, hqkRawFull, hdoteq, hqScaleEq, hkscEq]
  -- mask cell: ge(qStart + i, c*64 + jL)
  have hmaskcell : ∀ (ir : Fin 128) (jL : Fin 64),
      maskT.data (ir, jL, PUnit.unit)
        = ComparableDType.nat.ge (qStart + ir.val) (c * 64 + jL.val) := by
    intro ir jL
    rw [hmaskd]; simp only [Tile.vec_data, hi]
  -- qk6 cell in bridge form
  have hqk6cell : ∀ (ir : Fin 128) (jL : Fin 64),
      qk6T.data (ir, jL, PUnit.unit)
        = if maskT.data (ir, jL, PUnit.unit) then qkRawFull.data (ir, jL, PUnit.unit)
          else WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ)) := by
    intro ir jL; rw [hqk6d, hqkRawEq]
  -- assemble the new invariant
  simp only [aftInvariant, ← hqStart]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF, hpids']
  · omega
  · omega
  · -- m_i = aftRunningMax (i+64)
    rw [hFmi, hmijd]; refine congrArg some ?_; refine Tile.ext (fun r => ?_)
    obtain ⟨ir, ⟨⟩⟩ := r
    have hbr := aft_mij_reg_eq s0 Q K V keyScale qsc ksc c hc1 ir hkeyBlock hsb
      (qLoadedAFT s0 Q) kfullT qkRawFull maskT
      ⟨fun r : TileIndex [128] => aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart i r.1 ⟨0, by norm_num⟩⟩
      rmaxT rmaxT qk6T rfl hkfull hqkRawFull
      (fun jL => hmaskcell ir jL) (fun jL => hqk6cell ir jL)
      (by simp only [hi]; rfl) hrm
    rw [hbr, show ((c + 1) * 64 : Nat) = i + 64 from by omega]
  · -- l_i = aftStateBotK (i+64) .2.1
    rw [hFli]; refine congrArg some ?_; refine Tile.ext (fun r => ?_)
    obtain ⟨ir, ⟨⟩⟩ := r
    have hmijcell : mijT.data (ir, PUnit.unit)
        = aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart ((c + 1) * 64) ir ⟨0, by norm_num⟩ := by
      rw [hmijd]
      exact aft_mij_reg_eq s0 Q K V keyScale qsc ksc c hc1 ir hkeyBlock hsb
        (qLoadedAFT s0 Q) kfullT qkRawFull maskT
        ⟨fun r : TileIndex [128] => aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart i r.1 ⟨0, by norm_num⟩⟩
        rmaxT rmaxT qk6T rfl hkfull hqkRawFull
        (fun jL => hmaskcell ir jL) (fun jL => hqk6cell ir jL)
        (by simp only [hi]; rfl) hrm
    have hpc : ∀ jL : Fin 64, pT.data (ir, jL, PUnit.unit)
        = some (if c * 64 + jL.val ≤ qStart + ir.val then
            pow2 (aftBlockScore s0 Q K keyScale ir ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩
              - (aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart ((c + 1) * 64) ir ⟨0, by norm_num⟩).unbotD 0)
            else 0) := by
      intro jL
      exact aft_pT_cell s0 Q K V keyScale qsc ksc c ir jL (by have := jL.isLt; omega) (hkeyBlock jL)
        (qLoadedAFT s0 Q) kfullT qkRawFull qk6T maskT mijT pT rfl hkfull hqkRawFull
        (hmaskcell ir jL) (hqk6cell ir jL) hmijcell
        (by rw [hpd])
        (fun hkp => aftRunningMax_succ_ne_bot_of_kept s0 Q K V keyScale c ir ⟨0, by norm_num⟩ jL (by have := jL.isLt; omega) hkp)
    have hbr := aft_denom_reg_eq s0 Q K V keyScale c hc1 ir
      ⟨fun r : TileIndex [128] => ((aftStateBotK (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart i r.1 ⟨0, by norm_num⟩).2.1 : ℝ)⟩
      ⟨fun r : TileIndex [128] => aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart i r.1 ⟨0, by norm_num⟩⟩
      mijT alphaT pT
      (by simp only [hi]; rfl) (by simp only [hi]; rfl) hmijcell halphad hpc
    show (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil) _ lijT).data (ir, PUnit.unit)
      = ((aftStateBotK (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart (i + 64) ir ⟨0, by norm_num⟩).2.1 : ℝ)
    rw [hlijd, show ((i + 64) : Nat) = (c + 1) * 64 from by omega,
      show aftStateBotK (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart ((c + 1) * 64) ir ⟨0, by norm_num⟩
          = aftStateBot (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart ((c + 1) * 64) ir ⟨0, by norm_num⟩
        from by unfold aftStateBotK; exact if_neg (show (c + 1) * 64 ≠ 0 by omega)]
    exact hbr
  · -- acc = aftStateBotK (i+64) .2.2
    rw [hFacc]; refine congrArg some ?_; refine Tile.ext (fun idx => ?_)
    obtain ⟨ir, id, ⟨⟩⟩ := idx
    have hmijcell : mijT.data (ir, PUnit.unit)
        = aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart ((c + 1) * 64) ir ⟨0, by norm_num⟩ := by
      rw [hmijd]
      exact aft_mij_reg_eq s0 Q K V keyScale qsc ksc c hc1 ir hkeyBlock hsb
        (qLoadedAFT s0 Q) kfullT qkRawFull maskT
        ⟨fun r : TileIndex [128] => aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart i r.1 ⟨0, by norm_num⟩⟩
        rmaxT rmaxT qk6T rfl hkfull hqkRawFull
        (fun jL => hmaskcell ir jL) (fun jL => hqk6cell ir jL)
        (by simp only [hi]; rfl) hrm
    have hpc : ∀ jL : Fin 64, pT.data (ir, jL, PUnit.unit)
        = some (if c * 64 + jL.val ≤ qStart + ir.val then
            pow2 (aftBlockScore s0 Q K keyScale ir ⟨c * 64 + jL.val, by have := jL.isLt; omega⟩
              - (aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart ((c + 1) * 64) ir ⟨0, by norm_num⟩).unbotD 0)
            else 0) := by
      intro jL
      exact aft_pT_cell s0 Q K V keyScale qsc ksc c ir jL (by have := jL.isLt; omega) (hkeyBlock jL)
        (qLoadedAFT s0 Q) kfullT qkRawFull qk6T maskT mijT pT rfl hkfull hqkRawFull
        (hmaskcell ir jL) (hqk6cell ir jL) hmijcell
        (by rw [hpd])
        (fun hkp => aftRunningMax_succ_ne_bot_of_kept s0 Q K V keyScale c ir ⟨0, by norm_num⟩ jL (by have := jL.isLt; omega) hkp)
    -- the masked v-load cell = vMaskedAFT value
    have hvload : ∀ jL : Fin 64,
        (⟨fun i => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (pT.data i))⟩ : Tile .real [128, 64]).data (ir, jL, PUnit.unit)
          = pT.data (ir, jL, PUnit.unit) := by
      intro jL
      simp only [FloatDType.cast, FloatDType.real_toWithBot, FloatDType.fp16_ofWithBot,
        FloatDType.fp16_toWithBot, FloatDType.real_ofWithBot]
    -- the masked v-load: full readback form expected by aft_acc_reg_eq (`hv`)
    have hvfull : ∀ idx : TileIndex [64, 128],
        vtileB.data idx = some (vMaskedAFT s0 V
          (⟨c * 64 + idx.1.val, by have := idx.1.isLt; omega⟩, idx.2.1, PUnit.unit)) := by
      rintro ⟨jr, jd, ⟨⟩⟩
      rw [hvtileB, hvmaskTile]; simp only [Tile.vec_data, vPtrsAFT, Region.cast, vMaskedAFT, vTileAFT]
      by_cases hjd : jd.val < 96
      · have hmt : ((ComparableDType.nat.lt jr.val (128 - i)) && (ComparableDType.nat.lt jd.val 96)) = Bool.true := by
          rw [Bool.and_eq_true]
          refine ⟨?_, ?_⟩
          · rw [ComparableDType.nat_lt_eq_true]; have := jr.isLt; omega
          · rw [ComparableDType.nat_lt_eq_true]; exact hjd
        rw [if_pos hmt, if_pos hjd]
        refine congrArg some ?_
        show sin.readMem V _ = s0.readMem V _
        rw [show baseOffsetAFT s0 + jr.val * 128 + jd.val + i * 128
              = baseOffsetAFT s0 + (c * 64 + jr.val) * 128 + jd.val from by rw [hi]; ring]
        unfold BlockState.readMem; rw [hmem']
      · have hmf : ((ComparableDType.nat.lt jr.val (128 - i)) && (ComparableDType.nat.lt jd.val 96)) = Bool.false := by
          rw [Bool.and_eq_false_iff]; right
          rw [← Bool.not_eq_true, ComparableDType.nat_lt_eq_true]; omega
        rw [if_neg (by rw [hmf]; simp), if_neg hjd]
        exact congrArg some (hundef' _ _)
    have hbr := aft_acc_reg_eq s0 Q K V keyScale c hc1 ir id
      ⟨fun idx : TileIndex [128, 128] => ((aftStateBotK (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart i idx.1 idx.2.1).2.2 : ℝ)⟩
      (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        ⟨fun idx : TileIndex [128, 128] => ((aftStateBotK (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart i idx.1 idx.2.1).2.2 : ℝ)⟩
        (Tile.expandDim ⟨1, by simp⟩ alphaT))
      ⟨fun ix => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (pT.data ix))⟩
      vtileB
      ⟨fun r : TileIndex [128] => aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart i r.1 ⟨0, by norm_num⟩⟩
      mijT alphaT
      hvfull
      (by simp only [hi]; rfl) (by simp only [hi]; rfl) hmijcell halphad rfl
      (fun jL => by
        show (⟨fun ix => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (pT.data ix))⟩ : Tile .real [128, 64]).data (ir, jL, PUnit.unit) = _
        rw [hvload jL]; exact hpc jL)
    show (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) _
        (Tile.dot [] _ vtileB)).data (ir, id, PUnit.unit)
      = ((aftStateBotK (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart (i + 64) ir id).2.2 : ℝ)
    rw [show ((i + 64) : Nat) = (c + 1) * 64 from by omega,
      show aftStateBotK (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart ((c + 1) * 64) ir id
          = aftStateBot (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale qStart ((c + 1) * 64) ir id
        from by unfold aftStateBotK; exact if_neg (show (c + 1) * 64 ≠ 0 by omega)]
    exact hbr
  · -- offs_m (unassigned by the body → frame preserved)
    rw [stepStmts_regs_frame (d := .nat) (sh := [128]) (name := "offs_m")
      (aftLoopBody_stmtNotAssign _ (by decide)) hchain]
    rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffsm
  · -- offs_n
    rw [stepStmts_regs_frame (d := .nat) (sh := [64]) (name := "offs_n")
      (aftLoopBody_stmtNotAssign _ (by decide)) hchain]
    rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffsn
  · -- offs_k
    rw [stepStmts_regs_frame (d := .nat) (sh := [128]) (name := "offs_k")
      (aftLoopBody_stmtNotAssign _ (by decide)) hchain]
    rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffsk
  · -- start_m
    rw [stepStmts_regs_frame (d := .nat) (sh := []) (name := "start_m")
      (aftLoopBody_stmtNotAssign _ (by decide)) hchain]
    rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hstartm
  · -- off_hz
    rw [stepStmts_regs_frame (d := .nat) (sh := []) (name := "off_hz")
      (aftLoopBody_stmtNotAssign _ (by decide)) hchain]
    rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffhz
  · -- q (unassigned)
    rw [stepStmts_regs_frame (d := .real) (sh := [128, 128]) (name := "q")
      (aftLoopBody_stmtNotAssign _ (by decide)) hchain]
    rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq
  · -- q_scale (unassigned)
    rw [stepStmts_regs_frame (d := .real) (sh := []) (name := "q_scale")
      (aftLoopBody_stmtNotAssign _ (by decide)) hchain]
    rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hqs
  · -- K_ptrs (advanced)
    rw [hFKp, kPtrsAFT_succ]
  · -- K_scale_ptr (advanced)
    rw [hFKsp, kScalePtrAFT_succ s0 KScale i himod]
  · -- V_ptrs (advanced)
    rw [hFVp, vPtrsAFT_succ]
  · -- O_block_ptr (unassigned)
    rw [stepStmts_regs_frame (d := .ptr) (sh := [128, 128]) (name := "O_block_ptr")
      (aftLoopBody_stmtNotAssign _ (by decide)) hchain]
    rw [hsin, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hOp
  · intro rg o; exact hundefF rg o
  · rw [hmemF, hmem']

/-! ## FINAL Part 4 — `aftPostLoop_eval` (finalize + masked O store off the closed form) -/

/-- Injectivity of the `O_block_ptr` per-lane output addresses (Python shape). -/
theorem aft_O_offset_injective (s0 : BlockState) :
    Function.Injective
      (fun idx : TileIndex [128, 128] =>
        baseOffsetAFT s0 + (qStartAFT s0 + idx.1.val) * 128 + idx.2.1.val) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp only at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb; subst db; rfl

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **PostLoop evaluation.** From a loop-end state satisfying `aftInvariant … 128`
(with the running max `≠ ⊥` on every output lane — the causal mask keeps key `0`),
stepping the 2 `aftPostLoop` statements (`acc = acc / l_i[:,None]` finalize + the
masked `tl.store(O_block_ptr, acc, mask=active)`) lands the genuine closed-form
attention ratio `attnFwdTritonOutSpec` at every active output lane. -/
theorem aftPostLoop_eval (Q K V QScale KScale Out : RegionName) (s0 : BlockState) (s : BlockState)
    (hne : ∀ i d : Fin 128, aftRunningMax (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V)
      (aftKeyScale s0 QScale KScale) (qStartAFT s0) 128 i d ≠ ⊥)
    (hinv : aftInvariant Q K V QScale KScale Out s0 (aftKeyScale s0 QScale KScale) 128 s) :
    ∃ sP, stepStmts (AftFoundation.aftPostLoop Out) s = some sP
      ∧ (∀ idx : TileIndex [128, 128],
          active s0 128 96 128 idx →
          sP.readMem Out (outOffset s0 4 65536 16384 128 1 128 idx)
            = attnFwdTritonOutSpec s0 Q K V (aftKeyScale s0 QScale KScale) idx) := by
  set keyScale := aftKeyScale s0 QScale KScale with hkeyScale
  simp only [aftInvariant] at hinv
  obtain ⟨_, _, _, _, hli, hacc, hoffsm, _, _, _, _, _, _, _, _, _, hOp, _, _⟩ := hinv
  set liTile : Tile .real [128] :=
    ⟨fun r : TileIndex [128] => ((aftStateBotK (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale (qStartAFT s0) 128 r.1 ⟨0, by norm_num⟩).2.1 : ℝ)⟩ with hliTile
  set accTile : Tile .real [128, 128] :=
    ⟨fun idx : TileIndex [128, 128] => ((aftStateBotK (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale (qStartAFT s0) 128 idx.1 idx.2.1).2.2 : ℝ)⟩ with haccTile
  unfold AftFoundation.aftPostLoop
  set accFin : Tile .real [128, 128] :=
    Tile.bop NumericDType.real.div (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      accTile (Tile.expandDim ⟨1, by simp⟩ liTile) with haccFin
  -- step 1: acc = acc / l_i[:, None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.div NumericDType.real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref TileDType.real [128, 128] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.real [128] "l_i"))) s
        = some accFin from by
      rw [evalOp_div]
      have hexp : @evalOp TileDType.real [128, 1]
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.real [128] "l_i")) s
          = some (Tile.expandDim ⟨1, by simp⟩ liTile) :=
        evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hli
      simp only [evalOp_ref, hacc, hexp, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  set s2 := s.setReg "acc" .real [128, 128] accFin with hs2
  have hOp2 : s2.regs .ptr [128, 128] "O_block_ptr" = some ⟨fun idx : TileIndex [128, 128] =>
      (Region.cast Out, baseOffsetAFT s0 + (qStartAFT s0 + idx.1.val) * 128 + idx.2.1.val)⟩ := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hOp
  have hacc2 : s2.regs .real [128, 128] "acc" = some accFin := by
    rw [hs2, BlockState.setReg_same]
  have hoffsm2 : s2.regs .nat [128] "offs_m" = some (Tile.vec (fun r : Fin 128 => qStartAFT s0 + r.val)) := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffsm
  set oOffFn : TileIndex [128, 128] → Nat :=
    fun idx => baseOffsetAFT s0 + (qStartAFT s0 + idx.1.val) * 128 + idx.2.1.val with hoOffFn
  set valFn : TileIndex [128, 128] → ℝ := fun idx => (accFin.data idx).unbotD 0 with hvalFn
  set P : TileIndex [128, 128] → Prop := fun idx => active s0 128 96 128 idx with hP
  -- step 2: masked tl.store reduces to the masked scatter foldl
  rw [stepStmts.cons_some
    (show stepStmt (Stmt.store .real [128, 128] (.ptr (.ref .ptr [128, 128] "O_block_ptr"))
        (Op.ref .real [128, 128] "acc")
        (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) (Op.constNat 128))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange 128) (Op.constNat 96)))))) s2
        = some ((TileShape.allIndices [128, 128]).foldl
            (fun acc idx => if P idx then acc.writeMem Out (oOffFn idx) (valFn idx) else acc) s2)
      from by
        simp only [stepStmt, evalOp, evalOp.eq_def, evalOp_ref, hacc2, hOp2, hoffsm2,
          evalOp_constNat, evalOp_arange, Option.bind_eq_bind, Option.bind_some, Option.map_some,
          Region.cast_id]
        refine congrArg some (List.foldl_ext _ _ s2 (fun acc idx _ => ?_))
        obtain ⟨ir, id, ⟨⟩⟩ := idx
        have hPunfold : P (ir, id, PUnit.unit) ↔ (s0.pids 0 * 128 + ir.val < 128 ∧ id.val < 96) := by
          rw [hP]; simp only [active, mIndex, kIndex]
        have hmaskeq : ((ComparableDType.nat.lt (qStartAFT s0 + ir.val) 128)
              && (ComparableDType.nat.lt id.val 96)) = decide (P (ir, id, PUnit.unit)) := by
          rw [Bool.eq_iff_iff, Bool.and_eq_true, ComparableDType.nat_lt_eq_true,
            ComparableDType.nat_lt_eq_true, decide_eq_true_eq, hPunfold, qStartAFT]
        simp only [Tile.bop_data, Tile.cop_data, Tile.expandDim_data, Tile.vec_data, Tile.scalar,
          Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex, hmaskeq]
        by_cases hpi : P (ir, id, PUnit.unit)
        · rw [if_pos (by rw [hmaskeq] at *; simpa using hpi), if_pos hpi,
            BlockState.writeMemTyped_real]
          simp only [FloatDType.real_storeValue, hoOffFn, hvalFn]
        · rw [if_neg (by simp [hpi]), if_neg hpi])]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx hactive
  obtain ⟨ir, id, ⟨⟩⟩ := idx
  have houtOff : outOffset s0 4 65536 16384 128 1 128 (ir, id, PUnit.unit) = oOffFn (ir, id, PUnit.unit) := by
    simp only [outOffset, offZ, offH, mIndex, kIndex, hoOffFn, baseOffsetAFT, qStartAFT]; ring
  rw [houtOff, BlockState.scatter_readback_prop_masked_nd s2 oOffFn valFn P
    (aft_O_offset_injective s0) (ir, id, PUnit.unit), if_pos hactive]
  -- valFn = accFin ratio = attnFwdTritonOutSpec
  have hlival : (aftStateBotK (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale (qStartAFT s0) 128 ir ⟨0, by norm_num⟩).2.1
      = (aftStateBot (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale (qStartAFT s0) 128 ir ⟨0, by norm_num⟩).2.1 := by
    unfold aftStateBotK; rw [if_neg (by norm_num)]
  have haccval : (aftStateBotK (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale (qStartAFT s0) 128 ir id).2.2
      = (aftStateBot (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale (qStartAFT s0) 128 ir id).2.2 := by
    unfold aftStateBotK; rw [if_neg (by norm_num)]
  have hdenom : (aftStateBot (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale (qStartAFT s0) 128 ir id).2.1
      = (aftStateBot (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale (qStartAFT s0) 128 ir ⟨0, by norm_num⟩).2.1 := by
    rw [aftStateBot_snd_fst_indep]
  rw [hvalFn, haccFin]
  simp only [Tile.bop_data, Tile.expandDim_data, TileShape.dropInsertedIndex,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div, WithBot.realDiv,
    Option.map₂, Option.bind, Option.map, haccTile, hliTile, hlival, haccval]
  rw [← hdenom]
  show (aftStateBot (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale (qStartAFT s0) 128 ir id).2.2
      / (aftStateBot (qMaskedAFT s0 Q) (kTileAFT s0 K) (vMaskedAFT s0 V) keyScale (qStartAFT s0) 128 ir id).2.1
    = attnFwdTritonOutSpec s0 Q K V keyScale (ir, id, PUnit.unit)
  exact aftStateBot_full_eq_spec s0 Q K V keyScale ir id (hne ir id)

/-! ## FINAL Part 4 (top) — genuine closed-form exec correctness + summary -/

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




/-- `evalOp` helper for `tl.math.exp2` (`Op.exp2`). -/
theorem aftg_evalOp_exp2 {shape : TileShape} (a : Op .real shape) (s : BlockState) :
    evalOp (.exp2 a) s = (do let va ← evalOp a s; some (Tile.uop WithBot.realExp2 va)) := by
  simp [evalOp]

/-- `evalOp` helper for the identity `real → real` cast (the `.to(tl.float32)`
fp32 wrapper around `tl.dot`, which is a no-op at the algorithm layer): given the
inner op's value, the cast returns the same tile (`FloatDType.cast real real` is
the identity through `WithBot ℝ`). -/
theorem aftg_castReal_eval {shape : TileShape} (a : Op .real shape) (s : BlockState)
    (va : Tile .real shape) (ha : evalOp a s = some va) :
    evalOp (Op.castFloat FloatDType.real FloatDType.real a) s
      = some ⟨fun i => va.data i⟩ := by
  rw [evalOp_castFloat]
  simp only [FloatDType.toTileDType_real]
  rw [ha]; rfl

/-- `evalOp` helper for the `>=` predicate (`Op.ge`), no `@[simp]` form. -/
theorem aftg_evalOp_ge {dtype a b shape} (h : ComparableDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

/-- `evalOp` helper for `Op.boolAnd`, no `@[simp]` form. -/
theorem aftg_evalOp_boolAnd {a b shape} (bc : Broadcast a b shape)
    (x : Op .bool a) (y : Op .bool b) (s : BlockState) :
    evalOp (.boolAnd bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s;
      some (Tile.bop (fun p q : Bool => p && q) bc vx vy)) := by
  simp [evalOp]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- `evalOp (Op.ptrBase R)` — the base pointer tile `(R, 0)` at the empty shape. -/
theorem aftg_evalOp_ptrBase {d : TileDType} (region : Region d) (s : BlockState) :
    evalOp (Op.ptrBase region) s = some (Tile.scalar (Region.cast region, 0)) := by
  simp only [evalOp]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Bind-aware `ptrAdd` reducer.** Given the operand evaluations
`evalOp ptr s = some ptrs` and `evalOp off s = some offs`, fire the `ptrAdd`
reduction inside the `Option.bind` do-block. Threads each `expandDim`-bearing
offset (pre-proved via `evalOp_expandDim_ref_of_regs` / `evalOp_add` / `evalOp_mul`)
into the elementwise `Tile.ptrAdd`. -/
theorem aftg_evalOp_ptrAdd_of {a b shape : TileShape}
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
Generalizes `aftg_load_k_eval`/`aftg_load_v_eval` to inline (non-`ref`) ptr/mask ops,
unblocking the preLoop `q` load (inline `boolAnd` mask). -/
theorem aftg_evalOp_load_ptr_mask_of {shape : TileShape}
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
theorem aftg_evalOp_load_ptr_none_of {shape : TileShape}
    (ptrOp : Op .ptr shape) (s : BlockState) (ptrs : Tile .ptr shape)
    (hptr : evalOp ptrOp s = some ptrs) :
    evalOp (.load .real (.ptr ptrOp) .none) s
      = some ⟨fun i : TileIndex shape =>
          some (s.readMem (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, hptr, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue_real, if_true]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L3: `k = tl.load(K_ptrs, mask=k_mask)`** — elementwise masked pointer load.
`K_ptrs` is a plain `[BM, BN]` pointer tile (per-lane `(region, offset)`); each
kept lane reads `s.readMem`, each masked-out lane reads `s.undef`. -/
theorem aftg_load_k_eval
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
theorem aftg_load_kscale_eval
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
theorem aftg_load_v_eval
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
/-- **L5: `qk = (castFloat (tl.dot(q, k))) * q_scale * k_scale`** — the `q·k` dot
(through an identity `real → real` fp32 cast) scaled by the two scalar
quantization factors `q_scale`, `k_scale`, each broadcast on the right. -/
theorem aftg_qk_dot_eval (s : BlockState) (BM BN BD : Nat)
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
    aftg_castReal_eval
      (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k"))
      s (Tile.dot [] qtile ktile) hdot
  rw [evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, hcast, hqsc, hksc, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L18: `acc += tl.dot(p.to(fp16).to(real), v)`** — numerator accumulation. The
fp16-cast `p` is cast back to real (identity at the algorithm layer) inside the
dot; `v` is real. -/
theorem aftg_acc_eval (s : BlockState) (BM BN BD : Nat)
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
/-- **L21: `K_scale_ptr += 1`** — advance the scalar K-scale pointer by one key
block. -/
theorem aftg_advance_kscale_eval (s : BlockState) (name : RegName) (ptr : Tile .ptr [])
    (hptr : s.regs .ptr [] name = some ptr) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ref .ptr [] name) (Op.constNat 1)) s
      = some (Tile.ptrAdd Broadcast.nil ptr (Tile.scalar 1)) := by
  simp only [evalOp, evalOp_ref, evalOp_constNat, hptr, Option.bind_eq_bind, Option.bind_some]

/-- The running `max` component of an `osStepBot` fold is the `WithBot ⊔`-fold. -/
theorem aftgStateBot_fst
    (xs : List (ℝ × ℝ)) (m₀ : WithBot ℝ) (l₀ acc₀ : ℝ) :
    (xs.foldl osStepBot (m₀, l₀, acc₀)).1
      = (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldl (· ⊔ ·) m₀ := by
  induction xs generalizing m₀ l₀ acc₀ with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl_cons, List.map_cons]; rw [ih]; rfl

/-- The `WithBot ⊔`-fold is seed/direction-agnostic. -/
theorem aftg_foldl_sup_bot_eq_foldr (L : List (WithBot ℝ)) :
    L.foldl (· ⊔ ·) (⊥ : WithBot ℝ) = L.foldr (· ⊔ ·) (⊥ : WithBot ℝ) := by
  have gen : ∀ (m : WithBot ℝ), L.foldl (· ⊔ ·) m = m ⊔ L.foldr (· ⊔ ·) ⊥ := by
    induction L with
    | nil => intro m; simp
    | cons a t ih => intro m; simp only [List.foldl_cons, List.foldr_cons, ih]; rw [max_assoc]
  rw [gen ⊥, bot_sup_eq]

/-- Generic threshold-split for a `.val`-ascending list. Ported from flash-attn. -/
private theorem aftg_filterMap_window_split {n : Nat} (l : List (Fin n))
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

/-- `filterMap`-then-map-and-sum over `finRange n` equals the masked `Finset.sum`. -/
theorem aftg_filterMap_finRange_sum {α : Type*} (n : Nat)
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
theorem aftg_mem_le_foldr_sup (a : WithBot ℝ) :
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
theorem aftg_filterMap_foldr_sup (n : Nat) (P : Fin n → Prop) [DecidablePred P]
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

/-- Helper: `WithBot.realSub (some 0) (some 1e6) = some (-1000000)`. -/
theorem aftg_sentinel_eq : WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ)) = some (-1000000.0 : ℝ) := by
  simp only [WithBot.realSub, Option.map₂, Option.bind, Option.map]; norm_num

namespace AftgFoundation

open VeriTile.Triton

/-- General loop body (symbolic `BLOCK_M`/`BLOCK_N`/`BLOCK_DMODEL`/`HEAD_DIM`/
`N_CTX`/`HEAD_ACTIVE`). Mirrors `aftgLoopBody` with the test-shape numerals replaced
by the corresponding dimension parameters. -/
def aftgLoopBodyG (N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) : List Stmt :=
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

/-- General preLoop (22 deterministic prefix statements), symbolic dims with
contiguous strides (`stride_qm = HEAD_DIM`, `stride_qk = 1`, `stride_kn = HEAD_DIM`).
Mirrors `aftgPreLoop` with the test-shape numerals replaced by the corresponding
dimension parameters. -/
def aftgPreLoopG (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) : List Stmt :=
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
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N_CTX) (Op.constNat BLOCK_M)) (Op.constNat 1))
          (Op.constNat BLOCK_M))),
    Stmt.assign .nat [] "k_scale_offset"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N_CTX) (Op.constNat BLOCK_N)) (Op.constNat 1))
          (Op.constNat BLOCK_N))),
    Stmt.assign .nat [BLOCK_M] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)) (Op.arange BLOCK_M)),
    Stmt.assign .nat [BLOCK_N] "offs_n" (Op.arange BLOCK_N),
    Stmt.assign .nat [BLOCK_DMODEL] "offs_k" (Op.arange BLOCK_DMODEL),
    Stmt.assign .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [] "Q_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase QScale)
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "q_scale_offset") (Op.ref .nat [] "start_m"))),
    Stmt.assign .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM)))),
    Stmt.assign .ptr [] "K_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase KScale) (Op.ref .nat [] "k_scale_offset")),
    Stmt.assign .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .real [BLOCK_M] "m_i"
      (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf),
    Stmt.assign .real [BLOCK_M] "l_i"
      (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) (Op.const 1.0)),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc" (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "q"
      (Op.load .real (.ptr (.ref .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"))
        (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat N_CTX))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))))),
    Stmt.assign .real [] "q_scale"
      (Op.load .real (.ptr (.ref .ptr [] "Q_scale_ptr")) .none) ]

/-- General preLoop head — statements 0–10 of `aftgPreLoopG`. -/
def aftgPreLoopHeadG (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) : List Stmt :=
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
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N_CTX) (Op.constNat BLOCK_M)) (Op.constNat 1))
          (Op.constNat BLOCK_M))),
    Stmt.assign .nat [] "k_scale_offset"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N_CTX) (Op.constNat BLOCK_N)) (Op.constNat 1))
          (Op.constNat BLOCK_N))),
    Stmt.assign .nat [BLOCK_M] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)) (Op.arange BLOCK_M)),
    Stmt.assign .nat [BLOCK_N] "offs_n" (Op.arange BLOCK_N),
    Stmt.assign .nat [BLOCK_DMODEL] "offs_k" (Op.arange BLOCK_DMODEL) ]

/-- General preLoop tail — statements 11–21 of `aftgPreLoopG`. -/
def aftgPreLoopTailG (Q K V QScale KScale Out : RegionName)
    (HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) : List Stmt :=
  [ Stmt.assign .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [] "Q_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase QScale)
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "q_scale_offset") (Op.ref .nat [] "start_m"))),
    Stmt.assign .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM)))),
    Stmt.assign .ptr [] "K_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase KScale) (Op.ref .nat [] "k_scale_offset")),
    Stmt.assign .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .real [BLOCK_M] "m_i"
      (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf),
    Stmt.assign .real [BLOCK_M] "l_i"
      (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) (Op.const 1.0)),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc" (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "q"
      (Op.load .real (.ptr (.ref .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"))
        (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat N_CTX))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))))),
    Stmt.assign .real [] "q_scale"
      (Op.load .real (.ptr (.ref .ptr [] "Q_scale_ptr")) .none) ]

theorem aftgPreLoopG_eq_head_tail (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) :
    aftgPreLoopG Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE
      = aftgPreLoopHeadG stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL
        ++ aftgPreLoopTailG Q K V QScale KScale Out HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE :=
  rfl

/-- General postLoop (2 statements): `acc /= l_i[:, None]` then masked store to `O_block_ptr`. -/
def aftgPostLoopG (Out : RegionName) (N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE : Nat) : List Stmt :=
  [ Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc"
      (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "l_i"))),
    Stmt.store .real [BLOCK_M, BLOCK_DMODEL] (.ptr (.ref .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"))
      (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
      (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat N_CTX))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE))))) ]

/-- General loop-body head (statements 0–10). -/
def aftgLoopBodyHeadG (N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) : List Stmt :=
  List.take 11 (aftgLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE)

/-- General loop-body tail (statements 11–21). -/
def aftgLoopBodyTailG (N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) : List Stmt :=
  List.drop 11 (aftgLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE)

theorem aftgLoopBodyG_eq_head_tail (N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) :
    aftgLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE
      = aftgLoopBodyHeadG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE
        ++ aftgLoopBodyTailG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE := by
  rw [aftgLoopBodyHeadG, aftgLoopBodyTailG, List.take_append_drop]

end AftgFoundation

section General

open VeriTile.Triton

/-! ### General spec layer (genuine causal closed form over symbolic dims) -/

/-- General per-plane base offset: `off_z·stride_qz + off_h·stride_qh` with
`off_z = pids1 / H`, `off_h = pids1 % H`. -/
def baseOffsetAFT2G (s : BlockState) (stride_qz stride_qh H : Nat) : Nat :=
  s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh

/-- General query tile: row `i` = global query `pids0·BLOCK_M + i`, head lane `e`
(`stride_qm = HEAD_DIM`, `stride_qk = 1`). -/
noncomputable def qTileAFT2G (s : BlockState) (Q : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL : Nat) :
    TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
  fun (i, e, _) => s.readMem Q (baseOffsetAFT2G s stride_qz stride_qh H
    + (s.pids 0 * BLOCK_M + i.val) * HEAD_DIM + e.val)

/-- General key tile: row `j` (global key), head lane `e` (`K[base + j·HEAD_DIM + e]`). -/
noncomputable def kTileAFT2G (s : BlockState) (K : RegionName)
    (stride_qz stride_qh H HEAD_DIM SEQ BLOCK_DMODEL : Nat) :
    TileIndex [SEQ, BLOCK_DMODEL] → ℝ :=
  fun (j, e, _) => s.readMem K (baseOffsetAFT2G s stride_qz stride_qh H + j.val * HEAD_DIM + e.val)

/-- General value tile: row `j` (global key), head lane `d` (`V[base + j·HEAD_DIM + d]`). -/
noncomputable def vTileAFT2G (s : BlockState) (V : RegionName)
    (stride_qz stride_qh H HEAD_DIM SEQ BLOCK_DMODEL : Nat) :
    TileIndex [SEQ, BLOCK_DMODEL] → ℝ :=
  fun (j, d, _) => s.readMem V (baseOffsetAFT2G s stride_qz stride_qh H + j.val * HEAD_DIM + d.val)

/-- General global query row for output tile-row `i`. -/
def qStartAFT2G (s : BlockState) (BLOCK_M : Nat) : Nat := s.pids 0 * BLOCK_M

/-- General masked query tile: head-active (`e < HEAD_ACTIVE`) and query-row
boundary (`qStart + i < N_CTX`) masking, mirroring the kernel's `q` load mask. -/
noncomputable def qTileAFT2mG (s : BlockState) (Q : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE : Nat) :
    TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
  fun (i, e, u) =>
    if qStartAFT2G s BLOCK_M + i.val < N_CTX ∧ e.val < HEAD_ACTIVE then
      qTileAFT2G s Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL (i, e, u) else 0

/-- General masked value tile: head-active (`d < HEAD_ACTIVE`) masking. -/
noncomputable def vTileAFT2mG (s : BlockState) (V : RegionName)
    (stride_qz stride_qh H HEAD_DIM SEQ BLOCK_DMODEL HEAD_ACTIVE : Nat) :
    TileIndex [SEQ, BLOCK_DMODEL] → ℝ :=
  fun (j, d, u) =>
    if d.val < HEAD_ACTIVE then
      vTileAFT2G s V stride_qz stride_qh H HEAD_DIM SEQ BLOCK_DMODEL (j, d, u) else 0

/-- **General genuine closed form** (exp2, causal): predicate-masked base-2
per-key-scale attention with the `causalKeep qStart` mask, over the kernel's
actually-loaded masked q/v tiles. -/
noncomputable def attnFwdTritonOutSpecG
    (s : BlockState) (Q K V : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (keyScale : Fin (BLOCK_N * numKVBlocks) → ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  attentionRealBase2PerKeyScalePred
    (qTileAFT2mG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
    (kTileAFT2G s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
    (vTileAFT2mG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
    keyScale (fun i j => causalKeep (qStartAFT2G s BLOCK_M) i j) idx

/-- General per-key score scale carrier `q_scale · k_scale` (block `j / BLOCK_N`). -/
noncomputable def keyScaleAFT2G (s : BlockState) (QScale KScale : RegionName)
    (N_CTX BLOCK_M BLOCK_N numKVBlocks : Nat) : Fin (BLOCK_N * numKVBlocks) → ℝ :=
  fun j => s.readMem QScale (s.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M) + s.pids 0)
            * s.readMem KScale (s.pids 1 * ((N_CTX + BLOCK_N - 1) / BLOCK_N) + j.val / BLOCK_N)

/-- General streaming bridge: the closed form equals the `osStep` online-softmax
fold over the causal-masked key list. -/
theorem attnFwdTritonOutSpecG_eq_streaming
    (s : BlockState) (Q K V : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (keyScale : Fin (BLOCK_N * numKVBlocks) → ℝ)
    (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    attnFwdTritonOutSpecG s Q K V stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N
        BLOCK_DMODEL HEAD_ACTIVE numKVBlocks keyScale (i, d, PUnit.unit)
      = (let st := (attnKeyListPred
            (qTileAFT2mG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
            (kTileAFT2G s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
            (vTileAFT2mG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
            keyScale (fun i j => causalKeep (qStartAFT2G s BLOCK_M) i j) i d).foldl
              osStep (0, 0, 0)
         st.2.2 / st.2.1) := by
  simpa [attnFwdTritonOutSpecG] using
    VeriTile.Triton.attentionRealBase2PerKeyScalePred_eq_streaming
      (qTileAFT2mG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
      (kTileAFT2G s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
      (vTileAFT2mG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
      keyScale (fun i j => causalKeep (qStartAFT2G s BLOCK_M) i j) i d

/-! ### General ⊥-seed online-softmax foundation math

Mirrors the pinned `aftgKV`/`aftgKeysUpto`/`aftgBlock`/`aftgRunningMax`/`aftgStateBot`/
`aftgStateBot1` over symbolic `BLOCK_M`(query rows)/`BLOCK_DMODEL`(channels)/
`SEQ`(keys)/`BLOCK_N`(block stride). Reuses the dim-agnostic generic core
(`osStepBot`, `osStepBot_foldl_consistent`, `osStepBot_block_eq`,
`osStepBot_bot_seed_indep`, `aftg_filterMap_window_split`, `aftg_filterMap_foldr_sup`,
`aftg_mem_le_foldr_sup`, `aftg_filterMap_finRange_sum`, `aftg_foldl_sup_bot_eq_foldr`). -/

variable {BLOCK_M BLOCK_DMODEL SEQ : Nat}

/-- General `(score, value)` pair the kernel streams for output `(i, d)` at key `j`. -/
noncomputable def aftgKVG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) (j : Fin SEQ) : ℝ × ℝ :=
  (keyScale j * Finset.univ.sum (fun e : Fin BLOCK_DMODEL => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)),
   vT (j, d, PUnit.unit))

/-- General causal per-row key list over `[0, hi)`: keys `j < hi` with `j ≤ qStart + i`. -/
noncomputable def aftgKeysUptoG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) : List (ℝ × ℝ) :=
  (List.finRange SEQ).filterMap (fun j : Fin SEQ =>
    if j.val < hi ∧ j.val ≤ qStart + i.val then
      some (aftgKVG qT kT vT keyScale i d j)
    else none)

/-- General block-`c` per-row key list (`c·BLOCK_N ≤ j < (c+1)·BLOCK_N`, causal). -/
noncomputable def aftgBlockG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) : List (ℝ × ℝ) :=
  (List.finRange SEQ).filterMap (fun j : Fin SEQ =>
    if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val then
      some (aftgKVG qT kT vT keyScale i d j)
    else none)

/-- General ⊥-seeded running max of the streamed key prefix `[0, hi)`. -/
noncomputable def aftgRunningMaxG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) : WithBot ℝ :=
  ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).map
    (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥

/-- General ⊥-seeded running `(max, denom, acc)` after streaming `[0, hi)` (seed-`0`). -/
noncomputable def aftgStateBotG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) : WithBot ℝ × ℝ × ℝ :=
  (aftgKeysUptoG qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 0, 0)

/-- General ⊥-seeded running state from the kernel's `l_i = 1` seed. -/
noncomputable def aftgStateBot1G
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) : WithBot ℝ × ℝ × ℝ :=
  (aftgKeysUptoG qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 1, 0)

variable (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
  (keyScale : Fin SEQ → ℝ)

/-- General: ⊥-seeded running max of `aftgStateBotG` is `aftgRunningMaxG`. -/
theorem aftgStateBotG_fst_eq_runningMax (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    (aftgStateBotG qT kT vT keyScale qStart hi i d).1
      = aftgRunningMaxG qT kT vT keyScale qStart hi i d := by
  rw [aftgStateBotG, aftgStateBot_fst, aftgRunningMaxG, aftg_foldl_sup_bot_eq_foldr]

/-- General ⊥-seeded denominator. -/
theorem aftgStateBotG_snd_fst (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    (aftgStateBotG qT kT vT keyScale qStart hi i d).2.1
      = ((aftgRunningMaxG qT kT vT keyScale qStart hi i d).elim 0 (fun r => pow2 (-r)))
        * (0 + ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1)).sum) := by
  rw [aftgStateBotG]
  rw [(osStepBot_foldl_consistent (aftgKeysUptoG qT kT vT keyScale qStart hi i d) ⊥ 0 0 0 0
    (by simp) (by simp) (by simp) (by simp)).1]
  rw [show ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 0, 0)).1
        = aftgRunningMaxG qT kT vT keyScale qStart hi i d from by
    rw [aftgStateBot_fst, aftgRunningMaxG, aftg_foldl_sup_bot_eq_foldr]]

/-- General ⊥-seeded accumulator. -/
theorem aftgStateBotG_snd_snd (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    (aftgStateBotG qT kT vT keyScale qStart hi i d).2.2
      = ((aftgRunningMaxG qT kT vT keyScale qStart hi i d).elim 0 (fun r => pow2 (-r)))
        * (0 + ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1 * p.2)).sum) := by
  rw [aftgStateBotG]
  rw [(osStepBot_foldl_consistent (aftgKeysUptoG qT kT vT keyScale qStart hi i d) ⊥ 0 0 0 0
    (by simp) (by simp) (by simp) (by simp)).2]
  rw [show ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 0, 0)).1
        = aftgRunningMaxG qT kT vT keyScale qStart hi i d from by
    rw [aftgStateBot_fst, aftgRunningMaxG, aftg_foldl_sup_bot_eq_foldr]]

/-- General ⊥-seeded ratio (max factor cancels) on a nonempty window. -/
theorem aftgStateBotG_ratio_eq (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hne : aftgRunningMaxG qT kT vT keyScale qStart hi i d ≠ ⊥) :
    (aftgStateBotG qT kT vT keyScale qStart hi i d).2.2
        / (aftgStateBotG qT kT vT keyScale qStart hi i d).2.1
      = ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1 * p.2)).sum
        / ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1)).sum := by
  rw [aftgStateBotG_snd_fst, aftgStateBotG_snd_snd, zero_add, zero_add]
  cases hM : aftgRunningMaxG qT kT vT keyScale qStart hi i d with
  | bot => exact absurd hM hne
  | coe r =>
    have hκ : ((r : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-r) := rfl
    rw [hκ]
    have hpos : pow2 (-r) ≠ 0 := ne_of_gt (pow2_pos _)
    rw [mul_div_mul_left _ _ hpos]

/-- General ⊥-seeded running max at the empty window is `⊥`. -/
theorem aftgRunningMaxG_zero (qStart : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    aftgRunningMaxG qT kT vT keyScale qStart 0 i d = ⊥ := by
  unfold aftgRunningMaxG aftgKeysUptoG
  rw [show (List.finRange SEQ).filterMap
        (fun j : Fin SEQ => if j.val < 0 ∧ j.val ≤ qStart + i.val
          then some (aftgKVG qT kT vT keyScale i d j) else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-- General window split (`hi = c·BLOCK_N`). -/
theorem aftgKeysUptoG_succ (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    aftgKeysUptoG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i d
      = aftgKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i d
        ++ aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d := by
  unfold aftgKeysUptoG aftgBlockG
  rw [show (List.finRange SEQ).filterMap
        (fun j : Fin SEQ => if j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val
          then some (aftgKVG qT kT vT keyScale i d j) else none)
      = (List.finRange SEQ).filterMap
        (fun j : Fin SEQ => if j.val ≤ qStart + i.val ∧ j.val < (c + 1) * BLOCK_N
          then some (aftgKVG qT kT vT keyScale i d j) else none)
      from List.filterMap_congr (fun j _ => by simp only [and_comm])]
  rw [aftg_filterMap_window_split (List.finRange SEQ) (List.pairwise_lt_finRange SEQ)
    (c * BLOCK_N) ((c + 1) * BLOCK_N) (fun j => j.val ≤ qStart + i.val)
    (fun j => aftgKVG qT kT vT keyScale i d j) (by nlinarith [Nat.zero_le BLOCK_N])]
  refine congrArg₂ (· ++ ·) ?_ ?_
  · apply List.filterMap_congr; intro j _; simp only [and_comm]
  · apply List.filterMap_congr; intro j _
    by_cases h1 : c * BLOCK_N ≤ j.val <;> by_cases h2 : j.val < (c + 1) * BLOCK_N <;>
      by_cases h3 : j.val ≤ qStart + i.val <;> simp [h1, h2, h3, and_assoc]

/-- General one-block advance of `aftgStateBotG`. -/
theorem aftgStateBotG_succ (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    aftgStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i d
      = (aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d).foldl osStepBot
          (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d) := by
  unfold aftgStateBotG
  rw [aftgKeysUptoG_succ, List.foldl_append]

/-- General running max one-block advance. -/
theorem aftgRunningMaxG_succ (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i d
      = aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i d
        ⊔ ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d).map
            (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  unfold aftgRunningMaxG
  rw [aftgKeysUptoG_succ, List.map_append]
  induction (aftgKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i d) with
  | nil => simp
  | cons a t ih => simp only [List.map_cons, List.foldr_cons, List.cons_append, ih, max_assoc]

/-- General `aftgRunningMaxG` is channel-independent. -/
theorem aftgRunningMaxG_eq (qStart hi : Nat) (i : Fin BLOCK_M) (d d' : Fin BLOCK_DMODEL) :
    aftgRunningMaxG qT kT vT keyScale qStart hi i d
      = aftgRunningMaxG qT kT vT keyScale qStart hi i d' := by
  unfold aftgRunningMaxG aftgKeysUptoG
  congr 1
  rw [List.map_filterMap, List.map_filterMap]
  apply List.filterMap_congr
  intro j _
  by_cases hj : j.val < hi ∧ j.val ≤ qStart + i.val <;> simp [aftgKVG, hj]

/-- General `aftgRunningMaxG` over a nonempty causal window ≠ ⊥ (key 0 always kept). -/
theorem aftgRunningMaxG_ne_bot (qStart hi : Nat) (hhi : 1 ≤ hi) (hSEQ : 0 < SEQ)
    (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    aftgRunningMaxG qT kT vT keyScale qStart hi i d ≠ ⊥ := by
  unfold aftgRunningMaxG aftgKeysUptoG
  set sc0 : ℝ := keyScale (⟨0, hSEQ⟩ : Fin SEQ) *
      Finset.univ.sum (fun e : Fin BLOCK_DMODEL => qT (i, e, PUnit.unit) *
        kT (⟨0, hSEQ⟩, e, PUnit.unit)) with hsc0
  have hmem : ((sc0 : ℝ) : WithBot ℝ) ∈
      ((List.finRange SEQ).filterMap (fun j : Fin SEQ =>
        if j.val < hi ∧ j.val ≤ qStart + i.val
        then some (aftgKVG qT kT vT keyScale i d j) else none)).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ)) := by
    rw [List.mem_map]
    refine ⟨aftgKVG qT kT vT keyScale i d ⟨0, hSEQ⟩, ?_, rfl⟩
    rw [List.mem_filterMap]
    refine ⟨⟨0, hSEQ⟩, List.mem_finRange _, ?_⟩
    rw [if_pos ⟨show (0:Nat) < hi from by omega, Nat.zero_le _⟩]
  have hle := aftg_mem_le_foldr_sup _ _ hmem
  intro hbot
  exact absurd (le_bot_iff.mp (hbot ▸ hle)) WithBot.coe_ne_bot

/-- General seed-1 = seed-0 on nonempty windows. -/
theorem aftgStateBot1G_eq_aftgStateBotG (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hne : aftgRunningMaxG qT kT vT keyScale qStart hi i d ≠ ⊥) :
    aftgStateBot1G qT kT vT keyScale qStart hi i d
      = aftgStateBotG qT kT vT keyScale qStart hi i d := by
  have hxs : aftgKeysUptoG qT kT vT keyScale qStart hi i d ≠ [] := by
    intro h; apply hne; unfold aftgRunningMaxG; rw [h]; rfl
  unfold aftgStateBot1G aftgStateBotG
  exact osStepBot_bot_seed_indep _ hxs 1 0 0 0

/-- General seed-1 state at the empty window is `(⊥, 1, 0)`. -/
theorem aftgStateBot1G_zero (qStart : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    aftgStateBot1G qT kT vT keyScale qStart 0 i d = (⊥, 1, 0) := by
  unfold aftgStateBot1G aftgKeysUptoG
  rw [show (List.finRange SEQ).filterMap
        (fun j : Fin SEQ => if j.val < 0 ∧ j.val ≤ qStart + i.val
          then some (aftgKVG qT kT vT keyScale i d j) else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-- General denominator channel-independence. -/
theorem aftgStateBotG_snd_fst_indep (qStart hi : Nat) (i : Fin BLOCK_M) (d d' : Fin BLOCK_DMODEL) :
    (aftgStateBotG qT kT vT keyScale qStart hi i d).2.1
      = (aftgStateBotG qT kT vT keyScale qStart hi i d').2.1 := by
  rw [aftgStateBotG_snd_fst, aftgStateBotG_snd_fst,
    aftgRunningMaxG_eq qT kT vT keyScale qStart hi i d d']
  congr 2
  unfold aftgKeysUptoG
  rw [List.map_filterMap, List.map_filterMap]
  refine congrArg List.sum (List.filterMap_congr ?_)
  intro j _
  by_cases hj : j.val < hi ∧ j.val ≤ qStart + i.val <;> simp [aftgKVG, hj]

/-- General full-window bridge: at `hi = SEQ`, the causal ⊥-seeded key list equals
the predicate-filtered `attnKeyListPred` with `causalKeep qStart`. -/
theorem aftgKeysUptoG_full_eq_pred (qStart : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    aftgKeysUptoG qT kT vT keyScale qStart SEQ i d
      = attnKeyListPred qT kT vT keyScale (fun a b => causalKeep qStart a b) i d := by
  unfold aftgKeysUptoG attnKeyListPred aftgKVG
  apply List.filterMap_congr
  intro j _
  have hjlt : j.val < SEQ := j.isLt
  have hiff : causalKeep qStart i j ↔ j.val ≤ qStart + i.val := by
    unfold causalKeep; omega
  by_cases hc : causalKeep qStart i j
  · rw [if_pos ⟨hjlt, hiff.mp hc⟩, if_pos hc]
  · rw [if_neg (fun hh => hc (hiff.mpr hh.2)), if_neg hc]

/-- General sentinel boundedness side-condition (**causal-only, non-strict**).

The kernel masks every non-causal key with the `-1e6` `tl.where` sentinel, so only
the **kept** (causal) keys `j ≤ qStart + i` need a boundedness assumption, and only
`≥ -1e6` (non-strict) is required: the masked keys are pinned to the sentinel and
never raise the running max. This mirrors the pinned `aftScoreBound` form. -/
def aftgScoreBoundG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart : Nat) : Prop :=
  ∀ (j : Fin SEQ) (i : Fin BLOCK_M), j.val ≤ qStart + i.val →
    (0:ℝ) - 1000000.0
      ≤ keyScale j * Finset.univ.sum (fun e : Fin BLOCK_DMODEL => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit))

/-- General: under the **causal-only, non-strict** `aftgScoreBoundG`, the running
max over a nonempty causal window is `≥` the `-1e6` sentinel. Key `0` is always
causal-kept (`0 ≤ qStart + i`), so its kept score `≥ -1e6` (the weaker bound) is a
member of the window's `foldr ⊔ ⊥`, hence a lower bound on the running max. -/
theorem aftgRunningMaxG_gt_sentinel (qStart hi : Nat) (hhi : 1 ≤ hi) (hSEQ : 0 < SEQ)
    (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hsb : aftgScoreBoundG qT kT vT keyScale qStart) :
    aftgRunningMaxG qT kT vT keyScale qStart hi i d ≥ some (-1000000.0 : ℝ) := by
  unfold aftgRunningMaxG aftgKeysUptoG
  have hkey0 : ((-1000000.0 : ℝ)) ≤ ((aftgKVG qT kT vT keyScale i d ⟨0, hSEQ⟩).1 : ℝ) := by
    have := hsb ⟨0, hSEQ⟩ i (by simp)
    simpa [aftgKVG] using this
  have hmem : ((((aftgKVG qT kT vT keyScale i d ⟨0, hSEQ⟩).1 : ℝ)) : WithBot ℝ) ∈
      ((List.finRange SEQ).filterMap (fun j : Fin SEQ =>
        if j.val < hi ∧ j.val ≤ qStart + i.val
        then some (aftgKVG qT kT vT keyScale i d j) else none)).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ)) := by
    rw [List.mem_map]
    refine ⟨aftgKVG qT kT vT keyScale i d ⟨0, hSEQ⟩, ?_, rfl⟩
    rw [List.mem_filterMap]
    refine ⟨⟨0, hSEQ⟩, List.mem_finRange _, ?_⟩
    rw [if_pos ⟨show (0:Nat) < hi from by omega, Nat.zero_le _⟩]
  have hle := aftg_mem_le_foldr_sup _ _ hmem
  refine le_trans ?_ hle
  exact (WithBot.coe_le_coe).mpr hkey0

/-- General `aftgBlockG` map-and-sum: reindex block `c`'s causal window onto
`Fin BLOCK_N` lanes (global key `c·BLOCK_N + jL`). -/
theorem aftgBlockG_map_sum (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hwin : (c + 1) * BLOCK_N ≤ SEQ) (h : ℝ × ℝ → ℝ) :
    ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d).map h).sum
      = ∑ jL : Fin BLOCK_N,
          (if (c * BLOCK_N + jL.val) ≤ qStart + i.val then
            h (aftgKVG qT kT vT keyScale i d ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩)
           else 0) := by
  rw [aftgBlockG, aftg_filterMap_finRange_sum SEQ
    (fun j => c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val)
    (fun j => aftgKVG qT kT vT keyScale i d j) h]
  rw [show (∑ j : Fin SEQ, if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val
            then h (aftgKVG qT kT vT keyScale i d j) else 0)
        = ∑ j ∈ Finset.univ.filter (fun j : Fin SEQ => c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N),
            (if j.val ≤ qStart + i.val then h (aftgKVG qT kT vT keyScale i d j) else 0) from by
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

/-- General `aftgBlockG` running-sup bridge. -/
theorem aftgBlockG_blockSup (qStart BLOCK_N c : Nat) (i d : Fin BLOCK_M) (d2 : Fin BLOCK_DMODEL)
    (hwin : (c + 1) * BLOCK_N ≤ SEQ) :
    ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d2).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = Finset.univ.sup (fun jL : Fin BLOCK_N =>
          if (c * BLOCK_N + jL.val) ≤ qStart + i.val then
            (((aftgKVG qT kT vT keyScale i d2 ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else (⊥ : WithBot ℝ)) := by
  rw [show (aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d2).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
      = ((List.finRange SEQ).filterMap (fun j : Fin SEQ =>
          if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val
          then some ((aftgKVG qT kT vT keyScale i d2 j).1) else none)).map
            (fun x => ((x : ℝ) : WithBot ℝ)) from by
    unfold aftgBlockG
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr
    intro j _
    by_cases hj : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val <;> simp [hj]]
  rw [aftg_filterMap_foldr_sup SEQ
    (fun j => c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val)
    (fun j => (aftgKVG qT kT vT keyScale i d2 j).1)]
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
abbrev aftgAx1G (BLOCK_M BLOCK_N : Nat) : Fin [BLOCK_M, BLOCK_N].length := ⟨1, by simp⟩

/-- General `reduceMax` row. -/
theorem aftg_reduceMaxDrop_rowG (BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (qk : Tile .real [BLOCK_M, BLOCK_N]) (rmaxT : Tile .real [BLOCK_M])
    (hrm : Tile.reduceMaxDrop (aftgAx1G BLOCK_M BLOCK_N) qk = some rmaxT)
    (i : Fin BLOCK_M) (g : Fin BLOCK_N → WithBot ℝ)
    (hqk : ∀ jL : Fin BLOCK_N,
        qk.data (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (aftgAx1G BLOCK_M BLOCK_N) (i, PUnit.unit) jL) = g jL) :
    rmaxT.data (i, PUnit.unit) = Finset.univ.sup g := by
  unfold Tile.reduceMaxDrop at hrm
  rw [dif_pos (show 0 < TileShape.axisDim [BLOCK_M, BLOCK_N] (aftgAx1G BLOCK_M BLOCK_N) from hBN)] at hrm
  rw [← Option.some.inj hrm]
  simp only [Finset.sup'_eq_sup]
  exact Finset.sup_congr rfl (fun jL _ => hqk jL)

/-- General `q·k` score cell (HEAD_ACTIVE). -/
theorem aftg_score_cellG (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat)
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

/-! ### General per-statement op-eval recipes -/

set_option maxHeartbeats 1600000 in
/-- General `k_mask` recipe ([BLOCK_DMODEL, BLOCK_N]). -/
theorem aftg_kmask_evalG (s : BlockState) (SN NCTX HA BLOCK_N BLOCK_DMODEL : Nat)
    (offsn : Tile .nat [BLOCK_N])
    (hoffsn : s.regs .nat [BLOCK_N] "offs_n" = some offsn)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOp (Op.boolAnd (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat NCTX) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HA)))) s
      = some ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
          (ComparableDType.nat.lt (offsn.data (idx.2.1, PUnit.unit)) (NCTX - SN))
            && (ComparableDType.nat.lt idx.1.val HA)⟩ := by
  rw [aftg_evalOp_boolAnd]
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
/-- General `mask` (causal keep) recipe. -/
theorem aftg_mask_evalG (s : BlockState) (SN BLOCK_M BLOCK_N : Nat)
    (offsm : Tile .nat [BLOCK_M]) (offsn : Tile .nat [BLOCK_N])
    (hoffsm : s.regs .nat [BLOCK_M] "offs_m" = some offsm)
    (hoffsn : s.regs .nat [BLOCK_N] "offs_n" = some offsn)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOp (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")))) s
      = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          ComparableDType.nat.ge (offsm.data (idx.1, PUnit.unit))
            (SN + offsn.data (idx.2.1, PUnit.unit))⟩ := by
  rw [aftg_evalOp_ge]
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
/-- General `m_ij = maximum(m_i, max(qk,1))` recipe. -/
theorem aftg_mij_evalG (s : BlockState) (BLOCK_M BLOCK_N : Nat)
    (mtile : Tile .real [BLOCK_M]) (qktile : Tile .real [BLOCK_M, BLOCK_N]) (rmaxT : Tile .real [BLOCK_M])
    (hmi : s.regs .real [BLOCK_M] "m_i" = some mtile)
    (hqk : s.regs .real [BLOCK_M, BLOCK_N] "qk" = some qktile)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) qktile = some rmaxT) :
    evalOp (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BLOCK_M] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false
            (Op.ref .real [BLOCK_M, BLOCK_N] "qk")))
        (Op.ref .real [BLOCK_M] "m_i")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false
          (Op.ref .real [BLOCK_M, BLOCK_N] "qk"))) s
      = some (Tile.select
          (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT)
          mtile rmaxT) := by
  have hrmax : @evalOp TileDType.real [BLOCK_M]
      (Op.reduceMax (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false
        (Op.ref .real [BLOCK_M, BLOCK_N] "qk")) s = some rmaxT := by
    unfold evalOp
    simp only [evalOp_ref, hqk, Option.bind_eq_bind, Option.bind_some]
    exact hrm
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hmi, hrmax, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
/-- General `qk = qk - m_ij[:,None]` recipe. -/
theorem aftg_qk_sub_evalG (s : BlockState) (BLOCK_M BLOCK_N : Nat) (hax : 1 < [BLOCK_M].length.succ)
    (qktile : Tile .real [BLOCK_M, BLOCK_N]) (mc : Tile .real [BLOCK_M])
    (hqk : s.regs .real [BLOCK_M, BLOCK_N] "qk" = some qktile)
    (hmij : s.regs .real [BLOCK_M] "m_ij" = some mc) :
    evalOp (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_N] "qk") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "m_ij"))) s
      = some (Tile.bop NumericDType.real.sub
          (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          qktile (Tile.expandDim ⟨1, hax⟩ mc)) := by
  have hexp : @evalOp TileDType.real [BLOCK_M, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "m_ij")) s
      = some (Tile.expandDim ⟨1, hax⟩ mc) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hmij
  rw [evalOp_sub]
  simp only [evalOp_ref, hqk, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
/-- General `p = exp2(qk)` recipe. -/
theorem aftg_p_evalG (s : BlockState) (BLOCK_M BLOCK_N : Nat) (qktile : Tile .real [BLOCK_M, BLOCK_N])
    (hqk : s.regs .real [BLOCK_M, BLOCK_N] "qk" = some qktile) :
    evalOp (Op.exp2 (Op.ref .real [BLOCK_M, BLOCK_N] "qk")) s
      = some (Tile.uop WithBot.realExp2 qktile) := by
  rw [aftg_evalOp_exp2]; simp only [evalOp_ref, hqk, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
/-- General `p = where(mask, p, 0)` recipe. -/
theorem aftg_p_mask_evalG (s : BlockState) (BLOCK_M BLOCK_N : Nat) (masktile : Tile .bool [BLOCK_M, BLOCK_N])
    (ptile : Tile .real [BLOCK_M, BLOCK_N])
    (hmask : s.regs .bool [BLOCK_M, BLOCK_N] "mask" = some masktile)
    (hp : s.regs .real [BLOCK_M, BLOCK_N] "p" = some ptile) :
    evalOp (Op.where (Op.ref .bool [BLOCK_M, BLOCK_N] "mask")
        (Op.ref .real [BLOCK_M, BLOCK_N] "p") (Op.broadcast (Op.const 0.0) [BLOCK_M, BLOCK_N])) s
      = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          if masktile.data idx then ptile.data idx else (some (0.0 : ℝ) : WithBot ℝ)⟩ := by
  have hbcast : @evalOp TileDType.real [BLOCK_M, BLOCK_N] (Op.broadcast (Op.const 0.0) [BLOCK_M, BLOCK_N]) s
      = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => (some (0.0 : ℝ) : WithBot ℝ)⟩ :
          Tile .real [BLOCK_M, BLOCK_N]) := by
    simp only [evalOp, evalOp_const, Option.bind_eq_bind, Option.bind_some]; rfl
  rw [evalOp_where]
  simp only [evalOp_ref, hmask, hp, hbcast, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.select_data, Tile.scalar]

set_option maxHeartbeats 1600000 in
/-- General `l_ij = sum(p, 1)` recipe. -/
theorem aftg_lij_evalG (s : BlockState) (BLOCK_M BLOCK_N : Nat) (ptile : Tile .real [BLOCK_M, BLOCK_N])
    (hp : s.regs .real [BLOCK_M, BLOCK_N] "p" = some ptile) :
    evalOp (Op.reduceSum (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false
        (Op.ref .real [BLOCK_M, BLOCK_N] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) ptile) := by
  rw [evalOp_reduceSum]
  simp only [evalOp_ref, hp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
/-- General `alpha = exp2(m_i - m_ij)` recipe. -/
theorem aftg_alpha_evalG (s : BlockState) (BLOCK_M : Nat) (mi mij : Tile .real [BLOCK_M])
    (hmi : s.regs .real [BLOCK_M] "m_i" = some mi)
    (hmij : s.regs .real [BLOCK_M] "m_ij" = some mij) :
    evalOp (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BLOCK_M] "m_i") (Op.ref .real [BLOCK_M] "m_ij"))) s
      = some (Tile.uop WithBot.realExp2
          (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mi mij)) := by
  rw [aftg_evalOp_exp2, evalOp_sub]
  simp only [evalOp_ref, hmi, hmij, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
/-- General `l_i = l_i * alpha + l_ij` recipe. -/
theorem aftg_li_evalG (s : BlockState) (BLOCK_M : Nat) (li alpha lij : Tile .real [BLOCK_M])
    (hli : s.regs .real [BLOCK_M] "l_i" = some li)
    (halpha : s.regs .real [BLOCK_M] "alpha" = some alpha)
    (hlij : s.regs .real [BLOCK_M] "l_ij" = some lij) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BLOCK_M] "l_i") (Op.ref .real [BLOCK_M] "alpha"))
        (Op.ref .real [BLOCK_M] "l_ij")) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
          (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) li alpha) lij) := by
  rw [evalOp_add, evalOp_mul]
  simp only [evalOp_ref, hli, halpha, hlij, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
/-- General `acc = acc * alpha[:,None]` recipe. -/
theorem aftg_acc_rescale_evalG (s : BlockState) (BLOCK_M BLOCK_DMODEL : Nat) (hax : 1 < [BLOCK_M].length.succ)
    (acctile : Tile .real [BLOCK_M, BLOCK_DMODEL]) (alpha : Tile .real [BLOCK_M])
    (hacc : s.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some acctile)
    (halpha : s.regs .real [BLOCK_M] "alpha" = some alpha) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "alpha"))) s
      = some (Tile.bop NumericDType.real.mul
          (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          acctile (Tile.expandDim ⟨1, hax⟩ alpha)) := by
  have hexp : @evalOp TileDType.real [BLOCK_M, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "alpha")) s
      = some (Tile.expandDim ⟨1, hax⟩ alpha) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ halpha
  rw [evalOp_mul]
  simp only [evalOp_ref, hacc, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
/-- General `p = p.to(fp16)` recipe. -/
theorem aftg_p_fp16_evalG (s : BlockState) (BLOCK_M BLOCK_N : Nat) (ptile : Tile .real [BLOCK_M, BLOCK_N])
    (hp : s.regs .real [BLOCK_M, BLOCK_N] "p" = some ptile) :
    evalOp (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BLOCK_M, BLOCK_N] "p")) s
      = some ⟨fun i => FloatDType.real.cast FloatDType.fp16 (ptile.data i)⟩ := by
  simp only [evalOp_castFloat, FloatDType.toTileDType_real, evalOp_ref, hp,
    Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
/-- General `m_i = m_ij` recipe. -/
theorem aftg_mi_carry_evalG (s : BlockState) (BLOCK_M : Nat) (mij : Tile .real [BLOCK_M])
    (hmij : s.regs .real [BLOCK_M] "m_ij" = some mij) :
    evalOp (Op.ref .real [BLOCK_M] "m_ij") s = some mij := by
  rw [evalOp_ref, hmij]

set_option maxHeartbeats 1600000 in
/-- General pointer-advance recipe (second factor symbolic = HEAD_DIM). -/
theorem aftg_advance_ptr_evalG (s : BlockState) (BT BS d e : Nat) (name : RegName)
    (ptrs : Tile .ptr [BT, BS])
    (hptr : s.regs .ptr [BT, BS] name = some ptrs) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BT, BS] name)
        (Op.mul .nat Broadcast.nil (Op.constNat d) (Op.constNat e))) s
      = some (Tile.ptrAdd Broadcast.scalarR ptrs
          (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar d) (Tile.scalar e))) := by
  simp only [evalOp, evalOp_ref, evalOp_constNat, hptr, Option.bind_eq_bind, Option.bind_some]

/-- General `reduceMaxDrop` over axis 1 of `[BLOCK_M, BLOCK_N]` succeeds (`0 < BLOCK_N`). -/
theorem aftg_reduceMaxDrop1_someG (BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N) (x : Tile .real [BLOCK_M, BLOCK_N]) :
    ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) x = some t := by
  unfold Tile.reduceMaxDrop
  rw [dif_pos (show 0 < TileShape.axisDim [BLOCK_M, BLOCK_N] (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) from hBN)]
  exact ⟨_, rfl⟩

set_option maxHeartbeats 1600000 in
/-- General `m_ij = aftgRunningMaxG((c+1)·BLOCK_N)` (masked with -1e6 sentinel). -/
theorem aftg_mij_reg_eq_maskedG (BLOCK_N : Nat) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M)
    (hsb : aftgScoreBoundG qT kT vT keyScale qStart)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mtile rmaxT mijT : Tile .real [BLOCK_M])
    (hsent : ∀ jL : Fin BLOCK_N, qkSentT.data (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (aftgAx1G BLOCK_M BLOCK_N) (i, PUnit.unit) jL)
        = if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hrmax : Tile.reduceMaxDrop (aftgAx1G BLOCK_M BLOCK_N) qkSentT = some rmaxT)
    (hmtile : mtile.data (i, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩)
    (hmij : mijT = Tile.select
        (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT) :
    mijT.data (i, PUnit.unit)
      = aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ := by
  have hSEQ : 0 < SEQ := by
    have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega
  have hrmaxcell : rmaxT.data (i, PUnit.unit)
      = Finset.univ.sup (fun jL : Fin BLOCK_N =>
          if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ)) :=
    aftg_reduceMaxDrop_rowG BLOCK_M BLOCK_N hBN qkSentT rmaxT hrmax i _ hsent
  rw [aftgRunningMaxG_succ qT kT vT keyScale qStart BLOCK_N c i ⟨0, hBD⟩]
  rw [aftgBlockG_blockSup qT kT vT keyScale qStart BLOCK_N c i i ⟨0, hBD⟩ hc1]
  set BSk : WithBot ℝ := Finset.univ.sup (fun jL : Fin BLOCK_N =>
      if (c * BLOCK_N + jL.val) ≤ qStart + i.val then
        (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
      else (⊥ : WithBot ℝ)) with hBSk
  set M := aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩ with hMdef
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
      rw [WithBot.coe_le_coe]
      have hbnd := hsb ⟨0 * BLOCK_N + (⟨0, hBN⟩ : Fin BLOCK_N).val, by simp only [Nat.zero_mul, Nat.add_zero]; exact hSEQ⟩ i (by simp)
      simpa [aftgKVG] using hbnd
    · refine le_sup_of_le_left ?_
      rw [hMdef]
      exact aftgRunningMaxG_gt_sentinel qT kT vT keyScale qStart (c * BLOCK_N) (by
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
theorem aftg_pmT_cell_maskedG (BLOCK_N : Nat) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M) (jL : Fin BLOCK_N) (Mc1 : WithBot ℝ)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mijT : Tile .real [BLOCK_M]) (pT : Tile .real [BLOCK_M, BLOCK_N])
    (kept : Bool)
    (_hkept : kept = decide (qStart + i.val ≥ c * BLOCK_N + jL.val))
    (hsent : qkSentT.data (i, jL, PUnit.unit)
        = if kept then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩
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
          pow2 ((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩
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
          (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩
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
theorem aftg_nume_row_sum_maskedG (BLOCK_N : Nat) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hsb : aftgScoreBoundG qT kT vT keyScale qStart)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mijT : Tile .real [BLOCK_M]) (pT : Tile .real [BLOCK_M, BLOCK_N])
    (hsent : ∀ jL : Fin BLOCK_N, qkSentT.data (i, jL, PUnit.unit)
        = if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hmij : mijT.data (i, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩)
    (hpT : ∀ jL : Fin BLOCK_N, pT.data (i, jL, PUnit.unit)
        = if (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ)) :
    (Tile.reduceSumDrop (aftgAx1G BLOCK_M BLOCK_N) pT).data (i, PUnit.unit)
      = some ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d).map
          (fun p => pow2 (p.1 - (aftgRunningMaxG qT kT vT keyScale
            qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩).unbotD 0))).sum := by
  have hc1' : 1 ≤ (c + 1) * BLOCK_N := by
    have : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega
  set Mc1 := aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ with hMc1
  have hMc1bot : Mc1 ≠ ⊥ := by
    rw [hMc1]; exact aftgRunningMaxG_ne_bot qT kT vT keyScale qStart ((c + 1) * BLOCK_N) hc1' (by omega) i ⟨0, hBD⟩
  rw [Tile.reduceSumDrop_data]
  have hcell : ∀ jL : Fin BLOCK_N,
      pT.data (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (aftgAx1G BLOCK_M BLOCK_N) (i, PUnit.unit) jL)
        = some (if decide (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            pow2 ((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 - Mc1.unbotD 0)
            else 0) := by
    intro jL
    rw [show (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (aftgAx1G BLOCK_M BLOCK_N) (i, PUnit.unit) jL) = (i, jL, PUnit.unit) from rfl]
    have hsentR : qkSentT.data (i, jL, PUnit.unit)
        = if decide (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
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
    exact aftg_pmT_cell_maskedG qT kT vT keyScale BLOCK_N hBD qStart c hc1 i jL Mc1 qkSentT mijT pT
      (decide (qStart + i.val ≥ c * BLOCK_N + jL.val)) (by rfl) hsentR
      (by rw [hmij]) (fun _ => hMc1bot) hpTR
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [aftgBlockG_map_sum qT kT vT keyScale qStart BLOCK_N c i d hc1
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
theorem aftg_acc_dot_block_maskedG (BLOCK_N : Nat) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mijT : Tile .real [BLOCK_M]) (pT : Tile .real [BLOCK_M, BLOCK_N])
    (vtile : Tile .real [BLOCK_N, BLOCK_DMODEL]) (vval : Fin BLOCK_N → ℝ)
    (hMc1bot : aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ ≠ ⊥)
    (hsent : ∀ jL : Fin BLOCK_N, qkSentT.data (i, jL, PUnit.unit)
        = if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hmij : mijT.data (i, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩)
    (hpT : ∀ jL : Fin BLOCK_N, pT.data (i, jL, PUnit.unit)
        = if (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ))
    (hv : ∀ jL : Fin BLOCK_N, vtile.data (jL, d, PUnit.unit) = some (vval jL))
    (hvval : ∀ jL : Fin BLOCK_N, vval jL
        = (aftgKVG qT kT vT keyScale i d ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).2) :
    (Tile.dot [] pT vtile).data (i, d, PUnit.unit)
      = some ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d).map
          (fun p => pow2 (p.1 - (aftgRunningMaxG qT kT vT keyScale
            qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩).unbotD 0) * p.2)).sum := by
  set Mc1 := aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ with hMc1
  have hpcell : ∀ jL : Fin BLOCK_N, pT.data (i, jL, PUnit.unit)
      = some (if decide (qStart + i.val ≥ c * BLOCK_N + jL.val) then
          pow2 ((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 - Mc1.unbotD 0)
          else 0) := by
    intro jL
    have hsentR : qkSentT.data (i, jL, PUnit.unit)
        = if decide (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
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
    have := aftg_pmT_cell_maskedG qT kT vT keyScale BLOCK_N hBD qStart c hc1 i jL Mc1 qkSentT mijT pT
      (decide (qStart + i.val ≥ c * BLOCK_N + jL.val)) (by rfl) hsentR (by rw [hmij]) (fun _ => hMc1bot) hpTR
    rw [this]
  rw [Tile.dot_nil_data]
  have hterm : ∀ jL : Fin BLOCK_N,
      Option.map₂ (· * ·) (pT.data (i, jL, PUnit.unit)) (vtile.data (jL, d, PUnit.unit))
        = some (if (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            pow2 ((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 - Mc1.unbotD 0)
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
              pow2 ((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 - Mc1.unbotD 0)
                * vval jL else 0) : WithBot ℝ))
      from Finset.sum_congr rfl (fun jL _ => hterm jL)]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [aftgBlockG_map_sum qT kT vT keyScale qStart BLOCK_N c i d hc1
      (fun p => pow2 (p.1 - Mc1.unbotD 0) * p.2)]
  refine Finset.sum_congr rfl (fun jL _ => ?_)
  by_cases hkp : (c * BLOCK_N + jL.val) ≤ qStart + i.val
  · rw [if_pos (show qStart + i.val ≥ c * BLOCK_N + jL.val from by omega),
        if_pos (show c * BLOCK_N + jL.val ≤ qStart + i.val from hkp)]
    rw [hvval jL,
      show (aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1
          = (aftgKVG qT kT vT keyScale i d ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 from by
        simp only [aftgKVG]]
  · rw [if_neg (show ¬ qStart + i.val ≥ c * BLOCK_N + jL.val from by omega),
        if_neg (show ¬ c * BLOCK_N + jL.val ≤ qStart + i.val from hkp)]

/-- General seed-1 vs seed-0 carry cancellation. -/
theorem aftgStateBot1G_cancel (qStart BLOCK_N c : Nat) (hBN : 0 < BLOCK_N) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) (Mc1 : WithBot ℝ)
    (hSEQ : 0 < SEQ) :
    let m := (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).1
    let α := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
    (aftgStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i d).2.1 * α
        = (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.1 * α
      ∧ (aftgStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2 * α
        = (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2 * α := by
  intro m α
  by_cases hc0 : c = 0
  · subst hc0
    have hmbot : m = ⊥ := by
      show (aftgStateBotG qT kT vT keyScale qStart (0 * BLOCK_N) i d).1 = ⊥
      rw [aftgStateBotG_fst_eq_runningMax, Nat.zero_mul, aftgRunningMaxG_zero]
    have hα0 : α = 0 := by
      show (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 = 0
      rw [hmbot, WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
    rw [hα0]; simp
  · have hne : aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i d ≠ ⊥ :=
      aftgRunningMaxG_ne_bot qT kT vT keyScale qStart (c * BLOCK_N)
        (by calc 1 ≤ c := by omega
              _ ≤ c * BLOCK_N := Nat.le_mul_of_pos_right c hBN) hSEQ i d
    rw [aftgStateBot1G_eq_aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d hne]
    exact ⟨rfl, rfl⟩

/-- General empty-window sum-zero. -/
theorem aftgKeysUptoG_sum_zero_of_bot (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hbot : aftgRunningMaxG qT kT vT keyScale qStart hi i d = ⊥) (h : ℝ × ℝ → ℝ) :
    ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).map h).sum = 0 := by
  rw [show aftgKeysUptoG qT kT vT keyScale qStart hi i d = [] from ?_, List.map_nil, List.sum_nil]
  by_contra hne
  obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ hne
  have hmem : ((p.1 : ℝ) : WithBot ℝ) ∈
      (aftgKeysUptoG qT kT vT keyScale qStart hi i d).map (fun q => ((q.1 : ℝ) : WithBot ℝ)) :=
    List.mem_map_of_mem hp
  have := aftg_mem_le_foldr_sup _ _ hmem
  rw [← aftgRunningMaxG, hbot] at this
  exact absurd (le_bot_iff.mp this) WithBot.coe_ne_bot

/-- General denom anchor. -/
theorem aftg_denom_anchorG (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    (aftgStateBotG qT kT vT keyScale qStart hi i d).2.1
      = ((aftgStateBotG qT kT vT keyScale qStart hi i d).1.elim 0 (fun r => pow2 (-r)))
        * (0 + ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1)).sum) := by
  rw [aftgStateBotG_snd_fst, aftgStateBotG_fst_eq_runningMax]

/-- General acc anchor. -/
theorem aftg_acc_anchorG (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    (aftgStateBotG qT kT vT keyScale qStart hi i d).2.2
      = ((aftgStateBotG qT kT vT keyScale qStart hi i d).1.elim 0 (fun r => pow2 (-r)))
        * (0 + ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1 * p.2)).sum) := by
  rw [aftgStateBotG_snd_snd, aftgStateBotG_fst_eq_runningMax]

set_option maxHeartbeats 1600000 in
/-- General `l_i' = aftgStateBotG((c+1)·BLOCK_N).2.1` (masked). -/
theorem aftg_denom_reg_eq_maskedG (BLOCK_N : Nat) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M)
    (hsb : aftgScoreBoundG qT kT vT keyScale qStart)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mtile mijT alphaT litile lijT : Tile .real [BLOCK_M])
    (pT : Tile .real [BLOCK_M, BLOCK_N])
    (hsent : ∀ jL : Fin BLOCK_N, qkSentT.data (i, jL, PUnit.unit)
        = if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hlitile : litile.data (i, PUnit.unit) = some
        ((aftgStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.1))
    (hmtile : mtile.data (i, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩)
    (hmij : mijT.data (i, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩)
    (halpha : alphaT = Tile.uop WithBot.realExp2
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
    (hlijT : lijT = Tile.reduceSumDrop (aftgAx1G BLOCK_M BLOCK_N) pT)
    (hpT : ∀ jL : Fin BLOCK_N, pT.data (i, jL, PUnit.unit)
        = if (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ)) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
        (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) litile alphaT) lijT).data (i, PUnit.unit)
      = some ((aftgStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩).2.1) := by
  have hSEQ : 0 < SEQ := by
    have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega
  set m := (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).1 with hm_def
  set Mc := aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩ with hMc
  set Mc1 := aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ with hMc1
  have hmMc : m = Mc := by rw [hm_def, hMc, aftgStateBotG_fst_eq_runningMax]
  have hMsucc : Mc1 = m ⊔ ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i ⟨0, hBD⟩).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    rw [hMc1, aftgRunningMaxG_succ, hmMc, ← hMc]
  have halphaVal : alphaT.data (i, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [halpha]; show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmtile, hmij,
      NumericDType.sub, ← hMc, ← hMc1, hmMc]
  have hsum := aftg_nume_row_sum_maskedG qT kT vT keyScale BLOCK_N hBN hBM hBD qStart c hc1 i ⟨0, hBD⟩ hsb qkSentT mijT pT hsent hmij hpT
  have hblockEq := osStepBot_block_eq m
    ((aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.1)
    ((aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.2)
    ((aftgKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).map (fun p => pow2 p.1 * p.2)).sum
    ((aftgKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).map (fun p => pow2 p.1)).sum
    (aftgBlockG qT kT vT keyScale qStart BLOCK_N c i ⟨0, hBD⟩)
    (by rw [aftg_denom_anchorG, zero_add, hm_def])
    (by rw [aftg_acc_anchorG, zero_add, hm_def])
    (fun hbot => aftgKeysUptoG_sum_zero_of_bot qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩
      (by rw [← aftgStateBotG_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => aftgKeysUptoG_sum_zero_of_bot qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩
      (by rw [← aftgStateBotG_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  rw [show (aftgStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩).2.1
        = (Mc1, (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.1
              * (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
              + ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i ⟨0, hBD⟩).map (fun p => pow2 (p.1 - Mc1.unbotD 0))).sum,
            _).2.1 from by
    rw [aftgStateBotG_succ]; rw [← hblockEq]]
  set α : ℝ := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp2 (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  have hcancel := (aftgStateBot1G_cancel qT kT vT keyScale qStart BLOCK_N c hBN i ⟨0, hBD⟩ Mc1 hSEQ).1
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex]
  rw [hlijT]
  erw [hsum]
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    hlitile, halphaVal, hαsome]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [show (aftgStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.1 * α
        = (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.1 * α from by
    have := hcancel; simp only [← hm_def, ← hαdef] at this ⊢; exact this]

set_option maxHeartbeats 1600000 in
/-- General `acc' = aftgStateBotG((c+1)·BLOCK_N).2.2` (masked). -/
theorem aftg_acc_reg_eq_maskedG (BLOCK_N : Nat) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hsb : aftgScoreBoundG qT kT vT keyScale qStart)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mtile mijT alphaT : Tile .real [BLOCK_M])
    (acctile acc1T : Tile .real [BLOCK_M, BLOCK_DMODEL]) (pT : Tile .real [BLOCK_M, BLOCK_N])
    (vtile : Tile .real [BLOCK_N, BLOCK_DMODEL]) (vval : Fin BLOCK_N → ℝ)
    (hsent : ∀ jL : Fin BLOCK_N, qkSentT.data (i, jL, PUnit.unit)
        = if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hacctile : acctile.data (i, d, PUnit.unit) = some
        ((aftgStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2))
    (hmtile : mtile.data (i, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩)
    (hmij : mijT.data (i, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩)
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
        = (aftgKVG qT kT vT keyScale i d ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).2) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        acc1T (Tile.dot [] pT vtile)).data (i, d, PUnit.unit)
      = some ((aftgStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i d).2.2) := by
  have hSEQ : 0 < SEQ := by
    have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega
  have hc1' : 1 ≤ (c + 1) * BLOCK_N := by
    have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega
  set m := (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).1 with hm_def
  set Mc := aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩ with hMc
  set Mc1 := aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ with hMc1
  have hmMc : m = Mc := by
    rw [hm_def, hMc, aftgStateBotG_fst_eq_runningMax,
      aftgRunningMaxG_eq qT kT vT keyScale qStart (c * BLOCK_N) i d ⟨0, hBD⟩]
  have hmd : m = aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i d := by
    rw [hm_def, aftgStateBotG_fst_eq_runningMax]
  have hMsucc : Mc1 = m ⊔ ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    rw [hMc1, aftgRunningMaxG_eq qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ d,
      aftgRunningMaxG_succ, hmd]
  have halphaVal : alphaT.data (i, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [halpha]; show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmtile, hmij,
      NumericDType.sub, ← hMc, ← hMc1, hmMc]
  have hMc1bot : Mc1 ≠ ⊥ := by
    rw [hMc1]; exact aftgRunningMaxG_ne_bot qT kT vT keyScale qStart ((c + 1) * BLOCK_N) hc1' hSEQ i ⟨0, hBD⟩
  have hdot := aftg_acc_dot_block_maskedG qT kT vT keyScale BLOCK_N hBM hBD qStart c hc1 i d qkSentT mijT pT vtile vval
    (by rw [← hMc1]; exact hMc1bot) hsent hmij hpT hv hvval
  have hblockEq := osStepBot_block_eq m
    ((aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.1)
    ((aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2)
    ((aftgKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i d).map (fun p => pow2 p.1 * p.2)).sum
    ((aftgKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i d).map (fun p => pow2 p.1)).sum
    (aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d)
    (by rw [aftg_denom_anchorG, zero_add, hm_def])
    (by rw [aftg_acc_anchorG, zero_add, hm_def])
    (fun hbot => aftgKeysUptoG_sum_zero_of_bot qT kT vT keyScale qStart (c * BLOCK_N) i d
      (by rw [← aftgStateBotG_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => aftgKeysUptoG_sum_zero_of_bot qT kT vT keyScale qStart (c * BLOCK_N) i d
      (by rw [← aftgStateBotG_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  rw [show (aftgStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i d).2.2
        = (Mc1, _,
            (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2
              * (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
              + ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d).map (fun p => pow2 (p.1 - Mc1.unbotD 0) * p.2)).sum).2.2
        from by rw [aftgStateBotG_succ]; rw [← hblockEq]]
  set α : ℝ := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp2 (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  have hcancel := (aftgStateBot1G_cancel qT kT vT keyScale qStart BLOCK_N c hBN i d Mc1 hSEQ).2
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex]
  erw [hdot]
  rw [hacc1, Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, hacctile, halphaVal, hαsome]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [show (aftgStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2 * α
        = (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2 * α from by
    have := hcancel; simp only [← hm_def, ← hαdef] at this ⊢; exact this]

/-- **General ⊥-seeded full-window state reads off the genuine closed-form spec.** -/
theorem aftgStateBotG_full_eq_spec
    (s : BlockState) (Q K V : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (keyScale : Fin (BLOCK_N * numKVBlocks) → ℝ)
    (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hne : aftgRunningMaxG
        (qTileAFT2mG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
        (kTileAFT2G s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
        (vTileAFT2mG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
        keyScale (qStartAFT2G s BLOCK_M) (BLOCK_N * numKVBlocks) i d ≠ ⊥) :
    (let st := aftgStateBotG
        (qTileAFT2mG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
        (kTileAFT2G s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
        (vTileAFT2mG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
        keyScale (qStartAFT2G s BLOCK_M) (BLOCK_N * numKVBlocks) i d
     st.2.2 / st.2.1)
      = attnFwdTritonOutSpecG s Q K V stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N
          BLOCK_DMODEL HEAD_ACTIVE numKVBlocks keyScale (i, d, PUnit.unit) := by
  simp only
  rw [aftgStateBotG_ratio_eq _ _ _ _ _ _ _ _ hne]
  rw [aftgKeysUptoG_full_eq_pred]
  rw [attnFwdTritonOutSpecG_eq_streaming]
  rw [VeriTile.Triton.osStep_foldl_eq_batch]

set_option maxRecDepth 8000 in
/-- **General body split** — the lowered general AFT2 body decomposes as
`take 22 ++ (forRange "start_n" 0 N_CTX BLOCK_N aftgLoopBodyG :: drop 23)`. The
static `forRange` (NOT `forRangeDyn`) sits at index 22 regardless of dimension
values (the AST structure is dim-value-independent), checked by `rfl`. -/
theorem aftg_body_splitG
    (Q K V QScale KScale Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat) :
    (attn_fwd_triton_surface Q K V QScale KScale Out
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.body
      = (attn_fwd_triton_surface Q K V QScale KScale Out
          sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
          Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.body.take 22
        ++ (Stmt.forRange "start_n" 0 N_CTX BLOCK_N
              (AftgFoundation.aftgLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE)
            :: (attn_fwd_triton_surface Q K V QScale KScale Out
                sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
                Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.body.drop 23) :=
  rfl

set_option maxRecDepth 8000 in
/-- `body.take 22 = aftgPreLoopG` at the contiguous symbolic layout. Checked by `rfl`. -/
theorem aftgPreLoopG_check (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE Z : Nat) :
    (attn_fwd_triton_surface Q K V QScale KScale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.body.take 22
      = AftgFoundation.aftgPreLoopG Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE :=
  rfl

set_option maxRecDepth 8000 in
/-- `body.drop 23 = aftgPostLoopG` at the contiguous symbolic layout. Checked by `rfl`. -/
theorem aftgPostLoopG_check (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE Z : Nat) :
    (attn_fwd_triton_surface Q K V QScale KScale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.body.drop 23
      = AftgFoundation.aftgPostLoopG Out N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE :=
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General preLoop head execution.** -/
theorem aftgPreLoopHeadG_eval
    (s : BlockState) (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s1, stepStmts (AftgFoundation.aftgPreLoopHeadG stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL) s = some s1
      ∧ s1.pids = s.pids ∧ s1.mem = s.mem ∧ (∀ rg o, s1.undef rg o = 0)
      ∧ s1.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s1.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 1))
      ∧ s1.regs .nat [] "qvk_offset" = some (Tile.scalar (baseOffsetAFT2G s stride_qz stride_qh H))
      ∧ s1.regs .nat [] "q_scale_offset"
          = some (Tile.scalar (s.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M)))
      ∧ s1.regs .nat [] "k_scale_offset"
          = some (Tile.scalar (s.pids 1 * ((N_CTX + BLOCK_N - 1) / BLOCK_N)))
      ∧ s1.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val))
      ∧ s1.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
      ∧ s1.regs .nat [BLOCK_DMODEL] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val)) := by
  unfold AftgFoundation.aftgPreLoopHeadG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)) _
        = some (Tile.scalar (s.pids 1 / H)) from by
      rw [evalOp]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mod .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)) _
        = some (Tile.scalar (s.pids 1 % H)) from by
      rw [evalOp]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat stride_qz))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat stride_qh))) _
        = some (Tile.scalar (baseOffsetAFT2G s stride_qz stride_qh H)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "qvk_offset") (Op.constNat HEAD_DIM)) _
        = some (Tile.scalar (baseOffsetAFT2G s stride_qz stride_qh H / HEAD_DIM)) from by
      rw [evalOp]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N_CTX) (Op.constNat BLOCK_M)) (Op.constNat 1))
          (Op.constNat BLOCK_M))) _
        = some (Tile.scalar (s.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M))) from by
      rw [evalOp_mul, evalOp_div, evalOp_sub, evalOp_add]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N_CTX) (Op.constNat BLOCK_N)) (Op.constNat 1))
          (Op.constNat BLOCK_N))) _
        = some (Tile.scalar (s.pids 1 * ((N_CTX + BLOCK_N - 1) / BLOCK_N))) from by
      rw [evalOp_mul, evalOp_div, evalOp_sub, evalOp_add]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)) (Op.arange BLOCK_M)) _
        = some (Tile.vec (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val)) from by
      rw [evalOp_add, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, evalOp_arange, BlockState.setReg_same,
        BlockState.setReg_pids, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BLOCK_N) _ = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) from
      evalOp_arange BLOCK_N _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BLOCK_DMODEL) _ = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val)) from
      evalOp_arange BLOCK_DMODEL _))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · intro rg o; exact hundef rg o
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]

/-! ### General streamed-pointer closed cell-forms -/

/-- General `K_ptrs` after `c` blocks: `[BLOCK_DMODEL, BLOCK_N]`, cell `(e, jL)` →
`K[base + e + (c·BLOCK_N + jL)·HEAD_DIM]`. -/
noncomputable def kPtrsAFT2G (s0 : BlockState) (K : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL c : Nat) : Tile .ptr [BLOCK_DMODEL, BLOCK_N] :=
  ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
    (K.cast, baseOffsetAFT2G s0 stride_qz stride_qh H + idx.1.val + (c * BLOCK_N + idx.2.1.val) * HEAD_DIM)⟩

/-- General `K_scale_ptr` after `c` blocks. -/
noncomputable def kScalePtrAFT2G (s0 : BlockState) (KScale : RegionName)
    (N_CTX BLOCK_N c : Nat) : Tile .ptr [] :=
  ⟨fun _ : TileIndex [] => (KScale.cast, s0.pids 1 * ((N_CTX + BLOCK_N - 1) / BLOCK_N) + c)⟩

/-- General `V_ptrs` after `c` blocks: `[BLOCK_N, BLOCK_DMODEL]`, cell `(jL, d)` →
`V[base + (c·BLOCK_N + jL)·HEAD_DIM + d]`. -/
noncomputable def vPtrsAFT2G (s0 : BlockState) (V : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL c : Nat) : Tile .ptr [BLOCK_N, BLOCK_DMODEL] :=
  ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
    (V.cast, baseOffsetAFT2G s0 stride_qz stride_qh H + (c * BLOCK_N + idx.1.val) * HEAD_DIM + idx.2.1.val)⟩

/-- General `O_block_ptr` (constant): `[BLOCK_M, BLOCK_DMODEL]`, cell `(i, e)` →
`Out[base + (pids0·BLOCK_M + i)·HEAD_DIM + e]`. -/
noncomputable def oBlockPtrAFT2G (s0 : BlockState) (Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL : Nat) : Tile .ptr [BLOCK_M, BLOCK_DMODEL] :=
  ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
    (Out.cast, baseOffsetAFT2G s0 stride_qz stride_qh H + (s0.pids 0 * BLOCK_M + idx.1.val) * HEAD_DIM + idx.2.1.val)⟩

theorem kPtrsAFT2G_succ (s0 : BlockState) (K : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL c : Nat) :
    Tile.ptrAdd Broadcast.scalarR (kPtrsAFT2G s0 K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL c)
        (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar BLOCK_N) (Tile.scalar HEAD_DIM))
      = kPtrsAFT2G s0 K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL (c + 1) := by
  ext idx
  · rfl
  · simp only [kPtrsAFT2G, Tile.ptrAdd_data, Tile.bop_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR, Broadcast.leftIndex_nil,
      Broadcast.rightIndex_nil, NumericDType.nat_mul]
    ring

theorem kScalePtrAFT2G_succ (s0 : BlockState) (KScale : RegionName) (N_CTX BLOCK_N c : Nat) :
    Tile.ptrAdd Broadcast.nil (kScalePtrAFT2G s0 KScale N_CTX BLOCK_N c) (Tile.scalar 1)
      = kScalePtrAFT2G s0 KScale N_CTX BLOCK_N (c + 1) := by
  ext idx
  · rfl
  · simp only [kScalePtrAFT2G, Tile.ptrAdd_data, Tile.scalar,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    omega

theorem vPtrsAFT2G_succ (s0 : BlockState) (V : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL c : Nat) :
    Tile.ptrAdd Broadcast.scalarR (vPtrsAFT2G s0 V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL c)
        (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar BLOCK_N) (Tile.scalar HEAD_DIM))
      = vPtrsAFT2G s0 V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL (c + 1) := by
  ext idx
  · rfl
  · simp only [vPtrsAFT2G, Tile.ptrAdd_data, Tile.bop_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR, Broadcast.leftIndex_nil,
      Broadcast.rightIndex_nil, NumericDType.nat_mul]
    ring

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General preLoop tail execution.** -/
theorem aftgPreLoopTailG_eval
    (s1 : BlockState) (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat)
    (hundef : ∀ rg o, s1.undef rg o = 0)
    (hstartm : s1.regs .nat [] "start_m" = some (Tile.scalar (s1.pids 0)))
    (hoffhz : s1.regs .nat [] "off_hz" = some (Tile.scalar (s1.pids 1)))
    (hqvk : s1.regs .nat [] "qvk_offset" = some (Tile.scalar (baseOffsetAFT2G s1 stride_qz stride_qh H)))
    (hqso : s1.regs .nat [] "q_scale_offset"
        = some (Tile.scalar (s1.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M))))
    (hkso : s1.regs .nat [] "k_scale_offset"
        = some (Tile.scalar (s1.pids 1 * ((N_CTX + BLOCK_N - 1) / BLOCK_N))))
    (hoffsm : s1.regs .nat [BLOCK_M] "offs_m"
        = some (Tile.vec (fun r : Fin BLOCK_M => s1.pids 0 * BLOCK_M + r.val)))
    (hoffsn : s1.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hoffsk : s1.regs .nat [BLOCK_DMODEL] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))) :
    ∃ s0, stepStmts (AftgFoundation.aftgPreLoopTailG Q K V QScale KScale Out HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) s1 = some s0
      ∧ s0.pids = s1.pids ∧ s0.mem = s1.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "start_m" = some (Tile.scalar (s1.pids 0))
      ∧ s0.regs .nat [] "off_hz" = some (Tile.scalar (s1.pids 1))
      ∧ s0.regs .real [BLOCK_M] "m_i" = some ⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩
      ∧ s0.regs .real [BLOCK_M] "l_i" = some ⟨fun _ : TileIndex [BLOCK_M] => (some (1 : ℝ) : WithBot ℝ)⟩
      ∧ s0.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some ⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => (some (0 : ℝ) : WithBot ℝ)⟩
      ∧ s0.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => s1.pids 0 * BLOCK_M + r.val))
      ∧ s0.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
      ∧ s0.regs .nat [BLOCK_DMODEL] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))
      ∧ s0.regs .real [] "q_scale" = some ⟨fun _ : TileIndex [] =>
          some (s1.readMem QScale ((s1.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M) + s1.pids 0)))⟩
      ∧ s0.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some (kPtrsAFT2G s1 K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL 0)
      ∧ s0.regs .ptr [] "K_scale_ptr" = some (kScalePtrAFT2G s1 KScale N_CTX BLOCK_N 0)
      ∧ s0.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some (vPtrsAFT2G s1 V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL 0)
      ∧ s0.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr" = some (oBlockPtrAFT2G s1 Out stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL)
      ∧ s0.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          if s1.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE
          then some (qTileAFT2G s1 Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL idx) else some (0.0 : ℝ)⟩ := by
  unfold AftgFoundation.aftgPreLoopTailG
  -- stmt 11: Q_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aftg_evalOp_ptrAdd_of Broadcast.scalarL (Op.ptrBase Q) _ _ _ _
      (aftg_evalOp_ptrBase Q _)
      (show evalOp (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1))) _
          = some _ from by
        rw [evalOp_add, evalOp_add, evalOp_mul, evalOp_mul]
        erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hoffsm,
          evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hoffsk]
        rw [evalOp_ref, hqvk]
        simp only [evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
        rfl)))]
  -- stmt 12: Q_scale_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aftg_evalOp_ptrAdd_of Broadcast.nil (Op.ptrBase QScale) _ _ _ _
      (aftg_evalOp_ptrBase QScale _)
      (show evalOp (Op.add .nat Broadcast.nil (Op.ref .nat [] "q_scale_offset") (Op.ref .nat [] "start_m")) _
          = some _ from by
        simp only [evalOp_add, evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, hqso, hstartm, Option.bind_some, Option.bind_eq_bind]
        rfl)))]
  -- stmt 13: K_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aftg_evalOp_ptrAdd_of Broadcast.scalarL (Op.ptrBase K) _ _ _ _
      (aftg_evalOp_ptrBase K _)
      (show evalOp (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM))) _
          = some _ from by
        rw [evalOp_add, evalOp_add, evalOp_mul]
        erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hoffsk)),
            evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hoffsn))]
        rw [evalOp_ref, regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hqvk)]
        simp only [evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
        rfl)))]
  -- stmt 14: K_scale_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aftg_evalOp_ptrAdd_of Broadcast.nil (Op.ptrBase KScale) _ _ _ _
      (aftg_evalOp_ptrBase KScale _)
      (show evalOp (Op.ref .nat [] "k_scale_offset") _ = some _ from by
        simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true]
        rw [hkso])))]
  -- stmt 15: V_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aftg_evalOp_ptrAdd_of Broadcast.scalarL (Op.ptrBase V) _ _ _ _
      (aftg_evalOp_ptrBase V _)
      (show evalOp (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1))) _
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
  -- stmt 16: O_block_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aftg_evalOp_ptrAdd_of Broadcast.scalarL (Op.ptrBase Out) _ _ _ _
      (aftg_evalOp_ptrBase Out _)
      (show evalOp (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1))) _
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
        = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => (some (0 : ℝ) : WithBot ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_DMODEL]) from by
      simp only [evalOp_full, evalOp_const]
      rfl))]
  -- stmt 20: q = masked load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aftg_evalOp_load_ptr_mask_of (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs") _ _ _ _
      (by rw [evalOp_ref,
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_same])
      (show evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat N_CTX))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))) _
          = some _ from by
        rw [aftg_evalOp_boolAnd, evalOp_lt]
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
  -- stmt 21: q_scale = load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aftg_evalOp_load_ptr_none_of (Op.ref .ptr [] "Q_scale_ptr") _ _
      (by rw [evalOp_ref,
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_same])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · intro rg o; exact hundef rg o
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids, hstartm]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids, hoffhz]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids, hoffsm]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, hoffsn]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, hoffsk]
  · -- q_scale
    simp only [BlockState.setReg_same, BlockState.setReg_pids]
    refine congrArg some ?_
    ext _
    simp only [BlockState.readMem, BlockState.setReg_mem, castTile_self,
      Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
      Tile.bop_data, Region.cast, NumericDType.nat_add, Nat.zero_add]
  · -- K_ptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    · simp only [kPtrsAFT2G, Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex_scalarL,
        Broadcast.rightIndex_scalarL, Region.cast]
    · simp only [kPtrsAFT2G, Tile.ptrAdd_data, Tile.scalar, Tile.bop_data, Tile.expandDim_data,
        Tile.vec_data, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
        Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, NumericDType.nat_add,
        NumericDType.nat_mul, baseOffsetAFT2G]
      ring_nf
  · -- K_scale_ptr
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    · simp only [kScalePtrAFT2G, Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex_nil,
        Broadcast.rightIndex_nil, Region.cast]
    · simp only [kScalePtrAFT2G, Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex_nil,
        Broadcast.rightIndex_nil, Nat.zero_add, Nat.add_zero]
  · -- V_ptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    · simp only [vPtrsAFT2G, Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex_scalarL,
        Broadcast.rightIndex_scalarL, Region.cast]
    · simp only [vPtrsAFT2G, Tile.ptrAdd_data, Tile.scalar, Tile.bop_data, Tile.expandDim_data,
        Tile.vec_data, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
        Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, NumericDType.nat_add,
        NumericDType.nat_mul, baseOffsetAFT2G]
      ring_nf
  · -- O_block_ptr
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    · simp only [oBlockPtrAFT2G, Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex_scalarL,
        Broadcast.rightIndex_scalarL, Region.cast]
    · simp only [oBlockPtrAFT2G, Tile.ptrAdd_data, Tile.scalar, Tile.bop_data, Tile.expandDim_data,
        Tile.vec_data, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
        Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, NumericDType.nat_add,
        NumericDType.nat_mul, baseOffsetAFT2G]
      ring_nf
  · -- q
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
    rw [show (ComparableDType.nat.lt (s1.pids 0 * BLOCK_M + idx.1.val) N_CTX
          && ComparableDType.nat.lt idx.2.1.val HEAD_ACTIVE)
        = decide (s1.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE) from by
      rw [Bool.eq_iff_iff]; simp only [Bool.and_eq_true, ComparableDType.nat_lt_eq_true,
        decide_eq_true_eq]]
    by_cases hk : s1.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE
    · rw [if_pos (by simp only [decide_eq_true_eq]; exact hk), if_pos hk]
      refine congrArg some ?_
      simp only [qTileAFT2G, BlockState.readMem, BlockState.setReg_mem, castTile_self,
        Tile.ptrAdd_data, Tile.scalar, Tile.bop_data,
        Tile.expandDim_data, Tile.vec_data, Broadcast.leftIndex_scalarL,
        Broadcast.rightIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul, NumericDType.nat_add, NumericDType.nat_mul,
        Region.cast, baseOffsetAFT2G]
      ring_nf
    · rw [if_neg (by simp only [decide_eq_true_eq]; exact hk), if_neg hk]
      simp only [BlockState.setReg_undef, hundef]
      norm_num

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General preLoop execution** (compose head + tail). -/
theorem aftgPreLoopG_eval
    (s : BlockState) (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (AftgFoundation.aftgPreLoopG Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) s = some s0
      ∧ s0.pids = s.pids ∧ s0.mem = s.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 1))
      ∧ s0.regs .real [BLOCK_M] "m_i" = some ⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩
      ∧ s0.regs .real [BLOCK_M] "l_i" = some ⟨fun _ : TileIndex [BLOCK_M] => (some (1 : ℝ) : WithBot ℝ)⟩
      ∧ s0.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some ⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => (some (0 : ℝ) : WithBot ℝ)⟩
      ∧ s0.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val))
      ∧ s0.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
      ∧ s0.regs .nat [BLOCK_DMODEL] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))
      ∧ s0.regs .real [] "q_scale" = some ⟨fun _ : TileIndex [] =>
          some (s.readMem QScale ((s.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M) + s.pids 0)))⟩
      ∧ s0.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some (kPtrsAFT2G s K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL 0)
      ∧ s0.regs .ptr [] "K_scale_ptr" = some (kScalePtrAFT2G s KScale N_CTX BLOCK_N 0)
      ∧ s0.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some (vPtrsAFT2G s V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL 0)
      ∧ s0.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr" = some (oBlockPtrAFT2G s Out stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL)
      ∧ s0.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          if s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE
          then some (qTileAFT2G s Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL idx) else some (0.0 : ℝ)⟩ := by
  rw [AftgFoundation.aftgPreLoopG_eq_head_tail]
  obtain ⟨s1, hHead, hpids1, hmem1, hundef1, hstartm1, hoffhz1, hqvk1, hqso1, hkso1,
    hoffsm1, hoffsn1, hoffsk1⟩ := aftgPreLoopHeadG_eval s stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL hundef
  rw [stepStmts.append_some hHead]
  have hstartm1' : s1.regs .nat [] "start_m" = some (Tile.scalar (s1.pids 0)) := by
    rw [hpids1]; exact hstartm1
  have hoffhz1' : s1.regs .nat [] "off_hz" = some (Tile.scalar (s1.pids 1)) := by
    rw [hpids1]; exact hoffhz1
  have hqvk1' : s1.regs .nat [] "qvk_offset" = some (Tile.scalar (baseOffsetAFT2G s1 stride_qz stride_qh H)) := by
    rw [show baseOffsetAFT2G s1 stride_qz stride_qh H = baseOffsetAFT2G s stride_qz stride_qh H from by
      simp only [baseOffsetAFT2G, hpids1]]; exact hqvk1
  have hqso1' : s1.regs .nat [] "q_scale_offset"
      = some (Tile.scalar (s1.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M))) := by
    rw [hpids1]; exact hqso1
  have hkso1' : s1.regs .nat [] "k_scale_offset"
      = some (Tile.scalar (s1.pids 1 * ((N_CTX + BLOCK_N - 1) / BLOCK_N))) := by
    rw [hpids1]; exact hkso1
  have hoffsm1' : s1.regs .nat [BLOCK_M] "offs_m"
      = some (Tile.vec (fun r : Fin BLOCK_M => s1.pids 0 * BLOCK_M + r.val)) := by
    rw [hpids1]; exact hoffsm1
  obtain ⟨s0, hTail, hpids0, hmem0, hundef0, hstartm0, hoffhz0, hmi0, hli0, hacc0,
    hoffsm0, hoffsn0, hoffsk0, hqscale0, hkp0, hksp0, hvp0, hop0, hq0⟩ :=
    aftgPreLoopTailG_eval s1 Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE hundef1
      hstartm1' hoffhz1' hqvk1' hqso1' hkso1' hoffsm1' hoffsn1 hoffsk1
  have hbase : baseOffsetAFT2G s1 stride_qz stride_qh H = baseOffsetAFT2G s stride_qz stride_qh H := by
    simp only [baseOffsetAFT2G, hpids1]
  have hkpEq : kPtrsAFT2G s1 K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL 0 = kPtrsAFT2G s K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL 0 := by
    simp only [kPtrsAFT2G, hbase]
  have hkspEq : kScalePtrAFT2G s1 KScale N_CTX BLOCK_N 0 = kScalePtrAFT2G s KScale N_CTX BLOCK_N 0 := by
    simp only [kScalePtrAFT2G, hpids1]
  have hvpEq : vPtrsAFT2G s1 V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL 0 = vPtrsAFT2G s V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL 0 := by
    simp only [vPtrsAFT2G, hbase]
  have hopEq : oBlockPtrAFT2G s1 Out stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL = oBlockPtrAFT2G s Out stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL := by
    simp only [oBlockPtrAFT2G, hbase, hpids1]
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
    simp only [hpids1, qTileAFT2G, BlockState.readMem, hmem1, hbase]

/-- **General loop invariant** for the AFT2 streaming loop (counter `i = c·BLOCK_N`).
Binds the running registers to the seed-1 ⊥-state over the first `i` keys, the
static index vectors, loaded `q`/`q_scale`, the three streamed pointers, and
preserves `undef`/`mem`. -/
noncomputable def aftgInvariantG
    (Q K V QScale KScale Out : RegionName) (s0 : BlockState)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (keyScale : Fin (BLOCK_N * numKVBlocks) → ℝ) (hBD : 0 < BLOCK_DMODEL)
    (i : Nat) (s : BlockState) : Prop :=
  let qStart := qStartAFT2G s0 BLOCK_M
  let qT := qTileAFT2mG s0 Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE
  let kT := kTileAFT2G s0 K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL
  let vT := vTileAFT2mG s0 V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE
  s.pids = s0.pids ∧ i % BLOCK_N = 0 ∧ i ≤ N_CTX ∧
  (s.regs .real [BLOCK_M] "m_i" = some ⟨fun r : TileIndex [BLOCK_M] =>
      aftgRunningMaxG qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩⟩) ∧
  (s.regs .real [BLOCK_M] "l_i" = some ⟨fun r : TileIndex [BLOCK_M] =>
      ((aftgStateBot1G qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩).2.1 : ℝ)⟩) ∧
  (s.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
      ((aftgStateBot1G qT kT vT keyScale qStart i idx.1 idx.2.1).2.2 : ℝ)⟩) ∧
  (s.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => qStart + r.val))) ∧
  (s.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))) ∧
  (s.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
      if qStart + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE then
        some (qTileAFT2G s0 Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL idx) else some (0.0 : ℝ)⟩) ∧
  (s.regs .real [] "q_scale" = some (Tile.scalar
      (some (s0.readMem QScale (s0.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M) + s0.pids 0))))) ∧
  (s.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some (kPtrsAFT2G s0 K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL (i / BLOCK_N))) ∧
  (s.regs .ptr [] "K_scale_ptr" = some (kScalePtrAFT2G s0 KScale N_CTX BLOCK_N (i / BLOCK_N))) ∧
  (s.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some (vPtrsAFT2G s0 V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL (i / BLOCK_N))) ∧
  (s.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr" = some (oBlockPtrAFT2G s0 Out stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL)) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General preLoop ⇒ invariant base case.** -/
theorem aftgPreLoopG_invariant
    (s : BlockState) (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (hN : N_CTX = BLOCK_N * numKVBlocks) (hBN : 0 < BLOCK_N) (hBD : 0 < BLOCK_DMODEL)
    (keyScale : Fin (BLOCK_N * numKVBlocks) → ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (AftgFoundation.aftgPreLoopG Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) s = some s0
      ∧ aftgInvariantG Q K V QScale KScale Out s stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks keyScale hBD 0 s0 := by
  subst hN
  obtain ⟨s0, hstep, hpids, hmem, hundef0, hstartm, hoffhz, hmi, hli, hacc,
    hoffsm, hoffsn, hoffsk, hqscale, hkp, hksp, hvp, hop, hq⟩ :=
    aftgPreLoopG_eval s Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE hundef
  refine ⟨s0, hstep, ?_⟩
  simp only [aftgInvariantG, qStartAFT2G]
  refine ⟨hpids, by norm_num, by norm_num, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, fun rg o => hundef0 rg o, hmem⟩
  · rw [hmi]; refine congrArg some ?_; refine Tile.ext (fun r => ?_)
    simp only [aftgRunningMaxG_zero]
  · rw [hli]; refine congrArg some ?_; refine Tile.ext (fun r => ?_)
    simp only [aftgStateBot1G_zero]; rfl
  · rw [hacc]; refine congrArg some ?_; refine Tile.ext (fun idx => ?_)
    simp only [aftgStateBot1G_zero]; rfl
  · rw [hoffsm]
  · exact hoffsn
  · rw [hq]; rfl
  · rw [hqscale]; rfl
  · rw [hkp, Nat.zero_div]
  · rw [hksp, Nat.zero_div]
  · rw [hvp, Nat.zero_div]
  · rw [hop]

/-! ### General loop-body execution chain -/

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- General loop-body head execution (statements 0–10). -/
theorem aftgLoopBodyHeadG_steps
    (N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) (hBN : 0 < BLOCK_N)
    (sin : BlockState) (SN : Nat)
    (offsm : Tile .nat [BLOCK_M]) (offsn : Tile .nat [BLOCK_N])
    (kptrs : Tile .ptr [BLOCK_DMODEL, BLOCK_N]) (ksptr : Tile .ptr [])
    (mtile : Tile .real [BLOCK_M]) (qtile : Tile .real [BLOCK_M, BLOCK_DMODEL]) (qsc : Tile .real [])
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoffsm : sin.regs .nat [BLOCK_M] "offs_m" = some offsm)
    (hoffsn : sin.regs .nat [BLOCK_N] "offs_n" = some offsn)
    (hmi : sin.regs .real [BLOCK_M] "m_i" = some mtile)
    (hkp : sin.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some kptrs)
    (hksp : sin.regs .ptr [] "K_scale_ptr" = some ksptr)
    (hq : sin.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some qtile)
    (hqsc : sin.regs .real [] "q_scale" = some qsc) :
    ∃ s1, stepStmts (AftgFoundation.aftgLoopBodyHeadG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) sin = some s1
      ∧ s1.pids = sin.pids ∧ s1.mem = sin.mem ∧ (∀ rg o, s1.undef rg o = sin.undef rg o)
      ∧ ∃ (kmaskT : Tile .bool [BLOCK_DMODEL, BLOCK_N]) (ktile : Tile .real [BLOCK_DMODEL, BLOCK_N])
          (kscT : Tile .real []) (qkdotT : Tile .real [BLOCK_M, BLOCK_N])
          (maskT : Tile .bool [BLOCK_M, BLOCK_N]) (qkSentT : Tile .real [BLOCK_M, BLOCK_N])
          (rmaxT mijT : Tile .real [BLOCK_M]) (qkShiftT pExpT pT : Tile .real [BLOCK_M, BLOCK_N]),
        (kmaskT = ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
            (ComparableDType.nat.lt (offsn.data (idx.2.1, PUnit.unit)) (N_CTX - SN))
              && (ComparableDType.nat.lt idx.1.val HEAD_ACTIVE)⟩)
        ∧ (ktile = ⟨fun i : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
            if kmaskT.data i then some (sin.readMem (kptrs.data i).1 (kptrs.data i).2)
            else some (sin.undef (kptrs.data i).1 (kptrs.data i).2)⟩)
        ∧ (kscT = ⟨fun _ : TileIndex [] =>
            some (sin.readMem (ksptr.data PUnit.unit).1 (ksptr.data PUnit.unit).2)⟩)
        ∧ (qkdotT = Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.bop NumericDType.real.mul Broadcast.scalarR
              ⟨fun i => (Tile.dot [] qtile ktile).data i⟩ qsc) kscT)
        ∧ (maskT = ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            ComparableDType.nat.ge (offsm.data (idx.1, PUnit.unit))
              (SN + offsn.data (idx.2.1, PUnit.unit))⟩)
        ∧ (qkSentT = ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            if maskT.data idx then qkdotT.data idx
            else WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ))⟩)
        ∧ (Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) qkSentT = some rmaxT)
        ∧ (mijT = Tile.select
            (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT)
            mtile rmaxT)
        ∧ (qkShiftT = Tile.bop NumericDType.real.sub
            (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))
        ∧ (pExpT = Tile.uop WithBot.realExp2 qkShiftT)
        ∧ (pT = ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            if maskT.data idx then pExpT.data idx else (some (0.0 : ℝ) : WithBot ℝ)⟩)
        ∧ s1.regs .real [BLOCK_M] "m_i" = some mtile
        ∧ s1.regs .real [BLOCK_M] "m_ij" = some mijT
        ∧ s1.regs .bool [BLOCK_M, BLOCK_N] "mask" = some maskT
        ∧ s1.regs .real [BLOCK_M, BLOCK_N] "p" = some pT
        ∧ s1.regs .real [BLOCK_M] "l_i" = sin.regs .real [BLOCK_M] "l_i"
        ∧ s1.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = sin.regs .real [BLOCK_M, BLOCK_DMODEL] "acc"
        ∧ s1.regs .real [BLOCK_N, BLOCK_DMODEL] "v" = sin.regs .real [BLOCK_N, BLOCK_DMODEL] "v"
        ∧ s1.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = sin.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
        ∧ s1.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some kptrs
        ∧ s1.regs .ptr [] "K_scale_ptr" = some ksptr
        ∧ s1.regs .nat [BLOCK_M] "offs_m" = some offsm
        ∧ s1.regs .nat [BLOCK_N] "offs_n" = some offsn
        ∧ s1.regs .nat [] "start_n" = some (Tile.scalar SN)
        ∧ s1.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some qtile
        ∧ s1.regs .real [] "q_scale" = some qsc := by
  set kmaskT : Tile .bool [BLOCK_DMODEL, BLOCK_N] := ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
      (ComparableDType.nat.lt (offsn.data (idx.2.1, PUnit.unit)) (N_CTX - SN))
        && (ComparableDType.nat.lt idx.1.val HEAD_ACTIVE)⟩ with hkmaskT
  set ktile : Tile .real [BLOCK_DMODEL, BLOCK_N] := ⟨fun i : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
      if kmaskT.data i then some (sin.readMem (kptrs.data i).1 (kptrs.data i).2)
      else some (sin.undef (kptrs.data i).1 (kptrs.data i).2)⟩ with hktile
  set kscT : Tile .real [] := ⟨fun _ : TileIndex [] =>
      some (sin.readMem (ksptr.data PUnit.unit).1 (ksptr.data PUnit.unit).2)⟩ with hkscT
  set qkdotT : Tile .real [BLOCK_M, BLOCK_N] := Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        ⟨fun i => (Tile.dot [] qtile ktile).data i⟩ qsc) kscT with hqkdotT
  set maskT : Tile .bool [BLOCK_M, BLOCK_N] := ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      ComparableDType.nat.ge (offsm.data (idx.1, PUnit.unit))
        (SN + offsn.data (idx.2.1, PUnit.unit))⟩ with hmaskT
  set qkSentT : Tile .real [BLOCK_M, BLOCK_N] := ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      if maskT.data idx then qkdotT.data idx
      else WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ))⟩ with hqkSentT
  obtain ⟨rmaxT, hrm⟩ := aftg_reduceMaxDrop1_someG BLOCK_M BLOCK_N hBN qkSentT
  set mijT : Tile .real [BLOCK_M] := Tile.select
      (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT)
      mtile rmaxT with hmijT
  set qkShiftT : Tile .real [BLOCK_M, BLOCK_N] := Tile.bop NumericDType.real.sub
      (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT) with hqkShiftT
  set pExpT : Tile .real [BLOCK_M, BLOCK_N] := Tile.uop WithBot.realExp2 qkShiftT with hpExpT
  set pT : Tile .real [BLOCK_M, BLOCK_N] := ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      if maskT.data idx then pExpT.data idx else (some (0.0 : ℝ) : WithBot ℝ)⟩ with hpT
  unfold AftgFoundation.aftgLoopBodyHeadG AftgFoundation.aftgLoopBodyG
  simp only [List.take_succ_cons, List.take_zero]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_n") sin = some (Tile.scalar SN) from by rw [evalOp_ref, hsn]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some kmaskT from by
      rw [aftg_kmask_evalG _ SN N_CTX HEAD_ACTIVE BLOCK_N BLOCK_DMODEL offsn
        (by simp [BlockState.setReg_ne_name, hoffsn])
        (by rw [BlockState.setReg_same])]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some ktile from by
      rw [aftg_load_k_eval _ BLOCK_DMODEL BLOCK_N "K_ptrs" "k_mask" kptrs kmaskT
        (by simp [BlockState.setReg_ne_name, hkp]) (by rw [BlockState.setReg_same])]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some kscT from by
      rw [aftg_load_kscale_eval _ "K_scale_ptr" ksptr
        (by simp [BlockState.setReg_ne_name, hksp])]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some qkdotT from by
      rw [aftg_qk_dot_eval _ BLOCK_M BLOCK_N BLOCK_DMODEL qtile ktile qsc kscT
        (by simp [BlockState.setReg_ne_name, hq]) (by simp [BlockState.setReg_ne_name])
        (by simp [BlockState.setReg_ne_name, hqsc]) (by rw [BlockState.setReg_same])]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some maskT from by
      rw [aftg_mask_evalG _ SN BLOCK_M BLOCK_N offsm offsn
        (by simp [BlockState.setReg_ne_name, hoffsm]) (by simp [BlockState.setReg_ne_name, hoffsn])
        (by simp [BlockState.setReg_ne_name, hsn])]))]
  have hbcast6 : ∀ t : BlockState, @evalOp TileDType.real [BLOCK_M, BLOCK_N]
      (Op.broadcast (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1000000.0)) [BLOCK_M, BLOCK_N]) t
      = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] =>
          WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ))⟩ : Tile .real [BLOCK_M, BLOCK_N]) := by
    intro t
    simp only [evalOp, evalOp_sub, evalOp_const, Option.bind_eq_bind, Option.bind_some]
    rfl
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.where (Op.ref .bool [BLOCK_M, BLOCK_N] "mask")
        (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
        (Op.broadcast (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1000000.0)) [BLOCK_M, BLOCK_N])) _
        = some qkSentT from by
      rw [evalOp_where]
      simp only [evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        hbcast6, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_
      ext idx
      simp only [hqkSentT, Tile.select_data, Tile.scalar]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some mijT from by
      rw [aftg_mij_evalG _ BLOCK_M BLOCK_N mtile qkSentT rmaxT
        (by simp [BlockState.setReg_ne_name, hmi]) (by rw [BlockState.setReg_same]) hrm]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some qkShiftT from by
      rw [aftg_qk_sub_evalG _ BLOCK_M BLOCK_N (by simp) qkSentT mijT
        (by simp [BlockState.setReg_ne_name]) (by rw [BlockState.setReg_same])]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some pExpT from by
      rw [aftg_p_evalG _ BLOCK_M BLOCK_N qkShiftT (by rw [BlockState.setReg_same])]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some pT from by
      rw [aftg_p_mask_evalG _ BLOCK_M BLOCK_N maskT pExpT
        (by simp [BlockState.setReg_ne_name]) (by rw [BlockState.setReg_same])]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, kmaskT, ktile, kscT, qkdotT, maskT, qkSentT, rmaxT, mijT,
    qkShiftT, pExpT, pT, rfl, rfl, rfl, rfl, rfl, rfl, hrm, rfl, rfl, rfl, rfl,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_pids]
  · funext rg o; simp [BlockState.setReg_mem]
  · intro rg o; simp [BlockState.setReg_undef]
  · simp [BlockState.setReg_ne_name, hmi]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name, hkp]
  · simp [BlockState.setReg_ne_name, hksp]
  · simp [BlockState.setReg_ne_name, hoffsm]
  · simp [BlockState.setReg_ne_name, hoffsn]
  · simp [BlockState.setReg_ne_name, hsn]
  · simp [BlockState.setReg_ne_name, hq]
  · simp [BlockState.setReg_ne_name, hqsc]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- General loop-body tail execution (statements 11–21). -/
theorem aftgLoopBodyTailG_steps
    (N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat)
    (s1 : BlockState) (SN : Nat)
    (offsn : Tile .nat [BLOCK_N])
    (kptrs : Tile .ptr [BLOCK_DMODEL, BLOCK_N]) (ksptr : Tile .ptr []) (vptrs : Tile .ptr [BLOCK_N, BLOCK_DMODEL])
    (mtile mijT : Tile .real [BLOCK_M]) (maskT : Tile .bool [BLOCK_M, BLOCK_N])
    (pT : Tile .real [BLOCK_M, BLOCK_N]) (litile : Tile .real [BLOCK_M]) (acctile : Tile .real [BLOCK_M, BLOCK_DMODEL])
    (hsn : s1.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoffsn : s1.regs .nat [BLOCK_N] "offs_n" = some offsn)
    (hmi : s1.regs .real [BLOCK_M] "m_i" = some mtile)
    (hmij : s1.regs .real [BLOCK_M] "m_ij" = some mijT)
    (hmask : s1.regs .bool [BLOCK_M, BLOCK_N] "mask" = some maskT)
    (hp : s1.regs .real [BLOCK_M, BLOCK_N] "p" = some pT)
    (hli : s1.regs .real [BLOCK_M] "l_i" = some litile)
    (hacc : s1.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some acctile)
    (hvp : s1.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some vptrs)
    (hkp : s1.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some kptrs)
    (hksp : s1.regs .ptr [] "K_scale_ptr" = some ksptr) :
    ∃ sF, stepStmts (AftgFoundation.aftgLoopBodyTailG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) s1 = some sF
      ∧ sF.pids = s1.pids ∧ sF.mem = s1.mem ∧ (∀ rg o, sF.undef rg o = s1.undef rg o)
      ∧ sF.regs .nat [BLOCK_M] "offs_m" = s1.regs .nat [BLOCK_M] "offs_m"
      ∧ sF.regs .nat [BLOCK_N] "offs_n" = s1.regs .nat [BLOCK_N] "offs_n"
      ∧ sF.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = s1.regs .real [BLOCK_M, BLOCK_DMODEL] "q"
      ∧ sF.regs .real [] "q_scale" = s1.regs .real [] "q_scale"
      ∧ ∃ (lijT alphaT : Tile .real [BLOCK_M]) (vmaskT : Tile .bool [BLOCK_N, BLOCK_DMODEL])
          (vtile : Tile .real [BLOCK_N, BLOCK_DMODEL]) (pf16 : Tile .fp16 [BLOCK_M, BLOCK_N]),
        (lijT = Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) pT)
        ∧ (alphaT = Tile.uop WithBot.realExp2
            (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
        ∧ (vmaskT = ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
            (ComparableDType.nat.lt (offsn.data (idx.1, PUnit.unit)) (N_CTX - SN))
              && (ComparableDType.nat.lt idx.2.1.val HEAD_ACTIVE)⟩)
        ∧ (vtile = ⟨fun i : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
            if vmaskT.data i then some (s1.readMem (vptrs.data i).1 (vptrs.data i).2)
            else some (s1.undef (vptrs.data i).1 (vptrs.data i).2)⟩)
        ∧ (pf16 = ⟨fun i => FloatDType.real.cast FloatDType.fp16 (pT.data i)⟩)
        ∧ sF.regs .real [BLOCK_M] "m_i" = some mijT
        ∧ sF.regs .real [BLOCK_M] "l_i" = some (Tile.bop NumericDType.real.add
            (Broadcast.consSame Broadcast.nil)
            (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) litile alphaT) lijT)
        ∧ sF.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some (Tile.bop NumericDType.real.add
            (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
            (Tile.dot [] ⟨fun i => FloatDType.fp16.cast FloatDType.real (pf16.data i)⟩ vtile))
        ∧ sF.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some
            (Tile.ptrAdd Broadcast.scalarR kptrs
              (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar BLOCK_N) (Tile.scalar HEAD_DIM)))
        ∧ sF.regs .ptr [] "K_scale_ptr" = some
            (Tile.ptrAdd Broadcast.nil ksptr (Tile.scalar 1))
        ∧ sF.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some
            (Tile.ptrAdd Broadcast.scalarR vptrs
              (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar BLOCK_N) (Tile.scalar HEAD_DIM))) := by
  set lijT : Tile .real [BLOCK_M] := Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) pT with hlijT
  set alphaT : Tile .real [BLOCK_M] := Tile.uop WithBot.realExp2
      (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT) with halphaT
  set vmaskT : Tile .bool [BLOCK_N, BLOCK_DMODEL] := ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
      (ComparableDType.nat.lt (offsn.data (idx.1, PUnit.unit)) (N_CTX - SN))
        && (ComparableDType.nat.lt idx.2.1.val HEAD_ACTIVE)⟩ with hvmaskT
  set vtile : Tile .real [BLOCK_N, BLOCK_DMODEL] := ⟨fun i : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
      if vmaskT.data i then some (s1.readMem (vptrs.data i).1 (vptrs.data i).2)
      else some (s1.undef (vptrs.data i).1 (vptrs.data i).2)⟩ with hvtile
  set pf16 : Tile .fp16 [BLOCK_M, BLOCK_N] := ⟨fun i => FloatDType.real.cast FloatDType.fp16 (pT.data i)⟩ with hpf16
  unfold AftgFoundation.aftgLoopBodyTailG AftgFoundation.aftgLoopBodyG
  simp only [List.drop_succ_cons, List.drop_zero]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some lijT from aftg_lij_evalG _ BLOCK_M BLOCK_N pT hp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some alphaT from by
      rw [aftg_alpha_evalG _ BLOCK_M mtile mijT
        (by simp [BlockState.setReg_ne_name, hmi]) (by simp [BlockState.setReg_ne_name, hmij])]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      aftg_li_evalG _ BLOCK_M litile alphaT lijT
        (by simp [BlockState.setReg_ne_name, hli]) (by rw [BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      aftg_acc_rescale_evalG _ BLOCK_M BLOCK_DMODEL (by simp) acctile alphaT
        (by simp [BlockState.setReg_ne_name, hacc]) (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some vtile from by
      rw [aftg_evalOp_load_ptr_mask_of (Op.ref .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs") _ _ vptrs vmaskT
        (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, hvp])
        (show evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
              (Op.sub .nat Broadcast.nil (Op.constNat N_CTX) (Op.ref .nat [] "start_n")))
            (Op.expandDim ⟨0, by simp⟩
              (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))) _
            = some vmaskT from by
          rw [aftg_evalOp_boolAnd, evalOp_lt]
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
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some pf16 from
      aftg_p_fp16_evalG _ BLOCK_M BLOCK_N pT (by simp [BlockState.setReg_ne_name, hp])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      aftg_acc_eval _ BLOCK_M BLOCK_N BLOCK_DMODEL
        (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
        pf16 vtile
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
        (by rw [BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some mijT from
      aftg_mi_carry_evalG _ BLOCK_M mijT (by simp [BlockState.setReg_ne_name, hmij])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      aftg_advance_ptr_evalG _ BLOCK_DMODEL BLOCK_N BLOCK_N HEAD_DIM "K_ptrs" kptrs
        (by simp [BlockState.setReg_ne_name, hkp])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      aftg_advance_kscale_eval _ "K_scale_ptr" ksptr
        (by simp [BlockState.setReg_ne_name, hksp])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      aftg_advance_ptr_evalG _ BLOCK_N BLOCK_DMODEL BLOCK_N HEAD_DIM "V_ptrs" vptrs
        (by simp [BlockState.setReg_ne_name, hvp])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, lijT, alphaT, vmaskT, vtile, pf16,
    rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_pids]
  · funext rg o; simp [BlockState.setReg_mem]
  · intro rg o; simp [BlockState.setReg_undef]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]


set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- General loop-body execution chain (22 statements). -/
theorem aftgLoopBody_stepsG
    (N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) (hBN : 0 < BLOCK_N)
    (sin : BlockState) (SN : Nat)
    (offsm : Tile .nat [BLOCK_M]) (offsn : Tile .nat [BLOCK_N])
    (kptrs : Tile .ptr [BLOCK_DMODEL, BLOCK_N]) (ksptr : Tile .ptr []) (vptrs : Tile .ptr [BLOCK_N, BLOCK_DMODEL])
    (mtile : Tile .real [BLOCK_M]) (qtile : Tile .real [BLOCK_M, BLOCK_DMODEL]) (qsc : Tile .real [])
    (litile : Tile .real [BLOCK_M]) (acctile : Tile .real [BLOCK_M, BLOCK_DMODEL])
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoffsm : sin.regs .nat [BLOCK_M] "offs_m" = some offsm)
    (hoffsn : sin.regs .nat [BLOCK_N] "offs_n" = some offsn)
    (hmi : sin.regs .real [BLOCK_M] "m_i" = some mtile)
    (hli : sin.regs .real [BLOCK_M] "l_i" = some litile)
    (hacc : sin.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some acctile)
    (hq : sin.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some qtile)
    (hqsc : sin.regs .real [] "q_scale" = some qsc)
    (hkp : sin.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some kptrs)
    (hksp : sin.regs .ptr [] "K_scale_ptr" = some ksptr)
    (hvp : sin.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some vptrs) :
    ∃ sF, stepStmts (AftgFoundation.aftgLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = sin.undef rg o)
      ∧ ∃ (kmaskT : Tile .bool [BLOCK_DMODEL, BLOCK_N]) (ktile : Tile .real [BLOCK_DMODEL, BLOCK_N])
          (kscT : Tile .real []) (qkdotT : Tile .real [BLOCK_M, BLOCK_N])
          (maskT : Tile .bool [BLOCK_M, BLOCK_N]) (qkSentT : Tile .real [BLOCK_M, BLOCK_N])
          (rmaxT mijT : Tile .real [BLOCK_M]) (pT : Tile .real [BLOCK_M, BLOCK_N])
          (lijT alphaT : Tile .real [BLOCK_M]) (vmaskT : Tile .bool [BLOCK_N, BLOCK_DMODEL])
          (vtile : Tile .real [BLOCK_N, BLOCK_DMODEL]) (pf16 : Tile .fp16 [BLOCK_M, BLOCK_N]),
        (kmaskT = ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
            (ComparableDType.nat.lt (offsn.data (idx.2.1, PUnit.unit)) (N_CTX - SN))
              && (ComparableDType.nat.lt idx.1.val HEAD_ACTIVE)⟩)
        ∧ (ktile = ⟨fun i : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
            if kmaskT.data i then some (sin.readMem (kptrs.data i).1 (kptrs.data i).2)
            else some (sin.undef (kptrs.data i).1 (kptrs.data i).2)⟩)
        ∧ (kscT = ⟨fun _ : TileIndex [] =>
            some (sin.readMem (ksptr.data PUnit.unit).1 (ksptr.data PUnit.unit).2)⟩)
        ∧ (qkdotT = Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.bop NumericDType.real.mul Broadcast.scalarR
              ⟨fun i => (Tile.dot [] qtile ktile).data i⟩ qsc) kscT)
        ∧ (maskT = ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            ComparableDType.nat.ge (offsm.data (idx.1, PUnit.unit))
              (SN + offsn.data (idx.2.1, PUnit.unit))⟩)
        ∧ (qkSentT = ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            if maskT.data idx then qkdotT.data idx
            else WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ))⟩)
        ∧ (Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) qkSentT = some rmaxT)
        ∧ (mijT = Tile.select
            (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT)
            mtile rmaxT)
        ∧ (pT = ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            if maskT.data idx then
              (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
                (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data idx
            else (some (0.0 : ℝ) : WithBot ℝ)⟩)
        ∧ (lijT = Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) pT)
        ∧ (alphaT = Tile.uop WithBot.realExp2
            (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
        ∧ (vmaskT = ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
            (ComparableDType.nat.lt (offsn.data (idx.1, PUnit.unit)) (N_CTX - SN))
              && (ComparableDType.nat.lt idx.2.1.val HEAD_ACTIVE)⟩)
        ∧ (vtile = ⟨fun i : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
            if vmaskT.data i then some (sin.readMem (vptrs.data i).1 (vptrs.data i).2)
            else some (sin.undef (vptrs.data i).1 (vptrs.data i).2)⟩)
        ∧ (pf16 = ⟨fun i => FloatDType.real.cast FloatDType.fp16 (pT.data i)⟩)
        ∧ sF.regs .real [BLOCK_M] "m_i" = some mijT
        ∧ sF.regs .real [BLOCK_M] "l_i" = some (Tile.bop NumericDType.real.add
            (Broadcast.consSame Broadcast.nil)
            (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) litile alphaT) lijT)
        ∧ sF.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some (Tile.bop NumericDType.real.add
            (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
            (Tile.dot [] ⟨fun i => FloatDType.fp16.cast FloatDType.real (pf16.data i)⟩ vtile))
        ∧ sF.regs .nat [BLOCK_M] "offs_m" = some offsm
        ∧ sF.regs .nat [BLOCK_N] "offs_n" = some offsn
        ∧ sF.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some qtile
        ∧ sF.regs .real [] "q_scale" = some qsc
        ∧ sF.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some
            (Tile.ptrAdd Broadcast.scalarR kptrs
              (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar BLOCK_N) (Tile.scalar HEAD_DIM)))
        ∧ sF.regs .ptr [] "K_scale_ptr" = some
            (Tile.ptrAdd Broadcast.nil ksptr (Tile.scalar 1))
        ∧ sF.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some
            (Tile.ptrAdd Broadcast.scalarR vptrs
              (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar BLOCK_N) (Tile.scalar HEAD_DIM))) := by
  rw [AftgFoundation.aftgLoopBodyG_eq_head_tail]
  obtain ⟨s1, hHead, hpids1, hmem1, hundef1, kmaskT, ktile, kscT, qkdotT, maskT, qkSentT,
    rmaxT, mijT, qkShiftT, pExpT, pT, hkmaskT, hktile, hkscT, hqkdotT, hmaskT, hqkSentT,
    hrm, hmijT, hqkShiftT, hpExpT, hpT, hs1mi, hs1mij, hs1mask, hs1p, hs1li, hs1acc,
    hs1v, hs1vp, hs1kp, hs1ksp, hs1offsm, hs1offsn, hs1sn, hs1q, hs1qsc⟩ :=
    aftgLoopBodyHeadG_steps N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE hBN sin SN offsm offsn kptrs ksptr mtile qtile qsc
      hsn hoffsm hoffsn hmi hkp hksp hq hqsc
  rw [stepStmts.append_some hHead]
  have hs1li' : s1.regs .real [BLOCK_M] "l_i" = some litile := by rw [hs1li]; exact hli
  have hs1acc' : s1.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some acctile := by rw [hs1acc]; exact hacc
  have hs1vp' : s1.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some vptrs := by rw [hs1vp]; exact hvp
  obtain ⟨sF, hTail, hpidsF, hmemF, hundefF, hFoffsm, hFoffsn, hFq, hFqsc,
    lijT, alphaT, vmaskT, vtile, pf16,
    hlijT, halphaT, hvmaskT, hvtile, hpf16, hFmi, hFli, hFacc, hFkp, hFksp, hFvp⟩ :=
    aftgLoopBodyTailG_steps N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE s1 SN offsn kptrs ksptr vptrs mtile mijT maskT pT litile acctile
      hs1sn hs1offsn hs1mi hs1mij hs1mask hs1p hs1li' hs1acc' hs1vp' hs1kp hs1ksp
  have hpT' : pT = ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      if maskT.data idx then
        (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
          (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data idx
      else (some (0.0 : ℝ) : WithBot ℝ)⟩ := by
    rw [hpT, hpExpT, hqkShiftT]
  have hsinmem : s1.mem = sin.mem := hmem1
  have hsinundef : ∀ rg o, s1.undef rg o = sin.undef rg o := hundef1
  refine ⟨sF, hTail, ?_, ?_, ?_, kmaskT, ktile, kscT, qkdotT, maskT, qkSentT, rmaxT, mijT,
    pT, lijT, alphaT, vmaskT, vtile, pf16,
    hkmaskT, hktile, hkscT, hqkdotT, hmaskT, hqkSentT, hrm, hmijT, hpT', hlijT, halphaT,
    ?_, ?_, hpf16, hFmi, hFli, hFacc, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF, hpids1]
  · rw [hmemF, hmem1]
  · intro rg o; rw [hundefF, hundef1]
  · rw [hvmaskT]
  · rw [hvtile]
    refine congrArg _ ?_
    ext i
    rw [show s1.readMem (vptrs.data i).1 (vptrs.data i).2
          = sin.readMem (vptrs.data i).1 (vptrs.data i).2 from by
      unfold BlockState.readMem; rw [hsinmem],
      hsinundef]
  · rw [hFoffsm, hs1offsm]
  · rw [hFoffsn, hs1offsn]
  · rw [hFq, hs1q]
  · rw [hFqsc, hs1qsc]
  · exact hFkp
  · exact hFksp
  · exact hFvp

/-- General loop body preserves `O_block_ptr`. -/
theorem aftgLoopBody_preserves_OblockPtrG
    {N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat} {s s' : BlockState}
    (h : stepStmts (AftgFoundation.aftgLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) s = some s') :
    s'.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr" = s.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr" := by
  refine stepStmts_regs_preserved (fun st s1 s2 hmem hstep => ?_) h
  fin_cases hmem <;> exact stepStmt_assign_regs_ne (by decide) hstep

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **General step lemma (AFT2, causal, masked).** -/
theorem aftg_attn_stepG
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (hN : N_CTX = BLOCK_N * numKVBlocks)
    (s0 : BlockState) (i : Nat) (s : BlockState) (hilt : i < N_CTX) (himod : i % BLOCK_N = 0)
    (hsb : aftgScoreBoundG
      (qTileAFT2mG s0 Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
      (kTileAFT2G s0 K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
      (vTileAFT2mG s0 V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
      (keyScaleAFT2G s0 QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks) (qStartAFT2G s0 BLOCK_M))
    (hinv : aftgInvariantG Q K V QScale KScale Out s0 stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks
      (keyScaleAFT2G s0 QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks) hBD i s) :
    ∃ s', stepStmts (AftgFoundation.aftgLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ aftgInvariantG Q K V QScale KScale Out s0 stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks
          (keyScaleAFT2G s0 QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks) hBD (i + BLOCK_N) s' := by
  subst hN
  set keyScale := keyScaleAFT2G s0 QScale KScale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N numKVBlocks with hkeyScale
  set qStart := qStartAFT2G s0 BLOCK_M with hqStart
  set qT := qTileAFT2mG s0 Q stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_DMODEL HEAD_ACTIVE with hqT
  set kT := kTileAFT2G s0 K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL with hkT
  set vT := vTileAFT2mG s0 V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE with hvT
  simp only [aftgInvariantG] at hinv
  obtain ⟨hpids, hmod, hile, hmi, hli, hacc, hoffsm, hoffsn,
    hq, hqs, hKp, hKsp, hVp, hOp, hundef, hmem⟩ := hinv
  set c := i / BLOCK_N with hc_def
  have hi : i = c * BLOCK_N := by
    rw [hc_def]; exact (Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero himod)).symm
  have hcnum : c < numKVBlocks := by
    have hlt2 : c * BLOCK_N < BLOCK_N * numKVBlocks := by rw [← hi]; exact hilt
    by_contra hge; push_neg at hge
    have : BLOCK_N * numKVBlocks ≤ BLOCK_N * c := Nat.mul_le_mul_left BLOCK_N hge
    rw [Nat.mul_comm BLOCK_N c] at this; omega
  have hc1 : (c + 1) * BLOCK_N ≤ BLOCK_N * numKVBlocks := by
    calc (c + 1) * BLOCK_N = BLOCK_N * (c + 1) := by ring
      _ ≤ BLOCK_N * numKVBlocks := Nat.mul_le_mul_left BLOCK_N (by omega)
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
    aftgLoopBody_stepsG (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE hBN
      sin i
      (Tile.vec (fun r : Fin BLOCK_M => qStart + r.val)) (Tile.vec (fun j : Fin BLOCK_N => j.val))
      (kPtrsAFT2G s0 K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL c)
      (kScalePtrAFT2G s0 KScale (BLOCK_N * numKVBlocks) BLOCK_N c)
      (vPtrsAFT2G s0 V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL c)
      ⟨fun r : TileIndex [BLOCK_M] => aftgRunningMaxG qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩⟩
      ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        if qStart + idx.1.val < BLOCK_N * numKVBlocks ∧ idx.2.1.val < HEAD_ACTIVE then
          some (qTileAFT2G s0 Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL idx) else some (0.0 : ℝ)⟩
      (Tile.scalar (some (s0.readMem QScale (s0.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_M - 1) / BLOCK_M) + s0.pids 0))))
      ⟨fun r : TileIndex [BLOCK_M] => ((aftgStateBot1G qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩).2.1 : ℝ)⟩
      ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => ((aftgStateBot1G qT kT vT keyScale qStart i idx.1 idx.2.1).2.2 : ℝ)⟩
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
  set qsc : ℝ := s0.readMem QScale (s0.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_M - 1) / BLOCK_M) + s0.pids 0) with hqscv
  set ksc : ℝ := s0.readMem KScale (s0.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + c) with hkscv
  have hkscData : kscT.data PUnit.unit = some ksc := by
    rw [hkscT]; simp only [kScalePtrAFT2G, Region.cast, hkscv]
    refine congrArg some ?_
    unfold BlockState.readMem; rw [hmem']
  have hkeyBlock : ∀ jL : Fin BLOCK_N,
      keyScale (⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩ : Fin (BLOCK_N * numKVBlocks)) = qsc * ksc := by
    intro jL
    have hdiv : (c * BLOCK_N + jL.val) / BLOCK_N = c := by
      rw [Nat.mul_comm c BLOCK_N, Nat.mul_add_div hBN, Nat.div_eq_of_lt jL.isLt, Nat.add_zero]
    simp only [hkeyScale, keyScaleAFT2G, hqscv, hkscv, hdiv]
  have hqcell : ∀ (ir : Fin BLOCK_M) (e : Fin BLOCK_DMODEL),
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        if qStart + idx.1.val < BLOCK_N * numKVBlocks ∧ idx.2.1.val < HEAD_ACTIVE then
          some (qTileAFT2G s0 Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL idx) else some (0.0 : ℝ)⟩
          : Tile .real [BLOCK_M, BLOCK_DMODEL]).data (ir, e, PUnit.unit)
        = some (if e.val < HEAD_ACTIVE then qT (ir, e, PUnit.unit) else 0) := by
    intro ir e
    show (if qStart + ir.val < BLOCK_N * numKVBlocks ∧ e.val < HEAD_ACTIVE then some (qTileAFT2G s0 Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL (ir, e, PUnit.unit)) else some (0.0 : ℝ))
        = some (if e.val < HEAD_ACTIVE then qT (ir, e, PUnit.unit) else 0)
    rw [hqT]; simp only [qTileAFT2mG, hqStart]
    by_cases he : e.val < HEAD_ACTIVE
    · by_cases hb : qStartAFT2G s0 BLOCK_M + ir.val < BLOCK_N * numKVBlocks
      · rw [if_pos ⟨hb, he⟩, if_pos he, if_pos ⟨hb, he⟩]
      · rw [if_neg (fun h => hb h.1), if_pos he, if_neg (fun h => hb h.1)]; norm_num
    · rw [if_neg (fun h => he h.2), if_neg he]; norm_num
  have hkcell : ∀ (e : Fin BLOCK_DMODEL) (jL : Fin BLOCK_N),
      ktile.data (e, jL, PUnit.unit)
        = some (if jL.val < BLOCK_N * numKVBlocks - i ∧ e.val < HEAD_ACTIVE then
            kT (⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩, e, PUnit.unit) else 0) := by
    intro e jL
    rw [hktile]; simp only [hkmaskT, kPtrsAFT2G, Region.cast, Tile.vec]
    rw [show (ComparableDType.nat.lt jL.val (BLOCK_N * numKVBlocks - i) && ComparableDType.nat.lt e.val HEAD_ACTIVE)
          = decide (jL.val < BLOCK_N * numKVBlocks - i ∧ e.val < HEAD_ACTIVE) from by
      rw [Bool.eq_iff_iff]; simp only [Bool.and_eq_true, ComparableDType.nat_lt_eq_true,
        decide_eq_true_eq]]
    by_cases hb : jL.val < BLOCK_N * numKVBlocks - i ∧ e.val < HEAD_ACTIVE
    · rw [if_pos (by simp only [decide_eq_true_eq]; exact hb), if_pos hb]
      refine congrArg some ?_
      rw [hkT, kTileAFT2G]
      rw [show baseOffsetAFT2G s0 stride_qz stride_qh H + e.val + (c * BLOCK_N + jL.val) * HEAD_DIM
            = baseOffsetAFT2G s0 stride_qz stride_qh H + (c * BLOCK_N + jL.val) * HEAD_DIM + e.val from by omega]
      show sin.readMem _ _ = s0.readMem K _
      unfold BlockState.readMem; rw [hmem']
    · rw [if_neg (by simp only [decide_eq_true_eq]; exact hb), if_neg hb]
      refine congrArg some ?_
      exact hundef' _ _
  have hqTmask : ∀ (ir : Fin BLOCK_M) (e : Fin BLOCK_DMODEL), (if e.val < HEAD_ACTIVE then qT (ir, e, PUnit.unit) else 0) = qT (ir, e, PUnit.unit) := by
    intro ir e
    by_cases he : e.val < HEAD_ACTIVE
    · rw [if_pos he]
    · rw [if_neg he, hqT]; simp only [qTileAFT2mG]; rw [if_neg (fun h => he h.2)]
  have hjLwin : ∀ jL : Fin BLOCK_N, jL.val < BLOCK_N * numKVBlocks - i := by
    intro jL; have := jL.isLt
    have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N
    omega
  have hqkcell : ∀ (ir : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) (jL : Fin BLOCK_N),
      qkdotT.data (ir, jL, PUnit.unit)
        = some ((aftgKVG qT kT vT keyScale ir d ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1) := by
    intro ir d jL
    rw [hqkdotT]
    have hsc := aftg_score_cellG BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE qStart (BLOCK_N * numKVBlocks - i) jL ir qsc ksc qT
      (fun e jL => kT (⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩, e, PUnit.unit))
      ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        if qStart + idx.1.val < BLOCK_N * numKVBlocks ∧ idx.2.1.val < HEAD_ACTIVE then
          some (qTileAFT2G s0 Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL idx) else some (0.0 : ℝ)⟩
      ktile kscT (hjLwin jL)
      (fun e => hqcell ir e) (fun e => hkcell e jL) hkscData
    rw [hsc]
    refine congrArg some ?_
    simp only [aftgKVG, hkeyBlock jL]
    rw [show (Finset.univ.sum (fun e : Fin BLOCK_DMODEL => (if e.val < HEAD_ACTIVE then qT (ir, e, PUnit.unit) else 0)
            * kT (⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩, e, PUnit.unit)))
          = Finset.univ.sum (fun e : Fin BLOCK_DMODEL => qT (ir, e, PUnit.unit)
            * kT (⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩, e, PUnit.unit))
        from Finset.sum_congr rfl (fun e _ => by rw [hqTmask ir e])]
    ring
  have hmaskcell : ∀ (ir : Fin BLOCK_M) (jL : Fin BLOCK_N),
      maskT.data (ir, jL, PUnit.unit) = decide (qStart + ir.val ≥ c * BLOCK_N + jL.val) := by
    intro ir jL
    rw [hmaskT]; simp only [Tile.vec]
    rw [show (ComparableDType.nat.ge (qStart + ir.val) (i + jL.val))
          = decide (qStart + ir.val ≥ c * BLOCK_N + jL.val) from by
      rw [Bool.eq_iff_iff]; simp only [ComparableDType.nat_ge_eq_true, decide_eq_true_eq]
      omega]
  have hsentcell : ∀ (ir : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) (jL : Fin BLOCK_N),
      qkSentT.data (ir, jL, PUnit.unit)
        = if qStart + ir.val ≥ c * BLOCK_N + jL.val then
            (((aftgKVG qT kT vT keyScale ir d ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ) := by
    intro ir d jL
    rw [hqkSentT]; simp only [hmaskcell ir jL, decide_eq_true_eq]
    by_cases hk : qStart + ir.val ≥ c * BLOCK_N + jL.val
    · rw [if_pos hk, if_pos hk]
      exact hqkcell ir d jL
    · rw [if_neg hk, if_neg hk]
      exact aftg_sentinel_eq
  simp only [aftgInvariantG, ← hqStart, ← hqT, ← hkT, ← hvT]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF, hpids']
  · rw [Nat.add_mod_right]; exact himod
  · have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N
    rw [hi]; omega
  · rw [hFmi]; refine congrArg some ?_; refine Tile.ext (fun r => ?_)
    obtain ⟨ir, ⟨⟩⟩ := r
    have hbr := aftg_mij_reg_eq_maskedG qT kT vT keyScale BLOCK_N hBN hBM hBD qStart c hc1 ir hsb
      qkSentT ⟨fun r : TileIndex [BLOCK_M] =>
        aftgRunningMaxG qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩⟩ rmaxT mijT
      (fun jL => by
        rw [show (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (aftgAx1G BLOCK_M BLOCK_N) (ir, PUnit.unit) jL)
              = (ir, jL, PUnit.unit) from rfl]
        rw [hsentcell ir ⟨0, hBD⟩ jL])
      hrm (by simp only [hi]) hmijT
    rw [hbr, show ((c + 1) * BLOCK_N : Nat) = i + BLOCK_N from by have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega]
  · rw [hFli]; refine congrArg some ?_; refine Tile.ext (fun r => ?_)
    obtain ⟨ir, ⟨⟩⟩ := r
    have hmijcell : mijT.data (ir, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) ir ⟨0, hBD⟩ := by
      refine aftg_mij_reg_eq_maskedG qT kT vT keyScale BLOCK_N hBN hBM hBD qStart c hc1 ir hsb
        qkSentT ⟨fun r : TileIndex [BLOCK_M] =>
          aftgRunningMaxG qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩⟩ rmaxT mijT
        (fun jL => by
          rw [show (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (aftgAx1G BLOCK_M BLOCK_N) (ir, PUnit.unit) jL)
                = (ir, jL, PUnit.unit) from rfl]
          rw [hsentcell ir ⟨0, hBD⟩ jL])
        hrm (by simp only [hi]) hmijT
    have hbr := aftg_denom_reg_eq_maskedG qT kT vT keyScale BLOCK_N hBN hBM hBD qStart c hc1 ir hsb
      qkSentT ⟨fun r : TileIndex [BLOCK_M] => aftgRunningMaxG qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩⟩
      mijT alphaT ⟨fun r : TileIndex [BLOCK_M] => ((aftgStateBot1G qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩).2.1 : ℝ)⟩
      lijT pT
      (fun jL => by rw [hsentcell ir ⟨0, hBD⟩ jL])
      (by simp only [hi]; rfl) (by simp only [hi]) hmijcell
      halphaT hlijT
      (fun jL => by
        rw [hpT]; simp only [hmaskcell ir jL, decide_eq_true_eq])
    have hne : aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) ir ⟨0, hBD⟩ ≠ ⊥ :=
      aftgRunningMaxG_ne_bot qT kT vT keyScale qStart ((c + 1) * BLOCK_N) (by have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega) (by have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega) ir ⟨0, hBD⟩
    show (Tile.bop NumericDType.real.add _ _ _).data _ = _
    rw [hbr, show ((i + BLOCK_N) : Nat) = (c + 1) * BLOCK_N from by have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega]
    exact congrArg (fun st : WithBot ℝ × ℝ × ℝ => (st.2.1 : WithBot ℝ))
      (aftgStateBot1G_eq_aftgStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) ir ⟨0, hBD⟩ hne).symm
  · rw [hFacc, hpf16]
    simp only [FloatDType.cast, FloatDType.real_toWithBot, FloatDType.fp16_ofWithBot,
      FloatDType.fp16_toWithBot, FloatDType.real_ofWithBot]
    refine congrArg some ?_; refine Tile.ext (fun idx => ?_)
    obtain ⟨ir, id, ⟨⟩⟩ := idx
    have hmijcell : mijT.data (ir, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) ir ⟨0, hBD⟩ := by
      refine aftg_mij_reg_eq_maskedG qT kT vT keyScale BLOCK_N hBN hBM hBD qStart c hc1 ir hsb
        qkSentT ⟨fun r : TileIndex [BLOCK_M] =>
          aftgRunningMaxG qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩⟩ rmaxT mijT
        (fun jL => by
          rw [show (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (aftgAx1G BLOCK_M BLOCK_N) (ir, PUnit.unit) jL)
                = (ir, jL, PUnit.unit) from rfl]
          rw [hsentcell ir ⟨0, hBD⟩ jL])
        hrm (by simp only [hi]) hmijT
    have hvload : ∀ jL : Fin BLOCK_N,
        vtile.data (jL, id, PUnit.unit)
          = some ((aftgKVG qT kT vT keyScale ir id ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).2) := by
      intro jL
      rw [hvtile]; simp only [hvmaskT, Tile.vec, vPtrsAFT2G, Region.cast]
      rw [show (ComparableDType.nat.lt jL.val (BLOCK_N * numKVBlocks - i) && ComparableDType.nat.lt id.val HEAD_ACTIVE)
            = decide (jL.val < BLOCK_N * numKVBlocks - i ∧ id.val < HEAD_ACTIVE) from by
        rw [Bool.eq_iff_iff]; simp only [Bool.and_eq_true, ComparableDType.nat_lt_eq_true,
          decide_eq_true_eq]]
      simp only [aftgKVG, hvT, vTileAFT2mG]
      by_cases hid : id.val < HEAD_ACTIVE
      · rw [if_pos (by simp only [decide_eq_true_eq]; exact ⟨hjLwin jL, hid⟩), if_pos hid]
        refine congrArg some ?_
        show sin.readMem _ (baseOffsetAFT2G s0 stride_qz stride_qh H + (c * BLOCK_N + jL.val) * HEAD_DIM + id.val) = _
        rw [vTileAFT2G]
        unfold BlockState.readMem; rw [hmem']
      · rw [if_neg (by simp only [decide_eq_true_eq, not_and]; intro _ h; exact hid h), if_neg hid]
        exact congrArg some (hundef' _ _)
    have hbr := aftg_acc_reg_eq_maskedG qT kT vT keyScale BLOCK_N hBN hBM hBD qStart c hc1 ir id hsb
      qkSentT ⟨fun r : TileIndex [BLOCK_M] => aftgRunningMaxG qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩⟩
      mijT alphaT
      ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => ((aftgStateBot1G qT kT vT keyScale qStart i idx.1 idx.2.1).2.2 : ℝ)⟩
      (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => ((aftgStateBot1G qT kT vT keyScale qStart i idx.1 idx.2.1).2.2 : ℝ)⟩
        (Tile.expandDim ⟨1, by simp⟩ alphaT))
      pT vtile (fun jL => (aftgKVG qT kT vT keyScale ir id ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).2)
      (fun jL => by rw [hsentcell ir ⟨0, hBD⟩ jL])
      (by simp only [hi]; rfl) (by simp only [hi]) hmijcell halphaT rfl
      (fun jL => by
        rw [hpT]; simp only [hmaskcell ir jL, decide_eq_true_eq])
      hvload (fun jL => rfl)
    have hne : aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) ir id ≠ ⊥ := by
      rw [aftgRunningMaxG_eq qT kT vT keyScale qStart ((c + 1) * BLOCK_N) ir id ⟨0, hBD⟩]
      exact aftgRunningMaxG_ne_bot qT kT vT keyScale qStart ((c + 1) * BLOCK_N) (by have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega) (by have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega) ir ⟨0, hBD⟩
    show (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) _
      (Tile.dot [] pT vtile)).data _ = _
    rw [hbr, show ((i + BLOCK_N) : Nat) = (c + 1) * BLOCK_N from by have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega]
    exact congrArg (fun st : WithBot ℝ × ℝ × ℝ => (st.2.2 : WithBot ℝ))
      (aftgStateBot1G_eq_aftgStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) ir id hne).symm
  · rw [hFoffsm]
  · rw [hFoffsn]
  · rw [hFq]
  · rw [hFqsc]
  · rw [hFKp, kPtrsAFT2G_succ, show (i + BLOCK_N) / BLOCK_N = c + 1 from by rw [hi, Nat.add_div_right _ hBN, Nat.mul_div_cancel _ hBN]]
  · rw [hFKsp, kScalePtrAFT2G_succ, show (i + BLOCK_N) / BLOCK_N = c + 1 from by rw [hi, Nat.add_div_right _ hBN, Nat.mul_div_cancel _ hBN]]
  · rw [hFVp, vPtrsAFT2G_succ, show (i + BLOCK_N) / BLOCK_N = c + 1 from by rw [hi, Nat.add_div_right _ hBN, Nat.mul_div_cancel _ hBN]]
  · rw [aftgLoopBody_preserves_OblockPtrG hchain, hsin,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hOp
  · intro rg o; rw [hundefF, hundef']
  · rw [hmemF, hmem']

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General postLoop evaluation (AFT2, causal).** From a loop-end state satisfying
`aftgInvariantG … N_CTX`, the 2 postLoop statements write the genuine closed form
`attnFwdTritonOutSpecG` to `Out` at every active output lane and preserve `Out` on
inactive lanes. -/
theorem aftgPostLoopG_eval
    (Q K V QScale KScale Out : RegionName) (s0 : BlockState) (s : BlockState)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (hBD : 0 < BLOCK_DMODEL) (hN : N_CTX = BLOCK_N * numKVBlocks) (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (keyScale : Fin (BLOCK_N * numKVBlocks) → ℝ)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s0 H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx))
    (hinv : aftgInvariantG Q K V QScale KScale Out s0 stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks keyScale hBD N_CTX s) :
    ∃ sP, stepStmts (AftgFoundation.aftgPostLoopG Out N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE) s = some sP
      ∧ ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
          sP.readMem Out (outOffset s0 H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx)
            = if active s0 N_CTX HEAD_ACTIVE BLOCK_M idx then
                attnFwdTritonOutSpecG s0 Q K V stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks keyScale idx
              else s.readMem Out (outOffset s0 H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx) := by
  have hSEQ : 0 < BLOCK_N * numKVBlocks := Nat.mul_pos hBN hnum
  have hNpos : 0 < N_CTX := by rw [hN]; exact hSEQ
  simp only [aftgInvariantG] at hinv
  obtain ⟨hpids, _, _, _hmi, hli, hacc, hoffsm, _hoffsn,
    _hq, _hqs, _hKp, _hKsp, _hVp, hOp, hundef, hmem⟩ := hinv
  set qStart := qStartAFT2G s0 BLOCK_M with hqStart
  set qT := qTileAFT2mG s0 Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE with hqT
  set kT := kTileAFT2G s0 K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL with hkT
  set vT := vTileAFT2mG s0 V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE with hvT
  set liTile : Tile .real [BLOCK_M] :=
    ⟨fun r : TileIndex [BLOCK_M] => ((aftgStateBot1G qT kT vT keyScale qStart N_CTX r.1 ⟨0, hBD⟩).2.1 : ℝ)⟩
    with hliTile
  set accTile : Tile .real [BLOCK_M, BLOCK_DMODEL] :=
    ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => ((aftgStateBot1G qT kT vT keyScale qStart N_CTX idx.1 idx.2.1).2.2 : ℝ)⟩
    with haccTile
  set accFin : Tile .real [BLOCK_M, BLOCK_DMODEL] :=
    Tile.bop NumericDType.real.div (Broadcast.consSame (Broadcast.consR Broadcast.nil)) accTile
      (Tile.expandDim ⟨1, by simp⟩ liTile) with haccFin
  unfold AftgFoundation.aftgPostLoopG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "l_i"))) s
        = some accFin from by
      have hexp : @evalOp TileDType.real [BLOCK_M, 1]
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.real [BLOCK_M] "l_i")) s
          = some (Tile.expandDim ⟨1, by simp⟩ liTile) :=
        evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hli
      rw [evalOp_div]
      simp only [evalOp_ref, hexp, hacc, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  set s2 := s.setReg "acc" .real [BLOCK_M, BLOCK_DMODEL] accFin with hs2
  have hOp2 : s2.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr" = some (oBlockPtrAFT2G s0 Out stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL) := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hOp
  have hacc2 : s2.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some accFin := by
    rw [hs2, BlockState.setReg_same]
  have hoffsm2 : s2.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => qStart + r.val)) := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffsm
  set oOffFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx => baseOffsetAFT2G s0 stride_qz stride_qh H + (s0.pids 0 * BLOCK_M + idx.1.val) * HEAD_DIM + idx.2.1.val with hoOffFn
  set P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx => qStart + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE with hP
  have hopEval : evalOp (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr") s2
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out.cast, oOffFn idx)⟩ : Tile .ptr [BLOCK_M, BLOCK_DMODEL]) := by
    rw [evalOp_ref, hOp2]
    refine congrArg some ?_; ext idx
    · rfl
    · simp only [oBlockPtrAFT2G, hoOffFn]
  have hmaskEval : evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat N_CTX))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))) s2
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => decide (P idx)⟩ : Tile .bool [BLOCK_M, BLOCK_DMODEL]) := by
    rw [aftg_evalOp_boolAnd, evalOp_lt]
    erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hoffsm2, evalOp_expandDim]
    simp only [evalOp_lt, evalOp_arange, evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
    refine congrArg some ?_; ext idx
    simp only [Tile.bop_data, Tile.cop_data, Tile.expandDim_data, Tile.vec_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
      Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil, TileShape.dropInsertedIndex]
    rw [show (ComparableDType.nat.lt (qStart + idx.1.val) N_CTX && ComparableDType.nat.lt idx.2.1.val HEAD_ACTIVE)
          = decide (P idx) from by
      rw [Bool.eq_iff_iff]; simp only [hP, Bool.and_eq_true, ComparableDType.nat_lt_eq_true,
        decide_eq_true_eq]]
  have hstore : stepStmt (Stmt.store .real [BLOCK_M, BLOCK_DMODEL] (.ptr (.ref .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"))
      (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
      (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat N_CTX))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))))) s2
      = some ((TileShape.allIndices [BLOCK_M, BLOCK_DMODEL]).foldl
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
    have : oOffFn = (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => outOffset s0 H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx) := by
      funext idx
      simp only [hoOffFn, outOffset, offZ, offH, mIndex, kIndex, baseOffsetAFT2G, Nat.mul_one]
    rw [this]; exact hOutInj
  have houtOff : outOffset s0 H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx = oOffFn idx := by
    simp only [outOffset, offZ, offH, mIndex, kIndex, hoOffFn, baseOffsetAFT2G, Nat.mul_one]
  rw [houtOff]
  rw [BlockState.scatter_readback_prop_masked_nd _ oOffFn
    (fun idx => (accFin.data idx).unbotD 0) P hinjO idx]
  have hactiveP : active s0 N_CTX HEAD_ACTIVE BLOCK_M idx ↔ P idx := by
    simp only [active, mIndex, kIndex, hP, hqStart, qStartAFT2G]
  by_cases hk : P idx
  · rw [if_pos hk, if_pos (hactiveP.mpr hk)]
    obtain ⟨ir, id, ⟨⟩⟩ := idx
    subst hN
    have hne : aftgRunningMaxG qT kT vT keyScale qStart (BLOCK_N * numKVBlocks) ir id ≠ ⊥ :=
      aftgRunningMaxG_ne_bot qT kT vT keyScale qStart (BLOCK_N * numKVBlocks) hSEQ hSEQ ir id
    have hne' : aftgRunningMaxG qT kT vT keyScale qStart (BLOCK_N * numKVBlocks) ir ⟨0, hBD⟩ ≠ ⊥ :=
      aftgRunningMaxG_ne_bot qT kT vT keyScale qStart (BLOCK_N * numKVBlocks) hSEQ hSEQ ir ⟨0, hBD⟩
    simp only [haccFin, Tile.bop_data, Tile.expandDim_data, TileShape.dropInsertedIndex,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div, WithBot.realDiv,
      Option.map₂, Option.bind, Option.map, haccTile, hliTile, WithBot.unbotD_coe]
    rw [aftgStateBot1G_eq_aftgStateBotG qT kT vT keyScale qStart (BLOCK_N * numKVBlocks) ir id hne,
      aftgStateBot1G_eq_aftgStateBotG qT kT vT keyScale qStart (BLOCK_N * numKVBlocks) ir ⟨0, hBD⟩ hne']
    rw [aftgStateBotG_snd_fst_indep qT kT vT keyScale qStart (BLOCK_N * numKVBlocks) ir ⟨0, hBD⟩ id]
    have hbridge := aftgStateBotG_full_eq_spec s0 Q K V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks keyScale ir id
      (by rw [← hqStart, ← hqT, ← hkT, ← hvT]; exact hne)
    rw [← hbridge]
    simp only [hqStart, hqT, hkT, hvT, WithBot.unbotD_some]
  · rw [if_neg hk, if_neg (fun h => hk (hactiveP.mp h)), hs2]
    simp only [BlockState.readMem, BlockState.setReg_mem]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **General full kernel execution (AFT2, causal).** The lowered general AFT2 surface
body (contiguous symbolic layout) steps a clean, score-bounded state through
preLoop + the `forRange` streaming loop (`forRange_inv` with `aftgInvariantG`,
advanced by `aftg_attn_stepG`) + postLoop, leaving the `Out` writeback at every
active lane equal to the genuine causal closed form `attnFwdTritonOutSpecG`. -/
theorem aftg_exec_generalG
    (Q K V QScale KScale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE Z numKVBlocks : Nat)
    (hBD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M)
    (hN : N_CTX = BLOCK_N * numKVBlocks) (hnum : 0 < numKVBlocks)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx))
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hsb : aftgScoreBoundG
      (qTileAFT2mG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
      (kTileAFT2G s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
      (vTileAFT2mG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
      (keyScaleAFT2G s QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks) (qStartAFT2G s BLOCK_M)) :
    ∃ sF, stepStmts (attn_fwd_triton_surface Q K V QScale KScale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.body s = some sF
      ∧ ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
          active s N_CTX HEAD_ACTIVE BLOCK_M idx →
            sF.readMem Out (outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx)
              = attnFwdTritonOutSpecG s Q K V stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks (keyScaleAFT2G s QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks) idx := by
  set keyScale := keyScaleAFT2G s QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks with hkeyScale
  rw [aftg_body_splitG, aftgPreLoopG_check, aftgPostLoopG_check]
  obtain ⟨sp, hpre, hinv0⟩ :=
    aftgPreLoopG_invariant s Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks hN hBN hBD keyScale hundef
  rw [stepStmts.append_some hpre]
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    forRange_inv (idx := "start_n") (start := 0) (stop := N_CTX) (step := BLOCK_N)
      (body := AftgFoundation.aftgLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE)
      (P := fun i st => aftgInvariantG Q K V QScale KScale Out s stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks keyScale hBD i st)
      (s_init := sp)
      (by omega)
      hinv0
      (fun i st hi hP =>
        aftg_attn_stepG Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks hBN hBM hBD hN s i st hi
          (by simp only [aftgInvariantG] at hP; exact hP.2.1) hsb hP)
  rw [stepStmts.cons_some hloop]
  have hfinal : final = N_CTX := by
    obtain ⟨_, hmod, hle, _⟩ := hinvL
    omega
  rw [hfinal] at hinvL
  obtain ⟨sF, hpost, hO⟩ :=
    aftgPostLoopG_eval Q K V QScale KScale Out s sL stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks hBD hN hBN hnum keyScale hOutInj hinvL
  refine ⟨sF, hpost, ?_⟩
  intro idx hact
  rw [hO idx, if_pos hact]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Dimension-general genuine causal closed-form output summary for `attn_fwd_triton`.**

Removes the Python test-shape pin (`Z = 2`, `H = 4`,
`N_CTX = HEAD_DIM = BLOCK_M = BLOCK_DMODEL = 128`, `BLOCK_N = 64`,
`HEAD_ACTIVE = 96`): for arbitrary symbolic block/head dimensions at the
contiguous layout (`stride_qm = HEAD_DIM`, `stride_qk = 1`, `stride_kn = HEAD_DIM`),
the full causal surface lowers to the algorithm layer and its masked `Out`
writeback realizes the genuine closed-form causal attention
`attnFwdTritonOutSpecG` (= `attentionRealBase2PerKeyScalePred … (causalKeep)` over
INPUT memory) at every active output lane. Side conditions: positive blocks,
`N_CTX = BLOCK_N · numKVBlocks` (`numKVBlocks > 0`), clean input (`undef = 0`), the
`-1e6` sentinel score bound `aftgScoreBoundG`, and output-offset injectivity. -/
theorem attn_fwd_triton_output_summary_general
    (Q K V QScale KScale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE Z numKVBlocks : Nat)
    (hBD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M)
    (hN : N_CTX = BLOCK_N * numKVBlocks) (hnum : 0 < numKVBlocks)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx))
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hsb : aftgScoreBoundG
      (qTileAFT2mG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
      (kTileAFT2G s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
      (vTileAFT2mG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
      (keyScaleAFT2G s QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks) (qStartAFT2G s BLOCK_M)) :
    (∃ alg, (attn_fwd_triton_surface Q K V QScale KScale Out
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attn_fwd_triton_surface Q K V QScale KScale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => active s N_CTX HEAD_ACTIVE BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        attnFwdTritonOutSpecG s Q K V stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks (keyScaleAFT2G s QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks) idx) := by
  refine ⟨?_, ?_⟩
  · exact attn_fwd_triton_surface_toAlgorithm_supported Q K V QScale KScale Out
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
  · rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [attn_fwd_triton_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro idx hActive
    obtain ⟨sF, hstep, hO⟩ := aftg_exec_generalG Q K V QScale KScale Out s
      stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE Z numKVBlocks
      hBD hBN hBM hN hnum hOutInj hundef hsb
    rw [exec] at hExec
    rw [hstep] at hExec
    obtain rfl : sF = s' := Option.some.inj hExec
    exact hO idx hActive

section TestShape

/-- **Python test-shape summary for `attn_fwd_triton.py` (genuine closed form).**

The Python wrapper fixes `STAGE = 3`; this summary combines the supported full
surface with the observable `Out` writes, stated against the **genuine closed-form
attention** `attnFwdTritonOutSpec` (masked base-2 per-key-scale causal softmax with
`keyScale = aftKeyScale = q_scale · k_scale`). The two genuine side conditions are
explicit: `aftScoreBound` (the kernel's `-1e6` `tl.where` mask sentinel never raises
a row max above the true running max, i.e. every *kept* causal score is `≥ -1e6`)
and the clean `undef ≡ 0` initial state.

This is now a **corollary** of the dimension-general theorem
`attn_fwd_triton_output_summary_general`, instantiated at the Python test shape
(`Z = 2`, `H = 4`, `N_CTX = HEAD_DIM = BLOCK_M = BLOCK_DMODEL = 128`, `BLOCK_N = 64`,
`numKVBlocks = 2`, `HEAD_ACTIVE = 96`, `STAGE = 3`, contiguous strides
`(65536, 16384, 128, 1)`). At those concrete args the general masked tiles, score
carrier, and closed-form spec all reduce defeq to the pinned
`attnFwdTritonOutSpec`/`aftKeyScale` layer, and the pinned causal-only `aftScoreBound`
is exactly the (now causal-only) general `aftgScoreBoundG`. -/
theorem attn_fwd_triton_python_test_shape_output_summary
    (Q K V QScale KScale Out : RegionName) (s : BlockState)
    (hsb : aftScoreBound s Q K (aftKeyScale s QScale KScale))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (attn_fwd_triton_surface Q K V QScale KScale Out
      65536 16384 128 1
      65536 16384 128 1
      65536 16384 128 1
      65536 16384 128 1
      2 4 128 128 128 64 128 96 3).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attn_fwd_triton_surface Q K V QScale KScale Out
        65536 16384 128 1
        65536 16384 128 1
        65536 16384 128 1
        65536 16384 128 1
        2 4 128 128 128 64 128 96 3)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] => active s 128 96 128 idx)
        (fun idx : TileIndex [128, 128] => (Out,
          outOffset s 4 65536 16384 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        attnFwdTritonOutSpec s Q K V (aftKeyScale s QScale KScale) idx) := by
  -- output-offset injectivity at the contiguous test-shape strides
  have hOutInj : Function.Injective
      (fun idx : TileIndex [128, 128] =>
        outOffset s 4 65536 16384 128 1 128 idx) := by
    rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
    simp only [outOffset, offZ, offH, mIndex, kIndex, Nat.mul_one] at h
    have hm : ma = mb := by omega
    have hd : da = db := by omega
    subst mb; subst db; rfl
  -- pinned causal-only `aftScoreBound` ⟹ (now causal-only) general `aftgScoreBoundG`
  have hsbG : aftgScoreBoundG
      (qTileAFT2mG s Q 65536 16384 4 128 128 128 128 96)
      (kTileAFT2G s K 65536 16384 4 128 (64 * 2) 128)
      (vTileAFT2mG s V 65536 16384 4 128 (64 * 2) 128 96)
      (keyScaleAFT2G s QScale KScale 128 128 64 2) (qStartAFT2G s 128) := by
    intro j i hjle
    have h := hsb i j hjle
    rw [aftBlockScore] at h
    -- pinned tiles/keyScale are defeq to the general ones at the concrete test-shape
    -- args; only the `0.0`/`0` literal differs, closed by `norm_num`
    refine le_trans (le_of_eq (by norm_num)) h
  -- instantiate the dimension-general theorem at the Python test shape
  have hgen := attn_fwd_triton_output_summary_general Q K V QScale KScale Out s
    65536 16384 4 128 128 128 64 128 96 3 2 2
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    hOutInj hundef hsbG
  refine ⟨hgen.1, ?_⟩
  -- the general `expected` (attnFwdTritonOutSpecG …) reduces defeq to the pinned
  -- `attnFwdTritonOutSpec` at the concrete test-shape args
  convert hgen.2 using 2 with idx

end TestShape

end General


end VeriTile.Bench.TritonBenchG.AttnFwdTriton
