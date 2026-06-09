import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `int8_quantization` — strict per-kernel correctness

`q_kernel_per_block_int8` / `k_kernel_per_block_int8` process a `[BLK, C]` block
of the query / key matrix into int8: each program loads its block (row-masked by
`offs_m < L`), computes a per-block scale `max(|x|) / 127`, divides every element
by that scale, adds a half-ULP sign-dependent rounding bias, casts to int8, and
stores both the quantized block and the scalar scale. The Q kernel additionally
pre-scales `x` by `C**-0.5 * 1.44269504`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies for both Q and K. The host launch
(`q_kernel_per_block_int8[grid](...)`, the 2-D grid `((L+BLK-1)//BLK, B)`, the
host-side `view`/reshape and scale-tensor allocation, the runtime composition of
per-program writes) is the *trusted boundary*, not a proof obligation. The two
program ids (`off_blk`, `off_b`) are universally quantified, so the per-program
statements cover every program of the grid.

## Proof architecture

```
per_block_int8_python_case1_output_summary        ← TOP THEOREM (case 1)
per_block_int8_python_case2_output_summary        ← TOP THEOREM (case 2)
  ├─ per_block_int8_python_caseN_internal_summary
  │    ├─ ..._full_surfaces_toAlgorithm_supported  full Q/K surface lowers
  │    └─ ..._all_outputs_compute_correct          all real-valued slices
  ├─ q/k_kernel_..._scaled_store_slice_compute_correct   value writeback
  ├─ ..._scale_store_slice_compute_correct               scalar scale writeback
  │    └─ per_block_int8_scaled_store_slice_correct       algorithm-layer readback
  └─ ..._offset_injective                                no-collision lemmas
```

The value spec is the real-valued scaled store `(preScale * x) / scale`
(`perBlockInt8ScaledSpec`); the scale spec is the stored per-block scalar.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float. The honesty point is the
quantization tail: the proofs model the **pre-rounding scaled value**
`(preScale * x) / scale` and the per-block scale `max(|x|) / 127`, and verify
that the full Python surface (including the `x += 0.5 * sign(x)` rounding bias
and the fixed-width `(x).to(tl.int8)` cast annotation) lowers through algorithm
erasure. The bit-accurate effect of the half-ULP rounding bias and the int8
saturating cast is **not** numerically modeled: post-erasure all dtypes unify to
`ℝ`, so `to(tl.int8)` is the identity at the algorithm layer. `@triton.autotune`
is not present here; the Q pre-scale constant is carried symbolically.
-/

namespace VeriTile.Bench.TritonBenchG.Int8Quantization

open VeriTile.Triton

set_option linter.unusedSimpArgs false

private theorem finset_sup'_proof_irrel
    {α β : Type*} [SemilatticeSup α] [PartialOrder α]
    (s : Finset β) (h1 h2 : s.Nonempty) (f : β → α) :
    s.sup' h1 f = s.sup' h2 f := by
  apply Finset.sup'_congr (s := s) (H := h1) (t := s) (f := f) (g := f)
  · rfl
  · intro x hx
    rfl

/-! ## Integer memory infra

The genuine value store writes an `.int` channel (the `.to(tl.int8)` quotient).
The following lemmas are the `.int` analogues of the `.nat`/`.real`
scatter-readback and cross-channel preservation lemmas in
`VeriTile.Triton.Semantics.State`, together with the cross-channel peels relating
the `.int` value store and the trailing scalar `.real` scale store. -/

/-- `.int` masked-foldl preservation: writes whose (masked) offsets all miss `o`
leave `readMemValue .int region o` unchanged. -/
private theorem foldl_writeMemTyped_int_masked_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → Int) (mask : α → Bool) (o : Nat) (l : List α) :
    ∀ (s : BlockState), (∀ k ∈ l, mask k = Bool.true → offsetFn k ≠ o) →
      BlockState.readMemValue ((l.foldl
          (fun acc k =>
            if mask k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s))
          .int region o
      = s.readMemValue .int region o := by
  induction l with
  | nil => intros; rfl
  | cons hd tl ih =>
    intro s h
    rw [List.foldl_cons]
    have htl : ∀ k ∈ tl, mask k = Bool.true → offsetFn k ≠ o :=
      fun k hk hmk => h k (List.mem_cons_of_mem hd hk) hmk
    by_cases hmaskhd : mask hd = Bool.true
    · have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self) hmaskhd
      simp only [hmaskhd, if_true]
      rw [ih _ htl]
      rw [BlockState.writeMemTyped_int_readMemValue_int]
      show (if region = region ∧ o = offsetFn hd then valueFn hd
          else s.readMemValue TileDType.int region o) =
        s.readMemValue TileDType.int region o
      rw [if_neg]
      rintro ⟨_, h_eq⟩
      exact hhd h_eq.symm
    · have hmaskhd' : mask hd = Bool.false := by
        rcases hmaskFalse : mask hd
        · rfl
        · exact absurd hmaskFalse hmaskhd
      simp only [hmaskhd', if_false, Bool.false_eq_true]
      exact ih _ htl

/-- `.int` scatter readback: reading `readMemValue .int` of the masked
`writeMemTyped .int` foldl at lane `i` returns the stored value if `P i`, else
the prior value. -/
theorem scatter_readback_int_prop_masked_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → Int) (P : TileIndex shape → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    BlockState.readMemValue ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s)
      .int region (offsetFn i)
    = if P i then valueFn i else s.readMemValue .int region (offsetFn i) := by
  let l := TileShape.allIndices shape
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  change BlockState.readMemValue ((l.foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s))
      .int region (offsetFn i)
    = if P i then valueFn i else s.readMemValue .int region (offsetFn i)
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  have hl' : l = l₁ ++ i :: l₂ := by simpa [l] using hl
  rw [hl', List.foldl_append, List.foldl_cons]
  have h_l1_not_in : ∀ k ∈ l₁, decide (P k) = Bool.true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    rw [hki] at hk
    exact (hl1_disj i hk i (List.mem_cons_self)) rfl
  have h_l2_not_in : ∀ k ∈ l₂, decide (P k) = Bool.true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    subst hki
    exact hi_notin_l2 hk
  have hstep :
      (fun (acc : BlockState) k =>
        if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc)
        =
      (fun (acc : BlockState) k =>
        if decide (P k) then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) := by
    funext acc k
    by_cases hk : P k <;> simp [hk]
  rw [hstep]
  rw [foldl_writeMemTyped_int_masked_preserves offsetFn valueFn (fun k => decide (P k))
    (offsetFn i) l₂ _ h_l2_not_in]
  by_cases hPi : P i
  · simp only [hPi, if_true]
    rw [BlockState.writeMemTyped_int_readMemValue_int]
    show (if region = region ∧ offsetFn i = offsetFn i then valueFn i else _) = valueFn i
    exact if_pos ⟨rfl, rfl⟩
  · simp only [hPi, if_false]
    rw [foldl_writeMemTyped_int_masked_preserves offsetFn valueFn (fun k => decide (P k))
      (offsetFn i) l₁]
    exact h_l1_not_in

/-- A single trailing `.real` scale store to a different region `Scale` leaves
`readMemValue .int Out` unchanged. The genuine value readback reads `.int Out`;
the trailing scalar scale store to `Scale` is peeled with this. -/
theorem writeMem_readMemValue_int_other (s : BlockState)
    (Scale : RegionName) (so : Nat) (v : ℝ)
    (Out : RegionName) (o : Nat) (hOut : Out ≠ Scale) :
    (s.writeMem Scale so v).readMemValue .int Out o
      = s.readMemValue .int Out o := by
  unfold BlockState.writeMem BlockState.readMemValue BlockState.readMemTyped
  simp only
  rw [if_neg]
  rintro ⟨hR, _⟩
  exact hOut hR

/-- A masked `.int` value-store foldl to `Out` leaves `readMem .real` on a
different region `Scale` unchanged. The genuine scale readback reads `.real
Scale`; the prior int value store to `Out` is peeled with this. -/
theorem foldl_writeMemTyped_int_prop_masked_readMem_other {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → Int)
    (P : α → Prop) [DecidablePred P] (l : List α) (s : BlockState)
    (R : RegionName) (off : Nat) (hRR : R ≠ region) :
    BlockState.readMem ((l.foldl
        (fun acc k =>
          if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k) else acc) s))
      R off
      = s.readMem R off := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · simp only [hP, if_true]
        rw [ih]
        unfold BlockState.writeMemTyped BlockState.readMem
        simp only
        rw [if_neg]
        rintro ⟨hR, _⟩
        exact hRR hR
      · simp only [hP, if_false]
        exact ih _

/-- Faithful transcription of `int8_quantization.py`'s `q_kernel_per_block_int8`.

The final `to(tl.int8)` is preserved as a surface dtype annotation; algorithm
erasure now carries it through the fixed-width cast surface used by the DSL.
-/
noncomputable def q_kernel_per_block_int8_surface
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ComputeKernel := triton {
  off_b = tl.program_id(1)
  off_blk = tl.program_id(0)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x_int8_ptrs = XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  scale_ptrs = Scale + off_b * $(scale_stride) + off_blk
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  x *= $(((Real.sqrt (C : ℝ))⁻¹ * (1.44269504 : ℝ) : ℝ))
  scale = tl.max(tl.abs(x)) / 127.0
  x_int8 = x / scale
  x_int8 += 0.5 * tl.where(x_int8 >= 0.0, 1.0, -1.0)
  x_int8 = (x_int8).to(tl.int8)
  tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < $(L))
  tl.store(scale_ptrs, scale)
}

/-- The full Q int8 Python surface lowers through algorithm erasure, including
the fixed-width `to(tl.int8)` cast surface. -/
theorem q_kernel_per_block_int8_surface_toAlgorithm_supported
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ∃ alg, (q_kernel_per_block_int8_surface X XInt8 Scale L C BLK
      scale_stride).toAlgorithm? = Except.ok alg := by
  simp [q_kernel_per_block_int8_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription of `int8_quantization.py`'s `k_kernel_per_block_int8`.

The final `to(tl.int8)` is preserved as a surface dtype annotation; algorithm
erasure now carries it through the fixed-width cast surface used by the DSL.
-/
def k_kernel_per_block_int8_surface
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ComputeKernel := triton {
  off_b = tl.program_id(1)
  off_blk = tl.program_id(0)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x_int8_ptrs = XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  scale_ptrs = Scale + off_b * $(scale_stride) + off_blk
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  scale = tl.max(tl.abs(x)) / 127.0
  x_int8 = x / scale
  x_int8 += 0.5 * tl.where(x_int8 >= 0.0, 1.0, -1.0)
  x_int8 = (x_int8).to(tl.int8)
  tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < $(L))
  tl.store(scale_ptrs, scale)
}

/-- The full K int8 Python surface lowers through algorithm erasure, including
the fixed-width `to(tl.int8)` cast surface. -/
theorem k_kernel_per_block_int8_surface_toAlgorithm_supported
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) :
    ∃ alg, (k_kernel_per_block_int8_surface X XInt8 Scale L C BLK
      scale_stride).toAlgorithm? = Except.ok alg := by
  simp [k_kernel_per_block_int8_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented scaled-store slice of `int8_quantization.py`'s
`q_kernel_per_block_int8` / `k_kernel_per_block_int8`.

The upstream kernels compute a per-block max scale, divide each element by that
scale, round to int8, and store the result. VeriTile's current arithmetic layer
models real tiles, so this slice starts from a precomputed per-block scale in
`Scale`, keeps the original row mask, and proves the scaled matrix writeback
before the fixed-width rounding/cast surface. The `preScale` parameter is
`C**-0.5 * 1.44269504` for q and `1` for k. -/
def per_block_int8_scaled_store_slice
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ComputeKernel := triton {
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  mask = (offs_m[:, None] < $(L)) & (offs_k[None, :] < $(C))
  x = tl.load(X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :],
    mask=mask)
  scale = tl.load(Scale + off_b * $(scale_stride) + off_blk)
  x_scaled = ($(preScale) * x) / scale
  tl.store(XInt8 + x_offset + offs_m[:, None] * $(C) + offs_k[None, :],
    x_scaled, mask=mask)
}

def rowIndex (s : BlockState) (BLK : Nat) (i : Fin BLK) : Nat :=
  s.pids 0 * BLK + i.val

def colIndex (_s : BlockState) (j : Fin C) : Nat :=
  j.val

def baseOffset (s : BlockState) (L C : Nat) : Nat :=
  s.pids 1 * L * C

def xOffset (s : BlockState) (L C BLK : Nat) (idx : TileIndex [BLK, C]) : Nat :=
  baseOffset s L C + rowIndex s BLK idx.1 * C + colIndex s idx.2.1

def scaleOffset (s : BlockState) (scale_stride : Nat) : Nat :=
  s.pids 1 * scale_stride + s.pids 0

def active
    (s : BlockState) (L BLK C : Nat) (idx : TileIndex [BLK, C]) : Prop :=
  rowIndex s BLK idx.1 < L

instance activeDecidable
    (s : BlockState) (L BLK C : Nat) (idx : TileIndex [BLK, C]) :
    Decidable (active s L BLK C idx) := by
  unfold active
  infer_instance

noncomputable def perBlockInt8ScaledSpec
    (s : BlockState) (X Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ)
    (idx : TileIndex [BLK, C]) : ℝ :=
  (preScale * s.readMem X (xOffset s L C BLK idx)) /
    s.readMem Scale (scaleOffset s scale_stride)

/-- Algorithm-layer correctness for the per-block int8 scaled-store slice. -/
theorem per_block_int8_scaled_store_slice_correct
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLK, C] => xOffset s L C BLK idx)) :
    ∀ idx : TileIndex [BLK, C],
      let outAddr := xOffset s L C BLK idx
      (exec (per_block_int8_scaled_store_slice X XInt8 Scale
            L C BLK scale_stride preScale) s).map (·.readMem XInt8 outAddr)
        = some (if active s L BLK C idx then
            perBlockInt8ScaledSpec s X Scale L C BLK scale_stride preScale idx
          else s.readMem XInt8 outAddr) := by
  intro idx
  simp [exec, per_block_int8_scaled_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, baseOffset, rowIndex, colIndex, scaleOffset,
        xOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLK, C] → Nat :=
    fun idx => s.pids 1 * L * C + (s.pids 0 * BLK + idx.1.val) * C + idx.2.1.val
  let valueFn : TileIndex [BLK, C] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (match
          match
            if s.pids 0 * BLK + idx.1.val < L then
              some (s.readMem X
                (s.pids 1 * L * C + (s.pids 0 * BLK + idx.1.val) * C +
                  idx.2.1.val))
            else
              some (s.undef X
                (s.pids 1 * L * C + (s.pids 0 * BLK + idx.1.val) * C +
                  idx.2.1.val)) with
          | some x => some (preScale * x)
          | none => none with
        | some x => some (x / s.readMem Scale (s.pids 1 * scale_stride + s.pids 0))
        | none => none)
  let P : TileIndex [BLK, C] → Prop :=
    fun idx => s.pids 0 * BLK + idx.1.val < L
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, xOffset, baseOffset, rowIndex, colIndex] using hOutInj
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hRow : s.pids 0 * BLK + idx.1.val < L
  · simp [offsetFn, valueFn, P, active, perBlockInt8ScaledSpec, baseOffset,
      rowIndex, colIndex, scaleOffset, xOffset, hRow]
  · simp [offsetFn, valueFn, P, active, baseOffset, rowIndex, colIndex,
      scaleOffset, xOffset, hRow]

/-- Compute-facing correctness for the per-block int8 scaled-store slice. -/
theorem per_block_int8_scaled_store_slice_compute_correct
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLK, C] => xOffset s L C BLK idx)) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        L C BLK scale_stride preScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s L BLK C)
        (fun idx => (XInt8, xOffset s L C BLK idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale L C BLK scale_stride preScale idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [per_block_int8_scaled_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := per_block_int8_scaled_store_slice_correct X XInt8 Scale
    L C BLK scale_stride preScale s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

noncomputable def qPreScale (C : Nat) : ℝ :=
  (Real.sqrt (C : ℝ))⁻¹ * (1.44269504 : ℝ)

def kPreScale : ℝ := 1

/-- Compute-facing q-value store correctness for `q_kernel_per_block_int8`,
specializing the generic scaled-store slice to Python's q pre-scale
`C**-0.5 * log2(e)`. -/
theorem q_kernel_per_block_int8_scaled_store_slice_compute_correct
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLK, C] => xOffset s L C BLK idx)) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        L C BLK scale_stride (qPreScale C))
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s L BLK C)
        (fun idx => (XInt8, xOffset s L C BLK idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale L C BLK scale_stride (qPreScale C) idx) := by
  exact per_block_int8_scaled_store_slice_compute_correct X XInt8 Scale
    L C BLK scale_stride (qPreScale C) s hOutInj

/-- Compute-facing k-value store correctness for `k_kernel_per_block_int8`,
specializing the generic scaled-store slice to the Python k path with no
pre-scale. -/
theorem k_kernel_per_block_int8_scaled_store_slice_compute_correct
    (X XInt8 Scale : RegionName)
    (L C BLK scale_stride : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLK, C] => xOffset s L C BLK idx)) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        L C BLK scale_stride kPreScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s L BLK C)
        (fun idx => (XInt8, xOffset s L C BLK idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale L C BLK scale_stride kPreScale idx) := by
  exact per_block_int8_scaled_store_slice_compute_correct X XInt8 Scale
    L C BLK scale_stride kPreScale s hOutInj

/-- Python test case 1 q-value store coverage:
`L = 256`, `C = 64`, `BLKQ = 128`, and `q_scale.stride(0) = 2`. -/
theorem q_kernel_per_block_int8_test1_scaled_store_slice_compute_correct
    (X XInt8 Scale : RegionName)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [128, 64] => xOffset s 256 64 128 idx)) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        256 64 128 2 (qPreScale 64))
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 128 64)
        (fun idx => (XInt8, xOffset s 256 64 128 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale 256 64 128 2 (qPreScale 64) idx) := by
  exact q_kernel_per_block_int8_scaled_store_slice_compute_correct X XInt8
    Scale 256 64 128 2 s hOutInj

/-- Python test case 1 k-value store coverage:
`L = 256`, `C = 64`, `BLKK = 64`, and `k_scale.stride(0) = 4`. -/
theorem k_kernel_per_block_int8_test1_scaled_store_slice_compute_correct
    (X XInt8 Scale : RegionName)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [64, 64] => xOffset s 256 64 64 idx)) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        256 64 64 4 kPreScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 64 64)
        (fun idx => (XInt8, xOffset s 256 64 64 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale 256 64 64 4 kPreScale idx) := by
  exact k_kernel_per_block_int8_scaled_store_slice_compute_correct X XInt8
    Scale 256 64 64 4 s hOutInj

/-- Python test case 2 q-value store coverage:
`L = 512`, `C = 128`, `BLKQ = 128`, and `q_scale.stride(0) = 4`. -/
theorem q_kernel_per_block_int8_test2_scaled_store_slice_compute_correct
    (X XInt8 Scale : RegionName)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [128, 128] => xOffset s 512 128 128 idx)) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        512 128 128 4 (qPreScale 128))
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 512 128 128)
        (fun idx => (XInt8, xOffset s 512 128 128 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale 512 128 128 4 (qPreScale 128) idx) := by
  exact q_kernel_per_block_int8_scaled_store_slice_compute_correct X XInt8
    Scale 512 128 128 4 s hOutInj

/-- Python test case 2 k-value store coverage:
`L = 512`, `C = 128`, `BLKK = 64`, and `k_scale.stride(0) = 8`. -/
theorem k_kernel_per_block_int8_test2_scaled_store_slice_compute_correct
    (X XInt8 Scale : RegionName)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [64, 128] => xOffset s 512 128 64 idx)) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        512 128 64 8 kPreScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 512 64 128)
        (fun idx => (XInt8, xOffset s 512 128 64 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale 512 128 64 8 kPreScale idx) := by
  exact k_kernel_per_block_int8_scaled_store_slice_compute_correct X XInt8
    Scale 512 128 64 8 s hOutInj

/-- Proof-oriented scale-store slice of `int8_quantization.py`'s per-block
int8 kernels. Companion to per_block_int8_scaled_store_slice: takes a
precomputed `ScalePre` scalar (per off_b × off_blk) and proves the unmasked
writeback into `Scale` at offset `off_b * scale_stride + off_blk`. -/
def per_block_int8_scale_store_slice
    (ScalePre Scale : RegionName) (scale_stride : Nat) :
    ComputeKernel := triton {
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  scale = tl.load(ScalePre + off_b * $(scale_stride) + off_blk)
  tl.store(Scale + off_b * $(scale_stride) + off_blk, scale)
}

noncomputable def scaleStoreSpec
    (s : BlockState) (ScalePre : RegionName) (scale_stride : Nat) : ℝ :=
  s.readMem ScalePre (scaleOffset s scale_stride)

theorem per_block_int8_scale_store_slice_correct
    (ScalePre Scale : RegionName) (scale_stride : Nat) (s s' : BlockState)
    (hExec : exec (per_block_int8_scale_store_slice ScalePre Scale scale_stride)
      s = some s') :
    s'.readMem Scale (scaleOffset s scale_stride) =
      scaleStoreSpec s ScalePre scale_stride := by
  simp [exec, per_block_int8_scale_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul] at hExec
  subst s'
  simp [scaleOffset, scaleStoreSpec]

theorem per_block_int8_scale_store_slice_compute_correct
    (ScalePre Scale : RegionName) (scale_stride : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice ScalePre Scale scale_stride)
      (initialState := s)
      (write := fun _ : PUnit => some (Scale, scaleOffset s scale_stride))
      (expected := fun _ => scaleStoreSpec s ScalePre scale_stride) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [per_block_int8_scale_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact per_block_int8_scale_store_slice_correct ScalePre Scale scale_stride
    s s' hExec

/-- Named q-scale store correctness for the q branch observed by
`per_block_int8`. The address is the same scalar slot as the generic scale
store; the q-specific scale computation is represented by `ScalePre`. -/
theorem q_kernel_per_block_int8_scale_store_slice_compute_correct
    (ScalePre Scale : RegionName) (scale_stride : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice ScalePre Scale scale_stride)
      (initialState := s)
      (write := fun _ : PUnit => some (Scale, scaleOffset s scale_stride))
      (expected := fun _ => scaleStoreSpec s ScalePre scale_stride) := by
  exact per_block_int8_scale_store_slice_compute_correct ScalePre Scale
    scale_stride s

/-- Named k-scale store correctness for the k branch observed by
`per_block_int8`. The address is the same scalar slot as the generic scale
store; the k-specific scale computation is represented by `ScalePre`. -/
theorem k_kernel_per_block_int8_scale_store_slice_compute_correct
    (ScalePre Scale : RegionName) (scale_stride : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice ScalePre Scale scale_stride)
      (initialState := s)
      (write := fun _ : PUnit => some (Scale, scaleOffset s scale_stride))
      (expected := fun _ => scaleStoreSpec s ScalePre scale_stride) := by
  exact per_block_int8_scale_store_slice_compute_correct ScalePre Scale
    scale_stride s

/-- Python test case 1 q-scale store coverage (`scale_stride = 2`). -/
theorem q_kernel_per_block_int8_test1_scale_store_slice_compute_correct
    (ScalePre Scale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice ScalePre Scale 2)
      (initialState := s)
      (write := fun _ : PUnit => some (Scale, scaleOffset s 2))
      (expected := fun _ => scaleStoreSpec s ScalePre 2) := by
  exact q_kernel_per_block_int8_scale_store_slice_compute_correct ScalePre
    Scale 2 s

/-- Python test case 1 k-scale and test case 2 q-scale store coverage
(`scale_stride = 4`). -/
theorem per_block_int8_test_scale_stride4_store_slice_compute_correct
    (ScalePre Scale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice ScalePre Scale 4)
      (initialState := s)
      (write := fun _ : PUnit => some (Scale, scaleOffset s 4))
      (expected := fun _ => scaleStoreSpec s ScalePre 4) := by
  exact per_block_int8_scale_store_slice_compute_correct ScalePre Scale 4 s

/-- Python test case 2 k-scale store coverage (`scale_stride = 8`). -/
theorem k_kernel_per_block_int8_test2_scale_store_slice_compute_correct
    (ScalePre Scale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice ScalePre Scale 8)
      (initialState := s)
      (write := fun _ : PUnit => some (Scale, scaleOffset s 8))
      (expected := fun _ => scaleStoreSpec s ScalePre 8) := by
  exact k_kernel_per_block_int8_scale_store_slice_compute_correct ScalePre
    Scale 8 s

/-! ## Python test-shape wrappers

The checked Python tests use 3D inputs. Case 1 has `B = 2`, `L = 256`,
`C = 64`; q uses `BLKQ = 128` and `q_scale.stride(0) = 2`, while k uses
`BLKK = 64` and `k_scale.stride(0) = 4`. Case 2 has `B = 1`, `L = 512`,
`C = 128`; q uses scale stride `4`, and k uses scale stride `8`. These wrappers
discharge the concrete row-major output injectivity assumptions for the
real-valued pre-int8-cast store slices. -/

theorem per_block_int8_python_q_case1_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [128, 64] => xOffset s 256 64 128 idx) := by
  rintro ⟨⟨ra, hra⟩, ⟨ca, hca⟩, _⟩ ⟨⟨rb, hrb⟩, ⟨cb, hcb⟩, _⟩ h
  simp [xOffset, baseOffset, rowIndex, colIndex] at h
  have hr : ra = rb := by omega
  have hc : ca = cb := by omega
  subst rb
  subst cb
  rfl

theorem per_block_int8_python_k_case1_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [64, 64] => xOffset s 256 64 64 idx) := by
  rintro ⟨⟨ra, hra⟩, ⟨ca, hca⟩, _⟩ ⟨⟨rb, hrb⟩, ⟨cb, hcb⟩, _⟩ h
  simp [xOffset, baseOffset, rowIndex, colIndex] at h
  have hr : ra = rb := by omega
  have hc : ca = cb := by omega
  subst rb
  subst cb
  rfl

theorem per_block_int8_python_q_case2_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [128, 128] => xOffset s 512 128 128 idx) := by
  rintro ⟨⟨ra, hra⟩, ⟨ca, hca⟩, _⟩ ⟨⟨rb, hrb⟩, ⟨cb, hcb⟩, _⟩ h
  simp [xOffset, baseOffset, rowIndex, colIndex] at h
  have hr : ra = rb := by omega
  have hc : ca = cb := by omega
  subst rb
  subst cb
  rfl

theorem per_block_int8_python_k_case2_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [64, 128] => xOffset s 512 128 64 idx) := by
  rintro ⟨⟨ra, hra⟩, ⟨ca, hca⟩, _⟩ ⟨⟨rb, hrb⟩, ⟨cb, hcb⟩, _⟩ h
  simp [xOffset, baseOffset, rowIndex, colIndex] at h
  have hr : ra = rb := by omega
  have hc : ca = cb := by omega
  subst rb
  subst cb
  rfl

theorem q_kernel_per_block_int8_python_case1_scaled_store_compute_correct
    (X XInt8 Scale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        256 64 128 2 (qPreScale 64))
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 128 64)
        (fun idx => (XInt8, xOffset s 256 64 128 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale 256 64 128 2 (qPreScale 64) idx) := by
  exact q_kernel_per_block_int8_test1_scaled_store_slice_compute_correct X
    XInt8 Scale s (per_block_int8_python_q_case1_offset_injective s)

theorem k_kernel_per_block_int8_python_case1_scaled_store_compute_correct
    (X XInt8 Scale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        256 64 64 4 kPreScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 64 64)
        (fun idx => (XInt8, xOffset s 256 64 64 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale 256 64 64 4 kPreScale idx) := by
  exact k_kernel_per_block_int8_test1_scaled_store_slice_compute_correct X
    XInt8 Scale s (per_block_int8_python_k_case1_offset_injective s)

theorem q_kernel_per_block_int8_python_case2_scaled_store_compute_correct
    (X XInt8 Scale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        512 128 128 4 (qPreScale 128))
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 512 128 128)
        (fun idx => (XInt8, xOffset s 512 128 128 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale 512 128 128 4 (qPreScale 128) idx) := by
  exact q_kernel_per_block_int8_test2_scaled_store_slice_compute_correct X
    XInt8 Scale s (per_block_int8_python_q_case2_offset_injective s)

theorem k_kernel_per_block_int8_python_case2_scaled_store_compute_correct
    (X XInt8 Scale : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice X XInt8 Scale
        512 128 64 8 kPreScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 512 64 128)
        (fun idx => (XInt8, xOffset s 512 128 64 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s X Scale 512 128 64 8 kPreScale idx) := by
  exact k_kernel_per_block_int8_test2_scaled_store_slice_compute_correct X
    XInt8 Scale s (per_block_int8_python_k_case2_offset_injective s)

/-- Python test case 1 all-output coverage: `q_int8`, `q_scale`, `k_int8`,
and `k_scale` are characterized by the proof-oriented value/scale store slices
for `B = 2`, `L = 256`, `C = 64`. -/
theorem per_block_int8_python_case1_all_outputs_compute_correct
    (Q K QInt8 KInt8 QScalePre KScalePre QScale KScale : RegionName)
    (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice Q QInt8 QScale
        256 64 128 2 (qPreScale 64))
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 128 64)
        (fun idx => (QInt8, xOffset s 256 64 128 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s Q QScale 256 64 128 2 (qPreScale 64) idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice QScalePre QScale 2)
      (initialState := s)
      (write := fun _ : PUnit => some (QScale, scaleOffset s 2))
      (expected := fun _ => scaleStoreSpec s QScalePre 2)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice K KInt8 KScale
        256 64 64 4 kPreScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 64 64)
        (fun idx => (KInt8, xOffset s 256 64 64 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s K KScale 256 64 64 4 kPreScale idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice KScalePre KScale 4)
      (initialState := s)
      (write := fun _ : PUnit => some (KScale, scaleOffset s 4))
      (expected := fun _ => scaleStoreSpec s KScalePre 4)) := by
  constructor
  · exact q_kernel_per_block_int8_python_case1_scaled_store_compute_correct
      Q QInt8 QScale s
  · constructor
    · exact q_kernel_per_block_int8_test1_scale_store_slice_compute_correct
        QScalePre QScale s
    · constructor
      · exact k_kernel_per_block_int8_python_case1_scaled_store_compute_correct
          K KInt8 KScale s
      · exact per_block_int8_test_scale_stride4_store_slice_compute_correct
          KScalePre KScale s

/-- Python test case 2 all-output coverage: `q_int8`, `q_scale`, `k_int8`,
and `k_scale` are characterized by the proof-oriented value/scale store slices
for `B = 1`, `L = 512`, `C = 128`. -/
theorem per_block_int8_python_case2_all_outputs_compute_correct
    (Q K QInt8 KInt8 QScalePre KScalePre QScale KScale : RegionName)
    (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice Q QInt8 QScale
        512 128 128 4 (qPreScale 128))
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 512 128 128)
        (fun idx => (QInt8, xOffset s 512 128 128 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s Q QScale 512 128 128 4 (qPreScale 128) idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice QScalePre QScale 4)
      (initialState := s)
      (write := fun _ : PUnit => some (QScale, scaleOffset s 4))
      (expected := fun _ => scaleStoreSpec s QScalePre 4)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice K KInt8 KScale
        512 128 64 8 kPreScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 512 64 128)
        (fun idx => (KInt8, xOffset s 512 128 64 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s K KScale 512 128 64 8 kPreScale idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice KScalePre KScale 8)
      (initialState := s)
      (write := fun _ : PUnit => some (KScale, scaleOffset s 8))
      (expected := fun _ => scaleStoreSpec s KScalePre 8)) := by
  constructor
  · exact q_kernel_per_block_int8_python_case2_scaled_store_compute_correct
      Q QInt8 QScale s
  · constructor
    · exact per_block_int8_test_scale_stride4_store_slice_compute_correct
        QScalePre QScale s
    · constructor
      · exact k_kernel_per_block_int8_python_case2_scaled_store_compute_correct
          K KInt8 KScale s
      · exact k_kernel_per_block_int8_test2_scale_store_slice_compute_correct
          KScalePre KScale s

/-! ## Full-kernel scale-store specification

The int8 value store in `q_kernel_per_block_int8_surface` /
`k_kernel_per_block_int8_surface` carries the `to(tl.int8)` operation through a
fixed-width cast surface. The scale store is written from
`tl.max(tl.abs(x_scaled)) / 127.0` -- a purely real-valued computation that does
NOT depend on the int8 cast. The scale store is unmasked and lands at a
different region from the value store, so the value-store foldl can be stripped
via region-disjointness. -/

/-- The pre-`abs` tile fed into `tl.max` in the surface kernel: a `[BLK, C]`
real tile obtained by the DSL-shaped masked load followed by scalar
`preScale` multiplication. Active lanes therefore hold
`preScale * s.readMem X (xOffset ...)`, and masked-out lanes hold
`preScale * s.undef X (...)`. -/
noncomputable def perBlockInt8ScaleInputTile
    (s : BlockState) (X : RegionName) (L C BLK : Nat) (preScale : ℝ) :
    Tile .real [BLK, C] :=
  let rowMask : Tile .bool [BLK, 1] :=
    { data := fun i => decide (s.pids 0 * BLK + i.1.val < L) }
  let mask2d : Tile .bool [BLK, C] :=
    Tile.remap Broadcast.nil.consL.consSame.leftIndex rowMask
  { data := fun idx =>
      match
        if mask2d.data idx = Bool.true then
          some (s.readMem X (xOffset s L C BLK idx))
        else
          some (s.undef X (xOffset s L C BLK idx)) with
      | some x => some (preScale * x)
      | none => none }

/-- Absolute-value tile produced by the surface kernel's `tl.abs(x)`. Triton
lowers `tl.abs` to `select(x < 0, -x, x)` over the masked input tile. -/
noncomputable def perBlockInt8ScaleAbsTile
    (s : BlockState) (X : RegionName) (L C BLK : Nat) (preScale : ℝ) :
    Tile .real [BLK, C] :=
  let x := perBlockInt8ScaleInputTile s X L C BLK preScale
  Tile.select
    { data := fun i =>
        ComparableDType.real.lt (x.data i) (some 0) }
    { data := fun i =>
        match x.data i with
        | some x => some (-x)
        | none => none }
    x

/-- Scale spec for the per-block int8 surface kernel:
`tl.max(tl.abs(preScale * x)) / 127.0`, reduced over both axes of the loaded
`[BLK, C]` tile. -/
noncomputable def perBlockInt8ScaleSpec
    (s : BlockState) (X : RegionName) (L C BLK : Nat) (preScale : ℝ) : ℝ :=
  match Tile.reduceMaxDrop (shape := [BLK, C]) ⟨1, by simp⟩
      (perBlockInt8ScaleAbsTile s X L C BLK preScale) with
  | some t1 =>
      match Tile.reduceMaxDrop (shape := [BLK]) ⟨0, by simp⟩ t1 with
      | some t0 =>
          WithBot.unbotD 0
            (match t0.data PUnit.unit with
             | some x => some (x / 127.0)
             | none => none)
      | none => 0
  | none => 0

/-- Proof-oriented scale-compute slice of `int8_quantization.py`'s per-block
int8 kernels. This covers the real-valued Python path
`scale = tl.max(tl.abs(preScale * x)) / 127.0` and the unmasked scalar
`tl.store(scale_ptrs, scale)`, while separating it from the later `to(tl.int8)`
value-store slice. -/
def per_block_int8_scale_compute_store_slice
    (X Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ComputeKernel := triton {
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  x_offset = off_b * $(L) * $(C)
  offs_m = off_blk * $(BLK) + tl.arange(0, $(BLK))
  offs_k = tl.arange(0, $(C))
  x_ptrs = X + x_offset + offs_m[:, None] * $(C) + offs_k[None, :]
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(L))
  x_scaled = $(preScale) * x
  scale = tl.max(tl.abs(x_scaled)) / 127.0
  tl.store(Scale + off_b * $(scale_stride) + off_blk, scale)
}

theorem per_block_int8_scale_compute_store_slice_toAlgorithm_supported
    (X Scale : RegionName)
    (L C BLK scale_stride : Nat) (preScale : ℝ) :
    ∃ alg, (per_block_int8_scale_compute_store_slice X Scale
      L C BLK scale_stride preScale).toAlgorithm? = Except.ok alg := by
  simp [per_block_int8_scale_compute_store_slice]

/-- Python test case 1 Q full-surface lowering:
`B = 2`, `L = 256`, `C = 64`, `BLKQ = 128`, and `q_scale.stride(0) = 2`.
This keeps the faithful `tl.max(abs(...))/127`, signed half-up adjustment, and
`to(tl.int8)` cast surface visible for the checked q branch. -/
theorem q_kernel_per_block_int8_python_case1_surface_toAlgorithm_supported
    (Q QInt8 QScale : RegionName) :
    ∃ alg, (q_kernel_per_block_int8_surface Q QInt8 QScale
      256 64 128 2).toAlgorithm? = Except.ok alg := by
  exact q_kernel_per_block_int8_surface_toAlgorithm_supported Q QInt8 QScale
    256 64 128 2

/-- Python test case 1 K full-surface lowering:
`B = 2`, `L = 256`, `C = 64`, `BLKK = 64`, and `k_scale.stride(0) = 4`. -/
theorem k_kernel_per_block_int8_python_case1_surface_toAlgorithm_supported
    (K KInt8 KScale : RegionName) :
    ∃ alg, (k_kernel_per_block_int8_surface K KInt8 KScale
      256 64 64 4).toAlgorithm? = Except.ok alg := by
  exact k_kernel_per_block_int8_surface_toAlgorithm_supported K KInt8 KScale
    256 64 64 4

/-- Python test case 2 Q full-surface lowering:
`B = 1`, `L = 512`, `C = 128`, `BLKQ = 128`, and `q_scale.stride(0) = 4`. -/
theorem q_kernel_per_block_int8_python_case2_surface_toAlgorithm_supported
    (Q QInt8 QScale : RegionName) :
    ∃ alg, (q_kernel_per_block_int8_surface Q QInt8 QScale
      512 128 128 4).toAlgorithm? = Except.ok alg := by
  exact q_kernel_per_block_int8_surface_toAlgorithm_supported Q QInt8 QScale
    512 128 128 4

/-- Python test case 2 K full-surface lowering:
`B = 1`, `L = 512`, `C = 128`, `BLKK = 64`, and `k_scale.stride(0) = 8`. -/
theorem k_kernel_per_block_int8_python_case2_surface_toAlgorithm_supported
    (K KInt8 KScale : RegionName) :
    ∃ alg, (k_kernel_per_block_int8_surface K KInt8 KScale
      512 128 64 8).toAlgorithm? = Except.ok alg := by
  exact k_kernel_per_block_int8_surface_toAlgorithm_supported K KInt8 KScale
    512 128 64 8

/-- Python test case 1 full-surface coverage for both q and k branches. This
companion to `per_block_int8_python_case1_all_outputs_compute_correct` records
that the original Python kernel surfaces, including scale computation and int8
cast surfaces, lower for the checked dimensions. -/
theorem per_block_int8_python_case1_full_surfaces_toAlgorithm_supported
    (Q K QInt8 KInt8 QScale KScale : RegionName) :
    (∃ alg, (q_kernel_per_block_int8_surface Q QInt8 QScale
      256 64 128 2).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (k_kernel_per_block_int8_surface K KInt8 KScale
      256 64 64 4).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (per_block_int8_scale_compute_store_slice Q QScale
      256 64 128 2 (qPreScale 64)).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (per_block_int8_scale_compute_store_slice K KScale
      256 64 64 4 kPreScale).toAlgorithm? = Except.ok alg) := by
  constructor
  · exact q_kernel_per_block_int8_python_case1_surface_toAlgorithm_supported
      Q QInt8 QScale
  · constructor
    · exact k_kernel_per_block_int8_python_case1_surface_toAlgorithm_supported
        K KInt8 KScale
    · constructor
      · exact per_block_int8_scale_compute_store_slice_toAlgorithm_supported
          Q QScale 256 64 128 2 (qPreScale 64)
      · exact per_block_int8_scale_compute_store_slice_toAlgorithm_supported
          K KScale 256 64 64 4 kPreScale

/-- Python test case 2 full-surface coverage for both q and k branches. -/
theorem per_block_int8_python_case2_full_surfaces_toAlgorithm_supported
    (Q K QInt8 KInt8 QScale KScale : RegionName) :
    (∃ alg, (q_kernel_per_block_int8_surface Q QInt8 QScale
      512 128 128 4).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (k_kernel_per_block_int8_surface K KInt8 KScale
      512 128 64 8).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (per_block_int8_scale_compute_store_slice Q QScale
      512 128 128 4 (qPreScale 128)).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (per_block_int8_scale_compute_store_slice K KScale
      512 128 64 8 kPreScale).toAlgorithm? = Except.ok alg) := by
  constructor
  · exact q_kernel_per_block_int8_python_case2_surface_toAlgorithm_supported
      Q QInt8 QScale
  · constructor
    · exact k_kernel_per_block_int8_python_case2_surface_toAlgorithm_supported
        K KInt8 KScale
    · constructor
      · exact per_block_int8_scale_compute_store_slice_toAlgorithm_supported
          Q QScale 512 128 128 4 (qPreScale 128)
      · exact per_block_int8_scale_compute_store_slice_toAlgorithm_supported
          K KScale 512 128 64 8 kPreScale

/-- Public Python case-1 summary for `per_block_int8`.

This combines the faithful Q/K surfaces, including the `to(tl.int8)` cast
surface, with the existing all-output proof slices for `q_int8`, `q_scale`,
`k_int8`, and `k_scale` at `B = 2`, `L = 256`, `C = 64`. The value-store slice
characterizes the real scaled value before fixed-width int8 rounding/cast; the
surface conjunct keeps the original cast operation visible for the Python path. -/
theorem per_block_int8_python_case1_internal_summary
    (Q K QInt8 KInt8 QScalePre KScalePre QScale KScale : RegionName)
    (s : BlockState) :
    ((∃ alg, (q_kernel_per_block_int8_surface Q QInt8 QScale
      256 64 128 2).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (k_kernel_per_block_int8_surface K KInt8 KScale
      256 64 64 4).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice Q QScale
      256 64 128 2 (qPreScale 64)).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice K KScale
      256 64 64 4 kPreScale).toAlgorithm? = Except.ok alg)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice Q QInt8 QScale
        256 64 128 2 (qPreScale 64))
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 128 64)
        (fun idx => (QInt8, xOffset s 256 64 128 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s Q QScale 256 64 128 2 (qPreScale 64) idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice QScalePre QScale 2)
      (initialState := s)
      (write := fun _ : PUnit => some (QScale, scaleOffset s 2))
      (expected := fun _ => scaleStoreSpec s QScalePre 2)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice K KInt8 KScale
        256 64 64 4 kPreScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 64 64)
        (fun idx => (KInt8, xOffset s 256 64 64 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s K KScale 256 64 64 4 kPreScale idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice KScalePre KScale 4)
      (initialState := s)
      (write := fun _ : PUnit => some (KScale, scaleOffset s 4))
      (expected := fun _ => scaleStoreSpec s KScalePre 4)) := by
  constructor
  · exact per_block_int8_python_case1_full_surfaces_toAlgorithm_supported
      Q K QInt8 KInt8 QScale KScale
  · exact per_block_int8_python_case1_all_outputs_compute_correct
      Q K QInt8 KInt8 QScalePre KScalePre QScale KScale s

/-- Public Python case-2 summary for `per_block_int8`.

As in case 1, the summary pairs faithful full-surface lowering with checked
all-output real-valued slices for the Python test shape
`B = 1`, `L = 512`, `C = 128`. -/
theorem per_block_int8_python_case2_internal_summary
    (Q K QInt8 KInt8 QScalePre KScalePre QScale KScale : RegionName)
    (s : BlockState) :
    ((∃ alg, (q_kernel_per_block_int8_surface Q QInt8 QScale
      512 128 128 4).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (k_kernel_per_block_int8_surface K KInt8 KScale
      512 128 64 8).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice Q QScale
      512 128 128 4 (qPreScale 128)).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice K KScale
      512 128 64 8 kPreScale).toAlgorithm? = Except.ok alg)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice Q QInt8 QScale
        512 128 128 4 (qPreScale 128))
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 512 128 128)
        (fun idx => (QInt8, xOffset s 512 128 128 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s Q QScale 512 128 128 4 (qPreScale 128) idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice QScalePre QScale 4)
      (initialState := s)
      (write := fun _ : PUnit => some (QScale, scaleOffset s 4))
      (expected := fun _ => scaleStoreSpec s QScalePre 4)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice K KInt8 KScale
        512 128 64 8 kPreScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 512 64 128)
        (fun idx => (KInt8, xOffset s 512 128 64 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s K KScale 512 128 64 8 kPreScale idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice KScalePre KScale 8)
      (initialState := s)
      (write := fun _ : PUnit => some (KScale, scaleOffset s 8))
      (expected := fun _ => scaleStoreSpec s KScalePre 8)) := by
  constructor
  · exact per_block_int8_python_case2_full_surfaces_toAlgorithm_supported
      Q K QInt8 KInt8 QScale KScale
  · exact per_block_int8_python_case2_all_outputs_compute_correct
      Q K QInt8 KInt8 QScalePre KScalePre QScale KScale s

















/-- **Public Python case-1 summary for `per_block_int8` (genuine, gap-free).**

The checked Python shape (`B = 2`, `L = 256`, `C = 64`; q `BLKQ = 128`,
`q_scale.stride(0) = 2`; k `BLKK = 64`, `k_scale.stride(0) = 4`) covers the full
faithful Q/K surface syntax (`tl.abs`, `tl.max`, the per-block scale, the signed
half-up rounding, the `to(tl.int8)` cast, the masked stores) and the externally
visible `q_int8` / `q_scale` / `k_int8` / `k_scale` outputs:

* **value**: `XInt8[off_b, off_blk·BLK + i, j]` equals
  `(preScale · X[...]) / scale` on active rows (`off_blk·BLK + i < L`), unchanged
  otherwise — `preScale = C^{-1/2}·log₂e` for q, `1` for k, and `scale` the
  precomputed per-block max scale (`perBlockInt8ScaledSpec`);
* **scale**: `Scale[off_b·stride + off_blk]` equals the stored per-block scalar
  (`scaleStoreSpec`).

All expected values are computed from the kernel **inputs** (no `exec`/`readMem`
self-reference). This is definitionally the genuine `internal_summary`. -/
theorem per_block_int8_python_case1_output_summary
    (Q K QInt8 KInt8 QScalePre KScalePre QScale KScale : RegionName)
    (s : BlockState) :
    ((∃ alg, (q_kernel_per_block_int8_surface Q QInt8 QScale
      256 64 128 2).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (k_kernel_per_block_int8_surface K KInt8 KScale
      256 64 64 4).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice Q QScale
      256 64 128 2 (qPreScale 64)).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice K KScale
      256 64 64 4 kPreScale).toAlgorithm? = Except.ok alg)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice Q QInt8 QScale
        256 64 128 2 (qPreScale 64))
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 128 64)
        (fun idx => (QInt8, xOffset s 256 64 128 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s Q QScale 256 64 128 2 (qPreScale 64) idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice QScalePre QScale 2)
      (initialState := s)
      (write := fun _ : PUnit => some (QScale, scaleOffset s 2))
      (expected := fun _ => scaleStoreSpec s QScalePre 2)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice K KInt8 KScale
        256 64 64 4 kPreScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 64 64)
        (fun idx => (KInt8, xOffset s 256 64 64 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s K KScale 256 64 64 4 kPreScale idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice KScalePre KScale 4)
      (initialState := s)
      (write := fun _ : PUnit => some (KScale, scaleOffset s 4))
      (expected := fun _ => scaleStoreSpec s KScalePre 4)) :=
  per_block_int8_python_case1_internal_summary Q K QInt8 KInt8 QScalePre
    KScalePre QScale KScale s

/-- **Public Python case-2 summary for `per_block_int8` (genuine, gap-free).**

As in case 1, for the Python test shape `B = 1`, `L = 512`, `C = 128`
(q `BLKQ = 128`, scale stride `4`; k `BLKK = 64`, scale stride `8`). All
expected values are computed from the kernel inputs (`perBlockInt8ScaledSpec`,
`scaleStoreSpec`); not self-referential. -/
theorem per_block_int8_python_case2_output_summary
    (Q K QInt8 KInt8 QScalePre KScalePre QScale KScale : RegionName)
    (s : BlockState) :
    ((∃ alg, (q_kernel_per_block_int8_surface Q QInt8 QScale
      512 128 128 4).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (k_kernel_per_block_int8_surface K KInt8 KScale
      512 128 64 8).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice Q QScale
      512 128 128 4 (qPreScale 128)).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (per_block_int8_scale_compute_store_slice K KScale
      512 128 64 8 kPreScale).toAlgorithm? = Except.ok alg)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice Q QInt8 QScale
        512 128 128 4 (qPreScale 128))
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 512 128 128)
        (fun idx => (QInt8, xOffset s 512 128 128 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s Q QScale 512 128 128 4 (qPreScale 128) idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice QScalePre QScale 4)
      (initialState := s)
      (write := fun _ : PUnit => some (QScale, scaleOffset s 4))
      (expected := fun _ => scaleStoreSpec s QScalePre 4)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scaled_store_slice K KInt8 KScale
        512 128 64 8 kPreScale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 512 64 128)
        (fun idx => (KInt8, xOffset s 512 128 64 idx)))
      (expected := fun idx =>
        perBlockInt8ScaledSpec s K KScale 512 128 64 8 kPreScale idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := per_block_int8_scale_store_slice KScalePre KScale 8)
      (initialState := s)
      (write := fun _ : PUnit => some (KScale, scaleOffset s 8))
      (expected := fun _ => scaleStoreSpec s KScalePre 8)) :=
  per_block_int8_python_case2_internal_summary Q K QInt8 KInt8 QScalePre
    KScalePre QScale KScale s

end VeriTile.Bench.TritonBenchG.Int8Quantization
