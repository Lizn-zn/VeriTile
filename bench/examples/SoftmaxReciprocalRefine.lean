import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Softmax: per-element-divide vs precomputed-reciprocal — rounding-invariant refinement

Self-contained showcase, read top to bottom: **kernels** first, the
**supporting lemma** in the middle (`private` plumbing — a bf16-scatter
congruence), the **theorem** last (one public headline
`softmax_reciprocal_refinement_view`), then a compile-time **trust audit**. The
three real sections are `SoftmaxReciprocal.kernels`, `SoftmaxReciprocal.lemmas`,
`SoftmaxReciprocal.theorems`.

Both stable-softmax kernels compute `y = exp(x − m) / Σ exp(x − m)` per row and
**store the result rounded to bf16** (`(y).to(tl.bfloat16)`):
`stableSoftmaxKernel` divides each lane by the sum; `softmaxRecipKernel`
precomputes `1 / Σ` once and multiplies. The reductions run in ℝ, so both
produce the **same** per-lane ℝ value (`e / S = e · S⁻¹`), and the only rounding
is the shared bf16 output store, which quantizes equal values identically.

## The public result (bottom of file)

The single public headline is **`softmax_reciprocal_refinement_view`** — a
kernel-vs-kernel refinement on `ComputeRefine.Refines R` (the rounding-model
surface): for the rounding model `R`, from the same state the divide and
reciprocal kernels perform the same writes (no scratch, no input precondition).
The compositional rounding pattern is `bench/examples/FusedSwiglu.lean`.

## Exact-ℝ companions (bottom of file)

The file also carries the exact-ℝ (raw-store, no bf16 rounding) variants
`stableSoftmaxKernelReal` / `softmaxRecipKernelReal` together with their
ARM-in-Lean-style story: the reciprocal-side closed form
(`softmax_recip_correct` against `stableRecipSpec`), the load-bearing math
identity (`div_eq_mul_inv_real`), the pointwise `Y`-memory refinement
(`softmax_reciprocal_refinement`), and its `TensorView` surface
(`softmax_reciprocal_refinement_exec_view`).
-/

namespace VeriTile.Bench.Examples.SoftmaxReciprocal

open VeriTile.Triton VeriTile.Triton.TiledSoftmax

/-! ## Kernels -/
section SoftmaxReciprocal.kernels

/-- Numerically-stable softmax with a per-lane division `y = e / S`, stored bf16. -/
def stableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

/-- Optimized stable softmax: precompute `1 / S` once, then multiply per lane
(`y = e · S⁻¹`), stored bf16. -/
def softmaxRecipKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x      := tl.load($(xReg) + offs)
  m      := tl.max(x, axis=0)
  e      := tl.exp(x - m)
  s      := tl.sum(e, axis=0)
  inv_s  := 1 / s
  y      := e * inv_s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

/-! ### Exact-ℝ variants (raw store, no rounding)

Term-identical to the bf16 kernels above except that the final store writes the
raw ℝ value `y` instead of `(y).to(tl.bfloat16)`. These are the kernels of the
exact-ℝ companion theorems at the bottom of the file. The divide-side kernel
mirrors `stableSoftmaxKernelReal` in `bench/examples/SoftmaxEqRefine.lean`
(bench showcases are self-contained, so this file carries its own copy). -/

/-- Stable softmax, exact-ℝ store: `y = e / S` with per-lane division. -/
def stableSoftmaxKernelReal (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, y)
}

/-- Stable softmax with precomputed reciprocal, exact-ℝ store. Saves N-1
    divisions vs the per-element-divide form (`stableSoftmaxKernelReal`). -/
def softmaxRecipKernelReal (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x      := tl.load($(xReg) + offs)
  m      := tl.max(x, axis=0)
  e      := tl.exp(x - m)
  s      := tl.sum(e, axis=0)
  inv_s  := 1 / s
  y      := e * inv_s
  tl.store($(yReg) + offs, y)
}

end SoftmaxReciprocal.kernels

/-! ## Supporting lemma (private plumbing) -/
section SoftmaxReciprocal.lemmas

/-- Two `writeMemAsR` scatters over the same offsets agree cell-by-cell when
their per-lane values agree — the rounding-store analogue of
`BlockState.foldl_writeMem_mem_congr`. -/
private theorem foldl_writeMemAsR_mem_congr {α : Type} (R : RoundingModel)
    (dtype : FloatDType) {region : RegionName} (l : List α) (offsetFn : α → Nat)
    (vL vR : α → TileCarrier dtype.toTileDType)
    (hv : ∀ k ∈ l, vL k = vR k) (r : RegionName) (o : Nat) :
    ∀ sL sR : BlockState, sL.mem r o = sR.mem r o →
      (l.foldl (fun acc k => acc.writeMemAsR R dtype region (offsetFn k) (vL k)) sL).mem r o
        = (l.foldl (fun acc k => acc.writeMemAsR R dtype region (offsetFn k) (vR k)) sR).mem r o := by
  induction l with
  | nil => intro sL sR h; exact h
  | cons hd tl ih =>
      intro sL sR h
      refine ih (fun k hk => hv k (List.mem_cons_of_mem _ hk)) _ _ ?_
      rw [BlockState.writeMemAsR_mem, BlockState.writeMemAsR_mem,
          hv hd List.mem_cons_self]
      by_cases hc : r = region ∧ o = offsetFn hd
      · rw [if_pos hc, if_pos hc]
      · rw [if_neg hc, if_neg hc]; exact h

end SoftmaxReciprocal.lemmas

/-! ## The headline theorem -/
section SoftmaxReciprocal.theorems

/- Shared parameters of the headline: the two regions, the block size
(nonempty), the state, and the rounding model `R`. -/
variable (xReg yReg : RegionName) (N : Nat) (hN : 0 < N) (s : BlockState)
variable (R : RoundingModel)

include hN in
/-- **divide refines reciprocal** (`ComputeRefine.Refines R`, no scratch): for
the rounding model `R`, from the same initial state the per-element-divide and
precomputed-reciprocal kernels perform the same writes. Both compute the same
per-lane ℝ value (`e / S = e · S⁻¹`) and round it at the same bf16 store. -/
specification softmax_reciprocal_refinement_view :
    ComputeRefine.Refines R
      (stableSoftmaxKernel xReg yReg N)
      (softmaxRecipKernel xReg yReg N) s [] := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  apply ComputeKernel.computeRefineR_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp [execR, stableSoftmaxKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComputeExpr.toAlgorithm?] at hL
  simp [Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hL
  simp [execR, softmaxRecipKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComputeExpr.toAlgorithm?] at hR
  simp [Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hR
  subst lhs'
  subst rhs'
  refine foldl_writeMemAsR_mem_congr R .bf16 _ _ _ _ ?_ r o _ _ rfl
  intro k _
  -- per-lane value equality `e / S = e * S⁻¹`, rounded identically at the store
  exact congrArg (fun v : ℝ => R.cast FloatDType.real FloatDType.bf16 (some v))
    (div_eq_mul_inv _ _)

/-! ## Trust audit (compile-time gate)

If either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See `VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in the public theorem's transitive proof.
#axiomsClean softmax_reciprocal_refinement_view

-- (2) The headline is a *kernel-vs-kernel* refinement: its statement may mention
-- ONLY the two kernels, the rounding-model surface, and the state/region types.
#stmtSurfaceSubset softmax_reciprocal_refinement_view ⊆
  [stableSoftmaxKernel, softmaxRecipKernel, ComputeRefine.Refines, RoundingModel,
   BlockState, RegionName]

end SoftmaxReciprocal.theorems

/-! ## Exact-ℝ companion theorems

Real Triton optimization: replace `y = e/s` (per-element division) with
`inv_s = 1/s; y = e * inv_s` (one division total + one multiply per element).
Algorithmically equivalent in ℝ since `e/s = e * (1/s)` when `s ≠ 0`.

The math denotation (`stableSpec`, `tileMax`) lives in
`VeriTile.Triton.Math.Softmax`; the observation vocabulary (`InputLoadedAt`,
`observeAt`, `programTileView`) in `VeriTile.Examples.Common`. The divide side
of the comparison is `stableSoftmaxKernelReal`, proven correct against
`stableSpec` in `bench/examples/SoftmaxEqRefine.lean`; this file carries a
`private` copy of that closed form (bench showcases are self-contained). -/
section SoftmaxReciprocal.theoremsReal

open VeriTile.Examples

/-- The load-bearing math identity: division equals multiplication by
    reciprocal (for non-zero divisor). -/
theorem div_eq_mul_inv_real (a s : ℝ) (hs : s ≠ 0) : a / s = a * (1 / s) := by
  field_simp

/-- Closed-form spec for `softmaxRecipKernelReal`'s `Y[pid*N+i]` cell. -/
noncomputable def stableRecipSpec {N : Nat} (xs : Fin N → ℝ) (m : ℝ) (i : Fin N) : ℝ :=
  Real.exp (xs i - m) * (1 / ∑ j, Real.exp (xs j - m))

/-- Local copy of the divide-side closed form (plumbing; the showcased
original is `softmax_stable_correct` in `bench/examples/SoftmaxEqRefine.lean`). -/
private theorem softmax_stable_correct
    (xReg yReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState) (xs : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs) :
    ∀ i : Fin blockSize,
      observeAt (exec (stableSoftmaxKernelReal xReg yReg blockSize) s) yReg blockSize s.pid i
        = some (stableSpec xs (tileMax hN xs) i) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [n + 1] => s.pid * (n + 1) + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, stableSoftmaxKernelReal, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, stableSpec, tileMax]
  repeat unfold evalOp
  simp [observeAt, exec, stableSoftmaxKernelReal, stepStmts, stepStmt,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, stableSpec, tileMax]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  unfold InputLoadedAt at _h_x
  simp [_h_x]
  rfl

/-- **Reciprocal-form softmax kernel correctness (exact ℝ).** -/
theorem softmax_recip_correct
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs) :
    ∀ i : Fin N,
      observeAt (exec (softmaxRecipKernelReal xReg yReg N) s) yReg N s.pid i
        = some (stableRecipSpec xs (tileMax hN xs) i) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [n + 1] => s.pid * (n + 1) + idx.1.val) :=
    injective_offset_singleton (s.pid * (n + 1))
  simp [observeAt, exec, softmaxRecipKernelReal, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, stableRecipSpec, tileMax]
  repeat unfold evalOp
  simp [observeAt, exec, softmaxRecipKernelReal, stepStmts, stepStmt,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, stableRecipSpec, tileMax]
  unfold InputLoadedAt at _h_x
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x]
  rfl

/-- **Refinement (exact ℝ): stable softmax with per-element division ≡ stable
    softmax with precomputed reciprocal.** Composes via `div_eq_mul_inv_real`. -/
theorem softmax_reciprocal_refinement
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoadedAt s xReg N xs) :
    ∀ i : Fin N,
      observeAt (exec (stableSoftmaxKernelReal xReg yReg N) s) yReg N s.pid i =
      observeAt (exec (softmaxRecipKernelReal  xReg yReg N) s) yReg N s.pid i := by
  intro i
  rw [softmax_stable_correct xReg yReg N hN s xs h_x i,
      softmax_recip_correct  xReg yReg N hN s xs h_x i]
  congr 1
  -- Goal: stableSpec xs (tileMax hN xs) i = stableRecipSpec xs (tileMax hN xs) i
  -- These differ only by a / b vs a * (1/b).
  unfold stableSpec stableRecipSpec
  have h_sum_pos : 0 < ∑ j, Real.exp (xs j - tileMax hN xs) := by
    apply Finset.sum_pos
    · intro j _; exact Real.exp_pos _
    · obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
      exact ⟨⟨0, hN⟩, Finset.mem_univ _⟩
  exact div_eq_mul_inv_real _ _ (ne_of_gt h_sum_pos)

/-- View-level surface for `softmax_reciprocal_refinement`. -/
theorem softmax_reciprocal_refinement_exec_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    ∀ idx : TileIndex [N],
      TensorView.observe (exec (stableSoftmaxKernelReal xReg yReg N) s)
          (programTileView s yReg N) idx =
      TensorView.observe (exec (softmaxRecipKernelReal  xReg yReg N) s)
          (programTileView s yReg N) idx := by
  intro idx
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := N) (xs := xs) h_x
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt]
    using softmax_reciprocal_refinement xReg yReg N hN s xs hx idx.1

/-! ### Trust audit for the exact-ℝ companions -/

#axiomsClean div_eq_mul_inv_real
#axiomsClean softmax_recip_correct
#axiomsClean softmax_reciprocal_refinement
#axiomsClean softmax_reciprocal_refinement_exec_view

end SoftmaxReciprocal.theoremsReal

end VeriTile.Bench.Examples.SoftmaxReciprocal
