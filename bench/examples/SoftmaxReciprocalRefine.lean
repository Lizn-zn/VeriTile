import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Softmax: per-element-divide vs precomputed-reciprocal — writes-equality refinement

Self-contained showcase, read top to bottom: **kernels** first (what we
built), then the **theorem** (one public headline
`softmax_reciprocal_refinement_view`; the writes-equality proof is direct, so
there is no `private` lemma section), then a compile-time **trust audit**. The
two real sections below are `SoftmaxReciprocal.kernels` and
`SoftmaxReciprocal.theorems`.

Both stable-softmax kernels compute `y = exp(x − m) / Σ exp(x − m)` per row:
`stableSoftmaxKernel` divides each lane by the sum; `softmaxRecipKernel`
precomputes `1 / Σ` once and multiplies (saving per-lane divisions). On the ℝ
abstraction both are exactly equal — `e / S = e · S⁻¹` — so from the same state
they perform the same per-lane writes to `y`.

## The public result (bottom of file)

The single public headline is **`softmax_reciprocal_refinement_view`** — a
kernel-vs-kernel refinement on `ComputeRefine.Refines_without_Rounding`: from the same state the
divide and reciprocal kernels perform the same writes (no scratch regions, so
the scratch list is `[]`). Its statement mentions only the two kernels, the
writes-equality surface, and the state/region types — **no spec, and no input
precondition** (the `#stmtSurfaceSubset` gate below enforces this). For the
rounding-model (∀R) analogue of this surface see
`bench/examples/FusedSwiglu.lean`.
-/

namespace VeriTile.Bench.Examples.SoftmaxReciprocal

open VeriTile.Triton VeriTile.Triton.TiledSoftmax
open VeriTile.Examples (programTileView)

/-! ## Kernels -/
section SoftmaxReciprocal.kernels

/-- Numerically-stable softmax with a per-lane division `y = e / S`. -/
def stableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, y)
}

/-- Optimized stable softmax: precompute `1 / S` once, then multiply per lane
(`y = e · S⁻¹`), saving the per-lane divisions. -/
def softmaxRecipKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
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

/-! ## The headline theorem -/
section SoftmaxReciprocal.theorems

/- Shared parameters of the headline. Hoisted to a `variable` block so the
signature carries only its genuine hypothesis (block nonemptiness — the writes
are equal for *any* input, so no loaded-input contract is needed). -/
variable (xReg yReg : RegionName) (N : Nat) (hN : 0 < N) (s : BlockState)

include hN in
/-- **divide refines reciprocal** (`ComputeRefine.Refines_without_Rounding`, no scratch): from
the same initial state, the per-element-divide and precomputed-reciprocal
stable softmax kernels perform the same writes — their final memories agree at
every cell. The per-lane written-value equality is `e / S = e · S⁻¹`, which
holds for any input, so no loaded-input hypothesis is required. -/
theorem softmax_reciprocal_refinement_view :
    ComputeRefine.Refines_without_Rounding
      (stableSoftmaxKernel xReg yReg N)
      (softmaxRecipKernel xReg yReg N) s [] := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  apply ComputeKernel.computeRefine_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp [exec, stableSoftmaxKernel, stepStmts, stepStmt, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div] at hL
  simp [exec, softmaxRecipKernel, stepStmts, stepStmt, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div] at hR
  repeat unfold evalOp at hL
  repeat unfold evalOp at hR
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hL
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hR
  subst lhs'
  subst rhs'
  refine BlockState.foldl_writeMem_mem_congr _ _ _ _ ?_ r o _ _ rfl
  intro k _
  -- Per-lane value equality: `e / S = e * S⁻¹` (simp normalized `1 / S` to
  -- `S⁻¹`), the reciprocal rewrite `div_eq_mul_inv_real` in inverse form.
  exact div_eq_mul_inv _ _

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in the public theorem's transitive proof.
#axiomsClean softmax_reciprocal_refinement_view

-- (2) The headline is a *kernel-vs-kernel* refinement: its statement may mention
-- ONLY the two kernels, the loaded-input contract, the writes-equality surface,
-- and the state/region types — NO spec.
#stmtSurfaceSubset softmax_reciprocal_refinement_view ⊆
  [stableSoftmaxKernel, softmaxRecipKernel, ComputeRefine.Refines_without_Rounding, BlockState, RegionName]

end SoftmaxReciprocal.theorems

end VeriTile.Bench.Examples.SoftmaxReciprocal
