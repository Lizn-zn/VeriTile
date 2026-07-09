import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Softmax: naive vs stable — writes-equality refinement

Self-contained showcase, read top to bottom: **kernels** first (what we
built), then the **theorem** (one public headline
`softmax_kernels_refinement_view`; the writes-equality proof is direct, so
there is no `private` lemma section), then a compile-time **trust audit**. The
two real sections below are `Softmax.kernels` and `Softmax.theorems`.

Two block-parallel softmax kernels compute `y = exp(x) / Σ exp(x)` per row:
`naiveSoftmaxKernel` exponentiates `x` raw; `stableSoftmaxKernel` subtracts the
row max first (`exp(x − m) / Σ exp(x − m)`), the numerically-stable rewrite. On
the ℝ abstraction both are exactly equal — the shift-cancellation of softmax
(`naive_eq_stable`, a generic library identity) — so from the same state they
perform the same per-lane writes to `y`.

## The public result (bottom of file)

The single public headline is **`softmax_kernels_refinement_view`** — a
kernel-vs-kernel refinement on `ComputeRefine.Refines`: from the same state the
naive and stable kernels perform the same writes (no scratch regions, so the
scratch list is `[]`). Its statement mentions only the two kernels, the
writes-equality surface, and the state/region types — **no spec, and no input
precondition** (the `#stmtSurfaceSubset` gate below enforces this). For the
rounding-model (∀R) analogue of this surface see
`bench/examples/FusedSwiglu.lean`.
-/

namespace VeriTile.Bench.Examples.Softmax

open VeriTile.Triton VeriTile.Triton.TiledSoftmax
open VeriTile.Examples (programTileView)

/-! ## Kernels -/
section Softmax.kernels

/-- Naive softmax: `y = exp(x) / Σ exp(x)`, exponentiated raw (no max shift). -/
def naiveSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, y)
}

/-- Numerically-stable softmax: subtract the row max first,
`y = exp(x − m) / Σ exp(x − m)`. -/
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

end Softmax.kernels

/-! ## The headline theorem -/
section Softmax.theorems

/- Shared parameters of the headline. Hoisted to a `variable` block so the
signature carries only its genuine hypothesis (block nonemptiness — the writes
are equal for *any* input, so no loaded-input contract is needed). -/
variable (xReg yReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)

include hN in
/-- **naive refines stable** (`ComputeRefine.Refines`, no scratch): from the
same initial state, the naive and stable softmax kernels perform the same
writes — their final memories agree at every cell. The per-lane written-value
equality is the shift-cancellation of softmax, which holds for any input, so no
loaded-input hypothesis is required. -/
theorem softmax_kernels_refinement_view :
    ComputeRefine.Refines
      (naiveSoftmaxKernel xReg yReg blockSize)
      (stableSoftmaxKernel xReg yReg blockSize) s [] := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  apply ComputeKernel.computeRefine_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp [exec, naiveSoftmaxKernel, stepStmts, stepStmt, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.div] at hL
  simp [exec, stableSoftmaxKernel, stepStmts, stepStmt, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div] at hR
  repeat unfold evalOp at hL
  repeat unfold evalOp at hR
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hL
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hR
  subst lhs'
  subst rhs'
  refine BlockState.foldl_writeMem_mem_congr _ _ _ _ ?_ r o _ _ rfl
  intro k _
  exact congrFun (naive_eq_stable
    (fun i : Fin (n + 1) => s.readMem xReg (s.pids 0 * (n + 1) + i.val)) _) k.1

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in the public theorem's transitive proof.
#axiomsClean softmax_kernels_refinement_view

-- (2) The headline is a *kernel-vs-kernel* refinement: its statement may mention
-- ONLY the two kernels, the loaded-input contract, the writes-equality surface,
-- and the state/region types — NO spec.
#stmtSurfaceSubset softmax_kernels_refinement_view ⊆
  [naiveSoftmaxKernel, stableSoftmaxKernel, ComputeRefine.Refines, BlockState, RegionName]

end Softmax.theorems

end VeriTile.Bench.Examples.Softmax
