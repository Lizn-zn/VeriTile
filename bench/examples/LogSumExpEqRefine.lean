import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Log-sum-exp: direct vs shift-trick — writes-equality refinement

Self-contained showcase, read top to bottom: **kernels** first (what we
built), then the **theorem** (one public headline
`log_sum_exp_refinement_view`; the writes-equality proof is direct, so there
is no `private` lemma section), then a compile-time **trust audit**. The two
real sections below are `LogSumExp.kernels` and `LogSumExp.theorems`.

Two block-parallel log-sum-exp kernels compute `y = log Σ exp(x)` per row:
`directLSEKernel` sums `exp(x)` raw; `stableLSEKernel` subtracts the row max
first (`m + log Σ exp(x − m)`), the numerically-stable rewrite. On the ℝ
abstraction both are exactly equal — the shift-invariance of log-sum-exp
(`log_sum_exp_shift_invariant`, a generic library identity) — so from the same
state they perform the same single write to `y`.

## The public result (bottom of file)

The single public headline is **`log_sum_exp_refinement_view`** — a
kernel-vs-kernel refinement on `ComputeRefine.Refines_without_Rounding`: from the same state
the direct and shift-trick kernels perform the same writes (no scratch
regions, so the scratch list is `[]`). Its statement mentions only the two
kernels, the writes-equality surface, and the state/region types — **no spec,
and no input precondition** (the `#stmtSurfaceSubset` gate below enforces
this). For the rounding-model (∀R) analogue of this surface see
`bench/examples/FusedSwiglu.lean`.
-/

namespace VeriTile.Bench.Examples.LogSumExp

open VeriTile.Triton VeriTile.Triton.TiledLogSumExp VeriTile.Triton.TiledSoftmax
open VeriTile.Examples (programTileView)

/-! ## Kernels -/
section LogSumExp.kernels

/-- Direct log-sum-exp: `y = log Σ exp(x)`, summed raw (no max shift). -/
def directLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e, axis=0)
  y    := tl.log(s)
  tl.store($(yReg) + pid, y)
}

/-- Numerically-stable log-sum-exp: subtract the row max first,
`y = m + log Σ exp(x − m)`. -/
def stableLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := m + tl.log(s)
  tl.store($(yReg) + pid, y)
}

end LogSumExp.kernels

/-! ## The headline theorem -/
section LogSumExp.theorems

/- Shared parameters of the headline. Hoisted to a `variable` block so the
theorem signature carries only its genuine hypotheses (here just block
nonemptiness — the writes are equal for *any* input, so no loaded-input
contract is needed). -/
variable (xReg yReg : RegionName) (N : Nat) (hN : 0 < N) (s : BlockState)

include hN in
/-- **direct refines shift-trick** (`ComputeRefine.Refines_without_Rounding`, no scratch):
from the same initial state, the direct and shift-trick LSE kernels perform
the same writes — their final memories agree at every cell. The written-value
equality is the shift-invariance of log-sum-exp, which holds for any input, so
no loaded-input hypothesis is required. -/
theorem log_sum_exp_refinement_view :
    ComputeRefine.Refines_without_Rounding
      (directLSEKernel xReg yReg N)
      (stableLSEKernel xReg yReg N) s [] := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  apply ComputeKernel.computeRefine_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp [exec, directLSEKernel, stepStmts, stepStmt, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul] at hL
  simp [exec, stableLSEKernel, stepStmts, stepStmt, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub] at hR
  repeat unfold evalOp at hL
  repeat unfold evalOp at hR
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hL
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hR
  subst lhs'
  subst rhs'
  -- Both sides are a single scalar write `writeMem yReg s.pid v` on top of a
  -- register-only (mem-preserving) setReg chain.
  rw [BlockState.writeMem_mem, BlockState.writeMem_mem]
  by_cases hc : r = yReg ∧ o = s.pids 0
  · rw [if_pos hc, if_pos hc]
    -- Written-value equality: the shift-invariance of log-sum-exp.
    exact congrArg MemCell.real (log_sum_exp_shift_invariant hN _ _)
  · rw [if_neg hc, if_neg hc]
    rfl

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in the public theorem's transitive proof.
#axiomsClean log_sum_exp_refinement_view

-- (2) The headline is a *kernel-vs-kernel* refinement: its statement may mention
-- ONLY the two kernels, the loaded-input contract, the writes-equality surface,
-- and the state/region types — NO spec. If a spec-like definition ever creeps
-- into the statement, this fails.
#stmtSurfaceSubset log_sum_exp_refinement_view ⊆
  [directLSEKernel, stableLSEKernel, ComputeRefine.Refines_without_Rounding, BlockState, RegionName]

end LogSumExp.theorems

end VeriTile.Bench.Examples.LogSumExp
