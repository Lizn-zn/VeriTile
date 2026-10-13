import bench.tritonbench_g.kldiv_triton.Proofs.Algorithm
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivTriton.FullAlgorithm
open VeriTile.Triton

noncomputable def forwardInitial (s : BlockState) (Y T O : RegionName)
    (sy st so B : Nat) : BlockState :=
  ((((s.setReg "pid" .int [] (Tile.scalar (s.pids 0 : Int))).setReg "Y" .ptr []
    (Tile.scalar (Y, s.pids 0 * sy))).setReg "T" .ptr []
    (Tile.scalar (T, s.pids 0 * st))).setReg "O" .ptr []
    (Tile.scalar (O, s.pids 0 * so))).setReg "base_offsets" .nat [B]
    (Tile.vec fun i ↦ i.val)

/-- All five actual initialization statements execute before the first column block. -/
theorem forward_initialize (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (logTarget : Bool) (reduction : Nat) :
    stepStmts ((forward Y T O sy st so N B logTarget reduction).body.take 5) s =
      some (forwardInitial s Y T O sy st so B) := by
  simp [forward, forwardSurface, forwardInitial, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    stepStmts, stepStmt, evalOp.eq_def, Tile.ptrAdd, Tile.bop, Tile.uop,
    NumericDType.mul, Tile.scalar, Tile.vec, -Tile.scalar_eta, Function.comp_def,
    Region.cast, Except.bind, bind]
  simp only [← Int.natCast_mul, Int.toNat_natCast]

/-- Initialization preserves all memory and program identifiers. -/
theorem forward_initialize_frame (s : BlockState) (Y T O : RegionName) (sy st so B : Nat) :
    (forwardInitial s Y T O sy st so B).mem = s.mem ∧
    (forwardInitial s Y T O sy st so B).pids = s.pids := by
  exact ⟨rfl, rfl⟩

noncomputable def backwardInitial (s : BlockState) (T O : RegionName)
    (st so N B : Nat) : BlockState :=
  ((((s.setReg "pid" .int [] (Tile.scalar (s.pids 0 : Int))).setReg "O" .ptr []
    (Tile.scalar (O, s.pids 0 * so))).setReg "T" .ptr []
    (Tile.scalar (T, s.pids 0 * st))).setReg "offsets" .int [B]
    (Tile.vec fun i ↦ (i.val : Int))).setReg "mask" .bool [B]
    (Tile.vec fun i ↦ decide (i.val < N))

/-- The backward pre-loop offsets and mask are evaluated even for an empty row. -/
theorem backward_initialize (s : BlockState) (T O : RegionName)
    (st so N B : Nat) (logTarget : Bool) :
    stepStmts ((backward T O st so N B logTarget).body.take 5) s =
      some (backwardInitial s T O st so N B) := by
  cases logTarget <;> simp [backward, backwardSurface, backwardInitial, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    stepStmts, stepStmt, evalOp.eq_def, Tile.ptrAdd, Tile.bop, Tile.uop, Tile.cop,
    NumericDType.mul, ComparableDType.lt, Tile.scalar, Tile.vec, -Tile.scalar_eta,
    Function.comp_def, Region.cast, Except.bind, bind]
  all_goals simp only [← Int.natCast_mul, Int.toNat_natCast]

theorem backward_initialize_frame (s : BlockState) (T O : RegionName) (st so N B : Nat) :
    (backwardInitial s T O st so N B).mem = s.mem ∧
    (backwardInitial s T O st so N B).pids = s.pids := by
  exact ⟨rfl, rfl⟩

end VeriTile.Bench.TritonBenchG.KldivTriton.FullAlgorithm

#guard_msgs (drop info) in
#auditModuleAxioms
