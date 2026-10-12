import bench.tritonbench_g.diag_ssm_triton.Proofs.Candidate
import VeriTile.Triton.Semantics.Step
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroTime
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- With no recurrence step, the real forward source still executes its
masked initial-state and transition loads, and preserves every memory cell. -/
theorem forward_memory (storage : Storage) (swapped : Bool) (A : FlatAlloc)
    (s x l y : RegionName) (batch dim block : Nat) (u : BlockState) :
    ∃ v, exec (A.flattenKernel
      (TypedSource.forward storage swapped s x l y 0 batch dim block).eraseDType.toAlgKernel) u = some v ∧
      ∀ r o, v.mem r o = u.mem r o := by
  cases swapped
  case «true» => simpa only [← Candidate.forward_flat_exec] using
    forward_memory storage Bool.false A s x l y batch dim block u
  case «false» =>
    simp only [exec,FlatAlloc.flattenKernel,Erasure.forward_body]
    refine ⟨?_, ?_, ?_⟩
    rotate_left
    · simp only [Erasure.forward,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
        ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,ite_false,Bool.false_eq_true,↓reduceIte]
      simp only [FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
        stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
        Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
        String.reduceEq,FlatAlloc.shiftBroadcast,
        Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,
        IntegralDType.mod,ComparableDType.lt,stepForLoopAux.forLoop_unfold,
        stepForLoopAux.step_ge,Region.cast,BlockState.setReg_same,
        BlockState.setReg_ne_name,BlockState.setReg_pids,BlockState.setReg_readMem,
        BlockState.setReg_readMemValue,Option.bind_some,Option.map_some,
        Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true,Nat.le_refl]
      rfl
    · intro r o; rfl
termination_by if swapped then 1 else 0

/-- Complex64 has two component loads for each static input, also at T=0. -/
theorem complex_forward_memory (swapped : Bool) (A : FlatAlloc)
    (s x y l : RegionName) (batch dim block : Nat) (u : BlockState) :
    ∃ v, exec (A.flattenKernel
      (TypedSource.complexForward swapped s x y l 0 batch dim block).eraseDType.toAlgKernel) u = some v ∧
      ∀ r o, v.mem r o = u.mem r o := by
  cases swapped
  case «true» => simpa only [← Candidate.complexForward_flat_exec] using
    complex_forward_memory Bool.false A s x y l batch dim block u
  case «false» =>
    simp only [exec,FlatAlloc.flattenKernel,Erasure.complexForward_body]
    refine ⟨?_, ?_, ?_⟩
    rotate_left
    · simp only [Erasure.complexForward,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
        ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,ite_false,Bool.false_eq_true,↓reduceIte]
      simp only [FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
        stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
        Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
        String.reduceEq,FlatAlloc.shiftBroadcast,
        Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,
        IntegralDType.mod,ComparableDType.lt,stepForLoopAux.forLoop_unfold,
        stepForLoopAux.step_ge,Region.cast,BlockState.setReg_same,
        BlockState.setReg_ne_name,BlockState.setReg_pids,BlockState.setReg_readMem,
        BlockState.setReg_readMemValue,Option.bind_some,Option.map_some,
        Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true,Nat.le_refl]
      rfl
    · intro r o; rfl
termination_by if swapped then 1 else 0

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroTime
