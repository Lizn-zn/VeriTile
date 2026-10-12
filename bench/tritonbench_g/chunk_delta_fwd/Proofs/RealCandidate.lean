import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealExecution
import VeriTile.Triton.Math.MatrixRewrite
import VeriTile.Triton.Semantics.ProgramRewrite

/-! The sum-left candidate changes only the outer carry addition. Real addition
commutes in every reached iteration, so complete execution, including failure,
is unchanged. The proof also applies after allocation flattening and does not
use FP assumptions. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealCandidate
open VeriTile.Triton RealProgram
set_option maxHeartbeats 8000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def source (p : Layout) (r : Regions) (kd value initial final : TypedSource.Storage)
    (swapped useSeed storeFinal : Bool) : ComputeKernel :=
  TypedSource.specialize kd value initial final swapped r.key r.value r.delta r.corrected r.history r.initial r.final
    p.keyHead p.keyTime p.keyDim p.valueHead p.valueTime p.valueDim p.historyHead p.historyTime
    p.H p.T p.K p.V p.BT p.BC p.BK p.BV p.NT useSeed storeFinal

def carry (p : Layout) (swapped : Bool) : Stmt :=
  .assign .real [p.BK,p.BV] "b_h" (if swapped then
    .add .real (.consSame (.consSame .nil)) (.ref .real [p.BK,p.BV] "b_h_cumsum") (.ref .real [p.BK,p.BV] "b_h")
  else RealOuterStep.carryOp p)

def beforeCarry (p : Layout) (r : Regions) : List Stmt :=
  RealOuterStep.prelude p r ++ [RealInnerLoop.statement p r]

theorem body_parts (p : Layout) (r : Regions) (kd value initial final : TypedSource.Storage)
    (swapped useSeed storeFinal : Bool) :
    (source p r kd value initial final swapped useSeed storeFinal).eraseDType.toAlgKernel.body =
      cdfPrologue r.initial p.K p.V p.BK p.BV useSeed ++
      [.forRange "i_t" 0 p.NT 1
        (beforeCarry p r ++ [carry p swapped])] ++
      cdfEpilogue r.final p.K p.V p.BK p.BV storeFinal := by
  cases swapped
  · exact RealExecution.body_parts p r kd value initial final useSeed storeFinal
  · cases value <;> cases final
    all_goals
      simp only [source, TypedSource.specialize, beforeCarry, carry, RealOuterStep.prelude,
        RealOuterStep.pointerOp, RealInnerLoop.statement, RealProgram.innerBody,
        cdfPrologue, cdfEpilogue, chunkDeltaInnerBody, cdfStopOp,
        ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
        ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
        ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
        Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
        NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
        Except.bind, bind, List.map_cons, List.map_nil, List.cons_append, List.nil_append,
        eraseDType_float, Bool.true_eq, ite_true]
    all_goals simp only [VeriTile.Triton.eraseDType]
    all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
      MaskOpt.eraseDType.eq_def, eraseDType_float]
    all_goals cases kd <;> cases initial <;>
      simp [TypedSource.Storage.dtype, VeriTile.Triton.eraseDType, FloatDType.toTileDType]

theorem carry_execution (p : Layout) (s : BlockState) :
    stepStmts [carry p Bool.false] s = stepStmts [carry p Bool.true] s := by
  apply Option.ext
  intro t
  exact ⟨MatrixRewrite.add_commute [] [] "b_h" _ _ s t,
    MatrixRewrite.add_commute [] [] "b_h" _ _ s t⟩

theorem loop_execution (p : Layout) (before : List Stmt) (s : BlockState) :
    stepStmt (.forRange "i_t" 0 p.NT 1
      (before ++ [carry p Bool.false])) s =
    stepStmt (.forRange "i_t" 0 p.NT 1
      (before ++ [carry p Bool.true])) s := by
  simp only [stepStmt]
  apply ProgramRewrite.range_congr
  intro u
  simp only [ProgramRewrite.append, carry_execution]

theorem execution_eq (p : Layout) (r : Regions) (kd value initial final : TypedSource.Storage)
    (useSeed storeFinal : Bool) (s : BlockState) :
    exec (source p r kd value initial final Bool.false useSeed storeFinal).eraseDType.toAlgKernel s =
    exec (source p r kd value initial final Bool.true useSeed storeFinal).eraseDType.toAlgKernel s := by
  simp only [exec, body_parts, List.append_assoc, List.singleton_append, ProgramRewrite.append]
  congr 1
  funext u
  simp only [stepStmts, loop_execution]

private theorem flatten_append (A : FlatAlloc) (xs ys : List Stmt) :
    A.flattenStmts (xs ++ ys) = A.flattenStmts xs ++ A.flattenStmts ys := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp only [List.cons_append, FlatAlloc.flattenStmts, ih]

private theorem flatten_carry (A : FlatAlloc) (p : Layout) (swapped : Bool) :
    A.flattenStmt (carry p swapped) = carry p swapped := by
  cases swapped <;> simp [carry, RealOuterStep.carryOp, FlatAlloc.flattenStmt, FlatAlloc.flattenOp]

theorem flat_execution_eq (A : FlatAlloc) (p : Layout) (r : Regions)
    (kd value initial final : TypedSource.Storage) (useSeed storeFinal : Bool) (s : BlockState) :
    exec (A.flattenKernel (source p r kd value initial final Bool.false useSeed storeFinal).eraseDType.toAlgKernel) s =
    exec (A.flattenKernel (source p r kd value initial final Bool.true useSeed storeFinal).eraseDType.toAlgKernel) s := by
  simp only [exec, FlatAlloc.flattenKernel, body_parts, flatten_append,
    FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp, flatten_carry,
    List.append_assoc, List.singleton_append, ProgramRewrite.append]
  congr 1
  funext u
  simp only [stepStmts, loop_execution]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealCandidate
