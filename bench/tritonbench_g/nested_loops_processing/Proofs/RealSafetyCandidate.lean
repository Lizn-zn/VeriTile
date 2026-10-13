import bench.tritonbench_g.nested_loops_processing.Proofs.RealCandidate
import bench.tritonbench_g.nested_loops_processing.Proofs.RealSafetySource
import VeriTile.Triton.Memory.TraceRewrite

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.RealCandidate
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem pointer_safe (bounds : RegionBounds) (version : Bool)
    (src dst : RegionName) (sm sn : Int) (s : BlockState) :
    Stmt.TraceSafe bounds (initialPointer version src dst sm sn) s := by
  cases version
  all_goals
    simp [initialPointer, algorithm, kernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  all_goals simp [Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def]

/-- Safety is transported using both the pure prefix's safety and its exact
intermediate-state equality, not merely equality of final states. -/
theorem trace_equivalence (bounds : RegionBounds) (storage : Storage)
    (src dst : RegionName) (sm sn : Int) (s : BlockState) :
    (algorithm storage Bool.false src dst sm sn).TraceSafe bounds s ↔
      (algorithm storage Bool.true src dst sm sn).TraceSafe bounds s := by
  unfold Kernel.TraceSafe
  conv_lhs => rw [decomposition storage Bool.false]
  conv_rhs => rw [decomposition storage Bool.true]
  simp only [List.append_assoc]
  conv_lhs => rw [TraceRewrite.append_iff]
  conv_rhs => rw [TraceRewrite.append_iff]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply forall_congr'
  intro _
  simp only [List.singleton_append, Stmt.TraceSafeList,
    pointer_safe bounds Bool.false src dst sm sn t,
    pointer_safe bounds Bool.true src dst sm sn t, pointer_step, true_and]

theorem candidate_safe (bounds : RegionBounds) (src dst : RegionName) (sm sn : Nat)
    (s : BlockState) (hsrc : sm + 27 * sn < bounds src)
    (hdst : sm + 47 * sn < bounds dst) :
    (algorithm .int32 Bool.true src dst sm sn).TraceSafe bounds s := by
  exact (trace_equivalence bounds .int32 src dst sm sn s).mp
    (FullReal.source_safe bounds src dst sm sn s hsrc hdst)

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.RealCandidate

#guard_msgs (drop info) in
#auditModuleAxioms
