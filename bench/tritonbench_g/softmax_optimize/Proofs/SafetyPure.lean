import bench.tritonbench_g.softmax_optimize.Proofs.SignedBodies
import VeriTile.Triton.Memory.TraceProofs

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
open VeriTile.Triton SignedTransport

theorem index_prefix_safe (bounds : RegionBounds) (X : RegionName) (N B : Nat)
    (s : BlockState) : Stmt.TraceSafeList bounds (indexPrefix X N B) s := by
  apply TraceProofs.all_safe
  intro st hst t
  simp only [indexPrefix, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl <;>
    simp [Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def]

theorem reduction_safe (bounds : RegionBounds) (B : Nat) (s : BlockState) :
    Stmt.TraceSafeList bounds (reductionStmts B) s := by
  apply TraceProofs.all_safe
  intro st hst t
  simp only [reductionStmts, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl <;>
    simp [Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def]

/-- The numerical update after the full-tile load has no memory access. -/
theorem input_full_update_safe (bounds : RegionBounds) (X : RegionName) (N B : Nat)
    (s : BlockState) : Stmt.TraceSafeList bounds ((loopAbody X N B).drop 4) s := by
  apply TraceProofs.all_safe
  intro st hst t
  simp only [loopAbody, List.drop, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl <;>
    simp [Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def]

/-- The masked input loop has the same memory-free numerical update. -/
theorem input_tail_update_safe (bounds : RegionBounds) (X : RegionName) (N B : Nat)
    (s : BlockState) : Stmt.TraceSafeList bounds ((loopBbody X N B).drop 5) s := by
  apply TraceProofs.all_safe
  intro st hst t
  simp only [loopBbody, List.drop, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl <;>
    simp [Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
