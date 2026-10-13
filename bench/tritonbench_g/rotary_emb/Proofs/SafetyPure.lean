import bench.tritonbench_g.rotary_emb.Proofs.Arithmetic
import VeriTile.Triton.Memory.TraceProofs

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton

/-- Index construction cannot access memory, including degenerate tiles. -/
theorem prefix_safe (bounds : RegionBounds) (p : Parameters) (s : BlockState) :
    Stmt.TraceSafeList bounds (prefix p) s := by
  apply TraceProofs.all_safe
  intro st hst t
  simp only [prefix, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def]

theorem q_index_safe (bounds : RegionBounds) (p : Parameters)
    (Q K Cos Sin : RegionName) (s : BlockState) :
    Stmt.TraceSafeList bounds (qIndexBody p Q K Cos Sin) s := by
  change Stmt.TraceSafeList bounds [_,_,_,_] s
  apply TraceProofs.all_safe
  intro st hst t
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl <;>
    simp [Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def]

theorem q_arithmetic_safe (bounds : RegionBounds) (p : Parameters)
    (Q K Cos Sin : RegionName) (s : BlockState) :
    Stmt.TraceSafeList bounds (qArithmeticBody p Q K Cos Sin) s := by
  change Stmt.TraceSafeList bounds [_,_] s
  apply TraceProofs.all_safe
  intro st hst t
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl <;>
    simp [Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
