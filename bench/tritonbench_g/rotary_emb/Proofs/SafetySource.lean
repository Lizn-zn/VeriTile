import bench.tritonbench_g.rotary_emb.Proofs.SafetyLoads
import bench.tritonbench_g.rotary_emb.Proofs.SourceRun

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option maxRecDepth 12000

theorem k_index_safe (bounds : RegionBounds) (p : Parameters)
    (Q K Cos Sin : RegionName) (s : BlockState) :
    Stmt.TraceSafeList bounds (kIndexBody p Q K Cos Sin) s := by
  change Stmt.TraceSafeList bounds [_,_,_,_] s
  apply TraceProofs.all_safe
  intro st hst t
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl <;>
    simp [Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def]

theorem k_arithmetic_safe (bounds : RegionBounds) (p : Parameters)
    (Q K Cos Sin : RegionName) (s : BlockState) :
    Stmt.TraceSafeList bounds (kArithmeticBody p Q K Cos Sin) s := by
  change Stmt.TraceSafeList bounds [_,_] s
  apply TraceProofs.all_safe
  intro st hst t
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl <;>
    simp [Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def]

theorem q_stores_safe (bounds : RegionBounds) (p : Parameters)
    (Q K Cos Sin : RegionName) (s : BlockState) (hb : p.Bounds s bounds Q K Cos Sin) :
    Stmt.TraceSafeList bounds (qStoreBody p Q K Cos Sin) (qArithmeticState p s Q Cos Sin) := by
  change Stmt.TraceSafeList bounds [(mathKernel p Q K Cos Sin).body[18]!,
    (mathKernel p Q K Cos Sin).body[19]!] (qArithmeticState p s Q Cos Sin)
  rw [TraceProofs.cons_of_run_iff (q_even_store_run p Q K Cos Sin s),
    TraceProofs.cons_of_run_iff (q_odd_store_run p Q K Cos Sin s)]
  refine ⟨?_, ?_, Stmt.TraceSafeList.nil_intro⟩
  · solveRotaryAccess using hb.1 Bool.false
  · solveRotaryAccess using hb.1 Bool.true

theorem k_stores_safe (bounds : RegionBounds) (p : Parameters)
    (Q K Cos Sin : RegionName) (s : BlockState) (hb : p.Bounds s bounds Q K Cos Sin) :
    Stmt.TraceSafeList bounds (kStoreBody p Q K Cos Sin) (kArithmeticState p s Q K Cos Sin) := by
  change Stmt.TraceSafeList bounds [(mathKernel p Q K Cos Sin).body[32]!,
    (mathKernel p Q K Cos Sin).body[33]!] (kArithmeticState p s Q K Cos Sin)
  rw [TraceProofs.cons_of_run_iff (k_even_store_run p Q K Cos Sin s),
    TraceProofs.cons_of_run_iff (k_odd_store_run p Q K Cos Sin s)]
  refine ⟨?_, ?_, Stmt.TraceSafeList.nil_intro⟩
  · solveRotaryAccess using hb.2.1 Bool.false
  · solveRotaryAccess using hb.2.1 Bool.true

/-- Every actual source access is safe under its own mask-specific bound.
No collision-freedom, alias-separation, or numerical-output premise is needed. -/
theorem source_safe (bounds : RegionBounds) (p : Parameters)
    (Q K Cos Sin : RegionName) (s : BlockState) (hb : p.Bounds s bounds Q K Cos Sin) :
    Kernel.TraceSafe bounds (mathKernel p Q K Cos Sin) s := by
  unfold Kernel.TraceSafe
  rw [source_parts]
  apply TraceProofs.append_safe _ _ (prefix_safe bounds p s)
  intro t ht
  simp only [prefix_run, Option.some.injEq] at ht
  subst t
  apply TraceProofs.append_safe _ _ (q_index_safe bounds p Q K Cos Sin _)
  intro t ht
  simp only [q_index_run, Option.some.injEq] at ht
  subst t
  apply TraceProofs.append_safe _ _ (q_loads_safe bounds p Q K Cos Sin s hb)
  intro t ht
  simp only [q_load_run, Option.some.injEq] at ht
  subst t
  apply TraceProofs.append_safe _ _ (q_arithmetic_safe bounds p Q K Cos Sin _)
  intro t ht
  simp only [q_arithmetic_run, Option.some.injEq] at ht
  subst t
  apply TraceProofs.append_safe _ _ (q_stores_safe bounds p Q K Cos Sin s hb)
  intro t ht
  simp only [q_store_run, Option.some.injEq] at ht
  subst t
  apply TraceProofs.append_safe _ _ (k_index_safe bounds p Q K Cos Sin _)
  intro t ht
  simp only [k_index_run, Option.some.injEq] at ht
  subst t
  apply TraceProofs.append_safe _ _ (k_loads_safe bounds p Q K Cos Sin s hb)
  intro t ht
  simp only [k_load_run, Option.some.injEq] at ht
  subst t
  apply TraceProofs.append_safe _ _ (k_arithmetic_safe bounds p Q K Cos Sin _)
  intro t ht
  simp only [k_arithmetic_run, Option.some.injEq] at ht
  subst t
  exact k_stores_safe bounds p Q K Cos Sin s hb

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
