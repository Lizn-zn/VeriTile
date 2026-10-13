import bench.tritonbench_g.rotary_emb.Proofs.SafetySource
import bench.tritonbench_g.rotary_emb.Proofs.Projection
import bench.tritonbench_g.softmax_optimize.Proofs.SafetyCongruence

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton

def headPrefix (v : Bool) (BH : Nat) : List Stmt :=
  [.assign .nat [] "cur_head_index" (.programId 0),
   .assign .nat [] "cur_seq_index" (.programId 1), headStatement v BH]

theorem variant_decomposition (v : Bool) (p : Parameters) (Q K Cos Sin : RegionName) :
    (mathVariant v p Q K Cos Sin).body =
      headPrefix v p.blockHead ++ (mathKernel p Q K Cos Sin).body.drop 3 := rfl

theorem head_prefix_safe (bounds : RegionBounds) (v : Bool) (BH : Nat) (s : BlockState) :
    Stmt.TraceSafeList bounds (headPrefix v BH) s := by
  apply TraceProofs.all_safe
  intro st hst t
  simp only [headPrefix, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl
  all_goals cases v <;> simp [headStatement, Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem head_prefix_run (BH : Nat) (s : BlockState) :
    stepStmts (headPrefix Bool.true BH) s = stepStmts (headPrefix Bool.false BH) s := by
  simp only [headPrefix, stepStmts, ← head_statement_run]

/-- The candidate preserves the actual complete trace-safety proposition,
including arbitrary initial registers and degenerate mathematical shapes. -/
theorem math_candidate_trace (bounds : RegionBounds) (p : Parameters)
    (Q K Cos Sin : RegionName) (s : BlockState) :
    Kernel.TraceSafe bounds (mathVariant Bool.true p Q K Cos Sin) s ↔
      Kernel.TraceSafe bounds (mathKernel p Q K Cos Sin) s := by
  rw [← math_original p Q K Cos Sin]
  simp only [Kernel.TraceSafe, variant_decomposition]
  apply SoftmaxOptimize.Safety.trace_prefix_congr
  · intro t
    exact iff_of_true (head_prefix_safe bounds Bool.true p.blockHead t)
      (head_prefix_safe bounds Bool.false p.blockHead t)
  · exact head_prefix_run p.blockHead

theorem typed_safe (q k cos sin : FullSource.Storage) (v : Bool)
    (bounds : RegionBounds) (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState)
    (hb : p.Bounds s bounds Q K Cos Sin) :
    Kernel.TraceSafe bounds
      (FullSource.kernel q k cos sin v Q K Cos Sin p.qBatch p.qHead p.qDim
        p.kBatch p.kHead p.kDim p.cosBatch p.cosDim p.sinBatch p.sinDim p.length
        p.headsQ p.headsK p.blockHead p.blockSeq p.dimension).eraseDType.toAlgKernel s := by
  unfold Kernel.TraceSafe
  rw [FullSource.kernel_math_body]
  cases v
  · rw [math_original]
    exact source_safe bounds p Q K Cos Sin s hb
  · exact (math_candidate_trace bounds p Q K Cos Sin s).2
      (source_safe bounds p Q K Cos Sin s hb)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
