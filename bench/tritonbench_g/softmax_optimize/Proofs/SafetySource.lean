import bench.tritonbench_g.softmax_optimize.Proofs.SafetyLoopStatements

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
open VeriTile.Triton SignedTransport FullSource

theorem previous_safe (bounds : RegionBounds) (N B : Nat) (s : BlockState) :
    Stmt.TraceSafe bounds (.assign .int [] "prev_multiple" (previousExpression N B)) s := by
  simp [Stmt.TraceSafe, previousExpression, signedQuotient, Op.SafeAt.eq_def]

theorem output_pass_safe (bounds : RegionBounds) (Y X : RegionName)
    (M N B p : Nat) (hB : 0 < B) (hp : p < M)
    (hX : M * N ≤ bounds X) (hY : M * N ≤ bounds Y)
    (s : BlockState) (hc : Control N B p s) :
    Stmt.TraceSafeList bounds
      [fullLoop B (outputFull Y X N B), tailLoop N B (outputTail Y X N B)] s := by
  apply Stmt.TraceSafeList.cons_intro
  · apply full_loop_safe bounds N B p _ s hc
    exact output_full_range_safe bounds Y X M N B p 0 s hB hp hX hY (dvd_zero B) hc.1
  · intro t hr
    have ht := full_loop_control Y X N B p (outputFull Y X N B) (by simp) s t hc hr
    apply Stmt.TraceSafeList.cons_intro
    · apply tail_loop_safe bounds N B p _ t ht
      exact output_tail_range_safe bounds Y X M N B p (prevMultipleNat N B) t
        hB hp hX hY ht.1
    · intro u _
      exact Stmt.TraceSafeList.nil_intro

theorem after_seed_safe (bounds : RegionBounds) (Y X : RegionName)
    (M N B p : Nat) (hN : 0 < N) (hB : 0 < B) (hp : p < M)
    (hX : M * N ≤ bounds X) (hY : M * N ≤ bounds Y)
    (s : BlockState) (hc : Control N B p s) :
    Stmt.TraceSafeList bounds
      ([fullLoop B (inputFull X N B), tailLoop N B (inputTail X N B)] ++
        reductionStmts B ++
        [.assign .int [] "prev_multiple" (previousExpression N B),
         fullLoop B (outputFull Y X N B), tailLoop N B (outputTail Y X N B)]) s := by
  simp only [List.cons_append, List.nil_append]
  apply Stmt.TraceSafeList.cons_intro
  · apply full_loop_safe bounds N B p _ s hc
    exact input_full_range_safe bounds X M N B p 0 s hB hp hX (dvd_zero B) hc.1
  · intro t hr
    have ht := full_loop_control Y X N B p (inputFull X N B) (by simp) s t hc hr
    apply Stmt.TraceSafeList.cons_intro
    · apply tail_loop_safe bounds N B p _ t ht
      exact input_tail_range_safe bounds X M N B p (prevMultipleNat N B) t hB hp hX ht.1
    · intro u hu
      have hcu := tail_loop_control Y X N B p (inputTail X N B) (by simp) t u ht hu
      apply TraceProofs.append_safe (reductionStmts B) u (reduction_safe bounds B u)
      intro v hv
      have hcv := body_control_preserved Y X N B p (reductionStmts B) (by simp) u v hcu hv
      rw [TraceProofs.cons_of_run_iff (previous_run N B hN hB v)]
      refine ⟨previous_safe bounds N B v, ?_⟩
      apply output_pass_safe bounds Y X M N B p hB hp hX hY
      constructor
      · simpa using hcv.1
      · simp

/-- Complete original signed source is access-safe on every in-bounds row.
No numerical state, successful execution or output-value premise is required. -/
theorem original_source_safe (input : Storage) (output : Accumulation)
    (bounds : RegionBounds) (s : BlockState) (Y X : RegionName) (M N B : Nat)
    (hN : 0 < N) (hB : 0 < B) (hr : s.pids 0 < M)
    (hX : M * N ≤ bounds X) (hY : M * N ≤ bounds Y) :
    (kernel input output Bool.false Y X M N B).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [projection]
  unfold Kernel.TraceSafe
  rw [source_body]
  simp only [List.cons_append, List.nil_append]
  rw [TraceProofs.cons_of_run_iff
    (stepStmt_assign_eq_some (v := Tile.scalar (s.pids 0)) (by simp [evalOp.eq_def]))]
  constructor
  · simp [Stmt.TraceSafe, Op.SafeAt.eq_def]
  · apply TraceProofs.assign_cons
    · simp [Op.SafeAt.eq_def]
    · intro m
      apply TraceProofs.assign_cons
      · simp [Op.SafeAt.eq_def]
      · intro z
        rw [TraceProofs.cons_of_run_iff (previous_run N B hN hB _)]
        refine ⟨previous_safe bounds N B _, ?_⟩
        apply after_seed_safe bounds Y X M N B (s.pids 0) hN hB hr hX hY
        constructor <;> simp

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
