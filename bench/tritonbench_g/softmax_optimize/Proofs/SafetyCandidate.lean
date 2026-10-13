import bench.tritonbench_g.softmax_optimize.Proofs.SafetyCongruence
import bench.tritonbench_g.softmax_optimize.Proofs.SafetySource
import bench.tritonbench_g.softmax_optimize.Proofs.SignedCandidateEquality

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
open VeriTile.Triton SignedTransport FullSource

theorem candidate_index_prefix_safe (bounds : RegionBounds) (X : RegionName) (N B : Nat)
    (s : BlockState) : Stmt.TraceSafeList bounds (indexPrefixCandidate X N B) s := by
  apply TraceProofs.all_safe
  intro st hst t
  simp only [indexPrefixCandidate, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl <;>
    simp [Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def]

theorem candidate_prefix_trace (bounds : RegionBounds) (X : RegionName) (N B : Nat)
    (suffix : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (indexPrefixCandidate X N B ++ suffix) s ↔
      Stmt.TraceSafeList bounds (indexPrefix X N B ++ suffix) s := by
  apply trace_prefix_congr
  · intro t
    exact iff_of_true (candidate_index_prefix_safe bounds X N B t)
      (index_prefix_safe bounds X N B t)
  · exact candidate_prefix_eq X N B

theorem candidate_input_full_trace (bounds : RegionBounds) (X : RegionName) (N B : Nat)
    (s : BlockState) :
    Stmt.TraceSafeList bounds (inputFullCandidate X N B) s ↔
      Stmt.TraceSafeList bounds (inputFull X N B) s := by
  exact candidate_prefix_trace bounds X N B _ s

theorem candidate_input_tail_trace (bounds : RegionBounds) (X : RegionName) (N B : Nat)
    (s : BlockState) :
    Stmt.TraceSafeList bounds (inputTailCandidate X N B) s ↔
      Stmt.TraceSafeList bounds (inputTail X N B) s := by
  simpa only [inputTailCandidate, inputTail, List.append_assoc] using
    candidate_prefix_trace bounds X N B
      ([maskStatement N B] ++ (loopBbody X N B).drop 4) s

theorem candidate_output_full_trace (bounds : RegionBounds) (Y X : RegionName) (N B : Nat)
    (s : BlockState) :
    Stmt.TraceSafeList bounds (outputFullCandidate Y X N B) s ↔
      Stmt.TraceSafeList bounds (outputFull Y X N B) s := by
  simpa only [outputFullCandidate, outputFull, List.append_assoc] using
    candidate_prefix_trace bounds X N B
      (((loopApass2body Y X N B).drop 3).take 3 ++
        ([outputPointer Y B] ++ (loopApass2body Y X N B).drop 7)) s

theorem candidate_output_tail_trace (bounds : RegionBounds) (Y X : RegionName) (N B : Nat)
    (s : BlockState) :
    Stmt.TraceSafeList bounds (outputTailCandidate Y X N B) s ↔
      Stmt.TraceSafeList bounds (outputTail Y X N B) s := by
  simpa only [outputTailCandidate, outputTail, List.append_assoc] using
    candidate_prefix_trace bounds X N B
      ([maskStatement N B] ++ (((loopBpass2body Y X N B).drop 4).take 3 ++
        ([outputPointer Y B] ++ (loopBpass2body Y X N B).drop 8))) s

/-- Trace equivalence uses both local access safety and optional state equality;
it does not infer access safety from final-state equivalence alone. -/
theorem candidate_source_trace (bounds : RegionBounds) (Y X : RegionName) (M N B : Nat)
    (s : BlockState) :
    (mathematical Bool.true Y X M N B).TraceSafe bounds s ↔
      (mathematical Bool.false Y X M N B).TraceSafe bounds s := by
  unfold Kernel.TraceSafe
  rw [candidate_source_body, source_body]
  have ha := signed_statement_trace_congr bounds "start_n" (.constInt 0)
    (.ref .int [] "prev_multiple") (.constNat B) _ _
    (candidate_input_full_trace bounds X N B) (candidate_input_full_eq X N B)
  have hb := signed_statement_trace_congr bounds "start_n" (.ref .int [] "prev_multiple")
    (.constInt N) (.constNat B) _ _
    (candidate_input_tail_trace bounds X N B) (candidate_input_tail_eq X N B)
  have hc := signed_statement_trace_congr bounds "start_n" (.constInt 0)
    (.ref .int [] "prev_multiple") (.constNat B) _ _
    (candidate_output_full_trace bounds Y X N B) (candidate_output_full_eq Y X N B)
  have hd := signed_statement_trace_congr bounds "start_n" (.ref .int [] "prev_multiple")
    (.constInt N) (.constNat B) _ _
    (candidate_output_tail_trace bounds Y X N B) (candidate_output_tail_eq Y X N B)
  have ea := signed_loop_congr "start_n" (.constInt 0) (.ref .int [] "prev_multiple")
    (.constNat B) _ _ (candidate_input_full_eq X N B)
  have eb := signed_loop_congr "start_n" (.ref .int [] "prev_multiple") (.constInt N)
    (.constNat B) _ _ (candidate_input_tail_eq X N B)
  have ec := signed_loop_congr "start_n" (.constInt 0) (.ref .int [] "prev_multiple")
    (.constNat B) _ _ (candidate_output_full_eq Y X N B)
  have ed := signed_loop_congr "start_n" (.ref .int [] "prev_multiple") (.constInt N)
    (.constNat B) _ _ (candidate_output_tail_eq Y X N B)
  simp only [reductionStmts, List.cons_append, List.nil_append, Stmt.TraceSafeList,
    ha, hb, hc, hd, ea, eb, ec, ed]

theorem candidate_source_safe (input : Storage) (output : Accumulation)
    (bounds : RegionBounds) (s : BlockState) (Y X : RegionName) (M N B : Nat)
    (hN : 0 < N) (hB : 0 < B) (hr : s.pids 0 < M)
    (hX : M * N ≤ bounds X) (hY : M * N ≤ bounds Y) :
    (kernel input output Bool.true Y X M N B).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [projection]
  apply (candidate_source_trace bounds Y X M N B s).mpr
  simpa only [projection] using
    original_source_safe input output bounds s Y X M N B hN hB hr hX hY

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
