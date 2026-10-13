import bench.tritonbench_g.softmax_optimize.Proofs.SignedCandidate
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton FullSource
set_option maxHeartbeats 3000000

theorem integer_product_commutes (a b : Op .int []) (s : BlockState) :
    evalOp (.mul .int .nil a b) s = evalOp (.mul .int .nil b a) s := by
  simp only [evalOp_mul]
  cases evalOp a s <;> cases evalOp b s <;> try simp only [Option.bind_some, Option.bind_none]
  all_goals first | rfl | {
    apply congrArg some
    apply Tile.ext
    intro i
    exact Int.mul_comm _ _ }

theorem candidate_prefix_eq (X : RegionName) (N B : Nat) (s : BlockState) :
    stepStmts (indexPrefixCandidate X N B) s = stepStmts (indexPrefix X N B) s := by
  unfold indexPrefixCandidate indexPrefix
  simp only [stepStmts]
  cases stepStmt
    (.assign .int [B] "n_offsets"
      (.add .int .scalarL (.ref .int [] "start_n") (.castNatToInt (.arange B)))) s with
  | none => rfl
  | some t =>
    simp only [stepStmt, evalOp_add, integer_product_commutes (.constInt N)
      (.castNatToInt (.ref .nat [] "pid_m")) t]

theorem candidate_input_full_eq (X : RegionName) (N B : Nat) (s : BlockState) :
    stepStmts (inputFullCandidate X N B) s = stepStmts (inputFull X N B) s := by
  simp only [inputFullCandidate, inputFull, ProgramRewrite.append, candidate_prefix_eq]

theorem candidate_input_tail_eq (X : RegionName) (N B : Nat) (s : BlockState) :
    stepStmts (inputTailCandidate X N B) s = stepStmts (inputTail X N B) s := by
  simp only [inputTailCandidate, inputTail, ProgramRewrite.append, candidate_prefix_eq]

theorem candidate_output_full_eq (Y X : RegionName) (N B : Nat) (s : BlockState) :
    stepStmts (outputFullCandidate Y X N B) s = stepStmts (outputFull Y X N B) s := by
  simp only [outputFullCandidate, outputFull, ProgramRewrite.append, candidate_prefix_eq]

theorem candidate_output_tail_eq (Y X : RegionName) (N B : Nat) (s : BlockState) :
    stepStmts (outputTailCandidate Y X N B) s = stepStmts (outputTail Y X N B) s := by
  simp only [outputTailCandidate, outputTail, ProgramRewrite.append, candidate_prefix_eq]

theorem signed_range_congr (name : RegName) (cur stop : Int) (stride : Nat)
    (left right : List Stmt) (heq : ∀ s, stepStmts left s = stepStmts right s)
    (s : BlockState) :
    stepForRangeIntAux name cur stop stride left s =
      stepForRangeIntAux name cur stop stride right s := by
  by_cases hz : stride=0
  · subst stride
    rw [stepForRangeIntAux.step_zero_step, stepForRangeIntAux.step_zero_step]
  by_cases hc : cur<stop
  · rw [stepForRangeIntAux.step_lt hz hc, stepForRangeIntAux.step_lt hz hc, heq]
    cases stepStmts right (s.setReg name .int [] (Tile.scalar cur)) with
    | none => rfl
    | some t => exact signed_range_congr name (cur+stride) stop stride left right heq t
  · rw [stepForRangeIntAux.step_ge hz (Int.le_of_not_gt hc),
      stepForRangeIntAux.step_ge hz (Int.le_of_not_gt hc)]
termination_by (stop-cur).toNat
decreasing_by omega

theorem signed_loop_congr (name : RegName) (start stop : Op .int []) (stride : Op .nat [])
    (left right : List Stmt) (heq : ∀ s, stepStmts left s = stepStmts right s)
    (s : BlockState) :
    stepStmt (.forRangeInt name start stop stride left) s =
      stepStmt (.forRangeInt name start stop stride right) s := by
  simp only [stepStmt]
  cases evalOp start s <;> cases evalOp stop s <;> cases evalOp stride s <;>
    try simp only [Option.bind_some, Option.bind_none]
  all_goals first | rfl | exact signed_range_congr _ _ _ _ _ _ heq _


/-- Candidate equality covers complete state and failure, including negative
starts when N=0; no real or floating numerical identity is used. -/
theorem candidate_exec_eq (input : Storage) (output : Accumulation)
    (Y X : RegionName) (M N B : Nat) (s : BlockState) :
    exec (kernel input output Bool.true Y X M N B).eraseDType.toAlgKernel s =
      exec (kernel input output Bool.false Y X M N B).eraseDType.toAlgKernel s := by
  rw [projection, projection]
  unfold exec
  rw [candidate_source_body, source_body]
  have ha := signed_loop_congr "start_n" (.constInt 0) (.ref .int [] "prev_multiple") (.constNat B)
    _ _ (candidate_input_full_eq X N B)
  have hb := signed_loop_congr "start_n" (.ref .int [] "prev_multiple") (.constInt N) (.constNat B)
    _ _ (candidate_input_tail_eq X N B)
  have hc := signed_loop_congr "start_n" (.constInt 0) (.ref .int [] "prev_multiple") (.constNat B)
    _ _ (candidate_output_full_eq Y X N B)
  have hd := signed_loop_congr "start_n" (.ref .int [] "prev_multiple") (.constInt N) (.constNat B)
    _ _ (candidate_output_tail_eq Y X N B)
  simp only [ProgramRewrite.append, stepStmts, ha, hb, hc, hd]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
