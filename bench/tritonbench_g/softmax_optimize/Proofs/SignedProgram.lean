import bench.tritonbench_g.softmax_optimize.Proofs.SignedBodyRelation
import bench.tritonbench_g.softmax_optimize.Proofs.SignedPrevious

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton FullSource
set_option maxHeartbeats 4000000

def signedRest (Y X : RegionName) (N B : Nat) : List Stmt :=
  [.forRangeInt "start_n" (.constInt 0) (.ref .int [] "prev_multiple") (.constNat B) (inputFull X N B),
   .forRangeInt "start_n" (.ref .int [] "prev_multiple") (.constInt N) (.constNat B) (inputTail X N B)] ++
  reductionStmts B ++
  [.assign .int [] "prev_multiple" (previousExpression N B),
   .forRangeInt "start_n" (.constInt 0) (.ref .int [] "prev_multiple") (.constNat B) (outputFull Y X N B),
   .forRangeInt "start_n" (.ref .int [] "prev_multiple") (.constInt N) (.constNat B) (outputTail Y X N B)]

def naturalRest (Y X : RegionName) (N B : Nat) : List Stmt :=
  [.forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "prev_multiple") (.constNat B) (loopAbody X N B),
   .forRangeDyn "start_n" (.ref .nat [] "prev_multiple") (.constNat N) (.constNat B) (loopBbody X N B)] ++
  reductionStmts B ++
  [.assign .nat [] "prev_multiple" (naturalPrevious N B),
   .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "prev_multiple") (.constNat B) (loopApass2body Y X N B),
   .forRangeDyn "start_n" (.ref .nat [] "prev_multiple") (.constNat N) (.constNat B) (loopBpass2body Y X N B)]

theorem live_append {a b c d : List Stmt}
    (h₁ : LiveStatementsRelated a b) (h₂ : LiveStatementsRelated c d) :
    LiveStatementsRelated (a++c) (b++d) := by
  intro s t h
  rw [ProgramRewrite.append, ProgramRewrite.append]
  exact results_bind (h₁ s t h) _ _ h₂

theorem rest_related (Y X : RegionName) (N B : Nat) (hN : 0<N) (hB : 0<B) :
    LiveStatementsRelated (signedRest Y X N B) (naturalRest Y X N B) := by
  unfold signedRest naturalRest
  apply live_append
  · apply live_append
    · exact live_cons (full_range_related B _ _ (input_full_related X N B))
        (live_cons (tail_range_related N B _ _ (input_tail_related X N B)) live_nil)
    · exact reduction_related B
  · exact live_cons (previous_related N B hN hB)
      (live_cons (full_range_related B _ _ (output_full_related Y X N B))
        (live_cons (tail_range_related N B _ _ (output_tail_related Y X N B)) live_nil))

def numericalSeed (B : Nat) (s : BlockState) : BlockState :=
  ((s.setReg "pid_m" .nat [] (Tile.scalar s.pid)).setReg "m" .real [B]
    ⟨fun _ ↦ (⊥ : WithBot ℝ)⟩).setReg "z" .real [B] ⟨fun _ ↦ ((0:ℝ):WithBot ℝ)⟩

theorem source_positive_related (Y X : RegionName) (M N B : Nat)
    (hN : 0<N) (hB : 0<B) (s : BlockState) : ResultsRelated LiveRelated
      (exec (mathematical Bool.false Y X M N B) s)
      (exec (softmax_kernel_online_v2_surface Y X M N B).toAlgKernel s) := by
  unfold exec
  rw [source_body]
  change ResultsRelated LiveRelated
    (stepStmts
      ([.assign .nat [] "pid_m" (.programId 0),
        .assign .real [B] "m" (.full [B] .negInf),
        .assign .real [B] "z" (.full [B] (.const 0)),
        .assign .int [] "prev_multiple" (previousExpression N B)] ++ signedRest Y X N B) s)
    (stepStmts
      ([.assign .nat [] "pid_m" (.programId 0),
        .assign .real [B] "m" (.full [B] .negInf),
        .assign .real [B] "z" (.full [B] (.const 0)),
        .assign .nat [] "prev_multiple" (naturalPrevious N B)] ++ naturalRest Y X N B) s)
  simp only [List.cons_append, List.nil_append]
  have hp : stepStmt (.assign .nat [] "pid_m" (.programId 0)) s =
      some (s.setReg "pid_m" .nat [] (Tile.scalar s.pid)) := by
    apply stepStmt_assign_eq_some
    exact evalOp_programId _ _
  rw [stepStmts.cons_some hp, stepStmts.cons_some hp]
  have hm : ∀ st : BlockState, stepStmt (.assign .real [B] "m" (.full [B] .negInf)) st =
      some (st.setReg "m" .real [B] ⟨fun _ ↦ (⊥ : WithBot ℝ)⟩) := by
    intro st
    apply stepStmt_assign_eq_some
    simp only [evalOp_full, evalOp_negInf]
    rfl
  rw [stepStmts.cons_some (hm _), stepStmts.cons_some (hm _)]
  have hz : ∀ st : BlockState, stepStmt (.assign .real [B] "z" (.full [B] (.const 0))) st =
      some (st.setReg "z" .real [B] ⟨fun _ ↦ ((0:ℝ):WithBot ℝ)⟩) := by
    intro st
    apply stepStmt_assign_eq_some
    simp only [evalOp_full, evalOp_const]
    rfl
  rw [stepStmts.cons_some (hz _), stepStmts.cons_some (hz _)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (previous_eval N B hN hB _)),
    stepStmts.cons_some (stepStmt_assign_eq_some (natural_previous_eval N B _))]
  exact rest_related Y X N B hN hB _ _ (seed_previous _ _)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
