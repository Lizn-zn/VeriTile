import bench.tritonbench_g.softmax_optimize.Proofs.SafetyOutput

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
open VeriTile.Triton SignedTransport

theorem output_full_safe (bounds : RegionBounds) (Y X : RegionName) (N B i p : Nat)
    (s : BlockState) (hpid : s.regs .nat [] "pid_m" = some (Tile.scalar p))
    (hX : ∀ lane : Fin B, p * N + (i + lane.val) < bounds X)
    (hY : ∀ lane : Fin B, p * N + (i + lane.val) < bounds Y) :
    Stmt.TraceSafeList bounds (outputFull Y X N B)
      (s.setReg "start_n" .int [] (Tile.scalar (i : Int))) := by
  unfold outputFull
  simp only [List.append_assoc]
  apply TraceProofs.append_safe _ _ (index_prefix_safe bounds X N B _)
  intro t ht
  have heq := Option.some.inj ((prefix_int_run X N B i p s hpid).symm.trans ht)
  subst t
  change Stmt.TraceSafeList bounds
    [.assign .real [B] "inp" (.load .real (.ptr (.ref .ptr [B] "input_ptrs")) .none),
     .assign .real [B] "e" (.exp (.sub .real .scalarR
       (.ref .real [B] "inp") (.ref .real [] "m"))),
     .assign .real [B] "out" (.div .real .scalarR
       (.ref .real [B] "e") (.ref .real [] "z")),
     outputPointer Y B,
     .store .real [B] (.ptr (.ref .ptr [B] "output_ptrs")) (.ref .real [B] "out") .none]
    (prefixIntState X N B i p s)
  apply TraceProofs.assign_cons
  · exact full_load_safe bounds X N B i p _ (by simp [prefixIntState]) hX
  · intro inp
    apply TraceProofs.assign_cons
    · simp [Op.SafeAt.eq_def]
    · intro e
      apply TraceProofs.assign_cons
      · simp [Op.SafeAt.eq_def]
      · intro out
        exact full_output_end_safe bounds Y N B i p _ (by simp [prefixIntState]) hY

theorem output_tail_safe (bounds : RegionBounds) (Y X : RegionName) (N B i p : Nat)
    (s : BlockState) (hpid : s.regs .nat [] "pid_m" = some (Tile.scalar p))
    (hX : ∀ lane : Fin B, i + lane.val < N → p * N + (i + lane.val) < bounds X)
    (hY : ∀ lane : Fin B, i + lane.val < N → p * N + (i + lane.val) < bounds Y) :
    Stmt.TraceSafeList bounds (outputTail Y X N B)
      (s.setReg "start_n" .int [] (Tile.scalar (i : Int))) := by
  unfold outputTail
  simp only [List.append_assoc]
  apply TraceProofs.append_safe _ _ (index_prefix_safe bounds X N B _)
  intro t ht
  have heq := Option.some.inj ((prefix_int_run X N B i p s hpid).symm.trans ht)
  subst t
  change Stmt.TraceSafeList bounds
    (maskStatement N B ::
     [.assign .real [B] "inp" (.load .real (.ptr (.ref .ptr [B] "input_ptrs"))
        (.maskOther (.ref .bool [B] "mask") (Op.negInf.broadcast [B]))),
      .assign .real [B] "e" (.exp (.sub .real .scalarR
        (.ref .real [B] "inp") (.ref .real [] "m"))),
      .assign .real [B] "out" (.div .real .scalarR
        (.ref .real [B] "e") (.ref .real [] "z")),
      outputPointer Y B,
      .store .real [B] (.ptr (.ref .ptr [B] "output_ptrs")) (.ref .real [B] "out")
        (.mask (.ref .bool [B] "mask"))]) (prefixIntState X N B i p s)
  rw [TraceProofs.cons_of_run_iff (mask_run X N B i p s)]
  constructor
  · simp [maskStatement, Stmt.TraceSafe, Op.SafeAt.eq_def]
  · apply TraceProofs.assign_cons
    · exact tail_load_safe bounds X N B i p _ (by simp [prefixIntState]) (by simp) hX
    · intro inp
      apply TraceProofs.assign_cons
      · simp [Op.SafeAt.eq_def]
      · intro e
        apply TraceProofs.assign_cons
        · simp [Op.SafeAt.eq_def]
        · intro out
          exact tail_output_end_safe bounds Y N B i p _
            (by simp [prefixIntState]) (by simp) hY

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
