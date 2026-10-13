import bench.tritonbench_g.softmax_optimize.Proofs.SafetyPure
import bench.tritonbench_g.softmax_optimize.Proofs.SafetyAccess
import bench.tritonbench_g.softmax_optimize.Proofs.SignedPrefix

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
open VeriTile.Triton SignedTransport

theorem full_load_safe (bounds : RegionBounds) (X : RegionName) (N B i p : Nat)
    (s : BlockState)
    (hptr : s.regs .ptr [B] "input_ptrs" = some (pointers X N B i p))
    (hbound : ∀ lane : Fin B, p * N + (i + lane.val) < bounds X) :
    (Op.load .real (.ptr (.ref .ptr [B] "input_ptrs")) .none).SafeAt bounds s := by
  apply pointer_load_safe bounds B "input_ptrs" s (pointers X N B i p) hptr
  intro lane
  exact hbound lane.1

theorem tail_load_safe (bounds : RegionBounds) (X : RegionName) (N B i p : Nat)
    (s : BlockState)
    (hptr : s.regs .ptr [B] "input_ptrs" = some (pointers X N B i p))
    (hmask : s.regs .bool [B] "mask" =
      some (Tile.vec fun lane ↦ decide (i + lane.val < N)))
    (hbound : ∀ lane : Fin B, i + lane.val < N → p * N + (i + lane.val) < bounds X) :
    (Op.load .real (.ptr (.ref .ptr [B] "input_ptrs"))
      (.maskOther (.ref .bool [B] "mask") (Op.negInf.broadcast [B]))).SafeAt bounds s := by
  apply masked_pointer_load_safe bounds B "input_ptrs" "mask" s
    (pointers X N B i p) _ hptr hmask
  intro lane hactive
  exact hbound lane.1 (of_decide_eq_true hactive)

/-- Every full input-loop body is safe from its address facts alone; the
numerical registers are unrestricted, including missing-register failure. -/
theorem input_full_safe (bounds : RegionBounds) (X : RegionName) (N B i p : Nat)
    (s : BlockState)
    (hpid : s.regs .nat [] "pid_m" = some (Tile.scalar p))
    (hbound : ∀ lane : Fin B, p * N + (i + lane.val) < bounds X) :
    Stmt.TraceSafeList bounds (inputFull X N B)
      (s.setReg "start_n" .int [] (Tile.scalar (i : Int))) := by
  unfold inputFull
  apply TraceProofs.append_safe _ _ (index_prefix_safe bounds X N B _)
  intro t ht
  have heq := Option.some.inj ((prefix_int_run X N B i p s hpid).symm.trans ht)
  subst t
  change Stmt.TraceSafeList bounds
    (.assign .real [B] "inp" (.load .real (.ptr (.ref .ptr [B] "input_ptrs")) .none) ::
      (loopAbody X N B).drop 4) (prefixIntState X N B i p s)
  apply TraceProofs.assign_cons
  · exact full_load_safe bounds X N B i p _ (by simp [prefixIntState]) hbound
  · intro value
    exact input_full_update_safe bounds X N B _

theorem mask_run (X : RegionName) (N B i p : Nat) (s : BlockState) :
    stepStmt (maskStatement N B) (prefixIntState X N B i p s) =
      some ((prefixIntState X N B i p s).setReg "mask" .bool [B]
        (Tile.vec fun lane ↦ decide (i + lane.val < N))) := by
  apply stepStmt_assign_eq_some
  simp only [maskStatement, evalOp.eq_def, prefixIntState,
    BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
    not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro lane
  change decide (((i + lane.1.val : Nat) : Int) < (N : Int)) =
    decide (i + lane.1.val < N)
  simp only [Int.ofNat_lt]

/-- Tail safety uses the source's signed mask and checks only active lanes. -/
theorem input_tail_safe (bounds : RegionBounds) (X : RegionName) (N B i p : Nat)
    (s : BlockState)
    (hpid : s.regs .nat [] "pid_m" = some (Tile.scalar p))
    (hbound : ∀ lane : Fin B, i + lane.val < N → p * N + (i + lane.val) < bounds X) :
    Stmt.TraceSafeList bounds (inputTail X N B)
      (s.setReg "start_n" .int [] (Tile.scalar (i : Int))) := by
  rw [inputTail, List.append_assoc]
  apply TraceProofs.append_safe _ _ (index_prefix_safe bounds X N B _)
  intro t ht
  have heq := Option.some.inj ((prefix_int_run X N B i p s hpid).symm.trans ht)
  subst t
  change Stmt.TraceSafeList bounds
    (maskStatement N B :: (loopBbody X N B).drop 4) (prefixIntState X N B i p s)
  rw [TraceProofs.cons_of_run_iff (mask_run X N B i p s)]
  constructor
  · simp [maskStatement, Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def]
  · change Stmt.TraceSafeList bounds
      (.assign .real [B] "inp" (.load .real (.ptr (.ref .ptr [B] "input_ptrs"))
        (.maskOther (.ref .bool [B] "mask") (Op.negInf.broadcast [B]))) ::
        (loopBbody X N B).drop 5) _
    apply TraceProofs.assign_cons
    · exact tail_load_safe bounds X N B i p _ (by simp [prefixIntState])
        (by simp) hbound
    · intro value
      exact input_tail_update_safe bounds X N B _

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
