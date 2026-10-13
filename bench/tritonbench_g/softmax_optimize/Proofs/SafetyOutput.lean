import bench.tritonbench_g.softmax_optimize.Proofs.SafetyInput

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
open VeriTile.Triton SignedTransport

theorem output_pointer_run (Y : RegionName) (N B i p : Nat) (s : BlockState)
    (hoff : s.regs .int [B] "offset" = some (castIndex (offsets N B i p))) :
    stepStmt (outputPointer Y B) s =
      some (s.setReg "output_ptrs" .ptr [B] (pointers Y N B i p)) := by
  apply stepStmt_assign_eq_some
  simp only [evalOp.eq_def, hoff, Option.bind_some, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro lane
  change (Y, 0 + Int.toNat ((p * N + (i + lane.1.val) : Nat) : Int)) =
    (Y, p * N + (i + lane.1.val))
  rw [Int.toNat_natCast, Nat.zero_add]

theorem full_output_end_safe (bounds : RegionBounds) (Y : RegionName)
    (N B i p : Nat) (s : BlockState)
    (hoff : s.regs .int [B] "offset" = some (castIndex (offsets N B i p)))
    (hb : ∀ lane : Fin B, p * N + (i + lane.val) < bounds Y) :
    Stmt.TraceSafeList bounds [outputPointer Y B,
      .store .real [B] (.ptr (.ref .ptr [B] "output_ptrs"))
        (.ref .real [B] "out") .none] s := by
  rw [TraceProofs.cons_of_run_iff (output_pointer_run Y N B i p s hoff)]
  constructor
  · simp [outputPointer, Stmt.TraceSafe, Op.SafeAt.eq_def]
  · apply Stmt.TraceSafeList.cons_intro
    · apply pointer_store_safe bounds B "output_ptrs" "out" _ (pointers Y N B i p)
        (by simp)
      intro lane
      exact hb lane.1
    · intro t _
      exact Stmt.TraceSafeList.nil_intro

theorem tail_output_end_safe (bounds : RegionBounds) (Y : RegionName)
    (N B i p : Nat) (s : BlockState)
    (hoff : s.regs .int [B] "offset" = some (castIndex (offsets N B i p)))
    (hm : s.regs .bool [B] "mask" = some (Tile.vec fun lane ↦ decide (i + lane.val < N)))
    (hb : ∀ lane : Fin B, i + lane.val < N → p * N + (i + lane.val) < bounds Y) :
    Stmt.TraceSafeList bounds [outputPointer Y B,
      .store .real [B] (.ptr (.ref .ptr [B] "output_ptrs"))
        (.ref .real [B] "out") (.mask (.ref .bool [B] "mask"))] s := by
  rw [TraceProofs.cons_of_run_iff (output_pointer_run Y N B i p s hoff)]
  constructor
  · simp [outputPointer, Stmt.TraceSafe, Op.SafeAt.eq_def]
  · apply Stmt.TraceSafeList.cons_intro
    · apply masked_pointer_store_safe bounds B "output_ptrs" "out" "mask" _
        (pointers Y N B i p) (Tile.vec fun lane ↦ decide (i + lane.val < N))
        (by simp) (by simpa using hm)
      intro lane hactive
      exact hb lane.1 (of_decide_eq_true hactive)
    · intro t _
      exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
