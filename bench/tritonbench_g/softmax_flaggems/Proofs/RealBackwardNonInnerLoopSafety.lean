import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerSafety

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton

theorem pass1_loop_safe (bounds : RegionBounds) (sb st : BlockState) (P G Y : RegionName)
    (M row N K kbase B C : Nat) (hN : 0<N) (hB : 0<B)
    (hr : row<M) (hP : M*N*K≤bounds P) (hG : M*N*K≤bounds G)
    (h : Pass1Inv sb P G row N K kbase B C 0 st) :
    Stmt.forRangeTraceSafe bounds "unused_start_n" 0 N B (pass1Body P G Y M N K B C) st := by
  suffices hs : Stmt.forRangeTraceSafe bounds "unused_start_n" (0*B) N B
      (pass1Body P G Y M N K B C) st by simpa only [Nat.zero_mul] using hs
  refine RealLoop.range_tiles_safe (P := fun c s ↦ Pass1Inv sb P G row N K kbase B C c s)
    hN hB ?_ ?_ 0 st (Nat.zero_le _) h
  · intro c s _ hs
    apply pass1_safe bounds P G Y M N K B C row kbase (c*B) _
      (laneSum sb P G row N K kbase B C c)
      (by simpa using hs.columnReg) (by simpa using hs.nReg)
      (by simpa using hs.offsetReg) (by simpa using hs.sumReg)
    · intro i hn hk
      exact Nat.lt_of_lt_of_le (RealAddress.non_inner_bound hr hn hk) hP
    · intro i hn hk
      exact Nat.lt_of_lt_of_le (RealAddress.non_inner_bound hr hn hk) hG
  · intro c s _ hs
    exact pass1_invariant_step sb s P G Y M row N K kbase B C c hs

theorem pass2_loop_safe (bounds : RegionBounds) (sb st : BlockState) (P G Y : RegionName)
    (M row N K kbase B C : Nat) (hN : 0<N) (hB : 0<B) (hPY : P≠Y) (hGY : G≠Y)
    (hr : row<M) (hP : M*N*K≤bounds P) (hG : M*N*K≤bounds G) (hY : M*N*K≤bounds Y)
    (h : Pass2Inv sb P G Y row N K kbase B C 0 st) :
    Stmt.forRangeTraceSafe bounds "unused_start_n" 0 N B (pass2Body P G Y M N K B C) st := by
  suffices hs : Stmt.forRangeTraceSafe bounds "unused_start_n" (0*B) N B
      (pass2Body P G Y M N K B C) st by simpa only [Nat.zero_mul] using hs
  refine RealLoop.range_tiles_safe (P := fun c s ↦ Pass2Inv sb P G Y row N K kbase B C c s)
    hN hB ?_ ?_ 0 st (Nat.zero_le _) h
  · intro c s _ hs
    apply pass2_safe bounds P G Y M N K B C row kbase (c*B) _
      (finalSum sb P G row N K kbase B C)
      (by simpa using hs.columnReg) (by simpa using hs.nReg)
      (by simpa using hs.offsetReg) (by simpa using hs.sumReg)
    · intro i hn hk
      exact Nat.lt_of_lt_of_le (RealAddress.non_inner_bound hr hn hk) hP
    · intro i hn hk
      exact Nat.lt_of_lt_of_le (RealAddress.non_inner_bound hr hn hk) hG
    · intro i hn hk
      exact Nat.lt_of_lt_of_le (RealAddress.non_inner_bound hr hn hk) hY
  · intro c s _ hs
    exact pass2_invariant_step sb s P G Y M row N K kbase B C c hB hPY hGY hs

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
