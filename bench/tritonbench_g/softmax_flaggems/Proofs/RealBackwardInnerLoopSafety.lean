import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardInnerSafety

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton

theorem pass1_loop_safe (bounds : RegionBounds) (sb st : BlockState) (P G Y : RegionName)
    (M N rbase B C : Nat) (hN : 0<N) (hB : 0<B)
    (hP : M*N≤bounds P) (hG : M*N≤bounds G)
    (h : Pass1Inv sb P G M N rbase B C 0 st) :
    Stmt.forRangeTraceSafe bounds "unused_start_n" 0 N B (pass1Body P G Y M N B C) st := by
  suffices hs : Stmt.forRangeTraceSafe bounds "unused_start_n" (0*B) N B
      (pass1Body P G Y M N B C) st by simpa only [Nat.zero_mul] using hs
  refine RealLoop.range_tiles_safe (P := fun c s ↦ Pass1Inv sb P G M N rbase B C c s)
    hN hB ?_ ?_ 0 st (Nat.zero_le _) h
  · intro c s _ hs
    apply pass1_safe bounds P G Y M N B C rbase (c*B) _
      (laneSum sb P G M N rbase B C c)
      (by simpa using hs.columnReg) (by simpa using hs.nReg)
      (by simpa using hs.offsetReg) (by simpa using hs.sumReg)
    · intro i hn hk
      exact Nat.lt_of_lt_of_le (tile_access_bound i (by exact decide_eq_true (And.intro hn hk))) hP
    · intro i hn hk
      exact Nat.lt_of_lt_of_le (tile_access_bound i (by exact decide_eq_true (And.intro hn hk))) hG
  · intro c s _ hs
    exact pass1_invariant_step sb s P G Y M N rbase B C c hs

theorem pass2_loop_safe (bounds : RegionBounds) (sb st : BlockState) (P G Y : RegionName)
    (M N rbase B C : Nat) (hN : 0<N) (hB : 0<B) (hPY : P≠Y) (hGY : G≠Y)
    (hP : M*N≤bounds P) (hG : M*N≤bounds G) (hY : M*N≤bounds Y)
    (h : Pass2Inv sb P G Y M N rbase B C 0 st) :
    Stmt.forRangeTraceSafe bounds "unused_start_n" 0 N B (pass2Body P G Y M N B C) st := by
  suffices hs : Stmt.forRangeTraceSafe bounds "unused_start_n" (0*B) N B
      (pass2Body P G Y M N B C) st by simpa only [Nat.zero_mul] using hs
  refine RealLoop.range_tiles_safe (P := fun c s ↦ Pass2Inv sb P G Y M N rbase B C c s)
    hN hB ?_ ?_ 0 st (Nat.zero_le _) h
  · intro c s _ hs
    apply pass2_safe bounds P G Y M N B C rbase (c*B) _
      (finalSum sb P G M N rbase B C)
      (by simpa using hs.columnReg) (by simpa using hs.nReg)
      (by simpa using hs.offsetReg) (by simpa using hs.sumReg)
    · intro i hn hk
      exact Nat.lt_of_lt_of_le (tile_access_bound i (by exact decide_eq_true (And.intro hn hk))) hP
    · intro i hn hk
      exact Nat.lt_of_lt_of_le (tile_access_bound i (by exact decide_eq_true (And.intro hn hk))) hG
    · intro i hn hk
      exact Nat.lt_of_lt_of_le (tile_access_bound i (by exact decide_eq_true (And.intro hn hk))) hY
  · intro c s _ hs
    exact pass2_invariant_step sb s P G Y M N rbase B C c hB hPY hGY hs

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
