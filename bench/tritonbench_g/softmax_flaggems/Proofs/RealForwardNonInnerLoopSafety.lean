import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerSafety

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton

theorem pass1_loop_safe (bounds : RegionBounds) (sb st : BlockState) (Y X : RegionName)
    (M row N K kbase B C : Nat) (hN : 0 < N) (hB : 0 < B)
    (hr : row < M) (hX : M * N * K ≤ bounds X)
    (h : Pass1Inv sb X row N K kbase B C 0 st) :
    Stmt.forRangeTraceSafe bounds "start_n" 0 N B (pass1Body Y X M N K B C) st := by
  suffices hs : Stmt.forRangeTraceSafe bounds "start_n" (0 * B) N B
      (pass1Body Y X M N K B C) st by simpa only [Nat.zero_mul] using hs
  refine RealLoop.range_tiles_safe (P := fun c s ↦ Pass1Inv sb X row N K kbase B C c s)
    hN hB ?_ ?_ 0 st (Nat.zero_le _) h
  · intro c s _ hs
    apply pass1_safe bounds Y X M N K B C row kbase (c * B) _
      (fun i ↦ (laneState sb X row N K kbase B C c i).1)
      (fun i ↦ (laneState sb X row N K kbase B C c i).2)
      (by simpa using hs.rowReg) (by simpa using hs.columnReg)
      (by simp) (by simpa using hs.maxReg) (by simpa using hs.sumReg)
    intro i hn hk
    exact Nat.lt_of_lt_of_le (tile_access_bound hr i (by simpa [tileActive] using And.intro hn hk)) hX
  · intro c s _ hs
    exact pass1_invariant_step sb s Y X M row N K kbase B C c hs

theorem pass2_loop_safe (bounds : RegionBounds) (sb st : BlockState) (Y X : RegionName)
    (M row N K kbase B C : Nat) (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y)
    (hr : row < M) (hX : M * N * K ≤ bounds X) (hY : M * N * K ≤ bounds Y)
    (h : Pass2Inv sb Y X row N K kbase B C hB 0 st) :
    Stmt.forRangeTraceSafe bounds "start_n" 0 N B (pass2Body Y X M N K B C) st := by
  suffices hs : Stmt.forRangeTraceSafe bounds "start_n" (0 * B) N B
      (pass2Body Y X M N K B C) st by simpa only [Nat.zero_mul] using hs
  refine RealLoop.range_tiles_safe (P := fun c s ↦ Pass2Inv sb Y X row N K kbase B C hB c s)
    hN hB ?_ ?_ 0 st (Nat.zero_le _) h
  · intro c s _ hs
    apply pass2_safe bounds Y X M N K B C row kbase (RealAddress.previous N B) (c * B) _
      (finalMax sb X row N K kbase B C hB) (finalSum sb X row N K kbase B C hB)
      (by simpa using hs.rowReg) (by simpa using hs.columnReg) (by simp)
      (by simpa using hs.previousReg) (by simpa using hs.maxReg) (by simpa using hs.sumReg)
    · intro i hn hk
      exact Nat.lt_of_lt_of_le (tile_access_bound hr i (by simpa [tileActive] using And.intro hn hk)) hX
    · intro i hn hk
      exact Nat.lt_of_lt_of_le (tile_access_bound hr i (by simpa [tileActive] using And.intro hn hk)) hY
  · intro c s hc hs
    exact pass2_invariant_step sb s Y X M row N K kbase B C c hN hB hXY hc hs

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
