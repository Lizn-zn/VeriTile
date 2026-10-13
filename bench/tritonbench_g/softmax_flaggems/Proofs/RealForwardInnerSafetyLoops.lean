import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerSafetyStep

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton

 theorem range_aligned_safe
    {bounds : RegionBounds} {idx : RegName} {q B : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} (hB : 0 < B)
    (hsafe : ∀ c s, c < q → P c s →
      Stmt.TraceSafeList bounds body (s.setReg idx .nat [] (Tile.scalar (c * B))))
    (hstep : ∀ c s, c < q → P c s →
      ∃ s', stepStmts body (s.setReg idx .nat [] (Tile.scalar (c * B))) = some s' ∧
        P (c + 1) s') :
    ∀ c s, c ≤ q → P c s →
      Stmt.forRangeTraceSafe bounds idx (c * B) (q * B) B body s := by
  intro c s
  induction h : (q - c) using Nat.strong_induction_on generalizing c s with
  | _ remaining ih =>
    intro hc hp
    by_cases hdone : c = q
    · subst c
      rw [Stmt.forRangeTraceSafe, if_neg (Nat.ne_of_gt hB), if_neg (Nat.lt_irrefl _)]
      trivial
    · have hlt : c < q := by omega
      obtain ⟨next, hbody, hnext⟩ := hstep c s hlt hp
      rw [Stmt.forRangeTraceSafe, if_neg (Nat.ne_of_gt hB),
        if_pos (Nat.mul_lt_mul_of_pos_right hlt hB)]
      refine ⟨hsafe c s hlt hp, ?_⟩
      rw [hbody]
      have ht := ih (q - (c + 1)) (by omega) (c + 1) next rfl (by omega) hnext
      simpa only [Nat.add_mul, Nat.one_mul] using ht

 theorem row_address_bound {row M N n : Nat} (hr : row < M) (hn : n < N) :
    row * N + n < M * N := by
  simpa using RealAddress.non_inner_bound hr (show 0 < 1 by omega) hn

 theorem full_loop_safe (bounds : RegionBounds) (sb st : BlockState) (Y X : RegionName)
    (M row N B : Nat) (hN : 0 < N) (hB : 0 < B)
    (hr : row < M) (hX : M * N ≤ bounds X)
    (h : Pass1Inv sb X Y row N B 0 st) :
    Stmt.forRangeTraceSafe bounds "start_n" 0 (RealAddress.previous N B) B
      (fullBody Y X M N B) st := by
  rw [RealAddress.previous_eq]
  have hfull : ∀ c, c < RealAddress.rounds N B - 1 → c * B + B ≤ N := by
    intro c hc
    have hl := RealAddress.previous_lt hN hB
    rw [RealAddress.previous_eq] at hl
    have hm := Nat.mul_le_mul_right B (show c + 1 ≤ RealAddress.rounds N B - 1 by omega)
    simp only [Nat.add_mul, Nat.one_mul] at hm
    omega
  have hs := range_aligned_safe (bounds := bounds) (idx := "start_n")
    (q := RealAddress.rounds N B - 1) (P := fun c s ↦ Pass1Inv sb X Y row N B c s)
    hB (fun c s hc hs ↦ ?_) (fun c s hc hs ↦
      full_invariant_step sb s Y X M row N B c (hfull c hc) hs) 0 st (Nat.zero_le _) h
  · simpa only [Nat.zero_mul] using hs
  · apply full_safe bounds Y X M N B row (c * B) _
      (fun i ↦ (laneState sb X row N B c i).1)
      (fun i ↦ (laneState sb X row N B c i).2)
      (by simpa using hs.rowReg) (by simpa using hs.inputReg)
      (by simp) (by simpa using hs.maxReg) (by simpa using hs.sumReg)
    intro i
    exact Nat.lt_of_lt_of_le (row_address_bound hr (by have := i.1.isLt; have := hfull c hc; omega)) hX

 theorem tail_loop_safe (bounds : RegionBounds) (sb st : BlockState) (Y X : RegionName)
    (M row N B : Nat) (hN : 0 < N) (hB : 0 < B)
    (hr : row < M) (hX : M * N ≤ bounds X)
    (h : Pass1Inv sb X Y row N B (RealAddress.rounds N B - 1) st) :
    Stmt.forRangeTraceSafe bounds "start_n" (RealAddress.previous N B) N B
      (pass1Body Y X M N B) st := by
  rw [RealAddress.previous_eq]
  refine RealLoop.range_tiles_safe (P := fun c s ↦ Pass1Inv sb X Y row N B c s)
    hN hB ?_ ?_ _ st (Nat.sub_le _ _) h
  · intro c s _ hs
    apply pass1_safe bounds Y X M N B row (c * B) _
      (fun i ↦ (laneState sb X row N B c i).1)
      (fun i ↦ (laneState sb X row N B c i).2)
      (by simpa using hs.rowReg) (by simpa using hs.inputReg)
      (by simp) (by simpa using hs.maxReg) (by simpa using hs.sumReg)
    intro i hn
    exact Nat.lt_of_lt_of_le (row_address_bound hr hn) hX
  · intro c s _ hs
    exact pass1_invariant_step sb s Y X M row N B c hs

 theorem masked_invariant_safe (bounds : RegionBounds) (sb st : BlockState) (Y X : RegionName)
    (M row N B c : Nat) (hB : 0 < B)
    (hr : row < M) (hX : M * N ≤ bounds X) (hY : M * N ≤ bounds Y)
    (h : Pass2Inv sb Y X row N B hB c st) :
    Stmt.TraceSafeList bounds (pass2Body Y X M N B)
      (st.setReg "start_n" .nat [] (Tile.scalar (c * B))) := by
  apply pass2_safe bounds Y X M N B row (RealAddress.previous N B) (c * B) _
    (finalMax sb X row N B hB) (finalSum sb X row N B hB)
    (by simpa using h.rowReg) (by simpa using h.inputReg) (by simp)
    (by simpa using h.previousReg) (by simpa using h.outputReg)
    (by simpa using h.maxReg) (by simpa using h.sumReg)
  · intro i hn
    exact Nat.lt_of_lt_of_le (row_address_bound hr hn) hX
  · intro i hn
    exact Nat.lt_of_lt_of_le (row_address_bound hr hn) hY

 theorem full_store_invariant_safe (bounds : RegionBounds) (sb st : BlockState) (Y X : RegionName)
    (M row N B c : Nat) (hN : 0 < N) (hB : 0 < B) (hc : 0 < c) (hcr : c < RealAddress.rounds N B)
    (hr : row < M) (hX : M * N ≤ bounds X) (hY : M * N ≤ bounds Y)
    (h : Pass2Inv sb Y X row N B hB c st) :
    Stmt.TraceSafeList bounds (fullStoreBody Y X M N B)
      (st.setReg "start_n" .nat [] (Tile.scalar (c * B))) := by
  have hfull : RealAddress.previous N B - c * B + B ≤ N := by
    rw [RealAddress.reverse_base]
    have hlast := RealAddress.previous_lt hN hB
    rw [RealAddress.previous_eq] at hlast
    have hmul := Nat.mul_le_mul_right B
      (show RealAddress.rounds N B - 1 - c + 1 ≤ RealAddress.rounds N B - 1 by omega)
    simp only [Nat.add_mul, Nat.one_mul] at hmul
    omega
  apply full_store_safe bounds Y X M N B row (RealAddress.previous N B) (c * B) _
    (finalMax sb X row N B hB) (finalSum sb X row N B hB)
    (by simpa using h.rowReg) (by simpa using h.inputReg) (by simp)
    (by simpa using h.previousReg) (by simpa using h.outputReg)
    (by simpa using h.maxReg) (by simpa using h.sumReg)
  · intro i
    exact Nat.lt_of_lt_of_le (row_address_bound hr (by have := i.1.isLt; omega)) hX
  · intro i
    exact Nat.lt_of_lt_of_le (row_address_bound hr (by have := i.1.isLt; omega)) hY

 theorem masked_write_loop_safe (bounds : RegionBounds) (sb st : BlockState) (Y X : RegionName)
    (M row N B : Nat) (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y)
    (hr : row < M) (hX : M * N ≤ bounds X) (hY : M * N ≤ bounds Y)
    (h : Pass2Inv sb Y X row N B hB 0 st) :
    Stmt.forRangeTraceSafe bounds "start_n" 0 B B (pass2Body Y X M N B) st := by
  obtain ⟨next, he, _⟩ := masked_invariant_step sb st Y X M row N B 0 hN hB hXY
    (RealAddress.rounds_positive hN hB) h
  rw [Stmt.forRangeTraceSafe, if_neg (Nat.ne_of_gt hB), if_pos hB]
  refine ⟨?_, ?_⟩
  · simpa only [Nat.zero_mul] using masked_invariant_safe bounds sb st Y X M row N B 0 hB hr hX hY h
  · simp only [Nat.zero_mul] at he
    rw [he]
    simp only [Nat.zero_add]
    rw [Stmt.forRangeTraceSafe, if_neg (Nat.ne_of_gt hB), if_neg (Nat.lt_irrefl B)]
    trivial

 theorem full_write_loop_safe (bounds : RegionBounds) (sb st : BlockState) (Y X : RegionName)
    (M row N B : Nat) (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y)
    (hr : row < M) (hX : M * N ≤ bounds X) (hY : M * N ≤ bounds Y)
    (h : Pass2Inv sb Y X row N B hB 1 st) :
    Stmt.forRangeTraceSafe bounds "start_n" B N B (fullStoreBody Y X M N B) st := by
  have hs := RealLoop.range_tiles_safe
    (P := fun c s ↦ Pass2Inv sb Y X row N B hB c s ∧ 0 < c) hN hB
    (fun c s hc hs ↦ full_store_invariant_safe bounds sb s Y X M row N B c hN hB hs.2 hc hr hX hY hs.1)
    (fun c s hc hs ↦ ?_) 1 st (RealAddress.rounds_positive hN hB) ⟨h, by omega⟩
  · simpa only [Nat.one_mul] using hs
  · obtain ⟨next, he, hn⟩ := full_store_invariant_step sb s Y X M row N B c hN hB hXY hc hs.2 hs.1
    exact ⟨next, he, hn, by omega⟩

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner

#guard_msgs (drop info) in
#auditModuleAxioms
