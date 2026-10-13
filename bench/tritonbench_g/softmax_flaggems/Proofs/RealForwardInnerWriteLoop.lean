import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerWriteback
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerFullStore

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton

 theorem Pass2Inv.setStart {sb st : BlockState} {Y X : RegionName}
    {row N B c start : Nat} {hB : 0 < B} (h : Pass2Inv sb Y X row N B hB c st) :
    Pass2Inv sb Y X row N B hB c (st.setReg "start_n" .nat [] (Tile.scalar start)) := by
  refine ⟨h.frame, h.input, h.pids, h.numPids, h.undef, ?_, ?_, ?_, ?_, ?_, ?_, h.values⟩
  · simpa using h.rowReg
  · simpa using h.inputReg
  · simpa using h.outputReg
  · simpa using h.previousReg
  · simpa using h.maxReg
  · simpa using h.sumReg

 theorem Pass2Inv.storeMid {sb st : BlockState} {Y X : RegionName}
    {row N B c base : Nat} {hB : 0 < B} (mx : WithBot ℝ) (z : ℝ)
    (h : Pass2Inv sb Y X row N B hB c st) :
    Pass2Inv sb Y X row N B hB c (storeMid st X row N base B mx z) := by
  refine ⟨h.frame, h.input, h.pids, h.numPids, h.undef, ?_, ?_, ?_, ?_, ?_, ?_, h.values⟩
  · simpa [RealForwardInner.storeMid] using h.rowReg
  · simpa [RealForwardInner.storeMid] using h.inputReg
  · simpa [RealForwardInner.storeMid] using h.outputReg
  · simpa [RealForwardInner.storeMid] using h.previousReg
  · simpa [RealForwardInner.storeMid] using h.maxReg
  · simpa [RealForwardInner.storeMid] using h.sumReg

 theorem Pass2Inv.fullStoreMid {sb st : BlockState} {Y X : RegionName}
    {row N B c base : Nat} {hB : 0 < B} (mx : WithBot ℝ) (z : ℝ)
    (h : Pass2Inv sb Y X row N B hB c st) :
    Pass2Inv sb Y X row N B hB c (fullStoreMid st X row N base B mx z) := by
  refine ⟨h.frame, h.input, h.pids, h.numPids, h.undef, ?_, ?_, ?_, ?_, ?_, ?_, h.values⟩
  · simpa [RealForwardInner.fullStoreMid] using h.rowReg
  · simpa [RealForwardInner.fullStoreMid] using h.inputReg
  · simpa [RealForwardInner.fullStoreMid] using h.outputReg
  · simpa [RealForwardInner.fullStoreMid] using h.previousReg
  · simpa [RealForwardInner.fullStoreMid] using h.maxReg
  · simpa [RealForwardInner.fullStoreMid] using h.sumReg

 theorem masked_invariant_step (sb st : BlockState) (Y X : RegionName)
    (M row N B c : Nat) (hN : 0 < N) (hB : 0 < B)
    (hXY : X ≠ Y) (hc : c < RealAddress.rounds N B)
    (h : Pass2Inv sb Y X row N B hB c st) :
    ∃ next, stepStmts (pass2Body Y X M N B)
        (st.setReg "start_n" .nat [] (Tile.scalar (c * B))) = some next ∧
      Pass2Inv sb Y X row N B hB (c + 1) next := by
  let st' := st.setReg "start_n" .nat [] (Tile.scalar (c * B))
  have hs : Pass2Inv sb Y X row N B hB c st' := h.setStart
  have he := pass2_step Y X M N B row (RealAddress.previous N B) (c * B) st'
    (finalMax sb X row N B hB) (finalSum sb X row N B hB)
    hs.inputReg hs.outputReg hs.previousReg (by simp [st']) hs.maxReg hs.sumReg
  refine ⟨_, he, ?_⟩
  exact scatter_invariant_step sb _ Y X row N B c hN hB hXY hc
    (hs.storeMid (finalMax sb X row N B hB) (finalSum sb X row N B hB))

 theorem full_store_invariant_step (sb st : BlockState) (Y X : RegionName)
    (M row N B c : Nat) (hN : 0 < N) (hB : 0 < B)
    (hXY : X ≠ Y) (hc : c < RealAddress.rounds N B) (hc0 : 0 < c)
    (h : Pass2Inv sb Y X row N B hB c st) :
    ∃ next, stepStmts (fullStoreBody Y X M N B)
        (st.setReg "start_n" .nat [] (Tile.scalar (c * B))) = some next ∧
      Pass2Inv sb Y X row N B hB (c + 1) next := by
  let st' := st.setReg "start_n" .nat [] (Tile.scalar (c * B))
  have hs : Pass2Inv sb Y X row N B hB c st' := h.setStart
  have hfull : RealAddress.previous N B - c * B + B ≤ N := by
    rw [RealAddress.reverse_base]
    have hlast := RealAddress.previous_lt hN hB
    rw [RealAddress.previous_eq] at hlast
    have hmul := Nat.mul_le_mul_right B (show RealAddress.rounds N B - 1 - c + 1 ≤ RealAddress.rounds N B - 1 by omega)
    simp only [Nat.add_mul, Nat.one_mul] at hmul
    omega
  have he := fullStore_step Y X M N B row (RealAddress.previous N B) (c * B) st'
    (finalMax sb X row N B hB) (finalSum sb X row N B hB) hfull
    hs.inputReg hs.outputReg hs.previousReg (by simp [st']) hs.maxReg hs.sumReg
  refine ⟨_, he, ?_⟩
  exact scatter_invariant_step sb _ Y X row N B c hN hB hXY hc
    (hs.fullStoreMid (finalMax sb X row N B hB) (finalSum sb X row N B hB))


theorem masked_write_loop (sb st : BlockState) (Y X : RegionName)
    (M row N B : Nat) (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y)
    (h : Pass2Inv sb Y X row N B hB 0 st) :
    ∃ next, stepForRangeAux "start_n" 0 B B (pass2Body Y X M N B) st = some next ∧
      Pass2Inv sb Y X row N B hB 1 next := by
  obtain ⟨next, hexec, hn⟩ := masked_invariant_step sb st Y X M row N B 0 hN hB hXY
    (RealAddress.rounds_positive hN hB) h
  refine ⟨next, ?_, hn⟩
  rw [stepForRangeAux.step_lt (Nat.ne_of_gt hB) hB]
  have he : stepStmts (pass2Body Y X M N B) (st.setReg "start_n" .nat [] (Tile.scalar 0)) = some next := by
    simpa only [Nat.zero_mul] using hexec
  rw [he]
  simp only [Option.bind_some, Nat.zero_add, stepForRangeAux.step_ge (Nat.ne_of_gt hB) (Nat.le_refl B)]

theorem full_write_loop (sb st : BlockState) (Y X : RegionName)
    (M row N B : Nat) (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y)
    (h : Pass2Inv sb Y X row N B hB 1 st) :
    ∃ next, stepForRangeAux "start_n" B N B (fullStoreBody Y X M N B) st = some next ∧
      Pass2Inv sb Y X row N B hB (RealAddress.rounds N B) next := by
  have step : ∀ c s, c < RealAddress.rounds N B →
      (Pass2Inv sb Y X row N B hB c s ∧ 0 < c) →
      ∃ next, stepStmts (fullStoreBody Y X M N B) (s.setReg "start_n" .nat [] (Tile.scalar (c * B))) = some next ∧
        (Pass2Inv sb Y X row N B hB (c + 1) next ∧ 0 < c + 1) := by
    intro c s hc hs
    obtain ⟨next, he, hn⟩ := full_store_invariant_step sb s Y X M row N B c hN hB hXY hc hs.2 hs.1
    exact ⟨next, he, hn, by omega⟩
  obtain ⟨next, he, hn, _⟩ := RealLoop.range_tiles hN hB step 1 st
    (RealAddress.rounds_positive hN hB) ⟨h, by omega⟩
  exact ⟨next, by simpa only [Nat.one_mul] using he, hn⟩

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
