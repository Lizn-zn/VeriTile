import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerScatter

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton

def outputFootprint (Y : RegionName) (row N K kbase C : Nat) : WriteFootprint :=
  fun a ↦ a.1 = Y ∧ ∃ n k : Nat,
    n < N ∧ k < C ∧ kbase + k < K ∧
      a.2 = row * N * K + n * K + (kbase + k)

structure Pass2Inv (sb : BlockState) (Y X : RegionName)
    (row N K kbase B C : Nat) (hB : 0 < B) (c : Nat) (st : BlockState) : Prop where
  frame : BlockState.WriteWithin (outputFootprint Y row N K kbase C) sb st
  input : ∀ o, st.mem X o = sb.mem X o
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  rowReg : st.regs .nat [] "pid_m" = some (Tile.scalar row)
  columnReg : st.regs .nat [C] "k_offsets" = some ⟨fun i ↦ kbase + i.1.val⟩
  previousReg : st.regs .nat [] "previous_multiple" = some (Tile.scalar (RealAddress.previous N B))
  maxReg : st.regs .real [C] "m" = some ⟨fun i ↦ finalMax sb X row N K kbase B C hB i.1⟩
  sumReg : st.regs .real [C] "z" = some ⟨fun i ↦ some (finalSum sb X row N K kbase B C hB i.1)⟩
  values : ∀ n k, n < N → k < C → kbase + k < K →
    RealAddress.rounds N B - c ≤ n / B →
    st.readMem Y (row * N * K + n * K + (kbase + k)) =
      expected sb X row N K n (kbase + k)

theorem pass2_invariant_step (sb st : BlockState) (Y X : RegionName)
    (M row N K kbase B C c : Nat) (hN : 0 < N) (hB : 0 < B)
    (hXY : X ≠ Y) (hc : c < RealAddress.rounds N B)
    (h : Pass2Inv sb Y X row N K kbase B C hB c st) :
    ∃ next, stepStmts (pass2Body Y X M N K B C)
        (st.setReg "start_n" .nat [] (Tile.scalar (c * B))) = some next ∧
      Pass2Inv sb Y X row N K kbase B C hB (c + 1) next := by
  let st' := st.setReg "start_n" .nat [] (Tile.scalar (c * B))
  let base := RealAddress.previous N B - c * B
  let val : TileIndex [B, C] → ℝ := storeValue st' X row N K kbase base
    (finalMax sb X row N K kbase B C hB) (finalSum sb X row N K kbase B C hB)
  obtain ⟨mid, hexec, hmem, hpids, hnum, hundef, hrow, hk, hprev, hm, hz⟩ :=
    pass2_step Y X M N K B C row kbase (RealAddress.previous N B) (c * B)
      st' (finalMax sb X row N K kbase B C hB) (finalSum sb X row N K kbase B C hB)
      (by simpa [st'] using h.rowReg) (by simpa [st'] using h.columnReg)
      (by simp [st']) (by simpa [st'] using h.previousReg)
      (by simpa [st'] using h.maxReg) (by simpa [st'] using h.sumReg)
  let next := scatter mid Y (tileOffset row N K kbase base) val (tileActive N K kbase base)
  have hframe : BlockState.WriteWithin (outputFootprint Y row N K kbase C) mid next := by
    apply scatter_frame
    intro i hi
    have hi' : base + i.1.val < N ∧ kbase + i.2.1.val < K := of_decide_eq_true hi
    exact ⟨rfl, base + i.1.val, i.2.1.val, hi'.1, i.2.1.isLt, hi'.2, rfl⟩
  have hmidmem : mid.mem = st.mem := hmem
  have hmidframe : BlockState.WriteWithin (outputFootprint Y row N K kbase C) sb mid := by
    intro r o ho
    rw [hmidmem]
    exact h.frame r o ho
  have hregs := scatter_regs mid Y (tileOffset row N K kbase base) val (tileActive N K kbase base)
  have hlaunch := scatter_launch_padding mid Y (tileOffset row N K kbase base) val
    (tileActive N K kbase base)
  refine ⟨next, hexec, ?_⟩
  refine ⟨BlockState.writeWithin_trans hmidframe hframe, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro o
    rw [scatter_preserve_input _ _ _ _ _ _ hXY, hmidmem]
    exact h.input o
  · exact (scatter_pids _ _ _ _ _).trans (hpids.trans h.pids)
  · exact hlaunch.1.trans (hnum.trans h.numPids)
  · exact hlaunch.2.trans (hundef.trans h.undef)
  · rw [hregs, hrow]
    simpa [st'] using h.rowReg
  · rw [hregs, hk]
    simpa [st'] using h.columnReg
  · rw [hregs, hprev]
    simpa [st'] using h.previousReg
  · rw [hregs, hm]
    simpa [st'] using h.maxReg
  · rw [hregs, hz]
    simpa [st'] using h.sumReg
  · intro n k hn hkC hkK hprocessed
    rcases RealAddress.reverse_new_or_old hc hprocessed with hnew | hold
    · obtain ⟨i, hi⟩ := RealAddress.active_reverse_lane hB hnew
      let ix : TileIndex [B, C] := (i, ⟨k, hkC⟩, PUnit.unit)
      have hactive : tileActive N K kbase base ix = Bool.true := by
        apply decide_eq_true
        exact ⟨by simpa [base, ix, ← hi] using hn, hkK⟩
      have hvalue := scatter_value mid Y row N K kbase base val ix hactive
      have hformula := storeValue_expected sb st' X row N K kbase base hN hB
        (fun o ↦ by simp only [st', BlockState.setReg_readMem, BlockState.readMem, BlockState.setReg_mem, h.input o]) ix hactive
      simpa only [next, val, tileOffset, ix, base, ← hi] using hvalue.trans hformula
    · have hcell := scatter_preserve_processed mid Y row N K kbase c n (kbase + k)
        val hB hc hold hkK
      have hread : next.readMem Y (row * N * K + n * K + (kbase + k)) =
          st.readMem Y (row * N * K + n * K + (kbase + k)) := by
        unfold BlockState.readMem
        rw [hcell, hmidmem]
      exact hread.trans (h.values n k hn hkC hkK hold)

theorem pass2_loop (sb st : BlockState) (Y X : RegionName)
    (M row N K kbase B C : Nat) (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y)
    (h : Pass2Inv sb Y X row N K kbase B C hB 0 st) :
    ∃ next, stepForRangeAux "start_n" 0 N B (pass2Body Y X M N K B C) st = some next ∧
      Pass2Inv sb Y X row N K kbase B C hB (RealAddress.rounds N B) next := by
  simpa only [Nat.zero_mul] using RealLoop.range_tiles hN hB
    (fun c s hc hs ↦ pass2_invariant_step sb s Y X M row N K kbase B C c hN hB hXY hc hs)
    0 st (Nat.zero_le _) h

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
