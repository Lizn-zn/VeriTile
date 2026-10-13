import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerScatter

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton

structure Pass2Inv (sb : BlockState) (Y X : RegionName)
    (row N B : Nat) (hB : 0 < B) (c : Nat) (st : BlockState) : Prop where
  frame : BlockState.WriteWithin (outputFootprint Y row N) sb st
  input : ∀ o, st.mem X o = sb.mem X o
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  rowReg : st.regs .nat [] "pid_m" = some (Tile.scalar row)
  inputReg : st.regs .ptr [] "input_ptr" = some (Tile.scalar (X, row * N))
  outputReg : st.regs .ptr [] "output_ptr" = some (Tile.scalar (Y, row * N))
  previousReg : st.regs .nat [] "previous_multiple" = some (Tile.scalar (RealAddress.previous N B))
  maxReg : st.regs .real [] "m" = some (Tile.scalar (finalMax sb X row N B hB))
  sumReg : st.regs .real [] "z" = some (Tile.scalar (some (finalSum sb X row N B hB)))
  values : ∀ n, n < N →
    RealAddress.rounds N B - c ≤ n / B →
    st.readMem Y (row * N + n) =
      expected sb X row N n

theorem scatter_invariant_step (sb mid : BlockState) (Y X : RegionName)
    (row N B c : Nat) (hN : 0 < N) (hB : 0 < B)
    (hXY : X ≠ Y) (hc : c < RealAddress.rounds N B)
    (h : Pass2Inv sb Y X row N B hB c mid) :
    Pass2Inv sb Y X row N B hB (c + 1)
      (scatter (B := B) mid Y (tileOffset row N (RealAddress.previous N B - c * B))
        (storeValue mid X row N (RealAddress.previous N B - c * B)
          (finalMax sb X row N B hB) (finalSum sb X row N B hB))
        (tileActive N (RealAddress.previous N B - c * B))) := by
  let base := RealAddress.previous N B - c * B
  let val : TileIndex [B] → ℝ := storeValue mid X row N base
    (finalMax sb X row N B hB) (finalSum sb X row N B hB)
  let next := scatter mid Y (tileOffset row N base) val (tileActive N base)
  have hframe : BlockState.WriteWithin (outputFootprint Y row N) mid next := by
    apply scatter_frame
    intro i hi
    exact ⟨rfl, base + i.1.val, of_decide_eq_true hi, rfl⟩
  have hregs := scatter_regs mid Y (tileOffset row N base) val (tileActive N base)
  have hlaunch := scatter_launch_padding mid Y (tileOffset row N base) val (tileActive N base)
  refine ⟨BlockState.writeWithin_trans h.frame hframe, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro o
    rw [scatter_preserve_input _ _ _ _ _ _ hXY]
    exact h.input o
  · exact (scatter_pids _ _ _ _ _).trans h.pids
  · exact hlaunch.1.trans h.numPids
  · exact hlaunch.2.trans h.undef
  · rw [hregs]; exact h.rowReg
  · rw [hregs]; exact h.inputReg
  · rw [hregs]; exact h.outputReg
  · rw [hregs]; exact h.previousReg
  · rw [hregs]; exact h.maxReg
  · rw [hregs]; exact h.sumReg
  · intro n hn hprocessed
    rcases RealAddress.reverse_new_or_old hc hprocessed with hnew | hold
    · obtain ⟨i, hi⟩ := RealAddress.active_reverse_lane hB hnew
      let ix : TileIndex [B] := (i, PUnit.unit)
      have hactive : tileActive N base ix = Bool.true := by
        apply decide_eq_true
        simpa [base, ix, ← hi] using hn
      have hvalue := scatter_value mid Y row N base val ix hactive
      have hformula := storeValue_expected sb mid X row N base hN hB
        (fun o ↦ by simp only [BlockState.readMem, h.input o]) ix hactive
      simpa only [next, val, tileOffset, ix, base, ← hi] using hvalue.trans hformula
    · have hcell := scatter_preserve_processed mid Y row N c n val hB hc hold
      have hread : next.readMem Y (row * N + n) = mid.readMem Y (row * N + n) := by
        unfold BlockState.readMem
        rw [hcell]
      exact hread.trans (h.values n hn hold)

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
