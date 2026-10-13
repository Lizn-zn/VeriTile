import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerDenominator

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource

noncomputable def finalMax (s : BlockState) (X : RegionName)
    (row N K kbase B C : Nat) (hB : 0 < B) : Fin C → WithBot ℝ :=
  columnMax hB (fun i ↦ (laneState s X row N K kbase B C (RealAddress.rounds N B) i).1)

noncomputable def finalSum (s : BlockState) (X : RegionName)
    (row N K kbase B C : Nat) (hB : 0 < B) : Fin C → ℝ :=
  columnSum
    (fun i ↦ (laneState s X row N K kbase B C (RealAddress.rounds N B) i).1)
    (fun i ↦ (laneState s X row N K kbase B C (RealAddress.rounds N B) i).2)
    (finalMax s X row N K kbase B C hB)

structure ReducedInv (sb : BlockState) (X : RegionName)
    (row N K kbase B C : Nat) (hB : 0 < B) (st : BlockState) : Prop where
  memory : st.mem = sb.mem
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  rowReg : st.regs .nat [] "pid_m" = some (Tile.scalar row)
  columnReg : st.regs .nat [C] "k_offsets" = some ⟨fun i ↦ kbase + i.1.val⟩
  maxReg : st.regs .real [C] "m" = some ⟨fun i ↦ finalMax sb X row N K kbase B C hB i.1⟩
  sumReg : st.regs .real [C] "z" = some ⟨fun i ↦ some (finalSum sb X row N K kbase B C hB i.1)⟩

/-- The complete accumulation and reduction extracted from the actual source. -/
def prepareBody (Y X : RegionName) (M N K B C : Nat) : List Stmt :=
  firstPass Y X M N K B C ++ reductionBody Y X M N K B C

theorem prepare_run (s : BlockState) (Y X : RegionName) (M N K B C : Nat)
    (hN : 0 < N) (hB : 0 < B) :
    ∃ next, stepStmts (prepareBody Y X M N K B C) s = some next ∧
      ReducedInv s X (s.pids 0) N K (s.pids 1 * C) B C hB next := by
  obtain ⟨st, hfirst, hinv⟩ := firstPass_run s Y X M N K B C hN hB
  obtain ⟨next, hreduce, hmem, hpids, hnum, hundef, hrow, hk, hm, hz⟩ :=
    reduction_step Y X M N K B C hB st _ _ hinv.maxReg hinv.sumReg
  refine ⟨next, ?_, ?_⟩
  · rw [prepareBody, ProgramRewrite.append, hfirst]
    exact hreduce
  · exact ⟨hmem.trans hinv.memory, hpids.trans hinv.pids, hnum.trans hinv.numPids,
      hundef.trans hinv.undef, hrow.trans hinv.rowReg, hk.trans hinv.columnReg, hm, hz⟩

theorem final_denominator (s : BlockState) (X : RegionName)
    (row N K kbase B C : Nat) (hN : 0 < N) (hB : 0 < B)
    (k : Fin C) (hk : kbase + k.val < K) :
    ∃ m : ℝ, finalMax s X row N K kbase B C hB k = some m ∧
      finalSum s X row N K kbase B C hB k =
        Real.exp (-m) * denominator s X row N K (kbase + k.val) :=
  denominator_closed s X row N K kbase B C (RealAddress.rounds N B) hN hB
    (RealAddress.rounds_positive hN hB) (RealAddress.coverage hB) k hk

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
