import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerReduce
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerInitialize

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource

structure ReducedInv (sb : BlockState) (X Y : RegionName)
    (row N B : Nat) (hB : 0 < B) (st : BlockState) : Prop where
  memory : st.mem = sb.mem
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  rowReg : st.regs .nat [] "pid_m" = some (Tile.scalar row)
  inputReg : st.regs .ptr [] "input_ptr" = some (Tile.scalar (X, row * N))
  outputReg : st.regs .ptr [] "output_ptr" = some (Tile.scalar (Y, row * N))
  previousReg : st.regs .nat [] "previous_multiple" = some (Tile.scalar (RealAddress.previous N B))
  maxReg : st.regs .real [] "m" = some (Tile.scalar (finalMax sb X row N B hB))
  sumReg : st.regs .real [] "z" = some (Tile.scalar (some (finalSum sb X row N B hB)))

/-- The complete accumulation and reduction extracted from the actual source. -/
def prepareBody (Y X : RegionName) (M N B : Nat) : List Stmt :=
  firstPass Y X M N B ++ reductionBody Y X M N B

theorem prepare_run (s : BlockState) (Y X : RegionName) (M N B : Nat)
    (hN : 0 < N) (hB : 0 < B) :
    ∃ next, stepStmts (prepareBody Y X M N B) s = some next ∧
      ReducedInv s X Y (s.pids 0) N B hB next := by
  obtain ⟨st, hfirst, hinv⟩ := firstPass_run s Y X M N B hN hB
  obtain ⟨next, hreduce, hmem, hpids, hnum, hundef, hrow, hp, ho, hprev, hm, hz⟩ :=
    reduction_step Y X M N B hB st _ _ hinv.maxReg hinv.sumReg
  refine ⟨next, ?_, ?_⟩
  · rw [prepareBody, ProgramRewrite.append, hfirst]
    exact hreduce
  · exact ⟨hmem.trans hinv.memory, hpids.trans hinv.pids, hnum.trans hinv.numPids,
      hundef.trans hinv.undef, hrow.trans hinv.rowReg, hp.trans hinv.inputReg, ho.trans hinv.outputReg, hprev.trans hinv.previousReg, hm, hz⟩

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
