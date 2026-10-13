import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardInnerScatter
import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardInnerPrepare

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def outputFootprint (Y : RegionName) (M N rbase C : Nat) : WriteFootprint :=
  fun a ↦ a.1=Y ∧ ∃ r n, r<C ∧ rbase+r<M ∧ n<N ∧ a.2=offset N (rbase+r) n

noncomputable def expected (s : BlockState) (P G : RegionName)
    (M N padded r n : Nat) : ℝ :=
  RealMath.backwardPadded padded (fun j ↦ sample s P M N r j)
    (fun j ↦ sample s G M N r j) n

theorem storeValue_expected {B C : Nat} (sb st : BlockState) (P G : RegionName)
    (M N rbase start : Nat) (hB : 0<B)
    (hp : ∀ o, st.readMem P o=sb.readMem P o)
    (hg : ∀ o, st.readMem G o=sb.readMem G o)
    (i : TileIndex [C,B]) (hi : tileActive M N rbase start i=Bool.true) :
    storeValue st P G M N rbase start (finalSum sb P G M N rbase B C) i =
      expected sb P G M N (RealAddress.rounds N B*B) (rbase+i.1.val) (start+i.2.1.val) := by
  have hi' : rbase+i.1.val<M ∧ start+i.2.1.val<N := of_decide_eq_true hi
  simp only [storeValue, tileSample, finalSum_eq sb P G M N rbase B C hB,
    expected, RealMath.backwardPadded]
  simp only [sample, if_pos hi', hp, hg]

structure Pass2Inv (sb : BlockState) (P G Y : RegionName)
    (M N rbase B C c : Nat) (st : BlockState) : Prop where
  frame : BlockState.WriteWithin (outputFootprint Y M N rbase C) sb st
  inputP : ∀ o, st.mem P o=sb.mem P o
  inputG : ∀ o, st.mem G o=sb.mem G o
  pids : st.pids=sb.pids
  numPids : st.numPids=sb.numPids
  undef : st.undef=sb.undef
  columnReg : st.regs .nat [C] "m_offsets" = some ⟨fun i ↦ rbase+i.1.val⟩
  nReg : st.regs .nat [B] "n_offsets" = some ⟨fun i ↦ c*B+i.1.val⟩
  offsetReg : st.regs .nat [C,B] "offsets" = some ⟨tileOffset N rbase (c*B)⟩
  sumReg : st.regs .real [C] "scale" = some ⟨fun i ↦ some (finalSum sb P G M N rbase B C i.1)⟩
  values : ∀ r n, r<C → rbase+r<M → n<N → n/B<c →
    st.readMem Y (offset N (rbase+r) n) =
      expected sb P G M N (RealAddress.rounds N B*B) (rbase+r) n

theorem pass2_invariant_step (sb st : BlockState) (P G Y : RegionName)
    (M N rbase B C c : Nat) (hB : 0<B) (hPY : P≠Y) (hGY : G≠Y)
    (h : Pass2Inv sb P G Y M N rbase B C c st) :
    ∃ next, stepStmts (pass2Body P G Y M N B C)
      (st.setReg "unused_start_n" .nat [] (Tile.scalar (c*B))) = some next ∧
      Pass2Inv sb P G Y M N rbase B C (c+1) next := by
  let st' := st.setReg "unused_start_n" .nat [] (Tile.scalar (c*B))
  let z := finalSum sb P G M N rbase B C
  let mid := storeMid (B := B) st' P G M N rbase (c*B) z
  let value : TileIndex [C,B] → ℝ := storeValue st' P G M N rbase (c*B) z
  let sc := scatter mid Y (tileOffset N rbase (c*B)) value (tileActive M N rbase (c*B))
  let next := advance (B := B) (C := C) sc M N rbase (c*B)
  have he := pass2_step P G Y M N B C rbase (c*B) st' z
    (by simpa [st'] using h.columnReg) (by simpa [st'] using h.nReg)
    (by simpa [st'] using h.offsetReg) (by simpa [st', z] using h.sumReg)
  have hf : BlockState.WriteWithin (outputFootprint Y M N rbase C) mid sc := by
    apply RealForwardNonInner.scatter_frame
    intro i hi
    have hi' : rbase+i.1.val<M ∧ c*B+i.2.1.val<N := of_decide_eq_true hi
    exact ⟨rfl, i.1.val, c*B+i.2.1.val, i.1.isLt, hi'.1, hi'.2, rfl⟩
  have hm : mid.mem=st.mem := rfl
  have hr := RealForwardNonInner.scatter_regs mid Y (tileOffset N rbase (c*B)) value
    (tileActive M N rbase (c*B))
  have hl := RealForwardNonInner.scatter_launch_padding mid Y
    (tileOffset N rbase (c*B)) value (tileActive M N rbase (c*B))
  refine ⟨next, he, ?_⟩
  constructor
  · intro r o ho
    exact (h.frame r o ho).trans (hf r o ho)
  · intro o
    exact (RealForwardNonInner.scatter_preserve_input mid Y P _ value _ hPY o).trans (h.inputP o)
  · intro o
    exact (RealForwardNonInner.scatter_preserve_input mid Y G _ value _ hGY o).trans (h.inputG o)
  · exact (RealForwardNonInner.scatter_pids mid Y _ value _).trans h.pids
  · exact hl.1.trans h.numPids
  · exact hl.2.trans h.undef
  · simp only [next, advance, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true]
    rw [hr]
    simpa [mid, storeMid, st'] using h.columnReg
  · simp [next, advance, Nat.add_mul, Nat.one_mul]
  · simp [next, advance, Nat.add_mul, Nat.one_mul]
  · simp only [next, advance, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true]
    rw [hr]
    simpa [mid, storeMid, st', z] using h.sumReg
  · intro r n hrC hrM hn hpast
    by_cases hold : n/B<c
    · have hc := scatter_preserve_processed mid Y M N rbase c (rbase+r) n value hB hold hn
      have hread : next.readMem Y (offset N (rbase+r) n) = st.readMem Y (offset N (rbase+r) n) := by
        unfold BlockState.readMem
        simp only [next, advance, BlockState.setReg_mem, sc, hc, hm]
      exact hread.trans (h.values r n hrC hrM hn hold)
    · have heq : n/B=c := by omega
      let i : TileIndex [C,B] := (⟨r,hrC⟩, ⟨n%B,Nat.mod_lt n hB⟩, PUnit.unit)
      have hn' : c*B+n%B=n := by rw [←heq]; simpa [Nat.mul_comm] using Nat.div_add_mod n B
      have ha : tileActive M N rbase (c*B) i=Bool.true := by
        apply decide_eq_true
        exact ⟨hrM, by simpa [i, hn'] using hn⟩
      have hv := scatter_value mid Y M N rbase (c*B) value i ha
      have hv' := storeValue_expected sb st' P G M N rbase (c*B) hB
        (fun o ↦ by simp only [st', BlockState.setReg_readMem, BlockState.readMem,
          BlockState.setReg_mem, h.inputP o])
        (fun o ↦ by simp only [st', BlockState.setReg_readMem, BlockState.readMem,
          BlockState.setReg_mem, h.inputG o]) i ha
      simpa only [next, advance, BlockState.setReg_readMem, sc, value, tileOffset, i, hn', offset]
        using hv.trans hv'

theorem pass2_loop (sb st : BlockState) (P G Y : RegionName)
    (M N rbase B C : Nat) (hN : 0<N) (hB : 0<B) (hPY : P≠Y) (hGY : G≠Y)
    (h : Pass2Inv sb P G Y M N rbase B C 0 st) :
    ∃ next, stepForRangeAux "unused_start_n" 0 N B (pass2Body P G Y M N B C) st = some next ∧
      Pass2Inv sb P G Y M N rbase B C (RealAddress.rounds N B) next := by
  simpa only [Nat.zero_mul] using RealLoop.range_tiles hN hB
    (fun c s _ hs ↦ pass2_invariant_step sb s P G Y M N rbase B C c hB hPY hGY hs)
    0 st (Nat.zero_le _) h

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
