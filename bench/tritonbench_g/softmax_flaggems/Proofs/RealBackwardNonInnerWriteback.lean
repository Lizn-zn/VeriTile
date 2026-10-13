import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerStore
import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerPrepare
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerWriteback

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton
open scoped BigOperators
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

abbrev outputFootprint := RealForwardNonInner.outputFootprint

noncomputable def expected (sb : BlockState) (P G : RegionName)
    (row N K padded n k : Nat) : ℝ :=
  RealMath.backwardPadded padded
    (fun j ↦ sample sb P row N K j k) (fun j ↦ sample sb G row N K j k) n

theorem storeValue_expected {B C : Nat} (sb st : BlockState) (P G : RegionName)
    (row N K kbase start : Nat) (hB : 0 < B)
    (hp : ∀ o, st.readMem P o = sb.readMem P o)
    (hg : ∀ o, st.readMem G o = sb.readMem G o)
    (i : TileIndex [B,C]) (hi : tileActive N K kbase start i = Bool.true) :
    storeValue st P G row N K kbase start (finalSum sb P G row N K kbase B C) i =
      expected sb P G row N K (RealAddress.rounds N B*B) (start+i.1.val)
        (kbase+i.2.1.val) := by
  have hi' : start+i.1.val<N ∧ kbase+i.2.1.val<K := of_decide_eq_true hi
  simp only [storeValue, tileSample, finalSum_eq sb P G row N K kbase B C hB,
    expected, RealMath.backwardPadded]
  simp only [sample, if_pos hi', hp, hg]

structure Pass2Inv (sb : BlockState) (P G Y : RegionName)
    (row N K kbase B C c : Nat) (st : BlockState) : Prop where
  frame : BlockState.WriteWithin (outputFootprint Y row N K kbase C) sb st
  inputP : ∀ o, st.mem P o = sb.mem P o
  inputG : ∀ o, st.mem G o = sb.mem G o
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  columnReg : st.regs .nat [C] "offsets_k" = some ⟨fun i ↦ kbase+i.1.val⟩
  nReg : st.regs .nat [B] "offsets_n" = some ⟨fun i ↦ c*B+i.1.val⟩
  offsetReg : st.regs .nat [B,C] "offsets" = some ⟨tileOffset row N K kbase (c*B)⟩
  sumReg : st.regs .real [C] "scale" =
    some ⟨fun i ↦ some (finalSum sb P G row N K kbase B C i.1)⟩
  values : ∀ n k, n < N → k < C → kbase+k<K → n / B < c →
    st.readMem Y (offset row N K n (kbase+k)) =
      expected sb P G row N K (RealAddress.rounds N B*B) n (kbase+k)

theorem scatter_preserve_processed {B C : Nat} (s : BlockState) (Y : RegionName)
    (row N K kbase c n k : Nat) (value : TileIndex [B,C] → ℝ)
    (hB : 0 < B) (hold : n/B<c) (hk : k<K) :
    (scatter s Y (tileOffset row N K kbase (c*B)) value
      (tileActive N K kbase (c*B))).mem Y (offset row N K n k) =
      s.mem Y (offset row N K n k) := by
  apply foldl_store_preserve_cell _ value (fun i ↦ tileActive N K kbase (c*B) i = Bool.true)
  intro i _ hi heq
  have hi' : c*B+i.1.val<N ∧ kbase+i.2.1.val<K := of_decide_eq_true hi
  have ha := heq.2
  unfold tileOffset offset at ha
  have hcanc : (c*B+i.1.val)*K+(kbase+i.2.1.val)=n*K+k := by omega
  have he := (RealAddress.row_column_injective hi'.2 hk hcanc).1
  have hsmall := (Nat.div_lt_iff_lt_mul hB).mp hold
  omega

theorem pass2_invariant_step (sb st : BlockState) (P G Y : RegionName)
    (M row N K kbase B C c : Nat) (hB : 0 < B) (hPY : P ≠ Y) (hGY : G ≠ Y)
    (h : Pass2Inv sb P G Y row N K kbase B C c st) :
    ∃ next, stepStmts (pass2Body P G Y M N K B C)
      (st.setReg "unused_start_n" .nat [] (Tile.scalar (c*B))) = some next ∧
      Pass2Inv sb P G Y row N K kbase B C (c+1) next := by
  let st' := st.setReg "unused_start_n" .nat [] (Tile.scalar (c*B))
  let z := finalSum sb P G row N K kbase B C
  let mid := storeMid (B := B) st' P G row N K kbase (c*B) z
  let value : TileIndex [B,C] → ℝ := storeValue st' P G row N K kbase (c*B) z
  let sc := scatter mid Y (tileOffset row N K kbase (c*B)) value (tileActive N K kbase (c*B))
  let next := advance (B := B) (C := C) sc row N K kbase (c*B)
  have he := pass2_step P G Y M N K B C row kbase (c*B) st' z
    (by simpa [st'] using h.columnReg) (by simpa [st'] using h.nReg)
    (by simpa [st'] using h.offsetReg) (by simpa [st', z] using h.sumReg)
  have hf : BlockState.WriteWithin (outputFootprint Y row N K kbase C) mid sc := by
    apply RealForwardNonInner.scatter_frame
    intro i hi
    have hi' : c*B+i.1.val<N ∧ kbase+i.2.1.val<K := of_decide_eq_true hi
    exact ⟨rfl, c*B+i.1.val, i.2.1.val, hi'.1, i.2.1.isLt, hi'.2, rfl⟩
  have hm : mid.mem=st.mem := rfl
  have hr := RealForwardNonInner.scatter_regs mid Y (tileOffset row N K kbase (c*B)) value
    (tileActive N K kbase (c*B))
  have hl := RealForwardNonInner.scatter_launch_padding mid Y
    (tileOffset row N K kbase (c*B)) value (tileActive N K kbase (c*B))
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
  · intro n k hn hkC hkK hpast
    by_cases hold : n/B<c
    · have hc := scatter_preserve_processed mid Y row N K kbase c n (kbase+k) value hB hold hkK
      have hr : next.readMem Y (offset row N K n (kbase+k)) =
          st.readMem Y (offset row N K n (kbase+k)) := by
        unfold BlockState.readMem
        simp only [next, advance, BlockState.setReg_mem, sc, hc, hm]
      exact hr.trans (h.values n k hn hkC hkK hold)
    · have heq : n/B=c := by omega
      let i : TileIndex [B,C] := (⟨n%B, Nat.mod_lt n hB⟩, ⟨k,hkC⟩, PUnit.unit)
      have hn' : c*B+n%B=n := by rw [← heq]; simpa [Nat.mul_comm] using Nat.div_add_mod n B
      have ha : tileActive N K kbase (c*B) i = Bool.true := by
        apply decide_eq_true
        exact ⟨by simpa [i, hn'] using hn, hkK⟩
      have hv := RealForwardNonInner.scatter_value mid Y row N K kbase (c*B) value i ha
      have hv' := storeValue_expected sb st' P G row N K kbase (c*B) hB
        (fun o ↦ by simp only [st', BlockState.setReg_readMem, BlockState.readMem, BlockState.setReg_mem, h.inputP o])
        (fun o ↦ by simp only [st', BlockState.setReg_readMem, BlockState.readMem, BlockState.setReg_mem, h.inputG o]) i ha
      simpa only [next, advance, BlockState.setReg_readMem, sc, value, RealForwardNonInner.tileOffset,
        tileOffset, i, hn', offset]
        using hv.trans hv'

theorem pass2_loop (sb st : BlockState) (P G Y : RegionName)
    (M row N K kbase B C : Nat) (hN : 0<N) (hB : 0<B) (hPY : P≠Y) (hGY : G≠Y)
    (h : Pass2Inv sb P G Y row N K kbase B C 0 st) :
    ∃ next, stepForRangeAux "unused_start_n" 0 N B (pass2Body P G Y M N K B C) st = some next ∧
      Pass2Inv sb P G Y row N K kbase B C (RealAddress.rounds N B) next := by
  simpa only [Nat.zero_mul] using RealLoop.range_tiles hN hB
    (fun c s _ hs ↦ pass2_invariant_step sb s P G Y M row N K kbase B C c hB hPY hGY hs)
    0 st (Nat.zero_le _) h

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
