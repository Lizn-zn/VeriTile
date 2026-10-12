import FlashAttnDraft.Formula
import VeriTile.Triton.Memory.FlattenTolerant
namespace VeriTile.Bench.TritonBenchG.FlashAttn.WitnessData
open VeriTile.Triton Formula
noncomputable section
set_option maxHeartbeats 2000000

/-- Non-contiguous layout; `SEQLEN=3` with `BLOCK_N=2` deliberately exercises
the unmasked final tile. This file checks data and formulas; the full finite
execution theorem must additionally be connected to the real IO contract. -/
def layout (causal : Bool) : Layout where
  stride_q_head := 20
  stride_q_seqlen := 5
  stride_q_dim := 2
  stride_k_seqlen := 7
  stride_k_dim := 2
  stride_v_dim := 3
  stride_o_seqlen := 5
  stride_o_dim := 2
  SEQLEN := 3
  BLOCK_M := 2
  DIM := 2
  BLOCK_N := 2
  causal := causal
  sm_scale := 1

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["Q","K","V","L","O"]
  base := fun r => if r="Q" then 1 else if r="K" then 41 else if r="V" then 91 else if r="L" then 141 else 151
  extent := fun r => if r="Q" then 28 else if r="K" then 44 else if r="V" then 45 else if r="L" then 6 else if r="O" then 29 else 0

def initial : BlockState where
  mem := fun r o => if r="O" then .of .int 99 else if r="L" then .of .int 98
    else .real (if r="V" ∧ (o=41 ∨ o=44) then 4 else 0)
  regs := fun _ _ _ => none
  pids := fun ax => if ax=1 then 1 else 0
  numPids := fun _ => 2
  undef := fun _ _ => 0

theorem allocation_disjoint : allocation.Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl|rfl|rfl|rfl|rfl
  all_goals rcases hs with rfl|rfl|rfl|rfl|rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem allocation_coverage : ∀ r,r∉allocation.regions → allocation.extent r=0 := by
  intro r hr
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [allocation,hr.1,hr.2.1,hr.2.2.1,hr.2.2.2.1,hr.2.2.2.2]

theorem input_bounds (causal : Bool) (r : RegionName) (o : Nat)
    (h : readWindow (layout causal) "Q" "K" "V" 0 1 r o) : o<allocation.extent r := by
  rcases h with ⟨rfl,a,b,rfl⟩|⟨rfl,a,b,rfl⟩|⟨rfl,a,b,rfl⟩
  all_goals first | change _<28 | change _<44 | change _<45
  all_goals
    have ha := a.isLt
    have hb := b.isLt
    cases causal
    all_goals
      norm_num [qOffset,kOffset,vOffset,base,row,layout,allocation,keyCount,hi,
        Geometry.stop,Geometry.visited,Geometry.tiles,Nat.ceilDiv_eq_add_pred_div,Region.mk.injEq,String.reduceEq] at ha hb ⊢
      omega


theorem output_bounds (causal : Bool) (r : RegionName) (o : Nat)
    (h : writeWindow (layout causal) "L" "O" 0 1 r o) : o<allocation.extent r := by
  rcases h with ⟨rfl,i,rfl⟩|⟨rfl,i,d,rfl⟩
  · have hi := i.isLt
    simp [lOffset,row,layout,allocation]
    change i.val<2 at hi
    omega
  · have hi := i.isLt;have hd := d.isLt
    simp [oOffset,base,row,layout,allocation]
    change i.val<2 at hi
    change d.val<2 at hd
    omega

theorem readQ (o : Nat) : initial.readMem "Q" o=0 := by rfl
theorem readK (o : Nat) : initial.readMem "K" o=0 := by rfl
theorem readV (o : Nat) : initial.readMem "V" o=(if o=41 ∨ o=44 then 4 else 0) := by
  by_cases h : o=41 ∨ o=44
  all_goals simp [BlockState.readMem,initial,h,MemCell.real,MemCell.of,MemCell.readAs,
    (by decide : ("V" : RegionName)≠"O"),(by decide : ("V" : RegionName)≠"L")]

/-- The loaded fourth key has value 4. The three logical keys have value 0,
so truncating the reference to `SEQLEN` would produce a wrong zero result. -/
theorem noncausal_output (i d : Fin 2) :
    output (layout Bool.false) initial.readMem "Q" "K" "V" 0 1 i.val d.val=1 := by
  fin_cases i <;> fin_cases d
  all_goals norm_num [output,numerator,denominator,keys,keyCount,hi,Geometry.stop,
    Geometry.visited,Geometry.tiles,Nat.ceilDiv_eq_add_pred_div,layout,score,
    qOffset,kOffset,vOffset,base,row,readQ,readK,readV,
    Fin.sum_univ_succ,pow2,List.range_succ]

theorem causal_output (i d : Fin 2) :
    output (layout Bool.true) initial.readMem "Q" "K" "V" 0 1 i.val d.val=0 := by
  fin_cases i <;> fin_cases d
  all_goals norm_num [output,numerator,denominator,keys,keyCount,hi,Geometry.stop,
    Geometry.visited,Geometry.tiles,Nat.ceilDiv_eq_add_pred_div,layout,score,
    qOffset,kOffset,vOffset,base,row,readQ,readK,readV,
    Fin.sum_univ_succ,pow2,List.range_succ]

theorem noncausal_denominator (i : Fin 2) :
    denominator (layout Bool.false) initial.readMem "Q" "K" 0 1 i.val=4 := by
  fin_cases i
  all_goals norm_num [denominator,keys,keyCount,hi,Geometry.stop,Geometry.visited,
    Geometry.tiles,Nat.ceilDiv_eq_add_pred_div,layout,score,readQ,readK,
    Fin.sum_univ_succ,pow2,List.range_succ]

theorem noncausal_lse (i : Fin 2) :
    lse (layout Bool.false) initial.readMem "Q" "K" 0 1 i.val=2 := by
  unfold lse
  rw [noncausal_denominator]
  have hl : Real.log (4:ℝ)=2*Real.log 2 := by
    convert Real.log_pow (2:ℝ) 2 using 1 <;> norm_num
  rw [hl]
  have hn : Real.log (2:ℝ)≠0 := ne_of_gt (Real.log_pos (by norm_num))
  field_simp

theorem causal_lse (i : Fin 2) :
    lse (layout Bool.true) initial.readMem "Q" "K" 0 1 i.val=i.val := by
  have hn : Real.log (2:ℝ)≠0 := ne_of_gt (Real.log_pos (by norm_num))
  fin_cases i
  all_goals norm_num [lse,denominator,keys,keyCount,hi,Geometry.stop,Geometry.visited,
    Geometry.tiles,Nat.ceilDiv_eq_add_pred_div,layout,score,readQ,readK,
    Fin.sum_univ_succ,pow2,List.range_succ,row,hn]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.WitnessData
