import FifthSphDraft.Correct
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.Execution
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["x","y","dx","g"]
  base := fun r => if r="x" then 1 else if r="y" then 21 else if r="dx" then 81 else 101
  extent := fun r => if r="x" then 9 else if r="y" then 37 else if r="dx" then 10 else if r="g" then 36 else 0

def initial : BlockState where
  mem := fun r o => if (r="y" ∧ o=36) ∨ (r="dx" ∧ o=9) then .of .int 99
    else .real (if r="x" then (if o=6 then 1 else 0) else if r="g" then (if o=27 then 1 else 0) else 5)
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun _ _ => 0

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl|rfl|rfl|rfl
  all_goals rcases hs with rfl|rfl|rfl|rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem coverage : ∀ r,r∉allocation.regions → allocation.extent r=0 := by
  intro r hr
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [allocation,hr.1,hr.2.1,hr.2.2.1,hr.2.2.2]

def spectrum (c : Fin 11) : ℝ :=
  if c.val=0 then 232681380862329/100000000000000
  else if c.val=2 then 34686092313779/20000000000000
  else if c.val=4 then 80282703616657/50000000000000 else 0

def ForwardResult (kernel : Kernel) : Prop :=
  ∃ v,exec (allocation.flattenKernel kernel) (allocation.flattenState initial)=some v ∧
    (∀ c : Fin 11,v.readMem "flat" (46+c.val)=spectrum c) ∧
    v.mem "flat" 57=MemCell.of .int 99

/-- Nonzero block index, nonzero output column, partial tile, all eleven
outputs and an untouched integer-tagged cell. Masked coordinate tails are
unallocated. The theorem instantiates the public finite-memory contract. -/
theorem forward_witness (kernel : Kernel)
    (h : Correctness.forward_correct_target kernel "x" "y" 2 9 36 1 12) : ForwardResult kernel := by
  have hb : Correctness.ForwardBounds "x" "y" 2 9 36 1 12 initial allocation.extent := by
    refine ⟨?_,?_⟩
    · intro c j hj;change j.val*3+2*3*1+c.val<9 at hj;exact hj
    · intro c j hj;change j.val*12+2*12*1+1+c.val<36 at hj
      change j.val*12+2*12*1+1+c.val<37
      omega
  have hc : Correctness.ForwardCover 2 9 36 1 12 := by
    intro pid j hj
    change j.val*12+2*12*pid+1+0<36 at hj
    change j.val*3+2*3*pid+2<9
    omega
  obtain ⟨v,he,hv,hf⟩ := h allocation disjoint coverage initial rfl hb (by decide) hc
  refine ⟨v,he,?_,?_⟩
  · intro c
    have ho := hv c 0 (by change 0*12+2*12*1+1+c.val<36;omega)
    change v.readMem "flat" (21+(25+c.val))=Correctness.output c 1 0 0 at ho
    have ha : 21+(25+c.val)=46+c.val := by omega
    rw [ha] at ho
    rw [ho]
    fin_cases c <;> norm_num [spectrum,Correctness.output]

  · rw [hf "flat" 57 (Or.inr (by
      intro c j hj
      change j.val*12+2*12*1+1+c.val<36 at hj
      change 57≠21+(j.val*12+2*12*1+1+c.val)
      omega))]
    rfl

theorem forward_executes (input output : Storage) (swapped : Bool) :
    ForwardResult (forward input output swapped "x" "y" 2 9 36 1 12).eraseDType.toAlgKernel := by
  cases swapped
  · exact forward_witness _ (Correctness.forward_correctness input output "x" "y" 2 9 36 1 12)
  · exact forward_witness _ (Correctness.forward_index_commuted_correctness input output "x" "y" 2 9 36 1 12)

def BackwardResult (kernel : Kernel) : Prop :=
  ∃ v,exec (allocation.flattenKernel kernel) (allocation.flattenState initial)=some v ∧
    v.readMem "flat" 87=1367152307844476/100000000000000 ∧
    v.readMem "flat" 88=5 ∧ v.readMem "flat" 89=5 ∧
    v.mem "flat" 90=MemCell.of .int 99 ∧
    v.readMem "flat" 87-5-(867152307844475/100000000000000 : ℝ)=1/100000000000000

/-- Complete backward execution adds to old gradient 5. Its x update at
(1,0,0), with only upstream channel 2 active, differs by 10^-14 from the
exact derivative of the forward decimal table. This is an exact-decimal real
model discrepancy, not a physical floating-point error measurement. -/
theorem backward_witness (kernel : Kernel)
    (h : Correctness.backward_correct_target kernel "x" "dx" "g" 2 9 36 1 12) : BackwardResult kernel := by
  have hb : Correctness.BackwardBounds "x" "dx" "g" 2 9 36 1 12 initial allocation.extent := by
    refine ⟨?_,?_,?_⟩
    · intro c j hj;exact hj
    · intro c j hj;change j.val*3+2*3*1+c.val<9 at hj
      change j.val*3+2*3*1+c.val<10
      omega
    · intro c j hj;exact hj
  have hc : Correctness.BackwardCover 1 2 9 36 1 12 := by
    intro j hj
    change j.val*3+2*3*1+0<9 at hj
    change j.val*3+2*3*1+2<9 ∧ j.val*12+2*12*1+1+10<36
    omega
  obtain ⟨v,he,hv,hf⟩ := h allocation disjoint coverage initial rfl hb hc
  have hx := hv 0 0 (by decide)
  have hy := hv 1 0 (by decide)
  have hz := hv 2 0 (by decide)
  change v.readMem "flat" 87=Correctness.gradient 0 1 0 0 5 (fun d => if 25+d.val=27 then 1 else 0) at hx
  change v.readMem "flat" 88=Correctness.gradient 1 1 0 0 5 (fun d => if 25+d.val=27 then 1 else 0) at hy
  change v.readMem "flat" 89=Correctness.gradient 2 1 0 0 5 (fun d => if 25+d.val=27 then 1 else 0) at hz
  norm_num [Correctness.gradient,Correctness.backwardWeight,Fin.sum_univ_succ] at hx hy hz
  refine ⟨v,he,?_,hy,hz,?_,?_⟩
  · simpa only [show (1367152307844476/100000000000000 : ℝ)=341788076961119/25000000000000 by norm_num] using hx
  · rw [hf "flat" 90 (Or.inr (by
      intro c j hj
      change j.val*3+2*3*1+c.val<9 at hj
      change 90≠81+(j.val*3+2*3*1+c.val)
      omega))]
    rfl
  · rw [hx]
    norm_num

theorem backward_executes (input gradient upstream : Storage) (swapped : Bool) :
    BackwardResult (backward input gradient upstream swapped "x" "dx" "g" 2 9 36 1 12).eraseDType.toAlgKernel := by
  cases swapped
  · exact backward_witness _ (Correctness.backward_correctness input gradient upstream "x" "dx" "g" 2 9 36 1 12)
  · exact backward_witness _ (Correctness.backward_index_commuted_correctness input gradient upstream "x" "dx" "g" 2 9 36 1 12)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.Execution
