import bench.tritonbench_g.fast_layernorm.Correct
namespace VeriTile.Bench.TritonBenchG.FastLayernorm.ForwardExecution
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 20000

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["x","y","w","b","r","mu"]
  base := fun r => if r="x" then 1 else if r="y" then 11 else if r="w" then 21
    else if r="b" then 31 else if r="r" then 41 else 51
  extent := fun r => if r="x" then 7 else if r="y" then 8 else if r="w" ∨ r="b" then 3
    else if r="r" ∨ r="mu" then 2 else 0

def initial : BlockState where
  mem := fun r o => if r="y" ∧ o=7 then .of .int 99
    else .real (if r="x" then 3 else if r="w" then 2 else if r="b" then 1 else 0)
  regs := fun _ _ _ => none
  pids := fun a => if a=0 then 1 else 0
  numPids := fun _ => 2
  undef := fun _ _ => 0

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl|rfl|rfl|rfl|rfl|rfl
  all_goals rcases hs with rfl|rfl|rfl|rfl|rfl|rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem coverage : ∀ r,r∉allocation.regions → allocation.extent r=0 := by
  intro r hr
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [allocation,hr.1,hr.2.1,hr.2.2.1,hr.2.2.2.1,hr.2.2.2.2.1,hr.2.2.2.2.2]

def Result (kernel : Kernel) : Prop :=
  ∃ v,exec (allocation.flattenKernel kernel) (allocation.flattenState initial)=some v ∧
    (∀ j : Fin 3,v.readMem "flat" (15+j.val)=1) ∧ v.readMem "flat" 42=1/2 ∧
    v.readMem "flat" 52=3 ∧ v.mem "flat" 18=MemCell.of .int 99

/-- Nonzero program id, host-sized padded row, finite allocations and a
masked integer-tagged neighbor. The variance is three, not zero. -/
theorem contract_witness (kernel : Kernel)
    (h : Correctness.forward_correct_target kernel "y" "x" "w" "b" "r" "mu" 4 4 3 4 1) : Result kernel := by
  have hb : Correctness.ForwardBounds "y" "x" "w" "b" "r" "mu" 4 4 3 4 initial allocation.extent := by
    refine ⟨?_,?_,?_,?_,by decide,by decide⟩
    all_goals intro j hj
    · change 1*4+j.val<7; omega
    · exact hj
    · exact hj
    · change 1*4+j.val<8; omega
  obtain ⟨v,he,hy,hr,hm,hf⟩ := h allocation disjoint coverage initial rfl hb
    (by decide) (by decide) (by decide) (by decide) (by decide) (by
      change 0<RealMath.variance 3 (fun _ : Fin 4 => (3 : ℝ))+1
      norm_num [RealMath.host_padding_example.2])
  refine ⟨v,he,?_,?_,?_,?_⟩
  · intro j
    have hv := hy ⟨j.val,by omega⟩ j.isLt
    change v.readMem "flat" (11+(1*4+j.val))=_ at hv
    rw [show 11+(1*4+j.val)=15+j.val by omega] at hv
    rw [hv]
    change RealMath.output 3 1 (fun _ : Fin 4 => 3) (fun _ => 2) (fun _ => 1) ⟨j.val,by omega⟩=1
    norm_num [RealMath.output,RealMath.host_padding_example.1]
  · change v.readMem "flat" 42=_ at hr
    rw [hr]
    change RealMath.invStd 3 1 (fun _ : Fin 4 => 3)=1/2
    norm_num [RealMath.invStd,RealMath.host_padding_example.2]
  · change v.readMem "flat" 52=_ at hm
    rw [hm]
    exact RealMath.host_padding_example.1
  · rw [hf "flat" 18 (Or.inr ⟨?_,by decide,by decide⟩)]
    · rfl
    · intro j hj
      change 18≠11+(1*4+j.val)
      omega

theorem executes (input weight bias : Storage) (swapped : Bool) :
    Result (forward input weight bias swapped "y" "x" "w" "b" "r" "mu" 4 4 3 4 1).eraseDType.toAlgKernel := by
  cases swapped
  · exact contract_witness _ (Correctness.forward_correctness input weight bias "y" "x" "w" "b" "r" "mu" 4 4 3 4 1)
  · exact contract_witness _ (Correctness.forward_index_commuted_correctness input weight bias "y" "x" "w" "b" "r" "mu" 4 4 3 4 1)

def backwardInitial : BlockState where
  mem := fun r o => if r="y" ∧ o=7 then .of .int 99
    else .real (if r="x" ∨ r="mu" then 3 else if r="w" then (o : ℝ)+1
      else if r="r" then 2 else 1)
  regs := fun _ _ _ => none
  pids := fun a => if a=0 then 1 else 0
  numPids := fun _ => 2
  undef := fun _ _ => 0

def BackwardResult (kernel : Kernel) : Prop :=
  ∃ v,exec (allocation.flattenKernel kernel) (allocation.flattenState backwardInitial)=some v ∧
    (∀ j : Fin 3,v.readMem "flat" (15+j.val)=(j.val : ℝ)*2-2) ∧
    v.mem "flat" 18=MemCell.of .int 99

/-- Independent caller statistics and varying weights give gradients -2,0,2.
The unused bias load is allocated, while the masked input tail is not. -/
theorem backward_contract_witness (kernel : Kernel)
    (h : Correctness.backward_correct_target kernel "y" "x" "w" "b" "r" "mu" 4 4 3 4) : BackwardResult kernel := by
  have hb : Correctness.BackwardBounds "y" "x" "w" "b" "r" "mu" 4 4 3 4 backwardInitial allocation.extent := by
    refine ⟨?_,?_,?_,?_,by decide,by decide⟩
    all_goals intro j hj
    · change 1*4+j.val<8; omega
    · change 1*4+j.val<7; omega
    · exact hj
    · exact hj
  obtain ⟨v,he,hy,hf⟩ := h allocation disjoint coverage backwardInitial rfl hb (by decide) (by decide)
  refine ⟨v,he,?_,?_⟩
  · intro j
    have hv := hy ⟨j.val,by omega⟩ j.isLt
    change v.readMem "flat" (11+(1*4+j.val))=_ at hv
    rw [show 11+(1*4+j.val)=15+j.val by omega] at hv
    rw [hv]
    have hdy (k : Fin 4) : backwardInitial.readMem "y" (4+k.val)=
        (if k.val<3 then (1 : ℝ) else 0) := by
      fin_cases k <;> rfl
    change RealMath.gradient 3 2 3 (fun _ : Fin 4 => 3)
      (fun k : Fin 4 => backwardInitial.readMem "y" (4+k.val))
      (fun k : Fin 4 => (k.val : ℝ)+1) ⟨j.val,by omega⟩=(j.val : ℝ)*2-2
    simp_rw [hdy]
    fin_cases j
    all_goals norm_num [RealMath.gradient,RealMath.padded,Fin.sum_univ_succ]
  · rw [hf "flat" 18 (Or.inr (by
      intro j hj
      change 18≠11+(1*4+j.val)
      omega))]
    rfl

theorem backward_executes (input weight bias : Storage) (swapped : Bool) :
    BackwardResult (backward input weight bias swapped "y" "x" "w" "b" "r" "mu" 4 4 3 4 1).eraseDType.toAlgKernel := by
  cases swapped
  · exact backward_contract_witness _ (Correctness.backward_correctness input weight bias "y" "x" "w" "b" "r" "mu" 4 4 3 4 1)
  · exact backward_contract_witness _ (Correctness.backward_index_commuted_correctness input weight bias "y" "x" "w" "b" "r" "mu" 4 4 3 4 1)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastLayernorm.ForwardExecution
