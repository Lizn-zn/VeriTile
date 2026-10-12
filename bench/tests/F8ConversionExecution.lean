import bench.tritonbench_g.f8_conversion_utils.Correct
namespace VeriTile.Bench.TritonBenchG.F8ConversionUtils.RealWitness
open VeriTile.Triton TypedSource Correctness
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 20000

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["x","y"]
  base := fun r => if r="x" then 1 else 9
  extent := fun r => if r="x" then 2 else if r="y" then 3 else 0

def initial : BlockState where
  mem := fun r o => if r="x" then .real (if o=0 then -3 else 2) else .of .int 9
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun _ _ => 0

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl|rfl
  all_goals rcases hs with rfl|rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem coverage : ∀ r, r ∉ allocation.regions → allocation.extent r=0 := by
  intro r hr
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [allocation,hr.1,hr.2]

def Result (kernel : ComputeKernel) : Prop :=
  ∃ v, exec (allocation.flattenKernel kernel.toAlgKernel) (allocation.flattenState initial)=some v ∧
    v.readMem "flat" 9= -3 ∧ v.readMem "flat" 10=2 ∧ v.mem "flat" 11=MemCell.of .int 9

theorem contract_witness (kernel : ComputeKernel) (h : correct_target kernel "y" "x" 2 4) : Result kernel := by
  have hb : Bounds "y" "x" 2 4 initial allocation.extent := by
    constructor
    · intro j hj; exact hj
    · intro j hj; change 0*4+j.val<2 at hj; change 0*4+j.val<3; omega
  obtain ⟨v,he,hv,hf⟩ := h allocation disjoint coverage initial rfl hb
  refine ⟨v,he,?_,?_,?_⟩
  · exact hv ⟨0,by decide⟩ (by change 0<2; decide)
  · exact hv ⟨1,by decide⟩ (by change 1<2; decide)
  · rw [hf "flat" 11 (Or.inr (by
      intro j hj
      change 0*4+j.val<2 at hj
      change 11≠9+(0*4+j.val)
      omega))]
    rfl

/-- Both original/candidate conversions copy active values after erasure and
preserve the masked integer-tag tail in the finite flat allocation. -/
theorem to_half_executes (format : Format) (swapped : Bool) :
    Result (toHalf format swapped "y" "x" 2 4).eraseDType := by
  cases swapped
  · exact contract_witness _ (f8_to_f16_correctness format "y" "x" 2 4)
  · exact contract_witness _ (f8_to_f16_index_commuted_correctness format "y" "x" 2 4)

theorem to_fp8_executes (input : Input) (format : Format) (swapped : Bool) :
    Result (toFP8 input format swapped "y" "x" 2 4).eraseDType := by
  cases swapped
  · exact contract_witness _ (to_fp8_correctness input format "y" "x" 2 4)
  · exact contract_witness _ (to_fp8_index_commuted_correctness input format "y" "x" 2 4)

def emptyAllocation : FlatAlloc := { allocation with extent := fun _ => 0 }

def EmptyResult (kernel : ComputeKernel) : Prop :=
  ∃ v, exec (emptyAllocation.flattenKernel kernel.toAlgKernel) (emptyAllocation.flattenState initial)=some v ∧
    v.mem=(emptyAllocation.flattenState initial).mem

theorem empty_witness (kernel : ComputeKernel) (h : correct_target kernel "y" "x" 0 4) : EmptyResult kernel := by
  have hd : emptyAllocation.Disjoint := by
    intro r hr s hs hn
    simp only [emptyAllocation]
    omega
  have hb : Bounds "y" "x" 0 4 initial emptyAllocation.extent := by
    constructor <;> intro j hj <;> exact False.elim (Nat.not_lt_zero _ hj)
  obtain ⟨v,he,_,hf⟩ := h emptyAllocation hd (by intro r hr; rfl) initial rfl hb
  refine ⟨v,he,?_⟩
  funext r o
  exact hf r o (Or.inr (by intro j hj; exact False.elim (Nat.not_lt_zero _ hj)))

theorem to_half_inactive_executes (format : Format) (swapped : Bool) :
    EmptyResult (toHalf format swapped "y" "x" 0 4).eraseDType := by
  cases swapped
  · exact empty_witness _ (f8_to_f16_correctness format "y" "x" 0 4)
  · exact empty_witness _ (f8_to_f16_index_commuted_correctness format "y" "x" 0 4)

theorem to_fp8_inactive_executes (input : Input) (format : Format) (swapped : Bool) :
    EmptyResult (toFP8 input format swapped "y" "x" 0 4).eraseDType := by
  cases swapped
  · exact empty_witness _ (to_fp8_correctness input format "y" "x" 0 4)
  · exact empty_witness _ (to_fp8_index_commuted_correctness input format "y" "x" 0 4)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.F8ConversionUtils.RealWitness
