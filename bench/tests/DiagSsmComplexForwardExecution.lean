import bench.tritonbench_g.diag_ssm_triton.Correct
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardWitness
open VeriTile.Triton ComplexForwardLoop
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

/-- Two time steps, an in-place input/output, and a four-lane tile whose
last lane is inactive. Odd region bases exercise component-wise relocation. -/
def layout : Layout := ⟨"s","x","x","l",1,3,4⟩
def allocation : FlatAlloc where
  flat := "flat"
  regions := ["s","l","x"]
  base := fun r => if r="s" then 1 else if r="l" then 9 else 17
  extent := fun r => if r="s" then 6 else if r="l" then 6 else if r="x" then 14 else 0

def launch : BlockState where
  mem := fun r o =>
    if r="s" then .real (if o%2=0 then 1 else 2)
    else if r="l" then .real (if o%2=0 then 2 else -1)
    else if r="x" then
      if o<12 then .real (if o%2=0 then -3 else 4) else .mk .int 9
    else .mk .nat 11
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun _ _ => 0

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl | rfl | rfl
  all_goals rcases hs with rfl | rfl | rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem coverage : ∀ r, r ∉ allocation.regions → allocation.extent r=0 := by
  intro r hr
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [allocation,hr.1,hr.2.1,hr.2.2]

theorem bounds : ComplexForwardSafety.Bounds layout launch 2 allocation.extent := by
  constructor
  · intro j hj
    simp [mask,ComplexForwardInit.active,ComplexForwardInit.columns,layout,launch] at hj
    simp [columns,ComplexForwardInit.columns,layout,launch,allocation]
    omega
  · intro j hj
    simp [mask,ComplexForwardInit.active,ComplexForwardInit.columns,layout,launch] at hj
    simp [columns,ComplexForwardInit.columns,layout,launch,allocation,Nat.mod_eq_of_lt hj]
    omega
  · intro t ht j hj
    simp [mask,ComplexForwardInit.active,ComplexForwardInit.columns,layout,launch] at hj
    simp [offset,ComplexForwardStep.offset,columns,ComplexForwardInit.columns,layout,launch,allocation]
    omega
  · intro t ht j hj
    simp [mask,ComplexForwardInit.active,ComplexForwardInit.columns,layout,launch] at hj
    simp [offset,ComplexForwardStep.offset,columns,ComplexForwardInit.columns,layout,launch,allocation]
    omega

/-- 1+2i evolves through 1+7i to 6+17i for transition 2-i and input -3+4i.
This evaluates the independent formula, not a sampled kernel execution. -/
theorem expected_second_state : state layout launch 2 ⟨⟨0,by decide⟩,PUnit.unit⟩ = (⟨6,17⟩ : ℂ) := by
  let j : Lane layout := ⟨⟨0,by decide⟩,PUnit.unit⟩
  have hs : initial layout launch j = (⟨1,2⟩ : ℂ) := rfl
  have hl : transition layout launch j = (⟨2,-1⟩ : ℂ) := rfl
  have hx0 : input layout launch 0 j = (⟨-3,4⟩ : ℂ) := rfl
  have hx1 : input layout launch 1 j = (⟨-3,4⟩ : ℂ) := by
    have hoff : offset layout launch 1 j = 3 := rfl
    have hm : mask layout launch j = Bool.true := rfl
    have hc0 : launch.mem "x" 6 = MemCell.real (-3) := by simp [launch]
    have hc1 : launch.mem "x" 7 = MemCell.real 4 := by simp [launch]
    apply Complex.ext
    · simp only [input,ComplexForwardStep.input,hm,↓reduceIte,hoff]
      change launch.readMem "x" 6 = -3
      unfold BlockState.readMem
      rw [hc0]
      rfl
    · simp only [input,ComplexForwardStep.input,hm,↓reduceIte,hoff]
      change launch.readMem "x" 7 = 4
      unfold BlockState.readMem
      rw [hc1]
      rfl
  change state layout launch 2 j = _
  simp only [state,Reference.forwardState,hs,hl,hx0,hx1]
  norm_num [Complex.ext_iff,Complex.mul_re,Complex.mul_im,Complex.add_re,Complex.add_im]

theorem executes (swapped : Bool) :
    ∃ v, exec (allocation.flattenKernel
      (TypedSource.complexForward swapped "s" "x" "x" "l" 2 1 3 4).eraseDType.toAlgKernel)
        (allocation.flattenState launch)=some v ∧
      ComplexForwardIO.readFlat allocation v "x" 3 = (⟨6,17⟩ : ℂ) ∧
      v.mem "flat" 29 = MemCell.of .int 9 := by
  have h : ComplexForwardIO.Implements
      (TypedSource.complexForward swapped "s" "x" "x" "l" 2 1 3 4).eraseDType layout 2 := by
    cases swapped
    · intro A hd hc u hu hb
      exact Correctness.ComplexForward.complex_forward_correctness layout 2 A hd hc u hu ⟨hb.initial,hb.transition,hb.input,hb.output⟩
    · intro A hd hc u hu hb
      exact Correctness.ComplexForward.complex_forward_index_commuted_correctness layout 2 A hd hc u hu ⟨hb.initial,hb.transition,hb.input,hb.output⟩
  obtain ⟨_,v,he,hv,hf⟩ := h allocation disjoint coverage launch rfl bounds
  refine ⟨v,he,?_,?_⟩
  · have hout := hv 1 (by decide) ⟨⟨0,by decide⟩,PUnit.unit⟩ (by decide)
    exact hout.trans expected_second_state
  · have ho : ComplexForwardIO.Outside layout allocation launch 2 "flat" 29 := by
      right
      intro t ht j hj
      simp [mask,ComplexForwardInit.active,ComplexForwardInit.columns,layout,launch] at hj
      simp [FlatAlloc.addr,allocation,offset,ComplexForwardStep.offset,columns,ComplexForwardInit.columns,layout,launch]
      omega
    rw [hf "flat" 29 ho]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardWitness
