import bench.tritonbench_g.diag_ssm_triton.Correct
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardWitness
open VeriTile.Triton ComplexBackwardLoop
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

/-- Two reverse steps exercise both previous-state branches and a partial
four-lane tile. Odd bases require separate relocation of both components. -/
def layout : Layout := ⟨"s","l","y","gs","gx","gl","gy",1,3,4⟩
def allocation : FlatAlloc where
  flat := "flat"
  regions := ["s","l","y","gy","gx","gs","gl"]
  base := fun r => if r="s" then 1 else if r="l" then 9 else if r="y" then 17
    else if r="gy" then 25 else if r="gx" then 41 else if r="gs" then 57 else 69
  extent := fun r => if r="s" then 6 else if r="l" then 6 else if r="y" then 6
    else if r="gy" then 12 else if r="gx" then 14 else if r="gs" then 8 else if r="gl" then 8 else 0

def launch : BlockState where
  mem := fun r o =>
    if r="s" then .real (if o%2=0 then 1 else 2)
    else if r="l" then .real (if o%2=0 then 2 else -1)
    else if r="y" then .real (if o%2=0 then 1 else 7)
    else if r="gy" then .real (if o%2=0 then 1 else -1)
    else .of .int 9
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun _ _ => 0

theorem separation : ComplexBackwardRun.Separation layout := by
  exact ⟨⟨by decide,by decide,by decide⟩,by decide,by decide,by decide⟩

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem coverage : ∀ r, r ∉ allocation.regions → allocation.extent r=0 := by
  intro r hr
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [allocation,hr.1,hr.2.1,hr.2.2.1,hr.2.2.2.1,hr.2.2.2.2.1,hr.2.2.2.2.2.1,hr.2.2.2.2.2.2]

theorem bounds : ComplexBackwardSafety.Bounds layout launch 2 allocation.extent := by
  constructor
  · intro _ j hj
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
  · intro t ht j hj
    simp [mask,ComplexForwardInit.active,ComplexForwardInit.columns,layout,launch] at hj
    simp [offset,ComplexForwardStep.offset,columns,ComplexForwardInit.columns,layout,launch,allocation]
    omega
  · intro j hj
    simp [mask,ComplexForwardInit.active,ComplexForwardInit.columns,layout,launch] at hj
    simp [columns,ComplexForwardInit.columns,layout,launch,allocation]
    omega
  · intro j hj
    simp [mask,ComplexForwardInit.active,ComplexForwardInit.columns,layout,launch] at hj
    simp [columns,ComplexForwardInit.columns,layout,launch,allocation]
    omega

def lane : Lane layout := ⟨⟨0,by decide⟩,PUnit.unit⟩

/-- Read the two components after first resolving the complete tagged cell;
this avoids treating the memory type tag as an implicit real input. -/
theorem masked_input {B : Nat} (r : RegionName) (off : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) (j : TileIndex [B]) (hm : mask j=Bool.true) (z : ℂ)
    (hc : ∀ o, launch.mem r o=MemCell.real (if o%2=0 then z.re else z.im)) :
    ComplexForwardStep.input launch r off mask j=z := by
  apply Complex.ext
  · simp only [ComplexForwardStep.input,hm,↓reduceIte]
    unfold BlockState.readMem
    rw [hc]
    simp [MemCell.real,MemCell.of,MemCell.readAs]
  · simp only [ComplexForwardStep.input,hm,↓reduceIte]
    unfold BlockState.readMem
    rw [hc]
    simp [Nat.add_mod,MemCell.real,MemCell.of,MemCell.readAs]

theorem initial_value : initial layout launch lane=(⟨1,2⟩ : ℂ) := by
  exact masked_input "s" (columns layout launch) (mask layout launch) lane (by decide) (⟨1,2⟩ : ℂ) (by intro o;simp [launch])
theorem transition_value : transition layout launch lane=(⟨2,-1⟩ : ℂ) := by
  exact masked_input "l" (fun j => columns layout launch j % layout.dim) (mask layout launch) lane (by decide) (⟨2,-1⟩ : ℂ) (by intro o;simp [launch])
theorem output_value (t : Nat) : output layout launch t lane=(⟨1,7⟩ : ℂ) := by
  exact masked_input "y" (offset layout launch t) (mask layout launch) lane (by decide) (⟨1,7⟩ : ℂ) (by intro o;simp [launch])
theorem grad_value (t : Nat) : gradOutput layout launch t lane=(⟨1,-1⟩ : ℂ) := by
  exact masked_input "gy" (offset layout launch t) (mask layout launch) lane (by decide) (⟨1,-1⟩ : ℂ) (by intro o;simp [launch])

/-- The reference recurrence gives grad_x[0]=4-2i, grad_x[1]=1-i,
grad_s=10 and the per-column grad_lambda=-6-18i. -/
theorem executes (swapped : Bool) :
    ∃ v, exec (allocation.flattenKernel
      (TypedSource.complexBackward swapped "s" "l" "y" "gs" "gx" "gl" "gy" 2 1 3 4).eraseDType.toAlgKernel)
      (allocation.flattenState launch)=some v ∧
      ComplexBackwardIO.readFlat allocation v "gx" 0=(⟨4,-2⟩ : ℂ) ∧
      ComplexBackwardIO.readFlat allocation v "gx" 3=(⟨1,-1⟩ : ℂ) ∧
      ComplexBackwardIO.readFlat allocation v "gs" 0=(⟨10,0⟩ : ℂ) ∧
      ComplexBackwardIO.readFlat allocation v "gl" 0=(⟨-6,-18⟩ : ℂ) ∧
      v.mem "flat" 53=MemCell.of .int 9 := by
  have h : ComplexBackwardIO.Implements
      (TypedSource.complexBackward swapped "s" "l" "y" "gs" "gx" "gl" "gy" 2 1 3 4).eraseDType layout 2 := by
    cases swapped
    · intro hsep A hd hc u hu hb
      exact Correctness.ComplexBackward.complex_backward_correctness layout 2 (fun hp => let s := hsep hp; ⟨s.initial,s.output,s.gradient,s.terminal,s.initialGradient,s.transitionGradient⟩) A hd hc u hu ⟨hb.initial,hb.transition,hb.previous,hb.gradOutput,hb.gradInput,hb.gradInitial,hb.gradTransition⟩
    · intro hsep A hd hc u hu hb
      exact Correctness.ComplexBackward.complex_backward_index_commuted_correctness layout 2 (fun hp => let s := hsep hp; ⟨s.initial,s.output,s.gradient,s.terminal,s.initialGradient,s.transitionGradient⟩) A hd hc u hu ⟨hb.initial,hb.transition,hb.previous,hb.gradOutput,hb.gradInput,hb.gradInitial,hb.gradTransition⟩
  obtain ⟨_,v,he,hx,hs,hl,hf⟩ := h (fun _ => separation) allocation disjoint coverage launch rfl bounds
  refine ⟨v,he,?_,?_,?_,?_,?_⟩
  · have hv := hx 0 (by decide) lane (by decide)
    change ComplexBackwardIO.readFlat allocation v "gx" 0 = _ at hv
    rw [hv]
    norm_num [Reference.inputGradient,Reference.reverseCarry,transition_value,grad_value,
      Complex.ext_iff,Complex.mul_re,Complex.mul_im,Complex.add_re,Complex.add_im]
  · have hv := hx 1 (by decide) lane (by decide)
    change ComplexBackwardIO.readFlat allocation v "gx" 3 = _ at hv
    rw [hv]
    norm_num [Reference.inputGradient,Reference.reverseCarry,transition_value,grad_value,
      Complex.ext_iff,Complex.mul_re,Complex.mul_im,Complex.add_re,Complex.add_im]
  · have hv := hs lane (by decide)
    change ComplexBackwardIO.readFlat allocation v "gs" 0 = _ at hv
    rw [hv]
    norm_num [Reference.reverseCarry,transition_value,grad_value,
      Complex.ext_iff,Complex.mul_re,Complex.mul_im,Complex.add_re,Complex.add_im]
  · have hv := hl lane (by decide)
    change ComplexBackwardIO.readFlat allocation v "gl" 0 = _ at hv
    rw [hv]
    norm_num [Reference.transitionGradient,Reference.reverseCarry,Reference.previousState,
      transition_value,initial_value,grad_value,output_value,
      Complex.ext_iff,Complex.mul_re,Complex.mul_im,Complex.add_re,Complex.add_im]
  · have ho : ComplexBackwardIO.Outside layout allocation launch 2 "flat" 53 := by
      right
      refine ⟨?_,?_,?_⟩
      · intro t ht j hj
        simp [mask,ComplexForwardInit.active,ComplexForwardInit.columns,layout,launch] at hj
        simp [FlatAlloc.addr,allocation,offset,ComplexForwardStep.offset,columns,ComplexForwardInit.columns,layout,launch]
        omega
      · intro j hj
        simp [mask,ComplexForwardInit.active,ComplexForwardInit.columns,layout,launch] at hj
        simp [FlatAlloc.addr,allocation,columns,ComplexForwardInit.columns,layout,launch]
        omega
      · intro j hj
        simp [mask,ComplexForwardInit.active,ComplexForwardInit.columns,layout,launch] at hj
        simp [FlatAlloc.addr,allocation,columns,ComplexForwardInit.columns,layout,launch]
        omega
    rw [hf "flat" 53 ho]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardWitness
