import bench.tritonbench_g.diag_ssm_triton.Correct
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.BackwardWitness
open VeriTile.Triton TypedSource Correctness.RealBackward
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

/-- Two reverse iterations over three columns, including both previous-state
branches and a masked fourth lane. Only one previous output row is allocated. -/
def layout : BackwardLayout := ⟨"s","l","y","gs","gx","gl","gy",1,3,4⟩

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["s","l","y","gs","gx","gl","gy"]
  base := fun r => if r="s" then 1 else if r="l" then 9 else if r="y" then 17
    else if r="gy" then 25 else if r="gx" then 41 else if r="gs" then 57 else 69
  extent := fun r => if r="s" then 3 else if r="l" then 3 else if r="y" then 3
    else if r="gy" then 6 else if r="gx" then 8 else if r="gs" then 4 else if r="gl" then 4 else 0

def launch : BlockState where
  mem := fun r _ => if r="s" then .real 1 else if r="l" then .real 2
    else if r="y" then .real (-1) else if r="gy" then .real (-1) else .of .int 9
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun _ _ => 0

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

theorem read_constant (r : RegionName) (v : ℝ) (h : ∀ o, launch.mem r o=MemCell.real v) (o : Nat) :
    launch.readMem r o=v := by
  unfold BlockState.readMem
  rw [h o]
  rfl

theorem original_executes (storage : Storage) :
    ∃ v, exec (allocation.flattenKernel
      (TypedSource.backward storage Bool.false "s" "l" "y" "gs" "gx" "gl" "gy" 2 1 3 4).eraseDType.toAlgKernel)
      (allocation.flattenState launch)=some v ∧
      v.readMem "flat" 41 = -3 ∧ v.readMem "flat" 44 = -1 ∧
      v.readMem "flat" 57 = -6 ∧ v.readMem "flat" 69 = -2 ∧
      v.mem "flat" 47=MemCell.of .int 9 := by
  have hb : Bounds layout launch 2 allocation.extent := by
    constructor
    · intro _ j hj
      change 0*4+j.val<1*3 at hj
      change (0*4+j.val)<3
      omega
    · intro j hj
      change 0*4+j.val<1*3 at hj
      change (0*4+j.val)%3<3
      omega
    · intro t ht j hj
      change 0*4+j.val<1*3 at hj
      change offset layout launch t j<3
      simp only [column,offset,layout,launch]
      omega
    · intro t ht j hj
      change 0*4+j.val<1*3 at hj
      change offset layout launch t j<6
      simp only [column,offset,layout,launch]
      omega
    · intro t ht j hj
      change 0*4+j.val<1*3 at hj
      change offset layout launch t j<8
      simp only [column,offset,layout,launch]
      omega
    · intro j hj
      change 0*4+j.val<1*3 at hj
      change (0*4+j.val)<4
      omega
    · intro j hj
      change 0*4+j.val<1*3 at hj
      change (0*4+j.val)<4
      omega
  have hsep : Separation layout := ⟨by decide,by decide,by decide,by decide,by decide,by decide,by decide⟩
  obtain ⟨v,he,hx,hs,hl,hf⟩ := Correctness.RealBackward.backward_correctness storage layout 2
    (fun _ => hsep) allocation disjoint rfl coverage launch rfl hb
  have rs (o : Nat) : launch.readMem "s" o=1 := read_constant "s" 1 (by intro o;simp [launch]) o
  have rl (o : Nat) : launch.readMem "l" o=2 := read_constant "l" 2 (by intro o;simp [launch]) o
  have ry (o : Nat) : launch.readMem "y" o= -1 := read_constant "y" (-1) (by intro o;simp [launch]) o
  have rg (o : Nat) : launch.readMem "gy" o= -1 := read_constant "gy" (-1) (by intro o;simp [launch]) o
  refine ⟨v,he,?_,?_,?_,?_,?_⟩
  · have hout := hx 0 (by decide) ⟨0,by decide⟩ (by change 0<3; decide)
    change v.readMem "flat" 41 = _ at hout
    rw [hout]
    norm_num [inputValue,Reference.inputGradient,Reference.reverseCarry,layout,rl,rg]
  · have hout := hx 1 (by decide) ⟨0,by decide⟩ (by change 0<3; decide)
    change v.readMem "flat" 44 = _ at hout
    rw [hout]
    norm_num [inputValue,Reference.inputGradient,Reference.reverseCarry,layout,rl,rg]
  · have hout := hs ⟨0,by decide⟩ (by change 0<3; decide)
    change v.readMem "flat" 57 = _ at hout
    rw [hout]
    norm_num [initialValue,Reference.reverseCarry,layout,rl,rg]
  · have hout := hl ⟨0,by decide⟩ (by change 0<3; decide)
    change v.readMem "flat" 69 = _ at hout
    rw [hout]
    norm_num [transitionValue,Reference.transitionGradient,Reference.reverseCarry,Reference.previousState,layout,rl,rg,rs,ry]
  · rw [hf "flat" 47 (Or.inr (by
      refine ⟨?_,?_,?_⟩
      · intro t ht j hj
        change 0*4+j.val<1*3 at hj
        change 47 ≠ 41+(t*(1*3)+(0*4+j.val))
        omega
      · intro j hj
        change 47 ≠ 57+(0*4+j.val)
        omega
      · intro j hj
        change 47 ≠ 69+(0*4+j.val)
        omega))]
    rfl

theorem executes (storage : Storage) (swapped : Bool) :
    ∃ v, exec (allocation.flattenKernel
      (TypedSource.backward storage swapped "s" "l" "y" "gs" "gx" "gl" "gy" 2 1 3 4).eraseDType.toAlgKernel)
      (allocation.flattenState launch)=some v ∧
      v.readMem "flat" 41 = -3 ∧ v.readMem "flat" 44 = -1 ∧
      v.readMem "flat" 57 = -6 ∧ v.readMem "flat" 69 = -2 ∧
      v.mem "flat" 47=MemCell.of .int 9 := by
  cases swapped
  · exact original_executes storage
  · simpa only [← Candidate.backward_flat_exec] using original_executes storage

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.BackwardWitness
