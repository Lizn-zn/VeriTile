import bench.tritonbench_g.diag_ssm_triton.Correct
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ForwardWitness
open VeriTile.Triton TypedSource Correctness.RealForward
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 20000

/-- Two time steps over three columns with a four-lane block. Only active
input cells are allocated; the output has an integer-tagged tail cell. -/
def layout : ForwardLayout := ⟨"s","x","y","l",1,3,4⟩

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["s","l","x","y"]
  base := fun r => if r="s" then 1 else if r="l" then 9 else if r="x" then 17 else 33
  extent := fun r => if r="s" then 3 else if r="l" then 3 else if r="x" then 6 else if r="y" then 8 else 0

def launch : BlockState where
  mem := fun r _ => if r="s" then .real 1 else if r="l" then .real 2
    else if r="x" then .real (-3) else .of .int 9
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun _ _ => 0

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl | rfl | rfl | rfl
  all_goals rcases hs with rfl | rfl | rfl | rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem coverage : ∀ r, r ∉ allocation.regions → allocation.extent r=0 := by
  intro r hr
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [allocation,hr.1,hr.2.1,hr.2.2.1,hr.2.2.2]

theorem read_constant (r : RegionName) (v : ℝ) (h : ∀ o, launch.mem r o=MemCell.real v) (o : Nat) :
    launch.readMem r o=v := by
  unfold BlockState.readMem
  rw [h o]
  rfl

theorem original_executes (storage : Storage) :
    ∃ v, exec (allocation.flattenKernel
      (TypedSource.forward storage Bool.false "s" "x" "l" "y" 2 1 3 4).eraseDType.toAlgKernel)
      (allocation.flattenState launch)=some v ∧
      v.readMem "flat" 33 = -1 ∧ v.readMem "flat" 36 = -5 ∧
      v.mem "flat" 39=MemCell.of .int 9 := by
  have hb : Bounds layout launch 2 allocation.extent := by
    constructor
    · intro j hj
      change 0*4+j.val<1*3 at hj
      change (0*4+j.val)<3
      omega
    · intro j hj
      change 0*4+j.val<1*3 at hj
      change (0*4+j.val)%3<3
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
  obtain ⟨v,he,hv,hf⟩ := Correctness.RealForward.forward_correctness storage layout 2
    (fun _ => by decide) allocation disjoint rfl coverage launch rfl hb
  have rs (o : Nat) : launch.readMem "s" o=1 := read_constant "s" 1 (by intro o;simp [launch]) o
  have rl (o : Nat) : launch.readMem "l" o=2 := read_constant "l" 2 (by intro o;simp [launch]) o
  have rx (o : Nat) : launch.readMem "x" o= -3 := read_constant "x" (-3) (by intro o;simp [launch]) o
  refine ⟨v,he,?_,?_,?_⟩
  · have hout := hv 0 (by decide) ⟨0,by decide⟩ (by change 0<3; decide)
    change v.readMem "flat" 33 = _ at hout
    rw [hout]
    norm_num [value,Reference.forwardState,layout,rs,rl,rx]
  · have hout := hv 1 (by decide) ⟨0,by decide⟩ (by change 0<3; decide)
    change v.readMem "flat" 36 = _ at hout
    rw [hout]
    norm_num [value,Reference.forwardState,layout,rs,rl,rx]
  · rw [hf "flat" 39 (Or.inr (by
      intro t ht j hj
      change 0*4+j.val<1*3 at hj
      change 39 ≠ 33+(t*(1*3)+(0*4+j.val))
      omega))]
    rfl

/-- All three storage choices and the complete index-commuted candidate
have the same finite real execution witness, after precision erasure. -/
theorem executes (storage : Storage) (swapped : Bool) :
    ∃ v, exec (allocation.flattenKernel
      (TypedSource.forward storage swapped "s" "x" "l" "y" 2 1 3 4).eraseDType.toAlgKernel)
      (allocation.flattenState launch)=some v ∧
      v.readMem "flat" 33 = -1 ∧ v.readMem "flat" 36 = -5 ∧
      v.mem "flat" 39=MemCell.of .int 9 := by
  cases swapped
  · exact original_executes storage
  · simpa only [← Candidate.forward_flat_exec] using original_executes storage

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ForwardWitness
