import bench.tritonbench_g.decay_cumsum.Proofs.ForwardCorrect
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.ForwardWitness
open VeriTile.Triton TypedSource
noncomputable section

def initial : BlockState where
  mem := fun _ _ => .real 2
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun _ _ => 0

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["g","go"]
  base := fun r => if r="g" then 0 else 32
  extent := fun r => if r ∈ (["g","go"] : List RegionName) then 32 else 0

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl | rfl
  all_goals rcases hs with rfl | rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem coverage : ∀ r, r ∉ allocation.regions → allocation.extent r=0 := by
  intro r hr
  simp only [allocation] at hr ⊢
  rw [if_neg hr]

/-- Both variants execute two iterations on finite arrays, with nonzero program
IDs and a partially active lane mask. Registers start empty. -/
theorem executes (storage : Storage) (swapped : Bool) :
    ∃ final, exec (allocation.flattenKernel
      (forward storage swapped "g" "go" 16 3 1 2 2 4 2 2 3 1).eraseDType.toAlgKernel)
      (allocation.flattenState initial) = some final := by
  cases swapped
  all_goals
    have h := ForwardCorrect.original_correct storage "g" "go" 16 3 1 2 2 4 2 2 3 1 (by decide) (by decide)
  all_goals
    obtain ⟨v,he,_,_⟩ := h allocation disjoint rfl coverage 1 1 1 (fun _ _ => 2)
      initial rfl rfl rfl rfl
      (by intro t j hj; simp [ForwardCorrect.originalIO,allocation]; omega)
      (by intro t j hj; simp [ForwardCorrect.originalIO,allocation]; omega)
      (by intros; rfl)
    first
    | exact ⟨v,he⟩
    | exact ⟨v,(ForwardCandidate.flat_exec_equal allocation storage "g" "go" 16 3 1 2 2 4 2 2 3 1 (allocation.flattenState initial)).symm.trans he⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DecayCumsum.ForwardWitness
