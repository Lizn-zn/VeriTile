import bench.tritonbench_g.decay_cumsum.Proofs.PrepareCorrect
import bench.tritonbench_g.decay_cumsum.Proofs.ForwardWitness
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.PrepareWitness
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["q","k","g","qg","kg"]
  base := fun r => if r="q" then 0 else if r="k" then 32
    else if r="g" then 64 else if r="qg" then 96 else 128
  extent := fun r => if r ∈ (["q","k","g","qg","kg"] : List RegionName) then 32 else 0

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl | rfl | rfl | rfl | rfl
  all_goals rcases hs with rfl | rfl | rfl | rfl | rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem coverage : ∀ r, r ∉ allocation.regions → allocation.extent r=0 := by
  intro r hr
  simp only [allocation] at hr ⊢
  rw [if_neg hr]

/-- Two preparation iterations on finite arrays, including the otherwise masked
lane of the unmasked final gate load, execute for every storage pair. -/
theorem executes (source gate : Storage) (swapped : Bool) :
    ∃ final, exec (allocation.flattenKernel
      (prepare source gate swapped "q" "k" "g" "qg" "kg"
        16 3 1 2 2 4 2 2 3 1).eraseDType.toAlgKernel)
      (allocation.flattenState ForwardWitness.initial) = some final := by
  have h := PrepareCorrect.original_correct source gate "q" "k" "g" "qg" "kg"
    16 3 1 2 2 4 3 2 2 1 (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  obtain ⟨v,he,_,_⟩ := h allocation disjoint rfl coverage 1 1 1 (fun _ _ _ => 2)
    ForwardWitness.initial rfl rfl rfl rfl
    (by intro i t j hj; fin_cases i <;> simp [PrepareCorrect.originalIO,allocation] <;> omega)
    (by intro o t j hj; fin_cases o <;> simp [PrepareCorrect.originalIO,allocation] <;> omega)
    (by intros; rfl)
  cases swapped
  · exact ⟨v,he⟩
  · exact ⟨v,(PrepareCandidate.flat_exec_equal allocation source gate "q" "k" "g" "qg" "kg"
      16 3 1 2 2 4 2 2 3 1 (allocation.flattenState ForwardWitness.initial)).symm.trans he⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DecayCumsum.PrepareWitness
