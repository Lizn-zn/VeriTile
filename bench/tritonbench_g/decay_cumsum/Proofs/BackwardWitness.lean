import bench.tritonbench_g.decay_cumsum.Proofs.BackwardCorrect
import bench.tritonbench_g.decay_cumsum.Proofs.ForwardWitness
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.BackwardWitness
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["dqi","dqo","dki","dko","q","k","g","dg"]
  base := fun r => if r="dqi" then 0 else if r="dqo" then 32
    else if r="dki" then 64 else if r="dko" then 96 else if r="q" then 128
    else if r="k" then 160 else if r="g" then 192 else 224
  extent := fun r => if r ∈ (["dqi","dqo","dki","dko","q","k","g","dg"] : List RegionName) then 32 else 0

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem coverage : ∀ r, r ∉ allocation.regions → allocation.extent r=0 := by
  intro r hr
  simp only [allocation] at hr ⊢
  rw [if_neg hr]

/-- Two reverse iterations on finite arrays, with nonzero program IDs,
one active and one masked lane, for every independent storage pair. -/
theorem executes (source gate : Storage) (swapped : Bool) :
    ∃ final, exec (allocation.flattenKernel
      (backward source gate swapped "dqi" "dqo" "dki" "dko" "q" "k" "g" "dg"
        16 3 1 2 2 4 2 2 3 1).eraseDType.toAlgKernel)
      (allocation.flattenState ForwardWitness.initial) = some final := by
  have h := BackwardCorrect.original_correct source gate "dqi" "dqo" "dki" "dko" "q" "k" "g" "dg"
    16 3 1 2 2 4 3 2 2 1
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  obtain ⟨v,he,_,_⟩ := h allocation disjoint rfl coverage 1 1 1 (fun _ _ _ => 2)
    ForwardWitness.initial rfl rfl rfl rfl
    (by intro i t j hj; fin_cases i <;> simp [BackwardCorrect.originalIO,allocation] <;> omega)
    (by intro o t j hj; fin_cases o <;> simp [BackwardCorrect.originalIO,allocation] <;> omega)
    (by intros; rfl)
  cases swapped
  · exact ⟨v,he⟩
  · exact ⟨v,(BackwardCandidate.flat_exec_equal allocation source gate "dqi" "dqo" "dki" "dko" "q" "k" "g" "dg"
      16 3 1 2 2 4 2 2 3 1 (allocation.flattenState ForwardWitness.initial)).symm.trans he⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DecayCumsum.BackwardWitness
