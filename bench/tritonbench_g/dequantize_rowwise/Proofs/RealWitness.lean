import bench.tritonbench_g.dequantize_rowwise.Proofs.RealIO
namespace VeriTile.Bench.TritonBenchG.DequantizeRowwise.RealWitness
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 1000000

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["x","scale","out"]
  base := fun r => if r="x" then 0 else if r="scale" then 8 else 16
  extent := fun r => if r="x" then 6 else if r="scale" then 2 else if r="out" then 6 else 0

def initial : BlockState where
  mem := fun r _ => if r="x" then .mk .int (-3) else .mk .real (some 2)
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
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

/-- The second finite row has three active integer cells and a masked fourth
lane; the unmasked row scale is in bounds. Both complete versions execute. -/
theorem executes (swapped : Bool) :
    ∃ final, exec (allocation.flattenKernel
      ((if swapped then indexCommuted else original) "x" "scale" "out" (1/127) 6 3 4).eraseDType.toAlgKernel)
      (allocation.flattenState initial) = some final := by
  have h := RealIO.original_correct "x" "scale" "out" (1/127) 6 3 4 (by decide) (by decide)
  obtain ⟨v,he,_,_⟩ := h allocation disjoint rfl coverage 1 1
    (by intro j hj; change j.val<3 at hj; simp [RealIO.originalIO,allocation]; omega)
    (by intro j hj; simp [RealIO.originalIO,allocation])
    (by intro j hj; change j.val<3 at hj; simp [RealIO.originalIO,allocation]; omega)
    (by intro p hp; cases hp)
    (fun _ => -3) (fun _ => 2) initial rfl rfl rfl
    (by intros; rfl) (by intros; rfl)
  cases swapped
  · exact ⟨v,he⟩
  · exact ⟨v,(Projection.candidate_exec allocation "x" "scale" "out" (1/127) 6 3 4
      (allocation.flattenState initial)).symm.trans he⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DequantizeRowwise.RealWitness
