import bench.tritonbench_g.dequantize_matmul.Proofs.RealIO
namespace VeriTile.Bench.TritonBenchG.DequantizeMatmul.RealWitness
open VeriTile.Triton TypedSource RealReference
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["b","scale","out"]
  base := fun r => if r="b" then 0 else if r="scale" then 16 else 32
  extent := fun r => if r="b" then 16 else if r="scale" then 4 else if r="out" then 16 else 0

def initial : BlockState where
  mem := fun r _ => if r="b" then .mk .int (-3) else .mk .real (some 2)
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

/-- A finite 2x2 tail tile at nonzero program IDs, loaded from genuine signed
integer cells. Both versions execute with every supported output storage. -/
theorem executes (storage : Storage) (swapped : Bool) :
    ∃ final, exec (allocation.flattenKernel
      (kernel storage swapped "b" "scale" "out" 3 3 3 1 3 1 2 2).eraseDType.toAlgKernel)
      (allocation.flattenState initial) = some final := by
  have hi : ∀ p₀ p₁ : Nat, Function.Injective
      (fun j : Fin (2*2) => fpbOffset p₀ p₁ 3 1 2 2 (laneIdx 2 2 j)) := by
    intro p₀ p₁ i j h
    fin_cases i <;> fin_cases j <;> simp [fpbOffset,laneIdx] at h ⊢ <;> omega
  have h := RealIO.original_correct storage "b" "scale" "out" 3 3 3 1 3 1 2 2 (by decide) hi
  obtain ⟨v,he,_,_⟩ := h allocation disjoint rfl coverage 1 1
    (by intro j hj; fin_cases j <;> simp [RealIO.originalIO,bOffset,laneIdx,allocation])
    (by intro j hj; fin_cases j <;> simp [RealIO.originalIO,nOffset,laneIdx,allocation])
    (by intro j hj; fin_cases j <;> simp [RealIO.originalIO,fpbOffset,laneIdx,allocation])
    (by intro p hp; cases hp)
    (fun _ => -3) (fun _ => 2) initial rfl rfl rfl
    (by intros; rfl) (by intros; rfl)
  cases swapped
  · exact ⟨v,he⟩
  · exact ⟨v,(Candidate.flat_exec_equal allocation storage "b" "scale" "out" 3 3 3 1 3 1 2 2
      (allocation.flattenState initial)).symm.trans he⟩

/-- Empty weight/output allocations are legal when the K mask is empty,
but the column-scale allocation remains necessary and is actually read. -/
def emptyRows : FlatAlloc := { allocation with
  extent := fun r => if r="scale" then 4 else 0 }

theorem empty_rows_execute (storage : Storage) (swapped : Bool) :
    ∃ final, exec (emptyRows.flattenKernel
      (kernel storage swapped "b" "scale" "out" 0 3 3 1 3 1 2 2).eraseDType.toAlgKernel)
      (emptyRows.flattenState initial) = some final := by
  have hd : emptyRows.Disjoint := by
    intro r hr s hs hn
    simp only [emptyRows,allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
    rcases hr with rfl | rfl | rfl
    all_goals rcases hs with rfl | rfl | rfl
    all_goals first | exact False.elim (hn rfl) | decide
  have hc : ∀ r, r ∉ emptyRows.regions → emptyRows.extent r=0 := by
    intro r hr
    simp only [emptyRows,allocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
    simp [emptyRows,hr.2.1]
  have hi : ∀ p₀ p₁ : Nat, Function.Injective
      (fun j : Fin (2*2) => fpbOffset p₀ p₁ 3 1 2 2 (laneIdx 2 2 j)) := by
    intro p₀ p₁ i j h
    fin_cases i <;> fin_cases j <;> simp [fpbOffset,laneIdx] at h ⊢ <;> omega
  have h := RealIO.original_correct storage "b" "scale" "out" 0 3 3 1 3 1 2 2 (by decide) hi
  obtain ⟨v,he,_,_⟩ := h emptyRows hd rfl hc 1 1
    (by intro j hj; simp [RealIO.originalIO,dequantizeActive] at hj)
    (by intro j hj; fin_cases j <;> simp [RealIO.originalIO,nOffset,laneIdx,emptyRows])
    (by intro j hj; simp [RealIO.originalIO,dequantizeActive] at hj)
    (by intro p hp; cases hp)
    (fun _ => -3) (fun _ => 2) initial rfl rfl rfl
    (by intros; rfl) (by intros; rfl)
  cases swapped
  · exact ⟨v,he⟩
  · exact ⟨v,(Candidate.flat_exec_equal emptyRows storage "b" "scale" "out" 0 3 3 1 3 1 2 2
      (emptyRows.flattenState initial)).symm.trans he⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DequantizeMatmul.RealWitness
