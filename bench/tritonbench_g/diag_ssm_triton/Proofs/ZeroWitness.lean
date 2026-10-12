import bench.tritonbench_g.diag_ssm_triton.Proofs.ZeroIO
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroWitness
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 1000000

/-- Time-dependent inputs/outputs have zero allocation. The two backward
terminal outputs intentionally share g. A fourth lane is masked, with
integer-tagged initial cells to exercise the full memory frame. -/
def allocation : FlatAlloc where
  flat := "flat"
  regions := ["s","l","g"]
  base := fun r => if r="s" then 0 else if r="l" then 6 else 12
  extent := fun r => if r="s" then 6 else if r="l" then 6 else if r="g" then 8 else 0

def initial : BlockState where
  mem := fun r _ => if r="g" then .mk .int 7 else .mk .real (some (-3))
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

theorem forward_bounds : ZeroSafety.forwardBounds allocation.extent initial "s" "l" 1 3 4 := by
  constructor <;> intro j hj
  all_goals simp only [initial,Nat.zero_mul,Nat.zero_add,Nat.one_mul] at hj
  all_goals simp [allocation,initial,Nat.mod_eq_of_lt hj]; omega

theorem complex_forward_bounds : ZeroSafety.complexForwardBounds allocation.extent initial "s" "l" 1 3 4 := by
  constructor <;> intro j hj
  all_goals simp only [initial,Nat.zero_mul,Nat.zero_add,Nat.one_mul] at hj
  all_goals simp [allocation,initial,Nat.mod_eq_of_lt hj]; omega

theorem backward_bounds : ZeroSafety.backwardBounds allocation.extent initial "l" "g" "g" 1 3 4 := by
  refine ⟨?_,?_,?_⟩ <;> intro j hj
  all_goals simp only [initial,Nat.zero_mul,Nat.zero_add,Nat.one_mul] at hj
  all_goals simp [allocation,initial,Nat.mod_eq_of_lt hj]; omega

theorem complex_backward_bounds : ZeroSafety.complexBackwardBounds allocation.extent initial "l" "g" "g" 1 3 4 := by
  refine ⟨?_,?_,?_⟩ <;> intro j hj
  all_goals simp only [initial,Nat.zero_mul,Nat.zero_add,Nat.one_mul] at hj
  all_goals simp [allocation,initial,Nat.mod_eq_of_lt hj]; omega

theorem forward_executes (storage : Storage) (swapped : Bool) :
    ∃ v, exec (allocation.flattenKernel
      (TypedSource.forward storage swapped "s" "x" "l" "y" 0 1 3 4).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some v ∧
      ∀ r o, v.mem r o = (allocation.flattenState initial).mem r o := by
  obtain ⟨_,v,he,_,hf⟩ := ZeroIO.forward_correct storage swapped "s" "x" "l" "y" 1 3 4
    allocation disjoint coverage initial forward_bounds
  exact ⟨v,he,fun r o => hf r o (by simp)⟩

theorem complex_forward_executes (swapped : Bool) :
    ∃ v, exec (allocation.flattenKernel
      (TypedSource.complexForward swapped "s" "x" "y" "l" 0 1 3 4).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some v ∧
      ∀ r o, v.mem r o = (allocation.flattenState initial).mem r o := by
  obtain ⟨_,v,he,_,hf⟩ := ZeroIO.complexForward_correct swapped "s" "x" "y" "l" 1 3 4
    allocation disjoint coverage initial complex_forward_bounds
  exact ⟨v,he,fun r o => hf r o (by simp)⟩

theorem backward_executes (storage : Storage) (swapped : Bool) :
    ∃ v, exec (allocation.flattenKernel
      (TypedSource.backward storage swapped "s" "l" "y" "g" "gx" "g" "gy" 0 1 3 4).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some v ∧
      (∀ j : Fin 3, v.readMem allocation.flat (allocation.addr "g" j.val)=0) ∧
      (∀ r o, ¬ ZeroBackward.written allocation initial 1 3 4 "g" "g" r o →
        v.mem r o = (allocation.flattenState initial).mem r o) := by
  obtain ⟨_,v,he,hz,hf⟩ := ZeroIO.backward_correct storage swapped "s" "l" "y" "g" "gx" "g" "gy" 1 3 4
    allocation disjoint coverage initial backward_bounds
  refine ⟨v,he,?_,hf⟩
  intro j
  apply ZeroIO.zero_readback (hz _ _ ?_)
  refine ⟨rfl,Or.inl ⟨⟨j.val,by omega⟩,?_,?_⟩⟩
  · simp [initial]
  · simp [initial]

theorem complex_backward_executes (swapped : Bool) :
    ∃ v, exec (allocation.flattenKernel
      (TypedSource.complexBackward swapped "s" "l" "y" "g" "gx" "g" "gy" 0 1 3 4).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some v ∧
      (∀ (j : Fin 3) (c : Fin 2), v.readMem allocation.flat (allocation.addr "g" (j.val*2+c.val))=0) ∧
      (∀ r o, ¬ ZeroComplexBackward.written allocation initial 1 3 4 "g" "g" r o →
        v.mem r o = (allocation.flattenState initial).mem r o) := by
  obtain ⟨_,v,he,hz,hf⟩ := ZeroIO.complexBackward_correct swapped "s" "l" "y" "g" "gx" "g" "gy" 1 3 4
    allocation disjoint coverage initial complex_backward_bounds
  refine ⟨v,he,?_,hf⟩
  intro j c
  apply ZeroIO.zero_readback (hz _ _ ?_)
  fin_cases c
  · refine ⟨rfl,Or.inl ⟨⟨j.val,by omega⟩,(by simp [initial]),?_⟩⟩; simp [initial]
  · refine ⟨rfl,Or.inr (Or.inl ⟨⟨j.val,by omega⟩,(by simp [initial]),?_⟩)⟩; simp [initial]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroWitness
