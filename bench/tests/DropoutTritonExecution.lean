import bench.tritonbench_g.dropout_triton.Correct
namespace VeriTile.Bench.TritonBenchG.DropoutTriton.RealWitness
open VeriTile.Triton TypedSource Correctness
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 20000

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["x","keep","out"]
  base := fun r => if r="x" then 1 else if r="keep" then 9 else 17
  extent := fun r => if r="x" then 3 else if r="keep" then 3 else if r="out" then 4 else 0

def initial (keep : KeepStorage) : BlockState where
  mem := fun r o => if r="x" then .real (-3) else if r="keep" then
    match keep with
    | .int32 => .of .int (if o=0 then 0 else if o=1 then 2 else -1)
    | .bool => .of .bool (decide (o≠0))
    else .of .int 9
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

/-- All storage/keep combinations execute the full source on a partial tile.
Zero, positive-even and negative integer conditions exercise nonzero truthiness;
the untouched tail retains its integer tag. -/
theorem original_executes (storage : Storage) (keep : KeepStorage) :
    ∃ v, exec (allocation.flattenKernel
      (source storage keep Bool.false "x" "keep" "out" 3 4 (1/2)).eraseDType.toAlgKernel)
      (allocation.flattenState (initial keep))=some v ∧
      v.readMem "flat" 17=0 ∧ v.readMem "flat" 18= -6 ∧ v.readMem "flat" 19= -6 ∧
      v.mem "flat" 20=MemCell.of .int 9 := by
  have hb : Bounds "x" "keep" "out" 3 4 (initial keep) allocation.extent := by
    constructor
    · intro j hj; exact hj
    · intro j hj; exact hj
    · intro j hj; change 0*4+j.val<3 at hj; change 0*4+j.val<4; omega
  obtain ⟨v,he,hv,hf⟩ := dropout_correctness storage keep "x" "keep" "out" 3 4 (1/2)
    allocation disjoint coverage (initial keep) rfl hb
  refine ⟨v,he,?_,?_,?_,?_⟩
  · have h := hv ⟨0,by decide⟩ (by change 0<3; decide)
    change v.readMem "flat" 17=_ at h
    rw [h]
    cases keep <;> rfl
  · have h := hv ⟨1,by decide⟩ (by change 1<3; decide)
    change v.readMem "flat" 18=_ at h
    rw [h]
    cases keep <;> change (-3 : ℝ)/(1-1/2) = -6
    all_goals norm_num
  · have h := hv ⟨2,by decide⟩ (by change 2<3; decide)
    change v.readMem "flat" 19=_ at h
    rw [h]
    cases keep <;> change (-3 : ℝ)/(1-1/2) = -6
    all_goals norm_num
  · rw [hf "flat" 20 (Or.inr (by
      intro j hj
      change 0*4+j.val<3 at hj
      change 20≠17+(0*4+j.val)
      omega))]
    rfl

theorem candidate_executes (storage : Storage) (keep : KeepStorage) :
    ∃ v, exec (allocation.flattenKernel
      (source storage keep Bool.true "x" "keep" "out" 3 4 (1/2)).eraseDType.toAlgKernel)
      (allocation.flattenState (initial keep))=some v ∧
      v.readMem "flat" 17=0 ∧ v.readMem "flat" 18= -6 ∧ v.readMem "flat" 19= -6 ∧
      v.mem "flat" 20=MemCell.of .int 9 := by
  simpa only [← Candidate.flat_exec] using original_executes storage keep

def emptyAllocation : FlatAlloc := { allocation with extent := fun _ => 0 }

/-- Fully masked programs need no allocated cells, even with p=1; this
statement concerns erased real semantics, whose division is total. -/
theorem inactive_executes (storage : Storage) (keep : KeepStorage) (swapped : Bool) :
    ∃ v, exec (emptyAllocation.flattenKernel
      (source storage keep swapped "x" "keep" "out" 0 4 1).eraseDType.toAlgKernel)
      (emptyAllocation.flattenState (initial keep))=some v ∧
      v.mem=(emptyAllocation.flattenState (initial keep)).mem := by
  have hd : emptyAllocation.Disjoint := by
    intro r hr s hs hn
    simp only [emptyAllocation]
    omega
  have hb : Bounds "x" "keep" "out" 0 4 (initial keep) emptyAllocation.extent := by
    constructor <;> intro j hj <;> exact False.elim (Nat.not_lt_zero _ hj)
  obtain ⟨v,he,_,hf⟩ := dropout_correctness storage keep "x" "keep" "out" 0 4 1
    emptyAllocation hd (by intro r hr; rfl) (initial keep) rfl hb
  refine ⟨v,?_,?_⟩
  · cases swapped
    · exact he
    · rw [← Candidate.flat_exec]; exact he
  · funext r o
    exact hf r o (Or.inr (by intro j hj; exact False.elim (Nat.not_lt_zero _ hj)))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DropoutTriton.RealWitness
