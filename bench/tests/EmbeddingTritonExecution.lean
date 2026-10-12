import bench.tritonbench_g.embedding_triton_kernel.Correct
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.RealWitness
open VeriTile.Triton TypedSource Correctness
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["weight","ids","out"]
  base := fun r => if r="weight" then 1 else if r="ids" then 17 else 33
  extent := fun r => if r="weight" then 6 else if r="ids" then 4 else if r="out" then 9 else 0

def initial : BlockState where
  mem := fun r o => if r="weight" then .real (-3) else if r="ids" then
    .of .int (if o=0 then -3 else if o=1 then -1 else if o=2 then 0 else 1)
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

theorem bounds : Bounds initial "weight" "ids" "out" (-2) 1 2 2 4 2 4 3 2 allocation.extent := by
  constructor
  · intro t ht jr hn
    exact hn
  · intro t ht jr jd hn hd hlo hhi
    simp only [token,BlockState.readMemValue,BlockState.readMemTyped,initial] at hlo hhi ⊢
    simp only [↓reduceIte,MemCell.of,MemCell.readAs] at hlo hhi ⊢
    split_ifs at hlo hhi ⊢ <;> norm_num at hlo hhi ⊢ <;> change _<6 <;> omega
  · intro t ht jr jd hn hd
    change row initial.pid 3 2 t jr.val*2+jd.val<9
    omega

/-- Two-row chunks, a final chunk extending past BLOCK_N, signed invalid
and valid IDs, and padding width four over physical row stride two. Every
storage choice executes with real outputs 0, -3, -3, 0 and an untouched tag. -/
theorem original_executes (storage : Storage) :
    ∃ v, exec (allocation.flattenKernel
      (source storage Bool.false "weight" "ids" "out" (-2) 1 2 2 4 2 4 3 2).eraseDType.toAlgKernel)
      (allocation.flattenState initial)=some v ∧
      v.readMem "flat" 33=0 ∧ v.readMem "flat" 35= -3 ∧
      v.readMem "flat" 37= -3 ∧ v.readMem "flat" 39=0 ∧
      v.mem "flat" 41=MemCell.of .int 9 := by
  obtain ⟨v,he,hv,hf⟩ := embedding_correctness storage "weight" "ids" "out" (-2) 1 2 2 4 2 4 3 2
    (by decide) (by decide) (by decide) allocation disjoint coverage initial rfl bounds
  refine ⟨v,he,?_,?_,?_,?_,?_⟩
  · have h := hv 0 (by decide) ⟨0,by decide⟩ ⟨0,by decide⟩ (by change 0<4; decide) (by decide)
    change v.readMem "flat" 33=_ at h
    exact h
  · have h := hv 0 (by decide) ⟨1,by decide⟩ ⟨0,by decide⟩ (by change 1<4; decide) (by decide)
    change v.readMem "flat" 35=_ at h
    exact h
  · have h := hv 1 (by decide) ⟨0,by decide⟩ ⟨0,by decide⟩ (by change 2<4; decide) (by decide)
    change v.readMem "flat" 37=_ at h
    exact h
  · have h := hv 1 (by decide) ⟨1,by decide⟩ ⟨0,by decide⟩ (by change 3<4; decide) (by decide)
    change v.readMem "flat" 39=_ at h
    exact h
  · rw [hf "flat" 41 (Or.inr (by
      rintro z ⟨t,ht,jr,hjr,jd,hjd,hn,hd,rfl⟩
      change 41≠33+(row initial.pid 3 2 t jr*2+jd)
      omega))]
    rfl

theorem candidate_executes (storage : Storage) :
    ∃ v, exec (allocation.flattenKernel
      (source storage Bool.true "weight" "ids" "out" (-2) 1 2 2 4 2 4 3 2).eraseDType.toAlgKernel)
      (allocation.flattenState initial)=some v ∧
      v.readMem "flat" 33=0 ∧ v.readMem "flat" 35= -3 ∧
      v.readMem "flat" 37= -3 ∧ v.readMem "flat" 39=0 ∧
      v.mem "flat" 41=MemCell.of .int 9 := by
  simpa only [← Candidate.flat_exec] using original_executes storage

def emptyAllocation : FlatAlloc := { allocation with extent := fun _ => 0 }

/-- All-masked rows require no allocations, even though the full source
still executes both chunks and initializes all its registers. -/
theorem inactive_executes (storage : Storage) (swapped : Bool) :
    ∃ v, exec (emptyAllocation.flattenKernel
      (source storage swapped "weight" "ids" "out" (-2) 1 2 2 0 2 4 3 2).eraseDType.toAlgKernel)
      (emptyAllocation.flattenState initial)=some v ∧
      v.mem=(emptyAllocation.flattenState initial).mem := by
  have hd : emptyAllocation.Disjoint := by
    intro r hr s hs hn
    simp only [emptyAllocation]
    omega
  have hb : Bounds initial "weight" "ids" "out" (-2) 1 2 2 0 2 4 3 2 emptyAllocation.extent := by
    constructor
    · intro t ht jr hn; exact False.elim (Nat.not_lt_zero _ hn)
    · intro t ht jr jd hn; exact False.elim (Nat.not_lt_zero _ hn)
    · intro t ht jr jd hn; exact False.elim (Nat.not_lt_zero _ hn)
  obtain ⟨v,he,_,hf⟩ := embedding_correctness storage "weight" "ids" "out" (-2) 1 2 2 0 2 4 3 2
    (by decide) (by decide) (by decide) emptyAllocation hd (by intro r hr; rfl) initial rfl hb
  refine ⟨v,?_,?_⟩
  · cases swapped
    · exact he
    · rw [← Candidate.flat_exec]; exact he
  · funext r o
    exact hf r o (Or.inr (by
      rintro z ⟨t,ht,jr,hjr,jd,hjd,hn,hd,heq⟩
      exact False.elim (Nat.not_lt_zero _ hn)))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.RealWitness
