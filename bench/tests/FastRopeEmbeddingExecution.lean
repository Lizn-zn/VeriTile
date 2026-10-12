import bench.tritonbench_g.fast_rope_embedding.Correct
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Execution
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 25000

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["q","c","s"]
  base := fun r => if r="q" then 1 else if r="c" then 101 else 111
  extent := fun r => if r="q" then 98 else if r="c" ∨ r="s" then 6 else 0

def initial : BlockState where
  mem := fun r o => if r="q" ∧ o=97 then .of .int 99 else .real (if r="q" then 2 else if r="c" then 3 else 1)
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun _ _ => 0

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl|rfl|rfl
  all_goals rcases hs with rfl|rfl|rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem coverage : ∀ r,r∉allocation.regions → allocation.extent r=0 := by
  intro r hr
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [allocation,hr.1,hr.2.1,hr.2.2]

def Result (backwards : Bool) (kernel : Kernel) : Prop :=
  ∃ v,exec (allocation.flattenKernel kernel) (allocation.flattenState initial)=some v ∧
    (∀ i j : Fin 3,v.readMem "flat" (78+i.val*7+j.val)=(if backwards then 8 else 4) ∧
      v.readMem "flat" (81+i.val*7+j.val)=(if backwards then 4 else 8)) ∧
    v.mem "flat" 98=MemCell.of .int 99 ∧ v.mem "flat" 50=MemCell.real 2

/-- Three loop iterations in a partial group; nonzero row/group, odd head
size, unallocated masked coefficient tails and a tagged untouched final channel. -/
theorem witness (backwards : Bool) (kernel : Kernel)
    (h : Correctness.correct_target kernel "q" "c" "s" 49 3 3 2 7 7 4 backwards) : Result backwards kernel := by
  have hb : Correctness.Bounds initial "q" "c" "s" 49 3 3 2 7 7 4 allocation.extent := by
    refine ⟨?_,?_,?_,?_⟩
    · intro j hj;change 1%2*3+j.val<6;omega
    · intro j hj;change 1%2*3+j.val<6;omega
    · intro k hk ht j hj
      change 4≤k at hk
      change k<7 at ht
      change 1*49+k*7+j.val<98
      omega
    · intro k hk ht j hj
      change 4≤k at hk
      change k<7 at ht
      change 1*49+k*7+j.val+3<98
      omega
  obtain ⟨v,he,hv,hf⟩ := h allocation disjoint coverage initial rfl hb (by decide) (by decide)
  refine ⟨v,he,?_,?_,?_⟩
  · intro i j
    have ho := hv (4+i.val) (by change 4≤4+i.val;omega) (by change 4+i.val<7;omega)
      ⟨j.val,by omega⟩ j.isLt
    change v.readMem "flat" (1+(49+(4+i.val)*7+j.val))=_ ∧
      v.readMem "flat" (1+(49+(4+i.val)*7+j.val+3))=_ at ho
    have ha : 1+(49+(4+i.val)*7+j.val)=78+i.val*7+j.val := by omega
    have hb : 1+(49+(4+i.val)*7+j.val+3)=81+i.val*7+j.val := by omega
    rw [ha,hb] at ho
    cases backwards <;> fin_cases i <;> fin_cases j
    all_goals
      change _ = (2 : ℝ)*3-2*(if _ then -(1 : ℝ) else 1) ∧
        _ = (2 : ℝ)*3+2*(if _ then -(1 : ℝ) else 1) at ho
      norm_num at ho ⊢
      exact ho
  · rw [hf "flat" 98 (Or.inr (by
      intro k hk ht j hj
      change 4≤k at hk
      change k<7 at ht
      change 98≠1+(49+k*7+j.val) ∧ 98≠1+(49+k*7+j.val+3)
      omega))]
    rfl
  · rw [hf "flat" 50 (Or.inr (by
      intro k hk ht j hj
      change 4≤k at hk
      change k<7 at ht
      change 50≠1+(49+k*7+j.val) ∧ 50≠1+(49+k*7+j.val+3)
      omega))]
    rfl

theorem executes (input cosine sine : Storage) (swapped backwards : Bool) :
    Result backwards (rope input cosine sine swapped "q" "c" "s" 49 3 3 2 7 7 4 backwards).eraseDType.toAlgKernel := by
  cases swapped
  · exact witness backwards _ (Correctness.original_correctness input cosine sine "q" "c" "s" 49 3 3 2 7 7 4 backwards)
  · exact witness backwards _ (Correctness.index_commuted_correctness input cosine sine "q" "c" "s" 49 3 3 2 7 7 4 backwards)

def AliasResult (backwards : Bool) (kernel : Kernel) : Prop :=
  ∃ v,exec (allocation.flattenKernel kernel) (allocation.flattenState initial)=some v ∧
    (∀ i j : Fin 3,v.readMem "flat" (78+i.val*7+j.val)=(if backwards then 8 else 0) ∧
      v.readMem "flat" (81+i.val*7+j.val)=(if backwards then 0 else 8)) ∧
    v.mem "flat" 98=MemCell.of .int 99 ∧ v.mem "flat" 50=MemCell.real 2

/-- Both coefficient tables alias the first head's written Q cells. They
must stay at their initial value 2 for the two later head iterations. -/
theorem alias_witness (backwards : Bool) (kernel : Kernel)
    (h : Correctness.correct_target kernel "q" "q" "q" 49 77 77 2 7 7 4 backwards) : AliasResult backwards kernel := by
  have hb : Correctness.Bounds initial "q" "q" "q" 49 77 77 2 7 7 4 allocation.extent := by
    refine ⟨?_,?_,?_,?_⟩
    · intro j hj;change 1%2*77+j.val<98;omega
    · intro j hj;change 1%2*77+j.val<98;omega
    · intro k hk ht j hj
      change 4≤k at hk
      change k<7 at ht
      change 1*49+k*7+j.val<98
      omega
    · intro k hk ht j hj
      change 4≤k at hk
      change k<7 at ht
      change 1*49+k*7+j.val+3<98
      omega
  obtain ⟨v,he,hv,hf⟩ := h allocation disjoint coverage initial rfl hb (by decide) (by decide)
  refine ⟨v,he,?_,?_,?_⟩
  · intro i j
    have ho := hv (4+i.val) (by change 4≤4+i.val;omega) (by change 4+i.val<7;omega)
      ⟨j.val,by omega⟩ j.isLt
    change v.readMem "flat" (1+(49+(4+i.val)*7+j.val))=_ ∧
      v.readMem "flat" (1+(49+(4+i.val)*7+j.val+3))=_ at ho
    have ha : 1+(49+(4+i.val)*7+j.val)=78+i.val*7+j.val := by omega
    have hb : 1+(49+(4+i.val)*7+j.val+3)=81+i.val*7+j.val := by omega
    rw [ha,hb] at ho
    cases backwards <;> fin_cases i <;> fin_cases j
    all_goals
      change _ = (2 : ℝ)*2-2*(if _ then -(2 : ℝ) else 2) ∧
        _ = (2 : ℝ)*2+2*(if _ then -(2 : ℝ) else 2) at ho
      norm_num at ho ⊢
      exact ho
  · rw [hf "flat" 98 (Or.inr (by
      intro k hk ht j hj
      change 4≤k at hk
      change k<7 at ht
      change 98≠1+(49+k*7+j.val) ∧ 98≠1+(49+k*7+j.val+3)
      omega))]
    rfl
  · rw [hf "flat" 50 (Or.inr (by
      intro k hk ht j hj
      change 4≤k at hk
      change k<7 at ht
      change 50≠1+(49+k*7+j.val) ∧ 50≠1+(49+k*7+j.val+3)
      omega))]
    rfl

theorem alias_executes (storage : Storage) (swapped backwards : Bool) :
    AliasResult backwards (rope storage storage storage swapped "q" "q" "q" 49 77 77 2 7 7 4 backwards).eraseDType.toAlgKernel := by
  cases swapped
  · exact alias_witness backwards _ (Correctness.original_correctness storage storage storage "q" "q" "q" 49 77 77 2 7 7 4 backwards)
  · exact alias_witness backwards _ (Correctness.index_commuted_correctness storage storage storage "q" "q" "q" 49 77 77 2 7 7 4 backwards)


def EmptyResult (kernel : Kernel) : Prop :=
  ∃ v,exec (allocation.flattenKernel kernel) (allocation.flattenState initial)=some v ∧
    ∀ r o,v.mem r o=(allocation.flattenState initial).mem r o

/-- Coefficients are loaded with their active mask even when the head group
is empty; the complete source succeeds and writes no cell. -/
theorem empty_witness (backwards : Bool) (kernel : Kernel)
    (h : Correctness.correct_target kernel "q" "c" "s" 49 3 3 2 7 4 4 backwards) : EmptyResult kernel := by
  have hb : Correctness.Bounds initial "q" "c" "s" 49 3 3 2 7 4 4 allocation.extent := by
    refine ⟨?_,?_,?_,?_⟩
    · intro j hj;change 1%2*3+j.val<6;omega
    · intro j hj;change 1%2*3+j.val<6;omega
    · intro k hk ht
      change 4≤k at hk
      change k<4 at ht
      omega
    · intro k hk ht
      change 4≤k at hk
      change k<4 at ht
      omega
  obtain ⟨v,he,_,hf⟩ := h allocation disjoint coverage initial rfl hb (by decide) (by decide)
  refine ⟨v,he,?_⟩
  intro r o
  apply hf r o (Or.inr ?_)
  intro k hk ht
  change 4≤k at hk
  change k<4 at ht
  omega

theorem empty_executes (input cosine sine : Storage) (swapped backwards : Bool) :
    EmptyResult (rope input cosine sine swapped "q" "c" "s" 49 3 3 2 7 4 4 backwards).eraseDType.toAlgKernel := by
  cases swapped
  · exact empty_witness backwards _ (Correctness.original_correctness input cosine sine "q" "c" "s" 49 3 3 2 7 4 4 backwards)
  · exact empty_witness backwards _ (Correctness.index_commuted_correctness input cosine sine "q" "c" "s" 49 3 3 2 7 4 4 backwards)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Execution
