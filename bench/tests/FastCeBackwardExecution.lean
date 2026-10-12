import bench.tritonbench_g.fast_ce_loss.Proofs.BackwardIO
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.BackwardWitness
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 20000

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["x","z","l"]
  base := fun r => if r="x" then 1 else if r="z" then 10 else 20
  extent := fun r => if r="x" then 2 else if r="z" ∨ r="l" then 1 else 0

def initial : BlockState where
  mem := fun r o => if r="l" then .of .int (-100)
    else if r="x" ∧ o=1 then .of .int 99 else .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun _ _ => 0

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hn
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl|rfl|rfl
  all_goals rcases hs with rfl|rfl|rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem coverage : ∀ r, r ∉ allocation.regions → allocation.extent r=0 := by
  intro r hr
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [allocation,hr.1,hr.2.1,hr.2.2]

def Result (kernel : Kernel) : Prop :=
  ∃ v, exec (allocation.flattenKernel kernel) (allocation.flattenState initial)=some v ∧
    v.readMem "flat" 1=0 ∧ v.mem "flat" 2=MemCell.of .int 99

/-- An ignored label needs no upstream-gradient allocation. The active gradient
is zero, and the masked integer-tagged neighboring cell is preserved. -/
theorem contract_witness (kernel : Kernel) (cap scale : Bool)
    (h : BackwardIO.Implements kernel cap scale "x" "d" "z" ⟨"l"⟩ 1 1 3 2 1 2) : Result kernel := by
  have hl : initial.readMemValue .int (Region.cast (⟨"l"⟩ : Region .int)) 0 = -100 := rfl
  have hb : BackwardIO.Bounds "x" "d" "z" ⟨"l"⟩ 1 1 3 2 initial allocation.extent := by
    refine ⟨by decide,⟨-100,rfl⟩,Or.inl rfl,by decide,?_,?_⟩
    · intro hi
      exact False.elim (hi hl)
    · intro j hj
      have hj' : j.val<1 := by simpa [BackwardReal.col,initial] using hj
      simpa [BackwardReal.address,BackwardReal.col,initial,allocation] using (show j.val<2 by omega)
  obtain ⟨v,he,hv,hf⟩ := h (by intro _; norm_num)
    allocation disjoint coverage initial rfl hb
  refine ⟨v,he,?_,?_⟩
  · have hx := hv ⟨0,by decide⟩ (by change 0<1; decide)
    change v.readMem "flat" 1 = _ at hx
    rw [hx]
    change BackwardReal.gradient cap scale 1 2 (-100) 0 0 0 0=0
    simp [BackwardReal.gradient]
  · rw [hf "flat" 2 (Or.inr (by
      intro j hj
      have hj' : j.val<1 := by simpa [BackwardReal.col,initial] using hj
      change 2≠1+(0*1+(0*2+j.val))
      omega))]
    rfl

theorem executes (storage : Storage) (cap scale swapped : Bool) :
    Result (backward storage cap scale swapped "x" "d" "z" ⟨"l"⟩ 1 1 3 2 1 2).eraseDType.toAlgKernel := by
  cases swapped
  · exact contract_witness _ cap scale (BackwardIO.typed_correct storage cap scale "x" "d" "z" ⟨"l"⟩ 1 1 3 2 1 2)
  · exact contract_witness _ cap scale (BackwardIO.candidate_correct storage cap scale "x" "d" "z" ⟨"l"⟩ 1 1 3 2 1 2)

def signedAllocation : FlatAlloc where
  flat := "flat"
  regions := ["x","z","l","d"]
  base := fun r => if r="x" then 1 else if r="z" then 10 else if r="l" then 20 else 30
  extent := fun r => if r="x" then 2 else if r="z" ∨ r="l" ∨ r="d" then 1 else 0

def signedInitial : BlockState where
  mem := fun r o => if r="l" then .of .int 1
    else if r="x" ∧ o=1 then .of .int 99 else .real (if r="d" then 1 else 0)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun _ _ => 0

theorem signed_disjoint : signedAllocation.Disjoint := by
  intro r hr s hs hn
  simp only [signedAllocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl|rfl|rfl|rfl
  all_goals rcases hs with rfl|rfl|rfl|rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem signed_coverage : ∀ r, r ∉ signedAllocation.regions → signedAllocation.extent r=0 := by
  intro r hr
  simp only [signedAllocation,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [signedAllocation,hr.1,hr.2.1,hr.2.2.1,hr.2.2.2]

/-- Negative enabled scaling and softcap are valid for the backward formula.
An active non-target logit zero with LSE zero and upstream one produces -3;
the adjacent integer-tagged cell belongs to another chunk and is preserved. -/
theorem negative_parameters (storage : Storage) (swapped : Bool) :
    ∃ v, exec (signedAllocation.flattenKernel
      (backward storage Bool.true Bool.true swapped "x" "d" "z" ⟨"l"⟩ 2 2 1 1 (-2) (-3)).eraseDType.toAlgKernel)
      (signedAllocation.flattenState signedInitial)=some v ∧
      v.readMem "flat" 1 = -3 ∧ v.mem "flat" 2=MemCell.of .int 99 := by
  have h : BackwardIO.Implements
      (backward storage Bool.true Bool.true swapped "x" "d" "z" ⟨"l"⟩ 2 2 1 1 (-2) (-3)).eraseDType.toAlgKernel
      Bool.true Bool.true "x" "d" "z" ⟨"l"⟩ 2 2 1 1 (-2) (-3) := by
    cases swapped
    · exact BackwardIO.typed_correct storage Bool.true Bool.true "x" "d" "z" ⟨"l"⟩ 2 2 1 1 (-2) (-3)
    · exact BackwardIO.candidate_correct storage Bool.true Bool.true "x" "d" "z" ⟨"l"⟩ 2 2 1 1 (-2) (-3)
  have hb : BackwardIO.Bounds "x" "d" "z" ⟨"l"⟩ 2 2 1 1 signedInitial signedAllocation.extent := by
    refine ⟨by decide,⟨1,rfl⟩,?_,by decide,?_,?_⟩
    · change (1 : Int) = -100 ∨ (0≤(1 : Int) ∧ (1 : Int)<2)
      norm_num
    · intro _; decide
    · intro j _
      have hj := j.isLt
      change 0*2+(0*1+j.val)<2
      omega
  obtain ⟨v,he,hv,hf⟩ := h (by intro _; norm_num)
    signedAllocation signed_disjoint signed_coverage signedInitial rfl hb
  refine ⟨v,he,?_,?_⟩
  · have hx := hv ⟨0,by decide⟩ (by change 0<2; decide)
    change v.readMem "flat" 1 = _ at hx
    rw [hx]
    change BackwardReal.gradient Bool.true Bool.true (-2) (-3) 1 0 0 0 1 = -3
    norm_num [BackwardReal.gradient,Real.tanh_zero]
  · rw [hf "flat" 2 (Or.inr (by
      intro j _
      have hj := j.isLt
      change 2≠1+(0*2+(0*1+j.val))
      omega))]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastCeLoss.BackwardWitness
