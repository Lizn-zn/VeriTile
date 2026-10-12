import bench.tritonbench_g.cross_entropy1.Proofs.ForwardIO
import bench.tritonbench_g.cross_entropy1.Proofs.BackwardIO
namespace VeriTile.Bench.TritonBenchG.CrossEntropy1.RealWitness
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def initial : BlockState where
  mem := fun r _ => if r="label" then .of .int 2 else .real 1
  regs := fun _ _ _ => none
  pids := fun d => if d=0 then 1 else 0
  numPids := fun d => if d=0 then 2 else 1
  undef := fun _ _ => 0

def forwardAllocation : FlatAlloc where
  flat := "flat"
  regions := ["label","logits","loss","lse"]
  base := fun r => if r="label" then 0 else if r="logits" then 16 else if r="loss" then 32 else 48
  extent := fun r => if r ∈ (["label","logits","loss","lse"] : List RegionName) then 16 else 0

theorem forward_disjoint : forwardAllocation.Disjoint := by
  intro r hr s hs hn
  simp only [forwardAllocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl | rfl | rfl | rfl
  all_goals rcases hs with rfl | rfl | rfl | rfl
  all_goals first | exact False.elim (hn rfl) | decide

theorem forward_coverage : ∀ r, r ∉ forwardAllocation.regions → forwardAllocation.extent r=0 := by
  intro r hr
  simpa only [forwardAllocation] using (if_neg hr :
    (if r ∈ forwardAllocation.regions then 16 else 0)=0)

/-- A nonzero row, active label and partial block execute on every storage and
smoothing/split branch, from empty registers and finite initial arrays. -/
theorem forward_executes (storage : Storage) (swapped smooth split : Bool) :
    ∃ final, exec (forwardAllocation.flattenKernel
      (forward storage swapped "loss" "lse" "logits" ⟨"label"⟩
        1 2 (-100) 3 0 3 2 4 4 smooth split).eraseDType.toAlgKernel)
      (forwardAllocation.flattenState initial)=some final := by
  have h := ForwardIO.implements storage swapped "loss" "lse" "logits" ⟨"label"⟩
    1 2 (-100) 3 0 3 2 4 4 smooth split (by decide) (by decide) (by decide)
  obtain ⟨v,he,_,_,_⟩ := h forwardAllocation forward_disjoint rfl forward_coverage
    1 0 2 (fun _ => 1) 1 initial rfl rfl rfl (by simp [ForwardIO.sourceIO,forwardAllocation,Region.cast])
    (by intro j hj; simp [ForwardIO.sourceIO,forwardAllocation] at hj ⊢; omega)
    (by intro _; simp [ForwardIO.sourceIO,forwardAllocation])
    (by intro _; simp [ForwardIO.sourceIO,forwardAllocation])
    (by intro _; simp [ForwardIO.sourceIO,forwardAllocation])
    rfl (by intros; rfl) (by intros; rfl)
  exact ⟨v,he⟩

def backwardLayout (smooth ignored : Bool) : BackwardFormula.Layout where
  nCols := 3
  B := 4
  logitsStride := 4
  gradientStride := 4
  dlossStride := 1
  totalClasses := 3
  classStart := 0
  ignored := if ignored then 2 else -100
  smoothing := 1
  squareScale := 2
  hasSmoothing := smooth

def backwardRegions : BackwardProgram.Regions := ⟨"gradient","dloss","logits","lse",⟨"label"⟩⟩

def backwardAllocation (ignored : Bool) : FlatAlloc where
  flat := "flat"
  regions := ["label","logits","lse","dloss","gradient"]
  base := fun r => if r="label" then 0 else if r="logits" then 16 else if r="lse" then 32
    else if r="dloss" then 48 else 64
  extent := fun r =>
    if r ∈ (["label","logits","lse","dloss","gradient"] : List RegionName) then
      if ignored && r="dloss" then 0 else 16
    else 0

theorem backward_disjoint (ignored : Bool) : (backwardAllocation ignored).Disjoint := by
  cases ignored
  all_goals
    intro r hr s hs hn
    simp only [backwardAllocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
    rcases hr with rfl | rfl | rfl | rfl | rfl
    all_goals rcases hs with rfl | rfl | rfl | rfl | rfl
    all_goals first | exact False.elim (hn rfl) | decide

theorem backward_coverage (ignored : Bool) :
    ∀ r, r ∉ (backwardAllocation ignored).regions → (backwardAllocation ignored).extent r=0 := by
  intro r hr
  simp only [backwardAllocation] at hr ⊢
  rw [if_neg hr]

theorem backward_bounds (smooth ignored : Bool) :
    BackwardSafety.Bounds (backwardLayout smooth ignored) backwardRegions
      (BackwardProgram.launch initial) (BackwardProgram.inputs initial backwardRegions)
      (backwardAllocation ignored).extent := by
  cases ignored
  · constructor
    · change 1 < 16; decide
    · change 1 < 16; decide
    · intro _; change 1 < 16; decide
    · intro i hi
      have hi : i.val < 3 := by simpa [BackwardFormula.active,BackwardFormula.column,backwardLayout,BackwardProgram.launch,initial] using hi
      simpa [BackwardFormula.logitAddress,BackwardFormula.gradientAddress,BackwardFormula.column,
        backwardLayout,backwardRegions,BackwardProgram.launch,initial,backwardAllocation,
        show ("logits" : RegionName) ≠ "dloss" by decide,
        show ("gradient" : RegionName) ≠ "dloss" by decide]
        using (show 4+i.val < 16 by omega)
    · intro i hi
      have hi : i.val < 3 := by simpa [BackwardFormula.active,BackwardFormula.column,backwardLayout,BackwardProgram.launch,initial] using hi
      simpa [BackwardFormula.logitAddress,BackwardFormula.gradientAddress,BackwardFormula.column,
        backwardLayout,backwardRegions,BackwardProgram.launch,initial,backwardAllocation,
        show ("logits" : RegionName) ≠ "dloss" by decide,
        show ("gradient" : RegionName) ≠ "dloss" by decide]
        using (show 4+i.val < 16 by omega)
  · constructor
    · change 1 < 16; decide
    · change 1 < 16; decide
    · intro hi
      change (2 : Int) ≠ 2 at hi
      exact False.elim (hi rfl)
    · intro i hi
      have hi : i.val < 3 := by simpa [BackwardFormula.active,BackwardFormula.column,backwardLayout,BackwardProgram.launch,initial] using hi
      simpa [BackwardFormula.logitAddress,BackwardFormula.gradientAddress,BackwardFormula.column,
        backwardLayout,backwardRegions,BackwardProgram.launch,initial,backwardAllocation,
        show ("logits" : RegionName) ≠ "dloss" by decide,
        show ("gradient" : RegionName) ≠ "dloss" by decide]
        using (show 4+i.val < 16 by omega)
    · intro i hi
      have hi : i.val < 3 := by simpa [BackwardFormula.active,BackwardFormula.column,backwardLayout,BackwardProgram.launch,initial] using hi
      simpa [BackwardFormula.logitAddress,BackwardFormula.gradientAddress,BackwardFormula.column,
        backwardLayout,backwardRegions,BackwardProgram.launch,initial,backwardAllocation,
        show ("logits" : RegionName) ≠ "dloss" by decide,
        show ("gradient" : RegionName) ≠ "dloss" by decide]
        using (show 4+i.val < 16 by omega)

/-- The ignored-label instance has an empty upstream-loss allocation. It still
reads LSE/logits, and the successful full execution verifies the conditional
dloss access obligation without assuming a precomputed output. -/
theorem backward_executes (storage : Storage) (swapped smooth ignored : Bool) :
    ∃ final, exec ((backwardAllocation ignored).flattenKernel
      (BackwardProgram.source (backwardLayout smooth ignored) backwardRegions storage swapped).eraseDType.toAlgKernel)
      ((backwardAllocation ignored).flattenState initial)=some final := by
  obtain ⟨v,he,_,_⟩ := BackwardFlat.correct (backwardAllocation ignored) (backward_disjoint ignored)
    (backward_coverage ignored) (backwardLayout smooth ignored) backwardRegions storage swapped initial
    (backward_bounds smooth ignored)
  exact ⟨v,he⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CrossEntropy1.RealWitness
