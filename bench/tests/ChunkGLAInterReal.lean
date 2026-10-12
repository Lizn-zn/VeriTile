import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterFlat

/-! Complete inter-subblock regressions. Nonzero launch coordinates, two K
blocks, partial rows/features, and timeStride != K exercise the source's two
different gate layouts. Tests cover both addition orders, both skip branches,
K=0, and finite flat memory with a tagged neighbour outside the store mask. -/
namespace VeriTile.Tests.ChunkGLAInterReal
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open RealInterMath RealInterProgram RealInterLoop
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private abbrev p : Layout := ⟨40,5,7,3,4,2,2,2⟩
private abbrev regions : Regions := ⟨"q","k","g","A"⟩
private def initial : BlockState where
  mem := fun r off => if r = "g" then MemCell.real off
    else if r = "q" ∨ r = "k" then MemCell.real 1 else MemCell.of .nat 99
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 1 then 2 else 1
  numPids := fun _ => 2
  undef := fun _ _ => 0

private theorem formula :
    expected p (launch initial) (inputs regions initial) 2 0 0 = 6*Real.exp 10 ∧
    expected p (launch initial) (inputs regions initial) 2 0 1 = 6*Real.exp 5 := by
  change expected p ⟨1,2,1⟩ ⟨fun _ => 1,fun _ => 1,fun off => off⟩ 2 0 0 = _ ∧
    expected p ⟨1,2,1⟩ ⟨fun _ => 1,fun _ => 1,fun off => off⟩ 2 0 1 = _
  norm_num [expected,partialSum,numBlocks,p,Finset.sum_range_succ,Finset.sum_range_zero,
    Fin.sum_univ_succ,term,matrixValue,matrixAddress,rowStart,columnStart,rowBlock,columnBlock]
  constructor <;> ring

theorem complete_region (swapped : Bool) : ∃ sf,
    exec (source p regions .fp16 .bf16 .fp16 .bf16 2 swapped).eraseDType.toAlgKernel initial = some sf ∧
    sf.readMem "A" 52 = 6*Real.exp 10 ∧ sf.readMem "A" 53 = 6*Real.exp 5 ∧
    sf.mem "A" 54 = MemCell.of .nat 99 := by
  obtain ⟨sf,he,ho,hf⟩ := RealInterExecution.source_executes p regions .fp16 .bf16 .fp16 .bf16 2 swapped initial
    (by intros; rfl)
  refine ⟨sf,he,(ho (by unfold enabled; decide) (0,0,PUnit.unit) (by decide)).trans formula.1,
    (ho (by unfold enabled; decide) (0,1,PUnit.unit) (by decide)).trans formula.2,?_⟩
  rw [← hf "A" 54]
  · rfl
  · rintro ⟨_,_,i,ha,heq⟩
    dsimp [active,outputAddress,rowStart,rowBlock,columnBlock,launch,p,initial] at ha heq
    omega

private def triangleSkip : BlockState := {initial with pids := fun ax => if ax = 1 then 3 else 1}
private def timeSkip : BlockState := {initial with pids := fun ax => if ax = 2 then 1 else 2}

theorem skipped (swapped : Bool) (s : BlockState) (hs : s = triangleSkip ∨ s = timeSkip) : ∃ sf,
    exec (source p regions .fp16 .bf16 .fp16 .bf16 2 swapped).eraseDType.toAlgKernel s = some sf ∧ sf.mem = s.mem := by
  have hz : ∀ region off, s.undef region off = 0 := by rcases hs with rfl | rfl <;> intros <;> rfl
  have hn : ¬ enabled p (launch s) := by rcases hs with rfl | rfl <;> unfold enabled <;> decide
  obtain ⟨sf,he,_,hf⟩ := RealInterExecution.source_executes p regions .fp16 .bf16 .fp16 .bf16 2 swapped s hz
  refine ⟨sf,he,?_⟩
  funext r off
  exact (hf r off (by rintro ⟨_,hen,_⟩; exact hn hen)).symm

theorem zero_features (swapped : Bool) : ∃ sf,
    exec (source {p with K := 0} regions .fp16 .bf16 .fp16 .bf16 2 swapped).eraseDType.toAlgKernel initial = some sf ∧
    sf.readMem "A" 52 = 0 := by
  obtain ⟨sf,he,ho,_⟩ := RealInterExecution.source_executes {p with K := 0} regions .fp16 .bf16 .fp16 .bf16 2 swapped initial
    (by intros; rfl)
  refine ⟨sf,he,?_⟩
  simpa [expected,numBlocks,partialSum,p] using ho (by unfold enabled; decide) (0,0,PUnit.unit) (by decide)

private def alloc : FlatAlloc where
  flat := "flat"
  regions := ["q","k","g","A"]
  base := fun r => if r = "q" then 0 else if r = "k" then 80 else if r = "g" then 160 else 240
  extent := fun r => if r = "q" ∨ r = "k" ∨ r = "g" then 80 else if r = "A" then 56 else 0
private theorem disjoint : alloc.Disjoint := by simp +decide [FlatAlloc.Disjoint,alloc,Region.mk.injEq]
private theorem covered : ∀ r, r ∉ alloc.regions → alloc.extent r = 0 := by
  intro r hn
  simp [alloc] at hn ⊢
  simp [hn]
private theorem bounds : RealInterSafety.Bounds p (launch initial) regions alloc.extent := by
  constructor
  · intro _ kb hkb
    have hkb' : kb < 2 := hkb
    constructor
    all_goals
      simp +decide [RealInterAccessSafety.QueryBounds,RealInterAccessSafety.KeyBounds,
        RealInterAccessSafety.ReferenceBounds,RealInterAccessSafety.MatrixBounds,
        referenceAddress,rowStart,columnStart,rowBlock,columnBlock,p,launch,initial,regions,alloc,Region.mk.injEq]
    all_goals intros; omega
  · intro _ i hi
    simp +decide [p,launch,initial,rowStart,rowBlock,columnBlock,active,outputAddress,regions,alloc,Region.mk.injEq] at hi ⊢
    omega

theorem complete_flat (swapped : Bool) : ∃ sf,
    exec (alloc.flattenKernel (source p regions .fp16 .bf16 .fp16 .bf16 2 swapped).eraseDType.toAlgKernel)
      (alloc.flattenState initial) = some sf ∧
    sf.readMem "flat" 292 = 6*Real.exp 10 ∧ sf.readMem "flat" 293 = 6*Real.exp 5 ∧
    sf.mem "flat" 294 = MemCell.of .nat 99 := by
  obtain ⟨sf,he,ho,hf⟩ := RealInterFlat.source_executes p regions .fp16 .bf16 .fp16 .bf16 2 swapped
    alloc disjoint covered initial rfl bounds
  refine ⟨sf,he,(ho (by unfold enabled; decide) (0,0,PUnit.unit) (by decide)).trans formula.1,
    (ho (by unfold enabled; decide) (0,1,PUnit.unit) (by decide)).trans formula.2,?_⟩
  rw [← hf "flat" 294]
  · change (alloc.flattenState initial).mem alloc.flat (alloc.addr "A" 54) = _
    rw [alloc.flattenState_mem_addr disjoint initial (by simp [alloc]) (by decide)]
    rfl
  · rintro ⟨_,_,i,ha,heq⟩
    dsimp [alloc,FlatAlloc.addr,active,outputAddress,rowStart,rowBlock,columnBlock,launch,p,initial] at ha heq
    norm_num at heq
    omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.ChunkGLAInterReal
