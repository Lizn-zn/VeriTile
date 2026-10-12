import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealSplitFlat

/-! Complete split intra-subblock executions with three columns, both causal arms,
a masked row, a partial K tile, nonzero launch and timeStride != K. Arbitrary
padding must not change real output. Both scaling orders and the early return
are covered, plus finite flat memory and tagged untouched cells. -/
namespace VeriTile.Tests.ChunkGLASplitReal
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open RealSplitMath RealSplitProgram
open RealSplitSetup (inputs)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private abbrev p : Layout := ⟨40,9,15,7,8,4,4,2⟩
private abbrev regions : Regions := ⟨"q","k","g","A"⟩
private def initial (padding : ℝ) : BlockState where
  mem := fun r off => if r = "g" then MemCell.real off
    else if r = "q" ∨ r = "k" then MemCell.real 1 else MemCell.of .nat 99
  regs := fun _ _ _ => none
  pids := fun d => if d = 1 then 3 else 1
  numPids := fun _ => 2
  undef := fun _ _ => padding

private theorem formula (padding : ℝ) :
    expected p (launch p (initial padding)) (inputs regions (initial padding)) 2 0 0 = 6*Real.exp 24 ∧
    expected p (launch p (initial padding)) (inputs regions (initial padding)) 2 0 1 = 0 ∧
    expected p (launch p (initial padding)) (inputs regions (initial padding)) 2 1 1 = 6*Real.exp 26 ∧
    expected p (launch p (initial padding)) (inputs regions (initial padding)) 2 2 2 = 6*Real.exp 28 := by
  have hl : launch p (initial padding) = ⟨1,1,1,1,2⟩ := by norm_num [launch,p,initial]
  rw [hl]
  change expected p ⟨1,1,1,1,2⟩ ⟨fun _ => 1,fun _ => 1,fun off => off⟩ 2 0 0 = _ ∧
    expected p ⟨1,1,1,1,2⟩ ⟨fun _ => 1,fun _ => 1,fun off => off⟩ 2 0 1 = _ ∧
    expected p ⟨1,1,1,1,2⟩ ⟨fun _ => 1,fun _ => 1,fun off => off⟩ 2 1 1 = _ ∧
    expected p ⟨1,1,1,1,2⟩ ⟨fun _ => 1,fun _ => 1,fun off => off⟩ 2 2 2 = _
  norm_num [expected,p,Finset.sum_range_succ,Finset.sum_range_zero,term,matrixAddress,keyAddress,rowStart,featureIndex,Fin.sum_univ_succ]
  repeat' apply And.intro
  all_goals ring

theorem complete_region (swapped : Bool) (padding : ℝ) : ∃ sf,
    exec (source p regions .fp16 .bf16 .fp16 2 swapped).eraseDType.toAlgKernel (initial padding) = some sf ∧
    sf.readMem "A" 228 = 6*Real.exp 24 ∧ sf.readMem "A" 229 = 0 ∧
    sf.readMem "A" 233 = 6*Real.exp 26 ∧ sf.readMem "A" 238 = 6*Real.exp 28 ∧
    sf.mem "A" 231 = MemCell.of .nat 99 ∧ sf.mem "A" 240 = MemCell.of .nat 99 := by
  obtain ⟨sf,he,ho,hf⟩ := RealSplitExecution.source_executes p regions .fp16 .bf16 .fp16 2 swapped (initial padding)
    (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive]) (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive])
  refine ⟨sf,he,(by simpa [regions,outputAddress,launch,initial,p,rowStart] using (ho (by norm_num [enabled,rowStart,launch,initial,p]) 0 (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive]) 0 (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive])).trans (formula padding).1),
    (by simpa [regions,outputAddress,launch,initial,p,rowStart] using (ho (by norm_num [enabled,rowStart,launch,initial,p]) 1 (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive]) 0 (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive])).trans (formula padding).2.1),
    (by simpa [regions,outputAddress,launch,initial,p,rowStart] using (ho (by norm_num [enabled,rowStart,launch,initial,p]) 1 (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive]) 1 (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive])).trans (formula padding).2.2.1),
    (by simpa [regions,outputAddress,launch,initial,p,rowStart] using (ho (by norm_num [enabled,rowStart,launch,initial,p]) 2 (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive]) 2 (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive])).trans (formula padding).2.2.2),?_,?_⟩
  all_goals
    rw [← hf "A" _]
    · rfl
    · rintro ⟨_,_,j,hj,i,ha,heq⟩
      dsimp [columns,rowActive,outputAddress,rowStart,launch,p,initial] at hj ha heq
      omega

theorem skipped (swapped : Bool) (padding : ℝ) : ∃ sf,
    exec (source p regions .fp16 .bf16 .fp16 2 swapped).eraseDType.toAlgKernel
      { initial padding with pids := fun d => if d = 1 then 5 else 1 } = some sf ∧ sf.mem = (initial padding).mem := by
  let s : BlockState := { initial padding with pids := fun d => if d = 1 then 5 else 1 }
  have hn : ¬enabled p (launch p s) := by norm_num [enabled,rowStart,launch,initial,p,s]
  obtain ⟨sf,he,_,hf⟩ := RealSplitExecution.source_executes p regions .fp16 .bf16 .fp16 2 swapped s
    (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive]) (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive])
  refine ⟨sf,he,?_⟩
  funext r off
  exact (hf r off (by rintro ⟨hen,_⟩; exact hn hen)).symm

theorem zero_features (swapped : Bool) (padding : ℝ) : ∃ sf,
    exec (source {p with K := 0} regions .fp16 .bf16 .fp16 2 swapped).eraseDType.toAlgKernel (initial padding) = some sf ∧
    sf.readMem "A" 228 = 0 := by
  obtain ⟨sf,he,ho,_⟩ := RealSplitExecution.source_executes {p with K := 0} regions .fp16 .bf16 .fp16 2 swapped (initial padding)
    (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive]) (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive])
  refine ⟨sf,he,?_⟩
  simpa [expected,p,featureIndex,outputAddress,rowStart,launch,initial,regions] using ho (by norm_num [enabled,rowStart,launch,initial,p]) 0 (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive]) 0 (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive])

private def alloc : FlatAlloc where
  flat := "flat"
  regions := ["q","k","g","A"]
  base := fun r => if r = "q" then 0 else if r = "k" then 180 else if r = "g" then 360 else 540
  extent := fun r => if r = "q" ∨ r = "k" ∨ r = "g" then 180 else if r = "A" then 240 else 0
private theorem disjoint : alloc.Disjoint := by simp +decide [FlatAlloc.Disjoint,alloc,Region.mk.injEq]
private theorem covered : ∀ r, r ∉ alloc.regions → alloc.extent r = 0 := by
  intro r hn
  simp [alloc] at hn ⊢
  simp [hn]
private theorem bounds : RealSplitSafety.Bounds p (launch p (initial 0)) regions alloc.extent := by
  constructor
  · intro _ i e hi he
    simp +decide [matrixAddress,featureIndex,rowStart,rowActive,p,launch,initial,regions,alloc,Region.mk.injEq] at hi he ⊢
    omega
  · intro _ i e hi he
    simp +decide [matrixAddress,featureIndex,rowStart,rowActive,p,launch,initial,regions,alloc,Region.mk.injEq] at hi he ⊢
    omega
  · intro _ j hj
    have hj' : j < 3 := hj
    constructor
    all_goals
      simp +decide [RealSplitAccessSafety.KeyBounds,RealSplitAccessSafety.OutputBounds,
        keyAddress,featureIndex,rowStart,rowActive,outputAddress,p,launch,initial,regions,alloc,Region.mk.injEq]
    all_goals intros; omega

theorem complete_flat (swapped : Bool) : ∃ sf,
    exec (alloc.flattenKernel (source p regions .fp16 .bf16 .fp16 2 swapped).eraseDType.toAlgKernel)
      (alloc.flattenState (initial 0)) = some sf ∧
    sf.readMem "flat" 768 = 6*Real.exp 24 ∧ sf.readMem "flat" 769 = 0 ∧
    sf.readMem "flat" 778 = 6*Real.exp 28 ∧ sf.mem "flat" 771 = MemCell.of .nat 99 := by
  obtain ⟨sf,he,ho,hf⟩ := RealSplitFlat.source_executes p regions .fp16 .bf16 .fp16 2 swapped
    alloc disjoint covered (initial 0) (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive]) (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive]) rfl bounds
  refine ⟨sf,he,(by simpa [regions,outputAddress,launch,initial,p,rowStart,alloc,FlatAlloc.addr] using (ho (by norm_num [enabled,rowStart,launch,initial,p]) 0 (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive]) 0 (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive])).trans (formula 0).1),
    (by simpa [regions,outputAddress,launch,initial,p,rowStart,alloc,FlatAlloc.addr] using (ho (by norm_num [enabled,rowStart,launch,initial,p]) 1 (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive]) 0 (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive])).trans (formula 0).2.1),
    (by simpa [regions,outputAddress,launch,initial,p,rowStart,alloc,FlatAlloc.addr] using (ho (by norm_num [enabled,rowStart,launch,initial,p]) 2 (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive]) 2 (by norm_num +decide [p,regions,Region.mk.injEq,columns,launch,initial,rowStart,rowActive])).trans (formula 0).2.2.2),?_⟩
  rw [← hf "flat" 771]
  · change (alloc.flattenState (initial 0)).mem alloc.flat (alloc.addr "A" 231) = _
    rw [alloc.flattenState_mem_addr disjoint (initial 0) (by simp [alloc]) (by decide)]
    rfl
  · rintro ⟨_,_,j,hj,i,ha,heq⟩
    dsimp [alloc,FlatAlloc.addr,columns,rowActive,outputAddress,rowStart,launch,p,initial] at hj ha heq
    omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.ChunkGLASplitReal
