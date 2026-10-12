import bench.tritonbench_g.chunk_gated_attention.Correct

/-! Complete-source execution at nonzero head/key/value program IDs, with
16-wide tiles and partial time/key/value tails. The gate is nonzero: the full
first chunk uses exp(2), whereas the final chunk's checked-off endpoint uses
zero and its sole valid product uses exp(-2). All switches and both source
versions are covered. This is a real-semantics test, not a GPU observation. -/
namespace VeriTile.Tests.GatedAttentionReal
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

private abbrev p : Layout := ⟨323,19,1,323,19,1,722,19,1,17,19,19,16,16,16,2⟩
private def r : Regions := ⟨"k","v","g","h","h0","ht"⟩
private def a : Inputs := ⟨fun _ => 1,fun _ => 2,fun _ => 2,fun _ => 3⟩
private def u : BlockState where
  mem := fun region _ =>
    if region = "k" then MemCell.real 1
    else if region = "v" ∨ region = "g" then MemCell.real 2
    else if region = "h0" then MemCell.real 3
    else MemCell.of .nat 99
  regs := fun _ _ _ => none
  pids := fun _ => 1
private def l := launch u
private def seedValue (hasSeed : Bool) : ℝ := if hasSeed then 3 else 0

private theorem history_injective : RealHLoop.HistoryNoCollision p l := by
  intro c d e f i j _ _ ha hb he
  dsimp [RealHStore.active,keyIndex,valueIndex,RealHLoop.address,RealHStep.historyBase,
    RealHStore.address,p,l,launch,u] at ha hb he
  have h : c = d ∧ e.val = f.val ∧ i.val = j.val := by omega
  exact ⟨h.1,Fin.ext h.2.1,Fin.ext h.2.2⟩
private theorem final_injective : RealHStore.NoCollision p l (RealHFinal.base p l) p.V 1 := by
  intro e f i j ha hb he
  dsimp [RealHStore.active,keyIndex,valueIndex,RealHStore.address,RealHFinal.base,p,l,launch,u] at ha hb he
  have h : e.val = f.val ∧ i.val = j.val := by omega
  exact ⟨Fin.ext h.1,Fin.ext h.2⟩

private theorem state_zero (gateK hasSeed : Bool) : closed p l a gateK hasSeed 0 16 16 = seedValue hasSeed := by
  cases hasSeed <;> simp [seedValue,seed,p,a]
private theorem state_one (gateK hasSeed : Bool) :
    closed p l a gateK hasSeed 1 16 16 = seedValue hasSeed * Real.exp 2 + 32 := by
  rw [closed_step p l a gateK hasSeed 0,state_zero]
  cases gateK <;> norm_num [gateFactor,endpoint,endpointIndex,contribution,keyValue,valueValue,
    RealHMath.gateValue,timeIndex,p,a,Fin.sum_univ_succ]
private theorem state_two (gateK hasSeed : Bool) :
    closed p l a gateK hasSeed 2 16 16 = seedValue hasSeed * Real.exp 2 + 32 + 2*Real.exp (-2) := by
  rw [closed_step p l a gateK hasSeed 1,state_one]
  cases gateK <;> norm_num [gateFactor,endpoint,endpointIndex,contribution,keyValue,valueValue,
    RealHMath.gateValue,timeIndex,p,a,Fin.sum_univ_succ]
  ring

/-- Both full erased sources, including disabled final-store behavior and a
neighboring tagged cell outside the active history tile. -/
theorem both_complete_sources (gateK hasSeed storeFinal swapped : Bool) : ∃ v : BlockState,
    exec (source p r .fp16 .bf16 .bf16 .bf16 .fp32 gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel u = some v ∧
    v.readMem "h" 1042 = seedValue hasSeed ∧
    v.readMem "h" 1403 = seedValue hasSeed * Real.exp 2 + 32 ∧
    (storeFinal = Bool.true → v.readMem "ht" 681 = seedValue hasSeed * Real.exp 2 + 32 + 2*Real.exp (-2)) ∧
    v.mem "h" 1041 = MemCell.of .nat 99 ∧
    (storeFinal = Bool.false → v.mem "ht" 681 = MemCell.of .nat 99) := by
  obtain ⟨v,he,ho,hf⟩ := RealHExecution.executes p r .fp16 .bf16 .bf16 .bf16 .fp32
    gateK hasSeed storeFinal swapped a u rfl (by constructor <;> decide)
    (by intro _; decide) history_injective (fun _ => final_injective)
    (by intro off; simp [u,r,a,BlockState.readMem,MemCell.readAs,MemCell.real,MemCell.of])
    (by intro off; simp [u,r,a,BlockState.readMem,MemCell.readAs,MemCell.real,MemCell.of])
    (by intro off; simp [u,r,a,BlockState.readMem,MemCell.readAs,MemCell.real,MemCell.of])
    (by intro off; simp [u,r,a,BlockState.readMem,MemCell.readAs,MemCell.real,MemCell.of])
  refine ⟨v,he,(ho.history 0 0 0 (by decide) (by decide)).trans (state_zero gateK hasSeed),
    (ho.history 1 0 0 (by decide) (by decide)).trans (state_one gateK hasSeed),?_,?_,?_⟩
  · intro hon
    exact (ho.final hon 0 0 (by decide)).trans (state_two gateK hasSeed)
  · have h := hf "h" 1041 (by
      rintro (hh | hh)
      · rcases hh with ⟨_,c,e,j,hc,ha,he⟩
        dsimp [RealHStore.active,keyIndex,valueIndex,RealHLoop.address,RealHStep.historyBase,
          RealHStore.address,p,launch,u] at ha he
        omega
      · exact (by decide : ("h" : RegionName) ≠ "ht") hh.2.1)
    exact h.symm
  · intro hoff
    have h := hf "ht" 681 (by
      rintro (hh | hh)
      · exact (by decide : ("ht" : RegionName) ≠ "h") hh.1
      · simp [RealHFinal.footprint,hoff] at hh)
    exact h.symm


private def A : FlatAlloc where
  flat := "flat"
  regions := ["k","v","g","h","h0","ht"]
  base := fun region =>
    if region = "k" then 100 else if region = "v" then 2100
    else if region = "g" then 4100 else if region = "h" then 6100
    else if region = "h0" then 8100 else 10100
  extent := fun region =>
    if region = "k" ∨ region = "v" ∨ region = "g" then 646
    else if region = "h" then 1444 else if region = "h0" ∨ region = "ht" then 722 else 0

private theorem allocations_disjoint : A.Disjoint := by
  simp +decide [FlatAlloc.Disjoint,A,Region.mk.injEq]

private theorem allocations_covered : ∀ rg, rg ∉ A.regions → A.extent rg = 0 := by
  intro rg hn
  simp [A] at hn ⊢
  simp [hn]

/-- The finite allocations cover actual endpoint-vector accesses as well as
matrix tiles. In particular, no feature mask is added to the flattened endpoint. -/
private theorem allocated (gateK hasSeed storeFinal : Bool) :
    RealHSafety.Bounds p l r gateK hasSeed storeFinal A.extent := by
  constructor
  · intro _ i j ha
    simp +decide [RealHStore.active,RealHAccessSafety.StateBounds,RealHAccessSafety.MatrixBounds,
      keyIndex,valueIndex,p,l,launch,u,r,A,Region.mk.injEq] at ha ⊢
    omega
  · intro c hc
    constructor
    · intro i j ha
      simp +decide [RealHAccessSafety.KeyBounds,RealHAccessSafety.MatrixBounds,p,l,launch,u,r,A,Region.mk.injEq] at ha ⊢
      omega
    · intro i j ha
      simp +decide [RealHAccessSafety.ValueBounds,RealHAccessSafety.MatrixBounds,p,l,launch,u,r,A,Region.mk.injEq] at ha ⊢
      omega
    · intro i j ha
      simp +decide [RealHAccessSafety.HistoryBounds,RealHAccessSafety.MatrixBounds,p,l,launch,u,r,A,Region.mk.injEq] at ha ⊢
      change c < 2 at hc
      omega
    · constructor
      · intro _ i j ha
        simp +decide [RealHAccessSafety.KeyBounds,RealHAccessSafety.MatrixBounds,p,l,launch,u,r,A,Region.mk.injEq] at ha ⊢
        omega
      · intro _ i ha
        simp +decide [RealHAccessSafety.EndpointBounds,RealHAccessSafety.VectorBounds,p,l,launch,u,r,A,Region.mk.injEq] at ha ⊢
        omega
      · intro _ i j ha
        simp +decide [RealHAccessSafety.ValueBounds,RealHAccessSafety.MatrixBounds,p,l,launch,u,r,A,Region.mk.injEq] at ha ⊢
        omega
      · intro _ i ha
        simp +decide [RealHAccessSafety.EndpointBounds,RealHAccessSafety.VectorBounds,p,l,launch,u,r,A,Region.mk.injEq] at ha ⊢
        omega
  · intro _ i j ha
    simp +decide [RealHAccessSafety.StateBounds,RealHAccessSafety.MatrixBounds,p,l,launch,u,r,A,Region.mk.injEq] at ha ⊢
    omega

/-- Exercise both public real specifications with finite disjoint allocations,
including all optional branches. No successful run or output is a premise. -/
theorem both_complete_flat (gateK hasSeed storeFinal swapped : Bool) : ∃ v : BlockState,
    exec (A.flattenKernel (source p r .fp16 .bf16 .bf16 .bf16 .fp32 gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel)
      (A.flattenState u) = some v ∧
    v.readMem "flat" 7142 = seedValue hasSeed ∧
    v.readMem "flat" 7503 = seedValue hasSeed * Real.exp 2 + 32 ∧
    (storeFinal = Bool.true → v.readMem "flat" 10781 = seedValue hasSeed * Real.exp 2 + 32 + 2*Real.exp (-2)) ∧
    v.mem "flat" 7141 = MemCell.of .nat 99 ∧
    (storeFinal = Bool.false → v.mem "flat" 10781 = MemCell.of .nat 99) := by
  let io := RealHIO.sourceIO p r .fp16 .bf16 .bf16 .bf16 .fp32 gateK hasSeed storeFinal swapped
  have hi : RealHIO.IO.Implements io (Correctness.recurrentExpected p gateK hasSeed) := by
    cases swapped
    · exact Correctness.recurrent_correctness p r .fp16 .bf16 .bf16 .bf16 .fp32 gateK hasSeed storeFinal
        (by constructor <;> decide) (by intro _; decide)
    · exact Correctness.recurrent_sum_left_correctness p r .fp16 .bf16 .bf16 .bf16 .fp32 gateK hasSeed storeFinal
        (by constructor <;> decide) (by intro _; decide)
  obtain ⟨v,he,hh,hfinal,hf⟩ := hi A allocations_disjoint allocations_covered u rfl
    ⟨history_injective,fun _ => final_injective⟩ (allocated gateK hasSeed storeFinal)
  have hinputs : io.read u = a := rfl
  rw [hinputs] at hh hfinal
  refine ⟨v,he,(hh 0 0 0 (by decide) (by change 16 < 19 ∧ 16 < 19; decide)).trans (state_zero gateK hasSeed),
    (hh 1 0 0 (by decide) (by change 16 < 19 ∧ 16 < 19; decide)).trans (state_one gateK hasSeed),?_,?_,?_⟩
  · intro hon
    exact (hfinal hon 0 0 (by change 16 < 19 ∧ 16 < 19; decide)).trans (state_two gateK hasSeed)
  · rw [← hf "flat" 7141]
    · change (A.flattenState u).mem A.flat (A.addr "h" 1041) = _
      rw [A.flattenState_mem_addr allocations_disjoint u (by simp [A]) (by decide)]
      rfl
    · rintro ⟨_,rg,off,hw,heq⟩
      rcases hw with ⟨rfl,c,e,j,hc,ha,rfl⟩ | ⟨_,rfl,e,j,ha,rfl⟩
      all_goals
        simp +decide [io,RealHIO.sourceIO,historyAddress,stateAddress,keyIndex,valueIndex,
          RealHStore.active,FlatAlloc.addr,p,l,launch,u,r,A,Region.mk.injEq] at ha heq
        omega
  · intro hoff
    rw [← hf "flat" 10781]
    · change (A.flattenState u).mem A.flat (A.addr "ht" 681) = _
      rw [A.flattenState_mem_addr allocations_disjoint u (by simp [A]) (by decide)]
      rfl
    · rintro ⟨_,rg,off,hw,heq⟩
      rcases hw with ⟨rfl,c,e,j,hc,ha,rfl⟩ | ⟨hon,_,_⟩
      · simp +decide [io,RealHIO.sourceIO,historyAddress,keyIndex,valueIndex,
          RealHStore.active,FlatAlloc.addr,p,l,launch,u,r,A,Region.mk.injEq] at ha heq
        change c < 2 at hc
        omega
      · simp [io,RealHIO.sourceIO,hoff] at hon

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.GatedAttentionReal
