import bench.tritonbench_g.chunk_linear_attn.Proofs.ForwardOutputFlat
import bench.tritonbench_g.chunk_linear_attn.Proofs.ForwardStateFlat
import bench.tritonbench_g.chunk_linear_attn.Proofs.BackwardStateFlat
import bench.tritonbench_g.chunk_linear_attn.Proofs.BackwardGradientsFlat

/-! Two K blocks, checked time/key/value tails, nonzero program IDs, complete
nonunit key/value feature strides, flat execution and preservation of a neighboring non-real memory tag. -/
namespace VeriTile.Tests.ChunkLinearAttnReal.ForwardOutput
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkLinearAttn OutputMath RealOutput
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
private def p : Layout := ⟨18,6,2,18,6,2,18,3,3,3,3,2,2,2⟩
private def regions : Regions := ⟨"q","k","v","h","o"⟩
private def initial : BlockState where
  mem := fun r _ => if r = "o" then MemCell.of .nat 99 else MemCell.real 1
  regs := fun _ _ _ => none
  pids := fun _ => 1
private def alloc : FlatAlloc where
  flat := "flat"
  regions := ["q","k","v","h","o"]
  base := fun r => if r = "q" then 0 else if r = "k" then 100 else if r = "v" then 200
    else if r = "h" then 300 else 500
  extent := fun r => if r = "h" then 36
    else if r = "q" ∨ r = "k" ∨ r = "v" ∨ r = "o" then 36 else 0
private theorem disjoint : alloc.Disjoint := by simp +decide [FlatAlloc.Disjoint,alloc,Region.mk.injEq]
private theorem covered : ∀ r, r ∉ alloc.regions → alloc.extent r = 0 := by
  intro r hn
  simp [alloc] at hn ⊢
  simp [hn]
private theorem injective : NoCollision p (launch initial) := by
  rintro ⟨i,j,⟨⟩⟩ ⟨k,l,⟨⟩⟩ _ _ he
  dsimp [outputAddress,row,column,launch,p,initial] at he
  have hi : i = k := Fin.ext (by omega)
  have hj : j = l := Fin.ext (by omega)
  subst k; subst l; rfl
private theorem bounds : OutputSafe.Bounds p (launch initial) regions alloc.extent := by
  constructor
  · intro kb _
    constructor <;> intro i j hi
    all_goals simp +decide [p,launch,initial,regions,alloc,Region.mk.injEq] at hi ⊢
    all_goals omega
  · constructor
    · intro i j hi
      simp +decide [p,launch,initial,regions,alloc,Region.mk.injEq] at hi ⊢
      omega
    · intro i j hi
      simp +decide [p,launch,initial,regions,alloc,Region.mk.injEq] at hi ⊢
      omega
private theorem formula : expected p (launch initial) (inputs regions initial) 2 0 0 = 12 := by
  change expected p ⟨1,1,1⟩ ⟨fun _ => 1,fun _ => 1,fun _ => 1,fun _ => 1⟩ 2 0 0 = 12
  dsimp [expected,qhSum,qkSum,score,numBlocks,p]
  norm_num [query,key,history,value,active,row,column,keyAddress,historyAddress,outputAddress,
    Finset.sum_range_succ,Finset.sum_range_zero,Fin.sum_univ_two]

theorem both_complete_flat (swapped : Bool) : ∃ sf,
    exec (alloc.flattenKernel (program p regions .fp16 .bf16 .fp16 2 swapped).eraseDType.toAlgKernel)
      (alloc.flattenState initial) = some sf ∧
    sf.readMem "flat" 534 = 12 ∧ sf.mem "flat" 533 = MemCell.of .nat 99 := by
  obtain ⟨sf,he,ho,hf⟩ := OutputFlat.source_executes p regions .fp16 .bf16 .fp16 2 swapped
    alloc disjoint covered initial rfl injective bounds
  refine ⟨sf,he,(ho (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit) (by decide)).trans formula,?_⟩
  rw [← hf "flat" 533]
  · change (alloc.flattenState initial).mem alloc.flat (alloc.addr "o" 33) = _
    rw [alloc.flattenState_mem_addr disjoint initial (by simp [alloc]) (by decide)]
    rfl
  · rintro ⟨_,idx,ha,heq⟩
    dsimp [OutputFlat.footprint,alloc,regions,outputAddress,active,row,column,launch,p,initial,FlatAlloc.addr] at heq ha
    omega

end
end VeriTile.Tests.ChunkLinearAttnReal.ForwardOutput

/-! Both complete sources, all seed/final branches, two 16-wide chunks,
partial time/key/value tails and nonzero program IDs. Full finite flat
execution preserves adjacent tagged cells and disabled final-state storage. -/
namespace VeriTile.Tests.ChunkLinearAttnReal.ForwardState
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkLinearAttn
open RealHMath RealHProgram
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

private abbrev p : Layout := ⟨323,19,1,323,19,1,722,19,17,19,19,16,16,16,2⟩
private def r : Regions := ⟨"k","v","h","h0","ht"⟩
private def a : Inputs := ⟨fun _ => 1,fun _ => 2,fun _ => 3⟩
private def u : BlockState where
  mem := fun region _ =>
    if region = "k" then MemCell.real 1
    else if region = "v" then MemCell.real 2
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

private theorem state_zero (hasSeed : Bool) : closed p l a hasSeed 0 16 16 = seedValue hasSeed := by
  cases hasSeed <;> simp [seedValue,seed,p,a]
private theorem state_one (hasSeed : Bool) :
    closed p l a hasSeed 1 16 16 = seedValue hasSeed + 32 := by
  rw [closed_step p l a hasSeed 0,state_zero]
  norm_num [contribution,keyValue,valueValue,timeIndex,p,a,Fin.sum_univ_succ]
private theorem state_two (hasSeed : Bool) :
    closed p l a hasSeed 2 16 16 = seedValue hasSeed + 32 + 2 := by
  rw [closed_step p l a hasSeed 1,state_one]
  norm_num [contribution,keyValue,valueValue,timeIndex,p,a,Fin.sum_univ_succ]

private def A : FlatAlloc where
  flat := "flat"
  regions := ["k","v","h","h0","ht"]
  base := fun region =>
    if region = "k" then 100 else if region = "v" then 2100
    else if region = "h" then 6100
    else if region = "h0" then 8100 else 10100
  extent := fun region =>
    if region = "k" ∨ region = "v" then 646
    else if region = "h" then 1444 else if region = "h0" ∨ region = "ht" then 722 else 0

private theorem allocations_disjoint : A.Disjoint := by
  simp +decide [FlatAlloc.Disjoint,A,Region.mk.injEq]

private theorem allocations_covered : ∀ rg, rg ∉ A.regions → A.extent rg = 0 := by
  intro rg hn
  simp [A] at hn ⊢
  simp [hn]

/- Actual active matrix bounds in finite allocations. -/
private theorem allocated (hasSeed storeFinal : Bool) :
    RealHSafety.Bounds p l r hasSeed storeFinal A.extent := by
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
  · intro _ i j ha
    simp +decide [RealHAccessSafety.StateBounds,RealHAccessSafety.MatrixBounds,p,l,launch,u,r,A,Region.mk.injEq] at ha ⊢
    omega

/-- Exercise both full real contracts with finite disjoint allocations,
including all optional branches. No successful run or output is a premise. -/
theorem both_complete_flat (hasSeed storeFinal swapped : Bool) : ∃ v : BlockState,
    exec (A.flattenKernel (source p r .fp16 .bf16 .fp32 hasSeed storeFinal swapped).eraseDType.toAlgKernel)
      (A.flattenState u) = some v ∧
    v.readMem "flat" 7142 = seedValue hasSeed ∧
    v.readMem "flat" 7503 = seedValue hasSeed + 32 ∧
    (storeFinal = Bool.true → v.readMem "flat" 10781 = seedValue hasSeed + 32 + 2) ∧
    v.mem "flat" 7141 = MemCell.of .nat 99 ∧
    (storeFinal = Bool.false → v.mem "flat" 10781 = MemCell.of .nat 99) := by
  let io := RealHIO.sourceIO p r .fp16 .bf16 .fp32 hasSeed storeFinal swapped
  have hi : RealHIO.IO.Implements io (RealHIO.expected p hasSeed) :=
    RealHIO.implements p r .fp16 .bf16 .fp32 hasSeed storeFinal swapped
      (by constructor <;> decide) (by intro _; decide)
  obtain ⟨v,he,hh,hfinal,hf⟩ := hi A allocations_disjoint allocations_covered u rfl
    ⟨history_injective,fun _ => final_injective⟩ (allocated hasSeed storeFinal)
  have hinputs : io.read u = a := rfl
  rw [hinputs] at hh hfinal
  refine ⟨v,he,(hh 0 0 0 (by decide) (by change 16 < 19 ∧ 16 < 19; decide)).trans (state_zero hasSeed),
    (hh 1 0 0 (by decide) (by change 16 < 19 ∧ 16 < 19; decide)).trans (state_one hasSeed),?_,?_,?_⟩
  · intro hon
    exact (hfinal hon 0 0 (by change 16 < 19 ∧ 16 < 19; decide)).trans (state_two hasSeed)
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

end
end VeriTile.Tests.ChunkLinearAttnReal.ForwardState

/-! Reverse traversal with two 16-wide chunks and a one-element time tail.
Both complete sources use mixed query/gradient storage, nonzero program IDs,
finite allocations, and a scale different from one. Also check NT=0. -/
namespace VeriTile.Tests.ChunkLinearAttnReal.BackwardState
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkLinearAttn
open RealHMath (Layout keyIndex valueIndex historyAddress)
open RealDhMath RealDhProgram
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

private abbrev p : Layout := ⟨323,19,1,323,19,1,722,19,17,19,19,16,16,16,2⟩
private def r : Regions := ⟨"q","do","dh"⟩
private def a : Inputs := ⟨fun _ => 1,fun _ => 2⟩
private def u : BlockState where
  mem := fun region _ =>
    if region = "q" then MemCell.real 1
    else if region = "do" then MemCell.real 2
    else MemCell.of .nat 99
  regs := fun _ _ _ => none
  pids := fun _ => 1
private def l := launch u

private theorem history_injective : RealDhLoop.HistoryNoCollision p l := by
  intro c d e f i j _ _ ha hb he
  dsimp [RealDhStore.active,keyIndex,valueIndex,RealDhLoop.address,RealDhStep.historyBase,
    RealDhStore.address,p,l,launch,u] at ha hb he
  have h : c = d ∧ e.val = f.val ∧ i.val = j.val := by omega
  exact ⟨h.1,Fin.ext h.2.1,Fin.ext h.2.2⟩

private theorem first_value : closed p l a 3 0 16 16 = 6 := by
  norm_num [closed,contribution,query,incoming,RealHMath.timeIndex,p,a,Fin.sum_univ_succ]
private theorem last_value : closed p l a 3 1 16 16 = 0 := by
  norm_num [closed,p,Fin.sum_univ_succ]

private def A : FlatAlloc where
  flat := "flat"
  regions := ["q","do","dh"]
  base := fun region => if region = "q" then 100 else if region = "do" then 2100 else 6100
  extent := fun region =>
    if region = "q" ∨ region = "do" then 646 else if region = "dh" then 1444 else 0

private theorem allocations_disjoint : A.Disjoint := by
  simp +decide [FlatAlloc.Disjoint,A,Region.mk.injEq]
private theorem allocations_covered : ∀ rg, rg ∉ A.regions → A.extent rg = 0 := by
  intro rg hn
  simp [A] at hn ⊢
  simp [hn]

private theorem allocated : RealDhSafety.Bounds p l r A.extent := by
  intro c hc
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

theorem both_complete_flat (swapped : Bool) : ∃ v : BlockState,
    exec (A.flattenKernel (source p r .fp16 .bf16 .fp16 3 swapped).eraseDType.toAlgKernel)
      (A.flattenState u) = some v ∧
    v.readMem "flat" 7142 = 6 ∧ v.readMem "flat" 7503 = 0 ∧
    v.mem "flat" 7141 = MemCell.of .nat 99 := by
  let io := RealDhIO.sourceIO p r .fp16 .bf16 .fp16 3 swapped
  have hi : RealDhIO.IO.Implements io (RealDhIO.expected p 3) :=
    RealDhIO.implements p r .fp16 .bf16 .fp16 3 swapped (by constructor <;> decide)
  obtain ⟨v,he,hh,hf⟩ := hi A allocations_disjoint allocations_covered u rfl history_injective allocated
  have hinputs : io.read u = a := rfl
  rw [hinputs] at hh
  refine ⟨v,he,(hh 0 0 0 (by decide) (by change 16 < 19 ∧ 16 < 19; decide)).trans first_value,
    (hh 1 0 0 (by decide) (by change 16 < 19 ∧ 16 < 19; decide)).trans last_value,?_⟩
  rw [← hf "flat" 7141]
  · change (A.flattenState u).mem A.flat (A.addr "dh" 1041) = _
    rw [A.flattenState_mem_addr allocations_disjoint u (by simp [A]) (by decide)]
    rfl
  · rintro ⟨_,rg,off,⟨rfl,c,e,j,hc,ha,rfl⟩,heq⟩
    simp +decide [io,RealDhIO.sourceIO,historyAddress,keyIndex,valueIndex,
      RealDhStore.active,FlatAlloc.addr,p,l,launch,u,r,A,Region.mk.injEq] at ha heq
    omega

/-- Empty reverse range performs no history write or input load. -/
theorem empty_loop (swapped : Bool) : ∃ v : BlockState,
    exec (A.flattenKernel (source {p with NT := 0} r .fp16 .bf16 .fp16 3 swapped).eraseDType.toAlgKernel)
      (A.flattenState u) = some v ∧ v.mem = (A.flattenState u).mem := by
  obtain ⟨v,he,_,hf⟩ := RealDhFlat.source_executes A allocations_disjoint allocations_covered
    {p with NT := 0} r .fp16 .bf16 .fp16 3 swapped a u rfl (by constructor <;> decide)
    (by intro c d e f i j hc; exact (Nat.not_lt_zero c hc).elim)
    (by intro off; rfl) (by intro off; rfl)
    (by intro c hc; exact (Nat.not_lt_zero c hc).elim)
  refine ⟨v,he,?_⟩
  funext region off
  symm
  apply hf region off
  rintro ⟨_,rg,k,⟨_,c,e,j,hc,_⟩,_⟩
  exact Nat.not_lt_zero c hc

end
end VeriTile.Tests.ChunkLinearAttnReal.BackwardState

/-! Both complete sources with two value blocks, 16-wide tiles, a one-row
time tail, a three-feature key tail and nonzero launch IDs. Runtime head count
is two, so dV must use the extra key-block slice. All three outputs and adjacent
tagged-cell preservation are checked in finite flat memory. -/
namespace VeriTile.Tests.ChunkLinearAttnReal.BackwardGradients
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkLinearAttn
open RealHMath (Layout)
open RealQKVMath RealQKVProgram
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private abbrev p : Layout := ⟨323,19,1,323,19,1,722,19,17,19,19,16,16,16,2⟩
private def r : Regions := ⟨"q","k","v","h","do","dh","dq","dk","dv"⟩
private def a : Inputs := ⟨fun _ => 1,fun _ => 2,fun _ => 3,fun _ => 4,fun _ => 5,fun _ => 6⟩
private def u : BlockState where
  mem := fun region _ =>
    if region = "q" then MemCell.real 1 else if region = "k" then MemCell.real 2
    else if region = "v" then MemCell.real 3 else if region = "h" then MemCell.real 4
    else if region = "do" then MemCell.real 5 else if region = "dh" then MemCell.real 6
    else MemCell.of .nat 99
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
private def l := launch u

private theorem query_value : dq p l a 2 0 0 = 1900 := by
  norm_num [dq,dqSum,ds,dsSum,incoming,value,history,key,row,keyIndex,valueIndex,historyIndex,
    numBlocks,p,a,l,launch,u,Finset.sum_range_succ,Fin.sum_univ_succ]
private theorem key_value : dk p l a 2 0 0 = 912 := by
  norm_num [dk,dkSum,ds,dsSum,incoming,value,historyGradient,query,row,keyIndex,valueIndex,historyIndex,
    numBlocks,p,a,l,launch,u,Finset.sum_range_succ,Fin.sum_univ_succ]
private theorem value_zero : dv p l a 2 0 0 0 = 96 := by
  norm_num [dv,RealQKVMath.score,incoming,key,query,historyGradient,row,keyIndex,valueIndex,historyIndex,
    p,a,l,launch,u,Fin.sum_univ_succ]
private theorem value_one : dv p l a 2 1 0 0 = 96 := by
  norm_num [dv,RealQKVMath.score,incoming,key,query,historyGradient,row,keyIndex,valueIndex,historyIndex,
    p,a,l,launch,u,Fin.sum_univ_succ]

private theorem value_injective : RealQKVLoop.NoCollision p l r := by
  intro c d i e j f hc hd ha hb he
  dsimp [RealQKVStep.valueWindow,RealMatrixStore.Window.active,RealMatrixStore.Window.address,p,l,launch,u] at ha hb he
  have h : c = d ∧ i.val = e.val ∧ j.val = f.val := by omega
  exact ⟨h.1,Fin.ext h.2.1,Fin.ext h.2.2⟩
private theorem key_injective : (RealQKVFinal.keyWindow p l r.dq).NoCollision := by
  intro i e j f ha hb he
  dsimp [RealQKVFinal.keyWindow,RealMatrixStore.Window.active,RealMatrixStore.Window.address,p,l,launch,u] at ha hb he
  have h : i.val = e.val ∧ j.val = f.val := by omega
  exact ⟨Fin.ext h.1,Fin.ext h.2⟩

private def A : FlatAlloc where
  flat := "flat"
  regions := ["q","k","v","h","do","dh","dq","dk","dv"]
  base := fun region =>
    if region = "q" then 100 else if region = "k" then 2100 else if region = "v" then 4100
    else if region = "h" then 6100 else if region = "do" then 8100 else if region = "dh" then 10100
    else if region = "dq" then 12100 else if region = "dk" then 14100 else 16100
  extent := fun region =>
    if region = "q" ∨ region = "k" ∨ region = "v" ∨ region = "do" ∨ region = "dq" ∨ region = "dk" then 646
    else if region = "h" ∨ region = "dh" then 1444 else if region = "dv" then 1292 else 0
private theorem allocations_disjoint : A.Disjoint := by
  simp +decide [FlatAlloc.Disjoint,A,Region.mk.injEq]
private theorem allocations_covered : ∀ rg, rg ∉ A.regions → A.extent rg = 0 := by
  intro rg hn
  simp [A] at hn ⊢
  simp [hn]
private theorem allocated : RealQKVSafety.Bounds p l r A.extent := by
  constructor
  · intro i j ha
    simp +decide [RealQKVSafety.QueryBounds,RealHAccessSafety.MatrixBounds,p,l,launch,u,r,A,Region.mk.injEq] at ha ⊢
    omega
  · intro i j ha
    simp +decide [RealQKVSafety.KeyBounds,RealHAccessSafety.MatrixBounds,p,l,launch,u,r,A,Region.mk.injEq] at ha ⊢
    omega
  · intro n hn
    constructor
    all_goals
      intro i j ha
      simp +decide [RealQKVSafety.ValueBounds,RealQKVSafety.HistoryBounds,RealQKVSafety.GradientBounds,
        RealQKVSafety.DVBounds,RealHAccessSafety.MatrixBounds,p,l,launch,u,r,A,Region.mk.injEq] at ha ⊢
      omega
  · intro i j ha
    simp +decide [RealQKVSafety.KeyBounds,RealHAccessSafety.MatrixBounds,p,l,launch,u,r,A,Region.mk.injEq] at ha ⊢
    omega
  · intro i j ha
    simp +decide [RealQKVSafety.KeyBounds,RealHAccessSafety.MatrixBounds,p,l,launch,u,r,A,Region.mk.injEq] at ha ⊢
    omega

theorem both_complete_flat (swapped : Bool) : ∃ w : BlockState,
    exec (A.flattenKernel (source p r .bf16 2 swapped).eraseDType.toAlgKernel) (A.flattenState u) = some w ∧
    w.readMem "flat" 12743 = 1900 ∧ w.readMem "flat" 14743 = 912 ∧
    w.readMem "flat" 17373 = 96 ∧ w.readMem "flat" 17389 = 96 ∧
    w.mem "flat" 12742 = MemCell.of .nat 99 := by
  let io := RealQKVIO.sourceIO p r .bf16 2 swapped
  have hi : RealQKVIO.IO.Implements io (RealQKVIO.expected p 2) :=
    RealQKVIO.implements p r .bf16 2 swapped (by constructor; constructor <;> decide; all_goals decide)
  obtain ⟨w,he,hq,hk,hv,hf⟩ := hi A allocations_disjoint allocations_covered u rfl
    ⟨value_injective,key_injective⟩ allocated
  have hinputs : io.read u = a := rfl
  rw [hinputs] at hq hk hv
  refine ⟨w,he,(hq 0 0 (by change 16 < 17 ∧ 16 < 19; decide)).trans query_value,(hk 0 0 (by change 16 < 17 ∧ 16 < 19; decide)).trans key_value,
    (hv 0 0 0 (by decide) (by change 16 < 17 ∧ 0 < 19; decide)).trans value_zero,
    (hv 1 0 0 (by decide) (by change 16 < 17 ∧ 16 < 19; decide)).trans value_one,?_⟩
  rw [← hf "flat" 12742]
  · change (A.flattenState u).mem A.flat (A.addr "dq" 642) = _
    rw [A.flattenState_mem_addr allocations_disjoint u (by simp [A]) (by decide)]
    rfl
  · rintro ⟨_,rg,off,hw,heq⟩
    rcases hw with ⟨rfl,n,i,j,hn,ha,rfl⟩ | (⟨rfl,i,e,ha,rfl⟩ | ⟨rfl,i,e,ha,rfl⟩)
    all_goals
      simp +decide [io,RealQKVIO.sourceIO,keyAddress,dvAddress,keyActive,valueActive,row,keyIndex,valueIndex,
        FlatAlloc.addr,p,l,launch,u,r,A,Region.mk.injEq] at ha heq
      omega

end
end VeriTile.Tests.ChunkLinearAttnReal.BackwardGradients

#guard_msgs (drop info) in
#auditModuleAxioms
