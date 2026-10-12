import bench.tritonbench_g.chunk_retention.Proofs.ForwardStateFlat
import bench.tritonbench_g.chunk_retention.Proofs.ForwardOutputFlat
import bench.tritonbench_g.chunk_retention.Proofs.BackwardStateFlat
import bench.tritonbench_g.chunk_retention.Proofs.BackwardGradientsFlat

/-! Both complete sources, all seed/final branches, two 16-wide chunks,
partial time/key/value tails and nonzero program IDs. Full finite flat
execution preserves adjacent tagged cells and disabled final-state storage. -/
namespace VeriTile.Tests.ChunkRetentionReal.ForwardState
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkRetention
open RealHMath RealHProgram
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

private abbrev p : Layout := ⟨323,19,1,323,19,1,722,19,2,17,19,19,16,16,16,2⟩
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
private def rate : ℝ := 63 / 64
private def afterFirst (hasSeed : Bool) : ℝ := rate ^ (16 : Nat) * seedValue hasSeed + 128 * (1 - rate ^ (16 : Nat))
private def afterLast (hasSeed : Bool) : ℝ := rate ^ (17 : Nat) * seedValue hasSeed + 128 * (1 - rate ^ (17 : Nat))

private theorem base_value : base p l = Real.log rate / Real.log 2 := by
  have he : Real.exp ((-6 : ℝ) * Real.log 2) = 1 / 64 := by
    rw [show (-6 : ℝ) * Real.log 2 = -((6 : Nat) * Real.log 2) by ring,
      Real.exp_neg,Real.exp_nat_mul,Real.exp_log (by norm_num : (0 : ℝ) < 2)]
    norm_num
  dsimp [base,Decay.headBase,p,l,launch,u]
  norm_num only [Nat.reduceMod,Nat.cast_one,mul_one,sub_neg_eq_add] at *
  unfold Decay.exp2
  rw [he]
  norm_num [rate]

private theorem factor_value (x : ℝ) : Decay.factor (base p l) x = rate ^ x := by
  rw [Decay.factor,Decay.exp2,base_value,Real.rpow_def_of_pos (by norm_num [rate] : 0 < rate)]
  congr 1
  have hlog : Real.log (2 : ℝ) ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  field_simp

private theorem state_one (hasSeed : Bool) :
    closed p l a hasSeed 1 16 16 = afterFirst hasSeed := by
  rw [closed_step p l a hasSeed 0,state_zero]
  norm_num [contribution,keyValue,valueValue,timeIndex,carryFactor,length,Decay.chunkLength,
    Decay.key,factor_value,p,a,Fin.sum_univ_succ,afterFirst,rate]
private theorem state_two (hasSeed : Bool) :
    closed p l a hasSeed 2 16 16 = afterLast hasSeed := by
  rw [closed_step p l a hasSeed 1,state_one]
  norm_num [contribution,keyValue,valueValue,timeIndex,carryFactor,length,Decay.chunkLength,
    Decay.key,factor_value,p,a,Fin.sum_univ_succ,afterFirst,afterLast,rate]
  ring

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
    v.readMem "flat" 7503 = afterFirst hasSeed ∧
    (storeFinal = Bool.true → v.readMem "flat" 10781 = afterLast hasSeed) ∧
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
end VeriTile.Tests.ChunkRetentionReal.ForwardState

/-! Two K blocks, checked time/key/value tails, nonzero program IDs, complete
nonunit key/value feature strides, flat execution and preservation of a neighboring non-real memory tag. -/
namespace VeriTile.Tests.ChunkRetentionReal.ForwardOutput
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkRetention OutputMath RealOutput
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
private abbrev p : Layout := ⟨722,38,2,722,38,2,722,19,2,19,19,19,16,16,16⟩
private def regions : Regions := ⟨"q","k","v","h","o"⟩
private def initial : BlockState where
  mem := fun r _ => if r = "o" then MemCell.of .nat 99 else MemCell.real 1
  regs := fun _ _ _ => none
  pids := fun _ => 1
private def alloc : FlatAlloc where
  flat := "flat"
  regions := ["q","k","v","h","o"]
  base := fun r => if r = "q" then 0 else if r = "k" then 2000 else if r = "v" then 4000
    else if r = "h" then 6000 else 8000
  extent := fun r => if r = "h" then 1444
    else if r = "q" ∨ r = "k" ∨ r = "v" ∨ r = "o" then 1444 else 0
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
private def rate : ℝ := 63 / 64
private theorem base_value : base p (launch initial) = Real.log rate / Real.log 2 := by
  have he : Real.exp ((-6 : ℝ) * Real.log 2) = 1 / 64 := by
    rw [show (-6 : ℝ) * Real.log 2 = -((6 : Nat) * Real.log 2) by ring,
      Real.exp_neg,Real.exp_nat_mul,Real.exp_log (by norm_num : (0 : ℝ) < 2)]
    norm_num
  dsimp [base,Decay.headBase,p,launch,initial]
  norm_num only [Nat.reduceMod,Nat.cast_one,mul_one,sub_neg_eq_add] at *
  unfold Decay.exp2
  rw [he]
  norm_num [rate]

private theorem factor_value (x : ℝ) : Decay.factor (base p (launch initial)) x = rate ^ x := by
  rw [Decay.factor,Decay.exp2,base_value,Real.rpow_def_of_pos (by norm_num [rate] : 0 < rate)]
  congr 1
  have hlog : Real.log (2 : ℝ) ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  field_simp

private theorem formula : expected p (launch initial) (inputs regions initial) 2 0 0 = 2413 / 32 := by
  change expected p (launch initial) ⟨fun _ => 1,fun _ => 1,fun _ => 1,fun _ => 1⟩ 2 _ _ = _
  simp only [expected,qhSum,score,qkSum,Decay.query,Decay.score,factor_value]
  dsimp only [p,numBlocks]
  norm_num only [Nat.reduceAdd,Nat.reduceSub,Nat.reduceDiv]
  norm_num [query,key,history,value,active,row,column,keyAddress,historyAddress,outputAddress,
    Decay.query,Decay.score,factor_value,p,launch,initial,inputs,regions,rate,
    Finset.sum_range_succ,Finset.sum_range_zero,Fin.sum_univ_succ]

private theorem second_row : expected p (launch initial) (inputs regions initial) 2 1 0 = 229843 / 2048 := by
  change expected p (launch initial) ⟨fun _ => 1,fun _ => 1,fun _ => 1,fun _ => 1⟩ 2 _ _ = _
  simp only [expected,qhSum,score,qkSum,Decay.query,Decay.score,factor_value]
  dsimp only [p,numBlocks]
  norm_num only [Nat.reduceAdd,Nat.reduceSub,Nat.reduceDiv]
  norm_num [query,key,history,value,active,row,column,keyAddress,historyAddress,outputAddress,
    Decay.query,Decay.score,factor_value,p,launch,initial,inputs,regions,rate,
    Finset.sum_range_succ,Finset.sum_range_zero,Fin.sum_univ_succ]

theorem both_complete_flat (swapped : Bool) : ∃ sf,
    exec (alloc.flattenKernel (program p regions .fp16 .bf16 .fp16 2 swapped).eraseDType.toAlgKernel)
      (alloc.flattenState initial) = some sf ∧
    sf.readMem "flat" 9362 = 2413 / 32 ∧ sf.readMem "flat" 9400 = 229843 / 2048 ∧ sf.mem "flat" 9361 = MemCell.of .nat 99 := by
  obtain ⟨sf,he,ho,hf⟩ := OutputFlat.source_executes p regions .fp16 .bf16 .fp16 2 swapped
    alloc disjoint covered initial rfl injective bounds
  refine ⟨sf,he,(ho (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit) (by decide)).trans formula,
    (ho (⟨1,by decide⟩,⟨0,by decide⟩,PUnit.unit) (by decide)).trans second_row,?_⟩
  rw [← hf "flat" 9361]
  · change (alloc.flattenState initial).mem alloc.flat (alloc.addr "o" 1361) = _
    rw [alloc.flattenState_mem_addr disjoint initial (by simp [alloc]) (by decide)]
    rfl
  · rintro ⟨_,idx,ha,heq⟩
    dsimp [OutputFlat.footprint,alloc,regions,outputAddress,active,row,column,launch,p,initial,FlatAlloc.addr] at heq ha
    omega

end
end VeriTile.Tests.ChunkRetentionReal.ForwardOutput

/-! Two reverse chunks, 16-wide square tiles, nonzero program IDs and partial
time/value/key tails. Checks the supplied final-store address, a zero-valued
stored column, and an untouched neighboring non-real memory tag. -/
namespace VeriTile.Tests.ChunkRetentionReal.BackwardState
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkRetention RealDhMath RealDhProgram RealDhStep RealDhFinal
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
private abbrev p : Layout := ⟨7,11,13,722,38,2,722,19,2,19,19,19,16,2⟩
private def regions : Regions := ⟨"q","do","dh","v"⟩
private def initial : BlockState where
  mem := fun r _ => if r = "do" then MemCell.real 2 else if r = "v" then MemCell.real 3 else MemCell.of .nat 99
  regs := fun _ _ _ => none
  pids := fun _ => 1
private def alloc : FlatAlloc where
  flat := "flat"
  regions := ["do","v","dh"]
  base := fun r => if r = "do" then 100 else if r = "v" then 2100 else 6000
  extent := fun r => if r = "do" ∨ r = "v" ∨ r = "dh" then 1444 else 0
private theorem disjoint : alloc.Disjoint := by simp +decide [FlatAlloc.Disjoint,alloc,Region.mk.injEq]
private theorem covered : ∀ r, r ∉ alloc.regions → alloc.extent r = 0 := by
  intro r hn
  simp [alloc] at hn ⊢
  simp [hn]
private theorem injective : NoCollision p (launch initial) := by
  rintro ⟨i,j,⟨⟩⟩ ⟨k,l,⟨⟩⟩ _ _ he
  dsimp [outputAddress,launch,p,initial] at he
  have hi : i = k := Fin.ext (by omega)
  have hj : j = l := Fin.ext (by omega)
  subst k; subst l; rfl
private theorem bounds : RealDhSafety.Bounds p (launch initial) regions alloc.extent := by
  constructor
  · intro c hc
    constructor <;> intro i j hi
    all_goals simp +decide [p,launch,initial,regions,alloc,Region.mk.injEq] at hi ⊢
    all_goals change c < 2 at hc
    all_goals omega
  · intro i j hi
    simp +decide [p,launch,initial,regions,alloc,Region.mk.injEq] at hi ⊢
    omega
private def rate : ℝ := 63 / 64
private theorem base_value : base p (launch initial) = Real.log rate / Real.log 2 := by
  have he : Real.exp ((-6 : ℝ) * Real.log 2) = 1 / 64 := by
    rw [show (-6 : ℝ) * Real.log 2 = -((6 : Nat) * Real.log 2) by ring,
      Real.exp_neg,Real.exp_nat_mul,Real.exp_log (by norm_num : (0 : ℝ) < 2)]
    norm_num
  dsimp [base,Decay.headBase,p,launch,initial]
  norm_num only [Nat.reduceMod,Nat.cast_one,mul_one,sub_neg_eq_add] at *
  unfold Decay.exp2
  rw [he]
  norm_num [rate]

private theorem factor_value (x : ℝ) : Decay.factor (base p (launch initial)) x = rate ^ x := by
  rw [Decay.factor,Decay.exp2,base_value,Real.rpow_def_of_pos (by norm_num [rate] : 0 < rate)]
  congr 1
  have hlog : Real.log (2 : ℝ) ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  field_simp

private theorem formula : expected p (launch initial) (inputs regions initial) 0 0 = 36 * rate ^ (17 : Nat) := by
  change expected p (launch initial) ⟨fun _ => 2,fun _ => 3⟩ 0 0 = _
  simp only [expected,RealDhMath.carry,contribution,Decay.query,factor_value]
  norm_num [p,incoming,value,address,launch,initial,Finset.sum_range_succ,Fin.sum_univ_succ,rate]

private theorem zero_column : expected p (launch initial) (inputs regions initial) 0 15 = 0 := by
  change expected p (launch initial) ⟨fun _ => 2,fun _ => 3⟩ 0 15 = _
  simp only [expected,RealDhMath.carry,contribution,Decay.query,factor_value]
  norm_num [p,incoming,value,address,launch,initial,Finset.sum_range_succ,Fin.sum_univ_succ,rate]

theorem both_complete_flat (swapped : Bool) : ∃ sf,
    exec (alloc.flattenKernel (source p regions .bf16 .fp16 17 swapped).eraseDType.toAlgKernel)
      (alloc.flattenState initial) = some sf ∧
    sf.readMem "flat" 7387 = 36 * rate ^ (17 : Nat) ∧
    sf.readMem "flat" 7402 = 0 ∧ sf.mem "flat" 7386 = MemCell.of .nat 99 := by
  obtain ⟨sf,he,ho,hf⟩ := RealDhFlat.source_executes p regions .bf16 .fp16 17 swapped
    alloc disjoint covered initial rfl injective bounds
  refine ⟨sf,he,(ho (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit) (by decide)).trans formula,
    (ho (⟨0,by decide⟩,⟨15,by decide⟩,PUnit.unit) (by decide)).trans zero_column,?_⟩
  rw [← hf "flat" 7386]
  · change (alloc.flattenState initial).mem alloc.flat (alloc.addr "dh" 1386) = _
    rw [alloc.flattenState_mem_addr disjoint initial (by simp [alloc]) (by decide)]
    rfl
  · rintro ⟨_,idx,ha,heq⟩
    dsimp [RealDhFlat.footprint,alloc,regions,outputAddress,active,launch,p,initial,FlatAlloc.addr] at heq ha
    omega
end
end VeriTile.Tests.ChunkRetentionReal.BackwardState

/-! Both complete sources with two value blocks, 16-wide tiles, a three-row
time tail, a three-feature key tail and nonzero launch IDs. Runtime head count
is two, so dV must use the extra key-block slice. All three outputs and adjacent
tagged-cell preservation are checked in finite flat memory. -/
namespace VeriTile.Tests.ChunkRetentionReal.BackwardGradients
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkRetention
open RealHMath (Layout)
open RealQKVMath RealQKVProgram
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private abbrev p : Layout := ⟨361,19,1,361,19,1,722,19,2,19,19,19,16,16,16,2⟩
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

private def rate : ℝ := 63 / 64
private theorem base_value : base p l = Real.log rate / Real.log 2 := by
  have he : Real.exp ((-6 : ℝ) * Real.log 2) = 1 / 64 := by
    rw [show (-6 : ℝ) * Real.log 2 = -((6 : Nat) * Real.log 2) by ring,
      Real.exp_neg,Real.exp_nat_mul,Real.exp_log (by norm_num : (0 : ℝ) < 2)]
    norm_num
  dsimp [base,Decay.headBase,p,l,launch,u]
  norm_num only [Nat.reduceMod,Nat.cast_one,mul_one,sub_neg_eq_add] at *
  unfold Decay.exp2
  rw [he]
  norm_num [rate]

private theorem factor_value (x : ℝ) : Decay.factor (base p l) x = rate ^ x := by
  rw [Decay.factor,Decay.exp2,base_value,Real.rpow_def_of_pos (by norm_num [rate] : 0 < rate)]
  congr 1
  have hlog : Real.log (2 : ℝ) ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  field_simp

private def queryResult : ℝ := 760 * rate + 1140
private def queryNext : ℝ := 760 * rate^ (2 : Nat) + 1140 * (rate+1)
private def keyResult : ℝ := 342 * rate^ (15 : Nat) + 570 * (1+rate+rate^ (2 : Nat))
private def valueResult : ℝ := 36 * rate^ (15 : Nat) + 60 * (1+rate+rate^ (2 : Nat))

private theorem query_value : dq p l a 2 0 0 = queryResult := by
  simp only [dq,dk,dv,dqSum,dkSum,ds,dsSum,RealQKVMath.score,
    queryDecay,keyDecay,scoreDecay,Decay.query,Decay.key,Decay.score,factor_value]
  norm_num [incoming,value,history,historyGradient,query,key,row,keyIndex,valueIndex,historyIndex,
    numBlocks,p,a,l,launch,u,Finset.sum_range_succ,Fin.sum_univ_succ,
    queryResult,queryNext,keyResult,valueResult,rate]

private theorem query_next : dq p l a 2 1 0 = queryNext := by
  simp only [dq,dk,dv,dqSum,dkSum,ds,dsSum,RealQKVMath.score,
    queryDecay,keyDecay,scoreDecay,Decay.query,Decay.key,Decay.score,factor_value]
  norm_num [incoming,value,history,historyGradient,query,key,row,keyIndex,valueIndex,historyIndex,
    numBlocks,p,a,l,launch,u,Finset.sum_range_succ,Fin.sum_univ_succ,
    queryResult,queryNext,keyResult,valueResult,rate]

private theorem key_value : dk p l a 2 0 0 = keyResult := by
  simp only [dq,dk,dv,dqSum,dkSum,ds,dsSum,RealQKVMath.score,
    queryDecay,keyDecay,scoreDecay,Decay.query,Decay.key,Decay.score,factor_value]
  norm_num [incoming,value,history,historyGradient,query,key,row,keyIndex,valueIndex,historyIndex,
    numBlocks,p,a,l,launch,u,Finset.sum_range_succ,Fin.sum_univ_succ,
    queryResult,queryNext,keyResult,valueResult,rate]

private theorem value_zero : dv p l a 2 0 0 0 = valueResult := by
  simp only [dq,dk,dv,dqSum,dkSum,ds,dsSum,RealQKVMath.score,
    queryDecay,keyDecay,scoreDecay,Decay.query,Decay.key,Decay.score,factor_value]
  norm_num [incoming,value,history,historyGradient,query,key,row,keyIndex,valueIndex,historyIndex,
    numBlocks,p,a,l,launch,u,Finset.sum_range_succ,Fin.sum_univ_succ,
    queryResult,queryNext,keyResult,valueResult,rate]

private theorem value_one : dv p l a 2 1 0 0 = valueResult := by
  simp only [dq,dk,dv,dqSum,dkSum,ds,dsSum,RealQKVMath.score,
    queryDecay,keyDecay,scoreDecay,Decay.query,Decay.key,Decay.score,factor_value]
  norm_num [incoming,value,history,historyGradient,query,key,row,keyIndex,valueIndex,historyIndex,
    numBlocks,p,a,l,launch,u,Finset.sum_range_succ,Fin.sum_univ_succ,
    queryResult,queryNext,keyResult,valueResult,rate]

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
    if region = "q" ∨ region = "k" ∨ region = "v" ∨ region = "do" ∨ region = "dq" ∨ region = "dk" then 722
    else if region = "h" ∨ region = "dh" then 1444 else if region = "dv" then 1444 else 0
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
    w.readMem "flat" 12781 = queryResult ∧ w.readMem "flat" 14781 = keyResult ∧
    w.readMem "flat" 17487 = valueResult ∧ w.readMem "flat" 17503 = valueResult ∧
    w.readMem "flat" 12800 = queryNext ∧ w.mem "flat" 12780 = MemCell.of .nat 99 := by
  let io := RealQKVIO.sourceIO p r .bf16 2 swapped
  have hi : RealQKVIO.IO.Implements io (RealQKVIO.expected p 2) :=
    RealQKVIO.implements p r .bf16 2 swapped (by constructor; constructor <;> decide; all_goals decide)
  obtain ⟨w,he,hq,hk,hv,hf⟩ := hi A allocations_disjoint allocations_covered u rfl
    ⟨value_injective,key_injective⟩ allocated
  have hinputs : io.read u = a := rfl
  rw [hinputs] at hq hk hv
  refine ⟨w,he,(hq 0 0 (by change 16 < 19 ∧ 16 < 19; decide)).trans query_value,(hk 0 0 (by change 16 < 19 ∧ 16 < 19; decide)).trans key_value,
    (hv 0 0 0 (by decide) (by change 16 < 19 ∧ 0 < 19; decide)).trans value_zero,
    (hv 1 0 0 (by decide) (by change 16 < 19 ∧ 16 < 19; decide)).trans value_one,(hq 1 0 (by change 17 < 19 ∧ 16 < 19; decide)).trans query_next,?_⟩
  rw [← hf "flat" 12780]
  · change (A.flattenState u).mem A.flat (A.addr "dq" 680) = _
    rw [A.flattenState_mem_addr allocations_disjoint u (by simp [A]) (by decide)]
    rfl
  · rintro ⟨_,rg,off,hw,heq⟩
    rcases hw with ⟨rfl,n,i,j,hn,ha,rfl⟩ | (⟨rfl,i,e,ha,rfl⟩ | ⟨rfl,i,e,ha,rfl⟩)
    all_goals
      simp +decide [io,RealQKVIO.sourceIO,keyAddress,dvAddress,keyActive,valueActive,row,keyIndex,valueIndex,
        FlatAlloc.addr,p,l,launch,u,r,A,Region.mk.injEq] at ha heq
      omega

end
end VeriTile.Tests.ChunkRetentionReal.BackwardGradients

#guard_msgs (drop info) in
#auditModuleAxioms
