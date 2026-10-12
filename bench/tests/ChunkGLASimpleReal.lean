import bench.tritonbench_g.chunk_gla_simple.Proofs.RealFlat

/-! Two K blocks, checked time/key/value tails, nonzero program IDs, complete
flat execution and preservation of a neighboring non-real memory tag. -/
namespace VeriTile.Tests.ChunkGLASimpleReal
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math Real
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
private def p : Layout := ⟨9,3,9,3,18,3,3,3,3,2,2,2⟩
private def regions : Regions := ⟨"q","k","v","h","g","o"⟩
private def initial : BlockState where
  mem := fun r _ => if r = "g" then MemCell.real 0 else if r = "o" then MemCell.of .nat 99 else MemCell.real 1
  regs := fun _ _ _ => none
  pids := fun _ => 1
private def alloc : FlatAlloc where
  flat := "flat"
  regions := ["q","k","v","h","g","o"]
  base := fun r => if r = "q" then 0 else if r = "k" then 100 else if r = "v" then 200
    else if r = "h" then 300 else if r = "g" then 400 else 500
  extent := fun r => if r = "h" then 36 else if r = "g" then 6
    else if r = "q" ∨ r = "k" ∨ r = "v" ∨ r = "o" then 18 else 0
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
private theorem bounds : Safe.Bounds p (launch initial) regions alloc.extent := by
  constructor
  · intro kb _
    constructor <;> intro i j hi
    all_goals simp +decide [p,launch,initial,regions,alloc,Region.mk.injEq] at hi ⊢
    all_goals omega
  · constructor
    · intro i hi
      simp +decide [row,gateAddress,p,launch,initial,regions,alloc,Region.mk.injEq] at hi ⊢
      omega
    · intro i j hi
      simp +decide [p,launch,initial,regions,alloc,Region.mk.injEq] at hi ⊢
      omega
    · intro i j hi
      simp +decide [p,launch,initial,regions,alloc,Region.mk.injEq] at hi ⊢
      omega
private theorem formula : expected p (launch initial) (inputs regions initial) 2 0 0 = 12 := by
  change expected p ⟨1,1,1⟩ ⟨fun _ => 1,fun _ => 1,fun _ => 1,fun _ => 1,fun _ => 0⟩ 2 0 0 = 12
  dsimp [expected,qhSum,qkSum,score,numBlocks,p]
  norm_num [query,key,history,value,gate,active,row,column,keyAddress,historyAddress,outputAddress,gateAddress,
    Finset.sum_range_succ,Finset.sum_range_zero,Fin.sum_univ_two]

theorem both_complete_flat (swapped : Bool) : ∃ sf,
    exec (alloc.flattenKernel (program p regions .fp16 .bf16 .bf16 .fp16 2 swapped).eraseDType.toAlgKernel)
      (alloc.flattenState initial) = some sf ∧
    sf.readMem "flat" 517 = 12 ∧ sf.mem "flat" 516 = MemCell.of .nat 99 := by
  obtain ⟨sf,he,ho,hf⟩ := Flat.source_executes p regions .fp16 .bf16 .bf16 .fp16 2 swapped
    alloc disjoint covered initial rfl injective bounds
  refine ⟨sf,he,(ho (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit) (by decide)).trans formula,?_⟩
  rw [← hf "flat" 516]
  · change (alloc.flattenState initial).mem alloc.flat (alloc.addr "o" 16) = _
    rw [alloc.flattenState_mem_addr disjoint initial (by simp [alloc]) (by decide)]
    rfl
  · rintro ⟨_,idx,ha,heq⟩
    dsimp [Flat.footprint,alloc,regions,outputAddress,active,row,column,launch,p,initial,FlatAlloc.addr] at heq ha
    omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.ChunkGLASimpleReal
