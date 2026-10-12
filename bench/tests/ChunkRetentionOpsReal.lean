import bench.tritonbench_g.chunk_retention_ops.Proofs.BackwardStateFlat

namespace VeriTile.Tests.ChunkRetentionOpsReal
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkRetention
open VeriTile.Bench.TritonBenchG.ChunkRetentionOps
open RealHMath (Layout keyIndex valueIndex historyAddress)
open RealDhMath RealDhProgram
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

private abbrev p : Layout := ⟨627,19,1,627,19,1,1083,19,2,33,19,19,16,16,16,3⟩
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

private def rate : ℝ := 63 / 64
private theorem base_value : RealHMath.base p l = Real.log rate / Real.log 2 := by
  have he : Real.exp ((-6 : ℝ) * Real.log 2) = 1 / 64 := by
    rw [show (-6 : ℝ) * Real.log 2 = -((6 : Nat) * Real.log 2) by ring,
      Real.exp_neg,Real.exp_nat_mul,Real.exp_log (by norm_num : (0 : ℝ) < 2)]
    norm_num
  dsimp [RealHMath.base,Decay.headBase,p,l,launch,u]
  norm_num only [Nat.reduceMod,Nat.cast_one,mul_one,sub_neg_eq_add] at *
  unfold Decay.exp2
  rw [he]
  norm_num [rate]
private theorem factor_value (x : ℝ) : Decay.factor (RealHMath.base p l) x = rate ^ x := by
  rw [Decay.factor,Decay.exp2,base_value,Real.rpow_def_of_pos (by norm_num [rate] : 0 < rate)]
  congr 1
  field_simp

private def firstExpected : ℝ := 6 * rate^17 + 6 * ∑ t : Fin 16, rate^(t.val+1)
private theorem first_value : closed p l a 3 0 16 16 = firstExpected := by
  norm_num [closed,RealDhMath.carry,contribution,query,incoming,Decay.query,factor_value,
    RealHMath.timeIndex,p,a,Fin.sum_univ_succ,rate,firstExpected]
private theorem middle_value : closed p l a 3 1 16 16 = 189 / 32 := by
  norm_num [closed,RealDhMath.carry,contribution,query,incoming,Decay.query,factor_value,
    RealHMath.timeIndex,p,a,Fin.sum_univ_succ,rate]
private theorem last_value : closed p l a 3 2 16 16 = 0 := by
  norm_num [closed,RealDhMath.carry,p]

private def A : FlatAlloc where
  flat := "flat"
  regions := ["q","do","dh"]
  base := fun region => if region = "q" then 100 else if region = "do" then 2100 else 6100
  extent := fun region =>
    if region = "q" ∨ region = "do" then 1254 else if region = "dh" then 2166 else 0

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
    change c < 3 at hc
    omega

theorem both_complete_flat (swapped : Bool) : ∃ v : BlockState,
    exec (A.flattenKernel (source p r .fp16 .fp16 3 swapped).eraseDType.toAlgKernel)
      (A.flattenState u) = some v ∧
    v.readMem "flat" 7503 = firstExpected ∧ v.readMem "flat" 7864 = 189 / 32 ∧
    v.readMem "flat" 8225 = 0 ∧
    v.mem "flat" 7502 = MemCell.of .nat 99 := by
  let io := RealDhIO.sourceIO p r .fp16 .fp16 3 swapped
  have hi : RealDhIO.IO.Implements io (RealDhIO.expected p 3) :=
    RealDhIO.implements p r .fp16 .fp16 3 swapped (by constructor <;> decide)
  obtain ⟨v,he,hh,hf⟩ := hi A allocations_disjoint allocations_covered u rfl history_injective allocated
  have hinputs : io.read u = a := rfl
  rw [hinputs] at hh
  refine ⟨v,he,(hh 0 0 0 (by decide) (by change 16 < 19 ∧ 16 < 19; decide)).trans first_value,
    (hh 1 0 0 (by decide) (by change 16 < 19 ∧ 16 < 19; decide)).trans middle_value,
    (hh 2 0 0 (by decide) (by change 16 < 19 ∧ 16 < 19; decide)).trans last_value,?_⟩
  rw [← hf "flat" 7502]
  · change (A.flattenState u).mem A.flat (A.addr "dh" 1402) = _
    rw [A.flattenState_mem_addr allocations_disjoint u (by simp [A]) (by decide)]
    rfl
  · rintro ⟨_,rg,off,⟨rfl,c,e,j,hc,ha,rfl⟩,heq⟩
    simp +decide [io,RealDhIO.sourceIO,historyAddress,keyIndex,valueIndex,
      RealDhStore.active,FlatAlloc.addr,p,l,launch,u,r,A,Region.mk.injEq] at ha heq
    omega

/-- Empty reverse range performs no history write or input load. -/
theorem empty_loop (swapped : Bool) : ∃ v : BlockState,
    exec (A.flattenKernel (source {p with NT := 0} r .fp16 .fp16 3 swapped).eraseDType.toAlgKernel)
      (A.flattenState u) = some v ∧ v.mem = (A.flattenState u).mem := by
  obtain ⟨v,he,_,hf⟩ := RealDhFlat.source_executes A allocations_disjoint allocations_covered
    {p with NT := 0} r .fp16 .fp16 3 swapped a u rfl (by constructor <;> decide)
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
end VeriTile.Tests.ChunkRetentionOpsReal

#guard_msgs (drop info) in
#auditModuleAxioms
