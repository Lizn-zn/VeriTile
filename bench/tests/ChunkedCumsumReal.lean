import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.Flat
import VeriTile.Meta.StatementAudit

/-! Complete region execution of both sources. Bias and finite clamping
produce dt=[0,2,3,0]; rate=2 produces cumulative=[0,4,10,10]. The last lane
is sequence padding and the second head lane is masked out. -/
namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Regression
open VeriTile.Triton Math Program Initialization Execution
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

private abbrev p : Layout := ⟨2,7,3,4,2,4,21,3,1,1,1,24,4,8,1,24,4,8,1⟩
private def r : Regions := ⟨"dt","A","bias","out","sum"⟩
private def a : Inputs where
  dt := fun off => if off = 35 then -2 else if off = 38 then 1 else 4
  rate := fun _ => 2
  bias := fun _ => 1
private def u : BlockState where
  mem := fun region off =>
    if region = "dt" then MemCell.real (a.dt off)
    else if region = "A" then MemCell.real 2
    else if region = "bias" then MemCell.real 1
    else MemCell.of .nat 99
  regs := fun _ _ _ => none
  pids := fun _ => 1
private def l := launch u

private theorem injective : Stores.NoCollision p l (outOffsets p l) := by
  intro i k j m ha hb he
  dsimp [activeOutput,head,outOffsets,outAddress,p,l,launch,u] at ha hb he
  have h : i.val = k.val ∧ j.val = m.val := by omega
  exact ⟨Fin.ext h.1,Fin.ext h.2⟩

private theorem dt_formula (j : Fin 4) :
    prepared p l a Bool.true Bool.false 0 3 0 j.val = ![0,2,3,0] j := by
  fin_cases j <;> norm_num [UpperBound.cap,prepared,activeInput,head,dtAddress,p,l,launch,u,a]
private theorem sum_formula (j : Fin 4) :
    cumulative p l a Bool.true Bool.false 0 3 0 j = ![0,4,10,10] j := by
  fin_cases j <;> norm_num [UpperBound.cap,cumulative,prepared,activeInput,head,dtAddress,rate,
    p,l,launch,u,a,Fin.sum_univ_succ]

theorem both_complete (swapped : Bool) : ∃ v : BlockState,
    exec (kernel p r .bf16 swapped .bf16 .fp16 Bool.true Bool.false 0 3).eraseDType.toAlgKernel u = some v ∧
    (∀ j : Fin 4, v.readMem "out" (44+j.val) = ![0,2,3,0] j ∧
      v.readMem "sum" (44+j.val) = ![0,4,10,10] j) ∧
    v.mem "out" 48 = MemCell.of .nat 99 ∧ v.mem "sum" 48 = MemCell.of .nat 99 := by
  obtain ⟨v,he,hv,hf⟩ := Execution.correct p r .bf16 swapped .bf16 .fp16 a Bool.true Bool.false 0 3 u
    (by intro off; simp [u,r,BlockState.readMem])
    (by intro off; simp [u,r,a,BlockState.readMem])
    (by intro off; simp [u,r,a,BlockState.readMem]) (by decide) (by decide) injective injective
  refine ⟨v,he,?_,?_,?_⟩
  · intro j
    have h := hv 0 j (by change 2 < 3 ∧ j.val < 4; exact ⟨by decide,j.isLt⟩)
    constructor
    · simpa [r,outAddress,head,p,launch,u] using h.1.trans (dt_formula j)
    · simpa [r,sumAddress,head,p,launch,u] using h.2.trans (sum_formula j)
  all_goals
    rw [← hf]
    · rfl
    · rintro (⟨hr,i,j,ha,heq⟩ | ⟨hr,i,j,ha,heq⟩)
      all_goals simp +decide [Stores.footprint,footprint,outOffsets,sumOffsets,outAddress,sumAddress,
        activeOutput,head,p,r,l,launch,u,Region.mk.injEq] at *
      all_goals omega

/-- Exact finite allocations include adjacent untouched sentinel cells. -/
private def allocation : FlatAlloc where
  flat := "flat"
  regions := ["dt","A","bias","out","sum"]
  base := fun r => if r = "dt" then 0 else if r = "A" then 64 else
    if r = "bias" then 128 else if r = "out" then 192 else 256
  extent := fun r => if r = "dt" then 42 else if r = "A" ∨ r = "bias" then 3 else
    if r = "out" ∨ r = "sum" then 49 else 0

private theorem allocation_disjoint : allocation.Disjoint := by
  simp +decide [FlatAlloc.Disjoint,allocation]

private theorem allocation_covers : ∀ rg, rg ∉ allocation.regions → allocation.extent rg = 0 := by
  intro rg hn
  simp only [allocation,List.mem_cons,List.mem_nil_iff,or_false,not_or] at hn
  simp [allocation,hn.1,hn.2.1,hn.2.2.1,hn.2.2.2.1,hn.2.2.2.2]

private theorem bounds : Safety.Bounds p l r Bool.true allocation.extent := by
  constructor
  · intro i j hi
    simp +decide [allocation,r,p,activeInput,head,dtAddress,l,launch,u] at hi ⊢
    omega
  · intro _ i hi
    simp +decide [allocation,r,p,head,l,launch,u] at hi ⊢
    omega
  · intro i hi
    simp +decide [allocation,r,p,head,l,launch,u] at hi ⊢
    omega
  · intro i j hi
    simp +decide [allocation,r,p,activeOutput,head,outAddress,l,launch,u] at hi ⊢
    omega
  · intro i j hi
    simp +decide [allocation,r,p,activeOutput,head,sumAddress,l,launch,u] at hi ⊢
    omega

/-- Nonzero launch IDs, partial head/time tails and mixed storage, now run
inside finite disjoint allocations. Both limits and both source versions have
successful full executions and a frame on every tagged flat-memory cell. -/
theorem flat_complete (swapped : Bool) (upper : UpperBound) : ∃ v : BlockState,
    exec (allocation.flattenKernel
      (kernel p r .bf16 swapped .bf16 .fp16 Bool.true Bool.false 0 upper).eraseDType.toAlgKernel)
      (allocation.flattenState u) = some v ∧
    (∀ j : Fin 4,
      v.readMem "flat" (236+j.val) = prepared p l a Bool.true Bool.false 0 upper 0 j.val ∧
      v.readMem "flat" (300+j.val) = cumulative p l a Bool.true Bool.false 0 upper 0 j) ∧
    BlockState.WriteWithin (Flat.footprint allocation p l r) (allocation.flattenState u) v := by
  obtain ⟨v,he,hv,hf⟩ := Flat.correct allocation allocation_disjoint allocation_covers
    p r .bf16 swapped .bf16 .fp16 a Bool.true Bool.false 0 upper u
    (by intro off; simp [u,r,BlockState.readMem])
    (by intro off; simp [u,r,a,BlockState.readMem])
    (by intro off; simp [u,r,a,BlockState.readMem])
    (by decide) (by decide) injective injective bounds
  refine ⟨v,he,?_,hf⟩
  intro j
  have h := hv 0 j (by change 2 < 3 ∧ j.val < 4; exact ⟨by decide,j.isLt⟩)
  simpa [FlatAlloc.addr,allocation,r,outAddress,sumAddress,head,p,l,launch,u,← Nat.add_assoc] using h

/-- At the default infinite upper bound, the source retains 5 instead of
clamping it to 3; the padded cumulative lane retains the preceding sum. -/
theorem default_values (j : Fin 4) :
    prepared p l a Bool.true Bool.false 0 .posInf 0 j.val = ![0,2,5,0] j ∧
    cumulative p l a Bool.true Bool.false 0 .posInf 0 j = ![0,4,14,14] j := by
  fin_cases j <;> norm_num [UpperBound.cap,cumulative,prepared,activeInput,head,dtAddress,rate,
    p,l,launch,u,a,Fin.sum_univ_succ]

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Regression

#axiomsClean VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Regression.both_complete
#axiomsClean VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Regression.flat_complete
#axiomsClean VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Regression.default_values
