import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealMergeFlat

/-! Complete merge regressions: three K slices, runtime head count three,
partial rows, arbitrary padding in region semantics, early return and NK=0.
The output formula uses initial memory even if the output aliases A. -/
namespace VeriTile.Tests.ChunkGLAMergeReal
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkGlaFwd RealMergeProgram
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

private abbrev p : Layout := ⟨5,4,2,3⟩
private def initial : BlockState where
  mem := fun r off => if r = "A" then MemCell.real off else MemCell.of .nat 99
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 1 then 0 else 1
  numPids := fun _ => 3
  undef := fun _ off => 1234+off
private theorem injective (s : BlockState) : NoCollision p (launch s) := by
  rintro ⟨i,j,⟨⟩⟩ ⟨k,l,⟨⟩⟩ _ _ he
  dsimp [outputAddress,p] at he
  have hi : i = k := Fin.ext (by omega)
  have hj : j = l := Fin.ext (by omega)
  subst k; subst l; rfl
private theorem formula : expected p (launch initial) (initial.readMem "A") (0,0,PUnit.unit) = 144 ∧
    expected p (launch initial) (initial.readMem "A") (0,1,PUnit.unit) = 147 := by
  have hr (off : Nat) : initial.readMem "A" off = off := rfl
  simp only [expected,show p.NK = 3 from rfl,Finset.sum_range_succ,Finset.sum_range_zero,zero_add,inputAddress,hr]
  norm_num [p,rowStart,launch,initial]

theorem complete_region (out : RegionName) (swapped : Bool) : ∃ sf,
    exec (source p "A" out .bf16 swapped).eraseDType.toAlgKernel initial = some sf ∧
    sf.readMem out 36 = 144 ∧ sf.readMem out 37 = 147 ∧ sf.mem out 38 = initial.mem out 38 := by
  obtain ⟨sf,he,ho,hf⟩ := RealMergeExecution.source_executes p "A" out .bf16 swapped initial (injective initial)
  refine ⟨sf,he,(ho (0,0,PUnit.unit) (by decide)).trans formula.1,
    (ho (0,1,PUnit.unit) (by decide)).trans formula.2,?_⟩
  apply (hf out 38 ?_).symm
  rintro ⟨_,_,i,ha,heq⟩
  dsimp [active,outputAddress,rowStart,launch,p,initial] at ha heq
  omega

private def disabled : BlockState := {initial with pids := fun _ => 2}
theorem early_return (swapped : Bool) : ∃ sf,
    exec (source p "A" "out" .fp16 swapped).eraseDType.toAlgKernel disabled = some sf ∧ sf.mem = disabled.mem := by
  obtain ⟨sf,he,_,hf⟩ := RealMergeExecution.source_executes p "A" "out" .fp16 swapped disabled (injective disabled)
  refine ⟨sf,he,?_⟩
  funext r off
  exact (hf r off (by rintro ⟨_,hen,_⟩; exact (by decide : ¬ enabled p (launch disabled)) hen)).symm

theorem zero_slices (swapped : Bool) : ∃ sf,
    exec (source {p with NK := 0} "A" "out" .fp32 swapped).eraseDType.toAlgKernel initial = some sf ∧
    sf.readMem "out" 36 = 0 := by
  obtain ⟨sf,he,ho,_⟩ := RealMergeExecution.source_executes {p with NK := 0} "A" "out" .fp32 swapped initial (injective initial)
  refine ⟨sf,he,?_⟩
  simpa [expected] using ho (0,0,PUnit.unit) (by decide)

private def flatInitial : BlockState := {initial with undef := fun _ _ => 0}
private def alloc : FlatAlloc where
  flat := "flat"
  regions := ["A","out"]
  base := fun r => if r = "A" then 100 else 200
  extent := fun r => if r = "A" then 80 else if r = "out" then 40 else 0
private theorem disjoint : alloc.Disjoint := by simp +decide [FlatAlloc.Disjoint,alloc,Region.mk.injEq]
private theorem covered : ∀ r, r ∉ alloc.regions → alloc.extent r = 0 := by
  intro r hn
  simp [alloc] at hn ⊢
  simp [hn]
private theorem bounds : RealMergeSafety.Bounds p (launch flatInitial) "A" "out" alloc.extent := by
  constructor
  · intro _ kb hkb i hi
    dsimp [p] at hkb
    simp +decide [p,launch,flatInitial,initial,rowStart,inputAddress,alloc,Region.mk.injEq] at hi ⊢
    omega
  · intro _ i hi
    simp +decide [p,launch,flatInitial,initial,rowStart,active,outputAddress,alloc,Region.mk.injEq] at hi ⊢
    omega

theorem complete_flat (swapped : Bool) : ∃ sf,
    exec (alloc.flattenKernel (source p "A" "out" .bf16 swapped).eraseDType.toAlgKernel) (alloc.flattenState flatInitial) = some sf ∧
    sf.readMem "flat" 236 = 144 ∧ sf.readMem "flat" 237 = 147 ∧ sf.mem "flat" 238 = MemCell.of .nat 99 := by
  obtain ⟨sf,he,ho,hf⟩ := RealMergeFlat.source_executes p "A" "out" .bf16 swapped alloc disjoint covered flatInitial rfl
    (injective flatInitial) bounds
  refine ⟨sf,he,(ho (0,0,PUnit.unit) (by decide)).trans formula.1,
    (ho (0,1,PUnit.unit) (by decide)).trans formula.2,?_⟩
  rw [← hf "flat" 238]
  · change (alloc.flattenState flatInitial).mem alloc.flat (alloc.addr "out" 38) = _
    rw [alloc.flattenState_mem_addr disjoint flatInitial (by simp [alloc]) (by decide)]
    rfl
  · rintro ⟨_,_,i,ha,heq⟩
    dsimp [alloc,FlatAlloc.addr,active,outputAddress,rowStart,launch,p,flatInitial,initial] at ha heq
    norm_num at heq
    omega

/-- For the host's K=512, T=128, BT=BC=16, B*H=4 case, the temporary
contains 4096 cells. The second K slice begins at cell 8192 for head zero.
This rejects that host allocation; it does not assume a GPU memory result. -/
theorem host_temporary_allocation_rejected :
    ¬ RealMergeSafety.Bounds ⟨128,16,16,4⟩ ⟨0,0,0,4⟩ "A" "out"
      (fun r => if r = "A" then 4096 else 8192) := by
  intro h
  have bad := h.input (by decide) 1 (by decide) (0,0,PUnit.unit) (by decide)
  norm_num [inputAddress,rowStart] at bad

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.ChunkGLAMergeReal
