import bench.tritonbench_g.chunk_gated_attention.Proofs.RealCumFlat

/-! Complete cumulative kernels on a partial time/feature tile, nonzero program
IDs and nonunit row stride. The source aliases the output. Arbitrary padding
cannot change valid prefixes; the neighboring inactive tagged cell survives. -/
namespace VeriTile.Tests.GatedCumulativeReal
open VeriTile.Triton VeriTile.Bench.TritonBenchG VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealCumProgram RealCumExecution
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
open scoped BigOperators

private abbrev p : Layout := ⟨18,3,1,6,3,4,2⟩
private def u : BlockState where
  mem := fun _ off => if off = 33 then MemCell.of .nat 77 else MemCell.real (off : ℝ)
  regs := fun _ _ _ => none
  pids := fun _ => 1
  undef := fun _ _ => 43
private def A : FlatAlloc where
  flat := "flat"
  regions := ["x"]
  base := fun _ => 100
  extent := fun r => if r = "x" then 36 else 0

private theorem disjoint : A.Disjoint := by
  intro r hr s hs hne
  simp [A] at hr hs
  exact (hne (hr.trans hs.symm)).elim
private theorem covered : ∀ r, r ∉ A.regions → A.extent r = 0 := by
  intro r h
  simpa [A] using h
private theorem collision_free : NoCollision p (launch u) (chunk u) := by
  rintro ⟨i,j,⟨⟩⟩ ⟨k,e,⟨⟩⟩ hi hk he
  have hi' := i.isLt
  have hk' := k.isLt
  have hj := j.isLt
  have he' := e.isLt
  dsimp [offset,ChunkCumsumVector.RealStep.address,p,launch,chunk,u] at he
  have hij : i = k ∧ j = e := ⟨Fin.ext (by omega),Fin.ext (by omega)⟩
  rcases hij with ⟨rfl,rfl⟩
  rfl
private theorem bounds : RealCumSafety.Bounds p (launch u) (chunk u) "x" "x" A.extent := by
  constructor <;> rintro ⟨i,j,⟨⟩⟩ hi
  all_goals
    dsimp [tileActive,ChunkCumsumVector.RealStep.active,p,launch,chunk,u] at hi
    norm_num [offset,ChunkCumsumVector.RealStep.address,p,launch,chunk,u,A] 
    omega
private def first : TileIndex [p.BT,p.BS] := (0,0,PUnit.unit)
private def second : TileIndex [p.BT,p.BS] := (1,0,PUnit.unit)
private theorem first_value : expected p (launch u) (chunk u) (u.readMem "x") first = 32 := by
  norm_num [expected,first,p,launch,chunk,u,BlockState.readMem,
    ChunkCumsumVector.RealStep.address,Fin.sum_univ_succ,Finset.sum_filter]
private theorem second_value : expected p (launch u) (chunk u) (u.readMem "x") second = 67 := by
  norm_num [expected,second,p,launch,chunk,u,BlockState.readMem,
    ChunkCumsumVector.RealStep.address,Fin.sum_univ_succ,Finset.sum_filter]

private theorem inactive_neighbor : ¬ RealCumFlat.footprint A p (launch u) (chunk u) "x" (A.flat,133) := by
  rintro ⟨_,⟨i,j,⟨⟩⟩,hi,he⟩
  dsimp [tileActive,ChunkCumsumVector.RealStep.active,p,launch,chunk,u] at hi
  norm_num [FlatAlloc.addr,A,offset,ChunkCumsumVector.RealStep.address,p,launch,chunk,u] at he
  omega

theorem both_complete_flat (swapped : Bool) : ∃ v : BlockState,
    exec (A.flattenKernel (RealCumSource.source swapped .fp16 .bf16 "x" "x"
      p.headStride p.timeStride p.channelStride p.T p.S p.BT p.BS).eraseDType.toAlgKernel)
      (A.flattenState u) = some v ∧
    v.readMem A.flat 132 = 32 ∧ v.readMem A.flat 135 = 67 ∧
    v.mem A.flat 133 = MemCell.of .nat 77 := by
  obtain ⟨v,he,ho,hf⟩ := RealCumFlat.source_executes A disjoint covered p "x" "x"
    swapped .fp16 .bf16 u collision_free bounds
  refine ⟨v,he,(ho first (by decide)).trans first_value,(ho second (by decide)).trans second_value,?_⟩
  rw [← hf A.flat 133 inactive_neighbor]
  change (A.flattenState u).mem A.flat (A.addr "x" 33) = _
  rw [A.flattenState_mem_addr disjoint u (by simp [A]) (by decide)]
  rfl

theorem both_complete_region (swapped : Bool) : ∃ v : BlockState,
    exec (RealCumSource.source swapped .fp16 .bf16 "x" "x"
      p.headStride p.timeStride p.channelStride p.T p.S p.BT p.BS).eraseDType.toAlgKernel u = some v ∧
    v.readMem "x" 32 = 32 ∧ v.readMem "x" 35 = 67 ∧ v.mem "x" 33 = MemCell.of .nat 77 := by
  obtain ⟨v,he,ho,hf⟩ := RealCumExecution.source_executes p "x" "x" swapped .fp16 .bf16 u collision_free
  refine ⟨v,he,(ho first (by decide)).trans first_value,(ho second (by decide)).trans second_value,?_⟩
  rw [← hf "x" 33]
  · rfl
  · rintro ⟨_,i,hi,he⟩
    apply inactive_neighbor
    refine ⟨rfl,i,hi,?_⟩
    rw [← he]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.GatedCumulativeReal
