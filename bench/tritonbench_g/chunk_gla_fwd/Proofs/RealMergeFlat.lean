import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealMergeSafety

/-! Both complete merge sources run in finite flat allocations and implement
the independent initial-array sum, preserving every other tagged cell. The
zero-oracle premise here belongs to the shared flattening adapter: the region
execution theorem itself permits arbitrary unspecified real values. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeFlat
open VeriTile.Triton RealMergeProgram
noncomputable section
set_option maxRecDepth 14000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

def footprint (alloc : FlatAlloc) (p : Layout) (l : Launch) (A2 : RegionName) : WriteFootprint :=
  fun cell => cell.1 = alloc.flat ∧ enabled p l ∧ ∃ i, active p l i ∧ cell.2 = alloc.addr A2 (outputAddress p l i)

theorem flatten_frame (alloc : FlatAlloc) (p : Layout) (l : Launch) (A2 : RegionName) (s t : BlockState)
    (hf : BlockState.WriteWithin (RealMergeProgram.footprint p l A2) s t) :
    BlockState.WriteWithin (footprint alloc p l A2) (alloc.flattenState s) (alloc.flattenState t) := by
  intro region off hn
  by_cases hr : region = alloc.flat
  · subst region
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases hd : alloc.decode off with
    | none => rfl
    | some cell =>
      rcases cell with ⟨r,k⟩
      obtain ⟨_,heq,_⟩ := alloc.decode_sound hd
      apply congrArg alloc.trCell
      apply hf
      rintro ⟨hr,hen,i,hi,haddr⟩
      apply hn
      refine ⟨rfl,hen,i,hi,?_⟩
      rw [← hr,← haddr,← heq]
  · simp only [FlatAlloc.flattenState,if_neg hr]

theorem source_executes (p : Layout) (A A2 : RegionName) (out : FloatDType) (swapped : Bool)
    (alloc : FlatAlloc) (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (s : BlockState) (hu : s.undef = fun _ _ => 0)
    (hinj : NoCollision p (launch s))
    (hb : RealMergeSafety.Bounds p (launch s) A A2 alloc.extent) :
    ∃ sf, exec (alloc.flattenKernel (source p A A2 out swapped).eraseDType.toAlgKernel) (alloc.flattenState s) = some sf ∧
      (∀ i, active p (launch s) i → sf.readMem alloc.flat (alloc.addr A2 (outputAddress p (launch s) i)) =
        expected p (launch s) (s.readMem A) i) ∧
      BlockState.WriteWithin (footprint alloc p (launch s) A2) (alloc.flattenState s) sf := by
  obtain ⟨t,he,ho,hf⟩ := RealMergeExecution.source_executes p A A2 out swapped s hinj
  have hflat := alloc.exec_flatten hd hcov _ s
    (RealMergeSafety.source_safe p A A2 out swapped s alloc.extent hb)
    (RealMergeSafety.source_flatten_ok p A A2 out swapped) hu
  rw [he,Option.map_some] at hflat
  refine ⟨alloc.flattenState t,hflat,?_,flatten_frame alloc p (launch s) A2 s t hf⟩
  intro i hi
  have hen : enabled p (launch s) := Nat.lt_of_le_of_lt (Nat.le_add_right _ _) hi.1
  have hoff := hb.output hen i hi
  have hr : A2 ∈ alloc.regions := by
    by_contra hn
    rw [hcov A2 hn] at hoff
    exact Nat.not_lt_zero _ hoff
  rw [alloc.flattenState_readMem hd t hr hoff]
  exact ho i hi

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeFlat
