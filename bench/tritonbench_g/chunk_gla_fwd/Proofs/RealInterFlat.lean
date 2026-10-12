import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterSafety

/-! Complete inter-subblock execution in finite flat allocations. On the
active branch the output is the independent initial-array contraction; every
other tagged cell is preserved. Input regions need not be separate from output. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterFlat
open VeriTile.Triton RealInterMath RealInterProgram RealInterLoop
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

def footprint (alloc : FlatAlloc) (p : Layout) (l : Launch) (r : RegionName) : WriteFootprint :=
  fun cell => cell.1 = alloc.flat ∧ enabled p l ∧ ∃ i : TileIndex [p.BC,p.BC],
    active p l i.1.val i.2.1.val ∧ cell.2 = alloc.addr r (outputAddress p l i.1.val i.2.1.val)

theorem flatten_frame (alloc : FlatAlloc) (p : Layout) (l : Launch) (r : RegionName) (s t : BlockState)
    (hf : BlockState.WriteWithin (RealInterFinal.footprint p l r) s t) :
    BlockState.WriteWithin (footprint alloc p l r) (alloc.flattenState s) (alloc.flattenState t) := by
  intro region off hn
  by_cases hr : region = alloc.flat
  · subst region
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases hd : alloc.decode off with
    | none => rfl
    | some cell =>
      rcases cell with ⟨r',k⟩
      obtain ⟨_,heq,_⟩ := alloc.decode_sound hd
      apply congrArg alloc.trCell
      apply hf
      rintro ⟨hr,hen,i,hi,haddr⟩
      apply hn
      refine ⟨rfl,hen,i,hi,?_⟩
      rw [← hr,← haddr,← heq]
  · simp only [FlatAlloc.flattenState,if_neg hr]

theorem source_executes (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) (swapped : Bool)
    (alloc : FlatAlloc) (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (s : BlockState) (hu : s.undef = fun _ _ => 0) (hb : RealInterSafety.Bounds p (launch s) r alloc.extent) :
    ∃ sf, exec (alloc.flattenKernel (source p r qs ks gs out scale swapped).eraseDType.toAlgKernel) (alloc.flattenState s) = some sf ∧
      (enabled p (launch s) → ∀ i : TileIndex [p.BC,p.BC], active p (launch s) i.1.val i.2.1.val →
        sf.readMem alloc.flat (alloc.addr r.output (outputAddress p (launch s) i.1.val i.2.1.val)) =
          expected p (launch s) (inputs r s) scale i.1.val i.2.1.val) ∧
      BlockState.WriteWithin (footprint alloc p (launch s) r.output) (alloc.flattenState s) sf := by
  have hz : ∀ region off, s.undef region off = 0 := by rw [hu]; intros; rfl
  obtain ⟨t,he,ho,hf⟩ := RealInterExecution.source_executes p r qs ks gs out scale swapped s hz
  have hflat := alloc.exec_flatten hd hcov _ s
    (RealInterSafety.source_safe p r qs ks gs out scale swapped s hz alloc.extent hb)
    (RealInterSafety.source_flatten_ok p r qs ks gs out scale swapped) hu
  rw [he,Option.map_some] at hflat
  refine ⟨alloc.flattenState t,hflat,?_,flatten_frame alloc p (launch s) r.output s t hf⟩
  intro hen i hi
  have hoff := hb.output hen i hi
  have hr : r.output ∈ alloc.regions := by
    by_contra hn
    rw [hcov r.output hn] at hoff
    exact Nat.not_lt_zero _ hoff
  rw [alloc.flattenState_readMem hd t hr hoff]
  exact ho hen i hi

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterFlat
