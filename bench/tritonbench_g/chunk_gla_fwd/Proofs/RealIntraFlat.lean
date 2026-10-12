import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealIntraSafety

/-! Complete intra-subblock execution in finite flat allocations. On the
active branch the output is the independent initial-array contraction; every
other tagged cell is preserved. The shared flat adapter uses a zero-oracle premise; the region execution
proof itself allows arbitrary padding. Key/gate remain separate from output. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraFlat
open VeriTile.Triton RealIntraMath RealIntraProgram RealIntraLoop
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

def footprint (alloc : FlatAlloc) (p : Layout) (l : Launch) (r : RegionName) : WriteFootprint :=
  fun cell => cell.1 = alloc.flat ∧ enabled p l ∧ ∃ j, j < columns p l ∧ ∃ i : TileIndex [p.BC],
    rowActive p l i.1.val ∧ cell.2 = alloc.addr r (outputAddress p l i.1.val j)

theorem flatten_frame (alloc : FlatAlloc) (p : Layout) (l : Launch) (r : RegionName) (s t : BlockState)
    (hf : BlockState.WriteWithin (RealIntraExecution.footprint p l r) s t) :
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
      rintro ⟨hen,hr,j,hj,i,hi,haddr⟩
      apply hn
      refine ⟨rfl,hen,j,hj,i,hi,?_⟩
      rw [← hr,← haddr,← heq]
  · simp only [FlatAlloc.flattenState,if_neg hr]

theorem source_executes (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ) (swapped : Bool)
    (alloc : FlatAlloc) (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (s : BlockState)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) (hcols : columns p (launch s) ≤ p.BT) (hK : p.K ≤ p.BK)
    (hu : s.undef = fun _ _ => 0) (hb : RealIntraSafety.Bounds p (launch s) r alloc.extent) :
    ∃ sf, exec (alloc.flattenKernel (source p r qs ks gs scale swapped).eraseDType.toAlgKernel) (alloc.flattenState s) = some sf ∧
      (enabled p (launch s) → ∀ j, j < columns p (launch s) → ∀ i : Fin p.BC, rowActive p (launch s) i.val →
        sf.readMem alloc.flat (alloc.addr r.output (outputAddress p (launch s) i.val j)) =
          expected p (launch s) (RealIntraSetup.inputs r s) scale i.val j) ∧
      BlockState.WriteWithin (footprint alloc p (launch s) r.output) (alloc.flattenState s) sf := by
  obtain ⟨t,he,ho,hf⟩ := RealIntraExecution.source_executes p r qs ks gs scale swapped s hkey hgate hcols hK
  have hflat := alloc.exec_flatten hd hcov _ s
    (RealIntraSafety.source_safe p r qs ks gs scale swapped s hkey hgate hcols alloc.extent hb)
    (RealIntraSafety.source_flatten_ok p r qs ks gs scale swapped) hu
  rw [he,Option.map_some] at hflat
  refine ⟨alloc.flattenState t,hflat,?_,flatten_frame alloc p (launch s) r.output s t hf⟩
  intro hen j hj i hi
  have hoff := (hb.column hen j hj).output i hi
  have hr : r.output ∈ alloc.regions := by
    by_contra hn
    rw [hcov r.output hn] at hoff
    exact Nat.not_lt_zero _ hoff
  rw [alloc.flattenState_readMem hd t hr hoff]
  exact ho hen j hj i hi

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraFlat
