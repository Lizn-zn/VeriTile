import bench.tritonbench_g.chunk_gla_simple.Proofs.RealSafety
import bench.tritonbench_g.chunk_gla_fwd.Proofs.IO

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Flat
open VeriTile.Triton Math Real
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def footprint (alloc : FlatAlloc) (p : Layout) (l : Launch) (r : Regions) : WriteFootprint :=
  fun cell => cell.1 = alloc.flat ∧ ∃ i : TileIndex [p.BT,p.BV],
    active p l i.1.val i.2.1.val ∧ cell.2 = alloc.addr r.output (outputAddress p l i.1.val i.2.1.val)

theorem flatten_frame (alloc : FlatAlloc) (p : Layout) (l : Launch) (r : Regions) (s t : BlockState)
    (hf : BlockState.WriteWithin (Real.footprint p l r) s t) :
    BlockState.WriteWithin (footprint alloc p l r) (alloc.flattenState s) (alloc.flattenState t) := by
  intro region off hn
  by_cases hr : region = alloc.flat
  · subst region
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases hd : alloc.decode off with
    | none => rfl
    | some cell =>
      rcases cell with ⟨rg,k⟩
      obtain ⟨_,heq,_⟩ := alloc.decode_sound hd
      apply congrArg alloc.trCell
      apply hf
      rintro ⟨hr,i,hi,haddr⟩
      apply hn
      refine ⟨rfl,i,hi,?_⟩
      rw [← hr,← haddr,← heq]
  · simp only [FlatAlloc.flattenState,if_neg hr]

theorem source_executes (p : Layout) (r : Regions) (dot value gate : Storage) (out : FloatDType) (scale : ℝ)
    (swapped : Bool) (alloc : FlatAlloc) (hd : alloc.Disjoint)
    (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0) (s : BlockState)
    (hu : s.undef = fun _ _ => 0) (hinj : NoCollision p (launch s))
    (hb : Safe.Bounds p (launch s) r alloc.extent) :
    ∃ sf, exec (alloc.flattenKernel (program p r dot value gate out scale swapped).eraseDType.toAlgKernel)
        (alloc.flattenState s) = some sf ∧
      (∀ i : TileIndex [p.BT,p.BV], active p (launch s) i.1.val i.2.1.val →
        sf.readMem alloc.flat (alloc.addr r.output (outputAddress p (launch s) i.1.val i.2.1.val)) =
          expected p (launch s) (inputs r s) scale i.1.val i.2.1.val) ∧
      BlockState.WriteWithin (footprint alloc p (launch s) r) (alloc.flattenState s) sf := by
  obtain ⟨t,he,ho,hf⟩ := program_executes p r dot value gate out scale swapped s hu hinj
  have hflat := alloc.exec_flatten hd hcov _ s
    (Safe.source_safe p r dot value gate out scale swapped s hu alloc.extent hb)
    (Safe.source_flatten_ok p r dot value gate out scale swapped) hu
  rw [he,Option.map_some] at hflat
  refine ⟨alloc.flattenState t,hflat,?_,flatten_frame alloc p (launch s) r s t hf⟩
  intro i hi
  have hoff : outputAddress p (launch s) i.1.val i.2.1.val < alloc.extent r.output := by
    simpa only [outputAddress,row,column,Nat.mul_one] using hb.tail.output i.1 i.2.1 hi
  have hr : r.output ∈ alloc.regions := by
    by_contra hn
    rw [hcov r.output hn] at hoff
    exact Nat.not_lt_zero _ hoff
  rw [alloc.flattenState_readMem hd t hr hoff]
  exact ho i hi

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Flat

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.RealIO
open VeriTile.Triton Math Real ChunkGlaFwd.RealIO
open scoped ChunkGlaFwd.RealIO
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000

theorem projection (p : Layout) (r : Regions) (dot value gate : Storage) (out : FloatDType) (scale : ℝ)
    (swapped : Bool) :
    (program p r dot value gate out scale swapped).eraseDType.toAlgorithm? =
      .ok (program p r dot value gate out scale swapped).eraseDType.toAlgKernel := by
  cases swapped
  all_goals simp only [program,Bool.false_eq_true,↓reduceIte,kernel,candidate,original,scaleFirst,
    ComputeKernel.eraseDType,RealSource.projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def sourceIO (p : Layout) (r : Regions) (dot value gate : Storage) (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    IO Launch Inputs (TileIndex [p.BT,p.BV]) where
  kernel := (program p r dot value gate out scale swapped).eraseDType
  projection := projection p r dot value gate out scale swapped
  grid := launch
  read := inputs r
  valid := fun s => NoCollision p (launch s)
  bounds := fun s => Safe.Bounds p (launch s) r
  out := r.output
  write := fun l i => outputAddress p l i.1.val i.2.1.val
  mask := fun l i => active p l i.1.val i.2.1.val

theorem implements (p : Layout) (r : Regions) (dot value gate : Storage) (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    sourceIO p r dot value gate out scale swapped ⊨
      (fun l a i => expected p l a scale i.1.val i.2.1.val) := by
  intro alloc hd hcov s hu hi hb
  exact Flat.source_executes p r dot value gate out scale swapped alloc hd hcov s hu hi hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.RealIO
