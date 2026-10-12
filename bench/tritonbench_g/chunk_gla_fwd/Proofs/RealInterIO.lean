import bench.tritonbench_g.chunk_gla_fwd.Proofs.IO
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterFlat

/-! Explicit inter-subblock IO, connected to complete finite-memory execution. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterIO
open VeriTile.Triton RealInterMath RealInterProgram RealIO
open scoped RealIO
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

theorem projection (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    (source p r qs ks gs out scale swapped).eraseDType.toAlgorithm? =
      .ok (source p r qs ks gs out scale swapped).eraseDType.toAlgKernel := by
  simp only [source,ComputeKernel.eraseDType,RealInterSource.projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def sourceIO (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) (swapped : Bool) : IO Launch Inputs (TileIndex [p.BC,p.BC]) where
  kernel := (source p r qs ks gs out scale swapped).eraseDType
  projection := projection p r qs ks gs out scale swapped
  grid := launch
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.gate⟩
  valid := fun _ => True
  bounds := fun s => RealInterSafety.Bounds p (launch s) r
  out := r.output
  write := fun l i => outputAddress p l i.1.val i.2.1.val
  mask := fun l i => enabled p l ∧ active p l i.1.val i.2.1.val

def formula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : TileIndex [p.BC,p.BC]) : ℝ :=
  expected p l a scale i.1.val i.2.1.val

theorem implements (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    sourceIO p r qs ks gs out scale swapped ⊨ formula p scale := by
  intro alloc hd hcov s hu _ hb
  obtain ⟨sf,he,ho,hf⟩ := RealInterFlat.source_executes p r qs ks gs out scale swapped alloc hd hcov s hu hb
  refine ⟨sf,he,?_,?_⟩
  · intro i hi
    exact ho hi.1 i hi.2
  · intro region off hn
    apply hf region off
    rintro ⟨hr,hen,i,hi,haddr⟩
    exact hn ⟨hr,i,⟨hen,hi⟩,haddr⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterIO
