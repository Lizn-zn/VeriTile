import bench.tritonbench_g.chunk_gla_fwd.Proofs.IO
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealMergeFlat

/-! Connect the merge kernel's explicit IO to complete flat execution. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeIO
open VeriTile.Triton RealMergeProgram RealIO
open scoped RealIO
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

theorem projection (p : Layout) (A A2 : RegionName) (out : FloatDType) (swapped : Bool) :
    (source p A A2 out swapped).eraseDType.toAlgorithm? =
      .ok (source p A A2 out swapped).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType,projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def sourceIO (p : Layout) (A A2 : RegionName) (out : FloatDType) (swapped : Bool) :
    IO Launch (Nat → ℝ) (TileIndex [p.BC,p.BC]) where
  kernel := (source p A A2 out swapped).eraseDType
  projection := projection p A A2 out swapped
  grid := launch
  read := fun s => s.readMem A
  valid := fun s => NoCollision p (launch s)
  bounds := fun s => RealMergeSafety.Bounds p (launch s) A A2
  out := A2
  write := outputAddress p
  mask := active p

theorem implements (p : Layout) (A A2 : RegionName) (out : FloatDType) (swapped : Bool) :
    sourceIO p A A2 out swapped ⊨ expected p := by
  intro alloc hd hcov s hu hi hb
  obtain ⟨sf,he,ho,hf⟩ := RealMergeFlat.source_executes p A A2 out swapped alloc hd hcov s hu hi hb
  refine ⟨sf,he,ho,?_⟩
  intro region off hn
  apply hf region off
  rintro ⟨hr,_,i,hi,haddr⟩
  exact hn ⟨hr,i,hi,haddr⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeIO
