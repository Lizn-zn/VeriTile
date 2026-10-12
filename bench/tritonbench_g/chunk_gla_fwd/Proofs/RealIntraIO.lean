import bench.tritonbench_g.chunk_gla_fwd.Proofs.IO
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealIntraFlat

/-! Explicit streamed-column IO, connected to the full source and tagged frame. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraIO
open VeriTile.Triton RealIntraMath RealIntraProgram RealIO
open scoped RealIO
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

theorem projection (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ) (swapped : Bool) :
    (source p r qs ks gs scale swapped).eraseDType.toAlgorithm? = .ok (source p r qs ks gs scale swapped).eraseDType.toAlgKernel := by
  simp only [source,ComputeKernel.eraseDType,RealIntraSource.projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def sourceIO (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ) (swapped : Bool) : IO Launch Inputs (Nat × Fin p.BC) where
  kernel := (source p r qs ks gs scale swapped).eraseDType
  projection := projection p r qs ks gs scale swapped
  grid := launch
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.gate⟩
  valid := fun s => columns p (launch s) ≤ p.BT
  bounds := fun s => RealIntraSafety.Bounds p (launch s) r
  out := r.output
  write := fun l i => outputAddress p l i.2.val i.1
  mask := fun l i => enabled p l ∧ i.1 < columns p l ∧ rowActive p l i.2.val

def formula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : Nat × Fin p.BC) : ℝ :=
  expected p l a scale i.2.val i.1

theorem implements (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ) (swapped : Bool) (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) (hK : p.K ≤ p.BK) :
    sourceIO p r qs ks gs scale swapped ⊨ formula p scale := by
  intro alloc hd hcov s hu hi hb
  obtain ⟨sf,he,ho,hf⟩ := RealIntraFlat.source_executes p r qs ks gs scale swapped alloc hd hcov s hkey hgate hi hK hu hb
  refine ⟨sf,he,?_,?_⟩
  · intro i ha
    exact ho ha.1 i.1 ha.2.1 i.2 ha.2.2
  · intro region off hn
    apply hf region off
    rintro ⟨hr,hen,j,hj,i,hi,haddr⟩
    exact hn ⟨hr,(j,i.1),⟨hen,hj,hi⟩,haddr⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraIO
