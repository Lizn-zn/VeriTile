import bench.tritonbench_g.kldiv_compute.Proofs.Candidate

namespace VeriTile.Bench.TritonBenchG.KldivCompute
open VeriTile.Triton VeriTile.Examples TypedSource
open VeriTile.Triton.TiledLoss
open scoped VeriTile.Triton.MaskedKernelIO₂

def typedIO (x y output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) : MaskedKernelIO₂ :=
  { kldivIO X Y O N B with
    kernel := (kernel x y output v X Y O N B).eraseDType
    projection := typed_supported x y output v X Y O N B }

/-- Full flat execution and raw-cell frame for the independent pointwise real KL formula. -/
theorem typed_real_correct_bridge (x y output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) :
    VeriTile.Spec.Real (typedIO x y output v X Y O N B ⊨
      fun xs ys i => klDivSpec (xs i) (ys i)) := by
  intro A hd hr hc pid hx hy ho hscr xs ys s hp hu hxs hys
  obtain ⟨t, he, hv, hf⟩ := kldivergence_kernel_correctness_bridge X Y O N B
    A hd hr hc pid hx hy ho hscr xs ys s hp hu hxs hys
  refine ⟨t, ?_, hv, hf⟩
  exact (typed_flat_exec A x y output v X Y O N B (A.flattenState s)).trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivCompute
