import bench.tritonbench_g.kldiv_compute.Kernels
import bench.tritonbench_g.sin_kernel.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.KldivCompute.FPProof
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

/-- The full suffix is identical: both loads, division, logarithm, multiplication and store. -/
theorem decomposition (x y output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) :
    (kernel x y output v X Y O N B).surfaceBody =
      SinKernel.FPProof.sourcePrefix v B ++
        (kernel x y output Bool.false X Y O N B).surfaceBody.drop 2 := by
  cases x <;> cases y <;> cases output <;> cases v <;> rfl

/-- Exact index arithmetic transports every complete execution, including failure and aliases. -/
theorem source_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (x y output : Storage) (X Y O : RegionName) (N B : Nat) (s : State α) :
    run (engine M) (kernel x y output Bool.false X Y O N B).surfaceBody s =
      run (engine M) (kernel x y output Bool.true X Y O N B).surfaceBody s := by
  rw [decomposition x y output Bool.false, decomposition x y output Bool.true]
  exact SinKernel.FPProof.prefix_run M s B _

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivCompute.FPProof
