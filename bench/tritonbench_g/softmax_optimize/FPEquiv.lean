import bench.tritonbench_g.softmax_optimize.Kernels
import bench.tritonbench_g.softmax_optimize.Proofs.FP

/-! Complete signed-source equivalence, including the negative tail start at N=0.
The candidate only commutes exact integer factors in each of the four loop bodies.
All floating operations, casts, masks, named pointer registers and stores are retained.
This theorem preserves the entire state and failure; it asserts no IEEE softmax
correctness at padded never-active lanes or invalid native launch boundaries. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Equivalence
open VeriTile Triton FP.ProfiledRewrite FullSource
open scoped VeriTile.Spec
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def program (k : ComputeKernel) : Program := ⟨k, []⟩

specification full_source_equiv (input : Storage) (output : Accumulation)
    (Y X : RegionName) (M N B : Nat) :
    program (kernel input output Bool.false Y X M N B) ≡[assumptions]
    program (kernel input output Bool.true Y X M N B) := by
  cases input <;> cases output <;> equiv_decompose <;> fp_prove

#print_fp_assumptions full_source_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Equivalence
