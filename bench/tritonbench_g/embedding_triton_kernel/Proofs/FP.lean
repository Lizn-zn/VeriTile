import bench.tritonbench_g.embedding_triton_kernel.Kernels
import bench.tritonbench_g.destindex_copy.Proofs.IndexRewrite
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.FPProof
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (storage : Storage) (swapped : Bool) (weight input_ids out : RegionName)
    (first last : Int) (sw so n h D N NN : Nat) : Program where
  kernel := source storage swapped weight input_ids out first last sw so n h D N NN
  domain := []

theorem equivalent (storage : Storage) (weight input_ids out : RegionName)
    (first last : Int) (sw so n h D N NN : Nat) :
    program storage Bool.false weight input_ids out first last sw so n h D N NN ≡[assumptions]
      program storage Bool.true weight input_ids out first last sw so n h D N NN := by
  cases storage <;> equiv_decompose <;> fp_prove

#print_fp_assumptions equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.FPProof
