import bench.tritonbench_g.destindex_copy_kv1.Kernels
import bench.tritonbench_g.destindex_copy.Proofs.IndexRewrite
namespace VeriTile.Bench.TritonBenchG.DestindexCopyKv1.FPProof
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def program (input output : Storage) (swapped : Bool)
    (K : RegionName) (Dest_loc : Region .int) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat) : Program where
  kernel := TypedSource.kernel input output swapped K Dest_loc Out stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num head_dim BLOCK_DMODEL BLOCK_HEAD
  domain := []

theorem equivalent (input output : Storage)
    (K : RegionName) (Dest_loc : Region .int) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat) :
    program input output Bool.false K Dest_loc Out stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num head_dim BLOCK_DMODEL BLOCK_HEAD ≡[assumptions]
      program input output Bool.true K Dest_loc Out stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num head_dim BLOCK_DMODEL BLOCK_HEAD := by
  cases input <;> cases output <;> equiv_decompose <;> fp_prove

#print_fp_assumptions equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DestindexCopyKv1.FPProof
