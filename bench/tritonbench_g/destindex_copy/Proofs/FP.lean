import bench.tritonbench_g.destindex_copy.Kernels
import bench.tritonbench_g.destindex_copy.Proofs.IndexRewrite
namespace VeriTile.Bench.TritonBenchG.DestindexCopy.FPProof
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (nopeType ropeType : Storage) (swapped : Bool)
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .int) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat) : Program where
  kernel := TypedSource.kernel nopeType ropeType swapped KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE
  domain := []

theorem equivalent (nopeType ropeType : Storage)
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .int) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat) :
    program nopeType ropeType Bool.false KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE ≡[assumptions]
      program nopeType ropeType Bool.true KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE := by
  cases nopeType <;> cases ropeType <;> equiv_decompose <;> fp_prove

#print_fp_assumptions equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DestindexCopy.FPProof
