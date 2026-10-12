import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.FP
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.BroadcastRewrite
import VeriTile.Triton.Float.RewriteTactics

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 6000000
set_option maxRecDepth 14000

section
variable (biasStorage : BiasStorage) (dtype out_dtype : FloatDType)
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr : RegionName)
    (batch seqlen nheads chunk_size : Nat)
    (dt_min : ℝ) (dt_max : UpperBound)
    (stride_dt_batch stride_dt_seqlen stride_dt_head
      stride_A_head
      stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize : Nat)
    (DT_SOFTPLUS HAS_DT_BIAS : Bool)
    (BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)

/-- The whole prefix runs through both loads, bias/softplus, both clamps,
remasking and the first output store. The domain checks the actual reached
fp32 dt tile and A row for the only changed multiplication. Bias/A/input
storage stay independent, both outputs are fp32, and scan order is unchanged. -/
def original : Program where
  kernel := VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.original biasStorage dtype out_dtype dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK
  domain := [.mulBroadcast
    (Source.prefixBefore (VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.original biasStorage dtype out_dtype dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK).surfaceBody "dA")
    [BLOCK_SIZE_H,BLOCK_SIZE_CHUNK] [BLOCK_SIZE_H,1]
    (.ref .real [BLOCK_SIZE_H,BLOCK_SIZE_CHUNK] "dt")
    (.expandDim ⟨1,by simp⟩ (.ref .real [BLOCK_SIZE_H] "A"))]

def scaleFirst : Program :=
  { original biasStorage dtype out_dtype dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK with
    kernel := VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.scaleFirst biasStorage dtype out_dtype dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK }

specification scaleFirst_equiv (R : FP.ScalarArithmetic.Rules) :
    original biasStorage dtype out_dtype dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK ≡[R]
    scaleFirst biasStorage dtype out_dtype dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK := by
  exact FPProof.equivalent biasStorage dtype out_dtype dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr
    batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch stride_dt_seqlen stride_dt_head
    stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head
    stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
    DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK R
end
#print_fp_assumptions scaleFirst_equiv
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.FPEquiv
