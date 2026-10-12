import bench.tritonbench_g.context_attn_bloom.Kernels
import bench.tritonbench_g.context_attn_bloom.Proofs.FP
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.FPEquivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

/-- At each reached signed-loop iteration, the actual qk lanes and scale are
finite at the multiplication. Query/key dot, exp, normalization, precision
casts, gathers and stores remain common context. No exp identity is used. -/
def sites (code : List ComputeStmt) (BM BN : Nat) (scale : ℝ) : List Site :=
  match Source.intRange? code "start_n" with
  | none => []
  | some loop => [.intRange loop.before "start_n" loop.start loop.stop loop.stride loop.body
      (.mulScalarFP32 (Source.prefixBefore loop.body "qk" 2) BM [BN]
        (.ref .real [BM,BN] "qk") (.const scale))]

section
variable (storage : Storage) 
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : RegionName) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) 

def original : Program where
  kernel := VeriTile.Bench.TritonBenchG.ContextAttnBloom.original storage Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
    stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
    kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N
  domain := sites (VeriTile.Bench.TritonBenchG.ContextAttnBloom.original storage Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
    stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
    kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N).surfaceBody
    BLOCK_M BLOCK_N sm_scale

def scaleFirst : Program :=
  { original storage Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
    stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
    kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N with
    kernel := VeriTile.Bench.TritonBenchG.ContextAttnBloom.scaleFirst storage Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len
      stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N }

specification scaleFirst_equiv (R : FP.ScalarArithmetic.Rules) :
    original storage Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
    stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
    kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N ≡[R]
    scaleFirst storage Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
    stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
    kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N := by
  exact FPProof.equivalent storage Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
    stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
    kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N R
end
#print_fp_assumptions scaleFirst_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.FPEquivalence
