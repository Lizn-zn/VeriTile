import bench.tritonbench_g.context_attn_fwd.Kernels
import bench.tritonbench_g.context_attn_fwd.Proofs.FP
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPEquivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

/-- Preserve the actual first-dot precision on each source storage branch. -/
def scoreOperand (storage : Storage) (BM BN : Nat) : Op .real [BM,BN] :=
  match storage with
  | .fp16 | .bf16 => .castFloat .fp32 .real (.ref .fp32 [BM,BN] "qk")
  | .fp32 => .ref .real [BM,BN] "qk"

/-- Every reached loop iteration checks the actual finite qk/scale operands
in all lanes of the eager where arm, including lanes whose mask is false.
The dot, exp2, reductions, accumulator-dot, casts, masks and stores are common
context; no exponential or dot/add identity is assumed. -/
def sites (storage : Storage) (code : List ComputeStmt) (BM BN : Nat) (scale : ℝ) : List Site :=
  match Source.intRange? code "start_n" with
  | none => []
  | some loop => [.intRange loop.before "start_n" loop.start loop.stop loop.stride loop.body
      (.mulScalar (Source.prefixBefore loop.body "qk" 1) BM [BN]
        (scoreOperand storage BM BN) (.const scale))]

section
variable (storage : Storage)
    (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd
      stride_obs stride_oh stride_od
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)

def original : Program where
  kernel := VeriTile.Bench.TritonBenchG.ContextAttnFwd.original storage Q K V sm_scale Out B_Start_Loc B_Seqlen b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd
    stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N
  domain := sites storage (VeriTile.Bench.TritonBenchG.ContextAttnFwd.original storage Q K V sm_scale Out B_Start_Loc B_Seqlen b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd
    stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N).surfaceBody
    BLOCK_M BLOCK_N sm_scale

def scaleFirst : Program :=
  { original storage Q K V sm_scale Out B_Start_Loc B_Seqlen b_prompt_cache_len
      stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N with
    kernel := VeriTile.Bench.TritonBenchG.ContextAttnFwd.scaleFirst storage Q K V sm_scale Out B_Start_Loc B_Seqlen b_prompt_cache_len
      stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N }

specification scaleFirst_equiv (R : FP.ScalarArithmetic.Rules) :
    original storage Q K V sm_scale Out B_Start_Loc B_Seqlen b_prompt_cache_len
      stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N ≡[R]
    scaleFirst storage Q K V sm_scale Out B_Start_Loc B_Seqlen b_prompt_cache_len
      stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N := by
  exact FPProof.equivalent storage Q K V sm_scale Out B_Start_Loc B_Seqlen b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd
    stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N R
end
#print_fp_assumptions scaleFirst_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPEquivalence
