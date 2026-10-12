import FlashAttnDraft.FP
/-! Complete typed-state/failure equivalence for both causal modes. Q and O
storage are independent fp16/bf16/fp32, K/V fp16 and L fp32. The kernel retains
all source strides, unchecked block accesses, casts, fp32 accumulators, exp2,
log2 and the denom*0+alpha expression. Only the natural head-base product is
commuted. The profiled structural engine uses fp32 default precision. There
are no input-domain restrictions or numerical FP atoms; successful nonempty
executions are validated separately from this equivalence statement. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.Equivalence
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 30000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original (input output : Storage) (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (causal : Bool) : Program where
  kernel := kernel input output Bool.false Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal
  domain := []

def indexCommuted (input output : Storage) (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (causal : Bool) : Program where
  kernel := kernel input output Bool.true Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal
  domain := []

specification source_equivalence (input output : Storage) (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (causal : Bool) :
    original input output Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal ≡[assumptions]
    indexCommuted input output Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal := by
  exact FPProof.equivalent input output Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.Equivalence
