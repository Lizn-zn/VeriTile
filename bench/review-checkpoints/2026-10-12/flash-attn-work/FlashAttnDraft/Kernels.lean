import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! Source: flash_attn.py, _fwd_kernel.
The full source and candidate differ only in the natural-number head-base
product. Q and O storage are independently fp16, bf16 or fp32. K and V are
fp16 because both dot left operands are explicitly converted to fp16. L is
fp32, as allocated by the host. Source stride choices, unchecked block loads
and stores, the literal 1.44269504 and denom * 0 + alpha are retained.
No host-layout correction or replacement with ordinary softmax is made. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.TypedSource
open Lean VeriTile.Triton
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq
syntax "flashAttnSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(flashAttnSource%[$inputTy:tritonDType,$outputTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let baseOffset : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then
        `(tritonStmt| qkv_base_offset = $(stride_q_head) * off_bs_head)
      else `(tritonStmt| qkv_base_offset = off_bs_head * $(stride_q_head))
    `(fun (Q K V L O : RegionName) (sm_scale : ℝ)
      (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
        stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
        stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
        stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
        BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (IS_CAUSAL : Bool) =>
      ComputeKernel.mk [Q,K,V] [L,O] (triton {
  tl.region Q = $inputTy, K = tl.float16, V = tl.float16, L = tl.float32, O = $outputTy

  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)

  $baseOffset
  Q_block_ptr = tl.make_block_ptr(base=Q + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_q_seqlen), $(stride_q_dim)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(DIM)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + qkv_base_offset,
    shape=($(DIM), $(SEQLEN)),
    strides=($(stride_k_dim), $(stride_k_seqlen)),
    offsets=(0, 0),
    block_shape=($(DIM), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_k_seqlen), $(stride_v_dim)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(DIM)),
    order=(1, 0))
  off_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(BLOCK_N))
  max = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  denom = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  out_buffer = tl.zeros([$(BLOCK_M), $(DIM)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(tl.float16)
  lo = 0
  hi = ((start_m + $(1)) * $(BLOCK_M) if IS_CAUSAL else $(SEQLEN))
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    if IS_CAUSAL {
      qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
    }
    qk += tl.dot(q, k)

    max_new = tl.maximum(max, tl.max(qk, 1))
    alpha = tl.math.exp2(max - max_new)
    nume = tl.math.exp2(qk - max_new[:, None])
    out_scale = denom * 0 + alpha
    out_buffer *= out_scale[:, None]
    out_buffer += tl.dot((nume).to(tl.float16), v)
    denom = denom * alpha + tl.sum(nume, 1)
    max = max_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  out_buffer = out_buffer / denom[:, None]
  l_ptr = L + off_bs_head * $(SEQLEN) + off_m
  tl.store(l_ptr, (max + tl.math.log2(denom)).to(tl.float32))
  O_block_ptr = tl.make_block_ptr(base=O + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_o_seqlen), $(stride_o_dim)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(DIM)),
    order=(1, 0))
  tl.store(O_block_ptr, ((out_buffer).to(tl.float16)).to($outputTy:tritonDType))

}).surfaceBody)

abbrev kernel_fp16_fp16 (swapped : Bool) := if swapped then flashAttnSource%[tl.float16,tl.float16,indexCommuted] else flashAttnSource%[tl.float16,tl.float16,original]
abbrev kernel_fp16_bf16 (swapped : Bool) := if swapped then flashAttnSource%[tl.float16,tl.bfloat16,indexCommuted] else flashAttnSource%[tl.float16,tl.bfloat16,original]
abbrev kernel_fp16_fp32 (swapped : Bool) := if swapped then flashAttnSource%[tl.float16,tl.float32,indexCommuted] else flashAttnSource%[tl.float16,tl.float32,original]
abbrev kernel_bf16_fp16 (swapped : Bool) := if swapped then flashAttnSource%[tl.bfloat16,tl.float16,indexCommuted] else flashAttnSource%[tl.bfloat16,tl.float16,original]
abbrev kernel_bf16_bf16 (swapped : Bool) := if swapped then flashAttnSource%[tl.bfloat16,tl.bfloat16,indexCommuted] else flashAttnSource%[tl.bfloat16,tl.bfloat16,original]
abbrev kernel_bf16_fp32 (swapped : Bool) := if swapped then flashAttnSource%[tl.bfloat16,tl.float32,indexCommuted] else flashAttnSource%[tl.bfloat16,tl.float32,original]
abbrev kernel_fp32_fp16 (swapped : Bool) := if swapped then flashAttnSource%[tl.float32,tl.float16,indexCommuted] else flashAttnSource%[tl.float32,tl.float16,original]
abbrev kernel_fp32_bf16 (swapped : Bool) := if swapped then flashAttnSource%[tl.float32,tl.bfloat16,indexCommuted] else flashAttnSource%[tl.float32,tl.bfloat16,original]
abbrev kernel_fp32_fp32 (swapped : Bool) := if swapped then flashAttnSource%[tl.float32,tl.float32,indexCommuted] else flashAttnSource%[tl.float32,tl.float32,original]

def kernel (input output : Storage) (swapped : Bool) := match input,output with
  | .fp16,.fp16 => kernel_fp16_fp16 swapped
  | .fp16,.bf16 => kernel_fp16_bf16 swapped
  | .fp16,.fp32 => kernel_fp16_fp32 swapped
  | .bf16,.fp16 => kernel_bf16_fp16 swapped
  | .bf16,.bf16 => kernel_bf16_bf16 swapped
  | .bf16,.fp32 => kernel_bf16_fp32 swapped
  | .fp32,.fp16 => kernel_fp32_fp16 swapped
  | .fp32,.bf16 => kernel_fp32_bf16 swapped
  | .fp32,.fp32 => kernel_fp32_fp32 swapped
end
end VeriTile.Bench.TritonBenchG.FlashAttn.TypedSource
