import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Source: context_attn_llama.py, _fwd_kernel.
Complete original and scale-first candidate with typed floating Q/K/V/Out,
signed metadata and gathered page indices. Both preserve the finite causal
sentinel, base-2 exponential, fused accumulator dot and source masks/casts.
The candidate only commutes qk and sm_scale inside the eager where arm. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama
open Lean VeriTile.Triton
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq
syntax "contextLlamaSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(contextLlamaSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scaleFirst do
      Macro.throwError "expected original or scaleFirst"
    let scaling : TSyntax `tritonStmt ←
      if version.getId == `scaleFirst then
        `(tritonStmt| qk = tl.where(mask, $((sm_scale : ℝ)) * qk, -1.0e8))
      else `(tritonStmt| qk = tl.where(mask, qk * $((sm_scale : ℝ)), -1.0e8))
    `(fun
    (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) => ComputeKernel.mk
      [Q,K,V,B_Start_Loc,B_Seqlen,Req_to_tokens,B_req_idx,b_prompt_cache_len] [Out] (triton {
  tl.region Q = $storage, K = $storage, V = $storage, Out = $storage,
    B_Start_Loc = tl.int32, B_Seqlen = tl.int32, b_prompt_cache_len = tl.int32,
    Req_to_tokens = tl.int32, B_req_idx = tl.int32

  start_m = tl.program_id(0)
  cur_bh = tl.program_id(1)
  cur_batch = cur_bh // $(H)
  cur_head = cur_bh % $(H)

  cur_kv_head = cur_head // $(kv_group_num)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = block_start_loc + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)
  q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum(block_start_loc + $(BLOCK_M) + prompt_cache_len,
    cur_batch_seq_len + prompt_cache_len)
  for start_n in range($(0), block_mask * block_end_loc, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    kv_loc = tl.load(Req_to_tokens + $(stride_req_to_tokens_b) * cur_batch_req_idx +
      $(stride_req_to_tokens_s) * (start_n + offs_n),
      mask=(start_n + offs_n) < block_end_loc, other=$(0))
    off_k = kv_loc[None, :] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
      offs_d[:, None] * $(stride_kd)
    k = tl.load(K + off_k,
      mask=(start_n + offs_n[None, :]) < block_end_loc,
      other=0.0)

    qk = tl.dot(q, k)
    mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
    $scaling
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)

    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    off_v = kv_loc[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
      offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=(start_n + offs_n[:, None]) < block_end_loc,
      other=0.0)

    p = (p).to($storage:tritonDType)
    acc = tl.dot(p, v, acc)
    m_i = m_ij
  }

  acc = acc / l_i[:, None]
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to($storage:tritonDType), mask=offs_m[:, None] < cur_batch_seq_len)
}).surfaceBody)

def original (storage : Storage) := match storage with
  | .fp16 => contextLlamaSource%[tl.float16,original]
  | .bf16 => contextLlamaSource%[tl.bfloat16,original]
  | .fp32 => contextLlamaSource%[tl.float32,original]
def scaleFirst (storage : Storage) := match storage with
  | .fp16 => contextLlamaSource%[tl.float16,scaleFirst]
  | .bf16 => contextLlamaSource%[tl.bfloat16,scaleFirst]
  | .fp32 => contextLlamaSource%[tl.float32,scaleFirst]
end VeriTile.Bench.TritonBenchG.ContextAttnLlama
