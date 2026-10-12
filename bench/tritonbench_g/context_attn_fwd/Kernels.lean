import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Source: context_attn_fwd.py, _fwd_kernel_int8kv.
Complete typed floating-storage source and scale-first candidate. Despite the
source name, the supplied Python test uses fp16 Q/K/V rather than an int8
conversion path. Metadata remains signed int32. The finite -1e8 sentinel,
base-2 exponential, signed range, grouped KV heads, all masks and casts are
preserved. The candidate only commutes qk and sm_scale inside the where arm. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd
open Lean VeriTile.Triton
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq
syntax "contextFwdSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(contextFwdSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scaleFirst do
      Macro.throwError "expected original or scaleFirst"
    let scaling : TSyntax `tritonStmt ←
      if version.getId == `scaleFirst then
        `(tritonStmt| qk = tl.where(mask, $((sm_scale : ℝ)) * qk, -1.0e8))
      else `(tritonStmt| qk = tl.where(mask, qk * $((sm_scale : ℝ)), -1.0e8))
    `(fun
    (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd
      stride_obs stride_oh stride_od
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) => ComputeKernel.mk
      [Q,K,V,B_Start_Loc,B_Seqlen,b_prompt_cache_len] [Out] (triton {
  tl.region Q = $storage, K = $storage, V = $storage, Out = $storage,
    B_Start_Loc = tl.int32, B_Seqlen = tl.int32, b_prompt_cache_len = tl.int32

  start_m = tl.program_id(0)
  cur_bh = tl.program_id(1)
  cur_batch = cur_bh // $(H)
  cur_head = cur_bh % $(H)

  cur_kv_head = cur_head // $(kv_group_num)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len

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
    off_k = cur_batch * $(stride_kb) + (start_n + offs_n[None, :]) * $(stride_ks) +
      cur_kv_head * $(stride_kh) + offs_d[:, None] * $(stride_kd)
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
    off_v = cur_batch * $(stride_vb) + (start_n + offs_n[:, None]) * $(stride_vs) +
      cur_kv_head * $(stride_vh) + offs_d[None, :] * $(stride_vd)
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
  | .fp16 => contextFwdSource%[tl.float16,original]
  | .bf16 => contextFwdSource%[tl.bfloat16,original]
  | .fp32 => contextFwdSource%[tl.float32,original]
def scaleFirst (storage : Storage) := match storage with
  | .fp16 => contextFwdSource%[tl.float16,scaleFirst]
  | .bf16 => contextFwdSource%[tl.bfloat16,scaleFirst]
  | .fp32 => contextFwdSource%[tl.float32,scaleFirst]
end VeriTile.Bench.TritonBenchG.ContextAttnFwd
