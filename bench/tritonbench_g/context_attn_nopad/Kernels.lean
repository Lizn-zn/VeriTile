import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Source: context_attn_nopad.py, _fwd_kernel.
The complete original and scale-first candidate retain signed metadata,
typed storage/casts, causal negative-infinity masks, natural exp and the
normalized online accumulator. Only qk and sm_scale are commuted. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad
open Lean VeriTile.Triton
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq
syntax "contextNopadSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(contextNopadSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scaleFirst do
      Macro.throwError "expected original or scaleFirst"
    let scaling : TSyntax `tritonStmt ←
      if version.getId == `scaleFirst then
        `(tritonStmt| qk = $((sm_scale : ℝ)) * qk)
      else `(tritonStmt| qk *= $((sm_scale : ℝ)))
    `(fun (Q K V : RegionName) (sm_scale : ℝ)
      (B_Start_Loc B_Seqlen Out : RegionName)
      (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od : Nat)
      (BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) => ComputeKernel.mk
      [Q,K,V,B_Start_Loc,B_Seqlen] [Out] (triton {
  tl.region Q = $storage, K = $storage, V = $storage, Out = $storage,
    B_Start_Loc = tl.int32, B_Seqlen = tl.int32

  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)
  off_k = offs_n[None, :] * $(stride_kbs) + cur_head * $(stride_kh) +
    offs_d[:, None] * $(stride_kd)
  off_v = offs_n[:, None] * $(stride_vbs) + cur_head * $(stride_vh) +
    offs_d[None, :] * $(stride_vd)

  q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

  k_ptrs = K + off_k
  v_ptrs = V + off_v

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))

  for start_n in range($((0 : Int)), block_mask * (start_m + $(1)) * $(BLOCK_M), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * $(stride_kbs),
      mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
      other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    $scaling
    qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)

    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij

    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc = acc * acc_scale[:, None]
    v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * $(stride_vbs),
      mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
      other=0.0)

    p = (p).to($storage:tritonDType)
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to($storage:tritonDType), mask=offs_m[:, None] < cur_batch_seq_len)
}).surfaceBody)

def original (storage : Storage) := match storage with
  | .fp16 => contextNopadSource%[tl.float16,original]
  | .bf16 => contextNopadSource%[tl.bfloat16,original]
  | .fp32 => contextNopadSource%[tl.float32,original]
def scaleFirst (storage : Storage) := match storage with
  | .fp16 => contextNopadSource%[tl.float16,scaleFirst]
  | .bf16 => contextNopadSource%[tl.bfloat16,scaleFirst]
  | .fp32 => contextNopadSource%[tl.float32,scaleFirst]
end VeriTile.Bench.TritonBenchG.ContextAttnNopad
