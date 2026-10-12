import VeriTile.Triton.DSL

/-! Complete typed port of `attention_llama.py`'s `_fwd_kernel`.
Typed source for both K/V storage specializations.
IS_CAUSAL and nonnegative start_position remain parameters. Both causal
conditionals, source key-mask indexing, explicit zero padding, probability cast,
fp32 accumulation and masked fp16 store are retained. Positional other=0 in
Python is written as a keyword; Q.dtype.element_ty resolves to fp16. The store's
implicit conversion to Out's fp16 storage is written as an explicit cast,
because the frontend does not infer store dtype through a pointer register.
Inputs and output are declared explicitly below: automatic region scanning
does not follow these pointer aliases through to the load/store sites.

The first pair uses fp16 K/V. The FP8 pair below specializes USE_FP8=True,
loads signed int8 bytes and retains both reinterpretation and conversion.
The candidate swaps only sum(p,1)+l_prev into l_prev+sum(p,1). No exp, dot,
reduction-order or reassociation rewrite occurs.

Translation-surface blocker: USE_FP8 is specialized into two complete sources;
IS_CAUSAL remains a Bool parameter. Positional zero-fill arguments are spelled
other=, Q.dtype.element_ty is resolved to fp16, and the implicit output cast is
explicit. The textual source scan cannot compare this dtype specialization;
both byte reinterpretations, all branches and all memory operations are present. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.TypedSource
open VeriTile.Triton

def original
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel := .mk [Q, K, V] [Out] (triton {
  tl.region Q = tl.float16, K = tl.float16, V = tl.float16, Out = tl.float16
  start_m = tl.program_id(0)

  head_idx = tl.program_id(1)
  batch_id = head_idx // $(N_HEAD)
  off_hz = head_idx % $(N_HEAD)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_q = batch_id * $(stride_qz) + off_hz * $(stride_qh) + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  off_k = batch_id * $(stride_kz) + off_hz * $(stride_kh) + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kk)
  off_v = batch_id * $(stride_vz) + off_hz * $(stride_vh) + offs_n[:, None] * $(stride_vk) + offs_d[None, :] * $(stride_vn)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(q_ptrs, offs_m[:, None] < $(H), other=0.0)
  block_n_end = $(N_CTX)
  if IS_CAUSAL {
    block_n_end = (start_m + $(1)) * $(BLOCK_N) + $(start_position)
  }
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk = tl.where(offs_n[None, :] < $(N_CTX), qk, float("-inf"))
    qk *= $((sm_scale : ℝ))
    if IS_CAUSAL {
      qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + $(start_position)), qk, float("-inf"))
    }
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_ptrs += $(BLOCK_N) * $(stride_kn)
    v_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))

  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_o = batch_id * $(stride_oz) + off_hz * $(stride_oh) + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_on)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to(tl.float16), offs_m[:, None] < $(H))
}).surfaceBody


def carryFirst
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel := .mk [Q, K, V] [Out] (triton {
  tl.region Q = tl.float16, K = tl.float16, V = tl.float16, Out = tl.float16
  start_m = tl.program_id(0)

  head_idx = tl.program_id(1)
  batch_id = head_idx // $(N_HEAD)
  off_hz = head_idx % $(N_HEAD)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_q = batch_id * $(stride_qz) + off_hz * $(stride_qh) + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  off_k = batch_id * $(stride_kz) + off_hz * $(stride_kh) + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kk)
  off_v = batch_id * $(stride_vz) + off_hz * $(stride_vh) + offs_n[:, None] * $(stride_vk) + offs_d[None, :] * $(stride_vn)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(q_ptrs, offs_m[:, None] < $(H), other=0.0)
  block_n_end = $(N_CTX)
  if IS_CAUSAL {
    block_n_end = (start_m + $(1)) * $(BLOCK_N) + $(start_position)
  }
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk = tl.where(offs_n[None, :] < $(N_CTX), qk, float("-inf"))
    qk *= $((sm_scale : ℝ))
    if IS_CAUSAL {
      qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + $(start_position)), qk, float("-inf"))
    }
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = l_prev + tl.sum(p, 1)
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_ptrs += $(BLOCK_N) * $(stride_kn)
    v_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))

  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_o = batch_id * $(stride_oz) + off_hz * $(stride_oh) + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_on)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to(tl.float16), offs_m[:, None] < $(H))
}).surfaceBody


/-- USE_FP8=True specialization. K and V store signed int8 payloads; each
load is followed by E5M2 bit reinterpretation and a separate fp16 conversion.
Both causal conditionals and every other source statement are retained. -/
def originalFP8
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel := .mk [Q, K, V] [Out] (triton {
  tl.region Q = tl.float16, K = tl.int8, V = tl.int8, Out = tl.float16
  start_m = tl.program_id(0)

  head_idx = tl.program_id(1)
  batch_id = head_idx // $(N_HEAD)
  off_hz = head_idx % $(N_HEAD)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_q = batch_id * $(stride_qz) + off_hz * $(stride_qh) + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  off_k = batch_id * $(stride_kz) + off_hz * $(stride_kh) + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kk)
  off_v = batch_id * $(stride_vz) + off_hz * $(stride_vh) + offs_n[:, None] * $(stride_vk) + offs_d[None, :] * $(stride_vn)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(q_ptrs, offs_m[:, None] < $(H), other=0.0)
  block_n_end = $(N_CTX)
  if IS_CAUSAL {
    block_n_end = (start_m + $(1)) * $(BLOCK_N) + $(start_position)
  }
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    k = k.to(tl.float8e5, bitcast=True)
    k = k.to(tl.float16)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk = tl.where(offs_n[None, :] < $(N_CTX), qk, float("-inf"))
    qk *= $((sm_scale : ℝ))
    if IS_CAUSAL {
      qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + $(start_position)), qk, float("-inf"))
    }
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    v = v.to(tl.float8e5, bitcast=True)
    v = v.to(tl.float16)
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_ptrs += $(BLOCK_N) * $(stride_kn)
    v_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))

  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_o = batch_id * $(stride_oz) + off_hz * $(stride_oh) + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_on)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to(tl.float16), offs_m[:, None] < $(H))
}).surfaceBody


def carryFirstFP8
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel := .mk [Q, K, V] [Out] (triton {
  tl.region Q = tl.float16, K = tl.int8, V = tl.int8, Out = tl.float16
  start_m = tl.program_id(0)

  head_idx = tl.program_id(1)
  batch_id = head_idx // $(N_HEAD)
  off_hz = head_idx % $(N_HEAD)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_q = batch_id * $(stride_qz) + off_hz * $(stride_qh) + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  off_k = batch_id * $(stride_kz) + off_hz * $(stride_kh) + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kk)
  off_v = batch_id * $(stride_vz) + off_hz * $(stride_vh) + offs_n[:, None] * $(stride_vk) + offs_d[None, :] * $(stride_vn)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(q_ptrs, offs_m[:, None] < $(H), other=0.0)
  block_n_end = $(N_CTX)
  if IS_CAUSAL {
    block_n_end = (start_m + $(1)) * $(BLOCK_N) + $(start_position)
  }
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    k = k.to(tl.float8e5, bitcast=True)
    k = k.to(tl.float16)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk = tl.where(offs_n[None, :] < $(N_CTX), qk, float("-inf"))
    qk *= $((sm_scale : ℝ))
    if IS_CAUSAL {
      qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + $(start_position)), qk, float("-inf"))
    }
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = l_prev + tl.sum(p, 1)
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    v = v.to(tl.float8e5, bitcast=True)
    v = v.to(tl.float16)
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_ptrs += $(BLOCK_N) * $(stride_kn)
    v_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))

  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_o = batch_id * $(stride_oz) + off_hz * $(stride_oh) + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_on)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to(tl.float16), offs_m[:, None] < $(H))
}).surfaceBody


end VeriTile.Bench.TritonBenchG.AttentionLlama.TypedSource
