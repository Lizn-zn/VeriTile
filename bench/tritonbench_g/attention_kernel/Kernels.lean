import VeriTile.Triton.DSL

/-! Typed transcription of `attention_kernel.py`'s `_fwd_kernel_aligned`.
The fp16 and bf16 specializations match the two dtypes accepted by the host.
Both dot products use their source default fp32 output; Q scaling, probability
casts, storage-precision bias arithmetic, exp2, and unchecked block-pointer
loads/stores remain visible. V uses the source's K-head offset, and Out uses its
Q-head offset. No contiguity or fixed benchmark shape is built into this source.
Correct.lean and FPEquiv.lean use these same implementations.

Translation-surface blocker: both host storage dtypes are specialized explicitly;
region types, scalar conversions and source-dependent casts are stated in Lean.
The source regression checks these precision boundaries. README.md documents
the complete source mapping and the two bias windows.

Each reciprocal candidate changes only the final fp32 row normalization.
The bias scalar's implicit storage-dtype conversion is written explicitly:
https://triton-lang.org/main/python-api/triton-semantics.html#type-promotion
The mathematical projection erases this conversion; the FP source retains it.
-/
namespace VeriTile.Bench.TritonBenchG.AttentionKernel.TypedSource
open VeriTile.Triton

inductive Storage where
  | fp16
  | bf16
  deriving DecidableEq

namespace FP16

def original
    (q k v bias out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk
      stride_kh stride_kn stride_kk
      _stride_vh stride_vk stride_vn
      _stride_oh stride_om stride_on
      stride_b0h stride_b0m
      _Z _H N_CTX P_SEQ BIAS_LAST_SIZE _B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    : ComputeKernel :=
  let Q : Region .fp16 := q.cast
  let K : Region .fp16 := k.cast
  let V : Region .fp16 := v.cast
  let B0 : Region .fp16 := bias.cast
  let Out : Region .fp16 := out.cast
  triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  q_offset = off_hz * $(stride_qh)
  kv_offset = off_hz * $(stride_kh)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + kv_offset,
    shape=($((BLOCK_DMODEL : Nat)), $((N_CTX + P_SEQ : Nat))),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + kv_offset,
    shape=($((N_CTX + P_SEQ : Nat)), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(tl.float16)
  lo = 0
  hi = $((N_CTX + P_SEQ : Nat))

  b_ptr_offsets_m = tl.arange(0, $(BLOCK_M))
  b_offset = off_hz * $(stride_b0h)
  b_ptr_offsets_n_1 = (tl.arange(0, $(BLOCK_N)) % $(BIAS_LAST_SIZE)) +
    $(BIAS_LAST_SIZE)
  b1 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
    $(stride_b0m))[:, None] + b_ptr_offsets_n_1[None, :])
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float16)
    qk += tl.dot(q, k)

    b0 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
      $(stride_b0m))[:, None] + start_n // $(BLOCK_N))
    -- A floating scalar does not widen the bias tensor in Triton.
    -- Resolve its implicit conversion to the storage dtype explicitly.
    qk += ((b0 + b1) * (1.44269504).to(tl.float16))

    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc *= alpha[:, None]
    acc += tl.dot((p).to(tl.float16), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  acc = acc / l_i[:, None]
  O_block_ptr = tl.make_block_ptr(base=Out + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  tl.store(O_block_ptr, (acc).to(tl.float16))
}


def reciprocal
    (q k v bias out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk
      stride_kh stride_kn stride_kk
      _stride_vh stride_vk stride_vn
      _stride_oh stride_om stride_on
      stride_b0h stride_b0m
      _Z _H N_CTX P_SEQ BIAS_LAST_SIZE _B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    : ComputeKernel :=
  let Q : Region .fp16 := q.cast
  let K : Region .fp16 := k.cast
  let V : Region .fp16 := v.cast
  let B0 : Region .fp16 := bias.cast
  let Out : Region .fp16 := out.cast
  triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  q_offset = off_hz * $(stride_qh)
  kv_offset = off_hz * $(stride_kh)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + kv_offset,
    shape=($((BLOCK_DMODEL : Nat)), $((N_CTX + P_SEQ : Nat))),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + kv_offset,
    shape=($((N_CTX + P_SEQ : Nat)), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(tl.float16)
  lo = 0
  hi = $((N_CTX + P_SEQ : Nat))

  b_ptr_offsets_m = tl.arange(0, $(BLOCK_M))
  b_offset = off_hz * $(stride_b0h)
  b_ptr_offsets_n_1 = (tl.arange(0, $(BLOCK_N)) % $(BIAS_LAST_SIZE)) +
    $(BIAS_LAST_SIZE)
  b1 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
    $(stride_b0m))[:, None] + b_ptr_offsets_n_1[None, :])
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float16)
    qk += tl.dot(q, k)

    b0 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
      $(stride_b0m))[:, None] + start_n // $(BLOCK_N))
    -- A floating scalar does not widen the bias tensor in Triton.
    -- Resolve its implicit conversion to the storage dtype explicitly.
    qk += ((b0 + b1) * (1.44269504).to(tl.float16))

    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc *= alpha[:, None]
    acc += tl.dot((p).to(tl.float16), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  acc = acc * (1.0 / l_i[:, None])
  O_block_ptr = tl.make_block_ptr(base=Out + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  tl.store(O_block_ptr, (acc).to(tl.float16))
}


end FP16

namespace BF16

def original
    (q k v bias out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk
      stride_kh stride_kn stride_kk
      _stride_vh stride_vk stride_vn
      _stride_oh stride_om stride_on
      stride_b0h stride_b0m
      _Z _H N_CTX P_SEQ BIAS_LAST_SIZE _B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    : ComputeKernel :=
  let Q : Region .bf16 := q.cast
  let K : Region .bf16 := k.cast
  let V : Region .bf16 := v.cast
  let B0 : Region .bf16 := bias.cast
  let Out : Region .bf16 := out.cast
  triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  q_offset = off_hz * $(stride_qh)
  kv_offset = off_hz * $(stride_kh)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + kv_offset,
    shape=($((BLOCK_DMODEL : Nat)), $((N_CTX + P_SEQ : Nat))),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + kv_offset,
    shape=($((N_CTX + P_SEQ : Nat)), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(tl.bfloat16)
  lo = 0
  hi = $((N_CTX + P_SEQ : Nat))

  b_ptr_offsets_m = tl.arange(0, $(BLOCK_M))
  b_offset = off_hz * $(stride_b0h)
  b_ptr_offsets_n_1 = (tl.arange(0, $(BLOCK_N)) % $(BIAS_LAST_SIZE)) +
    $(BIAS_LAST_SIZE)
  b1 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
    $(stride_b0m))[:, None] + b_ptr_offsets_n_1[None, :])
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.bfloat16)
    qk += tl.dot(q, k)

    b0 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
      $(stride_b0m))[:, None] + start_n // $(BLOCK_N))
    -- A floating scalar does not widen the bias tensor in Triton.
    -- Resolve its implicit conversion to the storage dtype explicitly.
    qk += ((b0 + b1) * (1.44269504).to(tl.bfloat16))

    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc *= alpha[:, None]
    acc += tl.dot((p).to(tl.bfloat16), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  acc = acc / l_i[:, None]
  O_block_ptr = tl.make_block_ptr(base=Out + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  tl.store(O_block_ptr, (acc).to(tl.bfloat16))
}


def reciprocal
    (q k v bias out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk
      stride_kh stride_kn stride_kk
      _stride_vh stride_vk stride_vn
      _stride_oh stride_om stride_on
      stride_b0h stride_b0m
      _Z _H N_CTX P_SEQ BIAS_LAST_SIZE _B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    : ComputeKernel :=
  let Q : Region .bf16 := q.cast
  let K : Region .bf16 := k.cast
  let V : Region .bf16 := v.cast
  let B0 : Region .bf16 := bias.cast
  let Out : Region .bf16 := out.cast
  triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  q_offset = off_hz * $(stride_qh)
  kv_offset = off_hz * $(stride_kh)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + kv_offset,
    shape=($((BLOCK_DMODEL : Nat)), $((N_CTX + P_SEQ : Nat))),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + kv_offset,
    shape=($((N_CTX + P_SEQ : Nat)), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(tl.bfloat16)
  lo = 0
  hi = $((N_CTX + P_SEQ : Nat))

  b_ptr_offsets_m = tl.arange(0, $(BLOCK_M))
  b_offset = off_hz * $(stride_b0h)
  b_ptr_offsets_n_1 = (tl.arange(0, $(BLOCK_N)) % $(BIAS_LAST_SIZE)) +
    $(BIAS_LAST_SIZE)
  b1 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
    $(stride_b0m))[:, None] + b_ptr_offsets_n_1[None, :])
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.bfloat16)
    qk += tl.dot(q, k)

    b0 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
      $(stride_b0m))[:, None] + start_n // $(BLOCK_N))
    -- A floating scalar does not widen the bias tensor in Triton.
    -- Resolve its implicit conversion to the storage dtype explicitly.
    qk += ((b0 + b1) * (1.44269504).to(tl.bfloat16))

    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc *= alpha[:, None]
    acc += tl.dot((p).to(tl.bfloat16), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  acc = acc * (1.0 / l_i[:, None])
  O_block_ptr = tl.make_block_ptr(base=Out + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  tl.store(O_block_ptr, (acc).to(tl.bfloat16))
}


end BF16

/-- Select the host-supported storage type before constructing the complete source. -/
def original (storage : Storage) :=
  match storage with
  | .fp16 => FP16.original
  | .bf16 => BF16.original

def reciprocal (storage : Storage) :=
  match storage with
  | .fp16 => FP16.reciprocal
  | .bf16 => BF16.reciprocal

end VeriTile.Bench.TritonBenchG.AttentionKernel.TypedSource
