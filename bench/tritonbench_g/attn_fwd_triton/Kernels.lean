import VeriTile.Triton.DSL

/-! Complete original and reciprocal Triton kernels. The candidate replaces only
the final acc / l_i[:, None] with acc * (1.0 / l_i[:, None]).

Both defined outer-stage choices of
`attn_fwd_triton.py`'s `_attn_fwd`: STAGE=3 runs prefix then diagonal,
STAGE=2 runs the diagonal twice. Other choices leave the first helper's lo/hi
undefined in Python; they are excluded explicitly by the source contract.

Each helper invocation has its own pointer copies, so the first call cannot
advance the caller's bases used by the second call. The diagonal stage adds
the finite -1e6 penalty to qk. It never replaces a score outright or zeros p.
Q/K/V/Out have bf16 storage, Q_scale/K_scale fp32. Both dot output precisions,
probability cast, exp2, load masks and the final store remain explicit.
The source constants 128/96 are represented by BLOCK_DMODEL/HEAD_ACTIVE;
unused source strides remain arguments. Host launch and fixed-width integer
overflow are outside this per-program model. Separate real and FP proofs must
use these complete sources, including both helper ranges.

Translation-surface blocker: the two JIT helper calls are inlined with separate
local pointer copies. The supplied bf16 Q/K/V/Out and fp32 scale types are
explicit; Python head constants 128/96 are generalized to BLOCK_DMODEL and
HEAD_ACTIVE. Unsupported outer stages that leave lo/hi undefined are excluded.
Host launch and fixed-width address overflow remain outside this per-program
model. Correct.lean and FPEquiv.lean use these same two sources. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTriton.TypedSource
open VeriTile.Triton

def original
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
    (_hStage : STAGE = 2 ∨ STAGE = 3) : ComputeKernel :=
  let FIRST_DIAGONAL := decide (STAGE = 2)
  .mk [Q,K,V,Q_scale,K_scale] [Out] (triton {
  tl.region Q = tl.bfloat16, K = tl.bfloat16, V = tl.bfloat16,
    Q_scale = tl.float32, K_scale = tl.float32, Out = tl.bfloat16
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  qvk_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  vk_offset = qvk_offset // $(stride_qm)
  q_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_M))
  k_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_N))

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  Q_scale_ptr = Q_scale + q_scale_offset + start_m
  K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * $(stride_kn)
  K_scale_ptr = K_scale + k_scale_offset
  V_ptrs = V + qvk_offset + offs_n[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  O_block_ptr = Out + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(Q_ptrs,
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
  q_scale = tl.load(Q_scale_ptr)
  -- Local helper pointers: the first call does not mutate the caller's bases.
  first_lo = 0
  first_hi = start_m * $(BLOCK_M)
  K_ptrs_first = K_ptrs
  K_scale_ptr_first = K_scale_ptr
  V_ptrs_first = V_ptrs
  if FIRST_DIAGONAL {
    first_lo = start_m * $(BLOCK_M)
    first_hi = (start_m + $(1)) * $(BLOCK_M)
    first_lo = tl.multiple_of(first_lo, $(BLOCK_M))
    K_scale_ptr_first += first_lo // $(BLOCK_N)
    K_ptrs_first += $(HEAD_DIM) * first_lo
    V_ptrs_first += $(HEAD_DIM) * first_lo
  }
  for start_n in range(first_lo, first_hi, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs_first, mask=k_mask)
    k_scale = tl.load(K_scale_ptr_first)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    if FIRST_DIAGONAL {
      mask_first = offs_m[:, None] >= (start_n + offs_n[None, :])
      qk = qk + tl.where(mask_first, 0, -1.0e6)
      m_ij = tl.maximum(m_i, tl.max(qk, 1))
      qk -= m_ij[:, None]
    } else {
      m_ij = tl.maximum(m_i, tl.max(qk, 1))
      qk = qk - m_ij[:, None]
    }
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs_first,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, (v).to(tl.float16), out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs_first += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr_first += $(1)
    V_ptrs_first += $(BLOCK_N) * $(HEAD_DIM)
  }
  -- The second helper starts from the caller's original pointers and advances
  -- to the diagonal block, exactly as its STAGE=2 prefix does.
  second_lo = start_m * $(BLOCK_M)
  second_hi = (start_m + $(1)) * $(BLOCK_M)
  second_lo = tl.multiple_of(second_lo, $(BLOCK_M))
  K_scale_ptr_second = K_scale_ptr + second_lo // $(BLOCK_N)
  K_ptrs_second = K_ptrs + $(HEAD_DIM) * second_lo
  V_ptrs_second = V_ptrs + $(HEAD_DIM) * second_lo
  for start_n in range(second_lo, second_hi, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs_second, mask=k_mask)
    k_scale = tl.load(K_scale_ptr_second)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    qk = qk + tl.where(mask, 0, -1.0e6)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs_second,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, (v).to(tl.float16), out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs_second += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr_second += $(1)
    V_ptrs_second += $(BLOCK_N) * $(HEAD_DIM)
  }
  acc = acc / l_i[:, None]
  -- Out.type.element_ty is bf16 in the Python wrapper.
  tl.store(O_block_ptr, (acc).to(tl.bfloat16),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}).surfaceBody

/-- Candidate changes only final row division to reciprocal multiplication. -/
def reciprocal
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
    (_hStage : STAGE = 2 ∨ STAGE = 3) : ComputeKernel :=
  let FIRST_DIAGONAL := decide (STAGE = 2)
  .mk [Q,K,V,Q_scale,K_scale] [Out] (triton {
  tl.region Q = tl.bfloat16, K = tl.bfloat16, V = tl.bfloat16,
    Q_scale = tl.float32, K_scale = tl.float32, Out = tl.bfloat16
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  qvk_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  vk_offset = qvk_offset // $(stride_qm)
  q_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_M))
  k_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_N))

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  Q_scale_ptr = Q_scale + q_scale_offset + start_m
  K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * $(stride_kn)
  K_scale_ptr = K_scale + k_scale_offset
  V_ptrs = V + qvk_offset + offs_n[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  O_block_ptr = Out + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(Q_ptrs,
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
  q_scale = tl.load(Q_scale_ptr)
  -- Local helper pointers: the first call does not mutate the caller's bases.
  first_lo = 0
  first_hi = start_m * $(BLOCK_M)
  K_ptrs_first = K_ptrs
  K_scale_ptr_first = K_scale_ptr
  V_ptrs_first = V_ptrs
  if FIRST_DIAGONAL {
    first_lo = start_m * $(BLOCK_M)
    first_hi = (start_m + $(1)) * $(BLOCK_M)
    first_lo = tl.multiple_of(first_lo, $(BLOCK_M))
    K_scale_ptr_first += first_lo // $(BLOCK_N)
    K_ptrs_first += $(HEAD_DIM) * first_lo
    V_ptrs_first += $(HEAD_DIM) * first_lo
  }
  for start_n in range(first_lo, first_hi, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs_first, mask=k_mask)
    k_scale = tl.load(K_scale_ptr_first)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    if FIRST_DIAGONAL {
      mask_first = offs_m[:, None] >= (start_n + offs_n[None, :])
      qk = qk + tl.where(mask_first, 0, -1.0e6)
      m_ij = tl.maximum(m_i, tl.max(qk, 1))
      qk -= m_ij[:, None]
    } else {
      m_ij = tl.maximum(m_i, tl.max(qk, 1))
      qk = qk - m_ij[:, None]
    }
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs_first,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, (v).to(tl.float16), out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs_first += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr_first += $(1)
    V_ptrs_first += $(BLOCK_N) * $(HEAD_DIM)
  }
  -- The second helper starts from the caller's original pointers and advances
  -- to the diagonal block, exactly as its STAGE=2 prefix does.
  second_lo = start_m * $(BLOCK_M)
  second_hi = (start_m + $(1)) * $(BLOCK_M)
  second_lo = tl.multiple_of(second_lo, $(BLOCK_M))
  K_scale_ptr_second = K_scale_ptr + second_lo // $(BLOCK_N)
  K_ptrs_second = K_ptrs + $(HEAD_DIM) * second_lo
  V_ptrs_second = V_ptrs + $(HEAD_DIM) * second_lo
  for start_n in range(second_lo, second_hi, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs_second, mask=k_mask)
    k_scale = tl.load(K_scale_ptr_second)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    qk = qk + tl.where(mask, 0, -1.0e6)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs_second,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, (v).to(tl.float16), out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs_second += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr_second += $(1)
    V_ptrs_second += $(BLOCK_N) * $(HEAD_DIM)
  }
  acc = acc * (1.0 / l_i[:, None])
  -- Out.type.element_ty is bf16 in the Python wrapper.
  tl.store(O_block_ptr, (acc).to(tl.bfloat16),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}).surfaceBody

end VeriTile.Bench.TritonBenchG.AttnFwdTriton.TypedSource
