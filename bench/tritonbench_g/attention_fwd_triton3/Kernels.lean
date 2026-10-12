import VeriTile.Triton.DSL

/-! Complete typed source for `attention_fwd_triton3.py`'s `_attn_fwd`,
with `_attn_fwd_inner` inlined.
The supplied Python tests use fp16 Q/K/V/Out and fp32 M/L; the source-dependent
v.dtype and Out.type.element_ty casts are therefore explicit tl.float16 casts.
The reciprocal candidate changes only the final normalization in the END arm.
The resume load inherits fp16 storage from Out, then casts the loaded value
to fp32; load/cast inference retains both operations.

Window arithmetic uses mathematical signed integers, runtime division truncates
toward zero, and the loop retains signed bounds and indices. Hardware integer
overflow is outside this model; source execution requires positive block sizes
and valid head divisors. Every original flag, mask, cast and M/L/Out store is
retained. Correct.lean and FPEquiv.lean state the independent real IO contract and
conditional FP equivalence for these same implementations.

Translation-surface blocker: the helper JIT is inlined, region storage types
and the V/Out element casts are explicit, and constexpr parameters use Lean
antiquotations. The outer-call text sequence cannot align the inlined source;
README.md documents the source mapping and corresponding Lean regressions.

The real projection deliberately erases IEEE NaN behavior; the FP interpreter
keeps `tmp != tmp` as a backend comparison, including the alpha repair.
-/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.TypedSource
open VeriTile.Triton

def original
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW : Nat)
    (sliding_window_offset sliding_window_size : Int) :
    ComputeKernel := triton {
  tl.region Q = tl.float16, K = tl.float16, V = tl.float16, Out = tl.float16, M = tl.float32, L = tl.float32
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  off_hkv = off_h // ($(H) // $(H_KV))
  q_offset = off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh)
  k_offset = off_z.to(tl.int64) * $(stride_kz) + off_hkv.to(tl.int64) * $(stride_kh)
  v_offset = off_z.to(tl.int64) * $(stride_vz) + off_hkv.to(tl.int64) * $(stride_vh)
  o_offset = off_z.to(tl.int64) * $(stride_oz) + off_h.to(tl.int64) * $(stride_oh)

  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)), offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)), order=(1, 0))
  V_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=($(NKV_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)), offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)), order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=($(BLOCK_DMODEL), $(NKV_CTX)),
    strides=($(stride_kk), $(stride_kn)), offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)), order=(0, 1))
  O_block_ptr = tl.make_block_ptr(base=Out + o_offset, shape=($(ROUND_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)), offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)), order=(1, 0))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  m_ptrs = M + off_hz * $(ROUND_CTX) + offs_m
  l_ptrs = L + off_hz * $(ROUND_CTX) + offs_m
  if $(INIT) != $(0) {
    m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
    acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  } else {
    m_i = tl.load(m_ptrs).to(tl.float32)
    l_i = tl.load(l_ptrs).to(tl.float32)
    acc = tl.load(O_block_ptr).to(tl.float32)
  }
  qk_scale = $((sm_scale : ℝ))
  qk_scale *= 1.4426950408889634
  if $(IS_EVEN_M) != $(0) {
    q = tl.load(Q_block_ptr)
  } else {
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  if ($(SLIDING_WINDOW) != $(0)) & ($(COMPLEMENT_SLIDING_WINDOW) == $(0)) {
    if $(COMPLEMENT_SLIDING_WINDOW) != $(0) {
      lo = $((0 : Int))
      hi = (((start_m + 1) * $(BLOCK_M) + $((sliding_window_offset : Int))
        - $((sliding_window_size : Int)) + $(BLOCK_N) - 1) // $(BLOCK_N)) * $(BLOCK_N)
    } else {
      lo = ((start_m * $(BLOCK_M) + $((sliding_window_offset : Int))
        - $((sliding_window_size : Int)) + 1) // $(BLOCK_N)) * $(BLOCK_N)
      hi = ((((start_m + 1) * $(BLOCK_M) - 1) + $((sliding_window_offset : Int))
        + $(BLOCK_N)) // $(BLOCK_N)) * $(BLOCK_N)
      if lo < $((0 : Int)) {
        lo = $((0 : Int))
      }
      if hi > $(NKV_CTX) {
        hi = $(((NKV_CTX : Int) : Int))
      }
      lo = tl.multiple_of(lo, $(BLOCK_N))
      K_block_ptr = tl.advance(K_block_ptr, (0, lo))
      V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    }
  } else {
    lo = $((0 : Int))
    hi = $(((NKV_CTX : Int) : Int))
  }
  for start_n in range(lo, hi, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    if $(IS_EVEN_N) != $(0) {
      k = tl.load(K_block_ptr)
    } else {
      k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk * qk_scale
    if $(SLIDING_WINDOW) != $(0) {
      dist = tl.arange(0, $(BLOCK_M))[:, None] - tl.arange(0, $(BLOCK_N))[None, :]
        + start_m * $(BLOCK_M) - start_n + $((sliding_window_offset : Int))
      if $(COMPLEMENT_SLIDING_WINDOW) != $(0) {
        mask = dist >= $((sliding_window_size : Int))
      } else {
        mask = (dist >= $(0)) & (dist < $((sliding_window_size : Int)))
      }
      qk = tl.where(mask, qk, float("-inf"))
    }
    if $(IS_EVEN_N) == $(0) {
      qk = tl.where(((tl.arange(0, $(BLOCK_N)) + start_n) < $(NKV_CTX))[None, :], qk, float("-inf"))
    }
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    if $(SLIDING_WINDOW) != $(0) {
      p = tl.where(mask, p, 0.0)
    }
    if $(IS_EVEN_N) == $(0) {
      p = tl.where(((tl.arange(0, $(BLOCK_N)) + start_n) < $(NKV_CTX))[None, :], p, 0.0)
    }
    l_ij = tl.sum(p, 1)
    tmp = m_i - m_ij
    alpha_mask = tmp != tmp
    alpha = tl.math.exp2(tmp)
    alpha = tl.where(alpha_mask, 1.0, alpha)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    if $(IS_EVEN_N) != $(0) {
      v = tl.load(V_block_ptr)
    } else {
      v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    acc += tl.dot((p).to(tl.float16), v)
    m_i = m_ij
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
  }
  if $(END) != $(0) {
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
  } else {
    tl.store(l_ptrs, l_i)
  }
  tl.store(m_ptrs, m_i)
  tl.store(O_block_ptr, (acc).to(tl.float16))
}

/-- Reciprocal normalization, preserving the full original context. -/
def reciprocal
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW : Nat)
    (sliding_window_offset sliding_window_size : Int) :
    ComputeKernel := triton {
  tl.region Q = tl.float16, K = tl.float16, V = tl.float16, Out = tl.float16, M = tl.float32, L = tl.float32
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  off_hkv = off_h // ($(H) // $(H_KV))
  q_offset = off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh)
  k_offset = off_z.to(tl.int64) * $(stride_kz) + off_hkv.to(tl.int64) * $(stride_kh)
  v_offset = off_z.to(tl.int64) * $(stride_vz) + off_hkv.to(tl.int64) * $(stride_vh)
  o_offset = off_z.to(tl.int64) * $(stride_oz) + off_h.to(tl.int64) * $(stride_oh)

  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)), offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)), order=(1, 0))
  V_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=($(NKV_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)), offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)), order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=($(BLOCK_DMODEL), $(NKV_CTX)),
    strides=($(stride_kk), $(stride_kn)), offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)), order=(0, 1))
  O_block_ptr = tl.make_block_ptr(base=Out + o_offset, shape=($(ROUND_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)), offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)), order=(1, 0))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  m_ptrs = M + off_hz * $(ROUND_CTX) + offs_m
  l_ptrs = L + off_hz * $(ROUND_CTX) + offs_m
  if $(INIT) != $(0) {
    m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
    acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  } else {
    m_i = tl.load(m_ptrs).to(tl.float32)
    l_i = tl.load(l_ptrs).to(tl.float32)
    acc = tl.load(O_block_ptr).to(tl.float32)
  }
  qk_scale = $((sm_scale : ℝ))
  qk_scale *= 1.4426950408889634
  if $(IS_EVEN_M) != $(0) {
    q = tl.load(Q_block_ptr)
  } else {
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  if ($(SLIDING_WINDOW) != $(0)) & ($(COMPLEMENT_SLIDING_WINDOW) == $(0)) {
    if $(COMPLEMENT_SLIDING_WINDOW) != $(0) {
      lo = $((0 : Int))
      hi = (((start_m + 1) * $(BLOCK_M) + $((sliding_window_offset : Int))
        - $((sliding_window_size : Int)) + $(BLOCK_N) - 1) // $(BLOCK_N)) * $(BLOCK_N)
    } else {
      lo = ((start_m * $(BLOCK_M) + $((sliding_window_offset : Int))
        - $((sliding_window_size : Int)) + 1) // $(BLOCK_N)) * $(BLOCK_N)
      hi = ((((start_m + 1) * $(BLOCK_M) - 1) + $((sliding_window_offset : Int))
        + $(BLOCK_N)) // $(BLOCK_N)) * $(BLOCK_N)
      if lo < $((0 : Int)) {
        lo = $((0 : Int))
      }
      if hi > $(NKV_CTX) {
        hi = $(((NKV_CTX : Int) : Int))
      }
      lo = tl.multiple_of(lo, $(BLOCK_N))
      K_block_ptr = tl.advance(K_block_ptr, (0, lo))
      V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    }
  } else {
    lo = $((0 : Int))
    hi = $(((NKV_CTX : Int) : Int))
  }
  for start_n in range(lo, hi, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    if $(IS_EVEN_N) != $(0) {
      k = tl.load(K_block_ptr)
    } else {
      k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk * qk_scale
    if $(SLIDING_WINDOW) != $(0) {
      dist = tl.arange(0, $(BLOCK_M))[:, None] - tl.arange(0, $(BLOCK_N))[None, :]
        + start_m * $(BLOCK_M) - start_n + $((sliding_window_offset : Int))
      if $(COMPLEMENT_SLIDING_WINDOW) != $(0) {
        mask = dist >= $((sliding_window_size : Int))
      } else {
        mask = (dist >= $(0)) & (dist < $((sliding_window_size : Int)))
      }
      qk = tl.where(mask, qk, float("-inf"))
    }
    if $(IS_EVEN_N) == $(0) {
      qk = tl.where(((tl.arange(0, $(BLOCK_N)) + start_n) < $(NKV_CTX))[None, :], qk, float("-inf"))
    }
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    if $(SLIDING_WINDOW) != $(0) {
      p = tl.where(mask, p, 0.0)
    }
    if $(IS_EVEN_N) == $(0) {
      p = tl.where(((tl.arange(0, $(BLOCK_N)) + start_n) < $(NKV_CTX))[None, :], p, 0.0)
    }
    l_ij = tl.sum(p, 1)
    tmp = m_i - m_ij
    alpha_mask = tmp != tmp
    alpha = tl.math.exp2(tmp)
    alpha = tl.where(alpha_mask, 1.0, alpha)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    if $(IS_EVEN_N) != $(0) {
      v = tl.load(V_block_ptr)
    } else {
      v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    acc += tl.dot((p).to(tl.float16), v)
    m_i = m_ij
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
  }
  if $(END) != $(0) {
    m_i += tl.math.log2(l_i)
    acc = acc * (1.0 / l_i[:, None])
  } else {
    tl.store(l_ptrs, l_i)
  }
  tl.store(m_ptrs, m_i)
  tl.store(O_block_ptr, (acc).to(tl.float16))
}

end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.TypedSource
