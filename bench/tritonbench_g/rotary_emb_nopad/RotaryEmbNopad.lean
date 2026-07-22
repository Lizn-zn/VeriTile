import VeriTile.Triton

/-!
# `rotary_emb_nopad` — strict per-kernel correctness

`rotary_embedding_kernel` applies the half-split (non-interleaved) rotary
position embedding in place to padding-free Q/K token buffers: each program owns
one head (`program_id(0)`) and one token-block (`program_id(1)`), loads the
per-token `cos`/`sin` half-dim vectors, and rewrites the two head-dim halves via
`(q0*cos - q1*sin, q0*sin + q1*cos)` for Q, and the same for K only on the GQA
group leader (`cur_head_idx % KV_GROUP_NUM == 0`). The file also covers the
`fused_rotary_embedding_kernel_v2` KV-cache writeback variant.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (the `(Q_HEAD_NUM, token-blocks)` grid, the
`HEAD_DIM`/`BLOCK_TOKENS`/`KV_GROUP_NUM` choices, the token/head stride
bookkeeping, the v2 paged-KV-cache block/last-block index computation, and how
the runtime composes per-program writes into one buffer) is the *trusted
boundary*, not a proof obligation here. Because the head, token-block, and
per-lane indices are universally quantified, the per-program statement covers
every program of the grid.

## Proof architecture

```
rotary_emb_nopad_output_summary_general               ← TOP (dimension-general)
      ├─ rotary_embedding_kernel_surface_toAlgorithm_supported
      ├─ rotary_embedding_q_surface_q0_compute_correct → rotary_embedding_q_surface_q0_correct   (Q first half)
      ├─ rotary_embedding_q_surface_q1_compute_correct → rotary_embedding_q_surface_q1_correct   (Q second half)
      ├─ rotary_embedding_k_surface_k0_compute_correct → rotary_embedding_k_surface_k0_correct   (K first half)
      └─ rotary_embedding_k_surface_k1_compute_correct → rotary_embedding_k_surface_k1_correct   (K second half)

block-level lemmas (single head/token-block scope):
  rotary_embedding_q0/q1/k0/k1_block_compute_correct → *_block_correct

fused-v2 KV-cache + Q writeback track:
  fused_rotary_v2_kv_cache_first/second_half_store_slice_compute_correct → *_correct
  fused_rotary_v2_context_kv_cache_first/second_half_store_slice_compute_correct
  fused_rotary_v2_q_first/second_half_store_slice_compute_correct → *_correct
```

Offset injectivity/disjointness (for both the `q_*` and `k_*` families) is
taken as hypotheses of the main theorem.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; dtype casts erase to the
identity at the algorithm layer (post-erasure all dtypes unify to `ℝ`).
`cos`/`sin` are modeled as **precomputed inputs** loaded from memory, not
computed. Scoping is **one half-store at a time** (q0/q1/k0/k1), each over the
active lanes (`tokens_range < q_total_tokens`, `cur_head_idx < Q_HEAD_NUM`, and
for K the GQA-leader predicate); out-of-bounds lanes are preserved verbatim. The
top summary is dimension-general, covering arbitrary symbolic token/head/KV-group
counts, head-dim half, block size, and strides for the no-cache
`_rotary_embedding_kernel`. The `fused_rotary_embedding_kernel_v2` KV-cache/Q
track is verified at the per-store-slice level (including the context paged-cache
offsets) but is **not folded into a single top-level v2 summary** here — honestly,
only the no-cache kernel has a public `output_summary`. `@triton.autotune` is not
modeled.
-/

namespace VeriTile.Bench.TritonBenchG.RotaryEmbNopad

open VeriTile.Triton
open scoped VeriTile.Triton.GroupedMasked2DKernelIO

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `rotary_emb_nopad.py`'s
`rotary_embedding_kernel`.

This keeps the unconditional Q rotary writes and the conditional K rotary path
in one kernel; the smaller Q/K kernels below remain available for local
correctness arguments. -/
def rotary_embedding_kernel_surface
    (q k cos sin : RegionName)
    (q_token_stride q_head_stride k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride q_total_tokens Q_HEAD_NUM KV_GROUP_NUM
      HEAD_DIM BLOCK_TOKENS : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_block_idx = tl.program_id(1)

  tokens_range = cur_token_block_idx * $(BLOCK_TOKENS) + tl.arange(0, $(BLOCK_TOKENS))
  dim_range0 = tl.arange(0, $(HEAD_DIM) // $(2))
  dim_range1 = tl.arange($(HEAD_DIM) // $(2), $(HEAD_DIM))

  off_cos_sin = tokens_range[:, None] * $(cos_token_stride) +
    dim_range0[None, :] * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin,
    mask=tokens_range[:, None] < $(q_total_tokens), other=0.0)
  loaded_sin = tl.load(sin + off_cos_sin,
    mask=tokens_range[:, None] < $(q_total_tokens), other=0.0)

  off_q0 = tokens_range[:, None, None] * $(q_token_stride) +
    cur_head_idx * $(q_head_stride) +
    dim_range0[None, None, :] * $(head_dim_stride)
  off_q1 = tokens_range[:, None, None] * $(q_token_stride) +
    cur_head_idx * $(q_head_stride) +
    dim_range1[None, None, :] * $(head_dim_stride)
  loaded_q0 = tl.load(q + off_q0,
    mask=((cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens))),
    other=0.0)
  loaded_q1 = tl.load(q + off_q1,
    mask=((cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens))),
    other=0.0)
  out_q0 = loaded_q0 * loaded_cos[:, None, :] - loaded_q1 * loaded_sin[:, None, :]
  out_q1 = loaded_q0 * loaded_sin[:, None, :] + loaded_q1 * loaded_cos[:, None, :]

  tl.store(q + off_q0, out_q0,
    mask=((cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens))))
  tl.store(q + off_q1, out_q1,
    mask=((cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens))))

  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    off_k0 = tokens_range[:, None, None] * $(k_token_stride) +
      k_head_idx * $(k_head_stride) +
      dim_range0[None, None, :] * $(head_dim_stride)
    off_k1 = tokens_range[:, None, None] * $(k_token_stride) +
      k_head_idx * $(k_head_stride) +
      dim_range1[None, None, :] * $(head_dim_stride)
    loaded_k0 = tl.load(k + off_k0,
      mask=tokens_range[:, None, None] < $(q_total_tokens),
      other=0.0)
    loaded_k1 = tl.load(k + off_k1,
      mask=tokens_range[:, None, None] < $(q_total_tokens),
      other=0.0)
    out_k0 = loaded_k0 * loaded_cos[:, None, :] - loaded_k1 * loaded_sin[:, None, :]
    out_k1 = loaded_k0 * loaded_sin[:, None, :] + loaded_k1 * loaded_cos[:, None, :]
    tl.store(k + off_k0, out_k0,
      mask=tokens_range[:, None, None] < $(q_total_tokens))
    tl.store(k + off_k1, out_k1,
      mask=tokens_range[:, None, None] < $(q_total_tokens))
  }
}

/-- The full Python-shaped rotary kernel surface lowers through algorithm
translation, including both Q stores and the conditional K stores. -/
theorem rotary_embedding_kernel_surface_toAlgorithm_supported
    (q k cos sin : RegionName)
    (q_token_stride q_head_stride k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride q_total_tokens Q_HEAD_NUM KV_GROUP_NUM
      HEAD_DIM BLOCK_TOKENS : Nat) :
    ∃ alg, (rotary_embedding_kernel_surface q k cos sin q_token_stride
      q_head_stride k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride q_total_tokens Q_HEAD_NUM KV_GROUP_NUM
      HEAD_DIM BLOCK_TOKENS).toAlgorithm? = Except.ok alg := by
  simp [rotary_embedding_kernel_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription of the Q part of `rotary_emb_nopad.py`'s
`rotary_embedding_kernel`.

This writes both rotary halves for Q over the full
`[BLOCK_TOKENS, 1, HEAD_HALF]` token/head/dimension tile. The conditional K
branch is represented by `rotary_embedding_k_surface`; the cache-writing v2
branch is represented by `fused_rotary_embedding_v2_surface`. -/
def rotary_embedding_q_surface
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_block_idx = tl.program_id(1)
  tokens_range = cur_token_block_idx * $(BLOCK_TOKENS) + tl.arange(0, $(BLOCK_TOKENS))
  dim_range0 = tl.arange(0, $(HEAD_HALF))
  dim_range1 = dim_range0 + $(HEAD_HALF)
  off_cos_sin = tokens_range[:, None] * $(cos_token_stride) +
    dim_range0[None, :] * $(cos_stride)
  loaded_cos = tl.load(Cos + off_cos_sin,
    mask=tokens_range[:, None] < $(q_total_tokens), other=0.0)
  loaded_sin = tl.load(Sin + off_cos_sin,
    mask=tokens_range[:, None] < $(q_total_tokens), other=0.0)
  off_q0 = tokens_range[:, None, None] * $(q_token_stride) +
    cur_head_idx * $(q_head_stride) +
    dim_range0[None, None, :] * $(head_dim_stride)
  off_q1 = tokens_range[:, None, None] * $(q_token_stride) +
    cur_head_idx * $(q_head_stride) +
    dim_range1[None, None, :] * $(head_dim_stride)
  loaded_q0 = tl.load(Q + off_q0,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens)),
    other=0.0)
  loaded_q1 = tl.load(Q + off_q1,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens)),
    other=0.0)
  out_q0 = loaded_q0 * loaded_cos[:, None, :] - loaded_q1 * loaded_sin[:, None, :]
  out_q1 = loaded_q0 * loaded_sin[:, None, :] + loaded_q1 * loaded_cos[:, None, :]
  tl.store(Q + off_q0, out_q0,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens)))
  tl.store(Q + off_q1, out_q1,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens)))
}

/-- The standalone full-Q rotary surface lowers to the algorithm layer. -/
theorem rotary_embedding_q_surface_toAlgorithm_supported
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    ∃ alg, (rotary_embedding_q_surface Q Cos Sin q_token_stride q_head_stride
      head_dim_stride cos_token_stride cos_stride q_total_tokens Q_HEAD_NUM
      HEAD_HALF BLOCK_TOKENS).toAlgorithm? = Except.ok alg := by
  simp [rotary_embedding_q_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription of the conditional K part of
`rotary_emb_nopad.py`'s `rotary_embedding_kernel` over the full
`[BLOCK_TOKENS, 1, HEAD_HALF]` token/head/dimension tile. -/
def rotary_embedding_k_surface
    (K Cos Sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_block_idx = tl.program_id(1)
  handle_kv = (cur_head_idx % $(KV_GROUP_NUM)) == 0
  if handle_kv {
    k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    tokens_range = cur_token_block_idx * $(BLOCK_TOKENS) + tl.arange(0, $(BLOCK_TOKENS))
    dim_range0 = tl.arange(0, $(HEAD_HALF))
    dim_range1 = dim_range0 + $(HEAD_HALF)
    off_cos_sin = tokens_range[:, None] * $(cos_token_stride) +
      dim_range0[None, :] * $(cos_stride)
    loaded_cos = tl.load(Cos + off_cos_sin,
      mask=tokens_range[:, None] < $(q_total_tokens), other=0.0)
    loaded_sin = tl.load(Sin + off_cos_sin,
      mask=tokens_range[:, None] < $(q_total_tokens), other=0.0)
    off_k0 = tokens_range[:, None, None] * $(k_token_stride) +
      k_head_idx * $(k_head_stride) +
      dim_range0[None, None, :] * $(head_dim_stride)
    off_k1 = tokens_range[:, None, None] * $(k_token_stride) +
      k_head_idx * $(k_head_stride) +
      dim_range1[None, None, :] * $(head_dim_stride)
    loaded_k0 = tl.load(K + off_k0,
      mask=tokens_range[:, None, None] < $(q_total_tokens), other=0.0)
    loaded_k1 = tl.load(K + off_k1,
      mask=tokens_range[:, None, None] < $(q_total_tokens), other=0.0)
    out_k0 = loaded_k0 * loaded_cos[:, None, :] - loaded_k1 * loaded_sin[:, None, :]
    out_k1 = loaded_k0 * loaded_sin[:, None, :] + loaded_k1 * loaded_cos[:, None, :]
    tl.store(K + off_k0, out_k0,
      mask=tokens_range[:, None, None] < $(q_total_tokens))
    tl.store(K + off_k1, out_k1,
      mask=tokens_range[:, None, None] < $(q_total_tokens))
  }
}

/-- The standalone conditional-K rotary surface lowers to the algorithm layer. -/
theorem rotary_embedding_k_surface_toAlgorithm_supported
    (K Cos Sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    ∃ alg, (rotary_embedding_k_surface K Cos Sin k_token_stride k_head_stride
      head_dim_stride cos_token_stride cos_stride q_total_tokens KV_GROUP_NUM
      HEAD_HALF BLOCK_TOKENS).toAlgorithm? = Except.ok alg := by
  simp [rotary_embedding_k_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription of `rotary_emb_nopad.py`'s
`fused_rotary_embedding_kernel_v2`.

Python returns early when `block_head_index >= Q_HEAD_NUM`; this surface
represents that with a guarded body. The benchmark initializes
`context_lengths >= 1`, so the `past_kv_seq_len = context_lengths[...] - 1`
Nat subtraction follows the exercised path. -/
def fused_rotary_embedding_v2_surface
    (Q K Cos Sin KVCache : RegionName) (BlockTables ContextLengths : Region .nat)
    (q_token_stride q_head_stride k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride cacheb_stride cacheh_stride cachebs_stride
      cached_stride bts_stride btb_stride block_size q_total_tokens
      Q_HEAD_NUM HEAD_HALF : Nat) :
    ComputeKernel := triton {
  block_head_index = tl.program_id(0)
  if block_head_index < $(Q_HEAD_NUM) {
    block_token_index = tl.program_id(1)
    dim = tl.arange(0, $(HEAD_HALF))
    dim1 = dim + $(HEAD_HALF)

    off_q0 = block_token_index * $(q_token_stride) +
      block_head_index * $(q_head_stride) + dim * $(head_dim_stride)
    off_q1 = block_token_index * $(q_token_stride) +
      block_head_index * $(q_head_stride) + dim1 * $(head_dim_stride)
    off_k0 = block_token_index * $(k_token_stride) +
      block_head_index * $(k_head_stride) + dim * $(head_dim_stride)
    off_k1 = block_token_index * $(k_token_stride) +
      block_head_index * $(k_head_stride) + dim1 * $(head_dim_stride)

    loaded_q0 = tl.load(Q + off_q0)
    loaded_q1 = tl.load(Q + off_q1)
    loaded_k0 = tl.load(K + off_k0)
    loaded_k1 = tl.load(K + off_k1)

    off_cos_sin = block_token_index * $(cos_token_stride) + dim * $(cos_stride)
    loaded_cos = tl.load(Cos + off_cos_sin,
      mask=block_token_index < $(q_total_tokens), other=0.0)
    loaded_sin = tl.load(Sin + off_cos_sin,
      mask=block_token_index < $(q_total_tokens), other=0.0)

    out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
    out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
    out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos

    past_kv_seq_len = tl.load(ContextLengths + block_token_index) - $(1)
    last_block_idx = past_kv_seq_len // $(block_size)
    block_table_ptr = BlockTables + block_token_index * $(bts_stride)
    block_ids = tl.load(block_table_ptr + last_block_idx * $(btb_stride),
      mask=block_token_index < $(q_total_tokens))
    offsets_in_last_block = (past_kv_seq_len % $(block_size)) * $(cachebs_stride)

    kv_range0 = block_ids * $(cacheb_stride) +
      block_head_index * $(cacheh_stride) + offsets_in_last_block +
      dim * $(cached_stride)
    kv_range1 = block_ids * $(cacheb_stride) +
      block_head_index * $(cacheh_stride) + offsets_in_last_block +
      dim1 * $(cached_stride)

    tl.store(KVCache + kv_range0, out_k0)
    tl.store(KVCache + kv_range1, out_k1)
    tl.store(Q + off_q0, out_q0)
    tl.store(Q + off_q1, out_q1)
  }
}

/-- The full v2 fused rotary surface lowers to the algorithm layer, including
Q stores plus context-length/block-table based KV-cache stores. -/
theorem fused_rotary_embedding_v2_surface_toAlgorithm_supported
    (Q K Cos Sin KVCache : RegionName) (BlockTables ContextLengths : Region .nat)
    (q_token_stride q_head_stride k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride cacheb_stride cacheh_stride cachebs_stride
      cached_stride bts_stride btb_stride block_size q_total_tokens
      Q_HEAD_NUM HEAD_HALF : Nat) :
    ∃ alg, (fused_rotary_embedding_v2_surface Q K Cos Sin KVCache BlockTables
      ContextLengths q_token_stride q_head_stride k_token_stride k_head_stride
      head_dim_stride cos_token_stride cos_stride cacheb_stride cacheh_stride
      cachebs_stride cached_stride bts_stride btb_stride block_size q_total_tokens
      Q_HEAD_NUM HEAD_HALF).toAlgorithm? = Except.ok alg := by
  simp [fused_rotary_embedding_v2_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented Q first-half slice of `rotary_emb_nopad.py`'s
`rotary_embedding_kernel`.

This captures one token block and one Q head for the first Q store:
`out_q0 = q0 * cos - q1 * sin`. -/
def rotary_embedding_q0_block
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_block_idx = tl.program_id(1)
  token = cur_token_block_idx * $(BLOCK_TOKENS)
  dim = tl.arange(0, $(HEAD_HALF))
  off_cos_sin = token * $(cos_token_stride) + dim * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin,
    mask=token < $(q_total_tokens), other=0.0)
  loaded_sin = tl.load(sin + off_cos_sin,
    mask=token < $(q_total_tokens), other=0.0)
  off_q0 = token * $(q_token_stride) + cur_head_idx * $(q_head_stride) +
    dim * $(head_dim_stride)
  off_q1 = token * $(q_token_stride) + cur_head_idx * $(q_head_stride) +
    (dim + $(HEAD_HALF)) * $(head_dim_stride)
  loaded_q0 = tl.load(q + off_q0,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) and (token < $(q_total_tokens)),
    other=0.0)
  loaded_q1 = tl.load(q + off_q1,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) and (token < $(q_total_tokens)),
    other=0.0)
  out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
  tl.store(q + off_q0, out_q0,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) and (token < $(q_total_tokens)))
}

def tokenIndex (s : BlockState) (BLOCK_TOKENS : Nat) : Nat :=
  s.pids 1 * BLOCK_TOKENS

def headIndex (s : BlockState) : Nat :=
  s.pids 0

def dimIndex (i : Fin HEAD_HALF) : Nat :=
  i.val

def active (s : BlockState) (q_total_tokens Q_HEAD_NUM BLOCK_TOKENS : Nat) :
    Prop :=
  headIndex s < Q_HEAD_NUM ∧ tokenIndex s BLOCK_TOKENS < q_total_tokens

instance activeDecidable
    (s : BlockState) (q_total_tokens Q_HEAD_NUM BLOCK_TOKENS : Nat) :
    Decidable (active s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS) := by
  unfold active
  infer_instance

def qOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride
      BLOCK_TOKENS : Nat) (dim : Nat) : Nat :=
  tokenIndex s BLOCK_TOKENS * q_token_stride +
    headIndex s * q_head_stride + dim * head_dim_stride

def cosOffset
    (s : BlockState) (cos_token_stride cos_stride BLOCK_TOKENS : Nat)
    (i : Fin HEAD_HALF) : Nat :=
  tokenIndex s BLOCK_TOKENS * cos_token_stride + dimIndex i * cos_stride

noncomputable def rotaryNopadQ0Spec
    (s : BlockState) (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HEAD_HALF BLOCK_TOKENS : Nat) (i : Fin HEAD_HALF) : ℝ :=
  s.readMem q
      (qOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
        (dimIndex i)) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride BLOCK_TOKENS i) -
  s.readMem q
      (qOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
        (dimIndex i + HEAD_HALF)) *
    s.readMem sin (cosOffset s cos_token_stride cos_stride BLOCK_TOKENS i)

/-- Algorithm-layer correctness for the Q first-half nopad rotary store. -/
theorem rotary_embedding_q0_block_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        qOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
          (dimIndex i)))
    (hExec : exec (rotary_embedding_q0_block q cos sin q_token_stride
        q_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS) s = some s') :
    ∀ i : Fin HEAD_HALF,
      s'.readMem q
          (qOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
            (dimIndex i)) =
        if active s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS then
          rotaryNopadQ0Spec s q cos sin q_token_stride q_head_stride
            head_dim_stride cos_token_stride cos_stride HEAD_HALF
            BLOCK_TOKENS i
        else
          s.readMem q
            (qOffset s q_token_stride q_head_stride head_dim_stride
              BLOCK_TOKENS (dimIndex i)) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HEAD_HALF] =>
        s.pids 1 * BLOCK_TOKENS * q_token_stride +
          s.pids 0 * q_head_stride + idx.1.val * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [qOffset, tokenIndex, headIndex, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hHH : 0 < HEAD_HALF
  · simp [exec, rotary_embedding_q0_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hHH] at hExec
    rw [← hExec]
    simp only [qOffset, tokenIndex, headIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hHead : s.pids 0 < Q_HEAD_NUM
    · by_cases hTok : s.pids 1 * BLOCK_TOKENS < q_total_tokens
      · simp [active, rotaryNopadQ0Spec, qOffset, cosOffset, tokenIndex,
              headIndex, dimIndex, hHead, hTok, Option.bind, Option.map]
      · simp [active, tokenIndex, headIndex, hHead, hTok]
    · simp [active, headIndex, hHead]
  · exact False.elim (hHH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the Q first-half nopad rotary store. -/
theorem rotary_embedding_q0_block_compute_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        qOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
          (dimIndex i))) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rotary_embedding_q0_block q cos sin q_token_stride
        q_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HEAD_HALF =>
          active s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS)
        (fun i => (q,
          qOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
            (dimIndex i))))
      (expected := fun i =>
        rotaryNopadQ0Spec s q cos sin q_token_stride q_head_stride
          head_dim_stride cos_token_stride cos_stride HEAD_HALF BLOCK_TOKENS i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_embedding_q0_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rotary_embedding_q0_block_correct q cos sin q_token_stride
    q_head_stride head_dim_stride cos_token_stride cos_stride q_total_tokens
    Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS s s' hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented Q second-half slice of `rotary_emb_nopad.py`'s
`rotary_embedding_kernel`.

This captures one token block and one Q head for the second Q store:
`out_q1 = q0 * sin + q1 * cos` written to `q + off_q1`. -/
def rotary_embedding_q1_block
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_block_idx = tl.program_id(1)
  token = cur_token_block_idx * $(BLOCK_TOKENS)
  dim = tl.arange(0, $(HEAD_HALF))
  off_cos_sin = token * $(cos_token_stride) + dim * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin,
    mask=token < $(q_total_tokens), other=0.0)
  loaded_sin = tl.load(sin + off_cos_sin,
    mask=token < $(q_total_tokens), other=0.0)
  off_q0 = token * $(q_token_stride) + cur_head_idx * $(q_head_stride) +
    dim * $(head_dim_stride)
  off_q1 = token * $(q_token_stride) + cur_head_idx * $(q_head_stride) +
    (dim + $(HEAD_HALF)) * $(head_dim_stride)
  loaded_q0 = tl.load(q + off_q0,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) and (token < $(q_total_tokens)),
    other=0.0)
  loaded_q1 = tl.load(q + off_q1,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) and (token < $(q_total_tokens)),
    other=0.0)
  out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
  tl.store(q + off_q1, out_q1,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) and (token < $(q_total_tokens)))
}

def qSecondOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride
      BLOCK_TOKENS HEAD_HALF : Nat) (i : Fin HEAD_HALF) : Nat :=
  tokenIndex s BLOCK_TOKENS * q_token_stride +
    headIndex s * q_head_stride + (dimIndex i + HEAD_HALF) * head_dim_stride

noncomputable def rotaryNopadQ1Spec
    (s : BlockState) (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HEAD_HALF BLOCK_TOKENS : Nat) (i : Fin HEAD_HALF) : ℝ :=
  s.readMem q
      (qOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
        (dimIndex i)) *
    s.readMem sin (cosOffset s cos_token_stride cos_stride BLOCK_TOKENS i) +
  s.readMem q
      (qOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
        (dimIndex i + HEAD_HALF)) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride BLOCK_TOKENS i)

/-- Algorithm-layer correctness for the Q second-half nopad rotary store. -/
theorem rotary_embedding_q1_block_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        qSecondOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS HEAD_HALF i))
    (hExec : exec (rotary_embedding_q1_block q cos sin q_token_stride
        q_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS) s = some s') :
    ∀ i : Fin HEAD_HALF,
      s'.readMem q
          (qSecondOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF i) =
        if active s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS then
          rotaryNopadQ1Spec s q cos sin q_token_stride q_head_stride
            head_dim_stride cos_token_stride cos_stride HEAD_HALF
            BLOCK_TOKENS i
        else
          s.readMem q
            (qSecondOffset s q_token_stride q_head_stride head_dim_stride
              BLOCK_TOKENS HEAD_HALF i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HEAD_HALF] =>
        s.pids 1 * BLOCK_TOKENS * q_token_stride +
          s.pids 0 * q_head_stride + (idx.1.val + HEAD_HALF) * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [qSecondOffset, tokenIndex, headIndex, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hHH : 0 < HEAD_HALF
  · simp [exec, rotary_embedding_q1_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hHH] at hExec
    rw [← hExec]
    simp only [qSecondOffset, tokenIndex, headIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hHead : s.pids 0 < Q_HEAD_NUM
    · by_cases hTok : s.pids 1 * BLOCK_TOKENS < q_total_tokens
      · simp [active, rotaryNopadQ1Spec, qOffset, cosOffset, tokenIndex,
              headIndex, dimIndex, hHead, hTok, Option.bind, Option.map]
      · simp [active, tokenIndex, headIndex, hHead, hTok]
    · simp [active, headIndex, hHead]
  · exact False.elim (hHH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the Q second-half nopad rotary store. -/
theorem rotary_embedding_q1_block_compute_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        qSecondOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS HEAD_HALF i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rotary_embedding_q1_block q cos sin q_token_stride
        q_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HEAD_HALF =>
          active s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS)
        (fun i => (q,
          qSecondOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF i)))
      (expected := fun i =>
        rotaryNopadQ1Spec s q cos sin q_token_stride q_head_stride
          head_dim_stride cos_token_stride cos_stride HEAD_HALF BLOCK_TOKENS i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_embedding_q1_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rotary_embedding_q1_block_correct q cos sin q_token_stride
    q_head_stride head_dim_stride cos_token_stride cos_stride q_total_tokens
    Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS s s' hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented K first-half slice of `rotary_emb_nopad.py`'s
`rotary_embedding_kernel`. K analog of `rotary_embedding_q0_block` writing
`out_k0 = k0 * cos - k1 * sin` to `k + off_k0`. The slice does NOT include the
`KV_GROUP_NUM` modular gate (this slice models a one-head-block view; the
group gate is captured separately when proving the full surface). -/
def rotary_embedding_k0_block
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_block_idx = tl.program_id(1)
  token = cur_token_block_idx * $(BLOCK_TOKENS)
  dim = tl.arange(0, $(HEAD_HALF))
  off_cos_sin = token * $(cos_token_stride) + dim * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin,
    mask=token < $(q_total_tokens), other=0.0)
  loaded_sin = tl.load(sin + off_cos_sin,
    mask=token < $(q_total_tokens), other=0.0)
  off_k0 = token * $(k_token_stride) + cur_head_idx * $(k_head_stride) +
    dim * $(head_dim_stride)
  off_k1 = token * $(k_token_stride) + cur_head_idx * $(k_head_stride) +
    (dim + $(HEAD_HALF)) * $(head_dim_stride)
  loaded_k0 = tl.load(k + off_k0,
    mask=(cur_head_idx < $(K_HEAD_NUM)) and (token < $(q_total_tokens)),
    other=0.0)
  loaded_k1 = tl.load(k + off_k1,
    mask=(cur_head_idx < $(K_HEAD_NUM)) and (token < $(q_total_tokens)),
    other=0.0)
  out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
  tl.store(k + off_k0, out_k0,
    mask=(cur_head_idx < $(K_HEAD_NUM)) and (token < $(q_total_tokens)))
}

def kOffset
    (s : BlockState) (k_token_stride k_head_stride head_dim_stride
      BLOCK_TOKENS : Nat) (dim : Nat) : Nat :=
  tokenIndex s BLOCK_TOKENS * k_token_stride +
    headIndex s * k_head_stride + dim * head_dim_stride

def activeK (s : BlockState) (q_total_tokens K_HEAD_NUM BLOCK_TOKENS : Nat) :
    Prop :=
  headIndex s < K_HEAD_NUM ∧ tokenIndex s BLOCK_TOKENS < q_total_tokens

instance activeKDecidable
    (s : BlockState) (q_total_tokens K_HEAD_NUM BLOCK_TOKENS : Nat) :
    Decidable (activeK s q_total_tokens K_HEAD_NUM BLOCK_TOKENS) := by
  unfold activeK
  infer_instance

noncomputable def rotaryNopadK0Spec
    (s : BlockState) (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HEAD_HALF BLOCK_TOKENS : Nat) (i : Fin HEAD_HALF) : ℝ :=
  s.readMem k
      (kOffset s k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS
        (dimIndex i)) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride BLOCK_TOKENS i) -
  s.readMem k
      (kOffset s k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS
        (dimIndex i + HEAD_HALF)) *
    s.readMem sin (cosOffset s cos_token_stride cos_stride BLOCK_TOKENS i)

theorem rotary_embedding_k0_block_correct
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        kOffset s k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS
          (dimIndex i)))
    (hExec : exec (rotary_embedding_k0_block k cos sin k_token_stride
        k_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS) s = some s') :
    ∀ i : Fin HEAD_HALF,
      s'.readMem k
          (kOffset s k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS
            (dimIndex i)) =
        if activeK s q_total_tokens K_HEAD_NUM BLOCK_TOKENS then
          rotaryNopadK0Spec s k cos sin k_token_stride k_head_stride
            head_dim_stride cos_token_stride cos_stride HEAD_HALF
            BLOCK_TOKENS i
        else
          s.readMem k
            (kOffset s k_token_stride k_head_stride head_dim_stride
              BLOCK_TOKENS (dimIndex i)) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HEAD_HALF] =>
        s.pids 1 * BLOCK_TOKENS * k_token_stride +
          s.pids 0 * k_head_stride + idx.1.val * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kOffset, tokenIndex, headIndex, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hHH : 0 < HEAD_HALF
  · simp [exec, rotary_embedding_k0_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hHH] at hExec
    rw [← hExec]
    simp only [kOffset, tokenIndex, headIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hHead : s.pids 0 < K_HEAD_NUM
    · by_cases hTok : s.pids 1 * BLOCK_TOKENS < q_total_tokens
      · simp [activeK, rotaryNopadK0Spec, kOffset, cosOffset, tokenIndex,
              headIndex, dimIndex, hHead, hTok, Option.bind, Option.map]
      · simp [activeK, tokenIndex, headIndex, hHead, hTok]
    · simp [activeK, headIndex, hHead]
  · exact False.elim (hHH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem rotary_embedding_k0_block_compute_correct
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        kOffset s k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS
          (dimIndex i))) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rotary_embedding_k0_block k cos sin k_token_stride
        k_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HEAD_HALF =>
          activeK s q_total_tokens K_HEAD_NUM BLOCK_TOKENS)
        (fun i => (k,
          kOffset s k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS
            (dimIndex i))))
      (expected := fun i =>
        rotaryNopadK0Spec s k cos sin k_token_stride k_head_stride
          head_dim_stride cos_token_stride cos_stride HEAD_HALF BLOCK_TOKENS i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_embedding_k0_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rotary_embedding_k0_block_correct k cos sin k_token_stride
    k_head_stride head_dim_stride cos_token_stride cos_stride q_total_tokens
    K_HEAD_NUM HEAD_HALF BLOCK_TOKENS s s' hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented K second-half slice of `rotary_emb_nopad.py`'s
`rotary_embedding_kernel`. Writes `out_k1 = k0 * sin + k1 * cos` to
`k + off_k1`. -/
def rotary_embedding_k1_block
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_block_idx = tl.program_id(1)
  token = cur_token_block_idx * $(BLOCK_TOKENS)
  dim = tl.arange(0, $(HEAD_HALF))
  off_cos_sin = token * $(cos_token_stride) + dim * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin,
    mask=token < $(q_total_tokens), other=0.0)
  loaded_sin = tl.load(sin + off_cos_sin,
    mask=token < $(q_total_tokens), other=0.0)
  off_k0 = token * $(k_token_stride) + cur_head_idx * $(k_head_stride) +
    dim * $(head_dim_stride)
  off_k1 = token * $(k_token_stride) + cur_head_idx * $(k_head_stride) +
    (dim + $(HEAD_HALF)) * $(head_dim_stride)
  loaded_k0 = tl.load(k + off_k0,
    mask=(cur_head_idx < $(K_HEAD_NUM)) and (token < $(q_total_tokens)),
    other=0.0)
  loaded_k1 = tl.load(k + off_k1,
    mask=(cur_head_idx < $(K_HEAD_NUM)) and (token < $(q_total_tokens)),
    other=0.0)
  out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos
  tl.store(k + off_k1, out_k1,
    mask=(cur_head_idx < $(K_HEAD_NUM)) and (token < $(q_total_tokens)))
}

def kSecondOffset
    (s : BlockState) (k_token_stride k_head_stride head_dim_stride
      BLOCK_TOKENS HEAD_HALF : Nat) (i : Fin HEAD_HALF) : Nat :=
  tokenIndex s BLOCK_TOKENS * k_token_stride +
    headIndex s * k_head_stride + (dimIndex i + HEAD_HALF) * head_dim_stride

noncomputable def rotaryNopadK1Spec
    (s : BlockState) (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HEAD_HALF BLOCK_TOKENS : Nat) (i : Fin HEAD_HALF) : ℝ :=
  s.readMem k
      (kOffset s k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS
        (dimIndex i)) *
    s.readMem sin (cosOffset s cos_token_stride cos_stride BLOCK_TOKENS i) +
  s.readMem k
      (kOffset s k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS
        (dimIndex i + HEAD_HALF)) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride BLOCK_TOKENS i)

theorem rotary_embedding_k1_block_correct
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        kSecondOffset s k_token_stride k_head_stride head_dim_stride
          BLOCK_TOKENS HEAD_HALF i))
    (hExec : exec (rotary_embedding_k1_block k cos sin k_token_stride
        k_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS) s = some s') :
    ∀ i : Fin HEAD_HALF,
      s'.readMem k
          (kSecondOffset s k_token_stride k_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF i) =
        if activeK s q_total_tokens K_HEAD_NUM BLOCK_TOKENS then
          rotaryNopadK1Spec s k cos sin k_token_stride k_head_stride
            head_dim_stride cos_token_stride cos_stride HEAD_HALF
            BLOCK_TOKENS i
        else
          s.readMem k
            (kSecondOffset s k_token_stride k_head_stride head_dim_stride
              BLOCK_TOKENS HEAD_HALF i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HEAD_HALF] =>
        s.pids 1 * BLOCK_TOKENS * k_token_stride +
          s.pids 0 * k_head_stride + (idx.1.val + HEAD_HALF) * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kSecondOffset, tokenIndex, headIndex, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hHH : 0 < HEAD_HALF
  · simp [exec, rotary_embedding_k1_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hHH] at hExec
    rw [← hExec]
    simp only [kSecondOffset, tokenIndex, headIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hHead : s.pids 0 < K_HEAD_NUM
    · by_cases hTok : s.pids 1 * BLOCK_TOKENS < q_total_tokens
      · simp [activeK, rotaryNopadK1Spec, kOffset, cosOffset, tokenIndex,
              headIndex, dimIndex, hHead, hTok, Option.bind, Option.map]
      · simp [activeK, tokenIndex, headIndex, hHead, hTok]
    · simp [activeK, headIndex, hHead]
  · exact False.elim (hHH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem rotary_embedding_k1_block_compute_correct
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        kSecondOffset s k_token_stride k_head_stride head_dim_stride
          BLOCK_TOKENS HEAD_HALF i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rotary_embedding_k1_block k cos sin k_token_stride
        k_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HEAD_HALF =>
          activeK s q_total_tokens K_HEAD_NUM BLOCK_TOKENS)
        (fun i => (k,
          kSecondOffset s k_token_stride k_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF i)))
      (expected := fun i =>
        rotaryNopadK1Spec s k cos sin k_token_stride k_head_stride
          head_dim_stride cos_token_stride cos_stride HEAD_HALF BLOCK_TOKENS i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_embedding_k1_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rotary_embedding_k1_block_correct k cos sin k_token_stride
    k_head_stride head_dim_stride cos_token_stride cos_stride q_total_tokens
    K_HEAD_NUM HEAD_HALF BLOCK_TOKENS s s' hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented kv_cache first-half store slice of
`rotary_emb_nopad.py`'s `fused_rotary_embedding_kernel_v2`. Takes a
precomputed `OutK0Pre` tile and proves the writeback into `kv_cache` at the
cache-layout first-half offsets. Parameterized over
`(block_ids, offsets_in_last_block)`. -/
def fused_rotary_v2_kv_cache_first_half_store_slice
    (OutK0Pre kv_cache : RegionName)
    (block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
      HEAD_HALF : Nat) :
    ComputeKernel := triton {
  block_head_index = tl.program_id(0)
  dim_range0 = tl.arange(0, $(HEAD_HALF))
  out_k0 = tl.load(OutK0Pre + dim_range0)
  kv_range0 = $(block_ids) * $(cacheb_stride) +
    block_head_index * $(cacheh_stride) +
    $(offsets_in_last_block) +
    dim_range0 * $(cached_stride)
  tl.store(kv_cache + kv_range0, out_k0)
}

def kvCacheFirstOffset
    (s : BlockState)
    (block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride : Nat)
    (i : Fin HEAD_HALF) : Nat :=
  block_ids * cacheb_stride + s.pids 0 * cacheh_stride +
    offsets_in_last_block + i.val * cached_stride

noncomputable def kvCacheFirstStoreSpec
    (s : BlockState) (OutK0Pre : RegionName) (i : Fin HEAD_HALF) : ℝ :=
  s.readMem OutK0Pre i.val

theorem fused_rotary_v2_kv_cache_first_half_store_slice_correct
    (OutK0Pre kv_cache : RegionName)
    (block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
      HEAD_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        kvCacheFirstOffset s block_ids offsets_in_last_block cacheb_stride
          cacheh_stride cached_stride i))
    (hExec : exec (fused_rotary_v2_kv_cache_first_half_store_slice OutK0Pre
        kv_cache block_ids offsets_in_last_block cacheb_stride cacheh_stride
        cached_stride HEAD_HALF) s = some s') :
    ∀ i : Fin HEAD_HALF,
      s'.readMem kv_cache
          (kvCacheFirstOffset s block_ids offsets_in_last_block cacheb_stride
            cacheh_stride cached_stride i) =
        kvCacheFirstStoreSpec s OutK0Pre i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HEAD_HALF] =>
        block_ids * cacheb_stride + s.pids 0 * cacheh_stride +
          offsets_in_last_block + idx.1.val * cached_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kvCacheFirstOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, fused_rotary_v2_kv_cache_first_half_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul] at hExec
  rw [← hExec]
  simp only [kvCacheFirstOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [kvCacheFirstStoreSpec]

theorem fused_rotary_v2_kv_cache_first_half_store_slice_compute_correct
    (OutK0Pre kv_cache : RegionName)
    (block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
      HEAD_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        kvCacheFirstOffset s block_ids offsets_in_last_block cacheb_stride
          cacheh_stride cached_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_rotary_v2_kv_cache_first_half_store_slice OutK0Pre
        kv_cache block_ids offsets_in_last_block cacheb_stride cacheh_stride
        cached_stride HEAD_HALF)
      (initialState := s)
      (write := fun i : Fin HEAD_HALF => some (kv_cache,
        kvCacheFirstOffset s block_ids offsets_in_last_block cacheb_stride
          cacheh_stride cached_stride i))
      (expected := fun i => kvCacheFirstStoreSpec s OutK0Pre i) := by
  unfold ComputeCorrect.Realizes_without_Rounding
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_rotary_v2_kv_cache_first_half_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact fused_rotary_v2_kv_cache_first_half_store_slice_correct OutK0Pre kv_cache
    block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
    HEAD_HALF s s' hOutInj hExec i

/-- Proof-oriented kv_cache second-half store slice of
`rotary_emb_nopad.py`'s `fused_rotary_embedding_kernel_v2`. Companion to
the first-half kv_cache slice. -/
def fused_rotary_v2_kv_cache_second_half_store_slice
    (OutK1Pre kv_cache : RegionName)
    (block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
      HEAD_HALF : Nat) :
    ComputeKernel := triton {
  block_head_index = tl.program_id(0)
  dim_range1 = tl.arange(0, $(HEAD_HALF)) + $(HEAD_HALF)
  out_k1 = tl.load(OutK1Pre + dim_range1)
  kv_range1 = $(block_ids) * $(cacheb_stride) +
    block_head_index * $(cacheh_stride) +
    $(offsets_in_last_block) +
    dim_range1 * $(cached_stride)
  tl.store(kv_cache + kv_range1, out_k1)
}

def kvCacheSecondOffset
    (s : BlockState)
    (block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
      HEAD_HALF : Nat) (i : Fin HEAD_HALF) : Nat :=
  block_ids * cacheb_stride + s.pids 0 * cacheh_stride +
    offsets_in_last_block + (i.val + HEAD_HALF) * cached_stride

noncomputable def kvCacheSecondStoreSpec
    (s : BlockState) (OutK1Pre : RegionName) (HEAD_HALF : Nat) (i : Fin HEAD_HALF) : ℝ :=
  s.readMem OutK1Pre (i.val + HEAD_HALF)

theorem fused_rotary_v2_kv_cache_second_half_store_slice_correct
    (OutK1Pre kv_cache : RegionName)
    (block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
      HEAD_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        kvCacheSecondOffset s block_ids offsets_in_last_block cacheb_stride
          cacheh_stride cached_stride HEAD_HALF i))
    (hExec : exec (fused_rotary_v2_kv_cache_second_half_store_slice OutK1Pre
        kv_cache block_ids offsets_in_last_block cacheb_stride cacheh_stride
        cached_stride HEAD_HALF) s = some s') :
    ∀ i : Fin HEAD_HALF,
      s'.readMem kv_cache
          (kvCacheSecondOffset s block_ids offsets_in_last_block cacheb_stride
            cacheh_stride cached_stride HEAD_HALF i) =
        kvCacheSecondStoreSpec s OutK1Pre HEAD_HALF i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HEAD_HALF] =>
        block_ids * cacheb_stride + s.pids 0 * cacheh_stride +
          offsets_in_last_block + (idx.1.val + HEAD_HALF) * cached_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kvCacheSecondOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, fused_rotary_v2_kv_cache_second_half_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul] at hExec
  rw [← hExec]
  simp only [kvCacheSecondOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [kvCacheSecondStoreSpec]

theorem fused_rotary_v2_kv_cache_second_half_store_slice_compute_correct
    (OutK1Pre kv_cache : RegionName)
    (block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
      HEAD_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        kvCacheSecondOffset s block_ids offsets_in_last_block cacheb_stride
          cacheh_stride cached_stride HEAD_HALF i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_rotary_v2_kv_cache_second_half_store_slice OutK1Pre
        kv_cache block_ids offsets_in_last_block cacheb_stride cacheh_stride
        cached_stride HEAD_HALF)
      (initialState := s)
      (write := fun i : Fin HEAD_HALF => some (kv_cache,
        kvCacheSecondOffset s block_ids offsets_in_last_block cacheb_stride
          cacheh_stride cached_stride HEAD_HALF i))
      (expected := fun i => kvCacheSecondStoreSpec s OutK1Pre HEAD_HALF i) := by
  unfold ComputeCorrect.Realizes_without_Rounding
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_rotary_v2_kv_cache_second_half_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact fused_rotary_v2_kv_cache_second_half_store_slice_correct OutK1Pre kv_cache
    block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
    HEAD_HALF s s' hOutInj hExec i

/-- Python v2 metadata: `past_kv_seq_len =
context_lengths[block_token_index] - 1`. -/
def v2PastKvSeqLen (s : BlockState) (ContextLengths : RegionName) : Nat :=
  s.readMemValue .nat ContextLengths (s.pids 1) - 1

/-- Python v2 metadata: `last_block_idx = past_kv_seq_len // block_size`. -/
def v2LastBlockIdx
    (s : BlockState) (ContextLengths : RegionName) (block_size : Nat) : Nat :=
  v2PastKvSeqLen s ContextLengths / block_size

/-- Python v2 metadata: `block_ids = block_tables[token, last_block_idx]`. -/
def v2BlockId
    (s : BlockState) (BlockTables ContextLengths : RegionName)
    (bts_stride btb_stride block_size : Nat) : Nat :=
  s.readMemValue .nat BlockTables
    (s.pids 1 * bts_stride +
      v2LastBlockIdx s ContextLengths block_size * btb_stride)

/-- Python v2 metadata:
`offsets_in_last_block = (past_kv_seq_len % block_size) * cachebs_stride`. -/
def v2OffsetsInLastBlock
    (s : BlockState) (ContextLengths : RegionName)
    (block_size cachebs_stride : Nat) : Nat :=
  (v2PastKvSeqLen s ContextLengths % block_size) * cachebs_stride

def v2KvCacheFirstOffset
    (s : BlockState) (BlockTables ContextLengths : RegionName)
    (cacheb_stride cacheh_stride cachebs_stride cached_stride bts_stride
      btb_stride block_size : Nat) (i : Fin HEAD_HALF) : Nat :=
  kvCacheFirstOffset s
    (v2BlockId s BlockTables ContextLengths bts_stride btb_stride block_size)
    (v2OffsetsInLastBlock s ContextLengths block_size cachebs_stride)
    cacheb_stride cacheh_stride cached_stride i

def v2KvCacheSecondOffset
    (s : BlockState) (BlockTables ContextLengths : RegionName)
    (cacheb_stride cacheh_stride cachebs_stride cached_stride bts_stride
      btb_stride block_size HEAD_HALF : Nat) (i : Fin HEAD_HALF) : Nat :=
  kvCacheSecondOffset s
    (v2BlockId s BlockTables ContextLengths bts_stride btb_stride block_size)
    (v2OffsetsInLastBlock s ContextLengths block_size cachebs_stride)
    cacheb_stride cacheh_stride cached_stride HEAD_HALF i

/-- Public v2 cache theorem with `block_ids` and `offsets_in_last_block`
specialized to the Python metadata loads from `BlockTables` and
`ContextLengths`. -/
theorem fused_rotary_v2_context_kv_cache_first_half_store_slice_compute_correct
    (OutK0Pre kv_cache BlockTables ContextLengths : RegionName)
    (cacheb_stride cacheh_stride cachebs_stride cached_stride bts_stride
      btb_stride block_size HEAD_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        v2KvCacheFirstOffset s BlockTables ContextLengths cacheb_stride
          cacheh_stride cachebs_stride cached_stride bts_stride btb_stride
          block_size i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_rotary_v2_kv_cache_first_half_store_slice OutK0Pre
        kv_cache
        (v2BlockId s BlockTables ContextLengths bts_stride btb_stride block_size)
        (v2OffsetsInLastBlock s ContextLengths block_size cachebs_stride)
        cacheb_stride cacheh_stride cached_stride HEAD_HALF)
      (initialState := s)
      (write := fun i : Fin HEAD_HALF => some (kv_cache,
        v2KvCacheFirstOffset s BlockTables ContextLengths cacheb_stride
          cacheh_stride cachebs_stride cached_stride bts_stride btb_stride
          block_size i))
      (expected := fun i => kvCacheFirstStoreSpec s OutK0Pre i) := by
  simpa [v2KvCacheFirstOffset]
    using fused_rotary_v2_kv_cache_first_half_store_slice_compute_correct
      OutK0Pre kv_cache
      (v2BlockId s BlockTables ContextLengths bts_stride btb_stride block_size)
      (v2OffsetsInLastBlock s ContextLengths block_size cachebs_stride)
      cacheb_stride cacheh_stride cached_stride HEAD_HALF s hOutInj

/-- Public v2 second-half cache theorem with `block_ids` and
`offsets_in_last_block` specialized to the Python metadata loads. -/
theorem fused_rotary_v2_context_kv_cache_second_half_store_slice_compute_correct
    (OutK1Pre kv_cache BlockTables ContextLengths : RegionName)
    (cacheb_stride cacheh_stride cachebs_stride cached_stride bts_stride
      btb_stride block_size HEAD_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        v2KvCacheSecondOffset s BlockTables ContextLengths cacheb_stride
          cacheh_stride cachebs_stride cached_stride bts_stride btb_stride
          block_size HEAD_HALF i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_rotary_v2_kv_cache_second_half_store_slice OutK1Pre
        kv_cache
        (v2BlockId s BlockTables ContextLengths bts_stride btb_stride block_size)
        (v2OffsetsInLastBlock s ContextLengths block_size cachebs_stride)
        cacheb_stride cacheh_stride cached_stride HEAD_HALF)
      (initialState := s)
      (write := fun i : Fin HEAD_HALF => some (kv_cache,
        v2KvCacheSecondOffset s BlockTables ContextLengths cacheb_stride
          cacheh_stride cachebs_stride cached_stride bts_stride btb_stride
          block_size HEAD_HALF i))
      (expected := fun i => kvCacheSecondStoreSpec s OutK1Pre HEAD_HALF i) := by
  simpa [v2KvCacheSecondOffset]
    using fused_rotary_v2_kv_cache_second_half_store_slice_compute_correct
      OutK1Pre kv_cache
      (v2BlockId s BlockTables ContextLengths bts_stride btb_stride block_size)
      (v2OffsetsInLastBlock s ContextLengths block_size cachebs_stride)
      cacheb_stride cacheh_stride cached_stride HEAD_HALF s hOutInj

/-! ## Fused v2 Q writeback slices -/

/-- Proof-oriented Q first-half store slice for
`fused_rotary_embedding_kernel_v2`.

The full v2 kernel first checks `block_head_index < Q_HEAD_NUM`, then writes
the rotated Q halves for the selected single token. This slice starts after the
rotary arithmetic has produced a precomputed `OutQ0Pre` vector and proves the
first-half Q writeback under the same head guard. -/
def fused_rotary_v2_q_first_half_store_slice
    (OutQ0Pre Q : RegionName)
    (q_token_stride q_head_stride head_dim_stride Q_HEAD_NUM HEAD_HALF : Nat) :
    ComputeKernel := triton {
  block_head_index = tl.program_id(0)
  if block_head_index < $(Q_HEAD_NUM) {
    block_token_index = tl.program_id(1)
    dim = tl.arange(0, $(HEAD_HALF))
    out_q0 = tl.load(OutQ0Pre + dim)
    off_q0 = block_token_index * $(q_token_stride) +
      block_head_index * $(q_head_stride) + dim * $(head_dim_stride)
    tl.store(Q + off_q0, out_q0)
  }
}

/-- Proof-oriented Q second-half store slice for
`fused_rotary_embedding_kernel_v2`. Companion to
`fused_rotary_v2_q_first_half_store_slice`. -/
def fused_rotary_v2_q_second_half_store_slice
    (OutQ1Pre Q : RegionName)
    (q_token_stride q_head_stride head_dim_stride Q_HEAD_NUM HEAD_HALF : Nat) :
    ComputeKernel := triton {
  block_head_index = tl.program_id(0)
  if block_head_index < $(Q_HEAD_NUM) {
    block_token_index = tl.program_id(1)
    dim = tl.arange(0, $(HEAD_HALF))
    dim1 = dim + $(HEAD_HALF)
    out_q1 = tl.load(OutQ1Pre + dim)
    off_q1 = block_token_index * $(q_token_stride) +
      block_head_index * $(q_head_stride) + dim1 * $(head_dim_stride)
    tl.store(Q + off_q1, out_q1)
  }
}

def v2QActive (s : BlockState) (Q_HEAD_NUM : Nat) : Prop :=
  s.pids 0 < Q_HEAD_NUM

instance v2QActiveDecidable (s : BlockState) (Q_HEAD_NUM : Nat) :
    Decidable (v2QActive s Q_HEAD_NUM) := by
  unfold v2QActive
  infer_instance

def v2QFirstOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride : Nat)
    (i : Fin HEAD_HALF) : Nat :=
  s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
    i.val * head_dim_stride

def v2QSecondOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride
      HEAD_HALF : Nat) (i : Fin HEAD_HALF) : Nat :=
  s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
    (i.val + HEAD_HALF) * head_dim_stride

noncomputable def v2QStoreSpec
    (s : BlockState) (OutQPre : RegionName) (i : Fin HEAD_HALF) : ℝ :=
  s.readMem OutQPre i.val

theorem fused_rotary_v2_q_first_half_store_slice_correct
    (OutQ0Pre Q : RegionName)
    (q_token_stride q_head_stride head_dim_stride Q_HEAD_NUM HEAD_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        v2QFirstOffset s q_token_stride q_head_stride head_dim_stride i))
    (hExec : exec (fused_rotary_v2_q_first_half_store_slice OutQ0Pre Q
        q_token_stride q_head_stride head_dim_stride Q_HEAD_NUM HEAD_HALF) s =
      some s') :
    ∀ i : Fin HEAD_HALF,
      s'.readMem Q
          (v2QFirstOffset s q_token_stride q_head_stride head_dim_stride i) =
        if v2QActive s Q_HEAD_NUM then
          v2QStoreSpec s OutQ0Pre i
        else
          s.readMem Q
            (v2QFirstOffset s q_token_stride q_head_stride head_dim_stride i) := by
  intro i
  by_cases hHead : s.pids 0 < Q_HEAD_NUM
  · have hRawInj : Function.Injective
        (fun idx : TileIndex [HEAD_HALF] =>
          s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
            idx.1.val * head_dim_stride) := by
      intro a b h
      have hab : a.1 = b.1 := by
        apply hOutInj
        simpa [v2QFirstOffset] using h
      cases a; cases b
      simp only at hab; cases hab; rfl
    simp [exec, fused_rotary_v2_q_first_half_store_slice, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hHead] at hExec
    rw [← hExec]
    simp only [v2QFirstOffset]
    rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
    simp [v2QActive, v2QStoreSpec, hHead]
  · simp [exec, fused_rotary_v2_q_first_half_store_slice, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, ComparableDType.lt,
          hHead] at hExec
    rw [← hExec]
    simp [v2QActive, hHead]

theorem fused_rotary_v2_q_first_half_store_slice_compute_correct
    (OutQ0Pre Q : RegionName)
    (q_token_stride q_head_stride head_dim_stride Q_HEAD_NUM HEAD_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        v2QFirstOffset s q_token_stride q_head_stride head_dim_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_rotary_v2_q_first_half_store_slice OutQ0Pre Q
        q_token_stride q_head_stride head_dim_stride Q_HEAD_NUM HEAD_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HEAD_HALF => v2QActive s Q_HEAD_NUM)
        (fun i => (Q,
          v2QFirstOffset s q_token_stride q_head_stride head_dim_stride i)))
      (expected := fun i => v2QStoreSpec s OutQ0Pre i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_rotary_v2_q_first_half_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fused_rotary_v2_q_first_half_store_slice_correct OutQ0Pre Q
    q_token_stride q_head_stride head_dim_stride Q_HEAD_NUM HEAD_HALF s s'
    hOutInj hExec i
  simpa [hActive] using h

theorem fused_rotary_v2_q_second_half_store_slice_correct
    (OutQ1Pre Q : RegionName)
    (q_token_stride q_head_stride head_dim_stride Q_HEAD_NUM HEAD_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        v2QSecondOffset s q_token_stride q_head_stride head_dim_stride
          HEAD_HALF i))
    (hExec : exec (fused_rotary_v2_q_second_half_store_slice OutQ1Pre Q
        q_token_stride q_head_stride head_dim_stride Q_HEAD_NUM HEAD_HALF) s =
      some s') :
    ∀ i : Fin HEAD_HALF,
      s'.readMem Q
          (v2QSecondOffset s q_token_stride q_head_stride head_dim_stride
            HEAD_HALF i) =
        if v2QActive s Q_HEAD_NUM then
          v2QStoreSpec s OutQ1Pre i
        else
          s.readMem Q
            (v2QSecondOffset s q_token_stride q_head_stride head_dim_stride
              HEAD_HALF i) := by
  intro i
  by_cases hHead : s.pids 0 < Q_HEAD_NUM
  · have hRawInj : Function.Injective
        (fun idx : TileIndex [HEAD_HALF] =>
          s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
            (idx.1.val + HEAD_HALF) * head_dim_stride) := by
      intro a b h
      have hab : a.1 = b.1 := by
        apply hOutInj
        simpa [v2QSecondOffset] using h
      cases a; cases b
      simp only at hab; cases hab; rfl
    simp [exec, fused_rotary_v2_q_second_half_store_slice, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hHead] at hExec
    rw [← hExec]
    simp only [v2QSecondOffset]
    rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
    simp [v2QActive, v2QStoreSpec, hHead]
  · simp [exec, fused_rotary_v2_q_second_half_store_slice, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, ComparableDType.lt,
          hHead] at hExec
    rw [← hExec]
    simp [v2QActive, hHead]

theorem fused_rotary_v2_q_second_half_store_slice_compute_correct
    (OutQ1Pre Q : RegionName)
    (q_token_stride q_head_stride head_dim_stride Q_HEAD_NUM HEAD_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        v2QSecondOffset s q_token_stride q_head_stride head_dim_stride
          HEAD_HALF i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_rotary_v2_q_second_half_store_slice OutQ1Pre Q
        q_token_stride q_head_stride head_dim_stride Q_HEAD_NUM HEAD_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HEAD_HALF => v2QActive s Q_HEAD_NUM)
        (fun i => (Q,
          v2QSecondOffset s q_token_stride q_head_stride head_dim_stride
            HEAD_HALF i)))
      (expected := fun i => v2QStoreSpec s OutQ1Pre i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_rotary_v2_q_second_half_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fused_rotary_v2_q_second_half_store_slice_correct OutQ1Pre Q
    q_token_stride q_head_stride head_dim_stride Q_HEAD_NUM HEAD_HALF s s'
    hOutInj hExec i
  simpa [hActive] using h

/-! ## Full-kernel Q-surface store proofs

The full Q surface `rotary_embedding_q_surface` writes the Q first-half and
Q second-half tiles consecutively over the 3D `[BLOCK_TOKENS, 1, HEAD_HALF]`
tile. We characterize each store at the algorithm layer using the substrate
helpers `BlockState.scatter_readback_prop_masked_nd` (for the active scatter)
and `BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem` (for
stripping the sibling Q store, which targets the same region at disjoint
offsets). -/

/-- Per-tile-index Q first-half offset for the full Q surface kernel.
The 3D tile shape is `[BLOCK_TOKENS, 1, HEAD_HALF]`. -/
def qFullFirstOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride
      BLOCK_TOKENS : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Nat :=
  (s.pids 1 * BLOCK_TOKENS + idx.1.val) * q_token_stride +
    s.pids 0 * q_head_stride + idx.2.2.1.val * head_dim_stride

/-- Per-tile-index Q second-half offset for the full Q surface kernel. -/
def qFullSecondOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride
      BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Nat :=
  (s.pids 1 * BLOCK_TOKENS + idx.1.val) * q_token_stride +
    s.pids 0 * q_head_stride + (idx.2.2.1.val + HEAD_HALF) * head_dim_stride

/-- Per-tile-index cos/sin offset (the cos/sin tile is `[BLOCK_TOKENS,
HEAD_HALF]` and broadcasts into the 3D Q tile along the singleton head
axis). -/
def cosFullOffset
    (s : BlockState) (cos_token_stride cos_stride BLOCK_TOKENS : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Nat :=
  (s.pids 1 * BLOCK_TOKENS + idx.1.val) * cos_token_stride +
    idx.2.2.1.val * cos_stride

/-- Active predicate for the Q stores in the full Q surface kernel: head
index in range AND the per-tile token index in range. -/
def activeFull (s : BlockState) (q_total_tokens Q_HEAD_NUM BLOCK_TOKENS : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Prop :=
  s.pids 0 < Q_HEAD_NUM ∧
    s.pids 1 * BLOCK_TOKENS + idx.1.val < q_total_tokens

instance activeFullDecidable (s : BlockState)
    (q_total_tokens Q_HEAD_NUM BLOCK_TOKENS : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) :
    Decidable (activeFull s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS idx) := by
  unfold activeFull
  infer_instance

/-- Q first-half rotary spec for the full Q surface kernel:
`q0 * cos - q1 * sin`. -/
noncomputable def rotaryNopadQ0FullSpec
    (s : BlockState) (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : ℝ :=
  s.readMem q
      (qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
        BLOCK_TOKENS idx) *
    s.readMem cos (cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS idx) -
  s.readMem q
      (qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
        BLOCK_TOKENS HEAD_HALF idx) *
    s.readMem sin (cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS idx)

/-- Q second-half rotary spec for the full Q surface kernel:
`q0 * sin + q1 * cos`. -/
noncomputable def rotaryNopadQ1FullSpec
    (s : BlockState) (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : ℝ :=
  s.readMem q
      (qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
        BLOCK_TOKENS idx) *
    s.readMem sin (cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS idx) +
  s.readMem q
      (qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
        BLOCK_TOKENS HEAD_HALF idx) *
    s.readMem cos (cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS idx)

/-- Algorithm-layer correctness for the Q first-half store in the full
3D-tile Q surface kernel `rotary_embedding_q_surface`. The Q second-half
store (same region, disjoint offsets) is stripped via
`foldl_writeMem_same_region_disjoint_offsets_readMem`. -/
theorem rotary_embedding_q_surface_q0_correct
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS idx))
    (hOutDisjoint :
      ∀ idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS idx ≠
          qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF idx')
    (hExec : exec (rotary_embedding_q_surface Q Cos Sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS) s = some s') :
    ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      s'.readMem Q
          (qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS idx) =
        if activeFull s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS idx then
          rotaryNopadQ0FullSpec s Q Cos Sin q_token_stride q_head_stride
            head_dim_stride cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF idx
        else
          s.readMem Q
            (qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
              BLOCK_TOKENS idx) := by
  intro idx
  by_cases hTok : 0 < BLOCK_TOKENS
  · by_cases hHalf : 0 < HEAD_HALF
    · have hRawInj : Function.Injective
          (fun j : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
            (s.pids 1 * BLOCK_TOKENS + j.1.val) * q_token_stride +
              s.pids 0 * q_head_stride + j.2.2.1.val * head_dim_stride) := by
        simpa [qFullFirstOffset] using hOutInj
      simp [exec, rotary_embedding_q_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop, Tile.remap,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            ComparableDType.lt, hTok, hHalf,
            TileShape.insertAxis, TileShape.dropInsertedIndex,
            TileShape.dropInsertedIndex_two_pair,
            TileShape.dropInsertedIndex_zero_pair] at hExec
      rw [← hExec]
      simp only [qFullFirstOffset]
      -- Strip the outer Q second-half foldl: same region Q, disjoint offsets.
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (region := Q)
            (P := fun j : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
              s.pids 0 < Q_HEAD_NUM ∧
                s.pids 1 * BLOCK_TOKENS + j.1.val < q_total_tokens)
            (offsetFn := fun j : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
                (s.pids 1 * BLOCK_TOKENS + j.1.val) * q_token_stride +
                  s.pids 0 * q_head_stride +
                  (j.2.2.1.val + HEAD_HALF) * head_dim_stride)
            (hOff := fun k _ _ => by
              have := hOutDisjoint idx k
              simpa [qFullFirstOffset, qFullSecondOffset] using this)]
      -- The inner foldl is the Q first-half scatter.
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj idx]
      by_cases hHead : s.pids 0 < Q_HEAD_NUM
      · by_cases hTokIdx : s.pids 1 * BLOCK_TOKENS + idx.1.val < q_total_tokens
        · simp [activeFull, rotaryNopadQ0FullSpec, qFullFirstOffset,
                qFullSecondOffset, cosFullOffset, hHead, hTokIdx,
                Option.bind, Option.map]
        · simp [activeFull, hHead, hTokIdx]
      · simp [activeFull, hHead]
    · exact False.elim (hHalf (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.2.2.1.isLt))
  · exact False.elim (hTok (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.1.isLt))

/-- Compute-facing correctness for the Q first-half store in the full
`rotary_embedding_q_surface` kernel. -/
theorem rotary_embedding_q_surface_q0_compute_correct
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS idx))
    (hOutDisjoint :
      ∀ idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS idx ≠
          qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF idx') :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rotary_embedding_q_surface Q Cos Sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
          activeFull s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS idx)
        (fun idx => (Q,
          qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS idx)))
      (expected := fun idx =>
        rotaryNopadQ0FullSpec s Q Cos Sin q_token_stride q_head_stride
          head_dim_stride cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_embedding_q_surface]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := rotary_embedding_q_surface_q0_correct Q Cos Sin
    q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
    q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS s s' hOutInj
    hOutDisjoint hExec idx
  simpa [hActive] using h

/-- Algorithm-layer correctness for the Q second-half store in the full
3D-tile Q surface kernel `rotary_embedding_q_surface`. -/
theorem rotary_embedding_q_surface_q1_correct
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS HEAD_HALF idx))
    (hOutDisjoint :
      ∀ idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS idx ≠
          qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF idx')
    (hExec : exec (rotary_embedding_q_surface Q Cos Sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS) s = some s') :
    ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      s'.readMem Q
          (qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF idx) =
        if activeFull s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS idx then
          rotaryNopadQ1FullSpec s Q Cos Sin q_token_stride q_head_stride
            head_dim_stride cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF idx
        else
          s.readMem Q
            (qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
              BLOCK_TOKENS HEAD_HALF idx) := by
  intro idx
  by_cases hTok : 0 < BLOCK_TOKENS
  · by_cases hHalf : 0 < HEAD_HALF
    · have hRawInj : Function.Injective
          (fun j : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
            (s.pids 1 * BLOCK_TOKENS + j.1.val) * q_token_stride +
              s.pids 0 * q_head_stride +
              (j.2.2.1.val + HEAD_HALF) * head_dim_stride) := by
        simpa [qFullSecondOffset] using hOutInj
      simp [exec, rotary_embedding_q_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop, Tile.remap,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            ComparableDType.lt, hTok, hHalf,
            TileShape.insertAxis, TileShape.dropInsertedIndex,
            TileShape.dropInsertedIndex_two_pair,
            TileShape.dropInsertedIndex_zero_pair] at hExec
      rw [← hExec]
      simp only [qFullSecondOffset]
      -- The outer foldl is the Q second-half scatter.
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj idx]
      by_cases hHead : s.pids 0 < Q_HEAD_NUM
      · by_cases hTokIdx : s.pids 1 * BLOCK_TOKENS + idx.1.val < q_total_tokens
        · simp [activeFull, rotaryNopadQ1FullSpec, qFullFirstOffset,
                qFullSecondOffset, cosFullOffset, hHead, hTokIdx,
                Option.bind, Option.map]
        · rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
              (region := Q)
              (P := fun j : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
                s.pids 0 < Q_HEAD_NUM ∧
                  s.pids 1 * BLOCK_TOKENS + j.1.val < q_total_tokens)
              (offsetFn := fun j : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
                  (s.pids 1 * BLOCK_TOKENS + j.1.val) * q_token_stride +
                    s.pids 0 * q_head_stride +
                    j.2.2.1.val * head_dim_stride)
              (hOff := fun k _ _ => by
                have := hOutDisjoint k idx
                simpa [qFullFirstOffset, qFullSecondOffset] using this.symm)]
          simp [activeFull, hHead, hTokIdx]
      · rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (region := Q)
            (P := fun j : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
              s.pids 0 < Q_HEAD_NUM ∧
                s.pids 1 * BLOCK_TOKENS + j.1.val < q_total_tokens)
            (offsetFn := fun j : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
                (s.pids 1 * BLOCK_TOKENS + j.1.val) * q_token_stride +
                  s.pids 0 * q_head_stride +
                  j.2.2.1.val * head_dim_stride)
            (hOff := fun k _ _ => by
              have := hOutDisjoint k idx
              simpa [qFullFirstOffset, qFullSecondOffset] using this.symm)]
        simp [activeFull, hHead]
    · exact False.elim (hHalf (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.2.2.1.isLt))
  · exact False.elim (hTok (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.1.isLt))

/-- Compute-facing correctness for the Q second-half store in the full
`rotary_embedding_q_surface` kernel. -/
theorem rotary_embedding_q_surface_q1_compute_correct
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS HEAD_HALF idx))
    (hOutDisjoint :
      ∀ idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS idx ≠
          qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF idx') :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rotary_embedding_q_surface Q Cos Sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
          activeFull s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS idx)
        (fun idx => (Q,
          qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF idx)))
      (expected := fun idx =>
        rotaryNopadQ1FullSpec s Q Cos Sin q_token_stride q_head_stride
          head_dim_stride cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_embedding_q_surface]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := rotary_embedding_q_surface_q1_correct Q Cos Sin
    q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
    q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS s s' hOutInj
    hOutDisjoint hExec idx
  simpa [hActive] using h

/-! ## Full-kernel K-surface store proofs -/

/-- Per-tile-index K first-half offset for the full K surface kernel.
The K head is selected by the Python kernel as `cur_head_idx // KV_GROUP_NUM`
after the modular `handle_kv` gate has fired. -/
def kFullFirstOffset
    (s : BlockState) (k_token_stride k_head_stride head_dim_stride
      KV_GROUP_NUM BLOCK_TOKENS : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Nat :=
  (s.pids 1 * BLOCK_TOKENS + idx.1.val) * k_token_stride +
    (s.pids 0 / KV_GROUP_NUM) * k_head_stride +
    idx.2.2.1.val * head_dim_stride

/-- Per-tile-index K second-half offset for the full K surface kernel. -/
def kFullSecondOffset
    (s : BlockState) (k_token_stride k_head_stride head_dim_stride
      KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Nat :=
  (s.pids 1 * BLOCK_TOKENS + idx.1.val) * k_token_stride +
    (s.pids 0 / KV_GROUP_NUM) * k_head_stride +
    (idx.2.2.1.val + HEAD_HALF) * head_dim_stride

/-- Active predicate for the K stores in the full K surface kernel: the modular
KV-group gate fires and the per-tile token index is in range. -/
def activeKFull (s : BlockState) (q_total_tokens KV_GROUP_NUM BLOCK_TOKENS : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Prop :=
  s.pids 0 % KV_GROUP_NUM = 0 ∧
    s.pids 1 * BLOCK_TOKENS + idx.1.val < q_total_tokens

instance activeKFullDecidable (s : BlockState)
    (q_total_tokens KV_GROUP_NUM BLOCK_TOKENS : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) :
    Decidable (activeKFull s q_total_tokens KV_GROUP_NUM BLOCK_TOKENS idx) := by
  unfold activeKFull
  infer_instance

/-- K first-half rotary spec for the full K surface kernel:
`k0 * cos - k1 * sin`. -/
noncomputable def rotaryNopadK0FullSpec
    (s : BlockState) (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : ℝ :=
  s.readMem k
      (kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
        KV_GROUP_NUM BLOCK_TOKENS idx) *
    s.readMem cos (cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS idx) -
  s.readMem k
      (kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
        KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx) *
    s.readMem sin (cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS idx)

/-- K second-half rotary spec for the full K surface kernel:
`k0 * sin + k1 * cos`. -/
noncomputable def rotaryNopadK1FullSpec
    (s : BlockState) (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : ℝ :=
  s.readMem k
      (kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
        KV_GROUP_NUM BLOCK_TOKENS idx) *
    s.readMem sin (cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS idx) +
  s.readMem k
      (kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
        KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx) *
    s.readMem cos (cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS idx)

/-- Algorithm-layer correctness for the K first-half store in the full
3D-tile K surface kernel `rotary_embedding_k_surface`. -/
theorem rotary_embedding_k_surface_k0_correct
    (K Cos Sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
          KV_GROUP_NUM BLOCK_TOKENS idx))
    (hOutDisjoint :
      ∀ idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
        kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS idx ≠
          kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx')
    (hExec : exec (rotary_embedding_k_surface K Cos Sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS) s = some s') :
    ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      s'.readMem K
          (kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS idx) =
        if activeKFull s q_total_tokens KV_GROUP_NUM BLOCK_TOKENS idx then
          rotaryNopadK0FullSpec s K Cos Sin k_token_stride k_head_stride
            head_dim_stride cos_token_stride cos_stride KV_GROUP_NUM
            BLOCK_TOKENS HEAD_HALF idx
        else
          s.readMem K
            (kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
              KV_GROUP_NUM BLOCK_TOKENS idx) := by
  intro idx
  by_cases hTok : 0 < BLOCK_TOKENS
  · by_cases hHalf : 0 < HEAD_HALF
    · have hRawInj : Function.Injective
          (fun j : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
            (s.pids 1 * BLOCK_TOKENS + j.1.val) * k_token_stride +
              (s.pids 0 / KV_GROUP_NUM) * k_head_stride +
              j.2.2.1.val * head_dim_stride) := by
        simpa [kFullFirstOffset] using hOutInj
      by_cases hGroup : s.pids 0 % KV_GROUP_NUM = 0
      · simp [exec, rotary_embedding_k_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop, Tile.remap,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
            ComparableDType.eq, hTok, hHalf, hGroup,
            TileShape.insertAxis, TileShape.dropInsertedIndex,
            TileShape.dropInsertedIndex_two_pair,
            TileShape.dropInsertedIndex_zero_pair] at hExec
        rw [← hExec]
        simp only [kFullFirstOffset]
        rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
              (region := K)
              (P := fun j : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
                s.pids 1 * BLOCK_TOKENS + j.1.val < q_total_tokens)
              (offsetFn := fun j : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
                  (s.pids 1 * BLOCK_TOKENS + j.1.val) * k_token_stride +
                    (s.pids 0 / KV_GROUP_NUM) * k_head_stride +
                    (j.2.2.1.val + HEAD_HALF) * head_dim_stride)
              (hOff := fun k _ _ => by
                have := hOutDisjoint idx k
                simpa [kFullFirstOffset, kFullSecondOffset] using this)]
        rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj idx]
        by_cases hTokIdx : s.pids 1 * BLOCK_TOKENS + idx.1.val < q_total_tokens
        · simp [activeKFull, rotaryNopadK0FullSpec, kFullFirstOffset,
                kFullSecondOffset, cosFullOffset, hGroup, hTokIdx,
                Option.bind, Option.map]
        · simp [activeKFull, hGroup, hTokIdx]
      · simp [exec, rotary_embedding_k_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map, Tile.bop, IntegralDType.mod,
            ComparableDType.eq, hGroup] at hExec
        rw [← hExec]
        simp [activeKFull, hGroup]
    · exact False.elim (hHalf (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.2.2.1.isLt))
  · exact False.elim (hTok (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.1.isLt))

/-- Compute-facing correctness for the K first-half store in the full
`rotary_embedding_k_surface` kernel. -/
theorem rotary_embedding_k_surface_k0_compute_correct
    (K Cos Sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
          KV_GROUP_NUM BLOCK_TOKENS idx))
    (hOutDisjoint :
      ∀ idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
        kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS idx ≠
          kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx') :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rotary_embedding_k_surface K Cos Sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
          activeKFull s q_total_tokens KV_GROUP_NUM BLOCK_TOKENS idx)
        (fun idx => (K,
          kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS idx)))
      (expected := fun idx =>
        rotaryNopadK0FullSpec s K Cos Sin k_token_stride k_head_stride
          head_dim_stride cos_token_stride cos_stride KV_GROUP_NUM
          BLOCK_TOKENS HEAD_HALF idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_embedding_k_surface]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := rotary_embedding_k_surface_k0_correct K Cos Sin
    k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
    q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS s s' hOutInj
    hOutDisjoint hExec idx
  simpa [hActive] using h

/-- Algorithm-layer correctness for the K second-half store in the full
3D-tile K surface kernel `rotary_embedding_k_surface`. -/
theorem rotary_embedding_k_surface_k1_correct
    (K Cos Sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
          KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx))
    (hOutDisjoint :
      ∀ idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
        kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS idx ≠
          kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx')
    (hExec : exec (rotary_embedding_k_surface K Cos Sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS) s = some s') :
    ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      s'.readMem K
          (kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx) =
        if activeKFull s q_total_tokens KV_GROUP_NUM BLOCK_TOKENS idx then
          rotaryNopadK1FullSpec s K Cos Sin k_token_stride k_head_stride
            head_dim_stride cos_token_stride cos_stride KV_GROUP_NUM
            BLOCK_TOKENS HEAD_HALF idx
        else
          s.readMem K
            (kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
              KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx) := by
  intro idx
  by_cases hTok : 0 < BLOCK_TOKENS
  · by_cases hHalf : 0 < HEAD_HALF
    · have hRawInj : Function.Injective
          (fun j : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
            (s.pids 1 * BLOCK_TOKENS + j.1.val) * k_token_stride +
              (s.pids 0 / KV_GROUP_NUM) * k_head_stride +
              (j.2.2.1.val + HEAD_HALF) * head_dim_stride) := by
        simpa [kFullSecondOffset] using hOutInj
      by_cases hGroup : s.pids 0 % KV_GROUP_NUM = 0
      · simp [exec, rotary_embedding_k_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop, Tile.remap,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
            ComparableDType.eq, hTok, hHalf, hGroup,
            TileShape.insertAxis, TileShape.dropInsertedIndex,
            TileShape.dropInsertedIndex_two_pair,
            TileShape.dropInsertedIndex_zero_pair] at hExec
        rw [← hExec]
        simp only [kFullSecondOffset]
        rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj idx]
        by_cases hTokIdx : s.pids 1 * BLOCK_TOKENS + idx.1.val < q_total_tokens
        · simp [activeKFull, rotaryNopadK1FullSpec, kFullFirstOffset,
                kFullSecondOffset, cosFullOffset, hGroup, hTokIdx,
                Option.bind, Option.map]
        · rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
              (region := K)
              (P := fun j : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
                s.pids 1 * BLOCK_TOKENS + j.1.val < q_total_tokens)
              (offsetFn := fun j : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
                  (s.pids 1 * BLOCK_TOKENS + j.1.val) * k_token_stride +
                    (s.pids 0 / KV_GROUP_NUM) * k_head_stride +
                    j.2.2.1.val * head_dim_stride)
              (hOff := fun k _ _ => by
                have := hOutDisjoint k idx
                simpa [kFullFirstOffset, kFullSecondOffset] using this.symm)]
          simp [activeKFull, hGroup, hTokIdx]
      · simp [exec, rotary_embedding_k_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map, Tile.bop, IntegralDType.mod,
            ComparableDType.eq, hGroup] at hExec
        rw [← hExec]
        simp [activeKFull, hGroup]
    · exact False.elim (hHalf (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.2.2.1.isLt))
  · exact False.elim (hTok (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.1.isLt))

/-- Compute-facing correctness for the K second-half store in the full
`rotary_embedding_k_surface` kernel. -/
theorem rotary_embedding_k_surface_k1_compute_correct
    (K Cos Sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
          KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx))
    (hOutDisjoint :
      ∀ idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
        kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS idx ≠
          kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx') :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rotary_embedding_k_surface K Cos Sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
          activeKFull s q_total_tokens KV_GROUP_NUM BLOCK_TOKENS idx)
        (fun idx => (K,
          kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx)))
      (expected := fun idx =>
        rotaryNopadK1FullSpec s K Cos Sin k_token_stride k_head_stride
          head_dim_stride cos_token_stride cos_stride KV_GROUP_NUM
          BLOCK_TOKENS HEAD_HALF idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_embedding_k_surface]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := rotary_embedding_k_surface_k1_correct K Cos Sin
    k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
    q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS s s' hOutInj
    hOutDisjoint hExec idx
  simpa [hActive] using h

/-! ## The `⊨` specification surface (`GroupedMasked2DKernelIO`)

`rotary_embedding_kernel` is exposed as two grouped-masked Hoare triples, one
per data buffer. The Q face reads two Q half-windows plus `Cos`/`Sin` and writes
two Q half-windows in place (`nIn = 4`, `nOut = 2`); the K face is the GQA analog
whose two stores fire only on the group leader (`cur_head_idx % KV_GROUP_NUM = 0`),
that runtime predicate living entirely inside each channel's `writeMask`/`readMask`.

The kernels store a genuinely three-dimensional `[BLOCK_TOKENS, 1, HEAD_HALF]`
token/head/dim tile; the singleton head axis is inert, so the flat `⊨` lane space
`Fin (BLOCK_TOKENS * HEAD_HALF)` identifies with it row-major via the shared
`Lane2D` bridge (token = row, dim = column) lifted through the singleton with
`lane3`. -/

/-- Lift a flat lane `j` to the 3-D tile index `[BLOCK_TOKENS, 1, HEAD_HALF]`
via the row-major `Lane2D` decode (token = `j / HEAD_HALF`, dim = `j % HEAD_HALF`),
the inert head axis pinned to its sole inhabitant. -/
def lane3 (BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] :=
  ((Lane2D.decode j).1, ⟨0, Nat.one_pos⟩, (Lane2D.decode j).2.1, PUnit.unit)

/-- The rotary output of output channel `o` from the four loaded lane values
`(dataFirst, dataSecond, cos, sin) = (a, b, c, d)`: first half `a·c − b·d`,
second half `a·d + b·c`. Factored into one definition so both `⊨` faces and the
two headline conjuncts share a single matcher (a bare inline `match o` generates
a fresh per-site matcher that blocks the cross-declaration `exact`). -/
def rotaryPair (a b c d : ℝ) (o : Fin 2) : ℝ :=
  match o with
  | ⟨0, _⟩ => a * c - b * d
  | ⟨_ + 1, _⟩ => a * d + b * c

/-- Q/K first-half output/read address of flat lane `j` for program
`(pid₀, pid₁)`. -/
def dataFirstP (pid₀ pid₁ token_stride head_stride head_dim_stride
    BLOCK_TOKENS HEAD_HALF : Nat) (j : Fin (BLOCK_TOKENS * HEAD_HALF)) : Nat :=
  (pid₁ * BLOCK_TOKENS + (Lane2D.decode j).1.val) * token_stride +
    pid₀ * head_stride + (Lane2D.decode j).2.1.val * head_dim_stride

/-- Q/K second-half output/read address of flat lane `j`. -/
def dataSecondP (pid₀ pid₁ token_stride head_stride head_dim_stride
    BLOCK_TOKENS HEAD_HALF : Nat) (j : Fin (BLOCK_TOKENS * HEAD_HALF)) : Nat :=
  (pid₁ * BLOCK_TOKENS + (Lane2D.decode j).1.val) * token_stride +
    pid₀ * head_stride + ((Lane2D.decode j).2.1.val + HEAD_HALF) * head_dim_stride

/-- Cos/Sin read address of flat lane `j` (the trig tile broadcasts along the
inert head axis). -/
def cosP (pid₁ cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) : Nat :=
  (pid₁ * BLOCK_TOKENS + (Lane2D.decode j).1.val) * cos_token_stride +
    (Lane2D.decode j).2.1.val * cos_stride

/-- Token-in-range predicate of flat lane `j` (the `Cos`/`Sin` load mask). -/
def tokP (pid₁ q_total_tokens BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) : Prop :=
  pid₁ * BLOCK_TOKENS + (Lane2D.decode j).1.val < q_total_tokens

/-- Q active predicate of flat lane `j`: head in range and token in range. -/
def activeQP (pid₀ pid₁ q_total_tokens Q_HEAD_NUM BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) : Prop :=
  pid₀ < Q_HEAD_NUM ∧ pid₁ * BLOCK_TOKENS + (Lane2D.decode j).1.val < q_total_tokens

/-- K active predicate of flat lane `j`: the GQA-leader modular gate fires and
the token is in range. -/
def activeKP (pid₀ pid₁ q_total_tokens KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) : Prop :=
  pid₀ % KV_GROUP_NUM = 0 ∧
    pid₁ * BLOCK_TOKENS + (Lane2D.decode j).1.val < q_total_tokens

instance activeQPDecidable
    (pid₀ pid₁ q_total_tokens Q_HEAD_NUM BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) :
    Decidable (activeQP pid₀ pid₁ q_total_tokens Q_HEAD_NUM BLOCK_TOKENS HEAD_HALF j) := by
  unfold activeQP; infer_instance

instance activeKPDecidable
    (pid₀ pid₁ q_total_tokens KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) :
    Decidable (activeKP pid₀ pid₁ q_total_tokens KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF j) := by
  unfold activeKP; infer_instance

instance tokPDecidable (pid₁ q_total_tokens BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) :
    Decidable (tokP pid₁ q_total_tokens BLOCK_TOKENS HEAD_HALF j) := by
  unfold tokP; infer_instance

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for one masked store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (st : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      st).mem r o = st.mem r o := by
  induction l generalizing st with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

/-! ### Pid-indexed ↔ state-indexed bridges (`rfl`)

The grouped skin's windows/masks are functions of the program ids and the flat
lane; the sections above phrased everything over a `BlockState` and a 3-D
`TileIndex`. These record that the two agree definitionally under
`pid := s.pids` and `idx := lane3 j`. -/

theorem dataFirstP_eq (s : BlockState)
    (token_stride head_stride head_dim_stride BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) :
    dataFirstP (s.pids 0) (s.pids 1) token_stride head_stride head_dim_stride
        BLOCK_TOKENS HEAD_HALF j
      = qFullFirstOffset s token_stride head_stride head_dim_stride BLOCK_TOKENS
          (lane3 BLOCK_TOKENS HEAD_HALF j) := rfl

theorem dataSecondP_eq (s : BlockState)
    (token_stride head_stride head_dim_stride BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) :
    dataSecondP (s.pids 0) (s.pids 1) token_stride head_stride head_dim_stride
        BLOCK_TOKENS HEAD_HALF j
      = qFullSecondOffset s token_stride head_stride head_dim_stride BLOCK_TOKENS
          HEAD_HALF (lane3 BLOCK_TOKENS HEAD_HALF j) := rfl

theorem cosP_eq (s : BlockState) (cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) :
    cosP (s.pids 1) cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF j
      = cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS
          (lane3 BLOCK_TOKENS HEAD_HALF j) := rfl

theorem kFirstP_eq (s : BlockState)
    (k_token_stride k_head_stride head_dim_stride KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) :
    dataFirstP (s.pids 0 / KV_GROUP_NUM) (s.pids 1) k_token_stride k_head_stride
        head_dim_stride BLOCK_TOKENS HEAD_HALF j
      = kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
          KV_GROUP_NUM BLOCK_TOKENS (lane3 BLOCK_TOKENS HEAD_HALF j) := rfl

theorem kSecondP_eq (s : BlockState)
    (k_token_stride k_head_stride head_dim_stride KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) :
    dataSecondP (s.pids 0 / KV_GROUP_NUM) (s.pids 1) k_token_stride k_head_stride
        head_dim_stride BLOCK_TOKENS HEAD_HALF j
      = kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
          KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF (lane3 BLOCK_TOKENS HEAD_HALF j) := rfl

theorem activeQP_eq (s : BlockState)
    (q_total_tokens Q_HEAD_NUM BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) :
    activeQP (s.pids 0) (s.pids 1) q_total_tokens Q_HEAD_NUM BLOCK_TOKENS HEAD_HALF j
      = activeFull s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS
          (lane3 BLOCK_TOKENS HEAD_HALF j) := rfl

theorem activeKP_eq (s : BlockState)
    (q_total_tokens KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) :
    activeKP (s.pids 0) (s.pids 1) q_total_tokens KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF j
      = activeKFull s q_total_tokens KV_GROUP_NUM BLOCK_TOKENS
          (lane3 BLOCK_TOKENS HEAD_HALF j) := rfl

/-- The flat lane of a 3-D tile index (inverse of `lane3` up to the inert head
axis): the row-major encode of the `(token, dim)` projection. -/
def encode3 (BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Fin (BLOCK_TOKENS * HEAD_HALF) :=
  Lane2D.encode (idx.1, idx.2.2.1, PUnit.unit)

@[simp] theorem decode_encode3 (BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) :
    Lane2D.decode (encode3 BLOCK_TOKENS HEAD_HALF idx)
      = (idx.1, idx.2.2.1, PUnit.unit) := by
  simp [encode3, Lane2D.decode_encode]

theorem dataFirstP_encode3 (p q token_stride head_stride head_dim_stride
    BLOCK_TOKENS HEAD_HALF : Nat) (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) :
    dataFirstP p q token_stride head_stride head_dim_stride BLOCK_TOKENS HEAD_HALF
        (encode3 BLOCK_TOKENS HEAD_HALF idx)
      = (q * BLOCK_TOKENS + idx.1.val) * token_stride + p * head_stride +
          idx.2.2.1.val * head_dim_stride := by
  simp [dataFirstP]

theorem dataSecondP_encode3 (p q token_stride head_stride head_dim_stride
    BLOCK_TOKENS HEAD_HALF : Nat) (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) :
    dataSecondP p q token_stride head_stride head_dim_stride BLOCK_TOKENS HEAD_HALF
        (encode3 BLOCK_TOKENS HEAD_HALF idx)
      = (q * BLOCK_TOKENS + idx.1.val) * token_stride + p * head_stride +
          (idx.2.2.1.val + HEAD_HALF) * head_dim_stride := by
  simp [dataSecondP]

theorem cosP_encode3 (q cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) :
    cosP q cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF
        (encode3 BLOCK_TOKENS HEAD_HALF idx)
      = (q * BLOCK_TOKENS + idx.1.val) * cos_token_stride +
          idx.2.2.1.val * cos_stride := by
  simp [cosP]

theorem activeQP_encode3 (p q q_total_tokens Q_HEAD_NUM BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) :
    activeQP p q q_total_tokens Q_HEAD_NUM BLOCK_TOKENS HEAD_HALF
        (encode3 BLOCK_TOKENS HEAD_HALF idx)
      = (p < Q_HEAD_NUM ∧ q * BLOCK_TOKENS + idx.1.val < q_total_tokens) := by
  simp [activeQP]

theorem activeKP_encode3 (p q q_total_tokens KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) :
    activeKP p q q_total_tokens KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF
        (encode3 BLOCK_TOKENS HEAD_HALF idx)
      = (p % KV_GROUP_NUM = 0 ∧ q * BLOCK_TOKENS + idx.1.val < q_total_tokens) := by
  simp [activeKP]

theorem tokP_encode3 (q q_total_tokens BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) :
    tokP q q_total_tokens BLOCK_TOKENS HEAD_HALF (encode3 BLOCK_TOKENS HEAD_HALF idx)
      = (q * BLOCK_TOKENS + idx.1.val < q_total_tokens) := by
  simp [tokP]

/-! ### Q face -/

/-- Termination: the full Q rotary surface executes to completion from any state. -/
private theorem rotary_embedding_q_surface_exec_isSome
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat) (s : BlockState) :
    ∃ s1, exec ((rotary_embedding_q_surface Q Cos Sin q_token_stride q_head_stride
        head_dim_stride cos_token_stride cos_stride q_total_tokens Q_HEAD_NUM
        HEAD_HALF BLOCK_TOKENS).toAlgKernel) s = some s1 := by
  simp [exec, rotary_embedding_q_surface, ComputeKernel.toAlgKernel,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop, Tile.remap,
        Tile.expandDim, TileShape.insertAxis, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]

/-- The full Q rotary surface sits inside the flat-memory bridge's covered
fragment. -/
private theorem rotary_embedding_q_surface_flattenOk
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    ((rotary_embedding_q_surface Q Cos Sin q_token_stride q_head_stride
        head_dim_stride cos_token_stride cos_stride q_total_tokens Q_HEAD_NUM
        HEAD_HALF BLOCK_TOKENS).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [rotary_embedding_q_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Frame half: every memory cell outside the two masked Q output windows (the
first-half `off_q0` scatter and the second-half `off_q1` scatter) is preserved. -/
private theorem rotary_embedding_q_surface_frame
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat) (s s1 : BlockState)
    (hExec : exec ((rotary_embedding_q_surface Q Cos Sin q_token_stride q_head_stride
        head_dim_stride cos_token_stride cos_stride q_total_tokens Q_HEAD_NUM
        HEAD_HALF BLOCK_TOKENS).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmissFirst : ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      activeFull s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS idx →
      ¬(Q = r ∧ qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS idx = o))
    (hmissSecond : ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      activeFull s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS idx →
      ¬(Q = r ∧ qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS HEAD_HALF idx = o)) :
    s1.mem r o = s.mem r o := by
  simp only [activeFull, qFullFirstOffset, qFullSecondOffset] at hmissFirst hmissSecond
  simp [exec, rotary_embedding_q_surface, ComputeKernel.toAlgKernel,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop, Tile.remap,
        Tile.expandDim, TileShape.insertAxis, TileShape.dropInsertedIndex,
        TileShape.dropInsertedIndex_two_pair, TileShape.dropInsertedIndex_zero_pair,
        NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) ?_
  · intro k _ hmk hc
    exact hmissSecond k hmk hc
  · simp only [BlockState.setReg]
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
    intro k _ hmk hc
    exact hmissFirst k hmk hc

private theorem rotary_embedding_q_surface_traceSafe
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hcos : ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      s.pids 1 * BLOCK_TOKENS + idx.1.val < q_total_tokens →
      cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS idx < bounds Cos)
    (hsin : ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      s.pids 1 * BLOCK_TOKENS + idx.1.val < q_total_tokens →
      cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS idx < bounds Sin)
    (hq0 : ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      activeFull s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS idx →
      qFullFirstOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS idx
        < bounds Q)
    (hq1 : ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      activeFull s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS idx →
      qFullSecondOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
        HEAD_HALF idx < bounds Q) :
    Kernel.TraceSafe bounds
      ((rotary_embedding_q_surface Q Cos Sin q_token_stride q_head_stride
        head_dim_stride cos_token_stride cos_stride q_total_tokens Q_HEAD_NUM
        HEAD_HALF BLOCK_TOKENS).toAlgKernel) s := by
  simp only [activeFull, qFullFirstOffset, qFullSecondOffset, cosFullOffset] at hcos hsin hq0 hq1
  unfold Kernel.TraceSafe
  simp [rotary_embedding_q_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.remap, Tile.expandDim,
    TileShape.insertAxis, TileShape.dropInsertedIndex,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]
  exact ⟨fun a a_1 ht => hcos (a, ⟨0, Nat.one_pos⟩, a_1, PUnit.unit) ht,
    fun a a_1 ht => hsin (a, ⟨0, Nat.one_pos⟩, a_1, PUnit.unit) ht,
    fun a a_1 hh ht => hq0 (a, ⟨0, Nat.one_pos⟩, a_1, PUnit.unit) ⟨hh, ht⟩,
    fun a a_1 hh ht => hq1 (a, ⟨0, Nat.one_pos⟩, a_1, PUnit.unit) ⟨hh, ht⟩,
    fun a a_1 hh ht => hq0 (a, ⟨0, Nat.one_pos⟩, a_1, PUnit.unit) ⟨hh, ht⟩,
    fun a a_1 hh ht => hq1 (a, ⟨0, Nat.one_pos⟩, a_1, PUnit.unit) ⟨hh, ht⟩⟩

/-- The **IO signature** of the Q face of `rotary_embedding_kernel` — four read
windows (`q0`/`q1` half-tiles + `Cos`/`Sin`) and two in-place Q half-stores over
the flat `Fin (BLOCK_TOKENS * HEAD_HALF)` lane space. -/
def rotaryNopadQIO (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    GroupedMasked2DKernelIO where
  kernel := rotary_embedding_q_surface Q Cos Sin q_token_stride q_head_stride
    head_dim_stride cos_token_stride cos_stride q_total_tokens Q_HEAD_NUM HEAD_HALF
    BLOCK_TOKENS
  nIn := 4
  nOut := 2
  bufs := [Q, Cos, Sin]
  inp := fun i => match i with
    | ⟨0, _⟩ => Q | ⟨1, _⟩ => Q | ⟨2, _⟩ => Cos | ⟨_ + 3, _⟩ => Sin
  out := fun _ => Q
  B := BLOCK_TOKENS * HEAD_HALF
  read := fun i _pid₀ pid₁ j => match i with
    | ⟨0, _⟩ => dataFirstP _pid₀ pid₁ q_token_stride q_head_stride head_dim_stride
        BLOCK_TOKENS HEAD_HALF j
    | ⟨1, _⟩ => dataSecondP _pid₀ pid₁ q_token_stride q_head_stride head_dim_stride
        BLOCK_TOKENS HEAD_HALF j
    | ⟨2, _⟩ => cosP pid₁ cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF j
    | ⟨_ + 3, _⟩ => cosP pid₁ cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF j
  readMask := fun i pid₀ pid₁ j => match i with
    | ⟨0, _⟩ => activeQP pid₀ pid₁ q_total_tokens Q_HEAD_NUM BLOCK_TOKENS HEAD_HALF j
    | ⟨1, _⟩ => activeQP pid₀ pid₁ q_total_tokens Q_HEAD_NUM BLOCK_TOKENS HEAD_HALF j
    | ⟨2, _⟩ => tokP pid₁ q_total_tokens BLOCK_TOKENS HEAD_HALF j
    | ⟨_ + 3, _⟩ => tokP pid₁ q_total_tokens BLOCK_TOKENS HEAD_HALF j
  write := fun o pid₀ pid₁ j => match o with
    | ⟨0, _⟩ => dataFirstP pid₀ pid₁ q_token_stride q_head_stride head_dim_stride
        BLOCK_TOKENS HEAD_HALF j
    | ⟨_ + 1, _⟩ => dataSecondP pid₀ pid₁ q_token_stride q_head_stride head_dim_stride
        BLOCK_TOKENS HEAD_HALF j
  writeMask := fun o pid₀ pid₁ j => match o with
    | ⟨0, _⟩ => activeQP pid₀ pid₁ q_total_tokens Q_HEAD_NUM BLOCK_TOKENS HEAD_HALF j
    | ⟨_ + 1, _⟩ => activeQP pid₀ pid₁ q_total_tokens Q_HEAD_NUM BLOCK_TOKENS HEAD_HALF j

/-- **`rotaryNopadQIO ⊨ rotary`** — the Q face of `rotary_embedding_kernel` as one
grouped masked Hoare triple: every `activeFull`-active lane of the first Q half
ends up holding `q0·cos − q1·sin` and of the second half `q0·sin + q1·cos`,
computed from the **old** window contents, with every other flat cell untouched.
The three `∀ s` side conditions (first-half injective, second-half injective,
first-vs-second disjoint) are the honest layout collision-freedom the value legs
require; universal in the launch state because `⊨` ranges over every program. -/
theorem rotary_embedding_q_surface_implements
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (hInjFirst : ∀ s : BlockState, Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS idx))
    (hInjSecond : ∀ s : BlockState, Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        qFullSecondOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
          HEAD_HALF idx))
    (hDisjoint : ∀ (s : BlockState) (idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]),
      qFullFirstOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS idx ≠
        qFullSecondOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
          HEAD_HALF idx') :
    rotaryNopadQIO Q Cos Sin q_token_stride q_head_stride head_dim_stride
        cos_token_stride cos_stride q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS
      ⊨ fun _pid₀ _pid₁ xs o j =>
          rotaryPair (xs (⟨0, by decide⟩ : Fin 4) j) (xs (⟨1, by decide⟩ : Fin 4) j)
            (xs (⟨2, by decide⟩ : Fin 4) j) (xs (⟨3, by decide⟩ : Fin 4) j) o := by
  refine GroupedMasked2DKernelIO.Implements.intro _ ?_ ?_ ?_ ?_
  · intro o
    simp [rotaryNopadQIO]
  · exact rotary_embedding_q_surface_flattenOk Q Cos Sin q_token_stride q_head_stride
      head_dim_stride cos_token_stride cos_stride q_total_tokens Q_HEAD_NUM HEAD_HALF
      BLOCK_TOKENS
  · intro bounds s hib hob
    simp only [rotaryNopadQIO] at hib hob
    refine rotary_embedding_q_surface_traceSafe Q Cos Sin q_token_stride q_head_stride
      head_dim_stride cos_token_stride cos_stride q_total_tokens Q_HEAD_NUM HEAD_HALF
      BLOCK_TOKENS bounds s ?_ ?_ ?_ ?_
    · intro idx htok
      have h := hib (⟨2, by decide⟩ : Fin 4) (encode3 BLOCK_TOKENS HEAD_HALF idx)
        (by rw [tokP_encode3]; exact htok)
      rw [cosP_encode3] at h
      simpa [cosFullOffset] using h
    · intro idx htok
      have h := hib (⟨3, by decide⟩ : Fin 4) (encode3 BLOCK_TOKENS HEAD_HALF idx)
        (by rw [tokP_encode3]; exact htok)
      rw [cosP_encode3] at h
      simpa [cosFullOffset] using h
    · intro idx hact
      have h := hob (⟨0, by decide⟩ : Fin 2) (encode3 BLOCK_TOKENS HEAD_HALF idx)
        (by rw [activeQP_encode3]; exact hact)
      rw [dataFirstP_encode3] at h
      simpa [qFullFirstOffset] using h
    · intro idx hact
      have h := hob (⟨1, by decide⟩ : Fin 2) (encode3 BLOCK_TOKENS HEAD_HALF idx)
        (by rw [activeQP_encode3]; exact hact)
      rw [dataSecondP_encode3] at h
      simpa [qFullSecondOffset] using h
  · intro s₀ xs hx
    simp only [rotaryNopadQIO] at hx
    obtain ⟨s1, hs1⟩ := rotary_embedding_q_surface_exec_isSome Q Cos Sin q_token_stride
      q_head_stride head_dim_stride cos_token_stride cos_stride q_total_tokens Q_HEAD_NUM
      HEAD_HALF BLOCK_TOKENS s₀
    refine ⟨s1, hs1, ?_, ?_⟩
    · rintro ⟨o, ho⟩ j hj
      match o, ho with
      | 0, _ =>
          have hact : activeFull s₀ q_total_tokens Q_HEAD_NUM BLOCK_TOKENS
            (lane3 BLOCK_TOKENS HEAD_HALF j) := hj
          have htok : tokP (s₀.pids 1) q_total_tokens BLOCK_TOKENS HEAD_HALF j := hact.2
          have h := rotary_embedding_q_surface_q0_correct Q Cos Sin q_token_stride
            q_head_stride head_dim_stride cos_token_stride cos_stride q_total_tokens
            Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS s₀ s1 (hInjFirst s₀) (hDisjoint s₀) hs1
            (lane3 BLOCK_TOKENS HEAD_HALF j)
          rw [if_pos hact] at h
          show s1.readMem Q (dataFirstP (s₀.pids 0) (s₀.pids 1) q_token_stride
            q_head_stride head_dim_stride BLOCK_TOKENS HEAD_HALF j) = _
          rw [dataFirstP_eq, h]
          simp only [rotaryNopadQ0FullSpec, ← dataFirstP_eq, ← dataSecondP_eq, ← cosP_eq]
          rw [hx (⟨0, by decide⟩ : Fin 4) j hact, hx (⟨1, by decide⟩ : Fin 4) j hact,
            hx (⟨2, by decide⟩ : Fin 4) j htok, hx (⟨3, by decide⟩ : Fin 4) j htok]
          rfl
      | _ + 1, _ =>
          have hact : activeFull s₀ q_total_tokens Q_HEAD_NUM BLOCK_TOKENS
            (lane3 BLOCK_TOKENS HEAD_HALF j) := hj
          have htok : tokP (s₀.pids 1) q_total_tokens BLOCK_TOKENS HEAD_HALF j := hact.2
          have h := rotary_embedding_q_surface_q1_correct Q Cos Sin q_token_stride
            q_head_stride head_dim_stride cos_token_stride cos_stride q_total_tokens
            Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS s₀ s1 (hInjSecond s₀) (hDisjoint s₀) hs1
            (lane3 BLOCK_TOKENS HEAD_HALF j)
          rw [if_pos hact] at h
          show s1.readMem Q (dataSecondP (s₀.pids 0) (s₀.pids 1) q_token_stride
            q_head_stride head_dim_stride BLOCK_TOKENS HEAD_HALF j) = _
          rw [dataSecondP_eq, h]
          simp only [rotaryNopadQ1FullSpec, ← dataFirstP_eq, ← dataSecondP_eq, ← cosP_eq]
          rw [hx (⟨0, by decide⟩ : Fin 4) j hact, hx (⟨1, by decide⟩ : Fin 4) j hact,
            hx (⟨2, by decide⟩ : Fin 4) j htok, hx (⟨3, by decide⟩ : Fin 4) j htok]
          rfl
    · intro r o' hcond
      simp only [rotaryNopadQIO] at hcond
      refine rotary_embedding_q_surface_frame Q Cos Sin q_token_stride q_head_stride
        head_dim_stride cos_token_stride cos_stride q_total_tokens Q_HEAD_NUM HEAD_HALF
        BLOCK_TOKENS s₀ s1 hs1 r o' ?_ ?_
      · intro idx hact hc
        have h := hcond (⟨0, by decide⟩ : Fin 2) (encode3 BLOCK_TOKENS HEAD_HALF idx)
          (by rw [activeQP_encode3]; exact hact)
        rw [dataFirstP_encode3] at h
        exact h.elim (fun hne => hne hc.1.symm)
          (fun hne => hne (by rw [← qFullFirstOffset]; exact hc.2.symm))
      · intro idx hact hc
        have h := hcond (⟨1, by decide⟩ : Fin 2) (encode3 BLOCK_TOKENS HEAD_HALF idx)
          (by rw [activeQP_encode3]; exact hact)
        rw [dataSecondP_encode3] at h
        exact h.elim (fun hne => hne hc.1.symm)
          (fun hne => hne (by rw [← qFullSecondOffset]; exact hc.2.symm))

/-! ### K face

The GQA-leader predicate `cur_head_idx % KV_GROUP_NUM = 0` gates the whole store
body (`if handle_kv`), so every K support lemma below splits on it: on the group
leader the two half-stores fire (their in-branch masks are token-only, the head
gate having already been consumed by `handle_kv`), otherwise the kernel is a
register-only no-op leaving memory untouched. The K head index the stores address
is `cur_head_idx / KV_GROUP_NUM`, entering the windows as `dataFirstP (pid₀ /
KV_GROUP_NUM) …`. -/

/-- Termination: the full K rotary surface executes to completion from any state. -/
private theorem rotary_embedding_k_surface_exec_isSome
    (K Cos Sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS : Nat) (s : BlockState) :
    ∃ s1, exec ((rotary_embedding_k_surface K Cos Sin k_token_stride k_head_stride
        head_dim_stride cos_token_stride cos_stride q_total_tokens KV_GROUP_NUM
        HEAD_HALF BLOCK_TOKENS).toAlgKernel) s = some s1 := by
  by_cases hGroup : s.pids 0 % KV_GROUP_NUM = 0
  · simp [exec, rotary_embedding_k_surface, ComputeKernel.toAlgKernel,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop, Tile.remap,
          Tile.expandDim, TileShape.insertAxis, TileShape.dropInsertedIndex,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
          ComparableDType.eq, hGroup]
  · simp [exec, rotary_embedding_k_surface, ComputeKernel.toAlgKernel,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
          Tile.bop, IntegralDType.mod, ComparableDType.eq, hGroup]

/-- The full K rotary surface sits inside the flat-memory bridge's covered fragment. -/
private theorem rotary_embedding_k_surface_flattenOk
    (K Cos Sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    ((rotary_embedding_k_surface K Cos Sin k_token_stride k_head_stride
        head_dim_stride cos_token_stride cos_stride q_total_tokens KV_GROUP_NUM
        HEAD_HALF BLOCK_TOKENS).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [rotary_embedding_k_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Frame half for the K face: every cell outside the two masked K windows is
preserved (vacuously on a non-leader program, where nothing is stored). -/
private theorem rotary_embedding_k_surface_frame
    (K Cos Sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS : Nat) (s s1 : BlockState)
    (hExec : exec ((rotary_embedding_k_surface K Cos Sin k_token_stride k_head_stride
        head_dim_stride cos_token_stride cos_stride q_total_tokens KV_GROUP_NUM
        HEAD_HALF BLOCK_TOKENS).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmissFirst : ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      activeKFull s q_total_tokens KV_GROUP_NUM BLOCK_TOKENS idx →
      ¬(K = r ∧ kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
          KV_GROUP_NUM BLOCK_TOKENS idx = o))
    (hmissSecond : ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      activeKFull s q_total_tokens KV_GROUP_NUM BLOCK_TOKENS idx →
      ¬(K = r ∧ kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
          KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx = o)) :
    s1.mem r o = s.mem r o := by
  simp only [activeKFull, kFullFirstOffset, kFullSecondOffset] at hmissFirst hmissSecond
  by_cases hGroup : s.pids 0 % KV_GROUP_NUM = 0
  · simp [exec, rotary_embedding_k_surface, ComputeKernel.toAlgKernel,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop, Tile.remap,
          Tile.expandDim, TileShape.insertAxis, TileShape.dropInsertedIndex,
          TileShape.dropInsertedIndex_two_pair, TileShape.dropInsertedIndex_zero_pair,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
          ComparableDType.eq, hGroup] at hExec
    subst hExec
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) ?_
    · intro k _ hmk hc
      exact hmissSecond k ⟨hGroup, hmk⟩ hc
    · simp only [BlockState.setReg]
      refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
      intro k _ hmk hc
      exact hmissFirst k ⟨hGroup, hmk⟩ hc
  · simp [exec, rotary_embedding_k_surface, ComputeKernel.toAlgKernel,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
          Tile.bop, IntegralDType.mod, ComparableDType.eq, hGroup] at hExec
    subst hExec
    simp only [BlockState.setReg]

private theorem rotary_embedding_k_surface_traceSafe
    (K Cos Sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hcos : ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      activeKFull s q_total_tokens KV_GROUP_NUM BLOCK_TOKENS idx →
      cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS idx < bounds Cos)
    (hsin : ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      activeKFull s q_total_tokens KV_GROUP_NUM BLOCK_TOKENS idx →
      cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS idx < bounds Sin)
    (hk0 : ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      activeKFull s q_total_tokens KV_GROUP_NUM BLOCK_TOKENS idx →
      kFullFirstOffset s k_token_stride k_head_stride head_dim_stride KV_GROUP_NUM
        BLOCK_TOKENS idx < bounds K)
    (hk1 : ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      activeKFull s q_total_tokens KV_GROUP_NUM BLOCK_TOKENS idx →
      kFullSecondOffset s k_token_stride k_head_stride head_dim_stride KV_GROUP_NUM
        BLOCK_TOKENS HEAD_HALF idx < bounds K) :
    Kernel.TraceSafe bounds
      ((rotary_embedding_k_surface K Cos Sin k_token_stride k_head_stride
        head_dim_stride cos_token_stride cos_stride q_total_tokens KV_GROUP_NUM
        HEAD_HALF BLOCK_TOKENS).toAlgKernel) s := by
  simp only [activeKFull, kFullFirstOffset, kFullSecondOffset, cosFullOffset]
    at hcos hsin hk0 hk1
  unfold Kernel.TraceSafe
  by_cases hGroup : s.pids 0 % KV_GROUP_NUM = 0
  · simp [rotary_embedding_k_surface, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
      MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
      MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
      BlockState.setReg, tile_elementwise, Bool.and_eq_true,
      Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.remap, Tile.expandDim,
      TileShape.insertAxis, TileShape.dropInsertedIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
      ComparableDType.eq, hGroup]
    refine ⟨⟨fun a a_1 ht => hcos (a, ⟨0, Nat.one_pos⟩, a_1, PUnit.unit) ⟨hGroup, ht⟩,
      fun a a_1 ht => hsin (a, ⟨0, Nat.one_pos⟩, a_1, PUnit.unit) ⟨hGroup, ht⟩,
      fun a a_1 ht => hk0 (a, ⟨0, Nat.one_pos⟩, a_1, PUnit.unit) ⟨hGroup, ht⟩,
      fun a a_1 ht => hk1 (a, ⟨0, Nat.one_pos⟩, a_1, PUnit.unit) ⟨hGroup, ht⟩,
      fun a a_1 ht => hk0 (a, ⟨0, Nat.one_pos⟩, a_1, PUnit.unit) ⟨hGroup, ht⟩,
      fun a a_1 ht => hk1 (a, ⟨0, Nat.one_pos⟩, a_1, PUnit.unit) ⟨hGroup, ht⟩⟩, ?_⟩
    cases stepStmts _ _ <;> simp [Stmt.TraceSafeList]
  · simp [rotary_embedding_k_surface, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
      MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
      MaskOpt.Active, BlockState.setReg,
      Tile.bop, IntegralDType.mod, ComparableDType.eq, hGroup]

/-- The **IO signature** of the K face of `rotary_embedding_kernel` — four read
windows (`k0`/`k1` half-tiles + `Cos`/`Sin`) and two in-place K half-stores over
the flat `Fin (BLOCK_TOKENS * HEAD_HALF)` lane space, every channel gated by the
GQA-leader predicate `pid₀ % KV_GROUP_NUM = 0` and addressed at K head
`pid₀ / KV_GROUP_NUM`. -/
def rotaryNopadKIO (K Cos Sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    GroupedMasked2DKernelIO where
  kernel := rotary_embedding_k_surface K Cos Sin k_token_stride k_head_stride
    head_dim_stride cos_token_stride cos_stride q_total_tokens KV_GROUP_NUM HEAD_HALF
    BLOCK_TOKENS
  nIn := 4
  nOut := 2
  bufs := [K, Cos, Sin]
  inp := fun i => match i with
    | ⟨0, _⟩ => K | ⟨1, _⟩ => K | ⟨2, _⟩ => Cos | ⟨_ + 3, _⟩ => Sin
  out := fun _ => K
  B := BLOCK_TOKENS * HEAD_HALF
  read := fun i pid₀ pid₁ j => match i with
    | ⟨0, _⟩ => dataFirstP (pid₀ / KV_GROUP_NUM) pid₁ k_token_stride k_head_stride
        head_dim_stride BLOCK_TOKENS HEAD_HALF j
    | ⟨1, _⟩ => dataSecondP (pid₀ / KV_GROUP_NUM) pid₁ k_token_stride k_head_stride
        head_dim_stride BLOCK_TOKENS HEAD_HALF j
    | ⟨2, _⟩ => cosP pid₁ cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF j
    | ⟨_ + 3, _⟩ => cosP pid₁ cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF j
  readMask := fun i pid₀ pid₁ j => match i with
    | ⟨0, _⟩ => activeKP pid₀ pid₁ q_total_tokens KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF j
    | ⟨1, _⟩ => activeKP pid₀ pid₁ q_total_tokens KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF j
    | ⟨2, _⟩ => activeKP pid₀ pid₁ q_total_tokens KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF j
    | ⟨_ + 3, _⟩ => activeKP pid₀ pid₁ q_total_tokens KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF j
  write := fun o pid₀ pid₁ j => match o with
    | ⟨0, _⟩ => dataFirstP (pid₀ / KV_GROUP_NUM) pid₁ k_token_stride k_head_stride
        head_dim_stride BLOCK_TOKENS HEAD_HALF j
    | ⟨_ + 1, _⟩ => dataSecondP (pid₀ / KV_GROUP_NUM) pid₁ k_token_stride k_head_stride
        head_dim_stride BLOCK_TOKENS HEAD_HALF j
  writeMask := fun o pid₀ pid₁ j => match o with
    | ⟨0, _⟩ => activeKP pid₀ pid₁ q_total_tokens KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF j
    | ⟨_ + 1, _⟩ => activeKP pid₀ pid₁ q_total_tokens KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF j

/-- **`rotaryNopadKIO ⊨ rotary`** — the K face of `rotary_embedding_kernel` as one
grouped masked Hoare triple. On every GQA-leader lane (`pid₀ % KV_GROUP_NUM = 0`
and token in range) the two K halves end up holding `k0·cos − k1·sin` and
`k0·sin + k1·cos` from the old contents; on a non-leader program nothing is
written and the frame is vacuous. The three `∀ s` side conditions are the honest
layout collision-freedom the value legs require. -/
theorem rotary_embedding_k_surface_implements
    (K Cos Sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (hInjFirst : ∀ s : BlockState, Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        kFullFirstOffset s k_token_stride k_head_stride head_dim_stride KV_GROUP_NUM
          BLOCK_TOKENS idx))
    (hInjSecond : ∀ s : BlockState, Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        kFullSecondOffset s k_token_stride k_head_stride head_dim_stride KV_GROUP_NUM
          BLOCK_TOKENS HEAD_HALF idx))
    (hDisjoint : ∀ (s : BlockState) (idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]),
      kFullFirstOffset s k_token_stride k_head_stride head_dim_stride KV_GROUP_NUM
          BLOCK_TOKENS idx ≠
        kFullSecondOffset s k_token_stride k_head_stride head_dim_stride KV_GROUP_NUM
          BLOCK_TOKENS HEAD_HALF idx') :
    rotaryNopadKIO K Cos Sin k_token_stride k_head_stride head_dim_stride
        cos_token_stride cos_stride q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS
      ⊨ fun _pid₀ _pid₁ xs o j =>
          rotaryPair (xs (⟨0, by decide⟩ : Fin 4) j) (xs (⟨1, by decide⟩ : Fin 4) j)
            (xs (⟨2, by decide⟩ : Fin 4) j) (xs (⟨3, by decide⟩ : Fin 4) j) o := by
  refine GroupedMasked2DKernelIO.Implements.intro _ ?_ ?_ ?_ ?_
  · intro o
    simp [rotaryNopadKIO]
  · exact rotary_embedding_k_surface_flattenOk K Cos Sin k_token_stride k_head_stride
      head_dim_stride cos_token_stride cos_stride q_total_tokens KV_GROUP_NUM HEAD_HALF
      BLOCK_TOKENS
  · intro bounds s hib hob
    simp only [rotaryNopadKIO] at hib hob
    refine rotary_embedding_k_surface_traceSafe K Cos Sin k_token_stride k_head_stride
      head_dim_stride cos_token_stride cos_stride q_total_tokens KV_GROUP_NUM HEAD_HALF
      BLOCK_TOKENS bounds s ?_ ?_ ?_ ?_
    · intro idx hact
      have h := hib (⟨2, by decide⟩ : Fin 4) (encode3 BLOCK_TOKENS HEAD_HALF idx)
        (by rw [activeKP_encode3]; exact hact)
      rw [cosP_encode3] at h
      simpa [cosFullOffset] using h
    · intro idx hact
      have h := hib (⟨3, by decide⟩ : Fin 4) (encode3 BLOCK_TOKENS HEAD_HALF idx)
        (by rw [activeKP_encode3]; exact hact)
      rw [cosP_encode3] at h
      simpa [cosFullOffset] using h
    · intro idx hact
      have h := hob (⟨0, by decide⟩ : Fin 2) (encode3 BLOCK_TOKENS HEAD_HALF idx)
        (by rw [activeKP_encode3]; exact hact)
      rw [dataFirstP_encode3] at h
      simpa [kFullFirstOffset] using h
    · intro idx hact
      have h := hob (⟨1, by decide⟩ : Fin 2) (encode3 BLOCK_TOKENS HEAD_HALF idx)
        (by rw [activeKP_encode3]; exact hact)
      rw [dataSecondP_encode3] at h
      simpa [kFullSecondOffset] using h
  · intro s₀ xs hx
    simp only [rotaryNopadKIO] at hx
    obtain ⟨s1, hs1⟩ := rotary_embedding_k_surface_exec_isSome K Cos Sin k_token_stride
      k_head_stride head_dim_stride cos_token_stride cos_stride q_total_tokens KV_GROUP_NUM
      HEAD_HALF BLOCK_TOKENS s₀
    refine ⟨s1, hs1, ?_, ?_⟩
    · rintro ⟨o, ho⟩ j hj
      match o, ho with
      | 0, _ =>
          have hact : activeKFull s₀ q_total_tokens KV_GROUP_NUM BLOCK_TOKENS
            (lane3 BLOCK_TOKENS HEAD_HALF j) := hj
          have h := rotary_embedding_k_surface_k0_correct K Cos Sin k_token_stride
            k_head_stride head_dim_stride cos_token_stride cos_stride q_total_tokens
            KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS s₀ s1 (hInjFirst s₀) (hDisjoint s₀) hs1
            (lane3 BLOCK_TOKENS HEAD_HALF j)
          rw [if_pos hact] at h
          show s1.readMem K (dataFirstP (s₀.pids 0 / KV_GROUP_NUM) (s₀.pids 1)
            k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS HEAD_HALF j) = _
          rw [kFirstP_eq, h]
          simp only [rotaryNopadK0FullSpec, ← kFirstP_eq, ← kSecondP_eq, ← cosP_eq]
          rw [hx (⟨0, by decide⟩ : Fin 4) j hact, hx (⟨1, by decide⟩ : Fin 4) j hact,
            hx (⟨2, by decide⟩ : Fin 4) j hact, hx (⟨3, by decide⟩ : Fin 4) j hact]
          rfl
      | _ + 1, _ =>
          have hact : activeKFull s₀ q_total_tokens KV_GROUP_NUM BLOCK_TOKENS
            (lane3 BLOCK_TOKENS HEAD_HALF j) := hj
          have h := rotary_embedding_k_surface_k1_correct K Cos Sin k_token_stride
            k_head_stride head_dim_stride cos_token_stride cos_stride q_total_tokens
            KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS s₀ s1 (hInjSecond s₀) (hDisjoint s₀) hs1
            (lane3 BLOCK_TOKENS HEAD_HALF j)
          rw [if_pos hact] at h
          show s1.readMem K (dataSecondP (s₀.pids 0 / KV_GROUP_NUM) (s₀.pids 1)
            k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS HEAD_HALF j) = _
          rw [kSecondP_eq, h]
          simp only [rotaryNopadK1FullSpec, ← kFirstP_eq, ← kSecondP_eq, ← cosP_eq]
          rw [hx (⟨0, by decide⟩ : Fin 4) j hact, hx (⟨1, by decide⟩ : Fin 4) j hact,
            hx (⟨2, by decide⟩ : Fin 4) j hact, hx (⟨3, by decide⟩ : Fin 4) j hact]
          rfl
    · intro r o' hcond
      simp only [rotaryNopadKIO] at hcond
      refine rotary_embedding_k_surface_frame K Cos Sin k_token_stride k_head_stride
        head_dim_stride cos_token_stride cos_stride q_total_tokens KV_GROUP_NUM HEAD_HALF
        BLOCK_TOKENS s₀ s1 hs1 r o' ?_ ?_
      · intro idx hact hc
        have h := hcond (⟨0, by decide⟩ : Fin 2) (encode3 BLOCK_TOKENS HEAD_HALF idx)
          (by rw [activeKP_encode3]; exact hact)
        rw [dataFirstP_encode3] at h
        exact h.elim (fun hne => hne hc.1.symm)
          (fun hne => hne (by rw [← kFullFirstOffset]; exact hc.2.symm))
      · intro idx hact hc
        have h := hcond (⟨1, by decide⟩ : Fin 2) (encode3 BLOCK_TOKENS HEAD_HALF idx)
          (by rw [activeKP_encode3]; exact hact)
        rw [dataSecondP_encode3] at h
        exact h.elim (fun hne => hne hc.1.symm)
          (fun hne => hne (by rw [← kFullSecondOffset]; exact hc.2.symm))

/-- **Dimension-general public output summary for `rotary_emb_nopad.py`**
(genuine, not self-referential), on the `⊨` (`GroupedMasked2DKernelIO`) surface.

Every token count, head count, KV-group count, head-dim half, block size, and
stride is a `Nat` parameter rather than a pinned Python literal. The per-lane
output-offset injectivity / first-vs-second-half disjointness side-conditions are
taken as hypotheses, now universal over the launch state because the `⊨` face
ranges over every program (`(pid₀, pid₁)`) rather than a single initial state.

For ANY shape, the full `rotary_embedding_kernel_surface` (both Q stores plus the
conditional GQA-leader K stores) lowers to the algorithm layer, and each data
buffer implements its grouped masked rotary Hoare triple: `rotaryNopadQIO` writes
`q0·cos − q1·sin` / `q0·sin + q1·cos` on every `activeFull` Q lane, and
`rotaryNopadKIO` the K analogues on every GQA-leader lane
(`cur_head_idx % KV_GROUP_NUM = 0`, that runtime predicate living entirely inside
each channel's `writeMask`) — both from the **old** window contents, the actual
embedding read from the precomputed `cos`/`sin` cache, with every other flat cell
untouched.

The host launch remains the trusted boundary. -/
specification rotary_emb_nopad_output_summary_general
    (Q K Cos Sin : RegionName)
    (surf_q_token_stride surf_q_head_stride surf_k_token_stride surf_k_head_stride
      surf_head_dim_stride surf_cos_token_stride surf_cos_stride
      surf_q_total_tokens surf_Q_HEAD_NUM surf_KV_GROUP_NUM surf_HEAD_DIM
      surf_BLOCK_TOKENS : Nat)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (k_token_stride k_head_stride k_q_total_tokens KV_GROUP_NUM : Nat)
    (hQInjFirst : ∀ s : BlockState, Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS idx))
    (hQInjSecond : ∀ s : BlockState, Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS HEAD_HALF idx))
    (hQDisjoint : ∀ (s : BlockState)
        (idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]),
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS idx ≠
          qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF idx')
    (hKInjFirst : ∀ s : BlockState, Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
          KV_GROUP_NUM BLOCK_TOKENS idx))
    (hKInjSecond : ∀ s : BlockState, Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
          KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx))
    (hKDisjoint : ∀ (s : BlockState)
        (idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]),
        kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS idx ≠
          kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx') :
    (∃ alg, (rotary_embedding_kernel_surface Q K Cos Sin
      surf_q_token_stride surf_q_head_stride surf_k_token_stride
      surf_k_head_stride surf_head_dim_stride surf_cos_token_stride
      surf_cos_stride surf_q_total_tokens surf_Q_HEAD_NUM surf_KV_GROUP_NUM
      surf_HEAD_DIM surf_BLOCK_TOKENS).toAlgorithm? = Except.ok alg) ∧
    (rotaryNopadQIO Q Cos Sin q_token_stride q_head_stride head_dim_stride
        cos_token_stride cos_stride q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS
      ⊨ fun _pid₀ _pid₁ xs o j =>
          rotaryPair (xs (⟨0, by decide⟩ : Fin 4) j) (xs (⟨1, by decide⟩ : Fin 4) j)
            (xs (⟨2, by decide⟩ : Fin 4) j) (xs (⟨3, by decide⟩ : Fin 4) j) o) ∧
    (rotaryNopadKIO K Cos Sin k_token_stride k_head_stride head_dim_stride
        cos_token_stride cos_stride k_q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS
      ⊨ fun _pid₀ _pid₁ xs o j =>
          rotaryPair (xs (⟨0, by decide⟩ : Fin 4) j) (xs (⟨1, by decide⟩ : Fin 4) j)
            (xs (⟨2, by decide⟩ : Fin 4) j) (xs (⟨3, by decide⟩ : Fin 4) j) o) := by
  refine ⟨rotary_embedding_kernel_surface_toAlgorithm_supported Q K Cos Sin
      surf_q_token_stride surf_q_head_stride surf_k_token_stride
      surf_k_head_stride surf_head_dim_stride surf_cos_token_stride
      surf_cos_stride surf_q_total_tokens surf_Q_HEAD_NUM surf_KV_GROUP_NUM
      surf_HEAD_DIM surf_BLOCK_TOKENS, ?_, ?_⟩
  · exact rotary_embedding_q_surface_implements Q Cos Sin
      q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS hQInjFirst hQInjSecond hQDisjoint
  · exact rotary_embedding_k_surface_implements K Cos Sin
      k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      k_q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS hKInjFirst hKInjSecond hKDisjoint

end VeriTile.Bench.TritonBenchG.RotaryEmbNopad
