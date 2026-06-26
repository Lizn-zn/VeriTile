import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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
rotary_nopad_python_case1_output_summary              ← TOP (abbrev alias)
  = rotary_nopad_python_case1_all_outputs_surface_summary
      ├─ rotary_nopad_python_case1_surface_toAlgorithm_supported
      │     └─ rotary_embedding_kernel_surface_toAlgorithm_supported
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

Offset injectivity/disjointness is supplied by
`rotary_nopad_python_q_first/second_offset_injective`,
`rotary_nopad_python_q_offsets_disjoint`, and the matching `k_*` lemmas.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; dtype casts erase to the
identity at the algorithm layer (post-erasure all dtypes unify to `ℝ`).
`cos`/`sin` are modeled as **precomputed inputs** loaded from memory, not
computed. Scoping is **one half-store at a time** (q0/q1/k0/k1), each over the
active lanes (`tokens_range < q_total_tokens`, `cur_head_idx < Q_HEAD_NUM`, and
for K the GQA-leader predicate); out-of-bounds lanes are preserved verbatim. The
top summary covers the no-cache `case1` Python shape (contiguous
`(32, 8, 64)` Q / `(32, 4, 64)` K, `HEAD_DIM = 64`, `BLOCK_TOKENS = 4`,
`KV_GROUP_NUM = 2`). The `fused_rotary_embedding_kernel_v2` KV-cache/Q track is verified
at the per-store-slice level (including the context paged-cache offsets) but is
**not folded into a single top-level v2 summary** here — honestly, only `case1`
has a public `output_summary`. `@triton.autotune` is not modeled.
-/

namespace VeriTile.Bench.TritonBenchG.RotaryEmbNopad

open VeriTile.Triton

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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
      (kernel := fused_rotary_v2_kv_cache_first_half_store_slice OutK0Pre
        kv_cache block_ids offsets_in_last_block cacheb_stride cacheh_stride
        cached_stride HEAD_HALF)
      (initialState := s)
      (write := fun i : Fin HEAD_HALF => some (kv_cache,
        kvCacheFirstOffset s block_ids offsets_in_last_block cacheb_stride
          cacheh_stride cached_stride i))
      (expected := fun i => kvCacheFirstStoreSpec s OutK0Pre i) := by
  unfold ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
      (kernel := fused_rotary_v2_kv_cache_second_half_store_slice OutK1Pre
        kv_cache block_ids offsets_in_last_block cacheb_stride cacheh_stride
        cached_stride HEAD_HALF)
      (initialState := s)
      (write := fun i : Fin HEAD_HALF => some (kv_cache,
        kvCacheSecondOffset s block_ids offsets_in_last_block cacheb_stride
          cacheh_stride cached_stride HEAD_HALF i))
      (expected := fun i => kvCacheSecondStoreSpec s OutK1Pre HEAD_HALF i) := by
  unfold ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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

/-! ## Python test-shape wrappers -/

theorem rotary_nopad_python_q_first_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [4, 1, 32] =>
        qFullFirstOffset s 512 64 1 4 idx) := by
  intro a b h
  simp [qFullFirstOffset] at h
  ext <;> omega

theorem rotary_nopad_python_q_second_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [4, 1, 32] =>
        qFullSecondOffset s 512 64 1 4 32 idx) := by
  intro a b h
  simp [qFullSecondOffset] at h
  ext <;> omega

theorem rotary_nopad_python_q_offsets_disjoint
    (s : BlockState) :
    ∀ idx idx' : TileIndex [4, 1, 32],
      qFullFirstOffset s 512 64 1 4 idx ≠
        qFullSecondOffset s 512 64 1 4 32 idx' := by
  intro idx idx' h
  simp [qFullFirstOffset, qFullSecondOffset] at h
  omega

theorem rotary_nopad_python_k_first_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [4, 1, 32] =>
        kFullFirstOffset s 256 64 1 2 4 idx) := by
  intro a b h
  simp [kFullFirstOffset] at h
  ext <;> omega

theorem rotary_nopad_python_k_second_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [4, 1, 32] =>
        kFullSecondOffset s 256 64 1 2 4 32 idx) := by
  intro a b h
  simp [kFullSecondOffset] at h
  ext <;> omega

theorem rotary_nopad_python_k_offsets_disjoint
    (s : BlockState) :
    ∀ idx idx' : TileIndex [4, 1, 32],
      kFullFirstOffset s 256 64 1 2 4 idx ≠
        kFullSecondOffset s 256 64 1 2 4 32 idx' := by
  intro idx idx' h
  simp [kFullFirstOffset, kFullSecondOffset] at h
  omega

/-- Python no-cache case full surface lowering for `total_tokens = 32`,
`q_head_num = 8`, `kv_head_num = 4`, and `head_dim = 64`. -/
theorem rotary_nopad_python_case1_surface_toAlgorithm_supported
    (Q K Cos Sin : RegionName) :
    ∃ alg, (rotary_embedding_kernel_surface Q K Cos Sin
      512 64 256 64 1 64 1 32 8 2 64 4).toAlgorithm? =
        Except.ok alg := by
  exact rotary_embedding_kernel_surface_toAlgorithm_supported Q K Cos Sin
    512 64 256 64 1 64 1 32 8 2 64 4

/-- Public Python no-cache case coverage summary: the full Q/K rotary surface
lowers, and all four Q/K half-output stores realize the checked tensor strides. -/
theorem rotary_nopad_python_case1_all_outputs_surface_summary
    (Q K Cos Sin : RegionName) (s : BlockState) :
    (∃ alg, (rotary_embedding_kernel_surface Q K Cos Sin
      512 64 256 64 1 64 1 32 8 2 64 4).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_embedding_q_surface Q Cos Sin
        512 64 1 64 1 32 8 32 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [4, 1, 32] => activeFull s 32 8 4 idx)
        (fun idx => (Q, qFullFirstOffset s 512 64 1 4 idx)))
      (expected := fun idx =>
        rotaryNopadQ0FullSpec s Q Cos Sin 512 64 1 64 1 4 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_embedding_q_surface Q Cos Sin
        512 64 1 64 1 32 8 32 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [4, 1, 32] => activeFull s 32 8 4 idx)
        (fun idx => (Q, qFullSecondOffset s 512 64 1 4 32 idx)))
      (expected := fun idx =>
        rotaryNopadQ1FullSpec s Q Cos Sin 512 64 1 64 1 4 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_embedding_k_surface K Cos Sin
        256 64 1 64 1 32 2 32 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [4, 1, 32] => activeKFull s 32 2 4 idx)
        (fun idx => (K, kFullFirstOffset s 256 64 1 2 4 idx)))
      (expected := fun idx =>
        rotaryNopadK0FullSpec s K Cos Sin 256 64 1 64 1 2 4 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_embedding_k_surface K Cos Sin
        256 64 1 64 1 32 2 32 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [4, 1, 32] => activeKFull s 32 2 4 idx)
        (fun idx => (K, kFullSecondOffset s 256 64 1 2 4 32 idx)))
      (expected := fun idx =>
        rotaryNopadK1FullSpec s K Cos Sin 256 64 1 64 1 2 4 32 idx)) := by
  constructor
  · exact rotary_nopad_python_case1_surface_toAlgorithm_supported Q K Cos Sin
  constructor
  · exact rotary_embedding_q_surface_q0_compute_correct Q Cos Sin
      512 64 1 64 1 32 8 32 4 s
      (rotary_nopad_python_q_first_offset_injective s)
      (rotary_nopad_python_q_offsets_disjoint s)
  constructor
  · exact rotary_embedding_q_surface_q1_compute_correct Q Cos Sin
      512 64 1 64 1 32 8 32 4 s
      (rotary_nopad_python_q_second_offset_injective s)
      (rotary_nopad_python_q_offsets_disjoint s)
  constructor
  · exact rotary_embedding_k_surface_k0_compute_correct K Cos Sin
      256 64 1 64 1 32 2 32 4 s
      (rotary_nopad_python_k_first_offset_injective s)
      (rotary_nopad_python_k_offsets_disjoint s)
  · exact rotary_embedding_k_surface_k1_compute_correct K Cos Sin
      256 64 1 64 1 32 2 32 4 s
      (rotary_nopad_python_k_second_offset_injective s)
      (rotary_nopad_python_k_offsets_disjoint s)

/-- `output_summary` alias for the Python no-cache rotary embedding path. -/
abbrev rotary_nopad_python_case1_output_summary
    (Q K Cos Sin : RegionName) (s : BlockState) :=
  rotary_nopad_python_case1_all_outputs_surface_summary Q K Cos Sin s

/-- **Dimension-general public output summary for `rotary_emb_nopad.py`**
(genuine, not self-referential).

Symbolic-dimension companion of
`rotary_nopad_python_case1_all_outputs_surface_summary`: every token count,
head count, KV-group count, head-dim half, block size, and stride is a `Nat`
parameter rather than a pinned Python literal, and the per-lane
output-offset injectivity / first-vs-second-half disjointness side-conditions
are taken as hypotheses (the concrete-shape variants
`rotary_nopad_python_q/k_*_offset_injective` / `_offsets_disjoint` discharge
them at the Python case-1 shape).

For ANY shape, the full `rotary_embedding_kernel_surface` (both Q stores plus
the conditional GQA-leader K stores) lowers to the algorithm layer, and all
four half-output stores realize the genuine rotary closed forms: Q first half
`q0·cos − q1·sin` (`rotaryNopadQ0FullSpec`), Q second half `q0·sin + q1·cos`
(`rotaryNopadQ1FullSpec`), and the K analogues
(`rotaryNopadK0FullSpec`/`rotaryNopadK1FullSpec`) over the active GQA-leader
lanes — the actual embedding read from the precomputed `cos`/`sin` cache, NOT
the kernel's own re-executed value.

The host launch remains the trusted boundary. -/
theorem rotary_emb_nopad_output_summary_general
    (Q K Cos Sin : RegionName) (s : BlockState)
    (surf_q_token_stride surf_q_head_stride surf_k_token_stride surf_k_head_stride
      surf_head_dim_stride surf_cos_token_stride surf_cos_stride
      surf_q_total_tokens surf_Q_HEAD_NUM surf_KV_GROUP_NUM surf_HEAD_DIM
      surf_BLOCK_TOKENS : Nat)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (k_token_stride k_head_stride k_q_total_tokens KV_GROUP_NUM : Nat)
    (hQ0Inj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS idx))
    (hQ1Inj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS HEAD_HALF idx))
    (hQDisjoint :
      ∀ idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS idx ≠
          qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF idx')
    (hK0Inj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
          KV_GROUP_NUM BLOCK_TOKENS idx))
    (hK1Inj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
          KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx))
    (hKDisjoint :
      ∀ idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
        kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS idx ≠
          kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx') :
    (∃ alg, (rotary_embedding_kernel_surface Q K Cos Sin
      surf_q_token_stride surf_q_head_stride surf_k_token_stride
      surf_k_head_stride surf_head_dim_stride surf_cos_token_stride
      surf_cos_stride surf_q_total_tokens surf_Q_HEAD_NUM surf_KV_GROUP_NUM
      surf_HEAD_DIM surf_BLOCK_TOKENS).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_embedding_q_surface Q Cos Sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
          activeFull s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS idx)
        (fun idx => (Q, qFullFirstOffset s q_token_stride q_head_stride
          head_dim_stride BLOCK_TOKENS idx)))
      (expected := fun idx =>
        rotaryNopadQ0FullSpec s Q Cos Sin q_token_stride q_head_stride
          head_dim_stride cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_embedding_q_surface Q Cos Sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
          activeFull s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS idx)
        (fun idx => (Q, qFullSecondOffset s q_token_stride q_head_stride
          head_dim_stride BLOCK_TOKENS HEAD_HALF idx)))
      (expected := fun idx =>
        rotaryNopadQ1FullSpec s Q Cos Sin q_token_stride q_head_stride
          head_dim_stride cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_embedding_k_surface K Cos Sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
        k_q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
          activeKFull s k_q_total_tokens KV_GROUP_NUM BLOCK_TOKENS idx)
        (fun idx => (K, kFullFirstOffset s k_token_stride k_head_stride
          head_dim_stride KV_GROUP_NUM BLOCK_TOKENS idx)))
      (expected := fun idx =>
        rotaryNopadK0FullSpec s K Cos Sin k_token_stride k_head_stride
          head_dim_stride cos_token_stride cos_stride KV_GROUP_NUM
          BLOCK_TOKENS HEAD_HALF idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := rotary_embedding_k_surface K Cos Sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
        k_q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
          activeKFull s k_q_total_tokens KV_GROUP_NUM BLOCK_TOKENS idx)
        (fun idx => (K, kFullSecondOffset s k_token_stride k_head_stride
          head_dim_stride KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx)))
      (expected := fun idx =>
        rotaryNopadK1FullSpec s K Cos Sin k_token_stride k_head_stride
          head_dim_stride cos_token_stride cos_stride KV_GROUP_NUM
          BLOCK_TOKENS HEAD_HALF idx)) := by
  refine ⟨rotary_embedding_kernel_surface_toAlgorithm_supported Q K Cos Sin
      surf_q_token_stride surf_q_head_stride surf_k_token_stride
      surf_k_head_stride surf_head_dim_stride surf_cos_token_stride
      surf_cos_stride surf_q_total_tokens surf_Q_HEAD_NUM surf_KV_GROUP_NUM
      surf_HEAD_DIM surf_BLOCK_TOKENS, ?_, ?_, ?_, ?_⟩
  · exact rotary_embedding_q_surface_q0_compute_correct Q Cos Sin
      q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS s hQ0Inj hQDisjoint
  · exact rotary_embedding_q_surface_q1_compute_correct Q Cos Sin
      q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS s hQ1Inj hQDisjoint
  · exact rotary_embedding_k_surface_k0_compute_correct K Cos Sin
      k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      k_q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS s hK0Inj hKDisjoint
  · exact rotary_embedding_k_surface_k1_compute_correct K Cos Sin
      k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      k_q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS s hK1Inj hKDisjoint

end VeriTile.Bench.TritonBenchG.RotaryEmbNopad
