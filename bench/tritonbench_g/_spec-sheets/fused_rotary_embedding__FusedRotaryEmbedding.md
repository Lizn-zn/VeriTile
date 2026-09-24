# Spec sheet — `bench/tritonbench_g/fused_rotary_embedding/FusedRotaryEmbedding.lean`

**Python source:** `bench/tritonbench_g/fused_rotary_embedding/fused_rotary_embedding.py`

## Public theorem: `decoding_fused_rotary_embedding_q_correctness`

<details><summary>docstring</summary>

```
/-- **`decodingRotaryQIO ⊨ rotary`** — the unconditional Q rotary face of
`decoding_fused_rotary_embedding_kernel` as one Hoare triple over flat pointer
memory: for every disjoint base-pointer placement of `q`/`cos`/`sin`, every
program `(pid₀, pid₁)` whose windows are in bounds, and every launch state
whose four input windows hold `q0`, `q1`, `cos`, `sin`, the translated pointer
kernel terminates, the Q first half ends up holding `q0 · cos - q1 · sin`, the
Q second half `q0 · sin + q1 · cos`, and every flat cell outside the two Q
half-windows is untouched.

Dimension-general: the rotary half width `HALF_DIM` and all four strides are
free `Nat` parameters. The single side condition `0 < head_dim_stride` is what
makes the two in-place half-windows disjoint (and each of them injective) —
without it the two stores would alias and no closed form could hold. -/
```
</details>

**Statement:**
```lean
specification decoding_fused_rotary_embedding_q_correctness
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (hstride : 0 < head_dim_stride) :
    decodingRotaryQIO q cos sin q_token_stride q_head_stride head_dim_stride
        cos_token_stride cos_stride HALF_DIM ⊨
      fun _pid₀ _pid₁ xs o j =>
        let q0 := xs (⟨0, by decide⟩ : Fin 4) j
        let q1 := xs (⟨1, by decide⟩ : Fin 4) j
        let c := xs (⟨2, by decide⟩ : Fin 4) j
        let sn := xs (⟨3, by decide⟩ : Fin 4) j
        match o with
        | ⟨0, _⟩ => q0 * c - q1 * sn
        | ⟨_ + 1, _⟩ => q0 * sn + q1 * c
```

**Assumptions / layout contracts:**
- `hstride : 0 < head_dim_stride`

**Closed-form spec defs (transitive):** `decodingRotaryQIO`, `decoding_fused_rotary_embedding_q_surface`, `qFirstAddr`, `qSecondAddr`, `cosSinAddr`

<details><summary><code>decodingRotaryQIO</code></summary>

```
/-- The Q rotary surface's **IO signature** — the whole kernel-specific audit
surface of the `⊨` headline: which buffer is which channel, where program
`(pid₀, pid₁)` reads and writes, and the (here total) activity predicates.

Four input channels — `0` the Q first half, `1` the Q second half, `2` `cos`,
`3` `sin` — and two output channels, `0` the Q first half and `1` the Q second
half. Both output channels name the **same** `q` buffer as input channels `0`
and `1`: the rotation is in place, which is why `bufs` (the allocation list,
`[q, cos, sin]`) is decoupled from the channel tables. Every lane is active:
the Python kernel carries no mask on this face. -/
```
```lean
def decodingRotaryQIO (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat) : GroupedMasked2DKernelIO where
  kernel := decoding_fused_rotary_embedding_q_surface q cos sin q_token_stride
    q_head_stride head_dim_stride cos_token_stride cos_stride (HALF_DIM * 2)
    HALF_DIM
  nIn := 4
  nOut := 2
  bufs := [q, cos, sin]
  inp := fun
    | ⟨0, _⟩ => q
    | ⟨1, _⟩ => q
    | ⟨2, _⟩ => cos
    | ⟨_ + 3, _⟩ => sin
  out := fun
    | ⟨0, _⟩ => q
    | ⟨_ + 1, _⟩ => q
  B := HALF_DIM
  read := fun
    | ⟨0, _⟩ => fun p₀ p₁ j =>
        qFirstAddr p₀ p₁ q_token_stride q_head_stride head_dim_stride HALF_DIM j
    | ⟨1, _⟩ => fun p₀ p₁ j =>
        qSecondAddr p₀ p₁ q_token_stride q_head_stride head_dim_stride HALF_DIM j
    | ⟨2, _⟩ => fun _p₀ p₁ j =>
        cosSinAddr p₁ cos_token_stride cos_stride HALF_DIM j
    | ⟨_ + 3, _⟩ => fun _p₀ p₁ j =>
        cosSinAddr p₁ cos_token_stride cos_stride HALF_DIM j
  readMask := fun _ _ _ _ => True
  write := fun
    | ⟨0, _⟩ => fun p₀ p₁ j =>
        qFirstAddr p₀ p₁ q_token_stride q_head_stride head_dim_stride HALF_DIM j
    | ⟨_ + 1, _⟩ => fun p₀ p₁ j =>
        qSecondAddr p₀ p₁ q_token_stride q_head_stride head_dim_stride HALF_DIM j
  writeMask := fun _ _ _ _ => True
```
</details>

<details><summary><code>decoding_fused_rotary_embedding_q_surface</code></summary>

```
/-- Surface transcription of the unconditional Q rotary part of
`fused_rotary_embedding.py`'s `decoding_fused_rotary_embedding_kernel`.

The full Python kernel also conditionally rotates K and fills K/V caches when
`cur_head_idx % KV_GROUP_NUM == 0`. That branch depends on context-length
metadata and cache block tables, so this surface covers the unconditional Q
updates that every program instance performs: both the first and second rotary
halves are written. -/
```
```lean
def decoding_fused_rotary_embedding_q_surface
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      _HEAD_DIM HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_q = cur_token_idx * $(q_token_stride) + cur_head_idx * $(q_head_stride)
  off_q0 = off_q + dim_range0 * $(head_dim_stride)
  off_q1 = off_q + dim_range1 * $(head_dim_stride)
  loaded_q0 = tl.load(Q + off_q0)
  loaded_q1 = tl.load(Q + off_q1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(Cos + off_cos_sin)
  loaded_sin = tl.load(Sin + off_cos_sin)
  out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
  out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
  tl.store(Q + off_q0, out_q0)
  tl.store(Q + off_q1, out_q1)
}
```
</details>

<details><summary><code>qFirstAddr</code></summary>

```
/-- Program `(pid₀, pid₁)`'s Q first-half lane address — the pid-level twin of
`qFirstOffset` (which reads the ids out of a `BlockState`). -/
```
```lean
def qFirstAddr
    (pid₀ pid₁ q_token_stride q_head_stride head_dim_stride HALF_DIM : Nat)
    (j : Fin HALF_DIM) : Nat :=
  pid₁ * q_token_stride + pid₀ * q_head_stride + j.val * head_dim_stride
```
</details>

<details><summary><code>qSecondAddr</code></summary>

```
/-- Program `(pid₀, pid₁)`'s Q second-half lane address — the pid-level twin of
`qSecondOffset`. -/
```
```lean
def qSecondAddr
    (pid₀ pid₁ q_token_stride q_head_stride head_dim_stride HALF_DIM : Nat)
    (j : Fin HALF_DIM) : Nat :=
  pid₁ * q_token_stride + pid₀ * q_head_stride +
    (j.val + HALF_DIM) * head_dim_stride
```
</details>

<details><summary><code>cosSinAddr</code></summary>

```
/-- Program `pid₁`'s rotary-factor lane address — the pid-level twin of
`cosOffset` / `sinOffset` (`cos` and `sin` share one addressing scheme). -/
```
```lean
def cosSinAddr (pid₁ cos_token_stride cos_stride HALF_DIM : Nat)
    (j : Fin HALF_DIM) : Nat :=
  pid₁ * cos_token_stride + j.val * cos_stride
```
</details>

## Public theorem: `decoding_fused_rotary_embedding_vcache_chain_correctness`

<details><summary>docstring</summary>

```
/-- **`decodingVCacheChainIO ⊨ decodingVCacheChainSpec`** — the paged-V-cache
copy face of `decoding_fused_rotary_embedding_kernel` as one Hoare triple over
flat pointer memory.

The destination is reached through a **two-link metadata chain**:
`context_lengths[pid₁]` is the sequence length `m₁`, and
`BLOCK_TABLES[pid₁ · bts_stride + ((m₁ - 1) / block_size) · btb_stride]` is the
physical block index `m₂`. The V row of token `pid₁`, head
`pid₀ / KV_GROUP_NUM` then lands at
`m₂ · vcb_stride + (pid₀ / KV_GROUP_NUM) · vch_stride +
((m₁ - 1) % block_size) · vcs_stride + j · vcd_stride`.

Only KV-group leaders are active: `pid₀ % KV_GROUP_NUM = 0` gates both the load
and the store. The stored value is the loaded row **verbatim** —
`decodingVCacheChainSpec` is the identity on `xs` — and every flat cell outside
the written window is untouched.

Dimension-general: `HEAD_DIM`, `block_size`, `KV_GROUP_NUM` and all eight
strides are free `Nat` parameters. The headline carries no side conditions of
its own; the disjoint base-pointer placement, the in-bounds windows and the
launch state pinning the input row and both metadata cells are all part of the
`⊨` triple. -/
```
</details>

**Statement:**
```lean
specification decoding_fused_rotary_embedding_vcache_chain_correctness
    (v v_cache : RegionName) (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM k_token_stride k_head_stride head_dim_stride
      vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride
      block_size HEAD_DIM : Nat) :
    decodingVCacheChainIO v v_cache BLOCK_TABLES context_lengths KV_GROUP_NUM
        k_token_stride k_head_stride head_dim_stride vcb_stride vch_stride vcs_stride
        vcd_stride bts_stride btb_stride block_size HEAD_DIM
      ⊨ decodingVCacheChainSpec
```

**Closed-form spec defs (transitive):** `decodingVCacheChainIO`, `decodingVCacheChainSpec`, `decoding_fused_rotary_embedding_vcache_chain`

<details><summary><code>decodingVCacheChainIO</code></summary>

```lean
def decodingVCacheChainIO
    (v v_cache : RegionName) (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM k_token_stride k_head_stride head_dim_stride
      vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride
      block_size HEAD_DIM : Nat) :
    ChainMetaGroupedMasked2DKernelIO where
  kernel := decoding_fused_rotary_embedding_vcache_chain v v_cache BLOCK_TABLES context_lengths KV_GROUP_NUM
    k_token_stride k_head_stride head_dim_stride vcb_stride vch_stride vcs_stride
    vcd_stride bts_stride btb_stride block_size HEAD_DIM
  nIn := 1
  nOut := 1
  bufs := [v, v_cache, context_lengths, BLOCK_TABLES]
  mbuf1 := context_lengths
  mbuf2 := BLOCK_TABLES
  inp := fun _ => v
  out := fun _ => v_cache
  B := HEAD_DIM
  mwin1 := fun _ pid₁ => pid₁
  mwin2 := fun _ pid₁ m₁ => pid₁ * bts_stride + ((m₁ - 1) / block_size) * btb_stride
  read := fun _ pid₀ pid₁ _ _ j =>
    pid₁ * k_token_stride + (pid₀ / KV_GROUP_NUM) * k_head_stride +
      j.val * head_dim_stride
  readMask := fun _ pid₀ _ _ _ _ => pid₀ % KV_GROUP_NUM = 0
  write := fun _ pid₀ _ m₁ m₂ j =>
    m₂ * vcb_stride + (pid₀ / KV_GROUP_NUM) * vch_stride +
      ((m₁ - 1) % block_size) * vcs_stride + j.val * vcd_stride
  writeMask := fun _ pid₀ _ _ _ _ => pid₀ % KV_GROUP_NUM = 0
```
</details>

<details><summary><code>decodingVCacheChainSpec</code></summary>

```lean
def decodingVCacheChainSpec : Nat → Nat → Nat → Nat → (Fin 1 → Fin HEAD_DIM → ℝ) → Fin 1 →
    Fin HEAD_DIM → ℝ :=
  fun _ _ _ _ xs _ j => xs 0 j
```
</details>

<details><summary><code>decoding_fused_rotary_embedding_vcache_chain</code></summary>

```
/-! ## V-cache honest chained face (single store, copy, B = HEAD_DIM) -/
```
```lean
def decoding_fused_rotary_embedding_vcache_chain
    (v v_cache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM k_token_stride k_head_stride head_dim_stride
      vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride
      block_size HEAD_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    cur_k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    dim_range = tl.arange(0, $(HEAD_DIM))
    off_kv = cur_token_idx * $(k_token_stride) + cur_k_head_idx * $(k_head_stride)
    loaded_v = tl.load(v + off_kv + dim_range * $(head_dim_stride))
    past_kv_seq_len = tl.load(context_lengths + cur_token_idx) - $(1)
    last_block_idx = past_kv_seq_len // $(block_size)
    block_ids = tl.load(BLOCK_TABLES + cur_token_idx * $(bts_stride) +
      last_block_idx * $(btb_stride))
    offsets_in_last_block = past_kv_seq_len % $(block_size)
    v_range = block_ids * $(vcb_stride) + cur_k_head_idx * $(vch_stride) +
      offsets_in_last_block * $(vcs_stride) + dim_range * $(vcd_stride)
    tl.store(v_cache + v_range, loaded_v)
  }
}
```
</details>

## Also present (pinned special-case summaries)
- `decoding_fused_rotary_embedding_q_first_half_compute_correct`
- `decoding_fused_rotary_embedding_q_second_half_compute_correct`
- `decoding_fused_rotary_embedding_k_first_half_compute_correct`
- `decoding_fused_rotary_embedding_k_second_half_compute_correct`
- `decoding_fused_rotary_embedding_v_cache_store_slice_compute_correct`
- `decoding_fused_rotary_embedding_v_cache_guarded_store_slice_compute_correct`
- `decoding_fused_rotary_embedding_k_cache_first_half_store_slice_compute_correct`
- `decoding_fused_rotary_embedding_k_cache_second_half_store_slice_compute_correct`
- `decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice_compute_correct`
- `decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice_compute_correct`
- `decoding_fused_rotary_embedding_context_k_cache_first_half_guarded_store_slice_compute_correct`
- `decoding_fused_rotary_embedding_context_k_cache_second_half_guarded_store_slice_compute_correct`
- `decoding_fused_rotary_embedding_context_v_cache_guarded_store_slice_compute_correct`
- `decoding_fused_rotary_embedding_all_outputs_compute_correct_general`
