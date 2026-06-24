# Spec sheet — `bench/tritonbench_g/fused_rotary_embedding/FusedRotaryEmbedding.lean`

**Python source:** `bench/tritonbench_g/fused_rotary_embedding/fused_rotary_embedding.py`

## Public theorem: `decoding_fused_rotary_embedding_output_summary_general`

<details><summary>docstring</summary>

```
/-- **General top theorem (genuine closed form).** Dimension-general all-outputs
summary built on `decoding_fused_rotary_embedding_all_outputs_compute_correct_general`.
Over symbolic strides/dims (`x`, all Q/K/cos/sin/cache strides, the rotary half
width `HALF_DIM`, the V head dim `HEAD_DIM`, `block_size`, `KV_GROUP_NUM`): the
faithful full kernel surface lowers to the algorithm layer, AND every output —
Q/K rotary writebacks plus the `handle_kv`-guarded paged K/V cache stores driven
by `context_lengths`/`BLOCK_TABLES` — realizes its genuine rotary closed form
reading input memory. No hardcoded test-shape literals; the only side conditions
are the per-region store-offset injectivity hypotheses. -/
```
</details>

**Statement:**
```lean
theorem decoding_fused_rotary_embedding_output_summary_general
    (q k v cos sin OutK0Pre OutK1Pre LoadedV k_cache v_cache BLOCK_TABLES
      context_lengths : RegionName)
    (BLOCK_TABLES_nat context_lengths_nat : Region .nat)
    (q_token_stride q_head_stride k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride vcb_stride vch_stride vcs_stride vcd_stride
      bts_stride btb_stride block_size KV_GROUP_NUM HALF_DIM HEAD_DIM : Nat)
    (s : BlockState)
    (hQ1Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        qFirstOffset s q_token_stride q_head_stride head_dim_stride i))
    (hQ2Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i))
    (hK1Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kFirstOffset s k_token_stride k_head_stride head_dim_stride i))
    (hK2Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i))
    (hKC1Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        decodingKCacheFirstGuardedOffset s BLOCK_TABLES context_lengths
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride bts_stride btb_stride block_size i))
    (hKC2Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        decodingKCacheSecondGuardedOffset s BLOCK_TABLES context_lengths
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride bts_stride btb_stride block_size HALF_DIM i))
    (hVCInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        decodingVCacheGuardedOffset s BLOCK_TABLES context_lengths KV_GROUP_NUM
          vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride
          block_size i)) :
    (∃ alg, (decoding_fused_rotary_embedding_kernel_surface q k v cos sin
      k_cache v_cache BLOCK_TABLES_nat context_lengths_nat
      x q_token_stride q_head_stride k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride vcb_stride vch_stride vcs_stride vcd_stride
      bts_stride btb_stride block_size KV_GROUP_NUM (HALF_DIM * 2)
      ).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_q_first_half q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin HALF_DIM => True)
        (fun i => (q, qFirstOffset s q_token_stride q_head_stride head_dim_stride i)))
      (expected := fun i =>
        qFirstSpec s q cos sin q_token_stride q_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_q_second_half q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin HALF_DIM => True)
        (fun i => (q,
          qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i)))
      (expected := fun i =>
        qSecondSpec s q cos sin q_token_stride q_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_k_first_half k cos sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM)
      (initialState := s)
      (write := fun i : Fin HALF_DIM => some (k,
        kFirstOffset s k_token_stride k_head_stride head_dim_stride i))
      (expected := fun i =>
        kFirstSpec s k cos sin k_token_stride k_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_k_second_half k cos sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM)
      (initialState := s)
      (write := fun i : Fin HALF_DIM => some (k,
        kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i))
      (expected := fun i =>
        kSecondSpec s k cos sin k_token_stride k_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i)) ∧
    (ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
          OutK0Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
            block_size)
          (decodingOffsetsInLastBlock s context_lengths block_size)
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HALF_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (k_cache,
          decodingKCacheFirstGuardedOffset s BLOCK_TABLES context_lengths
            KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
            kcd_stride bts_stride btb_stride block_size i)))
      (expected := fun i => kCacheFirstStoreSpec s OutK0Pre i)) ∧
    (ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
          OutK1Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
            block_size)
          (decodingOffsetsInLastBlock s context_lengths block_size)
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HALF_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (k_cache,
          decodingKCacheSecondGuardedOffset s BLOCK_TABLES context_lengths
            KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
            kcd_stride bts_stride btb_stride block_size HALF_DIM i)))
      (expected := fun i => kCacheSecondStoreSpec s OutK1Pre HALF_DIM i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_v_cache_guarded_store_slice
        LoadedV v_cache
        (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
          block_size)
        (decodingOffsetsInLastBlock s context_lengths block_size)
        KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride HEAD_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HEAD_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (v_cache,
          decodingVCacheGuardedOffset s BLOCK_TABLES context_lengths
            KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride bts_stride
            btb_stride block_size i)))
      (expected := fun i => vCacheStoreSpec s LoadedV i))
```

**Assumptions / layout contracts:**
- `hQ1Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        qFirstOffset s q_token_stride q_head_stride head_dim_stride i)`
- `hQ2Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i)`
- `hK1Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kFirstOffset s k_token_stride k_head_stride head_dim_stride i)`
- `hK2Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i)`
- `hKC1Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        decodingKCacheFirstGuardedOffset s BLOCK_TABLES context_lengths
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride bts_stride btb_stride block_size i)`
- `hKC2Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        decodingKCacheSecondGuardedOffset s BLOCK_TABLES context_lengths
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride bts_stride btb_stride block_size HALF_DIM i)`
- `hVCInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        decodingVCacheGuardedOffset s BLOCK_TABLES context_lengths KV_GROUP_NUM
          vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride
          block_size i)`
- `kernel : = decoding_fused_rotary_embedding_q_first_half q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM`
- `initialState : = s`
- `fun _i : Fin HALF_DIM => True`
- `expected : = fun i =>
        qFirstSpec s q cos sin q_token_stride q_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i`
- `kernel : = decoding_fused_rotary_embedding_q_second_half q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM`
- `initialState : = s`
- `fun _i : Fin HALF_DIM => True`
- `expected : = fun i =>
        qSecondSpec s q cos sin q_token_stride q_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i`
- `kernel : = decoding_fused_rotary_embedding_k_first_half k cos sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM`
- `initialState : = s`
- `write : = fun i : Fin HALF_DIM => some (k,
        kFirstOffset s k_token_stride k_head_stride head_dim_stride i)`
- `expected : = fun i =>
        kFirstSpec s k cos sin k_token_stride k_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i`
- `kernel : = decoding_fused_rotary_embedding_k_second_half k cos sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM`
- `initialState : = s`
- `write : = fun i : Fin HALF_DIM => some (k,
        kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i)`
- `expected : = fun i =>
        kSecondSpec s k cos sin k_token_stride k_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i`
- `kernel : =
        decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
          OutK0Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
            block_size)
          (decodingOffsetsInLastBlock s context_lengths block_size)
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride HALF_DIM`
- `initialState : = s`
- `fun _ : Fin HALF_DIM => handleKv s KV_GROUP_NUM`
- `expected : = fun i => kCacheFirstStoreSpec s OutK0Pre i`
- `kernel : =
        decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
          OutK1Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
            block_size)
          (decodingOffsetsInLastBlock s context_lengths block_size)
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride HALF_DIM`
- `initialState : = s`
- `fun _ : Fin HALF_DIM => handleKv s KV_GROUP_NUM`
- `expected : = fun i => kCacheSecondStoreSpec s OutK1Pre HALF_DIM i`
- `kernel : = decoding_fused_rotary_embedding_v_cache_guarded_store_slice
        LoadedV v_cache
        (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
          block_size)
        (decodingOffsetsInLastBlock s context_lengths block_size)
        KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride HEAD_DIM`
- `initialState : = s`
- `fun _ : Fin HEAD_DIM => handleKv s KV_GROUP_NUM`
- `expected : = fun i => vCacheStoreSpec s LoadedV i`

**Closed-form spec defs (transitive):** `qFirstOffset`, `qSecondOffset`, `kFirstOffset`, `kSecondOffset`, `decodingKCacheFirstGuardedOffset`, `decodingKCacheSecondGuardedOffset`, `decodingVCacheGuardedOffset`, `decoding_fused_rotary_embedding_kernel_surface`, `decoding_fused_rotary_embedding_q_first_half`, `qFirstSpec`, `decoding_fused_rotary_embedding_q_second_half`, `qSecondSpec`, `decoding_fused_rotary_embedding_k_first_half`, `kFirstSpec`, `decoding_fused_rotary_embedding_k_second_half`, `kSecondSpec`, `decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice`, `decodingBlockId`, `decodingOffsetsInLastBlock`, `handleKv`, `kCacheFirstStoreSpec`, `decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice`, `kCacheSecondStoreSpec`, `decoding_fused_rotary_embedding_v_cache_guarded_store_slice`, `vCacheStoreSpec`, `qBase`, `dimIndex`, `kBase`, `kCacheFirstGuardedOffset`, `kCacheSecondGuardedOffset`, `vCacheGuardedOffset`, `cosOffset`, `sinOffset`, `decodingLastBlockIdx`, `decodingPastKvSeqLen`

<details><summary><code>qFirstOffset</code></summary>

```lean
def qFirstOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride : Nat)
    (i : Fin HALF_DIM) : Nat :=
  qBase s q_token_stride q_head_stride + dimIndex i * head_dim_stride
```
</details>

<details><summary><code>qSecondOffset</code></summary>

```lean
def qSecondOffset
    (s : BlockState)
    (q_token_stride q_head_stride head_dim_stride HALF_DIM : Nat)
    (i : Fin HALF_DIM) : Nat :=
  qBase s q_token_stride q_head_stride + (dimIndex i + HALF_DIM) * head_dim_stride
```
</details>

<details><summary><code>kFirstOffset</code></summary>

```lean
def kFirstOffset
    (s : BlockState) (k_token_stride k_head_stride head_dim_stride : Nat)
    (i : Fin HALF_DIM) : Nat :=
  kBase s k_token_stride k_head_stride + dimIndex i * head_dim_stride
```
</details>

<details><summary><code>kSecondOffset</code></summary>

```lean
def kSecondOffset
    (s : BlockState)
    (k_token_stride k_head_stride head_dim_stride HALF_DIM : Nat)
    (i : Fin HALF_DIM) : Nat :=
  kBase s k_token_stride k_head_stride + (dimIndex i + HALF_DIM) * head_dim_stride
```
</details>

<details><summary><code>decodingKCacheFirstGuardedOffset</code></summary>

```lean
def decodingKCacheFirstGuardedOffset
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
      kcd_stride bts_stride btb_stride block_size : Nat)
    (i : Fin HALF_DIM) : Nat :=
  kCacheFirstGuardedOffset s
    (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
      block_size)
    (decodingOffsetsInLastBlock s context_lengths block_size)
    KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
    kcd_stride i
```
</details>

<details><summary><code>decodingKCacheSecondGuardedOffset</code></summary>

```lean
def decodingKCacheSecondGuardedOffset
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
      kcd_stride bts_stride btb_stride block_size HALF_DIM : Nat)
    (i : Fin HALF_DIM) : Nat :=
  kCacheSecondGuardedOffset s
    (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
      block_size)
    (decodingOffsetsInLastBlock s context_lengths block_size)
    KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
    kcd_stride HALF_DIM i
```
</details>

<details><summary><code>decodingVCacheGuardedOffset</code></summary>

```lean
def decodingVCacheGuardedOffset
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride bts_stride
      btb_stride block_size : Nat)
    (i : Fin HEAD_DIM) : Nat :=
  vCacheGuardedOffset s
    (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
      block_size)
    (decodingOffsetsInLastBlock s context_lengths block_size)
    KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride i
```
</details>

<details><summary><code>decoding_fused_rotary_embedding_kernel_surface</code></summary>

```
/-- Faithful transcription of `fused_rotary_embedding.py`'s
`decoding_fused_rotary_embedding_kernel`.

This keeps the unconditional Q rotary writes plus the conditional K rotary and
K/V cache-fill path guarded by `cur_head_idx % KV_GROUP_NUM == 0`. -/
```
```lean
def decoding_fused_rotary_embedding_kernel_surface
    (q k v cos sin k_cache v_cache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (x q_token_stride q_head_stride k_token_stride k_head_stride
      head_dim_stride cos_token_stride cos_stride kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride vcb_stride vch_stride vcs_stride
      vcd_stride bts_stride btb_stride block_size KV_GROUP_NUM HEAD_DIM
      : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)

  dim_range = tl.arange(0, $(HEAD_DIM))
  dim_range0 = tl.arange(0, $(HEAD_DIM) // $(2))
  dim_range1 = tl.arange($(HEAD_DIM) // $(2), $(HEAD_DIM))

  off_q = cur_token_idx * $(q_token_stride) + cur_head_idx * $(q_head_stride)
  off_q0 = off_q + dim_range0 * $(head_dim_stride)
  off_q1 = off_q + dim_range1 * $(head_dim_stride)

  loaded_q0 = tl.load(q + off_q0)
  loaded_q1 = tl.load(q + off_q1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin)
  loaded_sin = tl.load(sin + off_cos_sin)

  out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
  out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
  tl.store(q + off_q0, out_q0)
  tl.store(q + off_q1, out_q1)

  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    cur_k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    off_kv = cur_token_idx * $(k_token_stride) + cur_k_head_idx * $(k_head_stride)
    off_k0 = off_kv + dim_range0 * $(head_dim_stride)
    off_k1 = off_kv + dim_range1 * $(head_dim_stride)
    loaded_k0 = tl.load(k + off_k0)
    loaded_k1 = tl.load(k + off_k1)

    out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
    out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos

    past_kv_seq_len = tl.load(context_lengths + cur_token_idx) - $(1)

    last_block_idx = past_kv_seq_len // $(block_size)
    block_ids = tl.load(BLOCK_TABLES + cur_token_idx * $(bts_stride) +
      last_block_idx * $(btb_stride))
    offsets_in_last_block = past_kv_seq_len % $(block_size)
    offsets_cache_base = block_ids * $(kcb_stride) +
      cur_k_head_idx * $(kch_stride)
    k_range0 = offsets_cache_base +
      offsets_in_last_block * $(kcs_stride) +
      (dim_range0 // $(x)) * $(kcsplit_x_stride) +
      (dim_range0 % $(x)) * $(kcd_stride)
    k_range1 = offsets_cache_base +
      offsets_in_last_block * $(kcs_stride) +
      (dim_range1 // $(x)) * $(kcsplit_x_stride) +
      (dim_range1 % $(x)) * $(kcd_stride)
    tl.store(k_cache + k_range0, out_k0)
    tl.store(k_cache + k_range1, out_k1)

    off_v = off_kv + dim_range * $(head_dim_stride)
    loaded_v = tl.load(v + off_v)
    v_range = block_ids * $(vcb_stride) +
      cur_k_head_idx * $(vch_stride) +
      offsets_in_last_block * $(vcs_stride) +
      dim_range * $(vcd_stride)
    tl.store(v_cache + v_range, loaded_v)
  }
}
```
</details>

<details><summary><code>decoding_fused_rotary_embedding_q_first_half</code></summary>

```
/-- Proof-oriented Q first-half slice of `fused_rotary_embedding.py`'s
`decoding_fused_rotary_embedding_kernel`.

The full kernel updates Q, conditionally rotates K, and fills K/V caches. This
slice captures the unconditional Q first-half rotary writeback:
`q0 * cos - q1 * sin`. -/
```
```lean
def decoding_fused_rotary_embedding_q_first_half
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_q = cur_token_idx * $(q_token_stride) + cur_head_idx * $(q_head_stride)
  off_q0 = off_q + dim_range0 * $(head_dim_stride)
  off_q1 = off_q + dim_range1 * $(head_dim_stride)
  loaded_q0 = tl.load(q + off_q0)
  loaded_q1 = tl.load(q + off_q1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin)
  loaded_sin = tl.load(sin + off_cos_sin)
  out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
  tl.store(q + off_q0, out_q0)
}
```
</details>

<details><summary><code>qFirstSpec</code></summary>

```lean
noncomputable def qFirstSpec
    (s : BlockState) (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (i : Fin HALF_DIM) : ℝ :=
  s.readMem q (qFirstOffset s q_token_stride q_head_stride head_dim_stride i) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride i) -
  s.readMem q
      (qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i) *
    s.readMem sin (sinOffset s cos_token_stride cos_stride i)
```
</details>

<details><summary><code>decoding_fused_rotary_embedding_q_second_half</code></summary>

```
/-- Proof-oriented Q second-half slice of `fused_rotary_embedding.py`'s
`decoding_fused_rotary_embedding_kernel`. Captures the unconditional Q
second-half rotary writeback: `q1' = q1 * cos + q0 * sin`. -/
```
```lean
def decoding_fused_rotary_embedding_q_second_half
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_q = cur_token_idx * $(q_token_stride) + cur_head_idx * $(q_head_stride)
  off_q0 = off_q + dim_range0 * $(head_dim_stride)
  off_q1 = off_q + dim_range1 * $(head_dim_stride)
  loaded_q0 = tl.load(q + off_q0)
  loaded_q1 = tl.load(q + off_q1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin)
  loaded_sin = tl.load(sin + off_cos_sin)
  out_q1 = loaded_q1 * loaded_cos + loaded_q0 * loaded_sin
  tl.store(q + off_q1, out_q1)
}
```
</details>

<details><summary><code>qSecondSpec</code></summary>

```lean
noncomputable def qSecondSpec
    (s : BlockState) (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (i : Fin HALF_DIM) : ℝ :=
  s.readMem q
      (qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride i) +
  s.readMem q (qFirstOffset s q_token_stride q_head_stride head_dim_stride i) *
    s.readMem sin (sinOffset s cos_token_stride cos_stride i)
```
</details>

<details><summary><code>decoding_fused_rotary_embedding_k_first_half</code></summary>

```
/-- Proof-oriented K first-half slice of
`decoding_fused_rotary_embedding_kernel`. K analog of the Q first-half slice
writing `out_k0 = k0 * cos - k1 * sin` to `k + off_k0`. The KV_GROUP gating
that ungates the K branch in the full kernel is omitted here; the slice models
the K rotary stores assuming the gate is active. -/
```
```lean
def decoding_fused_rotary_embedding_k_first_half
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_k_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_kv = cur_token_idx * $(k_token_stride) + cur_k_head_idx * $(k_head_stride)
  off_k0 = off_kv + dim_range0 * $(head_dim_stride)
  off_k1 = off_kv + dim_range1 * $(head_dim_stride)
  loaded_k0 = tl.load(k + off_k0)
  loaded_k1 = tl.load(k + off_k1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin)
  loaded_sin = tl.load(sin + off_cos_sin)
  out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
  tl.store(k + off_k0, out_k0)
}
```
</details>

<details><summary><code>kFirstSpec</code></summary>

```lean
noncomputable def kFirstSpec
    (s : BlockState) (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (i : Fin HALF_DIM) : ℝ :=
  s.readMem k (kFirstOffset s k_token_stride k_head_stride head_dim_stride i) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride i) -
  s.readMem k
      (kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i) *
    s.readMem sin (sinOffset s cos_token_stride cos_stride i)
```
</details>

<details><summary><code>decoding_fused_rotary_embedding_k_second_half</code></summary>

```
/-- Proof-oriented K second-half slice of
`decoding_fused_rotary_embedding_kernel`. Writes `out_k1 = k0 * sin + k1 * cos`
to `k + off_k1`. -/
```
```lean
def decoding_fused_rotary_embedding_k_second_half
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_k_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_kv = cur_token_idx * $(k_token_stride) + cur_k_head_idx * $(k_head_stride)
  off_k0 = off_kv + dim_range0 * $(head_dim_stride)
  off_k1 = off_kv + dim_range1 * $(head_dim_stride)
  loaded_k0 = tl.load(k + off_k0)
  loaded_k1 = tl.load(k + off_k1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin)
  loaded_sin = tl.load(sin + off_cos_sin)
  out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos
  tl.store(k + off_k1, out_k1)
}
```
</details>

<details><summary><code>kSecondSpec</code></summary>

```lean
noncomputable def kSecondSpec
    (s : BlockState) (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (i : Fin HALF_DIM) : ℝ :=
  s.readMem k (kFirstOffset s k_token_stride k_head_stride head_dim_stride i) *
    s.readMem sin (sinOffset s cos_token_stride cos_stride i) +
  s.readMem k
      (kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride i)
```
</details>

<details><summary><code>decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice</code></summary>

```
/-- Guarded K-cache first-half store slice for the `handle_kv` branch. -/
```
```lean
def decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
    (OutK0Pre k_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    cur_k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    dim_range0 = tl.arange(0, $(HALF_DIM))
    out_k0 = tl.load(OutK0Pre + dim_range0)
    k_range0 = $(block_id) * $(kcb_stride) +
      cur_k_head_idx * $(kch_stride) +
      $(offsets_in_last_block) * $(kcs_stride) +
      (dim_range0 // $(x)) * $(kcsplit_x_stride) +
      (dim_range0 % $(x)) * $(kcd_stride)
    tl.store(k_cache + k_range0, out_k0)
  }
}
```
</details>

<details><summary><code>decodingBlockId</code></summary>

```
/-- Python decode metadata:
`block_ids = BLOCK_TABLES[cur_token_idx, last_block_idx]`. -/
```
```lean
def decodingBlockId
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (bts_stride btb_stride block_size : Nat) : Nat :=
  s.readMemValue .nat BLOCK_TABLES
    (s.pids 1 * bts_stride +
      decodingLastBlockIdx s context_lengths block_size * btb_stride)
```
</details>

<details><summary><code>decodingOffsetsInLastBlock</code></summary>

```
/-- Python decode metadata:
`offsets_in_last_block = past_kv_seq_len % block_size`. -/
```
```lean
def decodingOffsetsInLastBlock
    (s : BlockState) (context_lengths : RegionName) (block_size : Nat) : Nat :=
  decodingPastKvSeqLen s context_lengths % block_size
```
</details>

<details><summary><code>handleKv</code></summary>

```lean
def handleKv (s : BlockState) (KV_GROUP_NUM : Nat) : Prop :=
  s.pids 0 % KV_GROUP_NUM = 0
```
</details>

<details><summary><code>kCacheFirstStoreSpec</code></summary>

```lean
noncomputable def kCacheFirstStoreSpec
    (s : BlockState) (OutK0Pre : RegionName) (i : Fin HALF_DIM) : ℝ :=
  s.readMem OutK0Pre i.val
```
</details>

<details><summary><code>decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice</code></summary>

```
/-- Guarded K-cache second-half store slice for the `handle_kv` branch. -/
```
```lean
def decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
    (OutK1Pre k_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    cur_k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
    out_k1 = tl.load(OutK1Pre + dim_range1)
    k_range1 = $(block_id) * $(kcb_stride) +
      cur_k_head_idx * $(kch_stride) +
      $(offsets_in_last_block) * $(kcs_stride) +
      (dim_range1 // $(x)) * $(kcsplit_x_stride) +
      (dim_range1 % $(x)) * $(kcd_stride)
    tl.store(k_cache + k_range1, out_k1)
  }
}
```
</details>

<details><summary><code>kCacheSecondStoreSpec</code></summary>

```lean
noncomputable def kCacheSecondStoreSpec
    (s : BlockState) (OutK1Pre : RegionName) (HALF_DIM : Nat) (i : Fin HALF_DIM) : ℝ :=
  s.readMem OutK1Pre (i.val + HALF_DIM)
```
</details>

<details><summary><code>decoding_fused_rotary_embedding_v_cache_guarded_store_slice</code></summary>

```
/-- Guarded V-cache store slice for the `handle_kv` branch of
`decoding_fused_rotary_embedding_kernel`.

This keeps the Python branch guard `cur_head_idx % KV_GROUP_NUM == 0` and uses
the derived `cur_k_head_idx = cur_head_idx // KV_GROUP_NUM` in the cache
address. `block_id` and `offsets_in_last_block` remain parameters here; the
companion full surface above records their origin from `context_lengths` and
`BLOCK_TABLES`. -/
```
```lean
def decoding_fused_rotary_embedding_v_cache_guarded_store_slice
    (LoadedV v_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM vcb_stride vch_stride
      vcs_stride vcd_stride HEAD_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    cur_k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    dim_range = tl.arange(0, $(HEAD_DIM))
    loaded_v = tl.load(LoadedV + dim_range)
    v_range = $(block_id) * $(vcb_stride) +
      cur_k_head_idx * $(vch_stride) +
      $(offsets_in_last_block) * $(vcs_stride) +
      dim_range * $(vcd_stride)
    tl.store(v_cache + v_range, loaded_v)
  }
}
```
</details>

<details><summary><code>vCacheStoreSpec</code></summary>

```lean
noncomputable def vCacheStoreSpec
    (s : BlockState) (LoadedV : RegionName) (i : Fin HEAD_DIM) : ℝ :=
  s.readMem LoadedV i.val
```
</details>

<details><summary><code>qBase</code></summary>

```lean
def qBase (s : BlockState) (q_token_stride q_head_stride : Nat) : Nat :=
  s.pids 1 * q_token_stride + s.pids 0 * q_head_stride
```
</details>

<details><summary><code>dimIndex</code></summary>

```lean
def dimIndex (i : Fin HALF_DIM) : Nat :=
  i.val
```
</details>

<details><summary><code>kBase</code></summary>

```lean
def kBase (s : BlockState) (k_token_stride k_head_stride : Nat) : Nat :=
  s.pids 1 * k_token_stride + s.pids 0 * k_head_stride
```
</details>

<details><summary><code>kCacheFirstGuardedOffset</code></summary>

```lean
def kCacheFirstGuardedOffset
    (s : BlockState)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride : Nat)
    (i : Fin HALF_DIM) : Nat :=
  block_id * kcb_stride + (s.pids 0 / KV_GROUP_NUM) * kch_stride +
    offsets_in_last_block * kcs_stride +
    (i.val / x) * kcsplit_x_stride + (i.val % x) * kcd_stride
```
</details>

<details><summary><code>kCacheSecondGuardedOffset</code></summary>

```lean
def kCacheSecondGuardedOffset
    (s : BlockState)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat)
    (i : Fin HALF_DIM) : Nat :=
  block_id * kcb_stride + (s.pids 0 / KV_GROUP_NUM) * kch_stride +
    offsets_in_last_block * kcs_stride +
    ((i.val + HALF_DIM) / x) * kcsplit_x_stride +
    ((i.val + HALF_DIM) % x) * kcd_stride
```
</details>

<details><summary><code>vCacheGuardedOffset</code></summary>

```lean
def vCacheGuardedOffset
    (s : BlockState)
    (block_id offsets_in_last_block KV_GROUP_NUM vcb_stride vch_stride
      vcs_stride vcd_stride : Nat)
    (i : Fin HEAD_DIM) : Nat :=
  block_id * vcb_stride + (s.pids 0 / KV_GROUP_NUM) * vch_stride +
    offsets_in_last_block * vcs_stride + i.val * vcd_stride
```
</details>

<details><summary><code>cosOffset</code></summary>

```lean
def cosOffset
    (s : BlockState) (cos_token_stride cos_stride : Nat) (i : Fin HALF_DIM) : Nat :=
  s.pids 1 * cos_token_stride + dimIndex i * cos_stride
```
</details>

<details><summary><code>sinOffset</code></summary>

```lean
def sinOffset
    (s : BlockState) (cos_token_stride cos_stride : Nat) (i : Fin HALF_DIM) : Nat :=
  s.pids 1 * cos_token_stride + dimIndex i * cos_stride
```
</details>

<details><summary><code>decodingLastBlockIdx</code></summary>

```
/-- Python decode metadata: `last_block_idx = past_kv_seq_len // block_size`. -/
```
```lean
def decodingLastBlockIdx
    (s : BlockState) (context_lengths : RegionName) (block_size : Nat) : Nat :=
  decodingPastKvSeqLen s context_lengths / block_size
```
</details>

<details><summary><code>decodingPastKvSeqLen</code></summary>

```
/-- Python decode metadata:
`past_kv_seq_len = context_lengths[cur_token_idx] - 1`. -/
```
```lean
def decodingPastKvSeqLen (s : BlockState) (context_lengths : RegionName) : Nat :=
  s.readMemValue .nat context_lengths (s.pids 1) - 1
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
