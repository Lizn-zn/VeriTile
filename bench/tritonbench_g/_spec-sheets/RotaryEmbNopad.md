# Spec sheet — `bench/tritonbench_g/rotary_emb_nopad/RotaryEmbNopad.lean`

**Python source:** `bench/tritonbench_g/rotary_emb_nopad/rotary_emb_nopad.py`

## Public theorem: `rotary_emb_nopad_output_summary_general`

<details><summary>docstring</summary>

```
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

The kernel-2 (`fused_rotary_embedding_kernel_v2`) faces extend this over the
1-D `Fin HEAD_HALF` `dim` tile: the full v2 surface lowers, `fusedV2QIO` writes
the in-place rotary `Q` halves on every head-active program (with `Cos`/`Sin`
token-masked to `0`), and `fusedV2CacheIO` — on the chained-metadata surface,
whose two `.nat` slots load `context_lengths[token]` then
`BLOCK_TABLES[token·bts + ((m₁−1)/block_size)·btb]` — scatters the rotary `K`
halves into the paged cache at block `m₂`, in-block offset `(m₁−1) % block_size`.

The host launch remains the trusted boundary. -/
```
</details>

**Statement:**
```lean
specification rotary_emb_nopad_output_summary_general
    (Q K Cos Sin KVCache : RegionName)
    (BlockTables ContextLengths : Region .nat)
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
            KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx')
    (v2_q_token_stride v2_q_head_stride v2_head_dim_stride v2_cos_token_stride
      v2_cos_stride v2_q_total_tokens v2_Q_HEAD_NUM v2_HEAD_HALF : Nat)
    (v2_k_token_stride v2_k_head_stride cacheb_stride cacheh_stride cachebs_stride
      cached_stride bts_stride btb_stride block_size : Nat)
    (hV2QInjFirst : ∀ s : BlockState, Function.Injective
      (fun i : Fin v2_HEAD_HALF =>
        v2QFirstOffset s v2_q_token_stride v2_q_head_stride v2_head_dim_stride i))
    (hV2QInjSecond : ∀ s : BlockState, Function.Injective
      (fun i : Fin v2_HEAD_HALF =>
        v2QSecondOffset s v2_q_token_stride v2_q_head_stride v2_head_dim_stride
          v2_HEAD_HALF i))
    (hV2QDisjoint : ∀ (s : BlockState) (i i' : Fin v2_HEAD_HALF),
      v2QFirstOffset s v2_q_token_stride v2_q_head_stride v2_head_dim_stride i ≠
        v2QSecondOffset s v2_q_token_stride v2_q_head_stride v2_head_dim_stride
          v2_HEAD_HALF i')
    (hV2CacheDisjoint : ∀ (s : BlockState) (i i' : Fin v2_HEAD_HALF),
      v2CacheOff0 s BlockTables ContextLengths cacheb_stride cacheh_stride
          cachebs_stride cached_stride bts_stride btb_stride block_size
          v2_q_total_tokens i ≠
        v2CacheOff1 s BlockTables ContextLengths cacheb_stride cacheh_stride
          cachebs_stride cached_stride bts_stride btb_stride block_size
          v2_q_total_tokens v2_HEAD_HALF i') :
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
            (xs (⟨2, by decide⟩ : Fin 4) j) (xs (⟨3, by decide⟩ : Fin 4) j) o) ∧
    (∃ alg, (fused_rotary_embedding_v2_surface Q K Cos Sin KVCache BlockTables
      ContextLengths surf_q_token_stride surf_q_head_stride surf_k_token_stride
      surf_k_head_stride surf_head_dim_stride surf_cos_token_stride surf_cos_stride
      cacheb_stride cacheh_stride cachebs_stride cached_stride bts_stride btb_stride
      block_size surf_q_total_tokens surf_Q_HEAD_NUM surf_HEAD_DIM).toAlgorithm?
      = Except.ok alg) ∧
    (fusedV2QIO Q Cos Sin v2_q_token_stride v2_q_head_stride v2_head_dim_stride
        v2_cos_token_stride v2_cos_stride v2_q_total_tokens v2_Q_HEAD_NUM v2_HEAD_HALF
      ⊨ fun _pid₀ pid₁ xs o j =>
          rotaryPair (xs (⟨0, by decide⟩ : Fin 4) j) (xs (⟨1, by decide⟩ : Fin 4) j)
            (if pid₁ < v2_q_total_tokens then xs (⟨2, by decide⟩ : Fin 4) j else 0)
            (if pid₁ < v2_q_total_tokens then xs (⟨3, by decide⟩ : Fin 4) j else 0) o) ∧
    (ChainMetaGroupedMasked2DKernelIO.Implements
      (fusedV2CacheIO KVCache K Cos Sin BlockTables ContextLengths v2_k_token_stride
        v2_k_head_stride v2_head_dim_stride v2_cos_token_stride v2_cos_stride
        cacheb_stride cacheh_stride cachebs_stride cached_stride bts_stride btb_stride
        block_size v2_q_total_tokens v2_Q_HEAD_NUM v2_HEAD_HALF)
      (fun _pid₀ pid₁ _s1 _s2 xs o j =>
          rotaryPair (xs (⟨0, by decide⟩ : Fin 4) j) (xs (⟨1, by decide⟩ : Fin 4) j)
            (if pid₁ < v2_q_total_tokens then xs (⟨2, by decide⟩ : Fin 4) j else 0)
            (if pid₁ < v2_q_total_tokens then xs (⟨3, by decide⟩ : Fin 4) j else 0) o))
```

**Assumptions / layout contracts:**
- `hQInjFirst : ∀ s : BlockState, Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS idx)`
- `hQInjSecond : ∀ s : BlockState, Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS HEAD_HALF idx)`
- `hQDisjoint : ∀ (s : BlockState)
        (idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]),
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS idx ≠
          qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF idx'`
- `hKInjFirst : ∀ s : BlockState, Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
          KV_GROUP_NUM BLOCK_TOKENS idx)`
- `hKInjSecond : ∀ s : BlockState, Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
          KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx)`
- `hKDisjoint : ∀ (s : BlockState)
        (idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]),
        kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS idx ≠
          kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx'`
- `hV2QInjFirst : ∀ s : BlockState, Function.Injective
      (fun i : Fin v2_HEAD_HALF =>
        v2QFirstOffset s v2_q_token_stride v2_q_head_stride v2_head_dim_stride i)`
- `hV2QInjSecond : ∀ s : BlockState, Function.Injective
      (fun i : Fin v2_HEAD_HALF =>
        v2QSecondOffset s v2_q_token_stride v2_q_head_stride v2_head_dim_stride
          v2_HEAD_HALF i)`
- `hV2QDisjoint : ∀ (s : BlockState) (i i' : Fin v2_HEAD_HALF),
      v2QFirstOffset s v2_q_token_stride v2_q_head_stride v2_head_dim_stride i ≠
        v2QSecondOffset s v2_q_token_stride v2_q_head_stride v2_head_dim_stride
          v2_HEAD_HALF i'`
- `hV2CacheDisjoint : ∀ (s : BlockState) (i i' : Fin v2_HEAD_HALF),
      v2CacheOff0 s BlockTables ContextLengths cacheb_stride cacheh_stride
          cachebs_stride cached_stride bts_stride btb_stride block_size
          v2_q_total_tokens i ≠
        v2CacheOff1 s BlockTables ContextLengths cacheb_stride cacheh_stride
          cachebs_stride cached_stride bts_stride btb_stride block_size
          v2_q_total_tokens v2_HEAD_HALF i'`

**Closed-form spec defs (transitive):** `qFullFirstOffset`, `qFullSecondOffset`, `kFullFirstOffset`, `kFullSecondOffset`, `v2QFirstOffset`, `v2QSecondOffset`, `v2CacheOff0`, `v2CacheOff1`, `rotary_embedding_kernel_surface`, `rotaryNopadQIO`, `rotaryPair`, `rotaryNopadKIO`, `fused_rotary_embedding_v2_surface`, `fusedV2QIO`, `fusedV2CacheIO`, `v2BlockId`, `v2OffsetsInLastBlock`, `rotary_embedding_q_surface`, `dataFirstP`, `dataSecondP`, `cosP`, `activeQP`, `tokP`, `rotary_embedding_k_surface`, `activeKP`, `fused_rotary_embedding_v2_q_surface`, `fused_rotary_embedding_v2_cache_surface`, `v2LastBlockIdx`, `v2PastKvSeqLen`

<details><summary><code>qFullFirstOffset</code></summary>

```
/-- Per-tile-index Q first-half offset for the full Q surface kernel.
The 3D tile shape is `[BLOCK_TOKENS, 1, HEAD_HALF]`. -/
```
```lean
def qFullFirstOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride
      BLOCK_TOKENS : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Nat :=
  (s.pids 1 * BLOCK_TOKENS + idx.1.val) * q_token_stride +
    s.pids 0 * q_head_stride + idx.2.2.1.val * head_dim_stride
```
</details>

<details><summary><code>qFullSecondOffset</code></summary>

```
/-- Per-tile-index Q second-half offset for the full Q surface kernel. -/
```
```lean
def qFullSecondOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride
      BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Nat :=
  (s.pids 1 * BLOCK_TOKENS + idx.1.val) * q_token_stride +
    s.pids 0 * q_head_stride + (idx.2.2.1.val + HEAD_HALF) * head_dim_stride
```
</details>

<details><summary><code>kFullFirstOffset</code></summary>

```
/-- Per-tile-index K first-half offset for the full K surface kernel.
The K head is selected by the Python kernel as `cur_head_idx // KV_GROUP_NUM`
after the modular `handle_kv` gate has fired. -/
```
```lean
def kFullFirstOffset
    (s : BlockState) (k_token_stride k_head_stride head_dim_stride
      KV_GROUP_NUM BLOCK_TOKENS : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Nat :=
  (s.pids 1 * BLOCK_TOKENS + idx.1.val) * k_token_stride +
    (s.pids 0 / KV_GROUP_NUM) * k_head_stride +
    idx.2.2.1.val * head_dim_stride
```
</details>

<details><summary><code>kFullSecondOffset</code></summary>

```
/-- Per-tile-index K second-half offset for the full K surface kernel. -/
```
```lean
def kFullSecondOffset
    (s : BlockState) (k_token_stride k_head_stride head_dim_stride
      KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Nat :=
  (s.pids 1 * BLOCK_TOKENS + idx.1.val) * k_token_stride +
    (s.pids 0 / KV_GROUP_NUM) * k_head_stride +
    (idx.2.2.1.val + HEAD_HALF) * head_dim_stride
```
</details>

<details><summary><code>v2QFirstOffset</code></summary>

```lean
def v2QFirstOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride : Nat)
    (i : Fin HEAD_HALF) : Nat :=
  s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
    i.val * head_dim_stride
```
</details>

<details><summary><code>v2QSecondOffset</code></summary>

```lean
def v2QSecondOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride
      HEAD_HALF : Nat) (i : Fin HEAD_HALF) : Nat :=
  s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
    (i.val + HEAD_HALF) * head_dim_stride
```
</details>

<details><summary><code>v2CacheOff0</code></summary>

```
/-- Cache first-half scatter address of lane `i`. The block-id contribution is
`0` on a token-out program (the padded `BLOCK_TABLES` load), matching the
`if`-distributed shape the kernel evaluates to. -/
```
```lean
def v2CacheOff0 (s : BlockState) (BlockTables ContextLengths : RegionName)
    (cacheb_stride cacheh_stride cachebs_stride cached_stride bts_stride btb_stride
      block_size q_total_tokens : Nat) (i : Fin HEAD_HALF) : Nat :=
  (if s.pids 1 < q_total_tokens then
      v2BlockId s BlockTables ContextLengths bts_stride btb_stride block_size *
        cacheb_stride
    else 0) +
    s.pids 0 * cacheh_stride +
    v2OffsetsInLastBlock s ContextLengths block_size cachebs_stride +
    i.val * cached_stride
```
</details>

<details><summary><code>v2CacheOff1</code></summary>

```
/-- Cache second-half scatter address of lane `i`. -/
```
```lean
def v2CacheOff1 (s : BlockState) (BlockTables ContextLengths : RegionName)
    (cacheb_stride cacheh_stride cachebs_stride cached_stride bts_stride btb_stride
      block_size q_total_tokens HEAD_HALF : Nat) (i : Fin HEAD_HALF) : Nat :=
  (if s.pids 1 < q_total_tokens then
      v2BlockId s BlockTables ContextLengths bts_stride btb_stride block_size *
        cacheb_stride
    else 0) +
    s.pids 0 * cacheh_stride +
    v2OffsetsInLastBlock s ContextLengths block_size cachebs_stride +
    (i.val + HEAD_HALF) * cached_stride
```
</details>

<details><summary><code>rotary_embedding_kernel_surface</code></summary>

```
/-- Faithful transcription of `rotary_emb_nopad.py`'s
`rotary_embedding_kernel`.

This keeps the unconditional Q rotary writes and the conditional K rotary path
in one kernel; the smaller Q/K kernels below remain available for local
correctness arguments. -/
```
```lean
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
```
</details>

<details><summary><code>rotaryNopadQIO</code></summary>

```
/-- The **IO signature** of the Q face of `rotary_embedding_kernel` — four read
windows (`q0`/`q1` half-tiles + `Cos`/`Sin`) and two in-place Q half-stores over
the flat `Fin (BLOCK_TOKENS * HEAD_HALF)` lane space. -/
```
```lean
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
```
</details>

<details><summary><code>rotaryPair</code></summary>

```
/-- The rotary output of output channel `o` from the four loaded lane values
`(dataFirst, dataSecond, cos, sin) = (a, b, c, d)`: first half `a·c − b·d`,
second half `a·d + b·c`. Factored into one definition so both `⊨` faces and the
two headline conjuncts share a single matcher (a bare inline `match o` generates
a fresh per-site matcher that blocks the cross-declaration `exact`). -/
```
```lean
def rotaryPair (a b c d : ℝ) (o : Fin 2) : ℝ :=
  match o with
  | ⟨0, _⟩ => a * c - b * d
  | ⟨_ + 1, _⟩ => a * d + b * c
```
</details>

<details><summary><code>rotaryNopadKIO</code></summary>

```
/-- The **IO signature** of the K face of `rotary_embedding_kernel` — four read
windows (`k0`/`k1` half-tiles + `Cos`/`Sin`) and two in-place K half-stores over
the flat `Fin (BLOCK_TOKENS * HEAD_HALF)` lane space, every channel gated by the
GQA-leader predicate `pid₀ % KV_GROUP_NUM = 0` and addressed at K head
`pid₀ / KV_GROUP_NUM`. -/
```
```lean
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
```
</details>

<details><summary><code>fused_rotary_embedding_v2_surface</code></summary>

```
/-- Surface transcription of `rotary_emb_nopad.py`'s
`fused_rotary_embedding_kernel_v2`.

Python returns early when `block_head_index >= Q_HEAD_NUM`; this surface
represents that with a guarded body. The benchmark initializes
`context_lengths >= 1`, so the `past_kv_seq_len = context_lengths[...] - 1`
Nat subtraction follows the exercised path. -/
```
```lean
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
```
</details>

<details><summary><code>fusedV2QIO</code></summary>

```
/-- The **IO signature** of the Q face of `fused_rotary_embedding_kernel_v2` —
four read channels (`q0`/`q1` half-lanes + `Cos`/`Sin`) and two in-place `Q`
half-stores over the flat `Fin HEAD_HALF` `dim` lane space, gated by the head
predicate `pid₀ < Q_HEAD_NUM`. -/
```
```lean
def fusedV2QIO (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF : Nat) :
    GroupedMasked2DKernelIO where
  kernel := fused_rotary_embedding_v2_q_surface Q Cos Sin q_token_stride
    q_head_stride head_dim_stride cos_token_stride cos_stride q_total_tokens
    Q_HEAD_NUM HEAD_HALF
  nIn := 4
  nOut := 2
  bufs := [Q, Cos, Sin]
  inp := fun i => match i with
    | ⟨0, _⟩ => Q | ⟨1, _⟩ => Q | ⟨2, _⟩ => Cos | ⟨_ + 3, _⟩ => Sin
  out := fun _ => Q
  B := HEAD_HALF
  read := fun i _pid₀ pid₁ j => match i with
    | ⟨0, _⟩ => pid₁ * q_token_stride + _pid₀ * q_head_stride + j.val * head_dim_stride
    | ⟨1, _⟩ => pid₁ * q_token_stride + _pid₀ * q_head_stride +
        (j.val + HEAD_HALF) * head_dim_stride
    | ⟨2, _⟩ => pid₁ * cos_token_stride + j.val * cos_stride
    | ⟨_ + 3, _⟩ => pid₁ * cos_token_stride + j.val * cos_stride
  readMask := fun i pid₀ pid₁ _j => match i with
    | ⟨0, _⟩ => pid₀ < Q_HEAD_NUM
    | ⟨1, _⟩ => pid₀ < Q_HEAD_NUM
    | ⟨2, _⟩ => pid₁ < q_total_tokens
    | ⟨_ + 3, _⟩ => pid₁ < q_total_tokens
  write := fun o pid₀ pid₁ j => match o with
    | ⟨0, _⟩ => pid₁ * q_token_stride + pid₀ * q_head_stride + j.val * head_dim_stride
    | ⟨_ + 1, _⟩ => pid₁ * q_token_stride + pid₀ * q_head_stride +
        (j.val + HEAD_HALF) * head_dim_stride
  writeMask := fun _o pid₀ _pid₁ _j => pid₀ < Q_HEAD_NUM
```
</details>

<details><summary><code>fusedV2CacheIO</code></summary>

```
/-- The **IO signature** of the cache face of `fused_rotary_embedding_kernel_v2`
on the chained-metadata surface: two `.nat` slots (`ContextLengths` then, eating
the first, `BLOCK_TABLES`), four float read channels (`k0`/`k1` + `Cos`/`Sin`),
and two paged-cache scatters over the flat `Fin HEAD_HALF` `dim` lane space, all
gated by the head predicate `pid₀ < Q_HEAD_NUM`. -/
```
```lean
def fusedV2CacheIO (KVCache K Cos Sin : RegionName)
    (BlockTables ContextLengths : Region .nat)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      cacheb_stride cacheh_stride cachebs_stride cached_stride bts_stride btb_stride
      block_size q_total_tokens Q_HEAD_NUM HEAD_HALF : Nat) :
    ChainMetaGroupedMasked2DKernelIO where
  kernel := fused_rotary_embedding_v2_cache_surface KVCache K Cos Sin BlockTables
    ContextLengths k_token_stride k_head_stride head_dim_stride cos_token_stride
    cos_stride cacheb_stride cacheh_stride cachebs_stride cached_stride bts_stride
    btb_stride block_size q_total_tokens Q_HEAD_NUM HEAD_HALF
  nIn := 4
  nOut := 2
  bufs := [KVCache, K, Cos, Sin, BlockTables, ContextLengths]
  mbuf1 := ContextLengths
  mbuf2 := BlockTables
  inp := fun i => match i with
    | ⟨0, _⟩ => K | ⟨1, _⟩ => K | ⟨2, _⟩ => Cos | ⟨_ + 3, _⟩ => Sin
  out := fun _ => KVCache
  B := HEAD_HALF
  mwin1 := fun _pid₀ pid₁ => pid₁
  mwin2 := fun _pid₀ pid₁ s1 =>
    pid₁ * bts_stride + (s1 - 1) / block_size * btb_stride
  read := fun i pid₀ pid₁ _s1 _s2 j => match i with
    | ⟨0, _⟩ => pid₁ * k_token_stride + pid₀ * k_head_stride + j.val * head_dim_stride
    | ⟨1, _⟩ => pid₁ * k_token_stride + pid₀ * k_head_stride +
        (j.val + HEAD_HALF) * head_dim_stride
    | ⟨2, _⟩ => pid₁ * cos_token_stride + j.val * cos_stride
    | ⟨_ + 3, _⟩ => pid₁ * cos_token_stride + j.val * cos_stride
  readMask := fun i pid₀ pid₁ _s1 _s2 _j => match i with
    | ⟨0, _⟩ => pid₀ < Q_HEAD_NUM
    | ⟨1, _⟩ => pid₀ < Q_HEAD_NUM
    | ⟨2, _⟩ => pid₁ < q_total_tokens
    | ⟨_ + 3, _⟩ => pid₁ < q_total_tokens
  write := fun o pid₀ pid₁ s1 s2 j => match o with
    | ⟨0, _⟩ => (if pid₁ < q_total_tokens then s2 * cacheb_stride else 0) +
        pid₀ * cacheh_stride + (s1 - 1) % block_size * cachebs_stride +
        j.val * cached_stride
    | ⟨_ + 1, _⟩ => (if pid₁ < q_total_tokens then s2 * cacheb_stride else 0) +
        pid₀ * cacheh_stride + (s1 - 1) % block_size * cachebs_stride +
        (j.val + HEAD_HALF) * cached_stride
  writeMask := fun _o pid₀ _pid₁ _s1 _s2 _j => pid₀ < Q_HEAD_NUM
```
</details>

<details><summary><code>v2BlockId</code></summary>

```
/-- Python v2 metadata: `block_ids = block_tables[token, last_block_idx]`. -/
```
```lean
def v2BlockId
    (s : BlockState) (BlockTables ContextLengths : RegionName)
    (bts_stride btb_stride block_size : Nat) : Nat :=
  s.readMemValue .nat BlockTables
    (s.pids 1 * bts_stride +
      v2LastBlockIdx s ContextLengths block_size * btb_stride)
```
</details>

<details><summary><code>v2OffsetsInLastBlock</code></summary>

```
/-- Python v2 metadata:
`offsets_in_last_block = (past_kv_seq_len % block_size) * cachebs_stride`. -/
```
```lean
def v2OffsetsInLastBlock
    (s : BlockState) (ContextLengths : RegionName)
    (block_size cachebs_stride : Nat) : Nat :=
  (v2PastKvSeqLen s ContextLengths % block_size) * cachebs_stride
```
</details>

<details><summary><code>rotary_embedding_q_surface</code></summary>

```
/-- Surface transcription of the Q part of `rotary_emb_nopad.py`'s
`rotary_embedding_kernel`.

This writes both rotary halves for Q over the full
`[BLOCK_TOKENS, 1, HEAD_HALF]` token/head/dimension tile. The conditional K
branch is represented by `rotary_embedding_k_surface`; the cache-writing v2
branch is represented by `fused_rotary_embedding_v2_surface`. -/
```
```lean
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
```
</details>

<details><summary><code>dataFirstP</code></summary>

```
/-- Q/K first-half output/read address of flat lane `j` for program
`(pid₀, pid₁)`. -/
```
```lean
def dataFirstP (pid₀ pid₁ token_stride head_stride head_dim_stride
    BLOCK_TOKENS HEAD_HALF : Nat) (j : Fin (BLOCK_TOKENS * HEAD_HALF)) : Nat :=
  (pid₁ * BLOCK_TOKENS + (Lane2D.decode j).1.val) * token_stride +
    pid₀ * head_stride + (Lane2D.decode j).2.1.val * head_dim_stride
```
</details>

<details><summary><code>dataSecondP</code></summary>

```
/-- Q/K second-half output/read address of flat lane `j`. -/
```
```lean
def dataSecondP (pid₀ pid₁ token_stride head_stride head_dim_stride
    BLOCK_TOKENS HEAD_HALF : Nat) (j : Fin (BLOCK_TOKENS * HEAD_HALF)) : Nat :=
  (pid₁ * BLOCK_TOKENS + (Lane2D.decode j).1.val) * token_stride +
    pid₀ * head_stride + ((Lane2D.decode j).2.1.val + HEAD_HALF) * head_dim_stride
```
</details>

<details><summary><code>cosP</code></summary>

```
/-- Cos/Sin read address of flat lane `j` (the trig tile broadcasts along the
inert head axis). -/
```
```lean
def cosP (pid₁ cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) : Nat :=
  (pid₁ * BLOCK_TOKENS + (Lane2D.decode j).1.val) * cos_token_stride +
    (Lane2D.decode j).2.1.val * cos_stride
```
</details>

<details><summary><code>activeQP</code></summary>

```
/-- Q active predicate of flat lane `j`: head in range and token in range. -/
```
```lean
def activeQP (pid₀ pid₁ q_total_tokens Q_HEAD_NUM BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) : Prop :=
  pid₀ < Q_HEAD_NUM ∧ pid₁ * BLOCK_TOKENS + (Lane2D.decode j).1.val < q_total_tokens
```
</details>

<details><summary><code>tokP</code></summary>

```
/-- Token-in-range predicate of flat lane `j` (the `Cos`/`Sin` load mask). -/
```
```lean
def tokP (pid₁ q_total_tokens BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) : Prop :=
  pid₁ * BLOCK_TOKENS + (Lane2D.decode j).1.val < q_total_tokens
```
</details>

<details><summary><code>rotary_embedding_k_surface</code></summary>

```
/-- Surface transcription of the conditional K part of
`rotary_emb_nopad.py`'s `rotary_embedding_kernel` over the full
`[BLOCK_TOKENS, 1, HEAD_HALF]` token/head/dimension tile. -/
```
```lean
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
```
</details>

<details><summary><code>activeKP</code></summary>

```
/-- K active predicate of flat lane `j`: the GQA-leader modular gate fires and
the token is in range. -/
```
```lean
def activeKP (pid₀ pid₁ q_total_tokens KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF : Nat)
    (j : Fin (BLOCK_TOKENS * HEAD_HALF)) : Prop :=
  pid₀ % KV_GROUP_NUM = 0 ∧
    pid₁ * BLOCK_TOKENS + (Lane2D.decode j).1.val < q_total_tokens
```
</details>

<details><summary><code>fused_rotary_embedding_v2_q_surface</code></summary>

```
/-- Standalone Q face of `fused_rotary_embedding_kernel_v2`: the two in-place
`Q` half-stores over the 1-D `Fin HEAD_HALF` `dim` tile, head-gated by
`block_head_index < Q_HEAD_NUM`, with `Cos`/`Sin` loaded under the per-program
`block_token_index < q_total_tokens` mask (padded to `0`). -/
```
```lean
def fused_rotary_embedding_v2_q_surface
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF : Nat) :
    ComputeKernel := triton {
  block_head_index = tl.program_id(0)
  block_token_index = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HEAD_HALF))
  dim_range1 = dim_range0 + $(HEAD_HALF)
  off_cos_sin = block_token_index * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(Cos + off_cos_sin,
    mask=block_token_index < $(q_total_tokens), other=0.0)
  loaded_sin = tl.load(Sin + off_cos_sin,
    mask=block_token_index < $(q_total_tokens), other=0.0)
  off_q0 = block_token_index * $(q_token_stride) +
    block_head_index * $(q_head_stride) + dim_range0 * $(head_dim_stride)
  off_q1 = block_token_index * $(q_token_stride) +
    block_head_index * $(q_head_stride) + dim_range1 * $(head_dim_stride)
  loaded_q0 = tl.load(Q + off_q0,
    mask=block_head_index < $(Q_HEAD_NUM), other=0.0)
  loaded_q1 = tl.load(Q + off_q1,
    mask=block_head_index < $(Q_HEAD_NUM), other=0.0)
  out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
  out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
  tl.store(Q + off_q0, out_q0, mask=block_head_index < $(Q_HEAD_NUM))
  tl.store(Q + off_q1, out_q1, mask=block_head_index < $(Q_HEAD_NUM))
}
```
</details>

<details><summary><code>fused_rotary_embedding_v2_cache_surface</code></summary>

```
/-- Standalone paged-KV cache face of `fused_rotary_embedding_kernel_v2`, over
the 1-D `Fin HEAD_HALF` `dim` tile, head-gated by `block_head_index < Q_HEAD_NUM`,
with the chained `context_lengths → BLOCK_TABLES` metadata loads. -/
```
```lean
def fused_rotary_embedding_v2_cache_surface
    (KVCache K Cos Sin : RegionName) (BlockTables ContextLengths : Region .nat)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      cacheb_stride cacheh_stride cachebs_stride cached_stride bts_stride btb_stride
      block_size q_total_tokens Q_HEAD_NUM HEAD_HALF : Nat) :
    ComputeKernel := triton {
  block_head_index = tl.program_id(0)
  block_token_index = tl.program_id(1)
  dim = tl.arange(0, $(HEAD_HALF))
  dim1 = dim + $(HEAD_HALF)
  off_cos_sin = block_token_index * $(cos_token_stride) + dim * $(cos_stride)
  loaded_cos = tl.load(Cos + off_cos_sin,
    mask=block_token_index < $(q_total_tokens), other=0.0)
  loaded_sin = tl.load(Sin + off_cos_sin,
    mask=block_token_index < $(q_total_tokens), other=0.0)
  off_k0 = block_token_index * $(k_token_stride) +
    block_head_index * $(k_head_stride) + dim * $(head_dim_stride)
  off_k1 = block_token_index * $(k_token_stride) +
    block_head_index * $(k_head_stride) + dim1 * $(head_dim_stride)
  loaded_k0 = tl.load(K + off_k0, mask=block_head_index < $(Q_HEAD_NUM), other=0.0)
  loaded_k1 = tl.load(K + off_k1, mask=block_head_index < $(Q_HEAD_NUM), other=0.0)
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
  tl.store(KVCache + kv_range0, out_k0, mask=block_head_index < $(Q_HEAD_NUM))
  tl.store(KVCache + kv_range1, out_k1, mask=block_head_index < $(Q_HEAD_NUM))
}
```
</details>

<details><summary><code>v2LastBlockIdx</code></summary>

```
/-- Python v2 metadata: `last_block_idx = past_kv_seq_len // block_size`. -/
```
```lean
def v2LastBlockIdx
    (s : BlockState) (ContextLengths : RegionName) (block_size : Nat) : Nat :=
  v2PastKvSeqLen s ContextLengths / block_size
```
</details>

<details><summary><code>v2PastKvSeqLen</code></summary>

```
/-- Python v2 metadata: `past_kv_seq_len =
context_lengths[block_token_index] - 1`. -/
```
```lean
def v2PastKvSeqLen (s : BlockState) (ContextLengths : RegionName) : Nat :=
  s.readMemValue .nat ContextLengths (s.pids 1) - 1
```
</details>

## Also present (pinned special-case summaries)
- `rotary_embedding_q0_block_compute_correct`
- `rotary_embedding_q1_block_compute_correct`
- `rotary_embedding_k0_block_compute_correct`
- `rotary_embedding_k1_block_compute_correct`
- `fused_rotary_v2_kv_cache_first_half_store_slice_compute_correct`
- `fused_rotary_v2_kv_cache_second_half_store_slice_compute_correct`
- `fused_rotary_v2_context_kv_cache_first_half_store_slice_compute_correct`
- `fused_rotary_v2_context_kv_cache_second_half_store_slice_compute_correct`
- `fused_rotary_v2_q_first_half_store_slice_compute_correct`
- `fused_rotary_v2_q_second_half_store_slice_compute_correct`
- `rotary_embedding_q_surface_q0_compute_correct`
- `rotary_embedding_q_surface_q1_compute_correct`
- `rotary_embedding_k_surface_k0_compute_correct`
- `rotary_embedding_k_surface_k1_compute_correct`
