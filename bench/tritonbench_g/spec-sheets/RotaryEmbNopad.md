# Spec sheet — `bench/tritonbench_g/rotary_emb_nopad/RotaryEmbNopad.lean`

**Python source:** `bench/tritonbench_g/rotary_emb_nopad/rotary_emb_nopad.py`

## Public theorem: `rotary_emb_nopad_output_summary_general`

<details><summary>docstring</summary>

```
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
```
</details>

**Statement:**
```lean
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
          BLOCK_TOKENS HEAD_HALF idx))
```

**Assumptions / layout contracts:**
- `hQ0Inj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS idx)`
- `hQ1Inj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS HEAD_HALF idx)`
- `hQDisjoint : ∀ idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS idx ≠
          qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF idx'`
- `hK0Inj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
          KV_GROUP_NUM BLOCK_TOKENS idx)`
- `hK1Inj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
          KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx)`
- `hKDisjoint : ∀ idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
        kFullFirstOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS idx ≠
          kFullSecondOffset s k_token_stride k_head_stride head_dim_stride
            KV_GROUP_NUM BLOCK_TOKENS HEAD_HALF idx'`
- `fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
          activeFull s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS idx`
- `fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
          activeFull s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS idx`
- `fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
          activeKFull s k_q_total_tokens KV_GROUP_NUM BLOCK_TOKENS idx`
- `fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
          activeKFull s k_q_total_tokens KV_GROUP_NUM BLOCK_TOKENS idx`

**Closed-form spec defs (transitive):** `qFullFirstOffset`, `qFullSecondOffset`, `kFullFirstOffset`, `kFullSecondOffset`, `rotary_embedding_kernel_surface`, `rotary_embedding_q_surface`, `activeFull`, `rotaryNopadQ0FullSpec`, `rotaryNopadQ1FullSpec`, `rotary_embedding_k_surface`, `activeKFull`, `rotaryNopadK0FullSpec`, `rotaryNopadK1FullSpec`, `cosFullOffset`

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

<details><summary><code>activeFull</code></summary>

```
/-- Active predicate for the Q stores in the full Q surface kernel: head
index in range AND the per-tile token index in range. -/
```
```lean
def activeFull (s : BlockState) (q_total_tokens Q_HEAD_NUM BLOCK_TOKENS : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Prop :=
  s.pids 0 < Q_HEAD_NUM ∧
    s.pids 1 * BLOCK_TOKENS + idx.1.val < q_total_tokens
```
</details>

<details><summary><code>rotaryNopadQ0FullSpec</code></summary>

```
/-- Q first-half rotary spec for the full Q surface kernel:
`q0 * cos - q1 * sin`. -/
```
```lean
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
```
</details>

<details><summary><code>rotaryNopadQ1FullSpec</code></summary>

```
/-- Q second-half rotary spec for the full Q surface kernel:
`q0 * sin + q1 * cos`. -/
```
```lean
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

<details><summary><code>activeKFull</code></summary>

```
/-- Active predicate for the K stores in the full K surface kernel: the modular
KV-group gate fires and the per-tile token index is in range. -/
```
```lean
def activeKFull (s : BlockState) (q_total_tokens KV_GROUP_NUM BLOCK_TOKENS : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Prop :=
  s.pids 0 % KV_GROUP_NUM = 0 ∧
    s.pids 1 * BLOCK_TOKENS + idx.1.val < q_total_tokens
```
</details>

<details><summary><code>rotaryNopadK0FullSpec</code></summary>

```
/-- K first-half rotary spec for the full K surface kernel:
`k0 * cos - k1 * sin`. -/
```
```lean
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
```
</details>

<details><summary><code>rotaryNopadK1FullSpec</code></summary>

```
/-- K second-half rotary spec for the full K surface kernel:
`k0 * sin + k1 * cos`. -/
```
```lean
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
```
</details>

<details><summary><code>cosFullOffset</code></summary>

```
/-- Per-tile-index cos/sin offset (the cos/sin tile is `[BLOCK_TOKENS,
HEAD_HALF]` and broadcasts into the 3D Q tile along the singleton head
axis). -/
```
```lean
def cosFullOffset
    (s : BlockState) (cos_token_stride cos_stride BLOCK_TOKENS : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Nat :=
  (s.pids 1 * BLOCK_TOKENS + idx.1.val) * cos_token_stride +
    idx.2.2.1.val * cos_stride
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
- `rotary_nopad_python_case1_all_outputs_surface_summary`
