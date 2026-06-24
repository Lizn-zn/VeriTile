# Spec sheet — `bench/tritonbench_g/rotary_emb_nopad/RotaryEmbNopad.lean`

**Python source:** `bench/tritonbench_g/rotary_emb_nopad/rotary_emb_nopad.py`

## Public theorem: `rotary_nopad_python_case1_all_outputs_surface_summary`

<details><summary>docstring</summary>

```
/-- Public Python no-cache case coverage summary: the full Q/K rotary surface
lowers, and all four Q/K half-output stores realize the checked tensor strides. -/
```
</details>

**Statement:**
```lean
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
        rotaryNopadK1FullSpec s K Cos Sin 256 64 1 64 1 2 4 32 idx))
```

**Assumptions / layout contracts:**
- `kernel : = rotary_embedding_q_surface Q Cos Sin
        512 64 1 64 1 32 8 32 4`
- `initialState : = s`
- `fun idx : TileIndex [4, 1, 32] => activeFull s 32 8 4 idx`
- `expected : = fun idx =>
        rotaryNopadQ0FullSpec s Q Cos Sin 512 64 1 64 1 4 32 idx`
- `kernel : = rotary_embedding_q_surface Q Cos Sin
        512 64 1 64 1 32 8 32 4`
- `initialState : = s`
- `fun idx : TileIndex [4, 1, 32] => activeFull s 32 8 4 idx`
- `expected : = fun idx =>
        rotaryNopadQ1FullSpec s Q Cos Sin 512 64 1 64 1 4 32 idx`
- `kernel : = rotary_embedding_k_surface K Cos Sin
        256 64 1 64 1 32 2 32 4`
- `initialState : = s`
- `fun idx : TileIndex [4, 1, 32] => activeKFull s 32 2 4 idx`
- `expected : = fun idx =>
        rotaryNopadK0FullSpec s K Cos Sin 256 64 1 64 1 2 4 32 idx`
- `kernel : = rotary_embedding_k_surface K Cos Sin
        256 64 1 64 1 32 2 32 4`
- `initialState : = s`
- `fun idx : TileIndex [4, 1, 32] => activeKFull s 32 2 4 idx`
- `expected : = fun idx =>
        rotaryNopadK1FullSpec s K Cos Sin 256 64 1 64 1 2 4 32 idx`

**Closed-form spec defs (transitive):** `rotary_embedding_kernel_surface`, `rotary_embedding_q_surface`, `activeFull`, `qFullFirstOffset`, `rotaryNopadQ0FullSpec`, `qFullSecondOffset`, `rotaryNopadQ1FullSpec`, `rotary_embedding_k_surface`, `activeKFull`, `kFullFirstOffset`, `rotaryNopadK0FullSpec`, `kFullSecondOffset`, `rotaryNopadK1FullSpec`, `cosFullOffset`

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
