# Spec sheet — `bench/tritonbench_g/mixed_sparse_attention/MixedSparseAttention.lean`

**Python source:** `bench/tritonbench_g/mixed_sparse_attention/mixed_sparse_attention.py`

## Public theorem: `mixed_sparse_attention_python_case1_store_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 1 summary: full surface plus seqlen-masked output store. -/
```
</details>

**Statement:**
```lean
theorem mixed_sparse_attention_python_case1_store_summary
    (Q K V Out Acc : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState) :
    (∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
      (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 64 64 64 FloatDType.fp16).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_output_store_slice Acc Seqlens Out 4
        32768 8192 64 1 32768 8192 64 1 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 4 Seqlens 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        accStoreValue s Acc Seqlens 4 32768 8192 64 1 64 idx))
```

**Assumptions / layout contracts:**
- `fun idx : TileIndex [64, 64] => active s 4 Seqlens 64 idx`
- `fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)`

**Closed-form spec defs (transitive):** `mixed_sparse_attention_fwd_kernel_surface`, `mixed_sparse_attention_output_store_slice`, `active`, `outOffset`, `accStoreValue`, `mIndex`, `seqLen`, `offZ`, `offH`, `dIndex`, `accOffset`

<details><summary><code>mixed_sparse_attention_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `mixed_sparse_attention.py`'s
`_triton_mixed_sparse_attn_fwd_kernel`. -/
```
```lean
def mixed_sparse_attention_fwd_kernel_surface
    (Q K V : RegionName) (seqlens : Region .nat) (sm_scale : ℝ)
    (block_count block_offset column_count column_index : Region .nat)
    (Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vn stride_vk
      stride_oz stride_oh stride_om stride_ok
      Z H N_CTX NUM_ROWS NNZ_S NNZ_V
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat)
    (dtype : FloatDType) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  seqlen = tl.load(seqlens + off_hz // $(H))
  if start_m * $(BLOCK_M) >= seqlen {
    return
  }

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))

  qo_offset = (off_hz // $(H)) * $(stride_qz) + (off_hz % $(H)) * $(stride_qh)
  kv_offset = (off_hz // $(H)) * $(stride_kz) + (off_hz % $(H)) * $(stride_kh)

  q_ptrs = Q + qo_offset + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  k_ptrs = K + kv_offset + offs_d[:, None] * $(stride_kk)
  v_ptrs = V + kv_offset + offs_d[None, :] * $(stride_vk)
  o_ptrs = Out + qo_offset + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_ok)

  num_blks = tl.load(block_count + off_hz * $(NUM_ROWS) + start_m)
  blks_ptr = block_offset + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_S)
  num_cols = tl.load(column_count + off_hz * $(NUM_ROWS) + start_m)
  cols_ptr = column_index + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_V)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(q_ptrs)
  q = (q * qk_scale).to(DTYPE)

  m_mask = offs_m[:, None] < seqlen

  max_num_blks = $(8)
  for block_index in range(max_num_blks) {
    cond = block_index < num_blks
    start_n = tl.load(blks_ptr + block_index, mask=cond)
    cols = start_n + offs_n
    n_mask = (cols < seqlen) & cond
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    causal_mask = cols[None, :] <= offs_m[:, None]
    qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  max_num_cols = $(16)
  for start_n in range($(0), max_num_cols, $(BLOCK_N)) {
    cond = start_n < num_cols
    n_mask = (start_n + offs_n < num_cols) & cond
    cols = tl.load(cols_ptr + start_n + offs_n, mask=cond[:, None], other=0)
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk = tl.where(m_mask & n_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  acc /= l_i[:, None]
  tl.store(o_ptrs, (acc).to(DTYPE), mask=m_mask)
}
```
</details>

<details><summary><code>mixed_sparse_attention_output_store_slice</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of
`mixed_sparse_attention.py`'s `_triton_mixed_sparse_attn_fwd_kernel`.

The full kernel combines block-sparse and column-sparse attention updates. This
slice starts from a precomputed normalized `Acc` tile and proves the final
`seqlens`-masked writeback into `Out`. The kernel-level early return for
`start_m * BLOCK_M >= seqlen` is represented at this surface by the same
all-false row mask; the sparse block/column softmax loops remain separate
modeling work, including their `tl.float32` accumulators. -/
```
```lean
def mixed_sparse_attention_output_store_slice
    (Acc : RegionName) (Seqlens : Region .nat) (Out : RegionName)
    (H
      stride_acc_z stride_acc_h stride_acc_m stride_acc_d
      stride_qz stride_qh stride_om stride_ok
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  seqlen = tl.load(Seqlens + off_z)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < seqlen) & (offs_d[None, :] < $(BLOCK_DMODEL))
  acc = tl.load(Acc + off_z * $(stride_acc_z) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_ok),
      (acc).to(Out.dtype.element_ty), mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (H : Nat) (Seqlens : RegionName) (BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H Seqlens
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_om stride_ok BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_om + dIndex idx * stride_ok
```
</details>

<details><summary><code>accStoreValue</code></summary>

```lean
noncomputable def accStoreValue
    (s : BlockState) (Acc Seqlens : RegionName)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s H Seqlens BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (H : Nat) (Seqlens : RegionName) : Nat :=
  s.readMemValue .nat Seqlens (offZ s H)
```
</details>

<details><summary><code>offZ</code></summary>

```lean
def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>offH</code></summary>

```lean
def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>accOffset</code></summary>

```lean
def accOffset
    (s : BlockState)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_acc_z + offH s H * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d
```
</details>

## Public theorem: `mixed_sparse_attention_python_case2_store_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 2 summary. -/
```
</details>

**Statement:**
```lean
theorem mixed_sparse_attention_python_case2_store_summary
    (Q K V Out Acc : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState) :
    (∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
      (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 32 32 64 FloatDType.fp16).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_output_store_slice Acc Seqlens Out 4
        32768 8192 64 1 32768 8192 64 1 32 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [32, 64] => active s 4 Seqlens 32 idx)
        (fun idx : TileIndex [32, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 32 idx)))
      (expected := fun idx : TileIndex [32, 64] =>
        accStoreValue s Acc Seqlens 4 32768 8192 64 1 32 idx))
```

**Assumptions / layout contracts:**
- `fun idx : TileIndex [32, 64] => active s 4 Seqlens 32 idx`
- `fun idx : TileIndex [32, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 32 idx)`

**Closed-form spec defs (transitive):** `mixed_sparse_attention_fwd_kernel_surface`, `mixed_sparse_attention_output_store_slice`, `active`, `outOffset`, `accStoreValue`, `mIndex`, `seqLen`, `offZ`, `offH`, `dIndex`, `accOffset`

<details><summary><code>mixed_sparse_attention_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `mixed_sparse_attention.py`'s
`_triton_mixed_sparse_attn_fwd_kernel`. -/
```
```lean
def mixed_sparse_attention_fwd_kernel_surface
    (Q K V : RegionName) (seqlens : Region .nat) (sm_scale : ℝ)
    (block_count block_offset column_count column_index : Region .nat)
    (Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vn stride_vk
      stride_oz stride_oh stride_om stride_ok
      Z H N_CTX NUM_ROWS NNZ_S NNZ_V
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat)
    (dtype : FloatDType) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  seqlen = tl.load(seqlens + off_hz // $(H))
  if start_m * $(BLOCK_M) >= seqlen {
    return
  }

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))

  qo_offset = (off_hz // $(H)) * $(stride_qz) + (off_hz % $(H)) * $(stride_qh)
  kv_offset = (off_hz // $(H)) * $(stride_kz) + (off_hz % $(H)) * $(stride_kh)

  q_ptrs = Q + qo_offset + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  k_ptrs = K + kv_offset + offs_d[:, None] * $(stride_kk)
  v_ptrs = V + kv_offset + offs_d[None, :] * $(stride_vk)
  o_ptrs = Out + qo_offset + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_ok)

  num_blks = tl.load(block_count + off_hz * $(NUM_ROWS) + start_m)
  blks_ptr = block_offset + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_S)
  num_cols = tl.load(column_count + off_hz * $(NUM_ROWS) + start_m)
  cols_ptr = column_index + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_V)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(q_ptrs)
  q = (q * qk_scale).to(DTYPE)

  m_mask = offs_m[:, None] < seqlen

  max_num_blks = $(8)
  for block_index in range(max_num_blks) {
    cond = block_index < num_blks
    start_n = tl.load(blks_ptr + block_index, mask=cond)
    cols = start_n + offs_n
    n_mask = (cols < seqlen) & cond
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    causal_mask = cols[None, :] <= offs_m[:, None]
    qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  max_num_cols = $(16)
  for start_n in range($(0), max_num_cols, $(BLOCK_N)) {
    cond = start_n < num_cols
    n_mask = (start_n + offs_n < num_cols) & cond
    cols = tl.load(cols_ptr + start_n + offs_n, mask=cond[:, None], other=0)
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk = tl.where(m_mask & n_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  acc /= l_i[:, None]
  tl.store(o_ptrs, (acc).to(DTYPE), mask=m_mask)
}
```
</details>

<details><summary><code>mixed_sparse_attention_output_store_slice</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of
`mixed_sparse_attention.py`'s `_triton_mixed_sparse_attn_fwd_kernel`.

The full kernel combines block-sparse and column-sparse attention updates. This
slice starts from a precomputed normalized `Acc` tile and proves the final
`seqlens`-masked writeback into `Out`. The kernel-level early return for
`start_m * BLOCK_M >= seqlen` is represented at this surface by the same
all-false row mask; the sparse block/column softmax loops remain separate
modeling work, including their `tl.float32` accumulators. -/
```
```lean
def mixed_sparse_attention_output_store_slice
    (Acc : RegionName) (Seqlens : Region .nat) (Out : RegionName)
    (H
      stride_acc_z stride_acc_h stride_acc_m stride_acc_d
      stride_qz stride_qh stride_om stride_ok
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  seqlen = tl.load(Seqlens + off_z)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < seqlen) & (offs_d[None, :] < $(BLOCK_DMODEL))
  acc = tl.load(Acc + off_z * $(stride_acc_z) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_ok),
      (acc).to(Out.dtype.element_ty), mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (H : Nat) (Seqlens : RegionName) (BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H Seqlens
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_om stride_ok BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_om + dIndex idx * stride_ok
```
</details>

<details><summary><code>accStoreValue</code></summary>

```lean
noncomputable def accStoreValue
    (s : BlockState) (Acc Seqlens : RegionName)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s H Seqlens BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (H : Nat) (Seqlens : RegionName) : Nat :=
  s.readMemValue .nat Seqlens (offZ s H)
```
</details>

<details><summary><code>offZ</code></summary>

```lean
def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>offH</code></summary>

```lean
def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>accOffset</code></summary>

```lean
def accOffset
    (s : BlockState)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_acc_z + offH s H * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d
```
</details>

## Public theorem: `mixed_sparse_attention_python_case3_store_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 3 summary. -/
```
</details>

**Statement:**
```lean
theorem mixed_sparse_attention_python_case3_store_summary
    (Q K V Out Acc : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState) :
    (∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
      (0.2 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 64 64 64 FloatDType.fp16).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_output_store_slice Acc Seqlens Out 4
        32768 8192 64 1 32768 8192 64 1 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 4 Seqlens 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        accStoreValue s Acc Seqlens 4 32768 8192 64 1 64 idx))
```

**Assumptions / layout contracts:**
- `fun idx : TileIndex [64, 64] => active s 4 Seqlens 64 idx`
- `fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)`

**Closed-form spec defs (transitive):** `mixed_sparse_attention_fwd_kernel_surface`, `mixed_sparse_attention_output_store_slice`, `active`, `outOffset`, `accStoreValue`, `mIndex`, `seqLen`, `offZ`, `offH`, `dIndex`, `accOffset`

<details><summary><code>mixed_sparse_attention_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `mixed_sparse_attention.py`'s
`_triton_mixed_sparse_attn_fwd_kernel`. -/
```
```lean
def mixed_sparse_attention_fwd_kernel_surface
    (Q K V : RegionName) (seqlens : Region .nat) (sm_scale : ℝ)
    (block_count block_offset column_count column_index : Region .nat)
    (Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vn stride_vk
      stride_oz stride_oh stride_om stride_ok
      Z H N_CTX NUM_ROWS NNZ_S NNZ_V
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat)
    (dtype : FloatDType) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  seqlen = tl.load(seqlens + off_hz // $(H))
  if start_m * $(BLOCK_M) >= seqlen {
    return
  }

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))

  qo_offset = (off_hz // $(H)) * $(stride_qz) + (off_hz % $(H)) * $(stride_qh)
  kv_offset = (off_hz // $(H)) * $(stride_kz) + (off_hz % $(H)) * $(stride_kh)

  q_ptrs = Q + qo_offset + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  k_ptrs = K + kv_offset + offs_d[:, None] * $(stride_kk)
  v_ptrs = V + kv_offset + offs_d[None, :] * $(stride_vk)
  o_ptrs = Out + qo_offset + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_ok)

  num_blks = tl.load(block_count + off_hz * $(NUM_ROWS) + start_m)
  blks_ptr = block_offset + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_S)
  num_cols = tl.load(column_count + off_hz * $(NUM_ROWS) + start_m)
  cols_ptr = column_index + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_V)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(q_ptrs)
  q = (q * qk_scale).to(DTYPE)

  m_mask = offs_m[:, None] < seqlen

  max_num_blks = $(8)
  for block_index in range(max_num_blks) {
    cond = block_index < num_blks
    start_n = tl.load(blks_ptr + block_index, mask=cond)
    cols = start_n + offs_n
    n_mask = (cols < seqlen) & cond
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    causal_mask = cols[None, :] <= offs_m[:, None]
    qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  max_num_cols = $(16)
  for start_n in range($(0), max_num_cols, $(BLOCK_N)) {
    cond = start_n < num_cols
    n_mask = (start_n + offs_n < num_cols) & cond
    cols = tl.load(cols_ptr + start_n + offs_n, mask=cond[:, None], other=0)
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk = tl.where(m_mask & n_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  acc /= l_i[:, None]
  tl.store(o_ptrs, (acc).to(DTYPE), mask=m_mask)
}
```
</details>

<details><summary><code>mixed_sparse_attention_output_store_slice</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of
`mixed_sparse_attention.py`'s `_triton_mixed_sparse_attn_fwd_kernel`.

The full kernel combines block-sparse and column-sparse attention updates. This
slice starts from a precomputed normalized `Acc` tile and proves the final
`seqlens`-masked writeback into `Out`. The kernel-level early return for
`start_m * BLOCK_M >= seqlen` is represented at this surface by the same
all-false row mask; the sparse block/column softmax loops remain separate
modeling work, including their `tl.float32` accumulators. -/
```
```lean
def mixed_sparse_attention_output_store_slice
    (Acc : RegionName) (Seqlens : Region .nat) (Out : RegionName)
    (H
      stride_acc_z stride_acc_h stride_acc_m stride_acc_d
      stride_qz stride_qh stride_om stride_ok
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  seqlen = tl.load(Seqlens + off_z)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < seqlen) & (offs_d[None, :] < $(BLOCK_DMODEL))
  acc = tl.load(Acc + off_z * $(stride_acc_z) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_ok),
      (acc).to(Out.dtype.element_ty), mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (H : Nat) (Seqlens : RegionName) (BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H Seqlens
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_om stride_ok BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_om + dIndex idx * stride_ok
```
</details>

<details><summary><code>accStoreValue</code></summary>

```lean
noncomputable def accStoreValue
    (s : BlockState) (Acc Seqlens : RegionName)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s H Seqlens BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (H : Nat) (Seqlens : RegionName) : Nat :=
  s.readMemValue .nat Seqlens (offZ s H)
```
</details>

<details><summary><code>offZ</code></summary>

```lean
def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>offH</code></summary>

```lean
def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>accOffset</code></summary>

```lean
def accOffset
    (s : BlockState)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_acc_z + offH s H * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d
```
</details>

## Public theorem: `mixed_sparse_attention_python_case4_store_summary`

<details><summary>docstring</summary>

```
/-- Public Python case 4 summary. -/
```
</details>

**Statement:**
```lean
theorem mixed_sparse_attention_python_case4_store_summary
    (Q K V Out Acc : RegionName)
    (SeqlensAlt Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState) :
    (∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V SeqlensAlt
      (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 64 64 64 FloatDType.fp16).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_output_store_slice Acc SeqlensAlt Out 4
        32768 8192 64 1 32768 8192 64 1 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 4 SeqlensAlt 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        accStoreValue s Acc SeqlensAlt 4 32768 8192 64 1 64 idx))
```

**Assumptions / layout contracts:**
- `fun idx : TileIndex [64, 64] => active s 4 SeqlensAlt 64 idx`
- `fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)`

**Closed-form spec defs (transitive):** `mixed_sparse_attention_fwd_kernel_surface`, `mixed_sparse_attention_output_store_slice`, `active`, `outOffset`, `accStoreValue`, `mIndex`, `seqLen`, `offZ`, `offH`, `dIndex`, `accOffset`

<details><summary><code>mixed_sparse_attention_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `mixed_sparse_attention.py`'s
`_triton_mixed_sparse_attn_fwd_kernel`. -/
```
```lean
def mixed_sparse_attention_fwd_kernel_surface
    (Q K V : RegionName) (seqlens : Region .nat) (sm_scale : ℝ)
    (block_count block_offset column_count column_index : Region .nat)
    (Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vn stride_vk
      stride_oz stride_oh stride_om stride_ok
      Z H N_CTX NUM_ROWS NNZ_S NNZ_V
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat)
    (dtype : FloatDType) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  seqlen = tl.load(seqlens + off_hz // $(H))
  if start_m * $(BLOCK_M) >= seqlen {
    return
  }

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))

  qo_offset = (off_hz // $(H)) * $(stride_qz) + (off_hz % $(H)) * $(stride_qh)
  kv_offset = (off_hz // $(H)) * $(stride_kz) + (off_hz % $(H)) * $(stride_kh)

  q_ptrs = Q + qo_offset + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  k_ptrs = K + kv_offset + offs_d[:, None] * $(stride_kk)
  v_ptrs = V + kv_offset + offs_d[None, :] * $(stride_vk)
  o_ptrs = Out + qo_offset + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_ok)

  num_blks = tl.load(block_count + off_hz * $(NUM_ROWS) + start_m)
  blks_ptr = block_offset + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_S)
  num_cols = tl.load(column_count + off_hz * $(NUM_ROWS) + start_m)
  cols_ptr = column_index + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_V)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(q_ptrs)
  q = (q * qk_scale).to(DTYPE)

  m_mask = offs_m[:, None] < seqlen

  max_num_blks = $(8)
  for block_index in range(max_num_blks) {
    cond = block_index < num_blks
    start_n = tl.load(blks_ptr + block_index, mask=cond)
    cols = start_n + offs_n
    n_mask = (cols < seqlen) & cond
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    causal_mask = cols[None, :] <= offs_m[:, None]
    qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  max_num_cols = $(16)
  for start_n in range($(0), max_num_cols, $(BLOCK_N)) {
    cond = start_n < num_cols
    n_mask = (start_n + offs_n < num_cols) & cond
    cols = tl.load(cols_ptr + start_n + offs_n, mask=cond[:, None], other=0)
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk = tl.where(m_mask & n_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  acc /= l_i[:, None]
  tl.store(o_ptrs, (acc).to(DTYPE), mask=m_mask)
}
```
</details>

<details><summary><code>mixed_sparse_attention_output_store_slice</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of
`mixed_sparse_attention.py`'s `_triton_mixed_sparse_attn_fwd_kernel`.

The full kernel combines block-sparse and column-sparse attention updates. This
slice starts from a precomputed normalized `Acc` tile and proves the final
`seqlens`-masked writeback into `Out`. The kernel-level early return for
`start_m * BLOCK_M >= seqlen` is represented at this surface by the same
all-false row mask; the sparse block/column softmax loops remain separate
modeling work, including their `tl.float32` accumulators. -/
```
```lean
def mixed_sparse_attention_output_store_slice
    (Acc : RegionName) (Seqlens : Region .nat) (Out : RegionName)
    (H
      stride_acc_z stride_acc_h stride_acc_m stride_acc_d
      stride_qz stride_qh stride_om stride_ok
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  seqlen = tl.load(Seqlens + off_z)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < seqlen) & (offs_d[None, :] < $(BLOCK_DMODEL))
  acc = tl.load(Acc + off_z * $(stride_acc_z) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_ok),
      (acc).to(Out.dtype.element_ty), mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (H : Nat) (Seqlens : RegionName) (BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H Seqlens
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_om stride_ok BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_om + dIndex idx * stride_ok
```
</details>

<details><summary><code>accStoreValue</code></summary>

```lean
noncomputable def accStoreValue
    (s : BlockState) (Acc Seqlens : RegionName)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s H Seqlens BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (H : Nat) (Seqlens : RegionName) : Nat :=
  s.readMemValue .nat Seqlens (offZ s H)
```
</details>

<details><summary><code>offZ</code></summary>

```lean
def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>offH</code></summary>

```lean
def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>accOffset</code></summary>

```lean
def accOffset
    (s : BlockState)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_acc_z + offH s H * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d
```
</details>

## Public theorem: `mixed_sparse_attention_python_case1_output_closed_form_summary`

<details><summary>docstring</summary>

```
/-- **Genuine Python case 1 closed-form summary** (`BLOCK_M=BLOCK_N=64`,
`sm_scale=0.1`). The executed surface kernel's `fp16` `Out` cell at every active
output lane equals `some (mixedSparseAttnClosedForm …)`. -/
```
</details>

**Statement:**
```lean
theorem mixed_sparse_attention_python_case1_output_closed_form_summary
    (Q K V Out : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hactive : s.pids 0 * 64 < seqLen s 4 (Region.cast Seqlens))
    (hNC64 : s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * 2 + s.pids 0) ≤ 64)
    (hpos : ∀ i : Fin 64, s.pids 0 * 64 + i.val < seqLen s 4 (Region.cast Seqlens) →
      0 < msaDenomUpto 64 64
        (msaCatScore0 Q K V Seqlens Blocks BlockOffsets ColCounts Cols s) 9 i) :
    ∃ sF, exec (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
        (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
        32768 8192 64 1 32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
        2 4 128 2 4 8 64 64 64 FloatDType.fp16).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [64, 64],
          active s 4 Seqlens 64 idx →
            sF.readMemValue .fp16 Out (outOffset s 4 32768 8192 64 1 64 idx)
              = (some (mixedSparseAttnClosedForm s Q K V BlockOffsets Cols 4
                  32768 8192 64 32768 8192 64 32768 8192 64 2 4 8
                  (s.readMemValue .nat (Region.cast Blocks) (s.pids 1 * 2 + s.pids 0))
                  (s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * 2 + s.pids 0))
                  (seqLen s 4 (Region.cast Seqlens)) 64 64 64 0.1 idx.1 (dIndex idx)) : WithBot ℝ)
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg o, s.undef rg o = 0`
- `hactive : s.pids 0 * 64 < seqLen s 4 (Region.cast Seqlens)`
- `hNC64 : s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * 2 + s.pids 0) ≤ 64`
- `hpos : ∀ i : Fin 64, s.pids 0 * 64 + i.val < seqLen s 4 (Region.cast Seqlens) →
      0 < msaDenomUpto 64 64
        (msaCatScore0 Q K V Seqlens Blocks BlockOffsets ColCounts Cols s) 9 i`

**Closed-form spec defs (transitive):** `seqLen`, `msaDenomUpto`, `msaCatScore0`, `mixed_sparse_attention_fwd_kernel_surface`, `active`, `outOffset`, `mixedSparseAttnClosedForm`, `dIndex`, `offZ`, `msaE`, `msaCatScore`, `msaScoreA0`, `msaQVal`, `msaKPtr`, `msaScoreB0`, `mIndex`, `offH`, `rawScore`, `blockStartN`, `effScale`, `vRow`, `colKeyGlobal`, `msaScoreLaneA`, `msaSN0`, `msaScoreLaneB`, `msaGcol0`, `qRow`, `kRow`, `qoBase`, `msaKLaneA`, `msaKLaneB`, `msaColLaneB`

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (H : Nat) (Seqlens : RegionName) : Nat :=
  s.readMemValue .nat Seqlens (offZ s H)
```
</details>

<details><summary><code>msaDenomUpto</code></summary>

```
/-- Direct (unshifted) running denominator: `Σ_{l<k} Σ_j exp2(score l i j)`. -/
```
```lean
noncomputable def msaDenomUpto (BM BN : Nat)
    (score : Nat → Fin BM → Fin BN → WithBot ℝ) (k : Nat) (i : Fin BM) : ℝ :=
  (Finset.range k).sum (fun l =>
    (Finset.univ : Finset (Fin BN)).sum (fun j => msaE (score l i j)))
```
</details>

<details><summary><code>msaCatScore0</code></summary>

```
/-- The two pinned cat streams used by the loop assembly. -/
```
```lean
noncomputable abbrev msaCatScore0
    (Q K V : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (s0 : BlockState) (sm_scale : ℝ := 0.1) : Nat → Fin 64 → Fin 64 → WithBot ℝ :=
  msaCatScore 64 64 8
    (msaScoreA0 Q K Seqlens Blocks BlockOffsets (msaQVal Q s0 sm_scale) (msaKPtr K s0) s0)
    (msaScoreB0 Q K Seqlens Blocks BlockOffsets ColCounts Cols (msaQVal Q s0 sm_scale) (msaKPtr K s0) s0)
```
</details>

<details><summary><code>mixed_sparse_attention_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `mixed_sparse_attention.py`'s
`_triton_mixed_sparse_attn_fwd_kernel`. -/
```
```lean
def mixed_sparse_attention_fwd_kernel_surface
    (Q K V : RegionName) (seqlens : Region .nat) (sm_scale : ℝ)
    (block_count block_offset column_count column_index : Region .nat)
    (Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vn stride_vk
      stride_oz stride_oh stride_om stride_ok
      Z H N_CTX NUM_ROWS NNZ_S NNZ_V
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat)
    (dtype : FloatDType) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  seqlen = tl.load(seqlens + off_hz // $(H))
  if start_m * $(BLOCK_M) >= seqlen {
    return
  }

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))

  qo_offset = (off_hz // $(H)) * $(stride_qz) + (off_hz % $(H)) * $(stride_qh)
  kv_offset = (off_hz // $(H)) * $(stride_kz) + (off_hz % $(H)) * $(stride_kh)

  q_ptrs = Q + qo_offset + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  k_ptrs = K + kv_offset + offs_d[:, None] * $(stride_kk)
  v_ptrs = V + kv_offset + offs_d[None, :] * $(stride_vk)
  o_ptrs = Out + qo_offset + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_ok)

  num_blks = tl.load(block_count + off_hz * $(NUM_ROWS) + start_m)
  blks_ptr = block_offset + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_S)
  num_cols = tl.load(column_count + off_hz * $(NUM_ROWS) + start_m)
  cols_ptr = column_index + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_V)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(q_ptrs)
  q = (q * qk_scale).to(DTYPE)

  m_mask = offs_m[:, None] < seqlen

  max_num_blks = $(8)
  for block_index in range(max_num_blks) {
    cond = block_index < num_blks
    start_n = tl.load(blks_ptr + block_index, mask=cond)
    cols = start_n + offs_n
    n_mask = (cols < seqlen) & cond
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    causal_mask = cols[None, :] <= offs_m[:, None]
    qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  max_num_cols = $(16)
  for start_n in range($(0), max_num_cols, $(BLOCK_N)) {
    cond = start_n < num_cols
    n_mask = (start_n + offs_n < num_cols) & cond
    cols = tl.load(cols_ptr + start_n + offs_n, mask=cond[:, None], other=0)
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk = tl.where(m_mask & n_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  acc /= l_i[:, None]
  tl.store(o_ptrs, (acc).to(DTYPE), mask=m_mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (H : Nat) (Seqlens : RegionName) (BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H Seqlens
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_om stride_ok BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_om + dIndex idx * stride_ok
```
</details>

<details><summary><code>mixedSparseAttnClosedForm</code></summary>

```
/-- **Genuine (FAITHFUL) closed-form mixed-sparse attention output** for one
program/row. This mirrors *exactly* what
`_triton_mixed_sparse_attn_fwd_kernel` computes — including the faithfulness
quirk that **Loop A always runs `max_num_blks = 8` iterations regardless of
`num_blks`**.

For each iteration `b < 8` the kernel forms `cond = b < num_blks`, the masked
block start `start_n = blockStartN` (the masked default `0` when `cond` is
false), then for each lane `j < BLOCK_N` the key `n = start_n + j`:

* the K-load is masked by `n_mask = (n < seqlen) ∧ cond`, so the effective key
  vector is `K[n]` when `n < seqlen ∧ cond` and the **zero vector** otherwise;
* `qk = where(m_mask ∧ (n ≤ offs_m i), 0, -inf) + dot(q, K_masked)`, so the lane
  contributes weight `w = exp(effScale · rawMasked)` exactly when
  `offs_m i < seqlen ∧ n ≤ offs_m i`, and `0` otherwise. Here `rawMasked = raw n`
  when `n < seqlen ∧ cond` and `rawMasked = 0` (so `w = exp(0) = 1`) otherwise —
  this is the **spurious-block weight-1 path**: a block `b ≥ num_blks` has
  `start_n = 0`, `cond = false`, hence `n = j`, `n ≤ offs_m i` and
  (for active rows) `offs_m i < seqlen`, so it adds `exp(effScale·0) = 1` to the
  DENOMINATOR while its V is the zero vector, leaving the numerator unchanged.

Loop B (column phase) is correctly `n_mask`-guarded (its `where` masks
non-selected lanes to `⊥`), so spurious column lanes contribute nothing.

`numer/denom` is therefore the kernel's true output, **not** the naive
`num_blks`-only union softmax. The natural-exp scale is `effScale sm_scale =
sm_scale · 1.44269504 · log 2` (the faithful `exp2 → exp` bridge). -/
```
```lean
noncomputable def mixedSparseAttnClosedForm
    (s : BlockState) (Q K V : RegionName)
    (block_offset column_index : Region .nat)
    (H stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
      stride_vz stride_vh stride_vn
      NUM_ROWS NNZ_S NNZ_V
      num_blks num_cols seqlen
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (sm_scale : ℝ) (i : Fin BLOCK_M) (d : Nat) : ℝ :=
  let raw := fun n : Nat =>
    rawScore s Q K H stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
      BLOCK_DMODEL BLOCK_M i n
  -- block-sparse phase over ALL 8 = max_num_blks kernel iterations.
  -- `keep` = lane kept (causal + active row); `inSeq` = K/V actually loaded.
  let wBlock := fun (b : Fin 8) (j : Fin BLOCK_N) =>
    let SN := blockStartN s block_offset NUM_ROWS NNZ_S num_blks b.val
    let n := SN + j.val
    let inSeq := n < seqlen ∧ b.val < num_blks
    let rawMasked := if inSeq then raw n else 0
    if mIndex s BLOCK_M i < seqlen ∧ n ≤ mIndex s BLOCK_M i then
      Real.exp (effScale sm_scale * rawMasked) else 0
  let vBlock := fun (b : Fin 8) (j : Fin BLOCK_N) =>
    let SN := blockStartN s block_offset NUM_ROWS NNZ_S num_blks b.val
    let n := SN + j.val
    if n < seqlen ∧ b.val < num_blks then
      vRow s V H stride_vz stride_vh stride_vn n d else 0
  -- column-sparse phase weights. Faithful because the kernel `n_mask`-guards
  -- Loop B's `where`: a column lane `c < num_cols` is kept iff the row is active
  -- (`offs_m i < seqlen`). The kernel applies NO `cols < seqlen` mask to the
  -- column keys (only `c < num_cols ∧ 0 < num_cols`), so neither does this.
  let wCol := fun (c : Fin num_cols) =>
    let n := colKeyGlobal s column_index NUM_ROWS NNZ_V c.val
    if mIndex s BLOCK_M i < seqlen then
      Real.exp (effScale sm_scale * raw n) else 0
  let denom :=
    Finset.univ.sum (fun b : Fin 8 =>
      Finset.univ.sum (fun j : Fin BLOCK_N => wBlock b j)) +
    Finset.univ.sum (fun c : Fin num_cols => wCol c)
  let numer :=
    Finset.univ.sum (fun b : Fin 8 =>
      Finset.univ.sum (fun j : Fin BLOCK_N => wBlock b j * vBlock b j)) +
    Finset.univ.sum (fun c : Fin num_cols =>
      wCol c *
        vRow s V H stride_vz stride_vh stride_vn
          (colKeyGlobal s column_index NUM_ROWS NNZ_V c.val) d)
  numer / denom
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>offZ</code></summary>

```lean
def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>msaE</code></summary>

```
/-- `msaE x = exp2(x)` with `exp2(⊥) = 0` — the per-key softmax weight carrier. -/
```
```lean
noncomputable def msaE (x : WithBot ℝ) : ℝ := (WithBot.realExp2 x).unbotD 0
```
</details>

<details><summary><code>msaCatScore</code></summary>

```
/-- Concatenated score stream: first `bF` iterations from `scoreA`, then `scoreB`. -/
```
```lean
noncomputable def msaCatScore (BM BN bF : Nat)
    (scoreA scoreB : Nat → Fin BM → Fin BN → WithBot ℝ) :
    Nat → Fin BM → Fin BN → WithBot ℝ :=
  fun k => if k < bF then scoreA k else scoreB (k - bF)
```
</details>

<details><summary><code>msaScoreA0</code></summary>

```
/-- Concrete block-A score stream (pinned to the kernel lanes). -/
```
```lean
noncomputable def msaScoreA0 (Q K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (qF : TileIndex [64, 64] → WithBot ℝ) (kpF : TileIndex [64, 1] → RegionName × Nat)
    (s0 : BlockState) : Nat → Fin 64 → Fin 64 → WithBot ℝ :=
  fun c i j => msaScoreLaneA Q K Seqlens Blocks BlockOffsets qF kpF s0 c
    (msaSN0 s0 Blocks BlockOffsets c) i j
```
</details>

<details><summary><code>msaQVal</code></summary>

```
/-- The scaled `q` value the setup computes (as a `WithBot ℝ` carrier), lane
`(i,e)`: the raw `Q` read (via `readMemValue .real`) times `qk_scale =
0.1·1.44269504`, before the fp16 cast that `msaInvariantA` applies. Written with
`WithBot.realMul` so it matches the loop-body `mul` carrier exactly. -/
```
```lean
noncomputable def msaQVal (Q : RegionName) (s0 : BlockState)
    (sm_scale : ℝ := 0.1) : TileIndex [64, 64] → WithBot ℝ :=
  fun idx => WithBot.realMul
    (s0.readMemValue .real Q (((s0.pids 1 / 4) * 32768 + (s0.pids 1 % 4) * 8192)
      + (s0.pids 0 * 64 + idx.1.val) * 64 + idx.2.1.val))
    (some (sm_scale * 1.44269504))
```
</details>

<details><summary><code>msaKPtr</code></summary>

```
/-- The `k_ptrs` pointer tile `(e channel axis, j=1)`: `K` at `kv_offset + e·1`. -/
```
```lean
def msaKPtr (K : RegionName) (s0 : BlockState) : TileIndex [64, 1] → RegionName × Nat :=
  fun idx => (K, ((s0.pids 1 / 4) * 32768 + (s0.pids 1 % 4) * 8192) + idx.1.val)
```
</details>

<details><summary><code>msaScoreB0</code></summary>

```
/-- Concrete column-B score stream (loop value `sv = c·64`). -/
```
```lean
noncomputable def msaScoreB0 (Q K : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (qF : TileIndex [64, 64] → WithBot ℝ) (kpF : TileIndex [64, 1] → RegionName × Nat)
    (s0 : BlockState) : Nat → Fin 64 → Fin 64 → WithBot ℝ :=
  fun c i j => msaScoreLaneB Blocks ColCounts Seqlens qF kpF s0 (c * 64)
    (msaGcol0 s0 Cols ColCounts (c * 64)) i j
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>offH</code></summary>

```lean
def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>rawScore</code></summary>

```
/-- Unscaled raw score `Σ_{e<BLOCK_DMODEL} Q[row,e] · K[n,e]` at global key `n`. -/
```
```lean
noncomputable def rawScore (s : BlockState) (Q K : RegionName)
    (H stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
      BLOCK_DMODEL BLOCK_M : Nat) (i : Fin BLOCK_M) (n : Nat) : ℝ :=
  Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
    qRow s Q H stride_qz stride_qh stride_qm BLOCK_M i e.val *
      kRow s K H stride_kz stride_kh stride_kn n e.val)
```
</details>

<details><summary><code>blockStartN</code></summary>

```
/-- The masked block start `start_n` the kernel reads at block `b` (Loop A's
`tl.load(blks_ptr + b, mask = b < num_blks)`): the real offset for a visited
block, the masked default `0` for a spurious block `b ≥ num_blks`. -/
```
```lean
noncomputable def blockStartN (s : BlockState) (block_offset : Region .nat)
    (NUM_ROWS NNZ_S num_blks b : Nat) : Nat :=
  if b < num_blks then
    s.readMemValue .nat (Region.cast block_offset)
      ((s.pids 1 * NUM_ROWS + s.pids 0) * NNZ_S + b)
  else BlockState.defaultCarrier .nat
```
</details>

<details><summary><code>effScale</code></summary>

```
/-- **Faithful exp2→exp scale.** The kernel sets `qk_scale = sm_scale ·
1.44269504` and exponentiates with `exp2`. Since the semantics give
`exp2(x) = exp(x · log 2)`, the per-key weight the loop computes is
`exp2(qk_scale · raw) = exp(qk_scale · log 2 · raw)`. Hence the natural-exp
scale to instantiate the closed form with is
`effScale sm_scale = sm_scale · 1.44269504 · log 2`. (`1.44269504 · log 2 ≈ 1`,
the floating-point approximation of `log2(e) · ln 2 = 1`.) -/
```
```lean
noncomputable def effScale (sm_scale : ℝ) : ℝ :=
  sm_scale * 1.44269504 * Real.log 2
```
</details>

<details><summary><code>vRow</code></summary>

```
/-- V row at global key position `n`, channel `d`, at `kvBase + n·stride_vn + d`. -/
```
```lean
noncomputable def vRow (s : BlockState) (V : RegionName)
    (H stride_vz stride_vh stride_vn : Nat) (n d : Nat) : ℝ :=
  s.readMem V (qoBase s H stride_vz stride_vh + n * stride_vn + d)
```
</details>

<details><summary><code>colKeyGlobal</code></summary>

```
/-- Global key position of the `c`-th visited sparse column:
`column_index[off_hz·NUM_ROWS·NNZ_V + start_m·NNZ_V + c]`. -/
```
```lean
def colKeyGlobal (s : BlockState) (column_index : Region .nat)
    (NUM_ROWS NNZ_V c : Nat) : Nat :=
  s.readMemValue .nat (Region.cast column_index)
    ((s.pids 1 * NUM_ROWS + s.pids 0) * NNZ_V + c)
```
</details>

<details><summary><code>msaScoreLaneA</code></summary>

```
/-- The masked `qk` lane `(i,j)` the Loop-A body computes at block `c`:
`if (offs_m i < seqlen ∧ SN+j ≤ offs_m i) then some(Σ_e fp16(q)·K) else ⊥`. -/
```
```lean
noncomputable def msaScoreLaneA (Q K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (qF : TileIndex [64, 64] → WithBot ℝ) (kpF : TileIndex [64, 1] → RegionName × Nat)
    (s0 : BlockState) (c SN : Nat) (i j : Fin 64) : WithBot ℝ :=
  Option.map₂ (· + ·)
    (if (s0.pids 0 * 64 + i.val < seqLen s0 4 (Region.cast Seqlens)
        ∧ SN + j.val ≤ s0.pids 0 * 64 + i.val) then
      (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))
    (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
      (fun e : Fin 64 => Option.map₂ (· * ·)
        (FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 (qF (i, e, PUnit.unit))))
        (msaKLaneA K Seqlens Blocks BlockOffsets kpF s0 c SN e j)))
```
</details>

<details><summary><code>msaSN0</code></summary>

```
/-- The masked block start `start_n` Loop-A reads at block `c`. -/
```
```lean
noncomputable def msaSN0 (s0 : BlockState) (Blocks BlockOffsets : Region .nat) (c : Nat) : Nat :=
  if c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * 2 + s0.pids 0) then
    s0.readMemValue .nat (Region.cast BlockOffsets) ((s0.pids 1 * 2 + s0.pids 0) * 4 + c)
  else BlockState.defaultCarrier .nat
```
</details>

<details><summary><code>msaScoreLaneB</code></summary>

```
/-- The masked `qk` lane `(i,j)` the Loop-B body computes at loop value `sv`
(NON-causal): `if (offs_m i < seqlen ∧ (sv+j < num_cols ∧ cond)) then some(Σ q·K) else ⊥`. -/
```
```lean
noncomputable def msaScoreLaneB (Blocks ColCounts : Region .nat) (Seqlens : Region .nat)
    (qF : TileIndex [64, 64] → WithBot ℝ) (kpF : TileIndex [64, 1] → RegionName × Nat)
    (s0 : BlockState) (sv : Nat) (gcol : Fin 64 → Nat) (i j : Fin 64) : WithBot ℝ :=
  Option.map₂ (· + ·)
    (if (s0.pids 0 * 64 + i.val < seqLen s0 4 (Region.cast Seqlens)
        ∧ (sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0)
          ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0))) then
      (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))
    (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
      (fun e : Fin 64 => Option.map₂ (· * ·)
        (FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 (qF (i, e, PUnit.unit))))
        (msaKLaneB Blocks ColCounts kpF s0 sv gcol e j)))
```
</details>

<details><summary><code>msaGcol0</code></summary>

```
/-- The gathered columns Loop-B reads at loop value `sv`. -/
```
```lean
noncomputable def msaGcol0 (s0 : BlockState) (Cols ColCounts : Region .nat) (sv : Nat) :
    Fin 64 → Nat :=
  fun j => msaColLaneB Cols ColCounts s0 sv j
```
</details>

<details><summary><code>qRow</code></summary>

```
/-- Q row `start_m·BLOCK_M + i`, channel `e`, at `qoBase + row·stride_qm + e`. -/
```
```lean
noncomputable def qRow (s : BlockState) (Q : RegionName)
    (H stride_qz stride_qh stride_qm BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Nat) :
    ℝ :=
  s.readMem Q (qoBase s H stride_qz stride_qh + mIndex s BLOCK_M i * stride_qm + e)
```
</details>

<details><summary><code>kRow</code></summary>

```
/-- K row at global key position `n`, channel `e`, at `kvBase + n·stride_kn + e`.
The kernel reads K with `k_ptrs = K + kv_offset + offs_d·stride_kk` then
`+ cols·stride_kn`; here `kv_offset = off_z·stride_kz + off_h·stride_kh` and
`stride_kk = 1` (head channel `e` contiguous). -/
```
```lean
noncomputable def kRow (s : BlockState) (K : RegionName)
    (H stride_kz stride_kh stride_kn : Nat) (n e : Nat) : ℝ :=
  s.readMem K (qoBase s H stride_kz stride_kh + n * stride_kn + e)
```
</details>

<details><summary><code>qoBase</code></summary>

```
/-- Q/out tile base offset `off_z · stride_z + off_h · stride_h`. -/
```
```lean
def qoBase (s : BlockState) (H stride_z stride_h : Nat) : Nat :=
  offZ s H * stride_z + offH s H * stride_h
```
</details>

<details><summary><code>msaKLaneA</code></summary>

```
/-- The masked K-load lane `(e,j)` the Loop-A body computes at block `c`:
`if (SN+j < seqlen ∧ cb) then K[kpF(e)·base + (SN+j)·64] else 0`, where `cb`
is the loop guard `c < num_blks`. -/
```
```lean
noncomputable def msaKLaneA (K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (kpF : TileIndex [64, 1] → RegionName × Nat) (s0 : BlockState) (c SN : Nat)
    (e j : Fin 64) : WithBot ℝ :=
  if (SN + j.val < seqLen s0 4 (Region.cast Seqlens)
      ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * 2 + s0.pids 0)) then
    (s0.readMemValue .real (kpF (e, ⟨0, by simp⟩, PUnit.unit)).1
      ((kpF (e, ⟨0, by simp⟩, PUnit.unit)).2 + (SN + j.val) * 64) : WithBot ℝ)
  else (some (0.0 : ℝ) : WithBot ℝ)
```
</details>

<details><summary><code>msaKLaneB</code></summary>

```
/-- The masked K-load lane `(e,j)` at loop value `sv` (column-B): the gathered
column index `gcol j` scales the K pointer; mask is `(sv+j < num_cols) & cond`. -/
```
```lean
noncomputable def msaKLaneB (Blocks ColCounts : Region .nat)
    (kpF : TileIndex [64, 1] → RegionName × Nat) (s0 : BlockState) (sv : Nat)
    (gcol : Fin 64 → Nat) (e j : Fin 64) : WithBot ℝ :=
  if (sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0)
      ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0)) then
    (s0.readMemValue .real (kpF (e, ⟨0, by simp⟩, PUnit.unit)).1
      ((kpF (e, ⟨0, by simp⟩, PUnit.unit)).2 + gcol j * 64) : WithBot ℝ)
  else (some (0.0 : ℝ) : WithBot ℝ)
```
</details>

<details><summary><code>msaColLaneB</code></summary>

```
/-- The gathered column lane `j` at loop value `sv`: `cols_ptr[cpOff + sv + j]`
when the loop guard `sv < num_cols` holds, else `0`. -/
```
```lean
noncomputable def msaColLaneB (Cols : Region .nat) (ColCounts : Region .nat)
    (s0 : BlockState) (sv : Nat) (j : Fin 64) : Nat :=
  if sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0) then
    s0.readMemValue .nat (Region.cast Cols) ((s0.pids 1 * 2 + s0.pids 0) * 8 + sv + j.val)
  else 0
```
</details>

## Public theorem: `mixed_sparse_attention_python_case4_output_closed_form_summary`

<details><summary>docstring</summary>

```
/-- **Genuine Python case 4 closed-form summary** (block64, `sm_scale=0.1`,
alternate `seqlens`). Same genuine guarantee as case 1 at the alternate
`SeqlensAlt` input. -/
```
</details>

**Statement:**
```lean
theorem mixed_sparse_attention_python_case4_output_closed_form_summary
    (Q K V Out : RegionName)
    (SeqlensAlt Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hactive : s.pids 0 * 64 < seqLen s 4 (Region.cast SeqlensAlt))
    (hNC64 : s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * 2 + s.pids 0) ≤ 64)
    (hpos : ∀ i : Fin 64, s.pids 0 * 64 + i.val < seqLen s 4 (Region.cast SeqlensAlt) →
      0 < msaDenomUpto 64 64
        (msaCatScore0 Q K V SeqlensAlt Blocks BlockOffsets ColCounts Cols s) 9 i) :
    ∃ sF, exec (mixed_sparse_attention_fwd_kernel_surface Q K V SeqlensAlt
        (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
        32768 8192 64 1 32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
        2 4 128 2 4 8 64 64 64 FloatDType.fp16).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [64, 64],
          active s 4 SeqlensAlt 64 idx →
            sF.readMemValue .fp16 Out (outOffset s 4 32768 8192 64 1 64 idx)
              = (some (mixedSparseAttnClosedForm s Q K V BlockOffsets Cols 4
                  32768 8192 64 32768 8192 64 32768 8192 64 2 4 8
                  (s.readMemValue .nat (Region.cast Blocks) (s.pids 1 * 2 + s.pids 0))
                  (s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * 2 + s.pids 0))
                  (seqLen s 4 (Region.cast SeqlensAlt)) 64 64 64 0.1 idx.1 (dIndex idx)) : WithBot ℝ)
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg o, s.undef rg o = 0`
- `hactive : s.pids 0 * 64 < seqLen s 4 (Region.cast SeqlensAlt)`
- `hNC64 : s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * 2 + s.pids 0) ≤ 64`
- `hpos : ∀ i : Fin 64, s.pids 0 * 64 + i.val < seqLen s 4 (Region.cast SeqlensAlt) →
      0 < msaDenomUpto 64 64
        (msaCatScore0 Q K V SeqlensAlt Blocks BlockOffsets ColCounts Cols s) 9 i`

**Closed-form spec defs (transitive):** `seqLen`, `msaDenomUpto`, `msaCatScore0`, `mixed_sparse_attention_fwd_kernel_surface`, `active`, `outOffset`, `mixedSparseAttnClosedForm`, `dIndex`, `offZ`, `msaE`, `msaCatScore`, `msaScoreA0`, `msaQVal`, `msaKPtr`, `msaScoreB0`, `mIndex`, `offH`, `rawScore`, `blockStartN`, `effScale`, `vRow`, `colKeyGlobal`, `msaScoreLaneA`, `msaSN0`, `msaScoreLaneB`, `msaGcol0`, `qRow`, `kRow`, `qoBase`, `msaKLaneA`, `msaKLaneB`, `msaColLaneB`

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (H : Nat) (Seqlens : RegionName) : Nat :=
  s.readMemValue .nat Seqlens (offZ s H)
```
</details>

<details><summary><code>msaDenomUpto</code></summary>

```
/-- Direct (unshifted) running denominator: `Σ_{l<k} Σ_j exp2(score l i j)`. -/
```
```lean
noncomputable def msaDenomUpto (BM BN : Nat)
    (score : Nat → Fin BM → Fin BN → WithBot ℝ) (k : Nat) (i : Fin BM) : ℝ :=
  (Finset.range k).sum (fun l =>
    (Finset.univ : Finset (Fin BN)).sum (fun j => msaE (score l i j)))
```
</details>

<details><summary><code>msaCatScore0</code></summary>

```
/-- The two pinned cat streams used by the loop assembly. -/
```
```lean
noncomputable abbrev msaCatScore0
    (Q K V : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (s0 : BlockState) (sm_scale : ℝ := 0.1) : Nat → Fin 64 → Fin 64 → WithBot ℝ :=
  msaCatScore 64 64 8
    (msaScoreA0 Q K Seqlens Blocks BlockOffsets (msaQVal Q s0 sm_scale) (msaKPtr K s0) s0)
    (msaScoreB0 Q K Seqlens Blocks BlockOffsets ColCounts Cols (msaQVal Q s0 sm_scale) (msaKPtr K s0) s0)
```
</details>

<details><summary><code>mixed_sparse_attention_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `mixed_sparse_attention.py`'s
`_triton_mixed_sparse_attn_fwd_kernel`. -/
```
```lean
def mixed_sparse_attention_fwd_kernel_surface
    (Q K V : RegionName) (seqlens : Region .nat) (sm_scale : ℝ)
    (block_count block_offset column_count column_index : Region .nat)
    (Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vn stride_vk
      stride_oz stride_oh stride_om stride_ok
      Z H N_CTX NUM_ROWS NNZ_S NNZ_V
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat)
    (dtype : FloatDType) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  seqlen = tl.load(seqlens + off_hz // $(H))
  if start_m * $(BLOCK_M) >= seqlen {
    return
  }

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))

  qo_offset = (off_hz // $(H)) * $(stride_qz) + (off_hz % $(H)) * $(stride_qh)
  kv_offset = (off_hz // $(H)) * $(stride_kz) + (off_hz % $(H)) * $(stride_kh)

  q_ptrs = Q + qo_offset + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  k_ptrs = K + kv_offset + offs_d[:, None] * $(stride_kk)
  v_ptrs = V + kv_offset + offs_d[None, :] * $(stride_vk)
  o_ptrs = Out + qo_offset + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_ok)

  num_blks = tl.load(block_count + off_hz * $(NUM_ROWS) + start_m)
  blks_ptr = block_offset + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_S)
  num_cols = tl.load(column_count + off_hz * $(NUM_ROWS) + start_m)
  cols_ptr = column_index + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_V)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(q_ptrs)
  q = (q * qk_scale).to(DTYPE)

  m_mask = offs_m[:, None] < seqlen

  max_num_blks = $(8)
  for block_index in range(max_num_blks) {
    cond = block_index < num_blks
    start_n = tl.load(blks_ptr + block_index, mask=cond)
    cols = start_n + offs_n
    n_mask = (cols < seqlen) & cond
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    causal_mask = cols[None, :] <= offs_m[:, None]
    qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  max_num_cols = $(16)
  for start_n in range($(0), max_num_cols, $(BLOCK_N)) {
    cond = start_n < num_cols
    n_mask = (start_n + offs_n < num_cols) & cond
    cols = tl.load(cols_ptr + start_n + offs_n, mask=cond[:, None], other=0)
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk = tl.where(m_mask & n_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  acc /= l_i[:, None]
  tl.store(o_ptrs, (acc).to(DTYPE), mask=m_mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (H : Nat) (Seqlens : RegionName) (BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H Seqlens
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_om stride_ok BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_om + dIndex idx * stride_ok
```
</details>

<details><summary><code>mixedSparseAttnClosedForm</code></summary>

```
/-- **Genuine (FAITHFUL) closed-form mixed-sparse attention output** for one
program/row. This mirrors *exactly* what
`_triton_mixed_sparse_attn_fwd_kernel` computes — including the faithfulness
quirk that **Loop A always runs `max_num_blks = 8` iterations regardless of
`num_blks`**.

For each iteration `b < 8` the kernel forms `cond = b < num_blks`, the masked
block start `start_n = blockStartN` (the masked default `0` when `cond` is
false), then for each lane `j < BLOCK_N` the key `n = start_n + j`:

* the K-load is masked by `n_mask = (n < seqlen) ∧ cond`, so the effective key
  vector is `K[n]` when `n < seqlen ∧ cond` and the **zero vector** otherwise;
* `qk = where(m_mask ∧ (n ≤ offs_m i), 0, -inf) + dot(q, K_masked)`, so the lane
  contributes weight `w = exp(effScale · rawMasked)` exactly when
  `offs_m i < seqlen ∧ n ≤ offs_m i`, and `0` otherwise. Here `rawMasked = raw n`
  when `n < seqlen ∧ cond` and `rawMasked = 0` (so `w = exp(0) = 1`) otherwise —
  this is the **spurious-block weight-1 path**: a block `b ≥ num_blks` has
  `start_n = 0`, `cond = false`, hence `n = j`, `n ≤ offs_m i` and
  (for active rows) `offs_m i < seqlen`, so it adds `exp(effScale·0) = 1` to the
  DENOMINATOR while its V is the zero vector, leaving the numerator unchanged.

Loop B (column phase) is correctly `n_mask`-guarded (its `where` masks
non-selected lanes to `⊥`), so spurious column lanes contribute nothing.

`numer/denom` is therefore the kernel's true output, **not** the naive
`num_blks`-only union softmax. The natural-exp scale is `effScale sm_scale =
sm_scale · 1.44269504 · log 2` (the faithful `exp2 → exp` bridge). -/
```
```lean
noncomputable def mixedSparseAttnClosedForm
    (s : BlockState) (Q K V : RegionName)
    (block_offset column_index : Region .nat)
    (H stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
      stride_vz stride_vh stride_vn
      NUM_ROWS NNZ_S NNZ_V
      num_blks num_cols seqlen
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (sm_scale : ℝ) (i : Fin BLOCK_M) (d : Nat) : ℝ :=
  let raw := fun n : Nat =>
    rawScore s Q K H stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
      BLOCK_DMODEL BLOCK_M i n
  -- block-sparse phase over ALL 8 = max_num_blks kernel iterations.
  -- `keep` = lane kept (causal + active row); `inSeq` = K/V actually loaded.
  let wBlock := fun (b : Fin 8) (j : Fin BLOCK_N) =>
    let SN := blockStartN s block_offset NUM_ROWS NNZ_S num_blks b.val
    let n := SN + j.val
    let inSeq := n < seqlen ∧ b.val < num_blks
    let rawMasked := if inSeq then raw n else 0
    if mIndex s BLOCK_M i < seqlen ∧ n ≤ mIndex s BLOCK_M i then
      Real.exp (effScale sm_scale * rawMasked) else 0
  let vBlock := fun (b : Fin 8) (j : Fin BLOCK_N) =>
    let SN := blockStartN s block_offset NUM_ROWS NNZ_S num_blks b.val
    let n := SN + j.val
    if n < seqlen ∧ b.val < num_blks then
      vRow s V H stride_vz stride_vh stride_vn n d else 0
  -- column-sparse phase weights. Faithful because the kernel `n_mask`-guards
  -- Loop B's `where`: a column lane `c < num_cols` is kept iff the row is active
  -- (`offs_m i < seqlen`). The kernel applies NO `cols < seqlen` mask to the
  -- column keys (only `c < num_cols ∧ 0 < num_cols`), so neither does this.
  let wCol := fun (c : Fin num_cols) =>
    let n := colKeyGlobal s column_index NUM_ROWS NNZ_V c.val
    if mIndex s BLOCK_M i < seqlen then
      Real.exp (effScale sm_scale * raw n) else 0
  let denom :=
    Finset.univ.sum (fun b : Fin 8 =>
      Finset.univ.sum (fun j : Fin BLOCK_N => wBlock b j)) +
    Finset.univ.sum (fun c : Fin num_cols => wCol c)
  let numer :=
    Finset.univ.sum (fun b : Fin 8 =>
      Finset.univ.sum (fun j : Fin BLOCK_N => wBlock b j * vBlock b j)) +
    Finset.univ.sum (fun c : Fin num_cols =>
      wCol c *
        vRow s V H stride_vz stride_vh stride_vn
          (colKeyGlobal s column_index NUM_ROWS NNZ_V c.val) d)
  numer / denom
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>offZ</code></summary>

```lean
def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>msaE</code></summary>

```
/-- `msaE x = exp2(x)` with `exp2(⊥) = 0` — the per-key softmax weight carrier. -/
```
```lean
noncomputable def msaE (x : WithBot ℝ) : ℝ := (WithBot.realExp2 x).unbotD 0
```
</details>

<details><summary><code>msaCatScore</code></summary>

```
/-- Concatenated score stream: first `bF` iterations from `scoreA`, then `scoreB`. -/
```
```lean
noncomputable def msaCatScore (BM BN bF : Nat)
    (scoreA scoreB : Nat → Fin BM → Fin BN → WithBot ℝ) :
    Nat → Fin BM → Fin BN → WithBot ℝ :=
  fun k => if k < bF then scoreA k else scoreB (k - bF)
```
</details>

<details><summary><code>msaScoreA0</code></summary>

```
/-- Concrete block-A score stream (pinned to the kernel lanes). -/
```
```lean
noncomputable def msaScoreA0 (Q K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (qF : TileIndex [64, 64] → WithBot ℝ) (kpF : TileIndex [64, 1] → RegionName × Nat)
    (s0 : BlockState) : Nat → Fin 64 → Fin 64 → WithBot ℝ :=
  fun c i j => msaScoreLaneA Q K Seqlens Blocks BlockOffsets qF kpF s0 c
    (msaSN0 s0 Blocks BlockOffsets c) i j
```
</details>

<details><summary><code>msaQVal</code></summary>

```
/-- The scaled `q` value the setup computes (as a `WithBot ℝ` carrier), lane
`(i,e)`: the raw `Q` read (via `readMemValue .real`) times `qk_scale =
0.1·1.44269504`, before the fp16 cast that `msaInvariantA` applies. Written with
`WithBot.realMul` so it matches the loop-body `mul` carrier exactly. -/
```
```lean
noncomputable def msaQVal (Q : RegionName) (s0 : BlockState)
    (sm_scale : ℝ := 0.1) : TileIndex [64, 64] → WithBot ℝ :=
  fun idx => WithBot.realMul
    (s0.readMemValue .real Q (((s0.pids 1 / 4) * 32768 + (s0.pids 1 % 4) * 8192)
      + (s0.pids 0 * 64 + idx.1.val) * 64 + idx.2.1.val))
    (some (sm_scale * 1.44269504))
```
</details>

<details><summary><code>msaKPtr</code></summary>

```
/-- The `k_ptrs` pointer tile `(e channel axis, j=1)`: `K` at `kv_offset + e·1`. -/
```
```lean
def msaKPtr (K : RegionName) (s0 : BlockState) : TileIndex [64, 1] → RegionName × Nat :=
  fun idx => (K, ((s0.pids 1 / 4) * 32768 + (s0.pids 1 % 4) * 8192) + idx.1.val)
```
</details>

<details><summary><code>msaScoreB0</code></summary>

```
/-- Concrete column-B score stream (loop value `sv = c·64`). -/
```
```lean
noncomputable def msaScoreB0 (Q K : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (qF : TileIndex [64, 64] → WithBot ℝ) (kpF : TileIndex [64, 1] → RegionName × Nat)
    (s0 : BlockState) : Nat → Fin 64 → Fin 64 → WithBot ℝ :=
  fun c i j => msaScoreLaneB Blocks ColCounts Seqlens qF kpF s0 (c * 64)
    (msaGcol0 s0 Cols ColCounts (c * 64)) i j
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>offH</code></summary>

```lean
def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>rawScore</code></summary>

```
/-- Unscaled raw score `Σ_{e<BLOCK_DMODEL} Q[row,e] · K[n,e]` at global key `n`. -/
```
```lean
noncomputable def rawScore (s : BlockState) (Q K : RegionName)
    (H stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
      BLOCK_DMODEL BLOCK_M : Nat) (i : Fin BLOCK_M) (n : Nat) : ℝ :=
  Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
    qRow s Q H stride_qz stride_qh stride_qm BLOCK_M i e.val *
      kRow s K H stride_kz stride_kh stride_kn n e.val)
```
</details>

<details><summary><code>blockStartN</code></summary>

```
/-- The masked block start `start_n` the kernel reads at block `b` (Loop A's
`tl.load(blks_ptr + b, mask = b < num_blks)`): the real offset for a visited
block, the masked default `0` for a spurious block `b ≥ num_blks`. -/
```
```lean
noncomputable def blockStartN (s : BlockState) (block_offset : Region .nat)
    (NUM_ROWS NNZ_S num_blks b : Nat) : Nat :=
  if b < num_blks then
    s.readMemValue .nat (Region.cast block_offset)
      ((s.pids 1 * NUM_ROWS + s.pids 0) * NNZ_S + b)
  else BlockState.defaultCarrier .nat
```
</details>

<details><summary><code>effScale</code></summary>

```
/-- **Faithful exp2→exp scale.** The kernel sets `qk_scale = sm_scale ·
1.44269504` and exponentiates with `exp2`. Since the semantics give
`exp2(x) = exp(x · log 2)`, the per-key weight the loop computes is
`exp2(qk_scale · raw) = exp(qk_scale · log 2 · raw)`. Hence the natural-exp
scale to instantiate the closed form with is
`effScale sm_scale = sm_scale · 1.44269504 · log 2`. (`1.44269504 · log 2 ≈ 1`,
the floating-point approximation of `log2(e) · ln 2 = 1`.) -/
```
```lean
noncomputable def effScale (sm_scale : ℝ) : ℝ :=
  sm_scale * 1.44269504 * Real.log 2
```
</details>

<details><summary><code>vRow</code></summary>

```
/-- V row at global key position `n`, channel `d`, at `kvBase + n·stride_vn + d`. -/
```
```lean
noncomputable def vRow (s : BlockState) (V : RegionName)
    (H stride_vz stride_vh stride_vn : Nat) (n d : Nat) : ℝ :=
  s.readMem V (qoBase s H stride_vz stride_vh + n * stride_vn + d)
```
</details>

<details><summary><code>colKeyGlobal</code></summary>

```
/-- Global key position of the `c`-th visited sparse column:
`column_index[off_hz·NUM_ROWS·NNZ_V + start_m·NNZ_V + c]`. -/
```
```lean
def colKeyGlobal (s : BlockState) (column_index : Region .nat)
    (NUM_ROWS NNZ_V c : Nat) : Nat :=
  s.readMemValue .nat (Region.cast column_index)
    ((s.pids 1 * NUM_ROWS + s.pids 0) * NNZ_V + c)
```
</details>

<details><summary><code>msaScoreLaneA</code></summary>

```
/-- The masked `qk` lane `(i,j)` the Loop-A body computes at block `c`:
`if (offs_m i < seqlen ∧ SN+j ≤ offs_m i) then some(Σ_e fp16(q)·K) else ⊥`. -/
```
```lean
noncomputable def msaScoreLaneA (Q K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (qF : TileIndex [64, 64] → WithBot ℝ) (kpF : TileIndex [64, 1] → RegionName × Nat)
    (s0 : BlockState) (c SN : Nat) (i j : Fin 64) : WithBot ℝ :=
  Option.map₂ (· + ·)
    (if (s0.pids 0 * 64 + i.val < seqLen s0 4 (Region.cast Seqlens)
        ∧ SN + j.val ≤ s0.pids 0 * 64 + i.val) then
      (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))
    (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
      (fun e : Fin 64 => Option.map₂ (· * ·)
        (FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 (qF (i, e, PUnit.unit))))
        (msaKLaneA K Seqlens Blocks BlockOffsets kpF s0 c SN e j)))
```
</details>

<details><summary><code>msaSN0</code></summary>

```
/-- The masked block start `start_n` Loop-A reads at block `c`. -/
```
```lean
noncomputable def msaSN0 (s0 : BlockState) (Blocks BlockOffsets : Region .nat) (c : Nat) : Nat :=
  if c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * 2 + s0.pids 0) then
    s0.readMemValue .nat (Region.cast BlockOffsets) ((s0.pids 1 * 2 + s0.pids 0) * 4 + c)
  else BlockState.defaultCarrier .nat
```
</details>

<details><summary><code>msaScoreLaneB</code></summary>

```
/-- The masked `qk` lane `(i,j)` the Loop-B body computes at loop value `sv`
(NON-causal): `if (offs_m i < seqlen ∧ (sv+j < num_cols ∧ cond)) then some(Σ q·K) else ⊥`. -/
```
```lean
noncomputable def msaScoreLaneB (Blocks ColCounts : Region .nat) (Seqlens : Region .nat)
    (qF : TileIndex [64, 64] → WithBot ℝ) (kpF : TileIndex [64, 1] → RegionName × Nat)
    (s0 : BlockState) (sv : Nat) (gcol : Fin 64 → Nat) (i j : Fin 64) : WithBot ℝ :=
  Option.map₂ (· + ·)
    (if (s0.pids 0 * 64 + i.val < seqLen s0 4 (Region.cast Seqlens)
        ∧ (sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0)
          ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0))) then
      (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))
    (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
      (fun e : Fin 64 => Option.map₂ (· * ·)
        (FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 (qF (i, e, PUnit.unit))))
        (msaKLaneB Blocks ColCounts kpF s0 sv gcol e j)))
```
</details>

<details><summary><code>msaGcol0</code></summary>

```
/-- The gathered columns Loop-B reads at loop value `sv`. -/
```
```lean
noncomputable def msaGcol0 (s0 : BlockState) (Cols ColCounts : Region .nat) (sv : Nat) :
    Fin 64 → Nat :=
  fun j => msaColLaneB Cols ColCounts s0 sv j
```
</details>

<details><summary><code>qRow</code></summary>

```
/-- Q row `start_m·BLOCK_M + i`, channel `e`, at `qoBase + row·stride_qm + e`. -/
```
```lean
noncomputable def qRow (s : BlockState) (Q : RegionName)
    (H stride_qz stride_qh stride_qm BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Nat) :
    ℝ :=
  s.readMem Q (qoBase s H stride_qz stride_qh + mIndex s BLOCK_M i * stride_qm + e)
```
</details>

<details><summary><code>kRow</code></summary>

```
/-- K row at global key position `n`, channel `e`, at `kvBase + n·stride_kn + e`.
The kernel reads K with `k_ptrs = K + kv_offset + offs_d·stride_kk` then
`+ cols·stride_kn`; here `kv_offset = off_z·stride_kz + off_h·stride_kh` and
`stride_kk = 1` (head channel `e` contiguous). -/
```
```lean
noncomputable def kRow (s : BlockState) (K : RegionName)
    (H stride_kz stride_kh stride_kn : Nat) (n e : Nat) : ℝ :=
  s.readMem K (qoBase s H stride_kz stride_kh + n * stride_kn + e)
```
</details>

<details><summary><code>qoBase</code></summary>

```
/-- Q/out tile base offset `off_z · stride_z + off_h · stride_h`. -/
```
```lean
def qoBase (s : BlockState) (H stride_z stride_h : Nat) : Nat :=
  offZ s H * stride_z + offH s H * stride_h
```
</details>

<details><summary><code>msaKLaneA</code></summary>

```
/-- The masked K-load lane `(e,j)` the Loop-A body computes at block `c`:
`if (SN+j < seqlen ∧ cb) then K[kpF(e)·base + (SN+j)·64] else 0`, where `cb`
is the loop guard `c < num_blks`. -/
```
```lean
noncomputable def msaKLaneA (K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (kpF : TileIndex [64, 1] → RegionName × Nat) (s0 : BlockState) (c SN : Nat)
    (e j : Fin 64) : WithBot ℝ :=
  if (SN + j.val < seqLen s0 4 (Region.cast Seqlens)
      ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * 2 + s0.pids 0)) then
    (s0.readMemValue .real (kpF (e, ⟨0, by simp⟩, PUnit.unit)).1
      ((kpF (e, ⟨0, by simp⟩, PUnit.unit)).2 + (SN + j.val) * 64) : WithBot ℝ)
  else (some (0.0 : ℝ) : WithBot ℝ)
```
</details>

<details><summary><code>msaKLaneB</code></summary>

```
/-- The masked K-load lane `(e,j)` at loop value `sv` (column-B): the gathered
column index `gcol j` scales the K pointer; mask is `(sv+j < num_cols) & cond`. -/
```
```lean
noncomputable def msaKLaneB (Blocks ColCounts : Region .nat)
    (kpF : TileIndex [64, 1] → RegionName × Nat) (s0 : BlockState) (sv : Nat)
    (gcol : Fin 64 → Nat) (e j : Fin 64) : WithBot ℝ :=
  if (sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0)
      ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0)) then
    (s0.readMemValue .real (kpF (e, ⟨0, by simp⟩, PUnit.unit)).1
      ((kpF (e, ⟨0, by simp⟩, PUnit.unit)).2 + gcol j * 64) : WithBot ℝ)
  else (some (0.0 : ℝ) : WithBot ℝ)
```
</details>

<details><summary><code>msaColLaneB</code></summary>

```
/-- The gathered column lane `j` at loop value `sv`: `cols_ptr[cpOff + sv + j]`
when the loop guard `sv < num_cols` holds, else `0`. -/
```
```lean
noncomputable def msaColLaneB (Cols : Region .nat) (ColCounts : Region .nat)
    (s0 : BlockState) (sv : Nat) (j : Fin 64) : Nat :=
  if sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0) then
    s0.readMemValue .nat (Region.cast Cols) ((s0.pids 1 * 2 + s0.pids 0) * 8 + sv + j.val)
  else 0
```
</details>

## Public theorem: `mixed_sparse_attention_python_case3_output_closed_form_summary`

<details><summary>docstring</summary>

```
/-- **Genuine Python case 3 closed-form summary** (`BLOCK_M=BLOCK_N=64`,
`sm_scale=0.2`). Same genuine guarantee as case 1, at the larger softmax scale:
the executed surface kernel's `fp16` `Out` cell at every active output lane equals
`some (mixedSparseAttnClosedForm …)` with `sm_scale = 0.2`. The whole streaming
foundation is parameterized by `sm_scale`, so this is `msa_exec` instantiated at
`0.2` — NOT the store-only `accStoreValue` fact, the genuine non-self-referential
mixed-sparse closed form. -/
```
</details>

**Statement:**
```lean
theorem mixed_sparse_attention_python_case3_output_closed_form_summary
    (Q K V Out : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hactive : s.pids 0 * 64 < seqLen s 4 (Region.cast Seqlens))
    (hNC64 : s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * 2 + s.pids 0) ≤ 64)
    (hpos : ∀ i : Fin 64, s.pids 0 * 64 + i.val < seqLen s 4 (Region.cast Seqlens) →
      0 < msaDenomUpto 64 64
        (msaCatScore0 Q K V Seqlens Blocks BlockOffsets ColCounts Cols s (0.2 : ℝ)) 9 i) :
    ∃ sF, exec (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
        (0.2 : ℝ) Blocks BlockOffsets ColCounts Cols Out
        32768 8192 64 1 32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
        2 4 128 2 4 8 64 64 64 FloatDType.fp16).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [64, 64],
          active s 4 Seqlens 64 idx →
            sF.readMemValue .fp16 Out (outOffset s 4 32768 8192 64 1 64 idx)
              = (some (mixedSparseAttnClosedForm s Q K V BlockOffsets Cols 4
                  32768 8192 64 32768 8192 64 32768 8192 64 2 4 8
                  (s.readMemValue .nat (Region.cast Blocks) (s.pids 1 * 2 + s.pids 0))
                  (s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * 2 + s.pids 0))
                  (seqLen s 4 (Region.cast Seqlens)) 64 64 64 0.2 idx.1 (dIndex idx)) : WithBot ℝ)
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg o, s.undef rg o = 0`
- `hactive : s.pids 0 * 64 < seqLen s 4 (Region.cast Seqlens)`
- `hNC64 : s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * 2 + s.pids 0) ≤ 64`

**Closed-form spec defs (transitive):** `seqLen`, `msaDenomUpto`, `msaCatScore0`, `mixed_sparse_attention_fwd_kernel_surface`, `active`, `outOffset`, `mixedSparseAttnClosedForm`, `dIndex`, `offZ`, `msaE`, `msaCatScore`, `msaScoreA0`, `msaQVal`, `msaKPtr`, `msaScoreB0`, `mIndex`, `offH`, `rawScore`, `blockStartN`, `effScale`, `vRow`, `colKeyGlobal`, `msaScoreLaneA`, `msaSN0`, `msaScoreLaneB`, `msaGcol0`, `qRow`, `kRow`, `qoBase`, `msaKLaneA`, `msaKLaneB`, `msaColLaneB`

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (H : Nat) (Seqlens : RegionName) : Nat :=
  s.readMemValue .nat Seqlens (offZ s H)
```
</details>

<details><summary><code>msaDenomUpto</code></summary>

```
/-- Direct (unshifted) running denominator: `Σ_{l<k} Σ_j exp2(score l i j)`. -/
```
```lean
noncomputable def msaDenomUpto (BM BN : Nat)
    (score : Nat → Fin BM → Fin BN → WithBot ℝ) (k : Nat) (i : Fin BM) : ℝ :=
  (Finset.range k).sum (fun l =>
    (Finset.univ : Finset (Fin BN)).sum (fun j => msaE (score l i j)))
```
</details>

<details><summary><code>msaCatScore0</code></summary>

```
/-- The two pinned cat streams used by the loop assembly. -/
```
```lean
noncomputable abbrev msaCatScore0
    (Q K V : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (s0 : BlockState) (sm_scale : ℝ := 0.1) : Nat → Fin 64 → Fin 64 → WithBot ℝ :=
  msaCatScore 64 64 8
    (msaScoreA0 Q K Seqlens Blocks BlockOffsets (msaQVal Q s0 sm_scale) (msaKPtr K s0) s0)
    (msaScoreB0 Q K Seqlens Blocks BlockOffsets ColCounts Cols (msaQVal Q s0 sm_scale) (msaKPtr K s0) s0)
```
</details>

<details><summary><code>mixed_sparse_attention_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `mixed_sparse_attention.py`'s
`_triton_mixed_sparse_attn_fwd_kernel`. -/
```
```lean
def mixed_sparse_attention_fwd_kernel_surface
    (Q K V : RegionName) (seqlens : Region .nat) (sm_scale : ℝ)
    (block_count block_offset column_count column_index : Region .nat)
    (Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vn stride_vk
      stride_oz stride_oh stride_om stride_ok
      Z H N_CTX NUM_ROWS NNZ_S NNZ_V
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat)
    (dtype : FloatDType) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  seqlen = tl.load(seqlens + off_hz // $(H))
  if start_m * $(BLOCK_M) >= seqlen {
    return
  }

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))

  qo_offset = (off_hz // $(H)) * $(stride_qz) + (off_hz % $(H)) * $(stride_qh)
  kv_offset = (off_hz // $(H)) * $(stride_kz) + (off_hz % $(H)) * $(stride_kh)

  q_ptrs = Q + qo_offset + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  k_ptrs = K + kv_offset + offs_d[:, None] * $(stride_kk)
  v_ptrs = V + kv_offset + offs_d[None, :] * $(stride_vk)
  o_ptrs = Out + qo_offset + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_ok)

  num_blks = tl.load(block_count + off_hz * $(NUM_ROWS) + start_m)
  blks_ptr = block_offset + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_S)
  num_cols = tl.load(column_count + off_hz * $(NUM_ROWS) + start_m)
  cols_ptr = column_index + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_V)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(q_ptrs)
  q = (q * qk_scale).to(DTYPE)

  m_mask = offs_m[:, None] < seqlen

  max_num_blks = $(8)
  for block_index in range(max_num_blks) {
    cond = block_index < num_blks
    start_n = tl.load(blks_ptr + block_index, mask=cond)
    cols = start_n + offs_n
    n_mask = (cols < seqlen) & cond
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    causal_mask = cols[None, :] <= offs_m[:, None]
    qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  max_num_cols = $(16)
  for start_n in range($(0), max_num_cols, $(BLOCK_N)) {
    cond = start_n < num_cols
    n_mask = (start_n + offs_n < num_cols) & cond
    cols = tl.load(cols_ptr + start_n + offs_n, mask=cond[:, None], other=0)
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk = tl.where(m_mask & n_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  acc /= l_i[:, None]
  tl.store(o_ptrs, (acc).to(DTYPE), mask=m_mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (H : Nat) (Seqlens : RegionName) (BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H Seqlens
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_om stride_ok BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_om + dIndex idx * stride_ok
```
</details>

<details><summary><code>mixedSparseAttnClosedForm</code></summary>

```
/-- **Genuine (FAITHFUL) closed-form mixed-sparse attention output** for one
program/row. This mirrors *exactly* what
`_triton_mixed_sparse_attn_fwd_kernel` computes — including the faithfulness
quirk that **Loop A always runs `max_num_blks = 8` iterations regardless of
`num_blks`**.

For each iteration `b < 8` the kernel forms `cond = b < num_blks`, the masked
block start `start_n = blockStartN` (the masked default `0` when `cond` is
false), then for each lane `j < BLOCK_N` the key `n = start_n + j`:

* the K-load is masked by `n_mask = (n < seqlen) ∧ cond`, so the effective key
  vector is `K[n]` when `n < seqlen ∧ cond` and the **zero vector** otherwise;
* `qk = where(m_mask ∧ (n ≤ offs_m i), 0, -inf) + dot(q, K_masked)`, so the lane
  contributes weight `w = exp(effScale · rawMasked)` exactly when
  `offs_m i < seqlen ∧ n ≤ offs_m i`, and `0` otherwise. Here `rawMasked = raw n`
  when `n < seqlen ∧ cond` and `rawMasked = 0` (so `w = exp(0) = 1`) otherwise —
  this is the **spurious-block weight-1 path**: a block `b ≥ num_blks` has
  `start_n = 0`, `cond = false`, hence `n = j`, `n ≤ offs_m i` and
  (for active rows) `offs_m i < seqlen`, so it adds `exp(effScale·0) = 1` to the
  DENOMINATOR while its V is the zero vector, leaving the numerator unchanged.

Loop B (column phase) is correctly `n_mask`-guarded (its `where` masks
non-selected lanes to `⊥`), so spurious column lanes contribute nothing.

`numer/denom` is therefore the kernel's true output, **not** the naive
`num_blks`-only union softmax. The natural-exp scale is `effScale sm_scale =
sm_scale · 1.44269504 · log 2` (the faithful `exp2 → exp` bridge). -/
```
```lean
noncomputable def mixedSparseAttnClosedForm
    (s : BlockState) (Q K V : RegionName)
    (block_offset column_index : Region .nat)
    (H stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
      stride_vz stride_vh stride_vn
      NUM_ROWS NNZ_S NNZ_V
      num_blks num_cols seqlen
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (sm_scale : ℝ) (i : Fin BLOCK_M) (d : Nat) : ℝ :=
  let raw := fun n : Nat =>
    rawScore s Q K H stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
      BLOCK_DMODEL BLOCK_M i n
  -- block-sparse phase over ALL 8 = max_num_blks kernel iterations.
  -- `keep` = lane kept (causal + active row); `inSeq` = K/V actually loaded.
  let wBlock := fun (b : Fin 8) (j : Fin BLOCK_N) =>
    let SN := blockStartN s block_offset NUM_ROWS NNZ_S num_blks b.val
    let n := SN + j.val
    let inSeq := n < seqlen ∧ b.val < num_blks
    let rawMasked := if inSeq then raw n else 0
    if mIndex s BLOCK_M i < seqlen ∧ n ≤ mIndex s BLOCK_M i then
      Real.exp (effScale sm_scale * rawMasked) else 0
  let vBlock := fun (b : Fin 8) (j : Fin BLOCK_N) =>
    let SN := blockStartN s block_offset NUM_ROWS NNZ_S num_blks b.val
    let n := SN + j.val
    if n < seqlen ∧ b.val < num_blks then
      vRow s V H stride_vz stride_vh stride_vn n d else 0
  -- column-sparse phase weights. Faithful because the kernel `n_mask`-guards
  -- Loop B's `where`: a column lane `c < num_cols` is kept iff the row is active
  -- (`offs_m i < seqlen`). The kernel applies NO `cols < seqlen` mask to the
  -- column keys (only `c < num_cols ∧ 0 < num_cols`), so neither does this.
  let wCol := fun (c : Fin num_cols) =>
    let n := colKeyGlobal s column_index NUM_ROWS NNZ_V c.val
    if mIndex s BLOCK_M i < seqlen then
      Real.exp (effScale sm_scale * raw n) else 0
  let denom :=
    Finset.univ.sum (fun b : Fin 8 =>
      Finset.univ.sum (fun j : Fin BLOCK_N => wBlock b j)) +
    Finset.univ.sum (fun c : Fin num_cols => wCol c)
  let numer :=
    Finset.univ.sum (fun b : Fin 8 =>
      Finset.univ.sum (fun j : Fin BLOCK_N => wBlock b j * vBlock b j)) +
    Finset.univ.sum (fun c : Fin num_cols =>
      wCol c *
        vRow s V H stride_vz stride_vh stride_vn
          (colKeyGlobal s column_index NUM_ROWS NNZ_V c.val) d)
  numer / denom
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>offZ</code></summary>

```lean
def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>msaE</code></summary>

```
/-- `msaE x = exp2(x)` with `exp2(⊥) = 0` — the per-key softmax weight carrier. -/
```
```lean
noncomputable def msaE (x : WithBot ℝ) : ℝ := (WithBot.realExp2 x).unbotD 0
```
</details>

<details><summary><code>msaCatScore</code></summary>

```
/-- Concatenated score stream: first `bF` iterations from `scoreA`, then `scoreB`. -/
```
```lean
noncomputable def msaCatScore (BM BN bF : Nat)
    (scoreA scoreB : Nat → Fin BM → Fin BN → WithBot ℝ) :
    Nat → Fin BM → Fin BN → WithBot ℝ :=
  fun k => if k < bF then scoreA k else scoreB (k - bF)
```
</details>

<details><summary><code>msaScoreA0</code></summary>

```
/-- Concrete block-A score stream (pinned to the kernel lanes). -/
```
```lean
noncomputable def msaScoreA0 (Q K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (qF : TileIndex [64, 64] → WithBot ℝ) (kpF : TileIndex [64, 1] → RegionName × Nat)
    (s0 : BlockState) : Nat → Fin 64 → Fin 64 → WithBot ℝ :=
  fun c i j => msaScoreLaneA Q K Seqlens Blocks BlockOffsets qF kpF s0 c
    (msaSN0 s0 Blocks BlockOffsets c) i j
```
</details>

<details><summary><code>msaQVal</code></summary>

```
/-- The scaled `q` value the setup computes (as a `WithBot ℝ` carrier), lane
`(i,e)`: the raw `Q` read (via `readMemValue .real`) times `qk_scale =
0.1·1.44269504`, before the fp16 cast that `msaInvariantA` applies. Written with
`WithBot.realMul` so it matches the loop-body `mul` carrier exactly. -/
```
```lean
noncomputable def msaQVal (Q : RegionName) (s0 : BlockState)
    (sm_scale : ℝ := 0.1) : TileIndex [64, 64] → WithBot ℝ :=
  fun idx => WithBot.realMul
    (s0.readMemValue .real Q (((s0.pids 1 / 4) * 32768 + (s0.pids 1 % 4) * 8192)
      + (s0.pids 0 * 64 + idx.1.val) * 64 + idx.2.1.val))
    (some (sm_scale * 1.44269504))
```
</details>

<details><summary><code>msaKPtr</code></summary>

```
/-- The `k_ptrs` pointer tile `(e channel axis, j=1)`: `K` at `kv_offset + e·1`. -/
```
```lean
def msaKPtr (K : RegionName) (s0 : BlockState) : TileIndex [64, 1] → RegionName × Nat :=
  fun idx => (K, ((s0.pids 1 / 4) * 32768 + (s0.pids 1 % 4) * 8192) + idx.1.val)
```
</details>

<details><summary><code>msaScoreB0</code></summary>

```
/-- Concrete column-B score stream (loop value `sv = c·64`). -/
```
```lean
noncomputable def msaScoreB0 (Q K : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (qF : TileIndex [64, 64] → WithBot ℝ) (kpF : TileIndex [64, 1] → RegionName × Nat)
    (s0 : BlockState) : Nat → Fin 64 → Fin 64 → WithBot ℝ :=
  fun c i j => msaScoreLaneB Blocks ColCounts Seqlens qF kpF s0 (c * 64)
    (msaGcol0 s0 Cols ColCounts (c * 64)) i j
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>offH</code></summary>

```lean
def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>rawScore</code></summary>

```
/-- Unscaled raw score `Σ_{e<BLOCK_DMODEL} Q[row,e] · K[n,e]` at global key `n`. -/
```
```lean
noncomputable def rawScore (s : BlockState) (Q K : RegionName)
    (H stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
      BLOCK_DMODEL BLOCK_M : Nat) (i : Fin BLOCK_M) (n : Nat) : ℝ :=
  Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
    qRow s Q H stride_qz stride_qh stride_qm BLOCK_M i e.val *
      kRow s K H stride_kz stride_kh stride_kn n e.val)
```
</details>

<details><summary><code>blockStartN</code></summary>

```
/-- The masked block start `start_n` the kernel reads at block `b` (Loop A's
`tl.load(blks_ptr + b, mask = b < num_blks)`): the real offset for a visited
block, the masked default `0` for a spurious block `b ≥ num_blks`. -/
```
```lean
noncomputable def blockStartN (s : BlockState) (block_offset : Region .nat)
    (NUM_ROWS NNZ_S num_blks b : Nat) : Nat :=
  if b < num_blks then
    s.readMemValue .nat (Region.cast block_offset)
      ((s.pids 1 * NUM_ROWS + s.pids 0) * NNZ_S + b)
  else BlockState.defaultCarrier .nat
```
</details>

<details><summary><code>effScale</code></summary>

```
/-- **Faithful exp2→exp scale.** The kernel sets `qk_scale = sm_scale ·
1.44269504` and exponentiates with `exp2`. Since the semantics give
`exp2(x) = exp(x · log 2)`, the per-key weight the loop computes is
`exp2(qk_scale · raw) = exp(qk_scale · log 2 · raw)`. Hence the natural-exp
scale to instantiate the closed form with is
`effScale sm_scale = sm_scale · 1.44269504 · log 2`. (`1.44269504 · log 2 ≈ 1`,
the floating-point approximation of `log2(e) · ln 2 = 1`.) -/
```
```lean
noncomputable def effScale (sm_scale : ℝ) : ℝ :=
  sm_scale * 1.44269504 * Real.log 2
```
</details>

<details><summary><code>vRow</code></summary>

```
/-- V row at global key position `n`, channel `d`, at `kvBase + n·stride_vn + d`. -/
```
```lean
noncomputable def vRow (s : BlockState) (V : RegionName)
    (H stride_vz stride_vh stride_vn : Nat) (n d : Nat) : ℝ :=
  s.readMem V (qoBase s H stride_vz stride_vh + n * stride_vn + d)
```
</details>

<details><summary><code>colKeyGlobal</code></summary>

```
/-- Global key position of the `c`-th visited sparse column:
`column_index[off_hz·NUM_ROWS·NNZ_V + start_m·NNZ_V + c]`. -/
```
```lean
def colKeyGlobal (s : BlockState) (column_index : Region .nat)
    (NUM_ROWS NNZ_V c : Nat) : Nat :=
  s.readMemValue .nat (Region.cast column_index)
    ((s.pids 1 * NUM_ROWS + s.pids 0) * NNZ_V + c)
```
</details>

<details><summary><code>msaScoreLaneA</code></summary>

```
/-- The masked `qk` lane `(i,j)` the Loop-A body computes at block `c`:
`if (offs_m i < seqlen ∧ SN+j ≤ offs_m i) then some(Σ_e fp16(q)·K) else ⊥`. -/
```
```lean
noncomputable def msaScoreLaneA (Q K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (qF : TileIndex [64, 64] → WithBot ℝ) (kpF : TileIndex [64, 1] → RegionName × Nat)
    (s0 : BlockState) (c SN : Nat) (i j : Fin 64) : WithBot ℝ :=
  Option.map₂ (· + ·)
    (if (s0.pids 0 * 64 + i.val < seqLen s0 4 (Region.cast Seqlens)
        ∧ SN + j.val ≤ s0.pids 0 * 64 + i.val) then
      (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))
    (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
      (fun e : Fin 64 => Option.map₂ (· * ·)
        (FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 (qF (i, e, PUnit.unit))))
        (msaKLaneA K Seqlens Blocks BlockOffsets kpF s0 c SN e j)))
```
</details>

<details><summary><code>msaSN0</code></summary>

```
/-- The masked block start `start_n` Loop-A reads at block `c`. -/
```
```lean
noncomputable def msaSN0 (s0 : BlockState) (Blocks BlockOffsets : Region .nat) (c : Nat) : Nat :=
  if c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * 2 + s0.pids 0) then
    s0.readMemValue .nat (Region.cast BlockOffsets) ((s0.pids 1 * 2 + s0.pids 0) * 4 + c)
  else BlockState.defaultCarrier .nat
```
</details>

<details><summary><code>msaScoreLaneB</code></summary>

```
/-- The masked `qk` lane `(i,j)` the Loop-B body computes at loop value `sv`
(NON-causal): `if (offs_m i < seqlen ∧ (sv+j < num_cols ∧ cond)) then some(Σ q·K) else ⊥`. -/
```
```lean
noncomputable def msaScoreLaneB (Blocks ColCounts : Region .nat) (Seqlens : Region .nat)
    (qF : TileIndex [64, 64] → WithBot ℝ) (kpF : TileIndex [64, 1] → RegionName × Nat)
    (s0 : BlockState) (sv : Nat) (gcol : Fin 64 → Nat) (i j : Fin 64) : WithBot ℝ :=
  Option.map₂ (· + ·)
    (if (s0.pids 0 * 64 + i.val < seqLen s0 4 (Region.cast Seqlens)
        ∧ (sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0)
          ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0))) then
      (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))
    (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
      (fun e : Fin 64 => Option.map₂ (· * ·)
        (FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 (qF (i, e, PUnit.unit))))
        (msaKLaneB Blocks ColCounts kpF s0 sv gcol e j)))
```
</details>

<details><summary><code>msaGcol0</code></summary>

```
/-- The gathered columns Loop-B reads at loop value `sv`. -/
```
```lean
noncomputable def msaGcol0 (s0 : BlockState) (Cols ColCounts : Region .nat) (sv : Nat) :
    Fin 64 → Nat :=
  fun j => msaColLaneB Cols ColCounts s0 sv j
```
</details>

<details><summary><code>qRow</code></summary>

```
/-- Q row `start_m·BLOCK_M + i`, channel `e`, at `qoBase + row·stride_qm + e`. -/
```
```lean
noncomputable def qRow (s : BlockState) (Q : RegionName)
    (H stride_qz stride_qh stride_qm BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Nat) :
    ℝ :=
  s.readMem Q (qoBase s H stride_qz stride_qh + mIndex s BLOCK_M i * stride_qm + e)
```
</details>

<details><summary><code>kRow</code></summary>

```
/-- K row at global key position `n`, channel `e`, at `kvBase + n·stride_kn + e`.
The kernel reads K with `k_ptrs = K + kv_offset + offs_d·stride_kk` then
`+ cols·stride_kn`; here `kv_offset = off_z·stride_kz + off_h·stride_kh` and
`stride_kk = 1` (head channel `e` contiguous). -/
```
```lean
noncomputable def kRow (s : BlockState) (K : RegionName)
    (H stride_kz stride_kh stride_kn : Nat) (n e : Nat) : ℝ :=
  s.readMem K (qoBase s H stride_kz stride_kh + n * stride_kn + e)
```
</details>

<details><summary><code>qoBase</code></summary>

```
/-- Q/out tile base offset `off_z · stride_z + off_h · stride_h`. -/
```
```lean
def qoBase (s : BlockState) (H stride_z stride_h : Nat) : Nat :=
  offZ s H * stride_z + offH s H * stride_h
```
</details>

<details><summary><code>msaKLaneA</code></summary>

```
/-- The masked K-load lane `(e,j)` the Loop-A body computes at block `c`:
`if (SN+j < seqlen ∧ cb) then K[kpF(e)·base + (SN+j)·64] else 0`, where `cb`
is the loop guard `c < num_blks`. -/
```
```lean
noncomputable def msaKLaneA (K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (kpF : TileIndex [64, 1] → RegionName × Nat) (s0 : BlockState) (c SN : Nat)
    (e j : Fin 64) : WithBot ℝ :=
  if (SN + j.val < seqLen s0 4 (Region.cast Seqlens)
      ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * 2 + s0.pids 0)) then
    (s0.readMemValue .real (kpF (e, ⟨0, by simp⟩, PUnit.unit)).1
      ((kpF (e, ⟨0, by simp⟩, PUnit.unit)).2 + (SN + j.val) * 64) : WithBot ℝ)
  else (some (0.0 : ℝ) : WithBot ℝ)
```
</details>

<details><summary><code>msaKLaneB</code></summary>

```
/-- The masked K-load lane `(e,j)` at loop value `sv` (column-B): the gathered
column index `gcol j` scales the K pointer; mask is `(sv+j < num_cols) & cond`. -/
```
```lean
noncomputable def msaKLaneB (Blocks ColCounts : Region .nat)
    (kpF : TileIndex [64, 1] → RegionName × Nat) (s0 : BlockState) (sv : Nat)
    (gcol : Fin 64 → Nat) (e j : Fin 64) : WithBot ℝ :=
  if (sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0)
      ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0)) then
    (s0.readMemValue .real (kpF (e, ⟨0, by simp⟩, PUnit.unit)).1
      ((kpF (e, ⟨0, by simp⟩, PUnit.unit)).2 + gcol j * 64) : WithBot ℝ)
  else (some (0.0 : ℝ) : WithBot ℝ)
```
</details>

<details><summary><code>msaColLaneB</code></summary>

```
/-- The gathered column lane `j` at loop value `sv`: `cols_ptr[cpOff + sv + j]`
when the loop guard `sv < num_cols` holds, else `0`. -/
```
```lean
noncomputable def msaColLaneB (Cols : Region .nat) (ColCounts : Region .nat)
    (s0 : BlockState) (sv : Nat) (j : Fin 64) : Nat :=
  if sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0) then
    s0.readMemValue .nat (Region.cast Cols) ((s0.pids 1 * 2 + s0.pids 0) * 8 + sv + j.val)
  else 0
```
</details>

## Public theorem: `mixed_sparse_attention_python_case2_output_closed_form_summary`

<details><summary>docstring</summary>

```
/-- **Genuine Python case 2 closed-form summary** (`BLOCK_M=BLOCK_N=32`,
`BLOCK_DMODEL=64`, `sm_scale=0.1`). The executed surface kernel's `fp16` `Out` cell
at every active output lane equals `some (mixedSparseAttnClosedForm …)` — the
genuine non-self-referential mixed-sparse closed form at the case-2 tile shape, NOT
the store-only `accStoreValue` fact. Faithful side conditions (`num_cols ≤ BLOCK_N =
32` and a positive online-softmax denominator per active lane) are supplied as
hypotheses. -/
```
</details>

**Statement:**
```lean
theorem mixed_sparse_attention_python_case2_output_closed_form_summary
    (Q K V Out : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hactive : s.pids 0 * 32 < seqLen s 4 (Region.cast Seqlens))
    (hNC32 : s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * 2 + s.pids 0) ≤ 32)
    (hpos : ∀ i : Fin 32, s.pids 0 * 32 + i.val < seqLen s 4 (Region.cast Seqlens) →
      0 < msaDenomUpto 32 32
        (msaCatScore032 Q K V Seqlens Blocks BlockOffsets ColCounts Cols s (0.1 : ℝ)) 9 i) :
    ∃ sF, exec (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
        (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
        32768 8192 64 1 32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
        2 4 128 2 4 8 32 32 64 FloatDType.fp16).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [32, 64],
          active s 4 Seqlens 32 idx →
            sF.readMemValue .fp16 Out (outOffset s 4 32768 8192 64 1 32 idx)
              = (some (mixedSparseAttnClosedForm s Q K V BlockOffsets Cols 4
                  32768 8192 64 32768 8192 64 32768 8192 64 2 4 8
                  (s.readMemValue .nat (Region.cast Blocks) (s.pids 1 * 2 + s.pids 0))
                  (s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * 2 + s.pids 0))
                  (seqLen s 4 (Region.cast Seqlens)) 64 32 32 0.1 idx.1 (dIndex idx)) : WithBot ℝ)
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg o, s.undef rg o = 0`
- `hactive : s.pids 0 * 32 < seqLen s 4 (Region.cast Seqlens)`
- `hNC32 : s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * 2 + s.pids 0) ≤ 32`

**Closed-form spec defs (transitive):** `seqLen`, `msaDenomUpto`, `msaCatScore032`, `mixed_sparse_attention_fwd_kernel_surface`, `active`, `outOffset`, `mixedSparseAttnClosedForm`, `dIndex`, `offZ`, `msaE`, `msaCatScore`, `msaScoreA032`, `msaQVal32`, `msaKPtr`, `msaScoreB032`, `mIndex`, `offH`, `rawScore`, `blockStartN`, `effScale`, `vRow`, `colKeyGlobal`, `msaScoreLaneA32`, `msaSN0`, `msaScoreLaneB32`, `msaGcol032`, `qRow`, `kRow`, `qoBase`, `msaKLaneA32`, `msaKLaneB32`, `msaColLaneB32`

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (H : Nat) (Seqlens : RegionName) : Nat :=
  s.readMemValue .nat Seqlens (offZ s H)
```
</details>

<details><summary><code>msaDenomUpto</code></summary>

```
/-- Direct (unshifted) running denominator: `Σ_{l<k} Σ_j exp2(score l i j)`. -/
```
```lean
noncomputable def msaDenomUpto (BM BN : Nat)
    (score : Nat → Fin BM → Fin BN → WithBot ℝ) (k : Nat) (i : Fin BM) : ℝ :=
  (Finset.range k).sum (fun l =>
    (Finset.univ : Finset (Fin BN)).sum (fun j => msaE (score l i j)))
```
</details>

<details><summary><code>msaCatScore032</code></summary>

```
/-- Case-2 pinned cat score stream. -/
```
```lean
noncomputable abbrev msaCatScore032
    (Q K V : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (s0 : BlockState) (sm_scale : ℝ := 0.1) : Nat → Fin 32 → Fin 32 → WithBot ℝ :=
  msaCatScore 32 32 8
    (msaScoreA032 Q K Seqlens Blocks BlockOffsets (msaQVal32 Q s0 sm_scale) (msaKPtr K s0) s0)
    (msaScoreB032 Q K Seqlens Blocks BlockOffsets ColCounts Cols (msaQVal32 Q s0 sm_scale) (msaKPtr K s0) s0)
```
</details>

<details><summary><code>mixed_sparse_attention_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `mixed_sparse_attention.py`'s
`_triton_mixed_sparse_attn_fwd_kernel`. -/
```
```lean
def mixed_sparse_attention_fwd_kernel_surface
    (Q K V : RegionName) (seqlens : Region .nat) (sm_scale : ℝ)
    (block_count block_offset column_count column_index : Region .nat)
    (Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vn stride_vk
      stride_oz stride_oh stride_om stride_ok
      Z H N_CTX NUM_ROWS NNZ_S NNZ_V
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat)
    (dtype : FloatDType) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  seqlen = tl.load(seqlens + off_hz // $(H))
  if start_m * $(BLOCK_M) >= seqlen {
    return
  }

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))

  qo_offset = (off_hz // $(H)) * $(stride_qz) + (off_hz % $(H)) * $(stride_qh)
  kv_offset = (off_hz // $(H)) * $(stride_kz) + (off_hz % $(H)) * $(stride_kh)

  q_ptrs = Q + qo_offset + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  k_ptrs = K + kv_offset + offs_d[:, None] * $(stride_kk)
  v_ptrs = V + kv_offset + offs_d[None, :] * $(stride_vk)
  o_ptrs = Out + qo_offset + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_ok)

  num_blks = tl.load(block_count + off_hz * $(NUM_ROWS) + start_m)
  blks_ptr = block_offset + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_S)
  num_cols = tl.load(column_count + off_hz * $(NUM_ROWS) + start_m)
  cols_ptr = column_index + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_V)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(q_ptrs)
  q = (q * qk_scale).to(DTYPE)

  m_mask = offs_m[:, None] < seqlen

  max_num_blks = $(8)
  for block_index in range(max_num_blks) {
    cond = block_index < num_blks
    start_n = tl.load(blks_ptr + block_index, mask=cond)
    cols = start_n + offs_n
    n_mask = (cols < seqlen) & cond
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    causal_mask = cols[None, :] <= offs_m[:, None]
    qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  max_num_cols = $(16)
  for start_n in range($(0), max_num_cols, $(BLOCK_N)) {
    cond = start_n < num_cols
    n_mask = (start_n + offs_n < num_cols) & cond
    cols = tl.load(cols_ptr + start_n + offs_n, mask=cond[:, None], other=0)
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk = tl.where(m_mask & n_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  acc /= l_i[:, None]
  tl.store(o_ptrs, (acc).to(DTYPE), mask=m_mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (H : Nat) (Seqlens : RegionName) (BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H Seqlens
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_om stride_ok BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_om + dIndex idx * stride_ok
```
</details>

<details><summary><code>mixedSparseAttnClosedForm</code></summary>

```
/-- **Genuine (FAITHFUL) closed-form mixed-sparse attention output** for one
program/row. This mirrors *exactly* what
`_triton_mixed_sparse_attn_fwd_kernel` computes — including the faithfulness
quirk that **Loop A always runs `max_num_blks = 8` iterations regardless of
`num_blks`**.

For each iteration `b < 8` the kernel forms `cond = b < num_blks`, the masked
block start `start_n = blockStartN` (the masked default `0` when `cond` is
false), then for each lane `j < BLOCK_N` the key `n = start_n + j`:

* the K-load is masked by `n_mask = (n < seqlen) ∧ cond`, so the effective key
  vector is `K[n]` when `n < seqlen ∧ cond` and the **zero vector** otherwise;
* `qk = where(m_mask ∧ (n ≤ offs_m i), 0, -inf) + dot(q, K_masked)`, so the lane
  contributes weight `w = exp(effScale · rawMasked)` exactly when
  `offs_m i < seqlen ∧ n ≤ offs_m i`, and `0` otherwise. Here `rawMasked = raw n`
  when `n < seqlen ∧ cond` and `rawMasked = 0` (so `w = exp(0) = 1`) otherwise —
  this is the **spurious-block weight-1 path**: a block `b ≥ num_blks` has
  `start_n = 0`, `cond = false`, hence `n = j`, `n ≤ offs_m i` and
  (for active rows) `offs_m i < seqlen`, so it adds `exp(effScale·0) = 1` to the
  DENOMINATOR while its V is the zero vector, leaving the numerator unchanged.

Loop B (column phase) is correctly `n_mask`-guarded (its `where` masks
non-selected lanes to `⊥`), so spurious column lanes contribute nothing.

`numer/denom` is therefore the kernel's true output, **not** the naive
`num_blks`-only union softmax. The natural-exp scale is `effScale sm_scale =
sm_scale · 1.44269504 · log 2` (the faithful `exp2 → exp` bridge). -/
```
```lean
noncomputable def mixedSparseAttnClosedForm
    (s : BlockState) (Q K V : RegionName)
    (block_offset column_index : Region .nat)
    (H stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
      stride_vz stride_vh stride_vn
      NUM_ROWS NNZ_S NNZ_V
      num_blks num_cols seqlen
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (sm_scale : ℝ) (i : Fin BLOCK_M) (d : Nat) : ℝ :=
  let raw := fun n : Nat =>
    rawScore s Q K H stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
      BLOCK_DMODEL BLOCK_M i n
  -- block-sparse phase over ALL 8 = max_num_blks kernel iterations.
  -- `keep` = lane kept (causal + active row); `inSeq` = K/V actually loaded.
  let wBlock := fun (b : Fin 8) (j : Fin BLOCK_N) =>
    let SN := blockStartN s block_offset NUM_ROWS NNZ_S num_blks b.val
    let n := SN + j.val
    let inSeq := n < seqlen ∧ b.val < num_blks
    let rawMasked := if inSeq then raw n else 0
    if mIndex s BLOCK_M i < seqlen ∧ n ≤ mIndex s BLOCK_M i then
      Real.exp (effScale sm_scale * rawMasked) else 0
  let vBlock := fun (b : Fin 8) (j : Fin BLOCK_N) =>
    let SN := blockStartN s block_offset NUM_ROWS NNZ_S num_blks b.val
    let n := SN + j.val
    if n < seqlen ∧ b.val < num_blks then
      vRow s V H stride_vz stride_vh stride_vn n d else 0
  -- column-sparse phase weights. Faithful because the kernel `n_mask`-guards
  -- Loop B's `where`: a column lane `c < num_cols` is kept iff the row is active
  -- (`offs_m i < seqlen`). The kernel applies NO `cols < seqlen` mask to the
  -- column keys (only `c < num_cols ∧ 0 < num_cols`), so neither does this.
  let wCol := fun (c : Fin num_cols) =>
    let n := colKeyGlobal s column_index NUM_ROWS NNZ_V c.val
    if mIndex s BLOCK_M i < seqlen then
      Real.exp (effScale sm_scale * raw n) else 0
  let denom :=
    Finset.univ.sum (fun b : Fin 8 =>
      Finset.univ.sum (fun j : Fin BLOCK_N => wBlock b j)) +
    Finset.univ.sum (fun c : Fin num_cols => wCol c)
  let numer :=
    Finset.univ.sum (fun b : Fin 8 =>
      Finset.univ.sum (fun j : Fin BLOCK_N => wBlock b j * vBlock b j)) +
    Finset.univ.sum (fun c : Fin num_cols =>
      wCol c *
        vRow s V H stride_vz stride_vh stride_vn
          (colKeyGlobal s column_index NUM_ROWS NNZ_V c.val) d)
  numer / denom
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>offZ</code></summary>

```lean
def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>msaE</code></summary>

```
/-- `msaE x = exp2(x)` with `exp2(⊥) = 0` — the per-key softmax weight carrier. -/
```
```lean
noncomputable def msaE (x : WithBot ℝ) : ℝ := (WithBot.realExp2 x).unbotD 0
```
</details>

<details><summary><code>msaCatScore</code></summary>

```
/-- Concatenated score stream: first `bF` iterations from `scoreA`, then `scoreB`. -/
```
```lean
noncomputable def msaCatScore (BM BN bF : Nat)
    (scoreA scoreB : Nat → Fin BM → Fin BN → WithBot ℝ) :
    Nat → Fin BM → Fin BN → WithBot ℝ :=
  fun k => if k < bF then scoreA k else scoreB (k - bF)
```
</details>

<details><summary><code>msaScoreA032</code></summary>

```
/-- Case-2 concrete block-A score stream. -/
```
```lean
noncomputable def msaScoreA032 (Q K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (qF : TileIndex [32, 64] → WithBot ℝ) (kpF : TileIndex [64, 1] → RegionName × Nat)
    (s0 : BlockState) : Nat → Fin 32 → Fin 32 → WithBot ℝ :=
  fun c i j => msaScoreLaneA32 Q K Seqlens Blocks BlockOffsets qF kpF s0 c
    (msaSN0 s0 Blocks BlockOffsets c) i j
```
</details>

<details><summary><code>msaQVal32</code></summary>

```
/-- Case-2 scaled `q` value lane `(i,e)`. -/
```
```lean
noncomputable def msaQVal32 (Q : RegionName) (s0 : BlockState)
    (sm_scale : ℝ := 0.1) : TileIndex [32, 64] → WithBot ℝ :=
  fun idx => WithBot.realMul
    (s0.readMemValue .real Q (((s0.pids 1 / 4) * 32768 + (s0.pids 1 % 4) * 8192)
      + (s0.pids 0 * 32 + idx.1.val) * 64 + idx.2.1.val))
    (some (sm_scale * 1.44269504))
```
</details>

<details><summary><code>msaKPtr</code></summary>

```
/-- The `k_ptrs` pointer tile `(e channel axis, j=1)`: `K` at `kv_offset + e·1`. -/
```
```lean
def msaKPtr (K : RegionName) (s0 : BlockState) : TileIndex [64, 1] → RegionName × Nat :=
  fun idx => (K, ((s0.pids 1 / 4) * 32768 + (s0.pids 1 % 4) * 8192) + idx.1.val)
```
</details>

<details><summary><code>msaScoreB032</code></summary>

```
/-- Case-2 concrete column-B score stream (loop value `sv = c·32`). -/
```
```lean
noncomputable def msaScoreB032 (Q K : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (qF : TileIndex [32, 64] → WithBot ℝ) (kpF : TileIndex [64, 1] → RegionName × Nat)
    (s0 : BlockState) : Nat → Fin 32 → Fin 32 → WithBot ℝ :=
  fun c i j => msaScoreLaneB32 Blocks ColCounts Seqlens qF kpF s0 (c * 32)
    (msaGcol032 s0 Cols ColCounts (c * 32)) i j
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>offH</code></summary>

```lean
def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>rawScore</code></summary>

```
/-- Unscaled raw score `Σ_{e<BLOCK_DMODEL} Q[row,e] · K[n,e]` at global key `n`. -/
```
```lean
noncomputable def rawScore (s : BlockState) (Q K : RegionName)
    (H stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
      BLOCK_DMODEL BLOCK_M : Nat) (i : Fin BLOCK_M) (n : Nat) : ℝ :=
  Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
    qRow s Q H stride_qz stride_qh stride_qm BLOCK_M i e.val *
      kRow s K H stride_kz stride_kh stride_kn n e.val)
```
</details>

<details><summary><code>blockStartN</code></summary>

```
/-- The masked block start `start_n` the kernel reads at block `b` (Loop A's
`tl.load(blks_ptr + b, mask = b < num_blks)`): the real offset for a visited
block, the masked default `0` for a spurious block `b ≥ num_blks`. -/
```
```lean
noncomputable def blockStartN (s : BlockState) (block_offset : Region .nat)
    (NUM_ROWS NNZ_S num_blks b : Nat) : Nat :=
  if b < num_blks then
    s.readMemValue .nat (Region.cast block_offset)
      ((s.pids 1 * NUM_ROWS + s.pids 0) * NNZ_S + b)
  else BlockState.defaultCarrier .nat
```
</details>

<details><summary><code>effScale</code></summary>

```
/-- **Faithful exp2→exp scale.** The kernel sets `qk_scale = sm_scale ·
1.44269504` and exponentiates with `exp2`. Since the semantics give
`exp2(x) = exp(x · log 2)`, the per-key weight the loop computes is
`exp2(qk_scale · raw) = exp(qk_scale · log 2 · raw)`. Hence the natural-exp
scale to instantiate the closed form with is
`effScale sm_scale = sm_scale · 1.44269504 · log 2`. (`1.44269504 · log 2 ≈ 1`,
the floating-point approximation of `log2(e) · ln 2 = 1`.) -/
```
```lean
noncomputable def effScale (sm_scale : ℝ) : ℝ :=
  sm_scale * 1.44269504 * Real.log 2
```
</details>

<details><summary><code>vRow</code></summary>

```
/-- V row at global key position `n`, channel `d`, at `kvBase + n·stride_vn + d`. -/
```
```lean
noncomputable def vRow (s : BlockState) (V : RegionName)
    (H stride_vz stride_vh stride_vn : Nat) (n d : Nat) : ℝ :=
  s.readMem V (qoBase s H stride_vz stride_vh + n * stride_vn + d)
```
</details>

<details><summary><code>colKeyGlobal</code></summary>

```
/-- Global key position of the `c`-th visited sparse column:
`column_index[off_hz·NUM_ROWS·NNZ_V + start_m·NNZ_V + c]`. -/
```
```lean
def colKeyGlobal (s : BlockState) (column_index : Region .nat)
    (NUM_ROWS NNZ_V c : Nat) : Nat :=
  s.readMemValue .nat (Region.cast column_index)
    ((s.pids 1 * NUM_ROWS + s.pids 0) * NNZ_V + c)
```
</details>

<details><summary><code>msaScoreLaneA32</code></summary>

```
/-- Case-2 masked `qk` lane `(i,j)` (Loop A). `i,j : Fin 32`, channel `e : Fin 64`. -/
```
```lean
noncomputable def msaScoreLaneA32 (Q K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (qF : TileIndex [32, 64] → WithBot ℝ) (kpF : TileIndex [64, 1] → RegionName × Nat)
    (s0 : BlockState) (c SN : Nat) (i j : Fin 32) : WithBot ℝ :=
  Option.map₂ (· + ·)
    (if (s0.pids 0 * 32 + i.val < seqLen s0 4 (Region.cast Seqlens)
        ∧ SN + j.val ≤ s0.pids 0 * 32 + i.val) then
      (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))
    (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
      (fun e : Fin 64 => Option.map₂ (· * ·)
        (FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 (qF (i, e, PUnit.unit))))
        (msaKLaneA32 K Seqlens Blocks BlockOffsets kpF s0 c SN e j)))
```
</details>

<details><summary><code>msaSN0</code></summary>

```
/-- The masked block start `start_n` Loop-A reads at block `c`. -/
```
```lean
noncomputable def msaSN0 (s0 : BlockState) (Blocks BlockOffsets : Region .nat) (c : Nat) : Nat :=
  if c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * 2 + s0.pids 0) then
    s0.readMemValue .nat (Region.cast BlockOffsets) ((s0.pids 1 * 2 + s0.pids 0) * 4 + c)
  else BlockState.defaultCarrier .nat
```
</details>

<details><summary><code>msaScoreLaneB32</code></summary>

```
/-- Case-2 masked `qk` lane `(i,j)` (Loop B, NON-causal). -/
```
```lean
noncomputable def msaScoreLaneB32 (Blocks ColCounts : Region .nat) (Seqlens : Region .nat)
    (qF : TileIndex [32, 64] → WithBot ℝ) (kpF : TileIndex [64, 1] → RegionName × Nat)
    (s0 : BlockState) (sv : Nat) (gcol : Fin 32 → Nat) (i j : Fin 32) : WithBot ℝ :=
  Option.map₂ (· + ·)
    (if (s0.pids 0 * 32 + i.val < seqLen s0 4 (Region.cast Seqlens)
        ∧ (sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0)
          ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0))) then
      (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))
    (@Finset.sum (Fin 64) (WithBot ℝ) _ Finset.univ
      (fun e : Fin 64 => Option.map₂ (· * ·)
        (FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 (qF (i, e, PUnit.unit))))
        (msaKLaneB32 Blocks ColCounts kpF s0 sv gcol e j)))
```
</details>

<details><summary><code>msaGcol032</code></summary>

```
/-- Case-2 gathered columns at loop value `sv`. -/
```
```lean
noncomputable def msaGcol032 (s0 : BlockState) (Cols ColCounts : Region .nat) (sv : Nat) :
    Fin 32 → Nat :=
  fun j => msaColLaneB32 Cols ColCounts s0 sv j
```
</details>

<details><summary><code>qRow</code></summary>

```
/-- Q row `start_m·BLOCK_M + i`, channel `e`, at `qoBase + row·stride_qm + e`. -/
```
```lean
noncomputable def qRow (s : BlockState) (Q : RegionName)
    (H stride_qz stride_qh stride_qm BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Nat) :
    ℝ :=
  s.readMem Q (qoBase s H stride_qz stride_qh + mIndex s BLOCK_M i * stride_qm + e)
```
</details>

<details><summary><code>kRow</code></summary>

```
/-- K row at global key position `n`, channel `e`, at `kvBase + n·stride_kn + e`.
The kernel reads K with `k_ptrs = K + kv_offset + offs_d·stride_kk` then
`+ cols·stride_kn`; here `kv_offset = off_z·stride_kz + off_h·stride_kh` and
`stride_kk = 1` (head channel `e` contiguous). -/
```
```lean
noncomputable def kRow (s : BlockState) (K : RegionName)
    (H stride_kz stride_kh stride_kn : Nat) (n e : Nat) : ℝ :=
  s.readMem K (qoBase s H stride_kz stride_kh + n * stride_kn + e)
```
</details>

<details><summary><code>qoBase</code></summary>

```
/-- Q/out tile base offset `off_z · stride_z + off_h · stride_h`. -/
```
```lean
def qoBase (s : BlockState) (H stride_z stride_h : Nat) : Nat :=
  offZ s H * stride_z + offH s H * stride_h
```
</details>

<details><summary><code>msaKLaneA32</code></summary>

```
/-- Case-2 masked K-load lane `(e,j)` (Loop A). `e : Fin 64` (channel), `j : Fin 32`. -/
```
```lean
noncomputable def msaKLaneA32 (K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (kpF : TileIndex [64, 1] → RegionName × Nat) (s0 : BlockState) (c SN : Nat)
    (e : Fin 64) (j : Fin 32) : WithBot ℝ :=
  if (SN + j.val < seqLen s0 4 (Region.cast Seqlens)
      ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * 2 + s0.pids 0)) then
    (s0.readMemValue .real (kpF (e, ⟨0, by simp⟩, PUnit.unit)).1
      ((kpF (e, ⟨0, by simp⟩, PUnit.unit)).2 + (SN + j.val) * 64) : WithBot ℝ)
  else (some (0.0 : ℝ) : WithBot ℝ)
```
</details>

<details><summary><code>msaKLaneB32</code></summary>

```
/-- Case-2 masked K-load lane `(e,j)` (Loop B). -/
```
```lean
noncomputable def msaKLaneB32 (Blocks ColCounts : Region .nat)
    (kpF : TileIndex [64, 1] → RegionName × Nat) (s0 : BlockState) (sv : Nat)
    (gcol : Fin 32 → Nat) (e : Fin 64) (j : Fin 32) : WithBot ℝ :=
  if (sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0)
      ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0)) then
    (s0.readMemValue .real (kpF (e, ⟨0, by simp⟩, PUnit.unit)).1
      ((kpF (e, ⟨0, by simp⟩, PUnit.unit)).2 + gcol j * 64) : WithBot ℝ)
  else (some (0.0 : ℝ) : WithBot ℝ)
```
</details>

<details><summary><code>msaColLaneB32</code></summary>

```
/-- Case-2 gathered column at loop value `sv`. `j : Fin 32`. -/
```
```lean
noncomputable def msaColLaneB32 (Cols : Region .nat) (ColCounts : Region .nat)
    (s0 : BlockState) (sv : Nat) (j : Fin 32) : Nat :=
  if sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * 2 + s0.pids 0) then
    s0.readMemValue .nat (Region.cast Cols) ((s0.pids 1 * 2 + s0.pids 0) * 8 + sv + j.val)
  else 0
```
</details>

## Also present (pinned special-case summaries)
- `mixed_sparse_attention_output_store_slice_compute_correct`
- `mixed_sparse_attention_output_store_python_block64_compute_correct`
- `mixed_sparse_attention_output_store_python_block32_compute_correct`
