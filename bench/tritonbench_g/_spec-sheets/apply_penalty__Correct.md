# Spec sheet — `bench/tritonbench_g/apply_penalty/Correct.lean`

**Python source:** `bench/tritonbench_g/apply_penalty/apply_penalty.py`

## Public theorem: `apply_penalty_correctness`

<details><summary>docstring</summary>

```
/-- The masked contract requires distinct scatter addresses on active lanes.
Inactive padded ids may repeat. Bounds apply to the five scalar-slot loads and
the actual active gather/scatter addresses. All other memory is preserved. -/
```
</details>

**Statement:**
```lean
specification apply_penalty_correctness
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    Spec.Real (applyPenaltyIO Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
      ⊨ fun _ _ presence frequency repetition _ _ _ counts logits j =>
        TiledOptimizer.lionPenalty (logits j) (counts j : ℝ) repetition frequency presence)
```

**Closed-form spec defs (transitive):** `applyPenaltyIO`, `apply_penalty`

<details><summary><code>applyPenaltyIO</code></summary>

```
/-- Scalar penalties at pid; cumsum bounds at pid and pid+1; ids and counts at
start+j. Logits are read and overwritten at pid * stride_logit_b + ids[j], with
the same tail mask start+j < stop. The IO contract checks injectivity only on
active write lanes, so padding does not need distinct token ids. -/
```
```lean
def applyPenaltyIO
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    MetaScatterMasked2DKernelIO₁ where
  kernel := apply_penalty Logits presence_penalty freqency_penalty
    repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
    stride_logit_b stride_logit_s BLOCK_P
  fbuf1 := presence_penalty
  fbuf2 := freqency_penalty
  fbuf3 := repetition_penalty
  mbuf := p_cumsum_seq_len
  idbuf := p_token_ids
  cntbuf := p_token_counts
  inp := Logits
  out := Logits
  B := BLOCK_P
  fwin1 := fun pid₀ _ => pid₀
  fwin2 := fun pid₀ _ => pid₀
  fwin3 := fun pid₀ _ => pid₀
  mwin1 := fun pid₀ _ => pid₀
  mwin2 := fun pid₀ _ => pid₀ + 1
  readi := fun _ _ m₁ _ j => m₁ + j.val
  readc := fun _ _ m₁ _ j => m₁ + j.val
  read := fun pid₀ _ _ _ ids j => pid₀ * stride_logit_b + ids j
  mask := fun _ _ m₁ m₂ j => m₁ + j.val < m₂
  write := fun pid₀ _ _ _ ids j => pid₀ * stride_logit_b + ids j
```
</details>

<details><summary><code>apply_penalty</code></summary>

```
/-- Faithful transcription of `apply_penalty.py`'s `_fwd_kernel_apply_penalty`.
Ids, counts and cumsum metadata use Nat channels; the count is promoted to real
by the original multiplication expression. -/
```
```lean
def apply_penalty
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b _stride_logit_s BLOCK_P : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_freqency = tl.load(freqency_penalty + cur_batch)
  cur_presence = tl.load(presence_penalty + cur_batch)
  cur_repetition = tl.load(repetition_penalty + cur_batch)
  cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch)
  cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1)
  cur_batch_id_offset = cur_batch_start_index + tl.arange(0, $(BLOCK_P))
  batch_ids = tl.load(p_token_ids + cur_batch_id_offset,
    mask=cur_batch_id_offset < cur_batch_end_index, other=0)
  batch_ids_count = tl.load(p_token_counts + cur_batch_id_offset,
    mask=cur_batch_id_offset < cur_batch_end_index, other=0)
  row_start_ptr = Logits + cur_batch * $(stride_logit_b)
  cur_offset = row_start_ptr + batch_ids
  cur_logits = tl.load(cur_offset,
    mask=cur_batch_id_offset < cur_batch_end_index, other=0.0)
  rep_logits = tl.where(cur_logits > 0, cur_logits / cur_repetition,
    cur_logits * cur_repetition)
  freq_logits = rep_logits - batch_ids_count * cur_freqency
  pre_logits = freq_logits - cur_presence
  output_ptr = Logits + cur_batch * $(stride_logit_b) + batch_ids
  tl.store(output_ptr, pre_logits,
    mask=cur_batch_id_offset < cur_batch_end_index)
}
```
</details>

## Public theorem: `apply_penalty_reciprocal_correctness`

<details><summary>docstring</summary>

```
/-- The masked contract requires distinct scatter addresses on active lanes.
Inactive padded ids may repeat. Bounds apply to the five scalar-slot loads and
the actual active gather/scatter addresses. All other memory is preserved. -/
```
</details>

**Statement:**
```lean
specification apply_penalty_reciprocal_correctness
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    Spec.Real (reciprocalIO Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
      ⊨ fun _ _ presence frequency repetition _ _ _ counts logits j =>
        TiledOptimizer.lionPenalty (logits j) (counts j : ℝ) repetition frequency presence)
```

**Closed-form spec defs (transitive):** `reciprocalIO`, `applyPenaltyIO`, `apply_penalty_reciprocal`, `apply_penalty`

<details><summary><code>reciprocalIO</code></summary>

```
/-- Identical metadata slots, masks, in-place gather/scatter and active-index requirement. -/
```
```lean
def reciprocalIO (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) : MetaScatterMasked2DKernelIO₁ :=
  { applyPenaltyIO Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P with
    kernel := apply_penalty_reciprocal Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
    projection := by rfl }
```
</details>

<details><summary><code>applyPenaltyIO</code></summary>

```
/-- Scalar penalties at pid; cumsum bounds at pid and pid+1; ids and counts at
start+j. Logits are read and overwritten at pid * stride_logit_b + ids[j], with
the same tail mask start+j < stop. The IO contract checks injectivity only on
active write lanes, so padding does not need distinct token ids. -/
```
```lean
def applyPenaltyIO
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    MetaScatterMasked2DKernelIO₁ where
  kernel := apply_penalty Logits presence_penalty freqency_penalty
    repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
    stride_logit_b stride_logit_s BLOCK_P
  fbuf1 := presence_penalty
  fbuf2 := freqency_penalty
  fbuf3 := repetition_penalty
  mbuf := p_cumsum_seq_len
  idbuf := p_token_ids
  cntbuf := p_token_counts
  inp := Logits
  out := Logits
  B := BLOCK_P
  fwin1 := fun pid₀ _ => pid₀
  fwin2 := fun pid₀ _ => pid₀
  fwin3 := fun pid₀ _ => pid₀
  mwin1 := fun pid₀ _ => pid₀
  mwin2 := fun pid₀ _ => pid₀ + 1
  readi := fun _ _ m₁ _ j => m₁ + j.val
  readc := fun _ _ m₁ _ j => m₁ + j.val
  read := fun pid₀ _ _ _ ids j => pid₀ * stride_logit_b + ids j
  mask := fun _ _ m₁ m₂ j => m₁ + j.val < m₂
  write := fun pid₀ _ _ _ ids j => pid₀ * stride_logit_b + ids j
```
</details>

<details><summary><code>apply_penalty_reciprocal</code></summary>

```
/-- The reciprocal candidate keeps the sign comparison and the negative arm. -/
```
```lean
def apply_penalty_reciprocal
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b _stride_logit_s BLOCK_P : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_freqency = tl.load(freqency_penalty + cur_batch)
  cur_presence = tl.load(presence_penalty + cur_batch)
  cur_repetition = tl.load(repetition_penalty + cur_batch)
  cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch)
  cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1)
  cur_batch_id_offset = cur_batch_start_index + tl.arange(0, $(BLOCK_P))
  batch_ids = tl.load(p_token_ids + cur_batch_id_offset,
    mask=cur_batch_id_offset < cur_batch_end_index, other=0)
  batch_ids_count = tl.load(p_token_counts + cur_batch_id_offset,
    mask=cur_batch_id_offset < cur_batch_end_index, other=0)
  row_start_ptr = Logits + cur_batch * $(stride_logit_b)
  cur_offset = row_start_ptr + batch_ids
  cur_logits = tl.load(cur_offset,
    mask=cur_batch_id_offset < cur_batch_end_index, other=0.0)
  rep_logits = tl.where(cur_logits > 0, cur_logits * (1 / cur_repetition),
    cur_logits * cur_repetition)
  freq_logits = rep_logits - batch_ids_count * cur_freqency
  pre_logits = freq_logits - cur_presence
  output_ptr = Logits + cur_batch * $(stride_logit_b) + batch_ids
  tl.store(output_ptr, pre_logits,
    mask=cur_batch_id_offset < cur_batch_end_index)
}
```
</details>

<details><summary><code>apply_penalty</code></summary>

```
/-- Faithful transcription of `apply_penalty.py`'s `_fwd_kernel_apply_penalty`.
Ids, counts and cumsum metadata use Nat channels; the count is promoted to real
by the original multiplication expression. -/
```
```lean
def apply_penalty
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b _stride_logit_s BLOCK_P : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_freqency = tl.load(freqency_penalty + cur_batch)
  cur_presence = tl.load(presence_penalty + cur_batch)
  cur_repetition = tl.load(repetition_penalty + cur_batch)
  cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch)
  cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1)
  cur_batch_id_offset = cur_batch_start_index + tl.arange(0, $(BLOCK_P))
  batch_ids = tl.load(p_token_ids + cur_batch_id_offset,
    mask=cur_batch_id_offset < cur_batch_end_index, other=0)
  batch_ids_count = tl.load(p_token_counts + cur_batch_id_offset,
    mask=cur_batch_id_offset < cur_batch_end_index, other=0)
  row_start_ptr = Logits + cur_batch * $(stride_logit_b)
  cur_offset = row_start_ptr + batch_ids
  cur_logits = tl.load(cur_offset,
    mask=cur_batch_id_offset < cur_batch_end_index, other=0.0)
  rep_logits = tl.where(cur_logits > 0, cur_logits / cur_repetition,
    cur_logits * cur_repetition)
  freq_logits = rep_logits - batch_ids_count * cur_freqency
  pre_logits = freq_logits - cur_presence
  output_ptr = Logits + cur_batch * $(stride_logit_b) + batch_ids
  tl.store(output_ptr, pre_logits,
    mask=cur_batch_id_offset < cur_batch_end_index)
}
```
</details>
