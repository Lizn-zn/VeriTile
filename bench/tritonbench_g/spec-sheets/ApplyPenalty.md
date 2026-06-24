# Spec sheet — `bench/tritonbench_g/apply_penalty/ApplyPenalty.lean`

**Python source:** `bench/tritonbench_g/apply_penalty/apply_penalty.py`

## Public theorem: `apply_penalty_output_summary`

<details><summary>docstring</summary>

```
/-- **Per-kernel output summary for `apply_penalty`.** The DSL surface lowers to
the algorithm layer, and (under distinct active token ids) the masked in-place
`Logits` store is compute-correct against the Lion penalty value. -/
```
</details>

**Statement:**
```lean
theorem apply_penalty_output_summary
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) (s : BlockState)
    (hUniq : ∀ i j : Fin BLOCK_P,
      active s p_cumsum_seq_len i → active s p_cumsum_seq_len j →
      tokenId s p_token_ids p_cumsum_seq_len i =
        tokenId s p_token_ids p_cumsum_seq_len j → i = j) :
    (∃ alg, (apply_penalty Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := apply_penalty Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b stride_logit_s BLOCK_P)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_P => active s p_cumsum_seq_len i)
          (fun i => ((Logits : RegionName),
            activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b i)))
      (expected := fun i => penaltyValue s Logits presence_penalty
        freqency_penalty repetition_penalty p_token_ids p_token_counts
        p_cumsum_seq_len stride_logit_b i)
```

**Assumptions / layout contracts:**
- `hUniq : ∀ i j : Fin BLOCK_P,
      active s p_cumsum_seq_len i → active s p_cumsum_seq_len j →
      tokenId s p_token_ids p_cumsum_seq_len i =
        tokenId s p_token_ids p_cumsum_seq_len j → i = j`
- `fun i : Fin BLOCK_P => active s p_cumsum_seq_len i`

**Closed-form spec defs (transitive):** `active`, `tokenId`, `apply_penalty`, `activeStoreAddr`, `penaltyValue`, `tokenOffset`, `batchEnd`, `batchStart`

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (p_cumsum_seq_len : RegionName)
    (i : Fin BLOCK_P) : Prop :=
  tokenOffset s p_cumsum_seq_len i < batchEnd s p_cumsum_seq_len
```
</details>

<details><summary><code>tokenId</code></summary>

```lean
def tokenId (s : BlockState) (p_token_ids p_cumsum_seq_len : RegionName)
    (i : Fin BLOCK_P) : Nat :=
  s.readMemValue .nat p_token_ids (tokenOffset s p_cumsum_seq_len i)
```
</details>

<details><summary><code>apply_penalty</code></summary>

```
/-- Faithful transcription of `apply_penalty.py`'s
`_fwd_kernel_apply_penalty`.

`p_token_counts` is loaded as a Nat channel; the DSL infers the
integer-to-float promotion in `batch_ids_count * cur_freqency`, matching the
Python surface expression without adding an explicit cast. -/
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

<details><summary><code>activeStoreAddr</code></summary>

```
/-- The "active" store address (no mask conditional) used for the readback
spec. When `active s i` holds, this equals `storeOffset s ... i`. -/
```
```lean
def activeStoreAddr
    (s : BlockState) (p_token_ids p_cumsum_seq_len : RegionName)
    (stride_logit_b : Nat) (i : Fin BLOCK_P) : Nat :=
  s.pids 0 * stride_logit_b + tokenId s p_token_ids p_cumsum_seq_len i
```
</details>

<details><summary><code>penaltyValue</code></summary>

```
/-- Per-lane `Logits` output spec: the reusable Lion penalty oracle
(`VeriTile.Triton.Math.Optimizer.lionPenalty`) applied to the values this lane
loads — the logit at the gathered token, its count, and the three penalties. -/
```
```lean
noncomputable def penaltyValue
    (s : BlockState)
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b : Nat) (i : Fin BLOCK_P) : ℝ :=
  TiledOptimizer.lionPenalty
    (s.readMem Logits
      (s.pids 0 * stride_logit_b + tokenId s p_token_ids p_cumsum_seq_len i))
    (s.readMemValue .nat p_token_counts (tokenOffset s p_cumsum_seq_len i) : ℝ)
    (s.readMem repetition_penalty (s.pids 0))
    (s.readMem freqency_penalty (s.pids 0))
    (s.readMem presence_penalty (s.pids 0))
```
</details>

<details><summary><code>tokenOffset</code></summary>

```lean
def tokenOffset (s : BlockState) (p_cumsum_seq_len : RegionName)
    (i : Fin BLOCK_P) : Nat :=
  batchStart s p_cumsum_seq_len + i.val
```
</details>

<details><summary><code>batchEnd</code></summary>

```lean
def batchEnd (s : BlockState) (p_cumsum_seq_len : RegionName) : Nat :=
  s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1)
```
</details>

<details><summary><code>batchStart</code></summary>

```lean
def batchStart (s : BlockState) (p_cumsum_seq_len : RegionName) : Nat :=
  s.readMemValue .nat p_cumsum_seq_len (s.pids 0)
```
</details>

## Also present (pinned special-case summaries)
- `apply_penalty_compute_correct`
