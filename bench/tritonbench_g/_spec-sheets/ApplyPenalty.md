# Spec sheet — `bench/tritonbench_g/apply_penalty/ApplyPenalty.lean`

**Python source:** `bench/tritonbench_g/apply_penalty/apply_penalty.py`

## Public theorem: `apply_penalty_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `apply_penalty` implements the pure Lion penalty on
its penalty gather–scatter IO signature — for every disjoint flat placement
of the buffers, every program `cur_batch = pid₀` whose slot cells and
active lanes are in bounds, and every launch state whose penalty slots hold
`g₁`/`g₂`/`g₃`, whose cumsum cells hold `m₁`/`m₂`, whose token-id/count
windows hold `ids`/`cnts`, and whose gathered `Logits` window holds `xs`,
the translated pointer kernel terminates, every active (`m₁ + j < m₂`)
scatter lane `pid₀ * stride_logit_b + ids j` of `Logits` holds
`penaltyValuePure g₁ g₂ g₃ cnts xs j` — **guarded by the skin's `WriteInj`
antecedent** (distinct active token ids over the pinned values: the
write-map injectivity the pre-`⊨` summary carried as its `hUniq`
hypothesis) — and every other memory cell is unchanged. Proof:
`MetaScatterMasked2DKernelIO₁.Implements.intro` assembles the region-model
triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification apply_penalty_correctness
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    applyPenaltyIO Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b stride_logit_s BLOCK_P
      ⊨ fun _ _ g₁ g₂ g₃ _ _ _ cnts xs j =>
          penaltyValuePure g₁ g₂ g₃ cnts xs j
```

**Closed-form spec defs (transitive):** `applyPenaltyIO`, `penaltyValuePure`, `apply_penalty`

<details><summary><code>applyPenaltyIO</code></summary>

```
/-- `apply_penalty`'s **penalty gather–scatter IO signature** — the whole
kernel-specific audit surface of the `⊨` headline
(`MetaScatterMasked2DKernelIO₁`, the metadata genre's `Meta` slots +
`Scatter` writes skin):

* `fbuf1`/`fbuf2`/`fbuf3` — the three per-batch float penalty slots
  (`presence_penalty`/`freqency_penalty`/`repetition_penalty`), all read at
  cell `pid₀` (`fwin1 = fwin2 = fwin3 = pid₀`), yielding the named scalars
  `g₁ = cur_presence`, `g₂ = cur_freqency`, `g₃ = cur_repetition`;
* `mbuf = p_cumsum_seq_len` — the `.nat` metadata buffer carrying both
  cumsum slots: `mwin1 = pid₀` and `mwin2 = pid₀ + 1`, yielding
  `m₁ = cur_batch_start_index` and `m₂ = cur_batch_end_index`;
* `idbuf = p_token_ids`, `cntbuf = p_token_counts` — the `.nat` tiles, lane
  `j` at the batch-window cell `m₁ + j` (`readi = readc`), yielding the
  index tile `ids` (= `batch_ids`) and the counts `cnts`;
* `inp = out = Logits` — the float data channel is gather-read **and**
  scatter-written **in place** (duplicate-region wiring), lane `j` at the
  data-dependent address `pid₀ * stride_logit_b + ids j` (`read = write`);
  `B = BLOCK_P`;
* `mask` — the active lanes `m₁ + j < m₂`
  (`cur_batch_id_offset < cur_batch_end_index`), shared by all masked
  accesses (`writeMask` defaults to `mask`).

`stride_logit_s` is unused by the kernel body (rows are contiguous), and
the 1-D launch ignores the family's second program id. The slot cells,
windows, and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual slot loads, addressing, and masking match
them. Buffer sizes are not signature content: the headline quantifies over
every allocation whose extents cover the slot cells and the active lanes. -/
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

<details><summary><code>penaltyValuePure</code></summary>

```
/-- Per-lane `Logits` output spec as a **pure** function of the pinned values
of the `⊨` headline: the reusable Lion penalty oracle
(`VeriTile.Triton.Math.Optimizer.lionPenalty`) applied to the gathered logit
`xs j`, its count `cnts j`, and the three loaded penalty scalars —
repetition `g₃`, frequency `g₂`, presence `g₁`. This is `penaltyValue` with
the state-coupled reads replaced by the named binders of the headline. -/
```
```lean
noncomputable def penaltyValuePure (g₁ g₂ g₃ : ℝ)
    (cnts : Fin BLOCK_P → Nat) (xs : Fin BLOCK_P → ℝ) (j : Fin BLOCK_P) : ℝ :=
  TiledOptimizer.lionPenalty (xs j) (cnts j : ℝ) g₃ g₂ g₁
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
