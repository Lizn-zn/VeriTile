# Spec sheet — `bench/tritonbench_g/token_softmax_bloom/TokenSoftmaxBloom.lean`

**Python source:** `bench/tritonbench_g/token_softmax_bloom/token_softmax_bloom.py`

## Public theorem: `token_softmax_bloom_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `token_softmax_surface` implements the pure stable
softmax on its metadata-driven IO signature — for every disjoint flat
placement of the four buffers, every program `(cur_batch, cur_head)` whose
slot cells and active window lanes are in bounds, and every launch state
whose `B_Start_Loc`/`B_Seqlen` slot cells hold `m₁`/`m₂` and whose active
`Logics` window holds `xs`, the translated pointer kernel terminates, every
active (`j < m₂`) `Prob_Out` lane holds `tokenSoftmaxSpecPure m₂ xs j`, and
every other memory cell is unchanged. The loaded scalars `m₁`/`m₂` are honest
named binders of the spec, pinned to the slot cells by the skin's
`readMemValue .nat` preconditions. `hOutInj` is the trusted host-layout
side condition (output-offset injectivity at any head id and loaded start
index); `0 < BLOCK_SIZE` is required because the kernel's `max` reduce (like
`Finset.sup'`) is only defined on non-empty tiles. Proof:
`MetaMasked2DKernelIO₁.Implements.intro` assembles the region-model triple
with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification token_softmax_bloom_correctness
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE)
    (hOutInj : ∀ pid₁ m₁, Function.Injective
      (fun j : Fin BLOCK_SIZE =>
        pid₁ * stride_prob_h + (m₁ + j.val) * stride_prob_bs)) :
    tokenSoftmaxIO Logics B_Start_Loc B_Seqlen Prob_Out
        stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE
      ⊨ fun _ _ _ m₂ xs j => tokenSoftmaxSpecPure m₂ xs j
```

**Assumptions / layout contracts:**
- `hB : 0 < BLOCK_SIZE`
- `fun j : Fin BLOCK_SIZE =>
        pid₁ * stride_prob_h + (m₁ + j.val) * stride_prob_bs`

**Closed-form spec defs (transitive):** `tokenSoftmaxIO`, `tokenSoftmaxSpecPure`, `token_softmax_surface`

<details><summary><code>tokenSoftmaxIO</code></summary>

```
/-- `token_softmax_surface`'s **metadata-driven IO signature** — the whole
kernel-specific audit surface of the `⊨` headline (`MetaMasked2DKernelIO₁`,
the metadata genre: the data windows and the mask read two per-program `.nat`
scalar slots):

* `mbuf1 = B_Start_Loc`, `mbuf2 = B_Seqlen` — the scalar slots; program
  `(cur_batch, cur_head) = (pid₀, pid₁)` reads both at cell `pid₀`
  (`mwin1 = mwin2 = pid₀`), yielding the named scalars
  `m₁ = cur_batch_in_all_start_index` and `m₂ = cur_batch_seq_len`;
* `inp = Logics`, `out = Prob_Out` — the data channels, `B = BLOCK_SIZE`;
* `read`/`write` — lane `j` at
  `pid₁ * stride_logic_h + (m₁ + j) * stride_logic_bs` resp.
  `pid₁ * stride_prob_h + (m₁ + j) * stride_prob_bs`: the `m₁`-shifted token
  window of the batch's slice, head-indexed by `pid₁`;
* `mask` — the active lanes `j < m₂` (`col_offsets < cur_batch_seq_len`),
  shared by load and store (`writeMask` defaults to `mask`).

The slot cells, windows, and mask are declared, not parsed from the kernel;
the headline **proves** the kernel's actual slot loads, addressing, and
masking match them. Buffer sizes are not signature content: the headline
quantifies over every allocation whose extents cover the slot cells and the
active lanes. -/
```
```lean
def tokenSoftmaxIO (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs
      BLOCK_SIZE : Nat) : MetaMasked2DKernelIO₁ where
  kernel := token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
    stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE
  mbuf1 := B_Start_Loc
  mbuf2 := B_Seqlen
  inp := Logics
  out := Prob_Out
  B := BLOCK_SIZE
  mwin1 := fun pid₀ _ => pid₀
  mwin2 := fun pid₀ _ => pid₀
  read := fun _ pid₁ m₁ _ j =>
    pid₁ * stride_logic_h + (m₁ + j.val) * stride_logic_bs
  write := fun _ pid₁ m₁ _ j =>
    pid₁ * stride_prob_h + (m₁ + j.val) * stride_prob_bs
  mask := fun _ _ _ m₂ j => j.val < m₂
```
</details>

<details><summary><code>tokenSoftmaxSpecPure</code></summary>

```
/-- Exact stable-softmax value at lane `j`, as a **pure** function of the
loaded sequence length `m₂` and the input row `xs`: input tile lane `k` holds
`xs k` when `k < m₂` and `⊥` otherwise (matching `other=-float("inf")`), then
reduceMax-shift, `exp`, `/ reduceSum`. This is `tokenSoftmaxSpec` with the
state-coupled row replaced by the named binders of the `⊨` headline. -/
```
```lean
noncomputable def tokenSoftmaxSpecPure (m₂ : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) (j : Fin BLOCK_SIZE) : ℝ :=
  let row : Tile .real [BLOCK_SIZE] :=
    { data := fun idx => if idx.1.val < m₂ then some (xs idx.1) else none }
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false numerator
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR numerator denominator).data
          (j, PUnit.unit))
  | none => 0
```
</details>

<details><summary><code>token_softmax_surface</code></summary>

```
/-- Faithful transcription of `token_softmax_bloom.py`'s
`_fwd_kernel_token_softmax`.

The metadata buffers are typed Nat regions so their `tl.load` calls do not need
extra `dtype=` kwargs. -/
```
```lean
def token_softmax_surface
    (Logics : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  row = (tl.load(Logics + cur_head * $(stride_logic_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_logic_bs),
    mask=col_offsets < cur_batch_seq_len, other=-float("inf"))).to(tl.float32)
  row_minus_max = row - tl.max(row, axis=0)
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  tl.store(Prob_Out + cur_head * $(stride_prob_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_prob_bs),
    softmax_output, mask=col_offsets < cur_batch_seq_len)
}
```
</details>

## Also present (pinned special-case summaries)
- `token_softmax_final_store_slice_compute_correct`
