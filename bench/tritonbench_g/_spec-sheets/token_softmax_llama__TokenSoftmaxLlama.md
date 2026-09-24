# Spec sheet — `bench/tritonbench_g/token_softmax_llama/TokenSoftmaxLlama.lean`

**Python source:** `bench/tritonbench_g/token_softmax_llama/token_softmax_llama.py`

## Public theorem: `token_softmax_llama_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `_fwd_kernel_token_softmax` implements the pure
masked-row stable softmax on its metadata-driven IO signature — for every
disjoint flat placement of the four buffers, every program id whose slot
cells and active window lanes are in bounds, and every launch state whose
`B_Start_Loc`/`B_Seqlen` slots hold `m₁`/`m₂` and whose active `Logics`
window holds `xs`, the translated pointer kernel terminates, every active
lane `j < m₂` of `Prob_Out` holds `tokenSoftmaxRowSpec BLOCK_SIZE m₂ xs j`
(reduceMax-shift, `exp`, `/ reduceSum` over the masked row), and every other
memory cell is unchanged. The loaded scalars appear as honest named binders
of the spec, pinned to the slot cells by the skin's contract. `hOutInj` is
the trusted host-layout side condition (`Prob_Out` window injectivity, e.g.
`stride_prob_bs ≠ 0` row-major layouts); `0 < BLOCK_SIZE` is required by the
`max` reduce. Proof: `MetaMasked2DKernelIO₁.Implements.intro` assembles the
region-model metadata triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification token_softmax_llama_correctness
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs
      BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE)
    (hOutInj : ∀ pid₁ m₁ : Nat, Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        pid₁ * stride_prob_h + (m₁ + i.val) * stride_prob_bs)) :
    tokenSoftmaxLlamaIO Logics B_Start_Loc B_Seqlen Prob_Out
        stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE
      ⊨ fun _ _ _ m₂ xs j => tokenSoftmaxRowSpec BLOCK_SIZE m₂ xs j
```

**Assumptions / layout contracts:**
- `hB : 0 < BLOCK_SIZE`
- `fun i : Fin BLOCK_SIZE =>
        pid₁ * stride_prob_h + (m₁ + i.val) * stride_prob_bs`

**Closed-form spec defs (transitive):** `tokenSoftmaxLlamaIO`, `tokenSoftmaxRowSpec`, `token_softmax_surface`, `tokenSoftmaxRowTile`

<details><summary><code>tokenSoftmaxLlamaIO</code></summary>

```
/-- `_fwd_kernel_token_softmax`'s metadata-driven **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `mbuf1 = B_Start_Loc` / `mbuf2 = B_Seqlen` — the two per-program `.nat`
  scalar slots, both at cell `cur_batch = pid₀` (`mwin1 = mwin2 = pid₀`);
* `inp = Logics` / `out = Prob_Out` — which buffer is which argument;
* `B = BLOCK_SIZE` — the row window each program owns;
* `read`/`write` — lane `j` of program `(pid₀, pid₁)` reads
  `Logics[pid₁ * stride_logic_h + (m₁ + j) * stride_logic_bs]` and writes
  `Prob_Out[pid₁ * stride_prob_h + (m₁ + j) * stride_prob_bs]`, where
  `m₁` is the **loaded** `B_Start_Loc[pid₀]` (`cur_head = pid₁` picks the
  row, the start index shifts into the packed token dimension);
* `mask` — the active lanes `j < m₂`, where `m₂` is the **loaded**
  `B_Seqlen[pid₀]` (`col_offsets < cur_batch_seq_len`, both load and store).

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer
sizes are not signature content: the headline quantifies over every
allocation whose extents cover the slot cells and the active lanes. -/
```
```lean
def tokenSoftmaxLlamaIO
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs
      BLOCK_SIZE : Nat) :
    MetaMasked2DKernelIO₁ where
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

<details><summary><code>tokenSoftmaxRowSpec</code></summary>

```
/-- **Pure** stable-softmax spec of the `⊨` headline: the exact softmax value
at lane `i` of the masked row — reduceMax-shift, `exp`, `/ reduceSum` — built
only from the loaded scalar `m₂` and the row `xs` (no launch state). -/
```
```lean
noncomputable def tokenSoftmaxRowSpec (BLOCK_SIZE m₂ : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  let row := tokenSoftmaxRowTile BLOCK_SIZE m₂ xs
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false numerator
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR numerator denominator).data
          (i, PUnit.unit))
  | none => 0
```
</details>

<details><summary><code>token_softmax_surface</code></summary>

```
/-- Faithful transcription of `token_softmax_llama.py`'s
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

<details><summary><code>tokenSoftmaxRowTile</code></summary>

```
/-- Masked input row as a **pure** function of the loaded sequence length
`m₂` and the row values `xs`: lane `j < m₂` holds `xs j`, inactive lanes are
`⊥`, matching the `other=-float("inf")` load. -/
```
```lean
noncomputable def tokenSoftmaxRowTile (BLOCK_SIZE m₂ : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx => if idx.1.val < m₂ then some (xs idx.1) else none }
```
</details>

## Also present (pinned special-case summaries)
- `token_softmax_final_store_slice_compute_correct`
