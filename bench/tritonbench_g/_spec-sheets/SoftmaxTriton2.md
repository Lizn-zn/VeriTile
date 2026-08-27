# Spec sheet — `bench/tritonbench_g/softmax_triton2/SoftmaxTriton2.lean`

**Python source:** `bench/tritonbench_g/softmax_triton2/softmax_triton2.py`

## Public theorem: `softmax_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `softmax_kernel` implements the exact stable softmax
over the active row prefix on its masked IO signature — for every disjoint
flat placement of the two buffers, every program id whose active lanes are in
bounds, and every launch state whose active input-row lanes hold `xs`, the
translated pointer kernel terminates, every active output-row lane `j` holds
`softmaxSpec n_cols BLOCK_SIZE xs j`, and every other memory cell is
unchanged. `0 < BLOCK_SIZE` is required: the kernel's `max` reduce (like
`Finset.sup'`) is only defined on non-empty tiles. Proof: `Implements.intro`
assembles the region-model masked triple with the bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification softmax_kernel_correctness
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    softmaxIO output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE ⊨
      fun xs i => softmaxSpec n_cols BLOCK_SIZE xs i
```

**Assumptions / layout contracts:**
- `hB : 0 < BLOCK_SIZE`

**Closed-form spec defs (transitive):** `softmaxIO`, `softmaxSpec`, `softmax_kernel`, `softmaxInputTile`

<details><summary><code>softmaxIO</code></summary>

```
/-- `softmax_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_SIZE` — the row window each program owns;
* `read`/`write` — program `pid` reads its row at `pid * input_row_stride` and
  writes it at `pid * output_row_stride` (the host-side one-program-per-row
  launch convention);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding
  of `BLOCK_SIZE = next_power_of_2(n_cols)`) carry no obligations on either
  side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def softmaxIO (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    MaskedKernelIO₁ where
  kernel := softmax_kernel output_ptr input_ptr input_row_stride
    output_row_stride n_cols BLOCK_SIZE
  inp := input_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * input_row_stride
  write := fun pid => pid * output_row_stride
  mask := fun _ j => j.val < n_cols
```
</details>

<details><summary><code>softmaxSpec</code></summary>

```
/-- Exact stable-softmax value computed by the kernel at lane `idx`, as a pure
function of the active row prefix `xs j`, `j < n_cols` (masked lanes enter the
reductions as `⊥`, neutral for both `max` and the `exp`-sum). -/
```
```lean
noncomputable def softmaxSpec (n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) (idx : Fin BLOCK_SIZE) : ℝ :=
  let row := softmaxInputTile n_cols BLOCK_SIZE xs
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false numerator
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR numerator denominator).data
          (idx, PUnit.unit))
  | none => 0
```
</details>

<details><summary><code>softmax_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `softmax_triton2.py`'s `softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def softmax_kernel
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(input_row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
  row_minus_max = row - tl.max(row, axis=0)
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  output_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
}
```
</details>

<details><summary><code>softmaxInputTile</code></summary>

```
/-- Masked input row tile used by `softmax_kernel`: lane `j < n_cols` holds
`xs j`, masked lanes are `⊥`, matching `other=-float("inf")`. -/
```
```lean
noncomputable def softmaxInputTile (n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx => if idx.1.val < n_cols then some (xs idx.1) else none }
```
</details>
