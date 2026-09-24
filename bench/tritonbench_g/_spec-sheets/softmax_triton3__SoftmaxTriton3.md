# Spec sheet — `bench/tritonbench_g/softmax_triton3/SoftmaxTriton3.lean`

**Python source:** `bench/tritonbench_g/softmax_triton3/softmax_triton3.py`

## Public theorem: `softmax_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **Headline, `HAS_MASK = false`** (the Python `mask_ptr is None` compile-time
branch): `softmax_kernel` implements the exact stable softmax over the active
row prefix on its masked IO signature — for every disjoint flat placement of
the two buffers, every program id whose active lanes are in bounds, and every
launch state whose active input-row lanes hold `xs`, the translated pointer
kernel terminates, every active output-row lane `j` holds
`softmaxSpec n_cols BLOCK_SIZE none xs j`, and every other memory cell is
unchanged. `0 < BLOCK_SIZE` is required: the kernel's `max` reduce (like
`Finset.sup'`) is only defined on non-empty tiles. Proof: `Implements.intro`
assembles the region-model masked triple with the bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification softmax_kernel_correctness
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    softmaxIO output_ptr input_ptr mask_ptr row_stride n_cols BLOCK_SIZE ⊨
      fun xs i => softmaxSpec n_cols BLOCK_SIZE none xs i
```

**Assumptions / layout contracts:**
- `hB : 0 < BLOCK_SIZE`

**Closed-form spec defs (transitive):** `softmaxIO`, `softmaxSpec`, `softmax_kernel`, `softmaxInputTile`, `softmaxMaskTile`

<details><summary><code>softmaxIO</code></summary>

```
/-- `softmax_kernel`'s masked **IO signature for the `HAS_MASK = false`
branch** — the whole kernel-specific audit surface of the unmasked `⊨`
headline:

* `inp`/`out` — which buffer is which argument (the wiring); `mask_ptr` is
  wired into the kernel but never touched on this branch, so it is not an
  interface buffer;
* `B = BLOCK_SIZE` — the row window each program owns;
* `read`/`write` — program `pid` reads and writes its row at
  `pid * row_stride` (the host-side one-program-per-row launch convention;
  this kernel uses one `row_stride` for both buffers);
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
def softmaxIO (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := softmax_kernel output_ptr input_ptr mask_ptr row_stride n_cols
    BLOCK_SIZE Bool.false
  inp := input_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * row_stride
  write := fun pid => pid * row_stride
  mask := fun _ j => j.val < n_cols
```
</details>

<details><summary><code>softmaxSpec</code></summary>

```
/-- Exact stable-softmax value computed by `softmax_kernel` at lane `idx`, as a
pure function of the active row prefix `xs j`, `j < n_cols`, and the optional
additive-mask row `mask?` (`some ms` ↔ the `HAS_MASK` branch; masked input
lanes enter the reductions as `⊥`, masked mask lanes as `0`). -/
```
```lean
noncomputable def softmaxSpec (n_cols BLOCK_SIZE : Nat)
    (mask? : Option (Fin BLOCK_SIZE → ℝ)) (xs : Fin BLOCK_SIZE → ℝ)
    (idx : Fin BLOCK_SIZE) : ℝ :=
  let row := softmaxInputTile n_cols BLOCK_SIZE xs
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let shifted :=
        match mask? with
        | some ms =>
            Tile.bop (NumericDType.add .real) (Broadcast.consSame Broadcast.nil)
              shifted (softmaxMaskTile n_cols BLOCK_SIZE ms)
        | none => shifted
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
/-- Faithful transcription of `softmax_triton3.py`'s `softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter.
- Python `mask_ptr is not None` -> Lean `HAS_MASK : Bool` constexpr gate.
- Python `.to(tl.float32)` casts are represented explicitly in the Compute
  layer; the algorithm-layer theorem observes their Real projection. -/
```
```lean
def softmax_kernel
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = (tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))).to(tl.float32)
  row_minus_max = row - tl.max(row, axis=0)
  if HAS_MASK {
    mask_ptrs = (mask_ptr + (row_idx * $(row_stride))) + col_offsets
    mask = (tl.load(mask_ptrs, mask=col_offsets < $(n_cols), other=0)).to(tl.float32)
    row_minus_max = row_minus_max + mask
  }
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  output_row_start_ptr = output_ptr + row_idx * $(row_stride)
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

<details><summary><code>softmaxMaskTile</code></summary>

```
/-- Optional additive-mask row tile: lane `j < n_cols` holds `ms j`, inactive
lanes are `0`, matching `tl.load(..., other=0)` (the final store is still
masked by `n_cols`). -/
```
```lean
noncomputable def softmaxMaskTile (n_cols BLOCK_SIZE : Nat)
    (ms : Fin BLOCK_SIZE → ℝ) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx => if idx.1.val < n_cols then some (ms idx.1) else some 0 }
```
</details>

## Public theorem: `softmax_kernel_masked_correctness`

<details><summary>docstring</summary>

```
/-- **Headline, `HAS_MASK = true`** (the Python `mask_ptr is not None`
compile-time branch): `softmax_kernel` implements the exact stable softmax
with the additive mask on its masked IO signature — for every disjoint flat
placement of the three buffers, every program id whose active lanes are in
bounds, and every launch state whose active input-row lanes hold `xs` and
active mask-row lanes hold `ms`, the translated pointer kernel terminates,
every active output-row lane `j` holds
`softmaxSpec n_cols BLOCK_SIZE (some ms) xs j`, and every other memory cell is
unchanged. `0 < BLOCK_SIZE` is required: the kernel's `max` reduce (like
`Finset.sup'`) is only defined on non-empty tiles. Proof: `Implements.intro`
assembles the region-model masked triple with the bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification softmax_kernel_masked_correctness
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    softmaxMaskedIO output_ptr input_ptr mask_ptr row_stride n_cols BLOCK_SIZE ⊨
      fun xs ms i => softmaxSpec n_cols BLOCK_SIZE (some ms) xs i
```

**Assumptions / layout contracts:**
- `hB : 0 < BLOCK_SIZE`

**Closed-form spec defs (transitive):** `softmaxMaskedIO`, `softmaxSpec`, `softmax_kernel`, `softmaxInputTile`, `softmaxMaskTile`

<details><summary><code>softmaxMaskedIO</code></summary>

```
/-- `softmax_kernel`'s masked **IO signature for the `HAS_MASK = true`
branch** — as `softmaxIO`, but the additive-mask row is a genuine second input
buffer: `in1` is the input matrix, `in2` the mask matrix (both read at
`pid * row_stride`), `out` the output matrix. Active lanes are `j < n_cols`
for every program, as on the unmasked branch. -/
```
```lean
def softmaxMaskedIO (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := softmax_kernel output_ptr input_ptr mask_ptr row_stride n_cols
    BLOCK_SIZE Bool.true
  in1 := input_ptr
  in2 := mask_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read1 := fun pid => pid * row_stride
  read2 := fun pid => pid * row_stride
  write := fun pid => pid * row_stride
  mask := fun _ j => j.val < n_cols
```
</details>

<details><summary><code>softmaxSpec</code></summary>

```
/-- Exact stable-softmax value computed by `softmax_kernel` at lane `idx`, as a
pure function of the active row prefix `xs j`, `j < n_cols`, and the optional
additive-mask row `mask?` (`some ms` ↔ the `HAS_MASK` branch; masked input
lanes enter the reductions as `⊥`, masked mask lanes as `0`). -/
```
```lean
noncomputable def softmaxSpec (n_cols BLOCK_SIZE : Nat)
    (mask? : Option (Fin BLOCK_SIZE → ℝ)) (xs : Fin BLOCK_SIZE → ℝ)
    (idx : Fin BLOCK_SIZE) : ℝ :=
  let row := softmaxInputTile n_cols BLOCK_SIZE xs
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let shifted :=
        match mask? with
        | some ms =>
            Tile.bop (NumericDType.add .real) (Broadcast.consSame Broadcast.nil)
              shifted (softmaxMaskTile n_cols BLOCK_SIZE ms)
        | none => shifted
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
/-- Faithful transcription of `softmax_triton3.py`'s `softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter.
- Python `mask_ptr is not None` -> Lean `HAS_MASK : Bool` constexpr gate.
- Python `.to(tl.float32)` casts are represented explicitly in the Compute
  layer; the algorithm-layer theorem observes their Real projection. -/
```
```lean
def softmax_kernel
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = (tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))).to(tl.float32)
  row_minus_max = row - tl.max(row, axis=0)
  if HAS_MASK {
    mask_ptrs = (mask_ptr + (row_idx * $(row_stride))) + col_offsets
    mask = (tl.load(mask_ptrs, mask=col_offsets < $(n_cols), other=0)).to(tl.float32)
    row_minus_max = row_minus_max + mask
  }
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  output_row_start_ptr = output_ptr + row_idx * $(row_stride)
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

<details><summary><code>softmaxMaskTile</code></summary>

```
/-- Optional additive-mask row tile: lane `j < n_cols` holds `ms j`, inactive
lanes are `0`, matching `tl.load(..., other=0)` (the final store is still
masked by `n_cols`). -/
```
```lean
noncomputable def softmaxMaskTile (n_cols BLOCK_SIZE : Nat)
    (ms : Fin BLOCK_SIZE → ℝ) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx => if idx.1.val < n_cols then some (ms idx.1) else some 0 }
```
</details>
