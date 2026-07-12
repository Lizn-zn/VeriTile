# Spec sheet — `bench/tritonbench_g/softmax_triton3/SoftmaxTriton3.lean`

**Python source:** `bench/tritonbench_g/softmax_triton3/softmax_triton3.py`

## Public theorem: `softmax_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `softmax_kernel`: the DSL surface lowers to the
algorithm layer, and the masked store to `output_ptr` is compute-correct — every
in-bounds lane holds `softmaxSpec` (including the optional additive-mask branch),
out-of-bounds lanes are preserved. -/
```
</details>

**Statement:**
```lean
specification softmax_kernel_output_summary
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool)
    (s : BlockState) :
    (∃ alg, (softmax_kernel output_ptr input_ptr mask_ptr row_stride
        n_cols BLOCK_SIZE HAS_MASK).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := softmax_kernel output_ptr input_ptr mask_ptr row_stride
        n_cols BLOCK_SIZE HAS_MASK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => i.val < n_cols)
          (fun i => (output_ptr, s.pid * row_stride + i.val)))
      (expected := fun i =>
        softmaxSpec s input_ptr mask_ptr row_stride n_cols BLOCK_SIZE HAS_MASK i)
```

**Assumptions / layout contracts:**
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`

**Closed-form spec defs (transitive):** `softmax_kernel`, `softmaxSpec`, `softmaxInputTile`, `softmaxMaskTile`

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

<details><summary><code>softmaxSpec</code></summary>

```
/-- Exact stable-softmax value computed by `softmax_kernel` at lane `idx`,
including the optional additive mask path from `softmax_triton3.py`. -/
```
```lean
noncomputable def softmaxSpec
    (s : BlockState) (input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) (idx : Fin BLOCK_SIZE) : ℝ :=
  let row := softmaxInputTile s input_ptr row_stride n_cols BLOCK_SIZE
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let shifted :=
        if HAS_MASK then
          Tile.bop (NumericDType.add .real) (Broadcast.consSame Broadcast.nil) shifted
            (softmaxMaskTile s mask_ptr row_stride n_cols BLOCK_SIZE)
        else shifted
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false numerator
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR numerator denominator).data
          (idx, PUnit.unit))
  | none => 0
```
</details>

<details><summary><code>softmaxInputTile</code></summary>

```
/-- Masked input row tile used by `softmax_kernel`. Masked lanes are `⊥`,
matching `other=-float("inf")`. -/
```
```lean
noncomputable def softmaxInputTile
    (s : BlockState) (input_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pid * row_stride + idx.1.val
      if idx.1.val < n_cols then some (s.readMem input_ptr off) else none }
```
</details>

<details><summary><code>softmaxMaskTile</code></summary>

```
/-- Optional additive mask tile. Inactive lanes are `0`, matching
`tl.load(..., other=0)`, but the final store is still masked by `n_cols`. -/
```
```lean
noncomputable def softmaxMaskTile
    (s : BlockState) (mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pid * row_stride + idx.1.val
      if idx.1.val < n_cols then some (s.readMem mask_ptr off) else some 0 }
```
</details>

## Also present (pinned special-case summaries)
- `softmax_kernel_compute_correct`
