# Spec sheet — `bench/tritonbench_g/softmax_triton1/SoftmaxTriton1.lean`

**Python source:** `bench/tritonbench_g/softmax_triton1/softmax_triton1.py`

## Public theorem: `softmax_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `softmax_kernel`: the DSL surface lowers to the
algorithm layer, and the masked store to `output_ptr` is compute-correct — every
in-bounds lane holds `softmaxSpec`, out-of-bounds lanes are preserved. -/
```
</details>

**Statement:**
```lean
theorem softmax_kernel_output_summary
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState) :
    (∃ alg, (softmax_kernel output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := softmax_kernel output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => i.val < n_cols)
          (fun i => (output_ptr, s.pid * output_row_stride + i.val)))
      (expected := fun i =>
        softmaxSpec s input_ptr input_row_stride n_cols BLOCK_SIZE i)
```

**Assumptions / layout contracts:**
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`

**Closed-form spec defs (transitive):** `softmax_kernel`, `softmaxSpec`, `softmaxInputTile`

<details><summary><code>softmax_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `softmax_triton1.py`'s `softmax_kernel`.

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

<details><summary><code>softmaxSpec</code></summary>

```
/-- Exact stable-softmax value computed by the kernel at lane `idx`. -/
```
```lean
noncomputable def softmaxSpec
    (s : BlockState) (input_ptr : RegionName)
    (input_row_stride n_cols BLOCK_SIZE : Nat) (idx : Fin BLOCK_SIZE) : ℝ :=
  let row := softmaxInputTile s input_ptr input_row_stride n_cols BLOCK_SIZE
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

<details><summary><code>softmaxInputTile</code></summary>

```
/-- Masked input row tile used by `softmax_kernel`. Masked lanes are `⊥`,
matching `other=-float("inf")`. -/
```
```lean
noncomputable def softmaxInputTile
    (s : BlockState) (input_ptr : RegionName)
    (input_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pid * input_row_stride + idx.1.val
      if idx.1.val < n_cols then some (s.readMem input_ptr off) else none }
```
</details>

## Also present (pinned special-case summaries)
- `softmax_kernel_compute_correct`
