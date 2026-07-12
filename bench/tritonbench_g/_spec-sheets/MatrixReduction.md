# Spec sheet — `bench/tritonbench_g/matrix_reduction/MatrixReduction.lean`

**Python source:** `bench/tritonbench_g/matrix_reduction/matrix_reduction.py`

## Public theorem: `load_reduce_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `load_reduce_kernel`: the DSL surface lowers to
the algorithm layer, and the row-vector store to `y_ptr` is compute-correct —
every row lane `i` holds the row-wise maximum `matrixReduceSpec`. -/
```
</details>

**Statement:**
```lean
specification load_reduce_kernel_output_summary
    (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat)
    (s : BlockState) :
    (∃ alg, (load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y
        BLOCK_M BLOCK_N).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y
        BLOCK_M BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (y_ptr, i.val))
      (expected := fun i =>
        matrixReduceSpec s x_ptr stride_xm stride_xn BLOCK_M BLOCK_N i)
```

**Closed-form spec defs (transitive):** `load_reduce_kernel`, `matrixReduceSpec`, `matrixReduceInputTile`

<details><summary><code>load_reduce_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `matrix_reduction.py`'s `load_reduce_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_M: tl.constexpr` / `BLOCK_N: tl.constexpr` → Lean `Nat`
  parameters.
- `stride_y` is kept as `_stride_y`: the upstream Triton kernel accepts it but
  stores to `y_ptr + tl.arange(0, BLOCK_M)` and does not use it. -/
```
```lean
def load_reduce_kernel
    (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn _stride_y BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  x_ptr = tl.make_block_ptr(base=x_ptr,
    shape=($(BLOCK_M), $(BLOCK_N)),
    strides=($(stride_xm), $(stride_xn)),
    offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_N)),
    order=(1, 0))
  x = tl.load(x_ptr)
  y = tl.max(x, axis=1)
  tl.store(y_ptr + tl.arange(0, $(BLOCK_M)), y)
}
```
</details>

<details><summary><code>matrixReduceSpec</code></summary>

```
/-- Exact row-wise max written by `load_reduce_kernel` at row lane `i`. -/
```
```lean
noncomputable def matrixReduceSpec
    (s : BlockState) (x_ptr : RegionName)
    (stride_xm stride_xn BLOCK_M BLOCK_N : Nat) (i : Fin BLOCK_M) : ℝ :=
  match Tile.reduceMax (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.false
      (matrixReduceInputTile s x_ptr stride_xm stride_xn BLOCK_M BLOCK_N) with
  | some out => WithBot.unbotD 0 (out.data (i, PUnit.unit))
  | none => 0
```
</details>

<details><summary><code>matrixReduceInputTile</code></summary>

```
/-- Input tile read by the block pointer in `load_reduce_kernel`. -/
```
```lean
noncomputable def matrixReduceInputTile
    (s : BlockState) (x_ptr : RegionName)
    (stride_xm stride_xn BLOCK_M BLOCK_N : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      let bp : BlockPtr :=
        { region := x_ptr, baseOffset := 0, parentShape := [BLOCK_M, BLOCK_N],
          blockShape := [BLOCK_M, BLOCK_N], strides := [stride_xm, stride_xn],
          offsets := [0, 0] }
      some (s.readMem x_ptr (bp.address (TileShape.indexToList [BLOCK_M, BLOCK_N] idx))) }
```
</details>

## Also present (pinned special-case summaries)
- `load_reduce_kernel_compute_correct`
