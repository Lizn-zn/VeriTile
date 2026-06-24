# Spec sheet — `bench/tritonbench_g/dropout_triton/DropoutTriton.lean`

**Python source:** `bench/tritonbench_g/dropout_triton/dropout_triton.py`

## Public theorem: `dropout_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_dropout`: the DSL surface lowers to the
algorithm layer, and the masked store to `output_ptr` is compute-correct — every
active lane holds `dropoutSpec` (the keep-gated scaled input), out-of-bounds
lanes are preserved. -/
```
</details>

**Statement:**
```lean
theorem dropout_kernel_output_summary
    (x_ptr x_keep_ptr output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat)
    (s : BlockState) :
    (∃ alg, (dropout_kernel x_ptr x_keep_ptr output_ptr
        n_elements p BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := dropout_kernel x_ptr x_keep_ptr output_ptr
        n_elements p BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => dropoutOffset s BLOCK_SIZE i < n_elements)
        (fun i => (output_ptr, dropoutOffset s BLOCK_SIZE i)))
      (expected := fun i => dropoutSpec s x_ptr x_keep_ptr p BLOCK_SIZE i)
```

**Assumptions / layout contracts:**
- `kernel : = dropout_kernel x_ptr x_keep_ptr output_ptr
        n_elements p BLOCK_SIZE`
- `initialState : = s`
- `fun i : Fin BLOCK_SIZE => dropoutOffset s BLOCK_SIZE i < n_elements`
- `expected : = fun i => dropoutSpec s x_ptr x_keep_ptr p BLOCK_SIZE i`

**Closed-form spec defs (transitive):** `dropout_kernel`, `dropoutOffset`, `dropoutSpec`

<details><summary><code>dropout_kernel</code></summary>

```
/-- Faithful transcription of `dropout_triton.py`'s `_dropout`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- `x_keep_ptr` is a typed Lean boolean region so its `tl.load` call does not
  need an extra `dtype=` kwarg. -/
```
```lean
def dropout_kernel
    (x_ptr : RegionName) (x_keep_ptr : Region .bool) (output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
  output = tl.where(x_keep, x / (1 - $(p)), 0.0)
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>dropoutOffset</code></summary>

```lean
def dropoutOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val
```
</details>

<details><summary><code>dropoutSpec</code></summary>

```lean
noncomputable def dropoutSpec
    (s : BlockState) (x_ptr x_keep_ptr : RegionName)
    (p : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  if s.readMemValue .bool x_keep_ptr (dropoutOffset s BLOCK_SIZE i) then
    s.readMem x_ptr (dropoutOffset s BLOCK_SIZE i) / (1 - p)
  else
    0.0
```
</details>

## Also present (pinned special-case summaries)
- `dropout_kernel_compute_correct`
