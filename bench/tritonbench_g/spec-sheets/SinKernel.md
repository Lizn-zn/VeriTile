# Spec sheet — `bench/tritonbench_g/sin_kernel/SinKernel.lean`

**Python source:** `bench/tritonbench_g/sin_kernel/sin_kernel.py`

## Public theorem: `kernel_function_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `kernel_function`: the DSL surface lowers to
the algorithm layer, and the masked store to `output_ptr` is compute-correct —
every active lane holds `Real.sin (xs i)`, out-of-bounds lanes are preserved. -/
```
</details>

**Statement:**
```lean
theorem kernel_function_output_summary
    (x_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs) :
    (∃ alg, (kernel_function x_ptr output_ptr n_elements BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := kernel_function x_ptr output_ptr n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
          (fun i => (output_ptr, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => Real.sin (xs i))
```

**Assumptions / layout contracts:**
- `hBlockSize : 0 < BLOCK_SIZE`
- `xs : Fin BLOCK_SIZE → ℝ`
- `h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs`
- `kernel : = kernel_function x_ptr output_ptr n_elements BLOCK_SIZE`
- `initialState : = s`
- `fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements`
- `expected : = fun i => Real.sin (xs i)`

**Closed-form spec defs (transitive):** `kernel_function`

<details><summary><code>kernel_function</code></summary>

```
/-- Faithful 1:1 transcription of `sin_kernel.py`'s `kernel_function`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def kernel_function
    (x_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  output = tl.math.sin(x)
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

## Also present (pinned special-case summaries)
- `kernel_function_compute_correct`
