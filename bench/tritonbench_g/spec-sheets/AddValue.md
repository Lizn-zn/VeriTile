# Spec sheet — `bench/tritonbench_g/add_value/AddValue.lean`

**Python source:** `bench/tritonbench_g/add_value/add_value.py`

## Public theorem: `puzzle1_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `puzzle1_kernel`: the DSL surface lowers to the
algorithm layer, and the masked store to `output_ptr` is compute-correct — every
active lane holds `xs i + value`, out-of-bounds lanes are preserved. -/
```
</details>

**Statement:**
```lean
theorem puzzle1_kernel_output_summary
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs) :
    (∃ alg, (puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < N)
          (fun i => (output_ptr, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => xs i + value)
```

**Assumptions / layout contracts:**
- `hBlockSize : 0 < BLOCK_SIZE`
- `xs : Fin BLOCK_SIZE → ℝ`
- `h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs`
- `fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < N`

**Closed-form spec defs (transitive):** `puzzle1_kernel`

<details><summary><code>puzzle1_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `add_value.py`'s `puzzle1_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- `value` (Lean `ℝ` parameter) injected via `$(...)`. -/
```
```lean
def puzzle1_kernel
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  output = x + $(value)
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

## Also present (pinned special-case summaries)
- `puzzle1_kernel_compute_correct`
