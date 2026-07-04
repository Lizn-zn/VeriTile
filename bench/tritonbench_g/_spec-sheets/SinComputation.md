# Spec sheet — `bench/tritonbench_g/sin_computation/SinComputation.lean`

**Python source:** `bench/tritonbench_g/sin_computation/sin_computation.py`

## Public theorem: `sin_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `sin_kernel`: the DSL surface lowers to the
algorithm layer, and the masked store to `out_ptr` is compute-correct — every
active lane holds `Real.sin (xs i)`, out-of-bounds lanes are preserved. -/
```
</details>

**Statement:**
```lean
theorem sin_kernel_output_summary
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs) :
    (∃ alg, (sin_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := sin_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
          (fun i => (out_ptr, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => Real.sin (xs i))
```

**Assumptions / layout contracts:**
- `hBlockSize : 0 < BLOCK_SIZE`
- `xs : Fin BLOCK_SIZE → ℝ`
- `h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs`
- `fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements`

**Closed-form spec defs (transitive):** `sin_kernel`

<details><summary><code>sin_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `sin_computation.py`'s `sin_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def sin_kernel
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  output = tl.sin(x)
  tl.store(out_ptr + offsets, output, mask=mask)
}
```
</details>

## Also present (pinned special-case summaries)
- `sin_kernel_compute_correct`
