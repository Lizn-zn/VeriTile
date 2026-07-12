# Spec sheet — `bench/tritonbench_g/add_example/AddExample.lean`

**Python source:** `bench/tritonbench_g/add_example/add_example.py`

## Public theorem: `add_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `add_kernel`: the DSL surface lowers to the
algorithm layer, and the masked store to `out_ptr` is compute-correct — every
active lane holds `xs i + ys i`, out-of-bounds lanes are preserved. -/
```
</details>

**Statement:**
```lean
specification add_kernel_output_summary
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs)
    (h_y : InputLoadedAt s in_ptr1 BLOCK_SIZE ys) :
    (∃ alg, (add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
          (fun i => (out_ptr, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => xs i + ys i)
```

**Assumptions / layout contracts:**
- `hBlockSize : 0 < BLOCK_SIZE`
- `xs ys : Fin BLOCK_SIZE → ℝ`
- `h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs`
- `h_y : InputLoadedAt s in_ptr1 BLOCK_SIZE ys`
- `fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements`

**Closed-form spec defs (transitive):** `add_kernel`

<details><summary><code>add_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `add_example.py`'s `add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` annotation → Lean `Nat` parameter
  (the `tl.constexpr` is implicit in Lean params).

Everything else is verbatim from the upstream kernel. -/
```
```lean
def add_kernel
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  y = tl.load(in_ptr1 + offsets, mask=mask)
  output = x + y
  tl.store(out_ptr + offsets, output, mask=mask)
}
```
</details>

## Also present (pinned special-case summaries)
- `add_kernel_compute_correct`
