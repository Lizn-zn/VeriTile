# Spec sheet — `bench/tritonbench_g/triton_mul2/TritonMul2.lean`

**Python source:** `bench/tritonbench_g/triton_mul2/triton_mul2.py`

## Public theorem: `mul2_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `triton_mul2`'s two programs: both DSL surfaces
lower to the algorithm layer, and both masked stores are compute-correct — every
active lane holds `2 * xs i`, out-of-bounds lanes are preserved. The first
conjunct covers the out-of-place `mul2_kernel` store to `out_ptr`; the second
covers the in-place `mul2_inplace_kernel` store to `ptr`. -/
```
</details>

**Statement:**
```lean
specification mul2_kernel_output_summary
    (in_ptr0 out_ptr ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs xsInplace : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs)
    (h_xInplace : InputLoadedAt s ptr BLOCK_SIZE xsInplace) :
    ((∃ alg, (mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
      ComputeCorrect.Realizes_without_Rounding
        (kernel := mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
            (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
            (fun i => (out_ptr, s.pid * BLOCK_SIZE + i.val)))
        (expected := fun i => 2 * xs i)) ∧
    ((∃ alg, (mul2_inplace_kernel ptr n_elements BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
      ComputeCorrect.Realizes_without_Rounding
        (kernel := mul2_inplace_kernel ptr n_elements BLOCK_SIZE)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
            (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
            (fun i => (ptr, s.pid * BLOCK_SIZE + i.val)))
        (expected := fun i => 2 * xsInplace i))
```

**Assumptions / layout contracts:**
- `hBlockSize : 0 < BLOCK_SIZE`
- `xs xsInplace : Fin BLOCK_SIZE → ℝ`
- `h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs`
- `h_xInplace : InputLoadedAt s ptr BLOCK_SIZE xsInplace`
- `fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements`
- `fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements`

**Closed-form spec defs (transitive):** `mul2_kernel`, `mul2_inplace_kernel`

<details><summary><code>mul2_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `triton_mul2.py`'s `mul2_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def mul2_kernel
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  output = 2 * x
  tl.store(out_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>mul2_inplace_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `triton_mul2.py`'s `mul2_inplace_kernel`.

Same allowed mechanical Lean-syntax-only changes as `mul2_kernel`. -/
```
```lean
def mul2_inplace_kernel
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(ptr + offsets, mask=mask)
  output = 2 * x
  tl.store(ptr + offsets, output, mask=mask)
}
```
</details>

## Also present (pinned special-case summaries)
- `mul2_kernel_compute_correct`
- `mul2_inplace_kernel_compute_correct`
