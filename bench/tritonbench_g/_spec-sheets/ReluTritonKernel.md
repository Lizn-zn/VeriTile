# Spec sheet — `bench/tritonbench_g/relu_triton_kernel/ReluTritonKernel.lean`

**Python source:** `bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py`

## Public theorem: `relu_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `relu_kernel`: the DSL surface lowers to the
algorithm layer, and the `pid == 0`-gated masked store to `out_ptr` is
compute-correct — every active lane (with `pid = 0` and in bounds) holds
`TiledActivation.relu (xs i)`, all other observed cells are preserved. -/
```
</details>

**Statement:**
```lean
specification relu_kernel_output_summary
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) (hBlockSize : 0 < block_size)
    (s : BlockState) (xs : Fin block_size → ℝ)
    (h_x : InputLoadedAt s x_ptr block_size xs) :
    (∃ alg, (relu_kernel x_ptr out_ptr N block_size).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := relu_kernel x_ptr out_ptr N block_size)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin block_size => s.pid = 0 ∧ s.pid * block_size + i.val < N)
          (fun i => (out_ptr, s.pid * block_size + i.val)))
      (expected := fun i => TiledActivation.relu (xs i))
```

**Assumptions / layout contracts:**
- `hBlockSize : 0 < block_size`
- `xs : Fin block_size → ℝ`
- `h_x : InputLoadedAt s x_ptr block_size xs`
- `fun i : Fin block_size => s.pid = 0 ∧ s.pid * block_size + i.val < N`

**Closed-form spec defs (transitive):** `relu_kernel`

<details><summary><code>relu_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `relu_triton_kernel.py`'s `relu_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `N: tl.constexpr` / `block_size: tl.constexpr` → Lean `Nat`.
- Python `if cond: body` → `if cond { body }`, the DSL-side gate (block syntax
  uses braces because Lean is whitespace-insensitive; the `if` keyword itself
  is shared). -/
```
```lean
def relu_kernel
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = pid * $(block_size)
  offsets = block_start + tl.arange(0, $(block_size))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  result = tl.where(x >= 0, x, 0.0)
  if pid == 0 {
    tl.store(out_ptr + offsets, result, mask=mask)
  }
}
```
</details>

## Also present (pinned special-case summaries)
- `relu_kernel_compute_correct`
