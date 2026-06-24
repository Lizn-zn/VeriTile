# Spec sheet — `bench/tritonbench_g/mul_exponent_compensator/MulExponentCompensator.lean`

**Python source:** `bench/tritonbench_g/mul_exponent_compensator/mul_exponent_compensator.py`

## Public theorem: `mul_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `mul_kernel`: the DSL surface lowers to the
algorithm layer, and the unmasked store to `dst` is compute-correct — every lane
holds `xs i * exponentCompensator`. -/
```
</details>

**Statement:**
```lean
theorem mul_kernel_output_summary
    (src dst : RegionName)
    (BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s src BLOCK_SIZE xs) :
    (∃ alg, (mul_kernel src dst BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := mul_kernel src dst BLOCK_SIZE)
      (initialState := s)
      (write := fun i : Fin BLOCK_SIZE =>
          some (dst, s.pid * BLOCK_SIZE + i.val))
      (expected := fun i => xs i * exponentCompensator)
```

**Assumptions / layout contracts:**
- `hBlockSize : 0 < BLOCK_SIZE`
- `xs : Fin BLOCK_SIZE → ℝ`
- `h_x : InputLoadedAt s src BLOCK_SIZE xs`
- `kernel : = mul_kernel src dst BLOCK_SIZE`
- `initialState : = s`
- `write : = fun i : Fin BLOCK_SIZE =>
          some (dst, s.pid * BLOCK_SIZE + i.val)`
- `expected : = fun i => xs i * exponentCompensator`

**Closed-form spec defs (transitive):** `mul_kernel`, `exponentCompensator`

<details><summary><code>mul_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `mul_exponent_compensator.py`'s
`mul_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python local constexpr literal `2.0 ** (127 - 15)` is represented by the
  Lean constant `exponentCompensator` and injected as a real scalar antiquote.
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
noncomputable def mul_kernel
    (src dst : RegionName)
    (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  exponent_compensator = $((exponentCompensator : ℝ))
  idxs = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(src + idxs)
  y = x * exponent_compensator
  tl.store(dst + idxs, y)
}
```
</details>

<details><summary><code>exponentCompensator</code></summary>

```
/-- The constexpr multiplier from `mul_exponent_compensator.py`. -/
```
```lean
noncomputable def exponentCompensator : ℝ :=
  (2 : ℝ) ^ (127 - 15)
```
</details>

## Also present (pinned special-case summaries)
- `mul_kernel_compute_correct`
