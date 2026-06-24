# Spec sheet — `bench/tritonbench_g/fused_activation/FusedActivation.lean`

**Python source:** `bench/tritonbench_g/fused_activation/fused_activation.py`

## Public theorem: `fused_add_mul_activation_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `fused_add_mul_activation_kernel`: the DSL
surface lowers to the algorithm layer, and the in-place masked store to `x_ptr`
is compute-correct — every active lane holds `fusedActivationSpec` (the selected
activation of `multiplier · input + x + bias`), out-of-bounds lanes are
preserved. -/
```
</details>

**Statement:**
```lean
theorem fused_add_mul_activation_kernel_output_summary
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool)
    (s : BlockState)
    (xs inputs : Fin BLOCK_SIZE → ℝ)
    (biases : Fin BLOCK_SIZE → ℝ)
    (h_x : ∀ i : Fin BLOCK_SIZE,
      s.readMem x_ptr (fusedActivationOffset s BLOCK_SIZE i) = xs i)
    (h_in : ∀ i : Fin BLOCK_SIZE,
      s.readMem in_ptr (fusedActivationOffset s BLOCK_SIZE i) = inputs i)
    (h_bias : ∀ i : Fin BLOCK_SIZE,
      s.readMem bias_ptr ((fusedActivationOffset s BLOCK_SIZE i) % num_weights) = biases i) :
    (∃ alg, (fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
        num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID).toAlgorithm? =
          Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
        num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => fusedActivationOffset s BLOCK_SIZE i < xnumel)
          (fun i => (x_ptr, fusedActivationOffset s BLOCK_SIZE i)))
      (expected := fun i =>
        fusedActivationSpec ACTIVATION_SIGMOID (xs i) (biases i) (inputs i) multiplier)
```

**Assumptions / layout contracts:**
- `xs inputs : Fin BLOCK_SIZE → ℝ`
- `biases : Fin BLOCK_SIZE → ℝ`
- `h_x : ∀ i : Fin BLOCK_SIZE,
      s.readMem x_ptr (fusedActivationOffset s BLOCK_SIZE i) = xs i`
- `h_in : ∀ i : Fin BLOCK_SIZE,
      s.readMem in_ptr (fusedActivationOffset s BLOCK_SIZE i) = inputs i`
- `kernel : = fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
        num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID`
- `initialState : = s`
- `fun i : Fin BLOCK_SIZE => fusedActivationOffset s BLOCK_SIZE i < xnumel`
- `expected : = fun i =>
        fusedActivationSpec ACTIVATION_SIGMOID (xs i) (biases i) (inputs i) multiplier`

**Closed-form spec defs (transitive):** `fusedActivationOffset`, `fused_add_mul_activation_kernel`, `fusedActivationSpec`, `fusedActivationInput`

<details><summary><code>fusedActivationOffset</code></summary>

```lean
def fusedActivationOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val
```
</details>

<details><summary><code>fused_add_mul_activation_kernel</code></summary>

```
/-- Faithful transcription of `fused_activation.py`'s
`fused_add_mul_activation_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `num_weights/xnumel/multiplier/activation/BLOCK_SIZE: tl.constexpr` ->
  Lean parameters.
- Python `activation == "sigmoid"` / `elif activation == "relu"` -> Lean
  Boolean `activation`; `true` selects sigmoid and `false` selects ReLU. -/
```
```lean
def fused_add_mul_activation_kernel
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (activation : Bool) :
    ComputeKernel := triton {
  xoffset = tl.program_id(0) * $(BLOCK_SIZE)
  index = xoffset + tl.arange(0, $(BLOCK_SIZE))[:]
  mask = index < $(xnumel)
  bias_index = index % $(num_weights)
  tmp0 = tl.load(x_ptr + index, mask)
  tmp1 = tl.load(bias_ptr + bias_index, mask, eviction_policy="evict_last")
  tmp3 = tl.load(in_ptr + index, mask)
  activ_input = $(multiplier) * tmp3 + tmp0 + tmp1
  if activation {
    ma_result = tl.sigmoid(activ_input)
  } else {
    ma_result = tl.maximum(0, activ_input)
  }
  tl.store(x_ptr + index, ma_result, mask)
}
```
</details>

<details><summary><code>fusedActivationSpec</code></summary>

```
/-- Algorithm-layer branch form of the activation selector. `false` is the
`tl.maximum(0, x)` ReLU branch. -/
```
```lean
noncomputable def fusedActivationSpec
    (ACTIVATION_SIGMOID : Bool) (x bias input multiplier : ℝ) : ℝ :=
  let z := fusedActivationInput x bias input multiplier
  if ACTIVATION_SIGMOID then
    Real.sigmoid z
  else
    WithBot.unbotD 0
      (if ComparableDType.real.gt (some 0) (some z) then
        (some 0 : WithBot ℝ)
      else
        (some z : WithBot ℝ))
```
</details>

<details><summary><code>fusedActivationInput</code></summary>

```lean
noncomputable def fusedActivationInput
    (x bias input multiplier : ℝ) : ℝ :=
  multiplier * input + x + bias
```
</details>

## Also present (pinned special-case summaries)
- `fused_add_mul_activation_kernel_compute_correct`
