# Spec sheet — `bench/tritonbench_g/fused_activation/FusedActivation.lean`

**Python source:** `bench/tritonbench_g/fused_activation/fused_activation.py`

## Public theorem: `fused_add_mul_activation_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `fused_add_mul_activation_kernel` implements the fused
biased combine + activation on its grouped IO signature — for every disjoint
flat placement of the three buffers, every program id whose masked lanes are in
bounds, and every launch state whose three read windows (including the
broadcast bias window at `index % num_weights`) hold `xs`, the translated
pointer kernel terminates, every active lane `j` (`pid₀·BLOCK_SIZE + j <
xnumel`) of the *same* buffer `x_ptr` ends up holding
`fusedActivationSpec` — the selected activation of
`multiplier·in + x + bias` on the originally-loaded values — and every other
memory cell, including the out-of-bounds lanes of the window, is unchanged.
Both activation branches (`true` = `tl.sigmoid`, `false` = `tl.maximum(0, ·)`)
are covered: `ACTIVATION_SIGMOID` is a free parameter. Proof:
`GroupedMasked2DKernelIO.Implements.intro` assembles the region-model grouped
triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification fused_add_mul_activation_kernel_correctness
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool) :
    fusedActivationIO x_ptr bias_ptr in_ptr num_weights xnumel BLOCK_SIZE
        multiplier ACTIVATION_SIGMOID
      ⊨ fun _ _ xs _ j =>
          fusedActivationSpec ACTIVATION_SIGMOID
            (xs (⟨0, by decide⟩ : Fin 3) j)
            (xs (⟨1, by decide⟩ : Fin 3) j)
            (xs (⟨2, by decide⟩ : Fin 3) j) multiplier
```

**Closed-form spec defs (transitive):** `fusedActivationIO`, `fusedActivationSpec`, `fused_add_mul_activation_kernel`, `fusedActivationInput`

<details><summary><code>fusedActivationIO</code></summary>

```
/-- `fused_add_mul_activation_kernel`'s grouped masked **IO signature** — the
whole kernel-specific audit surface of the `⊨` headline
(`GroupedMasked2DKernelIO`, the vector-channel genre; the general per-lane
windows are what the broadcast bias read needs):

* `bufs = [x_ptr, bias_ptr, in_ptr]` — every buffer once; the output channel
  names `x_ptr` again, i.e. the update is **in place**;
* `nIn = 3` — channel 0 = `x_ptr` at `pid₀·BLOCK_SIZE + j`, channel 1 =
  `bias_ptr` at `(pid₀·BLOCK_SIZE + j) % num_weights` (the broadcast bias),
  channel 2 = `in_ptr` at `pid₀·BLOCK_SIZE + j`;
* `nOut = 1` — `x_ptr` at `pid₀·BLOCK_SIZE + j`;
* every read/write gate is the kernel's single mask `pid₀·BLOCK_SIZE + j <
  xnumel`;
* `B = BLOCK_SIZE`; the kernel is a 1-D launch, so the family's second program
  id is ignored by every field.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the declared lanes. -/
```
```lean
def fusedActivationIO
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool) : GroupedMasked2DKernelIO where
  kernel := fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr num_weights
    xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID
  nIn := 3
  nOut := 1
  bufs := [x_ptr, bias_ptr, in_ptr]
  inp := fun i => match i with
    | ⟨0, _⟩ => x_ptr
    | ⟨1, _⟩ => bias_ptr
    | ⟨_ + 2, _⟩ => in_ptr
  out := fun _ => x_ptr
  B := BLOCK_SIZE
  read := fun i pid₀ _ j => match i with
    | ⟨0, _⟩ => pid₀ * BLOCK_SIZE + j.val
    | ⟨1, _⟩ => (pid₀ * BLOCK_SIZE + j.val) % num_weights
    | ⟨_ + 2, _⟩ => pid₀ * BLOCK_SIZE + j.val
  readMask := fun _ pid₀ _ j => pid₀ * BLOCK_SIZE + j.val < xnumel
  write := fun _ pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  writeMask := fun _ pid₀ _ j => pid₀ * BLOCK_SIZE + j.val < xnumel
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

<details><summary><code>fusedActivationInput</code></summary>

```lean
noncomputable def fusedActivationInput
    (x bias input multiplier : ℝ) : ℝ :=
  multiplier * input + x + bias
```
</details>
