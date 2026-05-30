import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `fused_activation` — strict per-kernel correctness

`fused_add_mul_activation_kernel` fuses a biased linear combine with an
activation: program `pid` loads block `[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of
`x_ptr` and `in_ptr`, plus a broadcast bias (`bias_ptr` indexed by
`index % num_weights`), forms `multiplier · in + x + bias`, applies either
`tl.sigmoid` or `tl.maximum(0, ·)` (ReLU) by the `activation` selector, and
stores **in place** to `x_ptr`, masked by `index < xnumel`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`fused_add_mul_activation_kernel[grid](...)`, the grid
size `cdiv(numel, BLOCK_SIZE)`, and how the runtime composes per-program writes
into one buffer) is the *trusted boundary*, not a proof obligation here. Because
`pid` is universally quantified, the per-program statement covers every program
of the grid. The Python `if activation == "sigmoid" / elif "relu"` string branch
is modeled by a Boolean `activation` (`true` = sigmoid, `false` = ReLU); both
branches are verified.

## Proof architecture

```
fused_add_mul_activation_kernel_output_summary       ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)                    surface lowers to the algorithm layer
  └─ fused_add_mul_activation_kernel_compute_correct ← ComputeCorrect over the masked store
       └─ fused_add_mul_activation_kernel_correct    ← algorithm-layer readback per lane
```

The spec `fusedActivationSpec` applies the selected activation to
`fusedActivationInput = multiplier · input + x + bias`. The loaded values are
presented by explicit memory-read hypotheses `h_x` / `h_in` / `h_bias` (the
last at the broadcast index `offset % num_weights`).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `tl.sigmoid` is modeled by
`Real.sigmoid` and the ReLU branch by the `tl.maximum` algorithm-layer form;
`@triton.autotune` is not modeled. The kernel writes in place to `x_ptr`; the
loaded tiles `xs`/`inputs`/`biases` are read into registers before the scatter,
so correctness holds for the in-place store. Aliasing of `bias_ptr`/`in_ptr`
with `x_ptr` is governed by the supplied read hypotheses.
-/

namespace VeriTile.Bench.TritonBenchG.FusedActivation

open VeriTile.Triton

/-- Faithful transcription of `fused_activation.py`'s
`fused_add_mul_activation_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `num_weights/xnumel/multiplier/activation/BLOCK_SIZE: tl.constexpr` ->
  Lean parameters.
- Python `activation == "sigmoid"` / `elif activation == "relu"` -> Lean
  Boolean `activation`; `true` selects sigmoid and `false` selects ReLU. -/
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

noncomputable def fusedActivationInput
    (x bias input multiplier : ℝ) : ℝ :=
  multiplier * input + x + bias

/-- Algorithm-layer branch form of the activation selector. `false` is the
`tl.maximum(0, x)` ReLU branch. -/
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

def fusedActivationOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

/-- Algorithm-layer correctness for `fused_add_mul_activation_kernel`. -/
theorem fused_add_mul_activation_kernel_correct
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool)
    (s s' : BlockState)
    (xs inputs : Fin BLOCK_SIZE → ℝ)
    (biases : Fin BLOCK_SIZE → ℝ)
    (h_x : ∀ i : Fin BLOCK_SIZE,
      s.readMem x_ptr (fusedActivationOffset s BLOCK_SIZE i) = xs i)
    (h_in : ∀ i : Fin BLOCK_SIZE,
      s.readMem in_ptr (fusedActivationOffset s BLOCK_SIZE i) = inputs i)
    (h_bias : ∀ i : Fin BLOCK_SIZE,
      s.readMem bias_ptr ((fusedActivationOffset s BLOCK_SIZE i) % num_weights) = biases i)
    (hExec : exec (fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
          num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := fusedActivationOffset s BLOCK_SIZE i
      s'.readMem x_ptr outAddr =
        if outAddr < xnumel then
          fusedActivationSpec ACTIVATION_SIGMOID (xs i) (biases i) (inputs i) multiplier
        else s.readMem x_ptr outAddr := by
  intro i
  cases ACTIVATION_SIGMOID
  · simp [exec, fused_add_mul_activation_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          tile_elementwise] at hExec
    subst s'
    simp only [fusedActivationOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : s.pid * BLOCK_SIZE + i.val < xnumel
    · have hx := h_x i
      have hin := h_in i
      have hb := h_bias i
      simp [fusedActivationOffset] at hx hin hb
      by_cases hlt : multiplier * inputs i + xs i + biases i < 0
      · simp [hi, fusedActivationSpec, fusedActivationInput, hx, hin, hb,
          ComparableDType.gt]
      · simp [hi, fusedActivationSpec, fusedActivationInput, hx, hin, hb,
          ComparableDType.gt]
    · simp [hi]
  · simp [exec, fused_add_mul_activation_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          tile_elementwise] at hExec
    subst s'
    simp only [fusedActivationOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : s.pid * BLOCK_SIZE + i.val < xnumel
    · have hx := h_x i
      have hin := h_in i
      have hb := h_bias i
      simp [fusedActivationOffset] at hx hin hb
      simp [hi, fusedActivationSpec, fusedActivationInput, hx, hin, hb]
    · simp [hi]

/-- Compute-facing correctness for `fused_add_mul_activation_kernel`. -/
theorem fused_add_mul_activation_kernel_compute_correct
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
    ComputeCorrect.Realizes
      (kernel := fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
        num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => fusedActivationOffset s BLOCK_SIZE i < xnumel)
          (fun i => (x_ptr, fusedActivationOffset s BLOCK_SIZE i)))
      (expected := fun i =>
        fusedActivationSpec ACTIVATION_SIGMOID (xs i) (biases i) (inputs i) multiplier) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_add_mul_activation_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fused_add_mul_activation_kernel_correct x_ptr bias_ptr in_ptr
    num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID
    s s' xs inputs biases h_x h_in h_bias hExec i
  simpa [hActive] using h

/-- Per-kernel output summary for `fused_add_mul_activation_kernel`: the DSL
surface lowers to the algorithm layer, and the in-place masked store to `x_ptr`
is compute-correct — every active lane holds `fusedActivationSpec` (the selected
activation of `multiplier · input + x + bias`), out-of-bounds lanes are
preserved. -/
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
        fusedActivationSpec ACTIVATION_SIGMOID (xs i) (biases i) (inputs i) multiplier) := by
  refine ⟨?_, ?_⟩
  · simp [fused_add_mul_activation_kernel, ComputeExpr.toAlgorithm?]
  · exact fused_add_mul_activation_kernel_compute_correct x_ptr bias_ptr in_ptr
      num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID
      s xs inputs biases h_x h_in h_bias

end VeriTile.Bench.TritonBenchG.FusedActivation
