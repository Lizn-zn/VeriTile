import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.FusedActivation

open VeriTile.Triton

/-- Faithful transcription of `fused_activation.py`'s
`fused_add_mul_activation_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `num_weights/xnumel/multiplier/activation/BLOCK_SIZE: tl.constexpr` ->
  Lean parameters.
- Python `activation == "sigmoid"` / `"relu"` -> one-hot Lean constexpr gates
  `ACTIVATION_SIGMOID` / `ACTIVATION_RELU`.
- Python positional `tl.load(..., mask)` -> explicit `mask=mask`.
- Python `eviction_policy='evict_last'` is omitted because it is a cache hint
  with no algorithm-layer effect. -/
def fused_add_mul_activation_kernel
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID ACTIVATION_RELU : Bool) :
    ComputeKernel := triton {
  xoffset = tl.program_id(0) * $(BLOCK_SIZE)
  index = xoffset + tl.arange(0, $(BLOCK_SIZE))
  mask = index < $(xnumel)
  bias_index = index % $(num_weights)
  tmp0 = tl.load(x_ptr + index, mask=mask)
  tmp1 = tl.load(bias_ptr + bias_index, mask=mask)
  tmp3 = tl.load(in_ptr + index, mask=mask)
  activ_input = $(multiplier) * tmp3 + tmp0 + tmp1
  if ACTIVATION_SIGMOID {
    ma_result = tl.sigmoid(activ_input)
    tl.store(x_ptr + index, ma_result, mask=mask)
  }
  if ACTIVATION_RELU {
    ma_result = tl.maximum(0, activ_input)
    tl.store(x_ptr + index, ma_result, mask=mask)
  }
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
    (multiplier : ℝ) (ACTIVATION_SIGMOID ACTIVATION_RELU : Bool)
    (s s' : BlockState)
    (xs inputs : Fin BLOCK_SIZE → ℝ)
    (biases : Fin BLOCK_SIZE → ℝ)
    (h_x : ∀ i : Fin BLOCK_SIZE,
      s.readMem x_ptr (fusedActivationOffset s BLOCK_SIZE i) = xs i)
    (h_in : ∀ i : Fin BLOCK_SIZE,
      s.readMem in_ptr (fusedActivationOffset s BLOCK_SIZE i) = inputs i)
    (h_bias : ∀ i : Fin BLOCK_SIZE,
      s.readMem bias_ptr ((fusedActivationOffset s BLOCK_SIZE i) % num_weights) = biases i)
    (hSigmoidBranch : ACTIVATION_SIGMOID = Bool.true ∧ ACTIVATION_RELU = Bool.false ∨
      ACTIVATION_SIGMOID = Bool.false ∧ ACTIVATION_RELU = Bool.true)
    (hExec : exec (fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
          num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID ACTIVATION_RELU) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := fusedActivationOffset s BLOCK_SIZE i
      s'.readMem x_ptr outAddr =
        if outAddr < xnumel then
          fusedActivationSpec ACTIVATION_SIGMOID (xs i) (biases i) (inputs i) multiplier
        else s.readMem x_ptr outAddr := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * BLOCK_SIZE + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  rcases hSigmoidBranch with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · simp [exec, fused_add_mul_activation_kernel, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.uop,
          NumericDType.add, NumericDType.mul, IntegralDType.mod,
          ComparableDType.lt, ComparableDType.gt] at hExec
    subst s'
    simp only [fusedActivationOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : s.pid * BLOCK_SIZE + i.val < xnumel
    · have hx := h_x i
      have hin := h_in i
      have hb := h_bias i
      simp [fusedActivationOffset] at hx hin hb
      simp [hi, fusedActivationSpec, fusedActivationInput, hx, hin, hb]
    · simp [hi]
  · simp [exec, fused_add_mul_activation_kernel, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.uop,
          NumericDType.add, NumericDType.mul, IntegralDType.mod,
          ComparableDType.lt, ComparableDType.gt] at hExec
    subst s'
    simp only [fusedActivationOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : s.pid * BLOCK_SIZE + i.val < xnumel
    · have hx := h_x i
      have hin := h_in i
      have hb := h_bias i
      simp [fusedActivationOffset] at hx hin hb
      simp [hi, fusedActivationSpec, fusedActivationInput, hx, hin, hb,
        ComparableDType.gt]
    · simp [hi]

/-- Compute-facing correctness for `fused_add_mul_activation_kernel`. -/
theorem fused_add_mul_activation_kernel_compute_correct
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID ACTIVATION_RELU : Bool)
    (s : BlockState)
    (xs inputs : Fin BLOCK_SIZE → ℝ)
    (biases : Fin BLOCK_SIZE → ℝ)
    (h_x : ∀ i : Fin BLOCK_SIZE,
      s.readMem x_ptr (fusedActivationOffset s BLOCK_SIZE i) = xs i)
    (h_in : ∀ i : Fin BLOCK_SIZE,
      s.readMem in_ptr (fusedActivationOffset s BLOCK_SIZE i) = inputs i)
    (h_bias : ∀ i : Fin BLOCK_SIZE,
      s.readMem bias_ptr ((fusedActivationOffset s BLOCK_SIZE i) % num_weights) = biases i)
    (hSigmoidBranch : ACTIVATION_SIGMOID = Bool.true ∧ ACTIVATION_RELU = Bool.false ∨
      ACTIVATION_SIGMOID = Bool.false ∧ ACTIVATION_RELU = Bool.true) :
    ComputeCorrect.Realizes
      (kernel := fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
        num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID ACTIVATION_RELU)
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
    num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID ACTIVATION_RELU
    s s' xs inputs biases h_x h_in h_bias hSigmoidBranch hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.FusedActivation
