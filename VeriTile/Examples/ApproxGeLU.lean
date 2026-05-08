/-
VeriTile.Examples.ApproxGeLU

Embedded Triton example for the tanh-style approximate GeLU commonly used
in transformer MLP blocks, together with the exact erf-based mathematical
GeLU specification it approximates:

    gelu_exact(x)  = 0.5 * x * (1 + erf(x / sqrt(2)))
    gelu_approx(x) = 0.5 * x *
      (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

The reference Triton snippet often computes `tanh` through sigmoid:

```python
@triton.jit
def gelu_approx_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x3 = x * x * x
    u = 0.7978845608028654 * (x + 0.044715 * x3)
    tanh_u = 2.0 * tl.sigmoid(2.0 * u) - 1.0
    y = 0.5 * x * (1.0 + tanh_u)

    tl.store(out_ptr + offsets, y, mask=mask)
```

This file embeds the aligned single-block, unmasked approximate Triton
implementation in VeriTile's DSL. The exact erf-based GeLU appears as the
Lean mathematical spec, not as an embedded kernel: the current DSL has
`tl.sigmoid` but not `tl.erf` or masks.

Constants are written as rational expressions because the DSL currently
supports natural numeric literals plus arithmetic, not decimal literals.
For example, `7978845608028654 / 10000000000000000` represents the usual
`sqrt(2 / pi)` decimal approximation used by the Triton reference.
-/

import VeriTile.Examples.ApproxGeLU.Certified

namespace VeriTile.Examples

open VeriTile.Triton VeriTile.Math

/-! ## Correctness and Approximation Statements -/

/-- **`approxGeLUKernel` correctness against the approximate GeLU expression.**

This is the operational statement: the embedded Triton kernel computes exactly
the tanh/sigmoid approximation represented by `approxGeLUSpec`. -/
theorem approx_gelu_kernel_correct
    (xReg outReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (xs : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs) :
    ∀ i : Fin blockSize,
      observeAt (exec (approxGeLUKernel xReg outReg blockSize) s)
          outReg blockSize s.pid i
        = some (approxGeLUSpec xs i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) :=
    injective_offset_singleton (s.pid * blockSize)
  simp [observeAt, exec, approxGeLUKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, approxGeLUSpec, approxGeLUScalar]
  unfold InputLoadedAt at _h_x
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x]

/-- View-level surface for `approx_gelu_kernel_correct`. -/
theorem approx_gelu_kernel_correct_exec_view
    (xReg outReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
    (xs : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1)) :
    ∀ idx : TileIndex [blockSize],
      TensorView.observe (exec (approxGeLUKernel xReg outReg blockSize) s)
          (programTileView s outReg blockSize) idx
        = some (approxGeLUSpec xs idx.1) := by
  intro idx
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := blockSize) (xs := xs) h_x
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt]
    using approx_gelu_kernel_correct xReg outReg blockSize hN s xs hx idx.1

/-- Compute-facing view-level surface for `approx_gelu_kernel_correct`. -/
theorem approx_gelu_kernel_correct_view
    (xReg outReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
    (xs : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1)) :
    ComputeCorrect.Realizes
      (kernel := approxGeLUKernel xReg outReg blockSize)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.ofTensorView
        (programTileView s outReg blockSize))
      (expected := fun idx : TileIndex [blockSize] => approxGeLUSpec xs idx.1) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have hview := approx_gelu_kernel_correct_exec_view xReg outReg blockSize hN s xs h_x idx
  rw [hExec] at hview
  simpa [ComputeCorrect.WriteMap.ofTensorView, TensorView.observe,
    observeTileAt] using hview

/-- Target error tolerance for the standard tanh/sigmoid GeLU approximation. -/
noncomputable def approxGeLUEps : ℝ := 1 / 1000

/-- Quantitative gap for the global `approxGeLUEps = 1e-3` bound, restricted
to the medium range `1/1000 < |x| < 19/5`. The proof dispatches on whether
`|x| ≤ 83/100`:

* **`|x| ≤ 83/100`** — closed by `approx_gelu_error_bound_083`, which applies
  the 9th-order tanh Taylor decomposition `gelu_gap_taylor9_decomposition`
  with explicit numerical bounds on each piece.
* **Mid `83/100 < |x| < 19/5`** — closed by the standalone degree-20 Taylor
  certificate from `VeriTile.Math.GeluTaylor20Cert` at midpoint `463/200`. -/
theorem approx_gelu_error_bound_medium {x : ℝ}
    (_hxlow : approxGeLUEps < |x|) (hxhigh : |x| < 19/5) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ approxGeLUEps := by
  exact approx_gelu_error_bound_medium_taylor20 _hxlow hxhigh

/-- The `|x| > 1/1000` half of `approx_gelu_error_bound`. Dispatches via
`approx_gelu_error_bound_tail_19_5` for `|x| ≥ 19/5` and
`approx_gelu_error_bound_medium` for `1/1000 < |x| < 19/5`. -/
theorem approx_gelu_error_bound_large {x : ℝ} (hx : approxGeLUEps < |x|) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ approxGeLUEps := by
  rcases lt_or_ge (|x|) (19/5) with hmedium | hlarge
  · exact approx_gelu_error_bound_medium hx hmedium
  · -- `|x| ≥ 19/5`: discharge from the sharpened asymptotic tail bound.
    have h := approx_gelu_error_bound_tail_19_5 hlarge
    show |approxGeLUScalar x - exactGeLUScalar x| ≤ 1 / 1000
    exact h

/-- The analytic/numerical approximation claim we need:
    on this input tile, the tanh/sigmoid approximate GeLU differs from the
    exact erf-based GeLU by at most `1e-3`.

The proof case-splits on `|xs i|` against `approxGeLUEps = 1/1000`. The small
case `|xs i| ≤ 1/1000` is closed unconditionally by the trivial linear bound
`|approx − exact| ≤ |x|`. The large case is reduced to the still-pending
quantitative gap statement `approx_gelu_error_bound_large`. -/
theorem approx_gelu_error_bound {N : Nat} (xs : Fin N → ℝ) :
    ∀ i : Fin N, |approxGeLUSpec xs i - exactGeLUSpec xs i| ≤ approxGeLUEps := by
  intro i
  unfold approxGeLUSpec exactGeLUSpec
  rcases le_or_gt (|xs i|) approxGeLUEps with hsmall | hlarge
  · exact (abs_approxGeLUScalar_sub_exactGeLUScalar_le_abs (xs i)).trans hsmall
  · exact approx_gelu_error_bound_large hlarge

/-- Every value written by the Triton approximate GeLU kernel is within `1e-3`
    of the exact erf-based GeLU spec, assuming the scalar approximation theorem
    above. -/
theorem approx_gelu_kernel_approximates_exact
    (xReg outReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
    (xs : Fin blockSize → ℝ)
    (h_x : InputLoadedAt s xReg blockSize xs) :
    ∀ i : Fin blockSize,
      (observeAt (exec (approxGeLUKernel xReg outReg blockSize) s)
          outReg blockSize s.pid i).map (fun y => |y - exactGeLUSpec xs i|)
        = some (|approxGeLUSpec xs i - exactGeLUSpec xs i|)
      ∧ |approxGeLUSpec xs i - exactGeLUSpec xs i| ≤ approxGeLUEps := by
  intro i
  constructor
  · rw [approx_gelu_kernel_correct xReg outReg blockSize hN s xs h_x i]
    rfl
  · exact approx_gelu_error_bound xs i

/-- Sorry-free linear approximation: every value written by the Triton
    approximate GeLU kernel is within `|xs i|` of the exact erf-based GeLU
    spec. This is a strictly weaker bound than `approxGeLUEps = 1e-3` — the
    constant is `|xs i|` rather than `1/1000` — but it depends only on
    `|tanh| ≤ 1` and `|realErf| ≤ 1` and is therefore established without the
    quantitative `tanh`-vs-`realErf` analytic content still pending in
    `approx_gelu_error_bound`. -/
theorem approx_gelu_kernel_approximates_linearly
    (xReg outReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
    (xs : Fin blockSize → ℝ)
    (h_x : InputLoadedAt s xReg blockSize xs) :
    ∀ i : Fin blockSize,
      (observeAt (exec (approxGeLUKernel xReg outReg blockSize) s)
          outReg blockSize s.pid i).map (fun y => |y - exactGeLUSpec xs i|)
        = some (|approxGeLUSpec xs i - exactGeLUSpec xs i|)
      ∧ |approxGeLUSpec xs i - exactGeLUSpec xs i| ≤ |xs i| := by
  intro i
  refine ⟨?_, ?_⟩
  · rw [approx_gelu_kernel_correct xReg outReg blockSize hN s xs h_x i]; rfl
  · exact abs_approxGeLUScalar_sub_exactGeLUScalar_le_abs (xs i)


end VeriTile.Examples
