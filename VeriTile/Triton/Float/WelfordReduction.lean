/- Centered sums and variance recentering derived from admitted scalar atoms.
The count is the explicit reduction of ones. Binding it to the original
kernel's integer conversion remains a separate primitive-level obligation. -/
import VeriTile.Triton.Float.Welford

namespace VeriTile.Triton.FP.WelfordReduction
open Structural Guarded ScalarArithmetic ScalarReduction
open Equational (ReductionTree)

def count {α : Type} (M : Algebra α) (tree : ReductionTree N) : α :=
  value M (fun _ => one M) (zero M) tree

def mean {α : Type} (M : Algebra α) (xs : Fin N → α) (tree : ReductionTree N) : α :=
  div M (value M xs (zero M) tree) (count M tree)

/-- Only domain predicates occur here. The centered-sum identity is derived
below rather than included as an invariant or a numerical premise. -/
structure MeanDomain {α : Type} (M : Algebra α) (D : Domain α)
    (xs : Fin N → α) (tree : ReductionTree N) : Prop where
  inputs : ∀ i, D .finite (xs i)
  zero : D .finite (ScalarArithmetic.zero M)
  inputTree : FiniteTree M D xs (ScalarArithmetic.zero M) tree
  onesTree : FiniteTree M D (fun _ => one M) (ScalarArithmetic.zero M) tree
  countNonzero : D .nonzero (count M tree)
  inverseCount : D .finite (div M (one M) (count M tree))
  mean : D .finite (WelfordReduction.mean M xs tree)
  scaledZero : D .finite (mul M (WelfordReduction.mean M xs tree) (ScalarArithmetic.zero M))
  negativeScaledZero : D .finite
    (sub M (ScalarArithmetic.zero M) (mul M (WelfordReduction.mean M xs tree) (ScalarArithmetic.zero M)))
  deviations : FiniteTree M D (fun i => sub M (xs i) (WelfordReduction.mean M xs tree))
    (ScalarArithmetic.zero M) tree
  center : FiniteTree M D (fun _ => WelfordReduction.mean M xs tree) (ScalarArithmetic.zero M) tree
  negativeCenterSum : D .finite
    (sub M (ScalarArithmetic.zero M)
      (value M (fun _ => WelfordReduction.mean M xs tree) (ScalarArithmetic.zero M) tree))

/-- The scalar theory implies a zero sum of deviations when the mean uses
the same tree's floating count. No equality with fromNat(N) is assumed. -/
theorem centered_sum_zero {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (xs : Fin N → α) (tree : ReductionTree N) (hd : MeanDomain M D xs tree) :
    value M (fun i => sub M (xs i) (mean M xs tree)) (zero M) tree = zero M := by
  have hc : value M (fun _ => mean M xs tree) (zero M) tree = value M xs (zero M) tree := by
    rw [constant_value R M D hM s _ hd.mean hd.zero hd.scaledZero hd.negativeScaledZero tree hd.onesTree]
    exact div_mul_cancel R M D hM s _ _ (value_finite M D xs _ tree hd.inputTree)
      (value_finite M D _ _ tree hd.onesTree) hd.countNonzero hd.inverseCount
  apply add_right_cancel R M D hM s _ _ (value M (fun _ => mean M xs tree) (zero M) tree)
    (value_finite M D _ _ tree hd.deviations) hd.zero (value_finite M D _ _ tree hd.center)
    hd.zero hd.negativeCenterSum
  rw [zero_add R M D hM s _ (value_finite M D _ _ tree hd.center) hd.zero]
  exact (deviations_add_center R M D hM s xs _ tree hd.inputs hd.mean hd.zero
    hd.inputTree hd.deviations hd.center).trans hc.symm

/-- Domains for moving the center of a row whose old center is its derived
tree mean. All transformed partial sums remain visible in squareShift. -/
structure RecenterDomain {α : Type} (M : Algebra α) (D : Domain α)
    (xs : Fin N → α) (tree : ReductionTree N) (m' : α) : Prop where
  centering : MeanDomain M D xs tree
  squareShift : Welford.CenterShiftTreeDomain M D xs (mean M xs tree) m' tree
  shift : D .finite (sub M (mean M xs tree) m')
  scaledZero : D .finite (mul M (sub M (mean M xs tree) m') (zero M))
  negativeScaledZero : D .finite
    (sub M (zero M) (mul M (sub M (mean M xs tree) m') (zero M)))

/-- The cross term vanishes by factoring the shift out of a centered sum.
There is no whole-row cancellation assumption. -/
theorem cross_sum_zero {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (xs : Fin N → α) (tree : ReductionTree N) (m' : α)
    (hd : RecenterDomain M D xs tree m') :
    value M (Welford.crossTerms M xs (mean M xs tree) m') (zero M) tree = zero M := by
  unfold Welford.crossTerms
  rw [factor_right R M D hM s _ hd.shift hd.centering.zero hd.scaledZero hd.negativeScaledZero
    _ (fun i => (hd.squareShift.pointwise i).oldDeviation) tree hd.centering.deviations,
    centered_sum_zero R M D hM s xs tree hd.centering,
    mul_comm R M D hM s _ _ hd.centering.zero hd.shift]
  exact mul_zero R M D hM s _ hd.shift hd.centering.zero hd.scaledZero hd.negativeScaledZero

/-- Recenter a row about any new mean after deriving both vanishing cross
terms. The count-dependent shift contribution remains an explicit sum; it is
not replaced by multiplication by the converted row length. -/
theorem centered_square_shift {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (xs : Fin N → α) (tree : ReductionTree N) (m' : α)
    (hd : RecenterDomain M D xs tree m') :
    value M (Welford.deviationSquares M xs m') (zero M) tree =
      add M (value M (Welford.deviationSquares M xs (mean M xs tree)) (zero M) tree)
        (value M (fun _ => Welford.square M (sub M (mean M xs tree) m')) (zero M) tree) := by
  rw [Welford.square_shift_tree R M D hM s xs _ m' tree hd.squareShift,
    cross_sum_zero R M D hM s xs tree m' hd,
    add_zero R M D hM s _ (value_finite M D _ _ tree hd.squareShift.oldSquares),
    zero_add R M D hM s _ (value_finite M D _ _ tree hd.squareShift.shiftSquares) hd.centering.zero]

end VeriTile.Triton.FP.WelfordReduction
