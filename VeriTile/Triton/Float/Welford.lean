/- Local Welford identities derived from the admitted fp32 scalar atoms.
The floating count is explicit. Identifying it with an integer conversion is
a separate obligation, not an assumption hidden in this algebraic proof. -/
import VeriTile.Triton.Float.ScalarArithmetic
import VeriTile.Triton.Float.ScalarReduction

namespace VeriTile.Triton.FP.Welford
open Structural Guarded ScalarArithmetic

def nextCount {α : Type} (M : Algebra α) (n : α) : α := add M n (one M)
def difference {α : Type} (M : Algebra α) (x mean : α) : α := sub M x mean
def correction {α : Type} (M : Algebra α) (x mean n : α) : α :=
  div M (difference M x mean) (nextCount M n)
def nextMean {α : Type} (M : Algebra α) (x mean n : α) : α :=
  add M mean (correction M x mean n)
def residual {α : Type} (M : Algebra α) (x mean n : α) : α :=
  sub M x (nextMean M x mean n)
def square {α : Type} (M : Algebra α) (x : α) : α := mul M x x
def increment {α : Type} (M : Algebra α) (x mean n : α) : α :=
  mul M (difference M x mean) (residual M x mean n)

/-- Only finite/nonzero predicates occur here. In particular the updated mean
or its weighted sum is never supplied as an equality premise. -/
structure MeanStepDomain {α : Type} (M : Algebra α) (D : Domain α)
    (x m n : α) : Prop where
  input : D .finite x
  mean : D .finite m
  count : D .finite n
  one : D .finite (ScalarArithmetic.one M)
  difference : D .finite (Welford.difference M x m)
  nextCount : D .finite (Welford.nextCount M n)
  countNonzero : D .nonzero (Welford.nextCount M n)
  inverseCount : D .finite (div M (ScalarArithmetic.one M) (Welford.nextCount M n))
  correction : D .finite (Welford.correction M x m n)
  nextMean : D .finite (Welford.nextMean M x m n)
  weightedMean : D .finite (mul M m n)

/-- The mean invariant's arithmetic step, under the selected scalar theory:
(m + (x-m)/(n+1)) * (n+1) = m*n + x. There is no Welford admission atom. -/
theorem mean_step {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (x mean n : α) (hd : MeanStepDomain M D x mean n) :
    mul M (nextMean M x mean n) (nextCount M n) = add M (mul M mean n) x := by
  unfold nextMean
  rw [add_mul R M D hM s mean _ _ hd.mean hd.correction hd.nextCount hd.nextMean]
  have hdiv : mul M (correction M x mean n) (nextCount M n) = difference M x mean :=
    div_mul_cancel R M D hM s _ _ hd.difference hd.nextCount hd.countNonzero hd.inverseCount
  rw [hdiv]
  rw [nextCount, mul_distrib R M D hM s mean n (ScalarArithmetic.one M)
    hd.mean hd.count hd.one, mul_one R M D hM s mean hd.mean]
  rw [add_assoc R M D hM s _ mean _ hd.weightedMean hd.mean hd.difference,
    add_comm R M D hM s mean _ hd.mean hd.difference]
  rw [difference, sub_add_cancel R M D hM s x mean hd.input hd.mean]

/-- Additional domains used when cancelling the updated mean and expanding
the variance increment. These remain predicates on values, not equations. -/
structure VarianceStepDomain {α : Type} (M : Algebra α) (D : Domain α)
    (x m n : α) : Prop extends MeanStepDomain M D x m n where
  residual : D .finite (Welford.residual M x m n)
  scaledCorrection : D .finite (mul M n (Welford.correction M x m n))
  zero : D .finite (ScalarArithmetic.zero M)
  negativeNextMean : D .finite (sub M (ScalarArithmetic.zero M) (Welford.nextMean M x m n))

section Variance
variable {α : Type} [Inhabited α] (R : Rules) (M : Algebra α) (D : Domain α)
  (hM : Models R.assumptions M D) (s : State α)
  (x m n : α) (hd : VarianceStepDomain M D x m n)
include R hM s hd

/-- The correction recovers the old deviation after multiplication by the
updated count, then distribution splits the old-count part from one copy. -/
theorem split_correction :
    difference M x m = add M (mul M n (correction M x m n)) (correction M x m n) := by
  have recover : mul M (nextCount M n) (correction M x m n) = difference M x m := by
    rw [mul_comm R M D hM s _ _ hd.nextCount hd.correction]
    exact div_mul_cancel R M D hM s _ _ hd.difference hd.nextCount hd.countNonzero hd.inverseCount
  rw [nextCount, add_mul R M D hM s n (one M) _ hd.count hd.one hd.correction hd.nextCount,
    mul_comm R M D hM s (one M) _ hd.one hd.correction,
    mul_one R M D hM s _ hd.correction] at recover
  exact recover.symm

/-- The new deviation is n times the correction. This is the scalar identity
used by Welford's variance invariant, derived without an integer-count law. -/
theorem residual_step : residual M x m n = mul M n (correction M x m n) := by
  apply add_right_cancel R M D hM s _ _ (nextMean M x m n)
    hd.residual hd.scaledCorrection hd.nextMean hd.zero hd.negativeNextMean
  have recover : add M (mul M n (correction M x m n)) (nextMean M x m n) = x := by
    unfold nextMean
    rw [add_comm R M D hM s m _ hd.mean hd.correction,
      ← add_assoc R M D hM s _ _ m hd.scaledCorrection hd.correction hd.mean,
      ← split_correction R M D hM s x m n hd]
    exact sub_add_cancel R M D hM s x m hd.input hd.mean
  exact (sub_add_cancel R M D hM s x _ hd.input hd.nextMean).trans recover.symm

/-- Separate the newly appended sample's square from the shift contribution
of the previous n samples. This identity is derived, never admitted whole. -/
theorem variance_step :
    increment M x m n = add M (square M (residual M x m n))
      (mul M n (square M (correction M x m n))) := by
  have split : difference M x m = add M (residual M x m n) (correction M x m n) := by
    rw [residual_step R M D hM s x m n hd]
    exact split_correction R M D hM s x m n hd
  have hsum : D .finite (add M (residual M x m n) (correction M x m n)) :=
    split ▸ hd.difference
  unfold increment square
  rw [split, add_mul R M D hM s _ _ _ hd.residual hd.correction hd.residual hsum]
  congr 1
  rw [residual_step R M D hM s x m n hd,
    ← mul_assoc R M D hM s _ n _ hd.correction hd.count hd.correction,
    mul_comm R M D hM s _ n hd.correction hd.count,
    mul_assoc R M D hM s n _ _ hd.count hd.correction hd.correction]

end Variance

/-- Domains for recentering an arbitrary sample from m to m'. -/
structure CenterShiftDomain {α : Type} (M : Algebra α) (D : Domain α)
    (x m m' : α) : Prop where
  input : D .finite x
  oldMean : D .finite m
  newMean : D .finite m'
  oldDeviation : D .finite (sub M x m)
  meanShift : D .finite (sub M m m')
  joinedDeviation : D .finite (add M (sub M x m) (sub M m m'))
  newDeviation : D .finite (sub M x m')
  zero : D .finite (ScalarArithmetic.zero M)
  negativeNewMean : D .finite (sub M (ScalarArithmetic.zero M) m')

/-- Per-sample square recentering. The two cross terms stay explicit, avoiding
an unproved identification of literal 2 with floating addition of two ones. -/
theorem square_shift {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (x m m' : α) (hd : CenterShiftDomain M D x m m') :
    square M (sub M x m') =
      add M (add M (square M (sub M x m)) (mul M (sub M x m) (sub M m m')))
        (add M (mul M (sub M x m) (sub M m m')) (square M (sub M m m'))) := by
  have join := sub_add_sub_cancel R M D hM s x m m'
    hd.input hd.oldMean hd.newMean hd.oldDeviation hd.meanShift hd.joinedDeviation
    hd.newDeviation hd.zero hd.negativeNewMean
  unfold square
  rw [← join, add_mul R M D hM s _ _ _ hd.oldDeviation hd.meanShift hd.joinedDeviation hd.joinedDeviation,
    mul_distrib R M D hM s _ _ _ hd.oldDeviation hd.oldDeviation hd.meanShift,
    mul_distrib R M D hM s _ _ _ hd.meanShift hd.oldDeviation hd.meanShift,
    mul_comm R M D hM s _ _ hd.meanShift hd.oldDeviation]

def deviationSquares {α : Type} (M : Algebra α) (xs : Fin N → α) (m : α) : Fin N → α :=
  fun i => square M (sub M (xs i) m)

def crossTerms {α : Type} (M : Algebra α) (xs : Fin N → α) (m m' : α) : Fin N → α :=
  fun i => mul M (sub M (xs i) m) (sub M m m')

def oldPlusCross {α : Type} (M : Algebra α) (xs : Fin N → α) (m m' : α) : Fin N → α :=
  fun i => add M (deviationSquares M xs m i) (crossTerms M xs m m' i)

def crossPlusShift {α : Type} (M : Algebra α) (xs : Fin N → α) (m m' : α) : Fin N → α :=
  fun i => add M (crossTerms M xs m m' i) (square M (sub M m m'))

/-- Domains for all explicit addition trees used in square recentering.
The transformed partial sums are checked too; finite input leaves alone are
insufficient for the association rules used by reduction linearity. -/
structure CenterShiftTreeDomain {α : Type} (M : Algebra α) (D : Domain α)
    (xs : Fin N → α) (m m' : α) (tree : Equational.ReductionTree N) : Prop where
  pointwise : ∀ i, CenterShiftDomain M D (xs i) m m'
  zero : D .finite (ScalarArithmetic.zero M)
  oldSquares : ScalarReduction.FiniteTree M D (deviationSquares M xs m) (ScalarArithmetic.zero M) tree
  cross : ScalarReduction.FiniteTree M D (crossTerms M xs m m') (ScalarArithmetic.zero M) tree
  shiftSquares : ScalarReduction.FiniteTree M D (fun _ => square M (sub M m m'))
    (ScalarArithmetic.zero M) tree
  leftSum : ScalarReduction.FiniteTree M D (oldPlusCross M xs m m') (ScalarArithmetic.zero M) tree
  rightSum : ScalarReduction.FiniteTree M D (crossPlusShift M xs m m') (ScalarArithmetic.zero M) tree
  newSquares : ScalarReduction.FiniteTree M D (deviationSquares M xs m') (ScalarArithmetic.zero M) tree

/-- Lift per-sample square recentering through an arbitrary explicit addition
tree. Padding stays zero, the two cross terms remain separate, and the sum of
constant shift squares is not replaced by an unproved integer-count product. -/
theorem square_shift_tree {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (xs : Fin N → α) (m m' : α) (tree : Equational.ReductionTree N)
    (hd : CenterShiftTreeDomain M D xs m m' tree) :
    ScalarReduction.value M (deviationSquares M xs m') (zero M) tree =
      add M
        (add M (ScalarReduction.value M (deviationSquares M xs m) (zero M) tree)
          (ScalarReduction.value M (crossTerms M xs m m') (zero M) tree))
        (add M (ScalarReduction.value M (crossTerms M xs m m') (zero M) tree)
          (ScalarReduction.value M (fun _ => square M (sub M m m')) (zero M) tree)) := by
  have hp (i : Fin N) : deviationSquares M xs m' i =
      add M (oldPlusCross M xs m m' i) (crossPlusShift M xs m m' i) :=
    square_shift R M D hM s (xs i) m m' (hd.pointwise i)
  have hf := (ScalarReduction.finiteTree_congr M D _ _ (zero M) (zero M) hp rfl tree).mp hd.newSquares
  rw [ScalarReduction.value_congr M _ _ (zero M) (zero M) hp rfl tree,
    ScalarReduction.value_add R M D hM s _ _ tree hd.zero hd.leftSum hd.rightSum hf]
  unfold oldPlusCross crossPlusShift
  rw [ScalarReduction.value_add R M D hM s _ _ tree hd.zero hd.oldSquares hd.cross hd.leftSum,
    ScalarReduction.value_add R M D hM s _ _ tree hd.zero hd.cross hd.shiftSquares hd.rightSum]

end VeriTile.Triton.FP.Welford
