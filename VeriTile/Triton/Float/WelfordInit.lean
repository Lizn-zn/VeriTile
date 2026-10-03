/- Initialization of Welford's scalar recurrence and its singleton row tree.
The initial count is floating zero. This does not assert that the original
kernel's opaque fromNat(0) conversion has that value. -/
import VeriTile.Triton.Float.WelfordAppend

namespace VeriTile.Triton.FP.WelfordInit
open Structural Guarded ScalarArithmetic ScalarReduction WelfordReduction
open WelfordAppend (appendTree value_append)
open Equational (ReductionTree)

structure InitDomain {α : Type} (M : Algebra α) (D : Domain α) (x : α) : Prop where
  input : D .finite x
  zero : D .finite (ScalarArithmetic.zero M)
  one : D .finite (ScalarArithmetic.one M)
  difference : D .finite (sub M x (ScalarArithmetic.zero M))
  diagonal : D .finite (sub M x x)
  negativeInput : D .finite (sub M (ScalarArithmetic.zero M) x)
  scaledZero : D .finite (mul M x (ScalarArithmetic.zero M))
  negativeScaledZero : D .finite
    (sub M (ScalarArithmetic.zero M) (mul M x (ScalarArithmetic.zero M)))
  zeroSquare : D .finite (Welford.square M (ScalarArithmetic.zero M))
  negativeZeroSquare : D .finite
    (sub M (ScalarArithmetic.zero M) (Welford.square M (ScalarArithmetic.zero M)))

section Initialization
variable {α : Type} [Inhabited α] (R : Rules) (M : Algebra α) (D : Domain α)
  (hM : Models R.assumptions M D) (s : State α)
  (x : α) (hd : InitDomain M D x)
include R hM s hd

/-- The first update from literal floating zeros returns the input mean. -/
theorem mean_init : Welford.nextMean M x (zero M) (zero M) = x := by
  rw [Welford.nextMean, Welford.correction, Welford.difference, Welford.nextCount,
    zero_add R M D hM s _ hd.one hd.zero,
    sub_zero R M D hM s x hd.input hd.zero hd.difference,
    div_one R M D hM s x hd.input, zero_add R M D hM s x hd.input hd.zero]

/-- The first update's unnormalized variance is zero. No initialization
identity is admitted: self-subtraction and zero multiplication are derived. -/
theorem variance_init :
    add M (zero M) (Welford.increment M x (zero M) (zero M)) = zero M := by
  rw [Welford.increment, Welford.difference, Welford.residual,
    mean_init R M D hM s x hd,
    sub_zero R M D hM s x hd.input hd.zero hd.difference,
    sub_self R M D hM s x hd.input hd.zero hd.diagonal hd.negativeInput,
    mul_zero R M D hM s x hd.input hd.zero hd.scaledZero hd.negativeScaledZero,
    add_zero R M D hM s _ hd.zero]

end Initialization

section Singleton
variable {α : Type} [Inhabited α] (R : Rules) (M : Algebra α) (D : Domain α)
  (hM : Models R.assumptions M D) (s : State α)
include R hM s

/-- An arbitrary empty padding tree is retained when the first lane is added. -/
theorem singleton_value (x : α) (hx : D .finite x) (hz : D .finite (zero M))
    (empty : ReductionTree 0) :
    value M (fun _ : Fin 1 => x) (zero M) (appendTree empty) = x := by
  rw [value_append, value_empty R M D hM s _ empty hz]
  exact zero_add R M D hM s x hx hz

theorem singleton_count (hz : D .finite (zero M)) (ho : D .finite (one M))
    (empty : ReductionTree 0) : count M (appendTree empty) = one M :=
  singleton_value R M D hM s _ ho hz empty

theorem singleton_mean (x : α) (hd : InitDomain M D x) (empty : ReductionTree 0) :
    mean M (fun _ : Fin 1 => x) (appendTree empty) = x := by
  rw [mean, singleton_value R M D hM s x hd.input hd.zero empty,
    singleton_count R M D hM s hd.zero hd.one empty, div_one R M D hM s x hd.input]

theorem singleton_square_sum (x : α) (hd : InitDomain M D x) (empty : ReductionTree 0) :
    value M (Welford.deviationSquares M (fun _ : Fin 1 => x)
      (mean M (fun _ : Fin 1 => x) (appendTree empty))) (zero M) (appendTree empty) = zero M := by
  rw [singleton_mean R M D hM s x hd empty]
  refine (value_congr M _ (fun _ => zero M) _ _ ?_ rfl _).trans
    (value_zero R M D hM s _ hd.zero)
  intro i
  change mul M (sub M x x) (sub M x x) = zero M
  rw [sub_self R M D hM s x hd.input hd.zero hd.diagonal hd.negativeInput]
  exact mul_zero R M D hM s _ hd.zero hd.zero hd.zeroSquare hd.negativeZeroSquare

theorem singleton_variance (x : α) (hd : InitDomain M D x) (empty : ReductionTree 0) :
    div M (value M (Welford.deviationSquares M (fun _ : Fin 1 => x)
        (mean M (fun _ : Fin 1 => x) (appendTree empty))) (zero M) (appendTree empty))
      (count M (appendTree empty)) = zero M := by
  rw [singleton_square_sum R M D hM s x hd empty,
    singleton_count R M D hM s hd.zero hd.one empty, div_one R M D hM s _ hd.zero]

/-- The literal-zero initialization agrees with the singleton row statistics.
This is the induction base for the tree-count recurrence, not a proof of the
original integer conversion or of the original whole kernel. -/
theorem initial_statistics (x : α) (hd : InitDomain M D x) (empty : ReductionTree 0) :
    (Welford.nextMean M x (zero M) (zero M),
      add M (zero M) (Welford.increment M x (zero M) (zero M))) =
    (mean M (fun _ : Fin 1 => x) (appendTree empty),
      value M (Welford.deviationSquares M (fun _ : Fin 1 => x)
        (mean M (fun _ : Fin 1 => x) (appendTree empty))) (zero M) (appendTree empty)) :=
  Prod.ext ((mean_init R M D hM s x hd).trans (singleton_mean R M D hM s x hd empty).symm)
    ((variance_init R M D hM s x hd).trans (singleton_square_sum R M D hM s x hd empty).symm)

end Singleton
end VeriTile.Triton.FP.WelfordInit
