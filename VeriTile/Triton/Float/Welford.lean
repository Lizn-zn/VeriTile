/- Local Welford identities derived from the admitted fp32 scalar atoms.
The floating count is explicit. Identifying it with an integer conversion is
a separate obligation, not an assumption hidden in this algebraic proof. -/
import VeriTile.Triton.Float.ScalarArithmetic

namespace VeriTile.Triton.FP.Welford
open Structural Guarded ScalarArithmetic

def nextCount {α : Type} (M : Algebra α) (n : α) : α := add M n (one M)
def difference {α : Type} (M : Algebra α) (x mean : α) : α := sub M x mean
def correction {α : Type} (M : Algebra α) (x mean n : α) : α :=
  div M (difference M x mean) (nextCount M n)
def nextMean {α : Type} (M : Algebra α) (x mean n : α) : α :=
  add M mean (correction M x mean n)

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

end VeriTile.Triton.FP.Welford
