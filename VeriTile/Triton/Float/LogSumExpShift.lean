/- Scalar-derived logsumexp shift using libdevice.exp. EXP-SUB is supplied
by Float/Exponential; tl.log product and tl.log(libdevice.exp(a)) cancellation
still require matching admission. The old tl.log(tl.exp(a)) report cannot
establish the latter relation. -/
import VeriTile.Triton.Float.SoftmaxShift

namespace VeriTile.Triton.FP.LogSumExpShift
open Structural Guarded ScalarArithmetic ScalarReduction GuardExpression
open SoftmaxShift (exp exponentials shifted scale LibdeviceExpSub)
open WelfordConditions
open Equational (ReductionTree)

def log {α : Type} (M : Algebra α) (a : α) : α := M.unary (some .fp32) .log a

structure IntrinsicLogMul {α : Type} (M : Algebra α) (D : Domain α) : Prop where
  apply : ∀ a b, D .finite a → D .finite b → D .positive a → D .positive b →
    log M (mul M a b) = add M (log M a) (log M b)

structure LogLibdeviceExp {α : Type} (M : Algebra α) (D : Domain α) : Prop where
  apply : ∀ a, D .finite a → log M (exp M a) = a

/-- Only domain predicates on concrete operands and intermediate results.
No reduction or logarithm equation is concealed in the domain. -/
structure ShiftDomain {α : Type} (M : Algebra α) (D : Domain α)
    (xs : Fin N → α) (m : α) (tree : ReductionTree N) : Prop where
  inputs : ∀ i, D .finite (xs i)
  center : D .finite m
  centerExp : D .finite (exp M m)
  centerExpNonzero : D .nonzero (exp M m)
  centerExpPositive : D .positive (exp M m)
  expInputs : ∀ i, D .finite (exponentials M xs i)
  inverseCenterExp : D .finite (scale M m)
  zeroFinite : D .finite (zero M)
  scaledZero : D .finite (mul M (scale M m) (zero M))
  negScaledZero : D .finite (sub M (zero M) (mul M (scale M m) (zero M)))
  partialSums : FiniteTree M D (exponentials M xs) (zero M) tree
  shiftedSum : D .finite (value M (shifted M xs m) (zero M) tree)
  shiftedSumPositive : D .positive (value M (shifted M xs m) (zero M) tree)
  logShiftedSum : D .finite (log M (value M (shifted M xs m) (zero M) tree))

theorem recover_sum {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (hExp : LibdeviceExpSub M D) (xs : Fin N → α) (m : α) (tree : ReductionTree N)
    (hd : ShiftDomain M D xs m tree) :
    mul M (value M (shifted M xs m) (zero M) tree) (exp M m) =
      value M (exponentials M xs) (zero M) tree := by
  have hlane (i : Fin N) : shifted M xs m i = mul M (exponentials M xs i) (scale M m) := by
    unfold shifted scale exponentials
    rw [hExp.apply _ _ (hd.inputs i) hd.center]
    exact div_mul_rcp R M D hM s _ _ (hd.expInputs i) hd.centerExp hd.centerExpNonzero
  have hsum := value_congr M _ _ (zero M) (zero M) hlane rfl tree
  rw [hsum, factor_right R M D hM s _ hd.inverseCenterExp hd.zeroFinite hd.scaledZero
    hd.negScaledZero _ hd.expInputs tree hd.partialSums]
  have hf := value_finite M D _ (zero M) tree hd.partialSums
  rw [mul_assoc R M D hM s _ _ _ hf hd.inverseCenterExp hd.centerExp,
    mul_comm R M D hM s _ _ hd.inverseCenterExp hd.centerExp]
  rw [show mul M (exp M m) (scale M m) = one M from
    mul_rcp_cancel R M D hM s _ hd.centerExp hd.centerExpNonzero,
    mul_one R M D hM s _ hf]

/-- Recover the unshifted sum, apply the scalar log-product relation, then
cancel log(exp(m)). Max is still opaque and no whole-row identity is assumed. -/
theorem shifted_result {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (hExp : LibdeviceExpSub M D) (hMul : IntrinsicLogMul M D) (hLogExp : LogLibdeviceExp M D)
    (xs : Fin N → α) (m : α) (tree : ReductionTree N) (hd : ShiftDomain M D xs m tree) :
    add M m (log M (value M (shifted M xs m) (zero M) tree)) =
      log M (value M (exponentials M xs) (zero M) tree) := by
  rw [← recover_sum R M D hM s hExp xs m tree hd,
    hMul.apply _ _ hd.shiftedSum hd.centerExp hd.shiftedSumPositive hd.centerExpPositive,
    hLogExp.apply m hd.center]
  exact add_comm R M D hM s _ _ hd.center hd.logShiftedSum

def checks {α : Type} (M : Algebra α) (xs : Fin N → α) (m : α)
    (tree : ReductionTree N) : Requirements α :=
  .all [.each N (fun i => .guard .finite (xs i)), .guard .finite m,
    .guard .finite (exp M m), .guard .nonzero (exp M m), .guard .positive (exp M m),
    .each N (fun i => .guard .finite (exponentials M xs i)), .guard .finite (scale M m),
    .guard .finite (zero M), .guard .finite (mul M (scale M m) (zero M)),
    .guard .finite (sub M (zero M) (mul M (scale M m) (zero M))),
    finiteTree M (exponentials M xs) (zero M) tree,
    .guard .finite (value M (shifted M xs m) (zero M) tree),
    .guard .positive (value M (shifted M xs m) (zero M) tree),
    .guard .finite (log M (value M (shifted M xs m) (zero M) tree))]

@[simp] theorem checks_holds {α : Type} (M : Algebra α) (D : Domain α)
    (xs : Fin N → α) (m : α) (tree : ReductionTree N) :
    (checks M xs m tree).Holds D ↔ ShiftDomain M D xs m tree := by
  simp only [checks, Requirements.holds_all, List.mem_cons, List.not_mem_nil,
    forall_eq_or_imp, false_implies, implies_true, and_true, Requirements.holds_guard,
    Requirements.holds_each, finiteTree_holds]
  constructor
  · rintro ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩
    exact ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩
  · intro h
    exact ⟨h.inputs, h.center, h.centerExp, h.centerExpNonzero, h.centerExpPositive,
      h.expInputs, h.inverseCenterExp, h.zeroFinite, h.scaledZero, h.negScaledZero,
      h.partialSums, h.shiftedSum, h.shiftedSumPositive, h.logShiftedSum⟩

@[simp] theorem eval_log {α Input : Type} (M : Algebra α) (read : Input → α) (x : Expr Input) :
    (log (GuardExpression.algebra Input) x).eval M read = log M (x.eval M read) := rfl

@[simp] theorem map_checks {α Input : Type} (M : Algebra α) (read : Input → α)
    (xs : Fin N → Expr Input) (m : Expr Input) (tree : ReductionTree N) :
    (checks (GuardExpression.algebra Input) xs m tree).map (Expr.eval M read) =
      checks M (fun i => (xs i).eval M read) (m.eval M read) tree := by
  simp [checks, Requirements.map, exponentials, shifted, scale]
  rfl

end VeriTile.Triton.FP.LogSumExpShift
