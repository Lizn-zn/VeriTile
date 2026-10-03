/- Conditional softmax shift derived through scalar arithmetic and an explicit
sum tree. The original tl.exp EXP-SUB relation still needs matching admission:
the published libdevice.exp row is deliberately not bound to this symbol. -/
import VeriTile.Triton.Float.WelfordConditions

namespace VeriTile.Triton.FP.SoftmaxShift
open Structural Guarded ScalarArithmetic ScalarReduction GuardExpression
open Equational (ReductionTree)
open WelfordConditions

def exp {α : Type} (M : Algebra α) (a : α) : α := M.unary (some .fp32) .exp a

def exponentials {α : Type} (M : Algebra α) (xs : Fin N → α) : Fin N → α :=
  fun i => exp M (xs i)

def shifted {α : Type} (M : Algebra α) (xs : Fin N → α) (m : α) : Fin N → α :=
  fun i => exp M (sub M (xs i) m)

def scale {α : Type} (M : Algebra α) (m : α) : α := div M (one M) (exp M m)

/-- A scalar obligation for the original intrinsic, not an admitted rule or
whole-softmax premise. Its two operands retain their finite-value guards. -/
structure IntrinsicExpSub {α : Type} (M : Algebra α) (D : Domain α) : Prop where
  apply : ∀ a b, D .finite a → D .finite b →
    exp M (sub M a b) = div M (exp M a) (exp M b)

/-- All other requirements are domain predicates. The common shift may be
any opaque finite value; no algebraic property of reduceMax is assumed. -/
structure ShiftDomain {α : Type} (M : Algebra α) (D : Domain α)
    (xs : Fin N → α) (m : α) (tree : ReductionTree N) : Prop where
  inputs : ∀ i, D .finite (xs i)
  center : D .finite m
  centerExp : D .finite (exp M m)
  centerExpNonzero : D .nonzero (exp M m)
  normalization : NormalizationDomain M D (exponentials M xs) (scale M m) tree

theorem shifted_lane {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (hExp : IntrinsicExpSub M D) (xs : Fin N → α) (m : α) (tree : ReductionTree N)
    (hd : ShiftDomain M D xs m tree) (i : Fin N) :
    shifted M xs m i = mul M (exponentials M xs i) (scale M m) := by
  unfold shifted scale exponentials
  rw [hExp.apply _ _ (hd.inputs i) hd.center]
  exact div_mul_rcp R M D hM s _ _ (hd.normalization.inputs i) hd.centerExp hd.centerExpNonzero

/-- The reduction identity and shared-scale cancellation are derived from
scalar atoms; neither a reduction nor a softmax identity is assumed. -/
theorem normalized {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (hExp : IntrinsicExpSub M D) (xs : Fin N → α) (m : α) (tree : ReductionTree N)
    (hd : ShiftDomain M D xs m tree) (i : Fin N) :
    div M (shifted M xs m i) (value M (shifted M xs m) (zero M) tree) =
      div M (exponentials M xs i) (value M (exponentials M xs) (zero M) tree) := by
  have hrow := shifted_lane R M D hM s hExp xs m tree hd
  rw [value_congr M _ _ (zero M) (zero M) hrow rfl tree, hrow i]
  exact normalize_reduction R M D hM s _ _ tree hd.normalization i

/-- Reify the exact normalization record, including reciprocal values and
intermediate tree sums. Only finite/nonzero checks enter the public domain. -/
def normalizationChecks {α : Type} (M : Algebra α) (xs : Fin N → α) (c : α)
    (tree : ReductionTree N) : Requirements α :=
  .all [
    .each N (fun i => .guard .finite (xs i)),
    .guard .finite c,
    .guard .nonzero c,
    .guard .finite (zero M),
    .guard .finite (mul M c (zero M)),
    .guard .finite (sub M (zero M) (mul M c (zero M))),
    finiteTree M xs (zero M) tree,
    .guard .nonzero (value M xs (zero M) tree),
    .each N (fun i => .guard .finite (mul M (xs i) c)),
    .guard .finite (mul M (value M xs (zero M) tree) c),
    .guard .nonzero (mul M (value M xs (zero M) tree) c),
    .guard .finite (div M (one M) (value M xs (zero M) tree)),
    .guard .finite (div M (one M) c),
    .guard .finite (div M (one M) (mul M (value M xs (zero M) tree) c)),
    .each N (fun i => .guard .finite (div M (xs i) (value M xs (zero M) tree))),
    .each N (fun i => .guard .finite
      (div M (mul M (xs i) c) (mul M (value M xs (zero M) tree) c))),
    .each N (fun i => .guard .finite
      (mul M (div M (mul M (xs i) c) (mul M (value M xs (zero M) tree) c))
        (value M xs (zero M) tree)))]

@[simp] theorem normalizationChecks_holds {α : Type} (M : Algebra α) (D : Domain α)
    (xs : Fin N → α) (c : α) (tree : ReductionTree N) :
    (normalizationChecks M xs c tree).Holds D ↔ NormalizationDomain M D xs c tree := by
  simp only [normalizationChecks, Requirements.holds_all, List.mem_cons, List.not_mem_nil,
    forall_eq_or_imp, false_implies, implies_true, and_true, Requirements.holds_guard,
    Requirements.holds_each, finiteTree_holds]
  constructor
  · rintro ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16⟩
    exact ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16⟩
  · intro h
    exact ⟨h.inputs, h.scale, h.scaleNonzero, h.zeroFinite, h.scaledZero, h.negScaledZero,
      h.partialSums, h.sumNonzero, h.scaledInputs, h.scaledSum, h.scaledSumNonzero,
      h.inverseSum, h.inverseScale, h.inverseScaledSum, h.quotient, h.scaledQuotient,
      h.recoveredNumerator⟩

@[simp] theorem map_normalizationChecks {α Input : Type} (M : Algebra α) (read : Input → α)
    (xs : Fin N → Expr Input) (c : Expr Input) (tree : ReductionTree N) :
    (normalizationChecks (GuardExpression.algebra Input) xs c tree).map (Expr.eval M read) =
      normalizationChecks M (fun i => (xs i).eval M read) (c.eval M read) tree := by
  simp [normalizationChecks, Requirements.map]

def checks {α : Type} (M : Algebra α) (xs : Fin N → α) (m : α)
    (tree : ReductionTree N) : Requirements α :=
  .all [.each N (fun i => .guard .finite (xs i)), .guard .finite m,
    .guard .finite (exp M m), .guard .nonzero (exp M m),
    normalizationChecks M (exponentials M xs) (scale M m) tree]

@[simp] theorem checks_holds {α : Type} (M : Algebra α) (D : Domain α)
    (xs : Fin N → α) (m : α) (tree : ReductionTree N) :
    (checks M xs m tree).Holds D ↔ ShiftDomain M D xs m tree := by
  simp only [checks, Requirements.holds_all, List.mem_cons, List.not_mem_nil,
    forall_eq_or_imp, false_implies, implies_true, and_true, Requirements.holds_guard,
    Requirements.holds_each, normalizationChecks_holds]
  exact ⟨fun h => ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2⟩,
    fun h => ⟨h.inputs, h.center, h.centerExp, h.centerExpNonzero, h.normalization⟩⟩

@[simp] theorem eval_exp {α Input : Type} (M : Algebra α) (read : Input → α) (x : Expr Input) :
    (exp (GuardExpression.algebra Input) x).eval M read = exp M (x.eval M read) := rfl

@[simp] theorem map_checks {α Input : Type} (M : Algebra α) (read : Input → α)
    (xs : Fin N → Expr Input) (m : Expr Input) (tree : ReductionTree N) :
    (checks (GuardExpression.algebra Input) xs m tree).map (Expr.eval M read) =
      checks M (fun i => (xs i).eval M read) (m.eval M read) tree := by
  simp [checks, Requirements.map, exponentials, scale]
  rfl

end VeriTile.Triton.FP.SoftmaxShift
