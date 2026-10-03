/- Syntactic domain requirements for the Welford derivation. Every record
field is compiled to a finite/nonzero check; loop indices and reduction
rewrite operands are enumerated explicitly. No equality can be a check. -/
import VeriTile.Triton.Float.GuardExpression
import VeriTile.Triton.Float.WelfordSchedule

namespace VeriTile.Triton.FP.WelfordConditions
open Structural Guarded ScalarArithmetic ScalarReduction WelfordReduction
open Welford (deviationSquares crossTerms square oldPlusCross crossPlusShift)
open WelfordAppend (appendTree)
open GuardExpression
open Equational (ReductionTree ReductionPlan)

variable {α Input : Type}

/-- Check every node of this concrete reduction tree, including padding. -/
def finiteTree (M : Algebra α) (xs : Fin N → α) (seed : α) :
    ReductionTree N → Requirements α
  | .input i => .guard .finite (xs i)
  | .zero => .guard .finite seed
  | .add a b => .all [finiteTree M xs seed a, finiteTree M xs seed b,
      .guard .finite (add M (value M xs seed a) (value M xs seed b))]

@[simp] theorem finiteTree_holds (M : Algebra α) (D : Domain α)
    (xs : Fin N → α) (seed : α) (t : ReductionTree N) :
    (finiteTree M xs seed t).Holds D ↔ FiniteTree M D xs seed t := by
  induction t with
  | input => rfl
  | zero => rfl
  | add a b ha hb => simp [finiteTree, FiniteTree, ha, hb]

@[simp] theorem eval_zero (M : Algebra α) (read : Input → α) :
    (zero (GuardExpression.algebra Input)).eval M read = zero M := rfl
@[simp] theorem eval_one (M : Algebra α) (read : Input → α) :
    (one (GuardExpression.algebra Input)).eval M read = one M := rfl
@[simp] theorem eval_add (M : Algebra α) (read : Input → α) (a b : Expr Input) :
    (add (GuardExpression.algebra Input) a b).eval M read =
      add M (a.eval M read) (b.eval M read) := rfl
@[simp] theorem eval_sub (M : Algebra α) (read : Input → α) (a b : Expr Input) :
    (sub (GuardExpression.algebra Input) a b).eval M read =
      sub M (a.eval M read) (b.eval M read) := rfl
@[simp] theorem eval_mul (M : Algebra α) (read : Input → α) (a b : Expr Input) :
    (mul (GuardExpression.algebra Input) a b).eval M read =
      mul M (a.eval M read) (b.eval M read) := rfl
@[simp] theorem eval_div (M : Algebra α) (read : Input → α) (a b : Expr Input) :
    (div (GuardExpression.algebra Input) a b).eval M read =
      div M (a.eval M read) (b.eval M read) := rfl

@[simp] theorem eval_value (M : Algebra α) (read : Input → α)
    (xs : Fin N → Expr Input) (seed : Expr Input) (t : ReductionTree N) :
    (value (GuardExpression.algebra Input) xs seed t).eval M read =
      value M (fun i => (xs i).eval M read) (seed.eval M read) t := by
  induction t with
  | input => rfl
  | zero => rfl
  | add a b ha hb => simp only [value, eval_add, ha, hb]

@[simp] theorem eval_snoc (M : Algebra α) (read : Input → α)
    (xs : Fin N → Expr Input) (x : Expr Input) :
    (fun i : Fin (N + 1) => (Fin.snoc (α := fun _ => Expr Input) xs x i).eval M read) =
      Fin.snoc (fun i => (xs i).eval M read) (x.eval M read) := by
  funext i
  refine Fin.lastCases ?_ (fun j => ?_) i <;> simp

@[simp] theorem map_finiteTree (M : Algebra α) (read : Input → α)
    (xs : Fin N → Expr Input) (seed : Expr Input) (t : ReductionTree N) :
    (finiteTree (GuardExpression.algebra Input) xs seed t).map (Expr.eval M read) =
      finiteTree M (fun i => (xs i).eval M read) (seed.eval M read) t := by
  induction t with
  | input => rfl
  | zero => rfl
  | add a b ha hb =>
      simp only [finiteTree, Requirements.map_all, List.map_cons, List.map_nil,
        Requirements.map, eval_add, eval_value, ha, hb]

def initial (M : Algebra α) (x : α) : Requirements α :=
  .all [
    .guard .finite x,
    .guard .finite (ScalarArithmetic.zero M),
    .guard .finite (ScalarArithmetic.one M),
    .guard .finite (sub M x (ScalarArithmetic.zero M)),
    .guard .finite (sub M x x),
    .guard .finite (sub M (ScalarArithmetic.zero M) x),
    .guard .finite (mul M x (ScalarArithmetic.zero M)),
    .guard .finite (sub M (ScalarArithmetic.zero M) (mul M x (ScalarArithmetic.zero M))),
    .guard .finite (Welford.square M (ScalarArithmetic.zero M)),
    .guard .finite (sub M (ScalarArithmetic.zero M) (Welford.square M (ScalarArithmetic.zero M)))]

@[simp] theorem initial_holds (M : Algebra α) (D : Domain α) (x : α) :
    (initial M x).Holds D ↔ WelfordInit.InitDomain M D x := by
  simp only [initial, Requirements.holds_all, List.mem_cons, List.not_mem_nil,
    forall_eq_or_imp, false_implies, implies_true, and_true, Requirements.holds_guard]
  constructor
  · rintro ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9⟩
    exact ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9⟩
  · intro h
    exact ⟨h.input, h.zero, h.one, h.difference, h.diagonal, h.negativeInput, h.scaledZero, h.negativeScaledZero, h.zeroSquare, h.negativeZeroSquare⟩

@[simp] theorem map_initial (M : Algebra α) (read : Input → α)
    (x : Expr Input) :
    (initial (GuardExpression.algebra Input) x).map (Expr.eval M read) =
      initial M (x.eval M read) := by
  simp [initial, Requirements.map, Welford.square]

def meanStep (M : Algebra α) (x m n : α) : Requirements α :=
  .all [
    .guard .finite x,
    .guard .finite m,
    .guard .finite n,
    .guard .finite (ScalarArithmetic.one M),
    .guard .finite (Welford.difference M x m),
    .guard .finite (Welford.nextCount M n),
    .guard .nonzero (Welford.nextCount M n),
    .guard .finite (div M (ScalarArithmetic.one M) (Welford.nextCount M n)),
    .guard .finite (Welford.correction M x m n),
    .guard .finite (Welford.nextMean M x m n),
    .guard .finite (mul M m n)]

@[simp] theorem meanStep_holds (M : Algebra α) (D : Domain α) (x m n : α) :
    (meanStep M x m n).Holds D ↔ Welford.MeanStepDomain M D x m n := by
  simp only [meanStep, Requirements.holds_all, List.mem_cons, List.not_mem_nil,
    forall_eq_or_imp, false_implies, implies_true, and_true, Requirements.holds_guard]
  constructor
  · rintro ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩
    exact ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩
  · intro h
    exact ⟨h.input, h.mean, h.count, h.one, h.difference, h.nextCount, h.countNonzero, h.inverseCount, h.correction, h.nextMean, h.weightedMean⟩

@[simp] theorem map_meanStep (M : Algebra α) (read : Input → α)
    (x m n : Expr Input) :
    (meanStep (GuardExpression.algebra Input) x m n).map (Expr.eval M read) =
      meanStep M (x.eval M read) (m.eval M read) (n.eval M read) := by
  simp [meanStep, Requirements.map, Welford.nextCount, Welford.difference, Welford.correction, Welford.nextMean]

def varianceStep (M : Algebra α) (x m n : α) : Requirements α :=
  .all [
    meanStep M x m n,
    .guard .finite (Welford.residual M x m n),
    .guard .finite (mul M n (Welford.correction M x m n)),
    .guard .finite (ScalarArithmetic.zero M),
    .guard .finite (sub M (ScalarArithmetic.zero M) (Welford.nextMean M x m n))]

@[simp] theorem varianceStep_holds (M : Algebra α) (D : Domain α) (x m n : α) :
    (varianceStep M x m n).Holds D ↔ Welford.VarianceStepDomain M D x m n := by
  simp only [varianceStep, Requirements.holds_all, List.mem_cons, List.not_mem_nil,
    forall_eq_or_imp, false_implies, implies_true, and_true, Requirements.holds_guard, meanStep_holds]
  constructor
  · rintro ⟨h0, h1, h2, h3, h4⟩
    exact ⟨h0, h1, h2, h3, h4⟩
  · intro h
    exact ⟨h.toMeanStepDomain, h.residual, h.scaledCorrection, h.zero, h.negativeNextMean⟩

@[simp] theorem map_varianceStep (M : Algebra α) (read : Input → α)
    (x m n : Expr Input) :
    (varianceStep (GuardExpression.algebra Input) x m n).map (Expr.eval M read) =
      varianceStep M (x.eval M read) (m.eval M read) (n.eval M read) := by
  simp [varianceStep, Requirements.map, Welford.nextCount, Welford.difference, Welford.correction, Welford.nextMean,
      Welford.residual]

def centerShift (M : Algebra α) (x m m' : α) : Requirements α :=
  .all [
    .guard .finite x,
    .guard .finite m,
    .guard .finite m',
    .guard .finite (sub M x m),
    .guard .finite (sub M m m'),
    .guard .finite (add M (sub M x m) (sub M m m')),
    .guard .finite (sub M x m'),
    .guard .finite (ScalarArithmetic.zero M),
    .guard .finite (sub M (ScalarArithmetic.zero M) m')]

@[simp] theorem centerShift_holds (M : Algebra α) (D : Domain α) (x m m' : α) :
    (centerShift M x m m').Holds D ↔ Welford.CenterShiftDomain M D x m m' := by
  simp only [centerShift, Requirements.holds_all, List.mem_cons, List.not_mem_nil,
    forall_eq_or_imp, false_implies, implies_true, and_true, Requirements.holds_guard]
  constructor
  · rintro ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8⟩
    exact ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8⟩
  · intro h
    exact ⟨h.input, h.oldMean, h.newMean, h.oldDeviation, h.meanShift, h.joinedDeviation, h.newDeviation, h.zero, h.negativeNewMean⟩

@[simp] theorem map_centerShift (M : Algebra α) (read : Input → α)
    (x m m' : Expr Input) :
    (centerShift (GuardExpression.algebra Input) x m m').map (Expr.eval M read) =
      centerShift M (x.eval M read) (m.eval M read) (m'.eval M read) := by
  simp [centerShift, Requirements.map]

def centerTree (M : Algebra α) (xs : Fin N → α) (m m' : α) (tree : ReductionTree N) : Requirements α :=
  .all [
    .each N (fun i => centerShift M (xs i) m m'),
    .guard .finite (ScalarArithmetic.zero M),
    finiteTree M (deviationSquares M xs m) (ScalarArithmetic.zero M) tree,
    finiteTree M (crossTerms M xs m m') (ScalarArithmetic.zero M) tree,
    finiteTree M (fun _ => square M (sub M m m')) (ScalarArithmetic.zero M) tree,
    finiteTree M (oldPlusCross M xs m m') (ScalarArithmetic.zero M) tree,
    finiteTree M (crossPlusShift M xs m m') (ScalarArithmetic.zero M) tree,
    finiteTree M (deviationSquares M xs m') (ScalarArithmetic.zero M) tree]

@[simp] theorem centerTree_holds (M : Algebra α) (D : Domain α) (xs : Fin N → α) (m m' : α) (tree : ReductionTree N) :
    (centerTree M xs m m' tree).Holds D ↔ Welford.CenterShiftTreeDomain M D xs m m' tree := by
  simp only [centerTree, Requirements.holds_all, List.mem_cons, List.not_mem_nil,
    forall_eq_or_imp, false_implies, implies_true, and_true, Requirements.holds_guard, Requirements.holds_each,
    centerShift_holds, finiteTree_holds]
  constructor
  · rintro ⟨h0, h1, h2, h3, h4, h5, h6, h7⟩
    exact ⟨h0, h1, h2, h3, h4, h5, h6, h7⟩
  · intro h
    exact ⟨h.pointwise, h.zero, h.oldSquares, h.cross, h.shiftSquares, h.leftSum, h.rightSum, h.newSquares⟩

@[simp] theorem map_centerTree (M : Algebra α) (read : Input → α)
    (xs : Fin N → Expr Input) (m m' : Expr Input) (tree : ReductionTree N) :
    (centerTree (GuardExpression.algebra Input) xs m m' tree).map (Expr.eval M read) =
      centerTree M (fun i => (xs i).eval M read) (m.eval M read) (m'.eval M read) tree := by
  simp [centerTree, Requirements.map, Welford.square, Welford.deviationSquares, Welford.crossTerms,
      Welford.oldPlusCross, Welford.crossPlusShift]
  rfl

def meanDomain (M : Algebra α) (xs : Fin N → α) (tree : ReductionTree N) : Requirements α :=
  .all [
    .each N (fun i => .guard .finite (xs i)),
    .guard .finite (ScalarArithmetic.zero M),
    finiteTree M xs (ScalarArithmetic.zero M) tree,
    finiteTree M (fun _ => one M) (ScalarArithmetic.zero M) tree,
    .guard .nonzero (count M tree),
    .guard .finite (div M (one M) (count M tree)),
    .guard .finite (WelfordReduction.mean M xs tree),
    .guard .finite (mul M (WelfordReduction.mean M xs tree) (ScalarArithmetic.zero M)),
    .guard .finite (sub M (ScalarArithmetic.zero M) (mul M (WelfordReduction.mean M xs tree) (ScalarArithmetic.zero M))),
    finiteTree M (fun i => sub M (xs i) (WelfordReduction.mean M xs tree)) (ScalarArithmetic.zero M) tree,
    finiteTree M (fun _ => WelfordReduction.mean M xs tree) (ScalarArithmetic.zero M) tree,
    .guard .finite (sub M (ScalarArithmetic.zero M) (value M (fun _ => WelfordReduction.mean M xs tree) (ScalarArithmetic.zero M) tree))]

@[simp] theorem meanDomain_holds (M : Algebra α) (D : Domain α) (xs : Fin N → α) (tree : ReductionTree N) :
    (meanDomain M xs tree).Holds D ↔ WelfordReduction.MeanDomain M D xs tree := by
  simp only [meanDomain, Requirements.holds_all, List.mem_cons, List.not_mem_nil,
    forall_eq_or_imp, false_implies, implies_true, and_true, Requirements.holds_guard, Requirements.holds_each,
    finiteTree_holds]
  constructor
  · rintro ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩
    exact ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩
  · intro h
    exact ⟨h.inputs, h.zero, h.inputTree, h.onesTree, h.countNonzero, h.inverseCount, h.mean, h.scaledZero, h.negativeScaledZero, h.deviations, h.center, h.negativeCenterSum⟩

@[simp] theorem map_meanDomain (M : Algebra α) (read : Input → α)
    (xs : Fin N → Expr Input) (tree : ReductionTree N) :
    (meanDomain (GuardExpression.algebra Input) xs tree).map (Expr.eval M read) =
      meanDomain M (fun i => (xs i).eval M read) tree := by
  simp [meanDomain, Requirements.map, mean, count]

def recenter (M : Algebra α) (xs : Fin N → α) (tree : ReductionTree N) (m' : α) : Requirements α :=
  .all [
    meanDomain M xs tree,
    centerTree M xs (mean M xs tree) m' tree,
    .guard .finite (sub M (mean M xs tree) m'),
    .guard .finite (mul M (sub M (mean M xs tree) m') (zero M)),
    .guard .finite (sub M (zero M) (mul M (sub M (mean M xs tree) m') (zero M)))]

@[simp] theorem recenter_holds (M : Algebra α) (D : Domain α) (xs : Fin N → α) (tree : ReductionTree N) (m' : α) :
    (recenter M xs tree m').Holds D ↔ WelfordReduction.RecenterDomain M D xs tree m' := by
  simp only [recenter, Requirements.holds_all, List.mem_cons, List.not_mem_nil,
    forall_eq_or_imp, false_implies, implies_true, and_true, Requirements.holds_guard, centerTree_holds, meanDomain_holds]
  constructor
  · rintro ⟨h0, h1, h2, h3, h4⟩
    exact ⟨h0, h1, h2, h3, h4⟩
  · intro h
    exact ⟨h.centering, h.squareShift, h.shift, h.scaledZero, h.negativeScaledZero⟩

@[simp] theorem map_recenter (M : Algebra α) (read : Input → α)
    (xs : Fin N → Expr Input) (tree : ReductionTree N) (m' : Expr Input) :
    (recenter (GuardExpression.algebra Input) xs tree m').map (Expr.eval M read) =
      recenter M (fun i => (xs i).eval M read) tree (m'.eval M read) := by
  simp [recenter, Requirements.map, mean, count]

def shiftSquare (M : Algebra α) (x m n : α) : Requirements α :=
  .all [
    .guard .finite (ScalarArithmetic.zero M),
    .guard .finite (WelfordAppend.shift M x m n),
    .guard .finite (add M (WelfordAppend.shift M x m n) (Welford.correction M x m n)),
    .guard .finite (sub M (ScalarArithmetic.zero M) m),
    .guard .finite (Welford.square M (WelfordAppend.shift M x m n)),
    .guard .finite (Welford.square M (Welford.correction M x m n)),
    .guard .finite (mul M (WelfordAppend.shift M x m n) (Welford.correction M x m n)),
    .guard .finite (sub M (ScalarArithmetic.zero M) (mul M (WelfordAppend.shift M x m n) (Welford.correction M x m n))),
    .guard .finite (mul M (WelfordAppend.shift M x m n) (ScalarArithmetic.zero M)),
    .guard .finite (sub M (ScalarArithmetic.zero M) (mul M (WelfordAppend.shift M x m n) (ScalarArithmetic.zero M))),
    .guard .finite (mul M (Welford.correction M x m n) (ScalarArithmetic.zero M)),
    .guard .finite (sub M (ScalarArithmetic.zero M) (mul M (Welford.correction M x m n) (ScalarArithmetic.zero M)))]

@[simp] theorem shiftSquare_holds (M : Algebra α) (D : Domain α) (x m n : α) :
    (shiftSquare M x m n).Holds D ↔ WelfordAppend.ShiftSquareDomain M D x m n := by
  simp only [shiftSquare, Requirements.holds_all, List.mem_cons, List.not_mem_nil,
    forall_eq_or_imp, false_implies, implies_true, and_true, Requirements.holds_guard]
  constructor
  · rintro ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩
    exact ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩
  · intro h
    exact ⟨h.zero, h.shift, h.joined, h.negativeMean, h.shiftSquare, h.correctionSquare, h.cross, h.negativeCross, h.shiftZero, h.negativeShiftZero, h.correctionZero, h.negativeCorrectionZero⟩

@[simp] theorem map_shiftSquare (M : Algebra α) (read : Input → α)
    (x m n : Expr Input) :
    (shiftSquare (GuardExpression.algebra Input) x m n).map (Expr.eval M read) =
      shiftSquare M (x.eval M read) (m.eval M read) (n.eval M read) := by
  simp [shiftSquare, Requirements.map, Welford.nextCount, Welford.difference, Welford.correction, Welford.nextMean,
      Welford.square, WelfordAppend.shift]

def varianceAppend (M : Algebra α) (xs : Fin N → α) (tree : ReductionTree N) (x : α) : Requirements α :=
  .all [
    varianceStep M x (mean M xs tree) (count M tree),
    recenter M xs tree (Welford.nextMean M x (mean M xs tree) (count M tree)),
    shiftSquare M x (mean M xs tree) (count M tree),
    .guard .finite (add M (value M xs (zero M) tree) x),
    .guard .finite (mean M (Fin.snoc xs x) (appendTree tree)),
    .guard .finite (Welford.square M (Welford.residual M x (mean M xs tree) (count M tree))),
    .guard .finite (mul M (Welford.square M (WelfordAppend.shift M x (mean M xs tree) (count M tree))) (zero M)),
    .guard .finite (sub M (zero M) (mul M (Welford.square M (WelfordAppend.shift M x (mean M xs tree) (count M tree))) (zero M)))]

@[simp] theorem varianceAppend_holds (M : Algebra α) (D : Domain α) (xs : Fin N → α) (tree : ReductionTree N) (x : α) :
    (varianceAppend M xs tree x).Holds D ↔ WelfordAppend.VarianceAppendDomain M D xs tree x := by
  simp only [varianceAppend, Requirements.holds_all, List.mem_cons, List.not_mem_nil,
    forall_eq_or_imp, false_implies, implies_true, and_true, Requirements.holds_guard, recenter_holds, shiftSquare_holds, varianceStep_holds]
  constructor
  · rintro ⟨h0, h1, h2, h3, h4, h5, h6, h7⟩
    exact ⟨h0, h1, h2, h3, h4, h5, h6, h7⟩
  · intro h
    exact ⟨h.step, h.recenter, h.shift, h.newSum, h.newMean, h.residualSquare, h.shiftSquareZero, h.negativeShiftSquareZero⟩

@[simp] theorem map_varianceAppend (M : Algebra α) (read : Input → α)
    (xs : Fin N → Expr Input) (tree : ReductionTree N) (x : Expr Input) :
    (varianceAppend (GuardExpression.algebra Input) xs tree x).map (Expr.eval M read) =
      varianceAppend M (fun i => (xs i).eval M read) tree (x.eval M read) := by
  simp [varianceAppend, Requirements.map, Welford.nextCount, Welford.difference, Welford.correction, Welford.nextMean,
      Welford.residual, Welford.square, WelfordAppend.shift,
      mean, count]

/-- Every noninitial loop step receives its own checks, even if the final
state alone would satisfy the domain. -/
def iterations (M : Algebra α) (xs : Nat → α) (empty : ReductionTree 0)
    (N : Nat) : Requirements α :=
  .both (initial M (xs 0)) (.each N (fun i =>
    if 0 < i.val then
      varianceAppend M (WelfordInduction.rowPrefix xs i.val)
        (WelfordInduction.tree empty i.val) (xs i.val)
    else .top))

@[simp] theorem iterations_holds (M : Algebra α) (D : Domain α) (xs : Nat → α)
    (empty : ReductionTree 0) (N : Nat) :
    (iterations M xs empty N).Holds D ↔ WelfordInduction.IterationDomain M D xs empty N := by
  simp only [iterations, Requirements.holds_both, initial_holds, Requirements.holds_each]
  constructor
  · rintro ⟨hi, hs⟩
    refine ⟨hi, fun i hpos hn => ?_⟩
    simpa only [if_pos hpos, varianceAppend_holds] using hs ⟨i, hn⟩
  · intro h
    refine ⟨h.initial, fun i => ?_⟩
    split
    · exact (varianceAppend_holds M D _ _ _).mpr (h.steps i.val ‹_› i.isLt)
    · trivial

@[simp] theorem map_iterations (M : Algebra α) (read : Input → α)
    (xs : Nat → Expr Input) (empty : ReductionTree 0) (N : Nat) :
    (iterations (GuardExpression.algebra Input) xs empty N).map (Expr.eval M read) =
      iterations M (fun i => (xs i).eval M read) empty N := by
  simp only [iterations, Requirements.map, map_initial]
  congr 2
  funext i
  split
  · exact map_varianceAppend M read _ _ _
  · rfl

/-- The actual normalization path's operand checks, including intermediate
values introduced by reassociation. -/
def rewritePath (M : Algebra α) (xs : Fin N → α) (t : ReductionTree N) : Requirements α :=
  .all ((ReductionSchedule.normalize t).operands.map
    (fun node => .guard .finite (value M xs (zero M) node)))

@[simp] theorem rewritePath_holds (M : Algebra α) (D : Domain α)
    (xs : Fin N → α) (t : ReductionTree N) :
    (rewritePath M xs t).Holds D ↔ (ReductionSchedule.normalize t).Domain M D xs := by
  simp [rewritePath, ReductionSchedule.Rewrite.Domain]

@[simp] theorem map_rewritePath (M : Algebra α) (read : Input → α)
    (xs : Fin N → Expr Input) (t : ReductionTree N) :
    (rewritePath (GuardExpression.algebra Input) xs t).map (Expr.eval M read) =
      rewritePath M (fun i => (xs i).eval M read) t := by
  simp [rewritePath, List.map_map, Requirements.map, Function.comp_def]

def schedule (M : Algebra α) (xs : Fin N → α) (a b : ReductionPlan N) : Requirements α :=
  .both (rewritePath M xs a.tree) (rewritePath M xs b.tree)

@[simp] theorem schedule_holds (M : Algebra α) (D : Domain α)
    (xs : Fin N → α) (a b : ReductionPlan N) :
    (schedule M xs a b).Holds D ↔ ReductionSchedule.ScheduleDomain M D xs a b := by
  simp only [schedule, Requirements.holds_both, rewritePath_holds]
  exact ⟨fun h => ⟨h.1, h.2⟩, fun h => ⟨h.left, h.right⟩⟩

@[simp] theorem map_schedule (M : Algebra α) (read : Input → α)
    (xs : Fin N → Expr Input) (a b : ReductionPlan N) :
    (schedule (GuardExpression.algebra Input) xs a b).map (Expr.eval M read) =
      schedule M (fun i => (xs i).eval M read) a b := by
  simp [schedule, Requirements.map]

def statistics (M : Algebra α) (xs : Fin N → α) (a b : ReductionPlan N) : Requirements α :=
  .all [schedule M xs a b, schedule M (fun _ => one M) a b,
    schedule M (Welford.deviationSquares M xs (mean M xs a.tree)) a b]

@[simp] theorem statistics_holds (M : Algebra α) (D : Domain α)
    (xs : Fin N → α) (a b : ReductionPlan N) :
    (statistics M xs a b).Holds D ↔ WelfordSchedule.StatisticsDomain M D xs a b := by
  simp only [statistics, Requirements.holds_all, List.mem_cons, List.not_mem_nil,
    forall_eq_or_imp, false_implies, implies_true, and_true, schedule_holds]
  exact ⟨fun h => ⟨h.1, h.2.1, h.2.2⟩, fun h => ⟨h.inputs, h.counts, h.squares⟩⟩

@[simp] theorem map_statistics (M : Algebra α) (read : Input → α)
    (xs : Fin N → Expr Input) (a b : ReductionPlan N) :
    (statistics (GuardExpression.algebra Input) xs a b).map (Expr.eval M read) =
      statistics M (fun i => (xs i).eval M read) a b := by
  simp [statistics, Welford.deviationSquares, Welford.square, mean, count]
  rfl

/-- The complete requirements for the existing Welford comparison proof.
The count-conversion equations are deliberately absent: they require admission. -/
def complete (M : Algebra α) (xs : Nat → α) (empty : ReductionPlan 0)
    (batch : ReductionPlan N) : Requirements α :=
  .both (iterations M xs empty.tree N)
    (statistics M (WelfordInduction.rowPrefix xs N) (WelfordInduction.plan empty N) batch)

@[simp] theorem complete_holds (M : Algebra α) (D : Domain α)
    (xs : Nat → α) (empty : ReductionPlan 0) (batch : ReductionPlan N) :
    (complete M xs empty batch).Holds D ↔
      WelfordInduction.IterationDomain M D xs empty.tree N ∧
      WelfordSchedule.StatisticsDomain M D (WelfordInduction.rowPrefix xs N)
        (WelfordInduction.plan empty N) batch := by
  simp [complete]

@[simp] theorem map_complete (M : Algebra α) (read : Input → α)
    (xs : Nat → Expr Input) (empty : ReductionPlan 0) (batch : ReductionPlan N) :
    (complete (GuardExpression.algebra Input) xs empty batch).map (Expr.eval M read) =
      complete M (fun i => (xs i).eval M read) empty batch := by
  simp [complete, Requirements.map, WelfordInduction.rowPrefix]
  rfl

end VeriTile.Triton.FP.WelfordConditions
