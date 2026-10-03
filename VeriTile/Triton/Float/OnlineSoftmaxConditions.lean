/- Reified arithmetic domains for every iteration of online softmax. -/
import VeriTile.Triton.Float.OnlineSoftmax

namespace VeriTile.Triton.FP.OnlineSoftmaxConditions
open Structural Guarded ScalarArithmetic GuardExpression OnlineSoftmax
open SoftmaxShift (exp)
open WelfordConditions

def initial {α : Type} (M : Algebra α) (x : α) : Requirements α :=
  let m := maximum M M.negInf x
  let c := exp M (sub M M.negInf m)
  .all [.guard .finite x, .guard .finite m, .guard .finite (exp M x),
    .guard .finite (exp M m), .guard .nonzero (exp M m),
    .guard .finite (div M (one M) (exp M m)), .guard .finite (zero M),
    .guard .finite c, .guard .finite (mul M c (zero M)),
    .guard .finite (sub M (zero M) (mul M c (zero M))),
    .guard .finite (exp M (sub M x m))]

@[simp] theorem initial_holds {α : Type} (M : Algebra α) (D : Domain α) (x : α) :
    (initial M x).Holds D ↔ InitDomain M D x := by
  simp only [initial, Requirements.holds_all, List.mem_cons, List.not_mem_nil,
    forall_eq_or_imp, false_implies, implies_true, and_true, Requirements.holds_guard]
  constructor
  · rintro ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩
    exact ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩
  · intro h
    exact ⟨h.input, h.center, h.inputExp, h.centerExp, h.centerExpNonzero,
      h.inverseCenterExp, h.zero, h.seedFactor, h.seedProduct, h.negativeSeedProduct, h.newTerm⟩

def step {α : Type} (M : Algebra α) (x m l : α) : Requirements α :=
  let n := maximum M m x
  .all [.guard .finite x, .guard .finite m, .guard .finite n,
    .guard .finite (exp M x), .guard .finite (exp M m), .guard .finite (exp M n),
    .guard .nonzero (exp M n), .guard .finite (div M (one M) (exp M n)),
    .guard .finite (exp M (sub M m n)), .guard .finite (exp M (sub M x n)),
    .guard .finite l, .guard .finite (mul M (exp M (sub M m n)) l),
    .guard .finite (update M x (m, l)).2]

@[simp] theorem step_holds {α : Type} (M : Algebra α) (D : Domain α) (x m l : α) :
    (step M x m l).Holds D ↔ StepDomain M D x m l := by
  simp only [step, Requirements.holds_all, List.mem_cons, List.not_mem_nil,
    forall_eq_or_imp, false_implies, implies_true, and_true, Requirements.holds_guard]
  constructor
  · rintro ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12⟩
    exact ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12⟩
  · intro h
    exact ⟨h.input, h.oldCenter, h.newCenter, h.inputExp, h.oldCenterExp, h.newCenterExp,
      h.newCenterExpNonzero, h.inverseNewCenterExp, h.oldScale, h.newTerm,
      h.oldTotal, h.scaledOldTotal, h.newTotal⟩

@[simp] theorem eval_maximum {α Input : Type} (M : Algebra α) (read : Input → α) (a b : Expr Input) :
    (maximum (GuardExpression.algebra Input) a b).eval M read = maximum M (a.eval M read) (b.eval M read) := rfl

@[simp] theorem eval_state {α Input : Type} (M : Algebra α) (read : Input → α)
    (xs : Nat → Expr Input) (N : Nat) :
    (((state (GuardExpression.algebra Input) xs N).1).eval M read,
      ((state (GuardExpression.algebra Input) xs N).2).eval M read) =
      state M (fun i => (xs i).eval M read) N := by
  induction N with
  | zero => rfl
  | succ n ih =>
    simp only [state]
    rw [← ih]
    rfl

@[simp] theorem map_initial {α Input : Type} (M : Algebra α) (read : Input → α) (x : Expr Input) :
    (initial (GuardExpression.algebra Input) x).map (Expr.eval M read) = initial M (x.eval M read) := by
  simp [initial, Requirements.map]
  rfl

@[simp] theorem map_step {α Input : Type} (M : Algebra α) (read : Input → α) (x m l : Expr Input) :
    (step (GuardExpression.algebra Input) x m l).map (Expr.eval M read) =
      step M (x.eval M read) (m.eval M read) (l.eval M read) := by
  simp [step, Requirements.map, update]

def iterations {α : Type} (M : Algebra α) (xs : Nat → α) (N : Nat) : Requirements α :=
  .both (initial M (xs 0)) (.each N (fun i =>
    if 0 < i.val then step M (xs i.val) (state M xs i.val).1 (state M xs i.val).2 else .top))

@[simp] theorem iterations_holds {α : Type} (M : Algebra α) (D : Domain α) (xs : Nat → α) (N : Nat) :
    (iterations M xs N).Holds D ↔ IterationDomain M D xs N := by
  simp only [iterations, Requirements.holds_both, initial_holds, Requirements.holds_each]
  constructor
  · rintro ⟨hi, hs⟩
    refine ⟨hi, fun i hpos hn => ?_⟩
    simpa only [if_pos hpos, step_holds] using hs ⟨i, hn⟩
  · intro h
    refine ⟨h.initial, fun i => ?_⟩
    split
    · exact (step_holds M D _ _ _).mpr (h.steps i.val ‹_› i.isLt)
    · trivial

@[simp] theorem map_iterations {α Input : Type} (M : Algebra α) (read : Input → α)
    (xs : Nat → Expr Input) (N : Nat) :
    (iterations (GuardExpression.algebra Input) xs N).map (Expr.eval M read) =
      iterations M (fun i => (xs i).eval M read) N := by
  simp only [iterations, Requirements.map, map_initial]
  congr 2
  funext i
  split
  · rw [map_step]
    have h := eval_state M read xs i.val
    rw [← h]
  · rfl

end VeriTile.Triton.FP.OnlineSoftmaxConditions
