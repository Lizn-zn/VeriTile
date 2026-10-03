/- Induction for the original Welford arithmetic recurrence. The conversion
of the natural loop index remains explicit, with its two scalar conversion laws
named as premises. No count law, reduction identity or kernel equivalence is
admitted by this module. -/
import VeriTile.Triton.Float.WelfordInit

namespace VeriTile.Triton.FP.WelfordInduction
open Structural Guarded ScalarArithmetic ScalarReduction WelfordReduction
open Equational (ReductionTree ReductionPlan)

/-- Append each lane in loop order, preserving the chosen empty padding tree. -/
def tree (empty : ReductionTree 0) : (n : Nat) → ReductionTree n
  | 0 => empty
  | n + 1 => WelfordAppend.appendTree (tree empty n)

def plan (empty : ReductionPlan 0) : (n : Nat) → ReductionPlan n
  | 0 => empty
  | n + 1 => WelfordAppend.appendPlan (plan empty n)

theorem plan_tree (empty : ReductionPlan 0) (n : Nat) :
    (plan empty n).tree = tree empty.tree n := by
  induction n with
  | zero => rfl
  | succ n ih => exact congrArg WelfordAppend.appendTree ih

def rowPrefix {α : Type} (xs : Nat → α) (n : Nat) : Fin n → α := fun i => xs i.val

theorem rowPrefix_succ {α : Type} (xs : Nat → α) (n : Nat) :
    rowPrefix xs (n + 1) = Fin.snoc (rowPrefix xs n) (xs n) := by
  funext i
  refine Fin.lastCases ?_ (fun i => ?_) i
  · simp [rowPrefix]
  · simp [rowPrefix]

def statistics {α : Type} (M : Algebra α) (xs : Fin n → α) (t : ReductionTree n) : α × α :=
  (mean M xs t, value M (Welford.deviationSquares M xs (mean M xs t)) (zero M) t)

/-- The arithmetic of the original loop, including fromNat at fp32 precision. -/
def update {α : Type} (M : Algebra α) (i : Nat) (x : α) (acc : α × α) : α × α :=
  (Welford.nextMean M x acc.1 (M.fromNat (some .fp32) i),
    add M acc.2 (Welford.increment M x acc.1 (M.fromNat (some .fp32) i)))

def state {α : Type} (M : Algebra α) (xs : Nat → α) : Nat → α × α
  | 0 => (zero M, zero M)
  | n + 1 => update M n (xs n) (state M xs n)

/-- Primitive conversion obligations, bounded by the actual row length.
Float/CountConversion derives this record from the accepted bounded atoms.
It has no default instance and is not a whole-row equality. -/
structure CountConversion {α : Type} (M : Algebra α) (N : Nat) : Prop where
  zero : M.fromNat (some .fp32) 0 = ScalarArithmetic.zero M
  successor : ∀ i, i < N → M.fromNat (some .fp32) (i + 1) =
    add M (M.fromNat (some .fp32) i) (one M)

theorem CountConversion.restrict {α : Type} {M : Algebra α} {N k : Nat}
    (h : CountConversion M N) (hk : k ≤ N) : CountConversion M k :=
  ⟨h.zero, fun i hi => h.successor i (by omega)⟩

/-- Domains for every arithmetic rewrite in the induction, not just the last
iteration. All fields contain finite/nonzero predicates, never invariants. -/
structure IterationDomain {α : Type} (M : Algebra α) (D : Domain α)
    (xs : Nat → α) (empty : ReductionTree 0) (N : Nat) : Prop where
  initial : WelfordInit.InitDomain M D (xs 0)
  steps : ∀ i, 0 < i → i < N →
    WelfordAppend.VarianceAppendDomain M D (rowPrefix xs i) (tree empty i) (xs i)

theorem IterationDomain.restrict {α : Type} {M : Algebra α} {D : Domain α}
    {xs : Nat → α} {empty : ReductionTree 0} {N k : Nat}
    (h : IterationDomain M D xs empty N) (hk : k ≤ N) : IterationDomain M D xs empty k :=
  ⟨h.initial, fun i hi hn => h.steps i hi (by omega)⟩

section Induction
variable {α : Type} [Inhabited α] (R : Rules) (M : Algebra α) (D : Domain α)
  (hM : Models R.assumptions M D) (s : State α)
include R hM s

/-- Convert an integer count to the prefix tree's count using exactly the
zero and successor obligations. No reduction-count equality is a premise. -/
theorem converted_count (empty : ReductionTree 0) (N : Nat)
    (hc : CountConversion M N) (hz : D .finite (zero M)) (i : Nat) (hi : i ≤ N) :
    M.fromNat (some .fp32) i = count M (tree empty i) := by
  induction i with
  | zero =>
    exact hc.zero.trans (value_empty R M D hM s _ empty hz).symm
  | succ i ih =>
    rw [hc.successor i (by omega), ih (by omega), tree, WelfordAppend.count_append]
    rfl

/-- Both statistics of the converted-index recurrence equal those of the
explicit prefix tree. Every iteration is covered. This is conditional on the
two primitive conversion laws; it is not a completed kernel specification. -/
theorem state_statistics (xs : Nat → α) (empty : ReductionTree 0) (N : Nat)
    (hN : 0 < N) (hc : CountConversion M N) (hd : IterationDomain M D xs empty N) :
    state M xs N = statistics M (rowPrefix xs N) (tree empty N) := by
  induction N with
  | zero => omega
  | succ n ih =>
    cases n with
    | zero =>
      have hp : rowPrefix xs 1 = fun _ => xs 0 := by
        funext i
        simp [rowPrefix]
      change update M 0 (xs 0) (zero M, zero M) = _
      unfold update
      rw [hc.zero, hp]
      exact WelfordInit.initial_statistics R M D hM s (xs 0) hd.initial empty
    | succ n =>
      have hs := hd.steps (n + 1) (by omega) (by omega)
      have hprev := ih (by omega) (hc.restrict (by omega)) (hd.restrict (by omega))
      rw [state, hprev]
      unfold update
      rw [converted_count R M D hM s empty (n + 2) hc hd.initial.zero (n + 1) (by omega)]
      change (Welford.nextMean M (xs (n + 1)) _ _, add M _ (Welford.increment M _ _ _)) = _
      rw [rowPrefix_succ xs (n + 1)]
      exact Prod.ext
        (WelfordAppend.mean_append R M D hM s _ _ _
          ⟨hs.recenter.centering, hs.step.toMeanStepDomain, hs.newSum, hs.newMean⟩)
        (WelfordAppend.variance_append R M D hM s _ _ _ hs)

/-- Division by the actual converted row length is preserved on the online
side and bound to the explicit tree count on the batch side. -/
theorem normalized_statistics (xs : Nat → α) (empty : ReductionTree 0) (N : Nat)
    (hN : 0 < N) (hc : CountConversion M N) (hd : IterationDomain M D xs empty N) :
    ((state M xs N).1, div M (state M xs N).2 (M.fromNat (some .fp32) N)) =
      (mean M (rowPrefix xs N) (tree empty N),
        div M (statistics M (rowPrefix xs N) (tree empty N)).2 (count M (tree empty N))) := by
  rw [state_statistics R M D hM s xs empty N hN hc hd,
    converted_count R M D hM s empty N hc hd.initial.zero N le_rfl]
  rfl

end Induction
end VeriTile.Triton.FP.WelfordInduction
