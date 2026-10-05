/- Compare arbitrary valid addition schedules by an explicit scalar rewrite
path. Its finite-value obligations are computed from that path, not inferred
from finite leaves or supplied as a whole-reduction equality. -/
import VeriTile.Triton.Float.ScalarReduction
import Mathlib.Data.List.Sort

namespace VeriTile.Triton.FP.ReductionSchedule
open Structural Guarded ScalarArithmetic ScalarReduction
open Equational (ReductionTree ReductionPlan)

/-- Syntactic steps only. Their numerical meaning is derived in `sound`. -/
inductive Rewrite : ReductionTree n → ReductionTree n → Type where
  | refl (a) : Rewrite a a
  | commute (a b) : Rewrite (.add a b) (.add b a)
  | associate (a b c) : Rewrite (.add (.add a b) c) (.add a (.add b c))
  | zeroRight (a) : Rewrite (.add a .zero) a
  | zeroLeft (a) : Rewrite (.add .zero a) a
  | congr {a b c d} : Rewrite a b → Rewrite c d → Rewrite (.add a c) (.add b d)
  | symm {a b} : Rewrite a b → Rewrite b a
  | trans {a b c} : Rewrite a b → Rewrite b c → Rewrite a c

namespace Rewrite

/-- The exact operands of every scalar rewrite, including intermediate trees.
Congruence and transitivity add no numerical assumptions of their own. -/
def operands {a b : ReductionTree n} : Rewrite a b → List (ReductionTree n)
  | .refl _ => []
  | .commute a b => [a, b]
  | .associate a b c => [a, b, c]
  | .zeroRight a => [a]
  | .zeroLeft a => [.zero, a]
  | .congr p q => p.operands ++ q.operands
  | .symm p => p.operands
  | .trans p q => p.operands ++ q.operands

def Domain {α : Type} (M : Algebra α) (D : Guarded.Domain α) (xs : Fin n → α)
    {a b : ReductionTree n} (p : Rewrite a b) : Prop :=
  ∀ t ∈ p.operands, D .finite (value M xs (zero M) t)

theorem sound {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Guarded.Domain α) (hM : Models R.assumptions M D) (s : State α)
    (xs : Fin n → α) {a b : ReductionTree n} (p : Rewrite a b) (hd : p.Domain M D xs) :
    value M xs (zero M) a = value M xs (zero M) b := by
  induction p with
  | refl => rfl
  | commute a b =>
    exact add_comm R M D hM s _ _ (hd a (by simp [operands])) (hd b (by simp [operands]))
  | associate a b c =>
    exact add_assoc R M D hM s _ _ _ (hd a (by simp [operands]))
      (hd b (by simp [operands])) (hd c (by simp [operands]))
  | zeroRight a => exact add_zero R M D hM s _ (hd a (by simp [operands]))
  | zeroLeft a =>
    exact zero_add R M D hM s _ (hd a (by simp [operands])) (hd .zero (by simp [operands]))
  | congr p q ihp ihq =>
    exact congrArg₂ (add M)
      (ihp (fun t ht => hd t (List.mem_append_left _ ht)))
      (ihq (fun t ht => hd t (List.mem_append_right _ ht)))
  | symm p ih => exact (ih hd).symm
  | trans p q ihp ihq =>
    exact (ihp (fun t ht => hd t (List.mem_append_left _ ht))).trans
      (ihq (fun t ht => hd t (List.mem_append_right _ ht)))

end Rewrite

/-- Padding is retained in the original tree. Omitting it from this lane list
does not remove it numerically: normalization emits zero-law steps for it. -/
def lanes : ReductionTree n → List (Fin n)
  | .input i => [i]
  | .zero => []
  | .add a b => lanes a ++ lanes b

theorem lanes_filter (t : ReductionTree n) : lanes t = t.leaves.filterMap id := by
  induction t with
  | input => rfl
  | zero => rfl
  | add a b ih₁ ih₂ => simp [lanes, ReductionTree.leaves, List.filterMap_append, ih₁, ih₂]

theorem plan_lanes (p : ReductionPlan n) : (lanes p.tree).Perm (List.finRange n) := by
  have h := p.valid.filterMap id
  simpa [lanes_filter, List.filterMap_append, List.filterMap_map, Function.comp_def] using h

def canonical (xs : List (Fin n)) : ReductionTree n := ReductionTree.ofList xs

def appendRewrite (xs ys : List (Fin n)) :
    Rewrite (canonical (xs ++ ys)) (.add (canonical xs) (canonical ys)) :=
  match xs with
  | [] => (Rewrite.zeroLeft (canonical ys)).symm
  | x :: xs =>
    (Rewrite.congr (.refl (.input x)) (appendRewrite xs ys)).trans
      (Rewrite.associate (.input x) (canonical xs) (canonical ys)).symm

/-- Flatten the tree without changing lane order. Every padding removal and
every inserted trailing zero is justified by an explicit identity step. -/
def flatten (t : ReductionTree n) : Rewrite t (canonical (lanes t)) :=
  match t with
  | .input i => (Rewrite.zeroRight (.input i)).symm
  | .zero => .refl .zero
  | .add a b => (Rewrite.congr (flatten a) (flatten b)).trans (appendRewrite (lanes a) (lanes b)).symm

def insertRewrite (x : Fin n) (xs : List (Fin n)) :
    Rewrite (.add (.input x) (canonical xs)) (canonical (xs.orderedInsert (· ≤ ·) x)) :=
  match xs with
  | [] => .refl _
  | y :: ys =>
    if h : x ≤ y then by
      simpa only [List.orderedInsert_cons, if_pos h] using
        (Rewrite.refl (.add (.input x) (canonical (y :: ys))))
    else by
      rw [List.orderedInsert_cons, if_neg h]
      exact (Rewrite.associate (.input x) (.input y) (canonical ys)).symm.trans
        ((Rewrite.congr (.commute (.input x) (.input y)) (.refl (canonical ys))).trans
          ((Rewrite.associate (.input y) (.input x) (canonical ys)).trans
            (.congr (.refl (.input y)) (insertRewrite x ys))))

def sortRewrite (xs : List (Fin n)) :
    Rewrite (canonical xs) (canonical (xs.insertionSort (· ≤ ·))) :=
  match xs with
  | [] => .refl _
  | x :: xs =>
    (Rewrite.congr (.refl (.input x)) (sortRewrite xs)).trans (insertRewrite x (xs.insertionSort (· ≤ ·)))

/-- A deterministic path, computable from the schedule without floating data. -/
def normalize (t : ReductionTree n) :
    Rewrite t (canonical ((lanes t).insertionSort (· ≤ ·))) :=
  (flatten t).trans (sortRewrite (lanes t))

theorem sorted_lanes (p : ReductionPlan n) :
    (lanes p.tree).insertionSort (· ≤ ·) = (List.finRange n).insertionSort (· ≤ ·) := by
  apply List.Perm.eq_of_pairwise' (List.pairwise_insertionSort _ _) (List.pairwise_insertionSort _ _)
  exact (List.perm_insertionSort _ _).trans ((plan_lanes p).trans (List.perm_insertionSort _ _).symm)

/-- Conditions for the two concrete normalization paths. This quantifies over
their listed operands only, not over every conceivable intermediate tree. -/
structure ScheduleDomain {α : Type} (M : Algebra α) (D : Guarded.Domain α)
    (xs : Fin n → α) (a b : ReductionPlan n) : Prop where
  left : (normalize a.tree).Domain M D xs
  right : (normalize b.tree).Domain M D xs

/-- Arbitrary valid schedules, including different padding counts, agree
under the admitted scalar laws and the domains of the constructed rewrites. -/
theorem plans_value {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Guarded.Domain α) (hM : Models R.assumptions M D) (s : State α)
    (xs : Fin n → α) (a b : ReductionPlan n) (hd : ScheduleDomain M D xs a b) :
    value M xs (zero M) a.tree = value M xs (zero M) b.tree := by
  have ha := (normalize a.tree).sound R M D hM s xs hd.left
  have hb := (normalize b.tree).sound R M D hM s xs hd.right
  rw [sorted_lanes a] at ha
  rw [sorted_lanes b] at hb
  exact ha.trans hb.symm

/-- Rename input lanes without changing the tree or its explicit padding. -/
def reindexTree (σ : Equiv.Perm (Fin n)) : ReductionTree n → ReductionTree n
  | .input i => .input (σ i)
  | .zero => .zero
  | .add a b => .add (reindexTree σ a) (reindexTree σ b)

theorem reindexTree_leaves (σ : Equiv.Perm (Fin n)) (t : ReductionTree n) :
    (reindexTree σ t).leaves = t.leaves.map (Option.map σ) := by
  induction t with
  | input => rfl
  | zero => rfl
  | add a b ha hb => simp [reindexTree, ReductionTree.leaves, ha, hb]

def reindex (p : ReductionPlan n) (σ : Equiv.Perm (Fin n)) : ReductionPlan n where
  tree := reindexTree σ p.tree
  padding := p.padding
  valid := by
    rw [reindexTree_leaves]
    apply (p.valid.map (Option.map σ)).trans
    simpa [List.map_append, List.map_map, Function.comp_def, List.map_replicate] using
      (σ.map_finRange_perm.map some).append_right (List.replicate p.padding none)

theorem value_reindex {α : Type} (M : Algebra α) (xs : Fin n → α) (seed : α)
    (σ : Equiv.Perm (Fin n)) (t : ReductionTree n) :
    value M xs seed (reindexTree σ t) = value M (xs ∘ σ) seed t := by
  induction t with
  | input => rfl
  | zero => rfl
  | add a b ha hb => simp [reindexTree, value, ha, hb]

end VeriTile.Triton.FP.ReductionSchedule
