/- Numerical call trees for the FP equational theory. Sum reduction expands
an explicit, arbitrary valid addition schedule; its padding is retained. -/
import VeriTile.Triton.Float.Equational
import Mathlib.Data.List.FinRange

namespace VeriTile.Triton.FP.Equational

inductive ReductionTree (n : Nat) where
  | input (lane : Fin n)
  | zero
  | add (left right : ReductionTree n)

def ReductionTree.leaves : ReductionTree n → List (Option (Fin n))
  | .input i => [some i]
  | .zero => [none]
  | .add a b => a.leaves ++ b.leaves

def ReductionTree.expand (xs : Fin n → Term α) (zero : Term α) : ReductionTree n → SumTree α
  | .input i => .leaf (xs i)
  | .zero => .leaf zero
  | .add a b => .join (a.expand xs zero) (b.expand xs zero)

theorem ReductionTree.expand_leaves (xs : Fin n → Term α) (zero : Term α) (t : ReductionTree n) :
    (t.expand xs zero).leaves = t.leaves.map (fun i => i.elim zero xs) := by
  induction t with
  | input i => rfl
  | zero => rfl
  | add a b ih₁ ih₂ => simp [expand, SumTree.leaves, leaves, ih₁, ih₂]

/-- Padding is explicit. Validity records each input exactly once and preserves
the padding count; it asserts no numerical identity about the zero leaf. -/
structure ReductionPlan (n : Nat) where
  tree : ReductionTree n
  padding : Nat
  valid : tree.leaves.Perm ((List.finRange n).map some ++ List.replicate padding none)

def ReductionTree.ofList (xs : List (Fin n)) : ReductionTree n :=
  xs.foldr (fun i rest => .add (.input i) rest) .zero

theorem ReductionTree.ofList_leaves (xs : List (Fin n)) :
    (ReductionTree.ofList xs).leaves = xs.map some ++ [none] := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    change some x :: (ofList xs).leaves = some x :: (xs.map some ++ [none])
    rw [ih]

/-- An explicit witness exists for every length, including zero. Its single
zero leaf is retained; this is a valid example schedule, not a required one. -/
def ReductionPlan.withSeed (n : Nat) : ReductionPlan n where
  tree := ReductionTree.ofList (List.finRange n)
  padding := 1
  valid := by rw [ReductionTree.ofList_leaves]; exact .refl _

theorem ReductionPlan.expanded_leaves (plan : ReductionPlan n)
    (xs : Fin n → Term α) (zero : Term α) :
    (plan.tree.expand xs zero).leaves.Perm
      ((List.finRange n).map xs ++ List.replicate plan.padding zero) := by
  rw [ReductionTree.expand_leaves]
  simpa only [List.map_append, List.map_map, Function.comp_def, Option.elim_some,
    List.map_replicate, Option.elim_none] using
      plan.valid.map (fun i => i.elim zero xs)

/-- The input permutation changes neither multiplicity nor explicit padding. -/
theorem ReductionPlan.reorder {R : Spec.Assumptions ComputeStmt}
    (hc : Spec.Derivation R commuteLHS commuteRHS)
    (ha : Spec.Derivation R associateLHS associateRHS)
    (plan : ReductionPlan n) (xs : Fin n → Term α) (zero : Term α) (σ : Equiv.Perm (Fin n)) :
    TermEq R (plan.tree.expand xs zero).eval (plan.tree.expand (xs ∘ σ) zero).eval := by
  apply SumTree.equiv_of_perm hc ha
  apply (plan.expanded_leaves xs zero).trans
  apply List.Perm.trans _ (plan.expanded_leaves (xs ∘ σ) zero).symm
  apply List.Perm.append_right
  simpa only [List.map_map, Function.comp_def] using (σ.map_finRange_perm.map xs).symm

def SumTree.evalAt (precision : Option ComputeDType) : SumTree α → Term α
  | .leaf a => a
  | .join a b => .app (.binary precision .real .add) [a.evalAt precision, b.evalAt precision]

theorem SumTree.evalAt_fp32 (t : SumTree α) : t.evalAt (some .fp32) = t.eval := by
  induction t with
  | leaf => rfl
  | join a b ih₁ ih₂ => simp [evalAt, eval, Term.add, ih₁, ih₂]

/-- The schedule can depend on precision, full layout, axis, keepDims and
output position, but never on numerical input values. -/
abbrev Schedules := (precision : Option ComputeDType) → (shape : TileShape) →
  (axis : Fin shape.length) → (keepDims : Bool) →
  TileIndex (TileShape.reduceShape shape axis keepDims) →
  ReductionPlan (TileShape.axisDim shape axis)

def seededSchedules : Schedules := fun _ shape axis _ _ => .withSeed (TileShape.axisDim shape axis)

def reductionInputs (shape : TileShape) (axis : Fin shape.length) (keepDims : Bool)
    (xs : Structural.Values (Term α) .real shape) :
    TileIndex (TileShape.reduceShape shape axis keepDims) →
      Fin (TileShape.axisDim shape axis) → Term α :=
  match keepDims with
  | false => fun i k => xs (TileShape.insertAxisIndex shape axis i k)
  | true => fun i k => xs (TileShape.replaceAxisIndex shape axis i k)

def algebra (plans : Schedules) : Structural.Algebra (Term α) where
  literal := fun p d x => .app (.literal p d x) []
  negInf := .app .negInf []
  binary := fun p d op a b => .app (.binary p d op) [a, b]
  unary := fun p op a => .app (.unary p op) [a]
  cast := fun p src dst a => .app (.cast p src dst) [a]
  fromNat := fun p n => .app (.fromNat p n) []
  fromInt := fun p n => .app (.fromInt p n) []
  fp32Bits := fun b => .app (.fp32Bits b) []
  fp32Load := fun a => .app .fp32Load [a]
  reduceMax := fun p {shape} axis keepDims xs i =>
    .app (.reduceMax p shape axis keepDims i) ((TileShape.allIndices shape).map xs)
  reduceSum := fun p {shape} axis keepDims xs i =>
    ((plans p shape axis keepDims i).tree.expand (reductionInputs shape axis keepDims xs i)
      (.app (.literal p .real 0) [])).evalAt p

end VeriTile.Triton.FP.Equational
