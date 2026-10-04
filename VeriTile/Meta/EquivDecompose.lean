import Lean
import VeriTile.Spec

/-!
`equiv_decompose` exposes local differences in sequential program equivalence.
It uses only definitional equality and the existing syntax composition rules;
it does not inspect numerical atoms. Signature and context obligations remain
visible when they are not definitionally true. Changed loops, branches, casts
and memory operations remain whole statements, without an assumed congruence.
-/

namespace VeriTile.Spec.Derivation

/-- Compose adjacent fragment derivations in the same assumption context. -/
theorem append_congr {Statement : Type u} {R : Assumptions Statement}
    {a b c d : List Statement} (left : Derivation R a b)
    (right : Derivation R c d) : Derivation R (a ++ c) (b ++ d) := by
  apply Derivation.trans (middle := b ++ c)
  · simpa using Derivation.frame [] c left
  · simpa using Derivation.frame b [] right

end VeriTile.Spec.Derivation

namespace VeriTile.Meta.EquivDecompose
open Lean Meta Elab Tactic

/-- Compare without assigning any metavariables belonging to the user's goal. -/
def same (a b : Expr) : MetaM Bool :=
  withNewMCtxDepth <| isDefEq a b

/-- Read the list spine, retaining statements themselves as typed expressions. -/
partial def listItems? (e : Expr) : MetaM (Option (Array Expr)) := do
  let e ← whnf e
  if e.isAppOfArity ``List.nil 1 then return some #[]
  if e.isAppOfArity ``List.cons 3 then
    let some tail ← listItems? e.getAppArgs[2]! | return none
    return some (#[e.getAppArgs[1]!] ++ tail)
  return none

/-- Longest common subsequence, used only to choose a proof decomposition.
Every resulting concatenation is checked again by Lean's type checker. -/
def anchors (left right : Array Expr) : MetaM (Array (Nat × Nat)) := do
  let n := left.size
  let m := right.size
  let mut equalPairs := Array.replicate (n * m) false
  for i in [:n] do
    for j in [:m] do
      equalPairs := equalPairs.set! (i * m + j) (← same left[i]! right[j]!)
  let width := m + 1
  let mut lengths := Array.replicate ((n + 1) * width) 0
  for ii in [:n] do
    let i := n - 1 - ii
    for jj in [:m] do
      let j := m - 1 - jj
      let length := if equalPairs[i * m + j]! then lengths[(i+1)*width+j+1]! + 1
        else max lengths[(i+1)*width+j]! lengths[i*width+j+1]!
      lengths := lengths.set! (i*width+j) length
  let mut i := 0
  let mut j := 0
  let mut result := #[]
  while i < n && j < m do
    if equalPairs[i*m+j]! then
      result := result.push (i, j)
      i := i + 1
      j := j + 1
    else if lengths[(i+1)*width+j]! >= lengths[i*width+j+1]! then
      i := i + 1
    else
      j := j + 1
  return result

def trivial? (goal : MVarId) : MetaM Bool := do
  let saved ← saveState
  try
    goal.refl
    return true
  catch _ =>
    saved.restore
    let target ← whnf (← goal.getType)
    if target.isConstOf ``True then
      goal.assign (mkConst ``True.intro)
      return true
    return false

/-- Split a syntax derivation; unsupported or symbolic list spines stay goals. -/
def decomposeList (goal : MVarId) (splitChanges : Bool) : MetaM (List MVarId) := goal.withContext do
  let target ← whnf (← goal.getType)
  unless target.isAppOfArity ``Spec.Derivation 4 do return [goal]
  let #[statement, rules, lhs, rhs] := target.getAppArgs | return [goal]
  if ← same lhs rhs then
    goal.assign (← mkAppOptM ``Spec.Derivation.refl #[some statement, some rules, some lhs])
    return []
  let some left ← listItems? lhs | return [goal]
  let some right ← listItems? rhs | return [goal]
  let common ← anchors left right
  let mut chunks : Array (Array Expr × Array Expr) := #[]
  let mut i := 0
  let mut j := 0
  for (a, b) in common.push (left.size, right.size) do
    let xs := left.extract i a
    let ys := right.extract j b
    if splitChanges && xs.size == ys.size then
      for k in [:xs.size] do chunks := chunks.push (#[xs[k]!], #[ys[k]!])
    else if !xs.isEmpty || !ys.isEmpty then
      chunks := chunks.push (xs, ys)
    if a < left.size && b < right.size then
      chunks := chunks.push (#[left[a]!], #[right[b]!])
    i := a + 1
    j := b + 1
  let mut goals := []
  let mut proofs := #[]
  for (xs, ys) in chunks do
    let a ← mkListLit statement xs.toList
    let b ← mkListLit statement ys.toList
    if ← same a b then
      proofs := proofs.push (← mkAppOptM ``Spec.Derivation.refl #[some statement, some rules, some a])
    else
      let hole ← mkFreshExprSyntheticOpaqueMVar
        (← mkAppM ``Spec.Derivation #[rules, a, b])
        (tag := (← goal.getTag).appendAfter s!"diff_{goals.length + 1}")
      goals := goals ++ [hole.mvarId!]
      proofs := proofs.push hole
  let nil ← mkListLit statement []
  let mut proof ← mkAppOptM ``Spec.Derivation.refl #[some statement, some rules, some nil]
  for p in proofs.reverse do
    proof ← mkAppM ``Spec.Derivation.append_congr #[p, proof]
  goal.assign proof
  return goals

def decompose (goal : MVarId) (splitChanges : Bool) : MetaM (List MVarId) := goal.withContext do
  let target ← whnf (← goal.getType)
  if target.isAppOfArity ``Spec.Derivation 4 then
    return ← decomposeList goal splitChanges
  if target.isAppOfArity ``Spec.FloatingPoint 5 then
    let children ← goal.apply (← mkConstWithFreshMVarLevels ``Spec.FloatingPoint.ofDerivation)
    let mut remaining := []
    for child in children do
      if !(← trivial? child) then
        remaining := remaining ++ (← decomposeList child splitChanges)
    return remaining
  if target.isAppOfArity ``Spec.ProgramDerivation 6 then
    let children ← goal.apply (← mkConstWithFreshMVarLevels ``Spec.ProgramDerivation.syntax)
    let mut remaining := []
    for child in children do
      if !(← trivial? child) then
        remaining := remaining ++ (← decomposeList child splitChanges)
    return remaining
  throwError "equiv_decompose expects program equivalence or a syntax derivation"

/-- Expose changed fragments; `(split := false)` keeps adjacent changes together. -/
syntax (name := equivDecompose) "equiv_decompose" : tactic
syntax (name := equivDecomposeSplit) "equiv_decompose" " (" "split" " := "
  (&"true" <|> &"false") ")" : tactic

elab_rules : tactic
  | `(tactic| equiv_decompose) => liftMetaTactic fun goal => decompose goal true
  | `(tactic| equiv_decompose (split := true)) => liftMetaTactic fun goal => decompose goal true
  | `(tactic| equiv_decompose (split := false)) => liftMetaTactic fun goal => decompose goal false

end VeriTile.Meta.EquivDecompose
