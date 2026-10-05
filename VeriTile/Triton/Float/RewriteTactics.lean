import VeriTile.Meta.FPProve
import VeriTile.Triton.Float.ProfiledRewrite

/-! Optional semantic adapter for contextual program rewrites. Decomposition
only compares source statements and retains their prefixes. Numerical steps
are discharged separately by fp_prove using checked contextual lemmas. -/
namespace VeriTile.Triton.FP.RewriteTactics
open Lean Meta Elab Tactic
open VeriTile.Meta.EquivDecompose

private def domainFacts (goal : MVarId) (domain hypothesis : Expr) : MetaM MVarId := goal.withContext do
  let some sites ← listItems? domain | return goal
  let mut goal := goal
  for i in [:sites.size] do
    goal ← goal.withContext do
      let applied := mkApp hypothesis sites[i]!
      let t ← whnf (← inferType applied)
      let membership ← mkFreshExprSyntheticOpaqueMVar t.bindingDomain!
      let facts ← #[``List.mem_cons_self, ``List.mem_cons_of_mem].mapM mkConstWithFreshMVarLevels
      unless ← VeriTile.Meta.FPProve.solveUsing membership.mvarId! facts (sites.size + 2) do
        throwError "could not retain the declared domain at rewrite site {i + 1}"
      let proof := mkApp applied membership
      let next ← goal.assert (Name.mkSimple s!"site_{i + 1}") (← inferType proof) proof
      return (← next.intro1P).2
  return goal

private def differences (goal : MVarId) (model state lhs rhs : Expr) : MetaM (List MVarId) := goal.withContext do
  let some left ← listItems? lhs | throwError "the source statement list is symbolic"
  let some right ← listItems? rhs | throwError "the target statement list is symbolic"
  unless left.size == right.size do
    throwError "contextual decomposition requires aligned statement lists; insertion/deletion needs an execution lemma"
  let statement := mkConst ``ComputeStmt
  let mut current := left
  let mut proof : Option Expr := none
  let mut remaining := []
  for i in [:left.size] do
    if ← same current[i]! right[i]! then continue
    let before ← mkListLit statement (current.extract 0 i).toList
    let after ← mkListLit statement (current.extract (i + 1) current.size).toList
    let target ← mkAppM ``ProfiledRewrite.Contextual #[model, state, before, after, current[i]!, right[i]!]
    let hole ← mkFreshExprSyntheticOpaqueMVar target
      (tag := (← goal.getTag).appendAfter s!"diff_{remaining.length + 1}")
    remaining := remaining ++ [hole.mvarId!]
    proof ← match proof with
      | none => pure (some hole)
      | some previous => pure (some (← mkAppM ``Eq.trans #[previous, hole]))
    current := current.set! i right[i]!
  match proof with
  | none => goal.refl
  | some complete => goal.assign complete
  return remaining

private def decompose (goal : MVarId) : MetaM (List MVarId) := goal.withContext do
  let target ← whnf (← goal.getType)
  let lhs := target.getAppArgs[3]!
  let rhs := target.getAppArgs[4]!
  let left ← mkAppM ``ComputeKernel.surfaceBody #[← mkAppM ``ProfiledRewrite.Program.kernel #[lhs]]
  let right ← mkAppM ``ComputeKernel.surfaceBody #[← mkAppM ``ProfiledRewrite.Program.kernel #[rhs]]
  let domain ← mkAppM ``ProfiledRewrite.Program.domain #[lhs]
  let children ← goal.apply (← mkConstWithFreshMVarLevels ``Spec.FloatingPoint.ofNumerical)
  for child in children do
    unless ← child.isAssigned do discard <| trivial? child
  let mut remaining := []
  for child in children do
    if ← child.isAssigned then continue
    let t ← whnf (← child.getType)
    if !t.isForall then
      remaining := remaining ++ [child]
      continue
    let (vars, body) ← child.introNP 7
    let body ← domainFacts body domain (mkFVar vars[6]!)
    remaining := remaining ++ (← differences body (mkFVar vars[2]!) (mkFVar vars[5]!) left right)
  return remaining

elab_rules : tactic
  | `(tactic| equiv_decompose) => do
    let goal ← getMainGoal
    goal.withContext do
      let target ← whnf (← goal.getType)
      unless target.isAppOfArity ``Spec.FloatingPoint 5 &&
          (← same target.getAppArgs[0]! (mkConst ``ProfiledRewrite.Program)) do
        throwUnsupportedSyntax
      liftMetaTactic decompose

end VeriTile.Triton.FP.RewriteTactics
