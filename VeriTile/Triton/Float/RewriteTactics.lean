import VeriTile.Meta.FPProve
import VeriTile.Triton.Float.BranchRewrite

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

private def differences (goal : MVarId) (contextual : Name) (model state lhs rhs : Expr)
    (reverse : Bool := false) : MetaM (List MVarId) := goal.withContext do
  let some left ← listItems? lhs | throwError "the source statement list is symbolic"
  let some right ← listItems? rhs | throwError "the target statement list is symbolic"
  unless left.size == right.size do
    throwError "contextual decomposition requires aligned statement lists; insertion/deletion needs an execution lemma"
  let statement := mkConst ``ComputeStmt
  let mut current := left
  let mut proof : Option Expr := none
  let mut remaining := []
  for offset in [:left.size] do
    let i := if reverse then left.size - 1 - offset else offset
    if ← same current[i]! right[i]! then continue
    let before ← mkListLit statement (current.extract 0 i).toList
    let after ← mkListLit statement (current.extract (i + 1) current.size).toList
    let target ← mkAppM contextual #[model, state, before, after, current[i]!, right[i]!]
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

private def contextPlan (lhs rhs : Expr) : MetaM (Option (Name × Nat)) := do
  let left := lhs.getAppArgs
  let right := rhs.getAppArgs
  if lhs.isAppOfArity ``ComputeStmt.forRange 5 &&
      rhs.isAppOfArity ``ComputeStmt.forRange 5 then
    for i in [:4] do
      unless ← same left[i]! right[i]! do return none
    return some (``ProfiledRewrite.contextual_static_range, 4)
  if lhs.isAppOfArity ``ComputeStmt.forRangeDyn 5 &&
      rhs.isAppOfArity ``ComputeStmt.forRangeDyn 5 then
    for i in [:4] do
      unless ← same left[i]! right[i]! do return none
    return some (``ProfiledRewrite.contextual_range, 4)
  if lhs.isAppOfArity ``ComputeStmt.forRangeInt 5 &&
      rhs.isAppOfArity ``ComputeStmt.forRangeInt 5 then
    for i in [:4] do
      unless ← same left[i]! right[i]! do return none
    return some (``ProfiledRewrite.contextual_int_range, 4)
  if lhs.isAppOfArity ``ComputeStmt.ifThen 2 &&
      rhs.isAppOfArity ``ComputeStmt.ifThen 2 then
    unless ← same left[0]! right[0]! do return none
    return some (``ProfiledRewrite.contextual_if_then, 1)
  if lhs.isAppOfArity ``ComputeStmt.ifThenElse 3 &&
      rhs.isAppOfArity ``ComputeStmt.ifThenElse 3 then
    unless ← same left[0]! right[0]! do return none
    if ← same left[2]! right[2]! then
      return some (``ProfiledRewrite.contextual_then, 1)
    if ← same left[1]! right[1]! then
      return some (``ProfiledRewrite.contextual_else, 2)
  return none

/-- Descend through ranges and matching conditional arms only when the declared
site supplies that execution context's guard. Numerical obligations remain for
fp_prove; this traversal does not manufacture conditions or rewrite a test. -/
private partial def contexts (goal : MVarId) (reverse : Bool := false) : MetaM (List MVarId) := goal.withContext do
  let target ← instantiateMVars (← goal.getType)
  unless target.isAppOfArity ``ProfiledRewrite.Contextual 8 do return [goal]
  let args := target.getAppArgs
  let lhs ← whnf args[6]!
  let rhs ← whnf args[7]!
  let left := lhs.getAppArgs
  let right := rhs.getAppArgs
  let some (ruleName, bodyIndex) ← contextPlan lhs rhs | return [goal]
  let saved ← saveState
  try
    let children ← goal.apply (← mkConstWithFreshMVarLevels ruleName)
    -- Resolve the actual site first: it supplies the inner domain after the
    -- original prefix and only in reachable iterations/selected arms.
    for child in children do
      unless ← child.isAssigned do
        try child.assumption catch _ => pure ()
    let mut remaining := []
    for child in children do
      if ← child.isAssigned then continue
      let type ← whnf (← child.getType)
      unless type.isForall do throwError "execution-context domain was not supplied"
      let (vars, body) ← child.introNP 2
      let nested ← differences body ``ProfiledRewrite.Contextual args[2]!
        (mkFVar vars[0]!) left[bodyIndex]! right[bodyIndex]! reverse
      for difference in nested do remaining := remaining ++ (← contexts difference reverse)
    return remaining
  catch _ =>
    saved.restore
    return [goal]

private def decompose (kernelField domainField contextual : Name) (goal : MVarId)
    (reverse : Bool := false) : MetaM (List MVarId) := goal.withContext do
  let target ← whnf (← goal.getType)
  let lhs := target.getAppArgs[3]!
  let rhs := target.getAppArgs[4]!
  let left ← mkAppM ``ComputeKernel.surfaceBody #[← mkAppM kernelField #[lhs]]
  let right ← mkAppM ``ComputeKernel.surfaceBody #[← mkAppM kernelField #[rhs]]
  let domain ← mkAppM domainField #[lhs]
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
    let nested ← differences body contextual (mkFVar vars[2]!) (mkFVar vars[5]!) left right reverse
    for difference in nested do remaining := remaining ++ (← contexts difference reverse)
  return remaining

/-- Prove later differences first, so each numerical guard still refers to
the original program prefix. This changes proof composition order only;
source execution order, suffixes and all domain obligations are preserved. -/
syntax "equiv_decompose" " (" "reverse" " := " &"true" ")" : tactic

elab_rules : tactic
  | `(tactic| equiv_decompose (reverse := true)) => do
    let goal ← getMainGoal
    goal.withContext do
      let target ← whnf (← goal.getType)
      unless target.isAppOfArity ``Spec.FloatingPoint 5 do throwUnsupportedSyntax
      if ← same target.getAppArgs[0]! (mkConst ``ProfiledRewrite.Program) then
        liftMetaTactic fun goal => decompose ``ProfiledRewrite.Program.kernel
          ``ProfiledRewrite.Program.domain ``ProfiledRewrite.Contextual goal Bool.true
      else if ← same target.getAppArgs[0]! (mkConst ``GuardedRewrite.Program) then
        liftMetaTactic fun goal => decompose ``GuardedRewrite.Program.kernel
          ``GuardedRewrite.Program.domain ``GuardedRewrite.Contextual goal Bool.true
      else throwUnsupportedSyntax

elab_rules : tactic
  | `(tactic| equiv_decompose) => do
    let goal ← getMainGoal
    goal.withContext do
      let target ← whnf (← goal.getType)
      unless target.isAppOfArity ``Spec.FloatingPoint 5 do throwUnsupportedSyntax
      if ← same target.getAppArgs[0]! (mkConst ``ProfiledRewrite.Program) then
        liftMetaTactic (decompose ``ProfiledRewrite.Program.kernel
          ``ProfiledRewrite.Program.domain ``ProfiledRewrite.Contextual)
      else if ← same target.getAppArgs[0]! (mkConst ``GuardedRewrite.Program) then
        liftMetaTactic (decompose ``GuardedRewrite.Program.kernel
          ``GuardedRewrite.Program.domain ``GuardedRewrite.Contextual)
      else throwUnsupportedSyntax

end VeriTile.Triton.FP.RewriteTactics
