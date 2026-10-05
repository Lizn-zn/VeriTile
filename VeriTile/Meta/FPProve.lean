import VeriTile.Meta.EquivDecompose
import VeriTile.Meta.Specification

/-!
`fp_prove` builds proof terms from local hypotheses, explicit lemma hints and
`@[spec_rule]` lemmas. Syntax derivations additionally support bounded search
through symmetry, transitivity and common statement contexts. No report data
is turned into an admission proof, and no real arithmetic tactic is invoked.
-/

namespace VeriTile.Meta.FPProve
open Lean Meta Elab Tactic
open EquivDecompose

/-- Imported and local registered rules, in deterministic name order. -/
def registeredRules : MetaM (Array Expr) := do
  let env ← getEnv
  let mut names := (specRuleAttr.ext.getState env).toArray
  for i in [:env.header.moduleNames.size] do
    names := names ++ specRuleAttr.ext.getModuleEntries env i
  names := names.qsort Name.quickLt
  names.mapM mkConstWithFreshMVarLevels

/-- Try a lemma and solve all its premises; failed attempts restore unification.
Admission premises are supplied by hypotheses or lemmas, never manufactured.
The depth bound also makes cyclic user-supplied lemmas terminate. -/
partial def solveUsing (goal : MVarId) (facts : Array Expr) (fuel : Nat) : MetaM Bool :=
  goal.withContext do
    if ← goal.isAssigned then return true
    let saved ← saveState
    try
      goal.assumption
      return true
    catch _ => saved.restore
    if ← trivial? goal then return true
    if fuel == 0 then return false
    for fact in facts do
      let saved ← saveState
      try
        let children ← goal.apply fact
        let mut solved := true
        for child in children do
          if !(← solveUsing child facts (fuel - 1)) then
            solved := false
            break
        if solved then return true
      catch _ => pure ()
      saved.restore
    return false

/-- Instantiate one directed rule at a concrete fragment. An unconstrained
intermediate is not a search node: all its parameters must be resolved. -/
def step? (statement rules code fact : Expr) (backwards : Bool)
    (facts : Array Expr) : MetaM (Option (Expr × Expr)) := do
  let saved ← saveState
  let result ← try
    let next ← mkFreshExprMVar (← mkAppM ``List #[statement])
    let target ← mkAppM ``Spec.Derivation
      (if backwards then #[rules, next, code] else #[rules, code, next])
    let hole ← mkFreshExprSyntheticOpaqueMVar target
    let children ← hole.mvarId!.apply fact
    let mut solved := true
    for child in children do
      if !(← solveUsing child facts 6) then
        solved := false
        break
    let next ← instantiateMVars next
    let proof ← instantiateMVars hole
    if !solved || next.hasMVar || proof.hasMVar then pure none
    else
      let proof ← if backwards then mkAppM ``Spec.Derivation.symm #[proof] else pure proof
      pure (some (next, proof))
  catch _ => pure none
  saved.restore
  return result

structure SearchNode where
  code : Expr
  proof : Expr
  depth : Nat
  deriving Inhabited

/-- Breadth-first search through concrete fragment rewrites, including rules
inside a larger fragment. Search is bounded; failure does not assert inequivalence. -/
def search (goal : MVarId) (facts : Array Expr) (maxSteps : Nat) : MetaM Bool := goal.withContext do
  let target ← whnf (← goal.getType)
  unless target.isAppOfArity ``Spec.Derivation 4 do return false
  let #[statement, rules, lhs, rhs] := target.getAppArgs | return false
  let proof ← mkAppOptM ``Spec.Derivation.refl #[some statement, some rules, some lhs]
  let mut queue : Array SearchNode := #[⟨lhs, proof, 0⟩]
  let mut cursor := 0
  while cursor < queue.size && cursor < 256 do
    let node := queue[cursor]!
    cursor := cursor + 1
    if ← same node.code rhs then
      goal.assign node.proof
      return true
    if node.depth >= maxSteps then continue
    -- Try the entire fragment first, then proper nonempty subfragments.
    let mut slices : Array (Expr × Expr × Expr) := #[]
    let nil ← mkListLit statement []
    slices := slices.push (nil, node.code, nil)
    if let some items ← listItems? node.code then
      for start in [:items.size] do
        for stop in [start + 1:items.size + 1] do
          if start == 0 && stop == items.size then continue
          slices := slices.push (← mkListLit statement (items.extract 0 start).toList,
            ← mkListLit statement (items.extract start stop).toList,
            ← mkListLit statement (items.extract stop items.size).toList)
    for (before, fragment, after) in slices do
      for fact in facts do
        for backwards in [false, true] do
          if let some (next, edge) ← step? statement rules fragment fact backwards facts then
            let next ← mkAppM ``List.append #[before, (← mkAppM ``List.append #[next, after])]
            if ← queue.anyM (fun old => same old.code next) then continue
            let edge ← mkAppM ``Spec.Derivation.frame #[before, after, edge]
            let proof ← mkAppM ``Spec.Derivation.trans #[node.proof, edge]
            if ← same next rhs then
              goal.assign proof
              return true
            if queue.size < 256 then queue := queue.push ⟨next, proof, node.depth + 1⟩
  return false

/-- Prove using registered rules and optional explicit hints. The step bound
controls the number of atomic edges, independently of premise-search depth. -/
syntax (name := fpProve) "fp_prove" (" (" &"maxSteps" " := " num ")")?
  (" [" term,* "]")? : tactic

elab_rules : tactic
  | `(tactic| fp_prove $[ (maxSteps := $steps:num)]? $[ [$hints:term,*]]?) => do
    let goal ← getMainGoal
    goal.withContext do
      let mut facts := #[]
      if let some hints := hints then
        for hint in hints.getElems do facts := facts.push (← elabTerm hint none)
      facts := facts ++ (← registeredRules)
      for localDecl in ← getLCtx do
        unless localDecl.isImplementationDetail do
          facts := facts.push localDecl.toExpr
      let saved ← saveState
      if (← solveUsing goal facts 6) || (← search goal facts (steps.map (·.getNat) |>.getD 8)) then
        replaceMainGoal []
      else
        saved.restore
        throwTacticEx `fp_prove goal "fp_prove could not close this goal using the available hypotheses and @[spec_rule] lemmas.\nA required atomic rule, domain premise or structural lemma may be missing, or the search bound was reached.\nUse an explicit lemma hint, increase (maxSteps := ...), or prove the remaining goal manually."

end VeriTile.Meta.FPProve
