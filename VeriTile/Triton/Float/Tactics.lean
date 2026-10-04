import VeriTile.Meta.FPProve
import VeriTile.Triton.Float.Observation
import VeriTile.Triton.Float.ReductionAutomation
import Mathlib.Data.Fin.Rev

/-!
Optional numerical-model adapters for the generic equivalence tactics.
IO decomposition uses certified single-cell execution summaries; reduction
search uses scalar-derived permutation and opaque-call congruence lemmas.
No example names, complete kernel-equivalence lemmas or report data are used.
-/
namespace VeriTile.Triton.FP.Tactics
open Lean Meta Elab Tactic
open VeriTile.Meta.EquivDecompose
open Structural Equational

initialize executionAttr : TagAttribute ←
  registerTagAttribute `equiv_exec "A successful execution summary used by equiv_decompose's IO adapter."

def executionFacts : MetaM (Array Expr) := do
  let env ← getEnv
  let mut names := (executionAttr.ext.getState env).toArray
  for i in [:env.header.moduleNames.size] do
    names := names ++ executionAttr.ext.getModuleEntries env i
  (names.qsort Name.quickLt).mapM mkConstWithFreshMVarLevels

/-- Execution premises may be instantiated from the actual initial state.
Only hypotheses, reflexivity, introductions and certified summaries are used. -/
partial def execute (goal : MVarId) (facts : Array Expr) (fuel : Nat) : MetaM Bool :=
  goal.withContext do
    if ← goal.isAssigned then return true
    let saved ← saveState
    try
      goal.assumption
      return true
    catch _ => saved.restore
    if ← trivial? goal then return true
    if fuel == 0 then return false
    if (← whnf (← goal.getType)).isForall then
      let (_, body) ← goal.intros
      if ← execute body facts (fuel - 1) then return true
      saved.restore
    for fact in facts do
      let attempt ← saveState
      try
        let children ← goal.apply fact
        if ← children.allM (fun g => execute g facts (fuel - 1)) then return true
      catch _ => pure ()
      attempt.restore
    return false

def privateScratch (goal : MVarId) : MetaM (List MVarId) := goal.withContext do
  let children ← goal.apply (← mkConstWithFreshMVarLevels ``privateScratch_of_nil)
  children.filterM fun child => return !(← trivial? child)

/-- The bridge retains successful execution, both frames, the signature and
both output lengths. Only observed numerical values become FP proof goals. -/
def decomposeIO (goal : MVarId) : MetaM (List MVarId) := goal.withContext do
  let children ← goal.apply (← mkConstWithFreshMVarLevels ``Spec.FloatingPoint.ofNumerical)
  let mut remaining := []
  for child in children do
    if ← trivial? child then continue
    let target ← child.getType
    let target ← whnf target
    if !target.isAppOfArity ``And 2 then
      remaining := remaining ++ [child]
      continue
    let [leftPrivate, rest] ← child.apply (← mkConstWithFreshMVarLevels ``And.intro)
      | throwError "expected the numerical IO contract"
    let [rightPrivate, runs] ← rest.apply (← mkConstWithFreshMVarLevels ``And.intro)
      | throwError "expected both execution obligations"
    remaining := remaining ++ (← privateScratch leftPrivate) ++ (← privateScratch rightPrivate)
    let (_, body) ← runs.introNP 4
    let pending ← body.withContext do
      let pending ← body.apply (← mkConstWithFreshMVarLevels ``scalar_runs)
      let facts ← executionFacts
      -- Resolve result metavariables from the two executions before reducing
      -- their relation. No unspecified observation is allowed to disappear.
      for g in pending do
        if ← g.isAssigned then continue
        let t ← instantiateMVars (← g.getType)
        if t.isAppOfArity ``ScalarRun 6 then
          unless ← execute g facts 8 do
            throwTacticEx `equiv_decompose g
              "No applicable @[equiv_exec] summary proves this successful run and memory frame."
      let mut results := []
      for g in pending do
        if ← g.isAssigned then continue
        if ← trivial? g then continue
        let saved ← saveState
        try
          let values ← g.apply (← mkConstWithFreshMVarLevels ``cellRelated_mk)
          results := results ++ values
        catch _ =>
          saved.restore
          results := results ++ [g]
      return results
    remaining := remaining ++ pending
  return remaining

-- This elaborator is optional: importing the generic tactic alone does not
-- load an FP model. All other program views retain the original tactic.
elab_rules : tactic
  | `(tactic| equiv_decompose) => do
    let goal ← getMainGoal
    goal.withContext do
      let target ← whnf (← goal.getType)
      unless target.isAppOfArity ``Spec.FloatingPoint 5 &&
          (← same target.getAppArgs[0]! (mkConst ``KernelIO₁)) do
        throwUnsupportedSyntax
      liftMetaTactic decomposeIO

/-- Candidate permutations are typed equivalences, not arbitrary index maps.
The actual input-vector equation is checked separately for every candidate. -/
def permutations (goal : MVarId) (facts : Array Expr) : MetaM (Array Expr) := goal.withContext do
  let t ← whnf (← goal.getType)
  unless t.isAppOfArity ``Equiv 2 do return #[]
  let source := t.getAppArgs[0]!
  let mut result := #[]
  for fact in facts do
    if ← same (← inferType fact) t then
      result := result.push fact
      result := result.push (← mkAppM ``Equiv.symm #[fact])
  if ← same source t.getAppArgs[1]! then
    result := result.push (← mkAppM ``Equiv.refl #[source])
    let source ← whnf source
    if source.isAppOfArity ``Fin 1 then
      result := result.push (← mkAppOptM ``Fin.revPerm #[some source.getAppArgs[0]!])
  return result

/-- Expose an actual one-dimensional reduction through definitional wrappers.
The final definitional-equality check validates every extracted field, including
precision, axis, keepDims, output index and the numerical interpreter. -/
def asSum (value : Expr) : MetaM Expr := do
  let value ← instantiateMVars value
  let mut current := value
  for _ in [:32] do
    if current.isAppOfArity ``Algebra.reduceSum 8 then
      let args := current.getAppArgs
      let model := args[1]!
      if model.isAppOfArity ``algebra 2 then
        if let some shape ← listItems? args[3]! then
          if shape.size == 1 then
            let inputs ← mkAppM ``rowInputs #[args[6]!]
            let result ← mkAppM ``fp32Sum #[model.getAppArgs[1]!, shape[0]!, inputs]
            if ← same result value then return result
      return value
    let some next ← unfoldDefinition? current | return value
    current := next
  return value

def reorder? (goal : MVarId) (facts : Array Expr) : MetaM Bool := goal.withContext do
  let original ← saveState
  try
    let t ← whnf (← goal.getType)
    let args := t.getAppArgs
    let lhs ← asSum args[2]!
    let rhs ← asSum args[3]!
    let goal ← goal.replaceTargetDefEq (mkAppN t.getAppFn #[args[0]!, args[1]!, lhs, rhs])
    let children ← goal.apply (← mkConstWithFreshMVarLevels ``fp32Sum.reindex) {newGoals := .all}
    let some permGoal ← children.findM? (fun g => do
      return !(← g.isAssigned) && (← whnf (← g.getType)).isAppOfArity ``Equiv 2)
      | original.restore; return false
    let choices ← permutations permGoal facts
    let applied ← saveState
    for choice in choices do
      applied.restore
      permGoal.assign choice
      if ← children.allM (fun g => VeriTile.Meta.FPProve.solveUsing g facts 6) then return true
  catch _ => pure ()
  original.restore
  return false

/-- Opaque-call congruence preserves exact labels and argument multiplicity.
Reduction rewriting is attempted before unfolding a symbolic sum tree. -/
partial def proveValues (goal : MVarId) (facts : Array Expr) (fuel : Nat) : MetaM Bool :=
  goal.withContext do
    if ← goal.isAssigned then return true
    if ← VeriTile.Meta.FPProve.solveUsing goal facts 6 then return true
    if fuel == 0 then return false
    let target ← whnf (← goal.getType)
    if target.isAppOfArity ``TermEq 4 then
      let saved ← saveState
      try
        let children ← goal.apply (← mkConstWithFreshMVarLevels ``TermEq.refl)
        if children.isEmpty then return true
      catch _ => pure ()
      saved.restore
      if ← reorder? goal facts then return true
      try
        let children ← goal.apply (← mkConstWithFreshMVarLevels ``TermEq.app_congr)
        if ← children.allM (fun g => proveValues g facts (fuel - 1)) then return true
      catch _ => pure ()
      saved.restore
    else if target.isAppOfArity ``List.Forall₂ 5 then
      for name in [``List.Forall₂.nil, ``List.Forall₂.cons] do
        let saved ← saveState
        try
          let children ← goal.apply (← mkConstWithFreshMVarLevels name)
          if ← children.allM (fun g => proveValues g facts (fuel - 1)) then return true
        catch _ => pure ()
        saved.restore
    return false

elab_rules : tactic
  | `(tactic| fp_prove $[ (maxSteps := $steps:num)]? $[ [$hints:term,*]]?) => do
    let goal ← getMainGoal
    goal.withContext do
      unless (← whnf (← goal.getType)).isAppOfArity ``TermEq 4 do throwUnsupportedSyntax
      let mut facts := #[]
      if let some hints := hints then
        for hint in hints.getElems do facts := facts.push (← elabTerm hint none)
      facts := facts ++ (← VeriTile.Meta.FPProve.registeredRules)
      for localDecl in ← getLCtx do
        unless localDecl.isImplementationDetail do facts := facts.push localDecl.toExpr
      let saved ← saveState
      if ← proveValues goal facts (steps.map (·.getNat) |>.getD 8) then
        replaceMainGoal []
      else
        saved.restore
        throwTacticEx `fp_prove goal
          "fp_prove could not close this FP output relation. Reduction reordering requires the fp32 scalar addition rules and a checked permutation of all inputs; precision, casts and padding remain observable."

end VeriTile.Triton.FP.Tactics
