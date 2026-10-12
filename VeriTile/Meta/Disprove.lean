import Lean

/-! Proof-producing counterexample search. The core is independent of Triton
and numerical models. Semantic adapters register refutation lemmas; callers
supply proved model/domain/execution facts, or register reusable hints. Search
failure is inconclusive. No oracle, axiom or failed proof attempt is evidence. -/
namespace VeriTile.Meta.Disprove
open Lean Meta Elab Tactic

/-- Declare a refutation of the proposition after the colon. Parameters stay
outside the negation: `refutation name (x : α) : P x := proof` expands to
`theorem name (x : α) : ¬ (P x) := proof`. The ordinary theorem elaborator and
kernel check the result; any proof term or tactic may supply the refutation. -/
syntax (name := refutation) declModifiers
  "refutation " declId ppIndent(declSig) declVal : command

macro_rules
  | `(command| $mods:declModifiers refutation $name:declId $sig:declSig $proof:declVal) => do
    let `(declSig| $binders* : $claim) := sig | Macro.throwUnsupported
    let negated ← `(declSig| $binders* : ¬ ($claim))
    `(command| $mods:declModifiers theorem $name:declId $negated:declSig $proof:declVal)

/-- One checked witness suffices to refute a universal claim, including a
dependent implication whose witness is a proof of its premise. -/
theorem not_forall_of_witness {α : Sort u} {P : α → Prop} (a : α) (h : ¬ P a) :
    ¬ ∀ x, P x := fun all => h (all a)

theorem ne_of_evaluations {α : Sort u} {a b x y : α}
    (left : a = x) (right : b = y) (different : x ≠ y) : a ≠ b :=
  fun same => different (left.symm.trans (same.trans right))

initialize ruleAttr : TagAttribute ←
  registerTagAttribute `disprove_rule "A checked refutation lemma used by disprove."
initialize hintAttr : TagAttribute ←
  registerTagAttribute `disprove_hint "A reusable proved fact or witness used by disprove."
initialize registerTraceClass `VeriTile.disprove

private def registered (attr : TagAttribute) : MetaM (Array Expr) := do
  let env ← getEnv
  let mut names := (attr.ext.getState env).toArray
  for i in [:env.header.moduleNames.size] do
    names := names ++ attr.ext.getModuleEntries env i
  (names.qsort Name.quickLt).mapM mkConstWithFreshMVarLevels

/-- Flatten conjunctions so a bundle of execution facts remains usable. -/
private partial def expandFact (fact : Expr) (fuel : Nat := 8) : MetaM (Array Expr) := do
  let ty ← whnf (← inferType fact)
  if fuel > 0 && ty.isAppOfArity ``And 2 then
    return #[fact] ++ (← expandFact (← mkAppM ``And.left #[fact]) (fuel - 1)) ++
      (← expandFact (← mkAppM ``And.right #[fact]) (fuel - 1))
  return #[fact]

private def negated? (ty : Expr) : MetaM (Option Expr) := do
  let ty ← whnf ty
  if let .forallE _ domain body _ := ty then
    if body.isConstOf ``False then
      if ← isProp domain then return some domain
  return none

/-- Inspect a lemma's conclusion without unfolding model certificates into
their interpreter definitions or inventing values for their parameters. -/
private partial def conclusionHead (ty : Expr) : Name :=
  let ty := ty.consumeMData
  if ty.isForall then
    if ty.bindingBody!.consumeMData.isConstOf ``False then ``Not else conclusionHead ty.bindingBody!
  else if ty.isAppOfArity ``Ne 3 then ``Not
  else match ty.getAppFn with
    | .const name _ => name
    | _ => .anonymous

private def closeBy (goal : MVarId) (tactic : Syntax) : TacticM Bool := do
  let saved ← saveState
  let oldGoals ← getGoals
  try
    setGoals [goal]
    evalTactic tactic
    if (← getGoals).isEmpty then
      setGoals oldGoals
      return true
  catch _ => pure ()
  saved.restore
  return false

/-- Bounded witness enumeration; arbitrary values can also be supplied as
hints. Fin witnesses retain their checked bound, rather than truncating it. -/
private def witnesses (ty : Expr) (count : Nat) : MetaM (Array Expr) := do
  let ty ← whnf ty
  if ty.isConstOf ``Nat then return (Array.range count).map mkNatLit
  if ty.isConstOf ``Bool then return #[mkConst ``Bool.false, mkConst ``Bool.true]
  if ty.isConstOf ``Unit then return #[mkConst ``Unit.unit]
  if ty.isAppOfArity ``Fin 1 then
    let mut values := #[]
    for i in [:count] do
      try
        let bound ← mkLt (mkNatLit i) ty.getAppArgs[0]!
        unless ← isDefEq (← mkDecide bound) (mkConst ``Bool.true) do continue
        let h ← mkDecideProof bound
        values := values.push (← mkAppM ``Fin.mk #[mkNatLit i, h])
      catch _ => pure ()
    return values
  return #[]

private def instantiateHints (facts : Array Expr) (count : Nat) : MetaM (Array Expr) := do
  let mut result := facts
  for fact in facts do
    let ty ← inferType fact
    if ty.isForall && (← isProp ty) then
      for value in ← witnesses ty.bindingDomain! count do
        result := result ++ (← expandFact (mkApp fact value))
  return result

/-- Equality hints match with Lean's definitional equality, including wrappers
and dependent projections. The final inequality is still a checked proof. -/
private def closeFromEvaluations (goal : MVarId) (facts : Array Expr) : TacticM Bool :=
    goal.withContext do
  let some p ← negated? (← goal.getType) | return false
  let p ← whnf p
  unless p.isAppOfArity ``Eq 3 do return false
  let #[_, a, b] := p.getAppArgs | return false
  if ← closeBy goal (← `(tactic| solve | (intro h; injections; with_unfolding_all contradiction))) then
    return true
  let mut left := #[(a, ← mkEqRefl a)]
  let mut right := #[(b, ← mkEqRefl b)]
  for fact in facts do
    let t ← inferType fact
    if t.isAppOfArity ``Eq 3 then
      let args := t.getAppArgs
      if ← isDefEq args[1]! a then left := left.push (args[2]!, fact)
      if ← isDefEq args[1]! b then right := right.push (args[2]!, fact)
  for (x, hx) in left do
    for (y, hy) in right do
      let attempt ← saveState
      try
        let ne ← mkAppM ``Ne #[x, y]
        let proof ← mkFreshExprSyntheticOpaqueMVar ne
        if ← closeBy proof.mvarId! (← `(tactic| solve | assumption | contradiction | decide | simp_all)) then
          goal.assign (← mkAppM ``ne_of_evaluations #[hx, hy, proof])
          return true
      catch _ => pure ()
      attempt.restore
  return false

structure Config where
  maxDepth : Nat := 24
  maxNodes : Nat := 512
  maxWitnesses : Nat := 4

/-- Search all premises together so a failed observation backtracks to the
next model, input or lane. Unresolved parameters are never silently discarded. -/
private partial def search (pending : List MVarId) (facts rules : Array Expr)
    (cfg : Config) (nodes : IO.Ref Nat) (depth : Nat) : TacticM Bool := do
  let pending ← pending.filterM fun g => return !(← g.isAssigned)
  if pending.isEmpty then return true
  if depth == 0 || (← nodes.get) >= cfg.maxNodes then return false
  nodes.modify (· + 1)
  trace[VeriTile.disprove] "node {← nodes.get}: {← pending.toArray.mapM fun g => g.getType}"
  -- A later model certificate may determine an earlier carrier or interpreter.
  for idx in [:pending.length] do
    let goal := pending[idx]!
    let rest := pending.take idx ++ pending.drop (idx + 1)
    let (done, stepped) ← goal.withContext do
      let mut stepped := false
      let ty ← instantiateMVars (← goal.getType)
      let propositionGoal ← isProp ty
      let snapshot ← saveState
      if !ty.hasMVar then
        if (← closeBy goal (← `(tactic| solve | assumption | rfl | infer_instance | contradiction | decide | simp_all))) ||
            (← closeFromEvaluations goal facts) then
          stepped := true
          if ← search rest facts rules cfg nodes (depth - 1) then return (true, true)
          snapshot.restore
      for fact in facts ++ rules do
        let attempt ← saveState
        try
          let factType ← inferType fact
          -- Prefer exact certificates: their types infer the model and input.
          if ← isDefEq factType ty then
            goal.assign fact
            stepped := true
            if ← search rest facts rules cfg nodes (depth - 1) then return (true, true)
          else if propositionGoal && conclusionHead factType == conclusionHead ty then
            attempt.restore
            let children ← goal.apply fact (cfg := { newGoals := .all, allowSynthFailures := true })
            if ← search (children ++ rest) facts rules cfg nodes (depth - 1) then return (true, true)
        catch _ => pure ()
        attempt.restore
      if let some proposition ← negated? ty then
        if (← whnf proposition).isForall then
          let attempt ← saveState
          try
            let children ← goal.apply (← mkConstWithFreshMVarLevels ``not_forall_of_witness)
              (cfg := { newGoals := .all })
            if ← search (children ++ rest) facts rules cfg nodes (depth - 1) then return (true, true)
          catch _ => pure ()
          attempt.restore
      else if propositionGoal && !ty.hasMVar && ty.isForall then
        let attempt ← saveState
        try
          let (_, child) ← goal.intro1P
          if ← search (child :: rest) facts rules cfg nodes (depth - 1) then return (true, true)
        catch _ => pure ()
        attempt.restore
      if !ty.hasMVar && !(← isProp ty) then
        for value in ← witnesses ty cfg.maxWitnesses do
          let attempt ← saveState
          try
            goal.assign value
            stepped := true
            if ← search rest facts rules cfg nodes (depth - 1) then return (true, true)
          catch _ => pure ()
          attempt.restore
      return (false, stepped)
    if done then return true
    -- Every available way to solve this obligation has already been tried
    -- together with the remaining premises; do not permute the same choices.
    if stepped then return false
    -- Solving unrelated pending goals cannot change a closed proposition.
    -- Stop this branch instead of permuting the same failed obligations.
    let ty ← goal.withContext (instantiateMVars (← goal.getType))
    if !ty.hasMVar then
      if ← goal.withContext (isProp ty) then return false
  return false

/-- Refute using checked facts, semantic adapters and bounded witnesses.
Options limit search only; they never weaken a domain or model obligation. -/
declare_syntax_cat disproveOption
syntax "(" &"maxDepth" " := " num ")" : disproveOption
syntax "(" &"maxNodes" " := " num ")" : disproveOption
syntax "(" &"maxWitnesses" " := " num ")" : disproveOption
syntax (name := disprove) "disprove" disproveOption* (" [" term,* "]")? : tactic

elab_rules : tactic
  | `(tactic| disprove $options:disproveOption* $[ [$hints:term,*]]?) => withMainContext do
    let original ← getMainGoal
    let target ← whnf (← original.getType)
    unless (← negated? target).isSome || target.isConstOf ``False do
      throwTacticEx `disprove original "disprove expects a negated proposition or False."
    let saved ← saveState
    let mut goal := original
    let mut facts := #[]
    let mut supplied ← registered hintAttr
    if let some hints := hints then
      for hint in hints.getElems do supplied := supplied.push (← elabTerm hint none)
    for fact in supplied do
      if ← isProp (← inferType fact) then
        let next ← goal.assert (← mkFreshUserName `disprove_fact) (← inferType fact) fact
        let (_, next) ← next.intro1P
        goal := next
      else facts := facts.push fact
    let locals ← goal.withContext do
      let mut result := #[]
      for localDecl in ← getLCtx do
        unless localDecl.isImplementationDetail do
          result := result ++ (← expandFact localDecl.toExpr)
      return result
    facts := facts ++ locals
    let rules ← registered ruleAttr
    let mut cfg : Config := {}
    for option in options do
      match option with
      | `(disproveOption| (maxDepth := $n:num)) => cfg := { cfg with maxDepth := n.getNat }
      | `(disproveOption| (maxNodes := $n:num)) => cfg := { cfg with maxNodes := n.getNat }
      | `(disproveOption| (maxWitnesses := $n:num)) => cfg := { cfg with maxWitnesses := n.getNat }
      | _ => throwUnsupportedSyntax
    facts ← goal.withContext (instantiateHints facts cfg.maxWitnesses)
    let used ← IO.mkRef 0
    if ← search [goal] facts rules cfg used cfg.maxDepth then
      let proof ← instantiateMVars (mkMVar original)
      unless proof.hasMVar do
        replaceMainGoal []
        return
    saved.restore
    throwTacticEx `disprove original
      "disprove found no checked counterexample within the search bounds.\nSupply model, domain and execution facts with disprove [...], or import/register a @[disprove_rule] semantic adapter.\nFailure is inconclusive; it proves neither equivalence nor inequivalence."

end VeriTile.Meta.Disprove
