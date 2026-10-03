/-
VeriTile.Meta.StatementAudit

Machine-checkable trust audit for theorem statements. The soundness of a
theorem depends ONLY on (a) the constants in its *statement* (type) and (b) its
proof axiom footprint — never on the defs/lemmas used only in its *proof* (Lean's
kernel checks the proof against the stated type). These commands automate that
audit:

* `#stmtConsts T`          — every constant in `T`'s statement.
* `#auditStmt T`           — just the project (non-core) constants: the surface
                             a human must read.
* `#print_fp_assumptions T` — atomic assumptions referenced by an FP proof.
* `#print_spec T`          — reader-facing claim, premises and atomic rules.
* `#print_spec T full`     — configuration/evidence, transitive dependencies
                             and the complete trust audit.
* `#stmtSurfaceSubset T ⊆ [a, b, …]` — GATE: fail if the statement mentions a
                             project constant outside the allowlist (e.g. a spec
                             sneaking into a spec-free headline).
* `#axiomsClean T`         — GATE: fail unless the axiom footprint ⊆
                             {propext, Classical.choice, Quot.sound} (no
                             `sorryAx`, no smuggled axiom).
* `#specNonCircular s avoiding [k, …]` — GATE: fail if the *definition* of `s`
                             transitively references any kernel in the list
                             (a self-referential spec is a circular proof).
-/
import VeriTile.Meta.Specification

open Lean Elab Command

namespace VeriTile.Meta

/-- Explicit registration for mathematical specifications with arbitrary names. -/
initialize independentSpecAttr : TagAttribute ←
  registerTagAttribute `kernel_spec "An independent mathematical kernel specification."

/-- Execution denotations are inventoried separately, not independent specifications. -/
initialize kernelDenotationAttr : TagAttribute ←
  registerTagAttribute `kernel_denotation "A kernel execution denotation."

/-- Declarations originating in the trusted Lean/Mathlib dependencies. Use the
environment's defining module, never the declaration's spelling: project code
can extend `Nat`, `Real`, or an `inst…` namespace. Current-module declarations
and declarations from other imports always remain in the audited surface.
Like the rest of this audit, this assumes the imported dependencies are trusted. -/
def isCoreConst (env : Environment) (n : Name) : Bool :=
  match env.getModuleIdxFor? n with
  | none => false
  | some idx =>
    let origin := env.header.modules[idx.toNat]!.module
    #[`Init, `Std, `Lean, `Mathlib].any (·.isPrefixOf origin)

/-- Constants appearing in a declaration's STATEMENT (its type). -/
def stmtConsts (name : Name) : CommandElabM (Array Name) := do
  let info ← liftCoreM <| getConstInfo name
  return info.type.getUsedConstants

/-- Constants in the VALUE (body) of a declaration, or `#[]` if it has none. -/
def valueConsts (env : Environment) (name : Name) : Array Name :=
  match env.find? name with
  | some info => (info.value?.map (·.getUsedConstants)).getD #[]
  | none => #[]

/-- Transitive closure of `valueConsts`, restricted to project (non-core)
constants — deliberately does not descend into Mathlib. -/
partial def projValueClosure (env : Environment) : Array Name → NameSet → NameSet
  | work, seen =>
    match work.toList with
    | [] => seen
    | n :: rest =>
      if seen.contains n then
        projValueClosure env rest.toArray seen
      else
        let seen := seen.insert n
        let next := if isCoreConst env n then #[] else valueConsts env n
        projValueClosure env (rest.toArray ++ next) seen

elab "#stmtConsts " id:ident : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverload id
  let cs := (← stmtConsts name).qsort (·.toString < ·.toString)
  logInfo m!"{name} — statement uses {cs.size} constants:\n{cs}"

elab "#auditStmt " id:ident : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverload id
  let env ← getEnv
  let cs := ((← stmtConsts name).filter (fun n => ! isCoreConst env n)).qsort (·.toString < ·.toString)
  logInfo m!"{name} — trusted project surface ({cs.size}):\n{cs}"

elab "#stmtSurfaceSubset " id:ident " ⊆ " "[" allow:ident,* "]" : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverload id
  let allowNames ← allow.getElems.mapM (fun a => liftCoreM <| realizeGlobalConstNoOverload a)
  let env ← getEnv
  let proj := (← stmtConsts name).filter (fun n => ! isCoreConst env n)
  let bad := proj.filter (fun n => ! allowNames.contains n)
  if bad.isEmpty then
    logInfo m!"{name}: statement's project surface ⊆ allowlist ✓"
  else
    throwError m!"{name}: statement mentions non-allowlisted project constants:\n{bad}"

/-- Shared axiom check for explicit names and environment-discovered headlines. -/
def auditAxioms (name : Name) : CommandElabM Unit := do
  let env ← getEnv
  let (_, st) := ((CollectAxioms.collect name).run env).run {}
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  let bad := st.axioms.filter (fun a => ! allowed.contains a)
  if bad.isEmpty then
    logInfo m!"{name}: axiom footprint ⊆ standard base ✓  ({st.axioms})"
  else
    throwError m!"{name}: DISALLOWED axioms {bad} (sorryAx ⇒ fake proof)"

elab "#axiomsClean " id:ident : command => do
  auditAxioms (← liftCoreM <| realizeGlobalConstNoOverload id)

/-- Check all registered headlines and legacy theorem suffixes in this module.
Discovery uses the elaborated environment, including private and Unicode names. -/
elab "#auditModuleAxioms" : command => do
  let env ← getEnv
  let suffixes := #["_compute_correct", "_correct", "_output_summary_general", "_output_summary"]
  let mut names : Array Name := #[]
  for (name, info) in env.constants.toList do
    let userName := ((privateToUserName? name).getD name).eraseMacroScopes
    if (env.getModuleIdxFor? name).isSome then
      continue
    let .thmInfo _ := info | continue
    if headlineAttr.hasTag env name || suffixes.any (fun suffix => userName.getString!.endsWith suffix) then
      names := names.push name
  names := names.qsort (·.toString < ·.toString)
  for name in names do auditAxioms name
  let display := names.map fun name => (privateToUserName? name).getD name
  logInfo m!"Axiom audit: headlines={names.size}\nheadlines: {display}"

elab "#specNonCircular " spec:ident " avoiding " "[" ks:ident,* "]" : command => do
  let specName ← liftCoreM <| realizeGlobalConstNoOverload spec
  let kernelNames ← ks.getElems.mapM (fun k => liftCoreM <| realizeGlobalConstNoOverload k)
  let env ← getEnv
  let closure := projValueClosure env #[specName] {}
  let hit := kernelNames.filter (fun k => closure.contains k)
  if hit.isEmpty then
    logInfo m!"{specName}: definition does not reference {kernelNames} — non-circular ✓"
  else
    throwError m!"{specName}: SELF-REFERENTIAL — definition transitively uses kernel(s) {hit}"

/-- Discover kernels by their elaborated result type, including parameterized and
multiline declarations. Specs use the existing `*Spec` convention or `kernel_spec`;
execution denotations must instead be registered with `kernel_denotation`.
Every discovered pair is checked, and the actual inventory is printed. -/
elab "#auditModuleSpecs" : command => do
  let env ← getEnv
  let mut kernels : Array Name := #[]
  let mut specs : Array Name := #[]
  let mut denotations : Array Name := #[]
  for (name, info) in env.constants.toList do
    let userName := ((privateToUserName? name).getD name).eraseMacroScopes
    if (env.getModuleIdxFor? name).isSome then
      continue
    unless info.isDefinition do continue
    let isKernel ← liftTermElabM do
      Meta.forallTelescopeReducing info.type fun _ result => do
        return result.isConstOf `VeriTile.Triton.ComputeKernel
    if isKernel then kernels := kernels.push name
    let isDenotation := kernelDenotationAttr.hasTag env name
    if isDenotation then denotations := denotations.push name
    if independentSpecAttr.hasTag env name ||
        (!isDenotation && userName.getString!.endsWith "Spec") then
      specs := specs.push name
  kernels := kernels.qsort (·.toString < ·.toString)
  specs := specs.qsort (·.toString < ·.toString)
  denotations := denotations.qsort (·.toString < ·.toString)
  if !specs.isEmpty && kernels.isEmpty then
    throwError "Spec audit found {specs.size} specifications but no ComputeKernel declarations"
  for spec in specs do
    let closure := projValueClosure env #[spec] {}
    let hit := kernels.filter (closure.contains ·)
    unless hit.isEmpty do
      throwError "{spec}: SELF-REFERENTIAL — definition transitively uses kernel(s) {hit}"
  let display := fun (names : Array Name) =>
    (names.map fun name => (privateToUserName? name).getD name).qsort (·.toString < ·.toString)
  logInfo m!"Spec audit: kernels={kernels.size}, independentSpecs={specs.size}, denotations={denotations.size}\nkernels: {display kernels}\nindependent specs: {display specs}\ndenotations: {display denotations}"

/-! ## Public specification reports

This is inspection, not an acceptance checker. Traversal follows both types
and proof/definition bodies in project modules, stopping at trusted library
boundaries. It is conservative: all branches of a reachable definition count.
No trace of actual GPU operations or minimal logical dependency is claimed.
-/

/-- `getUsedConstants` omits raw `Expr.proj` nodes. Recover their projection
functions too, so a model/typeclass law cannot disappear from the report. -/
def specExprConsts (env : Environment) (e : Expr) : CoreM (Array Name) := do
  let visit : StateRefT (Array Name) CoreM Unit := e.forEach fun node => do
    if let .proj structureName index _ := node then
      if let some info := getStructureInfo? env structureName then
        if let some projection := info.getProjFn? index then
          modify (·.push projection)
  return (← visit.run e.getUsedConstants).2

structure SpecDependencies where
  project : Array Name := #[]
  library : Array Name := #[]

/-- Transitive syntactic dependency closure, with an explicit trusted-library
boundary. Origins, not namespace spellings, determine the boundary. -/
partial def specDependencies (env : Environment) (work : List Name)
    (seen : NameSet := {}) (found : SpecDependencies := {}) : CoreM SpecDependencies := do
  match work with
  | [] => return found
  | n :: rest =>
    if seen.contains n then return ← specDependencies env rest seen found
    let seen := seen.insert n
    if isCoreConst env n then
      return ← specDependencies env rest seen { found with library := found.library.push n }
    let some info := env.find? n | return ← specDependencies env rest seen found
    let mut next ← specExprConsts env info.type
    if let some value := info.value? then
      next := next ++ (← specExprConsts env value)
    specDependencies env (next.toList ++ rest) seen { found with project := found.project.push n }

private def legacyPrimitive (n : Name) : Bool :=
  #[`VeriTile.Triton.NumericDType.add, `VeriTile.Triton.NumericDType.sub,
    `VeriTile.Triton.NumericDType.mul, `VeriTile.Triton.NumericDType.div,
    `VeriTile.Triton.WithBot.realAdd, `VeriTile.Triton.WithBot.realSub,
    `VeriTile.Triton.WithBot.realMul, `VeriTile.Triton.WithBot.realDiv,
    `VeriTile.Triton.FloatDType.cast, `VeriTile.Triton.RoundingModel.cast,
    `VeriTile.Triton.RoundingModel.storeValue, `VeriTile.Triton.Tile.reduceSum,
    `VeriTile.Triton.Tile.dot].contains n

/-- Syntax-level equivalence need not call an evaluator. Its operation and
statement constructors are still primitive dependencies readers must see. -/
private def syntaxPrimitive (env : Environment) (n : Name) : Bool :=
  match env.find? n with
  | some (.ctorInfo info) =>
      #[`VeriTile.Triton.Op, `VeriTile.Triton.ComputeOp,
        `VeriTile.Triton.Stmt, `VeriTile.Triton.ComputeStmt].contains info.induct
  | _ => false

private def fpProofType (name : Name) : Bool :=
  #[``VeriTile.Spec.Derivation, ``VeriTile.Spec.ProgramDerivation,
    ``VeriTile.Spec.FloatingPoint, ``VeriTile.Spec.ProgramSyntax.Derivation,
    ``VeriTile.Spec.ProgramSyntax.numerical,
    `VeriTile.Triton.FP.Equational.TermEq, `VeriTile.Triton.FP.Equational.OptionalEq,
    `VeriTile.Triton.FP.Structural.IO₁NumericalEquiv,
    `VeriTile.Triton.FP.Guarded.Equivalent,
    `VeriTile.Triton.FP.Guarded.Equivalent₁ₓ₂,
    `VeriTile.Triton.FP.Scheduled.Equivalent₁ₓ₂,
    `VeriTile.Triton.FP.Structural.CellRelated,
    `VeriTile.Triton.FP.Structural.ValueRelated].contains name

/-- Stop before unfolding either public relation. Follow aliases one step at
a time; a bounded failure is reported as an unwrapped legacy declaration. -/
private def specConclusion (e : Expr) : MetaM Expr := do
  let mut e := e
  for _ in [:64] do
    if e.isAppOf ``VeriTile.Spec.Real ||
        (match e.getAppFn with | .const name _ => fpProofType name | _ => false) then
      return e
    match ← Meta.unfoldDefinition? e (ignoreTransparency := true) with
    | some next => e := next
    | none => return e
  return e

private def ppSpecField (label : String) (projection : Name) (value : Expr) : MetaM Unit := do
  let env ← getEnv
  let mut field ← Meta.mkAppM projection #[value]
  -- Reduce the record, then select the field without unfolding its contents.
  -- In particular a program stays a named program, not a printed AST dump.
  if let some info := env.getProjectionFnInfo? projection then
    let record ← Meta.withTransparency .all <| Meta.whnf value
    if record.isAppOf info.ctorName then
      if let some selected := record.getAppArgs[info.numParams + info.i]? then
        field := selected
  if projection == ``VeriTile.Spec.Evidence.bias || projection == ``VeriTile.Spec.Evidence.vars then
    let rendered ← Meta.withTransparency .all <| Meta.whnf
      (← Meta.mkAppM ``VeriTile.Spec.GateStatus.label #[field])
    if let some status := Meta.getStringValue? rendered then
      logInfo m!"  {label}: {status}"
      return
  -- Imported report rows use named projections for their IDs and digest keys.
  -- Print the actual identity rather than an unevaluated `row.ruleID`.
  if projection == ``VeriTile.Spec.Contract.ruleID ||
      projection == ``VeriTile.Spec.Contract.instanceKey ||
      projection == ``VeriTile.Spec.Contract.description ||
      projection == ``VeriTile.Spec.Evidence.instanceKey then
    let rendered ← Meta.withTransparency .all <| Meta.whnf field
    if let some value := Meta.getStringValue? rendered then
      logInfo m!"  {label}: {repr value}"
      return
  if projection == ``VeriTile.Spec.Evidence.artifact then
    let rendered ← Meta.withTransparency .all <| Meta.whnf field
    if rendered.isAppOf ``Option.some then
      let value ← Meta.withTransparency .all <| Meta.whnf rendered.getAppArgs.back!
      if let some artifact := Meta.getStringValue? value then
        logInfo m!"  {label}: some {repr artifact}"
        return
  logInfo m!"  {label}: {← Meta.ppExpr field}"

private def printAtomicEntry (entry : Expr) : MetaM Unit := do
  let rule ← Meta.mkAppM ``VeriTile.Spec.RuleEntry.rule #[entry]
  let evidence ← Meta.mkAppM ``VeriTile.Spec.RuleEntry.evidence #[entry]
  let contract ← Meta.mkAppM ``VeriTile.Spec.AtomicRule.contract #[rule]
  ppSpecField "rule ID" ``VeriTile.Spec.Contract.ruleID contract
  ppSpecField "atom lhs" ``VeriTile.Spec.AtomicRule.lhs rule
  ppSpecField "atom rhs" ``VeriTile.Spec.AtomicRule.rhs rule
  ppSpecField "instance key" ``VeriTile.Spec.Contract.instanceKey contract
  ppSpecField "configuration" ``VeriTile.Spec.Contract.configuration contract
  ppSpecField "configuration summary" ``VeriTile.Spec.Contract.description contract
  ppSpecField "warning policy" ``VeriTile.Spec.Contract.warningPolicy contract
  ppSpecField "evidence key" ``VeriTile.Spec.Evidence.instanceKey evidence
  ppSpecField "artifact" ``VeriTile.Spec.Evidence.artifact evidence
  ppSpecField "bias gate" ``VeriTile.Spec.Evidence.bias evidence
  ppSpecField "vars gate" ``VeriTile.Spec.Evidence.vars evidence
  let validated ← Meta.mkAppM ``VeriTile.Spec.EvidenceValidated #[rule, evidence]
  logInfo m!"  atomic validation obligation: {← Meta.ppExpr validated}"

private def printFloatingPointTheory (assumptions lhs rhs : Expr) : MetaM Unit := do
  logInfo m!"Implementation lhs: {← Meta.ppExpr lhs}"
  logInfo m!"Implementation rhs: {← Meta.ppExpr rhs}"
  logInfo m!"Atomic assumptions: {← Meta.ppExpr assumptions}"
  logInfo "Declared atom scope (conservative; may include unused entries):"
  let mut rest := assumptions
  for i in [:256] do
    let reduced ← Meta.withTransparency .all <| Meta.whnf rest
    if reduced.isAppOf ``List.nil then
      if i == 0 then logInfo "  none"
      break
    if reduced.isAppOf ``List.cons then
      let args := reduced.getAppArgs
      logInfo m!"Atom {i + 1}:"
      printAtomicEntry args[1]!
      rest := args[2]!
    else
      logInfo m!"  symbolic remainder: {← Meta.ppExpr rest}"
      break
    if i == 255 then logInfo "  further entries omitted (use the named table definition)."
  logInfo "Meaning: formal implementation equivalence under admitted atomic assumptions."
  logInfo "Symmetry, composition and common context are formal proof rules, not statistical gate guarantees."
  logInfo "No IEEE value equality or whole-kernel two-gates result is implied."

/-- Reduce a string-valued projection, without evaluating arbitrary code. -/
private def specString? (projection : Name) (value : Expr) : MetaM (Option String) := do
  let field ← Meta.mkAppM projection #[value]
  return Meta.getStringValue? (← Meta.withTransparency .all <| Meta.whnf field)

private def shortSpecName : Name → String
  | .str _ name => name
  | name => name.toString

private def printFPAtom (entry : Expr) (details : Bool := false) : MetaM Unit := do
  let rule ← Meta.mkAppM ``VeriTile.Spec.RuleEntry.rule #[entry]
  let contract ← Meta.mkAppM ``VeriTile.Spec.AtomicRule.contract #[rule]
  let id ← specString? ``VeriTile.Spec.Contract.ruleID contract
  unless details do
    match id with
    | some id => logInfo m!"  {id.toLower.replace "-" "_"}"
    | none => logInfo m!"  {← Meta.ppExpr entry} [symbolic atom]"
    return
  let evidence ← Meta.mkAppM ``VeriTile.Spec.RuleEntry.evidence #[entry]
  let scope ← specString? ``VeriTile.Spec.Contract.description contract
  let status (projection : Name) : MetaM String := do
    let gate ← Meta.mkAppM projection #[evidence]
    return (← specString? ``VeriTile.Spec.GateStatus.label gate).getD "symbolic"
  match id with
  | some id => logInfo m!"  {id} [bias {← status ``VeriTile.Spec.Evidence.bias}; vars {← status ``VeriTile.Spec.Evidence.vars}]"
  | none => logInfo m!"  {← Meta.ppExpr entry} [symbolic atom]"
  if let some scope := scope then
    unless scope.isEmpty do logInfo m!"    {scope}"

private structure FPAssumptionState where
  seen : Std.HashSet Expr := {}
  internalProofVars : Std.HashSet Expr := {}
  entries : Array Expr := #[]
  printedAdmissions : Array (Expr × Expr) := #[]
  unresolved : Bool := false
  fpConstants : Std.HashMap Name Bool := {}
  remaining : Nat := 10000

/-- Conservative reachability, including declaration types and projection
functions. A closed dependency graph without an FP proof type cannot hide an
atomic derivation. Cache negative results only after the entire graph has been
searched; mutual recursion must not turn a back-edge into a false negative. -/
private partial def fpConstantReachable (root : Name) (work : List Name)
    (seen : NameSet := {}) : StateRefT FPAssumptionState MetaM Bool := do
  match work with
  | [] =>
    modify fun s => { s with fpConstants := seen.foldl (fun m n => m.insert n false) s.fpConstants }
    return false
  | name :: rest =>
    if seen.contains name then return ← fpConstantReachable root rest seen
    if fpProofType name || (← get).fpConstants[name]? == some true then
      modify fun s => { s with fpConstants := s.fpConstants.insert root true }
      return true
    if (← get).fpConstants[name]? == some false then
      return ← fpConstantReachable root rest seen
    let seen := seen.insert name
    let env ← getEnv
    if isCoreConst env name then return ← fpConstantReachable root rest seen
    let some info := env.find? name | return true
    let mut next ← specExprConsts env info.type
    if let some value := info.value? then next := next ++ (← specExprConsts env value)
    fpConstantReachable root (next.toList ++ rest) seen

private def fpExprReachable (proof : Expr) : StateRefT FPAssumptionState MetaM Bool := do
  let env ← getEnv
  let mut names ← specExprConsts env proof
  -- Actual proof arguments can introduce FP dependencies into otherwise
  -- generic helpers. Include free-variable types instead of inspecting only
  -- the helper's declaration or the result type of this proof.
  let visit : StateRefT (Bool × Array Expr) MetaM Unit := proof.forEach fun e => do
    if e.isFVar then modify fun s => (s.1, s.2.push e)
    if e.isMVar then modify fun s => (true, s.2)
  let (_, localState) ← visit.run (false, #[])
  if localState.1 then return true
  for localProof in localState.2 do
    let type ← Meta.inferType localProof
    if type.getAppFn.isFVar || type.isMVar then return true
    names := names ++ (← specExprConsts env type)
  for name in names do
    if ← fpConstantReachable name [name] then return true
  return false

/-- Walk instantiated proof bodies, not theorem types or the declared table.
We inspect all reachable proof branches, not a minimal logical dependency set.
An opaque derivation is reported explicitly rather than guessed from its table.
Printing within each local context also supports atoms beneath proof lambdas. -/
private partial def visitFPAssumptions (proof : Expr) :
    StateRefT FPAssumptionState MetaM Unit := do
  let proof := proof.headBeta
  if (← get).seen.contains proof then return
  if (← get).remaining == 0 then
    throwError "FP assumption inspection exceeded its proof traversal limit"
  modify fun s => { s with seen := s.seen.insert proof, remaining := s.remaining - 1 }
  match proof with
  | .mdata _ inner => visitFPAssumptions inner
  | .letE _ _ value body _ => visitFPAssumptions (body.instantiate1 value)
  | .lam .. => Meta.lambdaTelescope proof fun params body => do
      -- These binders belong to a proof body (e.g. recursion hypotheses).
      -- The headline's external premises have already been instantiated by
      -- the command's outer telescope and must still be reported if opaque.
      let previous := (← get).internalProofVars
      modify fun s => { s with internalProofVars := params.foldl (·.insert ·) previous }
      visitFPAssumptions body
      modify fun s => { s with internalProofVars := previous }
  | _ =>
    unless ← Meta.isProof proof do return
    unless ← fpExprReachable proof do return
    let args := proof.getAppArgs
    if proof.isAppOf ``VeriTile.Spec.Derivation.atom then
      let entry := args[2]!
      -- Track each syntax instantiation, but print the same concrete admission
      -- once when it is applied at several sites. Different experimental
      -- contracts/evidence stay distinct even when their textual IDs coincide.
      let normalized ← Meta.withTransparency .all <| Meta.whnf entry
      unless (← get).entries.contains normalized do
        modify fun s => { s with entries := s.entries.push normalized }
        let rule ← Meta.mkAppM ``VeriTile.Spec.RuleEntry.rule #[entry]
        let contract ← Meta.mkAppM ``VeriTile.Spec.AtomicRule.contract #[rule]
        let evidence ← Meta.mkAppM ``VeriTile.Spec.RuleEntry.evidence #[entry]
        let concrete := (← specString? ``VeriTile.Spec.Contract.ruleID contract).isSome
        let identity := (← Meta.withTransparency .all <| Meta.whnf contract,
          ← Meta.withTransparency .all <| Meta.whnf evidence)
        -- Weak-head reduction may leave record projections inside fields.
        -- Compare closed admissions definitionally, so record updates that
        -- only rename registers do not produce duplicate printed assumptions.
        let closed := !identity.1.hasFVar && !identity.1.hasMVar &&
          !identity.2.hasFVar && !identity.2.hasMVar
        let duplicate ← if concrete && closed then
          (← get).printedAdmissions.anyM fun previous =>
            Meta.withTransparency .all do
              if previous.1.hasFVar || previous.1.hasMVar ||
                  previous.2.hasFVar || previous.2.hasMVar then return false
              return (← Meta.isDefEq previous.1 identity.1) &&
                (← Meta.isDefEq previous.2 identity.2)
          else pure false
        unless duplicate do
          modify fun s => { s with printedAdmissions := s.printedAdmissions.push identity }
          printFPAtom entry
      return
    let env ← getEnv
    if let .const name levels := proof.getAppFn then
      unless isCoreConst env name do
        let info ← getConstInfo name
        if let some value := info.value? then
          let body := value.instantiateLevelParams info.levelParams levels
          visitFPAssumptions (body.beta args)
          return
    if let .proj .. := proof.getAppFn then
      if let some reduced ← Meta.withTransparency .all <| Meta.reduceProj? proof.getAppFn then
        visitFPAssumptions (mkAppN reduced args)
        return
    -- Constructors, transports and library combinators retain their proof
    -- arguments. Skip data arguments, including unused rule-table entries.
    for arg in args do
      if ← Meta.isProof arg then visitFPAssumptions arg
    -- A local/opaque FP proof may conceal further atoms. Never print "none"
    -- in this case. Core recursors have their branches inspected above.
    let opaqueHead := match proof.getAppFn with
      | .fvar .. | .proj .. => true
      | .const name _ => match env.find? name with
          | some (.axiomInfo _) | some (.opaqueInfo _) => true
          | _ => false
      | _ => false
    let rec projectionRoot : Expr → Expr
      | .proj _ _ base => projectionRoot base.getAppFn
      | head => head
    if opaqueHead && !(← get).internalProofVars.contains (projectionRoot proof.getAppFn) then
      let type ← specConclusion (← Meta.inferType proof)
      if (match type.getAppFn with | .const name _ => fpProofType name | _ => false) then
        modify fun s => { s with unresolved := true }
        logInfo m!"  unresolved FP proof: {← Meta.ppExpr proof} (atomic assumptions unavailable)"

/-- Print only the atomic assumptions referenced by a floating-point proof.
Works on ordinary theorems as well as registered `specification` headlines. -/
elab "#print_fp_assumptions " id:ident : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverload id
  let info ← liftCoreM <| getConstInfo name
  liftTermElabM <| Meta.forallTelescope info.type fun params conclusion => do
    let target ← specConclusion conclusion
    unless (match target.getAppFn with | .const name _ => fpProofType name | _ => false) do
      throwError "{name}: expected a floating-point equivalence or derivation"
    logInfo m!"FP assumptions used by {shortSpecName name}:"
    let proof := mkAppN (mkConst name (info.levelParams.map Level.param)) params
    let (_, state) ← (visitFPAssumptions proof).run {}
    if state.entries.isEmpty && !state.unresolved then logInfo "  none"

private def printCompactAtoms (assumptions : Expr) : MetaM Unit := do
  logInfo "Atomic assumptions (declared scope):"
  let mut rest := assumptions
  for i in [:256] do
    let reduced ← Meta.withTransparency .all <| Meta.whnf rest
    if reduced.isAppOf ``List.nil then
      if i == 0 then logInfo "  none"
      return
    unless reduced.isAppOf ``List.cons do
      logInfo m!"  {← Meta.ppExpr rest} (symbolic table)"
      return
    let args := reduced.getAppArgs
    printFPAtom args[1]! (details := true)
    rest := args[2]!
  logInfo "  further entries omitted; use #print_spec ... full."

private def printCompactSpec (name : Name) (info : ConstantInfo)
    (primitives rules : Array Name) : CommandElabM Unit := do
  logInfo m!"Specification: {shortSpecName name}"
  liftTermElabM <| Meta.forallTelescope info.type fun params conclusion => do
    let target ← specConclusion conclusion
    let floating := target.isAppOf ``VeriTile.Spec.FloatingPoint
    if floating then
      logInfo "Kind: floating-point equivalence under atomic assumptions"
      let args := target.getAppArgs
      let mut theory := (← Meta.ppExpr args[2]!).pretty
      -- Coercions elaborate the public `R` to `R.assumptions`. Keep the
      -- reader-facing model spelling when it is exactly that projection.
      for param in params do
        let model := (← Meta.ppExpr param).pretty
        if theory == model ++ ".assumptions" then theory := model
      logInfo m!"Claim: {← Meta.ppExpr args[3]!} ≡[{theory}] {← Meta.ppExpr args[4]!}"
    else
      logInfo "Kind: real-valued correctness"
      let claim := if target.isAppOf ``VeriTile.Spec.Real then target.getAppArgs.back! else conclusion
      logInfo m!"Claim: {← Meta.ppExpr claim}"
    unless params.isEmpty do
      logInfo "Parameters / premises:"
      for param in params do
        let decl ← Meta.getFVarLocalDecl param
        logInfo m!"  {decl.userName} : {← Meta.ppExpr decl.type}"
    if floating then
      printCompactAtoms target.getAppArgs[2]!
    else
      unless primitives.isEmpty do
        logInfo m!"Primitives: {primitives.map shortSpecName}"
      unless rules.isEmpty do
        logInfo m!"Rules: {rules.map shortSpecName}"
  -- Compact output must never hide a nonstandard axiom or `sorryAx`.
  let (_, ax) := ((CollectAxioms.collect name).run (← getEnv)).run {}
  let bad := ax.axioms.filter fun a => ! #[`propext, `Classical.choice, `Quot.sound].contains a
  unless bad.isEmpty do logWarning m!"Nonstandard axioms: {bad}"

/-- Shared report implementation, also exercised by the regression fixtures. -/
def printSpec (name : Name) (full : Bool := false) : CommandElabM Unit := do
  let env ← getEnv
  unless headlineAttr.hasTag env name do
    throwError "{name}: not a registered specification"
  let info ← liftCoreM <| getConstInfo name
  let deps ← liftCoreM <| specDependencies env [name]
  let order (ns : Array Name) := ns.qsort (·.toString < ·.toString)
  let primitives := order ((deps.project ++ deps.library).filter fun n =>
    specPrimitiveAttr.hasTag env n || legacyPrimitive n || syntaxPrimitive env n)
  let rules := order ((deps.project ++ deps.library).filter (specRuleAttr.hasTag env))
  unless full do
    printCompactSpec name info primitives rules
    return
  logInfo m!"Specification: {name}"
  liftTermElabM <| Meta.forallTelescope info.type fun params conclusion => do
    let target ← specConclusion conclusion
    if target.isAppOf ``VeriTile.Spec.FloatingPoint then
      logInfo "Kind: FLOATING_POINT_EQUIVALENCE (two-gates-admitted atom assumptions)"
    else if target.isAppOf ``VeriTile.Spec.Real then
      logInfo "Kind: REAL (explicit mathematical claim)"
    else
      logInfo "Kind: REAL (legacy unwrapped claim; inspect the semantic dependencies)"
    logInfo m!"Conclusion: {← Meta.ppExpr conclusion}"
    logInfo "Declared parameters and assumptions (including implicit/instance binders):"
    if params.isEmpty then logInfo "  none"
    for param in params do
      let decl ← Meta.getFVarLocalDecl param
      let kind ← if ← Meta.isProp decl.type then pure "assumption"
        else if (← Meta.isClass? decl.type).isSome then pure "class parameter"
        else pure "parameter"
      logInfo m!"  {kind} {decl.userName}: {← Meta.ppExpr decl.type}"
    if target.isAppOf ``VeriTile.Spec.FloatingPoint then
      let args := target.getAppArgs
      printFloatingPointTheory args[2]! args[3]! args[4]!
    if !params.isEmpty then
      logInfo "Scope: conditional on all declared assumptions; this report does not discharge them."
  logInfo m!"Reachable execution primitives ({primitives.size}): {primitives}"
  logInfo m!"Reachable registered rules ({rules.size}): {rules}"
  liftTermElabM do
    for n in rules do
      let ruleInfo ← getConstInfo n
      logInfo m!"  rule {n}: {← Meta.ppExpr ruleInfo.type}"
      if let .defnInfo ruleDef := ruleInfo then
        logInfo m!"    definition: {← Meta.ppExpr ruleDef.value}"
    logInfo "Reachable proposition-valued structure/class fields:"
    let mut any := false
    for n in order (deps.project ++ deps.library) do
      if (env.getProjectionFnInfo? n).isSome then
        let ty := (← getConstInfo n).type
        if ← Meta.isProp ty then
          any := true
          logInfo m!"  {n}: {← Meta.ppExpr ty}"
    unless any do logInfo "  none"
  if deps.project.contains `VeriTile.Triton.RoundingModel then
    logInfo "Semantic boundary: ABSTRACT CAST/STORE ROUNDING; this is not a concrete FP execution certificate."
  let (_, ax) := ((CollectAxioms.collect name).run env).run {}
  logInfo m!"Transitive axioms: {order ax.axioms}"
  let bad := ax.axioms.filter fun a => ! #[`propext, `Classical.choice, `Quot.sound].contains a
  unless bad.isEmpty do logWarning m!"Nonstandard axioms: {bad}"
  logInfo m!"Dependency boundary: {deps.project.size} project declarations; {deps.library.size} trusted library declarations."
  logInfo "Conservative type/proof dependency report; reachable definitions can include unexecuted branches."
  if full then
    logInfo m!"Project dependencies: {order deps.project}"
    logInfo m!"Trusted library boundary: {order deps.library}"

elab "#print_spec " id:ident : command => do
  printSpec (← liftCoreM <| realizeGlobalConstNoOverload id)

elab "#print_spec " id:ident " full" : command => do
  printSpec (← liftCoreM <| realizeGlobalConstNoOverload id) true

end VeriTile.Meta
