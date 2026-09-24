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

end VeriTile.Meta
