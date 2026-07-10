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
import Lean

open Lean Elab Command

namespace VeriTile.Meta

/-- Core / primitive namespaces that are axiomatically trusted — not project
definitions that could encode a cheat. -/
def isCoreConst (n : Name) : Bool :=
  let s := n.toString
  ["Eq", "Ne", "And", "Or", "Not", "Iff", "True", "False", "HEq",
   "Nat", "Int", "Fin", "List", "Option", "Prod", "PUnit", "Sum", "Subtype",
   "Real", "WithBot", "Bool", "Char", "String",
   "Membership", "OfNat", "HMul", "HAdd", "HSub", "HDiv", "HPow",
   "Mul", "Add", "Sub", "Div", "Neg", "Sort", "Exists", "Decidable",
   "Function", "Set", "LE", "LT", "instH", "inst"].any (fun p => s.startsWith p)

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
      if seen.contains n || isCoreConst n then
        projValueClosure env rest.toArray seen
      else
        let seen := seen.insert n
        let next := valueConsts env n
        projValueClosure env (rest.toArray ++ next) seen

elab "#stmtConsts " id:ident : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverload id
  let cs := (← stmtConsts name).qsort (·.toString < ·.toString)
  logInfo m!"{name} — statement uses {cs.size} constants:\n{cs}"

elab "#auditStmt " id:ident : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverload id
  let cs := ((← stmtConsts name).filter (fun n => ! isCoreConst n)).qsort (·.toString < ·.toString)
  logInfo m!"{name} — trusted project surface ({cs.size}):\n{cs}"

elab "#stmtSurfaceSubset " id:ident " ⊆ " "[" allow:ident,* "]" : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverload id
  let allowNames ← allow.getElems.mapM (fun a => liftCoreM <| realizeGlobalConstNoOverload a)
  let proj := (← stmtConsts name).filter (fun n => ! isCoreConst n)
  let bad := proj.filter (fun n => ! allowNames.contains n)
  if bad.isEmpty then
    logInfo m!"{name}: statement's project surface ⊆ allowlist ✓"
  else
    throwError m!"{name}: statement mentions non-allowlisted project constants:\n{bad}"

elab "#axiomsClean " id:ident : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverload id
  let env ← getEnv
  let (_, st) := ((CollectAxioms.collect name).run env).run {}
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  let bad := st.axioms.filter (fun a => ! allowed.contains a)
  if bad.isEmpty then
    logInfo m!"{name}: axiom footprint ⊆ standard base ✓  ({st.axioms})"
  else
    throwError m!"{name}: DISALLOWED axioms {bad} (sorryAx ⇒ fake proof)"

elab "#specNonCircular " spec:ident " avoiding " "[" ks:ident,* "]" : command => do
  let specName ← liftCoreM <| realizeGlobalConstNoOverload spec
  let kernelNames ← ks.getElems.mapM (fun k => liftCoreM <| realizeGlobalConstNoOverload k)
  let env ← getEnv
  let closure := projValueClosure env (valueConsts env specName) {}
  let hit := kernelNames.filter (fun k => closure.contains k)
  if hit.isEmpty then
    logInfo m!"{specName}: definition does not reference {kernelNames} — non-circular ✓"
  else
    throwError m!"{specName}: SELF-REFERENTIAL — definition transitively uses kernel(s) {hit}"

end VeriTile.Meta
