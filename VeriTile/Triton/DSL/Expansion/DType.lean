/-
Structured dtype resolution for constructor and cast expansion.

Identifiers are matched as complete Lean names. A spelling inside an unrelated
identifier, expression, or string is not a dtype declaration.
-/

import VeriTile.Triton.DSL.Expansion.Compute

open Lean

namespace VeriTile.Triton.DSL

/-- Recover a dtype name, allowing parentheses around the same identifier. -/
partial def dtypeTermName? (term : TSyntax `term) : Option Name :=
  match term with
  | `($name:ident) => some name.getId.eraseMacroScopes
  | `(($inner:term)) => dtypeTermName? inner
  | `($base:term.$field:ident) => do
      return (← dtypeTermName? base) ++ field.getId.eraseMacroScopes
  | _ => none

/-- Decode the same explicit spellings accepted by the `tritonDType` category. -/
def dtypeSyntaxOfName? (name : Name) : MacroM (Option (TSyntax `tritonDType)) := do
  let dtype ← match name with
    | `tl.float64 => `(tritonDType| tl.float64)
    | `tl.float32 => `(tritonDType| tl.float32)
    | `tl.float16 => `(tritonDType| tl.float16)
    | `tl.bfloat16 => `(tritonDType| tl.bfloat16)
    | `tl.float8e4nv => `(tritonDType| tl.float8e4nv)
    | `tl.float8e5 => `(tritonDType| tl.float8e5)
    | `tl.int1 => `(tritonDType| tl.int1)
    | `tl.int8 => `(tritonDType| tl.int8)
    | `tl.int16 => `(tritonDType| tl.int16)
    | `tl.int32 => `(tritonDType| tl.int32)
    | `tl.int64 => `(tritonDType| tl.int64)
    | `tl.uint8 => `(tritonDType| tl.uint8)
    | `tl.uint16 => `(tritonDType| tl.uint16)
    | `tl.uint32 => `(tritonDType| tl.uint32)
    | `tl.uint64 => `(tritonDType| tl.uint64)
    | `OUT_DTYPE => `(tritonDType| OUT_DTYPE)
    | `DTYPE => `(tritonDType| DTYPE)
    | _ => return none
  return some dtype

inductive DTypeTermSource where
  | explicit (dtype : TSyntax `tritonDType)
  | pointerElement (pointer : Name)

/-- Preserve the distinction between an explicit dtype and a pointer's element
type. The latter keeps the existing constructor inference policy. -/
def resolveDTypeTerm? (term : TSyntax `term) : MacroM (Option DTypeTermSource) := do
  let some name := dtypeTermName? term | return none
  if let some dtype ← dtypeSyntaxOfName? name then
    return some (.explicit dtype)
  match name with
  | .str (.str pointer "dtype") "element_ty" =>
      if pointer.isAnonymous then return none
      return some (.pointerElement pointer)
  | _ => return none

/-- Read only the target of an attribute-style `.to(...)` node. The expression
being cast may itself contain dtype spellings and must not affect this result. -/
def methodCastTargetName? (stx : TSyntax `tritonExpr) : Option Name := do
  let args := stx.raw.getArgs
  let opening ← args.findIdx? (fun arg => arg.isAtom && arg.getAtomVal == "(")
  let closing ← args.back?
  unless closing.isAtom && closing.getAtomVal == ")" do failure
  let target := args.extract (opening + 1) (args.size - 1)
  if target.size == 1 then
    return ← dtypeTermName? ⟨target[0]!⟩
  if target.isEmpty || target.size % 2 == 0 then failure
  let mut name := Name.anonymous
  for i in [:target.size] do
    let part := target[i]!
    if i % 2 == 0 then
      unless part.isIdent do failure
      name := name ++ part.getId.eraseMacroScopes
    else
      unless part.isAtom && part.getAtomVal == "." do failure
  return name

def methodCastTargetDType? (stx : TSyntax `tritonExpr) : MacroM (Option DInfo) := do
  let some name := methodCastTargetName? stx | return none
  let some dtype ← dtypeSyntaxOfName? name | return none
  return some (← expandDType dtype)

def expandFullDTypeTerm (expandExpr : ExprExpander) (env : Env)
    (dims : Array (TSyntax `tritonExpr)) (value : TSyntax `tritonExpr)
    (dtype : TSyntax `term) : MacroM EOut := do
  match ← resolveDTypeTerm? dtype with
  | some (.explicit dtype) => expandComputeFull expandExpr env dims value dtype
  | some (.pointerElement _) => expandFull expandExpr env dims value
  | none =>
      Macro.throwError "tl.full: expected `dtype=<ptr>.dtype.element_ty` or a Triton dtype"

def expandZerosDTypeTerm (expandExpr : ExprExpander) (env : Env)
    (dims : Array (TSyntax `tritonExpr)) (dtype : TSyntax `term) : MacroM EOut := do
  match ← resolveDTypeTerm? dtype with
  | some (.explicit dtype) => expandComputeZeros expandExpr env dims dtype
  | some (.pointerElement _) =>
      expandFull expandExpr env dims (← `(tritonExpr| 0))
  | none =>
      Macro.throwError "tl.zeros: expected `dtype=<ptr>.dtype.element_ty` or a Triton dtype"

end VeriTile.Triton.DSL
