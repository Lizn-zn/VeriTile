/- Regression coverage for exact dtype names and constructor/cast dispatch. -/

import VeriTile.Triton.DSL
import Lean

open Lean Elab Command Term
open VeriTile.Triton VeriTile.Triton.DSL

namespace VeriTile.Bench.Tests.DTypeResolution

run_cmd liftTermElabM do
  liftMacroM do
    let cases : Array (Name × DInfo) := #[
      (`tl.float64, .real), (`tl.float32, .real), (`tl.float16, .fp16),
      (`tl.bfloat16, .bf16), (`tl.float8e4nv, .f8e4), (`tl.float8e5, .f8e5),
      (`tl.int1, .bool), (`tl.int8, .int), (`tl.int16, .int),
      (`tl.int32, .int), (`tl.int64, .int), (`tl.uint8, .nat),
      (`tl.uint16, .nat), (`tl.uint32, .nat), (`tl.uint64, .nat),
      (`OUT_DTYPE, .floatVar "out_dtype"), (`DTYPE, .floatVar "dtype")]
    for (name, expected) in cases do
      let term : TSyntax `term := ⟨mkIdent name⟩
      let value ← if expected == .bool then `(tritonExpr| true) else `(tritonExpr| 7)
      let full ← expandFullDTypeTerm expandExpr [] #[] value term
      unless full.dtype == expected do
        Macro.throwError s!"Wrong tl.full dtype for {name}"
      unless full.computeDType? == (if name == `tl.float32 then some .fp32 else none) do
        Macro.throwError s!"Wrong compute projection for {name}"
      -- Numeric zero construction must use the same dtype dispatch as full.
      if expected != .bool then
        let zeros ← expandZerosDTypeTerm expandExpr [] #[] term
        unless zeros.dtype == expected && zeros.computeDType? == full.computeDType? do
          Macro.throwError s!"Wrong tl.zeros dtype for {name}"

run_cmd liftTermElabM do
  liftMacroM do
    let pointer : TSyntax `term := ⟨mkIdent `out.dtype.element_ty⟩
    for term in #[pointer, ← `(($pointer))] do
      let some (.pointerElement name) ← resolveDTypeTerm? term
        | Macro.throwError "Expected a pointer-element dtype"
      unless name == `out do Macro.throwError "Wrong pointer name"
      let full ← expandFullDTypeTerm expandExpr [] #[] (← `(tritonExpr| 7)) term
      let zeros ← expandZerosDTypeTerm expandExpr [] #[] term
      unless full.dtype == .real && zeros.dtype == .real do
        Macro.throwError "Pointer-element constructor inference changed"

    let invalid : Array (TSyntax `term) := #[
      ⟨mkIdent `NOT_OUT_DTYPE⟩, ⟨mkIdent `tl.float32_extra⟩,
      ⟨mkIdent `tl.int128⟩, ⟨mkIdent `tl.uint128⟩,
      ⟨mkIdent `fake_dtype_element_ty⟩, ⟨mkIdent `out.dtype.element_type⟩,
      ← `("tl.float32"), ← `("dtype.element_ty")]
    for term in invalid do
      unless (← resolveDTypeTerm? term).isNone do
        Macro.throwError "An unrelated name or string was accepted as a dtype"
      let fullRejected ← try
        let _ ← expandFullDTypeTerm expandExpr [] #[] (← `(tritonExpr| 7)) term
        pure Bool.false
      catch _ => pure Bool.true
      let zerosRejected ← try
        let _ ← expandZerosDTypeTerm expandExpr [] #[] term
        pure Bool.false
      catch _ => pure Bool.true
      unless fullRejected && zerosRejected do
        Macro.throwError "Unsupported constructor dtype did not fail"

run_cmd liftTermElabM do
  let parserEnv ← getEnv
  for (text, expected) in [
      ("(x).to(tl . int16)", DInfo.int),
      ("(x).to(tl . uint32)", DInfo.nat)] do
    let .ok parsed := Parser.runParserCategory parserEnv `tritonExpr text
      | throwError "Cannot parse explicit cast target regression"
    let alternatives := if parsed.getKind == `choice then parsed.getArgs else #[parsed]
    let some parsed := alternatives.find? (·.isOfKind ``tritonMethodCastDTypeIdent)
      | throwError "Expected the qualified-identifier cast syntax"
    liftMacroM do
      unless (← methodCastTargetDType? ⟨parsed⟩) == some expected do
        Macro.throwError "Explicit cast target was not resolved"
  for text in ["(tl.intensity).to(out.dtype.element_ty)",
               "(tl.uintensity).to(other.dtype)"] do
    let .ok parsed := Parser.runParserCategory parserEnv `tritonExpr text
      | throwError "Cannot parse dtype cast regression"
    liftMacroM do
      let env : Env := [("tl.intensity", .real, SInfo.scalar, none),
                        ("tl.uintensity", .real, SInfo.scalar, none)]
      let expr : TSyntax `tritonExpr := ⟨parsed⟩
      unless (← methodCastTargetDType? expr).isNone do
        Macro.throwError "Cast operand spelling leaked into its target dtype"
      let out ← expandExpr env expr
      unless out.dtype == .real do
        Macro.throwError "Attribute cast changed an unrelated real value"

-- Exercise the public parser as well as the shared expansion entry points.
def signedFull (out : Region .int) : ComputeKernel := triton {
  value = tl.full([], 7, dtype=tl.int16)
  tl.store($((out : Region .int)), value)
}

def signedFullReordered (out : Region .int) : ComputeKernel := triton {
  value = tl.full([], dtype=tl.int16, value=7)
  tl.store($((out : Region .int)), value)
}

def signedFullValueFirst (out : Region .int) : ComputeKernel := triton {
  value = tl.full([], value=7, dtype=tl.int16)
  tl.store($((out : Region .int)), value)
}

theorem full_keyword_order_preserves_kernel (out : Region .int) :
    signedFull out = signedFullReordered out ∧
      signedFull out = signedFullValueFirst out := ⟨rfl, rfl⟩

def signedZeros (out : Region .int) : ComputeKernel := triton {
  value = tl.zeros([], dtype=tl.int16)
  tl.store($((out : Region .int)), value)
}

def signedZeroFull (out : Region .int) : ComputeKernel := triton {
  value = tl.full([], 0, dtype=tl.int16)
  tl.store($((out : Region .int)), value)
}

theorem zeros_preserves_integer_kernel (out : Region .int) :
    signedZeros out = signedZeroFull out := rfl

/-- error: tl.full: expected `dtype=<ptr>.dtype.element_ty` or a Triton dtype -/
#guard_msgs (error, drop info) in
#check fun (out_dtype : FloatDType) => (triton {
  value = tl.full([], 7, dtype=NOT_OUT_DTYPE)
} : ComputeKernel)

/-- error: tl.zeros: expected `dtype=<ptr>.dtype.element_ty` or a Triton dtype -/
#guard_msgs (error, drop info) in
#check fun (out_dtype : FloatDType) => (triton {
  value = tl.zeros([], dtype=NOT_OUT_DTYPE)
} : ComputeKernel)

end VeriTile.Bench.Tests.DTypeResolution
