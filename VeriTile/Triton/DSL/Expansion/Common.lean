/-
VeriTile.Triton.DSL.Expansion.Common

Shared macro expansion helpers for the Triton DSL.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.DSL.Metadata
import VeriTile.Triton.DSL.Syntax
import VeriTile.Triton.DSL.Typing

open Lean

namespace VeriTile.Triton.DSL

def identAsStr (i : TSyntax `ident) : MacroM (TSyntax `term) :=
  pure (Syntax.mkStrLit i.getId.toString)

/-- Extract a static region's identifier name from the term emitted by
`expandStaticPtrExpr`. Returns `none` for `$(R)` antiquotes (which are
arbitrary Lean terms, not syntactic region idents). Used by the region-
dtype lookup in `expandLoad` / `expandStore`. -/
def regionTermName (regionTerm : TSyntax `term) : Option String :=
  let rec go (fuel : Nat) (stx : Syntax) : Option String :=
    match fuel with
    | 0 => none
    | fuel + 1 =>
        if stx.isIdent then
          some stx.getId.toString
        else
          let args := stx.getArgs
          if args.size == 1 then go fuel args[0]! else none
  go 16 regionTerm.raw

structure EOut where
  term : TSyntax `term
  dtype : DInfo
  shape : SInfo
  computeTerm : Option (TSyntax `term) := none
  deriving Inhabited

def expandLeanAntiquoteAs? (dtype : DInfo) (e : TSyntax `tritonExpr) :
    MacroM (Option EOut) := do
  match e with
  | `(tritonExpr| $($t:term)) =>
      let out ←
        match dtype with
        | .real => pure (← `(Op.const $t), .real)
        | .nat => pure (← `(Op.constNat $t), .nat)
        | .int => pure (← `(Op.constInt $t), .int)
        | .bool => pure (← `(Op.constBool $t), .bool)
        | _ =>
            Macro.throwError
              "$(...): Lean antiquotation can only be inferred as real/nat/int/bool in the current context"
      pure (some ⟨out.1, out.2, SInfo.scalar, none⟩)
  | _ => pure none

def expandLeanAntiquoteAs (dtype : DInfo) (e : TSyntax `tritonExpr) :
    MacroM EOut := do
  match ← expandLeanAntiquoteAs? dtype e with
  | some out => pure out
  | none => Macro.throwError "$(...): expected a Lean antiquotation"

def realMathTerm (ctx : String) (e : EOut) : MacroM (TSyntax `term) := do
  match e.dtype with
  | .real => pure e.term
  | .fp32 | .fp16 | .bf16 =>
      let srcProof ← e.dtype.floatProof
      pure (← `(Op.castFloat $srcProof FloatDType.real $e.term))
  | _ =>
      ensureDType .real e.dtype ctx
      pure e.term

def coerceRealArithOperands (ctx : String) (a b : EOut) : MacroM (EOut × EOut) := do
  match a.dtype, b.dtype with
  | .real, .fp32 | .real, .fp16 | .real, .bf16 =>
      pure (a, { b with term := ← realMathTerm (ctx ++ " rhs") b, dtype := .real })
  | .fp32, .real | .fp16, .real | .bf16, .real =>
      pure ({ a with term := ← realMathTerm (ctx ++ " lhs") a, dtype := .real }, b)
  | _, _ =>
      pure (a, b)

inductive CInfo where
  | uint32
  | int32
  | fp32
  deriving BEq

structure StaticPtrOut where
  region : TSyntax `term
  offset : TSyntax `term
  shape : SInfo
  baseOnly : Bool

abbrev ExprExpander := Env → TSyntax `tritonExpr → MacroM EOut

abbrev StaticPtrExpander := Env → TSyntax `tritonExpr → MacroM (Option StaticPtrOut)

abbrev StmtExpansion := TSyntax `term × TSyntax `term × Env × Bool

def methodCast? (stx : TSyntax `tritonExpr) :
    Option (TSyntax `tritonExpr × TSyntax `tritonDType) :=
  let k := stx.raw.getKind
  if k == ``tritonMethodCast then
    let args := stx.raw.getArgs
    if h : args.size = 5 then
      some (⟨args[0]⟩, ⟨args[3]⟩)
    else if h : args.size = 6 then
      some (⟨args[0]⟩, ⟨args[4]⟩)
    else
      none
  else
    none

def isRealDType : DInfo → Bool
  | .real => Bool.true
  | _ => Bool.false

def natDimTerm (ctx : String) (d : TSyntax `tritonExpr) :
    MacroM (TSyntax `term) := do
  match d with
  | `(tritonExpr| $($t:term)) => `(($t : Nat))
  | `(tritonExpr| $n:num) => `(($n : Nat))
  | _ => Macro.throwError (ctx ++ ": each entry must be a numeric literal or `$(t)` antiquote")

def natListTerm (ctx : String) (dims : Array (TSyntax `tritonExpr)) :
    MacroM (TSyntax `term × List (TSyntax `term)) := do
  let mut dimTerms : Array (TSyntax `term) := #[]
  for d in dims do
    dimTerms := dimTerms.push (← natDimTerm ctx d)
  let rec go : List (TSyntax `term) → MacroM (TSyntax `term)
    | [] => `(([] : List Nat))
    | d :: rest => do
        let tail ← go rest
        `($d :: $tail)
  pure (← go dimTerms.toList, dimTerms.toList)

def paddingOptionTerm (s : String) : MacroM (TSyntax `term) := do
  match s with
  | "zero" => `(PaddingOption.zero)
  | other => Macro.throwError ("padding_option: unsupported value `" ++ other ++ "`; only \"zero\" is modeled")

def uint32BitsTerm (bits : Nat) : MacroM (TSyntax `term) := do
  let bitsNat : TSyntax `term := ⟨Syntax.mkNumLit (toString bits)⟩
  `(({ bits := BitVec.ofNat 32 $bitsNat } : UInt32Bits))

def int32BitsTerm (bits : Nat) : MacroM (TSyntax `term) := do
  let bitsNat : TSyntax `term := ⟨Syntax.mkNumLit (toString bits)⟩
  `(({ bits := BitVec.ofNat 32 $bitsNat } : Int32Bits))

def float32BitsTerm (bits : Nat) : MacroM (TSyntax `term) := do
  let bitsNat : TSyntax `term := ⟨Syntax.mkNumLit (toString bits)⟩
  `(({ bits := BitVec.ofNat 32 $bitsNat } : Float32Bits))

def CInfo.term : CInfo → MacroM (TSyntax `term)
  | .uint32 => `(ComputeDType.uint32)
  | .int32 => `(ComputeDType.int32)
  | .fp32 => `(ComputeDType.fp32)

def CInfo.algDType : CInfo → DInfo
  | .uint32 => .nat
  | .int32 => .int
  | .fp32 => .real

def CInfo.placeholderAlgTerm (dtype : CInfo) (shape : SInfo) :
    MacroM (TSyntax `term) := do
  let sh ← shape.term
  match dtype with
  | .uint32 => `(Op.ref TileDType.nat $sh "__compute_bitcast_unprojected__")
  | .int32 => `(Op.ref TileDType.int $sh "__compute_bitcast_unprojected__")
  | .fp32 => `(Op.ref TileDType.real $sh "__compute_bitcast_unprojected__")

def expandComputeDType : TSyntax `tritonDType → MacroM CInfo
  | `(tritonDType| tl.uint32) => pure .uint32
  | `(tritonDType| tl.int32) => pure .int32
  | `(tritonDType| tl.float32) => pure .fp32
  | _ =>
      Macro.throwError
        "tl.bitcast: only 32-bit payload dtypes are modeled (`tl.uint32`, `tl.int32`, `tl.float32`)"

def inferComputeSourceDType (ctx : String) (dtype : DInfo) : MacroM CInfo := do
  match dtype with
  | .nat => pure .uint32
  | .int => pure .int32
  | .real => pure .fp32
  | .fp32 => pure .fp32
  | _ =>
      Macro.throwError
        (ctx ++ ": source dtype must erase from a modeled 32-bit payload (`tl.uint32`, `tl.int32`, `tl.float32`)")

def computeLiteralPayloadTerm (dtype : CInfo) (bits : Nat) :
    MacroM (TSyntax `term) := do
  match dtype with
  | .uint32 => uint32BitsTerm bits
  | .int32 => int32BitsTerm bits
  | .fp32 => float32BitsTerm bits

def ensureAlgorithmOnly (ctx : String) (e : EOut) : MacroM Unit := do
  if e.computeTerm.isSome then
    Macro.throwError
      (ctx ++ ": compute-only expressions such as runtime tl.bitcast must be assigned or stored directly before algorithm-layer composition")

/-- Macro-time IEEE 754 binary32 bit-pattern triage. Throws on encodings
that AlgorithmCorrect does not model (subnormal/zero `exp = 0`, NaN/Inf
`exp = 255`), so the algorithm-side `Op.const` term emitted below is always
backed by a `decodeRat` value that elaborates to `some _`. This is the
single point that decides whether a `tl.bitcast` literal is admissible in
AlgorithmCorrect — there is no runtime `Except.error` fallback downstream. -/
def validateFp32BitsForAlg (bits : Nat) : MacroM Unit := do
  let exp := (bits / (2 ^ 23)) % 256
  if exp = 0 then
    Macro.throwError
      "tl.bitcast(..., tl.float32): zero/subnormal fp32 (exponent field 0) is not modeled in AlgorithmCorrect"
  if exp = 255 then
    Macro.throwError
      "tl.bitcast(..., tl.float32): NaN/Inf fp32 (exponent field 255) is not modeled in AlgorithmCorrect"

/-- Emit the algorithm-side `Op.const` term for a validated fp32 bit pattern.

`Float32Bits.decodeRat` is the single authoritative decoder; the elaborator
reduces it on the concrete `BitVec 32` literal. The `Option.get` proof
obligation `(by decide)` is a defense-in-depth check: if `validateFp32BitsForAlg`
ever lets through an `exp ∈ {0, 255}` pattern, elaboration fails loudly here
rather than silently substituting a placeholder constant. -/
def fp32ConstFromBitsTerm (bits : Nat) : MacroM (TSyntax `term) := do
  let bitsNat : TSyntax `term := ⟨Syntax.mkNumLit (toString bits)⟩
  `(Op.const
      ((((Float32Bits.decodeRat
            { bits := BitVec.ofNat 32 $bitsNat }).get (by decide)) : ℝ)))

def computeLiteralAlgTerm (dtype : CInfo) (bits : Nat) :
    MacroM (TSyntax `term) := do
  match dtype with
  | .uint32 =>
      let payload ← uint32BitsTerm bits
      `(Op.constNat ($payload).toNat)
  | .int32 =>
      let payload ← int32BitsTerm bits
      `(Op.constInt ($payload).toInt)
  | .fp32 =>
      fp32ConstFromBitsTerm bits

end VeriTile.Triton.DSL
