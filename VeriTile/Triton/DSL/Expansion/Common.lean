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
          some stx.getId.eraseMacroScopes.toString
        else
          let args := stx.getArgs
          if args.size == 1 then go fuel args[0]! else none
  go 16 regionTerm.raw

structure EOut where
  term : TSyntax `term
  dtype : DInfo
  shape : SInfo
  computeTerm : Option (TSyntax `term) := none
  computeDType? : Option DInfo := none
  deriving Inhabited

/-- Recover the structural floating expression retained by an elaborated operand. -/
def EOut.floatingTerm (e : EOut) : MacroM (TSyntax `term) := do
  match e.computeTerm with
  | none => pure e.term
  | some ce =>
      match ce with
      | `(ComputeExpr.alg $op) => pure op
      | `(ComputeExpr.compute (ComputeOp.alg $_ $op)) => pure op
      | `(ComputeExpr.compute (ComputeOp.load $_ $mem $mask $padding)) =>
          `(Op.load TileDType.real $mem $mask $padding)
      | `(ComputeExpr.compute (ComputeOp.full $shape (ComputeOp.alg $_ $value))) =>
          `(Op.full $shape $value)
      | _ => Macro.throwError "floating composition requires a structural floating expression"

/-- Preserve an operand's local arithmetic context before enclosing promotion.
Unannotated scalar literals acquire the enclosing operation's precision. -/
def EOut.scopedFloatingTerm (e : EOut) : MacroM (TSyntax `term) := do
  let op ← e.floatingTerm
  if e.computeDType?.isNone && e.computeTerm.isNone then
    match e.term with
    | `(Op.const $_) | `(Op.constFloat $_ $_) => return op
    | _ => pure ()
  let dtype ← e.dtype.floatProof
  let precision ← if e.computeDType? == some .real then
    `(IndexComputePrecision.fp64) else `(IndexComputePrecision.fp32)
  `(Op.withPrecision $dtype $precision $op)

/-- Remove only precision scopes equal to the enclosing explicit context.
The normalization also runs over the final emitted compute statement tree. -/
partial def normalizePrecisionScopes (precision : DInfo) (raw : Syntax) : MacroM Syntax := do
  let term : TSyntax `term := ⟨raw⟩
  match term with
  | `(Op.withPrecision $dtype IndexComputePrecision.fp32 $op) =>
      let child : TSyntax `term := ⟨← normalizePrecisionScopes .fp32 op.raw⟩
      if precision == .fp32 then return child.raw
      return (← `(Op.withPrecision $dtype IndexComputePrecision.fp32 $child)).raw
  | `(Op.withPrecision $dtype IndexComputePrecision.fp64 $op) =>
      let child : TSyntax `term := ⟨← normalizePrecisionScopes .real op.raw⟩
      if precision == .real then return child.raw
      return (← `(Op.withPrecision $dtype IndexComputePrecision.fp64 $child)).raw
  | `(ComputeOp.alg ComputeDType.fp32 $op) =>
      let child : TSyntax `term := ⟨← normalizePrecisionScopes .fp32 op.raw⟩
      return (← `(ComputeOp.alg ComputeDType.fp32 $child)).raw
  | `(ComputeOp.alg ComputeDType.fp64 $op) =>
      let child : TSyntax `term := ⟨← normalizePrecisionScopes .real op.raw⟩
      return (← `(ComputeOp.alg ComputeDType.fp64 $child)).raw
  | _ =>
      match raw with
      | .node info kind args =>
          return .node info kind (← args.mapM (normalizePrecisionScopes precision))
      | other => return other

/-- An explicit cast has a destination compute context and preserves the
source expression's independent context. Narrow destinations use fp32 profile;
the destination FloatDType still determines their rounding format. -/
def explicitFloatCast (dst : DInfo) (e : EOut) : MacroM EOut := do
  let srcProof ← e.dtype.floatProof
  let dstAlg := if dst == .fp32 then DInfo.real else dst
  let dstProof ← dstAlg.floatProof
  let alg ← `(Op.castFloat $srcProof $dstProof $e.term)
  let child ← e.scopedFloatingTerm
  let fp ← `(Op.castFloat $srcProof $dstProof $child)
  match dst with
  | .real => return ⟨alg, .real, e.shape,
      some (← `(ComputeExpr.compute (ComputeOp.alg ComputeDType.fp64 $fp))), some .real⟩
  | .fp32 => return ⟨alg, .real, e.shape,
      some (← `(ComputeExpr.compute (ComputeOp.alg ComputeDType.fp32 $fp))), some .fp32⟩
  | _ =>
      let normalized : TSyntax `term := ⟨← normalizePrecisionScopes .fp32 fp.raw⟩
      let ct ← if normalized.raw == alg.raw then pure none
        else pure (some (← `(ComputeExpr.alg $normalized)))
      return ⟨alg, dst, e.shape, ct, none⟩

/-- Forget storage width when entering the mathematical integer arithmetic
channel. In particular, its results cannot later impersonate byte loads. -/
def EOut.forgetStorageWidth (e : EOut) : EOut :=
  if e.dtype == .int8 then { e with dtype := .int } else e

def expandLeanAntiquoteAs? (dtype : DInfo) (e : TSyntax `tritonExpr) :
    MacroM (Option EOut) := do
  match e with
  | `(tritonExpr| $($t:term)) =>
      let out ←
        match dtype with
        | .real => pure (← `(Op.const $t), .real)
        | .fp32 => pure (← `(Op.constFloat FloatDType.fp32 $t), .fp32)
        | .fp16 => pure (← `(Op.constFloat FloatDType.fp16 $t), .fp16)
        | .bf16 => pure (← `(Op.constFloat FloatDType.bf16 $t), .bf16)
        | .f8e4 => pure (← `(Op.constFloat FloatDType.f8e4 $t), .f8e4)
        | .f8e5 => pure (← `(Op.constFloat FloatDType.f8e5 $t), .f8e5)
        | .nat => pure (← `(Op.constNat $t), .nat)
        | .int => pure (← `(Op.constInt $t), .int)
        | .bool => pure (← `(Op.constBool $t), .bool)
        | _ =>
            Macro.throwError
              "$(...): Lean antiquotation can only be inferred as real/nat/int/bool/float in the current context"
      pure (some ⟨out.1, out.2, SInfo.scalar, none, none⟩)
  | _ => pure none

def expandLeanAntiquoteAs (dtype : DInfo) (e : TSyntax `tritonExpr) :
    MacroM EOut := do
  match ← expandLeanAntiquoteAs? dtype e with
  | some out => pure out
  | none => Macro.throwError "$(...): expected a Lean antiquotation"

def realMathTerm (ctx : String) (e : EOut) : MacroM (TSyntax `term) := do
  match e.dtype with
  | .real => pure e.term
  | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 | .floatVar _ =>
      let srcProof ← e.dtype.floatProof
      pure (← `(Op.castFloat $srcProof FloatDType.real $e.term))
  | _ =>
      ensureDType .real e.dtype ctx
      pure e.term

/-- Numerical widening for explicitly typed floating operands. -/
def coerceFloatOperand (dst : DInfo) (e : EOut) : MacroM EOut := do
  if e.dtype == dst then return e
  let srcProof ← e.dtype.floatProof
  let dstProof ← dst.floatProof
  let alg ← `(Op.castFloat $srcProof $dstProof $e.term)
  let child ← e.scopedFloatingTerm
  let fp ← `(Op.castFloat $srcProof $dstProof $child)
  return { e with term := alg, dtype := dst, computeTerm := some (← `(ComputeExpr.alg $fp)) }

def coerceRealArithOperands (ctx : String) (a b : EOut) : MacroM (EOut × EOut) := do
  let a := a.forgetStorageWidth
  let b := b.forgetStorageWidth
  match a.dtype, b.dtype with
  | .real, .nat =>
      pure (a, { b with term := ← `(Op.natToReal $b.term), dtype := .real })
  | .nat, .real =>
      pure ({ a with term := ← `(Op.natToReal $a.term), dtype := .real }, b)
  | .real, .int =>
      pure (a, { b with term := ← `(Op.intToReal $b.term), dtype := .real })
  | .int, .real =>
      pure ({ a with term := ← `(Op.intToReal $a.term), dtype := .real }, b)
  | .real, .fp32 | .real, .fp16 | .real, .bf16 | .real, .f8e4 | .real, .f8e5 | .real, .floatVar _ =>
      let promoted ← coerceFloatOperand .real b
      pure (a, { promoted with computeDType? := a.computeDType? })
  | .fp32, .real | .fp16, .real | .bf16, .real | .f8e4, .real | .f8e5, .real | .floatVar _, .real =>
      let promoted ← coerceFloatOperand .real a
      pure ({ promoted with computeDType? := b.computeDType? }, b)
  | .fp32, .fp16 | .fp32, .bf16 | .fp32, .f8e4 | .fp32, .f8e5 =>
      pure (a, ← coerceFloatOperand .fp32 b)
  | .fp16, .fp32 | .bf16, .fp32 | .f8e4, .fp32 | .f8e5, .fp32 =>
      pure (← coerceFloatOperand .fp32 a, b)
  | .fp16, .bf16 | .fp16, .f8e4 | .fp16, .f8e5 =>
      pure (a, ← coerceFloatOperand .fp16 b)
  | .bf16, .fp16 | .f8e4, .fp16 | .f8e5, .fp16 =>
      pure (← coerceFloatOperand .fp16 a, b)
  | .bf16, .f8e4 | .bf16, .f8e5 | .f8e4, .bf16 | .f8e5, .bf16 =>
      pure (← coerceFloatOperand .fp32 a, ← coerceFloatOperand .fp32 b)
  | .f8e4, .f8e5 | .f8e5, .f8e4 =>
      pure (← coerceFloatOperand .fp16 a, ← coerceFloatOperand .fp16 b)
  | _, _ =>
      pure (a, b)

def coerceNatIntOperands (_ctx : String) (a b : EOut) : MacroM (EOut × EOut) := do
  let a := a.forgetStorageWidth
  let b := b.forgetStorageWidth
  match a.dtype, b.dtype with
  | .nat, .int =>
      pure ({ a with term := ← `(Op.castNatToInt $a.term), dtype := .int }, b)
  | .int, .nat =>
      pure (a, { b with term := ← `(Op.castNatToInt $b.term), dtype := .int })
  | _, _ =>
      pure (a, b)

inductive CInfo where
  | uint32
  | int32
  | fp32
  deriving BEq

structure StaticPtrOut where
  region : TSyntax `term
  regionDType? : Option DInfo := none
  offset : TSyntax `term
  shape : SInfo
  baseOnly : Bool

abbrev ExprExpander := Env → TSyntax `tritonExpr → MacroM EOut

abbrev StaticPtrExpander := Env → TSyntax `tritonExpr → MacroM (Option StaticPtrOut)

abbrev StmtExpansion := TSyntax `term × TSyntax `term × Env × Bool

def dottedToBaseName? (i : TSyntax `ident) : Option String :=
  let name := i.getId.eraseMacroScopes.toString
  if name.endsWith ".to" then
    some (name.dropEnd 3).toString
  else
    none

def dottedToBaseIdent? (i : TSyntax `ident) : Option (TSyntax `tritonExpr) :=
  match dottedToBaseName? i with
  | some base =>
    some ⟨mkIdentFrom i (Name.mkSimple base)⟩
  | none => none

/-- The final four children are dtype, comma, bitcast=True, close.
Using this suffix handles both the dotted identifier and expression forms. -/
def methodBitcast? (stx : TSyntax `tritonExpr) :
    Option (TSyntax `tritonExpr × TSyntax `tritonDType) :=
  let k := stx.raw.getKind
  if k == ``tritonMethodBitcast || k == ``tritonIdentMethodBitcast ||
      k == ``tritonIdentMethodBitcastSpaced || k == ``tritonDottedIdentMethodBitcast then
    let args := stx.raw.getArgs
    if h : 4 ≤ args.size then
      let dt : TSyntax `tritonDType := ⟨args[args.size - 4]⟩
      if k == ``tritonDottedIdentMethodBitcast then
        (dottedToBaseIdent? ⟨args[0]⟩).map (fun e => (e, dt))
      else some (⟨args[0]⟩, dt)
    else none
  else none

def methodCast? (stx : TSyntax `tritonExpr) :
    Option (TSyntax `tritonExpr × TSyntax `tritonDType) :=
  let k := stx.raw.getKind
  if k == ``tritonMethodCast || k == ``tritonIdentMethodCast ||
      k == ``tritonIdentMethodCastSpaced then
    let args := stx.raw.getArgs
    if h : args.size = 5 then
      some (⟨args[0]⟩, ⟨args[3]⟩)
    else if h : args.size = 6 then
      some (⟨args[0]⟩, ⟨args[4]⟩)
    else
      none
  else if k == ``tritonDottedIdentMethodCast then
    let args := stx.raw.getArgs
    if h : args.size = 4 then
      let i : TSyntax `ident := ⟨args[0]⟩
      match dottedToBaseIdent? i with
      | some e => some (e, ⟨args[2]⟩)
      | none => none
    else
      none
  else
    none

/-- Extractor for the `.round_to(tl.floatXX)` rounding-channel cast — the
`roundCast` sibling of `methodCast?` (same argument layouts). -/
def roundCast? (stx : TSyntax `tritonExpr) :
    Option (TSyntax `tritonExpr × TSyntax `tritonDType) :=
  let k := stx.raw.getKind
  if k == ``tritonMethodRoundCast || k == ``tritonIdentMethodRoundCast ||
      k == ``tritonIdentMethodRoundCastSpaced then
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

partial def natDimTerm (ctx : String) (d : TSyntax `tritonExpr) :
    MacroM (TSyntax `term) := do
  match d with
  | `(tritonExpr| $($t:term)) => `(($t : Nat))
  | `(tritonExpr| $n:num) => `(($n : Nat))
  | `(tritonExpr| $i:ident) =>
      let t : TSyntax `term := ⟨i.raw⟩
      `(($t : Nat))
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => do
      let aTerm ← natDimTerm ctx a
      let bTerm ← natDimTerm ctx b
      `(($aTerm / $bTerm : Nat))
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

/-- Like `natListTerm` but produces a `List Int` whose entries are the
non-negative deltas coerced from `Nat` (`↑d`). Used for `tl.advance` deltas:
`BlockPtr.advance` takes signed `Int` deltas, but the common case is
non-negative, and `↑Nat`-coerced deltas keep the `advance_2d_offsets` normal
form (`offset + delta`) that downstream proofs expect. -/
def intCoeListTerm (ctx : String) (dims : Array (TSyntax `tritonExpr)) :
    MacroM (TSyntax `term) := do
  let mut dimTerms : Array (TSyntax `term) := #[]
  for d in dims do
    dimTerms := dimTerms.push (← natDimTerm ctx d)
  let rec go : List (TSyntax `term) → MacroM (TSyntax `term)
    | [] => `(([] : List Int))
    | d :: rest => do
        let tail ← go rest
        `((((($d : Nat) : Int))) :: $tail)
  go dimTerms.toList

def paddingOptionTerm (s : String) : MacroM (TSyntax `term) := do
  match s with
  | "" => `(PaddingOption.unspecified)
  | "zero" => `(PaddingOption.zero)
  | other => Macro.throwError ("padding_option: unsupported value `" ++ other ++ "`; only empty or zero padding is modeled")

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

def fp32ComputeExpr (term : TSyntax `term) : MacroM (TSyntax `term) :=
  `(ComputeExpr.compute (ComputeOp.alg ComputeDType.fp32 $term))

/-- In compute annotations, `.real` denotes an explicit `tl.float64` target;
`none` continues to mean that no compute precision has been specified. -/
def fp64ComputeExpr (term : TSyntax `term) : MacroM (TSyntax `term) :=
  `(ComputeExpr.compute (ComputeOp.alg ComputeDType.fp64 $term))

def fp32ComputeLoadExpr (mem mask padding : TSyntax `term) : MacroM (TSyntax `term) :=
  `(ComputeExpr.compute (ComputeOp.load ComputeDType.fp32 $mem $mask $padding))

def computeAnnotated? (dtype : DInfo) (term : TSyntax `term) :
    MacroM (Option (TSyntax `term) × Option DInfo) := do
  match dtype with
  | .fp32 => pure (some (← fp32ComputeExpr term), some .fp32)
  | _ => pure (none, none)

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
that AlgorithmCorrect_without_Rounding does not model (subnormal/zero `exp = 0`, NaN/Inf
`exp = 255`), so the algorithm-side `Op.const` term emitted below is always
backed by a `decodeRat` value that elaborates to `some _`. This is the
single point that decides whether a `tl.bitcast` literal is admissible in
AlgorithmCorrect_without_Rounding — there is no runtime `Except.error` fallback downstream. -/
def validateFp32BitsForAlg (bits : Nat) : MacroM Unit := do
  let exp := (bits / (2 ^ 23)) % 256
  if exp = 0 then
    Macro.throwError
      "tl.bitcast(..., tl.float32): zero/subnormal fp32 (exponent field 0) is not modeled in AlgorithmCorrect_without_Rounding"
  if exp = 255 then
    Macro.throwError
      "tl.bitcast(..., tl.float32): NaN/Inf fp32 (exponent field 255) is not modeled in AlgorithmCorrect_without_Rounding"

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
