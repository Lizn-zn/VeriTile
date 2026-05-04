/-
VeriTile.Triton.DSL.Expansion

A `triton { ... }` macro that embeds Triton-style kernel syntax inside Lean,
modelled on the `arm64 { ... }` style used in arm-in-lean.

Example:

  def naiveSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : ComputeKernel := triton {
    pid  := tl.program_id(0)
    offs := pid * $(N) + tl.arange($(N))
    x    := tl.load($(xReg) + offs)
    e    := tl.exp(x)
    s    := tl.sum(e)
    y    := e / s
    tl.store($(yReg) + offs, y)
  }

Conventions:
  * Bare identifiers in expression position → register references
    (`pid`, `x`, `e`, etc. become `Op.ref "pid"`, `Op.ref "x"`, ...).
  * Memory accesses use pointer-like kernel parameters:
    `$(xReg) + offs` builds a pointer tile in pointer context, and
    `tl.load(ptr)` / `tl.store(ptr, value)` consume pointer-valued expressions.
  * `$(<lean-term>)` antiquotes a Lean-level value; in numeric context it
    becomes `Op.constNat`, inside `tl.arange(...)` it is fed directly as
    the `Nat` length; in pointer context (for example `tl.load($(REGION) + i)`)
    it is used as a `RegionName` pointer base.
  * `$ℝ(<lean-term>)` antiquotes a Lean-level `ℝ` value into `Op.const`.
    Symmetric with `$(t)` (which always lowers to `Op.constNat`). Used
    when a kernel parameter has type `ℝ` (e.g. LayerNorm's `ε : ℝ`) and
    must appear in the kernel body. Bare numeric literals (`0`, `1`,
    `2.5`) still go to `Op.const` directly; this form is only needed
    for non-literal `ℝ` terms.
  * Numeric literals become `Op.const`.
  * Statements are separated by newlines (no explicit terminator).

Currently supported expressions: `tl.program_id(_)`, `tl.arange(_)` /
`tl.arange(start, end)`, `tl.exp(_)`, `tl.exp2(_)`, `tl.log(_)`, `tl.log2(_)`,
`tl.sigmoid(_)`, `tl.sqrt(_)`, `tl.tanh(_)`, `tl.sin(_)`, `tl.cos(_)`,
`tl.tan(_)`, `tl.atan(_)`, `tl.cosh(_)`, `tl.sinh(_)`, `tl.max(_)`,
`tl.sum(_)`, `tl.load(ptrExpr)`,
binary `+ - * /`, parens, identifiers, numerals, antiquotation (`$(t)` for
`Nat` in numeric context / `RegionName` in pointer context, `$ℝ(t)` for `ℝ`).
Pointer arithmetic supports `ptr + nat` and `nat + ptr`.

The two-argument `tl.arange(start, end)` lowers to `start + tl.arange(end - start)`
at macro time (no new AST constructor). The literal-0 special case
`tl.arange(0, e)` collapses to `tl.arange(e)`, producing an AST identical to the
single-argument form so existing proofs (e.g. via `scatter_readback`) remain
applicable verbatim.

Currently supported statements: assignment (`name := expr`),
`tl.store(ptrExpr, value)`.

`Kernel.inputs` / `Kernel.outputs` are auto-populated by scanning the body for
`tl.load(...)` (input regions) and `tl.store(...)` (output regions). Order
follows body occurrence; no macro-time dedup since regions are Lean terms
(possibly equal at runtime but not statically). `Kernel.inputs/outputs` is
metadata only (not consumed by `exec`), so duplicates are harmless.

This module owns macro expansion only. Surface syntax, macro-time typing
helpers, and metadata scanning live in sibling `DSL.*` modules.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.DSL.Metadata
import VeriTile.Triton.DSL.Syntax
import VeriTile.Triton.DSL.Typing

open Lean

namespace VeriTile.Triton.DSL

/-! ## Expansion -/

private def identAsStr (i : TSyntax `ident) : MacroM (TSyntax `term) :=
  pure (Syntax.mkStrLit i.getId.toString)

private structure EOut where
  term : TSyntax `term
  dtype : DInfo
  shape : SInfo
  computeTerm : Option (TSyntax `term) := none
  deriving Inhabited

private inductive CInfo where
  | uint32
  | int32
  | fp32
  deriving BEq

private structure StaticPtrOut where
  region : TSyntax `term
  offset : TSyntax `term
  shape : SInfo
  baseOnly : Bool

private def methodCast? (stx : TSyntax `tritonExpr) :
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

private def isRealDType : DInfo → Bool
  | .real => Bool.true
  | _ => Bool.false

private def natDimTerm (ctx : String) (d : TSyntax `tritonExpr) :
    MacroM (TSyntax `term) := do
  match d with
  | `(tritonExpr| $($t:term)) => `(($t : Nat))
  | `(tritonExpr| $n:num) => `(($n : Nat))
  | _ => Macro.throwError (ctx ++ ": each entry must be a numeric literal or `$(t)` antiquote")

private def natListTerm (ctx : String) (dims : Array (TSyntax `tritonExpr)) :
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

private def paddingOptionTerm (s : String) : MacroM (TSyntax `term) := do
  match s with
  | "zero" => `(PaddingOption.zero)
  | other => Macro.throwError ("padding_option: unsupported value `" ++ other ++ "`; only \"zero\" is modeled")

private def uint32BitsTerm (bits : Nat) : MacroM (TSyntax `term) := do
  let bitsNat : TSyntax `term := ⟨Syntax.mkNumLit (toString bits)⟩
  `(({ bits := BitVec.ofNat 32 $bitsNat } : UInt32Bits))

private def int32BitsTerm (bits : Nat) : MacroM (TSyntax `term) := do
  let bitsNat : TSyntax `term := ⟨Syntax.mkNumLit (toString bits)⟩
  `(({ bits := BitVec.ofNat 32 $bitsNat } : Int32Bits))

private def float32BitsTerm (bits : Nat) : MacroM (TSyntax `term) := do
  let bitsNat : TSyntax `term := ⟨Syntax.mkNumLit (toString bits)⟩
  `(({ bits := BitVec.ofNat 32 $bitsNat } : Float32Bits))

private def CInfo.term : CInfo → MacroM (TSyntax `term)
  | .uint32 => `(ComputeDType.uint32)
  | .int32 => `(ComputeDType.int32)
  | .fp32 => `(ComputeDType.fp32)

private def CInfo.algDType : CInfo → DInfo
  | .uint32 => .nat
  | .int32 => .int
  | .fp32 => .real

private def CInfo.placeholderAlgTerm (dtype : CInfo) (shape : SInfo) :
    MacroM (TSyntax `term) := do
  let sh ← shape.term
  match dtype with
  | .uint32 => `(Op.ref TileDType.nat $sh "__compute_bitcast_unprojected__")
  | .int32 => `(Op.ref TileDType.int $sh "__compute_bitcast_unprojected__")
  | .fp32 => `(Op.ref TileDType.real $sh "__compute_bitcast_unprojected__")

private def expandComputeDType : TSyntax `tritonDType → MacroM CInfo
  | `(tritonDType| tl.uint32) => pure .uint32
  | `(tritonDType| tl.int32) => pure .int32
  | `(tritonDType| tl.float32) => pure .fp32
  | _ =>
      Macro.throwError
        "tl.bitcast: only 32-bit payload dtypes are modeled (`tl.uint32`, `tl.int32`, `tl.float32`)"

private def inferComputeSourceDType (ctx : String) (dtype : DInfo) : MacroM CInfo := do
  match dtype with
  | .nat => pure .uint32
  | .int => pure .int32
  | .real => pure .fp32
  | .fp32 => pure .fp32
  | _ =>
      Macro.throwError
        (ctx ++ ": source dtype must erase from a modeled 32-bit payload (`tl.uint32`, `tl.int32`, `tl.float32`)")

private def computeLiteralPayloadTerm (dtype : CInfo) (bits : Nat) :
    MacroM (TSyntax `term) := do
  match dtype with
  | .uint32 => uint32BitsTerm bits
  | .int32 => int32BitsTerm bits
  | .fp32 => float32BitsTerm bits

private def ensureAlgorithmOnly (ctx : String) (e : EOut) : MacroM Unit := do
  if e.computeTerm.isSome then
    Macro.throwError
      (ctx ++ ": compute-only expressions such as runtime tl.bitcast must be assigned or stored directly before algorithm-layer composition")

/-- Macro-time IEEE 754 binary32 bit-pattern triage. Throws on encodings
that AlgorithmCorrect does not model (subnormal/zero `exp = 0`, NaN/Inf
`exp = 255`), so the algorithm-side `Op.const` term emitted below is always
backed by a `decodeRat` value that elaborates to `some _`. This is the
single point that decides whether a `tl.bitcast` literal is admissible in
AlgorithmCorrect — there is no runtime `Except.error` fallback downstream. -/
private def validateFp32BitsForAlg (bits : Nat) : MacroM Unit := do
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
private def fp32ConstFromBitsTerm (bits : Nat) : MacroM (TSyntax `term) := do
  let bitsNat : TSyntax `term := ⟨Syntax.mkNumLit (toString bits)⟩
  `(Op.const
      ((((Float32Bits.decodeRat
            { bits := BitVec.ofNat 32 $bitsNat }).get (by decide)) : ℝ)))

private def computeLiteralAlgTerm (dtype : CInfo) (bits : Nat) :
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

mutual

partial def expandStaticPtrExpr (env : Env) (stx : TSyntax `tritonExpr) :
    MacroM (Option StaticPtrOut) := do
  match stx with
  | `(tritonExpr| $($r:term)) =>
      let zero ← `(Op.constNat 0)
      pure (some ⟨r, zero, SInfo.scalar, Bool.true⟩)
  | `(tritonExpr| ($e:tritonExpr)) =>
      expandStaticPtrExpr env e
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
      match ← expandStaticPtrExpr env a with
      | some p =>
          let b' ← expandExpr env b
          ensureDType .nat b'.dtype "pointer offset"
          if p.baseOnly then
            pure (some ⟨p.region, b'.term, b'.shape, Bool.false⟩)
          else
            let (bc, outShape) ← broadcastTerm p.shape b'.shape "pointer offset"
            let off := p.offset
            let bTerm := b'.term
            let nextOff ← `(Op.add NumericDType.nat $bc $off $bTerm)
            pure (some ⟨p.region, nextOff, outShape, Bool.false⟩)
      | none => pure none
  | _ => pure none

partial def expandExpr (env : Env) (stx : TSyntax `tritonExpr) : MacroM EOut := do
  match methodCast? stx with
  | some (e, dt) =>
      let e' ← expandExpr env e
      let dst ← expandDType dt
      let srcProof ← e'.dtype.floatProof
      let dstProof ← dst.floatProof
      pure ⟨← `(Op.castFloat $srcProof $dstProof $e'.term), dst, e'.shape, none⟩
  | none =>
  match stx with
  | `(tritonExpr| $n:num) =>
      -- Bare numeric literals are `ℝ` data constants (e.g. `1` in `1 / s`).
      pure ⟨← `(Op.const $n), .real, SInfo.scalar, none⟩
  | `(tritonExpr| $i:ident) =>
      let name := i.getId.toString
      let (dtype, shape) ← lookupEnv env name
      let s ← identAsStr i
      let dt ← dtype.term
      let sh ← shape.term
      pure ⟨← `(Op.ref $dt $sh $s), dtype, shape, none⟩
  | `(tritonExpr| $($t:term)) =>
      -- `$(...)` antiquote is the address/size channel: `Nat`.
      pure ⟨← `(Op.constNat $t), .nat, SInfo.scalar, none⟩
  | `(tritonExpr| $ℝ($t:term)) =>
      -- `$ℝ(...)` antiquote is the data channel: `ℝ`. Symmetric with the
      -- `$(t) → Op.constNat` form, used for non-literal ℝ kernel params
      -- (e.g. LayerNorm's `ε`).
      pure ⟨← `(Op.const $t), .real, SInfo.scalar, none⟩
  | `(tritonExpr| ($e:tritonExpr)) =>
      expandExpr env e
  | `(tritonExpr| tl.program_id($e:tritonExpr)) =>
      -- axis must be a numeric literal or `$(t)` antiquote (Nat)
      let axisTerm : TSyntax `term ← match e with
        | `(tritonExpr| $($t:term)) => `(($t : Nat))
        | `(tritonExpr| $n:num)     => `(($n : Nat))
        | _ => Macro.throwError
                "tl.program_id(axis): axis must be a numeric literal or $(N)"
      pure ⟨← `(Op.programId $axisTerm), .nat, SInfo.scalar, none⟩
  | `(tritonExpr| tl.arange($e:tritonExpr)) =>
      -- arange takes a Nat; recognize $(t) and bare numerals specially
      match e with
      | `(tritonExpr| $($t:term)) => pure ⟨← `(Op.arange $t), .nat, SInfo.vec t, none⟩
      | `(tritonExpr| $n:num)     => pure ⟨← `(Op.arange $n), .nat, SInfo.vec (← `(($n : Nat))), none⟩
      | _ => Macro.throwError
              "tl.arange(...) expects a Lean Nat: either a numeric literal or $(N)"
  | `(tritonExpr| tl.arange($s:tritonExpr, $e:tritonExpr)) => do
      -- Two-argument form: lower to start + arange(end - start). Both args must
      -- be Nat-valued (numeric literal or $(t)).
      let sTerm : TSyntax `term ← match s with
        | `(tritonExpr| $($t:term)) => `(($t : Nat))
        | `(tritonExpr| $n:num)     => `(($n : Nat))
        | _ => Macro.throwError
                "tl.arange(start, end): start must be a numeric literal or $(N)"
      let eTerm : TSyntax `term ← match e with
        | `(tritonExpr| $($t:term)) => `(($t : Nat))
        | `(tritonExpr| $n:num)     => `(($n : Nat))
        | _ => Macro.throwError
                "tl.arange(start, end): end must be a numeric literal or $(N)"
      -- Literal-0 start collapses to single-arg arange so the AST is identical
      -- to `tl.arange(end)` (no `Op.add (Op.const 0) ...` wrapper).
      match s with
      | `(tritonExpr| $n:num) =>
          if n.getNat = 0 then
            pure ⟨← `(Op.arange $eTerm), .nat, SInfo.vec eTerm, none⟩
          else
            pure ⟨← `(Op.add NumericDType.nat Broadcast.scalarL (Op.constNat $sTerm) (Op.arange ($eTerm - $sTerm))),
              .nat, SInfo.vec (← `($eTerm - $sTerm)), none⟩
      | _ =>
          pure ⟨← `(Op.add NumericDType.nat Broadcast.scalarL (Op.constNat $sTerm) (Op.arange ($eTerm - $sTerm))),
            .nat, SInfo.vec (← `($eTerm - $sTerm)), none⟩
  | `(tritonExpr| tl.exp($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.exp"
      pure ⟨← `(Op.exp $e'.term), .real, e'.shape, none⟩
  | `(tritonExpr| tl.exp2($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.exp2"
      pure ⟨← `(Op.exp2 $e'.term), .real, e'.shape, none⟩
  | `(tritonExpr| tl.log($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.log"
      pure ⟨← `(Op.log $e'.term), .real, e'.shape, none⟩
  | `(tritonExpr| tl.log2($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.log2"
      pure ⟨← `(Op.log2 $e'.term), .real, e'.shape, none⟩
  | `(tritonExpr| tl.sigmoid($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.sigmoid"
      pure ⟨← `(Op.sigmoid $e'.term), .real, e'.shape, none⟩
  | `(tritonExpr| tl.sqrt($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.sqrt"
      pure ⟨← `(Op.sqrt $e'.term), .real, e'.shape, none⟩
  | `(tritonExpr| tl.tanh($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.tanh"
      pure ⟨← `(Op.tanh $e'.term), .real, e'.shape, none⟩
  | `(tritonExpr| tl.sin($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.sin"
      pure ⟨← `(Op.sin $e'.term), .real, e'.shape, none⟩
  | `(tritonExpr| tl.cos($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.cos"
      pure ⟨← `(Op.cos $e'.term), .real, e'.shape, none⟩
  | `(tritonExpr| tl.tan($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.tan"
      pure ⟨← `(Op.tan $e'.term), .real, e'.shape, none⟩
  | `(tritonExpr| tl.atan($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.atan"
      pure ⟨← `(Op.atan $e'.term), .real, e'.shape, none⟩
  | `(tritonExpr| tl.cosh($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.cosh"
      pure ⟨← `(Op.cosh $e'.term), .real, e'.shape, none⟩
  | `(tritonExpr| tl.sinh($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.sinh"
      pure ⟨← `(Op.sinh $e'.term), .real, e'.shape, none⟩
  | `(tritonExpr| tl.abs($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.abs"
      let (cmpBc, cmpShape) ← broadcastTerm e'.shape SInfo.scalar "tl.abs"
      ensureShape e'.shape cmpShape "tl.abs"
      let (subBc, subShape) ← broadcastTerm SInfo.scalar e'.shape "tl.abs"
      ensureShape e'.shape subShape "tl.abs"
      let zero ← `(Op.const 0)
      let cond ← `(Op.lt ComparableDType.real $cmpBc $e'.term $zero)
      let neg ← `(Op.sub NumericDType.real $subBc $zero $e'.term)
      pure ⟨← `(Op.where $cond $neg $e'.term), .real, e'.shape, none⟩
  | `(tritonExpr| tl.logical_and($a:tritonExpr, $b:tritonExpr)) => do
      expandBoolBin env "tl.logical_and" (← `(Op.boolAnd)) a b
  | `(tritonExpr| tl.logical_or($a:tritonExpr, $b:tritonExpr)) => do
      expandBoolBin env "tl.logical_or" (← `(Op.boolOr)) a b
  | `(tritonExpr| tl.logical_not($a:tritonExpr)) => do
      expandBoolNot env "tl.logical_not" a
  | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) => do
      expandCdiv env a b
  | `(tritonExpr| tl.max($a:tritonExpr, $b:tritonExpr)) => do
      let a' ← expandExpr env a
      let b' ← expandExpr env b
      ensureDType .real a'.dtype "tl.max"
      ensureDType .real b'.dtype "tl.max"
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "tl.max"
      pure ⟨← `(Op.max2 $bc $a'.term $b'.term), .real, outShape, none⟩
  | `(tritonExpr| tl.maximum($a:tritonExpr, $b:tritonExpr)) => do
      expandMinMax env "tl.maximum" (← `(Op.gt)) a b
  | `(tritonExpr| tl.minimum($a:tritonExpr, $b:tritonExpr)) => do
      expandMinMax env "tl.minimum" (← `(Op.lt)) a b
  | `(tritonExpr| tl.cumsum($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandScan env "tl.cumsum" (← `(ScanOp.sum)) e kwargs
  | `(tritonExpr| tl.cumprod($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandScan env "tl.cumprod" (← `(ScanOp.prod)) e kwargs
  | `(tritonExpr| tl.associative_scan($e:tritonExpr, $op:tritonScanOp $[, $kwargs:tritonReduceKwarg]*)) => do
      expandScan env "tl.associative_scan" (← expandScanOp op) e kwargs
  | `(tritonExpr| tl.argmax($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandArgReduce env "tl.argmax" (← `(Op.argMax)) e kwargs
  | `(tritonExpr| tl.argmin($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandArgReduce env "tl.argmin" (← `(Op.argMin)) e kwargs
  | `(tritonExpr| tl.sort($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandSort env e kwargs
  | `(tritonExpr| tl.sum($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandReduce env "tl.sum" (← `(Op.reduceSum)) e kwargs
  | `(tritonExpr| tl.max($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandReduce env "tl.max" (← `(Op.reduceMax)) e kwargs
  | `(tritonExpr| tl.toReal($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .nat e'.dtype "tl.toReal"
      pure ⟨← `(Op.natToReal $e'.term), .real, e'.shape, none⟩
  | `(tritonExpr| tl.bitcast($e:tritonExpr, $dt:tritonDType)) => do
      let dst ← expandComputeDType dt
      let dstTerm ← dst.term
      match e with
      | `(tritonExpr| $n:num) =>
          let bits := n.getNat
          unless bits < 2^32 do
            Macro.throwError "tl.bitcast(...): literal does not fit 32 bits"
          if dst == .fp32 then
            validateFp32BitsForAlg bits
          let payload ← computeLiteralPayloadTerm .uint32 bits
          let op ←
            `(ComputeOp.bitcast ComputeDType.uint32 $dstTerm rfl
                (ComputeOp.const $payload))
          let algTerm ← computeLiteralAlgTerm dst bits
          pure ⟨algTerm, dst.algDType, SInfo.scalar,
            some (← `(ComputeExpr.compute $op))⟩
      | _ =>
          let e' ← expandExpr env e
          let src ← inferComputeSourceDType "tl.bitcast" e'.dtype
          let srcTerm ← src.term
          let algTerm ← dst.placeholderAlgTerm e'.shape
          let op ←
            `(ComputeOp.bitcast $srcTerm $dstTerm rfl
                (ComputeOp.alg $srcTerm $e'.term))
          pure ⟨algTerm, dst.algDType, e'.shape,
            some (← `(ComputeExpr.compute $op))⟩
  | `(tritonExpr| tl.cast($e:tritonExpr, $dt:tritonDType)) => do
      let e' ← expandExpr env e
      let dst ← expandDType dt
      let srcProof ← e'.dtype.floatProof
      let dstProof ← dst.floatProof
      pure ⟨← `(Op.castFloat $srcProof $dstProof $e'.term), dst, e'.shape, none⟩
  | `(tritonExpr| -inf) =>
      pure ⟨← `(Op.negInf), .real, SInfo.scalar, none⟩
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr)) => do
      expandDot env a b
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr, $acc:tritonExpr)) => do
      -- Fused accumulator form: `tl.dot(a, b, acc) ≡ acc + tl.dot(a, b)`.
      -- Both `tl.dot(a, b)` and `acc` have shape `[M, N]`; their `+` uses
      -- the standard same-shape broadcast.
      let dot ← expandDot env a b
      let acc' ← expandExpr env acc
      ensureDType .real acc'.dtype "tl.dot accumulator"
      ensureShape dot.shape acc'.shape "tl.dot accumulator"
      let (bc, outShape) ← broadcastTerm dot.shape acc'.shape "tl.dot accumulator"
      pure ⟨← `(Op.add NumericDType.real $bc $dot.term $acc'.term),
            .real, outShape, none⟩
  | `(tritonExpr| tl.make_block_ptr($p:tritonExpr, $baseKw:ident=$base:tritonExpr,
        $shapeKw:ident=[$parentDims:tritonExpr,*], $stridesKw:ident=[$strideDims:tritonExpr,*],
        $offsetsKw:ident=[$offsetDims:tritonExpr,*], $blockShapeKw:ident=[$blockDims:tritonExpr,*])) => do
      unless baseKw.getId.toString == "base" &&
          shapeKw.getId.toString == "shape" &&
          stridesKw.getId.toString == "strides" &&
          offsetsKw.getId.toString == "offsets" &&
          blockShapeKw.getId.toString == "block_shape" do
        Macro.throwError "tl.make_block_ptr kwargs must be `base`, `shape`, `strides`, `offsets`, `block_shape` in that order"
      let sp ← expandStaticPtrExpr env p
      let region ← match sp with
        | some sp =>
            unless sp.baseOnly do
              Macro.throwError "tl.make_block_ptr: pointer base must be a region antiquote like `$(xReg)`"
            pure sp.region
        | none =>
            Macro.throwError "tl.make_block_ptr: first argument must be a region antiquote like `$(xReg)`"
      let baseTerm ← natDimTerm "tl.make_block_ptr base" base
      let (parentShape, _) ← natListTerm "tl.make_block_ptr shape" parentDims
      let (strides, _) ← natListTerm "tl.make_block_ptr strides" strideDims
      let (offsets, _) ← natListTerm "tl.make_block_ptr offsets" offsetDims
      let (blockShape, blockShapeInfo) ← natListTerm "tl.make_block_ptr block_shape" blockDims
      pure ⟨← `(Op.makeBlockPtr $region $baseTerm $parentShape $blockShape $strides $offsets),
            .blockPtr, .dims blockShapeInfo, none⟩
  | `(tritonExpr| tl.advance($p:tritonExpr, [$deltas:tritonExpr,*])) => do
      let p' ← expandExpr env p
      ensureDType .blockPtr p'.dtype "tl.advance pointer"
      let (deltasTerm, _) ← natListTerm "tl.advance offsets" deltas
      pure ⟨← `(Op.advanceBlockPtr $p'.term $deltasTerm), .blockPtr, p'.shape, none⟩
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      let mut maskTerm : Option (TSyntax `term × SInfo) := none
      let mut otherTerm : Option (TSyntax `term × DInfo × SInfo) := none
      let mut dtype? : Option DInfo := none
      let mut boundaryCheck? : Option (TSyntax `term) := none
      let mut padding : TSyntax `term ← `(PaddingOption.zero)
      for kw in kwargs do
        match kw with
        | `(tritonMemKwarg| boundary_check=$axes:term) =>
            if boundaryCheck?.isSome then
              Macro.throwError "tl.load: duplicate `boundary_check=` kwarg"
            boundaryCheck? := some axes
        | `(tritonMemKwarg| padding_option="zero") =>
            padding := ← `(PaddingOption.zero)
        | `(tritonMemKwarg| $name:ident = $val:tritonExpr) =>
            let val' ← expandExpr env val
            match name.getId.toString with
            | "mask"  =>
                ensureDType .bool val'.dtype "tl.load mask"
                maskTerm := some (val'.term, val'.shape)
            | "other" =>
                otherTerm := some (val'.term, val'.dtype, val'.shape)
            | unknown =>
                let msg : String :=
                  "tl.load: unknown kwarg `" ++ unknown ++
                  "`. Only `mask`, `other`, `dtype`, `boundary_check`, and `padding_option` are recognized."
                Macro.throwError msg
        | `(tritonMemKwarg| $name:ident = $dt:tritonDType) =>
            unless name.getId.toString == "dtype" do
              Macro.throwError
                ("tl.load: unknown kwarg `" ++ name.getId.toString ++
                 "`. Only `mask`, `other`, `dtype`, `boundary_check`, and `padding_option` are recognized.")
            if dtype?.isSome then
              Macro.throwError "tl.load: duplicate `dtype=` kwarg"
            dtype? := some (← expandDType dt)
        | _ => Macro.throwUnsupported
      let outDType := dtype?.getD .real
      if let some boundaryCheck := boundaryCheck? then
        if maskTerm.isSome || otherTerm.isSome then
          Macro.throwError "tl.load: block-pointer `boundary_check` cannot be combined with `mask` or `other`"
        let p' ← expandExpr env p
        ensureDType .blockPtr p'.dtype "tl.load block pointer"
        let dt ← outDType.term
        return ⟨← `(Op.load $dt (MemAccess.blockPtr $p'.term $boundaryCheck) MaskOpt.none),
          outDType, p'.shape, none⟩
      if let some (_, otherDType, _) := otherTerm then
        unless otherDType == outDType do
          Macro.throwError "tl.load other: dtype must match load result dtype"
      match maskTerm, otherTerm with
      | none, none =>
          match ← expandStaticPtrExpr env p with
          | some sp =>
              let r := sp.region
              let off := sp.offset
              let dt ← outDType.term
              pure ⟨← `(Op.load $dt (MemAccess.region $r $off) MaskOpt.none), outDType, sp.shape, none⟩
          | none =>
              let p' ← expandExpr env p
              ensureDType .ptr p'.dtype "tl.load pointer"
              let dt ← outDType.term
              pure ⟨← `(Op.load $dt (MemAccess.ptr $p'.term) MaskOpt.none), outDType, p'.shape, none⟩
      | some (m, mShape), none =>
          match ← expandStaticPtrExpr env p with
          | some sp =>
              let r := sp.region
              let off := sp.offset
              let m' ← coerceShape m mShape sp.shape "tl.load mask"
              let dt ← outDType.term
              pure ⟨← `(Op.load $dt (MemAccess.region $r $off) (MaskOpt.mask $m')), outDType, sp.shape, none⟩
          | none =>
              let p' ← expandExpr env p
              ensureDType .ptr p'.dtype "tl.load pointer"
              let m' ← coerceShape m mShape p'.shape "tl.load mask"
              let dt ← outDType.term
              pure ⟨← `(Op.load $dt (MemAccess.ptr $p'.term) (MaskOpt.mask $m')), outDType, p'.shape, none⟩
      | some (m, mShape), some (o, otherDType, oShape) =>
          match ← expandStaticPtrExpr env p with
          | some sp =>
              let r := sp.region
              let off := sp.offset
              let m' ← coerceShape m mShape sp.shape "tl.load mask"
              let o' ← coerceShape o oShape sp.shape "tl.load other"
              let dt ← otherDType.term
              pure ⟨← `(Op.load $dt (MemAccess.region $r $off) (MaskOpt.maskOther $m' $o')), otherDType, sp.shape, none⟩
          | none =>
              let p' ← expandExpr env p
              ensureDType .ptr p'.dtype "tl.load pointer"
              let m' ← coerceShape m mShape p'.shape "tl.load mask"
              let o' ← coerceShape o oShape p'.shape "tl.load other"
              let dt ← otherDType.term
              pure ⟨← `(Op.load $dt (MemAccess.ptr $p'.term) (MaskOpt.maskOther $m' $o')), otherDType, p'.shape, none⟩
      | none, some _ =>
          Macro.throwError
            "tl.load: `other=` requires `mask=`. (Triton: `other` is meaningful only when some lanes are masked off.)"
  | `(tritonExpr| $a:tritonExpr < $b:tritonExpr) => do
      expandCmp env "comparison" (← `(Op.lt)) a b
  | `(tritonExpr| $a:tritonExpr <= $b:tritonExpr) => do
      expandCmp env "comparison" (← `(Op.le)) a b
  | `(tritonExpr| $a:tritonExpr == $b:tritonExpr) => do
      expandCmp env "comparison" (← `(Op.eq)) a b
  | `(tritonExpr| $a:tritonExpr > $b:tritonExpr) => do
      expandCmp env "comparison" (← `(Op.gt)) a b
  | `(tritonExpr| $a:tritonExpr >= $b:tritonExpr) => do
      expandCmp env "comparison" (← `(Op.ge)) a b
  | `(tritonExpr| $a:tritonExpr != $b:tritonExpr) => do
      expandCmp env "comparison" (← `(Op.ne)) a b
  | `(tritonExpr| $a:tritonExpr & $b:tritonExpr) => do
      expandBoolOrNatBitwise env "`&`" (← `(Op.boolAnd)) (← `(Op.bitAnd)) a b
  | `(tritonExpr| $a:tritonExpr | $b:tritonExpr) => do
      expandBoolOrNatBitwise env "`|`" (← `(Op.boolOr)) (← `(Op.bitOr)) a b
  | `(tritonExpr| $a:tritonExpr ^ $b:tritonExpr) => do
      expandNatBitwise env "`^`" (← `(Op.bitXor)) a b
  | `(tritonExpr| $a:tritonExpr << $b:tritonExpr) => do
      expandNatBitwise env "`<<`" (← `(Op.shiftLeft)) a b
  | `(tritonExpr| $a:tritonExpr >> $b:tritonExpr) => do
      expandNatBitwise env "`>>`" (← `(Op.shiftRight)) a b
  | `(tritonExpr| ~$a:tritonExpr) => do
      expandBoolNot env "boolean ~" a
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
      match ← expandStaticPtrExpr env stx with
      | some sp =>
          let (bc, _) ← broadcastTerm SInfo.scalar sp.shape "pointer arithmetic"
          pure ⟨← `(Op.ptrAdd $bc (Op.ptrBase $sp.region) $sp.offset), .ptr, sp.shape, none⟩
      | none =>
          let a' ← expandExpr env a
          let b' ← expandExpr env b
          ensureAlgorithmOnly "arithmetic" a'
          ensureAlgorithmOnly "arithmetic" b'
          match a'.dtype, b'.dtype with
          | .ptr, .nat =>
              let (bc, outShape) ← broadcastTerm a'.shape b'.shape "pointer arithmetic"
              pure ⟨← `(Op.ptrAdd $bc $a'.term $b'.term), .ptr, outShape, none⟩
          | .nat, .ptr =>
              let (bc, outShape) ← broadcastTerm b'.shape a'.shape "pointer arithmetic"
              pure ⟨← `(Op.ptrAdd $bc $b'.term $a'.term), .ptr, outShape, none⟩
          | _, _ =>
              unless a'.dtype == b'.dtype do
                Macro.throwError "arithmetic: dtype mismatch"
              let np ← a'.dtype.numericProof
              let (bc, outShape) ← broadcastTerm a'.shape b'.shape "arithmetic"
              pure ⟨← `(Op.add $np $bc $a'.term $b'.term), a'.dtype, outShape, none⟩
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) => do
      expandArith env "arithmetic" (← `(Op.sub)) a b
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => do
      expandArith env "arithmetic" (← `(Op.mul)) a b
  | `(tritonExpr| $a:tritonExpr / $b:tritonExpr) => do
      expandArith env "arithmetic" (← `(Op.div)) a b
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => do
      expandIntegralArith env "integer floor division" (← `(Op.floorDiv)) a b
  | `(tritonExpr| $a:tritonExpr % $b:tritonExpr) => do
      expandIntegralArith env "integer remainder" (← `(Op.mod)) a b
  | `(tritonExpr| tl.where($c:tritonExpr, $a:tritonExpr, $b:tritonExpr)) => do
      -- All three must converge to a common shape. To stay aligned with
      -- the same-shape `Op.where` AST node, we accept only scalar →
      -- tile lifts here (via `coerceShape` / `Op.broadcast`). Non-scalar
      -- rank/dim mismatches raise a macro error; the user can lift
      -- through `tl.expand_dims` first.
      let c' ← expandExpr env c
      let a' ← expandExpr env a
      let b' ← expandExpr env b
      ensureDType .bool c'.dtype "tl.where condition"
      unless a'.dtype == b'.dtype do
        Macro.throwError "tl.where: branch dtype mismatch"
      -- Pick target shape: the first non-scalar among the three; require
      -- any other non-scalars to match.
      let nonScalars : List SInfo :=
        ([c'.shape, a'.shape, b'.shape]).filter (fun s =>
          match s with | .dims [] => Bool.false | _ => Bool.true)
      let target : SInfo ←
        match nonScalars with
        | [] => pure (SInfo.dims [])
        | first :: rest =>
            for s in rest do
              unless first.eq s do
                Macro.throwError
                  "tl.where: non-scalar shape mismatch — all tile-shaped operands must agree (lift with tl.expand_dims if needed)"
            pure first
      let cTerm ← coerceShape c'.term c'.shape target "tl.where condition"
      let aTerm ← coerceShape a'.term a'.shape target "tl.where then-branch"
      let bTerm ← coerceShape b'.term b'.shape target "tl.where else-branch"
      pure ⟨← `(Op.where $cTerm $aTerm $bTerm), a'.dtype, target, none⟩
  | `(tritonExpr| $e:tritonExpr[ : , None ]) => do
      -- `e[:, None]` — insert a unit axis at position 1: `[N] → [N, 1]`.
      expandSlicerNone env e (axisIdx := 1)
  | `(tritonExpr| $e:tritonExpr[ None , : ]) => do
      -- `e[None, :]` — insert a unit axis at position 0: `[N] → [1, N]`.
      expandSlicerNone env e (axisIdx := 0)
  | `(tritonExpr| tl.expand_dims($e:tritonExpr, $kw:tritonReduceKwarg)) => do
      match kw with
      | `(tritonReduceKwarg| axis = $n:num) =>
          expandExpandDims env e (axisIdx := n.getNat)
      | `(tritonReduceKwarg| $name:ident = $_) =>
          Macro.throwError
            ("tl.expand_dims: unknown kwarg `" ++ name.getId.toString ++
             "`. Only literal `axis = N` is supported.")
      | _ => Macro.throwUnsupported
  | `(tritonExpr| tl.expand_dims($e:tritonExpr, $n:num)) => do
      expandExpandDims env e (axisIdx := n.getNat)
  | `(tritonExpr| tl.trans($e:tritonExpr)) => do
      -- `tl.trans(e)` — transpose the trailing two axes (`Op.transpose`).
      expandTranspose env e
  | `(tritonExpr| tl.permute($e:tritonExpr, [$axes:num,*])) => do
      expandPermute env e axes.getElems.toList
  | `(tritonExpr| tl.reshape($e:tritonExpr, [$dims:tritonExpr,*])) => do
      expandReshapeLike env "tl.reshape" e dims.getElems
  | `(tritonExpr| tl.view($e:tritonExpr, [$dims:tritonExpr,*])) => do
      expandReshapeLike env "tl.view" e dims.getElems
  | `(tritonExpr| tl.ravel($e:tritonExpr)) => do
      expandRavel env e
  | `(tritonExpr| tl.flip($e:tritonExpr, $ax:num)) => do
      expandFlip env e ax.getNat
  | `(tritonExpr| tl.flip($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      let ax ← parseFlipKwargs kwargs
      expandFlip env e ax
  | `(tritonExpr| tl.join($a:tritonExpr, $b:tritonExpr)) => do
      expandJoin env a b
  | `(tritonExpr| tl.split($e:tritonExpr, $side:num)) => do
      expandSplit env e side.getNat
  | `(tritonExpr| tl.full([$dims:tritonExpr,*], $v:tritonExpr)) => do
      expandFull env dims.getElems v
  | `(tritonExpr| tl.zeros([$dims:tritonExpr,*])) => do
      -- `tl.zeros([dims])` ≡ `tl.full([dims], 0)`.
      let zero ← `(tritonExpr| 0)
      expandFull env dims.getElems zero
  | _ => Macro.throwUnsupported

partial def expandArith (env : Env) (ctx : String) (op : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureAlgorithmOnly ctx a'
  ensureAlgorithmOnly ctx b'
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let np ← a'.dtype.numericProof
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  pure ⟨← `($op $np $bc $a'.term $b'.term), a'.dtype, outShape, none⟩

partial def expandIntegralArith (env : Env) (ctx : String) (op : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureAlgorithmOnly ctx a'
  ensureAlgorithmOnly ctx b'
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let ip ← a'.dtype.integralProof
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  pure ⟨← `($op $ip $bc $a'.term $b'.term), a'.dtype, outShape, none⟩

partial def expandCdiv (env : Env)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureAlgorithmOnly "tl.cdiv" a'
  ensureAlgorithmOnly "tl.cdiv" b'
  ensureDType .nat a'.dtype "tl.cdiv lhs"
  ensureDType .nat b'.dtype "tl.cdiv rhs"
  let (addBc, outShape) ← broadcastTerm a'.shape b'.shape "tl.cdiv"
  let (subBc, subShape) ← broadcastTerm outShape SInfo.scalar "tl.cdiv"
  ensureShape outShape subShape "tl.cdiv"
  let (divBc, divShape) ← broadcastTerm outShape b'.shape "tl.cdiv"
  ensureShape outShape divShape "tl.cdiv"
  let sumTerm ← `(Op.add NumericDType.nat $addBc $a'.term $b'.term)
  let numerator ← `(Op.sub NumericDType.nat $subBc $sumTerm (Op.constNat 1))
  pure ⟨← `(Op.div NumericDType.nat $divBc $numerator $b'.term), .nat, outShape, none⟩

partial def expandBoolBin (env : Env) (ctx : String) (op : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureAlgorithmOnly ctx a'
  ensureAlgorithmOnly ctx b'
  ensureDType .bool a'.dtype ctx
  ensureDType .bool b'.dtype ctx
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  pure ⟨← `($op $bc $a'.term $b'.term), .bool, outShape, none⟩

partial def expandNatBitwise (env : Env) (ctx : String) (op : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureAlgorithmOnly ctx a'
  ensureAlgorithmOnly ctx b'
  ensureDType .nat a'.dtype ctx
  ensureDType .nat b'.dtype ctx
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  pure ⟨← `($op $bc $a'.term $b'.term), .nat, outShape, none⟩

partial def expandBoolOrNatBitwise (env : Env) (ctx : String)
    (boolOp natOp : TSyntax `term) (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureAlgorithmOnly ctx a'
  ensureAlgorithmOnly ctx b'
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  match a'.dtype with
  | .bool => pure ⟨← `($boolOp $bc $a'.term $b'.term), .bool, outShape, none⟩
  | .nat => pure ⟨← `($natOp $bc $a'.term $b'.term), .nat, outShape, none⟩
  | _ =>
      Macro.throwError
        (ctx ++ ": only Bool logical ops and Nat bitwise ops are modeled; signed integer bitwise ops require fixed-width integer semantics")

partial def expandBoolNot (env : Env) (ctx : String)
    (a : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  ensureAlgorithmOnly ctx a'
  ensureDType .bool a'.dtype ctx
  pure ⟨← `(Op.boolNot $a'.term), .bool, a'.shape, none⟩

partial def expandCmp (env : Env) (ctx : String) (op : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureAlgorithmOnly ctx a'
  ensureAlgorithmOnly ctx b'
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let cp ← a'.dtype.comparableProof
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  pure ⟨← `($op $cp $bc $a'.term $b'.term), .bool, outShape, none⟩

partial def expandMinMax (env : Env) (ctx : String) (cmp : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureAlgorithmOnly ctx a'
  ensureAlgorithmOnly ctx b'
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let cp ← a'.dtype.comparableProof
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  let aTerm ← coerceShape a'.term a'.shape outShape (ctx ++ " lhs")
  let bTerm ← coerceShape b'.term b'.shape outShape (ctx ++ " rhs")
  pure ⟨← `(Op.where ($cmp $cp $bc $a'.term $b'.term) $aTerm $bTerm), a'.dtype, outShape, none⟩

partial def expandScanOp : TSyntax `tritonScanOp → MacroM (TSyntax `term)
  | `(tritonScanOp| $name:ident) =>
      match name.getId.toString with
      | "sum" => `(ScanOp.sum)
      | "prod" => `(ScanOp.prod)
      | "max" => `(ScanOp.max)
      | "min" => `(ScanOp.min)
      | other =>
          Macro.throwError
            ("tl.associative_scan: unsupported op `" ++ other ++
             "`. Supported ops: sum, prod, max, min.")
  | _ => Macro.throwUnsupported

partial def expandScan (env : Env) (ctx : String) (op : TSyntax `term)
    (e : TSyntax `tritonExpr)
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM EOut := do
  let e' ← expandExpr env e
  ensureDType .real e'.dtype ctx
  let dims := match e'.shape with | .dims ds => ds
  if dims.isEmpty then
    Macro.throwError (ctx ++ ": rank-≥ 1 input required")
  let mut seenAxis : Bool := Bool.false
  let mut axisIdx : Nat := 0
  for kw in kwargs do
    match kw with
    | `(tritonReduceKwarg| axis = $n:num) =>
        if seenAxis then
          Macro.throwError (ctx ++ ": duplicate `axis=` kwarg")
        seenAxis := Bool.true
        if n.getNat ≥ dims.length then
          Macro.throwError
            (ctx ++ ": axis `" ++ toString n.getNat ++ "` out of bounds for rank "
             ++ toString dims.length)
        axisIdx := n.getNat
    | `(tritonReduceKwarg| keep_dims = false) =>
        Macro.throwError (ctx ++ ": `keep_dims` is not meaningful for prefix scans")
    | `(tritonReduceKwarg| keep_dims = true) =>
        Macro.throwError (ctx ++ ": `keep_dims` is not meaningful for prefix scans")
    | `(tritonReduceKwarg| $name:ident = $_) =>
        Macro.throwError
          (ctx ++ ": unsupported kwarg `" ++ name.getId.toString ++
           "`. Only `axis = N` is supported.")
    | _ =>
        Macro.throwUnsupported
  let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
  pure ⟨← `(Op.scan $op (⟨$axisLit, by simp⟩) $e'.term), .real, e'.shape, none⟩

partial def parseAxisOnlyKwargs (ctx : String) (dims : List (TSyntax `term))
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM Nat := do
  if dims.isEmpty then
    Macro.throwError (ctx ++ ": rank-≥ 1 input required")
  let mut seenAxis : Bool := Bool.false
  let mut axisIdx : Nat := 0
  for kw in kwargs do
    match kw with
    | `(tritonReduceKwarg| axis = $n:num) =>
        if seenAxis then
          Macro.throwError (ctx ++ ": duplicate `axis=` kwarg")
        seenAxis := Bool.true
        if n.getNat ≥ dims.length then
          Macro.throwError
            (ctx ++ ": axis `" ++ toString n.getNat ++ "` out of bounds for rank "
             ++ toString dims.length)
        axisIdx := n.getNat
    | `(tritonReduceKwarg| keep_dims = false) =>
        Macro.throwError (ctx ++ ": `keep_dims` is not supported")
    | `(tritonReduceKwarg| keep_dims = true) =>
        Macro.throwError (ctx ++ ": `keep_dims` is not supported")
    | `(tritonReduceKwarg| $name:ident = $_) =>
        Macro.throwError
          (ctx ++ ": unsupported kwarg `" ++ name.getId.toString ++
           "`. Only `axis = N` is supported.")
    | _ =>
        Macro.throwUnsupported
  pure axisIdx

partial def expandArgReduce (env : Env) (ctx : String) (op : TSyntax `term)
    (e : TSyntax `tritonExpr)
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM EOut := do
  let e' ← expandExpr env e
  ensureDType .real e'.dtype ctx
  let dims := match e'.shape with | .dims ds => ds
  let axisIdx ← parseAxisOnlyKwargs ctx dims kwargs
  let outDims ← eraseNth dims axisIdx
  let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
  pure ⟨← `($op (⟨$axisLit, by simp⟩) $e'.term), .nat, .dims outDims, none⟩

partial def expandSort (env : Env)
    (e : TSyntax `tritonExpr)
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM EOut := do
  let e' ← expandExpr env e
  ensureDType .real e'.dtype "tl.sort"
  let dims := match e'.shape with | .dims ds => ds
  let axisIdx ← parseAxisOnlyKwargs "tl.sort" dims kwargs
  let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
  pure ⟨← `(Op.sort (⟨$axisLit, by simp⟩) $e'.term), .real, e'.shape, none⟩

/-- Lower a `tl.sum(...)` / `tl.max(...)` expression with optional reduction
kwargs (`axis = K`, `keep_dims = true|false`) into the corresponding
`Op.reduceSum / .reduceMax` AST nodes.

Triton spec for omitted `axis` is `axis = None` → **reduce over all
dimensions**, not "reduce the last axis". We honor that: omitted-axis on a
rank-`N` tile lowers to `N` nested `axis = 0, keep_dims = keepDims` calls,
collapsing the tile to a scalar (`keep_dims = false`) or to all-`1`s
(`keep_dims = true`). For rank-1 tiles this degenerates to a single call,
so existing 1D `tl.sum(x)` / `tl.max(x)` kernels are unaffected. With an
explicit `axis = K` we lower to a single call against that axis.

`keep_dims = false` erases the reduced dim; `keep_dims = true` collapses
it to `1`. -/
partial def expandReduce (env : Env) (ctx : String) (op : TSyntax `term)
    (e : TSyntax `tritonExpr)
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM EOut := do
  let e' ← expandExpr env e
  ensureDType .real e'.dtype ctx
  let dims := match e'.shape with | .dims ds => ds
  if dims.isEmpty then
    Macro.throwError (ctx ++ ": reduction expects a tile, got scalar")
  let mut seenAxis : Bool := Bool.false
  let mut seenKeepDims : Bool := Bool.false
  let mut keepDims : Bool := Bool.false
  let mut axis? : Option Nat := none
  for kw in kwargs do
    match kw with
    | `(tritonReduceKwarg| axis = $n:num) =>
        if seenAxis then
          Macro.throwError (ctx ++ ": duplicate `axis=` kwarg")
        seenAxis := Bool.true
        if n.getNat ≥ dims.length then
          Macro.throwError
            (ctx ++ ": axis `" ++ toString n.getNat ++ "` out of bounds for rank "
             ++ toString dims.length)
        axis? := some n.getNat
    | `(tritonReduceKwarg| keep_dims = false) =>
        if seenKeepDims then
          Macro.throwError (ctx ++ ": duplicate `keep_dims=` kwarg")
        seenKeepDims := Bool.true
    | `(tritonReduceKwarg| keep_dims = true) =>
        if seenKeepDims then
          Macro.throwError (ctx ++ ": duplicate `keep_dims=` kwarg")
        seenKeepDims := Bool.true
        keepDims := Bool.true
    | `(tritonReduceKwarg| $name:ident = $_) =>
        let nm := name.getId.toString
        if nm == "axis" || nm == "keep_dims" then
          Macro.throwError
            (ctx ++ ": `" ++ nm ++ "=` value is not a recognized literal")
        else
          Macro.throwError
            (ctx ++ ": unknown kwarg `" ++ nm ++
             "`. Only `axis = N` and `keep_dims = true|false` are supported.")
    | _ => Macro.throwUnsupported
  let kdLit : TSyntax `term ← if keepDims then `(Bool.true) else `(Bool.false)
  match axis? with
  | some axisIdx =>
      -- Single-axis reduction with explicit `axis = K`.
      let outDims ←
        if keepDims then setNthOne dims axisIdx else eraseNth dims axisIdx
      let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
      pure ⟨← `($op (⟨$axisLit, by simp⟩) $kdLit $e'.term),
            .real, SInfo.dims outDims, none⟩
  | none =>
      -- `axis = None` (Triton default): reduce over all dimensions.
      -- Two regimes, because `keep_dims` changes how the rank evolves:
      --   * `keep_dims = false`: each call drops the front axis; emit
      --     `dims.length` nested calls with `axis = 0`.
      --   * `keep_dims = true`: rank is preserved; emit calls with
      --     `axis = 0, 1, …, dims.length - 1` so every axis is reduced.
      let mut term := e'.term
      for j in [:dims.length] do
        let axisIdx := if keepDims then j else 0
        let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
        term ← `($op (⟨$axisLit, by simp⟩) $kdLit $term)
      let outDims : List (TSyntax `term) ←
        if keepDims then
          let oneLit : TSyntax `term ← `((1 : Nat))
          pure (List.replicate dims.length oneLit)
        else
          pure []
      pure ⟨term, .real, SInfo.dims outDims, none⟩

/-- Lower a `tl.dot(a, b)` to `Op.dot a b`. Both operands must be rank-2
real tiles whose inner dim agrees syntactically (same dim term). The
result shape is `[outerDim a, innerDim b]`. -/
partial def expandDot (env : Env)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureDType .real a'.dtype "tl.dot"
  ensureDType .real b'.dtype "tl.dot"
  -- Both operands must be rank ≥ 2 (the trailing two dims are the matmul);
  -- any leading dims form a shared `batch` prefix that must agree
  -- syntactically. The inner dim `K` is the last of `a` / first-of-trailing
  -- of `b` and must match.
  let aDims := match a'.shape with | .dims ds => ds
  let bDims := match b'.shape with | .dims ds => ds
  if aDims.length < 2 then
    Macro.throwError "tl.dot: LHS must have rank ≥ 2"
  if bDims.length < 2 then
    Macro.throwError "tl.dot: RHS must have rank ≥ 2"
  if aDims.length != bDims.length then
    Macro.throwError
      ("tl.dot: rank mismatch — LHS rank " ++ toString aDims.length ++
       ", RHS rank " ++ toString bDims.length)
  let aBatch := aDims.dropLast.dropLast
  let bBatch := bDims.dropLast.dropLast
  unless termListEq aBatch bBatch do
    Macro.throwError "tl.dot: batch prefix mismatch"
  -- aDims = batch ++ [aM, aK], bDims = batch ++ [bK, bN]
  let aM := aDims[aDims.length - 2]!
  let aK := aDims[aDims.length - 1]!
  let bK := bDims[bDims.length - 2]!
  let bN := bDims[bDims.length - 1]!
  unless termKey aK == termKey bK do
    Macro.throwError
      ("tl.dot: inner dim mismatch — LHS innermost `" ++ termKey aK ++
       "` ≠ RHS outer-trailing `" ++ termKey bK ++ "`")
  -- Pin the implicit `batch` / `M` / `K` / `N` arguments via named
  -- positions; the unification `batch ++ [M, K] = ...` is not
  -- invertible automatically through `List.append`.
  let rec batchTerm : List (TSyntax `term) → MacroM (TSyntax `term)
    | [] => `(([] : TileShape))
    | d :: rest => do
        let tail ← batchTerm rest
        `($d :: $tail)
  let batchT ← batchTerm aBatch
  pure ⟨← `(Op.dot (batch := $batchT) (M := $aM) (K := $aK) (N := $bN)
              $a'.term $b'.term),
        .real, .dims (aBatch ++ [aM, bN]), none⟩

/-- Lower `tl.full([dims*], value)` to `Op.full`. The DSL value is
restricted to a real / nat scalar (its dtype propagates to the resulting
tile). The shape list may be any rank including empty (which degenerates
to the scalar value itself, matching `Op.full shape (Op.const v) = v`
for `shape = []`). -/
partial def expandFull (env : Env)
    (dims : Array (TSyntax `tritonExpr)) (v : TSyntax `tritonExpr) :
    MacroM EOut := do
  let v' ← expandExpr env v
  -- Value must be a scalar; tile-shaped values aren't broadcast here.
  match v'.shape with
  | .dims [] => pure ()
  | _ => Macro.throwError "tl.full: value must be a scalar (rank-0)"
  -- Each dim is a tritonExpr; its surface form may be `$(t)` (a Lean
  -- `Nat` antiquote) or a numeral. We extract a plain `term` per dim
  -- without going through `expandExpr` (which would promote the dim to
  -- an `Op`). The user is responsible for keeping dims consistent with
  -- the surrounding code; the macro just stitches them into the
  -- `TileShape` literal that types the result.
  let mut dimTerms : Array (TSyntax `term) := #[]
  -- Wrap every dim in `(_ : Nat)` so its `termKey` matches what
  -- `tl.arange(0, end)` emits (which also wraps); without this,
  -- `tl.full([\$(M)], …)` would not broadcast against `[M]`-shaped
  -- tiles produced by arange.
  for d in dims do
    match d with
    | `(tritonExpr| $($t:term)) =>
        dimTerms := dimTerms.push (← `(($t : Nat)))
    | `(tritonExpr| $n:num) =>
        dimTerms := dimTerms.push (← `(($n : Nat)))
    | _ =>
        Macro.throwError "tl.full: each dim must be a numeric literal or `$(t)` antiquote"
  let rec shapeTerm : List (TSyntax `term) → MacroM (TSyntax `term)
    | [] => `(([] : TileShape))
    | d :: rest => do
        let tail ← shapeTerm rest
        `($d :: $tail)
  let shape ← shapeTerm dimTerms.toList
  pure ⟨← `(Op.full $shape $v'.term), v'.dtype, .dims dimTerms.toList, none⟩

/-- Lower `tl.trans(e)` to `Op.transpose`. Rank-≥ 2 required; the
trailing two dims are swapped, any leading dims pass through as the
`batch` prefix. The input shape's `SInfo.dims` already lists
outermost-first, so picking off the last two and rebuilding
`batch ++ [N, M]` is straightforward. -/
partial def expandTranspose (env : Env)
    (e : TSyntax `tritonExpr) : MacroM EOut := do
  let e' ← expandExpr env e
  let dims := match e'.shape with | .dims ds => ds
  if dims.length < 2 then
    Macro.throwError
      ("tl.trans: rank-≥ 2 input required, got rank " ++ toString dims.length ++
       ". (Triton `.T` / `tl.trans` swaps the trailing two axes; on a rank-1 tile use `tl.expand_dims` first.)")
  let batch := dims.dropLast.dropLast
  let M := dims[dims.length - 2]!
  let N := dims[dims.length - 1]!
  -- Build a `TileShape` term for `batch` so the elaborator pins the
  -- implicit `batch` argument (the unification `batch ++ [M, N] = ...`
  -- is not invertible automatically through `List.append`).
  let rec batchTerm : List (TSyntax `term) → MacroM (TSyntax `term)
    | [] => `(([] : TileShape))
    | d :: rest => do
        let tail ← batchTerm rest
        `($d :: $tail)
  let batchT ← batchTerm batch
  pure ⟨← `(Op.transpose (batch := $batchT) (M := $M) (N := $N) $e'.term),
        e'.dtype, .dims (batch ++ [N, M]), none⟩

partial def shapeTermOfDims (dims : List (TSyntax `term)) : MacroM (TSyntax `term) := do
  (SInfo.dims dims).term

partial def termNatLit? (t : TSyntax `term) : Option Nat :=
  match t with
  | `($n:num) => some n.getNat
  | `(($e:term)) => termNatLit? e
  | `(($e:term : $_:term)) => termNatLit? e
  | _ => none

partial def literalProduct? (dims : List (TSyntax `term)) : Option Nat :=
  dims.foldl
    (fun acc d => do
      let p ← acc
      let n ← termNatLit? d
      some (p * n))
    (some 1)

partial def coordProjTerm (idx : TSyntax `term) (pos : Nat) :
    MacroM (TSyntax `term) := do
  let mut cursor := idx
  for _ in [:pos] do
    cursor ← `(($cursor).2)
  `(($cursor).1)

partial def indexTupleTerm : List (TSyntax `term) → MacroM (TSyntax `term)
  | [] => `(PUnit.unit)
  | c :: rest => do
      let tail ← indexTupleTerm rest
      `(($c, $tail))

partial def validatePermutation (rank : Nat) (axes : List Nat) (ctx : String) :
    MacroM Unit := do
  unless axes.length == rank do
    Macro.throwError
      (ctx ++ ": permutation length " ++ toString axes.length ++
       " does not match rank " ++ toString rank)
  for ax in axes do
    if ax ≥ rank then
      Macro.throwError
        (ctx ++ ": axis `" ++ toString ax ++ "` out of bounds for rank " ++
         toString rank)
  for i in [:rank] do
    unless axes.count i == 1 do
      Macro.throwError
        (ctx ++ ": axes must be a permutation of 0.." ++ toString (rank - 1))

partial def expandPermute (env : Env)
    (e : TSyntax `tritonExpr) (axesStx : List (TSyntax `num)) :
    MacroM EOut := do
  let e' ← expandExpr env e
  let dims := match e'.shape with | .dims ds => ds
  let axes := axesStx.map (fun n => n.getNat)
  validatePermutation dims.length axes "tl.permute"
  let outDims := axes.map (fun ax => dims[ax]!)
  let outShape ← shapeTermOfDims outDims
  let idx : TSyntax `term ← `(idx)
  let mut inputCoords : List (TSyntax `term) := []
  for inputAxis in [:dims.length] do
    let outPos := axes.idxOf inputAxis
    if outPos == axes.length then
      Macro.throwError "internal error: validated permutation lost an axis"
    inputCoords := inputCoords ++ [← coordProjTerm idx outPos]
  let mapBody ← indexTupleTerm inputCoords
  pure ⟨← `(Op.remap $outShape (fun idx => $mapBody) $e'.term),
        e'.dtype, .dims outDims, none⟩

partial def expandFlip (env : Env)
    (e : TSyntax `tritonExpr) (axisIdx : Nat) : MacroM EOut := do
  let e' ← expandExpr env e
  let dims := match e'.shape with | .dims ds => ds
  if dims.isEmpty then
    Macro.throwError "tl.flip: rank-≥ 1 input required"
  if axisIdx ≥ dims.length then
    Macro.throwError
      ("tl.flip: axis `" ++ toString axisIdx ++ "` out of bounds for rank " ++
       toString dims.length)
  let shape ← e'.shape.term
  let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
  pure ⟨← `(Op.remap $shape
              (fun idx => TileShape.flipIndex $shape (⟨$axisLit, by simp⟩) idx)
              $e'.term),
        e'.dtype, e'.shape, none⟩

partial def parseFlipKwargs (kwargs : TSyntaxArray `tritonReduceKwarg) :
    MacroM Nat := do
  let mut axis? : Option Nat := none
  for kw in kwargs do
    match kw with
    | `(tritonReduceKwarg| axis = $n:num) =>
        if axis?.isSome then
          Macro.throwError "tl.flip: duplicate axis/dim kwarg"
        axis? := some n.getNat
    | `(tritonReduceKwarg| $name:ident = $val:tritonExpr) =>
        let nm := name.getId.toString
        unless nm == "dim" do
          Macro.throwError
            ("tl.flip: unsupported kwarg `" ++ nm ++ "`. Only `dim = N` / `axis = N` is supported.")
        if axis?.isSome then
          Macro.throwError "tl.flip: duplicate axis/dim kwarg"
        match val with
        | `(tritonExpr| $n:num) => axis? := some n.getNat
        | _ => Macro.throwError "tl.flip: `dim=` must be a numeric literal"
    | `(tritonReduceKwarg| keep_dims = false) =>
        Macro.throwError "tl.flip: `keep_dims` is not supported"
    | `(tritonReduceKwarg| keep_dims = true) =>
        Macro.throwError "tl.flip: `keep_dims` is not supported"
    | _ => Macro.throwUnsupported
  match axis? with
  | some ax => pure ax
  | none => Macro.throwError "tl.flip: explicit `dim = N` / `axis = N` is required"

partial def expandReshapeLike (env : Env) (ctx : String)
    (e : TSyntax `tritonExpr) (dims : Array (TSyntax `tritonExpr)) :
    MacroM EOut := do
  let e' ← expandExpr env e
  let (outShape, outDims) ← natListTerm ctx dims
  let inDims := match e'.shape with | .dims ds => ds
  match literalProduct? inDims, literalProduct? outDims with
  | some inProduct, some outProduct =>
      unless inProduct == outProduct do
        Macro.throwError
          (ctx ++ ": literal element-count mismatch (" ++ toString inProduct ++
           " input elements vs " ++ toString outProduct ++ " output elements)")
  | _, _ => pure ()
  pure ⟨← `(Op.reshape $outShape $e'.term), e'.dtype, .dims outDims, none⟩

partial def expandRavel (env : Env)
    (e : TSyntax `tritonExpr) : MacroM EOut := do
  let e' ← expandExpr env e
  let dims := match e'.shape with | .dims ds => ds
  let product ←
    match dims with
    | [] => `((1 : Nat))
    | d :: rest => do
        let mut acc := d
        for next in rest do
          acc ← `($acc * $next)
        pure acc
  let outShape ← shapeTermOfDims [product]
  pure ⟨← `(Op.reshape $outShape $e'.term), e'.dtype, .dims [product], none⟩

partial def expandJoin (env : Env)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  unless a'.dtype == b'.dtype do
    Macro.throwError "tl.join: dtype mismatch"
  ensureShape a'.shape b'.shape "tl.join"
  let dims := match a'.shape with | .dims ds => ds
  let two : TSyntax `term ← `((2 : Nat))
  pure ⟨← `(Op.join $a'.term $b'.term), a'.dtype, .dims (dims ++ [two]), none⟩

partial def expandSplit (env : Env)
    (e : TSyntax `tritonExpr) (side : Nat) : MacroM EOut := do
  let e' ← expandExpr env e
  unless side == 0 || side == 1 do
    Macro.throwError "tl.split: side must be literal 0 or 1"
  let dims := match e'.shape with | .dims ds => ds
  if dims.isEmpty then
    Macro.throwError "tl.split: rank-≥ 1 input required"
  let lastDim := dims[dims.length - 1]!
  unless termNatLit? lastDim == some 2 do
    Macro.throwError "tl.split: final dimension must be the literal 2"
  let outDims := dims.dropLast
  let outShape ← shapeTermOfDims outDims
  let sideLit : TSyntax `num := ⟨Syntax.mkNumLit (toString side)⟩
  pure ⟨← `(Op.split (shape := $outShape) (⟨$sideLit, by decide⟩) $e'.term),
        e'.dtype, .dims outDims, none⟩

/-- Lower `tl.expand_dims(e, axis=N)` / `tl.expand_dims(e, N)` to
`Op.expandDim`. The axis must be a literal in `[0, rank]`; dimensions are
typed by inserting a unit axis into the macro-tracked shape. -/
partial def expandExpandDims (env : Env)
    (e : TSyntax `tritonExpr) (axisIdx : Nat) : MacroM EOut := do
  let e' ← expandExpr env e
  let dims := match e'.shape with | .dims ds => ds
  if axisIdx > dims.length then
    Macro.throwError
      ("tl.expand_dims: axis `" ++ toString axisIdx ++
       "` out of bounds for rank " ++ toString dims.length)
  let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
  let outDims : List (TSyntax `term) :=
    dims.take axisIdx ++ [← `((1 : Nat))] ++ dims.drop axisIdx
  pure ⟨← `(Op.expandDim (⟨$axisLit, by simp⟩) $e'.term),
        e'.dtype, .dims outDims, none⟩

/-- Lower `e[:, None]` / `e[None, :]` to `Op.expandDim` with the appropriate
axis. This postfix surface intentionally remains rank-1 only; use
`tl.expand_dims(e, axis=N)` for general-rank insertion. -/
partial def expandSlicerNone (env : Env)
    (e : TSyntax `tritonExpr) (axisIdx : Nat) : MacroM EOut := do
  let e' ← expandExpr env e
  let dims := match e'.shape with | .dims ds => ds
  if dims.length != 1 then
    Macro.throwError
      ("[:, None] / [None, :]: rank-1 input required, got rank " ++
       toString dims.length ++
       ". Use `tl.expand_dims(e, axis=N)` for general-rank insertion.")
  expandExpandDims env e (axisIdx := axisIdx)

end

mutual

partial def expandStmt (env : Env) (stx : TSyntax `tritonStmt) :
    MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
  match stx with
  | `(tritonStmt| $i:ident := $e:tritonExpr) => do
      let nameLit ← identAsStr i
      let e' ← expandExpr env e
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        (i.getId.toString, e'.dtype, e'.shape) :: env,
        e'.computeTerm.isSome)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      let mut maskTerm : Option (TSyntax `term × SInfo) := none
      let mut dtype? : Option DInfo := none
      let mut boundaryCheck? : Option (TSyntax `term) := none
      for kw in kwargs do
        match kw with
        | `(tritonMemKwarg| boundary_check=$axes:term) =>
            if boundaryCheck?.isSome then
              Macro.throwError "tl.store: duplicate `boundary_check=` kwarg"
            boundaryCheck? := some axes
        | `(tritonMemKwarg| padding_option="zero") =>
            Macro.throwError "tl.store: `padding_option` is only valid on tl.load"
        | `(tritonMemKwarg| $name:ident = $kval:tritonExpr) =>
            let kval' ← expandExpr env kval
            match name.getId.toString with
            | "mask"  =>
                ensureDType .bool kval'.dtype "tl.store mask"
                maskTerm := some (kval'.term, kval'.shape)
            | unknown =>
                let msg : String :=
                  "tl.store: unknown kwarg `" ++ unknown ++
                  "`. Only `mask`, `dtype`, and `boundary_check` are recognized (Triton's tl.store has no `other`; see issue #16)."
                Macro.throwError msg
        | `(tritonMemKwarg| $name:ident = $dt:tritonDType) =>
            unless name.getId.toString == "dtype" do
              Macro.throwError
                ("tl.store: unknown kwarg `" ++ name.getId.toString ++
                 "`. Only `mask`, `dtype`, and `boundary_check` are recognized (Triton's tl.store has no `other`; see issue #16).")
            if dtype?.isSome then
              Macro.throwError "tl.store: duplicate `dtype=` kwarg"
            dtype? := some (← expandDType dt)
        | _ => Macro.throwUnsupported
      let v' ← expandExpr env v
      let valueExpr ←
        match v'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $v'.term)
      let storeDType := dtype?.getD v'.dtype
      unless storeDType == v'.dtype do
        Macro.throwError "tl.store: `dtype=` must match the value dtype"
      if let some boundaryCheck := boundaryCheck? then
        if maskTerm.isSome then
          Macro.throwError "tl.store: block-pointer `boundary_check` cannot be combined with `mask`"
        let p' ← expandExpr env p
        ensureDType .blockPtr p'.dtype "tl.store block pointer"
        let vTerm ← coerceShape v'.term v'.shape p'.shape "tl.store value"
        let sh ← p'.shape.term
        let dt ← storeDType.term
        let valueExpr' ←
          match v'.computeTerm with
          | some _ => pure valueExpr
          | none => `(ComputeExpr.alg $vTerm)
        return (← `(Stmt.store $dt $sh (MemAccess.blockPtr $p'.term $boundaryCheck) $vTerm MaskOpt.none),
          ← `(ComputeStmt.store $dt $sh (MemAccess.blockPtr $p'.term $boundaryCheck) $valueExpr' MaskOpt.none),
          env, v'.computeTerm.isSome)
      match maskTerm with
      | none =>
          match ← expandStaticPtrExpr env p with
          | some sp =>
              let vTerm ← coerceShape v'.term v'.shape sp.shape "tl.store value"
              let r := sp.region
              let off := sp.offset
              let sh ← sp.shape.term
              let dt ← storeDType.term
              let valueExpr' ←
                match v'.computeTerm with
                | some _ => pure valueExpr
                | none => `(ComputeExpr.alg $vTerm)
              pure (← `(Stmt.store $dt $sh (MemAccess.region $r $off) $vTerm MaskOpt.none),
                ← `(ComputeStmt.store $dt $sh (MemAccess.region $r $off) $valueExpr' MaskOpt.none),
                env, v'.computeTerm.isSome)
          | none =>
              let p' ← expandExpr env p
              ensureDType .ptr p'.dtype "tl.store pointer"
              let vTerm ← coerceShape v'.term v'.shape p'.shape "tl.store value"
              let sh ← p'.shape.term
              let dt ← storeDType.term
              let valueExpr' ←
                match v'.computeTerm with
                | some _ => pure valueExpr
                | none => `(ComputeExpr.alg $vTerm)
              pure (← `(Stmt.store $dt $sh (MemAccess.ptr $p'.term) $vTerm MaskOpt.none),
                ← `(ComputeStmt.store $dt $sh (MemAccess.ptr $p'.term) $valueExpr' MaskOpt.none),
                env, v'.computeTerm.isSome)
      | some (m, mShape) =>
          match ← expandStaticPtrExpr env p with
          | some sp =>
              let vTerm ← coerceShape v'.term v'.shape sp.shape "tl.store value"
              let m' ← coerceShape m mShape sp.shape "tl.store mask"
              let r := sp.region
              let off := sp.offset
              let sh ← sp.shape.term
              let dt ← storeDType.term
              let valueExpr' ←
                match v'.computeTerm with
                | some _ => pure valueExpr
                | none => `(ComputeExpr.alg $vTerm)
              pure (← `(Stmt.store $dt $sh (MemAccess.region $r $off) $vTerm (MaskOpt.mask $m')),
                ← `(ComputeStmt.store $dt $sh (MemAccess.region $r $off) $valueExpr' (MaskOpt.mask $m')),
                env, v'.computeTerm.isSome)
          | none =>
              let p' ← expandExpr env p
              ensureDType .ptr p'.dtype "tl.store pointer"
              let vTerm ← coerceShape v'.term v'.shape p'.shape "tl.store value"
              let m' ← coerceShape m mShape p'.shape "tl.store mask"
              let sh ← p'.shape.term
              let dt ← storeDType.term
              let valueExpr' ←
                match v'.computeTerm with
                | some _ => pure valueExpr
                | none => `(ComputeExpr.alg $vTerm)
                pure (← `(Stmt.store $dt $sh (MemAccess.ptr $p'.term) $vTerm (MaskOpt.mask $m')),
                  ← `(ComputeStmt.store $dt $sh (MemAccess.ptr $p'.term) $valueExpr' (MaskOpt.mask $m')),
                  env, v'.computeTerm.isSome)
  | `(tritonStmt| tl.atomic_add($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      let mut maskTerm : Option (TSyntax `term × SInfo) := none
      let mut dtype? : Option DInfo := none
      for kw in kwargs do
        match kw with
        | `(tritonMemKwarg| boundary_check=$_:term) =>
            Macro.throwError "tl.atomic_add: `boundary_check=` is not supported; block-pointer atomics are deferred"
        | `(tritonMemKwarg| padding_option="zero") =>
            Macro.throwError "tl.atomic_add: `padding_option` is only valid on tl.load"
        | `(tritonMemKwarg| $name:ident = $kval:tritonExpr) =>
            let kval' ← expandExpr env kval
            match name.getId.toString with
            | "mask" =>
                ensureDType .bool kval'.dtype "tl.atomic_add mask"
                maskTerm := some (kval'.term, kval'.shape)
            | unknown =>
                Macro.throwError
                  ("tl.atomic_add: unknown kwarg `" ++ unknown ++
                   "`. Only `mask` and `dtype` are recognized.")
        | `(tritonMemKwarg| $name:ident = $dt:tritonDType) =>
            unless name.getId.toString == "dtype" do
              Macro.throwError
                ("tl.atomic_add: unknown kwarg `" ++ name.getId.toString ++
                 "`. Only `mask` and `dtype` are recognized.")
            if dtype?.isSome then
              Macro.throwError "tl.atomic_add: duplicate `dtype=` kwarg"
            dtype? := some (← expandDType dt)
        | _ => Macro.throwUnsupported
      let v' ← expandExpr env v
      let valueExpr ←
        match v'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $v'.term)
      let atomicDType := dtype?.getD v'.dtype
      unless atomicDType == v'.dtype do
        Macro.throwError "tl.atomic_add: `dtype=` must match the value dtype"
      let hnum ← atomicDType.numericProof
      match maskTerm with
      | none =>
          match ← expandStaticPtrExpr env p with
          | some sp =>
              let vTerm ← coerceShape v'.term v'.shape sp.shape "tl.atomic_add value"
              let sh ← sp.shape.term
              let valueExpr' ←
                match v'.computeTerm with
                | some _ => pure valueExpr
                | none => `(ComputeExpr.alg $vTerm)
              pure (← `(Stmt.atomicAdd $hnum $sh (MemAccess.region $sp.region $sp.offset) $vTerm MaskOpt.none),
                ← `(ComputeStmt.atomicAdd $hnum $sh (MemAccess.region $sp.region $sp.offset) $valueExpr' MaskOpt.none),
                env, v'.computeTerm.isSome)
          | none =>
              let p' ← expandExpr env p
              ensureDType .ptr p'.dtype "tl.atomic_add pointer"
              let vTerm ← coerceShape v'.term v'.shape p'.shape "tl.atomic_add value"
              let sh ← p'.shape.term
              let valueExpr' ←
                match v'.computeTerm with
                | some _ => pure valueExpr
                | none => `(ComputeExpr.alg $vTerm)
              pure (← `(Stmt.atomicAdd $hnum $sh (MemAccess.ptr $p'.term) $vTerm MaskOpt.none),
                ← `(ComputeStmt.atomicAdd $hnum $sh (MemAccess.ptr $p'.term) $valueExpr' MaskOpt.none),
                env, v'.computeTerm.isSome)
      | some (m, mShape) =>
          match ← expandStaticPtrExpr env p with
          | some sp =>
              let vTerm ← coerceShape v'.term v'.shape sp.shape "tl.atomic_add value"
              let m' ← coerceShape m mShape sp.shape "tl.atomic_add mask"
              let sh ← sp.shape.term
              let valueExpr' ←
                match v'.computeTerm with
                | some _ => pure valueExpr
                | none => `(ComputeExpr.alg $vTerm)
              pure (← `(Stmt.atomicAdd $hnum $sh (MemAccess.region $sp.region $sp.offset) $vTerm (MaskOpt.mask $m')),
                ← `(ComputeStmt.atomicAdd $hnum $sh (MemAccess.region $sp.region $sp.offset) $valueExpr' (MaskOpt.mask $m')),
                env, v'.computeTerm.isSome)
          | none =>
              let p' ← expandExpr env p
              ensureDType .ptr p'.dtype "tl.atomic_add pointer"
              let vTerm ← coerceShape v'.term v'.shape p'.shape "tl.atomic_add value"
              let m' ← coerceShape m mShape p'.shape "tl.atomic_add mask"
              let sh ← p'.shape.term
              let valueExpr' ←
                match v'.computeTerm with
                | some _ => pure valueExpr
                | none => `(ComputeExpr.alg $vTerm)
              pure (← `(Stmt.atomicAdd $hnum $sh (MemAccess.ptr $p'.term) $vTerm (MaskOpt.mask $m')),
                ← `(ComputeStmt.atomicAdd $hnum $sh (MemAccess.ptr $p'.term) $valueExpr' (MaskOpt.mask $m')),
                env, v'.computeTerm.isSome)
  | `(tritonStmt| tl.async_copy($dst:tritonExpr, $src:tritonExpr $[, $_kwargs:tritonMemKwarg]*)) => do
      discard <| expandExpr env dst
      discard <| expandExpr env src
      pure (← `(Stmt.ifThen (Op.constBool Bool.false) []),
        ← `(ComputeStmt.asyncMarker "tl.async_copy"), env, Bool.true)
    | `(tritonStmt| tl.for $i:ident in $($n:term) { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv stmts.toList
      pure (← `(Stmt.forLoop $nameLit $n [$algBody,*]),
        ← `(ComputeStmt.forLoop $nameLit $n [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| tl.for $i:ident in $n:num { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv stmts.toList
      pure (← `(Stmt.forLoop $nameLit $n [$algBody,*]),
        ← `(ComputeStmt.forLoop $nameLit $n [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| tl.static_range $i:ident in $($n:term) { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv stmts.toList
      pure (← `(Stmt.forLoop $nameLit $n [$algBody,*]),
        ← `(ComputeStmt.forLoop $nameLit $n [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| tl.static_range $i:ident in $n:num { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv stmts.toList
      pure (← `(Stmt.forLoop $nameLit $n [$algBody,*]),
        ← `(ComputeStmt.forLoop $nameLit $n [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| tl.if $cond:tritonExpr { $stmts:tritonStmt* }) => do
      let cond' ← expandExpr env cond
      ensureDType .bool cond'.dtype "tl.if condition"
      ensureShape SInfo.scalar cond'.shape "tl.if condition"
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts env stmts.toList
      pure (← `(Stmt.ifThen $cond'.term [$algBody,*]),
        ← `(ComputeStmt.ifThen $cond'.term [$computeBody,*]), env, bodyHasCompute)
  | _ => Macro.throwUnsupported

partial def expandStmts (env : Env) (stmts : List (TSyntax `tritonStmt)) :
    MacroM (Array (TSyntax `term) × Array (TSyntax `term) × Env × Bool) := do
  let mut algOut : Array (TSyntax `term) := #[]
  let mut computeOut : Array (TSyntax `term) := #[]
  let mut env' := env
  let mut hasCompute := Bool.false
  for st in stmts do
    let (algTerm, computeTerm, nextEnv, stmtHasCompute) ← expandStmt env' st
    algOut := algOut.push algTerm
    computeOut := computeOut.push computeTerm
    env' := nextEnv
    hasCompute := hasCompute || stmtHasCompute
  pure (algOut, computeOut, env', hasCompute)

end

/-! ## Block macro -/

macro_rules
  | `(triton { $stmts:tritonStmt* }) => do
      let (algStmtTerms, computeStmtTerms, _, hasCompute) ← expandStmts [] stmts.toList
      -- Auto-scan body: collect every region appearing in `tl.load(...)` (inputs)
      -- and `tl.store(...)` (outputs). Order = body occurrence; no macro-time
      -- dedup (a mix of literals and Lean terms can't be statically deduped, and
      -- `Kernel.inputs/outputs` is metadata-only, so duplicates are harmless).
      let (allIns, allOuts) := stmts.foldl
        (fun (acc : List (TSyntax `term) × List (TSyntax `term)) s =>
          let (i, o) := Metadata.stmtRegions s
          (acc.1 ++ i, acc.2 ++ o))
        ([], [])
      let insArr  : Array (TSyntax `term) := allIns.toArray
      let outsArr : Array (TSyntax `term) := allOuts.toArray
      if hasCompute then
        `(ComputeKernel.mk [$insArr,*] [$outsArr,*] [$computeStmtTerms,*])
      else
        `(ComputeKernel.fromAlg (Kernel.mk [$insArr,*] [$outsArr,*] [$algStmtTerms,*]))

end VeriTile.Triton.DSL
