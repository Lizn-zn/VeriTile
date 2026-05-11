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
  * `$(<lean-term>)` is context-sensitive: address/shape/index contexts lower it
    to `.nat`, while data/scalar contexts lower it to `.real`. Ambiguous uses
    should be rejected instead of guessed.
  * Numeric literals become `Op.const`.
  * Statements are separated by newlines (no explicit terminator).

Currently supported expressions: `tl.program_id(_)`, `tl.arange(_)` /
`tl.arange(start, end)`, `tl.exp(_)`, `tl.exp2(_)`, `tl.log(_)`, `tl.log2(_)`,
`tl.sigmoid(_)`, `tl.sqrt(_)`, `tl.math.rsqrt(_)`/`tl.rsqrt(_)`, `tl.tanh(_)`, `tl.sin(_)`, `tl.cos(_)`,
`tl.tan(_)`, `tl.atan(_)`, `tl.cosh(_)`, `tl.sinh(_)`, `tl.erf(_)`,
`tl.extra.cuda.libdevice.erf(_)`, `tl.max(_)`, `tl.sum(_)`, `tl.load(ptrExpr)`,
binary `+ - * /`, parens, identifiers, numerals, antiquotation (`$(t)` for
Lean values in typed contexts / `RegionName` in pointer context).
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
import VeriTile.Triton.DSL.Inference
import VeriTile.Triton.DSL.Expansion.Common
import VeriTile.Triton.DSL.Expansion.Memory
import VeriTile.Triton.DSL.Expansion.Compute
import VeriTile.Triton.DSL.Expansion.Control
import VeriTile.Triton.DSL.Syntax
import VeriTile.Triton.DSL.Typing

open Lean

namespace VeriTile.Triton.DSL

/-! ## Expansion -/

def extractDTypeKwarg (kwargs : TSyntaxArray `tritonMemKwarg) :
    MacroM (Option DInfo) := do
  for kw in kwargs do
    match kw with
    | `(tritonMemKwarg| $name:ident = $dt:tritonDType) =>
        if name.getId.toString == "dtype" then
          return some (← expandDType dt)
    | _ => pure ()
  return none

def parseMaxReturnIndicesKwargs (dims : List (TSyntax `term))
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM Nat := do
  let mut seenAxis : Bool := Bool.false
  let mut seenReturn : Bool := Bool.false
  let mut axisIdx : Nat := 0
  for kw in kwargs do
    match kw with
    | `(tritonReduceKwarg| $n:num) =>
        if seenAxis then
          Macro.throwError "tl.max return_indices: duplicate axis"
        seenAxis := Bool.true
        if n.getNat ≥ dims.length then
          Macro.throwError "tl.max return_indices: axis out of bounds"
        axisIdx := n.getNat
    | `(tritonReduceKwarg| axis = $n:num) =>
        if seenAxis then
          Macro.throwError "tl.max return_indices: duplicate axis"
        seenAxis := Bool.true
        if n.getNat ≥ dims.length then
          Macro.throwError "tl.max return_indices: axis out of bounds"
        axisIdx := n.getNat
    | `(tritonReduceKwarg| return_indices=True) =>
        seenReturn := Bool.true
    | `(tritonReduceKwarg| return_indices=False) =>
        Macro.throwError "tl.max tuple binding requires `return_indices=True`"
    | `(tritonReduceKwarg| keep_dims = false) =>
        Macro.throwError "tl.max return_indices: `keep_dims` is not supported"
    | `(tritonReduceKwarg| keep_dims = true) =>
        Macro.throwError "tl.max return_indices: `keep_dims` is not supported"
    | `(tritonReduceKwarg| $name:ident = $_) =>
        Macro.throwError
          ("tl.max return_indices: unsupported kwarg `" ++ name.getId.toString ++ "`")
    | _ => Macro.throwUnsupported
  unless seenReturn do
    Macro.throwError "tl.max tuple binding requires `return_indices=True`"
  unless seenAxis do
    Macro.throwError "tl.max return_indices requires an explicit axis"
  pure axisIdx

private def typedRegionAntiquote? (r : TSyntax `term) :
    MacroM (Option (TSyntax `term × DInfo)) := do
  let dtype? ←
    match r with
    | `(($_:term : Region .real)) => pure (some DInfo.real)
    | `(($_:term : Region .fp32)) => pure (some DInfo.fp32)
    | `(($_:term : Region .fp16)) => pure (some DInfo.fp16)
    | `(($_:term : Region .bf16)) => pure (some DInfo.bf16)
    | `(($_:term : Region .int)) => pure (some DInfo.int)
    | `(($_:term : Region .nat)) => pure (some DInfo.nat)
    | `(($_:term : Region .bool)) => pure (some DInfo.bool)
    | `(($_:term : Region TileDType.real)) => pure (some DInfo.real)
    | `(($_:term : Region TileDType.fp32)) => pure (some DInfo.fp32)
    | `(($_:term : Region TileDType.fp16)) => pure (some DInfo.fp16)
    | `(($_:term : Region TileDType.bf16)) => pure (some DInfo.bf16)
    | `(($_:term : Region TileDType.int)) => pure (some DInfo.int)
    | `(($_:term : Region TileDType.nat)) => pure (some DInfo.nat)
    | `(($_:term : Region TileDType.bool)) => pure (some DInfo.bool)
    | _ => pure none
  let some dtype := dtype?
    | pure none
  let regionTerm ← `(Region.name $r)
  pure (some (regionTerm, dtype))

mutual

partial def expandNatExpectedExpr (env : Env) (stx : TSyntax `tritonExpr) :
    MacroM EOut := do
  match stx with
  | `(tritonExpr| $n:num) =>
      pure ⟨← `(Op.constNat $n), .nat, SInfo.scalar, none, none⟩
  | `(tritonExpr| $($t:term)) =>
      pure ⟨← `(Op.constNat $t), .nat, SInfo.scalar, none, none⟩
  | `(tritonExpr| $i:ident) =>
      if env.any (fun entry => entry.1 == i.getId.toString) then
        let name := i.getId.toString
        let (dtype, shape) ← lookupEnv env name
        ensureDType .nat dtype "nat expression"
        let s ← identAsStr i
        let sh ← shape.term
        pure ⟨← `(Op.ref TileDType.nat $sh $s), .nat, shape, none, none⟩
      else
        let t : TSyntax `term := ⟨i.raw⟩
        pure ⟨← `(Op.constNat $t), .nat, SInfo.scalar, none, none⟩
  | `(tritonExpr| ($e:tritonExpr)) =>
      expandNatExpectedExpr env e
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
      let a' ← expandNatExpectedExpr env a
      let b' ← expandNatExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "nat expression"
      pure ⟨← `(Op.add NumericDType.nat $bc $a'.term $b'.term),
        .nat, outShape, none, none⟩
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) => do
      let a' ← expandNatExpectedExpr env a
      let b' ← expandNatExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "nat expression"
      pure ⟨← `(Op.sub NumericDType.nat $bc $a'.term $b'.term),
        .nat, outShape, none, none⟩
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => do
      let a' ← expandNatExpectedExpr env a
      let b' ← expandNatExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "nat expression"
      pure ⟨← `(Op.mul NumericDType.nat $bc $a'.term $b'.term),
        .nat, outShape, none, none⟩
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => do
      let a' ← expandNatExpectedExpr env a
      let b' ← expandNatExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "nat expression"
      pure ⟨← `(Op.floorDiv IntegralDType.nat $bc $a'.term $b'.term),
        .nat, outShape, none, none⟩
  | `(tritonExpr| $a:tritonExpr % $b:tritonExpr) => do
      let a' ← expandNatExpectedExpr env a
      let b' ← expandNatExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "nat expression"
      pure ⟨← `(Op.mod IntegralDType.nat $bc $a'.term $b'.term),
        .nat, outShape, none, none⟩
  | _ => do
      let e' ← expandExpr env stx
      ensureDType .nat e'.dtype "nat expression"
      pure e'

partial def expandStaticPtrExpr (env : Env) (stx : TSyntax `tritonExpr) :
    MacroM (Option StaticPtrOut) := do
  match stx with
  | `(tritonExpr| $($r:term)) =>
      let zero ← `(Op.constNat 0)
      match ← typedRegionAntiquote? r with
      | some (region, dtype) =>
          pure (some ⟨region, some dtype, zero, SInfo.scalar, Bool.true⟩)
      | none =>
          pure (some ⟨r, none, zero, SInfo.scalar, Bool.true⟩)
  | `(tritonExpr| $r:ident) =>
      if env.any (fun entry => entry.1 == r.getId.toString) then
        pure none
      else
        let zero ← `(Op.constNat 0)
        let region : TSyntax `term := ⟨r.raw⟩
        pure (some ⟨region, none, zero, SInfo.scalar, Bool.true⟩)
  | `(tritonExpr| ($e:tritonExpr)) =>
      expandStaticPtrExpr env e
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
      match ← expandStaticPtrExpr env a with
      | some p =>
          let b' ← expandNatExpectedExpr env b
          if p.baseOnly then
            pure (some ⟨p.region, p.regionDType?, b'.term, b'.shape, Bool.false⟩)
          else
            let (bc, outShape) ← broadcastTerm p.shape b'.shape "pointer offset"
            let off := p.offset
            let bTerm := b'.term
            let nextOff ← `(Op.add NumericDType.nat $bc $off $bTerm)
            pure (some ⟨p.region, p.regionDType?, nextOff, outShape, Bool.false⟩)
      | none => pure none
  | _ => pure none

partial def expandWhereFromCond (env : Env) (c' : EOut)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureDType .bool c'.dtype "tl.where condition"
  unless a'.dtype == b'.dtype do
    Macro.throwError "tl.where: branch dtype mismatch"
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
  pure ⟨← `(Op.where $cTerm $aTerm $bTerm), a'.dtype, target, none, none⟩

partial def expandBoolCondition (env : Env) (cond : TSyntax `tritonExpr) :
    MacroM EOut := do
  match cond with
  | `(tritonExpr| $b:ident) =>
      if env.any (fun entry => entry.1 == b.getId.toString) then
        expandExpr env cond
      else
        let boolTerm : TSyntax `term := ⟨b.raw⟩
        pure ⟨← `(Op.constBool $boolTerm), .bool, SInfo.scalar, none, none⟩
  | _ =>
      expandExpr env cond

partial def mergeBranchEnv (base thenEnv elseEnv : Env) : Env :=
  let baseNames := base.map (fun entry => entry.1)
  let thenNew := thenEnv.filter (fun entry => !(baseNames.contains entry.1))
  let elseNew := elseEnv.filter (fun entry => !(baseNames.contains entry.1))
  let mergedNew :=
    thenNew.filter (fun t =>
      let (tn, td, ts, tc) := t
      match elseNew.find? (fun e => e.1 == tn) with
      | some e =>
          let (_, ed, es, ec) := e
          td == ed && ts.eq es && tc == ec
      | none => Bool.false)
  if mergedNew.isEmpty then
    thenNew ++ base
  else
    mergedNew ++ base

partial def natOffsetsWithBaseAdd (env : Env) (ctx : String)
    (offsets strides : Array (TSyntax `tritonExpr)) :
    MacroM (TSyntax `term × Option (TSyntax `term)) := do
  unless offsets.size == strides.size do
    Macro.throwError (ctx ++ ": offsets and strides must have the same rank")
  let mut staticOffsets : Array (TSyntax `term) := #[]
  let mut dynamicAdds : Array (TSyntax `term) := #[]
  for i in [:offsets.size] do
    let off := offsets[i]!
    let strideTerm ← natDimTerm (ctx ++ " strides") strides[i]!
    match off with
    | `(tritonExpr| $n:num) =>
        staticOffsets := staticOffsets.push (⟨n.raw⟩ : TSyntax `term)
    | `(tritonExpr| $($t:term)) =>
        staticOffsets := staticOffsets.push (← `(($t : Nat)))
    | `(tritonExpr| $id:ident) =>
        if env.any (fun entry => entry.1 == id.getId.toString) then
          staticOffsets := staticOffsets.push (← `((0 : Nat)))
          let off' ← expandNatExpectedExpr env off
          ensureShape SInfo.scalar off'.shape (ctx ++ " offsets")
          dynamicAdds := dynamicAdds.push
            (← `(Op.mul NumericDType.nat Broadcast.nil $off'.term (Op.constNat $strideTerm)))
        else
          let t : TSyntax `term := ⟨id.raw⟩
          staticOffsets := staticOffsets.push (← `(($t : Nat)))
    | _ =>
        staticOffsets := staticOffsets.push (← `((0 : Nat)))
        let off' ← expandNatExpectedExpr env off
        ensureShape SInfo.scalar off'.shape (ctx ++ " offsets")
        dynamicAdds := dynamicAdds.push
          (← `(Op.mul NumericDType.nat Broadcast.nil $off'.term (Op.constNat $strideTerm)))
  let rec listTerm : List (TSyntax `term) → MacroM (TSyntax `term)
    | [] => `(([] : List Nat))
    | d :: rest => do
        let tail ← listTerm rest
        `($d :: $tail)
  let rec sumOps : List (TSyntax `term) → MacroM (Option (TSyntax `term))
    | [] => pure none
    | [x] => pure (some x)
    | x :: xs => do
        match ← sumOps xs with
        | none => pure (some x)
        | some rest =>
            pure (some (← `(Op.add NumericDType.nat Broadcast.nil $x $rest)))
  pure (← listTerm staticOffsets.toList, ← sumOps dynamicAdds.toList)

partial def expandExpr (env : Env) (stx : TSyntax `tritonExpr) : MacroM EOut := do
  if stx.raw.getKind == ``tritonMethodCastElementTy ||
      stx.raw.getKind == ``tritonMethodCastElementTyIdent ||
      stx.raw.getKind == ``tritonIdentMethodCastElementTyIdent ||
      stx.raw.getKind == ``tritonIdentMethodCastElementTyIdentSpaced then
    let args := stx.raw.getArgs
    if h : args.size = 5 then
      let e : TSyntax `tritonExpr := ⟨args[0]⟩
      let e' ← expandExpr env e
      pure e'
    else if h : args.size = 6 then
      let e : TSyntax `tritonExpr := ⟨args[0]⟩
      let e' ← expandExpr env e
      pure e'
    else
      Macro.throwUnsupported
  else
  match methodCast? stx with
  | some (e, dt) =>
      let e' ← expandExpr env e
      let dst ← expandDType dt
      match e'.dtype, dst with
      | .nat, .int =>
          pure e'
      | _, .fp32 =>
          let srcProof ← e'.dtype.floatProof
          let algTerm ← `(Op.castFloat $srcProof FloatDType.real $e'.term)
          pure ⟨algTerm, .real, e'.shape, some (← fp32ComputeExpr algTerm), some .fp32⟩
      | _, _ =>
          let srcProof ← e'.dtype.floatProof
          let dstProof ← dst.floatProof
          pure ⟨← `(Op.castFloat $srcProof $dstProof $e'.term), dst, e'.shape, none, none⟩
  | none =>
  match stx with
  | `(tritonExpr| $n:num) =>
      -- Bare numeric literals are `ℝ` data constants (e.g. `1` in `1 / s`).
      pure ⟨← `(Op.const $n), .real, SInfo.scalar, none, none⟩
  | `(tritonExpr| $n:scientific) =>
      pure ⟨← `(Op.const $n), .real, SInfo.scalar, none, none⟩
  | `(tritonExpr| $i:ident) =>
      let name := i.getId.toString
      let (dtype, shape) ← lookupEnv env name
      let s ← identAsStr i
      let dt ← dtype.term
      let sh ← shape.term
      let term ← `(Op.ref $dt $sh $s)
      match lookupComputeDType? env name with
      | some .fp32 =>
          pure ⟨term, dtype, shape,
            some (← fp32ComputeExpr term),
            some .fp32⟩
      | some _ =>
          Macro.throwError
            ("identifier `" ++ name ++ "` has unsupported compute dtype annotation")
      | none =>
          pure ⟨term, dtype, shape, none, none⟩
  | `(tritonExpr| $($t:term)) =>
      -- `$(...)` antiquote is the address/size channel: `Nat`.
      -- Data/scalar contexts reinterpret the same surface form through
      -- `expandLeanAntiquoteAs? .real` before this default is used.
      pure ⟨← `(Op.constNat $t), .nat, SInfo.scalar, none, none⟩
  | `(tritonExpr| ($e:tritonExpr)) =>
      expandExpr env e
  | `(tritonExpr| tl.program_id($e:tritonExpr)) =>
      expandProgramId e
  | `(tritonExpr| tl.program_id(axis=$e:tritonExpr)) =>
      expandProgramId e
  | `(tritonExpr| tl.arange($e:tritonExpr)) =>
      expandArange e
  | `(tritonExpr| tl.arange($s:tritonExpr, $e:tritonExpr)) => do
      expandArangeRange s e
  | `(tritonExpr| tl.exp($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.exp" e'
      pure ⟨← `(Op.exp $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.exp2($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.exp2" e'
      pure ⟨← `(Op.exp2 $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.math.exp2($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.math.exp2" e'
      pure ⟨← `(Op.exp2 $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.log($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.log" e'
      pure ⟨← `(Op.log $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.log2($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.log2" e'
      pure ⟨← `(Op.log2 $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.sigmoid($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.sigmoid" e'
      pure ⟨← `(Op.sigmoid $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.sqrt($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.sqrt" e'
      pure ⟨← `(Op.sqrt $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.math.rsqrt($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.math.rsqrt" e'
      pure ⟨← `(Op.rsqrt $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.rsqrt($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.rsqrt" e'
      pure ⟨← `(Op.rsqrt $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.tanh($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.tanh" e'
      pure ⟨← `(Op.tanh $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.sin($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.sin" e'
      pure ⟨← `(Op.sin $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.math.sin($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.math.sin" e'
      pure ⟨← `(Op.sin $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.cos($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.cos" e'
      pure ⟨← `(Op.cos $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.tan($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.tan" e'
      pure ⟨← `(Op.tan $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.atan($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.atan" e'
      pure ⟨← `(Op.atan $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.cosh($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.cosh" e'
      pure ⟨← `(Op.cosh $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.sinh($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.sinh" e'
      pure ⟨← `(Op.sinh $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.erf($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.erf" e'
      pure ⟨← `(Op.erf $eTerm), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.extra.cuda.libdevice.erf($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.extra.cuda.libdevice.erf" e'
      pure ⟨← `(Op.erf $eTerm), .real, e'.shape, none, none⟩
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
      pure ⟨← `(Op.where $cond $neg $e'.term), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.logical_and($a:tritonExpr, $b:tritonExpr)) => do
      expandBoolBin expandExpr env "tl.logical_and" (← `(Op.boolAnd)) a b
  | `(tritonExpr| tl.logical_or($a:tritonExpr, $b:tritonExpr)) => do
      expandBoolBin expandExpr env "tl.logical_or" (← `(Op.boolOr)) a b
  | `(tritonExpr| tl.logical_not($a:tritonExpr)) => do
      expandBoolNot expandExpr env "tl.logical_not" a
  | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) => do
      expandCdiv expandExpr env a b
  | `(tritonExpr| tl.max($e:tritonExpr, $n:num)) => do
      expandReduce expandExpr env "tl.max" (← `(Op.reduceMax)) e
        #[← `(tritonReduceKwarg| $n:num)]
  | `(tritonExpr| tl.max($a:tritonExpr, $b:tritonExpr)) => do
      match b with
      | `(tritonExpr| $n:num) =>
          expandReduce expandExpr env "tl.max" (← `(Op.reduceMax)) a
            #[← `(tritonReduceKwarg| $n:num)]
      | _ =>
          let a' ← expandExpr env a
          let b' ← expandExpr env b
          ensureDType .real a'.dtype "tl.max"
          ensureDType .real b'.dtype "tl.max"
          let (bc, outShape) ← broadcastTerm a'.shape b'.shape "tl.max"
          pure ⟨← `(Op.max2 $bc $a'.term $b'.term), .real, outShape, none, none⟩
  | `(tritonExpr| tl.maximum($a:tritonExpr, $b:tritonExpr)) => do
      expandMinMax expandExpr env "tl.maximum" (← `(Op.gt)) a b
  | `(tritonExpr| tl.minimum($a:tritonExpr, $b:tritonExpr)) => do
      expandMinMax expandExpr env "tl.minimum" (← `(Op.lt)) a b
  | `(tritonExpr| tl.cumsum($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandScan expandExpr env "tl.cumsum" (← `(ScanOp.sum)) e kwargs
  | `(tritonExpr| tl.cumprod($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandScan expandExpr env "tl.cumprod" (← `(ScanOp.prod)) e kwargs
  | `(tritonExpr| tl.associative_scan($e:tritonExpr, $op:tritonScanOp $[, $kwargs:tritonReduceKwarg]*)) => do
      expandScan expandExpr env "tl.associative_scan" (← expandScanOp op) e kwargs
  | `(tritonExpr| tl.argmax($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandArgReduce expandExpr env "tl.argmax" (← `(Op.argMax)) e kwargs
  | `(tritonExpr| tl.argmin($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandArgReduce expandExpr env "tl.argmin" (← `(Op.argMin)) e kwargs
  | `(tritonExpr| tl.sort($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandSort expandExpr env e kwargs
  | `(tritonExpr| tl.sum($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandReduce expandExpr env "tl.sum" (← `(Op.reduceSum)) e kwargs
  | `(tritonExpr| tl.max($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandReduce expandExpr env "tl.max" (← `(Op.reduceMax)) e kwargs
  | `(tritonExpr| tl.toReal($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .nat e'.dtype "tl.toReal"
      pure ⟨← `(Op.natToReal $e'.term), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.multiple_of($e:tritonExpr, $_align:tritonExpr)) => do
      expandExpr env e
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
            some (← `(ComputeExpr.compute $op)), none⟩
      | _ =>
          let e' ← expandExpr env e
          let src ← inferComputeSourceDType "tl.bitcast" e'.dtype
          let srcTerm ← src.term
          let algTerm ← dst.placeholderAlgTerm e'.shape
          let op ←
            `(ComputeOp.bitcast $srcTerm $dstTerm rfl
                (ComputeOp.alg $srcTerm $e'.term))
          pure ⟨algTerm, dst.algDType, e'.shape,
            some (← `(ComputeExpr.compute $op)), none⟩
  | `(tritonExpr| tl.cast($e:tritonExpr, $dt:tritonDType)) => do
      let e' ← expandExpr env e
      let dst ← expandDType dt
      match dst with
      | .fp32 =>
          let srcProof ← e'.dtype.floatProof
          let algTerm ← `(Op.castFloat $srcProof FloatDType.real $e'.term)
          pure ⟨algTerm, .real, e'.shape, some (← fp32ComputeExpr algTerm), some .fp32⟩
      | _ =>
          let srcProof ← e'.dtype.floatProof
          let dstProof ← dst.floatProof
          pure ⟨← `(Op.castFloat $srcProof $dstProof $e'.term), dst, e'.shape, none, none⟩
  | `(tritonExpr| -inf) =>
      pure ⟨← `(Op.negInf), .real, SInfo.scalar, none, none⟩
  | `(tritonExpr| - float ($arg:term)) =>
      let argString := toString arg.raw
      unless argString.contains "inf" do
        Macro.throwError "float(...): only `float(\"inf\")` is supported in Triton DSL expressions"
      pure ⟨← `(Op.negInf), .real, SInfo.scalar, none, none⟩
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr)) => do
      expandDot expandExpr env a b
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr, $acc:tritonExpr)) => do
      -- Fused accumulator form: `tl.dot(a, b, acc) ≡ acc + tl.dot(a, b)`.
      -- Both `tl.dot(a, b)` and `acc` have shape `[M, N]`; their `+` uses
      -- the standard same-shape broadcast.
      let dot ← expandDot expandExpr env a b
      let acc' ← expandExpr env acc
      ensureDType .real acc'.dtype "tl.dot accumulator"
      ensureShape dot.shape acc'.shape "tl.dot accumulator"
      let (bc, outShape) ← broadcastTerm dot.shape acc'.shape "tl.dot accumulator"
      pure ⟨← `(Op.add NumericDType.real $bc $dot.term $acc'.term),
            .real, outShape, none, none⟩
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
      let (region, baseFromPtr?) ← match sp with
        | some sp =>
            pure (sp.region, if sp.baseOnly then none else some sp.offset)
        | none =>
            Macro.throwError "tl.make_block_ptr: first argument must be a region antiquote like `$(xReg)`"
      let base' ←
        match ← expandLeanAntiquoteAs? .nat base with
        | some out => pure out
        | none =>
            match base with
            | `(tritonExpr| $n:num) =>
                pure ⟨← `(Op.constNat $n), .nat, SInfo.scalar, none, none⟩
            | _ => expandNatExpectedExpr env base
      ensureShape SInfo.scalar base'.shape "tl.make_block_ptr base"
      let baseTerm ←
        match baseFromPtr? with
        | none => pure base'.term
        | some ptrBase => do
            let (bc, outShape) ← broadcastTerm SInfo.scalar SInfo.scalar "tl.make_block_ptr base"
            ensureShape SInfo.scalar outShape "tl.make_block_ptr base"
            `(Op.add NumericDType.nat $bc $ptrBase $base'.term)
      let (parentShape, _) ← natListTerm "tl.make_block_ptr shape" parentDims
      let (strides, _) ← natListTerm "tl.make_block_ptr strides" strideDims
      let (offsets, dynOffsetAdd?) ←
        natOffsetsWithBaseAdd env "tl.make_block_ptr" offsetDims strideDims
      let baseTerm ←
        match dynOffsetAdd? with
        | none => pure baseTerm
        | some dyn =>
            `(Op.add NumericDType.nat Broadcast.nil $baseTerm $dyn)
      let (blockShape, blockShapeInfo) ← natListTerm "tl.make_block_ptr block_shape" blockDims
      pure ⟨← `(Op.makeBlockPtrDyn $region $baseTerm $parentShape $blockShape $strides $offsets),
            .blockPtr, .dims blockShapeInfo, none, none⟩
  | `(tritonExpr| tl.make_block_ptr($baseKw:ident=$p:tritonExpr,
        $shapeKw:ident=($parentDims:tritonExpr,*), $stridesKw:ident=($strideDims:tritonExpr,*),
        $offsetsKw:ident=($offsetDims:tritonExpr,*), $blockShapeKw:ident=($blockDims:tritonExpr,*),
        $orderKw:ident=($_order:num,*))) => do
      unless baseKw.getId.toString == "base" &&
          shapeKw.getId.toString == "shape" &&
          stridesKw.getId.toString == "strides" &&
          offsetsKw.getId.toString == "offsets" &&
          blockShapeKw.getId.toString == "block_shape" &&
          orderKw.getId.toString == "order" do
        Macro.throwError "tl.make_block_ptr kwargs must be `base`, `shape`, `strides`, `offsets`, `block_shape`, optional `order` in that order"
      let sp ← expandStaticPtrExpr env p
      let (region, baseTerm, baseShape) ← match sp with
        | some sp => pure (sp.region, sp.offset, sp.shape)
        | none => Macro.throwError "tl.make_block_ptr: `base=` must start from a region expression"
      ensureShape SInfo.scalar baseShape "tl.make_block_ptr base"
      let (parentShape, _) ← natListTerm "tl.make_block_ptr shape" parentDims
      let (strides, _) ← natListTerm "tl.make_block_ptr strides" strideDims
      let (offsets, dynOffsetAdd?) ←
        natOffsetsWithBaseAdd env "tl.make_block_ptr" offsetDims strideDims
      let baseTerm ←
        match dynOffsetAdd? with
        | none => pure baseTerm
        | some dyn =>
            `(Op.add NumericDType.nat Broadcast.nil $baseTerm $dyn)
      let (blockShape, blockShapeInfo) ← natListTerm "tl.make_block_ptr block_shape" blockDims
      pure ⟨← `(Op.makeBlockPtrDyn $region $baseTerm $parentShape $blockShape $strides $offsets),
            .blockPtr, .dims blockShapeInfo, none, none⟩
  | `(tritonExpr| tl.make_block_ptr($baseKw:ident=$p:tritonExpr,
        $shapeKw:ident=($parentDims:tritonExpr,*), $stridesKw:ident=($strideDims:tritonExpr,*),
        $offsetsKw:ident=($offsetDims:tritonExpr,*), $blockShapeKw:ident=($blockDims:tritonExpr,*))) => do
      unless baseKw.getId.toString == "base" &&
          shapeKw.getId.toString == "shape" &&
          stridesKw.getId.toString == "strides" &&
          offsetsKw.getId.toString == "offsets" &&
          blockShapeKw.getId.toString == "block_shape" do
        Macro.throwError "tl.make_block_ptr kwargs must be `base`, `shape`, `strides`, `offsets`, `block_shape` in that order"
      let sp ← expandStaticPtrExpr env p
      let (region, baseTerm, baseShape) ← match sp with
        | some sp => pure (sp.region, sp.offset, sp.shape)
        | none => Macro.throwError "tl.make_block_ptr: `base=` must start from a region expression"
      ensureShape SInfo.scalar baseShape "tl.make_block_ptr base"
      let (parentShape, _) ← natListTerm "tl.make_block_ptr shape" parentDims
      let (strides, _) ← natListTerm "tl.make_block_ptr strides" strideDims
      let (offsets, dynOffsetAdd?) ←
        natOffsetsWithBaseAdd env "tl.make_block_ptr" offsetDims strideDims
      let baseTerm ←
        match dynOffsetAdd? with
        | none => pure baseTerm
        | some dyn =>
            `(Op.add NumericDType.nat Broadcast.nil $baseTerm $dyn)
      let (blockShape, blockShapeInfo) ← natListTerm "tl.make_block_ptr block_shape" blockDims
      pure ⟨← `(Op.makeBlockPtrDyn $region $baseTerm $parentShape $blockShape $strides $offsets),
            .blockPtr, .dims blockShapeInfo, none, none⟩
  | `(tritonExpr| tl.advance($p:tritonExpr, [$deltas:tritonExpr,*])) => do
      let p' ← expandExpr env p
      ensureDType .blockPtr p'.dtype "tl.advance pointer"
      let (deltasTerm, _) ← natListTerm "tl.advance offsets" deltas
<<<<<<< Updated upstream
      pure ⟨← `(Op.advanceBlockPtr (d := TileDType.real) $p'.term $deltasTerm), .blockPtr, p'.shape, none⟩
=======
      pure ⟨← `(Op.advanceBlockPtr $p'.term $deltasTerm), .blockPtr, p'.shape, none, none⟩
  | `(tritonExpr| tl.advance($p:tritonExpr, $offsetsKw:ident=($deltas:tritonExpr,*))) => do
      unless offsetsKw.getId.toString == "offsets" do
        Macro.throwError "tl.advance kwarg must be `offsets`"
      let p' ← expandExpr env p
      ensureDType .blockPtr p'.dtype "tl.advance pointer"
      let (deltasTerm, _) ← natListTerm "tl.advance offsets" deltas
      pure ⟨← `(Op.advanceBlockPtr $p'.term $deltasTerm), .blockPtr, p'.shape, none, none⟩
>>>>>>> Stashed changes
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      expandLoad expandExpr expandStaticPtrExpr env p kwargs
  | `(tritonExpr| $a:tritonExpr < $b:tritonExpr) => do
      expandCmp expandExpr env "comparison" (← `(Op.lt)) a b
  | `(tritonExpr| $a:tritonExpr <= $b:tritonExpr) => do
      expandCmp expandExpr env "comparison" (← `(Op.le)) a b
  | `(tritonExpr| $a:tritonExpr == $b:tritonExpr) => do
      expandCmp expandExpr env "comparison" (← `(Op.eq)) a b
  | `(tritonExpr| $a:tritonExpr > $b:tritonExpr) => do
      expandCmp expandExpr env "comparison" (← `(Op.gt)) a b
  | `(tritonExpr| $a:tritonExpr >= $b:tritonExpr) => do
      expandCmp expandExpr env "comparison" (← `(Op.ge)) a b
  | `(tritonExpr| $a:tritonExpr != $b:tritonExpr) => do
      expandCmp expandExpr env "comparison" (← `(Op.ne)) a b
  | `(tritonExpr| $a:tritonExpr & $b:tritonExpr) => do
      expandBoolOrNatBitwise expandExpr env "`&`" (← `(Op.boolAnd)) (← `(Op.bitAnd)) a b
  | `(tritonExpr| $a:tritonExpr and $b:tritonExpr) => do
      expandBoolBin expandExpr env "`and`" (← `(Op.boolAnd)) a b
  | `(tritonExpr| $a:tritonExpr | $b:tritonExpr) => do
      expandBoolOrNatBitwise expandExpr env "`|`" (← `(Op.boolOr)) (← `(Op.bitOr)) a b
  | `(tritonExpr| $a:tritonExpr ^ $b:tritonExpr) => do
      expandNatBitwise expandExpr env "`^`" (← `(Op.bitXor)) a b
  | `(tritonExpr| $a:tritonExpr << $b:tritonExpr) => do
      expandNatBitwise expandExpr env "`<<`" (← `(Op.shiftLeft)) a b
  | `(tritonExpr| $a:tritonExpr >> $b:tritonExpr) => do
      expandNatBitwise expandExpr env "`>>`" (← `(Op.shiftRight)) a b
  | `(tritonExpr| ~$a:tritonExpr) => do
      expandBoolNot expandExpr env "boolean ~" a
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
      match ← expandStaticPtrExpr env stx with
      | some sp =>
          let (bc, _) ← broadcastTerm SInfo.scalar sp.shape "pointer arithmetic"
<<<<<<< Updated upstream
          pure ⟨← `(Op.ptrAdd (d := TileDType.real) $bc (Op.ptrBase $sp.region) $sp.offset), .ptr, sp.shape, none⟩
=======
          pure ⟨← `(Op.ptrAdd $bc (Op.ptrBase $sp.region) $sp.offset), .ptr, sp.shape, none, none⟩
>>>>>>> Stashed changes
      | none =>
          let a' ← expandExpr env a
          let b' ← expandExpr env b
          match a'.dtype, b'.dtype with
          | .ptr, .nat =>
              ensureAlgorithmOnly "pointer arithmetic" a'
              ensureAlgorithmOnly "pointer arithmetic" b'
              let (bc, outShape) ← broadcastTerm a'.shape b'.shape "pointer arithmetic"
<<<<<<< Updated upstream
              pure ⟨← `(Op.ptrAdd (d := TileDType.real) $bc $a'.term $b'.term), .ptr, outShape, none⟩
=======
              pure ⟨← `(Op.ptrAdd $bc $a'.term $b'.term), .ptr, outShape, none, none⟩
>>>>>>> Stashed changes
          | .nat, .ptr =>
              ensureAlgorithmOnly "pointer arithmetic" a'
              ensureAlgorithmOnly "pointer arithmetic" b'
              let (bc, outShape) ← broadcastTerm b'.shape a'.shape "pointer arithmetic"
<<<<<<< Updated upstream
              pure ⟨← `(Op.ptrAdd (d := TileDType.real) $bc $b'.term $a'.term), .ptr, outShape, none⟩
=======
              pure ⟨← `(Op.ptrAdd $bc $b'.term $a'.term), .ptr, outShape, none, none⟩
>>>>>>> Stashed changes
          | _, _ =>
              expandArith expandExpr env "arithmetic" (← `(Op.add)) a b
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) => do
      expandArith expandExpr env "arithmetic" (← `(Op.sub)) a b
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => do
      expandArith expandExpr env "arithmetic" (← `(Op.mul)) a b
  | `(tritonExpr| $a:tritonExpr / $b:tritonExpr) => do
      expandArith expandExpr env "arithmetic" (← `(Op.div)) a b
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => do
      expandIntegralArith expandExpr env "integer floor division" (← `(Op.floorDiv)) a b
  | `(tritonExpr| $a:tritonExpr % $b:tritonExpr) => do
      expandIntegralArith expandExpr env "integer remainder" (← `(Op.mod)) a b
  | `(tritonExpr| tl.where($c:tritonExpr, $a:tritonExpr, $b:tritonExpr)) => do
      let c' ← expandExpr env c
      expandWhereFromCond env c' a b
  | `(tritonExpr| $e:tritonExpr[ : , None ]) => do
      -- `e[:, None]` — insert a unit axis at position 1: `[N] → [N, 1]`.
      expandSlicerNone expandExpr env e (axisIdx := 1)
  | `(tritonExpr| $e:tritonExpr[ None , : ]) => do
      -- `e[None, :]` — insert a unit axis at position 0: `[N] → [1, N]`.
      expandSlicerNone expandExpr env e (axisIdx := 0)
  | `(tritonExpr| tl.expand_dims($e:tritonExpr, $kw:tritonReduceKwarg)) => do
      match kw with
      | `(tritonReduceKwarg| axis = $n:num) =>
          expandExpandDims expandExpr env e (axisIdx := n.getNat)
      | `(tritonReduceKwarg| $name:ident = $_) =>
          Macro.throwError
            ("tl.expand_dims: unknown kwarg `" ++ name.getId.toString ++
             "`. Only literal `axis = N` is supported.")
      | _ => Macro.throwUnsupported
  | `(tritonExpr| tl.expand_dims($e:tritonExpr, $n:num)) => do
      expandExpandDims expandExpr env e (axisIdx := n.getNat)
  | `(tritonExpr| tl.trans($e:tritonExpr)) => do
      -- `tl.trans(e)` — transpose the trailing two axes (`Op.transpose`).
      expandTranspose expandExpr env e
  | `(tritonExpr| tl.permute($e:tritonExpr, [$axes:num,*])) => do
      expandPermute expandExpr env e axes.getElems.toList
  | `(tritonExpr| tl.reshape($e:tritonExpr, [$dims:tritonExpr,*])) => do
      expandReshapeLike expandExpr env "tl.reshape" e dims.getElems
  | `(tritonExpr| tl.view($e:tritonExpr, [$dims:tritonExpr,*])) => do
      expandReshapeLike expandExpr env "tl.view" e dims.getElems
  | `(tritonExpr| tl.ravel($e:tritonExpr)) => do
      expandRavel expandExpr env e
  | `(tritonExpr| tl.flip($e:tritonExpr, $ax:num)) => do
      expandFlip expandExpr env e ax.getNat
  | `(tritonExpr| tl.flip($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      let ax ← parseFlipKwargs kwargs
      expandFlip expandExpr env e ax
  | `(tritonExpr| tl.join($a:tritonExpr, $b:tritonExpr)) => do
      expandJoin expandExpr env a b
  | `(tritonExpr| tl.split($e:tritonExpr, $side:num)) => do
      expandSplit expandExpr env e side.getNat
<<<<<<< Updated upstream
  | `(tritonExpr| tl.full([$dims:tritonExpr,*], $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      let dtypeHint ← extractDTypeKwarg kwargs
      expandFull expandExpr env dims.getElems v dtypeHint
  | `(tritonExpr| tl.zeros([$dims:tritonExpr,*] $[, $kwargs:tritonMemKwarg]*)) => do
      let dtypeHint ← extractDTypeKwarg kwargs
      let zero ← `(tritonExpr| 0)
      expandFull expandExpr env dims.getElems zero dtypeHint
=======
  | `(tritonExpr| tl.full([$dims:tritonExpr,*], $v:tritonExpr)) => do
      expandFull expandExpr env dims.getElems v
  | `(tritonExpr| tl.full([$dims:tritonExpr,*], $v:tritonExpr, $name:ident=$dt:tritonDType)) => do
      unless name.getId.getString! == "dtype" do
        Macro.throwError
          ("tl.full: unknown kwarg `" ++ name.getId.toString ++ "`. Only `dtype=` is recognized.")
      expandComputeFull expandExpr env dims.getElems v dt
  | `(tritonExpr| tl.zeros([$dims:tritonExpr,*])) => do
      -- `tl.zeros([dims])` ≡ `tl.full([dims], 0)`.
      let zero ← `(tritonExpr| 0)
      expandFull expandExpr env dims.getElems zero
  | `(tritonExpr| tl.zeros([$dims:tritonExpr,*], $name:ident=$dt:tritonDType)) => do
      unless name.getId.getString! == "dtype" do
        Macro.throwError
          ("tl.zeros: unknown kwarg `" ++ name.getId.toString ++ "`. Only `dtype=` is recognized.")
      expandComputeZeros expandExpr env dims.getElems dt
>>>>>>> Stashed changes
  | _ => (Macro.throwUnsupported : MacroM EOut)

end

mutual

partial def expandStmt (env : Env) (pinned : List String)
    (regionDTypes : Inference.RegionDTypes) (ptrElems : Inference.PtrElems)
    (stx : TSyntax `tritonStmt) :
    MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
  let unsupportedAtomic2 (op : String) (p v : TSyntax `tritonExpr) :
      MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
    discard <| expandExpr env p
    discard <| expandExpr env v
    let opTerm : TSyntax `term := ⟨Syntax.mkStrLit op⟩
    pure (← `(Stmt.ifThen (Op.constBool Bool.false) []),
      ← `(ComputeStmt.effectMarker $opTerm), env, Bool.true)
  let ensureNoAtomicKwargs (op : String) (kwargs : TSyntaxArray `tritonMemKwarg) :
      MacroM Unit := do
    unless kwargs.isEmpty do
      Macro.throwError (op ++ ": kwargs are not modeled for return-valued atomics yet")
  let expandAtomicRMWCore (opName : String) (opTerm : TSyntax `term)
      (dest : Option Ident) (p input : TSyntax `tritonExpr)
      (extraInput : Option (TSyntax `tritonExpr)) :
      MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
    let input' ← expandExpr env input
    ensureAlgorithmOnly opName input'
    let extra' ←
      match extraInput with
      | none => pure none
      | some extra => do
          let e ← expandExpr env extra
          ensureAlgorithmOnly opName e
          ensureDType input'.dtype e.dtype opName
          pure (some e)
    let mkTerms (memTerm : TSyntax `term) (targetShape : SInfo) :
        MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
      let inputTerm ← coerceShape input'.term input'.shape targetShape opName
      let extraTerm? ←
        match extra' with
        | none => pure none
        | some e => pure (some (← coerceShape e.term e.shape targetShape opName))
      let dt ← input'.dtype.term
      let sh ← targetShape.term
      let destTerm ←
        match dest with
        | none => `(Option.none)
        | some ident =>
            let lit ← identAsStr ⟨ident.raw⟩
            `(Option.some $lit)
      let extraAlgTerm ←
        match extraTerm? with
        | none => `(Option.none)
        | some t => `(Option.some $t)
      let extraComputeTerm ←
        match extraTerm? with
        | none => `(Option.none)
        | some t => `(Option.some (ComputeExpr.alg $t))
      let alg ←
        `(Stmt.atomicRMW $opTerm $dt $sh $memTerm $inputTerm $extraAlgTerm
            (MaskOpt.none) $destTerm)
      let compute ←
        `(ComputeStmt.atomicRMW $opTerm $dt $sh $memTerm
            (ComputeExpr.alg $inputTerm) $extraComputeTerm (MaskOpt.none) $destTerm)
      let nextEnv :=
        match dest with
        | none => env
        | some ident => (ident.getId.toString, input'.dtype, targetShape, none) :: env
      pure (alg, compute, nextEnv, Bool.false)
    match ← expandStaticPtrExpr env p with
    | some sp =>
        mkTerms (← `(MemAccess.region $sp.region $sp.offset)) sp.shape
    | none =>
        let p' ← expandExpr env p
        ensureDType .ptr p'.dtype (opName ++ " pointer")
        mkTerms (← `(MemAccess.ptr $p'.term)) p'.shape
  let loadDTypeHint (dest : Ident) (p : TSyntax `tritonExpr) : MacroM (Option DInfo) := do
    let regionHint : Option DInfo ←
      match ← expandStaticPtrExpr env p with
      | some sp =>
          match regionTermName sp.region with
          | some name => pure (Inference.lookupRegionDType regionDTypes name)
          | none => pure none
      | none =>
          match p with
          | `(tritonExpr| $r:ident) => pure (Inference.lookupPtrElem ptrElems r.getId.toString)
          | _ => pure none
    let pinHint : Option DInfo :=
      if pinned.contains dest.getId.toString then some .nat else none
    pure (regionHint.orElse (fun _ => pinHint))
  let emitAssign (dest : Ident) (e' : EOut) : MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
    let nameLit ← identAsStr dest
    let dt ← e'.dtype.term
    let sh ← e'.shape.term
    let exprTerm ←
      match e'.computeTerm with
      | some ce => pure ce
      | none => `(ComputeExpr.alg $e'.term)
    pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
      ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
      (dest.getId.toString, e'.dtype, e'.shape) :: env,
      e'.computeTerm.isSome)
  let expandLoadAssign (dest : Ident) (p : TSyntax `tritonExpr)
      (kwargs : TSyntaxArray `tritonMemKwarg) :
      MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
    let hint ← loadDTypeHint dest p
    let e' ← expandLoad expandExpr expandStaticPtrExpr env p kwargs (defaultDType := hint)
    emitAssign dest e'
  let expandLoadSubAssign (dest : Ident) (p : TSyntax `tritonExpr)
      (kwargs : TSyntaxArray `tritonMemKwarg) (rhs : TSyntax `tritonExpr) :
      MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
    let hint ← loadDTypeHint dest p
    let load' ← expandLoad expandExpr expandStaticPtrExpr env p kwargs (defaultDType := hint)
    ensureAlgorithmOnly "arithmetic" load'
    let rhs' ←
      match ← expandLeanAntiquoteAs? load'.dtype rhs with
      | some out => pure out
      | none => expandExpr env rhs
    ensureAlgorithmOnly "arithmetic" rhs'
    ensureDType load'.dtype rhs'.dtype "arithmetic"
    let np ← load'.dtype.numericProof
    let (bc, outShape) ← broadcastTerm load'.shape rhs'.shape "arithmetic"
    let term ← `(Op.sub $np $bc $load'.term $rhs'.term)
    emitAssign dest { term := term, dtype := load'.dtype, shape := outShape, computeTerm := none }
  match stx with
  | `(tritonStmt| $lhs0:ident, $lhs1:ident $[, $lhsRest:ident]* = $rhs0:tritonExpr, $rhs1:tritonExpr $[, $rhsRest:tritonExpr]*) => do
      let lhs := #[lhs0, lhs1] ++ lhsRest
      let rhs := #[rhs0, rhs1] ++ rhsRest
      unless lhs.size == rhs.size do
        Macro.throwError
          ("multiple assignment: left side has " ++ toString lhs.size ++
           " targets but right side has " ++ toString rhs.size ++ " values")
      let mut algStmts : Array (TSyntax `term) := #[]
      let mut computeStmts : Array (TSyntax `term) := #[]
      let mut env' := env
      let mut hasCompute := Bool.false
      let mut expanded : Array EOut := #[]
      for expr in rhs do
        let e' ← expandExpr env expr
        expanded := expanded.push e'
        hasCompute := hasCompute || e'.computeTerm.isSome
      for idx in [:lhs.size] do
        let name := lhs[idx]!
        let e' := expanded[idx]!
        let nameLit ← identAsStr name
        let dt ← e'.dtype.term
        let sh ← e'.shape.term
        let exprTerm ←
          match e'.computeTerm with
          | some ce => pure ce
          | none => `(ComputeExpr.alg $e'.term)
        algStmts := algStmts.push (← `(Stmt.assign $dt $sh $nameLit $e'.term))
        computeStmts := computeStmts.push
          (← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm))
        env' := (name.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env'
      pure (← `(Stmt.ifThen (Op.constBool Bool.true) [$algStmts,*]),
        ← `(ComputeStmt.ifThen (Op.constBool Bool.true) [$computeStmts,*]),
        env', hasCompute)
  | `(tritonStmt| $lhs0:ident, $lhs1:ident $[, $lhsRest:ident]* := $rhs0:tritonExpr, $rhs1:tritonExpr $[, $rhsRest:tritonExpr]*) => do
      let lhs := #[lhs0, lhs1] ++ lhsRest
      let rhs := #[rhs0, rhs1] ++ rhsRest
      unless lhs.size == rhs.size do
        Macro.throwError
          ("multiple assignment: left side has " ++ toString lhs.size ++
           " targets but right side has " ++ toString rhs.size ++ " values")
      let mut algStmts : Array (TSyntax `term) := #[]
      let mut computeStmts : Array (TSyntax `term) := #[]
      let mut env' := env
      let mut hasCompute := Bool.false
      let mut expanded : Array EOut := #[]
      for expr in rhs do
        let e' ← expandExpr env expr
        expanded := expanded.push e'
        hasCompute := hasCompute || e'.computeTerm.isSome
      for idx in [:lhs.size] do
        let name := lhs[idx]!
        let e' := expanded[idx]!
        let nameLit ← identAsStr name
        let dt ← e'.dtype.term
        let sh ← e'.shape.term
        let exprTerm ←
          match e'.computeTerm with
          | some ce => pure ce
          | none => `(ComputeExpr.alg $e'.term)
        algStmts := algStmts.push (← `(Stmt.assign $dt $sh $nameLit $e'.term))
        computeStmts := computeStmts.push
          (← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm))
        env' := (name.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env'
      pure (← `(Stmt.ifThen (Op.constBool Bool.true) [$algStmts,*]),
        ← `(ComputeStmt.ifThen (Op.constBool Bool.true) [$computeStmts,*]),
        env', hasCompute)
  | `(tritonStmt| $i:ident = tl.max($e:tritonExpr, $n:num)) => do
      let nameLit ← identAsStr i
      let e' ← expandReduce expandExpr env "tl.max" (← `(Op.reduceMax)) e
        #[← `(tritonReduceKwarg| $n:num)]
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        (i.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env,
        e'.computeTerm.isSome)
  | `(tritonStmt| $i:ident := tl.max($e:tritonExpr, $n:num)) => do
      let nameLit ← identAsStr i
      let e' ← expandReduce expandExpr env "tl.max" (← `(Op.reduceMax)) e
        #[← `(tritonReduceKwarg| $n:num)]
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        (i.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env,
        e'.computeTerm.isSome)
  | `(tritonStmt| $valueName:ident, $indexName:ident = tl.max($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.max return_indices"
      let dims := match e'.shape with | .dims ds => ds
      let axisIdx ← parseMaxReturnIndicesKwargs dims kwargs
      let outDims ← eraseNth dims axisIdx
      let outShape := SInfo.dims outDims
      let outShapeTerm ← outShape.term
      let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
      let valueLit ← identAsStr valueName
      let indexLit ← identAsStr indexName
      let valueOp ← `(Op.reduceMax (⟨$axisLit, by simp⟩) Bool.false $e'.term)
      let indexOp ← `(Op.argMax (⟨$axisLit, by simp⟩) $e'.term)
      let valueStmt ← `(Stmt.assign TileDType.real $outShapeTerm $valueLit $valueOp)
      let indexStmt ← `(Stmt.assign TileDType.nat $outShapeTerm $indexLit $indexOp)
      let valueCompute ← `(ComputeStmt.assign TileDType.real $outShapeTerm $valueLit (ComputeExpr.alg $valueOp))
      let indexCompute ← `(ComputeStmt.assign TileDType.nat $outShapeTerm $indexLit (ComputeExpr.alg $indexOp))
      let env' :=
        (indexName.getId.toString, DInfo.nat, outShape, none) ::
          (valueName.getId.toString, DInfo.real, outShape, none) :: env
      pure (← `(Stmt.ifThen (Op.constBool Bool.true) [$valueStmt, $indexStmt]),
        ← `(ComputeStmt.ifThen (Op.constBool Bool.true) [$valueCompute, $indexCompute]),
        env', Bool.false)
  | `(tritonStmt| $valueName:ident, $indexName:ident := tl.max($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.max return_indices"
      let dims := match e'.shape with | .dims ds => ds
      let axisIdx ← parseMaxReturnIndicesKwargs dims kwargs
      let outDims ← eraseNth dims axisIdx
      let outShape := SInfo.dims outDims
      let outShapeTerm ← outShape.term
      let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
      let valueLit ← identAsStr valueName
      let indexLit ← identAsStr indexName
      let valueOp ← `(Op.reduceMax (⟨$axisLit, by simp⟩) Bool.false $e'.term)
      let indexOp ← `(Op.argMax (⟨$axisLit, by simp⟩) $e'.term)
      let valueStmt ← `(Stmt.assign TileDType.real $outShapeTerm $valueLit $valueOp)
      let indexStmt ← `(Stmt.assign TileDType.nat $outShapeTerm $indexLit $indexOp)
      let valueCompute ← `(ComputeStmt.assign TileDType.real $outShapeTerm $valueLit (ComputeExpr.alg $valueOp))
      let indexCompute ← `(ComputeStmt.assign TileDType.nat $outShapeTerm $indexLit (ComputeExpr.alg $indexOp))
      let env' :=
        (indexName.getId.toString, DInfo.nat, outShape, none) ::
          (valueName.getId.toString, DInfo.real, outShape, none) :: env
      pure (← `(Stmt.ifThen (Op.constBool Bool.true) [$valueStmt, $indexStmt]),
        ← `(ComputeStmt.ifThen (Op.constBool Bool.true) [$valueCompute, $indexCompute]),
        env', Bool.false)
  | `(tritonStmt| $i:ident := tl.atomic_xchg($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      ensureNoAtomicKwargs "tl.atomic_xchg" kwargs
      expandAtomicRMWCore "tl.atomic_xchg" (← `(RMWOp.xchg)) (some i) p v none
  | `(tritonStmt| $i:ident := tl.atomic_cas($p:tritonExpr, $cmp:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      ensureNoAtomicKwargs "tl.atomic_cas" kwargs
      expandAtomicRMWCore "tl.atomic_cas" (← `(RMWOp.cas)) (some i) p cmp (some v)
  | `(tritonStmt| $i:ident = tl.atomic_xchg($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      ensureNoAtomicKwargs "tl.atomic_xchg" kwargs
      expandAtomicRMWCore "tl.atomic_xchg" (← `(RMWOp.xchg)) (some i) p v none
  | `(tritonStmt| $i:ident = tl.atomic_cas($p:tritonExpr, $cmp:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      ensureNoAtomicKwargs "tl.atomic_cas" kwargs
      expandAtomicRMWCore "tl.atomic_cas" (← `(RMWOp.cas)) (some i) p cmp (some v)
  | `(tritonStmt| $i:ident := tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*) - $rhs:tritonExpr) => do
      expandLoadSubAssign i p kwargs rhs
  | `(tritonStmt| $i:ident = tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*) - $rhs:tritonExpr) => do
      expandLoadSubAssign i p kwargs rhs
  | `(tritonStmt| $i:ident := tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
<<<<<<< Updated upstream
      expandLoadAssign i p kwargs
  | `(tritonStmt| $i:ident = tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      expandLoadAssign i p kwargs
=======
      let nameLit ← identAsStr i
      let hint : Option DInfo :=
        if pinned.contains i.getId.toString then some .nat else none
      let e' ← expandLoad expandExpr expandStaticPtrExpr env p kwargs (defaultDType := hint)
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        (i.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env,
        e'.computeTerm.isSome)
  | `(tritonStmt| $i:ident = tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      let nameLit ← identAsStr i
      let hint : Option DInfo :=
        if pinned.contains i.getId.toString then some .nat else none
      let e' ← expandLoad expandExpr expandStaticPtrExpr env p kwargs (defaultDType := hint)
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        (i.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env,
        e'.computeTerm.isSome)
>>>>>>> Stashed changes
  | `(tritonStmt| $i:ident := $e:tritonExpr) => do
      let nameLit ← identAsStr i
      let e' ←
        if pinned.contains i.getId.toString then
          expandNatExpectedExpr env e
        else
          expandExpr env e
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        (i.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env,
        e'.computeTerm.isSome)
  | `(tritonStmt| $i:ident = $e:tritonExpr) => do
      let nameLit ← identAsStr i
      let e' ←
        if pinned.contains i.getId.toString then
          expandNatExpectedExpr env e
        else
          expandExpr env e
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        (i.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env,
        e'.computeTerm.isSome)
  | `(tritonStmt| $i:ident += $e:tritonExpr) => do
      let nameLit ← identAsStr i
      let iExpr : TSyntax `tritonExpr := ⟨i.raw⟩
      let sum ← `(tritonExpr| $iExpr + $e)
      let e' ← expandExpr env sum
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        (i.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env,
        e'.computeTerm.isSome)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      expandStore expandExpr expandStaticPtrExpr env p v kwargs
  | `(tritonStmt| tl.atomic_add($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      expandAtomicAdd expandExpr expandStaticPtrExpr env p v kwargs
  | `(tritonStmt| tl.atomic_max($p:tritonExpr, $v:tritonExpr $[, $_kwargs:tritonMemKwarg]*)) =>
      unsupportedAtomic2 "tl.atomic_max" p v
  | `(tritonStmt| tl.atomic_min($p:tritonExpr, $v:tritonExpr $[, $_kwargs:tritonMemKwarg]*)) =>
      unsupportedAtomic2 "tl.atomic_min" p v
  | `(tritonStmt| tl.atomic_and($p:tritonExpr, $v:tritonExpr $[, $_kwargs:tritonMemKwarg]*)) =>
      unsupportedAtomic2 "tl.atomic_and" p v
  | `(tritonStmt| tl.atomic_or($p:tritonExpr, $v:tritonExpr $[, $_kwargs:tritonMemKwarg]*)) =>
      unsupportedAtomic2 "tl.atomic_or" p v
  | `(tritonStmt| tl.atomic_xor($p:tritonExpr, $v:tritonExpr $[, $_kwargs:tritonMemKwarg]*)) =>
      unsupportedAtomic2 "tl.atomic_xor" p v
  | `(tritonStmt| tl.atomic_xchg($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      ensureNoAtomicKwargs "tl.atomic_xchg" kwargs
      expandAtomicRMWCore "tl.atomic_xchg" (← `(RMWOp.xchg)) none p v none
  | `(tritonStmt| tl.atomic_cas($p:tritonExpr, $cmp:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      ensureNoAtomicKwargs "tl.atomic_cas" kwargs
      expandAtomicRMWCore "tl.atomic_cas" (← `(RMWOp.cas)) none p cmp (some v)
  | `(tritonStmt| tl.async_copy($dst:tritonExpr, $src:tritonExpr $[, $_kwargs:tritonMemKwarg]*)) => do
      discard <| expandExpr env dst
      discard <| expandExpr env src
      pure (← `(Stmt.ifThen (Op.constBool Bool.false) []),
        ← `(ComputeStmt.effectMarker "tl.async_copy"), env, Bool.true)
  | `(tritonStmt| tl.async_wait()) =>
      pure (← `(Stmt.ifThen (Op.constBool Bool.false) []),
        ← `(ComputeStmt.effectMarker "tl.async_wait"), env, Bool.true)
  | `(tritonStmt| tl.debug_barrier()) =>
      pure (← `(Stmt.ifThen (Op.constBool Bool.false) []),
        ← `(ComputeStmt.effectMarker "tl.debug_barrier"), env, Bool.true)
  | `(tritonStmt| tl.for $i:ident in $($n:term) { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
<<<<<<< Updated upstream
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned regionDTypes ptrElems stmts.toList
=======
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned stmts.toList
>>>>>>> Stashed changes
      pure (← `(Stmt.forLoop $nameLit $n [$algBody,*]),
        ← `(ComputeStmt.forLoop $nameLit $n [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| tl.for $i:ident in $n:num { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
<<<<<<< Updated upstream
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned regionDTypes ptrElems stmts.toList
=======
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned stmts.toList
>>>>>>> Stashed changes
      pure (← `(Stmt.forLoop $nameLit $n [$algBody,*]),
        ← `(ComputeStmt.forLoop $nameLit $n [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| for $i:ident in range(0, $($stop:term), $($step:term)) { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
<<<<<<< Updated upstream
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned regionDTypes ptrElems stmts.toList
=======
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned stmts.toList
>>>>>>> Stashed changes
      pure (← `(Stmt.forRange $nameLit 0 $stop $step [$algBody,*]),
        ← `(ComputeStmt.forRange $nameLit 0 $stop $step [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| for $i:ident in range($($start:term), $($stop:term), $($step:term)) { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
<<<<<<< Updated upstream
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned regionDTypes ptrElems stmts.toList
      pure (← `(Stmt.forRange $nameLit $start $stop $step [$algBody,*]),
        ← `(ComputeStmt.forRange $nameLit $start $stop $step [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| tl.static_range $i:ident in $($n:term) { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned regionDTypes ptrElems stmts.toList
=======
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned stmts.toList
      pure (← `(Stmt.forRange $nameLit $start $stop $step [$algBody,*]),
        ← `(ComputeStmt.forRange $nameLit $start $stop $step [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr, $step:tritonExpr) { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let start' ← expandNatExpectedExpr env start
      let stop' ← expandNatExpectedExpr env stop
      let step' ← expandNatExpectedExpr env step
      ensureShape SInfo.scalar start'.shape "range start"
      ensureShape SInfo.scalar stop'.shape "range stop"
      ensureShape SInfo.scalar step'.shape "range step"
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned stmts.toList
      pure (← `(Stmt.forRangeDyn $nameLit $start'.term $stop'.term $step'.term [$algBody,*]),
        ← `(ComputeStmt.forRangeDyn $nameLit $start'.term $stop'.term $step'.term [$computeBody,*]),
        env, bodyHasCompute)
  | `(tritonStmt| tl.static_range $i:ident in $($n:term) { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned stmts.toList
>>>>>>> Stashed changes
      pure (← `(Stmt.forLoop $nameLit $n [$algBody,*]),
        ← `(ComputeStmt.forLoop $nameLit $n [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| tl.static_range $i:ident in $n:num { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
<<<<<<< Updated upstream
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned regionDTypes ptrElems stmts.toList
=======
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned stmts.toList
>>>>>>> Stashed changes
      pure (← `(Stmt.forLoop $nameLit $n [$algBody,*]),
        ← `(ComputeStmt.forLoop $nameLit $n [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| tl.if $cond:tritonExpr { $thenStmts:tritonStmt* } else { $elseStmts:tritonStmt* }) => do
      let cond' ← expandBoolCondition env cond
      ensureDType .bool cond'.dtype "tl.if condition"
      ensureShape SInfo.scalar cond'.shape "tl.if condition"
<<<<<<< Updated upstream
      let (algThen, computeThen, _, thenHasCompute) ← expandStmts env pinned regionDTypes ptrElems thenStmts.toList
      let (algElse, computeElse, _, elseHasCompute) ← expandStmts env pinned regionDTypes ptrElems elseStmts.toList
=======
      let thenBaseEnv := env
      let (algThen, computeThen, thenEnv, thenHasCompute) ← expandStmts env pinned thenStmts.toList
      let (algElse, computeElse, elseEnv, elseHasCompute) ← expandStmts env pinned elseStmts.toList
      let nextEnv := mergeBranchEnv thenBaseEnv thenEnv elseEnv
>>>>>>> Stashed changes
      pure (← `(Stmt.ifThenElse $cond'.term [$algThen,*] [$algElse,*]),
        ← `(ComputeStmt.ifThenElse $cond'.term [$computeThen,*] [$computeElse,*]),
        nextEnv,
        cond'.computeTerm.isSome || thenHasCompute || elseHasCompute)
  | `(tritonStmt| tl.if $cond:tritonExpr { $stmts:tritonStmt* }) => do
      let cond' ← expandBoolCondition env cond
      ensureDType .bool cond'.dtype "tl.if condition"
      ensureShape SInfo.scalar cond'.shape "tl.if condition"
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts env pinned regionDTypes ptrElems stmts.toList
      pure (← `(Stmt.ifThen $cond'.term [$algBody,*]),
        ← `(ComputeStmt.ifThen $cond'.term [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| if $cond:tritonExpr { $thenStmts:tritonStmt* } else { $elseStmts:tritonStmt* }) => do
      let cond' ← expandBoolCondition env cond
      ensureDType .bool cond'.dtype "if condition"
      ensureShape SInfo.scalar cond'.shape "if condition"
<<<<<<< Updated upstream
      let (algThen, computeThen, _, thenHasCompute) ← expandStmts env pinned regionDTypes ptrElems thenStmts.toList
      let (algElse, computeElse, _, elseHasCompute) ← expandStmts env pinned regionDTypes ptrElems elseStmts.toList
=======
      let (algThen, computeThen, thenEnv, thenHasCompute) ← expandStmts env pinned thenStmts.toList
      let (algElse, computeElse, elseEnv, elseHasCompute) ← expandStmts env pinned elseStmts.toList
>>>>>>> Stashed changes
      pure (← `(Stmt.ifThenElse $cond'.term [$algThen,*] [$algElse,*]),
        ← `(ComputeStmt.ifThenElse $cond'.term [$computeThen,*] [$computeElse,*]),
        mergeBranchEnv env thenEnv elseEnv,
        cond'.computeTerm.isSome || thenHasCompute || elseHasCompute)
  | `(tritonStmt| if $cond:tritonExpr { $stmts:tritonStmt* }) => do
      let cond' ← expandBoolCondition env cond
      ensureDType .bool cond'.dtype "if condition"
      ensureShape SInfo.scalar cond'.shape "if condition"
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts env pinned regionDTypes ptrElems stmts.toList
      pure (← `(Stmt.ifThen $cond'.term [$algBody,*]),
        ← `(ComputeStmt.ifThen $cond'.term [$computeBody,*]), env, bodyHasCompute)
  | _ => Macro.throwUnsupported

partial def expandStmts (env : Env) (pinned : List String)
    (regionDTypes : Inference.RegionDTypes) (ptrElems : Inference.PtrElems)
    (stmts : List (TSyntax `tritonStmt)) :
    MacroM (Array (TSyntax `term) × Array (TSyntax `term) × Env × Bool) := do
  let mut algOut : Array (TSyntax `term) := #[]
  let mut computeOut : Array (TSyntax `term) := #[]
  let mut env' := env
  let mut hasCompute := Bool.false
  for st in stmts do
    if Inference.isRegionDirective st then
      continue
    let (algTerm, computeTerm, nextEnv, stmtHasCompute) ←
      expandStmt env' pinned regionDTypes ptrElems st
    algOut := algOut.push algTerm
    computeOut := computeOut.push computeTerm
    env' := nextEnv
    hasCompute := hasCompute || stmtHasCompute
  pure (algOut, computeOut, env', hasCompute)

end

/-! ## Block macro -/

macro_rules
  | `(triton { $stmts:tritonStmt* }) => do
      -- Pre-pass: scan the body for identifiers that appear in `.nat`-pinning
      -- positions (offsets of static-ptr-add chains). The `expandStmt` cases
      -- for `name = tl.load(p)` / `name := tl.load(p)` consult this list
      -- when no explicit `dtype=` kwarg is given, defaulting to `.nat`
      -- instead of `.real` so that `name` participates correctly in pointer
      -- arithmetic downstream.
      let pinned := Inference.collectNatPinned stmts.toList
      -- Pre-pass: collect `region <name> = <dtype>` directives so the
      -- macro can default `tl.load(R + offs)` / `tl.store(R + offs, _)`
      -- to the declared element dtype instead of `.real`.
      let regionDTypes := Inference.collectRegionDTypes stmts.toList
      -- Pre-pass: propagate region dtypes to chained pointer bindings
      -- (`p = R + offs; ... ; tl.load(p)` recovers `R`'s dtype on the load).
      let ptrElems := Inference.collectPtrElems regionDTypes stmts.toList
      let (_, computeStmtTerms, _, _) ←
        expandStmts [] pinned regionDTypes ptrElems stmts.toList
      -- Auto-scan body: collect every region appearing in `tl.load(...)` (inputs)
      -- and `tl.store(...)` (outputs). Order = body occurrence; no macro-time
      -- dedup (a mix of literals and Lean terms can't be statically deduped, and
      -- `Kernel.inputs/outputs` is metadata-only, so duplicates are harmless).
      let (allIns, allOuts) := Metadata.blockRegions stmts.toList
      let insArr  : Array (TSyntax `term) := allIns.toArray
      let outsArr : Array (TSyntax `term) := allOuts.toArray
      `(ComputeKernel.mk [$insArr,*] [$outsArr,*] [$computeStmtTerms,*])

end VeriTile.Triton.DSL
