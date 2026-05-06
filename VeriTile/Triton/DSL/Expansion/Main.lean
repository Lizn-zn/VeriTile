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
import VeriTile.Triton.DSL.Expansion.Common
import VeriTile.Triton.DSL.Expansion.Memory
import VeriTile.Triton.DSL.Expansion.Compute
import VeriTile.Triton.DSL.Expansion.Control
import VeriTile.Triton.DSL.Syntax
import VeriTile.Triton.DSL.Typing

open Lean

namespace VeriTile.Triton.DSL

/-! ## Expansion -/

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
      expandProgramId e
  | `(tritonExpr| tl.arange($e:tritonExpr)) =>
      expandArange e
  | `(tritonExpr| tl.arange($s:tritonExpr, $e:tritonExpr)) => do
      expandArangeRange s e
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
      expandBoolBin expandExpr env "tl.logical_and" (← `(Op.boolAnd)) a b
  | `(tritonExpr| tl.logical_or($a:tritonExpr, $b:tritonExpr)) => do
      expandBoolBin expandExpr env "tl.logical_or" (← `(Op.boolOr)) a b
  | `(tritonExpr| tl.logical_not($a:tritonExpr)) => do
      expandBoolNot expandExpr env "tl.logical_not" a
  | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) => do
      expandCdiv expandExpr env a b
  | `(tritonExpr| tl.max($a:tritonExpr, $b:tritonExpr)) => do
      let a' ← expandExpr env a
      let b' ← expandExpr env b
      ensureDType .real a'.dtype "tl.max"
      ensureDType .real b'.dtype "tl.max"
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "tl.max"
      pure ⟨← `(Op.max2 $bc $a'.term $b'.term), .real, outShape, none⟩
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
  | `(tritonExpr| tl.full([$dims:tritonExpr,*], $v:tritonExpr)) => do
      expandFull expandExpr env dims.getElems v
  | `(tritonExpr| tl.zeros([$dims:tritonExpr,*])) => do
      -- `tl.zeros([dims])` ≡ `tl.full([dims], 0)`.
      let zero ← `(tritonExpr| 0)
      expandFull expandExpr env dims.getElems zero
  | _ => Macro.throwUnsupported

end

mutual

partial def expandStmt (env : Env) (stx : TSyntax `tritonStmt) :
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
        | some ident => (ident.getId.toString, input'.dtype, targetShape) :: env
      pure (alg, compute, nextEnv, Bool.false)
    match ← expandStaticPtrExpr env p with
    | some sp =>
        mkTerms (← `(MemAccess.region $sp.region $sp.offset)) sp.shape
    | none =>
        let p' ← expandExpr env p
        ensureDType .ptr p'.dtype (opName ++ " pointer")
        mkTerms (← `(MemAccess.ptr $p'.term)) p'.shape
  match stx with
  | `(tritonStmt| $i:ident := tl.atomic_xchg($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      ensureNoAtomicKwargs "tl.atomic_xchg" kwargs
      expandAtomicRMWCore "tl.atomic_xchg" (← `(RMWOp.xchg)) (some i) p v none
  | `(tritonStmt| $i:ident := tl.atomic_cas($p:tritonExpr, $cmp:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      ensureNoAtomicKwargs "tl.atomic_cas" kwargs
      expandAtomicRMWCore "tl.atomic_cas" (← `(RMWOp.cas)) (some i) p cmp (some v)
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
  | `(tritonStmt| tl.if $cond:tritonExpr { $thenStmts:tritonStmt* } else { $elseStmts:tritonStmt* }) => do
      let cond' ← expandExpr env cond
      ensureDType .bool cond'.dtype "tl.if condition"
      ensureShape SInfo.scalar cond'.shape "tl.if condition"
      let (algThen, computeThen, _, thenHasCompute) ← expandStmts env thenStmts.toList
      let (algElse, computeElse, _, elseHasCompute) ← expandStmts env elseStmts.toList
      pure (← `(Stmt.ifThenElse $cond'.term [$algThen,*] [$algElse,*]),
        ← `(ComputeStmt.ifThenElse $cond'.term [$computeThen,*] [$computeElse,*]),
        env, cond'.computeTerm.isSome || thenHasCompute || elseHasCompute)
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
      let (_, computeStmtTerms, _, _) ← expandStmts [] stmts.toList
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
      `(ComputeKernel.mk [$insArr,*] [$outsArr,*] [$computeStmtTerms,*])

end VeriTile.Triton.DSL
