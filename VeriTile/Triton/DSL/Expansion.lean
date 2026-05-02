/-
VeriTile.Triton.DSL.Expansion

A `triton { ... }` macro that embeds Triton-style kernel syntax inside Lean,
modelled on the `arm64 { ... }` style used in arm-in-lean.

Example:

  def naiveSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : Kernel := triton {
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
`tl.arange(start, end)`, `tl.exp(_)`, `tl.log(_)`, `tl.sigmoid(_)`,
`tl.sqrt(_)`, `tl.tanh(_)`, `tl.max(_)`, `tl.sum(_)`, `tl.load(ptrExpr)`,
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
  deriving Inhabited

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
      pure ⟨← `(Op.castFloat $srcProof $dstProof $e'.term), dst, e'.shape⟩
  | none =>
  match stx with
  | `(tritonExpr| $n:num) =>
      -- Bare numeric literals are `ℝ` data constants (e.g. `1` in `1 / s`).
      pure ⟨← `(Op.const $n), .real, SInfo.scalar⟩
  | `(tritonExpr| $i:ident) =>
      let name := i.getId.toString
      let (dtype, shape) ← lookupEnv env name
      let s ← identAsStr i
      let dt ← dtype.term
      let sh ← shape.term
      pure ⟨← `(Op.ref $dt $sh $s), dtype, shape⟩
  | `(tritonExpr| $($t:term)) =>
      -- `$(...)` antiquote is the address/size channel: `Nat`.
      pure ⟨← `(Op.constNat $t), .nat, SInfo.scalar⟩
  | `(tritonExpr| $ℝ($t:term)) =>
      -- `$ℝ(...)` antiquote is the data channel: `ℝ`. Symmetric with the
      -- `$(t) → Op.constNat` form, used for non-literal ℝ kernel params
      -- (e.g. LayerNorm's `ε`).
      pure ⟨← `(Op.const $t), .real, SInfo.scalar⟩
  | `(tritonExpr| ($e:tritonExpr)) =>
      expandExpr env e
  | `(tritonExpr| tl.program_id($e:tritonExpr)) =>
      -- axis must be a numeric literal or `$(t)` antiquote (Nat)
      let axisTerm : TSyntax `term ← match e with
        | `(tritonExpr| $($t:term)) => `(($t : Nat))
        | `(tritonExpr| $n:num)     => `(($n : Nat))
        | _ => Macro.throwError
                "tl.program_id(axis): axis must be a numeric literal or $(N)"
      pure ⟨← `(Op.programId $axisTerm), .nat, SInfo.scalar⟩
  | `(tritonExpr| tl.arange($e:tritonExpr)) =>
      -- arange takes a Nat; recognize $(t) and bare numerals specially
      match e with
      | `(tritonExpr| $($t:term)) => pure ⟨← `(Op.arange $t), .nat, SInfo.vec t⟩
      | `(tritonExpr| $n:num)     => pure ⟨← `(Op.arange $n), .nat, SInfo.vec (← `(($n : Nat)))⟩
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
            pure ⟨← `(Op.arange $eTerm), .nat, SInfo.vec eTerm⟩
          else
            pure ⟨← `(Op.add NumericDType.nat Broadcast.scalarL (Op.constNat $sTerm) (Op.arange ($eTerm - $sTerm))),
              .nat, SInfo.vec (← `($eTerm - $sTerm))⟩
      | _ =>
          pure ⟨← `(Op.add NumericDType.nat Broadcast.scalarL (Op.constNat $sTerm) (Op.arange ($eTerm - $sTerm))),
            .nat, SInfo.vec (← `($eTerm - $sTerm))⟩
  | `(tritonExpr| tl.exp($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.exp"
      pure ⟨← `(Op.exp $e'.term), .real, e'.shape⟩
  | `(tritonExpr| tl.log($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.log"
      pure ⟨← `(Op.log $e'.term), .real, e'.shape⟩
  | `(tritonExpr| tl.sigmoid($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.sigmoid"
      pure ⟨← `(Op.sigmoid $e'.term), .real, e'.shape⟩
  | `(tritonExpr| tl.sqrt($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.sqrt"
      pure ⟨← `(Op.sqrt $e'.term), .real, e'.shape⟩
  | `(tritonExpr| tl.tanh($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .real e'.dtype "tl.tanh"
      pure ⟨← `(Op.tanh $e'.term), .real, e'.shape⟩
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
      pure ⟨← `(Op.where $cond $neg $e'.term), .real, e'.shape⟩
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
      pure ⟨← `(Op.max2 $bc $a'.term $b'.term), .real, outShape⟩
  | `(tritonExpr| tl.maximum($a:tritonExpr, $b:tritonExpr)) => do
      expandMinMax env "tl.maximum" (← `(Op.gt)) a b
  | `(tritonExpr| tl.minimum($a:tritonExpr, $b:tritonExpr)) => do
      expandMinMax env "tl.minimum" (← `(Op.lt)) a b
  | `(tritonExpr| tl.sum($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandReduce env "tl.sum" (← `(Op.reduceSum)) e kwargs
  | `(tritonExpr| tl.max($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandReduce env "tl.max" (← `(Op.reduceMax)) e kwargs
  | `(tritonExpr| tl.toReal($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .nat e'.dtype "tl.toReal"
      pure ⟨← `(Op.natToReal $e'.term), .real, e'.shape⟩
  | `(tritonExpr| tl.cast($e:tritonExpr, $dt:tritonDType)) => do
      let e' ← expandExpr env e
      let dst ← expandDType dt
      let srcProof ← e'.dtype.floatProof
      let dstProof ← dst.floatProof
      pure ⟨← `(Op.castFloat $srcProof $dstProof $e'.term), dst, e'.shape⟩
  | `(tritonExpr| -inf) =>
      pure ⟨← `(Op.negInf), .real, SInfo.scalar⟩
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
            .real, outShape⟩
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonKwarg]*)) => do
      let mut maskTerm : Option (TSyntax `term × SInfo) := none
      let mut otherTerm : Option (TSyntax `term × SInfo) := none
      for kw in kwargs do
        match kw with
        | `(tritonKwarg| $name:ident = $val:tritonExpr) =>
            let val' ← expandExpr env val
            match name.getId.toString with
            | "mask"  =>
                ensureDType .bool val'.dtype "tl.load mask"
                maskTerm := some (val'.term, val'.shape)
            | "other" =>
                ensureDType .real val'.dtype "tl.load other"
                otherTerm := some (val'.term, val'.shape)
            | unknown =>
                let msg : String :=
                  "tl.load: unknown kwarg `" ++ unknown ++
                  "`. Only `mask` and `other` are recognized (see GitHub issue #16)."
                Macro.throwError msg
        | _ => Macro.throwUnsupported
      match maskTerm, otherTerm with
      | none, none =>
          match ← expandStaticPtrExpr env p with
          | some sp =>
              let r := sp.region
              let off := sp.offset
              pure ⟨← `(Op.load $r $off), .real, sp.shape⟩
          | none =>
              let p' ← expandExpr env p
              ensureDType .ptr p'.dtype "tl.load pointer"
              pure ⟨← `(Op.loadPtr $p'.term), .real, p'.shape⟩
      | some (m, mShape), none =>
          match ← expandStaticPtrExpr env p with
          | some sp =>
              let r := sp.region
              let off := sp.offset
              let m' ← coerceShape m mShape sp.shape "tl.load mask"
              pure ⟨← `(Op.loadMask $r $off $m'), .real, sp.shape⟩
          | none =>
              let p' ← expandExpr env p
              ensureDType .ptr p'.dtype "tl.load pointer"
              let m' ← coerceShape m mShape p'.shape "tl.load mask"
              pure ⟨← `(Op.loadPtrMask $p'.term $m'), .real, p'.shape⟩
      | some (m, mShape), some (o, oShape) =>
          match ← expandStaticPtrExpr env p with
          | some sp =>
              let r := sp.region
              let off := sp.offset
              let m' ← coerceShape m mShape sp.shape "tl.load mask"
              let o' ← coerceShape o oShape sp.shape "tl.load other"
              pure ⟨← `(Op.loadMaskOther $r $off $m' $o'), .real, sp.shape⟩
          | none =>
              let p' ← expandExpr env p
              ensureDType .ptr p'.dtype "tl.load pointer"
              let m' ← coerceShape m mShape p'.shape "tl.load mask"
              let o' ← coerceShape o oShape p'.shape "tl.load other"
              pure ⟨← `(Op.loadPtrMaskOther $p'.term $m' $o'), .real, p'.shape⟩
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
      expandBoolBin env "boolean &" (← `(Op.boolAnd)) a b
  | `(tritonExpr| $a:tritonExpr | $b:tritonExpr) => do
      expandBoolBin env "boolean |" (← `(Op.boolOr)) a b
  | `(tritonExpr| ~$a:tritonExpr) => do
      expandBoolNot env "boolean ~" a
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
      match ← expandStaticPtrExpr env stx with
      | some sp =>
          let (bc, _) ← broadcastTerm SInfo.scalar sp.shape "pointer arithmetic"
          pure ⟨← `(Op.ptrAdd $bc (Op.ptrBase $sp.region) $sp.offset), .ptr, sp.shape⟩
      | none =>
          let a' ← expandExpr env a
          let b' ← expandExpr env b
          match a'.dtype, b'.dtype with
          | .ptr, .nat =>
              let (bc, outShape) ← broadcastTerm a'.shape b'.shape "pointer arithmetic"
              pure ⟨← `(Op.ptrAdd $bc $a'.term $b'.term), .ptr, outShape⟩
          | .nat, .ptr =>
              let (bc, outShape) ← broadcastTerm b'.shape a'.shape "pointer arithmetic"
              pure ⟨← `(Op.ptrAdd $bc $b'.term $a'.term), .ptr, outShape⟩
          | _, _ =>
              unless a'.dtype == b'.dtype do
                Macro.throwError "arithmetic: dtype mismatch"
              let np ← a'.dtype.numericProof
              let (bc, outShape) ← broadcastTerm a'.shape b'.shape "arithmetic"
              pure ⟨← `(Op.add $np $bc $a'.term $b'.term), a'.dtype, outShape⟩
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
      pure ⟨← `(Op.where $cTerm $aTerm $bTerm), a'.dtype, target⟩
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
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let np ← a'.dtype.numericProof
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  pure ⟨← `($op $np $bc $a'.term $b'.term), a'.dtype, outShape⟩

partial def expandIntegralArith (env : Env) (ctx : String) (op : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let ip ← a'.dtype.integralProof
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  pure ⟨← `($op $ip $bc $a'.term $b'.term), a'.dtype, outShape⟩

partial def expandCdiv (env : Env)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureDType .nat a'.dtype "tl.cdiv lhs"
  ensureDType .nat b'.dtype "tl.cdiv rhs"
  let (addBc, outShape) ← broadcastTerm a'.shape b'.shape "tl.cdiv"
  let (subBc, subShape) ← broadcastTerm outShape SInfo.scalar "tl.cdiv"
  ensureShape outShape subShape "tl.cdiv"
  let (divBc, divShape) ← broadcastTerm outShape b'.shape "tl.cdiv"
  ensureShape outShape divShape "tl.cdiv"
  let sum ← `(Op.add NumericDType.nat $addBc $a'.term $b'.term)
  let numerator ← `(Op.sub NumericDType.nat $subBc $sum (Op.constNat 1))
  pure ⟨← `(Op.div NumericDType.nat $divBc $numerator $b'.term), .nat, outShape⟩

partial def expandBoolBin (env : Env) (ctx : String) (op : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureDType .bool a'.dtype ctx
  ensureDType .bool b'.dtype ctx
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  pure ⟨← `($op $bc $a'.term $b'.term), .bool, outShape⟩

partial def expandBoolNot (env : Env) (ctx : String)
    (a : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  ensureDType .bool a'.dtype ctx
  pure ⟨← `(Op.boolNot $a'.term), .bool, a'.shape⟩

partial def expandCmp (env : Env) (ctx : String) (op : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let cp ← a'.dtype.comparableProof
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  pure ⟨← `($op $cp $bc $a'.term $b'.term), .bool, outShape⟩

partial def expandMinMax (env : Env) (ctx : String) (cmp : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let cp ← a'.dtype.comparableProof
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  let aTerm ← coerceShape a'.term a'.shape outShape (ctx ++ " lhs")
  let bTerm ← coerceShape b'.term b'.shape outShape (ctx ++ " rhs")
  pure ⟨← `(Op.where ($cmp $cp $bc $a'.term $b'.term) $aTerm $bTerm), a'.dtype, outShape⟩

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
            .real, SInfo.dims outDims⟩
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
      pure ⟨term, .real, SInfo.dims outDims⟩

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
        .real, .dims (aBatch ++ [aM, bN])⟩

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
  pure ⟨← `(Op.full $shape $v'.term), v'.dtype, .dims dimTerms.toList⟩

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
        e'.dtype, .dims (batch ++ [N, M])⟩

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
        e'.dtype, .dims outDims⟩

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
    MacroM (TSyntax `term × Env) := do
  match stx with
  | `(tritonStmt| $i:ident := $e:tritonExpr) => do
      let nameLit ← identAsStr i
      let e' ← expandExpr env e
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        (i.getId.toString, e'.dtype, e'.shape) :: env)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonKwarg]*)) => do
      let mut maskTerm : Option (TSyntax `term × SInfo) := none
      for kw in kwargs do
        match kw with
        | `(tritonKwarg| $name:ident = $kval:tritonExpr) =>
            let kval' ← expandExpr env kval
            match name.getId.toString with
            | "mask"  =>
                ensureDType .bool kval'.dtype "tl.store mask"
                maskTerm := some (kval'.term, kval'.shape)
            | unknown =>
                let msg : String :=
                  "tl.store: unknown kwarg `" ++ unknown ++
                  "`. Only `mask` is recognized (Triton's tl.store has no `other`; see issue #16)."
                Macro.throwError msg
        | _ => Macro.throwUnsupported
      match maskTerm with
      | none =>
          match ← expandStaticPtrExpr env p with
          | some sp =>
              let v' ← expandExpr env v
              ensureDType .real v'.dtype "tl.store value"
              let vTerm ← coerceShape v'.term v'.shape sp.shape "tl.store value"
              let r := sp.region
              let off := sp.offset
              let sh ← sp.shape.term
              pure (← `(Stmt.store $r $sh $off $vTerm), env)
          | none =>
              let p' ← expandExpr env p
              ensureDType .ptr p'.dtype "tl.store pointer"
              let v' ← expandExpr env v
              ensureDType .real v'.dtype "tl.store value"
              let vTerm ← coerceShape v'.term v'.shape p'.shape "tl.store value"
              let sh ← p'.shape.term
              pure (← `(Stmt.storePtr $sh $p'.term $vTerm), env)
      | some (m, mShape) =>
          match ← expandStaticPtrExpr env p with
          | some sp =>
              let v' ← expandExpr env v
              ensureDType .real v'.dtype "tl.store value"
              let vTerm ← coerceShape v'.term v'.shape sp.shape "tl.store value"
              let m' ← coerceShape m mShape sp.shape "tl.store mask"
              let r := sp.region
              let off := sp.offset
              let sh ← sp.shape.term
              pure (← `(Stmt.storeMask $r $sh $off $vTerm $m'), env)
          | none =>
              let p' ← expandExpr env p
              ensureDType .ptr p'.dtype "tl.store pointer"
              let v' ← expandExpr env v
              ensureDType .real v'.dtype "tl.store value"
              let vTerm ← coerceShape v'.term v'.shape p'.shape "tl.store value"
              let m' ← coerceShape m mShape p'.shape "tl.store mask"
              let sh ← p'.shape.term
              pure (← `(Stmt.storePtrMask $sh $p'.term $vTerm $m'), env)
  | `(tritonStmt| tl.for $i:ident in $($n:term) { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar) :: env
      let (body, _) ← expandStmts bodyEnv stmts.toList
      pure (← `(Stmt.forLoop $nameLit $n [$body,*]), env)
  | `(tritonStmt| tl.for $i:ident in $n:num { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar) :: env
      let (body, _) ← expandStmts bodyEnv stmts.toList
      pure (← `(Stmt.forLoop $nameLit $n [$body,*]), env)
  | `(tritonStmt| tl.static_range $i:ident in $($n:term) { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar) :: env
      let (body, _) ← expandStmts bodyEnv stmts.toList
      pure (← `(Stmt.forLoop $nameLit $n [$body,*]), env)
  | `(tritonStmt| tl.static_range $i:ident in $n:num { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar) :: env
      let (body, _) ← expandStmts bodyEnv stmts.toList
      pure (← `(Stmt.forLoop $nameLit $n [$body,*]), env)
  | `(tritonStmt| tl.if $cond:tritonExpr { $stmts:tritonStmt* }) => do
      let cond' ← expandExpr env cond
      ensureDType .bool cond'.dtype "tl.if condition"
      ensureShape SInfo.scalar cond'.shape "tl.if condition"
      let (body, _) ← expandStmts env stmts.toList
      pure (← `(Stmt.ifThen $cond'.term [$body,*]), env)
  | _ => Macro.throwUnsupported

partial def expandStmts (env : Env) (stmts : List (TSyntax `tritonStmt)) :
    MacroM (Array (TSyntax `term) × Env) := do
  let mut out : Array (TSyntax `term) := #[]
  let mut env' := env
  for st in stmts do
    let (term, nextEnv) ← expandStmt env' st
    out := out.push term
    env' := nextEnv
  pure (out, env')

end

/-! ## Block macro -/

macro_rules
  | `(triton { $stmts:tritonStmt* }) => do
      let (stmtTerms, _) ← expandStmts [] stmts.toList
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
      `(Kernel.mk [$insArr,*] [$outsArr,*] [$stmtTerms,*])

end VeriTile.Triton.DSL
