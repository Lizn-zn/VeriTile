/-
VeriTile.Triton.DSL

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
  * Memory accesses use a Triton-like pointer-plus-offset surface syntax:
    `tl.load($(xReg) + offs)` and `tl.store($(yReg) + offs, y)`. The pointer
    sits in its own syntax category `tritonPtr` (see RP1 / GH issue #1),
    so future pointer forms (masked load, 2D pointer) extend `tritonPtr`
    without touching `tl.load` / `tl.store` themselves. The pointer lowers
    to the internal `(region : RegionName, offset : Op)` pair feeding
    `Op.load` / `Stmt.store`. Scalar pointer sugar `tl.load($(xReg))` /
    `tl.store($(yReg), y)` lowers to offset `0`.
  * `$(<lean-term>)` antiquotes a Lean-level value; in numeric context it
    becomes `Op.constNat`, inside `tl.arange(...)` it is fed directly as
    the `Nat` length, and as the base pointer term in `tl.load($(REGION)
    + offset)` / `tl.store($(REGION) + offset, value)` or scalar-pointer
    `tl.load($(REGION))` / `tl.store($(REGION), value)` it is used as a
    `RegionName`.
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
`tl.sqrt(_)`, `tl.max(_)`, `tl.sum(_)`, `tl.load($(REGION) + offset)`,
binary `+ - * /`, parens, identifiers, numerals, antiquotation
(`$(t)` for `Nat`, `$ℝ(t)` for `ℝ`). `tl.load($(REGION))` is sugar for
offset `0`.

The two-argument `tl.arange(start, end)` lowers to `start + tl.arange(end - start)`
at macro time (no new AST constructor). The literal-0 special case
`tl.arange(0, e)` collapses to `tl.arange(e)`, producing an AST identical to the
single-argument form so existing proofs (e.g. via `scatter_readback`) remain
applicable verbatim.

Currently supported statements: assignment (`name := expr`),
`tl.store($(REGION) + offset, value)`. `tl.store($(REGION), value)` is
sugar for offset `0`.

`Kernel.inputs` / `Kernel.outputs` are auto-populated by scanning the body for
`tl.load(...)` (input regions) and `tl.store(...)` (output regions). Order
follows body occurrence; no macro-time dedup since regions are Lean terms
(possibly equal at runtime but not statically). `Kernel.inputs/outputs` is
metadata only (not consumed by `exec`), so duplicates are harmless.

Loops, conditionals, broadcasts, and several other ops are P2+ work.
-/

import VeriTile.Triton.Core

open Lean

namespace VeriTile.Triton.DSL

/-! ## Syntax declarations -/

declare_syntax_cat tritonExpr
declare_syntax_cat tritonStmt
declare_syntax_cat tritonPtr
declare_syntax_cat tritonKwarg
declare_syntax_cat tritonReduceKwarg

-- Pointer expressions (used by `tl.load` / `tl.store`).
-- Splitting the pointer surface syntax into its own category gives a single
-- extension point for future pointer forms (masked load, 2D pointer, etc.;
-- see Phase C). The internal AST is unchanged: `tritonPtr` lowers to a
-- `(RegionName, offset : Op)` pair, which feeds `Op.load` / `Stmt.store`.
syntax "$(" term ")" : tritonPtr
syntax "$(" term ")" " + " tritonExpr : tritonPtr

-- Expressions
syntax num : tritonExpr
syntax ident : tritonExpr
syntax "$(" term ")" : tritonExpr
syntax "$ℝ(" term ")" : tritonExpr
syntax "(" tritonExpr ")" : tritonExpr
syntax "tl.program_id(" tritonExpr ")" : tritonExpr
syntax "tl.arange(" tritonExpr ")" : tritonExpr
syntax "tl.arange(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.exp(" tritonExpr ")" : tritonExpr
syntax "tl.log(" tritonExpr ")" : tritonExpr
syntax "tl.sigmoid(" tritonExpr ")" : tritonExpr
syntax "tl.sqrt(" tritonExpr ")" : tritonExpr
syntax "tl.max(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.toReal(" tritonExpr ")" : tritonExpr
syntax "-inf" : tritonExpr

-- Block-level matrix multiply. Two-arg form `tl.dot(a, b)` produces `a @ b`;
-- three-arg form `tl.dot(a, b, acc)` is the fused-accumulator pattern (the
-- standard FA inner-loop shape) and desugars to `acc + tl.dot(a, b)`.
syntax "tl.dot(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.dot(" tritonExpr ", " tritonExpr ", " tritonExpr ")" : tritonExpr

-- kwarg: `name = expr`. Used for `mask=` / `other=` in tl.load / tl.store.
-- Per Issue #16: only `mask` and `other` are recognized; other names error.
syntax ident " = " tritonExpr : tritonKwarg

-- Reduction kwargs. Per Triton semantics:
--   - `axis = K` selects which axis to reduce. VeriTile currently only supports
--     `K = shape.length - 1` (the user's "last axis", i.e. innermost in
--     outermost-first storage). Other axes require an explicit transpose.
--   - `keep_dims = true | false` preserves vs strips the reduced rank dim;
--     defaults to `false` to match Triton.
syntax "axis" "=" num : tritonReduceKwarg
syntax "keep_dims" "=" "false" : tritonReduceKwarg
syntax "keep_dims" "=" "true" : tritonReduceKwarg
syntax ident "=" tritonExpr : tritonReduceKwarg

syntax "tl.sum(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.max(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr

-- `tl.load(ptr [, kwarg]*)` — kwargs are optional. With zero kwargs this is
-- the legacy unmasked form; with `mask=` / `other=` it lowers to the masked
-- AST (Slice 1 of mask extension). Other kwargs raise a parse error per
-- Issue #16 / `feedback_triton_user_first_class.md`.
syntax "tl.load(" tritonPtr ("," tritonKwarg)* ")" : tritonExpr

-- Comparison operators (priority 50 — below arithmetic 60/70 — non-associative).
-- Both ℝ × ℝ and Nat × Nat carriers are supported by typed `Op` constructors.
-- channel (scalarBool / tileBool). Triton-faithful: no chained comparisons
-- (a < b < c is a syntax error, not Python's "a < b and b < c").
syntax:50 tritonExpr:51 " < "  tritonExpr:51 : tritonExpr
syntax:50 tritonExpr:51 " <= " tritonExpr:51 : tritonExpr
syntax:50 tritonExpr:51 " == " tritonExpr:51 : tritonExpr
syntax:50 tritonExpr:51 " > "  tritonExpr:51 : tritonExpr
syntax:50 tritonExpr:51 " >= " tritonExpr:51 : tritonExpr
syntax:50 tritonExpr:51 " != " tritonExpr:51 : tritonExpr

syntax:60 tritonExpr:60 " + " tritonExpr:61 : tritonExpr
syntax:60 tritonExpr:60 " - " tritonExpr:61 : tritonExpr
syntax:70 tritonExpr:70 " * " tritonExpr:71 : tritonExpr
syntax:70 tritonExpr:70 " / " tritonExpr:71 : tritonExpr

-- Statements
syntax ident " := " tritonExpr : tritonStmt
-- `tl.store(ptr, value [, kwarg]*)` — kwargs are optional. Only `mask=` is
-- recognized (Triton's `tl.store` has no `other`). Per Issue #16: unknown
-- kwarg → parse error.
syntax "tl.store(" tritonPtr ", " tritonExpr ("," tritonKwarg)* ")" : tritonStmt
-- `tl.for i in $(n) { stmt* }` — bounded loop over `n` iterations,
-- binding the iteration index to register `i` (Nat-channel).
syntax "tl.for " ident " in " "$(" term ")" " { " tritonStmt* " }" : tritonStmt
-- Convenience form with a numeric literal in place of the antiquoted term.
syntax "tl.for " ident " in " num " { " tritonStmt* " }" : tritonStmt

-- Block (the user-facing entry point)
syntax (name := tritonBlock) "triton " "{" tritonStmt* "}" : term

/-! ## Expansion -/

private def identAsStr (i : TSyntax `ident) : MacroM (TSyntax `term) :=
  pure (Syntax.mkStrLit i.getId.toString)

private inductive DInfo where
  | real
  | nat
  | bool
  deriving BEq, Inhabited

private inductive SInfo where
  | dims : List (TSyntax `term) → SInfo
  deriving Inhabited

private abbrev Env := List (String × DInfo × SInfo)

private def termKey (t : TSyntax `term) : String :=
  toString t.raw

namespace SInfo

private def scalar : SInfo := .dims []

private def vec (n : TSyntax `term) : SInfo := .dims [n]

end SInfo

private def termListEq : List (TSyntax `term) → List (TSyntax `term) → Bool
  | [], [] => Bool.true
  | a :: as, b :: bs => (termKey a == termKey b) && termListEq as bs
  | _, _ => Bool.false

private def SInfo.eq : SInfo → SInfo → Bool
  | SInfo.dims a, SInfo.dims b => termListEq a b

private def DInfo.term : DInfo → MacroM (TSyntax `term)
  | .real => `(TileDType.real)
  | .nat => `(TileDType.nat)
  | .bool => `(TileDType.bool)

private def SInfo.term : SInfo → MacroM (TSyntax `term)
  | SInfo.dims ds => do
      let rec go : List (TSyntax `term) → MacroM (TSyntax `term)
        | [] => `(([] : TileShape))
        | d :: rest => do
            let tail ← go rest
            `($d :: $tail)
      go ds

private def DInfo.numericProof : DInfo → MacroM (TSyntax `term)
  | .real => `(NumericDType.real)
  | .nat => `(NumericDType.nat)
  | .bool => Macro.throwError "arithmetic on Bool values is not supported"

private def DInfo.comparableProof : DInfo → MacroM (TSyntax `term)
  | .real => `(ComparableDType.real)
  | .nat => `(ComparableDType.nat)
  | .bool => Macro.throwError "comparison on Bool values is not supported"

private def lookupEnv (env : Env) (name : String) : MacroM (DInfo × SInfo) := do
  match env.find? (fun entry => entry.1 == name) with
  | some (_, dtype, shape) => pure (dtype, shape)
  | none => Macro.throwError ("unknown Triton identifier `" ++ name ++ "`")

private def ensureDType (expected actual : DInfo) (ctx : String) : MacroM Unit := do
  unless expected == actual do
    Macro.throwError (ctx ++ ": dtype mismatch")

private def ensureShape (expected actual : SInfo) (ctx : String) : MacroM Unit := do
  unless expected.eq actual do
    Macro.throwError (ctx ++ ": shape mismatch")

/-- Recognize a dim term as the literal `1`, tolerating common syntactic
wrappings (`1`, `(1)`, `(1 : Nat)`, `((1 : Nat))`, …) so that unit-dim
broadcasts fire regardless of how the dim was emitted.

`tl.arange(1)` and `tl.max(..., keep_dims=true)` etc. emit slightly different
syntactic forms for the literal `1`; without this normalization the
`broadcastTerm` head check would miss `keep_dims=true` outputs and break
`(M, 1) + (M, N)` style broadcasts. -/
private partial def termIsOne (t : TSyntax `term) : Bool :=
  match t with
  | `($n:num) => n.getNat == 1
  | `(($e:term)) => termIsOne e
  | `(($e:term : $_:term)) => termIsOne e
  | _ => Bool.false

private def broadcastTerm (a b : SInfo) (ctx : String) :
    MacroM (TSyntax `term × SInfo) := do
  match a, b with
  | SInfo.dims [], SInfo.dims [] => pure (← `(Broadcast.nil), SInfo.dims [])
  | SInfo.dims [], SInfo.dims (_ :: _) => pure (← `(Broadcast.scalarL), b)
  | SInfo.dims (_ :: _), SInfo.dims [] => pure (← `(Broadcast.scalarR), a)
  | SInfo.dims (ad :: ads), SInfo.dims (bd :: bds) =>
      let (subBc, subShape) ← broadcastTerm (SInfo.dims ads) (SInfo.dims bds) ctx
      if termKey ad == termKey bd then
        match subShape with
        | SInfo.dims outRest => pure (← `(Broadcast.consSame $subBc), SInfo.dims (ad :: outRest))
      else if termIsOne ad then
        match subShape with
        | SInfo.dims outRest => pure (← `(Broadcast.consL $subBc), SInfo.dims (bd :: outRest))
      else if termIsOne bd then
        match subShape with
        | SInfo.dims outRest => pure (← `(Broadcast.consR $subBc), SInfo.dims (ad :: outRest))
      else
        Macro.throwError (ctx ++ ": incompatible shapes")

private def coerceShape (e : TSyntax `term) (src target : SInfo) (ctx : String) :
    MacroM (TSyntax `term) := do
  if src.eq target then
    pure e
  else
    match src with
    | SInfo.dims [] =>
        let st ← target.term
        `(Op.broadcast $e $st)
    | _ => Macro.throwError (ctx ++ ": cannot broadcast non-scalar to target shape")

private structure EOut where
  term : TSyntax `term
  dtype : DInfo
  shape : SInfo
  deriving Inhabited

mutual

/-- Lower a `tritonPtr` to its `(region, offset, offset-shape)` triple. -/
partial def expandPtr (env : Env) (stx : TSyntax `tritonPtr) :
    MacroM (TSyntax `term × TSyntax `term × SInfo) := do
  match stx with
  | `(tritonPtr| $($r:term)) =>
      -- Scalar pointer sugar: `$(R)` reads `R + 0`.
      let zero : TSyntax `term ← `(Op.constNat 0)
      pure (r, zero, SInfo.scalar)
  | `(tritonPtr| $($r:term) + $o:tritonExpr) => do
      let o' ← expandExpr env o
      ensureDType .nat o'.dtype "pointer offset"
      pure (r, o'.term, o'.shape)
  | _ => Macro.throwUnsupported

partial def expandExpr (env : Env) (stx : TSyntax `tritonExpr) : MacroM EOut := do
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
  | `(tritonExpr| tl.program_id($_)) =>
      pure ⟨← `(Op.programId), .nat, SInfo.scalar⟩
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
  | `(tritonExpr| tl.max($a:tritonExpr, $b:tritonExpr)) => do
      let a' ← expandExpr env a
      let b' ← expandExpr env b
      ensureDType .real a'.dtype "tl.max"
      ensureDType .real b'.dtype "tl.max"
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "tl.max"
      pure ⟨← `(Op.max2 $bc $a'.term $b'.term), .real, outShape⟩
  | `(tritonExpr| tl.sum($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandReduce env "tl.sum" (← `(Op.reduceSum)) e kwargs
  | `(tritonExpr| tl.max($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandReduce env "tl.max" (← `(Op.reduceMax)) e kwargs
  | `(tritonExpr| tl.toReal($e:tritonExpr)) => do
      let e' ← expandExpr env e
      ensureDType .nat e'.dtype "tl.toReal"
      pure ⟨← `(Op.natToReal $e'.term), .real, e'.shape⟩
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
  | `(tritonExpr| tl.load($p:tritonPtr $[, $kwargs:tritonKwarg]*)) => do
      -- Pointer surface syntax lowers to the internal `(region, offset)` AST.
      -- Optional `mask=` / `other=` kwargs lower to `LoadOptions`.
      -- Per Issue #16: any other kwarg name is a parse error. Per Triton spec
      -- ("If `other` is None, the masked-out value is undefined"): missing
      -- `other` lowers to `mask := some ..., other := none`; evalOp then
      -- uses `BlockState.undef` for masked-off lanes. `other` without `mask`
      -- is rejected (no Triton equivalent).
      let (r, off, offShape) ← expandPtr env p
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
          pure ⟨← `(Op.load $r $off), .real, offShape⟩
      | some (m, mShape), none =>
          -- No `other=`: masked-off lanes are undef in Triton. Keep that
          -- distinction in the AST; do not silently choose 0.
          let m' ← coerceShape m mShape offShape "tl.load mask"
          pure ⟨← `(Op.loadMask $r $off $m'), .real, offShape⟩
      | some (m, mShape), some (o, oShape) =>
          let m' ← coerceShape m mShape offShape "tl.load mask"
          let o' ← coerceShape o oShape offShape "tl.load other"
          pure ⟨← `(Op.loadMaskOther $r $off $m' $o'), .real, offShape⟩
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
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
      expandArith env "arithmetic" (← `(Op.add)) a b
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) => do
      expandArith env "arithmetic" (← `(Op.sub)) a b
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => do
      expandArith env "arithmetic" (← `(Op.mul)) a b
  | `(tritonExpr| $a:tritonExpr / $b:tritonExpr) => do
      expandArith env "arithmetic" (← `(Op.div)) a b
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

partial def expandCmp (env : Env) (ctx : String) (op : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let cp ← a'.dtype.comparableProof
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  pure ⟨← `($op $cp $bc $a'.term $b'.term), .bool, outShape⟩

/-- Lower a `tl.sum(...)` / `tl.max(...)` expression with optional reduction
kwargs (`axis = K`, `keep_dims = true|false`) into the corresponding
`Op.reduceSum / .reduceMax` AST node.

Validation: `axis` must be the user's last axis (`shape.length - 1`); we
currently only support innermost-axis reduction. `keep_dims` defaults to
`false` (Triton default). The output `SInfo` reflects the rank change:
`keep_dims = false` strips the trailing dim, `keep_dims = true` collapses
it to `1`. -/
partial def expandReduce (env : Env) (ctx : String) (op : TSyntax `term)
    (e : TSyntax `tritonExpr)
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM EOut := do
  let e' ← expandExpr env e
  ensureDType .real e'.dtype ctx
  let dims := match e'.shape with | .dims ds => ds
  if dims.isEmpty then
    Macro.throwError (ctx ++ ": reduction expects a tile, got scalar")
  let userLastAxis : Nat := dims.length - 1
  let mut seenAxis : Bool := Bool.false
  let mut seenKeepDims : Bool := Bool.false
  let mut keepDims : Bool := Bool.false
  for kw in kwargs do
    match kw with
    | `(tritonReduceKwarg| axis = $n:num) =>
        if seenAxis then
          Macro.throwError (ctx ++ ": duplicate `axis=` kwarg")
        seenAxis := Bool.true
        if n.getNat != userLastAxis then
          Macro.throwError
            (ctx ++ ": only `axis=" ++ toString userLastAxis ++ "` " ++
             "(the user-side last / innermost axis) is supported; got `axis=" ++
             toString n.getNat ++ "`. Reductions over interior or outermost " ++
             "axes require an explicit transpose (issue #26).")
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
  -- Compute output shape:
  -- keep_dims = false: dims = init ++ [last]  →  init
  -- keep_dims = true:  dims = init ++ [last]  →  init ++ [1]
  let initDims := dims.dropLast
  let oneLit : TSyntax `term ← `((1 : Nat))
  let outShape : SInfo :=
    if keepDims then SInfo.dims (initDims ++ [oneLit]) else SInfo.dims initDims
  let kdLit : TSyntax `term ← if keepDims then `(Bool.true) else `(Bool.false)
  pure ⟨← `($op $kdLit $e'.term), .real, outShape⟩

/-- Lower a `tl.dot(a, b)` to `Op.dot a b`. Both operands must be rank-2
real tiles whose inner dim agrees syntactically (same dim term). The
result shape is `[outerDim a, innerDim b]`. -/
partial def expandDot (env : Env)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureDType .real a'.dtype "tl.dot"
  ensureDType .real b'.dtype "tl.dot"
  match a'.shape, b'.shape with
  | .dims [aM, aK], .dims [bK, bN] =>
      unless termKey aK == termKey bK do
        Macro.throwError
          ("tl.dot: inner dim mismatch — LHS has shape `[..., "
            ++ termKey aK ++ "]` but RHS has shape `[" ++ termKey bK
            ++ ", ...]`")
      pure ⟨← `(Op.dot $a'.term $b'.term), .real, .dims [aM, bN]⟩
  | _, _ =>
      Macro.throwError
        "tl.dot: both operands must be rank-2 (2D matrices)"

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
  | `(tritonStmt| tl.store($p:tritonPtr, $v:tritonExpr $[, $kwargs:tritonKwarg]*)) => do
      -- Optional kwargs: only `mask=` is recognized for store (Triton's
      -- `tl.store` has no `other`). Per Issue #16: any other kwarg name
      -- (including `other=`) is a parse error.
      let v' ← expandExpr env v
      ensureDType .real v'.dtype "tl.store value"
      let (r, off, offShape) ← expandPtr env p
      let vTerm ← coerceShape v'.term v'.shape offShape "tl.store value"
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
          let sh ← offShape.term
          pure (← `(Stmt.store $r $sh $off $vTerm), env)
      | some (m, mShape) =>
          let m' ← coerceShape m mShape offShape "tl.store mask"
          let sh ← offShape.term
          pure (← `(Stmt.storeMask $r $sh $off $vTerm $m'), env)
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

/-! ## Region collection (for auto-populating Kernel.inputs / Kernel.outputs) -/

mutual

/-- Collect all region terms reachable from a `tritonExpr`. Returns `term`
    syntax — each element is the Lean term inside a `tl.load(...)`
    pointer (recursively in subexpressions). -/
private partial def exprRegions : TSyntax `tritonExpr → List (TSyntax `term) := fun stx =>
  match stx with
  | `(tritonExpr| tl.load($p:tritonPtr $[, $kwargs:tritonKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonKwarg) =>
            match kw with
            | `(tritonKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions val
            | _ => acc) []
      ptrRegions p ++ kwargRegions
  | `(tritonExpr| tl.exp($e:tritonExpr))         => exprRegions e
  | `(tritonExpr| tl.log($e:tritonExpr))         => exprRegions e
  | `(tritonExpr| tl.sigmoid($e:tritonExpr))     => exprRegions e
  | `(tritonExpr| tl.sqrt($e:tritonExpr))        => exprRegions e
  | `(tritonExpr| tl.max($a:tritonExpr, $b:tritonExpr))   =>
      exprRegions a ++ exprRegions b
  | `(tritonExpr| tl.sum($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions val
            | _ => acc) []
      exprRegions e ++ kwargRegions
  | `(tritonExpr| tl.max($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions val
            | _ => acc) []
      exprRegions e ++ kwargRegions
  | `(tritonExpr| tl.toReal($e:tritonExpr))      => exprRegions e
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions a ++ exprRegions b
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr, $c:tritonExpr)) =>
      exprRegions a ++ exprRegions b ++ exprRegions c
  | `(tritonExpr| ($e:tritonExpr))               => exprRegions e
  | `(tritonExpr| tl.program_id($e:tritonExpr))  => exprRegions e
  | `(tritonExpr| tl.arange($e:tritonExpr))      => exprRegions e
  | `(tritonExpr| tl.arange($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr <  $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr <= $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr == $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr >  $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr >= $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr != $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr +  $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr -  $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr *  $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr /  $b:tritonExpr) => exprRegions a ++ exprRegions b
  | _ => []  -- num, ident, $(...) — no regions to collect

/-- All region terms appearing in a `tritonPtr`: the base region plus any
    regions referenced inside the offset expression. -/
private partial def ptrRegions : TSyntax `tritonPtr → List (TSyntax `term) := fun stx =>
  match stx with
  | `(tritonPtr| $($r:term))                  => [r]
  | `(tritonPtr| $($r:term) + $o:tritonExpr)  => r :: exprRegions o
  | _ => []

end

/-- Just the base region of a `tritonPtr` (used as the *output* region for
    `tl.store(...)`, distinct from regions that appear *inside* the offset). -/
private def ptrBaseRegion : TSyntax `tritonPtr → List (TSyntax `term) := fun stx =>
  match stx with
  | `(tritonPtr| $($r:term))                  => [r]
  | `(tritonPtr| $($r:term) + $_:tritonExpr)  => [r]
  | _ => []

/-- Just the regions referenced inside the offset expression of a `tritonPtr`
    (excluding the base). For `tl.store(p, v)` these still count as inputs. -/
private def ptrOffsetRegions : TSyntax `tritonPtr → List (TSyntax `term) := fun stx =>
  match stx with
  | `(tritonPtr| $($_:term))                  => []
  | `(tritonPtr| $($_:term) + $o:tritonExpr)  => exprRegions o
  | _ => []

/-- Per-statement region split: `(input regions, output regions)`. -/
private partial def stmtRegions :
    TSyntax `tritonStmt → List (TSyntax `term) × List (TSyntax `term) := fun stx =>
  match stx with
  | `(tritonStmt| $_:ident := $e:tritonExpr) =>
      (exprRegions e, [])
  | `(tritonStmt| tl.store($p:tritonPtr, $v:tritonExpr $[, $kwargs:tritonKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonKwarg) =>
            match kw with
            | `(tritonKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions val
            | _ => acc) []
      (ptrOffsetRegions p ++ exprRegions v ++ kwargRegions, ptrBaseRegion p)
  | `(tritonStmt| tl.for $_:ident in $($_:term) { $stmts:tritonStmt* }) =>
      stmts.toList.foldl
        (fun (acc : List (TSyntax `term) × List (TSyntax `term)) st =>
          let (i, o) := stmtRegions st
          (acc.1 ++ i, acc.2 ++ o)) ([], [])
  | `(tritonStmt| tl.for $_:ident in $_:num { $stmts:tritonStmt* }) =>
      stmts.toList.foldl
        (fun (acc : List (TSyntax `term) × List (TSyntax `term)) st =>
          let (i, o) := stmtRegions st
          (acc.1 ++ i, acc.2 ++ o)) ([], [])
  | _ => ([], [])

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
          let (i, o) := stmtRegions s
          (acc.1 ++ i, acc.2 ++ o))
        ([], [])
      let insArr  : Array (TSyntax `term) := allIns.toArray
      let outsArr : Array (TSyntax `term) := allOuts.toArray
      `(Kernel.mk [$insArr,*] [$outsArr,*] [$stmtTerms,*])

end VeriTile.Triton.DSL
