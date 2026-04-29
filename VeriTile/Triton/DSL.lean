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
syntax "tl.max(" tritonExpr ")" : tritonExpr
syntax "tl.max(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.sum(" tritonExpr ")" : tritonExpr
syntax "tl.toReal(" tritonExpr ")" : tritonExpr
syntax "-inf" : tritonExpr
syntax "tl.load(" tritonPtr ")" : tritonExpr
syntax:60 tritonExpr:60 " + " tritonExpr:61 : tritonExpr
syntax:60 tritonExpr:60 " - " tritonExpr:61 : tritonExpr
syntax:70 tritonExpr:70 " * " tritonExpr:71 : tritonExpr
syntax:70 tritonExpr:70 " / " tritonExpr:71 : tritonExpr

-- Statements
syntax ident " := " tritonExpr : tritonStmt
syntax "tl.store(" tritonPtr ", " tritonExpr ")" : tritonStmt
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

mutual

/-- Lower a `tritonPtr` to its `(region, offset)` pair. -/
partial def expandPtr (stx : TSyntax `tritonPtr) :
    MacroM (TSyntax `term × TSyntax `term) := do
  match stx with
  | `(tritonPtr| $($r:term)) =>
      -- Scalar pointer sugar: `$(R)` reads `R + 0`.
      let zero : TSyntax `term ← `(Op.constNat 0)
      pure (r, zero)
  | `(tritonPtr| $($r:term) + $o:tritonExpr) => do
      let o' ← expandExpr o
      pure (r, o')
  | _ => Macro.throwUnsupported

partial def expandExpr (stx : TSyntax `tritonExpr) : MacroM (TSyntax `term) := do
  match stx with
  | `(tritonExpr| $n:num) =>
      -- Bare numeric literals are `ℝ` data constants (e.g. `1` in `1 / s`).
      `(Op.const $n)
  | `(tritonExpr| $i:ident) =>
      let s ← identAsStr i
      `(Op.ref $s)
  | `(tritonExpr| $($t:term)) =>
      -- `$(...)` antiquote is the address/size channel: `Nat`.
      `(Op.constNat $t)
  | `(tritonExpr| $ℝ($t:term)) =>
      -- `$ℝ(...)` antiquote is the data channel: `ℝ`. Symmetric with the
      -- `$(t) → Op.constNat` form, used for non-literal ℝ kernel params
      -- (e.g. LayerNorm's `ε`).
      `(Op.const $t)
  | `(tritonExpr| ($e:tritonExpr)) =>
      expandExpr e
  | `(tritonExpr| tl.program_id($_)) =>
      `(Op.programId)
  | `(tritonExpr| tl.arange($e:tritonExpr)) =>
      -- arange takes a Nat; recognize $(t) and bare numerals specially
      match e with
      | `(tritonExpr| $($t:term)) => `(Op.arange $t)
      | `(tritonExpr| $n:num)     => `(Op.arange $n)
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
            `(Op.arange $eTerm)
          else
            `(Op.add (Op.constNat $sTerm) (Op.arange ($eTerm - $sTerm)))
      | _ =>
          `(Op.add (Op.constNat $sTerm) (Op.arange ($eTerm - $sTerm)))
  | `(tritonExpr| tl.exp($e:tritonExpr)) => do
      let e' ← expandExpr e
      `(Op.exp $e')
  | `(tritonExpr| tl.log($e:tritonExpr)) => do
      let e' ← expandExpr e
      `(Op.log $e')
  | `(tritonExpr| tl.sigmoid($e:tritonExpr)) => do
      let e' ← expandExpr e
      `(Op.sigmoid $e')
  | `(tritonExpr| tl.sqrt($e:tritonExpr)) => do
      let e' ← expandExpr e
      `(Op.sqrt $e')
  | `(tritonExpr| tl.max($e:tritonExpr)) => do
      let e' ← expandExpr e
      `(Op.reduceMax $e')
  | `(tritonExpr| tl.max($a:tritonExpr, $b:tritonExpr)) => do
      let a' ← expandExpr a
      let b' ← expandExpr b
      `(Op.max2 $a' $b')
  | `(tritonExpr| tl.sum($e:tritonExpr)) => do
      let e' ← expandExpr e
      `(Op.reduceSum $e')
  | `(tritonExpr| tl.toReal($e:tritonExpr)) => do
      let e' ← expandExpr e
      `(Op.natToReal $e')
  | `(tritonExpr| -inf) =>
      `(Op.negInf)
  | `(tritonExpr| tl.load($p:tritonPtr)) => do
      -- Region is a Lean term of type RegionName (kernel parameter or value).
      -- The pointer-like surface syntax lowers to the internal region+offset AST.
      let (r, off) ← expandPtr p
      `(Op.load $r $off)
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
      let a' ← expandExpr a; let b' ← expandExpr b
      `(Op.add $a' $b')
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) => do
      let a' ← expandExpr a; let b' ← expandExpr b
      `(Op.sub $a' $b')
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => do
      let a' ← expandExpr a; let b' ← expandExpr b
      `(Op.mul $a' $b')
  | `(tritonExpr| $a:tritonExpr / $b:tritonExpr) => do
      let a' ← expandExpr a; let b' ← expandExpr b
      `(Op.div $a' $b')
  | _ => Macro.throwUnsupported

end

partial def expandStmt (stx : TSyntax `tritonStmt) : MacroM (TSyntax `term) := do
  match stx with
  | `(tritonStmt| $i:ident := $e:tritonExpr) => do
      let nameLit ← identAsStr i
      let e' ← expandExpr e
      `(Stmt.assign $nameLit $e')
  | `(tritonStmt| tl.store($p:tritonPtr, $v:tritonExpr)) => do
      let v' ← expandExpr v
      let (r, off) ← expandPtr p
      `(Stmt.store $r $off $v')
  | `(tritonStmt| tl.for $i:ident in $($n:term) { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let body ← stmts.mapM expandStmt
      `(Stmt.forLoop $nameLit $n [$body,*])
  | `(tritonStmt| tl.for $i:ident in $n:num { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let body ← stmts.mapM expandStmt
      `(Stmt.forLoop $nameLit $n [$body,*])
  | _ => Macro.throwUnsupported

/-! ## Region collection (for auto-populating Kernel.inputs / Kernel.outputs) -/

mutual

/-- Collect all region terms reachable from a `tritonExpr`. Returns `term`
    syntax — each element is the Lean term inside a `tl.load(...)`
    pointer (recursively in subexpressions). -/
private partial def exprRegions : TSyntax `tritonExpr → List (TSyntax `term) := fun stx =>
  match stx with
  | `(tritonExpr| tl.load($p:tritonPtr))         => ptrRegions p
  | `(tritonExpr| tl.exp($e:tritonExpr))         => exprRegions e
  | `(tritonExpr| tl.log($e:tritonExpr))         => exprRegions e
  | `(tritonExpr| tl.sigmoid($e:tritonExpr))     => exprRegions e
  | `(tritonExpr| tl.sqrt($e:tritonExpr))        => exprRegions e
  | `(tritonExpr| tl.max($e:tritonExpr))         => exprRegions e
  | `(tritonExpr| tl.max($a:tritonExpr, $b:tritonExpr))   =>
      exprRegions a ++ exprRegions b
  | `(tritonExpr| tl.sum($e:tritonExpr))         => exprRegions e
  | `(tritonExpr| tl.toReal($e:tritonExpr))      => exprRegions e
  | `(tritonExpr| ($e:tritonExpr))               => exprRegions e
  | `(tritonExpr| tl.program_id($e:tritonExpr))  => exprRegions e
  | `(tritonExpr| tl.arange($e:tritonExpr))      => exprRegions e
  | `(tritonExpr| tl.arange($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr / $b:tritonExpr) => exprRegions a ++ exprRegions b
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
  | `(tritonStmt| tl.store($p:tritonPtr, $v:tritonExpr)) =>
      (ptrOffsetRegions p ++ exprRegions v, ptrBaseRegion p)
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
      let stmtTerms ← stmts.mapM expandStmt
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
