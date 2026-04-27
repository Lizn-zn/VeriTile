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
    `tl.load($(xReg) + offs)` and `tl.store($(yReg) + offs, y)`.
    The base region is a Lean `RegionName` term written via `$(...)`;
    the offset is a DSL expression. The macro lowers this to the internal
    `(region, offset)` memory model. Scalar pointer sugar
    `tl.load($(xReg))` / `tl.store($(yReg), y)` lowers to offset `0`.
  * `$(<lean-term>)` antiquotes a Lean-level value; in numeric context it
    becomes `Op.const (·: ℝ)`, inside `tl.arange(...)` it is fed
    directly as the `Nat` length, and as the base pointer term in
    `tl.load($(REGION) + offset)` / `tl.store($(REGION) + offset, value)`
    or scalar-pointer `tl.load($(REGION))` / `tl.store($(REGION), value)`
    it is used as a `RegionName`.
  * Numeric literals become `Op.const`.
  * Statements are separated by newlines (no explicit terminator).

Currently supported expressions: `tl.program_id(_)`, `tl.arange(_)` /
`tl.arange(start, end)`, `tl.exp(_)`, `tl.log(_)`, `tl.sigmoid(_)`,
`tl.max(_)`, `tl.sum(_)`, `tl.load($(REGION) + offset)`, binary `+ - * /`,
parens, identifiers, numerals, antiquotation. `tl.load($(REGION))` is
sugar for offset `0`.

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

-- Expressions
syntax num : tritonExpr
syntax ident : tritonExpr
syntax "$(" term ")" : tritonExpr
syntax "(" tritonExpr ")" : tritonExpr
syntax "tl.program_id(" tritonExpr ")" : tritonExpr
syntax "tl.arange(" tritonExpr ")" : tritonExpr
syntax "tl.arange(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.exp(" tritonExpr ")" : tritonExpr
syntax "tl.log(" tritonExpr ")" : tritonExpr
syntax "tl.sigmoid(" tritonExpr ")" : tritonExpr
syntax "tl.max(" tritonExpr ")" : tritonExpr
syntax "tl.sum(" tritonExpr ")" : tritonExpr
syntax "tl.load($(" term ")" ")" : tritonExpr
syntax "tl.load($(" term ")" " + " tritonExpr ")" : tritonExpr
syntax:60 tritonExpr:60 " + " tritonExpr:61 : tritonExpr
syntax:60 tritonExpr:60 " - " tritonExpr:61 : tritonExpr
syntax:70 tritonExpr:70 " * " tritonExpr:71 : tritonExpr
syntax:70 tritonExpr:70 " / " tritonExpr:71 : tritonExpr

-- Statements
syntax ident " := " tritonExpr : tritonStmt
syntax "tl.store($(" term ")" ", " tritonExpr ")" : tritonStmt
syntax "tl.store($(" term ")" " + " tritonExpr ", " tritonExpr ")" : tritonStmt

-- Block (the user-facing entry point)
syntax (name := tritonBlock) "triton " "{" tritonStmt* "}" : term

/-! ## Expansion -/

private def identAsStr (i : TSyntax `ident) : MacroM (TSyntax `term) :=
  pure (Syntax.mkStrLit i.getId.toString)

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
  | `(tritonExpr| tl.max($e:tritonExpr)) => do
      let e' ← expandExpr e
      `(Op.reduceMax $e')
  | `(tritonExpr| tl.sum($e:tritonExpr)) => do
      let e' ← expandExpr e
      `(Op.reduceSum $e')
  | `(tritonExpr| tl.load($($r:term))) =>
      -- Scalar pointer sugar: `tl.load(ptr)` reads `ptr + 0`.
      `(Op.load $r (Op.constNat 0))
  | `(tritonExpr| tl.load($($r:term) + $o:tritonExpr)) => do
      -- Region is a Lean term of type RegionName (kernel parameter or value).
      -- The pointer-like surface syntax lowers to the internal region+offset AST.
      let o' ← expandExpr o
      `(Op.load $r $o')
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

partial def expandStmt (stx : TSyntax `tritonStmt) : MacroM (TSyntax `term) := do
  match stx with
  | `(tritonStmt| $i:ident := $e:tritonExpr) => do
      let nameLit ← identAsStr i
      let e' ← expandExpr e
      `(Stmt.assign $nameLit $e')
  | `(tritonStmt| tl.store($($r:term), $v:tritonExpr)) => do
      -- Scalar pointer sugar: `tl.store(ptr, value)` writes `ptr + 0`.
      let v' ← expandExpr v
      `(Stmt.store $r (Op.constNat 0) $v')
  | `(tritonStmt| tl.store($($r:term) + $o:tritonExpr, $v:tritonExpr)) => do
      -- Region is a Lean term of type RegionName (kernel parameter or value).
      -- The pointer-like surface syntax lowers to the internal region+offset AST.
      let o' ← expandExpr o
      let v' ← expandExpr v
      `(Stmt.store $r $o' $v')
  | _ => Macro.throwUnsupported

/-! ## Region collection (for auto-populating Kernel.inputs / Kernel.outputs) -/

/-- Collect all region terms reachable from a `tritonExpr`. Returns `term`
    syntax — each element is the Lean term inside a `tl.load($(R) + …)`
    base-region antiquote in this expression (or recursively in subexpressions). -/
private partial def exprRegions : TSyntax `tritonExpr → List (TSyntax `term) := fun stx =>
  match stx with
  | `(tritonExpr| tl.load($($r:term))) =>
      [r]
  | `(tritonExpr| tl.load($($r:term) + $o:tritonExpr)) =>
      r :: exprRegions o
  | `(tritonExpr| tl.exp($e:tritonExpr))         => exprRegions e
  | `(tritonExpr| tl.log($e:tritonExpr))         => exprRegions e
  | `(tritonExpr| tl.sigmoid($e:tritonExpr))     => exprRegions e
  | `(tritonExpr| tl.max($e:tritonExpr))         => exprRegions e
  | `(tritonExpr| tl.sum($e:tritonExpr))         => exprRegions e
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

/-- Per-statement region split: `(input regions, output regions)`. -/
private def stmtRegions :
    TSyntax `tritonStmt → List (TSyntax `term) × List (TSyntax `term) := fun stx =>
  match stx with
  | `(tritonStmt| $_:ident := $e:tritonExpr) =>
      (exprRegions e, [])
  | `(tritonStmt| tl.store($($r:term), $v:tritonExpr)) =>
      (exprRegions v, [r])
  | `(tritonStmt| tl.store($($r:term) + $o:tritonExpr, $v:tritonExpr)) =>
      (exprRegions o ++ exprRegions v, [r])
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
