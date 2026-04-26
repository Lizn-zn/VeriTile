/-
VeriTile.Triton.DSL

A `triton { ... }` macro that embeds Triton-style kernel syntax inside Lean,
modelled on the `arm64 { ... }` style used in arm-in-lean.

Example:

  def naiveSoftmaxKernel (N : Nat) : Kernel := triton {
    pid  := tl.program_id(0)
    offs := pid * $(N) + tl.arange($(N))
    x    := tl.load(X, offs)
    e    := tl.exp(x)
    s    := tl.sum(e)
    y    := e / s
    tl.store(Y, offs, y)
  }

Conventions:
  * Bare identifiers in expression position → register references
    (`pid`, `x`, `e`, etc. become `Op.ref "pid"`, `Op.ref "x"`, ...).
  * First argument of `tl.load` and `tl.store` is a bare identifier and
    is interpreted as a memory region name (string literal).
  * `$(<lean-term>)` antiquotes a Lean-level value; in numeric context it
    becomes `Op.const (·: ℝ)`, and inside `tl.arange(...)` it is fed
    directly as the `Nat` length.
  * Numeric literals become `Op.const`.
  * Statements are separated by newlines (no explicit terminator).

Currently supported expressions: `tl.program_id(_)`, `tl.arange(_)`,
`tl.exp(_)`, `tl.max(_)`, `tl.sum(_)`, `tl.load(REGION, offset)`,
binary `+ - * /`, parens, identifiers, numerals, antiquotation.

Currently supported statements: assignment (`name := expr`), `tl.store(...)`.

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
syntax "tl.exp(" tritonExpr ")" : tritonExpr
syntax "tl.log(" tritonExpr ")" : tritonExpr
syntax "tl.max(" tritonExpr ")" : tritonExpr
syntax "tl.sum(" tritonExpr ")" : tritonExpr
syntax "tl.load(" ident ", " tritonExpr ")" : tritonExpr
syntax:60 tritonExpr:60 " + " tritonExpr:61 : tritonExpr
syntax:60 tritonExpr:60 " - " tritonExpr:61 : tritonExpr
syntax:70 tritonExpr:70 " * " tritonExpr:71 : tritonExpr
syntax:70 tritonExpr:70 " / " tritonExpr:71 : tritonExpr

-- Statements
syntax ident " := " tritonExpr : tritonStmt
syntax "tl.store(" ident ", " tritonExpr ", " tritonExpr ")" : tritonStmt

-- Block (the user-facing entry point)
syntax (name := tritonBlock) "triton " "{" tritonStmt* "}" : term

/-! ## Expansion -/

private def identAsStr (i : TSyntax `ident) : MacroM (TSyntax `term) :=
  pure (Syntax.mkStrLit i.getId.toString)

partial def expandExpr (stx : TSyntax `tritonExpr) : MacroM (TSyntax `term) := do
  match stx with
  | `(tritonExpr| $n:num) =>
      `(Op.const (($n : Nat) : ℝ))
  | `(tritonExpr| $i:ident) =>
      let s ← identAsStr i
      `(Op.ref $s)
  | `(tritonExpr| $($t:term)) =>
      `(Op.const (($t : Nat) : ℝ))
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
  | `(tritonExpr| tl.exp($e:tritonExpr)) => do
      let e' ← expandExpr e
      `(Op.exp $e')
  | `(tritonExpr| tl.log($e:tritonExpr)) => do
      let e' ← expandExpr e
      `(Op.log $e')
  | `(tritonExpr| tl.max($e:tritonExpr)) => do
      let e' ← expandExpr e
      `(Op.reduceMax $e')
  | `(tritonExpr| tl.sum($e:tritonExpr)) => do
      let e' ← expandExpr e
      `(Op.reduceSum $e')
  | `(tritonExpr| tl.load($r:ident, $o:tritonExpr)) => do
      let regLit ← identAsStr r
      let o' ← expandExpr o
      `(Op.load $regLit $o')
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
  | `(tritonStmt| tl.store($r:ident, $o:tritonExpr, $v:tritonExpr)) => do
      let regLit ← identAsStr r
      let o' ← expandExpr o
      let v' ← expandExpr v
      `(Stmt.store $regLit $o' $v')
  | _ => Macro.throwUnsupported

/-! ## Block macro -/

macro_rules
  | `(triton { $stmts:tritonStmt* }) => do
      let stmtTerms ← stmts.mapM expandStmt
      -- For now, hardcode inputs/outputs. P2 work: scan the body for tl.load
      -- and tl.store regions and synthesise these lists automatically.
      `(Kernel.mk ["X"] ["Y"] [$stmtTerms,*])

end VeriTile.Triton.DSL
