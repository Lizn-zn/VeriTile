/-
VeriTile.Triton.DSL.Inference

Syntax-level inference of `.nat` dtype hints for `tl.load(...)` bindings.

The DSL macro expands top-to-bottom; when expanding a binding
`name = tl.load(p)` it must commit to a dtype before knowing how `name`
is later used. Without a hint, `tl.load` defaults to `.real`. If `name`
is then used as a pointer offset (e.g. `Region + name * stride + ...`),
that requires `.nat` and macro expansion fails.

This module pre-scans the kernel body and returns the set of identifier
names that appear in `.nat`-pinning syntactic positions. The macro looks
up the binding name in this set when no explicit `dtype=` kwarg is given
and uses `.nat` as the default in that case.
-/

import VeriTile.Triton.DSL.Syntax

open Lean

namespace VeriTile.Triton.DSL.Inference

/-- Walk an expression that occupies a `.nat`-required position and
collect every identifier appearing within it. Recurses through
arithmetic (`+ - * / // % << >> & | ^ ~`), parens, and the unit-axis
indexing forms `e[: , None]` / `e[None , :]` plus `tl.expand_dims`.
Stops at antiquotations (`$(...)`), `tl.load`, `tl.cast`, and any other
constructor whose result type is fixed independently. -/
partial def natExprIdents : TSyntax `tritonExpr → List String := fun stx =>
  match stx with
  | `(tritonExpr| $r:ident) => [r.getId.toString]
  | `(tritonExpr| ($e:tritonExpr)) => natExprIdents e
  | `(tritonExpr| $a:tritonExpr +  $b:tritonExpr) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| $a:tritonExpr -  $b:tritonExpr) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| $a:tritonExpr *  $b:tritonExpr) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| $a:tritonExpr /  $b:tritonExpr) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| $a:tritonExpr %  $b:tritonExpr) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| $a:tritonExpr << $b:tritonExpr) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| $a:tritonExpr >> $b:tritonExpr) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| $a:tritonExpr &  $b:tritonExpr) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| $a:tritonExpr |  $b:tritonExpr) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| $a:tritonExpr ^  $b:tritonExpr) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| ~ $e:tritonExpr) => natExprIdents e
  | `(tritonExpr| $e:tritonExpr[ : , None ]) => natExprIdents e
  | `(tritonExpr| $e:tritonExpr[ None , : ]) => natExprIdents e
  | `(tritonExpr| tl.expand_dims($e:tritonExpr, $_:tritonReduceKwarg)) => natExprIdents e
  | `(tritonExpr| tl.expand_dims($e:tritonExpr, $_:num)) => natExprIdents e
  | _ => []

/-- The names of identifiers that have been previously assigned in the
kernel body. Used to distinguish a static pointer base (a parameter
identifier) from a locally-bound pointer name. -/
abbrev Assigned := List String

/-- Test whether `stx` would be recognized as a static pointer
expression by the macro's `expandStaticPtrExpr`. Mirrors the cases there. -/
private partial def isStaticPtr (assigned : Assigned) :
    TSyntax `tritonExpr → Bool := fun stx =>
  match stx with
  | `(tritonExpr| $($_t:term)) => Bool.true
  | `(tritonExpr| $r:ident) => !(assigned.contains r.getId.toString)
  | `(tritonExpr| ($e:tritonExpr)) => isStaticPtr assigned e
  | `(tritonExpr| $a:tritonExpr + $_b:tritonExpr) => isStaticPtr assigned a
  | _ => Bool.false

/-- For a static-ptr-add chain (i.e. `isStaticPtr` returned `true`),
return the `.nat`-required offset sub-expressions. For a chain
`(((base + b1) + b2) + b3)`, this returns `[b1, b2, b3]`. -/
private partial def staticPtrChainOffsets :
    TSyntax `tritonExpr → List (TSyntax `tritonExpr) := fun stx =>
  match stx with
  | `(tritonExpr| ($e:tritonExpr)) => staticPtrChainOffsets e
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) =>
      b :: staticPtrChainOffsets a
  | _ => []

/-- For an expression in a "pointer expression" context (the address
argument of `tl.load`, `tl.store`, `tl.atomic_*`), collect the
identifiers pinned to `.nat` by the offsets of any static-ptr-add chain
inside it. -/
private partial def pinsFromPtrExpr (assigned : Assigned)
    (stx : TSyntax `tritonExpr) : List String :=
  if isStaticPtr assigned stx then
    let offsets := staticPtrChainOffsets stx
    offsets.foldl (fun acc o => acc ++ natExprIdents o) []
  else
    []

/-- For an expression in a value-position context, collect identifiers
pinned to `.nat`. The two ways an expression contributes pins:
1. The expression is itself a static-ptr-add chain (e.g. RHS of
   `name = Region + offset`); harvest its offset idents.
2. The expression syntactically contains a `tl.load(ptr_expr)` whose
   `ptr_expr` is a static-ptr-add chain; recurse into the load's
   pointer arg. -/
private partial def pinsFromExpr (assigned : Assigned) :
    TSyntax `tritonExpr → List String := fun stx =>
  let topPins :=
    if isStaticPtr assigned stx then
      let offsets := staticPtrChainOffsets stx
      offsets.foldl (fun acc o => acc ++ natExprIdents o) []
    else []
  let belowPins :=
    match stx with
    | `(tritonExpr| ($e:tritonExpr)) => pinsFromExpr assigned e
    | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
        let kwargPins :=
          kwargs.foldl (fun (acc : List String) kw =>
            match kw with
            | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ pinsFromExpr assigned val
            | _ => acc) []
        pinsFromPtrExpr assigned p ++ kwargPins
    | _ => []
  topPins ++ belowPins

private def memKwargPins (assigned : Assigned)
    (kwargs : TSyntaxArray `tritonMemKwarg) : List String :=
  kwargs.foldl (fun (acc : List String) kw =>
    match kw with
    | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ pinsFromExpr assigned val
    | _ => acc) []

mutual

private partial def pinsFromStmts (assigned : Assigned)
    (stmts : List (TSyntax `tritonStmt)) : List String × Assigned :=
  stmts.foldl
    (fun (acc : List String × Assigned) st =>
      let (pins, nextAssigned) := pinsFromStmt acc.2 st
      (acc.1 ++ pins, nextAssigned))
    ([], assigned)

private partial def pinsFromStmt (assigned : Assigned)
    (stx : TSyntax `tritonStmt) : List String × Assigned :=
  match stx with
  | `(tritonStmt| $lhs0:ident, $lhs1:ident $[, $lhsRest:ident]* = $rhs0:tritonExpr, $rhs1:tritonExpr $[, $rhsRest:tritonExpr]*) =>
      let lhs := #[lhs0, lhs1] ++ lhsRest
      let rhs := #[rhs0, rhs1] ++ rhsRest
      let pins := rhs.foldl (fun acc e => acc ++ pinsFromExpr assigned e) []
      let nextAssigned := lhs.foldl (fun acc i => i.getId.toString :: acc) assigned
      (pins, nextAssigned)
  | `(tritonStmt| $lhs0:ident, $lhs1:ident $[, $lhsRest:ident]* := $rhs0:tritonExpr, $rhs1:tritonExpr $[, $rhsRest:tritonExpr]*) =>
      let lhs := #[lhs0, lhs1] ++ lhsRest
      let rhs := #[rhs0, rhs1] ++ rhsRest
      let pins := rhs.foldl (fun acc e => acc ++ pinsFromExpr assigned e) []
      let nextAssigned := lhs.foldl (fun acc i => i.getId.toString :: acc) assigned
      (pins, nextAssigned)
  | `(tritonStmt| $i:ident := $e:tritonExpr) =>
      (pinsFromExpr assigned e, i.getId.toString :: assigned)
  | `(tritonStmt| $i:ident = $e:tritonExpr) =>
      (pinsFromExpr assigned e, i.getId.toString :: assigned)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (pinsFromPtrExpr assigned p ++ pinsFromExpr assigned v ++ memKwargPins assigned kwargs,
        assigned)
  | `(tritonStmt| tl.atomic_add($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (pinsFromPtrExpr assigned p ++ pinsFromExpr assigned v ++ memKwargPins assigned kwargs,
        assigned)
  | `(tritonStmt| tl.atomic_max($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (pinsFromPtrExpr assigned p ++ pinsFromExpr assigned v ++ memKwargPins assigned kwargs,
        assigned)
  | `(tritonStmt| tl.atomic_min($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (pinsFromPtrExpr assigned p ++ pinsFromExpr assigned v ++ memKwargPins assigned kwargs,
        assigned)
  | `(tritonStmt| tl.atomic_and($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (pinsFromPtrExpr assigned p ++ pinsFromExpr assigned v ++ memKwargPins assigned kwargs,
        assigned)
  | `(tritonStmt| tl.atomic_or($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (pinsFromPtrExpr assigned p ++ pinsFromExpr assigned v ++ memKwargPins assigned kwargs,
        assigned)
  | `(tritonStmt| tl.atomic_xor($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (pinsFromPtrExpr assigned p ++ pinsFromExpr assigned v ++ memKwargPins assigned kwargs,
        assigned)
  | `(tritonStmt| tl.atomic_xchg($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (pinsFromPtrExpr assigned p ++ pinsFromExpr assigned v ++ memKwargPins assigned kwargs,
        assigned)
  | `(tritonStmt| tl.atomic_cas($p:tritonExpr, $cmp:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (pinsFromPtrExpr assigned p ++ pinsFromExpr assigned cmp ++
        pinsFromExpr assigned v ++ memKwargPins assigned kwargs,
        assigned)
  | `(tritonStmt| tl.async_copy($dst:tritonExpr, $src:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (pinsFromPtrExpr assigned dst ++ pinsFromPtrExpr assigned src ++
        pinsFromExpr assigned src ++ memKwargPins assigned kwargs,
        assigned)
  | `(tritonStmt| tl.async_wait()) => ([], assigned)
  | `(tritonStmt| tl.debug_barrier()) => ([], assigned)
  | `(tritonStmt| tl.for $i:ident in $($_:term) { $stmts:tritonStmt* }) =>
      let bodyAssigned := i.getId.toString :: assigned
      let (pins, _) := pinsFromStmts bodyAssigned stmts.toList
      (pins, assigned)
  | `(tritonStmt| tl.for $i:ident in $_:num { $stmts:tritonStmt* }) =>
      let bodyAssigned := i.getId.toString :: assigned
      let (pins, _) := pinsFromStmts bodyAssigned stmts.toList
      (pins, assigned)
  | `(tritonStmt| tl.static_range $i:ident in $($_:term) { $stmts:tritonStmt* }) =>
      let bodyAssigned := i.getId.toString :: assigned
      let (pins, _) := pinsFromStmts bodyAssigned stmts.toList
      (pins, assigned)
  | `(tritonStmt| tl.static_range $i:ident in $_:num { $stmts:tritonStmt* }) =>
      let bodyAssigned := i.getId.toString :: assigned
      let (pins, _) := pinsFromStmts bodyAssigned stmts.toList
      (pins, assigned)
  | `(tritonStmt| tl.if $cond:tritonExpr { $thenStmts:tritonStmt* } else { $elseStmts:tritonStmt* }) =>
      let condPins := pinsFromExpr assigned cond
      let (thenPins, _) := pinsFromStmts assigned thenStmts.toList
      let (elsePins, _) := pinsFromStmts assigned elseStmts.toList
      (condPins ++ thenPins ++ elsePins, assigned)
  | `(tritonStmt| tl.if $cond:tritonExpr { $stmts:tritonStmt* }) =>
      let condPins := pinsFromExpr assigned cond
      let (pins, _) := pinsFromStmts assigned stmts.toList
      (condPins ++ pins, assigned)
  | `(tritonStmt| if $cond:tritonExpr { $thenStmts:tritonStmt* } else { $elseStmts:tritonStmt* }) =>
      let condPins := pinsFromExpr assigned cond
      let (thenPins, _) := pinsFromStmts assigned thenStmts.toList
      let (elsePins, _) := pinsFromStmts assigned elseStmts.toList
      (condPins ++ thenPins ++ elsePins, assigned)
  | `(tritonStmt| if $cond:tritonExpr { $stmts:tritonStmt* }) =>
      let condPins := pinsFromExpr assigned cond
      let (pins, _) := pinsFromStmts assigned stmts.toList
      (condPins ++ pins, assigned)
  | _ => ([], assigned)

end

/-- Set of identifier names that the body uses in a `.nat`-required
position. The DSL macro consults this when expanding a `name = tl.load(p)`
binding without an explicit `dtype=` kwarg: if `name` is in this set,
the load defaults to `.nat` instead of `.real`. -/
def collectNatPinned (stmts : List (TSyntax `tritonStmt)) : List String :=
  let (pins, _) := pinsFromStmts [] stmts
  pins

end VeriTile.Triton.DSL.Inference
