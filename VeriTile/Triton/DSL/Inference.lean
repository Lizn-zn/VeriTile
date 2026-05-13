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
import VeriTile.Triton.DSL.Typing

open Lean

namespace VeriTile.Triton.DSL.Inference

open VeriTile.Triton.DSL (DInfo)

/-- Walk an expression that occupies a `.nat`-required position and
collect every identifier appearing within it. Recurses through
arithmetic (`+ - * / // % << >> & | ^ ~ not`), parens, and the unit-axis
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
  | `(tritonExpr| _div_up($a:tritonExpr, $b:tritonExpr)) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| min($a:tritonExpr, $b:tritonExpr)) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| $a:tritonExpr << $b:tritonExpr) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| $a:tritonExpr >> $b:tritonExpr) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| $a:tritonExpr &  $b:tritonExpr) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| $a:tritonExpr |  $b:tritonExpr) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| $a:tritonExpr ^  $b:tritonExpr) => natExprIdents a ++ natExprIdents b
  | `(tritonExpr| ~ $e:tritonExpr) => natExprIdents e
  | `(tritonExpr| not $e:tritonExpr) => natExprIdents e
  | `(tritonExpr| $e:tritonExpr[ : , None ]) => natExprIdents e
  | `(tritonExpr| $e:tritonExpr[ None , : ]) => natExprIdents e
  | `(tritonExpr| $e:tritonExpr[ None ]) => natExprIdents e
  | `(tritonExpr| $e:tritonExpr[ : , None , None ]) => natExprIdents e
  | `(tritonExpr| $e:tritonExpr[ None , : , None ]) => natExprIdents e
  | `(tritonExpr| $e:tritonExpr[ None , None , : ]) => natExprIdents e
  | `(tritonExpr| tl.expand_dims($e:tritonExpr, $_:tritonReduceKwarg)) => natExprIdents e
  | `(tritonExpr| tl.expand_dims($e:tritonExpr, $_:num)) => natExprIdents e
  | _ => []

/-- The names of identifiers that have been previously assigned in the
kernel body. Used to distinguish a static pointer base (a parameter
identifier) from a locally-bound pointer name. -/
abbrev Assigned := List String

private def addUnique (xs : List String) (x : String) : List String :=
  if xs.contains x then xs else x :: xs

private def addManyUnique (xs ys : List String) : List String :=
  ys.foldl addUnique xs

private def anyContained (needles haystack : List String) : Bool :=
  needles.any (fun x => haystack.contains x)

private def typedRegionTermIsInt? (r : TSyntax `term) : Bool :=
  match r with
  | `(($_:term : Region .int)) => Bool.true
  | `(($_:term : Region TileDType.int)) => Bool.true
  | _ => Bool.false

private partial def staticPtrRootIsIntRegion? (stx : TSyntax `tritonExpr) : Bool :=
  match stx with
  | `(tritonExpr| $($r:term)) => typedRegionTermIsInt? r
  | `(tritonExpr| ($e:tritonExpr)) => staticPtrRootIsIntRegion? e
  | `(tritonExpr| $a:tritonExpr + $_b:tritonExpr) => staticPtrRootIsIntRegion? a
  | _ => Bool.false

private partial def intSourceExpr? : TSyntax `tritonExpr → Bool := fun stx =>
  match stx with
  | `(tritonExpr| $(($_:term : Int))) => Bool.true
  | `(tritonExpr| tl.load($p:tritonExpr, $_mask:tritonExpr $[, $_kwargs:tritonMemKwarg]*)) =>
      staticPtrRootIsIntRegion? p
  | `(tritonExpr| tl.load($p:tritonExpr $[, $_kwargs:tritonMemKwarg]*)) =>
      staticPtrRootIsIntRegion? p
  | `(tritonExpr| ($e:tritonExpr)) => intSourceExpr? e
  | _ => Bool.false

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
    match stx with
    | `(tritonExpr| ($e:tritonExpr)) => pinsFromPtrExpr assigned e
    | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) =>
        let basePins :=
          match a with
          | `(tritonExpr| $r:ident) =>
              if assigned.contains r.getId.toString then natExprIdents b else []
          | _ => []
        basePins ++ pinsFromPtrExpr assigned a ++ pinsFromPtrExpr assigned b
    | _ => []

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
    | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
        let kwargPins :=
          kwargs.foldl (fun (acc : List String) kw =>
            match kw with
            | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ pinsFromExpr assigned val
            | _ => acc) []
        pinsFromPtrExpr assigned p ++ pinsFromExpr assigned mask ++ kwargPins
    | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
        let kwargPins :=
          kwargs.foldl (fun (acc : List String) kw =>
            match kw with
            | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ pinsFromExpr assigned val
            | _ => acc) []
        pinsFromPtrExpr assigned p ++ kwargPins
    | `(tritonExpr| tl.make_block_ptr($_:ident=$p:tritonExpr,
        $_:ident=($_parent:tritonExpr,*), $_:ident=($_strides:tritonExpr,*),
        $_:ident=($_offsets:tritonExpr,*), $_:ident=($_block:tritonExpr,*))) =>
        pinsFromPtrExpr assigned p
    | `(tritonExpr| tl.make_block_ptr($_:ident=$p:tritonExpr,
        $_:ident=($_parent:tritonExpr,*), $_:ident=($_strides:tritonExpr,*),
        $_:ident=($_offsets:tritonExpr,*), $_:ident=($_block:tritonExpr,*),
        $_:ident=($_order:num,*))) =>
        pinsFromPtrExpr assigned p
    | _ => []
  topPins ++ belowPins

/-- Assignment-level `.nat` dependencies: if the LHS is later known to
be `.nat`, identifiers occurring in arithmetic/indexing positions on
the RHS must also be `.nat`. -/
private partial def natDepsFromAssignedExpr :
    TSyntax `tritonExpr → List String := fun stx =>
  natExprIdents stx

/-- A local pointer value can be built from another pointer plus a Nat
offset. If the LHS is later used as a pointer, the returned first list
contains local pointer bases that also become pointer-used, and the
second list contains Nat offsets that must be pinned. -/
private partial def ptrDepsFromAssignedExpr (assigned : Assigned) :
    TSyntax `tritonExpr → List String × List String := fun stx =>
  match stx with
  | `(tritonExpr| ($e:tritonExpr)) => ptrDepsFromAssignedExpr assigned e
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) =>
      match a with
      | `(tritonExpr| $r:ident) =>
          let name := r.getId.toString
          if assigned.contains name then
            ([name], natExprIdents b)
          else if isStaticPtr assigned stx then
            ([], staticPtrChainOffsets stx |>.foldl (fun acc o => acc ++ natExprIdents o) [])
          else
            ([], [])
      | _ =>
          if isStaticPtr assigned stx then
            ([], staticPtrChainOffsets stx |>.foldl (fun acc o => acc ++ natExprIdents o) [])
          else
            ([], [])
  | _ => ([], [])

/-- Local identifiers used directly as memory pointers, e.g. `tl.load(p)`
after `p = base + off`. -/
private partial def ptrUsesFromPtrExpr (assigned : Assigned) :
    TSyntax `tritonExpr → List String := fun stx =>
  match stx with
  | `(tritonExpr| ($e:tritonExpr)) => ptrUsesFromPtrExpr assigned e
  | `(tritonExpr| $r:ident) =>
      let name := r.getId.toString
      if assigned.contains name then [name] else []
  | `(tritonExpr| $a:tritonExpr + $_b:tritonExpr) =>
      ptrUsesFromPtrExpr assigned a
  | _ => []

/-- Conditional comparison dependencies. Comparisons are not always Nat
comparisons (`cur_logits > 0` is real), so we only use these after the
fixed point already knows at least one side is Nat. -/
private partial def cmpDepsFromExpr :
    TSyntax `tritonExpr → List (List String × List String) := fun stx =>
  let cmp (a b : TSyntax `tritonExpr) :=
    let ids := natExprIdents a ++ natExprIdents b
    [(ids, ids)]
  match stx with
  | `(tritonExpr| ($e:tritonExpr)) => cmpDepsFromExpr e
  | `(tritonExpr| $a:tritonExpr < $b:tritonExpr) => cmp a b
  | `(tritonExpr| $a:tritonExpr <= $b:tritonExpr) => cmp a b
  | `(tritonExpr| $a:tritonExpr == $b:tritonExpr) => cmp a b
  | `(tritonExpr| $a:tritonExpr > $b:tritonExpr) => cmp a b
  | `(tritonExpr| $a:tritonExpr >= $b:tritonExpr) => cmp a b
  | `(tritonExpr| $a:tritonExpr != $b:tritonExpr) => cmp a b
  | `(tritonExpr| tl.logical_and($a:tritonExpr, $b:tritonExpr)) =>
      cmpDepsFromExpr a ++ cmpDepsFromExpr b
  | `(tritonExpr| tl.logical_or($a:tritonExpr, $b:tritonExpr)) =>
      cmpDepsFromExpr a ++ cmpDepsFromExpr b
  | `(tritonExpr| $a:tritonExpr and $b:tritonExpr) =>
      cmpDepsFromExpr a ++ cmpDepsFromExpr b
  | `(tritonExpr| tl.where($c:tritonExpr, $t:tritonExpr, $f:tritonExpr)) =>
      cmpDepsFromExpr c ++ cmpDepsFromExpr t ++ cmpDepsFromExpr f
  | `(tritonExpr| tl.extra.cuda.libdevice.pow($e:tritonExpr, $_:num)) =>
      cmpDepsFromExpr e
  | _ => []

private partial def directPinsFromExpr (assigned : Assigned) :
    TSyntax `tritonExpr → List String := fun stx =>
  if stx.raw.getKind == ``VeriTile.Triton.DSL.tritonMethodCast ||
      stx.raw.getKind == ``VeriTile.Triton.DSL.tritonMethodCastDTypeIdent ||
      stx.raw.getKind == ``VeriTile.Triton.DSL.tritonMethodCastElementTy ||
      stx.raw.getKind == ``VeriTile.Triton.DSL.tritonMethodCastElementTyIdent ||
      stx.raw.getKind == ``VeriTile.Triton.DSL.tritonIdentMethodCast ||
      stx.raw.getKind == ``VeriTile.Triton.DSL.tritonIdentMethodCastSpaced ||
      stx.raw.getKind == ``VeriTile.Triton.DSL.tritonIdentMethodCastDTypeIdent ||
      stx.raw.getKind == ``VeriTile.Triton.DSL.tritonIdentMethodCastDTypeIdentSpaced ||
      stx.raw.getKind == ``VeriTile.Triton.DSL.tritonIdentMethodCastElementTyIdent ||
      stx.raw.getKind == ``VeriTile.Triton.DSL.tritonIdentMethodCastElementTyIdentSpaced then
    let args := stx.raw.getArgs
    if h : args.size = 5 then
      let e : TSyntax `tritonExpr := ⟨args[0]⟩
      directPinsFromExpr assigned e
    else if h : args.size = 6 then
      let e : TSyntax `tritonExpr := ⟨args[0]⟩
      directPinsFromExpr assigned e
    else
      []
  else
  match stx with
  | `(tritonExpr| ($e:tritonExpr)) => directPinsFromExpr assigned e
  | `(tritonExpr| tl.toReal($e:tritonExpr)) => natExprIdents e
  | `(tritonExpr| tl.multiple_of($e:tritonExpr, $align:tritonExpr)) =>
      directPinsFromExpr assigned e ++ directPinsFromExpr assigned align
  | `(tritonExpr| tl.max_contiguous($e:tritonExpr, $align:tritonExpr)) =>
      directPinsFromExpr assigned e ++ directPinsFromExpr assigned align
  | `(tritonExpr| tl.make_block_ptr($_:ident=$p:tritonExpr,
        $_:ident=($_parent:tritonExpr,*), $_:ident=($_strides:tritonExpr,*),
        $_:ident=($_offsets:tritonExpr,*), $_:ident=($_block:tritonExpr,*))) =>
      pinsFromPtrExpr assigned p
  | `(tritonExpr| tl.make_block_ptr($_:ident=$p:tritonExpr,
        $_:ident=($_parent:tritonExpr,*), $_:ident=($_strides:tritonExpr,*),
        $_:ident=($_offsets:tritonExpr,*), $_:ident=($_block:tritonExpr,*),
        $_:ident=($_order:num,*))) =>
      pinsFromPtrExpr assigned p
  | `(tritonExpr| tl.where($c:tritonExpr, $t:tritonExpr, $f:tritonExpr)) =>
      directPinsFromExpr assigned c ++ directPinsFromExpr assigned t ++ directPinsFromExpr assigned f
  | `(tritonExpr| tl.extra.cuda.libdevice.pow($e:tritonExpr, $_:num)) =>
      directPinsFromExpr assigned e
  | `(tritonExpr| $a:tritonExpr +  $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr -  $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr *  $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr /  $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr %  $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| _div_up($a:tritonExpr, $b:tritonExpr)) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| min($a:tritonExpr, $b:tritonExpr)) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr << $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr >> $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr &  $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr |  $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr ^  $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr and $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr <  $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr <= $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr == $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr >  $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr >= $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| $a:tritonExpr != $b:tritonExpr) => directPinsFromExpr assigned a ++ directPinsFromExpr assigned b
  | `(tritonExpr| ~ $e:tritonExpr) => directPinsFromExpr assigned e
  | `(tritonExpr| not $e:tritonExpr) => directPinsFromExpr assigned e
  | `(tritonExpr| - $e:tritonExpr) => directPinsFromExpr assigned e
  | `(tritonExpr| $e:tritonExpr[ None ]) => directPinsFromExpr assigned e
  | `(tritonExpr| $e:tritonExpr[ : , None , None ]) => directPinsFromExpr assigned e
  | `(tritonExpr| $e:tritonExpr[ None , : , None ]) => directPinsFromExpr assigned e
  | `(tritonExpr| $e:tritonExpr[ None , None , : ]) => directPinsFromExpr assigned e
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      let kwargPins :=
        kwargs.foldl (fun (acc : List String) kw =>
          match kw with
          | `(tritonMemKwarg| $_:ident = $val:tritonExpr) =>
              acc ++ directPinsFromExpr assigned val
          | _ => acc) []
      pinsFromPtrExpr assigned p ++ directPinsFromExpr assigned mask ++ kwargPins
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      let kwargPins :=
        kwargs.foldl (fun (acc : List String) kw =>
          match kw with
          | `(tritonMemKwarg| $_:ident = $val:tritonExpr) =>
              acc ++ directPinsFromExpr assigned val
          | _ => acc) []
      pinsFromPtrExpr assigned p ++ kwargPins
  | _ => pinsFromExpr assigned stx

private def memKwargPins (assigned : Assigned)
    (kwargs : TSyntaxArray `tritonMemKwarg) : List String :=
  kwargs.foldl (fun (acc : List String) kw =>
    match kw with
    | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ directPinsFromExpr assigned val
    | _ => acc) []

private def memKwargCmpDeps
    (kwargs : TSyntaxArray `tritonMemKwarg) : List (List String × List String) :=
  kwargs.foldl (fun (acc : List (List String × List String)) kw =>
    match kw with
    | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ cmpDepsFromExpr val
    | _ => acc) []

structure ScanInfo where
  directPins : List String := []
  natDeps : List (String × List String) := []
  ptrUses : List String := []
  ptrDeps : List (String × List String × List String) := []
  cmpDeps : List (List String × List String) := []
  intPins : List String := []
  deriving Inhabited

private def ScanInfo.append (a b : ScanInfo) : ScanInfo :=
  { directPins := a.directPins ++ b.directPins
    natDeps := a.natDeps ++ b.natDeps
    ptrUses := a.ptrUses ++ b.ptrUses
    ptrDeps := a.ptrDeps ++ b.ptrDeps
    cmpDeps := a.cmpDeps ++ b.cmpDeps
    intPins := a.intPins ++ b.intPins }

private def assignmentInfo (assigned : Assigned) (lhs : List String)
    (rhs : List (TSyntax `tritonExpr)) : ScanInfo :=
  let directPins := rhs.foldl (fun acc e => acc ++ directPinsFromExpr assigned e) []
  let cmpDeps := rhs.foldl (fun acc e => acc ++ cmpDepsFromExpr e) []
  let natDeps := (lhs.zip rhs).map (fun (l, e) => (l, natDepsFromAssignedExpr e))
  let ptrDeps := (lhs.zip rhs).map (fun (l, e) =>
    let (bases, offsets) := ptrDepsFromAssignedExpr assigned e
    (l, bases, offsets))
  let intPins := (lhs.zip rhs).foldl
    (fun acc (l, e) => if intSourceExpr? e then l :: acc else acc) []
  { directPins := directPins, natDeps := natDeps, ptrDeps := ptrDeps,
    cmpDeps := cmpDeps, intPins := intPins }

mutual

private partial def commonAssigned (base thenAssigned elseAssigned : Assigned) : Assigned :=
  thenAssigned.filter (fun name => !(base.contains name) && elseAssigned.contains name)

private partial def scanStmts (assigned : Assigned)
    (stmts : List (TSyntax `tritonStmt)) : ScanInfo × Assigned :=
  stmts.foldl
    (fun (acc : ScanInfo × Assigned) st =>
      let (info, nextAssigned) := scanStmt acc.2 st
      (acc.1.append info, nextAssigned))
    ({}, assigned)

private partial def scanStmt (assigned : Assigned)
    (stx : TSyntax `tritonStmt) : ScanInfo × Assigned :=
  match stx with
  | `(tritonStmt| $lhs0:ident, $lhs1:ident $[, $lhsRest:ident]* = $rhs0:tritonExpr, $rhs1:tritonExpr $[, $rhsRest:tritonExpr]*) =>
      let lhsSyntax := #[lhs0, lhs1] ++ lhsRest
      let lhs := lhsSyntax.toList.map (fun i => i.getId.toString)
      let rhs := #[rhs0, rhs1] ++ rhsRest
      let nextAssigned := lhs.foldl (fun acc i => i :: acc) assigned
      (assignmentInfo assigned lhs rhs.toList, nextAssigned)
  | `(tritonStmt| $lhs0:ident, $lhs1:ident $[, $lhsRest:ident]* := $rhs0:tritonExpr, $rhs1:tritonExpr $[, $rhsRest:tritonExpr]*) =>
      let lhsSyntax := #[lhs0, lhs1] ++ lhsRest
      let lhs := lhsSyntax.toList.map (fun i => i.getId.toString)
      let rhs := #[rhs0, rhs1] ++ rhsRest
      let nextAssigned := lhs.foldl (fun acc i => i :: acc) assigned
      (assignmentInfo assigned lhs rhs.toList, nextAssigned)
  | `(tritonStmt| $_i:ident % $e:tritonExpr) =>
      ({ directPins := directPinsFromExpr assigned e
         ptrUses := []
         cmpDeps := cmpDepsFromExpr e }, assigned)
  | `(tritonStmt| $i:ident := $e:tritonExpr) =>
      (assignmentInfo assigned [i.getId.toString] [e], i.getId.toString :: assigned)
  | `(tritonStmt| $i:ident = $e:tritonExpr) =>
      (assignmentInfo assigned [i.getId.toString] [e], i.getId.toString :: assigned)
  | `(tritonStmt| $i:ident += $e:tritonExpr) =>
      (assignmentInfo assigned [i.getId.toString] [e], i.getId.toString :: assigned)
  | `(tritonStmt| $i:ident -= $e:tritonExpr) =>
      (assignmentInfo assigned [i.getId.toString] [e], i.getId.toString :: assigned)
  | `(tritonStmt| $i:ident *= $e:tritonExpr) =>
      (assignmentInfo assigned [i.getId.toString] [e], i.getId.toString :: assigned)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      ({ directPins := pinsFromPtrExpr assigned p ++ directPinsFromExpr assigned v ++
            directPinsFromExpr assigned mask ++ memKwargPins assigned kwargs
         ptrUses := ptrUsesFromPtrExpr assigned p
         cmpDeps := cmpDepsFromExpr v ++ cmpDepsFromExpr mask ++ memKwargCmpDeps kwargs }, assigned)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr, $kw0:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*)) =>
      let kwargs' := #[kw0] ++ kwargs
      ({ directPins := pinsFromPtrExpr assigned p ++ directPinsFromExpr assigned v ++ memKwargPins assigned kwargs'
         ptrUses := ptrUsesFromPtrExpr assigned p
         cmpDeps := cmpDepsFromExpr v ++ memKwargCmpDeps kwargs' }, assigned)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr)) =>
      ({ directPins := pinsFromPtrExpr assigned p ++ directPinsFromExpr assigned v
         ptrUses := ptrUsesFromPtrExpr assigned p
         cmpDeps := cmpDepsFromExpr v }, assigned)
  | `(tritonStmt| tl.atomic_add($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      ({ directPins := pinsFromPtrExpr assigned p ++ directPinsFromExpr assigned v ++ memKwargPins assigned kwargs
         ptrUses := ptrUsesFromPtrExpr assigned p
         cmpDeps := cmpDepsFromExpr v ++ memKwargCmpDeps kwargs }, assigned)
  | `(tritonStmt| tl.atomic_max($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      ({ directPins := pinsFromPtrExpr assigned p ++ directPinsFromExpr assigned v ++ memKwargPins assigned kwargs
         ptrUses := ptrUsesFromPtrExpr assigned p
         cmpDeps := cmpDepsFromExpr v ++ memKwargCmpDeps kwargs }, assigned)
  | `(tritonStmt| tl.atomic_min($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      ({ directPins := pinsFromPtrExpr assigned p ++ directPinsFromExpr assigned v ++ memKwargPins assigned kwargs
         ptrUses := ptrUsesFromPtrExpr assigned p
         cmpDeps := cmpDepsFromExpr v ++ memKwargCmpDeps kwargs }, assigned)
  | `(tritonStmt| tl.atomic_and($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      ({ directPins := pinsFromPtrExpr assigned p ++ directPinsFromExpr assigned v ++ memKwargPins assigned kwargs
         ptrUses := ptrUsesFromPtrExpr assigned p
         cmpDeps := cmpDepsFromExpr v ++ memKwargCmpDeps kwargs }, assigned)
  | `(tritonStmt| tl.atomic_or($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      ({ directPins := pinsFromPtrExpr assigned p ++ directPinsFromExpr assigned v ++ memKwargPins assigned kwargs
         ptrUses := ptrUsesFromPtrExpr assigned p
         cmpDeps := cmpDepsFromExpr v ++ memKwargCmpDeps kwargs }, assigned)
  | `(tritonStmt| tl.atomic_xor($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      ({ directPins := pinsFromPtrExpr assigned p ++ directPinsFromExpr assigned v ++ memKwargPins assigned kwargs
         ptrUses := ptrUsesFromPtrExpr assigned p
         cmpDeps := cmpDepsFromExpr v ++ memKwargCmpDeps kwargs }, assigned)
  | `(tritonStmt| tl.atomic_xchg($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      ({ directPins := pinsFromPtrExpr assigned p ++ directPinsFromExpr assigned v ++ memKwargPins assigned kwargs
         ptrUses := ptrUsesFromPtrExpr assigned p
         cmpDeps := cmpDepsFromExpr v ++ memKwargCmpDeps kwargs }, assigned)
  | `(tritonStmt| tl.atomic_cas($p:tritonExpr, $cmp:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      ({ directPins := pinsFromPtrExpr assigned p ++ directPinsFromExpr assigned cmp ++
          directPinsFromExpr assigned v ++ memKwargPins assigned kwargs
         ptrUses := ptrUsesFromPtrExpr assigned p
         cmpDeps := cmpDepsFromExpr cmp ++ cmpDepsFromExpr v ++ memKwargCmpDeps kwargs }, assigned)
  | `(tritonStmt| tl.async_copy($dst:tritonExpr, $src:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      ({ directPins := pinsFromPtrExpr assigned dst ++ pinsFromPtrExpr assigned src ++
          directPinsFromExpr assigned src ++ memKwargPins assigned kwargs
         ptrUses := ptrUsesFromPtrExpr assigned dst ++ ptrUsesFromPtrExpr assigned src
         cmpDeps := cmpDepsFromExpr src ++ memKwargCmpDeps kwargs }, assigned)
  | `(tritonStmt| tl.async_wait()) => ({}, assigned)
  | `(tritonStmt| tl.debug_barrier()) => ({}, assigned)
  | `(tritonStmt| tl.static_print($_msg:term)) => ({}, assigned)
  | `(tritonStmt| tl.for $i:ident in $($_:term) { $stmts:tritonStmt* }) =>
      let bodyAssigned := i.getId.toString :: assigned
      let (info, _) := scanStmts bodyAssigned stmts.toList
      (info, assigned)
  | `(tritonStmt| tl.for $i:ident in $_:num { $stmts:tritonStmt* }) =>
      let bodyAssigned := i.getId.toString :: assigned
      let (info, _) := scanStmts bodyAssigned stmts.toList
      (info, assigned)
  | `(tritonStmt| for $i:ident in range(0, $($_:term), $($_:term)) { $stmts:tritonStmt* }) =>
      let bodyAssigned := i.getId.toString :: assigned
      let (info, _) := scanStmts bodyAssigned stmts.toList
      (info, assigned)
  | `(tritonStmt| for $i:ident in range($($_:term), $($_:term), $($_:term)) { $stmts:tritonStmt* }) =>
      let bodyAssigned := i.getId.toString :: assigned
      let (info, _) := scanStmts bodyAssigned stmts.toList
      (info, assigned)
  | `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr) { $stmts:tritonStmt* }) =>
      let bodyAssigned := i.getId.toString :: assigned
      let (info, _) := scanStmts bodyAssigned stmts.toList
      let rangeInfo : ScanInfo :=
        { directPins := natExprIdents start ++ natExprIdents stop ++
            directPinsFromExpr assigned start ++ directPinsFromExpr assigned stop
          cmpDeps := cmpDepsFromExpr start ++ cmpDepsFromExpr stop }
      (rangeInfo.append info, assigned)
  | `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr, $step:tritonExpr) { $stmts:tritonStmt* }) =>
      let bodyAssigned := i.getId.toString :: assigned
      let (info, _) := scanStmts bodyAssigned stmts.toList
      let rangeInfo : ScanInfo :=
        { directPins := natExprIdents start ++ natExprIdents stop ++ natExprIdents step ++
            directPinsFromExpr assigned start ++ directPinsFromExpr assigned stop ++
            directPinsFromExpr assigned step
          cmpDeps := cmpDepsFromExpr start ++ cmpDepsFromExpr stop ++ cmpDepsFromExpr step }
      (rangeInfo.append info, assigned)
  | `(tritonStmt| tl.static_range $i:ident in $($_:term) { $stmts:tritonStmt* }) =>
      let bodyAssigned := i.getId.toString :: assigned
      let (info, _) := scanStmts bodyAssigned stmts.toList
      (info, assigned)
  | `(tritonStmt| tl.static_range $i:ident in $_:num { $stmts:tritonStmt* }) =>
      let bodyAssigned := i.getId.toString :: assigned
      let (info, _) := scanStmts bodyAssigned stmts.toList
      (info, assigned)
  | `(tritonStmt| tl.if $cond:tritonExpr { $thenStmts:tritonStmt* } else { $elseStmts:tritonStmt* }) =>
      let (thenInfo, thenAssigned) := scanStmts assigned thenStmts.toList
      let (elseInfo, elseAssigned) := scanStmts assigned elseStmts.toList
      let condInfo : ScanInfo :=
        { directPins := directPinsFromExpr assigned cond
          cmpDeps := cmpDepsFromExpr cond }
      (condInfo.append thenInfo |>.append elseInfo,
        commonAssigned assigned thenAssigned elseAssigned ++ assigned)
  | `(tritonStmt| tl.if $cond:tritonExpr { $stmts:tritonStmt* }) =>
      let (info, _) := scanStmts assigned stmts.toList
      let condInfo : ScanInfo :=
        { directPins := directPinsFromExpr assigned cond
          cmpDeps := cmpDepsFromExpr cond }
      (condInfo.append info, assigned)
  | `(tritonStmt| if $cond:tritonExpr { $thenStmts:tritonStmt* } else { $elseStmts:tritonStmt* }) =>
      let (thenInfo, thenAssigned) := scanStmts assigned thenStmts.toList
      let (elseInfo, elseAssigned) := scanStmts assigned elseStmts.toList
      let condInfo : ScanInfo :=
        { directPins := directPinsFromExpr assigned cond
          cmpDeps := cmpDepsFromExpr cond }
      (condInfo.append thenInfo |>.append elseInfo,
        commonAssigned assigned thenAssigned elseAssigned ++ assigned)
  | `(tritonStmt| if $cond:tritonExpr { $stmts:tritonStmt* }) =>
      let (info, _) := scanStmts assigned stmts.toList
      let condInfo : ScanInfo :=
        { directPins := directPinsFromExpr assigned cond
          cmpDeps := cmpDepsFromExpr cond }
      (condInfo.append info, assigned)
  | _ => ({}, assigned)

end

private def propagateNatDeps (natPins : List String)
    (deps : List (String × List String)) : List String :=
  deps.foldl (fun acc (lhs, rhsIds) =>
    if acc.contains lhs then addManyUnique acc rhsIds else acc) natPins

private def propagateCmpDeps (natPins : List String)
    (deps : List (List String × List String)) : List String :=
  deps.foldl (fun acc (guards, rhsIds) =>
    if anyContained guards acc then addManyUnique acc rhsIds else acc) natPins

private def propagatePtrDeps (ptrUses natPins : List String)
    (deps : List (String × List String × List String)) :
    List String × List String :=
  deps.foldl
    (fun (acc : List String × List String) (lhs, bases, offsets) =>
      if acc.1.contains lhs then
        (addManyUnique acc.1 bases, addManyUnique acc.2 offsets)
      else
        acc)
    (ptrUses, natPins)

partial def closeInfo (info : ScanInfo) (fuel : Nat := 128) :
    List String × List String :=
  let rec loop (fuel : Nat) (ptrUses natPins : List String) :
      List String × List String :=
    match fuel with
    | 0 => (ptrUses, natPins)
    | fuel + 1 =>
        let natPins1 := propagateNatDeps natPins info.natDeps
        let natPins2 := propagateCmpDeps natPins1 info.cmpDeps
        let (ptrUses1, natPins3) := propagatePtrDeps ptrUses natPins2 info.ptrDeps
        if ptrUses1.length == ptrUses.length && natPins3.length == natPins.length then
          (ptrUses1, natPins3)
        else
          loop fuel ptrUses1 natPins3
  let (ptrUses, natPins) := loop fuel
    (info.ptrUses.foldl addUnique [])
    (info.directPins.foldl addUnique [])
  (ptrUses, natPins.filter (fun name => !(info.intPins.contains name)))

/-- Set of identifier names that the body uses in a `.nat`-required
position. The DSL macro consults this when expanding a `name = tl.load(p)`
binding without an explicit `dtype=` kwarg: if `name` is in this set,
the load defaults to `.nat` instead of `.real`. -/
def collectNatPinned (stmts : List (TSyntax `tritonStmt)) : List String :=
  let (info, _) := scanStmts [] stmts
  let (_, pins) := closeInfo info
  pins

/-- Per-region element-dtype declarations gathered from in-body
`region <name> : <dtype>` directives. Used by `expandLoad` /
`expandStore` to default the access dtype from the region's declared
element type, so kernels with non-real buffers don't need an explicit
`dtype=` kwarg on every load. -/
abbrev RegionDTypes := List (String × DInfo)

/-- Pure syntactic mapping from a `tritonDType` token to the macro-time
`DInfo` tag. Mirrors `VeriTile.Triton.DSL.expandDType` but operates
outside `MacroM` so the body scanner can run without monadic plumbing. -/
private def dtypeOfTritonDType (stx : TSyntax `tritonDType) : Option DInfo :=
  match stx with
  | `(tritonDType| tl.float64) => some .real
  | `(tritonDType| tl.float32) => some .fp32
  | `(tritonDType| tl.float16) => some .fp16
  | `(tritonDType| tl.bfloat16) => some .bf16
  | `(tritonDType| tl.int1) => some .bool
  | `(tritonDType| tl.int8) => some .int
  | `(tritonDType| tl.int16) => some .int
  | `(tritonDType| tl.int32) => some .int
  | `(tritonDType| tl.int64) => some .int
  | `(tritonDType| tl.uint8) => some .nat
  | `(tritonDType| tl.uint16) => some .nat
  | `(tritonDType| tl.uint32) => some .nat
  | `(tritonDType| tl.uint64) => some .nat
  | _ => none

mutual

private partial def regionDTypesFromStmts
    (stmts : List (TSyntax `tritonStmt)) : RegionDTypes :=
  stmts.foldl
    (fun acc st => acc ++ regionDTypesFromStmt st)
    []

private partial def regionDTypesFromStmt
    (stx : TSyntax `tritonStmt) : RegionDTypes :=
  match stx with
  | `(tritonStmt| tl.region $i:ident = $dt:tritonDType $[, $iRest:ident = $dtRest:tritonDType]*) =>
      let head : RegionDTypes :=
        match dtypeOfTritonDType dt with
        | some d => [(i.getId.toString, d)]
        | none => []
      let rest : RegionDTypes :=
        (iRest.zip dtRest).foldl (fun acc (j, dt') =>
          match dtypeOfTritonDType dt' with
          | some d => acc ++ [(j.getId.toString, d)]
          | none => acc) []
      head ++ rest
  | `(tritonStmt| tl.for $_:ident in $($_:term) { $stmts:tritonStmt* }) =>
      regionDTypesFromStmts stmts.toList
  | `(tritonStmt| tl.for $_:ident in $_:num { $stmts:tritonStmt* }) =>
      regionDTypesFromStmts stmts.toList
  | `(tritonStmt| tl.static_range $_:ident in $($_:term) { $stmts:tritonStmt* }) =>
      regionDTypesFromStmts stmts.toList
  | `(tritonStmt| tl.static_range $_:ident in $_:num { $stmts:tritonStmt* }) =>
      regionDTypesFromStmts stmts.toList
  | `(tritonStmt| tl.if $_:tritonExpr { $thenStmts:tritonStmt* } else { $elseStmts:tritonStmt* }) =>
      regionDTypesFromStmts thenStmts.toList ++ regionDTypesFromStmts elseStmts.toList
  | `(tritonStmt| tl.if $_:tritonExpr { $stmts:tritonStmt* }) =>
      regionDTypesFromStmts stmts.toList
  | `(tritonStmt| if $_:tritonExpr { $thenStmts:tritonStmt* } else { $elseStmts:tritonStmt* }) =>
      regionDTypesFromStmts thenStmts.toList ++ regionDTypesFromStmts elseStmts.toList
  | `(tritonStmt| if $_:tritonExpr { $stmts:tritonStmt* }) =>
      regionDTypesFromStmts stmts.toList
  | _ => []

end

/-- Collect all `region <name> : <dtype>` directives in a kernel body.
Returns a list of `(region_name, element_dtype)` pairs. The macro
consults this map when defaulting the dtype of `tl.load` / `tl.store`
accesses that target a declared region. -/
def collectRegionDTypes (stmts : List (TSyntax `tritonStmt)) : RegionDTypes :=
  regionDTypesFromStmts stmts

/-- Look up the declared element dtype for a region by name. -/
def lookupRegionDType (decls : RegionDTypes) (name : String) : Option DInfo :=
  match List.find? (fun entry : String × DInfo => entry.1 == name) decls with
  | some entry => some entry.2
  | none => none

/-- Test whether a `tritonStmt` is a region-dtype directive (so the
expansion driver can skip emitting code for it). -/
def isRegionDirective : TSyntax `tritonStmt → Bool := fun stx =>
  match stx with
  | `(tritonStmt| tl.region $_:ident = $_:tritonDType $[, $_:ident = $_:tritonDType]*) =>
      Bool.true
  | _ => Bool.false

/-- Per-binding ptr-element-dtype propagation. When a kernel body
binds `i = R + offset` (or chains through other already-propagated
bindings), `i` denotes a pointer whose element dtype is `R`'s declared
dtype. The DSL macro consults this map at `tl.load(i)` expansion to
default the load's dtype, in addition to the static-region case. -/
abbrev PtrElems := List (String × DInfo)

private def typedRegionTermDType? (r : TSyntax `term) : Option DInfo :=
  match r with
  | `(($_:term : Region .real)) => some DInfo.real
  | `(($_:term : Region .fp32)) => some DInfo.fp32
  | `(($_:term : Region .fp16)) => some DInfo.fp16
  | `(($_:term : Region .bf16)) => some DInfo.bf16
  | `(($_:term : Region .int)) => some DInfo.int
  | `(($_:term : Region .nat)) => some DInfo.nat
  | `(($_:term : Region .bool)) => some DInfo.bool
  | `(($_:term : Region TileDType.real)) => some DInfo.real
  | `(($_:term : Region TileDType.fp32)) => some DInfo.fp32
  | `(($_:term : Region TileDType.fp16)) => some DInfo.fp16
  | `(($_:term : Region TileDType.bf16)) => some DInfo.bf16
  | `(($_:term : Region TileDType.int)) => some DInfo.int
  | `(($_:term : Region TileDType.nat)) => some DInfo.nat
  | `(($_:term : Region TileDType.bool)) => some DInfo.bool
  | _ => none

/-- Walk an expression and try to determine its root static-pointer
region's declared dtype. Returns `some d` only when the expression is
a static-ptr-add chain rooted in a region whose dtype is known via
`regionDTypes` or whose chained-binding name is recorded in `ptrElems`. -/
private partial def rootedPtrDType (regionDTypes : RegionDTypes)
    (ptrElems : PtrElems) :
    TSyntax `tritonExpr → Option DInfo := fun stx =>
  match stx with
  | `(tritonExpr| $($r:term)) => typedRegionTermDType? r
  | `(tritonExpr| ($e:tritonExpr)) => rootedPtrDType regionDTypes ptrElems e
  | `(tritonExpr| $r:ident) =>
      let name := r.getId.toString
      match lookupRegionDType regionDTypes name with
      | some d => some d
      | none =>
          match List.find? (fun entry : String × DInfo => entry.1 == name) ptrElems with
          | some entry => some entry.2
          | none => none
  | `(tritonExpr| $a:tritonExpr + $_:tritonExpr) =>
      rootedPtrDType regionDTypes ptrElems a
  | _ => none

mutual

private partial def ptrElemsFromStmts (regionDTypes : RegionDTypes)
    (acc : PtrElems) (stmts : List (TSyntax `tritonStmt)) : PtrElems :=
  stmts.foldl (fun acc' st => ptrElemsFromStmt regionDTypes acc' st) acc

private partial def ptrElemsFromStmt (regionDTypes : RegionDTypes)
    (acc : PtrElems) (stx : TSyntax `tritonStmt) : PtrElems :=
  match stx with
  | `(tritonStmt| $i:ident := $e:tritonExpr) =>
      match rootedPtrDType regionDTypes acc e with
      | some d => (i.getId.toString, d) :: acc
      | none => acc
  | `(tritonStmt| $i:ident = $e:tritonExpr) =>
      match rootedPtrDType regionDTypes acc e with
      | some d => (i.getId.toString, d) :: acc
      | none => acc
  | `(tritonStmt| tl.for $_:ident in $($_:term) { $stmts:tritonStmt* }) =>
      ptrElemsFromStmts regionDTypes acc stmts.toList
  | `(tritonStmt| tl.for $_:ident in $_:num { $stmts:tritonStmt* }) =>
      ptrElemsFromStmts regionDTypes acc stmts.toList
  | `(tritonStmt| tl.static_range $_:ident in $($_:term) { $stmts:tritonStmt* }) =>
      ptrElemsFromStmts regionDTypes acc stmts.toList
  | `(tritonStmt| tl.static_range $_:ident in $_:num { $stmts:tritonStmt* }) =>
      ptrElemsFromStmts regionDTypes acc stmts.toList
  | `(tritonStmt| tl.if $_:tritonExpr { $thenStmts:tritonStmt* } else { $elseStmts:tritonStmt* }) =>
      ptrElemsFromStmts regionDTypes
        (ptrElemsFromStmts regionDTypes acc thenStmts.toList)
        elseStmts.toList
  | `(tritonStmt| tl.if $_:tritonExpr { $stmts:tritonStmt* }) =>
      ptrElemsFromStmts regionDTypes acc stmts.toList
  | `(tritonStmt| if $_:tritonExpr { $thenStmts:tritonStmt* } else { $elseStmts:tritonStmt* }) =>
      ptrElemsFromStmts regionDTypes
        (ptrElemsFromStmts regionDTypes acc thenStmts.toList)
        elseStmts.toList
  | `(tritonStmt| if $_:tritonExpr { $stmts:tritonStmt* }) =>
      ptrElemsFromStmts regionDTypes acc stmts.toList
  | _ => acc

end

/-- Pre-pass: from the body and the declared region dtypes, propagate
element dtypes to every `name = staticRegion + offset` binding (and
chains thereof). The DSL macro consults the resulting map at
`tl.load(name)` so the chained-pointer case recovers the same dtype as
the direct `tl.load(staticRegion + offset)` case. -/
def collectPtrElems (regionDTypes : RegionDTypes)
    (stmts : List (TSyntax `tritonStmt)) : PtrElems :=
  ptrElemsFromStmts regionDTypes [] stmts

/-- Look up the propagated element dtype for a bound pointer name. -/
def lookupPtrElem (ptrElems : PtrElems) (name : String) : Option DInfo :=
  match List.find? (fun entry : String × DInfo => entry.1 == name) ptrElems with
  | some entry => some entry.2
  | none => none

end VeriTile.Triton.DSL.Inference
