/-
VeriTile.Triton.DSL.Metadata

Syntax-level metadata collection for `triton { ... }` blocks.
-/

import VeriTile.Triton.DSL.Syntax

open Lean

namespace VeriTile.Triton.DSL.Metadata

private def methodCastExpr? (stx : TSyntax `tritonExpr) : Option (TSyntax `tritonExpr) :=
  let k := stx.raw.getKind
  if k == ``tritonMethodCast then
    let args := stx.raw.getArgs
    if h : args.size = 5 then
      some ⟨args[0]⟩
    else if h : args.size = 6 then
      some ⟨args[0]⟩
    else
      none
  else
    none

mutual

/-- Collect region terms from statically visible pointer expressions. -/
private partial def staticPtrRegions (assigned : List String) :
    TSyntax `tritonExpr → List (TSyntax `term) := fun stx =>
  match stx with
  | `(tritonExpr| $($r:term)) => [r]
  | `(tritonExpr| $r:ident) =>
      if assigned.contains r.getId.toString then
        []
      else
        [⟨r.raw⟩]
  | `(tritonExpr| ($e:tritonExpr)) => staticPtrRegions assigned e
  | `(tritonExpr| $a:tritonExpr + $_b:tritonExpr) =>
      staticPtrRegions assigned a
  | `(tritonExpr| tl.make_block_ptr($p:tritonExpr, $_:ident=$_:tritonExpr,
        $_:ident=[$_:tritonExpr,*], $_:ident=[$_:tritonExpr,*],
        $_:ident=[$_:tritonExpr,*], $_:ident=[$_:tritonExpr,*])) =>
      staticPtrRegions assigned p
  | `(tritonExpr| tl.advance($p:tritonExpr, [$_:tritonExpr,*])) =>
      staticPtrRegions assigned p
  | _ => []

/-- Collect all region terms reachable from a `tritonExpr`. Returns `term`
    syntax - each element is the Lean term inside a `tl.load(...)`
    pointer (recursively in subexpressions). -/
private partial def exprRegions (assigned : List String) :
    TSyntax `tritonExpr → List (TSyntax `term) := fun stx =>
  match methodCastExpr? stx with
  | some e => exprRegions assigned e
  | none =>
  match stx with
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonMemKwarg) =>
            match kw with
            | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      staticPtrRegions assigned p ++ exprRegions assigned mask ++ kwargRegions
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonMemKwarg) =>
            match kw with
            | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      staticPtrRegions assigned p ++ kwargRegions
  | `(tritonExpr| tl.atomic_xchg($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonMemKwarg) =>
            match kw with
            | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned v ++ kwargRegions ++ staticPtrRegions assigned p
  | `(tritonExpr| tl.atomic_cas($p:tritonExpr, $cmp:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonMemKwarg) =>
            match kw with
            | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned cmp ++ exprRegions assigned v ++ kwargRegions ++ staticPtrRegions assigned p
  | `(tritonExpr| tl.exp($e:tritonExpr))         => exprRegions assigned e
  | `(tritonExpr| tl.exp2($e:tritonExpr))        => exprRegions assigned e
  | `(tritonExpr| tl.math.exp2($e:tritonExpr))   => exprRegions assigned e
  | `(tritonExpr| tl.extra.cuda.libdevice.pow($e:tritonExpr, $_:num)) => exprRegions assigned e
  | `(tritonExpr| tl.log($e:tritonExpr))         => exprRegions assigned e
  | `(tritonExpr| tl.log2($e:tritonExpr))        => exprRegions assigned e
  | `(tritonExpr| tl.sigmoid($e:tritonExpr))     => exprRegions assigned e
  | `(tritonExpr| tl.sqrt($e:tritonExpr))        => exprRegions assigned e
  | `(tritonExpr| tl.math.rsqrt($e:tritonExpr)) => exprRegions assigned e
  | `(tritonExpr| tl.rsqrt($e:tritonExpr))      => exprRegions assigned e
  | `(tritonExpr| tl.tanh($e:tritonExpr))        => exprRegions assigned e
  | `(tritonExpr| tanh($e:tritonExpr))           => exprRegions assigned e
  | `(tritonExpr| tl.sin($e:tritonExpr))         => exprRegions assigned e
  | `(tritonExpr| tl.math.sin($e:tritonExpr))    => exprRegions assigned e
  | `(tritonExpr| tl.cos($e:tritonExpr))         => exprRegions assigned e
  | `(tritonExpr| tl.tan($e:tritonExpr))         => exprRegions assigned e
  | `(tritonExpr| tl.atan($e:tritonExpr))        => exprRegions assigned e
  | `(tritonExpr| tl.cosh($e:tritonExpr))        => exprRegions assigned e
  | `(tritonExpr| tl.sinh($e:tritonExpr))        => exprRegions assigned e
  | `(tritonExpr| tl.erf($e:tritonExpr))         => exprRegions assigned e
  | `(tritonExpr| tl.extra.cuda.libdevice.erf($e:tritonExpr)) => exprRegions assigned e
  | `(tritonExpr| tl.abs($e:tritonExpr))         => exprRegions assigned e
  | `(tritonExpr| tl.logical_and($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.logical_or($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.logical_not($e:tritonExpr)) => exprRegions assigned e
  | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.max($a:tritonExpr, $b:tritonExpr))   =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.maximum($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.minimum($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.cumsum($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.cumprod($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.associative_scan($e:tritonExpr, $_:tritonScanOp $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.argmax($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.argmin($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.sort($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.sum($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.max($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.toReal($e:tritonExpr))      => exprRegions assigned e
  | `(tritonExpr| tl.cast($e:tritonExpr, $_:tritonDType)) => exprRegions assigned e
  | `(tritonExpr| tl.multiple_of($e:tritonExpr, $align:tritonExpr)) =>
      exprRegions assigned e ++ exprRegions assigned align
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr, $c:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b ++ exprRegions assigned c
  | `(tritonExpr| tl.make_block_ptr($p:tritonExpr, $_:ident=$base:tritonExpr,
        $_:ident=[$_parent:tritonExpr,*], $_:ident=[$_strides:tritonExpr,*],
        $_:ident=[$_offsets:tritonExpr,*], $_:ident=[$_block:tritonExpr,*])) =>
      staticPtrRegions assigned p ++ exprRegions assigned base
  | `(tritonExpr| tl.make_block_ptr($_:ident=$p:tritonExpr,
        $_:ident=($_parent:tritonExpr,*), $_:ident=($_strides:tritonExpr,*),
        $_:ident=($_offsets:tritonExpr,*), $_:ident=($_block:tritonExpr,*))) =>
      staticPtrRegions assigned p
  | `(tritonExpr| tl.make_block_ptr($_:ident=$p:tritonExpr,
        $_:ident=($_parent:tritonExpr,*), $_:ident=($_strides:tritonExpr,*),
        $_:ident=($_offsets:tritonExpr,*), $_:ident=($_block:tritonExpr,*),
        $_:ident=($_order:num,*))) =>
      staticPtrRegions assigned p
  | `(tritonExpr| tl.advance($p:tritonExpr, [$_deltas:tritonExpr,*])) =>
      exprRegions assigned p
  | `(tritonExpr| tl.advance($p:tritonExpr, $_:ident=($_deltas:tritonExpr,*))) =>
      exprRegions assigned p
  | `(tritonExpr| tl.permute($e:tritonExpr, [$_:num,*])) =>
      exprRegions assigned e
  | `(tritonExpr| tl.reshape($e:tritonExpr, [$dims:tritonExpr,*])) =>
      dims.getElems.foldl (fun acc d => acc ++ exprRegions assigned d) (exprRegions assigned e)
  | `(tritonExpr| tl.view($e:tritonExpr, [$dims:tritonExpr,*])) =>
      dims.getElems.foldl (fun acc d => acc ++ exprRegions assigned d) (exprRegions assigned e)
  | `(tritonExpr| tl.ravel($e:tritonExpr)) =>
      exprRegions assigned e
  | `(tritonExpr| tl.flip($e:tritonExpr, $_:num)) =>
      exprRegions assigned e
  | `(tritonExpr| tl.flip($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.join($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.split($e:tritonExpr, $_:num)) =>
      exprRegions assigned e
  | `(tritonExpr| ($e:tritonExpr))               => exprRegions assigned e
  | `(tritonExpr| tl.program_id($e:tritonExpr))  => exprRegions assigned e
  | `(tritonExpr| tl.arange($e:tritonExpr))      => exprRegions assigned e
  | `(tritonExpr| tl.arange($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr <  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr <= $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr == $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr >  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr >= $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr != $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr &  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr and $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr ^  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr |  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| ~$e:tritonExpr) => exprRegions assigned e
  | `(tritonExpr| -$e:tritonExpr) => exprRegions assigned e
  | `(tritonExpr| $a:tritonExpr << $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr >> $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr +  $b:tritonExpr) =>
      staticPtrRegions assigned stx ++ exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr -  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr *  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr /  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr %  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.where($c:tritonExpr, $a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned c ++ exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $e:tritonExpr[ : , None ])     => exprRegions assigned e
  | `(tritonExpr| $e:tritonExpr[ None , : ])     => exprRegions assigned e
  | `(tritonExpr| tl.expand_dims($e:tritonExpr, $_:tritonReduceKwarg)) => exprRegions assigned e
  | `(tritonExpr| tl.expand_dims($e:tritonExpr, $_:num)) => exprRegions assigned e
  | `(tritonExpr| tl.trans($e:tritonExpr))       => exprRegions assigned e
  | `(tritonExpr| tl.broadcast($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.full([$_dims:tritonExpr,*], $v:tritonExpr)) => exprRegions assigned v
  | `(tritonExpr| tl.full([$_dims:tritonExpr,*], $v:tritonExpr, $_name:ident=$_dt:tritonDType)) =>
      exprRegions assigned v
  | `(tritonExpr| tl.full([$_dims:tritonExpr,*], $_dtypeName:ident=$_dt:tritonDType,
      $_valueName:ident=$v:tritonExpr)) =>
      exprRegions assigned v
  | `(tritonExpr| tl.full([$_dims:tritonExpr,*], $_valueName:ident=$v:tritonExpr,
      $_dtypeName:ident=$_dt:tritonDType)) =>
      exprRegions assigned v
  | `(tritonExpr| tl.zeros([$_dims:tritonExpr,*])) => []
  | `(tritonExpr| tl.zeros([$_dims:tritonExpr,*], $_name:ident=$_dt:tritonDType)) => []
  | _ => []
end

private def memKwargRegions (assigned : List String)
    (kwargs : TSyntaxArray `tritonMemKwarg) : List (TSyntax `term) :=
  kwargs.foldl
    (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonMemKwarg) =>
      match kw with
      | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
      | _ => acc) []

mutual

private partial def commonAssigned (base thenAssigned elseAssigned : List String) : List String :=
  thenAssigned.filter (fun name => !(base.contains name) && elseAssigned.contains name)

private partial def blockRegionsWith (assigned : List String)
    (stmts : List (TSyntax `tritonStmt)) :
    List (TSyntax `term) × List (TSyntax `term) × List String :=
  stmts.foldl
    (fun (acc : List (TSyntax `term) × List (TSyntax `term) × List String) st =>
      let (i, o, nextAssigned) := stmtRegionsWith acc.2.2 st
      (acc.1 ++ i, acc.2.1 ++ o, nextAssigned))
    ([], [], assigned)

/-- Per-statement region split plus the assigned-register context after the statement. -/
private partial def stmtRegionsWith (assigned : List String)
    (stx : TSyntax `tritonStmt) :
    List (TSyntax `term) × List (TSyntax `term) × List String :=
 match stx with
  | `(tritonStmt| $lhs0:ident, $lhs1:ident $[, $lhsRest:ident]* = $rhs0:tritonExpr, $rhs1:tritonExpr $[, $rhsRest:tritonExpr]*) =>
      let lhs := #[lhs0, lhs1] ++ lhsRest
      let rhs := #[rhs0, rhs1] ++ rhsRest
      let ins := rhs.foldl (fun acc e => acc ++ exprRegions assigned e) []
      let nextAssigned := lhs.foldl (fun acc i => i.getId.toString :: acc) assigned
      (ins, [], nextAssigned)
  | `(tritonStmt| $lhs0:ident, $lhs1:ident $[, $lhsRest:ident]* := $rhs0:tritonExpr, $rhs1:tritonExpr $[, $rhsRest:tritonExpr]*) =>
      let lhs := #[lhs0, lhs1] ++ lhsRest
      let rhs := #[rhs0, rhs1] ++ rhsRest
      let ins := rhs.foldl (fun acc e => acc ++ exprRegions assigned e) []
      let nextAssigned := lhs.foldl (fun acc i => i.getId.toString :: acc) assigned
      (ins, [], nextAssigned)
  | `(tritonStmt| $i:ident := $e:tritonExpr) =>
      (exprRegions assigned e, [], i.getId.toString :: assigned)
  | `(tritonStmt| $i:ident = $e:tritonExpr) =>
      (exprRegions assigned e, [], i.getId.toString :: assigned)
  | `(tritonStmt| $i:ident += $e:tritonExpr) =>
      (exprRegions assigned e, [], i.getId.toString :: assigned)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned v ++ exprRegions assigned mask ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr, $kw0:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*)) =>
      let kwargs' := #[kw0] ++ kwargs
      (exprRegions assigned v ++ memKwargRegions assigned kwargs',
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr)) =>
      (exprRegions assigned v,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.atomic_add($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned v ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.atomic_max($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned v ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.atomic_min($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned v ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.atomic_and($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned v ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.atomic_or($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned v ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.atomic_xor($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned v ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.atomic_xchg($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned v ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.atomic_cas($p:tritonExpr, $cmp:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned cmp ++ exprRegions assigned v ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.async_copy($dst:tritonExpr, $src:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (staticPtrRegions assigned src ++ exprRegions assigned src ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned dst, assigned)
  | `(tritonStmt| tl.async_wait()) =>
      ([], [], assigned)
  | `(tritonStmt| tl.debug_barrier()) =>
      ([], [], assigned)
  | `(tritonStmt| tl.for $idx:ident in $($_:term) { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (idx.getId.toString :: assigned) stmts.toList
      (i, o, assigned)
  | `(tritonStmt| tl.for $idx:ident in $_:num { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (idx.getId.toString :: assigned) stmts.toList
      (i, o, assigned)
  | `(tritonStmt| for $idx:ident in range(0, $($_:term), $($_:term)) { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (idx.getId.toString :: assigned) stmts.toList
      (i, o, assigned)
  | `(tritonStmt| for $idx:ident in range($($_:term), $($_:term), $($_:term)) { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (idx.getId.toString :: assigned) stmts.toList
      (i, o, assigned)
  | `(tritonStmt| for $idx:ident in range($start:tritonExpr, $stop:tritonExpr, $step:tritonExpr) { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (idx.getId.toString :: assigned) stmts.toList
      (exprRegions assigned start ++ exprRegions assigned stop ++ exprRegions assigned step ++ i, o, assigned)
  | `(tritonStmt| tl.static_range $idx:ident in $($_:term) { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (idx.getId.toString :: assigned) stmts.toList
      (i, o, assigned)
  | `(tritonStmt| tl.static_range $idx:ident in $_:num { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (idx.getId.toString :: assigned) stmts.toList
      (i, o, assigned)
  | `(tritonStmt| tl.if $cond:tritonExpr { $thenStmts:tritonStmt* } else { $elseStmts:tritonStmt* }) =>
      let (thenIns, thenOuts, thenAssigned) := blockRegionsWith assigned thenStmts.toList
      let (elseIns, elseOuts, elseAssigned) := blockRegionsWith assigned elseStmts.toList
      (exprRegions assigned cond ++ thenIns ++ elseIns, thenOuts ++ elseOuts,
        commonAssigned assigned thenAssigned elseAssigned ++ assigned)
  | `(tritonStmt| tl.if $cond:tritonExpr { $stmts:tritonStmt* }) =>
      let (ins, outs, _) := blockRegionsWith assigned stmts.toList
      (exprRegions assigned cond ++ ins, outs, assigned)
  | _ => ([], [], assigned)

end

/-- Per-statement region split: `(input regions, output regions)`. -/
partial def stmtRegions (stx : TSyntax `tritonStmt) :
    List (TSyntax `term) × List (TSyntax `term) :=
  let (i, o, _) := stmtRegionsWith [] stx
  (i, o)

/-- Per-block region split, tracking assigned registers so bare pointer
identifiers can be distinguished from pointer registers. -/
partial def blockRegions (stmts : List (TSyntax `tritonStmt)) :
    List (TSyntax `term) × List (TSyntax `term) :=
  let (i, o, _) := blockRegionsWith [] stmts
  (i, o)

end VeriTile.Triton.DSL.Metadata
