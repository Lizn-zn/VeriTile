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
private partial def staticPtrRegions : TSyntax `tritonExpr → List (TSyntax `term) := fun stx =>
  match stx with
  | `(tritonExpr| $($r:term)) => [r]
  | `(tritonExpr| ($e:tritonExpr)) => staticPtrRegions e
  | `(tritonExpr| $a:tritonExpr + $_b:tritonExpr) =>
      staticPtrRegions a
  | `(tritonExpr| tl.make_block_ptr($p:tritonExpr, $_:ident=$_:tritonExpr,
        $_:ident=[$_:tritonExpr,*], $_:ident=[$_:tritonExpr,*],
        $_:ident=[$_:tritonExpr,*], $_:ident=[$_:tritonExpr,*])) =>
      staticPtrRegions p
  | `(tritonExpr| tl.advance($p:tritonExpr, [$_:tritonExpr,*])) =>
      staticPtrRegions p
  | _ => []

/-- Collect all region terms reachable from a `tritonExpr`. Returns `term`
    syntax - each element is the Lean term inside a `tl.load(...)`
    pointer (recursively in subexpressions). -/
private partial def exprRegions : TSyntax `tritonExpr → List (TSyntax `term) := fun stx =>
  match methodCastExpr? stx with
  | some e => exprRegions e
  | none =>
  match stx with
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonMemKwarg) =>
            match kw with
            | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions val
            | _ => acc) []
      staticPtrRegions p ++ kwargRegions
  | `(tritonExpr| tl.exp($e:tritonExpr))         => exprRegions e
  | `(tritonExpr| tl.exp2($e:tritonExpr))        => exprRegions e
  | `(tritonExpr| tl.log($e:tritonExpr))         => exprRegions e
  | `(tritonExpr| tl.log2($e:tritonExpr))        => exprRegions e
  | `(tritonExpr| tl.sigmoid($e:tritonExpr))     => exprRegions e
  | `(tritonExpr| tl.sqrt($e:tritonExpr))        => exprRegions e
  | `(tritonExpr| tl.tanh($e:tritonExpr))        => exprRegions e
  | `(tritonExpr| tl.sin($e:tritonExpr))         => exprRegions e
  | `(tritonExpr| tl.cos($e:tritonExpr))         => exprRegions e
  | `(tritonExpr| tl.tan($e:tritonExpr))         => exprRegions e
  | `(tritonExpr| tl.atan($e:tritonExpr))        => exprRegions e
  | `(tritonExpr| tl.cosh($e:tritonExpr))        => exprRegions e
  | `(tritonExpr| tl.sinh($e:tritonExpr))        => exprRegions e
  | `(tritonExpr| tl.abs($e:tritonExpr))         => exprRegions e
  | `(tritonExpr| tl.logical_and($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions a ++ exprRegions b
  | `(tritonExpr| tl.logical_or($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions a ++ exprRegions b
  | `(tritonExpr| tl.logical_not($e:tritonExpr)) => exprRegions e
  | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions a ++ exprRegions b
  | `(tritonExpr| tl.max($a:tritonExpr, $b:tritonExpr))   =>
      exprRegions a ++ exprRegions b
  | `(tritonExpr| tl.maximum($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions a ++ exprRegions b
  | `(tritonExpr| tl.minimum($a:tritonExpr, $b:tritonExpr)) =>
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
  | `(tritonExpr| tl.cast($e:tritonExpr, $_:tritonDType)) => exprRegions e
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions a ++ exprRegions b
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr, $c:tritonExpr)) =>
      exprRegions a ++ exprRegions b ++ exprRegions c
  | `(tritonExpr| tl.make_block_ptr($p:tritonExpr, $_:ident=$base:tritonExpr,
        $_:ident=[$_parent:tritonExpr,*], $_:ident=[$_strides:tritonExpr,*],
        $_:ident=[$_offsets:tritonExpr,*], $_:ident=[$_block:tritonExpr,*])) =>
      staticPtrRegions p ++ exprRegions base
  | `(tritonExpr| tl.advance($p:tritonExpr, [$_deltas:tritonExpr,*])) =>
      exprRegions p
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
  | `(tritonExpr| $a:tritonExpr &  $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr ^  $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr |  $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| ~$e:tritonExpr) => exprRegions e
  | `(tritonExpr| $a:tritonExpr << $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr >> $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr +  $b:tritonExpr) =>
      staticPtrRegions stx ++ exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr -  $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr *  $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr /  $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| $a:tritonExpr %  $b:tritonExpr) => exprRegions a ++ exprRegions b
  | `(tritonExpr| tl.where($c:tritonExpr, $a:tritonExpr, $b:tritonExpr)) =>
      exprRegions c ++ exprRegions a ++ exprRegions b
  | `(tritonExpr| $e:tritonExpr[ : , None ])     => exprRegions e
  | `(tritonExpr| $e:tritonExpr[ None , : ])     => exprRegions e
  | `(tritonExpr| tl.expand_dims($e:tritonExpr, $_:tritonReduceKwarg)) => exprRegions e
  | `(tritonExpr| tl.expand_dims($e:tritonExpr, $_:num)) => exprRegions e
  | `(tritonExpr| tl.trans($e:tritonExpr))       => exprRegions e
  | `(tritonExpr| tl.full([$_dims:tritonExpr,*], $v:tritonExpr)) => exprRegions v
  | `(tritonExpr| tl.zeros([$_dims:tritonExpr,*])) => []
  | _ => []
end

/-- Per-statement region split: `(input regions, output regions)`. -/
partial def stmtRegions :
    TSyntax `tritonStmt → List (TSyntax `term) × List (TSyntax `term) := fun stx =>
  match stx with
  | `(tritonStmt| $_:ident := $e:tritonExpr) =>
      (exprRegions e, [])
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonMemKwarg) =>
            match kw with
            | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions val
            | _ => acc) []
      (exprRegions v ++ kwargRegions, staticPtrRegions p)
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
  | `(tritonStmt| tl.static_range $_:ident in $($_:term) { $stmts:tritonStmt* }) =>
      stmts.toList.foldl
        (fun (acc : List (TSyntax `term) × List (TSyntax `term)) st =>
          let (i, o) := stmtRegions st
          (acc.1 ++ i, acc.2 ++ o)) ([], [])
  | `(tritonStmt| tl.static_range $_:ident in $_:num { $stmts:tritonStmt* }) =>
      stmts.toList.foldl
        (fun (acc : List (TSyntax `term) × List (TSyntax `term)) st =>
          let (i, o) := stmtRegions st
          (acc.1 ++ i, acc.2 ++ o)) ([], [])
  | `(tritonStmt| tl.if $cond:tritonExpr { $stmts:tritonStmt* }) =>
      stmts.toList.foldl
        (fun (acc : List (TSyntax `term) × List (TSyntax `term)) st =>
          let (i, o) := stmtRegions st
          (acc.1 ++ i, acc.2 ++ o)) (exprRegions cond, [])
  | _ => ([], [])

end VeriTile.Triton.DSL.Metadata
