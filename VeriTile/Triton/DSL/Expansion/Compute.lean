/-
VeriTile.Triton.DSL.Expansion.Compute

Compute, reduction, and shape-view expansion helpers.
-/

import VeriTile.Triton.DSL.Expansion.Common

open Lean

namespace VeriTile.Triton.DSL

def ensureComputeArithComposable (ctx : String) (e : EOut) : MacroM Unit := do
  match e.computeTerm, e.computeDType? with
  | some _, some .fp32 => pure ()
  | some _, _ =>
      Macro.throwError
        (ctx ++ ": compute-only expressions without a composable fp32 annotation must be assigned or stored directly before algorithm-layer composition")
  | none, _ => pure ()

def fp32ComputeArith? (ctx : String) (algTerm : TSyntax `term) (a b : EOut) :
    MacroM (Option (TSyntax `term) × Option DInfo) := do
  match a.computeDType?, b.computeDType? with
  | none, none => pure (none, none)
  | some .fp32, none | none, some .fp32 | some .fp32, some .fp32 =>
      -- This is annotation tracking only: Algorithm semantics still owns the
      -- arithmetic term; Compute records that the result remains fp32.
      pure (some (← fp32ComputeExpr algTerm), some .fp32)
  | some _, _ | _, some _ =>
      Macro.throwError
        (ctx ++ ": only fp32 compute arithmetic annotations are supported")

def ensureComputeCmpComposable (ctx : String) (e : EOut) : MacroM Unit := do
  match e.computeTerm, e.computeDType? with
  | some _, some .fp32 => pure ()
  | some _, _ =>
      Macro.throwError
        (ctx ++ ": compute-only expressions without a composable fp32 annotation must be assigned or stored directly before algorithm-layer composition")
  | none, _ => pure ()

partial def expandArith (expandExpr : ExprExpander) (env : Env) (ctx : String) (op : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  let a' ←
    if a'.dtype == .nat && b'.dtype == .real then
      match ← expandLeanAntiquoteAs? .real a with
      | some out => pure out
      | none => pure a'
    else
      pure a'
  let b' ←
    if a'.dtype == .real && b'.dtype == .nat then
      match ← expandLeanAntiquoteAs? .real b with
      | some out => pure out
      | none => pure b'
    else
      pure b'
  let (a', b') ← coerceNatIntOperands ctx a' b'
  let (a', b') ← coerceRealArithOperands ctx a' b'
  ensureComputeArithComposable ctx a'
  ensureComputeArithComposable ctx b'
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let np ← a'.dtype.numericProof
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  let algTerm ← `($op $np $bc $a'.term $b'.term)
  let (computeTerm?, computeDType?) ← fp32ComputeArith? ctx algTerm a' b'
  pure ⟨algTerm, a'.dtype, outShape, computeTerm?, computeDType?⟩

partial def expandBoolMaskMul? (expandExpr : ExprExpander) (env : Env)
    (a b : TSyntax `tritonExpr) : MacroM (Option EOut) := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  let mk (cond val : EOut) : MacroM EOut := do
    ensureDType .bool cond.dtype "bool mask multiplication condition"
    ensureComputeArithComposable "bool mask multiplication" val
    ensureDType .real val.dtype "bool mask multiplication value"
    let (_, outShape) ← broadcastTerm cond.shape val.shape "bool mask multiplication"
    let condTerm ← coerceShape cond.term cond.shape outShape "bool mask multiplication condition"
    let valTerm ← coerceShape val.term val.shape outShape "bool mask multiplication value"
    let zeroTerm ← coerceShape (← `(Op.const 0)) SInfo.scalar outShape
      "bool mask multiplication zero"
    let algTerm ← `(Op.where $condTerm $valTerm $zeroTerm)
    let computeTerm? ←
      match val.computeDType? with
      | some .fp32 => pure (some (← fp32ComputeExpr algTerm))
      | some _ =>
          Macro.throwError "bool mask multiplication: only fp32 compute annotations are supported"
      | none => pure none
    pure ⟨algTerm, .real, outShape, computeTerm?, val.computeDType?⟩
  match a'.dtype, b'.dtype with
  | .bool, .real => pure (some (← mk a' b'))
  | .real, .bool => pure (some (← mk b' a'))
  | .bool, _ | _, .bool =>
      Macro.throwError "bool mask multiplication currently supports Bool * Real only"
  | _, _ => pure none

partial def expandIntegralArith (expandExpr : ExprExpander) (env : Env) (ctx : String) (op : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  let (a', b') ← coerceNatIntOperands ctx a' b'
  ensureAlgorithmOnly ctx a'
  ensureAlgorithmOnly ctx b'
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let ip ← a'.dtype.integralProof
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  pure ⟨← `($op $ip $bc $a'.term $b'.term), a'.dtype, outShape, none, none⟩

partial def expandCdiv (expandExpr : ExprExpander) (env : Env)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureAlgorithmOnly "tl.cdiv" a'
  ensureAlgorithmOnly "tl.cdiv" b'
  ensureDType .nat a'.dtype "tl.cdiv lhs"
  ensureDType .nat b'.dtype "tl.cdiv rhs"
  let (addBc, outShape) ← broadcastTerm a'.shape b'.shape "tl.cdiv"
  let (subBc, subShape) ← broadcastTerm outShape SInfo.scalar "tl.cdiv"
  ensureShape outShape subShape "tl.cdiv"
  let (divBc, divShape) ← broadcastTerm outShape b'.shape "tl.cdiv"
  ensureShape outShape divShape "tl.cdiv"
  let sumTerm ← `(Op.add NumericDType.nat $addBc $a'.term $b'.term)
  let numerator ← `(Op.sub NumericDType.nat $subBc $sumTerm (Op.constNat 1))
  pure ⟨← `(Op.div NumericDType.nat $divBc $numerator $b'.term), .nat, outShape, none, none⟩

partial def expandBoolBin (expandExpr : ExprExpander) (env : Env) (ctx : String) (op : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureAlgorithmOnly ctx a'
  ensureAlgorithmOnly ctx b'
  ensureDType .bool a'.dtype ctx
  ensureDType .bool b'.dtype ctx
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  pure ⟨← `($op $bc $a'.term $b'.term), .bool, outShape, none, none⟩

partial def expandNatBitwise (expandExpr : ExprExpander) (env : Env) (ctx : String) (op : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureAlgorithmOnly ctx a'
  ensureAlgorithmOnly ctx b'
  ensureDType .nat a'.dtype ctx
  ensureDType .nat b'.dtype ctx
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  pure ⟨← `($op $bc $a'.term $b'.term), .nat, outShape, none, none⟩

partial def expandBoolOrNatBitwise (expandExpr : ExprExpander) (env : Env) (ctx : String)
    (boolOp natOp : TSyntax `term) (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  ensureAlgorithmOnly ctx a'
  ensureAlgorithmOnly ctx b'
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  match a'.dtype with
  | .bool => pure ⟨← `($boolOp $bc $a'.term $b'.term), .bool, outShape, none, none⟩
  | .nat => pure ⟨← `($natOp $bc $a'.term $b'.term), .nat, outShape, none, none⟩
  | _ =>
      Macro.throwError
        (ctx ++ ": only Bool logical ops and Nat bitwise ops are modeled; signed integer bitwise ops require fixed-width integer semantics")

partial def expandBoolNot (expandExpr : ExprExpander) (env : Env) (ctx : String)
    (a : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  ensureAlgorithmOnly ctx a'
  ensureDType .bool a'.dtype ctx
  pure ⟨← `(Op.boolNot $a'.term), .bool, a'.shape, none, none⟩

partial def expandCmp (expandExpr : ExprExpander) (env : Env) (ctx : String) (op : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  let b' ←
    if a'.dtype == .nat && b'.dtype == .real then
      match b with
      | `(tritonExpr| $n:num) =>
          pure ⟨← `(Op.constNat $n), .nat, SInfo.scalar, none, none⟩
      | _ => pure b'
    else
      pure b'
  let a' ←
    if a'.dtype == .real && b'.dtype == .nat then
      match a with
      | `(tritonExpr| $n:num) =>
          pure ⟨← `(Op.constNat $n), .nat, SInfo.scalar, none, none⟩
      | _ => pure a'
    else
      pure a'
  let b' ←
    if a'.dtype == .real && b'.dtype == .nat then
      match ← expandLeanAntiquoteAs? .real b with
      | some out => pure out
      | none => pure b'
    else
      pure b'
  let a' ←
    if a'.dtype == .nat && b'.dtype == .real then
      match ← expandLeanAntiquoteAs? .real a with
      | some out => pure out
      | none => pure a'
    else
      pure a'
  let (a', b') ← coerceNatIntOperands ctx a' b'
  ensureComputeCmpComposable ctx a'
  ensureComputeCmpComposable ctx b'
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let cp ← a'.dtype.comparableProof
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  pure ⟨← `($op $cp $bc $a'.term $b'.term), .bool, outShape, none, none⟩

partial def expandMinMax (expandExpr : ExprExpander) (env : Env) (ctx : String) (cmp : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  let a' ←
    if b'.dtype == .nat && a'.dtype == .real then
      match a with
      | `(tritonExpr| $n:num) => pure ⟨← `(Op.constNat $n), .nat, SInfo.scalar, none, none⟩
      | _ => pure a'
    else
      pure a'
  let b' ←
    if a'.dtype == .nat && b'.dtype == .real then
      match b with
      | `(tritonExpr| $n:num) => pure ⟨← `(Op.constNat $n), .nat, SInfo.scalar, none, none⟩
      | _ => pure b'
    else
      pure b'
  let a' ←
    if a'.dtype == .nat && b'.dtype == .real then
      match ← expandLeanAntiquoteAs? .real a with
      | some out => pure out
      | none => pure a'
    else
      pure a'
  let b' ←
    if a'.dtype == .real && b'.dtype == .nat then
      match ← expandLeanAntiquoteAs? .real b with
      | some out => pure out
      | none => pure b'
    else
      pure b'
  let (a', b') ← coerceNatIntOperands ctx a' b'
  ensureComputeArithComposable ctx a'
  ensureComputeArithComposable ctx b'
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let cp ← a'.dtype.comparableProof
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  let aTerm ← coerceShape a'.term a'.shape outShape (ctx ++ " lhs")
  let bTerm ← coerceShape b'.term b'.shape outShape (ctx ++ " rhs")
  let algTerm ← `(Op.where ($cmp $cp $bc $a'.term $b'.term) $aTerm $bTerm)
  let (computeTerm?, computeDType?) ← fp32ComputeArith? ctx algTerm a' b'
  pure ⟨algTerm, a'.dtype, outShape, computeTerm?, computeDType?⟩

partial def expandScanOp : TSyntax `tritonScanOp → MacroM (TSyntax `term)
  | `(tritonScanOp| $name:ident) =>
      match name.getId.toString with
      | "sum" => `(ScanOp.sum)
      | "prod" => `(ScanOp.prod)
      | "max" => `(ScanOp.max)
      | "min" => `(ScanOp.min)
      | other =>
          Macro.throwError
            ("tl.associative_scan: unsupported op `" ++ other ++
             "`. Supported ops: sum, prod, max, min.")
  | _ => Macro.throwUnsupported

partial def expandScan (expandExpr : ExprExpander) (env : Env) (ctx : String) (op : TSyntax `term)
    (e : TSyntax `tritonExpr)
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM EOut := do
  let e' ← expandExpr env e
  let eTerm ← realMathTerm ctx e'
  let dims := match e'.shape with | .dims ds => ds
  if dims.isEmpty then
    Macro.throwError (ctx ++ ": rank-≥ 1 input required")
  let mut seenAxis : Bool := Bool.false
  let mut axisIdx : Nat := 0
  let mut dirTerm : TSyntax `term ← `(ScanDirection.forward)
  for kw in kwargs do
    match kw with
    | `(tritonReduceKwarg| reverse=True) =>
        dirTerm ← `(ScanDirection.reverse)
    | `(tritonReduceKwarg| reverse=False) =>
        dirTerm ← `(ScanDirection.forward)
    | `(tritonReduceKwarg| $n:num) =>
        if seenAxis then
          Macro.throwError (ctx ++ ": duplicate `axis=` kwarg")
        seenAxis := Bool.true
        if n.getNat ≥ dims.length then
          Macro.throwError
            (ctx ++ ": axis `" ++ toString n.getNat ++ "` out of bounds for rank "
             ++ toString dims.length)
        axisIdx := n.getNat
    | `(tritonReduceKwarg| axis = $n:num) =>
        if seenAxis then
          Macro.throwError (ctx ++ ": duplicate `axis=` kwarg")
        seenAxis := Bool.true
        if n.getNat ≥ dims.length then
          Macro.throwError
            (ctx ++ ": axis `" ++ toString n.getNat ++ "` out of bounds for rank "
             ++ toString dims.length)
        axisIdx := n.getNat
    | `(tritonReduceKwarg| keep_dims = false) =>
        Macro.throwError (ctx ++ ": `keep_dims` is not meaningful for prefix scans")
    | `(tritonReduceKwarg| keep_dims = true) =>
        Macro.throwError (ctx ++ ": `keep_dims` is not meaningful for prefix scans")
    | `(tritonReduceKwarg| $name:ident = $_) =>
        Macro.throwError
          (ctx ++ ": unsupported kwarg `" ++ name.getId.toString ++
           "`. Only `axis = N` and `reverse=True/False` are supported.")
    | _ =>
        Macro.throwUnsupported
  let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
  pure ⟨← `(Op.scan $op (⟨$axisLit, by simp⟩) $dirTerm $eTerm), .real, e'.shape, none, none⟩

partial def parseAxisOnlyKwargs (ctx : String) (dims : List (TSyntax `term))
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM Nat := do
  if dims.isEmpty then
    Macro.throwError (ctx ++ ": rank-≥ 1 input required")
  let mut seenAxis : Bool := Bool.false
  let mut axisIdx : Nat := 0
  for kw in kwargs do
    match kw with
    | `(tritonReduceKwarg| $n:num) =>
        if seenAxis then
          Macro.throwError (ctx ++ ": duplicate `axis=` kwarg")
        seenAxis := Bool.true
        if n.getNat ≥ dims.length then
          Macro.throwError
            (ctx ++ ": axis `" ++ toString n.getNat ++ "` out of bounds for rank "
             ++ toString dims.length)
        axisIdx := n.getNat
    | `(tritonReduceKwarg| axis = $n:num) =>
        if seenAxis then
          Macro.throwError (ctx ++ ": duplicate `axis=` kwarg")
        seenAxis := Bool.true
        if n.getNat ≥ dims.length then
          Macro.throwError
            (ctx ++ ": axis `" ++ toString n.getNat ++ "` out of bounds for rank "
             ++ toString dims.length)
        axisIdx := n.getNat
    | `(tritonReduceKwarg| keep_dims = false) =>
        Macro.throwError (ctx ++ ": `keep_dims` is not supported")
    | `(tritonReduceKwarg| keep_dims = true) =>
        Macro.throwError (ctx ++ ": `keep_dims` is not supported")
    | `(tritonReduceKwarg| $name:ident = $_) =>
        Macro.throwError
          (ctx ++ ": unsupported kwarg `" ++ name.getId.toString ++
           "`. Only `axis = N` is supported.")
    | _ =>
        Macro.throwUnsupported
  pure axisIdx

partial def expandArgReduce (expandExpr : ExprExpander) (env : Env) (ctx : String) (op : TSyntax `term)
    (e : TSyntax `tritonExpr)
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM EOut := do
  let e' ← expandExpr env e
  let eTerm ← realMathTerm ctx e'
  let dims := match e'.shape with | .dims ds => ds
  let axisIdx ← parseAxisOnlyKwargs ctx dims kwargs
  let outDims ← eraseNth dims axisIdx
  let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
  pure ⟨← `($op (⟨$axisLit, by simp⟩) $eTerm), .nat, .dims outDims, none, none⟩

partial def expandSort (expandExpr : ExprExpander) (env : Env)
    (e : TSyntax `tritonExpr)
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM EOut := do
  let e' ← expandExpr env e
  let eTerm ← realMathTerm "tl.sort" e'
  let dims := match e'.shape with | .dims ds => ds
  let axisIdx ← parseAxisOnlyKwargs "tl.sort" dims kwargs
  let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
  pure ⟨← `(Op.sort (⟨$axisLit, by simp⟩) $eTerm), .real, e'.shape, none, none⟩

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
partial def expandReduce (expandExpr : ExprExpander) (env : Env) (ctx : String) (op : TSyntax `term)
    (e : TSyntax `tritonExpr)
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM EOut := do
  let e' ← expandExpr env e
  let eTerm ← realMathTerm ctx e'
  let dims := match e'.shape with | .dims ds => ds
  if dims.isEmpty then
    Macro.throwError (ctx ++ ": reduction expects a tile, got scalar")
  let mut seenAxis : Bool := Bool.false
  let mut seenKeepDims : Bool := Bool.false
  let mut keepDims : Bool := Bool.false
  let mut axis? : Option Nat := none
  for kw in kwargs do
    match kw with
    | `(tritonReduceKwarg| $n:num) =>
        if seenAxis then
          Macro.throwError (ctx ++ ": duplicate `axis=` kwarg")
        seenAxis := Bool.true
        if n.getNat ≥ dims.length then
          Macro.throwError
            (ctx ++ ": axis `" ++ toString n.getNat ++ "` out of bounds for rank "
             ++ toString dims.length)
        axis? := some n.getNat
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
    | `(tritonReduceKwarg| return_indices=True) =>
        Macro.throwError (ctx ++ ": `return_indices=True` is only supported through tuple binding")
    | `(tritonReduceKwarg| return_indices=False) =>
        pure ()
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
      pure ⟨← `($op (⟨$axisLit, by simp⟩) $kdLit $eTerm),
            .real, SInfo.dims outDims, none, none⟩
  | none =>
      -- `axis = None` (Triton default): reduce over all dimensions.
      -- Two regimes, because `keep_dims` changes how the rank evolves:
      --   * `keep_dims = false`: each call drops the last remaining axis. This
      --     still reduces every dimension, and keeps the intermediate shapes
      --     syntactically reducible (`[M, N] -> [M] -> []`) so the generated
      --     `Fin shape.length` proof closes by `simp`.
      --   * `keep_dims = true`: rank is preserved; emit calls with
      --     `axis = 0, 1, …, dims.length - 1` so every axis is reduced.
      let mut term := eTerm
      for j in [:dims.length] do
        let axisIdx := if keepDims then j else (dims.length - 1 - j)
        let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
        term ← `($op (⟨$axisLit, by simp [TileShape.eraseAxis, TileShape.reduceShape]⟩) $kdLit $term)
      let outDims : List (TSyntax `term) ←
        if keepDims then
          let oneLit : TSyntax `term ← `((1 : Nat))
          pure (List.replicate dims.length oneLit)
        else
          pure []
      pure ⟨term, .real, SInfo.dims outDims, none, none⟩

/-- Lower `tl.max(...)` reductions. Real-valued tiles keep the existing
mathematical-real path; Nat tiles lower to `Op.reduceMaxNat`, which is needed
for index arithmetic such as `tl.max(tl.where(nat_cond, nat_values, 0))`. -/
partial def expandReduceMax (expandExpr : ExprExpander) (env : Env)
    (e : TSyntax `tritonExpr)
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM EOut := do
  let e' ← expandExpr env e
  let mut eTerm := e'.term
  let mut outDType : DInfo := .real
  let mut isNat : Bool := Bool.false
  match e'.dtype with
  | .nat =>
      outDType := .nat
      isNat := Bool.true
  | .real =>
      pure ()
  | .fp32 | .fp16 | .bf16 =>
      eTerm ← realMathTerm "tl.max" e'
  | _ =>
      ensureDType .real e'.dtype "tl.max"
  let dims := match e'.shape with | .dims ds => ds
  if dims.isEmpty then
    Macro.throwError "tl.max: reduction expects a tile, got scalar"
  let mut seenAxis : Bool := Bool.false
  let mut seenKeepDims : Bool := Bool.false
  let mut keepDims : Bool := Bool.false
  let mut axis? : Option Nat := none
  for kw in kwargs do
    match kw with
    | `(tritonReduceKwarg| $n:num) =>
        if seenAxis then
          Macro.throwError "tl.max: duplicate `axis=` kwarg"
        seenAxis := Bool.true
        if n.getNat ≥ dims.length then
          Macro.throwError
            ("tl.max: axis `" ++ toString n.getNat ++ "` out of bounds for rank "
             ++ toString dims.length)
        axis? := some n.getNat
    | `(tritonReduceKwarg| axis = $n:num) =>
        if seenAxis then
          Macro.throwError "tl.max: duplicate `axis=` kwarg"
        seenAxis := Bool.true
        if n.getNat ≥ dims.length then
          Macro.throwError
            ("tl.max: axis `" ++ toString n.getNat ++ "` out of bounds for rank "
             ++ toString dims.length)
        axis? := some n.getNat
    | `(tritonReduceKwarg| keep_dims = false) =>
        if seenKeepDims then
          Macro.throwError "tl.max: duplicate `keep_dims=` kwarg"
        seenKeepDims := Bool.true
    | `(tritonReduceKwarg| keep_dims = true) =>
        if seenKeepDims then
          Macro.throwError "tl.max: duplicate `keep_dims=` kwarg"
        seenKeepDims := Bool.true
        keepDims := Bool.true
    | `(tritonReduceKwarg| return_indices=True) =>
        Macro.throwError "tl.max: `return_indices=True` is only supported through tuple binding"
    | `(tritonReduceKwarg| return_indices=False) =>
        pure ()
    | `(tritonReduceKwarg| $name:ident = $_) =>
        let nm := name.getId.toString
        if nm == "axis" || nm == "keep_dims" then
          Macro.throwError
            ("tl.max: `" ++ nm ++ "=` value is not a recognized literal")
        else
          Macro.throwError
            ("tl.max: unknown kwarg `" ++ nm ++
             "`. Only `axis = N` and `keep_dims = true|false` are supported.")
    | _ => Macro.throwUnsupported
  let kdLit : TSyntax `term ← if keepDims then `(Bool.true) else `(Bool.false)
  match axis? with
  | some axisIdx =>
      let outDims ←
        if keepDims then setNthOne dims axisIdx else eraseNth dims axisIdx
      let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
      let term ←
        if isNat then
          `(Op.reduceMaxNat (⟨$axisLit, by simp⟩) $kdLit $eTerm)
        else
          `(Op.reduceMax (⟨$axisLit, by simp⟩) $kdLit $eTerm)
      pure ⟨term, outDType, SInfo.dims outDims, none, none⟩
  | none =>
      let mut term := eTerm
      for j in [:dims.length] do
        let axisIdx := if keepDims then j else (dims.length - 1 - j)
        let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
        term ←
          if isNat then
            `(Op.reduceMaxNat
                (⟨$axisLit, by simp [TileShape.eraseAxis, TileShape.reduceShape]⟩)
                $kdLit $term)
          else
            `(Op.reduceMax
                (⟨$axisLit, by simp [TileShape.eraseAxis, TileShape.reduceShape]⟩)
                $kdLit $term)
      let outDims : List (TSyntax `term) ←
        if keepDims then
          let oneLit : TSyntax `term ← `((1 : Nat))
          pure (List.replicate dims.length oneLit)
        else
          pure []
      pure ⟨term, outDType, SInfo.dims outDims, none, none⟩

/-- Lower a `tl.dot(a, b)` to `Op.dot a b`.

Algorithm-layer dot is real-valued. If the surface operands were explicitly
cast to fp32/fp16/bf16, first project those float tiles back to real so the
cast remains visible in the operand term before the dot. -/
partial def expandDot (expandExpr : ExprExpander) (env : Env)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a0 ← expandExpr env a
  let b0 ← expandExpr env b
  let a' ←
    match a0.dtype with
    | .real => pure a0
    | .fp32 | .fp16 | .bf16 | .floatVar _ =>
        pure { a0 with term := ← realMathTerm "tl.dot lhs" a0, dtype := .real }
    | _ =>
        ensureDType .real a0.dtype "tl.dot"
        pure a0
  let b' ←
    match b0.dtype with
    | .real => pure b0
    | .fp32 | .fp16 | .bf16 | .floatVar _ =>
        pure { b0 with term := ← realMathTerm "tl.dot rhs" b0, dtype := .real }
    | _ =>
        ensureDType .real b0.dtype "tl.dot"
        pure b0
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
        .real, .dims (aBatch ++ [aM, bN]), none, none⟩

partial def expandShapeDims (ctx : String) (dims : Array (TSyntax `tritonExpr)) :
    MacroM (TSyntax `term × SInfo) := do
  let mut dimTerms : Array (TSyntax `term) := #[]
  for d in dims do
    dimTerms := dimTerms.push (← natDimTerm (ctx ++ " dims") d)
  let rec shapeTerm : List (TSyntax `term) → MacroM (TSyntax `term)
    | [] => `(([] : TileShape))
    | d :: rest => do
        let tail ← shapeTerm rest
        `($d :: $tail)
  let shape ← shapeTerm dimTerms.toList
  pure (shape, .dims dimTerms.toList)

/-- Lower `tl.full([dims*], value)` to `Op.full`. The DSL value is
restricted to a real / nat scalar (its dtype propagates to the resulting
tile). The shape list may be any rank including empty (which degenerates
to the scalar value itself, matching `Op.full shape (Op.const v) = v`
for `shape = []`). -/
partial def expandFull (expandExpr : ExprExpander) (env : Env)
    (dims : Array (TSyntax `tritonExpr)) (v : TSyntax `tritonExpr)
    (dtypeHint : Option DInfo := none) :
    MacroM EOut := do
  let targetDType := dtypeHint.getD .real
  let v'' ←
    match dtypeHint with
    | some .nat =>
        match v with
        | `(tritonExpr| $n:num) =>
            pure ⟨← `(Op.constNat $n), .nat, SInfo.scalar, none, none⟩
        | _ =>
            match ← expandLeanAntiquoteAs? targetDType v with
            | some out => pure out
            | none => expandExpr env v
    | some .int =>
        match v with
        | `(tritonExpr| $n:num) =>
            let t : TSyntax `term := ⟨Syntax.mkNumLit (toString n.getNat)⟩
            pure ⟨← `(Op.constInt ($t : Int)), .int, SInfo.scalar, none, none⟩
        | _ =>
            match ← expandLeanAntiquoteAs? targetDType v with
            | some out => pure out
            | none => expandExpr env v
    | _ =>
        match ← expandLeanAntiquoteAs? targetDType v with
        | some out => pure out
        | none => expandExpr env v
  let v' ←
    if let some hint := dtypeHint then
      if v''.dtype == .real && hint != .real then
        let srcProof ← DInfo.floatProof .real
        let dstProof ← hint.floatProof
        pure ⟨← `(Op.castFloat $srcProof $dstProof $v''.term), hint, v''.shape, none, none⟩
      else pure v''
    else pure v''
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
  -- Wrap every dim in `(_ : Nat)` so its `termKey` matches what
  -- `tl.arange(0, end)` emits (which also wraps); without this,
  -- `tl.full([\$(M)], …)` would not broadcast against `[M]`-shaped
  -- tiles produced by arange.
  let (shape, shapeInfo) ← expandShapeDims "tl.full" dims
  pure ⟨← `(Op.full $shape $v'.term), v'.dtype, shapeInfo, none, none⟩

partial def expandComputeFull (expandExpr : ExprExpander) (env : Env)
    (dims : Array (TSyntax `tritonExpr)) (v : TSyntax `tritonExpr)
    (dt : TSyntax `tritonDType) : MacroM EOut := do
  if (← expandDType dt) != .fp32 then
    return ← expandFull expandExpr env dims v (dtypeHint := some (← expandDType dt))
  let value ←
    match ← expandLeanAntiquoteAs? .real v with
    | some out => pure out
    | none => expandExpr env v
  match value.shape with
  | .dims [] => pure ()
  | _ => Macro.throwError "tl.full: value must be a scalar (rank-0)"
  let cdtype ← expandComputeDType dt
  unless cdtype == .fp32 do
    Macro.throwError "tl.full: compute-preserving full currently supports `dtype=tl.float32`"
  let (shape, shapeInfo) ← expandShapeDims "tl.full" dims
  let cdt ← cdtype.term
  let algDType := cdtype.algDType
  let algFull ← `(Op.full $shape $value.term)
  let computeFull ←
    `(ComputeExpr.compute
        (ComputeOp.full $shape
          (ComputeOp.alg $cdt $value.term)))
  pure ⟨algFull, algDType, shapeInfo, some computeFull, some .fp32⟩

partial def expandComputeZeros (expandExpr : ExprExpander) (env : Env)
    (dims : Array (TSyntax `tritonExpr)) (dt : TSyntax `tritonDType) :
    MacroM EOut := do
  let zero ← `(tritonExpr| 0)
  expandComputeFull expandExpr env dims zero dt

/-- Lower `tl.trans(e)` to `Op.transpose`. Rank-≥ 2 required; the
trailing two dims are swapped, any leading dims pass through as the
`batch` prefix. The input shape's `SInfo.dims` already lists
outermost-first, so picking off the last two and rebuilding
`batch ++ [N, M]` is straightforward. -/
partial def expandTranspose (expandExpr : ExprExpander) (env : Env)
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
        e'.dtype, .dims (batch ++ [N, M]), none, none⟩

partial def shapeTermOfDims (dims : List (TSyntax `term)) : MacroM (TSyntax `term) := do
  (SInfo.dims dims).term

partial def termNatLit? (t : TSyntax `term) : Option Nat :=
  match t with
  | `($n:num) => some n.getNat
  | `(($e:term)) => termNatLit? e
  | `(($e:term : $_:term)) => termNatLit? e
  | _ => none

partial def literalProduct? (dims : List (TSyntax `term)) : Option Nat :=
  dims.foldl
    (fun acc d => do
      let p ← acc
      let n ← termNatLit? d
      some (p * n))
    (some 1)

partial def coordProjTerm (idx : TSyntax `term) (pos : Nat) :
    MacroM (TSyntax `term) := do
  let mut cursor := idx
  for _ in [:pos] do
    cursor ← `(($cursor).2)
  `(($cursor).1)

partial def indexTupleTerm : List (TSyntax `term) → MacroM (TSyntax `term)
  | [] => `(PUnit.unit)
  | c :: rest => do
      let tail ← indexTupleTerm rest
      `(($c, $tail))

partial def validatePermutation (rank : Nat) (axes : List Nat) (ctx : String) :
    MacroM Unit := do
  unless axes.length == rank do
    Macro.throwError
      (ctx ++ ": permutation length " ++ toString axes.length ++
       " does not match rank " ++ toString rank)
  for ax in axes do
    if ax ≥ rank then
      Macro.throwError
        (ctx ++ ": axis `" ++ toString ax ++ "` out of bounds for rank " ++
         toString rank)
  for i in [:rank] do
    unless axes.count i == 1 do
      Macro.throwError
        (ctx ++ ": axes must be a permutation of 0.." ++ toString (rank - 1))

partial def expandPermute (expandExpr : ExprExpander) (env : Env)
    (e : TSyntax `tritonExpr) (axesStx : List (TSyntax `num)) :
    MacroM EOut := do
  let e' ← expandExpr env e
  let dims := match e'.shape with | .dims ds => ds
  let axes := axesStx.map (fun n => n.getNat)
  validatePermutation dims.length axes "tl.permute"
  let outDims := axes.map (fun ax => dims[ax]!)
  let outShape ← shapeTermOfDims outDims
  let idx : TSyntax `term ← `(idx)
  let mut inputCoords : List (TSyntax `term) := []
  for inputAxis in [:dims.length] do
    let outPos := axes.idxOf inputAxis
    if outPos == axes.length then
      Macro.throwError "internal error: validated permutation lost an axis"
    inputCoords := inputCoords ++ [← coordProjTerm idx outPos]
  let mapBody ← indexTupleTerm inputCoords
  pure ⟨← `(Op.remap $outShape (fun idx => $mapBody) $e'.term),
        e'.dtype, .dims outDims, none, none⟩

partial def expandFlip (expandExpr : ExprExpander) (env : Env)
    (e : TSyntax `tritonExpr) (axisIdx : Nat) : MacroM EOut := do
  let e' ← expandExpr env e
  let dims := match e'.shape with | .dims ds => ds
  if dims.isEmpty then
    Macro.throwError "tl.flip: rank-≥ 1 input required"
  if axisIdx ≥ dims.length then
    Macro.throwError
      ("tl.flip: axis `" ++ toString axisIdx ++ "` out of bounds for rank " ++
       toString dims.length)
  let shape ← e'.shape.term
  let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
  pure ⟨← `(Op.remap $shape
              (fun idx => TileShape.flipIndex $shape (⟨$axisLit, by simp⟩) idx)
              $e'.term),
        e'.dtype, e'.shape, none, none⟩

partial def parseFlipKwargs (kwargs : TSyntaxArray `tritonReduceKwarg) :
    MacroM Nat := do
  let mut axis? : Option Nat := none
  for kw in kwargs do
    match kw with
    | `(tritonReduceKwarg| axis = $n:num) =>
        if axis?.isSome then
          Macro.throwError "tl.flip: duplicate axis/dim kwarg"
        axis? := some n.getNat
    | `(tritonReduceKwarg| $name:ident = $val:tritonExpr) =>
        let nm := name.getId.toString
        unless nm == "dim" do
          Macro.throwError
            ("tl.flip: unsupported kwarg `" ++ nm ++ "`. Only `dim = N` / `axis = N` is supported.")
        if axis?.isSome then
          Macro.throwError "tl.flip: duplicate axis/dim kwarg"
        match val with
        | `(tritonExpr| $n:num) => axis? := some n.getNat
        | _ => Macro.throwError "tl.flip: `dim=` must be a numeric literal"
    | `(tritonReduceKwarg| keep_dims = false) =>
        Macro.throwError "tl.flip: `keep_dims` is not supported"
    | `(tritonReduceKwarg| keep_dims = true) =>
        Macro.throwError "tl.flip: `keep_dims` is not supported"
    | _ => Macro.throwUnsupported
  match axis? with
  | some ax => pure ax
  | none => Macro.throwError "tl.flip: explicit `dim = N` / `axis = N` is required"

partial def expandReshapeLike (expandExpr : ExprExpander) (env : Env) (ctx : String)
    (e : TSyntax `tritonExpr) (dims : Array (TSyntax `tritonExpr)) :
    MacroM EOut := do
  let e' ← expandExpr env e
  let (outShape, outDims) ← natListTerm ctx dims
  let inDims := match e'.shape with | .dims ds => ds
  match literalProduct? inDims, literalProduct? outDims with
  | some inProduct, some outProduct =>
      unless inProduct == outProduct do
        Macro.throwError
          (ctx ++ ": literal element-count mismatch (" ++ toString inProduct ++
           " input elements vs " ++ toString outProduct ++ " output elements)")
  | _, _ => pure ()
  pure ⟨← `(Op.reshape $outShape $e'.term), e'.dtype, .dims outDims, none, none⟩

partial def expandRavel (expandExpr : ExprExpander) (env : Env)
    (e : TSyntax `tritonExpr) : MacroM EOut := do
  let e' ← expandExpr env e
  let dims := match e'.shape with | .dims ds => ds
  let product ←
    match dims with
    | [] => `((1 : Nat))
    | d :: rest => do
        let mut acc := d
        for next in rest do
          acc ← `($acc * $next)
        pure acc
  let outShape ← shapeTermOfDims [product]
  pure ⟨← `(Op.reshape $outShape $e'.term), e'.dtype, .dims [product], none, none⟩

partial def expandJoin (expandExpr : ExprExpander) (env : Env)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  unless a'.dtype == b'.dtype do
    Macro.throwError "tl.join: dtype mismatch"
  ensureShape a'.shape b'.shape "tl.join"
  let dims := match a'.shape with | .dims ds => ds
  let two : TSyntax `term ← `((2 : Nat))
  pure ⟨← `(Op.join $a'.term $b'.term), a'.dtype, .dims (dims ++ [two]), none, none⟩

partial def expandSplit (expandExpr : ExprExpander) (env : Env)
    (e : TSyntax `tritonExpr) (side : Nat) : MacroM EOut := do
  let e' ← expandExpr env e
  unless side == 0 || side == 1 do
    Macro.throwError "tl.split: side must be literal 0 or 1"
  let dims := match e'.shape with | .dims ds => ds
  if dims.isEmpty then
    Macro.throwError "tl.split: rank-≥ 1 input required"
  let lastDim := dims[dims.length - 1]!
  unless termNatLit? lastDim == some 2 do
    Macro.throwError "tl.split: final dimension must be the literal 2"
  let outDims := dims.dropLast
  let outShape ← shapeTermOfDims outDims
  let sideLit : TSyntax `num := ⟨Syntax.mkNumLit (toString side)⟩
  pure ⟨← `(Op.split (shape := $outShape) (⟨$sideLit, by decide⟩) $e'.term),
        e'.dtype, .dims outDims, none, none⟩

/-- Lower `tl.expand_dims(e, axis=N)` / `tl.expand_dims(e, N)` to
`Op.expandDim`. The axis must be a literal in `[0, rank]`; dimensions are
typed by inserting a unit axis into the macro-tracked shape. -/
partial def expandExpandDims (expandExpr : ExprExpander) (env : Env)
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
  pure ⟨← `(Op.expandDim (⟨$axisLit, by simp [TileShape.eraseAxis, TileShape.setAxisOne]⟩) $e'.term),
        e'.dtype, .dims outDims, none, none⟩

/-- Lower `e[:, None]` / `e[None, :]` to `Op.expandDim` with the appropriate
axis. TritonBench-G contains scalar guard masks written as `cond[:, None]`;
for rank-0 inputs we treat that spelling as inserting a single unit axis so
the scalar guard can still broadcast against a vector mask. For higher-rank
general insertion use `tl.expand_dims(e, axis=N)`. -/
partial def expandSlicerNone (expandExpr : ExprExpander) (env : Env)
    (e : TSyntax `tritonExpr) (axisIdx : Nat) : MacroM EOut := do
  let e' ← expandExpr env e
  let dims := match e'.shape with | .dims ds => ds
  if dims.length == 0 then
    return ← expandExpandDims expandExpr env e (axisIdx := 0)
  if dims.length != 1 then
    Macro.throwError
      ("[:, None] / [None, :]: rank-1 input required, got rank " ++
       toString dims.length ++
       ". Use `tl.expand_dims(e, axis=N)` for general-rank insertion.")
  expandExpandDims expandExpr env e (axisIdx := axisIdx)

end VeriTile.Triton.DSL
