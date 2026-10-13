/-
VeriTile.Triton.DSL.Expansion.Compute

Compute, reduction, and shape-view expansion helpers.
-/

import VeriTile.Triton.DSL.Expansion.Common

open Lean

namespace VeriTile.Triton.DSL

def ensureComputeArithComposable (ctx : String) (e : EOut) : MacroM Unit := do
  match e.computeTerm, e.computeDType? with
  | some _, some .fp32 | some _, some .real => pure ()
  | some ce, _ =>
      match ce with
      | `(ComputeExpr.alg $_) => pure ()
      | _ => Macro.throwError (ctx ++ ": compute-only expressions without a composable floating annotation must be assigned or stored directly before algorithm-layer composition")
  | none, _ => pure ()

private partial def replaceFloatingOperands (a aa b bb raw : Syntax) : Syntax :=
  if raw == a then aa else if raw == b then bb else
  match raw with
  | .node info kind args => .node info kind (args.map (replaceFloatingOperands a aa b bb))
  | other => other

def floatingComputeArith? (ctx : String) (algTerm : TSyntax `term) (a b : EOut) :
    MacroM (Option (TSyntax `term) × Option DInfo) := do
  let precision := if a.computeDType? == some .real || b.computeDType? == some .real
    then some DInfo.real else a.computeDType?.orElse (fun _ => b.computeDType?)
  if precision.isNone && a.computeTerm.isNone && b.computeTerm.isNone then
    return (none, none)
  let operand (e : EOut) : MacroM (TSyntax `term) := do
    match e.dtype with
    | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 | .floatVar _ => e.scopedFloatingTerm
    | _ => pure e.term
  let aa ← operand a
  let bb ← operand b
  let raw := replaceFloatingOperands a.term.raw aa.raw b.term.raw bb.raw algTerm.raw
  let fp : TSyntax `term := ⟨raw⟩
  match precision with
  | some .fp32 => return (some (← fp32ComputeExpr fp), some .fp32)
  | some .real => return (some (← fp64ComputeExpr fp), some .real)
  | none => return (some (← `(ComputeExpr.alg $fp)), none)
  | _ => Macro.throwError (ctx ++ ": unsupported floating compute precision")

def ensureComputeCmpComposable (ctx : String) (e : EOut) : MacroM Unit := do
  match e.computeTerm, e.computeDType? with
  | some _, some .fp32 | some _, some .real => pure ()
  | some ce, _ =>
      match ce with
      | `(ComputeExpr.alg $_) => pure ()
      | _ => Macro.throwError (ctx ++ ": compute-only expressions without a composable floating annotation must be assigned or stored directly before algorithm-layer composition")
  | none, _ => pure ()

-- Recognize source literals rather than inspecting an expanded subtraction:
-- runtime negation and explicit casts must retain their own execution steps.
private partial def floatingLiteralValue? (e : TSyntax `tritonExpr) :
    MacroM (Option (TSyntax `term)) := do
  match e with
  | `(tritonExpr| $n:num) => return some (← `(($n : ℝ)))
  | `(tritonExpr| $n:scientific) => return some (← `(($n : ℝ)))
  | `(tritonExpr| ($inner:tritonExpr)) => floatingLiteralValue? inner
  | `(tritonExpr| -$inner:tritonExpr) =>
      match ← floatingLiteralValue? inner with
      | some value => return some (← `(-$value))
      | none => return none
  | _ => return none

private def foldTypedFloatingLiteral (other : EOut) (source : TSyntax `tritonExpr)
    (e : EOut) : MacroM EOut := do
  let typed := other.computeDType?.isSome || match other.dtype with
    | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => Bool.true
    | _ => Bool.false
  unless typed do return e
  match ← floatingLiteralValue? source with
  | some value => return ⟨← `(Op.const $value), .real, SInfo.scalar, none, none⟩
  | none => return e

/-- A constexpr floating scalar does not promote a half/bfloat16 tensor.
Only literal syntax is retargeted: a loaded scalar or an explicit cast remains
an independently typed operand. Division promotes half operands to fp32. -/
private def scalarLiteralAt (dtype : DInfo) (e : EOut) : MacroM EOut := do
  if e.computeDType?.isSome then return e
  unless dtype == .fp32 || dtype == .fp16 || dtype == .bf16 ||
      dtype == .f8e4 || dtype == .f8e5 do return e
  match e.term with
  | `(Op.const $value) =>
      let proof ← dtype.floatProof
      return { e with term := ← `(Op.constFloat $proof $value), dtype := dtype }
  | _ => return e

partial def expandArith (expandExpr : ExprExpander) (env : Env) (ctx : String) (op : TSyntax `term)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  let a' ← foldTypedFloatingLiteral b' a a'
  let b' ← foldTypedFloatingLiteral a' b b'
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
  let isDivision := match op with | `(Op.div) => Bool.true | _ => Bool.false
  let (a', b') ← if isDivision then pure (a', b') else do
    let aa ← scalarLiteralAt b'.dtype a'
    let bb ← scalarLiteralAt a'.dtype b'
    pure (aa, bb)
  let (a', b') ← coerceNatIntOperands ctx a' b'
  let (a', b') ←
    -- Select division's common type from the original operands. Generic
    -- promotion first would round bf16 through fp16 for mixed half inputs.
    -- Real/fp32 operands already dominate through the ordinary promotion path.
    if isDivision &&
        (a'.dtype == .fp16 || a'.dtype == .bf16 ||
          b'.dtype == .fp16 || b'.dtype == .bf16) &&
        a'.dtype != .real && b'.dtype != .real &&
        a'.dtype != .fp32 && b'.dtype != .fp32 then
      let aa ← coerceFloatOperand .real a'
      let bb ← coerceFloatOperand .real b'
      pure ({ aa with computeDType? := some .fp32 },
        { bb with computeDType? := some .fp32 })
    else coerceRealArithOperands ctx a' b'
  ensureComputeArithComposable ctx a'
  ensureComputeArithComposable ctx b'
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let np ← a'.dtype.numericProof
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  let algTerm ← `($op $np $bc $a'.term $b'.term)
  let (computeTerm?, computeDType?) ← floatingComputeArith? ctx algTerm a' b'
  pure ⟨algTerm, a'.dtype, outShape, computeTerm?, computeDType?⟩

/-- `tl.extra.cuda.libdevice.pow(base, exponent)`: element-wise real power,
lowering to `Op.pow` (`Real.rpow` semantics). Operands are coerced onto the
ℝ channel like the other real math calls (`$(...)` antiquotes reinterpreted
as ℝ, `Nat` lifted via `Op.natToReal`, floats widened via `Op.castFloat`);
shapes broadcast like binary arithmetic, which covers the common
scalar-base × tensor-exponent form (e.g. RoPE's `pow(theta, freqs)`). -/
partial def expandPow (expandExpr : ExprExpander) (env : Env) (ctx : String)
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
  let (a', b') ← coerceRealArithOperands ctx a' b'
  ensureComputeArithComposable ctx a'
  ensureComputeArithComposable ctx b'
  let aTerm ← realMathTerm (ctx ++ " base") a'
  let bTerm ← realMathTerm (ctx ++ " exponent") b'
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  let algTerm ← `(Op.pow $bc $aTerm $bTerm)
  let (computeTerm?, computeDType?) ← floatingComputeArith? ctx algTerm a' b'
  pure ⟨algTerm, .real, outShape, computeTerm?, computeDType?⟩

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
      | some .real => pure (some (← fp64ComputeExpr algTerm))
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
  let aa ← scalarLiteralAt b'.dtype a'
  let bb ← scalarLiteralAt a'.dtype b'
  let (a', b') ← coerceNatIntOperands ctx aa bb
  let (a', b') ← coerceRealArithOperands ctx a' b'
  ensureComputeArithComposable ctx a'
  ensureComputeArithComposable ctx b'
  unless a'.dtype == b'.dtype do
    Macro.throwError (ctx ++ ": dtype mismatch")
  let cp ← a'.dtype.comparableProof
  let (bc, outShape) ← broadcastTerm a'.shape b'.shape ctx
  let aTerm ← coerceShape a'.term a'.shape outShape (ctx ++ " lhs")
  let bTerm ← coerceShape b'.term b'.shape outShape (ctx ++ " rhs")
  let algTerm ← `(Op.where ($cmp $cp $bc $a'.term $b'.term) $aTerm $bTerm)
  let (computeTerm?, computeDType?) ← floatingComputeArith? ctx algTerm a' b'
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
        Macro.throwError (ctx ++ ": unsupported kwarg `" ++ name.getId.toString ++
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
        Macro.throwError (ctx ++ ": unsupported kwarg `" ++ name.getId.toString ++
           "`. Only `axis = N` is supported.")
    | _ =>
        Macro.throwUnsupported
  pure axisIdx

partial def expandArgReduce (expandExpr : ExprExpander) (env : Env) (ctx : String) (op : TSyntax `term)
    (e : TSyntax `tritonExpr)
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM EOut := do
  let original ← expandExpr env e
  if ctx != "tl.argmax" then
    let eTerm ← realMathTerm ctx original
    let dims := match original.shape with | .dims ds => ds
    let axisIdx ← parseAxisOnlyKwargs ctx dims kwargs
    let outDims ← eraseNth dims axisIdx
    let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
    return ⟨← `($op (⟨$axisLit, by simp⟩) $eTerm), .nat, .dims outDims, none, none⟩
  let mut axisKwargs : TSyntaxArray `tritonReduceKwarg := #[]
  for kw in kwargs do
    match kw with
    | `(tritonReduceKwarg| $name:ident = $value:tritonExpr) =>
        if name.getId.toString == "tie_break_left" then
          match value with
          | `(tritonExpr| True) => pure ()
          | _ => Macro.throwError "tl.argmax: only literal True leftmost tie breaking is supported"
        else axisKwargs := axisKwargs.push kw
    | _ => axisKwargs := axisKwargs.push kw
  let e' ← if original.dtype == .bf16 then do
      let term ← realMathTerm ctx original
      pure (⟨term, .real, original.shape, some (← fp32ComputeExpr term), some .fp32⟩ : EOut)
    else pure original
  let dims := match e'.shape with | .dims ds => ds
  let axisIdx ← parseAxisOnlyKwargs ctx dims axisKwargs
  let outDims ← eraseNth dims axisIdx
  let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
  let term ← match e'.dtype with
    | .real => match e'.computeDType? with
      | some .fp32 => `(Op.argMaxTyped FloatDType.real (⟨$axisLit, by simp⟩) $e'.term (some IndexComputePrecision.fp32))
      | some .real => `(Op.argMaxTyped FloatDType.real (⟨$axisLit, by simp⟩) $e'.term (some IndexComputePrecision.fp64))
      | _ => `(Op.argMax (⟨$axisLit, by simp⟩) $e'.term)
    | .fp32 | .fp16 | .f8e4 | .f8e5 => do
        let dtype ← e'.dtype.floatProof
        `(Op.argMaxTyped $dtype (⟨$axisLit, by simp⟩) $e'.term)
    | _ => Macro.throwError "tl.argmax: unsupported or unspecialized storage dtype"
  pure ⟨term, .nat, .dims outDims, none, none⟩

partial def expandSort (expandExpr : ExprExpander) (env : Env)
    (e : TSyntax `tritonExpr)
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM EOut := do
  let e' ← expandExpr env e
  let eTerm ← realMathTerm "tl.sort" e'
  let dims := match e'.shape with | .dims ds => ds
  let axisIdx ← parseAxisOnlyKwargs "tl.sort" dims kwargs
  let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
  pure ⟨← `(Op.sort (⟨$axisLit, by simp⟩) $eTerm), .real, e'.shape, none, none⟩

/-- Lower floating reductions without changing their numerical operation.
A floating sum retains its input dtype unless an explicit `dtype` is supplied.
The source frontend flattens a whole tile before one reduction when `axis` is
omitted; nested axis reductions would change the opaque FP operation. Rank-one
inputs and explicit-axis reductions need no flattening. -/
partial def expandReduce (expandExpr : ExprExpander) (env : Env) (ctx : String) (op : TSyntax `term)
    (e : TSyntax `tritonExpr)
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM EOut := do
  let original ← expandExpr env e
  let dims := match original.shape with | .dims ds => ds
  if dims.isEmpty then
    Macro.throwError (ctx ++ ": reduction expects a tile, got scalar")
  let isSum := ctx == "tl.sum"
  let mut seenAxis : Bool := Bool.false
  let mut seenKeepDims : Bool := Bool.false
  let mut seenDType : Bool := Bool.false
  let mut keepDims : Bool := Bool.false
  let mut axis? : Option Nat := none
  let mut dtype? : Option DInfo := none
  for kw in kwargs do
    match kw with
    | `(tritonReduceKwarg| axis=None) =>
        if seenAxis then Macro.throwError (ctx ++ ": duplicate `axis=` kwarg")
        seenAxis := Bool.true
    | `(tritonReduceKwarg| $n:num)
    | `(tritonReduceKwarg| axis = $n:num) =>
        if seenAxis then Macro.throwError (ctx ++ ": duplicate `axis=` kwarg")
        seenAxis := Bool.true
        if n.getNat ≥ dims.length then
          Macro.throwError
            (ctx ++ ": axis `" ++ toString n.getNat ++ "` out of bounds for rank "
             ++ toString dims.length)
        axis? := some n.getNat
    | `(tritonReduceKwarg| keep_dims = false) =>
        if seenKeepDims then Macro.throwError (ctx ++ ": duplicate `keep_dims=` kwarg")
        seenKeepDims := Bool.true
    | `(tritonReduceKwarg| keep_dims = true) =>
        if seenKeepDims then Macro.throwError (ctx ++ ": duplicate `keep_dims=` kwarg")
        seenKeepDims := Bool.true
        keepDims := Bool.true
    | `(tritonReduceKwarg| return_indices=True) =>
        Macro.throwError (ctx ++ ": `return_indices=True` is only supported through tuple binding")
    | `(tritonReduceKwarg| return_indices=False) => pure ()
    | `(tritonReduceKwarg| $name:ident=$dt:tritonDType) =>
        unless isSum && name.getId == `dtype do
          Macro.throwError (ctx ++ ": unsupported dtype kwarg `" ++ name.getId.toString ++ "`")
        if seenDType then Macro.throwError (ctx ++ ": duplicate `dtype=` kwarg")
        seenDType := Bool.true
        dtype? := some (← expandDType dt)
    | `(tritonReduceKwarg| $name:ident = $value:tritonExpr) =>
        let nm := name.getId.toString
        if nm == "dtype" && isSum then
          if seenDType then Macro.throwError (ctx ++ ": duplicate `dtype=` kwarg")
          seenDType := Bool.true
          match value with
          | `(tritonExpr| None) => pure ()
          | _ => dtype? := some (← expandDTypeExpr value)
        else if nm == "axis" then
          if seenAxis then Macro.throwError (ctx ++ ": duplicate `axis=` kwarg")
          seenAxis := Bool.true
          match value with
          | `(tritonExpr| None) => pure ()
          | _ => Macro.throwError (ctx ++ ": `axis=` must be a numeric literal or None")
        else
          Macro.throwError (ctx ++ ": unknown or unsupported kwarg `" ++ nm ++ "`")
    | _ => Macro.throwUnsupported
  let e' ← match dtype? with
    | none => pure original
    | some dtype => do
        -- A `.real` carrier with an fp32/fp64 annotation already has that
        -- computation dtype. Do not insert a fictitious same-dtype cast.
        if original.dtype == .real && original.computeDType? == some dtype then
          pure original
        else
          let casted ← coerceFloatOperand dtype original
          if dtype == .real then
            pure { casted with
              computeTerm := some (← fp64ComputeExpr casted.term)
              computeDType? := some .real }
          else pure { casted with computeTerm := none, computeDType? := none }
  let eTerm ← if isSum then
      -- Check that this is a floating reduction without erasing its channel.
      let _ ← e'.dtype.floatProof
      pure e'.term
    else realMathTerm ctx e'
  let resultDType := if isSum then e'.dtype else DInfo.real
  let reducer ← if isSum && e'.dtype != .real then do
      let dtype ← e'.dtype.floatProof
      `(Op.reduceSumTyped $dtype)
    else pure op
  let finish (term : TSyntax `term) (outDims : List (TSyntax `term)) : MacroM EOut := do
    if resultDType == .real then
      let (computeTerm?, computeDType?) ← floatingComputeArith? ctx term e' e'
      pure ⟨term, resultDType, SInfo.dims outDims, computeTerm?, computeDType?⟩
    else
      pure ⟨term, resultDType, SInfo.dims outDims, none, none⟩
  let kdLit : TSyntax `term ← if keepDims then `(Bool.true) else `(Bool.false)
  match axis? with
  | some axisIdx =>
      let outDims ← if keepDims then setNthOne dims axisIdx else eraseNth dims axisIdx
      let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
      finish (← `($reducer (⟨$axisLit, by simp⟩) $kdLit $eTerm)) outDims
  | none =>
      if dims.length == 1 then
        let outDims ← if keepDims then setNthOne dims 0 else pure []
        finish (← `($reducer (⟨0, by simp⟩) $kdLit $eTerm)) outDims
      else
        let mut product := dims.head!
        for dim in dims.tail do
          product ← `($product * $dim)
        let flatShape ← (SInfo.dims [product]).term
        let term ← `($reducer (⟨0, by simp⟩) Bool.false (Op.reshape $flatShape $eTerm))
        if keepDims then
          let oneLit : TSyntax `term ← `((1 : Nat))
          let outDims := List.replicate dims.length oneLit
          let outShape ← (SInfo.dims outDims).term
          finish (← `(Op.reshape $outShape $term)) outDims
        else finish term []

/-- Lower `tl.max(...)` reductions. Natural and signed integer tiles retain
their exact discrete carrier; floating tiles retain their computation precision.
In particular, a signed metadata maximum is never clipped to natural zero. -/
partial def expandReduceMax (expandExpr : ExprExpander) (env : Env)
    (e : TSyntax `tritonExpr)
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM EOut := do
  let e' ← expandExpr env e
  let mut eTerm := e'.term
  let mut outDType : DInfo := .real
  let mut isNat : Bool := Bool.false
  let mut isInt : Bool := Bool.false
  let mut promotedToFP32 := Bool.false
  match e'.dtype with
  | .nat =>
      outDType := .nat
      isNat := Bool.true
  | .int =>
      outDType := .int
      isInt := Bool.true
  | .real =>
      pure ()
  | .fp32 =>
      eTerm ← realMathTerm "tl.max" e'
  | .fp16 | .bf16 | .f8e4 | .f8e5 =>
      -- Triton v3.5.1 standard.max promotes sub-32-bit floating inputs
      -- before a value-only reduction. Preserve the conversion and its fp32
      -- profile, so subsequent arithmetic also sees the promoted result.
      eTerm ← realMathTerm "tl.max" e'
      promotedToFP32 := Bool.true
  | .floatVar _ =>
      Macro.throwError "tl.max: specialize the floating storage dtype before reduction"
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
        else if isInt then
          `(Op.reduceMaxInt (⟨$axisLit, by simp⟩) $kdLit $eTerm)
        else
          `(Op.reduceMax (⟨$axisLit, by simp⟩) $kdLit $eTerm)
      let (ct, cp) ←
        if isNat || isInt then pure (none, none)
        else if promotedToFP32 then
          pure (some (← fp32ComputeExpr term), some DInfo.fp32)
        else floatingComputeArith? "tl.max" term e' e'
      pure ⟨term, outDType, SInfo.dims outDims, ct, cp⟩
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
          else if isInt then
            `(Op.reduceMaxInt
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
      let (ct, cp) ←
        if isNat || isInt then pure (none, none)
        else if promotedToFP32 then
          pure (some (← fp32ComputeExpr term), some DInfo.fp32)
        else floatingComputeArith? "tl.max" term e' e'
      pure ⟨term, outDType, SInfo.dims outDims, ct, cp⟩

/-- Lower a `tl.dot(a, b)` to `Op.dot a b` (or `Op.dotInt` on the `.int`
channel).

Algorithm-typed operands retain the real-channel dot. Explicit floating
operands/results use `Op.dotTyped`, so the original input channels and the
selected output type survive nested arithmetic and stores. Integer operands
use `Op.dotInt`. Mixed integer/floating operands are rejected. -/
partial def expandDot (expandExpr : ExprExpander) (env : Env)
    (a b : TSyntax `tritonExpr) (options : Option (TSyntax `term) := none)
    (resultHint : Option DInfo := none) : MacroM EOut := do
  let a0 := (← expandExpr env a).forgetStorageWidth
  let b0 := (← expandExpr env b).forgetStorageWidth
  if (a0.dtype == .int) != (b0.dtype == .int) then
    Macro.throwError
      ("tl.dot: mixed operand dtypes — both operands must be int "
        ++ "(integer tl.dot, `Op.dotInt`) or both float (`Op.dot`); "
        ++ "cast the odd side first (nat values: `tl.cast(x, tl.int32)`)")
  let outDType : DInfo := if a0.dtype == .int then .int else .real
  -- Typed dot consumes its operands directly, without artificial widening casts.
  let a' := a0
  let b' := b0
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
  let optionsT ← match options with
    | some term => pure term
    | none => `(({} : DotOptions))
  if outDType == .int then
    let opTerm ← `(Op.dotInt (options := $optionsT) (batch := $batchT)
      (M := $aM) (K := $aK) (N := $bN) $a'.term $b'.term)
    return ⟨opTerm, .int, .dims (aBatch ++ [aM, bN]), none, none⟩
  let typed := a'.dtype != .real || b'.dtype != .real ||
    (resultHint.isSome && resultHint != some .fp32 && resultHint != some .real)
  if !typed then
    let opTerm ← `(Op.dot (options := $optionsT) (batch := $batchT)
      (M := $aM) (K := $aK) (N := $bN) $a'.term $b'.term)
    return ⟨opTerm, .real, .dims (aBatch ++ [aM, bN]), none, none⟩
  -- Triton's fp32 and bf16 operands produce fp32; fp16/fp8 operands use the
  -- selected result type, defaulting to fp32. Result dtype is part of the dot.
  if resultHint == some .bf16 then
    Macro.throwError "tl.dot: bfloat16 output requires a float32/float16 dot followed by a cast"
  let out := if a'.dtype == .fp32 || a'.dtype == .bf16 || a'.computeDType? == some .fp32 then DInfo.fp32
    else resultHint.getD .fp32
  if out != .fp32 && out != .fp16 then
    Macro.throwError "tl.dot: typed floating results require float16 or float32"
  let leftProof ← a'.dtype.floatProof
  let rightProof ← b'.dtype.floatProof
  let outProof ← out.floatProof
  let opTerm ← `(Op.dotTyped $leftProof $rightProof $outProof
    (options := $optionsT) (batch := $batchT) (M := $aM) (K := $aK) (N := $bN)
    $a'.term $b'.term)
  return ⟨opTerm, out, .dims (aBatch ++ [aM, bN]), none, none⟩

/-- Preserve the explicit dot accumulator as part of the opaque operation.
Triton 3.7.1 defaults out_dtype to fp32 even when an accumulator is supplied;
fp16 accumulation requires an explicit matching out_dtype. -/
partial def expandDotAcc (expandExpr : ExprExpander) (env : Env)
    (a b acc : TSyntax `tritonExpr) (options : Option (TSyntax `term) := none)
    (resultHint : Option DInfo := none) : MacroM EOut := do
  let acc' ← expandExpr env acc
  let hint := resultHint.getD .fp32
  let dot ← expandDot expandExpr env a b options (some hint)
  ensureShape dot.shape acc'.shape "tl.dot accumulator"
  if dot.dtype == .int then
    ensureDType .int acc'.dtype "tl.dot accumulator"
    let (bc, outShape) ← broadcastTerm dot.shape acc'.shape "tl.dot accumulator"
    return ⟨← `(Op.add NumericDType.int $bc $dot.term $acc'.term),
      .int, outShape, none, none⟩
  unless acc'.dtype == .real || acc'.dtype == .fp32 || acc'.dtype == .fp16 do
    Macro.throwError "tl.dot accumulator must have float16 or float32 precision"
  let declared := if acc'.dtype == .real then DInfo.fp32 else acc'.dtype
  if hint != declared then
    Macro.throwError "tl.dot: out_dtype must match accumulator precision"
  let actual := if dot.dtype == .real then DInfo.fp32 else dot.dtype
  ensureDType actual declared "tl.dot accumulator"
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  let .dims aDims := a'.shape
  let .dims bDims := b'.shape
  let rec batchTerm : List (TSyntax `term) → MacroM (TSyntax `term)
    | [] => `(([] : TileShape))
    | d :: rest => do
        let tail ← batchTerm rest
        `($d :: $tail)
  let batchT ← batchTerm (aDims.dropLast.dropLast)
  let aM := aDims[aDims.length - 2]!
  let aK := aDims[aDims.length - 1]!
  let bN := bDims[bDims.length - 1]!
  let left ← a'.dtype.floatProof
  let right ← b'.dtype.floatProof
  let out ← acc'.dtype.floatProof
  let optionsT ← match options with
    | some term => pure term
    | none => `(({} : DotOptions))
  let term ← `(Op.dotAcc $left $right $out (batch := $batchT)
    (M := $aM) (K := $aK) (N := $bN) $a'.term $b'.term $acc'.term $optionsT)
  let (ct, cp) ← if acc'.dtype == .real then
    floatingComputeArith? "tl.dot accumulator" term acc' acc'
    else pure (none, none)
  return ⟨term, acc'.dtype, dot.shape, ct, cp⟩

/-- Dot options are compile-time choices, retained even in nested expressions.
Unknown or repeated options are rejected instead of silently losing precision. -/
def expandDotOptions (kwargs : Array (TSyntax `tritonReduceKwarg)) :
    MacroM (TSyntax `term × Option DInfo) := do
  let mut resultHint : Option DInfo := none
  let mut result ← `(({} : DotOptions))
  let mut seen : List String := []
  for kw in kwargs do
    let (name, updated) ← match kw with
      | `(tritonReduceKwarg| out_dtype=$dt:tritonDType) => do
          let info ← expandDType dt
          resultHint := some info
          let dtype ← info.term
          pure ("out_dtype", ← `({ $result with outDType := some $dtype }))
      | `(tritonReduceKwarg| input_precision=$mode:str) =>
          pure ("input_precision", ← `({ $result with inputPrecision := some $mode }))
      | `(tritonReduceKwarg| $key:ident=$value:tritonExpr) => do
          let name := key.getId.toString
          let term ← match value with
            | `(tritonExpr| True) | `(tritonExpr| true) => `(Bool.true)
            | `(tritonExpr| False) | `(tritonExpr| false) => `(Bool.false)
            | `(tritonExpr| $($t:term)) => pure t
            | `(tritonExpr| $n:num) => pure ⟨n.raw⟩
            | `(tritonExpr| $i:ident) => pure ⟨i.raw⟩
            | _ => Macro.throwError "tl.dot options require compile-time values"
          match name with
          | "allow_tf32" => pure (name, ← `({ $result with allowTF32 := some $term }))
          | "input_precision" => pure (name, ← `({ $result with inputPrecision := some $term }))
          | "max_num_imprecise_acc" =>
              pure (name, ← `({ $result with maxNumImpreciseAcc := some $term }))
          | _ => Macro.throwError ("tl.dot: unsupported option `" ++ name ++ "`")
      | _ => Macro.throwError "tl.dot: unsupported option"
    if seen.contains name then Macro.throwError ("tl.dot: repeated option `" ++ name ++ "`")
    seen := name :: seen
    result := updated
  if seen.contains "allow_tf32" && seen.contains "input_precision" then
    Macro.throwError "tl.dot: specify only one of allow_tf32 and input_precision"
  return (result, resultHint)

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
        Macro.throwError (ctx ++ ": literal element-count mismatch (" ++ toString inProduct ++
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
