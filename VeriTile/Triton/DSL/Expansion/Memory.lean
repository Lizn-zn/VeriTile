/-
VeriTile.Triton.DSL.Expansion.Memory

Memory-surface expansion helpers.
-/

import VeriTile.Triton.DSL.Expansion.Common

open Lean

namespace VeriTile.Triton.DSL

partial def expandLoad (expandExpr : ExprExpander)
    (expandStaticPtrExpr : StaticPtrExpander) (env : Env)
    (p : TSyntax `tritonExpr) (kwargs : TSyntaxArray `tritonMemKwarg) :
    MacroM EOut := do
  let mut maskTerm : Option (TSyntax `term × SInfo) := none
  let mut otherSyntax : Option (TSyntax `tritonExpr) := none
  let mut dtype? : Option DInfo := none
  let mut boundaryCheck? : Option (TSyntax `term) := none
  let mut padding : TSyntax `term ← `(PaddingOption.zero)
  for kw in kwargs do
    match kw with
    | `(tritonMemKwarg| boundary_check=$axes:term) =>
        if boundaryCheck?.isSome then
          Macro.throwError "tl.load: duplicate `boundary_check=` kwarg"
        boundaryCheck? := some axes
    | `(tritonMemKwarg| padding_option="zero") =>
        padding := ← `(PaddingOption.zero)
    | `(tritonMemKwarg| $name:ident = $val:tritonExpr) =>
        match name.getId.toString with
        | "mask"  =>
            let val' ← expandExpr env val
            ensureDType .bool val'.dtype "tl.load mask"
            maskTerm := some (val'.term, val'.shape)
        | "other" =>
            otherSyntax := some val
        | unknown =>
            let msg : String :=
              "tl.load: unknown kwarg `" ++ unknown ++
              "`. Only `mask`, `other`, `dtype`, `boundary_check`, and `padding_option` are recognized."
            Macro.throwError msg
    | `(tritonMemKwarg| $name:ident = $dt:tritonDType) =>
        unless name.getId.toString == "dtype" do
          Macro.throwError
            ("tl.load: unknown kwarg `" ++ name.getId.toString ++
             "`. Only `mask`, `other`, `dtype`, `boundary_check`, and `padding_option` are recognized.")
        if dtype?.isSome then
          Macro.throwError "tl.load: duplicate `dtype=` kwarg"
        dtype? := some (← expandDType dt)
    | _ => Macro.throwUnsupported
  let outDType := dtype?.getD .real
  if let some boundaryCheck := boundaryCheck? then
    if maskTerm.isSome || otherSyntax.isSome then
      Macro.throwError "tl.load: block-pointer `boundary_check` cannot be combined with `mask` or `other`"
    let p' ← expandExpr env p
    ensureDType .blockPtr p'.dtype "tl.load block pointer"
    let dt ← outDType.term
    return ⟨← `(Op.load $dt (MemAccess.blockPtr $p'.term $boundaryCheck) MaskOpt.none),
      outDType, p'.shape, none⟩
  let otherTerm ←
    match otherSyntax with
    | none => pure none
    | some other => do
        let other' ←
          match ← expandLeanAntiquoteAs? outDType other with
          | some out => pure out
          | none => expandExpr env other
        pure (some (other'.term, other'.dtype, other'.shape))
  if let some (_, otherDType, _) := otherTerm then
    unless otherDType == outDType do
      Macro.throwError "tl.load other: dtype must match load result dtype"
  match maskTerm, otherTerm with
  | none, none =>
      match ← expandStaticPtrExpr env p with
      | some sp =>
          let r := sp.region
          let off := sp.offset
          let dt ← outDType.term
          pure ⟨← `(Op.load $dt (MemAccess.region $r $off) MaskOpt.none), outDType, sp.shape, none⟩
      | none =>
          let p' ← expandExpr env p
          ensureDType .ptr p'.dtype "tl.load pointer"
          let dt ← outDType.term
          pure ⟨← `(Op.load $dt (MemAccess.ptr $p'.term) MaskOpt.none), outDType, p'.shape, none⟩
  | some (m, mShape), none =>
      match ← expandStaticPtrExpr env p with
      | some sp =>
          let r := sp.region
          let off := sp.offset
          let m' ← coerceShape m mShape sp.shape "tl.load mask"
          let dt ← outDType.term
          pure ⟨← `(Op.load $dt (MemAccess.region $r $off) (MaskOpt.mask $m')), outDType, sp.shape, none⟩
      | none =>
          let p' ← expandExpr env p
          ensureDType .ptr p'.dtype "tl.load pointer"
          let m' ← coerceShape m mShape p'.shape "tl.load mask"
          let dt ← outDType.term
          pure ⟨← `(Op.load $dt (MemAccess.ptr $p'.term) (MaskOpt.mask $m')), outDType, p'.shape, none⟩
  | some (m, mShape), some (o, otherDType, oShape) =>
      match ← expandStaticPtrExpr env p with
      | some sp =>
          let r := sp.region
          let off := sp.offset
          let m' ← coerceShape m mShape sp.shape "tl.load mask"
          let o' ← coerceShape o oShape sp.shape "tl.load other"
          let dt ← otherDType.term
          pure ⟨← `(Op.load $dt (MemAccess.region $r $off) (MaskOpt.maskOther $m' $o')), otherDType, sp.shape, none⟩
      | none =>
          let p' ← expandExpr env p
          ensureDType .ptr p'.dtype "tl.load pointer"
          let m' ← coerceShape m mShape p'.shape "tl.load mask"
          let o' ← coerceShape o oShape p'.shape "tl.load other"
          let dt ← otherDType.term
          pure ⟨← `(Op.load $dt (MemAccess.ptr $p'.term) (MaskOpt.maskOther $m' $o')), otherDType, p'.shape, none⟩
  | none, some _ =>
      Macro.throwError
        "tl.load: `other=` requires `mask=`. (Triton: `other` is meaningful only when some lanes are masked off.)"

partial def expandStore (expandExpr : ExprExpander)
    (expandStaticPtrExpr : StaticPtrExpander) (env : Env)
    (p v : TSyntax `tritonExpr) (kwargs : TSyntaxArray `tritonMemKwarg) :
    MacroM StmtExpansion := do
  let mut maskTerm : Option (TSyntax `term × SInfo) := none
  let mut dtype? : Option DInfo := none
  let mut boundaryCheck? : Option (TSyntax `term) := none
  for kw in kwargs do
    match kw with
    | `(tritonMemKwarg| boundary_check=$axes:term) =>
        if boundaryCheck?.isSome then
          Macro.throwError "tl.store: duplicate `boundary_check=` kwarg"
        boundaryCheck? := some axes
    | `(tritonMemKwarg| padding_option="zero") =>
        Macro.throwError "tl.store: `padding_option` is only valid on tl.load"
    | `(tritonMemKwarg| $name:ident = $kval:tritonExpr) =>
        let kval' ← expandExpr env kval
        match name.getId.toString with
        | "mask"  =>
            ensureDType .bool kval'.dtype "tl.store mask"
            maskTerm := some (kval'.term, kval'.shape)
        | unknown =>
            let msg : String :=
              "tl.store: unknown kwarg `" ++ unknown ++
              "`. Only `mask`, `dtype`, and `boundary_check` are recognized (Triton's tl.store has no `other`; see issue #16)."
            Macro.throwError msg
    | `(tritonMemKwarg| $name:ident = $dt:tritonDType) =>
        unless name.getId.toString == "dtype" do
          Macro.throwError
            ("tl.store: unknown kwarg `" ++ name.getId.toString ++
             "`. Only `mask`, `dtype`, and `boundary_check` are recognized (Triton's tl.store has no `other`; see issue #16).")
        if dtype?.isSome then
          Macro.throwError "tl.store: duplicate `dtype=` kwarg"
        dtype? := some (← expandDType dt)
    | _ => Macro.throwUnsupported
  let storeExpected := dtype?.getD .real
  let v' ←
    match ← expandLeanAntiquoteAs? storeExpected v with
    | some out => pure out
    | none => expandExpr env v
  let valueExpr ←
    match v'.computeTerm with
    | some ce => pure ce
    | none => `(ComputeExpr.alg $v'.term)
  let storeDType := dtype?.getD v'.dtype
  unless storeDType == v'.dtype do
    Macro.throwError "tl.store: `dtype=` must match the value dtype"
  if let some boundaryCheck := boundaryCheck? then
    if maskTerm.isSome then
      Macro.throwError "tl.store: block-pointer `boundary_check` cannot be combined with `mask`"
    let p' ← expandExpr env p
    ensureDType .blockPtr p'.dtype "tl.store block pointer"
    let vTerm ← coerceShape v'.term v'.shape p'.shape "tl.store value"
    let sh ← p'.shape.term
    let dt ← storeDType.term
    let valueExpr' ←
      match v'.computeTerm with
      | some _ => pure valueExpr
      | none => `(ComputeExpr.alg $vTerm)
    return (← `(Stmt.store $dt $sh (MemAccess.blockPtr $p'.term $boundaryCheck) $vTerm MaskOpt.none),
      ← `(ComputeStmt.store $dt $sh (MemAccess.blockPtr $p'.term $boundaryCheck) $valueExpr' MaskOpt.none),
      env, v'.computeTerm.isSome)
  match maskTerm with
  | none =>
      match ← expandStaticPtrExpr env p with
      | some sp =>
          let vTerm ← coerceShape v'.term v'.shape sp.shape "tl.store value"
          let r := sp.region
          let off := sp.offset
          let sh ← sp.shape.term
          let dt ← storeDType.term
          let valueExpr' ←
            match v'.computeTerm with
            | some _ => pure valueExpr
            | none => `(ComputeExpr.alg $vTerm)
          pure (← `(Stmt.store $dt $sh (MemAccess.region $r $off) $vTerm MaskOpt.none),
            ← `(ComputeStmt.store $dt $sh (MemAccess.region $r $off) $valueExpr' MaskOpt.none),
            env, v'.computeTerm.isSome)
      | none =>
          let p' ← expandExpr env p
          ensureDType .ptr p'.dtype "tl.store pointer"
          let vTerm ← coerceShape v'.term v'.shape p'.shape "tl.store value"
          let sh ← p'.shape.term
          let dt ← storeDType.term
          let valueExpr' ←
            match v'.computeTerm with
            | some _ => pure valueExpr
            | none => `(ComputeExpr.alg $vTerm)
          pure (← `(Stmt.store $dt $sh (MemAccess.ptr $p'.term) $vTerm MaskOpt.none),
            ← `(ComputeStmt.store $dt $sh (MemAccess.ptr $p'.term) $valueExpr' MaskOpt.none),
            env, v'.computeTerm.isSome)
  | some (m, mShape) =>
      match ← expandStaticPtrExpr env p with
      | some sp =>
          let vTerm ← coerceShape v'.term v'.shape sp.shape "tl.store value"
          let m' ← coerceShape m mShape sp.shape "tl.store mask"
          let r := sp.region
          let off := sp.offset
          let sh ← sp.shape.term
          let dt ← storeDType.term
          let valueExpr' ←
            match v'.computeTerm with
            | some _ => pure valueExpr
            | none => `(ComputeExpr.alg $vTerm)
          pure (← `(Stmt.store $dt $sh (MemAccess.region $r $off) $vTerm (MaskOpt.mask $m')),
            ← `(ComputeStmt.store $dt $sh (MemAccess.region $r $off) $valueExpr' (MaskOpt.mask $m')),
            env, v'.computeTerm.isSome)
      | none =>
          let p' ← expandExpr env p
          ensureDType .ptr p'.dtype "tl.store pointer"
          let vTerm ← coerceShape v'.term v'.shape p'.shape "tl.store value"
          let m' ← coerceShape m mShape p'.shape "tl.store mask"
          let sh ← p'.shape.term
          let dt ← storeDType.term
          let valueExpr' ←
            match v'.computeTerm with
            | some _ => pure valueExpr
            | none => `(ComputeExpr.alg $vTerm)
            pure (← `(Stmt.store $dt $sh (MemAccess.ptr $p'.term) $vTerm (MaskOpt.mask $m')),
              ← `(ComputeStmt.store $dt $sh (MemAccess.ptr $p'.term) $valueExpr' (MaskOpt.mask $m')),
              env, v'.computeTerm.isSome)

partial def expandAtomicAdd (expandExpr : ExprExpander)
    (expandStaticPtrExpr : StaticPtrExpander) (env : Env)
    (p v : TSyntax `tritonExpr) (kwargs : TSyntaxArray `tritonMemKwarg) :
    MacroM StmtExpansion := do
  let mut maskTerm : Option (TSyntax `term × SInfo) := none
  let mut dtype? : Option DInfo := none
  for kw in kwargs do
    match kw with
    | `(tritonMemKwarg| boundary_check=$_:term) =>
        Macro.throwError "tl.atomic_add: `boundary_check=` is not supported; block-pointer atomics are deferred"
    | `(tritonMemKwarg| padding_option="zero") =>
        Macro.throwError "tl.atomic_add: `padding_option` is only valid on tl.load"
    | `(tritonMemKwarg| $name:ident = $kval:tritonExpr) =>
        let kval' ← expandExpr env kval
        match name.getId.toString with
        | "mask" =>
            ensureDType .bool kval'.dtype "tl.atomic_add mask"
            maskTerm := some (kval'.term, kval'.shape)
        | unknown =>
            Macro.throwError
              ("tl.atomic_add: unknown kwarg `" ++ unknown ++
               "`. Only `mask` and `dtype` are recognized.")
    | `(tritonMemKwarg| $name:ident = $dt:tritonDType) =>
        unless name.getId.toString == "dtype" do
          Macro.throwError
            ("tl.atomic_add: unknown kwarg `" ++ name.getId.toString ++
             "`. Only `mask` and `dtype` are recognized.")
        if dtype?.isSome then
          Macro.throwError "tl.atomic_add: duplicate `dtype=` kwarg"
        dtype? := some (← expandDType dt)
    | _ => Macro.throwUnsupported
  let atomicExpected := dtype?.getD .real
  let v' ←
    match ← expandLeanAntiquoteAs? atomicExpected v with
    | some out => pure out
    | none => expandExpr env v
  let valueExpr ←
    match v'.computeTerm with
    | some ce => pure ce
    | none => `(ComputeExpr.alg $v'.term)
  let atomicDType := dtype?.getD v'.dtype
  unless atomicDType == v'.dtype do
    Macro.throwError "tl.atomic_add: `dtype=` must match the value dtype"
  let hnum ← atomicDType.numericProof
  match maskTerm with
  | none =>
      match ← expandStaticPtrExpr env p with
      | some sp =>
          let vTerm ← coerceShape v'.term v'.shape sp.shape "tl.atomic_add value"
          let sh ← sp.shape.term
          let valueExpr' ←
            match v'.computeTerm with
            | some _ => pure valueExpr
            | none => `(ComputeExpr.alg $vTerm)
          pure (← `(Stmt.atomicAdd $hnum $sh (MemAccess.region $sp.region $sp.offset) $vTerm MaskOpt.none),
            ← `(ComputeStmt.atomicAdd $hnum $sh (MemAccess.region $sp.region $sp.offset) $valueExpr' MaskOpt.none),
            env, v'.computeTerm.isSome)
      | none =>
          let p' ← expandExpr env p
          ensureDType .ptr p'.dtype "tl.atomic_add pointer"
          let vTerm ← coerceShape v'.term v'.shape p'.shape "tl.atomic_add value"
          let sh ← p'.shape.term
          let valueExpr' ←
            match v'.computeTerm with
            | some _ => pure valueExpr
            | none => `(ComputeExpr.alg $vTerm)
          pure (← `(Stmt.atomicAdd $hnum $sh (MemAccess.ptr $p'.term) $vTerm MaskOpt.none),
            ← `(ComputeStmt.atomicAdd $hnum $sh (MemAccess.ptr $p'.term) $valueExpr' MaskOpt.none),
            env, v'.computeTerm.isSome)
  | some (m, mShape) =>
      match ← expandStaticPtrExpr env p with
      | some sp =>
          let vTerm ← coerceShape v'.term v'.shape sp.shape "tl.atomic_add value"
          let m' ← coerceShape m mShape sp.shape "tl.atomic_add mask"
          let sh ← sp.shape.term
          let valueExpr' ←
            match v'.computeTerm with
            | some _ => pure valueExpr
            | none => `(ComputeExpr.alg $vTerm)
          pure (← `(Stmt.atomicAdd $hnum $sh (MemAccess.region $sp.region $sp.offset) $vTerm (MaskOpt.mask $m')),
            ← `(ComputeStmt.atomicAdd $hnum $sh (MemAccess.region $sp.region $sp.offset) $valueExpr' (MaskOpt.mask $m')),
            env, v'.computeTerm.isSome)
      | none =>
          let p' ← expandExpr env p
          ensureDType .ptr p'.dtype "tl.atomic_add pointer"
          let vTerm ← coerceShape v'.term v'.shape p'.shape "tl.atomic_add value"
          let m' ← coerceShape m mShape p'.shape "tl.atomic_add mask"
          let sh ← p'.shape.term
          let valueExpr' ←
            match v'.computeTerm with
            | some _ => pure valueExpr
            | none => `(ComputeExpr.alg $vTerm)
          pure (← `(Stmt.atomicAdd $hnum $sh (MemAccess.ptr $p'.term) $vTerm (MaskOpt.mask $m')),
            ← `(ComputeStmt.atomicAdd $hnum $sh (MemAccess.ptr $p'.term) $valueExpr' (MaskOpt.mask $m')),
            env, v'.computeTerm.isSome)

end VeriTile.Triton.DSL
