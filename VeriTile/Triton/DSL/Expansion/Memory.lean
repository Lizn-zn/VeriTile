/-
VeriTile.Triton.DSL.Expansion.Memory

Memory-surface expansion helpers.
-/

import VeriTile.Triton.DSL.Expansion.Common

open Lean

namespace VeriTile.Triton.DSL

private def numArrayAsNatListTerm (axes : TSyntaxArray `num) :
    MacroM (TSyntax `term) := do
  let axisTerms : Array (TSyntax `term) :=
    axes.map fun n => (⟨n.raw⟩ : TSyntax `term)
  let rec go : List (TSyntax `term) → MacroM (TSyntax `term)
    | [] => `(([] : List Nat))
    | d :: rest => do
        let tail ← go rest
        `(($d : Nat) :: $tail)
  go axisTerms.toList

private def expandLoadOtherAs? (dtype : DInfo) (e : TSyntax `tritonExpr) :
    MacroM (Option EOut) := do
  match dtype with
  | .nat =>
      match e with
      | `(tritonExpr| $n:num) =>
          pure (some ⟨← `(Op.constNat $n), .nat, SInfo.scalar, none, none⟩)
      | _ => pure none
  | _ =>
      pure none

partial def expandLoad (expandExpr : ExprExpander)
    (expandStaticPtrExpr : StaticPtrExpander) (env : Env)
    (p : TSyntax `tritonExpr) (kwargs : TSyntaxArray `tritonMemKwarg)
    (defaultDType : Option DInfo := none)
    (positionalMask : Option (TSyntax `tritonExpr) := none) :
    MacroM EOut := do
  let mut maskTerm : Option (TSyntax `term × SInfo) := none
  match positionalMask with
  | none => pure ()
  | some mask => do
      let mask' ← expandExpr env mask
      ensureDType .bool mask'.dtype "tl.load mask"
      maskTerm := some (mask'.term, mask'.shape)
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
    | `(tritonMemKwarg| boundary_check=($axes:term)) =>
        if boundaryCheck?.isSome then
          Macro.throwError "tl.load: duplicate `boundary_check=` kwarg"
        boundaryCheck? := some axes
    | `(tritonMemKwarg| boundary_check=([$axes:num,*] : $_ty:term)) =>
        if boundaryCheck?.isSome then
          Macro.throwError "tl.load: duplicate `boundary_check=` kwarg"
        boundaryCheck? := some (← numArrayAsNatListTerm axes)
    | `(tritonMemKwarg| boundary_check=($axes:num,*)) =>
        if boundaryCheck?.isSome then
          Macro.throwError "tl.load: duplicate `boundary_check=` kwarg"
        boundaryCheck? := some (← numArrayAsNatListTerm axes)
    | `(tritonMemKwarg| padding_option="zero") =>
        padding := ← `(PaddingOption.zero)
    | `(tritonMemKwarg| $name:ident = $dt:tritonDType) =>
        unless name.getId.getString! == "dtype" do
          Macro.throwError
            ("tl.load: unknown kwarg `" ++ name.getId.toString ++
             "`. Only `mask`, `other`, `dtype`, `boundary_check`, and `padding_option` are recognized.")
        if dtype?.isSome then
          Macro.throwError "tl.load: duplicate `dtype=` kwarg"
        dtype? := some (← expandDType dt)
    | `(tritonMemKwarg| $name:ident = $val:tritonExpr) =>
        match name.getId.getString! with
        | "mask"  =>
            let val' ← expandExpr env val
            ensureDType .bool val'.dtype "tl.load mask"
            maskTerm := some (val'.term, val'.shape)
        | "other" =>
            otherSyntax := some val
        | "dtype" =>
            if dtype?.isSome then
              Macro.throwError "tl.load: duplicate `dtype=` kwarg"
            dtype? := some (← expandDTypeExpr val)
        | unknown =>
            let msg : String :=
              "tl.load: unknown kwarg `" ++ unknown ++
              "`. Only `mask`, `other`, `dtype`, `boundary_check`, and `padding_option` are recognized."
            Macro.throwError msg
    | _ => Macro.throwUnsupported
  let explicitOrDefaultDType := dtype?.orElse (fun _ => defaultDType)
  let loadDType (spDType? : Option DInfo := none) : DInfo :=
    explicitOrDefaultDType.orElse (fun _ => spDType?) |>.getD .real
  let mkLoadOutWithDType (outDType : DInfo) (mem mask : TSyntax `term) (shape : SInfo) : MacroM EOut := do
    match outDType with
    | .fp32 =>
        let algTerm ← `(Op.load TileDType.real $mem $mask)
        pure ⟨algTerm, .real, shape,
          some (← fp32ComputeLoadExpr mem mask), some .fp32⟩
    | _ =>
        let dt ← outDType.term
        pure ⟨← `(Op.load $dt $mem $mask), outDType, shape, none, none⟩
  let staticPtr? ←
    if boundaryCheck?.isSome then
      pure none
    else
      expandStaticPtrExpr env p
  let outDType := loadDType (staticPtr?.bind (fun sp => sp.regionDType?))
  let mkLoadOut (mem mask : TSyntax `term) (shape : SInfo) : MacroM EOut := do
    mkLoadOutWithDType outDType mem mask shape
  if let some boundaryCheck := boundaryCheck? then
    if maskTerm.isSome || otherSyntax.isSome then
      Macro.throwError "tl.load: block-pointer `boundary_check` cannot be combined with `mask` or `other`"
    let p' ← expandExpr env p
    ensureDType .blockPtr p'.dtype "tl.load block pointer"
    return ← mkLoadOut (← `(MemAccess.blockPtr $p'.term $boundaryCheck)) (← `(MaskOpt.none)) p'.shape
  let otherTerm ←
    match otherSyntax with
    | none => pure none
    | some other => do
        let other' ←
          match ← expandLoadOtherAs? outDType other with
          | some out => pure out
          | none => expandExpr env other
        pure (some (other'.term, other'.dtype, other'.shape, other'.computeDType?))
  if let some (_, otherDType, _, _) := otherTerm then
    unless otherDType == outDType || (outDType == .fp32 && otherDType == .real) do
      Macro.throwError "tl.load other: dtype must match load result dtype"
  match maskTerm, otherTerm with
  | none, none =>
      match staticPtr? with
      | some sp =>
          let r := sp.region
          let off := sp.offset
          mkLoadOut (← `(MemAccess.region $r $off)) (← `(MaskOpt.none)) sp.shape
      | none =>
          let p' ← expandExpr env p
          if p'.dtype == .blockPtr then
            mkLoadOut (← `(MemAccess.blockPtr $p'.term ([] : List Nat))) (← `(MaskOpt.none)) p'.shape
          else
            ensureDType .ptr p'.dtype "tl.load pointer"
            mkLoadOut (← `(MemAccess.ptr $p'.term)) (← `(MaskOpt.none)) p'.shape
  | some (m, mShape), none =>
      match staticPtr? with
      | some sp =>
          let r := sp.region
          let off := sp.offset
          let m' ← coerceShape m mShape sp.shape "tl.load mask"
          mkLoadOut (← `(MemAccess.region $r $off)) (← `(MaskOpt.mask $m')) sp.shape
      | none =>
          let p' ← expandExpr env p
          ensureDType .ptr p'.dtype "tl.load pointer"
          let m' ← coerceShape m mShape p'.shape "tl.load mask"
          mkLoadOut (← `(MemAccess.ptr $p'.term)) (← `(MaskOpt.mask $m')) p'.shape
  | some (m, mShape), some (o, _, oShape, _) =>
      match staticPtr? with
      | some sp =>
          let r := sp.region
          let off := sp.offset
          let m' ← coerceShape m mShape sp.shape "tl.load mask"
          let o' ← coerceShape o oShape sp.shape "tl.load other"
          mkLoadOut (← `(MemAccess.region $r $off)) (← `(MaskOpt.maskOther $m' $o')) sp.shape
      | none =>
          let p' ← expandExpr env p
          ensureDType .ptr p'.dtype "tl.load pointer"
          let m' ← coerceShape m mShape p'.shape "tl.load mask"
          let o' ← coerceShape o oShape p'.shape "tl.load other"
          mkLoadOut (← `(MemAccess.ptr $p'.term)) (← `(MaskOpt.maskOther $m' $o')) p'.shape
  | none, some _ =>
      Macro.throwError
        "tl.load: `other=` requires `mask=`. (Triton: `other` is meaningful only when some lanes are masked off.)"

partial def expandStore (expandExpr : ExprExpander)
    (expandStaticPtrExpr : StaticPtrExpander) (env : Env)
    (p v : TSyntax `tritonExpr) (kwargs : TSyntaxArray `tritonMemKwarg)
    (positionalMask : Option (TSyntax `tritonExpr) := none) :
    MacroM StmtExpansion := do
  let mut maskTerm : Option (TSyntax `term × SInfo) := none
  match positionalMask with
  | none => pure ()
  | some mask => do
      let mask' ← expandExpr env mask
      ensureDType .bool mask'.dtype "tl.store mask"
      maskTerm := some (mask'.term, mask'.shape)
  let mut dtype? : Option DInfo := none
  let mut boundaryCheck? : Option (TSyntax `term) := none
  for kw in kwargs do
    match kw with
    | `(tritonMemKwarg| boundary_check=$axes:term) =>
        if boundaryCheck?.isSome then
          Macro.throwError "tl.store: duplicate `boundary_check=` kwarg"
        boundaryCheck? := some axes
    | `(tritonMemKwarg| boundary_check=($axes:term)) =>
        if boundaryCheck?.isSome then
          Macro.throwError "tl.store: duplicate `boundary_check=` kwarg"
        boundaryCheck? := some axes
    | `(tritonMemKwarg| boundary_check=([$axes:num,*] : $_ty:term)) =>
        if boundaryCheck?.isSome then
          Macro.throwError "tl.store: duplicate `boundary_check=` kwarg"
        boundaryCheck? := some (← numArrayAsNatListTerm axes)
    | `(tritonMemKwarg| boundary_check=($axes:num,*)) =>
        if boundaryCheck?.isSome then
          Macro.throwError "tl.store: duplicate `boundary_check=` kwarg"
        boundaryCheck? := some (← numArrayAsNatListTerm axes)
    | `(tritonMemKwarg| padding_option="zero") =>
        Macro.throwError "tl.store: `padding_option` is only valid on tl.load"
    | `(tritonMemKwarg| $name:ident = $dt:tritonDType) =>
        unless name.getId.getString! == "dtype" do
          Macro.throwError
            ("tl.store: unknown kwarg `" ++ name.getId.toString ++
             "`. Only `mask`, `dtype`, and `boundary_check` are recognized (Triton's tl.store has no `other`; see issue #16).")
        if dtype?.isSome then
          Macro.throwError "tl.store: duplicate `dtype=` kwarg"
        dtype? := some (← expandDType dt)
    | `(tritonMemKwarg| $name:ident = $kval:tritonExpr) =>
        match name.getId.getString! with
        | "mask"  =>
            if maskTerm.isSome then
              Macro.throwError "tl.store: duplicate `mask=` kwarg"
            let kval' ← expandExpr env kval
            ensureDType .bool kval'.dtype "tl.store mask"
            maskTerm := some (kval'.term, kval'.shape)
        | "dtype" =>
            if dtype?.isSome then
              Macro.throwError "tl.store: duplicate `dtype=` kwarg"
            dtype? := some (← expandDTypeExpr kval)
        | unknown =>
            let msg : String :=
              "tl.store: unknown kwarg `" ++ unknown ++
              "`. Only `mask`, `dtype`, and `boundary_check` are recognized (Triton's tl.store has no `other`; see issue #16)."
            Macro.throwError msg
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
    | `(tritonMemKwarg| $name:ident = $dt:tritonDType) =>
        unless name.getId.getString! == "dtype" do
          Macro.throwError
            ("tl.atomic_add: unknown kwarg `" ++ name.getId.toString ++
             "`. Only `mask` and `dtype` are recognized.")
        if dtype?.isSome then
          Macro.throwError "tl.atomic_add: duplicate `dtype=` kwarg"
        dtype? := some (← expandDType dt)
    | `(tritonMemKwarg| $name:ident = $kval:tritonExpr) =>
        match name.getId.getString! with
        | "mask" =>
            let kval' ← expandExpr env kval
            ensureDType .bool kval'.dtype "tl.atomic_add mask"
            maskTerm := some (kval'.term, kval'.shape)
        | "dtype" =>
            if dtype?.isSome then
              Macro.throwError "tl.atomic_add: duplicate `dtype=` kwarg"
            dtype? := some (← expandDTypeExpr kval)
        | unknown =>
            Macro.throwError
              ("tl.atomic_add: unknown kwarg `" ++ unknown ++
               "`. Only `mask` and `dtype` are recognized.")
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
