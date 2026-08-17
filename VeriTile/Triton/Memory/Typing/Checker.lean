/-
VeriTile.Triton.Memory.Typing.Checker

Optional well-formedness checker for Triton kernels.
-/

import VeriTile.Triton.Memory.Typing.Region

namespace VeriTile.Triton

/-- Optional region-dtype environment for the executable checker. -/
abbrev RegionEnv := RegionName → Option TileDType

/-- Checker mode for undeclared regions. -/
inductive CheckMode where
  | lax
  | strict
  deriving DecidableEq, Repr

/-- Executable checker errors. This layer is diagnostic; it is not part of
the mathematical proof boundary. -/
inductive CheckError where
  | undeclaredRegion (region : RegionName)
  | regionDTypeMismatch (region : RegionName)
      (expected actual : TileDType)
  | unboundRegister (name : RegName)
  | registerDTypeShapeMismatch (name : RegName)
      (oldDType newDType : TileDType) (oldShape newShape : TileShape)
  | missingPointerProvenance (name : RegName)
  | missingBlockPointerProvenance (name : RegName)
  | unsupportedPointerProvenance
  | unsupportedBlockPointerProvenance
  | pointerProvenanceConflict (left right : RegionName)
  | blockPointerProvenanceConflict (left right : RegionName)
  | blockPointerMetadataMismatch
      (parentRank strideRank offsetRank blockRank : Nat)
  | boundaryAxisOutOfRange (axis rank : Nat)
  | blockPointerAdvanceUnderflow (axis : Nat) (offset : Nat) (delta : Int)
  deriving DecidableEq, Repr

structure BlockPtrSummary where
  region : RegionName
  parentRank : Option Nat := none
  offsets : Option (List Nat) := none
  deriving DecidableEq, Repr

structure RegEntry where
  dtype : TileDType
  shape : TileShape
  ptrProvenance : Option RegionName := none
  blockPtrSummary : Option BlockPtrSummary := none
  deriving DecidableEq, Repr

structure CheckCtx where
  mode : CheckMode
  regions : RegionEnv
  regs : List (RegName × RegEntry) := []

namespace CheckCtx

def lookupReg (ctx : CheckCtx) (name : RegName) : Except CheckError RegEntry :=
  match ctx.regs.find? (fun entry => entry.1 == name) with
  | some (_, entry) => .ok entry
  | none => .error (.unboundRegister name)

def checkRegion (ctx : CheckCtx) (region : RegionName) (dtype : TileDType) :
    Except CheckError Unit :=
  match ctx.regions region with
  | some actual =>
      if actual = dtype then
        .ok ()
      else
        .error (.regionDTypeMismatch region dtype actual)
  | none =>
      match ctx.mode with
      | .lax => .ok ()
      | .strict => .error (.undeclaredRegion region)

def checkRegRef (ctx : CheckCtx) (name : RegName)
    (dtype : TileDType) (shape : TileShape) : Except CheckError RegEntry := do
  let entry ← ctx.lookupReg name
  if entry.dtype = dtype then
    if entry.shape = shape then
      .ok entry
    else
      .error (.registerDTypeShapeMismatch name entry.dtype dtype entry.shape shape)
  else
    .error (.registerDTypeShapeMismatch name entry.dtype dtype entry.shape shape)

def setReg (ctx : CheckCtx) (name : RegName) (entry : RegEntry) :
    Except CheckError CheckCtx :=
  match ctx.regs.find? (fun item => item.1 == name) with
  | none => .ok { ctx with regs := (name, entry) :: ctx.regs }
  | some (_, old) =>
      if old.dtype = entry.dtype then
        if old.shape = entry.shape then
          .ok { ctx with regs :=
            (name, entry) :: ctx.regs.filter (fun item => !(item.1 == name)) }
        else
          .error (.registerDTypeShapeMismatch name old.dtype entry.dtype old.shape entry.shape)
      else
        .error (.registerDTypeShapeMismatch name old.dtype entry.dtype old.shape entry.shape)

end CheckCtx

def checkBoundaryAxes (rank : Nat) : List Nat → Except CheckError Unit
  | [] => .ok ()
  | axis :: rest =>
      if BlockPtr.axisInRank rank axis then
        checkBoundaryAxes rank rest
      else
        .error (.boundaryAxisOutOfRange axis rank)

def checkBlockPtrMetadataRanks
    (parentRank strideRank offsetRank blockRank : Nat) :
    Except CheckError Unit :=
  if BlockPtr.metadataRanksValid parentRank strideRank offsetRank blockRank then
    .ok ()
  else
    .error (.blockPointerMetadataMismatch parentRank strideRank offsetRank blockRank)

def checkBlockPtrMetadata (parentShape : List Nat) (blockShape : TileShape)
    (strides offsets : List Nat) : Except CheckError Unit :=
  let parentRank := parentShape.length
  let strideRank := strides.length
  let offsetRank := offsets.length
  let blockRank := blockShape.length
  checkBlockPtrMetadataRanks parentRank strideRank offsetRank blockRank

private def nthDNat (xs : List Nat) (i : Nat) : Nat :=
  xs.getD i 0

private def nthDInt (xs : List Int) (i : Nat) : Int :=
  xs.getD i 0

def checkStaticAdvanceNonnegativeAxes
    (offsets : List Nat) (deltas : List Int) : List Nat → Except CheckError Unit
  | [] => .ok ()
  | axis :: rest =>
      let offset := nthDNat offsets axis
      let delta := nthDInt deltas axis
      if BlockPtr.advanceAxisNonnegative offsets deltas axis then
        checkStaticAdvanceNonnegativeAxes offsets deltas rest
      else
        .error (.blockPointerAdvanceUnderflow axis offset delta)

def checkStaticAdvanceNonnegative (offsets : List Nat) (deltas : List Int) :
    Except CheckError Unit :=
  checkStaticAdvanceNonnegativeAxes offsets deltas
    (List.range (max offsets.length deltas.length))

theorem checkBoundaryAxes_ok {rank : Nat} {axes : List Nat}
    (h : checkBoundaryAxes rank axes = .ok ()) :
    ∀ axis, axis ∈ axes → axis < rank := by
  induction axes with
  | nil =>
      intro axis hmem
      cases hmem
  | cons head tail ih =>
      simp [checkBoundaryAxes] at h
      split at h
      · rename_i hHead
        intro axis hmem
        simp at hmem
        cases hmem with
        | inl hEq =>
            subst hEq
            simpa [BlockPtr.axisInRank] using hHead
        | inr hTail =>
            exact ih h axis hTail
      · contradiction

theorem checkBlockPtrMetadataRanks_ok
    {parentRank strideRank offsetRank blockRank : Nat}
    (h : checkBlockPtrMetadataRanks parentRank strideRank offsetRank blockRank = .ok ()) :
    parentRank = strideRank ∧ parentRank = offsetRank ∧ parentRank = blockRank := by
  simp [checkBlockPtrMetadataRanks] at h
  simpa [BlockPtr.metadataRanksValid] using h

theorem checkBlockPtrMetadata_ok
    {parentShape blockShape strides offsets : List Nat}
    (h : checkBlockPtrMetadata parentShape blockShape strides offsets = .ok ()) :
    BlockPtr.MetadataWellFormed parentShape blockShape strides offsets := by
  simpa [checkBlockPtrMetadata, checkBlockPtrMetadataRanks,
    BlockPtr.MetadataWellFormed, BlockPtr.metadataValid] using h

theorem checkStaticAdvanceNonnegativeAxes_ok
    {offsets : List Nat} {deltas : List Int} :
    ∀ {axes : List Nat},
      checkStaticAdvanceNonnegativeAxes offsets deltas axes = .ok () →
        axes.all (BlockPtr.advanceAxisNonnegative offsets deltas) = true
  | [], _ => rfl
  | axis :: rest, h => by
      simp [checkStaticAdvanceNonnegativeAxes] at h
      split at h
      · rename_i hAxis
        simp [hAxis, checkStaticAdvanceNonnegativeAxes_ok h]
      · contradiction

theorem checkStaticAdvanceNonnegative_ok
    {offsets : List Nat} {deltas : List Int}
    (h : checkStaticAdvanceNonnegative offsets deltas = .ok ()) :
    BlockPtr.StaticAdvanceNonnegative offsets deltas := by
  exact checkStaticAdvanceNonnegativeAxes_ok h

namespace BlockPtrSummary

def ofStaticChecked (region : RegionName) (parentShape blockShape strides offsets : List Nat) :
    Except CheckError BlockPtrSummary := do
  checkBlockPtrMetadata parentShape blockShape strides offsets
  .ok { region := region, parentRank := some parentShape.length, offsets := some offsets }

def ofDynamicOffsetsChecked (region : RegionName)
    (parentShape blockShape strides : List Nat) (offsetRank : Nat) :
    Except CheckError BlockPtrSummary := do
  checkBlockPtrMetadataRanks parentShape.length strides.length offsetRank blockShape.length
  .ok { region := region, parentRank := some parentShape.length, offsets := none }

def checkedAdvance (summary : BlockPtrSummary) (deltas : List Int) :
    Except CheckError BlockPtrSummary := do
  match summary.offsets with
  | some offsets =>
      checkStaticAdvanceNonnegative offsets deltas
      .ok { summary with offsets := some (BlockPtr.advanceOffsets offsets deltas) }
  | none => .ok { summary with offsets := none }

def checkBoundary (summary : BlockPtrSummary) (axes : List Nat) :
    Except CheckError Unit :=
  match summary.parentRank with
  | some rank => checkBoundaryAxes rank axes
  | none => .ok ()

def merge (left right : BlockPtrSummary) : Except CheckError BlockPtrSummary :=
  if left.region = right.region then
    .ok
      { region := left.region
      , parentRank := if left.parentRank = right.parentRank then left.parentRank else none
      , offsets := if left.offsets = right.offsets then left.offsets else none }
  else
    .error (.blockPointerProvenanceConflict left.region right.region)

theorem ofStaticChecked_ok {region : RegionName}
    {parentShape blockShape strides offsets : List Nat}
    (h : ofStaticChecked region parentShape blockShape strides offsets = .ok summary) :
    BlockPtr.MetadataWellFormed parentShape blockShape strides offsets ∧
      summary.region = region ∧
      summary.parentRank = some parentShape.length ∧
      summary.offsets = some offsets := by
  simp [ofStaticChecked] at h
  cases hCheck : checkBlockPtrMetadata parentShape blockShape strides offsets with
  | ok u =>
      cases u
      simp [hCheck] at h
      cases h
      exact ⟨checkBlockPtrMetadata_ok hCheck, rfl, rfl, rfl⟩
  | error err =>
      rw [hCheck] at h
      cases h

theorem ofDynamicOffsetsChecked_ok {region : RegionName}
    {parentShape blockShape strides : List Nat} {offsetRank : Nat}
    (h : ofDynamicOffsetsChecked region parentShape blockShape strides offsetRank = .ok summary) :
    parentShape.length = strides.length ∧
      parentShape.length = offsetRank ∧
      parentShape.length = blockShape.length ∧
      summary.region = region ∧
      summary.parentRank = some parentShape.length ∧
      summary.offsets = none := by
  simp [ofDynamicOffsetsChecked] at h
  cases hCheck :
      checkBlockPtrMetadataRanks parentShape.length strides.length offsetRank blockShape.length with
  | ok u =>
      cases u
      simp [hCheck] at h
      cases h
      rcases checkBlockPtrMetadataRanks_ok hCheck with ⟨hStride, hOffset, hBlock⟩
      exact ⟨hStride, hOffset, hBlock, rfl, rfl, rfl⟩
  | error err =>
      rw [hCheck] at h
      cases h

theorem checkedAdvance_ok
    (h : checkedAdvance summary deltas = .ok summary') :
    match summary.offsets with
    | some offsets =>
        BlockPtr.StaticAdvanceNonnegative offsets deltas ∧
          summary'.region = summary.region ∧
          summary'.parentRank = summary.parentRank ∧
          summary'.offsets = some (BlockPtr.advanceOffsets offsets deltas)
    | none =>
        summary'.region = summary.region ∧
          summary'.parentRank = summary.parentRank ∧
          summary'.offsets = none := by
  cases summary with
  | mk region parentRank offsets? =>
      cases offsets? with
      | none =>
          simp [checkedAdvance] at h
          subst h
          simp
      | some offsets =>
          simp [checkedAdvance] at h
          cases hCheck : checkStaticAdvanceNonnegative offsets deltas with
          | ok u =>
              cases u
              simp [hCheck] at h
              cases h
              exact ⟨checkStaticAdvanceNonnegative_ok hCheck, rfl, rfl, rfl⟩
          | error err =>
              rw [hCheck] at h
              cases h

theorem checkBoundary_ok
    (h : checkBoundary summary axes = .ok ()) :
    match summary.parentRank with
    | some rank => ∀ axis, axis ∈ axes → axis < rank
    | none => True := by
  cases summary with
  | mk region parentRank offsets =>
      cases parentRank with
      | none =>
          simp
      | some rank =>
          simp [checkBoundary] at h
          exact checkBoundaryAxes_ok h

theorem merge_ok
    (h : merge left right = .ok summary) :
    left.region = right.region ∧ summary.region = left.region := by
  simp [merge] at h
  split at h
  · rename_i hRegion
    cases h
    exact ⟨hRegion, rfl⟩
  · contradiction

end BlockPtrSummary

mutual

def Op.check (ctx : CheckCtx) : Op dtype shape → Except CheckError Unit
  | .const _ => .ok ()
  | .constFloat _ _ => .ok ()
  | .constNat _ => .ok ()
  | .constInt _ => .ok ()
  | .constBool _ => .ok ()
  | .negInf => .ok ()
  | .programId _ => .ok ()
  | .numPrograms _ => .ok ()
  | .ref dtype shape name => do
      let _ ← ctx.checkRegRef name dtype shape
      .ok ()
  | .arange _ => .ok ()
  | .broadcast e _ => e.check ctx
  | .full _ e => e.check ctx
  | .castFloat _ _ e => e.check ctx
  | .castNatToInt e => e.check ctx
  | .castIntToNat e => e.check ctx
  | .castRealToInt8 e => e.check ctx
  | .add _ _ a b => a.check ctx *> b.check ctx
  | .sub _ _ a b => a.check ctx *> b.check ctx
  | .mul _ _ a b => a.check ctx *> b.check ctx
  | .div _ _ a b => a.check ctx *> b.check ctx
  | .floorDiv _ _ a b => a.check ctx *> b.check ctx
  | .mod _ _ a b => a.check ctx *> b.check ctx
  | .bitAnd _ a b => a.check ctx *> b.check ctx
  | .bitOr _ a b => a.check ctx *> b.check ctx
  | .bitXor _ a b => a.check ctx *> b.check ctx
  | .shiftLeft _ a b => a.check ctx *> b.check ctx
  | .shiftRight _ a b => a.check ctx *> b.check ctx
  | .exp a => a.check ctx
  | .exp2 a => a.check ctx
  | .log a => a.check ctx
  | .log2 a => a.check ctx
  | .sigmoid a => a.check ctx
  | .sqrt a => a.check ctx
  | .rsqrt a => a.check ctx
  | .tanh a => a.check ctx
  | .sin a => a.check ctx
  | .cos a => a.check ctx
  | .tan a => a.check ctx
  | .atan a => a.check ctx
  | .cosh a => a.check ctx
  | .sinh a => a.check ctx
  | .erf a => a.check ctx
  | .lt _ _ a b => a.check ctx *> b.check ctx
  | .le _ _ a b => a.check ctx *> b.check ctx
  | .eq _ _ a b => a.check ctx *> b.check ctx
  | .gt _ _ a b => a.check ctx *> b.check ctx
  | .ge _ _ a b => a.check ctx *> b.check ctx
  | .ne _ _ a b => a.check ctx *> b.check ctx
  | .boolAnd _ a b => a.check ctx *> b.check ctx
  | .boolOr _ a b => a.check ctx *> b.check ctx
  | .boolNot a => a.check ctx
  | .max2 _ a b => a.check ctx *> b.check ctx
  | .pow _ a b => a.check ctx *> b.check ctx
  | .where c a b => c.check ctx *> a.check ctx *> b.check ctx
  | .ite c a b => c.check ctx *> a.check ctx *> b.check ctx
  | .reduceMax _ _ a => a.check ctx
  | .reduceMaxNat _ _ a => a.check ctx
  | .reduceSum _ _ a => a.check ctx
  | .scan _ _ _ a => a.check ctx
  | .argMax _ a => a.check ctx
  | .argMin _ a => a.check ctx
  | .sort _ a => a.check ctx
  | .dot a b => a.check ctx *> b.check ctx
  | .transpose a => a.check ctx
  | .reshape _ a => a.check ctx
  | .remap _ _ a => a.check ctx
  | .join a b => a.check ctx *> b.check ctx
  | .split _ a => a.check ctx
  | .expandDim _ a => a.check ctx
  | .ptrBase _ => .ok ()
  | .ptrAdd _ ptr off => ptr.check ctx *> off.check ctx
  | .ptrSub _ ptr off => ptr.check ctx *> off.check ctx
  | .makeBlockPtr _ _ parentShape blockShape strides offsets =>
      checkBlockPtrMetadata parentShape blockShape strides offsets
  | .makeBlockPtrDyn _ base parentShape blockShape strides offsets =>
      base.check ctx *> checkBlockPtrMetadata parentShape blockShape strides offsets
  | .makeBlockPtrDynOffsets _ base parentShape blockShape strides offsets =>
      base.check ctx *>
      checkBlockPtrMetadataRanks parentShape.length strides.length offsets.length blockShape.length
  | .advanceBlockPtr ptr deltas => do
      ptr.check ctx
      .ok ()
  | .load dtype mem mask => mem.check ctx dtype *> mask.check ctx
  | .natToReal a => a.check ctx
  | .intToReal a => a.check ctx
termination_by op => sizeOf op
decreasing_by
  all_goals
    simp_wf
    try omega

def Op.ptrProvenance (ctx : CheckCtx) : Op dtype shape → Except CheckError RegionName
  | .ptrBase region => .ok (Region.cast region)
  | .ptrAdd _ ptr off => do
      let region ← ptr.ptrProvenance ctx
      off.check ctx
      .ok region
  | .ptrSub _ ptr off => do
      let region ← ptr.ptrProvenance ctx
      off.check ctx
      .ok region
  | .broadcast ptr _ => ptr.ptrProvenance ctx
  | .full _ ptr => ptr.ptrProvenance ctx
  | .where c a b => do
      c.check ctx
      let left ← a.ptrProvenance ctx
      let right ← b.ptrProvenance ctx
      if left = right then .ok left else .error (.pointerProvenanceConflict left right)
  | .ite c a b => do
      c.check ctx
      let left ← a.ptrProvenance ctx
      let right ← b.ptrProvenance ctx
      if left = right then .ok left else .error (.pointerProvenanceConflict left right)
  | .transpose ptr => ptr.ptrProvenance ctx
  | .expandDim _ ptr => ptr.ptrProvenance ctx
  | .ref .ptr shape name => do
      let entry ← ctx.checkRegRef name .ptr shape
      match entry.ptrProvenance with
      | some region => .ok region
      | none => .error (.missingPointerProvenance name)
  | _ => .error .unsupportedPointerProvenance
termination_by op => sizeOf op
decreasing_by
  all_goals
    simp_wf
    try omega

def Op.blockPtrSummary (ctx : CheckCtx) :
    Op dtype shape → Except CheckError BlockPtrSummary
  | .makeBlockPtr region _ parentShape blockShape strides offsets =>
      BlockPtrSummary.ofStaticChecked
        (Region.cast region) parentShape blockShape strides offsets
  | .makeBlockPtrDyn region base parentShape blockShape strides offsets => do
      base.check ctx
      BlockPtrSummary.ofStaticChecked region parentShape blockShape strides offsets
  | .makeBlockPtrDynOffsets region base parentShape blockShape strides offsets => do
      base.check ctx
      BlockPtrSummary.ofDynamicOffsetsChecked region parentShape blockShape strides offsets.length
  | .advanceBlockPtr ptr deltas => do
      let summary ← ptr.blockPtrSummary ctx
      summary.checkedAdvance deltas
  | .broadcast ptr _ => ptr.blockPtrSummary ctx
  | .full _ ptr => ptr.blockPtrSummary ctx
  | .where c a b => do
      c.check ctx
      let left ← a.blockPtrSummary ctx
      let right ← b.blockPtrSummary ctx
      BlockPtrSummary.merge left right
  | .ite c a b => do
      c.check ctx
      let left ← a.blockPtrSummary ctx
      let right ← b.blockPtrSummary ctx
      BlockPtrSummary.merge left right
  | .transpose ptr => ptr.blockPtrSummary ctx
  | .expandDim _ ptr => ptr.blockPtrSummary ctx
  | .ref .blockPtr shape name => do
      let entry ← ctx.checkRegRef name .blockPtr shape
      match entry.blockPtrSummary with
      | some summary => .ok summary
      | none => .error (.missingBlockPointerProvenance name)
  | _ => .error .unsupportedBlockPointerProvenance
termination_by op => sizeOf op
decreasing_by all_goals (simp_wf; try omega)

def Op.blockPtrProvenance (ctx : CheckCtx) :
    Op dtype shape → Except CheckError RegionName := fun op => do
  let summary ← op.blockPtrSummary ctx
  .ok summary.region

def Op.blockPtrParentRank? (ctx : CheckCtx) :
    Op dtype shape → Except CheckError (Option Nat) := fun op => do
  let summary ← op.blockPtrSummary ctx
  .ok summary.parentRank

def Op.blockPtrOffsets? (ctx : CheckCtx) :
    Op dtype shape → Except CheckError (Option (List Nat)) := fun op => do
  let summary ← op.blockPtrSummary ctx
  .ok summary.offsets

def MemAccess.check (ctx : CheckCtx) (dtype : TileDType) :
    MemAccess dtype shape → Except CheckError Unit
  | .region region off => off.check ctx *> ctx.checkRegion (Region.cast region) dtype
  | .ptr ptr => do
      let region ← ptr.ptrProvenance ctx
      ctx.checkRegion region dtype
  | .blockPtr ptr boundaryCheck => do
      let summary ← ptr.blockPtrSummary ctx
      checkBoundaryAxes shape.length boundaryCheck
      summary.checkBoundary boundaryCheck
      ctx.checkRegion summary.region dtype
termination_by mem => sizeOf mem
decreasing_by all_goals (simp_wf; try omega)

def MaskOpt.check (ctx : CheckCtx) : MaskOpt dtype shape → Except CheckError Unit
  | .none => .ok ()
  | .mask mask => mask.check ctx
  | .maskOther mask other => mask.check ctx *> other.check ctx
termination_by mask => sizeOf mask
decreasing_by all_goals (simp_wf; try omega)

end

def pointerEntry (dtype : TileDType) (shape : TileShape)
    (ptrProv : Option RegionName)
    (blockSummary : Option BlockPtrSummary := none) : RegEntry :=
  { dtype := dtype, shape := shape
  , ptrProvenance := ptrProv, blockPtrSummary := blockSummary }

def Stmt.assignEntry (ctx : CheckCtx) (dtype : TileDType) (shape : TileShape)
    (e : Op dtype shape) : Except CheckError RegEntry :=
  match dtype with
  | .ptr => do
      let region ← e.ptrProvenance ctx
      .ok (pointerEntry .ptr shape (some region))
  | .blockPtr => do
      let summary ← e.blockPtrSummary ctx
      .ok (pointerEntry .blockPtr shape none (some summary))
  | dtype => .ok (pointerEntry dtype shape none)

mutual

def Stmt.check (ctx : CheckCtx) : Stmt → Except CheckError CheckCtx
  | .assign dtype shape name e => do
      e.check ctx
      let entry ← Stmt.assignEntry ctx dtype shape e
      ctx.setReg name entry
  | .store dtype _ mem val mask => do
      val.check ctx
      mask.check ctx
      mem.check ctx dtype
      .ok ctx
  | @Stmt.atomicAdd dtype _ _ mem val mask => do
      val.check ctx
      mask.check ctx
      mem.check ctx dtype
      .ok ctx
  | .atomicRMW _ dtype shape mem input extraInput mask dest => do
      input.check ctx
      match extraInput with
      | none => pure ()
      | some extra => extra.check ctx
      mask.check ctx
      mem.check ctx dtype
      match dest with
      | none => .ok ctx
      | some name => ctx.setReg name (pointerEntry dtype shape none)
  | .forLoop idx _ body => do
      let ctx' ← ctx.setReg idx (pointerEntry .nat [] none)
      StmtList.check ctx' body
  | .forRange idx _ _ _ body => do
      let ctx' ← ctx.setReg idx (pointerEntry .nat [] none)
      StmtList.check ctx' body
  | .forRangeDyn idx start stop step body => do
      start.check ctx
      stop.check ctx
      step.check ctx
      let ctx' ← ctx.setReg idx (pointerEntry .nat [] none)
      StmtList.check ctx' body
  | .ifThen cond body => do
      cond.check ctx
      StmtList.check ctx body
  | .ifThenElse cond thenBody elseBody => do
      cond.check ctx
      let _ ← StmtList.check ctx thenBody
      let _ ← StmtList.check ctx elseBody
      .ok ctx
termination_by st => sizeOf st
decreasing_by all_goals (simp_wf; try omega)

def StmtList.check (ctx : CheckCtx) : List Stmt → Except CheckError CheckCtx
  | [] => .ok ctx
  | st :: rest => do
      let ctx' ← st.check ctx
      StmtList.check ctx' rest
termination_by body => sizeOf body
decreasing_by all_goals (simp_wf; try omega)

end

namespace Kernel

def checkWith (mode : CheckMode) (Γ : RegionEnv) (k : Kernel) :
    Except CheckError Unit := do
  let _ ← StmtList.check { mode := mode, regions := Γ } k.body
  .ok ()

def check (Γ : RegionEnv) (k : Kernel) : Except CheckError Unit :=
  checkWith .lax Γ k

def checkStrict (Γ : RegionEnv) (k : Kernel) : Except CheckError Unit :=
  checkWith .strict Γ k

end Kernel

end VeriTile.Triton
