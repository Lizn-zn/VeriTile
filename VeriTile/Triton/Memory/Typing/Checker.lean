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
  deriving DecidableEq, Repr

structure RegEntry where
  dtype : TileDType
  shape : TileShape
  ptrProvenance : Option RegionName := none
  blockPtrProvenance : Option RegionName := none
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
      if axis < rank then
        checkBoundaryAxes rank rest
      else
        .error (.boundaryAxisOutOfRange axis rank)

def checkBlockPtrMetadata (parentShape : List Nat) (blockShape : TileShape)
    (strides offsets : List Nat) : Except CheckError Unit :=
  let parentRank := parentShape.length
  let strideRank := strides.length
  let offsetRank := offsets.length
  let blockRank := blockShape.length
  if parentRank = strideRank ∧ parentRank = offsetRank ∧ blockRank ≤ parentRank then
    .ok ()
  else
    .error (.blockPointerMetadataMismatch parentRank strideRank offsetRank blockRank)

mutual

def Op.check (ctx : CheckCtx) : Op dtype shape → Except CheckError Unit
  | .const _ => .ok ()
  | .constFloat _ _ => .ok ()
  | .constNat _ => .ok ()
  | .constInt _ => .ok ()
  | .constBool _ => .ok ()
  | .negInf => .ok ()
  | .programId _ => .ok ()
  | .ref dtype shape name => do
      let _ ← ctx.checkRegRef name dtype shape
      .ok ()
  | .arange _ => .ok ()
  | .broadcast e _ => e.check ctx
  | .full _ e => e.check ctx
  | .castFloat _ _ e => e.check ctx
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
  | .where c a b => c.check ctx *> a.check ctx *> b.check ctx
  | .reduceMax _ _ a => a.check ctx
  | .reduceSum _ _ a => a.check ctx
  | .scan _ _ a => a.check ctx
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
  | .makeBlockPtr _ _ parentShape blockShape strides offsets =>
      checkBlockPtrMetadata parentShape blockShape strides offsets
  | .advanceBlockPtr ptr _ => ptr.check ctx
  | .load dtype mem mask => mem.check ctx dtype *> mask.check ctx
  | .natToReal a => a.check ctx
termination_by op => sizeOf op
decreasing_by all_goals (simp_wf; try omega)

def Op.ptrProvenance (ctx : CheckCtx) : Op dtype shape → Except CheckError RegionName
  | .ptrBase region => .ok (Region.cast region)
  | .ptrAdd _ ptr off => do
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
  | .transpose ptr => ptr.ptrProvenance ctx
  | .expandDim _ ptr => ptr.ptrProvenance ctx
  | .ref .ptr shape name => do
      let entry ← ctx.checkRegRef name .ptr shape
      match entry.ptrProvenance with
      | some region => .ok region
      | none => .error (.missingPointerProvenance name)
  | _ => .error .unsupportedPointerProvenance
termination_by op => sizeOf op
decreasing_by all_goals (simp_wf; try omega)

def Op.blockPtrProvenance (ctx : CheckCtx) :
    Op dtype shape → Except CheckError RegionName
  | .makeBlockPtr region _ parentShape blockShape strides offsets => do
      checkBlockPtrMetadata parentShape blockShape strides offsets
      .ok (Region.cast region)
  | .advanceBlockPtr ptr _ => ptr.blockPtrProvenance ctx
  | .broadcast ptr _ => ptr.blockPtrProvenance ctx
  | .full _ ptr => ptr.blockPtrProvenance ctx
  | .where c a b => do
      c.check ctx
      let left ← a.blockPtrProvenance ctx
      let right ← b.blockPtrProvenance ctx
      if left = right then .ok left else .error (.blockPointerProvenanceConflict left right)
  | .transpose ptr => ptr.blockPtrProvenance ctx
  | .expandDim _ ptr => ptr.blockPtrProvenance ctx
  | .ref .blockPtr shape name => do
      let entry ← ctx.checkRegRef name .blockPtr shape
      match entry.blockPtrProvenance with
      | some region => .ok region
      | none => .error (.missingBlockPointerProvenance name)
  | _ => .error .unsupportedBlockPointerProvenance
termination_by op => sizeOf op
decreasing_by all_goals (simp_wf; try omega)

def MemAccess.check (ctx : CheckCtx) (dtype : TileDType) :
    MemAccess dtype shape → Except CheckError Unit
  | .region region off => off.check ctx *> ctx.checkRegion (Region.cast region) dtype
  | .ptr ptr => do
      let region ← ptr.ptrProvenance ctx
      ctx.checkRegion region dtype
  | .blockPtr ptr boundaryCheck => do
      let region ← ptr.blockPtrProvenance ctx
      checkBoundaryAxes shape.length boundaryCheck
      ctx.checkRegion region dtype
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
    (ptrProv blockProv : Option RegionName) : RegEntry :=
  { dtype := dtype, shape := shape
  , ptrProvenance := ptrProv, blockPtrProvenance := blockProv }

def Stmt.assignEntry (ctx : CheckCtx) (dtype : TileDType) (shape : TileShape)
    (e : Op dtype shape) : Except CheckError RegEntry :=
  match dtype with
  | .ptr => do
      let region ← e.ptrProvenance ctx
      .ok (pointerEntry .ptr shape (some region) none)
  | .blockPtr => do
      let region ← e.blockPtrProvenance ctx
      .ok (pointerEntry .blockPtr shape none (some region))
  | dtype => .ok (pointerEntry dtype shape none none)

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
      | some name => ctx.setReg name (pointerEntry dtype shape none none)
  | .forLoop idx _ body => do
      let ctx' ← ctx.setReg idx (pointerEntry .nat [] none none)
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
