/-
VeriTile.Triton.Float.Erasure

Erasure from explicit floating dtype annotations to the mathematical real
channel used by the current Triton semantics.
-/

import VeriTile.Triton.Core

namespace VeriTile.Triton

/-! ## Floating-dtype erasure -/

/-- Erase hardware-shaped floating channels to the mathematical real channel. -/
def eraseDType : TileDType → TileDType
  | .fp32 => .real
  | .fp16 => .real
  | .bf16 => .real
  | dtype => dtype

/-- Erase a floating dtype tag to the real channel. -/
def FloatDType.eraseFloat (_ : FloatDType) : FloatDType :=
  .real

@[simp] theorem eraseDType_float (dtype : FloatDType) :
    VeriTile.Triton.eraseDType dtype.toTileDType = .real := by
  cases dtype <;> rfl

namespace NumericDType

/-- Numeric witness after floating-dtype erasure. -/
def eraseDType : NumericDType dtype → NumericDType (VeriTile.Triton.eraseDType dtype)
  | .real => .real
  | .fp32 => .real
  | .fp16 => .real
  | .bf16 => .real
  | .int => .int
  | .nat => .nat

end NumericDType

namespace IntegralDType

/-- Integral witness after floating-dtype erasure. -/
def eraseDType : IntegralDType dtype → IntegralDType (VeriTile.Triton.eraseDType dtype)
  | .int => .int
  | .nat => .nat

end IntegralDType

namespace ComparableDType

/-- Comparable witness after floating-dtype erasure. -/
def eraseDType : ComparableDType dtype → ComparableDType (VeriTile.Triton.eraseDType dtype)
  | .real => .real
  | .fp32 => .real
  | .fp16 => .real
  | .bf16 => .real
  | .int => .int
  | .nat => .nat

end ComparableDType

mutual

/-- Erase explicit floating dtype annotations from an expression. -/
def Op.eraseDType : Op dtype shape → Op (VeriTile.Triton.eraseDType dtype) shape
  | .const c => .const c
  | .constFloat h c => by
      rw [eraseDType_float h]
      exact .const c
  | .constNat n => .constNat n
  | .constInt n => .constInt n
  | .constBool b => .constBool b
  | .negInf => .negInf
  | .programId axis => .programId axis
  | .ref dtype shape name => .ref (VeriTile.Triton.eraseDType dtype) shape name
  | .arange n => .arange n
  | .broadcast e shape => .broadcast e.eraseDType shape
  | .full shape e => .full shape e.eraseDType
  | .castFloat src dst e => by
      simpa [eraseDType_float src, eraseDType_float dst] using e.eraseDType
  | .add h bc a b => .add h.eraseDType bc a.eraseDType b.eraseDType
  | .sub h bc a b => .sub h.eraseDType bc a.eraseDType b.eraseDType
  | .mul h bc a b => .mul h.eraseDType bc a.eraseDType b.eraseDType
  | .div h bc a b => .div h.eraseDType bc a.eraseDType b.eraseDType
  | .floorDiv h bc a b => .floorDiv h.eraseDType bc a.eraseDType b.eraseDType
  | .mod h bc a b => .mod h.eraseDType bc a.eraseDType b.eraseDType
  | .bitAnd bc a b => .bitAnd bc a.eraseDType b.eraseDType
  | .bitOr bc a b => .bitOr bc a.eraseDType b.eraseDType
  | .bitXor bc a b => .bitXor bc a.eraseDType b.eraseDType
  | .shiftLeft bc a b => .shiftLeft bc a.eraseDType b.eraseDType
  | .shiftRight bc a b => .shiftRight bc a.eraseDType b.eraseDType
  | .exp a => .exp a.eraseDType
  | .exp2 a => .exp2 a.eraseDType
  | .log a => .log a.eraseDType
  | .log2 a => .log2 a.eraseDType
  | .sigmoid a => .sigmoid a.eraseDType
  | .sqrt a => .sqrt a.eraseDType
  | .rsqrt a => .rsqrt a.eraseDType
  | .tanh a => .tanh a.eraseDType
  | .sin a => .sin a.eraseDType
  | .cos a => .cos a.eraseDType
  | .tan a => .tan a.eraseDType
  | .atan a => .atan a.eraseDType
  | .cosh a => .cosh a.eraseDType
  | .sinh a => .sinh a.eraseDType
  | .erf a => .erf a.eraseDType
  | .lt h bc a b => .lt h.eraseDType bc a.eraseDType b.eraseDType
  | .le h bc a b => .le h.eraseDType bc a.eraseDType b.eraseDType
  | .eq h bc a b => .eq h.eraseDType bc a.eraseDType b.eraseDType
  | .gt h bc a b => .gt h.eraseDType bc a.eraseDType b.eraseDType
  | .ge h bc a b => .ge h.eraseDType bc a.eraseDType b.eraseDType
  | .ne h bc a b => .ne h.eraseDType bc a.eraseDType b.eraseDType
  | .boolAnd bc a b => .boolAnd bc a.eraseDType b.eraseDType
  | .boolOr bc a b => .boolOr bc a.eraseDType b.eraseDType
  | .boolNot a => .boolNot a.eraseDType
  | .max2 bc a b => .max2 bc a.eraseDType b.eraseDType
  | .where c a b => .where c.eraseDType a.eraseDType b.eraseDType
  | .reduceMax axis keepDims a => .reduceMax axis keepDims a.eraseDType
  | .reduceSum axis keepDims a => .reduceSum axis keepDims a.eraseDType
  | .scan op axis a => .scan op axis a.eraseDType
  | .argMax axis a => .argMax axis a.eraseDType
  | .argMin axis a => .argMin axis a.eraseDType
  | .sort axis a => .sort axis a.eraseDType
  | .dot a b => .dot a.eraseDType b.eraseDType
  | .transpose a => .transpose a.eraseDType
  | .reshape outShape a => .reshape outShape a.eraseDType
  | .remap outShape map a => .remap outShape map a.eraseDType
  | .join a b => .join a.eraseDType b.eraseDType
  | .split side a => .split side a.eraseDType
  | .expandDim axis a => .expandDim axis a.eraseDType
  | .ptrBase region => .ptrBase region
  | @Op.ptrAdd a b out d bc ptr off =>
      @Op.ptrAdd a b out (VeriTile.Triton.eraseDType d) bc ptr.eraseDType off.eraseDType
  | .makeBlockPtr region baseOffset parentShape blockShape strides offsets =>
      .makeBlockPtr region baseOffset parentShape blockShape strides offsets
<<<<<<< Updated upstream
  | @Op.advanceBlockPtr shape d ptr deltas =>
      @Op.advanceBlockPtr shape (VeriTile.Triton.eraseDType d) ptr.eraseDType deltas
=======
  | .makeBlockPtrDyn region baseOffset parentShape blockShape strides offsets =>
      .makeBlockPtrDyn region baseOffset.eraseDType parentShape blockShape strides offsets
  | .advanceBlockPtr ptr deltas => .advanceBlockPtr ptr.eraseDType deltas
>>>>>>> Stashed changes
  | .load dtype mem mask => .load (VeriTile.Triton.eraseDType dtype) mem.eraseDType mask.eraseDType
  | .natToReal a => .natToReal a.eraseDType
termination_by e => sizeOf e
decreasing_by
  all_goals (simp_wf; try omega; try decreasing_trivial)

/-- Erase explicit floating dtype annotations from a memory access. The
region's phantom dtype is independent of the access dtype, so we keep
the region unchanged and only erase the offset / payload pointers. -/
def MemAccess.eraseDType :
    MemAccess dtype shape → MemAccess (VeriTile.Triton.eraseDType dtype) shape
  | .region region off =>
      .region (Region.cast (d' := VeriTile.Triton.eraseDType dtype) region) off.eraseDType
  | .ptr ptr => .ptr ptr.eraseDType
  | .blockPtr ptr boundaryCheck => .blockPtr ptr.eraseDType boundaryCheck
termination_by mem => sizeOf mem
decreasing_by all_goals (simp_wf; try omega)

/-- Erase explicit floating dtype annotations from a memory mask. -/
def MaskOpt.eraseDType : MaskOpt dtype shape → MaskOpt (VeriTile.Triton.eraseDType dtype) shape
  | .none => .none
  | .mask mask => .mask mask.eraseDType
  | .maskOther mask other => .maskOther mask.eraseDType other.eraseDType
termination_by mask => sizeOf mask
decreasing_by all_goals (simp_wf; try omega)

/-- Erase explicit floating dtype annotations from a statement. -/
def Stmt.eraseDType : Stmt → Stmt
  | .assign dtype shape name e =>
      .assign (VeriTile.Triton.eraseDType dtype) shape name e.eraseDType
  | .store dtype shape mem val mask =>
      .store (VeriTile.Triton.eraseDType dtype) shape mem.eraseDType val.eraseDType mask.eraseDType
  | .atomicAdd h shape mem val mask =>
      .atomicAdd h.eraseDType shape mem.eraseDType val.eraseDType mask.eraseDType
  | .atomicRMW op dtype shape mem input extraInput mask dest =>
      .atomicRMW op (VeriTile.Triton.eraseDType dtype) shape mem.eraseDType
        input.eraseDType (extraInput.map Op.eraseDType) mask.eraseDType dest
  | .forLoop idx n body =>
      .forLoop idx n (Stmt.eraseDTypeList body)
  | .forRange idx start stop step body =>
      .forRange idx start stop step (Stmt.eraseDTypeList body)
<<<<<<< Updated upstream
=======
  | .forRangeDyn idx start stop step body =>
      .forRangeDyn idx start.eraseDType stop.eraseDType step.eraseDType
        (Stmt.eraseDTypeList body)
>>>>>>> Stashed changes
  | .ifThen cond body =>
      .ifThen cond.eraseDType (Stmt.eraseDTypeList body)
  | .ifThenElse cond thenBody elseBody =>
      .ifThenElse cond.eraseDType (Stmt.eraseDTypeList thenBody)
        (Stmt.eraseDTypeList elseBody)
termination_by st => sizeOf st
decreasing_by all_goals (simp_wf; try omega)

/-- Erase explicit floating dtype annotations from a statement list. -/
def Stmt.eraseDTypeList : List Stmt → List Stmt
  | [] => []
  | st :: rest => st.eraseDType :: Stmt.eraseDTypeList rest
termination_by body => sizeOf body
decreasing_by all_goals (simp_wf; try omega)

end

@[simp] theorem MemAccess.eraseDType_region
    {dtype : TileDType} (region : Region dtype) (off : Op .nat shape) :
    (MemAccess.region (d := dtype) region off).eraseDType =
      MemAccess.region (d := VeriTile.Triton.eraseDType dtype)
        (Region.cast (d' := VeriTile.Triton.eraseDType dtype) region) off.eraseDType := by
  rw [MemAccess.eraseDType.eq_def]

@[simp] theorem MemAccess.eraseDType_ptr {dtype : TileDType}
    (ptr : Op .ptr shape) :
    (MemAccess.ptr (d := dtype) ptr).eraseDType =
      MemAccess.ptr (d := VeriTile.Triton.eraseDType dtype) ptr.eraseDType := by
  rw [MemAccess.eraseDType.eq_def]

@[simp] theorem MemAccess.eraseDType_blockPtr {dtype : TileDType}
    (ptr : Op .blockPtr shape) (boundaryCheck : List Nat) :
    (MemAccess.blockPtr (d := dtype) ptr boundaryCheck).eraseDType =
      MemAccess.blockPtr (d := VeriTile.Triton.eraseDType dtype) ptr.eraseDType
        boundaryCheck := by
  rw [MemAccess.eraseDType.eq_def]

@[simp] theorem MaskOpt.eraseDType_none {dtype : TileDType} :
    (MaskOpt.none : MaskOpt dtype shape).eraseDType = MaskOpt.none := by
  rw [MaskOpt.eraseDType.eq_def]

@[simp] theorem MaskOpt.eraseDType_mask
    (mask : Op .bool shape) :
    (MaskOpt.mask (dtype := dtype) mask).eraseDType =
      MaskOpt.mask mask.eraseDType := by
  rw [MaskOpt.eraseDType.eq_def]

@[simp] theorem MaskOpt.eraseDType_maskOther
    (mask : Op .bool shape) (other : Op dtype shape) :
    (MaskOpt.maskOther mask other).eraseDType =
      MaskOpt.maskOther mask.eraseDType other.eraseDType := by
  rw [MaskOpt.eraseDType.eq_def]

@[simp] theorem Op.eraseDType_ref
    (dtype : TileDType) (shape : TileShape) (name : RegName) :
    (Op.ref dtype shape name).eraseDType =
      Op.ref (VeriTile.Triton.eraseDType dtype) shape name := by
  rw [Op.eraseDType.eq_def]

@[simp] theorem Op.eraseDType_castFloat
    (src dst : FloatDType) (e : Op src.toTileDType shape) :
    (Op.castFloat src dst e).eraseDType =
      cast (by simp [eraseDType_float src, eraseDType_float dst])
        e.eraseDType := by
  rw [Op.eraseDType.eq_def]
  simp [VeriTile.Triton.eraseDType]

@[simp] theorem Op.eraseDType_load
    (dtype : TileDType) (mem : MemAccess dtype shape)
    (mask : MaskOpt dtype shape) :
    (Op.load dtype mem mask).eraseDType =
      Op.load (VeriTile.Triton.eraseDType dtype) mem.eraseDType mask.eraseDType := by
  rw [Op.eraseDType.eq_def]

@[simp] theorem Op.eraseDType_reduceMax
    (axis : Fin shape.length) (keepDims : Bool) (e : Op .real shape) :
    (Op.reduceMax axis keepDims e).eraseDType =
      cast (by simp [VeriTile.Triton.eraseDType])
        (@Op.reduceMax shape axis keepDims
          (cast (by simp [VeriTile.Triton.eraseDType]) e.eraseDType)) := by
  rw [Op.eraseDType.eq_def]
  simp [VeriTile.Triton.eraseDType]

@[simp] theorem Op.eraseDType_reduceSum
    (axis : Fin shape.length) (keepDims : Bool) (e : Op .real shape) :
    (Op.reduceSum axis keepDims e).eraseDType =
      cast (by simp [VeriTile.Triton.eraseDType])
        (@Op.reduceSum shape axis keepDims
          (cast (by simp [VeriTile.Triton.eraseDType]) e.eraseDType)) := by
  rw [Op.eraseDType.eq_def]
  simp [VeriTile.Triton.eraseDType]

@[simp] theorem Stmt.eraseDType_store
    (dtype : TileDType) (shape : TileShape)
    (mem : MemAccess dtype shape) (val : Op dtype shape)
    (mask : MaskOpt dtype shape) :
    (Stmt.store dtype shape mem val mask).eraseDType =
      Stmt.store (VeriTile.Triton.eraseDType dtype) shape mem.eraseDType val.eraseDType mask.eraseDType := by
  rw [Stmt.eraseDType.eq_def]

@[simp] theorem Stmt.eraseDType_atomicAdd
    (h : NumericDType dtype) (shape : TileShape)
    (mem : MemAccess dtype shape) (val : Op dtype shape)
    (mask : MaskOpt dtype shape) :
    (Stmt.atomicAdd h shape mem val mask).eraseDType =
      Stmt.atomicAdd h.eraseDType shape mem.eraseDType val.eraseDType mask.eraseDType := by
  rw [Stmt.eraseDType.eq_def]

@[simp] theorem Stmt.eraseDType_atomicRMW
    (op : RMWOp) (dtype : TileDType) (shape : TileShape)
    (mem : MemAccess dtype shape) (input : Op dtype shape)
    (extraInput : Option (Op dtype shape)) (mask : MaskOpt dtype shape)
    (dest : Option RegName) :
    (Stmt.atomicRMW op dtype shape mem input extraInput mask dest).eraseDType =
      Stmt.atomicRMW op (VeriTile.Triton.eraseDType dtype) shape mem.eraseDType
        input.eraseDType (extraInput.map Op.eraseDType) mask.eraseDType dest := by
  rw [Stmt.eraseDType.eq_def]

namespace Kernel

/-- Erase explicit floating dtype annotations from a kernel body. -/
def eraseDType (k : Kernel) : Kernel :=
  { k with body := Stmt.eraseDTypeList k.body }

end Kernel

namespace ComputeKernel

/-- Project a compute-facing kernel to the algorithm layer, then erase
algorithm dtype annotations. This keeps float-facing examples compute-first
while reusing the existing Real proof path. -/
def eraseDType (ck : ComputeKernel) : ComputeKernel :=
  let k := ck.toAlgKernel.eraseDType
  ComputeKernel.fromAlgBody k.inputs k.outputs k.body

end ComputeKernel

end VeriTile.Triton
