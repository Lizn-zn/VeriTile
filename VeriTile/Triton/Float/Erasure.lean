/-
VeriTile.Triton.Float.Erasure

Erasure from explicit floating dtype annotations to the mathematical real
channel used by the current Triton semantics.
-/

import VeriTile.Triton.Core

namespace VeriTile.Triton

/-! ## Floating-dtype erasure -/

/-- Erase hardware-shaped floating channels to the mathematical real channel. -/
def eraseFloatDType : TileDType → TileDType
  | .fp32 => .real
  | .fp16 => .real
  | .bf16 => .real
  | dtype => dtype

/-- Erase a floating dtype tag to the real channel. -/
def FloatDType.eraseFloat (_ : FloatDType) : FloatDType :=
  .real

@[simp] theorem eraseFloatDType_float (dtype : FloatDType) :
    eraseFloatDType dtype.toTileDType = .real := by
  cases dtype <;> rfl

namespace NumericDType

/-- Numeric witness after floating-dtype erasure. -/
def eraseFloat : NumericDType dtype → NumericDType (eraseFloatDType dtype)
  | .real => .real
  | .fp32 => .real
  | .fp16 => .real
  | .bf16 => .real
  | .int => .int
  | .nat => .nat

end NumericDType

namespace IntegralDType

/-- Integral witness after floating-dtype erasure. -/
def eraseFloat : IntegralDType dtype → IntegralDType (eraseFloatDType dtype)
  | .int => .int
  | .nat => .nat

end IntegralDType

namespace ComparableDType

/-- Comparable witness after floating-dtype erasure. -/
def eraseFloat : ComparableDType dtype → ComparableDType (eraseFloatDType dtype)
  | .real => .real
  | .fp32 => .real
  | .fp16 => .real
  | .bf16 => .real
  | .int => .int
  | .nat => .nat

end ComparableDType

mutual

/-- Erase explicit floating dtype annotations from an expression. -/
def Op.eraseFloat : Op dtype shape → Op (eraseFloatDType dtype) shape
  | .const c => .const c
  | .constFloat h c => by
      rw [eraseFloatDType_float h]
      exact .const c
  | .constNat n => .constNat n
  | .constBool b => .constBool b
  | .negInf => .negInf
  | .programId axis => .programId axis
  | .ref dtype shape name => .ref (eraseFloatDType dtype) shape name
  | .arange n => .arange n
  | .broadcast e shape => .broadcast e.eraseFloat shape
  | .full shape e => .full shape e.eraseFloat
  | .castFloat src dst e => by
      simpa [eraseFloatDType_float src, eraseFloatDType_float dst] using e.eraseFloat
  | .add h bc a b => .add h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .sub h bc a b => .sub h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .mul h bc a b => .mul h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .div h bc a b => .div h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .floorDiv h bc a b => .floorDiv h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .mod h bc a b => .mod h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .bitAnd bc a b => .bitAnd bc a.eraseFloat b.eraseFloat
  | .bitOr bc a b => .bitOr bc a.eraseFloat b.eraseFloat
  | .bitXor bc a b => .bitXor bc a.eraseFloat b.eraseFloat
  | .shiftLeft bc a b => .shiftLeft bc a.eraseFloat b.eraseFloat
  | .shiftRight bc a b => .shiftRight bc a.eraseFloat b.eraseFloat
  | .exp a => .exp a.eraseFloat
  | .exp2 a => .exp2 a.eraseFloat
  | .log a => .log a.eraseFloat
  | .log2 a => .log2 a.eraseFloat
  | .sigmoid a => .sigmoid a.eraseFloat
  | .sqrt a => .sqrt a.eraseFloat
  | .tanh a => .tanh a.eraseFloat
  | .sin a => .sin a.eraseFloat
  | .cos a => .cos a.eraseFloat
  | .tan a => .tan a.eraseFloat
  | .atan a => .atan a.eraseFloat
  | .cosh a => .cosh a.eraseFloat
  | .sinh a => .sinh a.eraseFloat
  | .lt h bc a b => .lt h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .le h bc a b => .le h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .eq h bc a b => .eq h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .gt h bc a b => .gt h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .ge h bc a b => .ge h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .ne h bc a b => .ne h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .boolAnd bc a b => .boolAnd bc a.eraseFloat b.eraseFloat
  | .boolOr bc a b => .boolOr bc a.eraseFloat b.eraseFloat
  | .boolNot a => .boolNot a.eraseFloat
  | .max2 bc a b => .max2 bc a.eraseFloat b.eraseFloat
  | .where c a b => .where c.eraseFloat a.eraseFloat b.eraseFloat
  | .reduceMax axis keepDims a => .reduceMax axis keepDims a.eraseFloat
  | .reduceSum axis keepDims a => .reduceSum axis keepDims a.eraseFloat
  | .dot a b => .dot a.eraseFloat b.eraseFloat
  | .transpose a => .transpose a.eraseFloat
  | .expandDim axis a => .expandDim axis a.eraseFloat
  | .ptrBase region => .ptrBase region
  | .ptrAdd bc ptr off => .ptrAdd bc ptr.eraseFloat off.eraseFloat
  | .makeBlockPtr region baseOffset parentShape blockShape strides offsets =>
      .makeBlockPtr region baseOffset parentShape blockShape strides offsets
  | .advanceBlockPtr ptr deltas => .advanceBlockPtr ptr.eraseFloat deltas
  | .load dtype mem mask => .load (eraseFloatDType dtype) mem.eraseFloat mask.eraseFloat
  | .natToReal a => .natToReal a.eraseFloat
termination_by e => sizeOf e
decreasing_by
  all_goals (simp_wf; try omega; try decreasing_trivial)

/-- Erase explicit floating dtype annotations from a memory access. -/
def MemAccess.eraseFloat : MemAccess shape → MemAccess shape
  | .region region off => .region region off.eraseFloat
  | .ptr ptr => .ptr ptr.eraseFloat
  | .blockPtr ptr boundaryCheck => .blockPtr ptr.eraseFloat boundaryCheck
termination_by mem => sizeOf mem
decreasing_by all_goals (simp_wf; try omega)

/-- Erase explicit floating dtype annotations from a memory mask. -/
def MaskOpt.eraseFloat : MaskOpt dtype shape → MaskOpt (eraseFloatDType dtype) shape
  | .none => .none
  | .mask mask => .mask mask.eraseFloat
  | .maskOther mask other => .maskOther mask.eraseFloat other.eraseFloat
termination_by mask => sizeOf mask
decreasing_by all_goals (simp_wf; try omega)

/-- Erase explicit floating dtype annotations from a statement. -/
def Stmt.eraseFloat : Stmt → Stmt
  | .assign dtype shape name e =>
      .assign (eraseFloatDType dtype) shape name e.eraseFloat
  | .store dtype shape mem val mask =>
      .store (eraseFloatDType dtype) shape mem.eraseFloat val.eraseFloat mask.eraseFloat
  | .forLoop idx n body =>
      .forLoop idx n (Stmt.eraseFloatList body)
  | .ifThen cond body =>
      .ifThen cond.eraseFloat (Stmt.eraseFloatList body)
termination_by st => sizeOf st
decreasing_by all_goals (simp_wf; try omega)

/-- Erase explicit floating dtype annotations from a statement list. -/
def Stmt.eraseFloatList : List Stmt → List Stmt
  | [] => []
  | st :: rest => st.eraseFloat :: Stmt.eraseFloatList rest
termination_by body => sizeOf body
decreasing_by all_goals (simp_wf; try omega)

end

@[simp] theorem MemAccess.eraseFloat_region
    (region : RegionName) (off : Op .nat shape) :
    (MemAccess.region region off).eraseFloat =
      MemAccess.region region off.eraseFloat := by
  rw [MemAccess.eraseFloat.eq_def]

@[simp] theorem MemAccess.eraseFloat_ptr (ptr : Op .ptr shape) :
    (MemAccess.ptr ptr).eraseFloat = MemAccess.ptr ptr.eraseFloat := by
  rw [MemAccess.eraseFloat.eq_def]

@[simp] theorem MemAccess.eraseFloat_blockPtr
    (ptr : Op .blockPtr shape) (boundaryCheck : List Nat) :
    (MemAccess.blockPtr ptr boundaryCheck).eraseFloat =
      MemAccess.blockPtr ptr.eraseFloat boundaryCheck := by
  rw [MemAccess.eraseFloat.eq_def]

@[simp] theorem MaskOpt.eraseFloat_none {dtype : TileDType} :
    (MaskOpt.none : MaskOpt dtype shape).eraseFloat = MaskOpt.none := by
  rw [MaskOpt.eraseFloat.eq_def]

@[simp] theorem MaskOpt.eraseFloat_mask
    (mask : Op .bool shape) :
    (MaskOpt.mask (dtype := dtype) mask).eraseFloat =
      MaskOpt.mask mask.eraseFloat := by
  rw [MaskOpt.eraseFloat.eq_def]

@[simp] theorem MaskOpt.eraseFloat_maskOther
    (mask : Op .bool shape) (other : Op dtype shape) :
    (MaskOpt.maskOther mask other).eraseFloat =
      MaskOpt.maskOther mask.eraseFloat other.eraseFloat := by
  rw [MaskOpt.eraseFloat.eq_def]

@[simp] theorem Op.eraseFloat_ref
    (dtype : TileDType) (shape : TileShape) (name : RegName) :
    (Op.ref dtype shape name).eraseFloat =
      Op.ref (eraseFloatDType dtype) shape name := by
  rw [Op.eraseFloat.eq_def]

@[simp] theorem Op.eraseFloat_castFloat
    (src dst : FloatDType) (e : Op src.toTileDType shape) :
    (Op.castFloat src dst e).eraseFloat =
      cast (by simp [eraseFloatDType_float src, eraseFloatDType_float dst])
        e.eraseFloat := by
  rw [Op.eraseFloat.eq_def]
  simp [eraseFloatDType]

@[simp] theorem Op.eraseFloat_load
    (dtype : TileDType) (mem : MemAccess shape)
    (mask : MaskOpt dtype shape) :
    (Op.load dtype mem mask).eraseFloat =
      Op.load (eraseFloatDType dtype) mem.eraseFloat mask.eraseFloat := by
  rw [Op.eraseFloat.eq_def]

@[simp] theorem Op.eraseFloat_reduceMax
    (axis : Fin shape.length) (keepDims : Bool) (e : Op .real shape) :
    (Op.reduceMax axis keepDims e).eraseFloat =
      cast (by simp [eraseFloatDType])
        (@Op.reduceMax shape axis keepDims
          (cast (by simp [eraseFloatDType]) e.eraseFloat)) := by
  rw [Op.eraseFloat.eq_def]
  simp [eraseFloatDType]

@[simp] theorem Op.eraseFloat_reduceSum
    (axis : Fin shape.length) (keepDims : Bool) (e : Op .real shape) :
    (Op.reduceSum axis keepDims e).eraseFloat =
      cast (by simp [eraseFloatDType])
        (@Op.reduceSum shape axis keepDims
          (cast (by simp [eraseFloatDType]) e.eraseFloat)) := by
  rw [Op.eraseFloat.eq_def]
  simp [eraseFloatDType]

@[simp] theorem Stmt.eraseFloat_store
    (dtype : TileDType) (shape : TileShape)
    (mem : MemAccess shape) (val : Op dtype shape)
    (mask : MaskOpt dtype shape) :
    (Stmt.store dtype shape mem val mask).eraseFloat =
      Stmt.store (eraseFloatDType dtype) shape mem.eraseFloat val.eraseFloat mask.eraseFloat := by
  rw [Stmt.eraseFloat.eq_def]

namespace Kernel

/-- Erase explicit floating dtype annotations from a kernel body. -/
def eraseFloat (k : Kernel) : Kernel :=
  { k with body := Stmt.eraseFloatList k.body }

end Kernel

end VeriTile.Triton
