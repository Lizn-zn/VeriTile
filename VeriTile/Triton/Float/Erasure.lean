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

namespace NumericDType

/-- Numeric witness after floating-dtype erasure. -/
def eraseFloat : NumericDType dtype → NumericDType (eraseFloatDType dtype)
  | .real => .real
  | .fp32 => .real
  | .fp16 => .real
  | .bf16 => .real
  | .int32 => .int32
  | .nat => .nat

end NumericDType

namespace ComparableDType

/-- Comparable witness after floating-dtype erasure. -/
def eraseFloat : ComparableDType dtype → ComparableDType (eraseFloatDType dtype)
  | .real => .real
  | .fp32 => .real
  | .fp16 => .real
  | .bf16 => .real
  | .int32 => .int32
  | .nat => .nat

end ComparableDType

mutual

/-- Erase explicit floating dtype annotations from an expression. -/
def Op.eraseFloat : Op dtype shape → Op (eraseFloatDType dtype) shape
  | .const c => .const c
  | .constFloat .real c => .const c
  | .constFloat .fp32 c => .const c
  | .constFloat .fp16 c => .const c
  | .constFloat .bf16 c => .const c
  | .constNat n => .constNat n
  | .constBool b => .constBool b
  | .negInf => .negInf
  | .programId axis => .programId axis
  | .ref dtype shape name => .ref (eraseFloatDType dtype) shape name
  | .arange n => .arange n
  | .broadcast e shape => .broadcast e.eraseFloat shape
  | .full shape e => .full shape e.eraseFloat
  | .castFloat .real .real e => e.eraseFloat
  | .castFloat .real .fp32 e => e.eraseFloat
  | .castFloat .real .fp16 e => e.eraseFloat
  | .castFloat .real .bf16 e => e.eraseFloat
  | .castFloat .fp32 .real e => e.eraseFloat
  | .castFloat .fp32 .fp32 e => e.eraseFloat
  | .castFloat .fp32 .fp16 e => e.eraseFloat
  | .castFloat .fp32 .bf16 e => e.eraseFloat
  | .castFloat .fp16 .real e => e.eraseFloat
  | .castFloat .fp16 .fp32 e => e.eraseFloat
  | .castFloat .fp16 .fp16 e => e.eraseFloat
  | .castFloat .fp16 .bf16 e => e.eraseFloat
  | .castFloat .bf16 .real e => e.eraseFloat
  | .castFloat .bf16 .fp32 e => e.eraseFloat
  | .castFloat .bf16 .fp16 e => e.eraseFloat
  | .castFloat .bf16 .bf16 e => e.eraseFloat
  | .add h bc a b => .add h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .sub h bc a b => .sub h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .mul h bc a b => .mul h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .div h bc a b => .div h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .exp a => .exp a.eraseFloat
  | .log a => .log a.eraseFloat
  | .sigmoid a => .sigmoid a.eraseFloat
  | .sqrt a => .sqrt a.eraseFloat
  | .tanh a => .tanh a.eraseFloat
  | .lt h bc a b => .lt h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .le h bc a b => .le h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .eq h bc a b => .eq h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .gt h bc a b => .gt h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .ge h bc a b => .ge h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .ne h bc a b => .ne h.eraseFloat bc a.eraseFloat b.eraseFloat
  | .boolAnd bc a b => .boolAnd bc a.eraseFloat b.eraseFloat
  | .max2 bc a b => .max2 bc a.eraseFloat b.eraseFloat
  | .where c a b => .where c.eraseFloat a.eraseFloat b.eraseFloat
  | .reduceMax axis keepDims a => .reduceMax axis keepDims a.eraseFloat
  | .reduceSum axis keepDims a => .reduceSum axis keepDims a.eraseFloat
  | .dot a b => .dot a.eraseFloat b.eraseFloat
  | .transpose a => .transpose a.eraseFloat
  | .expandDim axis a => .expandDim axis a.eraseFloat
  | .ptrBase region => .ptrBase region
  | .ptrAdd bc ptr off => .ptrAdd bc ptr.eraseFloat off.eraseFloat
  | .load region off => .load region off.eraseFloat
  | .loadMask region off mask => .loadMask region off.eraseFloat mask.eraseFloat
  | .loadMaskOther region off mask other =>
      .loadMaskOther region off.eraseFloat mask.eraseFloat other.eraseFloat
  | .loadPtr ptr => .loadPtr ptr.eraseFloat
  | .loadPtrMask ptr mask => .loadPtrMask ptr.eraseFloat mask.eraseFloat
  | .loadPtrMaskOther ptr mask other =>
      .loadPtrMaskOther ptr.eraseFloat mask.eraseFloat other.eraseFloat
  | .loadFloat .real region off => .load region off.eraseFloat
  | .loadFloat .fp32 region off => .load region off.eraseFloat
  | .loadFloat .fp16 region off => .load region off.eraseFloat
  | .loadFloat .bf16 region off => .load region off.eraseFloat
  | .loadFloatMask .real region off mask => .loadMask region off.eraseFloat mask.eraseFloat
  | .loadFloatMask .fp32 region off mask => .loadMask region off.eraseFloat mask.eraseFloat
  | .loadFloatMask .fp16 region off mask => .loadMask region off.eraseFloat mask.eraseFloat
  | .loadFloatMask .bf16 region off mask => .loadMask region off.eraseFloat mask.eraseFloat
  | .loadFloatMaskOther .real region off mask other =>
      .loadMaskOther region off.eraseFloat mask.eraseFloat other.eraseFloat
  | .loadFloatMaskOther .fp32 region off mask other =>
      .loadMaskOther region off.eraseFloat mask.eraseFloat other.eraseFloat
  | .loadFloatMaskOther .fp16 region off mask other =>
      .loadMaskOther region off.eraseFloat mask.eraseFloat other.eraseFloat
  | .loadFloatMaskOther .bf16 region off mask other =>
      .loadMaskOther region off.eraseFloat mask.eraseFloat other.eraseFloat
  | .loadPtrFloat .real ptr => .loadPtr ptr.eraseFloat
  | .loadPtrFloat .fp32 ptr => .loadPtr ptr.eraseFloat
  | .loadPtrFloat .fp16 ptr => .loadPtr ptr.eraseFloat
  | .loadPtrFloat .bf16 ptr => .loadPtr ptr.eraseFloat
  | .loadPtrFloatMask .real ptr mask => .loadPtrMask ptr.eraseFloat mask.eraseFloat
  | .loadPtrFloatMask .fp32 ptr mask => .loadPtrMask ptr.eraseFloat mask.eraseFloat
  | .loadPtrFloatMask .fp16 ptr mask => .loadPtrMask ptr.eraseFloat mask.eraseFloat
  | .loadPtrFloatMask .bf16 ptr mask => .loadPtrMask ptr.eraseFloat mask.eraseFloat
  | .loadPtrFloatMaskOther .real ptr mask other =>
      .loadPtrMaskOther ptr.eraseFloat mask.eraseFloat other.eraseFloat
  | .loadPtrFloatMaskOther .fp32 ptr mask other =>
      .loadPtrMaskOther ptr.eraseFloat mask.eraseFloat other.eraseFloat
  | .loadPtrFloatMaskOther .fp16 ptr mask other =>
      .loadPtrMaskOther ptr.eraseFloat mask.eraseFloat other.eraseFloat
  | .loadPtrFloatMaskOther .bf16 ptr mask other =>
      .loadPtrMaskOther ptr.eraseFloat mask.eraseFloat other.eraseFloat
  | .natToReal a => .natToReal a.eraseFloat
termination_by e => sizeOf e
decreasing_by all_goals (simp_wf; try omega)

/-- Erase explicit floating dtype annotations from a statement. -/
def Stmt.eraseFloat : Stmt → Stmt
  | .assign dtype shape name e =>
      .assign (eraseFloatDType dtype) shape name e.eraseFloat
  | .store region shape off val =>
      .store region shape off.eraseFloat val.eraseFloat
  | .storeMask region shape off val mask =>
      .storeMask region shape off.eraseFloat val.eraseFloat mask.eraseFloat
  | .storePtr shape ptr val =>
      .storePtr shape ptr.eraseFloat val.eraseFloat
  | .storePtrMask shape ptr val mask =>
      .storePtrMask shape ptr.eraseFloat val.eraseFloat mask.eraseFloat
  | .storeFloat .real region shape off val =>
      .store region shape off.eraseFloat val.eraseFloat
  | .storeFloat .fp32 region shape off val =>
      .store region shape off.eraseFloat val.eraseFloat
  | .storeFloat .fp16 region shape off val =>
      .store region shape off.eraseFloat val.eraseFloat
  | .storeFloat .bf16 region shape off val =>
      .store region shape off.eraseFloat val.eraseFloat
  | .storeFloatMask .real region shape off val mask =>
      .storeMask region shape off.eraseFloat val.eraseFloat mask.eraseFloat
  | .storeFloatMask .fp32 region shape off val mask =>
      .storeMask region shape off.eraseFloat val.eraseFloat mask.eraseFloat
  | .storeFloatMask .fp16 region shape off val mask =>
      .storeMask region shape off.eraseFloat val.eraseFloat mask.eraseFloat
  | .storeFloatMask .bf16 region shape off val mask =>
      .storeMask region shape off.eraseFloat val.eraseFloat mask.eraseFloat
  | .storePtrFloat .real shape ptr val =>
      .storePtr shape ptr.eraseFloat val.eraseFloat
  | .storePtrFloat .fp32 shape ptr val =>
      .storePtr shape ptr.eraseFloat val.eraseFloat
  | .storePtrFloat .fp16 shape ptr val =>
      .storePtr shape ptr.eraseFloat val.eraseFloat
  | .storePtrFloat .bf16 shape ptr val =>
      .storePtr shape ptr.eraseFloat val.eraseFloat
  | .storePtrFloatMask .real shape ptr val mask =>
      .storePtrMask shape ptr.eraseFloat val.eraseFloat mask.eraseFloat
  | .storePtrFloatMask .fp32 shape ptr val mask =>
      .storePtrMask shape ptr.eraseFloat val.eraseFloat mask.eraseFloat
  | .storePtrFloatMask .fp16 shape ptr val mask =>
      .storePtrMask shape ptr.eraseFloat val.eraseFloat mask.eraseFloat
  | .storePtrFloatMask .bf16 shape ptr val mask =>
      .storePtrMask shape ptr.eraseFloat val.eraseFloat mask.eraseFloat
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

namespace Kernel

/-- Erase explicit floating dtype annotations from a kernel body. -/
def eraseFloat (k : Kernel) : Kernel :=
  { k with body := Stmt.eraseFloatList k.body }

end Kernel

end VeriTile.Triton
