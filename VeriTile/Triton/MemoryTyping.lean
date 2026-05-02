/-
VeriTile.Triton.MemoryTyping

Lightweight region-dtype contracts for Triton kernels.

This module does not change `BlockState.mem`; memory is still modeled as
`RegionName → Nat → ℝ`. The predicates here only record the static contract
that a kernel's memory operations agree with a user-supplied region dtype map.
-/

import VeriTile.Triton.Core

namespace VeriTile.Triton

/-- User-supplied dtype assignment for named memory regions. -/
abbrev RegionTyping := RegionName → TileDType

namespace FloatDType

/-- The concrete dtype witnessed by a floating dtype proof. -/
def dtype : {dtype : TileDType} → FloatDType dtype → TileDType
  | _, .real => .real
  | _, .fp32 => .fp32
  | _, .fp16 => .fp16
  | _, .bf16 => .bf16

end FloatDType

mutual

/--
All statically visible pointer bases inside a pointer expression have dtype
`dtype` under `Γ`.

Pointer registers (`Op.ref .ptr ...`) are treated as external/dynamic pointer
values in this lightweight layer, so they impose no region constraint here.
-/
def Op.PointerRegionsHaveDType (Γ : RegionTyping) (dtype : TileDType) :
    Op .ptr shape → Prop
  | .ptrBase region => Γ region = dtype
  | .ptrAdd _ ptr off =>
      ptr.PointerRegionsHaveDType Γ dtype ∧ off.WellTypedMemory Γ
  | .broadcast ptr _ =>
      ptr.PointerRegionsHaveDType Γ dtype
  | .full _ ptr =>
      ptr.PointerRegionsHaveDType Γ dtype
  | .where c a b =>
      c.WellTypedMemory Γ ∧
      a.PointerRegionsHaveDType Γ dtype ∧
      b.PointerRegionsHaveDType Γ dtype
  | .transpose ptr =>
      ptr.PointerRegionsHaveDType Γ dtype
  | .expandDim _ ptr =>
      ptr.PointerRegionsHaveDType Γ dtype
  | .ref .ptr _ _ => True
termination_by ptr => sizeOf ptr
decreasing_by all_goals (simp_wf; try omega)

/-- Memory-region dtype contract for expressions. -/
def Op.WellTypedMemory (Γ : RegionTyping) : Op dtype shape → Prop
  | .const _ => True
  | .constFloat _ _ => True
  | .constNat _ => True
  | .constBool _ => True
  | .negInf => True
  | .programId _ => True
  | .ref _ _ _ => True
  | .arange _ => True
  | .broadcast e _ => e.WellTypedMemory Γ
  | .full _ e => e.WellTypedMemory Γ
  | .castFloat _ _ e => e.WellTypedMemory Γ
  | .add _ _ a b => a.WellTypedMemory Γ ∧ b.WellTypedMemory Γ
  | .sub _ _ a b => a.WellTypedMemory Γ ∧ b.WellTypedMemory Γ
  | .mul _ _ a b => a.WellTypedMemory Γ ∧ b.WellTypedMemory Γ
  | .div _ _ a b => a.WellTypedMemory Γ ∧ b.WellTypedMemory Γ
  | .exp a => a.WellTypedMemory Γ
  | .log a => a.WellTypedMemory Γ
  | .sigmoid a => a.WellTypedMemory Γ
  | .sqrt a => a.WellTypedMemory Γ
  | .tanh a => a.WellTypedMemory Γ
  | .lt _ _ a b => a.WellTypedMemory Γ ∧ b.WellTypedMemory Γ
  | .le _ _ a b => a.WellTypedMemory Γ ∧ b.WellTypedMemory Γ
  | .eq _ _ a b => a.WellTypedMemory Γ ∧ b.WellTypedMemory Γ
  | .gt _ _ a b => a.WellTypedMemory Γ ∧ b.WellTypedMemory Γ
  | .ge _ _ a b => a.WellTypedMemory Γ ∧ b.WellTypedMemory Γ
  | .ne _ _ a b => a.WellTypedMemory Γ ∧ b.WellTypedMemory Γ
  | .boolAnd _ a b => a.WellTypedMemory Γ ∧ b.WellTypedMemory Γ
  | .max2 _ a b => a.WellTypedMemory Γ ∧ b.WellTypedMemory Γ
  | .where c a b =>
      c.WellTypedMemory Γ ∧ a.WellTypedMemory Γ ∧ b.WellTypedMemory Γ
  | .reduceMax _ _ a => a.WellTypedMemory Γ
  | .reduceSum _ _ a => a.WellTypedMemory Γ
  | .dot a b => a.WellTypedMemory Γ ∧ b.WellTypedMemory Γ
  | .transpose a => a.WellTypedMemory Γ
  | .expandDim _ a => a.WellTypedMemory Γ
  | .ptrBase _ => True
  | .ptrAdd _ ptr off =>
      ptr.WellTypedMemory Γ ∧ off.WellTypedMemory Γ
  | .load region off =>
      Γ region = .real ∧ off.WellTypedMemory Γ
  | .loadMask region off mask =>
      Γ region = .real ∧ off.WellTypedMemory Γ ∧ mask.WellTypedMemory Γ
  | .loadMaskOther region off mask other =>
      Γ region = .real ∧
      off.WellTypedMemory Γ ∧ mask.WellTypedMemory Γ ∧ other.WellTypedMemory Γ
  | .loadPtr ptr =>
      ptr.WellTypedMemory Γ ∧ ptr.PointerRegionsHaveDType Γ .real
  | .loadPtrMask ptr mask =>
      ptr.WellTypedMemory Γ ∧ ptr.PointerRegionsHaveDType Γ .real ∧
      mask.WellTypedMemory Γ
  | .loadPtrMaskOther ptr mask other =>
      ptr.WellTypedMemory Γ ∧ ptr.PointerRegionsHaveDType Γ .real ∧
      mask.WellTypedMemory Γ ∧ other.WellTypedMemory Γ
  | .loadFloat h region off =>
      Γ region = h.dtype ∧ off.WellTypedMemory Γ
  | .loadFloatMask h region off mask =>
      Γ region = h.dtype ∧ off.WellTypedMemory Γ ∧ mask.WellTypedMemory Γ
  | .loadFloatMaskOther h region off mask other =>
      Γ region = h.dtype ∧
      off.WellTypedMemory Γ ∧ mask.WellTypedMemory Γ ∧ other.WellTypedMemory Γ
  | .loadPtrFloat h ptr =>
      ptr.WellTypedMemory Γ ∧ ptr.PointerRegionsHaveDType Γ h.dtype
  | .loadPtrFloatMask h ptr mask =>
      ptr.WellTypedMemory Γ ∧ ptr.PointerRegionsHaveDType Γ h.dtype ∧
      mask.WellTypedMemory Γ
  | .loadPtrFloatMaskOther h ptr mask other =>
      ptr.WellTypedMemory Γ ∧ ptr.PointerRegionsHaveDType Γ h.dtype ∧
      mask.WellTypedMemory Γ ∧ other.WellTypedMemory Γ
  | .natToReal a => a.WellTypedMemory Γ
termination_by op => sizeOf op
decreasing_by all_goals (simp_wf; try omega)

end

mutual

/-- Memory-region dtype contract for statements. -/
def Stmt.WellTypedMemory (Γ : RegionTyping) : Stmt → Prop
  | .assign _ _ _ e => e.WellTypedMemory Γ
  | .store region _ off val =>
      Γ region = .real ∧ off.WellTypedMemory Γ ∧ val.WellTypedMemory Γ
  | .storeMask region _ off val mask =>
      Γ region = .real ∧
      off.WellTypedMemory Γ ∧ val.WellTypedMemory Γ ∧ mask.WellTypedMemory Γ
  | .storePtr _ ptr val =>
      ptr.WellTypedMemory Γ ∧ ptr.PointerRegionsHaveDType Γ .real ∧
      val.WellTypedMemory Γ
  | .storePtrMask _ ptr val mask =>
      ptr.WellTypedMemory Γ ∧ ptr.PointerRegionsHaveDType Γ .real ∧
      val.WellTypedMemory Γ ∧ mask.WellTypedMemory Γ
  | .storeFloat h region _ off val =>
      Γ region = h.dtype ∧ off.WellTypedMemory Γ ∧ val.WellTypedMemory Γ
  | .storeFloatMask h region _ off val mask =>
      Γ region = h.dtype ∧
      off.WellTypedMemory Γ ∧ val.WellTypedMemory Γ ∧ mask.WellTypedMemory Γ
  | .storePtrFloat h _ ptr val =>
      ptr.WellTypedMemory Γ ∧ ptr.PointerRegionsHaveDType Γ h.dtype ∧
      val.WellTypedMemory Γ
  | .storePtrFloatMask h _ ptr val mask =>
      ptr.WellTypedMemory Γ ∧ ptr.PointerRegionsHaveDType Γ h.dtype ∧
      val.WellTypedMemory Γ ∧ mask.WellTypedMemory Γ
  | .forLoop _ _ body => StmtList.WellTypedMemory Γ body
  | .ifThen cond body =>
      cond.WellTypedMemory Γ ∧ StmtList.WellTypedMemory Γ body
termination_by st => sizeOf st
decreasing_by all_goals (simp_wf; try omega)

/-- Memory-region dtype contract for statement lists. -/
def StmtList.WellTypedMemory (Γ : RegionTyping) : List Stmt → Prop
  | [] => True
  | st :: rest => st.WellTypedMemory Γ ∧ StmtList.WellTypedMemory Γ rest
termination_by body => sizeOf body
decreasing_by all_goals (simp_wf; try omega)

end

namespace Kernel

/-- Memory-region dtype contract for kernels. -/
def WellTypedMemory (Γ : RegionTyping) (k : Kernel) : Prop :=
  StmtList.WellTypedMemory Γ k.body

end Kernel

end VeriTile.Triton
