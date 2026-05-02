/-
VeriTile.Triton.MemoryTyping.Contracts

Lightweight region-dtype contract predicates for Triton kernels.
-/

import VeriTile.Triton.MemoryTyping.Region

namespace VeriTile.Triton

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
      ptr.PointerRegionsHaveDType Γ dtype ∧ off.RespectsRegionTyping Γ
  | .broadcast ptr _ =>
      ptr.PointerRegionsHaveDType Γ dtype
  | .full _ ptr =>
      ptr.PointerRegionsHaveDType Γ dtype
  | .where c a b =>
      c.RespectsRegionTyping Γ ∧
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
def Op.RespectsRegionTyping (Γ : RegionTyping) : Op dtype shape → Prop
  | .const _ => True
  | .constFloat _ _ => True
  | .constNat _ => True
  | .constBool _ => True
  | .negInf => True
  | .programId _ => True
  | .ref _ _ _ => True
  | .arange _ => True
  | .broadcast e _ => e.RespectsRegionTyping Γ
  | .full _ e => e.RespectsRegionTyping Γ
  | .castFloat _ _ e => e.RespectsRegionTyping Γ
  | .add _ _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .sub _ _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .mul _ _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .div _ _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .floorDiv _ _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .mod _ _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .exp a => a.RespectsRegionTyping Γ
  | .log a => a.RespectsRegionTyping Γ
  | .sigmoid a => a.RespectsRegionTyping Γ
  | .sqrt a => a.RespectsRegionTyping Γ
  | .tanh a => a.RespectsRegionTyping Γ
  | .lt _ _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .le _ _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .eq _ _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .gt _ _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .ge _ _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .ne _ _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .boolAnd _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .boolOr _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .boolNot a => a.RespectsRegionTyping Γ
  | .max2 _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .where c a b =>
      c.RespectsRegionTyping Γ ∧ a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .reduceMax _ _ a => a.RespectsRegionTyping Γ
  | .reduceSum _ _ a => a.RespectsRegionTyping Γ
  | .dot a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .transpose a => a.RespectsRegionTyping Γ
  | .expandDim _ a => a.RespectsRegionTyping Γ
  | .ptrBase _ => True
  | .ptrAdd _ ptr off =>
      ptr.RespectsRegionTyping Γ ∧ off.RespectsRegionTyping Γ
  | .load region off =>
      Γ region = .real ∧ off.RespectsRegionTyping Γ
  | .loadMask region off mask =>
      Γ region = .real ∧ off.RespectsRegionTyping Γ ∧ mask.RespectsRegionTyping Γ
  | .loadMaskOther region off mask other =>
      Γ region = .real ∧
      off.RespectsRegionTyping Γ ∧ mask.RespectsRegionTyping Γ ∧ other.RespectsRegionTyping Γ
  | .loadPtr ptr =>
      ptr.RespectsRegionTyping Γ ∧ ptr.PointerRegionsHaveDType Γ .real
  | .loadPtrMask ptr mask =>
      ptr.RespectsRegionTyping Γ ∧ ptr.PointerRegionsHaveDType Γ .real ∧
      mask.RespectsRegionTyping Γ
  | .loadPtrMaskOther ptr mask other =>
      ptr.RespectsRegionTyping Γ ∧ ptr.PointerRegionsHaveDType Γ .real ∧
      mask.RespectsRegionTyping Γ ∧ other.RespectsRegionTyping Γ
  | .loadFloat h region off =>
      Γ region = h.dtype ∧ off.RespectsRegionTyping Γ
  | .loadFloatMask h region off mask =>
      Γ region = h.dtype ∧ off.RespectsRegionTyping Γ ∧ mask.RespectsRegionTyping Γ
  | .loadFloatMaskOther h region off mask other =>
      Γ region = h.dtype ∧
      off.RespectsRegionTyping Γ ∧ mask.RespectsRegionTyping Γ ∧ other.RespectsRegionTyping Γ
  | .loadPtrFloat h ptr =>
      ptr.RespectsRegionTyping Γ ∧ ptr.PointerRegionsHaveDType Γ h.dtype
  | .loadPtrFloatMask h ptr mask =>
      ptr.RespectsRegionTyping Γ ∧ ptr.PointerRegionsHaveDType Γ h.dtype ∧
      mask.RespectsRegionTyping Γ
  | .loadPtrFloatMaskOther h ptr mask other =>
      ptr.RespectsRegionTyping Γ ∧ ptr.PointerRegionsHaveDType Γ h.dtype ∧
      mask.RespectsRegionTyping Γ ∧ other.RespectsRegionTyping Γ
  | .natToReal a => a.RespectsRegionTyping Γ
termination_by op => sizeOf op
decreasing_by all_goals (simp_wf; try omega)

end

mutual

/-- Memory-region dtype contract for statements. -/
def Stmt.RespectsRegionTyping (Γ : RegionTyping) : Stmt → Prop
  | .assign _ _ _ e => e.RespectsRegionTyping Γ
  | .store region _ off val =>
      Γ region = .real ∧ off.RespectsRegionTyping Γ ∧ val.RespectsRegionTyping Γ
  | .storeMask region _ off val mask =>
      Γ region = .real ∧
      off.RespectsRegionTyping Γ ∧ val.RespectsRegionTyping Γ ∧ mask.RespectsRegionTyping Γ
  | .storePtr _ ptr val =>
      ptr.RespectsRegionTyping Γ ∧ ptr.PointerRegionsHaveDType Γ .real ∧
      val.RespectsRegionTyping Γ
  | .storePtrMask _ ptr val mask =>
      ptr.RespectsRegionTyping Γ ∧ ptr.PointerRegionsHaveDType Γ .real ∧
      val.RespectsRegionTyping Γ ∧ mask.RespectsRegionTyping Γ
  | .storeFloat h region _ off val =>
      Γ region = h.dtype ∧ off.RespectsRegionTyping Γ ∧ val.RespectsRegionTyping Γ
  | .storeFloatMask h region _ off val mask =>
      Γ region = h.dtype ∧
      off.RespectsRegionTyping Γ ∧ val.RespectsRegionTyping Γ ∧ mask.RespectsRegionTyping Γ
  | .storePtrFloat h _ ptr val =>
      ptr.RespectsRegionTyping Γ ∧ ptr.PointerRegionsHaveDType Γ h.dtype ∧
      val.RespectsRegionTyping Γ
  | .storePtrFloatMask h _ ptr val mask =>
      ptr.RespectsRegionTyping Γ ∧ ptr.PointerRegionsHaveDType Γ h.dtype ∧
      val.RespectsRegionTyping Γ ∧ mask.RespectsRegionTyping Γ
  | .forLoop _ _ body => StmtList.RespectsRegionTyping Γ body
  | .ifThen cond body =>
      cond.RespectsRegionTyping Γ ∧ StmtList.RespectsRegionTyping Γ body
termination_by st => sizeOf st
decreasing_by all_goals (simp_wf; try omega)

/-- Memory-region dtype contract for statement lists. -/
def StmtList.RespectsRegionTyping (Γ : RegionTyping) : List Stmt → Prop
  | [] => True
  | st :: rest => st.RespectsRegionTyping Γ ∧ StmtList.RespectsRegionTyping Γ rest
termination_by body => sizeOf body
decreasing_by all_goals (simp_wf; try omega)

end

namespace Kernel

/-- Memory-region dtype contract for kernels. -/
def RespectsRegionTyping (Γ : RegionTyping) (k : Kernel) : Prop :=
  StmtList.RespectsRegionTyping Γ k.body

end Kernel

end VeriTile.Triton
