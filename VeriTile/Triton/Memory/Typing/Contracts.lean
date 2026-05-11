/-
VeriTile.Triton.Memory.Typing.Contracts

Lightweight region-dtype contract predicates for Triton kernels.
-/

import VeriTile.Triton.Memory.Typing.Region

namespace VeriTile.Triton

mutual

/--
All statically visible pointer bases inside a pointer expression have dtype
`dtype` under `Γ`.

Pointer registers (`Op.ref .ptr ...`) are treated as external/dynamic pointer
values in this lightweight layer, so they impose no region constraint here.
-/
def Op.PointerRegionsHaveDType (Γ : RegionTyping) (dtype : TileDType) :
    Op exprDType shape → Prop
  | .ptrBase region => Γ (Region.cast region) = dtype
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
  | _ => True
termination_by ptr => sizeOf ptr
decreasing_by all_goals (simp_wf; try omega)

/--
All statically visible base regions inside a block pointer expression have dtype
`dtype` under `Γ`. Block-pointer registers are dynamic/external values at this
lightweight layer, like pointer registers.
-/
def Op.BlockPointerRegionHasDType (Γ : RegionTyping) (dtype : TileDType) :
    Op exprDType shape → Prop
  | .makeBlockPtr region _ _ _ _ _ => Γ (Region.cast region) = dtype
  | .makeBlockPtrDyn region base _ _ _ _ =>
      Γ region = dtype ∧ base.RespectsRegionTyping Γ
  | .advanceBlockPtr ptr _ => ptr.BlockPointerRegionHasDType Γ dtype
  | .broadcast ptr _ => ptr.BlockPointerRegionHasDType Γ dtype
  | .full _ ptr => ptr.BlockPointerRegionHasDType Γ dtype
  | .where c a b =>
      c.RespectsRegionTyping Γ ∧
      a.BlockPointerRegionHasDType Γ dtype ∧
      b.BlockPointerRegionHasDType Γ dtype
  | .transpose ptr => ptr.BlockPointerRegionHasDType Γ dtype
  | .expandDim _ ptr => ptr.BlockPointerRegionHasDType Γ dtype
  | _ => True
termination_by ptr => sizeOf ptr
decreasing_by all_goals (simp_wf; try omega)

/-- Memory-region dtype contract for memory address forms. -/
def MemAccess.RespectsRegionTyping (Γ : RegionTyping) (dtype : TileDType) :
    MemAccess dtype shape → Prop
  | .region region off => Γ (Region.cast region) = dtype ∧ off.RespectsRegionTyping Γ
  | .ptr ptr =>
      ptr.RespectsRegionTyping Γ ∧ ptr.PointerRegionsHaveDType Γ dtype
  | .blockPtr ptr _ =>
      ptr.RespectsRegionTyping Γ ∧ ptr.BlockPointerRegionHasDType Γ dtype
termination_by mem => sizeOf mem
decreasing_by all_goals (simp_wf; try omega)

/-- Memory-region dtype contract for memory masks. -/
def MaskOpt.RespectsRegionTyping (Γ : RegionTyping) : MaskOpt dtype shape → Prop
  | .none => True
  | .mask mask => mask.RespectsRegionTyping Γ
  | .maskOther mask other => mask.RespectsRegionTyping Γ ∧ other.RespectsRegionTyping Γ
termination_by mask => sizeOf mask
decreasing_by all_goals (simp_wf; try omega)

/-- Memory-region dtype contract for expressions. -/
def Op.RespectsRegionTyping (Γ : RegionTyping) : Op dtype shape → Prop
  | .const _ => True
  | .constFloat _ _ => True
  | .constNat _ => True
  | .constInt _ => True
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
  | .bitAnd _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .bitOr _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .bitXor _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .shiftLeft _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .shiftRight _ a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .exp a => a.RespectsRegionTyping Γ
  | .exp2 a => a.RespectsRegionTyping Γ
  | .log a => a.RespectsRegionTyping Γ
  | .log2 a => a.RespectsRegionTyping Γ
  | .sigmoid a => a.RespectsRegionTyping Γ
  | .sqrt a => a.RespectsRegionTyping Γ
  | .rsqrt a => a.RespectsRegionTyping Γ
  | .tanh a => a.RespectsRegionTyping Γ
  | .sin a => a.RespectsRegionTyping Γ
  | .cos a => a.RespectsRegionTyping Γ
  | .tan a => a.RespectsRegionTyping Γ
  | .atan a => a.RespectsRegionTyping Γ
  | .cosh a => a.RespectsRegionTyping Γ
  | .sinh a => a.RespectsRegionTyping Γ
  | .erf a => a.RespectsRegionTyping Γ
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
  | .scan _ _ a => a.RespectsRegionTyping Γ
  | .argMax _ a => a.RespectsRegionTyping Γ
  | .argMin _ a => a.RespectsRegionTyping Γ
  | .sort _ a => a.RespectsRegionTyping Γ
  | .dot a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .transpose a => a.RespectsRegionTyping Γ
  | .reshape _ a => a.RespectsRegionTyping Γ
  | .remap _ _ a => a.RespectsRegionTyping Γ
  | .join a b => a.RespectsRegionTyping Γ ∧ b.RespectsRegionTyping Γ
  | .split _ a => a.RespectsRegionTyping Γ
  | .expandDim _ a => a.RespectsRegionTyping Γ
  | .ptrBase _ => True
  | .ptrAdd _ ptr off =>
      ptr.RespectsRegionTyping Γ ∧ off.RespectsRegionTyping Γ
  | .makeBlockPtr _ _ _ _ _ _ => True
  | .makeBlockPtrDyn _ base _ _ _ _ => base.RespectsRegionTyping Γ
  | .advanceBlockPtr ptr _ => ptr.RespectsRegionTyping Γ
  | .load dtype mem mask =>
      mem.RespectsRegionTyping Γ dtype ∧ mask.RespectsRegionTyping Γ
  | .natToReal a => a.RespectsRegionTyping Γ
termination_by op => sizeOf op
decreasing_by all_goals (simp_wf; try omega)

end

mutual

/-- Memory-region dtype contract for statements. -/
def Stmt.RespectsRegionTyping (Γ : RegionTyping) : Stmt → Prop
  | .assign _ _ _ e => e.RespectsRegionTyping Γ
  | .store dtype _ mem val mask =>
      mem.RespectsRegionTyping Γ dtype ∧
      val.RespectsRegionTyping Γ ∧ mask.RespectsRegionTyping Γ
  | @Stmt.atomicAdd dtype _ _ mem val mask =>
      mem.RespectsRegionTyping Γ dtype ∧
      val.RespectsRegionTyping Γ ∧ mask.RespectsRegionTyping Γ
  | .atomicRMW _ dtype _ mem input extraInput mask _ =>
      mem.RespectsRegionTyping Γ dtype ∧
      input.RespectsRegionTyping Γ ∧
      (match extraInput with
        | none => True
        | some extra => extra.RespectsRegionTyping Γ) ∧
      mask.RespectsRegionTyping Γ
  | .forLoop _ _ body => StmtList.RespectsRegionTyping Γ body
  | .forRange _ _ _ _ body => StmtList.RespectsRegionTyping Γ body
  | .forRangeDyn _ start stop step body =>
      start.RespectsRegionTyping Γ ∧ stop.RespectsRegionTyping Γ ∧
        step.RespectsRegionTyping Γ ∧ StmtList.RespectsRegionTyping Γ body
  | .ifThen cond body =>
      cond.RespectsRegionTyping Γ ∧ StmtList.RespectsRegionTyping Γ body
  | .ifThenElse cond thenBody elseBody =>
      cond.RespectsRegionTyping Γ ∧
      StmtList.RespectsRegionTyping Γ thenBody ∧
      StmtList.RespectsRegionTyping Γ elseBody
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
