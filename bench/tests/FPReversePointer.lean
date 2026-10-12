import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Pointer decrement is discrete address arithmetic. It preserves the region,
broadcasts the stride and propagates missing registers; no FP atom is used.
The model uses natural offsets, including truncation below zero. Kernel proofs
must ensure that every actual read/write corresponds to a valid source address. -/
namespace FPReversePointerTests
open VeriTile.Triton FP.Structural
variable {α : Type} [Inhabited α]

private def pointers : Op .ptr [4] :=
  .ptrAdd .scalarL (.ptrBase ("data" : RegionName)) (.add .nat .scalarR (.arange 4) (.constNat 64))

theorem two_reverse_steps (M : Algebra α) (p : Option ComputeDType) (s : State α) :
    FP.Structural.evalOp M p
      (.ptrSub .scalarR (.ptrSub .scalarR pointers (.constNat 16)) (.constNat 16)) s =
      some (fun i : TileIndex [4] => (("data" : RegionName), 32 + i.1.val)) := by
  simp only [evalOp_unfold, pointers, Option.bind, bind, pure, bop, numeric, Region.cast]
  congr 1
  funext i
  dsimp [bop, Broadcast.leftIndex, Broadcast.rightIndex]
  congr 1
  omega

theorem natural_offset_boundary (M : Algebra α) (s : State α) :
    FP.Structural.evalOp M none (.ptrSub .nil (.ptrBase ("data" : RegionName)) (.constNat 16)) s =
      some (fun _ => (("data" : RegionName), 0)) := rfl

theorem missing_pointer_fails (M : Algebra α) (s : State α)
    (h : s.regs .ptr [4] "p" = none) :
    FP.Structural.evalOp M none
      (.ptrSub .scalarR (.ref .ptr [4] "p") (.constNat 16)) s = none := by
  simp [FP.Structural.evalOp, h]

theorem missing_stride_fails (M : Algebra α) (s : State α)
    (h : s.regs .nat [] "stride" = none) :
    FP.Structural.evalOp M none (.ptrSub .scalarR pointers (.ref .nat [] "stride")) s = none := by
  simp [FP.Structural.evalOp, pointers, h]

#guard_msgs (drop info) in
#auditModuleAxioms
end FPReversePointerTests
