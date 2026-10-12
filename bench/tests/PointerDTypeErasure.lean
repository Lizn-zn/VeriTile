import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace PointerDTypeErasureTests
open VeriTile.Triton

theorem fp16_base_becomes_real (r : Region .fp16) :
    (Op.ptrBase r).eraseDType = Op.ptrBase (r.cast : Region .real) := by
  rw [Op.eraseDType.eq_def]
  rfl

theorem integer_base_stays_integer (r : Region .int) :
    (Op.ptrBase r).eraseDType = Op.ptrBase r := by
  rw [Op.eraseDType.eq_def]
  rfl

theorem base_address_preserved (r : Region d) (s : BlockState) :
    evalOp (Op.ptrBase r).eraseDType s = evalOp (Op.ptrBase r) s := by
  have h : (Op.ptrBase r).eraseDType =
      Op.ptrBase (r.cast : Region (eraseDType d)) := by
    rw [Op.eraseDType.eq_def]
  rw [h]
  simp only [evalOp.eq_def]
  rfl

theorem block_descriptor_preserved (r : Region d) (offset : Nat)
    (parent shape strides offsets : List Nat) (s : BlockState) :
    evalOp (Op.makeBlockPtr r offset parent shape strides offsets).eraseDType s =
      evalOp (Op.makeBlockPtr r offset parent shape strides offsets) s := by
  have h : (Op.makeBlockPtr r offset parent shape strides offsets).eraseDType =
      Op.makeBlockPtr (r.cast : Region (eraseDType d)) offset parent shape strides offsets := by
    rw [Op.eraseDType.eq_def]
  rw [h]
  simp only [evalOp.eq_def]
  rfl

theorem fp16_block_becomes_real (r : Region .fp16) (offset : Nat)
    (parent shape strides offsets : List Nat) :
    (Op.makeBlockPtr r offset parent shape strides offsets).eraseDType =
      Op.makeBlockPtr (r.cast : Region .real) offset parent shape strides offsets := by
  rw [Op.eraseDType.eq_def]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end PointerDTypeErasureTests
