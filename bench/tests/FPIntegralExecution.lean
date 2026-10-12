import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Semantics.EvalOp
import VeriTile.Meta.StatementAudit

/-! Exact integer address operations in the FP interpreter. These checks make
no assumptions about floating arithmetic, and retain the real interpreter's
discrete conventions rather than claiming hardware signed-division semantics. -/
namespace FPIntegralExecutionTests
open VeriTile.Triton FP.Structural

variable {α : Type}

theorem nat_div_matches (x y : Nat) :
    integralFloorDiv (α := α) .nat x y = IntegralDType.floorDiv .nat x y := rfl

theorem nat_mod_matches (x y : Nat) :
    integralMod (α := α) .nat x y = IntegralDType.mod .nat x y := rfl

theorem int_div_matches (x y : Int) :
    integralFloorDiv (α := α) .int x y = IntegralDType.floorDiv .int x y := rfl

theorem int_mod_matches (x y : Int) :
    integralMod (α := α) .int x y = IntegralDType.mod .int x y := rfl

variable [Inhabited α]

/-- Signed window masks are discrete comparisons, independent of the chosen
floating backend and compute precision. The negative case must not fail or
become a truncated natural zero. -/
theorem signed_order (M : Algebra α) (p : Option ComputeDType)
    (s : State α) (x y : Int) :
    FP.Structural.evalOp M p (.lt .int .nil (.constInt x) (.constInt y)) s =
      some (fun _ => decide (x < y)) ∧
    FP.Structural.evalOp M p (.le .int .nil (.constInt x) (.constInt y)) s =
      some (fun _ => decide (x ≤ y)) ∧
    FP.Structural.evalOp M p (.gt .int .nil (.constInt x) (.constInt y)) s =
      some (fun _ => decide (y < x)) ∧
    FP.Structural.evalOp M p (.ge .int .nil (.constInt x) (.constInt y)) s =
      some (fun _ => decide (y ≤ x)) := by
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem future_key_is_masked (M : Algebra α) (s : State α) :
    FP.Structural.evalOp M none
      (.boolAnd .nil
        (.ge .int .nil (.sub .int .nil (.constInt 0) (.constInt 1)) (.constInt 0))
        (.lt .int .nil (.sub .int .nil (.constInt 0) (.constInt 1)) (.constInt 64))) s =
      some (fun _ => Bool.false) := by
  rfl

theorem missing_signed_operand_fails (M : Algebra α) (s : State α)
    (h : s.regs .int [] "distance" = none) :
    FP.Structural.evalOp M none
      (.ge .int .nil (.ref .int [] "distance") (.constInt 0)) s = none := by
  simp [FP.Structural.evalOp, numericLe, h]

-- Broadcast division and remainder partition an actual address tile.
theorem broadcast_div_mod (M : Algebra α) (p : Option ComputeDType)
    (s : State α) (n divisor : Nat) :
    FP.Structural.evalOp M p (.floorDiv .nat .scalarR (.arange n) (.constNat divisor)) s =
      some (fun i => i.1.val / divisor) ∧
    FP.Structural.evalOp M p (.mod .nat .scalarR (.arange n) (.constNat divisor)) s =
      some (fun i => i.1.val % divisor) := by
  constructor <;> rfl

-- Signed and zero-divisor cases are explicitly regression checked. In
-- particular Int.div is Lean's Euclidean convention, not truncation to zero.
theorem signed_div_mod (M : Algebra α) (s : State α) :
    FP.Structural.evalOp M none (.floorDiv .int .nil (.constInt (-7)) (.constInt 3)) s =
      some (fun _ => (-3 : Int)) ∧
    FP.Structural.evalOp M none (.mod .int .nil (.constInt (-7)) (.constInt 3)) s =
      some (fun _ => (2 : Int)) ∧
    FP.Structural.evalOp M none (.floorDiv .int .nil (.constInt 7) (.constInt (-3))) s =
      some (fun _ => (-2 : Int)) ∧
    FP.Structural.evalOp M none (.mod .int .nil (.constInt 7) (.constInt (-3))) s =
      some (fun _ => (1 : Int)) := by
  change (some (fun _ : PUnit => (-7 : Int) / 3) = some (fun _ => (-3 : Int))) ∧
    (some (fun _ : PUnit => (-7 : Int) % 3) = some (fun _ => (2 : Int))) ∧
    (some (fun _ : PUnit => (7 : Int) / (-3)) = some (fun _ => (-2 : Int))) ∧
    (some (fun _ : PUnit => (7 : Int) % (-3)) = some (fun _ => (1 : Int)))
  norm_num

theorem zero_divisor (M : Algebra α) (s : State α) :
    FP.Structural.evalOp M none (.floorDiv .nat .nil (.constNat 7) (.constNat 0)) s =
      some (fun _ => 0) ∧
    FP.Structural.evalOp M none (.mod .nat .nil (.constNat 7) (.constNat 0)) s =
      some (fun _ => 7) ∧
    FP.Structural.evalOp M none (.floorDiv .int .nil (.constInt (-7)) (.constInt 0)) s =
      some (fun _ => (0 : Int)) ∧
    FP.Structural.evalOp M none (.mod .int .nil (.constInt (-7)) (.constInt 0)) s =
      some (fun _ => (-7 : Int)) := by
  change (some (fun _ : PUnit => 7 / 0 : PUnit → Nat) = some (fun _ => 0)) ∧
    (some (fun _ : PUnit => 7 % 0 : PUnit → Nat) = some (fun _ => 7)) ∧
    (some (fun _ : PUnit => (-7 : Int) / 0) = some (fun _ => (0 : Int))) ∧
    (some (fun _ : PUnit => (-7 : Int) % 0) = some (fun _ => (-7 : Int)))
  norm_num

theorem missing_divisor_fails (M : Algebra α) (s : State α)
    (h : s.regs .nat [] "divisor" = none) :
    FP.Structural.evalOp M none (.floorDiv .nat .nil (.constNat 7)
      (.ref .nat [] "divisor")) s = none ∧
    FP.Structural.evalOp M none (.mod .nat .nil (.constNat 7)
      (.ref .nat [] "divisor")) s = none := by
  simp [FP.Structural.evalOp, h]

#guard_msgs (drop info) in
#auditModuleAxioms
end FPIntegralExecutionTests
