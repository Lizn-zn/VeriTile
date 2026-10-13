import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FlashAttn
open VeriTile.Triton
noncomputable section
/-- Algorithm semantics preserve the typed-dot casts without introducing rounding. -/
theorem legacy_typed_dot (M K N : Nat) (x : Op .fp16 [M,K]) (y : Op .real [K,N])
    (s : BlockState) :
    evalOp (.castFloat .fp32 .real (.dotTyped .fp16 .real .fp32 (batch := []) x y)) s =
    evalOp (.dot (batch := []) (.castFloat .fp16 .real x) y) s := by
  simp only [evalOp_castFloat,evalOp_dotTyped,evalOp_dot]
  erw [evalOp_castFloat]
  cases hx : evalOp x s <;> cases hy : evalOp y s <;>
    simp [hx,hy,FloatDType.cast]
  all_goals rfl

theorem legacy_typed_dot_head (M K N : Nat) (name : RegName)
    (acc : Op .real [M,N]) (x : Op .fp16 [M,K]) (y : Op .real [K,N])
    (tail : List Stmt) (s : BlockState) :
    stepStmts (.assign .real [M,N] name
      (.add .real (.consSame (.consSame .nil)) acc
        (.castFloat .fp32 .real (.dotTyped .fp16 .real .fp32 (batch := []) x y))) :: tail) s =
    stepStmts (.assign .real [M,N] name
      (.add .real (.consSame (.consSame .nil)) acc
        (.dot (batch := []) (.castFloat .fp16 .real x) y)) :: tail) s := by
  simp only [stepStmts,stepStmt,evalOp_add]
  erw [legacy_typed_dot]
  rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn
