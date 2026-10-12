import VeriTile.Triton.Math.BroadcastRewrite
import VeriTile.Triton.Memory.FlattenR

/-! The real reciprocal rewrite also preserves the safety contract used by
streamed IO under identity rounding. No nonidentity-rounding claim is made. -/
namespace VeriTile.Triton.RealBroadcastRewrite

/-- Rewriting the numerical expression introduces no new memory accesses. -/
theorem contextual_traceSafeR_triv (before after : List Stmt)
    (bc : Broadcast sa (n :: rest) shape) (out : RegName)
    (a : Op .real sa) (b : Op .real (n :: rest)) (one : ℝ) (hone : one = 1)
    (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds
        (before ++ .assign .real shape out (.div .real bc a b) :: after) s ↔
      Stmt.TraceSafeListR .triv bounds (before ++ .assign .real shape out
        (.mul .real bc a (.div .real .scalarL (.const one) b)) :: after) s := by
  induction before generalizing s with
  | nil =>
      have hs : Stmt.TraceSafeR .triv bounds (.assign .real shape out (.div .real bc a b)) s ↔
          Stmt.TraceSafeR .triv bounds (.assign .real shape out
            (.mul .real bc a (.div .real .scalarL (.const one) b))) s := by
        conv_lhs =>
          rw [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
          dsimp only
        conv_rhs =>
          rw [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
          dsimp only
          arg 2
          rw [Op.SafeAtR.eq_def]
          dsimp only
          arg 1
          rw [Op.SafeAtR.eq_def]
          dsimp only
        simp only [true_and]
      simp only [List.nil_append, Stmt.TraceSafeListR, hs, stepStmtR_triv,
        assign_div_mul_rcp bc out a b one hone]
  | cons st tail ih =>
      simp only [List.cons_append, Stmt.TraceSafeListR, stepStmtR_triv]
      cases stepStmt st s with
      | none => rfl
      | some t => exact and_congr_right (fun _ => ih t)

end VeriTile.Triton.RealBroadcastRewrite
