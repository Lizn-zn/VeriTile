import bench.tritonbench_g.softmax_optimize.Proofs.SignedState
import bench.tritonbench_g.softmax_optimize.Proofs.SignedBodies

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton
set_option maxHeartbeats 3000000

def LiveStatementsRelated (a b : List Stmt) : Prop := ∀ s t, LiveRelated s t →
  ResultsRelated LiveRelated (stepStmts a s) (stepStmts b t)

theorem live_nil : LiveStatementsRelated [] [] := by
  intro s t h
  simpa only [stepStmts.nil, ResultsRelated] using h

theorem live_cons {a b : Stmt} {left right : List Stmt}
    (hab : ∀ s t, LiveRelated s t → ResultsRelated LiveRelated (stepStmt a s) (stepStmt b t))
    (hrest : LiveStatementsRelated left right) : LiveStatementsRelated (a::left) (b::right) := by
  intro s t h
  have hh := results_bind (hab s t h) (stepStmts left) (stepStmts right) hrest
  cases hs : stepStmt a s <;> cases ht : stepStmt b t <;>
    simpa only [stepStmts, hs, ht, Option.bind] using hh

theorem live_assign {dtype : TileDType} {shape : TileShape} (name : RegName)
    (hn : Shared name) {a b : Op dtype shape}
    (he : ∀ s t, LiveRelated s t → evalOp a s = evalOp b t) :
    ∀ s t, LiveRelated s t → ResultsRelated LiveRelated
      (stepStmt (.assign dtype shape name a) s) (stepStmt (.assign dtype shape name b) t) := by
  intro s t h
  simp only [stepStmt, he s t h]
  cases evalOp b t with
  | none => exact True.intro
  | some v => exact h.setShared name dtype shape v hn

theorem reduction_related (B : Nat) :
    LiveStatementsRelated (reductionStmts B) (reductionStmts B) := by
  unfold reductionStmts
  apply live_cons (live_assign "final_m" (by simp [Shared,Scratch]) ?_)
  · apply live_cons (live_assign "z" (by simp [Shared,Scratch]) ?_)
    · apply live_cons (live_assign "m" (by simp [Shared,Scratch]) ?_) live_nil
      intro s t h
      simpa only [evalOp_ref] using h.regs_eq .real [] "final_m" (by simp [Shared,Scratch])
    · intro s t h
      have hm := h.regs_eq .real [B] "m" (by simp [Shared,Scratch])
      have hz := h.regs_eq .real [B] "z" (by simp [Shared,Scratch])
      have hf := h.regs_eq .real [] "final_m" (by simp [Shared,Scratch])
      simp only [evalOp.eq_def, hm, hz, hf]
  · intro s t h
    have hm := h.regs_eq .real [B] "m" (by simp [Shared,Scratch])
    simp only [evalOp.eq_def, hm]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
