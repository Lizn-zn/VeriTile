import bench.tritonbench_g.softmax_optimize.Proofs.SafetyState
import bench.tritonbench_g.softmax_optimize.Proofs.SignedBodies

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
open VeriTile.Triton SignedTransport

/-- Source loops and the intervening reduction retain both address-control registers. -/
theorem source_bodies_avoid (name : RegName)
    (hname : name = "pid_m" ∨ name = "prev_multiple")
    (Y X : RegionName) (N B : Nat) (body : List Stmt)
    (hbody : body ∈ [inputFull X N B, inputTail X N B,
      outputFull Y X N B, outputTail Y X N B, reductionStmts B]) :
    ∀ st ∈ body, AvoidsRegister name st := by
  rcases hname with rfl | rfl
  all_goals
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hbody
    rcases hbody with rfl | rfl | rfl | rfl | rfl <;>
      simp [inputFull, inputTail, outputFull, outputTail, indexPrefix,
      maskStatement, outputPointer, loopAbody, loopBbody, loopApass2body,
      loopBpass2body, reductionStmts, AvoidsRegister]

theorem source_bodies_preserve (name : RegName)
    (hname : name = "pid_m" ∨ name = "prev_multiple")
    (dtype : TileDType) (shape : TileShape)
    (Y X : RegionName) (N B : Nat) (body : List Stmt)
    (hbody : body ∈ [inputFull X N B, inputTail X N B,
      outputFull Y X N B, outputTail Y X N B, reductionStmts B])
    (s t : BlockState) (hrun : stepStmts body s = some t) :
    t.regs dtype shape name = s.regs dtype shape name := by
  exact list_preserves_register name dtype shape body s t
    (source_bodies_avoid name hname Y X N B body hbody) hrun

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
