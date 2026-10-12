import bench.tritonbench_g.context_attn_fwd.Proofs.WholeProgram
import bench.tritonbench_g.context_attn_fwd.Proofs.LoopExecution
import bench.tritonbench_g.context_attn_fwd.Proofs.EpilogueExecution
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.WholeExecution
open VeriTile.Triton Formula Program PreludeProgram PreludeComplete
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.Math (gStateBot gStateBot_score_congr)
noncomputable section
set_option maxHeartbeats 1400000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

def numerator (p : Layout) (r : Regions) (u : BlockState) (scale : ℝ)
    (i : Fin p.BM) (d : Fin p.D) : ℝ :=
  (gStateBot (count p (launch u) (inputs u r)*p.BN) (count p (launch u) (inputs u r)*p.BN)
    (Formula.pairs p (launch u) (inputs u r) scale i.val d.val)).2.2

def denominator (p : Layout) (r : Regions) (u : BlockState) (scale : ℝ) (i : Fin p.BM) : ℝ :=
  (gStateBot (count p (launch u) (inputs u r)*p.BN) (count p (launch u) (inputs u r)*p.BN)
    (Formula.pairs p (launch u) (inputs u r) scale i.val 0)).2.1

def output (p : Layout) (r : Regions) (u : BlockState) (scale : ℝ)
    (i : Fin p.BM) (d : Fin p.D) : ℝ :=
  Formula.expected p (launch u) (inputs u r) scale i.val d.val

theorem output_agrees (p : Layout) (r : Regions) (u : BlockState) (scale : ℝ)
    (i : Fin p.BM) (d : Fin p.D) :
    numerator p r u scale i d / denominator p r u scale i = output p r u scale i d := by
  unfold numerator denominator output
  rw [(gStateBot_score_congr _ _
    (Formula.pairs p (launch u) (inputs u r) scale i.val 0)
    (Formula.pairs p (launch u) (inputs u r) scale i.val d.val) (by intros; rfl)).2]
  exact Formula.formula_agrees p (launch u) (inputs u r) scale i.val d.val

theorem loop_executes (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) (u : BlockState) :
    stepStmt (WholeProgram.loop p r swapped scale) (initialized p r u) =
      stepForRangeIntAux "start_n" 0 (Bounds.stop p (launch u) (inputs u r)) p.BN
        (LoopProgram.body p r swapped scale) (initialized p r u) := by
  simp only [WholeProgram.loop,stepForRangeIntAux.forRangeInt_unfold,evalOp.eq_def,n,z,initialized]
  simp [PreludeQuery.queried,Tile.scalar,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,
    Bounds.stop,Formula.limit,Formula.sequence,Formula.batch,launch,NumericDType.mul,mul_comm,ite_mul,
    Nat.cast_mul,Nat.cast_add,-Tile.scalar_eta]

/-- Both storage-parametric sources execute successfully and reach their only
output store. The stored quotient equals the independent initial-array formula. -/
theorem complete (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) (scale : ℝ)
    (initial : BlockState) (hBN : 0 < p.BN) :
    ∃ mid,
      Invariant.Static p r initial mid ∧
      stepStmts (kernel p r storage swapped scale).eraseDType.toAlgKernel.body initial =
        some (EpilogueExecution.prepared p r (initial.pids 1%p.heads)
          ((inputs initial r).start (initial.pids 1/p.heads)) (Formula.sequence p (launch initial) (inputs initial r))
          (fun i => queryRow p (launch initial) i.val) (numerator p r initial scale) (denominator p r initial scale) mid) := by
  obtain ⟨mid,hl,hs,hd⟩ := LoopExecution.complete p r swapped scale initial hBN
  refine ⟨mid,hs,?_⟩
  rw [Program.storage_erasure,WholeProgram.source,List.append_assoc,stepStmts.append_some (PreludeComplete.executes p r initial)]
  simp only [List.cons_append,List.nil_append]
  rw [stepStmts.cons_some (by rw [loop_executes]; exact hl)]
  exact EpilogueExecution.executes p r _ _ _ _ _ _ mid hs.head hs.start hs.sequence hs.rows hs.channels
    hd.denominator hd.accumulator

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.WholeExecution
