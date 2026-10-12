import bench.tritonbench_g.context_attn_bloom.Proofs.WholeProgram
import bench.tritonbench_g.context_attn_bloom.Proofs.EpilogueExecution
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.WholeExecution
open VeriTile.Triton Math Program RealSetup PreludeProgram
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def output (p : Layout) (r : Regions) (u : BlockState) (scale : ℝ)
    (i : Fin p.BM) (d : Fin p.D) : ℝ :=
  Formula.expected p (launch u) (inputs u r) scale i.val d.val
    (count p (launch u) (inputs u r))

theorem final_accumulator (p : Layout) (r : Regions) (initial state : BlockState) (scale : ℝ)
    (hBN : 0 < p.BN) (hp : 0 ≤ (inputs initial r).prompt (initial.pids 0))
    (hd : Invariant.Dynamic p r initial scale (count p (launch initial) (inputs initial r)) state) :
    state.regs .real [p.BM,p.D] "acc" =
      some (Tile.mat (fun i d => some (output p r initial scale i d))) := by
  have h := hd.accumulator
  change state.regs .real [p.BM,p.D] "acc" =
    some (Tile.mat (fun i d => some (Math.formula p (launch initial) (inputs initial r) scale i.val d.val))) at h
  simpa only [Formula.formula_eq p (launch initial) (inputs initial r) scale _ _ hBN hp,output] using h

theorem loop_executes (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (u : BlockState) :
    stepStmt (WholeProgram.loop p r swapped scale) (initialized p r u) =
      stepForRangeIntAux "start_n" 0 (Bounds.stop p (launch u) (inputs u r)) p.BN
        (LoopProgram.body p r swapped scale) (initialized p r u) := by
  simp only [WholeProgram.loop,stepForRangeIntAux.forRangeInt_unfold,evalOp.eq_def,
    n,z,initialized]
  simp [Tile.scalar,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,
    Bounds.stop,launch,NumericDType.mul,mul_comm,ite_mul,-Tile.scalar_eta]

theorem complete (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) (scale : ℝ)
    (initial : BlockState) (hBN : 0 < p.BN)
    (hp : 0 ≤ (inputs initial r).prompt (initial.pids 0)) :
    ∃ mid,
      Invariant.Static p r initial mid ∧
      stepStmts (kernel p r storage swapped scale).eraseDType.toAlgKernel.body initial =
        some (EpilogueExecution.prepared p r (initial.pids 1)
          ((inputs initial r).start (initial.pids 0)) (sequence (inputs initial r) (launch initial))
          (fun i => queryRow p (launch initial) i.val) (output p r initial scale) mid) := by
  obtain ⟨mid,hl,hs,hd⟩ := LoopExecution.complete p r swapped scale initial hBN hp
  refine ⟨mid,hs,?_⟩
  rw [Program.storage_erasure,WholeProgram.source,List.append_assoc,stepStmts.append_some (prepares p r initial)]
  simp only [List.cons_append,List.nil_append]
  rw [stepStmts.cons_some (by rw [loop_executes]; exact hl)]
  exact EpilogueExecution.executes p r _ _ _ _ _ mid hs.head hs.start hs.sequence hs.rows hs.channels
    (final_accumulator p r initial mid scale hBN hp hd)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.WholeExecution
