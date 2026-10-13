import bench.tritonbench_g.flash_attn.Proofs.NumericTypedPost
import VeriTile.Triton.Float.Countermodel
namespace VeriTile.Bench.TritonBenchG.FlashAttn.NumericTypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem source_parts (causal : Bool) :
    (original causal).kernel.surfaceBody =
      (parts causal).before ++
      .forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi") (.constNat 2)
        (parts causal).body :: (parts causal).after := by
  rfl

theorem loop_executes (causal : Bool) :
    step (engine model) (.forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi")
      (.constNat 2) (parts causal).body) (entry causal).val =
      some (afterLoop causal) := by
  have hlo : (entry causal).val.regs .nat [] "lo"=some (fun _ => 0) := by
    cases causal <;> rfl
  have hhi : (entry causal).val.regs .nat [] "hi"=some (fun _ => if causal then 2 else 3) := by
    cases causal <;> rfl
  simp only [step,evalOp_unfold,hlo,hhi,Option.bind_some]
  cases causal
  · change range (engine model) "start_n" 0 3 2 (parts Bool.false).body
      (entry Bool.false).val=some (second).val
    rw [range]
    change (run (engine model) (parts Bool.false).body
      ((entry Bool.false).val.setReg "start_n" .nat [] (fun _ => 0))).bind
      (range (engine model) "start_n" 2 3 2 (parts Bool.false).body) = _
    simp only [(first Bool.false).property,Option.bind_some]
    rw [range]
    change (run (engine model) (parts Bool.false).body
      ((first Bool.false).val.setReg "start_n" .nat [] (fun _ => 2))).bind
      (range (engine model) "start_n" 4 3 2 (parts Bool.false).body) = _
    simp only [(second).property,Option.bind_some]
    simp [range]
  · change range (engine model) "start_n" 0 2 2 (parts Bool.true).body
      (entry Bool.true).val=some (first Bool.true).val
    rw [range]
    change (run (engine model) (parts Bool.true).body
      ((entry Bool.true).val.setReg "start_n" .nat [] (fun _ => 0))).bind
      (range (engine model) "start_n" 2 2 2 (parts Bool.true).body) = _
    simp only [(first Bool.true).property,Option.bind_some]
    simp [range]

/-- Complete nonempty executions: both causal modes, fp16 Q/O storage,
nonzero head ID, strided loads/stores, and an unaligned noncausal final KV tile.
The exact-real backend retains numerical values and negative infinity; it does not model IEEE rounding. -/
theorem original_execution (causal : Bool) :
    FP.Structural.exec (engine model) (original causal).kernel
      (initial .fp16 .fp16)=some (post causal).val := by
  change run (engine model) (original causal).kernel.surfaceBody _ = _
  rw [source_parts,run_append]
  simp only [(entry causal).property,Option.bind_some]
  simp only [run,loop_executes,bind,Option.bind_some,(post causal).property]

theorem candidate_execution (causal : Bool) :
    FP.Structural.exec (engine model)
      (FPProof.program Bool.true .fp16 .fp16 "Q" "K" "V" "L" "O" 1
        50 20 5 2 50 20 7 2 50 20 99 3 50 20 5 2 1 2 3 2 2 2 causal).kernel
      (initial .fp16 .fp16)=some (post causal).val := by
  have h := FP.ProfiledRewrite.sound
    (FPProof.equivalent .fp16 .fp16 "Q" "K" "V" "L" "O" 1
      50 20 5 2 50 20 7 2 50 20 99 3 50 20 5 2 1 2 3 2 2 2 causal)
    (WithBot ℝ) model (fun _ _ => True) (by apply FP.models_of_atoms;intro e he;cases he)
    (initial .fp16 .fp16) (by simp [FPProof.program])
  exact h.symm.trans (original_execution causal)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.NumericTypedExecution
