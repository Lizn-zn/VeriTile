import FlashAttnDraft.TypedPost
import VeriTile.Triton.Float.Countermodel
namespace VeriTile.Bench.TritonBenchG.FlashAttn.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem source_parts (input output : Storage) (causal : Bool) :
    (original input output causal).kernel.surfaceBody =
      (parts input output causal).before ++
      .forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi") (.constNat 2)
        (parts input output causal).body :: (parts input output causal).after := by
  cases input <;> cases output <;> rfl

theorem loop_executes (input output : Storage) (causal : Bool) :
    step (engine model) (.forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi")
      (.constNat 2) (parts input output causal).body) (entry input output causal).val =
      some (afterLoop input output causal) := by
  have hlo : (entry input output causal).val.regs .nat [] "lo"=some (fun _ => 0) := by
    cases input <;> cases output <;> cases causal <;> rfl
  have hhi : (entry input output causal).val.regs .nat [] "hi"=some (fun _ => if causal then 2 else 3) := by
    cases input <;> cases output <;> cases causal <;> rfl
  simp only [step,evalOp_unfold,hlo,hhi,Option.bind_some]
  cases causal
  · change range (engine model) "start_n" 0 3 2 (parts input output Bool.false).body
      (entry input output Bool.false).val=some (second input output Bool.false).val
    rw [range]
    change (run (engine model) (parts input output Bool.false).body
      ((entry input output Bool.false).val.setReg "start_n" .nat [] (fun _ => 0))).bind
      (range (engine model) "start_n" 2 3 2 (parts input output Bool.false).body) = _
    simp only [(first input output Bool.false).property,Option.bind_some]
    rw [range]
    change (run (engine model) (parts input output Bool.false).body
      ((first input output Bool.false).val.setReg "start_n" .nat [] (fun _ => 2))).bind
      (range (engine model) "start_n" 4 3 2 (parts input output Bool.false).body) = _
    simp only [(second input output Bool.false).property,Option.bind_some]
    simp [range]
  · change range (engine model) "start_n" 0 2 2 (parts input output Bool.true).body
      (entry input output Bool.true).val=some (first input output Bool.true).val
    rw [range]
    change (run (engine model) (parts input output Bool.true).body
      ((entry input output Bool.true).val.setReg "start_n" .nat [] (fun _ => 0))).bind
      (range (engine model) "start_n" 2 2 2 (parts input output Bool.true).body) = _
    simp only [(first input output Bool.true).property,Option.bind_some]
    simp [range]

/-- Complete nonempty executions: both causal modes, all Q/O storage choices,
nonzero head ID, strided loads/stores, and an unaligned noncausal final KV tile.
The singleton algebra witnesses typed execution only, not physical IEEE values. -/
theorem original_execution (input output : Storage) (causal : Bool) :
    FP.Structural.exec (engine model) (original input output causal).kernel
      (initial input output)=some (post input output causal).val := by
  change run (engine model) (original input output causal).kernel.surfaceBody _ = _
  rw [source_parts,run_append]
  simp only [(entry input output causal).property,Option.bind_some]
  simp only [run,loop_executes,bind,Option.bind_some,(post input output causal).property]

theorem candidate_execution (input output : Storage) (causal : Bool) :
    FP.Structural.exec (engine model)
      (FPProof.program Bool.true input output "Q" "K" "V" "L" "O" 1
        50 20 5 2 50 20 7 2 50 20 99 3 50 20 5 2 1 2 3 2 2 2 causal).kernel
      (initial input output)=some (post input output causal).val := by
  have h := FP.ProfiledRewrite.sound
    (FPProof.equivalent input output "Q" "K" "V" "L" "O" 1
      50 20 5 2 50 20 7 2 50 20 99 3 50 20 5 2 1 2 3 2 2 2 causal)
    PUnit model (fun _ _ => True) (by apply FP.models_of_atoms;intro e he;cases he)
    (initial input output) (by simp [FPProof.program])
  exact h.symm.trans (original_execution input output causal)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.TypedExecution
