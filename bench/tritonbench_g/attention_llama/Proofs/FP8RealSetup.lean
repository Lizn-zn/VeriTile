import bench.tritonbench_g.attention_llama.Proofs.FP8RealSimulation
import bench.tritonbench_g.attention_llama.Proofs.FP8RealLoop

/-! The actual prologue and causal bound establish the byte-loop cursor.
Only register and address facts are required by the execution simulation. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealSetup
open VeriTile.Triton RealKernel FP8RealMemory FP8RealLoop
set_option maxRecDepth 16000
set_option maxHeartbeats 500000

def stop (s : BlockState) (N SP BN : Nat) (causal : Bool) : Nat :=
  if causal then (s.pids 0 + 1) * BN + SP else N

section
variable (Q K V : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H N SP BM BN BD : Nat)
    (causal : Bool)

def body : List Stmt :=
  alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H N BM BN BD ++
    [.ifThen (.constBool causal) [alBnEndCausalStmt SP BN]]

/-- Initialization establishes the pointer cursor and the exact source bound;
no accumulator or output equation is assumed. -/
theorem runs_with_launch (s : BlockState) (hundef : ∀ r o, s.undef r o = 0) :
    ∃ entry, stepStmts (body Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
        NH H N SP BM BN BD causal) s = some entry ∧
      Cursor K V s (alBase s NH skz skh) (alBase s NH svz svh) skn skk svk svn BN BD 0 entry ∧
      entry.regs .nat [] "block_n_end" = some (Tile.scalar (stop s N SP BN causal)) ∧
      FP8RealLoopFrame.Launch s NH entry := by
  obtain ⟨p, hp, hpids, hm, _, hstart, hbatch, hhead, _, hn, _, _, _, _, hk, hv, hb⟩ :=
    alPreLoop_evalG s Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      NH H N BM BN BD hundef
  simp only [body, ProgramRewrite.append, hp, Option.bind_some,
    ProgramRewrite.constant_branch]
  cases causal with
  | «false» =>
    refine ⟨p, ?_, ⟨hm, hn, hk, hv⟩, hb, ⟨hpids, hbatch, hhead⟩⟩
    simp [stepStmts.nil]
  | «true» =>
    have he : evalOp
        (.add .nat .nil (.mul .nat .nil
          (.add .nat .nil (.ref .nat [] "start_m") (.constNat 1)) (.constNat BN))
          (.constNat SP)) p = some (Tile.scalar ((s.pids 0 + 1) * BN + SP)) := by
      rw [evalOp_add, evalOp_mul, evalOp_add]
      simp only [evalOp_ref, hstart, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      apply congrArg some
      apply Tile.ext
      intro i
      rfl
    refine ⟨p.setReg "block_n_end" .nat [] (Tile.scalar ((s.pids 0 + 1) * BN + SP)), ?_, ?_, ?_, ?_⟩
    · simp only [↓reduceIte, alBnEndCausalStmt,
        stepStmts.cons_some (stepStmt_assign_eq_some he), stepStmts.nil, Option.bind_some]
    · exact ⟨hm, by simpa using hn, by simpa using hk, by simpa using hv⟩
    · simp [stop]
    · exact ⟨hpids, by simpa using hbatch, by simpa using hhead⟩

/-- The input cursor and exact loop bound, without the extra output metadata. -/
theorem runs (s : BlockState) (hundef : ∀ r o, s.undef r o = 0) :
    ∃ entry, stepStmts (body Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
        NH H N SP BM BN BD causal) s = some entry ∧
      Cursor K V s (alBase s NH skz skh) (alBase s NH svz svh) skn skk svk svn BN BD 0 entry ∧
      entry.regs .nat [] "block_n_end" = some (Tile.scalar (stop s N SP BN causal)) := by
  obtain ⟨entry, he, hc, hb, _⟩ := runs_with_launch Q K V sqz sqh sqm sqk skz skh skn skk
    svz svh svk svn NH H N SP BM BN BD causal s hundef
  exact ⟨entry, he, hc, hb⟩

/-- The complete setup commutes with the decoded K/V view. -/
theorem decoded_execution (s : BlockState) (hQK : Q ≠ K) (hQV : Q ≠ V) :
    (stepStmts (body Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      NH H N SP BM BN BD causal) s).map (decoded K V) =
    stepStmts (body Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      NH H N SP BM BN BD causal) (decoded K V s) := by
  have hp := FP8RealSimulation.preamble Q K V s sqz sqh sqm sqk skz skh skn skk
    svz svh svk svn NH H N BM BN BD hQK hQV
  simp only [body, ProgramRewrite.append, ← hp]
  cases he : stepStmts (alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      NH H N BM BN BD) s with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.bind_some, ProgramRewrite.constant_branch]
    cases causal with
    | «false» => simp [stepStmts.nil]
    | «true» =>
      simp only [↓reduceIte, List.append_nil, alBnEndCausalStmt]
      apply pure_cons (hpure := by simp [Op.SafeAt.eq_def])
      intro tile
      simp [stepStmts.nil]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealSetup
