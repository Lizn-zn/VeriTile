import bench.tritonbench_g.flash_attn.Proofs.RealLoop
import VeriTile.Triton.Memory.FlattenTolerant
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula NormalizedSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem flattenOk (Q K V L O : RegionName) (p : Layout) :
    (algorithm Q K V L O p).FlattenOk := by
  change StmtList.FlattenOk (algorithm Q K V L O p).body
  run_tac do
    let g ← Lean.Elab.Tactic.getMainGoal
    let t ← g.getType
    let b ← Lean.Meta.whnf t.appArg!
    Lean.Elab.Tactic.replaceMainGoal [← g.change (Lean.mkApp t.appFn! b)]
  simp only [StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk,
    List.forall_mem_cons,List.forall_mem_nil]
  simp
  simp [↓Op.FlattenOk.eq_def]

theorem before_storeFree (Q K V L O : RegionName) (p : Layout) :
    (before Q K V L O p).all storeFree=Bool.true := by
  simp only [before,algorithm,reduced,List.take_succ_cons,List.take_zero]
  simp [List.all_cons,List.all_nil,storeFree]

theorem body_storeFree (p : Layout) : (body p).all storeFree=Bool.true := by
  simp only [body,algorithm,reduced,List.getElem?_cons_zero,List.getElem?_cons_succ]
  simp [List.all_cons,List.all_nil,storeFree]

theorem before_memory (Q K V L O : RegionName) (p : Layout) (s t : BlockState)
    (he : stepStmts (before Q K V L O p) s=some t) : t.mem=s.mem :=
  storeFree_stepStmts_mem _ _ _ (before_storeFree Q K V L O p) he

theorem body_memory (p : Layout) (s t : BlockState)
    (he : stepStmts (body p) s=some t) : t.mem=s.mem :=
  storeFree_stepStmts_mem _ _ _ (body_storeFree p) he

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
