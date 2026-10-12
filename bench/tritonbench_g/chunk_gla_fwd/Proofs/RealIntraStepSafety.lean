import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealIntraExecution
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealIntraAccessSafety

/-! Safety of the full prelude and a streamed column. Bounds apply to the
actual pointers/masks after prior statements, including every in-loop store. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraStepSafety
open VeriTile.Triton RealIntraMath RealIntraProgram RealIntraMemory RealIntraSetup
open RealIntraAccessSafety
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem setup_safe (p : Layout) (l : Launch) (r : Regions) (s : BlockState)
    (hg : Grid l s) (bounds : RegionBounds)
    (hq : MatrixBounds p l r.query bounds) (hgate : MatrixBounds p l r.gate bounds) :
    Stmt.TraceSafeList bounds (setup p r) s := by
  unfold setup
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (show evalOp (.arange p.BC) s = some (indices p.BC) from by rw [evalOp_arange]; rfl))]
  refine ⟨by simp [Stmt.TraceSafe,offsetsOp,rowsOp,queryPtr,keyPtr,timeOp,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (show evalOp (.arange p.BK) _ = some (indices p.BK) from by rw [evalOp_arange]; rfl))]
  refine ⟨by simp [Stmt.TraceSafe,offsetsOp,rowsOp,queryPtr,keyPtr,timeOp,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (offsets_eval p l _
    (by constructor <;> simp [hg.chunk,hg.head,hg.row,hg.column])))]
  refine ⟨by simp [Stmt.TraceSafe,offsetsOp,rowsOp,queryPtr,keyPtr,timeOp,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (key_mask_eval p _ (by simp)))]
  refine ⟨by simp [Stmt.TraceSafe,offsetsOp,rowsOp,queryPtr,keyPtr,timeOp,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (row_mask_eval p l _
    (by constructor <;> simp [hg.chunk,hg.head,hg.row,hg.column])))]
  refine ⟨by simp [Stmt.TraceSafe,offsetsOp,rowsOp,queryPtr,keyPtr,timeOp,times,Op.SafeAt],?_⟩
  change Stmt.TraceSafeList bounds (.assign _ _ "p_q" _ :: _) (indexed p l s)
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (query_eval p l r.query _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column])))]
  refine ⟨by simp [Stmt.TraceSafe,offsetsOp,rowsOp,queryPtr,keyPtr,timeOp,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (query_eval p l r.gate _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column])))]
  refine ⟨by simp [Stmt.TraceSafe,offsetsOp,rowsOp,queryPtr,keyPtr,timeOp,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (key_eval p l r.key _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column]) (by simp [indexed])))]
  refine ⟨by simp [Stmt.TraceSafe,offsetsOp,rowsOp,queryPtr,keyPtr,timeOp,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (key_eval p l r.gate _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column]) (by simp [indexed])))]
  refine ⟨by simp [Stmt.TraceSafe,offsetsOp,rowsOp,queryPtr,keyPtr,timeOp,times,Op.SafeAt],?_⟩
  change Stmt.TraceSafeList bounds (.assign _ _ "b_q" _ :: _) (pointed p l r s)
  apply TraceProofs.assign_cons (matrix_safe p l r.query "p_q" bounds _ hq (by simp [pointed]))
  intro bq
  apply TraceProofs.assign_cons (matrix_safe p l r.gate "p_g" bounds _ hgate (by simp [pointed]))
  intro bg
  exact Stmt.TraceSafeList.nil_intro

structure Bounds (p : Layout) (l : Launch) (r : Regions) (j : Nat) (bounds : RegionBounds) : Prop where
  key : KeyBounds p l r.key j bounds
  gate : KeyBounds p l r.gate j bounds
  output : OutputBounds p l r.output j bounds

theorem iteration_safe (p : Layout) (l : Launch) (r : Regions) (scale : ℝ)
    (j : Nat) (swapped : Bool) (s0 s : BlockState) (bounds : RegionBounds) (hb : Bounds p l r j bounds)
    (hc : Cached p l r s0 s)
    (hj : s.regs .nat [] "j" = some (Tile.scalar j))
    (hp : s.regs .ptr [p.BK] "p_k" = some (keyPointer p l r.key j))
    (hpg : s.regs .ptr [p.BK] "p_gk" = some (keyPointer p l r.gate j)) :
    Stmt.TraceSafeList bounds (iteration p r scale swapped) s := by
  unfold iteration
  apply TraceProofs.assign_cons (key_safe p l r.key j "p_k" bounds s hb.key hp hc.keyMask)
  intro bk
  apply TraceProofs.assign_cons (key_safe p l r.gate j "p_gk" bounds _ hb.gate
    (by simpa using hpg) (by simpa using hc.keyMask))
  intro bgk
  apply TraceProofs.assign_cons
  · simp [reducedOp,product,Op.SafeAt,Op.SafeAt.eq_def]
  intro bA
  apply TraceProofs.assign_cons
  · cases swapped <;> simp [scaledOp,Op.SafeAt]
  intro bA'
  apply Stmt.TraceSafeList.cons_intro (output_safe p l r.output j bounds _ hb.output
    (by simpa using hc.offsets) (by simpa using hj) (by simpa using hc.rowMask))
  intro t _
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe,advance,Op.SafeAt]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraStepSafety
