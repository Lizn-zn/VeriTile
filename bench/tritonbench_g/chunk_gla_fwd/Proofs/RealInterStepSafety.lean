import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterStep
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterAccessSafety

/-! Trace safety of the complete K iteration. Each memory bound is checked
at the actual state produced by the preceding source statements. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterStepSafety
open VeriTile.Triton RealInterMath RealInterProgram RealInterPointers RealInterLoads RealInterAlgebra
open RealInterStep RealInterAccessSafety
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (r : Regions) (kb : Nat) (bounds : RegionBounds) : Prop where
  query : QueryBounds p l r.query kb bounds
  queryGate : QueryBounds p l r.gate kb bounds
  key : KeyBounds p l r.key kb bounds
  keyGate : KeyBounds p l r.gate kb bounds
  reference : ReferenceBounds p l r.gate kb bounds

theorem iteration_safe (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ)
    (kb : Nat) (swapped : Bool) (s : BlockState) (bounds : RegionBounds) (hb : Bounds p l r kb bounds)
    (hg : Grid p l s) (hk : s.regs .nat [] "i_k" = some (Tile.scalar kb)) :
    Stmt.TraceSafeList bounds (iteration p r scale swapped) s := by
  unfold iteration
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (indices_eval p kb s hk))]
  refine ⟨by simp [Stmt.TraceSafe,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (mask_eval p kb _ (by simp)))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  change Stmt.TraceSafeList bounds (.assign _ _ "p_q" _ :: _) (indexed p l r a scale kb s)
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (query_eval p l r.query kb _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column]) (by simp [indexed,hk])))]
  refine ⟨by rw [Stmt.TraceSafe]; exact query_pointer_safe p _ bounds _,?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (query_eval p l r.gate kb _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column]) (by simp [indexed,hk])))]
  refine ⟨by rw [Stmt.TraceSafe]; exact query_pointer_safe p _ bounds _,?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (key_eval p l r.key kb _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column]) (by simp [indexed,hk])))]
  refine ⟨by rw [Stmt.TraceSafe]; exact key_pointer_safe p _ bounds _,?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (key_eval p l r.gate kb _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column]) (by simp [indexed,hk])))]
  refine ⟨by rw [Stmt.TraceSafe]; exact key_pointer_safe p _ bounds _,?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (reference_eval p l r.gate kb _
    (by constructor <;> simp [indexed,hg.chunk,hg.head,hg.row,hg.column]) (by simp [indexed])))]
  refine ⟨by rw [Stmt.TraceSafe]; exact reference_pointer_safe p _ bounds _,?_⟩
  change Stmt.TraceSafeList bounds (.assign _ _ "b_gn" _ :: _) (pointed p l r a scale kb s)
  apply TraceProofs.assign_cons (reference_safe p l r.gate kb bounds _ hb.reference
    (by simp [pointed]) (by simp [pointed,indexed]))
  intro bgn
  apply TraceProofs.assign_cons (load2_safe r.query _ _ _ p.BC p.BK _ _ _ _ "p_q" bounds _ hb.query (by simp [pointed,queryPointer,keyPointer]))
  intro bq
  apply TraceProofs.assign_cons (load2_safe r.gate _ _ _ p.BC p.BK _ _ _ _ "p_g" bounds _ hb.queryGate (by simp [pointed,queryPointer,keyPointer]))
  intro bg
  apply TraceProofs.assign_cons
  · exact query_gate_safe p scale bounds _
  intro bqg
  apply TraceProofs.assign_cons (load2_safe r.key _ _ _ p.BK p.BC _ _ _ _ "p_k" bounds _ hb.key (by simp [pointed,queryPointer,keyPointer]))
  intro bk
  apply TraceProofs.assign_cons (load2_safe r.gate _ _ _ p.BK p.BC _ _ _ _ "p_gk" bounds _ hb.keyGate (by simp [pointed,queryPointer,keyPointer]))
  intro bgk
  apply TraceProofs.assign_cons
  · exact key_gate_safe p bounds _
  intro bkg
  apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
  rw [Stmt.TraceSafe]
  exact update_safe p swapped bounds _

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterStepSafety
