import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealIntraStepSafety

/-! Whole-source trace safety. The early return avoids the complete prefix;
on the active branch the matrix loads and every streamed load/store use their
actual source bounds. These conditions do not presume the output equation. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraSafety
open VeriTile.Triton RealIntraMath RealIntraProgram RealIntraMemory RealIntraLoop
open RealIntraSetup (initial initial_grid initialization_executes)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (r : Regions) (bounds : RegionBounds) : Prop where
  query : enabled p l → RealIntraAccessSafety.MatrixBounds p l r.query bounds
  gate : enabled p l → RealIntraAccessSafety.MatrixBounds p l r.gate bounds
  column : enabled p l → ∀ j, j < columns p l → RealIntraStepSafety.Bounds p l r j bounds

theorem loop_safe (p : Layout) (r : Regions) (scale : ℝ) (s0 : BlockState) (swapped : Bool)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) (hcols : columns p (launch s0) ≤ p.BT)
    (bounds : RegionBounds)
    (hb : ∀ j, j < columns p (launch s0) → RealIntraStepSafety.Bounds p (launch s0) r j bounds)
    (s : BlockState) (hi : Holds p r scale s0 0 s) : Stmt.TraceSafe bounds (loop p r scale swapped) s := by
  rw [loop,Stmt.TraceSafe]
  have hs : (tripCount p).SafeAt bounds s := by simp [tripCount,remaining,times,Op.SafeAt]
  simp only [Op.SafeAt,hs,true_and,evalOp_constNat,tripCount_eval p (launch s0) s hi.cached.grid,Tile.scalar]
  apply TraceProofs.range_safe (P := Holds p r scale s0)
  · intro j u hn hu
    have ht := index_holds p r scale s0 j u hu
    exact RealIntraStepSafety.iteration_safe p (launch s0) r scale j swapped s0 _ bounds (hb j hn)
      ht.cached (by simp) ht.keyPtr ht.gatePtr
  · intro j u v hn hu he
    obtain ⟨v',he',hv⟩ := RealIntraLoop.iteration p r scale s0 swapped hkey hgate hcols j u hn hu
    rw [he] at he'
    cases he'
    exact hv
  · exact hi

theorem source_safe (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ) (swapped : Bool) (s : BlockState)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) (hcols : columns p (launch s) ≤ p.BT)
    (bounds : RegionBounds) (hb : Bounds p (launch s) r bounds) :
    (source p r qs ks gs scale swapped).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [Kernel.TraceSafe,source_body,body]
  apply TraceProofs.append_safe _ s
  · apply TraceProofs.all_safe
    simp [initialization,Stmt.TraceSafe,Op.SafeAt]
  · intro t he
    rw [initialization_executes] at he
    cases he
    apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
    rw [Stmt.TraceSafe]
    refine ⟨by simp [condition,timeOp,times,Op.SafeAt],?_⟩
    rw [RealIntraExecution.condition_eval p (launch s) (initial s) (initial_grid s)]
    simp only [Tile.scalar,decide_eq_true_eq]
    by_cases hen : enabled p (launch s)
    · have hrow : rowStart p (launch s) < p.T := hen
      simp only [hrow,↓reduceIte]
      apply TraceProofs.append_safe _ (initial s)
      · exact RealIntraStepSafety.setup_safe p (launch s) r (initial s) (initial_grid s) bounds (hb.query hen) (hb.gate hen)
      · intro t ht
        rw [RealIntraSetup.executes p (launch s) r (initial s) (initial_grid s)] at ht
        cases ht
        apply Stmt.TraceSafeList.cons_intro
          (loop_safe p r scale s swapped hkey hgate hcols bounds (hb.column hen) (start p r s) (start_holds p r scale s))
        intro _ _
        exact Stmt.TraceSafeList.nil_intro
    · have hrow : ¬rowStart p (launch s) < p.T := hen
      simp [hrow]

theorem source_flatten_ok (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ) (swapped : Bool) :
    (source p r qs ks gs scale swapped).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,source_body]
  cases swapped <;> simp [body,initialization,condition,setup,loop,tripCount,remaining,
    RealIntraProgram.iteration,queryPtr,keyPtr,timeOp,times,rowsOp,offsetsOp,outputOffsets,
    product,reducedOp,scaledOp,keyLoad,load2,advance,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraSafety
