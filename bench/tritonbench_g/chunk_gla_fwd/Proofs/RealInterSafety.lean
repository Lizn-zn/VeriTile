import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterExecution
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterStepSafety

/-! Full-source trace safety, including both skip branches, all K iterations,
and the final store. Bounds mention only source-active accesses. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterSafety
open VeriTile.Triton RealInterMath RealInterProgram RealInterPointers RealInterLoop
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (r : Regions) (bounds : RegionBounds) : Prop where
  input : enabled p l → ∀ kb, kb < numBlocks p → RealInterStepSafety.Bounds p l r kb bounds
  output : enabled p l → ∀ i : TileIndex [p.BC,p.BC], active p l i.1.val i.2.1.val →
    outputAddress p l i.1.val i.2.1.val < bounds r.output

theorem loop_safe (p : Layout) (r : Regions) (scale : ℝ) (s0 : BlockState) (swapped : Bool)
    (hu : ∀ region off, s0.undef region off = 0) (bounds : RegionBounds)
    (hb : ∀ kb, kb < numBlocks p → RealInterStepSafety.Bounds p (launch s0) r kb bounds)
    (u : BlockState) (hi : Holds p r scale s0 0 u) : Stmt.TraceSafe bounds (loop p r scale swapped) u := by
  rw [loop,Stmt.TraceSafe]
  have hs : (tripCount p).SafeAt bounds u := by simp [tripCount,Op.SafeAt]
  simp only [Op.SafeAt,hs,true_and,evalOp_constNat,tripCount_eval,Tile.scalar]
  apply TraceProofs.range_safe (P := Holds p r scale s0)
  · intro i s hn hs
    have ht := index_holds p r scale s0 i s hs
    exact RealInterStepSafety.iteration_safe p (launch s0) r (inputs r s0) scale i swapped _ bounds (hb i hn)
      ht.grid (by simp)
  · intro i s t hn hs he
    obtain ⟨t',he',ht⟩ := RealInterLoop.iteration p r scale s0 swapped hu i s hn hs
    rw [he] at he'
    cases he'
    exact ht
  · exact hi

theorem store_safe (p : Layout) (l : Launch) (r : RegionName) (s : BlockState) (bounds : RegionBounds)
    (hp : s.regs .blockPtr [p.BC,p.BC] "p_A" = some (outputPointer p l r))
    (hb : ∀ i : TileIndex [p.BC,p.BC], active p l i.1.val i.2.1.val → outputAddress p l i.1.val i.2.1.val < bounds r) :
    Stmt.TraceSafe bounds (.store .real [p.BC,p.BC]
      (.blockPtr (.ref .blockPtr [p.BC,p.BC] "p_A") [0,1]) (carry p) .none) s := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt,Op.SafeAt],by simp [carry,Op.SafeAt],trivial,?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he i _
  rw [evalOp_ref,hp] at he
  cases he
  simpa only [outputPointer,ChunkGatedAttention.RealHPointers.matrixPointer,TileShape.indexToList,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,decide_eq_true_eq,Nat.mul_one,
    outputAddress,active] using hb i

theorem final_safe (p : Layout) (l : Launch) (r : Regions) (s : BlockState) (bounds : RegionBounds)
    (hg : Grid p l s)
    (hb : ∀ i : TileIndex [p.BC,p.BC], active p l i.1.val i.2.1.val → outputAddress p l i.1.val i.2.1.val < bounds r.output) :
    Stmt.TraceSafeList bounds (finalization p r) s := by
  unfold finalization
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (output_eval p l r.output s hg))]
  refine ⟨by simp [Stmt.TraceSafe,outputPtr,timeOp,times,Op.SafeAt],?_⟩
  apply Stmt.TraceSafeList.cons_intro (store_safe p l r.output _ bounds (by simp) hb)
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

theorem source_safe (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) (swapped : Bool) (s : BlockState)
    (hu : ∀ region off, s.undef region off = 0) (bounds : RegionBounds) (hb : Bounds p (launch s) r bounds) :
    (source p r qs ks gs out scale swapped).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [Kernel.TraceSafe,source_body,body]
  apply TraceProofs.append_safe _ s
  · apply TraceProofs.all_safe
    simp [initialization,Stmt.TraceSafe,Op.SafeAt]
  · intro t he
    rw [initialization_executes] at he
    cases he
    apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
    rw [Stmt.TraceSafe]
    refine ⟨by simp [rowCondition,timeOp,times,Op.SafeAt],?_⟩
    rw [RealInterExecution.row_condition p (launch s) (initial p s) (initial_grid p s)]
    simp only [Tile.scalar,decide_eq_true_eq]
    by_cases hrow : rowStart p (launch s) < p.T
    · simp only [hrow,↓reduceIte]
      apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
      rw [Stmt.TraceSafe]
      refine ⟨by simp [triangleCondition,Op.SafeAt],?_⟩
      rw [RealInterExecution.triangle_condition p (launch s) (initial p s) (initial_grid p s)]
      simp only [Tile.scalar,decide_eq_true_eq]
      by_cases htri : columnBlock p (launch s) < rowBlock p (launch s)
      · simp only [htri,↓reduceIte]
        rw [List.append_assoc]
        apply TraceProofs.append_safe _ (initial p s)
        · apply TraceProofs.all_safe
          simp [setup,Stmt.TraceSafe,Op.SafeAt]
        · intro t ht
          rw [setup_executes] at ht
          cases ht
          obtain ⟨u,he,hi⟩ := RealInterLoop.executes p r scale s swapped hu (start p s) (start_holds p r scale s)
          change Stmt.TraceSafeList bounds (loop p r scale swapped :: finalization p r) (start p s)
          rw [TraceProofs.cons_of_run_iff he]
          exact ⟨loop_safe p r scale s swapped hu bounds (hb.input ⟨hrow,htri⟩) (start p s) (start_holds p r scale s),
            final_safe p (launch s) r u bounds hi.grid (hb.output ⟨hrow,htri⟩)⟩
      · simp [htri]
    · simp [hrow]

theorem source_flatten_ok (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    (source p r qs ks gs out scale swapped).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,source_body]
  cases swapped <;> simp [body,initialization,rowCondition,triangleCondition,setup,loop,tripCount,
    RealInterProgram.iteration,finalization,queryPtr,keyPtr,referencePtr,outputPtr,timeOp,times,
    queryGate,keyGate,referenceLoad,load2,carry,dot,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterSafety
