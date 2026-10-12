import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealMergeExecution
import VeriTile.Triton.Memory.TraceProofs

/-! Bounds for each actual NK slice and the final diagonal-subblock store.
The early-return branch requires no allocation. Runtime head count and partial
rows/columns retain their source meanings; arbitrary padding is permitted. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeSafety
open VeriTile.Triton RealMergeProgram RealMergeStep RealMergeLoop
noncomputable section
set_option maxRecDepth 14000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (A A2 : RegionName) (bounds : RegionBounds) : Prop where
  input : enabled p l → ∀ kb, kb < p.NK → ∀ i : TileIndex [p.BC,p.BC], rowStart p l+i.1.val < p.T →
    inputAddress p l kb i < bounds A
  output : enabled p l → ∀ i, active p l i → outputAddress p l i < bounds A2

theorem load_safe (p : Layout) (l : Launch) (A : RegionName) (kb : Nat) (s : BlockState) (bounds : RegionBounds)
    (hp : s.regs .blockPtr [p.BC,p.BC] "p_A" = some (pointer p l A kb))
    (hb : ∀ i : TileIndex [p.BC,p.BC], rowStart p l+i.1.val < p.T → inputAddress p l kb i < bounds A) :
    (loaded p).SafeAt bounds s := by
  rw [loaded,Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt],trivial,?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he i _
  rw [evalOp_ref,hp] at he
  cases he
  rcases i with ⟨i,j,⟨⟩⟩
  simpa only [pointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets,decide_eq_true_eq,Nat.zero_add,j.isLt,and_true,
    Nat.mul_one,inputAddress] using hb (i,j,PUnit.unit)

theorem iteration_safe (p : Layout) (l : Launch) (A : RegionName) (kb : Nat) (swapped : Bool)
    (s : BlockState) (bounds : RegionBounds) (hg : Grid l s)
    (hk : s.regs .nat [] "i_k" = some (Tile.scalar kb))
    (hb : ∀ i : TileIndex [p.BC,p.BC], rowStart p l+i.1.val < p.T → inputAddress p l kb i < bounds A) :
    Stmt.TraceSafeList bounds (RealMergeProgram.iteration p A swapped) s := by
  unfold RealMergeProgram.iteration
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (pointer_eval p l A kb s hg hk))]
  refine ⟨by simp [Stmt.TraceSafe,inputPointer,inputBase,rowOp,mulRef,Op.SafeAt],?_⟩
  apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
  cases swapped <;> simp only [Bool.false_eq_true,↓reduceIte,Stmt.TraceSafe]
  all_goals rw [Op.SafeAt.eq_def]
  · exact ⟨by simp [carry,Op.SafeAt],load_safe p l A kb _ bounds (by simp) hb⟩
  · exact ⟨load_safe p l A kb _ bounds (by simp) hb,by simp [carry,Op.SafeAt]⟩

theorem loop_safe (p : Layout) (s0 : BlockState) (A : RegionName) (swapped : Bool)
    (bounds : RegionBounds)
    (hb : ∀ kb, kb < p.NK → ∀ i : TileIndex [p.BC,p.BC], rowStart p (launch s0)+i.1.val < p.T →
      inputAddress p (launch s0) kb i < bounds A)
    (u : BlockState) (hi : Holds p s0 A 0 u) : Stmt.TraceSafe bounds (loop p A swapped) u := by
  rw [loop,Stmt.TraceSafe]
  simp only [Op.SafeAt,evalOp_constNat,Tile.scalar,true_and]
  apply TraceProofs.range_safe (P := Holds p s0 A)
  · intro i s hn hs
    exact iteration_safe p (launch s0) A i swapped _ bounds (index_holds p s0 A i s hs).grid (by simp) (hb i hn)
  · intro i s t hn hs he
    obtain ⟨t',he',ht⟩ := RealMergeLoop.iteration p s0 A swapped i s hn hs
    rw [he] at he'
    cases he'
    exact ht
  · exact hi

theorem store_safe (p : Layout) (l : Launch) (A2 : RegionName) (s : BlockState) (bounds : RegionBounds)
    (hp : s.regs .blockPtr [p.BC,p.BC] "p_A2" = some (RealMergeFinal.pointer p l A2))
    (hb : ∀ i, active p l i → outputAddress p l i < bounds A2) :
    Stmt.TraceSafe bounds (.store .real [p.BC,p.BC]
      (.blockPtr (.ref .blockPtr [p.BC,p.BC] "p_A2") [0,1]) (carry p) .none) s := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt,Op.SafeAt],by simp [carry,Op.SafeAt],trivial,?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he i _
  rw [evalOp_ref,hp] at he
  cases he
  simpa only [RealMergeFinal.pointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets,decide_eq_true_eq,Nat.mul_one,outputAddress,active] using hb i

theorem final_safe (p : Layout) (l : Launch) (A2 : RegionName) (s : BlockState) (bounds : RegionBounds)
    (hg : Grid l s) (hb : ∀ i, active p l i → outputAddress p l i < bounds A2) :
    Stmt.TraceSafeList bounds (finalization p A2) s := by
  unfold finalization
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (RealMergeFinal.pointer_eval p l A2 s hg))]
  refine ⟨by simp [Stmt.TraceSafe,outputPointer,outputBase,rowOp,mulRef,Op.SafeAt],?_⟩
  apply Stmt.TraceSafeList.cons_intro (store_safe p l A2 _ bounds (by simp) hb)
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

theorem source_safe (p : Layout) (A A2 : RegionName) (out : FloatDType) (swapped : Bool)
    (s : BlockState) (bounds : RegionBounds) (hb : Bounds p (launch s) A A2 bounds) :
    (source p A A2 out swapped).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [Kernel.TraceSafe,source_body,body]
  apply TraceProofs.append_safe _ s
  · apply TraceProofs.all_safe
    simp [initialization,Stmt.TraceSafe,Op.SafeAt]
  · intro t he
    rw [initialization_executes] at he
    cases he
    apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
    rw [Stmt.TraceSafe]
    refine ⟨by simp [condition,rowOp,mulRef,Op.SafeAt],?_⟩
    rw [RealMergeExecution.condition_eval p (launch s) (initial s) (by simp [initial,launch]) (by simp [initial,launch])]
    simp only [Tile.scalar,decide_eq_true_eq]
    by_cases hen : enabled p (launch s)
    · simp only [hen,↓reduceIte]
      rw [List.append_assoc]
      apply TraceProofs.append_safe _ (initial s)
      · apply TraceProofs.all_safe
        simp [setup,Stmt.TraceSafe,Op.SafeAt]
      · intro t ht
        rw [setup_executes] at ht
        cases ht
        obtain ⟨u,hu,hi⟩ := RealMergeLoop.executes p s A swapped (start p s) (start_holds p s A)
        change Stmt.TraceSafeList bounds (loop p A swapped :: finalization p A2) (start p s)
        rw [TraceProofs.cons_of_run_iff hu]
        exact ⟨loop_safe p s A swapped bounds (hb.input hen) (start p s) (start_holds p s A),
          final_safe p (launch s) A2 u bounds hi.grid (hb.output hen)⟩
    · simp [hen]

theorem source_flatten_ok (p : Layout) (A A2 : RegionName) (out : FloatDType) (swapped : Bool) :
    (source p A A2 out swapped).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,source_body]
  cases swapped <;> simp [body,initialization,condition,setup,loop,RealMergeProgram.iteration,
    finalization,inputPointer,inputBase,outputPointer,outputBase,rowOp,mulRef,carry,loaded,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeSafety
