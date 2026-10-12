import bench.tritonbench_g.chunk_gated_attention.Proofs.RealCumExecution
import VeriTile.Triton.Memory.TraceProofs

/-! Bounds apply only to the active input and output cells of this chunk.
Unspecified boundary lanes are not memory accesses. Aliased input/output
regions are allowed because the only store follows the complete input load. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealCumSafety
open VeriTile.Triton RealCumProgram RealCumExecution
open ChunkCumsumVector.RealStep (pointer)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (c : Nat) (s o : RegionName) (bounds : RegionBounds) : Prop where
  input : ∀ i, tileActive p l c i → offset p l c i < bounds s
  output : ∀ i, tileActive p l c i → offset p l c i < bounds o

private theorem pointer_bounds (p : Layout) (l : Launch) (c : Nat) (r : RegionName)
    (bounds : RegionBounds) (hb : ∀ i, tileActive p l c i → offset p l c i < bounds r)
    (i : TileIndex [p.BT,p.BS])
    (hi : ((pointer p l r c).data i).inBounds (TileShape.indexToList _ i) [0,1] = Bool.true) :
    ((pointer p l r c).data i).address (TileShape.indexToList _ i) < bounds r := by
  obtain ⟨e,j,⟨⟩⟩ := i
  simp only [pointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,decide_eq_true_eq] at hi
  simpa only [pointer,TileShape.indexToList,BlockPtr.address_2d_offsets,offset,
    ChunkCumsumVector.RealStep.address] using hb (e,j,PUnit.unit) hi

private theorem load_safe (p : Layout) (l : Launch) (c : Nat) (s : RegionName)
    (bounds : RegionBounds) (u : BlockState)
    (hp : u.regs .blockPtr [p.BT,p.BS] "p_s" = some (pointer p l s c))
    (hb : ∀ i, tileActive p l c i → offset p l c i < bounds s) :
    (Op.load .real (.blockPtr (.ref .blockPtr [p.BT,p.BS] "p_s") [0,1]) .none .unspecified).SafeAt bounds u := by
  rw [Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt],trivial,?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he i _
  rw [evalOp_ref,hp] at he
  cases he
  exact pointer_bounds p l c s bounds hb i

private theorem store_safe (p : Layout) (l : Launch) (c : Nat) (o : RegionName)
    (bounds : RegionBounds) (u : BlockState)
    (hp : u.regs .blockPtr [p.BT,p.BS] "p_o" = some (pointer p l o c))
    (hb : ∀ i, tileActive p l c i → offset p l c i < bounds o) :
    Stmt.TraceSafe bounds (.store .real [p.BT,p.BS]
      (.blockPtr (.ref .blockPtr [p.BT,p.BS] "p_o") [0,1]) (.ref .real [p.BT,p.BS] "b_o") .none) u := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt,Op.SafeAt],by simp [Op.SafeAt],trivial,?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he i _
  rw [evalOp_ref,hp] at he
  cases he
  exact pointer_bounds p l c o bounds hb i

theorem compute_safe (p : Layout) (l : Launch) (c : Nat) (s o : RegionName) (swapped : Bool)
    (bounds : RegionBounds) (hb : Bounds p l c s o bounds) (u : BlockState)
    (his : u.regs .nat [] "i_s" = some (Tile.scalar l.feature))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    Stmt.TraceSafeList bounds (computeCode p s o swapped) u := by
  unfold computeCode
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (pointer_eval p l s c swapped u his hih hit))]
  refine ⟨?_,?_⟩
  · cases swapped <;> simp [Stmt.TraceSafe,pointerOp,baseOp,Op.SafeAt]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (pointer_eval p l o c swapped _
    (by simp [his]) (by simp [hih]) (by simp [hit])))]
  refine ⟨?_,?_⟩
  · cases swapped <;> simp [Stmt.TraceSafe,pointerOp,baseOp,Op.SafeAt]
  apply TraceProofs.assign_cons (load_safe p l c s bounds _ (by simp) hb.input)
  intro bs
  apply TraceProofs.assign_cons
  · simp [Op.SafeAt,Op.SafeAt.eq_def]
  intro bo
  apply Stmt.TraceSafeList.cons_intro (store_safe p l c o bounds _ (by simp) hb.output)
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

theorem source_safe (p : Layout) (s o : RegionName) (swapped : Bool) (dtype out_dtype : FloatDType)
    (u : BlockState) (bounds : RegionBounds) (hb : Bounds p (launch u) (chunk u) s o bounds) :
    (RealCumSource.source swapped dtype out_dtype s o p.headStride p.timeStride p.channelStride
      p.T p.S p.BT p.BS).eraseDType.toAlgKernel.TraceSafe bounds u := by
  rw [Kernel.TraceSafe,source_body,body]
  apply TraceProofs.append_safe
  · apply TraceProofs.all_safe
    simp [initCode,Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]
  · intro v he
    rw [initialization] at he
    cases he
    exact compute_safe p (launch u) (chunk u) s o swapped bounds hb (initial p u)
      (by simp [initial,launch]) (by simp [initial,launch]) (by simp [initial,chunk])

theorem source_flatten_ok (p : Layout) (s o : RegionName) (swapped : Bool) (dtype out_dtype : FloatDType) :
    (RealCumSource.source swapped dtype out_dtype s o p.headStride p.timeStride p.channelStride
      p.T p.S p.BT p.BS).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,source_body]
  cases swapped <;> simp [body,initCode,computeCode,pointerOp,baseOp,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealCumSafety
