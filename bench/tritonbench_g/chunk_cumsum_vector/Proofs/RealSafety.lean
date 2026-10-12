import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealInit
import VeriTile.Triton.Memory.TraceProofs

/-! Address safety for all source iterations and both tail masks. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealSafety
open VeriTile.Triton RealProgram RealStep RealLoop
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (s o : RegionName) (bounds : RegionBounds) : Prop where
  input : ∀ (t : Nat) (j : Fin p.BS), RealStep.active p l t j.val → address p l t j.val < bounds s
  output : ∀ (t : Nat) (j : Fin p.BS), RealStep.active p l t j.val → address p l t j.val < bounds o

private theorem pointer_bounds (p : Layout) (l : Launch) (r : RegionName) (c : Nat)
    (bounds : RegionBounds)
    (hb : ∀ (t : Nat) (j : Fin p.BS), RealStep.active p l t j.val → address p l t j.val < bounds r)
    (idx : TileIndex [p.BT, p.BS])
    (hi : ((pointer p l r c).data idx).inBounds (TileShape.indexToList _ idx) [0,1] = Bool.true) :
    ((pointer p l r c).data idx).address (TileShape.indexToList _ idx) < bounds r := by
  obtain ⟨i,j,⟨⟩⟩ := idx
  simp only [pointer, TileShape.indexToList, BlockPtr.inBounds_2d_offsets, decide_eq_true_eq] at hi
  simpa only [pointer, TileShape.indexToList, BlockPtr.address_2d_offsets, address] using
    hb (c * p.BT + i.val) j hi

private theorem load_safe (p : Layout) (l : Launch) (r : RegionName) (c : Nat)
    (bounds : RegionBounds) (u : BlockState)
    (hp : u.regs .blockPtr [p.BT,p.BS] "p_s" = some (pointer p l r c))
    (hb : ∀ (t : Nat) (j : Fin p.BS), RealStep.active p l t j.val → address p l t j.val < bounds r) :
    (Op.load .real (.blockPtr (.ref .blockPtr [p.BT,p.BS] "p_s") [0,1]) .none .unspecified).SafeAt bounds u := by
  rw [Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt], trivial, ?_⟩
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ps he i _
  rw [evalOp_ref, hp] at he
  cases he
  exact pointer_bounds p l r c bounds hb i

private theorem store_safe (p : Layout) (l : Launch) (o : RegionName) (c : Nat)
    (bounds : RegionBounds) (u : BlockState)
    (hp : u.regs .blockPtr [p.BT,p.BS] "p_z" = some (pointer p l o c))
    (hb : ∀ (t : Nat) (j : Fin p.BS), RealStep.active p l t j.val → address p l t j.val < bounds o) :
    Stmt.TraceSafe bounds (.store .real [p.BT,p.BS]
      (.blockPtr (.ref .blockPtr [p.BT,p.BS] "p_z") [0,1]) (.ref .real [p.BT,p.BS] "b_c") .none) u := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt, Op.SafeAt], by simp [Op.SafeAt], trivial, ?_⟩
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ps he i _
  rw [evalOp_ref, hp] at he
  cases he
  exact pointer_bounds p l o c bounds hb i

theorem iteration_safe (p : Layout) (l : Launch) (s o : RegionName)
    (bounds : RegionBounds) (hb : Bounds p l s o bounds) (c : Nat) (u : BlockState)
    (his : u.regs .nat [] "i_s" = some (Tile.scalar l.feature))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    Stmt.TraceSafeList bounds (loopBody p s o Bool.false) u := by
  unfold loopBody
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (pointer_eval p l s c u his hih hit))]
  refine ⟨by simp [Stmt.TraceSafe, pointerOp, Op.SafeAt], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (pointer_eval p l o c _
    (by simp [his]) (by simp [hih]) (by simp [hit])))]
  refine ⟨by simp [Stmt.TraceSafe, pointerOp, Op.SafeAt], ?_⟩
  apply TraceProofs.assign_cons (load_safe p l s c bounds _ (by simp) hb.input)
  intro bs
  apply TraceProofs.assign_cons
  · simp [Op.SafeAt, Op.SafeAt.eq_def]
  intro bc
  apply Stmt.TraceSafeList.cons_intro (store_safe p l o c bounds _ (by simp) hb.output)
  intro v _
  apply Stmt.TraceSafeList.cons_intro
  · rw [Stmt.TraceSafe]
    refine ⟨by simp [Op.SafeAt], ?_⟩
    split
    · split
      · apply TraceProofs.all_safe
        simp [Stmt.TraceSafe, Op.SafeAt, carryOp]
        all_goals simp [Op.SafeAt.eq_def]
      · trivial
    · trivial
  · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem loop_safe (initial : BlockState) (p : Layout) (l : Launch) (s o : RegionName)
    (a : Nat → ℝ) (hSO : s ≠ o) (hBT : 0 < p.BT) (hinj : NoCollision p l)
    (bounds : RegionBounds) (hb : Bounds p l s o bounds) (u : BlockState)
    (h : Holds initial p l s o a 0 u) : Stmt.TraceSafe bounds (statement p s o) u := by
  rw [statement, Stmt.TraceSafe]
  have hs : (stopOp p).SafeAt bounds u := by simp [stopOp, Op.SafeAt]
  simp only [Op.SafeAt, hs, true_and, evalOp_constNat, RealLoop.stop_eval, Tile.scalar]
  apply TraceProofs.range_safe (P := Holds initial p l s o a)
  · intro c v _ hv
    apply iteration_safe p l s o bounds hb c
    · simp [hv.feature]
    · simp [hv.head]
    · simp
  · intro c v w hc hv he
    obtain ⟨w',he',hw⟩ := RealLoop.iteration initial p l s o a hSO hBT hinj c v hc hv
    rw [he] at he'
    cases he'
    exact hw
  · exact h

theorem source_safe (p : Layout) (s o : RegionName) (dtype out_dtype : FloatDType)
    (u : BlockState) (hSO : s ≠ o) (hBT : 0 < p.BT) (hinj : NoCollision p (RealInit.launch u))
    (bounds : RegionBounds) (hb : Bounds p (RealInit.launch u) s o bounds) :
    (RealSource.source Bool.false dtype out_dtype s o p.headStride p.timeStride p.channelStride
      p.T p.S p.BT p.BS).eraseDType.toAlgKernel.TraceSafe bounds u := by
  rw [Kernel.TraceSafe, source_body, body]
  apply TraceProofs.append_safe
  · apply TraceProofs.all_safe
    simp [initCode, Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def]
  · intro v he
    rw [RealInit.executes] at he
    cases he
    apply Stmt.TraceSafeList.cons_intro
    · exact loop_safe u p (RealInit.launch u) s o (u.readMem s) hSO hBT hinj bounds hb
        (RealInit.initial p u) (RealInit.invariant p u s o (u.readMem s) (fun _ => rfl))
    · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem source_flatten_ok (p : Layout) (s o : RegionName) (dtype out_dtype : FloatDType) :
    (RealSource.source Bool.false dtype out_dtype s o p.headStride p.timeStride p.channelStride
      p.T p.S p.BT p.BS).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk, source_body]
  simp [body, initCode, loopBody, carryOp, pointerOp, stopOp,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealSafety
