import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealInnerLoop
import VeriTile.Triton.Memory.TraceProofs

/-! Bounds for the complete inner loop, following the actual two-axis masks.
Only active addresses are constrained. No value-correctness fact is a safety
precondition, and the step proof preserves the register invariant between loops. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealInnerSafety
open VeriTile.Triton RealProgram
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (r : Regions) (bounds : RegionBounds) : Prop where
  key : ∀ (t : Nat) (e : Fin p.BK), keyIndex p l e.val < p.K ∧ t < p.T →
    keyAddress p l t e.val < bounds r.key
  delta : ∀ (t : Nat) (e : Fin p.BK), t < p.T ∧ keyIndex p l e.val < p.K →
    deltaAddress p l t e.val < bounds r.delta
  value : ∀ (t : Nat) (j : Fin p.BV), activeValue p l t j.val →
    valueAddress p l t j.val < bounds r.value
  output : ∀ (t : Nat) (j : Fin p.BV), activeValue p l t j.val →
    valueAddress p l t j.val < bounds r.corrected

private theorem load_safe {m n : Nat} (nm : RegName) (bp : Tile .blockPtr [m,n])
    (bounds : RegionBounds) (u : BlockState)
    (hp : u.regs .blockPtr [m,n] nm = some bp)
    (hb : ∀ i : TileIndex [m,n], (bp.data i).inBounds (TileShape.indexToList _ i) [0,1] = Bool.true →
      (bp.data i).address (TileShape.indexToList _ i) < bounds (bp.data i).region) :
    (Op.load .real (.blockPtr (.ref .blockPtr [m,n] nm) [0,1]) .none .unspecified).SafeAt bounds u := by
  rw [Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt], trivial, ?_⟩
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ps he i _
  rw [evalOp_ref, hp] at he
  cases he
  exact hb i

private theorem store_safe (p : Layout) (l : Launch) (r : Regions) (outer inner : Nat)
    (bounds : RegionBounds) (hb : Bounds p l r bounds) (u : BlockState)
    (hp : u.regs .blockPtr [p.BC,p.BV] "p_v_new" = some (valuePointer p l r.corrected outer inner)) :
    Stmt.TraceSafe bounds (.store .real [p.BC,p.BV]
      (.blockPtr (.ref .blockPtr [p.BC,p.BV] "p_v_new") [0,1]) (.ref .real [p.BC,p.BV] "b_v") .none) u := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt, Op.SafeAt], by simp [Op.SafeAt], trivial, ?_⟩
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ps he idx _
  rw [evalOp_ref, hp] at he
  cases he
  rcases idx with ⟨c,j,⟨⟩⟩
  simp only [valuePointer, TileShape.indexToList, BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets, decide_eq_true_eq]
  exact hb.output (timeBase p outer inner + c.val) j

theorem iteration_safe (p : Layout) (l : Launch) (r : Regions) (outer inner : Nat)
    (bounds : RegionBounds) (hb : Bounds p l r bounds) (u : BlockState)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar outer))
    (hic : u.regs .nat [] "i_c" = some (Tile.scalar inner)) :
    Stmt.TraceSafeList bounds (innerBody p r) u := by
  unfold innerBody chunkDeltaInnerBody
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.key u _ _ _ [p.K,p.T] [p.BK,p.BC] [p.keyDim,p.keyTime]
      (l.head * p.keyHead) (l.key * p.BK) (timeBase p outer inner)
      (mulConst_eval u "i_bh" l.head p.keyHead hih)
      (mulConst_eval u "i_k" l.key p.BK hik)
      (addMulMul_eval u "i_t" "i_c" outer p.BT inner p.BC hit hic)))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.delta _ _ _ _ [p.T,p.K] [p.BC,p.BK] [p.keyTime,p.keyDim]
      (l.head * p.keyHead) (timeBase p outer inner) (l.key * p.BK)
      (mulConst_eval _ "i_bh" l.head p.keyHead (by simp [hih]))
      (addMulMul_eval _ "i_t" "i_c" outer p.BT inner p.BC (by simp [hit]) (by simp [hic]))
      (mulConst_eval _ "i_k" l.key p.BK (by simp [hik]))))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.value _ _ _ _ [p.T,p.V] [p.BC,p.BV] [p.valueTime,p.valueDim]
      (l.head * p.valueHead) (timeBase p outer inner) (l.value * p.BV)
      (mulConst_eval _ "i_bh" l.head p.valueHead (by simp [hih]))
      (addMulMul_eval _ "i_t" "i_c" outer p.BT inner p.BC (by simp [hit]) (by simp [hic]))
      (mulConst_eval _ "i_v" l.value p.BV (by simp [hiv]))))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.corrected _ _ _ _ [p.T,p.V] [p.BC,p.BV] [p.valueTime,p.valueDim]
      (l.head * p.valueHead) (timeBase p outer inner) (l.value * p.BV)
      (mulConst_eval _ "i_bh" l.head p.valueHead (by simp [hih]))
      (addMulMul_eval _ "i_t" "i_c" outer p.BT inner p.BC (by simp [hit]) (by simp [hic]))
      (mulConst_eval _ "i_v" l.value p.BV (by simp [hiv]))))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  apply TraceProofs.assign_cons (load_safe "p_k" (keyPointer p l r.key outer inner) bounds _
    (by simp [keyPointer]) (by
      rintro ⟨e,c,⟨⟩⟩
      simp only [keyPointer, TileShape.indexToList, BlockPtr.inBounds_2d_offsets,
        BlockPtr.address_2d_offsets, decide_eq_true_eq]
      exact hb.key (timeBase p outer inner + c.val) e))
  intro bk
  apply TraceProofs.assign_cons (load_safe "p_d" (deltaPointer p l r.delta outer inner) bounds _
    (by simp [deltaPointer]) (by
      rintro ⟨c,e,⟨⟩⟩
      simp only [deltaPointer, TileShape.indexToList, BlockPtr.inBounds_2d_offsets,
        BlockPtr.address_2d_offsets, decide_eq_true_eq]
      exact hb.delta (timeBase p outer inner + c.val) e))
  intro bd
  apply TraceProofs.assign_cons (load_safe "p_v" (valuePointer p l r.value outer inner) bounds _
    (by simp [valuePointer]) (by
      rintro ⟨c,j,⟨⟩⟩
      simp only [valuePointer, TileShape.indexToList, BlockPtr.inBounds_2d_offsets,
        BlockPtr.address_2d_offsets, decide_eq_true_eq]
      exact hb.value (timeBase p outer inner + c.val) j))
  intro bv
  apply TraceProofs.assign_cons
  · simp [Op.SafeAt, Op.SafeAt.eq_def]
  intro corrected
  apply Stmt.TraceSafeList.cons_intro (store_safe p l r outer inner bounds hb _ (by simp [valuePointer]))
  intro v _
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def]

theorem loop_safe (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (outer : Nat) (frozen seed : StateValue p)
    (hu : initial.undef = fun _ _ => 0)
    (hk : r.key ≠ r.corrected) (hd : r.delta ≠ r.corrected) (hv : r.value ≠ r.corrected)
    (hinj : RealInnerLoop.NoCollision p l) (bounds : RegionBounds) (hb : Bounds p l r bounds)
    (u : BlockState) (h : RealInnerLoop.Holds initial p l r a outer frozen seed 0 u) :
    Stmt.TraceSafe bounds (RealInnerLoop.statement p r) u := by
  rw [RealInnerLoop.statement, Stmt.TraceSafe]
  have hs : (cdfStopOp p.BT p.BC).SafeAt bounds u := by simp [cdfStopOp, Op.SafeAt]
  simp only [Op.SafeAt, hs, true_and, evalOp_constNat, RealInnerLoop.stop_eval, Tile.scalar]
  apply TraceProofs.range_safe (P := RealInnerLoop.Holds initial p l r a outer frozen seed)
  · intro c v _ hc
    apply iteration_safe p l r outer c bounds hb
    · simp [hc.key]
    · simp [hc.value]
    · simp [hc.head]
    · simp [hc.time]
    · simp
  · intro c v w hc hv' he
    obtain ⟨w',he',hw⟩ := RealInnerLoop.iteration initial p l r a outer frozen seed hu
      hk hd hv hinj c v hc hv'
    rw [he] at he'
    cases he'
    exact hw
  · exact h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealInnerSafety
