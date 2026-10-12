import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealFinal
import VeriTile.Triton.Memory.TraceProofs

/-! Active address bounds for seed, history and final-state tiles. Checked-off
lanes do not access memory, regardless of their unspecified numerical values.
Both optional-state branches use the source's actual contiguous seed/final
layout, while history retains its separate row stride. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealStateSafety
open VeriTile.Triton RealProgram
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def Bounds (p : Layout) (l : Launch) (rg : RegionName) (base stride : Nat)
    (bounds : RegionBounds) : Prop :=
  ∀ (e : Fin p.BK) (j : Fin p.BV), RealStateStore.active p l e.val j.val →
    RealStateStore.address p l base stride e.val j.val < bounds rg

theorem load_safe (p : Layout) (l : Launch) (rg : RegionName) (base stride : Nat)
    (name : RegName) (bounds : RegionBounds) (u : BlockState)
    (hb : Bounds p l rg base stride bounds)
    (hp : u.regs .blockPtr [p.BK,p.BV] name = some (RealStateStore.pointer p l rg base stride)) :
    (Op.load .real (.blockPtr (.ref .blockPtr [p.BK,p.BV] name) [0,1]) .none .unspecified).SafeAt bounds u := by
  rw [Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt], trivial, ?_⟩
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ps he idx _
  rw [evalOp_ref, hp] at he
  cases he
  rcases idx with ⟨e,j,⟨⟩⟩
  simp only [RealStateStore.pointer, TileShape.indexToList,
    BlockPtr.inBounds_2d_offsets, BlockPtr.address_2d_offsets, decide_eq_true_eq]
  simpa only [
    RealStateStore.address, RealStateStore.active, keyIndex, valueIndex, Nat.mul_one]
    using hb e j

theorem store_safe (p : Layout) (l : Launch) (rg : RegionName) (base stride : Nat)
    (name : RegName) (bounds : RegionBounds) (u : BlockState)
    (hb : Bounds p l rg base stride bounds)
    (hp : u.regs .blockPtr [p.BK,p.BV] name = some (RealStateStore.pointer p l rg base stride)) :
    Stmt.TraceSafe bounds (.store .real [p.BK,p.BV]
      (.blockPtr (.ref .blockPtr [p.BK,p.BV] name) [0,1])
      (.ref .real [p.BK,p.BV] "b_h") .none) u := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt, Op.SafeAt], by simp [Op.SafeAt], trivial, ?_⟩
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ps he idx _
  rw [evalOp_ref, hp] at he
  cases he
  rcases idx with ⟨e,j,⟨⟩⟩
  simp only [RealStateStore.pointer, TileShape.indexToList,
    BlockPtr.inBounds_2d_offsets, BlockPtr.address_2d_offsets, decide_eq_true_eq]
  simpa only [
    RealStateStore.address, RealStateStore.active, keyIndex, valueIndex, Nat.mul_one]
    using hb e j

theorem init_safe (p : Layout) (r : Regions) (useSeed : Bool)
    (bounds : RegionBounds) (u : BlockState)
    (hb : useSeed = Bool.true → Bounds p (RealInit.launch u) r.initial
      (RealInit.base p (RealInit.launch u)) p.V bounds) :
    Stmt.TraceSafeList bounds (cdfPrologue r.initial p.K p.V p.BK p.BV useSeed) u := by
  unfold cdfPrologue
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 0 u))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  apply TraceProofs.assign_cons (by simp [Op.SafeAt])
  intro zero
  apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
  cases useSeed
  · rw [Stmt.TraceSafe]
    exact ⟨by simp [Op.SafeAt], by simp only [evalOp, Tile.scalar, Bool.false_eq_true, ite_false]⟩
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt, evalOp, Tile.scalar, ↓reduceIte, true_and]
    let entry := (((u.setReg "i_k" .nat [] (Tile.scalar (u.pids 0))).setReg
      "i_v" .nat [] (Tile.scalar (u.pids 1))).setReg
      "i_bh" .nat [] (Tile.scalar (u.pids 2))).setReg "b_h" .real [p.BK,p.BV] zero
    change Stmt.TraceSafeList bounds _ entry
    have hp := RealFinal.pointer_eval p (RealInit.launch u) r.initial entry
      (by simp [entry, RealInit.launch]) (by simp [entry, RealInit.launch]) (by simp [entry, RealInit.launch])
    unfold RealFinal.pointerOp at hp
    rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some hp)]
    refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
    apply TraceProofs.assign_cons (load_safe p (RealInit.launch u) r.initial
      (RealInit.base p (RealInit.launch u)) p.V "p_h0" bounds _ (hb rfl) (by simp))
    intro _
    exact Stmt.TraceSafeList.nil_intro

theorem final_safe (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool)
    (bounds : RegionBounds) (u : BlockState)
    (hb : storeFinal = Bool.true → Bounds p l r.final (RealInit.base p l) p.V bounds)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)) :
    Stmt.TraceSafeList bounds (cdfEpilogue r.final p.K p.V p.BK p.BV storeFinal) u := by
  unfold cdfEpilogue
  apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
  cases storeFinal
  · rw [Stmt.TraceSafe]
    exact ⟨by simp [Op.SafeAt], by simp only [evalOp, Tile.scalar, Bool.false_eq_true, ite_false]⟩
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt, evalOp, Tile.scalar, ↓reduceIte, true_and]
    have hp := RealFinal.pointer_eval p l r.final u hik hiv hih
    unfold RealFinal.pointerOp at hp
    rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some hp)]
    refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
    apply Stmt.TraceSafeList.cons_intro
      (store_safe p l r.final (RealInit.base p l) p.V "p_ht" bounds _ (hb rfl) (by simp))
    intro _ _
    exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealStateSafety
