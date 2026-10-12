import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealFrame
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealLoopSafety

/-! Bounds for post-loop q/k reads and the three gradient stores. Numerical
registers remain abstract; address registers are established by the actual
source. Only boundary-checked active lanes require allocation bounds. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealPostSafety
open VeriTile.Triton RealMath RealFrame
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (q k dq dk dg : RegionName)
    (bounds : RegionBounds) : Prop where
  query : ∀ i, qkActive p l i → qkAddress p l i < bounds q
  key : ∀ i, qkActive p l i → qkAddress p l i < bounds k
  dq : ∀ i, qkActive p l i → qkAddress p l i < bounds dq
  dk : ∀ i, qkActive p l i → qkAddress p l i < bounds dk
  dg : ∀ i, dgActive p l i → dgAddress p l i < bounds dg

private theorem qk_bounds (p : Layout) (l : Launch) (r : RegionName) (bounds : RegionBounds)
    (hb : ∀ i, qkActive p l i → qkAddress p l i < bounds r) :
    ∀ i, ((qkPointer p l r).data i).inBounds (TileShape.indexToList [p.BT,p.BK] i) [0,1] = Bool.true →
      ((qkPointer p l r).data i).address (TileShape.indexToList [p.BT,p.BK] i) < bounds r := by
  rintro ⟨i,j,⟨⟩⟩ hi
  have ha : qkActive p l (i,j,PUnit.unit) := by
    simpa only [qkPointer, TileShape.indexToList, BlockPtr.inBounds_2d_offsets,
      decide_eq_true_eq, qkActive] using hi
  simpa only [qkPointer, TileShape.indexToList, BlockPtr.address_2d_offsets,
    qkAddress, Nat.mul_one, Nat.add_assoc] using hb (i,j,PUnit.unit) ha

private theorem block_store_safe {shape : TileShape} (bounds : RegionBounds) (s : BlockState)
    (name value : RegName) (ptrs : Tile .blockPtr shape) (bc : List Nat)
    (hp : s.regs .blockPtr shape name = some ptrs)
    (hb : ∀ i, (ptrs.data i).inBounds (TileShape.indexToList shape i) bc = Bool.true →
      (ptrs.data i).address (TileShape.indexToList shape i) < bounds (ptrs.data i).region) :
    Stmt.TraceSafe bounds
      (.store .real shape (.blockPtr (.ref .blockPtr shape name) bc) (.ref .real shape value) .none) s := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt, Op.SafeAt], by simp [Op.SafeAt], trivial, ?_⟩
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ps he i _
  rw [evalOp_ref, hp] at he
  cases he
  exact hb i

theorem prefix_safe (s : BlockState) (p : Layout) (l : Launch)
    (q k dq dk dg : RegionName) (bounds : RegionBounds)
    (hb : Bounds p l q k dq dk dg bounds) (hg : Grid s l) :
    Stmt.TraceSafeList bounds
      ((cbdPostLoop q k dq dk dg p.s_k_h p.s_k_t p.scale p.T p.K p.BT p.BK).take 14) s := by
  simp only [cbdPostLoop, List.take]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (qk_pointer s p l q hg))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  have g1 := grid_set s l .blockPtr [p.BT,p.BK] "p_q" (qkPointer p l q) hg (by decide)
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (qk_pointer _ p l k g1))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  apply TraceProofs.assign_cons
    (RealLoopSafety.block_load_safe bounds _ "p_k" (qkPointer p l k) [0,1]
      (by simp) (qk_bounds p l k bounds hb.key))
  intro kvals
  apply TraceProofs.assign_cons
    (RealLoopSafety.block_load_safe bounds _ "p_q" (qkPointer p l q) [0,1]
      (by simp) (qk_bounds p l q bounds hb.query))
  intro qvals
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe, Op.SafeAt]
  all_goals simp [Op.SafeAt.eq_def]

theorem suffix_safe (s : BlockState) (p : Layout) (l : Launch)
    (q k dq dk dg : RegionName) (bounds : RegionBounds)
    (hb : Bounds p l q k dq dk dg bounds) (hg : Grid s l) :
    Stmt.TraceSafeList bounds
      ((cbdPostLoop q k dq dk dg p.s_k_h p.s_k_t p.scale p.T p.K p.BT p.BK).drop 14) s := by
  simp only [cbdPostLoop, List.drop]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (qk_pointer s p l dq hg))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  have g1 := grid_set s l .blockPtr [p.BT,p.BK] "p_dq" (qkPointer p l dq) hg (by decide)
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (qk_pointer _ p l dk g1))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  have g2 := grid_set _ l .blockPtr [p.BT,p.BK] "p_dk" (qkPointer p l dk) g1 (by decide)
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (dg_pointer _ p l dg g2))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  apply Stmt.TraceSafeList.cons_intro
    (block_store_safe bounds _ "p_dq" "b_dq" (qkPointer p l dq) [0,1] (by simp)
      (qk_bounds p l dq bounds hb.dq))
  intro t ht
  have htr := block_store_regs ht
  apply Stmt.TraceSafeList.cons_intro
    (block_store_safe bounds _ "p_dk" "b_dk" (qkPointer p l dk) [0,1] (by rw [htr]; simp)
      (qk_bounds p l dk bounds hb.dk))
  intro u hu
  have hur := block_store_regs hu
  apply Stmt.TraceSafeList.cons_intro
  · apply block_store_safe bounds _ "p_dg" "b_dg" (dgPointer p l dg) [0] (by rw [hur,htr]; simp)
    rintro ⟨i,⟨⟩⟩ hi
    have ha : dgActive p l (i,PUnit.unit) := by
      simpa only [dgPointer, TileShape.indexToList, BlockPtr.inBounds_1d,
        decide_eq_true_eq, dgActive] using hi
    simpa only [dgPointer, TileShape.indexToList, BlockPtr.address_1d,
      dgAddress, Nat.mul_one, Nat.add_assoc] using hb.dg (i,PUnit.unit) ha
  · intro _ _
    exact Stmt.TraceSafeList.nil_intro

theorem safe (s : BlockState) (p : Layout) (l : Launch)
    (q k dq dk dg : RegionName) (bounds : RegionBounds)
    (hb : Bounds p l q k dq dk dg bounds) (hg : Grid s l) :
    Stmt.TraceSafeList bounds
      (cbdPostLoop q k dq dk dg p.s_k_h p.s_k_t p.scale p.T p.K p.BT p.BK) s := by
  let code := cbdPostLoop q k dq dk dg p.s_k_h p.s_k_t p.scale p.T p.K p.BT p.BK
  change Stmt.TraceSafeList bounds code s
  rw [← List.take_append_drop 14 code]
  apply TraceProofs.append_safe _ s (prefix_safe s p l q k dq dk dg bounds hb hg)
  intro t ht
  apply suffix_safe t p l q k dq dk dg bounds hb
  constructor
  · exact (assignsAway_regs _ _ _ .nat [] "i_k"
      (by simp [code, cbdPostLoop, assignsAway]) ht).trans hg.key
  · exact (assignsAway_regs _ _ _ .nat [] "i_t"
      (by simp [code, cbdPostLoop, assignsAway]) ht).trans hg.time
  · exact (assignsAway_regs _ _ _ .nat [] "i_bh"
      (by simp [code, cbdPostLoop, assignsAway]) ht).trans hg.head
  · exact (assignsAway_regs _ _ _ .nat [] "n_bh"
      (by simp [code, cbdPostLoop, assignsAway]) ht).trans hg.heads

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealPostSafety
