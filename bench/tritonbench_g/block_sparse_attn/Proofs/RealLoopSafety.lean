import bench.tritonbench_g.block_sparse_attn.Proofs.RealVisitSafety
import bench.tritonbench_g.block_sparse_attn.Proofs.RealStoreSafety

/-! Inductively safe accesses of every signed CSR visit, composed with the
actual prologue and final stores. Bounds name individual input/output cells;
none assumes execution or safety of an entire body. -/

namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealLoopSafety
open VeriTile.Triton RealNormalizationValues
noncomputable section

/-- Concrete addresses visited by the source; these are later derived from
an allocation layout. No execution/safety or output property is a field. -/
structure Bounds (bounds : RegionBounds) (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) : Prop where
  column : ∀ visit, visit < RealLoop.count s p →
    RealColumn.address (RealSetup.head s p % p.layouts) p.columnStride
      (RealLoop.first s p + (visit : Int)) < bounds (Region.cast p.cols)
  key : ∀ visit, visit < RealLoop.count s p → RealKey.Bounds bounds (RealSetup.keyPointers s BN BD p)
    (RealVisit.start s (RealLoop.operands s BM BN BD p visit) p) p.kRow p.context p.evenN Bool.false
  secondKey : 2 ≤ p.blocks → ∀ visit, visit < RealLoop.count s p →
    RealKey.Bounds bounds (RealSetup.keyPointers s BN BD p)
      (RealVisit.start s (RealLoop.operands s BM BN BD p visit) p) p.kRow p.context p.evenN Bool.true
  value : ∀ visit, visit < RealLoop.count s p → RealValueSafety.Bounds bounds (RealSetup.valuePointers s BN BD p)
    (RealVisit.start s (RealLoop.operands s BM BN BD p visit) p) p.vRow p.context p.evenN Bool.false
  secondValue : 2 ≤ p.blocks → ∀ visit, visit < RealLoop.count s p →
    RealValueSafety.Bounds bounds (RealSetup.valuePointers s BN BD p)
      (RealVisit.start s (RealLoop.operands s BM BN BD p visit) p) p.vRow p.context p.evenN Bool.true

theorem body_safe (bounds : RegionBounds) (swapped : Bool) (s t : BlockState)
    (BM BN BD : Nat) (p : RealProgram.Parameters) (visit : Nat) (hv : visit < RealLoop.count s p)
    (hN : 0 < BN) (hb : Bounds bounds s BM BN BD p) (hf : RealLoop.Fixed s BM BN BD p t)
    (hc : RealLoop.CarryAt (RealLoop.carry s BM BN BD p visit) p.blocks t) :
    Stmt.TraceSafeList bounds (RealProgram.realPart (RealProgram.parts swapped BM BN BD p).body)
      (t.setReg "col_idx_idx" .int [] (Tile.scalar (RealLoop.first s p + (visit : Int)))) := by
  let u := t.setReg "col_idx_idx" .int [] (Tile.scalar (RealLoop.first s p + (visit : Int)))
  have hfu := RealLoop.fixed_set_index s t BM BN BD p (RealLoop.first s p + (visit : Int)) hf
  have hcu := RealLoop.carry_set_index (RealLoop.carry s BM BN BD p visit) p.blocks t
    (RealLoop.first s p + (visit : Int)) hc
  have hm := congrArg BlockState.mem hfu.frame
  change u.mem = s.mem at hm
  have hstart := RealLoop.start_same_memory s u BM BN BD p visit hm
  apply RealVisitSafety.source_safe bounds swapped hN u (RealLoop.operands s BM BN BD p visit) p
    (RealLoop.carry s BM BN BD p visit)
    hfu.layout (by simp [u, RealLoop.operands]) hfu.keys hfu.values hfu.rows hfu.lanes
    hfu.query hfu.secondQuery hcu.maximum hcu.normalizer hcu.accumulator hcu.secondAccumulator
  · exact hb.column visit hv
  · rw [hstart]
    exact hb.key visit hv
  · intro htwo
    rw [hstart]
    exact hb.secondKey htwo visit hv
  · rw [hstart]
    exact hb.value visit hv
  · intro htwo
    rw [hstart]
    exact hb.secondValue htwo visit hv

theorem loop_safe_from (bounds : RegionBounds) (swapped : Bool) (s t : BlockState)
    (BM BN BD : Nat) (p : RealProgram.Parameters) (visit : Nat) (hi : visit ≤ RealLoop.count s p)
    (hN : 0 < BN) (hb : Bounds bounds s BM BN BD p) (hf : RealLoop.Fixed s BM BN BD p t)
    (hc : RealLoop.CarryAt (RealLoop.carry s BM BN BD p visit) p.blocks t) :
    Stmt.forRangeIntTraceSafe bounds "col_idx_idx" (RealLoop.first s p + (visit : Int)) (RealLoop.last s p) 1
      (RealProgram.realPart (RealProgram.parts swapped BM BN BD p).body) t := by
  rw [Stmt.forRangeIntTraceSafe, if_neg (by decide)]
  by_cases hv : visit < RealLoop.count s p
  · rw [if_pos ((RealLoop.cursor_lt s p visit).mpr hv)]
    refine ⟨body_safe bounds swapped s t BM BN BD p visit hv hN hb hf hc, ?_⟩
    obtain ⟨u, he, hfu, hcu⟩ := RealLoop.body_step swapped s t BM BN BD p visit hN hf hc
    rw [he]
    simpa only [Nat.cast_add, Nat.cast_one, Int.add_assoc] using
      loop_safe_from bounds swapped s u BM BN BD p (visit+1) (by omega) hN hb hfu hcu
  · rw [if_neg (by have ht := RealLoop.cursor_lt s p visit; omega)]
    trivial
termination_by RealLoop.count s p - visit

theorem signed_loop_safe (bounds : RegionBounds) (swapped : Bool) (s : BlockState)
    (BM BN BD : Nat) (p : RealProgram.Parameters) (hN : 0 < BN) (hb : Bounds bounds s BM BN BD p) :
    Stmt.TraceSafe bounds
      (.forRangeInt "col_idx_idx" (.ref .int [] "start_l") (.ref .int [] "end_l") (.constNat 1)
        (RealProgram.realPart (RealProgram.parts swapped BM BN BD p).body)) (RealPrologue.result s BM BN BD p) := by
  have hpre := RealPrologue.outputs s BM BN BD p
  rw [Stmt.TraceSafe]
  simp only [Op.SafeAt.eq_def, evalOp_ref, hpre.first, hpre.last, evalOp_constNat,
    Tile.scalar_data, true_and]
  simpa only [Int.ofNat_zero, Int.add_zero, RealLoop.first, RealLoop.last] using
    loop_safe_from bounds swapped s (RealPrologue.result s BM BN BD p) BM BN BD p 0
      (Nat.zero_le _) hN hb (RealLoop.prologue_fixed s BM BN BD p) (RealLoop.prologue_carry s BM BN BD p)

/-- All accesses of the complete source are safe, deriving the signed-loop
safety by induction and composing the actual masked output stores. -/
theorem source_safe (bounds : RegionBounds) (swapped : Bool) (s : BlockState)
    (BM BN BD : Nat) (p : RealProgram.Parameters) (hN : 0 < BN) (hb : Bounds bounds s BM BN BD p)
    (hq : RealQuery.Bounds bounds (RealSetup.queryPointers s BM BD p)
      (RealSetup.rows s BM) p.context p.evenM Bool.false)
    (hq2 : 2 ≤ p.blocks → RealQuery.Bounds bounds (RealSetup.queryPointers s BM BD p)
      (RealSetup.rows s BM) p.context p.evenM Bool.true)
    (hrows : RealRows.address (RealSetup.head s p) p.layouts p.rowStride (s.pids 0) + 1 < bounds (Region.cast p.rows))
    (hout : RealStores.Bounds bounds s BM BD p Bool.false)
    (hout2 : 2 ≤ p.blocks → RealStores.Bounds bounds s BM BD p Bool.true) :
    Stmt.TraceSafeList bounds (RealProgram.kernel swapped BM BN BD p).toAlgKernel.body s := by
  have hpre := RealPrologue.source_safe bounds swapped s BM BN BD p hq hq2 hrows
  have hprefix : Stmt.TraceSafeList bounds
      (RealProgram.realPart (RealProgram.parts swapped BM BN BD p).before ++
        [.forRangeInt "col_idx_idx" (.ref .int [] "start_l") (.ref .int [] "end_l") (.constNat 1)
          (RealProgram.realPart (RealProgram.parts swapped BM BN BD p).body)]) s := by
    apply TraceProofs.append_safe _ s hpre
    intro t ht
    rw [RealPrologue.source_executes swapped s BM BN BD p] at ht
    cases ht
    exact Stmt.TraceSafeList.cons_intro (signed_loop_safe bounds swapped s BM BN BD p hN hb)
      (fun _ _ => Stmt.TraceSafeList.nil_intro)
  obtain ⟨t, he, hf, hc⟩ := RealLoop.prologue_loop_executes swapped s BM BN BD p hN
  rw [RealProgram.body_split]
  apply TraceProofs.append_safe _ s hprefix
  intro u hu
  rw [he] at hu
  cases hu
  exact RealStores.source_safe bounds swapped s t BM BN BD p _ hf hc hout hout2

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealLoopSafety
