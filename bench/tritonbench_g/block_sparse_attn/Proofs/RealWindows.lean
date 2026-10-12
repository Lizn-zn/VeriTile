import bench.tritonbench_g.block_sparse_attn.Proofs.RealFormula
import bench.tritonbench_g.block_sparse_attn.Proofs.RealLoopSafety

/-! Allocation windows expressed using initial arrays and program ids.
Checked lanes need no padding; unchecked Q/K/V lanes need allocated cells
even when they lie beyond the mathematical context. No execution or
intermediate-register premise occurs in these windows. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealWindows
open VeriTile.Triton RealAddressFormula
noncomputable section

def query (s : BlockState) (BM : Nat) (p : RealProgram.Parameters) (r : Fin BM) (d : Nat) : Nat :=
  RealSetup.qBase s p + (s.pids 0 * BM + r.val) * p.qRow + d

def key (s : BlockState) (BN : Nat) (p : RealProgram.Parameters) (xs : CSRMath.Inputs)
    (visit : Nat) (j : Fin BN) (d : Nat) : Nat :=
  RealSetup.kBase s p + (CSRMath.selectedKey (schedule s p) xs BN visit j).toNat * p.kRow + d

def value (s : BlockState) (BN : Nat) (p : RealProgram.Parameters) (xs : CSRMath.Inputs)
    (visit : Nat) (j : Fin BN) (d : Nat) : Nat :=
  RealSetup.vBase s p + (CSRMath.selectedKey (schedule s p) xs BN visit j).toNat * p.vRow + d

def output (s : BlockState) (BM : Nat) (p : RealProgram.Parameters) (r : Fin BM) (d : Nat) : Nat :=
  RealSetup.batch s p * p.oBatch + RealSetup.head s p * p.oHead +
    (s.pids 0 * BM + r.val) * p.oRow + d

/-- Only visited CSR cells and active data lanes require allocation. The
channel width is BD, or exactly 2*BD when the second source branch runs. -/
structure Bounds (bounds : RegionBounds) (s : BlockState) (BM BN BD : Nat)
    (p : RealProgram.Parameters) (xs : CSRMath.Inputs) : Prop where
  rows : ∀ next, RealInputs.rowAddress s p next < bounds (Region.cast p.rows)
  columns : ∀ visit, visit < CSRMath.count (schedule s p) xs →
    (RealInputs.columnSlot s p xs visit).toNat < bounds (Region.cast p.cols)
  query : ∀ r d, d < RealInputs.width BD p → (p.evenM = Bool.true ∨ s.pids 0 * BM + r.val < p.context) →
    query s BM p r d < bounds p.Q
  key : ∀ visit, visit < CSRMath.count (schedule s p) xs → ∀ j d, d < RealInputs.width BD p →
    (p.evenN = Bool.true ∨ CSRMath.selectedKey (schedule s p) xs BN visit j < (p.context : Int)) →
    key s BN p xs visit j d < bounds p.K
  value : ∀ visit, visit < CSRMath.count (schedule s p) xs → ∀ j d, d < RealInputs.width BD p →
    (p.evenN = Bool.true ∨ CSRMath.selectedKey (schedule s p) xs BN visit j < (p.context : Int)) →
    value s BN p xs visit j d < bounds p.V
  output : ∀ r d, d < RealInputs.width BD p → s.pids 0 * BM + r.val < p.context →
    output s BM p r d < bounds p.Out

theorem output_address (s : BlockState) (BM BD : Nat) (p : RealProgram.Parameters)
    (second : Bool) (r : Fin BM) (d : Fin BD) :
    RealStores.offset s BM BD p second r d = output s BM p r (channel BD second d) := by
  cases second <;> simp [RealStores.offset, output, RealSetup.rows, channel,
    Nat.add_assoc, Nat.add_comm]

theorem query_bounds (bounds : RegionBounds) (s : BlockState) (BM BN BD : Nat)
    (p : RealProgram.Parameters) (xs : CSRMath.Inputs) (hw : Bounds bounds s BM BN BD p xs)
    (second : Bool) (hb : second = Bool.true → 2 ≤ p.blocks) :
    RealQuery.Bounds bounds (RealSetup.queryPointers s BM BD p)
      (RealSetup.rows s BM) p.context p.evenM second := by
  intro r d ha
  rw [query_address]
  exact hw.query r _ (RealInputs.channel_lt BD p second d hb) ha

theorem output_bounds (bounds : RegionBounds) (s : BlockState) (BM BN BD : Nat)
    (p : RealProgram.Parameters) (xs : CSRMath.Inputs) (hw : Bounds bounds s BM BN BD p xs)
    (second : Bool) (hb : second = Bool.true → 2 ≤ p.blocks) :
    RealStores.Bounds bounds s BM BD p second := by
  intro r d hr
  rw [output_address]
  exact hw.output r _ (RealInputs.channel_lt BD p second d hb) hr

theorem loop_bounds (bounds : RegionBounds) (s : BlockState) (BM BN BD : Nat)
    (p : RealProgram.Parameters) (xs : CSRMath.Inputs) (hm : RealInputs.Memory s BD p xs)
    (hv : RealInputs.Valid s p xs) (hw : Bounds bounds s BM BN BD p xs) :
    RealLoopSafety.Bounds bounds s BM BN BD p := by
  have hcount := RealInputs.count_matches s BD p xs hm
  have hk (second : Bool) (hb : second = Bool.true → 2 ≤ p.blocks)
      (visit : Nat) (hn : visit < RealLoop.count s p) :
      RealKey.Bounds bounds (RealSetup.keyPointers s BN BD p)
        (RealVisit.start s (RealLoop.operands s BM BN BD p visit) p) p.kRow p.context p.evenN second := by
    rw [hcount] at hn
    rw [RealInputs.start_matches s BM BN BD p xs hm visit hn]
    intro d j ha
    rw [key_address s BN BD p second _ (RealInputs.start_nonnegative s BN p xs hv visit hn)]
    apply hw.key visit hn j _ (RealInputs.channel_lt BD p second d hb)
    simpa only [RealKey.active, CSRMath.selectedKey, RealValueFormula.keyStart, Int.add_comm] using ha
  have hval (second : Bool) (hb : second = Bool.true → 2 ≤ p.blocks)
      (visit : Nat) (hn : visit < RealLoop.count s p) :
      RealValueSafety.Bounds bounds (RealSetup.valuePointers s BN BD p)
        (RealVisit.start s (RealLoop.operands s BM BN BD p visit) p) p.vRow p.context p.evenN second := by
    rw [hcount] at hn
    rw [RealInputs.start_matches s BM BN BD p xs hm visit hn]
    intro j d ha
    rw [value_address s BN BD p second _ (RealInputs.start_nonnegative s BN p xs hv visit hn)]
    apply hw.value visit hn j _ (RealInputs.channel_lt BD p second d hb)
    simpa only [RealValueLoad.active, CSRMath.selectedKey, RealValueFormula.keyStart, Int.add_comm] using ha
  refine ⟨?_, hk Bool.false (by simp), fun hb => hk Bool.true (fun _ => hb),
    hval Bool.false (by simp), fun hb => hval Bool.true (fun _ => hb)⟩
  intro visit hn
  rw [hcount] at hn
  simpa [RealColumn.address, (RealInputs.interval_matches s BD p xs hm).1,
    RealInputs.columnSlot, schedule, Int.add_assoc] using hw.columns visit hn

/-- The arithmetic input windows imply safety of every source access,
including padding loads and the final row-masked stores. -/
theorem source_safe (bounds : RegionBounds) (swapped : Bool) (s : BlockState) (BM BN BD : Nat)
    (p : RealProgram.Parameters) (xs : CSRMath.Inputs) (hm : RealInputs.Memory s BD p xs)
    (hv : RealInputs.Valid s p xs) (hw : Bounds bounds s BM BN BD p xs) (hN : 0 < BN) :
    (RealProgram.kernel swapped BM BN BD p).toAlgKernel.TraceSafe bounds s := by
  apply RealLoopSafety.source_safe bounds swapped s BM BN BD p hN
    (loop_bounds bounds s BM BN BD p xs hm hv hw)
    (query_bounds bounds s BM BN BD p xs hw Bool.false (by simp))
    (fun hb => query_bounds bounds s BM BN BD p xs hw Bool.true (fun _ => hb))
    _ (output_bounds bounds s BM BN BD p xs hw Bool.false (by simp))
    (fun hb => output_bounds bounds s BM BN BD p xs hw Bool.true (fun _ => hb))
  simpa [RealRows.address, RealInputs.rowAddress, schedule] using hw.rows Bool.true

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealWindows
