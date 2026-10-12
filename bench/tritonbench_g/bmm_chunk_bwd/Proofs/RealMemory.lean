import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealFinish

/-! Read the complete kernel's actual stores as a formula over initial memory.
Only active output addresses must be distinct; every other tagged memory cell
is preserved. The independent-array/flat-memory contract is a subsequent layer. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealMemory
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def base (s : BlockState) (p : RealProgram.Parameters) (batch row head : Nat) : Nat :=
  RealResidual.baseValue p.chunk batch row head (s.pids 1) (RealSetup.chunk s p) (RealSetup.head s p)

def address (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (batch row head column : Nat) (i : TileIndex [BM, BN]) : Nat :=
  base s p batch row head + (RealSetup.rows s BM BN p i.1 * row + RealSetup.columns s BN p i.2.1 * column)

def active (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (i : TileIndex [BM, BN]) : Prop :=
  RealSetup.rows s BM BN p i.1 < RealSetup.limit s p ∧ RealSetup.columns s BN p i.2.1 < p.K

def formula (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (i : TileIndex [BM, BN]) : ℝ :=
  (∑ k ∈ Finset.range (RealSetup.limit s p), RealLoop.product s BM BN p i.1 i.2.1 k) +
    if p.hasResidual then s.readMem p.Res (address s BM BN p p.rBatch p.rRow p.rHead p.rColumn i) else 0

def final (initial current : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters) : BlockState :=
  RealFinish.result current BM BN p (initial.pids 1) (RealSetup.chunk initial p) (RealSetup.head initial p)
    (RealSetup.tileM initial BN p) (RealSetup.tileN initial BN p) (RealSetup.limit initial p)
    (RealLoop.carry initial BM BN BCS p (RealMath.blocks (RealSetup.limit initial p) BCS))

theorem stored_value (initial current : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (hBCS : 0 < BCS) (hm : current.mem = initial.mem) (i : TileIndex [BM, BN])
    (hi : active initial BM BN p i) :
    RealFinish.values
      (RealFinish.refreshed current BM BN (RealSetup.tileM initial BN p) (RealSetup.tileN initial BN p))
      BM BN p (initial.pids 1) (RealSetup.chunk initial p) (RealSetup.head initial p) (RealSetup.limit initial p)
      (RealFinish.rows BM (RealSetup.tileM initial BN p)) (RealFinish.columns BN (RealSetup.tileN initial BN p))
      (RealLoop.carry initial BM BN BCS p (RealMath.blocks (RealSetup.limit initial p) BCS)) i.1 i.2.1 =
      formula initial BM BN p i := by
  have hlen : RealSetup.limit initial p ≤ p.chunk := Nat.min_le_left _ _
  have hrow := lt_of_lt_of_le hi.1 hlen
  have hmask : RealSetup.rows initial BM BN p i.1 < p.chunk ∧
      RealSetup.columns initial BN p i.2.1 < p.K := ⟨hrow, hi.2⟩
  have ha : RealFinish.rows BM (RealSetup.tileM initial BN p) i.1 < RealSetup.limit initial p ∧
      RealFinish.columns BN (RealSetup.tileN initial BN p) i.2.1 < p.K := hi
  have hc : RealLoop.carry initial BM BN BCS p (RealMath.blocks (RealSetup.limit initial p) BCS) i.1 i.2.1 =
      ∑ k ∈ Finset.range (RealSetup.limit initial p), RealLoop.product initial BM BN p i.1 i.2.1 k := by
    unfold RealLoop.carry
    rw [if_pos hmask, RealMath.partial_complete _ _ hBCS]
  unfold RealFinish.values
  rw [hc]
  cases hf : p.hasResidual
  · simp [hf, formula]
  · simp only [hf, ↓reduceIte, RealResidual.residualValue, if_pos ha]
    simp only [formula, hf, ↓reduceIte]
    congr 1
    simp only [RealFinish.refreshed, BlockState.readMem, BlockState.setReg_mem, hm]
    rfl

theorem readback (initial current : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (hBCS : 0 < BCS) (hm : current.mem = initial.mem)
    (hinj : ∀ a b, active initial BM BN p a → active initial BM BN p b →
      address initial BM BN p p.oBatch p.oRow p.oHead p.oColumn a =
        address initial BM BN p p.oBatch p.oRow p.oHead p.oColumn b → a = b)
    (i : TileIndex [BM, BN]) (hi : active initial BM BN p i) :
    (final initial current BM BN BCS p).mem p.Db
      (address initial BM BN p p.oBatch p.oRow p.oHead p.oColumn i) =
      MemCell.real (formula initial BM BN p i) := by
  unfold final RealFinish.result
  apply Eq.trans (RealStore.readback _ _ _ _ _ _ _ _ _ _ _ _ i hi hinj)
  exact congrArg MemCell.real (stored_value initial current BM BN BCS p hBCS hm i hi)

theorem frame (initial current : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (hm : current.mem = initial.mem) (r : RegionName) (a : Nat)
    (h : ∀ i, active initial BM BN p i →
      ¬ (p.Db = r ∧ address initial BM BN p p.oBatch p.oRow p.oHead p.oColumn i = a)) :
    (final initial current BM BN BCS p).mem r a = initial.mem r a := by
  unfold final RealFinish.result
  apply Eq.trans (RealStore.frame _ _ _ _ _ _ _ _ _ _ _ _ r a h)
  simp only [RealFinish.outputPrepared, BlockState.setReg_mem]
  cases hf : p.hasResidual <;>
    simp [RealFinish.afterResidual, hf, RealResidual.residualResult, RealFinish.refreshed, hm]

/-- Both full source variants terminate, store the initial-memory formula at
every live output lane, and preserve every other tagged memory cell. -/
theorem executes (swapped : Bool) (initial : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (hBCS : 0 < BCS)
    (hinj : ∀ a b, active initial BM BN p a → active initial BM BN p b →
      address initial BM BN p p.oBatch p.oRow p.oHead p.oColumn a =
        address initial BM BN p p.oBatch p.oRow p.oHead p.oColumn b → a = b) :
    ∃ t, stepStmts (RealProgram.kernel swapped BM BN BCS p).toAlgKernel.body initial = some t ∧
      (∀ i, active initial BM BN p i → t.mem p.Db
        (address initial BM BN p p.oBatch p.oRow p.oHead p.oColumn i) = MemCell.real (formula initial BM BN p i)) ∧
      (∀ r a, (∀ i, active initial BM BN p i →
        ¬ (p.Db = r ∧ address initial BM BN p p.oBatch p.oRow p.oHead p.oColumn i = a)) →
        t.mem r a = initial.mem r a) := by
  obtain ⟨current, hc, he⟩ := RealFinish.kernel_executes swapped initial BM BN BCS p
  exact ⟨final initial current BM BN BCS p, he,
    fun i hi => readback initial current BM BN BCS p hBCS hc.mem hinj i hi,
    fun r a h => frame initial current BM BN BCS p hc.mem r a h⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealMemory
