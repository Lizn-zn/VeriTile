import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealExecution

/-! Both complete variants store the initial-memory contraction, zero padded
rows/columns and optionally filter signed sequence IDs. Causal-skipped tiles
preserve all tagged cells; active stores preserve every cell outside the actual
full-chunk output window. No executed output is used as a reference formula. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealMemory
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

def base (s : BlockState) (p : RealProgram.Parameters) : Nat :=
  s.pids 1 * p.oBatch + RealSetup.chunk s p * p.oChunk + RealSetup.head s p * p.oHead

def address (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (i : TileIndex [BM, BN]) : Nat :=
  base s p + (RealSetup.rows s BM BN p i.1 * p.oRow + RealSetup.columns s BN p i.2.1 * p.oColumn)

def active (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (i : TileIndex [BM, BN]) : Prop :=
  RealSetup.active s BM BN p = Bool.true ∧
    RealSetup.rows s BM BN p i.1 < p.chunk ∧ RealSetup.columns s BN p i.2.1 < p.chunk

def sequenceID (s : BlockState) (p : RealProgram.Parameters) (offset : Nat) : Int :=
  s.readMemValue .int (Region.cast p.Seq) (RealSetup.seqBase s p + offset * p.sRow)

def formula (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (i : TileIndex [BM, BN]) : ℝ :=
  if RealSetup.rows s BM BN p i.1 < RealSetup.limit s p ∧
     RealSetup.columns s BN p i.2.1 < RealSetup.limit s p ∧
     (p.hasSeq = Bool.false ∨ sequenceID s p (RealSetup.rows s BM BN p i.1) =
       sequenceID s p (RealSetup.columns s BN p i.2.1))
  then ∑ k ∈ Finset.range p.K, RealLoop.product s BM BN p i.1 i.2.1 k else 0

abbrev final := RealExecution.finished

theorem stored_value (initial current : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hBK : 0 < BK) (hm : current.mem = initial.mem) (i : TileIndex [BM, BN]) :
    RealFinish.values
      (RealPostlude.refreshed current BM BN (RealSetup.tileM initial BN p) (RealSetup.tileN initial BN p))
      BM BN p (RealSetup.chunk initial p) (RealSetup.seqBase initial p)
      (RealPostlude.rows BM (RealSetup.tileM initial BN p)) (RealPostlude.columns BN (RealSetup.tileN initial BN p))
      (RealLoop.carry initial BM BN BK p (BmmChunkBwd.RealMath.blocks p.K BK)) i.1 i.2.1 =
      formula initial BM BN p i := by
  simp only [RealFinish.values, RealLoop.carry, BmmChunkBwd.RealMath.partial_complete _ _ hBK,
    RealSequence.indexValues, Tile.vec, RealPostlude.refreshed, BmmChunkBwd.RealFinish.refreshed,
    BlockState.readMemValue, BlockState.readMemTyped, BlockState.setReg_mem, hm]
  change (if p.hasSeq then
      if (if RealSetup.rows initial BM BN p i.1 < RealSetup.limit initial p then
          sequenceID initial p (RealSetup.rows initial BM BN p i.1) else -1) =
         (if RealSetup.columns initial BN p i.2.1 < RealSetup.limit initial p then
          sequenceID initial p (RealSetup.columns initial BN p i.2.1) else -2)
      then (if RealSetup.rows initial BM BN p i.1 < RealSetup.limit initial p ∧
          RealSetup.columns initial BN p i.2.1 < RealSetup.limit initial p then
          ∑ k ∈ Finset.range p.K, RealLoop.product initial BM BN p i.1 i.2.1 k else 0) else 0
    else (if RealSetup.rows initial BM BN p i.1 < RealSetup.limit initial p ∧
          RealSetup.columns initial BN p i.2.1 < RealSetup.limit initial p then
          ∑ k ∈ Finset.range p.K, RealLoop.product initial BM BN p i.1 i.2.1 k else 0)) = _
  cases hs : p.hasSeq <;>
    by_cases hr : RealSetup.rows initial BM BN p i.1 < RealSetup.limit initial p <;>
    by_cases hc : RealSetup.columns initial BN p i.2.1 < RealSetup.limit initial p <;>
      simp [formula, hs, hr, hc]

theorem readback (initial current : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hBK : 0 < BK) (hm : current.mem = initial.mem)
    (hinj : ∀ a b, active initial BM BN p a → active initial BM BN p b →
      address initial BM BN p a = address initial BM BN p b → a = b)
    (i : TileIndex [BM, BN]) (hi : active initial BM BN p i) :
    (final initial current BM BN BK p).mem p.Out (address initial BM BN p i) =
      MemCell.real (formula initial BM BN p i) := by
  unfold final RealExecution.finished RealFinish.result
  apply Eq.trans (RealStore.readback _ _ _ _ _ _ _ _ _ _ _ i hi.2 ?_)
  · exact congrArg MemCell.real (stored_value initial current BM BN BK p hBK hm i)
  · intro a b ha hb he
    exact hinj a b ⟨hi.1, ha⟩ ⟨hi.1, hb⟩ he

theorem frame (initial current : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hm : current.mem = initial.mem) (hactive : RealSetup.active initial BM BN p = Bool.true)
    (r : RegionName) (a : Nat)
    (h : ∀ i, active initial BM BN p i → ¬ (p.Out = r ∧ address initial BM BN p i = a)) :
    (final initial current BM BN BK p).mem r a = initial.mem r a := by
  unfold final RealExecution.finished RealFinish.result
  apply Eq.trans (RealStore.frame _ _ _ _ _ _ _ _ _ _ _ r a ?_)
  · simp only [RealPostlude.outputPrepared, BlockState.setReg_mem]
    cases hs : p.hasSeq <;>
      simp [RealFinish.afterSequence, hs, RealSequence.result, RealPostlude.refreshed,
        BmmChunkBwd.RealFinish.refreshed, hm]
  · intro i hi
    exact h i ⟨hactive, hi⟩

theorem executes (swapped : Bool) (initial : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hBK : 0 < BK)
    (hinj : ∀ a b, active initial BM BN p a → active initial BM BN p b →
      address initial BM BN p a = address initial BM BN p b → a = b) :
    ∃ t, stepStmts (RealProgram.kernel swapped BM BN BK p).toAlgKernel.body initial = some t ∧
      (∀ i, active initial BM BN p i → t.mem p.Out (address initial BM BN p i) =
        MemCell.real (formula initial BM BN p i)) ∧
      (∀ r a, (∀ i, active initial BM BN p i → ¬ (p.Out = r ∧ address initial BM BN p i = a)) →
        t.mem r a = initial.mem r a) := by
  cases hactive : RealSetup.active initial BM BN p
  · refine ⟨RealSetup.prologueResult initial BM BN p,
      RealExecution.skipped_executes swapped initial BM BN BK p hactive, ?_, ?_⟩
    · intro i hi
      simp [active, hactive] at hi
    · intro r a _
      cases hc : p.causal <;> simp [RealSetup.prologueResult, hc]
  · obtain ⟨current, hc, he⟩ := RealExecution.kernel_executes swapped initial BM BN BK p hactive
    exact ⟨final initial current BM BN BK p, he,
      fun i hi => readback initial current BM BN BK p hBK hc.mem hinj i hi,
      fun r a h => frame initial current BM BN BK p hc.mem hactive r a h⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealMemory
