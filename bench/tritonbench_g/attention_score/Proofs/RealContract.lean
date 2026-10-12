import bench.tritonbench_g.attention_score.Proofs.RealIOBridge
import bench.tritonbench_g.attention_score.Proofs.RealSource

/-! Bind the complete source to finite Q/K/M arrays and actual strided windows.
Unchecked Q/K loads require their whole windows; checked loads require only
active lanes. M is unmasked, including padded rows of a partial final block.
Output bounds follow the source mask. These are address conditions only. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealContract
open VeriTile.Triton RealSource
set_option maxHeartbeats 1000000
noncomputable section

def io (B D : Nat) (p : Parameters) (candidate : Bool) : RealIO.IO (rows B p) B D where
  kernel := kernel B D p candidate
  projection := projection B D p candidate
  Q := p.Q
  K := p.K
  M := p.M
  Out := p.Out
  queryWindow := fun _ headBatch _ i =>
    headBatch / p.heads * p.queryBatch + headBatch % p.heads * p.queryHead +
      i.1.val * p.queryRow + i.2.1.val * p.queryDim
  keyWindow := fun startN headBatch _ i =>
    headBatch / p.heads * p.keyBatch + (headBatch % p.heads) / (p.heads / p.kvHeads) * p.keyHead +
      i.1.val * p.keyDim + (startN * B + i.2.1.val) * p.keyRow
  normalizerWindow := fun _ headBatch _ i => headBatch * p.roundedContext + i.1.val
  outputWindow := fun startN headBatch _ i => outputBase p headBatch + startN * B + i.val
  queryMask := fun _ _ _ i => p.evenM = Bool.true ∨ i.1.val < p.queryContext
  keyMask := fun startN _ _ i => p.evenN = Bool.true ∨ startN * B + i.2.1.val < p.keyContext
  outputMask := fun startN _ _ i => startN * B + i.val < p.keyContext

def expected (B D : Nat) (p : Parameters) (startN _headBatch : Nat)
    (xs : RealInputs.Inputs (rows B p) B D) : Fin B → ℝ :=
  RealInputs.column xs p.queryContext p.keyContext startN p.scale
    p.window p.complement p.evenM p.evenN p.offset p.size

theorem pins (B D : Nat) (p : Parameters) (candidate : Bool) (s : BlockState)
    (xs : RealInputs.Inputs (rows B p) B D) (hp : RealIO.Pins (io B D p candidate) s xs) :
    RealInputs.Pins (config B D p candidate s) s (rows B p) xs :=
  ⟨hp.query, hp.key, hp.normalizer⟩

theorem bounds (B D : Nat) (p : Parameters) (candidate : Bool) (s : BlockState)
    (b : RegionBounds) (hb : RealIO.Bounds (io B D p candidate) s b) :
    RealSafety.Bounds b (config B D p candidate s) p.roundedContext p.Out (outputBase p (s.pids 1)) := by
  constructor
  · intro j hj i hi
    have ha : p.evenM = Bool.true ∨ j*B+i.1.val < p.queryContext := by
      by_cases he : p.evenM = Bool.true
      · exact Or.inl he
      · right
        have he' : p.evenM = Bool.false := Bool.eq_false_iff.mpr he
        have hactive : j*B+i.1.val < p.queryContext ∧ i.2.1.val < D := by
          simpa [config, RealPrelude.config, RealLoads.queryPointer, RealLoads.checks, he'] using hi
        exact hactive.1
    let row : Fin (rows B p) := RealInputs.rowIndex (⟨j,hj⟩ : Fin (blocks B p)) i.1
    simpa [io, config, RealPrelude.config, RealLoads.queryPointer, row, RealInputs.rowIndex] using
      hb.query (row,i.2.1,PUnit.unit) ha
  · intro i hi
    have ha : p.evenN = Bool.true ∨ s.pids 0*B+i.2.1.val < p.keyContext := by
      by_cases he : p.evenN = Bool.true
      · exact Or.inl he
      · right
        have he' : p.evenN = Bool.false := Bool.eq_false_iff.mpr he
        have hactive : i.1.val < D ∧ s.pids 0*B+i.2.1.val < p.keyContext := by
          simpa [config, RealPrelude.config, RealLoads.keyPointer, RealLoads.checks, he'] using hi
        exact hactive.2
    simpa [io, config, RealPrelude.config, RealLoads.keyPointer] using hb.key i ha
  · intro row hr
    exact hb.normalizer (⟨row,hr⟩,PUnit.unit)
  · exact hb.output

theorem implements (B D : Nat) (p : Parameters) (candidate : Bool) (hB : 0 < B) :
    (io B D p candidate).Implements (expected B D p) := by
  apply RealIO.IO.Implements.intro
  · exact flatten_ok B D p candidate
  · intro b s hb
    exact safe b s B D p candidate hB (bounds B D p candidate s b hb)
  · intro s xs hp
    obtain ⟨t,he,hy,hframe⟩ := executes s B D p candidate hB
    refine ⟨t,he,?_,hframe⟩
    intro i ha
    have hcol := RealInputs.column_pinned (config B D p candidate s) s (blocks B p) xs
      (pins B D p candidate s xs hp) i
    exact (hy i).trans ((if_pos ha).trans hcol)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.RealContract
