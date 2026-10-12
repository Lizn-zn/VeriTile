import bench.tritonbench_g.context_attn_mistral.Proofs.NumericNormalizer
import bench.tritonbench_g.context_attn_mistral.Proofs.NumericScales
import bench.tritonbench_g.context_attn_mistral.Proofs.LoopState
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.NumericUpdate
open VeriTile.Triton Formula Program
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 6000

def update (p : Layout) (mi li : Tile .real [p.BM]) (acc : Tile .real [p.BM,p.D])
    (qk : Tile .real [p.BM,p.BN]) (value : Tile .real [p.BN,p.D]) (blockMax : Tile .real [p.BM]) :=
  let center := StatsExecution.center p blockMax
  let prob := StatsExecution.exponentials p qk center
  let lij := StatsExecution.denominator p qk center
  let mn := NormalizerExecution.newMaximum p mi center
  let alpha := NormalizerExecution.alpha p mi center
  let beta := NormalizerExecution.beta p mi center
  let ln := NormalizerExecution.guardedDenominator p mi center li lij
  let prob := ScalesExecution.probabilities p prob beta ln
  let acc := ScalesExecution.accumulator p li ln alpha acc
  (mn,ln,CommitExecution.accumulator p prob value acc)

theorem correct (p : Layout) (r : Regions) (initial : BlockState) (scale : ℝ) (c : Nat)
    (hBN : 0 < p.BN) (m : Fin p.BM → WithBot ℝ) (hbot : ∀ i, m i = ⊥ → c=0)
    (blockMax : Tile .real [p.BM])
    (hrm : Tile.reduceMaxDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length)
      (LoopState.score p r initial scale c) = some blockMax) :
    ∃ next : Fin p.BM → ℝ,
      update p (Tile.vec m) (Invariant.denominatorTile p r initial scale c m)
        (Invariant.accumulatorTile p r initial scale c) (LoopState.score p r initial scale c)
        (LoopState.value p r initial c) blockMax =
      (Tile.vec (fun i => some (next i)),
       Invariant.denominatorTile p r initial scale (c+1) (fun i => some (next i)),
       Invariant.accumulatorTile p r initial scale (c+1)) := by
  let scores := fun (i : Fin p.BM) (j : Fin p.BN) =>
    Formula.score p (launch initial) (inputs initial r) scale i.val (c*p.BN+j.val)
  let total := fun i : Fin p.BM => Formula.denominator p (launch initial) (inputs initial r) scale i.val c
  let numerator := fun (i : Fin p.BM) (d : Fin p.D) =>
    Formula.numerator p (launch initial) (inputs initial r) scale i.val d.val c
  let values := fun (j : Fin p.BN) (d : Fin p.D) => Formula.value p (launch initial) (inputs initial r) (c*p.BN+j.val) d.val
  let qk := LoopState.score p r initial scale c
  let center := StatsExecution.center p blockMax
  have hc : ∀ i : Fin p.BM, ∃ ctr : ℝ, center.data (i,PUnit.unit) = some ctr := by
    intro i
    exact NumericCells.center_finite p blockMax i (NumericCells.maximum_finite p scores qk blockMax hBN (by intros; rfl) hrm i)
  let ctr := fun i => Classical.choose (hc i)
  have hctr : ∀ i, center.data (i,PUnit.unit) = some (ctr i) := fun i => Classical.choose_spec (hc i)
  let next := fun i => NumericNormalizer.nextCenter (m i) (ctr i)
  let oldl := fun i => NormalizedStep.factor (m i)*total i
  let raw := fun i => NormalizedStep.alpha (m i) (next i)*oldl i+
    Real.exp (ctr i-next i)*NormalizedStep.blockSum (scores i) (ctr i)
  let mi : Tile .real [p.BM] := Tile.vec m
  let li := Invariant.denominatorTile p r initial scale c m
  let acc := Invariant.accumulatorTile p r initial scale c
  let prob := StatsExecution.exponentials p qk center
  let lij := StatsExecution.denominator p qk center
  let mn := NormalizerExecution.newMaximum p mi center
  let al := NormalizerExecution.alpha p mi center
  let be := NormalizerExecution.beta p mi center
  let ln := NormalizerExecution.guardedDenominator p mi center li lij
  let prob2 := ScalesExecution.probabilities p prob be ln
  let acc1 := ScalesExecution.accumulator p li ln al acc
  have htbot : ∀ i, m i = ⊥ → total i = 0 := by
    intro i hi
    dsimp [total]
    rw [hbot i hi,Formula.denominator_zero]
  have hrawpos : ∀ i, 0 < raw i := by
    intro i
    exact NormalizedStep.denominator_positive hBN (scores i) (m i) (ctr i) (next i) (oldl i) (total i)
      rfl (htbot i) (Formula.denominator_nonnegative _ _ _ _ _ _)
  have hmn : ∀ i, mn.data (i,PUnit.unit) = some (next i) := by
    intro i
    exact NumericNormalizer.maximum_cell p mi center i (m i) (ctr i) rfl (hctr i)
  have hal : ∀ i, al.data (i,PUnit.unit) = some (NormalizedStep.alpha (m i) (next i)) := by
    intro i
    exact NumericNormalizer.alpha_cell p mi center i (m i) (ctr i) rfl (hctr i)
  have hbe : ∀ i, be.data (i,PUnit.unit) = some (Real.exp (ctr i-next i)) := by
    intro i
    exact NumericNormalizer.beta_cell p mi center i (m i) (ctr i) rfl (hctr i)
  have hln : ∀ i, ln.data (i,PUnit.unit) = some (raw i) := by
    intro i
    apply NumericNormalizer.guard_inactive p mi center li lij i (raw i) (hrawpos i)
    exact NumericNormalizer.denominator_cell p mi center li lij i (m i) (ctr i) (oldl i)
      (NormalizedStep.blockSum (scores i) (ctr i)) rfl (hctr i) rfl
      (NumericCells.denominator_cell p qk center i (scores i) (ctr i) (by intros; rfl) (hctr i))
  have hprob : ∀ i j, prob2.data (i,j,PUnit.unit) =
      some (Real.exp (scores i j-ctr i)*(Real.exp (ctr i-next i)/raw i)) := by
    intro i j
    apply NumericScales.probability_cell p prob be ln i j
    · exact NumericCells.exponentials_cell p qk center i j _ _ rfl (hctr i)
    · exact hbe i
    · exact hln i
  have hacc : ∀ i d, acc1.data (i,d,PUnit.unit) =
      some ((numerator i d/total i)*(oldl i/raw i*NormalizedStep.alpha (m i) (next i))) := by
    intro i d
    exact NumericScales.accumulator_cell p li ln al acc i d (oldl i) (raw i)
      (NormalizedStep.alpha (m i) (next i)) (numerator i d/total i) rfl (hln i) (hal i) rfl
  have hformula (i : Fin p.BM) (d : Fin p.D) := FormulaStep.update p (launch initial) (inputs initial r)
    scale i.val d.val c (m i) (ctr i) (next i) (oldl i) (numerator i d/total i) hBN rfl (hbot i) rfl
  refine ⟨next,?_⟩
  change (mn,ln,CommitExecution.accumulator p prob2 (LoopState.value p r initial c) acc1) = _
  apply Prod.ext
  · apply Tile.ext
    rintro ⟨i,⟨⟩⟩
    exact hmn i
  · apply Prod.ext
    · apply Tile.ext
      rintro ⟨i,⟨⟩⟩
      rw [hln i]
      have hd := NormalizedStep.denominator_update (scores i) (m i) (ctr i) (next i) (oldl i) (total i) rfl (htbot i)
      apply congrArg some
      simpa only [raw,oldl,total,scores,← Formula.denominator_succ] using hd
    · apply Tile.ext
      rintro ⟨i,d,⟨⟩⟩
      rw [NumericScales.commit_cell p prob2 (LoopState.value p r initial c) acc1 i d
        (fun j => Real.exp (scores i j-ctr i)*(Real.exp (ctr i-next i)/raw i))
        (fun j => values j d)
        ((numerator i d/total i)*(oldl i/raw i*NormalizedStep.alpha (m i) (next i)))
        (hprob i) (by intros; rfl) (hacc i d)]
      apply congrArg some
      have he := (hformula i d).2
      simpa only [scores,values,numerator,total,raw,oldl,
        if_neg (ne_of_gt (hrawpos i))] using he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.NumericUpdate
