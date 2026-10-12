import bench.tritonbench_g.context_attn_nopad.Proofs.NumericNormalizer
import bench.tritonbench_g.context_attn_nopad.Proofs.NumericScales
import bench.tritonbench_g.context_attn_nopad.Proofs.LoopState
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.NumericUpdate
open VeriTile.Triton Formula Program
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 6000

def update (p : Layout) (mi li : Tile .real [p.BM]) (acc : Tile .real [p.BM,p.D])
    (qk : Tile .real [p.BM,p.BN]) (value : Tile .real [p.BN,p.D]) (blockMax : Tile .real [p.BM]) :=
  let center := blockMax
  let prob := StatsExecution.exponentials p qk center
  let lij := StatsExecution.denominator p qk center
  let mn := NormalizerExecution.newMaximum p mi center
  let alpha := NormalizerExecution.alpha p mi center
  let beta := NormalizerExecution.beta p mi center
  let ln := NormalizerExecution.newDenominator p mi center li lij
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
    FormulaStep.maskedScore p (launch initial) (inputs initial r) scale i.val (c*p.BN+j.val)
  let total := fun i : Fin p.BM => Formula.denominator p (launch initial) (inputs initial r) scale i.val c
  let numerator := fun (i : Fin p.BM) (d : Fin p.D) =>
    Formula.numerator p (launch initial) (inputs initial r) scale i.val d.val c
  let values := fun (j : Fin p.BN) (d : Fin p.D) => Formula.value p (launch initial) (inputs initial r) (c*p.BN+j.val) d.val
  let qk := LoopState.score p r initial scale c
  let center := blockMax
  let ctr := fun i : Fin p.BM => blockMax.data (i,PUnit.unit)
  have hc : ∀ i : Fin p.BM, ∃ next : ℝ, m i ⊔ ctr i = some next := by
    intro i
    have hn := NumericCells.next_finite p (launch initial) (inputs initial r) scale c m hbot
      qk blockMax hBN (by intros; rfl) hrm i
    obtain ⟨x,hx⟩ := WithBot.ne_bot_iff_exists.mp hn
    exact ⟨x,hx.symm⟩
  let next := fun i => Classical.choose (hc i)
  have hnext : ∀ i, m i ⊔ ctr i = some (next i) := fun i => Classical.choose_spec (hc i)
  have hcenter : ∀ i j, ctr i=⊥ → scores i j=⊥ := by
    intro i j h
    exact NumericCells.bottom_center p scores qk blockMax hBN (by intros; rfl) hrm i j h
  let oldl := fun i => NormalizedStep.factor (m i)*total i
  let raw := fun i => NormalizedStep.alpha (m i) (next i)*oldl i+
    NormalizedStep.alpha (ctr i) (next i)*(∑ j,NormalizedStep.shifted (scores i j) (ctr i))
  let mi : Tile .real [p.BM] := Tile.vec m
  let li := Invariant.denominatorTile p r initial scale c m
  let acc := Invariant.accumulatorTile p r initial scale c
  let prob := StatsExecution.exponentials p qk center
  let lij := StatsExecution.denominator p qk center
  let mn := NormalizerExecution.newMaximum p mi center
  let al := NormalizerExecution.alpha p mi center
  let be := NormalizerExecution.beta p mi center
  let ln := NormalizerExecution.newDenominator p mi center li lij
  let prob2 := ScalesExecution.probabilities p prob be ln
  let acc1 := ScalesExecution.accumulator p li ln al acc
  have htbot : ∀ i, m i = ⊥ → total i = 0 := by
    intro i hi
    dsimp [total]
    rw [hbot i hi,Formula.denominator_zero]
  have hmn : ∀ i, mn.data (i,PUnit.unit) = some (next i) := by
    intro i
    exact (NumericNormalizer.maximum_cell p mi center i (m i) (ctr i) rfl rfl).trans (hnext i)
  have hal : ∀ i, al.data (i,PUnit.unit) = some (NormalizedStep.alpha (m i) (next i)) := by
    intro i
    exact NumericNormalizer.alpha_cell p mi center i (m i) (ctr i) (next i) rfl rfl (hnext i)
  have hbe : ∀ i, be.data (i,PUnit.unit) = some (NormalizedStep.alpha (ctr i) (next i)) := by
    intro i
    exact NumericNormalizer.beta_cell p mi center i (m i) (ctr i) (next i) rfl rfl (hnext i)
  have hln : ∀ i, ln.data (i,PUnit.unit) = some (raw i) := by
    intro i
    exact NumericNormalizer.denominator_cell p mi center li lij i (m i) (ctr i) (next i) (oldl i)
      (∑ j,NormalizedStep.shifted (scores i j) (ctr i)) rfl rfl (hnext i) rfl
      (NumericCells.denominator_cell p qk center i (scores i) (ctr i) (by intros; rfl) rfl)
  have hprob : ∀ i j, prob2.data (i,j,PUnit.unit) =
      some (NormalizedStep.shifted (scores i j) (ctr i)*(NormalizedStep.alpha (ctr i) (next i)/raw i)) := by
    intro i j
    apply NumericScales.probability_cell p prob be ln i j
    · exact NumericCells.exponentials_cell p qk center i j _ _ rfl rfl
    · exact hbe i
    · exact hln i
  have hacc : ∀ i d, acc1.data (i,d,PUnit.unit) =
      some ((numerator i d/total i)*(oldl i/raw i*NormalizedStep.alpha (m i) (next i))) := by
    intro i d
    exact NumericScales.accumulator_cell p li ln al acc i d (oldl i) (raw i)
      (NormalizedStep.alpha (m i) (next i)) (numerator i d/total i) rfl (hln i) (hal i) rfl
  have hformula (i : Fin p.BM) (d : Fin p.D) := FormulaStep.update p (launch initial) (inputs initial r)
    scale i.val d.val c (m i) (ctr i) (next i) (oldl i) (numerator i d/total i) hBN rfl (hbot i) (hcenter i) rfl
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
      have hd := NormalizedStep.denominator_update (scores i) (m i) (ctr i) (next i) (oldl i) (total i) rfl (htbot i) (hcenter i)
      apply congrArg some
      simpa only [raw,oldl,total,scores,FormulaStep.weight_matches,← Formula.denominator_succ] using hd
    · apply Tile.ext
      rintro ⟨i,d,⟨⟩⟩
      rw [NumericScales.commit_cell p prob2 (LoopState.value p r initial c) acc1 i d
        (fun j => NormalizedStep.shifted (scores i j) (ctr i)*(NormalizedStep.alpha (ctr i) (next i)/raw i))
        (fun j => values j d)
        ((numerator i d/total i)*(oldl i/raw i*NormalizedStep.alpha (m i) (next i)))
        (hprob i) (by intros; rfl) (hacc i d)]
      apply congrArg some
      have he := (hformula i d).2
      simpa only [scores,values,numerator,total,raw,oldl] using he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.NumericUpdate
