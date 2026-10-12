import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealInputs

/-! The closed attention formula on six mathematical input arrays. Window
selection is over the finite global KV context. The no-iteration branch
preserves the initial carry, including INIT's denominator one; an executed
all-masked row instead follows the maximum/weighted-sum formula. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealInputFormula
open VeriTile.Triton RealPrologue RealInputs
noncomputable section

def row (BM BN BD start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext)
    (r : Fin BM) (initialValue : ℝ) (v : Fin p.layout.kvContext → ℝ) : RealFormula.RowState :=
  let seed := RealLoopFormula.rowSeed (initial BM BD p xs) r initialValue
  if (RealKeys.lower start BM BN p.window p.complement p.offset p.size : Int) <
      RealWindow.upper start BM BN p.layout.kvContext p.window p.complement p.offset then
    RealFormula.summarize seed
      (((List.finRange p.layout.kvContext).filter
        (fun key => RealKeys.keep start BM r.val p.window p.complement p.offset p.size key.val)).map
        (fun key => (score BM BD start p xs r key, v key)))
  else seed

def carry (BM BN BD start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext) :
    RealLoopFormula.Carry BM BD where
  center := fun r => (row BM BN BD start p xs r 0 (fun _ => 0)).center
  denominator := fun r => (row BM BN BD start p xs r 0 (fun _ => 0)).denominator
  numerator := fun i => (row BM BN BD start p xs i.1
    ((initial BM BD p xs).numerator i) (fun key => xs.value key i.2.1)).numerator

theorem row_pinned (BM BN BD : Nat) (p : Parameters) (s : BlockState)
    (xs : Inputs BM BD p.layout.kvContext) (hp : Pins BM BD p s xs)
    (r : Fin BM) (a : ℝ) (v : Nat → ℝ) (w : Fin p.layout.kvContext → ℝ)
    (hv : ∀ key, v key.val = w key) :
    RealLoopContext.fullRow s (loopParameters BM BN BD p s) (seed BM BD p s) r a v =
      row BM BN BD (s.pids 0) p xs r a w := by
  rw [RealLoopContext.fullRow, seed_pinned BM BD p s xs hp]
  unfold row
  have hm : (((List.range p.layout.kvContext).filter
        (RealLoopContext.keepKey (loopParameters BM BN BD p s) r)).map
        (fun key => (RealLoopContext.fullScore s (loopParameters BM BN BD p s) r key, v key))) =
      (((List.finRange p.layout.kvContext).filter
        (fun key => RealKeys.keep (s.pids 0) BM r.val p.window p.complement p.offset p.size key.val)).map
        (fun key => (score BM BD (s.pids 0) p xs r key, w key))) := by
    rw [← List.map_coe_finRange_eq_range, List.filter_map, List.map_map]
    apply List.map_congr_left
    intro key _
    exact Prod.ext (score_pinned BM BN BD p s xs hp r key) (hv key)
  change (if _ then RealFormula.summarize _ _ else _) = _
  erw [hm]
  rfl

theorem carry_pinned (BM BN BD : Nat) (p : Parameters) (s : BlockState)
    (xs : Inputs BM BD p.layout.kvContext) (hp : Pins BM BD p s xs) :
    RealExecution.finalCarry BM BN BD p s = carry BM BN BD (s.pids 0) p xs := by
  apply RealLoopFormula.Carry.ext
  · funext r
    exact congrArg RealFormula.RowState.center (row_pinned BM BN BD p s xs hp r 0 (fun _ => 0) (fun _ => 0) (fun _ => rfl))
  · funext r
    exact congrArg RealFormula.RowState.denominator (row_pinned BM BN BD p s xs hp r 0 (fun _ => 0) (fun _ => 0) (fun _ => rfl))
  · funext i
    have h := row_pinned BM BN BD p s xs hp i.1 ((seed BM BD p s).numerator i)
      (fun key => RealLoopContext.valueInput s (loopParameters BM BN BD p s) key i.2.1)
      (fun key => xs.value key i.2.1) (fun key => value_pinned BM BN BD p s xs hp key i.2.1)
    change (RealLoopContext.fullRow _ _ _ _ _ _).numerator = _
    simpa only [seed_pinned BM BD p s xs hp] using congrArg RealFormula.RowState.numerator h

/-- All three mathematical outputs, without memory reads or execution in the
formula. When END is enabled, L is preserved rather than assigned a value. -/
def center (BM BN BD finish start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext)
    (r : Fin BM) : ℝ :=
  (RealFormula.writeback (decide (finish ≠ 0))
    (RealLoopFormula.rowSeed (carry BM BN BD start p xs) r 0)).maxValue

def output (BM BN BD finish start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext)
    (i : TileIndex [BM, BD]) : ℝ :=
  let c := carry BM BN BD start p xs
  if finish ≠ 0 then c.numerator i / c.denominator i.1 else c.numerator i

theorem center_pinned (BM BN BD finish : Nat) (p : Parameters) (s : BlockState)
    (xs : Inputs BM BD p.layout.kvContext) (hp : Pins BM BD p s xs) (r : Fin BM) :
    RealEpilogue.centerValue finish (RealExecution.finalCarry BM BN BD p s) r =
      center BM BN BD finish (s.pids 0) p xs r := by
  rw [carry_pinned BM BN BD p s xs hp]
  rfl

theorem output_pinned (BM BN BD finish : Nat) (p : Parameters) (s : BlockState)
    (xs : Inputs BM BD p.layout.kvContext) (hp : Pins BM BD p s xs) (i : TileIndex [BM, BD]) :
    RealEpilogue.outputValue finish (RealExecution.finalCarry BM BN BD p s) i =
      output BM BN BD finish (s.pids 0) p xs i := by
  rw [carry_pinned BM BN BD p s xs hp]
  simp [RealEpilogue.outputValue, output, RealFormula.writeback, RealLoopFormula.rowSeed]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealInputFormula
