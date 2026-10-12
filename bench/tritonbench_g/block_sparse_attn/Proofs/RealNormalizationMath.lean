import bench.tritonbench_g.block_sparse_attn.Proofs.RealNormalization
import bench.tritonbench_g.block_sparse_attn.Proofs.RealAccumulationSource
import bench.tritonbench_g.block_sparse_attn.Proofs.CSRMath

/-! Connect normalization's executed row formulas to the independent CSR
attention recurrence. This is the arithmetic part of the loop invariant.
The score producers, value loads, loop induction and stores remain distinct
execution obligations; none is supplied as a completed-kernel assumption. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealNormalizationMath
open VeriTile.Triton RealNormalizationValues
open scoped BigOperators
noncomputable section

/-- Carry after k previous CSR visits, with two independent value-channel blocks. -/
def inputs {BM BN BD : Nat} (score : Nat → Fin BM → Fin BN → WithBot ℝ)
    (value secondValue : Nat → Fin BN → Fin BD → ℝ) (k : Nat) : Inputs BM BN BD where
  score := score k
  maximum := fun r => StreamingMath.maximum (fun n => score n r) k
  normalizer := fun r => StreamingMath.normalizer (fun n => score n r) k
  accumulator := fun r d => StreamingMath.accumulated (fun n => score n r) (fun n j => value n j d) k
  secondAccumulator := fun r d =>
    StreamingMath.accumulated (fun n => score n r) (fun n j => secondValue n j d) k

theorem maximum_next {BM BN BD : Nat} (score : Nat → Fin BM → Fin BN → WithBot ℝ)
    (value secondValue : Nat → Fin BN → Fin BD → ℝ) (k : Nat) (r : Fin BM) :
    (inputs score value secondValue k).nextMax r = StreamingMath.maximum (fun n => score n r) (k+1) := rfl

theorem normalizer_next {BM BN BD : Nat} (score : Nat → Fin BM → Fin BN → WithBot ℝ)
    (value secondValue : Nat → Fin BN → Fin BD → ℝ) (k : Nat) (r : Fin BM) :
    (inputs score value secondValue k).nextNormalizer r =
      StreamingMath.normalizer (fun n => score n r) (k+1) :=
  StreamingMath.block_normalizer (fun n => score n r) k

theorem accumulator_next {BM BN BD : Nat} (score : Nat → Fin BM → Fin BN → WithBot ℝ)
    (value secondValue : Nat → Fin BN → Fin BD → ℝ) (k : Nat) (r : Fin BM) (d : Fin BD) :
    (inputs score value secondValue k).scaledAccumulator r d +
      (∑ j : Fin BN, (inputs score value secondValue k).scaledProbability r j * value k j d) =
      StreamingMath.accumulated (fun n => score n r) (fun n j => value n j d) (k+1) := rfl

theorem second_accumulator_next {BM BN BD : Nat} (score : Nat → Fin BM → Fin BN → WithBot ℝ)
    (value secondValue : Nat → Fin BN → Fin BD → ℝ) (k : Nat) (r : Fin BM) (d : Fin BD) :
    (inputs score value secondValue k).scaledSecond r d +
      (∑ j : Fin BN, (inputs score value secondValue k).scaledProbability r j * secondValue k j d) =
      StreamingMath.accumulated (fun n => score n r) (fun n j => secondValue n j d) (k+1) := rfl

theorem matrix_dot {BM BN BD : Nat} (p : Fin BM → Fin BN → ℝ) (v : Fin BN → Fin BD → ℝ) :
    Tile.dot [] (matrix p) (matrix v) = matrix (fun r d => ∑ j : Fin BN, p r j * v j d) := by
  apply Tile.ext
  rintro ⟨r, d, ⟨⟩⟩
  change (∑ j : Fin BN, ((p r j * v j d : ℝ) : WithBot ℝ)) =
    ((∑ j : Fin BN, p r j * v j d : ℝ) : WithBot ℝ)
  exact (WithBot.coe_sum _ _).symm

theorem matrix_add {BM BD : Nat} (a b : Fin BM → Fin BD → ℝ) :
    Tile.bop NumericDType.real.add (.consSame (.consSame .nil)) (matrix a) (matrix b) =
      matrix (fun r d => a r d + b r d) := rfl

/-- Once a V load has supplied its actual tile, the source's acc += dot(p,v)
executes to the next independent row recurrence. This helper is used for both
channel names; no scalar surrogate replaces the matrix dot. -/
theorem value_dot_executes {BM BN BD : Nat} (s : BlockState) (name : RegName)
    (p : Fin BM → Fin BN → ℝ) (v : Fin BN → Fin BD → ℝ) (a : Fin BM → Fin BD → ℝ)
    (hp : s.regs .real [BM, BN] "p" = some (matrix p))
    (hv : s.regs .real [BN, BD] "v" = some (matrix v))
    (ha : s.regs .real [BM, BD] name = some (matrix a)) :
    stepStmt (RealAccumulationSource.valueDot BM BN BD name) s =
      some (s.setReg name .real [BM, BD] (matrix (fun r d => a r d + ∑ j : Fin BN, p r j * v j d))) := by
  apply stepStmt_assign_eq_some
  simp [evalOp.eq_def, hp, hv, ha, bind, Option.bind, matrix_dot, matrix_add]

/-- The executed normalization outputs give the source's next l/m carry. -/
theorem carry_outputs {BM BN BD : Nat} (score : Nat → Fin BM → Fin BN → WithBot ℝ)
    (value secondValue : Nat → Fin BN → Fin BD → ℝ) (k blocks : Nat) (t : BlockState)
    (h : RealNormalization.Outputs (inputs score value secondValue k) blocks t) :
    t.regs .real [BM] "m_i_new" = some (extendedRow (fun r => StreamingMath.maximum (fun n => score n r) (k+1))) ∧
    t.regs .real [BM] "l_i_new" = some (row (fun r => StreamingMath.normalizer (fun n => score n r) (k+1))) := by
  constructor
  · simpa only [funext (maximum_next score value secondValue k)] using h.maximum
  · simpa only [funext (normalizer_next score value secondValue k)] using h.normalizer

/-- Independent initial-array formula after a final visit. A zero normalizer
is covered by the real arithmetic model rather than ruled out as a premise. -/
theorem update_eq_formula {BM BN BD : Nat} (score : Nat → Fin BM → Fin BN → WithBot ℝ)
    (value secondValue : Nat → Fin BN → Fin BD → ℝ) (k : Nat) (r : Fin BM) (d : Fin BD) :
    (inputs score value secondValue k).scaledAccumulator r d +
      (∑ j : Fin BN, (inputs score value secondValue k).scaledProbability r j * value k j d) =
      StreamingMath.moment (fun n => score n r) (fun n j => value n j d) (k+1) /
        StreamingMath.mass (fun n => score n r) (k+1) := by
  rw [accumulator_next, StreamingMath.accumulated_eq_formula]


/-- Combine actual normalization readbacks with the checked first value-dot
assignment. The remaining premise is the V tile established by its source load. -/
theorem first_dot_next {BM BN BD : Nat} (t : BlockState)
    (score : Nat → Fin BM → Fin BN → WithBot ℝ)
    (value secondValue : Nat → Fin BN → Fin BD → ℝ) (k blocks : Nat)
    (h : RealNormalization.Outputs (inputs score value secondValue k) blocks t)
    (hv : t.regs .real [BN, BD] "v" = some (matrix (value k))) :
    stepStmt (RealAccumulationSource.valueDot BM BN BD "acc") t =
      some (t.setReg "acc" .real [BM, BD]
        (matrix (fun r d => StreamingMath.accumulated (fun n => score n r)
          (fun n j => value n j d) (k+1)))) := by
  have he := funext (fun r => funext (accumulator_next score value secondValue k r))
  simpa only [he] using value_dot_executes t "acc"
    (inputs score value secondValue k).scaledProbability (value k)
    (inputs score value secondValue k).scaledAccumulator h.probability hv h.accumulator

theorem second_dot_next {BM BN BD : Nat} (t : BlockState)
    (score : Nat → Fin BM → Fin BN → WithBot ℝ)
    (value secondValue : Nat → Fin BN → Fin BD → ℝ) (k : Nat)
    (hp : t.regs .real [BM, BN] "p" = some (matrix (inputs score value secondValue k).scaledProbability))
    (ha : t.regs .real [BM, BD] "acc2" = some (matrix (inputs score value secondValue k).scaledSecond))
    (hv : t.regs .real [BN, BD] "v" = some (matrix (secondValue k))) :
    stepStmt (RealAccumulationSource.valueDot BM BN BD "acc2") t =
      some (t.setReg "acc2" .real [BM, BD]
        (matrix (fun r d => StreamingMath.accumulated (fun n => score n r)
          (fun n j => secondValue n j d) (k+1)))) := by
  have he := funext (fun r => funext (second_accumulator_next score value secondValue k r))
  simpa only [he] using value_dot_executes t "acc2"
    (inputs score value secondValue k).scaledProbability (secondValue k)
    (inputs score value secondValue k).scaledSecond hp hv ha

/-- The two source assignments commit the computed l/m for the next signed visit. -/
theorem commit_executes {BM : Nat} (t : BlockState) (m : Fin BM → WithBot ℝ) (l : Fin BM → ℝ)
    (hm : t.regs .real [BM] "m_i_new" = some (extendedRow m))
    (hl : t.regs .real [BM] "l_i_new" = some (row l)) :
    stepStmts (RealAccumulationSource.commit BM) t =
      some ((t.setReg "l_i" .real [BM] (row l)).setReg "m_i" .real [BM] (extendedRow m)) := by
  simp [RealAccumulationSource.commit, stepStmts, stepStmt, evalOp.eq_def, hm, hl, bind, Option.bind]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealNormalizationMath
