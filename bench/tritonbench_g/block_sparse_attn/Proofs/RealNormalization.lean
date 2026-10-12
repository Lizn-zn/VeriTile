import bench.tritonbench_g.block_sparse_attn.Proofs.RealNormalizationSource
import bench.tritonbench_g.block_sparse_attn.Proofs.RealNormalizationValues
import VeriTile.Triton.Semantics.Step

/-! Execution of the checked normalization segment. Entry premises describe
qk and the old carry, to be supplied by the preceding source execution.
The segment writes only its listed registers and preserves the complete memory,
program IDs, grid dimensions and undefined-input interpretation. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealNormalization
open VeriTile.Triton RealNormalizationValues
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

/-- The register state produced by the actual assignment order. -/
def result {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD) (blocks : Nat) : BlockState :=
  let s := s.setReg "m_ij" .real [BM] (extendedRow x.blockMax)
  let s := s.setReg "p" .real [BM, BN] (matrix x.probability)
  let s := s.setReg "l_ij" .real [BM] (row x.blockNormalizer)
  let s := s.setReg "m_i_new" .real [BM] (extendedRow x.nextMax)
  let s := s.setReg "alpha" .real [BM] (row x.alpha)
  let s := s.setReg "beta" .real [BM] (row x.beta)
  let s := s.setReg "l_i_new" .real [BM] (row x.nextNormalizer)
  let s := s.setReg "p_scale" .real [BM] (row x.probabilityScale)
  let s := s.setReg "p" .real [BM, BN] (matrix x.scaledProbability)
  let s := s.setReg "acc_scale" .real [BM] (row x.accumulatorScale)
  let s := s.setReg "acc" .real [BM, BD] (matrix x.scaledAccumulator)
  let s := if 2 ≤ blocks then s.setReg "acc2" .real [BM, BD] (matrix x.scaledSecond) else s
  s.setReg "p" .real [BM, BN] (matrix x.scaledProbability)

theorem executes {BM BN BD : Nat} (hN : 0 < BN) (s : BlockState)
    (x : Inputs BM BN BD) (blocks : Nat)
    (hq : s.regs .real [BM, BN] "qk" = some (extendedMatrix x.score))
    (hm : s.regs .real [BM] "m_i" = some (extendedRow x.maximum))
    (hl : s.regs .real [BM] "l_i" = some (row x.normalizer))
    (ha : s.regs .real [BM, BD] "acc" = some (matrix x.accumulator))
    (ha2 : 2 ≤ blocks → s.regs .real [BM, BD] "acc2" = some (matrix x.secondAccumulator)) :
    stepStmts (RealNormalizationSource.code BM BN BD blocks) s = some (result s x blocks) := by
  have hmax : Tile.reduceMaxDrop (shape := [BM, BN]) (1 : Fin 2) (extendedMatrix x.score) =
      some (extendedRow x.blockMax) := reduce_max hN x.score
  have hprob : Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub (.consSame (.consR .nil))
        (extendedMatrix x.score) (Tile.expandDim (shape := [BM]) 1 (extendedRow x.blockMax))) =
      matrix x.probability := shifted_matrix x.score x.blockMax
  by_cases hb : 2 ≤ blocks
  all_goals
    simp [RealNormalizationSource.code, stepStmts, stepStmt, evalOp.eq_def, bind, Option.bind,
      hq, hm, hl, ha, ha2, hb, hmax, hprob, reduce_max hN, reduce_sum, shifted_matrix,
      shifted_row, maximum_row, row_add, row_mul, row_div, matrix_scale, result,
      Inputs.blockMax, Inputs.probability, Inputs.blockNormalizer, Inputs.nextMax,
      Inputs.alpha, Inputs.beta, Inputs.nextNormalizer, Inputs.probabilityScale,
      Inputs.scaledProbability, Inputs.accumulatorScale, Inputs.scaledAccumulator,
      Inputs.scaledSecond]
  all_goals
    repeat' first
      | erw [hprob]
      | erw [reduce_sum]
      | erw [matrix_scale]
      | erw [row_add]
      | erw [row_mul]
      | erw [row_div]
    rfl

def writes : List RegName :=
  ["m_ij", "p", "l_ij", "m_i_new", "alpha", "beta", "l_i_new", "p_scale", "acc_scale", "acc", "acc2"]

/-- Restoring only the register file recovers every other initial state field. -/
theorem state_frame {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD) (blocks : Nat) :
    { result s x blocks with regs := s.regs } = s := by
  by_cases hb : 2 ≤ blocks <;> simp [result, hb, BlockState.setReg]

theorem register_frame {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD) (blocks : Nat)
    (dtype : TileDType) (shape : TileShape) (name : RegName) (hn : name ∉ writes) :
    (result s x blocks).regs dtype shape name = s.regs dtype shape name := by
  simp only [writes, List.mem_cons, List.mem_singleton, not_or] at hn
  by_cases hb : 2 ≤ blocks <;> simp [result, hb, hn]


structure Outputs {BM BN BD : Nat} (x : Inputs BM BN BD) (blocks : Nat) (s : BlockState) : Prop where
  probability : s.regs .real [BM, BN] "p" = some (matrix x.scaledProbability)
  maximum : s.regs .real [BM] "m_i_new" = some (extendedRow x.nextMax)
  normalizer : s.regs .real [BM] "l_i_new" = some (row x.nextNormalizer)
  accumulator : s.regs .real [BM, BD] "acc" = some (matrix x.scaledAccumulator)
  secondAccumulator : 2 ≤ blocks →
    s.regs .real [BM, BD] "acc2" = some (matrix x.scaledSecond)

theorem outputs {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD) (blocks : Nat) :
    Outputs x blocks (result s x blocks) := by
  constructor <;> by_cases hb : 2 ≤ blocks <;> simp [result, hb]

/-- The absent second-channel arm does not need or alter an acc2 binding. -/
theorem second_disabled {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD)
    (blocks : Nat) (hb : ¬ 2 ≤ blocks) (dtype : TileDType) (shape : TileShape) :
    (result s x blocks).regs dtype shape "acc2" = s.regs dtype shape "acc2" := by
  simp [result, hb]

/-- Execution of the segment selected from either complete source, with exact
row formulas and preservation of everything outside its register write set.
The surrounding loop must establish the entry register premises. -/
theorem source_executes (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (hN : 0 < BN) (s : BlockState) (x : Inputs BM BN BD)
    (hq : s.regs .real [BM, BN] "qk" = some (extendedMatrix x.score))
    (hm : s.regs .real [BM] "m_i" = some (extendedRow x.maximum))
    (hl : s.regs .real [BM] "l_i" = some (row x.normalizer))
    (ha : s.regs .real [BM, BD] "acc" = some (matrix x.accumulator))
    (ha2 : 2 ≤ p.blocks → s.regs .real [BM, BD] "acc2" = some (matrix x.secondAccumulator)) :
    ∃ t, stepStmts (RealProgram.realPart (RealProgram.normalization swapped BM BN BD p)) s = some t ∧
      Outputs x p.blocks t ∧ { t with regs := s.regs } = s ∧
      (∀ dtype shape name, name ∉ writes → t.regs dtype shape name = s.regs dtype shape name) := by
  refine ⟨result s x p.blocks, ?_, outputs s x p.blocks, state_frame s x p.blocks, ?_⟩
  · rw [RealNormalizationSource.matches_source]
    exact executes hN s x p.blocks hq hm hl ha ha2
  · exact register_frame s x p.blocks

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealNormalization
