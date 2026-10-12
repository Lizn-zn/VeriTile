import bench.tritonbench_g.block_sparse_attn.Proofs.RealValueLoad
import bench.tritonbench_g.block_sparse_attn.Proofs.RealNormalizationMath

/-! Actual V loads, both matrix dots and carry writes after normalization.
Values come from memory, including arbitrary masked-load values. The earlier
source must supply pointer/index registers and normalization outputs; no V
register or execution of the complete body is assumed. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealAccumulation
open VeriTile.Triton RealNormalizationValues
open scoped BigOperators
noncomputable section

def update {BM BN BD : Nat} (probability : Fin BM → Fin BN → ℝ)
    (old : Fin BM → Fin BD → ℝ) (value : Fin BN → Fin BD → ℝ) : Fin BM → Fin BD → ℝ :=
  fun r d => old r d + ∑ j : Fin BN, probability r j * value j d

def loadDot (BM BN BD stride context : Nat) (even second : Bool) : List Stmt :=
  RealValueSource.code BN BD stride context even second ++
    [RealAccumulationSource.valueDot BM BN BD (if second then "acc2" else "acc")]

def dotResult {BM BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BN, BD])
    (start : Int) (stride context : Nat) (even second : Bool)
    (probability : Fin BM → Fin BN → ℝ) (old : Fin BM → Fin BD → ℝ) : BlockState :=
  let value := RealValueLoad.loaded s ptrs start stride context even second
  (s.setReg "v" .real [BN, BD] (matrix value)).setReg
    (if second then "acc2" else "acc") .real [BM, BD] (matrix (update probability old value))

theorem load_dot_executes {BM BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BN, BD])
    (start : Int) (stride context : Nat) (even second : Bool)
    (probability : Fin BM → Fin BN → ℝ) (old : Fin BM → Fin BD → ℝ)
    (hp : s.regs .ptr [BN, BD] "v_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hpp : s.regs .real [BM, BN] "p" = some (matrix probability))
    (ha : s.regs .real [BM, BD] (if second then "acc2" else "acc") = some (matrix old)) :
    stepStmts (loadDot BM BN BD stride context even second) s =
      some (dotResult s ptrs start stride context even second probability old) := by
  unfold loadDot
  rw [stepStmts.append_some (RealValueLoad.executes s ptrs start stride context even second hp hs hn)]
  have hd := RealNormalizationMath.value_dot_executes
    (s.setReg "v" .real [BN, BD] (matrix (RealValueLoad.loaded s ptrs start stride context even second)))
    (if second then "acc2" else "acc") probability
    (RealValueLoad.loaded s ptrs start stride context even second) old
    (by simpa using hpp) (by simp) (by cases second <;> simpa using ha)
  rw [stepStmts.cons_some hd]
  rw [stepStmts.nil]
  rfl

def code (BM BN BD stride context blocks : Nat) (even : Bool) : List Stmt :=
  loadDot BM BN BD stride context even Bool.false ++
    [.ifThen (.constBool (decide (2 ≤ blocks)))
      (loadDot BM BN BD stride context even Bool.true)] ++
    RealAccumulationSource.commit BM

theorem matches_source (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.accumulation swapped BM BN BD p) =
      code BM BN BD p.vRow p.context p.blocks p.evenN := by
  rw [RealAccumulationSource.matches_source, RealValueSource.first_matches,
    RealValueSource.second_matches]
  simp [code, loadDot, List.append_assoc]

def result {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context blocks : Nat) (even : Bool) : BlockState :=
  let t := dotResult s ptrs start stride context even Bool.false x.scaledProbability x.scaledAccumulator
  let t := if 2 ≤ blocks then
      dotResult t ptrs start stride context even Bool.true x.scaledProbability x.scaledSecond else t
  (t.setReg "l_i" .real [BM] (row x.nextNormalizer)).setReg
    "m_i" .real [BM] (extendedRow x.nextMax)

theorem executes {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context blocks : Nat) (even : Bool)
    (hp : s.regs .ptr [BN, BD] "v_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hx : RealNormalization.Outputs x blocks s) :
    stepStmts (code BM BN BD stride context blocks even) s =
      some (result s x ptrs start stride context blocks even) := by
  have hfirst := load_dot_executes s ptrs start stride context even Bool.false
    x.scaledProbability x.scaledAccumulator hp hs hn hx.probability hx.accumulator
  unfold code
  rw [List.append_assoc, stepStmts.append_some hfirst]
  let t := dotResult s ptrs start stride context even Bool.false x.scaledProbability x.scaledAccumulator
  change stepStmts ([.ifThen (.constBool (decide (2 ≤ blocks)))
      (loadDot BM BN BD stride context even Bool.true)] ++ RealAccumulationSource.commit BM) t = _
  by_cases hb : 2 ≤ blocks
  · have hsecond := load_dot_executes t ptrs start stride context even Bool.true
      x.scaledProbability x.scaledSecond
      (by simpa [t, dotResult] using hp) (by simpa [t, dotResult] using hs)
      (by simpa [t, dotResult] using hn) (by simpa [t, dotResult] using hx.probability)
      (by simpa [t, dotResult] using hx.secondAccumulator hb)
    have hi : stepStmt (.ifThen (.constBool (decide (2 ≤ blocks)))
        (loadDot BM BN BD stride context even Bool.true)) t =
        some (dotResult t ptrs start stride context even Bool.true x.scaledProbability x.scaledSecond) := by
      simpa only [stepStmt, evalOp.eq_def, hb, decide_true, Tile.scalar_data] using hsecond
    rw [List.singleton_append, stepStmts.cons_some hi]
    have hc := RealNormalizationMath.commit_executes
      (dotResult t ptrs start stride context even Bool.true x.scaledProbability x.scaledSecond)
      x.nextMax x.nextNormalizer
      (by simpa [t, dotResult] using hx.maximum) (by simpa [t, dotResult] using hx.normalizer)
    simpa only [result, hb, ↓reduceIte, t] using hc
  · have hi : stepStmt (.ifThen (.constBool (decide (2 ≤ blocks)))
        (loadDot BM BN BD stride context even Bool.true)) t = some t := by
      simp [stepStmt, evalOp.eq_def, hb]
    rw [List.singleton_append, stepStmts.cons_some hi]
    have hc := RealNormalizationMath.commit_executes t x.nextMax x.nextNormalizer
      (by simpa [t, dotResult] using hx.maximum) (by simpa [t, dotResult] using hx.normalizer)
    simpa only [result, hb, ↓reduceIte, t] using hc

def writes : List RegName := ["v", "acc", "acc2", "l_i", "m_i"]

theorem state_frame {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context blocks : Nat) (even : Bool) :
    { result s x ptrs start stride context blocks even with regs := s.regs } = s := by
  by_cases hb : 2 ≤ blocks <;> simp [result, dotResult, hb, BlockState.setReg]

theorem register_frame {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context blocks : Nat) (even : Bool)
    (dtype : TileDType) (shape : TileShape) (name : RegName) (hn : name ∉ writes) :
    (result s x ptrs start stride context blocks even).regs dtype shape name = s.regs dtype shape name := by
  simp only [writes, List.mem_cons, not_or] at hn
  by_cases hb : 2 ≤ blocks <;> simp [result, dotResult, hb, hn]

structure Outputs {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context blocks : Nat) (even : Bool)
    (t : BlockState) : Prop where
  maximum : t.regs .real [BM] "m_i" = some (extendedRow x.nextMax)
  normalizer : t.regs .real [BM] "l_i" = some (row x.nextNormalizer)
  accumulator : t.regs .real [BM, BD] "acc" = some (matrix
    (update x.scaledProbability x.scaledAccumulator
      (RealValueLoad.loaded s ptrs start stride context even Bool.false)))
  secondAccumulator : 2 ≤ blocks → t.regs .real [BM, BD] "acc2" = some (matrix
    (update x.scaledProbability x.scaledSecond
      (RealValueLoad.loaded s ptrs start stride context even Bool.true)))

theorem outputs {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context blocks : Nat) (even : Bool) :
    Outputs s x ptrs start stride context blocks even (result s x ptrs start stride context blocks even) := by
  constructor <;> by_cases hb : 2 ≤ blocks <;> simp [result, dotResult, hb]

theorem second_disabled {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context blocks : Nat) (even : Bool)
    (hb : ¬ 2 ≤ blocks) (dtype : TileDType) (shape : TileShape) :
    (result s x ptrs start stride context blocks even).regs dtype shape "acc2" =
      s.regs dtype shape "acc2" := by
  simp [result, dotResult, hb]

/-- The complete accumulation slice from either source executes with no
preloaded V assumption and preserves memory and all unrelated state. -/
theorem source_executes (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (s : BlockState) (x : Inputs BM BN BD) (ptrs : Tile .ptr [BN, BD]) (start : Int)
    (hp : s.regs .ptr [BN, BD] "v_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hx : RealNormalization.Outputs x p.blocks s) :
    ∃ t, stepStmts (RealProgram.realPart (RealProgram.accumulation swapped BM BN BD p)) s = some t ∧
      Outputs s x ptrs start p.vRow p.context p.blocks p.evenN t ∧
      { t with regs := s.regs } = s ∧
      (∀ dtype shape name, name ∉ writes → t.regs dtype shape name = s.regs dtype shape name) := by
  refine ⟨result s x ptrs start p.vRow p.context p.blocks p.evenN, ?_,
    outputs .., state_frame ..,
    register_frame s x ptrs start p.vRow p.context p.blocks p.evenN⟩
  rw [matches_source]
  exact executes s x ptrs start p.vRow p.context p.blocks p.evenN hp hs hn hx

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealAccumulation
