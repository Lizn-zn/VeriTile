import bench.tritonbench_g.block_sparse_attn.Proofs.RealValueFormula
import bench.tritonbench_g.block_sparse_attn.Proofs.RealValueSafety

/-! Compose the actual loop suffix from row normalization through both V
loads, matrix accumulation and carry writes. The score-producing prefix and
outer signed-loop invariant are still separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealLoopTail
open VeriTile.Triton RealNormalizationValues
noncomputable section

def result {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context blocks : Nat) (even : Bool) : BlockState :=
  RealAccumulation.result (RealNormalization.result s x blocks) x ptrs start stride context blocks even

theorem executes {BM BN BD : Nat} (hN : 0 < BN) (s : BlockState) (x : Inputs BM BN BD)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context blocks : Nat) (even : Bool)
    (hp : s.regs .ptr [BN, BD] "v_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hq : s.regs .real [BM, BN] "qk" = some (extendedMatrix x.score))
    (hm : s.regs .real [BM] "m_i" = some (extendedRow x.maximum))
    (hl : s.regs .real [BM] "l_i" = some (row x.normalizer))
    (ha : s.regs .real [BM, BD] "acc" = some (matrix x.accumulator))
    (ha2 : 2 ≤ blocks → s.regs .real [BM, BD] "acc2" = some (matrix x.secondAccumulator)) :
    stepStmts (RealNormalizationSource.code BM BN BD blocks ++
      RealAccumulation.code BM BN BD stride context blocks even) s =
      some (result s x ptrs start stride context blocks even) := by
  rw [stepStmts.append_some (RealNormalization.executes hN s x blocks hq hm hl ha ha2)]
  apply RealAccumulation.executes
  · rw [RealNormalization.register_frame s x blocks .ptr [BN, BD] "v_ptrs" (by simp [RealNormalization.writes])]
    exact hp
  · rw [RealNormalization.register_frame s x blocks .int [] "start_n" (by simp [RealNormalization.writes])]
    exact hs
  · rw [RealNormalization.register_frame s x blocks .nat [BN] "offs_n" (by simp [RealNormalization.writes])]
    exact hn
  · exact RealNormalization.outputs s x blocks

theorem state_frame {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context blocks : Nat) (even : Bool) :
    { result s x ptrs start stride context blocks even with regs := s.regs } = s := by
  have h := RealAccumulation.state_frame (RealNormalization.result s x blocks) x ptrs start stride context blocks even
  have hr := congrArg (fun t => { t with regs := s.regs }) h
  exact hr.trans (RealNormalization.state_frame s x blocks)

theorem register_frame {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context blocks : Nat) (even : Bool)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hn : name ∉ RealNormalization.writes ++ RealAccumulation.writes) :
    (result s x ptrs start stride context blocks even).regs dtype shape name = s.regs dtype shape name := by
  simp only [List.mem_append, not_or] at hn
  rw [result, RealAccumulation.register_frame _ _ _ _ _ _ _ _ _ _ _ hn.2,
    RealNormalization.register_frame _ _ _ _ _ _ hn.1]

theorem loaded_normalization {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context blocks : Nat) (even second : Bool) :
    RealValueLoad.loaded (RealNormalization.result s x blocks) ptrs start stride context even second =
      RealValueLoad.loaded s ptrs start stride context even second := by
  by_cases hb : 2 ≤ blocks <;> simp [RealNormalization.result, hb]

theorem outputs {BM BN BD : Nat} (s : BlockState) (x : Inputs BM BN BD)
    (ptrs : Tile .ptr [BN, BD]) (start : Int) (stride context blocks : Nat) (even : Bool) :
    RealAccumulation.Outputs s x ptrs start stride context blocks even
      (result s x ptrs start stride context blocks even) := by
  have h := RealAccumulation.outputs (RealNormalization.result s x blocks) x ptrs start stride context blocks even
  exact ⟨h.maximum, h.normalizer, by simpa only [loaded_normalization] using h.accumulator,
    fun hb => by simpa only [loaded_normalization] using h.secondAccumulator hb⟩

theorem source_executes (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (hN : 0 < BN) (s : BlockState) (x : Inputs BM BN BD) (ptrs : Tile .ptr [BN, BD]) (start : Int)
    (hp : s.regs .ptr [BN, BD] "v_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hq : s.regs .real [BM, BN] "qk" = some (extendedMatrix x.score))
    (hm : s.regs .real [BM] "m_i" = some (extendedRow x.maximum))
    (hl : s.regs .real [BM] "l_i" = some (row x.normalizer))
    (ha : s.regs .real [BM, BD] "acc" = some (matrix x.accumulator))
    (ha2 : 2 ≤ p.blocks → s.regs .real [BM, BD] "acc2" = some (matrix x.secondAccumulator)) :
    ∃ t, stepStmts (RealProgram.realPart (RealProgram.normalization swapped BM BN BD p) ++
        RealProgram.realPart (RealProgram.accumulation swapped BM BN BD p)) s = some t ∧
      RealAccumulation.Outputs s x ptrs start p.vRow p.context p.blocks p.evenN t ∧
      { t with regs := s.regs } = s ∧
      (∀ dtype shape name, name ∉ RealNormalization.writes ++ RealAccumulation.writes →
        t.regs dtype shape name = s.regs dtype shape name) := by
  refine ⟨result s x ptrs start p.vRow p.context p.blocks p.evenN, ?_, outputs ..,
    state_frame .., register_frame s x ptrs start p.vRow p.context p.blocks p.evenN⟩
  rw [RealNormalizationSource.matches_source, RealAccumulation.matches_source]
  exact executes hN s x ptrs start p.vRow p.context p.blocks p.evenN hp hs hn hq hm hl ha ha2

/-- Observe either actually written accumulator on a valid query row. The
result is the independent initial-array weighted sum through this CSR visit;
only active V addresses need input pinning. The execution theorem above
connects this same result state to the actual source suffix. -/
theorem result_formula (schedule : CSRMath.Schedule) (xs : CSRMath.Inputs)
    (BM BN BD blocks context visit stride : Nat) (scale : ℝ) (even second : Bool)
    (s : BlockState) (ptrs : Tile .ptr [BN, BD]) (r : Fin BM) (d : Fin BD)
    (hsecond : second = Bool.true → 2 ≤ blocks)
    (hq : schedule.queryBlock * BM + r.val < context)
    (hv : ∀ j d, RealValueLoad.active (RealValueFormula.keyStart schedule xs BN visit) context even j →
      s.readMem (RealValueLoad.address ptrs (RealValueFormula.keyStart schedule xs BN visit) stride second j d).1
        (RealValueLoad.address ptrs (RealValueFormula.keyStart schedule xs BN visit) stride second j d).2 =
      RealValueFormula.value schedule xs BN BD second visit j d) :
    let x := RealValueFormula.inputs schedule xs BM BN BD blocks scale visit
    let t := result s x ptrs (RealValueFormula.keyStart schedule xs BN visit) stride context blocks even
    ∃ a : Tile .real [BM, BD], t.regs .real [BM, BD] (if second then "acc2" else "acc") = some a ∧
      a.data (r, d, PUnit.unit) = some
        (StreamingMath.moment
          (CSRMath.score schedule xs BN BD blocks (schedule.queryBlock * BM + r.val) scale)
          (fun n j => RealValueFormula.value schedule xs BN BD second n j d) (visit + 1) /
        StreamingMath.mass
          (CSRMath.score schedule xs BN BD blocks (schedule.queryBlock * BM + r.val) scale) (visit + 1)) := by
  dsimp only
  let x := RealValueFormula.inputs schedule xs BM BN BD blocks scale visit
  have ho := outputs s x ptrs (RealValueFormula.keyStart schedule xs BN visit) stride context blocks even
  cases second
  · refine ⟨_, ho.accumulator, ?_⟩
    apply congrArg some
    exact RealValueFormula.update_eq_formula schedule xs BM BN BD blocks context visit stride scale
      even Bool.false s ptrs r d hq hv
  · refine ⟨_, ho.secondAccumulator (hsecond rfl), ?_⟩
    apply congrArg some
    exact RealValueFormula.update_eq_formula schedule xs BM BN BD blocks context visit stride scale
      even Bool.true s ptrs r d hq hv

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealLoopTail
