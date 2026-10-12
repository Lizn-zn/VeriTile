import bench.tritonbench_g.attention_score.Proofs.RealArithmetic
import bench.tritonbench_g.attention_score.Proofs.RealTiles

/-! The actual nine-statement arithmetic fragment updates o by the independent
column-score formula. The loaded-register premises are local invariants to be
established by the full memory/range proof; they are not a public kernel input
contract. Both source variants use the same result, without FP assumptions. -/

namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealArithmeticOutput
open VeriTile.Triton RealFormula
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem output_of_execution (s t : BlockState) {D : Nat} (xs : Inputs D)
    (B N NK startM startN : Nat) (offset size : Int) (scale : ℝ)
    (window complement evenM evenN : Bool) (old : Fin B → ℝ)
    (hq : s.regs .real [B,D] "q" = some (RealTiles.query xs B N startM evenM))
    (hk : s.regs .real [D,B] "k" = some (RealTiles.key xs B NK startN evenN))
    (hm : s.regs .real [B] "m" = some (RealTiles.normalizer xs B startM))
    (ho : s.regs .real [B] "o" = some (Tile.vec (fun k => some (old k))))
    (hs : s.regs .real [] "qk_scale" = some (Tile.scalar (some (scale * 1.4426950408889634))))
    (hsm : s.regs .nat [] "start_m" = some (Tile.scalar startM))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar startN))
    (he : stepStmts (RealArithmetic.body B D N offset size window complement evenN Bool.false) s = some t) :
    ∀ k : Fin B, (t.regs .real [B] "o").map (fun v => v.data (k,PUnit.unit)) =
      some (some (old k + ∑ r : Fin B,
        weight xs scale N NK window complement evenM evenN offset size
          (startM+r.val) (startN*B+k.val))) := by
  have hcolumns (j : Fin B) := RealTiles.column_sum xs scale B N NK startM startN window complement evenM evenN offset size j
  cases window <;> cases complement <;> cases evenN
  all_goals
    simp [RealArithmetic.body, stepStmts, stepStmt, evalOp.eq_def, hq, hk, hm, ho,
      hs, hsm, hsn, RealExpressions.distanceExpr, bind, Option.bind] at he
    subst t
    intro k
    have hc := hcolumns k
    simp only [BlockState.setReg_same, Option.map_some]
    change some (NumericDType.real.add (some (old k)) _) = _
    congr 1
    rw [show NumericDType.real.add = WithBot.realAdd from rfl]
    have ha (value : ℝ) : WithBot.realAdd (some (old k)) (some value) = some (old k + value) := rfl
    rw [← ha, ← hc]
    congr 1
  all_goals
    apply congrArg (fun p : Tile .real [B,B] => (Tile.reduceSumDrop ⟨0, by simp⟩ p).data (k,PUnit.unit))
    apply Tile.ext
    rintro ⟨r, j, ⟨⟩⟩
    have hdist : (r.val : Int) - j.val + startM - (startN : Int) * B + offset =
        (startM : Int) + r.val - ((startN : Int) * B + j.val) + offset := by ring
    simp [RealTiles.contributions, RealTiles.windowed, RealTiles.rawProbability,
      RealTiles.queryMask, RealTiles.windowMask, RealTiles.zero,
      Tile.select, Tile.bop_data, Tile.cop, Tile.remap, Tile.expandDim_data,
      TileShape.dropInsertedIndex, Tile.scalar, Tile.vec, NumericDType.add,
      NumericDType.sub, NumericDType.mul, ComparableDType.ge, ComparableDType.lt,
      RealFormula.distance, hdist, Nat.add_comm, -Tile.scalar_eta]
  all_goals simp [Int.add_comm]

/-- Both actual arithmetic fragments execute and update the running output
by the independent column formula. Loaded-register premises will be established
by the memory/loop invariant; no output equation is assumed. -/
theorem correct (s : BlockState) {D : Nat} (xs : Inputs D)
    (B N NK startM startN : Nat) (offset size : Int) (scale : ℝ)
    (window complement evenM evenN candidate : Bool) (old : Fin B → ℝ)
    (hq : s.regs .real [B,D] "q" = some (RealTiles.query xs B N startM evenM))
    (hk : s.regs .real [D,B] "k" = some (RealTiles.key xs B NK startN evenN))
    (hm : s.regs .real [B] "m" = some (RealTiles.normalizer xs B startM))
    (ho : s.regs .real [B] "o" = some (Tile.vec (fun k => some (old k))))
    (hs : s.regs .real [] "qk_scale" = some (Tile.scalar (some (scale * 1.4426950408889634))))
    (hsm : s.regs .nat [] "start_m" = some (Tile.scalar startM))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar startN)) :
    ∃ t, stepStmts (RealArithmetic.body B D N offset size window complement evenN candidate) s = some t ∧
      t.mem = s.mem ∧ t.pids = s.pids ∧ t.undef = s.undef ∧
      ∀ k : Fin B, (t.regs .real [B] "o").map (fun v => v.data (k,PUnit.unit)) =
      some (some (old k + ∑ r : Fin B,
        weight xs scale N NK window complement evenM evenN offset size
          (startM+r.val) (startN*B+k.val))) := by
  obtain ⟨t, he, hmem, hpids, hundef⟩ := RealArithmetic.executes s B D N startM startN
    offset size (scale * 1.4426950408889634) window complement evenN
    (RealTiles.query xs B N startM evenM) (RealTiles.key xs B NK startN evenN)
    (RealTiles.normalizer xs B startM) (Tile.vec (fun k => some (old k)))
    hq hk hm ho hs hsm hsn
  refine ⟨t, ?_, hmem, hpids, hundef, ?_⟩
  · cases candidate
    · exact he
    · rw [RealArithmetic.candidate_execution]
      exact he
  · exact output_of_execution s t xs B N NK startM startN offset size scale
      window complement evenM evenN old hq hk hm ho hs hsm hsn he

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionScore.RealArithmeticOutput
