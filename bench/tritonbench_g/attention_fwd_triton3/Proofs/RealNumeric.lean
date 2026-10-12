import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealBody
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealTiles

/-! Execution of numerical fragments from the checked source loop. Register
premises describe values established by preceding loads and loop steps; the
conclusions prove successful execution and independent real tile formulas. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealNumeric
open VeriTile.Triton
open RealTiles
open scoped BigOperators
noncomputable section

def scores (BM BN BD : Nat) (q : TileIndex [BM, BD] → ℝ)
    (k : TileIndex [BD, BN] → ℝ) (scale : ℝ) : TileIndex [BM, BN] → ℝ :=
  fun i => (∑ d : Fin BD, q (i.1, d, PUnit.unit) * k (d, i.2.1, PUnit.unit)) * scale

theorem eval_tail_mask (s : BlockState) (BM BN context : Nat) (startN : Int)
    (hn : s.regs .int [] "start_n" = some (Tile.scalar startN)) :
    evalOp (RealBody.tailMask BM BN context) s =
      some (⟨fun i => decide ((i.2.1.val : Int) + startN < context)⟩ : Tile .bool [BM, BN]) := by
  simp only [RealBody.tailMask, evalOp.eq_def, hn, bind, Option.bind]
  congr 1

theorem eval_zero_plus_dot (s : BlockState) (BM BN BD : Nat)
    (q : TileIndex [BM, BD] → ℝ) (k : TileIndex [BD, BN] → ℝ)
    (hq : s.regs .real [BM, BD] "q" = some (finite q))
    (hk : s.regs .real [BD, BN] "k" = some (finite k))
    (hz : s.regs .real [BM, BN] "qk" = some (finite (fun _ => 0))) :
    evalOp (.add .real (.consSame (.consSame .nil)) (.ref .real [BM, BN] "qk")
      (.dot (batch := []) (.ref .real [BM, BD] "q") (.ref .real [BD, BN] "k"))) s =
      some (finite (fun i : TileIndex [BM, BN] =>
        ∑ d : Fin BD, q (i.1, d, PUnit.unit) * k (d, i.2.1, PUnit.unit))) := by
  simp only [evalOp.eq_def, List.nil_append, hq, hk, hz, bind, Option.bind]
  erw [dot_finite BM BN BD q k]
  congr 1
  apply Tile.ext
  rintro ⟨row, key, ⟨⟩⟩
  simp [finite, Tile.bop_data, NumericDType.add]

theorem eval_scale (s : BlockState) (BM BN : Nat) (f : TileIndex [BM, BN] → ℝ) (scale : ℝ)
    (hqk : s.regs .real [BM, BN] "qk" = some (finite f))
    (hscale : s.regs .real [] "qk_scale" = some (Tile.scalar (some scale))) :
    evalOp (.mul .real .scalarR (.ref .real [BM, BN] "qk") (.ref .real [] "qk_scale")) s =
      some (finite (fun i => f i * scale)) := by
  simp only [evalOp_mul, evalOp_ref, hqk, hscale, bind, Option.bind]
  congr 1

/-- The three arithmetic assignments at the start of the checked score body
execute as the Q/K sum-product times the loaded scale. The final state also
keeps memory, block pointers and all registers other than qk unchanged. -/
theorem raw_scores_execute (s : BlockState) (BM BN BD context window complement even : Nat)
    (offset size : Int) (q : TileIndex [BM, BD] → ℝ) (k : TileIndex [BD, BN] → ℝ) (scale : ℝ)
    (hq : s.regs .real [BM, BD] "q" = some (finite q))
    (hk : s.regs .real [BD, BN] "k" = some (finite k))
    (hscale : s.regs .real [] "qk_scale" = some (Tile.scalar (some scale))) :
    ∃ s', stepStmts ((RealBody.scores BM BN BD context window complement even offset size).take 3) s =
        some s' ∧ s'.regs .real [BM, BN] "qk" = some (finite (scores BM BN BD q k scale)) ∧
      s'.mem = s.mem ∧ s'.pids = s.pids ∧ s'.undef = s.undef ∧
      (∀ dtype shape name, name ≠ "qk" → s'.regs dtype shape name = s.regs dtype shape name) := by
  have hz : evalOp (.full [BM, BN] (.const 0)) s = some (finite (fun _ => 0)) := by
    simp [evalOp_full, evalOp_const, finite]
  simp only [RealBody.scores, List.take_succ_cons, List.take_zero]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hz)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (eval_zero_plus_dot _ BM BN BD q k (by simp [hq]) (by simp [hk]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (eval_scale _ BM BN (fun i => ∑ d : Fin BD, q (i.1, d, PUnit.unit) * k (d, i.2.1, PUnit.unit))
      scale (by simp) (by simp [hscale])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, by rw [BlockState.setReg_same]; rfl, rfl, rfl, rfl, ?_⟩
  intro dtype shape name hname
  simp [hname]

theorem eval_self_inequality (s : BlockState) (BM : Nat) (tmp : Tile .real [BM])
    (ht : s.regs .real [BM] "tmp" = some tmp) :
    evalOp (.ne .real (.consSame .nil) (.ref .real [BM] "tmp") (.ref .real [BM] "tmp")) s =
      some (⟨fun _ => Bool.false⟩ : Tile .bool [BM]) := by
  simp only [evalOp.eq_def, ht, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨row, ⟨⟩⟩
  simp [Tile.cop_data, ComparableDType.ne]

theorem eval_alpha_repair (s : BlockState) (BM : Nat) (alpha : Tile .real [BM])
    (hm : s.regs .bool [BM] "alpha_mask" = some (⟨fun _ => Bool.false⟩ : Tile .bool [BM]))
    (ha : s.regs .real [BM] "alpha" = some alpha) :
    evalOp (.where (.ref .bool [BM] "alpha_mask")
      (.broadcast (.const 1.0) [BM]) (.ref .real [BM] "alpha")) s = some alpha := by
  simp only [evalOp.eq_def, hm, ha, bind, Option.bind]
  congr 1

/-- Execute the actual seven rescaling assignments, including the NaN repair
syntax. The result preserves memory and every register outside this fragment's
write set, so it can be composed with the surrounding load/loop proof. -/
theorem rescale_executes (s : BlockState) (BM BN BD : Nat)
    (m mij l : Tile .real [BM]) (p : Tile .real [BM, BN]) (acc : Tile .real [BM, BD])
    (hm : s.regs .real [BM] "m_i" = some m)
    (hmij : s.regs .real [BM] "m_ij" = some mij)
    (hl : s.regs .real [BM] "l_i" = some l)
    (hp : s.regs .real [BM, BN] "p" = some p)
    (hacc : s.regs .real [BM, BD] "acc" = some acc) :
    let alpha := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (.consSame .nil) m mij)
    let nextL := Tile.bop NumericDType.real.add (.consSame .nil)
      (Tile.bop NumericDType.real.mul (.consSame .nil) l alpha)
      (Tile.reduceSumDrop ⟨1, by simp⟩ p)
    let nextAcc := Tile.bop NumericDType.real.mul (.consSame (.consR .nil)) acc
      (Tile.expandDim ⟨1, by simp⟩ alpha)
    ∃ s', stepStmts (RealBody.rescale BM BN BD) s = some s' ∧
      s'.regs .real [BM] "l_i" = some nextL ∧
      s'.regs .real [BM, BD] "acc" = some nextAcc ∧
      s'.regs .real [BM] "alpha" = some alpha ∧
      s'.regs .bool [BM] "alpha_mask" = some (⟨fun _ => Bool.false⟩ : Tile .bool [BM]) ∧
      s'.mem = s.mem ∧ s'.pids = s.pids ∧ s'.undef = s.undef ∧
      (∀ dtype shape name, name ∉ ["l_ij", "tmp", "alpha_mask", "alpha", "l_i", "acc"] →
        s'.regs dtype shape name = s.regs dtype shape name) := by
  dsimp only
  simp only [RealBody.rescale]
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .real [BM] "l_ij" _ _ _
    (show evalOp (.reduceSum ⟨1, by simp⟩ Bool.false (.ref .real [BM, BN] "p")) s =
      some (Tile.reduceSumDrop ⟨1, by simp⟩ p) from by simp [evalOp.eq_def, hp]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.sub .real (.consSame .nil) (.ref .real [BM] "m_i") (.ref .real [BM] "m_ij")) _ =
      some (Tile.bop NumericDType.real.sub (.consSame .nil) m mij) from by
        simp [evalOp.eq_def, hm, hmij]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (eval_self_inequality _ BM
    (Tile.bop NumericDType.real.sub (.consSame .nil) m mij) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.exp2 (.ref .real [BM] "tmp")) _ =
      some (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (.consSame .nil) m mij)) from by
        simp [evalOp.eq_def]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (eval_alpha_repair _ BM (Tile.uop WithBot.realExp2
      (Tile.bop NumericDType.real.sub (.consSame .nil) m mij)) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.add .real (.consSame .nil)
      (.mul .real (.consSame .nil) (.ref .real [BM] "l_i") (.ref .real [BM] "alpha"))
      (.ref .real [BM] "l_ij")) _ = some (Tile.bop NumericDType.real.add (.consSame .nil)
        (Tile.bop NumericDType.real.mul (.consSame .nil) l
          (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (.consSame .nil) m mij)))
        (Tile.reduceSumDrop ⟨1, by simp⟩ p)) from by simp [evalOp.eq_def, hl]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.mul .real (.consSame (.consR .nil)) (.ref .real [BM, BD] "acc")
      (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "alpha"))) _ =
      some (Tile.bop NumericDType.real.mul (.consSame (.consR .nil)) acc
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (.consSame .nil) m mij)))) from by
        simp [evalOp.eq_def, hacc]; rfl))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, by simp, by simp, by simp, by simp, rfl, rfl, rfl, ?_⟩
  intro dtype shape name hname
  simp only [List.mem_cons, not_or] at hname
  simp [hname.1, hname.2.1, hname.2.2.1, hname.2.2.2.1, hname.2.2.2.2.1, hname.2.2.2.2.2]

/-- Connect successful rescaling to the independent denominator formula.
The numerator here is the rescaled carry; the subsequent V load/dot still
has to be composed to obtain the full numerator update. -/
theorem rescale_denominator_formula (s : BlockState) (BM BN BD : Nat)
    (score : Fin BM → Fin BN → ℝ) (keep : Fin BM → Fin BN → Bool)
    (old : Fin BM → WithBot ℝ) (l : Fin BM → ℝ) (acc : TileIndex [BM, BD] → ℝ)
    (hm : s.regs .real [BM] "m_i" = some (⟨fun i => old i.1⟩ : Tile .real [BM]))
    (hmij : s.regs .real [BM] "m_ij" = some (⟨fun i => center score keep old i.1⟩ : Tile .real [BM]))
    (hl : s.regs .real [BM] "l_i" = some (finite (fun i : TileIndex [BM] => l i.1)))
    (hp : s.regs .real [BM, BN] "p" =
      some (finite (fun i : TileIndex [BM, BN] => probability score keep old i.1 i.2.1)))
    (hacc : s.regs .real [BM, BD] "acc" = some (finite acc)) :
    ∃ s', stepStmts (RealBody.rescale BM BN BD) s = some s' ∧
      s'.regs .real [BM] "l_i" = some (finite (fun i : TileIndex [BM] =>
        (RealFormula.summarize ⟨old i.1, l i.1, 0⟩ (pairs (score i.1) (fun _ => 0) (keep i.1))).denominator)) ∧
      s'.regs .real [BM, BD] "acc" = some (finite (fun i : TileIndex [BM, BD] =>
        acc i * RealFormula.weight (old i.1) (center score keep old i.1))) ∧
      s'.mem = s.mem ∧ s'.pids = s.pids ∧ s'.undef = s.undef := by
  obtain ⟨s', hs, hli, hac, _, _, hmem, hpids, hundef, _⟩ :=
    rescale_executes s BM BN BD _ _ _ _ _ hm hmij hl hp hacc
  rw [alpha_matches BM BN score keep old] at hli hac
  refine ⟨s', hs, hli.trans (congrArg some ?_), hac.trans (congrArg some ?_), hmem, hpids, hundef⟩
  · apply Tile.ext
    rintro ⟨row, ⟨⟩⟩
    exact denominator_matches BM BN score keep old l row (fun _ => 0) 0
  · apply Tile.ext
    rintro ⟨row, lane, ⟨⟩⟩
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealNumeric
