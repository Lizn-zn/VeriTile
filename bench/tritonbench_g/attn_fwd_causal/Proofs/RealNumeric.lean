import bench.tritonbench_g.attn_fwd_causal.Proofs.RealBody
import bench.tritonbench_g.attn_fwd_causal.Proofs.RealTiles
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealProbabilities

/-! Execution of arithmetic fragments taken from the two source helpers.
Register premises are intermediate-state invariants; memory loads and loop
induction must establish them. No FP rewrite assumptions enter these proofs. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealNumeric
open VeriTile.Triton
open AttentionFwdTriton3.RealTiles (finite dot_finite)
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 800000
set_option linter.unusedSimpArgs false

def scores (BM BN BD : Nat) (q : TileIndex [BM,BD] → ℝ) (k : TileIndex [BD,BN] → ℝ)
    (qs ks : ℝ) : Fin BM → Fin BN → ℝ :=
  fun row key => (∑ d : Fin BD, q (row,d,PUnit.unit) * k (d,key,PUnit.unit)) * qs * ks

theorem eval_score (s : BlockState) (BM BN BD : Nat)
    (q : TileIndex [BM,BD] → ℝ) (k : TileIndex [BD,BN] → ℝ) (qs ks : ℝ)
    (hq : s.regs .real [BM,BD] "q" = some (finite q))
    (hk : s.regs .real [BD,BN] "k" = some (finite k))
    (hqs : s.regs .real [] "q_scale" = some (Tile.scalar (some qs)))
    (hks : s.regs .real [] "k_scale" = some (Tile.scalar (some ks))) :
    evalOp (RealBody.scoreExpr BM BN BD) s =
      some (finite (fun i : TileIndex [BM,BN] => scores BM BN BD q k qs ks i.1 i.2.1)) := by
  simp only [RealBody.scoreExpr, evalOp.eq_def, List.nil_append, hq, hk, hqs, hks, bind, Option.bind]
  erw [dot_finite BM BN BD q k]
  rfl

theorem score_executes (s : BlockState) (BM BN BD : Nat)
    (q : TileIndex [BM,BD] → ℝ) (k : TileIndex [BD,BN] → ℝ) (qs ks : ℝ)
    (hq : s.regs .real [BM,BD] "q" = some (finite q))
    (hk : s.regs .real [BD,BN] "k" = some (finite k))
    (hqs : s.regs .real [] "q_scale" = some (Tile.scalar (some qs)))
    (hks : s.regs .real [] "k_scale" = some (Tile.scalar (some ks))) :
    ∃ t, stepStmt (RealBody.score BM BN BD) s = some t ∧
      t.regs .real [BM,BN] "qk" =
        some (finite (fun i : TileIndex [BM,BN] => scores BM BN BD q k qs ks i.1 i.2.1)) ∧
      Frame ["qk"] s t := by
  refine ⟨_, stepStmt_assign_eq_some (eval_score s BM BN BD q k qs ks hq hk hqs hks), by simp, ?_⟩
  exact Frame.setReg _ s "qk" _ _ _ (by simp)

theorem diagonal_executes (s : BlockState) (BM BN startM startN : Nat)
    (label : RegName) (hl : label ≠ "qk") (score : Fin BM → Fin BN → ℝ)
    (hqk : s.regs .real [BM,BN] "qk" = some (finite (fun i : TileIndex [BM,BN] => score i.1 i.2.1)))
    (hm : s.regs .nat [BM] "offs_m" = some ⟨fun i => startM*BM+i.1.val⟩)
    (hn : s.regs .nat [BN] "offs_n" = some ⟨fun i => i.1.val⟩)
    (hs : s.regs .nat [] "start_n" = some (Tile.scalar startN)) :
    ∃ t, stepStmts (RealBody.diagonal BM BN label) s = some t ∧
      t.regs .real [BM,BN] "qk" = some (finite (fun i : TileIndex [BM,BN] =>
        score i.1 i.2.1 + RealFormula.penalty Bool.true (startM*BM+i.1.val) (startN+i.2.1.val))) ∧
      Frame [label,"qk"] s t := by
  let mask := RealExpressions.maskTile BM BN startM startN
  let sm := s.setReg label .bool [BM,BN] mask
  have hmask := RealExpressions.eval_mask s BM BN startM startN hm hn hs
  have hp := RealExpressions.eval_penalty sm BM BN startM startN label score
    (by simpa [sm, Ne.symm hl] using hqk) (by simp [sm, mask])
  simp only [RealBody.diagonal]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hmask),
    stepStmts.cons_some (stepStmt_assign_eq_some hp), stepStmts.nil]
  refine ⟨_, rfl, by simp [RealExpressions.scoreTile, finite], ?_⟩
  exact (Frame.setReg _ s label _ _ _ (by simp)).trans (Frame.setReg _ sm "qk" _ _ _ (by simp))

theorem probabilities_execute (s : BlockState) (BM BN : Nat) (hN : 0 < BN)
    (score : Fin BM → Fin BN → ℝ) (old : Fin BM → WithBot ℝ)
    (hm : s.regs .real [BM] "m_i" = some (⟨fun i => old i.1⟩ : Tile .real [BM]))
    (hqk : s.regs .real [BM,BN] "qk" = some (finite (fun i : TileIndex [BM,BN] => score i.1 i.2.1))) :
    ∃ t, stepStmts (RealBody.center BM BN ++ [RealBody.probability BM BN]) s = some t ∧
      t.regs .real [BM] "m_ij" = some (⟨fun i => RealTiles.center score old i.1⟩ : Tile .real [BM]) ∧
      t.regs .real [BM,BN] "p" =
        some (finite (fun i : TileIndex [BM,BN] => RealTiles.probability score old i.1 i.2.1)) ∧
      Frame ["m_ij","qk","p"] s t := by
  have hc : RealBody.center BM BN ++ [RealBody.probability BM BN] =
      (AttentionFwdTriton3.RealBody.probabilities BM BN 0 0 0).take 3 := rfl
  rw [hc]
  exact AttentionFwdTriton3.RealProbabilities.prefix_executes s BM BN 0 0 0 hN score
    RealTiles.keep old hm (by simpa only [RealTiles.keep_eq, Bool.true_eq, ↓reduceIte] using hqk)

theorem rescale_executes (s : BlockState) (BM BN BD : Nat)
    (m mij l : Tile .real [BM]) (p : Tile .real [BM,BN]) (acc : Tile .real [BM,BD])
    (hm : s.regs .real [BM] "m_i" = some m)
    (hmij : s.regs .real [BM] "m_ij" = some mij)
    (hl : s.regs .real [BM] "l_i" = some l)
    (hp : s.regs .real [BM,BN] "p" = some p)
    (ha : s.regs .real [BM,BD] "acc" = some acc) :
    let alpha := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (.consSame .nil) m mij)
    let nextL := Tile.bop NumericDType.real.add (.consSame .nil)
      (Tile.bop NumericDType.real.mul (.consSame .nil) l alpha) (Tile.reduceSumDrop ⟨1, by simp⟩ p)
    let nextA := Tile.bop NumericDType.real.mul (.consSame (.consR .nil)) acc (Tile.expandDim ⟨1, by simp⟩ alpha)
    ∃ t, stepStmts (RealBody.rescale BM BN BD) s = some t ∧
      t.regs .real [BM] "l_i" = some nextL ∧ t.regs .real [BM,BD] "acc" = some nextA ∧
      Frame ["l_ij","alpha","l_i","acc"] s t := by
  dsimp only
  simp only [RealBody.rescale]
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .real [BM] "l_ij" _ _ _
    (show evalOp (.reduceSum ⟨1, by simp⟩ Bool.false (.ref .real [BM,BN] "p")) s =
      some (Tile.reduceSumDrop ⟨1, by simp⟩ p) from by simp [evalOp.eq_def, hp]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.exp2 (.sub .real (.consSame .nil) (.ref .real [BM] "m_i") (.ref .real [BM] "m_ij"))) _ =
      some (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (.consSame .nil) m mij)) from by
        simp [evalOp.eq_def, hm, hmij]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.add .real (.consSame .nil)
      (.mul .real (.consSame .nil) (.ref .real [BM] "l_i") (.ref .real [BM] "alpha"))
      (.ref .real [BM] "l_ij")) _ = some (Tile.bop NumericDType.real.add (.consSame .nil)
        (Tile.bop NumericDType.real.mul (.consSame .nil) l
          (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (.consSame .nil) m mij)))
        (Tile.reduceSumDrop ⟨1, by simp⟩ p)) from by simp [evalOp.eq_def, hl]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.mul .real (.consSame (.consR .nil)) (.ref .real [BM,BD] "acc")
      (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "alpha"))) _ =
      some (Tile.bop NumericDType.real.mul (.consSame (.consR .nil)) acc
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (.consSame .nil) m mij)))) from by
        simp [evalOp.eq_def, ha]; rfl))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, by simp, by simp, rfl, rfl, rfl, ?_⟩
  intro dtype shape name hn
  simp only [List.mem_cons, not_or] at hn
  simp [hn.1, hn.2.1, hn.2.2.1, hn.2.2.2.1]

theorem accumulate_executes (s : BlockState) (BM BN BD : Nat)
    (p : TileIndex [BM,BN] → ℝ) (v : TileIndex [BN,BD] → ℝ) (a : TileIndex [BM,BD] → ℝ)
    (m : Tile .real [BM])
    (hp : s.regs .real [BM,BN] "p" = some (finite p))
    (hv : s.regs .real [BN,BD] "v" = some (finite v))
    (ha : s.regs .real [BM,BD] "acc" = some (finite a))
    (hm : s.regs .real [BM] "m_ij" = some m) :
    ∃ t, stepStmts (RealBody.accumulate BM BN BD) s = some t ∧
      t.regs .real [BM,BD] "acc" = some (finite (fun i : TileIndex [BM,BD] =>
        a i + ∑ j : Fin BN, p (i.1,j,PUnit.unit) * v (j,i.2.1,PUnit.unit))) ∧
      t.regs .real [BM] "m_i" = some m ∧ Frame ["p","acc","m_i"] s t := by
  let sp := s.setReg "p" .real [BM,BN] (finite p)
  have hp' : sp.regs .real [BM,BN] "p" = some (finite p) := by simp [sp]
  have hv' : sp.regs .real [BN,BD] "v" = some (finite v) := by simp [sp, hv]
  have ha' : sp.regs .real [BM,BD] "acc" = some (finite a) := by simp [sp, ha]
  have hdot : evalOp (.add .real (.consSame (.consSame .nil)) (.ref .real [BM,BD] "acc")
      (.dot (options := {outDType := some .fp16}) (batch := [])
        (.ref .real [BM,BN] "p") (.ref .real [BN,BD] "v"))) sp =
      some (finite (fun i : TileIndex [BM,BD] =>
        a i + ∑ j : Fin BN, p (i.1,j,PUnit.unit) * v (j,i.2.1,PUnit.unit))) := by
    simp only [evalOp.eq_def, List.nil_append, hp', hv', ha', bind, Option.bind]
    erw [dot_finite BM BD BN p v]
    rfl
  simp only [RealBody.accumulate]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.ref .real [BM,BN] "p") s =
    some (finite p) from by simpa using hp)), stepStmts.cons_some (stepStmt_assign_eq_some hdot)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.ref .real [BM] "m_ij") _ = some m from by simp [sp, hm])), stepStmts.nil]
  refine ⟨_, rfl, by simp, by simp, rfl, rfl, rfl, ?_⟩
  intro dtype shape name hn
  simp only [List.mem_cons, not_or] at hn
  simp [hn.1, hn.2.1, hn.2.2.1, sp]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealNumeric
