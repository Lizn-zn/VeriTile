import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealScores

/-! Execution of the source row maximum, exponentiation and probability masks.
Only the physical reduction dimension must be positive. The kept key set can
be empty, and the old row maximum can be bottom. All reasoning here belongs
to the precision-erased real semantics, without FP rewrite assumptions. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealProbabilities
open VeriTile.Triton
open RealTiles RealScores
noncomputable section

theorem eval_center (s : BlockState) (BM BN : Nat) (hN : 0 < BN)
    (score : Fin BM → Fin BN → ℝ) (active : Fin BM → Fin BN → Bool)
    (old : Fin BM → WithBot ℝ)
    (hm : s.regs .real [BM] "m_i" = some (⟨fun i => old i.1⟩ : Tile .real [BM]))
    (hqk : s.regs .real [BM, BN] "qk" = some
      (⟨fun i => if active i.1 i.2.1 then (score i.1 i.2.1 : WithBot ℝ) else ⊥⟩ : Tile .real [BM, BN])) :
    evalOp (.where (.gt .real (.consSame .nil) (.ref .real [BM] "m_i")
      (.reduceMax ⟨1, by simp⟩ Bool.false (.ref .real [BM, BN] "qk")))
      (.ref .real [BM] "m_i")
      (.reduceMax ⟨1, by simp⟩ Bool.false (.ref .real [BM, BN] "qk"))) s =
      some (⟨fun i => center score active old i.1⟩ : Tile .real [BM]) := by
  have hr : @evalOp .real [BM]
      (.reduceMax ⟨1, by simp⟩ Bool.false (.ref .real [BM, BN] "qk")) s =
      some (⟨fun i => RealFormula.maxScore (pairs (score i.1) (fun _ => 0) (active i.1))⟩ : Tile .real [BM]) := by
    erw [evalOp_reduceMax]
    simp only [evalOp_ref, hqk, bind, Option.bind]
    exact reduce_max_masked BM BN hN score active
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hm, hr, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨row, ⟨⟩⟩
  simp only [Tile.select_data, Tile.cop_data, ComparableDType.gt, center, decide_eq_true_eq]
  change (if _ < old row then old row else _) = max (old row) _
  split
  · exact (max_eq_left (le_of_lt ‹_›)).symm
  · exact (max_eq_right (le_of_not_gt ‹_›)).symm

/-- Execute the first three probability statements from a masked score tile.
The result is the independent center and finite probability formula. -/
theorem prefix_executes (s : BlockState) (BM BN context window even : Nat) (hN : 0 < BN)
    (score : Fin BM → Fin BN → ℝ) (active : Fin BM → Fin BN → Bool)
    (old : Fin BM → WithBot ℝ)
    (hm : s.regs .real [BM] "m_i" = some (⟨fun i => old i.1⟩ : Tile .real [BM]))
    (hqk : s.regs .real [BM, BN] "qk" = some
      (⟨fun i => if active i.1 i.2.1 then (score i.1 i.2.1 : WithBot ℝ) else ⊥⟩ : Tile .real [BM, BN])) :
    ∃ s', stepStmts ((RealBody.probabilities BM BN context window even).take 3) s = some s' ∧
      s'.regs .real [BM] "m_ij" = some (⟨fun i => center score active old i.1⟩ : Tile .real [BM]) ∧
      s'.regs .real [BM, BN] "p" =
        some (finite (fun i : TileIndex [BM, BN] => probability score active old i.1 i.2.1)) ∧
      Frame ["m_ij", "qk", "p"] s s' := by
  let c : Tile .real [BM] := ⟨fun i => center score active old i.1⟩
  let qkt : Tile .real [BM, BN] :=
    ⟨fun i => if active i.1 i.2.1 then (score i.1 i.2.1 : WithBot ℝ) else ⊥⟩
  let sc := s.setReg "m_ij" .real [BM] c
  let shifted := Tile.bop NumericDType.real.sub (.consSame (.consR .nil)) qkt
    (Tile.expandDim ⟨1, by simp⟩ c)
  let ss := sc.setReg "qk" .real [BM, BN] shifted
  let p := finite (fun i : TileIndex [BM, BN] => probability score active old i.1 i.2.1)
  have hc := eval_center s BM BN hN score active old hm hqk
  have hshift : evalOp (.sub .real (.consSame (.consR .nil)) (.ref .real [BM, BN] "qk")
      (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "m_ij"))) sc = some shifted := by
    simp [evalOp.eq_def, sc, hqk, shifted, qkt]
    rfl
  have hp : evalOp (.exp2 (.ref .real [BM, BN] "qk")) ss = some p := by
    simp only [evalOp.eq_def, ss, BlockState.setReg_same, bind, Option.bind]
    exact congrArg some (probabilities_match BM BN score active old)
  simp only [RealBody.probabilities, List.take_succ_cons, List.take_zero]
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .real [BM] "m_ij" _ _ _ hc),
    stepStmts.cons_some (stepStmt_assign_eq_some hshift),
    stepStmts.cons_some (stepStmt_assign_eq_some hp), stepStmts.nil]
  refine ⟨_, rfl, by simp [sc, ss, c], by simp [p], ?_⟩
  exact ((Frame.setReg _ s "m_ij" _ _ _ (by simp)).trans
    (Frame.setReg _ sc "qk" _ _ _ (by simp))).trans
    (Frame.setReg _ ss "p" _ _ _ (by simp))

/-- A later mask can only reject probabilities already zero from the score
mask. This is a predicate implication, not an assumed output equality. -/
theorem eval_remask (s : BlockState) (BM BN : Nat) (mask : Op .bool [BM, BN])
    (mt : Tile .bool [BM, BN]) (score : Fin BM → Fin BN → ℝ)
    (active : Fin BM → Fin BN → Bool) (old : Fin BM → WithBot ℝ)
    (hm : evalOp mask s = some mt)
    (hp : s.regs .real [BM, BN] "p" =
      some (finite (fun i : TileIndex [BM, BN] => probability score active old i.1 i.2.1)))
    (hkeep : ∀ row key, active row key = Bool.true → mt.data (row, key, PUnit.unit) = Bool.true) :
    evalOp (.where mask (.ref .real [BM, BN] "p") (.broadcast (.const 0.0) [BM, BN])) s =
      some (finite (fun i : TileIndex [BM, BN] => probability score active old i.1 i.2.1)) := by
  have hz : evalOp (.broadcast (.const 0.0) [BM, BN]) s =
      some (finite (fun _ : TileIndex [BM, BN] => 0)) := by
    norm_num [evalOp.eq_def, finite]
  rw [evalOp_where]
  simp only [hm, evalOp_ref, hp, hz, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨row, key, ⟨⟩⟩
  by_cases hk : active row key = Bool.true
  · simp [Tile.select_data, finite, hkeep row key hk]
  · simp [Tile.select_data, finite, probability, hk]

/-- Execute the two explicit probability masks. Their domains come from
the earlier score mask; all memory and all registers other than p survive. -/
theorem masks_execute (s : BlockState) (BM BN context window even : Nat) (startN : Int)
    (score : Fin BM → Fin BN → ℝ) (active : Fin BM → Fin BN → Bool)
    (old : Fin BM → WithBot ℝ)
    (hp : s.regs .real [BM, BN] "p" =
      some (finite (fun i : TileIndex [BM, BN] => probability score active old i.1 i.2.1)))
    (hn : s.regs .int [] "start_n" = some (Tile.scalar startN))
    (hw : window ≠ 0 → ∃ mt : Tile .bool [BM, BN], s.regs .bool [BM, BN] "mask" = some mt ∧
      ∀ row key, active row key = Bool.true → mt.data (row, key, PUnit.unit) = Bool.true)
    (ht : even = 0 → ∀ row key, active row key = Bool.true → (key.val : Int) + startN < context) :
    ∃ s', stepStmts ((RealBody.probabilities BM BN context window even).drop 3) s = some s' ∧
      s'.regs .real [BM, BN] "p" =
        some (finite (fun i : TileIndex [BM, BN] => probability score active old i.1 i.2.1)) ∧
      Frame ["p"] s s' := by
  let p := finite (fun i : TileIndex [BM, BN] => probability score active old i.1 i.2.1)
  have windowStep : ∃ sw,
      stepStmt (.ifThen (.ne .nat .nil (.constNat window) (.constNat 0))
        [.assign .real [BM, BN] "p" (.where (.ref .bool [BM, BN] "mask")
          (.ref .real [BM, BN] "p") (.broadcast (.const 0.0) [BM, BN]))]) s = some sw ∧
      sw.regs .real [BM, BN] "p" = some p ∧ Frame ["p"] s sw := by
    rw [step_if_nonzero]
    by_cases h : window ≠ 0
    · rw [if_pos h]
      obtain ⟨mt, hmt, hkeep⟩ := hw h
      have he := eval_remask s BM BN (.ref .bool [BM, BN] "mask") mt score active old
        (by simpa using hmt) hp hkeep
      rw [stepStmts.cons_some (stepStmt_assign_eq_some he), stepStmts.nil]
      exact ⟨_, rfl, by simp [p], Frame.setReg _ s "p" _ _ _ (by simp)⟩
    · rw [if_neg h]
      exact ⟨s, rfl, hp, Frame.refl _ _⟩
  obtain ⟨sw, hws, hwp, hwf⟩ := windowStep
  simp only [RealBody.probabilities, List.drop_succ_cons, List.drop_zero]
  rw [stepStmts.cons_some hws]
  have tailStep : ∃ sf,
      stepStmt (.ifThen (.eq .nat .nil (.constNat even) (.constNat 0))
        [.assign .real [BM, BN] "p" (.where (RealBody.tailMask BM BN context)
          (.ref .real [BM, BN] "p") (.broadcast (.const 0.0) [BM, BN]))]) sw = some sf ∧
      sf.regs .real [BM, BN] "p" = some p ∧ Frame ["p"] sw sf := by
    rw [step_if_zero]
    by_cases h : even = 0
    · rw [if_pos h]
      have he := eval_remask sw BM BN (RealBody.tailMask BM BN context)
        (⟨fun i => decide ((i.2.1.val : Int) + startN < context)⟩ : Tile .bool [BM, BN])
        score active old (RealNumeric.eval_tail_mask sw BM BN context startN
          ((hwf.regs .int [] "start_n" (by simp)).trans hn)) hwp
        (by intro row key hk; exact decide_eq_true (ht h row key hk))
      rw [stepStmts.cons_some (stepStmt_assign_eq_some he), stepStmts.nil]
      exact ⟨_, rfl, by simp [p], Frame.setReg _ sw "p" _ _ _ (by simp)⟩
    · rw [if_neg h]
      exact ⟨sw, rfl, hwp, Frame.refl _ _⟩
  obtain ⟨sf, hfs, hfp, hff⟩ := tailStep
  rw [stepStmts.cons_some hfs, stepStmts.nil]
  exact ⟨sf, rfl, hfp, hwf.trans hff⟩

/-- Compose the row maximum, shift, exp2 and both masks into the entire
source probability path. Mask premises refer to previously computed lanes. -/
theorem probabilities_execute (s : BlockState) (BM BN context window even : Nat) (hN : 0 < BN)
    (startN : Int) (score : Fin BM → Fin BN → ℝ) (active : Fin BM → Fin BN → Bool)
    (old : Fin BM → WithBot ℝ)
    (hm : s.regs .real [BM] "m_i" = some (⟨fun i => old i.1⟩ : Tile .real [BM]))
    (hqk : s.regs .real [BM, BN] "qk" = some
      (⟨fun i => if active i.1 i.2.1 then (score i.1 i.2.1 : WithBot ℝ) else ⊥⟩ : Tile .real [BM, BN]))
    (hn : s.regs .int [] "start_n" = some (Tile.scalar startN))
    (hw : window ≠ 0 → ∃ mt : Tile .bool [BM, BN], s.regs .bool [BM, BN] "mask" = some mt ∧
      ∀ row key, active row key = Bool.true → mt.data (row, key, PUnit.unit) = Bool.true)
    (ht : even = 0 → ∀ row key, active row key = Bool.true → (key.val : Int) + startN < context) :
    ∃ s', stepStmts (RealBody.probabilities BM BN context window even) s = some s' ∧
      s'.regs .real [BM] "m_ij" = some (⟨fun i => center score active old i.1⟩ : Tile .real [BM]) ∧
      s'.regs .real [BM, BN] "p" =
        some (finite (fun i : TileIndex [BM, BN] => probability score active old i.1 i.2.1)) ∧
      Frame ["m_ij", "qk", "p"] s s' := by
  obtain ⟨sr, hr, hc, hp, hf⟩ := prefix_executes s BM BN context window even hN score active old hm hqk
  have hwr : window ≠ 0 → ∃ mt : Tile .bool [BM, BN], sr.regs .bool [BM, BN] "mask" = some mt ∧
      ∀ row key, active row key = Bool.true → mt.data (row, key, PUnit.unit) = Bool.true := by
    intro h
    obtain ⟨mt, hmt, hkeep⟩ := hw h
    exact ⟨mt, (hf.regs .bool [BM, BN] "mask" (by simp)).trans hmt, hkeep⟩
  obtain ⟨sf, hs, hpf, hff⟩ := masks_execute sr BM BN context window even startN score active old hp
    ((hf.regs .int [] "start_n" (by simp)).trans hn) hwr ht
  refine ⟨sf, ?_, (hff.regs .real [BM] "m_ij" (by simp)).trans hc, hpf,
    hf.trans (hff.mono (by simp))⟩
  rw [← List.take_append_drop 3 (RealBody.probabilities BM BN context window even), stepStmts.append_some hr]
  exact hs

/-- Connect the actual Q/K score and probability paths. The mask implications
needed above are discharged from the executed signed-window/tail predicates;
no score or probability output equation is a precondition. -/
theorem scores_probabilities_execute (s : BlockState)
    (BM BN BD startM context window complement even : Nat) (hN : 0 < BN)
    (startN offset size : Int) (q : TileIndex [BM, BD] → ℝ)
    (k : TileIndex [BD, BN] → ℝ) (scale : ℝ) (old : Fin BM → WithBot ℝ)
    (hq : s.regs .real [BM, BD] "q" = some (finite q))
    (hk : s.regs .real [BD, BN] "k" = some (finite k))
    (hscale : s.regs .real [] "qk_scale" = some (Tile.scalar (some scale)))
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar startM))
    (hn : s.regs .int [] "start_n" = some (Tile.scalar startN))
    (hold : s.regs .real [BM] "m_i" = some (⟨fun i => old i.1⟩ : Tile .real [BM])) :
    let score := fun row key => RealNumeric.scores BM BN BD q k scale (row, key, PUnit.unit)
    let active := keep BM BN startM context window complement even startN offset size
    ∃ s', stepStmts (RealBody.scores BM BN BD context window complement even offset size ++
        RealBody.probabilities BM BN context window even) s = some s' ∧
      s'.regs .real [BM] "m_ij" = some (⟨fun i => center score active old i.1⟩ : Tile .real [BM]) ∧
      s'.regs .real [BM, BN] "p" =
        some (finite (fun i : TileIndex [BM, BN] => probability score active old i.1 i.2.1)) ∧
      Frame ["qk", "dist", "mask", "m_ij", "p"] s s' := by
  dsimp only
  obtain ⟨sr, hr, hqkr, hmr, hfr⟩ := scores_execute s BM BN BD startM context window complement even
    startN offset size q k scale hq hk hscale hm hn
  have hwindow : window ≠ 0 → ∃ mt : Tile .bool [BM, BN], sr.regs .bool [BM, BN] "mask" = some mt ∧
      ∀ row key, keep BM BN startM context window complement even startN offset size row key = Bool.true →
        mt.data (row, key, PUnit.unit) = Bool.true := by
    intro hw
    refine ⟨windowKeep BM BN startM window complement startN offset size, hmr hw, ?_⟩
    intro row key h
    exact (Bool.and_eq_true_iff.mp h).1
  have htail : even = 0 → ∀ row key,
      keep BM BN startM context window complement even startN offset size row key = Bool.true →
        (key.val : Int) + startN < context := by
    intro he row key h
    have htail := (Bool.and_eq_true_iff.mp h).2
    simpa [tailKeep, he] using htail
  obtain ⟨sf, hs, hc, hp, hff⟩ := probabilities_execute sr BM BN context window even hN startN
    (fun row key => RealNumeric.scores BM BN BD q k scale (row, key, PUnit.unit))
    (keep BM BN startM context window complement even startN offset size) old
    ((hfr.regs .real [BM] "m_i" (by simp)).trans hold) hqkr
    ((hfr.regs .int [] "start_n" (by simp)).trans hn) hwindow htail
  refine ⟨sf, (stepStmts.append_some hr).trans hs, hc, hp, ?_⟩
  exact (hfr.mono (by simp)).trans (hff.mono (by simp))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealProbabilities
