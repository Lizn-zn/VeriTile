import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealAccumulation

/-! Successful execution and independent row formulas for one complete
iteration of the checked source loop. Q and the carried M/L/Out registers are
inputs to the iteration; K and V are read through the actual block pointers.
The loop induction, prologue, writeback and global address contract remain
separate obligations. No FP assumption is used in these real-side lemmas. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealIteration
open VeriTile.Triton
open RealTiles RealScores RealAccumulation
open scoped BigOperators
noncomputable section

theorem values_of_same_memory {shape : TileShape} (s t : BlockState)
    (ptrs : Tile .blockPtr shape) (checked : List Nat) (hmem : s.mem = t.mem) :
    values s ptrs checked = values t ptrs checked := by
  funext i
  simp only [values, BlockState.readMem, hmem]

/-- A complete iteration computes the row update from the original memory
view and carry. Physical BLOCK_N is positive; window flags, signed positions,
partial blocks, pointer strides and empty kept-key sets remain unrestricted.
Address validity is not claimed here: it belongs to the final IO contract. -/
theorem iteration_executes (s : BlockState)
    (BM BN BD startM context window complement even : Nat) (hN : 0 < BN)
    (startN offset size : Int) (q : TileIndex [BM, BD] → ℝ) (scale : ℝ)
    (old : Fin BM → WithBot ℝ) (den : Fin BM → ℝ) (acc : TileIndex [BM, BD] → ℝ)
    (kp : Tile .blockPtr [BD, BN]) (vp : Tile .blockPtr [BN, BD])
    (hq : s.regs .real [BM, BD] "q" = some (finite q))
    (hscale : s.regs .real [] "qk_scale" = some (Tile.scalar (some scale)))
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar startM))
    (hn : s.regs .int [] "start_n" = some (Tile.scalar startN))
    (hold : s.regs .real [BM] "m_i" = some (⟨fun i => old i.1⟩ : Tile .real [BM]))
    (hden : s.regs .real [BM] "l_i" = some (finite (fun i : TileIndex [BM] => den i.1)))
    (hacc : s.regs .real [BM, BD] "acc" = some (finite acc))
    (hk : s.regs .blockPtr [BD, BN] "K_block_ptr" = some kp)
    (hv : s.regs .blockPtr [BN, BD] "V_block_ptr" = some vp) :
    let score := fun row key => RealNumeric.scores BM BN BD q (values s kp (checks even)) scale (row, key, PUnit.unit)
    let active := keep BM BN startM context window complement even startN offset size
    ∃ s', stepStmts (RealBody.body BM BN BD context window complement even offset size) s = some s' ∧
      s'.regs .real [BM] "m_i" = some (⟨fun i => center score active old i.1⟩ : Tile .real [BM]) ∧
      s'.regs .real [BM] "l_i" = some (finite (fun i : TileIndex [BM] =>
        (RealFormula.summarize ⟨old i.1, den i.1, 0⟩ (pairs (score i.1) (fun _ => 0) (active i.1))).denominator)) ∧
      s'.regs .real [BM, BD] "acc" = some (finite (fun i : TileIndex [BM, BD] =>
        (RealFormula.summarize ⟨old i.1, den i.1, acc i⟩
          (pairs (score i.1) (fun key => values s vp (checks even) (key, i.2.1, PUnit.unit))
            (active i.1))).numerator)) ∧
      s'.regs .blockPtr [BD, BN] "K_block_ptr" =
        some (⟨fun i => (kp.data i).advance [0, BN]⟩ : Tile .blockPtr [BD, BN]) ∧
      s'.regs .blockPtr [BN, BD] "V_block_ptr" =
        some (⟨fun i => (vp.data i).advance [BN, 0]⟩ : Tile .blockPtr [BN, BD]) ∧
      Frame ["start_n", "k", "qk", "dist", "mask", "m_ij", "p", "l_ij", "tmp",
        "alpha_mask", "alpha", "l_i", "acc", "v", "m_i", "V_block_ptr", "K_block_ptr"] s s' := by
  let kval := values s kp (checks even)
  let vval := values s vp (checks even)
  let score := fun row key => RealNumeric.scores BM BN BD q kval scale (row, key, PUnit.unit)
  let active := keep BM BN startM context window complement even startN offset size
  let sn := s.setReg "start_n" .int [] (Tile.scalar startN)
  let sk := sn.setReg "k" .real [BD, BN] (finite kval)
  have hstart : stepStmts [.assign .int [] "start_n" (.ref .int [] "start_n"),
      RealBody.loadKey BD BN even] s = some sk := by
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.ref .int [] "start_n") s =
      some (Tile.scalar startN) from by simpa using hn)),
      stepStmts.cons_some (load_key_executes sn BD BN even kp (by simp [sn, hk])), stepStmts.nil]
    rfl
  have hkframe : Frame ["start_n", "k"] s sk :=
    (Frame.setReg _ s "start_n" _ _ _ (by simp)).trans (Frame.setReg _ sn "k" _ _ _ (by simp))
  obtain ⟨sp, hpstep, hcenter, hprob, hpframe⟩ :=
    RealProbabilities.scores_probabilities_execute sk BM BN BD startM context window complement even hN
      startN offset size q kval scale old (by simp [sk, sn, hq]) (by simp [sk])
      (by simp [sk, sn, hscale]) (by simp [sk, sn, hm]) (by simp [sk, sn]) (by simp [sk, sn, hold])
  obtain ⟨sr, hrstep, hrden, hracc, _, _, hrmem, hrpids, hrundef, hrregs⟩ :=
    RealNumeric.rescale_executes sp BM BN BD
      (⟨fun i => old i.1⟩ : Tile .real [BM]) (⟨fun i => center score active old i.1⟩ : Tile .real [BM])
      (finite (fun i : TileIndex [BM] => den i.1))
      (finite (fun i : TileIndex [BM, BN] => probability score active old i.1 i.2.1)) (finite acc)
      ((hpframe.regs .real [BM] "m_i" (by simp)).trans (by simp [sk, sn, hold])) hcenter
      ((hpframe.regs .real [BM] "l_i" (by simp)).trans (by simp [sk, sn, hden])) hprob
      ((hpframe.regs .real [BM, BD] "acc" (by simp)).trans (by simp [sk, sn, hacc]))
  have hrframe : Frame ["l_ij", "tmp", "alpha_mask", "alpha", "l_i", "acc"] sp sr :=
    ⟨hrmem, hrpids, hrundef, hrregs⟩
  rw [alpha_matches BM BN score active old] at hrden hracc
  have hdenFormula : sr.regs .real [BM] "l_i" = some (finite (fun i : TileIndex [BM] =>
      (RealFormula.summarize ⟨old i.1, den i.1, 0⟩ (pairs (score i.1) (fun _ => 0) (active i.1))).denominator)) := by
    refine hrden.trans (congrArg some ?_)
    apply Tile.ext
    rintro ⟨row, ⟨⟩⟩
    exact denominator_matches BM BN score active old den row (fun _ => 0) 0
  have haccFormula : sr.regs .real [BM, BD] "acc" = some (finite (fun i : TileIndex [BM, BD] =>
      acc i * RealFormula.weight (old i.1) (center score active old i.1))) := by
    exact hracc.trans (congrArg some (by apply Tile.ext; rintro ⟨row, lane, ⟨⟩⟩; rfl))
  obtain ⟨sf, hfstep, hfacc, hfm, hfk, hfv, hfframe⟩ := accumulate_executes sr BM BN BD even
    (fun i : TileIndex [BM, BD] => acc i * RealFormula.weight (old i.1) (center score active old i.1))
    (fun i : TileIndex [BM, BN] => probability score active old i.1 i.2.1)
    (⟨fun i => center score active old i.1⟩ : Tile .real [BM]) kp vp haccFormula
    ((hrframe.regs .real [BM, BN] "p" (by simp)).trans hprob)
    ((hrframe.regs .real [BM] "m_ij" (by simp)).trans hcenter)
    ((hrframe.regs .blockPtr [BD, BN] "K_block_ptr" (by simp)).trans
      ((hpframe.regs .blockPtr [BD, BN] "K_block_ptr" (by simp)).trans (by simp [sk, sn, hk])))
    ((hrframe.regs .blockPtr [BN, BD] "V_block_ptr" (by simp)).trans
      ((hpframe.regs .blockPtr [BN, BD] "V_block_ptr" (by simp)).trans (by simp [sk, sn, hv])))
  have hvalues : values sr vp (checks even) = vval :=
    values_of_same_memory sr s vp (checks even) (hrmem.trans (hpframe.mem.trans hkframe.mem))
  rw [hvalues] at hfacc
  have hnumFormula : sf.regs .real [BM, BD] "acc" = some (finite (fun i : TileIndex [BM, BD] =>
      (RealFormula.summarize ⟨old i.1, den i.1, acc i⟩
        (pairs (score i.1) (fun key => vval (key, i.2.1, PUnit.unit)) (active i.1))).numerator)) := by
    refine hfacc.trans (congrArg some ?_)
    apply Tile.ext
    rintro ⟨row, lane, ⟨⟩⟩
    have h := numerator_matches BM BN BD score active old acc vval row lane (den row)
    rw [dot_finite] at h
    exact h
  refine ⟨sf, ?_, hfm, (hfframe.regs .real [BM] "l_i" (by simp)).trans hdenFormula,
    hnumFormula, hfk, hfv, ?_⟩
  · unfold RealBody.body
    rw [List.append_assoc _ (RealBody.scores BM BN BD context window complement even offset size)
      (RealBody.probabilities BM BN context window even)]
    have hpast := (stepStmts.append_some hstart).trans hpstep
    have hrescaled := (stepStmts.append_some hpast).trans hrstep
    exact (stepStmts.append_some hrescaled).trans hfstep
  · exact (((hkframe.mono (by simp)).trans (hpframe.mono (by simp))).trans
      (hrframe.mono (by simp))).trans (hfframe.mono (by simp))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealIteration
