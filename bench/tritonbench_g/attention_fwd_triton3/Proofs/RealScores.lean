import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealNumeric

/-! Execution of the complete score path, including the signed window and
the independent KV tail mask. These helpers use the loop body checked against
both source kernels in RealBody; sizes and all source flags remain symbolic. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores
open VeriTile.Triton
open RealTiles RealExpressions
noncomputable section

/-- The state components a register-only fragment must preserve. -/
structure Frame (writes : List RegName) (before after : BlockState) : Prop where
  mem : after.mem = before.mem
  pids : after.pids = before.pids
  undef : after.undef = before.undef
  regs : ∀ dtype shape name, name ∉ writes → after.regs dtype shape name = before.regs dtype shape name

theorem Frame.refl (writes : List RegName) (s : BlockState) : Frame writes s s :=
  ⟨rfl, rfl, rfl, fun _ _ _ _ => rfl⟩

theorem Frame.trans {writes : List RegName} {s t u : BlockState}
    (h : Frame writes s t) (h' : Frame writes t u) : Frame writes s u :=
  ⟨h'.mem.trans h.mem, h'.pids.trans h.pids, h'.undef.trans h.undef,
    fun dt sh name hn => (h'.regs dt sh name hn).trans (h.regs dt sh name hn)⟩

theorem Frame.mono {writes more : List RegName} {s t : BlockState}
    (h : Frame writes s t) (hsub : writes ⊆ more) : Frame more s t :=
  ⟨h.mem, h.pids, h.undef, fun dt sh name hn => h.regs dt sh name (fun hw => hn (hsub hw))⟩

theorem Frame.setReg (writes : List RegName) (s : BlockState)
    (name : RegName) (dtype : TileDType) (shape : TileShape) (value : Tile dtype shape)
    (hw : name ∈ writes) : Frame writes s (s.setReg name dtype shape value) := by
  refine ⟨rfl, rfl, rfl, ?_⟩
  intro dt sh readName hn
  have hne : readName ≠ name := by rintro rfl; exact hn hw
  simp [hne]

theorem step_if_nonzero (flag : Nat) (body : List Stmt) (s : BlockState) :
    stepStmt (.ifThen (.ne .nat .nil (.constNat flag) (.constNat 0)) body) s =
      if flag ≠ 0 then stepStmts body s else some s := by
  conv_lhs => unfold stepStmt
  simp only [evalOp.eq_def, bind, Option.bind, Tile.cop_data, Tile.scalar, ComparableDType.ne]
  simp

theorem step_if_zero (flag : Nat) (body : List Stmt) (s : BlockState) :
    stepStmt (.ifThen (.eq .nat .nil (.constNat flag) (.constNat 0)) body) s =
      if flag = 0 then stepStmts body s else some s := by
  conv_lhs => unfold stepStmt
  simp only [evalOp.eq_def, bind, Option.bind, Tile.cop_data, Tile.scalar, ComparableDType.eq]
  simp

def windowKeep (BM BN startM window complement : Nat) (startN offset size : Int) :
    Tile .bool [BM, BN] :=
  if window ≠ 0 then selectedMask (distances BM BN startM startN offset) complement size
  else ⟨fun _ => Bool.true⟩

def tailKeep (BM BN context even : Nat) (startN : Int) : Tile .bool [BM, BN] :=
  ⟨fun i => if even = 0 then decide ((i.2.1.val : Int) + startN < context) else Bool.true⟩

def keep (BM BN startM context window complement even : Nat) (startN offset size : Int) :
    Fin BM → Fin BN → Bool :=
  fun row key => (windowKeep BM BN startM window complement startN offset size).data (row, key, PUnit.unit) &&
    (tailKeep BM BN context even startN).data (row, key, PUnit.unit)

def masked {BM BN : Nat} (score : TileIndex [BM, BN] → ℝ)
    (active : Tile .bool [BM, BN]) : Tile .real [BM, BN] :=
  ⟨fun i => if active.data i then (score i : WithBot ℝ) else ⊥⟩

/-- Translate the executed masks to full-context key membership. An unchecked
KV block requires its lanes to be in range; checked blocks can have a tail. -/
theorem keep_matches_keys (BM BN startM startN context window complement even : Nat)
    (offset size : Int) (row : Fin BM) (key : Fin BN)
    (hfull : even ≠ 0 → startN + key.val < context) :
    keep BM BN startM context window complement even startN offset size row key =
      (RealKeys.keep startM BM row.val window complement offset size (startN + key.val) &&
        decide (startN + key.val < context)) := by
  have hw : (windowKeep BM BN startM window complement startN offset size).data
      (row, key, PUnit.unit) =
      RealKeys.keep startM BM row.val window complement offset size (startN + key.val) := by
    by_cases h : window ≠ 0
    · simpa [windowKeep, h] using
        selected_mask_matches_keep BM BN startM startN window complement offset size row key h
    · have hz : window = 0 := by omega
      simp [windowKeep, hz, RealKeys.keep]
  simp only [keep, hw, tailKeep]
  by_cases h : even = 0
  · have hcast : ((key.val : Int) + (startN : Int) < (context : Int)) ↔ startN + key.val < context := by omega
    simp [h, hcast]
  · simp [h, hfull h]

theorem eval_mask_bottom (s : BlockState) (BM BN : Nat) (mask : Op .bool [BM, BN])
    (active : Tile .bool [BM, BN]) (value : Tile .real [BM, BN])
    (hm : evalOp mask s = some active)
    (hv : s.regs .real [BM, BN] "qk" = some value) :
    evalOp (.where mask (.ref .real [BM, BN] "qk") (.broadcast .negInf [BM, BN])) s =
      some (⟨fun i => if active.data i then value.data i else ⊥⟩ : Tile .real [BM, BN]) := by
  simp only [hm, hv, evalOp.eq_def, bind, Option.bind]
  rfl

/-- The two mask statements following the dot/scale prefix execute, preserving
the actual window mask for its later use in the probability path. -/
theorem masks_execute (s : BlockState) (BM BN BD startM context window complement even : Nat)
    (startN offset size : Int) (score : TileIndex [BM, BN] → ℝ)
    (hqk : s.regs .real [BM, BN] "qk" = some (finite score))
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar startM))
    (hn : s.regs .int [] "start_n" = some (Tile.scalar startN)) :
    ∃ s', stepStmts ((RealBody.scores BM BN BD context window complement even offset size).drop 3) s = some s' ∧
      s'.regs .real [BM, BN] "qk" = some (masked score
        ⟨fun i => keep BM BN startM context window complement even startN offset size i.1 i.2.1⟩) ∧
      (window ≠ 0 → s'.regs .bool [BM, BN] "mask" =
        some (windowKeep BM BN startM window complement startN offset size)) ∧
      Frame ["qk", "dist", "mask"] s s' := by
  let wm := windowKeep BM BN startM window complement startN offset size
  have windowStep : ∃ sw,
      stepStmt (.ifThen (.ne .nat .nil (.constNat window) (.constNat 0))
        [.assign .int [BM, BN] "dist" (distanceExpr BM BN offset),
         maskStmt BM BN complement size,
         .assign .real [BM, BN] "qk" (.where (.ref .bool [BM, BN] "mask")
           (.ref .real [BM, BN] "qk") (.broadcast .negInf [BM, BN]))]) s = some sw ∧
      sw.regs .real [BM, BN] "qk" = some (masked score wm) ∧
      (window ≠ 0 → sw.regs .bool [BM, BN] "mask" = some wm) ∧
      Frame ["qk", "dist", "mask"] s sw := by
    by_cases hw : window = 0
    · refine ⟨s, ?_, ?_, ?_, Frame.refl _ _⟩
      · rw [step_if_nonzero]; simp [hw]
      · simpa [masked, wm, windowKeep, hw, finite] using hqk
      · simp [hw]
    · let sd := s.setReg "dist" .int [BM, BN] (distances BM BN startM startN offset)
      let sm := sd.setReg "mask" .bool [BM, BN] wm
      have hmask : stepStmts [.assign .int [BM, BN] "dist" (distanceExpr BM BN offset),
          maskStmt BM BN complement size] s = some sm := by
        simpa [sm, sd, wm, windowKeep, hw] using
          mask_fragment_executes s BM BN startM complement startN offset size hm hn
      have he := eval_mask_bottom sm BM BN (.ref .bool [BM, BN] "mask") wm (finite score)
        (by simp [sm]) (by simp [sm, sd, hqk])
      have he' : evalOp (.where (.ref .bool [BM, BN] "mask")
          (.ref .real [BM, BN] "qk") (.broadcast .negInf [BM, BN])) sm =
          some (masked score wm) := he
      refine ⟨sm.setReg "qk" .real [BM, BN] (masked score wm), ?_, by simp, ?_, ?_⟩
      · rw [step_if_nonzero, if_pos hw]
        rw [show [Stmt.assign .int [BM, BN] "dist" (distanceExpr BM BN offset),
          maskStmt BM BN complement size,
          Stmt.assign .real [BM, BN] "qk" (.where (.ref .bool [BM, BN] "mask")
            (.ref .real [BM, BN] "qk") (.broadcast .negInf [BM, BN]))] =
          [Stmt.assign .int [BM, BN] "dist" (distanceExpr BM BN offset),
            maskStmt BM BN complement size] ++ [_] from rfl]
        rw [stepStmts.append_some hmask, stepStmts.cons_some (stepStmt_assign_eq_some he'), stepStmts.nil]
      · intro _; simp [sm]
      · exact ((Frame.setReg _ s "dist" _ _ _ (by simp)).trans
          (Frame.setReg _ sd "mask" _ _ _ (by simp))).trans
          (Frame.setReg _ sm "qk" _ _ _ (by simp))
  obtain ⟨sw, hws, hwq, hwm, hwf⟩ := windowStep
  simp only [RealBody.scores, List.drop_succ_cons, List.drop_zero]
  rw [stepStmts.cons_some hws]
  by_cases heven : even = 0
  · let active : Tile .bool [BM, BN] :=
      ⟨fun i => decide ((i.2.1.val : Int) + startN < context)⟩
    have ht := RealNumeric.eval_tail_mask sw BM BN context startN
      ((hwf.regs .int [] "start_n" (by simp)).trans hn)
    have he := eval_mask_bottom sw BM BN (RealBody.tailMask BM BN context)
      active (masked score wm) ht hwq
    let result := masked score
      ⟨fun i => keep BM BN startM context window complement even startN offset size i.1 i.2.1⟩
    have he' : evalOp (.where (RealBody.tailMask BM BN context)
        (.ref .real [BM, BN] "qk") (.broadcast .negInf [BM, BN])) sw = some result := by
      rw [he]; congr 1; apply Tile.ext; rintro ⟨row, key, ⟨⟩⟩
      simp only [result, masked, keep, tailKeep, heven, if_true, wm, active]
      split <;> simp_all
    have hs : stepStmt (.ifThen (.eq .nat .nil (.constNat even) (.constNat 0))
        [.assign .real [BM, BN] "qk" (.where (RealBody.tailMask BM BN context)
          (.ref .real [BM, BN] "qk") (.broadcast .negInf [BM, BN]))]) sw =
        some (sw.setReg "qk" .real [BM, BN] result) := by
      rw [step_if_zero, if_pos heven]
      exact (stepStmts.cons_some (stepStmt_assign_eq_some he')).trans stepStmts.nil
    rw [stepStmts.cons_some hs, stepStmts.nil]
    exact ⟨_, rfl, by simp [result], fun h => by simpa using hwm h,
      hwf.trans (Frame.setReg _ sw "qk" _ _ _ (by simp))⟩
  · have hs : stepStmt (.ifThen (.eq .nat .nil (.constNat even) (.constNat 0))
        [.assign .real [BM, BN] "qk" (.where (RealBody.tailMask BM BN context)
          (.ref .real [BM, BN] "qk") (.broadcast .negInf [BM, BN]))]) sw = some sw := by
      rw [step_if_zero, if_neg heven]
    rw [stepStmts.cons_some hs, stepStmts.nil]
    refine ⟨sw, rfl, ?_, hwm, hwf⟩
    simpa [masked, keep, tailKeep, heven, wm] using hwq

/-- Execute the entire source score path from Q, K and the loaded scale,
including both optional masks and the register/memory frame. -/
theorem scores_execute (s : BlockState) (BM BN BD startM context window complement even : Nat)
    (startN offset size : Int) (q : TileIndex [BM, BD] → ℝ)
    (k : TileIndex [BD, BN] → ℝ) (scale : ℝ)
    (hq : s.regs .real [BM, BD] "q" = some (finite q))
    (hk : s.regs .real [BD, BN] "k" = some (finite k))
    (hscale : s.regs .real [] "qk_scale" = some (Tile.scalar (some scale)))
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar startM))
    (hn : s.regs .int [] "start_n" = some (Tile.scalar startN)) :
    ∃ s', stepStmts (RealBody.scores BM BN BD context window complement even offset size) s = some s' ∧
      s'.regs .real [BM, BN] "qk" = some (masked (RealNumeric.scores BM BN BD q k scale)
        ⟨fun i => keep BM BN startM context window complement even startN offset size i.1 i.2.1⟩) ∧
      (window ≠ 0 → s'.regs .bool [BM, BN] "mask" =
        some (windowKeep BM BN startM window complement startN offset size)) ∧
      Frame ["qk", "dist", "mask"] s s' := by
  obtain ⟨sr, hr, hqkr, hmem, hpids, hundef, hregs⟩ :=
    RealNumeric.raw_scores_execute s BM BN BD context window complement even offset size q k scale hq hk hscale
  obtain ⟨sf, hf, hqkf, hmask, hframe⟩ := masks_execute sr BM BN BD startM context window complement even
    startN offset size (RealNumeric.scores BM BN BD q k scale) hqkr
      ((hregs .nat [] "start_m" (by decide)).trans hm)
      ((hregs .int [] "start_n" (by decide)).trans hn)
  refine ⟨sf, ?_, hqkf, hmask, (Frame.mk hmem hpids hundef ?_).trans hframe⟩
  · rw [← List.take_append_drop 3 (RealBody.scores BM BN BD context window complement even offset size),
      stepStmts.append_some hr]
    exact hf
  · intro dtype shape name hname
    exact hregs dtype shape name (fun he => hname (by simp [he]))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores
