import bench.tritonbench_g.attn_fwd_causal.Proofs.RealMemory

/-! Source score preparation, through the center and probability registers.
The diagonal flag adds the finite penalty; it never removes a probability.
First-helper branch selection is handled separately from the arithmetic. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealPreparation
open VeriTile.Triton RealLoads RealMemory
open AttentionFwdTriton3.RealTiles (finite)
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 800000

def maskName (second : Bool) : RegName := if second then "mask" else "mask_first"

def prepare (BM BN BD N HA : Nat) (second diagonal : Bool) : List Stmt :=
  RealBody.loadPrefix (keyName second) (scaleName second) BD BN N HA ++
    [RealBody.score BM BN BD] ++
    (if diagonal then RealBody.diagonal BM BN (maskName second) else []) ++
    RealBody.center BM BN ++ [RealBody.probability BM BN]

def writes (second : Bool) : List RegName :=
  ["start_n","k_mask","k","k_scale","qk",maskName second,"m_ij","p"]

theorem prepares (s : BlockState) (c : Config) (BM BN BD start : Nat) (hN : 0 < BN)
    (second diagonal : Bool) (old : Fin BM → WithBot ℝ)
    (hq : s.regs .real [BM,BD] "q" = some (RealTiles.query (inputs s c BM BN BD) c.startM c.context c.activeHead))
    (hqs : s.regs .real [] "q_scale" = some (Tile.scalar (some (inputs s c BM BN BD).queryScale)))
    (hkp : s.regs .ptr [BD,BN] (keyName second) = some (keyPointer c BD BN start))
    (hsp : s.regs .ptr [] (scaleName second) = some (keyScalePointer c BN start))
    (hm : s.regs .nat [BM] "offs_m" = some ⟨fun i => c.startM*BM+i.1.val⟩)
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hold : s.regs .real [BM] "m_i" = some (⟨fun i => old i.1⟩ : Tile .real [BM]))
    (hu : ∀ r a, s.undef r a = 0) :
    let score := RealFormula.logit (inputs s c BM BN BD) c.startM c.context c.activeHead ⟨start,diagonal⟩
    ∃ t, stepStmts (prepare BM BN BD c.context c.activeHead second diagonal) s = some t ∧
      t.regs .real [BM] "m_ij" = some (⟨fun i => RealTiles.center score old i.1⟩ : Tile .real [BM]) ∧
      t.regs .real [BM,BN] "p" =
        some (finite (fun i : TileIndex [BM,BN] => RealTiles.probability score old i.1 i.2.1)) ∧
      t.regs .nat [] "start_n" = some (Tile.scalar start) ∧
      Frame (writes second) s t := by
  dsimp only
  let xs := inputs s c BM BN BD
  let score := RealFormula.logit xs c.startM c.context c.activeHead ⟨start,diagonal⟩
  obtain ⟨sl, hlstep, hlk, hlks, hlstart, hlf⟩ := loads_execute s c BM BN BD start second hkp hsp hn hs hu
  obtain ⟨ss, hsstep, hsq, hsf⟩ := RealNumeric.score_executes sl BM BN BD
    (fun i => if c.startM*BM+i.1.val < c.context ∧ i.2.1.val < c.activeHead then xs.query i.1 i.2.1 else 0)
    (fun i => if start+i.2.1.val < c.context ∧ i.1.val < c.activeHead then xs.key start i.2.1 i.1 else 0)
    xs.queryScale (xs.keyScale (start/BN))
    ((hlf.regs _ _ "q" (by simp)).trans hq) hlk
    ((hlf.regs _ _ "q_scale" (by simp)).trans hqs) hlks
  have hsf' : Frame (writes second) s ss :=
    (hlf.mono (by simp [writes])).trans (hsf.mono (by simp [writes]))
  have hsq' : ss.regs .real [BM,BN] "qk" = some (finite (fun i : TileIndex [BM,BN] =>
      RealFormula.score xs c.startM c.context c.activeHead ⟨start,diagonal⟩ i.1 i.2.1)) := hsq
  have hdr : ∃ sd,
      stepStmts (if diagonal then RealBody.diagonal BM BN (maskName second) else []) ss = some sd ∧
      sd.regs .real [BM,BN] "qk" = some (finite (fun i : TileIndex [BM,BN] => score i.1 i.2.1)) ∧
      Frame [maskName second,"qk"] ss sd := by
    cases diagonal with
    | «false» =>
      refine ⟨ss, by simp, ?_, Frame.refl _ _⟩
      simpa [score, RealFormula.logit, RealFormula.penalty] using hsq'
    | «true» =>
      obtain ⟨sd, hdstep, hdq, hdf⟩ := RealNumeric.diagonal_executes ss BM BN c.startM start
        (maskName second) (by cases second <;> decide)
        (RealFormula.score xs c.startM c.context c.activeHead ⟨start,Bool.true⟩) hsq'
        ((hsf'.regs _ _ "offs_m" (by cases second <;> simp [writes, maskName])).trans hm)
        ((hsf'.regs _ _ "offs_n" (by cases second <;> simp [writes, maskName])).trans hn)
        ((hsf.regs _ _ "start_n" (by simp)).trans hlstart)
      exact ⟨sd, hdstep, hdq, hdf⟩
  obtain ⟨sd, hdstep, hdq, hdf⟩ := hdr
  have hdf' : Frame (writes second) s sd := hsf'.trans (hdf.mono (by simp [writes]))
  obtain ⟨t, hpstep, hpm, hpp, hpf⟩ := RealNumeric.probabilities_execute sd BM BN hN score old
    ((hdf'.regs _ _ "m_i" (by cases second <;> simp [writes, maskName])).trans hold) hdq
  refine ⟨t, ?_, hpm, hpp,
    (hpf.regs _ _ "start_n" (by simp)).trans
      ((hdf.regs _ _ "start_n" (by cases second <;> simp [maskName])).trans
        ((hsf.regs _ _ "start_n" (by simp)).trans hlstart)),
    hdf'.trans (hpf.mono (by simp [writes]))⟩
  have hscore : stepStmts [RealBody.score BM BN BD] sl = some ss := by
    rw [stepStmts.cons_some hsstep, stepStmts.nil]
  have hpast := (stepStmts.append_some hlstep).trans hscore
  have hdiag := (stepStmts.append_some hpast).trans hdstep
  simpa only [prepare, List.append_assoc] using (stepStmts.append_some hdiag).trans hpstep

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealPreparation
