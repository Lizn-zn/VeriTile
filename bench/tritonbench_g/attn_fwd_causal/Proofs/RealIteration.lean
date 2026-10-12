import bench.tritonbench_g.attn_fwd_causal.Proofs.RealPreparation
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealLoopFormula

/-! One complete helper iteration, from actual K/V/scale memory loads to the
online carry and all three advanced pointers. The flat body below only selects
the constant first-helper branch. Source transport at the end checks both
original helper bodies. Address bounds and full-kernel IO are separate. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealIteration
open VeriTile.Triton RealLoads RealMemory
open AttentionFwdTriton3.RealTiles (finite pairs dot_finite)
open AttentionFwdTriton3.RealScores (Frame)
open AttentionFwdTriton3.RealLoopFormula (Carry update)
noncomputable section
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

def body (BM BN BD N HA HD : Nat) (second diagonal : Bool) : List Stmt :=
  RealPreparation.prepare BM BN BD N HA second diagonal ++ RealBody.rescale BM BN BD ++
    [RealBody.loadValue (valueName second) BN BD N HA] ++ RealBody.accumulate BM BN BD ++
    RealBody.advance (keyName second) (scaleName second) (valueName second) BN BD HD

def updated {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config)
    (v : RealFormula.Visit) (carry : Carry BM BD) : Carry BM BD :=
  update carry (RealFormula.logit xs c.startM c.context c.activeHead v) RealTiles.keep
    (fun i => RealFormula.loadedValue xs c.context c.activeHead v i.1 i.2.1)

structure Carries (s : BlockState) {BM BD : Nat} (carry : Carry BM BD) : Prop where
  center : s.regs .real [BM] "m_i" = some (⟨fun i => carry.center i.1⟩ : Tile .real [BM])
  denominator : s.regs .real [BM] "l_i" = some (finite (fun i : TileIndex [BM] => carry.denominator i.1))
  numerator : s.regs .real [BM,BD] "acc" = some (finite carry.numerator)

structure Pointers (s : BlockState) (c : Config) (BN BD start : Nat) (second : Bool) : Prop where
  key : s.regs .ptr [BD,BN] (keyName second) = some (keyPointer c BD BN start)
  scale : s.regs .ptr [] (scaleName second) = some (keyScalePointer c BN start)
  value : s.regs .ptr [BN,BD] (valueName second) = some (valuePointer c BN BD start)

def writes (second : Bool) : List RegName :=
  RealPreparation.writes second ++ ["l_ij","alpha","l_i","acc","v","m_i",
    keyName second,scaleName second,valueName second]

theorem executes (s : BlockState) (c : Config) (BM BN BD start : Nat) (hN : 0 < BN)
    (second diagonal : Bool) (carry : Carry BM BD)
    (hq : s.regs .real [BM,BD] "q" = some (RealTiles.query (inputs s c BM BN BD) c.startM c.context c.activeHead))
    (hqs : s.regs .real [] "q_scale" = some (Tile.scalar (some (inputs s c BM BN BD).queryScale)))
    (hm : s.regs .nat [BM] "offs_m" = some ⟨fun i => c.startM*BM+i.1.val⟩)
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hc : Carries s carry) (hp : Pointers s c BN BD start second)
    (hu : ∀ r a, s.undef r a = 0) :
    ∃ t, stepStmts (body BM BN BD c.context c.activeHead c.headDim second diagonal) s = some t ∧
      Carries t (updated (inputs s c BM BN BD) c ⟨start,diagonal⟩ carry) ∧
      Pointers t c BN BD (start+BN) second ∧ Frame (writes second) s t := by
  let xs := inputs s c BM BN BD
  let v : RealFormula.Visit := ⟨start,diagonal⟩
  let score := RealFormula.logit xs c.startM c.context c.activeHead v
  let values : TileIndex [BN,BD] → ℝ := fun i => RealFormula.loadedValue xs c.context c.activeHead v i.1 i.2.1
  obtain ⟨sp, hpstep, hcenter, hprob, hstart, hpf⟩ := RealPreparation.prepares s c BM BN BD start hN
    second diagonal carry.center hq hqs hp.key hp.scale hm hn hs hc.center hu
  obtain ⟨sr, hrstep, hrden, hracc, hrf⟩ := RealNumeric.rescale_executes sp BM BN BD
    (⟨fun i => carry.center i.1⟩ : Tile .real [BM])
    (⟨fun i => RealTiles.center score carry.center i.1⟩ : Tile .real [BM])
    (finite (fun i : TileIndex [BM] => carry.denominator i.1))
    (finite (fun i : TileIndex [BM,BN] => RealTiles.probability score carry.center i.1 i.2.1))
    (finite carry.numerator)
    ((hpf.regs _ _ "m_i" (by cases second <;> simp [RealPreparation.writes, RealPreparation.maskName])).trans hc.center)
    hcenter
    ((hpf.regs _ _ "l_i" (by cases second <;> simp [RealPreparation.writes, RealPreparation.maskName])).trans hc.denominator)
    hprob
    ((hpf.regs _ _ "acc" (by cases second <;> simp [RealPreparation.writes, RealPreparation.maskName])).trans hc.numerator)
  rw [RealTiles.alpha_matches score carry.center] at hrden hracc
  have hdenFormula : sr.regs .real [BM] "l_i" =
      some (finite (fun i : TileIndex [BM] => (updated xs c v carry).denominator i.1)) := by
    refine hrden.trans (congrArg some ?_)
    apply Tile.ext
    rintro ⟨row,⟨⟩⟩
    exact AttentionFwdTriton3.RealTiles.denominator_matches BM BN score RealTiles.keep
      carry.center carry.denominator row (fun _ => 0) 0
  let scaled := fun i : TileIndex [BM,BD] => carry.numerator i *
    AttentionFwdTriton3.RealFormula.weight (carry.center i.1) (RealTiles.center score carry.center i.1)
  have haccFormula : sr.regs .real [BM,BD] "acc" = some (finite scaled) := by
    exact hracc.trans (congrArg some (by apply Tile.ext; rintro ⟨row,d,⟨⟩⟩; rfl))
  have hrbase : Frame (RealPreparation.writes second ++ ["l_ij","alpha","l_i","acc"]) s sr :=
    (hpf.mono (by simp)).trans (hrf.mono (by simp))
  obtain ⟨sv, hvstep, hvval, hvf⟩ := value_executes sr c BM BN BD start diagonal second
    ((hrbase.regs _ _ (valueName second)
      (by cases second <;> simp [RealPreparation.writes, RealPreparation.maskName, valueName])).trans hp.value)
    ((hrbase.regs _ _ "offs_n" (by cases second <;> simp [RealPreparation.writes, RealPreparation.maskName])).trans hn)
    ((hrf.regs _ _ "start_n" (by simp)).trans hstart)
    (by rw [hrbase.undef]; exact hu)
  rw [inputs_same_memory sr s c BM BN BD hrbase.mem] at hvval
  obtain ⟨sa, haste, hanum, ham, haf⟩ := RealNumeric.accumulate_executes sv BM BN BD
    (fun i : TileIndex [BM,BN] => RealTiles.probability score carry.center i.1 i.2.1)
    values scaled (⟨fun i => RealTiles.center score carry.center i.1⟩ : Tile .real [BM])
    ((hvf.regs _ _ "p" (by simp)).trans ((hrf.regs _ _ "p" (by simp)).trans hprob)) hvval
    ((hvf.regs _ _ "acc" (by simp)).trans haccFormula)
    ((hvf.regs _ _ "m_ij" (by simp)).trans ((hrf.regs _ _ "m_ij" (by simp)).trans hcenter))
  have hnumFormula : sa.regs .real [BM,BD] "acc" = some (finite (updated xs c v carry).numerator) := by
    refine hanum.trans (congrArg some ?_)
    apply Tile.ext
    rintro ⟨row,d,⟨⟩⟩
    have h := AttentionFwdTriton3.RealTiles.numerator_matches BM BN BD score RealTiles.keep
      carry.center carry.numerator values row d (carry.denominator row)
    rw [dot_finite] at h
    exact h
  have habase : Frame (RealPreparation.writes second ++ ["l_ij","alpha","l_i","acc","v","p","m_i"]) s sa :=
    ((hrbase.mono (by simp)).trans (hvf.mono (by simp))).trans (haf.mono (by simp))
  obtain ⟨t, htstep, htk, hts, htv, htf⟩ := advances_execute sa c BN BD start hN second
    ((habase.regs _ _ (keyName second)
      (by cases second <;> simp [RealPreparation.writes, RealPreparation.maskName, keyName])).trans hp.key)
    ((habase.regs _ _ (scaleName second)
      (by cases second <;> simp [RealPreparation.writes, RealPreparation.maskName, scaleName])).trans hp.scale)
    ((habase.regs _ _ (valueName second)
      (by cases second <;> simp [RealPreparation.writes, RealPreparation.maskName, valueName])).trans hp.value)
  refine ⟨t, ?_, ⟨?_, ?_, ?_⟩, ⟨htk,hts,htv⟩,
    (habase.mono (by simp [writes, RealPreparation.writes])).trans (htf.mono (by simp [writes]))⟩
  · have hvlist : stepStmts [RealBody.loadValue (valueName second) BN BD c.context c.activeHead] sr = some sv := by
      rw [stepStmts.cons_some hvstep, stepStmts.nil]
    have hrescale := (stepStmts.append_some hpstep).trans hrstep
    have hvalue := (stepStmts.append_some hrescale).trans hvlist
    have hacc := (stepStmts.append_some hvalue).trans haste
    exact (stepStmts.append_some hacc).trans htstep
  · exact (htf.regs _ _ "m_i" (by cases second <;> decide)).trans ham
  · exact (htf.regs _ _ "l_i" (by cases second <;> decide)).trans
      ((haf.regs _ _ "l_i" (by simp)).trans ((hvf.regs _ _ "l_i" (by simp)).trans hdenFormula))
  · exact (htf.regs _ _ "acc" (by cases second <;> decide)).trans hnumFormula

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealIteration
