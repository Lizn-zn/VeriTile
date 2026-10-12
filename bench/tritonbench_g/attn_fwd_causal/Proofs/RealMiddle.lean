import bench.tritonbench_g.attn_fwd_causal.Proofs.RealMiddleSource

/-! Complete execution from the loaded query/scale and initial carry through
both helpers. The caller pointer contract precedes local helper setup; no
loop-output or successful-execution premise is required. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMiddle
open VeriTile.Triton RealLoads RealLoopFormula
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section

structure Entry {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config) (s : BlockState) : Prop where
  query : s.regs .real [BM,BD] "q" = some (RealTiles.query xs c.startM c.context c.activeHead)
  scale : s.regs .real [] "q_scale" = some (Tile.scalar (some xs.queryScale))
  rows : s.regs .nat [BM] "offs_m" = some ⟨fun i => c.startM*BM+i.1.val⟩
  keys : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val))
  carry : RealIteration.Carries s (initial BM BD)
  caller : RealTwoLoops.Caller s c BN BD
  undef : ∀ r a, s.undef r a = 0

def code (BM BN BD N HA HD : Nat) (diagonal : Bool) : List Stmt :=
  RealFirstSetup.code BM BN BD HD diagonal ++ RealTwoLoops.code BM BN BD N HA HD diagonal

def writes : List RegName := RealSetup.writes Bool.false ++ RealTwoLoops.writes

theorem executes (s : BlockState) (c : Config) (BM BN BD : Nat) (hN : 0 < BN) (diagonal : Bool)
    (entry : Entry (inputs s c BM BN BD) c s) :
    ∃ t, stepStmts (code BM BN BD c.context c.activeHead c.headDim diagonal) s = some t ∧
      RealIteration.Carries t (bothResult (inputs s c BM BN BD) c diagonal) ∧ Frame writes s t := by
  obtain ⟨sc, hc, hlo, hhi, hp, hf⟩ := RealFirstSetup.executes s c BM BN BD diagonal entry.caller
  have hmem := inputs_same_memory sc s c BM BN BD hf.mem
  have caller : RealTwoLoops.Caller sc c BN BD :=
    ⟨(hf.regs _ _ "start_m" (by decide)).trans entry.caller.row,
     (hf.regs _ _ "K_ptrs" (by decide)).trans entry.caller.key,
     (hf.regs _ _ "K_scale_ptr" (by decide)).trans entry.caller.scale,
     (hf.regs _ _ "V_ptrs" (by decide)).trans entry.caller.value⟩
  have hr : RealLoop.Registers (inputs sc c BM BN BD) c (initial BM BD)
      (firstLower c.startM BM diagonal) Bool.false sc := by
    rw [hmem]
    refine ⟨(hf.regs _ _ "q" (by decide)).trans entry.query,
      (hf.regs _ _ "q_scale" (by decide)).trans entry.scale,
      (hf.regs _ _ "offs_m" (by decide)).trans entry.rows,
      (hf.regs _ _ "offs_n" (by decide)).trans entry.keys, ?_, hp, ?_⟩
    · exact ⟨(hf.regs _ _ "m_i" (by decide)).trans entry.carry.center,
        (hf.regs _ _ "l_i" (by decide)).trans entry.carry.denominator,
        (hf.regs _ _ "acc" (by decide)).trans entry.carry.numerator⟩
    · rw [hf.undef]; exact entry.undef
  obtain ⟨t, ht, hcarry, _, hframe⟩ := RealTwoLoops.executes sc c BM BN BD hN diagonal caller hlo hhi hr
  rw [hmem] at hcarry
  exact ⟨t, (stepStmts.append_some hc).trans ht, hcarry,
    (hf.mono (by simp [writes])).trans (hframe.mono (by simp [writes]))⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMiddle
