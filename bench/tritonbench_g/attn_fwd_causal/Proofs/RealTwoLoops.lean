import bench.tritonbench_g.attn_fwd_causal.Proofs.RealSetupSource

/-! Execute the two actual dynamic ranges and the source reset between them.
The entry contract is established by the still-separate prologue proof. The
result is the independent two-range carry, before normalization and storage. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealTwoLoops
open VeriTile.Triton RealLoads RealMemory RealLoopFormula
open RealIteration (Carries Pointers)
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 1000000

structure Caller (s : BlockState) (c : Config) (BN BD : Nat) : Prop where
  row : s.regs .nat [] "start_m" = some (Tile.scalar c.startM)
  key : s.regs .ptr [BD,BN] "K_ptrs" = some (keyPointer c BD BN 0)
  scale : s.regs .ptr [] "K_scale_ptr" = some (keyScalePointer c BN 0)
  value : s.regs .ptr [BN,BD] "V_ptrs" = some (valuePointer c BN BD 0)

def code (BM BN BD N HA HD : Nat) (diagonal : Bool) : List Stmt :=
  [RealSetupSource.statement BM BN BD N HA HD Bool.false diagonal] ++
    RealSetup.code BM BN BD HD Bool.true ++
    [RealSetupSource.statement BM BN BD N HA HD Bool.true diagonal]

def writes : List RegName :=
  RealIteration.writes Bool.false ++ RealSetup.writes Bool.true ++ RealIteration.writes Bool.true

theorem executes (s : BlockState) (c : Config) (BM BN BD : Nat) (hN : 0 < BN) (diagonal : Bool)
    (caller : Caller s c BN BD)
    (hl : s.regs .nat [] "first_lo" = some (Tile.scalar (firstLower c.startM BM diagonal)))
    (hh : s.regs .nat [] "first_hi" = some (Tile.scalar (firstUpper c.startM BM diagonal)))
    (entry : RealLoop.Registers (inputs s c BM BN BD) c (initial BM BD)
      (firstLower c.startM BM diagonal) Bool.false s) :
    ∃ t, stepStmts (code BM BN BD c.context c.activeHead c.headDim diagonal) s = some t ∧
      Carries t (bothResult (inputs s c BM BN BD) c diagonal) ∧
      RealLoop.Registers (inputs s c BM BN BD) c (bothResult (inputs s c BM BN BD) c diagonal)
        (RealLoop.cursor (c.startM*BM) BN (RealFormula.tripCount (c.startM*BM) ((c.startM+1)*BM) BN)) Bool.true t ∧
      Frame writes s t := by
  obtain ⟨s1, h1, hr1, hf1⟩ := RealSetupSource.statement_executes s c
    (firstLower c.startM BM diagonal) (firstUpper c.startM BM diagonal) Bool.false diagonal
    (initial BM BD) hN hl hh entry
  obtain ⟨sr, hr, hlo, hhi, hptr, hfr⟩ := RealSetup.executes s1 c BM BN BD Bool.true
    ((hf1.regs _ _ "start_m" (by decide)).trans caller.row)
    ((hf1.regs _ _ "K_ptrs" (by decide)).trans caller.key)
    ((hf1.regs _ _ "K_scale_ptr" (by decide)).trans caller.scale)
    ((hf1.regs _ _ "V_ptrs" (by decide)).trans caller.value)
  have hrmem : inputs sr c BM BN BD = inputs s c BM BN BD :=
    inputs_same_memory sr s c BM BN BD (hfr.mem.trans hf1.mem)
  have secondEntry : RealLoop.Registers (inputs sr c BM BN BD) c
      (firstResult (inputs s c BM BN BD) c diagonal) (c.startM*BM) Bool.true sr := by
    rw [hrmem]
    refine ⟨(hfr.regs _ _ "q" (by decide)).trans hr1.query,
      (hfr.regs _ _ "q_scale" (by decide)).trans hr1.queryScale,
      (hfr.regs _ _ "offs_m" (by decide)).trans hr1.rows,
      (hfr.regs _ _ "offs_n" (by decide)).trans hr1.keys, ?_, hptr, ?_⟩
    · exact ⟨(hfr.regs _ _ "m_i" (by decide)).trans hr1.carry.center,
        (hfr.regs _ _ "l_i" (by decide)).trans hr1.carry.denominator,
        (hfr.regs _ _ "acc" (by decide)).trans hr1.carry.numerator⟩
    · rw [hfr.undef]; exact hr1.undef
  obtain ⟨t, h2, hr2, hf2⟩ := RealSetupSource.statement_executes sr c (c.startM*BM) ((c.startM+1)*BM)
    Bool.true diagonal (firstResult (inputs s c BM BN BD) c diagonal) hN hlo hhi secondEntry
  rw [hrmem] at hr2
  refine ⟨t, ?_, hr2.carry, hr2, ?_⟩
  · have hfirst : stepStmts [RealSetupSource.statement BM BN BD c.context c.activeHead c.headDim Bool.false diagonal] s = some s1 := by
      rw [stepStmts.cons_some h1, stepStmts.nil]
    have hpast := (stepStmts.append_some hfirst).trans hr
    exact (stepStmts.append_some hpast).trans ((stepStmts.cons_some h2).trans stepStmts.nil)
  · exact ((hf1.mono (by simp [writes])).trans (hfr.mono (by simp [writes]))).trans
      (hf2.mono (by simp [writes]))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealTwoLoops
