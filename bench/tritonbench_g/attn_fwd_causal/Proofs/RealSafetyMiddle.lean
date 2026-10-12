import bench.tritonbench_g.attn_fwd_causal.Proofs.RealSafetyLoop
import bench.tritonbench_g.attn_fwd_causal.Proofs.RealSafetyPrologue

/-! Compose both helper ranges and their actual pointer reset. Allocation
windows are restricted to the visited tiles in each stage, including stage 2's
repeated diagonal. The second helper uses preserved caller pointers. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyMiddle
open VeriTile.Triton RealLoads RealLoopFormula
open AttentionFwdTriton3.RealAccess
noncomputable section
set_option maxHeartbeats 1000000

theorem first_bounds (bounds : RegionBounds) (c : Config) (Out : RegionName) (BM BN BD : Nat)
    (diagonal : Bool) (h : RealAccessSafety.Windows c Out BM BN BD (if diagonal then 2 else 3) bounds) :
    RealSafetyLoop.Bounds bounds c BN BD (firstLower c.startM BM diagonal) (firstUpper c.startM BM diagonal) := by
  apply RealSafetyLoop.bounds_of_visits bounds c Out BM BN BD (if diagonal then 2 else 3)
    _ _ diagonal h
  cases diagonal <;> simp [firstLower, firstUpper, RealFormula.allVisits, RealFormula.firstVisits]

theorem second_bounds (bounds : RegionBounds) (c : Config) (Out : RegionName) (BM BN BD stage : Nat)
    (h : RealAccessSafety.Windows c Out BM BN BD stage bounds) :
    RealSafetyLoop.Bounds bounds c BN BD (c.startM*BM) ((c.startM+1)*BM) := by
  apply RealSafetyLoop.bounds_of_visits bounds c Out BM BN BD stage _ _ Bool.true h
  simp [RealFormula.allVisits, RealFormula.secondVisits]

theorem two_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (BM BN BD : Nat)
    (hN : 0 < BN) (diagonal : Bool)
    (hb1 : RealSafetyLoop.Bounds bounds c BN BD (firstLower c.startM BM diagonal) (firstUpper c.startM BM diagonal))
    (hb2 : RealSafetyLoop.Bounds bounds c BN BD (c.startM*BM) ((c.startM+1)*BM))
    (caller : RealTwoLoops.Caller s c BN BD)
    (hl : s.regs .nat [] "first_lo" = some (Tile.scalar (firstLower c.startM BM diagonal)))
    (hh : s.regs .nat [] "first_hi" = some (Tile.scalar (firstUpper c.startM BM diagonal)))
    (entry : RealLoop.Registers (inputs s c BM BN BD) c (initial BM BD)
      (firstLower c.startM BM diagonal) Bool.false s) :
    Stmt.TraceSafeList bounds (RealTwoLoops.code BM BN BD c.context c.activeHead c.headDim diagonal) s := by
  obtain ⟨s1, h1, hr1, hf1⟩ := RealSetupSource.statement_executes s c
    (firstLower c.startM BM diagonal) (firstUpper c.startM BM diagonal) Bool.false diagonal
    (initial BM BD) hN hl hh entry
  have hs1 := RealSafetyLoop.statement_safe bounds s c
    (firstLower c.startM BM diagonal) (firstUpper c.startM BM diagonal) Bool.false diagonal
    (initial BM BD) hN hb1 hl hh entry
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
  rw [RealTwoLoops.code, List.singleton_append]
  apply cons_of_run hs1 h1
  apply append_of_run (RealSafetyPrologue.setup_safe BM BN BD c.headDim Bool.true bounds s1) hr
  apply Stmt.TraceSafeList.cons_intro
    (RealSafetyLoop.statement_safe bounds sr c (c.startM*BM) ((c.startM+1)*BM) Bool.true diagonal
      (firstResult (inputs s c BM BN BD) c diagonal) hN hb2 hlo hhi secondEntry)
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

theorem safe (bounds : RegionBounds) (s : BlockState) (c : Config) (Out : RegionName) (BM BN BD : Nat)
    (hN : 0 < BN) (diagonal : Bool)
    (hb : RealAccessSafety.Windows c Out BM BN BD (if diagonal then 2 else 3) bounds)
    (entry : RealMiddle.Entry (inputs s c BM BN BD) c s) :
    Stmt.TraceSafeList bounds (RealMiddle.code BM BN BD c.context c.activeHead c.headDim diagonal) s := by
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
  exact append_of_run (RealSafetyPrologue.first_setup_safe BM BN BD c.headDim diagonal bounds s) hc
    (two_safe bounds sc c BM BN BD hN diagonal (first_bounds bounds c Out BM BN BD diagonal hb)
      (second_bounds bounds c Out BM BN BD _ hb) caller hlo hhi hr)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyMiddle
