import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealSafetyPrologue
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealSafetyLoop

/-! Whole-program trace safety from explicit active-address bounds. Metadata
is derived from initial program IDs and source parameters. The INIT and END
conditions retain the exact optional carry-read and denominator-write behavior.
The next IO layer supplies these inequalities from its allocation windows. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealSafety
open VeriTile.Triton RealAccess RealPrologue
noncomputable section

structure Bounds (BM BN BD finish : Nat) (p : Parameters) (s : BlockState) (bounds : RegionBounds) : Prop
    extends RealSafetyPrologue.Bounds BM BN BD p s bounds where
  keys : ∀ j, j < RealLoop.count (loopParameters BM BN BD p s) →
    BlockBounds bounds (RealLoop.keyPointer (loopParameters BM BN BD p s) j) (RealAccumulation.checks p.evenN)
  values : ∀ j, j < RealLoop.count (loopParameters BM BN BD p s) →
    BlockBounds bounds (RealLoop.valuePointer (loopParameters BM BN BD p s) j) (RealAccumulation.checks p.evenN)
  writeCenter : ∀ i, RealExecution.rowAddress BM p s i < bounds p.layout.M
  writeDenominator : finish = 0 → ∀ i, RealExecution.rowAddress BM p s i < bounds p.layout.L
  writeOutput : ∀ i, RealExecution.outputAddress BM BD p s i < bounds p.layout.Out

def code (BM BN BD finish : Nat) (p : Parameters) (reciprocal : Bool) : List Stmt :=
  RealPrologue.code BM BN BD p ++
    [RealLoopSource.statement BM BN BD p.layout.kvContext p.window p.complement p.evenN p.offset p.size] ++
    RealEpilogue.code BM BD finish reciprocal

theorem complete_safe (BM BN BD finish : Nat) (p : Parameters) (reciprocal : Bool)
    (s : BlockState) (bounds : RegionBounds) (hN : 0 < BN)
    (hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0) (hb : Bounds BM BN BD finish p s bounds) :
    Stmt.TraceSafeList bounds (code BM BN BD finish p reciprocal) s := by
  let lp := loopParameters BM BN BD p s
  obtain ⟨sp, ep, hp, hl, hh, _, ho⟩ := RealPrologue.executes s BM BN BD p
  obtain ⟨sf, ef, hf, ff⟩ := RealLoopSource.statement_executes sp lp (seed BM BD p s) hN hflag rfl
    (RealWindow.upper_matches _ _ _ _ _ _ _) hl hh hp
  have hloop : Stmt.TraceSafe bounds
      (RealLoopSource.statement BM BN BD p.layout.kvContext p.window p.complement p.evenN p.offset p.size) sp := by
    simp only [RealLoopSource.statement, Stmt.TraceSafe, Op.SafeAt.eq_def, true_and,
      evalOp.eq_def, hl, hh, Tile.scalar]
    exact RealSafetyLoop.loop_safe bounds sp lp (seed BM BD p s) hN hb.keys hb.values hp
  have hm : sf.regs .ptr [BM] "m_ptrs" = some
      (⟨fun i => (p.layout.M, RealExecution.rowAddress BM p s i)⟩ : Tile .ptr [BM]) :=
    (ff.regs _ _ _ (by simp [RealLoop.writes])).trans ho.center
  have hL : sf.regs .ptr [BM] "l_ptrs" = some
      (⟨fun i => (p.layout.L, RealExecution.rowAddress BM p s i)⟩ : Tile .ptr [BM]) :=
    (ff.regs _ _ _ (by simp [RealLoop.writes])).trans ho.denominator
  have hO : sf.regs .blockPtr [BM, BD] "O_block_ptr" = some
      (⟨fun _ => RealExecution.outputPointer BM BD p s⟩ : Tile .blockPtr [BM, BD]) :=
    (ff.regs _ _ _ (by simp [RealLoop.writes])).trans ho.output
  simp only [code, List.append_assoc, List.singleton_append]
  apply append_of_run (RealSafetyPrologue.prologue_safe BM BN BD p s bounds hb.toBounds) ep
  apply cons_of_run hloop ef
  exact RealMemorySafety.epilogue_safe bounds sf BM BD finish reciprocal _ p.layout.M p.layout.L
    (RealExecution.rowAddress BM p s) (RealExecution.outputPointer BM BD p s)
    ⟨hf.center, hf.denominator, hf.numerator⟩ hm hL hO hb.writeCenter hb.writeDenominator hb.writeOutput

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealSafety
