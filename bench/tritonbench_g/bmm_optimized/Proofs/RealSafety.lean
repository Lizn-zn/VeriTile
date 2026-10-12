import bench.tritonbench_g.bmm_optimized.Proofs.RealLoopSafety
import bench.tritonbench_g.bmm_optimized.Proofs.RealExecution

/-! Whole-source address safety: actual prefix, K loop and postlude. Input
and output bounds apply only to reached active lanes. Both CTA reorder
branches and every constexpr mask combination remain in the source. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealSafety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

structure Bounds (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) (bounds : RegionBounds) : Prop where
  input : RealLoopSafety.Bounds s BM BN p bounds
  output : ∀ (i : Fin BM) (j : Fin BN),
    RealMath.active p.divM (RealSetup.rows s BM p i) p.M →
    RealMath.active p.divN (RealSetup.columns s BN p j) p.N →
    RealSetup.oBase s p + RealSetup.rows s BM p i * p.N + RealSetup.columns s BN p j < bounds p.O

theorem safe (swapped : Bool) (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hBK : 0 < BK) (hdiv : p.divK = Bool.true → BK ∣ p.K) (hu : ∀ r o, s.undef r o = 0)
    (bounds : RegionBounds) (hbounds : Bounds s BM BN p bounds) :
    (RealProgram.kernel swapped BM BN BK p).toAlgKernel.TraceSafe bounds s := by
  obtain ⟨t,hprefix,h⟩ := RealLoop.prefix_executes swapped s BM BN BK p hBK hdiv hu
  have hpreSafe : Stmt.TraceSafeList bounds
      (RealProgram.realPart (RealProgram.parts swapped BM BN BK p).before ++
        [.forRangeDyn "_i" (.constNat 0) (.ref .nat [] "num_iters") (.constNat 1)
          (RealProgram.realPart (RealProgram.parts swapped BM BN BK p).body)]) s := by
    rw [RealSetup.original_candidate_setup, RealLoads.matches_source]
    apply TraceProofs.append_safe _ _ (RealPureSafety.setup_safe BM BN BK p bounds s)
    intro u he
    rw [RealSetup.executes] at he
    cases he
    apply Stmt.TraceSafeList.cons_intro
      (RealLoopSafety.loop_safe swapped s BM BN BK p hBK hdiv hu bounds hbounds.input)
    intro _ _
    exact Stmt.TraceSafeList.nil_intro
  have hm : p.divM = Bool.false → t.regs .bool [BM] "mask_m" =
      some (Tile.vec fun i => decide (RealSetup.rows s BM p i < p.M)) := by
    intro hdm
    have hv := h.fixed .bool [BM] "mask_m" (by decide)
    cases hdn : p.divN <;> simpa [RealSetup.result, hdm, hdn] using hv
  have hn : p.divN = Bool.false → t.regs .bool [BN] "mask_n" =
      some (Tile.vec fun j => decide (RealSetup.columns s BN p j < p.N)) := by
    intro hdn
    simpa [RealSetup.result, hdn] using h.fixed .bool [BN] "mask_n" (by decide)
  have hp : t.regs .ptr [BM,BN] "o_ptrs" = some (RealSetup.oPointers s BM BN p) := by
    simpa [RealSetup.result] using h.fixed .ptr [BM,BN] "o_ptrs" (by decide)
  rw [Kernel.TraceSafe, RealProgram.body_split]
  apply TraceProofs.append_safe _ _ hpreSafe
  intro u he
  rw [hprefix] at he
  cases he
  rw [RealStore.matches_source, RealStore.code]
  apply TraceProofs.append_safe _ _ (RealPureSafety.store_masks_safe BM BN p bounds t)
  intro v hv
  rw [RealStore.prepares t BM BN p (RealSetup.rows s BM p) (RealSetup.columns s BN p) hm hn] at hv
  cases hv
  apply Stmt.TraceSafeList.cons_intro
  · exact RealAccess.output_safe bounds t BM BN p (RealSetup.oPointers s BM BN p)
      (RealSetup.rows s BM p) (RealSetup.columns s BN p) hp hbounds.output
  · intro _ _
    exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealSafety
