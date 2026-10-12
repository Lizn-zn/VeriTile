import bench.tritonbench_g.bmm_optimized.Proofs.RealStore

/-! Successful execution, matrix-product readback and preservation of every
other tagged memory cell for the entire source. Flat allocation/safety is
proved separately; this layer does not assume successful execution. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealExecution
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def values (s₀ : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) :=
  RealLoop.carry s₀ BM BN BK p (BmmChunkBwd.RealMath.blocks p.K BK)

def finished (s₀ t : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) : BlockState :=
  RealStore.write (RealStore.prepared t p (RealSetup.rows s₀ BM p) (RealSetup.columns s₀ BN p))
    p (RealSetup.oBase s₀ p) (RealSetup.rows s₀ BM p) (RealSetup.columns s₀ BN p) (values s₀ BM BN BK p)

theorem kernel_executes (swapped : Bool) (s₀ : BlockState) (BM BN BK : Nat)
    (p : RealProgram.Parameters) (hBK : 0 < BK) (hdiv : p.divK = Bool.true → BK ∣ p.K)
    (hu : ∀ r o, s₀.undef r o = 0) :
    ∃ t, RealLoop.Holds s₀ BM BN BK p (BmmChunkBwd.RealMath.blocks p.K BK) t ∧
      stepStmts (RealProgram.kernel swapped BM BN BK p).toAlgKernel.body s₀ =
        some (finished s₀ t BM BN BK p) := by
  obtain ⟨t,hprefix,h⟩ := RealLoop.prefix_executes swapped s₀ BM BN BK p hBK hdiv hu
  have hm : p.divM = Bool.false → t.regs .bool [BM] "mask_m" =
      some (Tile.vec fun i => decide (RealSetup.rows s₀ BM p i < p.M)) := by
    intro hdm
    have hv := h.fixed .bool [BM] "mask_m" (by decide)
    cases hdn : p.divN <;> simpa [RealSetup.result, hdm, hdn] using hv
  have hn : p.divN = Bool.false → t.regs .bool [BN] "mask_n" =
      some (Tile.vec fun j => decide (RealSetup.columns s₀ BN p j < p.N)) := by
    intro hdn
    have hv := h.fixed .bool [BN] "mask_n" (by decide)
    simpa [RealSetup.result, hdn] using hv
  have hp : t.regs .ptr [BM,BN] "o_ptrs" = some (RealSetup.oPointers s₀ BM BN p) := by
    have hv := h.fixed .ptr [BM,BN] "o_ptrs" (by decide)
    simpa [RealSetup.result] using hv
  refine ⟨t,h,?_⟩
  rw [RealProgram.body_split, stepStmts.append_some hprefix, RealStore.matches_source]
  exact RealStore.executes t BM BN p (RealSetup.oBase s₀ p) (RealSetup.rows s₀ BM p)
    (RealSetup.columns s₀ BN p) (values s₀ BM BN BK p) hm hn hp h.acc

theorem value_formula (s₀ : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hBK : 0 < BK) (i : Fin BM) (j : Fin BN)
    (hi : RealMath.active p.divM (RealSetup.rows s₀ BM p i) p.M ∧
      RealMath.active p.divN (RealSetup.columns s₀ BN p j) p.N) :
    values s₀ BM BN BK p i j =
      RealMath.contraction p.K (RealLoop.aEntry s₀ BM p i) (RealLoop.bEntry s₀ BN p j) := by
  simp only [values, RealLoop.carry, RealMath.partial_complete _ _ _ _ _ _ _ _ hBK, hi, true_and, ↓reduceIte]

theorem readback (s₀ t : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hBK : 0 < BK) (i : TileIndex [BM,BN])
    (hi : RealStore.active p (RealSetup.rows s₀ BM p) (RealSetup.columns s₀ BN p) i)
    (hinj : ∀ a b, RealStore.active p (RealSetup.rows s₀ BM p) (RealSetup.columns s₀ BN p) a →
      RealStore.active p (RealSetup.rows s₀ BM p) (RealSetup.columns s₀ BN p) b →
      RealStore.offset (RealSetup.oBase s₀ p) p.N (RealSetup.rows s₀ BM p) (RealSetup.columns s₀ BN p) a =
      RealStore.offset (RealSetup.oBase s₀ p) p.N (RealSetup.rows s₀ BM p) (RealSetup.columns s₀ BN p) b → a = b) :
    (finished s₀ t BM BN BK p).mem p.O
      (RealStore.offset (RealSetup.oBase s₀ p) p.N (RealSetup.rows s₀ BM p) (RealSetup.columns s₀ BN p) i) =
      MemCell.real (RealMath.contraction p.K (RealLoop.aEntry s₀ BM p i.1) (RealLoop.bEntry s₀ BN p i.2.1)) := by
  rw [finished, RealStore.readback _ _ _ _ _ _ i hi hinj, value_formula s₀ BM BN BK p hBK i.1 i.2.1 hi]

theorem frame (s₀ t : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hm : t.mem = s₀.mem) (r : RegionName) (address : Nat)
    (h : ∀ i, RealStore.active p (RealSetup.rows s₀ BM p) (RealSetup.columns s₀ BN p) i →
      ¬ (p.O = r ∧ RealStore.offset (RealSetup.oBase s₀ p) p.N
        (RealSetup.rows s₀ BM p) (RealSetup.columns s₀ BN p) i = address)) :
    (finished s₀ t BM BN BK p).mem r address = s₀.mem r address := by
  rw [finished, RealStore.frame _ _ _ _ _ _ r address h]
  have hf := congrArg BlockState.mem
    (RealStore.prepared_frame t p (RealSetup.rows s₀ BM p) (RealSetup.columns s₀ BN p))
  change (RealStore.prepared t p (RealSetup.rows s₀ BM p) (RealSetup.columns s₀ BN p)).mem = t.mem at hf
  rw [hf, hm]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealExecution
