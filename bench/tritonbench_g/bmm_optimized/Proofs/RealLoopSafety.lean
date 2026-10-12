import bench.tritonbench_g.bmm_optimized.Proofs.RealAccess
import bench.tritonbench_g.bmm_optimized.Proofs.RealPureSafety

/-! Safety follows the actual loop invariant and host K-divisibility contract.
Bounds refer to initial matrix coordinates, rather than prepared registers or
assumed loop execution. Masked-off contraction lanes need no valid address. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealLoopSafety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

structure Bounds (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) (bounds : RegionBounds) : Prop where
  a : ∀ (i : Fin BM) (k : Nat), RealMath.active p.divM (RealSetup.rows s BM p i) p.M → k < p.K →
    RealSetup.aBase s p + RealSetup.rows s BM p i * p.K + k < bounds p.A
  b : ∀ (k : Nat) (j : Fin BN), k < p.K → RealMath.active p.divN (RealSetup.columns s BN p j) p.N →
    RealSetup.bBase s p + k * p.N + RealSetup.columns s BN p j < bounds p.B

theorem iteration_safe (swapped : Bool) (initial s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hBK : 0 < BK) (hdiv : p.divK = Bool.true → BK ∣ p.K)
    (bounds : RegionBounds) (hbounds : Bounds initial BM BN p bounds)
    (c : Nat) (hc : c < BmmChunkBwd.RealMath.blocks p.K BK) (h : RealLoop.Holds initial BM BN BK p c s) :
    Stmt.TraceSafeList bounds (RealLoads.body swapped BM BN BK p) (s.setReg "_i" .nat [] (Tile.scalar c)) := by
  let entry := s.setReg "_i" .nat [] (Tile.scalar c)
  have hm : p.divM = Bool.false → entry.regs .bool [BM] "mask_m" =
      some (Tile.vec fun i => decide (RealSetup.rows initial BM p i < p.M)) := by
    intro hdm
    have hv := h.fixed .bool [BM] "mask_m" (by decide)
    cases hdn : p.divN <;> simpa [entry, RealSetup.result, hdm, hdn] using hv
  have hn : p.divN = Bool.false → entry.regs .bool [BN] "mask_n" =
      some (Tile.vec fun j => decide (RealSetup.columns initial BN p j < p.N)) := by
    intro hdn
    simpa [entry, RealSetup.result, hdn] using h.fixed .bool [BN] "mask_n" (by decide)
  have hk : entry.regs .nat [BK] "offs_k" = some (Tile.vec fun k => c * BK + k.val) := by
    simpa [entry] using h.k
  have ha : entry.regs .ptr [BM,BK] "a_ptrs" = some (RealLoop.aPointers initial BM BK p c) := by
    simpa [entry] using h.a
  have hb : entry.regs .ptr [BK,BN] "b_ptrs" = some (RealLoop.bPointers initial BN BK p c) := by
    simpa [entry] using h.b
  have he := RealMasks.executes entry BM BN BK p (RealSetup.rows initial BM p)
    (RealSetup.columns initial BN p) (fun k => c * BK + k.val) hm hn hk
  rw [RealLoads.body, RealLoads.code, List.append_assoc]
  apply TraceProofs.append_safe _ _ (RealPureSafety.masks_safe BM BN BK p bounds entry)
  intro t ht
  rw [he] at ht
  cases ht
  simp only [List.cons_append, List.nil_append]
  apply TraceProofs.assign_cons
  · apply RealAccess.a_safe bounds entry BM BN BK p _ _ _ _ ha
    intro i k hi hkk
    exact hbounds.a i (c * BK + k.val) hi ((RealMath.contraction_mask p.K BK c p.divK hBK hdiv hc k).mp hkk)
  · intro av
    apply TraceProofs.assign_cons
    · apply RealAccess.b_safe bounds entry BM BN BK p _ _ _ _ av hb
      intro k j hkk hj
      exact hbounds.b (c * BK + k.val) j ((RealMath.contraction_mask p.K BK c p.divK hBK hdiv hc k).mp hkk) hj
    · intro _
      exact RealPureSafety.tail_safe swapped BM BN BK p bounds _

theorem loop_safe (swapped : Bool) (initial : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hBK : 0 < BK) (hdiv : p.divK = Bool.true → BK ∣ p.K) (hu : ∀ r o, initial.undef r o = 0)
    (bounds : RegionBounds) (hbounds : Bounds initial BM BN p bounds) :
    Stmt.TraceSafe bounds (RealLoop.statement swapped BM BN BK p) (RealSetup.result initial BM BN BK p) := by
  have hnum : evalOp (.ref .nat [] "num_iters") (RealSetup.result initial BM BN BK p) =
      some (Tile.scalar (BmmChunkBwd.RealMath.blocks p.K BK)) := by
    simp [RealSetup.result]
  rw [RealLoop.statement, Stmt.TraceSafe]
  simp only [Op.SafeAt.eq_def, true_and, evalOp_constNat, hnum, bind, Option.bind_some, Tile.scalar]
  apply TraceProofs.range_safe (P := RealLoop.Holds initial BM BN BK p)
  · intro c s hc h
    exact iteration_safe swapped initial s BM BN BK p hBK hdiv bounds hbounds c hc h
  · intro c s t hc h he
    obtain ⟨u,hu,hf⟩ := RealLoop.iteration swapped initial BM BN BK p hBK hdiv hu c s hc h
    rw [he] at hu
    cases hu
    exact hf
  · exact RealLoop.initial initial BM BN BK p

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealLoopSafety
