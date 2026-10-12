import bench.tritonbench_g.bmm_optimized.Proofs.RealStore
import VeriTile.Triton.Memory.TraceProofs

/-! Bounds for actual reached load addresses, retaining all mask-elision and
broadcast cases. No bounds are required for a masked-off lane. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealAccess
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

theorem a_safe (bounds : RegionBounds) (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (ptrs : Tile .ptr [BM,BK]) (rows : Fin BM → Nat) (cols : Fin BN → Nat) (ks : Fin BK → Nat)
    (hp : s.regs .ptr [BM,BK] "a_ptrs" = some ptrs)
    (hb : ∀ (i : Fin BM) (k : Fin BK), RealMath.active p.divM (rows i) p.M →
      RealMath.active p.divK (ks k) p.K →
      (ptrs.data (i,k,PUnit.unit)).2 < bounds (ptrs.data (i,k,PUnit.unit)).1) :
    (RealLoads.aLoad BM BK p).SafeAt bounds (RealMasks.result s p rows cols ks) := by
  rcases p with ⟨A,B,O,M,N,K,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk
  all_goals
    simp [RealLoads.aLoad, RealMasks.aMask, Op.SafeAt.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, Op.PointerAddressesSafeOn,
      Op.MemorySafe, MaskOpt.Active, RealMasks.result, RealMasks.aResult, RealMasks.bResult,
      evalOp.eq_def, hp, bind, Option.bind, Tile.mat, Tile.remap]
    intro i k tail
    cases tail
    simpa [RealMath.active] using hb i k

theorem b_safe (bounds : RegionBounds) (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (ptrs : Tile .ptr [BK,BN]) (rows : Fin BM → Nat) (cols : Fin BN → Nat) (ks : Fin BK → Nat)
    (a : Tile .real [BM,BK]) (hp : s.regs .ptr [BK,BN] "b_ptrs" = some ptrs)
    (hb : ∀ (k : Fin BK) (j : Fin BN), RealMath.active p.divK (ks k) p.K →
      RealMath.active p.divN (cols j) p.N →
      (ptrs.data (k,j,PUnit.unit)).2 < bounds (ptrs.data (k,j,PUnit.unit)).1) :
    (RealLoads.bLoad BN BK p).SafeAt bounds
      ((RealMasks.result s p rows cols ks).setReg "a" .real [BM,BK] a) := by
  rcases p with ⟨A,B,O,M,N,K,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk
  all_goals
    simp [RealLoads.bLoad, RealMasks.bMask, Op.SafeAt.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, Op.PointerAddressesSafeOn,
      Op.MemorySafe, MaskOpt.Active, RealMasks.result, RealMasks.aResult, RealMasks.bResult,
      evalOp.eq_def, hp, bind, Option.bind, Tile.mat, Tile.remap]
    intro k j tail
    cases tail
    simpa [RealMath.active] using hb k j

theorem output_safe (bounds : RegionBounds) (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (ptrs : Tile .ptr [BM,BN]) (rows : Fin BM → Nat) (cols : Fin BN → Nat)
    (hp : s.regs .ptr [BM,BN] "o_ptrs" = some ptrs)
    (hb : ∀ (i : Fin BM) (j : Fin BN), RealMath.active p.divM (rows i) p.M →
      RealMath.active p.divN (cols j) p.N →
      (ptrs.data (i,j,PUnit.unit)).2 < bounds (ptrs.data (i,j,PUnit.unit)).1) :
    Stmt.TraceSafe bounds (.store .real [BM,BN] (.ptr (.ref .ptr [BM,BN] "o_ptrs"))
      (.ref .real [BM,BN] "o") (RealStore.mask BM BN p.divM p.divN)) (RealStore.prepared s p rows cols) := by
  rcases p with ⟨A,B,O,M,N,K,G,dm,dn,dk⟩
  cases dm <;> cases dn
  all_goals
    simp [Stmt.TraceSafe, RealStore.mask, Op.SafeAt.eq_def, MemAccess.SafeAt, MaskOpt.SafeAt,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, Op.PointerAddressesSafeOn,
      Op.MemorySafe, MaskOpt.Active, RealStore.prepared,
      evalOp.eq_def, hp, bind, Option.bind, Tile.mat, Tile.remap]
    intro i j tail
    cases tail
    simpa [RealMath.active] using hb i j

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealAccess
