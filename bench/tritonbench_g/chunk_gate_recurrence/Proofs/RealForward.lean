import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForwardInit

/-! Complete forward region-memory correctness. The first store and all N-1
updates implement the independent formula; every other full tagged cell is
unchanged. Bounds and flattening are separate obligations, not implied here. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForward
open VeriTile.Triton RealProgram RealForwardProgram RealForwardLoop RealForwardInit
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem invariant (p : Layout) (l : Launch) (S D O seed : RegionName) (a : Inputs p)
    (hasSeed : Bool) (u : BlockState) (hN : 0 < p.N) (hSO : S ≠ O) (hDO : D ≠ O)
    (hi : NoCollision p l)
    (hS : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
      u.readMem S (address p l t e.val j.val) = a.source t e j)
    (hD : ∀ t, t < p.N → u.readMem D (gateAddress p l t) = a.gate t) :
    Holds u p l S D O a hasSeed 0 (initial p l S D O seed a hasSeed u) := by
  let pre := beforeStore p l S O seed a hasSeed u
  have hzero : RealMath.forward a hasSeed 0 = RealMath.initial a hasSeed := by
    funext e j
    exact RealMath.forward_zero a hasSeed e j
  refine ⟨Nat.zero_le _,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · cases hasSeed <;> simp [initial,scatter,beforeStore,seeded,registers]
  · simp [initial]
  · simp [initial]
  · rw [hzero]
    cases hasSeed <;> simp [initial,scatter,beforeStore,seeded,registers]
  · intro t e j ht
    change (scatter p l O 0 (RealMath.initial a hasSeed) pre).readMem S _ = _
    rw [scatter, BlockState.scatter_preserves_other_region _ _ _ _ hSO]
    cases hasSeed <;> simpa [pre,beforeStore,seeded,registers] using hS t e j ht
  · intro t ht
    change (scatter p l O 0 (RealMath.initial a hasSeed) pre).readMem D _ = _
    rw [scatter, BlockState.scatter_preserves_other_region _ _ _ _ hDO]
    cases hasSeed <;> simpa [pre,beforeStore,seeded,registers] using hD t ht
  · intro t e j ht
    have ht0 : t = 0 := by omega
    subst t
    change (scatter p l O 0 (RealMath.initial a hasSeed) pre).readMem O _ = _
    rw [scatter_read p l O 0 _ pre hi e j, RealMath.forward_zero]
  · have hs := scatter_frame p l O 0 hN (RealMath.initial a hasSeed) pre
    intro r off hn
    cases hasSeed <;> exact hs r off hn

theorem original_executes (p : Layout) (S D O seed : RegionName) (a : Inputs p)
    (hasSeed : Bool) (u : BlockState) (storage gate seedStorage : TypedSource.Storage) (H : Nat)
    (hN : 0 < p.N) (hBK : p.BK ≤ p.K) (hBV : p.BV ≤ p.V)
    (hSO : S ≠ O) (hDO : D ≠ O)
    (hS : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
      u.readMem S (address p (launch u) t e.val j.val) = a.source t e j)
    (hD : ∀ t, t < p.N → u.readMem D (gateAddress p (launch u) t) = a.gate t)
    (hSeed : hasSeed = Bool.true → ∀ e j,
      u.readMem seed (seedAddress p (launch u) e.val j.val) = a.seed e j) :
    ∃ v, exec (TypedSource.forward storage gate seedStorage S D O seed H
      p.N p.K p.V p.BK p.BV hasSeed).eraseDType.toAlgKernel u = some v ∧
      (∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
        v.readMem O (address p (launch u) t e.val j.val) = RealMath.forward a hasSeed t e j) ∧
      BlockState.WriteWithin (footprint p (launch u) O) u v := by
  have hi := noCollision p (launch u) hBK hBV
  obtain ⟨v,he,h⟩ := RealForwardLoop.executes u p (launch u) S D O a hasSeed hSO hDO hi
    (initial p (launch u) S D O seed a hasSeed u)
    (invariant p (launch u) S D O seed a hasSeed u hN hSO hDO hi hS hD)
  refine ⟨v,?_,?_,h.frame⟩
  · rw [exec,source_body,ProgramRewrite.append,RealForwardInit.executes p S D O seed a hasSeed u hSeed]
    change stepStmts [statement p] _ = some v
    rw [stepStmts.cons_some he,stepStmts.nil]
  · intro t e j ht
    exact h.output t e j (by omega)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForward
