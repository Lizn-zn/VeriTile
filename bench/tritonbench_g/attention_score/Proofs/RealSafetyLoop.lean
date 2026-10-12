import bench.tritonbench_g.attention_score.Proofs.RealSafetyParts

/-! Actual query-loop trace safety. The execution invariant supplies Q/M
pointers, and each visited block is bounded by the proved ceiling trip count.
No condition is imposed on numerical outputs or on unexecuted iterations. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealSafetyLoop
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccess
set_option maxHeartbeats 1000000
noncomputable section

theorem correct (bounds : RegionBounds) (c : RealLoop.Config) (initial s : BlockState)
    (stop : Nat) (hB : 0 < c.B) (hi : RealLoop.Invariant c initial 0 s)
    (hbQ : ∀ j, j < RealLoop.tripCount stop c.B →
      BlockBounds bounds (RealLoads.queryPointer c.Q c.B c.D c.N c.baseQ c.sqm c.sqk (j*c.B))
        (RealLoads.checks c.evenM))
    (hbM : ∀ row, row < RealLoop.tripCount stop c.B*c.B → c.baseM+row < bounds c.M) :
    Stmt.forRangeTraceSafe bounds "start_m" 0 stop c.B c.body s := by
  let count := RealLoop.tripCount stop c.B
  let P := fun row st => ∃ j, row = j*c.B ∧ j ≤ count ∧ RealLoop.Invariant c initial j st
  have hu : stop ≤ count*c.B := (RealLoop.tripCount_bounds stop c.B hB).1
  have visited : ∀ j, j ≤ count → j*c.B < stop → j < count := by
    intro j hj hrow
    by_contra hn
    have he : j = count := by omega
    rw [he] at hrow
    omega
  apply TraceProofs.range_safe (P := P)
  · intro row st hrow hp
    obtain ⟨j,rfl,hj,hregs⟩ := hp
    have hjlt := visited j hj hrow
    apply RealSafetyParts.body_safe bounds _ c.Q c.M c.B c.D c.N c.baseQ c.baseM c.sqm c.sqk
      (j*c.B) c.offset c.size c.window c.complement c.evenM c.evenN c.candidate
      (by simp [hregs.query]) (by simp [hregs.normalizer]) (hbQ j hjlt)
    intro i
    have hle := Nat.mul_le_mul_right c.B (Nat.succ_le_of_lt hjlt)
    have hi := i.isLt
    simp only [Nat.succ_mul] at hle
    simpa only [Nat.add_assoc] using hbM (j*c.B+i.val) (by
      change j*c.B+i.val < count*c.B
      omega)
  · intro row st t hrow hp he
    obtain ⟨j,rfl,hj,hregs⟩ := hp
    have hjlt := visited j hj hrow
    obtain ⟨u,hu,iu,_⟩ := RealLoop.iteration c initial st j hB hregs
    have hut : u = t := Option.some.inj (hu.symm.trans he)
    subst t
    exact ⟨j+1, by simp [Nat.add_mul], by omega, iu⟩
  · exact ⟨0, by simp, Nat.zero_le _, hi⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.RealSafetyLoop
