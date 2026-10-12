import bench.tritonbench_g.attn_fwd_causal.Proofs.RealSafetyParts

/-! Address safety of either complete helper body. Only the actual pointer
and index registers are used. The arithmetic branch preserves the value
pointer; its numerical result is not an address-safety assumption. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyBody
open VeriTile.Triton RealLoads RealMemory RealAccessSafety RealSafetyParts
open AttentionFwdTriton3.RealAccess
noncomputable section
set_option maxHeartbeats 1000000

theorem safe (bounds : RegionBounds) (s : BlockState) (c : Config) (BM BN BD start : Nat)
    (second diagonal : Bool) (hp : RealIteration.Pointers s c BN BD start second)
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hu : ∀ r a, s.undef r a = 0)
    (hbK : ∀ (j : Fin BN) (d : Fin BD), start+j.val < c.context → d.val < c.activeHead →
      c.base+d.val+j.val*c.strideKN+c.headDim*start < bounds c.K)
    (hbS : c.keyScaleBase+start/BN < bounds c.KS)
    (hbV : ∀ (j : Fin BN) (d : Fin BD), start+j.val < c.context → d.val < c.activeHead →
      c.base+j.val*c.strideM+d.val*c.strideD+c.headDim*start < bounds c.V) :
    Stmt.TraceSafeList bounds (RealLoop.sourceBody BM BN BD c.context c.activeHead c.headDim second diagonal) s := by
  obtain ⟨sl, el, _, _, hstart, fl⟩ := loads_execute s c BM BN BD start second hp.key hp.scale hn hs hu
  rw [source_split, List.append_assoc, List.append_assoc]
  apply append_of_run (loads_safe bounds s c BN BD start second hp.key hp.scale hn hs hbK hbS) el
  apply append_safe _ sl (middle_safe BM BN BD second diagonal bounds sl)
  intro sm em
  have fm := (middle_frame BM BN BD second diagonal).frame em
  have hv : sm.regs .ptr [BN,BD] (valueName second) = some (valuePointer c BN BD start) :=
    (fm.regs _ _ _ (by cases second <;> simp [middleWrites, valueName])).trans
      ((fl.regs _ _ _ (by cases second <;> simp [valueName])).trans hp.value)
  have hkeys : sm.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)) :=
    (fm.regs _ _ _ (by simp [middleWrites])).trans ((fl.regs _ _ _ (by simp)).trans hn)
  have hcur : sm.regs .nat [] "start_n" = some (Tile.scalar start) :=
    (fm.regs _ _ _ (by simp [middleWrites])).trans hstart
  apply Stmt.TraceSafeList.cons_intro (value_safe bounds sm c BN BD start (valueName second) hv hkeys hcur hbV)
  intro t _
  exact tail_safe BM BN BD c.headDim second bounds t

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyBody
