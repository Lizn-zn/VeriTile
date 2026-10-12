import bench.tritonbench_g.attn_fwd_causal.Proofs.RealSafetyMiddle

/-! Complete source trace safety from finite active input/output windows.
Program IDs determine the addresses; both legal stages, all source strides
and both normalization variants are preserved. The zero-undef convention is
the same initial-state convention used by the real IO execution contract. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafety
open VeriTile.Triton RealLoads
open AttentionFwdTriton3.RealAccess
noncomputable section
set_option maxHeartbeats 1000000

theorem safe (bounds : RegionBounds) (s : BlockState) (Q K V QS KS Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage : Nat) (hs : stage = 2 ∨ stage = 3)
    (candidate : Bool) (hBN : 0 < BN) (hu : ∀ r a, s.undef r a = 0)
    (hb : RealAccessSafety.Windows
      (RealPrologue.configuration Q K V QS KS sqm sqk skn N HD HA
        (RealCoordinates.values s H N BM BN sqz sqh sqm)) Out BM BN BD stage bounds) :
    Kernel.TraceSafe bounds ((if candidate then TypedSource.reciprocal else TypedSource.original)
      Q K V QS KS Out sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage hs).eraseDType.toAlgKernel s := by
  let c := RealPrologue.configuration Q K V QS KS sqm sqk skn N HD HA
    (RealCoordinates.values s H N BM BN sqz sqh sqm)
  let diagonal := decide (stage=2)
  have hstage : (if diagonal then 2 else 3) = stage := by
    rcases hs with h | h <;> simp [diagonal, h]
  obtain ⟨sp, hp, entry, hout, hpf⟩ := RealPrologue.executes s Q K V QS KS Out sqz sqh sqm sqk skn H N HD BM BN BD HA hu
  have hinputs := inputs_same_memory sp s c BM BN BD hpf.mem
  have entry' : RealMiddle.Entry (inputs sp c BM BN BD) c sp := by rw [hinputs]; exact entry
  obtain ⟨sm, hm, _, hmf⟩ := RealMiddle.executes sp c BM BN BD hBN diagonal entry'
  have hmid := RealSafetyMiddle.safe bounds sp c Out BM BN BD hBN diagonal
    (by rw [hstage]; exact hb) entry'
  have hlast := RealSafetyPrologue.epilogue_safe bounds sm c Out BM BD candidate
    ((hmf.regs _ _ "O_block_ptr" (by decide)).trans hout)
    ((hmf.regs _ _ "offs_m" (by decide)).trans entry.rows) hb.output
  have hall := append_of_run
    (RealSafetyPrologue.safe bounds s Q K V QS KS Out sqz sqh sqm sqk skn H N HD BM BN BD HA hb.query hb.queryScale) hp
    (append_of_run hmid hm hlast)
  rw [Kernel.TraceSafe, RealSource.body Q K V QS KS Out sqz sqh sqm sqk skz skh skn skk
    svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage hs candidate]
  simpa only [List.append_assoc] using hall

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafety
