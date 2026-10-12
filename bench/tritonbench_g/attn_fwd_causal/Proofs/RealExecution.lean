import bench.tritonbench_g.attn_fwd_causal.Proofs.RealEpilogue

/-! Complete real execution and output formula for both actual typed sources.
All intermediate register contracts are established by the source execution.
This region-memory theorem precedes physical address safety and flat IO. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealExecution
open VeriTile.Triton RealLoads
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 1000000

theorem executes (s : BlockState) (Q K V QS KS Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage : Nat) (hs : stage = 2 ∨ stage = 3)
    (candidate : Bool) (hBN : 0 < BN) (hu : ∀ r a, s.undef r a = 0) :
    let c := RealPrologue.configuration Q K V QS KS sqm sqk skn N HD HA
      (RealCoordinates.values s H N BM BN sqz sqh sqm)
    ∃ t, exec ((if candidate then TypedSource.reciprocal else TypedSource.original)
        Q K V QS KS Out sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        Z H N HD BM BN BD HA stage hs).eraseDType.toAlgKernel s = some t ∧
      (∀ i : TileIndex [BM,BD], RealEpilogue.active c BM BD i = Bool.true →
        (∀ a b, RealEpilogue.active c BM BD a = Bool.true → RealEpilogue.active c BM BD b = Bool.true →
          RealEpilogue.address c BM BD a = RealEpilogue.address c BM BD b → a = b) →
        t.readMem Out (RealEpilogue.address c BM BD i) =
          RealFormula.expected (inputs s c BM BN BD) c.startM N HA stage i.1 i.2.1) ∧
      (∀ r offset, (r ≠ Out ∨ ∀ i, RealEpilogue.active c BM BD i = Bool.true →
          offset ≠ RealEpilogue.address c BM BD i) → t.mem r offset = s.mem r offset) := by
  dsimp only
  let c := RealPrologue.configuration Q K V QS KS sqm sqk skn N HD HA
    (RealCoordinates.values s H N BM BN sqz sqh sqm)
  let diagonal := decide (stage=2)
  have hstage : (if diagonal then 2 else 3) = stage := by
    rcases hs with h | h <;> simp [diagonal, h]
  obtain ⟨sp, hp, entry, hout, hpf⟩ := RealPrologue.executes s Q K V QS KS Out sqz sqh sqm sqk skn H N HD BM BN BD HA hu
  have hinputs := inputs_same_memory sp s c BM BN BD hpf.mem
  obtain ⟨sm, hm, carry, hmf⟩ := RealMiddle.executes sp c BM BN BD hBN diagonal
    (by rw [hinputs]; exact entry)
  rw [hinputs] at carry
  let result := RealLoopFormula.bothResult (inputs s c BM BN BD) c diagonal
  obtain ⟨t, ht, hread, hframe⟩ := RealEpilogue.executes sm c Out BM BD candidate result.numerator result.denominator
    carry.numerator carry.denominator
    ((hmf.regs _ _ "O_block_ptr" (by decide)).trans hout)
    ((hmf.regs _ _ "offs_m" (by decide)).trans entry.rows)
  refine ⟨t, ?_, ?_, ?_⟩
  · rw [exec, RealSource.body Q K V QS KS Out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage hs candidate]
    have hprefix := (stepStmts.append_some hp).trans hm
    exact (stepStmts.append_some hprefix).trans ht
  · rintro ⟨row,lane,⟨⟩⟩ hi hinj
    rw [hread _ hi hinj]
    have h := RealLoopFormula.both_normalized (inputs s c BM BN BD) c diagonal row lane
    rw [hstage] at h
    exact h
  · intro r offset h
    rw [hframe r offset h, hmf.mem, hpf.mem]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealExecution
