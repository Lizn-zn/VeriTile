import bench.tritonbench_g.rotary_emb.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite FullSource FPProof
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (q k cos sin : Storage) (v : Bool) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) : Program :=
  ⟨kernel q k cos sin v Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd
    L HQ HK BH BS D, []⟩

/-- All four stores, every register, arbitrary aliases and undefined values, and
execution failure are preserved. The only change is exact natural multiplication;
no numerical admission or finite-input premise is required. -/
specification source_equivalence (q k cos sin : Storage) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    program q k cos sin Bool.false Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd
      L HQ HK BH BS D ≡[assumptions]
    program q k cos sin Bool.true Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd
      L HQ HK BH BS D := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl rfl
  intro α _ M _ _ s _
  change run (engine M)
      (kernel q k cos sin Bool.false Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd
        L HQ HK BH BS D).surfaceBody s =
    run (engine M)
      (kernel q k cos sin Bool.true Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd
        L HQ HK BH BS D).surfaceBody s
  rw [decomposition q k cos sin Bool.false, decomposition q k cos sin Bool.true]
  exact prefix_run M s BH _

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FPEquiv
