import FifthSphDraft.BackwardBase
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardReadback
open VeriTile.Triton BackwardBase
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem coord_injective (pid B c : Nat) :
    Function.Injective (fun j : TileIndex [B] => j.1.val*3+B*3*pid+c) := by
  intro a b h
  change a.1.val*3+B*3*pid+c=b.1.val*3+B*3*pid+c at h
  have hv : a.1.val=b.1.val := by omega
  obtain ⟨a,⟨⟩⟩ := a
  obtain ⟨b,⟨⟩⟩ := b
  have hab : a=b := Fin.ext hv
  subst b
  rfl

theorem coord_disjoint (pid B : Nat) (j : Fin B) (k : TileIndex [B]) (c d : Fin 3)
    (hcd : c≠d) : coord pid B j.val c.val≠coord pid B k.1.val d.val := by
  unfold coord
  have hc := c.isLt
  have hd := d.isLt
  intro h
  have he : c.val=d.val := by omega
  exact hcd (Fin.ext he)


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardReadback
