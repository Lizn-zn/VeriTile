import bench.tritonbench_g.softmax_flaggems.Proofs.RealLoop
import VeriTile.Triton.Memory.Flatten

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealLoop
open VeriTile Triton

/-- Access safety follows the same successful transitions as the numerical loop
invariant, including the final partial tile. -/
theorem range_tiles_safe
    {bounds : RegionBounds} {idx : RegName} {N B : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} (hN : 0 < N) (hB : 0 < B)
    (hsafe : ∀ c s, c < RealAddress.rounds N B → P c s →
      Stmt.TraceSafeList bounds body (s.setReg idx .nat [] (Tile.scalar (c * B))))
    (hstep : ∀ c s, c < RealAddress.rounds N B → P c s →
      ∃ s', stepStmts body (s.setReg idx .nat [] (Tile.scalar (c * B))) = some s' ∧
        P (c + 1) s') :
    ∀ c s, c ≤ RealAddress.rounds N B → P c s →
      Stmt.forRangeTraceSafe bounds idx (c * B) N B body s := by
  intro c s
  induction h : (RealAddress.rounds N B - c) using Nat.strong_induction_on
      generalizing c s with
  | _ remaining ih =>
    intro hc hp
    by_cases hdone : c = RealAddress.rounds N B
    · subst c
      rw [Stmt.forRangeTraceSafe, if_neg (Nat.ne_of_gt hB),
        if_neg (Nat.not_lt_of_ge (RealAddress.coverage hB))]
      exact True.intro
    · have hlt : c < RealAddress.rounds N B := by omega
      obtain ⟨next, hbody, hnext⟩ := hstep c s hlt hp
      rw [Stmt.forRangeTraceSafe, if_neg (Nat.ne_of_gt hB),
        if_pos ((RealAddress.tile_start_lt_iff hN hB).mpr hlt)]
      refine ⟨hsafe c s hlt hp, ?_⟩
      rw [hbody]
      have ht := ih (RealAddress.rounds N B - (c + 1)) (by omega)
        (c + 1) next rfl (by omega) hnext
      simpa only [Nat.add_mul, Nat.one_mul] using ht

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealLoop
