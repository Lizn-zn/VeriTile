import bench.tritonbench_g.flash_attn.Proofs.Formula

/-! Prefix/block algebra for the complete source's rounded-up key traversal.
The lists below contain only initial-memory score/value pairs. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealMath
open VeriTile.Triton
noncomputable section

/-- The online state for a nonempty list always has a finite maximum. -/
theorem maximum_eq_bot_imp_nil (xs : List (ℝ × ℝ)) (h : maximum xs = ⊥) : xs = [] := by
  by_contra hn
  obtain ⟨r, hr⟩ := maximum_finite xs hn
  rw [h] at hr
  exact WithBot.bot_ne_coe hr

/-- One block update of the actual arithmetic recurrence extends the prefix. -/
theorem running_append_block (xs block : List (ℝ × ℝ)) :
    let st := running xs
    let next := maximum xs ⊔ maximum block
    (next,
      st.2.1 * (WithBot.realExp2 (WithBot.realSub (maximum xs) next)).unbotD 0 +
        (block.map (fun v => pow2 (v.1 - next.unbotD 0))).sum,
      st.2.2 * (WithBot.realExp2 (WithBot.realSub (maximum xs) next)).unbotD 0 +
        (block.map (fun v => pow2 (v.1 - next.unbotD 0) * v.2)).sum) =
      running (xs ++ block) := by
  obtain ⟨hd, ha⟩ := anchored xs
  have hz : maximum xs = ⊥ → xs = [] := maximum_eq_bot_imp_nil xs
  have h := osStepBot_block_eq (maximum xs) (running xs).2.1 (running xs).2.2
    (numerator xs) (denominator xs) block hd ha
    (fun h => by simp [hz h, denominator]) (fun h => by simp [hz h, numerator])
  simpa only [running, List.foldl_append, ← running_max xs] using h

end
end VeriTile.Bench.TritonBenchG.FlashAttn.RealMath

namespace VeriTile.Bench.TritonBenchG.FlashAttn.Formula
open VeriTile.Triton
noncomputable section

def visible (p : Layout) (pid₀ i k : Nat) : Bool :=
  !p.causal || decide (k ≤ row p pid₀ i)

def prefixKeys (p : Layout) (pid₀ i c : Nat) : List Nat :=
  (List.range (c * p.BLOCK_N)).filter (visible p pid₀ i)

def blockKeys (p : Layout) (pid₀ i c : Nat) : List Nat :=
  ((List.range p.BLOCK_N).map (fun j => c * p.BLOCK_N + j)).filter (visible p pid₀ i)

def prefixPairs (p : Layout) (read : RegionName → Nat → ℝ) (Q K V : RegionName)
    (pid₀ pid₁ i d c : Nat) : List (ℝ × ℝ) :=
  (prefixKeys p pid₀ i c).map
    (fun k => (score p read Q K pid₀ pid₁ i k, read V (vOffset p pid₁ k d)))

def blockPairs (p : Layout) (read : RegionName → Nat → ℝ) (Q K V : RegionName)
    (pid₀ pid₁ i d c : Nat) : List (ℝ × ℝ) :=
  (blockKeys p pid₀ i c).map
    (fun k => (score p read Q K pid₀ pid₁ i k, read V (vOffset p pid₁ k d)))

theorem prefixKeys_succ (p : Layout) (pid₀ i c : Nat) :
    prefixKeys p pid₀ i (c + 1) = prefixKeys p pid₀ i c ++ blockKeys p pid₀ i c := by
  simp only [prefixKeys, blockKeys, Nat.add_mul, one_mul, List.range_add, List.filter_append]

theorem prefixPairs_succ (p : Layout) (read : RegionName → Nat → ℝ) (Q K V : RegionName)
    (pid₀ pid₁ i d c : Nat) :
    prefixPairs p read Q K V pid₀ pid₁ i d (c + 1) =
      prefixPairs p read Q K V pid₀ pid₁ i d c ++ blockPairs p read Q K V pid₀ pid₁ i d c := by
  simp only [prefixPairs, blockPairs, prefixKeys_succ, List.map_append]

@[simp] theorem prefixPairs_zero (p : Layout) (read : RegionName → Nat → ℝ)
    (Q K V : RegionName) (pid₀ pid₁ i d : Nat) :
    prefixPairs p read Q K V pid₀ pid₁ i d 0 = [] := by
  simp [prefixPairs, prefixKeys]

theorem prefixPairs_full (p : Layout) (read : RegionName → Nat → ℝ)
    (Q K V : RegionName) (pid₀ pid₁ i d : Nat) :
    prefixPairs p read Q K V pid₀ pid₁ i d (Geometry.tiles (hi p pid₀) p.BLOCK_N) =
      pairs p read Q K V pid₀ pid₁ i d := rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.Formula
