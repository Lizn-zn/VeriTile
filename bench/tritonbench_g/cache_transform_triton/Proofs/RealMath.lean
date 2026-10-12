import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit
/-! Independent prefill row-selection formula on signed cumulative lengths.
This is arithmetic preparation for the complete-source migration, not a kernel
execution or IO proof. Each selected entry is the original cumulative length
when it is at most the token index, and zero otherwise; a nonempty maximum is
required. Negative metadata remains signed rather than being clipped to zero. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.RealMath
noncomputable section

def selectedEnd {n : Nat} (hn : 0 < n) (token : Nat) (ends : Fin n → Int) : Int :=
  (Finset.univ : Finset (Fin n)).sup' ⟨⟨0,hn⟩, Finset.mem_univ _⟩
    (fun i => if ends i ≤ (token : Int) then ends i else 0)

def relativeRow {n : Nat} (hn : 0 < n) (token : Nat) (ends : Fin n → Int) : Int :=
  (token : Int) - selectedEnd hn token ends

theorem selectedEnd_le {n : Nat} (hn : 0 < n) (token : Nat) (ends : Fin n → Int) :
    selectedEnd hn token ends ≤ (token : Int) := by
  apply Finset.sup'_le
  intro i _
  split_ifs with h
  · exact h
  · exact Int.natCast_nonneg token

theorem relativeRow_nonnegative {n : Nat} (hn : 0 < n) (token : Nat) (ends : Fin n → Int) :
    0 ≤ relativeRow hn token ends := sub_nonneg.mpr (selectedEnd_le hn token ends)

theorem natural_row_exact {n : Nat} (hn : 0 < n) (token : Nat) (ends : Fin n → Int) :
    ((relativeRow hn token ends).toNat : Int) = relativeRow hn token ends :=
  Int.toNat_of_nonneg (relativeRow_nonnegative hn token ends)

-- Signed data is retained. A zero initializer for max would change this case.
theorem negative_metadata : relativeRow (by decide : 0 < 2) 3 ![-7,-2] = 5 := by
  norm_num [relativeRow, selectedEnd, Finset.univ_fin2]

theorem first_sequence : relativeRow (by decide : 0 < 2) 1 ![3,7] = 1 := by
  norm_num [relativeRow, selectedEnd, Finset.univ_fin2]

theorem second_sequence : relativeRow (by decide : 0 < 2) 5 ![3,7] = 2 := by
  norm_num [relativeRow, selectedEnd, Finset.univ_fin2]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.RealMath
