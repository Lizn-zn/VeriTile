import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealMath

/-! Independent initial arrays for one forward chunk tile. Metadata arrays
are signed, and the mathematical target contains no interpreter state. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealInputs
open scoped BigOperators
noncomputable section
structure Inputs (BM BN K : Nat) where
  a : Fin BM → Fin K → ℝ
  b : Fin K → Fin BN → ℝ
  rowIDs : Fin BM → Int
  columnIDs : Fin BN → Int

def formula {BM BN K : Nat} (length : Nat) (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (hasSeq : Bool) (xs : Inputs BM BN K) (i : Fin BM) (j : Fin BN) : ℝ :=
  if rows i < length ∧ columns j < length ∧ (hasSeq = Bool.false ∨ xs.rowIDs i = xs.columnIDs j)
  then ∑ k : Fin K, xs.a i k * xs.b k j else 0

theorem sum_of_pins (K : Nat) (a b : Fin K → ℝ) (left right : Nat → ℝ)
    (ha : ∀ k : Fin K, left k.val = a k) (hb : ∀ k : Fin K, right k.val = b k) :
    (∑ k ∈ Finset.range K, left k * right k) = ∑ k : Fin K, a k * b k := by
  rw [Finset.sum_range]
  apply Finset.sum_congr rfl
  intro k _
  rw [ha k, hb k]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealInputs
