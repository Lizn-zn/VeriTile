import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Independent delta-rule formulas and the geometry needed to relate the
source's inner blocks to a BT-wide chunk. The BT-wide recurrence requires
BC to divide BT; the source itself masks only by global T. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealMath
open scoped BigOperators
noncomputable section

/-- Independent arrays; the public IO contract will pin these to input cells. -/
structure Inputs where
  key : Nat → Nat → ℝ
  delta : Nat → Nat → ℝ
  value : Nat → Nat → ℝ
  seed : Nat → Nat → ℝ

def corrected (xs : Inputs) (K : Nat) (state : Nat → Nat → ℝ)
    (time column : Nat) : ℝ :=
  xs.value time column - ∑ e : Fin K, xs.delta time e.val * state e.val column

def state (xs : Inputs) (T K BT : Nat) (useSeed : Bool) :
    Nat → Nat → Nat → ℝ
  | 0, e, p => if useSeed then xs.seed e p else 0
  | chunk + 1, e, p =>
      state xs T K BT useSeed chunk e p +
        ∑ c : Fin BT, if chunk * BT + c.val < T then
          xs.key (chunk * BT + c.val) e *
            corrected xs K (state xs T K BT useSeed chunk) (chunk * BT + c.val) p
        else 0

/-- Partition a finite row sum into inner blocks; this is real arithmetic. -/
theorem block_sum (f : Nat → ℝ) (chunks width : Nat) :
    (∑ c : Fin chunks, ∑ j : Fin width, f (c.val * width + j.val)) =
      ∑ t : Fin (chunks * width), f t.val := by
  calc
    _ = ∑ p : Fin chunks × Fin width, f (p.1.val * width + p.2.val) := by
      rw [Fintype.sum_prod_type]
    _ = _ := Fintype.sum_equiv finProdFinEquiv _ _ (by intro p; simp [finProdFinEquiv, Nat.mul_comm, Nat.add_comm])


def count (BT BC : Nat) := (BT + BC - 1) / BC

def row (BT BC outer inner lane : Nat) := outer * BT + inner * BC + lane

/-- The host's cap can give this geometry: BT=65, BC=64.
The source masks only by T, so row 65 is visited twice when T=130. -/
theorem overlapping_source_visits :
    1 < count 65 64 ∧ 1 < 64 ∧ 0 < count 65 64 ∧ 0 < 64 ∧
    row 65 64 0 1 1 = row 65 64 1 0 0 ∧ row 65 64 0 1 1 < 130 := by
  norm_num [row, count]

/-- Divisibility makes the source's rounded-up inner span exactly BT. -/
theorem complete_inner_span (BT BC : Nat) (hBC : 0 < BC) (hd : BC ∣ BT) :
    count BT BC * BC = BT := by
  obtain ⟨n, rfl⟩ := hd
  unfold count
  have he : BC * n + BC - 1 = BC * n + (BC - 1) := by omega
  rw [he, Nat.mul_add_div hBC, Nat.div_eq_of_lt (by omega : BC - 1 < BC)]
  simp [Nat.mul_comm]

theorem row_inside_chunk (BT BC outer inner lane : Nat)
    (hBC : 0 < BC) (hd : BC ∣ BT) (hc : inner < count BT BC) (hl : lane < BC) :
    outer * BT ≤ row BT BC outer inner lane ∧
    row BT BC outer inner lane < (outer + 1) * BT := by
  have hi := Nat.mul_le_mul_right BC (show inner + 1 ≤ count BT BC by omega)
  rw [complete_inner_span BT BC hBC hd, Nat.add_mul, Nat.one_mul] at hi
  simp only [row, Nat.add_mul, Nat.one_mul]
  omega

theorem distinct_chunk_rows (BT BC a b c d i j : Nat)
    (hBC : 0 < BC) (hd : BC ∣ BT)
    (hc : c < count BT BC) (hi : i < BC)
    (he : d < count BT BC) (hj : j < BC)
    (hab : a ≠ b) : row BT BC a c i ≠ row BT BC b d j := by
  obtain ⟨ha0, ha1⟩ := row_inside_chunk BT BC a c i hBC hd hc hi
  obtain ⟨hb0, hb1⟩ := row_inside_chunk BT BC b d j hBC hd he hj
  intro eq
  rcases lt_or_gt_of_ne hab with h | h
  · have hh := Nat.mul_le_mul_right BT (show a + 1 ≤ b by omega)
    omega
  · have hh := Nat.mul_le_mul_right BT (show b + 1 ≤ a by omega)
    omega


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealMath
