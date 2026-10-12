import bench.tritonbench_g.context_attn_fwd.Proofs.Formula
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.Bounds
open VeriTile.Triton Formula
set_option maxHeartbeats 500000
set_option linter.unusedSimpArgs false

def ceilBlocks (n b : Nat) : Nat := n/b + if n%b=0 then 0 else 1

theorem lt_ceilBlocks (n b c : Nat) (hb : 0 < b) :
    c < ceilBlocks n b ↔ c*b < n := by
  have hd := Nat.div_add_mod n b
  have hr := Nat.mod_lt n hb
  by_cases hz : n%b=0
  · simp only [ceilBlocks,hz,↓reduceIte,add_zero]
    constructor
    · intro h
      have hm := Nat.mul_le_mul_right b (show c+1 ≤ n/b by omega)
      nlinarith
    · intro h
      by_contra hc
      have hm := Nat.mul_le_mul_right b (show n/b ≤ c by omega)
      nlinarith
  · have hpos : 0 < n%b := Nat.pos_of_ne_zero hz
    simp only [ceilBlocks,hz,↓reduceIte]
    constructor
    · intro h
      have hm := Nat.mul_le_mul_right b (show c ≤ n/b by omega)
      nlinarith
    · intro h
      by_contra hc
      have hm := Nat.mul_le_mul_right b (show n/b+1 ≤ c by omega)
      nlinarith

/-- The exact signed upper bound of the source range, including its empty
block selection. Negative bounds produce no iterations. -/
def stop (p : Layout) (l : Launch) (a : Inputs) : Int :=
  if (l.block*p.BM : Int) < sequence p l a then limit p l a else 0

theorem count_eq (p : Layout) (l : Launch) (a : Inputs) :
    count p l a = ceilBlocks (stop p l a).toNat p.BN := by
  unfold count stop ceilBlocks
  split_ifs <;> simp_all

/-- The mathematical block count is exactly the source's signed range test;
this statement also covers negative metadata and empty launches. -/
theorem loop_test (p : Layout) (l : Launch) (a : Inputs) (c : Nat) (hBN : 0 < p.BN) :
    c < count p l a ↔ ((c*p.BN : Nat) : Int) < stop p l a := by
  rw [count_eq,lt_ceilBlocks _ _ _ hBN]
  omega

theorem loop_finished (p : Layout) (l : Launch) (a : Inputs) (hBN : 0 < p.BN) :
    stop p l a ≤ ((count p l a*p.BN : Nat) : Int) := by
  have h := loop_test p l a (count p l a) hBN
  omega

theorem reached_window (p : Layout) (l : Launch) (a : Inputs) (c : Nat)
    (hc : c < count p l a) : (c+1)*p.BN ≤ count p l a*p.BN :=
  Nat.mul_le_mul_right p.BN (by omega)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.Bounds
