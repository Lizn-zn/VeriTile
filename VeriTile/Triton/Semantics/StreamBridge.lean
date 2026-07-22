/-
VeriTile.Triton.Semantics.StreamBridge

The reindexing vocabulary for the wave-5 **streaming** kernel genre: a kernel
that folds `T` loop steps, each step covering a block of `B` lanes.

Such a kernel's closed-form result is a sum over the flat `Nat` range
`T * B` (e.g. the `gemmSum`-style contraction of the matmul family), while
its per-step curried spec is the double sum

    ∑ t : Fin T, ∑ b : Fin B, h (t * B + b)

with the row-major block address `t * B + b`. This module owns that
interchange once (`sum_range_mul` for the `Finset.range` closed form,
`sum_univ_mul` for the `Fin (T * B)` lane-universe form), so every S1 fold
skin can move between the kernel's block-wise closed form and the spec's
per-step view without re-proving the reindexing.

The module will grow with the later wave-5 rounds (S2/S3).
-/

import Mathlib.Algebra.BigOperators.Fin

namespace VeriTile.Triton

/-! ## `StreamLane`: flat `T * B` sums ↔ per-step `B`-lane block sums -/
namespace StreamLane

variable {M : Type*} [AddCommMonoid M]

/-- **Block decomposition of a flat range sum.** A sum over the flat range
`T * B` is the double sum over `T` steps of `B`-lane blocks, the `b`-th lane
of step `t` sitting at the row-major address `t * B + b`. This is the core
interchange between a streaming kernel's closed-form accumulator and its
per-step curried spec. -/
theorem sum_range_mul (T B : Nat) (h : Nat → M) :
    ∑ k ∈ Finset.range (T * B), h k =
      ∑ t : Fin T, ∑ b : Fin B, h (t.val * B + b.val) := by
  induction T with
  | zero => simp
  | succ T ih =>
      rw [Nat.succ_mul, Finset.sum_range_add, ih, Fin.sum_univ_castSucc]
      simp [Finset.sum_range]

/-- Reverse orientation of `sum_range_mul`, for goals stated per-step first. -/
theorem sum_range_mul' (T B : Nat) (h : Nat → M) :
    ∑ t : Fin T, ∑ b : Fin B, h (t.val * B + b.val) =
      ∑ k ∈ Finset.range (T * B), h k :=
  (sum_range_mul T B h).symm

/-- `Fin`-universe version of `sum_range_mul`: a sum over the flat lane space
`Fin (T * B)` decomposes into `T` steps of `B`-lane blocks. -/
theorem sum_univ_mul (T B : Nat) (h : Nat → M) :
    ∑ l : Fin (T * B), h l.val =
      ∑ t : Fin T, ∑ b : Fin B, h (t.val * B + b.val) := by
  rw [Fin.sum_univ_eq_sum_range, sum_range_mul]

end StreamLane

end VeriTile.Triton
