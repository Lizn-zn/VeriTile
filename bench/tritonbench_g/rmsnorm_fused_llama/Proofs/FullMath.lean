import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.Source
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.General
open VeriTile.Triton Finset
set_option maxHeartbeats 2000000

/-- Independent logical-row formula for arbitrary numbers of source iterations. -/
noncomputable def rowSpec (s : BlockState) (X W : RegionName)
    (stride N : Nat) (eps : ℝ) (k : Nat) : ℝ :=
  s.readMem X (s.pids 0 * stride + k) *
    (Real.sqrt ((∑ j : Fin N, s.readMem X (s.pids 0 * stride + j.val) ^ 2) / (N : ℝ) + eps))⁻¹ *
    s.readMem W k

/-- Padded per-lane accumulator after exactly c complete source iterations. -/
noncomputable def accumulator (s : BlockState) (X : RegionName)
    (stride N B c : Nat) (j : Fin B) : ℝ :=
  ∑ b : Fin c, if b.val * B + j.val < N then
    s.readMem X (s.pids 0 * stride + (b.val * B + j.val)) ^ 2 else 0

theorem sum_blocks_lanes (BLOCK c : Nat) (hBLOCK : 0 < BLOCK) (H : Nat → ℝ) :
    (∑ b : Fin c, ∑ j : Fin BLOCK, H (b.val * BLOCK + j.val)) = ∑ k : Fin (c * BLOCK), H k.val := by
  rw [← Fintype.sum_prod_type']
  apply Finset.sum_nbij' (i := fun p : Fin c × Fin BLOCK => (⟨p.1.val * BLOCK + p.2.val, by
        have h1 := p.1.isLt; have h2 := p.2.isLt
        calc p.1.val * BLOCK + p.2.val < p.1.val * BLOCK + BLOCK := by omega
          _ = (p.1.val + 1) * BLOCK := by ring
          _ ≤ c * BLOCK := Nat.mul_le_mul_right _ (by omega)⟩ : Fin (c*BLOCK)))
    (j := fun k : Fin (c*BLOCK) => (⟨k.val / BLOCK,
        (Nat.div_lt_iff_lt_mul hBLOCK).mpr (by have := k.isLt; omega)⟩, ⟨k.val % BLOCK, Nat.mod_lt _ hBLOCK⟩))
  · intro p _; simp
  · intro k _; simp
  · intro p _
    apply Prod.ext
    · apply Fin.ext; show (p.1.val * BLOCK + p.2.val) / BLOCK = p.1.val
      rw [Nat.mul_comm p.1.val BLOCK, Nat.mul_add_div hBLOCK, Nat.div_eq_of_lt p.2.isLt, Nat.add_zero]
    · apply Fin.ext; show (p.1.val * BLOCK + p.2.val) % BLOCK = p.2.val
      rw [Nat.mul_comm p.1.val BLOCK, Nat.mul_add_mod, Nat.mod_eq_of_lt p.2.isLt]
  · intro k _; apply Fin.ext; show (k.val / BLOCK) * BLOCK + k.val % BLOCK = k.val
    rw [Nat.mul_comm (k.val / BLOCK) BLOCK]; exact Nat.div_add_mod k.val BLOCK
  · intro p _; rfl

theorem sum_fin_extend (N M : Nat) (hNM : N ≤ M) (f : Nat → ℝ) :
    (∑ k : Fin M, (if k.val < N then f k.val else 0)) = ∑ k : Fin N, f k.val := by
  rw [Fin.sum_univ_eq_sum_range (fun k => if k < N then f k else 0) M,
      Fin.sum_univ_eq_sum_range (fun k => f k) N]
  rw [← Finset.sum_subset (s₁ := range N) (s₂ := range M)
        (fun x hx => Finset.mem_range.mpr (lt_of_lt_of_le (Finset.mem_range.mp hx) hNM))
        (by intro k hk hknotN; simp only [Finset.mem_range] at hk hknotN; simp [Nat.not_lt.mp hknotN])]
  apply Finset.sum_congr rfl
  intro k hk; simp only [Finset.mem_range] at hk; simp [hk]

theorem sum_sq_mean (BLOCK c N : Nat) (hBLOCK : 0 < BLOCK) (hge : N ≤ c * BLOCK) (f : Nat → ℝ) :
    (∑ j : Fin BLOCK, ∑ b : Fin c, (if (b.val * BLOCK + j.val) < N then f (b.val * BLOCK + j.val) else 0))
      = ∑ k : Fin N, f k.val := by
  rw [Finset.sum_comm]
  rw [sum_blocks_lanes BLOCK c hBLOCK (fun m => if m < N then f m else 0)]
  exact sum_fin_extend N (c*BLOCK) hge f


/-- At loop termination, lane-wise accumulated squares equal the full logical sum. -/
theorem accumulator_sum (s : BlockState) (X : RegionName)
    (stride N B c : Nat) (hB : 0 < B) (hN : N ≤ c * B) :
    (∑ j : Fin B, accumulator s X stride N B c j) =
      ∑ k : Fin N, s.readMem X (s.pids 0 * stride + k.val) ^ 2 :=
  sum_sq_mean B c N hB hN (fun k => s.readMem X (s.pids 0 * stride + k) ^ 2)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.General
