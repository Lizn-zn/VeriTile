import bench.examples.StableLogSumExp.Contract
import VeriTile.Meta.StatementAudit

/-!
FP equivalence of optimizedLSEKernel and the unchanged directLSEKernel in
Kernels.lean. Both use tl.log and libdevice.exp and store a scalar bf16 result.
The conditional optimized preserves the product-log fallback near one and
the log-exp fallback near zero and outside the cancellation interval.

The contract records the finite/positive/nonzero intermediate operands used
by the scalar derivation, an arbitrary shared reduction schedule, and support
for fp32 comparisons. Branch outcomes are unrestricted. The row length is
symbolic. The specification includes successful execution and a memory frame.

The older stableLSEKernel omits both fallbacks. Its unconditional log_mul and
log_exp candidates are not admitted; Unconditional.lean records that separate goal.
-/

noncomputable section
namespace VeriTile.Bench.Examples.StableLogSumExpFPEquiv
open VeriTile Triton FP.Structural FP.Guarded
open StableLogSumExpFPContract StableLogSumExpFPExecution
open scoped VeriTile.Spec

/-- Actual source-to-source equivalence from admitted scalar atoms. -/
specification logsumexp_equiv (B : Nat) (hB : 0 < B) (R : FP.LogSumExp.Rules) :
    optimized "x" "y" B ≡[R] original "x" "y" B := by
  apply Spec.FloatingPoint.ofNumerical (lhs := optimized "x" "y" B)
    (rhs := original "x" "y" B) (structural := fun _ _ => False) rfl rfl
  refine ⟨by simp [optimized, IO₁PrivateScratch, optimizedIO, directIO],
    by simp [original, optimized, IO₁PrivateScratch, directIO], ?_⟩
  intro α _ M D hM comparisons plans s hd
  obtain ⟨lt, hlt⟩ : ∃ lt, M.compareLt (some .fp32) .real = some lt := by
    have h := comparisons (.lt .fp32 .real) (by simp [optimized])
    cases he : M.compareLt (some .fp32) .real with
    | none => simp [FP.Scheduled.Comparison.Supported, he] at h
    | some lt => exact ⟨lt, rfl⟩
  obtain ⟨le, hle⟩ : ∃ le, M.compareLe (some .fp32) .real = some le := by
    have h := comparisons (.le .fp32 .real) (by simp [optimized])
    cases he : M.compareLe (some .fp32) .real with
    | none => simp [FP.Scheduled.Comparison.Supported, he] at h
    | some le => exact ⟨le, rfl⟩
  obtain ⟨a, b, ha, hb, hout, hfa, hfb⟩ :=
    candidate_runs R M D hM s plans lt le hlt hle "x" "y" B hB hd
  refine ⟨a, b, ha, hb, ?_, hfa, hfb⟩
  intro i
  simpa only [optimized, original, optimizedIO, directIO, Fin.val_eq_zero, Nat.add_zero] using hout

#print_fp_assumptions logsumexp_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.StableLogSumExpFPEquiv
