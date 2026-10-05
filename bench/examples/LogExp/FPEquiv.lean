import bench.examples.LogExp.Kernels
import VeriTile.Triton.Float.LogExp
import VeriTile.Meta.StatementAudit

/-!
FP equivalence of originalKernel and optimizedKernel from Kernels.lean.
Each loaded operand must be finite. The only numerical assumption is the
admitted log_exp_elim scalar relation. Comparisons and casts retain fp32
precision. The symbolic tile size is independent of the experimental shape.
The proof lifts the scalar relation to equal writes and a memory frame.
-/

noncomputable section
namespace VeriTile.Bench.Examples.LogExp.FPEquiv
open Triton FP FP.Structural FP.Guarded FP.LogExp
open scoped VeriTile.Spec

set_option maxHeartbeats 1600000

/-- All untagged floating operations in this fp32 source use fp32 as well. -/
def engine {α : Type} (M : Algebra α) : Algebra α := M.withDefaultPrecision .fp32

def loaded {α : Type} (M : Algebra α) (a : α) : α := M.fp32Load a

def output {α : Type} (M : Algebra α) (a : α) : α :=
  M.cast (some .fp32) .real .real a

/-- The candidate prefix reads `a` and evaluates the branch selector. Successful
execution requires comparison support, for either branch outcome. The only
numerical input restriction is that each loaded `a` is finite. -/
def domain (xReg : RegionName) (B : Nat) : Precondition where
  code := (optimizedKernel xReg "y" B).surfaceBody.take 4
  guards := [⟨"a", [B], .finite⟩]

private theorem half_eq : (0.5 : ℝ) = 1 / 2 := by norm_num
private theorem upper_eq : (80.0 : ℝ) = 80 := by norm_num
private theorem zero_eq : (0.0 : ℝ) = 0 := by norm_num

theorem domain_values {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (xReg : RegionName) (B : Nat) (s : State α) (xs : Fin B → α)
    (hx : ∀ i : Fin B, (s.mem xReg (s.pids 0 * B + i.val)).read .real = xs i)
    (h : (domain xReg B).Holds (engine M) D s) :
    ∃ lt le, M.compareLt (some .fp32) .real = some lt ∧
      M.compareLe (some .fp32) .real = some le ∧
      ∀ i, D .finite (loaded M (xs i)) := by
  cases hlt : M.compareLt (some .fp32) .real with
  | none =>
    simp [Precondition.Holds, domain, optimizedKernel, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      engine, Algebra.withDefaultPrecision, resolvePrecision, numericLt, numericLe,
      numeric, hlt] at h
  | some lt =>
    cases hle : M.compareLe (some .fp32) .real with
    | none =>
      simp [Precondition.Holds, domain, optimizedKernel, ComputeKernel.surfaceBody,
        run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
        engine, Algebra.withDefaultPrecision, resolvePrecision, numericLt, numericLe,
        numeric, hlt, hle] at h
    | some le =>
      refine ⟨lt, le, rfl, rfl, ?_⟩
      simp [Precondition.Holds, domain, optimizedKernel, ComputeKernel.surfaceBody,
        run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
        engine, Algebra.withDefaultPrecision, resolvePrecision, numericLt, numericLe,
        numeric, bop, hlt, hle, hx, TileIndex] at h
      exact h

theorem original_run {α : Type} [Inhabited α] (M : Algebra α)
    (xReg yReg : RegionName) (B : Nat) (s : State α) (xs : Fin B → α)
    (hx : ∀ i : Fin B, (s.mem xReg (s.pids 0 * B + i.val)).read .real = xs i) :
    ∃ t, Structural.exec (engine M) (originalKernel xReg yReg B) s = some t ∧
      (∀ i : Fin B, t.mem yReg (s.pids 0 * B + i.val) =
        Cell.mk .real (output M (referenceValue M (loaded M (xs i))))) ∧
      (∀ (r : RegionName) o, (r ≠ yReg ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [originalKernel, Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, ComputeDType.eraseDType, engine, Algebra.withDefaultPrecision,
    resolvePrecision, numeric, bop, store, toFloat, ofFloat]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    simp [hx, output, loaded, referenceValue]
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

theorem optimized_run {α : Type} [Inhabited α] (M : Algebra α)
    (lt le : α → α → Bool)
    (hlt : M.compareLt (some .fp32) .real = some lt)
    (hle : M.compareLe (some .fp32) .real = some le)
    (xReg yReg : RegionName) (B : Nat) (s : State α) (xs : Fin B → α)
    (hx : ∀ i : Fin B, (s.mem xReg (s.pids 0 * B + i.val)).read .real = xs i) :
    ∃ t, Structural.exec (engine M) (optimizedKernel xReg yReg B) s = some t ∧
      (∀ i : Fin B, t.mem yReg (s.pids 0 * B + i.val) = Cell.mk .real
        (output M (value M lt le (loaded M (xs i))))) ∧
      (∀ (r : RegionName) o, (r ≠ yReg ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [optimizedKernel, Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, ComputeDType.eraseDType, engine, Algebra.withDefaultPrecision,
    resolvePrecision, numeric, numericLt, numericLe, bop, hlt, hle,
    store, toFloat, ofFloat]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    simp [hx, output, loaded, value, half_eq, upper_eq, zero_eq]
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

/-- The original kernel's input/output windows, domain and fp32 profile. -/
def originalIO (xReg yReg : RegionName) (B : Nat) : Guarded.IO where
  io := {
    kernel := originalKernel xReg yReg B
    inp := xReg, out := yReg, Bin := B, Bout := B
    read := fun pid => pid * B
    write := fun pid => pid * B }
  domain := domain xReg B
  defaultPrecision := some .fp32

/-- Same interface and domain; the implementation is `optimizedKernel`. -/
def optimizedIO (xReg yReg : RegionName) (B : Nat) : Guarded.IO :=
  { originalIO xReg yReg B with io := { (originalIO xReg yReg B).io with
      kernel := optimizedKernel xReg yReg B, projection := by rfl } }

/-- Under the admitted guarded relation, both kernels succeed, write equivalent
fp32 outputs and preserve all other memory. Dimensions remain symbolic. -/
specification log_exp_equiv (R : Rules)
    (xReg yReg : RegionName) (blockSize : Nat) :
    originalIO xReg yReg blockSize ≡[R] optimizedIO xReg yReg blockSize := by
  apply Spec.FloatingPoint.ofNumerical (lhs := originalIO xReg yReg blockSize)
    (rhs := optimizedIO xReg yReg blockSize) (structural := fun _ _ => False) rfl rfl
  refine ⟨?_, ?_, ?_⟩
  · simp [IO₁PrivateScratch, originalIO]
  · simp [IO₁PrivateScratch, originalIO, optimizedIO]
  · intro α _ M D hM s hs
    let xs : Fin blockSize → α := fun i => (s.mem xReg (s.pids 0 * blockSize + i.val)).read .real
    obtain ⟨lt, le, hlt, hle, hf⟩ := domain_values M D xReg blockSize s xs (fun _ => rfl) hs
    obtain ⟨a, ha, hva, hfa⟩ := original_run M xReg yReg blockSize s xs (fun _ => rfl)
    obtain ⟨b, hb, hvb, hfb⟩ := optimized_run M lt le hlt hle xReg yReg blockSize s xs (fun _ => rfl)
    refine ⟨a, b, ha, hb, ?_, ?_, ?_⟩
    · intro i
      change a.mem yReg (s.pids 0 * blockSize + i.val) = b.mem yReg (s.pids 0 * blockSize + i.val)
      rw [hva i, hvb i, apply_rule R (by decide) M D hM s lt le hlt hle _ (hf i)]
    · intro r o ho _
      exact hfa r o ho
    · intro r o ho _
      exact hfb r o ho

#print_fp_assumptions log_exp_equiv

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.LogExp.FPEquiv
