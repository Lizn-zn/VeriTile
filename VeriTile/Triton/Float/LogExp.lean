import VeriTile.Triton.DSL
import VeriTile.Triton.Float.LogAdmission
import VeriTile.Triton.Float.GuardedIO
import VeriTile.Meta.StatementAudit

/-!
The two kernels below read the same fp32 tile and write the same fp32 output
tile. `originalKernel` computes a piecewise log-exp; `optimizedKernel` copies
the input. The public specification `log_exp_expm1_equiv` compares these
actual kernels under the single admitted `log_exp_expm1` atom.

`expm1(x)` computes `exp(x)-1`, and `log1p(x)` computes `log(1+x)`. The near-zero
path preserves small inputs that plain `exp` can round to 1. Both unused
branch arguments are masked to zero because `tl.where` evaluates both arms.
The input condition is finite fp32 values; the threshold is not a domain filter.

PR #13 admitted this piecewise expression. Plain log-exp failed admission and
LOG-MUL remains inconclusive; this example does not supply those LSE premises.
`tl.exp` did not satisfy the gates; the source below uses `libdevice.exp`.
The symbolic block size is independent of the shape used to select the atom.
-/

noncomputable section
namespace VeriTile.Triton.FP.LogExp
open Structural Guarded
open scoped VeriTile.Spec

set_option maxHeartbeats 1600000

/-- Original PR #13 computation, applied elementwise to a symbolic tile. -/
def originalKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  a := tl.load($(xReg) + offs, dtype=tl.float32)
  near_zero := tl.abs(a) <= 0.5
  small_a := tl.where(near_zero, a, 0.0)
  other_a := tl.where(near_zero, 0.0, a)
  small := libdevice.log1p(libdevice.expm1(small_a))
  other := libdevice.log(libdevice.exp(other_a))
  out := tl.where(near_zero, small, other)
  tl.store($(yReg) + offs, (out).to(tl.float32))
}

/-- Replace the piecewise computation with the loaded input. -/
def optimizedKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  a := tl.load($(xReg) + offs, dtype=tl.float32)
  tl.store($(yReg) + offs, (a).to(tl.float32))
}

/- The scalar atom used at each lane of these kernels. -/

/-- Both fragments require the same finite input register `a`. -/
def guards : List OperandGuard := [⟨"a", .finite⟩]

/-- Same comparison as `tl.abs(a) <= 0.5`; no near-zero input restriction. -/
def nearZero (a : Op .real []) : Op .bool [] :=
  .le .real .nil
    (.where (.lt .real .nil a (.const 0)) (.sub .real .nil (.const 0) a) a)
    (.const (1 / 2))

/-- Both arms of `where` are evaluated. Mask the unused argument to zero,
as in the measured PR #13 source, before either libdevice call. -/
def expression (a : Op .real []) : Op .real [] :=
  let near := nearZero a
  let small_a := .where near a (.const 0)
  let other_a := .where near (.const 0) a
  let small := .libdeviceLog1p (.libdeviceExpm1 small_a)
  let other := .libdeviceLog (.libdeviceExp other_a)
  .where near small other

/-- Read one scalar register. `[]` is the scalar tile shape. -/
def input : Op .real [] := .ref .real [] "a"

/-- Write `out` with explicit fp32 computation. The AST's `.real` tag is the
floating carrier; `.compute (.alg .fp32 ...)` selects the numerical precision.
All arithmetic, comparisons and libdevice calls execute at that precision. -/
def assignOutput (e : Op .real []) : List ComputeStmt :=
  [.assign .real [] "out" (.compute (.alg .fp32 e))]

/-- The original scalar computation, with the input condition attached. -/
def piecewiseLogExp : GuardedFragment := ⟨guards, assignOutput (expression input)⟩

/-- The replacement `out = a`, with the same input condition and precision. -/
def identity : GuardedFragment := ⟨guards, assignOutput input⟩

/-- Bind the accepted PR #13 row to the two fragments written above. The
imported admission table supplies report data, not a hidden theorem. -/
def entry := LogAdmission.fp32_log_exp_expm1.bind piecewiseLogExp.code identity.code

/-- Check that the selected row is exactly this fp32 rule and input domain. -/
theorem report_matches :
    LogAdmission.fp32_log_exp_expm1.report.ruleID = "LOG-EXP-EXPM1" ∧
    LogAdmission.fp32_log_exp_expm1.report.input = "fp32" ∧
    LogAdmission.fp32_log_exp_expm1.report.compute = "fp32" ∧
    LogAdmission.fp32_log_exp_expm1.report.accumulator = "fp32" ∧
    LogAdmission.fp32_log_exp_expm1.report.output = "fp32" ∧
    LogAdmission.fp32_log_exp_expm1.guards = guards := by decide

/-- The one experiment-selected assumption used in this example. Its validity
is the premise supplied by the two-gates workflow, not proved by Lean here. -/
structure Rules where
  log_exp_expm1 : Spec.EvidenceValidated entry.rule entry.evidence

def Rules.assumptions (_ : Rules) : Spec.Assumptions GuardedFragment := [entry]

instance : CoeOut Rules (Spec.Assumptions GuardedFragment) := ⟨Rules.assumptions⟩

/-- One application of the admitted atom rewrites the original into `out = a`. -/
theorem admitted (R : Rules) : Spec.Derivation R.assumptions [piecewiseLogExp] [identity] :=
  .atom entry (by simp [Rules.assumptions])
    (LogAdmission.fp32_log_exp_expm1.admit _ _ R.log_exp_expm1)

/-- Scalar rewrite used by the kernel proof below. -/
theorem scalar_equiv (R : Rules) : [piecewiseLogExp] ≡[R] [identity] :=
  Spec.FloatingPoint.ofDerivation rfl trivial (admitted R)

/- Execution of the scalar atom, followed by its use in the kernels. -/

/-- Evaluate the original piecewise computation using the FP operations in M. -/
def value {α : Type} (M : Algebra α) (lt le : α → α → Bool) (a : α) : α :=
  let z := M.literal (some .fp32) .real 0
  let half := M.literal (some .fp32) .real (1 / 2)
  let absolute := if lt a z then M.binary (some .fp32) .real .sub z a else a
  let near := le absolute half
  let small := M.unary (some .fp32) .libdeviceLog1p
    (M.unary (some .fp32) .libdeviceExpm1 (if near then a else z))
  let other := M.unary (some .fp32) .libdeviceLog
    (M.unary (some .fp32) .libdeviceExp (if near then z else a))
  if near then small else other

set_option maxHeartbeats 1600000 in
/-- Instantiate the admitted scalar rule only after executing its comparisons
and both masked branches. Unsupported comparisons cannot discharge this law. -/
theorem apply_rule {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (lt le : α → α → Bool)
    (hlt : M.compareLt (some .fp32) .real = some lt)
    (hle : M.compareLe (some .fp32) .real = some le)
    (a : α) (ha : D .finite a) : value M lt le a = a := by
  let t := s.setReg "a" .real [] (fun _ => a)
  have hg : ScalarDomain D guards t := by
    intro g hg
    simp only [guards, List.mem_singleton] at hg
    subst g
    exact ⟨fun _ => a, by simp [t], ha⟩
  have h := hM piecewiseLogExp identity (admitted R) t hg hg
  simp only [piecewiseLogExp, identity, assignOutput, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, expression, nearZero, input,
    ComputeDType.eraseDType, numeric, numericLt, numericLe, hlt, hle,
    State.setReg_same, t] at h
  simp [State.setReg, bop, value] at h ⊢
  exact congrFun h PUnit.unit

/-- All untagged floating operations in this fp32 source use fp32 as well. -/
def engine {α : Type} (M : Algebra α) : Algebra α := M.withDefaultPrecision .fp32

def loaded {α : Type} (M : Algebra α) (a : α) : α := M.fp32Load a

def output {α : Type} (M : Algebra α) (a : α) : α :=
  M.cast (some .fp32) .real .real a

/-- The prefix reads `a` and evaluates the branch selector. Its successful
execution requires the interpreter to support comparisons. The only numerical
input restriction is that each loaded `a` is finite. -/
def domain (xReg : RegionName) (B : Nat) : Precondition where
  code := (originalKernel xReg "y" B).surfaceBody.take 4
  guards := [⟨"a", [B], .finite⟩]

private theorem half_eq : (0.5 : ℝ) = 1 / 2 := by norm_num
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
    simp [Precondition.Holds, domain, originalKernel, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      engine, Algebra.withDefaultPrecision, resolvePrecision, numericLt, numericLe,
      numeric, hlt] at h
  | some lt =>
    cases hle : M.compareLe (some .fp32) .real with
    | none =>
      simp [Precondition.Holds, domain, originalKernel, ComputeKernel.surfaceBody,
        run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
        engine, Algebra.withDefaultPrecision, resolvePrecision, numericLt, numericLe,
        numeric, hlt, hle] at h
    | some le =>
      refine ⟨lt, le, rfl, rfl, ?_⟩
      simp [Precondition.Holds, domain, originalKernel, ComputeKernel.surfaceBody,
        run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
        engine, Algebra.withDefaultPrecision, resolvePrecision, numericLt, numericLe,
        numeric, bop, hlt, hle, hx, TileIndex] at h
      exact h

theorem original_run {α : Type} [Inhabited α] (M : Algebra α)
    (lt le : α → α → Bool)
    (hlt : M.compareLt (some .fp32) .real = some lt)
    (hle : M.compareLe (some .fp32) .real = some le)
    (xReg yReg : RegionName) (B : Nat) (s : State α) (xs : Fin B → α)
    (hx : ∀ i : Fin B, (s.mem xReg (s.pids 0 * B + i.val)).read .real = xs i) :
    ∃ t, Structural.exec (engine M) (originalKernel xReg yReg B) s = some t ∧
      (∀ i : Fin B, t.mem yReg (s.pids 0 * B + i.val) = Cell.mk .real
        (output M (value M lt le (loaded M (xs i))))) ∧
      (∀ (r : RegionName) o, (r ≠ yReg ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [originalKernel, Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, ComputeDType.eraseDType, engine, Algebra.withDefaultPrecision,
    resolvePrecision, numeric, numericLt, numericLe, bop, hlt, hle,
    store, toFloat, ofFloat]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    simp [hx, output, loaded, value, half_eq, zero_eq]
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

theorem optimized_run {α : Type} [Inhabited α] (M : Algebra α)
    (xReg yReg : RegionName) (B : Nat) (s : State α) (xs : Fin B → α)
    (hx : ∀ i : Fin B, (s.mem xReg (s.pids 0 * B + i.val)).read .real = xs i) :
    ∃ t, Structural.exec (engine M) (optimizedKernel xReg yReg B) s = some t ∧
      (∀ i : Fin B, t.mem yReg (s.pids 0 * B + i.val) =
        Cell.mk .real (output M (loaded M (xs i)))) ∧
      (∀ (r : RegionName) o, (r ≠ yReg ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [optimizedKernel, Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, ComputeDType.eraseDType, engine, Algebra.withDefaultPrecision,
    resolvePrecision, numeric, bop, store, toFloat, ofFloat]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    simp [hx, output, loaded]
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

/-- The two actual Triton kernels succeed, write equal fp32 outputs and leave
all other memory unchanged. Only the scalar log-exp-expm1 atom is used. -/
specification log_exp_expm1_equiv (R : Rules) (xReg yReg : RegionName) (blockSize : Nat) :
    originalIO xReg yReg blockSize ≡[R] optimizedIO xReg yReg blockSize := by
  apply Spec.FloatingPoint.ofNumerical (lhs := originalIO xReg yReg blockSize)
    (rhs := optimizedIO xReg yReg blockSize) (structural := fun _ _ => False) rfl rfl
  refine ⟨?_, ?_, ?_⟩
  · simp [IO₁PrivateScratch, originalIO]
  · simp [IO₁PrivateScratch, originalIO, optimizedIO]
  · intro α _ M D hM s hs
    let xs : Fin blockSize → α := fun i => (s.mem xReg (s.pids 0 * blockSize + i.val)).read .real
    obtain ⟨lt, le, hlt, hle, hf⟩ := domain_values M D xReg blockSize s xs (fun _ => rfl) hs
    obtain ⟨a, ha, hva, hfa⟩ := original_run M lt le hlt hle xReg yReg blockSize s xs (fun _ => rfl)
    obtain ⟨b, hb, hvb, hfb⟩ := optimized_run M xReg yReg blockSize s xs (fun _ => rfl)
    refine ⟨a, b, ha, hb, ?_, ?_, ?_⟩
    · intro i
      change a.mem yReg (s.pids 0 * blockSize + i.val) = b.mem yReg (s.pids 0 * blockSize + i.val)
      rw [hva i, hvb i, apply_rule R M D hM s lt le hlt hle _ (hf i)]
    · intro r o ho _
      exact hfa r o ho
    · intro r o ho _
      exact hfb r o ho

#print_fp_assumptions log_exp_expm1_equiv

end VeriTile.Triton.FP.LogExp
