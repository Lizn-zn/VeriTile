import bench.examples.SoftmaxStable.Kernels
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Triton.Float.SoftmaxShift
import VeriTile.Triton.Float.ScheduledIO

/-! FP proof support for SoftmaxStable. The public specification and atomic
assumption report are in ../FPEquiv.lean; the shared sources are in ../Kernels.lean.
Execution, intermediate-value domains and their composition are kept together. -/

/- Use libdevice.exp for exp-sub rewrites: the measured fp32 tl.exp relation
has B = 0.1608954387 ULP > 0.05 under the configured Normal(1,1) probe.
That intrinsic relation failed admission; the libdevice EXP-SUB instance passed. -/
/- Opaque execution of the naive and stable libdevice.exp softmax kernels.
The outputs retain their bf16 casts; max, exp and sum have no implicit laws. -/

namespace VeriTile.Bench.Examples.SoftmaxStableFPExecution
open VeriTile.Bench.Examples.SoftmaxStable.Kernels
open VeriTile Triton FP.Structural

set_option maxHeartbeats 1600000


def maximum {α : Type} (M : Algebra α) (xs : Fin B → α) : α :=
  M.reduceMax none (shape := [B]) ⟨0, by simp⟩ Bool.false (fun i => xs i.1) PUnit.unit

def exponentials {α : Type} (M : Algebra α) (xs : Fin B → α) : Fin B → α :=
  fun i => M.unary none .libdeviceExp (xs i)

def shifted {α : Type} (M : Algebra α) (xs : Fin B → α) : Fin B → α :=
  fun i => M.unary none .libdeviceExp (M.binary none .real .sub (xs i) (maximum M xs))

def rowSum {α : Type} (M : Algebra α) (xs : Fin B → α) : α :=
  M.reduceSum none (shape := [B]) ⟨0, by simp⟩ Bool.false (fun i => xs i.1) PUnit.unit

def normalizedValue {α : Type} (M : Algebra α) (xs : Fin B → α) (i : Fin B) : α :=
  M.cast none .real .bf16 (M.binary none .real .div (xs i) (rowSum M xs))

theorem naive_run {α : Type} [Inhabited α] (M : Algebra α)
    (x y : RegionName) (B : Nat) (xs : Fin B → α) (s : State α)
    (hx : ∀ i : Fin B, (s.mem x (s.pids 0 * B + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (naiveSoftmaxKernel x y B) s = some t ∧
      (∀ i : Fin B, t.mem y (s.pids 0 * B + i.val) = .mk .bf16
        (normalizedValue M (exponentials M xs) i)) ∧
      (∀ r o, (r ≠ y ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) → t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [naiveSoftmaxKernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, TileShape.eraseAxis, Region.cast, ofFloat, toFloat,
    hx, normalizedValue, exponentials, rowSum]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    rfl
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

/-- Max reduction requires a nonempty source row. No max property is needed
for this execution theorem, and input/output aliasing is allowed. -/
theorem stable_run {α : Type} [Inhabited α] (M : Algebra α)
    (x y : RegionName) (B : Nat) (hB : 0 < B) (xs : Fin B → α) (s : State α)
    (hx : ∀ i : Fin B, (s.mem x (s.pids 0 * B + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (stableSoftmaxKernel x y B) s = some t ∧
      (∀ i : Fin B, t.mem y (s.pids 0 * B + i.val) = .mk .bf16
        (normalizedValue M (shifted M xs) i)) ∧
      (∀ r o, (r ≠ y ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) → t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [stableSoftmaxKernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, TileShape.axisDim, TileShape.eraseAxis, hB,
    Region.cast, ofFloat, toFloat, hx, normalizedValue, shifted, maximum, rowSum]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    rfl
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

def naiveIO (x y : RegionName) (B : Nat) : KernelIO₁ where
  kernel := naiveSoftmaxKernel x y B
  projection := by rfl
  inp := x
  out := y
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

def stableIO (x y : RegionName) (B : Nat) : KernelIO₁ :=
  { naiveIO x y B with kernel := stableSoftmaxKernel x y B, projection := by rfl }

theorem same_signature (x y : RegionName) (B : Nat) :
    io₁Signature (naiveIO x y B) = io₁Signature (stableIO x y B) := rfl

end VeriTile.Bench.Examples.SoftmaxStableFPExecution

/- The libdevice.exp softmax shift, derived from its scalar EXP-SUB law.
All reduction and normalization identities are derived below. -/

namespace VeriTile.Bench.Examples.SoftmaxStableFPContract
open VeriTile.Bench.Examples.SoftmaxStable.Kernels
open VeriTile Triton FP.Structural FP.Guarded FP.ScalarArithmetic FP.ScalarReduction
open FP.GuardExpression
open _root_.VeriTile.Triton.FP.Equational (ReductionPlan Schedules)
open SoftmaxStableFPExecution

/-- The original default arithmetic is resolved to fp32, with the sum's
layout-dependent addition schedule explicit and the max interpretation opaque. -/
def engine {α : Type} (M : Algebra α) (plans : Schedules) : Algebra α :=
  FP.Scheduled.fp32.algebra M plans

def rowPlan (plans : Schedules) (B : Nat) : ReductionPlan B :=
  plans (some .fp32) [B] ⟨0, by simp⟩ Bool.false PUnit.unit

def center {α : Type} (M : Algebra α) (xs : Fin B → α) : α :=
  M.reduceMax (some .fp32) (shape := [B]) ⟨0, by simp⟩ Bool.false (fun i => xs i.1) PUnit.unit

def rowExpressions (x : RegionName) (B : Nat) (i : Fin B) : Expr MemoryInput :=
  .input ⟨x, fun pid => pid * B + i.val, .real⟩

def rowValues {α : Type} [Inhabited α] (s : State α) (x : RegionName) (B : Nat) (i : Fin B) : α :=
  (s.mem x (s.pids 0 * B + i.val)).read .real

@[simp] theorem eval_center {α Input : Type} (M : Algebra α) (read : Input → α)
    (xs : Fin B → Expr Input) :
    (center (FP.GuardExpression.algebra Input) xs).eval M read =
      center M (fun i => (xs i).eval M read) := rfl

@[simp] theorem eval_row {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (x : RegionName) (B : Nat) (i : Fin B) :
    (rowExpressions x B i).eval M (MemoryInput.read s) = rowValues s x B i := rfl

def requirements (x : RegionName) (B : Nat) (plans : Schedules) : Condition :=
  let M := FP.GuardExpression.algebra MemoryInput
  FP.SoftmaxShift.checks M (rowExpressions x B) (center M (rowExpressions x B)) (rowPlan plans B).tree

theorem requirements_holds {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (s : State α) (x : RegionName) (B : Nat) (plans : Schedules) :
    (requirements x B plans).Holds M D s ↔
      FP.SoftmaxShift.ShiftDomain M D (rowValues s x B) (center M (rowValues s x B))
        (rowPlan plans B).tree := by
  unfold requirements Condition.Holds
  rw [← Requirements.holds_map]
  simp only [FP.SoftmaxShift.map_checks, eval_center, eval_row, FP.SoftmaxShift.checks_holds]

def stable (x y : RegionName) (B : Nat) : FP.Scheduled.IO₁ :=
  ⟨stableIO x y B, FP.Scheduled.fp32, requirements x B⟩

def naive (x y : RegionName) (B : Nat) : FP.Scheduled.IO₁ :=
  ⟨naiveIO x y B, FP.Scheduled.fp32, requirements x B⟩

theorem same_signature (x y : RegionName) (B : Nat) :
    Spec.ProgramSyntax.signature (stable x y B) = Spec.ProgramSyntax.signature (naive x y B) := rfl

theorem sum_value {α : Type} (M : Algebra α) (plans : Schedules) (xs : Fin B → α) :
    rowSum (engine M plans) xs = value M xs (zero M) (rowPlan plans B).tree := rfl

/-- The common output cast preserves the scalar-derived normalization
identity. Exp retains the tested libdevice implementation identity. -/
theorem original_values {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α) (plans : Schedules)
    (hExp : FP.SoftmaxShift.LibdeviceExpSub M D) (xs : Fin B → α)
    (hd : FP.SoftmaxShift.ShiftDomain M D xs (center M xs) (rowPlan plans B).tree) (i : Fin B) :
    normalizedValue (engine M plans) (shifted (engine M plans) xs) i =
      normalizedValue (engine M plans) (exponentials (engine M plans) xs) i := by
  unfold normalizedValue
  rw [sum_value, sum_value]
  exact congrArg (M.cast (some .fp32) .real .bf16)
    (FP.SoftmaxShift.normalized R M D hM s hExp xs (center M xs) (rowPlan plans B).tree hd i)

/-- The original programs succeed, agree in every bf16 output lane and frame
all other memory under the selected arithmetic theory, syntactic domain and
matching scalar EXP-SUB law. SoftmaxStableFPEquiv binds that law to admission. -/
theorem original_runs_under_exp {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α) (plans : Schedules)
    (x y : RegionName) (B : Nat) (hB : 0 < B)
    (hExp : FP.SoftmaxShift.LibdeviceExpSub M D) (hd : (requirements x B plans).Holds M D s) :
    IO₁PrivateScratch (stable x y B).io ∧ IO₁PrivateScratch (naive x y B).io ∧
    ∃ a b,
      FP.Structural.exec ((stable x y B).profile.algebra M plans) (stable x y B).io.kernel s = some a ∧
      FP.Structural.exec ((naive x y B).profile.algebra M plans) (naive x y B).io.kernel s = some b ∧
      (∀ i : Fin B, a.mem y (s.pids 0 * B + i.val) = b.mem y (s.pids 0 * B + i.val)) ∧
      IO₁Frame (stable x y B).io s a ∧ IO₁Frame (naive x y B).io s b := by
  let xs := rowValues s x B
  have hd' := (requirements_holds M D s x B plans).mp hd
  obtain ⟨a, ha, hva, hfa⟩ := stable_run (engine M plans) x y B hB xs s (fun _ => rfl)
  obtain ⟨b, hb, hvb, hfb⟩ := naive_run (engine M plans) x y B xs s (fun _ => rfl)
  refine ⟨by simp [stable, IO₁PrivateScratch, stableIO, naiveIO],
    by simp [naive, IO₁PrivateScratch, naiveIO], a, b, ha, hb, ?_,
    (fun r o ho _ => hfa r o ho), (fun r o ho _ => hfb r o ho)⟩
  intro i
  exact (hva i).trans ((congrArg (Cell.mk .bf16)
    (original_values R M D hM s plans hExp xs hd' i)).trans (hvb i).symm)

end VeriTile.Bench.Examples.SoftmaxStableFPContract
