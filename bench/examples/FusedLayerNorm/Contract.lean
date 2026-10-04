import bench.examples.FusedLayerNorm.Kernels
/- The original LayerNorm transformation under the selected scalar theory.
The statistics comparison is derived before the common affine suffix. Count
conversion is explicit in this reusable support module; FusedLayerNormFPEquiv
supplies it from the accepted bounded scalar atoms. -/
import bench.examples.FusedLayerNorm.Execution
import bench.examples.Welford.Contract

namespace VeriTile.Bench.Examples.LayerNormFPContract
open VeriTile.Bench.Examples.FusedLayerNorm.Kernels
open VeriTile Triton FP.Structural FP.Guarded FP.ScalarArithmetic
open FP.WelfordInduction FP.WelfordSchedule
open _root_.VeriTile.Triton.FP.Equational (ReductionPlan Schedules)
open LayerNormFPExecution
open WelfordFPComparison (engine rowPlan)

/-- The common affine computation needs only congruence once both unrounded
statistics agree. Sqrt, epsilon, reciprocal, gamma, beta and the bf16 cast
remain their original operations; none becomes an additional numerical law. -/
theorem original_values {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α) (plans : Schedules)
    (xs : Nat → α) (empty : ReductionPlan 0) (N : Nat) (hN : 0 < N)
    (hc : CountConversion M N) (hi : IterationDomain M D xs empty.tree N)
    (hs : StatisticsDomain M D (rowPrefix xs N) (plan empty N) (rowPlan plans N))
    (ε : ℝ) (x gamma beta : α) :
    outputValue (engine M plans) ε x gamma beta
        (WelfordFPExecution.recurrence (engine M plans) (rowPrefix xs N) N).1
        (onlineVariance (engine M plans) (rowPrefix xs N)) =
      outputValue (engine M plans) ε x gamma beta
        (WelfordFPExecution.twopassMean (engine M plans) (rowPrefix xs N))
        (WelfordFPExecution.twopassVariance (engine M plans) (rowPrefix xs N)) := by
  exact congrArg (fun stats : α × α => outputValue (engine M plans) ε x gamma beta stats.1 stats.2)
    (WelfordFPComparison.original_statistics R M D hM s plans xs empty N hN hc hi hs)

/-- Empty output rows execute both original programs and preserve all memory;
no arithmetic or count relation is needed for this boundary case. -/
theorem empty_structural (x g b y : RegionName) (stride : Nat) (ε : ℝ) :
    IO₃Equiv (fusedIO x g b y 0 stride ε) (twopassIO x g b y 0 stride ε) := by
  refine ⟨by simp [PrivateScratch, fusedIO, twopassIO],
    by simp [PrivateScratch, twopassIO], ?_⟩
  intro α _ M s
  obtain ⟨a, ha, _, hfa⟩ := fused_run M x g b y stride ε Fin.elim0 Fin.elim0 Fin.elim0 s
    (fun i => Fin.elim0 i) (fun i => Fin.elim0 i) (fun i => Fin.elim0 i)
  obtain ⟨b, hb, _, hfb⟩ := twopass_run M x g b y stride ε Fin.elim0 Fin.elim0 Fin.elim0 s
    (fun i => Fin.elim0 i) (fun i => Fin.elim0 i) (fun i => Fin.elim0 i)
  exact ⟨a, b, ha, hb, (fun i => Fin.elim0 i),
    (fun r o h _ => hfa r o h), (fun r o h _ => hfb r o h)⟩

/-- Nonempty rows use exactly Welford's compiled statistics domain. Empty
rows have no numerical output and require no domain checks. -/
def requirements (x : RegionName) (N stride : Nat) (empty : ReductionPlan 0)
    (plans : Schedules) : FP.GuardExpression.Condition :=
  if N = 0 then .top else WelfordFPContract.requirements x N stride empty plans

def fused (x g b y : RegionName) (N stride : Nat) (ε : ℝ) (empty : ReductionPlan 0) :
    FP.Scheduled.IO₃ :=
  ⟨fusedIO x g b y N stride ε, FP.Scheduled.fp32, requirements x N stride empty⟩

def twopass (x g b y : RegionName) (N stride : Nat) (ε : ℝ) (empty : ReductionPlan 0) :
    FP.Scheduled.IO₃ :=
  ⟨twopassIO x g b y N stride ε, FP.Scheduled.fp32, requirements x N stride empty⟩

theorem same_signature (x g b y : RegionName) (N stride : Nat) (ε : ℝ) (empty : ReductionPlan 0) :
    Spec.ProgramSyntax.signature (fused x g b y N stride ε empty) =
      Spec.ProgramSyntax.signature (twopass x g b y N stride ε empty) := rfl

/-- Successful executions, every typed output lane, and both frames for the
original transformation. The primitive count obligations remain explicit;
no output equality or whole-normalization atom is a premise. -/
theorem original_runs_under_count {α : Type} [Inhabited α] (R : Rules) (M : Algebra α)
    (D : Domain α) (hM : Models R.assumptions M D) (s : State α) (plans : Schedules)
    (x g b y : RegionName) (N stride : Nat) (ε : ℝ) (empty : ReductionPlan 0)
    (hc : CountConversion M N)
    (hd : (requirements x N stride empty plans).Holds M D s) :
    PrivateScratch (fused x g b y N stride ε empty).io ∧
    PrivateScratch (twopass x g b y N stride ε empty).io ∧
    ∃ a z,
      FP.Structural.exec ((fused x g b y N stride ε empty).profile.algebra M plans)
        (fused x g b y N stride ε empty).io.kernel s = some a ∧
      FP.Structural.exec ((twopass x g b y N stride ε empty).profile.algebra M plans)
        (twopass x g b y N stride ε empty).io.kernel s = some z ∧
      (∀ i : Fin N, a.mem y (s.pids 0 * stride + i.val) = z.mem y (s.pids 0 * stride + i.val)) ∧
      Frame (fused x g b y N stride ε empty).io s a ∧
      Frame (twopass x g b y N stride ε empty).io s z := by
  by_cases hzero : N = 0
  · subst N
    have h := empty_structural x g b y stride ε
    exact ⟨h.1, h.2.1, h.2.2 α (engine M plans) s⟩
  have hN : 0 < N := Nat.pos_of_ne_zero hzero
  have hd' : (WelfordFPContract.requirements x N stride empty plans).Holds M D s := by
    simpa only [requirements, if_neg hzero] using hd
  obtain ⟨hi, hs⟩ := (WelfordFPContract.requirements_holds M D s x N stride empty plans).mp hd'
  let xs := WelfordFPContract.rowValues s x stride
  let gs : Fin N → α := fun i => (s.mem g i.val).read .real
  let bs : Fin N → α := fun i => (s.mem b i.val).read .real
  obtain ⟨a, ha, hva, hfa⟩ := fused_run (engine M plans) x g b y stride ε
    (rowPrefix xs N) gs bs s (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
  obtain ⟨z, hz, hvz, hfz⟩ := twopass_run (engine M plans) x g b y stride ε
    (rowPrefix xs N) gs bs s (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
  refine ⟨by simp [fused, PrivateScratch, fusedIO, twopassIO],
    by simp [twopass, PrivateScratch, twopassIO], a, z, ha, hz, ?_,
    (fun r o ho _ => hfa r o ho), (fun r o ho _ => hfz r o ho)⟩
  intro i
  exact (hva i).trans ((congrArg (Cell.mk .bf16)
    (original_values R M D hM s plans xs empty N hN hc hi hs ε (xs i.val) (gs i) (bs i))).trans
      (hvz i).symm)

end VeriTile.Bench.Examples.LayerNormFPContract
