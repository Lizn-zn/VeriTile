import bench.examples.FusedLayerNorm.Kernels
import bench.examples.Welford.Proofs.FP

/-! FP proof support for FusedLayerNorm. The public specification and atomic
assumption report are in ../FPEquiv.lean; the shared sources are in ../Kernels.lean.
Execution, intermediate-value domains and their composition are kept together. -/

/- Opaque FP execution of the original two-pass and Welford-based LayerNorm
sources. No numerical identity is used to evaluate the affine suffix. The
shared sources are imported from Kernels.lean, without the real correctness proofs. -/

namespace VeriTile.Bench.Examples.LayerNormFPExecution
open VeriTile.Bench.Examples.FusedLayerNorm.Kernels
open VeriTile Triton FP.Structural
open WelfordFPExecution (recurrence twopassMean twopassVariance)

set_option maxHeartbeats 1600000


/-- The common affine suffix, with every operation and its order retained. -/
def outputValue {α : Type} (M : Algebra α) (ε : ℝ) (x gamma beta mean variance : α) : α :=
  let inverse := M.binary none .real .div (M.literal none .real 1)
    (M.unary none .sqrt (M.binary none .real .add variance (M.literal none .real ε)))
  M.cast none .real .bf16
    (M.binary none .real .add
      (M.binary none .real .mul
        (M.binary none .real .mul (M.binary none .real .sub x mean) inverse) gamma) beta)

def onlineVariance {α : Type} (M : Algebra α) (xs : Fin N → α) : α :=
  M.binary none .real .div (recurrence M xs N).2 (M.fromNat none N)

/-- This is the unchanged source after the Welford loop, not a replacement
kernel. The preceding four statements initialize and run the recurrence. -/
def suffix (x g b y : RegionName) (N stride : Nat) (ε : ℝ) : List ComputeStmt :=
  (fusedLayerNormKernel x g b y N stride ε).surfaceBody.drop 4

theorem fused_body (x g b y : RegionName) (N stride : Nat) (ε : ℝ) :
    (fusedLayerNormKernel x g b y N stride ε).surfaceBody =
      WelfordFPExecution.initialCode ++ [.forLoop "i" N (WelfordFPExecution.body x stride)] ++
        suffix x g b y N stride ε := rfl

/-- Both reductions stay opaque. The final store preserves bf16 and frames
all cells outside the original row; gamma and beta use feature offsets. -/
theorem twopass_run {α : Type} [Inhabited α] (M : Algebra α)
    (x g b y : RegionName) (stride : Nat) (ε : ℝ) (xs gs bs : Fin N → α) (s : State α)
    (hx : ∀ i : Fin N, (s.mem x (s.pids 0 * stride + i.val)).read .real = xs i)
    (hg : ∀ i : Fin N, (s.mem g i.val).read .real = gs i)
    (hb : ∀ i : Fin N, (s.mem b i.val).read .real = bs i) :
    ∃ t, FP.Structural.exec M (twoPassLayerNormKernel x g b y N stride ε) s = some t ∧
      (∀ i : Fin N, t.mem y (s.pids 0 * stride + i.val) =
        .mk .bf16 (outputValue M ε (xs i) (gs i) (bs i) (twopassMean M xs) (twopassVariance M xs))) ∧
      (∀ r o, (r ≠ y ∨ ∀ i : Fin N, o ≠ s.pids 0 * stride + i.val) → t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [N] => s.pids 0 * stride + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [twoPassLayerNormKernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, TileShape.eraseAxis, Region.cast, ofFloat, toFloat,
    hx, hg, hb, WelfordFPExecution.sumValue, twopassMean, twopassVariance, outputValue]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    rfl
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

/-- The loop leaves input memory and pid intact; the original suffix reads
x again and uses the unrounded statistics in its unchanged affine expression. -/
theorem fused_run {α : Type} [Inhabited α] (M : Algebra α)
    (x g b y : RegionName) (stride : Nat) (ε : ℝ) (xs gs bs : Fin N → α) (s : State α)
    (hx : ∀ i : Fin N, (s.mem x (s.pids 0 * stride + i.val)).read .real = xs i)
    (hg : ∀ i : Fin N, (s.mem g i.val).read .real = gs i)
    (hb : ∀ i : Fin N, (s.mem b i.val).read .real = bs i) :
    ∃ t, FP.Structural.exec M (fusedLayerNormKernel x g b y N stride ε) s = some t ∧
      (∀ i : Fin N, t.mem y (s.pids 0 * stride + i.val) =
        .mk .bf16 (outputValue M ε (xs i) (gs i) (bs i) (recurrence M xs N).1 (onlineVariance M xs))) ∧
      (∀ r o, (r ≠ y ∨ ∀ i : Fin N, o ≠ s.pids 0 * stride + i.val) → t.mem r o = s.mem r o) := by
  obtain ⟨u, hu, hm, hv, hpid, hmem, _⟩ :=
    WelfordFPExecution.loop_run_with_pid M x stride xs s hx
  have he : FP.Structural.exec M (fusedLayerNormKernel x g b y N stride ε) s =
      run M (WelfordFPExecution.initialCode ++ [.forLoop "i" N (WelfordFPExecution.body x stride)] ++
        suffix x g b y N stride ε) s :=
    congrArg (fun code => run M code s) (fused_body x g b y N stride ε)
  have hinj : Function.Injective (fun i : TileIndex [N] => s.pids 0 * stride + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  rw [he, run_append, hu]
  simp [suffix, fusedLayerNormKernel, ComputeKernel.surfaceBody, List.drop, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, Region.cast, ofFloat, toFloat,
    hm, hv, hpid, hmem, hx, hg, hb, outputValue, onlineVariance]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
  · trans u.mem r o
    · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
      rcases hmiss with hr | ho
      · exact Or.inl hr
      · exact Or.inr fun k _ => ho k.1
    · exact congrFun (congrFun hmem r) o

/-- Declare the three public input ports once each. The DSL's metadata lists
each antiquoted load occurrence, including the fused kernel's second x read;
the explicit IO declaration preserves the entire original statement body. -/
def twopassIO (x g b y : RegionName) (N stride : Nat) (ε : ℝ) : KernelIO₃ where
  kernel := .mk [x, g, b] [y] (twoPassLayerNormKernel x g b y N stride ε).surfaceBody
  projection := by rfl
  in1 := x
  in2 := g
  in3 := b
  out := y
  B1 := N
  B2 := N
  B3 := N
  Bout := N
  read1 := fun pid => pid * stride
  read2 := fun _ => 0
  read3 := fun _ => 0
  write := fun pid => pid * stride

def fusedIO (x g b y : RegionName) (N stride : Nat) (ε : ℝ) : KernelIO₃ :=
  { twopassIO x g b y N stride ε with
    kernel := .mk [x, g, b] [y] (fusedLayerNormKernel x g b y N stride ε).surfaceBody
    projection := by rfl }

theorem same_signature (x g b y : RegionName) (N stride : Nat) (ε : ℝ) :
    ioSignature (twopassIO x g b y N stride ε) =
      ioSignature (fusedIO x g b y N stride ε) := rfl

end VeriTile.Bench.Examples.LayerNormFPExecution

/- The original LayerNorm transformation under the selected scalar theory.
The statistics comparison is derived before the common affine suffix. Count
conversion is explicit in this reusable support module; FusedLayerNormFPEquiv
supplies it from the accepted bounded scalar atoms. -/

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
