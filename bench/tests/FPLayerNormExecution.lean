/- Boundaries of the original LayerNorm execution and scheduled contract.
These fixtures do not supply experimental evidence or admit count relations. -/
import bench.examples.FusedLayerNorm.Contract
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.NormNum

open VeriTile.Bench.Examples.FusedLayerNorm.Kernels

namespace FPLayerNormExecutionTests
open VeriTile Triton FP.Structural
open VeriTile.Bench.Examples
open LayerNormFPExecution
open scoped VeriTile.Spec

-- Preserve the original statement body while explicitly declaring the three
-- public input ports once, including the second x read in the fused source.
example (x g b y : RegionName) (N stride : Nat) (ε : ℝ) :
    (fusedIO x g b y N stride ε).kernel.surfaceBody =
      (fusedLayerNormKernel x g b y N stride ε).surfaceBody := rfl
example (x g b y : RegionName) (N stride : Nat) (ε : ℝ) :
    (twopassIO x g b y N stride ε).kernel.surfaceBody =
      (twoPassLayerNormKernel x g b y N stride ε).surfaceBody := rfl

-- The loop's memory preservation makes the second read sound even in-place;
-- the execution theorem needs no disjointness assumption on the input ports.
example {α : Type} [Inhabited α] (M : Algebra α) (N stride : Nat) (ε : ℝ)
    (xs gs bs : Fin N → α) (s : State α)
    (hx : ∀ i : Fin N, (s.mem "x" (s.pids 0 * stride + i.val)).read .real = xs i)
    (hg : ∀ i : Fin N, (s.mem "gamma" i.val).read .real = gs i)
    (hb : ∀ i : Fin N, (s.mem "beta" i.val).read .real = bs i) :
    ∃ t, FP.Structural.exec M (fusedLayerNormKernel "x" "gamma" "beta" "x" N stride ε) s = some t ∧
      (∀ i : Fin N, t.mem "x" (s.pids 0 * stride + i.val) =
        .mk .bf16 (outputValue M ε (xs i) (gs i) (bs i)
          (WelfordFPExecution.recurrence M xs N).1 (onlineVariance M xs))) ∧
      (∀ (r : RegionName) o, (r ≠ "x" ∨ ∀ i : Fin N, o ≠ s.pids 0 * stride + i.val) → t.mem r o = s.mem r o) :=
  fused_run M "x" "gamma" "beta" "x" stride ε xs gs bs s hx hg hb

private noncomputable def model : Algebra ℚ where
  literal := fun _ _ r => if r = 0 then 0 else 1
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a + b
    | .sub => a - b
    | .mul => a * b
    | .div => a / b
    | .max => max a b
    | .pow => a
  unary := fun _ _ _ => 1
  cast := fun _ _ _ a => if a = 0 then 0 else 1
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun b => b.bits.toNat
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

-- An opaque noninjective output cast can equate two means while the affine
-- results still differ. Equality of Welford's stored bf16 outputs alone is
-- therefore insufficient for the LayerNorm replacement.
theorem rounded_statistics_do_not_suffice :
    (model.cast none .real .bf16 1, model.cast none .real .bf16 0) =
      (model.cast none .real .bf16 2, model.cast none .real .bf16 0) ∧
    outputValue model 1 1 1 0 1 0 ≠ outputValue model 1 1 1 0 2 0 := by
  norm_num [model, outputValue]

private def empty : FP.Equational.ReductionPlan 0 := .withSeed 0
private def lhs : FP.Scheduled.IO₃ := LayerNormFPContract.fused "x" "gamma" "beta" "y" 2 8 1 empty
private def rhs : FP.Scheduled.IO₃ := LayerNormFPContract.twopass "x" "gamma" "beta" "y" 2 8 1 empty

theorem gamma_layout_is_observable :
    Spec.ProgramSyntax.signature lhs ≠ Spec.ProgramSyntax.signature
      { lhs with io := { lhs.io with read2 := fun pid => pid * 8 } } := by
  intro h
  have hr := congrArg (fun sig => sig.1.read2 1) h
  cases hr

theorem beta_port_is_observable :
    Spec.ProgramSyntax.signature lhs ≠ Spec.ProgramSyntax.signature
      { lhs with io := { lhs.io with in3 := "wrong_beta" } } := by
  intro h
  have hr := congrArg (fun sig => sig.1.in3.name) h
  exact (by decide : ("beta" : String) ≠ "wrong_beta") hr

theorem precision_is_observable :
    Spec.ProgramSyntax.signature lhs ≠ Spec.ProgramSyntax.signature
      { lhs with profile := ⟨.fp64, Bool.true⟩ } := by
  intro h
  have hp := congrArg (fun sig => sig.2.1.defaultPrecision) h
  cases hp

theorem reduction_profile_is_observable :
    Spec.ProgramSyntax.signature lhs ≠ Spec.ProgramSyntax.signature
      { lhs with profile := ⟨.fp32, Bool.false⟩ } := by
  intro h
  have hp := congrArg (fun sig => sig.2.1.fp32SumTrees) h
  cases hp

theorem statistics_domain_is_observable :
    Spec.ProgramSyntax.signature lhs ≠ Spec.ProgramSyntax.signature
      { lhs with domain := fun _ => .top } := by
  intro h
  have hp := congrArg (fun sig => sig.2.2 FP.Equational.seededSchedules) h
  cases hp

private def initial : State ℚ where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

theorem output_neighbor_stays_framed :
    ¬ Frame lhs.io initial (initial.write "y" 10 (.mk .bf16 0)) := by
  intro h
  have hv := h "y" 10 (Or.inr (by
    intro i
    have hi := i.isLt
    change i.val < 2 at hi
    change 10 ≠ 8 + i.val
    omega)) (by simp [lhs, LayerNormFPContract.fused, fusedIO, twopassIO])
  simp [initial, State.write] at hv

-- A nontrivial whole-kernel premise remains explicitly unresolved in the
-- assumption printer. A derived empty-row structural boundary prints none.
specification opaque_layernorm
    (h : FP.Scheduled.Equivalent₃ [] lhs rhs) : lhs ≡[[]] rhs :=
  Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl rfl h

specification empty_layernorm :
    (LayerNormFPContract.fused "x" "gamma" "beta" "y" 0 8 1 empty) ≡[[]]
      (LayerNormFPContract.twopass "x" "gamma" "beta" "y" 0 8 1 empty) :=
  Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl rfl
    (FP.Scheduled.Equivalent₃.ofStructural [] rfl (LayerNormFPContract.empty_structural _ _ _ _ _ _))

#axiomsClean WelfordFPExecution.loop_run_with_pid
#axiomsClean WelfordFPComparison.original_statistics
#axiomsClean LayerNormFPExecution.fused_run
#axiomsClean LayerNormFPExecution.twopass_run
#axiomsClean LayerNormFPContract.original_values
#axiomsClean LayerNormFPContract.original_runs_under_count
#axiomsClean rounded_statistics_do_not_suffice
#axiomsClean output_neighbor_stays_framed
#axiomsClean gamma_layout_is_observable
#axiomsClean beta_port_is_observable
#axiomsClean precision_is_observable
#axiomsClean reduction_profile_is_observable
#axiomsClean statistics_domain_is_observable
#axiomsClean empty_layernorm
#print_fp_assumptions opaque_layernorm
#print_fp_assumptions empty_layernorm

end FPLayerNormExecutionTests
