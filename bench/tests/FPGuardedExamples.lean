import bench.examples.VectorAdd.FPEquiv
import bench.examples.FlatVectorAdd.FPEquiv
import bench.examples.FloatDTypeAdd.FPEquiv
import bench.examples.TritonBenchVectorAddition.FPEquiv
import bench.examples.AdamUpdateGridLaunch.FPEquiv
import bench.examples.HyperConnectionsDepth.FPEquiv
import bench.examples.HyperConnectionsWidth.FPEquiv
import bench.examples.RowWiseSum.FPEquiv

/- Domain regressions exercise actual expression evaluation and intermediate
operands. Execution witnesses ensure that the contextual specifications are
not only comparing unsupported/failed computations. These are abstract models,
not new numerical admissions or claims about an IEEE implementation. -/
namespace FPGuardedExamplesTests
open VeriTile Triton FP.Structural FP.Guarded FP.GuardedRewrite
open VeriTile.Bench.Examples
set_option maxHeartbeats 1600000
set_option linter.unusedSimpArgs false

private def model : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ _ _ => 0
  unary := fun _ _ _ => 0
  compareEq := fun _ _ => some (fun _ _ => Bool.true)
  compareLt := fun _ _ => some (fun _ _ => Bool.false)
  compareLe := fun _ _ => some (fun _ _ => Bool.true)
  cast := fun _ _ _ _ => 0
  fromNat := fun _ _ => 0
  fromInt := fun _ _ => 0
  fp32Bits := fun _ => 0
  fp32Load := fun _ => 0
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

private def initial : State Nat where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

@[simp] private theorem write_regs (s : State Nat) (r : RegionName) (o : Nat) (v : Cell Nat) :
    (s.write r o v).regs = s.regs := rfl

-- All seven contextual examples have a concrete successful nonempty run.
example : ∃ t, FP.Structural.exec model (VectorAdd.Kernels.originalKernel 1) initial = some t := by
  simp [VectorAdd.Kernels.originalKernel, VectorAdd.Kernels.addKernel, FP.Structural.exec, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, model, initial,
    ComputeDType.eraseDType, numeric, numericEq, numericLt, natLt, bop, store,
    State.setReg, TileShape.allIndices, List.finRange_succ]
example : ∃ t, FP.Structural.exec model (FlatVectorAdd.Kernels.originalKernel 1 1) initial = some t := by
  simp [FlatVectorAdd.Kernels.originalKernel, FlatVectorAdd.Kernels.addKernelMasked, FP.Structural.exec, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, model, initial,
    ComputeDType.eraseDType, numeric, numericEq, numericLt, natLt, bop, store,
    State.setReg, TileShape.allIndices, List.finRange_succ]
example : ∃ t, FP.Structural.exec model (FloatDTypeAdd.Kernels.originalKernel 1) initial = some t := by
  simp [FloatDTypeAdd.Kernels.originalKernel, FloatDTypeAdd.Kernels.floatAddKernel, FP.Structural.exec, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, model, initial,
    ComputeDType.eraseDType, numeric, numericEq, numericLt, natLt, bop, store,
    State.setReg, TileShape.allIndices, List.finRange_succ]
example : ∃ t, FP.Structural.exec model (TritonBenchVectorAddition.Kernels.originalKernel 1 1) initial = some t := by
  simp [TritonBenchVectorAddition.Kernels.originalKernel, FP.Structural.exec, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, model, initial,
    ComputeDType.eraseDType, numeric, numericEq, numericLt, natLt, bop, store,
    State.setReg, TileShape.allIndices, List.finRange_succ]
example : ∃ t, FP.Structural.exec model (AdamUpdateGridLaunch.Kernels.originalKernel 0 0 0 0 1 1) initial = some t := by
  simp [AdamUpdateGridLaunch.Kernels.originalKernel, AdamUpdateGridLaunch.Kernels.update_fn_kernel, FP.Structural.exec, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, model, initial,
    ComputeDType.eraseDType, numeric, numericEq, numericLt, natLt, bop, store,
    State.setReg, TileShape.allIndices, List.finRange_succ]
example : ∃ t, FP.Structural.exec model (HyperConnectionsDepth.Kernels.originalKernel 1) initial = some t := by
  simp [HyperConnectionsDepth.Kernels.originalKernel, HyperConnectionsDepth.Kernels.mhcDepthConnectionKernel, FP.Structural.exec, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, model, initial,
    ComputeDType.eraseDType, numeric, numericEq, numericLt, natLt, bop, store,
    State.setReg, TileShape.allIndices, List.finRange_succ]
example : ∃ t, FP.Structural.exec model (HyperConnectionsWidth.Kernels.originalKernel 1) initial = some t := by
  simp [HyperConnectionsWidth.Kernels.originalKernel, HyperConnectionsWidth.Kernels.mhcWidthConnectionKernel, FP.Structural.exec, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, model, initial,
    ComputeDType.eraseDType, numeric, numericEq, numericLt, natLt, bop, store,
    State.setReg, TileShape.allIndices, List.finRange_succ]

-- The conditions are satisfiable under the same execution interpretation.
private def finite : Domain Nat := fun _ n => n = 0
example : (VectorAddFPEquiv.additionSite 1).Holds model finite initial := by
  intro t ht a b ha hb
  simp [VectorAddFPEquiv.additionSite, VectorAdd.Kernels.originalKernel,
    VectorAdd.Kernels.addKernel, ComputeKernel.surfaceBody, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, model, initial,
    ComputeDType.eraseDType, numeric, bop] at ht
  subst t
  simp [VectorAddFPEquiv.additionSite, evalOp_unfold, State.setReg] at ha hb
  subst a
  subst b
  exact ⟨fun _ => rfl, fun _ => rfl⟩

-- A finite leaf can produce a non-finite intermediate operand. The site
-- condition must reject that case rather than silently dropping the check.
private def overflow : Algebra Nat := { model with
  binary := fun _ _ op a b => match op with
    | .mul => 20
    | .add => a + b
    | _ => 0 }
private def bounded : Domain Nat := fun _ n => n < 10
private def operands : State Nat :=
  ((initial.setReg "a" .real [] (fun _ => 1)).setReg "b" .real [] (fun _ => 1)).setReg
    "c" .real [] (fun _ => 1)
private def productSite : Site :=
  ⟨[], [], .mul .real .nil (.ref .real [] "a") (.ref .real [] "b"), .ref .real [] "c"⟩

example : bounded .finite 1 := by norm_num [bounded]
example : ¬ productSite.Holds overflow bounded operands := by
  intro h
  have hp := h operands (by simp [productSite, run]) (fun _ => 20) (fun _ => 1) rfl rfl
  have impossible := hp.1 PUnit.unit
  norm_num [bounded] at impossible

-- This is the exact intermediate product required by the Adam addition.
example : (AdamUpdateGridLaunchFPEquiv.additionSite 0 0 0 2 1 1).left =
    .mul .real .scalarR (.ref .real [1] "diff") (.const 2) := rfl
-- Width checks exp(h/tau), and depth checks the complete weighted product.
example : (HyperConnectionsWidthFPEquiv.residualSite 1).left =
    .exp (.div .real .nil (.ref .real [] "h_res") (.const 1)) := rfl
example : (HyperConnectionsDepthFPEquiv.additionSite 1).right =
    .ref .real [] "branch_mix" := rfl

-- Finite reduction leaves alone do not justify commutation of partial sums.
private def largeSum : FP.Equational.ReductionTree 3 := .add (.input 0) (.input 1)
private def path := FP.ReductionSchedule.Rewrite.commute largeSum (.input (2 : Fin 3))
example : (∀ _ : Fin 3, bounded .finite (6 : Nat)) ∧
    ¬ path.Domain overflow bounded (fun _ => 6) := by
  constructor
  · intro _; norm_num [bounded]
  · intro h
    have impossible := h largeSum (by simp [path, FP.ReductionSchedule.Rewrite.operands])
    norm_num [largeSum, FP.ScalarReduction.value, FP.ScalarArithmetic.add,
      overflow, bounded] at impossible

-- Comparison backend identity and precision survive default-profile selection.
example : (model.withDefaultPrecision .fp32).compareEq none .real =
    model.compareEq (some .fp32) .real := rfl
-- Unordered equality is supplied independently from less-than.
private def unordered : Algebra Nat := { model with compareEq := fun _ _ => some (fun _ _ => Bool.false) }
example : FP.Structural.evalOp unordered (some .fp32)
    (.ne .real .nil (.const 0) (.const 0)) initial = some (fun _ => Bool.true) := rfl
example : FP.Structural.evalOp { model with compareEq := fun _ _ => none } (some .fp32)
    (.ne .real .nil (.const 0) (.const 0)) initial = none := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end FPGuardedExamplesTests
