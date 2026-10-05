import bench.examples.HyperConnectionsDepth.FPEquiv
import bench.examples.HyperConnectionsWidth.FPEquiv

/- Successful, nonsquare matrix executions with two normalization iterations.
These abstract execution witnesses are not numerical admission experiments. -/
namespace FPMatrixExamplesTests
open VeriTile Triton FP.Structural VeriTile.Bench.Examples
set_option maxHeartbeats 3200000
set_option linter.unusedSimpArgs false

private def model : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ _ _ => 0
  unary := fun _ _ _ => 0
  cast := fun _ _ _ _ => 0
  fromNat := fun _ _ => 0
  fromInt := fun _ _ => 0
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0
  dot := fun _ {_} {_} {_} {_} _ _ => some (fun _ => 0)

private def initial : State Nat where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 2
  numPids := fun _ => 4
  undef := fun d _ _ => defaultValue d

example : ∃ t, FP.Structural.exec model
    (HyperConnectionsDepth.Kernels.matrixOriginal 2 3 4 2 1) initial = some t := by
  simp [HyperConnectionsDepth.Kernels.matrixOriginal, HyperConnectionsDepth.Kernels.matrixKernel,
    FP.Structural.exec, run, step, loop, FP.Structural.evalExpr, evalOp_unfold, model,
    initial, State.setReg, bop, numeric, store, Option.bind]

example : ∃ t, FP.Structural.exec model
    (HyperConnectionsDepth.Kernels.matrixOptimized "res_mix" "branch_out" "h_post" "out" 2 3 4 2 1) initial = some t := by
  simp [HyperConnectionsDepth.Kernels.matrixOptimized,
    FP.Structural.exec, run, step, loop, FP.Structural.evalExpr, evalOp_unfold, model,
    initial, State.setReg, bop, numeric, store, Option.bind]

example : ∃ t, FP.Structural.exec model
    (HyperConnectionsWidth.Kernels.matrixOriginal 2 3 4 2 1) initial = some t := by
  simp (config := { maxSteps := 500000 }) [HyperConnectionsWidth.Kernels.matrixOriginal, HyperConnectionsWidth.Kernels.matrixKernel,
    FP.Structural.exec, run, step, loop, FP.Structural.evalExpr, evalOp_unfold, model,
    initial, State.setReg, bop, numeric, store, Option.bind]

example : ∃ t, FP.Structural.exec model
    (HyperConnectionsWidth.Kernels.matrixOptimized "res" "h_res" "h_pre" "res_mix" "branch_in" 2 3 4 2 1) initial = some t := by
  simp (config := { maxSteps := 500000 }) [HyperConnectionsWidth.Kernels.matrixOptimized,
    FP.Structural.exec, run, step, loop, FP.Structural.evalExpr, evalOp_unfold, model,
    initial, State.setReg, bop, numeric, store, Option.bind]

-- Missing backend support remains failure, rather than a made-up real dot.
example (s : State Nat) (a : Values Nat .real [2, 3]) (b : Values Nat .real [3, 4]) :
    FP.Structural.evalOp { model with dot := fun _ {_} {_} {_} {_} _ _ => none } none
      (.dot (batch := []) (.ref .real [2, 3] "a") (.ref .real [3, 4] "b"))
      ((s.setReg "a" .real [2, 3] a).setReg "b" .real [3, 4] b) = none := by
  simp [evalOp_unfold, State.setReg]

example (v : Values Nat .real [2, 3]) (i : Fin 2) (j : Fin 3) :
    transposeValues [] v (j, i, PUnit.unit) = v (i, j, PUnit.unit) := rfl

-- Precision selection reaches matrix operations too.
example (M : Algebra Nat) (a : Values Nat .real [2, 3]) (b : Values Nat .real [3, 4]) :
    (M.withDefaultPrecision .fp32).dot (batch := []) none a b = M.dot (batch := []) (some .fp32) a b := rfl

end FPMatrixExamplesTests
