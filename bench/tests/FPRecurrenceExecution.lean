/- These checks concern execution, not the pending batch/online numerical
equivalences. Neither source imports a Real correctness proof. -/
import bench.examples.support.WelfordExecution
import bench.examples.support.OnlineSoftmaxExecution
import VeriTile.Meta.StatementAudit

namespace FPRecurrenceExecutionTests
open VeriTile Triton FP.Structural
open VeriTile.Bench.Examples

-- Symbolic sizes and strides; no fixed GPU experiment shape enters execution.
example {α : Type} [Inhabited α] (M : Algebra α) (N stride : Nat)
    (xs : Fin N → α) (s : State α)
    (hx : ∀ i : Fin N, (s.mem "x" (s.pids 0 * stride + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (WelfordFPExecution.onlineWelfordKernel "x" "m" "v" N stride) s = some t ∧
      t.mem "m" 0 = .mk .bf16 (WelfordFPExecution.meanValue M xs) ∧
      t.mem "v" 0 = .mk .bf16 (WelfordFPExecution.varianceValue M xs) ∧
      (∀ (r : RegionName) o, (r ≠ "m" ∨ o ≠ 0) → (r ≠ "v" ∨ o ≠ 0) → t.mem r o = s.mem r o) :=
  WelfordFPExecution.online_run M "x" "m" "v" stride xs s (by decide) hx

-- The original empty-row source still reaches its two stores. There is no
-- fabricated nonzero count, nor a cancellation of the final opaque division.
example {α : Type} [Inhabited α] (M : Algebra α) (stride : Nat) (s : State α) :
    ∃ t, FP.Structural.exec M (WelfordFPExecution.onlineWelfordKernel "x" "m" "v" 0 stride) s = some t ∧
      t.mem "m" 0 = .mk .bf16 (M.cast none .real .bf16 (M.literal none .real 0)) ∧
      t.mem "v" 0 = .mk .bf16 (M.cast none .real .bf16
        (M.binary none .real .div (M.literal none .real 0) (M.fromNat none 0))) ∧
      (∀ (r : RegionName) o, (r ≠ "m" ∨ o ≠ 0) → (r ≠ "v" ∨ o ≠ 0) → t.mem r o = s.mem r o) := by
  exact WelfordFPExecution.online_run M "x" "m" "v" stride Fin.elim0 s (by decide)
    (fun i => Fin.elim0 i)

-- -inf is opaque and survives the empty online-softmax loop unchanged.
example {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, FP.Structural.exec M (OnlineSoftmaxFPExecution.onlineSoftmaxKernel "x" "y" 0) s = some t ∧
      t.regs .real [] "m" = some (fun _ => M.negInf) ∧
      t.regs .real [] "l" = some (fun _ => M.literal none .real 0) ∧
      t.mem = s.mem ∧ t.pids = s.pids :=
  OnlineSoftmaxFPExecution.online_run M "x" "y" Fin.elim0 s (fun i => Fin.elim0 i)

-- Merely having an integer conversion primitive supplies no floating count
-- law. This carrier distinguishes conversion from addition with literal one.
private def countModel : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ a b => a + b
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a
  fromNat := fun _ n => 20 + n
  fromInt := fun _ n => n.toNat
  fp32Bits := fun b => b.bits.toNat
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

example : countModel.fromNat none 2 ≠ countModel.binary none .real .add
    (countModel.fromNat none 1) (countModel.literal none .real 1) := by decide

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  for n in [`VeriTile.Bench.Examples.WelfordCorrect.onlineWelfordKernel,
            `VeriTile.Bench.Examples.OnlineSoftmax.onlineSoftmaxKernel] do
    if env.contains n then throwError "FP execution imported its Real correctness implementation"

#axiomsClean WelfordFPExecution.online_run
#axiomsClean WelfordFPExecution.twopass_run
#axiomsClean OnlineSoftmaxFPExecution.online_run

end FPRecurrenceExecutionTests
