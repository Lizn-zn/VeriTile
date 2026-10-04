import bench.examples.OnlineSoftmax.Kernels
/- Use libdevice.exp for exp-sub rewrites: the measured fp32 tl.exp relation
has B = 0.1608954387 ULP > 0.05 under the configured Normal(1,1) probe.
That intrinsic relation failed admission; the libdevice EXP-SUB instance passed. -/
/- The original OnlineSoftmax batch reference stores real-typed values.
The shared source retains its original behavior; no bf16 cast or online store is added. -/
import bench.examples.SoftmaxStable.Execution

namespace VeriTile.Bench.Examples.OnlineSoftmaxFPBatch
open VeriTile.Bench.Examples.OnlineSoftmax.Kernels
open VeriTile Triton FP.Structural
open SoftmaxStableFPExecution (maximum shifted rowSum)

set_option maxHeartbeats 1600000


def outputValue {α : Type} (M : Algebra α) (xs : Fin N → α) (i : Fin N) : α :=
  M.binary none .real .div (shifted M xs i) (rowSum M (shifted M xs))

theorem batch_run {α : Type} [Inhabited α] (M : Algebra α)
    (x y : RegionName) (N : Nat) (hN : 0 < N) (xs : Fin N → α) (s : State α)
    (hx : ∀ i : Fin N, (s.mem x (s.pids 0 * N + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (stableSoftmaxKernel x y N) s = some t ∧
      (∀ i : Fin N, t.mem y (s.pids 0 * N + i.val) = .mk .real (outputValue M xs i)) ∧
      (∀ r o, (r ≠ y ∨ ∀ i : Fin N, o ≠ s.pids 0 * N + i.val) → t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [N] => s.pids 0 * N + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [stableSoftmaxKernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, TileShape.axisDim, TileShape.eraseAxis, hN,
    Region.cast, hx, outputValue, shifted, maximum, rowSum]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    rfl
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

end VeriTile.Bench.Examples.OnlineSoftmaxFPBatch
