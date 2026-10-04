import bench.examples.StableLogSumExp.Kernels
/- Use libdevice.exp for exp-sub rewrites: the measured fp32 tl.exp relation
has B = 0.1608954387 ULP > 0.05 under the configured Normal(1,1) probe.
That intrinsic relation failed admission; the libdevice EXP-SUB instance passed. -/
/- Original direct and stable logsumexp execution under opaque FP operations.
The source's scalar output address and bf16 conversion remain explicit. -/
import bench.examples.SoftmaxStable.Execution

namespace VeriTile.Bench.Examples.StableLogSumExpFPExecution
open VeriTile.Bench.Examples.StableLogSumExp.Kernels
open VeriTile Triton FP.Structural
open SoftmaxStableFPExecution (maximum exponentials shifted rowSum)

set_option maxHeartbeats 1600000


def directValue {α : Type} (M : Algebra α) (xs : Fin B → α) : α :=
  M.cast none .real .bf16 (M.unary none .log (rowSum M (exponentials M xs)))

def stableValue {α : Type} (M : Algebra α) (xs : Fin B → α) : α :=
  M.cast none .real .bf16 (M.binary none .real .add (maximum M xs)
    (M.unary none .log (rowSum M (shifted M xs))))

theorem direct_run {α : Type} [Inhabited α] (M : Algebra α)
    (x y : RegionName) (B : Nat) (xs : Fin B → α) (s : State α)
    (hx : ∀ i : Fin B, (s.mem x (s.pids 0 * B + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (directLSEKernel x y B) s = some t ∧
      t.mem y (s.pids 0) = .mk .bf16 (directValue M xs) ∧
      (∀ r o, (r ≠ y ∨ o ≠ s.pids 0) → t.mem r o = s.mem r o) := by
  simp [directLSEKernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, State.write, TileShape.allIndices,
    TileShape.eraseAxis, Region.cast, ofFloat, toFloat, hx, directValue, exponentials, rowSum]
  refine ⟨rfl, ?_⟩
  intro r o hmiss hr ho
  exact (hmiss.elim (fun h => h hr) (fun h => h ho)).elim

theorem stable_run {α : Type} [Inhabited α] (M : Algebra α)
    (x y : RegionName) (B : Nat) (hB : 0 < B) (xs : Fin B → α) (s : State α)
    (hx : ∀ i : Fin B, (s.mem x (s.pids 0 * B + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (stableLSEKernel x y B) s = some t ∧
      t.mem y (s.pids 0) = .mk .bf16 (stableValue M xs) ∧
      (∀ r o, (r ≠ y ∨ o ≠ s.pids 0) → t.mem r o = s.mem r o) := by
  simp [stableLSEKernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, State.write, TileShape.allIndices,
    TileShape.axisDim, TileShape.eraseAxis, hB, Region.cast, ofFloat, toFloat,
    hx, stableValue, shifted, maximum, rowSum]
  refine ⟨rfl, ?_⟩
  intro r o hmiss hr ho
  exact (hmiss.elim (fun h => h hr) (fun h => h ho)).elim

def directIO (x y : RegionName) (B : Nat) : KernelIO₁ where
  kernel := directLSEKernel x y B
  projection := by rfl
  inp := x
  out := y
  Bin := B
  Bout := 1
  read := fun pid => pid * B
  write := fun pid => pid

def stableIO (x y : RegionName) (B : Nat) : KernelIO₁ :=
  { directIO x y B with kernel := stableLSEKernel x y B, projection := by rfl }

theorem same_signature (x y : RegionName) (B : Nat) :
    io₁Signature (directIO x y B) = io₁Signature (stableIO x y B) := rfl

end VeriTile.Bench.Examples.StableLogSumExpFPExecution
