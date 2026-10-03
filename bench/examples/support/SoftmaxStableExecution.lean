/- Opaque execution of the original naive and stable tl.exp softmax kernels.
The outputs retain their bf16 casts; max, exp and sum have no implicit laws. -/
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.StructuralIO

namespace VeriTile.Bench.Examples.SoftmaxStableFPExecution
open VeriTile Triton FP.Structural

set_option maxHeartbeats 1600000

def naiveSoftmaxKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(B) + tl.arange(0, $(B))
  x := tl.load($(xReg) + offs)
  e := tl.exp(x)
  s := tl.sum(e, axis=0)
  y := e / s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

def stableSoftmaxKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(B) + tl.arange(0, $(B))
  x := tl.load($(xReg) + offs)
  m := tl.max(x, axis=0)
  e := tl.exp(x - m)
  s := tl.sum(e, axis=0)
  y := e / s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

def maximum {α : Type} (M : Algebra α) (xs : Fin B → α) : α :=
  M.reduceMax none (shape := [B]) ⟨0, by simp⟩ Bool.false (fun i => xs i.1) PUnit.unit

def exponentials {α : Type} (M : Algebra α) (xs : Fin B → α) : Fin B → α :=
  fun i => M.unary none .exp (xs i)

def shifted {α : Type} (M : Algebra α) (xs : Fin B → α) : Fin B → α :=
  fun i => M.unary none .exp (M.binary none .real .sub (xs i) (maximum M xs))

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
