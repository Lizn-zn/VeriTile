import VeriTile.Triton
import VeriTile.Examples.SoftmaxReciprocal
import VeriTile.Meta.StatementAudit

/-!
# Float dtype erasure — the float-facing *refinement* policy

Self-contained showcase, read top to bottom: the **kernels** first (what we
built), the **supporting lemmas** in the middle (`private` plumbing — the
erasure bridges and the exec-level view), the **theorem** last (one public
headline `float_softmax_reciprocal_refinement_view`), then a compile-time
**trust audit**. The three real sections below are
`FloatDTypeRefine.kernels`, `FloatDTypeRefine.lemmas`,
`FloatDTypeRefine.theorems`.

VeriTile's float-facing theorem policy on the **refinement** surface
(`ComputeRefine`): two kernels carry explicit `tl.float32` memory annotations,
and the algorithmic equivalence theorem is discharged by **erasing** those
annotations to the Real channel and reusing the Real-valued refinement (here
the softmax reciprocal rewrite). This is the erased/ideal pathway — the
complement of the rounding-model pathway (`ComputeRefine.*R`) demonstrated in
`bench/examples/FusedSwiglu.lean`. The correctness counterpart of this policy
lives in `bench/examples/FloatDTypeCorrect.lean`.

## The public result (bottom of file)

The single public headline is **`float_softmax_reciprocal_refinement_view`** —
the erased equivalence statement (`ComputeRefine.General`): the fp32-annotated
per-element-divide and reciprocal-form softmaxes agree cell-by-cell after
erasure. Its statement's project surface is pinned by the `#stmtSurfaceSubset`
gate below — **no spec** (this is a refinement surface). The erasure bridges
and the exec-level view are `private` scaffolding.
-/

namespace VeriTile.Bench.Examples.FloatDTypeRefine

open VeriTile.Triton
open VeriTile.Examples (stableSoftmaxKernel softmaxRecipKernel
  softmax_reciprocal_refinement_exec_view programTileView)

/-- Local reduction helper: the algorithm-projection traversals
(`ComputeStmt.toAlgorithm?` and friends) are written in `do` notation over
`Except`, so their equation lemmas leave `Except.ok _ >>= f` redexes behind.
This discharges them for `simp`. -/
@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

/-! ## Kernels -/
section FloatDTypeRefine.kernels

/-- Stable softmax with explicit fp32 input/output annotations and per-element
division `y = e / s`. The reductions still run in the Real abstraction after
the input load is cast to `tl.float64`. -/
def floatStableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := (tl.load($(xReg) + offs, dtype=tl.float32)).to(tl.float64)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, (y).to(tl.float32))
}

/-- Optimized stable softmax: precompute `1 / s` and use multiplication,
saving per-lane divisions versus `floatStableSoftmaxKernel`. -/
def floatSoftmaxRecipKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x      := (tl.load($(xReg) + offs, dtype=tl.float32)).to(tl.float64)
  m      := tl.max(x, axis=0)
  e      := tl.exp(x - m)
  s      := tl.sum(e, axis=0)
  inv_s  := 1 / s
  y      := e * inv_s
  tl.store($(yReg) + offs, (y).to(tl.float32))
}

end FloatDTypeRefine.kernels

/-! ## Supporting lemmas (private plumbing) -/
section FloatDTypeRefine.lemmas

/-- The fp32 per-element-divide softmax erases to the existing Real kernel,
at the algorithm-projection level. -/
private theorem float_stable_softmax_erases_to_real
    (xReg yReg : RegionName) (blockSize : Nat) :
    (floatStableSoftmaxKernel xReg yReg blockSize).eraseDType.toAlgKernel =
      (stableSoftmaxKernel xReg yReg blockSize).toAlgKernel := by
  simp [floatStableSoftmaxKernel, stableSoftmaxKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType]
  repeat' apply And.intro
  all_goals
    rw [Op.eraseDType.eq_def]
    simp [MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      Op.eraseDType.eq_def, VeriTile.Triton.eraseDType]

/-- The fp32 reciprocal-form softmax erases to the existing Real optimized
kernel, at the algorithm-projection level. -/
private theorem float_softmax_recip_erases_to_real
    (xReg yReg : RegionName) (blockSize : Nat) :
    (floatSoftmaxRecipKernel xReg yReg blockSize).eraseDType.toAlgKernel =
      (softmaxRecipKernel xReg yReg blockSize).toAlgKernel := by
  simp [floatSoftmaxRecipKernel, softmaxRecipKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType]
  repeat' apply And.intro
  all_goals
    rw [Op.eraseDType.eq_def]
    simp [MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      Op.eraseDType.eq_def, VeriTile.Triton.eraseDType]

/-- Float-facing rewrite refinement, exec-level view: the fp32
per-element-divide softmax and the fp32 reciprocal-form softmax are
algorithmically equivalent after erasure. State the dtype-annotated rewrite,
prove it through the existing Real refinement. -/
private theorem float_softmax_reciprocal_refinement_exec_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    ∀ idx : TileIndex [N],
      TensorView.observe
          (exec (floatStableSoftmaxKernel xReg yReg N).eraseDType s)
          (programTileView s yReg N) idx =
      TensorView.observe
          (exec (floatSoftmaxRecipKernel xReg yReg N).eraseDType s)
          (programTileView s yReg N) idx := by
  intro idx
  rw [float_stable_softmax_erases_to_real xReg yReg N,
    float_softmax_recip_erases_to_real xReg yReg N]
  exact softmax_reciprocal_refinement_exec_view xReg yReg N hN s xs h_x idx

end FloatDTypeRefine.lemmas

/-! ## The headline theorem -/
section FloatDTypeRefine.theorems

/- Shared parameters of the headline, hoisted to a `variable` block so the
signature carries only its input hypothesis. (The input keeps the `loaded`
view form here because the reused Real reciprocal refinement expects it.) -/
variable (xReg yReg : RegionName) (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)

include hN in
/-- **float divide refines float reciprocal** (`ComputeRefine.General`, erased
view): the two fp32-annotated softmax kernels, projected to their erased
algorithm kernels, perform the same per-lane writes. Proved through the Real
reciprocal refinement via the erasure bridges. -/
theorem float_softmax_reciprocal_refinement_view
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    ComputeRefine.General
      (((floatStableSoftmaxKernel xReg yReg N).eraseDType))
      (((floatSoftmaxRecipKernel xReg yReg N).eraseDType))
      (fun s0 lhs' rhs' =>
        s0 = s →
        ∀ idx : TileIndex [N],
          TensorView.observe (some lhs')
              (programTileView s yReg N) idx =
          TensorView.observe (some rhs')
              (programTileView s yReg N) idx) := by
  apply ComputeKernel.computeRefine_of_toAlgKernel
    (by simp [ComputeKernel.eraseDType]) (by simp [ComputeKernel.eraseDType])
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro idx
  have hview := float_softmax_reciprocal_refinement_exec_view xReg yReg N hN s xs h_x idx
  rw [hL, hR] at hview
  simpa using hview

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in the public theorem's transitive proof.
#axiomsClean float_softmax_reciprocal_refinement_view

-- (2) The headline is a *kernel-vs-kernel* refinement: its statement's project
-- surface must stay within the allowlist below — NO spec.
#stmtSurfaceSubset float_softmax_reciprocal_refinement_view ⊆
  [floatStableSoftmaxKernel, floatSoftmaxRecipKernel, ComputeKernel.eraseDType,
   ComputeRefine.General, TensorView.observe, TensorView.loaded, programTileView,
   TileIndex, BlockState, RegionName]

end FloatDTypeRefine.theorems

end VeriTile.Bench.Examples.FloatDTypeRefine
