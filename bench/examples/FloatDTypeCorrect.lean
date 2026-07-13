import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Float dtype erasure — the float-facing *correctness* policy

Self-contained showcase, read top to bottom: the **kernel** first (what we
built), the **supporting lemmas** in the middle (`private` plumbing — the
erasure bridge and the honest region-typing facts), the **theorem** last (one
public headline `float_add_kernel_correct_view`), then a compile-time **trust
audit**. The three real sections below are `FloatDTypeCorrect.kernels`,
`FloatDTypeCorrect.lemmas`, `FloatDTypeCorrect.theorems`.

VeriTile's float-facing theorem policy on the **correctness** surface
(`ComputeCorrect`): a kernel carries explicit `tl.float32` memory annotations,
and the algorithmic correctness theorem is discharged by **erasing** those
annotations to the Real channel and reusing the Real-valued proof (here
VectorAdd's). This is the erased/ideal pathway — the complement of the
rounding-model pathway (`ComputeRefine.*R`) demonstrated in
`bench/examples/FusedSwiglu.lean`. The refinement counterpart of this policy
lives in `bench/examples/FloatDTypeRefine.lean`.

## The public result (bottom of file)

The single public headline is **`float_add_kernel_correct_view`** — the erased
correctness statement (`ComputeCorrect.General`): the fp32-annotated add kernel,
projected through `eraseDType`, realizes `addSpec`. Its statement's project
surface is pinned by the `#stmtSurfaceSubset` gate below (a correctness surface
legitimately mentions its spec `addSpec`). The erasure bridge, the exec-level
view, and the region-typing facts (including the honest negation
`float_add_not_respects_fp32_regions`) are all `private` scaffolding.
-/

namespace VeriTile.Bench.Examples.FloatDTypeCorrect

open VeriTile.Triton
open VeriTile.Examples  -- Common vocabulary: programTileView, InputLoadedAt, observeAt, …

/-- Local reduction helper: the algorithm-projection traversals
(`ComputeStmt.toAlgorithm?` and friends) are written in `do` notation over
`Except`, so their equation lemmas leave `Except.ok _ >>= f` redexes behind.
This discharges them for `simp`. -/
@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

/-! ## Kernel -/
section FloatDTypeCorrect.kernels

/-- This file's own copy of the aligned Real `addKernel` (each showcase is
self-contained; the showcased original lives in `bench/examples/VectorAdd.lean`).
Elementwise add of two `blockSize`-element tiles, single-block, no boundary
mask. The fp32-annotated `floatAddKernel` below erases to this kernel. -/
def addKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  y    := tl.load($(yReg) + offs)
  out  := x + y
  tl.store($(outReg) + offs, out)
}

/-- This file's own copy of the elementwise-add math denotation:
`out[i] = xs[i] + ys[i]`. -/
def addSpec {N : Nat} (xs ys : Fin N → ℝ) (i : Fin N) : ℝ :=
  xs i + ys i

/-- Elementwise add with fp32-annotated input/output memory.

The surface kernel carries Triton-like dtype information, while the proof path
below erases it back to the mathematical Real kernel. -/
def floatAddKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs, dtype=tl.float32)
  y    := tl.load($(yReg) + offs, dtype=tl.float32)
  out  := x + y
  tl.store($(outReg) + offs, out)
}

end FloatDTypeCorrect.kernels

/-! ## Supporting lemmas (private plumbing) -/
section FloatDTypeCorrect.lemmas

/-- This file's own copy of the aligned Real VectorAdd correctness theorem
(each showcase is self-contained; the showcased original lives in
`bench/examples/VectorAdd.lean`). -/
theorem add_kernel_correct
    (xReg yReg outReg : RegionName)
    (blockSize : Nat) (_hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs)
    (_h_y : InputLoadedAt s yReg blockSize ys) :
    ∀ i : Fin blockSize,
      observeAt (exec (addKernel xReg yReg outReg blockSize) s) outReg blockSize s.pid i
        = some (addSpec xs ys i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, addKernel, stepStmts, stepStmt, Tile.bop,
        NumericDType.add, NumericDType.mul, addSpec]
  unfold InputLoadedAt at _h_x _h_y
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x, _h_y]

/-- This file's own copy of the view-level surface for `add_kernel_correct`. -/
theorem add_kernel_correct_exec_view
    (xReg yReg outReg : RegionName)
    (blockSize : Nat) (hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : TensorView.loadedArray s (programTileView s xReg blockSize) xs)
    (h_y : TensorView.loadedArray s (programTileView s yReg blockSize) ys) :
    ∀ idx : TileIndex [blockSize],
      TensorView.observe (exec (addKernel xReg yReg outReg blockSize) s)
          (programTileView s outReg blockSize) idx
        = some (addSpec xs ys idx.1) := by
  intro idx
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := blockSize) (xs := xs) h_x
  have hy := inputLoadedAt_of_programTileView_loaded (s := s) (region := yReg)
    (N := blockSize) (xs := ys) h_y
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt]
    using add_kernel_correct xReg yReg outReg blockSize hBlockSize s xs ys hx hy idx.1

/-- The fp32-annotated kernel erases to the existing Real VectorAdd kernel,
at the algorithm-projection level: both sides project (via `toAlgKernel`) to
the same plain algorithm kernel. -/
private theorem float_add_erases_to_real
    (xReg yReg outReg : RegionName) (blockSize : Nat) :
    (floatAddKernel xReg yReg outReg blockSize).eraseDType.toAlgKernel =
      (addKernel xReg yReg outReg blockSize).toAlgKernel := by
  simp [floatAddKernel, addKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType]

/-- Float-facing VectorAdd correctness, exec-level view.

The statement starts from the fp32-annotated kernel, but the formal algorithmic
proof runs the erased Real semantics: state float, prove Real. -/
private theorem float_add_kernel_correct_exec_view
    (xReg yReg outReg : RegionName)
    (blockSize : Nat) (hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1))
    (h_y : TensorView.loaded s (programTileView s yReg blockSize)
      (fun idx : TileIndex [blockSize] => ys idx.1)) :
    ∀ idx : TileIndex [blockSize],
      TensorView.observe
          (exec (floatAddKernel xReg yReg outReg blockSize).eraseDType s)
          (programTileView s outReg blockSize) idx
        = some (addSpec xs ys idx.1) := by
  intro idx
  rw [float_add_erases_to_real xReg yReg outReg blockSize]
  exact add_kernel_correct_exec_view xReg yReg outReg blockSize hBlockSize
    s xs ys h_x h_y idx

/-- The original claim `Kernel.RespectsRegionTyping (fun _ => .fp32)
(floatAddKernel …)` is **false** under today's definitions and is therefore
recorded as a proved negation. `Kernel.RespectsRegionTyping` receives the
kernel through the `ComputeKernel → AlgKernel` coercion (`toAlgKernel`), and
the compute→algorithm projection (`ComputeOp.toAlgorithm?`) rewrites every
`tl.float32` load to `Op.load ComputeDType.fp32.eraseDType = Op.load .real`
before the typing contract ever sees it. The contract's load/store cases then
demand `Γ region = .real`, i.e. `TileDType.fp32 = TileDType.real`, which
reduces to `False`. -/
private theorem float_add_not_respects_fp32_regions
    (xReg yReg outReg : RegionName) (blockSize : Nat) :
    ¬ Kernel.RespectsRegionTyping (fun _ => .fp32)
      (floatAddKernel xReg yReg outReg blockSize) := by
  simp [floatAddKernel, ComputeKernel.toAlgKernel,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.RespectsRegionTyping, StmtList.RespectsRegionTyping,
    Stmt.RespectsRegionTyping, Op.RespectsRegionTyping,
    MemAccess.RespectsRegionTyping, MaskOpt.RespectsRegionTyping]

/-- The true region-typing contract on today's projected surface: after the
compute→algorithm projection the fp32 annotations are erased, so the kernel
respects the all-`.real` region typing. -/
private theorem float_add_respects_real_regions
    (xReg yReg outReg : RegionName) (blockSize : Nat) :
    Kernel.RespectsRegionTyping (fun _ => .real)
      (floatAddKernel xReg yReg outReg blockSize) := by
  simp [floatAddKernel, ComputeKernel.toAlgKernel,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.RespectsRegionTyping, StmtList.RespectsRegionTyping,
    Stmt.RespectsRegionTyping, Op.RespectsRegionTyping,
    MemAccess.RespectsRegionTyping, MaskOpt.RespectsRegionTyping]

end FloatDTypeCorrect.lemmas

/-! ## The headline theorem -/
section FloatDTypeCorrect.theorems

/- Shared parameters of the headline, hoisted to a `variable` block so the
signature carries only its input hypotheses. (The inputs keep the `loaded`
view form here because the reused Real VectorAdd theorem expects it.) -/
variable (xReg yReg outReg : RegionName)
variable (blockSize : Nat) (hBlockSize : 0 < blockSize)
variable (s : BlockState) (xs ys : Fin blockSize → ℝ)

include hBlockSize in
/-- **float add realizes addSpec** (`ComputeCorrect.General`, erased view):
the fp32-annotated add kernel, projected to the erased algorithm kernel,
computes `addSpec` at every lane. Proved through the Real VectorAdd theorem via
the erasure bridge. -/
specification float_add_kernel_correct_view
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1))
    (h_y : TensorView.loaded s (programTileView s yReg blockSize)
      (fun idx : TileIndex [blockSize] => ys idx.1)) :
    ComputeCorrect.General
      (((floatAddKernel xReg yReg outReg blockSize).eraseDType))
      (fun s0 s' =>
        s0 = s →
        ∀ idx : TileIndex [blockSize],
          TensorView.observe (some s')
              (programTileView s outReg blockSize) idx
            = some (addSpec xs ys idx.1)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
    (by simp [ComputeKernel.eraseDType])
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have hview := float_add_kernel_correct_exec_view xReg yReg outReg blockSize hBlockSize
    s xs ys h_x h_y idx
  rw [hExec] at hview
  simpa using hview

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in the public theorem's transitive proof.
#axiomsClean float_add_kernel_correct_view

-- (2) The headline is a *correctness* view: its statement's project surface must
-- stay within the allowlist below. A correctness surface legitimately names its
-- spec `addSpec`; the gate pins that this is the only spec-like constant.
#stmtSurfaceSubset float_add_kernel_correct_view ⊆
  [floatAddKernel, addSpec, ComputeKernel.eraseDType, ComputeCorrect.General,
   TensorView.observe, TensorView.loaded, programTileView, TileIndex,
   BlockState, RegionName]

end FloatDTypeCorrect.theorems

end VeriTile.Bench.Examples.FloatDTypeCorrect
