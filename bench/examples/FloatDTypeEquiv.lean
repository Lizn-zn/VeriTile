import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Float dtype erasure — the float-facing *refinement* policy, on the `≡[R]` surface

Self-contained showcase, read top to bottom: the **kernels** first (two
fp32-annotated softmaxes and the two plain Real kernels they erase to), the
**erasure bridges** in the middle (`private` plumbing), the region-level
refinement core next (`softmax_reciprocal_refinement_view`), the
**region-run plumbing** (termination + per-kernel frames), then the
**flat-memory bridge side conditions**, the **specification** last (one
public headline `float_softmax_reciprocal_equiv` on the `≡[R]` surface), and
a compile-time **trust audit**. The sections are `FloatDTypeEquiv.kernels`,
`FloatDTypeEquiv.lemmas`, `FloatDTypeEquiv.theorems`,
`FloatDTypeEquiv.regionRun`, `FloatDTypeEquiv.bridge`,
`FloatDTypeEquiv.spec`.

VeriTile's float-facing theorem policy on the refinement surface: the two
softmax kernels carry explicit float memory annotations (`.to(tl.float64)`
load casts, `.to(tl.float32)` store casts), and the equivalence headline is
stated about their **erasures** to the Real channel — the `kernel` fields of
the IO signatures below are literally
`(floatStableSoftmaxKernel …).eraseDType` /
`(floatSoftmaxRecipKernel …).eraseDType`, so the erased/ideal pathway is
visible in the audited statement surface itself. The erasure bridges prove
those erasures are exactly the plain Real divide/reciprocal kernels
(`stableSoftmaxKernel` / `softmaxRecipKernel` — raw stores, no casts), and
every `≡[R]` obligation is transported along the bridges to the plain
kernels.

## The public result (bottom of file)

The single public headline is **`float_softmax_reciprocal_equiv`** — kernel
equivalence on the shared one-input IO signature:

    floatSoftmaxDivIO B ≡[R] floatSoftmaxRecipIO B

`≡[R]` is the audit-once kernel-equivalence combinator (`KernelIO₁.Equiv`,
`VeriTile.Triton.Memory.KernelSpec`), the `⊨`-grade form of the refinement
surface. Spelled out, the headline says: for **every** disjoint flat
placement of the interface buffers `x`/`y` (∀ base pointers, ∀ buffer sizes;
neither kernel stages anything through memory, so both `scratch` lists are
empty), **every** program id whose whole-tile windows `[pid·B, pid·B + B)`
are in bounds, and **every** launch state — **no input hypotheses at all**:
"equal inputs" is simply "the same `s₀`" — both translated (erased) pointer
kernels terminate under `execR R`, their `y` windows agree cell-for-cell,
and each kernel leaves every cell outside its `y` window untouched. The only
side hypothesis is `0 < B`: `tl.max` over an empty tile has no value, so an
empty block genuinely diverges.

The headline quantifies over **every** rounding model `R` — but the erased
kernels are cast-free, so `execR R` never rounds. The ∀`R` quantification
keeps the surface uniform with the other seven equivalence showcases while
the content is the exact-ℝ rewrite `e / S = e · S⁻¹`. The correctness
counterpart of the erasure policy is `bench/examples/FloatDTypeCorrect.lean`;
the rounding-model (bf16) sibling of this same rewrite is
`bench/examples/SoftmaxReciprocalEquiv.lean`.

The statement mentions only the two IO signatures and the library
equivalence surface — **no spec** (the `#stmtSurfaceSubset` gate below
enforces this). Everything else — the erasure bridges, the region-level
refinement `softmax_reciprocal_refinement_view` (`ComputeRefine.Refines R`
on the plain kernels), the per-kernel frame lemmas, and the bridge side
conditions — is scaffolding.
-/

namespace VeriTile.Bench.Examples.FloatDTypeEquiv

open VeriTile.Triton
open scoped VeriTile.Triton.KernelIO₁

/-- Local reduction helper: the algorithm-projection traversals
(`ComputeStmt.toAlgorithm?` and friends) are written in `do` notation over
`Except`, so their equation lemmas leave `Except.ok _ >>= f` redexes behind.
This discharges them for `simp`. -/
@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

/-! ## Kernels -/
section FloatDTypeEquiv.kernels

/-- This file's own copy of the aligned Real per-element-divide stable softmax
(each showcase is self-contained; the bf16 sibling is showcased in
`bench/examples/SoftmaxStableEquiv.lean` — its raw-store twin was retired
there, the exact surface being the `≡[R]` headline at `R := .triv`). The
fp32-annotated `floatStableSoftmaxKernel` below erases to this kernel. -/
def stableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, y)
}

/-- This file's own copy of the aligned Real reciprocal-form stable softmax
(the bf16 sibling is showcased in `bench/examples/SoftmaxReciprocalEquiv.lean`;
its raw-store twin was retired there). The fp32-annotated
`floatSoftmaxRecipKernel` below erases to this kernel. -/
def softmaxRecipKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x      := tl.load($(xReg) + offs)
  m      := tl.max(x, axis=0)
  e      := tl.exp(x - m)
  s      := tl.sum(e, axis=0)
  inv_s  := 1 / s
  y      := e * inv_s
  tl.store($(yReg) + offs, y)
}

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

end FloatDTypeEquiv.kernels

/-! ## Supporting lemmas (private plumbing) -/
section FloatDTypeEquiv.lemmas

/-- The fp32 per-element-divide softmax erases to the plain Real kernel,
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

/-- The fp32 reciprocal-form softmax erases to the plain Real optimized
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

/-- A scatter-store `foldl` leaves every memory cell it does not hit
unchanged (cell-level frame for the plain kernels' raw unmasked stores;
this file's copy of VectorAdd's helper — the exact-store sibling of the
library `BlockState.foldl_writeMemAsR_preserve_cell`). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ)
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        acc.writeMem region (offsetFn k) (valueFn k)) s).mem r o
      = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons,
        ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
        BlockState.writeMem_mem]
      exact if_neg (fun hc =>
        hnot hd List.mem_cons_self ⟨hc.1.symm, hc.2.symm⟩)

end FloatDTypeEquiv.lemmas

/-! ## The region-level refinement core -/
section FloatDTypeEquiv.theorems

/- Shared parameters of the core: the two regions, the block size
(nonempty), the state, and the rounding model `R`. -/
variable (xReg yReg : RegionName) (N : Nat) (hN : 0 < N) (s : BlockState)
variable (R : RoundingModel)

include hN in
/-- **divide refines reciprocal** (`ComputeRefine.Refines R`, no scratch, on
the plain Real kernels): from the same initial state the per-element-divide
and precomputed-reciprocal kernels perform the same writes. Both compute the
same per-lane ℝ value (`e / S = e · S⁻¹`) and store it raw — no rounding
site, so `R` is inert. This is the mathematical core the `≡[R]`
specification's region-model obligation repackages, reached from the
fp32-annotated kernels through the erasure bridges. -/
theorem softmax_reciprocal_refinement_view :
    ComputeRefine.Refines R
      (stableSoftmaxKernel xReg yReg N)
      (softmaxRecipKernel xReg yReg N) s [] := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  apply ComputeKernel.computeRefineR_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp [execR, stableSoftmaxKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComputeExpr.toAlgorithm?,
        BlockState.writeMemTypedR, BlockState.writeMemTyped] at hL
  simp [Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hL
  simp [execR, softmaxRecipKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComputeExpr.toAlgorithm?,
        BlockState.writeMemTypedR, BlockState.writeMemTyped] at hR
  simp [Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hR
  subst lhs'
  subst rhs'
  refine BlockState.foldl_writeMem_mem_congr _ _ _ _ ?_ r o _ _ rfl
  intro k _
  -- per-lane value equality `e / S = e * S⁻¹`, stored raw
  exact div_eq_mul_inv _ _

end FloatDTypeEquiv.theorems

/-! ## Region-run plumbing (termination + per-kernel frames)

`≡[R]`'s region-model obligation starts from **any** state — no loaded
inputs, no clean undef — and additionally wants, per kernel, termination and
a frame ("only the program's `y` window is written"). All four lemmas fall
out of the same computational unfold the refinement core uses: `execR R`
reduces each straight-line body to one plain `writeMem` scatter over the
program's `offs` tile (`0 < N` because `tl.max` over an empty tile has no
value); the frames then close by the local scatter frame
`foldl_store_preserve_cell`. -/
section FloatDTypeEquiv.regionRun

variable (xReg yReg : RegionName) (N : Nat) (s : BlockState)
variable (R : RoundingModel)

/-- The plain divide kernel terminates under `execR R` from every state. -/
private theorem softmaxDiv_execR_isSome (hN : 0 < N) :
    ∃ s', execR R ((stableSoftmaxKernel xReg yReg N).toAlgKernel) s
      = some s' := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  simp [execR, stableSoftmaxKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComputeExpr.toAlgorithm?,
        Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]

/-- The plain reciprocal kernel terminates under `execR R` from every state. -/
private theorem softmaxRecip_execR_isSome (hN : 0 < N) :
    ∃ s', execR R ((softmaxRecipKernel xReg yReg N).toAlgKernel) s
      = some s' := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  simp [execR, softmaxRecipKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComputeExpr.toAlgorithm?,
        Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]

/-- Frame lemma for the plain divide kernel: every cell outside the program's
`y` window `[pid·N, pid·N + N)` is untouched. -/
private theorem softmaxDiv_preservesR (hN : 0 < N) (s' : BlockState)
    (hExec : execR R ((stableSoftmaxKernel xReg yReg N).toAlgKernel) s
      = some s')
    (r : RegionName) (o : Nat)
    (hcond : r ≠ yReg ∨ ∀ j : Fin N, o ≠ s.pid * N + j.val) :
    s'.mem r o = s.mem r o := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  simp only [BlockState.pid_eq] at hcond
  simp [execR, stableSoftmaxKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComputeExpr.toAlgorithm?,
        BlockState.writeMemTypedR, BlockState.writeMemTyped] at hExec
  simp [Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hExec
  subst s'
  refine Eq.trans (foldl_store_preserve_cell _ _ r o _ _ ?_) rfl
  intro k _ hc
  rcases hcond with hne | hno
  · exact hne hc.1.symm
  · exact hno k.1 hc.2.symm

/-- Frame lemma for the plain reciprocal kernel: every cell outside the
program's `y` window is untouched. -/
private theorem softmaxRecip_preservesR (hN : 0 < N) (s' : BlockState)
    (hExec : execR R ((softmaxRecipKernel xReg yReg N).toAlgKernel) s
      = some s')
    (r : RegionName) (o : Nat)
    (hcond : r ≠ yReg ∨ ∀ j : Fin N, o ≠ s.pid * N + j.val) :
    s'.mem r o = s.mem r o := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  simp only [BlockState.pid_eq] at hcond
  simp [execR, softmaxRecipKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComputeExpr.toAlgorithm?,
        BlockState.writeMemTypedR, BlockState.writeMemTyped] at hExec
  simp [Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hExec
  subst s'
  refine Eq.trans (foldl_store_preserve_cell _ _ r o _ _ ?_) rfl
  intro k _ hc
  rcases hcond with hne | hno
  · exact hne hc.1.symm
  · exact hno k.1 hc.2.symm

end FloatDTypeEquiv.regionRun

/-! ## Flat-memory bridge side conditions

Both plain kernels are register-indirect (`offs := pid * B + tl.arange(0, B);
tl.load(x + offs)`), so no ∀-state safety contract covers them; the
flat-memory bridge (v1.2, `execR` flavor) takes the per-execution
`Kernel.TraceSafeR` contract instead, plus `FlattenOk` (bridge fragment
membership). Each kernel makes exactly two accesses — the `x` load and the
raw `y` store, both through the unmasked whole-tile `offs` window — so each
buffer's bound must cover the program tile `[pid·B, pid·B + B)`; every other
statement (`program_id`, the `offs` index arithmetic, and the register ops
`max`/`exp`/`sum`/`div`/reciprocal) is memory-silent. The headline
discharges these for the plain kernels and transports them to the erased
kernels along the erasure bridges. -/
section FloatDTypeEquiv.bridge

variable (xReg yReg : RegionName) (N : Nat) (s : BlockState)
variable (R : RoundingModel)

/-- The plain divide kernel sits inside the bridge's covered fragment. -/
theorem softmaxDiv_flattenOk :
    ((stableSoftmaxKernel xReg yReg N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [stableSoftmaxKernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- The plain reciprocal kernel sits inside the bridge's covered fragment. -/
theorem softmaxRecip_flattenOk :
    ((softmaxRecipKernel xReg yReg N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [softmaxRecipKernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk for the plain divide kernel under `R`: one
computational unfold walks all eight statements, discharging every
memory-silent `SafeAtR` and reducing the two unmasked accesses' address
obligations to the whole-tile bounds hypotheses. -/
theorem softmaxDiv_traceSafeR (hN : 0 < N) (bounds : RegionBounds)
    (hx : s.pid * N + N ≤ bounds xReg)
    (hy : s.pid * N + N ≤ bounds yReg) :
    Kernel.TraceSafeR R bounds
      ((stableSoftmaxKernel xReg yReg N).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  unfold Kernel.TraceSafeR
  simp only [BlockState.pid_eq] at hx hy
  simp [stableSoftmaxKernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    MaskOpt.SafeAtR, MemAccess.SafeAtR, stepStmtR, evalOpR.eq_def,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MaskOpt.ActiveR, BlockState.setReg,
    Tile.bop, Tile.uop,
    Tile.reduceSumDrop, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div]
  exact ⟨fun i => lt_of_lt_of_le (Nat.add_lt_add_left i.isLt _) hx,
    fun i => lt_of_lt_of_le (Nat.add_lt_add_left i.isLt _) hy⟩

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk for the plain reciprocal kernel under `R`: same
computational unfold over its nine statements; the extra reciprocal
statement is memory-silent. -/
theorem softmaxRecip_traceSafeR (hN : 0 < N) (bounds : RegionBounds)
    (hx : s.pid * N + N ≤ bounds xReg)
    (hy : s.pid * N + N ≤ bounds yReg) :
    Kernel.TraceSafeR R bounds
      ((softmaxRecipKernel xReg yReg N).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  unfold Kernel.TraceSafeR
  simp only [BlockState.pid_eq] at hx hy
  simp [softmaxRecipKernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    MaskOpt.SafeAtR, MemAccess.SafeAtR, stepStmtR, evalOpR.eq_def,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MaskOpt.ActiveR, BlockState.setReg,
    Tile.bop, Tile.uop,
    Tile.reduceSumDrop, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div]
  exact ⟨fun i => lt_of_lt_of_le (Nat.add_lt_add_left i.isLt _) hx,
    fun i => lt_of_lt_of_le (Nat.add_lt_add_left i.isLt _) hy⟩

end FloatDTypeEquiv.bridge

/-! ## The spec: `floatSoftmaxDivIO ≡[R] floatSoftmaxRecipIO` -/
section FloatDTypeEquiv.spec

/-- The erased fp32 divide kernel's **IO signature** — the whole
kernel-specific audit surface of the headline: `kernel` is
**`(floatStableSoftmaxKernel …).eraseDType`** (the dtype-erasure policy is
part of the audited signature, not hidden in the proof), `x` is the input
buffer and `y` the output, the tile lengths are `Bin = Bout = B` (softmax is
a whole-tile map), program `pid` reads/writes its tile at `pid * B` (the
kernel's `offs = pid * B + tl.arange(0, B)` window), and no scratch (nothing
is staged through memory). The windows are declared, not parsed from the
kernel: the headline **proves** the kernel's actual addressing matches
them. -/
def floatSoftmaxDivIO (B : Nat) : KernelIO₁ where
  kernel := (floatStableSoftmaxKernel ⟨"x"⟩ ⟨"y"⟩ B).eraseDType
  inp := ⟨"x"⟩
  out := ⟨"y"⟩
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

/-- The erased fp32 reciprocal kernel's IO signature: the **same interface
by construction** (structure update of `floatSoftmaxDivIO` — buffers and
windows shared verbatim), with the erased reciprocal-multiply kernel
plugged in. -/
def floatSoftmaxRecipIO (B : Nat) : KernelIO₁ :=
  { floatSoftmaxDivIO B with
    kernel := (floatSoftmaxRecipKernel ⟨"x"⟩ ⟨"y"⟩ B).eraseDType }

/-- **The headline**: the fp32-annotated per-element-divide softmax and the
fp32-annotated precomputed-reciprocal softmax, projected through
`eraseDType`, are equivalent on their shared IO signature, for **every**
rounding model — see the module docstring for the full contract `≡[R]`
unfolds to (both erased kernels terminate from any state with a nonempty
tile, `y` windows agree, each frames outside its `y` window; no scratch on
either side). Proof: `KernelIO₁.Equiv.intro` assembles the region-model
equivalence, with each of the five obligations first rewritten along the
erasure bridges (`float_*_erases_to_real`) to the plain Real kernels —
termination and the frames from the computational unfold, the `y`-agreement
leg from the region-level refinement core
`softmax_reciprocal_refinement_view` (`e / S = e · S⁻¹`, stored raw),
unpacked at the execution pair by the library `ComputeRefine.Refines.out` —
plus the flat-memory bridge side conditions (`FlattenOk` + `TraceSafeR` per
kernel), also transported along the bridges. -/
specification float_softmax_reciprocal_equiv (R : RoundingModel) (B : Nat)
    (hB : 0 < B) :
    floatSoftmaxDivIO B ≡[R] floatSoftmaxRecipIO B := by
  have hbridge1 := float_stable_softmax_erases_to_real ⟨"x"⟩ ⟨"y"⟩ B
  have hbridge2 := float_softmax_recip_erases_to_real ⟨"x"⟩ ⟨"y"⟩ B
  refine KernelIO₁.Equiv.intro _ _ ?_ ?_ ?_ ?_ ?_
  · -- FlattenOk, divide (transported along the erasure bridge)
    show ((floatStableSoftmaxKernel ⟨"x"⟩ ⟨"y"⟩ B).eraseDType).toAlgKernel.FlattenOk
    rw [hbridge1]
    exact softmaxDiv_flattenOk ⟨"x"⟩ ⟨"y"⟩ B
  · -- FlattenOk, reciprocal
    show ((floatSoftmaxRecipKernel ⟨"x"⟩ ⟨"y"⟩ B).eraseDType).toAlgKernel.FlattenOk
    rw [hbridge2]
    exact softmaxRecip_flattenOk ⟨"x"⟩ ⟨"y"⟩ B
  · -- TraceSafeR, divide
    intro bounds t h1 h2 _
    simp only [floatSoftmaxDivIO] at h1 h2 ⊢
    rw [hbridge1]
    exact softmaxDiv_traceSafeR ⟨"x"⟩ ⟨"y"⟩ B t R hB bounds h1 h2
  · -- TraceSafeR, reciprocal (window hypotheses are io₁'s = the shared ones)
    intro bounds t h1 h2 _
    simp only [floatSoftmaxDivIO, floatSoftmaxRecipIO] at h1 h2 ⊢
    rw [hbridge2]
    exact softmaxRecip_traceSafeR ⟨"x"⟩ ⟨"y"⟩ B t R hB bounds h1 h2
  · -- the region-model equivalence, from ANY state s₀ (no input hypotheses)
    intro s₀
    simp only [floatSoftmaxDivIO, floatSoftmaxRecipIO]
    rw [hbridge1, hbridge2]
    obtain ⟨s1, hexec1⟩ := softmaxDiv_execR_isSome ⟨"x"⟩ ⟨"y"⟩ B s₀ R hB
    obtain ⟨s2, hexec2⟩ := softmaxRecip_execR_isSome ⟨"x"⟩ ⟨"y"⟩ B s₀ R hB
    -- the refinement core: the two final memories agree at EVERY cell
    have hmem12 : ∀ r, r ∉ ([] : List RegionName) →
        ∀ o, s1.mem r o = s2.mem r o :=
      (softmax_reciprocal_refinement_view ⟨"x"⟩ ⟨"y"⟩ B hB s₀ R).out
        rfl rfl hexec1 hexec2
    refine ⟨s1, s2, hexec1, hexec2, ?_, ?_, ?_⟩
    · -- the y windows agree (a fortiori: all cells agree)
      intro j
      have hmem := hmem12 ⟨"y"⟩ (by simp) (s₀.pid * B + j.val)
      unfold BlockState.readMem
      rw [hmem]
    · -- divide frame: writes only its y window (no scratch)
      intro r o hcond _
      exact softmaxDiv_preservesR ⟨"x"⟩ ⟨"y"⟩ B s₀ R hB s1 hexec1 r o hcond
    · -- reciprocal frame: writes only its y window (no scratch)
      intro r o hcond _
      exact softmaxRecip_preservesR ⟨"x"⟩ ⟨"y"⟩ B s₀ R hB s2 hexec2 r o hcond

end FloatDTypeEquiv.spec

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom — in the region-level core's and the
-- public headline's transitive proofs.
#axiomsClean softmax_reciprocal_refinement_view
#axiomsClean float_softmax_reciprocal_equiv

-- (2) The headline's statement surface is the two IO signatures plus the
-- audit-once kernel-equivalence combinator — NO spec (this is a refinement
-- surface), no other project constant. The dtype-erasure policy
-- (`eraseDType`) lives inside the IO signatures' definitions, one
-- definition-click away from the headline.
#stmtSurfaceSubset float_softmax_reciprocal_equiv ⊆
  [floatSoftmaxDivIO, floatSoftmaxRecipIO, VeriTile.Triton.KernelIO₁.Equiv,
   RoundingModel]

end VeriTile.Bench.Examples.FloatDTypeEquiv
