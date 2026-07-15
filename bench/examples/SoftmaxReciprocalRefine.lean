import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Softmax: per-element-divide vs precomputed-reciprocal — kernel equivalence `≡[R]`

Self-contained showcase, read top to bottom: **kernels** first, one
**supporting lemma** in the middle (`private` plumbing — a bf16-scatter
congruence; its cell-frame twin is the library's
`BlockState.foldl_writeMemAsR_preserve_cell`), the region-level refinement
core next (`softmax_reciprocal_refinement_view`), the **region-run
plumbing** (termination + per-kernel frames), then the **flat-memory bridge
side conditions**, the **specification** last (one public headline
`softmax_reciprocal_equiv` on the `≡[R]` surface), and a compile-time
**trust audit**. The rounding-layer sections are `SoftmaxReciprocal.kernels`,
`SoftmaxReciprocal.lemmas`, `SoftmaxReciprocal.theorems`,
`SoftmaxReciprocal.regionRun`, `SoftmaxReciprocal.bridge`,
`SoftmaxReciprocal.spec`; the exact-ℝ companions close the file.

Both stable-softmax kernels compute `y = exp(x − m) / Σ exp(x − m)` per row and
**store the result rounded to bf16** (`(y).to(tl.bfloat16)`):
`stableSoftmaxKernel` divides each lane by the sum; `softmaxRecipKernel`
precomputes `1 / Σ` once and multiplies. The reductions run in ℝ, so both
produce the **same** per-lane ℝ value (`e / S = e · S⁻¹`), and the only rounding
is the shared bf16 output store, which quantizes equal values identically — no
idempotence argument is even needed (contrast `bench/examples/FusedSwiglu.lean`,
where the two sides' rounding sites differ by a redundant re-round).

## The public result (bottom of the rounding layer)

The single public headline is **`softmax_reciprocal_equiv`** — kernel
equivalence on the shared one-input IO signature:

    softmaxDivIO B ≡[R] softmaxRecipIO B

`≡[R]` is the audit-once kernel-equivalence combinator (`KernelIO₁.Equiv`,
`VeriTile.Triton.Memory.KernelSpec`), the `⊨`-grade form of the refinement
surface. Spelled out, the headline says: for **every** disjoint flat placement
of the interface buffers `x`/`y` (∀ base pointers, ∀ buffer sizes; neither
kernel stages anything through memory, so both `scratch` lists are empty),
**every** program id whose whole-tile windows `[pid·B, pid·B + B)` are in
bounds, and **every** launch state — **no input hypotheses at all**, not even
loaded inputs: "equal inputs" is simply "the same `s₀`" — both translated
pointer kernels terminate under `execR R`, their `y` windows agree
cell-for-cell, and each kernel leaves every cell outside its `y` window
untouched. The only side hypothesis is `0 < B`: `tl.max` over an empty tile
has no value, so an empty block genuinely diverges.

The statement mentions only the two IO signatures and the library equivalence
surface — **no spec** (the `#stmtSurfaceSubset` gate below enforces this).
Everything else — the region-level refinement `softmax_reciprocal_refinement_view`
(`ComputeRefine.Refines R`, formerly the headline, now demoted to the
mathematical core the `≡[R]` obligation repackages), the per-kernel frame
lemmas, and the bridge side conditions — is scaffolding.

## Exact-ℝ companions (bottom of file)

The file also carries the exact-ℝ (raw-store, no bf16 rounding) variants
`stableSoftmaxKernelReal` / `softmaxRecipKernelReal` together with their
ARM-in-Lean-style story: the reciprocal-side closed form
(`softmax_recip_correct` against `stableRecipSpec`), the load-bearing math
identity (`div_eq_mul_inv_real`), the pointwise `Y`-memory refinement
(`softmax_reciprocal_refinement`), and its `TensorView` surface
(`softmax_reciprocal_refinement_exec_view`).
-/

namespace VeriTile.Bench.Examples.SoftmaxReciprocal

open VeriTile.Triton VeriTile.Triton.TiledSoftmax
open scoped VeriTile.Triton.KernelIO₁

/-! ## Kernels -/
section SoftmaxReciprocal.kernels

/-- Numerically-stable softmax with a per-lane division `y = e / S`, stored bf16. -/
def stableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

/-- Optimized stable softmax: precompute `1 / S` once, then multiply per lane
(`y = e · S⁻¹`), stored bf16. -/
def softmaxRecipKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x      := tl.load($(xReg) + offs)
  m      := tl.max(x, axis=0)
  e      := tl.exp(x - m)
  s      := tl.sum(e, axis=0)
  inv_s  := 1 / s
  y      := e * inv_s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

/-! ### Exact-ℝ variants (raw store, no rounding)

Term-identical to the bf16 kernels above except that the final store writes the
raw ℝ value `y` instead of `(y).to(tl.bfloat16)`. These are the kernels of the
exact-ℝ companion theorems at the bottom of the file. The divide-side kernel
mirrors `stableSoftmaxKernelReal` in `bench/examples/SoftmaxEqRefine.lean`
(bench showcases are self-contained, so this file carries its own copy). -/

/-- Stable softmax, exact-ℝ store: `y = e / S` with per-lane division. -/
def stableSoftmaxKernelReal (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, y)
}

/-- Stable softmax with precomputed reciprocal, exact-ℝ store. Saves N-1
    divisions vs the per-element-divide form (`stableSoftmaxKernelReal`). -/
def softmaxRecipKernelReal (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
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

end SoftmaxReciprocal.kernels

/-! ## Supporting lemma (private plumbing) -/
section SoftmaxReciprocal.lemmas

/-- Two `writeMemAsR` scatters over the same offsets agree cell-by-cell when
their per-lane values agree — the rounding-store analogue of
`BlockState.foldl_writeMem_mem_congr`. -/
private theorem foldl_writeMemAsR_mem_congr {α : Type} (R : RoundingModel)
    (dtype : FloatDType) {region : RegionName} (l : List α) (offsetFn : α → Nat)
    (vL vR : α → TileCarrier dtype.toTileDType)
    (hv : ∀ k ∈ l, vL k = vR k) (r : RegionName) (o : Nat) :
    ∀ sL sR : BlockState, sL.mem r o = sR.mem r o →
      (l.foldl (fun acc k => acc.writeMemAsR R dtype region (offsetFn k) (vL k)) sL).mem r o
        = (l.foldl (fun acc k => acc.writeMemAsR R dtype region (offsetFn k) (vR k)) sR).mem r o := by
  induction l with
  | nil => intro sL sR h; exact h
  | cons hd tl ih =>
      intro sL sR h
      refine ih (fun k hk => hv k (List.mem_cons_of_mem _ hk)) _ _ ?_
      rw [BlockState.writeMemAsR_mem, BlockState.writeMemAsR_mem,
          hv hd List.mem_cons_self]
      by_cases hc : r = region ∧ o = offsetFn hd
      · rw [if_pos hc, if_pos hc]
      · rw [if_neg hc, if_neg hc]; exact h

end SoftmaxReciprocal.lemmas

/-! ## The region-level refinement core -/
section SoftmaxReciprocal.theorems

/- Shared parameters of the core: the two regions, the block size
(nonempty), the state, and the rounding model `R`. -/
variable (xReg yReg : RegionName) (N : Nat) (hN : 0 < N) (s : BlockState)
variable (R : RoundingModel)

include hN in
/-- **divide refines reciprocal** (`ComputeRefine.Refines R`, no scratch): for
the rounding model `R`, from the same initial state the per-element-divide and
precomputed-reciprocal kernels perform the same writes. Both compute the same
per-lane ℝ value (`e / S = e · S⁻¹`) and round it at the same bf16 store.

Formerly the file's headline; now the mathematical core the `≡[R]`
specification's region-model obligation repackages (its `y`-window agreement
leg is exactly this whole-memory agreement read at the output window). -/
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
        NumericDType.div, ComputeExpr.toAlgorithm?] at hL
  simp [Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hL
  simp [execR, softmaxRecipKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComputeExpr.toAlgorithm?] at hR
  simp [Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hR
  subst lhs'
  subst rhs'
  refine foldl_writeMemAsR_mem_congr R .bf16 _ _ _ _ ?_ r o _ _ rfl
  intro k _
  -- per-lane value equality `e / S = e * S⁻¹`, rounded identically at the store
  exact congrArg (fun v : ℝ => R.cast FloatDType.real FloatDType.bf16 (some v))
    (div_eq_mul_inv _ _)

end SoftmaxReciprocal.theorems

/-! ## Region-run plumbing (termination + per-kernel frames)

`≡[R]`'s region-model obligation starts from **any** state — no loaded
inputs, no clean undef — and additionally wants, per kernel, termination and
a frame ("only the program's `y` window is written"). All four lemmas fall
out of the same computational unfold the refinement core uses: `execR R`
reduces each straight-line body to one `writeMemAsR` scatter over the
program's `offs` tile (`0 < N` because `tl.max` over an empty tile has no
value); the frames then close by the library scatter frame
`BlockState.foldl_writeMemAsR_preserve_cell`. -/
section SoftmaxReciprocal.regionRun

variable (xReg yReg : RegionName) (N : Nat) (s : BlockState)
variable (R : RoundingModel)

/-- The divide kernel terminates under `execR R` from every state. -/
private theorem softmaxDiv_execR_isSome (hN : 0 < N) :
    ∃ s', execR R ((stableSoftmaxKernel xReg yReg N).toAlgKernel) s
      = some s' := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  simp [execR, stableSoftmaxKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComputeExpr.toAlgorithm?,
        Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]

/-- The reciprocal kernel terminates under `execR R` from every state. -/
private theorem softmaxRecip_execR_isSome (hN : 0 < N) :
    ∃ s', execR R ((softmaxRecipKernel xReg yReg N).toAlgKernel) s
      = some s' := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  simp [execR, softmaxRecipKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComputeExpr.toAlgorithm?,
        Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]

/-- Frame lemma for the divide kernel: every cell outside the program's `y`
window `[pid·N, pid·N + N)` is untouched. -/
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
        NumericDType.div, ComputeExpr.toAlgorithm?] at hExec
  simp [Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hExec
  subst s'
  refine Eq.trans
    (BlockState.foldl_writeMemAsR_preserve_cell R .bf16 _ _ r o _ ?_ _) rfl
  intro k _ hc
  rcases hcond with hne | hno
  · exact hne hc.1.symm
  · exact hno k.1 hc.2.symm

/-- Frame lemma for the reciprocal kernel: every cell outside the program's
`y` window is untouched. -/
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
        NumericDType.div, ComputeExpr.toAlgorithm?] at hExec
  simp [Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hExec
  subst s'
  refine Eq.trans
    (BlockState.foldl_writeMemAsR_preserve_cell R .bf16 _ _ r o _ ?_ _) rfl
  intro k _ hc
  rcases hcond with hne | hno
  · exact hne hc.1.symm
  · exact hno k.1 hc.2.symm

end SoftmaxReciprocal.regionRun

/-! ## Flat-memory bridge side conditions

Both kernels are register-indirect (`offs := pid * B + tl.arange(0, B);
tl.load(x + offs)`), so no ∀-state safety contract covers them; the
flat-memory bridge (v1.2, `execR` flavor) takes the per-execution
`Kernel.TraceSafeR` contract instead, plus `FlattenOk` (bridge fragment
membership). Each kernel makes exactly two accesses — the `x` load and the
bf16 `y` store, both through the unmasked whole-tile `offs` window — so each
buffer's bound must cover the program tile `[pid·B, pid·B + B)`; every other
statement (`program_id`, the `offs` index arithmetic, and the register ops
`max`/`exp`/`sum`/`div`/reciprocal) is memory-silent. -/
section SoftmaxReciprocal.bridge

variable (xReg yReg : RegionName) (N : Nat) (s : BlockState)
variable (R : RoundingModel)

/-- The divide kernel sits inside the bridge's covered fragment. -/
theorem softmaxDiv_flattenOk :
    ((stableSoftmaxKernel xReg yReg N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [stableSoftmaxKernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- The reciprocal kernel sits inside the bridge's covered fragment. -/
theorem softmaxRecip_flattenOk :
    ((softmaxRecipKernel xReg yReg N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [softmaxRecipKernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk for the divide kernel under `R`: one
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
/-- Per-execution safety walk for the reciprocal kernel under `R`: same
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

end SoftmaxReciprocal.bridge

/-! ## The spec: `softmaxDivIO ≡[R] softmaxRecipIO` -/
section SoftmaxReciprocal.spec

/-- The divide kernel's **IO signature** — the whole kernel-specific audit
surface of the headline: which buffer is the input (`x`) and which the
output (`y`), the tile lengths (`Bin = Bout = B`: softmax is a whole-tile
map), where program `pid` reads/writes its tile (`pid * B`, the kernel's
`offs = pid * B + tl.arange(0, B)` window), and no scratch (nothing is
staged through memory). The windows are declared, not parsed from the
kernel: the headline **proves** the kernel's actual addressing matches
them. -/
def softmaxDivIO (B : Nat) : KernelIO₁ where
  kernel := stableSoftmaxKernel ⟨"x"⟩ ⟨"y"⟩ B
  inp := ⟨"x"⟩
  out := ⟨"y"⟩
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

/-- The reciprocal kernel's IO signature: the **same interface by
construction** (structure update of `softmaxDivIO` — buffers and windows
shared verbatim), with the reciprocal-multiply kernel plugged in. -/
def softmaxRecipIO (B : Nat) : KernelIO₁ :=
  { softmaxDivIO B with kernel := softmaxRecipKernel ⟨"x"⟩ ⟨"y"⟩ B }

/-- **The headline**: the per-element-divide softmax is equivalent to the
precomputed-reciprocal softmax on their shared IO signature, for **every**
rounding model — see the module docstring for the full contract `≡[R]`
unfolds to (both kernels terminate from any state with a nonempty tile,
`y` windows agree, each frames outside its `y` window; no scratch on either
side). Proof: `KernelIO₁.Equiv.intro` assembles the region-model
equivalence — termination and the frames from the computational unfold, the
`y`-agreement leg from the region-level refinement core
`softmax_reciprocal_refinement_view` (`e / S = e · S⁻¹` rounded at the same
bf16 store), unpacked at the execution pair by the library
`ComputeRefine.Refines.out` — with the flat-memory bridge side conditions
(`FlattenOk` + `TraceSafeR` per kernel). -/
specification softmax_reciprocal_equiv (R : RoundingModel) (B : Nat)
    (hB : 0 < B) :
    softmaxDivIO B ≡[R] softmaxRecipIO B := by
  refine KernelIO₁.Equiv.intro _ _ ?_ ?_ ?_ ?_ ?_
  · -- FlattenOk, divide
    exact softmaxDiv_flattenOk ⟨"x"⟩ ⟨"y"⟩ B
  · -- FlattenOk, reciprocal
    exact softmaxRecip_flattenOk ⟨"x"⟩ ⟨"y"⟩ B
  · -- TraceSafeR, divide
    intro bounds t h1 h2 _
    simp only [softmaxDivIO] at h1 h2 ⊢
    exact softmaxDiv_traceSafeR ⟨"x"⟩ ⟨"y"⟩ B t R hB bounds h1 h2
  · -- TraceSafeR, reciprocal (window hypotheses are io₁'s = the shared ones)
    intro bounds t h1 h2 _
    simp only [softmaxDivIO, softmaxRecipIO] at h1 h2 ⊢
    exact softmaxRecip_traceSafeR ⟨"x"⟩ ⟨"y"⟩ B t R hB bounds h1 h2
  · -- the region-model equivalence, from ANY state s₀ (no input hypotheses)
    intro s₀
    simp only [softmaxDivIO, softmaxRecipIO]
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

end SoftmaxReciprocal.spec

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom — in the region-level core's and the
-- public headline's transitive proofs.
#axiomsClean softmax_reciprocal_refinement_view
#axiomsClean softmax_reciprocal_equiv

-- (2) The headline's statement surface is the two IO signatures plus the
-- audit-once kernel-equivalence combinator — NO spec (there are none on the
-- rounding layer), no other project constant.
#stmtSurfaceSubset softmax_reciprocal_equiv ⊆
  [softmaxDivIO, softmaxRecipIO, VeriTile.Triton.KernelIO₁.Equiv,
   RoundingModel]

/-! ## Exact-ℝ companion theorems

Real Triton optimization: replace `y = e/s` (per-element division) with
`inv_s = 1/s; y = e * inv_s` (one division total + one multiply per element).
Algorithmically equivalent in ℝ since `e/s = e * (1/s)` when `s ≠ 0`.

The math denotation (`stableSpec`, `tileMax`) lives in
`VeriTile.Triton.Math.Softmax`; the observation vocabulary (`InputLoadedAt`,
`observeAt`, `programTileView`) in `VeriTile.Examples.Common`. The divide side
of the comparison is `stableSoftmaxKernelReal`, proven correct against
`stableSpec` in `bench/examples/SoftmaxEqRefine.lean`; this file carries a
`private` copy of that closed form (bench showcases are self-contained). -/
section SoftmaxReciprocal.theoremsReal

open VeriTile.Examples

/-- The load-bearing math identity: division equals multiplication by
    reciprocal (for non-zero divisor). -/
theorem div_eq_mul_inv_real (a s : ℝ) (hs : s ≠ 0) : a / s = a * (1 / s) := by
  field_simp

/-- Closed-form spec for `softmaxRecipKernelReal`'s `Y[pid*N+i]` cell. -/
noncomputable def stableRecipSpec {N : Nat} (xs : Fin N → ℝ) (m : ℝ) (i : Fin N) : ℝ :=
  Real.exp (xs i - m) * (1 / ∑ j, Real.exp (xs j - m))

/-- Local copy of the divide-side closed form (plumbing; the showcased
original is `softmax_stable_correct` in `bench/examples/SoftmaxEqRefine.lean`). -/
private theorem softmax_stable_correct
    (xReg yReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState) (xs : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs) :
    ∀ i : Fin blockSize,
      observeAt (exec (stableSoftmaxKernelReal xReg yReg blockSize) s) yReg blockSize s.pid i
        = some (stableSpec xs (tileMax hN xs) i) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [n + 1] => s.pid * (n + 1) + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, stableSoftmaxKernelReal, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, stableSpec, tileMax]
  repeat unfold evalOp
  simp [observeAt, exec, stableSoftmaxKernelReal, stepStmts, stepStmt,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, stableSpec, tileMax]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  unfold InputLoadedAt at _h_x
  simp [_h_x]
  rfl

/-- **Reciprocal-form softmax kernel correctness (exact ℝ).** -/
theorem softmax_recip_correct
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs) :
    ∀ i : Fin N,
      observeAt (exec (softmaxRecipKernelReal xReg yReg N) s) yReg N s.pid i
        = some (stableRecipSpec xs (tileMax hN xs) i) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [n + 1] => s.pid * (n + 1) + idx.1.val) :=
    injective_offset_singleton (s.pid * (n + 1))
  simp [observeAt, exec, softmaxRecipKernelReal, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, stableRecipSpec, tileMax]
  repeat unfold evalOp
  simp [observeAt, exec, softmaxRecipKernelReal, stepStmts, stepStmt,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, stableRecipSpec, tileMax]
  unfold InputLoadedAt at _h_x
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x]
  rfl

/-- **Refinement (exact ℝ): stable softmax with per-element division ≡ stable
    softmax with precomputed reciprocal.** Composes via `div_eq_mul_inv_real`. -/
theorem softmax_reciprocal_refinement
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoadedAt s xReg N xs) :
    ∀ i : Fin N,
      observeAt (exec (stableSoftmaxKernelReal xReg yReg N) s) yReg N s.pid i =
      observeAt (exec (softmaxRecipKernelReal  xReg yReg N) s) yReg N s.pid i := by
  intro i
  rw [softmax_stable_correct xReg yReg N hN s xs h_x i,
      softmax_recip_correct  xReg yReg N hN s xs h_x i]
  congr 1
  -- Goal: stableSpec xs (tileMax hN xs) i = stableRecipSpec xs (tileMax hN xs) i
  -- These differ only by a / b vs a * (1/b).
  unfold stableSpec stableRecipSpec
  have h_sum_pos : 0 < ∑ j, Real.exp (xs j - tileMax hN xs) := by
    apply Finset.sum_pos
    · intro j _; exact Real.exp_pos _
    · obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
      exact ⟨⟨0, hN⟩, Finset.mem_univ _⟩
  exact div_eq_mul_inv_real _ _ (ne_of_gt h_sum_pos)

/-- View-level surface for `softmax_reciprocal_refinement`. -/
theorem softmax_reciprocal_refinement_exec_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    ∀ idx : TileIndex [N],
      TensorView.observe (exec (stableSoftmaxKernelReal xReg yReg N) s)
          (programTileView s yReg N) idx =
      TensorView.observe (exec (softmaxRecipKernelReal  xReg yReg N) s)
          (programTileView s yReg N) idx := by
  intro idx
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := N) (xs := xs) h_x
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt]
    using softmax_reciprocal_refinement xReg yReg N hN s xs hx idx.1

/-! ### Trust audit for the exact-ℝ companions -/

#axiomsClean div_eq_mul_inv_real
#axiomsClean softmax_recip_correct
#axiomsClean softmax_reciprocal_refinement
#axiomsClean softmax_reciprocal_refinement_exec_view

end SoftmaxReciprocal.theoremsReal

end VeriTile.Bench.Examples.SoftmaxReciprocal
