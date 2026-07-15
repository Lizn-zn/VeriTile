import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Softmax: naive vs stable — kernel equivalence `≡[R]` on the shared IO surface

Self-contained showcase, read top to bottom: **kernels** first, the
**supporting lemma** in the middle (`private` plumbing — bf16-scatter
congruence), the region-level refinement core next
(`softmax_kernels_refinement_view`), the **termination/frame lemmas**, the
**flat-memory bridge side conditions**, and the **specification** (one public
headline `softmax_stable_equiv` on the `≡[R]` surface) with its compile-time
**trust audit**. The real sections below are `Softmax.kernels`,
`Softmax.lemmas`, `Softmax.theorems`, `Softmax.run`, `Softmax.bridge`,
`Softmax.spec`.

Two block-parallel softmax kernels compute `y = exp(x) / Σ exp(x)` per row and
**store the result rounded to bf16** (`(y).to(tl.bfloat16)`):
`naiveSoftmaxKernel` exponentiates `x` raw; `stableSoftmaxKernel` subtracts the
row max first. The reductions run in ℝ (no intermediate rounding), so both
kernels produce the **same** per-lane ℝ value — the shift-cancellation of
softmax (`naive_eq_stable`) — and the only rounding is the shared bf16 output
store, which quantizes equal values identically. No idempotence is needed
(contrast `bench/examples/FusedSwiglu.lean`, where a re-round must collapse):
here the rounding *sites* already coincide one-for-one.

## The public result (bottom of file)

The single public headline is **`softmax_stable_equiv`** — kernel equivalence
on the shared one-input IO signature:

    softmaxNaiveIO B ≡[R] softmaxStableIO B

`≡[R]` is the audit-once kernel-equivalence combinator (`KernelIO₁.Equiv`,
`VeriTile.Triton.Memory.KernelSpec`), the `⊨`-grade form of the refinement
surface. Spelled out, the headline says: for **every** disjoint flat placement
of the interface buffers `x`/`y` (∀ base pointers, ∀ buffer sizes — neither
kernel stages anything through memory, so there is no scratch), **every**
program id whose whole-tile windows `[pid * B, pid * B + B)` are in bounds,
and **every** launch state of that program — **no input hypotheses at all**,
not even loaded inputs: "equal inputs" is simply "the same `s₀`" — both
translated pointer kernels terminate under `execR R`, their output windows
hold equal values, and each kernel leaves every cell outside the output
window untouched.
Non-emptiness `0 < B` is required: the stable kernel's `max` reduction (like
`Finset.sup'`) is only defined on non-empty tiles.

The statement mentions only the two IO signatures and the library equivalence
surface — **no spec** (the `#stmtSurfaceSubset` gate below enforces this).
Everything else — the region-level refinement `softmax_kernels_refinement_view`
that the `≡[R]` obligation repackages, the termination/frame lemmas, and the
bridge side conditions — is scaffolding.

## The exact-ℝ surface

There is no separate exact layer: the headline quantifies over **every**
rounding model, and at `R := .triv` the `castTo` stores are inert
(`execR_triv`), so the exact-ℝ equivalence is the headline's degeneration —
no raw-store twin kernels needed.
-/

namespace VeriTile.Bench.Examples.Softmax

open VeriTile.Triton VeriTile.Triton.TiledSoftmax
open scoped VeriTile.Triton.KernelIO₁

/-! ## Kernels -/
section Softmax.kernels

/-- Naive softmax: `y = exp(x) / Σ exp(x)`, exponentiated raw, stored bf16. -/
def naiveSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

/-- Numerically-stable softmax: subtract the row max first,
`y = exp(x − m) / Σ exp(x − m)`, stored bf16. -/
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


end Softmax.kernels

/-! ## Supporting lemma (private plumbing) -/
section Softmax.lemmas

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

end Softmax.lemmas

/-! ## The region-level refinement core -/
section Softmax.theorems

/- Shared parameters of the refinement core: the two regions, the block size
(nonempty), the state, and the rounding model `R`. -/
variable (xReg yReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
variable (R : RoundingModel)

include hN in
/-- **naive refines stable** (`ComputeRefine.Refines R`, no scratch): for the
rounding model `R`, from the same initial state the naive and stable softmax
kernels perform the same writes. Both compute the same per-lane ℝ softmax value
(shift-cancellation) and round it at the same bf16 store.

Formerly the file's headline; now the mathematical core the `≡[R]`
specification's region-model obligation repackages (its output-window
agreement leg is exactly this whole-memory agreement read at the output
lanes). -/
theorem softmax_kernels_refinement_view :
    ComputeRefine.Refines R
      (naiveSoftmaxKernel xReg yReg blockSize)
      (stableSoftmaxKernel xReg yReg blockSize) s [] := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  apply ComputeKernel.computeRefineR_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp [execR, naiveSoftmaxKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.div,
        ComputeExpr.toAlgorithm?] at hL
  simp [Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hL
  simp [execR, stableSoftmaxKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComputeExpr.toAlgorithm?] at hR
  simp [Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hR
  subst lhs'
  subst rhs'
  refine foldl_writeMemAsR_mem_congr R .bf16 _ _ _ _ ?_ r o _ _ rfl
  intro k _
  exact congrArg (fun v : ℝ => R.cast FloatDType.real FloatDType.bf16 (some v))
    (congrFun (naive_eq_stable
      (fun i : Fin (n + 1) => s.readMem xReg (s.pids 0 * (n + 1) + i.val)) _) k.1)

-- No `sorry`, no smuggled axiom, in the refinement core's transitive proof.
#axiomsClean softmax_kernels_refinement_view

end Softmax.theorems

/-! ## Termination and frame (the region-model raw material for `≡[R]`)

`≡[R]`'s region-model obligation starts from **any** state — no loaded
inputs, no clean undef — so nothing here may take an input hypothesis.
Termination holds outright (both bodies are straight-line statement lists
whose every step is total once the tile is non-empty — the stable kernel's
`max` reduce is `none` on an empty axis, which is where `0 < blockSize`
earns its keep); the frame lemmas walk the same exec closed forms and apply
the library's unmasked-scatter cell frame
`BlockState.foldl_writeMemAsR_preserve_cell`. -/
section Softmax.run

variable (xReg yReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
variable (s : BlockState) (R : RoundingModel)

include hN in
/-- The naive kernel terminates under `execR R` from every state. -/
private theorem softmax_naive_execR_isSome :
    ∃ s', execR R ((naiveSoftmaxKernel xReg yReg blockSize).toAlgKernel) s
      = some s' := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  simp [execR, naiveSoftmaxKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.div,
        ComputeExpr.toAlgorithm?, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]

include hN in
/-- The stable kernel terminates under `execR R` from every state; the `max`
reduce needs the non-empty tile. -/
private theorem softmax_stable_execR_isSome :
    ∃ s', execR R ((stableSoftmaxKernel xReg yReg blockSize).toAlgKernel) s
      = some s' := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  simp [execR, stableSoftmaxKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComputeExpr.toAlgorithm?,
        Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]

include hN in
/-- Frame lemma for the naive kernel: every cell outside the output window
(`yReg` at `[pid * B, pid * B + B)`) is untouched. -/
private theorem softmax_naive_frameR (s' : BlockState) (r : RegionName) (o : Nat)
    (hcond : r ≠ yReg ∨ ∀ j : Fin blockSize, o ≠ s.pid * blockSize + j.val)
    (hExec : execR R ((naiveSoftmaxKernel xReg yReg blockSize).toAlgKernel) s
      = some s') :
    s'.mem r o = s.mem r o := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  simp [execR, naiveSoftmaxKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.div,
        ComputeExpr.toAlgorithm?, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hExec
  subst s'
  refine Eq.trans
    (BlockState.foldl_writeMemAsR_preserve_cell R .bf16 _ _ r o _ ?_ _) rfl
  intro k _ hc
  rcases hcond with hne | hno
  · exact hne hc.1.symm
  · exact hno k.1 hc.2.symm

include hN in
/-- Frame lemma for the stable kernel: every cell outside the output window
is untouched. -/
private theorem softmax_stable_frameR (s' : BlockState) (r : RegionName) (o : Nat)
    (hcond : r ≠ yReg ∨ ∀ j : Fin blockSize, o ≠ s.pid * blockSize + j.val)
    (hExec : execR R ((stableSoftmaxKernel xReg yReg blockSize).toAlgKernel) s
      = some s') :
    s'.mem r o = s.mem r o := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  simp [execR, stableSoftmaxKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComputeExpr.toAlgorithm?,
        Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hExec
  subst s'
  refine Eq.trans
    (BlockState.foldl_writeMemAsR_preserve_cell R .bf16 _ _ r o _ ?_ _) rfl
  intro k _ hc
  rcases hcond with hne | hno
  · exact hne hc.1.symm
  · exact hno k.1 hc.2.symm

end Softmax.run

/-! ## Flat-memory bridge side conditions

Both kernels are register-indirect (`offs = pid * B + tl.arange(0, B);
tl.load(x + offs)`), so no ∀-state safety contract covers them; the
flat-memory bridge (v1.2, `execR` flavor) takes the per-execution
`Kernel.TraceSafeR` contract instead, plus `FlattenOk` (bridge fragment
membership). Both accesses on both sides are **unmasked whole-tile**: the
load's bound is the whole window `pid * B + B ≤ bounds x`, the store's
`pid * B + B ≤ bounds y`; every other statement (`program_id`, the index
arithmetic, the reductions) is memory-silent. -/
section Softmax.bridge

variable (xReg yReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
variable (s : BlockState) (R : RoundingModel)

/-- The naive kernel sits inside the bridge's covered fragment. -/
theorem softmax_naive_flattenOk :
    ((naiveSoftmaxKernel xReg yReg blockSize).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [naiveSoftmaxKernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- The stable kernel sits inside the bridge's covered fragment. -/
theorem softmax_stable_flattenOk :
    ((stableSoftmaxKernel xReg yReg blockSize).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [stableSoftmaxKernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
include hN in
/-- Per-execution safety walk for the naive kernel under `R`: one
computational unfold walks all seven statements, discharging every
memory-silent statement's `SafeAtR` and reducing the two unmasked whole-tile
accesses' address obligations to the window bounds hypotheses. -/
theorem softmax_naive_traceSafeR (bounds : RegionBounds)
    (hx : s.pid * blockSize + blockSize ≤ bounds xReg)
    (hy : s.pid * blockSize + blockSize ≤ bounds yReg) :
    Kernel.TraceSafeR R bounds
      ((naiveSoftmaxKernel xReg yReg blockSize).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  unfold Kernel.TraceSafeR
  simp only [BlockState.pid_eq] at hx hy
  simp [naiveSoftmaxKernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    MaskOpt.SafeAtR, MemAccess.SafeAtR, stepStmtR, evalOpR.eq_def,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MaskOpt.ActiveR, BlockState.setReg,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.div,
    Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex]
  exact ⟨fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hx,
    fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hy⟩

set_option maxHeartbeats 1600000 in
include hN in
/-- Per-execution safety walk for the stable kernel under `R`: same
computational unfold over the eight statements (the extra `max` reduce is
memory-silent, but its step only computes on a non-empty tile). -/
theorem softmax_stable_traceSafeR (bounds : RegionBounds)
    (hx : s.pid * blockSize + blockSize ≤ bounds xReg)
    (hy : s.pid * blockSize + blockSize ≤ bounds yReg) :
    Kernel.TraceSafeR R bounds
      ((stableSoftmaxKernel xReg yReg blockSize).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  unfold Kernel.TraceSafeR
  simp only [BlockState.pid_eq] at hx hy
  simp [stableSoftmaxKernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    MaskOpt.SafeAtR, MemAccess.SafeAtR, stepStmtR, evalOpR.eq_def,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MaskOpt.ActiveR, BlockState.setReg,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
    NumericDType.div, Tile.reduceSumDrop, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hx,
    fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hy⟩

end Softmax.bridge

/-! ## The spec: `softmaxNaiveIO ≡[R] softmaxStableIO` -/
section Softmax.spec

/-- The stable kernel's **IO signature** — the whole kernel-specific audit
surface of the headline: interface `x` → `y`, whole-tile map
(`Bin = Bout = B`), each program owning the window `[pid * B, pid * B + B)`
on both buffers, no scratch (nothing is staged through memory). The windows
are declared, not parsed from the kernel: the headline **proves** the
kernel's actual addressing matches them. -/
def softmaxStableIO (B : Nat) : KernelIO₁ where
  kernel := stableSoftmaxKernel ⟨"x"⟩ ⟨"y"⟩ B
  inp := ⟨"x"⟩
  out := ⟨"y"⟩
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

/-- The naive kernel's IO signature: the **same interface by construction**
(structure update of `softmaxStableIO` — buffers and windows shared
verbatim), with the naive kernel plugged in. -/
def softmaxNaiveIO (B : Nat) : KernelIO₁ :=
  { softmaxStableIO B with kernel := naiveSoftmaxKernel ⟨"x"⟩ ⟨"y"⟩ B }

/-- **The headline**: naive softmax is equivalent to stable softmax on their
shared one-input IO signature, for **every** rounding model — see the module
docstring for the full contract `≡[R]` unfolds to (both kernels terminate
from any state, output windows agree, each frames outside the output
window). Non-emptiness `0 < B` is required by the stable kernel's `max`
reduce. Proof: `KernelIO₁.Equiv.intro` assembles the region-model
equivalence — termination is unconditional, the output-agreement leg is the
region-level refinement theorem's whole-memory agreement
(shift-cancellation + the shared bf16 store), the frames are the unmasked
scatter frame lemmas — with the flat-memory bridge side conditions
(`FlattenOk` + `TraceSafeR` per kernel). -/
specification softmax_stable_equiv (R : RoundingModel) (B : Nat) (hB : 0 < B) :
    softmaxNaiveIO B ≡[R] softmaxStableIO B := by
  refine KernelIO₁.Equiv.intro _ _ ?_ ?_ ?_ ?_ ?_
  · -- FlattenOk, naive
    exact softmax_naive_flattenOk ⟨"x"⟩ ⟨"y"⟩ B
  · -- FlattenOk, stable
    exact softmax_stable_flattenOk ⟨"x"⟩ ⟨"y"⟩ B
  · -- TraceSafeR, naive (no scratch: third hypothesis vacuous)
    intro bounds t h1 h2 _
    simp only [softmaxNaiveIO, softmaxStableIO] at h1 h2 ⊢
    exact softmax_naive_traceSafeR ⟨"x"⟩ ⟨"y"⟩ B hB t R bounds h1 h2
  · -- TraceSafeR, stable
    intro bounds t h1 h2 _
    simp only [softmaxNaiveIO, softmaxStableIO] at h1 h2 ⊢
    exact softmax_stable_traceSafeR ⟨"x"⟩ ⟨"y"⟩ B hB t R bounds h1 h2
  · -- the region-model equivalence, from ANY state s₀ (no input hypotheses)
    intro s₀
    simp only [softmaxNaiveIO, softmaxStableIO]
    -- termination of both sides
    obtain ⟨s1, hexec1⟩ := softmax_naive_execR_isSome ⟨"x"⟩ ⟨"y"⟩ B hB s₀ R
    obtain ⟨s2, hexec2⟩ := softmax_stable_execR_isSome ⟨"x"⟩ ⟨"y"⟩ B hB s₀ R
    -- the region-level refinement: memories agree EVERYWHERE (scratch = [])
    have hmem12 : ∀ r, r ∉ ([] : List RegionName) →
        ∀ o, s1.mem r o = s2.mem r o :=
      ComputeRefine.Refines.out
        (softmax_kernels_refinement_view ⟨"x"⟩ ⟨"y"⟩ B hB s₀ R)
        (by simp [naiveSoftmaxKernel, ComputeExpr.toAlgorithm?])
        (by simp [stableSoftmaxKernel, ComputeExpr.toAlgorithm?])
        hexec1 hexec2
    refine ⟨s1, s2, hexec1, hexec2, ?_, ?_, ?_⟩
    · -- output windows agree (a fortiori: the whole memories agree)
      intro j
      have hmem := hmem12 ⟨"y"⟩ (by simp) (s₀.pid * B + j.val)
      unfold BlockState.readMem
      rw [hmem]
    · -- naive frame: writes only the output window (no scratch)
      intro r o hcond _
      exact softmax_naive_frameR ⟨"x"⟩ ⟨"y"⟩ B hB s₀ R s1 r o hcond hexec1
    · -- stable frame: writes only the output window (no scratch)
      intro r o hcond _
      exact softmax_stable_frameR ⟨"x"⟩ ⟨"y"⟩ B hB s₀ R s2 r o hcond hexec2

end Softmax.spec

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in the public headline's transitive proof.
#axiomsClean softmax_stable_equiv

-- (2) The headline's statement surface is the two IO signatures plus the
-- audit-once kernel-equivalence combinator — NO spec (there are none), no
-- other project constant. If a spec-like definition ever creeps into the
-- statement, this fails.
#stmtSurfaceSubset softmax_stable_equiv ⊆
  [softmaxNaiveIO, softmaxStableIO, VeriTile.Triton.KernelIO₁.Equiv,
   RoundingModel]


end VeriTile.Bench.Examples.Softmax
