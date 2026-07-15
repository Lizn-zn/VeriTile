import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Log-sum-exp: direct vs shift-trick — kernel equivalence `≡[R]`

Self-contained showcase, read top to bottom: **kernels** first (what we
built), the region-level refinement core next (`log_sum_exp_refinement_view`,
proved directly — no `private` lemma section of its own), then the
**flat-memory bridge side conditions**, the **specification** (private
termination/frame plumbing plus one public headline `log_sum_exp_equiv` on
the `≡[R]` surface), a compile-time **trust audit**, and the exact-ℝ
companion layer last. The real
sections below are `LogSumExp.kernels`, `LogSumExp.theorems`,
`LogSumExp.bridge`, `LogSumExp.spec`, and `LogSumExp.exact` (dissolved from
the former library example `VeriTile.Examples.LogSumExpEq`).

Two block-parallel log-sum-exp kernels compute `y = log Σ exp(x)` per row and
**store the result rounded to bf16** (`(y).to(tl.bfloat16)`): `directLSEKernel`
sums `exp(x)` raw; `stableLSEKernel` subtracts the row max first
(`m + log Σ exp(x − m)`), the numerically-stable rewrite. The reductions run in
ℝ (no intermediate rounding), so both kernels produce the **same** ℝ value —
the shift-invariance of log-sum-exp (`log_sum_exp_shift_invariant`) — and the
only rounding is the shared bf16 output store. Any rounding model therefore
quantizes the two equal ℝ values identically, so the writes agree.

## The public result (bottom of `LogSumExp.spec`)

The single public headline is **`log_sum_exp_equiv`** — kernel equivalence on
the shared one-input IO signature:

    lseDirectIO B ≡[R] lseStableIO B    (0 < B)

`≡[R]` is the audit-once kernel-equivalence combinator (`KernelIO₁.Equiv`,
`VeriTile.Triton.Memory.KernelSpec`), the `⊨`-grade form of the refinement
surface. Spelled out, the headline says: for **every** disjoint flat
placement of the two interface buffers `x`/`y` (∀ base pointers, ∀ buffer
sizes), **every** program id whose windows are in bounds, and **every**
launch state — **no input hypotheses at all**, not even loaded inputs:
"equal inputs" is simply "the same `s₀`" — both translated pointer kernels
terminate under `execR R`, their output windows agree, and each kernel
leaves every cell outside its output window untouched. The shared signature
is a **reduction shape**: program `pid` reads the whole tile window
`[pid * B, pid * B + B)` of `x` and writes the single cell `y[pid]`
(`Bout = 1`, a scalar-per-program store — the `RowWiseSum` shape); neither
kernel stages anything through memory, so there is no scratch. The `0 < B`
hypothesis is genuinely forced: the shift-trick kernel's `tl.max` reduction
has no value on an empty tile, so at `B = 0` it does not terminate and the
equivalence is false.

The statement mentions only the two IO signatures and the library equivalence
surface — **no spec, and no input precondition** (the `#stmtSurfaceSubset`
gate below enforces this). The region-level refinement
`log_sum_exp_refinement_view` (`ComputeRefine.Refines`, the rounding-model
surface) is the mathematical core the headline's region-model obligation
repackages — unpacked at each execution pair by the library's
`ComputeRefine.Refines.out`. The compositional-rounding sibling (matched
intermediate casts, scratch staging) is `bench/examples/FusedSwiglu.lean`.

Alongside it, the `Exact` namespace (section `LogSumExp.exact`) keeps the
exact-store layer of the dissolved library example: the same two kernels
minus the bf16 output cast, their per-kernel closed-form correctness
(`Exact.direct_lse_correct`, `Exact.stable_lse_correct`), the exact-ℝ
refinement `Exact.log_sum_exp_refinement`, and its `TensorView` exec view.
-/

namespace VeriTile.Bench.Examples.LogSumExp

open VeriTile.Triton VeriTile.Triton.TiledLogSumExp VeriTile.Triton.TiledSoftmax
open scoped VeriTile.Triton.KernelIO₁
open VeriTile.Examples

/-! ## Kernels -/
section LogSumExp.kernels

/-- Direct log-sum-exp: `y = log Σ exp(x)`, summed raw (no max shift), stored
rounded to bf16. -/
def directLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e, axis=0)
  y    := tl.log(s)
  tl.store($(yReg) + pid, (y).to(tl.bfloat16))
}

/-- Numerically-stable log-sum-exp: subtract the row max first,
`y = m + log Σ exp(x − m)`, stored rounded to bf16. -/
def stableLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := m + tl.log(s)
  tl.store($(yReg) + pid, (y).to(tl.bfloat16))
}

end LogSumExp.kernels

/-! ## The region-level refinement core -/
section LogSumExp.theorems

/- Shared parameters of the refinement core, hoisted to a `variable` block: the
two regions, the block size (nonempty), the state, and the rounding model `R`. -/
variable (xReg yReg : RegionName) (N : Nat) (hN : 0 < N) (s : BlockState)
variable (R : RoundingModel)

include hN in
/-- **direct refines shift-trick** (`ComputeRefine.Refines R`, no scratch): for
the rounding model `R`, from the same initial state the direct and shift-trick
LSE kernels perform the same writes — their final memories agree at every cell.
Both compute the same ℝ log-sum-exp (shift-invariance) and round it at the same
bf16 store, so `R` quantizes equal values equally.

Formerly the file's headline; now the mathematical core the `≡[R]`
specification's region-model obligation repackages (its output-agreement leg
is exactly this memory agreement read at `y[pid]`). -/
theorem log_sum_exp_refinement_view :
    ComputeRefine.Refines R
      (directLSEKernel xReg yReg N)
      (stableLSEKernel xReg yReg N) s [] := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  apply ComputeKernel.computeRefineR_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp [execR, directLSEKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        ComputeExpr.toAlgorithm?] at hL
  repeat unfold evalOp at hL
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hL
  simp [execR, stableLSEKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComputeExpr.toAlgorithm?] at hR
  repeat unfold evalOp at hR
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hR
  subst lhs'
  subst rhs'
  rw [BlockState.writeMemAsR_mem, BlockState.writeMemAsR_mem]
  by_cases hc : r = yReg ∧ o = s.pids 0
  · rw [if_pos hc, if_pos hc]
    -- both store bf16-rounded copies of the SAME ℝ value (shift-invariance)
    exact congrArg (fun v : ℝ => MemCell.of _ (FloatDType.bf16.ofReal (R.storeValue .bf16 (FloatDType.bf16.ofReal (R.round .bf16 v)))))
      (log_sum_exp_shift_invariant hN _ _)
  · rw [if_neg hc, if_neg hc]
    rfl

end LogSumExp.theorems

/-! ## Flat-memory bridge side conditions

Both kernels are register-indirect (`offs = pid * B + tl.arange(0, B)` feeds
the load, the scalar `pid` register feeds the store), so no ∀-state safety
contract covers them; the flat-memory bridge (v1.2, `execR` flavor) takes the
per-execution `Kernel.TraceSafeR` contract instead, plus `FlattenOk` (bridge
fragment membership). Each kernel makes exactly two accesses — the unmasked
whole-tile load at `[pid * B, pid * B + B)` and the single-cell store at
`y[pid]` — and every other statement (`program_id`, the offset arithmetic,
`exp`/`log`, and the `sum`/`max` reductions) is memory-silent, so one
computational unfold walks every statement, reducing the two accesses'
address obligations to the window-bounds hypotheses. -/
section LogSumExp.bridge

/-- The direct kernel sits inside the bridge's covered fragment. -/
theorem direct_lse_flattenOk (xReg yReg : RegionName) (B : Nat) :
    ((directLSEKernel xReg yReg B).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [directLSEKernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- The shift-trick kernel sits inside the bridge's covered fragment. -/
theorem stable_lse_flattenOk (xReg yReg : RegionName) (B : Nat) :
    ((stableLSEKernel xReg yReg B).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [stableLSEKernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk for the direct kernel under `R`: one
computational unfold walks all seven statements (the five besides the load
and store — `program_id`, offsets, `exp`, `sum`, `log` — are memory-silent),
reducing the whole-tile load and the single-cell store to the two
window-bounds hypotheses. -/
theorem direct_lse_traceSafeR (xReg yReg : RegionName) (n : Nat)
    (s : BlockState) (R : RoundingModel) (bounds : RegionBounds)
    (hx : s.pid * (n + 1) + (n + 1) ≤ bounds xReg)
    (hy : s.pid + 1 ≤ bounds yReg) :
    Kernel.TraceSafeR R bounds
      ((directLSEKernel xReg yReg (n + 1)).toAlgKernel) s := by
  unfold Kernel.TraceSafeR
  simp only [BlockState.pid_eq] at hx hy
  simp [directLSEKernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    MaskOpt.SafeAtR, MemAccess.SafeAtR, stepStmtR, evalOpR.eq_def,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MaskOpt.ActiveR, BlockState.setReg,
    Tile.bop, Tile.uop, Tile.vec,
    NumericDType.add, NumericDType.mul,
    Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hx, hy⟩

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk for the shift-trick kernel under `R`: same
computational unfold over its eight statements; the extra `max` reduction is
memory-silent like the `sum`. -/
theorem stable_lse_traceSafeR (xReg yReg : RegionName) (n : Nat)
    (s : BlockState) (R : RoundingModel) (bounds : RegionBounds)
    (hx : s.pid * (n + 1) + (n + 1) ≤ bounds xReg)
    (hy : s.pid + 1 ≤ bounds yReg) :
    Kernel.TraceSafeR R bounds
      ((stableLSEKernel xReg yReg (n + 1)).toAlgKernel) s := by
  unfold Kernel.TraceSafeR
  simp only [BlockState.pid_eq] at hx hy
  simp [stableLSEKernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    MaskOpt.SafeAtR, MemAccess.SafeAtR, stepStmtR, evalOpR.eq_def,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MaskOpt.ActiveR, BlockState.setReg,
    Tile.bop, Tile.uop, Tile.vec,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    Tile.reduceSum, Tile.reduceSumDrop,
    Tile.reduceMax, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hx, hy⟩

end LogSumExp.bridge

/-! ## The spec: `lseDirectIO ≡[R] lseStableIO` -/
section LogSumExp.spec

/-- The shift-trick kernel's **IO signature** — the whole kernel-specific
audit surface of the headline: `x` in, `y` out; tile lengths `Bin = B`,
`Bout = 1` (a scalar-per-program reduction); program `pid` reads the whole
tile window at `pid * B` and writes the single cell `y[pid]`. No scratch:
neither kernel stages anything through memory. The windows are declared, not
parsed from the kernel: the headline **proves** the kernel's actual
addressing matches them. -/
def lseStableIO (B : Nat) : KernelIO₁ where
  kernel := stableLSEKernel ⟨"x"⟩ ⟨"y"⟩ B
  inp := ⟨"x"⟩
  out := ⟨"y"⟩
  Bin := B
  Bout := 1
  read := fun pid => pid * B
  write := fun pid => pid

/-- The direct kernel's IO signature: the **same interface by construction**
(structure update of `lseStableIO` — buffers and windows shared verbatim),
with the direct kernel plugged in. -/
def lseDirectIO (B : Nat) : KernelIO₁ :=
  { lseStableIO B with kernel := directLSEKernel ⟨"x"⟩ ⟨"y"⟩ B }

/-! Private plumbing for the headline's region-model obligation: `≡[R]` runs
both kernels from **any** state (no input hypotheses), so termination cannot
come from an input-conditioned realization — it holds outright (both bodies
are straight-line statement lists whose every step is total once `B = n + 1`
makes the reductions defined). The frames are one-`writeMemAsR` arguments
(single-cell `BlockState.writeMemAsR_mem`, no foldl): each kernel performs a
single scalar store at `y[pid]`, so everything else is untouched. -/

/-- The direct kernel terminates under `execR R` from every state. -/
private theorem direct_lse_execR_isSome (xReg yReg : RegionName) (n : Nat)
    (R : RoundingModel) (s₀ : BlockState) :
    ∃ s', execR R ((directLSEKernel xReg yReg (n + 1)).toAlgKernel) s₀
      = some s' := by
  simp [execR, directLSEKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        ComputeExpr.toAlgorithm?]
  simp [Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]

/-- The shift-trick kernel terminates under `execR R` from every state. -/
private theorem stable_lse_execR_isSome (xReg yReg : RegionName) (n : Nat)
    (R : RoundingModel) (s₀ : BlockState) :
    ∃ s', execR R ((stableLSEKernel xReg yReg (n + 1)).toAlgKernel) s₀
      = some s' := by
  simp [execR, stableLSEKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        NumericDType.sub, ComputeExpr.toAlgorithm?]
  simp [Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]

/-- Frame lemma for the direct kernel: its single store is the scalar cell
`y[pid]`, so every other cell is untouched (one `writeMemAsR`, no foldl). -/
private theorem direct_lse_frameR (xReg yReg : RegionName) (n : Nat)
    (R : RoundingModel) (s₀ s1 : BlockState)
    (hExec : execR R ((directLSEKernel xReg yReg (n + 1)).toAlgKernel) s₀
      = some s1)
    (r : RegionName) (o : Nat) (hmiss : ¬(r = yReg ∧ o = s₀.pids 0)) :
    s1.mem r o = s₀.mem r o := by
  simp [execR, directLSEKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        ComputeExpr.toAlgorithm?] at hExec
  simp [Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
    at hExec
  subst hExec
  rw [BlockState.writeMemAsR_mem]
  exact if_neg hmiss

/-- Frame lemma for the shift-trick kernel: same single-cell store at
`y[pid]`. -/
private theorem stable_lse_frameR (xReg yReg : RegionName) (n : Nat)
    (R : RoundingModel) (s₀ s2 : BlockState)
    (hExec : execR R ((stableLSEKernel xReg yReg (n + 1)).toAlgKernel) s₀
      = some s2)
    (r : RegionName) (o : Nat) (hmiss : ¬(r = yReg ∧ o = s₀.pids 0)) :
    s2.mem r o = s₀.mem r o := by
  simp [execR, stableLSEKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        NumericDType.sub, ComputeExpr.toAlgorithm?] at hExec
  simp [Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
    at hExec
  subst hExec
  rw [BlockState.writeMemAsR_mem]
  exact if_neg hmiss

/-- **The headline**: the direct LSE kernel is equivalent to the shift-trick
kernel on their shared one-input IO signature, for **every** rounding model
and **every** block size `0 < B` — see the module docstring for the full
contract `≡[R]` unfolds to (both kernels terminate from any state, the
single output cell `y[pid]` agrees, each frames outside it). `0 < B` is
genuinely forced: the `tl.max` reduction has no value on an empty tile.
Proof: `KernelIO₁.Equiv.intro` assembles the region-model equivalence —
termination is unconditional, the output-agreement leg is the region-level
refinement theorem's memory agreement (shift-invariance of log-sum-exp,
unpacked at the execution pair by `ComputeRefine.Refines.out`), the frames
are one-`writeMemAsR` arguments — with the flat-memory bridge side
conditions (`FlattenOk` + `TraceSafeR` per kernel). -/
specification log_sum_exp_equiv (R : RoundingModel) (B : Nat) (hB : 0 < B) :
    lseDirectIO B ≡[R] lseStableIO B := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  refine KernelIO₁.Equiv.intro _ _ ?_ ?_ ?_ ?_ ?_
  · -- FlattenOk, direct
    exact direct_lse_flattenOk ⟨"x"⟩ ⟨"y"⟩ (n + 1)
  · -- FlattenOk, shift-trick
    exact stable_lse_flattenOk ⟨"x"⟩ ⟨"y"⟩ (n + 1)
  · -- TraceSafeR, direct
    intro bounds t h1 h2 _
    simp only [lseDirectIO, lseStableIO] at h1 h2 ⊢
    exact direct_lse_traceSafeR ⟨"x"⟩ ⟨"y"⟩ n t R bounds h1 h2
  · -- TraceSafeR, shift-trick
    intro bounds t h1 h2 _
    simp only [lseDirectIO, lseStableIO] at h1 h2 ⊢
    exact stable_lse_traceSafeR ⟨"x"⟩ ⟨"y"⟩ n t R bounds h1 h2
  · -- the region-model equivalence, from ANY state s₀ (no input hypotheses)
    intro s₀
    simp only [lseDirectIO, lseStableIO]
    obtain ⟨s1, hexec1⟩ := direct_lse_execR_isSome ⟨"x"⟩ ⟨"y"⟩ n R s₀
    obtain ⟨s2, hexec2⟩ := stable_lse_execR_isSome ⟨"x"⟩ ⟨"y"⟩ n R s₀
    -- the region-level refinement: memories agree at EVERY cell (no scratch)
    have hmem12 : ∀ r, r ∉ ([] : List RegionName) →
        ∀ o, s1.mem r o = s2.mem r o :=
      (log_sum_exp_refinement_view ⟨"x"⟩ ⟨"y"⟩ (n + 1) n.succ_pos s₀ R).out
        rfl rfl hexec1 hexec2
    refine ⟨s1, s2, hexec1, hexec2, ?_, ?_, ?_⟩
    · -- the output cell y[pid] agrees
      intro j
      unfold BlockState.readMem
      rw [hmem12 ⟨"y"⟩ (by simp) (s₀.pid + j.val)]
    · -- direct frame: writes only the single cell y[pid]
      intro r o hcond _
      refine direct_lse_frameR ⟨"x"⟩ ⟨"y"⟩ n R s₀ s1 hexec1 r o ?_
      rintro ⟨rfl, rfl⟩
      rcases hcond with hne | hno
      · exact hne rfl
      · exact hno ⟨0, Nat.one_pos⟩ (by simp)
    · -- shift-trick frame: same single cell
      intro r o hcond _
      refine stable_lse_frameR ⟨"x"⟩ ⟨"y"⟩ n R s₀ s2 hexec2 r o ?_
      rintro ⟨rfl, rfl⟩
      rcases hcond with hne | hno
      · exact hne rfl
      · exact hno ⟨0, Nat.one_pos⟩ (by simp)

end LogSumExp.spec

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom — in the region-level core's and the
-- public headline's transitive proofs.
#axiomsClean log_sum_exp_refinement_view
#axiomsClean log_sum_exp_equiv

-- (2) The headline's statement surface is the two IO signatures plus the
-- audit-once kernel-equivalence combinator — NO spec (there are none), no
-- other project constant. If a spec-like definition ever creeps into the
-- statement, this fails.
#stmtSurfaceSubset log_sum_exp_equiv ⊆
  [lseDirectIO, lseStableIO, VeriTile.Triton.KernelIO₁.Equiv, RoundingModel]

-- (There is no `#specNonCircular` gate: the file defines no specs at all, so a
-- self-referential spec is impossible by construction.)

/-! ## Exact-ℝ layer (dissolved library example)

The former `VeriTile.Examples.LogSumExpEq` proved the same refinement for the
**exact-store** kernel variants — identical programs except the output store
writes the raw ℝ value (no `(y).to(tl.bfloat16)` cast). That layer lives on
here under the `Exact` namespace: single-cell observation `observeLSE`,
per-kernel correctness against the closed forms `LSE` /
`stableLSEWithShift`, the exact-ℝ refinement, and its `TensorView`
exec-level view. The math identity `log_sum_exp_shift_invariant` is the
load-bearing fact, exactly as in the rounded headline above. -/
section LogSumExp.exact

namespace Exact

/-- Direct log-sum-exp kernel, exact-store variant: `y = log(Σ exp(x))`,
stored unrounded. -/
def directLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e, axis=0)
  y    := tl.log(s)
  tl.store($(yReg) + pid, y)
}

/-- Shift-trick LSE kernel, exact-store variant: `y = m + log(Σ exp(x - m))`
where `m = max(x)`, stored unrounded. -/
def stableLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := m + tl.log(s)
  tl.store($(yReg) + pid, y)
}

/-- Read region `region` at offset `basePid` (single scalar per program_id).
    Unlike softmax which writes a tile, LSE writes one cell per pid. -/
noncomputable def observeLSE
    (sf : Option BlockState) (region : RegionName) (basePid : Nat) : Option ℝ :=
  sf.map (·.readMem region basePid)

/-- **Direct LSE kernel correctness.** Single-cell observation at Y[pid]. -/
theorem direct_lse_correct
    (xReg yReg : RegionName)
    (N : Nat) (_hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs) :
    observeLSE (exec (directLSEKernel xReg yReg N) s) yReg s.pid
      = some (LSE xs Bool.false 0) := by
  simp [observeLSE, exec, directLSEKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, LSE]
  repeat unfold evalOp
  simp [observeLSE, exec, directLSEKernel, stepStmts, stepStmt,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, LSE]
  unfold InputLoadedAt at _h_x
  simp_rw [_h_x]
  rfl

/-- **Stable LSE kernel correctness.** Single-cell observation; closed form
    is `m + log(Σ exp(x - m))` where `m = tileMax xs`. -/
theorem stable_lse_correct
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs) :
    observeLSE (exec (stableLSEKernel xReg yReg N) s) yReg s.pid
      = some (stableLSEWithShift xs (tileMax hN xs)) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  simp [observeLSE, exec, stableLSEKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        stableLSEWithShift, tileMax]
  repeat unfold evalOp
  simp [observeLSE, exec, stableLSEKernel, stepStmts, stepStmt,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        stableLSEWithShift, tileMax]
  unfold InputLoadedAt at _h_x
  simp_rw [_h_x]
  rfl

/-- **Refinement: `Exact.directLSEKernel` and `Exact.stableLSEKernel` produce
    the same `Y[pid]` value.** Composes the two correctness lemmas via the
    math identity `log_sum_exp_shift_invariant`. -/
theorem log_sum_exp_refinement
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoadedAt s xReg N xs) :
    observeLSE (exec (directLSEKernel xReg yReg N) s) yReg s.pid =
    observeLSE (exec (stableLSEKernel xReg yReg N) s) yReg s.pid := by
  rw [direct_lse_correct xReg yReg N hN s xs h_x,
      stable_lse_correct xReg yReg N hN s xs h_x]
  congr 1
  unfold LSE stableLSEWithShift
  simp
  exact log_sum_exp_shift_invariant hN xs (tileMax hN xs)

/-- View-level surface for `Exact.log_sum_exp_refinement`. -/
theorem log_sum_exp_refinement_exec_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    TensorView.observe (exec (directLSEKernel xReg yReg N) s)
        (scalarCellView yReg s.pid) PUnit.unit =
    TensorView.observe (exec (stableLSEKernel xReg yReg N) s)
        (scalarCellView yReg s.pid) PUnit.unit := by
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := N) (xs := xs) h_x
  simpa [TensorView.observe, observeTileAt, scalarCellView,
         TensorView.offset, Offset.strided, observeLSE]
    using log_sum_exp_refinement xReg yReg N hN s xs hx

end Exact

/-! ### Trust audit for the exact-ℝ layer -/

-- No `sorry`, no smuggled axiom, anywhere in the exact layer's proofs.
#axiomsClean Exact.direct_lse_correct
#axiomsClean Exact.stable_lse_correct
#axiomsClean Exact.log_sum_exp_refinement
#axiomsClean Exact.log_sum_exp_refinement_exec_view

end LogSumExp.exact

end VeriTile.Bench.Examples.LogSumExp
