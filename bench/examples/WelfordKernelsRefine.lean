import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Welford variance: two-pass vs online — writes-equality refinement

Self-contained showcase, read top to bottom: **kernels** first (the two Welford
kernels), the **supporting lemmas** in the middle (`private` plumbing — the
full Welford math, the loop invariant and its correctness, and the cell-level
memory characterizations), the **theorem** last (one public headline
`welford_kernels_refinement_view`), then a compile-time **trust audit**. The
three real sections below are `Welford.kernels`, `Welford.lemmas`,
`Welford.theorems`.

`twopassWelfordKernel` computes the row mean/variance with two `tl.sum` passes;
`onlineWelfordKernel` uses Welford's one-pass recurrence inside a `for` loop.
Like a real Triton kernel, both take a **`rowStride` argument** (stage 1 of
the address-layout roadmap): the input row lives at `pid * rowStride + [0,
blockSize)` (`InputRowLoadedAt`).
Both **store the mean and variance rounded to bf16** (`(·).to(tl.bfloat16)`); the
reductions run in ℝ (no intermediate rounding). The load-bearing identity is that
after processing all inputs the running `(M, S)` equals the two-pass `(μ, S)`
(`welford_eq_two_pass`), so from the same state both kernels produce the same ℝ
mean and variance, and the only rounding is the shared bf16 output stores, which
quantize equal values identically.

## The public result (bottom of file)

The single public headline is **`welford_kernels_refinement_view`** — a
kernel-vs-kernel refinement on `ComputeRefine.Refines R` (the rounding-model
surface): for the rounding model `R`, from the same state the two-pass and online
kernels perform the same writes (no scratch regions, so the scratch list is
`[]`). Its statement mentions only the two kernels, the loaded-input contract,
the writes-equality surface, and the state/region types — **no spec** (the
`#stmtSurfaceSubset` gate below enforces this; the Welford math and correctness
lemmas are all `private`). The compositional rounding pattern is
`bench/examples/FusedSwiglu.lean`.

## The exact originals (`Exact` namespace, end of file)

The final section is the former library example
`VeriTile.Examples.WelfordKernels`, dissolved into this showcase. Its kernels
are the exact-store originals (no `.to(tl.bfloat16)` boundary rounding, and the
row lives at `pid * blockSize` — no separate `rowStride` argument), so they are
**not** the kernels above; they and their per-kernel `exec` correctness results
(`Exact.twopass_welford_correct`, `Exact.online_welford_correct`) plus the
exact readMem/view-level refinements live under the `Exact` sub-namespace.
-/

namespace VeriTile.Bench.Examples.Welford

open VeriTile.Triton

open VeriTile.Triton.TiledReduction.WelfordRec
open VeriTile.Examples (InputRowLoadedAt inputLoadedAt_of_programTileView_loaded programTileView rowTileView inputRowLoadedAt_of_rowTileView_loaded castFin
  onlineWelfordLoopBody onlineWelfordLoopBody_castFree InputLoadedAt scalarCellView)

/-! ## Kernels -/
section Welford.kernels

/-- Two-pass mean/variance kernel: mean with one `tl.sum`, then population
variance with a second `tl.sum` over squared deviations. -/
def twopassWelfordKernel (xReg meanReg varReg : RegionName)
    (blockSize rowStride : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(rowStride) + tl.arange($(blockSize))
  x      := tl.load($(xReg) + offs)
  s_x    := tl.sum(x)
  μ      := s_x / tl.toReal($(blockSize))
  d      := x - μ
  s_d2   := tl.sum(d * d)
  v      := s_d2 / tl.toReal($(blockSize))
  tl.store($(meanReg), (μ).to(tl.bfloat16))
  tl.store($(varReg), (v).to(tl.bfloat16))
}

/-- Online Welford mean/variance kernel: the same population mean/variance via
Welford's one-pass recurrence inside a `for` loop (running mean `M`, running
sum-of-squared-deviations `S`). -/
def onlineWelfordKernel (xReg meanReg varReg : RegionName)
    (blockSize rowStride : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  M   := 0
  S   := 0
  tl.for i in $(blockSize) {
    xi     := tl.load($(xReg) + (pid * $(rowStride) + i))
    delta  := xi - M
    M      := M + delta / (tl.toReal(i) + 1)
    delta2 := xi - M
    S      := S + delta * delta2
  }
  tl.store($(meanReg), (M).to(tl.bfloat16))
  tl.store($(varReg), (S / tl.toReal($(blockSize))).to(tl.bfloat16))
}

end Welford.kernels

/- Variables shared by every lemma and the theorem below — declared **once**, at
namespace scope, so both `Welford.lemmas` and `Welford.theorems` inherit them
(`end <section>` only clears variables declared *inside* that section). Hoisted
out of the declarations so each signature carries only its genuine hypotheses:
the compact `InputRowLoadedAt` input contract and the `meanReg ≠ varReg`
aliasing constraint. -/
variable (xReg meanReg varReg : RegionName) (blockSize rowStride : Nat) (hN : 0 < blockSize)
variable (s : BlockState) (xs : Fin blockSize → ℝ) (R : RoundingModel)

/-! ## Supporting lemmas (private plumbing) -/
section Welford.lemmas

-- Welford two-pass/recurrence math (twoPassMean/twoPassS/welfordMean/welfordS +
-- welford_eq_two_pass) now lives in `VeriTile.Triton.TiledReduction.WelfordRec`,
-- opened above; shared with the LayerNorm showcase and VeriTile.Examples.WelfordKernels.
-- `onlineWelfordLoopBody` + its cast-free degeneration are shared from
-- `VeriTile.Examples.Common`; `stepForLoopAuxR_castFree` + `writeMemAsR_regs`
-- live in the library (`VeriTile.Triton.Float.StepR`).

/-- Loop invariant for `onlineWelfordKernel`: after `k` body iterations,
    register `M` holds `welfordMean xs k`, register `S` holds `welfordS xs k`,
    register `pid` holds the original program id, and the input region is
    unchanged. -/
private def P_welford {N : Nat} (rowStride : Nat) (xs : Fin N → ℝ)
    (xReg : RegionName)
    (origPid : Nat) (k : Nat) (s : BlockState) : Prop :=
  s.regs .real [] "M" = some (Tile.scalar (welfordMean xs k))
  ∧ s.regs .real [] "S" = some (Tile.scalar (welfordS xs k))
  ∧ s.regs .nat [] "pid" = some (Tile.scalar origPid)
  ∧ s.pid = origPid
  ∧ InputRowLoadedAt s xReg rowStride N xs

private theorem online_welford_step
    {N : Nat} (xs : Fin N → ℝ) (xReg : RegionName) (origPid i : Nat)
    (s : BlockState) (hi : i < N)
    (hP : P_welford rowStride xs xReg origPid i s) :
    ∃ s',
      stepStmts (onlineWelfordLoopBody xReg rowStride)
        (s.setReg "i" .nat [] (Tile.scalar i)) = some s' ∧
      P_welford rowStride xs xReg origPid (i + 1) s' := by
  rcases hP with ⟨hM, hS, hpidReg, hpid, hX⟩
  let xi : ℝ := s.readMem xReg (origPid * rowStride + i)
  have hxi : xi = xs ⟨i, hi⟩ := by
    have hx := hX ⟨i, hi⟩
    rw [hpid] at hx
    exact hx
  let m : ℝ := welfordMean xs i
  let ssum : ℝ := welfordS xs i
  let delta : ℝ := xi - m
  let m' : ℝ := m + delta / ((i : ℝ) + 1)
  let delta2 : ℝ := xi - m'
  let ssum' : ℝ := ssum + delta * delta2
  let s' :=
    (((((s.setReg "i" .nat [] (Tile.scalar i)).setReg
      "xi" .real [] (Tile.scalar xi)).setReg
      "delta" .real [] (Tile.scalar delta)).setReg
      "M" .real [] (Tile.scalar m')).setReg
      "delta2" .real [] (Tile.scalar delta2)).setReg
      "S" .real [] (Tile.scalar ssum')
  refine ⟨s', ?_, ?_⟩
  · -- Reduce loop body via simp; remaining goal is structural equality of
    -- WithBot ℝ arithmetic terms (↑a + ↑b vs ↑(a + b) etc.) which matches via
    -- WithBot.coe_add / coe_mul in reverse, plus rfl on the outer setReg shell.
    simp [onlineWelfordLoopBody, stepStmts, stepStmt, evalOp, Tile.bop,
      Tile.natToReal, NumericDType.add, NumericDType.mul, NumericDType.sub,
      NumericDType.div, BlockState.readMem, hM, hS, hpidReg,
      xi, m, ssum, delta, m', delta2, ssum', s',
      WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv,
      BlockState.setReg]
    rfl
  · simp [P_welford, s', InputRowLoadedAt, welfordMean, welfordS, hi, xi, m,
      ssum, delta, m', delta2, ssum', hpidReg, hpid, hxi]
    intro j
    have hx := hX j
    rw [hpid] at hx
    exact hx

/-- A successful `assign` step only touches registers, never memory. -/
private theorem stepStmt_assign_mem {dtype : TileDType} {shape : TileShape}
    {name : RegName} {e : Op dtype shape} {t t' : BlockState}
    (h : stepStmt (Stmt.assign dtype shape name e) t = some t')
    (r : RegionName) (o : Nat) : t'.mem r o = t.mem r o := by
  simp only [stepStmt] at h
  cases hv : evalOp e t with
  | none => rw [hv] at h; exact absurd h (by simp)
  | some v =>
      rw [hv] at h
      injection h with h
      subst h
      rfl

/-- A successful run of a list of `assign` statements leaves memory unchanged. -/
private theorem stepStmts_assigns_mem :
    ∀ (l : List Stmt),
      (∀ st ∈ l, ∃ (dtype : TileDType) (shape : TileShape) (name : RegName)
        (e : Op dtype shape), st = Stmt.assign dtype shape name e) →
      ∀ t t' : BlockState, stepStmts l t = some t' →
        ∀ (r : RegionName) (o : Nat), t'.mem r o = t.mem r o := by
  intro l
  induction l with
  | nil =>
      intro _ t t' h r o
      simp only [stepStmts.nil, Option.some.injEq] at h
      subst h
      rfl
  | cons hd tl ih =>
      intro hall t t' h r o
      unfold stepStmts at h
      cases hhd : stepStmt hd t with
      | none => simp [hhd] at h
      | some tmid =>
          simp only [hhd] at h
          obtain ⟨dt, sh, nm, e, heq⟩ := hall hd (by simp)
          subst heq
          rw [ih (fun st hst => hall st (List.mem_cons_of_mem _ hst)) tmid t' h r o,
            stepStmt_assign_mem hhd r o]

/-- Every statement of the online Welford loop body is an `assign`. -/
private theorem onlineWelfordLoopBody_assigns (xReg : RegionName)
    (stride : Nat) :
    ∀ st ∈ onlineWelfordLoopBody xReg stride,
      ∃ (dtype : TileDType) (shape : TileShape) (name : RegName)
        (e : Op dtype shape), st = Stmt.assign dtype shape name e := by
  intro st hst
  simp only [onlineWelfordLoopBody, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩

/-! ### Rounding-degeneration plumbing

The online kernel's loop body carries no `castFloat`, so it steps identically
under `execR R` and `exec`; only the two boundary bf16 output stores differ. -/

/-- The online kernel's whole `forLoop` statement steps identically under
`execR R` and `exec`. -/
private theorem online_forLoop_castFree (R : RoundingModel)
    (xReg : RegionName) (blockSize : Nat) (t : BlockState) :
    stepStmtR R (.forLoop "i" blockSize (onlineWelfordLoopBody xReg rowStride)) t
      = stepStmt (.forLoop "i" blockSize (onlineWelfordLoopBody xReg rowStride)) t := by
  simp only [stepStmtR, stepStmt,
    stepForLoopAuxR_castFree R (onlineWelfordLoopBody xReg rowStride)
      (onlineWelfordLoopBody_castFree R xReg rowStride) "i" 0 blockSize t]

/-- The bf16-rounded scalar output-store cell: the boundary store double-rounds
its ℝ value `v` (the `.to(bf16)` cast, then the buffer `storeValue`). -/
private noncomputable def roundedCell (R : RoundingModel) (v : ℝ) : MemCell :=
  MemCell.of FloatDType.bf16.toTileDType
    (FloatDType.bf16.ofReal (R.storeValue FloatDType.bf16
      (R.cast FloatDType.real FloatDType.bf16 (some v))))

/-- Cell-level memory of the two-pass kernel under `execR R`: the var store wins
at `(varReg, 0)`, the mean store at `(meanReg, 0)` (both bf16-rounded), and every
other cell is untouched. -/
private theorem twopass_execR_mem (R : RoundingModel)
    (xReg meanReg varReg : RegionName) (blockSize : Nat)
    (s lhs' : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputRowLoadedAt s xReg rowStride blockSize xs)
    (h_mv : meanReg ≠ varReg)
    (hL : execR R (twopassWelfordKernel xReg meanReg varReg blockSize rowStride) s = some lhs')
    (r : RegionName) (o : Nat) :
    lhs'.mem r o =
      if r = varReg ∧ o = 0 then roundedCell R (twoPassS xs / blockSize)
      else if r = meanReg ∧ o = 0 then roundedCell R (twoPassMean xs)
      else s.mem r o := by
  simp [execR, twopassWelfordKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
    Tile.bop, Tile.natToReal, NumericDType.add,
    NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComputeExpr.toAlgorithm?, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    WithBot.realMul] at hL
  unfold InputRowLoadedAt at h_x
  subst hL
  by_cases hvar : r = varReg ∧ o = 0
  · obtain ⟨rfl, rfl⟩ := hvar
    simp [BlockState.writeMemAsR_mem, roundedCell, twoPassS, twoPassMean, h_x, pow_two]
    rfl
  · by_cases hmean : r = meanReg ∧ o = 0
    · obtain ⟨rfl, rfl⟩ := hmean
      simp [BlockState.writeMemAsR_mem, roundedCell, twoPassMean, h_mv, h_x]
      rfl
    · simp [BlockState.writeMemAsR_mem, hvar, hmean]

/-- Cell-level memory of the online kernel under `execR R`: same shape as
`twopass_execR_mem` but keyed on the Welford running values. -/
private theorem online_execR_mem (R : RoundingModel)
    (xReg meanReg varReg : RegionName) (blockSize : Nat)
    (s rhs' : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputRowLoadedAt s xReg rowStride blockSize xs)
    (h_mv : meanReg ≠ varReg)
    (hR : execR R (onlineWelfordKernel xReg meanReg varReg blockSize rowStride) s = some rhs')
    (r : RegionName) (o : Nat) :
    rhs'.mem r o =
      if r = varReg ∧ o = 0 then roundedCell R (welfordS xs blockSize / blockSize)
      else if r = meanReg ∧ o = 0 then roundedCell R (welfordMean xs blockSize)
      else s.mem r o := by
  let s0 :=
    ((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
      "M" .real [] (Tile.scalar (some 0))).setReg
      "S" .real [] (Tile.scalar (some 0))
  have h_init : P_welford rowStride xs xReg s.pid 0 s0 ∧ ∀ r o, s0.mem r o = s.mem r o := by
    refine ⟨?_, fun r o => rfl⟩
    simp [P_welford, s0, welfordMean, welfordS]
    exact ⟨rfl, h_x⟩
  obtain ⟨sLoop, hLoop, hPloop, hMemLoop⟩ :=
    forLoop_inv
      (idx := "i") (n := blockSize)
      (body := onlineWelfordLoopBody xReg rowStride)
      (P := fun k st => P_welford rowStride xs xReg s.pid k st ∧ ∀ r o, st.mem r o = s.mem r o)
      (s_init := s0)
      h_init
      (fun i st hi hP => by
        obtain ⟨hPw, hMem⟩ := hP
        obtain ⟨st', hstep, hPw'⟩ := online_welford_step rowStride xs xReg s.pid i st hi hPw
        refine ⟨st', hstep, hPw', fun r o => ?_⟩
        rw [stepStmts_assigns_mem _ (onlineWelfordLoopBody_assigns xReg rowStride)
          _ _ hstep r o]
        exact hMem r o)
  rcases hPloop with ⟨hM, hS, _hpidReg, _hpid, _hX⟩
  have hLoopR :
      stepForLoopAuxR R "i" 0 blockSize (onlineWelfordLoopBody xReg rowStride) s0
        = some sLoop := by
    rw [stepForLoopAuxR_castFree R (onlineWelfordLoopBody xReg rowStride)
      (onlineWelfordLoopBody_castFree R xReg rowStride) "i" 0 blockSize s0]
    simpa [stepForLoopAux.forLoop_unfold] using hLoop
  simp [execR, onlineWelfordKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
    Tile.bop, Tile.natToReal, ComputeExpr.toAlgorithm?] at hR
  rw [show
      (((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
        "M" .real [] (Tile.scalar (some 0))).setReg
        "S" .real [] (Tile.scalar (some 0))) = s0 from rfl] at hR
  simp only [onlineWelfordLoopBody] at hLoopR
  rw [hLoopR] at hR
  simp [NumericDType.div, hM, hS] at hR
  subst hR
  by_cases hvar : r = varReg ∧ o = 0
  · obtain ⟨rfl, rfl⟩ := hvar
    simp [BlockState.writeMemAsR_mem, roundedCell]
  · by_cases hmean : r = meanReg ∧ o = 0
    · obtain ⟨rfl, rfl⟩ := hmean
      simp [BlockState.writeMemAsR_mem, roundedCell, h_mv]
      rfl
    · simp [BlockState.writeMemAsR_mem, hvar, hmean, hMemLoop r o]

end Welford.lemmas

/-! ## The headline theorem -/
section Welford.theorems

include hN in
/-- **two-pass refines online** (`ComputeRefine.Refines R`, no scratch): for the
rounding model `R`, from the same initial state `twopassWelfordKernel` and
`onlineWelfordKernel` perform the same writes — their final memories agree at
every cell. Both compute the same per-lane ℝ mean/variance (Welford's identity
`welford_eq_two_pass`) and round it at the shared bf16 output stores. -/
specification welford_kernels_refinement_view
    (hin : TensorView.ViewsLoaded s
      [TensorView.slot (rowTileView s xReg rowStride blockSize) xs])
    (h_mv : meanReg ≠ varReg) :
    ComputeRefine.Refines R
      (twopassWelfordKernel xReg meanReg varReg blockSize rowStride)
      (onlineWelfordKernel xReg meanReg varReg blockSize rowStride)
      s [] := by
  obtain ⟨h_x', -⟩ := hin
  have h_x := inputRowLoadedAt_of_rowTileView_loaded h_x'
  obtain ⟨hMeanEq, hSEq⟩ := welford_eq_two_pass hN xs
  apply ComputeKernel.computeRefineR_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r _hr o
  rw [twopass_execR_mem rowStride R xReg meanReg varReg blockSize s lhs' xs h_x h_mv hL r o,
    online_execR_mem rowStride R xReg meanReg varReg blockSize s rhs' xs h_x h_mv hR r o]
  by_cases hvar : r = varReg ∧ o = 0
  · simp only [if_pos hvar]
    rw [show twoPassS xs / (blockSize : ℝ) = welfordS xs blockSize / (blockSize : ℝ) from by
      rw [hSEq]]
  · simp only [if_neg hvar]
    by_cases hmean : r = meanReg ∧ o = 0
    · simp only [if_pos hmean]
      rw [show twoPassMean xs = welfordMean xs blockSize from hMeanEq.symm]
    · simp only [if_neg hmean]

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in the public theorem's transitive proof.
#axiomsClean welford_kernels_refinement_view

-- (2) The headline is a *kernel-vs-kernel* refinement: its statement may mention
-- ONLY the two kernels, the loaded-input contract, the writes-equality surface,
-- and the state/region types — NO spec.
#stmtSurfaceSubset welford_kernels_refinement_view ⊆
  [twopassWelfordKernel, onlineWelfordKernel, TensorView.ViewsLoaded,
   TensorView.Slot, TensorView.slot, TensorView, VeriTile.Examples.rowTileView,
   VeriTile.Triton.InputAt,
   ComputeRefine.Refines, RoundingModel, BlockState, RegionName]

end Welford.theorems

/-! ## Exact (unrounded) originals — the dissolved `VeriTile.Examples.WelfordKernels`

Everything below is the former library example, dissolved into this showcase.
Its kernels are the exact-store originals: no `.to(tl.bfloat16)` boundary
rounding, and the input row lives at `pid * blockSize` (no separate `rowStride`
argument) — so they genuinely differ from the rounding-surface kernels above
and are kept, with their theorems, under the `Exact` sub-namespace. The shared
two-pass/recurrence Welford math is still single-sourced in
`VeriTile.Triton.TiledReduction.WelfordRec` (opened above). -/
section Welford.exact
namespace Exact

/-- Two-pass mean/variance kernel.

For each program id, this kernel loads one aligned `blockSize`-element row,
computes the mean with one `tl.sum`, then computes the population variance
with a second `tl.sum` over squared deviations. It writes one scalar mean and
one scalar variance for the row. In the current named-region model those
outputs are observed at offset `0`; the theorem below is for one fixed
`BlockState.pid`.

Source Triton (`.py` reference, aligned single-block flavour; assumes
`BLOCK_SIZE` equals the row length and uses no boundary mask):

```python
@triton.jit
def twopass_welford_kernel(x_ptr, mean_ptr, var_ptr,
                           BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offs)
    s_x = tl.sum(x, axis=0)
    mu = s_x / BLOCK_SIZE
    d = x - mu
    s_d2 = tl.sum(d * d, axis=0)
    v = s_d2 / BLOCK_SIZE

    tl.store(mean_ptr, mu)
    tl.store(var_ptr, v)
```
-/
def twopassWelfordKernel (xReg meanReg varReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange($(blockSize))
  x      := tl.load($(xReg) + offs)
  s_x    := tl.sum(x)
  μ      := s_x / tl.toReal($(blockSize))
  d      := x - μ
  s_d2   := tl.sum(d * d)
  v      := s_d2 / tl.toReal($(blockSize))
  tl.store($(meanReg), μ)
  tl.store($(varReg), v)
}

/-- Online Welford mean/variance kernel.

This kernel computes the same population mean/variance as
`Exact.twopassWelfordKernel`, but uses Welford's one-pass recurrence inside a
Triton `for` loop. The loop maintains scalar registers:

* `M`: running mean after the processed prefix
* `S`: running sum of squared deviations

After all `blockSize` elements, it writes `M` and `S / blockSize`. As above,
this is the aligned single-block version used by the proof; no boundary mask
is modeled in this example.

Source Triton (`.py` reference):

```python
@triton.jit
def online_welford_kernel(x_ptr, mean_ptr, var_ptr,
                          BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    M = 0.0
    S = 0.0

    for i in range(0, BLOCK_SIZE):
        xi = tl.load(x_ptr + (pid * BLOCK_SIZE + i))
        delta = xi - M
        M = M + delta / (i + 1)
        delta2 = xi - M
        S = S + delta * delta2

    tl.store(mean_ptr, M)
    tl.store(var_ptr, S / BLOCK_SIZE)
```
-/
def onlineWelfordKernel (xReg meanReg varReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  M   := 0
  S   := 0
  tl.for i in $(blockSize) {
    xi     := tl.load($(xReg) + (pid * $(blockSize) + i))
    delta  := xi - M
    M      := M + delta / (tl.toReal(i) + 1)
    delta2 := xi - M
    S      := S + delta * delta2
  }
  tl.store($(meanReg), M)
  tl.store($(varReg), S / tl.toReal($(blockSize)))
}

/-- Per-row mean spec. Thin alias for `Triton.TiledReduction.welfordMean`. -/
noncomputable def welfordMeanSpec {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  Triton.TiledReduction.welfordMean xs

/-- Per-row population variance spec. Thin alias for
`Triton.TiledReduction.welfordVar`. -/
noncomputable def welfordVarSpec {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  Triton.TiledReduction.welfordVar xs

theorem twopass_welford_correct
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (_hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs)
    (_h_mv : meanReg ≠ varReg) :
    let final := exec (twopassWelfordKernel xReg meanReg varReg blockSize) s
    final.bind (fun s' => some (s'.readMem meanReg 0))
        = some (welfordMeanSpec xs)
    ∧ final.bind (fun s' => some (s'.readMem varReg 0))
        = some (welfordVarSpec xs) := by
  constructor
  · simp [exec, twopassWelfordKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        Tile.natToReal, NumericDType.add,
        NumericDType.mul, NumericDType.sub, NumericDType.div,
        welfordMeanSpec, Triton.TiledReduction.welfordMean,
        Triton.TiledReduction.tileSum]
    unfold InputLoadedAt at _h_x
    simp_rw [_h_x]
    repeat unfold evalOp
    simp [Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      WithBot.realAdd, WithBot.realMul, WithBot.realSub, WithBot.realDiv, _h_mv]
    rfl
  · simp [exec, twopassWelfordKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        Tile.natToReal, NumericDType.add,
        NumericDType.mul, NumericDType.sub, NumericDType.div,
        welfordVarSpec, Triton.TiledReduction.welfordVar,
        Triton.TiledReduction.welfordSumSq,
        Triton.TiledReduction.welfordMean, Triton.TiledReduction.tileSum]
    unfold InputLoadedAt at _h_x
    simp_rw [_h_x]
    repeat unfold evalOp
    simp [Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      WithBot.realAdd, WithBot.realMul, WithBot.realSub, WithBot.realDiv, _h_mv, pow_two]
    rfl

/-- Loop invariant for `Exact.onlineWelfordKernel`: after `k` body iterations,
    register `M` holds `welfordMean xs k`, register `S` holds `welfordS xs k`,
    register `pid` holds the original program id, and the input region is
    unchanged. -/
def P_welford {N : Nat} (xs : Fin N → ℝ) (xReg : RegionName)
    (origPid : Nat) (k : Nat) (s : BlockState) : Prop :=
  s.regs .real [] "M" = some (Tile.scalar (welfordMean xs k))
  ∧ s.regs .real [] "S" = some (Tile.scalar (welfordS xs k))
  ∧ s.regs .nat [] "pid" = some (Tile.scalar origPid)
  ∧ s.pid = origPid
  ∧ InputLoadedAt s xReg N xs

theorem online_welford_step
    {N : Nat} (xs : Fin N → ℝ) (xReg : RegionName) (origPid i : Nat)
    (s : BlockState) (hi : i < N)
    (hP : P_welford xs xReg origPid i s) :
    ∃ s',
      stepStmts (onlineWelfordLoopBody xReg N)
        (s.setReg "i" .nat [] (Tile.scalar i)) = some s' ∧
      P_welford xs xReg origPid (i + 1) s' := by
  rcases hP with ⟨hM, hS, hpidReg, hpid, hX⟩
  let xi : ℝ := s.readMem xReg (origPid * N + i)
  have hxi : xi = xs ⟨i, hi⟩ := by
    have hx := hX ⟨i, hi⟩
    rw [hpid] at hx
    exact hx
  let m : ℝ := welfordMean xs i
  let ssum : ℝ := welfordS xs i
  let delta : ℝ := xi - m
  let m' : ℝ := m + delta / ((i : ℝ) + 1)
  let delta2 : ℝ := xi - m'
  let ssum' : ℝ := ssum + delta * delta2
  let s' :=
    (((((s.setReg "i" .nat [] (Tile.scalar i)).setReg
      "xi" .real [] (Tile.scalar xi)).setReg
      "delta" .real [] (Tile.scalar delta)).setReg
      "M" .real [] (Tile.scalar m')).setReg
      "delta2" .real [] (Tile.scalar delta2)).setReg
      "S" .real [] (Tile.scalar ssum')
  refine ⟨s', ?_, ?_⟩
  · -- Reduce loop body via simp; remaining goal is structural equality of
    -- WithBot ℝ arithmetic terms (↑a + ↑b vs ↑(a + b) etc.) which matches via
    -- WithBot.coe_add / coe_mul in reverse, plus rfl on the outer setReg shell.
    simp [onlineWelfordLoopBody, stepStmts, stepStmt, evalOp, Tile.bop,
      Tile.natToReal, NumericDType.add, NumericDType.mul, NumericDType.sub,
      NumericDType.div, BlockState.readMem, hM, hS, hpidReg,
      xi, m, ssum, delta, m', delta2, ssum', s',
      WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv,
      BlockState.setReg]
    rfl
  · simp [P_welford, s', InputLoadedAt, welfordMean, welfordS, hi, xi, m,
      ssum, delta, m', delta2, ssum', hpidReg, hpid, hxi]
    intro j
    have hx := hX j
    rw [hpid] at hx
    exact hx

theorem online_welford_correct
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputLoadedAt s xReg blockSize xs)
    (h_mv : meanReg ≠ varReg) :
    let final := exec (onlineWelfordKernel xReg meanReg varReg blockSize) s
    final.bind (fun s' => some (s'.readMem meanReg 0))
        = some (welfordMeanSpec xs)
    ∧ final.bind (fun s' => some (s'.readMem varReg 0))
        = some (welfordVarSpec xs) := by
  let s0 :=
    ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
      "M" .real [] (Tile.scalar 0)).setReg
      "S" .real [] (Tile.scalar 0)
  have h_init : P_welford xs xReg s.pid 0 s0 := by
    simp [P_welford, s0, welfordMean, welfordS]
    exact h_x
  obtain ⟨sLoop, hLoop, hPloop⟩ :=
    forLoop_inv
      (idx := "i") (n := blockSize)
      (body := onlineWelfordLoopBody xReg blockSize)
      (P := P_welford xs xReg s.pid)
      (s_init := s0)
      h_init
      (fun i st hi hP => online_welford_step xs xReg s.pid i st hi hP)
  rcases hPloop with ⟨hM, hS, _hpidReg, _hpid, _hX⟩
  obtain ⟨hMeanEq, hSEq⟩ := welford_eq_two_pass hN xs
  have hLoopAux :
      stepForLoopAux "i" 0 blockSize (onlineWelfordLoopBody xReg blockSize) s0 =
        some sLoop := by
    simpa [stepForLoopAux.forLoop_unfold] using hLoop
  have hLoopAuxExpanded :
      stepForLoopAux "i" 0 blockSize
        [Stmt.assign .real [] "xi"
            (Op.load .real (MemAccess.region xReg
              (Op.add .nat .nil
                (Op.mul .nat .nil (Op.ref .nat [] "pid")
                  (Op.constNat blockSize))
                (Op.ref .nat [] "i"))) MaskOpt.none),
          Stmt.assign .real [] "delta"
            (Op.sub .real .nil (Op.ref .real [] "xi")
              (Op.ref .real [] "M")),
          Stmt.assign .real [] "M"
            (Op.add .real .nil (Op.ref .real [] "M")
              (Op.div .real .nil (Op.ref .real [] "delta")
                (Op.add .real .nil (Op.ref .nat [] "i").natToReal
                  (Op.const 1)))),
          Stmt.assign .real [] "delta2"
            (Op.sub .real .nil (Op.ref .real [] "xi")
              (Op.ref .real [] "M")),
          Stmt.assign .real [] "S"
            (Op.add .real .nil (Op.ref .real [] "S")
              (Op.mul .real .nil (Op.ref .real [] "delta")
                (Op.ref .real [] "delta2")))]
        s0 = some sLoop := by
    simpa [onlineWelfordLoopBody] using hLoopAux
  have hExec :
      exec (onlineWelfordKernel xReg meanReg varReg blockSize) s =
        some ((sLoop.writeMem meanReg 0 (welfordMean xs blockSize)).writeMem
          varReg 0 (welfordS xs blockSize / blockSize)) := by
    -- Walk through pre-loop assigns, forLoop, post-loop stores explicitly.
    have hpid : stepStmt (.assign .nat [] "pid" (.programId 0)) s
                  = some (s.setReg "pid" .nat [] (Tile.scalar s.pid)) := by
      simp [stepStmt, evalOp]
    have hM0 : stepStmt (.assign .real [] "M" (.const 0))
                  (s.setReg "pid" .nat [] (Tile.scalar s.pid))
                = some ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
                    "M" .real [] (Tile.scalar 0)) := by
      simp [stepStmt, evalOp]
      rfl
    have hS0 : stepStmt (.assign .real [] "S" (.const 0))
                  ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
                    "M" .real [] (Tile.scalar 0))
                = some s0 := by
      simp [stepStmt, evalOp, s0]
      rfl
    have hMeanStore : stepStmt
        (.store .real [] (MemAccess.region meanReg (.constNat 0)) (.ref .real [] "M") MaskOpt.none) sLoop
        = some (sLoop.writeMem meanReg 0 (welfordMean xs blockSize)) := by
      simp [stepStmt, evalOp, hM]
    set sMean : BlockState := sLoop.writeMem meanReg 0 (welfordMean xs blockSize)
    have hSat_S : sMean.regs .real [] "S"
                  = some (Tile.scalar (welfordS xs blockSize)) := by
      show sLoop.regs .real [] "S" = _
      exact hS
    have hVarStore : stepStmt
        (.store .real [] (MemAccess.region varReg (.constNat 0))
            (.div .real .nil (.ref .real [] "S")
              (.natToReal (.constNat blockSize))) MaskOpt.none) sMean
        = some (sMean.writeMem varReg 0 (welfordS xs blockSize / blockSize)) := by
      simp [stepStmt, evalOp, hSat_S, Tile.bop, NumericDType.div, Tile.natToReal,
            WithBot.realDiv]
    -- Chain: 3 assigns → s0; forLoop → sLoop; 2 stores → sMean.writeMem ...
    show stepStmts (onlineWelfordKernel xReg meanReg varReg blockSize).body s = _
    show stepStmts
        [ .assign .nat [] "pid" (.programId 0)
        , .assign .real [] "M" (.const 0)
        , .assign .real [] "S" (.const 0)
        , .forLoop "i" blockSize (onlineWelfordLoopBody xReg blockSize)
        , .store .real [] (MemAccess.region meanReg (.constNat 0)) (.ref .real [] "M") MaskOpt.none
        , .store .real [] (MemAccess.region varReg (.constNat 0))
            (.div .real .nil (.ref .real [] "S")
              (.natToReal (.constNat blockSize))) MaskOpt.none
        ] s = _
    rw [stepStmts.cons_some hpid]
    rw [stepStmts.cons_some hM0]
    rw [stepStmts.cons_some hS0]
    rw [stepStmts.cons_some hLoop]
    rw [stepStmts.cons_some hMeanStore]
    rw [stepStmts.cons_some hVarStore]
    exact stepStmts.nil
  constructor
  · rw [hExec]
    simp [BlockState.readMem, BlockState.writeMem, h_mv, welfordMeanSpec,
      Triton.TiledReduction.welfordMean, Triton.TiledReduction.tileSum,
      twoPassMean, hMeanEq]
  · rw [hExec]
    simp [BlockState.readMem, BlockState.writeMem, welfordVarSpec,
      Triton.TiledReduction.welfordVar, Triton.TiledReduction.welfordSumSq,
      Triton.TiledReduction.welfordMean, Triton.TiledReduction.tileSum,
      twoPassS, twoPassMean, hSEq]

theorem welford_kernels_refinement
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputLoadedAt s xReg blockSize xs)
    (h_mv : meanReg ≠ varReg) :
    let final_2p := exec (twopassWelfordKernel xReg meanReg varReg blockSize) s
    let final_on := exec (onlineWelfordKernel xReg meanReg varReg blockSize) s
    final_2p.bind (fun s' => some (s'.readMem meanReg 0))
        = final_on.bind (fun s' => some (s'.readMem meanReg 0))
    ∧ final_2p.bind (fun s' => some (s'.readMem varReg 0))
        = final_on.bind (fun s' => some (s'.readMem varReg 0)) := by
  obtain ⟨h_2p_mean, h_2p_var⟩ :=
    twopass_welford_correct xReg meanReg varReg blockSize hN s xs h_x h_mv
  obtain ⟨h_on_mean, h_on_var⟩ :=
    online_welford_correct xReg meanReg varReg blockSize hN s xs h_x h_mv
  exact ⟨h_2p_mean.trans h_on_mean.symm, h_2p_var.trans h_on_var.symm⟩

/-- View-level surface for `Exact.welford_kernels_refinement`. -/
theorem welford_kernels_refinement_exec_view
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1))
    (h_mv : meanReg ≠ varReg) :
    let final_2p := exec (twopassWelfordKernel xReg meanReg varReg blockSize) s
    let final_on := exec (onlineWelfordKernel xReg meanReg varReg blockSize) s
    TensorView.observe final_2p (scalarCellView meanReg 0) PUnit.unit
        = TensorView.observe final_on (scalarCellView meanReg 0) PUnit.unit
    ∧ TensorView.observe final_2p (scalarCellView varReg 0) PUnit.unit
        = TensorView.observe final_on (scalarCellView varReg 0) PUnit.unit := by
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := blockSize) (xs := xs) h_x
  obtain ⟨hm, hv⟩ :=
    welford_kernels_refinement xReg meanReg varReg blockSize hN s xs hx h_mv
  constructor
  · cases h2p : exec (twopassWelfordKernel xReg meanReg varReg blockSize) s <;>
      cases hon : exec (onlineWelfordKernel xReg meanReg varReg blockSize) s <;>
      simp [TensorView.observe, observeTileAt, scalarCellView,
            TensorView.offset, Offset.strided, BlockState.readMem,
            h2p, hon] at hm ⊢
    exact hm
  · cases h2p : exec (twopassWelfordKernel xReg meanReg varReg blockSize) s <;>
      cases hon : exec (onlineWelfordKernel xReg meanReg varReg blockSize) s <;>
      simp [TensorView.observe, observeTileAt, scalarCellView,
            TensorView.offset, Offset.strided, BlockState.readMem,
            h2p, hon] at hv ⊢
    exact hv

end Exact

/-! ### Trust audit for the `Exact` section: no `sorry`, no smuggled axiom, in
any public result's transitive proof. -/

#axiomsClean Exact.twopass_welford_correct
#axiomsClean Exact.online_welford_correct
#axiomsClean Exact.welford_kernels_refinement
#axiomsClean Exact.welford_kernels_refinement_exec_view

end Welford.exact

end VeriTile.Bench.Examples.Welford

