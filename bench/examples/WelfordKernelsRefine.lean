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
# Welford variance: two-pass vs online — kernel equivalence `≡[R]`

Self-contained showcase, read top to bottom: **kernels** first (the two Welford
kernels), the **supporting lemmas** in the middle (`private` plumbing — the
loop invariant and the cell-level memory characterizations; the Welford
recurrence math itself stays single-sourced in
`VeriTile.Triton.TiledReduction.WelfordRec`), the region-level refinement core
next (`welford_kernels_refinement_view`), then the **flat-memory bridge side
conditions**, the **specification** last (one public headline `welford_equiv`
on the `≡[R]` surface), and a compile-time **trust audit**. The real sections
below are `Welford.kernels`, `Welford.lemmas`, `Welford.theorems`,
`Welford.bridge`, `Welford.spec`.

`twopassWelfordKernel` computes the row mean/variance with two `tl.sum` passes;
`onlineWelfordKernel` uses Welford's one-pass recurrence inside a `for` loop.
Like a real Triton kernel, both take a **`rowStride` argument** (stage 1 of
the address-layout roadmap): the input row lives at `pid * rowStride + [0,
blockSize)`.
Both **store the mean and variance rounded to bf16** (`(·).to(tl.bfloat16)`); the
reductions run in ℝ (no intermediate rounding). The load-bearing identity is that
after processing all inputs the running `(M, S)` equals the two-pass `(μ, S)`
(`welford_eq_two_pass`), so from the same state both kernels produce the same ℝ
mean and variance, and the only rounding is the shared bf16 output stores, which
quantize equal values identically.

## The public result (bottom of file)

The single public headline is **`welford_equiv`** — kernel equivalence on the
shared one-input / **two-output** IO signature:

    welfordTwoPassIO blockSize rowStride ≡[R] welfordOnlineIO blockSize rowStride

`≡[R]` is the audit-once kernel-equivalence combinator (`KernelIO₁ₓ₂.Equiv`,
`VeriTile.Triton.Memory.KernelSpec`), the `⊨`-grade form of the refinement
surface, here in its one-input / two-output flavour: both kernels read the
input row `[pid * rowStride, pid * rowStride + blockSize)` from `x` and write
TWO one-cell outputs — the scalar mean at `mean[0]` and the scalar variance at
`var[0]` (the stores are plain `tl.store(reg, …)` with no offset expression,
so `write1 = write2 = fun _ => 0` and `Bout1 = Bout2 = 1`). Spelled out, the
headline says: for **every** rounding model `R`, **every** disjoint flat
placement of the three interface buffers `x`/`mean`/`var` (∀ base pointers,
∀ buffer sizes), **every** program id whose row window and output cells land
inside their buffers, and **every** launch state — **no input hypotheses at
all**, not even loaded inputs: "equal inputs" is simply "the same `s₀`" —
both translated pointer kernels terminate under `execR R`, **both output
cells agree**, and each kernel leaves every cell outside the **union of the
two one-cell output windows** `{mean[0], var[0]}` untouched (neither kernel
stages anything through memory, so both scratch lists are empty and the
scratch leg of the frame is vacuous).

The statement mentions only the two IO signatures and the library equivalence
surface — **no spec** (the `#stmtSurfaceSubset` gate below enforces this).
Everything else — the loop invariant and the per-kernel cell-level memory
characterizations that `hrun` reads **directly**, at the two output cells
(value agreement, via `welford_eq_two_pass`) and everywhere else (the
one-cell frames) — is scaffolding. The same characterizations also power the
region-level refinement `welford_kernels_refinement_view`, which is **not**
on the headline's proof path; it is kept (and audited) as the named
`ComputeRefine.Refines` region-level story. The headline carries **no
positivity hypothesis**: at `blockSize = 0` both recurrences degenerate to
`0` (`welford_eq_two_pass_total`). The compositional pattern is
`bench/examples/FusedSwiglu.lean` (the `≡[R]` pilot).

## The exact-ℝ surface

There is no separate exact layer: the headline quantifies over **every**
rounding model, and at `R := .triv` the `castTo` stores are inert
(`execR_triv`), so the exact-ℝ equivalence is the headline's degeneration —
no raw-store twin kernels needed.
-/

namespace VeriTile.Bench.Examples.Welford

open VeriTile.Triton
open scoped VeriTile.Triton.KernelIO₁ₓ₂

open VeriTile.Triton.TiledReduction.WelfordRec
open VeriTile.Examples (InputRowLoadedAt inputLoadedAt_of_programTileView_loaded programTileView rowTileView inputRowLoadedAt_of_rowTileView_loaded
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

/- Interface variables for the sections below — declared **once**, at namespace
scope, so `Welford.lemmas`, `Welford.theorems` and `Welford.bridge` can all
draw on them (`end <section>` only clears variables declared *inside* that
section). A declaration auto-includes only the variables it mentions; the
`{N}`-generic plumbing re-binds its own copies where a more general signature
is needed, and per-declaration hypotheses (the `InputRowLoadedAt` /
`ViewsLoaded` input contracts, the `meanReg ≠ varReg` aliasing constraint)
stay on the signatures that need them. -/
variable (xReg meanReg varReg : RegionName) (blockSize rowStride : Nat) (hN : 0 < blockSize)
variable (s : BlockState) (xs : Fin blockSize → ℝ) (R : RoundingModel)

/-! ## Supporting lemmas (private plumbing) -/
section Welford.lemmas

-- Welford two-pass/recurrence math (twoPassMean/twoPassS/welfordMean/welfordS +
-- welford_eq_two_pass) now lives in `VeriTile.Triton.TiledReduction.WelfordRec`,
-- opened above; shared with the `FusedLayerNorm` showcase.
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
under `execR R` and `exec` (`stepForLoopAuxR_castFree` below); only the two
boundary bf16 output stores differ. -/

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

/-! ### Totalized Welford identity and termination (for the `≡[R]` contract)

`≡[R]`'s region-model obligation starts from **any** state and carries no
side hypotheses, so (a) the Welford identity is extended to `blockSize = 0`
(where both recurrences are literally `0`), and (b) termination of both
kernels is established outright — the two-pass body is straight-line and
total, and the online loop runs to completion via the same `P_welford`
invariant, seeded with whatever input row the state happens to hold. -/

/-- `welford_eq_two_pass`, extended to `n = 0`: with an empty row the
recurrences and the two-pass forms all degenerate to `0`, so the identity
needs no positivity hypothesis. (A trivial case split, not new Welford math —
the `n > 0` case is the single-sourced library lemma.) -/
private theorem welford_eq_two_pass_total {n : Nat} (x : Fin n → ℝ) :
    welfordMean x n = twoPassMean x ∧ welfordS x n = twoPassS x := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · exact ⟨by simp [welfordMean, twoPassMean], by simp [welfordS, twoPassS]⟩
  · exact welford_eq_two_pass hn x

/-- The two-pass kernel terminates under `execR R` from **every** state: its
body is a straight-line statement list whose every step is total, so
`execR R` computes to `some _` by the same computational unfold the memory
characterization uses. -/
private theorem twopass_execR_isSome :
    ∃ s', execR R ((twopassWelfordKernel xReg meanReg varReg blockSize
      rowStride).toAlgKernel) s = some s' := by
  cases h : execR R ((twopassWelfordKernel xReg meanReg varReg blockSize
      rowStride).toAlgKernel) s with
  | some s' => exact ⟨s', rfl⟩
  | none =>
      -- impossible: the straight-line body computes to `some _`
      simp [execR, twopassWelfordKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.natToReal, NumericDType.add,
        NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComputeExpr.toAlgorithm?, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        WithBot.realMul] at h

/-- The online kernel terminates under `execR R` from every state (the
`InputRowLoadedAt` seed is trivially available at the use site: whatever the
state holds). The `forLoop` runs to completion via the `P_welford` invariant;
the pre-loop assigns and the two boundary stores are total. -/
private theorem online_execR_isSome (R : RoundingModel)
    (xReg meanReg varReg : RegionName) (blockSize : Nat)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputRowLoadedAt s xReg rowStride blockSize xs) :
    ∃ s', execR R ((onlineWelfordKernel xReg meanReg varReg blockSize
      rowStride).toAlgKernel) s = some s' := by
  let s0 :=
    ((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
      "M" .real [] (Tile.scalar (some 0))).setReg
      "S" .real [] (Tile.scalar (some 0))
  have h_init : P_welford rowStride xs xReg s.pid 0 s0 := by
    simp [P_welford, s0, welfordMean, welfordS]
    exact ⟨rfl, h_x⟩
  obtain ⟨sLoop, hLoop, hPloop⟩ :=
    forLoop_inv
      (idx := "i") (n := blockSize)
      (body := onlineWelfordLoopBody xReg rowStride)
      (P := P_welford rowStride xs xReg s.pid)
      (s_init := s0)
      h_init
      (fun i st hi hP => online_welford_step rowStride xs xReg s.pid i st hi hP)
  rcases hPloop with ⟨hM, hS, _hpidReg, _hpid, _hX⟩
  have hLoopR :
      stepForLoopAuxR R "i" 0 blockSize (onlineWelfordLoopBody xReg rowStride) s0
        = some sLoop := by
    rw [stepForLoopAuxR_castFree R (onlineWelfordLoopBody xReg rowStride)
      (onlineWelfordLoopBody_castFree R xReg rowStride) "i" 0 blockSize s0]
    simpa [stepForLoopAux.forLoop_unfold] using hLoop
  rw [← Option.isSome_iff_exists]
  simp [execR, onlineWelfordKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
    Tile.bop, Tile.natToReal, ComputeExpr.toAlgorithm?]
  rw [show
      (((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
        "M" .real [] (Tile.scalar (some 0))).setReg
        "S" .real [] (Tile.scalar (some 0))) = s0 from rfl]
  simp only [onlineWelfordLoopBody] at hLoopR
  rw [hLoopR]
  simp [NumericDType.div, hM, hS]

end Welford.lemmas

/-! ## The headline theorem -/
section Welford.theorems

include hN in
/-- **two-pass refines online** (`ComputeRefine.Refines R`, no scratch): for the
rounding model `R`, from the same initial state `twopassWelfordKernel` and
`onlineWelfordKernel` perform the same writes — their final memories agree at
every cell. Both compute the same per-lane ℝ mean/variance (Welford's identity
`welford_eq_two_pass`) and round it at the shared bf16 output stores.

Formerly the file's headline; now the named region-level story. The `≡[R]`
specification's region-model obligation does **not** route through this
statement — it reads the same per-kernel memory characterizations directly
(value agreement at the two output cells + the one-cell frames): `Refines`
compares only the two final memories, so it cannot supply the per-kernel
frames against the initial state, and this view carries the `0 < blockSize` /
loaded-row hypotheses the total obligation avoids. -/
theorem welford_kernels_refinement_view
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

end Welford.theorems

/-! ## Flat-memory bridge side conditions

Both kernels are register-indirect on the input side (the two-pass kernel
loads through the `offs` register, the online kernel's loop loads through
`pid * rowStride + i`), so no ∀-state safety contract covers them; the
flat-memory bridge (v1.2, `execR` flavor) takes the per-execution
`Kernel.TraceSafeR` contract instead, plus `FlattenOk` (bridge fragment
membership). The input bounds obligation is the whole row window
`pid * rowStride + blockSize ≤ bounds x`; the two scalar output stores hit
the single cell `0` of `mean` / `var`, so their obligations degenerate to
`0 < bounds mean` / `0 < bounds var`. The two-pass kernel is straight-line
(one `cons_intro` walk); the online kernel's Welford `forLoop` is walked
iteration-by-iteration (`welfordLoop_traceSafeR`), the same recipe as the
fused-LayerNorm showcase. -/
section Welford.bridge

/-- Inversion for a successful `assign` step under `R`. -/
private theorem stepStmtR_assign_inv {R : RoundingModel} {d : TileDType}
    {sh : TileShape} {nm : RegName} {e : Op d sh} {t t' : BlockState}
    (h : stepStmtR R (.assign d sh nm e) t = some t') :
    ∃ v, evalOpR R e t = some v ∧ t' = t.setReg nm d sh v := by
  simp only [stepStmtR] at h
  cases hv : evalOpR R e t with
  | none => rw [hv] at h; exact absurd h (by simp)
  | some v =>
      rw [hv] at h
      replace h : some (t.setReg nm d sh v) = some t' := h
      exact ⟨v, rfl, (Option.some_inj.mp h).symm⟩

/-- Inversion for a successful statement-list step at a cons. -/
private theorem stepStmtsR_cons_inv {R : RoundingModel} {st : Stmt}
    {rest : List Stmt} {t t' : BlockState}
    (h : stepStmtsR R (st :: rest) t = some t') :
    ∃ t1, stepStmtR R st t = some t1 ∧ stepStmtsR R rest t1 = some t' := by
  rw [stepStmtsR] at h
  cases h1 : stepStmtR R st t with
  | none => rw [h1] at h; exact absurd h (by simp)
  | some t1 => rw [h1] at h; exact ⟨t1, rfl, h⟩

private theorem stepStmtsR_nil_inv {R : RoundingModel} {t t' : BlockState}
    (h : stepStmtsR R [] t = some t') : t' = t := by
  rw [stepStmtsR] at h
  exact (Option.some_inj.mp h).symm

/-- Bounds discharge for the two-pass row load through the `offs` register:
any state whose `offs` register holds the row offsets `pid₀ * rowStride + i`
addresses below `pid₀ * rowStride + blockSize`. -/
private theorem offs_activeAddressSafeR (R : RoundingModel)
    (bounds : RegionBounds) (pid₀ : Nat) (t : BlockState)
    (active : TileIndex [blockSize] → Prop)
    (hread : t.regs .nat [blockSize] "offs"
      = some ⟨fun i => pid₀ * rowStride + i.1.val⟩)
    (reg : RegionName) (hreg : pid₀ * rowStride + blockSize ≤ bounds reg) :
    memAccessActiveAddressSafeR R bounds
      (MemAccess.region reg (Op.ref .nat [blockSize] "offs")) t active := by
  simp only [memAccessActiveAddressSafeR]
  intro offsets hoffs i _
  rw [show evalOpR R (Op.ref .nat [blockSize] "offs") t
      = some ⟨fun i => pid₀ * rowStride + i.1.val⟩ from by
    simp [evalOpR.eq_def, hread]] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact lt_of_lt_of_le (Nat.add_lt_add_left i.1.isLt _) hreg

/-- Bounds discharge for the scalar output stores at constant offset `0`. -/
private theorem const0_activeAddressSafeR (R : RoundingModel)
    (bounds : RegionBounds) (t : BlockState)
    (active : TileIndex [] → Prop)
    (reg : RegionName) (hreg : 0 < bounds reg) :
    memAccessActiveAddressSafeR R bounds
      (MemAccess.region reg (Op.constNat 0)) t active := by
  simp only [memAccessActiveAddressSafeR]
  intro offsets hoffs i _
  rw [show evalOpR R (Op.constNat 0) t = some (Tile.scalar 0) from by
    simp [evalOpR.eq_def]] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact hreg

/-- The two-pass kernel sits inside the bridge's covered fragment. -/
theorem twopass_flattenOk :
    ((twopassWelfordKernel xReg meanReg varReg blockSize rowStride
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [twopassWelfordKernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

/-- The online kernel (Welford `forLoop` included) sits inside the bridge's
covered fragment. -/
theorem online_flattenOk :
    ((onlineWelfordKernel xReg meanReg varReg blockSize rowStride
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [onlineWelfordKernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk for the two-pass kernel under `R`: the row load
through `offs` stays below `bounds x` along the actual execution; the two
scalar stores hit cell `0` of `mean` / `var`. -/
theorem twopass_traceSafeR (bounds : RegionBounds)
    (hxb : s.pids 0 * rowStride + blockSize ≤ bounds xReg)
    (hmb : 0 < bounds meanReg) (hvb : 0 < bounds varReg) :
    Kernel.TraceSafeR R bounds
      ((twopassWelfordKernel xReg meanReg varReg blockSize rowStride
        ).toAlgKernel) s := by
  unfold Kernel.TraceSafeR
  -- pid := program_id(0)
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmtR_assign_inv hs1
  rw [show evalOpR R (Op.programId 0) s = some (Tile.scalar (s.pids 0)) from by
    simp [evalOpR.eq_def]] at hv1
  obtain rfl := Option.some_inj.mp hv1
  -- offs := pid * rowStride + arange blockSize
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmtR_assign_inv hs2
  rw [show evalOpR R (Op.add .nat .scalarL
      (Op.mul .nat .nil (Op.ref .nat [] "pid") (Op.constNat rowStride))
      (Op.arange blockSize)) (s.setReg "pid" .nat [] (Tile.scalar (s.pids 0)))
      = some ⟨fun i => s.pids 0 * rowStride + i.1.val⟩ from by
    simp [evalOpR.eq_def, Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
      Tile.vec, BlockState.setReg]] at hv2
  obtain rfl := Option.some_inj.mp hv2
  -- x := load(xReg + offs)
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR]
    exact ⟨trivial, trivial,
      offs_activeAddressSafeR blockSize rowStride R bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) xReg hxb⟩
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmtR_assign_inv hs3
  -- s_x := sum(x)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmtR_assign_inv hs4
  -- μ := s_x / blockSize
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmtR_assign_inv hs5
  -- d := x - μ
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s6 hs6
  obtain ⟨v6, hv6, rfl⟩ := stepStmtR_assign_inv hs6
  -- s_d2 := sum(d * d)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s7 hs7
  obtain ⟨v7, hv7, rfl⟩ := stepStmtR_assign_inv hs7
  -- v := s_d2 / blockSize
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s8 hs8
  obtain ⟨v8, hv8, rfl⟩ := stepStmtR_assign_inv hs8
  -- store(meanReg, μ.to(bf16))
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR, Op.SafeAtR]
    exact ⟨trivial, by simp [Op.SafeAtR.eq_def], trivial,
      const0_activeAddressSafeR R bounds _ _ (Region.cast meanReg)
        (by simpa [Region.cast_self] using hmb)⟩
  intro s9 hs9
  -- store(varReg, v.to(bf16))
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR, Op.SafeAtR]
  exact ⟨trivial, by simp [Op.SafeAtR.eq_def], trivial,
    const0_activeAddressSafeR R bounds _ _ (Region.cast varReg)
      (by simpa [Region.cast_self] using hvb)⟩

/-- One Welford iteration is trace-safe when the `pid` register is pinned:
its single scalar load reads `pid₀ * rowStride + k < pid₀ * rowStride +
blockSize`. -/
private theorem welfordBody_traceSafeR (xReg : RegionName)
    (blockSize rowStride : Nat)
    (R : RoundingModel) (bounds : RegionBounds) (pid₀ k : Nat)
    (hk : k < blockSize)
    (hxb : pid₀ * rowStride + blockSize ≤ bounds xReg)
    (t : BlockState) (hpid : t.regs .nat [] "pid" = some (Tile.scalar pid₀)) :
    Stmt.TraceSafeListR R bounds (onlineWelfordLoopBody xReg rowStride)
      (t.setReg "i" .nat [] (Tile.scalar k)) := by
  unfold onlineWelfordLoopBody
  -- xi := load(x + (pid * rowStride + i))
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR]
    refine ⟨by simp, trivial, ?_⟩
    simp only [memAccessActiveAddressSafeR]
    intro offsets hoffs i _
    rw [show evalOpR R (Op.add .nat .nil
        (Op.mul .nat .nil (Op.ref .nat [] "pid") (Op.constNat rowStride))
        (Op.ref .nat [] "i")) (t.setReg "i" .nat [] (Tile.scalar k))
        = some (Tile.scalar (pid₀ * rowStride + k)) from by
      simp [evalOpR.eq_def, Tile.bop, NumericDType.nat_add,
        NumericDType.nat_mul, BlockState.setReg, hpid]] at hoffs
    obtain rfl := Option.some_inj.mp hoffs
    simp only [Region.cast_self]
    exact lt_of_lt_of_le (Nat.add_lt_add_left hk _) hxb
  intro t1 ht1
  obtain ⟨w1, hw1, rfl⟩ := stepStmtR_assign_inv ht1
  -- delta := xi - M
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro t2 ht2
  obtain ⟨w2, hw2, rfl⟩ := stepStmtR_assign_inv ht2
  -- M := M + delta / (toReal i + 1)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro t3 ht3
  obtain ⟨w3, hw3, rfl⟩ := stepStmtR_assign_inv ht3
  -- delta2 := xi - M
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro t4 ht4
  obtain ⟨w4, hw4, rfl⟩ := stepStmtR_assign_inv ht4
  -- S := S + delta * delta2
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun _ _ => .nil_intro)

/-- The Welford body writes only `xi`/`delta`/`M`/`delta2`/`S`: the `pid`
register survives one iteration. -/
private theorem welfordBody_pid_preserved (xReg : RegionName) (rowStride : Nat)
    {R : RoundingModel} {t t' : BlockState}
    (h : stepStmtsR R (onlineWelfordLoopBody xReg rowStride) t = some t') :
    t'.regs .nat [] "pid" = t.regs .nat [] "pid" := by
  unfold onlineWelfordLoopBody at h
  obtain ⟨t1, h1, h⟩ := stepStmtsR_cons_inv h
  obtain ⟨w1, _, rfl⟩ := stepStmtR_assign_inv h1
  obtain ⟨t2, h2, h⟩ := stepStmtsR_cons_inv h
  obtain ⟨w2, _, rfl⟩ := stepStmtR_assign_inv h2
  obtain ⟨t3, h3, h⟩ := stepStmtsR_cons_inv h
  obtain ⟨w3, _, rfl⟩ := stepStmtR_assign_inv h3
  obtain ⟨t4, h4, h⟩ := stepStmtsR_cons_inv h
  obtain ⟨w4, _, rfl⟩ := stepStmtR_assign_inv h4
  obtain ⟨t5, h5, h⟩ := stepStmtsR_cons_inv h
  obtain ⟨w5, _, rfl⟩ := stepStmtR_assign_inv h5
  obtain rfl := stepStmtsR_nil_inv h
  simp [BlockState.setReg]

/-- The whole Welford loop is trace-safe from any `pid`-pinned state. -/
private theorem welfordLoop_traceSafeR (xReg : RegionName)
    (blockSize rowStride : Nat)
    (R : RoundingModel) (bounds : RegionBounds) (pid₀ : Nat)
    (hxb : pid₀ * rowStride + blockSize ≤ bounds xReg) :
    ∀ (fuel k : Nat) (t : BlockState), blockSize - k ≤ fuel →
      t.regs .nat [] "pid" = some (Tile.scalar pid₀) →
      Stmt.forLoopTraceSafeR R bounds "i" k blockSize
        (onlineWelfordLoopBody xReg rowStride) t
  | 0, k, t, hf, hpid => by
      rw [Stmt.forLoopTraceSafeR]
      rw [if_neg (by omega)]
      trivial
  | fuel + 1, k, t, hf, hpid => by
      rw [Stmt.forLoopTraceSafeR]
      split
      next hlt =>
        refine ⟨welfordBody_traceSafeR xReg blockSize rowStride R bounds pid₀ k
          hlt hxb t hpid, ?_⟩
        split
        next s' hs' =>
          refine welfordLoop_traceSafeR xReg blockSize rowStride R bounds pid₀
            hxb fuel (k + 1) s' (by omega) ?_
          rw [welfordBody_pid_preserved xReg rowStride hs']
          simp [BlockState.setReg, hpid]
        next => trivial
      next => trivial

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk for the online kernel under `R`: the Welford
loop's `blockSize` scalar loads stay below `bounds x` (walked
iteration-by-iteration), and the two scalar stores hit cell `0` of
`mean` / `var`. -/
theorem online_traceSafeR (bounds : RegionBounds)
    (hxb : s.pids 0 * rowStride + blockSize ≤ bounds xReg)
    (hmb : 0 < bounds meanReg) (hvb : 0 < bounds varReg) :
    Kernel.TraceSafeR R bounds
      ((onlineWelfordKernel xReg meanReg varReg blockSize rowStride
        ).toAlgKernel) s := by
  unfold Kernel.TraceSafeR
  -- pid := program_id(0)
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmtR_assign_inv hs1
  rw [show evalOpR R (Op.programId 0) s = some (Tile.scalar (s.pids 0)) from by
    simp [evalOpR.eq_def]] at hv1
  obtain rfl := Option.some_inj.mp hv1
  -- M := 0
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmtR_assign_inv hs2
  -- S := 0
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmtR_assign_inv hs3
  -- for i in blockSize { … Welford … }
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR]
    exact welfordLoop_traceSafeR xReg blockSize rowStride R bounds (s.pids 0)
      hxb blockSize 0 _ (by omega) (by simp [BlockState.setReg])
  intro s4 hs4
  -- store(meanReg, M.to(bf16))
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR, Op.SafeAtR]
    exact ⟨trivial, by simp [Op.SafeAtR.eq_def], trivial,
      const0_activeAddressSafeR R bounds _ _ (Region.cast meanReg)
        (by simpa [Region.cast_self] using hmb)⟩
  intro s5 hs5
  -- store(varReg, (S / blockSize).to(bf16))
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR, Op.SafeAtR]
  exact ⟨trivial, by simp [Op.SafeAtR.eq_def], trivial,
    const0_activeAddressSafeR R bounds _ _ (Region.cast varReg)
      (by simpa [Region.cast_self] using hvb)⟩

end Welford.bridge

/-! ## The spec: `welfordTwoPassIO ≡[R] welfordOnlineIO` -/
section Welford.spec

/-- The two-pass kernel's one-input / two-output **IO signature** — the whole
kernel-specific audit surface of the headline: one input row `x` read at
`[pid * rowStride, pid * rowStride + blockSize)`, and TWO one-cell outputs —
the scalar mean at `mean[0]` and the scalar variance at `var[0]` (the
kernels' stores are plain `tl.store(reg, …)` with no offset expression, hence
`write1 = write2 = fun _ => 0` and `Bout1 = Bout2 = 1`). Neither kernel
stages anything through memory: no scratch. -/
def welfordTwoPassIO (blockSize rowStride : Nat) : KernelIO₁ₓ₂ where
  kernel := twopassWelfordKernel ⟨"x"⟩ ⟨"mean"⟩ ⟨"var"⟩ blockSize rowStride
  inp := ⟨"x"⟩
  out1 := ⟨"mean"⟩
  out2 := ⟨"var"⟩
  Bin := blockSize
  Bout1 := 1
  Bout2 := 1
  read := fun pid => pid * rowStride
  write1 := fun _ => 0
  write2 := fun _ => 0

/-- The online kernel's IO signature: the **same interface by construction**
(structure update of `welfordTwoPassIO` — buffers and windows shared
verbatim), with the online kernel plugged in. -/
def welfordOnlineIO (blockSize rowStride : Nat) : KernelIO₁ₓ₂ :=
  { welfordTwoPassIO blockSize rowStride with
    kernel := onlineWelfordKernel ⟨"x"⟩ ⟨"mean"⟩ ⟨"var"⟩ blockSize rowStride }

/-- **The headline**: the two-pass Welford kernel is equivalent to the online
(one-pass recurrence) kernel on their shared one-input / two-output IO
signature, for **every** rounding model and **every** `blockSize`/`rowStride`
— see the module docstring for the full contract `≡[R]` unfolds to (both
kernels terminate from any state, the mean and variance cells agree, each
frames outside the two one-cell output windows). Proof:
`KernelIO₁ₓ₂.Equiv.intro` assembles the region-model equivalence —
termination is unconditional, the two value-agreement legs are the per-kernel
memory characterizations glued by Welford's identity (`welford_eq_two_pass`,
totalized at `blockSize = 0`), the frames are those same characterizations
read everywhere outside the two output cells — with the flat-memory bridge side conditions
(`FlattenOk` + `TraceSafeR` per kernel). -/
specification welford_equiv (R : RoundingModel) (blockSize rowStride : Nat) :
    welfordTwoPassIO blockSize rowStride ≡[R] welfordOnlineIO blockSize rowStride := by
  refine KernelIO₁ₓ₂.Equiv.intro _ _ ?_ ?_ ?_ ?_ ?_
  · -- FlattenOk, two-pass
    exact twopass_flattenOk ⟨"x"⟩ ⟨"mean"⟩ ⟨"var"⟩ blockSize rowStride
  · -- FlattenOk, online
    exact online_flattenOk ⟨"x"⟩ ⟨"mean"⟩ ⟨"var"⟩ blockSize rowStride
  · -- TraceSafeR, two-pass
    intro bounds t h1 h2 h3 _
    simp only [welfordTwoPassIO] at h1 h2 h3
    exact twopass_traceSafeR ⟨"x"⟩ ⟨"mean"⟩ ⟨"var"⟩ blockSize rowStride t R
      bounds (by simpa using h1) (by omega) (by omega)
  · -- TraceSafeR, online
    intro bounds t h1 h2 h3 _
    simp only [welfordTwoPassIO] at h1 h2 h3
    exact online_traceSafeR ⟨"x"⟩ ⟨"mean"⟩ ⟨"var"⟩ blockSize rowStride t R
      bounds (by simpa using h1) (by omega) (by omega)
  · -- the region-model equivalence, from ANY state s₀ (no input hypotheses)
    intro s₀
    simp only [welfordTwoPassIO, welfordOnlineIO]
    -- the input row both kernels actually see: whatever s₀ holds
    have h_x : InputRowLoadedAt s₀ ⟨"x"⟩ rowStride blockSize
        (fun j => s₀.readMem ⟨"x"⟩ (s₀.pid * rowStride + j.val)) :=
      fun _ => rfl
    have h_mv : (⟨"mean"⟩ : RegionName) ≠ ⟨"var"⟩ := by decide
    -- termination of both sides
    obtain ⟨s1, hexec1⟩ :=
      twopass_execR_isSome ⟨"x"⟩ ⟨"mean"⟩ ⟨"var"⟩ blockSize rowStride s₀ R
    obtain ⟨s2, hexec2⟩ :=
      online_execR_isSome rowStride R ⟨"x"⟩ ⟨"mean"⟩ ⟨"var"⟩ blockSize s₀ _ h_x
    -- the per-kernel cell-level memory characterizations
    have hmem1 := twopass_execR_mem rowStride R ⟨"x"⟩ ⟨"mean"⟩ ⟨"var"⟩
      blockSize s₀ s1 _ h_x h_mv hexec1
    have hmem2 := online_execR_mem rowStride R ⟨"x"⟩ ⟨"mean"⟩ ⟨"var"⟩
      blockSize s₀ s2 _ h_x h_mv hexec2
    -- Welford's identity, totalized (no `0 < blockSize` needed)
    obtain ⟨hMeanEq, hSEq⟩ := welford_eq_two_pass_total (n := blockSize)
      (fun j => s₀.readMem ⟨"x"⟩ (s₀.pid * rowStride + j.val))
    refine ⟨s1, s2, hexec1, hexec2, ?_, ?_, ?_, ?_⟩
    · -- the mean cells agree
      intro j
      have hj : (0 : Nat) + j.val = 0 := by have := j.isLt; omega
      rw [hj]
      unfold BlockState.readMem
      rw [hmem1 ⟨"mean"⟩ 0, hmem2 ⟨"mean"⟩ 0]
      simp [h_mv, hMeanEq]
    · -- the variance cells agree
      intro j
      have hj : (0 : Nat) + j.val = 0 := by have := j.isLt; omega
      rw [hj]
      unfold BlockState.readMem
      rw [hmem1 ⟨"var"⟩ 0, hmem2 ⟨"var"⟩ 0]
      simp [hSEq]
    · -- two-pass frame: writes only {mean[0], var[0]} (no scratch)
      intro r o hc1 hc2 _
      rw [hmem1 r o]
      have hnv : ¬(r = (⟨"var"⟩ : RegionName) ∧ o = 0) := by
        rintro ⟨rfl, rfl⟩
        rcases hc2 with h | h
        · exact h rfl
        · exact h ⟨0, Nat.zero_lt_one⟩ rfl
      have hnm : ¬(r = (⟨"mean"⟩ : RegionName) ∧ o = 0) := by
        rintro ⟨rfl, rfl⟩
        rcases hc1 with h | h
        · exact h rfl
        · exact h ⟨0, Nat.zero_lt_one⟩ rfl
      simp [hnv, hnm]
    · -- online frame: identical shape (no scratch either)
      intro r o hc1 hc2 _
      rw [hmem2 r o]
      have hnv : ¬(r = (⟨"var"⟩ : RegionName) ∧ o = 0) := by
        rintro ⟨rfl, rfl⟩
        rcases hc2 with h | h
        · exact h rfl
        · exact h ⟨0, Nat.zero_lt_one⟩ rfl
      have hnm : ¬(r = (⟨"mean"⟩ : RegionName) ∧ o = 0) := by
        rintro ⟨rfl, rfl⟩
        rcases hc1 with h | h
        · exact h rfl
        · exact h ⟨0, Nat.zero_lt_one⟩ rfl
      simp [hnv, hnm]

end Welford.spec

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom — in the region-level core's and the
-- public headline's transitive proofs.
#axiomsClean welford_kernels_refinement_view
#axiomsClean welford_equiv

-- (2) The headline's statement surface is the two IO signatures plus the
-- audit-once kernel-equivalence combinator — NO spec (there are none), no
-- other project constant. If a spec-like definition ever creeps into the
-- statement, this fails.
#stmtSurfaceSubset welford_equiv ⊆
  [welfordTwoPassIO, welfordOnlineIO, VeriTile.Triton.KernelIO₁ₓ₂.Equiv,
   RoundingModel]


end VeriTile.Bench.Examples.Welford

