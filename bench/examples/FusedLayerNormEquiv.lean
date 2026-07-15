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
# LayerNorm: two-pass vs fused single-pass — kernel equivalence `≡[R]`

Self-contained showcase, read top to bottom: **kernels** first (the two
LayerNorm kernels), the **supporting lemmas** in the middle (`private` plumbing
— the Welford loop-invariant walk, store-freeness, the rounded-store
congruence; the two-pass/Welford identity itself is shared library math,
`TiledReduction.WelfordRec`), the region-level refinement next (the `private`
raw-contract core `layernorm_refines_core` and its public form
`layernorm_kernels_refinement_view`), then the **flat-memory bridge side
conditions**, the **specification** last (one public headline
`layernorm_equiv` on the `≡[R]` surface), and a compile-time **trust audit**.
The real sections below are `LayerNorm.kernels`, `LayerNorm.lemmas`,
`LayerNorm.theorems`, `LayerNorm.bridge`, `LayerNorm.spec`.

`twoPassLayerNormKernel` computes mean/variance with two `tl.sum` passes then the
affine `(x − μ)/√(var+ε)·γ + β`; `fusedLayerNormKernel` computes mean/variance in
a single Welford `for` loop then the same affine tail. Like a real Triton
layernorm, both kernels take a **`rowStride` argument** (stage 1 of the
address-layout roadmap): the input/output rows live at
`pid * rowStride + [0, N)` of an `M × rowStride` buffer, while the per-feature
`γ`/`β` vectors stay contiguous at `[0, N)`, **shared** by every program. Both
**store the output row rounded to bf16** (`(y).to(tl.bfloat16)`); the
mean/variance reductions and the affine arithmetic run in ℝ (no intermediate
rounding). Both compute the same per-lane ℝ output (Welford's running (M,S) =
two-pass (μ,S) via `welford_eq_two_pass`), so from the same state they produce
the same ℝ output row, and the only rounding is the shared bf16 output store,
which quantizes equal values identically — for **every** rounding model `R`.

## The public result (bottom of file)

The single public headline is **`layernorm_equiv`** — kernel equivalence on
the shared three-input IO signature:

    layerNormTwoPassIO N rowStride ε ≡[R] layerNormFusedIO N rowStride ε

`≡[R]` is the audit-once kernel-equivalence combinator (`KernelIO₃.Equiv`,
`VeriTile.Triton.Memory.KernelSpec`), the `⊨`-grade form of the refinement
surface. Spelled out, the headline says: for **every** disjoint flat placement
of the interface buffers `x`/`γ`/`β`/`y` (∀ base pointers, ∀ buffer sizes;
neither kernel has scratch), **every** program id whose row window
`[pid·rowStride, pid·rowStride + N)` fits in `x` and `y` and whose shared
feature window `[0, N)` fits in `γ` and `β`, and **every** launch state — **no
input hypotheses at all**, not even loaded inputs: "equal inputs" is simply
"the same `s₀`" — both translated pointer kernels terminate under `execR R`,
their `y` row windows hold equal values, and each kernel leaves every cell
outside its `y` row window untouched. There is no `0 < N` hypothesis either:
at `N = 0` every window is empty, so the row-agreement leg (∀ `j : Fin N`) is
vacuous and the frames say the kernels write nothing.

The statement mentions only the two IO signatures and the library equivalence
surface — **no spec** (the `#stmtSurfaceSubset` gate below enforces this).
Everything else — the Welford loop invariant, the per-execution safety walks,
and the region-level refinement `layernorm_kernels_refinement_view`
(`ComputeRefine.Refines R`, empty scratch list) whose writes-equality the
`hrun` obligation repackages — is scaffolding. There are **no spec
definitions**: the expected per-lane output never appears as a named function,
so a self-referential spec is impossible by construction (and there is no
`#specNonCircular` gate to run). The masked-IO pilot of this conversion is
`bench/examples/FusedSwigluEquiv.lean`.
-/

namespace VeriTile.Bench.Examples.LayerNorm

open VeriTile.Triton VeriTile.Examples
open VeriTile.Triton.TiledReduction.WelfordRec
open scoped VeriTile.Triton.KernelIO₃

/-! ## Kernels -/
section LayerNorm.kernels
/-- Two-pass LayerNorm kernel: `tl.sum` twice (mean and var), then affine. -/
def twoPassLayerNormKernel
    (xReg γReg βReg yReg : RegionName) (N rowStride : Nat) (ε : ℝ) :
    ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(rowStride) + tl.arange($(N))
  x      := tl.load($(xReg) + offs)
  s_x    := tl.sum(x)
  μ      := s_x / tl.toReal($(N))
  d      := x - μ
  s_d2   := tl.sum(d * d)
  v      := s_d2 / tl.toReal($(N))
  γ      := tl.load($(γReg) + tl.arange($(N)))
  β      := tl.load($(βReg) + tl.arange($(N)))
  σ_inv  := 1 / tl.sqrt(v + $(ε))
  y      := (x - μ) * σ_inv * γ + β
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

/-- Fused single-pass LayerNorm kernel: Welford `forLoop`, then affine. -/
def fusedLayerNormKernel
    (xReg γReg βReg yReg : RegionName) (N rowStride : Nat) (ε : ℝ) :
    ComputeKernel := triton {
  pid := tl.program_id(0)
  M   := 0
  S   := 0
  tl.for i in $(N) {
    xi      := tl.load($(xReg) + (pid * $(rowStride) + i))
    delta   := xi - M
    M       := M + delta / (tl.toReal(i) + 1)
    delta2  := xi - M
    S       := S + delta * delta2
  }
  μ       := M
  v       := S / tl.toReal($(N))
  σ_inv   := 1 / tl.sqrt(v + $(ε))
  -- Second pass to compute Y. The "fused" gain is that μ/var were
  -- computed in a single pass over `x`; the residual `(x − μ)` still
  -- needs the second read of x.
  offs    := pid * $(rowStride) + tl.arange($(N))
  x       := tl.load($(xReg) + offs)
  γ       := tl.load($(γReg) + tl.arange($(N)))
  β       := tl.load($(βReg) + tl.arange($(N)))
  y       := (x - μ) * σ_inv * γ + β
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

end LayerNorm.kernels

/- Variables shared by every lemma and the theorem below — declared **once**, at
namespace scope, so both `LayerNorm.lemmas` and `LayerNorm.theorems` inherit
them (`end <section>` only clears variables declared *inside* that section).
Hoisted out of the declarations so each signature carries only its genuine
hypotheses: the compact `InputLoadedAt` / `InputFeatureLoadedAt` input
contracts. -/
variable (xReg γReg βReg yReg : RegionName) (N rowStride : Nat) (hN : 0 < N) (ε : ℝ)
variable (s : BlockState) (xs γs βs : Fin N → ℝ)

/-! ## Supporting lemmas (private plumbing) -/
section LayerNorm.lemmas

-- Welford two-pass/recurrence math (twoPassMean/twoPassS/welfordMean/welfordS +
-- welford_eq_two_pass) now lives in `VeriTile.Triton.TiledReduction.WelfordRec`,
-- opened above; shared with the Welford showcase and VeriTile.Examples.WelfordKernels.
-- `onlineWelfordLoopBody` + its cast-free degeneration are shared from
-- `VeriTile.Examples.Common`; `stepForLoopAuxR_castFree` + `writeMemAsR_regs`
-- and the store-frame `BlockState.foldl_writeMemAsR_preserve_cell` live in
-- `VeriTile.Triton.Float.StepR`; the `Refines` unpacker
-- `ComputeRefine.Refines.out` in `VeriTile.Triton.Float.Refine`.

/-- **Loop invariant** for the fused kernel's Welford pass: after `k` iterations
of `onlineWelfordLoopBody`, the running `(M, S)` registers hold the Welford
recurrence values `welfordMean/welfordS xs k`, the block id is pinned, and every
input stays loaded. Declared as a `structure` (not an anonymous `∧`-chain) so
each clause is a *named field* — the invariant reads as an invariant and stands
out from the surrounding plumbing `def`s. -/
private structure LayerNormLoopInv {N : Nat} (rowStride : Nat)
    (xs γs βs : Fin N → ℝ) (xReg γReg βReg : RegionName)
    (origPid k : Nat) (s : BlockState) : Prop where
  /-- Running mean register `M` holds the Welford mean after `k` steps. -/
  M_eq     : s.regs .real [] "M" = some (Tile.scalar (welfordMean xs k))
  /-- Running sum-of-squares register `S` holds the Welford `S` after `k` steps. -/
  S_eq     : s.regs .real [] "S" = some (Tile.scalar (welfordS xs k))
  /-- The block-id register is pinned to `origPid`. -/
  pid_reg  : s.regs .nat [] "pid" = some (Tile.scalar origPid)
  /-- The block id itself is pinned to `origPid`. -/
  pid_eq   : s.pid = origPid
  /-- The input row `x` is still loaded in `xReg`. -/
  x_loaded : InputRowLoadedAt s xReg rowStride N xs
  /-- The scale vector `γ` is still loaded in `γReg`. -/
  γ_loaded : InputFeatureLoadedAt s γReg N γs
  /-- The bias vector `β` is still loaded in `βReg`. -/
  β_loaded : InputFeatureLoadedAt s βReg N βs

private theorem layernorm_welford_step
    {N : Nat} (rowStride : Nat) (xs γs βs : Fin N → ℝ)
    (xReg γReg βReg : RegionName) (origPid i : Nat)
    (s : BlockState) (hi : i < N)
    (hP : LayerNormLoopInv rowStride xs γs βs xReg γReg βReg origPid i s) :
    ∃ s',
      stepStmts (onlineWelfordLoopBody xReg rowStride)
        (s.setReg "i" .nat [] (Tile.scalar i)) = some s' ∧
      LayerNormLoopInv rowStride xs γs βs xReg γReg βReg origPid (i + 1) s' := by
  rcases hP with ⟨hM, hS, hpidReg, hpid, hX, hγ, hβ⟩
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
  · simp [onlineWelfordLoopBody, stepStmts, stepStmt, Tile.bop,
      Tile.natToReal, NumericDType.add, NumericDType.mul, NumericDType.sub,
      NumericDType.div, hM, hS, hpidReg,
      xi, m, ssum, delta, m', delta2, ssum', s',
      WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv]
    rfl
  · -- one-step unfoldings of the Welford recurrence at `i + 1`
    have hWM : welfordMean xs (i + 1)
        = welfordMean xs i + (xs ⟨i, hi⟩ - welfordMean xs i) / ((i : ℝ) + 1) := by
      simp [welfordMean, hi]
    have hWS : welfordS xs (i + 1)
        = welfordS xs i + (xs ⟨i, hi⟩ - welfordMean xs i)
            * (xs ⟨i, hi⟩ - welfordMean xs (i + 1)) := by
      simp [welfordS, hi]
    -- inputs stay loaded: `s'` only writes registers, preserving memory
    refine ⟨?_, ?_, ?_, ?_, hX, hγ, hβ⟩
    · simp [s', hWM, hxi, m, delta, m']
    · simp [s', hWS, hWM, hxi, m, ssum, delta, m', delta2, ssum']
    · simp [s', hpidReg]
    · simp [s', hpid]

private theorem layernorm_welford_loop
    (xReg γReg βReg : RegionName) (N rowStride : Nat)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (h_x : InputRowLoadedAt s xReg rowStride N xs)
    (h_γ : InputFeatureLoadedAt s γReg N γs)
    (h_β : InputFeatureLoadedAt s βReg N βs) :
    let s0 :=
      ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
        "M" .real [] (Tile.scalar 0)).setReg
        "S" .real [] (Tile.scalar 0)
    ∃ sLoop,
      stepStmt (.forLoop "i" N (onlineWelfordLoopBody xReg rowStride)) s0
        = some sLoop
      ∧ LayerNormLoopInv rowStride xs γs βs xReg γReg βReg s.pid N sLoop := by
  intro s0
  have h_init : LayerNormLoopInv rowStride xs γs βs xReg γReg βReg s.pid 0 s0 := by
    refine ⟨?_, ?_, ?_, ?_, h_x, h_γ, h_β⟩
    · simp [s0, welfordMean]
    · simp [s0, welfordS]
    · simp [s0]
    · simp [s0]
  exact forLoop_inv
    (idx := "i") (n := N)
    (body := onlineWelfordLoopBody xReg rowStride)
    (P := LayerNormLoopInv rowStride xs γs βs xReg γReg βReg s.pid)
    (s_init := s0)
    h_init
    (fun k st hk hP =>
      layernorm_welford_step rowStride xs γs βs xReg γReg βReg s.pid k st hk hP)

private theorem stepForLoopAux_mem_of_storeFree
    (idx : RegName) (body : List Stmt)
    (hsf : body.all (fun st => storeFree st) = Bool.true)
    (start n : Nat) (s s' : BlockState)
    (h : stepForLoopAux idx start n body s = some s') :
    s'.mem = s.mem := by
  by_cases hlt : start < n
  · rw [stepForLoopAux.step_lt hlt] at h
    cases hbody : stepStmts body (s.setReg idx .nat [] (Tile.scalar start)) <;>
      simp [hbody] at h
    rename_i mid
    have hmid : mid.mem = s.mem :=
      storeFree_stepStmts_mem body (s.setReg idx .nat [] (Tile.scalar start))
        mid hsf hbody
    exact (stepForLoopAux_mem_of_storeFree idx body hsf (start + 1) n mid s' h).trans
      hmid
  · have hge : n ≤ start := Nat.le_of_not_gt hlt
    rw [stepForLoopAux.step_ge hge] at h
    simp_all
termination_by n - start

/-- The Welford loop body is store-free: five register assignments. -/
private theorem onlineWelfordLoopBody_storeFree (xReg : RegionName) (N : Nat) :
    (onlineWelfordLoopBody xReg N).all (fun st => storeFree st) = Bool.true := by
  simp [onlineWelfordLoopBody, storeFree]

/-! ### Rounding-store congruence (the shared bf16 output scatter) -/

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

end LayerNorm.lemmas

/-! ## The headline theorem -/
section LayerNorm.theorems

include hN in
set_option maxHeartbeats 1600000 in
/-- The refinement core on the raw loaded-input contracts
(`InputRowLoadedAt` / `InputFeatureLoadedAt`) — all the mathematics of
two-pass vs fused. `layernorm_kernels_refinement_view` repackages it on the
`TensorView.ViewsLoaded` bundle; that is the form `layernorm_equiv`'s
region-model obligation consumes. -/
private theorem layernorm_refines_core (R : RoundingModel)
    (h_x : InputRowLoadedAt s xReg rowStride N xs)
    (h_γ : InputFeatureLoadedAt s γReg N γs)
    (h_β : InputFeatureLoadedAt s βReg N βs) :
    ComputeRefine.Refines R
      (twoPassLayerNormKernel xReg γReg βReg yReg N rowStride ε)
      (fusedLayerNormKernel xReg γReg βReg yReg N rowStride ε) s [] := by
  obtain ⟨hMeanEq, hSEq⟩ := welford_eq_two_pass hN xs
  have h_inj : Function.Injective
      (fun idx : TileIndex [N] => s.pid * rowStride + idx.1.val) :=
    injective_offset_singleton (s.pid * rowStride)
  -- Fused-side loop bookkeeping: run the Welford loop once (its state is
  -- cast-free, so it steps identically under `execR R`).
  obtain ⟨sLoop, hLoop, hPloop⟩ :=
    layernorm_welford_loop xReg γReg βReg N rowStride s xs γs βs h_x h_γ h_β
  have hMemLoop : sLoop.mem = s.mem := by
    have h := hLoop
    rw [stepForLoopAux.forLoop_unfold] at h
    exact stepForLoopAux_mem_of_storeFree "i"
      (onlineWelfordLoopBody xReg rowStride)
      (onlineWelfordLoopBody_storeFree xReg rowStride) 0 N
      (((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
          "M" .real [] (Tile.scalar 0)).setReg "S" .real [] (Tile.scalar 0))
      sLoop h
  rcases hPloop with ⟨hM, hS, hpidReg, _hpidLoop, hXl, hγl, hβl⟩
  let s0 :=
    ((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
      "M" .real [] (Tile.scalar (some 0))).setReg "S" .real [] (Tile.scalar (some 0))
  have hLoopR :
      stepForLoopAuxR R "i" 0 N (onlineWelfordLoopBody xReg rowStride) s0
        = some sLoop := by
    rw [stepForLoopAuxR_castFree R (onlineWelfordLoopBody xReg rowStride)
      (onlineWelfordLoopBody_castFree R xReg rowStride) "i" 0 N s0]
    have h := hLoop
    rw [stepForLoopAux.forLoop_unfold] at h
    exact h
  apply ComputeKernel.computeRefineR_of_toAlgKernel rfl rfl
  intro s0' lhs' rhs' hL hR hs0
  subst s0'
  intro r hr o
  unfold InputRowLoadedAt at hXl
  unfold InputFeatureLoadedAt at hγl hβl
  -- reduce the two-pass execution to one bf16 scatter
  simp [execR, twoPassLayerNormKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, Tile.natToReal,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComputeExpr.toAlgorithm?, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        WithBot.realMul] at hL
  -- reduce the fused execution: expose the loop, fold state, collapse to sLoop,
  -- then reduce the affine tail to one bf16 scatter
  simp [execR, fusedLayerNormKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, Tile.natToReal,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComputeExpr.toAlgorithm?] at hR
  rw [show
      (((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
        "M" .real [] (Tile.scalar (some 0))).setReg
        "S" .real [] (Tile.scalar (some 0))) = s0 from rfl] at hR
  simp only [onlineWelfordLoopBody] at hLoopR
  rw [hLoopR] at hR
  simp [hM, hS, hpidReg, hγl, hβl] at hR
  subst hL
  subst hR
  have hpids : s.pids 0 = s.pid := rfl
  -- both kernels end in ONE unmasked bf16 scatter over the same offsets
  refine foldl_writeMemAsR_mem_congr R .bf16 _ _ _ _ ?_ r o _ _ ?_
  · -- per-lane written values agree (same ℝ output, rounded identically)
    intro k _
    refine congrArg (fun w : ℝ => R.cast FloatDType.real FloatDType.bf16 (some w)) ?_
    unfold InputRowLoadedAt at h_x
    unfold InputFeatureLoadedAt at h_γ h_β
    rw [_hpidLoop] at hXl
    simp only [hXl, h_x, h_γ, h_β, hMeanEq, hSEq, twoPassMean, twoPassS, pow_two]
    rfl
  · -- base states: registers only differ; the fused loop preserved memory
    simp [BlockState.setReg_mem, hMemLoop]

include hN in
/-- **two-pass refines fused** (`ComputeRefine.Refines R`, no scratch): for the
rounding model `R`, from the same initial state the two-pass and fused LayerNorm
kernels perform the same writes — their final memories agree at every cell. Both
compute the same per-lane ℝ output `(x−μ)/√(var+ε)·γ+β` (Welford's identity
`welford_eq_two_pass`) and round it at the shared bf16 output store.

Formerly the file's headline (on the `TensorView.ViewsLoaded` input bundle);
now the citable region-level wrapper over `layernorm_refines_core` that the
`≡[R]` specification's region-model obligation consumes via the library
unpacker `ComputeRefine.Refines.out` — its `y`-agreement leg is exactly this
memory agreement read at the output row window. From the obligation's bare
`s₀` the views bundle holds by `rfl` once the arrays are instantiated with
the values `s₀` holds at the views' own address maps. -/
theorem layernorm_kernels_refinement_view
    (R : RoundingModel)
    (hin : TensorView.ViewsLoaded s
      [TensorView.slot (rowTileView s xReg rowStride N) xs,
       TensorView.slot (featureView γReg N) γs,
       TensorView.slot (featureView βReg N) βs]) :
    ComputeRefine.Refines R
      (twoPassLayerNormKernel xReg γReg βReg yReg N rowStride ε)
      (fusedLayerNormKernel xReg γReg βReg yReg N rowStride ε) s [] := by
  obtain ⟨h_x', h_γ', h_β', -⟩ := hin
  exact layernorm_refines_core xReg γReg βReg yReg N rowStride hN ε s xs γs βs R
    (inputRowLoadedAt_of_rowTileView_loaded h_x')
    (inputFeatureLoadedAt_of_featureView_loaded h_γ')
    (inputFeatureLoadedAt_of_featureView_loaded h_β')

end LayerNorm.theorems

/-! ## Flat-memory bridge side conditions

Both LayerNorm kernels are register-indirect (`offs := …; tl.load(x + offs)`),
so no ∀-state safety contract covers them; the flat-memory bridge (v1.2,
`execR` flavor) takes the per-execution `Kernel.TraceSafeR` contract instead,
plus `FlattenOk` (bridge fragment membership). These are exactly the side
conditions `≡[R]` consumes: the two-pass kernel makes 4 accesses (row loads of
`x`, shared loads of `γ`/`β`, row store of `y`); the fused kernel adds the
Welford loop's `N` scalar `x` loads, walked by an invariant that pins the
`pid` register across iterations. All row accesses stay below
`pid·rowStride + N`, the shared feature accesses below `N`. -/
section LayerNorm.bridge

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

/-- Bounds discharge for accesses through the `offs` register: any state whose
`offs` register holds the row offsets `pid₀ * rowStride + i` addresses below
`pid₀ * rowStride + N`. -/
private theorem offs_activeAddressSafeR (R : RoundingModel)
    (bounds : RegionBounds) (pid₀ : Nat) (t : BlockState)
    (active : TileIndex [N] → Prop)
    (hread : t.regs .nat [N] "offs"
      = some ⟨fun i => pid₀ * rowStride + i.1.val⟩)
    (reg : RegionName) (hreg : pid₀ * rowStride + N ≤ bounds reg) :
    memAccessActiveAddressSafeR R bounds
      (MemAccess.region reg (Op.ref .nat [N] "offs")) t active := by
  simp only [memAccessActiveAddressSafeR]
  intro offsets hoffs i _
  rw [show evalOpR R (Op.ref .nat [N] "offs") t
      = some ⟨fun i => pid₀ * rowStride + i.1.val⟩ from by
    simp [evalOpR.eq_def, hread]] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact lt_of_lt_of_le (Nat.add_lt_add_left i.1.isLt _) hreg

/-- Bounds discharge for the contiguous `tl.arange` accesses (`γ`/`β`). -/
private theorem arange_activeAddressSafeR (R : RoundingModel)
    (bounds : RegionBounds) (t : BlockState)
    (active : TileIndex [N] → Prop)
    (reg : RegionName) (hreg : N ≤ bounds reg) :
    memAccessActiveAddressSafeR R bounds
      (MemAccess.region reg (Op.arange N)) t active := by
  simp only [memAccessActiveAddressSafeR]
  intro offsets hoffs i _
  rw [show evalOpR R (Op.arange N) t = some ⟨fun i => i.1.val⟩ from by
    simp [evalOpR.eq_def, Tile.vec]] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact lt_of_lt_of_le i.1.isLt hreg

set_option maxHeartbeats 1600000 in
/-- The two-pass kernel is trace-safe: its three loads and one store stay
inside `bounds` along the actual execution. -/
private theorem twoPass_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (hxb : s.pids 0 * rowStride + N ≤ bounds xReg)
    (hγb : N ≤ bounds γReg) (hβb : N ≤ bounds βReg)
    (hyb : s.pids 0 * rowStride + N ≤ bounds yReg) :
    Kernel.TraceSafeR R bounds
      ((twoPassLayerNormKernel xReg γReg βReg yReg N rowStride ε).toAlgKernel)
      s := by
  unfold Kernel.TraceSafeR
  -- pid := program_id(0)
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmtR_assign_inv hs1
  rw [show evalOpR R (Op.programId 0) s = some (Tile.scalar (s.pids 0)) from by
    simp [evalOpR.eq_def]] at hv1
  obtain rfl := Option.some_inj.mp hv1
  -- offs := pid * rowStride + arange N
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmtR_assign_inv hs2
  rw [show evalOpR R (Op.add .nat .scalarL
      (Op.mul .nat .nil (Op.ref .nat [] "pid") (Op.constNat rowStride))
      (Op.arange N)) (s.setReg "pid" .nat [] (Tile.scalar (s.pids 0)))
      = some ⟨fun i => s.pids 0 * rowStride + i.1.val⟩ from by
    simp [evalOpR.eq_def, Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
      Tile.vec, BlockState.setReg]] at hv2
  obtain rfl := Option.some_inj.mp hv2
  -- x := load(xReg + offs)
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR]
    exact ⟨trivial, trivial,
      offs_activeAddressSafeR N rowStride R bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) xReg hxb⟩
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmtR_assign_inv hs3
  -- s_x := sum(x)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmtR_assign_inv hs4
  -- μ := s_x / N
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
  -- v := s_d2 / N
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s8 hs8
  obtain ⟨v8, hv8, rfl⟩ := stepStmtR_assign_inv hs8
  -- γ := load(γReg + arange N)
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR]
    exact ⟨trivial, trivial, arange_activeAddressSafeR N R bounds _ _ γReg hγb⟩
  intro s9 hs9
  obtain ⟨v9, hv9, rfl⟩ := stepStmtR_assign_inv hs9
  -- β := load(βReg + arange N)
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR]
    exact ⟨trivial, trivial, arange_activeAddressSafeR N R bounds _ _ βReg hβb⟩
  intro s10 hs10
  obtain ⟨v10, hv10, rfl⟩ := stepStmtR_assign_inv hs10
  -- σ_inv := 1 / sqrt(v + ε)
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s11 hs11
  obtain ⟨v11, hv11, rfl⟩ := stepStmtR_assign_inv hs11
  -- y := (x - μ) * σ_inv * γ + β
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s12 hs12
  obtain ⟨v12, hv12, rfl⟩ := stepStmtR_assign_inv hs12
  -- store(yReg + offs, y.to(bf16))
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR, Op.SafeAtR]
  refine ⟨trivial, by simp [Op.SafeAtR.eq_def], trivial,
    offs_activeAddressSafeR N rowStride R bounds (s.pids 0) _ _ ?_
      (Region.cast yReg) (by simpa [Region.cast_self] using hyb)⟩
  simp [BlockState.setReg]

/-- The two-pass kernel sits inside the bridge's covered fragment. -/
private theorem twoPass_flattenOk :
    ((twoPassLayerNormKernel xReg γReg βReg yReg N rowStride ε
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [twoPassLayerNormKernel, ComputeKernel.toAlgKernel, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### The fused kernel: walking the Welford loop -/

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

/-- One Welford iteration is trace-safe when the `pid` register is pinned:
its single scalar load reads `pid₀ * rowStride + k < pid₀ * rowStride + N`. -/
private theorem welfordBody_traceSafeR (xReg : RegionName) (N rowStride : Nat)
    (R : RoundingModel) (bounds : RegionBounds) (pid₀ k : Nat) (hk : k < N)
    (hxb : pid₀ * rowStride + N ≤ bounds xReg)
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
private theorem welfordLoop_traceSafeR (xReg : RegionName) (N rowStride : Nat)
    (R : RoundingModel) (bounds : RegionBounds) (pid₀ : Nat)
    (hxb : pid₀ * rowStride + N ≤ bounds xReg) :
    ∀ (fuel k : Nat) (t : BlockState), N - k ≤ fuel →
      t.regs .nat [] "pid" = some (Tile.scalar pid₀) →
      Stmt.forLoopTraceSafeR R bounds "i" k N
        (onlineWelfordLoopBody xReg rowStride) t
  | 0, k, t, hf, hpid => by
      rw [Stmt.forLoopTraceSafeR]
      rw [if_neg (by omega)]
      trivial
  | fuel + 1, k, t, hf, hpid => by
      rw [Stmt.forLoopTraceSafeR]
      split
      next hlt =>
        refine ⟨welfordBody_traceSafeR xReg N rowStride R bounds pid₀ k hlt
          hxb t hpid, ?_⟩
        split
        next s' hs' =>
          refine welfordLoop_traceSafeR xReg N rowStride R bounds pid₀ hxb
            fuel (k + 1) s' (by omega) ?_
          rw [welfordBody_pid_preserved xReg rowStride hs']
          simp [BlockState.setReg, hpid]
        next => trivial
      next => trivial

/-- The Welford loop preserves the `pid` register. -/
private theorem welfordLoop_pid_preserved (xReg : RegionName) (rowStride : Nat)
    (R : RoundingModel) :
    ∀ (fuel k n : Nat) (t t' : BlockState), n - k ≤ fuel →
      stepForLoopAuxR R "i" k n (onlineWelfordLoopBody xReg rowStride) t
        = some t' →
      t'.regs .nat [] "pid" = t.regs .nat [] "pid"
  | fuel, k, n, t, t', hf, h => by
      rw [stepForLoopAuxR] at h
      split at h
      next hlt =>
        cases fuel with
        | zero => omega
        | succ fuel =>
            cases hb : stepStmtsR R (onlineWelfordLoopBody xReg rowStride)
                (t.setReg "i" .nat [] (Tile.scalar k)) with
            | none =>
                rw [hb] at h
                exact absurd h (by simp)
            | some s1 =>
                rw [hb] at h
                replace h : stepForLoopAuxR R "i" (k + 1) n
                    (onlineWelfordLoopBody xReg rowStride) s1 = some t' := h
                rw [welfordLoop_pid_preserved xReg rowStride R fuel (k + 1) n
                    s1 t' (by omega) h,
                  welfordBody_pid_preserved xReg rowStride hb]
                simp [BlockState.setReg]
      next =>
        obtain rfl := Option.some_inj.mp h
        rfl

set_option maxHeartbeats 1600000 in
/-- The fused kernel is trace-safe: the Welford loop's `N` scalar loads and
the tail's loads/store stay inside `bounds` along the actual execution. -/
private theorem fused_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (hxb : s.pids 0 * rowStride + N ≤ bounds xReg)
    (hγb : N ≤ bounds γReg) (hβb : N ≤ bounds βReg)
    (hyb : s.pids 0 * rowStride + N ≤ bounds yReg) :
    Kernel.TraceSafeR R bounds
      ((fusedLayerNormKernel xReg γReg βReg yReg N rowStride ε).toAlgKernel)
      s := by
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
  -- for i in N { … Welford … }
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR]
    exact welfordLoop_traceSafeR xReg N rowStride R bounds (s.pids 0) hxb
      N 0 _ (by omega) (by simp [BlockState.setReg])
  intro s4 hs4
  have hpid4 : s4.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)) := by
    simp only [stepStmtR] at hs4
    rw [welfordLoop_pid_preserved xReg rowStride R N 0 N _ s4 (by omega) hs4]
    simp [BlockState.setReg]
  -- μ := M
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmtR_assign_inv hs5
  -- v := S / N
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s6 hs6
  obtain ⟨v6, hv6, rfl⟩ := stepStmtR_assign_inv hs6
  -- σ_inv := 1 / sqrt(v + ε)
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s7 hs7
  obtain ⟨v7, hv7, rfl⟩ := stepStmtR_assign_inv hs7
  -- offs := pid * rowStride + arange N
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s8 hs8
  obtain ⟨v8, hv8, rfl⟩ := stepStmtR_assign_inv hs8
  rw [show evalOpR R (Op.add .nat .scalarL
      (Op.mul .nat .nil (Op.ref .nat [] "pid") (Op.constNat rowStride))
      (Op.arange N))
      (((s4.setReg "μ" .real [] v5).setReg "v" .real [] v6).setReg
        "σ_inv" .real [] v7)
      = some ⟨fun i => s.pids 0 * rowStride + i.1.val⟩ from by
    simp [evalOpR.eq_def, Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
      Tile.vec, BlockState.setReg, hpid4]] at hv8
  obtain rfl := Option.some_inj.mp hv8
  -- x := load(xReg + offs)
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR]
    exact ⟨trivial, trivial,
      offs_activeAddressSafeR N rowStride R bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) xReg hxb⟩
  intro s9 hs9
  obtain ⟨v9, hv9, rfl⟩ := stepStmtR_assign_inv hs9
  -- γ := load(γReg + arange N)
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR]
    exact ⟨trivial, trivial, arange_activeAddressSafeR N R bounds _ _ γReg hγb⟩
  intro s10 hs10
  obtain ⟨v10, hv10, rfl⟩ := stepStmtR_assign_inv hs10
  -- β := load(βReg + arange N)
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR]
    exact ⟨trivial, trivial, arange_activeAddressSafeR N R bounds _ _ βReg hβb⟩
  intro s11 hs11
  obtain ⟨v11, hv11, rfl⟩ := stepStmtR_assign_inv hs11
  -- y := (x - μ) * σ_inv * γ + β
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
  intro s12 hs12
  obtain ⟨v12, hv12, rfl⟩ := stepStmtR_assign_inv hs12
  -- store(yReg + offs, y.to(bf16))
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR, Op.SafeAtR]
  refine ⟨trivial, by simp [Op.SafeAtR.eq_def], trivial,
    offs_activeAddressSafeR N rowStride R bounds (s.pids 0) _ _ ?_
      (Region.cast yReg) (by simpa [Region.cast_self] using hyb)⟩
  simp [BlockState.setReg]

/-- The fused kernel sits inside the bridge's covered fragment. -/
private theorem fused_flattenOk :
    ((fusedLayerNormKernel xReg γReg βReg yReg N rowStride ε
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [fusedLayerNormKernel, ComputeKernel.toAlgKernel, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk.eq_def]

end LayerNorm.bridge

/-! ## The spec: `layerNormTwoPassIO ≡[R] layerNormFusedIO` -/
section LayerNorm.spec

/-! ### Region-model runs (termination + frame, from ANY state)

`≡[R]`'s region-model obligation starts from **any** state — no loaded
inputs, no clean undef — so nothing below takes an input contract: where the
Welford loop invariant wants loaded values, the values `s` actually holds
(`fun j => s.readMem …`) satisfy the contract by `rfl`. Each run lemma
packages termination under `execR R` with the per-kernel frame (the single
unmasked bf16 output scatter touches exactly the row window
`[pid·rowStride, pid·rowStride + N)` of `yReg`). -/

/-- The two-pass kernel terminates under `execR R` from every state and
writes nothing outside its `yReg` row window. -/
private theorem twoPass_execR_run (R : RoundingModel) (s : BlockState) :
    ∃ s', execR R
        ((twoPassLayerNormKernel xReg γReg βReg yReg N rowStride ε
          ).toAlgKernel) s = some s'
      ∧ ∀ r o, (∀ j : Fin N, ¬(r = yReg ∧ o = s.pids 0 * rowStride + j.val)) →
          s'.mem r o = s.mem r o := by
  obtain ⟨s', hexec⟩ : ∃ s', execR R
      ((twoPassLayerNormKernel xReg γReg βReg yReg N rowStride ε
        ).toAlgKernel) s = some s' := by
    simp [execR, twoPassLayerNormKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
          Tile.bop, Tile.uop, Tile.natToReal,
          NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
          ComputeExpr.toAlgorithm?, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          WithBot.realMul]
  refine ⟨s', hexec, ?_⟩
  intro r o hmiss
  simp [execR, twoPassLayerNormKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, Tile.natToReal,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComputeExpr.toAlgorithm?, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        WithBot.realMul] at hexec
  subst hexec
  refine Eq.trans
    (BlockState.foldl_writeMemAsR_preserve_cell R .bf16 _ _ r o _
      (fun k _ hc => hmiss k.1 ⟨hc.1.symm, hc.2.symm⟩) _) ?_
  simp

/-- The fused kernel terminates under `execR R` from every state and writes
nothing outside its `yReg` row window — the Welford loop is store-free, so
the frame is again just the tail's output scatter. -/
private theorem fused_execR_run (R : RoundingModel) (s : BlockState) :
    ∃ s', execR R
        ((fusedLayerNormKernel xReg γReg βReg yReg N rowStride ε
          ).toAlgKernel) s = some s'
      ∧ ∀ r o, (∀ j : Fin N, ¬(r = yReg ∧ o = s.pids 0 * rowStride + j.val)) →
          s'.mem r o = s.mem r o := by
  -- run the Welford loop once, from the values `s` actually holds
  obtain ⟨sLoop, hLoop, hPloop⟩ :=
    layernorm_welford_loop xReg γReg βReg N rowStride s
      (fun j => s.readMem xReg (s.pid * rowStride + j.val))
      (fun j => s.readMem γReg j.val)
      (fun j => s.readMem βReg j.val)
      (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
  have hMemLoop : sLoop.mem = s.mem := by
    have h := hLoop
    rw [stepForLoopAux.forLoop_unfold] at h
    exact stepForLoopAux_mem_of_storeFree "i"
      (onlineWelfordLoopBody xReg rowStride)
      (onlineWelfordLoopBody_storeFree xReg rowStride) 0 N
      (((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
          "M" .real [] (Tile.scalar 0)).setReg "S" .real [] (Tile.scalar 0))
      sLoop h
  rcases hPloop with ⟨hM, hS, hpidReg, _hpidLoop, hXl, hγl, hβl⟩
  let s0 :=
    ((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
      "M" .real [] (Tile.scalar (some 0))).setReg "S" .real [] (Tile.scalar (some 0))
  have hLoopR :
      stepForLoopAuxR R "i" 0 N (onlineWelfordLoopBody xReg rowStride) s0
        = some sLoop := by
    rw [stepForLoopAuxR_castFree R (onlineWelfordLoopBody xReg rowStride)
      (onlineWelfordLoopBody_castFree R xReg rowStride) "i" 0 N s0]
    have h := hLoop
    rw [stepForLoopAux.forLoop_unfold] at h
    exact h
  unfold InputFeatureLoadedAt at hγl hβl
  simp only [onlineWelfordLoopBody] at hLoopR
  obtain ⟨s', hexec⟩ : ∃ s', execR R
      ((fusedLayerNormKernel xReg γReg βReg yReg N rowStride ε
        ).toAlgKernel) s = some s' := by
    simp [execR, fusedLayerNormKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
          Tile.bop, Tile.uop, Tile.natToReal,
          NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
          ComputeExpr.toAlgorithm?]
    rw [show
        (((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
          "M" .real [] (Tile.scalar (some 0))).setReg
          "S" .real [] (Tile.scalar (some 0))) = s0 from rfl]
    rw [hLoopR]
    simp [hM, hS, hpidReg, hγl, hβl]
  refine ⟨s', hexec, ?_⟩
  intro r o hmiss
  simp [execR, fusedLayerNormKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, Tile.natToReal,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComputeExpr.toAlgorithm?] at hexec
  rw [show
      (((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
        "M" .real [] (Tile.scalar (some 0))).setReg
        "S" .real [] (Tile.scalar (some 0))) = s0 from rfl] at hexec
  rw [hLoopR] at hexec
  simp [hM, hS, hpidReg, hγl, hβl] at hexec
  subst hexec
  refine Eq.trans
    (BlockState.foldl_writeMemAsR_preserve_cell R .bf16 _ _ r o _
      (fun k _ hc => hmiss k.1 ⟨hc.1.symm, hc.2.symm⟩) _) ?_
  simp [hMemLoop]

/-- The two-pass kernel's three-input **IO signature** — the whole
kernel-specific audit surface of the headline: interface `x`, `γ`, `β` → `y`;
program `pid` owns the row windows `[pid · rowStride, pid · rowStride + N)`
of `x` and `y`, while `γ` and `β` are **shared** — every program reads the
same constant window `[0, N)`. No scratch: the kernel stages nothing through
memory (the Welford state of the fused sibling lives in registers). -/
def layerNormTwoPassIO (N rowStride : Nat) (ε : ℝ) : KernelIO₃ where
  kernel := twoPassLayerNormKernel ⟨"x"⟩ ⟨"γ"⟩ ⟨"β"⟩ ⟨"y"⟩ N rowStride ε
  in1 := ⟨"x"⟩
  in2 := ⟨"γ"⟩
  in3 := ⟨"β"⟩
  out := ⟨"y"⟩
  B1 := N
  B2 := N
  B3 := N
  Bout := N
  read1 := fun pid => pid * rowStride
  read2 := fun _ => 0
  read3 := fun _ => 0
  write := fun pid => pid * rowStride

/-- The fused kernel's IO signature: the **same interface by construction**
(structure update of `layerNormTwoPassIO` — buffers, windows, and tile
lengths shared verbatim), with the fused kernel plugged in. Also no
scratch — the single-pass (M, S) state is register-resident. -/
def layerNormFusedIO (N rowStride : Nat) (ε : ℝ) : KernelIO₃ :=
  { layerNormTwoPassIO N rowStride ε with
    kernel := fusedLayerNormKernel ⟨"x"⟩ ⟨"γ"⟩ ⟨"β"⟩ ⟨"y"⟩ N rowStride ε }

/-- **The headline**: two-pass LayerNorm is equivalent to the fused
single-pass kernel on their shared three-input IO signature, for **every**
rounding model — see the module docstring for the full contract `≡[R]`
unfolds to (both kernels terminate from any state, the `y` row windows
agree, each frames outside its `y` row window; no scratch on either side).
Proof: `KernelIO₃.Equiv.intro` assembles the region-model equivalence —
termination and the frames are the run lemmas above, the `y`-agreement leg
is `layernorm_kernels_refinement_view`'s memory agreement, unpacked by the
library `ComputeRefine.Refines.out` (Welford's identity + the shared bf16
store; the `N = 0` degenerate window is vacuous) — with the flat-memory
bridge side conditions (`FlattenOk` + `TraceSafeR` per kernel). -/
specification layernorm_equiv (R : RoundingModel) (N rowStride : Nat) (ε : ℝ) :
    layerNormTwoPassIO N rowStride ε ≡[R] layerNormFusedIO N rowStride ε := by
  refine KernelIO₃.Equiv.intro _ _ ?_ ?_ ?_ ?_ ?_
  · -- FlattenOk, two-pass
    exact twoPass_flattenOk ⟨"x"⟩ ⟨"γ"⟩ ⟨"β"⟩ ⟨"y"⟩ N rowStride ε
  · -- FlattenOk, fused
    exact fused_flattenOk ⟨"x"⟩ ⟨"γ"⟩ ⟨"β"⟩ ⟨"y"⟩ N rowStride ε
  · -- TraceSafeR, two-pass
    intro bounds t h1 h2 h3 h4 _
    simp only [layerNormTwoPassIO] at h1 h2 h3 h4 ⊢
    exact twoPass_traceSafeR ⟨"x"⟩ ⟨"γ"⟩ ⟨"β"⟩ ⟨"y"⟩ N rowStride ε t R bounds
      h1 (by simpa using h2) (by simpa using h3) h4
  · -- TraceSafeR, fused
    intro bounds t h1 h2 h3 h4 _
    simp only [layerNormFusedIO, layerNormTwoPassIO] at h1 h2 h3 h4 ⊢
    exact fused_traceSafeR ⟨"x"⟩ ⟨"γ"⟩ ⟨"β"⟩ ⟨"y"⟩ N rowStride ε t R bounds
      h1 (by simpa using h2) (by simpa using h3) h4
  · -- the region-model equivalence, from ANY state s₀ (no input hypotheses)
    intro s₀
    simp only [layerNormTwoPassIO, layerNormFusedIO]
    obtain ⟨s1, hexec1, hframe1⟩ :=
      twoPass_execR_run ⟨"x"⟩ ⟨"γ"⟩ ⟨"β"⟩ ⟨"y"⟩ N rowStride ε R s₀
    obtain ⟨s2, hexec2, hframe2⟩ :=
      fused_execR_run ⟨"x"⟩ ⟨"γ"⟩ ⟨"β"⟩ ⟨"y"⟩ N rowStride ε R s₀
    refine ⟨s1, s2, hexec1, hexec2, ?_, ?_, ?_⟩
    · -- y row windows agree
      intro j
      rcases Nat.eq_zero_or_pos N with hN0 | hN
      · exact absurd j.isLt (by omega)
      · -- the region-level refinement: memories agree everywhere (no scratch);
        -- the inputs both kernels see are whatever s₀ holds at the views'
        -- own address maps, so the ViewsLoaded bundle is rfl
        have hmem12 : ∀ r, r ∉ ([] : List RegionName) →
            ∀ o, s1.mem r o = s2.mem r o :=
          (layernorm_kernels_refinement_view ⟨"x"⟩ ⟨"γ"⟩ ⟨"β"⟩ ⟨"y"⟩ N rowStride
              hN ε s₀
              (fun j => s₀.readMem ⟨"x"⟩
                ((rowTileView s₀ ⟨"x"⟩ rowStride N).offset (j, PUnit.unit)))
              (fun j => s₀.readMem ⟨"γ"⟩
                ((featureView ⟨"γ"⟩ N).offset (j, PUnit.unit)))
              (fun j => s₀.readMem ⟨"β"⟩
                ((featureView ⟨"β"⟩ N).offset (j, PUnit.unit)))
              R ⟨fun _ => rfl, fun _ => rfl, fun _ => rfl, trivial⟩).out
            rfl rfl hexec1 hexec2
        have hmem := hmem12 ⟨"y"⟩ (by simp) (s₀.pid * rowStride + j.val)
        unfold BlockState.readMem
        rw [hmem]
    · -- two-pass frame: writes only the y row window (no scratch)
      intro r o hcond _
      refine hframe1 r o (fun j hc => ?_)
      rcases hcond with hne | hno
      · exact hne hc.1
      · exact hno j hc.2
    · -- fused frame: same window, same argument
      intro r o hcond _
      refine hframe2 r o (fun j hc => ?_)
      rcases hcond with hne | hno
      · exact hne hc.1
      · exact hno j hc.2

end LayerNorm.spec

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom — in the public region-level
-- refinement's and the `≡[R]` headline's transitive proofs.
#axiomsClean layernorm_kernels_refinement_view
#axiomsClean layernorm_equiv

-- (2) The headline's statement surface is the two IO signatures plus the
-- audit-once kernel-equivalence combinator — NO spec (there are none), no
-- other project constant. If a spec-like definition ever creeps into the
-- statement, this fails.
#stmtSurfaceSubset layernorm_equiv ⊆
  [layerNormTwoPassIO, layerNormFusedIO, VeriTile.Triton.KernelIO₃.Equiv,
   RoundingModel]

-- (There is no `#specNonCircular` gate: the file defines no specs at all, so a
-- self-referential spec is impossible by construction.)

end VeriTile.Bench.Examples.LayerNorm

