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
# LayerNorm: two-pass vs fused single-pass — writes-equality refinement

Self-contained showcase, read top to bottom: **kernels** first (the two
LayerNorm kernels), the **supporting lemmas** in the middle (`private` plumbing
— the Welford math, the loop invariant, the affine-tail decomposition, and the
two per-kernel spec realizations), the **theorem** last (one public headline
`layernorm_kernels_refinement_view`), then a compile-time **trust audit**. The
three real sections below are `LayerNorm.kernels`, `LayerNorm.lemmas`,
`LayerNorm.theorems`.

`twoPassLayerNormKernel` computes mean/variance with two `tl.sum` passes then the
affine `(x − μ)/√(var+ε)·γ + β`; `fusedLayerNormKernel` computes mean/variance in
a single Welford `for` loop then the same affine tail. Both **store the output
row rounded to bf16** (`(y).to(tl.bfloat16)`); the mean/variance reductions and
the affine arithmetic run in ℝ (no intermediate rounding). Both realize the same
LayerNorm spec (Welford's running (M,S) = two-pass (μ,S) via
`welford_eq_two_pass`), so from the same state they produce the same ℝ output
row, and the only rounding is the shared bf16 output store, which quantizes
equal values identically.

## The public result (bottom of file)

The single public headline is **`layernorm_kernels_refinement_view`** — a
kernel-vs-kernel refinement on `ComputeRefine.Refines R` (the rounding-model
surface): for the rounding model `R`, from the same state the two-pass and fused
kernels perform the same writes (no scratch regions, so the scratch list is
`[]`). Its statement mentions only the two kernels, the loaded-input contracts,
the writes-equality surface, and the state/region types — **no spec** (the
`#stmtSurfaceSubset` gate below enforces this; the LayerNorm spec and per-kernel
correctness lemmas are all `private`). The compositional rounding pattern is
`bench/examples/FusedSwiglu.lean`.
-/

namespace VeriTile.Bench.Examples.LayerNorm

open VeriTile.Triton VeriTile.Examples
open VeriTile.Triton.TiledReduction.WelfordRec

/-! ## Kernels -/
section LayerNorm.kernels
/-- Two-pass LayerNorm kernel: `tl.sum` twice (mean and var), then affine. -/
def twoPassLayerNormKernel
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(N) + tl.arange($(N))
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
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) : ComputeKernel := triton {
  pid := tl.program_id(0)
  M   := 0
  S   := 0
  tl.for i in $(N) {
    xi      := tl.load($(xReg) + (pid * $(N) + i))
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
  offs    := pid * $(N) + tl.arange($(N))
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
variable (xReg γReg βReg yReg : RegionName) (N : Nat) (hN : 0 < N) (ε : ℝ)
variable (s : BlockState) (xs γs βs : Fin N → ℝ)

/-! ## Supporting lemmas (private plumbing) -/
section LayerNorm.lemmas

-- Welford two-pass/recurrence math (twoPassMean/twoPassS/welfordMean/welfordS +
-- welford_eq_two_pass) now lives in `VeriTile.Triton.TiledReduction.WelfordRec`,
-- opened above; shared with the Welford showcase and VeriTile.Examples.WelfordKernels.
-- `onlineWelfordLoopBody` + its cast-free degeneration are shared from
-- `VeriTile.Examples.Common`; `stepForLoopAuxR_castFree` + `writeMemAsR_regs`
-- live in the library (`VeriTile.Triton.Float.StepR`).

/-- **Faithfulness bridge.** The raw-AST `onlineWelfordLoopBody` that the loop
proofs below reason about is *exactly* the Welford `forLoop` body that the
readable `fusedLayerNormKernel` DSL compiles to — statement index 3 of its
algorithm projection (after `pid`, `M := 0`, `S := 0`). Machine-checked by
`rfl`, so the hand-written loop body need not be matched against the DSL by eye:
review the DSL kernel, and this lemma certifies the transcription is faithful. -/
private theorem onlineWelfordLoopBody_is_dsl_loop
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) :
    (fusedLayerNormKernel xReg γReg βReg yReg N ε).body[3]?
      = some (Stmt.forLoop "i" N (onlineWelfordLoopBody xReg N)) := rfl

/-- **Loop invariant** for the fused kernel's Welford pass: after `k` iterations
of `onlineWelfordLoopBody`, the running `(M, S)` registers hold the Welford
recurrence values `welfordMean/welfordS xs k`, the block id is pinned, and every
input stays loaded. Declared as a `structure` (not an anonymous `∧`-chain) so
each clause is a *named field* — the invariant reads as an invariant and stands
out from the surrounding plumbing `def`s. -/
private structure LayerNormLoopInv {N : Nat}
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
  x_loaded : InputLoadedAt s xReg N xs
  /-- The scale vector `γ` is still loaded in `γReg`. -/
  γ_loaded : InputFeatureLoadedAt s γReg N γs
  /-- The bias vector `β` is still loaded in `βReg`. -/
  β_loaded : InputFeatureLoadedAt s βReg N βs

private theorem layernorm_welford_step
    {N : Nat} (xs γs βs : Fin N → ℝ)
    (xReg γReg βReg : RegionName) (origPid i : Nat)
    (s : BlockState) (hi : i < N)
    (hP : LayerNormLoopInv xs γs βs xReg γReg βReg origPid i s) :
    ∃ s',
      stepStmts (onlineWelfordLoopBody xReg N)
        (s.setReg "i" .nat [] (Tile.scalar i)) = some s' ∧
      LayerNormLoopInv xs γs βs xReg γReg βReg origPid (i + 1) s' := by
  rcases hP with ⟨hM, hS, hpidReg, hpid, hX, hγ, hβ⟩
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
    (xReg γReg βReg : RegionName) (N : Nat)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (h_x : InputLoadedAt s xReg N xs)
    (h_γ : InputFeatureLoadedAt s γReg N γs)
    (h_β : InputFeatureLoadedAt s βReg N βs) :
    let s0 :=
      ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
        "M" .real [] (Tile.scalar 0)).setReg
        "S" .real [] (Tile.scalar 0)
    ∃ sLoop,
      stepStmt (.forLoop "i" N (onlineWelfordLoopBody xReg N)) s0 = some sLoop
      ∧ LayerNormLoopInv xs γs βs xReg γReg βReg s.pid N sLoop := by
  intro s0
  have h_init : LayerNormLoopInv xs γs βs xReg γReg βReg s.pid 0 s0 := by
    refine ⟨?_, ?_, ?_, ?_, h_x, h_γ, h_β⟩
    · simp [s0, welfordMean]
    · simp [s0, welfordS]
    · simp [s0]
    · simp [s0]
  exact forLoop_inv
    (idx := "i") (n := N)
    (body := onlineWelfordLoopBody xReg N)
    (P := LayerNormLoopInv xs γs βs xReg γReg βReg s.pid)
    (s_init := s0)
    h_init
    (fun k st hk hP =>
      layernorm_welford_step xs γs βs xReg γReg βReg s.pid k st hk hP)

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

/-! ### Rounding-degeneration plumbing (loop cast-free + bf16-store congruence) -/

/-- The fused kernel's `forLoop` statement steps identically under `execR R`
and `exec`. -/
private theorem layernorm_forLoop_castFree (R : RoundingModel)
    (xReg : RegionName) (N : Nat) (t : BlockState) :
    stepStmtR R (.forLoop "i" N (onlineWelfordLoopBody xReg N)) t
      = stepStmt (.forLoop "i" N (onlineWelfordLoopBody xReg N)) t := by
  simp only [stepStmtR, stepStmt,
    stepForLoopAuxR_castFree R (onlineWelfordLoopBody xReg N)
      (onlineWelfordLoopBody_castFree R xReg N) "i" 0 N t]

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
/-- **two-pass refines fused** (`ComputeRefine.Refines R`, no scratch): for the
rounding model `R`, from the same initial state the two-pass and fused LayerNorm
kernels perform the same writes — their final memories agree at every cell. Both
compute the same per-lane ℝ output `(x−μ)/√(var+ε)·γ+β` (Welford's identity
`welford_eq_two_pass`) and round it at the shared bf16 output store. -/
theorem layernorm_kernels_refinement_view
    (R : RoundingModel)
    (h_x : InputLoadedAt s xReg N xs)
    (h_γ : InputFeatureLoadedAt s γReg N γs)
    (h_β : InputFeatureLoadedAt s βReg N βs) :
    ComputeRefine.Refines R
      (twoPassLayerNormKernel xReg γReg βReg yReg N ε)
      (fusedLayerNormKernel xReg γReg βReg yReg N ε) s [] := by
  obtain ⟨hMeanEq, hSEq⟩ := welford_eq_two_pass hN xs
  have h_inj : Function.Injective
      (fun idx : TileIndex [N] => s.pid * N + idx.1.val) :=
    injective_offset_singleton (s.pid * N)
  -- Fused-side loop bookkeeping: run the Welford loop once (its state is
  -- cast-free, so it steps identically under `execR R`).
  obtain ⟨sLoop, hLoop, hPloop⟩ :=
    layernorm_welford_loop xReg γReg βReg N s xs γs βs h_x h_γ h_β
  have hMemLoop : sLoop.mem = s.mem := by
    have h := hLoop
    rw [stepForLoopAux.forLoop_unfold] at h
    exact stepForLoopAux_mem_of_storeFree "i" (onlineWelfordLoopBody xReg N)
      (onlineWelfordLoopBody_storeFree xReg N) 0 N
      (((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
          "M" .real [] (Tile.scalar 0)).setReg "S" .real [] (Tile.scalar 0))
      sLoop h
  rcases hPloop with ⟨hM, hS, hpidReg, _hpidLoop, hXl, hγl, hβl⟩
  let s0 :=
    ((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
      "M" .real [] (Tile.scalar (some 0))).setReg "S" .real [] (Tile.scalar (some 0))
  have hLoopR :
      stepForLoopAuxR R "i" 0 N (onlineWelfordLoopBody xReg N) s0 = some sLoop := by
    rw [stepForLoopAuxR_castFree R (onlineWelfordLoopBody xReg N)
      (onlineWelfordLoopBody_castFree R xReg N) "i" 0 N s0]
    have h := hLoop
    rw [stepForLoopAux.forLoop_unfold] at h
    exact h
  apply ComputeKernel.computeRefineR_of_toAlgKernel rfl rfl
  intro s0' lhs' rhs' hL hR hs0
  subst s0'
  intro r hr o
  unfold InputLoadedAt at hXl
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
    unfold InputLoadedAt at h_x
    unfold InputFeatureLoadedAt at h_γ h_β
    rw [_hpidLoop] at hXl
    simp only [hXl, h_x, h_γ, h_β, hMeanEq, hSEq, twoPassMean, twoPassS, pow_two]
    rfl
  · -- base states: registers only differ; the fused loop preserved memory
    simp [BlockState.setReg_mem, hMemLoop]

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in the public theorem's transitive proof.
#axiomsClean layernorm_kernels_refinement_view

-- (2) The headline is a *kernel-vs-kernel* refinement: its statement may mention
-- ONLY the two kernels, the loaded-input contracts, the writes-equality surface,
-- and the state/region types — NO spec.
#stmtSurfaceSubset layernorm_kernels_refinement_view ⊆
  [twoPassLayerNormKernel, fusedLayerNormKernel, InputLoadedAt,
   InputFeatureLoadedAt, ComputeRefine.Refines, RoundingModel, BlockState, RegionName]

end LayerNorm.theorems

end VeriTile.Bench.Examples.LayerNorm

