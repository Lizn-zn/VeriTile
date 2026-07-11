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
a single Welford `for` loop then the same affine tail. Like a real Triton
layernorm, both kernels take a **`rowStride` argument** (stage 1 of the
address-layout roadmap): the input/output rows live at
`pid * rowStride + [0, N)` of an `M × rowStride` buffer (`InputRowLoadedAt`),
while the per-feature `γ`/`β` vectors stay contiguous at `[0, N)`. Both **store
the output row rounded to bf16** (`(y).to(tl.bfloat16)`); the mean/variance
reductions and the affine arithmetic run in ℝ (no intermediate rounding). Both realize the same
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
-- live in the library (`VeriTile.Triton.Float.StepR`).

/-- **Faithfulness bridge.** The raw-AST `onlineWelfordLoopBody` that the loop
proofs below reason about is *exactly* the Welford `forLoop` body that the
readable `fusedLayerNormKernel` DSL compiles to — statement index 3 of its
algorithm projection (after `pid`, `M := 0`, `S := 0`). Machine-checked by
`rfl`, so the hand-written loop body need not be matched against the DSL by eye:
review the DSL kernel, and this lemma certifies the transcription is faithful. -/
private theorem onlineWelfordLoopBody_is_dsl_loop
    (xReg γReg βReg yReg : RegionName) (N rowStride : Nat) (ε : ℝ) :
    (fusedLayerNormKernel xReg γReg βReg yReg N rowStride ε).body[3]?
      = some (Stmt.forLoop "i" N (onlineWelfordLoopBody xReg rowStride)) := rfl

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

/-! ### Rounding-degeneration plumbing (loop cast-free + bf16-store congruence) -/

/-- The fused kernel's `forLoop` statement steps identically under `execR R`
and `exec`. -/
private theorem layernorm_forLoop_castFree (R : RoundingModel)
    (xReg : RegionName) (N rowStride : Nat) (t : BlockState) :
    stepStmtR R (.forLoop "i" N (onlineWelfordLoopBody xReg rowStride)) t
      = stepStmt (.forLoop "i" N (onlineWelfordLoopBody xReg rowStride)) t := by
  simp only [stepStmtR, stepStmt,
    stepForLoopAuxR_castFree R (onlineWelfordLoopBody xReg rowStride)
      (onlineWelfordLoopBody_castFree R xReg rowStride) "i" 0 N t]

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
  [twoPassLayerNormKernel, fusedLayerNormKernel, InputRowLoadedAt,
   InputFeatureLoadedAt, ComputeRefine.Refines, RoundingModel, BlockState, RegionName]

end LayerNorm.theorems

/-! ## Flat-memory corollary — the bridge (stage 2 + execR) applied

Both LayerNorm kernels are register-indirect (`offs := …; tl.load(x + offs)`),
so this is bridge **v1.2** territory: the per-execution `Kernel.TraceSafeR`
contracts are discharged below by walking the actual executions (including the
fused kernel's Welford `forLoop`), and `FlatAlloc.refineR_mem_flatten`
transports the writes-equality headline to the flat address space: one flat
region, real pointer arithmetic, same refinement. -/
section LayerNorm.flat

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
    Stmt.FlattenOk, Op.FlattenOk, Op.FlattenOk.eq_def]

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
    Stmt.FlattenOk, Op.FlattenOk, Op.FlattenOk.eq_def, onlineWelfordLoopBody]

include hN in
set_option maxHeartbeats 1600000 in
/-- **Flat-memory LayerNorm refinement** — the full composition: stage 1
(stride-parameterized addressing), stage 2 (the flat-memory bridge), and the
rounding model. For any disjoint allocation of the four regions whose extents
cover the accessed row and feature ranges, the *translated* kernels — one flat
address space, real pointer arithmetic `base + pid·rowStride + i` — still
perform the same writes from the flattened state: two-pass and fused LayerNorm
agree cell-for-cell in flat memory. The per-execution safety contracts and
fragment membership are discharged above, not assumed. -/
theorem layernorm_kernels_refinement_flat
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (R : RoundingModel)
    (h_x : InputRowLoadedAt s xReg rowStride N xs)
    (h_γ : InputFeatureLoadedAt s γReg N γs)
    (h_β : InputFeatureLoadedAt s βReg N βs)
    (hxb : s.pids 0 * rowStride + N ≤ A.extent xReg)
    (hγb : N ≤ A.extent γReg) (hβb : N ≤ A.extent βReg)
    (hyb : s.pids 0 * rowStride + N ≤ A.extent yReg)
    (hu : s.undef = (fun _ _ => 0)) :
    ∀ lhs' rhs',
      execR R (A.flattenKernel ((twoPassLayerNormKernel
          xReg γReg βReg yReg N rowStride ε).toAlgKernel))
        (A.flattenState s) = some lhs' →
      execR R (A.flattenKernel ((fusedLayerNormKernel
          xReg γReg βReg yReg N rowStride ε).toAlgKernel))
        (A.flattenState s) = some rhs' →
      ∀ r o, lhs'.mem r o = rhs'.mem r o := by
  refine A.refineR_mem_flatten hd hcov R _ _ s
    (twoPass_traceSafeR xReg γReg βReg yReg N rowStride ε s R A.extent
      hxb hγb hβb hyb)
    (twoPass_flattenOk xReg γReg βReg yReg N rowStride ε)
    (fused_traceSafeR xReg γReg βReg yReg N rowStride ε s R A.extent
      hxb hγb hβb hyb)
    (fused_flattenOk xReg γReg βReg yReg N rowStride ε) hu ?_
  intro lhs' rhs' hL hR r o
  obtain ⟨-, hproj⟩ := layernorm_kernels_refinement_view xReg γReg βReg yReg
    N rowStride hN ε s xs γs βs R h_x h_γ h_β
  replace hproj : Kernel.RefineR R
      ((twoPassLayerNormKernel xReg γReg βReg yReg N rowStride ε).toAlgKernel)
      ((fusedLayerNormKernel xReg γReg βReg yReg N rowStride ε).toAlgKernel)
      (fun s0 lhs' rhs' => s0 = s →
        ∀ r, r ∉ ([] : List RegionName) → ∀ o, lhs'.mem r o = rhs'.mem r o) :=
    hproj
  exact hproj s lhs' rhs' hL hR rfl r (by simp) o

-- Trust audit for the flat corollary.
#axiomsClean layernorm_kernels_refinement_flat

end LayerNorm.flat

end VeriTile.Bench.Examples.LayerNorm

