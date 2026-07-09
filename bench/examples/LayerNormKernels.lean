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
a single Welford `for` loop then the same affine tail. Both realize the same
LayerNorm spec (Welford's running (M,S) = two-pass (μ,S) via
`welford_eq_two_pass`), so from the same state they write the same output row.

## The public result (bottom of file)

The single public headline is **`layernorm_kernels_refinement_view`** — a
kernel-vs-kernel refinement on `ComputeRefine.Refines_without_Rounding`: from the same state the
two-pass and fused kernels perform the same writes (no scratch regions, so the
scratch list is `[]`). Its statement mentions only the two kernels, the
loaded-input contracts, the writes-equality surface, and the state/region types
— **no spec** (the `#stmtSurfaceSubset` gate below enforces this; the LayerNorm
spec and per-kernel correctness lemmas are all `private`). For the
rounding-model (∀R) analogue of this compositional pattern see
`bench/examples/FusedSwiglu.lean`.
-/

namespace VeriTile.Bench.Examples.LayerNorm

open VeriTile.Triton VeriTile.Examples

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
  tl.store($(yReg) + offs, y)
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
  tl.store($(yReg) + offs, y)
}

end LayerNorm.kernels

/-! ## Supporting lemmas (private plumbing) -/
section LayerNorm.lemmas

/-- Two-pass mean: μ = (∑ xᵢ) / n. -/
private noncomputable def twoPassMean {n : Nat} (x : Fin n → ℝ) : ℝ :=
  (∑ i, x i) / n

/-- Two-pass sum-of-squared-deviations: S = ∑ (xᵢ − μ)². -/
private noncomputable def twoPassS {n : Nat} (x : Fin n → ℝ) : ℝ :=
  ∑ i, (x i - twoPassMean x) ^ 2

/-- Welford recurrence: running mean M_k after processing x[0..k-1].
    M_0 = 0, M_{k+1} = M_k + (x_k − M_k) / (k+1).
    Returns 0 if k > n (out-of-range). -/
private noncomputable def welfordMean {n : Nat} (x : Fin n → ℝ) : Nat → ℝ
  | 0     => 0
  | k + 1 =>
      if h : k < n then
        let prev := welfordMean x k
        prev + (x ⟨k, h⟩ - prev) / (k + 1)
      else welfordMean x k

/-- Welford recurrence: running sum-of-squared-deviations S_k.
    S_0 = 0, S_{k+1} = S_k + (x_k − M_k) · (x_k − M_{k+1}). -/
private noncomputable def welfordS {n : Nat} (x : Fin n → ℝ) : Nat → ℝ
  | 0     => 0
  | k + 1 =>
      if h : k < n then
        let prevM := welfordMean x k
        let curM  := welfordMean x (k + 1)
        welfordS x k + (x ⟨k, h⟩ - prevM) * (x ⟨k, h⟩ - curM)
      else welfordS x k

/-- Helper: for any prefix length k ≤ n, the running Welford mean times k
    equals the sum of the first k inputs. Used to derive the final
    `welfordMean x n = twoPassMean x` claim. -/
private theorem welford_mean_mul_eq_sum {n : Nat} (x : Fin n → ℝ) :
    ∀ k : Nat, ∀ (h : k ≤ n),
      welfordMean x k * k = ∑ i : Fin k, x (castFin h i) := by
  intro k
  induction k with
  | zero =>
    intro _
    simp [welfordMean]
  | succ j ih =>
    intro hk
    have hj : j ≤ n := Nat.le_of_succ_le hk
    have hj_lt : j < n := hk
    have ih' := ih hj
    have hwm : welfordMean x (j + 1) =
        welfordMean x j + (x ⟨j, hj_lt⟩ - welfordMean x j) / (j + 1) := by
      simp [welfordMean, hj_lt]
    rw [hwm]
    have hjp1_ne : ((j : ℝ) + 1) ≠ 0 := by
      have : (0 : ℝ) ≤ j := Nat.cast_nonneg j
      linarith
    have hcast : ((j + 1 : Nat) : ℝ) = (j : ℝ) + 1 := by push_cast; ring
    rw [hcast]
    have hlhs : (welfordMean x j + (x ⟨j, hj_lt⟩ - welfordMean x j) / ((j : ℝ) + 1))
                  * ((j : ℝ) + 1) = welfordMean x j * j + x ⟨j, hj_lt⟩ := by
      field_simp
      ring
    rw [hlhs]
    rw [Fin.sum_univ_castSucc]
    have h_last : x (castFin hk (Fin.last j)) = x ⟨j, hj_lt⟩ := by
      rfl
    have h_cs : ∀ i : Fin j, x (castFin hk i.castSucc) = x (castFin hj i) := by
      intro i; rfl
    simp only [h_cs, h_last]
    rw [ih']

/-- Welford's variance identity. For any prefix length k ≤ n,
    the running Welford S_k equals the sum-of-squared-deviations from M_k. -/
private theorem welford_S_eq_sum_sq_dev {n : Nat} (x : Fin n → ℝ) :
    ∀ k : Nat, ∀ (hk : k ≤ n),
      welfordS x k = ∑ i : Fin k, (x (castFin hk i) - welfordMean x k) ^ 2 := by
  intro k
  induction k with
  | zero =>
    intro _
    simp [welfordS]
  | succ j ih =>
    intro hk
    have hj : j ≤ n := Nat.le_of_succ_le hk
    have hj_lt : j < n := hk
    have ih' := ih hj
    set M := welfordMean x j with hMdef
    set M' := welfordMean x (j + 1) with hM'def
    set xj := x ⟨j, hj_lt⟩ with hxjdef
    have hM' : M' = M + (xj - M) / ((j : ℝ) + 1) := by
      simp [hM'def, welfordMean, hj_lt, hMdef, hxjdef]
    have hS' : welfordS x (j + 1) = welfordS x j + (xj - M) * (xj - M') := by
      simp [welfordS, hj_lt, hMdef, hM'def, hxjdef]
    have hjp1_pos : (0 : ℝ) < (j : ℝ) + 1 := by
      have : (0 : ℝ) ≤ j := Nat.cast_nonneg j
      linarith
    have hjp1_ne : ((j : ℝ) + 1) ≠ 0 := ne_of_gt hjp1_pos
    have hMean := welford_mean_mul_eq_sum x j hj
    have hMM' : ((j : ℝ) + 1) * (M' - M) = xj - M := by
      rw [hM']; field_simp; ring
    have hxj_M' : xj - M' = (j : ℝ) * (M' - M) := by
      have : xj - M' = (xj - M) - (M' - M) := by ring
      rw [this, ← hMM']; ring
    have hxj_M : xj - M = ((j : ℝ) + 1) * (M' - M) := hMM'.symm
    rw [hS', ih']
    rw [Fin.sum_univ_castSucc]
    have h_last : x (castFin hk (Fin.last j)) = xj := rfl
    have h_cs : ∀ i : Fin j, x (castFin hk i.castSucc) = x (castFin hj i) := by
      intro i; rfl
    simp only [h_cs, h_last]
    have key :
        (∑ i : Fin j, (x (castFin hj i) - M') ^ 2) -
          (∑ i : Fin j, (x (castFin hj i) - M) ^ 2)
        = (j : ℝ) * (M - M') ^ 2 := by
      rw [← Finset.sum_sub_distrib]
      have h_per : ∀ i : Fin j,
          (x (castFin hj i) - M') ^ 2 - (x (castFin hj i) - M) ^ 2 =
          (M - M') * (2 * x (castFin hj i) - M - M') := by
        intro i; ring
      simp_rw [h_per]
      rw [← Finset.mul_sum]
      have h_sum_split : ∀ i : Fin j,
          2 * x (castFin hj i) - M - M' =
            2 * x (castFin hj i) + (- M - M') := by
        intro i; ring
      simp_rw [h_sum_split]
      rw [Finset.sum_add_distrib, ← Finset.mul_sum]
      rw [Finset.sum_const]
      simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      have hSumX : ∑ i : Fin j, x (castFin hj i) = M * j := hMean.symm
      rw [hSumX]
      ring
    have lhs_alg : (xj - M) * (xj - M') - (xj - M') ^ 2
                 = (j : ℝ) * (M - M') ^ 2 := by
      rw [hxj_M, hxj_M']
      ring
    linarith [key, lhs_alg]

/-- The load-bearing identity for Welford kernel refinement: after processing
all `n` inputs, Welford's running `(M, S)` equals the two-pass `(μ, S)`. -/
private theorem welford_eq_two_pass {n : Nat} (hn : 0 < n) (x : Fin n → ℝ) :
    welfordMean x n = twoPassMean x ∧ welfordS x n = twoPassS x := by
  refine ⟨?_, ?_⟩
  · have hMul := welford_mean_mul_eq_sum x n (le_refl n)
    have hn_pos : (0 : ℝ) < n := by exact_mod_cast hn
    have hn_ne : (n : ℝ) ≠ 0 := ne_of_gt hn_pos
    unfold twoPassMean
    rw [eq_div_iff hn_ne, hMul]
    apply Finset.sum_congr rfl
    intro i _
    rfl
  · have hS := welford_S_eq_sum_sq_dev x n (le_refl n)
    have hM : welfordMean x n = twoPassMean x := by
      have hMul := welford_mean_mul_eq_sum x n (le_refl n)
      have hn_pos : (0 : ℝ) < n := by exact_mod_cast hn
      have hn_ne : (n : ℝ) ≠ 0 := ne_of_gt hn_pos
      unfold twoPassMean
      rw [eq_div_iff hn_ne, hMul]
      apply Finset.sum_congr rfl
      intro i _; rfl
    rw [hS, hM]
    unfold twoPassS
    apply Finset.sum_congr rfl
    intro i _
    rfl

private def onlineWelfordLoopBody (xReg : RegionName) (blockSize : Nat) : List Stmt :=
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

private def layerNormAffineTailKernel
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) : ComputeKernel :=
  let body : List Stmt :=
      [ .assign .real [] "μ" (.ref .real [] "M")
      , .assign .real [] "v"
          (.div .real .nil (.ref .real [] "S") (Op.natToReal (.constNat N)))
      , .assign .real [] "σ_inv"
          (.div .real .nil (.const 1)
            (.sqrt (.add .real .nil (.ref .real [] "v") (.const ε))))
      , .assign .nat [N] "offs"
          (.add .nat .scalarL
            (.mul .nat .nil (.ref .nat [] "pid") (.constNat N))
            (.arange N))
      , .assign .real [N] "x"
          (.load .real (MemAccess.region xReg (.ref .nat [N] "offs")) MaskOpt.none)
      , .assign .real [N] "γ"
          (.load .real (MemAccess.region γReg (.arange N)) MaskOpt.none)
      , .assign .real [N] "β"
          (.load .real (MemAccess.region βReg (.arange N)) MaskOpt.none)
      , .assign .real [N] "y"
          (.add .real (.consSame .nil)
            (.mul .real (.consSame .nil)
              (.mul .real .scalarR
                (.sub .real .scalarR (.ref .real [N] "x") (.ref .real [] "μ"))
                (.ref .real [] "σ_inv"))
              (.ref .real [N] "γ"))
            (.ref .real [N] "β"))
      , .store .real [N] (MemAccess.region yReg (.ref .nat [N] "offs"))
          (.ref .real [N] "y") MaskOpt.none
      ]
  ComputeKernel.fromKernelBody [xReg, γReg, βReg] [yReg] body

/-- LayerNorm spec: `y_i = (x_i − μ) / √(var + ε) · γ_i + β_i`.

Body uses the kernel-internal `twoPassMean`/`twoPassS` since the proof bridges
through the running ↔ two-pass equivalence. The user-facing operator is
`Triton.TiledReduction.layerNorm`; see `layerNormSpec_eq_layerNorm` below. -/
private noncomputable def layerNormSpec {N : Nat}
    (xs γs βs : Fin N → ℝ) (ε : ℝ) (i : Fin N) : ℝ :=
  let μ : ℝ := twoPassMean xs
  let v : ℝ := twoPassS xs / N
  (xs i - μ) / Real.sqrt (v + ε) * γs i + βs i

/-- `layerNormSpec` agrees with `Triton.TiledReduction.layerNorm`; the local
spec is a beta-η variant that exposes `twoPassMean`/`twoPassS` as named
let-bindings to ease kernel-side proofs. -/
private theorem layerNormSpec_eq_layerNorm {N : Nat}
    (xs γs βs : Fin N → ℝ) (ε : ℝ) (i : Fin N) :
    layerNormSpec xs γs βs ε i =
      Triton.TiledReduction.layerNorm xs γs βs ε i := by
  simp [layerNormSpec, Triton.TiledReduction.layerNorm,
        Triton.TiledReduction.welfordMean, Triton.TiledReduction.welfordVar,
        Triton.TiledReduction.welfordSumSq, Triton.TiledReduction.tileSum,
        twoPassMean, twoPassS]

private def P_layernorm {N : Nat}
    (xs γs βs : Fin N → ℝ) (xReg γReg βReg : RegionName)
    (origPid : Nat) (k : Nat) (s : BlockState) : Prop :=
  s.regs .real [] "M" = some (Tile.scalar (welfordMean xs k))
  ∧ s.regs .real [] "S" = some (Tile.scalar (welfordS xs k))
  ∧ s.regs .nat [] "pid" = some (Tile.scalar origPid)
  ∧ s.pid = origPid
  ∧ InputLoadedAt s xReg N xs
  ∧ InputFeatureLoadedAt s γReg N γs
  ∧ InputFeatureLoadedAt s βReg N βs

private theorem layernorm_welford_step
    {N : Nat} (xs γs βs : Fin N → ℝ)
    (xReg γReg βReg : RegionName) (origPid i : Nat)
    (s : BlockState) (hi : i < N)
    (hP : P_layernorm xs γs βs xReg γReg βReg origPid i s) :
    ∃ s',
      stepStmts (onlineWelfordLoopBody xReg N)
        (s.setReg "i" .nat [] (Tile.scalar i)) = some s' ∧
      P_layernorm xs γs βs xReg γReg βReg origPid (i + 1) s' := by
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
  · simp [onlineWelfordLoopBody, stepStmts, stepStmt, evalOp, Tile.bop,
      Tile.natToReal, NumericDType.add, NumericDType.mul, NumericDType.sub,
      NumericDType.div, hM, hS, hpidReg,
      xi, m, ssum, delta, m', delta2, ssum', s',
      WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv]
    rfl
  · simp [P_layernorm, s', InputLoadedAt, InputFeatureLoadedAt, welfordMean,
      welfordS, hi, xi, m, ssum, delta, m', delta2, ssum', hpidReg, hpid, hxi]
    constructor
    · intro j
      have hx := hX j
      rw [hpid] at hx
      exact hx
    · exact ⟨hγ, hβ⟩

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
      ∧ P_layernorm xs γs βs xReg γReg βReg s.pid N sLoop := by
  intro s0
  have h_init : P_layernorm xs γs βs xReg γReg βReg s.pid 0 s0 := by
    simp [P_layernorm, s0, welfordMean, welfordS]
    exact ⟨h_x, h_γ, h_β⟩
  exact forLoop_inv
    (idx := "i") (n := N)
    (body := onlineWelfordLoopBody xReg N)
    (P := P_layernorm xs γs βs xReg γReg βReg s.pid)
    (s_init := s0)
    h_init
    (fun k st hk hP =>
      layernorm_welford_step xs γs βs xReg γReg βReg s.pid k st hk hP)

private theorem layernorm_affine_tail_correct
    (xReg γReg βReg yReg : RegionName) (N : Nat) (hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (hP : P_layernorm xs γs βs xReg γReg βReg s.pid N s) :
    ∀ i : Fin N,
      observeAt (exec (layerNormAffineTailKernel xReg γReg βReg yReg N ε) s)
                yReg N s.pid i
        = some (layerNormSpec xs γs βs ε i) := by
  intro i
  rcases hP with ⟨hM, hS, hpidReg, _hpid, hX, hγ, hβ⟩
  have hMean := (welford_eq_two_pass hN xs).1
  have hSEq := (welford_eq_two_pass hN xs).2
  have h_inj : Function.Injective
      (fun idx : TileIndex [N] => s.pid * N + idx.1.val) :=
    injective_offset_singleton (s.pid * N)
  simp [observeAt, exec, layerNormAffineTailKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.natToReal,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        layerNormSpec,
        hM, hS, hpidReg, hMean, hSEq]
  unfold InputLoadedAt at hX
  unfold InputFeatureLoadedAt at hγ hβ
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [hX, hγ, hβ, div_eq_mul_inv]

set_option maxHeartbeats 800000 in
private theorem twopass_layernorm_correct
    (xReg γReg βReg yReg : RegionName) (N : Nat) (_hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs)
    (_h_γ : InputFeatureLoadedAt s γReg N γs)
    (_h_β : InputFeatureLoadedAt s βReg N βs)
    (_h_yx : yReg ≠ xReg) (_h_yγ : yReg ≠ γReg) (_h_yβ : yReg ≠ βReg) :
    ∀ i : Fin N,
      observeAt (exec (twoPassLayerNormKernel xReg γReg βReg yReg N ε) s)
                yReg N s.pid i
        = some (layerNormSpec xs γs βs ε i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [N] => s.pid * N + idx.1.val) :=
    injective_offset_singleton (s.pid * N)
  simp [observeAt, exec, twoPassLayerNormKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        Tile.natToReal,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        layerNormSpec, twoPassMean, twoPassS]
  repeat unfold evalOp
  simp [observeAt, exec, twoPassLayerNormKernel, stepStmts, stepStmt,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        Tile.natToReal,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        layerNormSpec, twoPassMean, twoPassS]
  unfold InputLoadedAt at _h_x
  unfold InputFeatureLoadedAt at _h_γ _h_β
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x, _h_γ, _h_β, pow_two, div_eq_mul_inv]
  exact Or.inl rfl

private theorem fused_layernorm_correct
    (xReg γReg βReg yReg : RegionName) (N : Nat) (_hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs)
    (_h_γ : InputFeatureLoadedAt s γReg N γs)
    (_h_β : InputFeatureLoadedAt s βReg N βs)
    (_h_yx : yReg ≠ xReg) (_h_yγ : yReg ≠ γReg) (_h_yβ : yReg ≠ βReg) :
    ∀ i : Fin N,
      observeAt (exec (fusedLayerNormKernel xReg γReg βReg yReg N ε) s)
                yReg N s.pid i
        = some (layerNormSpec xs γs βs ε i) := by
  intro i
  let s0 :=
    ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
      "M" .real [] (Tile.scalar 0)).setReg
      "S" .real [] (Tile.scalar 0)
  obtain ⟨sLoop, hLoop, hPloop⟩ :=
    layernorm_welford_loop xReg γReg βReg N s xs γs βs _h_x _h_γ _h_β
  have hLoopAux :
      stepForLoopAux "i" 0 N (onlineWelfordLoopBody xReg N) s0 =
        some sLoop := by
    simpa [s0, stepForLoopAux.forLoop_unfold] using hLoop
  have hLoopAuxExpanded :
      stepForLoopAux "i" 0 N
        [Stmt.assign .real [] "xi"
            (Op.load .real (MemAccess.region xReg
              (Op.add .nat .nil
                (Op.mul .nat .nil (Op.ref .nat [] "pid")
                  (Op.constNat N))
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
  have h_exec_tail :
      exec (fusedLayerNormKernel xReg γReg βReg yReg N ε) s =
        exec (layerNormAffineTailKernel xReg γReg βReg yReg N ε) sLoop := by
    -- The fused kernel's body is `[pid; M:=0; S:=0; forLoop body; tail...]`.
    -- The affine tail kernel's body is the same `tail...`. Therefore:
    -- exec fused s = stepStmts (tail) sLoop = exec affineTail sLoop.
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
    show stepStmts (fusedLayerNormKernel xReg γReg βReg yReg N ε).body s
        = stepStmts (layerNormAffineTailKernel xReg γReg βReg yReg N ε).body sLoop
    -- Unfold kernel body literals so `stepStmts.cons_some` can match `::`.
    simp only [show (fusedLayerNormKernel xReg γReg βReg yReg N ε).body =
        ([ .assign .nat [] "pid" (.programId 0)
         , .assign .real [] "M" (.const 0)
         , .assign .real [] "S" (.const 0)
         , .forLoop "i" N (onlineWelfordLoopBody xReg N)
         ] ++ (layerNormAffineTailKernel xReg γReg βReg yReg N ε).body) from rfl]
    show stepStmts ([_, _, _, _] ++ _) s = _
    -- Reduce stepStmts on append: stepStmts (l ++ rest) s steps through l first
    rw [show ∀ (a b c d : Stmt) (rest : List Stmt) (s : BlockState),
        stepStmts ([a, b, c, d] ++ rest) s
          = stepStmts (a :: b :: c :: d :: rest) s from fun _ _ _ _ _ _ => rfl]
    rw [stepStmts.cons_some hpid]
    rw [stepStmts.cons_some hM0]
    rw [stepStmts.cons_some hS0]
    rw [stepStmts.cons_some hLoop]
  rw [h_exec_tail]
  have hpidLoop : sLoop.pid = s.pid := by
    exact hPloop.2.2.2.1
  have hPtail : P_layernorm xs γs βs xReg γReg βReg sLoop.pid N sLoop := by
    rcases hPloop with ⟨hM, hS, hpidReg, hpid, hX, hγ, hβ⟩
    rw [hpid]
    exact ⟨hM, hS, hpidReg, hpid, hX, hγ, hβ⟩
  rw [← hpidLoop]
  exact layernorm_affine_tail_correct xReg γReg βReg yReg N _hN ε sLoop
    xs γs βs hPtail i

private theorem layernorm_kernels_refinement
    (xReg γReg βReg yReg : RegionName) (N : Nat) (_hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs)
    (_h_γ : InputFeatureLoadedAt s γReg N γs)
    (_h_β : InputFeatureLoadedAt s βReg N βs)
    (_h_yx : yReg ≠ xReg) (_h_yγ : yReg ≠ γReg) (_h_yβ : yReg ≠ βReg) :
    ∀ i : Fin N,
      observeAt (exec (twoPassLayerNormKernel xReg γReg βReg yReg N ε) s)
                yReg N s.pid i
        = observeAt (exec (fusedLayerNormKernel xReg γReg βReg yReg N ε) s)
                yReg N s.pid i := by
  intro i
  rw [twopass_layernorm_correct
        xReg γReg βReg yReg N _hN ε s xs γs βs
        _h_x _h_γ _h_β _h_yx _h_yγ _h_yβ i,
      fused_layernorm_correct
        xReg γReg βReg yReg N _hN ε s xs γs βs
        _h_x _h_γ _h_β _h_yx _h_yγ _h_yβ i]

/-- View-level surface for `layernorm_kernels_refinement`. -/
private theorem layernorm_kernels_refinement_exec_view
    (xReg γReg βReg yReg : RegionName) (N : Nat) (hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1))
    (h_γ : TensorView.loaded s (featureView γReg N)
      (fun idx : TileIndex [N] => γs idx.1))
    (h_β : TensorView.loaded s (featureView βReg N)
      (fun idx : TileIndex [N] => βs idx.1))
    (h_yx : yReg ≠ xReg) (h_yγ : yReg ≠ γReg) (h_yβ : yReg ≠ βReg) :
    ∀ idx : TileIndex [N],
      TensorView.observe (exec (twoPassLayerNormKernel xReg γReg βReg yReg N ε) s)
          (programTileView s yReg N) idx
        = TensorView.observe (exec (fusedLayerNormKernel xReg γReg βReg yReg N ε) s)
          (programTileView s yReg N) idx := by
  intro idx
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := N) (xs := xs) h_x
  have hγ := inputFeatureLoadedAt_of_featureView_loaded (s := s) (region := γReg)
    (N := N) (xs := γs) h_γ
  have hβ := inputFeatureLoadedAt_of_featureView_loaded (s := s) (region := βReg)
    (N := N) (xs := βs) h_β
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt]
    using layernorm_kernels_refinement xReg γReg βReg yReg N hN ε s xs γs βs
      hx hγ hβ h_yx h_yγ h_yβ idx.1

/-- A `stepForLoopAux` run whose body is store-free preserves every memory
cell. Used to show the fused kernel's Welford loop (register-only assigns)
leaves memory untouched before the final store. -/
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


end LayerNorm.lemmas

/-! ## The headline theorem -/
section LayerNorm.theorems

/- Shared parameters of the headline. Hoisted to a `variable` block so the
signature carries only its genuine hypotheses: the compact `InputLoadedAt` /
`InputFeatureLoadedAt` input contracts and the three `yReg ≠ ·` aliasing
constraints (the output must not alias any input). -/
variable (xReg γReg βReg yReg : RegionName) (N : Nat) (hN : 0 < N) (ε : ℝ)
variable (s : BlockState) (xs γs βs : Fin N → ℝ)

include hN in
set_option maxHeartbeats 1600000 in
/-- Compute-facing writes-equality refinement surface for
`layernorm_kernels_refinement`: from the same initial state, the two-pass and
fused LayerNorm kernels perform THE SAME WRITES — their final memories agree
at every cell, with no scratch regions. -/
theorem layernorm_kernels_refinement_view
    (h_x : InputLoadedAt s xReg N xs)
    (h_γ : InputFeatureLoadedAt s γReg N γs)
    (h_β : InputFeatureLoadedAt s βReg N βs)
    (h_yx : yReg ≠ xReg) (h_yγ : yReg ≠ γReg) (h_yβ : yReg ≠ βReg) :
    ComputeRefine.Refines_without_Rounding
      (twoPassLayerNormKernel xReg γReg βReg yReg N ε)
      (fusedLayerNormKernel xReg γReg βReg yReg N ε) s [] := by
  have hx := h_x
  have hγ := h_γ
  have hβ := h_β
  have hTP := twopass_layernorm_correct xReg γReg βReg yReg N hN ε s xs γs βs
    hx hγ hβ h_yx h_yγ h_yβ
  have hFU := fused_layernorm_correct xReg γReg βReg yReg N hN ε s xs γs βs
    hx hγ hβ h_yx h_yγ h_yβ
  have h_inj : Function.Injective
      (fun idx : TileIndex [N] => s.pid * N + idx.1.val) :=
    injective_offset_singleton (s.pid * N)
  -- Fused-side loop bookkeeping: run the Welford loop once, characterize its
  -- final state, and record that the loop (register-only assigns) leaves all
  -- of memory untouched.
  obtain ⟨sLoop, hLoop, hPloop⟩ :=
    layernorm_welford_loop xReg γReg βReg N s xs γs βs hx hγ hβ
  have hMemLoop : sLoop.mem = s.mem := by
    have h := hLoop
    rw [stepForLoopAux.forLoop_unfold] at h
    exact stepForLoopAux_mem_of_storeFree "i" (onlineWelfordLoopBody xReg N)
      (onlineWelfordLoopBody_storeFree xReg N) 0 N
      (((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
          "M" .real [] (Tile.scalar 0)).setReg "S" .real [] (Tile.scalar 0))
      sLoop h
  -- `exec fused s` runs the loop to `sLoop`, then the straight-line tail.
  have h_exec_tail :
      exec (fusedLayerNormKernel xReg γReg βReg yReg N ε).toAlgKernel s =
        exec (layerNormAffineTailKernel xReg γReg βReg yReg N ε).toAlgKernel
          sLoop := by
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
                = some (((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
                    "M" .real [] (Tile.scalar 0)).setReg
                    "S" .real [] (Tile.scalar 0)) := by
      simp [stepStmt, evalOp]
      rfl
    show stepStmts (fusedLayerNormKernel xReg γReg βReg yReg N ε).body s
        = stepStmts (layerNormAffineTailKernel xReg γReg βReg yReg N ε).body sLoop
    simp only [show (fusedLayerNormKernel xReg γReg βReg yReg N ε).body =
        ([ .assign .nat [] "pid" (.programId 0)
         , .assign .real [] "M" (.const 0)
         , .assign .real [] "S" (.const 0)
         , .forLoop "i" N (onlineWelfordLoopBody xReg N)
         ] ++ (layerNormAffineTailKernel xReg γReg βReg yReg N ε).body) from rfl]
    show stepStmts ([_, _, _, _] ++ _) s = _
    rw [show ∀ (a b c d : Stmt) (rest : List Stmt) (s : BlockState),
        stepStmts ([a, b, c, d] ++ rest) s
          = stepStmts (a :: b :: c :: d :: rest) s from fun _ _ _ _ _ _ => rfl]
    rw [stepStmts.cons_some hpid]
    rw [stepStmts.cons_some hM0]
    rw [stepStmts.cons_some hS0]
    rw [stepStmts.cons_some hLoop]
  apply ComputeKernel.computeRefine_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  -- Per-lane written-value equality, transported from the two proven
  -- per-lane correctness theorems through the exec hypotheses.
  have hEq : ∀ i : Fin N,
      lhs'.readMem yReg (s.pid * N + i.val)
        = rhs'.readMem yReg (s.pid * N + i.val) := by
    intro i
    have h1 := hTP i
    have h2 := hFU i
    rw [hL] at h1
    rw [hR] at h2
    simp [observeAt] at h1 h2
    exact h1.trans h2.symm
  rw [h_exec_tail] at hR
  rcases hPloop with ⟨hM, hS, hpidReg, _hpidLoop, _hXl, _hγl, _hβl⟩
  -- Reduce both executions to explicit single-scatter final states.
  simp [exec, twoPassLayerNormKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.natToReal,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div] at hL
  repeat unfold evalOp at hL
  simp [Tile.bop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.mul] at hL
  simp [exec, layerNormAffineTailKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.natToReal,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        hM, hS, hpidReg] at hR
  subst lhs'
  subst rhs'
  -- Both kernels end in ONE unmasked scatter over the same offsets
  -- `s.pid * N + k`; compare memories cell-by-cell.
  refine BlockState.foldl_writeMem_mem_congr _ _ _ _ ?_ r o _ _ ?_
  · -- written values agree lane-by-lane (both equal `layerNormSpec`)
    intro k _
    have h := hEq k.1
    rw [BlockState.scatter_readback_nd _ _ _ h_inj (k.1, PUnit.unit),
        BlockState.scatter_readback_nd _ _ _ h_inj (k.1, PUnit.unit)] at h
    exact h
  · -- base states: registers only differ; the loop preserved memory
    simp [hMemLoop]

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
   InputFeatureLoadedAt, ComputeRefine.Refines_without_Rounding, BlockState, RegionName]

end LayerNorm.theorems

end VeriTile.Bench.Examples.LayerNorm

