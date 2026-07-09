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
      , .store .bf16 [N] (MemAccess.region yReg (.ref .nat [N] "offs"))
          (.castFloat .real .bf16 (.ref .real [N] "y")) MaskOpt.none
      ]
  ComputeKernel.fromKernelBody [xReg, γReg, βReg] [yReg] body

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

/-- `writeMemAsR` only rewrites `mem`, so register reads pass through it. -/
@[simp] private theorem writeMemAsR_regs (R : RoundingModel) (s : BlockState)
    (d : FloatDType) (reg : RegionName) (o : Nat) (v : TileCarrier d.toTileDType)
    (dt : TileDType) (sh : TileShape) (nm : RegName) :
    (s.writeMemAsR R d reg o v).regs dt sh nm = s.regs dt sh nm := rfl

/-- `stepForLoopAux` degenerates from `execR R` to `exec` when the loop body is
cast-free. Arbitrary-`R` generalization of `stepForLoopAuxR_triv`. -/
private theorem stepForLoopAuxR_castFree (R : RoundingModel) (body : List Stmt)
    (hbody : ∀ t : BlockState, stepStmtsR R body t = stepStmts body t) (idx : RegName) :
    ∀ (start n : Nat) (s : BlockState),
      stepForLoopAuxR R idx start n body s = stepForLoopAux idx start n body s
  | start, n, s => by
      rw [stepForLoopAuxR, stepForLoopAux]
      simp only [hbody (s.setReg idx .nat [] (Tile.scalar start))]
      split
      · cases stepStmts body (s.setReg idx .nat [] (Tile.scalar start)) with
        | none => rfl
        | some s' => exact stepForLoopAuxR_castFree R body hbody idx (start + 1) n s'
      · rfl
  termination_by start n _ => n - start
  decreasing_by omega

/-- The online Welford loop body is cast-free: it steps identically under
`execR R` and `exec`. -/
private theorem onlineWelfordLoopBody_castFree (R : RoundingModel)
    (xReg : RegionName) (N : Nat) (t : BlockState) :
    stepStmtsR R (onlineWelfordLoopBody xReg N) t
      = stepStmts (onlineWelfordLoopBody xReg N) t := by
  simp only [onlineWelfordLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp]
  rfl

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

/- Shared parameters of the headline. Hoisted to a `variable` block so the
signature carries only its genuine hypotheses: the compact `InputLoadedAt` /
`InputFeatureLoadedAt` input contracts and the three `yReg ≠ ·` aliasing
constraints (the output must not alias any input). -/
variable (xReg γReg βReg yReg : RegionName) (N : Nat) (hN : 0 < N) (ε : ℝ)
variable (s : BlockState) (xs γs βs : Fin N → ℝ)

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

