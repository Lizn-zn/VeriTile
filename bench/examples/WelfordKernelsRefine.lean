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
-/

namespace VeriTile.Bench.Examples.Welford

open VeriTile.Triton
open VeriTile.Examples (InputLoadedAt inputLoadedAt_of_programTileView_loaded programTileView castFin)

/-! ## Kernels -/
section Welford.kernels

/-- Two-pass mean/variance kernel: mean with one `tl.sum`, then population
variance with a second `tl.sum` over squared deviations. -/
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
  tl.store($(meanReg), (μ).to(tl.bfloat16))
  tl.store($(varReg), (v).to(tl.bfloat16))
}

/-- Online Welford mean/variance kernel: the same population mean/variance via
Welford's one-pass recurrence inside a `for` loop (running mean `M`, running
sum-of-squared-deviations `S`). -/
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
  tl.store($(meanReg), (M).to(tl.bfloat16))
  tl.store($(varReg), (S / tl.toReal($(blockSize))).to(tl.bfloat16))
}

end Welford.kernels

/-! ## Supporting lemmas (private plumbing) -/
section Welford.lemmas

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

/-- Loop invariant for `onlineWelfordKernel`: after `k` body iterations,
    register `M` holds `welfordMean xs k`, register `S` holds `welfordS xs k`,
    register `pid` holds the original program id, and the input region is
    unchanged. -/
private def P_welford {N : Nat} (xs : Fin N → ℝ) (xReg : RegionName)
    (origPid : Nat) (k : Nat) (s : BlockState) : Prop :=
  s.regs .real [] "M" = some (Tile.scalar (welfordMean xs k))
  ∧ s.regs .real [] "S" = some (Tile.scalar (welfordS xs k))
  ∧ s.regs .nat [] "pid" = some (Tile.scalar origPid)
  ∧ s.pid = origPid
  ∧ InputLoadedAt s xReg N xs

private theorem online_welford_step
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
private theorem onlineWelfordLoopBody_assigns (xReg : RegionName) (blockSize : Nat) :
    ∀ st ∈ onlineWelfordLoopBody xReg blockSize,
      ∃ (dtype : TileDType) (shape : TileShape) (name : RegName)
        (e : Op dtype shape), st = Stmt.assign dtype shape name e := by
  intro st hst
  simp only [onlineWelfordLoopBody, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩

/-! ### Rounding-degeneration plumbing

The online kernel's loop body carries no `castFloat`, so it steps identically
under `execR R` and `exec`; only the two boundary bf16 output stores differ. -/

/-- `stepForLoopAux` degenerates from `execR R` to `exec` whenever the loop body
does. Arbitrary-`R` generalization of `stepForLoopAuxR_triv`. -/
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
    (xReg : RegionName) (blockSize : Nat) (t : BlockState) :
    stepStmtsR R (onlineWelfordLoopBody xReg blockSize) t
      = stepStmts (onlineWelfordLoopBody xReg blockSize) t := by
  simp only [onlineWelfordLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp]
  rfl

/-- The online kernel's whole `forLoop` statement steps identically under
`execR R` and `exec`. -/
private theorem online_forLoop_castFree (R : RoundingModel)
    (xReg : RegionName) (blockSize : Nat) (t : BlockState) :
    stepStmtR R (.forLoop "i" blockSize (onlineWelfordLoopBody xReg blockSize)) t
      = stepStmt (.forLoop "i" blockSize (onlineWelfordLoopBody xReg blockSize)) t := by
  simp only [stepStmtR, stepStmt,
    stepForLoopAuxR_castFree R (onlineWelfordLoopBody xReg blockSize)
      (onlineWelfordLoopBody_castFree R xReg blockSize) "i" 0 blockSize t]

/-- `writeMemAsR` only rewrites `mem`, so register reads pass through it — needed
so the second (var) store can read its value register over the first store. -/
@[simp] private theorem writeMemAsR_regs (R : RoundingModel) (s : BlockState)
    (d : FloatDType) (reg : RegionName) (o : Nat) (v : TileCarrier d.toTileDType)
    (dt : TileDType) (sh : TileShape) (nm : RegName) :
    (s.writeMemAsR R d reg o v).regs dt sh nm = s.regs dt sh nm := rfl

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
    (h_x : InputLoadedAt s xReg blockSize xs)
    (h_mv : meanReg ≠ varReg)
    (hL : execR R (twopassWelfordKernel xReg meanReg varReg blockSize) s = some lhs')
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
  unfold InputLoadedAt at h_x
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
    (h_x : InputLoadedAt s xReg blockSize xs)
    (h_mv : meanReg ≠ varReg)
    (hR : execR R (onlineWelfordKernel xReg meanReg varReg blockSize) s = some rhs')
    (r : RegionName) (o : Nat) :
    rhs'.mem r o =
      if r = varReg ∧ o = 0 then roundedCell R (welfordS xs blockSize / blockSize)
      else if r = meanReg ∧ o = 0 then roundedCell R (welfordMean xs blockSize)
      else s.mem r o := by
  let s0 :=
    ((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
      "M" .real [] (Tile.scalar (some 0))).setReg
      "S" .real [] (Tile.scalar (some 0))
  have h_init : P_welford xs xReg s.pid 0 s0 ∧ ∀ r o, s0.mem r o = s.mem r o := by
    refine ⟨?_, fun r o => rfl⟩
    simp [P_welford, s0, welfordMean, welfordS]
    exact ⟨rfl, h_x⟩
  obtain ⟨sLoop, hLoop, hPloop, hMemLoop⟩ :=
    forLoop_inv
      (idx := "i") (n := blockSize)
      (body := onlineWelfordLoopBody xReg blockSize)
      (P := fun k st => P_welford xs xReg s.pid k st ∧ ∀ r o, st.mem r o = s.mem r o)
      (s_init := s0)
      h_init
      (fun i st hi hP => by
        obtain ⟨hPw, hMem⟩ := hP
        obtain ⟨st', hstep, hPw'⟩ := online_welford_step xs xReg s.pid i st hi hPw
        refine ⟨st', hstep, hPw', fun r o => ?_⟩
        rw [stepStmts_assigns_mem _ (onlineWelfordLoopBody_assigns xReg blockSize)
          _ _ hstep r o]
        exact hMem r o)
  rcases hPloop with ⟨hM, hS, _hpidReg, _hpid, _hX⟩
  have hLoopR :
      stepForLoopAuxR R "i" 0 blockSize (onlineWelfordLoopBody xReg blockSize) s0
        = some sLoop := by
    rw [stepForLoopAuxR_castFree R (onlineWelfordLoopBody xReg blockSize)
      (onlineWelfordLoopBody_castFree R xReg blockSize) "i" 0 blockSize s0]
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

/- Shared parameters of the headline. Hoisted to a `variable` block so the
signature carries only its genuine hypotheses: the compact `InputLoadedAt`
input contract and the `meanReg ≠ varReg` aliasing constraint. -/
variable (xReg meanReg varReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
variable (s : BlockState) (xs : Fin blockSize → ℝ) (R : RoundingModel)

include hN in
/-- **two-pass refines online** (`ComputeRefine.Refines R`, no scratch): for the
rounding model `R`, from the same initial state `twopassWelfordKernel` and
`onlineWelfordKernel` perform the same writes — their final memories agree at
every cell. Both compute the same per-lane ℝ mean/variance (Welford's identity
`welford_eq_two_pass`) and round it at the shared bf16 output stores. -/
theorem welford_kernels_refinement_view
    (h_x : InputLoadedAt s xReg blockSize xs)
    (h_mv : meanReg ≠ varReg) :
    ComputeRefine.Refines R
      (twopassWelfordKernel xReg meanReg varReg blockSize)
      (onlineWelfordKernel xReg meanReg varReg blockSize)
      s [] := by
  obtain ⟨hMeanEq, hSEq⟩ := welford_eq_two_pass hN xs
  apply ComputeKernel.computeRefineR_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r _hr o
  rw [twopass_execR_mem R xReg meanReg varReg blockSize s lhs' xs h_x h_mv hL r o,
    online_execR_mem R xReg meanReg varReg blockSize s rhs' xs h_x h_mv hR r o]
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
  [twopassWelfordKernel, onlineWelfordKernel, InputLoadedAt,
   ComputeRefine.Refines, RoundingModel, BlockState, RegionName]

end Welford.theorems

end VeriTile.Bench.Examples.Welford

