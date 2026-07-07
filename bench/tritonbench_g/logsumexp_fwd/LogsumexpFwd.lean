import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL
import VeriTile.Launch.Grid
import VeriTile.Math.LogSumExp
import VeriTile.Semantics.MaskedReduction
import VeriTile.Examples.Common
import VeriTile.Examples.LogSumExpEq

/-!
# `logsumexp_fwd` — strict per-kernel correctness

`logsumexp_fwd_kernel` is a row-wise log-sum-exp: program `(i_n, i_d)` loads the
`B`-sized block `[i_d·B, (i_d+1)·B)` of row `i_n` from `x`, optionally scales it
by `scale` (`HAS_SCALE`), computes the numerically-stable
`log(∑ exp(b_x − max b_x)) + max b_x`, and stores the per-block result to
`z[i_n·cdiv(D, B) + i_d]`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`logsumexp_fwd_kernel[grid](...)`, the grid size
`(N, cdiv(D, B))`, and how the runtime composes per-program writes into `z`) is
the *trusted boundary*. The grid-level theorem below is stated in the
per-program-local `Kernel.ForAllProgramsSome` form (one statement quantified
over every grid index), not a unified merged-state launch. Because the
program-id is universally quantified, the per-program statement covers every
program of the grid. The kernel deliberately stops at the Triton boundary; it
does **not** model the Python wrapper's later `z.logsumexp(-1)` cross-block
reduction.

## Proof architecture

```
logsumexp_fwd_kernel_output_summary               ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)                 surface lowers to the algorithm layer
  └─ logsumexp_fwd_kernel_compute_correct         ← ComputeCorrect, single-block row D=B=n+1
       └─ logsumexp_fwd_kernel_correct            ← full-row LSE readback
            └─ logsumexp_fwd_kernel_correct_full  ← tail-block masked-lane core
logsumexp_fwd_kernel_grid_blockLSE_correct        ← whole-grid per-block blockLSE
  └─ logsumexp_fwd_kernel_correct_full            (shared masked-lane core)
```

The spec is the standard mathematical `LSE` (from
`VeriTile.Math.LogSumExp`) together with the tiled
`blockLSE` / `partialLSE_full` / `scaledLane_full` / `validLanes` (from
`VeriTile.Semantics.TiledIndexing`), under the `TiledLogSumExp`
oracle — the math is *not* inline-duplicated here.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` and the
`@triton.heuristics` `HAS_SCALE` selection are not modeled (`HAS_SCALE` is a
plain `Bool` parameter, both branches proved). Out-of-range lanes in the tail
block are loaded with `other = -float("inf")` (= `⊥ : WithBot ℝ`); the reduction
is proved to depend only on the valid lanes (`withBot_sup'_partial`,
`sum_exp_masked_eq`), so padded-block values never contaminate the result. The
`.to(tl.int64)` casts on the program ids erase to identity at the algorithm
layer.
-/

namespace VeriTile.Bench.TritonBenchG.LogsumexpFwd

open VeriTile VeriTile.TiledLogSumExp VeriTile.Examples

/-- Faithful 1:1 transcription of `logsumexp_fwd.py`'s `logsumexp_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `D: tl.constexpr` / `B: tl.constexpr` / `HAS_SCALE: tl.constexpr` →
  Lean parameters; the `tl.constexpr` annotation is implicit on Lean params.
- Python `if cond: body` → `tl.if cond { body }`, the DSL-side gate equivalent.
- `scale` (Lean `ℝ` parameter) injected via `$(...)`. -/
def logsumexp_fwd_kernel
    (x z : RegionName)
    (D B : Nat) (HAS_SCALE : Bool) (scale : ℝ) :
    ComputeKernel := triton {
  i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
  o_d = i_d * $(B) + tl.arange(0, $(B))
  m_d = o_d < $(D)
  b_x = tl.load(x + i_n * $(D) + o_d, mask=m_d, other=-float("inf"))
  if HAS_SCALE {
    b_x = b_x * $(scale)
  }
  b_m = tl.max(b_x, 0)
  b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
  tl.store(z + i_n * tl.cdiv($(D), $(B)) + i_d, b_z)
}

/-! ## Correctness

The **tail block** is the last block when `D` is not divisible by `B`: pid axis-1
takes value `i_d = ⌊D / B⌋` and satisfies `i_d * B < D < (i_d + 1) * B`.
Some lanes are out-of-range; the kernel loads them with `other = -∞` (= `⊥`).

The masked-lane semantic chain is:
* `tl.load(..., mask=m_d, other=-float("inf"))` → `none : WithBot ℝ` for invalid lanes
* `tl.max(b_x, 0)` → `sup'` over the mixed tile; `withBot_sup'_partial` reduces it
  to `↑(validLanes.sup' ...)`, a finite real
* `tl.exp(b_x - b_m)` on an invalid lane → `exp(none - some m) = exp(none) = 0`
* `tl.sum(..., 0)` → `↑(∑_{valid i} exp(scaled i - m))`
* `tl.log + b_m` → `m + log(∑_{valid i} exp(scaled i - m))` -/

/-! ### Whole-grid launch correctness -/

/-- The Triton launch grid for `logsumexp_fwd_kernel`: one program per row and
one program per `B`-sized block of the row. -/
abbrev logsumexpGrid (N D B : Nat) : Grid :=
  { dims := [N, (D + B - 1) / B] }

private theorem mul_succ_lt_of_lt_cdiv {D n i : Nat}
    (h : i < (D + n) / (n+1)) :
    i * (n+1) < D := by
  have hlediv : i + 1 ≤ (D + n) / (n+1) := Nat.succ_le_of_lt h
  have hmul : (i+1) * (n+1) ≤ D + n :=
    Nat.mul_le_of_le_div (n+1) (i+1) (D+n) hlediv
  have hEq : (i+1) * (n+1) = i * (n+1) + (n+1) := by ring
  rw [hEq] at hmul
  omega

private theorem logsumexpGrid_tail_lt
    {N D n : Nat} (s : BlockState)
    (idx : GridIndex (logsumexpGrid N D (n+1))) :
    (s.withGridIndex idx).pids 1 * (n+1) < D := by
  let axis1 : Fin (logsumexpGrid N D (n+1)).rank := ⟨1, by simp [Grid.rank]⟩
  have hlt : (idx axis1).val < (D + n) / (n+1) := by
    simpa [axis1, logsumexpGrid, Grid.dim] using (idx axis1).isLt
  have hpid : (s.withGridIndex idx).pids 1 = (idx axis1).val := by
    simpa [axis1] using BlockState.withGridIndex_pids_in_rank s idx axis1
  simpa [hpid] using mul_succ_lt_of_lt_cdiv hlt

private theorem logsumexpGrid_input_loaded
    {N D n : Nat} (s : BlockState)
    (idx : GridIndex (logsumexpGrid N D (n+1)))
    (x : RegionName) (xs : Nat → Fin D → ℝ)
    (h_x : ∀ i_n : Nat, ∀ j : Fin D,
      s.readMem x (i_n * D + j.val) = xs i_n j) :
    ∀ j : Fin D,
      (s.withGridIndex idx).readMem x
          ((s.withGridIndex idx).pids 0 * D + j.val) =
        xs ((s.withGridIndex idx).pids 0) j := by
  intro j
  simpa [BlockState.withGridIndex, BlockState.withPids] using
    h_x ((s.withGridIndex idx).pids 0) j

/-! ### Private partial-block helper -/

/-- **Tail-block correctness for `logsumexp_fwd_kernel`.**

For pid `(i_n, i_d)` with at least one valid lane (`i_d * B < D`), the kernel
writes `partialLSE_full xs i_d h_tail HAS_SCALE scale` to region `z`. Covers
both `HAS_SCALE` cases and all mixed-tile masked-lane cases. -/
private theorem logsumexp_fwd_kernel_correct_full
    (x z : RegionName)
    {D : Nat} (n : Nat) (HAS_SCALE : Bool) (scale : ℝ)
    (s : BlockState)
    (xs : Fin D → ℝ)
    (h_x : ∀ j : Fin D, s.readMem x (s.pids 0 * D + j.val) = xs j)
    (h_tail : s.pids 1 * (n + 1) < D) :
    (exec (logsumexp_fwd_kernel x z D (n+1) HAS_SCALE scale) s).map
        (·.readMem z (s.pids 0 * ((D + n) / (n+1)) + s.pids 1)) =
      some (partialLSE_full xs (s.pids 1) h_tail HAS_SCALE scale) := by
  have h_ne : (Finset.univ : Finset (Fin (n+1))).Nonempty := Finset.univ_nonempty
  have h_filter : (validLanes n D (s.pids 1)).Nonempty := validLanes_nonempty h_tail
  have h_rm : ∀ (i : Fin (n+1)) (hi : s.pids 1 * (n+1) + i.val < D),
      s.readMem x (s.pids 0 * D + (s.pids 1 * (n+1) + i.val)) =
      xs ⟨s.pids 1 * (n+1) + i.val, hi⟩ := by
    intro i hi; simpa using h_x ⟨s.pids 1 * (n+1) + i.val, hi⟩
  -- Raw readMem-based lane function (definitionally equal to s.readMem x ...)
  let rm : Fin (n+1) → ℝ := fun i =>
    s.readMem x (s.pids 0 * D + (s.pids 1 * (n+1) + i.val))
  -- Fold helper: rewrite s.readMem occurrences to rm
  have h_fold : ∀ i : Fin (n+1),
      s.readMem x (s.pids 0 * D + (s.pids 1 * (n+1) + ↑i)) = rm i := fun _ => rfl
  rcases h_HAS_SCALE : HAS_SCALE with _ | _
  · -- HAS_SCALE = false
    simp [exec, logsumexp_fwd_kernel, stepStmts, stepStmt, evalOp.eq_def,
      Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
      Tile.reduceMax, Tile.reduceMaxDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt]
    simp only [← Int.natCast_one, ← Int.natCast_add, ← Int.natCast_mul,
      ← Int.natCast_ediv, Int.toNat_natCast, Int.ofNat_lt, if_true, h_fold]
    erw [sup'_masked_eq h_ne h_filter rm, sum_exp_masked_eq rm]
    simp only [WithBot.realLog_coe]
    erw [Option.map₂_coe_coe, WithBot.unbotD_coe]
    rw [add_comm]
    unfold partialLSE_full scaledLane_full validLanes
    simp only [Bool.false_eq_true, reduceIte]
    -- Common: the two sup' (max) sides agree (after unfolding partialLSE_full, the spec uses
    -- `validLanes.sup' ⋯ (scaledLane_full ... false scale)`, while the kernel side has
    -- `validLanes.sup' h_filter rm`; they agree on valid lanes).
    have h_m_eq : (validLanes n D (s.pids 1)).sup' h_filter rm =
        (validLanes n D (s.pids 1)).sup' h_filter
          (fun x => if h : s.pids 1 * (n+1) + x.val < D then xs ⟨s.pids 1 * (n+1) + x.val, h⟩ else 0) := by
      apply Finset.sup'_congr h_filter rfl
      intro i hi
      simp only [validLanes, Finset.mem_filter, Finset.mem_univ, true_and] at hi
      simp only [dif_pos hi]; exact h_rm i hi
    congr 1
    -- log of sum of exp parts agree (the sup' part was solved by congr 1 via rfl)
    congr 1
    erw [h_m_eq]
    apply Finset.sum_congr rfl
    intro i hi
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi
    congr 1  -- removes Real.exp, leaving subtraction equality
    simp only [dif_pos hi]  -- simplify if h : ... on RHS
    congr 1  -- splits a - b = c - d; denominator closes by def-eq
    -- rm i = xs ⟨...⟩: rm is transparent so exact closes via def-eq
    exact h_rm i hi
  · -- HAS_SCALE = true
    simp [exec, logsumexp_fwd_kernel, stepStmts, stepStmt, evalOp.eq_def,
      Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
      Tile.reduceMax, Tile.reduceMaxDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt]
    simp only [← Int.natCast_one, ← Int.natCast_add, ← Int.natCast_mul,
      ← Int.natCast_ediv, Int.toNat_natCast, Int.ofNat_lt, if_true, h_fold]
    erw [sup'_masked_map_eq h_ne h_filter rm (· * scale),
         sum_exp_masked_map_eq rm (· * scale)]
    simp only [WithBot.realLog_coe]
    erw [Option.map₂_coe_coe, WithBot.unbotD_coe]
    rw [add_comm]
    unfold partialLSE_full scaledLane_full validLanes
    simp only [reduceIte]
    -- Common: the two sup' (max) sides agree (scaled case)
    have h_m_eq_s : (validLanes n D (s.pids 1)).sup' h_filter (fun i => rm i * scale) =
        (validLanes n D (s.pids 1)).sup' h_filter
          (fun x => (if h : s.pids 1 * (n+1) + x.val < D then xs ⟨s.pids 1 * (n+1) + x.val, h⟩ else 0) * scale) := by
      apply Finset.sup'_congr h_filter rfl
      intro i hi
      simp only [validLanes, Finset.mem_filter, Finset.mem_univ, true_and] at hi
      simp only [dif_pos hi]; congr 1; exact h_rm i hi
    congr 1
    -- log of sum parts agree (the sup' part was solved by congr 1 via rfl)
    congr 1
    erw [h_m_eq_s]
    apply Finset.sum_congr rfl
    intro i hi
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi
    congr 1  -- removes Real.exp
    simp only [dif_pos hi]
    rw [show rm i * scale = xs ⟨s.pids 1 * (n+1) + i.val, hi⟩ * scale from by congr 1; exact h_rm i hi]
    congr 1  -- denominators: proof irrelevance

/-- **Complete row correctness for `logsumexp_fwd_kernel`.**

This is the public correctness theorem for the kernel in the single-block
configuration `D = B = n + 1` and `pid axis-1 = 0`. Under that launch shape the
kernel writes the complete row log-sum-exp, not a per-block partial. -/
theorem logsumexp_fwd_kernel_correct
    (x z : RegionName)
    (n : Nat) (HAS_SCALE : Bool) (scale : ℝ)
    (s : BlockState)
    (xs : Fin (n+1) → ℝ)
    (h_x : ∀ j : Fin (n+1), s.readMem x (s.pids 0 * (n+1) + j.val) = xs j)
    (h_pid1 : s.pids 1 = 0) :
    (exec (logsumexp_fwd_kernel x z (n+1) (n+1) HAS_SCALE scale) s).map
        (·.readMem z (s.pids 0)) =
      some (LSE xs HAS_SCALE scale) := by
  have h_tail : s.pids 1 * (n+1) < n+1 := by
    simp [h_pid1]
  have hview :=
    logsumexp_fwd_kernel_correct_full x z n HAS_SCALE scale s xs h_x h_tail
  have h_spec :
      partialLSE_full xs (s.pids 1) h_tail HAS_SCALE scale =
        LSE xs HAS_SCALE scale := by
    have h_stable :
        partialLSE_full xs (s.pids 1) h_tail HAS_SCALE scale =
          stableLSE xs (Nat.succ_pos n) HAS_SCALE scale := by
      simpa [h_pid1] using partialLSE_full_zero_self_eq_stableLSE xs HAS_SCALE scale
    exact h_stable.trans (stableLSE_eq_LSE xs (Nat.succ_pos n) HAS_SCALE scale)
  rw [h_spec] at hview
  have h_div : (n + 1 + n) / (n + 1) = 1 := by
    apply Nat.div_eq_of_lt_le
    · simp
    · omega
  simpa [h_pid1, h_div, Nat.mul_one] using hview

/-- **Compute-facing complete correctness for `logsumexp_fwd_kernel`.**

The theorem is intentionally complete-row only: `D = B = n + 1`, so one Triton
program covers the entire row and writes the standard mathematical `LSE` to
`z[i_n]`. -/
theorem logsumexp_fwd_kernel_compute_correct
    (x z : RegionName)
    (n : Nat) (HAS_SCALE : Bool) (scale : ℝ)
    (s : BlockState)
    (xs : Fin (n+1) → ℝ)
    (h_x : ∀ j : Fin (n+1), s.readMem x (s.pids 0 * (n+1) + j.val) = xs j)
    (h_pid1 : s.pids 1 = 0) :
    ComputeCorrect.Realizes
      (kernel := logsumexp_fwd_kernel x z (n+1) (n+1) HAS_SCALE scale)
      (initialState := s)
      (write := fun _ : PUnit => some (z, s.pids 0))
      (expected := fun _ => LSE xs HAS_SCALE scale) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  have hview :=
    logsumexp_fwd_kernel_correct x z n HAS_SCALE scale s xs h_x h_pid1
  rw [hExec] at hview
  simpa [ComputeCorrect.Realizes, ComputeKernel.ExecCorrect] using hview

/-- **Whole Triton-grid block-LSE correctness for `logsumexp_fwd_kernel`.**

For launch grid `(N, cdiv(D, B))`, every program instance `(i_n, i_d)` writes
the correct standard mathematical `blockLSE` for its row/block to
`z[i_n * cdiv(D, B) + i_d]`. This theorem covers both full blocks and the tail
block through the masked-lane theorem above. This is the faithful main theorem
for the Triton kernel itself. It deliberately stops at the Triton kernel
boundary; it does not model the Python wrapper's later
`z.logsumexp(-1)`.

Note: until VeriTile has a whole-grid `launchExec` returning a single merged
final state, this theorem is stated in the per-program-local form
`Kernel.ForAllProgramsSome` instead of the unified `ComputeCorrect.Realizes`
surface. Single-program kernels in this file already use `Realizes` /
`OutputScalar`; once the launcher lands, the grid case will move there too. -/
theorem logsumexp_fwd_kernel_grid_blockLSE_correct
    (x z : RegionName)
    (N : Nat) {D : Nat} (n : Nat) (HAS_SCALE : Bool) (scale : ℝ)
    (s : BlockState)
    (xs : Nat → Fin D → ℝ)
    (h_x : ∀ i_n : Nat, ∀ j : Fin D,
      s.readMem x (i_n * D + j.val) = xs i_n j) :
    Kernel.ForAllProgramsSome
      (logsumexp_fwd_kernel x z D (n+1) HAS_SCALE scale).toAlgKernel
      (logsumexpGrid N D (n+1))
      s
      (fun idx s' =>
        let sIdx := s.withGridIndex idx
        let h_tail := logsumexpGrid_tail_lt s idx
        s'.readMem z (sIdx.pids 0 * ((D + n) / (n+1)) + sIdx.pids 1)
          = blockLSE (n := n) (xs (sIdx.pids 0)) (sIdx.pids 1)
            h_tail HAS_SCALE scale) := by
  intro idx
  let sIdx := s.withGridIndex idx
  have h_tail : sIdx.pids 1 * (n+1) < D := logsumexpGrid_tail_lt s idx
  have h_loaded : ∀ j : Fin D, sIdx.readMem x (sIdx.pids 0 * D + j.val) =
      xs (sIdx.pids 0) j :=
    logsumexpGrid_input_loaded s idx x xs h_x
  have hview :=
    logsumexp_fwd_kernel_correct_full x z n HAS_SCALE scale
      sIdx (xs (sIdx.pids 0)) h_loaded h_tail
  cases hExec : exec (logsumexp_fwd_kernel x z D (n+1) HAS_SCALE scale).toAlgKernel sIdx with
  | none =>
      simp [hExec] at hview
  | some final =>
      refine ⟨final, ?_, ?_⟩
      · rfl
      rw [hExec] at hview
      rw [partialLSE_full_eq_blockLSE
        (xs (sIdx.pids 0)) (sIdx.pids 1) h_tail HAS_SCALE scale] at hview
      simpa [sIdx, h_tail] using hview

/-- Per-kernel output summary for `logsumexp_fwd_kernel`: the DSL surface lowers
to the algorithm layer, and in the single-block configuration `D = B = n + 1`
(pid axis-1 = 0) the store to `z[i_n]` is compute-correct — it holds the
standard mathematical row `LSE xs HAS_SCALE scale`. -/
theorem logsumexp_fwd_kernel_output_summary
    (x z : RegionName)
    (n : Nat) (HAS_SCALE : Bool) (scale : ℝ)
    (s : BlockState)
    (xs : Fin (n+1) → ℝ)
    (h_x : ∀ j : Fin (n+1), s.readMem x (s.pids 0 * (n+1) + j.val) = xs j)
    (h_pid1 : s.pids 1 = 0) :
    (∃ alg, (logsumexp_fwd_kernel x z (n+1) (n+1) HAS_SCALE scale).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := logsumexp_fwd_kernel x z (n+1) (n+1) HAS_SCALE scale)
      (initialState := s)
      (write := fun _ : PUnit => some (z, s.pids 0))
      (expected := fun _ => LSE xs HAS_SCALE scale) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact logsumexp_fwd_kernel_compute_correct x z n HAS_SCALE scale s xs h_x h_pid1

end VeriTile.Bench.TritonBenchG.LogsumexpFwd
