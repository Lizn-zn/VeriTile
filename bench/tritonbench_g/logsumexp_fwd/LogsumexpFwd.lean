import VeriTile.Triton

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
the *trusted boundary*. Both program-id axes are universally quantified in the
`⊨` headline, so the per-program statement covers every program of the grid
(and every out-of-grid program with an in-bounds store cell). The kernel
deliberately stops at the Triton boundary; it does **not** model the Python
wrapper's later `z.logsumexp(-1)` cross-block reduction.

## Proof architecture

```
logsumexp_fwd_kernel_correctness                  ← TOP THEOREM (logsumexpIO ⊨ per-block LSE)
  ├─ logsumexp_fwd_kernel_flattenOk               bridge fragment membership
  ├─ logsumexp_fwd_kernel_traceSafe               per-execution lane-wise safety walk
  └─ logsumexp_fwd_kernel_region_run              region-model masked Hoare triple
       ├─ logsumexp_fwd_kernel_correct_full       ← in-grid blocks (full + tail)
       │    └─ partialLSE_full_eq_blockLSE_local  row spec → block-local spec bridge
       ├─ logsumexp_fwd_kernel_store_zero         ← all-masked blocks store the ⊥-fallback 0
       └─ logsumexp_fwd_kernel_frame              single-cell store frame
logsumexp_fwd_kernel_grid_blockLSE_correct        ← whole-grid per-block blockLSE
  └─ logsumexp_fwd_kernel_correct_full            (shared masked-lane core)
```

The headline is the two-axis masked Hoare-triple combinator
`logsumexpIO … ⊨ f` (`Masked2DKernelIO₁.Implements`, pid-aware spec): for every
disjoint flat placement of the two buffers, every program `(i_n, i_d)` whose
active read lanes and scalar store cell are in bounds, and every launch state
whose active input lanes hold `xs`, the translated pointer kernel terminates,
the scalar cell `z[i_n·cdiv(D,B) + i_d]` holds the per-block spec value, and
every other memory cell is unchanged. The spec is `blockLSE_local`, the exact
log-sum-exp over the block's *active* lanes — a pure function of the loaded
block tile; the tiled row-level vocabulary (`validLanes` / `partialLSE_full` /
`blockLSE`, from `VeriTile.Triton.Semantics.TiledIndexing` under the
`TiledLogSumExp` oracle) is *not* inline-duplicated here.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` and the
`@triton.heuristics` `HAS_SCALE` selection are not modeled (`HAS_SCALE` is a
plain `Bool` parameter, both branches proved). Out-of-range lanes in the tail
block are loaded with `other = -float("inf")` (= `⊥ : WithBot ℝ`); the reduction
is proved to depend only on the valid lanes (`withBot_sup'_partial`,
`sum_exp_masked_eq`), so padded-block values never contaminate the result. An
*all*-masked program (`i_d·B ≥ D`, only reachable outside the real launch grid)
computes `b_z = ⊥` and its unmasked store writes the IEEE-faithful finite
fallback `0` (`FloatDType.storeValue ⊥ = 0`); the headline covers this honestly
via the pid-aware spec's `else 0` branch. The `.to(tl.int64)` casts on the
program ids erase to identity at the algorithm layer.
-/

namespace VeriTile.Bench.TritonBenchG.LogsumexpFwd

open VeriTile.Triton VeriTile.Triton.TiledLogSumExp
open scoped VeriTile.Triton.Masked2DKernelIO₁

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

/-- **Whole Triton-grid block-LSE correctness for `logsumexp_fwd_kernel`.**

For launch grid `(N, cdiv(D, B))`, every program instance `(i_n, i_d)` writes
the correct standard mathematical `blockLSE` for its row/block to
`z[i_n * cdiv(D, B) + i_d]`. This theorem covers both full blocks and the tail
block through the masked-lane theorem above. It deliberately stops at the
Triton kernel boundary; it does not model the Python wrapper's later
`z.logsumexp(-1)`.

Note: until VeriTile has a whole-grid `launchExec` returning a single merged
final state, this theorem is stated in the per-program-local form
`Kernel.ForAllProgramsSome`; the flat-memory per-program headline is the
`⊨` spec at the end of the file. -/
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

/-! ### The `⊨` specification -/

/-- Exact log-sum-exp over the **active lanes** of one `B`-sized block: lane
`j` of block `i_d` is active when `i_d * B + j < D` (the kernel's load mask).
This is the pure, block-local form of the tiled `blockLSE`; the bridge below
proves the two agree on loaded blocks. -/
noncomputable def blockLSE_local (D B i_d : Nat) (HAS_SCALE : Bool) (scale : ℝ)
    (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ i ∈ Finset.univ.filter (fun i : Fin B => i_d * B + i.val < D),
    Real.exp (if HAS_SCALE then xs i * scale else xs i))

/-- Row-level `partialLSE_full` at a row agreeing with the block tile `xs` on
the block's active lanes equals the block-local `blockLSE_local xs`. -/
private theorem partialLSE_full_eq_blockLSE_local
    {D : Nat} (n : Nat) (i_d : Nat) (h_tail : i_d * (n+1) < D)
    (HAS_SCALE : Bool) (scale : ℝ)
    (xsRow : Fin D → ℝ) (xs : Fin (n+1) → ℝ)
    (h : ∀ (j : Fin (n+1)) (hj : i_d * (n+1) + j.val < D),
      xsRow ⟨i_d * (n+1) + j.val, hj⟩ = xs j) :
    partialLSE_full xsRow i_d h_tail HAS_SCALE scale
      = blockLSE_local D (n+1) i_d HAS_SCALE scale xs := by
  rw [partialLSE_full_eq_blockLSE]
  unfold TiledLogSumExp.blockLSE blockLSE_local
  congr 1
  apply Finset.sum_congr
  · rfl
  · intro i hi
    have hmem : i_d * (n+1) + i.val < D := (Finset.mem_filter.mp hi).2
    unfold scaledLane_full
    simp only [dif_pos hmem, h i hmem]

set_option maxHeartbeats 1600000 in
/-- **All-masked-block behavior for `logsumexp_fwd_kernel`.**

When no lane of the block is in range (`¬ i_d * B < D`, only reachable for
programs outside the real launch grid), the load yields the all-`⊥` tile,
`b_m = ⊥`, hence `b_z = log(…) + ⊥ = ⊥`, and the unmasked store writes the
IEEE-faithful finite fallback `0` (`FloatDType.storeValue ⊥ = 0`). -/
private theorem logsumexp_fwd_kernel_store_zero
    (x z : RegionName)
    {D : Nat} (n : Nat) (HAS_SCALE : Bool) (scale : ℝ)
    (s : BlockState)
    (h_out : ¬ s.pids 1 * (n + 1) < D) :
    (exec (logsumexp_fwd_kernel x z D (n+1) HAS_SCALE scale) s).map
        (·.readMem z (s.pids 0 * ((D + n) / (n+1)) + s.pids 1)) =
      some 0 := by
  have hno : ∀ i : Fin (n+1), ¬ (s.pids 1 * (n+1) + i.val < D) := fun i hlt =>
    h_out (Nat.lt_of_le_of_lt (Nat.le_add_right _ _) hlt)
  rcases HAS_SCALE with _ | _
  · -- HAS_SCALE = false
    simp [exec, logsumexp_fwd_kernel, stepStmts, stepStmt, evalOp.eq_def,
      Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
      Tile.reduceMax, Tile.reduceMaxDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt]
    simp only [← Int.natCast_one, ← Int.natCast_add, ← Int.natCast_mul,
      ← Int.natCast_ediv, Int.toNat_natCast, Int.ofNat_lt, if_true]
    simp [hno, Finset.sup'_const]
    exact Or.inr rfl
  · -- HAS_SCALE = true
    simp [exec, logsumexp_fwd_kernel, stepStmts, stepStmt, evalOp.eq_def,
      Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
      Tile.reduceMax, Tile.reduceMaxDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt]
    simp only [← Int.natCast_one, ← Int.natCast_add, ← Int.natCast_mul,
      ← Int.natCast_ediv, Int.toNat_natCast, Int.ofNat_lt, if_true]
    simp [hno, Finset.sup'_const]
    exact Or.inr rfl

set_option maxHeartbeats 1600000 in
/-- Frame half: the single-cell scalar store at `z[i_n·cdiv(D,B) + i_d]`
leaves every other memory cell unchanged. -/
private theorem logsumexp_fwd_kernel_frame
    (x z : RegionName)
    {D : Nat} (n : Nat) (HAS_SCALE : Bool) (scale : ℝ)
    (s s1 : BlockState)
    (hExec : exec ((logsumexp_fwd_kernel x z D (n+1) HAS_SCALE scale).toAlgKernel) s
      = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ¬(z = r ∧ s.pids 0 * ((D + n) / (n+1)) + s.pids 1 = o)) :
    s1.mem r o = s.mem r o := by
  rcases HAS_SCALE with _ | _
  · simp [exec, logsumexp_fwd_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
      evalOp.eq_def, Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
      Tile.reduceMax, Tile.reduceMaxDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt] at hExec
    subst hExec
    simp only [← Int.natCast_one, ← Int.natCast_add, ← Int.natCast_mul,
      ← Int.natCast_ediv, Int.toNat_natCast]
    rw [BlockState.writeMem_mem]
    exact if_neg fun hc => hmiss ⟨hc.1.symm, hc.2.symm⟩
  · simp [exec, logsumexp_fwd_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
      evalOp.eq_def, Tile.bop, Tile.uop, Tile.cop, Tile.reduceSum, Tile.reduceSumDrop,
      Tile.reduceMax, Tile.reduceMaxDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt] at hExec
    subst hExec
    simp only [← Int.natCast_one, ← Int.natCast_add, ← Int.natCast_mul,
      ← Int.natCast_ediv, Int.toNat_natCast]
    rw [BlockState.writeMem_mem]
    exact if_neg fun hc => hmiss ⟨hc.1.symm, hc.2.symm⟩

/-- **The region-model masked Hoare triple** — termination, the scalar output
cell's per-block value (in-grid blocks: the active-lane `blockLSE_local`;
all-masked blocks: the `⊥`-store fallback `0`), and frame off the store cell,
from any launch state whose **active** input lanes are loaded. This is the
`hrun` obligation of the `⊨` headline. -/
theorem logsumexp_fwd_kernel_region_run
    (x z : RegionName)
    {D : Nat} (n : Nat) (HAS_SCALE : Bool) (scale : ℝ)
    (s₀ : BlockState) (xs : Fin (n+1) → ℝ)
    (hx : ∀ j : Fin (n+1), s₀.pids 1 * (n+1) + j.val < D →
      s₀.readMem x (s₀.pids 0 * D + (s₀.pids 1 * (n+1) + j.val)) = xs j) :
    ∃ s1, exec ((logsumexp_fwd_kernel x z D (n+1) HAS_SCALE scale).toAlgKernel) s₀
        = some s1
      ∧ s1.readMem z (s₀.pids 0 * ((D + n) / (n+1)) + s₀.pids 1)
          = (if s₀.pids 1 * (n+1) < D
             then blockLSE_local D (n+1) (s₀.pids 1) HAS_SCALE scale xs
             else 0)
      ∧ (∀ r o, (r ≠ z ∨ o ≠ s₀.pids 0 * ((D + n) / (n+1)) + s₀.pids 1) →
          s1.mem r o = s₀.mem r o) := by
  have hview : (exec (logsumexp_fwd_kernel x z D (n+1) HAS_SCALE scale) s₀).map
      (·.readMem z (s₀.pids 0 * ((D + n) / (n+1)) + s₀.pids 1))
      = some (if s₀.pids 1 * (n+1) < D
              then blockLSE_local D (n+1) (s₀.pids 1) HAS_SCALE scale xs
              else 0) := by
    by_cases h_tail : s₀.pids 1 * (n+1) < D
    · rw [if_pos h_tail]
      rw [logsumexp_fwd_kernel_correct_full x z n HAS_SCALE scale s₀
        (fun k : Fin D => s₀.readMem x (s₀.pids 0 * D + k.val)) (fun _ => rfl) h_tail]
      rw [partialLSE_full_eq_blockLSE_local n (s₀.pids 1) h_tail HAS_SCALE scale
        _ xs (fun j hj => hx j hj)]
    · rw [if_neg h_tail]
      exact logsumexp_fwd_kernel_store_zero x z n HAS_SCALE scale s₀ h_tail
  cases hExec : exec ((logsumexp_fwd_kernel x z D (n+1) HAS_SCALE scale).toAlgKernel) s₀ with
  | none => rw [hExec] at hview; simp at hview
  | some s1 =>
      rw [hExec] at hview
      refine ⟨s1, rfl, by simpa using hview, fun r o hro => ?_⟩
      refine logsumexp_fwd_kernel_frame x z n HAS_SCALE scale s₀ s1 hExec r o ?_
      rintro ⟨hr, ho⟩
      rcases hro with hne | hno
      · exact hne hr.symm
      · exact hno ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: the statements are memory-silent (program ids,
index staging, the reductions, the register arithmetic) except the masked row
load and the unmasked scalar store; the walk reduces them to the **lane-wise**
bounds hypotheses: every *active* load lane's address and the single store
cell are below the region bounds. -/
theorem logsumexp_fwd_kernel_traceSafe
    (x z : RegionName)
    {D : Nat} (n : Nat) (HAS_SCALE : Bool) (scale : ℝ)
    (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin (n+1), s.pids 1 * (n+1) + j.val < D →
      s.pids 0 * D + (s.pids 1 * (n+1) + j.val) < bounds x)
    (hout : s.pids 0 * ((D + n) / (n+1)) + s.pids 1 < bounds z) :
    Kernel.TraceSafe bounds
      ((logsumexp_fwd_kernel x z D (n+1) HAS_SCALE scale).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  rcases HAS_SCALE with _ | _
  · simp [logsumexp_fwd_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
      MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
      MaskOpt.Active, BlockState.setReg,
      Tile.bop, Tile.cop, Tile.uop,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt,
      Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
    simp only [← Int.natCast_one, ← Int.natCast_add, ← Int.natCast_mul,
      ← Int.natCast_ediv, Int.toNat_natCast, Int.ofNat_lt]
    exact ⟨fun a ha => hin a ha, hout⟩
  · simp [logsumexp_fwd_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
      MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
      MaskOpt.Active, BlockState.setReg,
      Tile.bop, Tile.cop, Tile.uop,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt,
      Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
    simp only [← Int.natCast_one, ← Int.natCast_add, ← Int.natCast_mul,
      ← Int.natCast_ediv, Int.toNat_natCast, Int.ofNat_lt]
    exact ⟨fun a ha => hin a ha, hout⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, casts, masked load with `other`, reductions, gated register
arithmetic, scalar store). -/
theorem logsumexp_fwd_kernel_flattenOk
    (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) (scale : ℝ) :
    ((logsumexp_fwd_kernel x z D B HAS_SCALE scale).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [logsumexp_fwd_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- `logsumexp_fwd_kernel`'s masked two-axis **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B` — the block window each program owns;
* `read` — program `(i_n, i_d)`'s lane `j` reads `x[i_n·D + i_d·B + j]`
  (row-major rows of length `D`, `B`-sized blocks along the row);
* `write` — the **scalar** store cell `z[i_n·cdiv(D,B) + i_d]`, the same for
  every lane;
* `mask` — the active read lanes `i_d·B + j < D`: the part of the block that
  actually lies inside the row;
* `writeMask` — lane `0` carries the scalar; the other lanes are
  write-inactive and carry no obligations on either side.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def logsumexpIO (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) (scale : ℝ) :
    Masked2DKernelIO₁ where
  kernel := logsumexp_fwd_kernel x z D B HAS_SCALE scale
  inp := x
  out := z
  B := B
  read := fun i_n i_d j => i_n * D + (i_d * B + j.val)
  write := fun i_n i_d _ => i_n * ((D + B - 1) / B) + i_d
  mask := fun _ i_d j => i_d * B + j.val < D
  writeMask := fun _ _ j => j.val = 0

/-- **The headline**: `logsumexp_fwd_kernel` implements the per-block
log-sum-exp on its masked two-axis IO signature — for every disjoint flat
placement of the two buffers, every program `(i_n, i_d)` whose active lanes
and store cell are in bounds, and every launch state whose active input lanes
hold `xs`, the translated pointer kernel terminates, the scalar cell
`z[i_n·cdiv(D,B) + i_d]` holds `blockLSE_local D B i_d HAS_SCALE scale xs`
(the exact LSE over the block's active lanes) when the block meets the row
(`i_d·B < D`), and the `⊥`-store fallback `0` otherwise (programs outside the
real launch grid), and every other memory cell is unchanged. `0 < B` is
required: the kernel's `max` reduce (like `Finset.sup'`) is only defined on
non-empty tiles. Proof: `Implements.intro` assembles the region-model masked
triple with the bridge side conditions. -/
specification logsumexp_fwd_kernel_correctness
    (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) (scale : ℝ)
    (hB : 0 < B) :
    logsumexpIO x z D B HAS_SCALE scale ⊨
      fun _ i_d xs _ =>
        if i_d * B < D then blockLSE_local D B i_d HAS_SCALE scale xs else 0 := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  refine Masked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact logsumexp_fwd_kernel_flattenOk x z D (n+1) HAS_SCALE scale
  · intro bounds s h1 h2 _
    exact logsumexp_fwd_kernel_traceSafe x z n HAS_SCALE scale bounds s
      (fun j hj => h1 j hj) (h2 ⟨0, Nat.succ_pos n⟩ rfl)
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      logsumexp_fwd_kernel_region_run x z n HAS_SCALE scale s₀ xs
        (fun j hj => hx j hj)
    -- scratch is empty, so its frame side condition is vacuous
    refine ⟨s1, hexec, fun j _ => hval, fun r o hout _ => ?_⟩
    refine hframe r o ?_
    rcases hout with hne | hno
    · exact Or.inl hne
    · exact Or.inr (hno ⟨0, Nat.succ_pos n⟩ rfl)

end VeriTile.Bench.TritonBenchG.LogsumexpFwd
