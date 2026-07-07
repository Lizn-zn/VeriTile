import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL
import VeriTile.Kernel.LoopInvariant

/-!
# `softmax_optimize` — strict per-kernel correctness

`softmax_kernel_online_v2` is an online (streaming) row softmax: program `pid_m`
sweeps a row of `N` columns in `TILE_N`-wide tiles, maintaining a running max `m`
and running normalizer `z` (the standard online-softmax recurrence), finalizes
`m`/`z`, then sweeps again to write `exp(inp - m) / z` per column.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`softmax_kernel_online_v2[(M,1,1)](...)`, the grid size,
the host-side `TILE_N = min(4096, next_power_of_2(N))` choice, and how the
runtime composes per-program writes into one buffer) is the *trusted boundary*,
not a proof obligation here. Because `pid_m` (= `s.pid`) is universally
quantified, the per-program statement covers every row of the grid.

## Proof architecture

```
softmax_kernel_online_v2_output_summary                ★ MAIN THEOREM (ComputeCorrect.Realizes headline)
  └─ softmax_kernel_online_v2_surface_exec_correct      internal engine lemma (full multi-tile surface)
       ├─ pass1LoopA_run / pass1LoopB_run               pass-1 online recurrence over all tiles
       ├─ reduction_step                                cross-lane (max, denom) collapse
       └─ pass2LoopA_run / pass2LoopB_run               pass-2 stores `softmaxOptimizeFullSpec`

softmax_kernel_online_v2_one_tile_output_summary       slice corollary (one-tile `N ≤ TILE_N` regime)
  ├─ (toAlgorithm? = Except.ok _)                       slice lowers to the algorithm layer
  └─ softmax_kernel_online_v2_one_tile_compute_correct  ← ComputeCorrect over the masked store
       └─ softmax_kernel_online_v2_one_tile_correct     ← algorithm-layer readback per lane

softmax_kernel_online_v2_surface_toAlgorithm_supported  full surface lowers (lowering only)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. Full-surface exec-level value correctness **is** established: the full
online-recurrence surface `softmax_kernel_online_v2_surface` is proved to write
the genuine full-row softmax `softmaxOptimizeFullSpec` to every active output
column. The public headline is `softmax_kernel_online_v2_output_summary`, stated
via `ComputeCorrect.Realizes` over the total output write map; it wraps the
exec-level engine lemma `softmax_kernel_online_v2_surface_exec_correct` (the
lowering fact `..._surface_toAlgorithm_supported` is a supporting layer). Cellwise value
correctness is also proved against the one-tile
specialization `softmax_kernel_online_v2_one_tile` (the `N ≤ TILE_N` regime where
the online loops collapse to a single masked tile), whose `softmaxOptimizeSpec`
is the stable softmax: `reduceMax` over the padded tile, `exp(row - rowMax)`,
`reduceSum`, then division. Masked lanes are `⊥` matching
`other=-float("inf")`; out-of-bounds lanes (`i ≥ N`) are preserved. The
`.to(output_ptr.dtype.element_ty)` cast is the identity post-erasure. The spec
references `Tile.reduceMax` / `Tile.reduceSum` directly, not
`VeriTile.Math.*`.
-/

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize

open VeriTile

set_option linter.unusedVariables false

/-- Faithful transcription of `softmax_optimize.py`'s
`softmax_kernel_online_v2`.

This keeps the online max/sum recurrence over full tiles, the masked tail pass,
the final scalar normalization, and the two writeback loops. -/
def softmax_kernel_online_v2_surface
    (output_ptr input_ptr : RegionName)
    (M N TILE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  m = tl.full([$(TILE_N)], value=-float("inf"), dtype=output_ptr.dtype.element_ty)
  z = tl.full([$(TILE_N)], value=0, dtype=output_ptr.dtype.element_ty)
  prev_multiple = $((N + TILE_N - 1)) // $(TILE_N) * $(TILE_N) - $(TILE_N)
  for start_n in range($(0), prev_multiple, $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    inp = (tl.load(input_ptrs)).to(output_ptr.dtype.element_ty)
    new_m = tl.maximum(m, inp)
    new_z = tl.exp(m - new_m) * z + tl.exp(inp - new_m)
    m = new_m
    z = new_z
  }
  for start_n in range(prev_multiple, $(N), $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    mask = n_offsets < $(N)
    inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(output_ptr.dtype.element_ty)
    new_m = tl.maximum(m, inp)
    new_z = tl.exp(m - new_m) * z + tl.exp(inp - new_m)
    m = new_m
    z = new_z
  }
  final_m = tl.max(m, 0)
  z = tl.sum(tl.exp(m - final_m) * z)
  m = final_m

  prev_multiple = $((N + TILE_N - 1)) // $(TILE_N) * $(TILE_N) - $(TILE_N)
  for start_n in range($(0), prev_multiple, $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    inp = (tl.load(input_ptrs)).to(output_ptr.dtype.element_ty)
    e = tl.exp(inp - m)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out)
  }
  for start_n in range(prev_multiple, $(N), $(TILE_N)) {
    n_offsets = start_n + tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    mask = n_offsets < $(N)
    inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(output_ptr.dtype.element_ty)
    e = tl.exp(inp - m)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out, mask=mask)
  }
}

/-- The full online-softmax surface lowers to the algorithm layer, including
the online recurrence loops, masked tail pass, final normalization, and both
writeback loops. -/
theorem softmax_kernel_online_v2_surface_toAlgorithm_supported
    (output_ptr input_ptr : RegionName)
    (M N TILE_N : Nat) :
    ∃ alg,
      (softmax_kernel_online_v2_surface output_ptr input_ptr M N TILE_N).toAlgorithm? =
        Except.ok alg := by
  simp [softmax_kernel_online_v2_surface, ComputeExpr.toAlgorithm?]

/-- Proof-oriented one-tile specialization of `softmax_kernel_online_v2`.

When `N <= TILE_N`, the online loops collapse to one masked row tile. This kernel
is kept as the small executable target for the existing algorithm proof. -/
def softmax_kernel_online_v2_one_tile
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  n_offsets = tl.arange(0, $(TILE_N))
  offset = pid_m * $(N) + n_offsets
  mask = n_offsets < $(N)
  input_ptrs = input_ptr + offset
  inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(output_ptr.dtype.element_ty)
  m = tl.max(inp, 0)
  e = tl.exp(inp - m)
  z = tl.sum(e, 0)
  out = e / z
  output_ptrs = output_ptr + offset
  tl.store(output_ptrs, out, mask=mask)
}


noncomputable def softmaxOptimizeInputTile
    (s : BlockState) (input_ptr : RegionName) (N TILE_N : Nat) :
    Tile .real [TILE_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem input_ptr (linearOffset s N idx.1))
      else none }

noncomputable def softmaxOptimizeSpec
    (s : BlockState) (input_ptr : RegionName)
    (N TILE_N : Nat) (idx : Fin TILE_N) : ℝ :=
  let row := softmaxOptimizeInputTile s input_ptr N TILE_N
  match Tile.reduceMax (shape := [TILE_N]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let e := Tile.uop WithBot.realExp shifted
      let z := Tile.reduceSum (shape := [TILE_N]) ⟨0, by simp⟩ Bool.false e
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR e z).data
          (idx, PUnit.unit))
  | none => 0

/-- Algorithm-layer correctness for the one-tile optimized softmax slice. -/
theorem softmax_kernel_online_v2_one_tile_correct
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat)
    (s s' : BlockState)
    (hExec : exec (softmax_kernel_online_v2_one_tile output_ptr input_ptr N TILE_N)
        s = some s') :
    ∀ i : Fin TILE_N,
      s'.readMem output_ptr (linearOffset s N i) =
        if i.val < N then
          softmaxOptimizeSpec s input_ptr N TILE_N i
        else s.readMem output_ptr (linearOffset s N i) := by
  intro i
  by_cases hT : 0 < TILE_N
  · simp [exec, softmax_kernel_online_v2_one_tile, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
          ComparableDType.lt, hT] at hExec
    subst s'
    simp [BlockState.pid_eq, linearOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, softmaxOptimizeSpec, softmaxOptimizeInputTile, linearOffset,
            Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum,
            Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
            TileShape.insertAxisIndex, hT]
      congr
    · simp [hi]
  · exact False.elim (hT (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-tile optimized softmax slice. -/
theorem softmax_kernel_online_v2_one_tile_compute_correct
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := softmax_kernel_online_v2_one_tile output_ptr input_ptr N TILE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin TILE_N => i.val < N)
        (fun i => (output_ptr, linearOffset s N i)))
      (expected := fun i => softmaxOptimizeSpec s input_ptr N TILE_N i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_kernel_online_v2_one_tile]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := softmax_kernel_online_v2_one_tile_correct output_ptr input_ptr N TILE_N
    s s' hExec i
  simpa [hActive] using h

/-- Per-kernel output summary for the one-tile slice of
`softmax_kernel_online_v2`: the slice lowers to the algorithm layer, and the
masked store to `output_ptr` is compute-correct — every active lane (`i < N`)
holds the stable-softmax value `softmaxOptimizeSpec`, out-of-bounds lanes are
preserved. (Covers the `N ≤ TILE_N` collapsed-loop regime; the full online
surface's exec-level value correctness is established separately, see
`softmax_kernel_online_v2_surface_exec_correct`.) -/
theorem softmax_kernel_online_v2_one_tile_output_summary
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat)
    (s : BlockState) :
    (∃ alg, (softmax_kernel_online_v2_one_tile output_ptr input_ptr N TILE_N).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := softmax_kernel_online_v2_one_tile output_ptr input_ptr N TILE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin TILE_N => i.val < N)
        (fun i => (output_ptr, linearOffset s N i)))
      (expected := fun i => softmaxOptimizeSpec s input_ptr N TILE_N i) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact softmax_kernel_online_v2_one_tile_compute_correct output_ptr input_ptr N TILE_N s

/-! ## Full multi-tile streaming softmax: genuine full-row spec and proof

The `softmax_kernel_online_v2_surface` kernel computes, for row `pid_m`, the
numerically-stabilized full-row softmax over all `N` columns. It does so by a
*per-lane* online recurrence: it sweeps the row in `TILE_N`-wide tiles keeping
`[TILE_N]`-vectors `m`/`z`, so that after the pass-1 loops, lane `ℓ` holds the
online-softmax `(max, denom)` state of the values `{input[t·TILE_N + ℓ]}` at
within-tile position `ℓ`. The reduction then collapses the lanes
(`final_m = max_ℓ m[ℓ]`, `z = Σ_ℓ exp(m[ℓ]-final_m)·z[ℓ]`), giving the global
max and the global denominator `Σ_{j<N} exp(input[j] - rowMax)`. Pass-2 writes
`exp(input[j] - rowMax) / z`.

The genuine spec below is the standard full-row softmax of the loaded input row
— a function of memory, never self-referential.

**Proven here (sorry-free):** the genuine spec `softmaxOptimizeFullSpec`; the
per-lane online-softmax `(max, denom)` recurrence `soStep` and its `⊥`-seeded
denominator-consistency theorem `soStep_foldl_consistent` (the load-bearing
online-softmax identity that the kernel's `exp(m-new_m)` rescaling realizes); the
per-lane streaming-state model (`laneScores`/`laneState`) over the kernel's
tiling; and `laneState_denom_closed`, the per-lane closed form
`z[ℓ] = κ(m[ℓ])·Σ exp` used to collapse the cross-lane reduction.

**Pass-1 exec stepping now proven sorry-free (both loops):** the per-statement
full-tile body step `loopA_body_step` and masked-tail body step `loopB_body_step`
(each running the eight/nine kernel statements through `stepStmts`), and the two
whole-loop invariant advancements `pass1LoopA_run` / `pass1LoopB_run` via
`forRangeDyn_inv` — so after both pass-1 loops the `m`/`z` registers provably hold
the per-lane `laneState` over all `⌈N/TILE_N⌉` tiles (`pass1InvA`). Supporting:
`loopA_inRange` (full-tile columns are in range), `loopA_bodyMv_eq` /
`loopB_bodyMvB_eq` (the body advances `laneState` by one tile, masked-out lanes
being `⊥`-load no-ops), and `laneState_congr` (state depends only on `pid`/`mem`).

**Full-surface top theorem complete (sorry-free):**
`softmax_kernel_online_v2_surface_exec_correct` assembles the seed, both pass-1
loops (`pass1LoopA_run`/`pass1LoopB_run`), the `tl.max`/`tl.sum` cross-lane
reduction collapse to `∑ over Fin N` (`reduction_step`, using the
lane×tile→`Fin N` partition `lane_tile_partition`), and the two
pass-2 store loops (`pass2LoopA_run`/`pass2LoopB_run`) with their
scatter-readback — establishing that the full `softmax_kernel_online_v2_surface`
writes `softmaxOptimizeFullSpec` to every active output column. -/

/-- The loaded input row as a function of the genuine column index `j : Fin N`:
`input[pid_m, j]` at memory address `linearOffset s N j = s.pid·N + j`. -/
noncomputable def softmaxOptimizeRow
    (s : BlockState) (input_ptr : RegionName) (N : Nat) (j : Fin N) : ℝ :=
  s.readMem input_ptr (linearOffset s N j)

/-- **Genuine full-row softmax closed form.** For column `j < N`, the standard
numerically-stabilized softmax over the `N` loaded inputs:
`exp(input[j] - rowMax) / Σ_{j'<N} exp(input[j'] - rowMax)`, where
`rowMax = max_{j'<N} input[j']`. This is exactly `TiledSoftmax.naiveSpec`
(`= stableSpec`, the numerically stabilized form). It is a function of the
loaded `input` memory, not of the kernel's own output. -/
noncomputable def softmaxOptimizeFullSpec
    (s : BlockState) (input_ptr : RegionName) (N : Nat) (j : Fin N) : ℝ :=
  Real.exp (softmaxOptimizeRow s input_ptr N j)
    / ∑ j' : Fin N, Real.exp (softmaxOptimizeRow s input_ptr N j')

/-! ### Per-lane online-softmax recurrence (running max + denominator)

The pass-1 loop body, restricted to a single lane `ℓ`, runs the recurrence
`m' = max(m, x); z' = exp(m - m')·z + exp(x - m')` with the convention
`exp(⊥) = 0`. We model the running max in `WithBot ℝ` (seed `⊥ = -∞`) and the
running denominator in `ℝ` (seed `0`). -/

/-- One online-softmax `(max, denom)` update absorbing score `x`. Mirrors the
kernel body `new_m = max(m,x); new_z = exp(m-new_m)·z + exp(x-new_m)`. -/
noncomputable def soStep (st : WithBot ℝ × ℝ) (x : ℝ) : WithBot ℝ × ℝ :=
  let m := st.1; let z := st.2
  let m' := m ⊔ ((x : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp (WithBot.realSub m m')).unbotD 0
  (m', z * α + Real.exp (x - m'.unbotD 0))

/-- The running max after folding `soStep` over a list of scores: it is the
`WithBot`-supremum of the seed and the scores. -/
theorem soStep_foldl_fst (xs : List ℝ) (m : WithBot ℝ) (z : ℝ) :
    (xs.foldl soStep (m, z)).1 = xs.foldl (fun a x => a ⊔ ((x : ℝ) : WithBot ℝ)) m := by
  induction xs generalizing m z with
  | nil => rfl
  | cons x xs ih => simpa [soStep] using ih (m ⊔ ((x : ℝ) : WithBot ℝ)) _

/-- **⊥-seeded denominator consistency.** Folding `soStep` from a state with
`z = κ(m)·L` (where `κ ⊥ = 0`, `κ (some r) = exp(-r)`, and `L` is the running
true denominator `Σ exp(x)`) keeps that invariant: the final
`z = κ(m_final)·(L + Σ_{x∈xs} exp x)`, with `m_final = ⊔ m xs`. -/
theorem soStep_foldl_consistent
    (xs : List ℝ) (m : WithBot ℝ) (z L : ℝ)
    (hz : z = (m.elim 0 (fun r => Real.exp (-r))) * L)
    (hmL : m = ⊥ → L = 0) :
    let st := xs.foldl soStep (m, z)
    let L' := L + (xs.map Real.exp).sum
    st.2 = (st.1.elim 0 (fun r => Real.exp (-r))) * L'
      ∧ (st.1 = ⊥ → L' = 0) := by
  induction xs generalizing m z L with
  | nil =>
    refine ⟨by simpa using hz, ?_⟩
    simpa using hmL
  | cons x xs ih =>
    simp only [List.foldl_cons, List.map_cons, List.sum_cons]
    -- one soStep then recurse
    set m' : WithBot ℝ := m ⊔ ((x : ℝ) : WithBot ℝ) with hm'
    -- m' is `some` (it is ≥ some x), so the new state is never ⊥ and L' > 0-side is fine
    have hm'def : m' = ((m.elim x (fun r => max r x) : ℝ) : WithBot ℝ) := by
      cases m with
      | bot => rw [hm']; simp [Option.elim]
      | coe r => rw [hm']; rfl
    have hz' : z * (WithBot.realExp (WithBot.realSub m m')).unbotD 0
          + Real.exp (x - m'.unbotD 0)
        = (m'.elim 0 (fun r => Real.exp (-r))) * (L + Real.exp x) := by
      cases m with
      | bot =>
        -- m' = some x, z = 0, L = 0
        have hL0 : L = 0 := hmL rfl
        rw [hz, hL0]
        rw [show m' = ((x : ℝ) : WithBot ℝ) from by rw [hm']; simp]
        simp only [WithBot.realSub, WithBot.unbotD, Option.elim, mul_zero, zero_mul,
          zero_add, WithBot.recBotCoe_coe, id_eq]
        rw [show x - x = (0:ℝ) from by ring, Real.exp_zero, Real.exp_neg, mul_comm,
          mul_inv_cancel₀ (Real.exp_ne_zero x)]
      | coe r =>
        have hmm : m' = ((max r x : ℝ) : WithBot ℝ) := by rw [hm']; rfl
        rw [hmm, hz]
        simp only [WithBot.realSub_coe_coe, WithBot.unbotD_coe]
        rw [show (((max r x : ℝ)) : WithBot ℝ).elim 0 (fun r => Real.exp (-r))
              = Real.exp (-(max r x)) from rfl]
        rw [show (((r : ℝ)) : WithBot ℝ).elim 0 (fun r => Real.exp (-r))
              = Real.exp (-r) from rfl]
        rw [show WithBot.realExp ((r - max r x : ℝ) : WithBot ℝ)
              = ((Real.exp (r - max r x) : ℝ) : WithBot ℝ) from rfl,
            WithBot.unbotD_coe]
        rw [mul_add]
        congr 1
        · rw [mul_comm (Real.exp (-r)), mul_assoc, ← Real.exp_add,
            show -r + (r - max r x) = -(max r x) from by ring, mul_comm]
        · rw [← Real.exp_add]; ring_nf
    have hm'L : m' = ⊥ → (L + Real.exp x) = 0 := by
      intro hbot; rw [hm'def] at hbot; exact absurd hbot (by simp)
    have hrec := ih m' (z * (WithBot.realExp (WithBot.realSub m m')).unbotD 0
          + Real.exp (x - m'.unbotD 0)) (L + Real.exp x) hz' hm'L
    simp only [soStep] at hrec ⊢
    have hassoc : L + (Real.exp x + (xs.map Real.exp).sum)
        = (L + Real.exp x) + (xs.map Real.exp).sum := by ring
    rw [hassoc]
    exact hrec

/-! ### Per-lane streaming state over the kernel's tiling

For lane `ℓ : Fin TILE_N`, the pass-1 loops absorb the scores
`input[pid_m, t·TILE_N + ℓ]` for tiles `t` with `t·TILE_N + ℓ < N`. After `c`
tiles, the running `(max, denom)` state is the `soStep`-fold over the first `c`
such scores. `laneScores` is the list of these scores in tile order. -/

/-- Score absorbed at lane `ℓ` in tile `t`: `input[pid_m, t·TILE_N + ℓ]` if the
column is in range, dropped otherwise (handled via `List.filterMap`). -/
noncomputable def laneScoreAt
    (s : BlockState) (input_ptr : RegionName) (N TILE_N : Nat)
    (ℓ : Fin TILE_N) (t : Nat) : Option ℝ :=
  if h : t * TILE_N + ℓ.val < N then
    some (s.readMem input_ptr (s.pid * N + (t * TILE_N + ℓ.val)))
  else none

/-- The list of scores absorbed at lane `ℓ` over the first `c` tiles. -/
noncomputable def laneScores
    (s : BlockState) (input_ptr : RegionName) (N TILE_N : Nat)
    (ℓ : Fin TILE_N) (c : Nat) : List ℝ :=
  (List.range c).filterMap (laneScoreAt s input_ptr N TILE_N ℓ)

/-- The per-lane `(max, denom)` streaming state after `c` tiles. -/
noncomputable def laneState
    (s : BlockState) (input_ptr : RegionName) (N TILE_N : Nat)
    (ℓ : Fin TILE_N) (c : Nat) : WithBot ℝ × ℝ :=
  (laneScores s input_ptr N TILE_N ℓ c).foldl soStep (⊥, 0)

/-! ### Pass-1 loop invariant -/

/-- Pass-1 invariant after counter `start_n = i` (i.e. `c = i / TILE_N` tiles).
The `m`/`z` registers hold the per-lane streaming state, `pid_m` is fixed, and
memory is unchanged. -/
noncomputable def pass1Inv
    (input_ptr : RegionName) (N TILE_N : Nat) (s0 : BlockState)
    (i : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ s.mem = s0.mem ∧ (∀ rg o, s.undef rg o = s0.undef rg o) ∧
  s.regs .nat [] "pid_m" = some (Tile.scalar (s0.pid)) ∧
  s.regs .real [TILE_N] "m" = some
    ⟨fun idx : TileIndex [TILE_N] =>
      (laneState s0 input_ptr N TILE_N idx.1 (i / TILE_N)).1⟩ ∧
  s.regs .real [TILE_N] "z" = some
    ⟨fun idx : TileIndex [TILE_N] =>
      (((laneState s0 input_ptr N TILE_N idx.1 (i / TILE_N)).2 : ℝ) : WithBot ℝ)⟩ ∧
  s.regs .nat [] "prev_multiple"
    = some (Tile.scalar ((N + TILE_N - 1) / TILE_N * TILE_N - TILE_N))

/-! ### Closed-form denominator: cross-lane reduction = full-row stable sum

The reduction `final_m = max_ℓ m[ℓ]`, `z = Σ_ℓ exp(m[ℓ] - final_m)·z[ℓ]` collapses
the per-lane streaming state to the global denominator. Using
`soStep_foldl_consistent` per lane, `exp(m[ℓ]-final_m)·z[ℓ] = exp(-final_m)·Σ_t exp(score)`,
and summing over lanes recovers `exp(-rowMax)·Σ_{j<N} exp(input[j])`, the
numerically-stabilized full-row denominator. The bridge below packages the
per-lane consistency for use in that collapse. -/

/-- Per-lane closed form for the running denominator after `c` tiles:
`z[ℓ] = κ(m[ℓ]) · Σ_{x ∈ laneScores} exp x`, where `κ ⊥ = 0`,
`κ (some r) = exp(-r)`. Direct corollary of `soStep_foldl_consistent`. -/
theorem laneState_denom_closed
    (s : BlockState) (input_ptr : RegionName) (N TILE_N : Nat)
    (ℓ : Fin TILE_N) (c : Nat) :
    (laneState s input_ptr N TILE_N ℓ c).2
      = ((laneState s input_ptr N TILE_N ℓ c).1.elim 0 (fun r => Real.exp (-r)))
        * ((laneScores s input_ptr N TILE_N ℓ c).map Real.exp).sum := by
  have h := soStep_foldl_consistent (laneScores s input_ptr N TILE_N ℓ c) ⊥ 0 0
    (by simp) (fun _ => rfl)
  simpa [laneState, Option.elim] using h.1

/-! ### exec-stepping: per-lane bridges and load eval helpers

Below we step the four `forRangeDyn` loops of `softmax_kernel_online_v2_surface`
through `forRangeDyn_inv`, advancing `pass1Inv` (pass 1) and a written-rows
invariant (pass 2). The first batch of lemmas are the per-statement / per-lane
bridges the loop bodies need. -/

/-- Unmasked load through a pointer register (`inp = tl.load(input_ptrs)` where
`input_ptrs : ptr` tile): lane `i` reads memory at the `(region, offset)` stored
in `input_ptrs[i]`. The pass-1/pass-2 full-tile loops use this shape. -/
theorem evalOp_load_ptrRef_none {dtype : TileDType} {n : Nat}
    (ptrName : RegName) (s : BlockState) (ptrTile : Tile .ptr [n])
    (hp : evalOp (Op.ref .ptr [n] ptrName) s = some ptrTile) :
    evalOp (.load dtype (MemAccess.ptr (Op.ref .ptr [n] ptrName)) MaskOpt.none) s
      = some ⟨fun i => s.readMemValue dtype (ptrTile.data i).1 (ptrTile.data i).2⟩ := by
  simp only [evalOp, hp, Option.bind_eq_bind, Option.bind_some, if_true]

/-- Masked load with `other` through a pointer register (`inp =
tl.load(input_ptrs, mask=m, other=o)`): lane `i` reads at `input_ptrs[i]` when
`m i`, else `o i`. The pass-1/pass-2 masked-tail loops use this shape. -/
theorem evalOp_load_ptrRef_maskOther {dtype : TileDType} {n : Nat}
    (ptrName : RegName) (mask : Op .bool [n]) (other : Op dtype [n]) (s : BlockState)
    (ptrTile : Tile .ptr [n]) (maskTile : Tile .bool [n]) (otherTile : Tile dtype [n])
    (hp : evalOp (Op.ref .ptr [n] ptrName) s = some ptrTile)
    (hmask : evalOp mask s = some maskTile)
    (hother : evalOp other s = some otherTile) :
    evalOp (.load dtype (MemAccess.ptr (Op.ref .ptr [n] ptrName))
        (MaskOpt.maskOther mask other)) s
      = some ⟨fun i => if maskTile.data i then
          s.readMemValue dtype (ptrTile.data i).1 (ptrTile.data i).2 else otherTile.data i⟩ := by
  simp only [evalOp, hp, hmask, hother, Option.bind_eq_bind, Option.bind_some, if_true]

/-- **Per-lane running-max bridge.** The kernel's `new_m = where(m > inp, m, inp)`
on lane `ℓ` (`m = m[ℓ] : WithBot ℝ`, `inp = some x`) equals `(soStep (m,z) x).1`. -/
theorem soStep_fst_match (m : WithBot ℝ) (zr x : ℝ) :
    (if ComparableDType.real.gt m (some x) = Bool.true then m else (some x : WithBot ℝ))
      = (soStep (m, zr) x).1 := by
  simp only [soStep]
  by_cases h : m > (some x : WithBot ℝ)
  · rw [if_pos (by simpa [ComparableDType.real_gt_eq_true] using h)]
    exact (sup_eq_left.mpr (le_of_lt h)).symm
  · rw [if_neg (by simpa [ComparableDType.real_gt_eq_true] using h)]
    exact (sup_eq_right.mpr (not_lt.mp h)).symm

/-- **Per-lane running-denominator bridge.** The kernel's
`new_z = exp(m - new_m)·z + exp(inp - new_m)` on lane `ℓ` (`m : WithBot ℝ`,
`z = some zr`, `inp = some x`, `new_m = m ⊔ some x`) equals `some (soStep (m,z) x).2`. -/
theorem soStep_snd_match (m : WithBot ℝ) (zr x : ℝ) :
    WithBot.realAdd
        (WithBot.realMul (WithBot.realExp (WithBot.realSub m (m ⊔ (some x)))) (some zr))
        (WithBot.realExp (WithBot.realSub (some x) (m ⊔ (some x))))
      = some (soStep (m, zr) x).2 := by
  simp only [soStep]
  cases m with
  | bot =>
    simp only [bot_sup_eq, WithBot.realSub, WithBot.realExp, WithBot.realMul, WithBot.realAdd,
      Option.map₂, Option.bind, WithBot.unbotD, WithBot.recBotCoe]
    norm_num
  | coe r =>
    have hmx : ((r:ℝ):WithBot ℝ) ⊔ (some x) = ((max r x : ℝ):WithBot ℝ) := rfl
    rw [hmx]
    show WithBot.realAdd
        (WithBot.realMul (WithBot.realExp (WithBot.realSub (↑r) (↑(max r x)))) (some zr))
        (WithBot.realExp (WithBot.realSub (↑x) (↑(max r x))))
      = some (zr * WithBot.unbotD 0 (WithBot.realExp (WithBot.realSub (↑r) (↑(max r x))))
          + Real.exp (x - WithBot.unbotD 0 (↑(max r x))))
    rw [WithBot.realSub_coe_coe, WithBot.realSub_coe_coe]
    rw [show WithBot.realExp ((r - max r x : ℝ):WithBot ℝ)
          = ((Real.exp (r - max r x):ℝ):WithBot ℝ) from rfl]
    rw [show WithBot.realExp ((x - max r x : ℝ):WithBot ℝ)
          = ((Real.exp (x - max r x):ℝ):WithBot ℝ) from rfl]
    rw [WithBot.unbotD_coe, WithBot.unbotD_coe]
    simp only [WithBot.realMul, WithBot.realAdd, Option.map₂, Option.bind, Option.map]
    ring

/-- **`laneState` one-tile recurrence.** Absorbing tile `c` either applies one
`soStep` (when column `c·TILE_N + ℓ < N`) or is a no-op (out of range). -/
theorem laneState_succ
    (s : BlockState) (input_ptr : RegionName) (N TILE_N : Nat)
    (ℓ : Fin TILE_N) (c : Nat) :
    laneState s input_ptr N TILE_N ℓ (c + 1)
      = (match laneScoreAt s input_ptr N TILE_N ℓ c with
         | some x => soStep (laneState s input_ptr N TILE_N ℓ c) x
         | none => laneState s input_ptr N TILE_N ℓ c) := by
  unfold laneState laneScores
  rw [List.range_succ, List.filterMap_append, List.filterMap_cons, List.filterMap_nil]
  cases laneScoreAt s input_ptr N TILE_N ℓ c <;> simp [List.foldl_append]

/-! ### Pass-1 full-tile loop body (loop A): per-statement exec stepping

`loopAbody` is the alg-erased body of the pass-1 full-tile `forRangeDyn`
(`for start_n in range(0, prev_multiple, TILE_N)`). `loopA_body_step` runs the
eight statements, advancing the per-lane `(m, z)` registers by one `soStep` per
lane against the loaded score `input[pid·N + (c·TILE_N + ℓ)]`. -/

/-- The alg-erased statement list of the pass-1 full-tile loop body. Matches the
`forRangeDyn` body produced by `softmax_kernel_online_v2_surface` loop A. -/
def loopAbody (input_ptr : RegionName) (N TILE_N : Nat) : List Stmt :=
  [ Stmt.assign .nat [TILE_N] "n_offsets"
      (Op.add NumericDType.nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.arange TILE_N)),
    Stmt.assign .nat [TILE_N] "offset"
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat N))
        (Op.ref .nat [TILE_N] "n_offsets")),
    Stmt.assign .ptr [TILE_N] "input_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase input_ptr) (Op.ref .nat [TILE_N] "offset")),
    Stmt.assign .real [TILE_N] "inp"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [TILE_N] "input_ptrs")) MaskOpt.none),
    Stmt.assign .real [TILE_N] "new_m"
      ((Op.gt ComparableDType.real Broadcast.nil.consSame (Op.ref .real [TILE_N] "m")
          (Op.ref .real [TILE_N] "inp")).where
        (Op.ref .real [TILE_N] "m") (Op.ref .real [TILE_N] "inp")),
    Stmt.assign .real [TILE_N] "new_z"
      (Op.add NumericDType.real Broadcast.nil.consSame
        (Op.mul NumericDType.real Broadcast.nil.consSame
          (Op.sub NumericDType.real Broadcast.nil.consSame (Op.ref .real [TILE_N] "m")
              (Op.ref .real [TILE_N] "new_m")).exp
          (Op.ref .real [TILE_N] "z"))
        (Op.sub NumericDType.real Broadcast.nil.consSame (Op.ref .real [TILE_N] "inp")
            (Op.ref .real [TILE_N] "new_m")).exp),
    Stmt.assign .real [TILE_N] "m" (Op.ref .real [TILE_N] "new_m"),
    Stmt.assign .real [TILE_N] "z" (Op.ref .real [TILE_N] "new_z") ]

/-- The lane-`idx` score absorbed at tile `c` in loop A: `input[pid·N + (c·TILE_N + ℓ)]`. -/
noncomputable def loopAscore (N TILE_N c : Nat) (s : BlockState) (input_ptr : RegionName)
    (idx : TileIndex [TILE_N]) : ℝ :=
  s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))

/-- Running max after the loop-A body, per lane: `(soStep (m,z) score).1`. -/
noncomputable def bodyMv (N TILE_N c : Nat) (s : BlockState) (input_ptr : RegionName)
    (mfun : TileIndex [TILE_N] → WithBot ℝ) (zfun : TileIndex [TILE_N] → ℝ)
    (idx : TileIndex [TILE_N]) : WithBot ℝ :=
  (soStep (mfun idx, zfun idx) (loopAscore N TILE_N c s input_ptr idx)).1

/-- Running denominator after the loop-A body, per lane: `(soStep (m,z) score).2`. -/
noncomputable def bodyZv (N TILE_N c : Nat) (s : BlockState) (input_ptr : RegionName)
    (mfun : TileIndex [TILE_N] → WithBot ℝ) (zfun : TileIndex [TILE_N] → ℝ)
    (idx : TileIndex [TILE_N]) : ℝ :=
  (soStep (mfun idx, zfun idx) (loopAscore N TILE_N c s input_ptr idx)).2

set_option maxHeartbeats 4000000 in
/-- **Loop-A body step.** Executing the eight statements of `loopAbody` with the
counter `start_n = c·TILE_N`, starting from registers `m = mfun`, `z = zfun`,
advances each lane by one `soStep` against the loaded score, leaving `pid_m`,
`pids`, `mem`, `undef` unchanged. -/
theorem loopA_body_step (N TILE_N : Nat) (input_ptr : RegionName) (c : Nat) (s : BlockState)
    (mfun : TileIndex [TILE_N] → WithBot ℝ) (zfun : TileIndex [TILE_N] → ℝ)
    (hpid : s.regs .nat [] "pid_m" = some (Tile.scalar (s.pid)))
    (hm : s.regs .real [TILE_N] "m" = some (Tile.mk mfun))
    (hz : s.regs .real [TILE_N] "z" = some (Tile.mk (fun idx => ((zfun idx : ℝ) : WithBot ℝ)))) :
    ∃ s', stepStmts (loopAbody input_ptr N TILE_N)
        (s.setReg "start_n" .nat [] (Tile.scalar (c * TILE_N))) = some s'
      ∧ s'.regs .nat [] "pid_m" = some (Tile.scalar (s.pid))
      ∧ s'.pids = s.pids ∧ s'.mem = s.mem ∧ (∀ rg o, s'.undef rg o = s.undef rg o)
      ∧ s'.regs .real [TILE_N] "m" = some (Tile.mk (bodyMv N TILE_N c s input_ptr mfun zfun))
      ∧ s'.regs .real [TILE_N] "z" = some
          (Tile.mk (fun idx => ((bodyZv N TILE_N c s input_ptr mfun zfun idx : ℝ) : WithBot ℝ)))
      ∧ s'.regs .nat [] "prev_multiple" = s.regs .nat [] "prev_multiple" := by
  simp only [loopAbody]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => c * TILE_N + idx.1.val)) (by
      rw [evalOp_add, evalOp_ref_setReg_same, evalOp_arange]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => s.pid * N + (c * TILE_N + idx.1.val))) (by
      rw [evalOp_add, evalOp_mul, evalOp_constNat]
      simp only [evalOp_ref, BlockState.setReg_ne_name, BlockState.setReg_same, hpid, ne_eq,
        String.reduceEq, not_false_eq_true]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      ((Region.cast input_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) (by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
      congr 1
      apply Tile.ext; intro idx
      show ((Region.cast input_ptr : RegionName), 0 + (s.pid * N + (c * TILE_N + idx.1.val))) = _
      rw [Nat.zero_add]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) : WithBot ℝ))) (by
      rw [evalOp_load_ptrRef_none "input_ptrs" _
        (Tile.mk (fun idx : TileIndex [TILE_N] =>
          ((Region.cast input_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val))))
        (by simp only [evalOp_ref, BlockState.setReg_same])]
      congr 1
      apply Tile.ext; intro idx
      simp only [Tile.mk]
      rw [BlockState.readMemValue_real]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (bodyMv N TILE_N c s input_ptr mfun zfun)) (by
      rw [evalOp_where, evalOp_gt]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind, hm]
      congr 1
      apply Tile.ext; intro idx
      simp only [Tile.select_data, Tile.cop_data, Tile.mk, bodyMv, loopAscore]
      exact soStep_fst_match (mfun _) (zfun _) _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx => ((bodyZv N TILE_N c s input_ptr mfun zfun idx : ℝ) : WithBot ℝ))) (by
      rw [evalOp_add, evalOp_mul, evalOp_exp, evalOp_sub, evalOp_exp, evalOp_sub]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind, hm, hz]
      congr 1
      apply Tile.ext; intro idx
      simp only [Tile.bop_data, Tile.uop_data, Tile.mk, bodyZv, bodyMv, loopAscore]
      exact soStep_snd_match (mfun _) (zfun _) _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (bodyMv N TILE_N c s input_ptr mfun zfun)) (by
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx => ((bodyZv N TILE_N c s input_ptr mfun zfun idx : ℝ) : WithBot ℝ))) (by
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hpid
  · rfl
  · rfl
  · intro rg o; rfl
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
  · simp only [BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]

/-! ### Loop-A range fact and `laneState` advancement

In loop A (`start_n < prev_multiple`, step-aligned), every lane column
`c·TILE_N + ℓ` is `< N`, so `laneScoreAt = some (loaded score)` and the body
advances `laneState` by exactly one `soStep`. -/

open VeriTile in
/-- **Aligned `forRangeAux` invariant principle.** When the counter is always a
multiple of `step` and `step ∣ stop`, the loop terminates with the counter
*exactly* at `stop` (not merely `≥ stop`), so the invariant holds at `stop`. -/
theorem forRangeAux_inv_aligned
    {idx : RegName} {stop step : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop}
    (hstep : step ≠ 0) (hdvdStop : step ∣ stop)
    (h_step :
      ∀ i s, i < stop → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + step) s') :
    ∀ cur s, cur ≤ stop → step ∣ cur → P cur s →
      ∃ s_final,
        stepForRangeAux idx cur stop step body s = some s_final ∧
        P stop s_final := by
  intro cur s
  induction h : (stop - cur) using Nat.strong_induction_on generalizing cur s with
  | _ k ih =>
    intro hle hdvdCur hP
    by_cases hdone : stop ≤ cur
    · have hcur : cur = stop := Nat.le_antisymm hle hdone
      subst hcur
      exact ⟨s, stepForRangeAux.step_ge hstep (le_refl _), hP⟩
    · have hlt : cur < stop := Nat.lt_of_not_ge hdone
      obtain ⟨s', h_body, hP'⟩ := h_step cur s hlt hP
      have hle' : cur + step ≤ stop := by
        rcases hdvdStop with ⟨a, ha⟩; rcases hdvdCur with ⟨b, hb⟩
        subst ha hb
        have : b < a := by
          rcases Nat.lt_or_ge b a with h' | h'
          · exact h'
          · exact absurd (Nat.mul_le_mul_left step h') (Nat.not_le.mpr hlt)
        calc step * b + step = step * (b + 1) := by ring
          _ ≤ step * a := Nat.mul_le_mul_left step this
      obtain ⟨s_final, h_aux, hPfinal⟩ :=
        ih (stop - (cur + step)) (by omega) (cur + step) s' rfl hle'
          (Dvd.dvd.add hdvdCur (dvd_refl _)) hP'
      refine ⟨s_final, ?_, hPfinal⟩
      rw [stepForRangeAux.step_lt hstep hlt, h_body]
      exact h_aux

open VeriTile in
/-- Aligned `forRangeDyn` invariant principle: the loop ends at `stop` exactly. -/
theorem forRangeDyn_inv_aligned
    {idx : RegName} {startOp stopOp stepOp : Op .nat []}
    {start stop step : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    (hStart : evalOp startOp s_init = some (Tile.scalar start))
    (hStop : evalOp stopOp s_init = some (Tile.scalar stop))
    (hStepOp : evalOp stepOp s_init = some (Tile.scalar step))
    (hstep : step ≠ 0) (hle : start ≤ stop)
    (hdvdStart : step ∣ start) (hdvdStop : step ∣ stop)
    (h_init : P start s_init)
    (h_step :
      ∀ i s, i < stop → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + step) s') :
    ∃ s_final,
      stepStmt (.forRangeDyn idx startOp stopOp stepOp body) s_init = some s_final ∧
      P stop s_final := by
  obtain ⟨s_final, h_aux, hP⟩ :=
    forRangeAux_inv_aligned hstep hdvdStop h_step start s_init hle hdvdStart h_init
  refine ⟨s_final, ?_, hP⟩
  rw [stepForRangeAux.forRangeDyn_unfold, hStart, hStop, hStepOp]
  simpa [Option.bind] using h_aux

/-- `prev_multiple = ⌈N/TILE_N⌉·TILE_N - TILE_N` for the surface. -/
def prevMultiple (N TILE_N : Nat) : Nat := (N + TILE_N - 1) / TILE_N * TILE_N - TILE_N

/-- `prevMultiple N TILE_N` is a multiple of `TILE_N`. -/
theorem TILE_N_dvd_prevMultiple (N TILE_N : Nat) : TILE_N ∣ prevMultiple N TILE_N := by
  unfold prevMultiple
  have hrw : (N + TILE_N - 1) / TILE_N * TILE_N - TILE_N
      = ((N + TILE_N - 1) / TILE_N - 1) * TILE_N := by
    rw [Nat.sub_one_mul]
  rw [hrw]
  exact Dvd.intro_left _ rfl

/-- In loop A, a full tile whose start counter is `< prevMultiple` has every lane
column in range. -/
theorem loopA_inRange (N TILE_N : Nat) (hT : 0 < TILE_N) (c : Nat) (ℓ : Fin TILE_N)
    (hc : c * TILE_N < prevMultiple N TILE_N) :
    c * TILE_N + ℓ.val < N := by
  set q := (N + TILE_N - 1) / TILE_N with hq
  have hpm : prevMultiple N TILE_N = (q - 1) * TILE_N := by
    unfold prevMultiple; rw [← hq, Nat.sub_one_mul]
  rw [hpm] at hc
  -- c*TILE_N < (q-1)*TILE_N ⟹ c < q-1 ⟹ c+1 ≤ q-1
  have hclt : c < q - 1 :=
    Nat.lt_of_mul_lt_mul_right hc
  have hcq : c * TILE_N + TILE_N ≤ (q - 1) * TILE_N := by
    have hcc : (c + 1) ≤ q - 1 := hclt
    calc c * TILE_N + TILE_N = (c + 1) * TILE_N := by rw [Nat.succ_mul]
      _ ≤ (q - 1) * TILE_N := Nat.mul_le_mul_right _ hcc
  -- (q-1)*TILE_N < N  since q*TILE_N < N + TILE_N (ceiling), i.e. q*TILE_N - TILE_N < N
  have hceil : q * TILE_N < N + TILE_N := by
    rw [hq]
    have := Nat.div_mul_le_self (N + TILE_N - 1) TILE_N
    omega
  have hlt : (q - 1) * TILE_N < N := by
    have h1 : (q - 1) * TILE_N = q * TILE_N - TILE_N := by rw [Nat.sub_one_mul]
    omega
  calc c * TILE_N + ℓ.val < c * TILE_N + TILE_N := by omega
    _ ≤ (q - 1) * TILE_N := hcq
    _ < N := hlt

/-- In loop A, `bodyMv`/`bodyZv` advance `laneState` by one tile. -/
theorem loopA_bodyMv_eq (N TILE_N : Nat) (hT : 0 < TILE_N) (c : Nat) (s : BlockState)
    (input_ptr : RegionName) (hc : c * TILE_N < prevMultiple N TILE_N)
    (idx : TileIndex [TILE_N]) :
    bodyMv N TILE_N c s input_ptr
        (fun i => (laneState s input_ptr N TILE_N i.1 c).1)
        (fun i => (laneState s input_ptr N TILE_N i.1 c).2) idx
      = (laneState s input_ptr N TILE_N idx.1 (c + 1)).1
    ∧ bodyZv N TILE_N c s input_ptr
        (fun i => (laneState s input_ptr N TILE_N i.1 c).1)
        (fun i => (laneState s input_ptr N TILE_N i.1 c).2) idx
      = (laneState s input_ptr N TILE_N idx.1 (c + 1)).2 := by
  have hin : laneScoreAt s input_ptr N TILE_N idx.1 c
      = some (loopAscore N TILE_N c s input_ptr idx) := by
    unfold laneScoreAt loopAscore
    rw [dif_pos (loopA_inRange N TILE_N hT c idx.1 hc)]
  rw [laneState_succ, hin]
  constructor
  · rfl
  · rfl

/-- `laneScoreAt` only depends on the state through `pid` and `mem` (`readMem`). -/
theorem laneScoreAt_congr (s s0 : BlockState) (input_ptr : RegionName) (N TILE_N : Nat)
    (hpid : s.pid = s0.pid) (hmem : s.mem = s0.mem) (ℓ : Fin TILE_N) (t : Nat) :
    laneScoreAt s input_ptr N TILE_N ℓ t = laneScoreAt s0 input_ptr N TILE_N ℓ t := by
  unfold laneScoreAt
  rw [hpid]
  simp only [BlockState.readMem, hmem]

theorem laneState_congr (s s0 : BlockState) (input_ptr : RegionName) (N TILE_N : Nat)
    (hpid : s.pid = s0.pid) (hmem : s.mem = s0.mem) (ℓ : Fin TILE_N) (c : Nat) :
    laneState s input_ptr N TILE_N ℓ c = laneState s0 input_ptr N TILE_N ℓ c := by
  unfold laneState laneScores
  have : laneScoreAt s input_ptr N TILE_N ℓ = laneScoreAt s0 input_ptr N TILE_N ℓ := by
    funext t; exact laneScoreAt_congr s s0 input_ptr N TILE_N hpid hmem ℓ t
  rw [this]

/-- Loop-A invariant: `pass1Inv` together with `TILE_N ∣ i` (the counter is
always a tile-aligned multiple, as produced by `forRange` from `0` step
`TILE_N`). -/
noncomputable def pass1InvA
    (input_ptr : RegionName) (N TILE_N : Nat) (s0 : BlockState)
    (i : Nat) (s : BlockState) : Prop :=
  pass1Inv input_ptr N TILE_N s0 i s ∧ TILE_N ∣ i

set_option maxHeartbeats 4000000 in
/-- **Loop-A whole-loop step (`forRangeDyn_inv`).** The pass-1 full-tile loop
`for start_n in range(0, prev_multiple, TILE_N)` preserves `pass1InvA`: starting
from the seeded state (`m = ⊥`, `z = 0`, tile count 0), after the loop the `m`/`z`
registers hold the per-lane `laneState` over all full tiles. -/
theorem pass1LoopA_run (N TILE_N : Nat) (hT : 0 < TILE_N) (input_ptr : RegionName)
    (s0 sInit : BlockState)
    (hStop : evalOp (Op.ref .nat [] "prev_multiple") sInit = some (Tile.scalar (prevMultiple N TILE_N)))
    (hInv : pass1InvA input_ptr N TILE_N s0 0 sInit) :
    ∃ sFinal,
      stepStmt (.forRangeDyn "start_n" (Op.constNat 0) (Op.ref .nat [] "prev_multiple")
          (Op.constNat TILE_N) (loopAbody input_ptr N TILE_N)) sInit = some sFinal ∧
      pass1InvA input_ptr N TILE_N s0 (prevMultiple N TILE_N) sFinal := by
  apply forRangeDyn_inv_aligned (start := 0) (stop := prevMultiple N TILE_N) (step := TILE_N)
    (P := pass1InvA input_ptr N TILE_N s0)
    (by simp [evalOp]) hStop (by simp [evalOp]) hT.ne' (Nat.zero_le _)
    (Dvd.intro 0 rfl) (TILE_N_dvd_prevMultiple N TILE_N)
    hInv
  intro i s hi hPi
  obtain ⟨hP, hdvd⟩ := hPi
  obtain ⟨hpids, hmem, hundef, hpid, hm, hz, hprev⟩ := hP
  -- s and s0 agree on pid and mem
  have hpidEq : s.pid = s0.pid := by unfold BlockState.pid; rw [hpids]
  have hcong : ∀ (ℓ : Fin TILE_N) (c : Nat),
      laneState s0 input_ptr N TILE_N ℓ c = laneState s input_ptr N TILE_N ℓ c :=
    fun ℓ c => (laneState_congr s s0 input_ptr N TILE_N hpidEq hmem ℓ c).symm
  -- i is tile-aligned: i = (i / TILE_N) * TILE_N
  have halign : i = (i / TILE_N) * TILE_N := (Nat.div_mul_cancel hdvd).symm
  set c := i / TILE_N with hc
  -- registers at counter i, expressed via laneState at tile count c (now over s)
  have hmc : s.regs .real [TILE_N] "m" = some
      (Tile.mk (fun idx : TileIndex [TILE_N] => (laneState s input_ptr N TILE_N idx.1 c).1)) := by
    rw [hm]; congr 1; apply Tile.ext; intro idx; simp only [Tile.mk]; rw [hcong]
  have hzc : s.regs .real [TILE_N] "z" = some
      (Tile.mk (fun idx : TileIndex [TILE_N] =>
        (((laneState s input_ptr N TILE_N idx.1 c).2 : ℝ) : WithBot ℝ))) := by
    rw [hz]; congr 1; apply Tile.ext; intro idx; simp only [Tile.mk]; rw [hcong]
  have hpidS : s.regs .nat [] "pid_m" = some (Tile.scalar s.pid) := by rw [hpid, hpidEq]
  obtain ⟨s', hstep, hpid', hpids', hmem', hundef', hm', hz', hprev'⟩ :=
    loopA_body_step N TILE_N input_ptr c s
      (fun idx => (laneState s input_ptr N TILE_N idx.1 c).1)
      (fun idx => (laneState s input_ptr N TILE_N idx.1 c).2)
      hpidS hmc hzc
  refine ⟨s', ?_, ?_, ?_⟩
  · rw [← halign] at hstep; exact hstep
  · have hcsucc : (i + TILE_N) / TILE_N = c + 1 := by
      rw [hc, Nat.add_div_right _ hT]
    have hcrange : c * TILE_N < prevMultiple N TILE_N := by rw [← halign]; exact hi
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hpids']; exact hpids
    · rw [hmem']; exact hmem
    · intro rg o; rw [hundef']; exact hundef rg o
    · rw [hpid', hpidEq]
    · rw [hm', hcsucc]
      congr 1; apply Tile.ext; intro idx
      simp only [Tile.mk]
      rw [hcong]
      exact (loopA_bodyMv_eq N TILE_N hT c s input_ptr hcrange idx).1
    · rw [hz', hcsucc]
      congr 1; apply Tile.ext; intro idx
      simp only [Tile.mk]
      rw [hcong, (loopA_bodyMv_eq N TILE_N hT c s input_ptr hcrange idx).2]
    · rw [hprev']; exact hprev
  · exact Dvd.dvd.add hdvd (dvd_refl _)

/-! ### Pass-1 masked-tail loop body (loop B): per-statement exec stepping

`loopBbody` is the alg-erased body of the pass-1 masked-tail `forRangeDyn`
(`for start_n in range(prev_multiple, N, TILE_N)`). It is `loopAbody` plus a
`mask = n_offsets < N` assignment and a masked load with `other = -inf`.
Out-of-range lanes load `⊥`, so `new_m`/`new_z` are no-ops there — matching the
`laneScoreAt = none` case of `laneState_succ`. -/

/-- The alg-erased statement list of the pass-1 masked-tail loop body. -/
def loopBbody (input_ptr : RegionName) (N TILE_N : Nat) : List Stmt :=
  [ Stmt.assign .nat [TILE_N] "n_offsets"
      (Op.add NumericDType.nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.arange TILE_N)),
    Stmt.assign .nat [TILE_N] "offset"
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat N))
        (Op.ref .nat [TILE_N] "n_offsets")),
    Stmt.assign .ptr [TILE_N] "input_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase input_ptr) (Op.ref .nat [TILE_N] "offset")),
    Stmt.assign .bool [TILE_N] "mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [TILE_N] "n_offsets") (Op.constNat N)),
    Stmt.assign .real [TILE_N] "inp"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [TILE_N] "input_ptrs"))
        (MaskOpt.maskOther (Op.ref .bool [TILE_N] "mask") (Op.negInf.broadcast [TILE_N]))),
    Stmt.assign .real [TILE_N] "new_m"
      ((Op.gt ComparableDType.real Broadcast.nil.consSame (Op.ref .real [TILE_N] "m")
          (Op.ref .real [TILE_N] "inp")).where
        (Op.ref .real [TILE_N] "m") (Op.ref .real [TILE_N] "inp")),
    Stmt.assign .real [TILE_N] "new_z"
      (Op.add NumericDType.real Broadcast.nil.consSame
        (Op.mul NumericDType.real Broadcast.nil.consSame
          (Op.sub NumericDType.real Broadcast.nil.consSame (Op.ref .real [TILE_N] "m")
              (Op.ref .real [TILE_N] "new_m")).exp
          (Op.ref .real [TILE_N] "z"))
        (Op.sub NumericDType.real Broadcast.nil.consSame (Op.ref .real [TILE_N] "inp")
            (Op.ref .real [TILE_N] "new_m")).exp),
    Stmt.assign .real [TILE_N] "m" (Op.ref .real [TILE_N] "new_m"),
    Stmt.assign .real [TILE_N] "z" (Op.ref .real [TILE_N] "new_z") ]

/-- `where(gt m inp, m, inp) = m ⊔ inp` for general `WithBot ℝ` inputs (loop B,
where `inp` may be `⊥` on masked-out lanes). -/
theorem real_where_gt_eq_sup_bot (m inp : WithBot ℝ) :
    (if ComparableDType.real.gt m inp = Bool.true then m else inp) = m ⊔ inp := by
  by_cases h : m > inp
  · rw [if_pos (by simpa [ComparableDType.real_gt_eq_true] using h)]
    exact (sup_eq_left.mpr (le_of_lt h)).symm
  · rw [if_neg (by simpa [ComparableDType.real_gt_eq_true] using h)]
    exact (sup_eq_right.mpr (not_lt.mp h)).symm

/-- Per-lane masked load value in loop B at tile `c`: the score when in range,
`⊥` (from `other = -inf`) otherwise. Equals `(laneScoreAt …)` lifted to `WithBot`. -/
noncomputable def loopBload (N TILE_N c : Nat) (s : BlockState) (input_ptr : RegionName)
    (idx : TileIndex [TILE_N]) : WithBot ℝ :=
  if c * TILE_N + idx.1.val < N then
    some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val)))
  else ⊥

/-- Loop-B per-lane running max after the body. -/
noncomputable def bodyMvB (N TILE_N c : Nat) (s : BlockState) (input_ptr : RegionName)
    (mfun : TileIndex [TILE_N] → WithBot ℝ) (idx : TileIndex [TILE_N]) : WithBot ℝ :=
  mfun idx ⊔ loopBload N TILE_N c s input_ptr idx

/-- Loop-B per-lane running denominator after the body. -/
noncomputable def bodyZvB (N TILE_N c : Nat) (s : BlockState) (input_ptr : RegionName)
    (mfun : TileIndex [TILE_N] → WithBot ℝ) (zfun : TileIndex [TILE_N] → ℝ)
    (idx : TileIndex [TILE_N]) : WithBot ℝ :=
  WithBot.realAdd
    (WithBot.realMul
      (WithBot.realExp (WithBot.realSub (mfun idx) (bodyMvB N TILE_N c s input_ptr mfun idx)))
      ((zfun idx : ℝ) : WithBot ℝ))
    (WithBot.realExp (WithBot.realSub (loopBload N TILE_N c s input_ptr idx)
      (bodyMvB N TILE_N c s input_ptr mfun idx)))

set_option maxHeartbeats 4000000 in
/-- **Loop-B body step.** Executing the nine statements of `loopBbody` with the
counter `start_n = c·TILE_N` advances each lane: masked-in lanes by `m ⊔ score`
and the matching denominator update, masked-out lanes are no-ops (load `⊥`). -/
theorem loopB_body_step (N TILE_N : Nat) (input_ptr : RegionName) (c : Nat) (s : BlockState)
    (mfun : TileIndex [TILE_N] → WithBot ℝ) (zfun : TileIndex [TILE_N] → ℝ)
    (hpid : s.regs .nat [] "pid_m" = some (Tile.scalar (s.pid)))
    (hm : s.regs .real [TILE_N] "m" = some (Tile.mk mfun))
    (hz : s.regs .real [TILE_N] "z" = some (Tile.mk (fun idx => ((zfun idx : ℝ) : WithBot ℝ)))) :
    ∃ s', stepStmts (loopBbody input_ptr N TILE_N)
        (s.setReg "start_n" .nat [] (Tile.scalar (c * TILE_N))) = some s'
      ∧ s'.regs .nat [] "pid_m" = some (Tile.scalar (s.pid))
      ∧ s'.pids = s.pids ∧ s'.mem = s.mem ∧ (∀ rg o, s'.undef rg o = s.undef rg o)
      ∧ s'.regs .real [TILE_N] "m" = some (Tile.mk (bodyMvB N TILE_N c s input_ptr mfun))
      ∧ s'.regs .real [TILE_N] "z" = some
          (Tile.mk (bodyZvB N TILE_N c s input_ptr mfun zfun))
      ∧ s'.regs .nat [] "prev_multiple" = s.regs .nat [] "prev_multiple" := by
  simp only [loopBbody]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => c * TILE_N + idx.1.val)) (by
      rw [evalOp_add, evalOp_ref_setReg_same, evalOp_arange]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => s.pid * N + (c * TILE_N + idx.1.val))) (by
      rw [evalOp_add, evalOp_mul, evalOp_constNat]
      simp only [evalOp_ref, BlockState.setReg_ne_name, BlockState.setReg_same, hpid, ne_eq,
        String.reduceEq, not_false_eq_true]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      ((Region.cast input_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) (by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
      congr 1
      apply Tile.ext; intro idx
      show ((Region.cast input_ptr : RegionName), 0 + (s.pid * N + (c * TILE_N + idx.1.val))) = _
      rw [Nat.zero_add]))]
  -- mask = n_offsets < N
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      decide (c * TILE_N + idx.1.val < N))) (by
      rw [evalOp_lt, evalOp_constNat]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
      rfl))]
  -- masked load: inp = if mask then readMem else ⊥
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => loopBload N TILE_N c s input_ptr idx)) (by
      rw [evalOp_load_ptrRef_maskOther "input_ptrs"
        (Op.ref .bool [TILE_N] "mask") (Op.negInf.broadcast [TILE_N]) _
        (Tile.mk (fun idx : TileIndex [TILE_N] =>
          ((Region.cast input_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val))))
        (Tile.mk (fun idx : TileIndex [TILE_N] => decide (c * TILE_N + idx.1.val < N)))
        (Tile.mk (fun _ : TileIndex [TILE_N] => (none : WithBot ℝ)))
        (by simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
              String.reduceEq, not_false_eq_true])
        (by simp only [evalOp_ref, BlockState.setReg_same])
        (by
          simp only [evalOp, Option.bind_eq_bind, Option.bind_some]
          rfl)]
      congr 1
      apply Tile.ext; intro idx
      simp only [Tile.mk, loopBload]
      by_cases hin : c * TILE_N + idx.1.val < N
      · simp only [hin, decide_true, if_true]
        rw [BlockState.readMemValue_real]; rfl
      · simp only [hin, decide_false, if_false]; rfl))]
  -- new_m = where(m > inp, m, inp) = bodyMvB
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (bodyMvB N TILE_N c s input_ptr mfun)) (by
      rw [evalOp_where, evalOp_gt]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind, hm]
      congr 1
      apply Tile.ext; intro idx
      simp only [Tile.select_data, Tile.cop_data, Tile.mk, bodyMvB]
      exact real_where_gt_eq_sup_bot (mfun _) _))]
  -- new_z = bodyZvB
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (bodyZvB N TILE_N c s input_ptr mfun zfun)) (by
      rw [evalOp_add, evalOp_mul, evalOp_exp, evalOp_sub, evalOp_exp, evalOp_sub]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind, hm, hz]
      congr 1))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (bodyMvB N TILE_N c s input_ptr mfun)) (by
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (bodyZvB N TILE_N c s input_ptr mfun zfun)) (by
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hpid
  · rfl
  · rfl
  · intro rg o; rfl
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
  · simp only [BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]

private theorem withBot_noop_zero :
    WithBot.realAdd (WithBot.realMul ((0:ℝ):WithBot ℝ) ((0:ℝ):WithBot ℝ)) ((0:ℝ):WithBot ℝ)
      = ((0:ℝ):WithBot ℝ) := by
  rw [WithBot.realMul_coe_coe, WithBot.realAdd_coe_coe]; norm_num

private theorem withBot_noop_id (z : ℝ) :
    WithBot.realAdd (WithBot.realMul (WithBot.realExp ((0:ℝ):WithBot ℝ)) ((z:ℝ):WithBot ℝ))
        ((0:ℝ):WithBot ℝ)
      = ((z:ℝ):WithBot ℝ) := by
  rw [show WithBot.realExp ((0:ℝ):WithBot ℝ) = ((Real.exp 0:ℝ):WithBot ℝ) from rfl, Real.exp_zero,
    WithBot.realMul_coe_coe, WithBot.realAdd_coe_coe]; norm_num

/-- Loop B advances `laneState` by one tile (per lane: `soStep` if in range, else
no-op), regardless of mask. -/
theorem loopB_bodyMvB_eq (N TILE_N : Nat) (c : Nat) (s : BlockState) (input_ptr : RegionName)
    (idx : TileIndex [TILE_N]) :
    bodyMvB N TILE_N c s input_ptr
        (fun i => (laneState s input_ptr N TILE_N i.1 c).1) idx
      = (laneState s input_ptr N TILE_N idx.1 (c + 1)).1
    ∧ bodyZvB N TILE_N c s input_ptr
        (fun i => (laneState s input_ptr N TILE_N i.1 c).1)
        (fun i => (laneState s input_ptr N TILE_N i.1 c).2) idx
      = (((laneState s input_ptr N TILE_N idx.1 (c + 1)).2 : ℝ) : WithBot ℝ) := by
  rw [laneState_succ]
  unfold bodyMvB bodyZvB loopBload laneScoreAt
  by_cases hin : c * TILE_N + idx.1.val < N
  · -- in range: loopBload = some score, matches soStep
    rw [dif_pos hin]
    simp only [hin, if_true]
    refine ⟨rfl, ?_⟩
    -- bodyZvB shape = soStep_snd_match (with score = readMem)
    have h := soStep_snd_match (laneState s input_ptr N TILE_N idx.1 c).1
      (laneState s input_ptr N TILE_N idx.1 c).2
      (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val)))
    rw [show bodyMvB N TILE_N c s input_ptr
            (fun i => (laneState s input_ptr N TILE_N i.1 c).1) idx
          = (laneState s input_ptr N TILE_N idx.1 c).1 ⊔
            (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) : WithBot ℝ) from by
        unfold bodyMvB loopBload; rw [if_pos hin]]
    exact h
  · -- out of range: loopBload = ⊥, no-op
    rw [dif_neg hin]
    simp only [hin, if_false]
    have hbot : loopBload N TILE_N c s input_ptr idx = ⊥ := by
      unfold loopBload; rw [if_neg hin]
    have hMvB : bodyMvB N TILE_N c s input_ptr
        (fun i => (laneState s input_ptr N TILE_N i.1 c).1) idx
        = (laneState s input_ptr N TILE_N idx.1 c).1 := by
      unfold bodyMvB; rw [hbot, sup_bot_eq]
    rw [sup_bot_eq]
    refine ⟨rfl, ?_⟩
    rw [hMvB]
    -- bodyZvB with ⊥ load: exp(m-m)·z + exp(⊥-m) = z
    cases hm0 : (laneState s input_ptr N TILE_N idx.1 c).1 with
    | bot =>
      have hz0 : (laneState s input_ptr N TILE_N idx.1 c).2 = 0 := by
        have := laneState_denom_closed s input_ptr N TILE_N idx.1 c
        rw [this, hm0]; simp [Option.elim]
      rw [hz0]
      simp only [WithBot.realSub_bot_left, WithBot.realSub_bot_right, WithBot.realExp_bot]
      exact withBot_noop_zero
    | coe r =>
      rw [WithBot.realSub_coe_coe, sub_self]
      simp only [WithBot.realSub_bot_left, WithBot.realExp_bot]
      exact withBot_noop_id _

set_option maxHeartbeats 4000000 in
/-- **Loop-B whole-loop step (`forRangeDyn_inv`).** The pass-1 masked-tail loop
`for start_n in range(prev_multiple, N, TILE_N)` preserves `pass1InvA`. -/
theorem pass1LoopB_run (N TILE_N : Nat) (hT : 0 < TILE_N) (input_ptr : RegionName)
    (s0 sInit : BlockState)
    (hStart : evalOp (Op.ref .nat [] "prev_multiple") sInit
        = some (Tile.scalar (prevMultiple N TILE_N)))
    (hInv : pass1InvA input_ptr N TILE_N s0 (prevMultiple N TILE_N) sInit) :
    ∃ final sFinal,
      stepStmt (.forRangeDyn "start_n" (Op.ref .nat [] "prev_multiple") (Op.constNat N)
          (Op.constNat TILE_N) (loopBbody input_ptr N TILE_N)) sInit = some sFinal ∧
      N ≤ final ∧ pass1InvA input_ptr N TILE_N s0 final sFinal := by
  apply forRangeDyn_inv (start := prevMultiple N TILE_N) (stop := N) (step := TILE_N)
    (P := pass1InvA input_ptr N TILE_N s0)
    hStart (by simp [evalOp]) (by simp [evalOp]) hT.ne' hInv
  intro i s hi hPi
  obtain ⟨hP, hdvd⟩ := hPi
  obtain ⟨hpids, hmem, hundef, hpid, hm, hz, hprev⟩ := hP
  have hpidEq : s.pid = s0.pid := by unfold BlockState.pid; rw [hpids]
  have hcong : ∀ (ℓ : Fin TILE_N) (c : Nat),
      laneState s0 input_ptr N TILE_N ℓ c = laneState s input_ptr N TILE_N ℓ c :=
    fun ℓ c => (laneState_congr s s0 input_ptr N TILE_N hpidEq hmem ℓ c).symm
  have halign : i = (i / TILE_N) * TILE_N := (Nat.div_mul_cancel hdvd).symm
  set c := i / TILE_N with hc
  have hpidS : s.regs .nat [] "pid_m" = some (Tile.scalar s.pid) := by rw [hpid, hpidEq]
  have hmc : s.regs .real [TILE_N] "m" = some
      (Tile.mk (fun idx : TileIndex [TILE_N] => (laneState s input_ptr N TILE_N idx.1 c).1)) := by
    rw [hm]; congr 1; apply Tile.ext; intro idx; simp only [Tile.mk]; rw [hcong]
  have hzc : s.regs .real [TILE_N] "z" = some
      (Tile.mk (fun idx : TileIndex [TILE_N] =>
        (((laneState s input_ptr N TILE_N idx.1 c).2 : ℝ) : WithBot ℝ))) := by
    rw [hz]; congr 1; apply Tile.ext; intro idx; simp only [Tile.mk]; rw [hcong]
  obtain ⟨s', hstep, hpid', hpids', hmem', hundef', hm', hz', hprev'⟩ :=
    loopB_body_step N TILE_N input_ptr c s
      (fun idx => (laneState s input_ptr N TILE_N idx.1 c).1)
      (fun idx => (laneState s input_ptr N TILE_N idx.1 c).2)
      hpidS hmc hzc
  refine ⟨s', ?_, ?_, ?_⟩
  · rw [← halign] at hstep; exact hstep
  · have hcsucc : (i + TILE_N) / TILE_N = c + 1 := by rw [hc, Nat.add_div_right _ hT]
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hpids']; exact hpids
    · rw [hmem']; exact hmem
    · intro rg o; rw [hundef']; exact hundef rg o
    · rw [hpid', hpidEq]
    · rw [hm', hcsucc]
      congr 1; apply Tile.ext; intro idx
      simp only [Tile.mk]
      rw [hcong]
      exact (loopB_bodyMvB_eq N TILE_N c s input_ptr idx).1
    · rw [hz', hcsucc]
      congr 1; apply Tile.ext; intro idx
      simp only [Tile.mk]
      rw [hcong, (loopB_bodyMvB_eq N TILE_N c s input_ptr idx).2]
    · rw [hprev']; exact hprev
  · exact Dvd.dvd.add hdvd (dvd_refl _)

/-! ### Cross-lane reduction collapse: `final_m`/`z` to the full-row closed form

After both pass-1 loops the `m`/`z` registers hold the per-lane `laneState` over
all tiles (`pass1LoopA_run` then `pass1LoopB_run`). The reduction
`final_m = tl.max(m, 0)`, `z = tl.sum(exp(m - final_m)·z)` collapses these to the
global `(max, denom)`. The key observation: because `softmaxOptimizeFullSpec` is
shift-invariant, we never need `final_m = rowMax` exactly — only that
`final_m = some c` for *some* real `c`, and that the same `c` is used to compute
`z`. The denominator then collapses to `exp(-c)·Σ_{j<N} exp(input[j])` via the
per-lane closed form and the lane×tile→`Fin N` partition. -/

/-- Sum of `exp` over `laneScores` as a tile-indexed `Finset.range` sum. -/
theorem laneScores_exp_sum
    (s : BlockState) (input_ptr : RegionName) (N TILE_N : Nat)
    (ℓ : Fin TILE_N) (c : Nat) :
    ((laneScores s input_ptr N TILE_N ℓ c).map Real.exp).sum
      = ∑ t ∈ Finset.range c,
          (if t * TILE_N + ℓ.val < N then
            Real.exp (s.readMem input_ptr (s.pid * N + (t * TILE_N + ℓ.val))) else 0) := by
  unfold laneScores
  induction c with
  | zero => simp
  | succ k ih =>
    rw [List.range_succ, List.filterMap_append, List.map_append, List.sum_append,
        Finset.sum_range_succ, ih]
    unfold laneScoreAt
    by_cases h : k * TILE_N + ℓ.val < N
    · simp [h]
    · simp [h]

/-- **Lane-major partition.** For full tile coverage (`N ≤ c·TILE_N`), summing a
column-indexed value `V` over `(lane, tile)` pairs in range recovers the sum over
all columns `j < N`. -/
theorem lane_tile_partition (N TILE_N : Nat) (hT : 0 < TILE_N) (c : Nat)
    (hc : N ≤ c * TILE_N) (V : Nat → ℝ) :
    ∑ ℓ : Fin TILE_N, ∑ t ∈ Finset.range c,
        (if t * TILE_N + ℓ.val < N then V (t * TILE_N + ℓ.val) else 0)
      = ∑ j ∈ Finset.range N, V j := by
  have step1 : ∀ ℓ : Fin TILE_N, ∑ t ∈ Finset.range c,
        (if t * TILE_N + ℓ.val < N then V (t * TILE_N + ℓ.val) else 0)
      = ∑ t ∈ (Finset.range c).filter (fun t => t * TILE_N + ℓ.val < N),
          V (t * TILE_N + ℓ.val) := by
    intro ℓ; rw [Finset.sum_filter]
  simp_rw [step1]
  rw [Finset.sum_sigma']
  apply Finset.sum_nbij'
    (i := fun p : (_ : Fin TILE_N) × Nat => p.2 * TILE_N + (p.1 : Nat))
    (j := fun j => (⟨⟨j % TILE_N, Nat.mod_lt _ hT⟩, j / TILE_N⟩ : (_ : Fin TILE_N) × Nat))
  · rintro ⟨ℓ, t⟩ hmem
    simp only [Finset.mem_sigma, Finset.mem_filter, Finset.mem_range] at hmem
    simp only [Finset.mem_range]; exact hmem.2.2
  · intro j hj
    simp only [Finset.mem_range] at hj
    simp only [Finset.mem_sigma, Finset.mem_filter, Finset.mem_range, Finset.mem_univ, true_and]
    constructor
    · rw [Nat.div_lt_iff_lt_mul hT]; omega
    · rw [Nat.div_add_mod' j TILE_N]; exact hj
  · rintro ⟨ℓ, t⟩ hmem
    have hℓ : ℓ.val < TILE_N := ℓ.isLt
    have hmod : (t * TILE_N + ℓ.val) % TILE_N = ℓ.val := Nat.mul_add_mod_of_lt hℓ
    have hdiv : (t * TILE_N + ℓ.val) / TILE_N = t := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hT, Nat.div_eq_of_lt hℓ, Nat.zero_add]
    apply Sigma.ext
    · apply Fin.ext; simpa using hmod
    · simpa using hdiv
  · intro j hj; rw [Nat.div_add_mod' j TILE_N]
  · rintro ⟨ℓ, t⟩ hmem; rfl

/-- Per-lane reduction term: `realMul (realExp (realSub m[ℓ] (some c))) z[ℓ]`,
under the closed form `z[ℓ] = κ(m[ℓ])·S` with the `⊥ → S = 0` guard, equals
`some (exp(-c)·S)` — independent of whether `m[ℓ] ≤ c`. -/
theorem redTerm_eq (m : WithBot ℝ) (S c : ℝ) (hmS : m = ⊥ → S = 0) :
    WithBot.realMul (WithBot.realExp (WithBot.realSub m ((c : ℝ) : WithBot ℝ)))
        (((m.elim 0 (fun r => Real.exp (-r))) * S : ℝ) : WithBot ℝ)
      = (((Real.exp (-c)) * S : ℝ) : WithBot ℝ) := by
  cases m with
  | bot =>
    have hS : S = 0 := hmS rfl
    subst hS
    rw [mul_zero, mul_zero, WithBot.realSub_bot_left, WithBot.realExp_bot,
        WithBot.realMul_coe_coe, mul_zero]
  | coe r =>
    simp only [WithBot.realSub_coe_coe, WithBot.realExp_coe, Option.elim,
      WithBot.realMul_coe_coe, WithBot.coe_eq_coe]
    rw [← mul_assoc, ← Real.exp_add]; ring_nf

/-- A `⊥` running max means the lane absorbed no scores, so its exp-sum is `0`. -/
theorem laneState_bot_imp_S_zero
    (s : BlockState) (input_ptr : RegionName) (N TILE_N : Nat)
    (ℓ : Fin TILE_N) (c : Nat)
    (hbot : (laneState s input_ptr N TILE_N ℓ c).1 = ⊥) :
    ((laneScores s input_ptr N TILE_N ℓ c).map Real.exp).sum = 0 := by
  have h := soStep_foldl_consistent (laneScores s input_ptr N TILE_N ℓ c) ⊥ 0 0
    (by simp) (fun _ => rfl)
  have h2 := h.2
  simp only [laneState, laneScores] at hbot
  simp only at h2
  have := h2 (by simpa [laneState, laneScores] using hbot)
  simpa using this

/-- The running-max fold of `⊔` from `⊥` over a list dominates `some x` for any
member `x`, so a nonempty list yields a non-`⊥` running max. -/
theorem foldl_sup_mem_ne_bot (xs : List ℝ) (x : ℝ) (hx : x ∈ xs) :
    xs.foldl (fun acc y => acc ⊔ ((y : ℝ) : WithBot ℝ)) ⊥ ≠ ⊥ := by
  have hmono : ∀ (ys : List ℝ) (a b : WithBot ℝ), a ≤ b →
      ys.foldl (fun acc y => acc ⊔ ((y : ℝ) : WithBot ℝ)) a
        ≤ ys.foldl (fun acc y => acc ⊔ ((y : ℝ) : WithBot ℝ)) b := by
    intro ys
    induction ys with
    | nil => intro a b h; simpa using h
    | cons z zs ih => intro a b h; exact ih _ _ (sup_le_sup_right h _)
  have hge : ∀ (ys : List ℝ) (a : WithBot ℝ),
      a ≤ ys.foldl (fun acc y => acc ⊔ ((y : ℝ) : WithBot ℝ)) a := by
    intro ys
    induction ys with
    | nil => intro a; simp
    | cons z zs ih => intro a; exact le_trans le_sup_left (ih _)
  -- isolate x
  obtain ⟨l, r, rfl⟩ := List.append_of_mem hx
  rw [List.foldl_append, List.foldl_cons]
  set base := (l.foldl (fun acc y => acc ⊔ ((y : ℝ) : WithBot ℝ)) ⊥) ⊔ ((x : ℝ) : WithBot ℝ)
    with hbasedef
  have h1 : ((x : ℝ) : WithBot ℝ) ≤ base := by rw [hbasedef]; exact le_sup_right
  have h2 := hge r base
  have h3 : ((x : ℝ) : WithBot ℝ)
      ≤ r.foldl (fun acc y => acc ⊔ ((y : ℝ) : WithBot ℝ)) base := le_trans h1 h2
  intro hbot
  rw [hbot] at h3
  exact absurd (le_antisymm h3 bot_le) (by simp)

/-- Lane `0` over the first tile absorbs column `0`, so when `0 < N` its running
max is not `⊥`. -/
theorem laneState_zero_ne_bot
    (s : BlockState) (input_ptr : RegionName) (N TILE_N : Nat)
    (hN : 0 < N) (hT : 0 < TILE_N) (c : Nat) (hc : 0 < c) :
    (laneState s input_ptr N TILE_N ⟨0, hT⟩ c).1 ≠ ⊥ := by
  unfold laneState
  rw [soStep_foldl_fst]
  have hsc : laneScoreAt s input_ptr N TILE_N ⟨0, hT⟩ 0
      = some (s.readMem input_ptr (s.pid * N + 0)) := by
    unfold laneScoreAt; rw [dif_pos (by simpa using hN)]; simp
  have hin : s.readMem input_ptr (s.pid * N + 0)
      ∈ laneScores s input_ptr N TILE_N ⟨0, hT⟩ c := by
    unfold laneScores
    rw [List.mem_filterMap]
    exact ⟨0, by simp [hc], hsc⟩
  exact foldl_sup_mem_ne_bot _ _ hin

/-- Canonical reduction axis `0 : Fin [TILE_N].length`. Factored into a single
named term so every `reduceMax`/`reduceSum`/`insertAxisIndex` occurrence shares
*syntactically identical* proof data (otherwise independent `⟨0, by simp⟩`
elaborations can break `rw` pattern matching nondeterministically). Marked
`@[reducible]` so `Tile.reduceSum`/`reduceMaxDrop`'s `dif_pos` axis-bound check
still reduces definitionally. -/
@[reducible] def redAxis (TILE_N : Nat) : Fin [TILE_N].length := ⟨0, by simp⟩

/-- The three reduction statements `final_m = tl.max(m,0)`,
`z = tl.sum(exp(m - final_m)·z)`, `m = final_m`, alg-erased. -/
def reductionStmts (TILE_N : Nat) : List Stmt :=
  [ Stmt.assign .real [] "final_m"
      (Op.reduceMax (shape := [TILE_N]) (redAxis TILE_N) Bool.false (Op.ref .real [TILE_N] "m")),
    Stmt.assign .real [] "z"
      (Op.reduceSum (shape := [TILE_N]) (redAxis TILE_N) Bool.false
        (Op.mul NumericDType.real Broadcast.nil.consSame
          (Op.sub NumericDType.real Broadcast.scalarR (Op.ref .real [TILE_N] "m")
              (Op.ref .real [] "final_m")).exp
          (Op.ref .real [TILE_N] "z"))),
    Stmt.assign .real [] "m" (Op.ref .real [] "final_m") ]

set_option maxHeartbeats 2000000 in
/-- **Cross-lane reduction collapse.** From a state whose `m`/`z` registers hold
the per-lane `laneState` over full coverage (`N ≤ cfin·TILE_N`, `0 < N`), running
the three reduction statements gives `final_m = m = some c` for some real `c`,
and `z = some (exp(-c)·Σ_{j<N} exp(input[j]))`. -/
theorem reduction_step (N TILE_N : Nat) (hN : 0 < N) (hT : 0 < TILE_N)
    (input_ptr : RegionName) (cfin : Nat) (hcfin : 0 < cfin)
    (hcov : N ≤ cfin * TILE_N) (sb s : BlockState)
    (hpidEq : s.pid = sb.pid)
    (hcong : ∀ (ℓ : Fin TILE_N) (cc : Nat),
        laneState sb input_ptr N TILE_N ℓ cc = laneState s input_ptr N TILE_N ℓ cc)
    (hm : s.regs .real [TILE_N] "m" = some
        ⟨fun idx : TileIndex [TILE_N] => (laneState sb input_ptr N TILE_N idx.1 cfin).1⟩)
    (hz : s.regs .real [TILE_N] "z" = some
        ⟨fun idx : TileIndex [TILE_N] =>
          (((laneState sb input_ptr N TILE_N idx.1 cfin).2 : ℝ) : WithBot ℝ)⟩) :
    ∃ (c : ℝ) (s' : BlockState),
      stepStmts (reductionStmts TILE_N) s = some s' ∧
      s'.pids = s.pids ∧ s'.mem = s.mem ∧
      (∀ rg o, s'.undef rg o = s.undef rg o) ∧
      s'.regs .nat [] "pid_m" = s.regs .nat [] "pid_m" ∧
      s'.regs .real [] "m" = some (Tile.scalar ((c : ℝ) : WithBot ℝ)) ∧
      s'.regs .real [] "z" = some (Tile.scalar
        (((Real.exp (-c) * ∑ j ∈ Finset.range N,
            Real.exp (s.readMem input_ptr (s.pid * N + j))) : ℝ) : WithBot ℝ)) := by
  -- the per-lane running max (over s, via congruence)
  set mvals : Fin TILE_N → WithBot ℝ :=
    fun ℓ => (laneState s input_ptr N TILE_N ℓ cfin).1 with hmvals
  set zvals : Fin TILE_N → ℝ :=
    fun ℓ => (laneState s input_ptr N TILE_N ℓ cfin).2 with hzvals
  set Sv : Fin TILE_N → ℝ :=
    fun ℓ => ((laneScores s input_ptr N TILE_N ℓ cfin).map Real.exp).sum with hSv
  -- the m register tile, rewritten over s
  have hmS : s.regs .real [TILE_N] "m" = some ⟨fun idx => mvals idx.1⟩ := by
    rw [hm]; congr 1; apply Tile.ext; intro idx; simp only [hmvals]; rw [hcong]
  have hzS : s.regs .real [TILE_N] "z" = some
      ⟨fun idx => ((zvals idx.1 : ℝ) : WithBot ℝ)⟩ := by
    rw [hz]; congr 1; apply Tile.ext; intro idx; simp only [hzvals]; rw [hcong]
  -- final_m = sup' over lanes of mvals, which is `some c`
  have hne0 : mvals ⟨0, hT⟩ ≠ ⊥ := by
    simp only [hmvals]
    rw [← hcong]
    exact laneState_zero_ne_bot sb input_ptr N TILE_N hN hT cfin hcfin
  -- the sup'
  set sup0 : WithBot ℝ := (Finset.univ : Finset (Fin TILE_N)).sup'
      (by exact ⟨⟨0, hT⟩, Finset.mem_univ _⟩) (fun k => mvals k) with hsup0
  have hsupne : sup0 ≠ ⊥ := by
    rw [hsup0]
    intro hbot
    have : mvals ⟨0, hT⟩ ≤ (⊥ : WithBot ℝ) := by
      rw [← hbot]; exact Finset.le_sup' _ (Finset.mem_univ _)
    exact hne0 (le_antisymm this bot_le)
  obtain ⟨c, hc⟩ : ∃ c : ℝ, sup0 = some c := by
    cases hcase : sup0 with
    | bot => exact absurd hcase hsupne
    | coe c => exact ⟨c, rfl⟩
  refine ⟨c, ?_⟩
  -- evalOp facts for the two reductions (with concrete Ops)
  have hevalMax : evalOp (Op.reduceMax (shape := [TILE_N]) (redAxis TILE_N) Bool.false
        (Op.ref .real [TILE_N] "m")) s = some (Tile.scalar ((c : ℝ) : WithBot ℝ)) := by
    simp only [evalOp_reduceMax]
    rw [evalOp_ref, hmS]
    simp only [Option.bind_some, Option.bind_eq_bind]
    show Tile.reduceMax (shape := [TILE_N]) (redAxis TILE_N) Bool.false ⟨fun idx => mvals idx.1⟩
        = some (Tile.scalar ((c : ℝ) : WithBot ℝ))
    unfold Tile.reduceMax Tile.reduceMaxDrop
    rw [dif_pos (by simpa using hT)]
    apply congrArg some
    apply Tile.ext; intro outIdx
    simp only [Tile.scalar]
    show (Finset.univ.sup' _ fun x =>
        mvals (TileShape.insertAxisIndex [TILE_N] (redAxis TILE_N) outIdx x).1) = ((c:ℝ):WithBot ℝ)
    -- the lambda is defeq to `fun k => mvals k`, so the sup' is defeq to `sup0`
    exact hc
  -- the post-reduceMax state, where m/z are unchanged and final_m = some c
  set s6 := s.setReg "final_m" .real [] (Tile.scalar ((c : ℝ) : WithBot ℝ)) with hs6
  have hmS6 : s6.regs .real [TILE_N] "m" = some ⟨fun idx => mvals idx.1⟩ := by
    rw [hs6]
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    exact hmS
  have hzS6 : s6.regs .real [TILE_N] "z" = some ⟨fun idx => ((zvals idx.1 : ℝ) : WithBot ℝ)⟩ := by
    rw [hs6]
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    exact hzS
  have hfmS6 : s6.regs .real [] "final_m" = some (Tile.scalar ((c : ℝ) : WithBot ℝ)) := by
    rw [hs6]; simp only [BlockState.setReg_same]
  -- evalOp fact for the reduceSum over s6
  have hevalSum : evalOp (Op.reduceSum (shape := [TILE_N]) (redAxis TILE_N) Bool.false
        (Op.mul NumericDType.real Broadcast.nil.consSame
          (Op.sub NumericDType.real Broadcast.scalarR (Op.ref .real [TILE_N] "m")
              (Op.ref .real [] "final_m")).exp
          (Op.ref .real [TILE_N] "z"))) s6
      = some (Tile.scalar (((Real.exp (-c) * ∑ j ∈ Finset.range N,
          Real.exp (s.readMem input_ptr (s.pid * N + j))) : ℝ) : WithBot ℝ)) := by
    simp only [evalOp_reduceSum, evalOp_mul, evalOp_exp, evalOp_sub,
      evalOp_ref, Option.bind_some, Option.bind_eq_bind, hmS6, hzS6, hfmS6]
    apply congrArg some
    apply Tile.ext; intro outIdx
    simp only [Tile.reduceSum_false, Tile.reduceSumDrop_data, Tile.scalar]
    have hterm : ∀ k : Fin TILE_N,
        (Tile.bop (NumericDType.mul .real) Broadcast.nil.consSame
          (Tile.uop WithBot.realExp
            (Tile.bop (NumericDType.sub .real) Broadcast.scalarR
              ⟨fun idx => mvals idx.1⟩ (Tile.scalar ((c:ℝ):WithBot ℝ))))
          ⟨fun idx => ((zvals idx.1 : ℝ) : WithBot ℝ)⟩).data
            (TileShape.insertAxisIndex [TILE_N] (redAxis TILE_N) outIdx k)
          = (((Real.exp (-c) * Sv k : ℝ)) : WithBot ℝ) := by
      intro k
      show WithBot.realMul (WithBot.realExp (WithBot.realSub (mvals k)
          ((c:ℝ):WithBot ℝ))) ((zvals k : ℝ) : WithBot ℝ) = _
      rw [show ((zvals k : ℝ) : WithBot ℝ)
            = (((mvals k).elim 0 (fun r => Real.exp (-r)) * Sv k : ℝ) : WithBot ℝ) from by
          congr 1
          simp only [hzvals, hmvals, hSv]
          exact laneState_denom_closed s input_ptr N TILE_N k cfin]
      apply redTerm_eq
      intro hbot
      simp only [hSv]
      exact laneState_bot_imp_S_zero s input_ptr N TILE_N k cfin (by simpa [hmvals] using hbot)
    refine Eq.trans (Finset.sum_congr rfl (fun k _ => hterm k)) ?_
    rw [WithBot.sum_some_eq_some]
    apply congrArg
    rw [← Finset.mul_sum]
    congr 1
    simp only [hSv]
    refine Eq.trans (Finset.sum_congr rfl
      (fun k _ => laneScores_exp_sum s input_ptr N TILE_N k cfin)) ?_
    exact lane_tile_partition N TILE_N hT cfin hcov
      (fun j => Real.exp (s.readMem input_ptr (s.pid * N + j)))
  -- evalOp fact for `m = final_m` over the post-reduceSum state
  set s7 := s6.setReg "z" .real [] (Tile.scalar (((Real.exp (-c) * ∑ j ∈ Finset.range N,
        Real.exp (s.readMem input_ptr (s.pid * N + j))) : ℝ) : WithBot ℝ)) with hs7
  have hevalM : evalOp (Op.ref .real [] "final_m") s7
      = some (Tile.scalar ((c : ℝ) : WithBot ℝ)) := by
    rw [evalOp_ref, hs7]
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    exact hfmS6
  -- step all three statements; the chain composes via `Eq.trans` (defeq-tolerant
  -- on the statement shapes, which differ only as `reduceShape … ≟ []`).
  refine ⟨s7.setReg "m" .real [] (Tile.scalar ((c : ℝ) : WithBot ℝ)), ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · show stepStmts (reductionStmts TILE_N) s = _
    refine (stepStmts.cons_some (stepStmt_assign_eq_some hevalMax)).trans ?_
    refine (stepStmts.cons_some (stepStmt_assign_eq_some hevalSum)).trans ?_
    refine (stepStmts.cons_some (stepStmt_assign_eq_some hevalM)).trans ?_
    exact stepStmts.nil
  · rfl
  · rfl
  · intro rg o; rfl
  · rw [hs7, hs6]
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
  · simp only [BlockState.setReg_same]
  · rw [hs7]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]

/-! ### Pass-2 store loops: writeback `exp(input[j] - rowMax) / z`

After the reduction the registers hold `m = some c` and `z = some (exp(-c)·S)`
with `S = Σ_{j<N} exp(input[j])`. The two pass-2 loops recompute, per column `j`,
`out = exp(input[j] - c) / (exp(-c)·S) = exp(input[j]) / S =
softmaxOptimizeFullSpec`, and store it to `output_ptr[pid·N + j]`. -/

/-- **Pass-2 store-value cancellation.** The demoted stored value at an in-range
lane: `(exp(x - c) / (exp(-c)·S)).unbotD 0 = exp x / S`. The shift `c` cancels;
holds even for `S = 0` (both sides `0`). -/
theorem pass2_store_value (x c S : ℝ) :
    (WithBot.realDiv (WithBot.realExp (WithBot.realSub ((x : ℝ) : WithBot ℝ) ((c : ℝ) : WithBot ℝ)))
        (((Real.exp (-c) * S : ℝ)) : WithBot ℝ)).unbotD 0
      = Real.exp x / S := by
  rw [WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.realDiv_coe_coe, WithBot.unbotD_coe]
  rw [Real.exp_sub, Real.exp_neg]
  by_cases hS : S = 0
  · simp [hS]
  · have hc : Real.exp c ≠ 0 := Real.exp_ne_zero c
    field_simp

/-- The genuine full-row spec, rewritten over the running denominator. With
`S = Σ_{j<N} exp(input[j])`, `softmaxOptimizeFullSpec s input N j = exp(input[j]) / S`. -/
theorem softmaxOptimizeFullSpec_eq_div
    (s : BlockState) (input_ptr : RegionName) (N : Nat) (j : Fin N) :
    softmaxOptimizeFullSpec s input_ptr N j
      = Real.exp (s.readMem input_ptr (s.pid * N + j.val))
        / ∑ j' ∈ Finset.range N, Real.exp (s.readMem input_ptr (s.pid * N + j')) := by
  unfold softmaxOptimizeFullSpec softmaxOptimizeRow linearOffset
  congr 1
  rw [← Fin.sum_univ_eq_sum_range (fun j' => Real.exp (s.readMem input_ptr (s.pid * N + j')))]

/-- The alg-erased statement list of the pass-2 full-tile store loop body. -/
def loopApass2body (output_ptr input_ptr : RegionName) (N TILE_N : Nat) : List Stmt :=
  [ Stmt.assign .nat [TILE_N] "n_offsets"
      (Op.add NumericDType.nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.arange TILE_N)),
    Stmt.assign .nat [TILE_N] "offset"
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat N))
        (Op.ref .nat [TILE_N] "n_offsets")),
    Stmt.assign .ptr [TILE_N] "input_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase input_ptr) (Op.ref .nat [TILE_N] "offset")),
    Stmt.assign .real [TILE_N] "inp"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [TILE_N] "input_ptrs")) MaskOpt.none),
    Stmt.assign .real [TILE_N] "e"
      (Op.sub NumericDType.real Broadcast.scalarR (Op.ref .real [TILE_N] "inp")
          (Op.ref .real [] "m")).exp,
    Stmt.assign .real [TILE_N] "out"
      (Op.div NumericDType.real Broadcast.scalarR (Op.ref .real [TILE_N] "e")
        (Op.ref .real [] "z")),
    Stmt.assign .ptr [TILE_N] "output_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase output_ptr) (Op.ref .nat [TILE_N] "offset")),
    Stmt.store .real [TILE_N] (MemAccess.ptr (Op.ref .ptr [TILE_N] "output_ptrs"))
      (Op.ref .real [TILE_N] "out") MaskOpt.none ]

/-- The alg-erased statement list of the pass-2 masked-tail store loop body. -/
def loopBpass2body (output_ptr input_ptr : RegionName) (N TILE_N : Nat) : List Stmt :=
  [ Stmt.assign .nat [TILE_N] "n_offsets"
      (Op.add NumericDType.nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.arange TILE_N)),
    Stmt.assign .nat [TILE_N] "offset"
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat N))
        (Op.ref .nat [TILE_N] "n_offsets")),
    Stmt.assign .ptr [TILE_N] "input_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase input_ptr) (Op.ref .nat [TILE_N] "offset")),
    Stmt.assign .bool [TILE_N] "mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [TILE_N] "n_offsets") (Op.constNat N)),
    Stmt.assign .real [TILE_N] "inp"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [TILE_N] "input_ptrs"))
        (MaskOpt.maskOther (Op.ref .bool [TILE_N] "mask") (Op.negInf.broadcast [TILE_N]))),
    Stmt.assign .real [TILE_N] "e"
      (Op.sub NumericDType.real Broadcast.scalarR (Op.ref .real [TILE_N] "inp")
          (Op.ref .real [] "m")).exp,
    Stmt.assign .real [TILE_N] "out"
      (Op.div NumericDType.real Broadcast.scalarR (Op.ref .real [TILE_N] "e")
        (Op.ref .real [] "z")),
    Stmt.assign .ptr [TILE_N] "output_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase output_ptr) (Op.ref .nat [TILE_N] "offset")),
    Stmt.store .real [TILE_N] (MemAccess.ptr (Op.ref .ptr [TILE_N] "output_ptrs"))
      (Op.ref .real [TILE_N] "out") (MaskOpt.mask (Op.ref .bool [TILE_N] "mask")) ]

/-- Pass-2 output offset for lane `idx` at tile `c`: `pid·N + (c·TILE_N + ℓ)`. -/
noncomputable def pass2Off (N TILE_N c : Nat) (s : BlockState) (idx : TileIndex [TILE_N]) : Nat :=
  s.pid * N + (c * TILE_N + idx.1.val)

/-- The pass-2 stored value (already demoted to `ℝ`) at lane `idx`, tile `c`, with
running scalars `m = cc`, `z = exp(-cc)·S`: `exp(input[pid·N+col] - cc)/(exp(-cc)·S)`,
which `pass2_store_value` collapses to `exp(input[...])/S`. -/
noncomputable def pass2Val (N TILE_N c : Nat) (cc S : ℝ) (s : BlockState)
    (input_ptr : RegionName) (idx : TileIndex [TILE_N]) : ℝ :=
  Real.exp (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) / S

set_option maxHeartbeats 4000000 in
/-- **Pass-2 loop-A body advance.** Executing `loopApass2body` with counter
`start_n = c·TILE_N` from scalar registers `m = cc`, `z = exp(-cc)·S` produces a
state whose `output_ptr` readback is the masked scatter of `pass2Val` over the
tile, with all other regions/registers preserved. -/
theorem loopApass2_body_step (N TILE_N : Nat) (output_ptr input_ptr : RegionName)
    (hne : output_ptr ≠ input_ptr) (c : Nat) (cc S : ℝ) (s : BlockState)
    (hpid : s.regs .nat [] "pid_m" = some (Tile.scalar (s.pid)))
    (hm : s.regs .real [] "m" = some (Tile.scalar ((cc : ℝ) : WithBot ℝ)))
    (hz : s.regs .real [] "z" = some (Tile.scalar (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))) :
    ∃ s', stepStmts (loopApass2body output_ptr input_ptr N TILE_N)
        (s.setReg "start_n" .nat [] (Tile.scalar (c * TILE_N))) = some s'
      ∧ s'.regs .nat [] "pid_m" = some (Tile.scalar (s.pid))
      ∧ s'.regs .real [] "m" = some (Tile.scalar ((cc : ℝ) : WithBot ℝ))
      ∧ s'.regs .real [] "z" = some (Tile.scalar (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))
      ∧ s'.pids = s.pids
      ∧ (∀ rg o, rg ≠ output_ptr → s'.readMem rg o = s.readMem rg o)
      ∧ (∀ i : TileIndex [TILE_N],
          s'.readMem output_ptr (pass2Off N TILE_N c s i) = pass2Val N TILE_N c cc S s input_ptr i)
      ∧ (∀ o, (∀ i : TileIndex [TILE_N], pass2Off N TILE_N c s i ≠ o) →
          s'.readMem output_ptr o = s.readMem output_ptr o)
      ∧ s'.regs .nat [] "prev_multiple" = s.regs .nat [] "prev_multiple" := by
  simp only [loopApass2body]
  -- n_offsets = c*TILE_N + arange
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => c * TILE_N + idx.1.val)) (by
      rw [evalOp_add, evalOp_ref_setReg_same, evalOp_arange]; rfl))]
  -- offset = pid*N + n_offsets
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => s.pid * N + (c * TILE_N + idx.1.val))) (by
      rw [evalOp_add, evalOp_mul, evalOp_constNat]
      simp only [evalOp_ref, BlockState.setReg_ne_name, BlockState.setReg_same, hpid, ne_eq,
        String.reduceEq, not_false_eq_true]
      rfl))]
  -- input_ptrs = ptrBase input_ptr + offset
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      ((Region.cast input_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) (by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
      congr 1
      apply Tile.ext; intro idx
      show ((Region.cast input_ptr : RegionName), 0 + (s.pid * N + (c * TILE_N + idx.1.val))) = _
      rw [Nat.zero_add]))]
  -- inp = load(input_ptrs) [unmasked]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) : WithBot ℝ))) (by
      rw [evalOp_load_ptrRef_none "input_ptrs" _
        (Tile.mk (fun idx : TileIndex [TILE_N] =>
          ((Region.cast input_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val))))
        (by simp only [evalOp_ref, BlockState.setReg_same])]
      congr 1
      apply Tile.ext; intro idx
      simp only [Tile.mk]
      rw [BlockState.readMemValue_real]
      rfl))]
  -- e = exp(inp - m)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      WithBot.realExp (WithBot.realSub
        (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) : WithBot ℝ)
        ((cc : ℝ) : WithBot ℝ)))) (by
      rw [evalOp_exp, evalOp_sub]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind, hm]
      rfl))]
  -- out = e / z
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      WithBot.realDiv
        (WithBot.realExp (WithBot.realSub
          (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) : WithBot ℝ)
          ((cc : ℝ) : WithBot ℝ)))
        (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))) (by
      rw [evalOp_div]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind, hz]
      rfl))]
  -- output_ptrs = ptrBase output_ptr + offset
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      ((Region.cast output_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) (by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
      congr 1
      apply Tile.ext; intro idx
      show ((Region.cast output_ptr : RegionName), 0 + (s.pid * N + (c * TILE_N + idx.1.val))) = _
      rw [Nat.zero_add]))]
  -- the pre-store state (after the seven assigns)
  set s7 := ((((((( s.setReg "start_n" .nat [] (Tile.scalar (c * TILE_N)) ).setReg "n_offsets" .nat [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => c * TILE_N + idx.1.val))).setReg "offset" .nat [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => s.pid * N + (c * TILE_N + idx.1.val)))).setReg "input_ptrs" .ptr [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => ((Region.cast input_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val))))).setReg "inp" .real [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) : WithBot ℝ)))).setReg "e" .real [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => WithBot.realExp (WithBot.realSub
        (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) : WithBot ℝ) ((cc : ℝ) : WithBot ℝ))))).setReg "out" .real [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => WithBot.realDiv (WithBot.realExp (WithBot.realSub
        (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) : WithBot ℝ) ((cc : ℝ) : WithBot ℝ)))
        (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ)))).setReg "output_ptrs" .ptr [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => ((Region.cast output_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) with hs7def
  -- evaluate the store
  have houtEval : evalOp (Op.ref .real [TILE_N] "out") s7
      = some (Tile.mk (fun idx : TileIndex [TILE_N] => WithBot.realDiv (WithBot.realExp (WithBot.realSub
          (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + idx.1.val))) : WithBot ℝ) ((cc : ℝ) : WithBot ℝ)))
          (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))) := by
    rw [evalOp_ref, hs7def]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
  have hptrEval : evalOp (Op.ref .ptr [TILE_N] "output_ptrs") s7
      = some (Tile.mk (fun idx : TileIndex [TILE_N] =>
          ((Region.cast output_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) := by
    rw [evalOp_ref, hs7def]
    simp only [BlockState.setReg_same]
  -- the post-store state is the ptr scatter foldl
  have hstore : stepStmt (Stmt.store .real [TILE_N] (MemAccess.ptr (Op.ref .ptr [TILE_N] "output_ptrs"))
      (Op.ref .real [TILE_N] "out") MaskOpt.none) s7
      = some ((TileShape.allIndices [TILE_N]).foldl
          (fun acc i => acc.writeMemTyped .real (Region.cast output_ptr)
            (s.pid * N + (c * TILE_N + i.1.val))
            (WithBot.realDiv (WithBot.realExp (WithBot.realSub
              (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + i.1.val))) : WithBot ℝ) ((cc : ℝ) : WithBot ℝ)))
              (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))) s7) := by
    unfold stepStmt
    rw [houtEval]
    simp only [Option.bind_some, Option.bind_eq_bind, hptrEval, if_true]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  -- rewrite the typed store foldl into the plain `writeMem` library form
  set ofn : TileIndex [TILE_N] → Nat := fun i => s.pid * N + (c * TILE_N + i.1.val) with hofn
  set vfn : TileIndex [TILE_N] → ℝ := fun i => pass2Val N TILE_N c cc S s input_ptr i with hvfn
  have hfold : (TileShape.allIndices [TILE_N]).foldl
        (fun acc i => acc.writeMemTyped .real (Region.cast output_ptr)
          (s.pid * N + (c * TILE_N + i.1.val))
          (WithBot.realDiv (WithBot.realExp (WithBot.realSub
            (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + i.1.val))) : WithBot ℝ) ((cc : ℝ) : WithBot ℝ)))
            (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))) s7
      = (TileShape.allIndices [TILE_N]).foldl
          (fun acc i => acc.writeMem output_ptr (ofn i) (vfn i)) s7 := by
    refine List.foldl_ext _ _ s7 (fun acc i _ => ?_)
    rw [Region.cast_id, BlockState.writeMemTyped_real]
    congr 1
    show FloatDType.real.storeValue _ = vfn i
    rw [FloatDType.real_storeValue]
    simp only [hvfn, pass2Val]
    exact pass2_store_value _ cc S
  rw [hfold]
  set sF := (TileShape.allIndices [TILE_N]).foldl
      (fun acc i => acc.writeMem output_ptr (ofn i) (vfn i)) s7 with hsF
  have hinj : Function.Injective ofn := by
    intro a b hab
    rw [hofn] at hab
    simp only at hab
    exact BlockState.tileIndex1d_base_offset_injective (s.pid * N + c * TILE_N)
      (by simp only [← Nat.add_assoc] at hab ⊢; exact hab)
  -- `.pids` preserved by the writeMem foldl
  have hpidsF : sF.pids = s.pids := by
    rw [hsF]
    have : ∀ (l : List (TileIndex [TILE_N])) (t : BlockState),
        (l.foldl (fun acc i => acc.writeMem output_ptr (ofn i) (vfn i)) t).pids = t.pids := by
      intro l; induction l with
      | nil => intro t; rfl
      | cons hd tl ih => intro t; rw [List.foldl_cons, ih]; simp
    rw [this, hs7def]; simp only [BlockState.setReg_pids]
  have hregF : ∀ (dtype : TileDType) (shape : TileShape) (name : RegName),
      sF.regs dtype shape name = s7.regs dtype shape name := by
    intro dtype shape name; rw [hsF, BlockState.foldl_writeMem_regs]
  have hpidF : sF.regs .nat [] "pid_m" = some (Tile.scalar (s.pid)) := by
    rw [hregF, hs7def]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
    exact hpid
  have hmF : sF.regs .real [] "m" = some (Tile.scalar ((cc : ℝ) : WithBot ℝ)) := by
    rw [hregF, hs7def]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
    exact hm
  have hzF : sF.regs .real [] "z" = some (Tile.scalar (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ)) := by
    rw [hregF, hs7def]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
    exact hz
  -- readMem at `output_ptr` for an offset never written
  have hpreserve : ∀ o, (∀ i : TileIndex [TILE_N], ofn i ≠ o) →
      sF.readMem output_ptr o = s7.readMem output_ptr o := by
    intro o hno
    rw [hsF]
    have : ∀ (l : List (TileIndex [TILE_N])) (t : BlockState),
        (∀ i ∈ l, ofn i ≠ o) →
        (l.foldl (fun acc i => acc.writeMem output_ptr (ofn i) (vfn i)) t).readMem output_ptr o
          = t.readMem output_ptr o := by
      intro l; induction l with
      | nil => intro t _; rfl
      | cons hd tl ih =>
        intro t hh
        rw [List.foldl_cons, ih _ (fun i hi => hh i (List.mem_cons_of_mem hd hi))]
        rw [BlockState.writeMem_readMem, if_neg]
        rintro ⟨_, heq⟩
        exact hh hd (List.mem_cons_self) heq.symm
    exact this _ s7 (fun i _ => hno i)
  refine ⟨sF, rfl, hpidF, hmF, hzF, hpidsF, ?_, ?_, ?_, ?_⟩
  · -- other regions untouched
    intro rg o hrg
    rw [hsF]
    have : ∀ (l : List (TileIndex [TILE_N])) (t : BlockState),
        (l.foldl (fun acc i => acc.writeMem output_ptr (ofn i) (vfn i)) t).readMem rg o
          = t.readMem rg o := by
      intro l; induction l with
      | nil => intro t; rfl
      | cons hd tl ih =>
        intro t; rw [List.foldl_cons, ih]
        exact BlockState.writeMem_readMem_of_ne_region t output_ptr (ofn hd) (vfn hd) rg o hrg
    rw [this, hs7def]; simp only [BlockState.setReg_readMem]
  · -- readback at written offsets
    intro i
    rw [hsF, show pass2Off N TILE_N c s i = ofn i from by rw [hofn, pass2Off]]
    rw [BlockState.scatter_readback_nd s7 ofn vfn hinj i, hvfn]
  · -- readback at untouched offsets
    intro o hno
    rw [hpreserve o hno, hs7def]
    simp only [BlockState.setReg_readMem]
  · -- prev_multiple preserved
    rw [hregF, hs7def]
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]

set_option maxHeartbeats 4000000 in
/-- **Pass-2 loop-B body advance.** Like `loopApass2_body_step` but masked: only
in-range lanes (`c·TILE_N + ℓ < N`) store `pass2Val`; the rest are skipped. -/
theorem loopBpass2_body_step (N TILE_N : Nat) (output_ptr input_ptr : RegionName)
    (hne : output_ptr ≠ input_ptr) (c : Nat) (cc S : ℝ) (s : BlockState)
    (hpid : s.regs .nat [] "pid_m" = some (Tile.scalar (s.pid)))
    (hm : s.regs .real [] "m" = some (Tile.scalar ((cc : ℝ) : WithBot ℝ)))
    (hz : s.regs .real [] "z" = some (Tile.scalar (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))) :
    ∃ s', stepStmts (loopBpass2body output_ptr input_ptr N TILE_N)
        (s.setReg "start_n" .nat [] (Tile.scalar (c * TILE_N))) = some s'
      ∧ s'.regs .nat [] "pid_m" = some (Tile.scalar (s.pid))
      ∧ s'.regs .real [] "m" = some (Tile.scalar ((cc : ℝ) : WithBot ℝ))
      ∧ s'.regs .real [] "z" = some (Tile.scalar (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))
      ∧ s'.pids = s.pids
      ∧ (∀ rg o, rg ≠ output_ptr → s'.readMem rg o = s.readMem rg o)
      ∧ (∀ i : TileIndex [TILE_N], c * TILE_N + i.1.val < N →
          s'.readMem output_ptr (pass2Off N TILE_N c s i) = pass2Val N TILE_N c cc S s input_ptr i)
      ∧ (∀ o, (∀ i : TileIndex [TILE_N], c * TILE_N + i.1.val < N → pass2Off N TILE_N c s i ≠ o) →
          s'.readMem output_ptr o = s.readMem output_ptr o)
      ∧ s'.regs .nat [] "prev_multiple" = s.regs .nat [] "prev_multiple" := by
  simp only [loopBpass2body]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => c * TILE_N + idx.1.val)) (by
      rw [evalOp_add, evalOp_ref_setReg_same, evalOp_arange]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => s.pid * N + (c * TILE_N + idx.1.val))) (by
      rw [evalOp_add, evalOp_mul, evalOp_constNat]
      simp only [evalOp_ref, BlockState.setReg_ne_name, BlockState.setReg_same, hpid, ne_eq,
        String.reduceEq, not_false_eq_true]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      ((Region.cast input_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) (by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, Option.bind_some, Option.bind_eq_bind]
      congr 1
      apply Tile.ext; intro idx
      show ((Region.cast input_ptr : RegionName), 0 + (s.pid * N + (c * TILE_N + idx.1.val))) = _
      rw [Nat.zero_add]))]
  -- mask = n_offsets < N
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => decide (c * TILE_N + idx.1.val < N))) (by
      rw [evalOp_lt, evalOp_constNat]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
      rfl))]
  -- masked load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] => loopBload N TILE_N c s input_ptr idx)) (by
      rw [evalOp_load_ptrRef_maskOther "input_ptrs"
        (Op.ref .bool [TILE_N] "mask") (Op.negInf.broadcast [TILE_N]) _
        (Tile.mk (fun idx : TileIndex [TILE_N] =>
          ((Region.cast input_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val))))
        (Tile.mk (fun idx : TileIndex [TILE_N] => decide (c * TILE_N + idx.1.val < N)))
        (Tile.mk (fun _ : TileIndex [TILE_N] => (none : WithBot ℝ)))
        (by simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
              String.reduceEq, not_false_eq_true])
        (by simp only [evalOp_ref, BlockState.setReg_same])
        (by simp only [evalOp, Option.bind_eq_bind, Option.bind_some]; rfl)]
      congr 1
      apply Tile.ext; intro idx
      simp only [Tile.mk, loopBload]
      by_cases hin : c * TILE_N + idx.1.val < N
      · simp only [hin, decide_true, if_true]
        rw [BlockState.readMemValue_real]; rfl
      · simp only [hin, decide_false, if_false]; rfl))]
  -- e = exp(inp - m)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      WithBot.realExp (WithBot.realSub (loopBload N TILE_N c s input_ptr idx) ((cc : ℝ) : WithBot ℝ)))) (by
      rw [evalOp_exp, evalOp_sub]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind, hm]
      rfl))]
  -- out = e / z
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      WithBot.realDiv (WithBot.realExp (WithBot.realSub (loopBload N TILE_N c s input_ptr idx) ((cc : ℝ) : WithBot ℝ)))
        (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))) (by
      rw [evalOp_div]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind, hz]
      rfl))]
  -- output_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.mk (fun idx : TileIndex [TILE_N] =>
      ((Region.cast output_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) (by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
      congr 1
      apply Tile.ext; intro idx
      show ((Region.cast output_ptr : RegionName), 0 + (s.pid * N + (c * TILE_N + idx.1.val))) = _
      rw [Nat.zero_add]))]
  -- pre-store state
  set s8 := (((((((( s.setReg "start_n" .nat [] (Tile.scalar (c * TILE_N)) ).setReg "n_offsets" .nat [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => c * TILE_N + idx.1.val))).setReg "offset" .nat [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => s.pid * N + (c * TILE_N + idx.1.val)))).setReg "input_ptrs" .ptr [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => ((Region.cast input_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val))))).setReg "mask" .bool [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => decide (c * TILE_N + idx.1.val < N)))).setReg "inp" .real [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => loopBload N TILE_N c s input_ptr idx))).setReg "e" .real [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => WithBot.realExp (WithBot.realSub (loopBload N TILE_N c s input_ptr idx) ((cc : ℝ) : WithBot ℝ))))).setReg "out" .real [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => WithBot.realDiv (WithBot.realExp (WithBot.realSub (loopBload N TILE_N c s input_ptr idx) ((cc : ℝ) : WithBot ℝ)))
        (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ)))).setReg "output_ptrs" .ptr [TILE_N]
    (Tile.mk (fun idx : TileIndex [TILE_N] => ((Region.cast output_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) with hs8def
  have houtEval : evalOp (Op.ref .real [TILE_N] "out") s8
      = some (Tile.mk (fun idx : TileIndex [TILE_N] => WithBot.realDiv (WithBot.realExp (WithBot.realSub
          (loopBload N TILE_N c s input_ptr idx) ((cc : ℝ) : WithBot ℝ)))
          (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ))) := by
    rw [evalOp_ref, hs8def]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
  have hptrEval : evalOp (Op.ref .ptr [TILE_N] "output_ptrs") s8
      = some (Tile.mk (fun idx : TileIndex [TILE_N] =>
          ((Region.cast output_ptr : RegionName), s.pid * N + (c * TILE_N + idx.1.val)))) := by
    rw [evalOp_ref, hs8def]; simp only [BlockState.setReg_same]
  have hmaskEval : evalOp (Op.ref .bool [TILE_N] "mask") s8
      = some (Tile.mk (fun idx : TileIndex [TILE_N] => decide (c * TILE_N + idx.1.val < N))) := by
    rw [evalOp_ref, hs8def]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]
  -- masked store
  set actfn : TileIndex [TILE_N] → Bool := fun idx => decide (c * TILE_N + idx.1.val < N) with hactfn
  set ofn : TileIndex [TILE_N] → Nat := fun i => s.pid * N + (c * TILE_N + i.1.val) with hofn
  set vfn : TileIndex [TILE_N] → ℝ := fun i => pass2Val N TILE_N c cc S s input_ptr i with hvfn
  have hstore : stepStmt (Stmt.store .real [TILE_N] (MemAccess.ptr (Op.ref .ptr [TILE_N] "output_ptrs"))
      (Op.ref .real [TILE_N] "out") (MaskOpt.mask (Op.ref .bool [TILE_N] "mask"))) s8
      = some ((TileShape.allIndices [TILE_N]).foldl
          (fun acc i => if actfn i then acc.writeMem output_ptr (ofn i) (vfn i) else acc) s8) := by
    unfold stepStmt
    rw [houtEval]
    simp only [Option.bind_some, Option.bind_eq_bind, hmaskEval, hptrEval, Option.map_some]
    congr 1
    refine List.foldl_ext _ _ s8 (fun acc i _ => ?_)
    show (if actfn i then acc.writeMemTyped .real (Region.cast output_ptr) (ofn i)
        (WithBot.realDiv (WithBot.realExp (WithBot.realSub (loopBload N TILE_N c s input_ptr i)
          ((cc : ℝ) : WithBot ℝ))) (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ)) else acc)
      = (if actfn i then acc.writeMem output_ptr (ofn i) (vfn i) else acc)
    cases ha : actfn i
    · simp only [Bool.false_eq_true, if_false]
    · simp only [if_true]
      rw [Region.cast_id, BlockState.writeMemTyped_real]
      congr 1
      show FloatDType.real.storeValue _ = vfn i
      rw [FloatDType.real_storeValue]
      have hin : c * TILE_N + i.1.val < N := by simpa [hactfn] using ha
      have hld : loopBload N TILE_N c s input_ptr i
          = (some (s.readMem input_ptr (s.pid * N + (c * TILE_N + i.1.val))) : WithBot ℝ) := by
        unfold loopBload; rw [if_pos hin]
      rw [hld]
      simp only [hvfn, pass2Val]
      exact pass2_store_value _ cc S
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  set sF := (TileShape.allIndices [TILE_N]).foldl
      (fun acc i => if actfn i then acc.writeMem output_ptr (ofn i) (vfn i) else acc) s8 with hsF
  have hinj : Function.Injective ofn := by
    intro a b hab
    rw [hofn] at hab; simp only at hab
    exact BlockState.tileIndex1d_base_offset_injective (s.pid * N + c * TILE_N)
      (by simp only [← Nat.add_assoc] at hab ⊢; exact hab)
  -- registers / pids / other regions
  have hpidsF : sF.pids = s.pids := by
    rw [hsF]
    have : ∀ (l : List (TileIndex [TILE_N])) (t : BlockState),
        (l.foldl (fun acc i => if actfn i then acc.writeMem output_ptr (ofn i) (vfn i) else acc) t).pids = t.pids := by
      intro l; induction l with
      | nil => intro t; rfl
      | cons hd tl ih => intro t; rw [List.foldl_cons]; cases actfn hd <;> simp [ih]
    rw [this, hs8def]; simp only [BlockState.setReg_pids]
  have hregF : ∀ (dtype : TileDType) (shape : TileShape) (name : RegName),
      sF.regs dtype shape name = s8.regs dtype shape name := by
    intro dtype shape name
    rw [hsF]
    have : ∀ (l : List (TileIndex [TILE_N])) (t : BlockState),
        (l.foldl (fun acc i => if actfn i then acc.writeMem output_ptr (ofn i) (vfn i) else acc) t).regs dtype shape name
          = t.regs dtype shape name := by
      intro l; induction l with
      | nil => intro t; rfl
      | cons hd tl ih => intro t; rw [List.foldl_cons]; cases actfn hd <;> simp [ih]
    rw [this]
  have hpidF : sF.regs .nat [] "pid_m" = some (Tile.scalar (s.pid)) := by
    rw [hregF, hs8def]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]; exact hpid
  have hmF : sF.regs .real [] "m" = some (Tile.scalar ((cc : ℝ) : WithBot ℝ)) := by
    rw [hregF, hs8def]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]; exact hm
  have hzF : sF.regs .real [] "z" = some (Tile.scalar (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ)) := by
    rw [hregF, hs8def]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true]; exact hz
  refine ⟨sF, rfl, hpidF, hmF, hzF, hpidsF, ?_, ?_, ?_, ?_⟩
  · -- other regions
    intro rg o hrg
    rw [hsF]
    have : ∀ (l : List (TileIndex [TILE_N])) (t : BlockState),
        (l.foldl (fun acc i => if actfn i then acc.writeMem output_ptr (ofn i) (vfn i) else acc) t).readMem rg o
          = t.readMem rg o := by
      intro l; induction l with
      | nil => intro t; rfl
      | cons hd tl ih =>
        intro t; rw [List.foldl_cons]
        cases h : actfn hd
        · simp only [h, Bool.false_eq_true, if_false]; exact ih _
        · simp only [h, if_true]; rw [ih]
          exact BlockState.writeMem_readMem_of_ne_region t output_ptr (ofn hd) (vfn hd) rg o hrg
    rw [this, hs8def]; simp only [BlockState.setReg_readMem]
  · -- readback at in-range written offsets
    intro i hin
    rw [hsF, show pass2Off N TILE_N c s i = ofn i from by rw [hofn, pass2Off]]
    rw [BlockState.scatter_readback_masked_nd s8 ofn vfn actfn hinj i]
    have ha : actfn i = Bool.true := by rw [hactfn]; exact decide_eq_true_eq.mpr hin
    rw [if_pos ha, hvfn]
  · -- readback at offsets never written by an active lane
    intro o hno
    rw [hsF]
    have hpres : ∀ (l : List (TileIndex [TILE_N])) (t : BlockState),
        (∀ i ∈ l, actfn i → ofn i ≠ o) →
        (l.foldl (fun acc i => if actfn i then acc.writeMem output_ptr (ofn i) (vfn i) else acc) t).readMem output_ptr o
          = t.readMem output_ptr o := by
      intro l; induction l with
      | nil => intro t _; rfl
      | cons hd tl ih =>
        intro t hh
        rw [List.foldl_cons]
        cases h : actfn hd
        · simp only [h, Bool.false_eq_true, if_false]
          exact ih _ (fun i hi => hh i (List.mem_cons_of_mem hd hi))
        · simp only [h, if_true]
          rw [ih _ (fun i hi => hh i (List.mem_cons_of_mem hd hi))]
          rw [BlockState.writeMem_readMem, if_neg]
          rintro ⟨_, heq⟩
          exact hh hd (List.mem_cons_self) (by rw [h]) heq.symm
    rw [hpres _ s8 (fun i _ ha => by
      have hin : c * TILE_N + i.1.val < N := by simpa [hactfn] using ha
      have := hno i hin
      rw [hofn]; rw [pass2Off] at this; exact this)]
    rw [hs8def]; simp only [BlockState.setReg_readMem]
  · -- prev_multiple preserved
    rw [hregF, hs8def]
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]

/-! ### Pass-2 whole-loop runs

The two pass-2 store loops run via `forRangeDyn_inv` carrying `pass2Inv`: the
scalar `m`/`z`/`pid_m` registers are preserved, the input region is untouched
(so loads read the original input), `pid`/`pids` are fixed, and the `output_ptr`
readback equals the genuine `softmaxOptimizeFullSpec` on the columns written so
far. `cc` is the (irrelevant) running max from the reduction; `S` is the running
denominator `Σ_{j<N} exp(input[pid·N+j])`. -/

/-- Pass-2 invariant after counter `start_n = i`: scalar regs preserved, input
untouched, `output_ptr[pid·N+j]` holds the spec for `j < i` (and `j < N`),
unchanged otherwise. -/
noncomputable def pass2Inv
    (output_ptr input_ptr : RegionName) (N TILE_N : Nat) (cc S : ℝ) (s0 : BlockState)
    (i : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧
  s.regs .nat [] "pid_m" = some (Tile.scalar (s0.pid)) ∧
  s.regs .real [] "m" = some (Tile.scalar ((cc : ℝ) : WithBot ℝ)) ∧
  s.regs .real [] "z" = some (Tile.scalar (((Real.exp (-cc) * S : ℝ)) : WithBot ℝ)) ∧
  (∀ o, s.readMem input_ptr o = s0.readMem input_ptr o) ∧
  (∀ j : Nat, j < N →
    s.readMem output_ptr (s0.pid * N + j)
      = if j < i then
          Real.exp (s0.readMem input_ptr (s0.pid * N + j)) / S
        else s0.readMem output_ptr (s0.pid * N + j)) ∧
  s.regs .nat [] "prev_multiple" = s0.regs .nat [] "prev_multiple"

/-- Pass-2 loop-A invariant: `pass2Inv` plus tile-alignment of the counter. -/
noncomputable def pass2InvA
    (output_ptr input_ptr : RegionName) (N TILE_N : Nat) (cc S : ℝ) (s0 : BlockState)
    (i : Nat) (s : BlockState) : Prop :=
  pass2Inv output_ptr input_ptr N TILE_N cc S s0 i s ∧ TILE_N ∣ i

set_option maxHeartbeats 4000000 in
/-- **Pass-2 loop-A whole-loop run.** The pass-2 full-tile store loop preserves
`pass2InvA`. -/
theorem pass2LoopA_run (N TILE_N : Nat) (hT : 0 < TILE_N)
    (output_ptr input_ptr : RegionName) (hne : output_ptr ≠ input_ptr)
    (cc S : ℝ) (s0 sInit : BlockState)
    (hStop : evalOp (Op.ref .nat [] "prev_multiple") sInit
        = some (Tile.scalar (prevMultiple N TILE_N)))
    (hInv : pass2InvA output_ptr input_ptr N TILE_N cc S s0 0 sInit) :
    ∃ sFinal,
      stepStmt (.forRangeDyn "start_n" (Op.constNat 0) (Op.ref .nat [] "prev_multiple")
          (Op.constNat TILE_N) (loopApass2body output_ptr input_ptr N TILE_N)) sInit = some sFinal ∧
      pass2InvA output_ptr input_ptr N TILE_N cc S s0 (prevMultiple N TILE_N) sFinal := by
  apply forRangeDyn_inv_aligned (start := 0) (stop := prevMultiple N TILE_N) (step := TILE_N)
    (P := pass2InvA output_ptr input_ptr N TILE_N cc S s0)
    (by simp [evalOp]) hStop (by simp [evalOp]) hT.ne' (Nat.zero_le _)
    (Dvd.intro 0 rfl) (TILE_N_dvd_prevMultiple N TILE_N) hInv
  intro i s hi hPi
  obtain ⟨⟨hpids, hpid, hm, hz, hinpEq, hout, hprevInv⟩, hdvd⟩ := hPi
  have hpidEq : s.pid = s0.pid := by unfold BlockState.pid; rw [hpids]
  have halign : i = (i / TILE_N) * TILE_N := (Nat.div_mul_cancel hdvd).symm
  set c := i / TILE_N with hc
  have hpidS : s.regs .nat [] "pid_m" = some (Tile.scalar s.pid) := by rw [hpid, hpidEq]
  obtain ⟨s', hstep, hpid', hm', hz', hpids', hother', hwritten, hpreserved, hprev'⟩ :=
    loopApass2_body_step N TILE_N output_ptr input_ptr hne c cc S s hpidS hm hz
  refine ⟨s', ?_, ?_, ?_⟩
  · rw [← halign] at hstep; exact hstep
  · -- new pass2Inv at i + TILE_N
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hpids']; exact hpids
    · rw [hpid', hpidEq]
    · exact hm'
    · exact hz'
    · -- input region untouched
      intro o
      rw [hother' input_ptr o hne.symm, hinpEq]
    · -- output readback advances by one tile
      intro j hj
      by_cases hjt : j < i + TILE_N
      · -- in the newly written tile, or earlier
        rw [if_pos hjt]
        by_cases hji : j < i
        · -- earlier tile: this lane's offset s0.pid*N+j was not touched by tile c
          rw [hpreserved (s0.pid * N + j) (by
            intro idx
            rw [pass2Off, hpidEq]
            -- s.pid*N + (c*TILE_N + idx) ≠ s0.pid*N + j  since c*TILE_N + idx ≥ i > j
            have : i ≤ c * TILE_N + idx.1.val := by rw [← halign]; exact Nat.le_add_right _ _
            omega)]
          rw [hout j hj, if_pos hji]
        · -- newly written tile: j = c*TILE_N + ℓ for ℓ = j - i
          have hjr : i ≤ j := Nat.le_of_not_lt hji
          have hlt : j - i < TILE_N := by omega
          set ℓ : TileIndex [TILE_N] := (⟨j - i, hlt⟩, PUnit.unit) with hℓ
          have hcol : c * TILE_N + ℓ.1.val = j := by
            show c * TILE_N + (j - i) = j; rw [← halign]; omega
          have hoff : pass2Off N TILE_N c s ℓ = s0.pid * N + j := by
            rw [pass2Off, hpidEq, hcol]
          have hw := hwritten ℓ
          rw [hoff] at hw
          rw [hw, pass2Val, hcol, hpidEq, hinpEq]
      · -- j ≥ i + TILE_N: untouched this tile, was untouched before too
        rw [if_neg hjt]
        rw [hpreserved (s0.pid * N + j) (by
          intro idx
          rw [pass2Off, hpidEq]
          have : c * TILE_N + idx.1.val < i + TILE_N := by
            rw [← halign]; have := idx.2; omega
          omega)]
        rw [hout j hj, if_neg (by omega)]
    · rw [hprev', hprevInv]
  · exact Dvd.dvd.add hdvd (dvd_refl _)

set_option maxHeartbeats 4000000 in
/-- **Pass-2 loop-B whole-loop run.** The pass-2 masked-tail store loop preserves
`pass2InvA`; in-range columns get the spec, masked-out columns are skipped. -/
theorem pass2LoopB_run (N TILE_N : Nat) (hT : 0 < TILE_N)
    (output_ptr input_ptr : RegionName) (hne : output_ptr ≠ input_ptr)
    (cc S : ℝ) (s0 sInit : BlockState)
    (hStart : evalOp (Op.ref .nat [] "prev_multiple") sInit
        = some (Tile.scalar (prevMultiple N TILE_N)))
    (hInv : pass2InvA output_ptr input_ptr N TILE_N cc S s0 (prevMultiple N TILE_N) sInit) :
    ∃ final sFinal,
      stepStmt (.forRangeDyn "start_n" (Op.ref .nat [] "prev_multiple") (Op.constNat N)
          (Op.constNat TILE_N) (loopBpass2body output_ptr input_ptr N TILE_N)) sInit = some sFinal ∧
      N ≤ final ∧
      pass2InvA output_ptr input_ptr N TILE_N cc S s0 final sFinal := by
  apply forRangeDyn_inv (start := prevMultiple N TILE_N) (stop := N) (step := TILE_N)
    (P := pass2InvA output_ptr input_ptr N TILE_N cc S s0)
    hStart (by simp [evalOp]) (by simp [evalOp]) hT.ne' hInv
  intro i s hi hPi
  obtain ⟨⟨hpids, hpid, hm, hz, hinpEq, hout, hprevInv⟩, hdvd⟩ := hPi
  have hpidEq : s.pid = s0.pid := by unfold BlockState.pid; rw [hpids]
  have halign : i = (i / TILE_N) * TILE_N := (Nat.div_mul_cancel hdvd).symm
  set c := i / TILE_N with hc
  have hpidS : s.regs .nat [] "pid_m" = some (Tile.scalar s.pid) := by rw [hpid, hpidEq]
  obtain ⟨s', hstep, hpid', hm', hz', hpids', hother', hwritten, hpreserved, hprev'⟩ :=
    loopBpass2_body_step N TILE_N output_ptr input_ptr hne c cc S s hpidS hm hz
  refine ⟨s', ?_, ?_, ?_⟩
  · rw [← halign] at hstep; exact hstep
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hpids']; exact hpids
    · rw [hpid', hpidEq]
    · exact hm'
    · exact hz'
    · intro o; rw [hother' input_ptr o hne.symm, hinpEq]
    · intro j hj
      by_cases hjt : j < i + TILE_N
      · rw [if_pos hjt]
        by_cases hji : j < i
        · -- earlier column: untouched by this tile
          rw [hpreserved (s0.pid * N + j) (by
            intro idx hidx
            rw [pass2Off, hpidEq]
            have : i ≤ c * TILE_N + idx.1.val := by rw [← halign]; exact Nat.le_add_right _ _
            omega)]
          rw [hout j hj, if_pos hji]
        · -- newly written column: j = c*TILE_N + ℓ, and j < N so mask active
          have hjr : i ≤ j := Nat.le_of_not_lt hji
          have hlt : j - i < TILE_N := by omega
          set ℓ : TileIndex [TILE_N] := (⟨j - i, hlt⟩, PUnit.unit) with hℓ
          have hcol : c * TILE_N + ℓ.1.val = j := by
            show c * TILE_N + (j - i) = j; rw [← halign]; omega
          have hinr : c * TILE_N + ℓ.1.val < N := by rw [hcol]; exact hj
          have hoff : pass2Off N TILE_N c s ℓ = s0.pid * N + j := by
            rw [pass2Off, hpidEq, hcol]
          have hw := hwritten ℓ hinr
          rw [hoff] at hw
          rw [hw, pass2Val, hcol, hpidEq, hinpEq]
      · rw [if_neg hjt]
        rw [hpreserved (s0.pid * N + j) (by
          intro idx hidx
          rw [pass2Off, hpidEq]
          have : c * TILE_N + idx.1.val < i + TILE_N := by
            rw [← halign]; have := idx.2; omega
          omega)]
        rw [hout j hj, if_neg (by omega)]
    · rw [hprev', hprevInv]
  · exact Dvd.dvd.add hdvd (dvd_refl _)

/-! ### Genuine top theorem for the full streaming surface

Assembling the seed, the two pass-1 loops (`pass1LoopA_run`/`pass1LoopB_run`),
the reduction collapse (`reduction_step`), and the two pass-2 store loops
(`pass2LoopA_run`/`pass2LoopB_run`), the full `softmax_kernel_online_v2_surface`
writes `softmaxOptimizeFullSpec` to every active output column. -/

set_option maxHeartbeats 8000000 in
/-- **exec-level full correctness.** For `0 < N`, `0 < TILE_N`,
`output_ptr ≠ input_ptr`, every successful run of the full surface kernel writes
the genuine full-row softmax `softmaxOptimizeFullSpec` to `output_ptr` at each
column `j < N` of row `pid`. -/
theorem softmax_kernel_online_v2_surface_exec_correct
    (output_ptr input_ptr : RegionName) (M N TILE_N : Nat)
    (hN : 0 < N) (hT : 0 < TILE_N) (hne : output_ptr ≠ input_ptr)
    (s s' : BlockState)
    (hExec : exec (softmax_kernel_online_v2_surface output_ptr input_ptr M N TILE_N).toAlgKernel s
        = some s') :
    ∀ j : Fin N,
      s'.readMem output_ptr (linearOffset s N j) = softmaxOptimizeFullSpec s input_ptr N j := by
  -- abbreviations
  set pm := prevMultiple N TILE_N with hpm
  -- `exec = stepStmts body`; expose the concrete erased statement list
  rw [exec] at hExec
  -- name the seed states
  set s1 := s.setReg "pid_m" .nat [] (Tile.scalar s.pid) with hs1
  -- the body is (defeq) the concrete statement list
  have hbody : stepStmts (softmax_kernel_online_v2_surface output_ptr input_ptr M N TILE_N).toAlgKernel.body s
      = stepStmts
        ([ Stmt.assign .nat [] "pid_m" (Op.programId 0),
           Stmt.assign .real [TILE_N] "m" (Op.full [TILE_N] Op.negInf),
           Stmt.assign .real [TILE_N] "z" (Op.full [TILE_N] (Op.const 0)),
           Stmt.assign .nat [] "prev_multiple"
             (Op.sub NumericDType.nat Broadcast.nil
               (Op.mul NumericDType.nat Broadcast.nil
                 (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat (N + TILE_N - 1)) (Op.constNat TILE_N))
                 (Op.constNat TILE_N))
               (Op.constNat TILE_N)),
           Stmt.forRangeDyn "start_n" (Op.constNat 0) (Op.ref .nat [] "prev_multiple")
             (Op.constNat TILE_N) (loopAbody input_ptr N TILE_N),
           Stmt.forRangeDyn "start_n" (Op.ref .nat [] "prev_multiple") (Op.constNat N)
             (Op.constNat TILE_N) (loopBbody input_ptr N TILE_N) ]
         ++ reductionStmts TILE_N
         ++ [ Stmt.assign .nat [] "prev_multiple"
                (Op.sub NumericDType.nat Broadcast.nil
                  (Op.mul NumericDType.nat Broadcast.nil
                    (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat (N + TILE_N - 1)) (Op.constNat TILE_N))
                    (Op.constNat TILE_N))
                  (Op.constNat TILE_N)),
              Stmt.forRangeDyn "start_n" (Op.constNat 0) (Op.ref .nat [] "prev_multiple")
                (Op.constNat TILE_N) (loopApass2body output_ptr input_ptr N TILE_N),
              Stmt.forRangeDyn "start_n" (Op.ref .nat [] "prev_multiple") (Op.constNat N)
                (Op.constNat TILE_N) (loopBpass2body output_ptr input_ptr N TILE_N) ]) s := rfl
  rw [hbody] at hExec
  -- ===== Phase 0: step the four seed assignments =====
  -- prev_multiple evaluation (used twice: seed + pass-2 reassign)
  have hpmEval : ∀ st : BlockState, evalOp
      (Op.sub NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil
          (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat (N + TILE_N - 1)) (Op.constNat TILE_N))
          (Op.constNat TILE_N))
        (Op.constNat TILE_N)) st = some (Tile.scalar (prevMultiple N TILE_N)) := by
    intro st
    have hfd : evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
            (Op.constNat (N + TILE_N - 1)) (Op.constNat TILE_N)) st
          = some (Tile.bop (IntegralDType.nat).floorDiv Broadcast.nil
              (Tile.scalar (N + TILE_N - 1)) (Tile.scalar TILE_N)) := by
      simp only [evalOp, evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
    simp only [evalOp_sub, evalOp_mul, evalOp_constNat, Option.bind_some, Option.bind_eq_bind, hfd]
    apply congrArg some
    apply Tile.ext; intro idx
    simp only [Tile.bop_data, Tile.scalar_data, prevMultiple]
    rfl
  -- flatten the `[..] ++ reductionStmts ++ [..]` literal into a cons-chain
  simp only [List.cons_append, List.nil_append, List.append_assoc] at hExec
  -- peel: pid_m, m, z, prev_multiple
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.programId 0) s = some (Tile.scalar s.pid) from by
          simp only [evalOp_programId]))] at hExec
  set sa := s.setReg "pid_m" .nat [] (Tile.scalar s.pid) with hsa
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.full [TILE_N] Op.negInf) sa
            = some ⟨fun _ : TileIndex [TILE_N] => (⊥ : WithBot ℝ)⟩ from by
          simp only [evalOp_full, evalOp_negInf]; rfl))] at hExec
  set sb := sa.setReg "m" .real [TILE_N] ⟨fun _ : TileIndex [TILE_N] => (⊥ : WithBot ℝ)⟩ with hsb
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.full [TILE_N] (Op.const 0)) sb
            = some ⟨fun _ : TileIndex [TILE_N] => ((0 : ℝ) : WithBot ℝ)⟩ from by
          simp only [evalOp_full, evalOp_const]; rfl))] at hExec
  set sc := sb.setReg "z" .real [TILE_N] ⟨fun _ : TileIndex [TILE_N] => ((0 : ℝ) : WithBot ℝ)⟩ with hsc
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (hpmEval sc))] at hExec
  set sSeed := sc.setReg "prev_multiple" .nat [] (Tile.scalar (prevMultiple N TILE_N)) with hsSeed
  -- ===== sSeed register facts =====
  have hSeed_pids : sSeed.pids = s.pids := by
    rw [hsSeed, hsc, hsb, hsa]
    simp only [BlockState.setReg_pids]
  have hSeed_mem : sSeed.mem = s.mem := by
    rw [hsSeed, hsc, hsb, hsa]; rfl
  have hSeed_pid : sSeed.pid = s.pid := by
    unfold BlockState.pid; rw [hSeed_pids]
  have hSeed_undef : ∀ rg o, sSeed.undef rg o = s.undef rg o := by
    intro rg o; rw [hsSeed, hsc, hsb, hsa]
    simp only [BlockState.setReg_undef]
  have hSeed_pidm : sSeed.regs .nat [] "pid_m" = some (Tile.scalar s.pid) := by
    rw [hsSeed, hsc, hsb]
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    rw [hsa]; simp only [BlockState.setReg_same]
  have hSeed_pm : evalOp (Op.ref .nat [] "prev_multiple") sSeed
      = some (Tile.scalar (prevMultiple N TILE_N)) := by
    rw [evalOp_ref, hsSeed]; simp only [BlockState.setReg_same]
  have hSeed_m : sSeed.regs .real [TILE_N] "m" = some
      ⟨fun idx : TileIndex [TILE_N] => (laneState s input_ptr N TILE_N idx.1 (0 / TILE_N)).1⟩ := by
    rw [hsSeed, hsc]
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    rw [hsb]; simp only [BlockState.setReg_same]
    congr 1; apply Tile.ext; intro idx
    simp only [Nat.zero_div, laneState, laneScores, List.range_zero, List.filterMap_nil,
      List.foldl_nil]
  have hSeed_z : sSeed.regs .real [TILE_N] "z" = some
      ⟨fun idx : TileIndex [TILE_N] =>
        (((laneState s input_ptr N TILE_N idx.1 (0 / TILE_N)).2 : ℝ) : WithBot ℝ)⟩ := by
    rw [hsSeed]
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    rw [hsc]; simp only [BlockState.setReg_same]
    congr 1; apply Tile.ext; intro idx
    simp only [Nat.zero_div, laneState, laneScores, List.range_zero, List.filterMap_nil,
      List.foldl_nil]
  have hSeed_prev : sSeed.regs .nat [] "prev_multiple"
      = some (Tile.scalar ((N + TILE_N - 1) / TILE_N * TILE_N - TILE_N)) := by
    rw [hsSeed]; simp only [BlockState.setReg_same]; rfl
  have hInv0 : pass1InvA input_ptr N TILE_N s 0 sSeed :=
    ⟨⟨hSeed_pids, hSeed_mem, hSeed_undef, hSeed_pidm, hSeed_m, hSeed_z, hSeed_prev⟩, ⟨0, rfl⟩⟩
  -- ===== Phase 1: the two pass-1 loops =====
  obtain ⟨sLoopA, hstepA, hInvA⟩ :=
    pass1LoopA_run N TILE_N hT input_ptr s sSeed hSeed_pm hInv0
  rw [stepStmts.cons_some hstepA] at hExec
  -- prev_multiple still readable in sLoopA from the carried invariant clause
  have hA_pm : evalOp (Op.ref .nat [] "prev_multiple") sLoopA
      = some (Tile.scalar (prevMultiple N TILE_N)) := by
    rw [evalOp_ref, hInvA.1.2.2.2.2.2.2]; rfl
  obtain ⟨finalB, sLoopB, hstepB, hfinalB, hInvB⟩ :=
    pass1LoopB_run N TILE_N hT input_ptr s sLoopA hA_pm hInvA
  rw [stepStmts.cons_some hstepB] at hExec
  -- after both pass-1 loops, m/z = laneState over `finalB` tiles, fully covering N
  obtain ⟨⟨hB_pids, hB_mem, hB_undef, hB_pidm, hB_m, hB_z, hB_prev⟩, hB_dvd⟩ := hInvB
  -- ===== Phase 2: the reduction collapse =====
  set cfin := finalB / TILE_N with hcfin_def
  have hcfin_mul : cfin * TILE_N = finalB := Nat.div_mul_cancel hB_dvd
  have hcov : N ≤ cfin * TILE_N := by rw [hcfin_mul]; exact hfinalB
  have hcfin_pos : 0 < cfin := by
    rcases Nat.eq_zero_or_pos cfin with h | h
    · exfalso; rw [h, Nat.zero_mul] at hcov; exact absurd hcov (Nat.not_le.mpr hN)
    · exact h
  have hB_pidEq : sLoopB.pid = s.pid := by unfold BlockState.pid; rw [hB_pids]
  have hB_cong : ∀ (ℓ : Fin TILE_N) (cc : Nat),
      laneState s input_ptr N TILE_N ℓ cc = laneState sLoopB input_ptr N TILE_N ℓ cc :=
    fun ℓ cc => (laneState_congr sLoopB s input_ptr N TILE_N hB_pidEq hB_mem ℓ cc).symm
  have hB_m' : sLoopB.regs .real [TILE_N] "m" = some
      ⟨fun idx : TileIndex [TILE_N] => (laneState s input_ptr N TILE_N idx.1 cfin).1⟩ := hB_m
  have hB_z' : sLoopB.regs .real [TILE_N] "z" = some
      ⟨fun idx : TileIndex [TILE_N] =>
        (((laneState s input_ptr N TILE_N idx.1 cfin).2 : ℝ) : WithBot ℝ)⟩ := hB_z
  obtain ⟨cmax, sRed, hstepRed, hRed_pids, hRed_mem, hRed_undef, hRed_pidm,
      hRed_m, hRed_z⟩ :=
    reduction_step N TILE_N hN hT input_ptr cfin hcfin_pos hcov s sLoopB
      hB_pidEq hB_cong hB_m' hB_z'
  -- the reduced denominator `S = Σ_{j<N} exp(input[pid·N+j])`
  set S : ℝ := ∑ j ∈ Finset.range N, Real.exp (sLoopB.readMem input_ptr (sLoopB.pid * N + j)) with hS_def
  -- thread `reductionStmts` through hExec
  rw [stepStmts.append_some hstepRed] at hExec
  -- ===== Phase 3: pass-2 reassign + two store loops =====
  -- peel the pass-2 `prev_multiple` reassignment
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (hpmEval sRed))] at hExec
  set sP0 := sRed.setReg "prev_multiple" .nat [] (Tile.scalar (prevMultiple N TILE_N)) with hsP0
  -- memory / pid bridges from `s` to the pass-2 seed `sP0`
  have hP0_pids : sP0.pids = s.pids := by
    rw [hsP0]; simp only [BlockState.setReg_pids]; rw [hRed_pids, hB_pids]
  have hP0_pidEq : sP0.pid = s.pid := by unfold BlockState.pid; rw [hP0_pids]
  have hP0_mem : sP0.mem = s.mem := by
    rw [hsP0]; show sRed.mem = s.mem; rw [hRed_mem, hB_mem]
  have hP0_readInp : ∀ o, sP0.readMem input_ptr o = s.readMem input_ptr o := by
    intro o; unfold BlockState.readMem; rw [hP0_mem]
  -- the reduced max value `cmax` and denominator `S` (over `s`)
  have hS_over_s : S = ∑ j ∈ Finset.range N, Real.exp (s.readMem input_ptr (s.pid * N + j)) := by
    rw [hS_def]; apply Finset.sum_congr rfl; intro j _
    have hmem : sLoopB.readMem input_ptr (sLoopB.pid * N + j) = s.readMem input_ptr (s.pid * N + j) := by
      unfold BlockState.readMem; rw [hB_mem, hB_pidEq]
    rw [hmem]
  -- pass-2 register facts on the seed `sP0`
  have hP0_pidm : sP0.regs .nat [] "pid_m" = some (Tile.scalar s.pid) := by
    rw [hsP0]
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    rw [hRed_pidm, hB_pidm]
  have hP0_m : sP0.regs .real [] "m" = some (Tile.scalar ((cmax : ℝ) : WithBot ℝ)) := by
    rw [hsP0]
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    exact hRed_m
  have hP0_z : sP0.regs .real [] "z" = some (Tile.scalar (((Real.exp (-cmax) * S : ℝ)) : WithBot ℝ)) := by
    rw [hsP0]
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    rw [hRed_z]
  have hP0_pm : evalOp (Op.ref .nat [] "prev_multiple") sP0
      = some (Tile.scalar (prevMultiple N TILE_N)) := by
    rw [evalOp_ref, hsP0]; simp only [BlockState.setReg_same]
  have hP0_prev : sP0.regs .nat [] "prev_multiple"
      = some (Tile.scalar (prevMultiple N TILE_N)) := by
    rw [hsP0]; simp only [BlockState.setReg_same]
  -- pass-2 invariant at counter 0 over seed `sP0`
  have hP2Inv0 : pass2InvA output_ptr input_ptr N TILE_N cmax S sP0 0 sP0 := by
    refine ⟨⟨rfl, ?_, hP0_m, hP0_z, fun o => rfl, ?_, rfl⟩, ⟨0, rfl⟩⟩
    · rw [hP0_pidEq]; exact hP0_pidm
    · intro j hj; rw [if_neg (Nat.not_lt.mpr (Nat.zero_le _))]
  -- pass-2 loop A (aligned → ends exactly at prevMultiple)
  obtain ⟨sP1, hstepP1, hP2InvA⟩ :=
    pass2LoopA_run N TILE_N hT output_ptr input_ptr hne cmax S sP0 sP0 hP0_pm hP2Inv0
  rw [stepStmts.cons_some hstepP1] at hExec
  -- prev_multiple still readable in sP1 from the carried invariant clause
  have hP1_pm : evalOp (Op.ref .nat [] "prev_multiple") sP1
      = some (Tile.scalar (prevMultiple N TILE_N)) := by
    rw [evalOp_ref, hP2InvA.1.2.2.2.2.2.2, hP0_prev]
  -- pass-2 loop B (single masked tail tile → ends at finalP2 ≥ N)
  obtain ⟨finalP2, sP2, hstepP2, hfinalP2, hP2InvB⟩ :=
    pass2LoopB_run N TILE_N hT output_ptr input_ptr hne cmax S sP0 sP1 hP1_pm hP2InvA
  rw [stepStmts.cons_some hstepP2, stepStmts.nil] at hExec
  -- hExec : some sP2 = some s'
  obtain rfl : sP2 = s' := by injection hExec
  -- final pass-2 invariant: output[s0.pid*N+j] = exp(input[j])/S for j < N
  obtain ⟨_, _, _, _, _, hP2_out, _⟩ := hP2InvB.1
  intro j
  have hjN : j.val < N := j.isLt
  have hjfin : j.val < finalP2 := Nat.lt_of_lt_of_le hjN hfinalP2
  have hread := hP2_out j.val hjN
  rw [if_pos hjfin] at hread
  -- rewrite the address `linearOffset s N j = s.pid * N + j = sP0.pid * N + j`
  rw [show linearOffset s N j = sP0.pid * N + j.val by
        rw [linearOffset, hP0_pidEq]]
  rw [hread, hP0_readInp, hP0_pidEq]
  rw [softmaxOptimizeFullSpec_eq_div, hS_over_s]

/-- **Public output summary (headline theorem).** The full online-softmax
surface `softmax_kernel_online_v2_surface` realizes the genuine full-row softmax
`softmaxOptimizeFullSpec` at every output column `j < N` of row `pid`: the total
write map sends column `j` to `output_ptr` at `linearOffset s N j`, and the
value read back there after any successful run is exactly the numerically
stabilized full-row softmax of the loaded input row. Stated via
`ComputeCorrect.Realizes` (per `bench/MAIN_THEOREM_CONVENTIONS.md` §4), wrapping
the exec-level engine lemma `softmax_kernel_online_v2_surface_exec_correct`. -/
theorem softmax_kernel_online_v2_output_summary
    (output_ptr input_ptr : RegionName) (M N TILE_N : Nat)
    (hN : 0 < N) (hT : 0 < TILE_N) (hne : output_ptr ≠ input_ptr)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := softmax_kernel_online_v2_surface output_ptr input_ptr M N TILE_N)
      (initialState := s)
      (write := fun j : Fin N => some (output_ptr, linearOffset s N j))
      (expected := fun j : Fin N => softmaxOptimizeFullSpec s input_ptr N j) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_kernel_online_v2_surface, ComputeExpr.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro j
  simp only [ComputeCorrect.OutputReadable.read_real]
  exact softmax_kernel_online_v2_surface_exec_correct output_ptr input_ptr M N TILE_N
    hN hT hne s s' hExec j

end VeriTile.Bench.TritonBenchG.SoftmaxOptimize



