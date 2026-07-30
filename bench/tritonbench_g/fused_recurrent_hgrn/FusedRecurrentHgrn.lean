import VeriTile.Triton

/-!
# `fused_recurrent_hgrn` — strict per-kernel correctness

`fused_recurrent_hgrn_fwd_kernel` is the HGRN forward recurrent state scan:
each program carries a hidden-state vector `b_h` across a `0..T` time loop,
updating `b_h = g_t * b_h + x_t` per step, optionally
seeded by an initial state and optionally storing the final state. The
companion backward kernel performs the reverse-time gradient scan.

## Scope

This file verifies **three hand-cut single-step slices** of the two
`@triton.jit` bodies: one forward loop body and the backward loop body's `dx`
and `dg` writebacks. It does **not** verify the launched kernels: the forward
surface `fused_recurrent_hgrn_fwd_surface` is only shown to lower to the
algorithm layer, and the backward surface `fused_recurrent_hgrn_bwd_surface`
appears in no correctness face at all. The host launch (grid shape,
`@triton.autotune` config selection over `BD`, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*. Because the
program ids are universally quantified, each per-program statement covers every
program of the grid.

## Proof architecture

```
fused_recurrent_hgrn_output_summary_general               ← TOP THEOREM
  ├─ fused_recurrent_hgrn_fwd_surface_toAlgorithm_supported       (lowering only)
  ├─ fused_recurrent_hgrn_forward_step_closed_form                (hgrnStateClosed forward)
  ├─ fused_recurrent_hgrn_bwd_dx_step_store_slice_compute_correct
  └─ fused_recurrent_hgrn_bwd_dg_step_store_slice_compute_correct
```

## Modeling boundary — read before trusting anything below

Arithmetic is over `ℝ`, not bit-accurate IEEE float; dtype `.to(...)` casts
erase to the identity post-erasure. `@triton.autotune` (the `BD` config sweep)
is not modeled — `BD` is a free parameter. The step slices reproduce Python's
`mask = o_d < D` / `other = 0` loads faithfully, so no full-tile side condition
is needed. What is **outside** every claim in this file:

* **The cross-step folds.** Neither the forward `range(0, T)` fold threading
  `b_h` nor the backward reverse-time fold threading `b_dh` is modeled. Each
  carry is presented to its slice as a materialized fiction region (`BHPrev`,
  `DHPrev`) and constrained — where constrained at all — by an *assumed*
  hypothesis. In particular the forward carry invariant
  `BHPrev = hgrnStateClosed(i_t)` is **not** propagated by any clause: clause 2
  writes `O` at the time-indexed `outOffset s i_t`, while the hypothesis is
  about `BHPrev` at the time-free `bhOffset` — a different region at a different
  offset. There is no bridging lemma and no base case.
* **The `STORE_FINAL_STATE` writeback.** No correctness face. The former
  `final_state_store_slice` face was a masked memcpy (load address
  `BHFinal + i_bh·D + offs_d`, store address `Ht + i_bh·D + offs_d` — identical
  under the same mask) whose only content was the assumption
  `BHFinal = hgrnStateClosed(T)`; it has been deleted rather than presented as a
  result about `ht`.
* **The backward `b_o` re-indexing.** Python reads `b_o` from the *previous*
  output row (`p_o` starts at row `T-2`, so at reverse step `i` it reads row
  `i-1`) and from `h0` at `i = 0`. The `dg` slice reads the fiction region `BO`
  at row `i_t`; that region is *defined* to present the previous row, and no
  theorem connects it to `O` or `h0`. The `i = 0` / `h0` branch is unmodelled.
* **Region distinctness.** No `≠` hypotheses are stated, and none are needed:
  each slice performs all of its loads before its single store, and every
  `expected` is a function of the *initial* state, so aliasing cannot falsify
  any face below.
-/

namespace VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `fused_recurrent_hgrn_output_summary_general` — shape-general,
genuine closed form `hgrnStateClosed` over the input regions, but scoped to three
single-step **slices** (one forward body, the backward `dx`/`dg` writebacks), not
to the launched kernels. -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-- Faithful transcription of `fused_recurrent_hgrn.py`'s
`fused_recurrent_hgrn_fwd_kernel`.

The forward recurrence is a regular `0..T` loop and is represented directly,
including the optional initial-state load and final-state store. -/
def fused_recurrent_hgrn_fwd_surface
    (x g o h0 ht : RegionName) (T D BD : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  o_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = o_d < $(D)
  p_x = x + i_bh * $(T) * $(D) + o_d
  p_g = g + i_bh * $(T) * $(D) + o_d
  p_o = o + i_bh * $(T) * $(D) + o_d
  b_h = tl.zeros([$(BD)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = h0 + i_bh * $(D) + o_d
    b_h += tl.load(p_h0, mask=mask, other=0).to(tl.float32)
  }
  for _i in range($(0), $(T), $(1)) {
    b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
    b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    b_h = b_g * b_h + b_x
    tl.store(p_o, (b_h).to(p_o.dtype.element_ty), mask=mask)
    p_x += $(D)
    p_g += $(D)
    p_o += $(D)
  }
  if STORE_FINAL_STATE {
    p_ht = ht + i_bh * $(D) + o_d
    tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty), mask=mask)
  }
}

/-- The forward HGRN recurrent surface lowers to the algorithm layer, including
optional initial-state load, recurrent updates, output stores, and optional
final-state store. -/
theorem fused_recurrent_hgrn_fwd_surface_toAlgorithm_supported
    (x g o h0 ht : RegionName) (T D BD : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ∃ alg, (fused_recurrent_hgrn_fwd_surface x g o h0 ht T D BD
      USE_INITIAL_STATE STORE_FINAL_STATE).toAlgorithm? = Except.ok alg := by
  simp [fused_recurrent_hgrn_fwd_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription of `fused_recurrent_hgrn.py`'s
`fused_recurrent_hgrn_bwd_kernel`.

The Python kernel traverses time in reverse and decrements pointers; the DSL
surface preserves that reverse range and pointer movement directly. -/
def fused_recurrent_hgrn_bwd_surface
    (G O H0 DX DG DO : RegionName) (T D BD : Nat)
    (USE_INITIAL_STATE : Bool) :
    ComputeKernel := triton {
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  o_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = o_d < $(D)
  p_g = G + (i_bh * $(T) + $(T) - $(1)) * $(D) + o_d
  p_o = O + (i_bh * $(T) + $(T) - $(2)) * $(D) + o_d
  p_dx = DX + (i_bh * $(T) + $(T) - $(1)) * $(D) + o_d
  p_dg = DG + (i_bh * $(T) + $(T) - $(1)) * $(D) + o_d
  p_do = DO + (i_bh * $(T) + $(T) - $(1)) * $(D) + o_d
  b_dh = tl.zeros([$(BD)], dtype=tl.float32)
  for i in range($(T) - $(1), -$(1), -$(1)) {
    b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)
    if i > 0 {
      b_o = tl.load(p_o, mask=mask, other=0).to(tl.float32)
    } else {
      if USE_INITIAL_STATE {
        b_o = tl.load(H0 + i_bh * $(D) + o_d, mask=mask, other=0).to(tl.float32)
      } else {
        b_o = tl.zeros([$(BD)], dtype=tl.float32)
      }
    }
    b_dh = b_dh + b_do
    b_dx = b_dh
    b_dg = b_dh * b_o
    b_dh = b_dh * b_g
    tl.store(p_dx, (b_dx).to(p_dx.dtype.element_ty), mask=mask)
    tl.store(p_dg, (b_dg).to(p_dg.dtype.element_ty), mask=mask)
    p_g -= $(D)
    p_o -= $(D)
    p_dx -= $(D)
    p_dg -= $(D)
    p_do -= $(D)
  }
}

/-- The backward HGRN surface lowers with the Python reverse
`range(T - 1, -1, -1)` loop, pointer decrements, and initial-state branch
preserved. -/
theorem fused_recurrent_hgrn_bwd_surface_toAlgorithm_supported
    (G O H0 DX DG DO : RegionName) (T D BD : Nat)
    (USE_INITIAL_STATE : Bool) :
    (fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO T D BD
        USE_INITIAL_STATE).toAlgorithm? =
      Except.ok
        (fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO T D BD
          USE_INITIAL_STATE).toAlgKernel := by
  simp [fused_recurrent_hgrn_bwd_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

def dIndex (s : BlockState) (BD : Nat) (i : Fin BD) : Nat :=
  s.pids 0 * BD + i.val

def active (s : BlockState) (D BD : Nat) (i : Fin BD) : Prop :=
  dIndex s BD i < D

instance activeDecidable (s : BlockState) (D BD : Nat) (i : Fin BD) :
    Decidable (active s D BD i) := by
  unfold active
  infer_instance

def bhOffset (s : BlockState) (D BD : Nat) (i : Fin BD) : Nat :=
  s.pids 1 * D + dIndex s BD i

def outOffset (s : BlockState) (i_t T D BD : Nat) (i : Fin BD) : Nat :=
  (s.pids 1 * T + i_t) * D + dIndex s BD i

/-- One forward recurrence step:
`b_h = b_g * b_h + b_x`, then masked store to the current output row. -/
def fused_recurrent_hgrn_forward_step_store_slice
    (BHPrev X G O : RegionName) (i_t T D BD : Nat) :
    ComputeKernel := triton {
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  prev = tl.load(BHPrev + i_bh * $(D) + offs_d, mask=mask, other=0.0)
  b_x = tl.load(X + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_g = tl.load(G + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_h = b_g * prev + b_x
  tl.store(O + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    (b_h).to(O.dtype.element_ty), mask=mask)
}

noncomputable def forwardStepValue
    (s : BlockState) (BHPrev X G : RegionName) (i_t T D BD : Nat)
    (i : Fin BD) : ℝ :=
  s.readMem G (outOffset s i_t T D BD i) *
    s.readMem BHPrev (bhOffset s D BD i) +
  s.readMem X (outOffset s i_t T D BD i)

theorem fused_recurrent_hgrn_forward_step_store_slice_correct
    (BHPrev X G O : RegionName) (i_t T D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i)) :
    ∀ i : Fin BD,
      let outAddr := outOffset s i_t T D BD i
      (exec (fused_recurrent_hgrn_forward_step_store_slice BHPrev X G O
          i_t T D BD) s).map (·.readMem O outAddr)
        = some (if active s D BD i then
            forwardStepValue s BHPrev X G i_t T D BD i
          else s.readMem O outAddr) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BD] =>
        (s.pids 1 * T + i_t) * D + (s.pids 0 * BD + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s i_t T D BD a = outOffset s i_t T D BD b := by
      simpa [outOffset, dIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  simp [exec, fused_recurrent_hgrn_forward_step_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        outOffset, bhOffset, dIndex, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
  by_cases hi : s.pids 0 * BD + i.val < D
  · simp [active, forwardStepValue, outOffset, bhOffset, dIndex, hi,
      NumericDType.add, NumericDType.mul]
  · simp [active, outOffset, dIndex, hi]

theorem fused_recurrent_hgrn_forward_step_store_slice_compute_correct
    (BHPrev X G O : RegionName) (i_t T D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_hgrn_forward_step_store_slice BHPrev X G O
        i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (O, outOffset s i_t T D BD i)))
      (expected := fun i => forwardStepValue s BHPrev X G i_t T D BD i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_hgrn_forward_step_store_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fused_recurrent_hgrn_forward_step_store_slice_correct BHPrev X G O
    i_t T D BD s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Genuine closed-form forward state (over the *input* regions)

The forward loop body is `b_h = g_t · b_h + x_t`, output `o_t = b_h`. Unrolling
the recurrence gives, for the state after processing time steps `0,1,…,n-1`,

```
b_h^(n)[i] = seed[i] · ∏_{j<n} g_j[i]  +  Σ_{t<n} x_t[i] · ∏_{t<j<n} g_j[i]
```

with `seed = h0` when `USE_INITIAL_STATE` else `0` — `hgrnStateClosed` below, a
standalone specification over the *input* regions `x, g, h0`, never a read-back
of the kernel's own output. The kernel's output at time row `i_t` is the
*post-update* state `b_h^(i_t + 1)`, and the optional final state is
`hgrnStateClosed` at `n = T`. -/

/-- `g_t[i]` at the kernel's exact time-row layout. -/
noncomputable def gVal (s : BlockState) (g : RegionName) (T D BD : Nat)
    (t : Nat) (i : Fin BD) : ℝ :=
  s.readMem g (outOffset s t T D BD i)

/-- `x_t[i]` at the kernel's exact time-row layout. -/
noncomputable def xVal (s : BlockState) (x : RegionName) (T D BD : Nat)
    (t : Nat) (i : Fin BD) : ℝ :=
  s.readMem x (outOffset s t T D BD i)

/-- Seeded initial state `b_h^(0)`: `h0` if `USE_INITIAL_STATE` else `0`. -/
noncomputable def stateSeed (s : BlockState) (h0 : RegionName)
    (USE_INITIAL_STATE : Bool) (D BD : Nat) (i : Fin BD) : ℝ :=
  if USE_INITIAL_STATE then s.readMem h0 (bhOffset s D BD i) else 0

/-- **Genuine closed form for the forward state after `n` steps**, channel `i`:
`seed · ∏_{j<n} g_j + Σ_{t<n} x_t · ∏_{t<j<n} g_j`. A standalone specification
over the input regions `x, g, h0` — never a read-back of the kernel's own
output. -/
noncomputable def hgrnStateClosed
    (s : BlockState) (x g h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (T D BD n : Nat) (i : Fin BD) : ℝ :=
  stateSeed s h0 USE_INITIAL_STATE D BD i *
      (∏ j ∈ Finset.range n, gVal s g T D BD j i) +
    ∑ t ∈ Finset.range n,
      xVal s x T D BD t i *
        (∏ j ∈ Finset.Ico (t + 1) n, gVal s g T D BD j i)

/-- **The forward state carry-fold recurrence.** Unrolling one step:
`b_h^(n+1) = g_n · b_h^(n) + x_n`. This is the exact closed-form counterpart of
the Python loop body `b_h = b_g * b_h + b_x`. -/
theorem hgrnStateClosed_succ
    (s : BlockState) (x g h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (T D BD n : Nat) (i : Fin BD) :
    hgrnStateClosed s x g h0 USE_INITIAL_STATE T D BD (n + 1) i
      = gVal s g T D BD n i *
          hgrnStateClosed s x g h0 USE_INITIAL_STATE T D BD n i
        + xVal s x T D BD n i := by
  unfold hgrnStateClosed
  rw [Finset.prod_range_succ, Finset.sum_range_succ]
  rw [show Finset.Ico (n + 1) (n + 1) = (∅ : Finset Nat) from by simp,
      Finset.prod_empty, mul_one]
  have hsum :
      (∑ t ∈ Finset.range n,
          xVal s x T D BD t i *
            ∏ j ∈ Finset.Ico (t + 1) (n + 1), gVal s g T D BD j i)
        = (∑ t ∈ Finset.range n,
            xVal s x T D BD t i *
              ∏ j ∈ Finset.Ico (t + 1) n, gVal s g T D BD j i)
          * gVal s g T D BD n i := by
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro t ht
    simp only [Finset.mem_range] at ht
    rw [Finset.prod_Ico_succ_top (by omega : t + 1 ≤ n)]
    ring
  rw [hsum]; ring

/-- **Forward carry-fold step (genuine).** If the materialized previous-state
buffer `BHPrev` holds the genuine `i_t`-step folded state
`hgrnStateClosed(i_t)`, then one loop body — `forwardStepValue`, i.e.
`g_{i_t} · BHPrev + x_{i_t}` — produces exactly the genuine `(i_t+1)`-step
folded state `hgrnStateClosed(i_t + 1)`, the value stored to the output row. -/
theorem forwardStepValue_eq_hgrnStateClosed_succ
    (s : BlockState) (BHPrev X G h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (i_t T D BD : Nat)
    (hPrev : ∀ i : Fin BD,
      s.readMem BHPrev (bhOffset s D BD i)
        = hgrnStateClosed s X G h0 USE_INITIAL_STATE T D BD i_t i)
    (i : Fin BD) :
    forwardStepValue s BHPrev X G i_t T D BD i
      = hgrnStateClosed s X G h0 USE_INITIAL_STATE T D BD (i_t + 1) i := by
  rw [hgrnStateClosed_succ]
  unfold forwardStepValue
  rw [hPrev i]
  simp only [gVal, xVal]

/-- **Genuine forward output step.** One forward loop body, with the materialized
pre-update state buffer `BHPrev = hgrnStateClosed(i_t)`, realizes the genuine
closed form `hgrnStateClosed(i_t + 1)` (over the input regions `x, g, h0`) into
the output row. -/
theorem fused_recurrent_hgrn_forward_step_closed_form
    (BHPrev X G h0 O : RegionName) (USE_INITIAL_STATE : Bool)
    (i_t T D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i))
    (hPrev : ∀ i : Fin BD,
      s.readMem BHPrev (bhOffset s D BD i)
        = hgrnStateClosed s X G h0 USE_INITIAL_STATE T D BD i_t i) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_hgrn_forward_step_store_slice BHPrev X G O
        i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (O, outOffset s i_t T D BD i)))
      (expected := fun i =>
        hgrnStateClosed s X G h0 USE_INITIAL_STATE T D BD (i_t + 1) i) := by
  have h := fused_recurrent_hgrn_forward_step_store_slice_compute_correct
    BHPrev X G O i_t T D BD s hOutInj
  have hcong : (fun i => forwardStepValue s BHPrev X G i_t T D BD i)
      = (fun i =>
        hgrnStateClosed s X G h0 USE_INITIAL_STATE T D BD (i_t + 1) i) := by
    funext i
    exact forwardStepValue_eq_hgrnStateClosed_succ s BHPrev X G h0
      USE_INITIAL_STATE i_t T D BD hPrev i
  rwa [hcong] at h

/-! ## Offset-injectivity side conditions (dimension-general + Python shape)

The per-time output address `(i_bh·T + i_t)·D + i_d·BD + i` and the state
address `i_bh·D + i_d·BD + i` are injective in the lane `i` whenever `0 < BD`
(lanes are contiguous in the low digit). Honest structural side conditions; the
Python regression (`BD = 32`) satisfies them. -/

/-- **Dimension-general** per-time output address injectivity, given `0 < BD`. -/
theorem fused_recurrent_hgrn_out_offset_injective_general
    (s : BlockState) (i_t T D BD : Nat) (_hBD : 0 < BD) :
    Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i) := by
  intro a b h
  apply Fin.ext
  simp [outOffset, dIndex] at h
  omega

/-- Backward one-step `dx` formula:
`b_dh = b_dh_prev + b_do`, `b_dx = b_dh`, then masked store to `DX`. -/
def fused_recurrent_hgrn_bwd_dx_step_store_slice
    (DHPrev DO DX : RegionName) (i_t T D BD : Nat) :
    ComputeKernel := triton {
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  dh_prev = tl.load(DHPrev + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0)
  b_do = tl.load(DO + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_dh = dh_prev + b_do
  tl.store(DX + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    (b_dh).to(DX.dtype.element_ty), mask=mask)
}

noncomputable def bwdDxStepValue
    (s : BlockState) (DHPrev DO : RegionName) (i_t T D BD : Nat)
    (i : Fin BD) : ℝ :=
  s.readMem DHPrev (outOffset s i_t T D BD i) +
    s.readMem DO (outOffset s i_t T D BD i)

theorem fused_recurrent_hgrn_bwd_dx_step_store_slice_correct
    (DHPrev DO DX : RegionName) (i_t T D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i)) :
    ∀ i : Fin BD,
      let outAddr := outOffset s i_t T D BD i
      (exec (fused_recurrent_hgrn_bwd_dx_step_store_slice DHPrev DO DX
          i_t T D BD) s).map (·.readMem DX outAddr)
        = some (if active s D BD i then
            bwdDxStepValue s DHPrev DO i_t T D BD i
          else s.readMem DX outAddr) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BD] =>
        (s.pids 1 * T + i_t) * D + (s.pids 0 * BD + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s i_t T D BD a = outOffset s i_t T D BD b := by
      simpa [outOffset, dIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  simp [exec, fused_recurrent_hgrn_bwd_dx_step_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        outOffset, dIndex, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
  by_cases hi : s.pids 0 * BD + i.val < D
  · simp [active, bwdDxStepValue, outOffset, dIndex, hi, NumericDType.add]
  · simp [active, outOffset, dIndex, hi]

theorem fused_recurrent_hgrn_bwd_dx_step_store_slice_compute_correct
    (DHPrev DO DX : RegionName) (i_t T D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_hgrn_bwd_dx_step_store_slice DHPrev DO DX
        i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (DX, outOffset s i_t T D BD i)))
      (expected := fun i => bwdDxStepValue s DHPrev DO i_t T D BD i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_hgrn_bwd_dx_step_store_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fused_recurrent_hgrn_bwd_dx_step_store_slice_correct DHPrev DO DX
    i_t T D BD s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Backward one-step `dg` formula:
`b_dh = b_dh_prev + b_do`, `b_dg = b_dh * b_o`, then masked store to `DG`. -/
def fused_recurrent_hgrn_bwd_dg_step_store_slice
    (DHPrev DO BO DG : RegionName) (i_t T D BD : Nat) :
    ComputeKernel := triton {
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  dh_prev = tl.load(DHPrev + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0)
  b_do = tl.load(DO + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_o = tl.load(BO + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_dh = dh_prev + b_do
  b_dg = b_dh * b_o
  tl.store(DG + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    (b_dg).to(DG.dtype.element_ty), mask=mask)
}

noncomputable def bwdDgStepValue
    (s : BlockState) (DHPrev DO BO : RegionName) (i_t T D BD : Nat)
    (i : Fin BD) : ℝ :=
  (s.readMem DHPrev (outOffset s i_t T D BD i) +
      s.readMem DO (outOffset s i_t T D BD i)) *
    s.readMem BO (outOffset s i_t T D BD i)

theorem fused_recurrent_hgrn_bwd_dg_step_store_slice_correct
    (DHPrev DO BO DG : RegionName) (i_t T D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i)) :
    ∀ i : Fin BD,
      let outAddr := outOffset s i_t T D BD i
      (exec (fused_recurrent_hgrn_bwd_dg_step_store_slice DHPrev DO BO DG
          i_t T D BD) s).map (·.readMem DG outAddr)
        = some (if active s D BD i then
            bwdDgStepValue s DHPrev DO BO i_t T D BD i
          else s.readMem DG outAddr) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BD] =>
        (s.pids 1 * T + i_t) * D + (s.pids 0 * BD + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s i_t T D BD a = outOffset s i_t T D BD b := by
      simpa [outOffset, dIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  simp [exec, fused_recurrent_hgrn_bwd_dg_step_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        outOffset, dIndex, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
  by_cases hi : s.pids 0 * BD + i.val < D
  · simp [active, bwdDgStepValue, outOffset, dIndex, hi, NumericDType.add,
      NumericDType.mul]
  · simp [active, outOffset, dIndex, hi]

theorem fused_recurrent_hgrn_bwd_dg_step_store_slice_compute_correct
    (DHPrev DO BO DG : RegionName) (i_t T D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_hgrn_bwd_dg_step_store_slice DHPrev DO BO DG
        i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (DG, outOffset s i_t T D BD i)))
      (expected := fun i => bwdDgStepValue s DHPrev DO BO i_t T D BD i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_hgrn_bwd_dg_step_store_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fused_recurrent_hgrn_bwd_dg_step_store_slice_correct DHPrev DO BO
    DG i_t T D BD s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ### ════════ ★ MAIN THEOREM ★ ════════

**SCOPE — this is a claim about three hand-cut single-step slices, not about the
launched kernels.** Clauses 2–4 are `Realizes` facts about
`fused_recurrent_hgrn_forward_step_store_slice`,
`fused_recurrent_hgrn_bwd_dx_step_store_slice` and
`fused_recurrent_hgrn_bwd_dg_step_store_slice`; the launched forward surface
appears only in clause 1, which says nothing more than "it lowers to the
algorithm layer", and the launched backward surface does not appear at all. The
`STORE_FINAL_STATE` writeback has **no** correctness face here (see the module
docstring).

Parameterized over the symbolic time/feature/tile sizes `T D BD`, the step index
`i_t`, and both flags `USE_INITIAL_STATE STORE_FINAL_STATE`. The forward face is
realized against the closed form `hgrnStateClosed` over the *input* regions
`x, g, h0` (never a read-back of the kernel's own output):

1. the full HGRN forward surface lowers to the algorithm layer;
2. one forward **output** body realizes `hgrnStateClosed(i_t + 1)` — the unrolled
   recurrence `b_h = g·b_h + x` — given the *assumed* carry invariant
   `BHPrev = hgrnStateClosed(i_t)`;
3. one backward `dx` body realizes the genuine `bwdDxStepValue` (`dh_prev + do`);
4. one backward `dg` body realizes the genuine `bwdDgStepValue` (`(dh_prev+do)·o`).

Honest structural side condition: `0 < BD` (contiguous lanes, giving offset
injectivity for every face). The flags flow through verbatim; clauses 2–4 hold
for every flag setting.

**The carry invariant `hPrev` is an assumption, not a conclusion.** Nothing in
this file propagates it: clause 2 writes `O` at the time-indexed
`outOffset s i_t`, whereas `hPrev` constrains `BHPrev` at the time-free
`bhOffset` — a different region at a different offset, with no bridging lemma.
Likewise nothing proves the base case `hgrnStateClosed(0) = seed`. The same holds
for the backward `b_dh` carry, which is presented as the fiction region `DHPrev`
and never chained. -/
specification fused_recurrent_hgrn_output_summary_general
    (X G O H0 Ht DX DG DO BHPrev DHPrev BO : RegionName)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (i_t T D BD : Nat) (s : BlockState) (hBD : 0 < BD)
    (hPrev : ∀ i : Fin BD,
      s.readMem BHPrev (bhOffset s D BD i)
        = hgrnStateClosed s X G H0 USE_INITIAL_STATE T D BD i_t i) :
    -- (1) the full forward surface lowers to the algorithm layer
    (∃ alg, (fused_recurrent_hgrn_fwd_surface X G O H0 Ht T D BD
      USE_INITIAL_STATE STORE_FINAL_STATE).toAlgorithm? = Except.ok alg) ∧
    -- (2) the forward output body realizes the genuine `hgrnStateClosed(i_t+1)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_hgrn_forward_step_store_slice BHPrev X G O
        i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (O, outOffset s i_t T D BD i)))
      (expected := fun i =>
        hgrnStateClosed s X G H0 USE_INITIAL_STATE T D BD (i_t + 1) i)) ∧
    -- (3) the backward `dx` body realizes the genuine `dh_prev + do`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_hgrn_bwd_dx_step_store_slice DHPrev DO DX
        i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (DX, outOffset s i_t T D BD i)))
      (expected := fun i => bwdDxStepValue s DHPrev DO i_t T D BD i)) ∧
    -- (4) the backward `dg` body realizes the genuine `(dh_prev+do)·o`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_hgrn_bwd_dg_step_store_slice DHPrev DO BO DG
        i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (DG, outOffset s i_t T D BD i)))
      (expected := fun i => bwdDgStepValue s DHPrev DO BO i_t T D BD i)) := by
  have hOutInj := fused_recurrent_hgrn_out_offset_injective_general s i_t T D BD hBD
  refine ⟨fused_recurrent_hgrn_fwd_surface_toAlgorithm_supported _ _ _ _ _ _ _ _ _ _,
      ?_, ?_, ?_⟩
  · exact fused_recurrent_hgrn_forward_step_closed_form BHPrev X G H0 O
      USE_INITIAL_STATE i_t T D BD s hOutInj hPrev
  · exact fused_recurrent_hgrn_bwd_dx_step_store_slice_compute_correct
      DHPrev DO DX i_t T D BD s hOutInj
  · exact fused_recurrent_hgrn_bwd_dg_step_store_slice_compute_correct
      DHPrev DO BO DG i_t T D BD s hOutInj

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn

