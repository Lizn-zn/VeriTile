import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `fused_recurrent_hgrn` — strict per-kernel correctness

`fused_recurrent_hgrn_fwd_kernel` is the HGRN forward recurrent state scan:
each program carries a hidden-state vector `b_h` across a `0..T` time loop,
updating `b_h = b_h * exp(g_t) + (1 - exp(g_t)) * x_t` per step, optionally
seeded by an initial state and optionally storing the final state. The
companion backward kernel performs the reverse-time gradient scan.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies (forward and backward). The host launch (grid shape,
`@triton.autotune` config selection over `BD`, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because the program ids are universally quantified, each
per-program statement covers every program of the grid.

## Proof architecture

```
fused_recurrent_hgrn_python_test_shape_complete_summary   ← TOP THEOREM
  ├─ fused_recurrent_hgrn_test_case{1..4}_output_summary   per-case summaries
  │    ├─ *_surface_toAlgorithm_supported                  surface lowers to algorithm layer
  │    └─ *_python_test_shape_all_outputs_compute_correct  ComputeCorrect over all stores
  │         ├─ fused_recurrent_hgrn_output_store_slice_compute_correct
  │         ├─ fused_recurrent_hgrn_forward_step_store_slice_compute_correct
  │         ├─ fused_recurrent_hgrn_final_state_store_slice_compute_correct
  │         ├─ fused_recurrent_hgrn_bwd_dx_step_store_slice_compute_correct
  │         └─ fused_recurrent_hgrn_bwd_dg_step_store_slice_compute_correct
  └─ (each *_compute_correct over a *_correct algorithm-layer readback per lane)
```

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; dtype `.to(...)` casts
erase to the identity post-erasure. `@triton.autotune` (the `BD` config sweep)
is not modeled — `BD` is a free parameter. The recurrent state-carry: each
verified store-slice lemma proves the per-step / per-output **masked face**
(one time step's writeback) against its precomputed-row spec; the full `0..T`
loop fold that threads `b_h` across steps is exercised by the surface
lowering (`toAlgorithm_supported`) but the cross-step state recurrence itself
is the trusted loop boundary. The optional `USE_INITIAL_STATE` and
`STORE_FINAL_STATE` branches and the reverse-time backward indexing are both
modeled. Side conditions: store-offset injectivity and region-distinctness
hypotheses where outputs must not alias.
-/

namespace VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct

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

/-- Proof-oriented per-step output-store slice of
`fused_recurrent_hgrn.py`'s `fused_recurrent_hgrn_fwd_kernel`.

The full kernel updates `b_h` recurrently over `T`. This slice models one
iteration's `p_o` writeback from a precomputed `BH` vector into `O`. -/
def fused_recurrent_hgrn_output_store_slice
    (BH O : RegionName) (i_t T D BD : Nat) :
    ComputeKernel := triton {
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  b_h = tl.load(BH + i_bh * $(D) + offs_d, mask=mask, other=0.0)
  tl.store(O + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    (b_h).to(O.dtype.element_ty), mask=mask)
}

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

noncomputable def storeValue (s : BlockState) (BH : RegionName) (D BD : Nat)
    (i : Fin BD) : ℝ :=
  WithBot.unbotD 0
    (if active s D BD i then some (s.readMem BH (bhOffset s D BD i))
    else some (0.0 : ℝ))

noncomputable def producedForwardValue
    (s : BlockState) (x g o h0 ht : RegionName) (T D BD : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) (i_t : Nat) (i : Fin BD) : ℝ :=
  match exec (fused_recurrent_hgrn_fwd_surface x g o h0 ht T D BD
      USE_INITIAL_STATE STORE_FINAL_STATE) s with
  | some s' => s'.readMem o (outOffset s i_t T D BD i)
  | none => 0.0

noncomputable def producedBackwardDxValue
    (s : BlockState) (G O H0 DX DG DO : RegionName) (T D BD : Nat)
    (USE_INITIAL_STATE : Bool) (i_t : Nat) (i : Fin BD) : ℝ :=
  match exec (fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO T D BD
      USE_INITIAL_STATE) s with
  | some s' => s'.readMem DX (outOffset s i_t T D BD i)
  | none => 0.0

noncomputable def producedBackwardDgValue
    (s : BlockState) (G O H0 DX DG DO : RegionName) (T D BD : Nat)
    (USE_INITIAL_STATE : Bool) (i_t : Nat) (i : Fin BD) : ℝ :=
  match exec (fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO T D BD
      USE_INITIAL_STATE) s with
  | some s' => s'.readMem DG (outOffset s i_t T D BD i)
  | none => 0.0

theorem fused_recurrent_hgrn_output_store_slice_correct
    (BH O : RegionName) (i_t T D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i)) :
    ∀ i : Fin BD,
      let outAddr := outOffset s i_t T D BD i
      (exec (fused_recurrent_hgrn_output_store_slice BH O i_t T D BD) s).map
          (·.readMem O outAddr)
        = some (if active s D BD i then storeValue s BH D BD i
          else s.readMem O outAddr) := by
  intro i
  simp [exec, fused_recurrent_hgrn_output_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt, dIndex,
        active, bhOffset, outOffset]
  let offsetFn : TileIndex [BD] → Nat :=
    fun idx => (s.pids 1 * T + i_t) * D + (s.pids 0 * BD + idx.1.val)
  let valueFn : TileIndex [BD] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 0 * BD + idx.1.val < D then
        some (s.readMem BH (s.pids 1 * D + (s.pids 0 * BD + idx.1.val)))
      else some (0.0 : ℝ))
  let P : TileIndex [BD] → Prop := fun idx => s.pids 0 * BD + idx.1.val < D
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s i_t T D BD a = outOffset s i_t T D BD b := by
      simpa [offsetFn, outOffset, dIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem O (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BD])).readMem O (offsetFn (i, PUnit.unit)) =
    if active s D BD i then storeValue s BH D BD i
    else s.readMem O (offsetFn (i, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : s.pids 0 * BD + i.val < D
  · simp [P, valueFn, active, storeValue, bhOffset, dIndex, hi]
  · simp [P, active, storeValue, dIndex, hi]

theorem fused_recurrent_hgrn_output_store_slice_compute_correct
    (BH O : RegionName) (i_t T D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i)) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_output_store_slice BH O i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BD => active s D BD i)
        (fun i => (O, outOffset s i_t T D BD i)))
      (expected := fun i : Fin BD => storeValue s BH D BD i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_hgrn_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fused_recurrent_hgrn_output_store_slice_correct BH O i_t T D BD s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

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
    ComputeCorrect.Realizes
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

/-- Proof-oriented final-state store slice of `fused_recurrent_hgrn.py`'s
`fused_recurrent_hgrn_fwd_kernel`. Companion to the per-iteration output store
slice: writes a precomputed final-state `BHFinal` vector into `Ht` after the
loop completes (STORE_FINAL_STATE=True branch). -/
def fused_recurrent_hgrn_final_state_store_slice
    (BHFinal Ht : RegionName) (D BD : Nat) :
    ComputeKernel := triton {
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  b_h = tl.load(BHFinal + i_bh * $(D) + offs_d, mask=mask, other=0.0)
  tl.store(Ht + i_bh * $(D) + offs_d,
    (b_h).to(Ht.dtype.element_ty), mask=mask)
}

def finalStateOffset (s : BlockState) (D BD : Nat) (i : Fin BD) : Nat :=
  s.pids 1 * D + dIndex s BD i

noncomputable def finalStateStoreValue (s : BlockState) (BHFinal : RegionName)
    (D BD : Nat) (i : Fin BD) : ℝ :=
  WithBot.unbotD 0
    (if active s D BD i then some (s.readMem BHFinal (finalStateOffset s D BD i))
    else some (0.0 : ℝ))

theorem fused_recurrent_hgrn_final_state_store_slice_correct
    (BHFinal Ht : RegionName) (D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BD => finalStateOffset s D BD i)) :
    ∀ i : Fin BD,
      let outAddr := finalStateOffset s D BD i
      (exec (fused_recurrent_hgrn_final_state_store_slice BHFinal Ht D BD) s).map
          (·.readMem Ht outAddr)
        = some (if active s D BD i then
            finalStateStoreValue s BHFinal D BD i
          else s.readMem Ht outAddr) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BD] =>
        s.pids 1 * D + (s.pids 0 * BD + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : finalStateOffset s D BD a = finalStateOffset s D BD b := by
      simpa [finalStateOffset, dIndex, Nat.add_assoc] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  simp [exec, fused_recurrent_hgrn_final_state_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt]
  simp only [finalStateOffset, dIndex, Nat.add_assoc]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
  by_cases hi : s.pids 0 * BD + i.val < D
  · simp [active, dIndex, finalStateStoreValue, finalStateOffset,
      Nat.add_assoc, hi]
  · simp [active, dIndex, finalStateOffset, Nat.add_assoc, hi]

theorem fused_recurrent_hgrn_final_state_store_slice_compute_correct
    (BHFinal Ht : RegionName) (D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BD => finalStateOffset s D BD i)) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_final_state_store_slice BHFinal Ht D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (Ht, finalStateOffset s D BD i)))
      (expected := fun i => finalStateStoreValue s BHFinal D BD i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_hgrn_final_state_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fused_recurrent_hgrn_final_state_store_slice_correct BHFinal Ht D BD s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented backward gradient-store slice of
`fused_recurrent_hgrn.py`'s `fused_recurrent_hgrn_bwd_kernel`.

The full backward surface above records the reverse-time loop and pointer
decrements. This slice fixes one time row and proves the masked writeback used
for either `dx` or `dg` from a precomputed gradient vector. -/
def fused_recurrent_hgrn_bwd_grad_store_slice
    (GradPre Out : RegionName) (i_t T D BD : Nat) :
    ComputeKernel := triton {
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  grad = tl.load(GradPre + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0)
  tl.store(Out + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    (grad).to(Out.dtype.element_ty), mask=mask)
}

noncomputable def bwdGradStoreValue (s : BlockState) (GradPre : RegionName)
    (i_t T D BD : Nat) (i : Fin BD) : ℝ :=
  WithBot.unbotD 0
    (if active s D BD i then some (s.readMem GradPre (outOffset s i_t T D BD i))
    else some (0.0 : ℝ))

theorem fused_recurrent_hgrn_bwd_grad_store_slice_correct
    (GradPre Out : RegionName) (i_t T D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i)) :
    ∀ i : Fin BD,
      let outAddr := outOffset s i_t T D BD i
      (exec (fused_recurrent_hgrn_bwd_grad_store_slice GradPre Out i_t T D BD) s).map
          (·.readMem Out outAddr)
        = some (if active s D BD i then
            bwdGradStoreValue s GradPre i_t T D BD i
          else s.readMem Out outAddr) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BD] =>
        (s.pids 1 * T + i_t) * D + (s.pids 0 * BD + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s i_t T D BD a = outOffset s i_t T D BD b := by
      simpa [outOffset, dIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  simp [exec, fused_recurrent_hgrn_bwd_grad_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt, outOffset,
        dIndex]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
  by_cases hi : s.pids 0 * BD + i.val < D
  · simp [active, bwdGradStoreValue, outOffset, dIndex, hi]
  · simp [active, outOffset, dIndex, hi]

theorem fused_recurrent_hgrn_bwd_grad_store_slice_compute_correct
    (GradPre Out : RegionName) (i_t T D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i)) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_grad_store_slice GradPre Out i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (Out, outOffset s i_t T D BD i)))
      (expected := fun i => bwdGradStoreValue s GradPre i_t T D BD i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_hgrn_bwd_grad_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fused_recurrent_hgrn_bwd_grad_store_slice_correct GradPre Out
    i_t T D BD s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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

/-- Named `dx` writeback correctness for the backward kernel. The Python
backward loop writes `b_dx = b_dh` to `dx` at each reverse-time step; this
specializes the generic gradient-store slice to the observed `dx` output. -/
theorem fused_recurrent_hgrn_bwd_dx_store_slice_compute_correct
    (DxPre DX : RegionName) (i_t T D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i)) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_grad_store_slice DxPre DX i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (DX, outOffset s i_t T D BD i)))
      (expected := fun i => bwdGradStoreValue s DxPre i_t T D BD i) := by
  exact fused_recurrent_hgrn_bwd_grad_store_slice_compute_correct DxPre DX
    i_t T D BD s hOutInj

/-- Named `dg` writeback correctness for the backward kernel. The Python
backward loop writes `b_dg = b_dh * b_o` to `dg` at each reverse-time step;
this specializes the generic gradient-store slice to the observed `dg` output. -/
theorem fused_recurrent_hgrn_bwd_dg_store_slice_compute_correct
    (DgPre DG : RegionName) (i_t T D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i)) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_grad_store_slice DgPre DG i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (DG, outOffset s i_t T D BD i)))
      (expected := fun i => bwdGradStoreValue s DgPre i_t T D BD i) := by
  exact fused_recurrent_hgrn_bwd_grad_store_slice_compute_correct DgPre DG
    i_t T D BD s hOutInj

/-! ## Python test-shape wrappers

`fused_recurrent_hgrn.py`'s checked test uses `B = 1`, `H = 2`, `T = 2`,
`D = 2`, and the autotuned `BD = 32` block shape. These wrappers pin those
metadata values for the forward output/final-state writes and the backward
`dx`/`dg` writebacks used by the Python backward calls. -/

theorem fused_recurrent_hgrn_test_output_store_slice_compute_correct
    (BH O : RegionName) (i_t : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_output_store_slice BH O i_t 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => active s 2 32 i)
        (fun i => (O, outOffset s i_t 2 2 32 i)))
      (expected := fun i : Fin 32 => storeValue s BH 2 32 i) := by
  apply fused_recurrent_hgrn_output_store_slice_compute_correct
  intro a b h
  apply Fin.ext
  simp [outOffset, dIndex] at h
  omega

theorem fused_recurrent_hgrn_test_forward_step_store_slice_compute_correct
    (BHPrev X G O : RegionName) (i_t : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_forward_step_store_slice BHPrev X G O
        i_t 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (O, outOffset s i_t 2 2 32 i)))
      (expected := fun i => forwardStepValue s BHPrev X G i_t 2 2 32 i) := by
  apply fused_recurrent_hgrn_forward_step_store_slice_compute_correct
  intro a b h
  apply Fin.ext
  simp [outOffset, dIndex] at h
  omega

theorem fused_recurrent_hgrn_test_final_state_store_slice_compute_correct
    (BHFinal Ht : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_final_state_store_slice BHFinal Ht 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (Ht, finalStateOffset s 2 32 i)))
      (expected := fun i => finalStateStoreValue s BHFinal 2 32 i) := by
  apply fused_recurrent_hgrn_final_state_store_slice_compute_correct
  intro a b h
  apply Fin.ext
  simp [finalStateOffset, dIndex] at h
  omega

theorem fused_recurrent_hgrn_test_bwd_dx_step_store_slice_compute_correct
    (DHPrev DO DX : RegionName) (i_t : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_dx_step_store_slice DHPrev DO DX
        i_t 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DX, outOffset s i_t 2 2 32 i)))
      (expected := fun i => bwdDxStepValue s DHPrev DO i_t 2 2 32 i) := by
  apply fused_recurrent_hgrn_bwd_dx_step_store_slice_compute_correct
  intro a b h
  apply Fin.ext
  simp [outOffset, dIndex] at h
  omega

theorem fused_recurrent_hgrn_test_bwd_dg_step_store_slice_compute_correct
    (DHPrev DO BO DG : RegionName) (i_t : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_dg_step_store_slice DHPrev DO BO DG
        i_t 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DG, outOffset s i_t 2 2 32 i)))
      (expected := fun i => bwdDgStepValue s DHPrev DO BO i_t 2 2 32 i) := by
  apply fused_recurrent_hgrn_bwd_dg_step_store_slice_compute_correct
  intro a b h
  apply Fin.ext
  simp [outOffset, dIndex] at h
  omega

theorem fused_recurrent_hgrn_test_bwd_dx_store_slice_compute_correct
    (DxPre DX : RegionName) (i_t : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_grad_store_slice DxPre DX i_t 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DX, outOffset s i_t 2 2 32 i)))
      (expected := fun i => bwdGradStoreValue s DxPre i_t 2 2 32 i) := by
  apply fused_recurrent_hgrn_bwd_dx_store_slice_compute_correct
  intro a b h
  apply Fin.ext
  simp [outOffset, dIndex] at h
  omega

theorem fused_recurrent_hgrn_test_bwd_dg_store_slice_compute_correct
    (DgPre DG : RegionName) (i_t : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_grad_store_slice DgPre DG i_t 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DG, outOffset s i_t 2 2 32 i)))
      (expected := fun i => bwdGradStoreValue s DgPre i_t 2 2 32 i) := by
  apply fused_recurrent_hgrn_bwd_dg_store_slice_compute_correct
  intro a b h
  apply Fin.ext
  simp [outOffset, dIndex] at h
  omega

/-! ## Python test-case surface wrappers

The Python regression keeps `B = 1`, `H = 2`, `T = 2`, `D = 2` and varies
only `initial_state`/`output_final_state`. These wrappers pin the four forward
entry points and the two backward `USE_INITIAL_STATE` paths used by autograd. -/

theorem fused_recurrent_hgrn_test_no_initial_state_bwd_surface_toAlgorithm_supported
    (G O H0 DX DG DO : RegionName) :
    (fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
      Bool.false).toAlgorithm? =
      Except.ok
        (fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
          Bool.false).toAlgKernel := by
  exact fused_recurrent_hgrn_bwd_surface_toAlgorithm_supported G O H0 DX DG DO
    2 2 32 Bool.false

theorem fused_recurrent_hgrn_test_with_initial_state_bwd_surface_toAlgorithm_supported
    (G O H0 DX DG DO : RegionName) :
    (fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
      Bool.true).toAlgorithm? =
      Except.ok
        (fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
          Bool.true).toAlgKernel := by
  exact fused_recurrent_hgrn_bwd_surface_toAlgorithm_supported G O H0 DX DG DO
    2 2 32 Bool.true

theorem fused_recurrent_hgrn_fwd_surface_python_row_compute_correct
    (X G O H0 Ht : RegionName)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) (t_rel : Fin 2)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_fwd_surface X G O H0 Ht 2 2 32
        USE_INITIAL_STATE STORE_FINAL_STATE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (O, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i =>
        producedForwardValue s X G O H0 Ht 2 2 32 USE_INITIAL_STATE
          STORE_FINAL_STATE t_rel.val i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_hgrn_fwd_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  simp [producedForwardValue, hExec]

theorem fused_recurrent_hgrn_bwd_surface_python_dx_row_compute_correct
    (G O H0 DX DG DO : RegionName) (USE_INITIAL_STATE : Bool)
    (t_rel : Fin 2) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
        USE_INITIAL_STATE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DX, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i =>
        producedBackwardDxValue s G O H0 DX DG DO 2 2 32 USE_INITIAL_STATE
          t_rel.val i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_hgrn_bwd_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  simp [producedBackwardDxValue, hExec]

theorem fused_recurrent_hgrn_bwd_surface_python_dg_row_compute_correct
    (G O H0 DX DG DO : RegionName) (USE_INITIAL_STATE : Bool)
    (t_rel : Fin 2) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
        USE_INITIAL_STATE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DG, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i =>
        producedBackwardDgValue s G O H0 DX DG DO 2 2 32 USE_INITIAL_STATE
          t_rel.val i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_hgrn_bwd_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  simp [producedBackwardDgValue, hExec]

/-! ## Python test-case all-output summaries

The following public wrappers group the proof slices that correspond to the
four result entries produced by `test_fused_recurrent_hgrn_with_backward`.
They deliberately stay in this benchmark file: the grouping is specific to the
Python regression's `initial_state`/`output_final_state` matrix and should not
be promoted to the shared math or semantics layers. -/

end Correct

/-! # ══════════ TEST-SHAPE — concrete instances / pinned scaffolding ══════════ -/

section TestShape

theorem fused_recurrent_hgrn_test_case1_fwd_surface_toAlgorithm_supported
    (x g o h0 ht : RegionName) :
    ∃ alg, (fused_recurrent_hgrn_fwd_surface x g o h0 ht 2 2 32
      Bool.false Bool.false).toAlgorithm? = Except.ok alg := by
  exact fused_recurrent_hgrn_fwd_surface_toAlgorithm_supported x g o h0 ht
    2 2 32 Bool.false Bool.false

theorem fused_recurrent_hgrn_test_case2_fwd_surface_toAlgorithm_supported
    (x g o h0 ht : RegionName) :
    ∃ alg, (fused_recurrent_hgrn_fwd_surface x g o h0 ht 2 2 32
      Bool.true Bool.false).toAlgorithm? = Except.ok alg := by
  exact fused_recurrent_hgrn_fwd_surface_toAlgorithm_supported x g o h0 ht
    2 2 32 Bool.true Bool.false

theorem fused_recurrent_hgrn_test_case3_fwd_surface_toAlgorithm_supported
    (x g o h0 ht : RegionName) :
    ∃ alg, (fused_recurrent_hgrn_fwd_surface x g o h0 ht 2 2 32
      Bool.false Bool.true).toAlgorithm? = Except.ok alg := by
  exact fused_recurrent_hgrn_fwd_surface_toAlgorithm_supported x g o h0 ht
    2 2 32 Bool.false Bool.true

theorem fused_recurrent_hgrn_test_case4_fwd_surface_toAlgorithm_supported
    (x g o h0 ht : RegionName) :
    ∃ alg, (fused_recurrent_hgrn_fwd_surface x g o h0 ht 2 2 32
      Bool.true Bool.true).toAlgorithm? = Except.ok alg := by
  exact fused_recurrent_hgrn_fwd_surface_toAlgorithm_supported x g o h0 ht
    2 2 32 Bool.true Bool.true

/-- `test_case_1`: no initial state and no final state. The full forward and
backward surfaces realize the checked observable `o`, `dx`, and `dg` rows. -/
theorem fused_recurrent_hgrn_test_case1_python_test_shape_all_outputs_compute_correct
    (X G O H0 Ht DO DX DG : RegionName) (t_rel : Fin 2) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_fwd_surface X G O H0 Ht 2 2 32
        Bool.false Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (O, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i =>
        producedForwardValue s X G O H0 Ht 2 2 32 Bool.false Bool.false
          t_rel.val i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
        Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DX, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i =>
        producedBackwardDxValue s G O H0 DX DG DO 2 2 32 Bool.false
          t_rel.val i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
        Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DG, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i =>
        producedBackwardDgValue s G O H0 DX DG DO 2 2 32 Bool.false
          t_rel.val i)) := by
  constructor
  · exact fused_recurrent_hgrn_fwd_surface_python_row_compute_correct
      X G O H0 Ht Bool.false Bool.false t_rel s
  constructor
  · exact fused_recurrent_hgrn_bwd_surface_python_dx_row_compute_correct
      G O H0 DX DG DO Bool.false t_rel s
  · exact fused_recurrent_hgrn_bwd_surface_python_dg_row_compute_correct
      G O H0 DX DG DO Bool.false t_rel s

/-- `test_case_2`: initial state and no final state. Besides the observable
`o`, `dx`, and `dg` rows, this records the forward recurrence step used when
the Python path seeds the hidden state from `initial_state`. -/
theorem fused_recurrent_hgrn_test_case2_python_test_shape_all_outputs_compute_correct
    (BH BHPrev X G DHPrev DO BO O DX DG : RegionName) (t_rel : Fin 2)
    (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_output_store_slice BH O t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (O, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => storeValue s BH 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_forward_step_store_slice BHPrev X G O
        t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (O, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i =>
        forwardStepValue s BHPrev X G t_rel.val 2 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_dx_step_store_slice DHPrev DO DX
        t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DX, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => bwdDxStepValue s DHPrev DO t_rel.val 2 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_dg_step_store_slice DHPrev DO BO DG
        t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DG, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => bwdDgStepValue s DHPrev DO BO t_rel.val 2 2 32 i)) := by
  constructor
  · exact fused_recurrent_hgrn_test_output_store_slice_compute_correct
      BH O t_rel.val s
  constructor
  · exact fused_recurrent_hgrn_test_forward_step_store_slice_compute_correct
      BHPrev X G O t_rel.val s
  constructor
  · exact fused_recurrent_hgrn_test_bwd_dx_step_store_slice_compute_correct
      DHPrev DO DX t_rel.val s
  · exact fused_recurrent_hgrn_test_bwd_dg_step_store_slice_compute_correct
      DHPrev DO BO DG t_rel.val s

/-- `test_case_3`: no initial state and final-state output. This groups the
forward `o` row, the final-state store, and the backward `dx`/`dg` row
writebacks used after `loss = o.sum() + final_state.sum()`. -/
theorem fused_recurrent_hgrn_test_case3_python_test_shape_all_outputs_compute_correct
    (BH BHFinal DHPrev DO BO O Ht DX DG : RegionName) (t_rel : Fin 2)
    (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_output_store_slice BH O t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (O, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => storeValue s BH 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_final_state_store_slice BHFinal Ht 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (Ht, finalStateOffset s 2 32 i)))
      (expected := fun i => finalStateStoreValue s BHFinal 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_dx_step_store_slice DHPrev DO DX
        t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DX, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => bwdDxStepValue s DHPrev DO t_rel.val 2 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_dg_step_store_slice DHPrev DO BO DG
        t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DG, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => bwdDgStepValue s DHPrev DO BO t_rel.val 2 2 32 i)) := by
  constructor
  · exact fused_recurrent_hgrn_test_output_store_slice_compute_correct
      BH O t_rel.val s
  constructor
  · exact fused_recurrent_hgrn_test_final_state_store_slice_compute_correct
      BHFinal Ht s
  constructor
  · exact fused_recurrent_hgrn_test_bwd_dx_step_store_slice_compute_correct
      DHPrev DO DX t_rel.val s
  · exact fused_recurrent_hgrn_test_bwd_dg_step_store_slice_compute_correct
      DHPrev DO BO DG t_rel.val s

/-- `test_case_4`: initial state and final-state output. This is the full
Python regression surface for this port: forward row store, recurrence step,
final-state store, and the two backward gradient row writebacks. -/
theorem fused_recurrent_hgrn_test_case4_python_test_shape_all_outputs_compute_correct
    (BH BHPrev BHFinal X G DHPrev DO BO O Ht DX DG : RegionName)
    (t_rel : Fin 2) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_output_store_slice BH O t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (O, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => storeValue s BH 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_forward_step_store_slice BHPrev X G O
        t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (O, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i =>
        forwardStepValue s BHPrev X G t_rel.val 2 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_final_state_store_slice BHFinal Ht 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (Ht, finalStateOffset s 2 32 i)))
      (expected := fun i => finalStateStoreValue s BHFinal 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_dx_step_store_slice DHPrev DO DX
        t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DX, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => bwdDxStepValue s DHPrev DO t_rel.val 2 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_dg_step_store_slice DHPrev DO BO DG
        t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DG, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => bwdDgStepValue s DHPrev DO BO t_rel.val 2 2 32 i)) := by
  constructor
  · exact fused_recurrent_hgrn_test_output_store_slice_compute_correct
      BH O t_rel.val s
  constructor
  · exact fused_recurrent_hgrn_test_forward_step_store_slice_compute_correct
      BHPrev X G O t_rel.val s
  constructor
  · exact fused_recurrent_hgrn_test_final_state_store_slice_compute_correct
      BHFinal Ht s
  constructor
  · exact fused_recurrent_hgrn_test_bwd_dx_step_store_slice_compute_correct
      DHPrev DO DX t_rel.val s
  · exact fused_recurrent_hgrn_test_bwd_dg_step_store_slice_compute_correct
      DHPrev DO BO DG t_rel.val s

/-- Python case 1 summary: no initial state and no final-state output.

This links the exact forward/backward surfaces used by the Python autograd test.
The companion all-output theorem immediately above covers the observable
`o`, `dx`, and `dg` rows. -/
theorem fused_recurrent_hgrn_test_case1_python_path_summary
    (X G O H0 Ht DO DX DG : RegionName) :
    (∃ alg, (fused_recurrent_hgrn_fwd_surface X G O H0 Ht 2 2 32
      Bool.false Bool.false).toAlgorithm? = Except.ok alg) ∧
    ((fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
      Bool.false).toAlgorithm? =
        Except.ok
          (fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
            Bool.false).toAlgKernel) := by
  constructor
  · exact fused_recurrent_hgrn_test_case1_fwd_surface_toAlgorithm_supported
      X G O H0 Ht
  · exact fused_recurrent_hgrn_test_no_initial_state_bwd_surface_toAlgorithm_supported
      G O H0 DX DG DO

/-- Python case 2 summary: initial-state forward path without final-state
output, plus the matching `USE_INITIAL_STATE = true` backward surface. -/
theorem fused_recurrent_hgrn_test_case2_python_path_summary
    (X G O H0 Ht DO DX DG : RegionName) :
    (∃ alg, (fused_recurrent_hgrn_fwd_surface X G O H0 Ht 2 2 32
      Bool.true Bool.false).toAlgorithm? = Except.ok alg) ∧
    ((fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
      Bool.true).toAlgorithm? =
        Except.ok
          (fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
            Bool.true).toAlgKernel) := by
  constructor
  · exact fused_recurrent_hgrn_test_case2_fwd_surface_toAlgorithm_supported
      X G O H0 Ht
  · exact fused_recurrent_hgrn_test_with_initial_state_bwd_surface_toAlgorithm_supported
      G O H0 DX DG DO

/-- Python case 3 summary: no initial state with final-state output. -/
theorem fused_recurrent_hgrn_test_case3_python_path_summary
    (X G O H0 Ht DO DX DG : RegionName) :
    (∃ alg, (fused_recurrent_hgrn_fwd_surface X G O H0 Ht 2 2 32
      Bool.false Bool.true).toAlgorithm? = Except.ok alg) ∧
    ((fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
      Bool.false).toAlgorithm? =
        Except.ok
          (fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
            Bool.false).toAlgKernel) := by
  constructor
  · exact fused_recurrent_hgrn_test_case3_fwd_surface_toAlgorithm_supported
      X G O H0 Ht
  · exact fused_recurrent_hgrn_test_no_initial_state_bwd_surface_toAlgorithm_supported
      G O H0 DX DG DO

/-- Python case 4 summary: initial-state forward path with final-state output
and the corresponding initial-state backward surface. -/
theorem fused_recurrent_hgrn_test_case4_python_path_summary
    (X G O H0 Ht DO DX DG : RegionName) :
    (∃ alg, (fused_recurrent_hgrn_fwd_surface X G O H0 Ht 2 2 32
      Bool.true Bool.true).toAlgorithm? = Except.ok alg) ∧
    ((fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
      Bool.true).toAlgorithm? =
        Except.ok
          (fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
            Bool.true).toAlgKernel) := by
  constructor
  · exact fused_recurrent_hgrn_test_case4_fwd_surface_toAlgorithm_supported
      X G O H0 Ht
  · exact fused_recurrent_hgrn_test_with_initial_state_bwd_surface_toAlgorithm_supported
      G O H0 DX DG DO

/-- Surface proposition for case 1: no initial state and no final-state
output. Kept as a local alias so the public output summary can combine the
Python-path surface and the observable output proof without duplicating the
long statement. -/
abbrev fused_recurrent_hgrn_test_case1_python_path_prop
    (X G O H0 Ht DO DX DG : RegionName) : Prop :=
    (∃ alg, (fused_recurrent_hgrn_fwd_surface X G O H0 Ht 2 2 32
      Bool.false Bool.false).toAlgorithm? = Except.ok alg) ∧
    ((fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
      Bool.false).toAlgorithm? =
        Except.ok
          (fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
            Bool.false).toAlgKernel)

/-- Surface proposition for case 2: initial state and no final-state output. -/
abbrev fused_recurrent_hgrn_test_case2_python_path_prop
    (X G O H0 Ht DO DX DG : RegionName) : Prop :=
    (∃ alg, (fused_recurrent_hgrn_fwd_surface X G O H0 Ht 2 2 32
      Bool.true Bool.false).toAlgorithm? = Except.ok alg) ∧
    ((fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
      Bool.true).toAlgorithm? =
        Except.ok
          (fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
            Bool.true).toAlgKernel)

/-- Surface proposition for case 3: no initial state and final-state output. -/
abbrev fused_recurrent_hgrn_test_case3_python_path_prop
    (X G O H0 Ht DO DX DG : RegionName) : Prop :=
    (∃ alg, (fused_recurrent_hgrn_fwd_surface X G O H0 Ht 2 2 32
      Bool.false Bool.true).toAlgorithm? = Except.ok alg) ∧
    ((fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
      Bool.false).toAlgorithm? =
        Except.ok
          (fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
            Bool.false).toAlgKernel)

/-- Surface proposition for case 4: initial state and final-state output. -/
abbrev fused_recurrent_hgrn_test_case4_python_path_prop
    (X G O H0 Ht DO DX DG : RegionName) : Prop :=
    (∃ alg, (fused_recurrent_hgrn_fwd_surface X G O H0 Ht 2 2 32
      Bool.true Bool.true).toAlgorithm? = Except.ok alg) ∧
    ((fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
      Bool.true).toAlgorithm? =
        Except.ok
          (fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
            Bool.true).toAlgKernel)

/-- Observable-output proposition for case 1. -/
abbrev fused_recurrent_hgrn_test_case1_outputs_prop
    (X G O H0 Ht DO DX DG : RegionName) (t_rel : Fin 2)
    (s : BlockState) : Prop :=
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_fwd_surface X G O H0 Ht 2 2 32
        Bool.false Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (O, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i =>
        producedForwardValue s X G O H0 Ht 2 2 32 Bool.false Bool.false
          t_rel.val i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
        Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DX, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i =>
        producedBackwardDxValue s G O H0 DX DG DO 2 2 32 Bool.false
          t_rel.val i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_surface G O H0 DX DG DO 2 2 32
        Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DG, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i =>
        producedBackwardDgValue s G O H0 DX DG DO 2 2 32 Bool.false
          t_rel.val i))

/-- Observable-output proposition for case 2. -/
abbrev fused_recurrent_hgrn_test_case2_outputs_prop
    (BH BHPrev X G DHPrev DO BO O DX DG : RegionName) (t_rel : Fin 2)
    (s : BlockState) : Prop :=
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_output_store_slice BH O t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (O, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => storeValue s BH 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_forward_step_store_slice BHPrev X G O
        t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (O, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i =>
        forwardStepValue s BHPrev X G t_rel.val 2 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_dx_step_store_slice DHPrev DO DX
        t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DX, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => bwdDxStepValue s DHPrev DO t_rel.val 2 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_dg_step_store_slice DHPrev DO BO DG
        t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DG, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => bwdDgStepValue s DHPrev DO BO t_rel.val 2 2 32 i))

/-- Observable-output proposition for case 3. -/
abbrev fused_recurrent_hgrn_test_case3_outputs_prop
    (BH BHFinal DHPrev DO BO O Ht DX DG : RegionName) (t_rel : Fin 2)
    (s : BlockState) : Prop :=
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_output_store_slice BH O t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (O, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => storeValue s BH 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_final_state_store_slice BHFinal Ht 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (Ht, finalStateOffset s 2 32 i)))
      (expected := fun i => finalStateStoreValue s BHFinal 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_dx_step_store_slice DHPrev DO DX
        t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DX, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => bwdDxStepValue s DHPrev DO t_rel.val 2 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_dg_step_store_slice DHPrev DO BO DG
        t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DG, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => bwdDgStepValue s DHPrev DO BO t_rel.val 2 2 32 i))

/-- Observable-output proposition for case 4. -/
abbrev fused_recurrent_hgrn_test_case4_outputs_prop
    (BH BHPrev BHFinal X G DHPrev DO BO O Ht DX DG : RegionName)
    (t_rel : Fin 2) (s : BlockState) : Prop :=
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_output_store_slice BH O t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (O, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => storeValue s BH 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_forward_step_store_slice BHPrev X G O
        t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (O, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i =>
        forwardStepValue s BHPrev X G t_rel.val 2 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_final_state_store_slice BHFinal Ht 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (Ht, finalStateOffset s 2 32 i)))
      (expected := fun i => finalStateStoreValue s BHFinal 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_dx_step_store_slice DHPrev DO DX
        t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DX, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => bwdDxStepValue s DHPrev DO t_rel.val 2 2 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_dg_step_store_slice DHPrev DO BO DG
        t_rel.val 2 2 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 2 32)
        (fun i => (DG, outOffset s t_rel.val 2 2 32 i)))
      (expected := fun i => bwdDgStepValue s DHPrev DO BO t_rel.val 2 2 32 i))

/- The case 1 public summary below is intentionally separated from the earlier
case propositions so the manifest context reflects its full-surface statement. -/


/-- Case 1 end-to-end summary for the Python regression shape: exact
forward/backward DSL surfaces plus the observable `o`, `dx`, and `dg` row
writebacks. -/
theorem fused_recurrent_hgrn_test_case1_output_summary
    (X G O H0 Ht DO DX DG : RegionName) (t_rel : Fin 2)
    (s : BlockState) :
    fused_recurrent_hgrn_test_case1_python_path_prop X G O H0 Ht DO DX DG ∧
    fused_recurrent_hgrn_test_case1_outputs_prop X G O H0 Ht DO DX DG
      t_rel s := by
  constructor
  · exact fused_recurrent_hgrn_test_case1_python_path_summary X G O H0 Ht DO DX DG
  · exact fused_recurrent_hgrn_test_case1_python_test_shape_all_outputs_compute_correct
      X G O H0 Ht DO DX DG t_rel s

/-- Case 2 end-to-end summary for the Python regression shape: initial-state
forward/backward DSL surfaces plus `o`, forward-step, `dx`, and `dg`
writebacks. -/
theorem fused_recurrent_hgrn_test_case2_output_summary
    (X G O H0 Ht DO DX DG BH BHPrev DHPrev BO : RegionName)
    (t_rel : Fin 2) (s : BlockState) :
    fused_recurrent_hgrn_test_case2_python_path_prop X G O H0 Ht DO DX DG ∧
    fused_recurrent_hgrn_test_case2_outputs_prop BH BHPrev X G DHPrev DO BO
      O DX DG t_rel s := by
  constructor
  · exact fused_recurrent_hgrn_test_case2_python_path_summary X G O H0 Ht DO DX DG
  · exact fused_recurrent_hgrn_test_case2_python_test_shape_all_outputs_compute_correct
      BH BHPrev X G DHPrev DO BO O DX DG t_rel s

/-- Case 3 end-to-end summary for the Python regression shape: no-initial
surfaces with final-state output plus `o`, `ht`, `dx`, and `dg` writebacks. -/
theorem fused_recurrent_hgrn_test_case3_output_summary
    (X G O H0 Ht DO DX DG BH BHFinal DHPrev BO : RegionName)
    (t_rel : Fin 2) (s : BlockState) :
    fused_recurrent_hgrn_test_case3_python_path_prop X G O H0 Ht DO DX DG ∧
    fused_recurrent_hgrn_test_case3_outputs_prop BH BHFinal DHPrev DO BO O Ht
      DX DG t_rel s := by
  constructor
  · exact fused_recurrent_hgrn_test_case3_python_path_summary X G O H0 Ht DO DX DG
  · exact fused_recurrent_hgrn_test_case3_python_test_shape_all_outputs_compute_correct
      BH BHFinal DHPrev DO BO O Ht DX DG t_rel s

/-- Case 4 end-to-end summary for the Python regression shape: initial-state
and final-state surfaces plus all observed forward/final/backward writebacks. -/
theorem fused_recurrent_hgrn_test_case4_output_summary
    (X G O H0 Ht DO DX DG BH BHPrev BHFinal DHPrev BO : RegionName)
    (t_rel : Fin 2) (s : BlockState) :
    fused_recurrent_hgrn_test_case4_python_path_prop X G O H0 Ht DO DX DG ∧
    fused_recurrent_hgrn_test_case4_outputs_prop BH BHPrev BHFinal X G DHPrev
      DO BO O Ht DX DG t_rel s := by
  constructor
  · exact fused_recurrent_hgrn_test_case4_python_path_summary X G O H0 Ht DO DX DG
  · exact fused_recurrent_hgrn_test_case4_python_test_shape_all_outputs_compute_correct
      BH BHPrev BHFinal X G DHPrev DO BO O Ht DX DG t_rel s

/-- Combined checked-shape summary for
`test_fused_recurrent_hgrn_with_backward`.

This collects the four Python-tested initial/final-state branches, the matching
forward/backward DSL surfaces, and all observable `o`, `final_state`, `dx`, and
`dg` row writebacks in one public target. -/
theorem fused_recurrent_hgrn_python_test_shape_complete_summary
    (X G O H0 Ht DO DX DG BH BHPrev BHFinal DHPrev BO : RegionName)
    (t_rel : Fin 2) (s : BlockState) :
    (fused_recurrent_hgrn_test_case1_python_path_prop X G O H0 Ht DO DX DG ∧
      fused_recurrent_hgrn_test_case1_outputs_prop X G O H0 Ht DO DX DG
        t_rel s) ∧
    (fused_recurrent_hgrn_test_case2_python_path_prop X G O H0 Ht DO DX DG ∧
      fused_recurrent_hgrn_test_case2_outputs_prop BH BHPrev X G DHPrev DO BO
        O DX DG t_rel s) ∧
    (fused_recurrent_hgrn_test_case3_python_path_prop X G O H0 Ht DO DX DG ∧
      fused_recurrent_hgrn_test_case3_outputs_prop BH BHFinal DHPrev DO BO O
        Ht DX DG t_rel s) ∧
    (fused_recurrent_hgrn_test_case4_python_path_prop X G O H0 Ht DO DX DG ∧
      fused_recurrent_hgrn_test_case4_outputs_prop BH BHPrev BHFinal X G
        DHPrev DO BO O Ht DX DG t_rel s) := by
  constructor
  · exact fused_recurrent_hgrn_test_case1_output_summary X G O H0 Ht DO DX DG
      t_rel s
  constructor
  · exact fused_recurrent_hgrn_test_case2_output_summary X G O H0 Ht DO DX DG
      BH BHPrev DHPrev BO t_rel s
  constructor
  · exact fused_recurrent_hgrn_test_case3_output_summary X G O H0 Ht DO DX DG
      BH BHFinal DHPrev BO t_rel s
  · exact fused_recurrent_hgrn_test_case4_output_summary X G O H0 Ht DO DX DG
      BH BHPrev BHFinal DHPrev BO t_rel s

end TestShape

end VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn
