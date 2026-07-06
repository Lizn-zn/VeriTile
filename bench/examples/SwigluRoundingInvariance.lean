import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Activation

/-!
# SwiGLU fused vs unfused — rounding-invariance pilot (#447 Phase C)

The first `ComputeRefine.RealizesR` theorems: two SwiGLU pipelines that are
**definitionally equal** under the erased/ideal semantics (`ComputeCorrect.*`)
but **distinct terms** under the rounding-model semantics, with their full
rounding-event structure pinned down for every `RoundingModel`.

## The two pipelines

* `swiglu_fused` — the Triton-kernel dataflow: `silu(x)·y` stays on the ℝ
  register channel, one bf16 materialization at the output
  (`(out).to(tl.bfloat16)` + bf16 store).
* `swiglu_unfused` — the eager dataflow: the intermediate `silu(x)` is
  materialized to bf16 (`(s).to(tl.bfloat16)`) before the multiply, exactly
  like an eager `F.silu(x)` writing a bf16 tensor.

## Rounding-event ledger (what the raw specs record)

Each `R.round .bf16` in a spec is one event; the mixed-dtype multiply's
automatic bf16→ℝ upcast is `R.round .real = id` — upcast exactness holds *by
the model's only axiom*, no grid hypotheses needed.

```
fused   : round(round(silu(x)·y))                two events: exit cast + store
unfused : round(round(round(silu(x))·y))         three: materialize + exit cast + store
```

Under `IdemRounding` (every real rounding is idempotent) the cast+store pairs
collapse and the pipelines take the shapes of the #447 narrative:

```
fused   : round(silu(x)·y)                       exactly ONE rounding of the ℝ core
unfused : round(round(silu(x))·y)                the extra intermediate rounding
```

## Theorems

* `swiglu_fused_realizesR` / `swiglu_unfused_realizesR` — the unconditional
  ∀-`R` realization theorems (the pilot's main results). Outputs are read at
  the `MemCell` layer (`MemCell.of .bf16 (some …)`): the store is bf16-tagged
  and `readMem`'s real-channel decode is tag-exact, so a narrow-float output
  cell is only observable as its typed `MemCell` — the same convention the
  classic bench proofs use for fp16/bf16 stores (`rmsnorm_fused_llama`,
  `scatter_memcell_fp16_prop_masked_nd`).
* `fusedSpec_single_rounding` / `unfusedSpec_double_rounding` — the Idem
  collapses above (pure spec algebra).
* `fusedSpec_eq_unfusedSpec_of_representable` — conditional coincidence: if
  the intermediate `silu(x)` is a fixed point of `R.round .bf16`, the two
  pipelines agree.
* `fusedSpec_ne_unfusedSpec` — the model *distinguishes* the pipelines: a
  concrete witness model on which the specs differ (unprovable before #447:
  both erased to the same closed form).
* `swiglu_fused_realizes_classic` — degeneration: the classic
  `ComputeCorrect.Realizes` statement with the ℝ closed form inside the bf16
  output cell, recovered from the invariance theorem at the trivial model.
-/

namespace VeriTile.Bench.Examples.SwigluRounding

open VeriTile.Triton

/-- Fused SwiGLU: compute on the ℝ channel, single bf16 materialization at
the output store. -/
def swiglu_fused (X Y OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel := triton {
  start_col = tl.program_id(0) * $(BLOCK_N)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0)
  out = x * tl.sigmoid(x) * y
  tl.store(OUT + cols, (out).to(tl.bfloat16), mask=cols < $(ncols))
}

/-- Unfused SwiGLU: the intermediate `silu(x)` is materialized to bf16 before
the multiply (the eager-pipeline dataflow). The multiply's automatic bf16→ℝ
upcast is exact (`round_real = id`). -/
def swiglu_unfused (X Y OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel := triton {
  start_col = tl.program_id(0) * $(BLOCK_N)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0)
  s = x * tl.sigmoid(x)
  s16 = (s).to(tl.bfloat16)
  out = s16 * y
  tl.store(OUT + cols, (out).to(tl.bfloat16), mask=cols < $(ncols))
}

/-- Lane address for this example's 1-D launch. -/
def laneOffset (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * BLOCK_N + i.val

/-- Raw fused spec: exit cast + bf16 store — two rounding events around the
shared ℝ core. -/
noncomputable def fusedSpec (xs ys : Fin BLOCK_N → ℝ) (R : RoundingModel) (i : Fin BLOCK_N) : ℝ :=
  R.round .bf16 (R.round .bf16 (TiledActivation.swiglu (xs i) (ys i)))

/-- Raw unfused spec: intermediate materialization + exit cast + store —
three rounding events; the extra one wraps `silu(x)` alone. -/
noncomputable def unfusedSpec (xs ys : Fin BLOCK_N → ℝ) (R : RoundingModel) (i : Fin BLOCK_N) : ℝ :=
  R.round .bf16 (R.round .bf16 (R.round .bf16 (TiledActivation.silu (xs i)) * ys i))

/-! ## Spec algebra (no kernel execution involved) -/

theorem fusedSpec_single_rounding {BLOCK_N : Nat} (xs ys : Fin BLOCK_N → ℝ)
    (R : RoundingModel) [IdemRounding R] (i : Fin BLOCK_N) :
    fusedSpec xs ys R i =
      R.round .bf16 (TiledActivation.swiglu (xs i) (ys i)) := by
  simp [fusedSpec, IdemRounding.idem]

theorem unfusedSpec_double_rounding {BLOCK_N : Nat} (xs ys : Fin BLOCK_N → ℝ)
    (R : RoundingModel) [IdemRounding R] (i : Fin BLOCK_N) :
    unfusedSpec xs ys R i =
      R.round .bf16 (R.round .bf16 (TiledActivation.silu (xs i)) * ys i) := by
  simp [unfusedSpec, IdemRounding.idem]

/-- Conditional coincidence: when the intermediate `silu(x)` is already
representable, the extra materialization is invisible and the two pipelines
agree. -/
theorem fusedSpec_eq_unfusedSpec_of_representable {BLOCK_N : Nat}
    (xs ys : Fin BLOCK_N → ℝ) (R : RoundingModel) (i : Fin BLOCK_N)
    (h : R.Representable .bf16 (TiledActivation.silu (xs i))) :
    fusedSpec xs ys R i = unfusedSpec xs ys R i := by
  unfold fusedSpec unfusedSpec
  rw [h, TiledActivation.swiglu]

/-- The model distinguishes the two pipelines: a concrete rounding model and
inputs on which the specs differ. (Under the erased semantics both pipelines
have the same closed form, so this statement was previously unwritable.) -/
theorem fusedSpec_ne_unfusedSpec :
    ∃ (R : RoundingModel) (xs ys : Fin 1 → ℝ) (i : Fin 1),
      fusedSpec xs ys R i ≠ unfusedSpec xs ys R i := by
  -- Witness model: rounding shifts every non-real channel by 1 (a legitimate
  -- `RoundingModel`: the only axiom constrains the `.real` channel).
  -- At `x = 0, y = 1`: `silu 0 = 0`, so
  --   fused   = round (round 0)           = 2
  --   unfused = round (round (round 0·1)) = 3.
  refine ⟨⟨fun dt x => if dt = .real then x else x + 1, by funext x; simp⟩,
    fun _ => 0, fun _ => 1, 0, ?_⟩
  simp [fusedSpec, unfusedSpec, TiledActivation.swiglu, TiledActivation.silu]

/-! ## The realization theorems -/

/-- The fused kernel realizes `fusedSpec` for **every** rounding model.

Outputs are read at the `MemCell` layer (`MemCell.of .bf16 (some …)`): the
store is bf16-tagged and `readMem`'s real-channel decode is tag-exact, so a
narrow-float output cell is only observable as its typed `MemCell` — mirroring
the classic convention for fp16/bf16 stores (`rmsnorm_fused_llama`,
`scatter_memcell_fp16_prop_masked_nd`). -/
theorem swiglu_fused_realizesR
    (X Y OUT : RegionName) (ncols BLOCK_N : Nat)
    (s : BlockState) (xs ys : Fin BLOCK_N → ℝ)
    (h_x : ∀ i : Fin BLOCK_N, s.readMem X (laneOffset s BLOCK_N i) = xs i)
    (h_y : ∀ i : Fin BLOCK_N, s.readMem Y (laneOffset s BLOCK_N i) = ys i) :
    ComputeRefine.RealizesR
      (kernel := swiglu_fused X Y OUT ncols BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => s.pids 0 * BLOCK_N + i.val < ncols)
        (fun i => (OUT, laneOffset s BLOCK_N i)))
      (expected := fun R i =>
        MemCell.of .bf16 (some (fusedSpec xs ys R i) : WithBot ℝ)) := by
  rw [ComputeRefine.realizesR_writeIf_iff]
  intro R
  apply ComputeKernel.computeCorrectR_of_toAlgKernel
  · simp [swiglu_fused, ComputeExpr.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * BLOCK_N + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have hab' : s.pids 0 * BLOCK_N + a.val = s.pids 0 * BLOCK_N + b.val := by
      simpa using hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab')
    rfl
  simp [execR, swiglu_fused, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  simp only [ComputeCorrect.OutputReadable.read_memcell, laneOffset]
  rw [BlockState.scatter_memcell_R_prop_masked_nd R .bf16 _ _ _ _ h_inj (i, PUnit.unit)]
  have hx := h_x i
  have hy := h_y i
  simp only [laneOffset] at hx hy
  simp [hActive, fusedSpec, TiledActivation.swiglu, TiledActivation.silu,
        tile_elementwise, hx, hy, RoundingModel.cast, RoundingModel.storeValue,
        FloatDType.storeValue, FloatDType.ofReal]

/-- The unfused kernel realizes `unfusedSpec` for **every** rounding model.
Same `MemCell`-layer readback convention as `swiglu_fused_realizesR`. -/
theorem swiglu_unfused_realizesR
    (X Y OUT : RegionName) (ncols BLOCK_N : Nat)
    (s : BlockState) (xs ys : Fin BLOCK_N → ℝ)
    (h_x : ∀ i : Fin BLOCK_N, s.readMem X (laneOffset s BLOCK_N i) = xs i)
    (h_y : ∀ i : Fin BLOCK_N, s.readMem Y (laneOffset s BLOCK_N i) = ys i) :
    ComputeRefine.RealizesR
      (kernel := swiglu_unfused X Y OUT ncols BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => s.pids 0 * BLOCK_N + i.val < ncols)
        (fun i => (OUT, laneOffset s BLOCK_N i)))
      (expected := fun R i =>
        MemCell.of .bf16 (some (unfusedSpec xs ys R i) : WithBot ℝ)) := by
  rw [ComputeRefine.realizesR_writeIf_iff]
  intro R
  apply ComputeKernel.computeCorrectR_of_toAlgKernel
  · simp [swiglu_unfused, ComputeExpr.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * BLOCK_N + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have hab' : s.pids 0 * BLOCK_N + a.val = s.pids 0 * BLOCK_N + b.val := by
      simpa using hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab')
    rfl
  simp [execR, swiglu_unfused, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  simp only [ComputeCorrect.OutputReadable.read_memcell, laneOffset]
  rw [BlockState.scatter_memcell_R_prop_masked_nd R .bf16 _ _ _ _ h_inj (i, PUnit.unit)]
  have hx := h_x i
  have hy := h_y i
  simp only [laneOffset] at hx hy
  simp [hActive, unfusedSpec, TiledActivation.silu,
        tile_elementwise, hx, hy, RoundingModel.cast, RoundingModel.storeValue,
        FloatDType.storeValue, FloatDType.ofReal]

/-! ## Degeneration to the classic surface -/

/-- At the trivial model the invariance theorem collapses to the classic
`ComputeCorrect.Realizes` statement with the ℝ closed form inside the bf16
output cell — the existing proof style is the shadow of the new one. -/
theorem swiglu_fused_realizes_classic
    (X Y OUT : RegionName) (ncols BLOCK_N : Nat)
    (s : BlockState) (xs ys : Fin BLOCK_N → ℝ)
    (h_x : ∀ i : Fin BLOCK_N, s.readMem X (laneOffset s BLOCK_N i) = xs i)
    (h_y : ∀ i : Fin BLOCK_N, s.readMem Y (laneOffset s BLOCK_N i) = ys i) :
    ComputeCorrect.Realizes
      (kernel := swiglu_fused X Y OUT ncols BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => s.pids 0 * BLOCK_N + i.val < ncols)
        (fun i => (OUT, laneOffset s BLOCK_N i)))
      (expected := fun i =>
        MemCell.of .bf16 (some (TiledActivation.swiglu (xs i) (ys i)) : WithBot ℝ)) := by
  have h := (swiglu_fused_realizesR X Y OUT ncols BLOCK_N s xs ys h_x h_y).toRealizes
  simpa [fusedSpec] using h

end VeriTile.Bench.Examples.SwigluRounding
