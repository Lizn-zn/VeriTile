import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# SwiGLU fused vs unfused — rounding-invariance pilot (#447 Phase C)

**One `R⟦·⟧` is ONE bf16 rounding event — count the brackets to count the
events.** What the theorems say, per active output lane:

```
fused              ↦ R⟦R⟦silu(x)·y⟧⟧                                (2 events)
A  = silu_step     ↦ R⟦R⟦silu(x)⟧⟧                       =: aSpec   (2 events)
B  = mul_step      ↦ R⟦R⟦z·y⟧⟧                                      (2 events)
unfused = B ∘ A    ↦ R⟦R⟦ R⟦R⟦silu(x)⟧⟧ · y ⟧⟧                      (4 events)
```

Each kernel's two events are its exit cast + bf16 store. Loads of bf16 cells
return the stored payload (tag-exact); the mixed-dtype multiply's automatic
bf16→ℝ upcast is `R.round .real = id` — exact by the model's only axiom.
Under `IdemRounding` the cast+store pairs collapse pairwise to the
#447-narrative shapes: `R⟦silu(x)·y⟧` vs `R⟦R⟦silu(x)⟧·y⟧`.

Compositional form, mirroring `VeriTile.Examples.FusedSiLU`: the unfused
pipeline is **two separate kernels** (`silu_step`, then `mul_step`) composed
through a real bf16 tensor in memory — exactly what an eager framework
launches — and the final theorems live on the two-kernel refinement surfaces:
the writes-equality `ComputeRefine.RefinesR` (the headline) and the pointwise
`ComputeRefine.RefinesAtR` (the event ledger).

## The kernels

* `swiglu_fused` — one kernel: `silu(x)·y` on the ℝ register channel, one
  bf16 materialization at the output store.
* `silu_step` (**A**) — writes the intermediate `silu(x)` to the bf16 tensor
  `S` in memory.
* `mul_step` (**B**) — loads `S` back (a bf16-typed load), multiplies by `y`,
  stores bf16.
* `swiglu_unfused` — the pipeline `B ∘ A` as `ComputeKernel.seq [X, Y] [OUT]
  [A, B]`: the concatenation of the two step kernels' bodies;
  `exec_swiglu_unfusedR` splits its execution into the sequential
  composition, so the pipeline theorem is *derived from* the two step
  theorems rather than proven monolithically.

## Honest launch semantics

Concatenation lets register bindings flow across the A/B seam — something no
real two-launch execution allows. The honest semantics is `execPipelineR R
[A, B] s`: registers reset (`BlockState.resetRegs`) before **every** launch.
The library bridge theorem `ComputeKernel.execR_seq_rel_execPipelineR`
discharges that register-leakage gap: because `silu_step`, `mul_step` (and
`swiglu_fused`) are proven `RegClosed` here — their executions never depend
on pre-existing register bindings — the concatenated pipeline and the honest
two-launch pipeline agree on everything but the final dead register file,
and the headline theorem transfers to real launches
(`swiglu_fused_launch_eq_pipeline_of_representable`).

## Public surface

Outputs are read at the `MemCell` layer (`floatCell .bf16 v = MemCell.of .bf16
(some v)`, from `VeriTile.Examples.Common`): the stores are bf16-tagged and
`readMem`'s real-channel decode is tag-exact, so a narrow-float cell is only
observable as its typed `MemCell` — the classic fp16/bf16 convention
(`rmsnorm_fused_llama`).

The reader-facing declarations are the four kernels above and:

* **`swiglu_fused_eq_unfused_of_representable`** — the headline "fused =
  unfused" on the writes-equality surface `ComputeRefine.RefinesR`: when the
  intermediate `silu(x)` is representable, the two pipelines perform the same
  writes — final memories agree everywhere outside the scratch tensor `S`, for
  every rounding model.
* **`swiglu_refines_classic`** — its degeneration at `triv`: the canonical
  `ComputeRefine.Refines` writes-equality statement.
* **`swiglu_fused_launch_eq_pipeline_of_representable`** — the honest-launch
  corollary: the fused single launch and the honest two-launch pipeline
  (`execPipelineR`, registers reset at every launch) agree outside `S`.
* `swiglu_refinesR` — the event-ledger pair theorem (`ComputeRefine.RefinesAtR`):
  for every rounding model, fused and unfused realize their respective
  `R⟦…⟧`-annotated terms side by side.
* Spec algebra (the ledger, no kernel execution): `fusedSpec_single_rounding`,
  `unfusedSpec_shape` (Idem collapses), `fusedSpec_ne_unfusedSpec` (a witness
  model distinguishing the pipelines — unstatable before #447).

Everything else is `private` supporting machinery: the per-kernel component
realizations (`silu_step_realizesR`, `mul_step_realizesR`,
`swiglu_fused_realizesR`, `swiglu_unfused_realizesR` — A and B verified
separately, then composed), the aliasing/frame lemmas, the register-closedness
proofs (`*_regClosed`, inputs to the launch bridge), and the exec split
`exec_swiglu_unfusedR`.
-/

namespace VeriTile.Bench.Examples.SwigluRounding

open VeriTile.Triton
open VeriTile.Examples (InputLoadedAt outWritesTo floatCell InputCellsLoadedAt programLaneOffset)

/-! ## Kernels -/

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

/-- Step A of the unfused pipeline: materialize `silu(x)` into the bf16
tensor `S`. -/
def silu_step (X S : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel := triton {
  start_col = tl.program_id(0) * $(BLOCK_N)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0)
  s = x * tl.sigmoid(x)
  tl.store(S + cols, (s).to(tl.bfloat16), mask=cols < $(ncols))
}

/-- Step B of the unfused pipeline: load the bf16 intermediate back and
multiply by `y`. The `S` load is bf16-typed (it reads bf16 cells); the
mixed-dtype multiply upcasts it to ℝ exactly. -/
def mul_step (S Y OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel := triton {
  start_col = tl.program_id(0) * $(BLOCK_N)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  z = tl.load(S + cols, mask=cols < $(ncols)).to(tl.bfloat16)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0)
  out = z * y
  tl.store(OUT + cols, (out).to(tl.bfloat16), mask=cols < $(ncols))
}

/-- The unfused pipeline `B ∘ A` as one kernel: `ComputeKernel.seq` of the
two step kernels — the concatenation of their bodies (registers flowing
across the seam; the honest per-launch semantics is `execPipelineR`, related
by the launch bridge below). -/
def swiglu_unfused (X Y S OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel :=
  ComputeKernel.seq [X, Y] [OUT]
    [silu_step X S ncols BLOCK_N, mul_step S Y OUT ncols BLOCK_N]

/- Pre-defined variables used in the following -/
variable (X Y S OUT : RegionName) (ncols BLOCK_N : Nat) (s : BlockState)
variable (xs ys : Fin BLOCK_N → ℝ)

/-! ## Sequential decomposition of the pipeline -/

/-- Execution of the composed pipeline `A ; B` as the sequential composition
of the two step executions — the two-stage instance of the library split
`execR_toAlgKernel_seq` (the no-reset fold over the `ComputeKernel.seq`
stages). -/
private theorem exec_swiglu_unfusedR (R : RoundingModel)
    (X Y S OUT : RegionName) (ncols BLOCK_N : Nat) (s : BlockState) :
    execR R (swiglu_unfused X Y S OUT ncols BLOCK_N) s =
      (execR R (silu_step X S ncols BLOCK_N) s).bind
        (fun s1 => execR R (mul_step S Y OUT ncols BLOCK_N) s1) := by
  show execR R (ComputeKernel.seq [X, Y] [OUT]
      [silu_step X S ncols BLOCK_N, mul_step S Y OUT ncols BLOCK_N]).toAlgKernel s = _
  rw [execR_toAlgKernel_seq]
  simp only [List.foldl_cons, List.foldl_nil, Option.bind_some]

/-! ## Lane addressing, readback carrier, and the shared write map

Input hypotheses use the shared `VeriTile.Examples.InputLoadedAt` contract
(region holds `xs` at `s.pid * N + i` — the canonical per-program 1-D tile);
output claims use its write-side counterpart `VeriTile.Examples.outWritesTo`
(the same lanes, masked by `< ncols`). `programLaneOffset` is that same address,
named for the proofs. -/

/-- Each `R⟦v⟧` is ONE bf16 rounding event — count the brackets to count
the events. -/
local notation:max R "⟦" v "⟧" => RoundingModel.round R FloatDType.bf16 v

/-! ## Specs (the rounding-event ledger, term by term) -/

/-- Step A's per-lane payload: `silu(x)` through A's exit cast + bf16 store. -/
noncomputable def aSpec (R : RoundingModel) (x : ℝ) : ℝ :=
  R⟦R⟦TiledActivation.silu x⟧⟧

/-- Fused spec: exit cast + store — two events around the shared ℝ core. -/
noncomputable def fusedSpec {BLOCK_N : Nat} (xs ys : Fin BLOCK_N → ℝ) (R : RoundingModel) (i : Fin BLOCK_N) : ℝ :=
  R⟦R⟦TiledActivation.swiglu (xs i) (ys i)⟧⟧

/-- Unfused (composed) spec: B's two events around `aSpec(x)·y` — four events
total, of which the inner two are A's. -/
noncomputable def unfusedSpec {BLOCK_N : Nat} (xs ys : Fin BLOCK_N → ℝ) (R : RoundingModel) (i : Fin BLOCK_N) : ℝ :=
  R⟦R⟦aSpec R (xs i) * ys i⟧⟧

/-! ## Spec algebra (no kernel execution involved) -/

theorem fusedSpec_single_rounding {BLOCK_N : Nat} (xs ys : Fin BLOCK_N → ℝ)
    (R : RoundingModel) [IdemRounding R] (i : Fin BLOCK_N) :
    fusedSpec xs ys R i = R⟦TiledActivation.swiglu (xs i) (ys i)⟧ := by
  simp [fusedSpec, IdemRounding.idem]

/-- Under idempotence the four unfused events collapse pairwise to the #447
narrative shape: `R⟦R⟦silu(x)⟧·y⟧`. -/
theorem unfusedSpec_shape {BLOCK_N : Nat} (xs ys : Fin BLOCK_N → ℝ)
    (R : RoundingModel) [IdemRounding R] (i : Fin BLOCK_N) :
    unfusedSpec xs ys R i = R⟦R⟦TiledActivation.silu (xs i)⟧ * ys i⟧ := by
  simp [unfusedSpec, aSpec, IdemRounding.idem]

/-- Conditional coincidence: when the intermediate `silu(x)` is representable,
A's materialization is invisible and the specs agree. -/
theorem fusedSpec_eq_unfusedSpec_of_representable {BLOCK_N : Nat}
    (xs ys : Fin BLOCK_N → ℝ) (R : RoundingModel) (i : Fin BLOCK_N)
    (h : R.Representable .bf16 (TiledActivation.silu (xs i))) :
    fusedSpec xs ys R i = unfusedSpec xs ys R i := by
  unfold fusedSpec unfusedSpec aSpec
  rw [h, h, TiledActivation.swiglu]

/-- The model distinguishes the two pipelines: a concrete rounding model and
inputs on which the specs differ. (Under the erased semantics both pipelines
have the same closed form, so this statement was previously unwritable.) -/
theorem fusedSpec_ne_unfusedSpec :
    ∃ (R : RoundingModel) (xs ys : Fin 1 → ℝ) (i : Fin 1),
      fusedSpec xs ys R i ≠ unfusedSpec xs ys R i := by
  -- Witness model: rounding shifts every non-real channel by 1 (a legitimate
  -- `RoundingModel`: the only axiom constrains the `.real` channel).
  -- At `x = 0, y = 1`: `silu 0 = 0`, so
  --   fused   = R⟦R⟦0⟧⟧             = 2
  --   unfused = R⟦R⟦aSpec 0 · 1⟧⟧   = R⟦R⟦2⟧⟧ = 4.
  refine ⟨⟨fun dt x => if dt = .real then x else x + 1, by funext x; simp⟩,
    fun _ => 0, fun _ => 1, 0, ?_⟩
  simp [fusedSpec, unfusedSpec, aSpec, TiledActivation.swiglu, TiledActivation.silu]

/-! ## Component theorems -/

/-- The fused kernel realizes `fusedSpec` for **every** rounding model. -/
private theorem swiglu_fused_realizesR
    (h_x : InputLoadedAt s X BLOCK_N xs) (h_y : InputLoadedAt s Y BLOCK_N ys) :
    ComputeRefine.RealizesR (swiglu_fused X Y OUT ncols BLOCK_N) s
      (outWritesTo s OUT ncols BLOCK_N)
      (fun R i => floatCell .bf16 (fusedSpec xs ys R i)) := by
  unfold InputLoadedAt at h_x h_y
  unfold outWritesTo
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
  simp only [ComputeCorrect.OutputReadable.read_memcell, BlockState.pid_eq]
  rw [BlockState.scatter_memcell_R_prop_masked_nd R .bf16 _ _ _ _ h_inj (i, PUnit.unit)]
  have hx := h_x i
  have hy := h_y i
  simp only [BlockState.pid_eq] at hx hy
  simp [hActive, floatCell, fusedSpec, TiledActivation.swiglu, TiledActivation.silu,
        tile_elementwise, hx, hy, RoundingModel.cast, RoundingModel.storeValue,
        FloatDType.storeValue, FloatDType.ofReal]

/-- Component theorem for step A: for every rounding model, the active lanes
of `S` receive `aSpec(x)` as bf16 cells. -/
private theorem silu_step_realizesR (h_x : InputLoadedAt s X BLOCK_N xs) :
    ComputeRefine.RealizesR (silu_step X S ncols BLOCK_N) s
      (outWritesTo s S ncols BLOCK_N)
      (fun R i => floatCell .bf16 (aSpec R (xs i))) := by
  unfold InputLoadedAt at h_x
  unfold outWritesTo
  rw [ComputeRefine.realizesR_writeIf_iff]
  intro R
  apply ComputeKernel.computeCorrectR_of_toAlgKernel
  · simp [silu_step, ComputeExpr.toAlgorithm?]
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
  simp [execR, silu_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  simp only [ComputeCorrect.OutputReadable.read_memcell, BlockState.pid_eq]
  rw [BlockState.scatter_memcell_R_prop_masked_nd R .bf16 _ _ _ _ h_inj (i, PUnit.unit)]
  have hx := h_x i
  simp only [BlockState.pid_eq] at hx
  simp [hActive, floatCell, aSpec, TiledActivation.silu,
        tile_elementwise, hx, RoundingModel.cast, RoundingModel.storeValue,
        FloatDType.storeValue, FloatDType.ofReal]

/-- Frame lemma for step A: every region other than `S` is untouched, for
every rounding model. -/
private theorem silu_step_preservesR (R : RoundingModel) (s' : BlockState)
    (keepReg : RegionName) (h_keep : keepReg ≠ S)
    (hExec : execR R (silu_step X S ncols BLOCK_N) s = some s') :
    ∀ offset : Nat, s'.mem keepReg offset = s.mem keepReg offset := by
  intro offset
  simp [execR, silu_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMemAsR_preserve_other_region R .bf16 _ _ _
        keepReg h_keep offset]
  rfl

/-- Step A leaves the program ids untouched (needed to carry the lane
addressing across the composition). -/
private theorem silu_step_execR_pids (R : RoundingModel) (s1 : BlockState)
    (hExec : execR R (silu_step X S ncols BLOCK_N) s = some s1) :
    s1.pids = s.pids := by
  simp [execR, silu_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s1
  rw [BlockState.foldl_writeMemAsR_masked_pids]
  rfl

/-- Component theorem for step B: given the bf16 intermediate `S` holding
payloads `zs`, the active lanes of `OUT` receive `R⟦R⟦z·y⟧⟧`. -/
private theorem mul_step_realizesR (zs : Fin BLOCK_N → ℝ)
    (h_z : InputCellsLoadedAt s S ncols BLOCK_N .bf16 zs) (h_y : InputLoadedAt s Y BLOCK_N ys) :
    ComputeRefine.RealizesR (mul_step S Y OUT ncols BLOCK_N) s
      (outWritesTo s OUT ncols BLOCK_N)
      (fun R i => floatCell .bf16 R⟦R⟦zs i * ys i⟧⟧) := by
  unfold InputCellsLoadedAt floatCell at h_z
  unfold InputLoadedAt at h_y
  unfold outWritesTo
  rw [ComputeRefine.realizesR_writeIf_iff]
  intro R
  apply ComputeKernel.computeCorrectR_of_toAlgKernel
  · simp [mul_step, ComputeExpr.toAlgorithm?]
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
  simp [execR, mul_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  simp only [ComputeCorrect.OutputReadable.read_memcell, BlockState.pid_eq]
  rw [BlockState.scatter_memcell_R_prop_masked_nd R .bf16 _ _ _ _ h_inj (i, PUnit.unit)]
  have hz := h_z i hActive
  have hy := h_y i
  simp only [BlockState.pid_eq] at hz hy
  simp [hActive, floatCell, BlockState.readMemValue_bf16_of_cell hz, hy,
        tile_elementwise, RoundingModel.cast, RoundingModel.storeValue,
        FloatDType.storeValue, FloatDType.ofReal]

/-! ## The composed pipeline theorem -/

/-- Unpack a `RealizesR` statement at one model and one successful execution
of the projected kernel: the raw per-lane output clause. -/
private theorem realizesR_out {ι : Type} {α : Type}
    [ComputeCorrect.OutputReadable α]
    {k : ComputeKernel} {st : BlockState}
    {write : ComputeCorrect.WriteMap ι} {expected : RoundingModel → ι → α}
    (h : ComputeRefine.RealizesR k st write expected) (R : RoundingModel)
    (hAlg : k.toAlgorithm? = Except.ok k.toAlgKernel)
    {s' : BlockState}
    (hExec : execR R k.toAlgKernel st = some s') :
    ∀ i : ι, match write i with
      | some addr => ComputeCorrect.OutputReadable.read s' addr = expected R i
      | none => True := by
  have h' := (h R).2
  unfold ComputeKernel.AlgorithmCorrectR at h'
  rw [hAlg] at h'
  exact h' st s' hExec rfl

/-- The unfused pipeline realizes `unfusedSpec` for **every** rounding model —
**derived by composing** `silu_step_realizesR` and `mul_step_realizesR`
through the intermediate state (with the frame lemma keeping `Y` intact;
`S ≠ Y` is the aliasing hypothesis the composition needs). -/
private theorem swiglu_unfused_realizesR
    (h_x : InputLoadedAt s X BLOCK_N xs) (h_y : InputLoadedAt s Y BLOCK_N ys)
    (h_SY : S ≠ Y) :
    ComputeRefine.RealizesR (swiglu_unfused X Y S OUT ncols BLOCK_N) s
      (outWritesTo s OUT ncols BLOCK_N)
      (fun R i => floatCell .bf16 (unfusedSpec xs ys R i)) := by
  unfold outWritesTo
  rw [ComputeRefine.realizesR_writeIf_iff]
  intro R
  apply ComputeKernel.computeCorrectR_of_toAlgKernel
  · simp [swiglu_unfused]
  intro s0 s' hExec hs0
  subst s0
  rw [exec_swiglu_unfusedR] at hExec
  intro i hActive
  cases hA : execR R (ComputeKernel.toAlgKernel (silu_step X S ncols BLOCK_N)) s with
  | none => rw [hA] at hExec; exact absurd hExec (by simp)
  | some s1 =>
      rw [hA] at hExec
      have hExecB : execR R (ComputeKernel.toAlgKernel (mul_step S Y OUT ncols BLOCK_N)) s1
          = some s' := hExec
      have hpids : s1.pids = s.pids := silu_step_execR_pids X S ncols BLOCK_N s R s1 hA
      -- intermediate `S` cells from step A's realization
      have hAout := realizesR_out (silu_step_realizesR X S ncols BLOCK_N s xs h_x) R
        (by simp [silu_step, ComputeExpr.toAlgorithm?]) hA
      have h_z1 : InputCellsLoadedAt s1 S ncols BLOCK_N .bf16 (fun j => aSpec R (xs j)) := by
        intro j hj
        rw [BlockState.pid_eq] at hj
        have hj' : s.pids 0 * BLOCK_N + j.val < ncols := by rwa [hpids] at hj
        have hout := hAout j
        simp only [outWritesTo, ComputeCorrect.WriteMap.writeIf, BlockState.pid_eq, hj', if_true] at hout
        simpa [BlockState.pid_eq, hpids] using hout
      -- `Y` survives step A (frame lemma; `S ≠ Y`)
      have h_y1 : InputLoadedAt s1 Y BLOCK_N ys := by
        intro j
        have hmem := silu_step_preservesR X S ncols BLOCK_N s R s1 Y (Ne.symm h_SY) hA
        have hread : s1.readMem Y (programLaneOffset s1 BLOCK_N j)
            = s.readMem Y (programLaneOffset s1 BLOCK_N j) := by
          unfold BlockState.readMem
          rw [hmem]
        show s1.readMem Y (programLaneOffset s1 BLOCK_N j) = ys j
        rw [hread]
        have hy := h_y j
        simp only [BlockState.pid_eq, programLaneOffset, hpids] at hy ⊢
        exact hy
      -- step B's realization at the intermediate state
      have hBout := realizesR_out
        (mul_step_realizesR Y S OUT ncols BLOCK_N s1 ys (fun j => aSpec R (xs j)) h_z1 h_y1)
        R (by simp [mul_step, ComputeExpr.toAlgorithm?]) hExecB
      have hAct1 : s1.pids 0 * BLOCK_N + i.val < ncols := by rw [hpids]; exact hActive
      have hout := hBout i
      simp only [outWritesTo, ComputeCorrect.WriteMap.writeIf, BlockState.pid_eq, hAct1, if_true] at hout
      simpa [programLaneOffset, hpids, unfusedSpec] using hout

/-! ## Frame lemmas (the writes each kernel does NOT perform) -/

/-- Frame lemma for the fused kernel: every region other than `OUT` is
untouched. -/
private theorem swiglu_fused_preservesR (R : RoundingModel) (s' : BlockState)
    (r : RegionName) (hrOUT : r ≠ OUT)
    (hExec : execR R (swiglu_fused X Y OUT ncols BLOCK_N) s = some s') :
    ∀ o : Nat, s'.mem r o = s.mem r o := by
  intro o
  simp [execR, swiglu_fused, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMemAsR_preserve_other_region R .bf16 _ _ _ r hrOUT o]
  rfl

/-- Frame lemma for the fused kernel: `OUT` offsets not hit by an active lane
are untouched. -/
private theorem swiglu_fused_preserves_unhitR (R : RoundingModel) (s' : BlockState)
    (o : Nat)
    (ho : ∀ i : Fin BLOCK_N, s.pids 0 * BLOCK_N + i.val < ncols →
      programLaneOffset s BLOCK_N i ≠ o)
    (hExec : execR R (swiglu_fused X Y OUT ncols BLOCK_N) s = some s') :
    s'.mem OUT o = s.mem OUT o := by
  simp [execR, swiglu_fused, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMemAsR_preserve_masked_prop R .bf16 _ _ _ o _
      (fun k _ hPk => ho k.1 hPk)]
  rfl

/-- Frame lemma for step B: every region other than `OUT` is untouched. -/
private theorem mul_step_preservesR (R : RoundingModel) (s' : BlockState)
    (r : RegionName) (hrOUT : r ≠ OUT)
    (hExec : execR R (mul_step S Y OUT ncols BLOCK_N) s = some s') :
    ∀ o : Nat, s'.mem r o = s.mem r o := by
  intro o
  simp [execR, mul_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMemAsR_preserve_other_region R .bf16 _ _ _ r hrOUT o]
  rfl

/-- Frame lemma for step B: `OUT` offsets not hit by an active lane are
untouched. -/
private theorem mul_step_preserves_unhitR (R : RoundingModel) (s' : BlockState)
    (o : Nat)
    (ho : ∀ i : Fin BLOCK_N, s.pids 0 * BLOCK_N + i.val < ncols →
      programLaneOffset s BLOCK_N i ≠ o)
    (hExec : execR R (mul_step S Y OUT ncols BLOCK_N) s = some s') :
    s'.mem OUT o = s.mem OUT o := by
  simp [execR, mul_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMemAsR_preserve_masked_prop R .bf16 _ _ _ o _
      (fun k _ hPk => ho k.1 hPk)]
  rfl

/-! ## The headline pair theorems -/

/-- **The event-ledger pair theorem** (on the pointwise surface
`ComputeRefine.RefinesAtR` — it genuinely needs per-side values, which the
writes-equality form cannot express): for every rounding model, the fused
kernel and the unfused pipeline realize their respective event-ledger terms
side by side on the shared output lanes. -/
theorem swiglu_refinesR
    (h_x : InputLoadedAt s X BLOCK_N xs) (h_y : InputLoadedAt s Y BLOCK_N ys)
    (h_SY : S ≠ Y) :
    ∀ R : RoundingModel,
      ComputeRefine.RefinesAtR R
        (swiglu_fused X Y OUT ncols BLOCK_N)
        (swiglu_unfused X Y S OUT ncols BLOCK_N) s
        (outWritesTo s OUT ncols BLOCK_N) (outWritesTo s OUT ncols BLOCK_N)
        (fun i lhs rhs =>
          lhs = floatCell .bf16 (fusedSpec xs ys R i) ∧
          rhs = floatCell .bf16 (unfusedSpec xs ys R i)) := by
  intro R
  apply ComputeKernel.computeRefineR_of_toAlgKernel
  · simp [swiglu_fused, ComputeExpr.toAlgorithm?]
  · simp [swiglu_unfused]
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro i
  have hFout := realizesR_out
    (swiglu_fused_realizesR X Y OUT ncols BLOCK_N s xs ys h_x h_y) R
    (by simp [swiglu_fused, ComputeExpr.toAlgorithm?]) hL
  have hUout := realizesR_out
    (swiglu_unfused_realizesR X Y S OUT ncols BLOCK_N s xs ys h_x h_y h_SY) R
    (by simp [swiglu_unfused]) hR
  by_cases hi : s.pids 0 * BLOCK_N + i.val < ncols
  · have hF := hFout i
    have hU := hUout i
    simp only [outWritesTo, ComputeCorrect.WriteMap.writeIf, BlockState.pid_eq, hi, if_true] at hF hU ⊢
    exact ⟨hF, hU⟩
  · simp only [outWritesTo, ComputeCorrect.WriteMap.writeIf, BlockState.pid_eq, hi, if_false]

/-- **"fused = unfused"** — the headline writes-equality theorem
(`ComputeRefine.RefinesR`, scratch `[S]`): when the intermediate `silu(x)` is
representable for the model, running the fused kernel and the unfused
pipeline from the same state performs THE SAME WRITES — the final memories
agree at every cell outside the scratch tensor `S`. -/
theorem swiglu_fused_eq_unfused_of_representable
    (h_x : InputLoadedAt s X BLOCK_N xs) (h_y : InputLoadedAt s Y BLOCK_N ys)
    (h_SY : S ≠ Y) :
    ∀ R : RoundingModel,
      (∀ i : Fin BLOCK_N, R.Representable .bf16 (TiledActivation.silu (xs i))) →
      ComputeRefine.RefinesR R
        (swiglu_fused X Y OUT ncols BLOCK_N)
        (swiglu_unfused X Y S OUT ncols BLOCK_N) s [S] := by
  intro R hRep
  apply ComputeKernel.computeRefineR_of_toAlgKernel
  · simp [swiglu_fused, ComputeExpr.toAlgorithm?]
  · simp [swiglu_unfused]
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  have hrS : r ≠ S := by simpa using hr
  -- split the pipeline execution into A ; B (keeping the original `hR`)
  have hRsplit := hR
  rw [exec_swiglu_unfusedR] at hRsplit
  cases hA : execR R (ComputeKernel.toAlgKernel (silu_step X S ncols BLOCK_N)) s with
  | none => rw [hA] at hRsplit; exact absurd hRsplit (by simp)
  | some s1 =>
      rw [hA] at hRsplit
      have hRB : execR R (ComputeKernel.toAlgKernel (mul_step S Y OUT ncols BLOCK_N)) s1
          = some rhs' := hRsplit
      have hpids : s1.pids = s.pids := silu_step_execR_pids X S ncols BLOCK_N s R s1 hA
      -- step A only touches `S`, and `r ≠ S`
      have hA_mem : ∀ o' : Nat, s1.mem r o' = s.mem r o' :=
        silu_step_preservesR X S ncols BLOCK_N s R s1 r hrS hA
      by_cases hrOUT : r = OUT
      · rw [hrOUT] at hA_mem ⊢
        by_cases hhit : ∃ i : Fin BLOCK_N,
            (s.pids 0 * BLOCK_N + i.val < ncols) ∧ o = programLaneOffset s BLOCK_N i
        · -- an active lane writes this cell on both sides; the written values
          -- coincide by the representability collapse of the two specs
          obtain ⟨i, hAct, rfl⟩ := hhit
          have hFout := realizesR_out
            (swiglu_fused_realizesR X Y OUT ncols BLOCK_N s xs ys h_x h_y) R
            (by simp [swiglu_fused, ComputeExpr.toAlgorithm?]) hL
          have hUout := realizesR_out
            (swiglu_unfused_realizesR X Y S OUT ncols BLOCK_N s xs ys h_x h_y h_SY) R
            (by simp [swiglu_unfused]) hR
          have hF := hFout i
          have hU := hUout i
          simp only [outWritesTo, ComputeCorrect.WriteMap.writeIf, BlockState.pid_eq, hAct, if_true,
            ComputeCorrect.OutputReadable.read_memcell] at hF hU
          simp only [programLaneOffset]
          rw [hF, hU, fusedSpec_eq_unfusedSpec_of_representable xs ys R i (hRep i)]
        · -- no active lane hits this `OUT` cell: preserved on both sides
          have hmiss : ∀ i : Fin BLOCK_N, s.pids 0 * BLOCK_N + i.val < ncols →
              programLaneOffset s BLOCK_N i ≠ o :=
            fun i hi heq => hhit ⟨i, hi, heq.symm⟩
          have hLmem : lhs'.mem OUT o = s.mem OUT o :=
            swiglu_fused_preserves_unhitR X Y OUT ncols BLOCK_N s R lhs' o hmiss hL
          have hBmem : rhs'.mem OUT o = s1.mem OUT o :=
            mul_step_preserves_unhitR Y S OUT ncols BLOCK_N s1 R rhs' o
              (fun i hi => by
                simp only [programLaneOffset, BlockState.pid_eq, hpids]
                exact hmiss i (by rwa [hpids] at hi)) hRB
          rw [hLmem, hBmem, hA_mem o]
      · -- `r ∉ {S, OUT}`: untouched by every store on both sides
        have hLmem : lhs'.mem r o = s.mem r o :=
          swiglu_fused_preservesR X Y OUT ncols BLOCK_N s R lhs' r hrOUT hL o
        have hBmem : rhs'.mem r o = s1.mem r o :=
          mul_step_preservesR Y S OUT ncols BLOCK_N s1 R rhs' r hrOUT hRB o
        rw [hLmem, hBmem, hA_mem o]

/-- Degeneration at the trivial model: the canonical two-kernel
`ComputeRefine.Refines` writes-equality statement — every rounding
collapses, and the fused kernel and the unfused pipeline perform the same
writes outside the scratch tensor `S`. -/
theorem swiglu_refines_classic
    (h_x : InputLoadedAt s X BLOCK_N xs) (h_y : InputLoadedAt s Y BLOCK_N ys)
    (h_SY : S ≠ Y) :
    ComputeRefine.Refines
      (swiglu_fused X Y OUT ncols BLOCK_N)
      (swiglu_unfused X Y S OUT ncols BLOCK_N) s [S] := by
  rw [← ComputeRefine.refinesR_triv_iff]
  exact swiglu_fused_eq_unfused_of_representable X Y S OUT ncols BLOCK_N s xs ys
    h_x h_y h_SY .triv (fun _ => rfl)

/-! ## Register-closedness of the kernels

The launch bridge (`ComputeKernel.execR_seq_rel_execPipelineR`) needs each
stage to be `RegClosed`: its execution must not depend on pre-existing
register bindings. All three kernels here are straight-line load/compute/
store programs, so their executions from two `MemPidsEq` states normalize to
the same masked scatter store over `MemPidsEq` base states. -/

private theorem silu_step_regClosed : (silu_step X S ncols BLOCK_N).RegClosed := by
  intro R s t h
  simp [execR, silu_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
    tile_elementwise, ComputeExpr.toAlgorithm?]
  simp only [h.readMem_eq, ← h.pids_eq]
  refine MemPidsEq.foldl_congr (fun a b hab k => ?_) _
    ((((h.setReg _ _ _ _).setReg _ _ _ _).setReg _ _ _ _).setReg _ _ _ _)
  dsimp only
  split_ifs with hP
  · exact hab.writeMemAsR _ _ _ _ _
  · exact hab

private theorem mul_step_regClosed : (mul_step S Y OUT ncols BLOCK_N).RegClosed := by
  intro R s t h
  simp [execR, mul_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
    tile_elementwise, ComputeExpr.toAlgorithm?]
  simp only [h.readMem_eq, h.readMemValue_eq, ← h.pids_eq, ← h.undef_eq]
  refine MemPidsEq.foldl_congr (fun a b hab k => ?_) _
    (((((h.setReg _ _ _ _).setReg _ _ _ _).setReg _ _ _ _).setReg _ _ _ _).setReg _ _ _ _)
  dsimp only
  split_ifs with hP
  · exact hab.writeMemAsR _ _ _ _ _
  · exact hab

private theorem swiglu_fused_regClosed : (swiglu_fused X Y OUT ncols BLOCK_N).RegClosed := by
  intro R s t h
  simp [execR, swiglu_fused, stepStmtsR, stepStmtR, evalOpR.eq_def,
    tile_elementwise, ComputeExpr.toAlgorithm?]
  simp only [h.readMem_eq, ← h.pids_eq]
  refine MemPidsEq.foldl_congr (fun a b hab k => ?_) _
    (((((h.setReg _ _ _ _).setReg _ _ _ _).setReg _ _ _ _).setReg _ _ _ _).setReg _ _ _ _)
  dsimp only
  split_ifs with hP
  · exact hab.writeMemAsR _ _ _ _ _
  · exact hab

/-! ## The honest-launch corollary -/

/-- Unpack a `RefinesR` statement at one successful execution pair: the raw
outside-scratch memory-agreement clause. -/
private theorem refinesR_mem {R : RoundingModel} {lhs rhs : ComputeKernel}
    {st : BlockState} {scratch : List RegionName}
    (h : ComputeRefine.RefinesR R lhs rhs st scratch)
    (hLAlg : lhs.toAlgorithm? = Except.ok lhs.toAlgKernel)
    (hRAlg : rhs.toAlgorithm? = Except.ok rhs.toAlgKernel)
    {l r : BlockState}
    (hEL : execR R lhs.toAlgKernel st = some l)
    (hER : execR R rhs.toAlgKernel st = some r) :
    ∀ reg ∉ scratch, ∀ o : Nat, l.mem reg o = r.mem reg o := by
  unfold ComputeRefine.RefinesR ComputeKernel.ExecRefineR
    ComputeKernel.ComputeRefineR ComputeKernel.ProjectedRefineR at h
  rw [hLAlg, hRAlg] at h
  exact h.2 st l r hEL hER rfl

/-- **The honest-launch corollary**: for every rounding model with the
intermediate `silu(x)` representable, the FUSED single launch and the HONEST
two-launch pipeline (`execPipelineR` — registers reset at every launch) end
with memories that agree at every cell outside the scratch tensor `S`.

Derived from the headline writes-equality theorem via the launch bridge
(`ComputeKernel.execR_seq_rel_execPipelineR`) and the three `RegClosed`
proofs — the pipeline is NOT reproven from scratch. -/
theorem swiglu_fused_launch_eq_pipeline_of_representable
    (h_x : InputLoadedAt s X BLOCK_N xs) (h_y : InputLoadedAt s Y BLOCK_N ys)
    (h_SY : S ≠ Y) (R : RoundingModel)
    (hRep : ∀ i : Fin BLOCK_N, R.Representable .bf16 (TiledActivation.silu (xs i)))
    {sF sP : BlockState}
    (hF : execPipelineR R [swiglu_fused X Y OUT ncols BLOCK_N] s = some sF)
    (hP : execPipelineR R [silu_step X S ncols BLOCK_N,
      mul_step S Y OUT ncols BLOCK_N] s = some sP) :
    ∀ r ∉ ([S] : List RegionName), ∀ o : Nat, sF.mem r o = sP.mem r o := by
  intro r hr o
  -- 1. The honest fused launch is one `execR` from the reset state.
  have hFexec : execR R (swiglu_fused X Y OUT ncols BLOCK_N).toAlgKernel s.resetRegs
      = some sF := by
    rw [execPipelineR_cons] at hF
    cases hE : execR R (swiglu_fused X Y OUT ncols BLOCK_N).toAlgKernel s.resetRegs with
    | none => rw [hE, Option.bind_none] at hF; exact absurd hF (by simp)
    | some w => rw [hE, Option.bind_some, execPipelineR_nil] at hF; exact hF
  -- 2. `RegClosed` aligns the fused reset launch with the no-reset execution.
  have hClF := swiglu_fused_regClosed X Y OUT ncols BLOCK_N R s.resetRegs s
    (memPidsEq_resetRegs s).symm
  rw [hFexec] at hClF
  cases hE0 : execR R (swiglu_fused X Y OUT ncols BLOCK_N).toAlgKernel s with
  | none => rw [hE0] at hClF; exact absurd hClF (memPidsEqO_some_none sF)
  | some sF0 =>
  rw [hE0] at hClF
  have hMPF : MemPidsEq sF sF0 := hClF
  -- 3. The launch bridge aligns the concatenated pipeline with the honest one.
  have hBr : MemPidsEqO
      (execR R (swiglu_unfused X Y S OUT ncols BLOCK_N).toAlgKernel s)
      (execPipelineR R [silu_step X S ncols BLOCK_N,
        mul_step S Y OUT ncols BLOCK_N] s) :=
    ComputeKernel.execR_seq_rel_execPipelineR R [X, Y] [OUT]
      (by
        intro k hk
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
        rcases hk with rfl | rfl
        · exact silu_step_regClosed X S ncols BLOCK_N
        · exact mul_step_regClosed Y S OUT ncols BLOCK_N) s
  rw [hP] at hBr
  cases hEU : execR R (swiglu_unfused X Y S OUT ncols BLOCK_N).toAlgKernel s with
  | none => rw [hEU] at hBr; exact absurd hBr (memPidsEqO_none_some sP)
  | some sU =>
  rw [hEU] at hBr
  have hMPU : MemPidsEq sU sP := hBr
  -- 4. The headline theorem relates the two no-reset executions outside `S`.
  have hHead := refinesR_mem
    (swiglu_fused_eq_unfused_of_representable X Y S OUT ncols BLOCK_N s xs ys
      h_x h_y h_SY R hRep)
    (by simp [swiglu_fused, ComputeExpr.toAlgorithm?])
    (by simp [swiglu_unfused])
    hE0 hEU r hr o
  -- 5. Chain the three memory agreements.
  rw [hMPF.mem_eq, ← hMPU.mem_eq]
  exact hHead

/-! ## Trust audit (compile-time gate)

These commands re-audit the public results every time the file is elaborated —
if any gate fails (a smuggled axiom / `sorry`, a foreign constant in a trusted
statement, or a self-referential spec) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in any result's transitive proof.
#axiomsClean swiglu_fused_eq_unfused_of_representable
#axiomsClean swiglu_refines_classic
#axiomsClean swiglu_fused_launch_eq_pipeline_of_representable
#axiomsClean swiglu_refinesR

-- (2) The headline is a *kernel-vs-kernel* writes-equality: its statement may
-- mention ONLY the kernels, the loaded-input contract, and the library
-- refinement surface — NO spec. If a spec ever leaks in, this fails.
#stmtSurfaceSubset swiglu_refines_classic ⊆
  [swiglu_fused, swiglu_unfused, InputLoadedAt, ComputeRefine.Refines,
   BlockState, RegionName]

-- (3) The specs the ledger theorem trusts are independent mathematics, not the
-- kernels' own execution (no circular "kernel does what the kernel does").
#specNonCircular fusedSpec avoiding [swiglu_fused, swiglu_unfused, silu_step, mul_step]
#specNonCircular unfusedSpec avoiding [swiglu_fused, swiglu_unfused, silu_step, mul_step]
#specNonCircular aSpec avoiding [swiglu_fused, swiglu_unfused, silu_step, mul_step]

end VeriTile.Bench.Examples.SwigluRounding
