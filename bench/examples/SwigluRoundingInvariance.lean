import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Activation

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
launches — and the final theorems live on the two-kernel refinement surface
`ComputeRefine.RefinesR`.

## The kernels

* `swiglu_fused` — one kernel: `silu(x)·y` on the ℝ register channel, one
  bf16 materialization at the output store.
* `silu_step` (**A**) — writes the intermediate `silu(x)` to the bf16 tensor
  `S` in memory.
* `mul_step` (**B**) — loads `S` back (a bf16-typed load), multiplies by `y`,
  stores bf16.
* `swiglu_unfused` — the pipeline `B ∘ A`: literally the concatenation of the
  two step kernels' bodies; `exec_swiglu_unfusedR` splits its execution into
  the sequential composition, so the pipeline theorem is *derived from* the
  two step theorems rather than proven monolithically.

## Theorems

Outputs are read at the `MemCell` layer (`bf16Cell v = MemCell.of .bf16
(some v)`): the stores are bf16-tagged and `readMem`'s real-channel decode is
tag-exact, so a narrow-float cell is only observable as its typed `MemCell` —
the classic fp16/bf16 convention (`rmsnorm_fused_llama`).

* `silu_step_realizesR` / `mul_step_realizesR` — the component theorems (A
  and B verified separately).
* `silu_step_preservesR` — A leaves every region other than `S` untouched
  (the aliasing frame the composition needs).
* `swiglu_unfused_realizesR` — the pipeline theorem, **derived by composing**
  the component theorems through the intermediate state.
* `swiglu_fused_realizesR` — the fused kernel's component theorem.
* **`swiglu_refinesR`** — the headline pair theorem on
  `ComputeRefine.RefinesR`: for every rounding model, fused and unfused
  realize their respective event-ledger terms side by side.
* **`swiglu_fused_eq_unfused_of_representable`** — "fused = unfused": when
  the intermediate `silu(x)` is representable, the relation strengthens to
  plain cell equality, for every rounding model.
* `swiglu_refines_classic` — degeneration at `triv`: the classic
  `ComputeRefine.Realizes` equality statement (the `FusedSiLU`-shaped
  theorem) recovered from the invariance theorem.
* Spec algebra: `fusedSpec_single_rounding`, `unfusedSpec_shape` (Idem
  collapses), `fusedSpec_ne_unfusedSpec` (a witness model distinguishing the
  pipelines — unstatable before #447).
-/

namespace VeriTile.Bench.Examples.SwigluRounding

open VeriTile.Triton

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

/-- The unfused pipeline `B ∘ A` as one kernel: the concatenation of the two
step bodies (what the eager framework's two launches execute in sequence). -/
def swiglu_unfused (X Y S OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel :=
  let body : List Stmt :=
    (silu_step X S ncols BLOCK_N).body ++
    (mul_step S Y OUT ncols BLOCK_N).body
  ComputeKernel.fromAlgBody [X, Y] [OUT] body

/-! ## Sequential decomposition of the pipeline -/

private theorem stepStmtsR_append (R : RoundingModel) (xs ys : List Stmt) (s : BlockState) :
    stepStmtsR R (xs ++ ys) s = (stepStmtsR R xs s).bind (fun s' => stepStmtsR R ys s') := by
  induction xs generalizing s with
  | nil => simp [stepStmtsR]
  | cons st rest ih =>
      simp [stepStmtsR]
      cases stepStmtR R st s <;> simp [ih]

/-- Execution of the composed pipeline under a rounding model, as the
sequential composition of the two step executions. -/
noncomputable def execUnfusedR (R : RoundingModel)
    (X Y S OUT : RegionName) (ncols BLOCK_N : Nat) (s : BlockState) :
    Option BlockState :=
  match execR R (silu_step X S ncols BLOCK_N) s with
  | none => none
  | some s1 => execR R (mul_step S Y OUT ncols BLOCK_N) s1

theorem exec_swiglu_unfusedR (R : RoundingModel)
    (X Y S OUT : RegionName) (ncols BLOCK_N : Nat) (s : BlockState) :
    execR R (swiglu_unfused X Y S OUT ncols BLOCK_N) s =
      execUnfusedR R X Y S OUT ncols BLOCK_N s := by
  simp [swiglu_unfused, execUnfusedR, execR, stepStmtsR_append]
  cases h1 : stepStmtsR R (silu_step X S ncols BLOCK_N).body s <;> simp

/-! ## Lane addressing, readback carrier, and the shared write map -/

/-- Lane address for this example's 1-D launch. -/
def laneOffset (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * BLOCK_N + i.val

/-- Each `R⟦v⟧` is ONE bf16 rounding event — count the brackets to count
the events. -/
local notation:max R "⟦" v "⟧" => RoundingModel.round R FloatDType.bf16 v

/-- `vals` is loaded at this program's lanes of `reg`. -/
def Loaded (s : BlockState) (reg : RegionName) (BLOCK_N : Nat)
    (vals : Fin BLOCK_N → ℝ) : Prop :=
  ∀ i : Fin BLOCK_N, s.readMem reg (laneOffset s BLOCK_N i) = vals i

/-- The output write map shared by every theorem: this program's active
lanes (`col < ncols`) of `reg`. -/
def outWrites (s : BlockState) (reg : RegionName) (ncols BLOCK_N : Nat) :
    ComputeCorrect.WriteMap (Fin BLOCK_N) :=
  ComputeCorrect.WriteMap.writeIf
    (fun i : Fin BLOCK_N => s.pids 0 * BLOCK_N + i.val < ncols)
    (fun i => (reg, laneOffset s BLOCK_N i))

/-- A bf16 output cell holding `v`. -/
def bf16Cell (v : ℝ) : MemCell := MemCell.of .bf16 (some v : WithBot ℝ)

/-- This program's active lanes of `reg` hold the bf16 cells `bf16Cell (zs i)`
(step B's intermediate-input hypothesis). -/
def LoadedCells (s : BlockState) (reg : RegionName) (ncols BLOCK_N : Nat)
    (zs : Fin BLOCK_N → ℝ) : Prop :=
  ∀ i : Fin BLOCK_N, s.pids 0 * BLOCK_N + i.val < ncols →
    s.mem reg (laneOffset s BLOCK_N i) = bf16Cell (zs i)

/-! ## Specs (the rounding-event ledger, term by term) -/

/-- Step A's per-lane payload: `silu(x)` through A's exit cast + bf16 store. -/
noncomputable def aSpec (R : RoundingModel) (x : ℝ) : ℝ :=
  R⟦R⟦TiledActivation.silu x⟧⟧

/-- Fused spec: exit cast + store — two events around the shared ℝ core. -/
noncomputable def fusedSpec (xs ys : Fin BLOCK_N → ℝ) (R : RoundingModel) (i : Fin BLOCK_N) : ℝ :=
  R⟦R⟦TiledActivation.swiglu (xs i) (ys i)⟧⟧

/-- Unfused (composed) spec: B's two events around `aSpec(x)·y` — four events
total, of which the inner two are A's. -/
noncomputable def unfusedSpec (xs ys : Fin BLOCK_N → ℝ) (R : RoundingModel) (i : Fin BLOCK_N) : ℝ :=
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

variable (X Y S OUT : RegionName) (ncols BLOCK_N : Nat) (s : BlockState)
variable (xs ys : Fin BLOCK_N → ℝ)

/-- The fused kernel realizes `fusedSpec` for **every** rounding model. -/
theorem swiglu_fused_realizesR (h_x : Loaded s X BLOCK_N xs) (h_y : Loaded s Y BLOCK_N ys) :
    ComputeRefine.RealizesR (swiglu_fused X Y OUT ncols BLOCK_N) s
      (outWrites s OUT ncols BLOCK_N)
      (fun R i => bf16Cell (fusedSpec xs ys R i)) := by
  unfold Loaded at h_x h_y
  unfold outWrites
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
  simp [hActive, bf16Cell, fusedSpec, TiledActivation.swiglu, TiledActivation.silu,
        tile_elementwise, hx, hy, RoundingModel.cast, RoundingModel.storeValue,
        FloatDType.storeValue, FloatDType.ofReal]

/-- Component theorem for step A: for every rounding model, the active lanes
of `S` receive `aSpec(x)` as bf16 cells. -/
theorem silu_step_realizesR (h_x : Loaded s X BLOCK_N xs) :
    ComputeRefine.RealizesR (silu_step X S ncols BLOCK_N) s
      (outWrites s S ncols BLOCK_N)
      (fun R i => bf16Cell (aSpec R (xs i))) := by
  unfold Loaded at h_x
  unfold outWrites
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
  simp only [ComputeCorrect.OutputReadable.read_memcell, laneOffset]
  rw [BlockState.scatter_memcell_R_prop_masked_nd R .bf16 _ _ _ _ h_inj (i, PUnit.unit)]
  have hx := h_x i
  simp only [laneOffset] at hx
  simp [hActive, bf16Cell, aSpec, TiledActivation.silu,
        tile_elementwise, hx, RoundingModel.cast, RoundingModel.storeValue,
        FloatDType.storeValue, FloatDType.ofReal]

/-- Frame lemma for step A: every region other than `S` is untouched, for
every rounding model. -/
theorem silu_step_preservesR (R : RoundingModel) (s' : BlockState)
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
theorem mul_step_realizesR (zs : Fin BLOCK_N → ℝ)
    (h_z : LoadedCells s S ncols BLOCK_N zs) (h_y : Loaded s Y BLOCK_N ys) :
    ComputeRefine.RealizesR (mul_step S Y OUT ncols BLOCK_N) s
      (outWrites s OUT ncols BLOCK_N)
      (fun R i => bf16Cell R⟦R⟦zs i * ys i⟧⟧) := by
  unfold LoadedCells bf16Cell at h_z
  unfold Loaded at h_y
  unfold outWrites
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
  simp only [ComputeCorrect.OutputReadable.read_memcell, laneOffset]
  rw [BlockState.scatter_memcell_R_prop_masked_nd R .bf16 _ _ _ _ h_inj (i, PUnit.unit)]
  have hz := h_z i hActive
  have hy := h_y i
  simp only [laneOffset] at hz hy
  simp [hActive, bf16Cell, BlockState.readMemValue_bf16_of_cell hz, hy,
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
theorem swiglu_unfused_realizesR (h_x : Loaded s X BLOCK_N xs) (h_y : Loaded s Y BLOCK_N ys)
    (h_SY : S ≠ Y) :
    ComputeRefine.RealizesR (swiglu_unfused X Y S OUT ncols BLOCK_N) s
      (outWrites s OUT ncols BLOCK_N)
      (fun R i => bf16Cell (unfusedSpec xs ys R i)) := by
  unfold outWrites
  rw [ComputeRefine.realizesR_writeIf_iff]
  intro R
  apply ComputeKernel.computeCorrectR_of_toAlgKernel
  · simp [swiglu_unfused]
  intro s0 s' hExec hs0
  subst s0
  rw [exec_swiglu_unfusedR] at hExec
  unfold execUnfusedR at hExec
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
      have h_z1 : LoadedCells s1 S ncols BLOCK_N (fun j => aSpec R (xs j)) := by
        intro j hj
        have hj' : s.pids 0 * BLOCK_N + j.val < ncols := by rwa [hpids] at hj
        have hout := hAout j
        simp only [outWrites, ComputeCorrect.WriteMap.writeIf, hj', if_true] at hout
        simpa [laneOffset, hpids] using hout
      -- `Y` survives step A (frame lemma; `S ≠ Y`)
      have h_y1 : Loaded s1 Y BLOCK_N ys := by
        intro j
        have hmem := silu_step_preservesR X S ncols BLOCK_N s R s1 Y (Ne.symm h_SY) hA
        have hread : s1.readMem Y (laneOffset s1 BLOCK_N j)
            = s.readMem Y (laneOffset s1 BLOCK_N j) := by
          unfold BlockState.readMem
          rw [hmem]
        show s1.readMem Y (laneOffset s1 BLOCK_N j) = ys j
        rw [hread]
        have hy := h_y j
        simp only [laneOffset, hpids] at hy ⊢
        exact hy
      -- step B's realization at the intermediate state
      have hBout := realizesR_out
        (mul_step_realizesR Y S OUT ncols BLOCK_N s1 ys (fun j => aSpec R (xs j)) h_z1 h_y1)
        R (by simp [mul_step, ComputeExpr.toAlgorithm?]) hExecB
      have hAct1 : s1.pids 0 * BLOCK_N + i.val < ncols := by rw [hpids]; exact hActive
      have hout := hBout i
      simp only [outWrites, ComputeCorrect.WriteMap.writeIf, hAct1, if_true] at hout
      simpa [laneOffset, hpids, unfusedSpec] using hout

/-! ## The headline pair theorems (`ComputeRefine.RefinesR`) -/

/-- **The pair theorem**: for every rounding model, the fused kernel and the
unfused pipeline realize their respective event-ledger terms side by side on
the shared output lanes. -/
theorem swiglu_refinesR (h_x : Loaded s X BLOCK_N xs) (h_y : Loaded s Y BLOCK_N ys)
    (h_SY : S ≠ Y) :
    ∀ R : RoundingModel,
      ComputeRefine.RefinesR R
        (swiglu_fused X Y OUT ncols BLOCK_N)
        (swiglu_unfused X Y S OUT ncols BLOCK_N) s
        (outWrites s OUT ncols BLOCK_N) (outWrites s OUT ncols BLOCK_N)
        (fun i lhs rhs =>
          lhs = bf16Cell (fusedSpec xs ys R i) ∧
          rhs = bf16Cell (unfusedSpec xs ys R i)) := by
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
    simp only [outWrites, ComputeCorrect.WriteMap.writeIf, hi, if_true] at hF hU ⊢
    exact ⟨hF, hU⟩
  · simp only [outWrites, ComputeCorrect.WriteMap.writeIf, hi, if_false]

/-- **"fused = unfused"**: when the intermediate `silu(x)` is representable
for the model, the pair relation strengthens to plain cell equality. -/
theorem swiglu_fused_eq_unfused_of_representable
    (h_x : Loaded s X BLOCK_N xs) (h_y : Loaded s Y BLOCK_N ys) (h_SY : S ≠ Y) :
    ∀ R : RoundingModel,
      (∀ i : Fin BLOCK_N, R.Representable .bf16 (TiledActivation.silu (xs i))) →
      ComputeRefine.RefinesR R
        (swiglu_fused X Y OUT ncols BLOCK_N)
        (swiglu_unfused X Y S OUT ncols BLOCK_N) s
        (outWrites s OUT ncols BLOCK_N) (outWrites s OUT ncols BLOCK_N)
        (fun _ (lhs rhs : MemCell) => lhs = rhs) := by
  intro R hRep
  refine ComputeRefine.RefinesR.mono
    (swiglu_refinesR X Y S OUT ncols BLOCK_N s xs ys h_x h_y h_SY R) ?_
  rintro i a b ⟨ha, hb⟩
  rw [ha, hb, fusedSpec_eq_unfusedSpec_of_representable xs ys R i (hRep i)]

/-- Degeneration at the trivial model: the classic two-kernel
`ComputeRefine.Realizes` equality statement (the `FusedSiLU`-shaped final
theorem) — every rounding collapses, both sides hold the ℝ closed form. -/
theorem swiglu_refines_classic
    (h_x : Loaded s X BLOCK_N xs) (h_y : Loaded s Y BLOCK_N ys) (h_SY : S ≠ Y) :
    ComputeRefine.Realizes
      (swiglu_fused X Y OUT ncols BLOCK_N)
      (swiglu_unfused X Y S OUT ncols BLOCK_N) s
      (outWrites s OUT ncols BLOCK_N) (outWrites s OUT ncols BLOCK_N)
      (fun _ (lhs rhs : MemCell) => lhs = rhs) := by
  rw [← ComputeRefine.refinesR_triv_iff]
  exact swiglu_fused_eq_unfused_of_representable X Y S OUT ncols BLOCK_N s xs ys
    h_x h_y h_SY .triv (fun _ => rfl)

end VeriTile.Bench.Examples.SwigluRounding
