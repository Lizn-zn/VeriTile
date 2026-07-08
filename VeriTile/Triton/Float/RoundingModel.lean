/-
VeriTile.Triton.Float.RoundingModel

Abstract rounding models (#447).

Floating-point casts and narrowing stores become *semantic events*: an
application of an opaque per-dtype rounding function `round : FloatDType →
ℝ → ℝ`. Nothing is assumed about `round` beyond identity on the `.real`
channel, so theorems parametric in a `RoundingModel` are *rounding-invariant*:
they pin down the structure of the rounding events (how many, where) while
saying nothing about their magnitude.

The trivial model (`RoundingModel.triv`, every `round` is the identity) is
exactly the current semantics; the degeneration lemmas in
`VeriTile.Triton.Float.EvalOpR` / `StepR` connect the parametric evaluators
back to `evalOp` / `stepStmt` at `triv`, so every existing
`ComputeCorrect.*` theorem is the `triv` shadow of a `ComputeRefine.*` one.

Useful properties of the rounding function (idempotence, monotonicity,
oddness, grid nesting) are opt-in mixin classes, cited per-theorem as
hypotheses — never global axioms. Both `triv` and real round-to-nearest-even
satisfy all of them, so adding a mixin hypothesis never vacuates a theorem
and never requires constructing a concrete rounding function.
-/

import VeriTile.Triton.Semantics.Scalar

namespace VeriTile.Triton

/-- An abstract rounding model: one opaque rounding function per floating
dtype, constrained only on the mathematical `.real` channel (which never
rounds). This is deliberately *not* a concrete IEEE model — see #447. -/
structure RoundingModel where
  /-- The rounding function for each floating dtype tag. -/
  round : FloatDType → ℝ → ℝ
  /-- The `.real` channel is the mathematical carrier: no rounding, ever. -/
  round_real : round .real = id
  /-- Idempotence: rounding an already-rounded value is a no-op. This is
  intrinsic to quantizing onto a discrete grid — the first cast lands on the
  grid, and re-casting a grid point is the identity — so *every* genuine
  rounding satisfies it (even stochastic rounding: grid points are fixed).
  A function without it is not a rounding, so it is a defining field of the
  model rather than an opt-in mixin like `MonoRounding`/`OddRounding`. -/
  round_idem : ∀ dtype x, round dtype (round dtype x) = round dtype x

namespace RoundingModel

/-- The trivial model: every cast/store is exact. This *is* the current
semantics — see the degeneration lemmas `evalOpR_triv` / `stepStmtR_triv`. -/
def triv : RoundingModel := ⟨fun _ => id, rfl, fun _ _ => rfl⟩

@[simp] theorem triv_round (dtype : FloatDType) : triv.round dtype = id := rfl

@[simp] theorem round_real_apply (R : RoundingModel) (x : ℝ) :
    R.round .real x = x := by
  rw [R.round_real, id]

/-- Rounding lifted to the `WithBot ℝ` carrier: `⊥` (undefined) passes
through untouched — rounding is a value event, not a definedness event. -/
def roundW (R : RoundingModel) (dtype : FloatDType) : WithBot ℝ → WithBot ℝ :=
  WithBot.map (R.round dtype)

@[simp] theorem roundW_bot (R : RoundingModel) (dtype : FloatDType) :
    R.roundW dtype ⊥ = ⊥ := rfl

@[simp] theorem roundW_coe (R : RoundingModel) (dtype : FloatDType) (x : ℝ) :
    R.roundW dtype (some x) = some (R.round dtype x) := rfl

@[simp] theorem triv_roundW (dtype : FloatDType) :
    triv.roundW dtype = id := by
  funext x
  cases x <;> rfl

@[simp] theorem roundW_real (R : RoundingModel) (x : WithBot ℝ) :
    R.roundW .real x = x := by
  cases x
  · rfl
  · exact congrArg some (R.round_real_apply _)

/-- `R`-aware float cast: the value crosses to the destination dtype and is
rounded *at the destination*. At `triv` this is `FloatDType.cast` (the
identity round-trip through `WithBot ℝ`). This is rounding-event site 1 of 2;
see `evalOpR` (`Op.castFloat`). -/
def cast (R : RoundingModel) (src dst : FloatDType)
    (x : TileCarrier src.toTileDType) : TileCarrier dst.toTileDType :=
  dst.ofWithBot (R.roundW dst (src.toWithBot x))

@[simp] theorem triv_cast (src dst : FloatDType) :
    triv.cast src dst = FloatDType.cast src dst := by
  funext x
  simp [cast, FloatDType.cast]

/-- `R`-aware store demotion: the stored value is quantized at the *buffer's*
dtype (after the `⊥ ↦ 0` finite fallback, matching `FloatDType.storeValue`).
This is rounding-event site 2 of 2; see `BlockState.writeMemAsR`. -/
def storeValue (R : RoundingModel) (dtype : FloatDType)
    (x : TileCarrier dtype.toTileDType) : ℝ :=
  R.round dtype (dtype.storeValue x)

@[simp] theorem triv_storeValue (dtype : FloatDType) :
    triv.storeValue dtype = dtype.storeValue := by
  funext x
  simp [storeValue]

@[simp] theorem storeValue_real (R : RoundingModel)
    (x : TileCarrier FloatDType.real.toTileDType) :
    R.storeValue .real x = FloatDType.real.storeValue x := by
  simp [storeValue]

/-- ℝ-level quantization to the `dtype` grid: the numerical core shared by the
typed cast (`RoundingModel.cast`) and the store demotion (`storeValue`). Both a
`.to(dtype)` cast and a store into a `dtype` buffer quantize a real to the
`dtype` grid, and on the `.real`-carried ℝ channel both are exactly this.
Specs written over ℝ use `castTo` in place of the two typed operations.
`@[reducible]` so lemmas about `round` (e.g. `round_idem`) fire through it. -/
@[reducible] def castTo (R : RoundingModel) (dtype : FloatDType) (x : ℝ) : ℝ :=
  R.round dtype x

end RoundingModel

/-! ## Opt-in property mixins

Idempotence is *not* here — it is a defining field of `RoundingModel`
(`round_idem`), because a non-idempotent function is not a rounding. The mixins
below are genuinely optional properties that some roundings lack (e.g.
stochastic rounding is not monotone). Cited per-theorem as instance hypotheses
(`[MonoRounding R]` …); the trivial model satisfies every mixin, so adding one
never vacuates a statement, and real round-to-nearest-even satisfies them too,
so statements remain true of actual hardware rounding. -/

/-- Rounding preserves order. Buys comparison/max/clamp stability. -/
class MonoRounding (R : RoundingModel) : Prop where
  mono : ∀ dtype, Monotone (R.round dtype)

/-- Rounding commutes with negation (hence fixes `0`). Together with
`MonoRounding` this makes e.g. ReLU commute with rounding. -/
class OddRounding (R : RoundingModel) : Prop where
  odd : ∀ dtype x, R.round dtype (-x) = -(R.round dtype x)

/-- The bf16/fp16 grids sit inside the fp32 grid: anything fixed by a
narrower rounding is fixed by the wider one. Buys upcast exactness
(`.to(tl.float32)` on a bf16 value is the identity). -/
class GridNested (R : RoundingModel) : Prop where
  nest_bf16 : ∀ x, R.round .bf16 x = x → R.round .fp32 x = x
  nest_fp16 : ∀ x, R.round .fp16 x = x → R.round .fp32 x = x

instance : MonoRounding RoundingModel.triv := ⟨fun _ => monotone_id⟩
instance : OddRounding RoundingModel.triv := ⟨fun _ _ => rfl⟩
instance : GridNested RoundingModel.triv := ⟨fun _ _ => rfl, fun _ _ => rfl⟩

theorem OddRounding.round_zero (R : RoundingModel) [OddRounding R]
    (dtype : FloatDType) : R.round dtype 0 = 0 := by
  have h := OddRounding.odd (R := R) dtype 0
  rw [neg_zero] at h
  linarith

end VeriTile.Triton
