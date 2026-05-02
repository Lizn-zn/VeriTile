/-
VeriTile.Triton.Semantics.Eval

Tile-level semantic operators, expression evaluation, statement stepping, and kernel execution.
-/

import Mathlib.Data.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import VeriTile.Triton.Semantics.State

namespace VeriTile.Triton

def Tile.bop {dtype a b out}
    (op : TileCarrier dtype → TileCarrier dtype → TileCarrier dtype)
    (bc : Broadcast a b out) (x : Tile dtype a) (y : Tile dtype b) :
    Tile dtype out :=
  ⟨fun i => op (x.data (bc.leftIndex i)) (y.data (bc.rightIndex i))⟩

def Tile.cop {dtype a b out}
    (op : TileCarrier dtype → TileCarrier dtype → Bool)
    (bc : Broadcast a b out) (x : Tile dtype a) (y : Tile dtype b) :
    Tile .bool out :=
  ⟨fun i => op (x.data (bc.leftIndex i)) (y.data (bc.rightIndex i))⟩

def Tile.ptrAdd {a b out}
    (bc : Broadcast a b out) (ptrs : Tile .ptr a) (offs : Tile .nat b) :
    Tile .ptr out :=
  ⟨fun i =>
    let p := ptrs.data (bc.leftIndex i)
    let o := offs.data (bc.rightIndex i)
    (p.1, p.2 + o)⟩

def Tile.uop {shape} (op : WithBot ℝ → WithBot ℝ) (x : Tile .real shape) : Tile .real shape :=
  ⟨fun i => op (x.data i)⟩

/-! ### Lifted unary functions on `WithBot ℝ`

`exp(-∞) = 0` and `sigmoid(-∞) = 0` are the IEEE-faithful values; this is what
makes the OnlineSoftmax `m`-seed flow correctly: with `m_0 = ⊥`, the first
iteration's `exp(m_0 - m_1) * l_0 = exp(⊥) * 0 = 0 * 0 = 0`, recovering the
correct base-case `l_1 = exp(x_0 - x_0) = 1`. -/

noncomputable def WithBot.realExp : WithBot ℝ → WithBot ℝ
  | none   => some 0           -- exp(-∞) = 0
  | some r => some (Real.exp r)

noncomputable def WithBot.realLog : WithBot ℝ → WithBot ℝ
  | none   => none
  | some r => some (Real.log r)

noncomputable def WithBot.realSigmoid : WithBot ℝ → WithBot ℝ
  | none   => some 0           -- sigmoid(-∞) = 0
  | some r => some (Real.sigmoid r)

noncomputable def WithBot.realSqrt : WithBot ℝ → WithBot ℝ
  | none   => none             -- sqrt(-∞) undefined
  | some r => some (Real.sqrt r)

noncomputable def WithBot.realTanh : WithBot ℝ → WithBot ℝ
  | none   => some (-1)        -- tanh(-∞) = -1
  | some r => some (Real.tanh r)

@[simp] theorem WithBot.realExp_some (r : ℝ) :
    WithBot.realExp (some r) = some (Real.exp r) := rfl
@[simp] theorem WithBot.realLog_some (r : ℝ) :
    WithBot.realLog (some r) = some (Real.log r) := rfl
@[simp] theorem WithBot.realSigmoid_some (r : ℝ) :
    WithBot.realSigmoid (some r) = some (Real.sigmoid r) := rfl
@[simp] theorem WithBot.realSqrt_some (r : ℝ) :
    WithBot.realSqrt (some r) = some (Real.sqrt r) := rfl
@[simp] theorem WithBot.realTanh_some (r : ℝ) :
    WithBot.realTanh (some r) = some (Real.tanh r) := rfl

@[simp] theorem WithBot.realExp_bot :
    WithBot.realExp (⊥ : WithBot ℝ) = ((0 : ℝ) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realSigmoid_bot :
    WithBot.realSigmoid (⊥ : WithBot ℝ) = ((0 : ℝ) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realTanh_bot :
    WithBot.realTanh (⊥ : WithBot ℝ) = ((-1 : ℝ) : WithBot ℝ) := rfl

@[simp] theorem WithBot.realExp_coe (r : ℝ) :
    WithBot.realExp ((r : ℝ) : WithBot ℝ) = (((Real.exp r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realLog_coe (r : ℝ) :
    WithBot.realLog ((r : ℝ) : WithBot ℝ) = (((Real.log r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realSigmoid_coe (r : ℝ) :
    WithBot.realSigmoid ((r : ℝ) : WithBot ℝ) = (((Real.sigmoid r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realSqrt_coe (r : ℝ) :
    WithBot.realSqrt ((r : ℝ) : WithBot ℝ) = (((Real.sqrt r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realTanh_coe (r : ℝ) :
    WithBot.realTanh ((r : ℝ) : WithBot ℝ) = (((Real.tanh r : ℝ)) : WithBot ℝ) := rfl

/-! ### Algebraic simp lemmas on `WithBot ℝ` arithmetic helpers -/

@[simp] theorem WithBot.realAdd_coe_coe (a b : ℝ) :
    WithBot.realAdd ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = (((a + b : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realSub_coe_coe (a b : ℝ) :
    WithBot.realSub ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = (((a - b : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realMul_coe_coe (a b : ℝ) :
    WithBot.realMul ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = (((a * b : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realDiv_coe_coe (a b : ℝ) :
    WithBot.realDiv ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = (((a / b : ℝ)) : WithBot ℝ) := rfl

/-- `realSub ⊥ x = ⊥` (and symmetrically). The corresponding `realAdd` and
`realMul` propagate `⊥` for the same reason — but only `Sub` is needed for the
OnlineSoftmax L proof at iter 0 (`exp(M_0 - M_1) = exp(⊥ - ↑x₀) = exp(⊥) = 0`). -/
@[simp] theorem WithBot.realSub_bot_left (x : WithBot ℝ) :
    WithBot.realSub (⊥ : WithBot ℝ) x = (⊥ : WithBot ℝ) := by
  cases x <;> rfl
@[simp] theorem WithBot.realSub_bot_right (x : WithBot ℝ) :
    WithBot.realSub x (⊥ : WithBot ℝ) = (⊥ : WithBot ℝ) := by
  cases x <;> rfl

@[simp] theorem WithBot.realMul_coe_zero (a : ℝ) :
    WithBot.realMul ((a : ℝ) : WithBot ℝ) (((0 : ℝ)) : WithBot ℝ)
      = (((0 : ℝ)) : WithBot ℝ) := by
  show ((a * 0 : ℝ) : WithBot ℝ) = _
  simp

/-- `↑a + ↑b = ↑(a + b)` for `WithBot ℝ`. Reverse of Mathlib's `WithBot.coe_add`.
NOT marked `@[simp]` to avoid loop with the forward direction; use as `←` rewrite
where consolidating coe outward is desired. -/
theorem WithBot.coe_add_coe_eq (a b : ℝ) :
    ((a : ℝ) : WithBot ℝ) + ((b : ℝ) : WithBot ℝ) = (((a + b : ℝ)) : WithBot ℝ) :=
  (WithBot.coe_add a b).symm

/-- `Option.map f ↑x = ↑(f x)` for `WithBot ℝ`. -/
@[simp] theorem WithBot.coe_map_coe_eq {α} (f : ℝ → α) (a : ℝ) :
    Option.map f ((a : ℝ) : WithBot ℝ) = some (f a) := rfl

/-- Reduce-sum with `keep_dims = false` along an arbitrary Triton axis. -/
noncomputable def Tile.reduceSumDrop {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile .real (TileShape.eraseAxis shape axis) :=
  ⟨fun outIdx =>
    @Finset.sum (Fin (TileShape.axisDim shape axis)) (WithBot ℝ) _ Finset.univ
      (fun k => x.data (TileShape.insertAxisIndex shape axis outIdx k))⟩

/-- Reduce-sum with `keep_dims = true` along an arbitrary Triton axis. -/
noncomputable def Tile.reduceSumKeep {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile .real (TileShape.setAxisOne shape axis) :=
  ⟨fun outIdx =>
    @Finset.sum (Fin (TileShape.axisDim shape axis)) (WithBot ℝ) _ Finset.univ
      (fun k => x.data (TileShape.replaceAxisIndex shape axis outIdx k))⟩

/-- Reduce-sum along an arbitrary Triton axis. -/
noncomputable def Tile.reduceSum {shape : TileShape}
    (axis : Fin shape.length) (keepDims : Bool) (x : Tile .real shape) :
    Tile .real (TileShape.reduceShape shape axis keepDims) := by
  cases keepDims
  · exact Tile.reduceSumDrop axis x
  · exact Tile.reduceSumKeep axis x

/-! ### `simp` lemmas for the structural cases of reduceSum -/

@[simp] theorem Tile.reduceSum_false {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile.reduceSum axis false x = Tile.reduceSumDrop axis x := rfl

@[simp] theorem Tile.reduceSum_true {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile.reduceSum axis true x = Tile.reduceSumKeep axis x := rfl

/-- Body of `reduceSumDrop` at an output index. Lets `simp` push past the
opaque `Tile` constructor so kernel proofs can reach the `Finset.sum` form. -/
@[simp] theorem Tile.reduceSumDrop_data {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape)
    (outIdx : TileIndex (TileShape.eraseAxis shape axis)) :
    (Tile.reduceSumDrop axis x).data outIdx =
      @Finset.sum (Fin (TileShape.axisDim shape axis)) (WithBot ℝ) _ Finset.univ
        (fun k => x.data (TileShape.insertAxisIndex shape axis outIdx k)) := rfl

@[simp] theorem Tile.reduceSumKeep_data {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape)
    (outIdx : TileIndex (TileShape.setAxisOne shape axis)) :
    (Tile.reduceSumKeep axis x).data outIdx =
      @Finset.sum (Fin (TileShape.axisDim shape axis)) (WithBot ℝ) _ Finset.univ
        (fun k => x.data (TileShape.replaceAxisIndex shape axis outIdx k)) := rfl

/-! ### `simp` helpers for the `WithBot ℝ` boundary

A typical kernel proof reduces `tl.sum(x)` to `∑ i, (some (xs i) : WithBot ℝ)`,
then `tl.store` demotes via `Option.getD … 0`. Mathlib's `WithBot.coe_sum`
gives us `↑(∑ f i) = ∑ ↑(f i)` (for `WithBot.coe = some`); read backwards, it
collapses the sum-of-`some`s to a `some` of the sum, which then evaluates the
`getD`. -/

/-- `WithBot.unbotD` on a `some _` projects out the value. Same as
`WithBot.unbotD_coe` but stated in the `some`-form `evalOp` produces. -/
@[simp] theorem WithBot.unbotD_some {α} (d a : α) :
    WithBot.unbotD d (some a : WithBot α) = a := rfl

/-- `Option.map₂` on two `some`-lifted values produces a `some`-lifted result.
Required because `Tile.bop` for `.real` arithmetic uses `Option.map₂`. -/
@[simp] theorem Option.map₂_some_some (f : ℝ → ℝ → ℝ) (a b : ℝ) :
    Option.map₂ f (some a : WithBot ℝ) (some b : WithBot ℝ)
      = (some (f a b) : WithBot ℝ) := rfl

@[simp] theorem Option.map_some_real (f : ℝ → ℝ) (a : ℝ) :
    Option.map f (some a : WithBot ℝ) = (some (f a) : WithBot ℝ) := rfl

/-- `↑`-form: when proofs land on the coe view of `WithBot ℝ`, the same
reductions apply. Simp matches syntactically, so we need both forms. -/
@[simp] theorem Option.map₂_coe_coe (f : ℝ → ℝ → ℝ) (a b : ℝ) :
    Option.map₂ f ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = ((f a b : ℝ) : WithBot ℝ) := rfl

@[simp] theorem Option.map_coe_real (f : ℝ → ℝ) (a : ℝ) :
    Option.map f ((a : ℝ) : WithBot ℝ) = ((f a : ℝ) : WithBot ℝ) := rfl

/-- Mixed-form: `some` on left, `↑` on right (and vice versa). Both forms appear
in goals after partial simp normalization. -/
@[simp] theorem Option.map₂_some_coe (f : ℝ → ℝ → ℝ) (a b : ℝ) :
    Option.map₂ f (some a : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = (some (f a b) : WithBot ℝ) := rfl

@[simp] theorem Option.map₂_coe_some (f : ℝ → ℝ → ℝ) (a b : ℝ) :
    Option.map₂ f ((a : ℝ) : WithBot ℝ) (some b : WithBot ℝ)
      = (some (f a b) : WithBot ℝ) := rfl


@[simp] theorem WithBot.sum_some_eq_some {ι} (s : Finset ι) (f : ι → ℝ) :
    (∑ i ∈ s, ((f i : ℝ) : WithBot ℝ)) = (((∑ i ∈ s, f i) : ℝ) : WithBot ℝ) :=
  (WithBot.coe_sum s f).symm

/-- `some`-form companion to `WithBot.sum_some_eq_some`. The explicit
`AddCommMonoid (WithBot ℝ)` ascription via `@Finset.sum` is what makes the
typeclass resolution fire — bare `some _ : WithBot ℝ` ascription elaborates
to `Option ℝ` for typeclass purposes. -/
@[simp] theorem WithBot.sum_someTerm_eq_some {ι} (s : Finset ι) (f : ι → ℝ) :
    @Finset.sum ι (WithBot ℝ) _ s (fun i => (some (f i) : WithBot ℝ))
      = (some (∑ i ∈ s, f i) : WithBot ℝ) := by
  show @Finset.sum ι (WithBot ℝ) _ s (fun i => ((f i : ℝ) : WithBot ℝ)) = _
  rw [WithBot.sum_some_eq_some]
  rfl


/-- After a `tl.sum` reduce on a `some`-lifted tile, demoting via `unbotD 0`
(used by `tl.store`) recovers the underlying ℝ-valued sum.

Stated in the `some`-form (rather than `↑`) because that's what `evalOp`
produces — `tl.load` lifts via `some (s.readMem ...)` and `tl.sum` on a tile
of `some`-values gives `∑ some (xs i)` literally. -/
@[simp] theorem WithBot.unbotD_sum_some {ι} (s : Finset ι) (f : ι → ℝ) :
    WithBot.unbotD (0 : ℝ) (∑ i ∈ s, (some (f i) : WithBot ℝ)) = ∑ i ∈ s, f i := by
  show WithBot.unbotD (0 : ℝ) (∑ i ∈ s, ((f i : ℝ) : WithBot ℝ)) = ∑ i ∈ s, f i
  rw [WithBot.sum_some_eq_some]
  rfl

/-- `sup'` over `↑`-lifted values equals `↑` of the `sup'`. -/
theorem WithBot.sup'_coe {ι} [LinearOrder ι] (s : Finset ι) (h : s.Nonempty)
    (f : ι → ℝ) :
    s.sup' h (fun i => ((f i : ℝ) : WithBot ℝ)) = ((s.sup' h f : ℝ) : WithBot ℝ) := by
  refine h.cons_induction ?_ ?_
  · intro a; rfl
  · intro a s' ha hne ih
    rw [Finset.sup'_cons hne, Finset.sup'_cons hne, ih]
    rfl

/-- `some`-form of `WithBot.sup'_coe`. Required because `evalOp` produces
literal `some _` calls, not `↑`. The `@Finset.sup' (WithBot ℝ) ι` ascription
forces typeclass resolution to use `SemilatticeSup (WithBot ℝ)` rather than
the elaborated `Option ℝ`. -/
@[simp] theorem WithBot.sup'_someTerm_eq_some {ι} [LinearOrder ι]
    (s : Finset ι) (h : s.Nonempty) (f : ι → ℝ) :
    @Finset.sup' (WithBot ℝ) ι _ s h (fun i => (some (f i) : WithBot ℝ))
      = (some (s.sup' h f) : WithBot ℝ) := by
  show @Finset.sup' (WithBot ℝ) ι _ s h (fun i => ((f i : ℝ) : WithBot ℝ)) = _
  rw [WithBot.sup'_coe]; rfl

/-- Companion to `WithBot.sup'_coe`: store-side demote on a max reduce. -/
@[simp] theorem WithBot.unbotD_sup'_some {ι} [LinearOrder ι] (s : Finset ι)
    (h : s.Nonempty) (f : ι → ℝ) :
    WithBot.unbotD (0 : ℝ) (s.sup' h (fun i => (some (f i) : WithBot ℝ)))
      = s.sup' h f := by
  show WithBot.unbotD (0 : ℝ) (s.sup' h (fun i => ((f i : ℝ) : WithBot ℝ))) = _
  rw [WithBot.sup'_coe]; rfl

/-! ### Reduce-max along an arbitrary axis

Returns `none` only when the reduced dimension is `0` (empty `sup'`
undefined). -/

noncomputable def Tile.reduceMaxDrop {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Option (Tile .real (TileShape.eraseAxis shape axis)) :=
  if h : 0 < TileShape.axisDim shape axis then
    some ⟨fun outIdx =>
      (Finset.univ : Finset (Fin (TileShape.axisDim shape axis))).sup'
        (by exact ⟨⟨0, h⟩, Finset.mem_univ _⟩)
        (fun k => x.data (TileShape.insertAxisIndex shape axis outIdx k))⟩
  else
    none

noncomputable def Tile.reduceMaxKeep {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Option (Tile .real (TileShape.setAxisOne shape axis)) :=
  if h : 0 < TileShape.axisDim shape axis then
    some ⟨fun outIdx =>
      (Finset.univ : Finset (Fin (TileShape.axisDim shape axis))).sup'
        (by exact ⟨⟨0, h⟩, Finset.mem_univ _⟩)
        (fun k => x.data (TileShape.replaceAxisIndex shape axis outIdx k))⟩
  else
    none

noncomputable def Tile.reduceMax {shape : TileShape}
    (axis : Fin shape.length) (keepDims : Bool) (x : Tile .real shape) :
    Option (Tile .real (TileShape.reduceShape shape axis keepDims)) := by
  cases keepDims
  · exact Tile.reduceMaxDrop axis x
  · exact Tile.reduceMaxKeep axis x

/-! ### `simp` lemmas for the structural cases of reduceMax -/

@[simp] theorem Tile.reduceMax_false {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile.reduceMax axis false x = Tile.reduceMaxDrop axis x := rfl

@[simp] theorem Tile.reduceMax_true {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile.reduceMax axis true x = Tile.reduceMaxKeep axis x := rfl

def Tile.natToReal {shape} (x : Tile .nat shape) : Tile .real shape :=
  ⟨fun i => some ((x.data i : ℝ))⟩

/-- Block-level (possibly batched) matrix multiply (Triton's `tl.dot`):
`c[…, m, n] = ∑_k a[…, m, k] * b[…, k, n]`.

Operates on the trailing two dims; any batch prefix `batch : TileShape` is
broadcast pointwise. Structural recursion on the batch fixes one outer
index per level and recurses on the remaining batch.

`⊥`-propagating: if any factor in any pair is `⊥` the corresponding
summand is `⊥`, and `Finset.sum` over `WithBot ℝ` propagates `⊥` through
the whole entry. Well-formed kernels load real (non-`⊥`) values into the
operands, so the dot product collapses to a regular `↑(∑ a*b)` term and
the `tl.store` demotion via `unbotD 0` recovers the underlying ℝ value. -/
noncomputable def Tile.dot : (batch : TileShape) → {M K N : Nat} →
    Tile .real (batch ++ [M, K]) → Tile .real (batch ++ [K, N]) →
    Tile .real (batch ++ [M, N])
  | [], _, K, _, a, b =>
      ⟨fun (m, n, _) =>
        @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
          (fun k => Option.map₂ (· * ·)
                      (a.data (m, k, PUnit.unit))
                      (b.data (k, n, PUnit.unit)))⟩
  | _ :: rest, M, K, N, a, b =>
      ⟨fun (i, restIdx) =>
        (Tile.dot rest (M := M) (K := K) (N := N)
            ⟨fun rIdx => a.data (i, rIdx)⟩
            ⟨fun rIdx => b.data (i, rIdx)⟩).data restIdx⟩

/-- Body of `Tile.dot` at the rank-2 base case (`batch = []`); lets `simp`
expose the `Finset.sum` form for kernel proofs. -/
@[simp] theorem Tile.dot_nil_data {M K N : Nat}
    (a : Tile .real [M, K]) (b : Tile .real [K, N])
    (m : Fin M) (n : Fin N) (rest : TileIndex []) :
    (Tile.dot [] a b).data (m, n, rest) =
      @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·)
                    (a.data (m, k, PUnit.unit))
                    (b.data (k, n, PUnit.unit))) := rfl

/-- Recursive step: `(Tile.dot (b :: rest) a b').data (i, restIdx)` slices
each operand at outer index `i` and recurses on the remaining batch. -/
@[simp] theorem Tile.dot_cons_data {b : Nat} {rest : TileShape} {M K N : Nat}
    (a : Tile .real ((b :: rest) ++ [M, K]))
    (b' : Tile .real ((b :: rest) ++ [K, N]))
    (i : Fin b) (restIdx : TileIndex (rest ++ [M, N])) :
    (Tile.dot (b :: rest) a b').data (i, restIdx) =
      (Tile.dot rest (M := M) (K := K) (N := N)
          ⟨fun rIdx => a.data (i, rIdx)⟩
          ⟨fun rIdx => b'.data (i, rIdx)⟩).data restIdx := rfl

/-- Trailing-two-axes transpose (`Tile.dot`-style framework: matrix dims
are the trailing two, leading dims are an unchanged `batch` prefix). -/
def Tile.transpose : {dtype : TileDType} → (batch : TileShape) →
    {M N : Nat} →
    Tile dtype (batch ++ [M, N]) → Tile dtype (batch ++ [N, M])
  | _, [],         _, _, x =>
      ⟨fun (n, m, _) => x.data (m, n, PUnit.unit)⟩
  | _, _ :: rest,  _, _, x =>
      ⟨fun (i, restIdx) =>
        (Tile.transpose rest
            ⟨fun rIdx => x.data (i, rIdx)⟩).data restIdx⟩

/-- Body of `Tile.transpose` at the rank-2 base case: swap `(m, n) ↔ (n, m)`. -/
@[simp] theorem Tile.transpose_nil_data {dtype : TileDType} {M N : Nat}
    (x : Tile dtype [M, N]) (n : Fin N) (m : Fin M) (rest : TileIndex []) :
    (Tile.transpose [] x).data (n, m, rest) =
      x.data (m, n, PUnit.unit) := rfl

/-- Recursive step: `(Tile.transpose (b :: rest) x).data (i, restIdx)`
slices at outer index `i` and recurses on the remaining batch. -/
@[simp] theorem Tile.transpose_cons_data {dtype : TileDType}
    {b : Nat} {rest : TileShape} {M N : Nat}
    (x : Tile dtype ((b :: rest) ++ [M, N]))
    (i : Fin b) (restIdx : TileIndex (rest ++ [N, M])) :
    (Tile.transpose (b :: rest) x).data (i, restIdx) =
      (Tile.transpose rest (M := M) (N := N)
          ⟨fun rIdx => x.data (i, rIdx)⟩).data restIdx := rfl

/-- Insert a unit-size axis at position `axis`. The output index is
projected back to the input by `dropInsertedIndex`, which throws away
the inserted slot's `Fin 1` coordinate. -/
def Tile.expandDim {dtype : TileDType} {shape : TileShape}
    (axis : Fin (shape.length + 1)) (x : Tile dtype shape) :
    Tile dtype (TileShape.insertAxis shape axis 1) :=
  ⟨fun idx => x.data (TileShape.dropInsertedIndex shape axis 1 idx)⟩

@[simp] theorem Tile.expandDim_data {dtype : TileDType} {shape : TileShape}
    (axis : Fin (shape.length + 1)) (x : Tile dtype shape)
    (idx : TileIndex (TileShape.insertAxis shape axis 1)) :
    (Tile.expandDim axis x).data idx =
      x.data (TileShape.dropInsertedIndex shape axis 1 idx) := rfl

/-- Lift a plain ℝ-valued tile-shaped function into a `Tile .real`. Useful
at the spec / boundary layer: `BlockState.mem` reads ℝ, never `⊥`, so a
`tl.load`-fed kernel input is naturally a `TileIndex shape → ℝ`. The
`⊥` sentinel of `WithBot ℝ` is reserved for `-inf` / masked-off /
`tl.full(_, -inf)` values introduced *inside* a kernel; spec-level
inputs and outputs should round-trip through `ofReal` / `unbotD 0`. -/
def Tile.ofReal {shape : TileShape} (x : TileIndex shape → ℝ) :
    Tile .real shape :=
  ⟨fun i => some (x i)⟩

@[simp] theorem Tile.ofReal_data {shape : TileShape}
    (x : TileIndex shape → ℝ) (i : TileIndex shape) :
    (Tile.ofReal x).data i = some (x i) := rfl

/-- Element-wise select (`tl.where(cond, a, b)`): per-cell, pick from `a`
when `cond` is `true`, else from `b`. Same-shape; broadcast lifting is
done at the DSL layer. Named `select` to avoid the Lean `where` keyword
in definition position; the AST constructor `Op.where` and DSL surface
`tl.where(...)` keep the user-facing Triton spelling. -/
def Tile.select {dtype : TileDType} {shape : TileShape}
    (c : Tile .bool shape) (a b : Tile dtype shape) :
    Tile dtype shape :=
  ⟨fun idx => if c.data idx then a.data idx else b.data idx⟩

@[simp] theorem Tile.select_data {dtype : TileDType} {shape : TileShape}
    (c : Tile .bool shape) (a b : Tile dtype shape) (idx : TileIndex shape) :
    (Tile.select c a b).data idx = if c.data idx then a.data idx else b.data idx := rfl

noncomputable def evalOp : Op dtype shape → BlockState → Option (Tile dtype shape)
  | .const c, _ => some (Tile.scalar (some c : WithBot ℝ))
  | .constFloat h c, _ => some (Tile.scalar (h.ofReal c))
  | .constNat n, _ => some (Tile.scalar n)
  | .constBool b, _ => some (Tile.scalar b)
  | .negInf, _ => some (Tile.scalar (none : WithBot ℝ))
  | .programId axis, s => some (Tile.scalar (s.pids axis))
  | .ref dtype shape name, s => s.regs dtype shape name
  | .arange n, _ => some (Tile.vec (fun i => i.val))
  | .broadcast e shape, s => do
      let v ← evalOp e s
      some ⟨fun _ => v.data PUnit.unit⟩
  | .full shape e, s => do
      let v ← evalOp e s
      some ⟨fun _ => v.data PUnit.unit⟩
  | .castFloat src dst a, s => do
      let va ← evalOp a s
      some ⟨fun i => src.cast dst (va.data i)⟩
  | .add h bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop h.add bc va vb)
  | .sub h bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop h.sub bc va vb)
  | .mul h bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop h.mul bc va vb)
  | .div h bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop h.div bc va vb)
  | .exp a, s => return Tile.uop WithBot.realExp (← evalOp a s)
  | .log a, s => return Tile.uop WithBot.realLog (← evalOp a s)
  | .sigmoid a, s => return Tile.uop WithBot.realSigmoid (← evalOp a s)
  | .sqrt a, s => return Tile.uop WithBot.realSqrt (← evalOp a s)
  | .tanh a, s => return Tile.uop WithBot.realTanh (← evalOp a s)
  | .lt h bc a b, s => return Tile.cop h.lt bc (← evalOp a s) (← evalOp b s)
  | .le h bc a b, s => return Tile.cop h.le bc (← evalOp a s) (← evalOp b s)
  | .eq h bc a b, s => return Tile.cop h.eq bc (← evalOp a s) (← evalOp b s)
  | .gt h bc a b, s => return Tile.cop h.gt bc (← evalOp a s) (← evalOp b s)
  | .ge h bc a b, s => return Tile.cop h.ge bc (← evalOp a s) (← evalOp b s)
  | .ne h bc a b, s => return Tile.cop h.ne bc (← evalOp a s) (← evalOp b s)
  | .boolAnd bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop (fun x y : Bool => x && y) bc va vb)
  | .max2 bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop max bc va vb)
  | .reduceMax axis keepDims a, s => do
      let va ← evalOp a s
      Tile.reduceMax axis keepDims va
  | .reduceSum axis keepDims a, s => return Tile.reduceSum axis keepDims (← evalOp a s)
  | .dot (batch := batch) a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.dot batch va vb)
  | .expandDim axis a, s => return Tile.expandDim axis (← evalOp a s)
  | .where c a b, s => do
      let vc ← evalOp c s
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.select vc va vb)
  | .transpose (batch := batch) a, s => do
      let va ← evalOp a s
      some (Tile.transpose batch va)
  | .ptrBase region, _ => some (Tile.scalar (region, 0))
  | .ptrAdd bc ptr off, s => do
      let ptrs ← evalOp ptr s
      let offs ← evalOp off s
      some (Tile.ptrAdd bc ptrs offs)
  | .load region off, s => do
      let offsets ← evalOp off s
      some ⟨fun i => some (s.readMem region (offsets.data i))⟩
  | .loadMask region off mask, s => do
      let offsets ← evalOp off s
      let masks ← evalOp mask s
      some ⟨fun i =>
        let addr := offsets.data i
        if masks.data i then some (s.readMem region addr) else some (s.undef region addr)⟩
  | .loadMaskOther region off mask other, s => do
      let offsets ← evalOp off s
      let masks ← evalOp mask s
      let others ← evalOp other s
      some ⟨fun i =>
        let addr := offsets.data i
        if masks.data i then some (s.readMem region addr) else others.data i⟩
  | .loadPtr ptr, s => do
      let ptrs ← evalOp ptr s
      some ⟨fun i =>
        let p := ptrs.data i
        some (s.readMem p.1 p.2)⟩
  | .loadPtrMask ptr mask, s => do
      let ptrs ← evalOp ptr s
      let masks ← evalOp mask s
      some ⟨fun i =>
        let p := ptrs.data i
        if masks.data i then some (s.readMem p.1 p.2) else some (s.undef p.1 p.2)⟩
  | .loadPtrMaskOther ptr mask other, s => do
      let ptrs ← evalOp ptr s
      let masks ← evalOp mask s
      let others ← evalOp other s
      some ⟨fun i =>
        let p := ptrs.data i
        if masks.data i then some (s.readMem p.1 p.2) else others.data i⟩
  | .loadFloat h region off, s => do
      let offsets ← evalOp off s
      some ⟨fun i => h.ofReal (s.readMem region (offsets.data i))⟩
  | .loadFloatMask h region off mask, s => do
      let offsets ← evalOp off s
      let masks ← evalOp mask s
      some ⟨fun i =>
        let addr := offsets.data i
        if masks.data i then h.ofReal (s.readMem region addr) else h.ofReal (s.undef region addr)⟩
  | .loadFloatMaskOther h region off mask other, s => do
      let offsets ← evalOp off s
      let masks ← evalOp mask s
      let others ← evalOp other s
      some ⟨fun i =>
        let addr := offsets.data i
        if masks.data i then h.ofReal (s.readMem region addr) else others.data i⟩
  | .loadPtrFloat h ptr, s => do
      let ptrs ← evalOp ptr s
      some ⟨fun i =>
        let p := ptrs.data i
        h.ofReal (s.readMem p.1 p.2)⟩
  | .loadPtrFloatMask h ptr mask, s => do
      let ptrs ← evalOp ptr s
      let masks ← evalOp mask s
      some ⟨fun i =>
        let p := ptrs.data i
        if masks.data i then h.ofReal (s.readMem p.1 p.2) else h.ofReal (s.undef p.1 p.2)⟩
  | .loadPtrFloatMaskOther h ptr mask other, s => do
      let ptrs ← evalOp ptr s
      let masks ← evalOp mask s
      let others ← evalOp other s
      some ⟨fun i =>
        let p := ptrs.data i
        if masks.data i then h.ofReal (s.readMem p.1 p.2) else others.data i⟩
  | .natToReal a, s => return Tile.natToReal (← evalOp a s)

mutual

noncomputable def stepStmt : Stmt → BlockState → Option BlockState
  | .assign dtype shape name e, s => do
      let v ← evalOp e s
      some (s.setReg name dtype shape v)
  | .store region shape off val, s => do
      let offsets ← evalOp off s
      let values ← evalOp val s
      -- `mem : RegionName → Nat → ℝ`; demote `WithBot ℝ → ℝ` via `unbotD 0`.
      -- Well-formed kernels never store `⊥`, so the default value is unobservable.
      some ((TileShape.allIndices shape).foldl
        (fun acc i => acc.writeMem region (offsets.data i)
                        ((values.data i).unbotD 0)) s)
  | .storeMask region shape off val mask, s => do
      let offsets ← evalOp off s
      let values ← evalOp val s
      let masks ← evalOp mask s
      some ((TileShape.allIndices shape).foldl
        (fun acc i =>
          if masks.data i then acc.writeMem region (offsets.data i)
                                ((values.data i).unbotD 0)
          else acc) s)
  | .storePtr shape ptr val, s => do
      let ptrs ← evalOp ptr s
      let values ← evalOp val s
      some ((TileShape.allIndices shape).foldl
        (fun acc i =>
          let p := ptrs.data i
          acc.writeMem p.1 p.2 ((values.data i).unbotD 0)) s)
  | .storePtrMask shape ptr val mask, s => do
      let ptrs ← evalOp ptr s
      let values ← evalOp val s
      let masks ← evalOp mask s
      some ((TileShape.allIndices shape).foldl
        (fun acc i =>
          let p := ptrs.data i
          if masks.data i then acc.writeMem p.1 p.2 ((values.data i).unbotD 0)
          else acc) s)
  | .storeFloat h region shape off val, s => do
      let offsets ← evalOp off s
      let values ← evalOp val s
      some ((TileShape.allIndices shape).foldl
        (fun acc i => acc.writeMem region (offsets.data i)
                        (h.storeValue (values.data i))) s)
  | .storeFloatMask h region shape off val mask, s => do
      let offsets ← evalOp off s
      let values ← evalOp val s
      let masks ← evalOp mask s
      some ((TileShape.allIndices shape).foldl
        (fun acc i =>
          if masks.data i then acc.writeMem region (offsets.data i)
                                (h.storeValue (values.data i))
          else acc) s)
  | .storePtrFloat h shape ptr val, s => do
      let ptrs ← evalOp ptr s
      let values ← evalOp val s
      some ((TileShape.allIndices shape).foldl
        (fun acc i =>
          let p := ptrs.data i
          acc.writeMem p.1 p.2 (h.storeValue (values.data i))) s)
  | .storePtrFloatMask h shape ptr val mask, s => do
      let ptrs ← evalOp ptr s
      let values ← evalOp val s
      let masks ← evalOp mask s
      some ((TileShape.allIndices shape).foldl
        (fun acc i =>
          let p := ptrs.data i
          if masks.data i then acc.writeMem p.1 p.2 (h.storeValue (values.data i))
          else acc) s)
  | .forLoop idx n body, s =>
      stepForLoopAux idx 0 n body s
  | .ifThen cond body, s => do
      let c ← evalOp cond s
      if c.data PUnit.unit then stepStmts body s else some s
termination_by st _ => (sizeOf st, 0)
decreasing_by
  all_goals (try (simp_wf; omega))
  simp_wf
  have h : 0 < sizeOf idx := by
    cases idx; simp
  omega

noncomputable def stepStmts : List Stmt → BlockState → Option BlockState
  | [], s => some s
  | st :: rest, s =>
      match stepStmt st s with
      | some s' => stepStmts rest s'
      | none => none
termination_by l _ => (sizeOf l, 0)
decreasing_by all_goals (simp_wf; omega)

noncomputable def stepForLoopAux
    (idx : RegName) (start n : Nat) (body : List Stmt) :
    BlockState → Option BlockState
  | s =>
      if start < n then
        match stepStmts body (s.setReg idx .nat [] (Tile.scalar start)) with
        | some s' => stepForLoopAux idx (start + 1) n body s'
        | none => none
      else some s
termination_by _ => (sizeOf body + 1, n - start)
decreasing_by all_goals omega

end

namespace stepStmts

@[simp] theorem nil {s : BlockState} :
    stepStmts [] s = some s := by
  unfold stepStmts
  rfl

theorem cons_some {st : Stmt} {rest : List Stmt} {s s' : BlockState}
    (h : stepStmt st s = some s') :
    stepStmts (st :: rest) s = stepStmts rest s' := by
  conv_lhs => unfold stepStmts
  rw [h]

theorem append_some {l1 l2 : List Stmt} {s s' : BlockState}
    (h : stepStmts l1 s = some s') :
    stepStmts (l1 ++ l2) s = stepStmts l2 s' := by
  induction l1 generalizing s with
  | nil =>
      unfold stepStmts at h
      injection h with hs
      subst hs
      rfl
  | cons st rest ih =>
      conv_lhs at h => unfold stepStmts
      cases hst : stepStmt st s with
      | none =>
          simp [hst] at h
      | some smid =>
          simp [hst] at h
          rw [List.cons_append, cons_some hst]
          exact ih h

end stepStmts

namespace stepForLoopAux

@[simp] theorem step_ge {idx} {start n} {body} {s} (h : n ≤ start) :
    stepForLoopAux idx start n body s = some s := by
  unfold stepForLoopAux
  simp [Nat.not_lt.mpr h]

@[simp] theorem step_eq_self {idx} {n} {body} (s) :
    stepForLoopAux idx n n body s = some s :=
  step_ge (le_refl n)

@[simp] theorem step_lt {idx} {start n} {body} {s} (h : start < n) :
    stepForLoopAux idx start n body s
      = (stepStmts body (s.setReg idx .nat [] (Tile.scalar start))).bind
          (stepForLoopAux idx (start + 1) n body) := by
  conv_lhs => unfold stepForLoopAux
  simp [h]
  cases hbody : stepStmts body (s.setReg idx .nat [] (Tile.scalar start)) <;> rfl

@[simp] theorem forLoop_unfold {idx} {n} {body} {s} :
    stepStmt (.forLoop idx n body) s
      = stepForLoopAux idx 0 n body s := by
  unfold stepStmt
  rfl

end stepForLoopAux

noncomputable def exec (k : Kernel) (s : BlockState) : Option BlockState :=
  stepStmts k.body s

mutual

theorem stepStmt_pid {st : Stmt} {s s' : BlockState}
    (h : stepStmt st s = some s') :
    s'.pid = s.pid := by
  cases st
  case assign dtype shape name e =>
    cases hv : evalOp e s with
    | none =>
        simp [stepStmt, hv] at h
    | some v =>
        simp [stepStmt, hv] at h
        cases h
        rfl
  case store region shape off val =>
    cases hoff : evalOp off s <;> simp [stepStmt, hoff] at h
    rename_i offsets
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases h
    simp [BlockState.foldl_writeMem_pid]
  case storeMask region shape off val mask =>
    cases hoff : evalOp off s <;> simp [stepStmt, hoff] at h
    rename_i offsets
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases hmask : evalOp mask s <;> simp [hmask] at h
    rename_i masks
    cases h
    simp [BlockState.foldl_writeMem_masked_pid]
  case storePtr shape ptr val =>
    cases hptr : evalOp ptr s <;> simp [stepStmt, hptr] at h
    rename_i ptrs
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases h
    simp [BlockState.foldl_writeMemAt_pid]
  case storePtrMask shape ptr val mask =>
    cases hptr : evalOp ptr s <;> simp [stepStmt, hptr] at h
    rename_i ptrs
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases hmask : evalOp mask s <;> simp [hmask] at h
    rename_i masks
    cases h
    simp [BlockState.foldl_writeMemAt_masked_pid]
  case storeFloat hfloat region shape off val =>
    cases hoff : evalOp off s <;> simp [stepStmt, hoff] at h
    rename_i offsets
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases h
    simp [BlockState.foldl_writeMem_pid]
  case storeFloatMask hfloat region shape off val mask =>
    cases hoff : evalOp off s <;> simp [stepStmt, hoff] at h
    rename_i offsets
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases hmask : evalOp mask s <;> simp [hmask] at h
    rename_i masks
    cases h
    simp [BlockState.foldl_writeMem_masked_pid]
  case storePtrFloat hfloat shape ptr val =>
    cases hptr : evalOp ptr s <;> simp [stepStmt, hptr] at h
    rename_i ptrs
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases h
    simp [BlockState.foldl_writeMemAt_pid]
  case storePtrFloatMask hfloat shape ptr val mask =>
    cases hptr : evalOp ptr s <;> simp [stepStmt, hptr] at h
    rename_i ptrs
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases hmask : evalOp mask s <;> simp [hmask] at h
    rename_i masks
    cases h
    simp [BlockState.foldl_writeMemAt_masked_pid]
  case forLoop idx n body =>
    simp at h
    exact stepForLoopAux_pid h
  case ifThen cond body =>
    cases hcond : evalOp cond s with
    | none => simp [stepStmt, hcond] at h
    | some c =>
        simp [stepStmt, hcond] at h
        by_cases hc : c.data PUnit.unit
        · simp [hc] at h
          exact stepStmts_pid h
        · simp [hc] at h
          rw [← h]

theorem stepStmts_pid {body : List Stmt} {s s' : BlockState}
    (h : stepStmts body s = some s') :
    s'.pid = s.pid := by
  cases body with
  | nil =>
      simp at h
      simp_all
  | cons st rest =>
      unfold stepStmts at h
      cases hst : stepStmt st s <;> simp [hst] at h
      rename_i mid
      exact (stepStmts_pid h).trans (stepStmt_pid hst)

theorem stepForLoopAux_pid {idx : RegName} {start n : Nat}
    {body : List Stmt} {s s' : BlockState}
    (h : stepForLoopAux idx start n body s = some s') :
    s'.pid = s.pid := by
  by_cases hlt : start < n
  · rw [stepForLoopAux.step_lt hlt] at h
    cases hbody : stepStmts body (s.setReg idx .nat [] (Tile.scalar start)) <;>
      simp [hbody] at h
    rename_i mid
    exact (stepForLoopAux_pid h).trans
      ((stepStmts_pid hbody).trans (BlockState.setReg_pid s idx .nat [] (Tile.scalar start)))
  · have hge : n ≤ start := Nat.le_of_not_gt hlt
    rw [stepForLoopAux.step_ge hge] at h
    simp_all

end

theorem exec_pid {k : Kernel} {s s' : BlockState}
    (h : exec k s = some s') :
    s'.pid = s.pid := by
  exact stepStmts_pid h

example : evalOp (.const 5) default = some (Tile.scalar (some 5)) := by
  unfold evalOp
  rfl

example : evalOp (.constNat 7) default = some (Tile.scalar 7) := by
  unfold evalOp
  rfl

example : evalOp (.programId 0) default = some (Tile.scalar 0) := by
  unfold evalOp
  rfl

example : evalOp (.add .nat .nil (.constNat 2) (.constNat 3)) default
    = some (Tile.scalar 5) := by
  unfold evalOp Tile.bop NumericDType.add
  rfl


end VeriTile.Triton
