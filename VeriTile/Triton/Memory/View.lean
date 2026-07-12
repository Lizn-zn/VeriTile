/-
VeriTile.Triton.Memory

Memory-contract layer for VeriTile's Triton model — the reusable bridge
between low-level named-region memory and high-level tensor-shaped
mathematical objects.

## Three-layer mental model

VeriTile's memory modeling sits in three layers, low → high:

  Storage      `readMem` / `writeMem` over named regions    -- scalar cells
                ↕
  Address map  `Offset.strided` / arbitrary `offsetFn`     -- "index → addr"
                ↕
  View         `TensorView { region, base, strides }`      -- typed fat pointer

The Storage layer is ground truth — `tl.load` / `tl.store` ultimately read
and write `s.readMem region addr`. The View layer is what kernel correctness
theorems quantify over; users name tensors, not byte addresses.

## What `TensorView` is (and isn't)

`TensorView` is **not** memory. It is a *lens* — typed metadata describing
how a logical N-dim tensor maps to memory addresses, in the same spirit as
NumPy `ndarray` (without the data buffer), PyTorch `Tensor.stride()`
metadata, C++ `std::mdspan`, or Triton's own `tl.make_block_ptr`:

  | piece    | role                                              |
  | -------- | ------------------------------------------------- |
  | shape    | type-level dimensions (`TileShape`)               |
  | region   | which named buffer the view addresses             |
  | base     | starting offset within that region                |
  | strides  | per-axis cell-count step (not byte step)          |

`view.offset idx` computes an address; `view.loaded s xs` declares "memory
at these addresses contains tensor `xs`"; `view.observe sf idx` reads back.
None of this stores data — data is accessed through `BlockState.readMem`.

## Coverage and escape hatches

`TensorView` covers the **strided-affine** address-map subset: every
`view.offset idx` decomposes as `base + Σⱼ idxⱼ * strides[j]`. This handles
the vast majority of production kernels (FA-1/2 forward and backward, GEMM,
LayerNorm, Welford, RMSNorm, sliding-window attention, …).

Non-affine addressing patterns use a sibling lens (still on top of the same
state-level memory API):

  * Paged KV (issue #42, vLLM-style):
      `addr = base + block_table[idx_block] * page_stride + …`
      — data-dependent indirection through `IndirectView`.
  * Scatter / gather with arbitrary index tensors — same family.
  * Atomic reduce — orthogonal (concurrency, not addressing; see issue #12).

For these, `IndirectView` packages the common read-only gather pattern:
load a Nat index tensor from one region, feed it to a data-address function,
then read from a data region. The executable Triton surface remains ordinary
typed `tl.load` plus pointer arithmetic rather than a special gather AST node.

## Module contents

* `InputAt` / `observeTileAt` for arbitrary offset maps (low-level).
* `Offset` families for linear, row-major, contiguous, and generic strided
  layouts, with injectivity lemmas (`StridesValid` + `strided_inj`).
* `TensorView`: typed fat-pointer metadata for strided-affine layouts.
* `IndirectView`: typed metadata for read-only, index-tensor-driven loads.

## API guidance

The preferred public memory-contract surface is `TensorView.loaded`: new
kernel correctness theorems should generally quantify over tensor views and
state inputs as `TensorView.loaded s view tensor`. `InputAt` remains the
low-level escape hatch for proof internals and one-off address maps that are
not yet packaged as tensor metadata.

`TensorView.base` is the view's storage offset, mirroring a tensor metadata
field. For tiled kernels, keep the view as the fixed logical tensor layout
and put per-`program_id` block offsets in the kernel's address expression or
in the theorem's local output-view helper. Constructing a fresh per-program
view is allowed, but the main path is fixed view metadata plus dynamic
offsets.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics

namespace VeriTile.Triton

/-! ## Generic ND tile-load / observe -/

/-- Region `region` holds an ND tile `xs` at the addresses given by
`offsetFn`. -/
def InputAt {shape : TileShape} (s : BlockState) (region : RegionName)
    (offsetFn : TileIndex shape → Nat)
    (xs : TileIndex shape → ℝ) : Prop :=
  ∀ idx : TileIndex shape, s.readMem region (offsetFn idx) = xs idx

/-- Generic tile-load bridge: if a tile-local address map factors through a
larger loaded memory contract, the concrete memory tile is `Tile.ofReal` of
the corresponding logical slice. This is the reusable replacement for
one-off 2D `InputLoadedAt` lemmas in examples. -/
theorem load_tile_eq_of_InputAt_map
    {tileShape fullShape : TileShape}
    (s : BlockState) (region : RegionName)
    (addr : TileIndex tileShape → Nat)
    (offsetFn : TileIndex fullShape → Nat)
    (embed : TileIndex tileShape → TileIndex fullShape)
    (xs : TileIndex fullShape → ℝ)
    (hAddr : ∀ idx : TileIndex tileShape, addr idx = offsetFn (embed idx))
    (hLoaded : InputAt s region offsetFn xs) :
    (⟨fun idx : TileIndex tileShape => some (s.readMem region (addr idx))⟩
      : Tile .real tileShape)
      =
    Tile.ofReal (fun idx : TileIndex tileShape => xs (embed idx)) := by
  ext idx
  rw [Tile.ofReal_data]
  change some (s.readMem region (addr idx)) = some (xs (embed idx))
  rw [hAddr idx]
  exact congrArg some (hLoaded (embed idx))

/-- Read the cell at `offsetFn idx` of an ND tile from the optional final
state of `exec`. -/
noncomputable def observeTileAt {shape : TileShape}
    (sf : Option BlockState) (region : RegionName)
    (offsetFn : TileIndex shape → Nat) (idx : TileIndex shape) : Option ℝ :=
  sf.map (·.readMem region (offsetFn idx))

/-! ## Standard offset families -/

namespace Offset

/-- ND strided offset: `base + Σⱼ idxⱼ * stridesⱼ`. -/
def strided : (shape : TileShape) → (strides : List Nat) → (base : Nat) →
    TileIndex shape → Nat
  | [],          _,        base, _              => base
  | _ :: _,      [],        base, _              => base
  | _ :: rest,  s :: ss,   base, (i, idxRest)  =>
      strided rest ss (base + i.val * s) idxRest

/-- Canonical contiguous (row-major, innermost-stride-1) strides derived from
a shape: `[shape[1] * shape[2] * …, shape[2] * …, …, shape[k], 1]`. -/
def contigStrides : TileShape → List Nat
  | []        => []
  | _ :: rest => (rest.foldr (· * ·) 1) :: contigStrides rest

/-- Fully-contiguous ND row-major offset. -/
def contig (shape : TileShape) (base : Nat) : TileIndex shape → Nat :=
  strided shape (contigStrides shape) base

/-- 1D linear offset `base + i.val`. -/
def linear1D {n : Nat} (base : Nat) : TileIndex [n] → Nat :=
  strided [n] [1] base

/-- 2D row-major offset with explicit row stride. -/
def rowMajor2D {rows cols : Nat} (base rowStride : Nat) :
    TileIndex [rows, cols] → Nat :=
  strided [rows, cols] [rowStride, 1] base

theorem linear1D_inj {n : Nat} (base : Nat) :
    Function.Injective (linear1D (n := n) base) := by
  rintro ⟨a, _⟩ ⟨b, _⟩ hab
  have h : base + a.val * 1 + 0 = base + b.val * 1 + 0 := by
    simpa [linear1D, strided] using hab
  obtain rfl : a = b := Fin.ext (by omega)
  rfl

/-- Injectivity of the 2D row-major offset map. Requires `cols ≤ rowStride`
so distinct rows do not overlap. -/
theorem rowMajor2D_inj {rows cols : Nat} (base rowStride : Nat)
    (h : cols ≤ rowStride) :
    Function.Injective (rowMajor2D (rows := rows) (cols := cols) base rowStride) := by
  rintro ⟨a₁, a₂, _⟩ ⟨b₁, b₂, _⟩ hab
  have ha₂ : a₂.val < rowStride := lt_of_lt_of_le a₂.isLt h
  have hb₂ : b₂.val < rowStride := lt_of_lt_of_le b₂.isLt h
  have hSpos : 0 < rowStride := lt_of_le_of_lt (Nat.zero_le _) ha₂
  have hsum : a₁.val * rowStride + a₂.val = b₁.val * rowStride + b₂.val := by
    have h := hab
    simp only [rowMajor2D, strided] at h
    omega
  have h_a₂ : a₂.val = b₂.val := by
    have h_mod : (a₁.val * rowStride + a₂.val) % rowStride
              = (b₁.val * rowStride + b₂.val) % rowStride := by rw [hsum]
    rw [Nat.mul_add_mod_self_right, Nat.mul_add_mod_self_right,
        Nat.mod_eq_of_lt ha₂, Nat.mod_eq_of_lt hb₂] at h_mod
    exact h_mod
  have h_a₁ : a₁.val = b₁.val := by
    have h_div : (a₁.val * rowStride + a₂.val) / rowStride
              = (b₁.val * rowStride + b₂.val) / rowStride := by rw [hsum]
    rw [show a₁.val * rowStride = rowStride * a₁.val from Nat.mul_comm _ _,
        show b₁.val * rowStride = rowStride * b₁.val from Nat.mul_comm _ _,
        Nat.mul_add_div hSpos, Nat.mul_add_div hSpos,
        Nat.div_eq_of_lt ha₂, Nat.div_eq_of_lt hb₂] at h_div
    omega
  obtain rfl : a₁ = b₁ := Fin.ext h_a₁
  obtain rfl : a₂ = b₂ := Fin.ext h_a₂
  rfl

/-! ## Store readback helpers -/

/-- Store-stage readback helper for an unmasked row-major 2D Real store whose
address tile and value tile are already present in registers. -/
theorem store_stage_readback_rowMajor2D
    (outReg : RegionName) (ptrName valueName : RegName)
    (rows cols : Nat) (s : BlockState)
    (valueFn : TileIndex [rows, cols] → ℝ)
    (hPtrs : s.regs .nat [rows, cols] ptrName =
      some (⟨Offset.rowMajor2D (rows := rows) (cols := cols) 0 cols⟩ :
        Tile .nat [rows, cols]))
    (hVal : s.regs .real [rows, cols] valueName = some (Tile.ofReal valueFn)) :
    ∀ idx : TileIndex [rows, cols],
      observeTileAt
        (stepStmt (Stmt.store .real [rows, cols]
          (MemAccess.region outReg (Op.ref .nat [rows, cols] ptrName))
          (Op.ref .real [rows, cols] valueName) MaskOpt.none) s)
        outReg (Offset.rowMajor2D (rows := rows) (cols := cols) 0 cols) idx =
      some (valueFn idx) := by
  intro idx
  have h_inj : Function.Injective
      (fun i : TileIndex [rows, cols] => i.1.val * cols + i.2.1.val) := by
    intro a b h
    have hrow : Offset.rowMajor2D (rows := rows) (cols := cols) 0 cols a =
        Offset.rowMajor2D (rows := rows) (cols := cols) 0 cols b := by
      simpa [Offset.rowMajor2D, Offset.strided] using h
    exact Offset.rowMajor2D_inj (base := 0) (rowStride := cols) (le_refl cols) hrow
  simp [observeTileAt, stepStmt, evalOp, hPtrs, hVal, Tile.ofReal,
    BlockState.writeMemTyped_real, Offset.rowMajor2D, Offset.strided]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj idx]

/-- Row-major store-stage readback with a nonzero base offset. -/
theorem store_stage_readback_rowMajor2D_base
    (outReg : RegionName) (ptrName valueName : RegName)
    (rows cols base : Nat) (s : BlockState)
    (valueFn : TileIndex [rows, cols] → ℝ)
    (hPtrs : s.regs .nat [rows, cols] ptrName =
      some (⟨Offset.rowMajor2D (rows := rows) (cols := cols) base cols⟩ :
        Tile .nat [rows, cols]))
    (hVal : s.regs .real [rows, cols] valueName = some (Tile.ofReal valueFn)) :
    ∀ idx : TileIndex [rows, cols],
      observeTileAt
        (stepStmt (Stmt.store .real [rows, cols]
          (MemAccess.region outReg (Op.ref .nat [rows, cols] ptrName))
          (Op.ref .real [rows, cols] valueName) MaskOpt.none) s)
        outReg (Offset.rowMajor2D (rows := rows) (cols := cols) base cols) idx =
      some (valueFn idx) := by
  intro idx
  have h_inj : Function.Injective
      (fun i : TileIndex [rows, cols] => base + i.1.val * cols + i.2.1.val) := by
    intro a b h
    have hrow : Offset.rowMajor2D (rows := rows) (cols := cols) base cols a =
        Offset.rowMajor2D (rows := rows) (cols := cols) base cols b := by
      simpa [Offset.rowMajor2D, Offset.strided] using h
    exact Offset.rowMajor2D_inj (base := base) (rowStride := cols) (le_refl cols) hrow
  simp [observeTileAt, stepStmt, evalOp, hPtrs, hVal, Tile.ofReal,
    BlockState.writeMemTyped_real, Offset.rowMajor2D, Offset.strided]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj idx]


/-- Maximum offset (relative to `base`) reachable by `strided shape strides`. -/
def maxOffset : TileShape → List Nat → Nat
  | [], _              => 0
  | _ :: _, []         => 0
  | d :: ds, s :: ss   => (d - 1) * s + maxOffset ds ss

/-- Sufficient non-overlap condition for strided layouts. -/
def StridesValid : TileShape → List Nat → Prop
  | [], _              => True
  | _ :: _, []         => False
  | _ :: ds, s :: ss   => maxOffset ds ss < s ∧ StridesValid ds ss

/-- Lower bound: `strided` always returns at least `base`. -/
theorem base_le_strided : ∀ {shape : TileShape} (strides : List Nat)
    (base : Nat) (idx : TileIndex shape),
    base ≤ strided shape strides base idx
  | [], _, base, _ => le_refl base
  | _ :: _, [], base, _ => le_refl base
  | _ :: ds, s :: ss, base, (i, idxRest) => by
      simp only [strided]
      have := base_le_strided ss (base + i.val * s) idxRest
      omega

/-- Upper bound: `strided` returns at most `base + maxOffset shape strides`. -/
theorem strided_le_maxOffset : ∀ {shape : TileShape} (strides : List Nat)
    (base : Nat) (idx : TileIndex shape),
    strided shape strides base idx ≤ base + maxOffset shape strides
  | [], _, base, _ => by simp [strided, maxOffset]
  | _ :: _, [], base, _ => by simp [strided, maxOffset]
  | d :: ds, s :: ss, base, (i, idxRest) => by
      simp only [strided, maxOffset]
      have hRec := strided_le_maxOffset ss (base + i.val * s) idxRest
      have hi : i.val ≤ d - 1 := Nat.le_sub_one_of_lt i.isLt
      have hStep : i.val * s ≤ (d - 1) * s := Nat.mul_le_mul_right s hi
      omega

/-- General ND injectivity for `Offset.strided` under `StridesValid`. -/
theorem strided_inj : ∀ {shape : TileShape} {strides : List Nat} (base : Nat),
    StridesValid shape strides →
    Function.Injective (strided shape strides base) := by
  intro shape
  induction shape with
  | nil =>
      intros _strides _base _hValid a b _hEq
      exact Subsingleton.elim a b
  | cons d ds ih =>
      intros strides base hValid
      match strides, hValid with
      | [], hValid => exact absurd hValid id
      | s :: ss, ⟨hMax, hValidRec⟩ =>
          rintro ⟨i, restA⟩ ⟨j, restB⟩ hEq
          simp only [strided] at hEq
          have hLowerA := base_le_strided ss (base + i.val * s) restA
          have hRangeA := strided_le_maxOffset ss (base + i.val * s) restA
          have hLowerB := base_le_strided ss (base + j.val * s) restB
          have hRangeB := strided_le_maxOffset ss (base + j.val * s) restB
          have hi_eq_j : i.val = j.val := by
            rcases lt_trichotomy i.val j.val with hlt | heq | hgt
            · exfalso
              have h1 : base + j.val * s ≤ base + i.val * s + maxOffset ds ss :=
                calc base + j.val * s
                    ≤ strided ds ss (base + j.val * s) restB := hLowerB
                  _ = strided ds ss (base + i.val * s) restA := hEq.symm
                  _ ≤ base + i.val * s + maxOffset ds ss := hRangeA
              have h2 : (i.val + 1) * s ≤ j.val * s := Nat.mul_le_mul_right s hlt
              have h2' : i.val * s + s ≤ j.val * s := by
                rw [Nat.add_mul, Nat.one_mul] at h2; exact h2
              omega
            · exact heq
            · exfalso
              have h1 : base + i.val * s ≤ base + j.val * s + maxOffset ds ss :=
                calc base + i.val * s
                    ≤ strided ds ss (base + i.val * s) restA := hLowerA
                  _ = strided ds ss (base + j.val * s) restB := hEq
                  _ ≤ base + j.val * s + maxOffset ds ss := hRangeB
              have h2 : (j.val + 1) * s ≤ i.val * s := Nat.mul_le_mul_right s hgt
              have h2' : j.val * s + s ≤ i.val * s := by
                rw [Nat.add_mul, Nat.one_mul] at h2; exact h2
              omega
          have hij_fin : i = j := Fin.ext hi_eq_j
          subst hij_fin
          have hRest := ih (base + i.val * s) hValidRec hEq
          exact Prod.ext rfl hRest

end Offset

/-! ## Tensor views -/

/-- A logical tensor view into VeriTile memory. `shape` is the logical tensor
shape, `region` selects the memory region, and `base`/`strides` define the
strided address map from logical indices to memory offsets. -/
structure TensorView (shape : TileShape) where
  region  : RegionName
  base    : Nat := 0
  strides : List Nat

namespace TensorView

/-- Address map for a tensor view. -/
def offset {shape : TileShape} (view : TensorView shape) : TileIndex shape → Nat :=
  Offset.strided shape view.strides view.base

/-- Memory-state contract: `view.region` contains tensor `xs` according to
`view`'s strided address map. -/
def loaded {shape : TileShape} (s : BlockState) (view : TensorView shape)
    (xs : TileIndex shape → ℝ) : Prop :=
  InputAt s view.region view.offset xs

/-- One-dimensional loaded-view shorthand for array-shaped specs. -/
def loadedArray {n : Nat} (s : BlockState) (view : TensorView [n])
    (xs : Fin n → ℝ) : Prop :=
  loaded s view (fun idx : TileIndex [n] => xs idx.1)

/-- Non-overlap condition for a tensor view's address map. -/
def Valid {shape : TileShape} (view : TensorView shape) : Prop :=
  Offset.StridesValid shape view.strides

/-- Valid tensor views have injective address maps. -/
theorem offset_injective {shape : TileShape} (view : TensorView shape)
    (hValid : view.Valid) : Function.Injective view.offset :=
  Offset.strided_inj view.base hValid

/-- Observe a tensor element through a view from an optional final state. -/
noncomputable def observe {shape : TileShape}
    (sf : Option BlockState) (view : TensorView shape)
    (idx : TileIndex shape) : Option ℝ :=
  observeTileAt sf view.region view.offset idx

/-- One-dimensional view equality shorthand for array-shaped specs. -/
def equalsArray {n : Nat} (sf : Option BlockState) (view : TensorView [n])
    (xs : Fin n → ℝ) : Prop :=
  ∀ idx : TileIndex [n], observe sf view idx = some (xs idx.1)

/-- One-dimensional concrete-state view equality shorthand for array-shaped
specs. -/
def matchesArray {n : Nat} (s' : BlockState) (view : TensorView [n])
    (xs : Fin n → ℝ) : Prop :=
  equalsArray (some s') view xs

/-- One input slot of a loaded-inputs contract: a 1-D view together with
the array it holds. The dimension is existential so one contract lists
inputs of different sizes. -/
def Slot : Type := (n : Nat) × TensorView [n] × (Fin n → ℝ)

/-- Build an input slot. -/
def slot {n : Nat} (view : TensorView [n]) (xs : Fin n → ℝ) : Slot :=
  ⟨n, view, xs⟩

/-- The loaded-inputs contract: every listed view holds its array.
`ViewsLoaded s [(v₁.slot xs), (v₂.slot ys), …]` is the single hypothesis a
specification takes in place of one loaded-input line per input; each slot
carries its own layout (`programTileView`, `rowTileView`, `featureView`,
…), so continuous, strided and shared-parameter inputs all speak the same
language. -/
def ViewsLoaded (s : BlockState) : List Slot → Prop
  | [] => True
  | ⟨_, v, xs⟩ :: rest => loadedArray s v xs ∧ ViewsLoaded s rest

@[simp] theorem viewsLoaded_nil (s : BlockState) : ViewsLoaded s [] := trivial

@[simp] theorem viewsLoaded_cons (s : BlockState) {n : Nat}
    (v : TensorView [n]) (xs : Fin n → ℝ) (rest : List Slot) :
    ViewsLoaded s (slot v xs :: rest) ↔
      loadedArray s v xs ∧ ViewsLoaded s rest := Iff.rfl

/-- A loaded view reads back its logical tensor value from the same state. -/
theorem readMem_of_loaded {shape : TileShape} {s : BlockState}
    {view : TensorView shape} {xs : TileIndex shape → ℝ}
    (h : loaded s view xs) (idx : TileIndex shape) :
    s.readMem view.region (view.offset idx) = xs idx := by
  exact h idx

/-- Observing a loaded view in a concrete state returns the corresponding
logical tensor value. -/
@[simp] theorem observe_some_of_loaded {shape : TileShape} {s : BlockState}
    {view : TensorView shape} {xs : TileIndex shape → ℝ}
    (h : loaded s view xs) (idx : TileIndex shape) :
    observe (some s) view idx = some (xs idx) := by
  simp [observe, observeTileAt, BlockState.readMem]
  exact h idx

end TensorView

/-! ## Indirect views -/

/--
A read-only logical view driven by an index tensor in memory.

`idxShape` indexes the HBM tensor that stores Nat indices. `innerShape`
indexes the data slice reached by each loaded index. This covers 1D embedding
lookup (`idxShape = [N]`, `innerShape = []`) and paged-KV style reads where a
loaded block id selects a physical page and an inner `D` coordinate selects a
lane within that page.
-/
structure IndirectView (idxShape innerShape : TileShape) where
  idxRegion  : RegionName
  dataRegion : RegionName
  idxOffset  : TileIndex idxShape → Nat
  dataAddr   : Nat → TileIndex innerShape → Nat

namespace IndirectView

/-- Read an element through an indirect view from a concrete state. -/
def observe {idxShape innerShape : TileShape}
    (view : IndirectView idxShape innerShape) (s : BlockState)
    (idx : TileIndex idxShape) (inner : TileIndex innerShape) : ℝ :=
  let loadedIdx := s.readMemValue .nat view.idxRegion (view.idxOffset idx)
  s.readMem view.dataRegion (view.dataAddr loadedIdx inner)

/-- Address helper for paged layouts: a logical token `t` first selects a
logical page, the block table maps that page to a physical block, and `d`
selects the inner element within that physical block. -/
def pagedAddr (block : Nat → Nat)
    (pageSize pageStride tokenStride dStride : Nat) (t d : Nat) : Nat :=
  block (t / pageSize) * pageStride
    + (t % pageSize) * tokenStride
    + d * dStride

end IndirectView

end VeriTile.Triton
