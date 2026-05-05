/-
VeriTile.Examples.FlashAttention1.Common.Spec

FA-1 forward Real specs, 4D layout wrappers, and view wrappers.
-/

import VeriTile.Examples.FlashAttention1.Common.Kernels

namespace VeriTile.Examples

open VeriTile.Triton

/-! ## Math model — softmax-attention

Spec layer is ℝ-valued: `BlockState.readMem` only reads ℝ, never `⊥`, so
the natural type of a `tl.load`-fed kernel input is
`TileIndex shape → ℝ`. We lift to `Tile .real` (whose carrier is
`WithBot ℝ`) only as an internal staging step — `Tile.ofReal` /
`unbotD 0` round-trip — to reuse the existing `Tile.dot` /
`Tile.transpose` machinery. The `⊥` sentinel is reserved for things
like `-inf` / `tl.full(_, -inf)` / masked-off lanes that arise
*inside* a kernel, not at its inputs/outputs. -/

/-- Row-wise softmax along the trailing axis on a `Tile .real`. (Math
reference — no row-max subtraction; correctness is what matters at
the spec layer.) -/
noncomputable def softmaxRow {M N : Nat} (s : Tile .real [M, N]) :
    Tile .real [M, N] :=
  ⟨fun (m, n, _) =>
    let row := fun j : Fin N => (s.data (m, j, PUnit.unit)).unbotD 0
    let num := Real.exp (row n)
    let denom := Finset.univ.sum (fun j : Fin N => Real.exp (row j))
    some (num / denom)⟩

/-- Internal `Tile`-level attention helper. Takes `Tile .real` operands
and reuses `Tile.dot` / `Tile.transpose`. The user-facing spec is
`attentionReal` below; this helper exists so the proof can still pivot
through tile-level lemmas. -/
noncomputable def attention {M S D : Nat}
    (Q : Tile .real [M, D]) (K V : Tile .real [S, D])
    (scale : ℝ) : Tile .real [M, D] :=
  let qkT : Tile .real [M, S] :=
    Tile.dot [] Q (Tile.transpose [] K)
  let scaled : Tile .real [M, S] :=
    ⟨fun idx => Option.map (· * scale) (qkT.data idx)⟩
  let p : Tile .real [M, S] := softmaxRow scaled
  Tile.dot [] p V

/-- ℝ-valued reference attention: `softmax(Q · Kᵀ · scale) · V` on
plain `TileIndex → ℝ` inputs. Lifts through `Tile.ofReal`, runs
`attention`, projects back via `unbotD 0`. This is what
`fa1_forward_correct` compares the kernel against. -/
noncomputable def attentionReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  fun idx =>
    ((attention (Tile.ofReal Q) (Tile.ofReal K) (Tile.ofReal V)
        scale).data idx).unbotD 0

/-! ## Causal math model

The causal spec is written directly over `ℝ` instead of routing through
`Tile .real`: masked scores are semantically `-inf`, hence contribute
zero to `exp(score)`. Using `WithBot.unbotD 0` in the generic
`softmaxRow` helper would be wrong for that case, because it would turn
`-inf` into the real number `0` before exponentiation.
-/

/-- Causal attention for one 2D slice. A key position `j` contributes to
query position `i` exactly when `j ≤ i`; future keys contribute zero
softmax mass. -/
noncomputable def attentionRealCausal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    let score := fun j : Fin S =>
      scale * Finset.univ.sum (fun d' : Fin D =>
        Q (i, d', PUnit.unit) * K (j, d', PUnit.unit))
    let weight := fun j : Fin S =>
      if j.val ≤ i.val then Real.exp (score j) else 0
    let denom := Finset.univ.sum (fun j : Fin S => weight j)
    let numer := Finset.univ.sum (fun j : Fin S =>
      weight j * V (j, d, PUnit.unit))
    numer / denom

/-- Causal attention for a local Q block whose row `i` corresponds to
global query row `qStart + i`. This is the spec shape that matches the
strided causal kernel: `offs_m = pid_qb * M + arange(M)` is global,
while the theorem observes only the local `[M, D]` output tile. -/
noncomputable def attentionRealCausalBlock {M S D : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    let score := fun j : Fin S =>
      scale * Finset.univ.sum (fun d' : Fin D =>
        Q (i, d', PUnit.unit) * K (j, d', PUnit.unit))
    let weight := fun j : Fin S =>
      if j.val ≤ qStart + i.val then Real.exp (score j) else 0
    let denom := Finset.univ.sum (fun j : Fin S => weight j)
    let numer := Finset.univ.sum (fun j : Fin S =>
      weight j * V (j, d, PUnit.unit))
    numer / denom

/-! ## 4D layout — `[B, H, S, D]` reference attention

Step 1 of the FA-1 realism roadmap (issue #39) lifts Q/K/V/O to real
Triton 4D layout `[B, H, S, D]`. The math spec stays simple: the per-
`(batch, head)` slice computes ordinary 2D `attentionReal`, and the 4D
spec is "do that on every slice". The kernel under verification only
touches a single `(b, h, q_block)` slot per program instance, so the
slice helper is also the natural pivot for the correctness proof. -/

/-- Slice a 4D `[B, H, S, D]` tile-as-function at fixed `(batch, head)`,
yielding a 2D `[S, D]` tile-as-function. Used to thread the 4D spec
through the existing 2D `attentionReal`.

Specialized to 2 leading axes (FA-1's batch + head). A general
ND-prefix slicer (`TileIndex (prefix ++ rest) → ℝ → TileIndex rest → ℝ`)
would require `TileIndex` append helpers; deferred until a kernel
needs more than two leading axes (e.g. grouped attention with an
extra group axis). -/
def sliceBH {B H S D : Nat}
    (T : TileIndex [B, H, S, D] → ℝ)
    (b : Fin B) (h : Fin H) : TileIndex [S, D] → ℝ :=
  fun (i, d, _) => T (b, h, i, d, PUnit.unit)

/-- 4D ℝ-valued reference attention. Each `(batch, head)` slice is
independent and computes the ordinary 2D `attentionReal`. The kernel
spec for Step 1 will only assert this on the single `(b, h, q_block)`
slot determined by the three `program_id` axes. -/
noncomputable def attentionReal4D {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) =>
    attentionReal (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
      scale (i, d, PUnit.unit)

/-- 4D ℝ-valued causal reference attention. As with
`attentionReal4D`, each `(batch, head)` slice is independent; the only
difference is the per-row causal restriction `key ≤ query`. -/
noncomputable def attentionReal4DCausal {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) =>
    attentionRealCausal (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
      scale (i, d, PUnit.unit)

/-- `attentionReal4D` projects to `attentionReal` on each `(b, h)` slice.
The natural bridge from the 4D spec to the existing 2D correctness
machinery; the Step 1 correctness proof reduces 4D ↦ 2D via this
lemma and then reuses `streaming_eq_attentionReal` unchanged. -/
@[simp] theorem attentionReal4D_slice {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D) :
    attentionReal4D Q K V scale (b, h, i, d, PUnit.unit)
      = attentionReal (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
          scale (i, d, PUnit.unit) := rfl

/-- `attentionReal4DCausal` projects to `attentionRealCausal` on each
`(b, h)` slice. -/
@[simp] theorem attentionReal4DCausal_slice {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D) :
    attentionReal4DCausal Q K V scale (b, h, i, d, PUnit.unit)
      = attentionRealCausal (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
          scale (i, d, PUnit.unit) := rfl

/-- M-row slice of the `(b, h)` plane of a 4D tensor: pick `M`
consecutive rows starting at `start * M`. The boundary
`start * M + M ≤ S` ensures every row index fits.

This is the natural Q-input view that FA-1's strided kernel sees:
each program-instance `(b, h, qb)` reads the `M` rows
`[qb*M, qb*M+1, ..., qb*M+M-1]` of `Q4D`'s `(b, h)` plane. -/
def slice4DQRows {B H S D : Nat} (M : Nat)
    (T : TileIndex [B, H, S, D] → ℝ) (b : Fin B) (h : Fin H)
    (start : Nat) (hBnd : start * M + M ≤ S) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    T (b, h, ⟨start * M + i.val, by
      have := i.isLt
      omega⟩, d, PUnit.unit)

/-- Boundary-masked M-row Q block. In-bounds rows read the logical Q tensor;
out-of-bounds rows are the `other=0` value supplied to `tl.load`. -/
def slice4DQRowsBoundary {B H S D : Nat} (M : Nat)
    (T : TileIndex [B, H, S, D] → ℝ) (b : Fin B) (h : Fin H)
    (start : Nat) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    if hIn : start * M + i.val < S then
      T (b, h, ⟨start * M + i.val, hIn⟩, d, PUnit.unit)
    else
      0

@[simp] theorem slice4DQRowsBoundary_of_lt {B H S D : Nat} (M : Nat)
    (T : TileIndex [B, H, S, D] → ℝ) (b : Fin B) (h : Fin H)
    (start : Nat) (i : Fin M) (d : Fin D)
    (hIn : start * M + i.val < S) :
    slice4DQRowsBoundary M T b h start (i, d, PUnit.unit) =
      T (b, h, ⟨start * M + i.val, hIn⟩, d, PUnit.unit) := by
  simp [slice4DQRowsBoundary, hIn]

@[simp] theorem slice4DQRowsBoundary_of_not_lt {B H S D : Nat} (M : Nat)
    (T : TileIndex [B, H, S, D] → ℝ) (b : Fin B) (h : Fin H)
    (start : Nat) (i : Fin M) (d : Fin D)
    (hOut : ¬ start * M + i.val < S) :
    slice4DQRowsBoundary M T b h start (i, d, PUnit.unit) = 0 := by
  simp [slice4DQRowsBoundary, hOut]

/-- Pad a logical hidden dimension `D` to a block hidden dimension `Bd`.
Out-of-range hidden lanes are zero, matching `tl.load(..., other=0)` under
the D-tail mask. -/
def padHeadD {S D Bd : Nat} (X : TileIndex [S, D] → ℝ) :
    TileIndex [S, Bd] → ℝ :=
  fun (i, d, _) =>
    if h : d.val < D then
      X (i, ⟨d.val, h⟩, PUnit.unit)
    else
      0

@[simp] theorem padHeadD_of_lt {S D Bd : Nat}
    (X : TileIndex [S, D] → ℝ) (i : Fin S) (d : Fin Bd)
    (h : d.val < D) :
    padHeadD X (i, d, PUnit.unit) = X (i, ⟨d.val, h⟩, PUnit.unit) := by
  simp [padHeadD, h]

@[simp] theorem padHeadD_of_not_lt {S D Bd : Nat}
    (X : TileIndex [S, D] → ℝ) (i : Fin S) (d : Fin Bd)
    (h : ¬ d.val < D) :
    padHeadD X (i, d, PUnit.unit) = 0 := by
  simp [padHeadD, h]

/-- A zero-padded hidden-dimension sum over `Fin Bd` reduces to the logical
sum over `Fin D` when the logical dimension is covered by the block width. -/
theorem sum_padHeadD_eq {D Bd : Nat} (hDLe : D ≤ Bd) (f : Fin D → ℝ) :
    (Finset.univ : Finset (Fin Bd)).sum (fun d : Fin Bd =>
      if h : d.val < D then f ⟨d.val, h⟩ else 0)
      = (Finset.univ : Finset (Fin D)).sum f := by
  rw [show (Finset.univ.sum (fun d : Fin Bd =>
        if h : d.val < D then f ⟨d.val, h⟩ else 0)) =
        ((Finset.univ : Finset (Fin Bd)).filter (fun d => d.val < D)).sum
          (fun d : Fin Bd => if h : d.val < D then f ⟨d.val, h⟩ else 0)
        from ?_]
  · refine (Finset.sum_bij (fun (d : Fin D) (_ : d ∈ Finset.univ) =>
      Fin.castLE hDLe d) ?_ ?_ ?_ ?_).symm
    · intro d _
      simp [Fin.val_castLE, d.isLt]
    · intro d₁ _ d₂ _ heq
      apply Fin.ext
      have := congrArg Fin.val heq
      simpa [Fin.val_castLE] using this
    · intro d hd
      simp at hd
      refine ⟨⟨d.val, hd⟩, Finset.mem_univ _, ?_⟩
      apply Fin.ext
      simp
    · intro d _
      have hLt : (Fin.castLE hDLe d).val < D := by
        rw [Fin.val_castLE]; exact d.isLt
      rw [dif_pos hLt]
      congr 1
  · refine (Finset.sum_filter_of_ne ?_).symm
    intro d _ hNe
    by_contra hLt
    apply hNe
    rw [dif_neg hLt]

/-- Reinterpret the `(b, h)` plane of a 4D `[B, H, S, D]` tensor as a
flat `[Bk * numKVBlocks, D]` view, given `Bk * numKVBlocks = S`. The K
and V inputs of FA-1 take this form: the kernel iterates over
`numKVBlocks` blocks of `Bk` rows each, covering all `S` rows of the
`(b, h)` plane. -/
def slice4DFlat {B H S D : Nat} (Bk numKVBlocks : Nat)
    (T : TileIndex [B, H, S, D] → ℝ) (b : Fin B) (h : Fin H)
    (hSk : Bk * numKVBlocks = S) : TileIndex [Bk * numKVBlocks, D] → ℝ :=
  fun (j, d, _) =>
    T (b, h, ⟨j.val, by
      have := j.isLt
      omega⟩, d, PUnit.unit)

/-! ## User-facing 4D layout wrapper

The low-level strided FA-1 theorems expose every stride as a separate
argument. That is useful for proof reuse, but awkward as a public theorem
surface. `FA1Layout4D` bundles the Q/K/V/O strides for `[B, H, S, D]`
style tensors, plus the output-layout validity proof needed to turn the
output offset expression into an injective scatter/readback map.

The wrapper theorems near the end of the file are thin corollaries over
`fa1_forward_correct_4D` / `fa1_forward_correct_4D_causal`; they keep the
same semantics while giving users one layout argument instead of sixteen
stride arguments plus a separate `hOValid`. The input memory contracts are
exposed through `TensorView.loaded`, so theorem users can talk in tensor-view
metadata rather than raw `InputAt` / `Offset.strided` terms. -/

/-- Stride bundle for FA-1 over 4D `[B, H, S, D]` Q/K/V/O tensors.

Fields `q*`, `k*`, `v*`, and `o*` are the batch/head/sequence-or-output-row
/ feature-dimension strides for Q, K, V, and output respectively. -/
structure FA1Layout4D (B H S_q S_k D : Nat) where
  qB : Nat
  qH : Nat
  qS : Nat
  qD : Nat
  kB : Nat
  kH : Nat
  kS : Nat
  kD : Nat
  vB : Nat
  vH : Nat
  vS : Nat
  vD : Nat
  oB : Nat
  oH : Nat
  oS : Nat
  oD : Nat
  hOValid : Offset.StridesValid [B, H, S_q, D] [oB, oH, oS, oD]

namespace FA1Layout4D

def qStrides (layout : FA1Layout4D B H S_q S_k D) : List Nat :=
  [layout.qB, layout.qH, layout.qS, layout.qD]

def kStrides (layout : FA1Layout4D B H S_q S_k D) : List Nat :=
  [layout.kB, layout.kH, layout.kS, layout.kD]

def vStrides (layout : FA1Layout4D B H S_q S_k D) : List Nat :=
  [layout.vB, layout.vH, layout.vS, layout.vD]

def oStrides (layout : FA1Layout4D B H S_q S_k D) : List Nat :=
  [layout.oB, layout.oH, layout.oS, layout.oD]

/-- Tensor view for Q under this FA-1 layout. -/
def qView (layout : FA1Layout4D B H S_q S_k D)
    (qReg : RegionName) : TensorView [B, H, S_q, D] :=
  { region := qReg, base := 0, strides := layout.qStrides }

/-- Tensor view for K under this FA-1 layout. -/
def kView (layout : FA1Layout4D B H S_q S_k D)
    (kReg : RegionName) : TensorView [B, H, S_k, D] :=
  { region := kReg, base := 0, strides := layout.kStrides }

/-- Tensor view for V under this FA-1 layout. -/
def vView (layout : FA1Layout4D B H S_q S_k D)
    (vReg : RegionName) : TensorView [B, H, S_k, D] :=
  { region := vReg, base := 0, strides := layout.vStrides }

/-- Tensor view for output under this FA-1 layout. -/
def oView (layout : FA1Layout4D B H S_q S_k D)
    (outReg : RegionName) : TensorView [B, H, S_q, D] :=
  { region := outReg, base := 0, strides := layout.oStrides }

/-- Full 4D Q offset for an input tensor. -/
def qOffset (layout : FA1Layout4D B H S_q S_k D) :
    TileIndex [B, H, S_q, D] → Nat :=
  Offset.strided [B, H, S_q, D] layout.qStrides 0

/-- Full 4D K offset for an input tensor. -/
def kOffset (layout : FA1Layout4D B H S_q S_k D) :
    TileIndex [B, H, S_k, D] → Nat :=
  Offset.strided [B, H, S_k, D] layout.kStrides 0

/-- Full 4D V offset for an input tensor. -/
def vOffset (layout : FA1Layout4D B H S_q S_k D) :
    TileIndex [B, H, S_k, D] → Nat :=
  Offset.strided [B, H, S_k, D] layout.vStrides 0

/-- Output offset for the local `[M, D]` tile written by the current
`BlockState.pids` program instance. -/
def outBlockOffset (layout : FA1Layout4D B H S_q S_k D)
    (s : BlockState) (M : Nat) : TileIndex [M, D] → Nat :=
  fun idx =>
    s.pids 2 * layout.oB + s.pids 1 * layout.oH + s.pids 0 * M * layout.oS
      + idx.1.val * layout.oS + idx.2.1.val * layout.oD

/-- Output offset for a block-width hidden tile `[M, Bd]`; only lanes
`d < D` are written by D-tail kernels. -/
def outBlockOffsetD (layout : FA1Layout4D B H S_q S_k D)
    (s : BlockState) (M Bd : Nat) : TileIndex [M, Bd] → Nat :=
  fun idx =>
    s.pids 2 * layout.oB + s.pids 1 * layout.oH + s.pids 0 * M * layout.oS
      + idx.1.val * layout.oS + idx.2.1.val * layout.oD

def kernel (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (M Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel :=
  fa1ForwardKernelStrided qReg kReg vReg outReg M D Bk numKVBlocks
    layout.qB layout.qH layout.qS layout.qD
    layout.kB layout.kH layout.kS layout.kD
    layout.vB layout.vH layout.vS layout.vD
    layout.oB layout.oH layout.oS layout.oD scale

def causalKernel (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (M Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel :=
  fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
    layout.qB layout.qH layout.qS layout.qD
    layout.kB layout.kH layout.kS layout.kD
    layout.vB layout.vH layout.vS layout.vD
    layout.oB layout.oH layout.oS layout.oD scale

def boundaryKernel (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (M Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel :=
  fa1ForwardKernelStridedBoundary qReg kReg vReg outReg M D Bk numKVBlocks S_q S_k
    layout.qB layout.qH layout.qS layout.qD
    layout.kB layout.kH layout.kS layout.kD
    layout.vB layout.vH layout.vS layout.vD
    layout.oB layout.oH layout.oS layout.oD scale

def causalBoundaryKernel (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (M Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel :=
  fa1ForwardKernelStridedCausalBoundary qReg kReg vReg outReg
    M D Bk numKVBlocks S_q S_k
    layout.qB layout.qH layout.qS layout.qD
    layout.kB layout.kH layout.kS layout.kD
    layout.vB layout.vH layout.vS layout.vD
    layout.oB layout.oH layout.oS layout.oD scale

def boundaryKernelD (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (M Bd Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel :=
  fa1ForwardKernelStridedBoundaryD qReg kReg vReg outReg
    M Bd Bk numKVBlocks S_q S_k D
    layout.qB layout.qH layout.qS layout.qD
    layout.kB layout.kH layout.kS layout.kD
    layout.vB layout.vH layout.vS layout.vD
    layout.oB layout.oH layout.oS layout.oD scale

def causalBoundaryKernelD (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (M Bd Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel :=
  fa1ForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
    M Bd Bk numKVBlocks S_q S_k D
    layout.qB layout.qH layout.qS layout.qD
    layout.kB layout.kH layout.kS layout.kD
    layout.vB layout.vH layout.vS layout.vD
    layout.oB layout.oH layout.oS layout.oD scale

def naiveBoundaryKernelD (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (M Bd : Nat) (scale : ℝ) : ComputeKernel :=
  fa1NaiveForwardKernelStridedBoundaryD qReg kReg vReg outReg
    M Bd S_q S_k D
    layout.qB layout.qH layout.qS layout.qD
    layout.kB layout.kH layout.kS layout.kD
    layout.vB layout.vH layout.vS layout.vD
    layout.oB layout.oH layout.oS layout.oD scale

def naiveCausalBoundaryKernelD (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (M Bd : Nat) (scale : ℝ) : ComputeKernel :=
  fa1NaiveForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
    M Bd S_q S_k D
    layout.qB layout.qH layout.qS layout.qD
    layout.kB layout.kH layout.kS layout.kD
    layout.vB layout.vH layout.vS layout.vD
    layout.oB layout.oH layout.oS layout.oD scale

end FA1Layout4D

/-! ## View-level theorem surface

`FA1Layout4D` removes the raw stride arguments, but still keeps the
Q/K/V/O region names separate. `FA1Views4D` is the preferred public
surface: it bundles the logical layout together with the concrete memory
regions and exposes the four resulting `TensorView`s directly. -/

/-- Full memory contract bundle for one FA-1 call over 4D
`[B, H, S, D]` tensors. -/
structure FA1Views4D (B H S_q S_k D : Nat) where
  layout : FA1Layout4D B H S_q S_k D
  qReg : RegionName
  kReg : RegionName
  vReg : RegionName
  outReg : RegionName

namespace FA1Views4D

def qView (views : FA1Views4D B H S_q S_k D) : TensorView [B, H, S_q, D] :=
  views.layout.qView views.qReg

def kView (views : FA1Views4D B H S_q S_k D) : TensorView [B, H, S_k, D] :=
  views.layout.kView views.kReg

def vView (views : FA1Views4D B H S_q S_k D) : TensorView [B, H, S_k, D] :=
  views.layout.vView views.vReg

def oView (views : FA1Views4D B H S_q S_k D) : TensorView [B, H, S_q, D] :=
  views.layout.oView views.outReg

def outBlockOffset (views : FA1Views4D B H S_q S_k D)
    (s : BlockState) (M : Nat) : TileIndex [M, D] → Nat :=
  views.layout.outBlockOffset s M

def outBlockOffsetD (views : FA1Views4D B H S_q S_k D)
    (s : BlockState) (M Bd : Nat) : TileIndex [M, Bd] → Nat :=
  views.layout.outBlockOffsetD s M Bd

def kernel (views : FA1Views4D B H S_q S_k D)
    (M Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel :=
  views.layout.kernel views.qReg views.kReg views.vReg views.outReg
    M Bk numKVBlocks scale

def causalKernel (views : FA1Views4D B H S_q S_k D)
    (M Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel :=
  views.layout.causalKernel views.qReg views.kReg views.vReg views.outReg
    M Bk numKVBlocks scale

def boundaryKernel (views : FA1Views4D B H S_q S_k D)
    (M Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel :=
  views.layout.boundaryKernel views.qReg views.kReg views.vReg views.outReg
    M Bk numKVBlocks scale

def causalBoundaryKernel (views : FA1Views4D B H S_q S_k D)
    (M Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel :=
  views.layout.causalBoundaryKernel views.qReg views.kReg views.vReg views.outReg
    M Bk numKVBlocks scale

def boundaryKernelD (views : FA1Views4D B H S_q S_k D)
    (M Bd Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel :=
  views.layout.boundaryKernelD views.qReg views.kReg views.vReg views.outReg
    M Bd Bk numKVBlocks scale

def causalBoundaryKernelD (views : FA1Views4D B H S_q S_k D)
    (M Bd Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel :=
  views.layout.causalBoundaryKernelD views.qReg views.kReg views.vReg views.outReg
    M Bd Bk numKVBlocks scale

def naiveBoundaryKernelD (views : FA1Views4D B H S_q S_k D)
    (M Bd : Nat) (scale : ℝ) : ComputeKernel :=
  views.layout.naiveBoundaryKernelD views.qReg views.kReg views.vReg views.outReg
    M Bd scale

def naiveCausalBoundaryKernelD (views : FA1Views4D B H S_q S_k D)
    (M Bd : Nat) (scale : ℝ) : ComputeKernel :=
  views.layout.naiveCausalBoundaryKernelD views.qReg views.kReg views.vReg views.outReg
    M Bd scale

end FA1Views4D

end VeriTile.Examples
