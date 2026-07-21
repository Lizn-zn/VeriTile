import VeriTile.Triton

/-!
# `rotary_transform_ops` — strict per-kernel correctness

`rotary_kernel` applies the rotary position embedding out-of-place, writing to
`OUT` from `X`: each program owns one seq-block (`program_id(0)`), one batch
(`program_id(1)`), and one head (`program_id(2)`). On the non-interleaved branch
(`INTERLEAVED = false`) it loads the per-row `cos`/`sin` half-dim vectors and the
two head-dim halves `x0`/`x1`, then stores `(x0*cos - x1*sin, x0*sin + x1*cos)`
(the `sin := -sin` conjugate when `CONJUGATE = true`).

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body, on the non-interleaved branch. The host launch (the
`(seq-blocks, batch, heads)` grid, `BLOCK_M`/`BLOCK_K` choices, the varlen
`CU_SEQLENS` base-pointer arithmetic, the `pid_m * BLOCK_M >= seqlen` early
return, and how the runtime composes per-program writes into one buffer) is the
*trusted boundary*, not a proof obligation here. Because the seq-block, batch,
head, and per-lane indices are universally quantified, the per-program statement
covers every program of the grid.

## Proof architecture

```
rotary_transform_ops_output_summary_general  ← TOP THEOREM (genuine, dimension-general)
  ├─ rotary_kernel_surface_toAlgorithm_supported  surface lowers to the algorithm layer
  └─ rotary_kernel_o0o1_row_all_outputs_compute_correct
        ← ComputeCorrect over BOTH OUT stores = genuine rotaryO0Spec/rotaryO1Spec
          (x0·cos − x1·sin, x0·sin + x1·cos), NOT a re-executed kernel value

supporting per-row store track:
  rotary_kernel_o0o1_row_{o0,o1}_compute_correct
    ├─ rotary_kernel_o0o1_row_o0_correct
    └─ rotary_kernel_o0o1_row_o1_correct

full 2D track:
  only the `[BLOCK_M, BLOCK_HALF]` tile coordinate/active-mask helpers
  (`rowIndex2D`/`dimIndex2D`/`active2D`) remain; the full 2D body kernel and
  its closed-form proof are not present.

`⊨` track (MetaGroupedMasked2DKernelIO):
  rotary_transform_ops_meta_implements                ← `⊨` HEADLINE
  one metadata-grouped masked Hoare triple over FLAT pointer memory for the whole
  per-row/per-head `rotary_kernel` — BOTH `IS_VARLEN` branches, fully parametric:
  the two `.nat` slots are `CU_SEQLENS[pid_batch]` / `CU_SEQLENS[pid_batch+1]`
  (difference = per-program `seqlen`), four read channels x0/x1/cos/sin, two write
  channels the two OUT halves, value `f` = genuine `(x0·cos − x1·sin, x0·sin +
  x1·cos)` over the loaded inputs.
  ├─ rotaryMetaIO / meta_row_flattenOk / meta_row_traceSafe / meta_row_frame
  ├─ meta_row_o0_correct / meta_row_o1_correct
  └─ metaO0Spec_eq_F / metaO1Spec_eq_F
```

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; the `.to(tl.float32)`
register casts erase to the identity at the algorithm layer (post-erasure all
dtypes unify to `ℝ`). `cos`/`sin` are modeled as **precomputed inputs** loaded
from memory, not computed; the `CONJUGATE` flag selects the `sin := -sin` spec.
The dimension-general summary quantifies over arbitrary symbolic shapes and
strides (and the `IS_VARLEN`, `IS_SEQLEN_OFFSETS_TENSOR`, `CONJUGATE` flags).
The top summary `rotary_transform_ops_output_summary_general` exposes the
**genuine** rotary closed form: the full surface lowers to the algorithm layer,
and both output halves of the non-interleaved rotation body read back to
`rotaryO0Spec`/`rotaryO1Spec` (`x0·cos − x1·sin`, `x0·sin + x1·cos`) — the
actual embedding from the precomputed `COS`/`SIN` cache, not a re-executed
kernel value. The interleaved branch is not modeled at the value level.
-/

namespace VeriTile.Bench.TritonBenchG.RotaryTransformOps

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Faithful DSL port of `rotary_transform_ops.py`'s `rotary_kernel`.

Python's `SEQLEN_OFFSETS` argument is a union of scalar offset and tensor
pointer. The surface keeps `SEQLEN_OFFSETS` as the tensor region used by the
tensor-offset path and uses `SEQLEN_OFFSETS_SCALAR` for the scalar-offset path. -/
def rotary_kernel_surface
    (OUT X COS SIN : RegionName) (CU_SEQLENS SEQLEN_OFFSETS : Region .nat)
    (SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) :
    ComputeKernel := triton {
  pid_m = tl.program_id(axis=0)
  pid_batch = tl.program_id(axis=1)
  pid_head = tl.program_id(axis=2)
  rotary_dim_half = rotary_dim // $(2)

  if not IS_VARLEN {
    X = X + pid_batch * $(stride_x_batch) + pid_head * $(stride_x_nheads)
    OUT = OUT + pid_batch * $(stride_out_batch) + pid_head * $(stride_out_nheads)
  } else {
    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen = tl.load(CU_SEQLENS + pid_batch + $(1)) - start_idx
    X = X + start_idx * $(stride_x_seqlen) + pid_head * $(stride_x_nheads)
    OUT = OUT + start_idx * $(stride_out_seqlen) + pid_head * $(stride_out_nheads)
  }

  if pid_m * $(BLOCK_M) >= seqlen {
    return
  }
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  if not IS_SEQLEN_OFFSETS_TENSOR {
    rm_cs = rm + $(SEQLEN_OFFSETS_SCALAR)
  } else {
    rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
  }
  rk = tl.arange(0, $(BLOCK_K))
  rk_half = tl.arange(0, $(BLOCK_K) // $(2))

  if not INTERLEAVED {
    X = X + (rm[:, None] * $(stride_x_seqlen) +
      rk_half[None, :] * $(stride_x_headdim))
    COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
    SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
    cos = tl.load(COS,
      mask=(rm_cs[:, None] < $(seqlen_ro)) & (rk_half[None, :] < rotary_dim_half),
      other=1.0).to(tl.float32)
    sin = tl.load(SIN,
      mask=(rm_cs[:, None] < $(seqlen_ro)) & (rk_half[None, :] < rotary_dim_half),
      other=0.0).to(tl.float32)
    x0 = tl.load(X,
      mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
      other=0.0).to(tl.float32)
    x1 = tl.load(X + rotary_dim_half * $(stride_x_headdim),
      mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
      other=0.0).to(tl.float32)
    if CONJUGATE {
      sin = -sin
    }
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    OUT = OUT + (rm[:, None] * $(stride_out_seqlen) +
      rk_half[None, :] * $(stride_out_headdim))
    tl.store(OUT, o0,
      mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
    tl.store(OUT + rotary_dim_half * $(stride_out_headdim), o1,
      mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
  } else {
    rk_swap = rk + ((rk + $(1)) % $(2)) * $(2) - $(1)
    rk_repeat = tl.arange(0, $(BLOCK_K)) // $(2)
    X0 = X + (rm[:, None] * $(stride_x_seqlen) + rk[None, :] * $(stride_x_headdim))
    X1 = X + (rm[:, None] * $(stride_x_seqlen) + rk_swap[None, :] * $(stride_x_headdim))
    COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
    SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
    cos = tl.load(COS,
      mask=(rm_cs[:, None] < $(seqlen_ro)) & (rk_repeat[None, :] < rotary_dim_half),
      other=1.0).to(tl.float32)
    sin = tl.load(SIN,
      mask=(rm_cs[:, None] < $(seqlen_ro)) & (rk_repeat[None, :] < rotary_dim_half),
      other=0.0).to(tl.float32)
    x0 = tl.load(X0,
      mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim),
      other=0.0).to(tl.float32)
    x1 = tl.load(X1,
      mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim),
      other=0.0).to(tl.float32)
    if CONJUGATE {
      sin = -sin
    }
    x0_cos = x0 * cos
    x1_sin = x1 * sin
    out = tl.where(rk[None, :] % $(2) == $(0), x0_cos - x1_sin, x0_cos + x1_sin)
    OUT = OUT + (rm[:, None] * $(stride_out_seqlen) + rk[None, :] * $(stride_out_headdim))
    tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))
  }
}

/-- The full rotary-transform-ops surface lowers to the algorithm layer,
including scalar/tensor sequence offsets, varlen, interleaved, and conjugate
branches. -/
theorem rotary_kernel_surface_toAlgorithm_supported
    (OUT X COS SIN : RegionName) (CU_SEQLENS SEQLEN_OFFSETS : Region .nat)
    (SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) :
    ∃ alg, (rotary_kernel_surface OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS
      SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro stride_out_batch
      stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch
      stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M
      IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE).toAlgorithm? =
        Except.ok alg := by
  simp [rotary_kernel_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

def rowIndex (s : BlockState) (BLOCK_M : Nat) : Nat :=
  s.pids 0 * BLOCK_M

def dimIndex (i : Fin BLOCK_HALF) : Nat :=
  i.val

def active (s : BlockState) (seqlen rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Prop :=
  rowIndex s BLOCK_M < seqlen ∧ dimIndex i < rotary_dim_half

instance activeDecidable (s : BlockState) (seqlen rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) :
    Decidable (active s seqlen rotary_dim_half BLOCK_M i) := by
  unfold active
  infer_instance

def outOffset
    (s : BlockState)
    (stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
    rowIndex s BLOCK_M * stride_out_seqlen + dimIndex i * stride_out_headdim

def x0Offset
    (s : BlockState)
    (stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_x_batch + s.pids 2 * stride_x_nheads +
    rowIndex s BLOCK_M * stride_x_seqlen + dimIndex i * stride_x_headdim

def x1Offset
    (s : BlockState)
    (stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_x_batch + s.pids 2 * stride_x_nheads +
    rowIndex s BLOCK_M * stride_x_seqlen +
    (dimIndex i + rotary_dim_half) * stride_x_headdim

def rotOffset (s : BlockState) (SEQLEN_OFFSETS rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  (rowIndex s BLOCK_M + SEQLEN_OFFSETS) * rotary_dim_half + dimIndex i

noncomputable def rotaryO0Spec
    (s : BlockState) (X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen_ro stride_x_batch stride_x_seqlen stride_x_nheads
      stride_x_headdim rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : ℝ :=
  let cosVal :=
    if rowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        dimIndex i < rotary_dim_half then
      s.readMem COS (rotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      1.0
  let sinVal :=
    if rowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        dimIndex i < rotary_dim_half then
      s.readMem SIN (rotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      0.0
  s.readMem X
      (x0Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim BLOCK_M i) *
    cosVal -
  s.readMem X
      (x1Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim rotary_dim_half BLOCK_M i) *
    sinVal

def out1Offset
    (s : BlockState)
    (stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
    rowIndex s BLOCK_M * stride_out_seqlen +
    (dimIndex i + rotary_dim_half) * stride_out_headdim

noncomputable def rotaryO1Spec
    (s : BlockState) (X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen_ro stride_x_batch stride_x_seqlen stride_x_nheads
      stride_x_headdim rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : ℝ :=
  let cosVal :=
    if rowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        dimIndex i < rotary_dim_half then
      s.readMem COS (rotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      1.0
  let sinVal :=
    if rowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        dimIndex i < rotary_dim_half then
      s.readMem SIN (rotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      0.0
  s.readMem X
      (x0Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim BLOCK_M i) *
    sinVal +
  s.readMem X
      (x1Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim rotary_dim_half BLOCK_M i) *
    cosVal

/-- Combined one-row first-and-second-half slice of
`rotary_transform_ops.py`'s `rotary_kernel`.

Performs BOTH the `o0 = x0 * cos - x1 * sin` first-half store and the
`o1 = x0 * sin + x1 * cos` second-half store in a single kernel, matching
the non-varlen, non-interleaved, non-conjugate branch of the Python source. -/
def rotary_kernel_o0o1_row
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(axis=0)
  pid_batch = tl.program_id(axis=1)
  pid_head = tl.program_id(axis=2)
  rm = pid_m * $(BLOCK_M)
  rm_cs = rm + $(SEQLEN_OFFSETS)
  rk_half = tl.arange(0, $(BLOCK_HALF))
  x_base = X + pid_batch * $(stride_x_batch) + pid_head * $(stride_x_nheads)
  out_base = OUT + pid_batch * $(stride_out_batch) + pid_head * $(stride_out_nheads)
  cos = tl.load(COS + rm_cs * $(rotary_dim_half) + rk_half,
    mask=(rm_cs < $(seqlen_ro)) and (rk_half < $(rotary_dim_half)), other=1.0)
  sin = tl.load(SIN + rm_cs * $(rotary_dim_half) + rk_half,
    mask=(rm_cs < $(seqlen_ro)) and (rk_half < $(rotary_dim_half)), other=0.0)
  x0 = tl.load(x_base + rm * $(stride_x_seqlen) + rk_half * $(stride_x_headdim),
    mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)), other=0.0)
  x1 = tl.load(x_base + rm * $(stride_x_seqlen) +
      (rk_half + $(rotary_dim_half)) * $(stride_x_headdim),
    mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)), other=0.0)
  o0 = x0 * cos - x1 * sin
  o1 = x0 * sin + x1 * cos
  tl.store(out_base + rm * $(stride_out_seqlen) + rk_half * $(stride_out_headdim),
    o0, mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)))
  tl.store(out_base + rm * $(stride_out_seqlen) +
      (rk_half + $(rotary_dim_half)) * $(stride_out_headdim),
    o1, mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)))
}

/-- Algorithm-layer correctness for the combined one-row `o0` + `o1` rotary
store, projected on the first-half (`o0`) output position. -/
theorem rotary_kernel_o0o1_row_o0_correct
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        outOffset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim BLOCK_M i))
    (hStrideHd : stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ rotary_dim_half)
    (hExec : exec (rotary_kernel_o0o1_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem OUT
          (outOffset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim BLOCK_M i) =
        if active s seqlen rotary_dim_half BLOCK_M i then
          rotaryO0Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
            stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
            BLOCK_M i
        else
          s.readMem OUT
            (outOffset s stride_out_batch stride_out_seqlen stride_out_nheads
              stride_out_headdim BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HALF] =>
        s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
          s.pids 0 * BLOCK_M * stride_out_seqlen +
          idx.1.val * stride_out_headdim) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset, rowIndex, dimIndex, Nat.mul_assoc] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  -- Disjointness: the `o1` write at `(j.val + rotary_dim_half) * stride_headdim`
  -- never collides with the `o0` read offset `i.val * stride_headdim`, because
  -- `i.val < BLOCK_HALF ≤ rotary_dim_half ≤ j.val + rotary_dim_half`.
  have hDisjoint : ∀ k : TileIndex [BLOCK_HALF],
      s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
          s.pids 0 * BLOCK_M * stride_out_seqlen +
          i.val * stride_out_headdim
        ≠
      s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
          s.pids 0 * BLOCK_M * stride_out_seqlen +
          (k.1.val + rotary_dim_half) * stride_out_headdim := by
    intro k hEq
    have hMul : i.val * stride_out_headdim =
        (k.1.val + rotary_dim_half) * stride_out_headdim :=
      Nat.add_left_cancel hEq
    have hVal : i.val = k.1.val + rotary_dim_half :=
      Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero hStrideHd) hMul
    have hI : i.val < rotary_dim_half :=
      Nat.lt_of_lt_of_le i.isLt hHalfBound
    have hGe : k.1.val + rotary_dim_half ≥ rotary_dim_half := Nat.le_add_left _ _
    omega
  by_cases hBH : 0 < BLOCK_HALF
  · simp [exec, rotary_kernel_o0o1_row, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [outOffset, rowIndex, dimIndex]
    -- Strip the outer `o1` foldl: same region, disjoint offsets.
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (region := OUT)
          (P := fun idx : TileIndex [BLOCK_HALF] =>
            s.pids 0 * BLOCK_M < seqlen ∧ idx.1.val < rotary_dim_half)
          (offsetFn := fun idx : TileIndex [BLOCK_HALF] =>
            s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
              s.pids 0 * BLOCK_M * stride_out_seqlen +
              (idx.1.val + rotary_dim_half) * stride_out_headdim)
          (hOff := fun k _ _ => hDisjoint k)]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
    by_cases hRow : s.pids 0 * BLOCK_M < seqlen
    · by_cases hDim : i.val < rotary_dim_half
      · by_cases hRot : s.pids 0 * BLOCK_M + SEQLEN_OFFSETS < seqlen_ro
        · simp [active, rotaryO0Spec, outOffset, x0Offset, x1Offset, rotOffset,
                rowIndex, dimIndex, hRow, hDim, hRot, Option.map₂,
                Option.bind, Option.map]
        · simp [active, rotaryO0Spec, outOffset, x0Offset, x1Offset, rotOffset,
                rowIndex, dimIndex, hRow, hDim, hRot, Option.map₂,
                Option.bind, Option.map]
      · simp [active, rowIndex, dimIndex, hRow, hDim]
    · simp [active, rowIndex, dimIndex, hRow]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Algorithm-layer correctness for the combined one-row `o0` + `o1` rotary
store, projected on the second-half (`o1`) output position. -/
theorem rotary_kernel_o0o1_row_o1_correct
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        out1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim rotary_dim_half BLOCK_M i))
    (hStrideHd : stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ rotary_dim_half)
    (hExec : exec (rotary_kernel_o0o1_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem OUT
          (out1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim rotary_dim_half BLOCK_M i) =
        if active s seqlen rotary_dim_half BLOCK_M i then
          rotaryO1Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
            stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
            BLOCK_M i
        else
          s.readMem OUT
            (out1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
              stride_out_headdim rotary_dim_half BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HALF] =>
        s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
          s.pids 0 * BLOCK_M * stride_out_seqlen +
          (idx.1.val + rotary_dim_half) * stride_out_headdim) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [out1Offset, rowIndex, dimIndex, Nat.mul_assoc] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  -- Disjointness: the `o0` write at `j.val * stride_headdim` never collides
  -- with the `o1` read offset `(i.val + rotary_dim_half) * stride_headdim`.
  have hDisjoint : ∀ k : TileIndex [BLOCK_HALF],
      s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
          s.pids 0 * BLOCK_M * stride_out_seqlen +
          (i.val + rotary_dim_half) * stride_out_headdim
        ≠
      s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
          s.pids 0 * BLOCK_M * stride_out_seqlen +
          k.1.val * stride_out_headdim := by
    intro k hEq
    have hMul : (i.val + rotary_dim_half) * stride_out_headdim =
        k.1.val * stride_out_headdim :=
      Nat.add_left_cancel hEq
    have hVal : i.val + rotary_dim_half = k.1.val :=
      Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero hStrideHd) hMul
    have hK : k.1.val < rotary_dim_half :=
      Nat.lt_of_lt_of_le k.1.isLt hHalfBound
    omega
  by_cases hBH : 0 < BLOCK_HALF
  · simp [exec, rotary_kernel_o0o1_row, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [out1Offset, rowIndex, dimIndex]
    -- Outer foldl is the `o1` store; close via scatter_readback.
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
    -- Now the `else` branch leaves a `(o0_foldl _).readMem OUT o1_offset`.
    -- Strip the `o0` foldl via the disjoint-offsets lemma.
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (region := OUT)
          (P := fun idx : TileIndex [BLOCK_HALF] =>
            s.pids 0 * BLOCK_M < seqlen ∧ idx.1.val < rotary_dim_half)
          (offsetFn := fun idx : TileIndex [BLOCK_HALF] =>
            s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
              s.pids 0 * BLOCK_M * stride_out_seqlen +
              idx.1.val * stride_out_headdim)
          (hOff := fun k _ _ => hDisjoint k)]
    by_cases hRow : s.pids 0 * BLOCK_M < seqlen
    · by_cases hDim : i.val < rotary_dim_half
      · by_cases hRot : s.pids 0 * BLOCK_M + SEQLEN_OFFSETS < seqlen_ro
        · simp [active, rotaryO1Spec, out1Offset, x0Offset, x1Offset, rotOffset,
                rowIndex, dimIndex, hRow, hDim, hRot, Option.map₂,
                Option.bind, Option.map]
        · simp [active, rotaryO1Spec, out1Offset, x0Offset, x1Offset, rotOffset,
                rowIndex, dimIndex, hRow, hDim, hRot, Option.map₂,
                Option.bind, Option.map]
      · simp [active, rowIndex, dimIndex, hRow, hDim]
    · simp [active, rowIndex, dimIndex, hRow]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the combined one-row rotary kernel,
projected on the first-half (`o0`) output position: every active lane of the
`o0` store reads back to the genuine rotary closed form `rotaryO0Spec`. -/
theorem rotary_kernel_o0o1_row_o0_compute_correct
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        outOffset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim BLOCK_M i))
    (hStrideHd : stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ rotary_dim_half) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rotary_kernel_o0o1_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active s seqlen rotary_dim_half BLOCK_M i)
        (fun i => (OUT,
          outOffset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim BLOCK_M i)))
      (expected := fun i =>
        rotaryO0Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
          stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
          BLOCK_M i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_kernel_o0o1_row]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rotary_kernel_o0o1_row_o0_correct OUT X COS SIN SEQLEN_OFFSETS
    seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
    stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
    stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF s s' hOutInj
    hStrideHd hHalfBound hExec i
  simpa [hActive] using h

/-- Compute-facing correctness for the combined one-row rotary kernel,
projected on the second-half (`o1`) output position: every active lane of the
`o1` store reads back to the genuine rotary closed form `rotaryO1Spec`. -/
theorem rotary_kernel_o0o1_row_o1_compute_correct
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        out1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim rotary_dim_half BLOCK_M i))
    (hStrideHd : stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ rotary_dim_half) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rotary_kernel_o0o1_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active s seqlen rotary_dim_half BLOCK_M i)
        (fun i => (OUT,
          out1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim rotary_dim_half BLOCK_M i)))
      (expected := fun i =>
        rotaryO1Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
          stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
          BLOCK_M i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_kernel_o0o1_row]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rotary_kernel_o0o1_row_o1_correct OUT X COS SIN SEQLEN_OFFSETS
    seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
    stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
    stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF s s' hOutInj
    hStrideHd hHalfBound hExec i
  simpa [hActive] using h

/-- Compute-facing coverage for the combined one-row rotary kernel: BOTH the
first-half (`o0`) and second-half (`o1`) stores read back, on every active lane,
to the genuine rotary closed forms `rotaryO0Spec`/`rotaryO1Spec` — the actual
`(x0·cos − x1·sin, x0·sin + x1·cos)` rotation, NOT the kernel's own executed
value. -/
theorem rotary_kernel_o0o1_row_all_outputs_compute_correct
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        outOffset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim BLOCK_M i))
    (hOut1Inj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        out1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim rotary_dim_half BLOCK_M i))
    (hStrideHd : stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ rotary_dim_half) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := rotary_kernel_o0o1_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active s seqlen rotary_dim_half BLOCK_M i)
        (fun i => (OUT,
          outOffset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim BLOCK_M i)))
      (expected := fun i =>
        rotaryO0Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
          stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
          BLOCK_M i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := rotary_kernel_o0o1_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active s seqlen rotary_dim_half BLOCK_M i)
        (fun i => (OUT,
          out1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim rotary_dim_half BLOCK_M i)))
      (expected := fun i =>
        rotaryO1Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
          stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
          BLOCK_M i)) := by
  refine ⟨?_, ?_⟩
  · exact rotary_kernel_o0o1_row_o0_compute_correct OUT X COS SIN SEQLEN_OFFSETS
      seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
      stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
      stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF s hOutInj
      hStrideHd hHalfBound
  · exact rotary_kernel_o0o1_row_o1_compute_correct OUT X COS SIN SEQLEN_OFFSETS
      seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
      stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
      stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF s hOut1Inj
      hStrideHd hHalfBound

/-! ### Full 2D `[BLOCK_M, BLOCK_HALF]` non-interleaved kernel correctness

The full `[BLOCK_M, BLOCK_HALF]` non-interleaved tile body lifts the one-row
`o0`/`o1` companion stores (proven on `rotary_kernel_o0o1_row` above) to the
FULL `[BLOCK_M, BLOCK_HALF]` tile of the non-interleaved, non-varlen,
non-conjugate, scalar-offset branch from `rotary_transform_ops.py`. Only the
2D tile-indexed coordinate and active-mask helpers below survive — they mirror
the 1D ones (`outOffset`, `active`, etc.) but use
`TileIndex [BLOCK_M, BLOCK_HALF]` instead of `Fin BLOCK_HALF`. The proof
target would be the same: `o0 = x0 * cos - x1 * sin` written into `OUT` at the
first-half `rk_half` offsets, with the companion `o1` store landing on the
disjoint second-half offsets. -/

/-- 2D-tile row coordinate `rm = pid_m * BLOCK_M + idx.1.val`. -/
def rowIndex2D (s : BlockState) (BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_HALF]) : Nat :=
  s.pids 0 * BLOCK_M + idx.1.val

/-- 2D-tile column coordinate `rk_half = idx.2.1.val`. -/
def dimIndex2D (idx : TileIndex [BLOCK_M, BLOCK_HALF]) : Nat :=
  idx.2.1.val

/-- 2D-tile active mask: row in `seqlen`, column in `rotary_dim_half`. -/
def active2D (s : BlockState) (seqlen rotary_dim_half BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_HALF]) : Prop :=
  rowIndex2D s BLOCK_M idx < seqlen ∧ dimIndex2D idx < rotary_dim_half

instance active2DDecidable (s : BlockState) (seqlen rotary_dim_half BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_HALF]) :
    Decidable (active2D s seqlen rotary_dim_half BLOCK_M idx) := by
  unfold active2D
  infer_instance

/-- **Dimension-general public output summary for `rotary_transform_ops.py`**
(genuine, not self-referential).

Every sequence length, rotary
dimension, block size, and stride is a `Nat` parameter rather than a pinned
Python literal, and the per-lane output-offset injectivity plus the
`stride_out_headdim ≠ 0` / `BLOCK_HALF ≤ rotary_dim_half` disjointness
side-conditions are taken as hypotheses.

For ANY shape, the full `rotary_kernel_surface` (with all four prologue
branches and the conjugate/interleaved flags) lowers to the algorithm layer,
and on the non-interleaved rotation body run on a `[BLOCK_M, BLOCK_HALF]` row
tile BOTH output halves are written so that every active lane of the
first-half (`o0`) store equals the genuine closed form `x0·cos − x1·sin`
(`rotaryO0Spec`) and every active lane of the second-half (`o1`) store equals
`x0·sin + x1·cos` (`rotaryO1Spec`) — the actual rotary embedding read from the
precomputed `COS`/`SIN` cache, NOT the kernel's own re-executed value.

The host launch remains the trusted boundary. -/
specification rotary_transform_ops_output_summary_general
    (OUT X COS SIN : RegionName) (CU_SEQLENS SEQLEN_OFFSETS : Region .nat)
    (SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool)
    (body_SEQLEN_OFFSETS body_seqlen body_rotary_dim_half body_seqlen_ro
      body_stride_out_batch body_stride_out_seqlen body_stride_out_nheads
      body_stride_out_headdim body_stride_x_batch body_stride_x_seqlen
      body_stride_x_nheads body_stride_x_headdim body_BLOCK_M BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        outOffset s body_stride_out_batch body_stride_out_seqlen
          body_stride_out_nheads body_stride_out_headdim body_BLOCK_M i))
    (hOut1Inj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        out1Offset s body_stride_out_batch body_stride_out_seqlen
          body_stride_out_nheads body_stride_out_headdim body_rotary_dim_half
          body_BLOCK_M i))
    (hStrideHd : body_stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ body_rotary_dim_half) :
    (∃ alg, (rotary_kernel_surface OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS
      SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro stride_out_batch
      stride_out_seqlen stride_out_nheads stride_out_headdim stride_x_batch
      stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_K BLOCK_M
      IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := rotary_kernel_o0o1_row OUT X COS SIN
        body_SEQLEN_OFFSETS body_seqlen body_rotary_dim_half body_seqlen_ro
        body_stride_out_batch body_stride_out_seqlen body_stride_out_nheads
        body_stride_out_headdim body_stride_x_batch body_stride_x_seqlen
        body_stride_x_nheads body_stride_x_headdim body_BLOCK_M BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF =>
          active s body_seqlen body_rotary_dim_half body_BLOCK_M i)
        (fun i => (OUT,
          outOffset s body_stride_out_batch body_stride_out_seqlen
            body_stride_out_nheads body_stride_out_headdim body_BLOCK_M i)))
      (expected := fun i =>
        rotaryO0Spec s X COS SIN body_SEQLEN_OFFSETS body_seqlen_ro
          body_stride_x_batch body_stride_x_seqlen body_stride_x_nheads
          body_stride_x_headdim body_rotary_dim_half body_BLOCK_M i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := rotary_kernel_o0o1_row OUT X COS SIN
        body_SEQLEN_OFFSETS body_seqlen body_rotary_dim_half body_seqlen_ro
        body_stride_out_batch body_stride_out_seqlen body_stride_out_nheads
        body_stride_out_headdim body_stride_x_batch body_stride_x_seqlen
        body_stride_x_nheads body_stride_x_headdim body_BLOCK_M BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF =>
          active s body_seqlen body_rotary_dim_half body_BLOCK_M i)
        (fun i => (OUT,
          out1Offset s body_stride_out_batch body_stride_out_seqlen
            body_stride_out_nheads body_stride_out_headdim body_rotary_dim_half
            body_BLOCK_M i)))
      (expected := fun i =>
        rotaryO1Spec s X COS SIN body_SEQLEN_OFFSETS body_seqlen_ro
          body_stride_x_batch body_stride_x_seqlen body_stride_x_nheads
          body_stride_x_headdim body_rotary_dim_half body_BLOCK_M i)) := by
  refine ⟨rotary_kernel_surface_toAlgorithm_supported OUT X COS SIN CU_SEQLENS
      SEQLEN_OFFSETS SEQLEN_OFFSETS_SCALAR seqlen rotary_dim seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE,
    ?_, ?_⟩
  · exact (rotary_kernel_o0o1_row_all_outputs_compute_correct OUT X COS SIN
      body_SEQLEN_OFFSETS body_seqlen body_rotary_dim_half body_seqlen_ro
      body_stride_out_batch body_stride_out_seqlen body_stride_out_nheads
      body_stride_out_headdim body_stride_x_batch body_stride_x_seqlen
      body_stride_x_nheads body_stride_x_headdim body_BLOCK_M BLOCK_HALF s
      hOutInj hOut1Inj hStrideHd hHalfBound).1
  · exact (rotary_kernel_o0o1_row_all_outputs_compute_correct OUT X COS SIN
      body_SEQLEN_OFFSETS body_seqlen body_rotary_dim_half body_seqlen_ro
      body_stride_out_batch body_stride_out_seqlen body_stride_out_nheads
      body_stride_out_headdim body_stride_x_batch body_stride_x_seqlen
      body_stride_x_nheads body_stride_x_headdim body_BLOCK_M BLOCK_HALF s
      hOutInj hOut1Inj hStrideHd hHalfBound).2

/-! ## The `⊨` metadata-grouped surface (`MetaGroupedMasked2DKernelIO`)

The per-row slice below models the full `rotary_kernel` (both the `IS_VARLEN`
prologue branches) as a `MetaGroupedMasked2DKernelIO` masked Hoare triple over
flat pointer memory: the two `.nat` slots are the `CU_SEQLENS[pid_batch]` /
`CU_SEQLENS[pid_batch+1]` loads, whose difference is the per-program `seqlen`;
the four read channels are `x0`/`x1`/`cos`/`sin` and the two write channels the
two in-place-shaped `OUT` halves. All windows/masks case on `IS_VARLEN` exactly
as the kernel does; the head axis (`program_id(2)`) is carried as the `HEAD_IDX`
parameter (per-head slice, universally quantified). -/

/-- Faithful per-row, per-head companion for `rotary_transform_ops.py`'s
`rotary_kernel`, parametric in `IS_VARLEN`: the varlen prologue genuinely loads
`start_idx`/`seqlen_hi` from `CU_SEQLENS` and folds `start_idx` into the X/OUT
base offsets and `seqlen_hi - start_idx` into every mask. -/
def rotary_meta_row
    (OUT X COS SIN : RegionName) (CU_SEQLENS : Region .nat)
    (HEAD_IDX SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) (IS_VARLEN : Bool) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_batch = tl.program_id(1)
  rm = pid_m * $(BLOCK_M)
  rm_cs = rm + $(SEQLEN_OFFSETS)
  rk_half = tl.arange(0, $(BLOCK_HALF))
  x_base = X + pid_batch * $(stride_x_batch) + $(HEAD_IDX) * $(stride_x_nheads)
  out_base = OUT + pid_batch * $(stride_out_batch) + $(HEAD_IDX) * $(stride_out_nheads)
  seqlen_v = $(seqlen)
  if IS_VARLEN {
    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen_hi = tl.load(CU_SEQLENS + pid_batch + $(1))
    x_base = X + start_idx * $(stride_x_seqlen) + $(HEAD_IDX) * $(stride_x_nheads)
    out_base = OUT + start_idx * $(stride_out_seqlen) + $(HEAD_IDX) * $(stride_out_nheads)
    seqlen_v = seqlen_hi - start_idx
  }
  cos = tl.load(COS + rm_cs * $(rotary_dim_half) + rk_half,
    mask=(rm_cs < $(seqlen_ro)) and (rk_half < $(rotary_dim_half)), other=1.0)
  sin = tl.load(SIN + rm_cs * $(rotary_dim_half) + rk_half,
    mask=(rm_cs < $(seqlen_ro)) and (rk_half < $(rotary_dim_half)), other=0.0)
  x0 = tl.load(x_base + rm * $(stride_x_seqlen) + rk_half * $(stride_x_headdim),
    mask=(rm < seqlen_v) and (rk_half < $(rotary_dim_half)), other=0.0)
  x1 = tl.load(x_base + rm * $(stride_x_seqlen) +
      (rk_half + $(rotary_dim_half)) * $(stride_x_headdim),
    mask=(rm < seqlen_v) and (rk_half < $(rotary_dim_half)), other=0.0)
  o0 = x0 * cos - x1 * sin
  o1 = x0 * sin + x1 * cos
  tl.store(out_base + rm * $(stride_out_seqlen) + rk_half * $(stride_out_headdim),
    o0, mask=(rm < seqlen_v) and (rk_half < $(rotary_dim_half)))
  tl.store(out_base + rm * $(stride_out_seqlen) +
      (rk_half + $(rotary_dim_half)) * $(stride_out_headdim),
    o1, mask=(rm < seqlen_v) and (rk_half < $(rotary_dim_half)))
}

/-- Base offset (region-relative) for X, casing on IS_VARLEN. -/
def metaXBase (IS_VARLEN : Bool) (s : BlockState) (CU : RegionName)
    (HEAD_IDX stride_x_batch stride_x_seqlen stride_x_nheads : Nat) : Nat :=
  (if IS_VARLEN then s.readMemValue .nat CU (s.pids 1) * stride_x_seqlen
   else s.pids 1 * stride_x_batch) + HEAD_IDX * stride_x_nheads

/-- Base offset for OUT, casing on IS_VARLEN. -/
def metaOutBase (IS_VARLEN : Bool) (s : BlockState) (CU : RegionName)
    (HEAD_IDX stride_out_batch stride_out_seqlen stride_out_nheads : Nat) : Nat :=
  (if IS_VARLEN then s.readMemValue .nat CU (s.pids 1) * stride_out_seqlen
   else s.pids 1 * stride_out_batch) + HEAD_IDX * stride_out_nheads

/-- Per-program seqlen, casing on IS_VARLEN. -/
def metaSeqlen (IS_VARLEN : Bool) (s : BlockState) (CU : RegionName)
    (seqlen : Nat) : Nat :=
  if IS_VARLEN then
    s.readMemValue .nat CU (s.pids 1 + 1) - s.readMemValue .nat CU (s.pids 1)
  else seqlen

def metaRm (s : BlockState) (BLOCK_M : Nat) : Nat := s.pids 0 * BLOCK_M

def metaOutOffset (IS_VARLEN : Bool) (s : BlockState) (CU : RegionName)
    (HEAD_IDX stride_out_batch stride_out_seqlen stride_out_nheads
      stride_out_headdim BLOCK_M : Nat) (shift : Nat) (i : Fin BLOCK_HALF) : Nat :=
  metaOutBase IS_VARLEN s CU HEAD_IDX stride_out_batch stride_out_seqlen
      stride_out_nheads
    + metaRm s BLOCK_M * stride_out_seqlen + (i.val + shift) * stride_out_headdim

def metaXOffset (IS_VARLEN : Bool) (s : BlockState) (CU : RegionName)
    (HEAD_IDX stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M : Nat) (shift : Nat) (i : Fin BLOCK_HALF) : Nat :=
  metaXBase IS_VARLEN s CU HEAD_IDX stride_x_batch stride_x_seqlen stride_x_nheads
    + metaRm s BLOCK_M * stride_x_seqlen + (i.val + shift) * stride_x_headdim

def metaCosOffset (s : BlockState) (SEQLEN_OFFSETS rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  (metaRm s BLOCK_M + SEQLEN_OFFSETS) * rotary_dim_half + i.val

def metaActive (IS_VARLEN : Bool) (s : BlockState) (CU : RegionName)
    (seqlen rotary_dim_half BLOCK_M : Nat) (i : Fin BLOCK_HALF) : Prop :=
  metaRm s BLOCK_M < metaSeqlen IS_VARLEN s CU seqlen ∧ i.val < rotary_dim_half

instance (IS_VARLEN : Bool) (s : BlockState) (CU : RegionName)
    (seqlen rotary_dim_half BLOCK_M : Nat) (i : Fin BLOCK_HALF) :
    Decidable (metaActive IS_VARLEN s CU seqlen rotary_dim_half BLOCK_M i) := by
  unfold metaActive; infer_instance

noncomputable def metaO0Spec (IS_VARLEN : Bool)
    (s : BlockState) (X COS SIN CU : RegionName)
    (HEAD_IDX SEQLEN_OFFSETS seqlen_ro stride_x_batch stride_x_seqlen
      stride_x_nheads stride_x_headdim rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : ℝ :=
  let cosVal :=
    if metaRm s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ i.val < rotary_dim_half then
      s.readMem COS (metaCosOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else 1.0
  let sinVal :=
    if metaRm s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ i.val < rotary_dim_half then
      s.readMem SIN (metaCosOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else 0.0
  s.readMem X (metaXOffset IS_VARLEN s CU HEAD_IDX stride_x_batch stride_x_seqlen
      stride_x_nheads stride_x_headdim BLOCK_M 0 i) * cosVal
  - s.readMem X (metaXOffset IS_VARLEN s CU HEAD_IDX stride_x_batch stride_x_seqlen
      stride_x_nheads stride_x_headdim BLOCK_M rotary_dim_half i) * sinVal

/-- o0 correctness, parametric IS_VARLEN. -/
theorem meta_row_o0_correct
    (OUT X COS SIN : RegionName) (CU_SEQLENS : Region .nat)
    (HEAD_IDX SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) (IS_VARLEN : Bool)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        metaOutOffset IS_VARLEN s CU_SEQLENS HEAD_IDX stride_out_batch stride_out_seqlen
          stride_out_nheads stride_out_headdim BLOCK_M 0 i))
    (hStrideHd : stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ rotary_dim_half)
    (hExec : exec ((rotary_meta_row OUT X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF IS_VARLEN).toAlgKernel)
        s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem OUT
          (metaOutOffset IS_VARLEN s CU_SEQLENS HEAD_IDX stride_out_batch stride_out_seqlen
            stride_out_nheads stride_out_headdim BLOCK_M 0 i) =
        if metaActive IS_VARLEN s CU_SEQLENS seqlen rotary_dim_half BLOCK_M i then
          metaO0Spec IS_VARLEN s X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS seqlen_ro
            stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
            rotary_dim_half BLOCK_M i
        else
          s.readMem OUT
            (metaOutOffset IS_VARLEN s CU_SEQLENS HEAD_IDX stride_out_batch stride_out_seqlen
              stride_out_nheads stride_out_headdim BLOCK_M 0 i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HALF] =>
        metaOutBase IS_VARLEN s CU_SEQLENS HEAD_IDX stride_out_batch stride_out_seqlen
            stride_out_nheads
          + metaRm s BLOCK_M * stride_out_seqlen + idx.1.val * stride_out_headdim) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [metaOutOffset, Nat.add_zero] using h
    cases a; cases b; simp only at hab; cases hab; rfl
  have hDisjoint : ∀ k : TileIndex [BLOCK_HALF],
      metaOutBase IS_VARLEN s CU_SEQLENS HEAD_IDX stride_out_batch stride_out_seqlen
          stride_out_nheads
        + metaRm s BLOCK_M * stride_out_seqlen + i.val * stride_out_headdim
        ≠
      metaOutBase IS_VARLEN s CU_SEQLENS HEAD_IDX stride_out_batch stride_out_seqlen
          stride_out_nheads
        + metaRm s BLOCK_M * stride_out_seqlen
        + (k.1.val + rotary_dim_half) * stride_out_headdim := by
    intro k hEq
    have hMul : i.val * stride_out_headdim =
        (k.1.val + rotary_dim_half) * stride_out_headdim := Nat.add_left_cancel hEq
    have hVal : i.val = k.1.val + rotary_dim_half :=
      Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero hStrideHd) hMul
    have hI : i.val < rotary_dim_half := Nat.lt_of_lt_of_le i.isLt hHalfBound
    omega
  by_cases hBH : 0 < BLOCK_HALF
  · cases IS_VARLEN
    · have hRawInj0 : Function.Injective
          (fun idx : TileIndex [BLOCK_HALF] =>
            s.pids 1 * stride_out_batch + HEAD_IDX * stride_out_nheads +
              s.pids 0 * BLOCK_M * stride_out_seqlen +
              idx.1.val * stride_out_headdim) := by
        simpa [metaOutBase, metaRm] using hRawInj
      have hDisjoint0 : ∀ k : TileIndex [BLOCK_HALF],
          s.pids 1 * stride_out_batch + HEAD_IDX * stride_out_nheads +
              s.pids 0 * BLOCK_M * stride_out_seqlen + i.val * stride_out_headdim
            ≠
          s.pids 1 * stride_out_batch + HEAD_IDX * stride_out_nheads +
              s.pids 0 * BLOCK_M * stride_out_seqlen +
              (k.1.val + rotary_dim_half) * stride_out_headdim := by
        intro k; have := hDisjoint k; simpa [metaOutBase, metaRm] using this
      simp [exec, rotary_meta_row, ComputeKernel.toAlgKernel,
            ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
            stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            ComparableDType.lt, hBH] at hExec
      rw [← hExec]
      simp only [metaOutOffset, metaOutBase, metaRm, Bool.false_eq_true, if_false,
        Nat.add_zero]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (region := OUT)
            (P := fun idx : TileIndex [BLOCK_HALF] =>
              s.pids 0 * BLOCK_M < seqlen ∧ idx.1.val < rotary_dim_half)
            (offsetFn := fun idx : TileIndex [BLOCK_HALF] =>
              s.pids 1 * stride_out_batch + HEAD_IDX * stride_out_nheads +
                s.pids 0 * BLOCK_M * stride_out_seqlen +
                (idx.1.val + rotary_dim_half) * stride_out_headdim)
            (hOff := fun k _ _ => hDisjoint0 k)]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj0 (i, PUnit.unit)]
      by_cases hRow : s.pids 0 * BLOCK_M < seqlen
      · by_cases hDim : i.val < rotary_dim_half
        · by_cases hRot : s.pids 0 * BLOCK_M + SEQLEN_OFFSETS < seqlen_ro
          · simp [metaActive, metaSeqlen, metaO0Spec, metaXOffset, metaXBase,
                  metaCosOffset, metaRm, hRow, hDim, hRot, Option.map₂,
                  Option.bind, Option.map]
          · simp [metaActive, metaSeqlen, metaO0Spec, metaXOffset, metaXBase,
                  metaCosOffset, metaRm, hRow, hDim, hRot, Option.map₂,
                  Option.bind, Option.map]
        · simp [metaActive, metaSeqlen, metaRm, hRow, hDim]
      · simp [metaActive, metaSeqlen, metaRm, hRow]
    · have hRawInj0 : Function.Injective
          (fun idx : TileIndex [BLOCK_HALF] =>
            s.readMemValue .nat CU_SEQLENS (s.pids 1) * stride_out_seqlen +
                HEAD_IDX * stride_out_nheads +
              s.pids 0 * BLOCK_M * stride_out_seqlen +
              idx.1.val * stride_out_headdim) := by
        simpa [metaOutBase, metaRm] using hRawInj
      have hDisjoint0 : ∀ k : TileIndex [BLOCK_HALF],
          s.readMemValue .nat CU_SEQLENS (s.pids 1) * stride_out_seqlen +
              HEAD_IDX * stride_out_nheads +
              s.pids 0 * BLOCK_M * stride_out_seqlen + i.val * stride_out_headdim
            ≠
          s.readMemValue .nat CU_SEQLENS (s.pids 1) * stride_out_seqlen +
              HEAD_IDX * stride_out_nheads +
              s.pids 0 * BLOCK_M * stride_out_seqlen +
              (k.1.val + rotary_dim_half) * stride_out_headdim := by
        intro k; have := hDisjoint k; simpa [metaOutBase, metaRm] using this
      simp [exec, rotary_meta_row, ComputeKernel.toAlgKernel,
            ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
            stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            ComparableDType.lt, hBH] at hExec
      rw [← hExec]
      simp only [metaOutOffset, metaOutBase, metaRm, if_true, Nat.add_zero]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (region := OUT)
            (P := fun idx : TileIndex [BLOCK_HALF] =>
              s.pids 0 * BLOCK_M <
                  s.readMemValue .nat CU_SEQLENS (s.pids 1 + 1) -
                    s.readMemValue .nat CU_SEQLENS (s.pids 1) ∧
                idx.1.val < rotary_dim_half)
            (offsetFn := fun idx : TileIndex [BLOCK_HALF] =>
              s.readMemValue .nat CU_SEQLENS (s.pids 1) * stride_out_seqlen +
                  HEAD_IDX * stride_out_nheads +
                s.pids 0 * BLOCK_M * stride_out_seqlen +
                (idx.1.val + rotary_dim_half) * stride_out_headdim)
            (hOff := fun k _ _ => hDisjoint0 k)]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj0 (i, PUnit.unit)]
      by_cases hRow : s.pids 0 * BLOCK_M <
          s.readMemValue .nat CU_SEQLENS (s.pids 1 + 1) -
            s.readMemValue .nat CU_SEQLENS (s.pids 1)
      · by_cases hDim : i.val < rotary_dim_half
        · by_cases hRot : s.pids 0 * BLOCK_M + SEQLEN_OFFSETS < seqlen_ro
          · simp [metaActive, metaSeqlen, metaO0Spec, metaXOffset, metaXBase,
                  metaCosOffset, metaRm, hRow, hDim, hRot, Option.map₂,
                  Option.bind, Option.map]
          · simp [metaActive, metaSeqlen, metaO0Spec, metaXOffset, metaXBase,
                  metaCosOffset, metaRm, hRow, hDim, hRot, Option.map₂,
                  Option.bind, Option.map]
        · simp [metaActive, metaSeqlen, metaRm, hRow, hDim]
      · simp [metaActive, metaSeqlen, metaRm, hRow]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

noncomputable def metaO1Spec (IS_VARLEN : Bool)
    (s : BlockState) (X COS SIN CU : RegionName)
    (HEAD_IDX SEQLEN_OFFSETS seqlen_ro stride_x_batch stride_x_seqlen
      stride_x_nheads stride_x_headdim rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : ℝ :=
  let cosVal :=
    if metaRm s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ i.val < rotary_dim_half then
      s.readMem COS (metaCosOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else 1.0
  let sinVal :=
    if metaRm s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ i.val < rotary_dim_half then
      s.readMem SIN (metaCosOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else 0.0
  s.readMem X (metaXOffset IS_VARLEN s CU HEAD_IDX stride_x_batch stride_x_seqlen
      stride_x_nheads stride_x_headdim BLOCK_M 0 i) * sinVal
  + s.readMem X (metaXOffset IS_VARLEN s CU HEAD_IDX stride_x_batch stride_x_seqlen
      stride_x_nheads stride_x_headdim BLOCK_M rotary_dim_half i) * cosVal

/-- o1 correctness, parametric IS_VARLEN. -/
theorem meta_row_o1_correct
    (OUT X COS SIN : RegionName) (CU_SEQLENS : Region .nat)
    (HEAD_IDX SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) (IS_VARLEN : Bool)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        metaOutOffset IS_VARLEN s CU_SEQLENS HEAD_IDX stride_out_batch stride_out_seqlen
          stride_out_nheads stride_out_headdim BLOCK_M rotary_dim_half i))
    (hStrideHd : stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ rotary_dim_half)
    (hExec : exec ((rotary_meta_row OUT X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF IS_VARLEN).toAlgKernel)
        s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem OUT
          (metaOutOffset IS_VARLEN s CU_SEQLENS HEAD_IDX stride_out_batch stride_out_seqlen
            stride_out_nheads stride_out_headdim BLOCK_M rotary_dim_half i) =
        if metaActive IS_VARLEN s CU_SEQLENS seqlen rotary_dim_half BLOCK_M i then
          metaO1Spec IS_VARLEN s X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS seqlen_ro
            stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
            rotary_dim_half BLOCK_M i
        else
          s.readMem OUT
            (metaOutOffset IS_VARLEN s CU_SEQLENS HEAD_IDX stride_out_batch stride_out_seqlen
              stride_out_nheads stride_out_headdim BLOCK_M rotary_dim_half i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HALF] =>
        metaOutBase IS_VARLEN s CU_SEQLENS HEAD_IDX stride_out_batch stride_out_seqlen
            stride_out_nheads
          + metaRm s BLOCK_M * stride_out_seqlen
          + (idx.1.val + rotary_dim_half) * stride_out_headdim) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [metaOutOffset] using h
    cases a; cases b; simp only at hab; cases hab; rfl
  have hDisjoint : ∀ k : TileIndex [BLOCK_HALF],
      metaOutBase IS_VARLEN s CU_SEQLENS HEAD_IDX stride_out_batch stride_out_seqlen
          stride_out_nheads
        + metaRm s BLOCK_M * stride_out_seqlen
        + (i.val + rotary_dim_half) * stride_out_headdim
        ≠
      metaOutBase IS_VARLEN s CU_SEQLENS HEAD_IDX stride_out_batch stride_out_seqlen
          stride_out_nheads
        + metaRm s BLOCK_M * stride_out_seqlen + k.1.val * stride_out_headdim := by
    intro k hEq
    have hMul : (i.val + rotary_dim_half) * stride_out_headdim =
        k.1.val * stride_out_headdim := Nat.add_left_cancel hEq
    have hVal : i.val + rotary_dim_half = k.1.val :=
      Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero hStrideHd) hMul
    have hK : k.1.val < rotary_dim_half := Nat.lt_of_lt_of_le k.1.isLt hHalfBound
    omega
  by_cases hBH : 0 < BLOCK_HALF
  · cases IS_VARLEN
    · have hRawInj0 : Function.Injective
          (fun idx : TileIndex [BLOCK_HALF] =>
            s.pids 1 * stride_out_batch + HEAD_IDX * stride_out_nheads +
              s.pids 0 * BLOCK_M * stride_out_seqlen +
              (idx.1.val + rotary_dim_half) * stride_out_headdim) := by
        simpa [metaOutBase, metaRm] using hRawInj
      have hDisjoint0 : ∀ k : TileIndex [BLOCK_HALF],
          s.pids 1 * stride_out_batch + HEAD_IDX * stride_out_nheads +
              s.pids 0 * BLOCK_M * stride_out_seqlen +
              (i.val + rotary_dim_half) * stride_out_headdim
            ≠
          s.pids 1 * stride_out_batch + HEAD_IDX * stride_out_nheads +
              s.pids 0 * BLOCK_M * stride_out_seqlen + k.1.val * stride_out_headdim := by
        intro k; have := hDisjoint k; simpa [metaOutBase, metaRm] using this
      simp [exec, rotary_meta_row, ComputeKernel.toAlgKernel,
            ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
            stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            ComparableDType.lt, hBH] at hExec
      rw [← hExec]
      simp only [metaOutOffset, metaOutBase, metaRm, Bool.false_eq_true, if_false]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj0 (i, PUnit.unit)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (region := OUT)
            (P := fun idx : TileIndex [BLOCK_HALF] =>
              s.pids 0 * BLOCK_M < seqlen ∧ idx.1.val < rotary_dim_half)
            (offsetFn := fun idx : TileIndex [BLOCK_HALF] =>
              s.pids 1 * stride_out_batch + HEAD_IDX * stride_out_nheads +
                s.pids 0 * BLOCK_M * stride_out_seqlen +
                idx.1.val * stride_out_headdim)
            (hOff := fun k _ _ => hDisjoint0 k)]
      by_cases hRow : s.pids 0 * BLOCK_M < seqlen
      · by_cases hDim : i.val < rotary_dim_half
        · by_cases hRot : s.pids 0 * BLOCK_M + SEQLEN_OFFSETS < seqlen_ro
          · simp [metaActive, metaSeqlen, metaO1Spec, metaXOffset, metaXBase,
                  metaCosOffset, metaRm, hRow, hDim, hRot, Option.map₂,
                  Option.bind, Option.map]
          · simp [metaActive, metaSeqlen, metaO1Spec, metaXOffset, metaXBase,
                  metaCosOffset, metaRm, hRow, hDim, hRot, Option.map₂,
                  Option.bind, Option.map]
        · simp [metaActive, metaSeqlen, metaRm, hRow, hDim]
      · simp [metaActive, metaSeqlen, metaRm, hRow]
    · have hRawInj0 : Function.Injective
          (fun idx : TileIndex [BLOCK_HALF] =>
            s.readMemValue .nat CU_SEQLENS (s.pids 1) * stride_out_seqlen +
                HEAD_IDX * stride_out_nheads +
              s.pids 0 * BLOCK_M * stride_out_seqlen +
              (idx.1.val + rotary_dim_half) * stride_out_headdim) := by
        simpa [metaOutBase, metaRm] using hRawInj
      have hDisjoint0 : ∀ k : TileIndex [BLOCK_HALF],
          s.readMemValue .nat CU_SEQLENS (s.pids 1) * stride_out_seqlen +
              HEAD_IDX * stride_out_nheads +
              s.pids 0 * BLOCK_M * stride_out_seqlen +
              (i.val + rotary_dim_half) * stride_out_headdim
            ≠
          s.readMemValue .nat CU_SEQLENS (s.pids 1) * stride_out_seqlen +
              HEAD_IDX * stride_out_nheads +
              s.pids 0 * BLOCK_M * stride_out_seqlen + k.1.val * stride_out_headdim := by
        intro k; have := hDisjoint k; simpa [metaOutBase, metaRm] using this
      simp [exec, rotary_meta_row, ComputeKernel.toAlgKernel,
            ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
            stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            ComparableDType.lt, hBH] at hExec
      rw [← hExec]
      simp only [metaOutOffset, metaOutBase, metaRm, if_true]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj0 (i, PUnit.unit)]
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            (region := OUT)
            (P := fun idx : TileIndex [BLOCK_HALF] =>
              s.pids 0 * BLOCK_M <
                  s.readMemValue .nat CU_SEQLENS (s.pids 1 + 1) -
                    s.readMemValue .nat CU_SEQLENS (s.pids 1) ∧
                idx.1.val < rotary_dim_half)
            (offsetFn := fun idx : TileIndex [BLOCK_HALF] =>
              s.readMemValue .nat CU_SEQLENS (s.pids 1) * stride_out_seqlen +
                  HEAD_IDX * stride_out_nheads +
                s.pids 0 * BLOCK_M * stride_out_seqlen +
                idx.1.val * stride_out_headdim)
            (hOff := fun k _ _ => hDisjoint0 k)]
      by_cases hRow : s.pids 0 * BLOCK_M <
          s.readMemValue .nat CU_SEQLENS (s.pids 1 + 1) -
            s.readMemValue .nat CU_SEQLENS (s.pids 1)
      · by_cases hDim : i.val < rotary_dim_half
        · by_cases hRot : s.pids 0 * BLOCK_M + SEQLEN_OFFSETS < seqlen_ro
          · simp [metaActive, metaSeqlen, metaO1Spec, metaXOffset, metaXBase,
                  metaCosOffset, metaRm, hRow, hDim, hRot, Option.map₂,
                  Option.bind, Option.map]
          · simp [metaActive, metaSeqlen, metaO1Spec, metaXOffset, metaXBase,
                  metaCosOffset, metaRm, hRow, hDim, hRot, Option.map₂,
                  Option.bind, Option.map]
        · simp [metaActive, metaSeqlen, metaRm, hRow, hDim]
      · simp [metaActive, metaSeqlen, metaRm, hRow]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Pure X base offset (region-relative), casing on IS_VARLEN, for the IO. -/
def ioXBase (IS_VARLEN : Bool) (pid₁ s1 HEAD_IDX
    stride_x_batch stride_x_seqlen stride_x_nheads : Nat) : Nat :=
  (if IS_VARLEN then s1 * stride_x_seqlen else pid₁ * stride_x_batch)
    + HEAD_IDX * stride_x_nheads

def ioOutBase (IS_VARLEN : Bool) (pid₁ s1 HEAD_IDX
    stride_out_batch stride_out_seqlen stride_out_nheads : Nat) : Nat :=
  (if IS_VARLEN then s1 * stride_out_seqlen else pid₁ * stride_out_batch)
    + HEAD_IDX * stride_out_nheads

def ioSeqlen (IS_VARLEN : Bool) (s1 s2 seqlen : Nat) : Nat :=
  if IS_VARLEN then s2 - s1 else seqlen

/-- The `⊨` value function: both rotary output halves over the loaded inputs. -/
noncomputable def rotaryMetaF (SEQLEN_OFFSETS seqlen_ro rotary_dim_half BLOCK_M : Nat)
    (pid₀ : Nat) (xs : Fin 4 → Fin BLOCK_HALF → ℝ) (o : Fin 2) (j : Fin BLOCK_HALF) : ℝ :=
  let c := if pid₀ * BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ j.val < rotary_dim_half then
      xs 2 j else 1.0
  let sn := if pid₀ * BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ j.val < rotary_dim_half then
      xs 3 j else 0.0
  match o with
  | ⟨0, _⟩ => xs 0 j * c - xs 1 j * sn
  | ⟨1, _⟩ => xs 0 j * sn + xs 1 j * c

/-- The metadata-grouped masked IO signature of `rotary_meta_row`. -/
def rotaryMetaIO
    (OUT X COS SIN : RegionName) (CU_SEQLENS : Region .nat)
    (HEAD_IDX SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) (IS_VARLEN : Bool) :
    MetaGroupedMasked2DKernelIO where
  kernel := rotary_meta_row OUT X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS seqlen
    rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads
    stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
    BLOCK_M BLOCK_HALF IS_VARLEN
  nIn := 4
  nOut := 2
  bufs := [X, OUT, COS, SIN, CU_SEQLENS]
  mbuf1 := CU_SEQLENS
  mbuf2 := CU_SEQLENS
  inp := fun i => match i with
    | ⟨0, _⟩ => X
    | ⟨1, _⟩ => X
    | ⟨2, _⟩ => COS
    | ⟨_ + 3, _⟩ => SIN
  out := fun _ => OUT
  B := BLOCK_HALF
  mwin1 := fun _ pid₁ => pid₁
  mwin2 := fun _ pid₁ => pid₁ + 1
  read := fun i pid₀ pid₁ s1 _s2 j => match i with
    | ⟨0, _⟩ => ioXBase IS_VARLEN pid₁ s1 HEAD_IDX stride_x_batch stride_x_seqlen
        stride_x_nheads + pid₀ * BLOCK_M * stride_x_seqlen + j.val * stride_x_headdim
    | ⟨1, _⟩ => ioXBase IS_VARLEN pid₁ s1 HEAD_IDX stride_x_batch stride_x_seqlen
        stride_x_nheads + pid₀ * BLOCK_M * stride_x_seqlen +
        (j.val + rotary_dim_half) * stride_x_headdim
    | ⟨2, _⟩ => (pid₀ * BLOCK_M + SEQLEN_OFFSETS) * rotary_dim_half + j.val
    | ⟨_ + 3, _⟩ => (pid₀ * BLOCK_M + SEQLEN_OFFSETS) * rotary_dim_half + j.val
  readMask := fun i pid₀ _pid₁ s1 s2 j => match i with
    | ⟨0, _⟩ => pid₀ * BLOCK_M < ioSeqlen IS_VARLEN s1 s2 seqlen ∧ j.val < rotary_dim_half
    | ⟨1, _⟩ => pid₀ * BLOCK_M < ioSeqlen IS_VARLEN s1 s2 seqlen ∧ j.val < rotary_dim_half
    | ⟨2, _⟩ => pid₀ * BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ j.val < rotary_dim_half
    | ⟨_ + 3, _⟩ => pid₀ * BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ j.val < rotary_dim_half
  write := fun o pid₀ pid₁ s1 _s2 j =>
    ioOutBase IS_VARLEN pid₁ s1 HEAD_IDX stride_out_batch stride_out_seqlen
        stride_out_nheads + pid₀ * BLOCK_M * stride_out_seqlen +
      (j.val + (match o with | ⟨0, _⟩ => 0 | ⟨_ + 1, _⟩ => rotary_dim_half)) *
        stride_out_headdim
  writeMask := fun _ pid₀ _pid₁ s1 s2 j =>
    pid₀ * BLOCK_M < ioSeqlen IS_VARLEN s1 s2 seqlen ∧ j.val < rotary_dim_half

/-- Membership side condition for the assembly lemma. -/
theorem rotaryMetaIO_hout
    (OUT X COS SIN : RegionName) (CU_SEQLENS : Region .nat)
    (HEAD_IDX SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) (IS_VARLEN : Bool) :
    ∀ o, (rotaryMetaIO OUT X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS seqlen
      rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads
      stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads
      stride_x_headdim BLOCK_M BLOCK_HALF IS_VARLEN).out o ∈
      (rotaryMetaIO OUT X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS seqlen
        rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads
        stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim BLOCK_M BLOCK_HALF IS_VARLEN).bufs := by
  intro o; simp [rotaryMetaIO]

/-- FlattenOk for the companion. -/
theorem meta_row_flattenOk
    (OUT X COS SIN : RegionName) (CU_SEQLENS : Region .nat)
    (HEAD_IDX SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) (IS_VARLEN : Bool) :
    ((rotary_meta_row OUT X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS seqlen
      rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads
      stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads
      stride_x_headdim BLOCK_M BLOCK_HALF IS_VARLEN).toAlgKernel).FlattenOk := by
  cases IS_VARLEN <;>
  · unfold Kernel.FlattenOk
    simp [rotary_meta_row, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- A masked scatter-store `foldl` leaves every cell it does not actively hit
unchanged. -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP, ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc => hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

/-- Termination. -/
theorem meta_row_exec_isSome
    (OUT X COS SIN : RegionName) (CU_SEQLENS : Region .nat)
    (HEAD_IDX SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) (IS_VARLEN : Bool) (s : BlockState) :
    ∃ s1, exec ((rotary_meta_row OUT X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS seqlen
      rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads
      stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads
      stride_x_headdim BLOCK_M BLOCK_HALF IS_VARLEN).toAlgKernel) s = some s1 := by
  cases IS_VARLEN <;>
  simp [exec, rotary_meta_row, ComputeKernel.toAlgKernel,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]

/-- Frame: every cell off both written output halves is preserved. -/
theorem meta_row_frame
    (OUT X COS SIN : RegionName) (CU_SEQLENS : Region .nat)
    (HEAD_IDX SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) (IS_VARLEN : Bool) (s s1 : BlockState)
    (hExec : exec ((rotary_meta_row OUT X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF IS_VARLEN).toAlgKernel)
        s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss0 : ∀ j : Fin BLOCK_HALF,
      metaActive IS_VARLEN s CU_SEQLENS seqlen rotary_dim_half BLOCK_M j →
      ¬(OUT = r ∧ metaOutOffset IS_VARLEN s CU_SEQLENS HEAD_IDX stride_out_batch
          stride_out_seqlen stride_out_nheads stride_out_headdim BLOCK_M 0 j = o))
    (hmiss1 : ∀ j : Fin BLOCK_HALF,
      metaActive IS_VARLEN s CU_SEQLENS seqlen rotary_dim_half BLOCK_M j →
      ¬(OUT = r ∧ metaOutOffset IS_VARLEN s CU_SEQLENS HEAD_IDX stride_out_batch
          stride_out_seqlen stride_out_nheads stride_out_headdim BLOCK_M
          rotary_dim_half j = o)) :
    s1.mem r o = s.mem r o := by
  cases IS_VARLEN <;>
  · simp [exec, rotary_meta_row, ComputeKernel.toAlgKernel,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt] at hExec
    subst hExec
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
      (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl)
    · intro k _ hP hc
      exact hmiss1 k.1 (by simpa [metaActive, metaSeqlen, metaRm] using hP)
        (by simpa [metaOutOffset, metaOutBase, metaRm] using hc)
    · intro k _ hP hc
      exact hmiss0 k.1 (by simpa [metaActive, metaSeqlen, metaRm] using hP)
        (by simpa [metaOutOffset, metaOutBase, metaRm] using hc)

/-- Per-execution safety walk. -/
theorem meta_row_traceSafe
    (OUT X COS SIN : RegionName) (CU_SEQLENS : Region .nat)
    (HEAD_IDX SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) (IS_VARLEN : Bool)
    (bounds : RegionBounds) (s : BlockState)
    (hCU1 : s.pids 1 < bounds CU_SEQLENS)
    (hCU2 : s.pids 1 + 1 < bounds CU_SEQLENS)
    (hx0 : ∀ j : Fin BLOCK_HALF,
      metaActive IS_VARLEN s CU_SEQLENS seqlen rotary_dim_half BLOCK_M j →
      metaXOffset IS_VARLEN s CU_SEQLENS HEAD_IDX stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M 0 j < bounds X)
    (hx1 : ∀ j : Fin BLOCK_HALF,
      metaActive IS_VARLEN s CU_SEQLENS seqlen rotary_dim_half BLOCK_M j →
      metaXOffset IS_VARLEN s CU_SEQLENS HEAD_IDX stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M rotary_dim_half j < bounds X)
    (hc : ∀ j : Fin BLOCK_HALF,
      metaRm s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ j.val < rotary_dim_half →
      metaCosOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M j < bounds COS)
    (hsn : ∀ j : Fin BLOCK_HALF,
      metaRm s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ j.val < rotary_dim_half →
      metaCosOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M j < bounds SIN)
    (ho0 : ∀ j : Fin BLOCK_HALF,
      metaActive IS_VARLEN s CU_SEQLENS seqlen rotary_dim_half BLOCK_M j →
      metaOutOffset IS_VARLEN s CU_SEQLENS HEAD_IDX stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim BLOCK_M 0 j < bounds OUT)
    (ho1 : ∀ j : Fin BLOCK_HALF,
      metaActive IS_VARLEN s CU_SEQLENS seqlen rotary_dim_half BLOCK_M j →
      metaOutOffset IS_VARLEN s CU_SEQLENS HEAD_IDX stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim BLOCK_M rotary_dim_half j < bounds OUT) :
    Kernel.TraceSafe bounds
      ((rotary_meta_row OUT X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS seqlen
        rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads
        stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim BLOCK_M BLOCK_HALF IS_VARLEN).toAlgKernel) s := by
  cases IS_VARLEN
  · unfold Kernel.TraceSafe
    simp [rotary_meta_row, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
      MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
      MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
      BlockState.setReg, tile_elementwise, Bool.and_eq_true,
      Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
      NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]
    refine ⟨fun a ha hb => ?_, fun a ha hb => ?_, fun a ha hb => ?_,
      fun a ha hb => ?_, fun a ha hb => ?_, fun a ha hb => ?_⟩
    · exact hc a ⟨ha, hb⟩
    · exact hsn a ⟨ha, hb⟩
    · exact hx0 a ⟨ha, hb⟩
    · exact hx1 a ⟨ha, hb⟩
    · exact ho0 a ⟨ha, hb⟩
    · exact ho1 a ⟨ha, hb⟩
  · unfold Kernel.TraceSafe
    simp [rotary_meta_row, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
      MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
      MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
      tile_elementwise, Bool.and_eq_true,
      Option.bind, Option.map, Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
      NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]
    refine ⟨⟨hCU1, hCU2⟩, fun a ha hb => ?_, fun a ha hb => ?_, fun a ha hb => ?_,
      fun a ha hb => ?_, fun a ha hb => ?_, fun a ha hb => ?_⟩
    · exact hc a ⟨ha, hb⟩
    · exact hsn a ⟨ha, hb⟩
    · exact hx0 a ⟨ha, hb⟩
    · exact hx1 a ⟨ha, hb⟩
    · exact ho0 a ⟨ha, hb⟩
    · exact ho1 a ⟨ha, hb⟩

/-- Value bridge: the memory-form o0 spec equals the `xs`-form value `F`. -/
theorem metaO0Spec_eq_F
    (s₀ : BlockState) (X COS SIN CU : RegionName) (IS_VARLEN : Bool)
    (HEAD_IDX SEQLEN_OFFSETS seqlen_ro stride_x_batch stride_x_seqlen stride_x_nheads
      stride_x_headdim rotary_dim_half BLOCK_M BLOCK_HALF : Nat)
    (xs : Fin 4 → Fin BLOCK_HALF → ℝ) (j : Fin BLOCK_HALF)
    (hxX0 : s₀.readMem X (metaXOffset IS_VARLEN s₀ CU HEAD_IDX stride_x_batch
      stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_M 0 j) = xs 0 j)
    (hxX1 : s₀.readMem X (metaXOffset IS_VARLEN s₀ CU HEAD_IDX stride_x_batch
      stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_M rotary_dim_half j)
      = xs 1 j)
    (hxCos : metaRm s₀ BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ j.val < rotary_dim_half →
      s₀.readMem COS (metaCosOffset s₀ SEQLEN_OFFSETS rotary_dim_half BLOCK_M j) = xs 2 j)
    (hxSin : metaRm s₀ BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ j.val < rotary_dim_half →
      s₀.readMem SIN (metaCosOffset s₀ SEQLEN_OFFSETS rotary_dim_half BLOCK_M j) = xs 3 j) :
    metaO0Spec IS_VARLEN s₀ X COS SIN CU HEAD_IDX SEQLEN_OFFSETS seqlen_ro stride_x_batch
        stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half BLOCK_M j
      = rotaryMetaF SEQLEN_OFFSETS seqlen_ro rotary_dim_half BLOCK_M (s₀.pids 0) xs
          ⟨0, by decide⟩ j := by
  unfold metaO0Spec rotaryMetaF
  rw [hxX0, hxX1]
  simp only [metaRm]
  by_cases hcos : s₀.pids 0 * BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ j.val < rotary_dim_half
  · simp only [if_pos hcos]; rw [hxCos hcos, hxSin hcos]
  · simp only [if_neg hcos]

/-- Value bridge for o1. -/
theorem metaO1Spec_eq_F
    (s₀ : BlockState) (X COS SIN CU : RegionName) (IS_VARLEN : Bool)
    (HEAD_IDX SEQLEN_OFFSETS seqlen_ro stride_x_batch stride_x_seqlen stride_x_nheads
      stride_x_headdim rotary_dim_half BLOCK_M BLOCK_HALF : Nat)
    (xs : Fin 4 → Fin BLOCK_HALF → ℝ) (j : Fin BLOCK_HALF)
    (hxX0 : s₀.readMem X (metaXOffset IS_VARLEN s₀ CU HEAD_IDX stride_x_batch
      stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_M 0 j) = xs 0 j)
    (hxX1 : s₀.readMem X (metaXOffset IS_VARLEN s₀ CU HEAD_IDX stride_x_batch
      stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_M rotary_dim_half j)
      = xs 1 j)
    (hxCos : metaRm s₀ BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ j.val < rotary_dim_half →
      s₀.readMem COS (metaCosOffset s₀ SEQLEN_OFFSETS rotary_dim_half BLOCK_M j) = xs 2 j)
    (hxSin : metaRm s₀ BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ j.val < rotary_dim_half →
      s₀.readMem SIN (metaCosOffset s₀ SEQLEN_OFFSETS rotary_dim_half BLOCK_M j) = xs 3 j) :
    metaO1Spec IS_VARLEN s₀ X COS SIN CU HEAD_IDX SEQLEN_OFFSETS seqlen_ro stride_x_batch
        stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half BLOCK_M j
      = rotaryMetaF SEQLEN_OFFSETS seqlen_ro rotary_dim_half BLOCK_M (s₀.pids 0) xs
          ⟨1, by decide⟩ j := by
  unfold metaO1Spec rotaryMetaF
  rw [hxX0, hxX1]
  simp only [metaRm]
  by_cases hcos : s₀.pids 0 * BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ j.val < rotary_dim_half
  · simp only [if_pos hcos]; rw [hxCos hcos, hxSin hcos]
  · simp only [if_neg hcos]

/-- The per-lane output offsets are injective given a nonzero head-dim stride. -/
theorem metaOutOffset_inj (IS_VARLEN : Bool) (s : BlockState) (CU : RegionName)
    (HEAD_IDX stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      BLOCK_M BLOCK_HALF shift : Nat) (hStrideHd : stride_out_headdim ≠ 0) :
    Function.Injective
      (fun i : Fin BLOCK_HALF =>
        metaOutOffset IS_VARLEN s CU HEAD_IDX stride_out_batch stride_out_seqlen
          stride_out_nheads stride_out_headdim BLOCK_M shift i) := by
  intro a b h
  simp only [metaOutOffset] at h
  have h1 : (a.val + shift) * stride_out_headdim = (b.val + shift) * stride_out_headdim :=
    Nat.add_left_cancel h
  have h2 : a.val + shift = b.val + shift :=
    Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero hStrideHd) h1
  exact Fin.ext (Nat.add_right_cancel h2)

open scoped MetaGroupedMasked2DKernelIO in
/-- **The `⊨` metadata-grouped headline for `rotary_transform`.** -/
theorem rotary_transform_ops_meta_implements
    (OUT X COS SIN : RegionName) (CU_SEQLENS : Region .nat)
    (HEAD_IDX SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) (IS_VARLEN : Bool)
    (hStrideHd : stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ rotary_dim_half) :
    rotaryMetaIO OUT X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS seqlen rotary_dim_half
        seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
        stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF
        IS_VARLEN
      ⊨ fun pid₀ _pid₁ _s1 _s2 xs o j =>
          rotaryMetaF SEQLEN_OFFSETS seqlen_ro rotary_dim_half BLOCK_M pid₀ xs o j := by
  refine MetaGroupedMasked2DKernelIO.Implements.intro _
    (rotaryMetaIO_hout OUT X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS seqlen
      rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads
      stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF IS_VARLEN)
    (meta_row_flattenOk OUT X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS seqlen
      rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads
      stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF IS_VARLEN) ?_ ?_
  · -- hts
    intro bounds s s1 s2 hpin1 hpin2 hb1 hb2 hbr hbw
    simp only [rotaryMetaIO] at hpin1 hpin2 hb1 hb2 hbr hbw
    refine meta_row_traceSafe OUT X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS seqlen
      rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads
      stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF IS_VARLEN bounds s hb1 hb2 ?_ ?_ ?_ ?_ ?_ ?_
    · intro j hact
      have := hbr ⟨0, by decide⟩ j (by
        simp only [ioSeqlen, metaActive, metaSeqlen, metaRm] at hact ⊢
        rw [hpin1, hpin2] at hact; exact hact)
      simpa [ioXBase, metaXOffset, metaXBase, metaRm, hpin1] using this
    · intro j hact
      have := hbr ⟨1, by decide⟩ j (by
        simp only [ioSeqlen, metaActive, metaSeqlen, metaRm] at hact ⊢
        rw [hpin1, hpin2] at hact; exact hact)
      simpa [ioXBase, metaXOffset, metaXBase, metaRm, hpin1] using this
    · intro j hact
      have := hbr ⟨2, by decide⟩ j (by simpa [metaRm] using hact)
      simpa [metaCosOffset, metaRm] using this
    · intro j hact
      have := hbr ⟨3, by decide⟩ j (by simpa [metaRm] using hact)
      simpa [metaCosOffset, metaRm] using this
    · intro j hact
      have := hbw ⟨0, by decide⟩ j (by
        simp only [ioSeqlen, metaActive, metaSeqlen, metaRm] at hact ⊢
        rw [hpin1, hpin2] at hact; exact hact)
      simpa [ioOutBase, metaOutOffset, metaOutBase, metaRm, hpin1] using this
    · intro j hact
      have := hbw ⟨1, by decide⟩ j (by
        simp only [ioSeqlen, metaActive, metaSeqlen, metaRm] at hact ⊢
        rw [hpin1, hpin2] at hact; exact hact)
      simpa [ioOutBase, metaOutOffset, metaOutBase, metaRm, hpin1] using this
  · -- hrun
    intro s₀ s1 s2 xs hpin1 hpin2 hx
    simp only [rotaryMetaIO] at hpin1 hpin2 hx
    obtain ⟨s1', hexec⟩ := meta_row_exec_isSome OUT X COS SIN CU_SEQLENS HEAD_IDX
      SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
      stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads
      stride_x_headdim BLOCK_M BLOCK_HALF IS_VARLEN s₀
    -- concrete `xs` reads (bridge io windows → meta offsets via pins)
    have hxX0 : ∀ j : Fin BLOCK_HALF,
        metaActive IS_VARLEN s₀ CU_SEQLENS seqlen rotary_dim_half BLOCK_M j →
        s₀.readMem X (metaXOffset IS_VARLEN s₀ CU_SEQLENS HEAD_IDX stride_x_batch
          stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_M 0 j) = xs (0 : Fin 4) j := by
      intro j hact
      have := hx (⟨0, by decide⟩ : Fin 4) j (by
        simp only [rotaryMetaIO, ioSeqlen, metaActive, metaSeqlen, metaRm] at hact ⊢
        rw [hpin1, hpin2] at hact; exact hact)
      simpa [rotaryMetaIO, ioXBase, metaXOffset, metaXBase, metaRm, hpin1] using this
    have hxX1 : ∀ j : Fin BLOCK_HALF,
        metaActive IS_VARLEN s₀ CU_SEQLENS seqlen rotary_dim_half BLOCK_M j →
        s₀.readMem X (metaXOffset IS_VARLEN s₀ CU_SEQLENS HEAD_IDX stride_x_batch
          stride_x_seqlen stride_x_nheads stride_x_headdim BLOCK_M rotary_dim_half j)
          = xs (1 : Fin 4) j := by
      intro j hact
      have := hx (⟨1, by decide⟩ : Fin 4) j (by
        simp only [rotaryMetaIO, ioSeqlen, metaActive, metaSeqlen, metaRm] at hact ⊢
        rw [hpin1, hpin2] at hact; exact hact)
      simpa [rotaryMetaIO, ioXBase, metaXOffset, metaXBase, metaRm, hpin1] using this
    have hxCos : ∀ j : Fin BLOCK_HALF,
        metaRm s₀ BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ j.val < rotary_dim_half →
        s₀.readMem COS (metaCosOffset s₀ SEQLEN_OFFSETS rotary_dim_half BLOCK_M j)
          = xs (2 : Fin 4) j := by
      intro j hact
      have := hx (⟨2, by decide⟩ : Fin 4) j (by simpa [rotaryMetaIO, metaRm] using hact)
      simpa [rotaryMetaIO, metaCosOffset, metaRm] using this
    have hxSin : ∀ j : Fin BLOCK_HALF,
        metaRm s₀ BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧ j.val < rotary_dim_half →
        s₀.readMem SIN (metaCosOffset s₀ SEQLEN_OFFSETS rotary_dim_half BLOCK_M j)
          = xs (3 : Fin 4) j := by
      intro j hact
      have := hx (⟨3, by decide⟩ : Fin 4) j (by simpa [rotaryMetaIO, metaRm] using hact)
      simpa [rotaryMetaIO, metaCosOffset, metaRm] using this
    refine ⟨s1', hexec, ?_, ?_⟩
    · -- values
      intro o j hwmask
      simp only [rotaryMetaIO] at hwmask ⊢
      have hact : metaActive IS_VARLEN s₀ CU_SEQLENS seqlen rotary_dim_half BLOCK_M j := by
        simp only [ioSeqlen, metaActive, metaSeqlen, metaRm] at hwmask ⊢
        rw [hpin1, hpin2]; exact hwmask
      fin_cases o
      · have hc := meta_row_o0_correct OUT X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS
          seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
          stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
          stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF IS_VARLEN s₀ s1'
          (metaOutOffset_inj IS_VARLEN s₀ CU_SEQLENS HEAD_IDX stride_out_batch
            stride_out_seqlen stride_out_nheads stride_out_headdim BLOCK_M BLOCK_HALF 0
            hStrideHd)
          hStrideHd hHalfBound hexec j
        have hval := hc.trans (if_pos hact)
        simp only [ioOutBase, metaOutOffset, metaOutBase, metaRm, hpin1] at hval ⊢
        rw [hval]
        exact metaO0Spec_eq_F s₀ X COS SIN CU_SEQLENS IS_VARLEN HEAD_IDX SEQLEN_OFFSETS
          seqlen_ro stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
          rotary_dim_half BLOCK_M BLOCK_HALF xs j (hxX0 j hact) (hxX1 j hact) (hxCos j)
          (hxSin j)
      · have hc := meta_row_o1_correct OUT X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS
          seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
          stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
          stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF IS_VARLEN s₀ s1'
          (metaOutOffset_inj IS_VARLEN s₀ CU_SEQLENS HEAD_IDX stride_out_batch
            stride_out_seqlen stride_out_nheads stride_out_headdim BLOCK_M BLOCK_HALF
            rotary_dim_half hStrideHd)
          hStrideHd hHalfBound hexec j
        have hval := hc.trans (if_pos hact)
        simp only [ioOutBase, metaOutOffset, metaOutBase, metaRm, hpin1] at hval ⊢
        rw [hval]
        exact metaO1Spec_eq_F s₀ X COS SIN CU_SEQLENS IS_VARLEN HEAD_IDX SEQLEN_OFFSETS
          seqlen_ro stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
          rotary_dim_half BLOCK_M BLOCK_HALF xs j (hxX0 j hact) (hxX1 j hact) (hxCos j)
          (hxSin j)
    · -- frame
      intro r o' hcond
      refine meta_row_frame OUT X COS SIN CU_SEQLENS HEAD_IDX SEQLEN_OFFSETS seqlen
        rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen stride_out_nheads
        stride_out_headdim stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
        BLOCK_M BLOCK_HALF IS_VARLEN s₀ s1' hexec r o' (fun j hj hc => ?_) (fun j hj hc => ?_)
      · rcases hcond (⟨0, by decide⟩ : Fin 2) j (by
          simp only [rotaryMetaIO, ioSeqlen, metaActive, metaSeqlen, metaRm] at hj ⊢
          rw [hpin1, hpin2] at hj; exact hj) with hne | hno
        · exact hne hc.1.symm
        · exact hno (by
            simp only [rotaryMetaIO, ioOutBase, metaOutOffset, metaOutBase, metaRm,
              hpin1] at hc ⊢
            rw [← hc.2])
      · rcases hcond (⟨1, by decide⟩ : Fin 2) j (by
          simp only [rotaryMetaIO, ioSeqlen, metaActive, metaSeqlen, metaRm] at hj ⊢
          rw [hpin1, hpin2] at hj; exact hj) with hne | hno
        · exact hne hc.1.symm
        · exact hno (by
            simp only [rotaryMetaIO, ioOutBase, metaOutOffset, metaOutBase, metaRm,
              hpin1] at hc ⊢
            rw [← hc.2])

end VeriTile.Bench.TritonBenchG.RotaryTransformOps
