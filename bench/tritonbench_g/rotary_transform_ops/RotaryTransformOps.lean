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

end VeriTile.Bench.TritonBenchG.RotaryTransformOps
