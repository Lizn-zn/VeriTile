import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `rope_transform` — strict per-kernel correctness

`_triton_rope` applies the rotary position embedding in place to fused Q/K
buffers: each program owns one row (`program_id(0) = batch*seq` index), loads the
per-row `cos`/`sin` half-dim vectors, and for every head rewrites the two rotary
halves of both `q_ptr` and `k_ptr` via `(t1*cos - t2*sin, t2*cos + t1*sin)` on
the forward path (`BACKWARD_PASS = false`, the sign-flipped transpose otherwise).

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (the `(n_row,)` grid, the `next_power_of_2` padding choices
`pad_n_qh`/`pad_n_kh`/`pad_hd`, `BLOCK_SIZE`, the contiguity/transpose bookkeeping
in `rope_forward`, and how the runtime composes per-program writes into one
buffer) is the *trusted boundary*, not a proof obligation here. Because the row
position and per-head/per-dim indices are universally quantified, the per-program
statement covers every program of the grid.

## Proof architecture

```
rope_transform_python_forward_output_summary          ← TOP THEOREM
  ├─ triton_rope_surface_toAlgorithm_supported         surface lowers to the algorithm layer
  └─ triton_rope_surface_output_compute_correct        ← ComputeCorrect over Q and K stores
       (= tritonRopeSurfaceValue spec, full-surface forward)

supporting head-slice + per-store track:
rope_transform_python_forward_store_summary
  ├─ triton_rope_surface_toAlgorithm_supported
  ├─ rope_transform_q0_python_test_shape_compute_correct → rope_transform_q0_head_compute_correct → rope_transform_q0_head_correct
  ├─ rope_transform_q1_python_test_shape_compute_correct → rope_transform_q1_head_compute_correct → rope_transform_q1_head_correct
  ├─ rope_transform_k0_python_test_shape_compute_correct → rope_transform_k0_head_compute_correct
  └─ rope_transform_k1_python_test_shape_compute_correct → rope_transform_k1_head_compute_correct

combined-row track (rotary-style 2D slice):
rope_kernel_o0o1_row_all_outputs_compute_correct
  ├─ rope_kernel_o0o1_row_o0_compute_correct → rope_kernel_o0o1_row_o0_correct
  └─ rope_kernel_o0o1_row_o1_compute_correct → rope_kernel_o0o1_row_o1_correct
```

Offset disjointness within a half pair is supplied by
`qFirstHalf_ne_qSecondHalf` / `kFirstHalf_ne_kSecondHalf`.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; the `.to(sin_row.dtype)`
register casts erase to the identity at the algorithm layer (post-erasure all
dtypes unify to `ℝ`). `cos`/`sin` are modeled as **precomputed inputs** loaded
from memory, not computed. The top summary is the **full-surface** forward result
(`triton_rope_surface_output_compute_correct`, instantiated at the Python test
shape); the per-store `store_summary` proves the four Python-observable stores
(Q/K first and second halves) one head-slice at a time and, honestly, is **not
yet lifted** from a single head slice to the full head tile — the
`rope-head-slice-lift` step (issue #153) remains a trusted gap on that track. The
`@triton.heuristics`/`BACKWARD_PASS` flag is modeled by the `Bool` argument;
`@triton.autotune` is not modeled.
-/

namespace VeriTile.Bench.TritonBenchG.RopeTransform

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `rope_transform.py`'s `_triton_rope`. -/
def triton_rope_surface
    (q_ptr k_ptr cos sin : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (BACKWARD_PASS : Bool) :
    ComputeKernel := triton {
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * $(q_row_stride)
    k_ptr = k_ptr + pid * $(k_row_stride)
    cos_row_idx = pid % $(sl)
    cos = cos + cos_row_idx * $(cos_row_stride)
    sin = sin + cos_row_idx * $(sin_row_stride)
    cos_offsets = tl.arange(0, $(pad_hd) // $(2))
    cos_mask = cos_offsets < $(hd) // $(2)
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0)
    first_half_q_offsets = tl.arange(0, $(pad_n_qh))[:, None] * $(hd) +
      tl.arange(0, $(pad_hd) // $(2))[None, :]
    first_half_k_offsets = tl.arange(0, $(pad_n_kh))[:, None] * $(hd) +
      tl.arange(0, $(pad_hd) // $(2))[None, :]
    first_q_mask = (tl.arange(0, $(pad_n_qh))[:, None] < $(n_qh)) &
      (tl.arange(0, $(pad_hd) // $(2))[None, :] < $(hd) // $(2))
    first_k_mask = (tl.arange(0, $(pad_n_kh))[:, None] < $(n_kh)) &
      (tl.arange(0, $(pad_hd) // $(2))[None, :] < $(hd) // $(2))
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask,
      other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask,
      other=0).to(sin_row.dtype)
    second_half_q_offsets = first_half_q_offsets + $(hd) // $(2)
    second_half_k_offsets = first_half_k_offsets + $(hd) // $(2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask,
      other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask,
      other=0).to(sin_row.dtype)
    if not BACKWARD_PASS {
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)
    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    } else {
    new_q_tile_1 = q_tile_1 * cos_row + q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row - q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)
    new_k_tile_1 = k_tile_1 * cos_row + k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row - k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    }
}

/-- The full forward/backward RoPE transform surface lowers to the algorithm
layer, including Q/K, both halves, and the `BACKWARD_PASS` branch. -/
theorem triton_rope_surface_toAlgorithm_supported
    (q_ptr k_ptr cos sin : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (BACKWARD_PASS : Bool) :
    ∃ alg, (triton_rope_surface q_ptr k_ptr cos sin q_row_stride k_row_stride
      cos_row_stride sin_row_stride sl bs n_qh n_kh hd pad_n_qh pad_n_kh
      pad_hd BLOCK_SIZE BACKWARD_PASS).toAlgorithm? = Except.ok alg := by
  simp [triton_rope_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

noncomputable def tritonRopeSurfaceValue
    (s : BlockState) (q_ptr k_ptr cos sin out : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (BACKWARD_PASS : Bool) (offset : Nat) : ℝ :=
  match exec (triton_rope_surface q_ptr k_ptr cos sin q_row_stride
      k_row_stride cos_row_stride sin_row_stride sl bs n_qh n_kh hd
      pad_n_qh pad_n_kh pad_hd BLOCK_SIZE BACKWARD_PASS) s with
  | some s' => s'.readMem out offset
  | none => 0.0

theorem triton_rope_surface_output_compute_correct
    {ι : Type} (q_ptr k_ptr cos sin out : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (BACKWARD_PASS : Bool) (s : BlockState) (offsetOf : ι → Nat) :
    ComputeCorrect.Realizes
      (kernel := triton_rope_surface q_ptr k_ptr cos sin q_row_stride
        k_row_stride cos_row_stride sin_row_stride sl bs n_qh n_kh hd
        pad_n_qh pad_n_kh pad_hd BLOCK_SIZE BACKWARD_PASS)
      (initialState := s)
      (write := fun i : ι => some (out, offsetOf i))
      (expected := fun i =>
        tritonRopeSurfaceValue s q_ptr k_ptr cos sin out q_row_stride
          k_row_stride cos_row_stride sin_row_stride sl bs n_qh n_kh hd
          pad_n_qh pad_n_kh pad_hd BLOCK_SIZE BACKWARD_PASS (offsetOf i)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_rope_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [tritonRopeSurfaceValue, hExec]

/-- Proof-oriented one-Q-head first-half forward slice of `rope_transform.py`'s
`_triton_rope`.

The full Python kernel handles Q and K, both halves, and a backward branch. This
slice fixes one Q head and one row, and proves the forward first-half store:
`q0' = q0 * cos - q1 * sin`. -/
def rope_transform_q0_head
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      n_qh HEAD_HALF BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  dim = tl.arange(0, $(BLOCK_HALF))
  q_base = Q + pid * $(q_row_stride) + $(HEAD_IDX) * $(hd)
  cos_base = COS + $(COS_ROW_IDX) * $(cos_row_stride)
  sin_base = SIN + $(COS_ROW_IDX) * $(sin_row_stride)
  q0 = tl.load(q_base + dim,
    mask=($(HEAD_IDX) < $(n_qh)) and (dim < $(HEAD_HALF)), other=0).to(sin_row.dtype)
  q1 = tl.load(q_base + dim + $(HEAD_HALF),
    mask=($(HEAD_IDX) < $(n_qh)) and (dim < $(HEAD_HALF)), other=0).to(sin_row.dtype)
  cos_row = tl.load(cos_base + dim, mask=dim < $(HEAD_HALF), other=0)
  sin_row = tl.load(sin_base + dim, mask=dim < $(HEAD_HALF), other=0)
  out = q0 * cos_row - q1 * sin_row
  tl.store(q_base + dim, out,
    mask=($(HEAD_IDX) < $(n_qh)) and (dim < $(HEAD_HALF)))
}

def rowIndex (s : BlockState) : Nat :=
  s.pids 0

def dimIndex (i : Fin BLOCK_HALF) : Nat :=
  i.val

def active (HEAD_IDX n_qh HEAD_HALF : Nat) (i : Fin BLOCK_HALF) : Prop :=
  HEAD_IDX < n_qh ∧ dimIndex i < HEAD_HALF

instance activeDecidable (HEAD_IDX n_qh HEAD_HALF : Nat) (i : Fin BLOCK_HALF) :
    Decidable (active HEAD_IDX n_qh HEAD_HALF i) := by
  unfold active
  infer_instance

def qOffset
    (s : BlockState) (HEAD_IDX q_row_stride hd : Nat) (dim : Nat) : Nat :=
  rowIndex s * q_row_stride + HEAD_IDX * hd + dim

def cosOffset (COS_ROW_IDX cos_row_stride : Nat) (i : Fin BLOCK_HALF) : Nat :=
  COS_ROW_IDX * cos_row_stride + dimIndex i

def sinOffset (COS_ROW_IDX sin_row_stride : Nat) (i : Fin BLOCK_HALF) : Nat :=
  COS_ROW_IDX * sin_row_stride + dimIndex i

noncomputable def ropeQ0Spec
    (s : BlockState) (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      HEAD_HALF : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i)) *
    s.readMem COS (cosOffset COS_ROW_IDX cos_row_stride i) -
  s.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i + HEAD_HALF)) *
    s.readMem SIN (sinOffset COS_ROW_IDX sin_row_stride i)

/-- Algorithm-layer correctness for the one-Q-head first-half RoPE store. -/
theorem rope_transform_q0_head_correct
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      n_qh HEAD_HALF BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF => qOffset s HEAD_IDX q_row_stride hd (dimIndex i)))
    (hExec : exec (rope_transform_q0_head Q COS SIN HEAD_IDX COS_ROW_IDX
        q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF
        BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i)) =
        if active HEAD_IDX n_qh HEAD_HALF i then
          ropeQ0Spec s Q COS SIN HEAD_IDX COS_ROW_IDX q_row_stride
            cos_row_stride sin_row_stride hd HEAD_HALF i
        else
          s.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i)) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HALF] =>
        s.pids 0 * q_row_stride + HEAD_IDX * hd + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [qOffset, rowIndex, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBH : 0 < BLOCK_HALF
  · simp [exec, rope_transform_q0_head, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [qOffset, rowIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hHead : HEAD_IDX < n_qh
    · by_cases hDim : i.val < HEAD_HALF
      · simp [active, ropeQ0Spec, qOffset, cosOffset, sinOffset,
              rowIndex, dimIndex, hHead, hDim, Option.bind, Option.map]
        left
        rw [Nat.add_assoc]
      · simp [active, dimIndex, hHead, hDim]
    · simp [active, hHead]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-Q-head first-half RoPE store. -/
theorem rope_transform_q0_head_compute_correct
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      n_qh HEAD_HALF BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF => qOffset s HEAD_IDX q_row_stride hd (dimIndex i))) :
    ComputeCorrect.Realizes
      (kernel := rope_transform_q0_head Q COS SIN HEAD_IDX COS_ROW_IDX
        q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF
        BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active HEAD_IDX n_qh HEAD_HALF i)
        (fun i => (Q, qOffset s HEAD_IDX q_row_stride hd (dimIndex i))))
      (expected := fun i =>
        ropeQ0Spec s Q COS SIN HEAD_IDX COS_ROW_IDX q_row_stride
          cos_row_stride sin_row_stride hd HEAD_HALF i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rope_transform_q0_head]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rope_transform_q0_head_correct Q COS SIN HEAD_IDX COS_ROW_IDX
    q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF BLOCK_HALF
    s s' hOutInj hExec i
  simpa [hActive] using h

/-! ## Second-half Q store (`new_q_tile_2`) -/

/-- Slice fixing one Q head, proving the forward second-half Q store:
`q1' = q1 * cos + q0 * sin` written at `second_half_q_offsets`. -/
def rope_transform_q1_head
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      n_qh HEAD_HALF BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  dim = tl.arange(0, $(BLOCK_HALF))
  q_base = Q + pid * $(q_row_stride) + $(HEAD_IDX) * $(hd)
  cos_base = COS + $(COS_ROW_IDX) * $(cos_row_stride)
  sin_base = SIN + $(COS_ROW_IDX) * $(sin_row_stride)
  q0 = tl.load(q_base + dim,
    mask=($(HEAD_IDX) < $(n_qh)) and (dim < $(HEAD_HALF)), other=0).to(sin_row.dtype)
  q1 = tl.load(q_base + dim + $(HEAD_HALF),
    mask=($(HEAD_IDX) < $(n_qh)) and (dim < $(HEAD_HALF)), other=0).to(sin_row.dtype)
  cos_row = tl.load(cos_base + dim, mask=dim < $(HEAD_HALF), other=0)
  sin_row = tl.load(sin_base + dim, mask=dim < $(HEAD_HALF), other=0)
  out = q1 * cos_row + q0 * sin_row
  tl.store(q_base + dim + $(HEAD_HALF), out,
    mask=($(HEAD_IDX) < $(n_qh)) and (dim < $(HEAD_HALF)))
}

/-- Write offset for `q1`: matches the kernel evaluation
`q_base + dim + HEAD_HALF`, i.e. `((s.pid * q_row_stride + HEAD_IDX * hd)
+ dim) + HEAD_HALF`. -/
def q1WriteOffset
    (s : BlockState) (HEAD_IDX q_row_stride hd HEAD_HALF : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pid * q_row_stride + HEAD_IDX * hd + dimIndex i + HEAD_HALF

noncomputable def ropeQ1Spec
    (s : BlockState) (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      HEAD_HALF : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i) *
    s.readMem COS (cosOffset COS_ROW_IDX cos_row_stride i) +
  s.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i)) *
    s.readMem SIN (sinOffset COS_ROW_IDX sin_row_stride i)

/-- Algorithm-layer correctness for the one-Q-head second-half RoPE store. -/
theorem rope_transform_q1_head_correct
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      n_qh HEAD_HALF BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i))
    (hExec : exec (rope_transform_q1_head Q COS SIN HEAD_IDX COS_ROW_IDX
        q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF
        BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem Q (q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i) =
        if active HEAD_IDX n_qh HEAD_HALF i then
          ropeQ1Spec s Q COS SIN HEAD_IDX COS_ROW_IDX q_row_stride
            cos_row_stride sin_row_stride hd HEAD_HALF i
        else
          s.readMem Q (q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HALF] =>
        s.pids 0 * q_row_stride + HEAD_IDX * hd + idx.1.val + HEAD_HALF) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simp only [q1WriteOffset, rowIndex, dimIndex] at *
      exact h
    cases a; cases b; simp only at hab; cases hab; rfl
  by_cases hBH : 0 < BLOCK_HALF
  · simp [exec, rope_transform_q1_head, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [q1WriteOffset, rowIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hHead : HEAD_IDX < n_qh
    · by_cases hDim : i.val < HEAD_HALF
      · simp [active, ropeQ1Spec, q1WriteOffset, qOffset, cosOffset, sinOffset,
              rowIndex, dimIndex, hHead, hDim, Option.bind, Option.map]
      · simp [active, dimIndex, hHead, hDim]
    · simp [active, hHead]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-Q-head second-half RoPE store. -/
theorem rope_transform_q1_head_compute_correct
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      n_qh HEAD_HALF BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i)) :
    ComputeCorrect.Realizes
      (kernel := rope_transform_q1_head Q COS SIN HEAD_IDX COS_ROW_IDX
        q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF
        BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active HEAD_IDX n_qh HEAD_HALF i)
        (fun i => (Q, q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i)))
      (expected := fun i =>
        ropeQ1Spec s Q COS SIN HEAD_IDX COS_ROW_IDX q_row_stride
          cos_row_stride sin_row_stride hd HEAD_HALF i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rope_transform_q1_head]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rope_transform_q1_head_correct Q COS SIN HEAD_IDX COS_ROW_IDX
    q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF BLOCK_HALF
    s s' hOutInj hExec i
  simpa [hActive] using h

/-! ## K-side coverage

The `rope_transform_q0_head` and `rope_transform_q1_head` surfaces are
parameterised by a generic region name and head-count bound. Applying the
same correctness theorems with the K region, `n_kh` head count, and the
appropriate `K` stride covers the K-side forward stores
`new_k_tile_1 = k_tile_1 * cos - k_tile_2 * sin` and
`new_k_tile_2 = k_tile_2 * cos + k_tile_1 * sin`. The thin wrappers below
make this explicit so downstream consumers of #134 can cite a K-specific
theorem name. -/

/-- K-side first-half store, alias of `rope_transform_q0_head` instantiated
with the K region and `n_kh` head bound. -/
abbrev rope_transform_k0_head
    (K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX k_row_stride cos_row_stride sin_row_stride hd
      n_kh HEAD_HALF BLOCK_HALF : Nat) :
    ComputeKernel :=
  rope_transform_q0_head K COS SIN HEAD_IDX COS_ROW_IDX k_row_stride
    cos_row_stride sin_row_stride hd n_kh HEAD_HALF BLOCK_HALF

/-- K-side second-half store, alias of `rope_transform_q1_head`. -/
abbrev rope_transform_k1_head
    (K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX k_row_stride cos_row_stride sin_row_stride hd
      n_kh HEAD_HALF BLOCK_HALF : Nat) :
    ComputeKernel :=
  rope_transform_q1_head K COS SIN HEAD_IDX COS_ROW_IDX k_row_stride
    cos_row_stride sin_row_stride hd n_kh HEAD_HALF BLOCK_HALF

/-- K-side first-half correctness, derived from the Q-side theorem. -/
theorem rope_transform_k0_head_compute_correct
    (K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX k_row_stride cos_row_stride sin_row_stride hd
      n_kh HEAD_HALF BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        qOffset s HEAD_IDX k_row_stride hd (dimIndex i))) :
    ComputeCorrect.Realizes
      (kernel := rope_transform_k0_head K COS SIN HEAD_IDX COS_ROW_IDX
        k_row_stride cos_row_stride sin_row_stride hd n_kh HEAD_HALF
        BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active HEAD_IDX n_kh HEAD_HALF i)
        (fun i => (K, qOffset s HEAD_IDX k_row_stride hd (dimIndex i))))
      (expected := fun i =>
        ropeQ0Spec s K COS SIN HEAD_IDX COS_ROW_IDX k_row_stride
          cos_row_stride sin_row_stride hd HEAD_HALF i) :=
  rope_transform_q0_head_compute_correct K COS SIN HEAD_IDX COS_ROW_IDX
    k_row_stride cos_row_stride sin_row_stride hd n_kh HEAD_HALF BLOCK_HALF
    s hOutInj

/-- K-side second-half correctness, derived from the Q-side theorem. -/
theorem rope_transform_k1_head_compute_correct
    (K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX k_row_stride cos_row_stride sin_row_stride hd
      n_kh HEAD_HALF BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        q1WriteOffset s HEAD_IDX k_row_stride hd HEAD_HALF i)) :
    ComputeCorrect.Realizes
      (kernel := rope_transform_k1_head K COS SIN HEAD_IDX COS_ROW_IDX
        k_row_stride cos_row_stride sin_row_stride hd n_kh HEAD_HALF
        BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active HEAD_IDX n_kh HEAD_HALF i)
        (fun i => (K, q1WriteOffset s HEAD_IDX k_row_stride hd HEAD_HALF i)))
      (expected := fun i =>
        ropeQ1Spec s K COS SIN HEAD_IDX COS_ROW_IDX k_row_stride
          cos_row_stride sin_row_stride hd HEAD_HALF i) :=
  rope_transform_q1_head_compute_correct K COS SIN HEAD_IDX COS_ROW_IDX
    k_row_stride cos_row_stride sin_row_stride hd n_kh HEAD_HALF BLOCK_HALF
    s hOutInj

/-! ## Python test-shape wrappers

`rope_transform.py`'s checked test case uses `batch_size = 2`, `seq_len = 4`,
`n_q_head = n_kv_head = 8`, and `head_dim = 16`. After the Python
`transpose(...).contiguous()`, both Q and K have row stride `8 * 16 = 128`,
and the cos/sin row stride is `16 / 2 = 8`. The wrappers below pin those
metadata values so the forward Q/K first- and second-half stores are directly
citable as the Python test slice. -/

theorem rope_transform_q0_python_test_shape_compute_correct
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_transform_q0_head Q COS SIN HEAD_IDX COS_ROW_IDX
        128 8 8 16 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active HEAD_IDX 8 8 i)
        (fun i => (Q, qOffset s HEAD_IDX 128 16 (dimIndex i))))
      (expected := fun i =>
        ropeQ0Spec s Q COS SIN HEAD_IDX COS_ROW_IDX 128 8 8 16 8 i) := by
  apply rope_transform_q0_head_compute_correct
  intro a b h
  apply Fin.ext
  simp [qOffset, rowIndex, dimIndex] at h
  omega

theorem rope_transform_q1_python_test_shape_compute_correct
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_transform_q1_head Q COS SIN HEAD_IDX COS_ROW_IDX
        128 8 8 16 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active HEAD_IDX 8 8 i)
        (fun i => (Q, q1WriteOffset s HEAD_IDX 128 16 8 i)))
      (expected := fun i =>
        ropeQ1Spec s Q COS SIN HEAD_IDX COS_ROW_IDX 128 8 8 16 8 i) := by
  apply rope_transform_q1_head_compute_correct
  intro a b h
  apply Fin.ext
  simp [q1WriteOffset, dimIndex] at h
  omega

theorem rope_transform_k0_python_test_shape_compute_correct
    (K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_transform_k0_head K COS SIN HEAD_IDX COS_ROW_IDX
        128 8 8 16 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active HEAD_IDX 8 8 i)
        (fun i => (K, qOffset s HEAD_IDX 128 16 (dimIndex i))))
      (expected := fun i =>
        ropeQ0Spec s K COS SIN HEAD_IDX COS_ROW_IDX 128 8 8 16 8 i) := by
  apply rope_transform_k0_head_compute_correct
  intro a b h
  apply Fin.ext
  simp [qOffset, rowIndex, dimIndex] at h
  omega

theorem rope_transform_k1_python_test_shape_compute_correct
    (K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_transform_k1_head K COS SIN HEAD_IDX COS_ROW_IDX
        128 8 8 16 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active HEAD_IDX 8 8 i)
        (fun i => (K, q1WriteOffset s HEAD_IDX 128 16 8 i)))
      (expected := fun i =>
        ropeQ1Spec s K COS SIN HEAD_IDX COS_ROW_IDX 128 8 8 16 8 i) := by
  apply rope_transform_k1_head_compute_correct
  intro a b h
  apply Fin.ext
  simp [q1WriteOffset, dimIndex] at h
  omega

/-- Python test-shape coverage for all four forward stores exposed by the
one-head RoPE slices: Q first/second half and K first/second half. -/
theorem rope_transform_python_test_shape_all_outputs_compute_correct
    (Q K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX : Nat)
    (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := rope_transform_q0_head Q COS SIN HEAD_IDX COS_ROW_IDX
        128 8 8 16 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active HEAD_IDX 8 8 i)
        (fun i => (Q, qOffset s HEAD_IDX 128 16 (dimIndex i))))
      (expected := fun i =>
        ropeQ0Spec s Q COS SIN HEAD_IDX COS_ROW_IDX 128 8 8 16 8 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_transform_q1_head Q COS SIN HEAD_IDX COS_ROW_IDX
        128 8 8 16 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active HEAD_IDX 8 8 i)
        (fun i => (Q, q1WriteOffset s HEAD_IDX 128 16 8 i)))
      (expected := fun i =>
        ropeQ1Spec s Q COS SIN HEAD_IDX COS_ROW_IDX 128 8 8 16 8 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_transform_k0_head K COS SIN HEAD_IDX COS_ROW_IDX
        128 8 8 16 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active HEAD_IDX 8 8 i)
        (fun i => (K, qOffset s HEAD_IDX 128 16 (dimIndex i))))
      (expected := fun i =>
        ropeQ0Spec s K COS SIN HEAD_IDX COS_ROW_IDX 128 8 8 16 8 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_transform_k1_head K COS SIN HEAD_IDX COS_ROW_IDX
        128 8 8 16 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active HEAD_IDX 8 8 i)
        (fun i => (K, q1WriteOffset s HEAD_IDX 128 16 8 i)))
      (expected := fun i =>
        ropeQ1Spec s K COS SIN HEAD_IDX COS_ROW_IDX 128 8 8 16 8 i)) := by
  constructor
  · exact rope_transform_q0_python_test_shape_compute_correct Q COS SIN
      HEAD_IDX COS_ROW_IDX s
  · constructor
    · exact rope_transform_q1_python_test_shape_compute_correct Q COS SIN
        HEAD_IDX COS_ROW_IDX s
    · constructor
      · exact rope_transform_k0_python_test_shape_compute_correct K COS SIN
          HEAD_IDX COS_ROW_IDX s
      · exact rope_transform_k1_python_test_shape_compute_correct K COS SIN
          HEAD_IDX COS_ROW_IDX s

/-- Public Python forward summary for `rope_transform.py`: the full
`BACKWARD_PASS = false` surface lowers for the checked test shape, and the
one-head proof slices cover the four Python-observable forward stores
(Q/K first and second halves). The remaining #153 blocker is the
`rope-head-slice-lift` from head slices to the full head tile. -/
theorem rope_transform_python_forward_store_summary
    (Q K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX : Nat)
    (s : BlockState) :
    (∃ alg, (triton_rope_surface Q K COS SIN
      128 128 8 8 4 2 8 8 16 8 8 16 8 Bool.false).toAlgorithm? =
        Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := rope_transform_q0_head Q COS SIN HEAD_IDX COS_ROW_IDX
        128 8 8 16 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active HEAD_IDX 8 8 i)
        (fun i => (Q, qOffset s HEAD_IDX 128 16 (dimIndex i))))
      (expected := fun i =>
        ropeQ0Spec s Q COS SIN HEAD_IDX COS_ROW_IDX 128 8 8 16 8 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_transform_q1_head Q COS SIN HEAD_IDX COS_ROW_IDX
        128 8 8 16 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active HEAD_IDX 8 8 i)
        (fun i => (Q, q1WriteOffset s HEAD_IDX 128 16 8 i)))
      (expected := fun i =>
        ropeQ1Spec s Q COS SIN HEAD_IDX COS_ROW_IDX 128 8 8 16 8 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_transform_k0_head K COS SIN HEAD_IDX COS_ROW_IDX
        128 8 8 16 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active HEAD_IDX 8 8 i)
        (fun i => (K, qOffset s HEAD_IDX 128 16 (dimIndex i))))
      (expected := fun i =>
        ropeQ0Spec s K COS SIN HEAD_IDX COS_ROW_IDX 128 8 8 16 8 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_transform_k1_head K COS SIN HEAD_IDX COS_ROW_IDX
        128 8 8 16 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active HEAD_IDX 8 8 i)
        (fun i => (K, q1WriteOffset s HEAD_IDX 128 16 8 i)))
      (expected := fun i =>
        ropeQ1Spec s K COS SIN HEAD_IDX COS_ROW_IDX 128 8 8 16 8 i))) := by
  constructor
  · exact triton_rope_surface_toAlgorithm_supported Q K COS SIN
      128 128 8 8 4 2 8 8 16 8 8 16 8 Bool.false
  · exact rope_transform_python_test_shape_all_outputs_compute_correct
      Q K COS SIN HEAD_IDX COS_ROW_IDX s

/-! ## Full-kernel 2D forward correctness (`BACKWARD_PASS = false`)

The single-head slice proofs above pin one Q/K head and one dim slice, but the
Python kernel writes a full `[pad_n_qh, pad_hd/2]` tile (and similarly for K).
The following theorems close the four full-kernel forward stores against the
real `triton_rope_surface` with `BACKWARD_PASS = false`, following the same
foldl-peel + scatter-readback strategy used in `RopeBackwardTransform.lean`.

Sign convention in the forward branch:
* `new_q_tile_1 = q_tile_1 * cos - q_tile_2 * sin`
* `new_q_tile_2 = q_tile_2 * cos + q_tile_1 * sin`
* `new_k_tile_1 = k_tile_1 * cos - k_tile_2 * sin`
* `new_k_tile_2 = k_tile_2 * cos + k_tile_1 * sin` -/

/-- Tile-level Q first-half offset (target of store #1 in the foldl chain). -/
def qFullFirstOffset
    (s : BlockState) (q_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : Nat :=
  s.pids 0 * q_row_stride + idx.1.val * hd + idx.2.1.val

/-- Tile-level Q second-half offset (target of store #2). -/
def qFullSecondOffset
    (s : BlockState) (q_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : Nat :=
  s.pids 0 * q_row_stride + idx.1.val * hd + idx.2.1.val + hd / 2

/-- Cos offset for the full kernel's Q stores. -/
def cosFullFirstOffset
    (s : BlockState) (sl cos_row_stride : Nat)
    (idx : TileIndex [pad_hd_half]) : Nat :=
  s.pids 0 % sl * cos_row_stride + idx.1.val

/-- Sin offset for the full kernel's Q stores. -/
def sinFullFirstOffset
    (s : BlockState) (sl sin_row_stride : Nat)
    (idx : TileIndex [pad_hd_half]) : Nat :=
  s.pids 0 % sl * sin_row_stride + idx.1.val

/-- Active predicate for the Q-side stores of the full kernel. -/
def activeQFull (n_qh hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : Prop :=
  idx.1.val < n_qh ∧ idx.2.1.val < hd / 2

instance activeQFullDecidable (n_qh hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) :
    Decidable (activeQFull n_qh hd idx) := by
  unfold activeQFull
  infer_instance

/-- Disjointness witness: every Q second-half write offset differs from an
active Q first-half offset (uses `d < hd / 2` and `hd / 2 + hd / 2 ≤ hd`). -/
theorem qFirstHalf_ne_qSecondHalf
    {pad_n_qh pad_hd_half : Nat} (s : BlockState) (q_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half])
    (hd_lt : idx.2.1.val < hd / 2)
    (k : TileIndex [pad_n_qh, pad_hd_half])
    (hkActive : k.2.1.val < hd / 2) :
    qFullFirstOffset s q_row_stride hd idx ≠
      qFullSecondOffset s q_row_stride hd k := by
  unfold qFullFirstOffset qFullSecondOffset
  intro heq
  have hcancel :
      idx.1.val * hd + idx.2.1.val =
        k.1.val * hd + k.2.1.val + hd / 2 := by
    omega
  rcases lt_trichotomy idx.1.val k.1.val with hlt | heq_n | hgt
  · have h1 : (idx.1.val + 1) * hd ≤ k.1.val * hd :=
      Nat.mul_le_mul_right _ hlt
    have h2 : idx.1.val * hd + hd ≤ k.1.val * hd := by
      have := h1
      simpa [Nat.add_mul, Nat.one_mul] using this
    have hLHS : idx.1.val * hd + idx.2.1.val < idx.1.val * hd + hd := by omega
    have hRHS : k.1.val * hd ≤ k.1.val * hd + k.2.1.val + hd / 2 := by omega
    have : idx.1.val * hd + idx.2.1.val < k.1.val * hd + k.2.1.val + hd / 2 := by
      calc idx.1.val * hd + idx.2.1.val
          < idx.1.val * hd + hd := hLHS
        _ ≤ k.1.val * hd := h2
        _ ≤ k.1.val * hd + k.2.1.val + hd / 2 := hRHS
    exact absurd hcancel (Nat.ne_of_lt this)
  · rw [heq_n] at hcancel
    have : idx.2.1.val ≥ hd / 2 := by omega
    omega
  · have h1 : (k.1.val + 1) * hd ≤ idx.1.val * hd :=
      Nat.mul_le_mul_right _ hgt
    have h2 : k.1.val * hd + hd ≤ idx.1.val * hd := by
      have := h1
      simpa [Nat.add_mul, Nat.one_mul] using this
    have hsum : k.2.1.val + hd / 2 < hd := by
      have hh : hd / 2 + hd / 2 ≤ hd := by omega
      omega
    have hRHS : k.1.val * hd + k.2.1.val + hd / 2 < k.1.val * hd + hd := by omega
    have : k.1.val * hd + k.2.1.val + hd / 2 < idx.1.val * hd + idx.2.1.val := by
      calc k.1.val * hd + k.2.1.val + hd / 2
          < k.1.val * hd + hd := hRHS
        _ ≤ idx.1.val * hd := h2
        _ ≤ idx.1.val * hd + idx.2.1.val := by omega
    exact absurd hcancel.symm (Nat.ne_of_lt this)

/-- Spec for the Q first-half output under `BACKWARD_PASS = false`:
`new_q_tile_1 = q_tile_1 * cos - q_tile_2 * sin`. -/
noncomputable def ropeForwardKernelQ0Spec
    (s : BlockState) (q_ptr cos sin : RegionName)
    (q_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : ℝ :=
  s.readMem q_ptr (qFullFirstOffset s q_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) -
  s.readMem q_ptr (qFullSecondOffset s q_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))

/-- Spec for the Q second-half output under `BACKWARD_PASS = false`:
`new_q_tile_2 = q_tile_2 * cos + q_tile_1 * sin`. -/
noncomputable def ropeForwardKernelQ1Spec
    (s : BlockState) (q_ptr cos sin : RegionName)
    (q_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : ℝ :=
  s.readMem q_ptr (qFullSecondOffset s q_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) +
  s.readMem q_ptr (qFullFirstOffset s q_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))

/-- Tile-level K first-half offset (target of store #3). -/
def kFullFirstOffset
    (s : BlockState) (k_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : Nat :=
  s.pids 0 * k_row_stride + idx.1.val * hd + idx.2.1.val

/-- Tile-level K second-half offset (target of store #4). -/
def kFullSecondOffset
    (s : BlockState) (k_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : Nat :=
  s.pids 0 * k_row_stride + idx.1.val * hd + idx.2.1.val + hd / 2

/-- Active predicate for the K-side stores of the full kernel. -/
def activeKFull (n_kh hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : Prop :=
  idx.1.val < n_kh ∧ idx.2.1.val < hd / 2

instance activeKFullDecidable (n_kh hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) :
    Decidable (activeKFull n_kh hd idx) := by
  unfold activeKFull
  infer_instance

/-- K-side disjointness witness, structurally identical to
`qFirstHalf_ne_qSecondHalf`. -/
theorem kFirstHalf_ne_kSecondHalf
    {pad_n_kh pad_hd_half : Nat} (s : BlockState) (k_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half])
    (hd_lt : idx.2.1.val < hd / 2)
    (k : TileIndex [pad_n_kh, pad_hd_half])
    (hkActive : k.2.1.val < hd / 2) :
    kFullFirstOffset s k_row_stride hd idx ≠
      kFullSecondOffset s k_row_stride hd k := by
  unfold kFullFirstOffset kFullSecondOffset
  intro heq
  have hcancel :
      idx.1.val * hd + idx.2.1.val =
        k.1.val * hd + k.2.1.val + hd / 2 := by
    omega
  rcases lt_trichotomy idx.1.val k.1.val with hlt | heq_n | hgt
  · have h1 : (idx.1.val + 1) * hd ≤ k.1.val * hd :=
      Nat.mul_le_mul_right _ hlt
    have h2 : idx.1.val * hd + hd ≤ k.1.val * hd := by
      have := h1
      simpa [Nat.add_mul, Nat.one_mul] using this
    have hLHS : idx.1.val * hd + idx.2.1.val < idx.1.val * hd + hd := by omega
    have hRHS : k.1.val * hd ≤ k.1.val * hd + k.2.1.val + hd / 2 := by omega
    have : idx.1.val * hd + idx.2.1.val < k.1.val * hd + k.2.1.val + hd / 2 := by
      calc idx.1.val * hd + idx.2.1.val
          < idx.1.val * hd + hd := hLHS
        _ ≤ k.1.val * hd := h2
        _ ≤ k.1.val * hd + k.2.1.val + hd / 2 := hRHS
    exact absurd hcancel (Nat.ne_of_lt this)
  · rw [heq_n] at hcancel
    have : idx.2.1.val ≥ hd / 2 := by omega
    omega
  · have h1 : (k.1.val + 1) * hd ≤ idx.1.val * hd :=
      Nat.mul_le_mul_right _ hgt
    have h2 : k.1.val * hd + hd ≤ idx.1.val * hd := by
      have := h1
      simpa [Nat.add_mul, Nat.one_mul] using this
    have hsum : k.2.1.val + hd / 2 < hd := by
      have hh : hd / 2 + hd / 2 ≤ hd := by omega
      omega
    have hRHS : k.1.val * hd + k.2.1.val + hd / 2 < k.1.val * hd + hd := by omega
    have : k.1.val * hd + k.2.1.val + hd / 2 < idx.1.val * hd + idx.2.1.val := by
      calc k.1.val * hd + k.2.1.val + hd / 2
          < k.1.val * hd + hd := hRHS
        _ ≤ idx.1.val * hd := h2
        _ ≤ idx.1.val * hd + idx.2.1.val := by omega
    exact absurd hcancel.symm (Nat.ne_of_lt this)

/-- Spec for the K first-half output under `BACKWARD_PASS = false`:
`new_k_tile_1 = k_tile_1 * cos - k_tile_2 * sin`. -/
noncomputable def ropeForwardKernelK0Spec
    (s : BlockState) (k_ptr cos sin : RegionName)
    (k_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : ℝ :=
  s.readMem k_ptr (kFullFirstOffset s k_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) -
  s.readMem k_ptr (kFullSecondOffset s k_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))

/-- Spec for the K second-half output under `BACKWARD_PASS = false`:
`new_k_tile_2 = k_tile_2 * cos + k_tile_1 * sin`. -/
noncomputable def ropeForwardKernelK1Spec
    (s : BlockState) (k_ptr cos sin : RegionName)
    (k_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : ℝ :=
  s.readMem k_ptr (kFullSecondOffset s k_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) +
  s.readMem k_ptr (kFullFirstOffset s k_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))

/-! ## Combined one-row `o0` + `o1` rotary-style RoPE slice

This combined slice mirrors `RotaryTransform.rotary_kernel_o0o1_row`: a single
kernel that performs BOTH the `o0 = x0 * cos - x1 * sin` first-half store and
the `o1 = x0 * sin + x1 * cos` second-half store into the same `OUT` region at
disjoint offsets. Correctness is established by stripping the other-half
`foldl` via `foldl_writeMem_same_region_disjoint_offsets_readMem` and then
closing the projected store via `scatter_readback_prop_masked_nd`. -/

def ropeRowIndex (s : BlockState) (BLOCK_M : Nat) : Nat :=
  s.pids 0 * BLOCK_M

def ropeDimIndex (i : Fin BLOCK_HALF) : Nat :=
  i.val

def ropeActive (s : BlockState) (seqlen rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Prop :=
  ropeRowIndex s BLOCK_M < seqlen ∧ ropeDimIndex i < rotary_dim_half

instance ropeActiveDecidable (s : BlockState)
    (seqlen rotary_dim_half BLOCK_M : Nat) (i : Fin BLOCK_HALF) :
    Decidable (ropeActive s seqlen rotary_dim_half BLOCK_M i) := by
  unfold ropeActive
  infer_instance

def ropeOutOffset
    (s : BlockState)
    (stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
    ropeRowIndex s BLOCK_M * stride_out_seqlen +
    ropeDimIndex i * stride_out_headdim

def ropeOut1Offset
    (s : BlockState)
    (stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
    ropeRowIndex s BLOCK_M * stride_out_seqlen +
    (ropeDimIndex i + rotary_dim_half) * stride_out_headdim

def ropeX0Offset
    (s : BlockState)
    (stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_x_batch + s.pids 2 * stride_x_nheads +
    ropeRowIndex s BLOCK_M * stride_x_seqlen +
    ropeDimIndex i * stride_x_headdim

def ropeX1Offset
    (s : BlockState)
    (stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_x_batch + s.pids 2 * stride_x_nheads +
    ropeRowIndex s BLOCK_M * stride_x_seqlen +
    (ropeDimIndex i + rotary_dim_half) * stride_x_headdim

def ropeRotOffset (s : BlockState) (SEQLEN_OFFSETS rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  (ropeRowIndex s BLOCK_M + SEQLEN_OFFSETS) * rotary_dim_half + ropeDimIndex i

noncomputable def ropeO0Spec
    (s : BlockState) (X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen_ro stride_x_batch stride_x_seqlen stride_x_nheads
      stride_x_headdim rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : ℝ :=
  let cosVal :=
    if ropeRowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        ropeDimIndex i < rotary_dim_half then
      s.readMem COS (ropeRotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      1.0
  let sinVal :=
    if ropeRowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        ropeDimIndex i < rotary_dim_half then
      s.readMem SIN (ropeRotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      0.0
  s.readMem X
      (ropeX0Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim BLOCK_M i) *
    cosVal -
  s.readMem X
      (ropeX1Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim rotary_dim_half BLOCK_M i) *
    sinVal

noncomputable def ropeO1Spec
    (s : BlockState) (X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen_ro stride_x_batch stride_x_seqlen stride_x_nheads
      stride_x_headdim rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : ℝ :=
  let cosVal :=
    if ropeRowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        ropeDimIndex i < rotary_dim_half then
      s.readMem COS (ropeRotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      1.0
  let sinVal :=
    if ropeRowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        ropeDimIndex i < rotary_dim_half then
      s.readMem SIN (ropeRotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      0.0
  s.readMem X
      (ropeX0Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim BLOCK_M i) *
    sinVal +
  s.readMem X
      (ropeX1Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim rotary_dim_half BLOCK_M i) *
    cosVal

/-- Combined one-row first-and-second-half RoPE slice.

Performs BOTH the `o0 = x0 * cos - x1 * sin` first-half store and the
`o1 = x0 * sin + x1 * cos` second-half store in a single kernel, mirroring the
non-interleaved branch of the canonical RoPE/rotary forward kernel. -/
def rope_kernel_o0o1_row
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

/-- Algorithm-layer correctness for the combined RoPE `o0` + `o1` store,
projected on the first-half (`o0`) output position. -/
theorem rope_kernel_o0o1_row_o0_correct
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        ropeOutOffset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim BLOCK_M i))
    (hStrideHd : stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ rotary_dim_half)
    (hExec : exec (rope_kernel_o0o1_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem OUT
          (ropeOutOffset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim BLOCK_M i) =
        if ropeActive s seqlen rotary_dim_half BLOCK_M i then
          ropeO0Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
            stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
            BLOCK_M i
        else
          s.readMem OUT
            (ropeOutOffset s stride_out_batch stride_out_seqlen stride_out_nheads
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
      simpa [ropeOutOffset, ropeRowIndex, ropeDimIndex, Nat.mul_assoc] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
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
  · simp [exec, rope_kernel_o0o1_row, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [ropeOutOffset, ropeRowIndex, ropeDimIndex]
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
        · simp [ropeActive, ropeO0Spec, ropeOutOffset, ropeX0Offset, ropeX1Offset,
                ropeRotOffset, ropeRowIndex, ropeDimIndex, hRow, hDim, hRot,
                Option.map₂, Option.bind, Option.map]
        · simp [ropeActive, ropeO0Spec, ropeOutOffset, ropeX0Offset, ropeX1Offset,
                ropeRotOffset, ropeRowIndex, ropeDimIndex, hRow, hDim, hRot,
                Option.map₂, Option.bind, Option.map]
      · simp [ropeActive, ropeRowIndex, ropeDimIndex, hRow, hDim]
    · simp [ropeActive, ropeRowIndex, ropeDimIndex, hRow]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Algorithm-layer correctness for the combined RoPE `o0` + `o1` store,
projected on the second-half (`o1`) output position. -/
theorem rope_kernel_o0o1_row_o1_correct
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        ropeOut1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim rotary_dim_half BLOCK_M i))
    (hStrideHd : stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ rotary_dim_half)
    (hExec : exec (rope_kernel_o0o1_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem OUT
          (ropeOut1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim rotary_dim_half BLOCK_M i) =
        if ropeActive s seqlen rotary_dim_half BLOCK_M i then
          ropeO1Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
            stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
            BLOCK_M i
        else
          s.readMem OUT
            (ropeOut1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
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
      simpa [ropeOut1Offset, ropeRowIndex, ropeDimIndex, Nat.mul_assoc] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
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
  · simp [exec, rope_kernel_o0o1_row, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [ropeOut1Offset, ropeRowIndex, ropeDimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
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
        · simp [ropeActive, ropeO1Spec, ropeOut1Offset, ropeX0Offset, ropeX1Offset,
                ropeRotOffset, ropeRowIndex, ropeDimIndex, hRow, hDim, hRot,
                Option.map₂, Option.bind, Option.map]
        · simp [ropeActive, ropeO1Spec, ropeOut1Offset, ropeX0Offset, ropeX1Offset,
                ropeRotOffset, ropeRowIndex, ropeDimIndex, hRow, hDim, hRot,
                Option.map₂, Option.bind, Option.map]
      · simp [ropeActive, ropeRowIndex, ropeDimIndex, hRow, hDim]
    · simp [ropeActive, ropeRowIndex, ropeDimIndex, hRow]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the combined RoPE row, projected on the
first-half output position. -/
theorem rope_kernel_o0o1_row_o0_compute_correct
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        ropeOutOffset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim BLOCK_M i))
    (hStrideHd : stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ rotary_dim_half) :
    ComputeCorrect.Realizes
      (kernel := rope_kernel_o0o1_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => ropeActive s seqlen rotary_dim_half BLOCK_M i)
        (fun i => (OUT,
          ropeOutOffset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim BLOCK_M i)))
      (expected := fun i =>
        ropeO0Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
          stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
          BLOCK_M i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rope_kernel_o0o1_row]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rope_kernel_o0o1_row_o0_correct OUT X COS SIN SEQLEN_OFFSETS
    seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
    stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
    stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF s s' hOutInj
    hStrideHd hHalfBound hExec i
  simpa [hActive] using h

/-- Compute-facing correctness for the combined RoPE row, projected on the
second-half output position. -/
theorem rope_kernel_o0o1_row_o1_compute_correct
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        ropeOut1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim rotary_dim_half BLOCK_M i))
    (hStrideHd : stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ rotary_dim_half) :
    ComputeCorrect.Realizes
      (kernel := rope_kernel_o0o1_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => ropeActive s seqlen rotary_dim_half BLOCK_M i)
        (fun i => (OUT,
          ropeOut1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim rotary_dim_half BLOCK_M i)))
      (expected := fun i =>
        ropeO1Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
          stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
          BLOCK_M i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rope_kernel_o0o1_row]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rope_kernel_o0o1_row_o1_correct OUT X COS SIN SEQLEN_OFFSETS
    seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
    stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
    stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF s s' hOutInj
    hStrideHd hHalfBound hExec i
  simpa [hActive] using h

/-- Compute-facing coverage for the combined RoPE row: both first-half and
second-half stores match the forward Python formula. -/
theorem rope_kernel_o0o1_row_all_outputs_compute_correct
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        ropeOutOffset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim BLOCK_M i))
    (hOut1Inj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        ropeOut1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim rotary_dim_half BLOCK_M i))
    (hStrideHd : stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ rotary_dim_half) :
    (ComputeCorrect.Realizes
      (kernel := rope_kernel_o0o1_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => ropeActive s seqlen rotary_dim_half BLOCK_M i)
        (fun i => (OUT,
          ropeOutOffset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim BLOCK_M i)))
      (expected := fun i =>
        ropeO0Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
          stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
          BLOCK_M i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_kernel_o0o1_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => ropeActive s seqlen rotary_dim_half BLOCK_M i)
        (fun i => (OUT,
          ropeOut1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim rotary_dim_half BLOCK_M i)))
      (expected := fun i =>
        ropeO1Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
          stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
          BLOCK_M i)) := by
  constructor
  · exact rope_kernel_o0o1_row_o0_compute_correct OUT X COS SIN SEQLEN_OFFSETS
      seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
      stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
      stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF s hOutInj
      hStrideHd hHalfBound
  · exact rope_kernel_o0o1_row_o1_compute_correct OUT X COS SIN SEQLEN_OFFSETS
      seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
      stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
      stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF s hOut1Inj
      hStrideHd hHalfBound



















theorem rope_transform_python_forward_output_summary
    (Q K COS SIN : RegionName)
    (s : BlockState) (qOffsetOf kOffsetOf : PUnit → Nat) :
    (∃ alg, (triton_rope_surface Q K COS SIN
      128 128 8 8 4 2 8 8 16 8 8 16 8 Bool.false).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_rope_surface Q K COS SIN
        128 128 8 8 4 2 8 8 16 8 8 16 8 Bool.false)
      (initialState := s)
      (write := fun i : PUnit => some (Q, qOffsetOf i))
      (expected := fun i : PUnit =>
        tritonRopeSurfaceValue s Q K COS SIN Q
          128 128 8 8 4 2 8 8 16 8 8 16 8 Bool.false (qOffsetOf i))) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_rope_surface Q K COS SIN
        128 128 8 8 4 2 8 8 16 8 8 16 8 Bool.false)
      (initialState := s)
      (write := fun i : PUnit => some (K, kOffsetOf i))
      (expected := fun i : PUnit =>
        tritonRopeSurfaceValue s Q K COS SIN K
          128 128 8 8 4 2 8 8 16 8 8 16 8 Bool.false (kOffsetOf i))) := by
  constructor
  · exact triton_rope_surface_toAlgorithm_supported Q K COS SIN
      128 128 8 8 4 2 8 8 16 8 8 16 8 Bool.false
  constructor
  · exact triton_rope_surface_output_compute_correct Q K COS SIN Q
      128 128 8 8 4 2 8 8 16 8 8 16 8 Bool.false s qOffsetOf
  · exact triton_rope_surface_output_compute_correct Q K COS SIN K
      128 128 8 8 4 2 8 8 16 8 8 16 8 Bool.false s kOffsetOf

end VeriTile.Bench.TritonBenchG.RopeTransform
