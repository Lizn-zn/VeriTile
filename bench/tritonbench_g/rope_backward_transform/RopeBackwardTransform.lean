import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.RopeBackwardTransform

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `rope_backward_transform.py`'s `_triton_rope`. -/
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

/-- The full backward RoPE transform surface lowers to the algorithm layer,
including Q/K, both halves, and the `BACKWARD_PASS` branch. -/
theorem triton_rope_surface_toAlgorithm_supported
    (q_ptr k_ptr cos sin : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (BACKWARD_PASS : Bool) :
    ∃ alg, (triton_rope_surface q_ptr k_ptr cos sin q_row_stride k_row_stride
      cos_row_stride sin_row_stride sl bs n_qh n_kh hd pad_n_qh pad_n_kh
      pad_hd BLOCK_SIZE BACKWARD_PASS).toAlgorithm? = Except.ok alg := by
  simp [triton_rope_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented one-Q-head first-half backward slice of
`rope_backward_transform.py`'s `_triton_rope`.

This fixes one Q head and one row in the `BACKWARD_PASS=True` branch and proves
the first-half store: `q0' = q0 * cos + q1 * sin`. -/
def rope_backward_q0_head
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
  out = q0 * cos_row + q1 * sin_row
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

noncomputable def ropeBackwardQ0Spec
    (s : BlockState) (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      HEAD_HALF : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i)) *
    s.readMem COS (cosOffset COS_ROW_IDX cos_row_stride i) +
  s.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i + HEAD_HALF)) *
    s.readMem SIN (sinOffset COS_ROW_IDX sin_row_stride i)

/-- Algorithm-layer correctness for the one-Q-head backward RoPE store. -/
theorem rope_backward_q0_head_correct
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      n_qh HEAD_HALF BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF => qOffset s HEAD_IDX q_row_stride hd (dimIndex i)))
    (hExec : exec (rope_backward_q0_head Q COS SIN HEAD_IDX COS_ROW_IDX
        q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF
        BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i)) =
        if active HEAD_IDX n_qh HEAD_HALF i then
          ropeBackwardQ0Spec s Q COS SIN HEAD_IDX COS_ROW_IDX q_row_stride
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
  · simp [exec, rope_backward_q0_head, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [qOffset, rowIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hHead : HEAD_IDX < n_qh
    · by_cases hDim : i.val < HEAD_HALF
      · simp [active, ropeBackwardQ0Spec, qOffset, cosOffset, sinOffset,
              rowIndex, dimIndex, hHead, hDim, Option.bind, Option.map]
        left
        rw [Nat.add_assoc]
      · simp [active, dimIndex, hHead, hDim]
    · simp [active, hHead]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-Q-head backward RoPE store. -/
theorem rope_backward_q0_head_compute_correct
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      n_qh HEAD_HALF BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF => qOffset s HEAD_IDX q_row_stride hd (dimIndex i))) :
    ComputeCorrect.Realizes
      (kernel := rope_backward_q0_head Q COS SIN HEAD_IDX COS_ROW_IDX
        q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF
        BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active HEAD_IDX n_qh HEAD_HALF i)
        (fun i => (Q, qOffset s HEAD_IDX q_row_stride hd (dimIndex i))))
      (expected := fun i =>
        ropeBackwardQ0Spec s Q COS SIN HEAD_IDX COS_ROW_IDX q_row_stride
          cos_row_stride sin_row_stride hd HEAD_HALF i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rope_backward_q0_head]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rope_backward_q0_head_correct Q COS SIN HEAD_IDX COS_ROW_IDX
    q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF BLOCK_HALF
    s s' hOutInj hExec i
  simpa [hActive] using h

/-! ## Second-half backward Q store (`new_q_tile_2`) -/

/-- Slice for the backward second-half Q store:
`q1' = q1 * cos - q0 * sin` at `second_half_q_offsets`. -/
def rope_backward_q1_head
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
  out = q1 * cos_row - q0 * sin_row
  tl.store(q_base + dim + $(HEAD_HALF), out,
    mask=($(HEAD_IDX) < $(n_qh)) and (dim < $(HEAD_HALF)))
}

def q1WriteOffset
    (s : BlockState) (HEAD_IDX q_row_stride hd HEAD_HALF : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pid * q_row_stride + HEAD_IDX * hd + dimIndex i + HEAD_HALF

noncomputable def ropeBackwardQ1Spec
    (s : BlockState) (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      HEAD_HALF : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i) *
    s.readMem COS (cosOffset COS_ROW_IDX cos_row_stride i) -
  s.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i)) *
    s.readMem SIN (sinOffset COS_ROW_IDX sin_row_stride i)

theorem rope_backward_q1_head_correct
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      n_qh HEAD_HALF BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i))
    (hExec : exec (rope_backward_q1_head Q COS SIN HEAD_IDX COS_ROW_IDX
        q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF
        BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem Q (q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i) =
        if active HEAD_IDX n_qh HEAD_HALF i then
          ropeBackwardQ1Spec s Q COS SIN HEAD_IDX COS_ROW_IDX q_row_stride
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
  · simp [exec, rope_backward_q1_head, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [q1WriteOffset, rowIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hHead : HEAD_IDX < n_qh
    · by_cases hDim : i.val < HEAD_HALF
      · simp [active, ropeBackwardQ1Spec, q1WriteOffset, qOffset, cosOffset,
              sinOffset, rowIndex, dimIndex, hHead, hDim, Option.bind, Option.map]
      · simp [active, dimIndex, hHead, hDim]
    · simp [active, hHead]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem rope_backward_q1_head_compute_correct
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      n_qh HEAD_HALF BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i)) :
    ComputeCorrect.Realizes
      (kernel := rope_backward_q1_head Q COS SIN HEAD_IDX COS_ROW_IDX
        q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF
        BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active HEAD_IDX n_qh HEAD_HALF i)
        (fun i => (Q, q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i)))
      (expected := fun i =>
        ropeBackwardQ1Spec s Q COS SIN HEAD_IDX COS_ROW_IDX q_row_stride
          cos_row_stride sin_row_stride hd HEAD_HALF i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rope_backward_q1_head]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rope_backward_q1_head_correct Q COS SIN HEAD_IDX COS_ROW_IDX
    q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF BLOCK_HALF
    s s' hOutInj hExec i
  simpa [hActive] using h

/-! ## K-side backward coverage (alias of Q-side, with K region and `n_kh`) -/

abbrev rope_backward_k0_head
    (K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX k_row_stride cos_row_stride sin_row_stride hd
      n_kh HEAD_HALF BLOCK_HALF : Nat) :
    ComputeKernel :=
  rope_backward_q0_head K COS SIN HEAD_IDX COS_ROW_IDX k_row_stride
    cos_row_stride sin_row_stride hd n_kh HEAD_HALF BLOCK_HALF

abbrev rope_backward_k1_head
    (K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX k_row_stride cos_row_stride sin_row_stride hd
      n_kh HEAD_HALF BLOCK_HALF : Nat) :
    ComputeKernel :=
  rope_backward_q1_head K COS SIN HEAD_IDX COS_ROW_IDX k_row_stride
    cos_row_stride sin_row_stride hd n_kh HEAD_HALF BLOCK_HALF

theorem rope_backward_k0_head_compute_correct
    (K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX k_row_stride cos_row_stride sin_row_stride hd
      n_kh HEAD_HALF BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF => qOffset s HEAD_IDX k_row_stride hd (dimIndex i))) :
    ComputeCorrect.Realizes
      (kernel := rope_backward_k0_head K COS SIN HEAD_IDX COS_ROW_IDX
        k_row_stride cos_row_stride sin_row_stride hd n_kh HEAD_HALF BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active HEAD_IDX n_kh HEAD_HALF i)
        (fun i => (K, qOffset s HEAD_IDX k_row_stride hd (dimIndex i))))
      (expected := fun i =>
        ropeBackwardQ0Spec s K COS SIN HEAD_IDX COS_ROW_IDX k_row_stride
          cos_row_stride sin_row_stride hd HEAD_HALF i) :=
  rope_backward_q0_head_compute_correct K COS SIN HEAD_IDX COS_ROW_IDX
    k_row_stride cos_row_stride sin_row_stride hd n_kh HEAD_HALF BLOCK_HALF s
    hOutInj

theorem rope_backward_k1_head_compute_correct
    (K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX k_row_stride cos_row_stride sin_row_stride hd
      n_kh HEAD_HALF BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        q1WriteOffset s HEAD_IDX k_row_stride hd HEAD_HALF i)) :
    ComputeCorrect.Realizes
      (kernel := rope_backward_k1_head K COS SIN HEAD_IDX COS_ROW_IDX
        k_row_stride cos_row_stride sin_row_stride hd n_kh HEAD_HALF BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active HEAD_IDX n_kh HEAD_HALF i)
        (fun i => (K, q1WriteOffset s HEAD_IDX k_row_stride hd HEAD_HALF i)))
      (expected := fun i =>
        ropeBackwardQ1Spec s K COS SIN HEAD_IDX COS_ROW_IDX k_row_stride
          cos_row_stride sin_row_stride hd HEAD_HALF i) :=
  rope_backward_q1_head_compute_correct K COS SIN HEAD_IDX COS_ROW_IDX
    k_row_stride cos_row_stride sin_row_stride hd n_kh HEAD_HALF BLOCK_HALF s
    hOutInj

/-! ## Python test-shape wrappers

`rope_backward_transform.py`'s checked test case uses `batch_size = 2`,
`seq_len = 4`, `n_q_head = n_kv_head = 8`, and `head_dim = 16`. After
`transpose(...).contiguous()`, both gradient tensors have row stride
`8 * 16 = 128`, while cos/sin have row stride `16 / 2 = 8`. These wrappers
pin those metadata values for the `BACKWARD_PASS=True` Q/K stores. -/

theorem rope_backward_q0_python_test_shape_compute_correct
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_backward_q0_head Q COS SIN HEAD_IDX COS_ROW_IDX
        128 8 8 16 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active HEAD_IDX 8 8 i)
        (fun i => (Q, qOffset s HEAD_IDX 128 16 (dimIndex i))))
      (expected := fun i =>
        ropeBackwardQ0Spec s Q COS SIN HEAD_IDX COS_ROW_IDX 128 8 8 16 8 i) := by
  apply rope_backward_q0_head_compute_correct
  intro a b h
  apply Fin.ext
  simp [qOffset, rowIndex, dimIndex] at h
  omega

theorem rope_backward_q1_python_test_shape_compute_correct
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_backward_q1_head Q COS SIN HEAD_IDX COS_ROW_IDX
        128 8 8 16 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active HEAD_IDX 8 8 i)
        (fun i => (Q, q1WriteOffset s HEAD_IDX 128 16 8 i)))
      (expected := fun i =>
        ropeBackwardQ1Spec s Q COS SIN HEAD_IDX COS_ROW_IDX 128 8 8 16 8 i) := by
  apply rope_backward_q1_head_compute_correct
  intro a b h
  apply Fin.ext
  simp [q1WriteOffset, dimIndex] at h
  omega

theorem rope_backward_k0_python_test_shape_compute_correct
    (K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_backward_k0_head K COS SIN HEAD_IDX COS_ROW_IDX
        128 8 8 16 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active HEAD_IDX 8 8 i)
        (fun i => (K, qOffset s HEAD_IDX 128 16 (dimIndex i))))
      (expected := fun i =>
        ropeBackwardQ0Spec s K COS SIN HEAD_IDX COS_ROW_IDX 128 8 8 16 8 i) := by
  apply rope_backward_k0_head_compute_correct
  intro a b h
  apply Fin.ext
  simp [qOffset, rowIndex, dimIndex] at h
  omega

theorem rope_backward_k1_python_test_shape_compute_correct
    (K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_backward_k1_head K COS SIN HEAD_IDX COS_ROW_IDX
        128 8 8 16 8 8 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 8 => active HEAD_IDX 8 8 i)
        (fun i => (K, q1WriteOffset s HEAD_IDX 128 16 8 i)))
      (expected := fun i =>
        ropeBackwardQ1Spec s K COS SIN HEAD_IDX COS_ROW_IDX 128 8 8 16 8 i) := by
  apply rope_backward_k1_head_compute_correct
  intro a b h
  apply Fin.ext
  simp [q1WriteOffset, dimIndex] at h
  omega

/-! ## Full-kernel Q first-half store correctness (`BACKWARD_PASS = true`)

Per the #139 audit, the slice proofs above (one Q head, one row) aren't
sufficient. This section closes the Q first-half store for the entire
`triton_rope_surface` kernel under the `BACKWARD_PASS = true` branch.

The kernel issues four stores in sequence:
  1. Q at `first_half_q_offsets`  (the target whose readback we prove)
  2. Q at `second_half_q_offsets = first_half_q_offsets + hd / 2`
  3. K at `first_half_k_offsets`
  4. K at `second_half_k_offsets = first_half_k_offsets + hd / 2`

For the Q first-half readback we
* strip the K-side foldls via `foldl_writeMem_const_region_prop_masked_readMem_other`
  (cross-region: `Q ≠ K`);
* strip the Q second-half foldl via
  `foldl_writeMem_same_region_disjoint_offsets_readMem` (intra-region,
  disjoint offsets thanks to the `+ hd / 2` shift);
* finally apply `scatter_readback_prop_masked_nd` to the Q first-half foldl.

Disjointness uses `hd / 2 + hd / 2 ≤ hd` and the active-region bound
`d < hd / 2` to rule out wrap-around between adjacent heads. -/

/-- Tile-level Q first-half offset in the full `triton_rope_surface` kernel.
Tile shape is `[pad_n_qh, pad_hd / 2]`. -/
def qFullFirstOffset
    (s : BlockState) (q_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : Nat :=
  s.pids 0 * q_row_stride + idx.1.val * hd + idx.2.1.val

/-- Tile-level Q second-half offset (writes go here in store #2). -/
def qFullSecondOffset
    (s : BlockState) (q_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : Nat :=
  s.pids 0 * q_row_stride + idx.1.val * hd + idx.2.1.val + hd / 2

/-- Cos offset for the full kernel's first-half Q store. -/
def cosFullFirstOffset
    (s : BlockState) (sl cos_row_stride : Nat)
    (idx : TileIndex [pad_hd_half]) : Nat :=
  s.pids 0 % sl * cos_row_stride + idx.1.val

/-- Sin offset for the full kernel's first-half Q store. -/
def sinFullFirstOffset
    (s : BlockState) (sl sin_row_stride : Nat)
    (idx : TileIndex [pad_hd_half]) : Nat :=
  s.pids 0 % sl * sin_row_stride + idx.1.val

/-- Active predicate for the Q first-half store of `triton_rope_surface`. -/
def activeQFull (n_qh hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : Prop :=
  idx.1.val < n_qh ∧ idx.2.1.val < hd / 2

instance activeQFullDecidable (n_qh hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) :
    Decidable (activeQFull n_qh hd idx) := by
  unfold activeQFull
  infer_instance

/-- Specification for the full kernel's Q first-half output, under
`BACKWARD_PASS = true` (i.e. `new_q_tile_1 = q_tile_1 * cos + q_tile_2 * sin`). -/
noncomputable def ropeBackwardKernelQ0Spec
    (s : BlockState) (q_ptr cos sin : RegionName)
    (q_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : ℝ :=
  s.readMem q_ptr (qFullFirstOffset s q_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) +
  s.readMem q_ptr (qFullSecondOffset s q_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))

/-- Disjointness witness: every Q second-half write offset differs from a
given active Q first-half read offset. The proof uses `d < hd / 2` (active)
and `hd / 2 + hd / 2 ≤ hd` to rule out cross-head wrap-around. -/
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
  -- s.pids 0 * q_row_stride cancels on both sides.
  have hcancel :
      idx.1.val * hd + idx.2.1.val =
        k.1.val * hd + k.2.1.val + hd / 2 := by
    omega
  -- Case split on n0 = idx.1.val vs n1 = k.1.val.
  rcases lt_trichotomy idx.1.val k.1.val with hlt | heq_n | hgt
  · -- n0 < n1: LHS < (n0+1)*hd ≤ n1*hd ≤ RHS, but RHS - LHS ≥ hd/2, fine.
    -- Actually need: n0*hd + d0 = n1*hd + d1 + hd/2 with n0 < n1.
    -- Then n1*hd + hd/2 ≤ n0*hd + d0 = LHS, so n1*hd ≤ n0*hd + d0 - hd/2.
    -- But n1 ≥ n0 + 1, so n1*hd ≥ n0*hd + hd, giving hd ≤ d0 - hd/2 if
    -- d0 ≥ hd/2; since d0 < hd/2, contradiction.
    have h1 : (idx.1.val + 1) * hd ≤ k.1.val * hd :=
      Nat.mul_le_mul_right _ hlt
    have h2 : idx.1.val * hd + hd ≤ k.1.val * hd := by
      have := h1
      simpa [Nat.add_mul, Nat.one_mul] using this
    have h3 : idx.1.val * hd + idx.2.1.val + hd / 2 < idx.1.val * hd + hd := by
      have hsum : idx.2.1.val + hd / 2 < hd := by
        have hh : hd / 2 + hd / 2 ≤ hd := by omega
        omega
      omega
    -- combine: LHS = idx.1*hd + d0, RHS = k.1*hd + d1 + hd/2.
    -- LHS < idx.1*hd + hd/2 + hd/2 ≤ idx.1*hd + hd ≤ k.1*hd ≤ RHS.
    -- We want LHS < RHS, contradicting hcancel.
    have hLHS : idx.1.val * hd + idx.2.1.val < idx.1.val * hd + hd := by omega
    have hRHS : k.1.val * hd ≤ k.1.val * hd + k.2.1.val + hd / 2 := by omega
    have : idx.1.val * hd + idx.2.1.val < k.1.val * hd + k.2.1.val + hd / 2 := by
      calc idx.1.val * hd + idx.2.1.val
          < idx.1.val * hd + hd := hLHS
        _ ≤ k.1.val * hd := h2
        _ ≤ k.1.val * hd + k.2.1.val + hd / 2 := hRHS
    exact absurd hcancel (Nat.ne_of_lt this)
  · -- n0 = n1: hcancel reduces to d0 = d1 + hd/2; since d0 < hd/2 and
    -- d1 + hd/2 ≥ hd/2, contradiction.
    rw [heq_n] at hcancel
    have hd0 : idx.2.1.val = k.2.1.val + hd / 2 := by omega
    have : idx.2.1.val ≥ hd / 2 := by omega
    omega
  · -- n0 > n1: symmetric. n0 ≥ n1 + 1, so n0*hd ≥ n1*hd + hd, and
    -- LHS = n0*hd + d0 ≥ n1*hd + hd > n1*hd + d1 + hd/2 = RHS.
    have h1 : (k.1.val + 1) * hd ≤ idx.1.val * hd :=
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

/-! ## Full-kernel Q second-half store correctness (`BACKWARD_PASS = true`)

Mirrors the Q first-half proof: target offset is `qFullSecondOffset`. The
foldl-stack is `Q1 . Q2 . K1 . K2` (innermost to outermost); we peel K2, K1
(cross-region), then apply `scatter_readback_prop_masked_nd` to Q2, and in
the inactive case peel Q1 via offset-disjointness. -/

/-- Spec for the full kernel's Q second-half output under
`BACKWARD_PASS = true`: `new_q_tile_2 = q_tile_2 * cos - q_tile_1 * sin`. -/
noncomputable def ropeBackwardKernelQ1Spec
    (s : BlockState) (q_ptr cos sin : RegionName)
    (q_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : ℝ :=
  s.readMem q_ptr (qFullSecondOffset s q_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) -
  s.readMem q_ptr (qFullFirstOffset s q_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))

/-! ## Full-kernel K first-half store correctness (`BACKWARD_PASS = true`)

Target region is `k_ptr` at `kFullFirstOffset` (= the `first_half_k_offsets`
write). Foldl-stack is `Q1 . Q2 . K1 . K2`; for reading at K1 offsets we
peel K2 via intra-region offset disjointness (K2 = K1 + hd/2), then apply
scatter to K1, and strip Q1, Q2 in the inactive case via cross-region. -/

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

/-- Active predicate for the K-side stores. -/
def activeKFull (n_kh hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : Prop :=
  idx.1.val < n_kh ∧ idx.2.1.val < hd / 2

instance activeKFullDecidable (n_kh hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) :
    Decidable (activeKFull n_kh hd idx) := by
  unfold activeKFull
  infer_instance

/-- K-side disjointness witness, structurally identical to
`qFirstHalf_ne_qSecondHalf` (only the variable names differ). -/
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
    have hd0 : idx.2.1.val = k.2.1.val + hd / 2 := by omega
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

/-- Spec for the full kernel's K first-half output, under
`BACKWARD_PASS = true`: `new_k_tile_1 = k_tile_1 * cos + k_tile_2 * sin`. -/
noncomputable def ropeBackwardKernelK0Spec
    (s : BlockState) (k_ptr cos sin : RegionName)
    (k_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : ℝ :=
  s.readMem k_ptr (kFullFirstOffset s k_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) +
  s.readMem k_ptr (kFullSecondOffset s k_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))

/-! ## Full-kernel K second-half store correctness (`BACKWARD_PASS = true`)

Target offset is `kFullSecondOffset` (= store #4, outermost foldl). We
apply scatter to K2 directly, then peel K1, Q2, Q1 in the inactive case. -/

/-- Spec for the full kernel's K second-half output, under
`BACKWARD_PASS = true`: `new_k_tile_2 = k_tile_2 * cos - k_tile_1 * sin`. -/
noncomputable def ropeBackwardKernelK1Spec
    (s : BlockState) (k_ptr cos sin : RegionName)
    (k_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : ℝ :=
  s.readMem k_ptr (kFullSecondOffset s k_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) -
  s.readMem k_ptr (kFullFirstOffset s k_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))

/-! ## Python full-kernel test-shape facts

The checked Python backward test uses `batch_size = 2`, `seq_len = 4`,
`n_q_head = n_kv_head = 8`, and `head_dim = 16`. After
`transpose(...).contiguous()`, Q and K rows have stride `8 * 16 = 128`, and
the full-kernel tiles have shape `[8, 8]` for heads by half-head dimension.
The facts below pin the active masks and no-collision obligations for the
full-kernel Q/K first- and second-half store offsets. -/

theorem activeQFull_python_test_shape
    (idx : TileIndex [8, 8]) :
    activeQFull 8 16 idx := by
  unfold activeQFull
  constructor <;> omega

theorem activeKFull_python_test_shape
    (idx : TileIndex [8, 8]) :
    activeKFull 8 16 idx := by
  unfold activeKFull
  constructor <;> omega

theorem qFullFirstOffset_python_test_shape_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [8, 8] => qFullFirstOffset s 128 16 idx) := by
  intro a b h
  rcases a with ⟨ha, da, hta⟩
  rcases b with ⟨hb, db, htb⟩
  rcases da with ⟨da, hda⟩
  rcases db with ⟨db, hdb⟩
  simp [qFullFirstOffset] at h
  have hh : ha = hb := by omega
  have hd : da = db := by omega
  subst hb
  subst db
  rfl

theorem qFullSecondOffset_python_test_shape_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [8, 8] => qFullSecondOffset s 128 16 idx) := by
  intro a b h
  rcases a with ⟨ha, da, hta⟩
  rcases b with ⟨hb, db, htb⟩
  rcases da with ⟨da, hda⟩
  rcases db with ⟨db, hdb⟩
  simp [qFullSecondOffset] at h
  have hh : ha = hb := by omega
  have hd : da = db := by omega
  subst hb
  subst db
  rfl

theorem kFullFirstOffset_python_test_shape_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [8, 8] => kFullFirstOffset s 128 16 idx) := by
  intro a b h
  rcases a with ⟨ha, da, hta⟩
  rcases b with ⟨hb, db, htb⟩
  rcases da with ⟨da, hda⟩
  rcases db with ⟨db, hdb⟩
  simp [kFullFirstOffset] at h
  have hh : ha = hb := by omega
  have hd : da = db := by omega
  subst hb
  subst db
  rfl

theorem kFullSecondOffset_python_test_shape_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [8, 8] => kFullSecondOffset s 128 16 idx) := by
  intro a b h
  rcases a with ⟨ha, da, hta⟩
  rcases b with ⟨hb, db, htb⟩
  rcases da with ⟨da, hda⟩
  rcases db with ⟨db, hdb⟩
  simp [kFullSecondOffset] at h
  have hh : ha = hb := by omega
  have hd : da = db := by omega
  subst hb
  subst db
  rfl

end VeriTile.Bench.TritonBenchG.RopeBackwardTransform
