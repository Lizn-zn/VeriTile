import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.RopeTransform

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Surface transcription of the `BACKWARD_PASS=False` branch of
`rope_transform.py`'s `_triton_rope`.

Allowed mechanical Lean-syntax-only changes:
- `PAD_HALF` is Python's `pad_hd // 2`.
- `HEAD_HALF` is Python's `hd // 2`.
- Casts to `sin_row.dtype` are represented as fp32 casts in the current
  surface dtype model. -/
def triton_rope_forward
    (Q K COS SIN : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl n_qh n_kh hd pad_n_qh pad_n_kh PAD_HALF HEAD_HALF : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  q_ptr = Q + pid * $(q_row_stride)
  k_ptr = K + pid * $(k_row_stride)
  cos_row_idx = pid % $(sl)
  cos_base = COS + cos_row_idx * $(cos_row_stride)
  sin_base = SIN + cos_row_idx * $(sin_row_stride)
  cos_offsets = tl.arange(0, $(PAD_HALF))
  cos_mask = cos_offsets < $(HEAD_HALF)
  cos_row = tl.load(cos_base + cos_offsets, mask=cos_mask, other=0)
  sin_row = tl.load(sin_base + cos_offsets, mask=cos_mask, other=0)
  q_heads = tl.arange(0, $(pad_n_qh))
  k_heads = tl.arange(0, $(pad_n_kh))
  dims = tl.arange(0, $(PAD_HALF))
  first_half_q_offsets = q_heads[:, None] * $(hd) + dims[None, :]
  first_half_k_offsets = k_heads[:, None] * $(hd) + dims[None, :]
  first_q_mask = (q_heads[:, None] < $(n_qh)) & (dims[None, :] < $(HEAD_HALF))
  first_k_mask = (k_heads[:, None] < $(n_kh)) & (dims[None, :] < $(HEAD_HALF))
  q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask,
    other=0).to(SIN.dtype.element_ty)
  k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask,
    other=0).to(SIN.dtype.element_ty)
  second_half_q_offsets = first_half_q_offsets + $(HEAD_HALF)
  second_half_k_offsets = first_half_k_offsets + $(HEAD_HALF)
  q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=first_q_mask,
    other=0).to(SIN.dtype.element_ty)
  k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=first_k_mask,
    other=0).to(SIN.dtype.element_ty)
  new_q_tile_1 = q_tile_1 * cos_row[None, :] - q_tile_2 * sin_row[None, :]
  new_q_tile_2 = q_tile_2 * cos_row[None, :] + q_tile_1 * sin_row[None, :]
  new_k_tile_1 = k_tile_1 * cos_row[None, :] - k_tile_2 * sin_row[None, :]
  new_k_tile_2 = k_tile_2 * cos_row[None, :] + k_tile_1 * sin_row[None, :]
  tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
  tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=first_q_mask)
  tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
  tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=first_k_mask)
}

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
    mask=($(HEAD_IDX) < $(n_qh)) and (dim < $(HEAD_HALF)), other=0).to(SIN.dtype.element_ty)
  q1 = tl.load(q_base + dim + $(HEAD_HALF),
    mask=($(HEAD_IDX) < $(n_qh)) and (dim < $(HEAD_HALF)), other=0).to(SIN.dtype.element_ty)
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
  · simp [exec, rope_transform_q0_head, stepStmts, stepStmt, evalOp,
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

end VeriTile.Bench.TritonBenchG.RopeTransform
