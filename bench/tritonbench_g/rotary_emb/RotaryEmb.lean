import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.RotaryEmb

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `rotary_emb.py`'s `_rotary_kernel`.

This keeps the block-shaped sequence/head/dimension tiles and writes both Q and
K rotary pairs over the full `[BLOCK_SEQ, BLOCK_HEAD, BLOCK_DMODEL / 2]`
surface. -/
def rotary_kernel_surface
    (Q K Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len
      HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)

  cur_head_range = cur_head_index * $(BLOCK_HEAD) + tl.arange(0, $(BLOCK_HEAD))
  cur_seq_range = cur_seq_index * $(BLOCK_SEQ) + tl.arange(0, $(BLOCK_SEQ))

  dim_range0 = tl.arange(0, $(BLOCK_DMODEL) // $(2)) * $(2)
  dim_range1 = tl.arange(0, $(BLOCK_DMODEL) // $(2)) * $(2) + $(1)

  off_q0 = cur_seq_range[:, None, None] * $(stride_qbs) +
    cur_head_range[None, :, None] * $(stride_qh) +
    dim_range0[None, None, :] * $(stride_qd)
  off_q1 = cur_seq_range[:, None, None] * $(stride_qbs) +
    cur_head_range[None, :, None] * $(stride_qh) +
    dim_range1[None, None, :] * $(stride_qd)

  off_dimcos_sin0 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range0[None, None, :] * $(stride_cosd)
  off_dimcos_sin1 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range1[None, None, :] * $(stride_cosd)

  q0 = tl.load(Q + off_q0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + off_q1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)),
    other=0.0)

  cos0 = tl.load(Cos + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  cos1 = tl.load(Cos + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin1 = tl.load(Sin + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  out0 = q0 * cos0 - q1 * sin0
  out1 = q0 * sin1 + q1 * cos1

  tl.store(Q + off_q0, out0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)))
  tl.store(Q + off_q1, out1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)))

  off_k0 = cur_seq_range[:, None, None] * $(stride_kbs) +
    cur_head_range[None, :, None] * $(stride_kh) +
    dim_range0[None, None, :] * $(stride_kd)
  off_k1 = cur_seq_range[:, None, None] * $(stride_kbs) +
    cur_head_range[None, :, None] * $(stride_kh) +
    dim_range1[None, None, :] * $(stride_kd)

  off_dimcos_sin0 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range0[None, None, :] * $(stride_cosd)
  off_dimcos_sin1 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range1[None, None, :] * $(stride_cosd)

  k0 = tl.load(K + off_k0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + off_k1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)),
    other=0.0)

  cos0 = tl.load(Cos + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  cos1 = tl.load(Cos + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin1 = tl.load(Sin + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  out_k0 = k0 * cos0 - k1 * sin0
  out_k1 = k0 * sin1 + k1 * cos1

  tl.store(K + off_k0, out_k0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)))
  tl.store(K + off_k1, out_k1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)))
}

/-- Surface transcription of the Q part of `rotary_emb.py`'s `_rotary_kernel`
under the existing one-sequence/one-head tile abstraction used in this file.

The Python kernel is block-shaped over sequence/head dimensions and also writes
K. This surface closes the Q-side gap left by the proof slice below by writing
both even and odd rotary halves. -/
def rotary_emb_q_surface
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  active = (cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q))
  off_q0 = cur_seq_index * $(stride_qbs) + cur_head_index * $(stride_qh) +
    dim0 * $(stride_qd)
  off_q1 = cur_seq_index * $(stride_qbs) + cur_head_index * $(stride_qh) +
    dim1 * $(stride_qd)
  off_cos0 = cur_seq_index * $(stride_cosbs) + dim0 * $(stride_cosd)
  off_sin0 = cur_seq_index * $(stride_sinbs) + dim0 * $(stride_sind)
  off_cos1 = cur_seq_index * $(stride_cosbs) + dim1 * $(stride_cosd)
  off_sin1 = cur_seq_index * $(stride_sinbs) + dim1 * $(stride_sind)
  q0 = tl.load(Q + off_q0, mask=active, other=0.0)
  q1 = tl.load(Q + off_q1, mask=active, other=0.0)
  cos0 = tl.load(Cos + off_cos0, mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + off_sin0, mask=cur_seq_index < $(max_total_len), other=0.0)
  cos1 = tl.load(Cos + off_cos1, mask=cur_seq_index < $(max_total_len), other=0.0)
  sin1 = tl.load(Sin + off_sin1, mask=cur_seq_index < $(max_total_len), other=0.0)
  out0 = q0 * cos0 - q1 * sin0
  out1 = q0 * sin1 + q1 * cos1
  tl.store(Q + off_q0, out0, mask=active)
  tl.store(Q + off_q1, out1, mask=active)
}

/-- Surface transcription of the K part of `rotary_emb.py`'s `_rotary_kernel`
under the existing one-sequence/one-head tile abstraction. -/
def rotary_emb_k_surface
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  active = (cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K))
  off_k0 = cur_seq_index * $(stride_kbs) + cur_head_index * $(stride_kh) +
    dim0 * $(stride_kd)
  off_k1 = cur_seq_index * $(stride_kbs) + cur_head_index * $(stride_kh) +
    dim1 * $(stride_kd)
  off_cos0 = cur_seq_index * $(stride_cosbs) + dim0 * $(stride_cosd)
  off_sin0 = cur_seq_index * $(stride_sinbs) + dim0 * $(stride_sind)
  off_cos1 = cur_seq_index * $(stride_cosbs) + dim1 * $(stride_cosd)
  off_sin1 = cur_seq_index * $(stride_sinbs) + dim1 * $(stride_sind)
  k0 = tl.load(K + off_k0, mask=active, other=0.0)
  k1 = tl.load(K + off_k1, mask=active, other=0.0)
  cos0 = tl.load(Cos + off_cos0, mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + off_sin0, mask=cur_seq_index < $(max_total_len), other=0.0)
  cos1 = tl.load(Cos + off_cos1, mask=cur_seq_index < $(max_total_len), other=0.0)
  sin1 = tl.load(Sin + off_sin1, mask=cur_seq_index < $(max_total_len), other=0.0)
  out0 = k0 * cos0 - k1 * sin0
  out1 = k0 * sin1 + k1 * cos1
  tl.store(K + off_k0, out0, mask=active)
  tl.store(K + off_k1, out1, mask=active)
}

/-- Proof-oriented Q-even-dimension slice of `rotary_emb.py`'s `_rotary_kernel`.

This models the first Q store for one sequence/head program tile:
`out0 = q0 * cos0 - q1 * sin0`, where even dimensions are paired with the
following odd dimension. -/
def rotary_emb_q0_block
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  q0 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim1 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  out0 = q0 * cos0 - q1 * sin0
  tl.store(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    out0,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)))
}

def seqIndex (s : BlockState) : Nat :=
  s.pids 1

def headIndex (s : BlockState) : Nat :=
  s.pids 0

def dimEven (i : Fin BLOCK_HALF) : Nat :=
  i.val * 2

def dimOdd (i : Fin BLOCK_HALF) : Nat :=
  i.val * 2 + 1

def active (s : BlockState) (max_total_len HEAD_Q : Nat) : Prop :=
  seqIndex s < max_total_len ∧ headIndex s < HEAD_Q

instance activeDecidable (s : BlockState) (max_total_len HEAD_Q : Nat) :
    Decidable (active s max_total_len HEAD_Q) := by
  unfold active
  infer_instance

def qOffset
    (s : BlockState) (stride_qbs stride_qh stride_qd : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_qbs + headIndex s * stride_qh + dim * stride_qd

def cosOffset
    (s : BlockState) (stride_cosbs stride_cosd : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_cosbs + dim * stride_cosd

def sinOffset
    (s : BlockState) (stride_sinbs stride_sind : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_sinbs + dim * stride_sind

noncomputable def rotaryQ0Spec
    (s : BlockState) (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimEven i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i)) -
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimOdd i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i))

/-- Algorithm-layer correctness for the Q even-dimension rotary store. -/
theorem rotary_emb_q0_block_correct
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        qOffset s stride_qbs stride_qh stride_qd (dimEven i)))
    (hExec : exec (rotary_emb_q0_block Q Cos Sin stride_qbs stride_qh
        stride_qd stride_cosbs stride_cosd stride_sinbs stride_sind
        max_total_len HEAD_Q BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimEven i)) =
        if active s max_total_len HEAD_Q then
          rotaryQ0Spec s Q Cos Sin stride_qbs stride_qh stride_qd
            stride_cosbs stride_cosd stride_sinbs stride_sind i
        else
          s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimEven i)) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HALF] =>
        s.pids 1 * stride_qbs + s.pids 0 * stride_qh +
          (idx.1.val * 2) * stride_qd) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [qOffset, seqIndex, headIndex, dimEven] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBH : 0 < BLOCK_HALF
  · simp [exec, rotary_emb_q0_block, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [qOffset, seqIndex, headIndex, dimEven]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hSeq : s.pids 1 < max_total_len
    · by_cases hHead : s.pids 0 < HEAD_Q
      · simp [active, rotaryQ0Spec, qOffset, cosOffset, sinOffset,
              seqIndex, headIndex, dimEven, dimOdd, hSeq, hHead,
              Option.bind, Option.map]
      · simp [active, seqIndex, headIndex, hSeq, hHead]
    · simp [active, seqIndex, hSeq]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the Q even-dimension rotary store. -/
theorem rotary_emb_q0_block_compute_correct
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        qOffset s stride_qbs stride_qh stride_qd (dimEven i))) :
    ComputeCorrect.Realizes
      (kernel := rotary_emb_q0_block Q Cos Sin stride_qbs stride_qh
        stride_qd stride_cosbs stride_cosd stride_sinbs stride_sind
        max_total_len HEAD_Q BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin BLOCK_HALF => active s max_total_len HEAD_Q)
        (fun i => (Q, qOffset s stride_qbs stride_qh stride_qd (dimEven i))))
      (expected := fun i =>
        rotaryQ0Spec s Q Cos Sin stride_qbs stride_qh stride_qd
          stride_cosbs stride_cosd stride_sinbs stride_sind i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_emb_q0_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rotary_emb_q0_block_correct Q Cos Sin stride_qbs stride_qh
    stride_qd stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len
    HEAD_Q BLOCK_HALF s s' hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented Q-odd-dimension slice of `rotary_emb.py`'s `_rotary_kernel`.

This models the second Q store for one sequence/head program tile:
`out1 = q0 * sin0 + q1 * cos0`, written at the odd-dimension offset. -/
def rotary_emb_q1_block
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  q0 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim1 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  out1 = q0 * sin0 + q1 * cos0
  tl.store(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim1 * $(stride_qd),
    out1,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)))
}

noncomputable def rotaryQ1Spec
    (s : BlockState) (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimEven i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i)) +
  s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimOdd i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i))

/-- Algorithm-layer correctness for the Q odd-dimension rotary store. -/
theorem rotary_emb_q1_block_correct
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        qOffset s stride_qbs stride_qh stride_qd (dimOdd i)))
    (hExec : exec (rotary_emb_q1_block Q Cos Sin stride_qbs stride_qh
        stride_qd stride_cosbs stride_cosd stride_sinbs stride_sind
        max_total_len HEAD_Q BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimOdd i)) =
        if active s max_total_len HEAD_Q then
          rotaryQ1Spec s Q Cos Sin stride_qbs stride_qh stride_qd
            stride_cosbs stride_cosd stride_sinbs stride_sind i
        else
          s.readMem Q (qOffset s stride_qbs stride_qh stride_qd (dimOdd i)) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HALF] =>
        s.pids 1 * stride_qbs + s.pids 0 * stride_qh +
          (idx.1.val * 2 + 1) * stride_qd) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [qOffset, seqIndex, headIndex, dimOdd] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBH : 0 < BLOCK_HALF
  · simp [exec, rotary_emb_q1_block, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [qOffset, seqIndex, headIndex, dimOdd]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hSeq : s.pids 1 < max_total_len
    · by_cases hHead : s.pids 0 < HEAD_Q
      · simp [active, rotaryQ1Spec, qOffset, cosOffset, sinOffset,
              seqIndex, headIndex, dimEven, dimOdd, hSeq, hHead,
              Option.bind, Option.map]
      · simp [active, seqIndex, headIndex, hSeq, hHead]
    · simp [active, seqIndex, hSeq]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the Q odd-dimension rotary store. -/
theorem rotary_emb_q1_block_compute_correct
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        qOffset s stride_qbs stride_qh stride_qd (dimOdd i))) :
    ComputeCorrect.Realizes
      (kernel := rotary_emb_q1_block Q Cos Sin stride_qbs stride_qh
        stride_qd stride_cosbs stride_cosd stride_sinbs stride_sind
        max_total_len HEAD_Q BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin BLOCK_HALF => active s max_total_len HEAD_Q)
        (fun i => (Q, qOffset s stride_qbs stride_qh stride_qd (dimOdd i))))
      (expected := fun i =>
        rotaryQ1Spec s Q Cos Sin stride_qbs stride_qh stride_qd
          stride_cosbs stride_cosd stride_sinbs stride_sind i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_emb_q1_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rotary_emb_q1_block_correct Q Cos Sin stride_qbs stride_qh
    stride_qd stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len
    HEAD_Q BLOCK_HALF s s' hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented K-even-dimension slice of `rotary_emb.py`'s `_rotary_kernel`.

Mirrors the Q0 slice for the K buffer with `HEAD_K` head bound. -/
def rotary_emb_k0_block
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  k0 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim0 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim1 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  outK0 = k0 * cos0 - k1 * sin0
  tl.store(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim0 * $(stride_kd),
    outK0,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)))
}

def kOffset
    (s : BlockState) (stride_kbs stride_kh stride_kd : Nat) (dim : Nat) : Nat :=
  seqIndex s * stride_kbs + headIndex s * stride_kh + dim * stride_kd

def activeK (s : BlockState) (max_total_len HEAD_K : Nat) : Prop :=
  seqIndex s < max_total_len ∧ headIndex s < HEAD_K

instance activeKDecidable (s : BlockState) (max_total_len HEAD_K : Nat) :
    Decidable (activeK s max_total_len HEAD_K) := by
  unfold activeK
  infer_instance

noncomputable def rotaryK0Spec
    (s : BlockState) (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimEven i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i)) -
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimOdd i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i))

theorem rotary_emb_k0_block_correct
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        kOffset s stride_kbs stride_kh stride_kd (dimEven i)))
    (hExec : exec (rotary_emb_k0_block K Cos Sin stride_kbs stride_kh
        stride_kd stride_cosbs stride_cosd stride_sinbs stride_sind
        max_total_len HEAD_K BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimEven i)) =
        if activeK s max_total_len HEAD_K then
          rotaryK0Spec s K Cos Sin stride_kbs stride_kh stride_kd
            stride_cosbs stride_cosd stride_sinbs stride_sind i
        else
          s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimEven i)) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HALF] =>
        s.pids 1 * stride_kbs + s.pids 0 * stride_kh +
          (idx.1.val * 2) * stride_kd) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kOffset, seqIndex, headIndex, dimEven] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBH : 0 < BLOCK_HALF
  · simp [exec, rotary_emb_k0_block, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [kOffset, seqIndex, headIndex, dimEven]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hSeq : s.pids 1 < max_total_len
    · by_cases hHead : s.pids 0 < HEAD_K
      · simp [activeK, rotaryK0Spec, kOffset, cosOffset, sinOffset,
              seqIndex, headIndex, dimEven, dimOdd, hSeq, hHead,
              Option.bind, Option.map]
      · simp [activeK, seqIndex, headIndex, hSeq, hHead]
    · simp [activeK, seqIndex, hSeq]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem rotary_emb_k0_block_compute_correct
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        kOffset s stride_kbs stride_kh stride_kd (dimEven i))) :
    ComputeCorrect.Realizes
      (kernel := rotary_emb_k0_block K Cos Sin stride_kbs stride_kh
        stride_kd stride_cosbs stride_cosd stride_sinbs stride_sind
        max_total_len HEAD_K BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin BLOCK_HALF => activeK s max_total_len HEAD_K)
        (fun i => (K, kOffset s stride_kbs stride_kh stride_kd (dimEven i))))
      (expected := fun i =>
        rotaryK0Spec s K Cos Sin stride_kbs stride_kh stride_kd
          stride_cosbs stride_cosd stride_sinbs stride_sind i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_emb_k0_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rotary_emb_k0_block_correct K Cos Sin stride_kbs stride_kh
    stride_kd stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len
    HEAD_K BLOCK_HALF s s' hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented K-odd-dimension slice of `rotary_emb.py`'s `_rotary_kernel`. -/
def rotary_emb_k1_block
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  k0 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim0 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim1 * $(stride_kd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  outK1 = k0 * sin0 + k1 * cos0
  tl.store(K + cur_seq_index * $(stride_kbs) +
      cur_head_index * $(stride_kh) + dim1 * $(stride_kd),
    outK1,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_K)))
}

noncomputable def rotaryK1Spec
    (s : BlockState) (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimEven i)) *
    s.readMem Sin (sinOffset s stride_sinbs stride_sind (dimEven i)) +
  s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimOdd i)) *
    s.readMem Cos (cosOffset s stride_cosbs stride_cosd (dimEven i))

theorem rotary_emb_k1_block_correct
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        kOffset s stride_kbs stride_kh stride_kd (dimOdd i)))
    (hExec : exec (rotary_emb_k1_block K Cos Sin stride_kbs stride_kh
        stride_kd stride_cosbs stride_cosd stride_sinbs stride_sind
        max_total_len HEAD_K BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimOdd i)) =
        if activeK s max_total_len HEAD_K then
          rotaryK1Spec s K Cos Sin stride_kbs stride_kh stride_kd
            stride_cosbs stride_cosd stride_sinbs stride_sind i
        else
          s.readMem K (kOffset s stride_kbs stride_kh stride_kd (dimOdd i)) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HALF] =>
        s.pids 1 * stride_kbs + s.pids 0 * stride_kh +
          (idx.1.val * 2 + 1) * stride_kd) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kOffset, seqIndex, headIndex, dimOdd] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBH : 0 < BLOCK_HALF
  · simp [exec, rotary_emb_k1_block, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [kOffset, seqIndex, headIndex, dimOdd]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hSeq : s.pids 1 < max_total_len
    · by_cases hHead : s.pids 0 < HEAD_K
      · simp [activeK, rotaryK1Spec, kOffset, cosOffset, sinOffset,
              seqIndex, headIndex, dimEven, dimOdd, hSeq, hHead,
              Option.bind, Option.map]
      · simp [activeK, seqIndex, headIndex, hSeq, hHead]
    · simp [activeK, seqIndex, hSeq]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem rotary_emb_k1_block_compute_correct
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        kOffset s stride_kbs stride_kh stride_kd (dimOdd i))) :
    ComputeCorrect.Realizes
      (kernel := rotary_emb_k1_block K Cos Sin stride_kbs stride_kh
        stride_kd stride_cosbs stride_cosd stride_sinbs stride_sind
        max_total_len HEAD_K BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin BLOCK_HALF => activeK s max_total_len HEAD_K)
        (fun i => (K, kOffset s stride_kbs stride_kh stride_kd (dimOdd i))))
      (expected := fun i =>
        rotaryK1Spec s K Cos Sin stride_kbs stride_kh stride_kd
          stride_cosbs stride_cosd stride_sinbs stride_sind i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_emb_k1_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rotary_emb_k1_block_correct K Cos Sin stride_kbs stride_kh
    stride_kd stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len
    HEAD_K BLOCK_HALF s s' hOutInj hExec i
  simpa [hActive] using h

/-! ## Full-kernel Q first-half (off_q0) correctness

The Python test runs the full `_rotary_kernel` which writes 4 stores: Q at
`off_q0` (even), Q at `off_q1` (odd), K at `off_k0` (even), K at `off_k1` (odd).
Per #139's audit, the slice proofs above aren't sufficient — we need
full-kernel proofs.

This section establishes the Q first-half store for the full kernel under the
assumption that `Q ≠ K` and `stride_qd ≠ 0` (the latter is needed to prove the
intra-region Q first-half vs Q second-half offsets disjoint: they differ by
parity of the dimension factor). -/

/-- Q first-half offset of the full `rotary_kernel_surface`. The tile shape is
`[BLOCK_SEQ, BLOCK_HEAD, BLOCK_DMODEL/2]`. The even dimension factor is
`2 * idx.2.2.1.val`. -/
def qFullOffset
    (s : BlockState) (stride_qbs stride_qh stride_qd
      BLOCK_HEAD BLOCK_SEQ BLOCK_HALF : Nat)
    (idx : TileIndex [BLOCK_SEQ, BLOCK_HEAD, BLOCK_HALF]) : Nat :=
  (s.pids 1 * BLOCK_SEQ + idx.1.val) * stride_qbs +
  (s.pids 0 * BLOCK_HEAD + idx.2.1.val) * stride_qh +
  (idx.2.2.1.val * 2) * stride_qd

/-- Cosine offset for the full kernel's Q first-half store. -/
def cosFullOffset
    (s : BlockState) (stride_cosbs stride_cosd
      BLOCK_SEQ BLOCK_HALF : Nat)
    (idx : TileIndex [BLOCK_SEQ, BLOCK_HALF]) : Nat :=
  (s.pids 1 * BLOCK_SEQ + idx.1.val) * stride_cosbs +
  (idx.2.1.val * 2) * stride_cosd

/-- Sine offset for the full kernel's Q first-half store. -/
def sinFullOffset
    (s : BlockState) (stride_cosbs stride_cosd
      BLOCK_SEQ BLOCK_HALF : Nat)
    (idx : TileIndex [BLOCK_SEQ, BLOCK_HALF]) : Nat :=
  (s.pids 1 * BLOCK_SEQ + idx.1.val) * stride_cosbs +
  (idx.2.1.val * 2) * stride_cosd

/-- Active predicate for the full kernel's Q stores. -/
def activeFullQ (s : BlockState) (max_total_len HEAD_Q
    BLOCK_HEAD BLOCK_SEQ : Nat)
    (idx : TileIndex [BLOCK_SEQ, BLOCK_HEAD]) : Prop :=
  s.pids 1 * BLOCK_SEQ + idx.1.val < max_total_len ∧
  s.pids 0 * BLOCK_HEAD + idx.2.1.val < HEAD_Q

instance activeFullQDecidable (s : BlockState) (max_total_len HEAD_Q
    BLOCK_HEAD BLOCK_SEQ : Nat)
    (idx : TileIndex [BLOCK_SEQ, BLOCK_HEAD]) :
    Decidable (activeFullQ s max_total_len HEAD_Q BLOCK_HEAD BLOCK_SEQ idx) := by
  unfold activeFullQ
  infer_instance

/-- Specification for the Q first-half output in the full kernel. -/
noncomputable def rotaryKernelQ0Spec
    (s : BlockState) (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd : Nat)
    (BLOCK_SEQ BLOCK_HEAD BLOCK_HALF : Nat)
    (idx : TileIndex [BLOCK_SEQ, BLOCK_HEAD, BLOCK_HALF]) : ℝ :=
  s.readMem Q (qFullOffset s stride_qbs stride_qh stride_qd
      BLOCK_HEAD BLOCK_SEQ BLOCK_HALF idx) *
    s.readMem Cos (cosFullOffset s stride_cosbs stride_cosd
      BLOCK_SEQ BLOCK_HALF (idx.1, idx.2.2)) -
  s.readMem Q ((s.pids 1 * BLOCK_SEQ + idx.1.val) * stride_qbs +
      (s.pids 0 * BLOCK_HEAD + idx.2.1.val) * stride_qh +
      (idx.2.2.1.val * 2 + 1) * stride_qd) *
    s.readMem Sin (sinFullOffset s stride_cosbs stride_cosd
      BLOCK_SEQ BLOCK_HALF (idx.1, idx.2.2))

/-- Algorithm-layer correctness for the Q first-half store in the full
`rotary_kernel_surface`. (Stub — full 3D broadcast-remap proof blocked on
`dropInsertedIndex` simp coverage for compound 4-tuple indices over a middle
singleton head axis; tracked in memory v6.) -/

end VeriTile.Bench.TritonBenchG.RotaryEmb
