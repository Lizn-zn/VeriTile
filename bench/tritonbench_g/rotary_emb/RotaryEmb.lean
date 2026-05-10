import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.RotaryEmb

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

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

end VeriTile.Bench.TritonBenchG.RotaryEmb
