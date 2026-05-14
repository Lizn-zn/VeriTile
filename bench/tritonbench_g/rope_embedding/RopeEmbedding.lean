import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.RopeEmbedding

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `rope_embedding.py`'s `_rope_embedding`.

The body preserves the group-head loop, the `BACKWARD_PASS` branch, and both
rotary-pair stores. -/
def rope_embedding_surface
    (Q : RegionName) (Q_row_stride : Nat)
    (cos : RegionName) (cos_row_stride : Nat)
    (sin : RegionName) (sin_row_stride seqlen head_dim n_heads : Nat)
    (BACKWARD_PASS : Bool) (BLOCK_SIZE ROPE_GROUP_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) +
    half_head_dim * $(0) + col_offsets, mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) +
    half_head_dim * $(0) + col_offsets, mask=mask, other=0)
  if BACKWARD_PASS {
    sin1 = -sin1
  }
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  head_end = min(head_start + $(ROPE_GROUP_SIZE), $(n_heads))
  for k in range(head_start, head_end, $(1)) {
    offs_q1 = row_position * $(Q_row_stride) + k * $(head_dim) + col_offsets
    offs_q2 = row_position * $(Q_row_stride) + k * $(head_dim) +
      col_offsets + half_head_dim
    Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
    Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
    tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask=mask)
    tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask=mask)
  }
}

/-- Proof-oriented forward first-half slice of `rope_embedding.py`'s
`_rope_embedding`.

The full kernel loops over `ROPE_GROUP_SIZE` heads and writes both halves of
the rotary pair. This slice captures one group's first head and first-half
store: `Q1 * cos - Q2 * sin`. -/
def rope_embedding_forward_first_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < $(head_dim / 2)
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + col_offsets,
    mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + col_offsets,
    mask=mask, other=0)
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q1 * cos1 - Q2 * sin1
  tl.store(Q + offs_q1, out, mask=mask and head_start < $(n_heads))
}

def colIndex (i : Fin BLOCK_SIZE) : Nat :=
  i.val

def headStart (s : BlockState) (ROPE_GROUP_SIZE : Nat) : Nat :=
  s.pids 1 * ROPE_GROUP_SIZE

def rowMod (s : BlockState) (seqlen : Nat) : Nat :=
  IntegralDType.nat.mod (s.pids 0) seqlen

def qFirstOffset
    (s : BlockState) (Q_row_stride head_dim ROPE_GROUP_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * Q_row_stride + headStart s ROPE_GROUP_SIZE * head_dim + colIndex i

def qSecondOffset
    (s : BlockState) (Q_row_stride head_dim ROPE_GROUP_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * Q_row_stride + headStart s ROPE_GROUP_SIZE * head_dim +
    colIndex i + head_dim / 2

def cosOffset
    (s : BlockState) (seqlen cos_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  rowMod s seqlen * cos_row_stride + colIndex i

def sinOffset
    (s : BlockState) (seqlen sin_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  rowMod s seqlen * sin_row_stride + colIndex i

def active (s : BlockState) (head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  colIndex i < head_dim / 2 ∧ headStart s ROPE_GROUP_SIZE < n_heads

instance activeDecidable
    (s : BlockState) (head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) :
    Decidable (active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i) := by
  unfold active
  infer_instance

noncomputable def ropeFirstSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) -
  s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem sin (sinOffset s seqlen sin_row_stride i)

/-- Algorithm-layer correctness for the first-half RoPE store. -/
theorem rope_embedding_forward_first_half_correct
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i))
    (hExec : exec (rope_embedding_forward_first_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE
        BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) =
        if active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i then
          ropeFirstSpec s Q cos sin Q_row_stride cos_row_stride sin_row_stride
            seqlen head_dim ROPE_GROUP_SIZE BLOCK_SIZE i
        else
          s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] =>
        s.pids 0 * Q_row_stride + s.pids 1 * ROPE_GROUP_SIZE * head_dim +
          idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [qFirstOffset, headStart, colIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBS : 0 < BLOCK_SIZE
  · simp [exec, rope_embedding_forward_first_half, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBS] at hExec
    rw [← hExec]
    simp only [qFirstOffset, headStart, colIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
    by_cases hCol : i.val < head_dim / 2
    · by_cases hHead : s.pids 1 * ROPE_GROUP_SIZE < n_heads
      · simp [active, ropeFirstSpec, qFirstOffset, qSecondOffset, cosOffset,
              sinOffset, rowMod, headStart, colIndex, hCol, hHead,
              Option.map₂, Option.bind, Option.map]
      · simp [active, headStart, colIndex, hCol, hHead]
    · simp [active, headStart, colIndex, hCol]
  · exact False.elim (hBS (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the first-half RoPE store. -/
theorem rope_embedding_forward_first_half_compute_correct
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i)) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_first_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE
        BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE =>
          active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i)
        (fun i => (Q, qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i)))
      (expected := fun i =>
        ropeFirstSpec s Q cos sin Q_row_stride cos_row_stride sin_row_stride
          seqlen head_dim ROPE_GROUP_SIZE BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rope_embedding_forward_first_half]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rope_embedding_forward_first_half_correct Q cos sin Q_row_stride
    cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE
    BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.RopeEmbedding
