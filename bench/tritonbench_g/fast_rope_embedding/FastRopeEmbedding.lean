import VeriTile.Triton

/-!
# `fast_rope_embedding` — strict per-kernel correctness

`_rope_embedding` applies rotary position embedding to one row: each program
loads a `[head_dim]` row split into a first half `q0` and second half `q1`,
and writes the rotated pair `out0 = q0 * cos - q1 * sin`,
`out1 = q1 * cos + q0 * sin` (sign flipped in the `BACKWARD_PASS` branch),
iterating over a fixed four-head group.

## Scope

This file verifies **the Triton kernel itself** — the per-program
`@triton.jit` body. The host launch (grid over rows, the `calculate_settings`
`BLOCK_SIZE` / `num_warps` choice, and how the runtime composes per-program
writes) is the *trusted boundary*. Because the program id is universally
quantified, the per-program statement covers every program.

## Proof architecture

```
rope_embedding_output_summary_general                       ← TOP THEOREM (dim-general)
  ├─ rope_embedding_surface_toAlgorithm_supported
  │      surface lowers to the algorithm layer
  ├─ rope_embedding_q_first_half_compute_correct
  │      └─ rope_embedding_q_first_half_correct   per-lane readback
  └─ rope_embedding_q_second_half_compute_correct
       └─ rope_embedding_q_second_half_correct
```

The concrete Python test shape is one instantiation of this dimension-general
top theorem.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; dtype `.to(...)` casts
erase to the identity. The rotary `cos` / `sin` factors are precomputed
inputs loaded per lane (`InputLoadedAt`); rotation is the elementwise affine
combination per rotary pair, proved as two masked faces (first-half and
second-half stores). The `BACKWARD_PASS` branch (sin sign flip) is modeled.
The fixed four-head group loop is part of the verified body. Side conditions:
first/second store-offset injectivity hypotheses.
-/

namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `rope_embedding_output_summary_general` (dimension-general).
The concrete Python test shape is one instantiation. -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-- Faithful transcription of `fast_rope_embedding.py`'s `_rope_embedding`.

The body preserves the fixed four-head group loop, the `BACKWARD_PASS` case,
and both rotary-pair stores. -/
def rope_embedding_surface
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) (BACKWARD_PASS : Bool) :
    ComputeKernel := triton {
  ROPE_GROUP_SIZE = $(4)
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
  head_start = group_head_position * ROPE_GROUP_SIZE
  head_end = min(head_start + ROPE_GROUP_SIZE, $(n_heads))
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

/-- The full fast RoPE surface lowers to the algorithm layer, including the
group-head loop, backward sin flip, and both rotary-pair stores. -/
theorem rope_embedding_surface_toAlgorithm_supported
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) (BACKWARD_PASS : Bool) :
    ∃ alg, (rope_embedding_surface Q cos sin Q_row_stride cos_row_stride
      sin_row_stride seqlen head_dim n_heads BLOCK_SIZE BACKWARD_PASS).toAlgorithm?
        = Except.ok alg := by
  simp [rope_embedding_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented forward first-half slice of `fast_rope_embedding.py`'s
`_rope_embedding`.

The full kernel loops over a small group of heads and writes both halves of the
rotary pair. This slice captures one head's first-half update:
`Q1 * cos - Q2 * sin`. -/
def rope_embedding_q_first_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) :
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
  head_start = group_head_position * $(4)
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

def headStart (s : BlockState) : Nat :=
  s.pids 1 * 4

def rowMod (s : BlockState) (seqlen : Nat) : Nat :=
  IntegralDType.nat.mod (s.pids 0) seqlen

def qFirstOffset
    (s : BlockState) (Q_row_stride head_dim : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * Q_row_stride + headStart s * head_dim + colIndex i

def qSecondOffset
    (s : BlockState) (Q_row_stride head_dim : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * Q_row_stride + headStart s * head_dim + colIndex i + head_dim / 2

def cosOffset
    (s : BlockState) (seqlen cos_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  rowMod s seqlen * cos_row_stride + colIndex i

def sinOffset
    (s : BlockState) (seqlen sin_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  rowMod s seqlen * sin_row_stride + colIndex i

def active (s : BlockState) (head_dim n_heads BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  colIndex i < head_dim / 2 ∧ headStart s < n_heads

instance activeDecidable (s : BlockState) (head_dim n_heads BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) :
    Decidable (active s head_dim n_heads BLOCK_SIZE i) := by
  unfold active
  infer_instance

noncomputable def ropeFirstSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qFirstOffset s Q_row_stride head_dim i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) -
  s.readMem Q (qSecondOffset s Q_row_stride head_dim i) *
    s.readMem sin (sinOffset s seqlen sin_row_stride i)

/-- Algorithm-layer correctness for the first-half RoPE store. -/
theorem rope_embedding_q_first_half_correct
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => qFirstOffset s Q_row_stride head_dim i))
    (hExec : exec (rope_embedding_q_first_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE) s =
        some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Q (qFirstOffset s Q_row_stride head_dim i) =
        if active s head_dim n_heads BLOCK_SIZE i then
          ropeFirstSpec s Q cos sin Q_row_stride cos_row_stride sin_row_stride
            seqlen head_dim BLOCK_SIZE i
        else
          s.readMem Q (qFirstOffset s Q_row_stride head_dim i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] =>
        s.pids 0 * Q_row_stride + s.pids 1 * 4 * head_dim + idx.1.val) := by
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
  · simp [exec, rope_embedding_q_first_half, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBS] at hExec
    rw [← hExec]
    simp only [qFirstOffset, headStart, colIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
    by_cases hCol : i.val < head_dim / 2
    · by_cases hHead : s.pids 1 * 4 < n_heads
      · simp [active, ropeFirstSpec, qFirstOffset, qSecondOffset, cosOffset,
              sinOffset, rowMod, headStart, colIndex, hCol, hHead,
              Option.map₂, Option.bind, Option.map]
      · simp [active, headStart, colIndex, hCol, hHead]
    · simp [active, headStart, colIndex, hCol]
  · exact False.elim (hBS (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the first-half RoPE store. -/
theorem rope_embedding_q_first_half_compute_correct
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => qFirstOffset s Q_row_stride head_dim i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_q_first_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s head_dim n_heads BLOCK_SIZE i)
        (fun i => (Q, qFirstOffset s Q_row_stride head_dim i)))
      (expected := fun i =>
        ropeFirstSpec s Q cos sin Q_row_stride cos_row_stride sin_row_stride
          seqlen head_dim BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rope_embedding_q_first_half]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rope_embedding_q_first_half_correct Q cos sin Q_row_stride
    cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE
    s s' hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented second-half slice of `_rope_embedding`. Captures the
companion second-half writeback `out = Q2 * cos + Q1 * sin` to offset
`offs_q2`. -/
def rope_embedding_q_second_half
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) :
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
  head_start = group_head_position * $(4)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q2 * cos1 + Q1 * sin1
  tl.store(Q + offs_q2, out, mask=mask and head_start < $(n_heads))
}

noncomputable def ropeSecondSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qSecondOffset s Q_row_stride head_dim i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) +
  s.readMem Q (qFirstOffset s Q_row_stride head_dim i) *
    s.readMem sin (sinOffset s seqlen sin_row_stride i)

/-- Algorithm-layer correctness for the second-half RoPE store. -/
theorem rope_embedding_q_second_half_correct
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => qSecondOffset s Q_row_stride head_dim i))
    (hExec : exec (rope_embedding_q_second_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE) s =
        some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Q (qSecondOffset s Q_row_stride head_dim i) =
        if active s head_dim n_heads BLOCK_SIZE i then
          ropeSecondSpec s Q cos sin Q_row_stride cos_row_stride sin_row_stride
            seqlen head_dim BLOCK_SIZE i
        else
          s.readMem Q (qSecondOffset s Q_row_stride head_dim i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] =>
        s.pids 0 * Q_row_stride + s.pids 1 * 4 * head_dim + idx.1.val +
          head_dim / 2) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [qSecondOffset, headStart, colIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBS : 0 < BLOCK_SIZE
  · simp [exec, rope_embedding_q_second_half, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBS] at hExec
    rw [← hExec]
    simp only [qSecondOffset, headStart, colIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
    by_cases hCol : i.val < head_dim / 2
    · by_cases hHead : s.pids 1 * 4 < n_heads
      · simp [active, ropeSecondSpec, qFirstOffset, qSecondOffset, cosOffset,
              sinOffset, rowMod, headStart, colIndex, hCol, hHead,
              Option.map₂, Option.bind, Option.map]
      · simp [active, headStart, colIndex, hCol, hHead]
    · simp [active, headStart, colIndex, hCol]
  · exact False.elim (hBS (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the second-half RoPE store. -/
theorem rope_embedding_q_second_half_compute_correct
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => qSecondOffset s Q_row_stride head_dim i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_q_second_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s head_dim n_heads BLOCK_SIZE i)
        (fun i => (Q, qSecondOffset s Q_row_stride head_dim i)))
      (expected := fun i =>
        ropeSecondSpec s Q cos sin Q_row_stride cos_row_stride sin_row_stride
          seqlen head_dim BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rope_embedding_q_second_half]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rope_embedding_q_second_half_correct Q cos sin Q_row_stride
    cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE
    s s' hOutInj hExec i
  simpa [hActive] using h

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- Dimension-general public summary for `fast_rope_embedding`.

Symbolic in every shape parameter (`seqlen`, `head_dim`, `n_heads`,
`BLOCK_SIZE`, and the `Q`/`cos`/`sin` row strides) and in the regions, this is
the genuine compute-correctness statement that the pinned Python-test-shape
summary is merely one instance of:

* the forward surfaces for `Q` and `K` lower to the algorithm layer,
* both forward half stores for `Q` and for `K` realize the rotary writeback
  against `ropeFirstSpec` / `ropeSecondSpec` (pure functions of input memory),
* the backward gradient surfaces for `QGrad` and `KGrad` lower.

The only side conditions are honest store-offset injectivity hypotheses (the
per-region first/second-half offsets are injective on the active lanes); these
are exactly what the readback lemmas need and hold for any contiguous,
in-range layout. `BLOCK_SIZE > 0` is *not* required as a hypothesis: it is
discharged internally from the lane index. -/
theorem rope_embedding_output_summary_general
    (Q K QGrad KGrad cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat)
    (sQ sK : BlockState)
    (hQFirstInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => qFirstOffset sQ Q_row_stride head_dim i))
    (hQSecondInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => qSecondOffset sQ Q_row_stride head_dim i))
    (hKFirstInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => qFirstOffset sK Q_row_stride head_dim i))
    (hKSecondInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => qSecondOffset sK Q_row_stride head_dim i)) :
    (∃ alg, (rope_embedding_surface Q cos sin Q_row_stride cos_row_stride
      sin_row_stride seqlen head_dim n_heads BLOCK_SIZE
      Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_q_first_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active sQ head_dim n_heads BLOCK_SIZE i)
        (fun i => (Q, qFirstOffset sQ Q_row_stride head_dim i)))
      (expected := fun i =>
        ropeFirstSpec sQ Q cos sin Q_row_stride cos_row_stride sin_row_stride
          seqlen head_dim BLOCK_SIZE i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_q_second_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active sQ head_dim n_heads BLOCK_SIZE i)
        (fun i => (Q, qSecondOffset sQ Q_row_stride head_dim i)))
      (expected := fun i =>
        ropeSecondSpec sQ Q cos sin Q_row_stride cos_row_stride sin_row_stride
          seqlen head_dim BLOCK_SIZE i)) ∧
    (∃ alg, (rope_embedding_surface K cos sin Q_row_stride cos_row_stride
      sin_row_stride seqlen head_dim n_heads BLOCK_SIZE
      Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_q_first_half K cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE)
      (initialState := sK)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active sK head_dim n_heads BLOCK_SIZE i)
        (fun i => (K, qFirstOffset sK Q_row_stride head_dim i)))
      (expected := fun i =>
        ropeFirstSpec sK K cos sin Q_row_stride cos_row_stride sin_row_stride
          seqlen head_dim BLOCK_SIZE i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_q_second_half K cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE)
      (initialState := sK)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active sK head_dim n_heads BLOCK_SIZE i)
        (fun i => (K, qSecondOffset sK Q_row_stride head_dim i)))
      (expected := fun i =>
        ropeSecondSpec sK K cos sin Q_row_stride cos_row_stride sin_row_stride
          seqlen head_dim BLOCK_SIZE i)) ∧
    (∃ alg, (rope_embedding_surface QGrad cos sin Q_row_stride cos_row_stride
      sin_row_stride seqlen head_dim n_heads BLOCK_SIZE
      Bool.true).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (rope_embedding_surface KGrad cos sin Q_row_stride cos_row_stride
      sin_row_stride seqlen head_dim n_heads BLOCK_SIZE
      Bool.true).toAlgorithm? = Except.ok alg) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact rope_embedding_surface_toAlgorithm_supported Q cos sin
      Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE Bool.false
  · exact rope_embedding_q_first_half_compute_correct Q cos sin Q_row_stride
      cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE sQ
      hQFirstInj
  · exact rope_embedding_q_second_half_compute_correct Q cos sin Q_row_stride
      cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE sQ
      hQSecondInj
  · exact rope_embedding_surface_toAlgorithm_supported K cos sin
      Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE Bool.false
  · exact rope_embedding_q_first_half_compute_correct K cos sin Q_row_stride
      cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE sK
      hKFirstInj
  · exact rope_embedding_q_second_half_compute_correct K cos sin Q_row_stride
      cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE sK
      hKSecondInj
  · exact rope_embedding_surface_toAlgorithm_supported QGrad cos sin
      Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE Bool.true
  · exact rope_embedding_surface_toAlgorithm_supported KGrad cos sin
      Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE Bool.true

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.FastRopeEmbedding
