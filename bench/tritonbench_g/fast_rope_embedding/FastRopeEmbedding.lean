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
fast_rope_embedding_io_correctness                          ← TOP THEOREM (`⊨`, both halves)
  ├─ rope_{first,second}_flattenOk                inside the flat-memory bridge
  ├─ rope_{first,second}_traceSafe                per-execution address safety
  └─ rope_{first,second}_region_run               region-model run
       ├─ rope_{first,second}_terminates
       ├─ rope_embedding_q_{first,second}_half_correct   per-lane readback (shared)
       ├─ rope{First,Second}Spec_eq_of            memory spec = value spec (lane-local)
       ├─ q{First,Second}_inj                     output injectivity, discharged not assumed
       └─ rope_{first,second}_frame               cell-level frame off the write window

rope_embedding_output_summary_general                       per-write-map summary (dim-general)
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
open scoped VeriTile.Triton.InPlaceMaskedTileKernelIO

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
specification rope_embedding_output_summary_general
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

/-! ## ════════ `⊨` IO face for the two RoPE half-updates ════════

The summaries above are stated per *declared write map*. This section restates
both half-slices on the audit-once IO surface
`InPlaceMaskedTileKernelIO.Implements` (`⊨`), which additionally pins the **flat
memory** placement.

RoPE is the shape that skin exists for: `Q` is read at **two** windows
(`offs_q1`, `offs_q2`), `cos` and `sin` are read-only auxiliaries, and the store
goes back into `Q` at one of the two windows. Every other family in
`KernelSpec.lean` lists its output buffer separately from its inputs, which
excludes `out = in` outright. `f`'s four arguments are the **pre-state** lane
values, which is the right reading for an in-place update: the two halves are
computed from the values `Q` held on entry. -/

section IOFace

/-- Cell-level frame of a masked scatter (private copy — `bench` files are
standalone). -/
private theorem foldl_writeMem_frame {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (R : RegionName) (off : Nat) :
    ∀ l : List α, (R ≠ region ∨ ∀ k ∈ l, P k → offsetFn k ≠ off) →
      ∀ s : BlockState,
        ((l.foldl (fun acc k =>
            if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
            s).mem R off) = s.mem R off := by
  intro l
  induction l with
  | nil => intro _ s; rfl
  | cons hd tl ih =>
      intro hc s
      have htl : R ≠ region ∨ ∀ k ∈ tl, P k → offsetFn k ≠ off := by
        rcases hc with h | h
        · exact Or.inl h
        · exact Or.inr fun k hk => h k (List.mem_cons_of_mem hd hk)
      rw [List.foldl_cons, ih htl]
      by_cases hP : P hd
      · rw [if_pos hP, BlockState.writeMem_mem, if_neg ?_]
        rintro ⟨h1, h2⟩
        rcases hc with h | h
        · exact h h1
        · exact h hd List.mem_cons_self hP h2.symm
      · rw [if_neg hP]

/-! ### First half -/

/-- Value-level first-half spec: `q1 · c - q2 · s`, over the *loaded values*. -/
noncomputable def ropeFirstSpecOf {BLOCK_SIZE : Nat}
    (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (i : TileIndex [BLOCK_SIZE]) : ℝ :=
  q1 i * c1 i - q2 i * s1 i

/-- Value-level second-half spec: `q2 · c + q1 · s`. -/
noncomputable def ropeSecondSpecOf {BLOCK_SIZE : Nat}
    (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (i : TileIndex [BLOCK_SIZE]) : ℝ :=
  q2 i * c1 i + q1 i * s1 i

/-- The memory-level and value-level first-half specs agree under the pins. -/
theorem ropeFirstSpec_eq_of (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim BLOCK_SIZE : Nat)
    (s : BlockState) (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (hq1 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem Q (qFirstOffset s Q_row_stride head_dim i.1) = q1 i)
    (hq2 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem Q (qSecondOffset s Q_row_stride head_dim i.1) = q2 i)
    (hc : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem cos (cosOffset s seqlen cos_row_stride i.1) = c1 i)
    (hs : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem sin (sinOffset s seqlen sin_row_stride i.1) = s1 i)
    (i : TileIndex [BLOCK_SIZE]) (hi : i.1.val < head_dim / 2) :
    ropeFirstSpec s Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen
        head_dim BLOCK_SIZE i.1
      = ropeFirstSpecOf q1 q2 c1 s1 i := by
  rw [ropeFirstSpec, ropeFirstSpecOf, hq1 i hi, hq2 i hi, hc i hi, hs i hi]

/-- The memory-level and value-level second-half specs agree under the pins. -/
theorem ropeSecondSpec_eq_of (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim BLOCK_SIZE : Nat)
    (s : BlockState) (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (hq1 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem Q (qFirstOffset s Q_row_stride head_dim i.1) = q1 i)
    (hq2 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem Q (qSecondOffset s Q_row_stride head_dim i.1) = q2 i)
    (hc : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem cos (cosOffset s seqlen cos_row_stride i.1) = c1 i)
    (hs : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem sin (sinOffset s seqlen sin_row_stride i.1) = s1 i)
    (i : TileIndex [BLOCK_SIZE]) (hi : i.1.val < head_dim / 2) :
    ropeSecondSpec s Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen
        head_dim BLOCK_SIZE i.1
      = ropeSecondSpecOf q1 q2 c1 s1 i := by
  rw [ropeSecondSpec, ropeSecondSpecOf, hq1 i hi, hq2 i hi, hc i hi, hs i hi]

/-- The first-half slice sits inside the flat-memory bridge's covered fragment. -/
theorem rope_first_flattenOk (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) :
    ((rope_embedding_q_first_half Q cos sin Q_row_stride cos_row_stride
      sin_row_stride seqlen head_dim n_heads BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [rope_embedding_q_first_half, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]

/-- The second-half slice sits inside the bridge's covered fragment. -/
theorem rope_second_flattenOk (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) :
    ((rope_embedding_q_second_half Q cos sin Q_row_stride cos_row_stride
      sin_row_stride seqlen head_dim n_heads BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [rope_embedding_q_second_half, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]

/-- Termination of the first-half slice. -/
theorem rope_first_terminates (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) (s : BlockState) :
    ∃ s1, exec (rope_embedding_q_first_half Q cos sin Q_row_stride
      cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE) s
      = some s1 := by
  simp [exec, rope_embedding_q_first_half, stepStmts, stepStmt, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]

/-- Termination of the second-half slice. -/
theorem rope_second_terminates (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) (s : BlockState) :
    ∃ s1, exec (rope_embedding_q_second_half Q cos sin Q_row_stride
      cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE) s
      = some s1 := by
  simp [exec, rope_embedding_q_second_half, stepStmts, stepStmt, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]

/-- Cell-level frame of the first-half slice. -/
theorem rope_first_frame (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) (s s' : BlockState)
    (hExec : exec (rope_embedding_q_first_half Q cos sin Q_row_stride
      cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE) s
      = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ Q ∨ ∀ i : Fin BLOCK_SIZE, active s head_dim n_heads BLOCK_SIZE i →
        o ≠ qFirstOffset s Q_row_stride head_dim i) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, rope_embedding_q_first_half, stepStmts, stepStmt, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    ComparableDType.lt] at hExec
  subst hExec
  rw [foldl_writeMem_frame (region := Q)
    (fun i : TileIndex [BLOCK_SIZE] =>
      s.pids 0 * Q_row_stride + s.pids 1 * 4 * head_dim + i.1.val)
    _ (fun i : TileIndex [BLOCK_SIZE] =>
      i.1.val < head_dim / 2 ∧ s.pids 1 * 4 < n_heads) r o
    (TileShape.allIndices [BLOCK_SIZE]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ hi => Ne.symm (h i.1 hi)

/-- Cell-level frame of the second-half slice. -/
theorem rope_second_frame (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) (s s' : BlockState)
    (hExec : exec (rope_embedding_q_second_half Q cos sin Q_row_stride
      cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE) s
      = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ Q ∨ ∀ i : Fin BLOCK_SIZE, active s head_dim n_heads BLOCK_SIZE i →
        o ≠ qSecondOffset s Q_row_stride head_dim i) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, rope_embedding_q_second_half, stepStmts, stepStmt, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    ComparableDType.lt] at hExec
  subst hExec
  rw [foldl_writeMem_frame (region := Q)
    (fun i : TileIndex [BLOCK_SIZE] =>
      s.pids 0 * Q_row_stride + s.pids 1 * 4 * head_dim + i.1.val
        + head_dim / 2)
    _ (fun i : TileIndex [BLOCK_SIZE] =>
      i.1.val < head_dim / 2 ∧ s.pids 1 * 4 < n_heads) r o
    (TileShape.allIndices [BLOCK_SIZE]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ hi => Ne.symm (h i.1 hi)

/-- The two `Q` window maps are injective outright — each is `base + lane` (the
second shifted by `head_dim / 2`) — so the closed-form readbacks' `hOutInj`
precondition is a *theorem* here, not a hypothesis the headline has to carry. -/
private theorem qFirst_inj (s : BlockState) (Q_row_stride head_dim BLOCK_SIZE : Nat) :
    Function.Injective
      (fun i : Fin BLOCK_SIZE => qFirstOffset s Q_row_stride head_dim i) := by
  intro a b h
  simp only [qFirstOffset, colIndex] at h
  exact Fin.ext (Nat.add_left_cancel h)

private theorem qSecond_inj (s : BlockState) (Q_row_stride head_dim BLOCK_SIZE : Nat) :
    Function.Injective
      (fun i : Fin BLOCK_SIZE => qSecondOffset s Q_row_stride head_dim i) := by
  intro a b h
  simp only [qSecondOffset, colIndex] at h
  exact Fin.ext (Nat.add_left_cancel (Nat.add_right_cancel h))

/-- Per-execution safety walk for the first-half slice. The four loads are gated
by the **column** guard only (`col < head_dim / 2`); the store additionally
carries `head_start < n_heads`. -/
theorem rope_first_traceSafe (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) (bounds : RegionBounds) (s : BlockState)
    (hq1 : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      qFirstOffset s Q_row_stride head_dim i < bounds Q)
    (hq2 : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      qSecondOffset s Q_row_stride head_dim i < bounds Q)
    (hc : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      cosOffset s seqlen cos_row_stride i < bounds cos)
    (hsn : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      sinOffset s seqlen sin_row_stride i < bounds sin)
    (hst : ∀ i : Fin BLOCK_SIZE, active s head_dim n_heads BLOCK_SIZE i →
      qFirstOffset s Q_row_stride head_dim i < bounds Q) :
    ((rope_embedding_q_first_half Q cos sin Q_row_stride cos_row_stride
      sin_row_stride seqlen head_dim n_heads BLOCK_SIZE).toAlgKernel).TraceSafe
      bounds s := by
  simp [Kernel.TraceSafe, rope_embedding_q_first_half, Stmt.TraceSafeList,
    Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active,
    MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
    memAccessMemorySafe, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop,
    Tile.cop, Tile.ptrAdd, Tile.uop, NumericDType.add, NumericDType.mul,
    NumericDType.sub, ComparableDType.lt]
  exact ⟨fun a ha => hsn a ha, fun a ha => hc a ha, fun a ha => hq1 a ha,
    fun a ha => hq2 a ha, fun a ha hh => hst a ⟨ha, hh⟩⟩

/-- Per-execution safety walk for the second-half slice. -/
theorem rope_second_traceSafe (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) (bounds : RegionBounds) (s : BlockState)
    (hq1 : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      qFirstOffset s Q_row_stride head_dim i < bounds Q)
    (hq2 : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      qSecondOffset s Q_row_stride head_dim i < bounds Q)
    (hc : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      cosOffset s seqlen cos_row_stride i < bounds cos)
    (hsn : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      sinOffset s seqlen sin_row_stride i < bounds sin)
    (hst : ∀ i : Fin BLOCK_SIZE, active s head_dim n_heads BLOCK_SIZE i →
      qSecondOffset s Q_row_stride head_dim i < bounds Q) :
    ((rope_embedding_q_second_half Q cos sin Q_row_stride cos_row_stride
      sin_row_stride seqlen head_dim n_heads BLOCK_SIZE).toAlgKernel).TraceSafe
      bounds s := by
  simp [Kernel.TraceSafe, rope_embedding_q_second_half, Stmt.TraceSafeList,
    Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active,
    MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
    memAccessMemorySafe, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop,
    Tile.cop, Tile.ptrAdd, Tile.uop, NumericDType.add, NumericDType.mul,
    NumericDType.sub, ComparableDType.lt]
  exact ⟨fun a ha => hsn a ha, fun a ha => hc a ha, fun a ha => hq1 a ha,
    fun a ha => hq2 a ha, fun a ha hh => hst a ⟨ha, hh⟩⟩

/-- Region-model run of the first-half slice. -/
theorem rope_first_region_run (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) (s₀ : BlockState)
    (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (hq1 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem Q (qFirstOffset s₀ Q_row_stride head_dim i.1) = q1 i)
    (hq2 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem Q (qSecondOffset s₀ Q_row_stride head_dim i.1) = q2 i)
    (hc : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem cos (cosOffset s₀ seqlen cos_row_stride i.1) = c1 i)
    (hs : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem sin (sinOffset s₀ seqlen sin_row_stride i.1) = s1 i) :
    ∃ s' , exec (rope_embedding_q_first_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE) s₀
        = some s'
      ∧ (∀ i : TileIndex [BLOCK_SIZE],
          active s₀ head_dim n_heads BLOCK_SIZE i.1 →
          s'.readMem Q (qFirstOffset s₀ Q_row_stride head_dim i.1)
            = ropeFirstSpecOf q1 q2 c1 s1 i)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ Q ∨ ∀ i : Fin BLOCK_SIZE,
            active s₀ head_dim n_heads BLOCK_SIZE i →
            o ≠ qFirstOffset s₀ Q_row_stride head_dim i) →
          s'.mem r o = s₀.mem r o) := by
  obtain ⟨s', hexec⟩ := rope_first_terminates Q cos sin Q_row_stride
    cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE s₀
  refine ⟨s', hexec, ?_, rope_first_frame Q cos sin Q_row_stride cos_row_stride
    sin_row_stride seqlen head_dim n_heads BLOCK_SIZE s₀ s' hexec⟩
  intro i hact
  rw [rope_embedding_q_first_half_correct Q cos sin Q_row_stride cos_row_stride
      sin_row_stride seqlen head_dim n_heads BLOCK_SIZE s₀ s'
      (qFirst_inj s₀ Q_row_stride head_dim BLOCK_SIZE) hexec i.1,
    if_pos hact,
    ropeFirstSpec_eq_of Q cos sin Q_row_stride cos_row_stride sin_row_stride
      seqlen head_dim BLOCK_SIZE s₀ q1 q2 c1 s1
      hq1 hq2 hc hs i hact.1]

/-- Region-model run of the second-half slice. -/
theorem rope_second_region_run (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) (s₀ : BlockState)
    (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (hq1 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem Q (qFirstOffset s₀ Q_row_stride head_dim i.1) = q1 i)
    (hq2 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem Q (qSecondOffset s₀ Q_row_stride head_dim i.1) = q2 i)
    (hc : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem cos (cosOffset s₀ seqlen cos_row_stride i.1) = c1 i)
    (hs : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem sin (sinOffset s₀ seqlen sin_row_stride i.1) = s1 i) :
    ∃ s' , exec (rope_embedding_q_second_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE) s₀
        = some s'
      ∧ (∀ i : TileIndex [BLOCK_SIZE],
          active s₀ head_dim n_heads BLOCK_SIZE i.1 →
          s'.readMem Q (qSecondOffset s₀ Q_row_stride head_dim i.1)
            = ropeSecondSpecOf q1 q2 c1 s1 i)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ Q ∨ ∀ i : Fin BLOCK_SIZE,
            active s₀ head_dim n_heads BLOCK_SIZE i →
            o ≠ qSecondOffset s₀ Q_row_stride head_dim i) →
          s'.mem r o = s₀.mem r o) := by
  obtain ⟨s', hexec⟩ := rope_second_terminates Q cos sin Q_row_stride
    cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE s₀
  refine ⟨s', hexec, ?_, rope_second_frame Q cos sin Q_row_stride cos_row_stride
    sin_row_stride seqlen head_dim n_heads BLOCK_SIZE s₀ s' hexec⟩
  intro i hact
  rw [rope_embedding_q_second_half_correct Q cos sin Q_row_stride cos_row_stride
      sin_row_stride seqlen head_dim n_heads BLOCK_SIZE s₀ s'
      (qSecond_inj s₀ Q_row_stride head_dim BLOCK_SIZE) hexec i.1,
    if_pos hact,
    ropeSecondSpec_eq_of Q cos sin Q_row_stride cos_row_stride sin_row_stride
      seqlen head_dim BLOCK_SIZE s₀ q1 q2 c1 s1
      hq1 hq2 hc hs i hact.1]

/-- IO signature of the first-half slice on the **in-place** tile surface: `Q` is
read at `offs_q1` and `offs_q2`, `cos` / `sin` are read-only, and the store goes
back to `Q` at `offs_q1`. Loads are gated by the column guard alone; the store
additionally carries `head_start < n_heads`. -/
def ropeFirstIO (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) : InPlaceMaskedTileKernelIO where
  kernel := rope_embedding_q_first_half Q cos sin Q_row_stride cos_row_stride
    sin_row_stride seqlen head_dim n_heads BLOCK_SIZE
  main := Q
  aux1 := cos
  aux2 := sin
  shape := [BLOCK_SIZE]
  readMain1 := fun p₀ p₁ i => p₀ * Q_row_stride + p₁ * 4 * head_dim + i.1.val
  readMain2 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * 4 * head_dim + i.1.val + head_dim / 2
  readAux1 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * cos_row_stride + i.1.val
  readAux2 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * sin_row_stride + i.1.val
  write := fun p₀ p₁ i => p₀ * Q_row_stride + p₁ * 4 * head_dim + i.1.val
  mask := fun _p₀ _p₁ i => i.1.val < head_dim / 2
  writeMask := fun _p₀ p₁ i => i.1.val < head_dim / 2 ∧ p₁ * 4 < n_heads

/-- IO signature of the second-half slice: same channels, the store now at
`offs_q2`. -/
def ropeSecondIO (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) : InPlaceMaskedTileKernelIO where
  kernel := rope_embedding_q_second_half Q cos sin Q_row_stride cos_row_stride
    sin_row_stride seqlen head_dim n_heads BLOCK_SIZE
  main := Q
  aux1 := cos
  aux2 := sin
  shape := [BLOCK_SIZE]
  readMain1 := fun p₀ p₁ i => p₀ * Q_row_stride + p₁ * 4 * head_dim + i.1.val
  readMain2 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * 4 * head_dim + i.1.val + head_dim / 2
  readAux1 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * cos_row_stride + i.1.val
  readAux2 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * sin_row_stride + i.1.val
  write := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * 4 * head_dim + i.1.val + head_dim / 2
  mask := fun _p₀ _p₁ i => i.1.val < head_dim / 2
  writeMask := fun _p₀ p₁ i => i.1.val < head_dim / 2 ∧ p₁ * 4 < n_heads

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `fast_rope_embedding.py`'s
`_rope_embedding`, both half-updates: for every disjoint flat placement of
`Q`/`cos`/`sin`, every program coordinate whose active lanes are in bounds, and
every launch state whose two `Q` windows and the `cos`/`sin` windows are pinned,
each slice terminates, every write-active lane of `Q` holds the genuine rotary
value — `q1·cos − q2·sin` for the first half, `q2·cos + q1·sin` for the second —
and every other memory cell is unchanged.

The four spec arguments are the **pre-state** lane values, which is the honest
reading of an in-place update. Dimension-general in every stride, `seqlen`,
`head_dim`, `n_heads` and `BLOCK_SIZE`, with **no** side-condition: the closed-form
readbacks' output-injectivity precondition is discharged outright by
`qFirst_inj` / `qSecond_inj` (each window is `base + lane`). -/
specification fast_rope_embedding_io_correctness (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      BLOCK_SIZE : Nat) :
    (ropeFirstIO Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen
        head_dim n_heads BLOCK_SIZE
      ⊨ fun _p₀ _p₁ q1 q2 c1 s1 i => ropeFirstSpecOf q1 q2 c1 s1 i) ∧
    (ropeSecondIO Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen
        head_dim n_heads BLOCK_SIZE
      ⊨ fun _p₀ _p₁ q1 q2 c1 s1 i => ropeSecondSpecOf q1 q2 c1 s1 i) := by
  constructor
  · refine InPlaceMaskedTileKernelIO.Implements.intro _ ?_ ?_ ?_
    · exact rope_first_flattenOk Q cos sin Q_row_stride cos_row_stride
        sin_row_stride seqlen head_dim n_heads BLOCK_SIZE
    · intro bounds s h1 h2 h3 h4 h5
      exact rope_first_traceSafe Q cos sin Q_row_stride cos_row_stride
        sin_row_stride seqlen head_dim n_heads BLOCK_SIZE bounds s
        (fun i hi => h1 (i, PUnit.unit) hi) (fun i hi => h2 (i, PUnit.unit) hi)
        (fun i hi => h3 (i, PUnit.unit) hi) (fun i hi => h4 (i, PUnit.unit) hi)
        (fun i hi => h5 (i, PUnit.unit) hi)
    · intro s₀ q1 q2 c1 s1 hp1 hp2 hp3 hp4
      obtain ⟨s', hexec, hval, hframe⟩ :=
        rope_first_region_run Q cos sin Q_row_stride cos_row_stride sin_row_stride
          seqlen head_dim n_heads BLOCK_SIZE s₀ q1 q2 c1 s1
          (fun i hi => hp1 i hi) (fun i hi => hp2 i hi) (fun i hi => hp3 i hi)
          (fun i hi => hp4 i hi)
      refine ⟨s', hexec, hval, fun r o hcond => hframe r o ?_⟩
      rcases hcond with h | h
      · exact Or.inl h
      · exact Or.inr fun i hi => h (i, PUnit.unit) hi
  · refine InPlaceMaskedTileKernelIO.Implements.intro _ ?_ ?_ ?_
    · exact rope_second_flattenOk Q cos sin Q_row_stride cos_row_stride
        sin_row_stride seqlen head_dim n_heads BLOCK_SIZE
    · intro bounds s h1 h2 h3 h4 h5
      exact rope_second_traceSafe Q cos sin Q_row_stride cos_row_stride
        sin_row_stride seqlen head_dim n_heads BLOCK_SIZE bounds s
        (fun i hi => h1 (i, PUnit.unit) hi) (fun i hi => h2 (i, PUnit.unit) hi)
        (fun i hi => h3 (i, PUnit.unit) hi) (fun i hi => h4 (i, PUnit.unit) hi)
        (fun i hi => h5 (i, PUnit.unit) hi)
    · intro s₀ q1 q2 c1 s1 hp1 hp2 hp3 hp4
      obtain ⟨s', hexec, hval, hframe⟩ :=
        rope_second_region_run Q cos sin Q_row_stride cos_row_stride sin_row_stride
          seqlen head_dim n_heads BLOCK_SIZE s₀ q1 q2 c1 s1
          (fun i hi => hp1 i hi) (fun i hi => hp2 i hi) (fun i hi => hp3 i hi)
          (fun i hi => hp4 i hi)
      refine ⟨s', hexec, hval, fun r o hcond => hframe r o ?_⟩
      rcases hcond with h | h
      · exact Or.inl h
      · exact Or.inr fun i hi => h (i, PUnit.unit) hi

end IOFace

end VeriTile.Bench.TritonBenchG.FastRopeEmbedding
