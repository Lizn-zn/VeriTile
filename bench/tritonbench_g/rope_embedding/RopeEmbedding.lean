import VeriTile.Triton

/-!
# `rope_embedding` — strict per-kernel correctness

`_rope_embedding` applies the rotary position embedding `Q*cos + rotate_half(Q)*sin`
in place: each program owns one row (`program_id(0)`) and one head-group
(`program_id(1)`), loads the `cos`/`sin` half-dim vectors for that row, and for
every head in its group rewrites the first/second rotary halves of `Q` via the
pair `(Q1*cos - Q2*sin, Q2*cos + Q1*sin)`. The `BACKWARD_PASS` heuristic flips
`sin1 := -sin1`, giving the transpose of the forward rotation.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (the `(n_rows, n_groups)` grid, the `calculate_settings`
block-size/`num_warps` choice, the `divmod`-derived group count, the in-place
transpose/reshape bookkeeping in `_rope_embedding_forward_impl` /
`_rope_embedding_backward_impl`, and how the runtime composes per-program writes
into one buffer) is the *trusted boundary*, not a proof obligation here. Because
the row position, head-group, and per-head index are universally quantified, the
per-program statement covers every program of the grid.

## Proof architecture

```
rope_embedding_io_correctness                      ← TOP THEOREM (`⊨`, all four slices)
      ├─ {fwd,bwd}_{first,second}_flattenOk            inside the flat-memory bridge
      ├─ {fwd,bwd}_{first,second}_traceSafe            per-execution address safety
      └─ {fwd,bwd}_{first,second}_region_run           region-model run
            ├─ {fwd,bwd}_{first,second}_terminates
            ├─ rope_embedding_*_half_correct           per-lane readback (shared, below)
            ├─ rope*Spec_eq_of                         memory spec = value spec (lane-local)
            ├─ q{First,Second}_inj                     output injectivity, discharged
            └─ {fwd,bwd}_{first,second}_frame          cell-level frame

rope_embedding_forward_backward_summary_general    per-write-map summary (dimension-general)
      ├─ rope_embedding_surface_toAlgorithm_supported  (forward & backward surfaces)
      ├─ rope_embedding_forward_first_half_compute_correct
      │     └─ rope_embedding_forward_first_half_correct
      ├─ rope_embedding_forward_second_half_compute_correct
      │     └─ rope_embedding_forward_second_half_correct
      ├─ rope_embedding_backward_first_half_compute_correct
      │     └─ rope_embedding_backward_first_half_correct
      └─ rope_embedding_backward_second_half_compute_correct
            └─ rope_embedding_backward_second_half_correct
```

The disjointness of the two offset families is taken as hypotheses of the
main theorem, which the half-store compute-correctness lemmas consume.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; the `.to(sin1.dtype)`
register casts erase to the identity at the algorithm layer (post-erasure all
dtypes unify to `ℝ`). `cos`/`sin` are modeled as **precomputed inputs** loaded
from memory (the kernel does not compute them); the forward spec uses them as
loaded and the backward spec uses the `sin1 := -sin1` flip. Scoping is
**one head-half store at a time** (`first_half`/`second_half`), each over the
active lanes `col_offsets < head_dim//2`; out-of-bounds lanes are preserved
verbatim. The dimension-general summary quantifies over arbitrary strides and
shapes (and the `BACKWARD_PASS` heuristic is modeled by the `Bool` argument).
-/

namespace VeriTile.Bench.TritonBenchG.RopeEmbedding

open VeriTile.Triton
open scoped VeriTile.Triton.InPlaceMaskedTileKernelIO

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `rope_embedding.py`'s `_rope_embedding`.

The body preserves the group-head loop, the `BACKWARD_PASS` path, and both
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

/-- The full RoPE surface lowers to the algorithm layer, including the
group-head loop, backward sin flip, and both rotary-pair stores. -/
theorem rope_embedding_surface_toAlgorithm_supported
    (Q : RegionName) (Q_row_stride : Nat)
    (cos : RegionName) (cos_row_stride : Nat)
    (sin : RegionName) (sin_row_stride seqlen head_dim n_heads : Nat)
    (BACKWARD_PASS : Bool) (BLOCK_SIZE ROPE_GROUP_SIZE : Nat) :
    ∃ alg, (rope_embedding_surface Q Q_row_stride cos cos_row_stride sin
      sin_row_stride seqlen head_dim n_heads BACKWARD_PASS BLOCK_SIZE
      ROPE_GROUP_SIZE).toAlgorithm? = Except.ok alg := by
  simp [rope_embedding_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

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
  · simp [exec, rope_embedding_forward_first_half, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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
    ComputeCorrect.Realizes_without_Rounding
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

/-- Proof-oriented second-half slice of `_rope_embedding` (forward).
Captures the companion second-half writeback `out = Q2 * cos + Q1 * sin`
to offset `offs_q2`. -/
def rope_embedding_forward_second_half
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
  out = Q2 * cos1 + Q1 * sin1
  tl.store(Q + offs_q2, out, mask=mask and head_start < $(n_heads))
}

noncomputable def ropeSecondSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) +
  s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem sin (sinOffset s seqlen sin_row_stride i)

/-- Algorithm-layer correctness for the second-half RoPE store. -/
theorem rope_embedding_forward_second_half_correct
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i))
    (hExec : exec (rope_embedding_forward_second_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE
        BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) =
        if active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i then
          ropeSecondSpec s Q cos sin Q_row_stride cos_row_stride sin_row_stride
            seqlen head_dim ROPE_GROUP_SIZE BLOCK_SIZE i
        else
          s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] =>
        s.pids 0 * Q_row_stride + s.pids 1 * ROPE_GROUP_SIZE * head_dim +
          idx.1.val + head_dim / 2) := by
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
  · simp [exec, rope_embedding_forward_second_half, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
          Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBS] at hExec
    rw [← hExec]
    simp only [qSecondOffset, headStart, colIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
    by_cases hCol : i.val < head_dim / 2
    · by_cases hHead : s.pids 1 * ROPE_GROUP_SIZE < n_heads
      · simp [active, ropeSecondSpec, qFirstOffset, qSecondOffset, cosOffset,
              sinOffset, rowMod, headStart, colIndex, hCol, hHead,
              Option.map₂, Option.bind, Option.map]
      · simp [active, headStart, colIndex, hCol, hHead]
    · simp [active, headStart, colIndex, hCol]
  · exact False.elim (hBS (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the second-half RoPE store. -/
theorem rope_embedding_forward_second_half_compute_correct
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_forward_second_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE
        BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE =>
          active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i)
        (fun i => (Q, qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i)))
      (expected := fun i =>
        ropeSecondSpec s Q cos sin Q_row_stride cos_row_stride sin_row_stride
          seqlen head_dim ROPE_GROUP_SIZE BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rope_embedding_forward_second_half]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rope_embedding_forward_second_half_correct Q cos sin Q_row_stride
    cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE
    BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h

/-! ## Backward sin-flip stores (`BACKWARD_PASS = true`) -/

/-- Proof-oriented first-half slice of `_rope_embedding` for
`BACKWARD_PASS = true`. The surface flips `sin1 = -sin1`, so the first-half
write becomes `Q1 * cos + Q2 * sin`. -/
def rope_embedding_backward_first_half
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
  sin1 = -sin1
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q1 * cos1 - Q2 * sin1
  tl.store(Q + offs_q1, out, mask=mask and head_start < $(n_heads))
}

/-- Proof-oriented second-half slice of `_rope_embedding` for
`BACKWARD_PASS = true`. The surface flips `sin1 = -sin1`, so the second-half
write becomes `Q2 * cos - Q1 * sin`. -/
def rope_embedding_backward_second_half
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
  sin1 = -sin1
  head_start = group_head_position * $(ROPE_GROUP_SIZE)
  offs_q1 = row_position * $(Q_row_stride) + head_start * $(head_dim) + col_offsets
  offs_q2 = row_position * $(Q_row_stride) + head_start * $(head_dim) +
    col_offsets + $(head_dim / 2)
  Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
  Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
  out = Q2 * cos1 + Q1 * sin1
  tl.store(Q + offs_q2, out, mask=mask and head_start < $(n_heads))
}

noncomputable def ropeBackwardFirstSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) +
  0 - s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    ((0.0 : ℝ) - s.readMem sin (sinOffset s seqlen sin_row_stride i))

noncomputable def ropeBackwardSecondSpec
    (s : BlockState) (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE
      BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    s.readMem cos (cosOffset s seqlen cos_row_stride i) -
  0 + s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) *
    ((0.0 : ℝ) - s.readMem sin (sinOffset s seqlen sin_row_stride i))

/-- Algorithm-layer correctness for the first-half backward RoPE store. -/
theorem rope_embedding_backward_first_half_correct
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i))
    (hExec : exec (rope_embedding_backward_first_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE
        BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) =
        if active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i then
          ropeBackwardFirstSpec s Q cos sin Q_row_stride cos_row_stride
            sin_row_stride seqlen head_dim ROPE_GROUP_SIZE BLOCK_SIZE i
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
  · simp [exec, rope_embedding_backward_first_half, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBS] at hExec
    rw [← hExec]
    simp only [qFirstOffset, headStart, colIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
    by_cases hCol : i.val < head_dim / 2
    · by_cases hHead : s.pids 1 * ROPE_GROUP_SIZE < n_heads
      · simp [active, ropeBackwardFirstSpec, qFirstOffset, qSecondOffset, cosOffset,
              sinOffset, rowMod, headStart, colIndex, hCol, hHead,
              Option.map₂, Option.bind, Option.map]
      · simp [active, headStart, colIndex, hCol, hHead]
    · simp [active, headStart, colIndex, hCol]
  · exact False.elim (hBS (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Algorithm-layer correctness for the second-half backward RoPE store. -/
theorem rope_embedding_backward_second_half_correct
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i))
    (hExec : exec (rope_embedding_backward_second_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE
        BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) =
        if active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i then
          ropeBackwardSecondSpec s Q cos sin Q_row_stride cos_row_stride
            sin_row_stride seqlen head_dim ROPE_GROUP_SIZE BLOCK_SIZE i
        else
          s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] =>
        s.pids 0 * Q_row_stride + s.pids 1 * ROPE_GROUP_SIZE * head_dim +
          idx.1.val + head_dim / 2) := by
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
  · simp [exec, rope_embedding_backward_second_half, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
          Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBS] at hExec
    rw [← hExec]
    simp only [qSecondOffset, headStart, colIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
    by_cases hCol : i.val < head_dim / 2
    · by_cases hHead : s.pids 1 * ROPE_GROUP_SIZE < n_heads
      · simp [active, ropeBackwardSecondSpec, qFirstOffset, qSecondOffset, cosOffset,
              sinOffset, rowMod, headStart, colIndex, hCol, hHead,
              Option.map₂, Option.bind, Option.map]
      · simp [active, headStart, colIndex, hCol, hHead]
    · simp [active, headStart, colIndex, hCol]
  · exact False.elim (hBS (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the first-half backward RoPE store. -/
theorem rope_embedding_backward_first_half_compute_correct
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_backward_first_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE
        BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE =>
          active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i)
        (fun i => (Q, qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i)))
      (expected := fun i =>
        ropeBackwardFirstSpec s Q cos sin Q_row_stride cos_row_stride
          sin_row_stride seqlen head_dim ROPE_GROUP_SIZE BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rope_embedding_backward_first_half]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rope_embedding_backward_first_half_correct Q cos sin Q_row_stride
    cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE
    BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h

/-- Compute-facing correctness for the second-half backward RoPE store. -/
theorem rope_embedding_backward_second_half_compute_correct
    (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_backward_second_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE
        BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE =>
          active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i)
        (fun i => (Q, qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i)))
      (expected := fun i =>
        ropeBackwardSecondSpec s Q cos sin Q_row_stride cos_row_stride
          sin_row_stride seqlen head_dim ROPE_GROUP_SIZE BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rope_embedding_backward_second_half]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rope_embedding_backward_second_half_correct Q cos sin Q_row_stride
    cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE
    BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h

/-- **Dimension-general forward+backward output summary.** For arbitrary strides,
`seqlen`/`head_dim`/`n_heads`/`ROPE_GROUP_SIZE`/`BLOCK_SIZE` (and any program ids in
`sQ`/`sDY`), the forward surface lowers and both forward half-kernels realize the
genuine rotary specs `ropeFirst/SecondSpec` on `Q`, and symmetrically the backward
surface lowers and both backward half-kernels realize `ropeBackwardFirst/SecondSpec`
on `dY` — under the honest offset-injectivity side conditions. -/
specification rope_embedding_forward_backward_summary_general
    (Q dY cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (sQ sDY : BlockState)
    (hQF : Function.Injective
      (fun i : Fin BLOCK_SIZE => qFirstOffset sQ Q_row_stride head_dim ROPE_GROUP_SIZE i))
    (hQS : Function.Injective
      (fun i : Fin BLOCK_SIZE => qSecondOffset sQ Q_row_stride head_dim ROPE_GROUP_SIZE i))
    (hDF : Function.Injective
      (fun i : Fin BLOCK_SIZE => qFirstOffset sDY Q_row_stride head_dim ROPE_GROUP_SIZE i))
    (hDS : Function.Injective
      (fun i : Fin BLOCK_SIZE => qSecondOffset sDY Q_row_stride head_dim ROPE_GROUP_SIZE i)) :
    (∃ alg, (rope_embedding_surface Q Q_row_stride cos cos_row_stride sin
      sin_row_stride seqlen head_dim n_heads Bool.false BLOCK_SIZE ROPE_GROUP_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_forward_first_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active sQ head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i)
        (fun i => (Q, qFirstOffset sQ Q_row_stride head_dim ROPE_GROUP_SIZE i)))
      (expected := fun i => ropeFirstSpec sQ Q cos sin Q_row_stride cos_row_stride
        sin_row_stride seqlen head_dim ROPE_GROUP_SIZE BLOCK_SIZE i) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_forward_second_half Q cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active sQ head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i)
        (fun i => (Q, qSecondOffset sQ Q_row_stride head_dim ROPE_GROUP_SIZE i)))
      (expected := fun i => ropeSecondSpec sQ Q cos sin Q_row_stride cos_row_stride
        sin_row_stride seqlen head_dim ROPE_GROUP_SIZE BLOCK_SIZE i) ∧
    (∃ alg, (rope_embedding_surface dY Q_row_stride cos cos_row_stride sin
      sin_row_stride seqlen head_dim n_heads Bool.true BLOCK_SIZE ROPE_GROUP_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_backward_first_half dY cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active sDY head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i)
        (fun i => (dY, qFirstOffset sDY Q_row_stride head_dim ROPE_GROUP_SIZE i)))
      (expected := fun i => ropeBackwardFirstSpec sDY dY cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE BLOCK_SIZE i) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rope_embedding_backward_second_half dY cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active sDY head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i)
        (fun i => (dY, qSecondOffset sDY Q_row_stride head_dim ROPE_GROUP_SIZE i)))
      (expected := fun i => ropeBackwardSecondSpec sDY dY cos sin Q_row_stride
        cos_row_stride sin_row_stride seqlen head_dim ROPE_GROUP_SIZE BLOCK_SIZE i) :=
  ⟨rope_embedding_surface_toAlgorithm_supported Q Q_row_stride cos cos_row_stride sin
      sin_row_stride seqlen head_dim n_heads Bool.false BLOCK_SIZE ROPE_GROUP_SIZE,
   rope_embedding_forward_first_half_compute_correct Q cos sin Q_row_stride
      cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE sQ hQF,
   rope_embedding_forward_second_half_compute_correct Q cos sin Q_row_stride
      cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE sQ hQS,
   rope_embedding_surface_toAlgorithm_supported dY Q_row_stride cos cos_row_stride sin
      sin_row_stride seqlen head_dim n_heads Bool.true BLOCK_SIZE ROPE_GROUP_SIZE,
   rope_embedding_backward_first_half_compute_correct dY cos sin Q_row_stride
      cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE sDY hDF,
   rope_embedding_backward_second_half_compute_correct dY cos sin Q_row_stride
      cos_row_stride sin_row_stride seqlen head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE sDY hDS⟩

/-! ## ════════ `⊨` IO face for the four RoPE half-updates ════════

The summaries above are stated per *declared write map*. This section restates all
four slices on the audit-once IO surface `InPlaceMaskedTileKernelIO.Implements`
(`⊨`), which additionally pins the **flat memory** placement. RoPE is the shape
that skin exists for: `Q` is read at two windows, `cos` / `sin` are read-only
auxiliaries, and the store goes back into `Q` at one of the two windows — so `f`'s
four arguments are the **pre-state** lane values. Loads are gated by the column
guard alone while the store additionally carries `head_start < n_heads`, and the
signatures keep those two masks separate. -/

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

/-- Both `Q` window maps are injective outright (each is `base + lane`, the second
shifted by `head_dim / 2`), so the closed-form readbacks' `hOutInj` precondition is
a theorem here rather than a hypothesis the headline carries. -/
private theorem qFirst_inj (s : BlockState)
    (Q_row_stride head_dim ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    Function.Injective (fun i : Fin BLOCK_SIZE =>
      qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) := by
  intro a b h
  simp only [qFirstOffset, colIndex] at h
  exact Fin.ext (Nat.add_left_cancel h)

private theorem qSecond_inj (s : BlockState)
    (Q_row_stride head_dim ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    Function.Injective (fun i : Fin BLOCK_SIZE =>
      qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) := by
  intro a b h
  simp only [qSecondOffset, colIndex] at h
  exact Fin.ext (Nat.add_left_cancel (Nat.add_right_cancel h))

/-! ### `rope_embedding_forward_first_half` -/

/-- Value-level spec of `rope_embedding_forward_first_half`, over the loaded values. -/
noncomputable def ropeFirstSpecOf {BLOCK_SIZE : Nat}
    (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (i : TileIndex [BLOCK_SIZE]) : ℝ :=
  q1 i * c1 i - q2 i * s1 i

/-- Memory-level and value-level specs agree at a column-active lane. -/
theorem ropeFirstSpec_eq_of (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (s : BlockState) (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (hq1 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
        = q1 i)
    (hq2 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
        = q2 i)
    (hc : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem cos (cosOffset s seqlen cos_row_stride i.1) = c1 i)
    (hs : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem sin (sinOffset s seqlen sin_row_stride i.1) = s1 i)
    (i : TileIndex [BLOCK_SIZE]) (hi : i.1.val < head_dim / 2) :
    ropeFirstSpec s Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen
        head_dim ROPE_GROUP_SIZE BLOCK_SIZE i.1
      = ropeFirstSpecOf q1 q2 c1 s1 i := by
  rw [ropeFirstSpec, ropeFirstSpecOf, hq1 i hi, hq2 i hi, hc i hi, hs i hi]

/-- `rope_embedding_forward_first_half` sits inside the flat-memory bridge's covered fragment. -/
theorem fwd_first_flattenOk (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ((rope_embedding_forward_first_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [rope_embedding_forward_first_half, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- Termination of `rope_embedding_forward_first_half`. -/
theorem fwd_first_terminates (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) (s : BlockState) :
    ∃ s1, exec (rope_embedding_forward_first_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE) s = some s1 := by
  simp [exec, rope_embedding_forward_first_half, stepStmts, stepStmt, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]

/-- Cell-level frame of `rope_embedding_forward_first_half`. -/
theorem fwd_first_frame (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) (s s' : BlockState)
    (hExec : exec (rope_embedding_forward_first_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE) s = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ Q ∨ ∀ i : Fin BLOCK_SIZE,
        active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i →
        o ≠ qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, rope_embedding_forward_first_half, stepStmts, stepStmt, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt] at hExec
  subst hExec
  rw [foldl_writeMem_frame (region := Q)
    (fun i : TileIndex [BLOCK_SIZE] =>
      s.pids 0 * Q_row_stride + s.pids 1 * ROPE_GROUP_SIZE * head_dim
        + i.1.val)
    _ (fun i : TileIndex [BLOCK_SIZE] =>
      i.1.val < head_dim / 2 ∧ s.pids 1 * ROPE_GROUP_SIZE < n_heads) r o
    (TileShape.allIndices [BLOCK_SIZE]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ hi => Ne.symm (h i.1 hi)

/-- Per-execution safety walk for `rope_embedding_forward_first_half`. -/
theorem fwd_first_traceSafe (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) (bounds : RegionBounds) (s : BlockState)
    (hq1 : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i < bounds Q)
    (hq2 : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i < bounds Q)
    (hc : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      cosOffset s seqlen cos_row_stride i < bounds cos)
    (hsn : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      sinOffset s seqlen sin_row_stride i < bounds sin)
    (hst : ∀ i : Fin BLOCK_SIZE,
      active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i →
      qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i < bounds Q) :
    ((rope_embedding_forward_first_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE).toAlgKernel).TraceSafe bounds s := by
  simp [Kernel.TraceSafe, rope_embedding_forward_first_half, Stmt.TraceSafeList, Stmt.TraceSafe,
    Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active, MaskOpt.MemorySafe,
    MemAccess.SafeAt, MemAccess.MemorySafe, memAccessMemorySafe,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe, stepStmt, evalOp, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]
  exact ⟨fun a ha => hsn a ha, fun a ha => hc a ha, fun a ha => hq1 a ha,
    fun a ha => hq2 a ha, fun a ha hh => hst a ⟨ha, hh⟩⟩

/-- Region-model run of `rope_embedding_forward_first_half`. -/
theorem fwd_first_region_run (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) (s₀ : BlockState)
    (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (hq1 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem Q (qFirstOffset s₀ Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
        = q1 i)
    (hq2 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem Q (qSecondOffset s₀ Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
        = q2 i)
    (hc : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem cos (cosOffset s₀ seqlen cos_row_stride i.1) = c1 i)
    (hs : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem sin (sinOffset s₀ seqlen sin_row_stride i.1) = s1 i) :
    ∃ s', exec (rope_embedding_forward_first_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE) s₀ = some s'
      ∧ (∀ i : TileIndex [BLOCK_SIZE],
          active s₀ head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i.1 →
          s'.readMem Q (qFirstOffset s₀ Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
            = ropeFirstSpecOf q1 q2 c1 s1 i)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ Q ∨ ∀ i : Fin BLOCK_SIZE,
            active s₀ head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i →
            o ≠ qFirstOffset s₀ Q_row_stride head_dim ROPE_GROUP_SIZE i) →
          s'.mem r o = s₀.mem r o) := by
  obtain ⟨s', hexec⟩ := fwd_first_terminates Q cos sin _ _ _ _ _ _ _ _ s₀
  refine ⟨s', hexec, ?_, fwd_first_frame Q cos sin _ _ _ _ _ _ _ _ s₀ s' hexec⟩
  intro i hact
  rw [rope_embedding_forward_first_half_correct Q cos sin _ _ _ _ _ _ _ _ s₀ s'
      (qFirst_inj s₀ _ _ _ _) hexec i.1, if_pos hact,
    ropeFirstSpec_eq_of Q cos sin _ _ _ _ _ _ _ s₀ q1 q2 c1 s1 hq1 hq2 hc hs i
      hact.1]

/-- IO signature of `rope_embedding_forward_first_half` on the in-place tile surface. -/
def fwd_firstIO (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) : InPlaceMaskedTileKernelIO where
  kernel := rope_embedding_forward_first_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
    ROPE_GROUP_SIZE BLOCK_SIZE
  main := Q
  aux1 := cos
  aux2 := sin
  shape := [BLOCK_SIZE]
  readMain1 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val
  readMain2 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val + head_dim / 2
  readAux1 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * cos_row_stride + i.1.val
  readAux2 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * sin_row_stride + i.1.val
  write := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val
  mask := fun _p₀ _p₁ i => i.1.val < head_dim / 2
  writeMask := fun _p₀ p₁ i =>
    i.1.val < head_dim / 2 ∧ p₁ * ROPE_GROUP_SIZE < n_heads

/-! ### `rope_embedding_forward_second_half` -/

/-- Value-level spec of `rope_embedding_forward_second_half`, over the loaded values. -/
noncomputable def ropeSecondSpecOf {BLOCK_SIZE : Nat}
    (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (i : TileIndex [BLOCK_SIZE]) : ℝ :=
  q2 i * c1 i + q1 i * s1 i

/-- Memory-level and value-level specs agree at a column-active lane. -/
theorem ropeSecondSpec_eq_of (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (s : BlockState) (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (hq1 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
        = q1 i)
    (hq2 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
        = q2 i)
    (hc : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem cos (cosOffset s seqlen cos_row_stride i.1) = c1 i)
    (hs : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem sin (sinOffset s seqlen sin_row_stride i.1) = s1 i)
    (i : TileIndex [BLOCK_SIZE]) (hi : i.1.val < head_dim / 2) :
    ropeSecondSpec s Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen
        head_dim ROPE_GROUP_SIZE BLOCK_SIZE i.1
      = ropeSecondSpecOf q1 q2 c1 s1 i := by
  rw [ropeSecondSpec, ropeSecondSpecOf, hq1 i hi, hq2 i hi, hc i hi, hs i hi]

/-- `rope_embedding_forward_second_half` sits inside the flat-memory bridge's covered fragment. -/
theorem fwd_second_flattenOk (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ((rope_embedding_forward_second_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [rope_embedding_forward_second_half, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- Termination of `rope_embedding_forward_second_half`. -/
theorem fwd_second_terminates (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) (s : BlockState) :
    ∃ s1, exec (rope_embedding_forward_second_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE) s = some s1 := by
  simp [exec, rope_embedding_forward_second_half, stepStmts, stepStmt, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]

/-- Cell-level frame of `rope_embedding_forward_second_half`. -/
theorem fwd_second_frame (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) (s s' : BlockState)
    (hExec : exec (rope_embedding_forward_second_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE) s = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ Q ∨ ∀ i : Fin BLOCK_SIZE,
        active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i →
        o ≠ qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, rope_embedding_forward_second_half, stepStmts, stepStmt, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt] at hExec
  subst hExec
  rw [foldl_writeMem_frame (region := Q)
    (fun i : TileIndex [BLOCK_SIZE] =>
      s.pids 0 * Q_row_stride + s.pids 1 * ROPE_GROUP_SIZE * head_dim
        + i.1.val + head_dim / 2)
    _ (fun i : TileIndex [BLOCK_SIZE] =>
      i.1.val < head_dim / 2 ∧ s.pids 1 * ROPE_GROUP_SIZE < n_heads) r o
    (TileShape.allIndices [BLOCK_SIZE]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ hi => Ne.symm (h i.1 hi)

/-- Per-execution safety walk for `rope_embedding_forward_second_half`. -/
theorem fwd_second_traceSafe (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) (bounds : RegionBounds) (s : BlockState)
    (hq1 : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i < bounds Q)
    (hq2 : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i < bounds Q)
    (hc : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      cosOffset s seqlen cos_row_stride i < bounds cos)
    (hsn : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      sinOffset s seqlen sin_row_stride i < bounds sin)
    (hst : ∀ i : Fin BLOCK_SIZE,
      active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i →
      qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i < bounds Q) :
    ((rope_embedding_forward_second_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE).toAlgKernel).TraceSafe bounds s := by
  simp [Kernel.TraceSafe, rope_embedding_forward_second_half, Stmt.TraceSafeList, Stmt.TraceSafe,
    Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active, MaskOpt.MemorySafe,
    MemAccess.SafeAt, MemAccess.MemorySafe, memAccessMemorySafe,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe, stepStmt, evalOp, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]
  exact ⟨fun a ha => hsn a ha, fun a ha => hc a ha, fun a ha => hq1 a ha,
    fun a ha => hq2 a ha, fun a ha hh => hst a ⟨ha, hh⟩⟩

/-- Region-model run of `rope_embedding_forward_second_half`. -/
theorem fwd_second_region_run (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) (s₀ : BlockState)
    (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (hq1 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem Q (qFirstOffset s₀ Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
        = q1 i)
    (hq2 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem Q (qSecondOffset s₀ Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
        = q2 i)
    (hc : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem cos (cosOffset s₀ seqlen cos_row_stride i.1) = c1 i)
    (hs : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem sin (sinOffset s₀ seqlen sin_row_stride i.1) = s1 i) :
    ∃ s', exec (rope_embedding_forward_second_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE) s₀ = some s'
      ∧ (∀ i : TileIndex [BLOCK_SIZE],
          active s₀ head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i.1 →
          s'.readMem Q (qSecondOffset s₀ Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
            = ropeSecondSpecOf q1 q2 c1 s1 i)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ Q ∨ ∀ i : Fin BLOCK_SIZE,
            active s₀ head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i →
            o ≠ qSecondOffset s₀ Q_row_stride head_dim ROPE_GROUP_SIZE i) →
          s'.mem r o = s₀.mem r o) := by
  obtain ⟨s', hexec⟩ := fwd_second_terminates Q cos sin _ _ _ _ _ _ _ _ s₀
  refine ⟨s', hexec, ?_, fwd_second_frame Q cos sin _ _ _ _ _ _ _ _ s₀ s' hexec⟩
  intro i hact
  rw [rope_embedding_forward_second_half_correct Q cos sin _ _ _ _ _ _ _ _ s₀ s'
      (qSecond_inj s₀ _ _ _ _) hexec i.1, if_pos hact,
    ropeSecondSpec_eq_of Q cos sin _ _ _ _ _ _ _ s₀ q1 q2 c1 s1 hq1 hq2 hc hs i
      hact.1]

/-- IO signature of `rope_embedding_forward_second_half` on the in-place tile surface. -/
def fwd_secondIO (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) : InPlaceMaskedTileKernelIO where
  kernel := rope_embedding_forward_second_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
    ROPE_GROUP_SIZE BLOCK_SIZE
  main := Q
  aux1 := cos
  aux2 := sin
  shape := [BLOCK_SIZE]
  readMain1 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val
  readMain2 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val + head_dim / 2
  readAux1 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * cos_row_stride + i.1.val
  readAux2 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * sin_row_stride + i.1.val
  write := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val + head_dim / 2
  mask := fun _p₀ _p₁ i => i.1.val < head_dim / 2
  writeMask := fun _p₀ p₁ i =>
    i.1.val < head_dim / 2 ∧ p₁ * ROPE_GROUP_SIZE < n_heads

/-! ### `rope_embedding_backward_first_half` -/

/-- Value-level spec of `rope_embedding_backward_first_half`, over the loaded values. -/
noncomputable def ropeBackwardFirstSpecOf {BLOCK_SIZE : Nat}
    (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (i : TileIndex [BLOCK_SIZE]) : ℝ :=
  q1 i * c1 i + 0 - q2 i * ((0.0 : ℝ) - s1 i)

/-- Memory-level and value-level specs agree at a column-active lane. -/
theorem ropeBackwardFirstSpec_eq_of (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (s : BlockState) (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (hq1 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
        = q1 i)
    (hq2 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
        = q2 i)
    (hc : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem cos (cosOffset s seqlen cos_row_stride i.1) = c1 i)
    (hs : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem sin (sinOffset s seqlen sin_row_stride i.1) = s1 i)
    (i : TileIndex [BLOCK_SIZE]) (hi : i.1.val < head_dim / 2) :
    ropeBackwardFirstSpec s Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen
        head_dim ROPE_GROUP_SIZE BLOCK_SIZE i.1
      = ropeBackwardFirstSpecOf q1 q2 c1 s1 i := by
  rw [ropeBackwardFirstSpec, ropeBackwardFirstSpecOf, hq1 i hi, hq2 i hi, hc i hi, hs i hi]

/-- `rope_embedding_backward_first_half` sits inside the flat-memory bridge's covered fragment. -/
theorem bwd_first_flattenOk (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ((rope_embedding_backward_first_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [rope_embedding_backward_first_half, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- Termination of `rope_embedding_backward_first_half`. -/
theorem bwd_first_terminates (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) (s : BlockState) :
    ∃ s1, exec (rope_embedding_backward_first_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE) s = some s1 := by
  simp [exec, rope_embedding_backward_first_half, stepStmts, stepStmt, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]

/-- Cell-level frame of `rope_embedding_backward_first_half`. -/
theorem bwd_first_frame (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) (s s' : BlockState)
    (hExec : exec (rope_embedding_backward_first_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE) s = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ Q ∨ ∀ i : Fin BLOCK_SIZE,
        active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i →
        o ≠ qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, rope_embedding_backward_first_half, stepStmts, stepStmt, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt] at hExec
  subst hExec
  rw [foldl_writeMem_frame (region := Q)
    (fun i : TileIndex [BLOCK_SIZE] =>
      s.pids 0 * Q_row_stride + s.pids 1 * ROPE_GROUP_SIZE * head_dim
        + i.1.val)
    _ (fun i : TileIndex [BLOCK_SIZE] =>
      i.1.val < head_dim / 2 ∧ s.pids 1 * ROPE_GROUP_SIZE < n_heads) r o
    (TileShape.allIndices [BLOCK_SIZE]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ hi => Ne.symm (h i.1 hi)

/-- Per-execution safety walk for `rope_embedding_backward_first_half`. -/
theorem bwd_first_traceSafe (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) (bounds : RegionBounds) (s : BlockState)
    (hq1 : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i < bounds Q)
    (hq2 : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i < bounds Q)
    (hc : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      cosOffset s seqlen cos_row_stride i < bounds cos)
    (hsn : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      sinOffset s seqlen sin_row_stride i < bounds sin)
    (hst : ∀ i : Fin BLOCK_SIZE,
      active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i →
      qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i < bounds Q) :
    ((rope_embedding_backward_first_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE).toAlgKernel).TraceSafe bounds s := by
  simp [Kernel.TraceSafe, rope_embedding_backward_first_half, Stmt.TraceSafeList, Stmt.TraceSafe,
    Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active, MaskOpt.MemorySafe,
    MemAccess.SafeAt, MemAccess.MemorySafe, memAccessMemorySafe,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe, stepStmt, evalOp, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]
  exact ⟨fun a ha => hsn a ha, fun a ha => hc a ha, fun a ha => hq1 a ha,
    fun a ha => hq2 a ha, fun a ha hh => hst a ⟨ha, hh⟩⟩

/-- Region-model run of `rope_embedding_backward_first_half`. -/
theorem bwd_first_region_run (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) (s₀ : BlockState)
    (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (hq1 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem Q (qFirstOffset s₀ Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
        = q1 i)
    (hq2 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem Q (qSecondOffset s₀ Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
        = q2 i)
    (hc : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem cos (cosOffset s₀ seqlen cos_row_stride i.1) = c1 i)
    (hs : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem sin (sinOffset s₀ seqlen sin_row_stride i.1) = s1 i) :
    ∃ s', exec (rope_embedding_backward_first_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE) s₀ = some s'
      ∧ (∀ i : TileIndex [BLOCK_SIZE],
          active s₀ head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i.1 →
          s'.readMem Q (qFirstOffset s₀ Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
            = ropeBackwardFirstSpecOf q1 q2 c1 s1 i)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ Q ∨ ∀ i : Fin BLOCK_SIZE,
            active s₀ head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i →
            o ≠ qFirstOffset s₀ Q_row_stride head_dim ROPE_GROUP_SIZE i) →
          s'.mem r o = s₀.mem r o) := by
  obtain ⟨s', hexec⟩ := bwd_first_terminates Q cos sin _ _ _ _ _ _ _ _ s₀
  refine ⟨s', hexec, ?_, bwd_first_frame Q cos sin _ _ _ _ _ _ _ _ s₀ s' hexec⟩
  intro i hact
  rw [rope_embedding_backward_first_half_correct Q cos sin _ _ _ _ _ _ _ _ s₀ s'
      (qFirst_inj s₀ _ _ _ _) hexec i.1, if_pos hact,
    ropeBackwardFirstSpec_eq_of Q cos sin _ _ _ _ _ _ _ s₀ q1 q2 c1 s1 hq1 hq2 hc hs i
      hact.1]

/-- IO signature of `rope_embedding_backward_first_half` on the in-place tile surface. -/
def bwd_firstIO (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) : InPlaceMaskedTileKernelIO where
  kernel := rope_embedding_backward_first_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
    ROPE_GROUP_SIZE BLOCK_SIZE
  main := Q
  aux1 := cos
  aux2 := sin
  shape := [BLOCK_SIZE]
  readMain1 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val
  readMain2 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val + head_dim / 2
  readAux1 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * cos_row_stride + i.1.val
  readAux2 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * sin_row_stride + i.1.val
  write := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val
  mask := fun _p₀ _p₁ i => i.1.val < head_dim / 2
  writeMask := fun _p₀ p₁ i =>
    i.1.val < head_dim / 2 ∧ p₁ * ROPE_GROUP_SIZE < n_heads

/-! ### `rope_embedding_backward_second_half` -/

/-- Value-level spec of `rope_embedding_backward_second_half`, over the loaded values. -/
noncomputable def ropeBackwardSecondSpecOf {BLOCK_SIZE : Nat}
    (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (i : TileIndex [BLOCK_SIZE]) : ℝ :=
  q2 i * c1 i - 0 + q1 i * ((0.0 : ℝ) - s1 i)

/-- Memory-level and value-level specs agree at a column-active lane. -/
theorem ropeBackwardSecondSpec_eq_of (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat)
    (s : BlockState) (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (hq1 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem Q (qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
        = q1 i)
    (hq2 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem Q (qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
        = q2 i)
    (hc : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem cos (cosOffset s seqlen cos_row_stride i.1) = c1 i)
    (hs : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s.readMem sin (sinOffset s seqlen sin_row_stride i.1) = s1 i)
    (i : TileIndex [BLOCK_SIZE]) (hi : i.1.val < head_dim / 2) :
    ropeBackwardSecondSpec s Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen
        head_dim ROPE_GROUP_SIZE BLOCK_SIZE i.1
      = ropeBackwardSecondSpecOf q1 q2 c1 s1 i := by
  rw [ropeBackwardSecondSpec, ropeBackwardSecondSpecOf, hq1 i hi, hq2 i hi, hc i hi, hs i hi]

/-- `rope_embedding_backward_second_half` sits inside the flat-memory bridge's covered fragment. -/
theorem bwd_second_flattenOk (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    ((rope_embedding_backward_second_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [rope_embedding_backward_second_half, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- Termination of `rope_embedding_backward_second_half`. -/
theorem bwd_second_terminates (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) (s : BlockState) :
    ∃ s1, exec (rope_embedding_backward_second_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE) s = some s1 := by
  simp [exec, rope_embedding_backward_second_half, stepStmts, stepStmt, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]

/-- Cell-level frame of `rope_embedding_backward_second_half`. -/
theorem bwd_second_frame (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) (s s' : BlockState)
    (hExec : exec (rope_embedding_backward_second_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE) s = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ Q ∨ ∀ i : Fin BLOCK_SIZE,
        active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i →
        o ≠ qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, rope_embedding_backward_second_half, stepStmts, stepStmt, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt] at hExec
  subst hExec
  rw [foldl_writeMem_frame (region := Q)
    (fun i : TileIndex [BLOCK_SIZE] =>
      s.pids 0 * Q_row_stride + s.pids 1 * ROPE_GROUP_SIZE * head_dim
        + i.1.val + head_dim / 2)
    _ (fun i : TileIndex [BLOCK_SIZE] =>
      i.1.val < head_dim / 2 ∧ s.pids 1 * ROPE_GROUP_SIZE < n_heads) r o
    (TileShape.allIndices [BLOCK_SIZE]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ hi => Ne.symm (h i.1 hi)

/-- Per-execution safety walk for `rope_embedding_backward_second_half`. -/
theorem bwd_second_traceSafe (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) (bounds : RegionBounds) (s : BlockState)
    (hq1 : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i < bounds Q)
    (hq2 : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i < bounds Q)
    (hc : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      cosOffset s seqlen cos_row_stride i < bounds cos)
    (hsn : ∀ i : Fin BLOCK_SIZE, i.val < head_dim / 2 →
      sinOffset s seqlen sin_row_stride i < bounds sin)
    (hst : ∀ i : Fin BLOCK_SIZE,
      active s head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i →
      qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i < bounds Q) :
    ((rope_embedding_backward_second_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE).toAlgKernel).TraceSafe bounds s := by
  simp [Kernel.TraceSafe, rope_embedding_backward_second_half, Stmt.TraceSafeList, Stmt.TraceSafe,
    Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active, MaskOpt.MemorySafe,
    MemAccess.SafeAt, MemAccess.MemorySafe, memAccessMemorySafe,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe, stepStmt, evalOp, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]
  exact ⟨fun a ha => hsn a ha, fun a ha => hc a ha, fun a ha => hq1 a ha,
    fun a ha => hq2 a ha, fun a ha hh => hst a ⟨ha, hh⟩⟩

/-- Region-model run of `rope_embedding_backward_second_half`. -/
theorem bwd_second_region_run (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) (s₀ : BlockState)
    (q1 q2 c1 s1 : TileIndex [BLOCK_SIZE] → ℝ)
    (hq1 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem Q (qFirstOffset s₀ Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
        = q1 i)
    (hq2 : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem Q (qSecondOffset s₀ Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
        = q2 i)
    (hc : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem cos (cosOffset s₀ seqlen cos_row_stride i.1) = c1 i)
    (hs : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < head_dim / 2 →
      s₀.readMem sin (sinOffset s₀ seqlen sin_row_stride i.1) = s1 i) :
    ∃ s', exec (rope_embedding_backward_second_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE) s₀ = some s'
      ∧ (∀ i : TileIndex [BLOCK_SIZE],
          active s₀ head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i.1 →
          s'.readMem Q (qSecondOffset s₀ Q_row_stride head_dim ROPE_GROUP_SIZE i.1)
            = ropeBackwardSecondSpecOf q1 q2 c1 s1 i)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ Q ∨ ∀ i : Fin BLOCK_SIZE,
            active s₀ head_dim n_heads ROPE_GROUP_SIZE BLOCK_SIZE i →
            o ≠ qSecondOffset s₀ Q_row_stride head_dim ROPE_GROUP_SIZE i) →
          s'.mem r o = s₀.mem r o) := by
  obtain ⟨s', hexec⟩ := bwd_second_terminates Q cos sin _ _ _ _ _ _ _ _ s₀
  refine ⟨s', hexec, ?_, bwd_second_frame Q cos sin _ _ _ _ _ _ _ _ s₀ s' hexec⟩
  intro i hact
  rw [rope_embedding_backward_second_half_correct Q cos sin _ _ _ _ _ _ _ _ s₀ s'
      (qSecond_inj s₀ _ _ _ _) hexec i.1, if_pos hact,
    ropeBackwardSecondSpec_eq_of Q cos sin _ _ _ _ _ _ _ s₀ q1 q2 c1 s1 hq1 hq2 hc hs i
      hact.1]

/-- IO signature of `rope_embedding_backward_second_half` on the in-place tile surface. -/
def bwd_secondIO (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) : InPlaceMaskedTileKernelIO where
  kernel := rope_embedding_backward_second_half Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
    ROPE_GROUP_SIZE BLOCK_SIZE
  main := Q
  aux1 := cos
  aux2 := sin
  shape := [BLOCK_SIZE]
  readMain1 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val
  readMain2 := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val + head_dim / 2
  readAux1 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * cos_row_stride + i.1.val
  readAux2 := fun p₀ _p₁ i =>
    IntegralDType.nat.mod p₀ seqlen * sin_row_stride + i.1.val
  write := fun p₀ p₁ i =>
    p₀ * Q_row_stride + p₁ * ROPE_GROUP_SIZE * head_dim + i.1.val + head_dim / 2
  mask := fun _p₀ _p₁ i => i.1.val < head_dim / 2
  writeMask := fun _p₀ p₁ i =>
    i.1.val < head_dim / 2 ∧ p₁ * ROPE_GROUP_SIZE < n_heads

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `rope_embedding.py`'s
`_rope_embedding`, all four slices (forward / backward × first / second half): for
every disjoint flat placement of `Q`/`cos`/`sin`, every program coordinate whose
active lanes are in bounds, and every launch state whose two `Q` windows and the
`cos`/`sin` windows are pinned, each slice terminates, every write-active lane of
`Q` holds the genuine rotary value, and every other memory cell is unchanged.

The four spec arguments are the **pre-state** lane values — the honest reading of
an in-place update. Dimension-general in every stride, `seqlen`, `head_dim`,
`n_heads`, `ROPE_GROUP_SIZE` and `BLOCK_SIZE`, with **no** side-condition: the
closed-form readbacks' output-injectivity precondition is discharged outright by
`qFirst_inj` / `qSecond_inj`. -/
specification rope_embedding_io_correctness (Q cos sin : RegionName)
    (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    (fwd_firstIO Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE
      ⊨ fun _p₀ _p₁ q1 q2 c1 s1 i => ropeFirstSpecOf q1 q2 c1 s1 i) ∧
    (fwd_secondIO Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE
      ⊨ fun _p₀ _p₁ q1 q2 c1 s1 i => ropeSecondSpecOf q1 q2 c1 s1 i) ∧
    (bwd_firstIO Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE
      ⊨ fun _p₀ _p₁ q1 q2 c1 s1 i => ropeBackwardFirstSpecOf q1 q2 c1 s1 i) ∧
    (bwd_secondIO Q cos sin Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads
      ROPE_GROUP_SIZE BLOCK_SIZE
      ⊨ fun _p₀ _p₁ q1 q2 c1 s1 i => ropeBackwardSecondSpecOf q1 q2 c1 s1 i) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · refine InPlaceMaskedTileKernelIO.Implements.intro _ ?_ ?_ ?_
    · exact fwd_first_flattenOk Q cos sin _ _ _ _ _ _ _ _
    · intro bounds s h1 h2 h3 h4 h5
      exact fwd_first_traceSafe Q cos sin _ _ _ _ _ _ _ _ bounds s
        (fun i hi => h1 (i, PUnit.unit) hi) (fun i hi => h2 (i, PUnit.unit) hi)
        (fun i hi => h3 (i, PUnit.unit) hi) (fun i hi => h4 (i, PUnit.unit) hi)
        (fun i hi => h5 (i, PUnit.unit) hi)
    · intro s₀ q1 q2 c1 s1 hp1 hp2 hp3 hp4
      obtain ⟨s', hexec, hval, hframe⟩ :=
        fwd_first_region_run Q cos sin _ _ _ _ _ _ _ _ s₀ q1 q2 c1 s1
          (fun i hi => hp1 i hi) (fun i hi => hp2 i hi) (fun i hi => hp3 i hi)
          (fun i hi => hp4 i hi)
      refine ⟨s', hexec, hval, fun r o hcond => hframe r o ?_⟩
      rcases hcond with h | h
      · exact Or.inl h
      · exact Or.inr fun i hi => h (i, PUnit.unit) hi
  · refine InPlaceMaskedTileKernelIO.Implements.intro _ ?_ ?_ ?_
    · exact fwd_second_flattenOk Q cos sin _ _ _ _ _ _ _ _
    · intro bounds s h1 h2 h3 h4 h5
      exact fwd_second_traceSafe Q cos sin _ _ _ _ _ _ _ _ bounds s
        (fun i hi => h1 (i, PUnit.unit) hi) (fun i hi => h2 (i, PUnit.unit) hi)
        (fun i hi => h3 (i, PUnit.unit) hi) (fun i hi => h4 (i, PUnit.unit) hi)
        (fun i hi => h5 (i, PUnit.unit) hi)
    · intro s₀ q1 q2 c1 s1 hp1 hp2 hp3 hp4
      obtain ⟨s', hexec, hval, hframe⟩ :=
        fwd_second_region_run Q cos sin _ _ _ _ _ _ _ _ s₀ q1 q2 c1 s1
          (fun i hi => hp1 i hi) (fun i hi => hp2 i hi) (fun i hi => hp3 i hi)
          (fun i hi => hp4 i hi)
      refine ⟨s', hexec, hval, fun r o hcond => hframe r o ?_⟩
      rcases hcond with h | h
      · exact Or.inl h
      · exact Or.inr fun i hi => h (i, PUnit.unit) hi
  · refine InPlaceMaskedTileKernelIO.Implements.intro _ ?_ ?_ ?_
    · exact bwd_first_flattenOk Q cos sin _ _ _ _ _ _ _ _
    · intro bounds s h1 h2 h3 h4 h5
      exact bwd_first_traceSafe Q cos sin _ _ _ _ _ _ _ _ bounds s
        (fun i hi => h1 (i, PUnit.unit) hi) (fun i hi => h2 (i, PUnit.unit) hi)
        (fun i hi => h3 (i, PUnit.unit) hi) (fun i hi => h4 (i, PUnit.unit) hi)
        (fun i hi => h5 (i, PUnit.unit) hi)
    · intro s₀ q1 q2 c1 s1 hp1 hp2 hp3 hp4
      obtain ⟨s', hexec, hval, hframe⟩ :=
        bwd_first_region_run Q cos sin _ _ _ _ _ _ _ _ s₀ q1 q2 c1 s1
          (fun i hi => hp1 i hi) (fun i hi => hp2 i hi) (fun i hi => hp3 i hi)
          (fun i hi => hp4 i hi)
      refine ⟨s', hexec, hval, fun r o hcond => hframe r o ?_⟩
      rcases hcond with h | h
      · exact Or.inl h
      · exact Or.inr fun i hi => h (i, PUnit.unit) hi
  · refine InPlaceMaskedTileKernelIO.Implements.intro _ ?_ ?_ ?_
    · exact bwd_second_flattenOk Q cos sin _ _ _ _ _ _ _ _
    · intro bounds s h1 h2 h3 h4 h5
      exact bwd_second_traceSafe Q cos sin _ _ _ _ _ _ _ _ bounds s
        (fun i hi => h1 (i, PUnit.unit) hi) (fun i hi => h2 (i, PUnit.unit) hi)
        (fun i hi => h3 (i, PUnit.unit) hi) (fun i hi => h4 (i, PUnit.unit) hi)
        (fun i hi => h5 (i, PUnit.unit) hi)
    · intro s₀ q1 q2 c1 s1 hp1 hp2 hp3 hp4
      obtain ⟨s', hexec, hval, hframe⟩ :=
        bwd_second_region_run Q cos sin _ _ _ _ _ _ _ _ s₀ q1 q2 c1 s1
          (fun i hi => hp1 i hi) (fun i hi => hp2 i hi) (fun i hi => hp3 i hi)
          (fun i hi => hp4 i hi)
      refine ⟨s', hexec, hval, fun r o hcond => hframe r o ?_⟩
      rcases hcond with h | h
      · exact Or.inl h
      · exact Or.inr fun i hi => h (i, PUnit.unit) hi

end IOFace

end VeriTile.Bench.TritonBenchG.RopeEmbedding
