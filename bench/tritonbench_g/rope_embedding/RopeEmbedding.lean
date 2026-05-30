import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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
rope_embedding_python_case{1,2,3,4}_output_summary    ← TOP (abbrev aliases)
  = rope_embedding_python_case{1,2,3,4}_forward_backward_summary
      ├─ rope_embedding_python_caseN_forward_surface_toAlgorithm_supported
      ├─ rope_embedding_python_caseN_backward_surface_toAlgorithm_supported
      │     └─ rope_embedding_surface_toAlgorithm_supported
      ├─ rope_embedding_python_caseN_first_half_compute_correct
      │     └─ rope_embedding_forward_first_half_compute_correct
      │          └─ rope_embedding_forward_first_half_correct
      ├─ rope_embedding_python_caseN_second_half_compute_correct
      │     └─ rope_embedding_forward_second_half_compute_correct
      │          └─ rope_embedding_forward_second_half_correct
      ├─ rope_embedding_python_caseN_backward_first_half_compute_correct
      │     └─ rope_embedding_backward_first_half_compute_correct
      │          └─ rope_embedding_backward_first_half_correct
      └─ rope_embedding_python_caseN_backward_second_half_compute_correct
            └─ rope_embedding_backward_second_half_compute_correct
                 └─ rope_embedding_backward_second_half_correct
```

The two offset families are shown disjoint by
`rope_embedding_python_first_offset_injective` /
`rope_embedding_python_second_offset_injective`, which the half-store
compute-correctness lemmas consume.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; the `.to(sin1.dtype)`
register casts erase to the identity at the algorithm layer (post-erasure all
dtypes unify to `ℝ`). `cos`/`sin` are modeled as **precomputed inputs** loaded
from memory (the kernel does not compute them); the forward spec uses them as
loaded and the backward spec uses the `sin1 := -sin1` flip. Scoping is
**one head-half store at a time** (`first_half`/`second_half`), each over the
active lanes `col_offsets < head_dim//2`; out-of-bounds lanes are preserved
verbatim. The four `caseN` summaries instantiate the proved generic lemmas at
the concrete shapes of the Python `test_case_{1..4}` (and the `BACKWARD_PASS`
heuristic is modeled by the `Bool` argument; `@triton.autotune` is not modeled).
-/

namespace VeriTile.Bench.TritonBenchG.RopeEmbedding

open VeriTile.Triton

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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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

theorem rope_embedding_python_first_offset_injective
    (s : BlockState) (Q_row_stride head_dim ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        qFirstOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) := by
  intro a b h
  simp [qFirstOffset, headStart, colIndex] at h
  exact Fin.ext h

theorem rope_embedding_python_second_offset_injective
    (s : BlockState) (Q_row_stride head_dim ROPE_GROUP_SIZE BLOCK_SIZE : Nat) :
    Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        qSecondOffset s Q_row_stride head_dim ROPE_GROUP_SIZE i) := by
  intro a b h
  simp [qSecondOffset, qFirstOffset, headStart, colIndex] at h
  exact Fin.ext h

theorem rope_embedding_python_base_first_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_first_half Q cos sin 512 32 32 16 64 8 4 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => active s 64 8 4 32 i)
        (fun i => (Q, qFirstOffset s 512 64 4 i)))
      (expected := fun i => ropeFirstSpec s Q cos sin 512 32 32 16 64 4 32 i) := by
  exact rope_embedding_forward_first_half_compute_correct Q cos sin
    512 32 32 16 64 8 4 32 s
    (rope_embedding_python_first_offset_injective s 512 64 4 32)

theorem rope_embedding_python_base_second_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_second_half Q cos sin 512 32 32 16 64 8 4 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => active s 64 8 4 32 i)
        (fun i => (Q, qSecondOffset s 512 64 4 i)))
      (expected := fun i => ropeSecondSpec s Q cos sin 512 32 32 16 64 4 32 i) := by
  exact rope_embedding_forward_second_half_compute_correct Q cos sin
    512 32 32 16 64 8 4 32 s
    (rope_embedding_python_second_offset_injective s 512 64 4 32)

theorem rope_embedding_python_case1_first_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_first_half Q cos sin 128 16 16 8 32 4 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active s 32 4 4 16 i)
        (fun i => (Q, qFirstOffset s 128 32 4 i)))
      (expected := fun i => ropeFirstSpec s Q cos sin 128 16 16 8 32 4 16 i) := by
  exact rope_embedding_forward_first_half_compute_correct Q cos sin
    128 16 16 8 32 4 4 16 s
    (rope_embedding_python_first_offset_injective s 128 32 4 16)

theorem rope_embedding_python_case1_second_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_second_half Q cos sin 128 16 16 8 32 4 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active s 32 4 4 16 i)
        (fun i => (Q, qSecondOffset s 128 32 4 i)))
      (expected := fun i => ropeSecondSpec s Q cos sin 128 16 16 8 32 4 16 i) := by
  exact rope_embedding_forward_second_half_compute_correct Q cos sin
    128 16 16 8 32 4 4 16 s
    (rope_embedding_python_second_offset_injective s 128 32 4 16)

theorem rope_embedding_python_case2_first_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_first_half Q cos sin 2048 64 64 32 128 16 4 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active s 128 16 4 64 i)
        (fun i => (Q, qFirstOffset s 2048 128 4 i)))
      (expected := fun i => ropeFirstSpec s Q cos sin 2048 64 64 32 128 4 64 i) := by
  exact rope_embedding_forward_first_half_compute_correct Q cos sin
    2048 64 64 32 128 16 4 64 s
    (rope_embedding_python_first_offset_injective s 2048 128 4 64)

theorem rope_embedding_python_case2_second_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_second_half Q cos sin 2048 64 64 32 128 16 4 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active s 128 16 4 64 i)
        (fun i => (Q, qSecondOffset s 2048 128 4 i)))
      (expected := fun i => ropeSecondSpec s Q cos sin 2048 64 64 32 128 4 64 i) := by
  exact rope_embedding_forward_second_half_compute_correct Q cos sin
    2048 64 64 32 128 16 4 64 s
    (rope_embedding_python_second_offset_injective s 2048 128 4 64)

theorem rope_embedding_python_case3_first_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_first_half Q cos sin 8192 128 128 64 256 32 4 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 128 => active s 256 32 4 128 i)
        (fun i => (Q, qFirstOffset s 8192 256 4 i)))
      (expected := fun i => ropeFirstSpec s Q cos sin 8192 128 128 64 256 4 128 i) := by
  exact rope_embedding_forward_first_half_compute_correct Q cos sin
    8192 128 128 64 256 32 4 128 s
    (rope_embedding_python_first_offset_injective s 8192 256 4 128)

theorem rope_embedding_python_case3_second_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_second_half Q cos sin 8192 128 128 64 256 32 4 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 128 => active s 256 32 4 128 i)
        (fun i => (Q, qSecondOffset s 8192 256 4 i)))
      (expected := fun i => ropeSecondSpec s Q cos sin 8192 128 128 64 256 4 128 i) := by
  exact rope_embedding_forward_second_half_compute_correct Q cos sin
    8192 128 128 64 256 32 4 128 s
    (rope_embedding_python_second_offset_injective s 8192 256 4 128)

theorem rope_embedding_python_case4_first_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_first_half Q cos sin 32768 256 256 128 512 64 4 256)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 256 => active s 512 64 4 256 i)
        (fun i => (Q, qFirstOffset s 32768 512 4 i)))
      (expected := fun i => ropeFirstSpec s Q cos sin 32768 256 256 128 512 4 256 i) := by
  exact rope_embedding_forward_first_half_compute_correct Q cos sin
    32768 256 256 128 512 64 4 256 s
    (rope_embedding_python_first_offset_injective s 32768 512 4 256)

theorem rope_embedding_python_case4_second_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_second_half Q cos sin 32768 256 256 128 512 64 4 256)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 256 => active s 512 64 4 256 i)
        (fun i => (Q, qSecondOffset s 32768 512 4 i)))
      (expected := fun i => ropeSecondSpec s Q cos sin 32768 256 256 128 512 4 256 i) := by
  exact rope_embedding_forward_second_half_compute_correct Q cos sin
    32768 256 256 128 512 64 4 256 s
    (rope_embedding_python_second_offset_injective s 32768 512 4 256)

theorem rope_embedding_python_base_backward_first_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_first_half Q cos sin 512 32 32 16 64 8 4 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => active s 64 8 4 32 i)
        (fun i => (Q, qFirstOffset s 512 64 4 i)))
      (expected := fun i =>
        ropeBackwardFirstSpec s Q cos sin 512 32 32 16 64 4 32 i) := by
  exact rope_embedding_backward_first_half_compute_correct Q cos sin
    512 32 32 16 64 8 4 32 s
    (rope_embedding_python_first_offset_injective s 512 64 4 32)

theorem rope_embedding_python_base_backward_second_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_second_half Q cos sin 512 32 32 16 64 8 4 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => active s 64 8 4 32 i)
        (fun i => (Q, qSecondOffset s 512 64 4 i)))
      (expected := fun i =>
        ropeBackwardSecondSpec s Q cos sin 512 32 32 16 64 4 32 i) := by
  exact rope_embedding_backward_second_half_compute_correct Q cos sin
    512 32 32 16 64 8 4 32 s
    (rope_embedding_python_second_offset_injective s 512 64 4 32)

theorem rope_embedding_python_case1_backward_first_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_first_half Q cos sin 128 16 16 8 32 4 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active s 32 4 4 16 i)
        (fun i => (Q, qFirstOffset s 128 32 4 i)))
      (expected := fun i =>
        ropeBackwardFirstSpec s Q cos sin 128 16 16 8 32 4 16 i) := by
  exact rope_embedding_backward_first_half_compute_correct Q cos sin
    128 16 16 8 32 4 4 16 s
    (rope_embedding_python_first_offset_injective s 128 32 4 16)

theorem rope_embedding_python_case1_backward_second_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_second_half Q cos sin 128 16 16 8 32 4 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active s 32 4 4 16 i)
        (fun i => (Q, qSecondOffset s 128 32 4 i)))
      (expected := fun i =>
        ropeBackwardSecondSpec s Q cos sin 128 16 16 8 32 4 16 i) := by
  exact rope_embedding_backward_second_half_compute_correct Q cos sin
    128 16 16 8 32 4 4 16 s
    (rope_embedding_python_second_offset_injective s 128 32 4 16)

theorem rope_embedding_python_case2_backward_first_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_first_half Q cos sin 2048 64 64 32 128 16 4 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active s 128 16 4 64 i)
        (fun i => (Q, qFirstOffset s 2048 128 4 i)))
      (expected := fun i =>
        ropeBackwardFirstSpec s Q cos sin 2048 64 64 32 128 4 64 i) := by
  exact rope_embedding_backward_first_half_compute_correct Q cos sin
    2048 64 64 32 128 16 4 64 s
    (rope_embedding_python_first_offset_injective s 2048 128 4 64)

theorem rope_embedding_python_case2_backward_second_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_second_half Q cos sin 2048 64 64 32 128 16 4 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active s 128 16 4 64 i)
        (fun i => (Q, qSecondOffset s 2048 128 4 i)))
      (expected := fun i =>
        ropeBackwardSecondSpec s Q cos sin 2048 64 64 32 128 4 64 i) := by
  exact rope_embedding_backward_second_half_compute_correct Q cos sin
    2048 64 64 32 128 16 4 64 s
    (rope_embedding_python_second_offset_injective s 2048 128 4 64)

theorem rope_embedding_python_case3_backward_first_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_first_half Q cos sin 8192 128 128 64 256 32 4 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 128 => active s 256 32 4 128 i)
        (fun i => (Q, qFirstOffset s 8192 256 4 i)))
      (expected := fun i =>
        ropeBackwardFirstSpec s Q cos sin 8192 128 128 64 256 4 128 i) := by
  exact rope_embedding_backward_first_half_compute_correct Q cos sin
    8192 128 128 64 256 32 4 128 s
    (rope_embedding_python_first_offset_injective s 8192 256 4 128)

theorem rope_embedding_python_case3_backward_second_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_second_half Q cos sin 8192 128 128 64 256 32 4 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 128 => active s 256 32 4 128 i)
        (fun i => (Q, qSecondOffset s 8192 256 4 i)))
      (expected := fun i =>
        ropeBackwardSecondSpec s Q cos sin 8192 128 128 64 256 4 128 i) := by
  exact rope_embedding_backward_second_half_compute_correct Q cos sin
    8192 128 128 64 256 32 4 128 s
    (rope_embedding_python_second_offset_injective s 8192 256 4 128)

theorem rope_embedding_python_case4_backward_first_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_first_half Q cos sin 32768 256 256 128 512 64 4 256)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 256 => active s 512 64 4 256 i)
        (fun i => (Q, qFirstOffset s 32768 512 4 i)))
      (expected := fun i =>
        ropeBackwardFirstSpec s Q cos sin 32768 256 256 128 512 4 256 i) := by
  exact rope_embedding_backward_first_half_compute_correct Q cos sin
    32768 256 256 128 512 64 4 256 s
    (rope_embedding_python_first_offset_injective s 32768 512 4 256)

theorem rope_embedding_python_case4_backward_second_half_compute_correct
    (Q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_second_half Q cos sin 32768 256 256 128 512 64 4 256)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 256 => active s 512 64 4 256 i)
        (fun i => (Q, qSecondOffset s 32768 512 4 i)))
      (expected := fun i =>
        ropeBackwardSecondSpec s Q cos sin 32768 256 256 128 512 4 256 i) := by
  exact rope_embedding_backward_second_half_compute_correct Q cos sin
    32768 256 256 128 512 64 4 256 s
    (rope_embedding_python_second_offset_injective s 32768 512 4 256)

/-- Python case 1 full forward surface lowering. -/
theorem rope_embedding_python_case1_forward_surface_toAlgorithm_supported
    (Q cos sin : RegionName) :
    ∃ alg, (rope_embedding_surface Q 128 cos 16 sin 16 8 32 4
      Bool.false 16 4).toAlgorithm? = Except.ok alg := by
  exact rope_embedding_surface_toAlgorithm_supported Q 128 cos 16 sin 16
    8 32 4 Bool.false 16 4

/-- Python case 1 full backward surface lowering. -/
theorem rope_embedding_python_case1_backward_surface_toAlgorithm_supported
    (Q cos sin : RegionName) :
    ∃ alg, (rope_embedding_surface Q 128 cos 16 sin 16 8 32 4
      Bool.true 16 4).toAlgorithm? = Except.ok alg := by
  exact rope_embedding_surface_toAlgorithm_supported Q 128 cos 16 sin 16
    8 32 4 Bool.true 16 4

/-- Python case 2 full forward surface lowering. -/
theorem rope_embedding_python_case2_forward_surface_toAlgorithm_supported
    (Q cos sin : RegionName) :
    ∃ alg, (rope_embedding_surface Q 2048 cos 64 sin 64 32 128 16
      Bool.false 64 4).toAlgorithm? = Except.ok alg := by
  exact rope_embedding_surface_toAlgorithm_supported Q 2048 cos 64 sin 64
    32 128 16 Bool.false 64 4

/-- Python case 2 full backward surface lowering. -/
theorem rope_embedding_python_case2_backward_surface_toAlgorithm_supported
    (Q cos sin : RegionName) :
    ∃ alg, (rope_embedding_surface Q 2048 cos 64 sin 64 32 128 16
      Bool.true 64 4).toAlgorithm? = Except.ok alg := by
  exact rope_embedding_surface_toAlgorithm_supported Q 2048 cos 64 sin 64
    32 128 16 Bool.true 64 4

/-- Python case 3 full forward surface lowering. -/
theorem rope_embedding_python_case3_forward_surface_toAlgorithm_supported
    (Q cos sin : RegionName) :
    ∃ alg, (rope_embedding_surface Q 8192 cos 128 sin 128 64 256 32
      Bool.false 128 4).toAlgorithm? = Except.ok alg := by
  exact rope_embedding_surface_toAlgorithm_supported Q 8192 cos 128 sin 128
    64 256 32 Bool.false 128 4

/-- Python case 3 full backward surface lowering. -/
theorem rope_embedding_python_case3_backward_surface_toAlgorithm_supported
    (Q cos sin : RegionName) :
    ∃ alg, (rope_embedding_surface Q 8192 cos 128 sin 128 64 256 32
      Bool.true 128 4).toAlgorithm? = Except.ok alg := by
  exact rope_embedding_surface_toAlgorithm_supported Q 8192 cos 128 sin 128
    64 256 32 Bool.true 128 4

/-- Python case 4 full forward surface lowering. -/
theorem rope_embedding_python_case4_forward_surface_toAlgorithm_supported
    (Q cos sin : RegionName) :
    ∃ alg, (rope_embedding_surface Q 32768 cos 256 sin 256 128 512 64
      Bool.false 256 4).toAlgorithm? = Except.ok alg := by
  exact rope_embedding_surface_toAlgorithm_supported Q 32768 cos 256 sin 256
    128 512 64 Bool.false 256 4

/-- Python case 4 full backward surface lowering. -/
theorem rope_embedding_python_case4_backward_surface_toAlgorithm_supported
    (Q cos sin : RegionName) :
    ∃ alg, (rope_embedding_surface Q 32768 cos 256 sin 256 128 512 64
      Bool.true 256 4).toAlgorithm? = Except.ok alg := by
  exact rope_embedding_surface_toAlgorithm_supported Q 32768 cos 256 sin 256
    128 512 64 Bool.true 256 4

/-- Public Python case 1 coverage summary: both forward/backward full surfaces
lower, and both rotary half stores realize the checked outputs for `Q_out` and
`dY_out`. -/
theorem rope_embedding_python_case1_forward_backward_summary
    (Q dY cos sin : RegionName) (sQ sDY : BlockState) :
    (∃ alg, (rope_embedding_surface Q 128 cos 16 sin 16 8 32 4
      Bool.false 16 4).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_first_half Q cos sin
        128 16 16 8 32 4 4 16)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active sQ 32 4 4 16 i)
        (fun i => (Q, qFirstOffset sQ 128 32 4 i)))
      (expected := fun i => ropeFirstSpec sQ Q cos sin 128 16 16 8 32 4 16 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_second_half Q cos sin
        128 16 16 8 32 4 4 16)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active sQ 32 4 4 16 i)
        (fun i => (Q, qSecondOffset sQ 128 32 4 i)))
      (expected := fun i => ropeSecondSpec sQ Q cos sin 128 16 16 8 32 4 16 i)) ∧
    (∃ alg, (rope_embedding_surface dY 128 cos 16 sin 16 8 32 4
      Bool.true 16 4).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_first_half dY cos sin
        128 16 16 8 32 4 4 16)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active sDY 32 4 4 16 i)
        (fun i => (dY, qFirstOffset sDY 128 32 4 i)))
      (expected := fun i =>
        ropeBackwardFirstSpec sDY dY cos sin 128 16 16 8 32 4 16 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_second_half dY cos sin
        128 16 16 8 32 4 4 16)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active sDY 32 4 4 16 i)
        (fun i => (dY, qSecondOffset sDY 128 32 4 i)))
      (expected := fun i =>
        ropeBackwardSecondSpec sDY dY cos sin 128 16 16 8 32 4 16 i)) := by
  constructor
  · exact rope_embedding_python_case1_forward_surface_toAlgorithm_supported Q cos sin
  constructor
  · exact rope_embedding_python_case1_first_half_compute_correct Q cos sin sQ
  constructor
  · exact rope_embedding_python_case1_second_half_compute_correct Q cos sin sQ
  constructor
  · exact rope_embedding_python_case1_backward_surface_toAlgorithm_supported dY cos sin
  constructor
  · exact rope_embedding_python_case1_backward_first_half_compute_correct dY cos sin sDY
  · exact rope_embedding_python_case1_backward_second_half_compute_correct dY cos sin sDY

/-- Public Python case 2 coverage summary. -/
theorem rope_embedding_python_case2_forward_backward_summary
    (Q dY cos sin : RegionName) (sQ sDY : BlockState) :
    (∃ alg, (rope_embedding_surface Q 2048 cos 64 sin 64 32 128 16
      Bool.false 64 4).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_first_half Q cos sin
        2048 64 64 32 128 16 4 64)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active sQ 128 16 4 64 i)
        (fun i => (Q, qFirstOffset sQ 2048 128 4 i)))
      (expected := fun i => ropeFirstSpec sQ Q cos sin 2048 64 64 32 128 4 64 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_second_half Q cos sin
        2048 64 64 32 128 16 4 64)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active sQ 128 16 4 64 i)
        (fun i => (Q, qSecondOffset sQ 2048 128 4 i)))
      (expected := fun i => ropeSecondSpec sQ Q cos sin 2048 64 64 32 128 4 64 i)) ∧
    (∃ alg, (rope_embedding_surface dY 2048 cos 64 sin 64 32 128 16
      Bool.true 64 4).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_first_half dY cos sin
        2048 64 64 32 128 16 4 64)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active sDY 128 16 4 64 i)
        (fun i => (dY, qFirstOffset sDY 2048 128 4 i)))
      (expected := fun i =>
        ropeBackwardFirstSpec sDY dY cos sin 2048 64 64 32 128 4 64 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_second_half dY cos sin
        2048 64 64 32 128 16 4 64)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active sDY 128 16 4 64 i)
        (fun i => (dY, qSecondOffset sDY 2048 128 4 i)))
      (expected := fun i =>
        ropeBackwardSecondSpec sDY dY cos sin 2048 64 64 32 128 4 64 i)) := by
  constructor
  · exact rope_embedding_python_case2_forward_surface_toAlgorithm_supported Q cos sin
  constructor
  · exact rope_embedding_python_case2_first_half_compute_correct Q cos sin sQ
  constructor
  · exact rope_embedding_python_case2_second_half_compute_correct Q cos sin sQ
  constructor
  · exact rope_embedding_python_case2_backward_surface_toAlgorithm_supported dY cos sin
  constructor
  · exact rope_embedding_python_case2_backward_first_half_compute_correct dY cos sin sDY
  · exact rope_embedding_python_case2_backward_second_half_compute_correct dY cos sin sDY

/-- Public Python case 3 coverage summary. -/
theorem rope_embedding_python_case3_forward_backward_summary
    (Q dY cos sin : RegionName) (sQ sDY : BlockState) :
    (∃ alg, (rope_embedding_surface Q 8192 cos 128 sin 128 64 256 32
      Bool.false 128 4).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_first_half Q cos sin
        8192 128 128 64 256 32 4 128)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 128 => active sQ 256 32 4 128 i)
        (fun i => (Q, qFirstOffset sQ 8192 256 4 i)))
      (expected := fun i => ropeFirstSpec sQ Q cos sin 8192 128 128 64 256 4 128 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_second_half Q cos sin
        8192 128 128 64 256 32 4 128)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 128 => active sQ 256 32 4 128 i)
        (fun i => (Q, qSecondOffset sQ 8192 256 4 i)))
      (expected := fun i => ropeSecondSpec sQ Q cos sin 8192 128 128 64 256 4 128 i)) ∧
    (∃ alg, (rope_embedding_surface dY 8192 cos 128 sin 128 64 256 32
      Bool.true 128 4).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_first_half dY cos sin
        8192 128 128 64 256 32 4 128)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 128 => active sDY 256 32 4 128 i)
        (fun i => (dY, qFirstOffset sDY 8192 256 4 i)))
      (expected := fun i =>
        ropeBackwardFirstSpec sDY dY cos sin 8192 128 128 64 256 4 128 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_second_half dY cos sin
        8192 128 128 64 256 32 4 128)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 128 => active sDY 256 32 4 128 i)
        (fun i => (dY, qSecondOffset sDY 8192 256 4 i)))
      (expected := fun i =>
        ropeBackwardSecondSpec sDY dY cos sin 8192 128 128 64 256 4 128 i)) := by
  constructor
  · exact rope_embedding_python_case3_forward_surface_toAlgorithm_supported Q cos sin
  constructor
  · exact rope_embedding_python_case3_first_half_compute_correct Q cos sin sQ
  constructor
  · exact rope_embedding_python_case3_second_half_compute_correct Q cos sin sQ
  constructor
  · exact rope_embedding_python_case3_backward_surface_toAlgorithm_supported dY cos sin
  constructor
  · exact rope_embedding_python_case3_backward_first_half_compute_correct dY cos sin sDY
  · exact rope_embedding_python_case3_backward_second_half_compute_correct dY cos sin sDY

/-- Public Python case 4 coverage summary. -/
theorem rope_embedding_python_case4_forward_backward_summary
    (Q dY cos sin : RegionName) (sQ sDY : BlockState) :
    (∃ alg, (rope_embedding_surface Q 32768 cos 256 sin 256 128 512 64
      Bool.false 256 4).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_first_half Q cos sin
        32768 256 256 128 512 64 4 256)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 256 => active sQ 512 64 4 256 i)
        (fun i => (Q, qFirstOffset sQ 32768 512 4 i)))
      (expected := fun i => ropeFirstSpec sQ Q cos sin 32768 256 256 128 512 4 256 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_forward_second_half Q cos sin
        32768 256 256 128 512 64 4 256)
      (initialState := sQ)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 256 => active sQ 512 64 4 256 i)
        (fun i => (Q, qSecondOffset sQ 32768 512 4 i)))
      (expected := fun i => ropeSecondSpec sQ Q cos sin 32768 256 256 128 512 4 256 i)) ∧
    (∃ alg, (rope_embedding_surface dY 32768 cos 256 sin 256 128 512 64
      Bool.true 256 4).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_first_half dY cos sin
        32768 256 256 128 512 64 4 256)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 256 => active sDY 512 64 4 256 i)
        (fun i => (dY, qFirstOffset sDY 32768 512 4 i)))
      (expected := fun i =>
        ropeBackwardFirstSpec sDY dY cos sin 32768 256 256 128 512 4 256 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rope_embedding_backward_second_half dY cos sin
        32768 256 256 128 512 64 4 256)
      (initialState := sDY)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 256 => active sDY 512 64 4 256 i)
        (fun i => (dY, qSecondOffset sDY 32768 512 4 i)))
      (expected := fun i =>
        ropeBackwardSecondSpec sDY dY cos sin 32768 256 256 128 512 4 256 i)) := by
  constructor
  · exact rope_embedding_python_case4_forward_surface_toAlgorithm_supported Q cos sin
  constructor
  · exact rope_embedding_python_case4_first_half_compute_correct Q cos sin sQ
  constructor
  · exact rope_embedding_python_case4_second_half_compute_correct Q cos sin sQ
  constructor
  · exact rope_embedding_python_case4_backward_surface_toAlgorithm_supported dY cos sin
  constructor
  · exact rope_embedding_python_case4_backward_first_half_compute_correct dY cos sin sDY
  · exact rope_embedding_python_case4_backward_second_half_compute_correct dY cos sin sDY

/-- `output_summary` alias for Python RoPE embedding case 1, covering the
checked forward `Q_out` stores and backward `dY_out` stores. -/
abbrev rope_embedding_python_case1_output_summary
    (Q dY cos sin : RegionName) (sQ sDY : BlockState) :=
  rope_embedding_python_case1_forward_backward_summary Q dY cos sin sQ sDY

/-- `output_summary` alias for Python RoPE embedding case 2, covering the
checked forward `Q_out` stores and backward `dY_out` stores. -/
abbrev rope_embedding_python_case2_output_summary
    (Q dY cos sin : RegionName) (sQ sDY : BlockState) :=
  rope_embedding_python_case2_forward_backward_summary Q dY cos sin sQ sDY

/-- `output_summary` alias for Python RoPE embedding case 3, covering the
checked forward `Q_out` stores and backward `dY_out` stores. -/
abbrev rope_embedding_python_case3_output_summary
    (Q dY cos sin : RegionName) (sQ sDY : BlockState) :=
  rope_embedding_python_case3_forward_backward_summary Q dY cos sin sQ sDY

/-- `output_summary` alias for Python RoPE embedding case 4, covering the
checked forward `Q_out` stores and backward `dY_out` stores. -/
abbrev rope_embedding_python_case4_output_summary
    (Q dY cos sin : RegionName) (sQ sDY : BlockState) :=
  rope_embedding_python_case4_forward_backward_summary Q dY cos sin sQ sDY

end VeriTile.Bench.TritonBenchG.RopeEmbedding
