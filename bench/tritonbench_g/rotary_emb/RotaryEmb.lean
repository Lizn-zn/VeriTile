import VeriTile.Triton

/-!
# `rotary_emb` — strict per-kernel correctness

`_rotary_kernel` applies the interleaved rotary position embedding **in place**
to Q and K: each program owns one head-block (`program_id(0)`) and one seq-block
(`program_id(1)`), and over even/odd dimension pairs rewrites
`(q0*cos - q1*sin, q0*sin + q1*cos)` for Q and the analogous pair for K, where
`cos`/`sin` are precomputed inputs loaded per sequence position at the even
(`dim_range0`) dimension lanes. There are **four stores, all in place** — Q even,
Q odd, K even, K odd — no separate output buffer exists.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (the `(head-blocks, seq-blocks)` grid, the
`BLOCK_HEAD`/`BLOCK_SEQ`/`BLOCK_DMODEL` choices, the stride bookkeeping, and how
the runtime composes per-program writes into one buffer) is the *trusted
boundary*, not a proof obligation here. Because the head-block, seq-block, and
per-lane indices are universally quantified, the per-program statement covers
every program of the grid.

## Proof architecture

```
rotary_emb_kernel_correctness            ← TOP SPECIFICATION (four ⊨ headlines)
  ├─ rotary_kernel_surface_toAlgorithm_supported   full Q+K surface lowers
  └─ for each of the four in-place stores  (X ∈ {q0, q1, k0, k1})
       ├─ rotary_emb_X_block_flattenOk    bridge fragment membership
       ├─ rotary_emb_X_block_traceSafe    per-execution lane-wise safety walk
       └─ rotary_emb_X_block_region_run   region-model masked in-place triple
            ├─ rotary_emb_X_block_exec_isSome  termination
            ├─ rotary_emb_X_block_correct      algorithm-layer readback per lane
            └─ rotary_emb_X_block_frame        masked scatter cell frame
```

Each headline is stated on the store's **IO signature** `rotaryStoreIO`
(`GroupedMasked2DKernelIO`, the grouped vector-channel skin: channel counts are
fields, so the four float read windows are one uniformly-indexed table rather
than four named fields). The signature declares, per store: the allocation list
`bufs = [Q/K, Cos, Sin]`; the four `B = BLOCK_HALF`-lane read windows — data at
the even dimension (channel `0`), data at the odd dimension (channel `1`),
`Cos` (channel `2`) and `Sin` (channel `3`) at the even dimension — with their
per-lane gates (`pid₁ < max_total_len ∧ pid₀ < HEAD` for the data channels,
`pid₁ < max_total_len` for `Cos`/`Sin`, exactly the kernel's two `tl.load`
masks); and the single output channel, **in place** on the data buffer itself
(`out 0 = Q` resp. `K`, a channel pointing back at an input region) at the even
or odd dimension window, gated by the store mask.

`⊨` (`GroupedMasked2DKernelIO.Implements`) is the audit-once masked Hoare-triple
combinator: for **every** disjoint flat placement of the three buffers,
**every** program id whose active lanes are in bounds, and **every** launch
state whose four read windows hold `xs`, the translated pointer kernel
terminates, every write-active lane of the data buffer holds its rotary value
computed from the **old** window contents (the standard before/after reading of
an in-place update), and every other memory cell is unchanged.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; dtype casts erase to the
identity at the algorithm layer (post-erasure all dtypes unify to `ℝ`).
`cos`/`sin` are modeled as **precomputed inputs** loaded from memory, not
computed. Scoping is **one block-store at a time** (q0/q1/k0/k1): each
proof-facing slice kernel performs a single store, so no even-vs-odd
disjointness is needed — only *within-family* offset injectivity
(`dim ↦ dim · stride_d` injective on the store's own lane family), which enters
as a hypothesis of the top specification. The full `_rotary_kernel` surface
(`rotary_kernel_surface`, and the intermediate Q-only/K-only two-store faces
`rotary_emb_q_surface`/`rotary_emb_k_surface`) is transcribed and shown to lower
to the algorithm layer; its four-stores-at-once value story would additionally
need cross-family disjointness and is not claimed here. All statements quantify
over arbitrary symbolic strides, sequence length, head counts, and block sizes.
-/

namespace VeriTile.Bench.TritonBenchG.RotaryEmb

open VeriTile.Triton
open scoped VeriTile.Triton.GroupedMasked2DKernelIO

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

/-- The full rotary Q/K surface lowers to the algorithm layer. -/
theorem rotary_kernel_surface_toAlgorithm_supported
    (Q K Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len
      HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ BLOCK_DMODEL : Nat) :
    ∃ alg, (rotary_kernel_surface Q K Cos Sin stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ
      BLOCK_DMODEL).toAlgorithm? = Except.ok alg := by
  simp [rotary_kernel_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

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

/-- The standalone Q rotary surface lowers to the algorithm layer. -/
theorem rotary_emb_q_surface_toAlgorithm_supported
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    ∃ alg, (rotary_emb_q_surface Q Cos Sin stride_qbs stride_qh stride_qd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q
      BLOCK_HALF).toAlgorithm? = Except.ok alg := by
  simp [rotary_emb_q_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

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

/-- The standalone K rotary surface lowers to the algorithm layer. -/
theorem rotary_emb_k_surface_toAlgorithm_supported
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat) :
    ∃ alg, (rotary_emb_k_surface K Cos Sin stride_kbs stride_kh stride_kd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len HEAD_K
      BLOCK_HALF).toAlgorithm? = Except.ok alg := by
  simp [rotary_emb_k_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

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
  · simp [exec, rotary_emb_q0_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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
  · simp [exec, rotary_emb_q1_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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
  · simp [exec, rotary_emb_k0_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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
  · simp [exec, rotary_emb_k1_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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


/-! ### The `⊨` specifications -/

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for the masked store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

/-- The shared **IO signature** of one in-place rotary store — the whole
kernel-specific audit surface of the four `⊨` headlines below.

* `bufs = [Buf, Cos, Sin]` — the allocation list; `Buf` is the data buffer
  (`Q` or `K`), and the single output channel points **back at it**
  (`out 0 = Buf`): the rotary update is in place, so the triple reads the
  *old* window contents into `xs` and asserts the *new* ones.
* `nIn = 4`, `nOut = 1`, `B = BLOCK_HALF` — four `BLOCK_HALF`-lane read
  windows, one write window.
* read channel `0` — the data buffer at the **even** dimension
  `pid₁·stride_bs + pid₀·stride_h + (2j)·stride_d`;
  read channel `1` — the data buffer at the **odd** dimension
  `… + (2j+1)·stride_d`;
  read channels `2`/`3` — `Cos`/`Sin` at `pid₁·stride_cosbs + (2j)·stride_cosd`
  resp. `pid₁·stride_sinbs + (2j)·stride_sind` (the kernel loads both
  trigonometric tiles at the even dimension lanes).
* `readMask` — `pid₁ < max_total_len ∧ pid₀ < HEAD` on the two data channels
  and `pid₁ < max_total_len` on `Cos`/`Sin`: exactly the kernel's two
  `tl.load` masks.
* `write`/`writeMask` — the data buffer at `storeDim j` (instantiated with
  `dimEven` for the even store and `dimOdd` for the odd one), gated by
  `pid₁ < max_total_len ∧ pid₀ < HEAD`, the kernel's `tl.store` mask.

The windows and masks are declared, not parsed from the kernel; the headlines
**prove** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def rotaryStoreIO (kernel : ComputeKernel) (Buf Cos Sin : RegionName)
    (stride_bs stride_h stride_d stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD BLOCK_HALF : Nat)
    (storeDim : Fin BLOCK_HALF → Nat) : GroupedMasked2DKernelIO where
  kernel := kernel
  nIn := 4
  nOut := 1
  bufs := [Buf, Cos, Sin]
  inp := fun i => match i with
    | ⟨0, _⟩ => Buf
    | ⟨1, _⟩ => Buf
    | ⟨2, _⟩ => Cos
    | ⟨_ + 3, _⟩ => Sin
  out := fun _ => Buf
  B := BLOCK_HALF
  read := fun i pid₀ pid₁ j => match i with
    | ⟨0, _⟩ => pid₁ * stride_bs + pid₀ * stride_h + dimEven j * stride_d
    | ⟨1, _⟩ => pid₁ * stride_bs + pid₀ * stride_h + dimOdd j * stride_d
    | ⟨2, _⟩ => pid₁ * stride_cosbs + dimEven j * stride_cosd
    | ⟨_ + 3, _⟩ => pid₁ * stride_sinbs + dimEven j * stride_sind
  readMask := fun i pid₀ pid₁ _ => match i with
    | ⟨0, _⟩ => pid₁ < max_total_len ∧ pid₀ < HEAD
    | ⟨1, _⟩ => pid₁ < max_total_len ∧ pid₀ < HEAD
    | ⟨2, _⟩ => pid₁ < max_total_len
    | ⟨_ + 3, _⟩ => pid₁ < max_total_len
  write := fun _ pid₀ pid₁ j =>
    pid₁ * stride_bs + pid₀ * stride_h + storeDim j * stride_d
  writeMask := fun _ pid₀ pid₁ _ => pid₁ < max_total_len ∧ pid₀ < HEAD

/-! #### `rotary_emb_q0_block` -/

/-- Termination: the Q even-dimension slice executes to completion from any state. -/
private theorem rotary_emb_q0_block_exec_isSome
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat)
    (s : BlockState) :
    ∃ s1, exec ((rotary_emb_q0_block Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF).toAlgKernel) s
      = some s1 := by
  simp [exec, rotary_emb_q0_block, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt]

/-- Frame half: every memory cell the Q even-dimension masked in-place store does not
actively hit — every cell of every region other than `Q`, and the lanes of
the output window itself that are inactive — is preserved by the run. -/
private theorem rotary_emb_q0_block_frame
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat)
    (s s1 : BlockState)
    (hExec : exec ((rotary_emb_q0_block Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF).toAlgKernel) s
      = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ j : Fin BLOCK_HALF, active s max_total_len HEAD_Q →
      ¬(Q = r ∧ qOffset s stride_qbs stride_qh stride_qd (dimEven j) = o)) :
    s1.mem r o = s.mem r o := by
  simp only [active, qOffset, seqIndex, headIndex, dimEven] at hmiss
  simp [exec, rotary_emb_q0_block, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 hmk hc

/-- Per-execution safety walk for the Q even-dimension slice: one computational unfold
walks all statements — the program ids, the `arange`, the even/odd dimension
arithmetic and the register rotary combination are memory-silent — and reduces
the four masked loads (data at the even and odd dimensions, `Cos`/`Sin` at the
even dimension) and the masked store to the lane-wise bounds hypotheses. -/
theorem rotary_emb_q0_block_traceSafe
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ j : Fin BLOCK_HALF, active s max_total_len HEAD_Q →
      qOffset s stride_qbs stride_qh stride_qd (dimEven j) < bounds Q)
    (h2 : ∀ j : Fin BLOCK_HALF, active s max_total_len HEAD_Q →
      qOffset s stride_qbs stride_qh stride_qd (dimOdd j) < bounds Q)
    (h3 : ∀ j : Fin BLOCK_HALF, seqIndex s < max_total_len →
      cosOffset s stride_cosbs stride_cosd (dimEven j) < bounds Cos)
    (h4 : ∀ j : Fin BLOCK_HALF, seqIndex s < max_total_len →
      sinOffset s stride_sinbs stride_sind (dimEven j) < bounds Sin)
    (h5 : ∀ j : Fin BLOCK_HALF, active s max_total_len HEAD_Q →
      qOffset s stride_qbs stride_qh stride_qd (dimEven j) < bounds Q) :
    Kernel.TraceSafe bounds
      ((rotary_emb_q0_block Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_Q
        BLOCK_HALF).toAlgKernel) s := by
  simp only [active, qOffset, cosOffset, sinOffset, seqIndex, headIndex,
    dimEven, dimOdd] at h1 h2 h3 h4 h5
  unfold Kernel.TraceSafe
  simp [rotary_emb_q0_block, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active,
    BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]
  exact ⟨fun a hs hh => h1 a ⟨hs, hh⟩, fun a hs hh => h2 a ⟨hs, hh⟩,
    fun a hs => h3 a hs, fun a hs => h4 a hs, fun a hs hh => h5 a ⟨hs, hh⟩⟩

/-- The Q even-dimension slice sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, masked loads, one masked in-place store). -/
theorem rotary_emb_q0_block_flattenOk
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    ((rotary_emb_q0_block Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_Q
        BLOCK_HALF).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [rotary_emb_q0_block, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

/-- **The region-model masked in-place Hoare triple** for the Q even-dimension slice —
termination, write-active-lane output values, and frame off the write-active
output lanes, from any launch state whose four read windows hold `xs`. This is
the `hrun` obligation of the `⊨` headline; the value half reuses
`rotary_emb_q0_block_correct` (the kernel reads the whole even/odd pair before the in-place
scatter, so the before/after reading is sound). -/
theorem rotary_emb_q0_block_region_run
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat)
    (s₀ : BlockState) (xs : Fin 4 → Fin BLOCK_HALF → ℝ)
    (hinj : Function.Injective (fun i : Fin BLOCK_HALF => dimEven i * stride_qd))
    (hx0 : ∀ j : Fin BLOCK_HALF, active s₀ max_total_len HEAD_Q →
      s₀.readMem Q (qOffset s₀ stride_qbs stride_qh stride_qd (dimEven j))
        = xs (⟨0, by decide⟩ : Fin 4) j)
    (hx1 : ∀ j : Fin BLOCK_HALF, active s₀ max_total_len HEAD_Q →
      s₀.readMem Q (qOffset s₀ stride_qbs stride_qh stride_qd (dimOdd j))
        = xs (⟨1, by decide⟩ : Fin 4) j)
    (hx2 : ∀ j : Fin BLOCK_HALF, seqIndex s₀ < max_total_len →
      s₀.readMem Cos (cosOffset s₀ stride_cosbs stride_cosd (dimEven j))
        = xs (⟨2, by decide⟩ : Fin 4) j)
    (hx3 : ∀ j : Fin BLOCK_HALF, seqIndex s₀ < max_total_len →
      s₀.readMem Sin (sinOffset s₀ stride_sinbs stride_sind (dimEven j))
        = xs (⟨3, by decide⟩ : Fin 4) j) :
    ∃ s1, exec ((rotary_emb_q0_block Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
          stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF).toAlgKernel)
        s₀ = some s1
      ∧ (∀ j : Fin BLOCK_HALF, active s₀ max_total_len HEAD_Q →
          s1.readMem Q (qOffset s₀ stride_qbs stride_qh stride_qd (dimEven j)) = xs (⟨0, by decide⟩ : Fin 4) j * xs (⟨2, by decide⟩ : Fin 4) j -
              xs (⟨1, by decide⟩ : Fin 4) j * xs (⟨3, by decide⟩ : Fin 4) j)
      ∧ (∀ r o',
          (∀ j : Fin BLOCK_HALF, active s₀ max_total_len HEAD_Q →
            r ≠ Q ∨ o' ≠ qOffset s₀ stride_qbs stride_qh stride_qd (dimEven j)) →
          s1.mem r o' = s₀.mem r o') := by
  obtain ⟨s1, hs1⟩ := rotary_emb_q0_block_exec_isSome Q Cos Sin stride_qbs stride_qh stride_qd
    stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q
    BLOCK_HALF s₀
  have hinj' : Function.Injective
      (fun i : Fin BLOCK_HALF => qOffset s₀ stride_qbs stride_qh stride_qd (dimEven i)) := by
    intro a b hab
    simp only [qOffset] at hab
    exact hinj (Nat.add_left_cancel hab)
  refine ⟨s1, hs1, fun j hj => ?_, fun r o' hc => ?_⟩
  · have h := rotary_emb_q0_block_correct Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs
      stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF
      s₀ s1 hinj' hs1 j
    rw [if_pos hj] at h
    rw [h]
    simp only [rotaryQ0Spec]
    rw [hx0 j hj, hx1 j hj, hx2 j hj.1, hx3 j hj.1]
  · exact rotary_emb_q0_block_frame Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
      stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF s₀ s1 hs1 r o'
      (fun j hj hcc =>
        (hc j hj).elim (fun h => h hcc.1.symm) (fun h => h hcc.2.symm))

/-- The Q even-dimension store's IO signature. -/
def rotaryQ0IO (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    GroupedMasked2DKernelIO :=
  rotaryStoreIO (rotary_emb_q0_block Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
      stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF)
    Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
    stride_sind max_total_len HEAD_Q BLOCK_HALF dimEven

theorem rotary_emb_q0_block_implements
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat)
    (hinj : Function.Injective (fun i : Fin BLOCK_HALF => dimEven i * stride_qd)) :
    rotaryQ0IO Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
        stride_sind max_total_len HEAD_Q BLOCK_HALF
      ⊨ fun _ _ xs _ j =>
          let dataEven := xs (⟨0, by decide⟩ : Fin 4) j
          let dataOdd := xs (⟨1, by decide⟩ : Fin 4) j
          let cosLane := xs (⟨2, by decide⟩ : Fin 4) j
          let sinLane := xs (⟨3, by decide⟩ : Fin 4) j
          dataEven * cosLane - dataOdd * sinLane := by
  refine GroupedMasked2DKernelIO.Implements.intro _ ?_ ?_ ?_ ?_
  · intro o
    simp [rotaryQ0IO, rotaryStoreIO]
  · exact rotary_emb_q0_block_flattenOk Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs
      stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF
  · intro bounds s hib hob
    exact rotary_emb_q0_block_traceSafe Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs
      stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF
      bounds s
      (fun j hj => hib (⟨0, by decide⟩ : Fin 4) j hj)
      (fun j hj => hib (⟨1, by decide⟩ : Fin 4) j hj)
      (fun j hj => hib (⟨2, by decide⟩ : Fin 4) j hj)
      (fun j hj => hib (⟨3, by decide⟩ : Fin 4) j hj)
      (fun j hj => hob (⟨0, by decide⟩ : Fin 1) j hj)
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hfr⟩ :=
      rotary_emb_q0_block_region_run Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF s₀ xs hinj
        (fun j hj => hx (⟨0, by decide⟩ : Fin 4) j hj)
        (fun j hj => hx (⟨1, by decide⟩ : Fin 4) j hj)
        (fun j hj => hx (⟨2, by decide⟩ : Fin 4) j hj)
        (fun j hj => hx (⟨3, by decide⟩ : Fin 4) j hj)
    exact ⟨s1, hexec, fun _ j hj => hval j hj,
      fun r o' hcond => hfr r o' (fun j hj => hcond (⟨0, by decide⟩ : Fin 1) j hj)⟩

/-! #### `rotary_emb_q1_block` -/

/-- Termination: the Q odd-dimension slice executes to completion from any state. -/
private theorem rotary_emb_q1_block_exec_isSome
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat)
    (s : BlockState) :
    ∃ s1, exec ((rotary_emb_q1_block Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF).toAlgKernel) s
      = some s1 := by
  simp [exec, rotary_emb_q1_block, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt]

/-- Frame half: every memory cell the Q odd-dimension masked in-place store does not
actively hit — every cell of every region other than `Q`, and the lanes of
the output window itself that are inactive — is preserved by the run. -/
private theorem rotary_emb_q1_block_frame
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat)
    (s s1 : BlockState)
    (hExec : exec ((rotary_emb_q1_block Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF).toAlgKernel) s
      = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ j : Fin BLOCK_HALF, active s max_total_len HEAD_Q →
      ¬(Q = r ∧ qOffset s stride_qbs stride_qh stride_qd (dimOdd j) = o)) :
    s1.mem r o = s.mem r o := by
  simp only [active, qOffset, seqIndex, headIndex, dimOdd] at hmiss
  simp [exec, rotary_emb_q1_block, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 hmk hc

/-- Per-execution safety walk for the Q odd-dimension slice: one computational unfold
walks all statements — the program ids, the `arange`, the even/odd dimension
arithmetic and the register rotary combination are memory-silent — and reduces
the four masked loads (data at the even and odd dimensions, `Cos`/`Sin` at the
even dimension) and the masked store to the lane-wise bounds hypotheses. -/
theorem rotary_emb_q1_block_traceSafe
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ j : Fin BLOCK_HALF, active s max_total_len HEAD_Q →
      qOffset s stride_qbs stride_qh stride_qd (dimEven j) < bounds Q)
    (h2 : ∀ j : Fin BLOCK_HALF, active s max_total_len HEAD_Q →
      qOffset s stride_qbs stride_qh stride_qd (dimOdd j) < bounds Q)
    (h3 : ∀ j : Fin BLOCK_HALF, seqIndex s < max_total_len →
      cosOffset s stride_cosbs stride_cosd (dimEven j) < bounds Cos)
    (h4 : ∀ j : Fin BLOCK_HALF, seqIndex s < max_total_len →
      sinOffset s stride_sinbs stride_sind (dimEven j) < bounds Sin)
    (h5 : ∀ j : Fin BLOCK_HALF, active s max_total_len HEAD_Q →
      qOffset s stride_qbs stride_qh stride_qd (dimOdd j) < bounds Q) :
    Kernel.TraceSafe bounds
      ((rotary_emb_q1_block Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_Q
        BLOCK_HALF).toAlgKernel) s := by
  simp only [active, qOffset, cosOffset, sinOffset, seqIndex, headIndex,
    dimEven, dimOdd] at h1 h2 h3 h4 h5
  unfold Kernel.TraceSafe
  simp [rotary_emb_q1_block, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active,
    BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]
  exact ⟨fun a hs hh => h1 a ⟨hs, hh⟩, fun a hs hh => h2 a ⟨hs, hh⟩,
    fun a hs => h3 a hs, fun a hs => h4 a hs, fun a hs hh => h5 a ⟨hs, hh⟩⟩

/-- The Q odd-dimension slice sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, masked loads, one masked in-place store). -/
theorem rotary_emb_q1_block_flattenOk
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    ((rotary_emb_q1_block Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_Q
        BLOCK_HALF).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [rotary_emb_q1_block, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

/-- **The region-model masked in-place Hoare triple** for the Q odd-dimension slice —
termination, write-active-lane output values, and frame off the write-active
output lanes, from any launch state whose four read windows hold `xs`. This is
the `hrun` obligation of the `⊨` headline; the value half reuses
`rotary_emb_q1_block_correct` (the kernel reads the whole even/odd pair before the in-place
scatter, so the before/after reading is sound). -/
theorem rotary_emb_q1_block_region_run
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat)
    (s₀ : BlockState) (xs : Fin 4 → Fin BLOCK_HALF → ℝ)
    (hinj : Function.Injective (fun i : Fin BLOCK_HALF => dimOdd i * stride_qd))
    (hx0 : ∀ j : Fin BLOCK_HALF, active s₀ max_total_len HEAD_Q →
      s₀.readMem Q (qOffset s₀ stride_qbs stride_qh stride_qd (dimEven j))
        = xs (⟨0, by decide⟩ : Fin 4) j)
    (hx1 : ∀ j : Fin BLOCK_HALF, active s₀ max_total_len HEAD_Q →
      s₀.readMem Q (qOffset s₀ stride_qbs stride_qh stride_qd (dimOdd j))
        = xs (⟨1, by decide⟩ : Fin 4) j)
    (hx2 : ∀ j : Fin BLOCK_HALF, seqIndex s₀ < max_total_len →
      s₀.readMem Cos (cosOffset s₀ stride_cosbs stride_cosd (dimEven j))
        = xs (⟨2, by decide⟩ : Fin 4) j)
    (hx3 : ∀ j : Fin BLOCK_HALF, seqIndex s₀ < max_total_len →
      s₀.readMem Sin (sinOffset s₀ stride_sinbs stride_sind (dimEven j))
        = xs (⟨3, by decide⟩ : Fin 4) j) :
    ∃ s1, exec ((rotary_emb_q1_block Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
          stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF).toAlgKernel)
        s₀ = some s1
      ∧ (∀ j : Fin BLOCK_HALF, active s₀ max_total_len HEAD_Q →
          s1.readMem Q (qOffset s₀ stride_qbs stride_qh stride_qd (dimOdd j)) = xs (⟨0, by decide⟩ : Fin 4) j * xs (⟨3, by decide⟩ : Fin 4) j +
              xs (⟨1, by decide⟩ : Fin 4) j * xs (⟨2, by decide⟩ : Fin 4) j)
      ∧ (∀ r o',
          (∀ j : Fin BLOCK_HALF, active s₀ max_total_len HEAD_Q →
            r ≠ Q ∨ o' ≠ qOffset s₀ stride_qbs stride_qh stride_qd (dimOdd j)) →
          s1.mem r o' = s₀.mem r o') := by
  obtain ⟨s1, hs1⟩ := rotary_emb_q1_block_exec_isSome Q Cos Sin stride_qbs stride_qh stride_qd
    stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q
    BLOCK_HALF s₀
  have hinj' : Function.Injective
      (fun i : Fin BLOCK_HALF => qOffset s₀ stride_qbs stride_qh stride_qd (dimOdd i)) := by
    intro a b hab
    simp only [qOffset] at hab
    exact hinj (Nat.add_left_cancel hab)
  refine ⟨s1, hs1, fun j hj => ?_, fun r o' hc => ?_⟩
  · have h := rotary_emb_q1_block_correct Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs
      stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF
      s₀ s1 hinj' hs1 j
    rw [if_pos hj] at h
    rw [h]
    simp only [rotaryQ1Spec]
    rw [hx0 j hj, hx1 j hj, hx2 j hj.1, hx3 j hj.1]
  · exact rotary_emb_q1_block_frame Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
      stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF s₀ s1 hs1 r o'
      (fun j hj hcc =>
        (hc j hj).elim (fun h => h hcc.1.symm) (fun h => h hcc.2.symm))

/-- The Q odd-dimension store's IO signature. -/
def rotaryQ1IO (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    GroupedMasked2DKernelIO :=
  rotaryStoreIO (rotary_emb_q1_block Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
      stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF)
    Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
    stride_sind max_total_len HEAD_Q BLOCK_HALF dimOdd

theorem rotary_emb_q1_block_implements
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat)
    (hinj : Function.Injective (fun i : Fin BLOCK_HALF => dimOdd i * stride_qd)) :
    rotaryQ1IO Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
        stride_sind max_total_len HEAD_Q BLOCK_HALF
      ⊨ fun _ _ xs _ j =>
          let dataEven := xs (⟨0, by decide⟩ : Fin 4) j
          let dataOdd := xs (⟨1, by decide⟩ : Fin 4) j
          let cosLane := xs (⟨2, by decide⟩ : Fin 4) j
          let sinLane := xs (⟨3, by decide⟩ : Fin 4) j
          dataEven * sinLane + dataOdd * cosLane := by
  refine GroupedMasked2DKernelIO.Implements.intro _ ?_ ?_ ?_ ?_
  · intro o
    simp [rotaryQ1IO, rotaryStoreIO]
  · exact rotary_emb_q1_block_flattenOk Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs
      stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF
  · intro bounds s hib hob
    exact rotary_emb_q1_block_traceSafe Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs
      stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF
      bounds s
      (fun j hj => hib (⟨0, by decide⟩ : Fin 4) j hj)
      (fun j hj => hib (⟨1, by decide⟩ : Fin 4) j hj)
      (fun j hj => hib (⟨2, by decide⟩ : Fin 4) j hj)
      (fun j hj => hib (⟨3, by decide⟩ : Fin 4) j hj)
      (fun j hj => hob (⟨0, by decide⟩ : Fin 1) j hj)
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hfr⟩ :=
      rotary_emb_q1_block_region_run Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF s₀ xs hinj
        (fun j hj => hx (⟨0, by decide⟩ : Fin 4) j hj)
        (fun j hj => hx (⟨1, by decide⟩ : Fin 4) j hj)
        (fun j hj => hx (⟨2, by decide⟩ : Fin 4) j hj)
        (fun j hj => hx (⟨3, by decide⟩ : Fin 4) j hj)
    exact ⟨s1, hexec, fun _ j hj => hval j hj,
      fun r o' hcond => hfr r o' (fun j hj => hcond (⟨0, by decide⟩ : Fin 1) j hj)⟩

/-! #### `rotary_emb_k0_block` -/

/-- Termination: the K even-dimension slice executes to completion from any state. -/
private theorem rotary_emb_k0_block_exec_isSome
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat)
    (s : BlockState) :
    ∃ s1, exec ((rotary_emb_k0_block K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF).toAlgKernel) s
      = some s1 := by
  simp [exec, rotary_emb_k0_block, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt]

/-- Frame half: every memory cell the K even-dimension masked in-place store does not
actively hit — every cell of every region other than `K`, and the lanes of
the output window itself that are inactive — is preserved by the run. -/
private theorem rotary_emb_k0_block_frame
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat)
    (s s1 : BlockState)
    (hExec : exec ((rotary_emb_k0_block K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF).toAlgKernel) s
      = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ j : Fin BLOCK_HALF, activeK s max_total_len HEAD_K →
      ¬(K = r ∧ kOffset s stride_kbs stride_kh stride_kd (dimEven j) = o)) :
    s1.mem r o = s.mem r o := by
  simp only [activeK, kOffset, seqIndex, headIndex, dimEven] at hmiss
  simp [exec, rotary_emb_k0_block, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 hmk hc

/-- Per-execution safety walk for the K even-dimension slice: one computational unfold
walks all statements — the program ids, the `arange`, the even/odd dimension
arithmetic and the register rotary combination are memory-silent — and reduces
the four masked loads (data at the even and odd dimensions, `Cos`/`Sin` at the
even dimension) and the masked store to the lane-wise bounds hypotheses. -/
theorem rotary_emb_k0_block_traceSafe
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ j : Fin BLOCK_HALF, activeK s max_total_len HEAD_K →
      kOffset s stride_kbs stride_kh stride_kd (dimEven j) < bounds K)
    (h2 : ∀ j : Fin BLOCK_HALF, activeK s max_total_len HEAD_K →
      kOffset s stride_kbs stride_kh stride_kd (dimOdd j) < bounds K)
    (h3 : ∀ j : Fin BLOCK_HALF, seqIndex s < max_total_len →
      cosOffset s stride_cosbs stride_cosd (dimEven j) < bounds Cos)
    (h4 : ∀ j : Fin BLOCK_HALF, seqIndex s < max_total_len →
      sinOffset s stride_sinbs stride_sind (dimEven j) < bounds Sin)
    (h5 : ∀ j : Fin BLOCK_HALF, activeK s max_total_len HEAD_K →
      kOffset s stride_kbs stride_kh stride_kd (dimEven j) < bounds K) :
    Kernel.TraceSafe bounds
      ((rotary_emb_k0_block K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_K
        BLOCK_HALF).toAlgKernel) s := by
  simp only [activeK, kOffset, cosOffset, sinOffset, seqIndex, headIndex,
    dimEven, dimOdd] at h1 h2 h3 h4 h5
  unfold Kernel.TraceSafe
  simp [rotary_emb_k0_block, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active,
    BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]
  exact ⟨fun a hs hh => h1 a ⟨hs, hh⟩, fun a hs hh => h2 a ⟨hs, hh⟩,
    fun a hs => h3 a hs, fun a hs => h4 a hs, fun a hs hh => h5 a ⟨hs, hh⟩⟩

/-- The K even-dimension slice sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, masked loads, one masked in-place store). -/
theorem rotary_emb_k0_block_flattenOk
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat) :
    ((rotary_emb_k0_block K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_K
        BLOCK_HALF).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [rotary_emb_k0_block, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

/-- **The region-model masked in-place Hoare triple** for the K even-dimension slice —
termination, write-active-lane output values, and frame off the write-active
output lanes, from any launch state whose four read windows hold `xs`. This is
the `hrun` obligation of the `⊨` headline; the value half reuses
`rotary_emb_k0_block_correct` (the kernel reads the whole even/odd pair before the in-place
scatter, so the before/after reading is sound). -/
theorem rotary_emb_k0_block_region_run
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat)
    (s₀ : BlockState) (xs : Fin 4 → Fin BLOCK_HALF → ℝ)
    (hinj : Function.Injective (fun i : Fin BLOCK_HALF => dimEven i * stride_kd))
    (hx0 : ∀ j : Fin BLOCK_HALF, activeK s₀ max_total_len HEAD_K →
      s₀.readMem K (kOffset s₀ stride_kbs stride_kh stride_kd (dimEven j))
        = xs (⟨0, by decide⟩ : Fin 4) j)
    (hx1 : ∀ j : Fin BLOCK_HALF, activeK s₀ max_total_len HEAD_K →
      s₀.readMem K (kOffset s₀ stride_kbs stride_kh stride_kd (dimOdd j))
        = xs (⟨1, by decide⟩ : Fin 4) j)
    (hx2 : ∀ j : Fin BLOCK_HALF, seqIndex s₀ < max_total_len →
      s₀.readMem Cos (cosOffset s₀ stride_cosbs stride_cosd (dimEven j))
        = xs (⟨2, by decide⟩ : Fin 4) j)
    (hx3 : ∀ j : Fin BLOCK_HALF, seqIndex s₀ < max_total_len →
      s₀.readMem Sin (sinOffset s₀ stride_sinbs stride_sind (dimEven j))
        = xs (⟨3, by decide⟩ : Fin 4) j) :
    ∃ s1, exec ((rotary_emb_k0_block K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
          stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF).toAlgKernel)
        s₀ = some s1
      ∧ (∀ j : Fin BLOCK_HALF, activeK s₀ max_total_len HEAD_K →
          s1.readMem K (kOffset s₀ stride_kbs stride_kh stride_kd (dimEven j)) = xs (⟨0, by decide⟩ : Fin 4) j * xs (⟨2, by decide⟩ : Fin 4) j -
              xs (⟨1, by decide⟩ : Fin 4) j * xs (⟨3, by decide⟩ : Fin 4) j)
      ∧ (∀ r o',
          (∀ j : Fin BLOCK_HALF, activeK s₀ max_total_len HEAD_K →
            r ≠ K ∨ o' ≠ kOffset s₀ stride_kbs stride_kh stride_kd (dimEven j)) →
          s1.mem r o' = s₀.mem r o') := by
  obtain ⟨s1, hs1⟩ := rotary_emb_k0_block_exec_isSome K Cos Sin stride_kbs stride_kh stride_kd
    stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len HEAD_K
    BLOCK_HALF s₀
  have hinj' : Function.Injective
      (fun i : Fin BLOCK_HALF => kOffset s₀ stride_kbs stride_kh stride_kd (dimEven i)) := by
    intro a b hab
    simp only [kOffset] at hab
    exact hinj (Nat.add_left_cancel hab)
  refine ⟨s1, hs1, fun j hj => ?_, fun r o' hc => ?_⟩
  · have h := rotary_emb_k0_block_correct K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs
      stride_cosd stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF
      s₀ s1 hinj' hs1 j
    rw [if_pos hj] at h
    rw [h]
    simp only [rotaryK0Spec]
    rw [hx0 j hj, hx1 j hj, hx2 j hj.1, hx3 j hj.1]
  · exact rotary_emb_k0_block_frame K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
      stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF s₀ s1 hs1 r o'
      (fun j hj hcc =>
        (hc j hj).elim (fun h => h hcc.1.symm) (fun h => h hcc.2.symm))

/-- The K even-dimension store's IO signature. -/
def rotaryK0IO (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat) :
    GroupedMasked2DKernelIO :=
  rotaryStoreIO (rotary_emb_k0_block K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
      stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF)
    K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
    stride_sind max_total_len HEAD_K BLOCK_HALF dimEven

theorem rotary_emb_k0_block_implements
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat)
    (hinj : Function.Injective (fun i : Fin BLOCK_HALF => dimEven i * stride_kd)) :
    rotaryK0IO K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
        stride_sind max_total_len HEAD_K BLOCK_HALF
      ⊨ fun _ _ xs _ j =>
          let dataEven := xs (⟨0, by decide⟩ : Fin 4) j
          let dataOdd := xs (⟨1, by decide⟩ : Fin 4) j
          let cosLane := xs (⟨2, by decide⟩ : Fin 4) j
          let sinLane := xs (⟨3, by decide⟩ : Fin 4) j
          dataEven * cosLane - dataOdd * sinLane := by
  refine GroupedMasked2DKernelIO.Implements.intro _ ?_ ?_ ?_ ?_
  · intro o
    simp [rotaryK0IO, rotaryStoreIO]
  · exact rotary_emb_k0_block_flattenOk K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs
      stride_cosd stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF
  · intro bounds s hib hob
    exact rotary_emb_k0_block_traceSafe K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs
      stride_cosd stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF
      bounds s
      (fun j hj => hib (⟨0, by decide⟩ : Fin 4) j hj)
      (fun j hj => hib (⟨1, by decide⟩ : Fin 4) j hj)
      (fun j hj => hib (⟨2, by decide⟩ : Fin 4) j hj)
      (fun j hj => hib (⟨3, by decide⟩ : Fin 4) j hj)
      (fun j hj => hob (⟨0, by decide⟩ : Fin 1) j hj)
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hfr⟩ :=
      rotary_emb_k0_block_region_run K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF s₀ xs hinj
        (fun j hj => hx (⟨0, by decide⟩ : Fin 4) j hj)
        (fun j hj => hx (⟨1, by decide⟩ : Fin 4) j hj)
        (fun j hj => hx (⟨2, by decide⟩ : Fin 4) j hj)
        (fun j hj => hx (⟨3, by decide⟩ : Fin 4) j hj)
    exact ⟨s1, hexec, fun _ j hj => hval j hj,
      fun r o' hcond => hfr r o' (fun j hj => hcond (⟨0, by decide⟩ : Fin 1) j hj)⟩

/-! #### `rotary_emb_k1_block` -/

/-- Termination: the K odd-dimension slice executes to completion from any state. -/
private theorem rotary_emb_k1_block_exec_isSome
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat)
    (s : BlockState) :
    ∃ s1, exec ((rotary_emb_k1_block K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF).toAlgKernel) s
      = some s1 := by
  simp [exec, rotary_emb_k1_block, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt]

/-- Frame half: every memory cell the K odd-dimension masked in-place store does not
actively hit — every cell of every region other than `K`, and the lanes of
the output window itself that are inactive — is preserved by the run. -/
private theorem rotary_emb_k1_block_frame
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat)
    (s s1 : BlockState)
    (hExec : exec ((rotary_emb_k1_block K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF).toAlgKernel) s
      = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ j : Fin BLOCK_HALF, activeK s max_total_len HEAD_K →
      ¬(K = r ∧ kOffset s stride_kbs stride_kh stride_kd (dimOdd j) = o)) :
    s1.mem r o = s.mem r o := by
  simp only [activeK, kOffset, seqIndex, headIndex, dimOdd] at hmiss
  simp [exec, rotary_emb_k1_block, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 hmk hc

/-- Per-execution safety walk for the K odd-dimension slice: one computational unfold
walks all statements — the program ids, the `arange`, the even/odd dimension
arithmetic and the register rotary combination are memory-silent — and reduces
the four masked loads (data at the even and odd dimensions, `Cos`/`Sin` at the
even dimension) and the masked store to the lane-wise bounds hypotheses. -/
theorem rotary_emb_k1_block_traceSafe
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ j : Fin BLOCK_HALF, activeK s max_total_len HEAD_K →
      kOffset s stride_kbs stride_kh stride_kd (dimEven j) < bounds K)
    (h2 : ∀ j : Fin BLOCK_HALF, activeK s max_total_len HEAD_K →
      kOffset s stride_kbs stride_kh stride_kd (dimOdd j) < bounds K)
    (h3 : ∀ j : Fin BLOCK_HALF, seqIndex s < max_total_len →
      cosOffset s stride_cosbs stride_cosd (dimEven j) < bounds Cos)
    (h4 : ∀ j : Fin BLOCK_HALF, seqIndex s < max_total_len →
      sinOffset s stride_sinbs stride_sind (dimEven j) < bounds Sin)
    (h5 : ∀ j : Fin BLOCK_HALF, activeK s max_total_len HEAD_K →
      kOffset s stride_kbs stride_kh stride_kd (dimOdd j) < bounds K) :
    Kernel.TraceSafe bounds
      ((rotary_emb_k1_block K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_K
        BLOCK_HALF).toAlgKernel) s := by
  simp only [activeK, kOffset, cosOffset, sinOffset, seqIndex, headIndex,
    dimEven, dimOdd] at h1 h2 h3 h4 h5
  unfold Kernel.TraceSafe
  simp [rotary_emb_k1_block, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active,
    BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]
  exact ⟨fun a hs hh => h1 a ⟨hs, hh⟩, fun a hs hh => h2 a ⟨hs, hh⟩,
    fun a hs => h3 a hs, fun a hs => h4 a hs, fun a hs hh => h5 a ⟨hs, hh⟩⟩

/-- The K odd-dimension slice sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, masked loads, one masked in-place store). -/
theorem rotary_emb_k1_block_flattenOk
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat) :
    ((rotary_emb_k1_block K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_K
        BLOCK_HALF).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [rotary_emb_k1_block, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

/-- **The region-model masked in-place Hoare triple** for the K odd-dimension slice —
termination, write-active-lane output values, and frame off the write-active
output lanes, from any launch state whose four read windows hold `xs`. This is
the `hrun` obligation of the `⊨` headline; the value half reuses
`rotary_emb_k1_block_correct` (the kernel reads the whole even/odd pair before the in-place
scatter, so the before/after reading is sound). -/
theorem rotary_emb_k1_block_region_run
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat)
    (s₀ : BlockState) (xs : Fin 4 → Fin BLOCK_HALF → ℝ)
    (hinj : Function.Injective (fun i : Fin BLOCK_HALF => dimOdd i * stride_kd))
    (hx0 : ∀ j : Fin BLOCK_HALF, activeK s₀ max_total_len HEAD_K →
      s₀.readMem K (kOffset s₀ stride_kbs stride_kh stride_kd (dimEven j))
        = xs (⟨0, by decide⟩ : Fin 4) j)
    (hx1 : ∀ j : Fin BLOCK_HALF, activeK s₀ max_total_len HEAD_K →
      s₀.readMem K (kOffset s₀ stride_kbs stride_kh stride_kd (dimOdd j))
        = xs (⟨1, by decide⟩ : Fin 4) j)
    (hx2 : ∀ j : Fin BLOCK_HALF, seqIndex s₀ < max_total_len →
      s₀.readMem Cos (cosOffset s₀ stride_cosbs stride_cosd (dimEven j))
        = xs (⟨2, by decide⟩ : Fin 4) j)
    (hx3 : ∀ j : Fin BLOCK_HALF, seqIndex s₀ < max_total_len →
      s₀.readMem Sin (sinOffset s₀ stride_sinbs stride_sind (dimEven j))
        = xs (⟨3, by decide⟩ : Fin 4) j) :
    ∃ s1, exec ((rotary_emb_k1_block K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
          stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF).toAlgKernel)
        s₀ = some s1
      ∧ (∀ j : Fin BLOCK_HALF, activeK s₀ max_total_len HEAD_K →
          s1.readMem K (kOffset s₀ stride_kbs stride_kh stride_kd (dimOdd j)) = xs (⟨0, by decide⟩ : Fin 4) j * xs (⟨3, by decide⟩ : Fin 4) j +
              xs (⟨1, by decide⟩ : Fin 4) j * xs (⟨2, by decide⟩ : Fin 4) j)
      ∧ (∀ r o',
          (∀ j : Fin BLOCK_HALF, activeK s₀ max_total_len HEAD_K →
            r ≠ K ∨ o' ≠ kOffset s₀ stride_kbs stride_kh stride_kd (dimOdd j)) →
          s1.mem r o' = s₀.mem r o') := by
  obtain ⟨s1, hs1⟩ := rotary_emb_k1_block_exec_isSome K Cos Sin stride_kbs stride_kh stride_kd
    stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len HEAD_K
    BLOCK_HALF s₀
  have hinj' : Function.Injective
      (fun i : Fin BLOCK_HALF => kOffset s₀ stride_kbs stride_kh stride_kd (dimOdd i)) := by
    intro a b hab
    simp only [kOffset] at hab
    exact hinj (Nat.add_left_cancel hab)
  refine ⟨s1, hs1, fun j hj => ?_, fun r o' hc => ?_⟩
  · have h := rotary_emb_k1_block_correct K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs
      stride_cosd stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF
      s₀ s1 hinj' hs1 j
    rw [if_pos hj] at h
    rw [h]
    simp only [rotaryK1Spec]
    rw [hx0 j hj, hx1 j hj, hx2 j hj.1, hx3 j hj.1]
  · exact rotary_emb_k1_block_frame K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
      stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF s₀ s1 hs1 r o'
      (fun j hj hcc =>
        (hc j hj).elim (fun h => h hcc.1.symm) (fun h => h hcc.2.symm))

/-- The K odd-dimension store's IO signature. -/
def rotaryK1IO (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat) :
    GroupedMasked2DKernelIO :=
  rotaryStoreIO (rotary_emb_k1_block K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
      stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF)
    K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
    stride_sind max_total_len HEAD_K BLOCK_HALF dimOdd

theorem rotary_emb_k1_block_implements
    (K Cos Sin : RegionName)
    (stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_K BLOCK_HALF : Nat)
    (hinj : Function.Injective (fun i : Fin BLOCK_HALF => dimOdd i * stride_kd)) :
    rotaryK1IO K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
        stride_sind max_total_len HEAD_K BLOCK_HALF
      ⊨ fun _ _ xs _ j =>
          let dataEven := xs (⟨0, by decide⟩ : Fin 4) j
          let dataOdd := xs (⟨1, by decide⟩ : Fin 4) j
          let cosLane := xs (⟨2, by decide⟩ : Fin 4) j
          let sinLane := xs (⟨3, by decide⟩ : Fin 4) j
          dataEven * sinLane + dataOdd * cosLane := by
  refine GroupedMasked2DKernelIO.Implements.intro _ ?_ ?_ ?_ ?_
  · intro o
    simp [rotaryK1IO, rotaryStoreIO]
  · exact rotary_emb_k1_block_flattenOk K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs
      stride_cosd stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF
  · intro bounds s hib hob
    exact rotary_emb_k1_block_traceSafe K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs
      stride_cosd stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF
      bounds s
      (fun j hj => hib (⟨0, by decide⟩ : Fin 4) j hj)
      (fun j hj => hib (⟨1, by decide⟩ : Fin 4) j hj)
      (fun j hj => hib (⟨2, by decide⟩ : Fin 4) j hj)
      (fun j hj => hib (⟨3, by decide⟩ : Fin 4) j hj)
      (fun j hj => hob (⟨0, by decide⟩ : Fin 1) j hj)
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hfr⟩ :=
      rotary_emb_k1_block_region_run K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
        stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF s₀ xs hinj
        (fun j hj => hx (⟨0, by decide⟩ : Fin 4) j hj)
        (fun j hj => hx (⟨1, by decide⟩ : Fin 4) j hj)
        (fun j hj => hx (⟨2, by decide⟩ : Fin 4) j hj)
        (fun j hj => hx (⟨3, by decide⟩ : Fin 4) j hj)
    exact ⟨s1, hexec, fun _ j hj => hval j hj,
      fun r o' hcond => hfr r o' (fun j hj => hcond (⟨0, by decide⟩ : Fin 1) j hj)⟩

/-- **The headline for `rotary_emb_fwd`.** For arbitrary symbolic strides,
sequence length, head counts, and block sizes: the full `_rotary_kernel`
surface lowers to the algorithm layer, and each of the four Python-observable
in-place stores (Q even, Q odd, K even, K odd) implements the interleaved
rotary rotation on its grouped IO signature — for every disjoint flat placement
of `[Q/K, Cos, Sin]`, every program id whose active lanes are in bounds, and
every launch state whose four read windows hold `xs`, the translated pointer
kernel terminates, every write-active lane of the data buffer ends up holding
`dataEven·cos - dataOdd·sin` (even store) resp. `dataEven·sin + dataOdd·cos`
(odd store) computed from the **old** window contents, and every other memory
cell is unchanged. The four `dim ↦ dim · stride_d` injectivity hypotheses are
the within-family (not even-vs-odd) collision-freedom of the host layout. -/
specification rotary_emb_kernel_correctness
    (Q K Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len
      HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ BLOCK_DMODEL BLOCK_HALF : Nat)
    (hQEven : Function.Injective
      (fun i : Fin BLOCK_HALF => dimEven i * stride_qd))
    (hQOdd : Function.Injective
      (fun i : Fin BLOCK_HALF => dimOdd i * stride_qd))
    (hKEven : Function.Injective
      (fun i : Fin BLOCK_HALF => dimEven i * stride_kd))
    (hKOdd : Function.Injective
      (fun i : Fin BLOCK_HALF => dimOdd i * stride_kd)) :
    (∃ alg, (rotary_kernel_surface Q K Cos Sin stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ
      BLOCK_DMODEL).toAlgorithm? = Except.ok alg) ∧
    (rotaryQ0IO Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs
        stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF
      ⊨ fun _ _ xs _ j =>
          let dataEven := xs (⟨0, by decide⟩ : Fin 4) j
          let dataOdd := xs (⟨1, by decide⟩ : Fin 4) j
          let cosLane := xs (⟨2, by decide⟩ : Fin 4) j
          let sinLane := xs (⟨3, by decide⟩ : Fin 4) j
          dataEven * cosLane - dataOdd * sinLane) ∧
    (rotaryQ1IO Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs
        stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF
      ⊨ fun _ _ xs _ j =>
          let dataEven := xs (⟨0, by decide⟩ : Fin 4) j
          let dataOdd := xs (⟨1, by decide⟩ : Fin 4) j
          let cosLane := xs (⟨2, by decide⟩ : Fin 4) j
          let sinLane := xs (⟨3, by decide⟩ : Fin 4) j
          dataEven * sinLane + dataOdd * cosLane) ∧
    (rotaryK0IO K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs
        stride_cosd stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF
      ⊨ fun _ _ xs _ j =>
          let dataEven := xs (⟨0, by decide⟩ : Fin 4) j
          let dataOdd := xs (⟨1, by decide⟩ : Fin 4) j
          let cosLane := xs (⟨2, by decide⟩ : Fin 4) j
          let sinLane := xs (⟨3, by decide⟩ : Fin 4) j
          dataEven * cosLane - dataOdd * sinLane) ∧
    (rotaryK1IO K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs
        stride_cosd stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF
      ⊨ fun _ _ xs _ j =>
          let dataEven := xs (⟨0, by decide⟩ : Fin 4) j
          let dataOdd := xs (⟨1, by decide⟩ : Fin 4) j
          let cosLane := xs (⟨2, by decide⟩ : Fin 4) j
          let sinLane := xs (⟨3, by decide⟩ : Fin 4) j
          dataEven * sinLane + dataOdd * cosLane) :=
  ⟨rotary_kernel_surface_toAlgorithm_supported Q K Cos Sin stride_qbs stride_qh
      stride_qd stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
      stride_sinbs stride_sind max_total_len HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ
      BLOCK_DMODEL,
   rotary_emb_q0_block_implements Q Cos Sin stride_qbs stride_qh stride_qd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q
      BLOCK_HALF hQEven,
   rotary_emb_q1_block_implements Q Cos Sin stride_qbs stride_qh stride_qd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q
      BLOCK_HALF hQOdd,
   rotary_emb_k0_block_implements K Cos Sin stride_kbs stride_kh stride_kd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len HEAD_K
      BLOCK_HALF hKEven,
   rotary_emb_k1_block_implements K Cos Sin stride_kbs stride_kh stride_kd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len HEAD_K
      BLOCK_HALF hKOdd⟩

end VeriTile.Bench.TritonBenchG.RotaryEmb
