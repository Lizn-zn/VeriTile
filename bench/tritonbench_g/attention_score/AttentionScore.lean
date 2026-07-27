import VeriTile.Triton

/-!
# `attention_score` — strict per-kernel correctness

`_score_kernel` computes attention *scores* (per-key column sums of the softmax
probabilities), not the attention output. Program `(start_n, off_hz)` fixes one
`BLOCK_N` block of keys for one (batch, head), loads `k`, then loops over the
query blocks (`start_m` by `BLOCK_M`) accumulating `o += sum(p, axis=0)` where
`p = exp2(dot(q,k)·qk_scale - m)` are the precomputed-`M`-normalized softmax
weights, with optional sliding-window/`IS_EVEN_N` masking via `where(mask, p,
0)`. The accumulated column sums `o` are stored to `Out`, masked by
`o_range < NKV_CTX`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_score_kernel[grid](...)`, the grid over
`(cdiv(NKV_CTX, BLOCK_N), Z·H)`, block scheduling, the `@triton.heuristics`
`IS_EVEN_M`/`IS_EVEN_N` selection, and how the runtime composes per-program
writes into one buffer) is the *trusted boundary*, not a proof obligation here.
Because `start_n`/`off_hz` are universally quantified, the per-program statement
covers every program of the grid.

## Genuine closed-form specification (case 1)

In addition to the surface/store coverage below, this file now carries a
**genuine, self-reference-free** closed form for the case-1 score column:
`case1OutClosedForm` (built from `case1RawScore`, `case1Mask`, `case1Weight`)
is the masked-`exp2` query-row column sum
`o[j] = Σ_{c<2} Σ_{i<64} mask(i,j,c)·exp2(sm_scale·log2e·⟨Q_{c·64+i},K_{·,start_n·64+j}⟩ − M[c·64+i])`,
defined directly over the input buffers `Q`/`K`/`M` with no reference to the
kernel's own `exec`. It is derived lane-by-lane from the elaborated `@triton.jit`
body (the nat-truncated `dist`/`mask`, the `exp2`/`where`/`reduceSum` recipe).
The control-flow stepping lemmas (`stepStmt_ifThenElse_true` etc.) that discharge
the lowered `IS_EVEN_M`/`SLIDING_WINDOW`/`IS_EVEN_N` guards are proved
sorry-free.  The full `exec`-side connection is now **complete and sorry-free**:
`attention_score_case1_exec_eq_closedForm` unfolds the elaborated body
(preLoop `score_preLoop_eval` → the 2-iteration `forRangeDyn` loop
`score_loop_eval`, built on the one-block step `score_loopBody_eval` →
post-loop masked store `score_post_eval`) and proves the kernel writes exactly
`case1OutClosedForm` to `Out` at every active column. The public
`attention_score_python_case1_output_summary_general` now states this genuine
closed form — the former self-referential `producedAttentionScoreCase1OutValue`
is removed.

## Proof architecture

The public summary is the dimension-general
`attention_score_python_case1_output_summary_general` (the `case1` compile-flag
configuration: sliding window, non-complement).

```
attention_score_python_case1_output_summary_general          ← GENERAL TOP THEOREM (dimension-parameterized, genuine closed form)
  ├─ attention_score_kernel_toAlgorithm_supported                 surface lowers to algorithm layer
  └─ attention_score_case1_genuine_compute_correct_general        masked Out store = case1OutClosedFormG
       └─ attention_score_case1_exec_eq_closedForm_general       full exec unfold (preLoop+loop+post)

(test-shape building blocks, no public summary root)
  └─ attention_score_case1_genuine_compute_correct (→ attention_score_case1_exec_eq_closedForm)

attention_score_final_store_slice_compute_correct            ← ComputeCorrect over the masked Out store
  └─ attention_score_final_store_slice_correct                ← algorithm-layer readback per lane
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the output-dtype cast
collapses to the identity post-erasure; `@triton.autotune`/`@triton.heuristics`
and `num_warps`/`num_stages` are not modeled. The case 1 main summary is
dimension-general (`attention_score_python_case1_output_summary_general`, symbolic
shape/strides); the Python test shape (`B=2, H=4, N_CTX=NKV_CTX=128, D_MODEL=64,
BLOCK_M=BLOCK_N=64`, contiguous strides) is recovered as the special case by
instantiating the general theorem. `sm_scale` is kept universally quantified. The `Out`
writeback is stated against the genuine `case1OutClosedForm` (the masked-`exp2`
query-row column sum over the input buffers `Q`/`K`/`M`), written at the actual
kernel store offset `case1OutStoreOffset` under the `o_range < NKV_CTX` mask
`case1OutActive`. This is a single-program (single key-block) scope;
cross-program composition into the full score buffer is the trusted host
boundary.
-/

namespace VeriTile.Bench.TritonBenchG.AttentionScore

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `attention_score_python_case1_output_summary_general` -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-- DSL port of `attention_score.py`'s `_score_kernel`.

The proof parameter `hBlockMN` carries the Python wrapper invariant
`BLOCK_M == BLOCK_N` so the DSL can type the source `tl.zeros([BLOCK_M])`
against the later `tl.sum(p, axis=0)` vector. -/
def attention_score_kernel
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh _stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      sliding_window_offset sliding_window_size
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (sm_scale : ℝ)
    (SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N : Bool)
    (_hBlockMN : BLOCK_M = BLOCK_N) :
    ComputeKernel := triton {
  start_n = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  off_hkv = off_h // ($(H) // $(H_KV))
  q_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  k_offset = (off_z).to(tl.int64) * $(stride_kz) + (off_hkv).to(tl.int64) * $(stride_kh)
  m_ptrs = M + off_hz * $(ROUND_CTX) + tl.arange(0, $(BLOCK_M))
  o = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(0, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_offset,
    shape=($(BLOCK_DMODEL), $(NKV_CTX)),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, start_n * $(BLOCK_N)),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  if IS_EVEN_N {
    k = tl.load(K_block_ptr)
  } else {
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  lo = 0
  hi = $(ROUND_CTX)
  qk_scale = $((sm_scale : ℝ))
  qk_scale *= 1.4426950408889634
  for start_m in range(lo, hi, $(BLOCK_M)) {
    start_m = tl.multiple_of(start_m, $(BLOCK_M))
    if IS_EVEN_M {
      q = tl.load(Q_block_ptr)
    } else {
      q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    m = tl.load(m_ptrs)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk * qk_scale
    if SLIDING_WINDOW {
      dist = tl.arange(0, $(BLOCK_M))[:, None] -
        tl.arange(0, $(BLOCK_N))[None, :] + start_m -
        start_n * $(BLOCK_N) + $(sliding_window_offset)
      if COMPLEMENT_SLIDING_WINDOW {
        mask = dist >= $(sliding_window_size)
      } else {
        mask = (dist >= 0) & (dist < $(sliding_window_size))
      }
    }
    qk = qk - m[:, None]
    p = tl.math.exp2(qk)
    if SLIDING_WINDOW {
      p = tl.where(mask, p, 0)
    }
    if not IS_EVEN_N {
      p = tl.where(((tl.arange(0, $(BLOCK_M)) + start_m) < $(N_CTX))[:, None],
        p, 0)
    }
    o += tl.sum(p, axis=0)
    Q_block_ptr = tl.advance(Q_block_ptr, offsets=($(BLOCK_M), 0))
    m_ptrs = m_ptrs + $(BLOCK_M)
  }
  o_offset = (off_z).to(tl.int64) * $(stride_oz) + (off_h).to(tl.int64) * $(stride_oh)
  o_range = tl.arange(0, $(BLOCK_N)) + start_n * $(BLOCK_N)
  o_ptrs = Out + o_offset + o_range
  tl.store(o_ptrs, (o).to(Out.type.element_ty),
    mask=o_range < $(NKV_CTX))
}

/-- The full attention-score surface lowers to the algorithm layer, with the
Python wrapper invariant `BLOCK_M = BLOCK_N` carried as an explicit parameter. -/
theorem attention_score_kernel_toAlgorithm_supported
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX
      sliding_window_offset sliding_window_size
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (sm_scale : ℝ)
    (SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N : Bool)
    (hBlockMN : BLOCK_M = BLOCK_N) :
    ∃ alg, (attention_score_kernel Q K M Out stride_qz stride_qh stride_qm
      stride_qk stride_kz stride_kh stride_kn stride_kk stride_oz stride_oh
      stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX sliding_window_offset
      sliding_window_size BLOCK_M BLOCK_DMODEL BLOCK_N sm_scale SLIDING_WINDOW
      COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N hBlockMN).toAlgorithm?
        = Except.ok alg := by
  simp [attention_score_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented final score-vector store slice of `attention_score.py`'s
`_score_kernel`.

The surface kernel above computes the score accumulator `o` from Q/K blocks,
max values, and optional sliding-window masking. This slice starts after that
reduction with a precomputed `Score` vector and proves the final
`o_range < NKV_CTX` masked writeback into `Out`. -/
def attention_score_final_store_slice
    (Score Out : RegionName)
    (stride_score_z stride_score_h stride_score_n
      stride_oz stride_oh
      NKV_CTX BLOCK_N : Nat) :
    ComputeKernel := triton {
  start_n = tl.program_id(0)
  off_z = tl.program_id(1)
  off_h = tl.program_id(2)
  offs_n = tl.arange(0, $(BLOCK_N))
  o_range = start_n * $(BLOCK_N) + offs_n
  score = tl.load(Score + off_z * $(stride_score_z) + off_h * $(stride_score_h) +
      o_range * $(stride_score_n), mask=o_range < $(NKV_CTX), other=0.0)
  tl.store(Out + off_z * $(stride_oz) + off_h * $(stride_oh) +
      o_range, score, mask=o_range < $(NKV_CTX))
}

def nIndex (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * BLOCK_N + i.val

def active (s : BlockState) (NKV_CTX BLOCK_N : Nat) (i : Fin BLOCK_N) : Prop :=
  nIndex s BLOCK_N i < NKV_CTX

instance activeDecidable (s : BlockState) (NKV_CTX BLOCK_N : Nat) (i : Fin BLOCK_N) :
    Decidable (active s NKV_CTX BLOCK_N i) := by
  unfold active
  infer_instance

def scoreOffset
    (s : BlockState) (stride_score_z stride_score_h stride_score_n BLOCK_N : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * stride_score_z + s.pids 2 * stride_score_h +
    nIndex s BLOCK_N i * stride_score_n

def outOffset
    (s : BlockState) (stride_oz stride_oh BLOCK_N : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * stride_oz + s.pids 2 * stride_oh + nIndex s BLOCK_N i

noncomputable def scoreStoreValue
    (s : BlockState) (Score : RegionName)
    (stride_score_z stride_score_h stride_score_n NKV_CTX BLOCK_N : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (if active s NKV_CTX BLOCK_N i then
      some (s.readMem Score
        (scoreOffset s stride_score_z stride_score_h stride_score_n BLOCK_N i))
    else some (0.0 : ℝ))

/-- Algorithm-layer correctness for the final attention-score store. -/
theorem attention_score_final_store_slice_correct
    (Score Out : RegionName)
    (stride_score_z stride_score_h stride_score_n
      stride_oz stride_oh
      NKV_CTX BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s stride_oz stride_oh BLOCK_N i)) :
    ∀ i : Fin BLOCK_N,
      let outAddr := outOffset s stride_oz stride_oh BLOCK_N i
      (exec (attention_score_final_store_slice Score Out stride_score_z
            stride_score_h stride_score_n stride_oz stride_oh NKV_CTX
            BLOCK_N) s).map (·.readMem Out outAddr)
        = some (if active s NKV_CTX BLOCK_N i then
            scoreStoreValue s Score stride_score_z stride_score_h stride_score_n
              NKV_CTX BLOCK_N i
          else s.readMem Out outAddr) := by
  intro i
  simp [exec, attention_score_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, nIndex, active, scoreOffset,
        outOffset]
  let offsetFn : TileIndex [BLOCK_N] → Nat :=
    fun idx => s.pids 1 * stride_oz + s.pids 2 * stride_oh +
      (s.pids 0 * BLOCK_N + idx.1.val)
  let valueFn : TileIndex [BLOCK_N] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 * BLOCK_N + idx.1.val < NKV_CTX then
          some (s.readMem Score
            (s.pids 1 * stride_score_z + s.pids 2 * stride_score_h +
              (s.pids 0 * BLOCK_N + idx.1.val) * stride_score_n))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_N] → Prop :=
    fun idx => s.pids 0 * BLOCK_N + idx.1.val < NKV_CTX
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s stride_oz stride_oh BLOCK_N a =
        outOffset s stride_oz stride_oh BLOCK_N b := by
      simpa [offsetFn, outOffset, nIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem Out (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BLOCK_N])).readMem Out (offsetFn (i, PUnit.unit)) =
    if active s NKV_CTX BLOCK_N i then
      scoreStoreValue s Score stride_score_z stride_score_h stride_score_n
        NKV_CTX BLOCK_N i
    else s.readMem Out (offsetFn (i, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : P (i, PUnit.unit)
  · rw [if_pos hi]
    simp [valueFn, P, active, scoreStoreValue, scoreOffset, nIndex, hi]
  · rw [if_neg hi]
    simp [P, active, scoreStoreValue, nIndex, hi]

/-- Compute-facing correctness for the final attention-score store. -/
theorem attention_score_final_store_slice_compute_correct
    (Score Out : RegionName)
    (stride_score_z stride_score_h stride_score_n
      stride_oz stride_oh
      NKV_CTX BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s stride_oz stride_oh BLOCK_N i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_score_final_store_slice Score Out stride_score_z
        stride_score_h stride_score_n stride_oz stride_oh NKV_CTX BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => active s NKV_CTX BLOCK_N i)
        (fun i => (Out, outOffset s stride_oz stride_oh BLOCK_N i)))
      (expected := fun i =>
        scoreStoreValue s Score stride_score_z stride_score_h stride_score_n
          NKV_CTX BLOCK_N i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_score_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := attention_score_final_store_slice_correct Score Out stride_score_z
    stride_score_h stride_score_n stride_oz stride_oh NKV_CTX BLOCK_N
    s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Genuine closed-form attention-score specification (case 1)

The following definitions give the *genuine* per-column attention-score value the
case-1 kernel produces — a closed form over the input buffers `Q`, `K`, `M`, with
**no reference to the kernel's own execution**. They mirror, lane-by-lane, the
`@triton.jit` body extracted via `#print` of the elaborated surface:

* `dist` is **nat-valued** (matching the elaborated body, whose `dist` op chain is
  entirely `Op.sub .nat`/`Op.add .nat`): the difference `arange[:,None] −
  arange[None,:]` is `(i − j : ℕ)`, then `+ start_m − start_n·BLOCK_N + 0` all in
  `ℕ` (nat-truncated, so never negative). For query block `c ∈ {0,1}`,
  `start_m = c·64`.
* `mask(i,j,c) = (0 ≤ dist) ∧ (dist < 64)` (both `ℕ` comparisons; the `0 ≤ dist`
  conjunct is vacuous, so effectively `dist < 64`).
* `qk[i,j] = exp2( sm_scale · log2e · rawScore(c·64+i, start_n·64+j) − M[c·64+i] )`
  with `exp2 x = Real.exp (x · log 2) = pow2 x` and
  `log2e = 1.4426950408889634`.
* `o[j] = Σ_{c<2} Σ_{i<64} (if mask(i,j,c) then qk[i,j] else 0)`
  (the `reduceSum` over axis 0 = query rows, accumulated across the 2 query
  blocks of the `for start_m in range(0,128,64)` loop). -/

/-- Raw, unscaled QK dot for query row `r`, global key column `n`:
`Σ_{d<64} Q[r,d]·K[d,n]` with the case-1 layout `Q[r,d] @ qoff+r·64+d`,
`K[d,n] @ koff+d+n·64`. -/
noncomputable def case1RawScore
    (s : BlockState) (Q K : RegionName) (qoff koff : Nat) (r n : Nat) : ℝ :=
  Finset.univ.sum (fun d : Fin 64 =>
    s.readMem Q (qoff + r * 64 + d.val) * s.readMem K (koff + d.val + n * 64))

/-- The case-1 base offset `off_z·32768 + off_h·8192` shared by `Q` and `K`
(`off_z = off_hz / 4`, `off_h = off_hz % 4`, `off_hz = s.pids 1`). -/
def case1QKOffset (s : BlockState) : Nat :=
  (s.pids 1 / 4) * 32768 + (s.pids 1 % 4) * 8192

/-- The `M` (precomputed max) offset for query row `r`: `off_hz·128 + r`. -/
def case1MOffset (s : BlockState) (r : Nat) : Nat := s.pids 1 * 128 + r

/-- Sliding-window distance for in-block query row `i`, key column `j`, query
block `c`, matching the **elaborated kernel** which computes `dist` in `ℕ` (the
`tl.arange[:,None] − tl.arange[None,:] + start_m − start_n·64 + 0` chain lowers to
`Op.sub .nat`/`Op.add .nat`, i.e. *nat-truncated* subtraction): with
`start_m = c·64`,
`dist = ((((i − j) + c·64) − start_n·64) + 0)` over `ℕ`.

Note: this is genuinely nat-truncated, *not* the real-valued form — for
`start_n ≥ 1` the two differ (e.g. `i = j = 0, c = 0, start_n = 1` gives nat
`dist = 0` but real `dist = −64`). The kernel masks-in this cell, so the nat form
is the faithful spec. -/
def case1Dist (s : BlockState) (c i j : Nat) : Nat :=
  (((i - j) + c * 64) - s.pids 0 * 64) + 0

/-- Sliding-window mask (case 1, non-complement): `dist ≥ 0 ∧ dist < 64` over `ℕ`
(matching the elaborated `boolAnd (ge dist 0) (lt dist 64)`; the `ge … 0` conjunct
is vacuously true in `ℕ`, so this is `dist < 64`). -/
def case1Mask (s : BlockState) (c i j : Nat) : Prop :=
  0 ≤ case1Dist s c i j ∧ case1Dist s c i j < 64

instance (s : BlockState) (c i j : Nat) :
    Decidable (case1Mask s c i j) := by unfold case1Mask; infer_instance

/-- Per-cell masked softmax weight `qk[i,j]` for query block `c`:
`exp2( sm_scale · log2e · rawScore(c·64+i, start_n·64+j) − M[c·64+i] )`. -/
noncomputable def case1Weight
    (s : BlockState) (Q K M : RegionName) (sm_scale : ℝ) (c i j : Nat) : ℝ :=
  pow2 (sm_scale * 1.4426950408889634 *
      case1RawScore s Q K (case1QKOffset s) (case1QKOffset s)
        (c * 64 + i) (s.pids 0 * 64 + j)
    - s.readMem M (case1MOffset s (c * 64 + i)))

/-- **Genuine closed-form attention score** for output key column `j`
(`j' = start_n·64 + j` globally): the masked-`exp2` column sum over the two query
blocks. This is the specification the case-1 kernel must satisfy. -/
noncomputable def case1OutClosedForm
    (s : BlockState) (Q K M : RegionName) (sm_scale : ℝ) (j : Fin 64) : ℝ :=
  Finset.univ.sum (fun c : Fin 2 =>
    Finset.univ.sum (fun i : Fin 64 =>
      if case1Mask s c.val i.val j.val
        then case1Weight s Q K M sm_scale c.val i.val j.val
        else 0))

/-! ### General (dimension-parameterized) case-1 closed-form spec

The `…G` defs below re-parameterize the literal-pinned case-1 spec (above) over
the kernel dimensions / strides.  The kernel lays out
`Q[r,d] @ qoff + r·stride_qm + d·stride_qk` (block-ptr `[N_CTX, BLOCK_DMODEL]`
strides `[stride_qm, stride_qk]`), `K[d,n] @ koff + d·stride_kk + n·stride_kn`
(block-ptr `[BLOCK_DMODEL, NKV_CTX]` strides `[stride_kk, stride_kn]`, key column
offset `start_n·BLOCK_N`).  The pinned defs are the `BLOCK_DMODEL = BLOCK_M =
BLOCK_N = 64`, `stride_qm = stride_kn = 64`, `stride_qk = stride_kk = 1`,
`stride_qz = stride_kz = 32768`, `stride_qh = stride_kh = 8192`,
`ROUND_CTX = NKV_CTX = N_CTX = 128`, `H = H_KV = 4`, `sliding_window_offset = 0`,
`sliding_window_size = 64` instance. -/

/-- Query element `Q[r, d]` at `qoff + r·stride_qm + d·stride_qk` (the `Q`
block-ptr `[ROUND_CTX, BLOCK_DMODEL]` layout; base `qoff = case1QKOffsetQG`). -/
noncomputable def case1QElemG (s : BlockState) (Q : RegionName)
    (stride_qm stride_qk qoff r d : Nat) : ℝ :=
  s.readMem Q (qoff + r * stride_qm + d * stride_qk)

/-- Key element `K[d, n]` at `koff + d·stride_kk + n·stride_kn` (the `K`
block-ptr `[BLOCK_DMODEL, NKV_CTX]` layout; base `koff = case1QKOffsetKG`,
global key column `n`). -/
noncomputable def case1KElemG (s : BlockState) (K : RegionName)
    (stride_kk stride_kn koff d n : Nat) : ℝ :=
  s.readMem K (koff + d * stride_kk + n * stride_kn)

/-- General raw, unscaled QK dot for query row `r`, global key column `n`:
`Σ_{d<BLOCK_DMODEL} Q[r,d]·K[d,n]` (elements via `case1QElemG`/`case1KElemG`). -/
noncomputable def case1RawScoreG
    (s : BlockState) (Q K : RegionName)
    (BLOCK_DMODEL stride_qm stride_qk stride_kk stride_kn : Nat)
    (qoff koff : Nat) (r n : Nat) : ℝ :=
  Finset.univ.sum (fun d : Fin BLOCK_DMODEL =>
    case1QElemG s Q stride_qm stride_qk qoff r d.val
      * case1KElemG s K stride_kk stride_kn koff d.val n)

/-- General `Q` base offset `off_z·stride_qz + off_h·stride_qh`
(`off_z = off_hz / H`, `off_h = off_hz % H`, `off_hz = s.pids 1`). -/
def case1QKOffsetQG (s : BlockState) (H stride_qz stride_qh : Nat) : Nat :=
  (s.pids 1 / H) * stride_qz + (s.pids 1 % H) * stride_qh

/-- General `K` base offset `off_z·stride_kz + off_hkv·stride_kh`
(`off_hkv = off_h / (H / H_KV)`). -/
def case1QKOffsetKG (s : BlockState) (H H_KV stride_kz stride_kh : Nat) : Nat :=
  (s.pids 1 / H) * stride_kz + ((s.pids 1 % H) / (H / H_KV)) * stride_kh

/-- General `M` offset for query row `r`: `off_hz·ROUND_CTX + r`. -/
def case1MOffsetG (s : BlockState) (ROUND_CTX r : Nat) : Nat := s.pids 1 * ROUND_CTX + r

/-- General nat-truncated sliding-window distance for in-block query row `i`, key
column `j`, query block `c` (`start_m = c·BLOCK_M`):
`dist = ((((i − j) + c·BLOCK_M) − start_n·BLOCK_N) + sliding_window_offset)` over `ℕ`. -/
def case1DistG (s : BlockState) (BLOCK_M BLOCK_N sliding_window_offset c i j : Nat) : Nat :=
  (((i - j) + c * BLOCK_M) - s.pids 0 * BLOCK_N) + sliding_window_offset

/-- General sliding-window mask (non-complement): `0 ≤ dist ∧ dist < sliding_window_size`. -/
def case1MaskG (s : BlockState)
    (BLOCK_M BLOCK_N sliding_window_offset sliding_window_size c i j : Nat) : Prop :=
  0 ≤ case1DistG s BLOCK_M BLOCK_N sliding_window_offset c i j
    ∧ case1DistG s BLOCK_M BLOCK_N sliding_window_offset c i j < sliding_window_size

instance (s : BlockState)
    (BLOCK_M BLOCK_N sliding_window_offset sliding_window_size c i j : Nat) :
    Decidable (case1MaskG s BLOCK_M BLOCK_N sliding_window_offset sliding_window_size c i j) := by
  unfold case1MaskG; infer_instance

/-- General per-cell masked softmax weight for query block `c`:
`exp2( sm_scale · log2e · rawScore(c·BLOCK_M+i, start_n·BLOCK_N+j) − M[c·BLOCK_M+i] )`. -/
noncomputable def case1WeightG
    (s : BlockState) (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV ROUND_CTX BLOCK_M BLOCK_N BLOCK_DMODEL
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn : Nat)
    (c i j : Nat) : ℝ :=
  pow2 (sm_scale * 1.4426950408889634 *
      case1RawScoreG s Q K BLOCK_DMODEL stride_qm stride_qk stride_kk stride_kn
        (case1QKOffsetQG s H stride_qz stride_qh)
        (case1QKOffsetKG s H H_KV stride_kz stride_kh)
        (c * BLOCK_M + i) (s.pids 0 * BLOCK_N + j)
    - s.readMem M (case1MOffsetG s ROUND_CTX (c * BLOCK_M + i)))

/-- General inner per-query-block column sum
`Σ_{i<BLOCK_M} (if mask(c,i,j) then weight(c,i,j) else 0)`. -/
noncomputable def case1ColSumG
    (s : BlockState) (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV ROUND_CTX BLOCK_M BLOCK_N BLOCK_DMODEL
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn
      sliding_window_offset sliding_window_size : Nat)
    (c : Nat) (j : Fin BLOCK_N) : ℝ :=
  Finset.univ.sum (fun i : Fin BLOCK_M =>
    if case1MaskG s BLOCK_M BLOCK_N sliding_window_offset sliding_window_size c i.val j.val
      then case1WeightG s Q K M sm_scale H H_KV ROUND_CTX BLOCK_M BLOCK_N BLOCK_DMODEL
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn
        c i.val j.val
      else 0)

/-- **General genuine closed-form attention score** for output key column `j`:
the masked-`exp2` query-row column sum over the `ROUND_CTX/BLOCK_M` query blocks. -/
noncomputable def case1OutClosedFormG
    (s : BlockState) (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV ROUND_CTX BLOCK_M BLOCK_N BLOCK_DMODEL
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn
      sliding_window_offset sliding_window_size : Nat)
    (j : Fin BLOCK_N) : ℝ :=
  Finset.univ.sum (fun c : Fin (ROUND_CTX / BLOCK_M) =>
    case1ColSumG s Q K M sm_scale H H_KV ROUND_CTX BLOCK_M BLOCK_N BLOCK_DMODEL
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn
      sliding_window_offset sliding_window_size c.val j)

/-! ### Validated stepping infrastructure (case-1 control flow)

The case-1 kernel wraps its q-load, sliding-window mask, and `where` in
`if (constBool _) { … }` guards (the lowered `IS_EVEN_M`/`SLIDING_WINDOW`/
`IS_EVEN_N` heuristic flags).  These lemmas discharge those guards so the body
threads like a flat statement list. -/

theorem evalOp_constBool' (b : Bool) (s : BlockState) :
    evalOp (Op.constBool b) s = some (Tile.scalar b) := by
  unfold evalOp; rfl

theorem stepStmt_ifThenElse_true (thenB elseB : List Stmt) (s : BlockState) :
    stepStmt (.ifThenElse (Op.constBool Bool.true) thenB elseB) s
      = stepStmts thenB s := by
  unfold stepStmt; rw [evalOp_constBool']; rfl

theorem stepStmt_ifThen_true (body : List Stmt) (s : BlockState) :
    stepStmt (.ifThen (Op.constBool Bool.true) body) s = stepStmts body s := by
  unfold stepStmt; rw [evalOp_constBool']; rfl

theorem stepStmt_ifThenElse_false (thenB elseB : List Stmt) (s : BlockState) :
    stepStmt (.ifThenElse (Op.constBool Bool.false) thenB elseB) s
      = stepStmts elseB s := by
  unfold stepStmt; rw [evalOp_constBool']; rfl

theorem stepStmt_ifThen_boolNot_true (body : List Stmt) (s : BlockState) :
    stepStmt (.ifThen (Op.boolNot (Op.constBool Bool.true)) body) s = some s := by
  unfold stepStmt
  have hb : evalOp (Op.boolNot (Op.constBool Bool.true)) s
      = some (Tile.uop (fun x : Bool => !x) (Tile.scalar Bool.true)) := by
    conv_lhs => unfold evalOp
    rw [evalOp_constBool']; rfl
  rw [hb]; rfl

/-! ### Verified elaborated-body decomposition (case 1)

The case-1 elaborated `@triton.jit` body (`(attention_score_kernel …).toAlgKernel.body`,
extracted by `simp [attention_score_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]`
and cross-checked structurally) is a **21-statement** list:

* **preLoop (16):** `start_n` (`programId 0`), `off_hz` (`programId 1`), `off_z`
  (`off_hz // 4`), `off_h` (`off_hz % 4`), `off_hkv` (`off_h // (4 // 4)`),
  `q_offset` (`off_z·32768 + off_h·8192`), `k_offset` (`off_z·32768 + off_hkv·8192`),
  `m_ptrs` (`M + off_hz·128 + arange 64`), `o` (`full [64] 0`), `Q_block_ptr`
  (`makeBlockPtrDyn Q q_offset [128,64] [64,64] [64,1] [0,0]`), `K_block_ptr`
  (`makeBlockPtrDynOffsets K k_offset [64,128] [64,64] [1,64] [0, start_n·64]`),
  `ifThenElse (constBool true)` (`IS_EVEN_N`) ⇒ `k = load(K_block_ptr)`,
  `lo` (`0`), `hi` (`128`), `qk_scale` (`sm_scale`), `qk_scale` (`qk_scale·log2e`).
* **loop (#16):** `forRangeDyn "start_m" (ref lo) (ref hi) (constNat 64) loopBody`,
  start=0, stop=128, step=64 — **exactly 2 iterations** (`start_m ∈ {0, 64}`).
  `loopBody` (per iteration): `start_m` (`multiple_of` ⇒ identity ref), `ifThenElse
  (constBool true)` ⇒ `q = load(Q_block_ptr)`, `m = load(m_ptrs)`, `qk` (`full 0`),
  `qk += dot q k`, `qk *= qk_scale`, `ifThen (constBool true)` (`SLIDING_WINDOW`) ⇒
  [`dist` (`(((arange[:,None] − arange[None,:]) + start_m) − start_n·64) + 0`, all
  `Op.sub/.add .nat`), `ifThenElse (constBool false)` (`COMPLEMENT`) ⇒ else-branch
  `mask = boolAnd (ge dist 0) (lt dist 64)`], `qk -= m[:,None]`, `p = exp2 qk`,
  `ifThen (constBool true)` ⇒ `p = where mask p 0`, `ifThen (constBool true).boolNot`
  (`not IS_EVEN_N` = `false`) ⇒ **skipped**, `o += reduceSum axis-0 p`,
  `Q_block_ptr = advance [64,0]`, `m_ptrs += 64`.
* **post (#17–20):** `o_offset` (`off_z·512 + off_h·128`), `o_range`
  (`arange 64 + start_n·64`), `o_ptrs` (`Out + o_offset + o_range`),
  `store o_ptrs o (mask: o_range < 128)`.

`attention_score_case1_body_split` exposes this as `take 16 ++ forRangeDyn-loop ::
post` so an exec proof threads the prefix with `stepStmts.append_some`, unfolds the
`forRangeDyn` (resolving `lo=0`/`hi=128`/`step=64`) with `stepForRangeAux.step_lt`
×2 + `step_ge`, and finishes on the masked store. -/
theorem attention_score_case1_body_split
    (Q K M Out : RegionName) (sm_scale : ℝ) :
    (attention_score_kernel Q K M Out
        32768 8192 64 1 32768 8192 64 1 512 128 1
        2 4 4 128 128 128 0 64 64 64 64 sm_scale
        Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel.body
      = (attention_score_kernel Q K M Out
          32768 8192 64 1 32768 8192 64 1 512 128 1
          2 4 4 128 128 128 0 64 64 64 64 sm_scale
          Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel.body.take 16
        ++ (attention_score_kernel Q K M Out
            32768 8192 64 1 32768 8192 64 1 512 128 1
            2 4 4 128 128 128 0 64 64 64 64 sm_scale
            Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel.body.drop 16 := by
  rw [List.take_append_drop]

/-! ### Local `evalOp` reduction helpers (case-1 closed-form exec proof)

The block-pointer / `floorDiv`/`mod`/`boolAnd`/`exp2`/`ge`/`ptrAdd` ops appearing
in the elaborated case-1 body have no `@[simp]` `evalOp_*` form; these mirror the
companions in `Examples.AttentionForwardClosedForm` so the prefix / loop body step
under `simp`. -/

theorem evalOp_floorDiv' {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.floorDiv h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.floorDiv bc vx vy)) := by
  simp [evalOp]

theorem evalOp_mod' {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.mod h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.mod bc vx vy)) := by
  simp [evalOp]

theorem evalOp_boolAnd' {a b shape} (bc : Broadcast a b shape)
    (x y : Op .bool _) (s : BlockState) :
    evalOp (.boolAnd bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s;
      some (Tile.bop (fun p q : Bool => p && q) bc vx vy)) := by
  simp [evalOp]

theorem evalOp_ge' {dtype a b shape} (h : ComparableDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

theorem evalOp_exp2' {shape : TileShape} (a : Op .real shape) (s : BlockState) :
    evalOp (.exp2 a) s = (do let va ← evalOp a s; some (Tile.uop WithBot.realExp2 va)) := by
  simp [evalOp]

theorem evalOp_ptrAdd' {a b shape} (bc : Broadcast a b shape)
    (ptr : Op .ptr a) (off : Op .nat b) (s : BlockState) :
    evalOp (.ptrAdd bc ptr off) s = (do
      let ptrs ← evalOp ptr s; let offs ← evalOp off s;
      some (Tile.ptrAdd bc ptrs offs)) := by simp [evalOp]

theorem evalOp_ptrBase' (region : RegionName) (s : BlockState) :
    evalOp (.ptrBase region) s = some (Tile.scalar (region.cast, 0)) := by simp [evalOp]

/-- **`makeBlockPtrDyn` eval** (the `Q_block_ptr` with static `[0,0]` offsets). -/
theorem makeBlockPtrDyn_eval (region : RegionName) (baseOffset : Op .nat [])
    (parentShape : List Nat) (blockShape : TileShape)
    (strides : List Nat) (offsets : List Nat) (s : BlockState) (base : Nat)
    (hb : evalOp baseOffset s = some (Tile.scalar base)) :
    evalOp (.makeBlockPtrDyn region baseOffset parentShape blockShape strides offsets) s
      = some (⟨fun _ =>
          { region := region, baseOffset := base, parentShape := parentShape,
            blockShape := blockShape, strides := strides, offsets := offsets }⟩
          : Tile .blockPtr blockShape) := by
  simp [evalOp, hb]

/-- **No-mask `.ptr` real load** (the `m = tl.load(m_ptrs)` non-block load):
reads `readMem` at each pointer lane. -/
theorem load_ptr_none_real_score {shape : TileShape}
    (ptrOp : Op .ptr shape) (s : BlockState) (ptrs : Tile .ptr shape)
    (hp : evalOp ptrOp s = some ptrs) :
    evalOp (.load .real (.ptr ptrOp) .none) s
      = some ⟨fun i => some (s.readMem (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, hp]
  refine congrArg some ?_
  ext i
  simp [BlockState.readMemValue_real]

/-! ### Column-sum value of one query block

`case1ColSum s Q K M sm_scale c j` is the inner per-query-block column sum
`Σ_{i<64} (if mask(c,i,j) then weight(c,i,j) else 0)` — exactly one summand of
`case1OutClosedForm`. -/
noncomputable def case1ColSum
    (s : BlockState) (Q K M : RegionName) (sm_scale : ℝ) (c : Nat) (j : Fin 64) : ℝ :=
  Finset.univ.sum (fun i : Fin 64 =>
    if case1Mask s c i.val j.val
      then case1Weight s Q K M sm_scale c i.val j.val
      else 0)

theorem case1OutClosedForm_eq_colSum
    (s : BlockState) (Q K M : RegionName) (sm_scale : ℝ) (j : Fin 64) :
    case1OutClosedForm s Q K M sm_scale j
      = case1ColSum s Q K M sm_scale 0 j + case1ColSum s Q K M sm_scale 1 j := by
  simp only [case1OutClosedForm, case1ColSum, Fin.sum_univ_two, Fin.val_zero, Fin.val_one]

/-- The 14-statement case-1 loop body (`start_m = c·64` already set on entry).
Extracted by `rfl` from `body.drop 16 = forRangeDyn … loopBody :: post`. -/
def attentionScoreCase1LoopBody : List Stmt :=
  [Stmt.assign TileDType.nat [] "start_m" (Op.ref TileDType.nat [] "start_m"),
    Stmt.ifThenElse (Op.constBool Bool.true)
      [Stmt.assign TileDType.real [64, 64] "q"
          (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "Q_block_ptr") [])
            MaskOpt.none)]
      [Stmt.assign TileDType.real [64, 64] "q"
          (Op.load TileDType.real
            (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "Q_block_ptr") [0, 1]) MaskOpt.none)],
    Stmt.assign TileDType.real [64] "m"
      (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [64] "m_ptrs")) MaskOpt.none),
    Stmt.assign TileDType.real [64, 64] "qk" (Op.full [64, 64] (Op.const 0)),
    Stmt.assign TileDType.real [64, 64] "qk"
      (Op.add NumericDType.real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref TileDType.real [64, 64] "qk")
        (Op.dot (batch := []) (Op.ref TileDType.real [64, 64] "q") (Op.ref TileDType.real [64, 64] "k"))),
    Stmt.assign TileDType.real [64, 64] "qk"
      (Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [64, 64] "qk")
        (Op.ref TileDType.real [] "qk_scale")),
    Stmt.ifThen (Op.constBool Bool.true)
      [Stmt.assign TileDType.nat [64, 64] "dist"
          (Op.add NumericDType.nat Broadcast.scalarR
            (Op.sub NumericDType.nat Broadcast.scalarR
              (Op.add NumericDType.nat Broadcast.scalarR
                (Op.sub NumericDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                  (Op.expandDim ⟨1, by simp⟩ (Op.arange 64)) (Op.expandDim ⟨0, by simp⟩ (Op.arange 64)))
                (Op.ref TileDType.nat [] "start_m"))
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_n") (Op.constNat 64)))
            (Op.constNat 0)),
        Stmt.ifThenElse (Op.constBool Bool.false)
          [Stmt.assign TileDType.bool [64, 64] "mask"
              (Op.ge ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [64, 64] "dist")
                (Op.constNat 64))]
          [Stmt.assign TileDType.bool [64, 64] "mask"
              (Op.boolAnd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                (Op.ge ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [64, 64] "dist")
                  (Op.constNat 0))
                (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [64, 64] "dist")
                  (Op.constNat 64)))]],
    Stmt.assign TileDType.real [64, 64] "qk"
      (Op.sub NumericDType.real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref TileDType.real [64, 64] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.real [64] "m"))),
    Stmt.assign TileDType.real [64, 64] "p" (Op.exp2 (Op.ref TileDType.real [64, 64] "qk")),
    Stmt.ifThen (Op.constBool Bool.true)
      [Stmt.assign TileDType.real [64, 64] "p"
          (Op.where (Op.ref TileDType.bool [64, 64] "mask") (Op.ref TileDType.real [64, 64] "p")
            (Op.broadcast (Op.const 0) [64, 64]))],
    Stmt.ifThen (Op.boolNot (Op.constBool Bool.true))
      [Stmt.assign TileDType.real [64, 64] "p"
          (Op.where
            (Op.remap [64, 64] (Broadcast.leftIndex (Broadcast.consSame (Broadcast.consL Broadcast.nil)))
              (Op.expandDim ⟨1, by simp⟩
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.add NumericDType.nat Broadcast.scalarR (Op.arange 64)
                    (Op.ref TileDType.nat [] "start_m"))
                  (Op.constNat 128))))
            (Op.ref TileDType.real [64, 64] "p") (Op.broadcast (Op.const 0) [64, 64]))],
    Stmt.assign TileDType.real [64] "o"
      (Op.add NumericDType.real (Broadcast.consSame Broadcast.nil) (Op.ref TileDType.real [64] "o")
        (Op.reduceSum ⟨0, by simp⟩ Bool.false (Op.ref TileDType.real [64, 64] "p"))),
    Stmt.assign TileDType.blockPtr [64, 64] "Q_block_ptr"
      (Op.advanceBlockPtr (Op.ref TileDType.blockPtr [64, 64] "Q_block_ptr") [(64:Nat), (0:Nat)]),
    Stmt.assign TileDType.ptr [64] "m_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [64] "m_ptrs") (Op.constNat 64))]

/-- **General case-1 loop body** (`start_m = c·BLOCK_M` already set on entry),
parameterized over the block dims (`BN = BLOCK_M = BLOCK_N`, `BD = BLOCK_DMODEL`),
sliding-window offset/size, and `N_CTX`.  The pinned `attentionScoreCase1LoopBody`
is the `BN = BD = 64`, `swo = 0`, `sws = 64`, `N_CTX = 128` instance. -/
def attentionScoreCase1LoopBodyG (BN BD swo sws N_CTX : Nat) : List Stmt :=
  [Stmt.assign TileDType.nat [] "start_m" (Op.ref TileDType.nat [] "start_m"),
    Stmt.ifThenElse (Op.constBool Bool.true)
      [Stmt.assign TileDType.real [BN, BD] "q"
          (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BN, BD] "Q_block_ptr") [])
            MaskOpt.none)]
      [Stmt.assign TileDType.real [BN, BD] "q"
          (Op.load TileDType.real
            (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BN, BD] "Q_block_ptr") [0, 1]) MaskOpt.none)],
    Stmt.assign TileDType.real [BN] "m"
      (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [BN] "m_ptrs")) MaskOpt.none),
    Stmt.assign TileDType.real [BN, BN] "qk" (Op.full [BN, BN] (Op.const 0)),
    Stmt.assign TileDType.real [BN, BN] "qk"
      (Op.add NumericDType.real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref TileDType.real [BN, BN] "qk")
        (Op.dot (batch := []) (Op.ref TileDType.real [BN, BD] "q") (Op.ref TileDType.real [BD, BN] "k"))),
    Stmt.assign TileDType.real [BN, BN] "qk"
      (Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [BN, BN] "qk")
        (Op.ref TileDType.real [] "qk_scale")),
    Stmt.ifThen (Op.constBool Bool.true)
      [Stmt.assign TileDType.nat [BN, BN] "dist"
          (Op.add NumericDType.nat Broadcast.scalarR
            (Op.sub NumericDType.nat Broadcast.scalarR
              (Op.add NumericDType.nat Broadcast.scalarR
                (Op.sub NumericDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                  (Op.expandDim ⟨1, by simp⟩ (Op.arange BN)) (Op.expandDim ⟨0, by simp⟩ (Op.arange BN)))
                (Op.ref TileDType.nat [] "start_m"))
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_n") (Op.constNat BN)))
            (Op.constNat swo)),
        Stmt.ifThenElse (Op.constBool Bool.false)
          [Stmt.assign TileDType.bool [BN, BN] "mask"
              (Op.ge ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BN, BN] "dist")
                (Op.constNat sws))]
          [Stmt.assign TileDType.bool [BN, BN] "mask"
              (Op.boolAnd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                (Op.ge ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BN, BN] "dist")
                  (Op.constNat 0))
                (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BN, BN] "dist")
                  (Op.constNat sws)))]],
    Stmt.assign TileDType.real [BN, BN] "qk"
      (Op.sub NumericDType.real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref TileDType.real [BN, BN] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.real [BN] "m"))),
    Stmt.assign TileDType.real [BN, BN] "p" (Op.exp2 (Op.ref TileDType.real [BN, BN] "qk")),
    Stmt.ifThen (Op.constBool Bool.true)
      [Stmt.assign TileDType.real [BN, BN] "p"
          (Op.where (Op.ref TileDType.bool [BN, BN] "mask") (Op.ref TileDType.real [BN, BN] "p")
            (Op.broadcast (Op.const 0) [BN, BN]))],
    Stmt.ifThen (Op.boolNot (Op.constBool Bool.true))
      [Stmt.assign TileDType.real [BN, BN] "p"
          (Op.where
            (Op.remap [BN, BN] (Broadcast.leftIndex (Broadcast.consSame (Broadcast.consL Broadcast.nil)))
              (Op.expandDim ⟨1, by simp⟩
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.add NumericDType.nat Broadcast.scalarR (Op.arange BN)
                    (Op.ref TileDType.nat [] "start_m"))
                  (Op.constNat N_CTX))))
            (Op.ref TileDType.real [BN, BN] "p") (Op.broadcast (Op.const 0) [BN, BN]))],
    Stmt.assign TileDType.real [BN] "o"
      (Op.add NumericDType.real (Broadcast.consSame Broadcast.nil) (Op.ref TileDType.real [BN] "o")
        (Op.reduceSum ⟨0, by simp⟩ Bool.false (Op.ref TileDType.real [BN, BN] "p"))),
    Stmt.assign TileDType.blockPtr [BN, BD] "Q_block_ptr"
      (Op.advanceBlockPtr (Op.ref TileDType.blockPtr [BN, BD] "Q_block_ptr") [BN, (0:Nat)]),
    Stmt.assign TileDType.ptr [BN] "m_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BN] "m_ptrs") (Op.constNat BN))]

/-- **`qk += dot q k` statement eval** (the raw-score accumulation onto the
zero-initialized `qk`). -/
theorem score_qkRaw_eval (st : BlockState) (qk0 : Tile .real [64,64])
    (qtile ktile : Tile .real [64,64])
    (hqk : st.regs .real [64,64] "qk" = some qk0)
    (hq : st.regs .real [64,64] "q" = some qtile)
    (hk : st.regs .real [64,64] "k" = some ktile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [64,64] "qk")
        (Op.dot (batch := []) (Op.ref .real [64,64] "q") (Op.ref .real [64,64] "k"))) st
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          qk0 (Tile.dot [] qtile ktile)) := by
  have hdot : @evalOp TileDType.real [64,64]
      (Op.dot (batch := []) (Op.ref .real [64,64] "q") (Op.ref .real [64,64] "k")) st
      = some (Tile.dot [] qtile ktile) := by
    conv_lhs => unfold evalOp
    simp [hq, hk]
  rw [evalOp_add]
  simp only [evalOp_ref, hqk, Option.bind_eq_bind, Option.bind_some]
  rw [hdot]; rfl

/-- **`qk *= qk_scale` statement eval** (scalar broadcast). -/
theorem score_qkScale_eval (st : BlockState) (qk1 : Tile .real [64,64]) (sc : ℝ)
    (hqk : st.regs .real [64,64] "qk" = some qk1)
    (hsc : st.regs .real [] "qk_scale" = some (Tile.scalar (some sc))) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [64,64] "qk") (Op.ref .real [] "qk_scale")) st
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qk1 (Tile.scalar (some sc))) := by
  rw [evalOp_mul]; simp only [evalOp_ref, hqk, hsc, Option.bind_eq_bind, Option.bind_some]

set_option maxRecDepth 8000 in
/-- **`dist` statement eval** (the nat-truncated sliding-window distance). -/
theorem score_dist_eval (st : BlockState) (c sn : Nat)
    (hsm : st.regs .nat [] "start_m" = some (Tile.scalar (c * 64)))
    (hsn : st.regs .nat [] "start_n" = some (Tile.scalar sn)) :
    evalOp (Op.add NumericDType.nat Broadcast.scalarR
        (Op.sub NumericDType.nat Broadcast.scalarR
          (Op.add NumericDType.nat Broadcast.scalarR
            (Op.sub NumericDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.expandDim ⟨1, by simp⟩ (Op.arange 64)) (Op.expandDim ⟨0, by simp⟩ (Op.arange 64)))
            (Op.ref TileDType.nat [] "start_m"))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_n") (Op.constNat 64)))
        (Op.constNat 0)) st
      = some (⟨fun idx : TileIndex [64, 64] =>
          (((idx.1.val - idx.2.1.val) + c * 64) - sn * 64) + 0⟩ : Tile .nat [64, 64]) := by
  have hexp1 : @evalOp TileDType.nat [64,1] (Op.expandDim ⟨1, by simp⟩ (Op.arange 64)) st
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun i : Fin 64 => i.val))) := by
    conv_lhs => unfold evalOp
    simp [evalOp_arange]
  have hexp0 : @evalOp TileDType.nat [1,64] (Op.expandDim ⟨0, by simp⟩ (Op.arange 64)) st
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun i : Fin 64 => i.val))) := by
    conv_lhs => unfold evalOp
    simp [evalOp_arange]
  simp only [evalOp_add, evalOp_sub, evalOp_mul, evalOp_constNat, evalOp_ref, hexp1, hexp0,
    hsm, hsn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.expandDim_data, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
    Tile.scalar, NumericDType.add, NumericDType.sub, NumericDType.mul, TileShape.dropInsertedIndex,
    TileShape.insertAxis, Fin.isValue]

/-- **`mask` statement eval** (the non-complement `(dist ≥ 0) ∧ (dist < 64)`). -/
theorem score_mask_eval (st : BlockState) (disttile : Tile .nat [64,64])
    (hd : st.regs .nat [64,64] "dist" = some disttile) :
    evalOp (Op.boolAnd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ge ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [64, 64] "dist")
          (Op.constNat 0))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [64, 64] "dist")
          (Op.constNat 64))) st
      = some (⟨fun idx : TileIndex [64, 64] =>
          (decide (0 ≤ disttile.data idx) && decide (disttile.data idx < 64))⟩ : Tile .bool [64, 64]) := by
  rw [evalOp_boolAnd', evalOp_ge', evalOp_lt]
  simp only [evalOp_ref, evalOp_constNat, hd, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp [Tile.bop, Tile.cop, ComparableDType.ge, ComparableDType.lt]

/-- **`qk -= m[:, None]` statement eval.** -/
theorem score_qk2_eval (st : BlockState) (qk2 : Tile .real [64,64]) (mt : Tile .real [64])
    (hqk : st.regs .real [64,64] "qk" = some qk2)
    (hm : st.regs .real [64] "m" = some mt) :
    evalOp (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [64,64] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [64] "m"))) st
      = some (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          qk2 (Tile.expandDim ⟨1, by simp⟩ mt)) := by
  have hexp : @evalOp TileDType.real [64,1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [64] "m")) st
      = some (Tile.expandDim ⟨1, by simp⟩ mt) := by
    conv_lhs => unfold evalOp
    simp [hm]
  rw [evalOp_sub]; simp only [evalOp_ref, hqk, Option.bind_eq_bind, Option.bind_some]
  rw [hexp]; rfl

/-- **`p = exp2 qk` statement eval.** -/
theorem score_p_eval (st : BlockState) (qkt : Tile .real [64,64])
    (hqk : st.regs .real [64,64] "qk" = some qkt) :
    evalOp (Op.exp2 (Op.ref .real [64,64] "qk")) st = some (Tile.uop WithBot.realExp2 qkt) := by
  rw [evalOp_exp2']; simp [hqk]

/-- **`p = where mask p 0` statement eval.** -/
theorem score_pwhere_eval (st : BlockState) (mkt : Tile .bool [64,64]) (pt : Tile .real [64,64])
    (hmask : st.regs .bool [64,64] "mask" = some mkt)
    (hp : st.regs .real [64,64] "p" = some pt) :
    evalOp (Op.where (Op.ref .bool [64,64] "mask") (Op.ref .real [64,64] "p")
        (Op.broadcast (Op.const 0) [64,64])) st
      = some (Tile.select mkt pt (⟨fun _ : TileIndex [64,64] => some (0:ℝ)⟩ : Tile .real [64,64])) := by
  have hbr : evalOp (Op.broadcast (Op.const 0) [64,64]) st
      = some (⟨fun _ : TileIndex [64,64] => some (0:ℝ)⟩ : Tile .real [64,64]) := by
    simp [evalOp]
  rw [evalOp_where]; simp only [evalOp_ref, hmask, hp, hbr, Option.bind_eq_bind, Option.bind_some]

/-- **`o += reduceSum(p, axis 0)` statement eval** (column sum). -/
theorem score_oAcc_eval (st : BlockState) (ot : Tile .real [64]) (pt : Tile .real [64,64])
    (ho : st.regs .real [64] "o" = some ot)
    (hp : st.regs .real [64,64] "p" = some pt) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [64] "o")
        (Op.reduceSum ⟨0, by simp⟩ Bool.false (Op.ref .real [64,64] "p"))) st
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil) ot
          (Tile.reduceSumDrop (⟨0, by simp⟩ : Fin [64,64].length) pt)) := by
  have hrs : @evalOp TileDType.real [64]
      (Op.reduceSum ⟨0, by simp⟩ Bool.false (Op.ref .real [64,64] "p")) st
      = some (Tile.reduceSumDrop (⟨0, by simp⟩ : Fin [64,64].length) pt) := by
    conv_lhs => unfold evalOp
    simp [hp]
  rw [evalOp_add]; simp only [evalOp_ref, ho, Option.bind_eq_bind, Option.bind_some]
  rw [hrs]; rfl

/-- **Per-cell weight value.** The `exp2(scaled-dot − m)` tile cell at `(i, j)`
equals the closed-form `case1Weight`, given the loaded `q`/`k`/`m` tiles. -/
theorem score_wReg_cell (s : BlockState) (Q K M : RegionName) (sm_scale : ℝ) (c : Nat)
    (i j : Fin 64)
    (qtile ktile : Tile .real [64,64]) (mtile : Tile .real [64])
    (hq : qtile.data (i, j, PUnit.unit) = some (s.readMem Q (case1QKOffset s + (c * 64 + i.val) * 64 + j.val * 1)))
    (hqd : ∀ d : Fin 64, qtile.data (i, d, PUnit.unit) = some (s.readMem Q (case1QKOffset s + (c * 64 + i.val) * 64 + d.val * 1)))
    (hkd : ∀ d : Fin 64, ktile.data (d, j, PUnit.unit) = some (s.readMem K (case1QKOffset s + d.val + (s.pids 0 * 64 + j.val) * 64)))
    (hm : mtile.data (i, PUnit.unit) = some (s.readMem M (case1MOffset s (c * 64 + i.val)))) :
    (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (⟨fun _ : TileIndex [64,64] => some (0:ℝ)⟩ : Tile .real [64,64]) (Tile.dot [] qtile ktile))
          (Tile.scalar (some (sm_scale * 1.4426950408889634))))
        (Tile.expandDim ⟨1, by simp⟩ mtile))).data (i, j, PUnit.unit)
      = some (case1Weight s Q K M sm_scale c i.val j.val) := by
  have hdot : (Tile.dot [] qtile ktile).data (i, j, PUnit.unit)
      = some (case1RawScore s Q K (case1QKOffset s) (case1QKOffset s) (c * 64 + i.val) (s.pids 0 * 64 + j.val)) := by
    rw [Tile.dot_nil_data, case1RawScore]
    have hterm : (fun d : Fin 64 => Option.map₂ (· * ·) (qtile.data (i, d, PUnit.unit)) (ktile.data (d, j, PUnit.unit)))
        = (fun d : Fin 64 => ((s.readMem Q (case1QKOffset s + (c * 64 + i.val) * 64 + d.val) *
              s.readMem K (case1QKOffset s + d.val + (s.pids 0 * 64 + j.val) * 64) : ℝ) : WithBot ℝ)) := by
      funext d; rw [hqd d, hkd d]; simp only [mul_one, Option.map₂]; norm_cast
    rw [hterm, ← WithBot.coe_sum]
    refine congrArg some (Finset.sum_congr rfl (fun d _ => by ring_nf))
  show WithBot.realExp2 _ = _
  simp only [Tile.uop_data, Tile.bop_data, Tile.expandDim_data, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex, hdot, hm, Tile.scalar, NumericDType.add, NumericDType.mul, NumericDType.sub,
    WithBot.realAdd, WithBot.realMul, WithBot.realSub, Option.map₂, Option.bind, Option.map]
  rw [case1Weight, WithBot.realExp2]
  refine congrArg some ?_
  rw [pow2]
  congr 1
  ring

/-! ### General per-statement eval helpers (dimension-parameterized)

Shape-general mirrors of the `score_*_eval` lemmas above. -/

/-- **General per-cell weight value.** The `exp2(scaled-dot − m)` tile cell at
`(i,j)` equals `case1WeightG`, given the loaded `q`/`k`/`m` tiles. -/
theorem score_wReg_cellG (s : BlockState) (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV ROUND_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn : Nat)
    (c : Nat) (i j : Fin BN)
    (qtile : Tile .real [BN,BD]) (ktile : Tile .real [BD,BN]) (mtile : Tile .real [BN])
    (hqd : ∀ d : Fin BD, qtile.data (i, d, PUnit.unit)
      = some (s.readMem Q (case1QKOffsetQG s H stride_qz stride_qh
          + (c * BN + i.val) * stride_qm + d.val * stride_qk)))
    (hkd : ∀ d : Fin BD, ktile.data (d, j, PUnit.unit)
      = some (s.readMem K (case1QKOffsetKG s H H_KV stride_kz stride_kh
          + d.val * stride_kk + (s.pids 0 * BN + j.val) * stride_kn)))
    (hm : mtile.data (i, PUnit.unit)
      = some (s.readMem M (case1MOffsetG s ROUND_CTX (c * BN + i.val)))) :
    (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (⟨fun _ : TileIndex [BN,BN] => some (0:ℝ)⟩ : Tile .real [BN,BN]) (Tile.dot [] qtile ktile))
          (Tile.scalar (some (sm_scale * 1.4426950408889634))))
        (Tile.expandDim ⟨1, by simp⟩ mtile))).data (i, j, PUnit.unit)
      = some (case1WeightG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
          stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn
          c i.val j.val) := by
  have hdot : (Tile.dot [] qtile ktile).data (i, j, PUnit.unit)
      = some (case1RawScoreG s Q K BD stride_qm stride_qk stride_kk stride_kn
          (case1QKOffsetQG s H stride_qz stride_qh) (case1QKOffsetKG s H H_KV stride_kz stride_kh)
          (c * BN + i.val) (s.pids 0 * BN + j.val)) := by
    rw [Tile.dot_nil_data, case1RawScoreG]
    have hterm : (fun d : Fin BD => Option.map₂ (· * ·) (qtile.data (i, d, PUnit.unit)) (ktile.data (d, j, PUnit.unit)))
        = (fun d : Fin BD => ((s.readMem Q (case1QKOffsetQG s H stride_qz stride_qh + (c * BN + i.val) * stride_qm + d.val * stride_qk) *
              s.readMem K (case1QKOffsetKG s H H_KV stride_kz stride_kh + d.val * stride_kk + (s.pids 0 * BN + j.val) * stride_kn) : ℝ) : WithBot ℝ)) := by
      funext d; rw [hqd d, hkd d]; simp only [Option.map₂]; norm_cast
    rw [hterm, ← WithBot.coe_sum]; rfl
  show WithBot.realExp2 _ = _
  simp only [Tile.uop_data, Tile.bop_data, Tile.expandDim_data, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex, hdot, hm, Tile.scalar, NumericDType.add, NumericDType.mul, NumericDType.sub,
    WithBot.realAdd, WithBot.realMul, WithBot.realSub, Option.map₂, Option.bind, Option.map]
  rw [case1WeightG, WithBot.realExp2]
  refine congrArg some ?_
  rw [pow2]
  congr 1
  ring

theorem score_qkRaw_evalG (BN BD : Nat) (st : BlockState) (qk0 : Tile .real [BN,BN])
    (qtile : Tile .real [BN,BD]) (ktile : Tile .real [BD,BN])
    (hqk : st.regs .real [BN,BN] "qk" = some qk0)
    (hq : st.regs .real [BN,BD] "q" = some qtile)
    (hk : st.regs .real [BD,BN] "k" = some ktile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BN,BN] "qk")
        (Op.dot (batch := []) (Op.ref .real [BN,BD] "q") (Op.ref .real [BD,BN] "k"))) st
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          qk0 (Tile.dot [] qtile ktile)) := by
  have hdot : @evalOp TileDType.real [BN,BN]
      (Op.dot (batch := []) (Op.ref .real [BN,BD] "q") (Op.ref .real [BD,BN] "k")) st
      = some (Tile.dot [] qtile ktile) := by
    conv_lhs => unfold evalOp
    simp [hq, hk]
  rw [evalOp_add]
  simp only [evalOp_ref, hqk, Option.bind_eq_bind, Option.bind_some]
  rw [hdot]; rfl

theorem score_qkScale_evalG (BN : Nat) (st : BlockState) (qk1 : Tile .real [BN,BN]) (sc : ℝ)
    (hqk : st.regs .real [BN,BN] "qk" = some qk1)
    (hsc : st.regs .real [] "qk_scale" = some (Tile.scalar (some sc))) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BN,BN] "qk") (Op.ref .real [] "qk_scale")) st
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qk1 (Tile.scalar (some sc))) := by
  rw [evalOp_mul]; simp only [evalOp_ref, hqk, hsc, Option.bind_eq_bind, Option.bind_some]

set_option maxRecDepth 8000 in
theorem score_dist_evalG (BN swo : Nat) (st : BlockState) (c sn : Nat)
    (hsm : st.regs .nat [] "start_m" = some (Tile.scalar (c * BN)))
    (hsn : st.regs .nat [] "start_n" = some (Tile.scalar sn)) :
    evalOp (Op.add NumericDType.nat Broadcast.scalarR
        (Op.sub NumericDType.nat Broadcast.scalarR
          (Op.add NumericDType.nat Broadcast.scalarR
            (Op.sub NumericDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.expandDim ⟨1, by simp⟩ (Op.arange BN)) (Op.expandDim ⟨0, by simp⟩ (Op.arange BN)))
            (Op.ref TileDType.nat [] "start_m"))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_n") (Op.constNat BN)))
        (Op.constNat swo)) st
      = some (⟨fun idx : TileIndex [BN, BN] =>
          (((idx.1.val - idx.2.1.val) + c * BN) - sn * BN) + swo⟩ : Tile .nat [BN, BN]) := by
  have hexp1 : @evalOp TileDType.nat [BN,1] (Op.expandDim ⟨1, by simp⟩ (Op.arange BN)) st
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun i : Fin BN => i.val))) := by
    conv_lhs => unfold evalOp
    simp [evalOp_arange]
  have hexp0 : @evalOp TileDType.nat [1,BN] (Op.expandDim ⟨0, by simp⟩ (Op.arange BN)) st
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun i : Fin BN => i.val))) := by
    conv_lhs => unfold evalOp
    simp [evalOp_arange]
  simp only [evalOp_add, evalOp_sub, evalOp_mul, evalOp_constNat, evalOp_ref, hexp1, hexp0,
    hsm, hsn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.expandDim_data, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
    Tile.scalar, NumericDType.add, NumericDType.sub, NumericDType.mul, TileShape.dropInsertedIndex,
    TileShape.insertAxis, Fin.isValue]

theorem score_mask_evalG (BN sws : Nat) (st : BlockState) (disttile : Tile .nat [BN,BN])
    (hd : st.regs .nat [BN,BN] "dist" = some disttile) :
    evalOp (Op.boolAnd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ge ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BN, BN] "dist")
          (Op.constNat 0))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BN, BN] "dist")
          (Op.constNat sws))) st
      = some (⟨fun idx : TileIndex [BN, BN] =>
          (decide (0 ≤ disttile.data idx) && decide (disttile.data idx < sws))⟩ : Tile .bool [BN, BN]) := by
  rw [evalOp_boolAnd', evalOp_ge', evalOp_lt]
  simp only [evalOp_ref, evalOp_constNat, hd, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp [Tile.bop, Tile.cop, ComparableDType.ge, ComparableDType.lt]

theorem score_qk2_evalG (BN : Nat) (st : BlockState) (qk2 : Tile .real [BN,BN]) (mt : Tile .real [BN])
    (hqk : st.regs .real [BN,BN] "qk" = some qk2)
    (hm : st.regs .real [BN] "m" = some mt) :
    evalOp (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BN,BN] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BN] "m"))) st
      = some (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          qk2 (Tile.expandDim ⟨1, by simp⟩ mt)) := by
  have hexp : @evalOp TileDType.real [BN,1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BN] "m")) st
      = some (Tile.expandDim ⟨1, by simp⟩ mt) := by
    conv_lhs => unfold evalOp
    simp [hm]
  rw [evalOp_sub]; simp only [evalOp_ref, hqk, Option.bind_eq_bind, Option.bind_some]
  rw [hexp]; rfl

theorem score_p_evalG (BN : Nat) (st : BlockState) (qkt : Tile .real [BN,BN])
    (hqk : st.regs .real [BN,BN] "qk" = some qkt) :
    evalOp (Op.exp2 (Op.ref .real [BN,BN] "qk")) st = some (Tile.uop WithBot.realExp2 qkt) := by
  rw [evalOp_exp2']; simp [hqk]

theorem score_pwhere_evalG (BN : Nat) (st : BlockState) (mkt : Tile .bool [BN,BN]) (pt : Tile .real [BN,BN])
    (hmask : st.regs .bool [BN,BN] "mask" = some mkt)
    (hp : st.regs .real [BN,BN] "p" = some pt) :
    evalOp (Op.where (Op.ref .bool [BN,BN] "mask") (Op.ref .real [BN,BN] "p")
        (Op.broadcast (Op.const 0) [BN,BN])) st
      = some (Tile.select mkt pt (⟨fun _ : TileIndex [BN,BN] => some (0:ℝ)⟩ : Tile .real [BN,BN])) := by
  have hbr : evalOp (Op.broadcast (Op.const 0) [BN,BN]) st
      = some (⟨fun _ : TileIndex [BN,BN] => some (0:ℝ)⟩ : Tile .real [BN,BN]) := by
    simp [evalOp]
  rw [evalOp_where]; simp only [evalOp_ref, hmask, hp, hbr, Option.bind_eq_bind, Option.bind_some]

theorem score_oAcc_evalG (BN : Nat) (st : BlockState) (ot : Tile .real [BN]) (pt : Tile .real [BN,BN])
    (ho : st.regs .real [BN] "o" = some ot)
    (hp : st.regs .real [BN,BN] "p" = some pt) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BN] "o")
        (Op.reduceSum ⟨0, by simp⟩ Bool.false (Op.ref .real [BN,BN] "p"))) st
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil) ot
          (Tile.reduceSumDrop (⟨0, by simp⟩ : Fin [BN,BN].length) pt)) := by
  have hrs : @evalOp TileDType.real [BN]
      (Op.reduceSum ⟨0, by simp⟩ Bool.false (Op.ref .real [BN,BN] "p")) st
      = some (Tile.reduceSumDrop (⟨0, by simp⟩ : Fin [BN,BN].length) pt) := by
    conv_lhs => unfold evalOp
    simp [hp]
  rw [evalOp_add]; simp only [evalOp_ref, ho, Option.bind_eq_bind, Option.bind_some]
  rw [hrs]; rfl

set_option maxHeartbeats 2000000 in
/-- **Loop-body execution + column-sum accumulation.** One iteration of the
case-1 loop body (with `start_m = c·64` already set), given the invariant register
readbacks, advances `o` by the per-query-block column sum `case1ColSum`, advances
`Q_block_ptr` by one row block, and `m_ptrs` by 64. -/
theorem score_loopBody_eval
    (s sin : BlockState) (Q K M : RegionName) (sm_scale : ℝ) (c : Nat)
    (oF : Fin 64 → ℝ)
    (hpids : sin.pids = s.pids) (hmem : sin.mem = s.mem)
    (hundef : ∀ rg o, sin.undef rg o = 0)
    (hsm : sin.regs .nat [] "start_m" = some (Tile.scalar (c * 64)))
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar (s.pids 0)))
    (hoz : sin.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / 4)))
    (hoh : sin.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % 4)))
    (hqks : sin.regs .real [] "qk_scale" = some (Tile.scalar (some (sm_scale * 1.4426950408889634))))
    (ho : sin.regs .real [64] "o" = some ⟨fun idx : TileIndex [64] => some (oF idx.1)⟩)
    (hk : sin.regs .real [64, 64] "k" = some ⟨fun idx : TileIndex [64, 64] =>
        some (s.readMem K (case1QKOffset s + idx.1.val + (s.pids 0 * 64 + idx.2.1.val) * 64))⟩)
    (hQbp : sin.regs .blockPtr [64, 64] "Q_block_ptr" = some ⟨fun _ : TileIndex [64, 64] =>
        { region := Q, baseOffset := case1QKOffset s, parentShape := [128, 64],
          blockShape := [64, 64], strides := [64, 1], offsets := [c * 64, 0] }⟩)
    (hmp : sin.regs .ptr [64] "m_ptrs" = some ⟨fun idx : TileIndex [64] =>
        (M.cast, s.pids 1 * 128 + c * 64 + idx.1.val)⟩) :
    ∃ sF, stepStmts attentionScoreCase1LoopBody sin = some sF
      ∧ sF.pids = s.pids ∧ sF.mem = s.mem ∧ (∀ rg o, sF.undef rg o = 0)
      ∧ sF.regs .nat [] "start_n" = some (Tile.scalar (s.pids 0))
      ∧ sF.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / 4))
      ∧ sF.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % 4))
      ∧ sF.regs .real [64, 64] "k" = some ⟨fun idx : TileIndex [64, 64] =>
          some (s.readMem K (case1QKOffset s + idx.1.val + (s.pids 0 * 64 + idx.2.1.val) * 64))⟩
      ∧ sF.regs .real [] "qk_scale" = some (Tile.scalar (some (sm_scale * 1.4426950408889634)))
      ∧ sF.regs .real [64] "o" = some ⟨fun idx : TileIndex [64] =>
          some (oF idx.1 + case1ColSum s Q K M sm_scale c idx.1)⟩
      ∧ sF.regs .blockPtr [64, 64] "Q_block_ptr" = some ⟨fun _ : TileIndex [64, 64] =>
          { region := Q, baseOffset := case1QKOffset s, parentShape := [128, 64],
            blockShape := [64, 64], strides := [64, 1], offsets := [c * 64 + 64, 0] }⟩
      ∧ sF.regs .ptr [64] "m_ptrs" = some ⟨fun idx : TileIndex [64] =>
          (M.cast, s.pids 1 * 128 + (c * 64 + 64) + idx.1.val)⟩ := by
  -- The loaded q tile (Q_block_ptr offsets [c·64,0], strides [64,1]).
  set qtile : Tile .real [64, 64] := ⟨fun idx : TileIndex [64, 64] =>
      some (s.readMem Q (case1QKOffset s + (c * 64 + idx.1.val) * 64 + idx.2.1.val * 1))⟩ with hqt
  set ktile : Tile .real [64, 64] := ⟨fun idx : TileIndex [64, 64] =>
      some (s.readMem K (case1QKOffset s + idx.1.val + (s.pids 0 * 64 + idx.2.1.val) * 64))⟩ with hkt
  set mtile : Tile .real [64] := ⟨fun idx : TileIndex [64] =>
      some (s.readMem M (s.pids 1 * 128 + c * 64 + idx.1.val))⟩ with hmt
  -- the dist tile and the mask register value produced by score_mask_eval
  set disttile : Tile .nat [64, 64] := ⟨fun idx : TileIndex [64, 64] =>
      (((idx.1.val - idx.2.1.val) + c * 64) - s.pids 0 * 64) + 0⟩ with hdistt
  set masktile : Tile .bool [64, 64] := ⟨fun idx : TileIndex [64, 64] =>
      decide (case1Mask s c idx.1.val idx.2.1.val)⟩ with hmask
  have hdistt_eq : ∀ idx : TileIndex [64, 64], disttile.data idx = case1Dist s c idx.1.val idx.2.1.val := by
    intro idx; simp [hdistt, case1Dist]
  have hmask_eq : ⟨fun idx : TileIndex [64, 64] =>
      (decide (0 ≤ disttile.data idx) && decide (disttile.data idx < 64))⟩ = masktile := by
    rw [hmask]; refine congrArg _ ?_; ext idx
    simp [hdistt_eq, case1Mask, decide_eq_decide]
  have hrm : sin.readMem = s.readMem := by funext rg o; simp [BlockState.readMem, hmem]
  -- q-load eval (parameterized over any state st whose Q_block_ptr / mem agree with sin)
  have hqEval : ∀ st : BlockState,
      st.regs .blockPtr [64,64] "Q_block_ptr" = some ⟨fun _ : TileIndex [64, 64] =>
        { region := Q, baseOffset := case1QKOffset s, parentShape := [128, 64],
          blockShape := [64, 64], strides := [64, 1], offsets := [c * 64, 0] }⟩ →
      st.readMem = s.readMem →
      evalOp (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [64,64] "Q_block_ptr") []) MaskOpt.none) st
        = some qtile := by
    intro st hq hrm'
    rw [load_blockPtr_Q_eval Q (case1QKOffset s) 128 64 64 64 64 1 (c * 64)
      (Op.ref .blockPtr [64, 64] "Q_block_ptr") _ (by rw [evalOp_ref]; exact hq)]
    rw [hqt]; refine congrArg some ?_; ext idx; simp [hrm']
  unfold attentionScoreCase1LoopBody
  -- stmt 0: start_m = ref start_m
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_m") sin = some (Tile.scalar (c * 64)) from by
      rw [evalOp_ref]; exact hsm))]
  -- stmt 1: ifThenElse true [q = load] _
  rw [stepStmts.cons_some
    (show stepStmt (Stmt.ifThenElse (Op.constBool Bool.true) _ _) _ = some _ from by
      rw [stepStmt_ifThenElse_true, stepStmts.cons_some (stepStmt_assign_eq_some
        (hqEval _ (by simp [hQbp]) (by funext rg o; simp [BlockState.readMem, hmem]))), stepStmts.nil])]
  -- stmt 2: m = load(m_ptrs)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [64] "m_ptrs")) MaskOpt.none) _ = some mtile from by
      rw [load_ptr_none_real_score (Op.ref .ptr [64] "m_ptrs") _
        (⟨fun idx : TileIndex [64] => (M.cast, s.pids 1 * 128 + c * 64 + idx.1.val)⟩ : Tile .ptr [64])
        (by rw [evalOp_ref]; simp [hmp])]
      rw [hmt]; refine congrArg some ?_; ext idx; simp [BlockState.readMem, hmem]))]
  -- stmt 3: qk = full [64,64] 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [64,64] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [64,64] => some (0:ℝ)⟩ : Tile .real [64,64]) from by
      simp [evalOp_full]))]
  -- stmt 4: qk = qk + dot q k  (the raw score)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (score_qkRaw_eval _ (⟨fun _ : TileIndex [64,64] => some (0:ℝ)⟩ : Tile .real [64,64]) qtile ktile
      (by simp) (by simp) (by simp [hk])))]
  -- stmt 5: qk = qk * qk_scale  (scaled raw score)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (score_qkScale_eval _
      (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (⟨fun _ : TileIndex [64,64] => some (0:ℝ)⟩ : Tile .real [64,64]) (Tile.dot [] qtile ktile))
      (sm_scale * 1.4426950408889634) (by simp) (by simp [hqks])))]
  -- stmt 6: ifThen true [dist; ifThenElse false [_] [mask]]
  rw [stepStmts.cons_some
    (show stepStmt (Stmt.ifThen (Op.constBool Bool.true) _) _ = some _ from by
      rw [stepStmt_ifThen_true, stepStmts.cons_some (stepStmt_assign_eq_some
        (score_dist_eval _ c (s.pids 0) (by simp [hsm]) (by simp [hsn])))]
      rw [stepStmts.cons_some
        (show stepStmt (Stmt.ifThenElse (Op.constBool Bool.false) _ _) _ = some _ from by
          rw [stepStmt_ifThenElse_false, stepStmts.cons_some (stepStmt_assign_eq_some
            ((score_mask_eval _ disttile (by simp [hdistt])).trans (congrArg some hmask_eq)))]
          exact stepStmts.nil)]
      exact stepStmts.nil)]
  -- the scaled raw-score tile (qk after stmt 5)
  set scaledqk : Tile .real [64, 64] := Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (⟨fun _ : TileIndex [64,64] => some (0:ℝ)⟩ : Tile .real [64,64]) (Tile.dot [] qtile ktile))
      (Tile.scalar (some (sm_scale * 1.4426950408889634))) with hsq
  -- stmt 7: qk -= m[:, None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (score_qk2_eval _ scaledqk mtile (by simp [hsq]) (by simp)))]
  -- stmt 8: p = exp2 qk
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (score_p_eval _ (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        scaledqk (Tile.expandDim ⟨1, by simp⟩ mtile)) (by simp)))]
  -- stmt 9: ifThen true [p = where mask p 0]
  rw [stepStmts.cons_some
    (show stepStmt (Stmt.ifThen (Op.constBool Bool.true) _) _ = some _ from by
      rw [stepStmt_ifThen_true, stepStmts.cons_some (stepStmt_assign_eq_some
        (score_pwhere_eval _ masktile
          (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
            (Broadcast.consSame (Broadcast.consR Broadcast.nil)) scaledqk (Tile.expandDim ⟨1, by simp⟩ mtile)))
          (by simp) (by simp)))]
      exact stepStmts.nil)]
  -- stmt 10: ifThen (boolNot (constBool true)) [...] → skipped
  rw [stepStmts.cons_some (stepStmt_ifThen_boolNot_true _ _)]
  -- stmt 11: o += reduceSum(p, axis 0)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (score_oAcc_eval _ (⟨fun idx : TileIndex [64] => some (oF idx.1)⟩ : Tile .real [64])
      (Tile.select masktile
        (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
          (Broadcast.consSame (Broadcast.consR Broadcast.nil)) scaledqk (Tile.expandDim ⟨1, by simp⟩ mtile)))
        (⟨fun _ : TileIndex [64,64] => some (0:ℝ)⟩ : Tile .real [64,64]))
      (by simp [ho]) (by simp)))]
  -- stmt 12: Q_block_ptr = advance [64, 0]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [64,64] "Q_block_ptr") [(64:Nat), (0:Nat)]) _
        = some (⟨fun _ : TileIndex [64, 64] =>
            { region := Q, baseOffset := case1QKOffset s, parentShape := [128, 64],
              blockShape := [64, 64], strides := [64, 1], offsets := [c * 64 + 64, 0] }⟩
            : Tile .blockPtr [64,64]) from by
      rw [advanceBlockPtr_eval]; simp [hQbp]
      funext i; congr 1 <;> omega))]
  -- stmt 13: m_ptrs += 64
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [64] "m_ptrs") (Op.constNat 64)) _
        = some (⟨fun idx : TileIndex [64] =>
            (M.cast, s.pids 1 * 128 + (c * 64 + 64) + idx.1.val)⟩ : Tile .ptr [64]) from by
      rw [evalOp_ptrAdd']; simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff, hmp,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      · rfl
      · simp only [Tile.ptrAdd_data, Tile.scalar, Broadcast.rightIndex, Broadcast.leftIndex]
        omega))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hpids]
  · funext rg o; simp only [BlockState.setReg_mem]; rw [hmem]
  · intro rg o; simp [hundef]
  · simp [hsn]
  · simp [hoz]
  · simp [hoh]
  · simp [hk]
  · simp [hqks]
  · -- o register = oF + column sum
    refine congrArg some ?_
    ext idx
    -- the masked-weight tile cell, lane-wise
    have hcell : ∀ ii : Fin 64,
        (Tile.select masktile
          (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
            (Broadcast.consSame (Broadcast.consR Broadcast.nil)) scaledqk (Tile.expandDim ⟨1, by simp⟩ mtile)))
          (⟨fun _ : TileIndex [64,64] => some (0:ℝ)⟩ : Tile .real [64,64])).data (ii, idx.1, PUnit.unit)
        = some (if case1Mask s c ii.val idx.1.val
            then case1Weight s Q K M sm_scale c ii.val idx.1.val else 0) := by
      intro ii
      rw [Tile.select_data]
      have hmaskcell : masktile.data (ii, idx.1, PUnit.unit) = decide (case1Mask s c ii.val idx.1.val) := by
        rw [hmask]
      rw [hmaskcell, hsq]
      by_cases hmk : case1Mask s c ii.val idx.1.val
      · simp only [hmk, decide_true, if_true]
        refine score_wReg_cell s Q K M sm_scale c ii idx.1 qtile ktile mtile
          (by simp [hqt]) (fun d => by simp [hqt]) (fun d => by simp [hkt]) ?_
        rw [hmt]; simp only [Tile.mk.injEq]; congr 2; simp [case1MOffset]; ring
      · simp only [hmk, decide_false, Bool.false_eq_true, if_false]
    obtain ⟨ix, ⟨⟩⟩ := idx
    show (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
        (⟨fun idx : TileIndex [64] => some (oF idx.1)⟩ : Tile .real [64])
        (Tile.reduceSumDrop (⟨0, by simp⟩ : Fin [64,64].length) _)).data (ix, PUnit.unit) = _
    rw [Tile.bop_data, Tile.reduceSumDrop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex]
    have hsumGoal : ∀ T : Tile .real [64, 64],
        (∀ ii : Fin 64, T.data (ii, ix, PUnit.unit)
          = some (if case1Mask s c ii.val ix.val
              then case1Weight s Q K M sm_scale c ii.val ix.val else 0)) →
        (∑ x : Fin 64, T.data
            (TileShape.insertAxisIndex [64,64] (⟨0, by simp⟩ : Fin [64,64].length) (ix, PUnit.unit) x))
          = some (case1ColSum s Q K M sm_scale c ix) := by
      intro T hT
      rw [case1ColSum]
      rw [show (fun k : Fin 64 => T.data
            (TileShape.insertAxisIndex [64,64] (⟨0, by simp⟩ : Fin [64,64].length) (ix, PUnit.unit) k))
          = (fun k : Fin 64 => ((if case1Mask s c k.val ix.val
              then case1Weight s Q K M sm_scale c k.val ix.val else 0 : ℝ) : WithBot ℝ))
        from funext (fun k => by
          rw [show TileShape.insertAxisIndex [64,64] (⟨0, by simp⟩ : Fin [64,64].length) (ix, PUnit.unit) k
                = (k, ix, PUnit.unit) from rfl, hT k]; rfl)]
      rw [← WithBot.coe_sum]; rfl
    rw [show some (oF ix + case1ColSum s Q K M sm_scale c ix)
          = NumericDType.real.add (some (oF ix)) (some (case1ColSum s Q K M sm_scale c ix)) from by
        simp only [NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map]]
    refine congrArg _ ?_
    exact hsumGoal _ hcell
  · simp
  · simp

set_option maxHeartbeats 4000000 in
/-- **General loop-body execution + column-sum accumulation.** One iteration of
the general case-1 loop body advances `o` by `case1ColSumG`, advances
`Q_block_ptr` by one `BN`-row block, and `m_ptrs` by `BN`. -/
theorem score_loopBody_evalG
    (s sin : BlockState) (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV N_CTX ROUND_CTX NKV_CTX BN BD swo sws
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn : Nat)
    (hBNpos : 0 < BN) (c : Nat)
    (oF : Fin BN → ℝ)
    (hpids : sin.pids = s.pids) (hmem : sin.mem = s.mem)
    (hundef : ∀ rg o, sin.undef rg o = 0)
    (hsm : sin.regs .nat [] "start_m" = some (Tile.scalar (c * BN)))
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar (s.pids 0)))
    (hoz : sin.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / H)))
    (hoh : sin.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % H)))
    (hqks : sin.regs .real [] "qk_scale" = some (Tile.scalar (some (sm_scale * 1.4426950408889634))))
    (ho : sin.regs .real [BN] "o" = some ⟨fun idx : TileIndex [BN] => some (oF idx.1)⟩)
    (hk : sin.regs .real [BD, BN] "k" = some ⟨fun idx : TileIndex [BD, BN] =>
        some (s.readMem K (case1QKOffsetKG s H H_KV stride_kz stride_kh
          + idx.1.val * stride_kk + (s.pids 0 * BN + idx.2.1.val) * stride_kn))⟩)
    (hQbp : sin.regs .blockPtr [BN, BD] "Q_block_ptr" = some ⟨fun _ : TileIndex [BN, BD] =>
        { region := Q, baseOffset := case1QKOffsetQG s H stride_qz stride_qh, parentShape := [N_CTX, BD],
          blockShape := [BN, BD], strides := [stride_qm, stride_qk], offsets := [c * BN, 0] }⟩)
    (hmp : sin.regs .ptr [BN] "m_ptrs" = some ⟨fun idx : TileIndex [BN] =>
        (M.cast, s.pids 1 * ROUND_CTX + c * BN + idx.1.val)⟩) :
    ∃ sF, stepStmts (attentionScoreCase1LoopBodyG BN BD swo sws N_CTX) sin = some sF
      ∧ sF.pids = s.pids ∧ sF.mem = s.mem ∧ (∀ rg o, sF.undef rg o = 0)
      ∧ sF.regs .nat [] "start_n" = some (Tile.scalar (s.pids 0))
      ∧ sF.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / H))
      ∧ sF.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % H))
      ∧ sF.regs .real [BD, BN] "k" = some ⟨fun idx : TileIndex [BD, BN] =>
          some (s.readMem K (case1QKOffsetKG s H H_KV stride_kz stride_kh
            + idx.1.val * stride_kk + (s.pids 0 * BN + idx.2.1.val) * stride_kn))⟩
      ∧ sF.regs .real [] "qk_scale" = some (Tile.scalar (some (sm_scale * 1.4426950408889634)))
      ∧ sF.regs .real [BN] "o" = some ⟨fun idx : TileIndex [BN] =>
          some (oF idx.1 + case1ColSumG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
            stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws c idx.1)⟩
      ∧ sF.regs .blockPtr [BN, BD] "Q_block_ptr" = some ⟨fun _ : TileIndex [BN, BD] =>
          { region := Q, baseOffset := case1QKOffsetQG s H stride_qz stride_qh, parentShape := [N_CTX, BD],
            blockShape := [BN, BD], strides := [stride_qm, stride_qk], offsets := [c * BN + BN, 0] }⟩
      ∧ sF.regs .ptr [BN] "m_ptrs" = some ⟨fun idx : TileIndex [BN] =>
          (M.cast, s.pids 1 * ROUND_CTX + (c * BN + BN) + idx.1.val)⟩ := by
  set qtile : Tile .real [BN, BD] := ⟨fun idx : TileIndex [BN, BD] =>
      some (s.readMem Q (case1QKOffsetQG s H stride_qz stride_qh + (c * BN + idx.1.val) * stride_qm + idx.2.1.val * stride_qk))⟩ with hqt
  set ktile : Tile .real [BD, BN] := ⟨fun idx : TileIndex [BD, BN] =>
      some (s.readMem K (case1QKOffsetKG s H H_KV stride_kz stride_kh + idx.1.val * stride_kk + (s.pids 0 * BN + idx.2.1.val) * stride_kn))⟩ with hkt
  set mtile : Tile .real [BN] := ⟨fun idx : TileIndex [BN] =>
      some (s.readMem M (s.pids 1 * ROUND_CTX + c * BN + idx.1.val))⟩ with hmt
  set disttile : Tile .nat [BN, BN] := ⟨fun idx : TileIndex [BN, BN] =>
      (((idx.1.val - idx.2.1.val) + c * BN) - s.pids 0 * BN) + swo⟩ with hdistt
  set masktile : Tile .bool [BN, BN] := ⟨fun idx : TileIndex [BN, BN] =>
      decide (case1MaskG s BN BN swo sws c idx.1.val idx.2.1.val)⟩ with hmask
  have hdistt_eq : ∀ idx : TileIndex [BN, BN], disttile.data idx = case1DistG s BN BN swo c idx.1.val idx.2.1.val := by
    intro idx; simp [hdistt, case1DistG]
  have hmask_eq : ⟨fun idx : TileIndex [BN, BN] =>
      (decide (0 ≤ disttile.data idx) && decide (disttile.data idx < sws))⟩ = masktile := by
    rw [hmask]; refine congrArg _ ?_; ext idx
    simp [hdistt_eq, case1MaskG, decide_eq_decide]
  have hrm : sin.readMem = s.readMem := by funext rg o; simp [BlockState.readMem, hmem]
  have hqEval : ∀ st : BlockState,
      st.regs .blockPtr [BN,BD] "Q_block_ptr" = some ⟨fun _ : TileIndex [BN, BD] =>
        { region := Q, baseOffset := case1QKOffsetQG s H stride_qz stride_qh, parentShape := [N_CTX, BD],
          blockShape := [BN, BD], strides := [stride_qm, stride_qk], offsets := [c * BN, 0] }⟩ →
      st.readMem = s.readMem →
      evalOp (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BN,BD] "Q_block_ptr") []) MaskOpt.none) st
        = some qtile := by
    intro st hq hrm'
    rw [load_blockPtr_Q_eval Q (case1QKOffsetQG s H stride_qz stride_qh) N_CTX BD BN BD stride_qm stride_qk (c * BN)
      (Op.ref .blockPtr [BN, BD] "Q_block_ptr") _ (by rw [evalOp_ref]; exact hq)]
    rw [hqt]; refine congrArg some ?_; ext idx; simp [hrm']
  unfold attentionScoreCase1LoopBodyG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_m") sin = some (Tile.scalar (c * BN)) from by
      rw [evalOp_ref]; exact hsm))]
  rw [stepStmts.cons_some
    (show stepStmt (Stmt.ifThenElse (Op.constBool Bool.true) _ _) _ = some _ from by
      rw [stepStmt_ifThenElse_true, stepStmts.cons_some (stepStmt_assign_eq_some
        (hqEval _ (by simp [hQbp]) (by funext rg o; simp [BlockState.readMem, hmem]))), stepStmts.nil])]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN] "m_ptrs")) MaskOpt.none) _ = some mtile from by
      rw [load_ptr_none_real_score (Op.ref .ptr [BN] "m_ptrs") _
        (⟨fun idx : TileIndex [BN] => (M.cast, s.pids 1 * ROUND_CTX + c * BN + idx.1.val)⟩ : Tile .ptr [BN])
        (by rw [evalOp_ref]; simp [hmp])]
      rw [hmt]; refine congrArg some ?_; ext idx; simp [BlockState.readMem, hmem]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BN,BN] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BN,BN] => some (0:ℝ)⟩ : Tile .real [BN,BN]) from by
      simp [evalOp_full]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (score_qkRaw_evalG BN BD _ (⟨fun _ : TileIndex [BN,BN] => some (0:ℝ)⟩ : Tile .real [BN,BN]) qtile ktile
      (by simp) (by simp) (by simp [hk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (score_qkScale_evalG BN _
      (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (⟨fun _ : TileIndex [BN,BN] => some (0:ℝ)⟩ : Tile .real [BN,BN]) (Tile.dot [] qtile ktile))
      (sm_scale * 1.4426950408889634) (by simp) (by simp [hqks])))]
  rw [stepStmts.cons_some
    (show stepStmt (Stmt.ifThen (Op.constBool Bool.true) _) _ = some _ from by
      rw [stepStmt_ifThen_true, stepStmts.cons_some (stepStmt_assign_eq_some
        (score_dist_evalG BN swo _ c (s.pids 0) (by simp [hsm]) (by simp [hsn])))]
      rw [stepStmts.cons_some
        (show stepStmt (Stmt.ifThenElse (Op.constBool Bool.false) _ _) _ = some _ from by
          rw [stepStmt_ifThenElse_false, stepStmts.cons_some (stepStmt_assign_eq_some
            ((score_mask_evalG BN sws _ disttile (by simp [hdistt])).trans (congrArg some hmask_eq)))]
          exact stepStmts.nil)]
      exact stepStmts.nil)]
  set scaledqk : Tile .real [BN, BN] := Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (⟨fun _ : TileIndex [BN,BN] => some (0:ℝ)⟩ : Tile .real [BN,BN]) (Tile.dot [] qtile ktile))
      (Tile.scalar (some (sm_scale * 1.4426950408889634))) with hsq
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (score_qk2_evalG BN _ scaledqk mtile (by simp [hsq]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (score_p_evalG BN _ (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        scaledqk (Tile.expandDim ⟨1, by simp⟩ mtile)) (by simp)))]
  rw [stepStmts.cons_some
    (show stepStmt (Stmt.ifThen (Op.constBool Bool.true) _) _ = some _ from by
      rw [stepStmt_ifThen_true, stepStmts.cons_some (stepStmt_assign_eq_some
        (score_pwhere_evalG BN _ masktile
          (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
            (Broadcast.consSame (Broadcast.consR Broadcast.nil)) scaledqk (Tile.expandDim ⟨1, by simp⟩ mtile)))
          (by simp) (by simp)))]
      exact stepStmts.nil)]
  rw [stepStmts.cons_some (stepStmt_ifThen_boolNot_true _ _)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (score_oAcc_evalG BN _ (⟨fun idx : TileIndex [BN] => some (oF idx.1)⟩ : Tile .real [BN])
      (Tile.select masktile
        (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
          (Broadcast.consSame (Broadcast.consR Broadcast.nil)) scaledqk (Tile.expandDim ⟨1, by simp⟩ mtile)))
        (⟨fun _ : TileIndex [BN,BN] => some (0:ℝ)⟩ : Tile .real [BN,BN]))
      (by simp [ho]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BN,BD] "Q_block_ptr") [BN, (0:Nat)]) _
        = some (⟨fun _ : TileIndex [BN, BD] =>
            { region := Q, baseOffset := case1QKOffsetQG s H stride_qz stride_qh, parentShape := [N_CTX, BD],
              blockShape := [BN, BD], strides := [stride_qm, stride_qk], offsets := [c * BN + BN, 0] }⟩
            : Tile .blockPtr [BN,BD]) from by
      rw [advanceBlockPtr_eval]; simp [hQbp]
      funext i; congr 1 <;> omega))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BN] "m_ptrs") (Op.constNat BN)) _
        = some (⟨fun idx : TileIndex [BN] =>
            (M.cast, s.pids 1 * ROUND_CTX + (c * BN + BN) + idx.1.val)⟩ : Tile .ptr [BN]) from by
      rw [evalOp_ptrAdd']; simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff, hmp,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      · rfl
      · simp only [Tile.ptrAdd_data, Tile.scalar, Broadcast.rightIndex, Broadcast.leftIndex]
        omega))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hpids]
  · funext rg o; simp only [BlockState.setReg_mem]; rw [hmem]
  · intro rg o; simp [hundef]
  · simp [hsn]
  · simp [hoz]
  · simp [hoh]
  · simp [hk]
  · simp [hqks]
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    have hcell : ∀ ii : Fin BN,
        (Tile.select masktile
          (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
            (Broadcast.consSame (Broadcast.consR Broadcast.nil)) scaledqk (Tile.expandDim ⟨1, by simp⟩ mtile)))
          (⟨fun _ : TileIndex [BN,BN] => some (0:ℝ)⟩ : Tile .real [BN,BN])).data (ii, idx.1, PUnit.unit)
        = some (if case1MaskG s BN BN swo sws c ii.val idx.1.val
            then case1WeightG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
              stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn c ii.val idx.1.val else 0) := by
      intro ii
      rw [Tile.select_data]
      have hmaskcell : masktile.data (ii, idx.1, PUnit.unit) = decide (case1MaskG s BN BN swo sws c ii.val idx.1.val) := by
        rw [hmask]
      rw [hmaskcell, hsq]
      by_cases hmk : case1MaskG s BN BN swo sws c ii.val idx.1.val
      · simp only [hmk, decide_true, if_true]
        refine score_wReg_cellG s Q K M sm_scale H H_KV ROUND_CTX BN BD
          stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn c ii idx.1 qtile ktile mtile
          (fun d => by simp [hqt]) (fun d => by simp [hkt]) ?_
        rw [hmt]; simp only [Tile.mk.injEq]; congr 2; simp only [case1MOffsetG]; omega
      · simp only [hmk, decide_false, Bool.false_eq_true, if_false]
    obtain ⟨ix, ⟨⟩⟩ := idx
    show (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
        (⟨fun idx : TileIndex [BN] => some (oF idx.1)⟩ : Tile .real [BN])
        (Tile.reduceSumDrop (⟨0, by simp⟩ : Fin [BN,BN].length) _)).data (ix, PUnit.unit) = _
    rw [Tile.bop_data, Tile.reduceSumDrop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex]
    have hsumGoal : ∀ T : Tile .real [BN, BN],
        (∀ ii : Fin BN, T.data (ii, ix, PUnit.unit)
          = some (if case1MaskG s BN BN swo sws c ii.val ix.val
              then case1WeightG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
                stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn c ii.val ix.val else 0)) →
        (∑ x : Fin BN, T.data
            (TileShape.insertAxisIndex [BN,BN] (⟨0, by simp⟩ : Fin [BN,BN].length) (ix, PUnit.unit) x))
          = some (case1ColSumG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
              stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws c ix) := by
      intro T hT
      rw [case1ColSumG]
      rw [show (fun k : Fin BN => T.data
            (TileShape.insertAxisIndex [BN,BN] (⟨0, by simp⟩ : Fin [BN,BN].length) (ix, PUnit.unit) k))
          = (fun k : Fin BN => ((if case1MaskG s BN BN swo sws c k.val ix.val
              then case1WeightG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
                stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn c k.val ix.val else 0 : ℝ) : WithBot ℝ))
        from funext (fun k => by
          rw [show TileShape.insertAxisIndex [BN,BN] (⟨0, by simp⟩ : Fin [BN,BN].length) (ix, PUnit.unit) k
                = (k, ix, PUnit.unit) from rfl, hT k]; rfl)]
      rw [← WithBot.coe_sum]; rfl
    rw [show some (oF ix + case1ColSumG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
            stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws c ix)
          = NumericDType.real.add (some (oF ix)) (some (case1ColSumG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
            stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws c ix)) from by
        simp only [NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map]]
    refine congrArg _ ?_
    exact hsumGoal _ hcell
  · simp
  · simp

theorem attention_score_case1_loopBody_check
    (Q K M Out : RegionName) (sm_scale : ℝ) :
    (attention_score_kernel Q K M Out
        32768 8192 64 1 32768 8192 64 1 512 128 1
        2 4 4 128 128 128 0 64 64 64 64 sm_scale
        Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel.body.drop 16
      = Stmt.forRangeDyn "start_m" (Op.ref .nat [] "lo") (Op.ref .nat [] "hi")
          (Op.constNat 64) attentionScoreCase1LoopBody
        :: (attention_score_kernel Q K M Out
            32768 8192 64 1 32768 8192 64 1 512 128 1
            2 4 4 128 128 128 0 64 64 64 64 sm_scale
            Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel.body.drop 17 := by
  rfl

/-- The 16 preLoop statements of the case-1 elaborated body. -/
def attentionScoreCase1PreLoop (Q K M : RegionName) (sm_scale : ℝ) : List Stmt :=
  [Stmt.assign TileDType.nat [] "start_n" (Op.programId 0),
    Stmt.assign TileDType.nat [] "off_hz" (Op.programId 1),
    Stmt.assign TileDType.nat [] "off_z"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat 4)),
    Stmt.assign TileDType.nat [] "off_h"
      (Op.mod IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat 4)),
    Stmt.assign TileDType.nat [] "off_hkv"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h")
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat 4) (Op.constNat 4))),
    Stmt.assign TileDType.nat [] "q_offset"
      (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat 32768))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h") (Op.constNat 8192))),
    Stmt.assign TileDType.nat [] "k_offset"
      (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat 32768))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hkv") (Op.constNat 8192))),
    Stmt.assign TileDType.ptr [64] "m_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase M)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat 128))
          (Op.arange 64))),
    Stmt.assign TileDType.real [64] "o" (Op.full [64] (Op.const 0)),
    Stmt.assign TileDType.blockPtr [64, 64] "Q_block_ptr"
      (Op.makeBlockPtrDyn Q (Op.ref TileDType.nat [] "q_offset") [128, 64] [64, 64] [64, 1] [0, 0]),
    Stmt.assign TileDType.blockPtr [64, 64] "K_block_ptr"
      (Op.makeBlockPtrDynOffsets K (Op.ref TileDType.nat [] "k_offset") [64, 128] [64, 64] [1, 64]
        [Op.constNat 0,
          Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_n") (Op.constNat 64)]),
    Stmt.ifThenElse (Op.constBool Bool.true)
      [Stmt.assign TileDType.real [64, 64] "k"
          (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "K_block_ptr") [])
            MaskOpt.none)]
      [Stmt.assign TileDType.real [64, 64] "k"
          (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [64, 64] "K_block_ptr") [0, 1])
            MaskOpt.none)],
    Stmt.assign TileDType.nat [] "lo" (Op.constNat 0),
    Stmt.assign TileDType.nat [] "hi" (Op.constNat 128),
    Stmt.assign TileDType.real [] "qk_scale" (Op.const sm_scale),
    Stmt.assign TileDType.real [] "qk_scale"
      (Op.mul NumericDType.real Broadcast.nil (Op.ref TileDType.real [] "qk_scale")
        (Op.const 1.4426950408889634))]

theorem attentionScoreCase1PreLoop_check (Q K M Out : RegionName) (sm_scale : ℝ) :
    (attention_score_kernel Q K M Out
        32768 8192 64 1 32768 8192 64 1 512 128 1
        2 4 4 128 128 128 0 64 64 64 64 sm_scale
        Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel.body.take 16
      = attentionScoreCase1PreLoop Q K M sm_scale := by
  rfl

/-- The 4 post-loop statements (`o_offset`, `o_range`, `o_ptrs`, masked store). -/
def attentionScoreCase1Post (Out : RegionName) : List Stmt :=
  [Stmt.assign TileDType.nat [] "o_offset"
      (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat 512))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h") (Op.constNat 128))),
    Stmt.assign TileDType.nat [64] "o_range"
      (Op.add NumericDType.nat Broadcast.scalarR (Op.arange 64)
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_n") (Op.constNat 64))),
    Stmt.assign TileDType.ptr [64] "o_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add NumericDType.nat Broadcast.scalarL (Op.ref TileDType.nat [] "o_offset")
          (Op.ref TileDType.nat [64] "o_range"))),
    Stmt.store TileDType.real [64] (MemAccess.ptr (Op.ref TileDType.ptr [64] "o_ptrs"))
      (Op.ref TileDType.real [64] "o")
      (MaskOpt.mask
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [64] "o_range")
          (Op.constNat 128)))]

theorem attentionScoreCase1Post_check (Q K M Out : RegionName) (sm_scale : ℝ) :
    (attention_score_kernel Q K M Out
        32768 8192 64 1 32768 8192 64 1 512 128 1
        2 4 4 128 128 128 0 64 64 64 64 sm_scale
        Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel.body.drop 17
      = attentionScoreCase1Post Out := by
  rfl

/-- **General case-1 preLoop** (16 statements), parameterized over the kernel
dims/strides.  The pinned `attentionScoreCase1PreLoop` is the test-shape instance. -/
def attentionScoreCase1PreLoopG (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk : Nat) :
    List Stmt :=
  [Stmt.assign TileDType.nat [] "start_n" (Op.programId 0),
    Stmt.assign TileDType.nat [] "off_hz" (Op.programId 1),
    Stmt.assign TileDType.nat [] "off_z"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat H)),
    Stmt.assign TileDType.nat [] "off_h"
      (Op.mod IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat H)),
    Stmt.assign TileDType.nat [] "off_hkv"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h")
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat H) (Op.constNat H_KV))),
    Stmt.assign TileDType.nat [] "q_offset"
      (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat stride_qz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h") (Op.constNat stride_qh))),
    Stmt.assign TileDType.nat [] "k_offset"
      (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat stride_kz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hkv") (Op.constNat stride_kh))),
    Stmt.assign TileDType.ptr [BN] "m_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase M)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat ROUND_CTX))
          (Op.arange BN))),
    Stmt.assign TileDType.real [BN] "o" (Op.full [BN] (Op.const 0)),
    Stmt.assign TileDType.blockPtr [BN, BD] "Q_block_ptr"
      (Op.makeBlockPtrDyn Q (Op.ref TileDType.nat [] "q_offset") [N_CTX, BD] [BN, BD]
        [stride_qm, stride_qk] [0, 0]),
    Stmt.assign TileDType.blockPtr [BD, BN] "K_block_ptr"
      (Op.makeBlockPtrDynOffsets K (Op.ref TileDType.nat [] "k_offset") [BD, NKV_CTX] [BD, BN]
        [stride_kk, stride_kn]
        [Op.constNat 0,
          Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_n") (Op.constNat BN)]),
    Stmt.ifThenElse (Op.constBool Bool.true)
      [Stmt.assign TileDType.real [BD, BN] "k"
          (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BD, BN] "K_block_ptr") [])
            MaskOpt.none)]
      [Stmt.assign TileDType.real [BD, BN] "k"
          (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BD, BN] "K_block_ptr") [0, 1])
            MaskOpt.none)],
    Stmt.assign TileDType.nat [] "lo" (Op.constNat 0),
    Stmt.assign TileDType.nat [] "hi" (Op.constNat ROUND_CTX),
    Stmt.assign TileDType.real [] "qk_scale" (Op.const sm_scale),
    Stmt.assign TileDType.real [] "qk_scale"
      (Op.mul NumericDType.real Broadcast.nil (Op.ref TileDType.real [] "qk_scale")
        (Op.const 1.4426950408889634))]

/-- **General case-1 post-loop store** (4 statements), parameterized over the
output strides / block size / `NKV_CTX`. -/
def attentionScoreCase1PostG (Out : RegionName) (BN NKV_CTX stride_oz stride_oh : Nat) : List Stmt :=
  [Stmt.assign TileDType.nat [] "o_offset"
      (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat stride_oz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h") (Op.constNat stride_oh))),
    Stmt.assign TileDType.nat [BN] "o_range"
      (Op.add NumericDType.nat Broadcast.scalarR (Op.arange BN)
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_n") (Op.constNat BN))),
    Stmt.assign TileDType.ptr [BN] "o_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add NumericDType.nat Broadcast.scalarL (Op.ref TileDType.nat [] "o_offset")
          (Op.ref TileDType.nat [BN] "o_range"))),
    Stmt.store TileDType.real [BN] (MemAccess.ptr (Op.ref TileDType.ptr [BN] "o_ptrs"))
      (Op.ref TileDType.real [BN] "o")
      (MaskOpt.mask
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BN] "o_range")
          (Op.constNat NKV_CTX)))]

theorem scalarBop {dtype : TileDType}
    (f : TileCarrier dtype → TileCarrier dtype → TileCarrier dtype)
    (a b : TileCarrier dtype) :
    Tile.bop f Broadcast.nil (Tile.scalar a) (Tile.scalar b) = Tile.scalar (f a b) := rfl

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 8000 in
/-- **PreLoop execution.** The 16 deterministic preLoop statements step a clean
state to the loop-entry state with all register readbacks `score_loopBody_eval`
needs (for the first iteration `c = 0`: `start_m` is set by the loop itself). -/
theorem score_preLoop_eval
    (s : BlockState) (Q K M : RegionName) (sm_scale : ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (attentionScoreCase1PreLoop Q K M sm_scale) s = some s0
      ∧ s0.pids = s.pids ∧ s0.mem = s.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "start_n" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / 4))
      ∧ s0.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % 4))
      ∧ s0.regs .real [] "qk_scale" = some (Tile.scalar (some (sm_scale * 1.4426950408889634)))
      ∧ s0.regs .real [64] "o" = some ⟨fun _ : TileIndex [64] => some (0 : ℝ)⟩
      ∧ s0.regs .real [64, 64] "k" = some ⟨fun idx : TileIndex [64, 64] =>
          some (s.readMem K (case1QKOffset s + idx.1.val + (s.pids 0 * 64 + idx.2.1.val) * 64))⟩
      ∧ s0.regs .blockPtr [64, 64] "Q_block_ptr" = some ⟨fun _ : TileIndex [64, 64] =>
          { region := Q, baseOffset := case1QKOffset s, parentShape := [128, 64],
            blockShape := [64, 64], strides := [64, 1], offsets := [0, 0] }⟩
      ∧ s0.regs .ptr [64] "m_ptrs" = some ⟨fun idx : TileIndex [64] =>
          (M.cast, s.pids 1 * 128 + idx.1.val)⟩
      ∧ s0.regs .nat [] "lo" = some (Tile.scalar 0)
      ∧ s0.regs .nat [] "hi" = some (Tile.scalar 128) := by
  unfold attentionScoreCase1PreLoop
  -- start_n = programId 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  -- off_hz = programId 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  -- off_z = off_hz / 4
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 4)) _
        = some (Tile.scalar (s.pids 1 / 4)) from by
      rw [evalOp_floorDiv']
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- off_h = off_hz % 4
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 4)) _
        = some (Tile.scalar (s.pids 1 % 4)) from by
      rw [evalOp_mod']
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- off_hkv = off_h / (4/4) = off_h
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_h")
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat 4) (Op.constNat 4))) _
        = some (Tile.scalar (s.pids 1 % 4)) from by
      rw [evalOp_floorDiv', evalOp_floorDiv']
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      rw [scalarBop, scalarBop]
      refine congrArg some (congrArg Tile.scalar ?_)
      show s.pids 1 % 4 / (4 / 4) = s.pids 1 % 4
      norm_num))]
  -- q_offset = off_z*32768 + off_h*8192 = case1QKOffset
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat 32768))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat 8192))) _
        = some (Tile.scalar (case1QKOffset s)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_constNat, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- k_offset = off_z*32768 + off_hkv*8192 = case1QKOffset
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat 32768))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_hkv") (Op.constNat 8192))) _
        = some (Tile.scalar (case1QKOffset s)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_constNat, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- m_ptrs = M + off_hz*128 + arange 64
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase M)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 128))
          (Op.arange 64))) _
        = some (⟨fun idx : TileIndex [64] => (M.cast, s.pids 1 * 128 + idx.1.val)⟩ : Tile .ptr [64]) from by
      rw [evalOp_ptrAdd', evalOp_add, evalOp_mul]
      simp only [evalOp_ptrBase', evalOp_constNat, evalOp_ref, evalOp_arange,
        BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_some]
      refine congrArg some ?_; ext idx
      · rfl
      · simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Tile.vec, Broadcast.leftIndex,
          Broadcast.rightIndex, NumericDType.add, NumericDType.mul, BlockState.setReg_pids]
        omega))]
  -- o = full [64] 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [64] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [64] => some (0 : ℝ)⟩ : Tile .real [64]) from by
      simp [evalOp_full]))]
  -- Q_block_ptr = makeBlockPtrDyn Q q_offset [128,64] [64,64] [64,1] [0,0]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtrDyn_eval Q (Op.ref .nat [] "q_offset") [128, 64] [64, 64] [64, 1] [0, 0] _
      (case1QKOffset s) (by rw [evalOp_ref]; simp)))]
  -- K_block_ptr = makeBlockPtrDynOffsets ...
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.makeBlockPtrDynOffsets K (Op.ref .nat [] "k_offset") [64, 128] [64, 64] [1, 64]
        [Op.constNat 0, Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 64)]) _
        = some (⟨fun _ : TileIndex [64, 64] =>
            { region := K, baseOffset := case1QKOffset s, parentShape := [64, 128],
              blockShape := [64, 64], strides := [1, 64], offsets := [0, s.pids 0 * 64] }⟩
            : Tile .blockPtr [64, 64]) from by
      rw [makeBlockPtr2_eval]
      simp only [evalOp_ref, evalOp_constNat, evalOp_mul, BlockState.setReg_same,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind,
        Option.bind_some, List.mapM_cons, List.mapM_nil, scalarBop]
      refine congrArg some ?_; ext idx <;> rfl))]
  -- ifThenElse true [k = load K_block_ptr] _
  rw [stepStmts.cons_some
    (show stepStmt (Stmt.ifThenElse (Op.constBool Bool.true) _ _) _ = some _ from by
      rw [stepStmt_ifThenElse_true, stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [64,64] "K_block_ptr") []) MaskOpt.none) _
            = some (⟨fun idx : TileIndex [64, 64] =>
                some (s.readMem K (case1QKOffset s + idx.1.val + (s.pids 0 * 64 + idx.2.1.val) * 64))⟩
                : Tile .real [64, 64]) from by
          rw [load_blockPtr_K_eval K (case1QKOffset s) 64 128 64 64 1 64 (s.pids 0 * 64)
            (Op.ref .blockPtr [64, 64] "K_block_ptr") _ (by rw [evalOp_ref]; simp)]
          refine congrArg some ?_; ext idx
          simp [BlockState.readMem, BlockState.setReg_mem]))]
      exact stepStmts.nil)]
  -- lo = 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.constNat 0) _ = _ from evalOp_constNat 0 _))]
  -- hi = 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.constNat 128) _ = _ from evalOp_constNat 128 _))]
  -- qk_scale = sm_scale
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.const sm_scale) _ = _ from evalOp_const sm_scale _))]
  -- qk_scale = qk_scale * log2e
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul NumericDType.real Broadcast.nil (Op.ref .real [] "qk_scale")
        (Op.const 1.4426950408889634)) _
        = some (Tile.scalar (some (sm_scale * 1.4426950408889634))) from by
      rw [evalOp_mul]
      simp only [evalOp_ref, evalOp_const, BlockState.setReg_same, Option.bind_eq_bind,
        Option.bind_some, scalarBop]
      refine congrArg some (congrArg Tile.scalar ?_)
      simp only [NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext rg o; simp [BlockState.setReg_mem]
  · intro rg o; simp [hundef]
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp

set_option maxHeartbeats 2000000 in
/-- **Loop execution (2 iterations).** From the loop-entry state `s0` (with the
preLoop readbacks), the `forRangeDyn` over `[0,128)` step `64` runs exactly two
iterations, leaving `o[j] = case1OutClosedForm s Q K M sm_scale j` and preserving
`pids`/`mem`, `off_z`/`off_h`/`start_n`. -/
theorem score_loop_eval
    (s s0 : BlockState) (Q K M : RegionName) (sm_scale : ℝ)
    (hpids : s0.pids = s.pids) (hmem : s0.mem = s.mem) (hundef : ∀ rg o, s0.undef rg o = 0)
    (hsn : s0.regs .nat [] "start_n" = some (Tile.scalar (s.pids 0)))
    (hoz : s0.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / 4)))
    (hoh : s0.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % 4)))
    (hqks : s0.regs .real [] "qk_scale" = some (Tile.scalar (some (sm_scale * 1.4426950408889634))))
    (ho : s0.regs .real [64] "o" = some ⟨fun _ : TileIndex [64] => some (0 : ℝ)⟩)
    (hk : s0.regs .real [64, 64] "k" = some ⟨fun idx : TileIndex [64, 64] =>
        some (s.readMem K (case1QKOffset s + idx.1.val + (s.pids 0 * 64 + idx.2.1.val) * 64))⟩)
    (hQbp : s0.regs .blockPtr [64, 64] "Q_block_ptr" = some ⟨fun _ : TileIndex [64, 64] =>
        { region := Q, baseOffset := case1QKOffset s, parentShape := [128, 64],
          blockShape := [64, 64], strides := [64, 1], offsets := [0, 0] }⟩)
    (hmp : s0.regs .ptr [64] "m_ptrs" = some ⟨fun idx : TileIndex [64] =>
        (M.cast, s.pids 1 * 128 + idx.1.val)⟩)
    (hlo : s0.regs .nat [] "lo" = some (Tile.scalar 0))
    (hhi : s0.regs .nat [] "hi" = some (Tile.scalar 128)) :
    ∃ sL, stepStmt (Stmt.forRangeDyn "start_m" (Op.ref .nat [] "lo") (Op.ref .nat [] "hi")
        (Op.constNat 64) attentionScoreCase1LoopBody) s0 = some sL
      ∧ sL.pids = s.pids ∧ sL.mem = s.mem
      ∧ sL.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / 4))
      ∧ sL.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % 4))
      ∧ sL.regs .nat [] "start_n" = some (Tile.scalar (s.pids 0))
      ∧ sL.regs .real [64] "o" = some ⟨fun idx : TileIndex [64] =>
          some (case1OutClosedForm s Q K M sm_scale idx.1)⟩ := by
  -- unfold the dynamic for-loop: lo = 0, hi = 128, step = 64
  rw [stepForRangeAux.forRangeDyn_unfold]
  rw [show evalOp (Op.ref .nat [] "lo") s0 = some (Tile.scalar 0) from by rw [evalOp_ref]; exact hlo]
  rw [show evalOp (Op.ref .nat [] "hi") s0 = some (Tile.scalar 128) from by rw [evalOp_ref]; exact hhi]
  rw [show evalOp (Op.constNat 64) s0 = some (Tile.scalar 64) from evalOp_constNat 64 s0]
  simp only [Option.bind_eq_bind, Option.bind_some, Tile.scalar_data]
  -- iteration c = 0 (cur = 0)
  rw [stepForRangeAux.step_lt (by norm_num) (by norm_num : (0 : Nat) < 128)]
  obtain ⟨s1, hstep1, hpids1, hmem1, hundef1, hsn1, hoz1, hoh1, hk1, hqks1, ho1, hQbp1, hmp1⟩ :=
    score_loopBody_eval s (s0.setReg "start_m" .nat [] (Tile.scalar 0)) Q K M sm_scale 0
      (fun _ => 0)
      (by simp only [BlockState.setReg_pids]; exact hpids)
      (by funext rg o; simp only [BlockState.setReg_mem]; rw [hmem])
      (by intro rg o; simp only [BlockState.setReg_undef]; exact hundef rg o)
      (by rw [BlockState.setReg_same])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsn)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoz)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoh)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hqks)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact ho)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hk)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hQbp)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmp)
  rw [hstep1]
  simp only [Option.bind_eq_bind, Option.bind_some]
  -- iteration c = 1 (cur = 64)
  rw [stepForRangeAux.step_lt (by norm_num) (by norm_num : (64 : Nat) < 128)]
  obtain ⟨s2, hstep2, hpids2, hmem2, hundef2, hsn2, hoz2, hoh2, hk2, hqks2, ho2, hQbp2, hmp2⟩ :=
    score_loopBody_eval s (s1.setReg "start_m" .nat [] (Tile.scalar 64)) Q K M sm_scale 1
      (fun j => case1ColSum s Q K M sm_scale 0 j)
      (by simp only [BlockState.setReg_pids]; exact hpids1)
      (by funext rg o; simp only [BlockState.setReg_mem]; rw [hmem1])
      (by intro rg o; simp only [BlockState.setReg_undef]; exact hundef1 rg o)
      (by rw [BlockState.setReg_same])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsn1)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoz1)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoh1)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hqks1)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
          rw [ho1]; refine congrArg some ?_; ext idx; simp)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hk1)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
          rw [hQbp1])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
          rw [hmp1])
  rw [hstep2]
  simp only [Option.bind_eq_bind, Option.bind_some]
  -- cur = 128 ≥ stop ⇒ terminate
  rw [stepForRangeAux.step_ge (by norm_num) (by norm_num : (128 : Nat) ≤ 64 + 64)]
  refine ⟨s2, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact hpids2
  · exact hmem2
  · exact hoz2
  · exact hoh2
  · exact hsn2
  · rw [ho2]; refine congrArg some ?_; ext idx
    show some (case1ColSum s Q K M sm_scale 0 idx.1 + case1ColSum s Q K M sm_scale 1 idx.1)
      = some (case1OutClosedForm s Q K M sm_scale idx.1)
    rw [case1OutClosedForm_eq_colSum]

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 8000 in
/-- **General preLoop execution.** The 16 deterministic preLoop statements step a
clean state to the loop-entry state with all register readbacks `score_loopBody_evalG`
needs. -/
theorem score_preLoop_evalG
    (s : BlockState) (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (attentionScoreCase1PreLoopG Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk) s = some s0
      ∧ s0.pids = s.pids ∧ s0.mem = s.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "start_n" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / H))
      ∧ s0.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % H))
      ∧ s0.regs .real [] "qk_scale" = some (Tile.scalar (some (sm_scale * 1.4426950408889634)))
      ∧ s0.regs .real [BN] "o" = some ⟨fun _ : TileIndex [BN] => some (0 : ℝ)⟩
      ∧ s0.regs .real [BD, BN] "k" = some ⟨fun idx : TileIndex [BD, BN] =>
          some (s.readMem K (case1QKOffsetKG s H H_KV stride_kz stride_kh
            + idx.1.val * stride_kk + (s.pids 0 * BN + idx.2.1.val) * stride_kn))⟩
      ∧ s0.regs .blockPtr [BN, BD] "Q_block_ptr" = some ⟨fun _ : TileIndex [BN, BD] =>
          { region := Q, baseOffset := case1QKOffsetQG s H stride_qz stride_qh, parentShape := [N_CTX, BD],
            blockShape := [BN, BD], strides := [stride_qm, stride_qk], offsets := [0, 0] }⟩
      ∧ s0.regs .ptr [BN] "m_ptrs" = some ⟨fun idx : TileIndex [BN] =>
          (M.cast, s.pids 1 * ROUND_CTX + idx.1.val)⟩
      ∧ s0.regs .nat [] "lo" = some (Tile.scalar 0)
      ∧ s0.regs .nat [] "hi" = some (Tile.scalar ROUND_CTX) := by
  unfold attentionScoreCase1PreLoopG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)) _
        = some (Tile.scalar (s.pids 1 / H)) from by
      rw [evalOp_floorDiv']
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)) _
        = some (Tile.scalar (s.pids 1 % H)) from by
      rw [evalOp_mod']
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_h")
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat H) (Op.constNat H_KV))) _
        = some (Tile.scalar ((s.pids 1 % H) / (H / H_KV))) from by
      rw [evalOp_floorDiv', evalOp_floorDiv']
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      rw [scalarBop, scalarBop]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat stride_qz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat stride_qh))) _
        = some (Tile.scalar (case1QKOffsetQG s H stride_qz stride_qh)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_constNat, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat stride_kz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_hkv") (Op.constNat stride_kh))) _
        = some (Tile.scalar (case1QKOffsetKG s H H_KV stride_kz stride_kh)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_constNat, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase M)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat ROUND_CTX))
          (Op.arange BN))) _
        = some (⟨fun idx : TileIndex [BN] => (M.cast, s.pids 1 * ROUND_CTX + idx.1.val)⟩ : Tile .ptr [BN]) from by
      rw [evalOp_ptrAdd', evalOp_add, evalOp_mul]
      simp only [evalOp_ptrBase', evalOp_constNat, evalOp_ref, evalOp_arange,
        BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_some]
      refine congrArg some ?_; ext idx
      · rfl
      · simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Tile.vec, Broadcast.leftIndex,
          Broadcast.rightIndex, NumericDType.add, NumericDType.mul, BlockState.setReg_pids]
        omega))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BN] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BN] => some (0 : ℝ)⟩ : Tile .real [BN]) from by
      simp [evalOp_full]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtrDyn_eval Q (Op.ref .nat [] "q_offset") [N_CTX, BD] [BN, BD] [stride_qm, stride_qk] [0, 0] _
      (case1QKOffsetQG s H stride_qz stride_qh) (by rw [evalOp_ref]; simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.makeBlockPtrDynOffsets K (Op.ref .nat [] "k_offset") [BD, NKV_CTX] [BD, BN] [stride_kk, stride_kn]
        [Op.constNat 0, Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BN)]) _
        = some (⟨fun _ : TileIndex [BD, BN] =>
            { region := K, baseOffset := case1QKOffsetKG s H H_KV stride_kz stride_kh, parentShape := [BD, NKV_CTX],
              blockShape := [BD, BN], strides := [stride_kk, stride_kn], offsets := [0, s.pids 0 * BN] }⟩
            : Tile .blockPtr [BD, BN]) from by
      rw [makeBlockPtr2_eval]
      simp only [evalOp_ref, evalOp_constNat, evalOp_mul, BlockState.setReg_same,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind,
        Option.bind_some, List.mapM_cons, List.mapM_nil, scalarBop]
      refine congrArg some ?_; ext idx <;> rfl))]
  rw [stepStmts.cons_some
    (show stepStmt (Stmt.ifThenElse (Op.constBool Bool.true) _ _) _ = some _ from by
      rw [stepStmt_ifThenElse_true, stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BD,BN] "K_block_ptr") []) MaskOpt.none) _
            = some (⟨fun idx : TileIndex [BD, BN] =>
                some (s.readMem K (case1QKOffsetKG s H H_KV stride_kz stride_kh
                  + idx.1.val * stride_kk + (s.pids 0 * BN + idx.2.1.val) * stride_kn))⟩
                : Tile .real [BD, BN]) from by
          rw [load_blockPtr_K_eval K (case1QKOffsetKG s H H_KV stride_kz stride_kh) BD NKV_CTX BD BN stride_kk stride_kn (s.pids 0 * BN)
            (Op.ref .blockPtr [BD, BN] "K_block_ptr") _ (by rw [evalOp_ref]; simp)]
          refine congrArg some ?_; ext idx
          simp [BlockState.readMem, BlockState.setReg_mem]))]
      exact stepStmts.nil)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.constNat 0) _ = _ from evalOp_constNat 0 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.constNat ROUND_CTX) _ = _ from evalOp_constNat ROUND_CTX _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.const sm_scale) _ = _ from evalOp_const sm_scale _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul NumericDType.real Broadcast.nil (Op.ref .real [] "qk_scale")
        (Op.const 1.4426950408889634)) _
        = some (Tile.scalar (some (sm_scale * 1.4426950408889634))) from by
      rw [evalOp_mul]
      simp only [evalOp_ref, evalOp_const, BlockState.setReg_same, Option.bind_eq_bind,
        Option.bind_some, scalarBop]
      refine congrArg some (congrArg Tile.scalar ?_)
      simp only [NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext rg o; simp [BlockState.setReg_mem]
  · intro rg o; simp [hundef]
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp

/-- Bridge: `case1OutClosedFormG` (a `Fin (ROUND_CTX/BN)` sum) equals the
`Finset.range (ROUND_CTX/BN)` partial sum of `case1ColSumG`. -/
theorem case1OutClosedFormG_eq_rangeSum
    (s : BlockState) (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV ROUND_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn
      swo sws : Nat) (j : Fin BN) :
    case1OutClosedFormG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws j
      = (Finset.range (ROUND_CTX / BN)).sum (fun c' =>
          case1ColSumG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
            stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws c' j) := by
  rw [case1OutClosedFormG, Finset.sum_range]

set_option maxHeartbeats 4000000 in
/-- **General loop execution.** Drives the `forRangeDyn` over `[0, ROUND_CTX)`
step `BN` via `forRangeDyn_inv`, leaving `o[j] = case1OutClosedFormG`. -/
theorem score_loop_evalG
    (s s0 : BlockState) (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV N_CTX ROUND_CTX NKV_CTX BN BD swo sws
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn : Nat)
    (hBNpos : 0 < BN) (hdvd : BN ∣ ROUND_CTX)
    (hpids : s0.pids = s.pids) (hmem : s0.mem = s.mem) (hundef : ∀ rg o, s0.undef rg o = 0)
    (hsn : s0.regs .nat [] "start_n" = some (Tile.scalar (s.pids 0)))
    (hoz : s0.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / H)))
    (hoh : s0.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % H)))
    (hqks : s0.regs .real [] "qk_scale" = some (Tile.scalar (some (sm_scale * 1.4426950408889634))))
    (ho : s0.regs .real [BN] "o" = some ⟨fun _ : TileIndex [BN] => some (0 : ℝ)⟩)
    (hk : s0.regs .real [BD, BN] "k" = some ⟨fun idx : TileIndex [BD, BN] =>
        some (s.readMem K (case1QKOffsetKG s H H_KV stride_kz stride_kh
          + idx.1.val * stride_kk + (s.pids 0 * BN + idx.2.1.val) * stride_kn))⟩)
    (hQbp : s0.regs .blockPtr [BN, BD] "Q_block_ptr" = some ⟨fun _ : TileIndex [BN, BD] =>
        { region := Q, baseOffset := case1QKOffsetQG s H stride_qz stride_qh, parentShape := [N_CTX, BD],
          blockShape := [BN, BD], strides := [stride_qm, stride_qk], offsets := [0, 0] }⟩)
    (hmp : s0.regs .ptr [BN] "m_ptrs" = some ⟨fun idx : TileIndex [BN] =>
        (M.cast, s.pids 1 * ROUND_CTX + idx.1.val)⟩)
    (hlo : s0.regs .nat [] "lo" = some (Tile.scalar 0))
    (hhi : s0.regs .nat [] "hi" = some (Tile.scalar ROUND_CTX)) :
    ∃ sL, stepStmt (Stmt.forRangeDyn "start_m" (Op.ref .nat [] "lo") (Op.ref .nat [] "hi")
        (Op.constNat BN) (attentionScoreCase1LoopBodyG BN BD swo sws N_CTX)) s0 = some sL
      ∧ sL.pids = s.pids ∧ sL.mem = s.mem
      ∧ sL.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / H))
      ∧ sL.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % H))
      ∧ sL.regs .nat [] "start_n" = some (Tile.scalar (s.pids 0))
      ∧ sL.regs .real [BN] "o" = some ⟨fun idx : TileIndex [BN] =>
          some (case1OutClosedFormG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
            stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws idx.1)⟩ := by
  -- invariant: after counter m (a multiple of BN, ≤ ROUND_CTX), o = partial range sum.
  set P : Nat → BlockState → Prop := fun m sc =>
    BN ∣ m ∧ m ≤ ROUND_CTX
      ∧ sc.pids = s.pids ∧ sc.mem = s.mem ∧ (∀ rg o, sc.undef rg o = 0)
      ∧ sc.regs .nat [] "start_n" = some (Tile.scalar (s.pids 0))
      ∧ sc.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / H))
      ∧ sc.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % H))
      ∧ sc.regs .real [] "qk_scale" = some (Tile.scalar (some (sm_scale * 1.4426950408889634)))
      ∧ sc.regs .real [BD, BN] "k" = some ⟨fun idx : TileIndex [BD, BN] =>
          some (s.readMem K (case1QKOffsetKG s H H_KV stride_kz stride_kh
            + idx.1.val * stride_kk + (s.pids 0 * BN + idx.2.1.val) * stride_kn))⟩
      ∧ sc.regs .blockPtr [BN, BD] "Q_block_ptr" = some ⟨fun _ : TileIndex [BN, BD] =>
          { region := Q, baseOffset := case1QKOffsetQG s H stride_qz stride_qh, parentShape := [N_CTX, BD],
            blockShape := [BN, BD], strides := [stride_qm, stride_qk], offsets := [m, 0] }⟩
      ∧ sc.regs .ptr [BN] "m_ptrs" = some ⟨fun idx : TileIndex [BN] =>
          (M.cast, s.pids 1 * ROUND_CTX + m + idx.1.val)⟩
      ∧ sc.regs .real [BN] "o" = some ⟨fun idx : TileIndex [BN] =>
          some ((Finset.range (m / BN)).sum (fun c' =>
            case1ColSumG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
              stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws c' idx.1))⟩
    with hP
  have hinit : P 0 s0 := by
    refine ⟨Dvd.intro 0 rfl, Nat.zero_le _, hpids, hmem, hundef, hsn, hoz, hoh, hqks, hk, ?_, hmp, ?_⟩
    · simpa using hQbp
    · rw [ho]; refine congrArg some ?_; ext idx; simp
  have hstepObl : ∀ m sc, m < ROUND_CTX → P m sc →
      ∃ s', stepStmts (attentionScoreCase1LoopBodyG BN BD swo sws N_CTX)
          (sc.setReg "start_m" .nat [] (Tile.scalar m)) = some s' ∧ P (m + BN) s' := by
    intro m sc hm hPm
    obtain ⟨hdvdm, hmle, hpidsm, hmemm, hundefm, hsnm, hozm, hohm, hqksm, hkm, hQbpm, hmpm, hom⟩ := hPm
    -- c = m / BN, with c·BN = m.
    have hcm : (m / BN) * BN = m := Nat.div_mul_cancel hdvdm
    obtain ⟨sF, hbody, hpidsF, hmemF, hundefF, hsnF, hozF, hohF, hkF, hqksF, hoF, hQbpF, hmpF⟩ :=
      score_loopBody_evalG s (sc.setReg "start_m" .nat [] (Tile.scalar m)) Q K M sm_scale
        H H_KV N_CTX ROUND_CTX NKV_CTX BN BD swo sws
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn hBNpos (m / BN)
        (fun j => (Finset.range (m / BN)).sum (fun c' =>
          case1ColSumG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
            stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws c' j))
        (by simp only [BlockState.setReg_pids]; exact hpidsm)
        (by funext rg o; simp only [BlockState.setReg_mem]; rw [hmemm])
        (by intro rg o; simp only [BlockState.setReg_undef]; exact hundefm rg o)
        (by rw [BlockState.setReg_same, hcm])
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsnm)
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hozm)
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hohm)
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hqksm)
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hom)
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hkm)
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hcm]; exact hQbpm)
        (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hcm]; exact hmpm)
    obtain ⟨qq, hqq⟩ := hdvd
    have hnextle : m + BN ≤ ROUND_CTX := by
      rw [← hcm] at hm ⊢
      rw [hqq] at hm ⊢
      have hlt : m / BN < qq := by
        by_contra hc; push_neg at hc
        have : BN * qq ≤ m / BN * BN := by rw [Nat.mul_comm BN qq]; exact Nat.mul_le_mul_right BN hc
        omega
      calc m / BN * BN + BN = BN * (m / BN + 1) := by ring
        _ ≤ BN * qq := Nat.mul_le_mul_left BN hlt
    have hmBN : m / BN * BN = m := hcm
    refine ⟨sF, hbody, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · exact Nat.dvd_add hdvdm (Dvd.intro 1 (by ring))
    · exact hnextle
    · rw [hpidsF]
    · rw [hmemF]
    · exact hundefF
    · exact hsnF
    · exact hozF
    · exact hohF
    · exact hqksF
    · exact hkF
    · rw [hQbpF, hmBN]
    · rw [hmpF]; refine congrArg some ?_; ext idx
      · rfl
      · simp only [hmBN]
    · rw [hoF]; refine congrArg some ?_; ext idx
      have hran : (m + BN) / BN = m / BN + 1 := by
        conv_lhs => rw [← hmBN]
        rw [Nat.add_div_right _ hBNpos, Nat.mul_div_cancel _ hBNpos]
      show some _ = some _
      rw [hran, Finset.sum_range_succ]
  obtain ⟨final, sL, hloop, hfinal, hPL⟩ :=
    forRangeDyn_inv (by rw [evalOp_ref]; exact hlo) (by rw [evalOp_ref]; exact hhi)
      (evalOp_constNat BN s0) (by omega) hinit hstepObl
  obtain ⟨hdvdF, hFle, hpidsL, hmemL, _, hsnL, hozL, hohL, _, _, _, _, hoL⟩ := hPL
  have hfinalEq : final = ROUND_CTX := le_antisymm hFle hfinal
  refine ⟨sL, hloop, hpidsL, hmemL, hozL, hohL, hsnL, ?_⟩
  rw [hoL]; refine congrArg some ?_; ext idx
  show some _ = some _
  rw [hfinalEq, case1OutClosedFormG_eq_rangeSum]

/-- The **genuine** case-1 store offset for output column `i` (the address the
kernel writes): `off_z·512 + off_h·128 + start_n·64 + i`. -/
def case1OutStoreOffset (s : BlockState) (i : Fin 64) : Nat :=
  (s.pids 1 / 4) * 512 + (s.pids 1 % 4) * 128 + (s.pids 0 * 64 + i.val)

/-- The store mask: the o_range `< NKV_CTX = 128`. -/
def case1OutActive (s : BlockState) (i : Fin 64) : Prop := s.pids 0 * 64 + i.val < 128

instance (s : BlockState) (i : Fin 64) : Decidable (case1OutActive s i) := by
  unfold case1OutActive; infer_instance

set_option maxHeartbeats 2000000 in
/-- **Post-loop store readback.** From the post-loop state `sL` (with `o` =
`case1OutClosedForm`, plus `off_z`/`off_h`/`start_n` readbacks), the 4 post
statements (`o_offset`, `o_range`, `o_ptrs`, masked store) write
`case1OutClosedForm` to `Out` at every active column. -/
theorem score_post_eval
    (s sL : BlockState) (Q K M Out : RegionName) (sm_scale : ℝ) (i : Fin 64)
    (hmem : sL.mem = s.mem)
    (hoz : sL.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / 4)))
    (hoh : sL.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % 4)))
    (hsn : sL.regs .nat [] "start_n" = some (Tile.scalar (s.pids 0)))
    (ho : sL.regs .real [64] "o" = some ⟨fun idx : TileIndex [64] =>
        some (case1OutClosedForm s Q K M sm_scale idx.1)⟩) :
    ∃ sF, stepStmts (attentionScoreCase1Post Out) sL = some sF
      ∧ sF.readMem Out (case1OutStoreOffset s i)
        = (if case1OutActive s i then case1OutClosedForm s Q K M sm_scale i
            else s.readMem Out (case1OutStoreOffset s i)) := by
  unfold attentionScoreCase1Post
  -- o_offset = off_z*512 + off_h*128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat 512))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat 128))) _
        = some (Tile.scalar ((s.pids 1 / 4) * 512 + (s.pids 1 % 4) * 128)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_constNat, evalOp_ref, hoz, hoh, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- o_range = arange 64 + start_n*64
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.scalarR (Op.arange 64)
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 64))) _
        = some (⟨fun idx : TileIndex [64] => idx.1.val + s.pids 0 * 64⟩ : Tile .nat [64]) from by
      rw [evalOp_add, evalOp_mul]
      simp only [evalOp_arange, evalOp_constNat, evalOp_ref, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, hsn, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul, Tile.vec_data]))]
  -- o_ptrs = Out + o_offset + o_range
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add NumericDType.nat Broadcast.scalarL (Op.ref .nat [] "o_offset")
          (Op.ref .nat [64] "o_range"))) _
        = some (⟨fun idx : TileIndex [64] =>
            (Out.cast, case1OutStoreOffset s idx.1)⟩ : Tile .ptr [64]) from by
      rw [evalOp_ptrAdd', evalOp_add]
      simp only [evalOp_ptrBase', evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      · rfl
      · simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Broadcast.leftIndex,
          Broadcast.rightIndex, NumericDType.add, case1OutStoreOffset]
        omega))]
  -- masked store: resolve to an explicit masked scatter foldl
  set sP := (((sL.setReg "o_offset" .nat [] (Tile.scalar ((s.pids 1 / 4) * 512 + (s.pids 1 % 4) * 128))).setReg
      "o_range" .nat [64] (⟨fun idx : TileIndex [64] => idx.1.val + s.pids 0 * 64⟩ : Tile .nat [64])).setReg
      "o_ptrs" .ptr [64] (⟨fun idx : TileIndex [64] => (Out.cast, case1OutStoreOffset s idx.1)⟩ : Tile .ptr [64]))
    with hsPdef
  set offFn : TileIndex [64] → Nat := fun idx => case1OutStoreOffset s idx.1 with hoffFn
  set valFn : TileIndex [64] → ℝ := fun idx => case1OutClosedForm s Q K M sm_scale idx.1 with hvalFn
  set mskFn : TileIndex [64] → Bool := fun idx => decide (idx.1.val + s.pids 0 * 64 < 128) with hmskFn
  have hStore : stepStmt (Stmt.store .real [64] (MemAccess.ptr (Op.ref .ptr [64] "o_ptrs"))
      (Op.ref .real [64] "o")
      (MaskOpt.mask (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [64] "o_range")
        (Op.constNat 128)))) sP
      = some ((TileShape.allIndices [64]).foldl
          (fun acc idx => if mskFn idx then acc.writeMem Out (offFn idx) (valFn idx) else acc) sP) := by
    have hoP : sP.regs .real [64] "o" = some ⟨fun idx : TileIndex [64] =>
        some (case1OutClosedForm s Q K M sm_scale idx.1)⟩ := by
      rw [hsPdef, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      exact ho
    have hrangeP : sP.regs .nat [64] "o_range"
        = some (⟨fun idx : TileIndex [64] => idx.1.val + s.pids 0 * 64⟩ : Tile .nat [64]) := by
      rw [hsPdef, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        BlockState.setReg_same]
    have hptrsP : sP.regs .ptr [64] "o_ptrs"
        = some (⟨fun idx : TileIndex [64] => (Out.cast, case1OutStoreOffset s idx.1)⟩ : Tile .ptr [64]) := by
      rw [hsPdef, BlockState.setReg_same]
    simp only [stepStmt, evalOp_ref, hoP, hrangeP, hptrsP, evalOp_lt, evalOp_constNat,
      Option.bind_eq_bind, Option.bind_some, Option.map_some]
    refine congrArg some ?_
    refine congrArg (fun f => List.foldl f sP (TileShape.allIndices [64])) ?_
    funext acc idx
    by_cases hm : idx.1.val + s.pids 0 * 64 < 128
    · simp only [Tile.cop, Tile.scalar, Broadcast.rightIndex, Broadcast.leftIndex,
        ComparableDType.lt, hmskFn, hm, decide_true, if_true, hoffFn, hvalFn, case1OutStoreOffset,
        Region.cast_id, BlockState.writeMemTyped_real, FloatDType.storeValue,
        FloatDType.real_toWithBot, WithBot.unbotD_coe, WithBot.unbotD_some]
    · simp only [Tile.cop, Tile.scalar, Broadcast.rightIndex, Broadcast.leftIndex,
        ComparableDType.lt, hmskFn, hm, decide_false, Bool.false_eq_true, if_false,
        decide_eq_true_eq]
  rw [stepStmts.cons_some hStore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  -- readback at column i
  have hinj : Function.Injective offFn := by
    intro a b hab
    simp only [hoffFn, case1OutStoreOffset] at hab
    exact Prod.ext (Fin.ext (by omega)) (by cases a.2; cases b.2; rfl)
  have hread := BlockState.scatter_readback_masked_nd (region := Out) sP offFn valFn mskFn
    hinj (i, PUnit.unit)
  rw [show case1OutStoreOffset s i = offFn (i, PUnit.unit) from rfl, hread]
  have hmsk_iff : (mskFn (i, PUnit.unit) = Bool.true) ↔ case1OutActive s i := by
    simp only [hmskFn, decide_eq_true_eq, case1OutActive]; omega
  by_cases hi : case1OutActive s i
  · rw [if_pos (hmsk_iff.mpr hi), if_pos hi]
  · rw [if_neg (fun h => hi (hmsk_iff.mp h)), if_neg hi]
    rw [show offFn (i, PUnit.unit) = case1OutStoreOffset s i from rfl]
    simp only [hsPdef, BlockState.setReg_readMem]
    unfold BlockState.readMem; rw [hmem]

/-- **General** case-1 store offset for output column `i`:
`off_z·stride_oz + off_h·stride_oh + (start_n·BN + i)`. -/
def case1OutStoreOffsetG (s : BlockState) (H BN stride_oz stride_oh : Nat) (i : Fin BN) : Nat :=
  (s.pids 1 / H) * stride_oz + (s.pids 1 % H) * stride_oh + (s.pids 0 * BN + i.val)

/-- **General** store mask: `o_range = start_n·BN + i < NKV_CTX`. -/
def case1OutActiveG (s : BlockState) (BN NKV_CTX : Nat) (i : Fin BN) : Prop :=
  s.pids 0 * BN + i.val < NKV_CTX

instance (s : BlockState) (BN NKV_CTX : Nat) (i : Fin BN) :
    Decidable (case1OutActiveG s BN NKV_CTX i) := by unfold case1OutActiveG; infer_instance

set_option maxHeartbeats 2000000 in
/-- **General post-loop store readback.** The 4 post statements write
`case1OutClosedFormG` to `Out` at every active column. -/
theorem score_post_evalG
    (s sL : BlockState) (Q K M Out : RegionName) (sm_scale : ℝ)
    (H H_KV ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn
      stride_oz stride_oh swo sws : Nat) (i : Fin BN)
    (hmem : sL.mem = s.mem)
    (hoz : sL.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / H)))
    (hoh : sL.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % H)))
    (hsn : sL.regs .nat [] "start_n" = some (Tile.scalar (s.pids 0)))
    (ho : sL.regs .real [BN] "o" = some ⟨fun idx : TileIndex [BN] =>
        some (case1OutClosedFormG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
          stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws idx.1)⟩) :
    ∃ sF, stepStmts (attentionScoreCase1PostG Out BN NKV_CTX stride_oz stride_oh) sL = some sF
      ∧ sF.readMem Out (case1OutStoreOffsetG s H BN stride_oz stride_oh i)
        = (if case1OutActiveG s BN NKV_CTX i
            then case1OutClosedFormG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
              stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws i
            else s.readMem Out (case1OutStoreOffsetG s H BN stride_oz stride_oh i)) := by
  unfold attentionScoreCase1PostG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat stride_oz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat stride_oh))) _
        = some (Tile.scalar ((s.pids 1 / H) * stride_oz + (s.pids 1 % H) * stride_oh)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_constNat, evalOp_ref, hoz, hoh, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.scalarR (Op.arange BN)
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BN))) _
        = some (⟨fun idx : TileIndex [BN] => idx.1.val + s.pids 0 * BN⟩ : Tile .nat [BN]) from by
      rw [evalOp_add, evalOp_mul]
      simp only [evalOp_arange, evalOp_constNat, evalOp_ref, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, hsn, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul, Tile.vec_data]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add NumericDType.nat Broadcast.scalarL (Op.ref .nat [] "o_offset")
          (Op.ref .nat [BN] "o_range"))) _
        = some (⟨fun idx : TileIndex [BN] =>
            (Out.cast, case1OutStoreOffsetG s H BN stride_oz stride_oh idx.1)⟩ : Tile .ptr [BN]) from by
      rw [evalOp_ptrAdd', evalOp_add]
      simp only [evalOp_ptrBase', evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      · rfl
      · simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Broadcast.leftIndex,
          Broadcast.rightIndex, NumericDType.add, case1OutStoreOffsetG]
        omega))]
  set sP := (((sL.setReg "o_offset" .nat [] (Tile.scalar ((s.pids 1 / H) * stride_oz + (s.pids 1 % H) * stride_oh))).setReg
      "o_range" .nat [BN] (⟨fun idx : TileIndex [BN] => idx.1.val + s.pids 0 * BN⟩ : Tile .nat [BN])).setReg
      "o_ptrs" .ptr [BN] (⟨fun idx : TileIndex [BN] => (Out.cast, case1OutStoreOffsetG s H BN stride_oz stride_oh idx.1)⟩ : Tile .ptr [BN]))
    with hsPdef
  set offFn : TileIndex [BN] → Nat := fun idx => case1OutStoreOffsetG s H BN stride_oz stride_oh idx.1 with hoffFn
  set valFn : TileIndex [BN] → ℝ := fun idx => case1OutClosedFormG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws idx.1 with hvalFn
  set mskFn : TileIndex [BN] → Bool := fun idx => decide (idx.1.val + s.pids 0 * BN < NKV_CTX) with hmskFn
  have hStore : stepStmt (Stmt.store .real [BN] (MemAccess.ptr (Op.ref .ptr [BN] "o_ptrs"))
      (Op.ref .real [BN] "o")
      (MaskOpt.mask (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "o_range")
        (Op.constNat NKV_CTX)))) sP
      = some ((TileShape.allIndices [BN]).foldl
          (fun acc idx => if mskFn idx then acc.writeMem Out (offFn idx) (valFn idx) else acc) sP) := by
    have hoP : sP.regs .real [BN] "o" = some ⟨fun idx : TileIndex [BN] =>
        some (case1OutClosedFormG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
          stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws idx.1)⟩ := by
      rw [hsPdef, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      exact ho
    have hrangeP : sP.regs .nat [BN] "o_range"
        = some (⟨fun idx : TileIndex [BN] => idx.1.val + s.pids 0 * BN⟩ : Tile .nat [BN]) := by
      rw [hsPdef, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        BlockState.setReg_same]
    have hptrsP : sP.regs .ptr [BN] "o_ptrs"
        = some (⟨fun idx : TileIndex [BN] => (Out.cast, case1OutStoreOffsetG s H BN stride_oz stride_oh idx.1)⟩ : Tile .ptr [BN]) := by
      rw [hsPdef, BlockState.setReg_same]
    simp only [stepStmt, evalOp_ref, hoP, hrangeP, hptrsP, evalOp_lt, evalOp_constNat,
      Option.bind_eq_bind, Option.bind_some, Option.map_some]
    refine congrArg some ?_
    refine congrArg (fun f => List.foldl f sP (TileShape.allIndices [BN])) ?_
    funext acc idx
    by_cases hm : idx.1.val + s.pids 0 * BN < NKV_CTX
    · simp only [Tile.cop, Tile.scalar, Broadcast.rightIndex, Broadcast.leftIndex,
        ComparableDType.lt, hmskFn, hm, decide_true, if_true, hoffFn, hvalFn, case1OutStoreOffsetG,
        Region.cast_id, BlockState.writeMemTyped_real, FloatDType.storeValue,
        FloatDType.real_toWithBot, WithBot.unbotD_coe, WithBot.unbotD_some]
    · simp only [Tile.cop, Tile.scalar, Broadcast.rightIndex, Broadcast.leftIndex,
        ComparableDType.lt, hmskFn, hm, decide_false, Bool.false_eq_true, if_false,
        decide_eq_true_eq]
  rw [stepStmts.cons_some hStore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  have hinj : Function.Injective offFn := by
    intro a b hab
    simp only [hoffFn, case1OutStoreOffsetG] at hab
    exact Prod.ext (Fin.ext (by omega)) (by cases a.2; cases b.2; rfl)
  have hread := BlockState.scatter_readback_masked_nd (region := Out) sP offFn valFn mskFn
    hinj (i, PUnit.unit)
  rw [show case1OutStoreOffsetG s H BN stride_oz stride_oh i = offFn (i, PUnit.unit) from rfl, hread]
  have hmsk_iff : (mskFn (i, PUnit.unit) = Bool.true) ↔ case1OutActiveG s BN NKV_CTX i := by
    simp only [hmskFn, decide_eq_true_eq, case1OutActiveG]; omega
  by_cases hi : case1OutActiveG s BN NKV_CTX i
  · rw [if_pos (hmsk_iff.mpr hi), if_pos hi]
  · rw [if_neg (fun h => hi (hmsk_iff.mp h)), if_neg hi]
    rw [show offFn (i, PUnit.unit) = case1OutStoreOffsetG s H BN stride_oz stride_oh i from rfl]
    simp only [hsPdef, BlockState.setReg_readMem]
    unfold BlockState.readMem; rw [hmem]

set_option maxHeartbeats 2000000 in
/-- **Genuine case-1 exec correctness.** Executing the full case-1 kernel from a
clean state writes the genuine closed-form attention score
`case1OutClosedForm s Q K M sm_scale i` to `Out` at every active output column
(`case1OutStoreOffset s i`, mask `case1OutActive s i`) — with **no reference to
the kernel's own output**. -/
theorem attention_score_case1_exec_eq_closedForm
    (Q K M Out : RegionName) (sm_scale : ℝ) (s : BlockState) (i : Fin 64)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (exec (attention_score_kernel Q K M Out
        32768 8192 64 1 32768 8192 64 1 512 128 1
        2 4 4 128 128 128 0 64 64 64 64 sm_scale
        Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel s).map
      (·.readMem Out (case1OutStoreOffset s i))
      = some (if case1OutActive s i then case1OutClosedForm s Q K M sm_scale i
          else s.readMem Out (case1OutStoreOffset s i)) := by
  -- body = preLoop ++ (forRangeDyn loopBody :: post)
  rw [exec, attention_score_case1_body_split Q K M Out sm_scale,
    attentionScoreCase1PreLoop_check Q K M Out sm_scale,
    attention_score_case1_loopBody_check Q K M Out sm_scale,
    attentionScoreCase1Post_check Q K M Out sm_scale]
  -- preLoop
  obtain ⟨s0, hpre, hpids0, hmem0, hundef0, hsn0, hoz0, hoh0, hqks0, ho0, hk0, hQbp0, hmp0, hlo0, hhi0⟩ :=
    score_preLoop_eval s Q K M sm_scale hundef
  rw [stepStmts.append_some hpre]
  -- loop (2 iterations)
  obtain ⟨sL, hloop, hpidsL, hmemL, hozL, hohL, hsnL, hoL⟩ :=
    score_loop_eval s s0 Q K M sm_scale hpids0 hmem0 hundef0 hsn0 hoz0 hoh0 hqks0 ho0 hk0 hQbp0 hmp0
      hlo0 hhi0
  rw [stepStmts.cons_some hloop]
  -- post (masked store)
  obtain ⟨sF, hpost, hread⟩ :=
    score_post_eval s sL Q K M Out sm_scale i hmemL hozL hohL hsnL hoL
  rw [hpost, Option.map_some]
  exact congrArg some hread

/-- **Genuine compute-facing correctness (case 1).** The case-1 kernel realizes
the genuine closed-form attention score `case1OutClosedForm` at every active
output column — stated with **no reference to the kernel's own execution**. -/
theorem attention_score_case1_genuine_compute_correct
    (Q K M Out : RegionName) (sm_scale : ℝ) (s : BlockState)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_score_kernel Q K M Out
        32768 8192 64 1 32768 8192 64 1 512 128 1
        2 4 4 128 128 128 0 64 64 64 64 sm_scale
        Bool.true Bool.false Bool.true Bool.true rfl)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => case1OutActive s i)
        (fun i => (Out, case1OutStoreOffset s i)))
      (expected := fun i : Fin 64 =>
        case1OutClosedForm s Q K M sm_scale i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_score_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := attention_score_case1_exec_eq_closedForm Q K M Out sm_scale s i hundef
  rw [hExec, Option.map_some] at h
  have h2 := Option.some.inj h
  simp only [ComputeCorrect.OutputReadable.read_real]
  rw [h2, if_pos hActive]

/-! ### General elaborated-body decomposition checks (`rfl`)

The general case-1 elaborated body splits exactly as the pinned one: 16 preLoop
statements (`attentionScoreCase1PreLoopG`), one `forRangeDyn` loop over `start_m`
with body `attentionScoreCase1LoopBodyG`, and 4 post statements
(`attentionScoreCase1PostG`).  All proved by `rfl` (the `hBlockMN = rfl` forces
`BLOCK_M = BLOCK_N = BN`). -/

theorem attentionScoreCase1PreLoopG_check
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
     stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
     BN BD : Nat) (sm_scale : ℝ) :
    (attention_score_kernel Q K M Out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
      BN BD BN sm_scale Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel.body.take 16
      = attentionScoreCase1PreLoopG Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
          stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk := by
  rfl

theorem attentionScoreCase1LoopBodyG_check
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
     stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
     BN BD : Nat) (sm_scale : ℝ) :
    (attention_score_kernel Q K M Out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
      BN BD BN sm_scale Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel.body.drop 16
      = Stmt.forRangeDyn "start_m" (Op.ref .nat [] "lo") (Op.ref .nat [] "hi")
          (Op.constNat BN) (attentionScoreCase1LoopBodyG BN BD swo sws N_CTX)
        :: attentionScoreCase1PostG Out BN NKV_CTX stride_oz stride_oh := by
  rfl

set_option maxHeartbeats 2000000 in
/-- **General genuine case-1 exec correctness.** Dimension-parameterized version of
`attention_score_case1_exec_eq_closedForm`: executing the full case-1 kernel from a
clean state writes the genuine closed-form attention score `case1OutClosedFormG` to
`Out` at every active output column (`case1OutStoreOffsetG`, mask
`case1OutActiveG`), with **no reference to the kernel's own output**.  Side
conditions: the case-1 flag instantiation (`SLIDING_WINDOW = true`,
`COMPLEMENT_SLIDING_WINDOW = false`, `IS_EVEN_M = IS_EVEN_N = true`),
`BLOCK_M = BLOCK_N = BN`, `0 < BN`, `BN ∣ ROUND_CTX`, and the zero-undef
hypothesis. -/
theorem attention_score_case1_exec_eq_closedForm_general
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
     stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
     BN BD : Nat) (sm_scale : ℝ)
    (hBNpos : 0 < BN) (hdvd : BN ∣ ROUND_CTX)
    (s : BlockState) (i : Fin BN) (hundef : ∀ rg o, s.undef rg o = 0) :
    (exec (attention_score_kernel Q K M Out
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
        stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
        BN BD BN sm_scale Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel s).map
      (·.readMem Out (case1OutStoreOffsetG s H BN stride_oz stride_oh i))
      = some (if case1OutActiveG s BN NKV_CTX i
          then case1OutClosedFormG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
            stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws i
          else s.readMem Out (case1OutStoreOffsetG s H BN stride_oz stride_oh i)) := by
  rw [exec,
    show (attention_score_kernel Q K M Out
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
        stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
        BN BD BN sm_scale Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel.body
      = _ ++ _ from (List.take_append_drop 16 _).symm,
    attentionScoreCase1PreLoopG_check Q K M Out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws BN BD sm_scale,
    attentionScoreCase1LoopBodyG_check Q K M Out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws BN BD sm_scale]
  obtain ⟨s0, hpre, hpids0, hmem0, hundef0, hsn0, hoz0, hoh0, hqks0, ho0, hk0, hQbp0, hmp0, hlo0, hhi0⟩ :=
    score_preLoop_evalG s Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk hundef
  rw [stepStmts.append_some hpre]
  obtain ⟨sL, hloop, hpidsL, hmemL, hozL, hohL, hsnL, hoL⟩ :=
    score_loop_evalG s s0 Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD swo sws
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn
      hBNpos hdvd hpids0 hmem0 hundef0 hsn0 hoz0 hoh0 hqks0 ho0 hk0 hQbp0 hmp0 hlo0 hhi0
  rw [stepStmts.cons_some hloop]
  obtain ⟨sF, hpost, hread⟩ :=
    score_post_evalG s sL Q K M Out sm_scale H H_KV ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn
      stride_oz stride_oh swo sws i hmemL hozL hohL hsnL hoL
  rw [hpost, Option.map_some]
  exact congrArg some hread

/-- **General genuine compute-facing correctness (case 1).** The dimension-
parameterized case-1 kernel realizes the genuine closed-form attention score
`case1OutClosedFormG` at every active output column, with **no reference to the
kernel's own execution**. -/
theorem attention_score_case1_genuine_compute_correct_general
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
     stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
     BN BD : Nat) (sm_scale : ℝ)
    (hBNpos : 0 < BN) (hdvd : BN ∣ ROUND_CTX)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_score_kernel Q K M Out
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
        stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
        BN BD BN sm_scale Bool.true Bool.false Bool.true Bool.true rfl)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BN => case1OutActiveG s BN NKV_CTX i)
        (fun i => (Out, case1OutStoreOffsetG s H BN stride_oz stride_oh i)))
      (expected := fun i : Fin BN =>
        case1OutClosedFormG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
          stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_score_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := attention_score_case1_exec_eq_closedForm_general Q K M Out
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws BN BD sm_scale
    hBNpos hdvd s i hundef
  rw [hExec, Option.map_some] at h
  have h2 := Option.some.inj h
  simp only [ComputeCorrect.OutputReadable.read_real]
  rw [h2, if_pos hActive]


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **Public general Python case-1 output summary (genuine closed form).** The full
attention-score surface lowers to the algorithm layer, and the kernel writes the
genuine closed-form score `case1OutClosedFormG` to every active output column —
the dimension-parameterized case-1 output summary (symbolic shape/strides). -/
specification attention_score_python_case1_output_summary_general
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
     stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
     BN BD : Nat) (sm_scale : ℝ)
    (hBNpos : 0 < BN) (hdvd : BN ∣ ROUND_CTX)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (attention_score_kernel Q K M Out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
      BN BD BN sm_scale Bool.true Bool.false Bool.true Bool.true rfl).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_score_kernel Q K M Out
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
        stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
        BN BD BN sm_scale Bool.true Bool.false Bool.true Bool.true rfl)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BN => case1OutActiveG s BN NKV_CTX i)
        (fun i => (Out, case1OutStoreOffsetG s H BN stride_oz stride_oh i)))
      (expected := fun i : Fin BN =>
        case1OutClosedFormG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
          stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws i) := by
  refine ⟨?_, ?_⟩
  · exact attention_score_kernel_toAlgorithm_supported Q K M Out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws BN BD BN sm_scale
      Bool.true Bool.false Bool.true Bool.true rfl
  · exact attention_score_case1_genuine_compute_correct_general Q K M Out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws BN BD sm_scale
      hBNpos hdvd s hundef

end Correct_without_Rounding

/-! # ══════════ CORRECT — `⊨[R]` streaming io headline (case 1) ══════════ -/

section IOFaceCase1

open scoped VeriTile.Triton.StreamMasked3DKernelIO₃

/-! ## `R`-side constexpr mirrors and cast-free collapses

The whole case-1 surface is cast-free post-erasure (zero `castFloat` ops: the
store-side `.to(Out.type.element_ty)` erases to the identity at translation
and every load/store is `.real`-typed), so `stepStmtsR R` collapses verbatim
onto the exact stepper on all three body segments and the exact
`score_preLoop_evalG` / `score_loop_evalG` / `score_post_evalG` stack is
reused unchanged under `execR R`. -/

private theorem score_evalOpR_constBool (R : RoundingModel) (b : Bool) (s : BlockState) :
    evalOpR R (Op.constBool b) s = some (Tile.scalar b) := by
  simp only [evalOpR]

private theorem score_evalOpR_constNat (R : RoundingModel) (n : Nat) (s : BlockState) :
    evalOpR R (Op.constNat n) s = some (Tile.scalar n) := by
  simp only [evalOpR]

private theorem score_evalOpR_boolNot_true (R : RoundingModel) (s : BlockState) :
    evalOpR R (Op.boolNot (Op.constBool Bool.true)) s = some (Tile.scalar Bool.false) := by
  simp only [evalOpR, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `stepStmtR` collapse of a constexpr-`true` `ifThenElse` onto its live
then-branch. -/
private theorem score_ifThenElse_trueR (R : RoundingModel) {t e : List Stmt}
    {s : BlockState} :
    stepStmtR R (.ifThenElse (Op.constBool Bool.true) t e) s = stepStmtsR R t s := by
  simp only [stepStmtR, score_evalOpR_constBool, Option.bind, Tile.scalar]
  rfl

/-- `stepStmtR` collapse of a constexpr-`false` `ifThenElse` onto its live
else-branch. -/
private theorem score_ifThenElse_falseR (R : RoundingModel) {t e : List Stmt}
    {s : BlockState} :
    stepStmtR R (.ifThenElse (Op.constBool Bool.false) t e) s = stepStmtsR R e s := by
  simp only [stepStmtR, score_evalOpR_constBool, Option.bind, Tile.scalar]
  rfl

/-- `stepStmtR` collapse of a constexpr-`true` `ifThen` onto its body. -/
private theorem score_ifThen_trueR (R : RoundingModel) {body : List Stmt}
    {s : BlockState} :
    stepStmtR R (.ifThen (Op.constBool Bool.true) body) s = stepStmtsR R body s := by
  simp only [stepStmtR, score_evalOpR_constBool, Option.bind, Tile.scalar]
  rfl

/-- `stepStmtR` skip of a false-condition `ifThen`. -/
private theorem score_ifThen_falseR (R : RoundingModel) {cond : Op .bool []}
    {body : List Stmt} {s : BlockState}
    (hc : evalOpR R cond s = some (Tile.scalar Bool.false)) :
    stepStmtR R (.ifThen cond body) s = some s := by
  simp only [stepStmtR, hc, Option.bind, Tile.scalar]
  rfl

/-- `mapM` congruence for cast-free dynamic block-pointer offset lists. -/
private theorem score_evalOpR_castFree_offsets (R : RoundingModel) :
    ∀ (offsets : List (Op .nat [])) (s : BlockState),
      (∀ o ∈ offsets, evalOpR R o s = evalOp o s) →
      offsets.mapM (fun off => do
          let v ← evalOpR R off s
          some (v.data PUnit.unit))
        = offsets.mapM (fun off => do
            let v ← evalOp off s
            some (v.data PUnit.unit))
  | [], _, _ => rfl
  | off :: rest, s, h => by
      simp only [List.mapM_cons, h off List.mem_cons_self,
        score_evalOpR_castFree_offsets R rest s
          (fun o ho => h o (List.mem_cons_of_mem _ ho))]

/-- `evalOpR` mirror of `Op.makeBlockPtrDynOffsets` for cast-free base/offset
ops (the `.nat` channels never round). -/
private theorem score_evalOpR_mbpdo_castFree (R : RoundingModel) (region : RegionName)
    (base : Op .nat []) (ps : List Nat) (bs : TileShape) (strides : List Nat)
    (offs : List (Op .nat [])) (s : BlockState)
    (hb : evalOpR R base s = evalOp base s)
    (ho : ∀ o ∈ offs, evalOpR R o s = evalOp o s) :
    evalOpR R (.makeBlockPtrDynOffsets region base ps bs strides offs) s
      = evalOp (.makeBlockPtrDynOffsets region base ps bs strides offs) s := by
  simp only [evalOpR, evalOp, hb, score_evalOpR_castFree_offsets R offs s ho]

/-- `evalOpR` mirror of `Op.makeBlockPtrDyn` (static offsets, cast-free base). -/
private theorem score_evalOpR_mbpd_castFree (R : RoundingModel) (region : RegionName)
    (base : Op .nat []) (ps : List Nat) (bs : TileShape) (strides offs : List Nat)
    (s : BlockState) (hb : evalOpR R base s = evalOp base s) :
    evalOpR R (.makeBlockPtrDyn region base ps bs strides offs) s
      = evalOp (.makeBlockPtrDyn region base ps bs strides offs) s := by
  simp only [evalOpR, evalOp, hb]

/-- Per-statement cast-free collapse lifts to statement lists (walks the
actual successor chain; a failing step collapses on both sides). -/
private theorem score_stepStmtsR_castFree_of_stmts (R : RoundingModel) :
    ∀ (l : List Stmt), (∀ st ∈ l, ∀ u, stepStmtR R st u = stepStmt st u) →
      ∀ s, stepStmtsR R l s = stepStmts l s
  | [], _, s => by simp only [stepStmtsR, stepStmts]
  | st :: rest, h, s => by
      simp only [stepStmtsR, stepStmts, h st List.mem_cons_self s]
      cases stepStmt st s with
      | none => rfl
      | some s' =>
          exact score_stepStmtsR_castFree_of_stmts R rest
            (fun st' h' u => h st' (List.mem_cons_of_mem _ h') u) s'

/-- Cons-level inversion for `stepStmtsR` (walks a successful list run one
statement at a time). -/
private theorem score_stepStmtsR_cons_inv {R : RoundingModel} {st : Stmt}
    {rest : List Stmt} {s s' : BlockState}
    (h : stepStmtsR R (st :: rest) s = some s') :
    ∃ u, stepStmtR R st s = some u ∧ stepStmtsR R rest u = some s' := by
  rw [stepStmtsR] at h
  cases hu : stepStmtR R st s with
  | none => rw [hu] at h; exact absurd h (by simp)
  | some u => rw [hu] at h; exact ⟨u, rfl, h⟩

/-- `stepStmtR` on `forRangeDyn` unfolds to the `R` loop auxiliary after
evaluating the dynamic bounds under `evalOpR R` (the R-mirror of
`stepForRangeAux.forRangeDyn_unfold`). -/
private theorem score_stepStmtR_forRangeDyn (R : RoundingModel) (idx : RegName)
    (startOp stopOp stepOp : Op .nat []) (body : List Stmt) (s : BlockState) :
    stepStmtR R (.forRangeDyn idx startOp stopOp stepOp body) s
      = (evalOpR R startOp s).bind (fun a =>
          (evalOpR R stopOp s).bind (fun b =>
            (evalOpR R stepOp s).bind (fun c =>
              stepForRangeAuxR R idx (a.data PUnit.unit) (b.data PUnit.unit)
                (c.data PUnit.unit) body s))) := by
  unfold stepStmtR
  rfl

set_option maxHeartbeats 4000000 in
/-- Every case-1 preLoop statement is cast-free, statement-by-statement (the
constexpr `IS_EVEN_N` `ifThenElse` is reduced to its live k-load branch first;
the two block-pointer assigns go through the dedicated mirrors). -/
private theorem score_preLoopG_stmt_castFree (R : RoundingModel)
    (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk : Nat) :
    ∀ st ∈ attentionScoreCase1PreLoopG Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [attentionScoreCase1PreLoopG, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · -- Q_block_ptr (`makeBlockPtrDyn`): dedicated mirror
    simp only [stepStmtR, stepStmt,
      score_evalOpR_mbpd_castFree R Q (Op.ref TileDType.nat [] "q_offset") [N_CTX, BD] [BN, BD]
        [stride_qm, stride_qk] [0, 0] u (by simp only [evalOpR, evalOp])]
  · -- K_block_ptr (`makeBlockPtrDynOffsets`): dedicated `mapM` mirror
    simp only [stepStmtR, stepStmt,
      score_evalOpR_mbpdo_castFree R K (Op.ref TileDType.nat [] "k_offset") [BD, NKV_CTX] [BD, BN]
        [stride_kk, stride_kn]
        [Op.constNat 0,
          Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_n") (Op.constNat BN)] u
        (by simp only [evalOpR, evalOp])
        (by
          intro o ho
          simp only [List.mem_cons, List.not_mem_nil, or_false] at ho
          rcases ho with rfl | rfl <;> simp only [evalOpR.eq_def, evalOp.eq_def])]
  · -- IS_EVEN_N `ifThenElse` (constexpr-true): collapse on the live k-load only
    rw [score_ifThenElse_trueR, stepStmt_ifThenElse_true]
    simp only [stepStmtsR, stepStmts, stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
    rfl
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]

/-- The case-1 preLoop is cast-free: it steps identically under `stepStmtsR R`. -/
private theorem score_preLoopG_castFree (R : RoundingModel)
    (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk : Nat)
    (t : BlockState) :
    stepStmtsR R (attentionScoreCase1PreLoopG Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk) t
      = stepStmts (attentionScoreCase1PreLoopG Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
          stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk) t :=
  score_stepStmtsR_castFree_of_stmts R _
    (score_preLoopG_stmt_castFree R Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk) t

set_option maxHeartbeats 4000000 in
/-- Every case-1 loop-body statement is cast-free (the three constexpr guards
are reduced to their live branches first; the dead `not IS_EVEN_N` `where` is
skipped on both sides). -/
private theorem score_loopBodyG_stmt_castFree (R : RoundingModel)
    (BN BD swo sws N_CTX : Nat) :
    ∀ st ∈ attentionScoreCase1LoopBodyG BN BD swo sws N_CTX,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [attentionScoreCase1LoopBodyG, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · -- IS_EVEN_M `ifThenElse` (constexpr-true): collapse on the live q-load only
    rw [score_ifThenElse_trueR, stepStmt_ifThenElse_true]
    simp only [stepStmtsR, stepStmts, stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
    rfl
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · -- SLIDING_WINDOW `ifThen` (constexpr-true): dist + the COMPLEMENT branch
    rw [score_ifThen_trueR, stepStmt_ifThen_true]
    refine score_stepStmtsR_castFree_of_stmts R _ ?_ u
    intro st' hst' u'
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hst'
    rcases hst' with rfl | rfl
    · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
    · rw [score_ifThenElse_falseR, stepStmt_ifThenElse_false]
      simp only [stepStmtsR, stepStmts, stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
      rfl
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · -- SLIDING_WINDOW `where` `ifThen` (constexpr-true)
    rw [score_ifThen_trueR, stepStmt_ifThen_true]
    simp only [stepStmtsR, stepStmts, stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
    rfl
  · -- `not IS_EVEN_N` `ifThen`: skipped on both sides
    rw [score_ifThen_falseR R (score_evalOpR_boolNot_true R u), stepStmt_ifThen_boolNot_true]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]

/-- The case-1 loop body is cast-free. -/
private theorem score_loopBodyG_castFree (R : RoundingModel)
    (BN BD swo sws N_CTX : Nat) (t : BlockState) :
    stepStmtsR R (attentionScoreCase1LoopBodyG BN BD swo sws N_CTX) t
      = stepStmts (attentionScoreCase1LoopBodyG BN BD swo sws N_CTX) t :=
  score_stepStmtsR_castFree_of_stmts R _
    (score_loopBodyG_stmt_castFree R BN BD swo sws N_CTX) t

set_option maxHeartbeats 2000000 in
/-- The case-1 postLoop (three scalar/pointer assigns + the masked `.real`
store) is cast-free: `writeMemTypedR R .real` *is* `writeMemTyped .real`. -/
private theorem score_postG_castFree (R : RoundingModel)
    (Out : RegionName) (BN NKV_CTX stride_oz stride_oh : Nat) (t : BlockState) :
    stepStmtsR R (attentionScoreCase1PostG Out BN NKV_CTX stride_oz stride_oh) t
      = stepStmts (attentionScoreCase1PostG Out BN NKV_CTX stride_oz stride_oh) t := by
  simp only [attentionScoreCase1PostG, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The case-1 elaborated body splits as
`preLoopG ++ (forRangeDyn loopBodyG :: postG)`. -/
private theorem score_case1_body_splitG
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
     stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
     BN BD : Nat) (sm_scale : ℝ) :
    (attention_score_kernel Q K M Out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
      BN BD BN sm_scale Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel.body
      = attentionScoreCase1PreLoopG Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
          stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
        ++ (Stmt.forRangeDyn "start_m" (Op.ref .nat [] "lo") (Op.ref .nat [] "hi")
              (Op.constNat BN) (attentionScoreCase1LoopBodyG BN BD swo sws N_CTX)
            :: attentionScoreCase1PostG Out BN NKV_CTX stride_oz stride_oh) := by
  conv_lhs =>
    rw [← List.take_append_drop 16 (attention_score_kernel Q K M Out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
      BN BD BN sm_scale Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel.body]
  rw [attentionScoreCase1PreLoopG_check Q K M Out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws BN BD sm_scale,
    attentionScoreCase1LoopBodyG_check Q K M Out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws BN BD sm_scale]

set_option maxHeartbeats 4000000 in
/-- The case-1 surface sits inside the flat-memory bridge's covered fragment
(no `ptrSub`; the block-pointer ops are structurally covered). -/
private theorem score_case1_flattenOkG
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
     stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
     BN BD : Nat) (sm_scale : ℝ) :
    ((attention_score_kernel Q K M Out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
      BN BD BN sm_scale Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [score_case1_body_splitG Q K M Out
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws BN BD sm_scale]
  simp [attentionScoreCase1PreLoopG, attentionScoreCase1LoopBodyG, attentionScoreCase1PostG,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  simp [Op.FlattenOk.eq_def]

/-! ## IO signature, stream-indexed tiles, and the spec `f`

Window transcription (case-1 flag instantiation, `BLOCK_M = BLOCK_N = BN`
via the port's `hBlockMN = rfl`):

* `read1` (`Q`, streamed one `BN`-row block per step via the advanced
  `Q_block_ptr`): step `t`, lane `j = (i, e)` row-major over `[BN, BD]`
  reads `pid₁/H·sqz + pid₁%H·sqh + (t·BN + i)·sqm + e·sqk`.
* `read2` (`K`, the **static** stream — loaded once pre-loop, the window
  ignores `t`): lane `j = (d, n)` over `[BD, BN]` reads
  `pid₁/H·skz + (pid₁%H)/(H/H_KV)·skh + d·skk + (pid₀·BN + n)·skn`.
* `read3` (`M`, streamed one `BN`-row block per step via the plain
  `m_ptrs`): step `t`, lane `i` reads `pid₁·ROUND_CTX + (t·BN + i)`.
* `write` (`Out`, the single terminal masked store): lane `i` writes
  `pid₁/H·soz + pid₁%H·soh + (pid₀·BN + i)`, active iff
  `pid₀·BN + i < NKV_CTX` (the kernel's `o_range < NKV_CTX` store mask).

All read masks are `True` (`IS_EVEN_M = IS_EVEN_N = 1` in case 1). `pid₂`
is unused (2-D grid). `outDType` is the `.real` default — the store-side
`.to(Out.type.element_ty)` cast is erased to the identity at translation
(see the file's Modeling boundary). -/

/-- **Streaming IO signature** of the case-1
(`(SLIDING_WINDOW, COMPLEMENT, IS_EVEN_M, IS_EVEN_N) = (1,0,1,1)`)
`_score_kernel` surface on the three-stream single-output fold skin. -/
def attentionScoreCase1IO (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
     stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
     BN BD : Nat) (sm_scale : ℝ) : StreamMasked3DKernelIO₃ where
  kernel := attention_score_kernel Q K M Out
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
    BN BD BN sm_scale Bool.true Bool.false Bool.true Bool.true rfl
  inp1 := Q
  inp2 := K
  inp3 := M
  out := Out
  T := ROUND_CTX / BN
  B1 := BN * BD
  B2 := BD * BN
  B3 := BN
  C := BN
  read1 := fun _ p₁ _ t j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (t.val * BN + j.val / BD) * stride_qm + (j.val % BD) * stride_qk
  read2 := fun p₀ p₁ _ _ j =>
    p₁ / H * stride_kz + p₁ % H / (H / H_KV) * stride_kh
      + (j.val / BN) * stride_kk + (p₀ * BN + j.val % BN) * stride_kn
  read3 := fun _ p₁ _ t j => p₁ * ROUND_CTX + (t.val * BN + j.val)
  write := fun p₀ p₁ _ j => p₁ / H * stride_oz + p₁ % H * stride_oh + (p₀ * BN + j.val)
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  writeMask := fun p₀ _ _ j => p₀ * BN + j.val < NKV_CTX

/-- Nat-truncated sliding-window distance on the streams — the **verbatim**
`case1DistG` arithmetic (`ℕ`-truncated subtraction, `start_m = c·BN`,
`start_n = p₀`), with the block index `c` in stream-step position. -/
def attentionScoreCase1IODist (p₀ BN swo : Nat) (c i j : Nat) : Nat :=
  (((i - j) + c * BN) - p₀ * BN) + swo

/-- Sliding-window mask on the streams (case 1, non-complement):
`0 ≤ dist ∧ dist < sws`, both over `ℕ` (the `0 ≤` conjunct is vacuous —
kept verbatim from the elaborated `boolAnd (ge dist 0) (lt dist sws)`). -/
def attentionScoreCase1IOMask (p₀ BN swo sws : Nat) (c i j : Nat) : Prop :=
  0 ≤ attentionScoreCase1IODist p₀ BN swo c i j
    ∧ attentionScoreCase1IODist p₀ BN swo c i j < sws

instance (p₀ BN swo sws c i j : Nat) :
    Decidable (attentionScoreCase1IOMask p₀ BN swo sws c i j) := by
  unfold attentionScoreCase1IOMask; infer_instance

/-- **Case-1 closed form on the streams**: the masked-`exp2` query-row column
sum (`case1OutClosedFormG` restated over the three streamed tiles) at output
key column `j` —
`Σ_{c<T} Σ_{i<BN} [mask(c,i,j)] · exp2(sm_scale·log2e·⟨Q-row, K-col⟩ − M[c·BN+i])`,
where the `Q` row is step `c`'s lane row `i`, the `K` column is the static
stream's column `j` (step-`0` slice — the window ignores `t`), and the `M`
value is step `c`'s lane `i`. -/
noncomputable def attentionScoreCase1IOSpec (BN BD T swo sws : Nat) (hT : 0 < T)
    (sm_scale : ℝ) (p₀ : Nat)
    (xs : Fin T → Fin (BN * BD) → ℝ) (ys : Fin T → Fin (BD * BN) → ℝ)
    (zs : Fin T → Fin BN → ℝ) (j : Fin BN) : ℝ :=
  Finset.univ.sum (fun c : Fin T =>
    Finset.univ.sum (fun i : Fin BN =>
      if attentionScoreCase1IOMask p₀ BN swo sws c.val i.val j.val then
        pow2 (sm_scale * 1.4426950408889634 *
            Finset.univ.sum (fun d : Fin BD =>
              xs c (Lane2D.encode (i, d, PUnit.unit))
                * ys ⟨0, hT⟩ (Lane2D.encode (d, j, PUnit.unit)))
          - zs c i)
      else 0))

/-- **Stream-pin bridge**: under the skin's input pins the exact stack's
`case1OutClosedFormG` *is* the stream-level spec, pointwise. -/
private theorem score_case1_closedFormG_eq_ioSpec
    (s₀ : BlockState) (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV ROUND_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn
      swo sws : Nat)
    (hT : 0 < ROUND_CTX / BN)
    (xs : Fin (ROUND_CTX / BN) → Fin (BN * BD) → ℝ)
    (ys : Fin (ROUND_CTX / BN) → Fin (BD * BN) → ℝ)
    (zs : Fin (ROUND_CTX / BN) → Fin BN → ℝ)
    (hx : ∀ (t : Fin (ROUND_CTX / BN)) (j : Fin (BN * BD)),
      s₀.readMem Q (s₀.pids 1 / H * stride_qz + s₀.pids 1 % H * stride_qh
          + (t.val * BN + j.val / BD) * stride_qm + (j.val % BD) * stride_qk) = xs t j)
    (hy : ∀ (t : Fin (ROUND_CTX / BN)) (j : Fin (BD * BN)),
      s₀.readMem K (s₀.pids 1 / H * stride_kz + s₀.pids 1 % H / (H / H_KV) * stride_kh
          + (j.val / BN) * stride_kk + (s₀.pids 0 * BN + j.val % BN) * stride_kn) = ys t j)
    (hz : ∀ (t : Fin (ROUND_CTX / BN)) (j : Fin BN),
      s₀.readMem M (s₀.pids 1 * ROUND_CTX + (t.val * BN + j.val)) = zs t j)
    (j : Fin BN) :
    case1OutClosedFormG s₀ Q K M sm_scale H H_KV ROUND_CTX BN BN BD
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn
        swo sws j
      = attentionScoreCase1IOSpec BN BD (ROUND_CTX / BN) swo sws hT sm_scale (s₀.pids 0)
          xs ys zs j := by
  unfold case1OutClosedFormG attentionScoreCase1IOSpec case1ColSumG
  refine Finset.sum_congr rfl (fun c _ => Finset.sum_congr rfl (fun i _ => ?_))
  by_cases hm : case1MaskG s₀ BN BN swo sws c.val i.val j.val
  · rw [if_pos hm, if_pos (show attentionScoreCase1IOMask (s₀.pids 0) BN swo sws
        c.val i.val j.val from hm)]
    have hraw : case1RawScoreG s₀ Q K BD stride_qm stride_qk stride_kk stride_kn
        (case1QKOffsetQG s₀ H stride_qz stride_qh)
        (case1QKOffsetKG s₀ H H_KV stride_kz stride_kh)
        (c.val * BN + i.val) (s₀.pids 0 * BN + j.val)
        = Finset.univ.sum (fun d : Fin BD =>
            xs c (Lane2D.encode (i, d, PUnit.unit))
              * ys ⟨0, hT⟩ (Lane2D.encode (d, j, PUnit.unit))) := by
      unfold case1RawScoreG case1QElemG case1KElemG case1QKOffsetQG case1QKOffsetKG
      refine Finset.sum_congr rfl (fun d _ => ?_)
      rw [← hx c (Lane2D.encode (i, d, PUnit.unit)),
        ← hy ⟨0, hT⟩ (Lane2D.encode (d, j, PUnit.unit))]
      simp only [Lane2D.encode_div, Lane2D.encode_mod]
    have hmv : s₀.readMem M (case1MOffsetG s₀ ROUND_CTX (c.val * BN + i.val)) = zs c i :=
      hz c i
    unfold case1WeightG
    rw [hraw, hmv]
  · rw [if_neg hm, if_neg (show ¬ attentionScoreCase1IOMask (s₀.pids 0) BN swo sws
        c.val i.val j.val from hm)]

/-! ## The safety walk (weak invariant)

The skin's `hts` obligation quantifies over **arbitrary** launch states, so
the clean-`undef` exact stack (`score_preLoop_evalG` needs `hundef`) is
unavailable there. The safety walk instead runs on the *shape* half: exact
pins for the loop-carried `Q_block_ptr`/`m_ptrs` (whose addresses are the
bound obligations) plus bare existence for the value registers, mirroring
the `attention_fwd_triton3` `aft3SafeInv` precedent. -/

/-- The 11 leading scalar/pointer/index preLoop statements (no memory
reads) — the prefix of `attentionScoreCase1PreLoopG` before the k-load. -/
private def scoreCase1PreScalarsG (Q K M : RegionName)
    (H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk : Nat) :
    List Stmt :=
  [Stmt.assign TileDType.nat [] "start_n" (Op.programId 0),
    Stmt.assign TileDType.nat [] "off_hz" (Op.programId 1),
    Stmt.assign TileDType.nat [] "off_z"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat H)),
    Stmt.assign TileDType.nat [] "off_h"
      (Op.mod IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat H)),
    Stmt.assign TileDType.nat [] "off_hkv"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h")
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat H) (Op.constNat H_KV))),
    Stmt.assign TileDType.nat [] "q_offset"
      (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat stride_qz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h") (Op.constNat stride_qh))),
    Stmt.assign TileDType.nat [] "k_offset"
      (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat stride_kz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hkv") (Op.constNat stride_kh))),
    Stmt.assign TileDType.ptr [BN] "m_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase M)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat ROUND_CTX))
          (Op.arange BN))),
    Stmt.assign TileDType.real [BN] "o" (Op.full [BN] (Op.const 0)),
    Stmt.assign TileDType.blockPtr [BN, BD] "Q_block_ptr"
      (Op.makeBlockPtrDyn Q (Op.ref TileDType.nat [] "q_offset") [N_CTX, BD] [BN, BD]
        [stride_qm, stride_qk] [0, 0]),
    Stmt.assign TileDType.blockPtr [BD, BN] "K_block_ptr"
      (Op.makeBlockPtrDynOffsets K (Op.ref TileDType.nat [] "k_offset") [BD, NKV_CTX] [BD, BN]
        [stride_kk, stride_kn]
        [Op.constNat 0,
          Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_n") (Op.constNat BN)])]

/-- The 5 trailing preLoop statements (the constexpr k-load `ifThenElse` and
the `lo`/`hi`/`qk_scale` scalars). -/
private def scoreCase1PreTailG (sm_scale : ℝ) (ROUND_CTX BN BD : Nat) : List Stmt :=
  [Stmt.ifThenElse (Op.constBool Bool.true)
      [Stmt.assign TileDType.real [BD, BN] "k"
          (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BD, BN] "K_block_ptr") [])
            MaskOpt.none)]
      [Stmt.assign TileDType.real [BD, BN] "k"
          (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BD, BN] "K_block_ptr") [0, 1])
            MaskOpt.none)],
    Stmt.assign TileDType.nat [] "lo" (Op.constNat 0),
    Stmt.assign TileDType.nat [] "hi" (Op.constNat ROUND_CTX),
    Stmt.assign TileDType.real [] "qk_scale" (Op.const sm_scale),
    Stmt.assign TileDType.real [] "qk_scale"
      (Op.mul NumericDType.real Broadcast.nil (Op.ref TileDType.real [] "qk_scale")
        (Op.const 1.4426950408889634))]

private theorem scoreCase1PreScalarsG_append (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk : Nat) :
    attentionScoreCase1PreLoopG Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      = scoreCase1PreScalarsG Q K M H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
          stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
        ++ scoreCase1PreTailG sm_scale ROUND_CTX BN BD := rfl

/-- The scalar prefix is cast-free. -/
private theorem score_preScalarsG_castFree (R : RoundingModel)
    (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk : Nat)
    (t : BlockState) :
    stepStmtsR R (scoreCase1PreScalarsG Q K M H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk) t
      = stepStmts (scoreCase1PreScalarsG Q K M H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
          stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk) t :=
  score_stepStmtsR_castFree_of_stmts R _
    (fun st hst =>
      score_preLoopG_stmt_castFree R Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk st
        (by
          rw [scoreCase1PreScalarsG_append Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
            stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk]
          exact List.mem_append_left _ hst)) t

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 8000 in
/-- **Weak scalar-prefix eval**: from an **arbitrary** launch state (no
clean-`undef` hypothesis) the 11 scalar preLoop statements step to a state
with the register pins the k-load safety and the loop entry need. -/
private theorem score_preScalars_evalW
    (s : BlockState) (Q K M : RegionName)
    (H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk : Nat) :
    ∃ s11, stepStmts (scoreCase1PreScalarsG Q K M H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk) s = some s11
      ∧ s11.regs .nat [] "start_n" = some (Tile.scalar (s.pids 0))
      ∧ s11.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / H))
      ∧ s11.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % H))
      ∧ s11.regs .real [BN] "o" = some ⟨fun _ : TileIndex [BN] => some (0 : ℝ)⟩
      ∧ s11.regs .blockPtr [BN, BD] "Q_block_ptr" = some ⟨fun _ : TileIndex [BN, BD] =>
          { region := Q, baseOffset := case1QKOffsetQG s H stride_qz stride_qh, parentShape := [N_CTX, BD],
            blockShape := [BN, BD], strides := [stride_qm, stride_qk], offsets := [0, 0] }⟩
      ∧ s11.regs .blockPtr [BD, BN] "K_block_ptr" = some ⟨fun _ : TileIndex [BD, BN] =>
          { region := K, baseOffset := case1QKOffsetKG s H H_KV stride_kz stride_kh, parentShape := [BD, NKV_CTX],
            blockShape := [BD, BN], strides := [stride_kk, stride_kn], offsets := [0, s.pids 0 * BN] }⟩
      ∧ s11.regs .ptr [BN] "m_ptrs" = some ⟨fun idx : TileIndex [BN] =>
          (M.cast, s.pids 1 * ROUND_CTX + idx.1.val)⟩ := by
  unfold scoreCase1PreScalarsG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)) _
        = some (Tile.scalar (s.pids 1 / H)) from by
      rw [evalOp_floorDiv']
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)) _
        = some (Tile.scalar (s.pids 1 % H)) from by
      rw [evalOp_mod']
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_h")
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat H) (Op.constNat H_KV))) _
        = some (Tile.scalar ((s.pids 1 % H) / (H / H_KV))) from by
      rw [evalOp_floorDiv', evalOp_floorDiv']
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      rw [scalarBop, scalarBop]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat stride_qz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat stride_qh))) _
        = some (Tile.scalar (case1QKOffsetQG s H stride_qz stride_qh)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_constNat, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat stride_kz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_hkv") (Op.constNat stride_kh))) _
        = some (Tile.scalar (case1QKOffsetKG s H H_KV stride_kz stride_kh)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_constNat, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase M)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat ROUND_CTX))
          (Op.arange BN))) _
        = some (⟨fun idx : TileIndex [BN] => (M.cast, s.pids 1 * ROUND_CTX + idx.1.val)⟩ : Tile .ptr [BN]) from by
      rw [evalOp_ptrAdd', evalOp_add, evalOp_mul]
      simp only [evalOp_ptrBase', evalOp_constNat, evalOp_ref, evalOp_arange,
        BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_some]
      refine congrArg some ?_; ext idx
      · rfl
      · simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Tile.vec, Broadcast.leftIndex,
          Broadcast.rightIndex, NumericDType.add, NumericDType.mul, BlockState.setReg_pids]
        omega))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BN] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BN] => some (0 : ℝ)⟩ : Tile .real [BN]) from by
      simp [evalOp_full]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtrDyn_eval Q (Op.ref .nat [] "q_offset") [N_CTX, BD] [BN, BD] [stride_qm, stride_qk] [0, 0] _
      (case1QKOffsetQG s H stride_qz stride_qh) (by rw [evalOp_ref]; simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.makeBlockPtrDynOffsets K (Op.ref .nat [] "k_offset") [BD, NKV_CTX] [BD, BN] [stride_kk, stride_kn]
        [Op.constNat 0, Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BN)]) _
        = some (⟨fun _ : TileIndex [BD, BN] =>
            { region := K, baseOffset := case1QKOffsetKG s H H_KV stride_kz stride_kh, parentShape := [BD, NKV_CTX],
              blockShape := [BD, BN], strides := [stride_kk, stride_kn], offsets := [0, s.pids 0 * BN] }⟩
            : Tile .blockPtr [BD, BN]) from by
      rw [makeBlockPtr2_eval]
      simp only [evalOp_ref, evalOp_constNat, evalOp_mul, BlockState.setReg_same,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind,
        Option.bind_some, List.mapM_cons, List.mapM_nil, scalarBop]
      refine congrArg some ?_; ext idx <;> rfl))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 8000 in
/-- **Weak preLoop eval**: from an arbitrary launch state the full case-1
preLoop steps to the loop-entry state, with exact pins for the safety-relevant
registers and bare existence for the loaded `k` tile. -/
private theorem score_preLoop_evalW
    (s : BlockState) (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk : Nat) :
    ∃ s0, stepStmts (attentionScoreCase1PreLoopG Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk) s = some s0
      ∧ s0.regs .nat [] "start_n" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / H))
      ∧ s0.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % H))
      ∧ s0.regs .real [] "qk_scale" = some (Tile.scalar (some (sm_scale * 1.4426950408889634)))
      ∧ s0.regs .real [BN] "o" = some ⟨fun _ : TileIndex [BN] => some (0 : ℝ)⟩
      ∧ (∃ kT : Tile .real [BD, BN], s0.regs .real [BD, BN] "k" = some kT)
      ∧ s0.regs .blockPtr [BN, BD] "Q_block_ptr" = some ⟨fun _ : TileIndex [BN, BD] =>
          { region := Q, baseOffset := case1QKOffsetQG s H stride_qz stride_qh, parentShape := [N_CTX, BD],
            blockShape := [BN, BD], strides := [stride_qm, stride_qk], offsets := [0, 0] }⟩
      ∧ s0.regs .ptr [BN] "m_ptrs" = some ⟨fun idx : TileIndex [BN] =>
          (M.cast, s.pids 1 * ROUND_CTX + idx.1.val)⟩
      ∧ s0.regs .nat [] "lo" = some (Tile.scalar 0)
      ∧ s0.regs .nat [] "hi" = some (Tile.scalar ROUND_CTX) := by
  obtain ⟨s11, h11, hsn, hoz, hoh, ho, hQp, hKp, hmp⟩ :=
    score_preScalars_evalW s Q K M H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
  rw [scoreCase1PreScalarsG_append Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk]
  rw [stepStmts.append_some h11]
  unfold scoreCase1PreTailG
  set ktile : Tile .real [BD, BN] := ⟨fun idx : TileIndex [BD, BN] =>
      some (s11.readMem K (case1QKOffsetKG s H H_KV stride_kz stride_kh
        + idx.1.val * stride_kk + (s.pids 0 * BN + idx.2.1.val) * stride_kn))⟩ with hkt
  rw [stepStmts.cons_some
    (show stepStmt (Stmt.ifThenElse (Op.constBool Bool.true) _ _) _ = some _ from by
      rw [stepStmt_ifThenElse_true, stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BD, BN] "K_block_ptr") []) MaskOpt.none) _
            = some ktile from
          load_blockPtr_K_eval K (case1QKOffsetKG s H H_KV stride_kz stride_kh) BD NKV_CTX BD BN
            stride_kk stride_kn (s.pids 0 * BN)
            (Op.ref .blockPtr [BD, BN] "K_block_ptr") _ (by rw [evalOp_ref]; exact hKp)))]
      exact stepStmts.nil)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.constNat 0) _ = _ from evalOp_constNat 0 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.constNat ROUND_CTX) _ = _ from evalOp_constNat ROUND_CTX _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.const sm_scale) _ = _ from evalOp_const sm_scale _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul NumericDType.real Broadcast.nil (Op.ref .real [] "qk_scale")
        (Op.const 1.4426950408889634)) _
        = some (Tile.scalar (some (sm_scale * 1.4426950408889634))) from by
      rw [evalOp_mul]
      simp only [evalOp_ref, evalOp_const, BlockState.setReg_same, Option.bind_eq_bind,
        Option.bind_some, scalarBop]
      refine congrArg some (congrArg Tile.scalar ?_)
      simp only [NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ⟨ktile, ?_⟩, ?_, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    exact hsn
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    exact hoz
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    exact hoh
  · simp only [BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    exact ho
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    exact hQp
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    exact hmp
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]

/-- Safety-walk loop invariant: counter bounds, the `start_n`/`off_z`/`off_h`
scalar pins (the post-loop store addresses), the exact `Q_block_ptr`/`m_ptrs`
shapes at counter `m`, and existence of the value registers. -/
private def scoreSafeInv (Q M : RegionName)
    (qbase mbase sn ozv ohv N_CTX ROUND_CTX BN BD stride_qm stride_qk : Nat)
    (m : Nat) (s : BlockState) : Prop :=
  BN ∣ m ∧ m ≤ ROUND_CTX ∧
  s.regs .nat [] "start_n" = some (Tile.scalar sn) ∧
  s.regs .nat [] "off_z" = some (Tile.scalar ozv) ∧
  s.regs .nat [] "off_h" = some (Tile.scalar ohv) ∧
  (∃ kT : Tile .real [BD, BN], s.regs .real [BD, BN] "k" = some kT) ∧
  (∃ sc : ℝ, s.regs .real [] "qk_scale" = some (Tile.scalar (some sc))) ∧
  (∃ oT : Tile .real [BN], s.regs .real [BN] "o" = some oT) ∧
  s.regs .blockPtr [BN, BD] "Q_block_ptr" = some ⟨fun _ : TileIndex [BN, BD] =>
    { region := Q, baseOffset := qbase, parentShape := [N_CTX, BD],
      blockShape := [BN, BD], strides := [stride_qm, stride_qk], offsets := [m, 0] }⟩ ∧
  s.regs .ptr [BN] "m_ptrs" = some ⟨fun idx : TileIndex [BN] =>
    (M.cast, mbase + m + idx.1.val)⟩

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Weak loop-body eval** (`score_loopBody_evalG` minus the clean-`undef`
and memory/pid tracking, value pins weakened to existence): the case-1 body
steps successfully from register pins alone, advancing `Q_block_ptr` and
`m_ptrs` by one `BN` block and preserving the scalar pins. -/
private theorem score_loopBody_evalW
    (sin : BlockState) (Q M : RegionName)
    (N_CTX ROUND_CTX BN BD swo sws stride_qm stride_qk qbase mbase sn ozv ohv : Nat)
    (c : Nat) (sc : ℝ)
    (kT : Tile .real [BD, BN]) (oT : Tile .real [BN])
    (hsm : sin.regs .nat [] "start_m" = some (Tile.scalar (c * BN)))
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar sn))
    (hoz : sin.regs .nat [] "off_z" = some (Tile.scalar ozv))
    (hoh : sin.regs .nat [] "off_h" = some (Tile.scalar ohv))
    (hqks : sin.regs .real [] "qk_scale" = some (Tile.scalar (some sc)))
    (ho : sin.regs .real [BN] "o" = some oT)
    (hk : sin.regs .real [BD, BN] "k" = some kT)
    (hQbp : sin.regs .blockPtr [BN, BD] "Q_block_ptr" = some ⟨fun _ : TileIndex [BN, BD] =>
        { region := Q, baseOffset := qbase, parentShape := [N_CTX, BD],
          blockShape := [BN, BD], strides := [stride_qm, stride_qk], offsets := [c * BN, 0] }⟩)
    (hmp : sin.regs .ptr [BN] "m_ptrs" = some ⟨fun idx : TileIndex [BN] =>
        (M.cast, mbase + c * BN + idx.1.val)⟩) :
    ∃ sF, stepStmts (attentionScoreCase1LoopBodyG BN BD swo sws N_CTX) sin = some sF
      ∧ sF.regs .nat [] "start_n" = some (Tile.scalar sn)
      ∧ sF.regs .nat [] "off_z" = some (Tile.scalar ozv)
      ∧ sF.regs .nat [] "off_h" = some (Tile.scalar ohv)
      ∧ sF.regs .real [BD, BN] "k" = some kT
      ∧ sF.regs .real [] "qk_scale" = some (Tile.scalar (some sc))
      ∧ (∃ oT' : Tile .real [BN], sF.regs .real [BN] "o" = some oT')
      ∧ sF.regs .blockPtr [BN, BD] "Q_block_ptr" = some ⟨fun _ : TileIndex [BN, BD] =>
          { region := Q, baseOffset := qbase, parentShape := [N_CTX, BD],
            blockShape := [BN, BD], strides := [stride_qm, stride_qk], offsets := [c * BN + BN, 0] }⟩
      ∧ sF.regs .ptr [BN] "m_ptrs" = some ⟨fun idx : TileIndex [BN] =>
          (M.cast, mbase + (c * BN + BN) + idx.1.val)⟩ := by
  set qtile : Tile .real [BN, BD] := ⟨fun idx : TileIndex [BN, BD] =>
      some (sin.readMem Q (qbase + (c * BN + idx.1.val) * stride_qm + idx.2.1.val * stride_qk))⟩ with hqt
  set mtile : Tile .real [BN] := ⟨fun idx : TileIndex [BN] =>
      some (sin.readMem M (mbase + c * BN + idx.1.val))⟩ with hmt
  set disttile : Tile .nat [BN, BN] := ⟨fun idx : TileIndex [BN, BN] =>
      (((idx.1.val - idx.2.1.val) + c * BN) - sn * BN) + swo⟩ with hdistt
  set masktile : Tile .bool [BN, BN] := ⟨fun idx : TileIndex [BN, BN] =>
      decide (disttile.data idx < sws)⟩ with hmask
  have hmask_eq : (⟨fun idx : TileIndex [BN, BN] =>
      (decide (0 ≤ disttile.data idx) && decide (disttile.data idx < sws))⟩ : Tile .bool [BN, BN])
      = masktile := by
    rw [hmask]
    refine congrArg _ ?_
    ext idx
    simp
  unfold attentionScoreCase1LoopBodyG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_m") sin = some (Tile.scalar (c * BN)) from by
      rw [evalOp_ref]; exact hsm))]
  rw [stepStmts.cons_some
    (show stepStmt (Stmt.ifThenElse (Op.constBool Bool.true) _ _) _ = some _ from by
      rw [stepStmt_ifThenElse_true, stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BN, BD] "Q_block_ptr") []) MaskOpt.none) _
            = some qtile from by
          rw [load_blockPtr_Q_eval Q qbase N_CTX BD BN BD stride_qm stride_qk (c * BN)
            (Op.ref .blockPtr [BN, BD] "Q_block_ptr") _
            (by rw [evalOp_ref]
                simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
                exact hQbp)]
          rw [hqt]; refine congrArg some ?_; ext idx
          simp only [BlockState.setReg_readMem]))]
      exact stepStmts.nil)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN] "m_ptrs")) MaskOpt.none) _ = some mtile from by
      rw [load_ptr_none_real_score (Op.ref .ptr [BN] "m_ptrs") _
        (⟨fun idx : TileIndex [BN] => (M.cast, mbase + c * BN + idx.1.val)⟩ : Tile .ptr [BN])
        (by rw [evalOp_ref]
            simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
            exact hmp)]
      rw [hmt]; refine congrArg some ?_; ext idx
      simp [BlockState.readMem, BlockState.setReg_mem]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BN,BN] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BN,BN] => some (0:ℝ)⟩ : Tile .real [BN,BN]) from by
      simp [evalOp_full]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (score_qkRaw_evalG BN BD _ (⟨fun _ : TileIndex [BN,BN] => some (0:ℝ)⟩ : Tile .real [BN,BN]) qtile kT
      (by simp) (by simp) (by simp [hk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (score_qkScale_evalG BN _
      (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (⟨fun _ : TileIndex [BN,BN] => some (0:ℝ)⟩ : Tile .real [BN,BN]) (Tile.dot [] qtile kT))
      sc (by simp) (by simp [hqks])))]
  rw [stepStmts.cons_some
    (show stepStmt (Stmt.ifThen (Op.constBool Bool.true) _) _ = some _ from by
      rw [stepStmt_ifThen_true, stepStmts.cons_some (stepStmt_assign_eq_some
        (score_dist_evalG BN swo _ c sn (by simp [hsm]) (by simp [hsn])))]
      rw [stepStmts.cons_some
        (show stepStmt (Stmt.ifThenElse (Op.constBool Bool.false) _ _) _ = some _ from by
          rw [stepStmt_ifThenElse_false, stepStmts.cons_some (stepStmt_assign_eq_some
            ((score_mask_evalG BN sws _ disttile (by simp [hdistt])).trans
              (congrArg some hmask_eq)))]
          exact stepStmts.nil)]
      exact stepStmts.nil)]
  set scaledqk : Tile .real [BN, BN] := Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (⟨fun _ : TileIndex [BN,BN] => some (0:ℝ)⟩ : Tile .real [BN,BN]) (Tile.dot [] qtile kT))
      (Tile.scalar (some sc)) with hsq
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (score_qk2_evalG BN _ scaledqk mtile (by simp [hsq]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (score_p_evalG BN _ (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        scaledqk (Tile.expandDim ⟨1, by simp⟩ mtile)) (by simp)))]
  rw [stepStmts.cons_some
    (show stepStmt (Stmt.ifThen (Op.constBool Bool.true) _) _ = some _ from by
      rw [stepStmt_ifThen_true, stepStmts.cons_some (stepStmt_assign_eq_some
        (score_pwhere_evalG BN _ masktile
          (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
            (Broadcast.consSame (Broadcast.consR Broadcast.nil)) scaledqk (Tile.expandDim ⟨1, by simp⟩ mtile)))
          (by simp) (by simp)))]
      exact stepStmts.nil)]
  rw [stepStmts.cons_some (stepStmt_ifThen_boolNot_true _ _)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (score_oAcc_evalG BN _ oT
      (Tile.select masktile
        (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
          (Broadcast.consSame (Broadcast.consR Broadcast.nil)) scaledqk (Tile.expandDim ⟨1, by simp⟩ mtile)))
        (⟨fun _ : TileIndex [BN,BN] => some (0:ℝ)⟩ : Tile .real [BN,BN]))
      (by simp [ho]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BN,BD] "Q_block_ptr") [BN, (0:Nat)]) _
        = some (⟨fun _ : TileIndex [BN, BD] =>
            { region := Q, baseOffset := qbase, parentShape := [N_CTX, BD],
              blockShape := [BN, BD], strides := [stride_qm, stride_qk], offsets := [c * BN + BN, 0] }⟩
            : Tile .blockPtr [BN,BD]) from by
      rw [advanceBlockPtr_eval]; simp [hQbp]
      funext i; congr 1 <;> omega))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BN] "m_ptrs") (Op.constNat BN)) _
        = some (⟨fun idx : TileIndex [BN] =>
            (M.cast, mbase + (c * BN + BN) + idx.1.val)⟩ : Tile .ptr [BN]) from by
      rw [evalOp_ptrAdd']; simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff, hmp,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      · rfl
      · simp only [Tile.ptrAdd_data, Tile.scalar, Broadcast.rightIndex, Broadcast.leftIndex]
        omega))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_,
    ⟨Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil) oT
      (Tile.reduceSumDrop (⟨0, by simp⟩ : Fin [BN,BN].length)
        (Tile.select masktile
          (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
            (Broadcast.consSame (Broadcast.consR Broadcast.nil)) scaledqk
            (Tile.expandDim ⟨1, by simp⟩ mtile)))
          (⟨fun _ : TileIndex [BN,BN] => some (0:ℝ)⟩ : Tile .real [BN,BN]))), ?_⟩, ?_, ?_⟩
  · simp [hsn]
  · simp [hoz]
  · simp [hoh]
  · simp [hk]
  · simp [hqks]
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        BlockState.setReg_same]
  · simp
  · simp

set_option maxHeartbeats 2000000 in
/-- **Weak invariant step**: one case-1 body iteration from `scoreSafeInv m`
steps successfully (exact stepper) and re-establishes `scoreSafeInv (m + BN)`. -/
private theorem score_attn_stepW (Q M : RegionName)
    (qbase mbase sn ozv ohv N_CTX ROUND_CTX BN BD swo sws stride_qm stride_qk : Nat)
    (hBN : 0 < BN) (hdvd : BN ∣ ROUND_CTX)
    (m : Nat) (s : BlockState) (hm : m < ROUND_CTX)
    (hP : scoreSafeInv Q M qbase mbase sn ozv ohv N_CTX ROUND_CTX BN BD stride_qm stride_qk m s) :
    ∃ s', stepStmts (attentionScoreCase1LoopBodyG BN BD swo sws N_CTX)
        (s.setReg "start_m" .nat [] (Tile.scalar m)) = some s'
      ∧ scoreSafeInv Q M qbase mbase sn ozv ohv N_CTX ROUND_CTX BN BD stride_qm stride_qk (m + BN) s' := by
  obtain ⟨hdvdm, hle, hsn, hoz, hoh, ⟨kT, hk⟩, ⟨sc, hqks⟩, ⟨oT, ho⟩, hQbp, hmp⟩ := hP
  have hcm : m / BN * BN = m := Nat.div_mul_cancel hdvdm
  obtain ⟨sF, hstep, hsnF, hozF, hohF, hkF, hqksF, hoF, hQbpF, hmpF⟩ :=
    score_loopBody_evalW (s.setReg "start_m" .nat [] (Tile.scalar m)) Q M
      N_CTX ROUND_CTX BN BD swo sws stride_qm stride_qk qbase mbase sn ozv ohv (m / BN) sc kT oT
      (by rw [BlockState.setReg_same, hcm])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsn)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoz)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoh)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hqks)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact ho)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hk)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hcm]; exact hQbp)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hcm]; exact hmp)
  obtain ⟨qq, hqq⟩ := hdvd
  have hnextle : m + BN ≤ ROUND_CTX := by
    rw [← hcm] at hm ⊢
    rw [hqq] at hm ⊢
    have hlt : m / BN < qq := by
      by_contra hc
      push_neg at hc
      have : BN * qq ≤ m / BN * BN := by rw [Nat.mul_comm BN qq]; exact Nat.mul_le_mul_right BN hc
      omega
    calc m / BN * BN + BN = BN * (m / BN + 1) := by ring
      _ ≤ BN * qq := Nat.mul_le_mul_left BN hlt
  have hQbpF' := hQbpF
  rw [hcm] at hQbpF'
  have hmpF' := hmpF
  rw [hcm] at hmpF'
  exact ⟨sF, hstep, Nat.dvd_add hdvdm (Dvd.intro 1 (by ring)), hnextle, hsnF, hozF, hohF,
    ⟨kT, hkF⟩, ⟨sc, hqksF⟩, hoF, hQbpF', hmpF'⟩

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Weak body safety**: the case-1 loop body is `TraceSafeListR` from any
`scoreSafeInv m` state — the `q` load is bounded by the `read1` window at
step `m / BN`, the `m` load by the `read3` window; everything else is
register-only. -/
private theorem score_bodySafeW (R : RoundingModel) (bounds : RegionBounds)
    (Q M : RegionName)
    (qbase mbase sn ozv ohv N_CTX ROUND_CTX BN BD swo sws stride_qm stride_qk : Nat)
    (hBN : 0 < BN) (hdvd : BN ∣ ROUND_CTX)
    (m : Nat) (s : BlockState) (hm : m < ROUND_CTX)
    (hP : scoreSafeInv Q M qbase mbase sn ozv ohv N_CTX ROUND_CTX BN BD stride_qm stride_qk m s)
    (hbQ : ∀ (t : Fin (ROUND_CTX / BN)) (j : Fin (BN * BD)),
      qbase + (t.val * BN + j.val / BD) * stride_qm + (j.val % BD) * stride_qk < bounds Q)
    (hbM : ∀ (t : Fin (ROUND_CTX / BN)) (j : Fin BN),
      mbase + (t.val * BN + j.val) < bounds M) :
    Stmt.TraceSafeListR R bounds (attentionScoreCase1LoopBodyG BN BD swo sws N_CTX)
      (s.setReg "start_m" .nat [] (Tile.scalar m)) := by
  obtain ⟨hdvdm, hle, hsn, hoz, hoh, ⟨kT, hk⟩, ⟨sc, hqks⟩, ⟨oT, ho⟩, hQbp, hmp⟩ := hP
  have hcm : m / BN * BN = m := Nat.div_mul_cancel hdvdm
  have hmdivlt : m / BN < ROUND_CTX / BN := by
    have h1 : m / BN * BN < ROUND_CTX / BN * BN := by
      rw [hcm, Nat.div_mul_cancel hdvd]
      exact hm
    exact lt_of_mul_lt_mul_right h1 (Nat.zero_le BN)
  unfold attentionScoreCase1LoopBodyG
  -- (0) start_m = ref start_m
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s1 h1 => ?_)
  obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv h1
  -- (1) the q-load ifThenElse: `read1` bound at step m / BN
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s2 h2 => ?_)
  · simp only [Stmt.TraceSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], ?_⟩
    rw [score_evalOpR_constBool]
    show Stmt.TraceSafeListR R bounds
      [Stmt.assign TileDType.real [BN, BD] "q"
        (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BN, BD] "Q_block_ptr") [])
          MaskOpt.none)] _
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs hptrs idx _
    rw [evalOpR_ref] at hptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] at hptrs
    rw [hQbp] at hptrs
    obtain rfl := Option.some.inj hptrs
    intro _
    have hbound := hbQ ⟨m / BN, hmdivlt⟩ (Lane2D.encode (idx.1, idx.2.1, PUnit.unit))
    simp only [Lane2D.encode_div, Lane2D.encode_mod] at hbound
    rw [hcm] at hbound
    simpa using hbound
  rw [score_ifThenElse_trueR] at h2
  obtain ⟨u2, hu2, h2⟩ := score_stepStmtsR_cons_inv h2
  obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv hu2
  rw [stepStmtsR_nil] at h2
  obtain rfl := Option.some.inj h2
  -- (2) m = load(m_ptrs): the `read3` bound at step m / BN
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s3 h3 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs hptrs i _
    rw [evalOpR_ref] at hptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] at hptrs
    rw [hmp] at hptrs
    obtain rfl := Option.some.inj hptrs
    have hbound := hbM ⟨m / BN, hmdivlt⟩ i.1
    rw [hcm] at hbound
    show mbase + m + i.1.val < bounds (Region.cast M)
    simpa [Nat.add_assoc] using hbound
  obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv h3
  -- (3) qk = zeros
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s4 h4 => ?_)
  obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv h4
  -- (4) qk += dot q k
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s5 h5 => ?_)
  obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv h5
  -- (5) qk *= qk_scale
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s6 h6 => ?_)
  obtain ⟨v6, -, rfl⟩ := stepStmtR_assign_inv h6
  -- (6) the SLIDING_WINDOW ifThen (dist + the COMPLEMENT ifThenElse)
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s7 h7 => ?_)
  · simp only [Stmt.TraceSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], ?_⟩
    rw [score_evalOpR_constBool]
    show Stmt.TraceSafeListR R bounds [_, _] _
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun u7 hu7 => ?_)
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], ?_⟩
    rw [score_evalOpR_constBool]
    show Stmt.TraceSafeListR R bounds [_] _
    exact Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  rw [score_ifThen_trueR] at h7
  obtain ⟨u7, hu7, h7⟩ := score_stepStmtsR_cons_inv h7
  obtain ⟨v7, -, rfl⟩ := stepStmtR_assign_inv hu7
  obtain ⟨u8, hu8, h7⟩ := score_stepStmtsR_cons_inv h7
  rw [score_ifThenElse_falseR] at hu8
  obtain ⟨u9, hu9, hu8⟩ := score_stepStmtsR_cons_inv hu8
  obtain ⟨v9, -, rfl⟩ := stepStmtR_assign_inv hu9
  rw [stepStmtsR_nil] at hu8
  obtain rfl := Option.some.inj hu8
  rw [stepStmtsR_nil] at h7
  obtain rfl := Option.some.inj h7
  -- (7) qk -= m[:, None]
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s8 h8 => ?_)
  obtain ⟨v8, -, rfl⟩ := stepStmtR_assign_inv h8
  -- (8) p = exp2 qk
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s9 h9 => ?_)
  obtain ⟨v9', -, rfl⟩ := stepStmtR_assign_inv h9
  -- (9) the SLIDING_WINDOW where ifThen
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s10 h10 => ?_)
  · simp only [Stmt.TraceSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], ?_⟩
    rw [score_evalOpR_constBool]
    show Stmt.TraceSafeListR R bounds [_] _
    exact Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  rw [score_ifThen_trueR] at h10
  obtain ⟨u10, hu10, h10⟩ := score_stepStmtsR_cons_inv h10
  obtain ⟨v10, -, rfl⟩ := stepStmtR_assign_inv hu10
  rw [stepStmtsR_nil] at h10
  obtain rfl := Option.some.inj h10
  -- (10) the dead `not IS_EVEN_N` ifThen (skipped)
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s11 h11 => ?_)
  · simp only [Stmt.TraceSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], ?_⟩
    rw [score_evalOpR_boolNot_true]
    exact trivial
  rw [score_ifThen_falseR R (score_evalOpR_boolNot_true R _)] at h11
  obtain rfl := Option.some.inj h11
  -- (11) o += reduceSum p
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s12 h12 => ?_)
  obtain ⟨v12, -, rfl⟩ := stepStmtR_assign_inv h12
  -- (12) Q_block_ptr advance (register-only)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s13 h13 => ?_)
  obtain ⟨v13, -, rfl⟩ := stepStmtR_assign_inv h13
  -- (13) m_ptrs += BN (register-only)
  exact Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (fun _ _ => Stmt.TraceSafeListR.nil_intro)

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **`TraceSafeListR` for the case-1 postLoop**: the three scalar/pointer
assigns are register-only; the terminal masked store's active lanes are
bounded by the `write` window (`hbO`, gated on the `o_range < NKV_CTX`
mask). -/
private theorem score_postSafeW (R : RoundingModel) (bounds : RegionBounds)
    (Out : RegionName)
    (sn ozv ohv NKV_CTX BN stride_oz stride_oh : Nat)
    (s : BlockState)
    (hoz : s.regs .nat [] "off_z" = some (Tile.scalar ozv))
    (hoh : s.regs .nat [] "off_h" = some (Tile.scalar ohv))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar sn))
    (hbO : ∀ j : Fin BN, sn * BN + j.val < NKV_CTX →
      ozv * stride_oz + ohv * stride_oh + (sn * BN + j.val) < bounds Out) :
    Stmt.TraceSafeListR R bounds (attentionScoreCase1PostG Out BN NKV_CTX stride_oz stride_oh) s := by
  unfold attentionScoreCase1PostG
  -- (1) o_offset (register-only; concrete successor via inversion)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s1 h1 => ?_)
  obtain ⟨v1, hv1, rfl⟩ := stepStmtR_assign_inv h1
  obtain rfl := Option.some.inj (hv1.symm.trans
    (show evalOpR R (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat stride_oz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h") (Op.constNat stride_oh))) s
        = some (Tile.scalar (ozv * stride_oz + ohv * stride_oh)) from by
      simp only [evalOpR, hoz, hoh, Option.bind_eq_bind, Option.bind_some]
      rfl))
  -- (2) o_range (register-only; concrete successor via inversion)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s2 h2 => ?_)
  obtain ⟨v2, hv2, rfl⟩ := stepStmtR_assign_inv h2
  obtain rfl := Option.some.inj (hv2.symm.trans
    (show evalOpR R (Op.add NumericDType.nat Broadcast.scalarR (Op.arange BN)
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_n") (Op.constNat BN)))
        (s.setReg "o_offset" TileDType.nat [] (Tile.scalar (ozv * stride_oz + ohv * stride_oh)))
        = some (⟨fun idx : TileIndex [BN] => idx.1.val + sn * BN⟩ : Tile .nat [BN]) from by
      simp only [evalOpR, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        hsn, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul, Tile.vec_data]))
  -- (3) o_ptrs (register-only; concrete successor via inversion)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s3 h3 => ?_)
  obtain ⟨v3, hv3, rfl⟩ := stepStmtR_assign_inv h3
  obtain rfl := Option.some.inj (hv3.symm.trans
    (show evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add NumericDType.nat Broadcast.scalarL (Op.ref TileDType.nat [] "o_offset")
          (Op.ref TileDType.nat [BN] "o_range")))
        ((s.setReg "o_offset" TileDType.nat [] (Tile.scalar (ozv * stride_oz + ohv * stride_oh))).setReg
          "o_range" TileDType.nat [BN] (⟨fun idx : TileIndex [BN] => idx.1.val + sn * BN⟩ : Tile .nat [BN]))
        = some (⟨fun idx : TileIndex [BN] =>
            (Out.cast, ozv * stride_oz + ohv * stride_oh + (sn * BN + idx.1.val))⟩ : Tile .ptr [BN]) from by
      simp only [evalOpR, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        BlockState.setReg_same, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      · rfl
      · simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Broadcast.leftIndex,
          Broadcast.rightIndex, NumericDType.add]
        omega))
  -- (4) the masked store: active lanes bounded by the write window
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR, Op.SafeAtR.eq_def,
    MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
  refine ⟨trivial, trivial, ⟨trivial, trivial⟩, ?_⟩
  intro ptrs hptrs i hact
  rw [evalOpR_ref, BlockState.setReg_same] at hptrs
  obtain rfl := Option.some.inj hptrs
  obtain ⟨masks, hmasks, hmi⟩ := hact
  rw [show evalOpR R (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BN] "o_range")
      (Op.constNat NKV_CTX)) _
      = some (Tile.cop ComparableDType.nat.lt Broadcast.scalarR
          (⟨fun idx : TileIndex [BN] => idx.1.val + sn * BN⟩ : Tile .nat [BN])
          (Tile.scalar NKV_CTX)) from by
    simp only [evalOpR, evalOpR_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, BlockState.setReg_same, Option.bind_eq_bind, Option.bind_some]
    rfl] at hmasks
  obtain rfl := Option.some.inj hmasks
  have hlt : i.1.val + sn * BN < NKV_CTX := by
    simpa only [Tile.cop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
      ComparableDType.lt, decide_eq_true_eq] using hmi
  have hbound := hbO i.1 (by omega)
  show ozv * stride_oz + ohv * stride_oh + (sn * BN + i.1.val) < bounds (Region.cast Out)
  simpa using hbound

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **PreLoop safety**: the 11 scalar assigns are safe at every state; the
pre-loop k-load is bounded by the (static, `t`-free) `read2` window at step
`0`. -/
private theorem score_preSafeW (R : RoundingModel) (bounds : RegionBounds)
    (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk : Nat)
    (s : BlockState) (hT : 0 < ROUND_CTX / BN)
    (hbK : ∀ (t : Fin (ROUND_CTX / BN)) (j : Fin (BD * BN)),
      case1QKOffsetKG s H H_KV stride_kz stride_kh
        + (j.val / BN) * stride_kk + (s.pids 0 * BN + j.val % BN) * stride_kn < bounds K) :
    Stmt.TraceSafeListR R bounds
      (attentionScoreCase1PreLoopG Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk) s := by
  rw [scoreCase1PreScalarsG_append Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- statements 0–10: register-only assigns, safe at every state
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro st hst u
    simp only [scoreCase1PreScalarsG, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR]
  · intro s1 hs1
    obtain ⟨s11, h11, hsn11, hoz11, hoh11, ho11, hQp11, hKp11, hmp11⟩ :=
      score_preScalars_evalW s Q K M H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    rw [score_preScalarsG_castFree R Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk s, h11] at hs1
    obtain rfl := Option.some.inj hs1
    unfold scoreCase1PreTailG
    -- the k-load ifThenElse: the static `read2` bound at step 0
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun s2 hs2 => ?_)
    · simp only [Stmt.TraceSafeR]
      refine ⟨by simp [Op.SafeAtR.eq_def], ?_⟩
      rw [score_evalOpR_constBool]
      show Stmt.TraceSafeListR R bounds
        [Stmt.assign TileDType.real [BD, BN] "k"
          (Op.load TileDType.real (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BD, BN] "K_block_ptr") [])
            MaskOpt.none)] _
      refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
      refine ⟨trivial, trivial, ?_⟩
      intro ptrs hptrs idx _
      rw [evalOpR_ref, hKp11] at hptrs
      obtain rfl := Option.some.inj hptrs
      intro _
      have hbound := hbK ⟨0, hT⟩ (Lane2D.encode (idx.1, idx.2.1, PUnit.unit))
      simp only [Lane2D.encode_div, Lane2D.encode_mod] at hbound
      simpa using hbound
    rw [score_ifThenElse_trueR] at hs2
    obtain ⟨u1, hu1, hs2⟩ := score_stepStmtsR_cons_inv hs2
    obtain ⟨vk, -, rfl⟩ := stepStmtR_assign_inv hu1
    rw [stepStmtsR_nil] at hs2
    obtain rfl := Option.some.inj hs2
    -- lo / hi / qk_scale ×2: register-only
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun u2 hu2 => ?_)
    obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv hu2
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun u3 hu3 => ?_)
    obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv hu3
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun u4 hu4 => ?_)
    obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv hu4
    exact Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
      (fun _ _ => Stmt.TraceSafeListR.nil_intro)

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The `TraceSafeR` walk for the whole case-1 kernel**: the 11 leading
scalar assigns are safe at every state, the pre-loop k-load is bounded by
the (static, `t`-free) `read2` window at step `0` — the pre-loop static load
is what forces `0 < ROUND_CTX` — the query loop runs
`Stmt.forRangeTraceSafeR_inv` over `scoreSafeInv` after resolving the
`forRangeDyn` bounds ops, and the masked terminal store is bounded by the
`write` window. -/
private theorem score_traceSafeR_case1 (R : RoundingModel) (bounds : RegionBounds)
    (Q K M Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
     stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
     BN BD : Nat) (s : BlockState)
    (hBN : 0 < BN) (hRC : 0 < ROUND_CTX) (hdvd : BN ∣ ROUND_CTX)
    (hbQ : ∀ (t : Fin (ROUND_CTX / BN)) (j : Fin (BN * BD)),
      case1QKOffsetQG s H stride_qz stride_qh
        + (t.val * BN + j.val / BD) * stride_qm + (j.val % BD) * stride_qk < bounds Q)
    (hbK : ∀ (t : Fin (ROUND_CTX / BN)) (j : Fin (BD * BN)),
      case1QKOffsetKG s H H_KV stride_kz stride_kh
        + (j.val / BN) * stride_kk + (s.pids 0 * BN + j.val % BN) * stride_kn < bounds K)
    (hbM : ∀ (t : Fin (ROUND_CTX / BN)) (j : Fin BN),
      s.pids 1 * ROUND_CTX + (t.val * BN + j.val) < bounds M)
    (hbO : ∀ j : Fin BN, case1OutActiveG s BN NKV_CTX j →
      case1OutStoreOffsetG s H BN stride_oz stride_oh j < bounds Out) :
    ((attention_score_kernel Q K M Out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
      BN BD BN sm_scale Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel).TraceSafeR R bounds s := by
  have hT : 0 < ROUND_CTX / BN := Nat.div_pos (Nat.le_of_dvd hRC hdvd) hBN
  unfold Kernel.TraceSafeR
  rw [score_case1_body_splitG Q K M Out
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws BN BD sm_scale]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- the preLoop
    exact score_preSafeW R bounds Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk s hT hbK
  · -- after the preLoop: the query loop, then the postLoop store
    intro s2 hs2
    obtain ⟨spW, hpreW, hsnW, hozW, hohW, hqksW, hoW, hkEW, hQbpW, hmpW, hloW, hhiW⟩ :=
      score_preLoop_evalW s Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    rw [score_preLoopG_castFree R Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk s,
      hpreW] at hs2
    obtain rfl := Option.some.inj hs2
    have hinv0 : scoreSafeInv Q M (case1QKOffsetQG s H stride_qz stride_qh)
        (s.pids 1 * ROUND_CTX) (s.pids 0) (s.pids 1 / H) (s.pids 1 % H)
        N_CTX ROUND_CTX BN BD stride_qm stride_qk 0 spW := by
      refine ⟨Dvd.intro 0 rfl, Nat.zero_le _, hsnW, hozW, hohW, hkEW,
        ⟨sm_scale * 1.4426950408889634, hqksW⟩, ⟨_, hoW⟩, ?_, ?_⟩
      · exact hQbpW
      · exact hmpW
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun s3 hs3 => ?_)
    · -- the query loop is trace-safe (invariant principle over `scoreSafeInv`)
      simp only [Stmt.TraceSafeR]
      refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
        by simp [Op.SafeAtR.eq_def], ?_⟩
      rw [evalOpR_ref, hloW, evalOpR_ref, hhiW, score_evalOpR_constNat]
      show Stmt.forRangeTraceSafeR R bounds "start_m" 0 ROUND_CTX BN
        (attentionScoreCase1LoopBodyG BN BD swo sws N_CTX) spW
      refine Stmt.forRangeTraceSafeR_inv R bounds "start_m" ROUND_CTX BN
        (attentionScoreCase1LoopBodyG BN BD swo sws N_CTX)
        (scoreSafeInv Q M (case1QKOffsetQG s H stride_qz stride_qh)
          (s.pids 1 * ROUND_CTX) (s.pids 0) (s.pids 1 / H) (s.pids 1 % H)
          N_CTX ROUND_CTX BN BD stride_qm stride_qk) ?_ 0 spW hinv0
      intro m st hm hP
      refine ⟨score_bodySafeW R bounds Q M (case1QKOffsetQG s H stride_qz stride_qh)
        (s.pids 1 * ROUND_CTX) (s.pids 0) (s.pids 1 / H) (s.pids 1 % H)
        N_CTX ROUND_CTX BN BD swo sws stride_qm stride_qk hBN hdvd m st hm hP hbQ hbM, ?_⟩
      obtain ⟨st', hstep, hP'⟩ :=
        score_attn_stepW Q M (case1QKOffsetQG s H stride_qz stride_qh)
          (s.pids 1 * ROUND_CTX) (s.pids 0) (s.pids 1 / H) (s.pids 1 % H)
          N_CTX ROUND_CTX BN BD swo sws stride_qm stride_qk hBN hdvd m st hm hP
      exact ⟨st', by rw [score_loopBodyG_castFree]; exact hstep, hP'⟩
    · -- identify the post-loop state and finish on the masked store
      obtain ⟨final, sfin, hLoop, hfinal, hPfin⟩ :=
        forRangeDyn_inv (idx := "start_m")
          (startOp := Op.ref .nat [] "lo") (stopOp := Op.ref .nat [] "hi")
          (stepOp := Op.constNat BN)
          (start := 0) (stop := ROUND_CTX) (step := BN)
          (body := attentionScoreCase1LoopBodyG BN BD swo sws N_CTX)
          (P := scoreSafeInv Q M (case1QKOffsetQG s H stride_qz stride_qh)
            (s.pids 1 * ROUND_CTX) (s.pids 0) (s.pids 1 / H) (s.pids 1 % H)
            N_CTX ROUND_CTX BN BD stride_qm stride_qk)
          (s_init := spW)
          (by rw [evalOp_ref]; exact hloW) (by rw [evalOp_ref]; exact hhiW)
          (evalOp_constNat BN spW) hBN.ne' hinv0
          (fun m st hm hP =>
            score_attn_stepW Q M (case1QKOffsetQG s H stride_qz stride_qh)
              (s.pids 1 * ROUND_CTX) (s.pids 0) (s.pids 1 / H) (s.pids 1 % H)
              N_CTX ROUND_CTX BN BD swo sws stride_qm stride_qk hBN hdvd m st hm hP)
      rw [show stepStmtR R (Stmt.forRangeDyn "start_m" (Op.ref .nat [] "lo") (Op.ref .nat [] "hi")
          (Op.constNat BN) (attentionScoreCase1LoopBodyG BN BD swo sws N_CTX)) spW = some sfin from by
        rw [score_stepStmtR_forRangeDyn, evalOpR_ref, hloW, evalOpR_ref, hhiW,
          score_evalOpR_constNat]
        simp only [Option.bind_eq_bind, Option.bind_some, Tile.scalar_data]
        rw [stepForRangeAuxR_castFree R _ (score_loopBodyG_castFree R BN BD swo sws N_CTX) "start_m"]
        rw [stepForRangeAux.forRangeDyn_unfold,
          show evalOp (Op.ref .nat [] "lo") spW = some (Tile.scalar 0) from by
            rw [evalOp_ref]; exact hloW,
          show evalOp (Op.ref .nat [] "hi") spW = some (Tile.scalar ROUND_CTX) from by
            rw [evalOp_ref]; exact hhiW,
          evalOp_constNat] at hLoop
        simpa only [Option.bind_eq_bind, Option.bind_some, Tile.scalar_data] using hLoop] at hs3
      obtain rfl := Option.some.inj hs3
      obtain ⟨-, -, hsnF, hozF, hohF, -, -, -, -, -⟩ := hPfin
      exact score_postSafeW R bounds Out (s.pids 0) (s.pids 1 / H) (s.pids 1 % H)
        NKV_CTX BN stride_oz stride_oh sfin hozF hohF hsnF hbO

/-! ## The postLoop frame and the rounded Hoare triple -/

/-- A masked single-region `writeMem` scatter preserves every cell it does
not actively hit. -/
private theorem score_foldl_writeMem_masked_frame {α : Type} (region : RegionName)
    (offFn : α → Nat) (valFn : α → ℝ) (mskFn : α → Bool) :
    ∀ (l : List α) (s : BlockState) (r : RegionName) (o : Nat),
      (r = region → ∀ k ∈ l, mskFn k = Bool.true → offFn k ≠ o) →
      ((l.foldl (fun acc k =>
          if mskFn k then acc.writeMem region (offFn k) (valFn k) else acc) s).mem r o
        = s.mem r o)
  | [], _, _, _, _ => rfl
  | k :: rest, s, r, o, h => by
      rw [List.foldl_cons,
        score_foldl_writeMem_masked_frame region offFn valFn mskFn rest _ r o
          (fun hr k' hk' => h hr k' (List.mem_cons_of_mem _ hk'))]
      by_cases hm : mskFn k
      · rw [if_pos hm, BlockState.writeMem_mem]
        rw [if_neg (fun hro => h hro.1 k List.mem_cons_self hm hro.2.symm)]
      · rw [if_neg hm]

set_option maxHeartbeats 2000000 in
/-- **General post-loop store readback + frame** (the `score_post_evalG`
walk extended with the single-output frame the io skin needs): the 4 post
statements write `case1OutClosedFormG` to `Out` at every active column and
touch no other cell. -/
private theorem score_post_evalFrameG
    (s sL : BlockState) (Q K M Out : RegionName) (sm_scale : ℝ)
    (H H_KV ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn
      stride_oz stride_oh swo sws : Nat)
    (hmem : sL.mem = s.mem)
    (hoz : sL.regs .nat [] "off_z" = some (Tile.scalar (s.pids 1 / H)))
    (hoh : sL.regs .nat [] "off_h" = some (Tile.scalar (s.pids 1 % H)))
    (hsn : sL.regs .nat [] "start_n" = some (Tile.scalar (s.pids 0)))
    (ho : sL.regs .real [BN] "o" = some ⟨fun idx : TileIndex [BN] =>
        some (case1OutClosedFormG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
          stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws idx.1)⟩) :
    ∃ sF, stepStmts (attentionScoreCase1PostG Out BN NKV_CTX stride_oz stride_oh) sL = some sF
      ∧ (∀ i : Fin BN, sF.readMem Out (case1OutStoreOffsetG s H BN stride_oz stride_oh i)
          = (if case1OutActiveG s BN NKV_CTX i
              then case1OutClosedFormG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
                stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws i
              else s.readMem Out (case1OutStoreOffsetG s H BN stride_oz stride_oh i)))
      ∧ (∀ r o, (r = Out → ∀ i : Fin BN, case1OutActiveG s BN NKV_CTX i →
            o ≠ case1OutStoreOffsetG s H BN stride_oz stride_oh i) →
          sF.mem r o = sL.mem r o) := by
  unfold attentionScoreCase1PostG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat stride_oz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat stride_oh))) _
        = some (Tile.scalar ((s.pids 1 / H) * stride_oz + (s.pids 1 % H) * stride_oh)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_constNat, evalOp_ref, hoz, hoh, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.scalarR (Op.arange BN)
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BN))) _
        = some (⟨fun idx : TileIndex [BN] => idx.1.val + s.pids 0 * BN⟩ : Tile .nat [BN]) from by
      rw [evalOp_add, evalOp_mul]
      simp only [evalOp_arange, evalOp_constNat, evalOp_ref, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, hsn, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul, Tile.vec_data]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add NumericDType.nat Broadcast.scalarL (Op.ref .nat [] "o_offset")
          (Op.ref .nat [BN] "o_range"))) _
        = some (⟨fun idx : TileIndex [BN] =>
            (Out.cast, case1OutStoreOffsetG s H BN stride_oz stride_oh idx.1)⟩ : Tile .ptr [BN]) from by
      rw [evalOp_ptrAdd', evalOp_add]
      simp only [evalOp_ptrBase', evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      · rfl
      · simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Broadcast.leftIndex,
          Broadcast.rightIndex, NumericDType.add, case1OutStoreOffsetG]
        omega))]
  set sP := (((sL.setReg "o_offset" .nat [] (Tile.scalar ((s.pids 1 / H) * stride_oz + (s.pids 1 % H) * stride_oh))).setReg
      "o_range" .nat [BN] (⟨fun idx : TileIndex [BN] => idx.1.val + s.pids 0 * BN⟩ : Tile .nat [BN])).setReg
      "o_ptrs" .ptr [BN] (⟨fun idx : TileIndex [BN] => (Out.cast, case1OutStoreOffsetG s H BN stride_oz stride_oh idx.1)⟩ : Tile .ptr [BN]))
    with hsPdef
  set offFn : TileIndex [BN] → Nat := fun idx => case1OutStoreOffsetG s H BN stride_oz stride_oh idx.1 with hoffFn
  set valFn : TileIndex [BN] → ℝ := fun idx => case1OutClosedFormG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws idx.1 with hvalFn
  set mskFn : TileIndex [BN] → Bool := fun idx => decide (idx.1.val + s.pids 0 * BN < NKV_CTX) with hmskFn
  have hStore : stepStmt (Stmt.store .real [BN] (MemAccess.ptr (Op.ref .ptr [BN] "o_ptrs"))
      (Op.ref .real [BN] "o")
      (MaskOpt.mask (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "o_range")
        (Op.constNat NKV_CTX)))) sP
      = some ((TileShape.allIndices [BN]).foldl
          (fun acc idx => if mskFn idx then acc.writeMem Out (offFn idx) (valFn idx) else acc) sP) := by
    have hoP : sP.regs .real [BN] "o" = some ⟨fun idx : TileIndex [BN] =>
        some (case1OutClosedFormG s Q K M sm_scale H H_KV ROUND_CTX BN BN BD
          stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws idx.1)⟩ := by
      rw [hsPdef, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      exact ho
    have hrangeP : sP.regs .nat [BN] "o_range"
        = some (⟨fun idx : TileIndex [BN] => idx.1.val + s.pids 0 * BN⟩ : Tile .nat [BN]) := by
      rw [hsPdef, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        BlockState.setReg_same]
    have hptrsP : sP.regs .ptr [BN] "o_ptrs"
        = some (⟨fun idx : TileIndex [BN] => (Out.cast, case1OutStoreOffsetG s H BN stride_oz stride_oh idx.1)⟩ : Tile .ptr [BN]) := by
      rw [hsPdef, BlockState.setReg_same]
    simp only [stepStmt, evalOp_ref, hoP, hrangeP, hptrsP, evalOp_lt, evalOp_constNat,
      Option.bind_eq_bind, Option.bind_some, Option.map_some]
    refine congrArg some ?_
    refine congrArg (fun f => List.foldl f sP (TileShape.allIndices [BN])) ?_
    funext acc idx
    by_cases hm : idx.1.val + s.pids 0 * BN < NKV_CTX
    · simp only [Tile.cop, Tile.scalar, Broadcast.rightIndex, Broadcast.leftIndex,
        ComparableDType.lt, hmskFn, hm, decide_true, if_true, hoffFn, hvalFn, case1OutStoreOffsetG,
        Region.cast_id, BlockState.writeMemTyped_real, FloatDType.storeValue,
        FloatDType.real_toWithBot, WithBot.unbotD_coe, WithBot.unbotD_some]
    · simp only [Tile.cop, Tile.scalar, Broadcast.rightIndex, Broadcast.leftIndex,
        ComparableDType.lt, hmskFn, hm, decide_false, Bool.false_eq_true, if_false,
        decide_eq_true_eq]
  rw [stepStmts.cons_some hStore, stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_⟩
  · -- per-column readback (the store offsets are injective in the lane)
    intro i
    have hinj : Function.Injective offFn := by
      intro a b hab
      simp only [hoffFn, case1OutStoreOffsetG] at hab
      exact Prod.ext (Fin.ext (by omega)) (by cases a.2; cases b.2; rfl)
    have hread := BlockState.scatter_readback_masked_nd (region := Out) sP offFn valFn mskFn
      hinj (i, PUnit.unit)
    rw [show case1OutStoreOffsetG s H BN stride_oz stride_oh i = offFn (i, PUnit.unit) from rfl, hread]
    have hmsk_iff : (mskFn (i, PUnit.unit) = Bool.true) ↔ case1OutActiveG s BN NKV_CTX i := by
      simp only [hmskFn, decide_eq_true_eq, case1OutActiveG]; omega
    by_cases hi : case1OutActiveG s BN NKV_CTX i
    · rw [if_pos (hmsk_iff.mpr hi), if_pos hi]
    · rw [if_neg (fun h => hi (hmsk_iff.mp h)), if_neg hi]
      rw [show offFn (i, PUnit.unit) = case1OutStoreOffsetG s H BN stride_oz stride_oh i from rfl]
      simp only [hsPdef, BlockState.setReg_readMem]
      unfold BlockState.readMem; rw [hmem]
  · -- the frame: cells outside the active write window are untouched
    intro r o hcond
    have hmiss : r = Out → ∀ k ∈ TileShape.allIndices [BN], mskFn k = Bool.true → offFn k ≠ o := by
      intro hr k _ hmk
      have hact : case1OutActiveG s BN NKV_CTX k.1 := by
        rw [hmskFn] at hmk
        have h2 := of_decide_eq_true hmk
        unfold case1OutActiveG
        omega
      exact fun hko => hcond hr k.1 hact hko.symm
    rw [score_foldl_writeMem_masked_frame Out offFn valFn mskFn (TileShape.allIndices [BN])
      sP r o hmiss, hsPdef]
    simp only [BlockState.setReg_mem]

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The case-1 `⊨[R]` io headline — the first consumer of the
single-output attention fold skin `StreamMasked3DKernelIO₃`.** For every
rounding model `R`, the case-1
(`(SLIDING_WINDOW, COMPLEMENT, IS_EVEN_M, IS_EVEN_N) = (1,0,1,1)`)
`_score_kernel` surface implements, on its three-stream single-output io
signature, the **ideal-ℝ masked-`exp2` column sum**
`attentionScoreCase1IOSpec` — the existing `case1OutClosedFormG` closed form
restated over the streamed `Q`/`K`/`M` tiles, sliding-window mask kept in its
**verbatim ℕ-truncated** `case1DistG` arithmetic. The output grid is the
`.real` default (the store-side `.to(Out.type.element_ty)` cast erases to the
identity at translation), so at every `R` the active `Out` cells carry the
exact fold values, gated by the kernel's own `o_range < NKV_CTX` store mask.

**Hypothesis provenance** (all truth-forced): `0 < BLOCK_N` and
`BLOCK_N ∣ ROUND_CTX` shape the query walk (`T = ROUND_CTX / BLOCK_N` full
blocks; inherited verbatim from the exact headline
`attention_score_python_case1_output_summary_general`); `0 < ROUND_CTX` is
**new** and forced by the io form itself — the `K` tile is loaded *before*
the loop, so its safety bound must come from some stream step, i.e. the
step space `Fin (ROUND_CTX / BLOCK_N)` must be inhabited. The exact
headline's `hundef` is **not** a hypothesis here — the skin's Hoare triple
carries the `undef` pin itself. The store-lane injectivity is arithmetic
(`omega`), so no injectivity hypotheses are needed.

**Scope disclosed**: this headline showcases the io face on **case 1 only**;
the complement-window / uneven cases 2–4 stay on the existing exact coverage
(`attention_score_final_store_slice_*` and the case-1 exact stack). The
port's known fidelity gaps (casts erased to identity,
`@triton.autotune`/`@triton.heuristics` not modeled) are inherited as-is
from the surface. -/
specification attention_score_case1_io_correctness (R : RoundingModel)
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
     stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
     BN BD : Nat) (sm_scale : ℝ)
    (hBN : 0 < BN) (hRC : 0 < ROUND_CTX) (hdvd : BN ∣ ROUND_CTX) :
    attentionScoreCase1IO Q K M Out
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
        stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws BN BD sm_scale ⊨[R]
      fun p₀ _ _ xs ys zs j =>
        attentionScoreCase1IOSpec BN BD (ROUND_CTX / BN) swo sws
          (Nat.div_pos (Nat.le_of_dvd hRC hdvd) hBN) sm_scale p₀ xs ys zs j := by
  have hT : 0 < ROUND_CTX / BN := Nat.div_pos (Nat.le_of_dvd hRC hdvd) hBN
  refine StreamMasked3DKernelIO₃.ImplementsR.intro _ ?_ ?_ ?_
  · exact score_case1_flattenOkG Q K M Out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws BN BD sm_scale
  · -- the safety walk
    intro bounds s xs ys zs _hx _hy _hz hbr1 hbr2 hbr3 hbw
    simp only [attentionScoreCase1IO] at hbr1 hbr2 hbr3 hbw ⊢
    exact score_traceSafeR_case1 R bounds Q K M Out sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws BN BD s
      hBN hRC hdvd
      (fun t j => hbr1 t j trivial) (fun t j => hbr2 t j trivial)
      (fun t j => hbr3 t j trivial) (fun j hj => hbw j hj)
  · -- the rounded Hoare triple: exact invariant stack + cast-free collapses
    intro s₀ xs ys zs hu hx hy hz
    simp only [attentionScoreCase1IO] at hx hy hz ⊢
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hu]
    obtain ⟨sp, hpre, hpids0, hmem0, hundef0, hsn0, hoz0, hoh0, hqks0, ho0, hk0, hQbp0, hmp0, hlo0, hhi0⟩ :=
      score_preLoop_evalG s₀ Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk hundef'
    obtain ⟨sL, hloop, hpidsL, hmemL, hozL, hohL, hsnL, hoL⟩ :=
      score_loop_evalG s₀ sp Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD swo sws
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn
        hBN hdvd hpids0 hmem0 hundef0 hsn0 hoz0 hoh0 hqks0 ho0 hk0 hQbp0 hmp0 hlo0 hhi0
    obtain ⟨sF, hpost, hread, hframe⟩ :=
      score_post_evalFrameG s₀ sL Q K M Out sm_scale H H_KV ROUND_CTX NKV_CTX BN BD
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn
        stride_oz stride_oh swo sws hmemL hozL hohL hsnL hoL
    refine ⟨sF, ?_, ?_, ?_⟩
    · -- termination under `execR R` (everything cast-free)
      show execR R (attention_score_kernel Q K M Out
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
        stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws
        BN BD BN sm_scale Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel s₀ = some sF
      unfold execR
      rw [score_case1_body_splitG Q K M Out
          stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
          stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX swo sws BN BD sm_scale,
        stepStmtsR_append,
        score_preLoopG_castFree R Q K M sm_scale H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
          stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk s₀,
        hpre, Option.bind_some,
        stepStmtsR_cons_some (show stepStmtR R (Stmt.forRangeDyn "start_m" (Op.ref .nat [] "lo")
            (Op.ref .nat [] "hi") (Op.constNat BN)
            (attentionScoreCase1LoopBodyG BN BD swo sws N_CTX)) sp = some sL from by
          rw [score_stepStmtR_forRangeDyn, evalOpR_ref, hlo0, evalOpR_ref, hhi0,
            score_evalOpR_constNat]
          simp only [Option.bind_eq_bind, Option.bind_some, Tile.scalar_data]
          rw [stepForRangeAuxR_castFree R _ (score_loopBodyG_castFree R BN BD swo sws N_CTX) "start_m"]
          rw [stepForRangeAux.forRangeDyn_unfold,
            show evalOp (Op.ref .nat [] "lo") sp = some (Tile.scalar 0) from by
              rw [evalOp_ref]; exact hlo0,
            show evalOp (Op.ref .nat [] "hi") sp = some (Tile.scalar ROUND_CTX) from by
              rw [evalOp_ref]; exact hhi0,
            evalOp_constNat] at hloop
          simpa only [Option.bind_eq_bind, Option.bind_some, Tile.scalar_data] using hloop),
        score_postG_castFree R Out BN NKV_CTX stride_oz stride_oh sL]
      exact hpost
    · -- terminal readback = the streamed closed form (rounded at `.real` = exact)
      intro j hj
      rw [BlockState.readMemAs_real]
      have hr := hread j
      rw [if_pos (show case1OutActiveG s₀ BN NKV_CTX j from hj)] at hr
      rw [show s₀.pids 1 / H * stride_oz + s₀.pids 1 % H * stride_oh + (s₀.pids 0 * BN + j.val)
          = case1OutStoreOffsetG s₀ H BN stride_oz stride_oh j from rfl, hr]
      rw [score_case1_closedFormG_eq_ioSpec s₀ Q K M sm_scale H H_KV ROUND_CTX BN BD
        stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws hT
        xs ys zs (fun t j' => hx t j' trivial) (fun t j' => hy t j' trivial)
        (fun t j' => hz t j' trivial) j]
      simp [FloatDType.ofReal]
    · -- the frame: cells outside the active write window are untouched
      intro r o hcond
      have hmiss : r = Out → ∀ i : Fin BN, case1OutActiveG s₀ BN NKV_CTX i →
          o ≠ case1OutStoreOffsetG s₀ H BN stride_oz stride_oh i := by
        intro hr i hact
        rcases hcond with hne | hall
        · exact absurd hr hne
        · exact hall i hact
      rw [hframe r o hmiss, hmemL]

end IOFaceCase1

end VeriTile.Bench.TritonBenchG.AttentionScore
