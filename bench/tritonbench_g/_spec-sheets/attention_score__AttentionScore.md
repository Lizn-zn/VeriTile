# Spec sheet — `bench/tritonbench_g/attention_score/AttentionScore.lean`

**Python source:** `bench/tritonbench_g/attention_score/attention_score.py`

## Public theorem: `attention_score_python_case1_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Public general Python case-1 output summary (genuine closed form).** The full
attention-score surface lowers to the algorithm layer, and the kernel writes the
genuine closed-form score `case1OutClosedFormG` to every active output column —
the dimension-parameterized case-1 output summary (symbolic shape/strides). -/
```
</details>

**Statement:**
```lean
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
          stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kk stride_kn swo sws i)
```

**Assumptions / layout contracts:**
- `hBNpos : 0 < BN`
- `hdvd : BN ∣ ROUND_CTX`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `fun i : Fin BN => case1OutActiveG s BN NKV_CTX i`

**Closed-form spec defs (transitive):** `attention_score_kernel`, `case1OutActiveG`, `case1OutStoreOffsetG`, `case1OutClosedFormG`, `case1ColSumG`, `case1MaskG`, `case1WeightG`, `case1DistG`, `case1RawScoreG`, `case1QKOffsetQG`, `case1QKOffsetKG`, `case1MOffsetG`, `case1QElemG`, `case1KElemG`

<details><summary><code>attention_score_kernel</code></summary>

```
/-- DSL port of `attention_score.py`'s `_score_kernel`.

The proof parameter `hBlockMN` carries the Python wrapper invariant
`BLOCK_M == BLOCK_N` so the DSL can type the source `tl.zeros([BLOCK_M])`
against the later `tl.sum(p, axis=0)` vector. -/
```
```lean
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
```
</details>

<details><summary><code>case1OutActiveG</code></summary>

```
/-- **General** store mask: `o_range = start_n·BN + i < NKV_CTX`. -/
```
```lean
def case1OutActiveG (s : BlockState) (BN NKV_CTX : Nat) (i : Fin BN) : Prop :=
  s.pids 0 * BN + i.val < NKV_CTX

instance (s : BlockState) (BN NKV_CTX : Nat) (i : Fin BN) :
    Decidable (case1OutActiveG s BN NKV_CTX i) := by unfold case1OutActiveG; infer_instance
```
</details>

<details><summary><code>case1OutStoreOffsetG</code></summary>

```
/-- **General** case-1 store offset for output column `i`:
`off_z·stride_oz + off_h·stride_oh + (start_n·BN + i)`. -/
```
```lean
def case1OutStoreOffsetG (s : BlockState) (H BN stride_oz stride_oh : Nat) (i : Fin BN) : Nat :=
  (s.pids 1 / H) * stride_oz + (s.pids 1 % H) * stride_oh + (s.pids 0 * BN + i.val)
```
</details>

<details><summary><code>case1OutClosedFormG</code></summary>

```
/-- **General genuine closed-form attention score** for output key column `j`:
the masked-`exp2` query-row column sum over the `ROUND_CTX/BLOCK_M` query blocks. -/
```
```lean
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
```
</details>

<details><summary><code>case1ColSumG</code></summary>

```
/-- General inner per-query-block column sum
`Σ_{i<BLOCK_M} (if mask(c,i,j) then weight(c,i,j) else 0)`. -/
```
```lean
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
```
</details>

<details><summary><code>case1MaskG</code></summary>

```
/-- General sliding-window mask (non-complement): `0 ≤ dist ∧ dist < sliding_window_size`. -/
```
```lean
def case1MaskG (s : BlockState)
    (BLOCK_M BLOCK_N sliding_window_offset sliding_window_size c i j : Nat) : Prop :=
  0 ≤ case1DistG s BLOCK_M BLOCK_N sliding_window_offset c i j
    ∧ case1DistG s BLOCK_M BLOCK_N sliding_window_offset c i j < sliding_window_size

instance (s : BlockState)
    (BLOCK_M BLOCK_N sliding_window_offset sliding_window_size c i j : Nat) :
    Decidable (case1MaskG s BLOCK_M BLOCK_N sliding_window_offset sliding_window_size c i j) := by
  unfold case1MaskG; infer_instance
```
</details>

<details><summary><code>case1WeightG</code></summary>

```
/-- General per-cell masked softmax weight for query block `c`:
`exp2( sm_scale · log2e · rawScore(c·BLOCK_M+i, start_n·BLOCK_N+j) − M[c·BLOCK_M+i] )`. -/
```
```lean
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
```
</details>

<details><summary><code>case1DistG</code></summary>

```
/-- General nat-truncated sliding-window distance for in-block query row `i`, key
column `j`, query block `c` (`start_m = c·BLOCK_M`):
`dist = ((((i − j) + c·BLOCK_M) − start_n·BLOCK_N) + sliding_window_offset)` over `ℕ`. -/
```
```lean
def case1DistG (s : BlockState) (BLOCK_M BLOCK_N sliding_window_offset c i j : Nat) : Nat :=
  (((i - j) + c * BLOCK_M) - s.pids 0 * BLOCK_N) + sliding_window_offset
```
</details>

<details><summary><code>case1RawScoreG</code></summary>

```
/-- General raw, unscaled QK dot for query row `r`, global key column `n`:
`Σ_{d<BLOCK_DMODEL} Q[r,d]·K[d,n]` (elements via `case1QElemG`/`case1KElemG`). -/
```
```lean
noncomputable def case1RawScoreG
    (s : BlockState) (Q K : RegionName)
    (BLOCK_DMODEL stride_qm stride_qk stride_kk stride_kn : Nat)
    (qoff koff : Nat) (r n : Nat) : ℝ :=
  Finset.univ.sum (fun d : Fin BLOCK_DMODEL =>
    case1QElemG s Q stride_qm stride_qk qoff r d.val
      * case1KElemG s K stride_kk stride_kn koff d.val n)
```
</details>

<details><summary><code>case1QKOffsetQG</code></summary>

```
/-- General `Q` base offset `off_z·stride_qz + off_h·stride_qh`
(`off_z = off_hz / H`, `off_h = off_hz % H`, `off_hz = s.pids 1`). -/
```
```lean
def case1QKOffsetQG (s : BlockState) (H stride_qz stride_qh : Nat) : Nat :=
  (s.pids 1 / H) * stride_qz + (s.pids 1 % H) * stride_qh
```
</details>

<details><summary><code>case1QKOffsetKG</code></summary>

```
/-- General `K` base offset `off_z·stride_kz + off_hkv·stride_kh`
(`off_hkv = off_h / (H / H_KV)`). -/
```
```lean
def case1QKOffsetKG (s : BlockState) (H H_KV stride_kz stride_kh : Nat) : Nat :=
  (s.pids 1 / H) * stride_kz + ((s.pids 1 % H) / (H / H_KV)) * stride_kh
```
</details>

<details><summary><code>case1MOffsetG</code></summary>

```
/-- General `M` offset for query row `r`: `off_hz·ROUND_CTX + r`. -/
```
```lean
def case1MOffsetG (s : BlockState) (ROUND_CTX r : Nat) : Nat := s.pids 1 * ROUND_CTX + r
```
</details>

<details><summary><code>case1QElemG</code></summary>

```
/-- Query element `Q[r, d]` at `qoff + r·stride_qm + d·stride_qk` (the `Q`
block-ptr `[ROUND_CTX, BLOCK_DMODEL]` layout; base `qoff = case1QKOffsetQG`). -/
```
```lean
noncomputable def case1QElemG (s : BlockState) (Q : RegionName)
    (stride_qm stride_qk qoff r d : Nat) : ℝ :=
  s.readMem Q (qoff + r * stride_qm + d * stride_qk)
```
</details>

<details><summary><code>case1KElemG</code></summary>

```
/-- Key element `K[d, n]` at `koff + d·stride_kk + n·stride_kn` (the `K`
block-ptr `[BLOCK_DMODEL, NKV_CTX]` layout; base `koff = case1QKOffsetKG`,
global key column `n`). -/
```
```lean
noncomputable def case1KElemG (s : BlockState) (K : RegionName)
    (stride_kk stride_kn koff d n : Nat) : ℝ :=
  s.readMem K (koff + d * stride_kk + n * stride_kn)
```
</details>

## Public theorem: `attention_score_case1_io_correctness`

<details><summary>docstring</summary>

```
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
```
</details>

**Statement:**
```lean
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
          (Nat.div_pos (Nat.le_of_dvd hRC hdvd) hBN) sm_scale p₀ xs ys zs j
```

**Assumptions / layout contracts:**
- `hBN : 0 < BN`
- `hRC : 0 < ROUND_CTX`
- `hdvd : BN ∣ ROUND_CTX`

**Closed-form spec defs (transitive):** `attentionScoreCase1IO`, `attentionScoreCase1IOSpec`, `attention_score_kernel`, `attentionScoreCase1IOMask`, `attentionScoreCase1IODist`

<details><summary><code>attentionScoreCase1IO</code></summary>

```
/-- **Streaming IO signature** of the case-1
(`(SLIDING_WINDOW, COMPLEMENT, IS_EVEN_M, IS_EVEN_N) = (1,0,1,1)`)
`_score_kernel` surface on the three-stream single-output fold skin. -/
```
```lean
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
```
</details>

<details><summary><code>attentionScoreCase1IOSpec</code></summary>

```
/-- **Case-1 closed form on the streams**: the masked-`exp2` query-row column
sum (`case1OutClosedFormG` restated over the three streamed tiles) at output
key column `j` —
`Σ_{c<T} Σ_{i<BN} [mask(c,i,j)] · exp2(sm_scale·log2e·⟨Q-row, K-col⟩ − M[c·BN+i])`,
where the `Q` row is step `c`'s lane row `i`, the `K` column is the static
stream's column `j` (step-`0` slice — the window ignores `t`), and the `M`
value is step `c`'s lane `i`. -/
```
```lean
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
```
</details>

<details><summary><code>attention_score_kernel</code></summary>

```
/-- DSL port of `attention_score.py`'s `_score_kernel`.

The proof parameter `hBlockMN` carries the Python wrapper invariant
`BLOCK_M == BLOCK_N` so the DSL can type the source `tl.zeros([BLOCK_M])`
against the later `tl.sum(p, axis=0)` vector. -/
```
```lean
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
```
</details>

<details><summary><code>attentionScoreCase1IOMask</code></summary>

```
/-- Sliding-window mask on the streams (case 1, non-complement):
`0 ≤ dist ∧ dist < sws`, both over `ℕ` (the `0 ≤` conjunct is vacuous —
kept verbatim from the elaborated `boolAnd (ge dist 0) (lt dist sws)`). -/
```
```lean
def attentionScoreCase1IOMask (p₀ BN swo sws : Nat) (c i j : Nat) : Prop :=
  0 ≤ attentionScoreCase1IODist p₀ BN swo c i j
    ∧ attentionScoreCase1IODist p₀ BN swo c i j < sws

instance (p₀ BN swo sws c i j : Nat) :
    Decidable (attentionScoreCase1IOMask p₀ BN swo sws c i j) := by
  unfold attentionScoreCase1IOMask; infer_instance
```
</details>

<details><summary><code>attentionScoreCase1IODist</code></summary>

```
/-- Nat-truncated sliding-window distance on the streams — the **verbatim**
`case1DistG` arithmetic (`ℕ`-truncated subtraction, `start_m = c·BN`,
`start_n = p₀`), with the block index `c` in stream-step position. -/
```
```lean
def attentionScoreCase1IODist (p₀ BN swo : Nat) (c i j : Nat) : Nat :=
  (((i - j) + c * BN) - p₀ * BN) + swo
```
</details>

## Also present (pinned special-case summaries)
- `attention_score_final_store_slice_compute_correct`
- `attention_score_case1_genuine_compute_correct`
- `attention_score_case1_genuine_compute_correct_general`
