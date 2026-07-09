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
theorem attention_score_python_case1_output_summary_general
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

## Also present (pinned special-case summaries)
- `attention_score_final_store_slice_compute_correct`
- `attention_score_case1_genuine_compute_correct`
- `attention_score_case1_genuine_compute_correct_general`
