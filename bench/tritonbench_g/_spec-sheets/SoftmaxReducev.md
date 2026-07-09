# Spec sheet — `bench/tritonbench_g/softmax_reducev/SoftmaxReducev.lean`

**Python source:** `bench/tritonbench_g/softmax_reducev/softmax_reducev.py`

## Public theorem: `softmax_reducev_final_store_slice_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the final softmax-reduce-v store slice. -/
```
</details>

**Statement:**
```lean
theorem softmax_reducev_final_store_slice_compute_correct
    (Acc ESum Out : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d
      stride_es_bs stride_es_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := softmax_reducev_final_store_slice Acc ESum Out
        stride_acc_bs stride_acc_h stride_acc_d stride_es_bs stride_es_h
        stride_obs stride_oh stride_od BLOCK_DMODEL)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i =>
        softmaxReducevFinalSpec s Acc ESum stride_acc_bs stride_acc_h
          stride_acc_d stride_es_bs stride_es_h i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)`

**Closed-form spec defs (transitive):** `outOffset`, `softmax_reducev_final_store_slice`, `softmaxReducevFinalSpec`, `dIndex`, `accOffset`, `eSumOffset`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (stride_obs stride_oh stride_od : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_obs + s.pids 1 * stride_oh + dIndex s i * stride_od
```
</details>

<details><summary><code>softmax_reducev_final_store_slice</code></summary>

```
/-- Proof-oriented final normalization/store slice of `softmax_reducev.py`'s
`_fwd_kernel`.

The full kernel performs a streaming softmax over V blocks, maintaining `acc`
and `e_sum`. This slice starts after that loop with precomputed `Acc` and `ESum`
regions, then proves the final `acc / e_sum` writeback to `Out` across the
`BLOCK_DMODEL` vector. -/
```
```lean
def softmax_reducev_final_store_slice
    (Acc ESum Out : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d
      stride_es_bs stride_es_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  e_sum = tl.load(ESum + cur_batch * $(stride_es_bs) + cur_head * $(stride_es_h))
  acc = tl.load(Acc + cur_batch * $(stride_acc_bs) + cur_head * $(stride_acc_h) +
      offs_d * $(stride_acc_d))
  out = acc / e_sum
  tl.store(Out + cur_batch * $(stride_obs) + cur_head * $(stride_oh) +
      offs_d * $(stride_od), out)
}
```
</details>

<details><summary><code>softmaxReducevFinalSpec</code></summary>

```lean
noncomputable def softmaxReducevFinalSpec
    (s : BlockState) (Acc ESum : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d stride_es_bs stride_es_h : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  s.readMem Acc (accOffset s stride_acc_bs stride_acc_h stride_acc_d i) /
    s.readMem ESum (eSumOffset s stride_es_bs stride_es_h)
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (_s : BlockState) (i : Fin BLOCK_DMODEL) : Nat :=
  i.val
```
</details>

<details><summary><code>accOffset</code></summary>

```lean
def accOffset
    (s : BlockState) (stride_acc_bs stride_acc_h stride_acc_d : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_acc_bs + s.pids 1 * stride_acc_h + dIndex s i * stride_acc_d
```
</details>

<details><summary><code>eSumOffset</code></summary>

```lean
def eSumOffset (s : BlockState) (stride_es_bs stride_es_h : Nat) : Nat :=
  s.pids 0 * stride_es_bs + s.pids 1 * stride_es_h
```
</details>

## Public theorem: `softmax_reducev_genuine_output_compute_correct_general`

<details><summary>docstring</summary>

```
/-- **General genuine closed-form `Out`-store correctness.** Every output lane of the
`softmax_reducev` kernel holds the genuine softmax-weighted V reduction
`softmaxReducevWeightedSum` of the loaded logits / gathered V rows. Fully
dimension-parameterized: `BLOCK_N` (key block), `BLOCK_DMODEL` (channel), and all
strides. Side conditions: `0 < BLOCK_N`, `0 < BLOCK_DMODEL`, `BLOCK_N ∣ srSeqLen`
(`% = 0`), `0 < srSeqLen`, contiguous output offset injectivity, `hundef`, and the
finite running max `mr`. -/
```
</details>

**Statement:**
```lean
theorem softmax_reducev_genuine_output_compute_correct_general
    (Logics V Out : RegionName) (BLoc : Region .int) (BStartLoc BSeqLen : Region .nat)
    (mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int)
    (hD : 0 < BLOCK_DMODEL) (hN : 0 < BLOCK_N)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (hseqmod : srSeqLen s BSeqLen.cast % BLOCK_N = 0) (hseqpos : 0 < srSeqLen s BSeqLen.cast)
    (hOutInj : ∀ s0 : BlockState, Function.Injective (fun i : Fin BLOCK_DMODEL => outOffsetG s0 sob soh sod i))
    (mr : ℝ)
    (hM : srRunningMax (srQkFG s Logics BStartLoc.cast BSeqLen.cast slh slb)
      (srVFG s V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL)
      (srSeqLen s BSeqLen.cast) (⟨0, hD⟩ : Fin BLOCK_DMODEL) = (mr : WithBot ℝ)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
        mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N other_kv_index)
      (initialState := s)
      (write := fun d : Fin BLOCK_DMODEL => some (Out, outOffsetG s sob soh sod d))
      (expected := fun d : Fin BLOCK_DMODEL =>
        softmaxReducevWeightedSum (srQkFG s Logics BStartLoc.cast BSeqLen.cast slh slb) mr
          (srVFG s V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL) d)
```

**Assumptions / layout contracts:**
- `hD : 0 < BLOCK_DMODEL`
- `hN : 0 < BLOCK_N`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `hseqmod : srSeqLen s BSeqLen.cast % BLOCK_N = 0`
- `hseqpos : 0 < srSeqLen s BSeqLen.cast`
- `hOutInj : ∀ s0 : BlockState, Function.Injective (fun i : Fin BLOCK_DMODEL => outOffsetG s0 sob soh sod i)`
- `hM : srRunningMax (srQkFG s Logics BStartLoc.cast BSeqLen.cast slh slb)
      (srVFG s V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL)
      (srSeqLen s BSeqLen.cast) (⟨0, hD⟩ : Fin BLOCK_DMODEL) = (mr : WithBot ℝ)`

**Closed-form spec defs (transitive):** `srSeqLen`, `outOffsetG`, `srRunningMax`, `srQkFG`, `srVFG`, `softmax_reducev_surface`, `softmaxReducevWeightedSum`, `srKeysUpto`, `srQkG`, `srVG`, `softmaxReducevAcc`, `softmaxReducevDenom`, `srStartLoc`, `srVIndexG`, `softmaxWeight`, `srOffBLocG`

<details><summary><code>srSeqLen</code></summary>

```
/-- The loop bound: `cur_batch_seq_len = BSeqLen[cur_batch]`. -/
```
```lean
def srSeqLen (s : BlockState) (BSeqLen : RegionName) : Nat :=
  s.readMemValue .nat BSeqLen (s.pids 0)
```
</details>

<details><summary><code>outOffsetG</code></summary>

```
/-- General output offset `cur_batch·stride_obs + cur_head·stride_oh + d·stride_od`. -/
```
```lean
def outOffsetG (s : BlockState) (stride_obs stride_oh stride_od : Nat) {BLOCK_DMODEL : Nat}
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_obs + s.pids 1 * stride_oh + i.val * stride_od
```
</details>

<details><summary><code>srRunningMax</code></summary>

```
/-- **⊥-seeded running max** of the streamed key prefix `[0, hi)`, exactly the
value the kernel carries in its `e_max` register (`float("-inf")` seeds at `⊥`).
The `WithBot` `⊔`-fold of the coerced per-key logits; `⊥` on the empty / `hi = 0`
window (the kernel's preLoop init). -/
```
```lean
noncomputable def srRunningMax {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (hi : Nat)
    (d : Fin BLOCK_DMODEL) : WithBot ℝ :=
  ((srKeysUpto qk v hi d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
```
</details>

<details><summary><code>srQkFG</code></summary>

```
/-- General per-token logit over valid tokens. -/
```
```lean
noncomputable def srQkFG (s0 : BlockState) (Logics BStartLoc BSeqLen : RegionName)
    (stride_logic_h stride_logic_bs : Nat) : Fin (srSeqLen s0 BSeqLen) → ℝ :=
  fun n => srQkG s0 Logics BStartLoc stride_logic_h stride_logic_bs n.val
```
</details>

<details><summary><code>srVFG</code></summary>

```
/-- General per-token gathered V-row over valid tokens and `Fin BLOCK_DMODEL`. -/
```
```lean
noncomputable def srVFG (s0 : BlockState) (V : RegionName) (BLoc : Region .int)
    (BSeqLen : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_vbs stride_vh stride_vd : Nat)
    (BLOCK_DMODEL : Nat) : Fin (srSeqLen s0 BSeqLen) → Fin BLOCK_DMODEL → ℝ :=
  fun n d => srVG s0 V BLoc BSeqLen max_input_len stride_b_loc_b stride_b_loc_s
    stride_vbs stride_vh stride_vd n.val d.val
```
</details>

<details><summary><code>softmax_reducev_surface</code></summary>

```
/-- Lean port of `softmax_reducev.py`'s `_fwd_kernel`.

This records the streaming softmax recurrence over token blocks, the signed
`B_Loc` gather with Python's `other_kv_index` sentinel, the V gather, and the
final normalized writeback. -/
```
```lean
def softmax_reducev_surface
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat)
    (max_input_len
      stride_logic_h stride_logic_bs
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_b_loc_b stride_b_loc_s
      BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  cur_batch_seq_len = tl.load(BSeqLen + cur_batch)
  cur_batch_start_loc = tl.load(BStartLoc + cur_batch)
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_v = cur_head * $(stride_vh) + offs_d[None, :] * $(stride_vd)
  off_b_loc = cur_batch * $(stride_b_loc_b) +
    ($(max_input_len) - cur_batch_seq_len) * $(stride_b_loc_s)
  v_ptrs = V + off_v
  e_max = float("-inf")
  e_sum = 0.0
  acc = tl.zeros([$(BLOCK_DMODEL)], dtype=tl.float32)
  for start_n in range($(0), cur_batch_seq_len, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    v_index = tl.load(BLoc + off_b_loc +
      (start_n + offs_n) * $(stride_b_loc_s),
      mask=(start_n + offs_n) < cur_batch_seq_len, other=$(other_kv_index))
    qk = tl.load(Logics + cur_head * $(stride_logic_h) +
        (cur_batch_start_loc + start_n + offs_n) * $(stride_logic_bs),
      mask=start_n + offs_n < cur_batch_seq_len, other=float("-inf"))
    n_e_max = tl.maximum(tl.max(qk, 0), e_max)
    old_scale = tl.exp(e_max - n_e_max)
    p = tl.exp(qk - n_e_max)
    e_sum = e_sum * old_scale + tl.sum(p, 0)
    v = tl.load(v_ptrs + v_index[:, None] * $(stride_vbs))
    acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
    e_max = n_e_max
  }
  acc = acc / e_sum
  off_o = cur_batch * $(stride_obs) + cur_head * $(stride_oh) + offs_d * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc)
}
```
</details>

<details><summary><code>softmaxReducevWeightedSum</code></summary>

```
/-- The full normalized closed form
`out[d] = acc[d] / e_sum = Σ_n softmax(qk)[n] · V[v_index[n], d]`. This is what
`softmax_reducev.py` stores to `Out[cur_batch, cur_head, d]`, stated purely over
the input logits `qk` and gathered value rows `v` — no reference to the executed
kernel. -/
```
```lean
noncomputable def softmaxReducevWeightedSum {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (mMax : ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ)
    (d : Fin BLOCK_DMODEL) : ℝ :=
  softmaxReducevAcc qk mMax v d / softmaxReducevDenom qk mMax
```
</details>

<details><summary><code>srKeysUpto</code></summary>

```
/-- Per-channel streamed key list over the window `[0, hi)`: the valid tokens
`n < hi`, in index order, each carrying `(qk[n], v[n][d])`. After `c` blocks
`hi = c · BLOCK_N`, this is the prefix the kernel has streamed for channel `d`. -/
```
```lean
noncomputable def srKeysUpto {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (hi : Nat)
    (d : Fin BLOCK_DMODEL) : List (ℝ × ℝ) :=
  (List.finRange S).filterMap (fun n : Fin S =>
    if n.val < hi then some (qk n, v n d) else none)
```
</details>

<details><summary><code>srQkG</code></summary>

```
/-- General logit for token `n`: `Logics[cur_head·stride_logic_h + (start_loc + n)·stride_logic_bs]`. -/
```
```lean
def srQkG (s : BlockState) (Logics BStartLoc : RegionName)
    (stride_logic_h stride_logic_bs : Nat) (n : Nat) : ℝ :=
  s.readMem Logics (s.pids 1 * stride_logic_h + (srStartLoc s BStartLoc + n) * stride_logic_bs)
```
</details>

<details><summary><code>srVG</code></summary>

```
/-- General gathered V-row entry: `V[v_index[n]·stride_vbs + cur_head·stride_vh + d·stride_vd]`. -/
```
```lean
def srVG (s : BlockState) (V : RegionName) (BLoc : Region .int) (BSeqLen : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_vbs stride_vh stride_vd : Nat)
    (n d : Nat) : ℝ :=
  s.readMem V ((srVIndexG s BLoc BSeqLen max_input_len stride_b_loc_b stride_b_loc_s n * stride_vbs).toNat
    + s.pids 1 * stride_vh + d * stride_vd)
```
</details>

<details><summary><code>softmaxReducevAcc</code></summary>

```
/-- The unnormalized weighted V reduction
`acc[d] = Σ_n exp(qk[n] - M)·V[v_index[n], d]` — the genuine closed form of the
streamed `Acc[d]` value: the gathered value rows weighted by the unnormalized
softmax probabilities. -/
```
```lean
noncomputable def softmaxReducevAcc {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (mMax : ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ)
    (d : Fin BLOCK_DMODEL) : ℝ :=
  ∑ n : Fin S, softmaxWeight qk mMax n * v n d
```
</details>

<details><summary><code>softmaxReducevDenom</code></summary>

```
/-- The softmax normalizer `e_sum = Σ_n exp(qk[n] - M)` — the genuine closed form
of the streamed `ESum` value. -/
```
```lean
noncomputable def softmaxReducevDenom {S : Nat} (qk : Fin S → ℝ) (mMax : ℝ) : ℝ :=
  ∑ n : Fin S, softmaxWeight qk mMax n
```
</details>

<details><summary><code>srStartLoc</code></summary>

```
/-- `cur_batch_start_loc = BStartLoc[cur_batch]`. -/
```
```lean
def srStartLoc (s : BlockState) (BStartLoc : RegionName) : Nat :=
  s.readMemValue .nat BStartLoc (s.pids 0)
```
</details>

<details><summary><code>srVIndexG</code></summary>

```
/-- General paged-KV index for token `n`: `BLoc[off_b_loc + n·stride_b_loc_s]`. -/
```
```lean
def srVIndexG (s : BlockState) (BLoc : Region .int) (BSeqLen : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s : Nat) (n : Nat) : Int :=
  s.readMemValue .int (Region.cast BLoc)
    (srOffBLocG s BSeqLen max_input_len stride_b_loc_b stride_b_loc_s + n * stride_b_loc_s)
```
</details>

<details><summary><code>softmaxWeight</code></summary>

```
/-- Unnormalized softmax weight `exp(qk[n] - M)` for token `n`, with running max
`M = mMax`. -/
```
```lean
noncomputable def softmaxWeight {S : Nat} (qk : Fin S → ℝ) (mMax : ℝ) (n : Fin S) : ℝ :=
  Real.exp (qk n - mMax)
```
</details>

<details><summary><code>srOffBLocG</code></summary>

```
/-- General `off_b_loc = cur_batch·stride_b_loc_b + (max_input_len − seqlen)·stride_b_loc_s`. -/
```
```lean
def srOffBLocG (s : BlockState) (BSeqLen : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s : Nat) : Nat :=
  s.pids 0 * stride_b_loc_b + (max_input_len - srSeqLen s BSeqLen) * stride_b_loc_s
```
</details>
