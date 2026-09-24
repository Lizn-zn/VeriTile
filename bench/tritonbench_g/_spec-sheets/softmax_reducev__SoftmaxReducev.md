# Spec sheet — `bench/tritonbench_g/softmax_reducev/SoftmaxReducev.lean`

**Python source:** `bench/tritonbench_g/softmax_reducev/softmax_reducev.py`

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
specification softmax_reducev_genuine_output_compute_correct_general
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

## Public theorem: `softmax_reducev_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` gather-skin headline** — `softmax_reducev` on
`StreamMetaGatherMasked3DKernelIO₂`, the *exemplar* consumer of the
gather-indexed two-stream fold skin, at fully symbolic per-axis strides and
a free signed sentinel `other_kv_index`. For every rounding model `R`, the
faithful surface implements, on its gather-indexed signature, the streamed
closed form `softmaxReducevIOSpec`: every output lane `d` holds
`Σₙ softmax(qk)[n] · v[n, d]` over the `m 0` live tokens, read off the two
pinned streams. The kernel has **zero rounding events** (`.nat` slot loads,
a `.int` page-table gather, `other=`-defaulted `.real` loads, `.real`
in-loop arithmetic, untyped terminal store), so the skin's boundary
quantization degenerates: the readback's `R.round .real` is the identity by
the model's defining `round_real`.

**The gather channel.** `B_Loc` enters as the skin's index channel
(`gty = .int`, `gother = other_kv_index`), and the `V` window `read2` eats
the gathered tile: `(G t jL · stride_vbs).toNat + …`. Because the Python `V`
load is **unmasked** (`mask2 ≡ True`), the triple's `read2` in-bounds
hypothesis covers *dead* lanes too, where the two-leg gather pin forces
`G t jL = other_kv_index` — the honest disclosure that this port dereferences
the sentinel address. No hypothesis on `other_kv_index` is needed *here*
precisely because the skin makes that bound the caller's obligation.

**Launch legality (`pre` = the trusted-launch boundary).** The triple is
guarded by `io.pre = (m 0 ≤ max_input_len ∧ 0 < m 0 ∧ m 0 % BLOCK_N = 0)`:
the first conjunct is the host's `max_input_len` contract (it is what makes
the pid-free budget `T = max_input_len / BLOCK_N` citable for every live
step, in both the safety walk and the value bridge); the last two are the
exact headline's `hseqpos` / `hseqmod` side conditions, **disclosed here as
launch restrictions** (the port's own test shape satisfies them). The `⊨[R]`
triple says nothing about launches outside this boundary.

**Hypothesis provenance**: `0 < BLOCK_DMODEL`, `0 < BLOCK_N` are the exact
headline's side conditions (nonempty tiles; `reduceMax` totality); `hOInj`
restates the exact headline's **open** output-offset injectivity side
condition in ∀-pids form (per-axis strides are symbolic, so no contiguity
discharge is available). The exact headline's free running max `mr` and its
`hM` hypothesis are **not** exported: the spec pins the shift at `0` and
`srWeightedSum_shift_invariant` bridges the two, with `mr` constructed
internally from `srRunningMax_ne_botG` (legal because `0 < m 0` is in
`pre`). The exact headline's `hundef` is not a hypothesis either — the
skin's Hoare triple carries the `undef` pin itself.

Relation to the exact surface: the `Realizes_without_Rounding` headline
above is retained unchanged; this `⊨[R]` face restates the same genuine
closed form on the gather skin, for every `R` at once. -/
```
</details>

**Statement:**
```lean
specification softmax_reducev_io_correctness (R : RoundingModel)
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat)
    (mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int) (hD : 0 < BLOCK_DMODEL) (hN : 0 < BLOCK_N)
    (hOInj : ∀ pid₀ pid₁ : Nat, Function.Injective
      (fun i : Fin BLOCK_DMODEL => pid₀ * sob + pid₁ * soh + i.val * sod)) :
    softmaxReducevIO Logics V Out BLoc BStartLoc BSeqLen mil slh slb svbs svh svd
        sob soh sod sb ss BLOCK_DMODEL BLOCK_N other_kv_index ⊨[R]
      fun _ _ _ m xs ys j =>
        softmaxReducevIOSpec BLOCK_N BLOCK_DMODEL (srIOT mil BLOCK_N)
          (m (⟨0, by omega⟩ : Fin 2)) hN xs ys j
```

**Assumptions / layout contracts:**
- `hD : 0 < BLOCK_DMODEL`
- `hN : 0 < BLOCK_N`
- `hOInj : ∀ pid₀ pid₁ : Nat, Function.Injective
      (fun i : Fin BLOCK_DMODEL => pid₀ * sob + pid₁ * soh + i.val * sod)`

**Closed-form spec defs (transitive):** `softmaxReducevIO`, `softmaxReducevIOSpec`, `srIOT`, `softmax_reducev_surface`, `srIOMetaBuf`, `softmaxReducevWeightedSum`, `srIOqk`, `srIOvv`, `softmaxReducevAcc`, `softmaxReducevDenom`, `softmaxWeight`

<details><summary><code>softmaxReducevIO</code></summary>

```
/-- **Gather-indexed IO signature** of `softmax_reducev` on the
gather-indexed two-stream fold skin (S1: online-softmax fold + terminal
store, 2-D pid grid `(cur_batch, cur_head)`), at fully **symbolic per-axis
strides** and a free signed sentinel `other_kv_index`.

Windows transcribe the kernel's pointer arithmetic VERBATIM, with the
loaded slot vector `m` in place of the in-state metadata reads (`m 0 =
cur_batch_seq_len`, `m 1 = cur_batch_start_loc`):

* `gread` (`B_Loc`, the index channel): lane `jL` of step `t` reads
  `pid₀·stride_b_loc_b + (max_input_len − m 0)·stride_b_loc_s +
  (t·BN + jL)·stride_b_loc_s` — the ℕ-truncated `max_input_len − seq_len`
  transcribed as the kernel spells it (`off_b_loc`); `gmask` is
  `t·BN + jL < m 0` and `gother = other_kv_index`.
* `read1` (`Logics`, the logits): lane `jL` of step `t` reads
  `pid₁·stride_logic_h + (m 1 + (t·BN + jL))·stride_logic_bs`; `mask1` is
  the same live-token guard (the kernel's `other = float("-inf")` only
  feeds masked-off lanes, which the spec never reads).
* `read2` (`V`, the **gather-addressed** value rows, lane
  `j = (jL, d)` row-major over `[BLOCK_N, BLOCK_DMODEL]`) reads
  `(G t jL · stride_vbs).toNat + pid₁·stride_vh + d·stride_vd` — the
  DSL's ℤ-multiply-then-`Int.toNat` transcribed verbatim. **`mask2 ≡ True`**:
  the Python `tl.load(v_ptrs + v_index[:, None] * stride_vbs)` carries no
  mask, so the sentinel-substituted address of a dead lane is a *real*
  access and its in-bounds obligation is part of the triple's hypotheses
  (the honest disclosure of the port's `other_kv_index` behaviour).
* `write` (`Out`, the terminal store after the post-loop divide): lane `i`
  writes `pid₀·stride_obs + pid₁·stride_oh + i·stride_od`, `writeMask ≡ True`
  (the store is unmasked).

`pre` is the launch-legality field: `m 0 ≤ max_input_len ∧ 0 < m 0 ∧
m 0 % BLOCK_N = 0`. The first conjunct is the host's `max_input_len`
contract (it is what makes the pid-free budget `T` citable); the last two
are inherited verbatim from the exact headline's `hseqpos` / `hseqmod` side
conditions — **disclosed launch restrictions**, not proved here.

`outDType` is the `.real` default: the terminal `tl.store` is untyped, so
there is no quantization event. -/
```
```lean
def softmaxReducevIO (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat)
    (mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int) : StreamMetaGatherMasked3DKernelIO₂ where
  kernel := softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
    mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N other_kv_index
  inp1 := Logics
  inp2 := V
  out := Out
  nMeta := 2
  sty := fun _ => ChanTy.nat
  mbuf := srIOMetaBuf BSeqLen BStartLoc
  mwin := fun _ pid₀ _ _ => pid₀
  gbuf := BLoc.cast
  gty := ChanTy.int
  Bg := BLOCK_N
  gother := other_kv_index
  T := srIOT mil BLOCK_N
  B1 := BLOCK_N
  B2 := BLOCK_N * BLOCK_DMODEL
  C := BLOCK_DMODEL
  outDType := .real
  pre := fun _ _ _ m =>
    m (⟨0, by omega⟩ : Fin 2) ≤ mil ∧ 0 < m (⟨0, by omega⟩ : Fin 2)
      ∧ m (⟨0, by omega⟩ : Fin 2) % BLOCK_N = 0
  gread := fun pid₀ _ _ m t jL =>
    pid₀ * sb + (mil - m (⟨0, by omega⟩ : Fin 2)) * ss
      + (t.val * BLOCK_N + jL.val) * ss
  gmask := fun _ _ _ m t jL =>
    t.val * BLOCK_N + jL.val < m (⟨0, by omega⟩ : Fin 2)
  read1 := fun _ pid₁ _ m t jL =>
    pid₁ * slh + (m (⟨1, by omega⟩ : Fin 2) + (t.val * BLOCK_N + jL.val)) * slb
  mask1 := fun _ _ _ m t jL =>
    t.val * BLOCK_N + jL.val < m (⟨0, by omega⟩ : Fin 2)
  read2 := fun _ pid₁ _ _ G t j =>
    (G t (Lane2D.decode j).1 * svbs).toNat + pid₁ * svh
      + (Lane2D.decode j).2.1.val * svd
  mask2 := fun _ _ _ _ _ _ => True
  write := fun pid₀ pid₁ _ _ i => pid₀ * sob + pid₁ * soh + i.val * sod
  writeMask := fun _ _ _ _ _ => True
```
</details>

<details><summary><code>softmaxReducevIOSpec</code></summary>

```
/-- **The streamed closed form**: `softmaxReducevWeightedSum` restated over
the two streamed tiles — `out[d] = Σₙ softmax(qk)[n] · v[n, d]` over the
`S = m 0` live tokens. The running-max argument is pinned at `0`: the
quotient is invariant under the shift (`srWeightedSum_shift_invariant`), so
the spec needs no free `mr` (this is how the exact headline's free-variable
`hM` hypothesis is discharged rather than exported). -/
```
```lean
noncomputable def softmaxReducevIOSpec (BLOCK_N BLOCK_DMODEL T S : Nat)
    (hBN : 0 < BLOCK_N) (xs : Fin T → Fin BLOCK_N → ℝ)
    (ys : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ) (d : Fin BLOCK_DMODEL) : ℝ :=
  softmaxReducevWeightedSum (srIOqk BLOCK_N T S hBN xs) 0
    (srIOvv BLOCK_N BLOCK_DMODEL T S hBN ys) d
```
</details>

<details><summary><code>srIOT</code></summary>

```
/-- The pid-free step budget `T = max_input_len / BLOCK_N`. At a `pre`-legal
launch (`seq_len ≤ max_input_len` and `BLOCK_N ∣ seq_len`) the live trip
count `seq_len` never outruns `T·BLOCK_N` (`srIOT_seq_le`), so floor
division suffices — no ceiling is needed. -/
```
```lean
def srIOT (max_input_len BLOCK_N : Nat) : Nat :=
  max_input_len / BLOCK_N
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

<details><summary><code>srIOMetaBuf</code></summary>

```
/-- Slot-region table of the two per-batch metadata slots, in the kernel's
own load order (`B_Seqlen` then `B_Start_Loc`). A shared def, never an
inline `match` in a window position. -/
```
```lean
def srIOMetaBuf (BSeqLen BStartLoc : Region .nat) : Fin 2 → RegionName
  | ⟨0, _⟩ => BSeqLen.cast
  | ⟨_ + 1, _⟩ => BStartLoc.cast
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

<details><summary><code>srIOqk</code></summary>

```
/-- The logit of live token `n` read off the first stream: step
`n / BLOCK_N`, lane `n % BLOCK_N` (`0` past the step budget — unreachable
at any `pre`-legal launch). -/
```
```lean
noncomputable def srIOqk (BLOCK_N T S : Nat) (hBN : 0 < BLOCK_N)
    (xs : Fin T → Fin BLOCK_N → ℝ) (n : Fin S) : ℝ :=
  if h : n.val / BLOCK_N < T then
    xs ⟨n.val / BLOCK_N, h⟩ ⟨n.val % BLOCK_N, Nat.mod_lt _ hBN⟩
  else 0
```
</details>

<details><summary><code>srIOvv</code></summary>

```
/-- The gathered value row of live token `n`, channel `d`, read off the
second stream at lane `(n % BLOCK_N, d)`. The page indirection lives in the
*window* (`read2` eats `G`), so the stream cell is already the gathered
row. -/
```
```lean
noncomputable def srIOvv (BLOCK_N BLOCK_DMODEL T S : Nat) (hBN : 0 < BLOCK_N)
    (ys : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ) (n : Fin S)
    (d : Fin BLOCK_DMODEL) : ℝ :=
  if h : n.val / BLOCK_N < T then
    ys ⟨n.val / BLOCK_N, h⟩
      (Lane2D.encode (⟨n.val % BLOCK_N, Nat.mod_lt _ hBN⟩, d, PUnit.unit))
  else 0
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

## Also present (pinned special-case summaries)
- `softmax_reducev_final_store_slice_compute_correct`
