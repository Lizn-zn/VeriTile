import VeriTile.Triton

/-!
# `softmax_reducev` — strict per-kernel correctness

`_fwd_kernel` is the fused online-softmax + reduce-over-V attention kernel:
program `(cur_batch, cur_head)` streams the sequence in `BLOCK_N` chunks,
gathers V rows through the `B_Loc` paged-KV index, and maintains the flash-style
running statistics — running max `e_max`, rescaled running sum `e_sum`, and
rescaled accumulator `acc` — then normalizes `acc / e_sum` and stores the
`BLOCK_DMODEL` result to `Out[cur_batch, cur_head, :]`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel[(batch, head)](...)`, the grid over
`(batch, head)`, the host `BLOCK = 64` / `BLOCK_DMODEL = v.shape[-1]` choices,
`num_warps` / `num_stages`, and how the runtime composes per-program writes into
`Out`) is the *trusted boundary*, not a proof obligation here. Because the
program ids `cur_batch`/`cur_head` are universally quantified (via `BlockState`),
the per-program statements cover every program of the grid.

## Proof architecture

```
softmax_reducev_genuine_output_compute_correct_general      ← TOP THEOREM
  └─ sr_execG                                  ← exec-side unfold of the streaming loop
       ├─ sr_attn_stepG                        ← per-block online-softmax step
       └─ srPostLoopG_eval                     ← per-lane normalized readback (final store)
```

The top theorem holds for `other_kv_index : Int` free (any sentinel) and at
symbolic `BLOCK_DMODEL`/`BLOCK_N`, covering both `other_kv_index ∈ {-1, 0}` test
variants as instances.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float, so the `-inf` init,
`exp`, running-max rescaling, and the final division are real-valued);
`num_warps` / `num_stages` are not modeled. The verified
statement is scoped to the **final normalized store** to `Out`: the expected
value is `acc / e_sum` (`softmaxReducevFinalSpec`), and — discharging the loop —
the genuine input-side `softmaxReducevWeightedSum` (`sr_execG`),
read off at each `outOffset`, over a one-block output footprint with the program
ids universally quantified. The streaming online-softmax loop (running max,
rescale-and-accumulate, paged-V gather) feeds those `Acc`/`ESum` values; the side
condition is the offset-injectivity of the `Out` slice (symbolic-`BLOCK_DMODEL`).

## Genuine closed form (input-side, non-self-referential)

`softmaxReducevWeightedSum` states the mathematical result over the *input*
logits and gathered value rows:
`out[d] = Σ_n softmax(qk)[n] · V[v_index[n], d]`
`= (Σ_n exp(qk[n] - M)·V[v_index[n], d]) / (Σ_n exp(qk[n] - M))`
(the genuine softmax, independent of the running max `M`).
`softmaxReducevFinalSpec_eq_weightedSum`
bridges the verified `acc / e_sum` store spec to this closed form, and
`softmax_reducev_final_store_slice_weighted_sum_correct` proves the verified
final-store slice *realizes* it under the loop-output contract (`Acc` / `ESum`
hold the closed-form numerator / denominator — exact over `ℝ`). The `exec`-side
unfold of the dynamic (`forRangeDyn`) online-softmax loop with the paged-V gather
is now discharged sorry-free by the dimension-general `sr_execG`,
so the top theorem `softmax_reducev_genuine_output_compute_correct_general`
realizes the genuine input-side closed form directly.
-/

namespace VeriTile.Bench.TritonBenchG.SoftmaxReducev

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `softmax_reducev_genuine_output_compute_correct_general` -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-- Lean port of `softmax_reducev.py`'s `_fwd_kernel`.

This records the streaming softmax recurrence over token blocks, the signed
`B_Loc` gather with Python's `other_kv_index` sentinel, the V gather, and the
final normalized writeback. -/
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

/-- The full softmax-reduceV surface lowers to the algorithm layer. -/
theorem softmax_reducev_surface_toAlgorithm_supported
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat)
    (max_input_len
      stride_logic_h stride_logic_bs
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_b_loc_b stride_b_loc_s
      BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int) :
    ∃ alg, (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      max_input_len stride_logic_h stride_logic_bs stride_vbs stride_vh
      stride_vd stride_obs stride_oh stride_od stride_b_loc_b stride_b_loc_s
      BLOCK_DMODEL BLOCK_N other_kv_index).toAlgorithm? = Except.ok alg := by
  simp [softmax_reducev_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented final normalization/store slice of `softmax_reducev.py`'s
`_fwd_kernel`.

The full kernel performs a streaming softmax over V blocks, maintaining `acc`
and `e_sum`. This slice starts after that loop with precomputed `Acc` and `ESum`
regions, then proves the final `acc / e_sum` writeback to `Out` across the
`BLOCK_DMODEL` vector. -/
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

def dIndex (_s : BlockState) (i : Fin BLOCK_DMODEL) : Nat :=
  i.val

def accOffset
    (s : BlockState) (stride_acc_bs stride_acc_h stride_acc_d : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_acc_bs + s.pids 1 * stride_acc_h + dIndex s i * stride_acc_d

def eSumOffset (s : BlockState) (stride_es_bs stride_es_h : Nat) : Nat :=
  s.pids 0 * stride_es_bs + s.pids 1 * stride_es_h

def outOffset
    (s : BlockState) (stride_obs stride_oh stride_od : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_obs + s.pids 1 * stride_oh + dIndex s i * stride_od

noncomputable def softmaxReducevFinalSpec
    (s : BlockState) (Acc ESum : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d stride_es_bs stride_es_h : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  s.readMem Acc (accOffset s stride_acc_bs stride_acc_h stride_acc_d i) /
    s.readMem ESum (eSumOffset s stride_es_bs stride_es_h)

/-! ## Genuine closed form: softmax-weighted reduction over V

`softmax_reducev.py` computes, for each output lane `d`, the softmax-weighted sum
of the gathered V rows:

```
out[d] = ( Σ_n exp(qk[n] - M) · V[v_index[n], d] ) / ( Σ_n exp(qk[n] - M) )
       = Σ_n softmax(qk)[n] · V[v_index[n], d]
```

where `n` ranges over the `cur_batch_seq_len` valid tokens, `qk[n]` are the
attention logits, `M = maxₙ qk[n]` is the running max maintained by the online
softmax, and `v_index[n]` is the paged-KV gather index from `B_Loc`. Because the
online-softmax recurrence (running max, rescale-and-accumulate) is numerically
exact over `ℝ`, the streamed `acc` / `e_sum` equal the batched numerator /
denominator above, independent of the block schedule. This is a *genuine*
input-side closed form: it names only the logits `qk` and the gathered value rows
`v`, never the kernel's own executed output.

These definitions are the mathematical content that the proof-region values `Acc`
and `ESum` (consumed by the verified final-store slice) are contracted to hold. -/

/-- Unnormalized softmax weight `exp(qk[n] - M)` for token `n`, with running max
`M = mMax`. -/
noncomputable def softmaxWeight {S : Nat} (qk : Fin S → ℝ) (mMax : ℝ) (n : Fin S) : ℝ :=
  Real.exp (qk n - mMax)

/-- The softmax normalizer `e_sum = Σ_n exp(qk[n] - M)` — the genuine closed form
of the streamed `ESum` value. -/
noncomputable def softmaxReducevDenom {S : Nat} (qk : Fin S → ℝ) (mMax : ℝ) : ℝ :=
  ∑ n : Fin S, softmaxWeight qk mMax n

/-- The unnormalized weighted V reduction
`acc[d] = Σ_n exp(qk[n] - M)·V[v_index[n], d]` — the genuine closed form of the
streamed `Acc[d]` value: the gathered value rows weighted by the unnormalized
softmax probabilities. -/
noncomputable def softmaxReducevAcc {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (mMax : ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ)
    (d : Fin BLOCK_DMODEL) : ℝ :=
  ∑ n : Fin S, softmaxWeight qk mMax n * v n d

/-- The full normalized closed form
`out[d] = acc[d] / e_sum = Σ_n softmax(qk)[n] · V[v_index[n], d]`. This is what
`softmax_reducev.py` stores to `Out[cur_batch, cur_head, d]`, stated purely over
the input logits `qk` and gathered value rows `v` — no reference to the executed
kernel. -/
noncomputable def softmaxReducevWeightedSum {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (mMax : ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ)
    (d : Fin BLOCK_DMODEL) : ℝ :=
  softmaxReducevAcc qk mMax v d / softmaxReducevDenom qk mMax

/-- **The normalized weighted sum is `acc / e_sum`.** Bridges the genuine
input-side closed form `softmaxReducevWeightedSum` to the verified final-store
spec `softmaxReducevFinalSpec`: whenever the proof regions `Acc` / `ESum` hold the
closed-form numerator / denominator at lane `d` (the contract the streaming
online-softmax loop establishes, exact over `ℝ`), the stored value equals the
softmax-weighted V reduction over the input. -/
theorem softmaxReducevFinalSpec_eq_weightedSum {S BLOCK_DMODEL : Nat}
    (s : BlockState) (Acc ESum : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d stride_es_bs stride_es_h : Nat)
    (qk : Fin S → ℝ) (mMax : ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ)
    (i : Fin BLOCK_DMODEL)
    (hAcc : s.readMem Acc (accOffset s stride_acc_bs stride_acc_h stride_acc_d i)
      = softmaxReducevAcc qk mMax v i)
    (hESum : s.readMem ESum (eSumOffset s stride_es_bs stride_es_h)
      = softmaxReducevDenom qk mMax) :
    softmaxReducevFinalSpec s Acc ESum stride_acc_bs stride_acc_h stride_acc_d
        stride_es_bs stride_es_h i
      = softmaxReducevWeightedSum qk mMax v i := by
  unfold softmaxReducevFinalSpec softmaxReducevWeightedSum
  rw [hAcc, hESum]

/-- Algorithm-layer correctness for the final softmax-reduce-v store slice. -/
theorem softmax_reducev_final_store_slice_correct
    (Acc ESum Out : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d
      stride_es_bs stride_es_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ∀ i : Fin BLOCK_DMODEL,
      let outAddr := outOffset s stride_obs stride_oh stride_od i
      (exec (softmax_reducev_final_store_slice Acc ESum Out
            stride_acc_bs stride_acc_h stride_acc_d stride_es_bs stride_es_h
            stride_obs stride_oh stride_od BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (softmaxReducevFinalSpec s Acc ESum stride_acc_bs stride_acc_h
          stride_acc_d stride_es_bs stride_es_h i) := by
  intro i
  simp [exec, softmax_reducev_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, NumericDType.div, dIndex, accOffset, eSumOffset,
        outOffset]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_DMODEL] =>
        s.pids 0 * stride_obs + s.pids 1 * stride_oh + idx.1.val * stride_od) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s stride_obs stride_oh stride_od a =
        outOffset s stride_obs stride_oh stride_od b := by
      simpa [outOffset, dIndex] using hab
    have hfin : a = b := hOutInj habFin
    subst hfin
    rfl
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [softmaxReducevFinalSpec, accOffset, eSumOffset, outOffset, dIndex]

/-- Compute-facing correctness for the final softmax-reduce-v store slice. -/
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
          stride_acc_d stride_es_bs stride_es_h i) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_reducev_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := softmax_reducev_final_store_slice_correct Acc ESum Out
    stride_acc_bs stride_acc_h stride_acc_d stride_es_bs stride_es_h
    stride_obs stride_oh stride_od BLOCK_DMODEL s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-- **Genuine closed-form compute correctness for the final softmax-reduce-v
store.** The expected output is the input-side softmax-weighted V reduction
`softmaxReducevWeightedSum` — `Σ_n softmax(qk)[n] · V[v_index[n], d]` — NOT the
self-referential executed value. The hypotheses `hAcc` / `hESum` are the contract
the streaming online-softmax loop establishes: that the proof regions `Acc` /
`ESum` hold the closed-form numerator / denominator (exact over `ℝ`). Composes the
verified store slice with `softmaxReducevFinalSpec_eq_weightedSum`. -/
theorem softmax_reducev_final_store_slice_weighted_sum_correct {S : Nat}
    (Acc ESum Out : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d
      stride_es_bs stride_es_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (qk : Fin S → ℝ) (mMax : ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i))
    (hAcc : ∀ i : Fin BLOCK_DMODEL,
      s.readMem Acc (accOffset s stride_acc_bs stride_acc_h stride_acc_d i)
        = softmaxReducevAcc qk mMax v i)
    (hESum : s.readMem ESum (eSumOffset s stride_es_bs stride_es_h)
      = softmaxReducevDenom qk mMax) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := softmax_reducev_final_store_slice Acc ESum Out
        stride_acc_bs stride_acc_h stride_acc_d stride_es_bs stride_es_h
        stride_obs stride_oh stride_od BLOCK_DMODEL)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i => softmaxReducevWeightedSum qk mMax v i) := by
  have hbase := softmax_reducev_final_store_slice_compute_correct Acc ESum Out
    stride_acc_bs stride_acc_h stride_acc_d stride_es_bs stride_es_h
    stride_obs stride_oh stride_od BLOCK_DMODEL s hOutInj
  have hExpEq : (fun i : Fin BLOCK_DMODEL =>
        softmaxReducevFinalSpec s Acc ESum stride_acc_bs stride_acc_h
          stride_acc_d stride_es_bs stride_es_h i)
      = fun i => softmaxReducevWeightedSum qk mMax v i := by
    funext i
    exact softmaxReducevFinalSpec_eq_weightedSum s Acc ESum stride_acc_bs
      stride_acc_h stride_acc_d stride_es_bs stride_es_h qk mMax v i
      (hAcc i) hESum
  rw [hExpEq] at hbase
  exact hbase

/-! ## ⊥-seeded online-softmax recurrence (faithful kernel register model)

These definitions mirror the FlashAttention exec-assembly's ⊥-seed pattern
(`osStepBot` / `flashStateBot` / `flashRunningMax`), retargeted to
`softmax_reducev`'s base-`e` online softmax (`tl.exp`, not `exp2`) and its paged-V
gather. They are the faithful model of the kernel's three live registers across
the dynamic `forRangeDyn` loop:

* `e_max` — seeded `float("-inf")` (= `⊥` in `WithBot ℝ`), the running maximum;
* `e_sum` — seeded `0.0`, the rescaled running denominator;
* `acc[d]` — seeded `0.0`, the rescaled running V-weighted accumulator.

For a *fixed* output channel `d`, the per-key data the loop streams is the pair
`(qk[n], v[n][d])`, so the running state lives in `WithBot ℝ × ℝ × ℝ` exactly as
in flash. The key list `srKeysUpto` over the streamed window `[0, hi)` is the
prefix of valid tokens; at the full window `hi = S` it is the full token list,
where the ⊥-seeded fold reads off the genuine closed form
`softmaxReducevAcc / softmaxReducevDenom`. -/

/-- One ⊥-seeded online-softmax step for `softmax_reducev` (base-`e`). The running
max lives in `WithBot ℝ` (seeded `⊥`), so the rescale factor
`old_scale = exp(m ⊖ m')` is `0` on the first block — faithful to the kernel's
`e_max` register (`float("-inf")`) and `e_sum`/`acc` (seeded `0`). `sv = (qk, v)`
is one streamed token's logit and gathered V-row entry. -/
noncomputable def srOsStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) :
    WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let q := sv.1; let v := sv.2
  let m' := m ⊔ ((q : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp (WithBot.realSub m m')).unbotD 0
  let p := Real.exp (q - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)

/-- Per-channel streamed key list over the window `[0, hi)`: the valid tokens
`n < hi`, in index order, each carrying `(qk[n], v[n][d])`. After `c` blocks
`hi = c · BLOCK_N`, this is the prefix the kernel has streamed for channel `d`. -/
noncomputable def srKeysUpto {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (hi : Nat)
    (d : Fin BLOCK_DMODEL) : List (ℝ × ℝ) :=
  (List.finRange S).filterMap (fun n : Fin S =>
    if n.val < hi then some (qk n, v n d) else none)

/-- `srStateBot` — the ⊥-seeded running `(max, e_sum, acc[d])` after streaming the
window `[0, hi)` for channel `d`. Faithful to the kernel's register recurrence
(`e_max` seeded `⊥`, `e_sum`/`acc` seeded `0`). -/
noncomputable def srStateBot {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (hi : Nat)
    (d : Fin BLOCK_DMODEL) : WithBot ℝ × ℝ × ℝ :=
  (srKeysUpto qk v hi d).foldl srOsStepBot (⊥, 0, 0)

/-- **⊥-seeded running max** of the streamed key prefix `[0, hi)`, exactly the
value the kernel carries in its `e_max` register (`float("-inf")` seeds at `⊥`).
The `WithBot` `⊔`-fold of the coerced per-key logits; `⊥` on the empty / `hi = 0`
window (the kernel's preLoop init). -/
noncomputable def srRunningMax {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (hi : Nat)
    (d : Fin BLOCK_DMODEL) : WithBot ℝ :=
  ((srKeysUpto qk v hi d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥

/-- Block-`c` per-channel key list: valid tokens with `c·BLOCK_N ≤ n < (c+1)·BLOCK_N`
— the tokens the loop's `c`-th iteration streams. -/
noncomputable def srBlock {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (BLOCK_N c : Nat)
    (d : Fin BLOCK_DMODEL) : List (ℝ × ℝ) :=
  (List.finRange S).filterMap (fun n : Fin S =>
    if c * BLOCK_N ≤ n.val ∧ n.val < (c + 1) * BLOCK_N then some (qk n, v n d) else none)

/-! ### ⊥-seed projection + consistency lemmas (base-`e`)

Mirror of FlashAttention's `flashStateBot_fst` / `osStepBot_foldl_consistent` /
`flashStateBot_snd_*`, with `exp2`/`pow2` replaced by base-`e` `Real.exp`. The
rescale factor is `κ(m) = m.elim 0 (fun r => Real.exp (-r))` (`κ ⊥ = 0`,
`κ (some r) = exp(-r)`); the per-key weight is `Real.exp (qk n)`. -/

/-- The running `max` component of an `srOsStepBot` fold is the `⊔`-fold of the
coerced logits. -/
theorem srOsStepBot_fst (xs : List (ℝ × ℝ)) (m₀ : WithBot ℝ) (l₀ acc₀ : ℝ) :
    (xs.foldl srOsStepBot (m₀, l₀, acc₀)).1
      = (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldl (· ⊔ ·) m₀ := by
  induction xs generalizing m₀ l₀ acc₀ with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl_cons, List.map_cons]; rw [ih]; rfl

/-- The `WithBot ⊔`-fold is seed/direction-agnostic. -/
theorem sr_foldl_sup_bot_eq_foldr (L : List (WithBot ℝ)) :
    L.foldl (· ⊔ ·) (⊥ : WithBot ℝ) = L.foldr (· ⊔ ·) (⊥ : WithBot ℝ) := by
  have gen : ∀ (m : WithBot ℝ), L.foldl (· ⊔ ·) m = m ⊔ L.foldr (· ⊔ ·) ⊥ := by
    induction L with
    | nil => intro m; simp
    | cons a t ih =>
      intro m; simp only [List.foldl_cons, List.foldr_cons, ih]; rw [max_assoc]
  rw [gen ⊥, bot_sup_eq]

/-- **⊥-seeded consistency** (base-`e`). Folding `srOsStepBot` from a start
`(m, l, acc)` anchored to the max-free batch denominator `L` / accumulator `T`
via `κ(m) = m.elim 0 (fun r => exp(-r))` (with `κ ⊥ = 0`) keeps that invariant:
the final `l`/`acc` are `κ(final max)·(L + Σ exp(qk))` and
`κ(final max)·(T + Σ exp(qk)·v)`. -/
theorem srOsStepBot_foldl_consistent (xs : List (ℝ × ℝ)) (m : WithBot ℝ)
    (l acc T L : ℝ)
    (hl : l = (m.elim 0 (fun r => Real.exp (-r))) * L)
    (hacc : acc = (m.elim 0 (fun r => Real.exp (-r))) * T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0) :
    let st := xs.foldl srOsStepBot (m, l, acc)
    st.2.1 = (st.1.elim 0 (fun r => Real.exp (-r))) * (L + (xs.map (fun p => Real.exp p.1)).sum) ∧
    st.2.2 = (st.1.elim 0 (fun r => Real.exp (-r))) * (T + (xs.map (fun p => Real.exp p.1 * p.2)).sum) := by
  induction xs generalizing m l acc T L with
  | nil => simp [hl, hacc]
  | cons x xs ih =>
    obtain ⟨s, v⟩ := x
    set m' : WithBot ℝ := m ⊔ ((s : ℝ) : WithBot ℝ) with hm'
    have hm'r : ∃ r : ℝ, m' = (r : WithBot ℝ) := by
      cases m with
      | bot => exact ⟨s, by rw [hm']; rfl⟩
      | coe a => exact ⟨max a s, by rw [hm']; rw [← WithBot.coe_max]⟩
    obtain ⟨mr, hmr⟩ := hm'r
    have hκm' : m'.elim 0 (fun r => Real.exp (-r)) = Real.exp (-mr) := by rw [hmr]; rfl
    have hunbot : m'.unbotD 0 = mr := by rw [hmr]; rfl
    have hp : Real.exp (s - m'.unbotD 0) = Real.exp (-mr) * Real.exp s := by
      rw [hunbot, ← Real.exp_add]; ring_nf
    have hl' : l * (WithBot.realExp (WithBot.realSub m m')).unbotD 0
        + Real.exp (s - m'.unbotD 0) = Real.exp (-mr) * (L + Real.exp s) := by
      cases m with
      | bot =>
        rw [hmL rfl]
        have hz : (WithBot.realExp (WithBot.realSub (⊥ : WithBot ℝ) m')).unbotD 0 = 0 := by
          rw [WithBot.realSub_bot_left, WithBot.realExp_bot]; rfl
        rw [hz, mul_zero, zero_add, hp]; ring
      | coe a =>
        have hm'a : m' = ((max a s : ℝ) : WithBot ℝ) := by rw [hm']; rw [← WithBot.coe_max]
        have hmra : mr = max a s := by rw [hm'a] at hmr; exact (WithBot.coe_inj.mp hmr.symm)
        have hαa : (Real.exp (-a)) * (WithBot.realExp (WithBot.realSub (↑a) m')).unbotD 0
            = Real.exp (-mr) := by
          rw [hm'a, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe]
          rw [← Real.exp_add, hmra]; ring_nf
        rw [hl, show ((↑a : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = Real.exp (-a) from rfl]
        rw [mul_right_comm, hαa, hp]; ring
    have hacc' : acc * (WithBot.realExp (WithBot.realSub m m')).unbotD 0
        + Real.exp (s - m'.unbotD 0) * v = Real.exp (-mr) * (T + Real.exp s * v) := by
      cases m with
      | bot =>
        rw [hmT rfl]
        have hz : (WithBot.realExp (WithBot.realSub (⊥ : WithBot ℝ) m')).unbotD 0 = 0 := by
          rw [WithBot.realSub_bot_left, WithBot.realExp_bot]; rfl
        rw [hz, mul_zero, zero_add, hp]; ring
      | coe a =>
        have hm'a : m' = ((max a s : ℝ) : WithBot ℝ) := by rw [hm']; rw [← WithBot.coe_max]
        have hmra : mr = max a s := by rw [hm'a] at hmr; exact (WithBot.coe_inj.mp hmr.symm)
        have hαa : (Real.exp (-a)) * (WithBot.realExp (WithBot.realSub (↑a) m')).unbotD 0
            = Real.exp (-mr) := by
          rw [hm'a, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe]
          rw [← Real.exp_add, hmra]; ring_nf
        rw [hacc, show ((↑a : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = Real.exp (-a) from rfl]
        rw [mul_right_comm, hαa, hp]; ring
    have step := ih m'
      (l * (WithBot.realExp (WithBot.realSub m m')).unbotD 0 + Real.exp (s - m'.unbotD 0))
      (acc * (WithBot.realExp (WithBot.realSub m m')).unbotD 0 + Real.exp (s - m'.unbotD 0) * v)
      (T + Real.exp s * v) (L + Real.exp s) (by rw [hl', hκm']) (by rw [hacc', hκm'])
      (by rw [hmr]; simp) (by rw [hmr]; simp)
    simpa [List.foldl_cons, srOsStepBot, hm', List.map_cons, add_assoc] using step

/-- The ⊥-seeded running `max` of `srStateBot` is exactly `srRunningMax`. -/
theorem srStateBot_fst_eq_runningMax {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (hi : Nat)
    (d : Fin BLOCK_DMODEL) :
    (srStateBot qk v hi d).1 = srRunningMax qk v hi d := by
  rw [srStateBot, srOsStepBot_fst, srRunningMax, sr_foldl_sup_bot_eq_foldr]

/-- The ⊥-seeded denominator equals `κ(srRunningMax)·Σ exp(qk)`. -/
theorem srStateBot_snd_fst {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (hi : Nat)
    (d : Fin BLOCK_DMODEL) :
    (srStateBot qk v hi d).2.1
      = ((srRunningMax qk v hi d).elim 0 (fun r => Real.exp (-r)))
        * ((srKeysUpto qk v hi d).map (fun p => Real.exp p.1)).sum := by
  have h := (srOsStepBot_foldl_consistent (srKeysUpto qk v hi d)
    ⊥ 0 0 0 0 (by simp) (by simp) (fun _ => rfl) (fun _ => rfl)).1
  rw [srStateBot]
  rw [show (List.foldl srOsStepBot (⊥, 0, 0) (srKeysUpto qk v hi d)).2.1 = _ from h]
  rw [show (List.foldl srOsStepBot (⊥, 0, 0) (srKeysUpto qk v hi d)).1
        = srRunningMax qk v hi d from by
    rw [srOsStepBot_fst, srRunningMax, sr_foldl_sup_bot_eq_foldr]]
  rw [zero_add]

/-- The ⊥-seeded accumulator equals `κ(srRunningMax)·Σ exp(qk)·v`. -/
theorem srStateBot_snd_snd {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (hi : Nat)
    (d : Fin BLOCK_DMODEL) :
    (srStateBot qk v hi d).2.2
      = ((srRunningMax qk v hi d).elim 0 (fun r => Real.exp (-r)))
        * ((srKeysUpto qk v hi d).map (fun p => Real.exp p.1 * p.2)).sum := by
  have h := (srOsStepBot_foldl_consistent (srKeysUpto qk v hi d)
    ⊥ 0 0 0 0 (by simp) (by simp) (fun _ => rfl) (fun _ => rfl)).2
  rw [srStateBot]
  rw [show (List.foldl srOsStepBot (⊥, 0, 0) (srKeysUpto qk v hi d)).2.2 = _ from h]
  rw [show (List.foldl srOsStepBot (⊥, 0, 0) (srKeysUpto qk v hi d)).1
        = srRunningMax qk v hi d from by
    rw [srOsStepBot_fst, srRunningMax, sr_foldl_sup_bot_eq_foldr]]
  rw [zero_add]

/-- At the full window `hi = S`, the streamed key list is every token in index
order, carrying `(qk n, v n d)`. -/
theorem srKeysUpto_full {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (d : Fin BLOCK_DMODEL) :
    srKeysUpto qk v S d = (List.finRange S).map (fun n : Fin S => (qk n, v n d)) := by
  unfold srKeysUpto
  rw [← List.filterMap_eq_map]
  apply List.filterMap_congr
  intro n _
  simp [n.isLt]

/-- The sum of per-key weights over the full window is the closed-form numerator
of the denominator with running-max shift `M`: `Σ exp(qk) = exp(M)·Σ exp(qk - M)`,
so `κ(M)·Σ exp(qk) = softmaxReducevDenom qk M`. -/
theorem srKeys_full_denom {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (d : Fin BLOCK_DMODEL) (M : ℝ) :
    Real.exp (-M) * ((srKeysUpto qk v S d).map (fun p => Real.exp p.1)).sum
      = softmaxReducevDenom qk M := by
  rw [srKeysUpto_full]
  unfold softmaxReducevDenom softmaxWeight
  rw [List.map_map,
    show ((List.finRange S).map ((fun p : ℝ × ℝ => Real.exp p.1) ∘ fun n => (qk n, v n d))).sum
        = ∑ n : Fin S, Real.exp (qk n) from by
      rw [← List.sum_ofFn (f := fun n : Fin S => Real.exp (qk n)), List.ofFn_eq_map]; rfl]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro n _
  rw [← Real.exp_add]; ring_nf

/-- The sum of per-key weighted V over the full window: `κ(M)·Σ exp(qk)·v =
softmaxReducevAcc qk M v d`. -/
theorem srKeys_full_acc {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (d : Fin BLOCK_DMODEL) (M : ℝ) :
    Real.exp (-M) * ((srKeysUpto qk v S d).map (fun p => Real.exp p.1 * p.2)).sum
      = softmaxReducevAcc qk M v d := by
  rw [srKeysUpto_full]
  unfold softmaxReducevAcc softmaxWeight
  rw [List.map_map,
    show ((List.finRange S).map ((fun p : ℝ × ℝ => Real.exp p.1 * p.2) ∘ fun n => (qk n, v n d))).sum
        = ∑ n : Fin S, Real.exp (qk n) * v n d from by
      rw [← List.sum_ofFn (f := fun n : Fin S => Real.exp (qk n) * v n d), List.ofFn_eq_map]; rfl]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro n _
  rw [← mul_assoc, ← Real.exp_add]; ring_nf

/-- **Full-window ⊥-seeded accumulator equals the closed-form numerator.** With
running max `M = srRunningMax.unbotD 0`, the streamed `acc[d]` is exactly
`softmaxReducevAcc qk M v d`. -/
theorem srStateBot_full_acc {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (d : Fin BLOCK_DMODEL)
    (mr : ℝ) (hM : srRunningMax qk v S d = (mr : WithBot ℝ)) :
    (srStateBot qk v S d).2.2 = softmaxReducevAcc qk mr v d := by
  rw [srStateBot_snd_snd, hM]
  rw [show ((↑mr : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = Real.exp (-mr) from rfl]
  exact srKeys_full_acc qk v d mr

/-- **Full-window ⊥-seeded denominator equals the closed-form denominator.** -/
theorem srStateBot_full_denom {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (d : Fin BLOCK_DMODEL)
    (mr : ℝ) (hM : srRunningMax qk v S d = (mr : WithBot ℝ)) :
    (srStateBot qk v S d).2.1 = softmaxReducevDenom qk mr := by
  rw [srStateBot_snd_fst, hM]
  rw [show ((↑mr : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = Real.exp (-mr) from rfl]
  exact srKeys_full_denom qk v d mr

/-- **The full-window ⊥-seeded final state reads off the genuine closed form.**
`srStateBot.acc / srStateBot.e_sum = softmaxReducevWeightedSum`, with the running
max `M = srRunningMax.unbotD 0`. -/
theorem srStateBot_full_eq_weightedSum {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (d : Fin BLOCK_DMODEL)
    (mr : ℝ) (hM : srRunningMax qk v S d = (mr : WithBot ℝ)) :
    (let st := srStateBot qk v S d; st.2.2 / st.2.1)
      = softmaxReducevWeightedSum qk mr v d := by
  simp only
  rw [srStateBot_full_acc qk v d mr hM, srStateBot_full_denom qk v d mr hM]
  rfl

/-! ### Loop-advance lemmas (base case + one-block window split)

These are the bridges the dynamic-loop invariant uses: the ⊥-seeded state at the
empty window is the kernel's preLoop init `(⊥, 0, 0)`, and streaming one more
`BLOCK_N`-block advances the window `[0, c·BLOCK_N)` to `[0, (c+1)·BLOCK_N)` by
`foldl`-appending that block's keys (the per-iteration `osBlockStep`). -/

/-- The ⊥-seeded running max at the empty window is `⊥` (the kernel's `e_max`
init, `float("-inf")`). -/
theorem srRunningMax_zero {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (d : Fin BLOCK_DMODEL) :
    srRunningMax qk v 0 d = ⊥ := by
  unfold srRunningMax srKeysUpto
  rw [show (List.finRange S).filterMap
        (fun n : Fin S => if n.val < 0 then some (qk n, v n d) else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro n _; simp]
  rfl

/-- The ⊥-seeded state at the empty window is `(⊥, 0, 0)` — the kernel's preLoop
init (`e_max = -inf`, `e_sum = 0`, `acc = 0`). -/
theorem srStateBot_zero {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (d : Fin BLOCK_DMODEL) :
    srStateBot qk v 0 d = (⊥, 0, 0) := by
  unfold srStateBot srKeysUpto
  rw [show (List.finRange S).filterMap
        (fun n : Fin S => if n.val < 0 then some (qk n, v n d) else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro n _; simp]
  rfl

/-- Threshold-split for a `.val`-ascending `Fin` list: the `n.val < hi₂` window
filterMap splits into the `n.val < t` prefix and the `t ≤ n.val < hi₂` block,
provided `t ≤ hi₂`. -/
private theorem sr_filterMap_window_split {n : Nat} (l : List (Fin n))
    (hsorted : l.Pairwise (fun a b => a.val < b.val))
    (t hi₂ : Nat) (g : Fin n → ℝ × ℝ) (hle : t ≤ hi₂) :
    l.filterMap (fun j => if j.val < hi₂ then some (g j) else none)
      = l.filterMap (fun j => if j.val < t then some (g j) else none)
        ++ l.filterMap (fun j => if t ≤ j.val ∧ j.val < hi₂ then some (g j) else none) := by
  induction l with
  | nil => simp
  | cons a tl ih =>
    have htl : tl.Pairwise (fun x y => x.val < y.val) := (List.pairwise_cons.mp hsorted).2
    have hahead : ∀ b ∈ tl, a.val < b.val := (List.pairwise_cons.mp hsorted).1
    rw [List.filterMap_cons, List.filterMap_cons, List.filterMap_cons]
    by_cases hlt : a.val < t
    · rw [ih htl]
      have hnb : ¬ (t ≤ a.val ∧ a.val < hi₂) := fun h => (Nat.not_le.mpr hlt) h.1
      rw [if_neg hnb]
      have h2 : a.val < hi₂ := lt_of_lt_of_le hlt hle
      rw [if_pos h2, if_pos hlt]; rfl
    · have hge : t ≤ a.val := Nat.not_lt.mp hlt
      have htail_prefix : tl.filterMap (fun j => if j.val < t then some (g j) else none) = [] := by
        apply List.filterMap_eq_nil_iff.mpr
        intro b hb
        have hab : a.val < b.val := hahead b hb
        have hbt : ¬ (b.val < t) := by omega
        simp [hbt]
      rw [ih htl, htail_prefix]
      rw [if_neg hlt]
      by_cases h2 : a.val < hi₂
      · rw [if_pos h2, if_pos (And.intro hge h2 : t ≤ a.val ∧ a.val < hi₂)]; rfl
      · rw [if_neg h2, if_neg (fun h : t ≤ a.val ∧ a.val < hi₂ => h2 h.2)]

/-- **Window split** (`hi = c·BLOCK_N`): the keys streamed through `c+1` blocks are
those through `c` blocks followed by block `c`. -/
theorem srKeysUpto_succ {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (BLOCK_N c : Nat)
    (d : Fin BLOCK_DMODEL) :
    srKeysUpto qk v ((c + 1) * BLOCK_N) d
      = srKeysUpto qk v (c * BLOCK_N) d ++ srBlock qk v BLOCK_N c d := by
  unfold srKeysUpto srBlock
  rw [sr_filterMap_window_split (List.finRange S) (List.pairwise_lt_finRange S)
    (c * BLOCK_N) ((c + 1) * BLOCK_N) (fun n => (qk n, v n d))
    (by nlinarith [Nat.zero_le BLOCK_N])]

/-- **One-block invariant advance** (pure math): `srStateBot` after `c+1` blocks
is the per-block update (`srOsStepBot`-fold of block `c`'s keys) applied to
`srStateBot` after `c` blocks. -/
theorem srStateBot_succ {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (BLOCK_N c : Nat)
    (d : Fin BLOCK_DMODEL) :
    srStateBot qk v ((c + 1) * BLOCK_N) d
      = (srBlock qk v BLOCK_N c d).foldl srOsStepBot (srStateBot qk v (c * BLOCK_N) d) := by
  unfold srStateBot
  rw [srKeysUpto_succ, List.foldl_append]

/-- Factor the running-max shift out of a block sum: `Σ exp(s-m')·g = exp(-m')·Σ exp(s)·g`. -/
theorem sr_sum_map_exp_sub (m' : ℝ) (block : List (ℝ × ℝ)) (g : ℝ × ℝ → ℝ) :
    (block.map (fun p => Real.exp (p.1 - m') * g p)).sum
      = Real.exp (-m') * (block.map (fun p => Real.exp p.1 * g p)).sum := by
  induction block with
  | nil => simp
  | cons a t ih =>
    have e : Real.exp (a.1 - m') = Real.exp (-m') * Real.exp a.1 := by
      rw [← Real.exp_add]; ring_nf
    simp only [List.map_cons, List.sum_cons, ih, e]; ring

/-- The `.1` (running max) of an `srOsStepBot` block-fold from `(m, l, acc)`. -/
theorem srOsStepBot_block_fst (m : WithBot ℝ) (l acc : ℝ) (block : List (ℝ × ℝ)) :
    (block.foldl srOsStepBot (m, l, acc)).1
      = m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  rw [srOsStepBot_fst]
  induction block generalizing m with
  | nil => simp
  | cons a t ih =>
    simp only [List.map_cons, List.foldl_cons, List.foldr_cons]
    rw [ih]
    rw [show (m ⊔ ((a.1 : ℝ) : WithBot ℝ)) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) t))
          = m ⊔ (((a.1 : ℝ) : WithBot ℝ) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) t))) from by
      rw [sup_assoc]]

/-- **The block-at-once update equals the key-by-key `srOsStepBot` fold** (base-`e`).
For a block with max `M' = m ⊔ blockSup` and a state `(m, l, acc)` anchored to the
true denominator/accumulator via `l = κ(m)·L`, `acc = κ(m)·T`, the kernel's one-shot
rescale-and-add (`l·α + Σ exp(s−M')`, `acc·α + Σ exp(s−M')·v`, with
`α = realExp(m ⊖ M')`) lands on `block.foldl srOsStepBot (m, l, acc)`. This is the
loop-body bridge relating the kernel's per-iteration register update to the
`srStateBot_succ` fold. -/
theorem srOsStepBot_block_eq (m : WithBot ℝ) (l acc T L : ℝ) (block : List (ℝ × ℝ))
    (hl : l = (m.elim 0 (fun r => Real.exp (-r))) * L)
    (hacc : acc = (m.elim 0 (fun r => Real.exp (-r))) * T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0) :
    let M' := m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
    (M',
     l * (WithBot.realExp (WithBot.realSub m M')).unbotD 0
       + (block.map (fun p => Real.exp (p.1 - M'.unbotD 0))).sum,
     acc * (WithBot.realExp (WithBot.realSub m M')).unbotD 0
       + (block.map (fun p => Real.exp (p.1 - M'.unbotD 0) * p.2)).sum)
      = block.foldl srOsStepBot (m, l, acc) := by
  intro M'
  have hfst : (block.foldl srOsStepBot (m, l, acc)).1 = M' := by
    rw [srOsStepBot_block_fst]
  obtain ⟨hfold_l, hfold_acc⟩ := srOsStepBot_foldl_consistent block m l acc T L hl hacc hmL hmT
  rw [hfst] at hfold_l hfold_acc
  have hM'eq : M' = m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := rfl
  cases hM' : M' with
  | bot =>
    have hempty : block = [] := by
      rcases block with _ | ⟨a, t⟩
      · rfl
      · exfalso
        have : ((a.1 : ℝ) : WithBot ℝ) ≤ M' := by
          rw [hM'eq]
          exact le_sup_of_le_right (by simp only [List.map_cons, List.foldr_cons]; exact le_sup_left)
        rw [hM'] at this
        exact absurd (le_bot_iff.mp this) (WithBot.coe_ne_bot)
    have hm0 : m = ⊥ := by
      rw [hM'eq, hempty] at hM'
      simpa only [List.map_nil, List.foldr_nil, sup_bot_eq] using hM'
    have hl0 : l = 0 := by rw [hl, hm0]; simp [hmL hm0]
    have hacc0 : acc = 0 := by rw [hacc, hm0]; simp [hmT hm0]
    subst hempty
    rw [hl0, hacc0]
    simp only [List.foldl_nil, List.map_nil, List.sum_nil, add_zero, mul_zero, zero_mul]
    rw [hm0]
  | coe Mr =>
    rw [hM'] at hfst hfold_l hfold_acc
    have hlα : l * (WithBot.realExp (WithBot.realSub m (↑Mr : WithBot ℝ))).unbotD 0 = Real.exp (-Mr) * L := by
      cases hm : m with
      | bot =>
        rw [hl, hm, show ((⊥ : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = 0 from rfl,
          zero_mul, hmL hm]; ring
      | coe a =>
        rw [hl, hm, show ((↑a : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = Real.exp (-a) from rfl]
        rw [WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe]
        rw [mul_right_comm, ← Real.exp_add]; ring_nf
    have haccα : acc * (WithBot.realExp (WithBot.realSub m (↑Mr : WithBot ℝ))).unbotD 0 = Real.exp (-Mr) * T := by
      cases hm : m with
      | bot =>
        rw [hacc, hm, show ((⊥ : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = 0 from rfl,
          zero_mul, hmT hm]; ring
      | coe a =>
        rw [hacc, hm, show ((↑a : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = Real.exp (-a) from rfl]
        rw [WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe]
        rw [mul_right_comm, ← Real.exp_add]; ring_nf
    have hsumL : (block.map (fun p => Real.exp (p.1 - (↑Mr : WithBot ℝ).unbotD 0))).sum
        = Real.exp (-Mr) * (block.map (fun p => Real.exp p.1)).sum := by
      have := sr_sum_map_exp_sub ((↑Mr : WithBot ℝ).unbotD 0) block (fun _ => 1)
      simp only [mul_one] at this
      rw [this, WithBot.unbotD_coe]
    have hsumT : (block.map (fun p => Real.exp (p.1 - (↑Mr : WithBot ℝ).unbotD 0) * p.2)).sum
        = Real.exp (-Mr) * (block.map (fun p => Real.exp p.1 * p.2)).sum := by
      rw [sr_sum_map_exp_sub ((↑Mr : WithBot ℝ).unbotD 0) block (fun p => p.2), WithBot.unbotD_coe]
    refine Prod.ext hfst.symm (Prod.ext ?_ ?_)
    · rw [hfold_l, hlα, hsumL, show ((↑Mr : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = Real.exp (-Mr) from rfl]; ring
    · rw [hfold_acc, haccα, hsumT, show ((↑Mr : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = Real.exp (-Mr) from rfl]; ring

/-! ### Memory-side data functions (Python test shape)

The per-program scalars and per-token data the kernel reads, at the test shape
(`stride_logic_h=256, stride_logic_bs=1, stride_vbs=8192, stride_vh=64,
stride_vd=1, stride_b_loc_b=128, stride_b_loc_s=1, max_input_len=128`).  The
online-softmax loop streams, for valid token `n < srSeqLen`, the pair
`(srQk n, srV n d)`. -/

/-- The loop bound: `cur_batch_seq_len = BSeqLen[cur_batch]`. -/
def srSeqLen (s : BlockState) (BSeqLen : RegionName) : Nat :=
  s.readMemValue .nat BSeqLen (s.pids 0)

/-- `cur_batch_start_loc = BStartLoc[cur_batch]`. -/
def srStartLoc (s : BlockState) (BStartLoc : RegionName) : Nat :=
  s.readMemValue .nat BStartLoc (s.pids 0)

set_option maxHeartbeats 1600000 in
/-- `castIntToNat` eval helper (pointwise `Int.toNat`). -/
theorem sr_evalOp_castIntToNat {shape : TileShape} (a : Op .int shape) (s : BlockState) :
    evalOp (.castIntToNat a) s = (do let va ← evalOp a s; some ⟨fun i => (va.data i).toNat⟩) := by
  simp [evalOp]

/-- **`old_scale` recipe** (`tl.exp(e_max − n_e_max)`, scalar). -/
theorem sr_oldscale_eval (s : BlockState) (em nem : WithBot ℝ)
    (hem : s.regs .real [] "e_max" = some (Tile.scalar em))
    (hnem : s.regs .real [] "n_e_max" = some (Tile.scalar nem)) :
    evalOp (Op.sub .real Broadcast.nil (Op.ref .real [] "e_max") (Op.ref .real [] "n_e_max")).exp s
      = some (Tile.scalar (WithBot.realExp (WithBot.realSub em nem))) := by
  rw [evalOp_exp]
  simp only [evalOp_sub, evalOp_ref, hem, hnem, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.uop_data, Tile.bop_data, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.sub]

/-! ### Block-reduction bridges (`srBlock` ↔ `Fin 64` masked reductions)

The loop body reduces a `64`-lane masked row over `Fin 64` (lane `jL` ↦ token
`c·64 + jL` when `c·64 + jL < S`, else masked); the `srOsStepBot` math uses the
`srBlock` list (a `Fin S` window filterMap of `[c·64, (c+1)·64)`). These bridges
equate the two by reindexing the window onto `Fin 64`, masked lanes (token ≥ S)
contributing `⊥`/`0`. -/

/-- filterMap-sum over `Fin n` with a guard collapses into the masked `Finset.sum`. -/
theorem sr_filterMap_finRange_sum {α : Type*} (n : Nat)
    (p : Fin n → Prop) [DecidablePred p] (g : Fin n → α) (h : α → ℝ) :
    (((List.finRange n).filterMap (fun j => if p j then some (g j) else none)).map h).sum
      = ∑ j : Fin n, if p j then h (g j) else 0 := by
  rw [List.map_filterMap]
  rw [show (fun j : Fin n => Option.map h (if p j then some (g j) else none))
        = (fun j : Fin n => if p j then some (h (g j)) else none) from by
    funext j; by_cases hj : p j <;> simp [hj]]
  rw [show (((List.finRange n).filterMap (fun j => if p j then some (h (g j)) else none))).sum
        = ((List.finRange n).map (fun j => if p j then h (g j) else 0)).sum from by
    induction (List.finRange n) with
    | nil => simp
    | cons a t ih => by_cases ha : p a <;> simp [ha, ih]]
  rw [← List.sum_ofFn]; congr 1; rw [List.ofFn_eq_map]

/-- The `WithBot` `foldr` of a guarded score list (coerced) equals the `Finset.sup`
over `Fin n` of the lane terms (`⊥` on filtered-out lanes). -/
theorem sr_filterMap_foldr_sup (n : Nat) (P : Fin n → Prop) [DecidablePred P]
    (sc : Fin n → ℝ) :
    (((List.finRange n).filterMap (fun j => if P j then some (sc j) else none)).map
        (fun x => ((x : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = Finset.univ.sup (fun j : Fin n => if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥) := by
  rw [show (((List.finRange n).filterMap (fun j => if P j then some (sc j) else none)).map
        (fun x => ((x : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = (List.finRange n).foldr (fun j a => (if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥) ⊔ a) ⊥ from by
    induction (List.finRange n) with
    | nil => simp
    | cons a t ih => by_cases ha : P a <;> simp [ha, ih]]
  apply le_antisymm
  · induction (List.finRange n) with
    | nil => simp
    | cons a t ih =>
      simp only [List.foldr_cons]
      exact sup_le (Finset.le_sup (f := fun j : Fin n => if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥)
        (Finset.mem_univ a)) ih
  · apply Finset.sup_le
    intro j _
    have key : ∀ (l : List (Fin n)), j ∈ l →
        (if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥)
          ≤ l.foldr (fun j a => (if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥) ⊔ a) ⊥ := by
      intro l hl
      induction l with
      | nil => simp at hl
      | cons a t ih =>
        simp only [List.foldr_cons]
        rcases List.mem_cons.mp hl with h | h
        · subst h; exact le_sup_left
        · exact le_trans (ih h) le_sup_right
    exact key _ (List.mem_finRange j)

/-- Any member of a `WithBot ℝ` list is `≤` its `foldr (⊔) ⊥`. -/
theorem sr_mem_le_foldr_sup (a : WithBot ℝ) :
    ∀ (L : List (WithBot ℝ)), a ∈ L → a ≤ L.foldr (· ⊔ ·) ⊥ := by
  intro L
  induction L with
  | nil => intro h; simp at h
  | cons b t ih =>
    intro h
    rcases List.mem_cons.mp h with h | h
    · subst h; exact le_sup_left
    · exact le_trans (ih h) le_sup_right

/-- `srSeqLen` depends only on `mem`/`pids`. -/
theorem srSeqLen_eq_of_mem_pids (s s0 : BlockState) (BSeqLen : RegionName)
    (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids) :
    srSeqLen s BSeqLen = srSeqLen s0 BSeqLen := by
  simp only [srSeqLen, BlockState.readMemValue, BlockState.readMemTyped, hmem, hpids]

/-- `srStartLoc` depends only on `mem`/`pids`. -/
theorem srStartLoc_eq_of_mem_pids (s s0 : BlockState) (BStartLoc : RegionName)
    (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids) :
    srStartLoc s BStartLoc = srStartLoc s0 BStartLoc := by
  simp only [srStartLoc, BlockState.readMemValue, BlockState.readMemTyped, hmem, hpids]

/-- General windowed `Finset.sup` reindex onto `Fin BLOCK_N` (block step `BLOCK_N`). -/
theorem sr_window_sup_reindexG (BLOCK_N c S : Nat) (hwin : (c + 1) * BLOCK_N ≤ S)
    (F : Nat → WithBot ℝ) :
    Finset.univ.sup (fun j : Fin S =>
        if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N then F j.val else ⊥)
      = Finset.univ.sup (fun jL : Fin BLOCK_N => F (c * BLOCK_N + jL.val)) := by
  apply le_antisymm
  · apply Finset.sup_le
    intro j _
    by_cases hj : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N
    · rw [if_pos hj]
      have hjL : j.val - c * BLOCK_N < BLOCK_N := by
        have := hj.2; rw [Nat.succ_mul] at this; omega
      refine le_trans ?_ (Finset.le_sup
        (f := fun jL : Fin BLOCK_N => F (c * BLOCK_N + jL.val))
        (Finset.mem_univ (⟨j.val - c * BLOCK_N, hjL⟩ : Fin BLOCK_N)))
      simp only
      rw [show c * BLOCK_N + (j.val - c * BLOCK_N) = j.val from by omega]
    · rw [if_neg hj]; exact bot_le
  · apply Finset.sup_le
    intro jL _
    have hb : c * BLOCK_N + jL.val < S := by
      have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this
      have := jL.isLt; omega
    refine le_trans ?_ (Finset.le_sup
      (f := fun j : Fin S =>
        if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N then F j.val else ⊥)
      (Finset.mem_univ (⟨c * BLOCK_N + jL.val, hb⟩ : Fin S)))
    simp only
    rw [if_pos (by have := jL.isLt; rw [Nat.succ_mul]; exact ⟨by omega, by omega⟩)]

/-- General `srBlock` running-sup bridge (`Fin BLOCK_N` lanes, channel `Fin BLOCK_DMODEL`). -/
theorem srBlock_sup_eqG {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (BLOCK_N c : Nat) (d : Fin BLOCK_DMODEL)
    (hwin : (c + 1) * BLOCK_N ≤ S) :
    Finset.univ.sup (fun jL : Fin BLOCK_N =>
        if c * BLOCK_N + jL.val < S then
          ((qk ⟨c * BLOCK_N + jL.val, by
            have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ : ℝ) : WithBot ℝ)
        else (⊥ : WithBot ℝ))
      = ((srBlock qk v BLOCK_N c d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  classical
  set F : Nat → WithBot ℝ := fun jg =>
    if h : jg < S then ((qk ⟨jg, h⟩ : ℝ) : WithBot ℝ) else ⊥ with hF
  rw [show (srBlock qk v BLOCK_N c d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
        = ((List.finRange S).filterMap (fun j : Fin S =>
            if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N then some (qk j) else none)).map
            (fun x => ((x : ℝ) : WithBot ℝ)) from by
    unfold srBlock
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr
    intro j _
    by_cases hj : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N <;> simp [hj]]
  rw [sr_filterMap_foldr_sup S
    (fun j => c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N) (fun j => qk j)]
  rw [show (Finset.univ.sup (fun j : Fin S =>
        if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N then ((qk j : ℝ) : WithBot ℝ) else ⊥))
      = Finset.univ.sup (fun j : Fin S =>
          if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N then F j.val else ⊥) from by
    apply Finset.sup_congr rfl
    intro j _
    by_cases hw : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N
    · rw [if_pos hw, if_pos hw, hF]; simp only [dif_pos j.isLt]
    · rw [if_neg hw, if_neg hw]]
  rw [sr_window_sup_reindexG BLOCK_N c S hwin F]
  apply Finset.sup_congr rfl
  intro jL _
  have hb : c * BLOCK_N + jL.val < S := by
    have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega
  by_cases hq : c * BLOCK_N + jL.val < S
  · rw [if_pos hq, hF]; simp only [dif_pos hb]
  · exact absurd hb hq

/-- General `srBlock` map-and-sum bridge. -/
theorem srBlock_map_sumG {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (BLOCK_N c : Nat) (d : Fin BLOCK_DMODEL)
    (hwin : (c + 1) * BLOCK_N ≤ S) (h : ℝ × ℝ → ℝ) :
    ((srBlock qk v BLOCK_N c d).map h).sum
      = ∑ jL : Fin BLOCK_N,
          (if c * BLOCK_N + jL.val < S then
            h (qk ⟨c * BLOCK_N + jL.val, by
                have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩,
               v ⟨c * BLOCK_N + jL.val, by
                have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ d)
           else 0) := by
  classical
  rw [srBlock, sr_filterMap_finRange_sum S
    (fun j => c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N)
    (fun j => (qk j, v j d)) h]
  rw [← Finset.sum_filter
        (fun j : Fin S => c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N)
        (fun j => h (qk j, v j d))]
  symm
  rw [← Finset.sum_filter
        (fun jL : Fin BLOCK_N => c * BLOCK_N + jL.val < S)
        (fun jL => h (qk ⟨c * BLOCK_N + jL.val, by
            have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩,
            v ⟨c * BLOCK_N + jL.val, by
            have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ d))]
  refine Finset.sum_bij
    (i := fun jL (_ : jL ∈ Finset.univ.filter (fun jL : Fin BLOCK_N => c * BLOCK_N + jL.val < S)) =>
      (⟨c * BLOCK_N + jL.val, by
        have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ : Fin S)) ?_ ?_ ?_ ?_
  · intro jL hjL
    have hlt : c * BLOCK_N + jL.val < S := (Finset.mem_filter.mp hjL).2
    rw [Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    show c * BLOCK_N ≤ c * BLOCK_N + jL.val ∧ c * BLOCK_N + jL.val < (c + 1) * BLOCK_N
    have := jL.isLt; rw [Nat.succ_mul]; omega
  · intro a _ b _ hab
    apply Fin.ext
    have : c * BLOCK_N + a.val = c * BLOCK_N + b.val := by simpa using congrArg Fin.val hab
    omega
  · intro j hj
    have hj2 : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N := (Finset.mem_filter.mp hj).2
    refine ⟨⟨j.val - c * BLOCK_N, by rw [Nat.succ_mul] at hj2; omega⟩, ?_, by apply Fin.ext; simp only; omega⟩
    rw [Finset.mem_filter]
    exact ⟨Finset.mem_univ _, by simp only; omega⟩
  · intro jL _; rfl

/-- General `srBlock` `e_sum` lane-sum bridge. -/
theorem srBlock_esum_sumG {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (BLOCK_N c : Nat) (d : Fin BLOCK_DMODEL) (Mr : ℝ)
    (hwin : (c + 1) * BLOCK_N ≤ S) :
    (∑ jL : Fin BLOCK_N, WithBot.realExp (WithBot.realSub
        (if c * BLOCK_N + jL.val < S then ((qk ⟨c * BLOCK_N + jL.val, by
            have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ : ℝ) : WithBot ℝ)
         else (⊥ : WithBot ℝ)) ((Mr : ℝ) : WithBot ℝ)))
      = some ((srBlock qk v BLOCK_N c d).map (fun p => Real.exp (p.1 - Mr))).sum := by
  have hcell : ∀ jL : Fin BLOCK_N,
      WithBot.realExp (WithBot.realSub
        (if c * BLOCK_N + jL.val < S then ((qk ⟨c * BLOCK_N + jL.val, by
            have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ : ℝ) : WithBot ℝ)
         else (⊥ : WithBot ℝ)) ((Mr : ℝ) : WithBot ℝ))
        = some (if c * BLOCK_N + jL.val < S
            then Real.exp (qk ⟨c * BLOCK_N + jL.val, by
              have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ - Mr) else 0) := by
    intro jL
    by_cases hj : c * BLOCK_N + jL.val < S
    · rw [if_pos hj, if_pos hj, WithBot.realSub_coe_coe, WithBot.realExp_coe]; rfl
    · rw [if_neg hj, if_neg hj, WithBot.realSub_bot_left, WithBot.realExp_bot]; rfl
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [srBlock_map_sumG qk v BLOCK_N c d hwin (fun p => Real.exp (p.1 - Mr))]

/-- General `srBlock` `acc` lane-sum bridge. -/
theorem srBlock_acc_sumG {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (BLOCK_N c : Nat) (d : Fin BLOCK_DMODEL) (Mr : ℝ)
    (rawV : Fin BLOCK_N → ℝ)
    (hwin : (c + 1) * BLOCK_N ≤ S)
    (hrawV : ∀ jL : Fin BLOCK_N, (hj : c * BLOCK_N + jL.val < S) →
      rawV jL = v ⟨c * BLOCK_N + jL.val, by
        have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ d) :
    (∑ jL : Fin BLOCK_N, WithBot.realMul
        (WithBot.realExp (WithBot.realSub
          (if c * BLOCK_N + jL.val < S then ((qk ⟨c * BLOCK_N + jL.val, by
            have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ : ℝ) : WithBot ℝ)
           else (⊥ : WithBot ℝ)) ((Mr : ℝ) : WithBot ℝ)))
        ((rawV jL : ℝ) : WithBot ℝ))
      = some ((srBlock qk v BLOCK_N c d).map (fun p => Real.exp (p.1 - Mr) * p.2)).sum := by
  have hcell : ∀ jL : Fin BLOCK_N,
      WithBot.realMul
        (WithBot.realExp (WithBot.realSub
          (if c * BLOCK_N + jL.val < S then ((qk ⟨c * BLOCK_N + jL.val, by
            have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ : ℝ) : WithBot ℝ)
           else (⊥ : WithBot ℝ)) ((Mr : ℝ) : WithBot ℝ)))
        ((rawV jL : ℝ) : WithBot ℝ)
        = some (if c * BLOCK_N + jL.val < S
            then Real.exp (qk ⟨c * BLOCK_N + jL.val, by
              have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ - Mr)
                  * v ⟨c * BLOCK_N + jL.val, by
              have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ d
            else 0) := by
    intro jL
    by_cases hj : c * BLOCK_N + jL.val < S
    · rw [if_pos hj, if_pos hj, WithBot.realSub_coe_coe, WithBot.realExp_coe,
        WithBot.realMul_coe_coe, hrawV jL hj]; rfl
    · rw [if_neg hj, if_neg hj, WithBot.realSub_bot_left, WithBot.realExp_bot]
      show WithBot.realMul ((0:ℝ):WithBot ℝ) ((rawV jL : ℝ):WithBot ℝ) = some 0
      rw [WithBot.realMul_coe_coe, zero_mul]; rfl
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [srBlock_map_sumG qk v BLOCK_N c d hwin (fun p => Real.exp (p.1 - Mr) * p.2)]

/-- General channel-independence of `srRunningMax`. -/
theorem srRunningMax_eqG {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (hi : Nat) (d d' : Fin BLOCK_DMODEL) :
    srRunningMax qk v hi d = srRunningMax qk v hi d' := by
  unfold srRunningMax srKeysUpto
  rw [List.map_filterMap, List.map_filterMap]
  congr 1
  apply List.filterMap_congr
  intro n _
  by_cases hn : n.val < hi <;> simp [hn]

/-- General one-block advance of the `⊥`-seeded running max (block step `BLOCK_N`). -/
theorem srRunningMax_succG {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (BLOCK_N c : Nat) (d : Fin BLOCK_DMODEL)
    (hwin : (c + 1) * BLOCK_N ≤ S) :
    srRunningMax qk v ((c + 1) * BLOCK_N) d
      = srRunningMax qk v (c * BLOCK_N) d
        ⊔ Finset.univ.sup (fun jL : Fin BLOCK_N =>
            if c * BLOCK_N + jL.val < S then
              ((qk ⟨c * BLOCK_N + jL.val, by
                have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ : ℝ) : WithBot ℝ)
            else (⊥ : WithBot ℝ)) := by
  unfold srRunningMax
  rw [srKeysUpto_succ, List.map_append, List.foldr_append]
  rw [srBlock_sup_eqG qk v BLOCK_N c d hwin]
  induction (srKeysUpto qk v (c * BLOCK_N) d).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) with
  | nil => simp
  | cons a t ih =>
    simp only [List.foldr_cons]
    rw [ih, ← sup_assoc]

/-- General non-`⊥` of `srRunningMax` over a nonempty window. -/
theorem srRunningMax_ne_botG {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (hi : Nat) (d : Fin BLOCK_DMODEL)
    (hpos : 0 < hi) (hle : hi ≤ S) :
    srRunningMax qk v hi d ≠ ⊥ := by
  have hS : 0 < S := lt_of_lt_of_le hpos hle
  have hmem : ((qk ⟨0, hS⟩ : ℝ) : WithBot ℝ) ∈
      (srKeysUpto qk v hi d).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) := by
    unfold srKeysUpto
    rw [List.map_filterMap]
    rw [List.mem_filterMap]
    refine ⟨⟨0, hS⟩, List.mem_finRange _, ?_⟩
    simp only [hpos, if_true, Option.map_some]
  have hle' := sr_mem_le_foldr_sup _ _ hmem
  rw [← srRunningMax] at hle'
  intro hbot
  rw [hbot] at hle'
  exact absurd (le_bot_iff.mp hle') WithBot.coe_ne_bot

/-- General ⊥-seeded `e_sum` anchor. -/
theorem sr_denom_anchorG {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (hi : Nat) (d : Fin BLOCK_DMODEL) :
    (srStateBot qk v hi d).2.1
      = ((srStateBot qk v hi d).1.elim 0 (fun r => Real.exp (-r)))
        * (0 + ((srKeysUpto qk v hi d).map (fun p => Real.exp p.1)).sum) := by
  rw [srStateBot_snd_fst, srStateBot_fst_eq_runningMax, zero_add]

/-- General ⊥-seeded `acc` anchor. -/
theorem sr_acc_anchorG {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (hi : Nat) (d : Fin BLOCK_DMODEL) :
    (srStateBot qk v hi d).2.2
      = ((srStateBot qk v hi d).1.elim 0 (fun r => Real.exp (-r)))
        * (0 + ((srKeysUpto qk v hi d).map (fun p => Real.exp p.1 * p.2)).sum) := by
  rw [srStateBot_snd_snd, srStateBot_fst_eq_runningMax, zero_add]

/-- General: ⊥ running max ⇒ key sum is `0`. -/
theorem srKeysUpto_sum_zero_of_botG {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (hi : Nat) (d : Fin BLOCK_DMODEL)
    (hbot : srRunningMax qk v hi d = ⊥) (h : ℝ × ℝ → ℝ) :
    ((srKeysUpto qk v hi d).map h).sum = 0 := by
  rw [show srKeysUpto qk v hi d = [] from ?_, List.map_nil, List.sum_nil]
  by_contra hne
  obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ hne
  have hmem : ((p.1 : ℝ) : WithBot ℝ) ∈
      (srKeysUpto qk v hi d).map (fun q => ((q.1 : ℝ) : WithBot ℝ)) :=
    List.mem_map_of_mem hp
  have := sr_mem_le_foldr_sup _ _ hmem
  rw [← srRunningMax, hbot] at this
  exact absurd (le_bot_iff.mp this) WithBot.coe_ne_bot

/-- General `n_e_max = srRunningMax((c+1)·BLOCK_N)`. Channel-0 witness via `0 < BLOCK_DMODEL`. -/
theorem sr_nemax_reg_eqG {S BLOCK_DMODEL : Nat} (hD : 0 < BLOCK_DMODEL)
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (BLOCK_N c : Nat) (d : Fin BLOCK_DMODEL)
    (hwin : (c + 1) * BLOCK_N ≤ S) (hN : (0 : Nat) < BLOCK_N)
    (hne : (Finset.univ : Finset (Fin BLOCK_N)).Nonempty) :
    srRunningMax qk v (c * BLOCK_N) d
        ⊔ Finset.univ.sup' hne (fun jL : Fin BLOCK_N =>
            if c * BLOCK_N + jL.val < S then
              ((qk ⟨c * BLOCK_N + jL.val, by
                have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ : ℝ) : WithBot ℝ)
            else (⊥ : WithBot ℝ))
      = srRunningMax qk v ((c + 1) * BLOCK_N) d := by
  rw [Finset.sup'_eq_sup, srRunningMax_succG qk v BLOCK_N c d hwin]

/-- General `e_sum = srStateBot((c+1)·BLOCK_N).2.1`. -/
theorem sr_esum_reg_eqG {S BLOCK_DMODEL : Nat} (hD : 0 < BLOCK_DMODEL)
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (BLOCK_N c : Nat) (d : Fin BLOCK_DMODEL)
    (hwin : (c + 1) * BLOCK_N ≤ S) (hN : (0 : Nat) < BLOCK_N) :
    WithBot.realAdd
      (WithBot.realMul ((srStateBot qk v (c * BLOCK_N) ⟨0, hD⟩).2.1 : WithBot ℝ)
        (WithBot.realExp (WithBot.realSub
          (srRunningMax qk v (c * BLOCK_N) ⟨0, hD⟩)
          (srRunningMax qk v ((c + 1) * BLOCK_N) ⟨0, hD⟩))))
      (∑ jL : Fin BLOCK_N, WithBot.realExp (WithBot.realSub
        (if c * BLOCK_N + jL.val < S then ((qk ⟨c * BLOCK_N + jL.val, by
            have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ : ℝ) : WithBot ℝ)
         else (⊥ : WithBot ℝ))
        (srRunningMax qk v ((c + 1) * BLOCK_N) ⟨0, hD⟩)))
      = ((srStateBot qk v ((c + 1) * BLOCK_N) d).2.1 : WithBot ℝ) := by
  set m := (srStateBot qk v (c * BLOCK_N) d).1 with hm_def
  set Mc := srRunningMax qk v (c * BLOCK_N) ⟨0, hD⟩ with hMc
  set Mc1 := srRunningMax qk v ((c + 1) * BLOCK_N) ⟨0, hD⟩ with hMc1
  have hmMc : m = Mc := by
    rw [hm_def, hMc, srStateBot_fst_eq_runningMax, srRunningMax_eqG qk v (c * BLOCK_N) d ⟨0, hD⟩]
  have hne : Mc1 ≠ ⊥ := srRunningMax_ne_botG qk v ((c + 1) * BLOCK_N) ⟨0, hD⟩
    (Nat.mul_pos (Nat.succ_pos c) hN) hwin
  obtain ⟨Mr, hMr⟩ : ∃ Mr : ℝ, Mc1 = (Mr : WithBot ℝ) := by
    cases hh : Mc1 with
    | bot => exact absurd hh hne
    | coe x => exact ⟨x, rfl⟩
  have hMsucc : Mc1 = m ⊔ ((srBlock qk v BLOCK_N c d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    have h1 : Mc1 = (srStateBot qk v ((c + 1) * BLOCK_N) d).1 := by
      rw [hMc1, srStateBot_fst_eq_runningMax, srRunningMax_eqG qk v ((c+1)*BLOCK_N) ⟨0, hD⟩ d]
    rw [h1, srStateBot_succ, srOsStepBot_block_fst m
      ((srStateBot qk v (c * BLOCK_N) d).2.1) ((srStateBot qk v (c * BLOCK_N) d).2.2)]
  have hsum := srBlock_esum_sumG qk v BLOCK_N c d Mr hwin
  have hblockEq := srOsStepBot_block_eq m
    ((srStateBot qk v (c * BLOCK_N) d).2.1) ((srStateBot qk v (c * BLOCK_N) d).2.2)
    ((srKeysUpto qk v (c * BLOCK_N) d).map (fun p => Real.exp p.1 * p.2)).sum
    ((srKeysUpto qk v (c * BLOCK_N) d).map (fun p => Real.exp p.1)).sum
    (srBlock qk v BLOCK_N c d)
    (by rw [sr_denom_anchorG, zero_add, hm_def])
    (by rw [sr_acc_anchorG, zero_add, hm_def])
    (fun hbot => srKeysUpto_sum_zero_of_botG qk v (c * BLOCK_N) d
      (by rw [← srStateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => srKeysUpto_sum_zero_of_botG qk v (c * BLOCK_N) d
      (by rw [← srStateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  rw [show (srStateBot qk v ((c + 1) * BLOCK_N) d).2.1
        = (Mc1, (srStateBot qk v (c * BLOCK_N) d).2.1
              * (WithBot.realExp (WithBot.realSub m Mc1)).unbotD 0
              + ((srBlock qk v BLOCK_N c d).map (fun p => Real.exp (p.1 - Mc1.unbotD 0))).sum, _).2.1
        from by rw [srStateBot_succ]; rw [← hblockEq]]
  set α : ℝ := (WithBot.realExp (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  have hkeq : ∀ (e e' : Fin BLOCK_DMODEL),
      ((srKeysUpto qk v (c * BLOCK_N) e).map (fun p => Real.exp p.1)).sum
        = ((srKeysUpto qk v (c * BLOCK_N) e').map (fun p => Real.exp p.1)).sum := by
    intro e e'
    unfold srKeysUpto
    rw [List.map_filterMap, List.map_filterMap]
    congr 1
    apply List.filterMap_congr; intro n _; by_cases hn : n.val < c*BLOCK_N <;> simp [hn]
  have hcind : ((srStateBot qk v (c * BLOCK_N) ⟨0, hD⟩).2.1 : WithBot ℝ)
        = ((srStateBot qk v (c * BLOCK_N) d).2.1 : WithBot ℝ) := by
    rw [srStateBot_snd_fst, srStateBot_snd_fst, srRunningMax_eqG qk v (c*BLOCK_N) ⟨0, hD⟩ d]
    rw [hkeq ⟨0, hD⟩ d]
  rw [hcind]
  rw [show WithBot.realSub Mc Mc1 = WithBot.realSub m Mc1 from by rw [hmMc]]
  rw [show (∑ jL : Fin BLOCK_N, WithBot.realExp (WithBot.realSub
        (if c * BLOCK_N + jL.val < S then ((qk ⟨c * BLOCK_N + jL.val, by
            have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ : ℝ) : WithBot ℝ)
         else (⊥ : WithBot ℝ)) Mc1))
      = some ((srBlock qk v BLOCK_N c d).map (fun p => Real.exp (p.1 - Mr))).sum from by
    rw [hMr]; exact hsum]
  rw [show (Mc1.unbotD 0) = Mr from by rw [hMr, WithBot.unbotD_coe]]
  rw [show ((srStateBot qk v (c * BLOCK_N) d).2.1 : WithBot ℝ)
        = some (srStateBot qk v (c * BLOCK_N) d).2.1 from rfl, hαsome]
  simp only [WithBot.realMul, WithBot.realAdd, Option.map₂, Option.bind, Option.map]
  rfl

/-- General `acc[d] = srStateBot((c+1)·BLOCK_N).2.2` (per channel `d`). -/
theorem sr_acc_reg_eqG {S BLOCK_DMODEL : Nat} (hD : 0 < BLOCK_DMODEL)
    (qk : Fin S → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (BLOCK_N c : Nat) (d : Fin BLOCK_DMODEL)
    (rawV : Fin BLOCK_N → ℝ)
    (hwin : (c + 1) * BLOCK_N ≤ S) (hN : (0 : Nat) < BLOCK_N)
    (hrawV : ∀ jL : Fin BLOCK_N, (hj : c * BLOCK_N + jL.val < S) →
      rawV jL = v ⟨c * BLOCK_N + jL.val, by
        have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ d) :
    WithBot.realAdd
      (WithBot.realMul ((srStateBot qk v (c * BLOCK_N) d).2.2 : WithBot ℝ)
        (WithBot.realExp (WithBot.realSub
          (srRunningMax qk v (c * BLOCK_N) ⟨0, hD⟩)
          (srRunningMax qk v ((c + 1) * BLOCK_N) ⟨0, hD⟩))))
      (∑ jL : Fin BLOCK_N, WithBot.realMul
        (WithBot.realExp (WithBot.realSub
          (if c * BLOCK_N + jL.val < S then ((qk ⟨c * BLOCK_N + jL.val, by
            have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ : ℝ) : WithBot ℝ)
           else (⊥ : WithBot ℝ))
          (srRunningMax qk v ((c + 1) * BLOCK_N) ⟨0, hD⟩)))
        ((rawV jL : ℝ) : WithBot ℝ))
      = ((srStateBot qk v ((c + 1) * BLOCK_N) d).2.2 : WithBot ℝ) := by
  set m := (srStateBot qk v (c * BLOCK_N) d).1 with hm_def
  set Mc := srRunningMax qk v (c * BLOCK_N) ⟨0, hD⟩ with hMc
  set Mc1 := srRunningMax qk v ((c + 1) * BLOCK_N) ⟨0, hD⟩ with hMc1
  have hmMc : m = Mc := by
    rw [hm_def, hMc, srStateBot_fst_eq_runningMax, srRunningMax_eqG qk v (c * BLOCK_N) d ⟨0, hD⟩]
  have hne : Mc1 ≠ ⊥ := srRunningMax_ne_botG qk v ((c + 1) * BLOCK_N) ⟨0, hD⟩
    (Nat.mul_pos (Nat.succ_pos c) hN) hwin
  obtain ⟨Mr, hMr⟩ : ∃ Mr : ℝ, Mc1 = (Mr : WithBot ℝ) := by
    cases hh : Mc1 with
    | bot => exact absurd hh hne
    | coe x => exact ⟨x, rfl⟩
  have hMsucc : Mc1 = m ⊔ ((srBlock qk v BLOCK_N c d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    have h1 : Mc1 = (srStateBot qk v ((c + 1) * BLOCK_N) d).1 := by
      rw [hMc1, srStateBot_fst_eq_runningMax, srRunningMax_eqG qk v ((c+1)*BLOCK_N) ⟨0, hD⟩ d]
    rw [h1, srStateBot_succ, srOsStepBot_block_fst m
      ((srStateBot qk v (c * BLOCK_N) d).2.1) ((srStateBot qk v (c * BLOCK_N) d).2.2)]
  have hsum := srBlock_acc_sumG qk v BLOCK_N c d Mr rawV hwin hrawV
  have hblockEq := srOsStepBot_block_eq m
    ((srStateBot qk v (c * BLOCK_N) d).2.1) ((srStateBot qk v (c * BLOCK_N) d).2.2)
    ((srKeysUpto qk v (c * BLOCK_N) d).map (fun p => Real.exp p.1 * p.2)).sum
    ((srKeysUpto qk v (c * BLOCK_N) d).map (fun p => Real.exp p.1)).sum
    (srBlock qk v BLOCK_N c d)
    (by rw [sr_denom_anchorG, zero_add, hm_def])
    (by rw [sr_acc_anchorG, zero_add, hm_def])
    (fun hbot => srKeysUpto_sum_zero_of_botG qk v (c * BLOCK_N) d
      (by rw [← srStateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => srKeysUpto_sum_zero_of_botG qk v (c * BLOCK_N) d
      (by rw [← srStateBot_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  rw [show (srStateBot qk v ((c + 1) * BLOCK_N) d).2.2
        = (Mc1, _, (srStateBot qk v (c * BLOCK_N) d).2.2
              * (WithBot.realExp (WithBot.realSub m Mc1)).unbotD 0
              + ((srBlock qk v BLOCK_N c d).map (fun p => Real.exp (p.1 - Mc1.unbotD 0) * p.2)).sum).2.2
        from by rw [srStateBot_succ]; rw [← hblockEq]]
  set α : ℝ := (WithBot.realExp (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  rw [show WithBot.realSub Mc Mc1 = WithBot.realSub m Mc1 from by rw [hmMc]]
  rw [show (∑ jL : Fin BLOCK_N, WithBot.realMul
        (WithBot.realExp (WithBot.realSub
          (if c * BLOCK_N + jL.val < S then ((qk ⟨c * BLOCK_N + jL.val, by
            have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ : ℝ) : WithBot ℝ)
           else (⊥ : WithBot ℝ)) Mc1))
        ((rawV jL : ℝ) : WithBot ℝ))
      = some ((srBlock qk v BLOCK_N c d).map (fun p => Real.exp (p.1 - Mr) * p.2)).sum from by
    rw [hMr]; exact hsum]
  rw [show (Mc1.unbotD 0) = Mr from by rw [hMr, WithBot.unbotD_coe]]
  rw [show ((srStateBot qk v (c * BLOCK_N) d).2.2 : WithBot ℝ)
        = some (srStateBot qk v (c * BLOCK_N) d).2.2 from rfl, hαsome]
  simp only [WithBot.realMul, WithBot.realAdd, Option.map₂, Option.bind, Option.map]
  rfl

/-! ### General stride-parameterized memory-side data functions -/

/-- General `off_b_loc = cur_batch·stride_b_loc_b + (max_input_len − seqlen)·stride_b_loc_s`. -/
def srOffBLocG (s : BlockState) (BSeqLen : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s : Nat) : Nat :=
  s.pids 0 * stride_b_loc_b + (max_input_len - srSeqLen s BSeqLen) * stride_b_loc_s

/-- General paged-KV index for token `n`: `BLoc[off_b_loc + n·stride_b_loc_s]`. -/
def srVIndexG (s : BlockState) (BLoc : Region .int) (BSeqLen : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s : Nat) (n : Nat) : Int :=
  s.readMemValue .int (Region.cast BLoc)
    (srOffBLocG s BSeqLen max_input_len stride_b_loc_b stride_b_loc_s + n * stride_b_loc_s)

/-- General logit for token `n`: `Logics[cur_head·stride_logic_h + (start_loc + n)·stride_logic_bs]`. -/
def srQkG (s : BlockState) (Logics BStartLoc : RegionName)
    (stride_logic_h stride_logic_bs : Nat) (n : Nat) : ℝ :=
  s.readMem Logics (s.pids 1 * stride_logic_h + (srStartLoc s BStartLoc + n) * stride_logic_bs)

/-- General gathered V-row entry: `V[v_index[n]·stride_vbs + cur_head·stride_vh + d·stride_vd]`. -/
def srVG (s : BlockState) (V : RegionName) (BLoc : Region .int) (BSeqLen : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_vbs stride_vh stride_vd : Nat)
    (n d : Nat) : ℝ :=
  s.readMem V ((srVIndexG s BLoc BSeqLen max_input_len stride_b_loc_b stride_b_loc_s n * stride_vbs).toNat
    + s.pids 1 * stride_vh + d * stride_vd)

/-- General per-token logit over valid tokens. -/
noncomputable def srQkFG (s0 : BlockState) (Logics BStartLoc BSeqLen : RegionName)
    (stride_logic_h stride_logic_bs : Nat) : Fin (srSeqLen s0 BSeqLen) → ℝ :=
  fun n => srQkG s0 Logics BStartLoc stride_logic_h stride_logic_bs n.val

/-- General per-token gathered V-row over valid tokens and `Fin BLOCK_DMODEL`. -/
noncomputable def srVFG (s0 : BlockState) (V : RegionName) (BLoc : Region .int)
    (BSeqLen : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_vbs stride_vh stride_vd : Nat)
    (BLOCK_DMODEL : Nat) : Fin (srSeqLen s0 BSeqLen) → Fin BLOCK_DMODEL → ℝ :=
  fun n d => srVG s0 V BLoc BSeqLen max_input_len stride_b_loc_b stride_b_loc_s
    stride_vbs stride_vh stride_vd n.val d.val

/-- General data-fn invariance under `mem`/`pids`. -/
theorem srVIndexG_eq_of_mem_pids (s s0 : BlockState) (BLoc : Region .int) (BSeqLen : RegionName)
    (mil sb ss : Nat) (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids) (n : Nat) :
    srVIndexG s BLoc BSeqLen mil sb ss n = srVIndexG s0 BLoc BSeqLen mil sb ss n := by
  simp only [srVIndexG, srOffBLocG, srSeqLen, BlockState.readMemValue, BlockState.readMemTyped, hmem, hpids]

theorem srQkG_eq_of_mem_pids (s s0 : BlockState) (Logics BStartLoc : RegionName)
    (slh slb : Nat) (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids) (n : Nat) :
    srQkG s Logics BStartLoc slh slb n = srQkG s0 Logics BStartLoc slh slb n := by
  simp only [srQkG, srStartLoc, BlockState.readMem, BlockState.readMemValue, BlockState.readMemTyped, hmem, hpids]

theorem srVG_eq_of_mem_pids (s s0 : BlockState) (V : RegionName) (BLoc : Region .int)
    (BSeqLen : RegionName) (mil sb ss svbs svh svd : Nat)
    (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids) (n d : Nat) :
    srVG s V BLoc BSeqLen mil sb ss svbs svh svd n d
      = srVG s0 V BLoc BSeqLen mil sb ss svbs svh svd n d := by
  simp only [srVG, srVIndexG, srOffBLocG, srSeqLen, BlockState.readMem, BlockState.readMemValue,
    BlockState.readMemTyped, hmem, hpids]

/-! ### General stride/shape-parameterized op-eval recipes -/

set_option maxHeartbeats 1600000 in
/-- General `qk` masked-load recipe over `Fin BLOCK_N` lanes and strides. -/
theorem sr_qk_load_evalG (BLOCK_N : Nat) (s : BlockState) (Logics BStartLoc BSeqLen : RegionName)
    (SN slh slb : Nat)
    (hsl : s.regs .nat [] "cur_batch_start_loc" = some (Tile.scalar (srStartLoc s BStartLoc)))
    (hch : s.regs .nat [] "cur_head" = some (Tile.scalar (s.pids 1)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hseq : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (srSeqLen s BSeqLen))) :
    evalOp (Op.load .real
        (MemAccess.region Logics
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat slh))
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_start_loc")
                  (Op.ref .nat [] "start_n"))
                (Op.ref .nat [BLOCK_N] "offs_n"))
              (Op.constNat slb))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BLOCK_N] "offs_n"))
            (Op.ref .nat [] "cur_batch_seq_len"))
          (Op.negInf.broadcast [BLOCK_N]))) s
      = some (⟨fun idx : TileIndex [BLOCK_N] =>
          if SN + idx.1.val < srSeqLen s BSeqLen then some (srQkG s Logics BStartLoc slh slb (SN + idx.1.val))
          else (⊥ : WithBot ℝ)⟩ : Tile .real [BLOCK_N]) := by
  simp only [evalOp, hsl, hch, hsn, hn, hseq, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, u⟩ := idx
  simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar, NumericDType.add,
    NumericDType.mul, ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex,
    BlockState.readMemValue_real, Region.cast_id, srQkG]
  have haddr : s.pids 1 * slh + (srStartLoc s BStartLoc + SN + j.val) * slb
      = s.pids 1 * slh + (srStartLoc s BStartLoc + (SN + j.val)) * slb := by ring
  by_cases hlt : SN + j.val < srSeqLen s BSeqLen
  · simp only [hlt, decide_true, if_true, if_pos hlt]
    rw [haddr]
  · simp only [hlt, decide_false, if_false, if_neg hlt, Bool.false_eq_true]
    rfl

set_option maxHeartbeats 1600000 in
/-- General `v_index` masked-gather recipe over `Fin BLOCK_N` lanes and strides. -/
theorem sr_vindex_gather_evalG (BLOCK_N : Nat) (s : BlockState) (BLoc : Region .int)
    (BSeqLen : RegionName) (SN mil sb ss : Nat) (other_kv_index : Int)
    (hoff : s.regs .nat [] "off_b_loc" = some (Tile.scalar (srOffBLocG s BSeqLen mil sb ss)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hseq : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (srSeqLen s BSeqLen))) :
    evalOp (Op.load .int
        (MemAccess.region BLoc
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "off_b_loc")
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BLOCK_N] "offs_n"))
              (Op.constNat ss))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BLOCK_N] "offs_n"))
            (Op.ref .nat [] "cur_batch_seq_len"))
          ((Op.constInt other_kv_index).broadcast [BLOCK_N]))) s
      = some (⟨fun idx : TileIndex [BLOCK_N] =>
          if SN + idx.1.val < srSeqLen s BSeqLen then srVIndexG s BLoc BSeqLen mil sb ss (SN + idx.1.val)
          else other_kv_index⟩ : Tile .int [BLOCK_N]) := by
  simp only [evalOp, hoff, hsn, hn, hseq, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, u⟩ := idx
  simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar, NumericDType.add,
    NumericDType.mul, ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex,
    srVIndexG]
  by_cases hlt : SN + j.val < srSeqLen s BSeqLen
  · simp only [hlt, decide_true, if_true, if_pos hlt]
  · simp only [hlt, decide_false, if_false, if_neg hlt, Bool.false_eq_true]

set_option maxHeartbeats 1600000 in
/-- General `off_v` recipe (`cur_head·stride_vh + offs_d[None,:]·stride_vd`, shape `[1, BLOCK_DMODEL]`). -/
theorem sr_offv_evalG (BLOCK_DMODEL : Nat) (s : BlockState) (head svh svd : Nat)
    (hch : s.regs .nat [] "cur_head" = some (Tile.scalar head))
    (hd : s.regs .nat [BLOCK_DMODEL] "offs_d" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat svh))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
          (Op.constNat svd))) s
      = some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] => head * svh + idx.2.1.val * svd⟩ : Tile .nat [1, BLOCK_DMODEL]) := by
  have hexp : @evalOp .nat [1, BLOCK_DMODEL] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d")) s
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BLOCK_DMODEL] ⟨0, by simp⟩ "offs_d" s _ hd
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hexp, hch,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp [Tile.bop, Tile.vec, Tile.expandDim_data, NumericDType.mul, NumericDType.add]

set_option maxHeartbeats 1600000 in
/-- General `v` paged-gather recipe (`tl.load(v_ptrs + v_index[:,None]·stride_vbs)`, unmasked,
shape `[BLOCK_N, BLOCK_DMODEL]`). -/
theorem sr_v_gather_evalG (BLOCK_N BLOCK_DMODEL : Nat) (s : BlockState) (V : RegionName)
    (svbs : Nat) (offFn : Fin BLOCK_DMODEL → Nat) (vIdxFn : Fin BLOCK_N → Int)
    (hvp : s.regs .ptr [1, BLOCK_DMODEL] "v_ptrs" =
      some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] => (V, offFn idx.2.1)⟩ : Tile .ptr [1, BLOCK_DMODEL]))
    (hvi : s.regs .int [BLOCK_N] "v_index" =
      some (⟨fun idx : TileIndex [BLOCK_N] => vIdxFn idx.1⟩ : Tile .int [BLOCK_N])) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consR.consL (Op.ref .ptr [1, BLOCK_DMODEL] "v_ptrs")
            (Op.mul .int Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .int [BLOCK_N] "v_index"))
                (Op.constNat svbs).castNatToInt).castIntToNat))
        MaskOpt.none) s
      = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
          some (s.readMem V ((vIdxFn idx.1 * svbs).toNat + offFn idx.2.1))⟩
          : Tile .real [BLOCK_N, BLOCK_DMODEL]) := by
  have hexp : @evalOp .int [BLOCK_N, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .int [BLOCK_N] "v_index")) s
      = some (Tile.expandDim ⟨1, by simp⟩
          (⟨fun idx : TileIndex [BLOCK_N] => vIdxFn idx.1⟩ : Tile .int [BLOCK_N])) :=
    evalOp_expandDim_ref_of_regs .int [BLOCK_N] ⟨1, by simp⟩ "v_index" s _ hvi
  have hc : @evalOp .int [] (Op.constNat svbs).castNatToInt s
      = some (Tile.scalar (svbs : Int)) := by
    simp only [evalOp, evalOp_constNat, Option.bind]; rfl
  have hmulint : @evalOp .int [BLOCK_N, 1]
        (Op.mul .int Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .int [BLOCK_N] "v_index"))
          (Op.constNat svbs).castNatToInt) s
      = some (⟨fun idx : TileIndex [BLOCK_N, 1] => vIdxFn idx.1 * svbs⟩ : Tile .int [BLOCK_N, 1]) := by
    rw [evalOp_mul, hexp, hc]
    simp only [Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    ext idx
    obtain ⟨j, d, u⟩ := idx
    simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.expandDim_data,
      TileShape.dropInsertedIndex_succ, TileShape.dropInsertedIndex_nil,
      TileShape.dropInsertedIndex_zero_cons, NumericDType.mul, NumericDType.int_mul,
      Broadcast.leftIndex, Broadcast.rightIndex]
  have hoff : @evalOp .nat [BLOCK_N, 1]
        (Op.mul .int Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .int [BLOCK_N] "v_index"))
            (Op.constNat svbs).castNatToInt).castIntToNat s
      = some (⟨fun idx : TileIndex [BLOCK_N, 1] => (vIdxFn idx.1 * svbs).toNat⟩ : Tile .nat [BLOCK_N, 1]) := by
    erw [sr_evalOp_castIntToNat, hmulint]
    rfl
  rw [show evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consR.consL (Op.ref .ptr [1, BLOCK_DMODEL] "v_ptrs")
            (Op.mul .int Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .int [BLOCK_N] "v_index"))
                (Op.constNat svbs).castNatToInt).castIntToNat))
        MaskOpt.none) s = _ from rfl]
  simp only [evalOp, hvp, hoff, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, d, u⟩ := idx
  simp only [Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    BlockState.readMemValue_real]
  refine congrArg some (congrArg (s.readMem V) ?_)
  omega

set_option maxHeartbeats 1600000 in
/-- General `n_e_max` recipe over `Fin BLOCK_N` lanes (needs `0 < BLOCK_N` for the reduce). -/
theorem sr_nemax_evalG (BLOCK_N : Nat) (hN : 0 < BLOCK_N) (s : BlockState)
    (qkFn : Fin BLOCK_N → WithBot ℝ) (em : WithBot ℝ)
    (hqk : s.regs .real [BLOCK_N] "qk" = some (⟨fun idx : TileIndex [BLOCK_N] => qkFn idx.1⟩ : Tile .real [BLOCK_N]))
    (hem : s.regs .real [] "e_max" = some (Tile.scalar em))
    (hne : (Finset.univ : Finset (Fin BLOCK_N)).Nonempty) :
    evalOp ((Op.gt .real Broadcast.nil
            (Op.reduceMax ⟨0, by simp⟩ Bool.false (Op.ref .real [BLOCK_N] "qk"))
            (Op.ref .real [] "e_max")).where
        (Op.reduceMax ⟨0, by simp⟩ Bool.false (Op.ref .real [BLOCK_N] "qk"))
        (Op.ref .real [] "e_max")) s
      = some (Tile.scalar (em ⊔ Finset.univ.sup' hne (fun k : Fin BLOCK_N => qkFn k))) := by
  have hrm : Tile.reduceMaxDrop (⟨0, by simp⟩ : Fin [BLOCK_N].length)
        (⟨fun idx : TileIndex [BLOCK_N] => qkFn idx.1⟩ : Tile .real [BLOCK_N])
      = some (Tile.scalar (Finset.univ.sup' hne (fun k : Fin BLOCK_N => qkFn k))) := by
    unfold Tile.reduceMaxDrop
    rw [dif_pos (show 0 < TileShape.axisDim [BLOCK_N] (⟨0, by simp⟩ : Fin [BLOCK_N].length) from hN)]
    refine congrArg some ?_
    ext idx
    simp only [Tile.scalar]
    rfl
  have hrmaxN :
        evalOp (Op.reduceMax (⟨0, by simp⟩ : Fin [BLOCK_N].length) Bool.false (Op.ref .real [BLOCK_N] "qk")) s
      = some (Tile.scalar (Finset.univ.sup' hne (fun k : Fin BLOCK_N => qkFn k))) := by
    rw [evalOp_reduceMax]; simp only [evalOp_ref, hqk]; exact hrm
  rw [evalOp_where, evalOp_gt]
  erw [hrmaxN]
  rw [evalOp_ref, hem]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.select_data, Tile.cop_data, Tile.scalar, ComparableDType.gt]
  set rm := Finset.univ.sup' hne (fun k : Fin BLOCK_N => qkFn k) with hrmdef
  by_cases h : em < rm
  · rw [if_pos (by simpa using h), max_eq_right (le_of_lt h)]
  · rw [if_neg (by simpa using h), max_eq_left (not_lt.mp h)]

/-- General `p` recipe over `Fin BLOCK_N` lanes. -/
theorem sr_p_evalG (BLOCK_N : Nat) (s : BlockState) (qkFn : Fin BLOCK_N → WithBot ℝ) (nem : WithBot ℝ)
    (hqk : s.regs .real [BLOCK_N] "qk" = some (⟨fun idx : TileIndex [BLOCK_N] => qkFn idx.1⟩ : Tile .real [BLOCK_N]))
    (hnem : s.regs .real [] "n_e_max" = some (Tile.scalar nem)) :
    evalOp (Op.sub .real Broadcast.scalarR (Op.ref .real [BLOCK_N] "qk") (Op.ref .real [] "n_e_max")).exp s
      = some (⟨fun idx : TileIndex [BLOCK_N] => WithBot.realExp (WithBot.realSub (qkFn idx.1) nem)⟩
          : Tile .real [BLOCK_N]) := by
  rw [evalOp_exp]
  simp only [evalOp_sub, evalOp_ref, hqk, hnem, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.uop_data, Tile.bop_data, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.sub]

set_option maxHeartbeats 1600000 in
/-- General `e_sum` recipe over `Fin BLOCK_N` lanes. -/
theorem sr_esum_evalG (BLOCK_N : Nat) (s : BlockState) (l α : WithBot ℝ) (pFn : Fin BLOCK_N → WithBot ℝ)
    (hl : s.regs .real [] "e_sum" = some (Tile.scalar l))
    (hos : s.regs .real [] "old_scale" = some (Tile.scalar α))
    (hp : s.regs .real [BLOCK_N] "p" = some (⟨fun idx : TileIndex [BLOCK_N] => pFn idx.1⟩ : Tile .real [BLOCK_N])) :
    evalOp (Op.add .real Broadcast.nil
        (Op.mul .real Broadcast.nil (Op.ref .real [] "e_sum") (Op.ref .real [] "old_scale"))
        (Op.reduceSum ⟨0, by simp⟩ Bool.false (Op.ref .real [BLOCK_N] "p"))) s
      = some (Tile.scalar (WithBot.realAdd (WithBot.realMul l α)
          (∑ j : Fin BLOCK_N, pFn j))) := by
  have hleft : evalOp (Op.mul .real Broadcast.nil (Op.ref .real [] "e_sum") (Op.ref .real [] "old_scale")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil (Tile.scalar l) (Tile.scalar α)) := by
    rw [evalOp_mul, evalOp_ref, hl, evalOp_ref, hos]; rfl
  have hright : evalOp (Op.reduceSum ⟨0, by simp⟩ Bool.false (Op.ref .real [BLOCK_N] "p")) s
      = some (Tile.reduceSumDrop (⟨0, by simp⟩ : Fin [BLOCK_N].length)
          (⟨fun idx : TileIndex [BLOCK_N] => pFn idx.1⟩ : Tile .real [BLOCK_N])) := by
    rw [evalOp_reduceSum, evalOp_ref, hp]; rfl
  rw [evalOp_add, hleft]
  show Option.bind (evalOp (Op.reduceSum ⟨0, by simp⟩ Bool.false (Op.ref .real [BLOCK_N] "p")) s) _ = _
  rw [hright]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.reduceSumDrop_data, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul, TileShape.insertAxisIndex,
    TileShape.axisDim]
  rfl

set_option maxHeartbeats 1600000 in
/-- General `acc` recipe (`acc·old_scale + tl.sum(p[:,None]·v, 0)`, shape `[BLOCK_DMODEL]`). -/
theorem sr_acc_evalG (BLOCK_N BLOCK_DMODEL : Nat) (s : BlockState) (α : WithBot ℝ)
    (accFn : Fin BLOCK_DMODEL → WithBot ℝ) (pFn : Fin BLOCK_N → WithBot ℝ)
    (vFn : Fin BLOCK_N → Fin BLOCK_DMODEL → WithBot ℝ)
    (hacc : s.regs .real [BLOCK_DMODEL] "acc" = some (⟨fun idx : TileIndex [BLOCK_DMODEL] => accFn idx.1⟩ : Tile .real [BLOCK_DMODEL]))
    (hos : s.regs .real [] "old_scale" = some (Tile.scalar α))
    (hp : s.regs .real [BLOCK_N] "p" = some (⟨fun idx : TileIndex [BLOCK_N] => pFn idx.1⟩ : Tile .real [BLOCK_N]))
    (hv : s.regs .real [BLOCK_N, BLOCK_DMODEL] "v" =
      some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] => vFn idx.1 idx.2.1⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL])) :
    evalOp (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.scalarR (Op.ref .real [BLOCK_DMODEL] "acc") (Op.ref .real [] "old_scale"))
        (Op.reduceSum ⟨0, by simp⟩ Bool.false
          (Op.mul .real Broadcast.nil.consL.consSame
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_N] "p"))
            (Op.ref .real [BLOCK_N, BLOCK_DMODEL] "v")))) s
      = some (⟨fun idx : TileIndex [BLOCK_DMODEL] =>
          WithBot.realAdd (WithBot.realMul (accFn idx.1) α)
            (∑ j : Fin BLOCK_N, WithBot.realMul (pFn j) (vFn j idx.1))⟩ : Tile .real [BLOCK_DMODEL]) := by
  have hexp : @evalOp .real [BLOCK_N, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_N] "p")) s
      = some (Tile.expandDim ⟨1, by simp⟩
          (⟨fun idx : TileIndex [BLOCK_N] => pFn idx.1⟩ : Tile .real [BLOCK_N])) :=
    evalOp_expandDim_ref_of_regs .real [BLOCK_N] ⟨1, by simp⟩ "p" s _ hp
  have hmul : evalOp (Op.mul .real Broadcast.nil.consL.consSame
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_N] "p"))
        (Op.ref .real [BLOCK_N, BLOCK_DMODEL] "v")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consL.consSame
          (Tile.expandDim ⟨1, by simp⟩ (⟨fun idx : TileIndex [BLOCK_N] => pFn idx.1⟩ : Tile .real [BLOCK_N]))
          (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] => vFn idx.1 idx.2.1⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL])) := by
    rw [evalOp_mul, hexp, evalOp_ref, hv]; rfl
  have hleft : evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BLOCK_DMODEL] "acc") (Op.ref .real [] "old_scale")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (⟨fun idx : TileIndex [BLOCK_DMODEL] => accFn idx.1⟩ : Tile .real [BLOCK_DMODEL]) (Tile.scalar α)) := by
    rw [evalOp_mul, evalOp_ref, hacc, evalOp_ref, hos]; rfl
  have hright : evalOp (Op.reduceSum ⟨0, by simp⟩ Bool.false
        (Op.mul .real Broadcast.nil.consL.consSame
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_N] "p"))
          (Op.ref .real [BLOCK_N, BLOCK_DMODEL] "v"))) s
      = some (Tile.reduceSumDrop (⟨0, by simp⟩ : Fin [BLOCK_N, BLOCK_DMODEL].length)
          (Tile.bop NumericDType.real.mul Broadcast.nil.consL.consSame
            (Tile.expandDim ⟨1, by simp⟩ (⟨fun idx : TileIndex [BLOCK_N] => pFn idx.1⟩ : Tile .real [BLOCK_N]))
            (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] => vFn idx.1 idx.2.1⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL]))) := by
    rw [evalOp_reduceSum, hmul]; rfl
  rw [evalOp_add, hleft]
  show Option.bind (evalOp (Op.reduceSum ⟨0, by simp⟩ Bool.false
      (Op.mul .real Broadcast.nil.consL.consSame
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_N] "p"))
        (Op.ref .real [BLOCK_N, BLOCK_DMODEL] "v"))) s) _ = _
  rw [hright]
  refine congrArg some ?_
  ext idx
  obtain ⟨d, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.reduceSumDrop_data, Tile.expandDim_data,
    TileShape.dropInsertedIndex_succ, TileShape.dropInsertedIndex_nil,
    TileShape.dropInsertedIndex_zero_cons, TileShape.insertAxisIndex, TileShape.axisDim,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]
  rfl

/-! ### General stride/shape-parameterized lowered statement lists -/

/-- General lowered preLoop statements. -/
def srPreLoopG (V : RegionName) (BStartLoc BSeqLen : Region .nat)
    (max_input_len stride_vh stride_vd stride_b_loc_b stride_b_loc_s BLOCK_DMODEL BLOCK_N : Nat) :
    List Stmt :=
  [ Stmt.assign .nat [] "cur_batch" (Op.programId 0),
    Stmt.assign .nat [] "cur_head" (Op.programId 1),
    Stmt.assign .nat [] "cur_batch_seq_len"
      (Op.load .nat (MemAccess.region BSeqLen (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "cur_batch_start_loc"
      (Op.load .nat (MemAccess.region BStartLoc (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [BLOCK_N] "offs_n" (Op.arange BLOCK_N),
    Stmt.assign .nat [BLOCK_DMODEL] "offs_d" (Op.arange BLOCK_DMODEL),
    Stmt.assign .nat [1, BLOCK_DMODEL] "off_v"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_vh))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
          (Op.constNat stride_vd))),
    Stmt.assign .nat [] "off_b_loc"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat stride_b_loc_b))
        (Op.mul .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.constNat max_input_len) (Op.ref .nat [] "cur_batch_seq_len"))
          (Op.constNat stride_b_loc_s))),
    Stmt.assign .ptr [1, BLOCK_DMODEL] "v_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V) (Op.ref .nat [1, BLOCK_DMODEL] "off_v")),
    Stmt.assign .real [] "e_max" Op.negInf,
    Stmt.assign .real [] "e_sum" (Op.const 0.0),
    Stmt.assign .real [BLOCK_DMODEL] "acc" (Op.full [BLOCK_DMODEL] (Op.const 0)) ]

/-- General lowered loop-body statements. -/
def srLoopBodyG (Logics : RegionName) (BLoc : Region .int)
    (stride_logic_h stride_logic_bs stride_vbs stride_b_loc_s BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int) : List Stmt :=
  [ Stmt.assign .nat [] "start_n" (Op.ref .nat [] "start_n"),
    Stmt.assign .int [BLOCK_N] "v_index"
      (Op.load .int
        (MemAccess.region BLoc
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "off_b_loc")
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BLOCK_N] "offs_n"))
              (Op.constNat stride_b_loc_s))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BLOCK_N] "offs_n"))
            (Op.ref .nat [] "cur_batch_seq_len"))
          ((Op.constInt other_kv_index).broadcast [BLOCK_N]))),
    Stmt.assign .real [BLOCK_N] "qk"
      (Op.load .real
        (MemAccess.region Logics
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_logic_h))
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_start_loc")
                  (Op.ref .nat [] "start_n"))
                (Op.ref .nat [BLOCK_N] "offs_n"))
              (Op.constNat stride_logic_bs))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BLOCK_N] "offs_n"))
            (Op.ref .nat [] "cur_batch_seq_len"))
          (Op.negInf.broadcast [BLOCK_N]))),
    Stmt.assign .real [] "n_e_max"
      ((Op.gt .real Broadcast.nil
            (Op.reduceMax ⟨0, by simp⟩ Bool.false (Op.ref .real [BLOCK_N] "qk"))
            (Op.ref .real [] "e_max")).where
        (Op.reduceMax ⟨0, by simp⟩ Bool.false (Op.ref .real [BLOCK_N] "qk"))
        (Op.ref .real [] "e_max")),
    Stmt.assign .real [] "old_scale"
      (Op.sub .real Broadcast.nil (Op.ref .real [] "e_max") (Op.ref .real [] "n_e_max")).exp,
    Stmt.assign .real [BLOCK_N] "p"
      (Op.sub .real Broadcast.scalarR (Op.ref .real [BLOCK_N] "qk") (Op.ref .real [] "n_e_max")).exp,
    Stmt.assign .real [] "e_sum"
      (Op.add .real Broadcast.nil
        (Op.mul .real Broadcast.nil (Op.ref .real [] "e_sum") (Op.ref .real [] "old_scale"))
        (Op.reduceSum ⟨0, by simp⟩ Bool.false (Op.ref .real [BLOCK_N] "p"))),
    Stmt.assign .real [BLOCK_N, BLOCK_DMODEL] "v"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consR.consL (Op.ref .ptr [1, BLOCK_DMODEL] "v_ptrs")
            (Op.mul .int Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .int [BLOCK_N] "v_index"))
                (Op.constNat stride_vbs).castNatToInt).castIntToNat))
        MaskOpt.none),
    Stmt.assign .real [BLOCK_DMODEL] "acc"
      (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.scalarR (Op.ref .real [BLOCK_DMODEL] "acc") (Op.ref .real [] "old_scale"))
        (Op.reduceSum ⟨0, by simp⟩ Bool.false
          (Op.mul .real Broadcast.nil.consL.consSame
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_N] "p"))
            (Op.ref .real [BLOCK_N, BLOCK_DMODEL] "v")))),
    Stmt.assign .real [] "e_max" (Op.ref .real [] "n_e_max") ]

/-- General lowered postLoop statements. -/
def srPostLoopG (Out : RegionName)
    (stride_obs stride_oh stride_od BLOCK_DMODEL : Nat) : List Stmt :=
  [ Stmt.assign .real [BLOCK_DMODEL] "acc"
      (Op.div .real Broadcast.scalarR (Op.ref .real [BLOCK_DMODEL] "acc") (Op.ref .real [] "e_sum")),
    Stmt.assign .nat [BLOCK_DMODEL] "off_o"
      (Op.add .nat Broadcast.scalarL
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat stride_obs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_oh)))
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_DMODEL] "offs_d") (Op.constNat stride_od))),
    Stmt.assign .ptr [BLOCK_DMODEL] "out_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BLOCK_DMODEL] "off_o")),
    Stmt.store .real [BLOCK_DMODEL] (MemAccess.ptr (Op.ref .ptr [BLOCK_DMODEL] "out_ptrs"))
      (Op.ref .real [BLOCK_DMODEL] "acc") MaskOpt.none ]

/-- General body split: the lowered general surface body is preLoop ++ forRangeDyn ++ postLoop. -/
theorem srBody_splitG
    (Logics V Out : RegionName) (BLoc : Region .int) (BStartLoc BSeqLen : Region .nat)
    (max_input_len stride_logic_h stride_logic_bs stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od stride_b_loc_b stride_b_loc_s BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int) :
    (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      max_input_len stride_logic_h stride_logic_bs stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od stride_b_loc_b stride_b_loc_s
      BLOCK_DMODEL BLOCK_N other_kv_index).toAlgKernel.body
      = srPreLoopG V BStartLoc BSeqLen max_input_len stride_vh stride_vd
          stride_b_loc_b stride_b_loc_s BLOCK_DMODEL BLOCK_N
        ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0) (Op.ref .nat [] "cur_batch_seq_len")
              (Op.constNat BLOCK_N)
              (srLoopBodyG Logics BLoc stride_logic_h stride_logic_bs stride_vbs
                stride_b_loc_s BLOCK_DMODEL BLOCK_N other_kv_index)
            :: srPostLoopG Out stride_obs stride_oh stride_od BLOCK_DMODEL) := by
  rfl

/-! ### General loop invariant + output offset -/

/-- General output offset `cur_batch·stride_obs + cur_head·stride_oh + d·stride_od`. -/
def outOffsetG (s : BlockState) (stride_obs stride_oh stride_od : Nat) {BLOCK_DMODEL : Nat}
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_obs + s.pids 1 * stride_oh + i.val * stride_od

/-- General loop invariant after `c` `BLOCK_N`-blocks. -/
noncomputable def srInvariantG
    (Logics V : RegionName) (BLoc : Region .int) (BStartLoc BSeqLen : RegionName)
    (max_input_len stride_logic_h stride_logic_bs stride_vbs stride_vh stride_vd
      stride_b_loc_b stride_b_loc_s BLOCK_DMODEL BLOCK_N : Nat)
    (s0 : BlockState) (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids
  ∧ s.mem = s0.mem
  ∧ (∀ rg o, s.undef rg o = 0)
  ∧ s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 0))
  ∧ s.regs .nat [] "cur_head" = some (Tile.scalar (s0.pids 1))
  ∧ s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (srSeqLen s0 BSeqLen))
  ∧ s.regs .nat [] "cur_batch_start_loc" = some (Tile.scalar (srStartLoc s0 BStartLoc))
  ∧ s.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
  ∧ s.regs .nat [BLOCK_DMODEL] "offs_d" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))
  ∧ s.regs .nat [] "off_b_loc" = some (Tile.scalar (srOffBLocG s0 BSeqLen max_input_len stride_b_loc_b stride_b_loc_s))
  ∧ s.regs .ptr [1, BLOCK_DMODEL] "v_ptrs" =
      some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] => (V, s0.pids 1 * stride_vh + idx.2.1.val * stride_vd)⟩ : Tile .ptr [1, BLOCK_DMODEL])
  ∧ (∀ hD : 0 < BLOCK_DMODEL, s.regs .real [] "e_max" =
      some (Tile.scalar (srRunningMax (srQkFG s0 Logics BStartLoc BSeqLen stride_logic_h stride_logic_bs)
        (srVFG s0 V BLoc BSeqLen max_input_len stride_b_loc_b stride_b_loc_s stride_vbs stride_vh stride_vd BLOCK_DMODEL)
        (c * BLOCK_N) ⟨0, hD⟩)))
  ∧ (∀ hD : 0 < BLOCK_DMODEL, s.regs .real [] "e_sum" =
      some (Tile.scalar ((srStateBot (srQkFG s0 Logics BStartLoc BSeqLen stride_logic_h stride_logic_bs)
        (srVFG s0 V BLoc BSeqLen max_input_len stride_b_loc_b stride_b_loc_s stride_vbs stride_vh stride_vd BLOCK_DMODEL)
        (c * BLOCK_N) ⟨0, hD⟩).2.1 : WithBot ℝ)))
  ∧ s.regs .real [BLOCK_DMODEL] "acc" =
      some (⟨fun idx : TileIndex [BLOCK_DMODEL] =>
        ((srStateBot (srQkFG s0 Logics BStartLoc BSeqLen stride_logic_h stride_logic_bs)
          (srVFG s0 V BLoc BSeqLen max_input_len stride_b_loc_b stride_b_loc_s stride_vbs stride_vh stride_vd BLOCK_DMODEL)
          (c * BLOCK_N) idx.1).2.2 : WithBot ℝ)⟩ : Tile .real [BLOCK_DMODEL])

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **General step lemma**: the loop body advances `srInvariantG` by one key block. -/
theorem sr_attn_stepG
    (Logics V : RegionName) (BLoc : Region .int) (BStartLoc BSeqLen : RegionName)
    (mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N : Nat)
    (hD : 0 < BLOCK_DMODEL) (hN : 0 < BLOCK_N)
    (other_kv_index : Int)
    (s0 : BlockState) (i : Nat) (s : BlockState)
    (hilt : i < srSeqLen s0 BSeqLen) (hhimod : srSeqLen s0 BSeqLen % BLOCK_N = 0)
    (hinv : srInvariantG Logics V BLoc BStartLoc BSeqLen mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N s0 (i / BLOCK_N) s)
    (hi : i = (i / BLOCK_N) * BLOCK_N) :
    ∃ s', stepStmts (srLoopBodyG Logics BLoc slh slb svbs ss BLOCK_DMODEL BLOCK_N other_kv_index)
        (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ srInvariantG Logics V BLoc BStartLoc BSeqLen mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N s0 (i / BLOCK_N + 1) s' := by
  set S := srSeqLen s0 BSeqLen with hSdef
  set c := i / BLOCK_N with hc_def
  set qk := srQkFG s0 Logics BStartLoc BSeqLen slh slb with hqkdef
  set vF := srVFG s0 V BLoc BSeqLen mil sb ss svbs svh svd BLOCK_DMODEL with hvFdef
  have hwin : (c + 1) * BLOCK_N ≤ S := by
    have hmul : c * BLOCK_N < S := by rw [← hi]; exact hilt
    have hSeq : (S / BLOCK_N) * BLOCK_N = S := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hhimod)
    have hclt : c < S / BLOCK_N := by
      apply Nat.lt_of_mul_lt_mul_right (a := BLOCK_N); rw [hSeq]; exact hmul
    calc (c + 1) * BLOCK_N ≤ (S / BLOCK_N) * BLOCK_N := Nat.mul_le_mul_right _ hclt
      _ = S := hSeq
  haveI hNe : Nonempty (Fin BLOCK_N) := ⟨⟨0, hN⟩⟩
  have hne : (Finset.univ : Finset (Fin BLOCK_N)).Nonempty := Finset.univ_nonempty
  simp only [srInvariantG, ← hqkdef, ← hvFdef, ← hSdef] at hinv
  obtain ⟨hpids, hmem, hundef, hcb, hch, hseq, hsl, hn, hd, hoff, hvp, hemaxA, hesumA, hacc⟩ := hinv
  have hemax := hemaxA hD
  have hesum := hesumA hD
  -- the post-setReg start_n state
  set s1 := s.setReg "start_n" .nat [] (Tile.scalar i) with hs1
  have hs1seq : s1.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar S) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hseq
  have hs1sl : s1.regs .nat [] "cur_batch_start_loc" = some (Tile.scalar (srStartLoc s0 BStartLoc)) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsl
  have hs1ch : s1.regs .nat [] "cur_head" = some (Tile.scalar (s0.pids 1)) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hch
  have hs1n : s1.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hn
  have hs1off : s1.regs .nat [] "off_b_loc" = some (Tile.scalar (srOffBLocG s0 BSeqLen mil sb ss)) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoff
  have hs1sn : s1.regs .nat [] "start_n" = some (Tile.scalar i) := by
    rw [hs1, BlockState.setReg_same]
  have hs1cb : s1.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 0)) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hcb
  have hs1d : s1.regs .nat [BLOCK_DMODEL] "offs_d" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val)) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hd
  have hs1emax : s1.regs .real [] "e_max"
      = some (Tile.scalar (srRunningMax qk vF (c * BLOCK_N) (⟨0, hD⟩ : Fin BLOCK_DMODEL))) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hemax
  have hs1esum : s1.regs .real [] "e_sum"
      = some (Tile.scalar ((srStateBot qk vF (c * BLOCK_N) (⟨0, hD⟩ : Fin BLOCK_DMODEL)).2.1 : WithBot ℝ)) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hesum
  have hs1acc : s1.regs .real [BLOCK_DMODEL] "acc"
      = some (⟨fun idx : TileIndex [BLOCK_DMODEL] =>
          ((srStateBot qk vF (c * BLOCK_N) idx.1).2.2 : WithBot ℝ)⟩ : Tile .real [BLOCK_DMODEL]) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hacc
  have hs1vp : s1.regs .ptr [1, BLOCK_DMODEL] "v_ptrs"
      = some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] => (V, s0.pids 1 * svh + idx.2.1.val * svd)⟩ : Tile .ptr [1, BLOCK_DMODEL]) := by
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hvp
  have hs1pids : s1.pids = s0.pids := by rw [hs1, BlockState.setReg_pids]; exact hpids
  unfold srLoopBodyG
  -- stmt 0: start_n = ref start_n
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_n") s1 = some (Tile.scalar i) from by
      rw [evalOp_ref]; exact hs1sn))]
  set s2 := s1.setReg "start_n" .nat [] (Tile.scalar i) with hs2
  have hs2mem : s2.mem = s0.mem := by
    funext rg o; rw [hs2, BlockState.setReg_mem, hs1, BlockState.setReg_mem]
    exact congrFun (congrFun hmem rg) o
  have hs2pids : s2.pids = s0.pids := by rw [hs2, BlockState.setReg_pids]; exact hs1pids
  have hs2seqEq : srSeqLen s2 BSeqLen = S := by rw [hSdef]; exact srSeqLen_eq_of_mem_pids s2 s0 BSeqLen hs2mem hs2pids
  have hs2offEq : srOffBLocG s2 BSeqLen mil sb ss = srOffBLocG s0 BSeqLen mil sb ss := by
    simp only [srOffBLocG, hs2pids, srSeqLen_eq_of_mem_pids s2 s0 BSeqLen hs2mem hs2pids]
  have e2 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "start_n" → s1.regs dt sh nm = some t → s2.regs dt sh nm = some t := by
    intro dt sh nm t hne' h; rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne']; exact h
  have hs2sn : s2.regs .nat [] "start_n" = some (Tile.scalar i) := by rw [hs2, BlockState.setReg_same]
  -- stmt 1: v_index = masked gather
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sr_vindex_gather_evalG BLOCK_N s2 BLoc BSeqLen i mil sb ss other_kv_index
      (by rw [hs2offEq]; exact e2 (by decide) hs1off) hs2sn (e2 (by decide) hs1n)
      (by rw [hs2seqEq]; exact e2 (by decide) hs1seq)))]
  set s3 := s2.setReg "v_index" .int [BLOCK_N]
    (⟨fun idx : TileIndex [BLOCK_N] => if i + idx.1.val < srSeqLen s2 BSeqLen
        then srVIndexG s2 BLoc BSeqLen mil sb ss (i + idx.1.val) else other_kv_index⟩ : Tile .int [BLOCK_N]) with hs3
  have hs3mem : s3.mem = s0.mem := by funext rg o; rw [hs3, BlockState.setReg_mem]; exact congrFun (congrFun hs2mem rg) o
  have hs3pids : s3.pids = s0.pids := by rw [hs3, BlockState.setReg_pids]; exact hs2pids
  have e3 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "v_index" → s2.regs dt sh nm = some t → s3.regs dt sh nm = some t := by
    intro dt sh nm t hne' h; rw [hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne']; exact h
  have hs3seqEq : srSeqLen s3 BSeqLen = S := by rw [hSdef]; exact srSeqLen_eq_of_mem_pids s3 s0 BSeqLen hs3mem hs3pids
  -- stmt 2: qk = masked load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sr_qk_load_evalG BLOCK_N s3 Logics BStartLoc BSeqLen i slh slb
      (by rw [show srStartLoc s3 BStartLoc = srStartLoc s0 BStartLoc from
            srStartLoc_eq_of_mem_pids s3 s0 BStartLoc hs3mem hs3pids]
          exact e3 (by decide) (e2 (by decide) hs1sl))
      (by rw [show s3.pids 1 = s0.pids 1 from by rw [hs3pids]]
          exact e3 (by decide) (e2 (by decide) hs1ch))
      (e3 (by decide) hs2sn)
      (e3 (by decide) (e2 (by decide) hs1n))
      (by rw [hs3seqEq]; exact e3 (by decide) (e2 (by decide) hs1seq))))]
  rw [hs3seqEq]
  set qkFn : Fin BLOCK_N → WithBot ℝ := fun jL =>
    if i + jL.val < S then some (srQkG s0 Logics BStartLoc slh slb (i + jL.val)) else (⊥ : WithBot ℝ)
    with hqkFn
  rw [show (⟨fun idx : TileIndex [BLOCK_N] =>
        if i + idx.1.val < S then some (srQkG s3 Logics BStartLoc slh slb (i + idx.1.val)) else (⊥ : WithBot ℝ)⟩
        : Tile .real [BLOCK_N])
      = (⟨fun idx : TileIndex [BLOCK_N] => qkFn idx.1⟩ : Tile .real [BLOCK_N]) from by
    refine Tile.ext (fun idx => ?_)
    simp only [hqkFn, srQkG_eq_of_mem_pids s3 s0 Logics BStartLoc slh slb hs3mem hs3pids]]
  set s4 := s3.setReg "qk" .real [BLOCK_N] (⟨fun idx : TileIndex [BLOCK_N] => qkFn idx.1⟩ : Tile .real [BLOCK_N]) with hs4
  have e4 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s3.regs dt sh nm = some t → s4.regs dt sh nm = some t := by
    intro dt sh nm t hne' h; rw [hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne']; exact h
  have hs4qk : s4.regs .real [BLOCK_N] "qk" = some (⟨fun idx : TileIndex [BLOCK_N] => qkFn idx.1⟩ : Tile .real [BLOCK_N]) := by
    rw [hs4, BlockState.setReg_same]
  set emax0 := srRunningMax qk vF (c * BLOCK_N) (⟨0, hD⟩ : Fin BLOCK_DMODEL) with hemax0
  have hs4emax : s4.regs .real [] "e_max" = some (Tile.scalar emax0) := by
    rw [hemax0]; exact e4 (by decide) (e3 (by decide) (e2 (by decide) hs1emax))
  -- stmt 3: n_e_max
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sr_nemax_evalG BLOCK_N hN s4 qkFn emax0 hs4qk hs4emax hne))]
  set nemaxVal := emax0 ⊔ Finset.univ.sup' hne (fun k : Fin BLOCK_N => qkFn k) with hnemaxVal
  set s5 := s4.setReg "n_e_max" .real [] (Tile.scalar nemaxVal) with hs5
  have e5 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "n_e_max" → s4.regs dt sh nm = some t → s5.regs dt sh nm = some t := by
    intro dt sh nm t hne' h; rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne']; exact h
  have hs5nem : s5.regs .real [] "n_e_max" = some (Tile.scalar nemaxVal) := by rw [hs5, BlockState.setReg_same]
  have hs5emax : s5.regs .real [] "e_max" = some (Tile.scalar emax0) := e5 (by decide) hs4emax
  -- stmt 4: old_scale
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sr_oldscale_eval s5 emax0 nemaxVal hs5emax hs5nem))]
  set αVal := WithBot.realExp (WithBot.realSub emax0 nemaxVal) with hαVal
  set s6 := s5.setReg "old_scale" .real [] (Tile.scalar αVal) with hs6
  have e6 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "old_scale" → s5.regs dt sh nm = some t → s6.regs dt sh nm = some t := by
    intro dt sh nm t hne' h; rw [hs6, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne']; exact h
  have hs6os : s6.regs .real [] "old_scale" = some (Tile.scalar αVal) := by rw [hs6, BlockState.setReg_same]
  have hs6qk : s6.regs .real [BLOCK_N] "qk" = some (⟨fun idx : TileIndex [BLOCK_N] => qkFn idx.1⟩ : Tile .real [BLOCK_N]) :=
    e6 (by decide) (e5 (by decide) hs4qk)
  have hs6nem : s6.regs .real [] "n_e_max" = some (Tile.scalar nemaxVal) := e6 (by decide) hs5nem
  -- stmt 5: p
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sr_p_evalG BLOCK_N s6 qkFn nemaxVal hs6qk hs6nem))]
  set pFn : Fin BLOCK_N → WithBot ℝ := fun jL => WithBot.realExp (WithBot.realSub (qkFn jL) nemaxVal) with hpFn
  set s7 := s6.setReg "p" .real [BLOCK_N] (⟨fun idx : TileIndex [BLOCK_N] => pFn idx.1⟩ : Tile .real [BLOCK_N]) with hs7
  have e7 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p" → s6.regs dt sh nm = some t → s7.regs dt sh nm = some t := by
    intro dt sh nm t hne' h; rw [hs7, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne']; exact h
  have hs7p : s7.regs .real [BLOCK_N] "p" = some (⟨fun idx : TileIndex [BLOCK_N] => pFn idx.1⟩ : Tile .real [BLOCK_N]) := by
    rw [hs7, BlockState.setReg_same]
  have hs7os : s7.regs .real [] "old_scale" = some (Tile.scalar αVal) := e7 (by decide) hs6os
  set esum0 := ((srStateBot qk vF (c * BLOCK_N) (⟨0, hD⟩ : Fin BLOCK_DMODEL)).2.1 : WithBot ℝ) with hesum0
  have hs7esum : s7.regs .real [] "e_sum" = some (Tile.scalar esum0) :=
    e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) hs1esum)))))
  -- stmt 6: e_sum
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sr_esum_evalG BLOCK_N s7 esum0 αVal pFn hs7esum hs7os hs7p))]
  set esumNew := WithBot.realAdd (WithBot.realMul esum0 αVal) (∑ j : Fin BLOCK_N, pFn j) with hesumNew
  set s8 := s7.setReg "e_sum" .real [] (Tile.scalar esumNew) with hs8
  have e8 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "e_sum" → s7.regs dt sh nm = some t → s8.regs dt sh nm = some t := by
    intro dt sh nm t hne' h; rw [hs8, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne']; exact h
  set vIdxFn : Fin BLOCK_N → Int := fun jL =>
    if i + jL.val < srSeqLen s2 BSeqLen then srVIndexG s2 BLoc BSeqLen mil sb ss (i + jL.val) else other_kv_index
    with hvIdxFn
  have hs8vp : s8.regs .ptr [1, BLOCK_DMODEL] "v_ptrs"
      = some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] => (V, s0.pids 1 * svh + idx.2.1.val * svd)⟩ : Tile .ptr [1, BLOCK_DMODEL]) :=
    e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide)
      (e3 (by decide) (e2 (by decide) hs1vp))))))
  have hs8vi : s8.regs .int [BLOCK_N] "v_index" = some (⟨fun idx : TileIndex [BLOCK_N] => vIdxFn idx.1⟩ : Tile .int [BLOCK_N]) :=
    e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide)
      (show s3.regs .int [BLOCK_N] "v_index" = _ from by rw [hs3, BlockState.setReg_same])))))
  have hs8pids : s8.pids 1 = s0.pids 1 := by
    rw [hs8, BlockState.setReg_pids, hs7, BlockState.setReg_pids, hs6, BlockState.setReg_pids,
      hs5, BlockState.setReg_pids, hs4, BlockState.setReg_pids, hs3pids]
  -- stmt 7: v gather
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consR.consL (Op.ref .ptr [1, BLOCK_DMODEL] "v_ptrs")
            (Op.mul .int Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .int [BLOCK_N] "v_index"))
                (Op.constNat svbs).castNatToInt).castIntToNat)) MaskOpt.none) s8
        = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
            some (s8.readMem V ((vIdxFn idx.1 * svbs).toNat + (s0.pids 1 * svh + idx.2.1.val * svd)))⟩
            : Tile .real [BLOCK_N, BLOCK_DMODEL]) from
      sr_v_gather_evalG BLOCK_N BLOCK_DMODEL s8 V svbs (fun d => s0.pids 1 * svh + d.val * svd) vIdxFn hs8vp hs8vi))]
  set vTileFn : Fin BLOCK_N → Fin BLOCK_DMODEL → WithBot ℝ := fun jL d =>
    some (s8.readMem V ((vIdxFn jL * svbs).toNat + (s0.pids 1 * svh + d.val * svd))) with hvTileFn
  set s9 := s8.setReg "v" .real [BLOCK_N, BLOCK_DMODEL]
    (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] => vTileFn idx.1 idx.2.1⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL]) with hs9
  have e9 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "v" → s8.regs dt sh nm = some t → s9.regs dt sh nm = some t := by
    intro dt sh nm t hne' h; rw [hs9, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne']; exact h
  have hs9v : s9.regs .real [BLOCK_N, BLOCK_DMODEL] "v" = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] => vTileFn idx.1 idx.2.1⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL]) := by
    rw [hs9, BlockState.setReg_same]
  have hs9os : s9.regs .real [] "old_scale" = some (Tile.scalar αVal) := e9 (by decide) (e8 (by decide) hs7os)
  have hs9p : s9.regs .real [BLOCK_N] "p" = some (⟨fun idx : TileIndex [BLOCK_N] => pFn idx.1⟩ : Tile .real [BLOCK_N]) :=
    e9 (by decide) (e8 (by decide) hs7p)
  set accFn : Fin BLOCK_DMODEL → WithBot ℝ := fun d => ((srStateBot qk vF (c * BLOCK_N) d).2.2 : WithBot ℝ) with haccFn
  have hs9acc : s9.regs .real [BLOCK_DMODEL] "acc" = some (⟨fun idx : TileIndex [BLOCK_DMODEL] => accFn idx.1⟩ : Tile .real [BLOCK_DMODEL]) :=
    e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide)
      (e4 (by decide) (e3 (by decide) (e2 (by decide) hs1acc)))))))
  -- stmt 8: acc
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sr_acc_evalG BLOCK_N BLOCK_DMODEL s9 αVal accFn pFn vTileFn hs9acc hs9os hs9p hs9v))]
  set accNew : Fin BLOCK_DMODEL → WithBot ℝ := fun d =>
    WithBot.realAdd (WithBot.realMul (accFn d) αVal) (∑ j : Fin BLOCK_N, WithBot.realMul (pFn j) (vTileFn j d))
    with haccNew
  set s10 := s9.setReg "acc" .real [BLOCK_DMODEL] (⟨fun idx : TileIndex [BLOCK_DMODEL] => accNew idx.1⟩ : Tile .real [BLOCK_DMODEL]) with hs10
  have e10 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc" → s9.regs dt sh nm = some t → s10.regs dt sh nm = some t := by
    intro dt sh nm t hne' h; rw [hs10, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne']; exact h
  have hs10nem : s10.regs .real [] "n_e_max" = some (Tile.scalar nemaxVal) :=
    e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) hs5nem))))
  -- stmt 9: e_max = n_e_max
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [] "n_e_max") s10 = some (Tile.scalar nemaxVal) from by
      rw [evalOp_ref]; exact hs10nem))]
  rw [stepStmts.nil]
  set s11 := s10.setReg "e_max" .real [] (Tile.scalar nemaxVal) with hs11
  refine ⟨s11, rfl, ?_⟩
  have e11 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "e_max" → s10.regs dt sh nm = some t → s11.regs dt sh nm = some t := by
    intro dt sh nm t hne' h; rw [hs11, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne']; exact h
  have hqkFnBridge : qkFn = fun jL : Fin BLOCK_N =>
      if c * BLOCK_N + jL.val < S then ((qk ⟨c * BLOCK_N + jL.val, by
        have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ : ℝ) : WithBot ℝ)
      else (⊥ : WithBot ℝ) := by
    funext jL; simp only [hqkFn]
    by_cases hj : i + jL.val < S
    · rw [if_pos hj, if_pos (by rw [hi] at hj; exact hj)]
      rw [show i + jL.val = c * BLOCK_N + jL.val from by rw [hi]]
      rfl
    · rw [if_neg hj, if_neg (by rw [hi] at hj; exact hj)]
  have hnemaxBridge : nemaxVal = srRunningMax qk vF ((c + 1) * BLOCK_N) (⟨0, hD⟩ : Fin BLOCK_DMODEL) := by
    rw [hnemaxVal, hemax0, hqkFnBridge, ← sr_nemax_reg_eqG hD qk vF BLOCK_N c (⟨0, hD⟩ : Fin BLOCK_DMODEL) hwin hN hne]
  have hpFnBridge : ∀ jL : Fin BLOCK_N, pFn jL = WithBot.realExp (WithBot.realSub
        (if c * BLOCK_N + jL.val < S then ((qk ⟨c * BLOCK_N + jL.val, by
          have := jL.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ : ℝ) : WithBot ℝ)
         else (⊥ : WithBot ℝ))
        (srRunningMax qk vF ((c + 1) * BLOCK_N) (⟨0, hD⟩ : Fin BLOCK_DMODEL))) := by
    intro jL; rw [hpFn, hqkFnBridge, hnemaxBridge]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hs11, BlockState.setReg_pids, hs10, BlockState.setReg_pids, hs9, BlockState.setReg_pids,
      hs8, BlockState.setReg_pids, hs7, BlockState.setReg_pids, hs6, BlockState.setReg_pids,
      hs5, BlockState.setReg_pids, hs4, BlockState.setReg_pids, hs3pids]
  · funext rg o
    rw [hs11, BlockState.setReg_mem, hs10, BlockState.setReg_mem, hs9, BlockState.setReg_mem,
      hs8, BlockState.setReg_mem, hs7, BlockState.setReg_mem, hs6, BlockState.setReg_mem,
      hs5, BlockState.setReg_mem, hs4, BlockState.setReg_mem]
    exact congrFun (congrFun hs3mem rg) o
  · intro rg o
    rw [hs11, BlockState.setReg_undef, hs10, BlockState.setReg_undef, hs9, BlockState.setReg_undef,
      hs8, BlockState.setReg_undef, hs7, BlockState.setReg_undef, hs6, BlockState.setReg_undef,
      hs5, BlockState.setReg_undef, hs4, BlockState.setReg_undef, hs3, BlockState.setReg_undef,
      hs2, BlockState.setReg_undef, hs1, BlockState.setReg_undef]
    exact hundef rg o
  · exact e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide)
      (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) hs1cb)))))))))
  · exact e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide)
      (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) hs1ch)))))))))
  · exact e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide)
      (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) hs1seq)))))))))
  · exact e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide)
      (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) hs1sl)))))))))
  · exact e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide)
      (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) hs1n)))))))))
  · exact e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide)
      (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) hs1d)))))))))
  · exact e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide)
      (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) hs1off)))))))))
  · exact e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide)
      (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) hs1vp)))))))))
  · -- e_max
    intro hD'
    rw [show s11.regs .real [] "e_max" = some (Tile.scalar nemaxVal) from by rw [hs11, BlockState.setReg_same]]
    rw [hnemaxBridge]
  · -- e_sum
    intro hD'
    rw [show s11.regs .real [] "e_sum" = some (Tile.scalar esumNew) from by
      rw [hs11, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hs10, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hs9, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hs8, BlockState.setReg_same]]
    refine congrArg some (congrArg Tile.scalar ?_)
    rw [hesumNew, hesum0, hαVal, hnemaxBridge, hemax0]
    rw [show (∑ j : Fin BLOCK_N, pFn j) = (∑ j : Fin BLOCK_N, WithBot.realExp (WithBot.realSub
        (if c * BLOCK_N + j.val < S then ((qk ⟨c * BLOCK_N + j.val, by
          have := j.isLt; have : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at this; omega⟩ : ℝ) : WithBot ℝ)
         else (⊥ : WithBot ℝ)) (srRunningMax qk vF ((c + 1) * BLOCK_N) (⟨0, hD⟩ : Fin BLOCK_DMODEL)))) from by
      apply Finset.sum_congr rfl; intro j _; rw [hpFnBridge j]]
    exact sr_esum_reg_eqG hD qk vF BLOCK_N c (⟨0, hD⟩ : Fin BLOCK_DMODEL) hwin hN
  · -- acc
    rw [show s11.regs .real [BLOCK_DMODEL] "acc" = some (⟨fun idx : TileIndex [BLOCK_DMODEL] => accNew idx.1⟩ : Tile .real [BLOCK_DMODEL]) from by
      rw [hs11, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [hs10, BlockState.setReg_same]]
    refine congrArg some (Tile.ext (fun idx => ?_))
    obtain ⟨d, u⟩ := idx
    show accNew d = ((srStateBot qk vF ((c + 1) * BLOCK_N) d).2.2 : WithBot ℝ)
    simp only [haccNew, haccFn]
    rw [hαVal, hnemaxBridge, hemax0]
    have hrawV : ∀ jL : Fin BLOCK_N, (hj : c * BLOCK_N + jL.val < S) →
        s8.readMem V ((vIdxFn jL * svbs).toNat + (s0.pids 1 * svh + d.val * svd))
          = vF ⟨c * BLOCK_N + jL.val, hj⟩ d := by
      intro jL hj
      rw [hvFdef]
      show s8.readMem V _ = srVG s0 V BLoc BSeqLen mil sb ss svbs svh svd (c * BLOCK_N + jL.val) d.val
      have hs8mem : s8.mem = s0.mem := by
        funext rg o; rw [hs8, BlockState.setReg_mem, hs7, BlockState.setReg_mem, hs6, BlockState.setReg_mem,
          hs5, BlockState.setReg_mem, hs4, BlockState.setReg_mem]; exact congrFun (congrFun hs3mem rg) o
      have hreadeq : s8.readMem V ((vIdxFn jL * svbs).toNat + (s0.pids 1 * svh + d.val * svd))
          = s0.readMem V ((vIdxFn jL * svbs).toNat + s0.pids 1 * svh + d.val * svd) := by
        rw [show (vIdxFn jL * svbs).toNat + (s0.pids 1 * svh + d.val * svd)
              = (vIdxFn jL * svbs).toNat + s0.pids 1 * svh + d.val * svd from by ac_rfl]
        simp only [BlockState.readMem, hs8mem]
      rw [hreadeq, srVG]
      simp only [hvIdxFn]
      rw [if_pos (show i + jL.val < srSeqLen s2 BSeqLen from by rw [hs2seqEq, hi]; exact hj),
        srVIndexG_eq_of_mem_pids s2 s0 BLoc BSeqLen mil sb ss hs2mem hs2pids,
        show i + jL.val = c * BLOCK_N + jL.val from by rw [hi]]
    have hsumacc : (∑ j : Fin BLOCK_N, WithBot.realMul (pFn j) (vTileFn j d))
        = (∑ j : Fin BLOCK_N, WithBot.realMul
            (WithBot.realExp (WithBot.realSub
              (if c * BLOCK_N + j.val < S then ((qk ⟨c * BLOCK_N + j.val, by
                have hlt := j.isLt; have hw : (c + 1) * BLOCK_N ≤ S := hwin; rw [Nat.succ_mul] at hw; omega⟩ : ℝ) : WithBot ℝ)
               else (⊥ : WithBot ℝ)) (srRunningMax qk vF ((c + 1) * BLOCK_N) (⟨0, hD⟩ : Fin BLOCK_DMODEL))))
            (((fun jL => s8.readMem V ((vIdxFn jL * svbs).toNat + (s0.pids 1 * svh + d.val * svd))) j : ℝ) : WithBot ℝ)) := by
      apply Finset.sum_congr rfl; intro j _
      rw [hpFnBridge j]; rfl
    rw [hsumacc]
    exact sr_acc_reg_eqG hD qk vF BLOCK_N c d
      (fun jL => s8.readMem V ((vIdxFn jL * svbs).toNat + (s0.pids 1 * svh + d.val * svd))) hwin hN hrawV

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General preLoop execution.** The 12 general preLoop statements step a clean
state to one satisfying `srInvariantG … 0`. -/
theorem srPreLoopG_eval
    (s : BlockState) (V : RegionName) (BLoc : Region .int) (BStartLoc BSeqLen : Region .nat)
    (mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (srPreLoopG V BStartLoc BSeqLen mil svh svd sb ss BLOCK_DMODEL BLOCK_N) s = some s0
      ∧ srInvariantG BSeqLen.cast V BLoc BStartLoc BSeqLen mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N s 0 s0 := by
  unfold srPreLoopG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region BSeqLen (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
        = some (Tile.scalar (srSeqLen s BSeqLen.cast)) from by
      simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind, Option.pure_def, srSeqLen]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region BStartLoc (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
        = some (Tile.scalar (srStartLoc s BStartLoc.cast)) from by
      simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind, Option.pure_def, srStartLoc]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BLOCK_N) _ = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) from evalOp_arange BLOCK_N _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BLOCK_DMODEL) _ = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val)) from evalOp_arange BLOCK_DMODEL _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat svh))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
            (Op.constNat svd))) _
        = some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] => s.pids 1 * svh + idx.2.1.val * svd⟩ : Tile .nat [1, BLOCK_DMODEL]) from
      sr_offv_evalG BLOCK_DMODEL _ (s.pids 1) svh svd (by simp) (by simp [Tile.vec])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat sb))
          (Op.mul .nat Broadcast.nil
            (Op.sub .nat Broadcast.nil (Op.constNat mil) (Op.ref .nat [] "cur_batch_seq_len"))
            (Op.constNat ss))) _
        = some (Tile.scalar (srOffBLocG s BSeqLen.cast mil sb ss)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul, evalOp_sub]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_pids, Option.bind_eq_bind,
        Option.bind_some, srOffBLocG, srSeqLen]
      refine congrArg some ?_
      ext idx
      simp only [Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V) (Op.ref .nat [1, BLOCK_DMODEL] "off_v")) _
        = some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] => (V, s.pids 1 * svh + idx.2.1.val * svd)⟩ : Tile .ptr [1, BLOCK_DMODEL]) from by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, BlockState.setReg_pids, Option.bind]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨z, e, u⟩ := idx
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp Op.negInf _ = some (Tile.scalar (⊥ : WithBot ℝ)) from by
      simp [evalOp_negInf]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.const 0.0) _ = some (Tile.scalar (some (0 : ℝ))) from by
      simp only [evalOp_const]; norm_num))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BLOCK_DMODEL] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BLOCK_DMODEL] => some (0 : ℝ)⟩ : Tile .real [BLOCK_DMODEL]) from by
      simp [evalOp_full, evalOp_const]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp, ?_, ?_, by simp, by simp, by simp, by simp, by simp [Tile.vec], by simp [Tile.vec],
    by simp, by simp, ?_, ?_, ?_⟩
  · funext rg o; simp
  · intro rg o; simp [hundef]
  · intro hD
    rw [Nat.zero_mul, srRunningMax_zero]
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true]
  · intro hD
    rw [Nat.zero_mul, srStateBot_zero]
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true]
    rfl
  · simp only [BlockState.setReg_same, Nat.zero_mul, srStateBot_zero]
    rfl

/-- General `c = 0` invariant independence of `Logics`/`BLoc`. -/
theorem srInvariantG_zero_logics_irrel
    (Logics Logics' V : RegionName) (BLoc BLoc' : Region .int) (BStartLoc BSeqLen : RegionName)
    (mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N : Nat)
    (s0 s : BlockState)
    (h : srInvariantG Logics V BLoc BStartLoc BSeqLen mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N s0 0 s) :
    srInvariantG Logics' V BLoc' BStartLoc BSeqLen mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N s0 0 s := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14⟩ := h
  refine ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, ?_, ?_, ?_⟩
  · intro hD; rw [h12 hD]; simp only [Nat.zero_mul, srRunningMax_zero]
  · intro hD; rw [h13 hD]; simp only [Nat.zero_mul, srStateBot_zero]
  · rw [h14]; simp only [Nat.zero_mul, srStateBot_zero]

/-- General `softmaxReducevWeightedSum` reindex. -/
theorem softmaxReducevWeightedSum_reindexG {S S' BLOCK_DMODEL : Nat} (hSS : S = S')
    (qk : Fin S → ℝ) (qk' : Fin S' → ℝ) (mr : ℝ)
    (v : Fin S → Fin BLOCK_DMODEL → ℝ) (v' : Fin S' → Fin BLOCK_DMODEL → ℝ) (d : Fin BLOCK_DMODEL)
    (hqk : ∀ n : Fin S, qk n = qk' (Fin.cast hSS n))
    (hv : ∀ (n : Fin S) (e : Fin BLOCK_DMODEL), v n e = v' (Fin.cast hSS n) e) :
    softmaxReducevWeightedSum qk mr v d = softmaxReducevWeightedSum qk' mr v' d := by
  subst hSS
  have hqk' : qk = qk' := by funext n; rw [hqk n]; rfl
  have hv' : v = v' := by funext n e; rw [hv n e]; rfl
  rw [hqk', hv']

/-- General `srRunningMax` reindex. -/
theorem srRunningMax_reindexG {S S' BLOCK_DMODEL : Nat} (hSS : S = S')
    (qk : Fin S → ℝ) (qk' : Fin S' → ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ) (v' : Fin S' → Fin BLOCK_DMODEL → ℝ)
    (hi : Nat) (d : Fin BLOCK_DMODEL)
    (hqk : ∀ n : Fin S, qk n = qk' (Fin.cast hSS n)) :
    srRunningMax qk v hi d = srRunningMax qk' v' hi d := by
  subst hSS
  have hqk' : qk = qk' := by funext n; rw [hqk n]; rfl
  subst hqk'
  unfold srRunningMax srKeysUpto
  congr 1
  rw [List.map_filterMap, List.map_filterMap]
  apply List.filterMap_congr; intro n _
  by_cases hn : n.val < hi <;> simp [hn]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General postLoop execution + genuine readback.** -/
theorem srPostLoopG_eval
    (Logics V Out : RegionName) (BLoc : Region .int) (BStartLoc BSeqLen : RegionName)
    (mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N : Nat)
    (hD : 0 < BLOCK_DMODEL)
    (s0 : BlockState) (c : Nat) (s : BlockState)
    (hfull : c * BLOCK_N = srSeqLen s0 BSeqLen) (_hpos : 0 < srSeqLen s0 BSeqLen)
    (hinv : srInvariantG Logics V BLoc BStartLoc BSeqLen mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N s0 c s)
    (mr : ℝ)
    (hM : srRunningMax (srQkFG s0 Logics BStartLoc BSeqLen slh slb)
      (srVFG s0 V BLoc BSeqLen mil sb ss svbs svh svd BLOCK_DMODEL)
      (srSeqLen s0 BSeqLen) (⟨0, hD⟩ : Fin BLOCK_DMODEL) = (mr : WithBot ℝ))
    (hOutInj : Function.Injective (fun i : Fin BLOCK_DMODEL => outOffsetG s0 sob soh sod i)) :
    ∃ sP, stepStmts (srPostLoopG Out sob soh sod BLOCK_DMODEL) s = some sP
      ∧ ∀ d : Fin BLOCK_DMODEL,
          sP.readMem Out (outOffsetG s0 sob soh sod d)
            = softmaxReducevWeightedSum (srQkFG s0 Logics BStartLoc BSeqLen slh slb) mr
                (srVFG s0 V BLoc BSeqLen mil sb ss svbs svh svd BLOCK_DMODEL) d := by
  set S := srSeqLen s0 BSeqLen with hSdef
  set qk := srQkFG s0 Logics BStartLoc BSeqLen slh slb with hqkdef
  set vF := srVFG s0 V BLoc BSeqLen mil sb ss svbs svh svd BLOCK_DMODEL with hvFdef
  simp only [srInvariantG, ← hqkdef, ← hvFdef, ← hSdef] at hinv
  obtain ⟨hpids, hmem, hundef, hcb, hch, hseq, hsl, hn, hd, hoff, hvp, hemaxA, hesumA, hacc⟩ := hinv
  have hesum := hesumA hD
  set denomVal := ((srStateBot qk vF (c * BLOCK_N) (⟨0, hD⟩ : Fin BLOCK_DMODEL)).2.1 : WithBot ℝ) with hdenomVal
  set accFn : Fin BLOCK_DMODEL → WithBot ℝ := fun d => ((srStateBot qk vF (c * BLOCK_N) d).2.2 : WithBot ℝ) with haccFn
  unfold srPostLoopG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.div .real Broadcast.scalarR (Op.ref .real [BLOCK_DMODEL] "acc") (Op.ref .real [] "e_sum")) s
        = some (⟨fun idx : TileIndex [BLOCK_DMODEL] => WithBot.realDiv (accFn idx.1) denomVal⟩ : Tile .real [BLOCK_DMODEL]) from by
      rw [evalOp_div]
      simp only [evalOp_ref, hacc, hesum, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.div, haccFn, hdenomVal]))]
  set s1 := s.setReg "acc" .real [BLOCK_DMODEL] (⟨fun idx : TileIndex [BLOCK_DMODEL] => WithBot.realDiv (accFn idx.1) denomVal⟩ : Tile .real [BLOCK_DMODEL]) with hs1
  have e1 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc" → s.regs dt sh nm = some t → s1.regs dt sh nm = some t := by
    intro dt sh nm t hne' h; rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne']; exact h
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat sob))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat soh)))
          (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_DMODEL] "offs_d") (Op.constNat sod))) s1
        = some (⟨fun idx : TileIndex [BLOCK_DMODEL] => outOffsetG s0 sob soh sod idx.1⟩ : Tile .nat [BLOCK_DMODEL]) from by
      rw [evalOp_add, evalOp_add, evalOp_mul, evalOp_mul, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat,
        e1 (by decide) hcb, e1 (by decide) hch, e1 (by decide) hd,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul, outOffsetG]))]
  set s2 := s1.setReg "off_o" .nat [BLOCK_DMODEL] (⟨fun idx : TileIndex [BLOCK_DMODEL] => outOffsetG s0 sob soh sod idx.1⟩ : Tile .nat [BLOCK_DMODEL]) with hs2
  have e2 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "off_o" → s1.regs dt sh nm = some t → s2.regs dt sh nm = some t := by
    intro dt sh nm t hne' h; rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne']; exact h
  have hs2offo : s2.regs .nat [BLOCK_DMODEL] "off_o" = some (⟨fun idx : TileIndex [BLOCK_DMODEL] => outOffsetG s0 sob soh sod idx.1⟩ : Tile .nat [BLOCK_DMODEL]) := by
    rw [hs2, BlockState.setReg_same]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BLOCK_DMODEL] "off_o")) s2
        = some (⟨fun idx : TileIndex [BLOCK_DMODEL] => (Out, outOffsetG s0 sob soh sod idx.1)⟩ : Tile .ptr [BLOCK_DMODEL]) from by
      simp only [evalOp, evalOp_ref, hs2offo, Option.bind]
      refine congrArg some (Tile.ext (fun idx => ?_))
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and]))]
  set s3 := s2.setReg "out_ptrs" .ptr [BLOCK_DMODEL] (⟨fun idx : TileIndex [BLOCK_DMODEL] => (Out, outOffsetG s0 sob soh sod idx.1)⟩ : Tile .ptr [BLOCK_DMODEL]) with hs3
  have hs3acc : s3.regs .real [BLOCK_DMODEL] "acc" = some (⟨fun idx : TileIndex [BLOCK_DMODEL] => WithBot.realDiv (accFn idx.1) denomVal⟩ : Tile .real [BLOCK_DMODEL]) := by
    rw [hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hs1, BlockState.setReg_same]
  have hs3ptr : s3.regs .ptr [BLOCK_DMODEL] "out_ptrs" = some (⟨fun idx : TileIndex [BLOCK_DMODEL] => (Out, outOffsetG s0 sob soh sod idx.1)⟩ : Tile .ptr [BLOCK_DMODEL]) := by
    rw [hs3, BlockState.setReg_same]
  have hstore : stepStmt (Stmt.store .real [BLOCK_DMODEL] (MemAccess.ptr (Op.ref .ptr [BLOCK_DMODEL] "out_ptrs"))
      (Op.ref .real [BLOCK_DMODEL] "acc") MaskOpt.none) s3
      = some ((TileShape.allIndices [BLOCK_DMODEL]).foldl
          (fun acc i => acc.writeMemTyped .real Out (outOffsetG s0 sob soh sod i.1)
            ((⟨fun idx : TileIndex [BLOCK_DMODEL] => WithBot.realDiv (accFn idx.1) denomVal⟩ : Tile .real [BLOCK_DMODEL]).data i)) s3) := by
    unfold stepStmt
    simp only [evalOp_ref, hs3acc, hs3ptr, Option.bind_eq_bind, Option.bind_some, Option.map_some]
    rfl
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  simp only [BlockState.writeMemTyped_real]
  refine ⟨_, rfl, ?_⟩
  intro d
  have hinj : Function.Injective (fun i : TileIndex [BLOCK_DMODEL] => outOffsetG s0 sob soh sod i.1) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : a = b := hOutInj hab
    subst this; rfl
  rw [show outOffsetG s0 sob soh sod d = (fun i : TileIndex [BLOCK_DMODEL] => outOffsetG s0 sob soh sod i.1) (d, PUnit.unit) from rfl]
  rw [BlockState.scatter_readback_nd s3 (fun i : TileIndex [BLOCK_DMODEL] => outOffsetG s0 sob soh sod i.1)
      (fun i : TileIndex [BLOCK_DMODEL] => FloatDType.real.storeValue ((⟨fun idx : TileIndex [BLOCK_DMODEL] => WithBot.realDiv (accFn idx.1) denomVal⟩ : Tile .real [BLOCK_DMODEL]).data i)) ?_ (d, PUnit.unit)]
  · simp only [FloatDType.real_storeValue]
    rw [hdenomVal, haccFn]
    rw [show c * BLOCK_N = S from hfull]
    rw [show WithBot.realDiv ((srStateBot qk vF S d).2.2 : WithBot ℝ) ((srStateBot qk vF S (⟨0, hD⟩ : Fin BLOCK_DMODEL)).2.1 : WithBot ℝ)
          = (((srStateBot qk vF S d).2.2 / (srStateBot qk vF S d).2.1 : ℝ) : WithBot ℝ) from by
      rw [show ((srStateBot qk vF S (⟨0, hD⟩ : Fin BLOCK_DMODEL)).2.1 : WithBot ℝ)
            = ((srStateBot qk vF S d).2.1 : WithBot ℝ) from by
        rw [srStateBot_snd_fst, srStateBot_snd_fst, srRunningMax_eqG qk vF S ⟨0, hD⟩ d]
        congr 2
        unfold srKeysUpto
        rw [List.map_filterMap, List.map_filterMap]
        congr 1
        apply List.filterMap_congr; intro n _; by_cases hn : n.val < S <;> simp [hn]]
      rfl]
    rw [WithBot.unbotD_coe]
    rw [show (srStateBot qk vF S d).2.2 / (srStateBot qk vF S d).2.1
          = softmaxReducevWeightedSum qk mr vF d from by
      have hMd : srRunningMax qk vF S d = (mr : WithBot ℝ) := by
        rw [srRunningMax_eqG qk vF S d ⟨0, hD⟩]; exact hM
      have := srStateBot_full_eq_weightedSum qk vF d mr hMd; simpa using this]
  · exact fun a b hab => hinj hab

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General full-kernel execution chain.** -/
theorem sr_execG
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
    ∃ sF, stepStmts ((softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
        mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N other_kv_index).toAlgKernel.body) s = some sF
      ∧ ∀ d : Fin BLOCK_DMODEL,
          sF.readMem Out (outOffsetG s sob soh sod d)
            = softmaxReducevWeightedSum (srQkFG s Logics BStartLoc.cast BSeqLen.cast slh slb) mr
                (srVFG s V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL) d := by
  rw [srBody_splitG]
  obtain ⟨s0, hpre, hinv0'⟩ := srPreLoopG_eval s V BLoc BStartLoc BSeqLen mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N hundef
  have hinv0 : srInvariantG Logics V BLoc BStartLoc.cast BSeqLen.cast mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N s 0 s0 :=
    srInvariantG_zero_logics_irrel BSeqLen.cast Logics V BLoc BLoc BStartLoc.cast BSeqLen.cast
      mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N s s0 hinv0'
  rw [stepStmts.append_some hpre]
  obtain ⟨hpids0, hmem0, hundef0, hcb0, hch0, hseq0, hsl0, hn0, hd0, hoff0, hvp0, h12, h13, h14⟩ := hinv0
  set S := srSeqLen s BSeqLen.cast with hSdef
  have hseqS : srSeqLen s0 BSeqLen.cast = S := by
    rw [hSdef]; exact srSeqLen_eq_of_mem_pids s0 s BSeqLen.cast hmem0 hpids0
  have hinv0' : srInvariantG Logics V BLoc BStartLoc.cast BSeqLen.cast mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N s0 0 s0 := by
    refine srInvariantG_zero_logics_irrel Logics Logics V BLoc BLoc BStartLoc.cast BSeqLen.cast
      mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N s0 s0 ?_
    refine ⟨rfl, rfl, hundef0, ?_, ?_, ?_, ?_, hn0, hd0, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hcb0, hpids0]
    · rw [hch0, hpids0]
    · rw [hseq0, srSeqLen_eq_of_mem_pids s0 s BSeqLen.cast hmem0 hpids0]
    · rw [hsl0, srStartLoc_eq_of_mem_pids s0 s BStartLoc.cast hmem0 hpids0]
    · rw [hoff0]; simp only [srOffBLocG, hpids0,
        srSeqLen_eq_of_mem_pids s0 s BSeqLen.cast hmem0 hpids0]
    · rw [hvp0, hpids0]
    · intro hD'; rw [h12 hD']; simp only [Nat.zero_mul, srRunningMax_zero]
    · intro hD'; rw [h13 hD']; simp only [Nat.zero_mul, srStateBot_zero]
    · rw [h14]; simp only [Nat.zero_mul, srStateBot_zero]
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    VeriTile.Triton.forRangeDyn_inv (idx := "start_n")
      (startOp := Op.constNat 0) (stopOp := Op.ref .nat [] "cur_batch_seq_len")
      (stepOp := Op.constNat BLOCK_N)
      (P := fun i st => srInvariantG Logics V BLoc BStartLoc.cast BSeqLen.cast mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N s0 (i / BLOCK_N) st
        ∧ i % BLOCK_N = 0 ∧ i ≤ srSeqLen s0 BSeqLen.cast)
      (s_init := s0)
      (start := 0) (stop := srSeqLen s0 BSeqLen.cast) (step := BLOCK_N)
      (by rw [evalOp_constNat])
      (by rw [evalOp_ref, hseq0, hseqS])
      (by rw [evalOp_constNat])
      hN.ne'
      ⟨by rw [Nat.zero_div]; exact hinv0', Nat.zero_mod _, by omega⟩
      (fun i st hi hP => by
        obtain ⟨hPinv, hPmod, hPle⟩ := hP
        obtain ⟨s', hstep, hinv'⟩ := sr_attn_stepG Logics V BLoc BStartLoc.cast BSeqLen.cast
          mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N hD hN other_kv_index s0 i st
          hi (by rw [hseqS]; exact hseqmod) hPinv
          ((Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hPmod)).symm)
        refine ⟨s', hstep, ?_, by
          rw [Nat.add_mod, hPmod, Nat.mod_self]; simp, by
          have hmod : srSeqLen s0 BSeqLen.cast % BLOCK_N = 0 := by rw [hseqS]; exact hseqmod
          have : (i / BLOCK_N + 1) * BLOCK_N ≤ srSeqLen s0 BSeqLen.cast := by
            have hi2 : i / BLOCK_N * BLOCK_N < srSeqLen s0 BSeqLen.cast := by
              rw [Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hPmod)]; exact hi
            have hclt : i / BLOCK_N < srSeqLen s0 BSeqLen.cast / BLOCK_N := by
              apply Nat.lt_of_mul_lt_mul_right (a := BLOCK_N)
              rw [Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)]; exact hi2
            calc (i / BLOCK_N + 1) * BLOCK_N ≤ (srSeqLen s0 BSeqLen.cast / BLOCK_N) * BLOCK_N :=
                  Nat.mul_le_mul_right _ hclt
              _ = srSeqLen s0 BSeqLen.cast := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)
          rw [Nat.succ_mul] at this
          have hieq : i / BLOCK_N * BLOCK_N = i := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hPmod)
          omega⟩
        rw [show (i + BLOCK_N) / BLOCK_N = i / BLOCK_N + 1 from by
          rw [Nat.add_div_right i hN]]; exact hinv')
  rw [stepStmts.cons_some hloop]
  obtain ⟨hinvLinv, hinvLmod, hinvLle⟩ := hinvL
  have hfinS : final = srSeqLen s0 BSeqLen.cast := by
    have hmod : srSeqLen s0 BSeqLen.cast % BLOCK_N = 0 := by rw [hseqS]; exact hseqmod
    omega
  subst hfinS
  have hfull : (srSeqLen s0 BSeqLen.cast / BLOCK_N) * BLOCK_N = srSeqLen s0 BSeqLen.cast := by
    have hmod : srSeqLen s0 BSeqLen.cast % BLOCK_N = 0 := by rw [hseqS]; exact hseqmod
    rw [Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)]
  have hMs0 : srRunningMax (srQkFG s0 Logics BStartLoc.cast BSeqLen.cast slh slb)
      (srVFG s0 V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL)
      (srSeqLen s0 BSeqLen.cast) (⟨0, hD⟩ : Fin BLOCK_DMODEL) = (mr : WithBot ℝ) := by
    rw [srRunningMax_reindexG (S := srSeqLen s0 BSeqLen.cast) (S' := srSeqLen s BSeqLen.cast)
      (by rw [hseqS])
      (srQkFG s0 Logics BStartLoc.cast BSeqLen.cast slh slb) (srQkFG s Logics BStartLoc.cast BSeqLen.cast slh slb)
      (srVFG s0 V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL) (srVFG s V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL) (srSeqLen s0 BSeqLen.cast) ⟨0, hD⟩
      (fun n => by simp only [srQkFG, Fin.val_cast]; rw [srQkG_eq_of_mem_pids s0 s Logics BStartLoc.cast slh slb hmem0 hpids0])]
    rw [hseqS]; exact hM
  obtain ⟨sP, hpost, hOut⟩ := srPostLoopG_eval Logics V Out BLoc BStartLoc.cast BSeqLen.cast
    mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N hD s0
    (srSeqLen s0 BSeqLen.cast / BLOCK_N) sL
    hfull (by rw [hseqS]; exact hseqpos) hinvLinv mr hMs0 (hOutInj s0)
  rw [hpost]
  refine ⟨sP, rfl, ?_⟩
  intro d
  have hoeq : outOffsetG s0 sob soh sod d = outOffsetG s sob soh sod d := by
    simp only [outOffsetG, hpids0]
  rw [← hoeq, hOut d]
  exact softmaxReducevWeightedSum_reindexG (by rw [hseqS])
    (srQkFG s0 Logics BStartLoc.cast BSeqLen.cast slh slb) (srQkFG s Logics BStartLoc.cast BSeqLen.cast slh slb) mr
    (srVFG s0 V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL) (srVFG s V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL) d
    (fun n => by simp only [srQkFG, Fin.val_cast]; rw [srQkG_eq_of_mem_pids s0 s Logics BStartLoc.cast slh slb hmem0 hpids0])
    (fun n e => by simp only [srVFG, Fin.val_cast]; rw [srVG_eq_of_mem_pids s0 s V BLoc BSeqLen.cast mil sb ss svbs svh svd hmem0 hpids0])

set_option maxHeartbeats 1000000 in
/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **General genuine closed-form `Out`-store correctness.** Every output lane of the
`softmax_reducev` kernel holds the genuine softmax-weighted V reduction
`softmaxReducevWeightedSum` of the loaded logits / gathered V rows. Fully
dimension-parameterized: `BLOCK_N` (key block), `BLOCK_DMODEL` (channel), and all
strides. Side conditions: `0 < BLOCK_N`, `0 < BLOCK_DMODEL`, `BLOCK_N ∣ srSeqLen`
(`% = 0`), `0 < srSeqLen`, contiguous output offset injectivity, `hundef`, and the
finite running max `mr`. -/
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
          (srVFG s V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL) d) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_reducev_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro d
  obtain ⟨sF, hstep, hOut⟩ := sr_execG Logics V Out BLoc BStartLoc BSeqLen
    mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N other_kv_index hD hN s hundef
    hseqmod hseqpos hOutInj mr hM
  rw [show exec _ s = stepStmts _ s from rfl, hstep] at hExec
  obtain rfl : sF = s' := Option.some.inj hExec
  simp only [ComputeCorrect.OutputReadable.read_real]
  exact hOut d

end Correct_without_Rounding

/-! # ══════════ IO-face (`⊨[R]`) — the gather skin ══════════ -/

section IOFace

open scoped VeriTile.Triton.StreamMetaGatherMasked3DKernelIO₂

/-! ## Slot table, step budget and IO signature

`softmax_reducev` is the exemplar consumer of
`StreamMetaGatherMasked3DKernelIO₂`: the **gather-indexed** two-stream fold
skin. The 2-D grid is `(cur_batch, cur_head) = (pid₀, pid₁)` (`pid₂`
unused); the two `.nat` metadata slots are read at cell `cur_batch = pid₀`
of their own regions, in the kernel's own load order — slot `0` =
`B_Seqlen[pid₀]` (`cur_batch_seq_len`, the loop's dynamic trip count), slot
`1` = `B_Start_Loc[pid₀]` (`cur_batch_start_loc`, the logits' row base).

The **gather channel** is the paged-KV page table `B_Loc` at `gty = .int`
(the port models it as a signed region, because the Python `other=`
sentinel `other_kv_index` is a *signed* host argument): per step `t` the
kernel loads `BLOCK_N` page indices, masked by `t·BN + jL < seq_len` with
`other = other_kv_index`, and those *values* address the `V` load. The
skin's two-leg pin is exactly what the port needs: on live lanes the
readback is `G t jL`, on dead lanes `G t jL = other_kv_index` — so the
**unmasked** `V` load's address is `read2 … G …` on *every* lane, dead ones
included (see `mask2 ≡ True` below). -/

/-- Slot-region table of the two per-batch metadata slots, in the kernel's
own load order (`B_Seqlen` then `B_Start_Loc`). A shared def, never an
inline `match` in a window position. -/
def srIOMetaBuf (BSeqLen BStartLoc : Region .nat) : Fin 2 → RegionName
  | ⟨0, _⟩ => BSeqLen.cast
  | ⟨_ + 1, _⟩ => BStartLoc.cast

/-- The pid-free step budget `T = max_input_len / BLOCK_N`. At a `pre`-legal
launch (`seq_len ≤ max_input_len` and `BLOCK_N ∣ seq_len`) the live trip
count `seq_len` never outruns `T·BLOCK_N` (`srIOT_seq_le`), so floor
division suffices — no ceiling is needed. -/
def srIOT (max_input_len BLOCK_N : Nat) : Nat :=
  max_input_len / BLOCK_N

/-- Under the launch bound the live window fits the budget:
`seq_len ≤ T·BLOCK_N`. -/
theorem srIOT_seq_le (S mil BLOCK_N : Nat) (hle : S ≤ mil)
    (hmod : S % BLOCK_N = 0) : S ≤ srIOT mil BLOCK_N * BLOCK_N := by
  rw [srIOT]
  calc S = S / BLOCK_N * BLOCK_N :=
        (Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)).symm
    _ ≤ mil / BLOCK_N * BLOCK_N :=
        Nat.mul_le_mul_right _ (Nat.div_le_div_right hle)

/-- Every live token's block index is a legal step index. -/
theorem srIO_step_lt (n S mil BLOCK_N : Nat) (hBN : 0 < BLOCK_N) (hn : n < S)
    (hle : S ≤ mil) (hmod : S % BLOCK_N = 0) :
    n / BLOCK_N < srIOT mil BLOCK_N := by
  have h1 : n / BLOCK_N < S / BLOCK_N := by
    refine (Nat.div_lt_iff_lt_mul hBN).mpr ?_
    rw [Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)]
    exact hn
  exact Nat.lt_of_lt_of_le h1 (Nat.div_le_div_right hle)

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

/-! ## Stream-indexed tiles and the streamed closed form -/

/-- The logit of live token `n` read off the first stream: step
`n / BLOCK_N`, lane `n % BLOCK_N` (`0` past the step budget — unreachable
at any `pre`-legal launch). -/
noncomputable def srIOqk (BLOCK_N T S : Nat) (hBN : 0 < BLOCK_N)
    (xs : Fin T → Fin BLOCK_N → ℝ) (n : Fin S) : ℝ :=
  if h : n.val / BLOCK_N < T then
    xs ⟨n.val / BLOCK_N, h⟩ ⟨n.val % BLOCK_N, Nat.mod_lt _ hBN⟩
  else 0

/-- The gathered value row of live token `n`, channel `d`, read off the
second stream at lane `(n % BLOCK_N, d)`. The page indirection lives in the
*window* (`read2` eats `G`), so the stream cell is already the gathered
row. -/
noncomputable def srIOvv (BLOCK_N BLOCK_DMODEL T S : Nat) (hBN : 0 < BLOCK_N)
    (ys : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ) (n : Fin S)
    (d : Fin BLOCK_DMODEL) : ℝ :=
  if h : n.val / BLOCK_N < T then
    ys ⟨n.val / BLOCK_N, h⟩
      (Lane2D.encode (⟨n.val % BLOCK_N, Nat.mod_lt _ hBN⟩, d, PUnit.unit))
  else 0

/-- **The streamed closed form**: `softmaxReducevWeightedSum` restated over
the two streamed tiles — `out[d] = Σₙ softmax(qk)[n] · v[n, d]` over the
`S = m 0` live tokens. The running-max argument is pinned at `0`: the
quotient is invariant under the shift (`srWeightedSum_shift_invariant`), so
the spec needs no free `mr` (this is how the exact headline's free-variable
`hM` hypothesis is discharged rather than exported). -/
noncomputable def softmaxReducevIOSpec (BLOCK_N BLOCK_DMODEL T S : Nat)
    (hBN : 0 < BLOCK_N) (xs : Fin T → Fin BLOCK_N → ℝ)
    (ys : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ) (d : Fin BLOCK_DMODEL) : ℝ :=
  softmaxReducevWeightedSum (srIOqk BLOCK_N T S hBN xs) 0
    (srIOvv BLOCK_N BLOCK_DMODEL T S hBN ys) d

/-! ## Running-max shift invariance

The exact headline carries `hM : srRunningMax … = mr` with `mr` free, and
its expected value is `softmaxReducevWeightedSum qk mr v d`. A skin's `f`
cannot carry a free real, so the io face proves what the docstring of
`softmaxReducevWeightedSum` already claims — the quotient is independent of
the shift — and pins `mr := 0`. -/

/-- `softmaxReducevDenom qk M = exp(−M)·Σ exp(qk n)`. -/
theorem srWeightedSum_denom_shift {S : Nat} (qk : Fin S → ℝ) (M : ℝ) :
    softmaxReducevDenom qk M = Real.exp (-M) * ∑ n : Fin S, Real.exp (qk n) := by
  unfold softmaxReducevDenom softmaxWeight
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl (fun n _ => ?_)
  rw [← Real.exp_add]; ring_nf

/-- `softmaxReducevAcc qk M v d = exp(−M)·Σ exp(qk n)·v n d`. -/
theorem srWeightedSum_acc_shift {S BLOCK_DMODEL : Nat} (qk : Fin S → ℝ) (M : ℝ)
    (v : Fin S → Fin BLOCK_DMODEL → ℝ) (d : Fin BLOCK_DMODEL) :
    softmaxReducevAcc qk M v d
      = Real.exp (-M) * ∑ n : Fin S, Real.exp (qk n) * v n d := by
  unfold softmaxReducevAcc softmaxWeight
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl (fun n _ => ?_)
  rw [← mul_assoc, ← Real.exp_add]; ring_nf

/-- The softmax-weighted sum in shift-free form. -/
theorem srWeightedSum_eq_unshifted {S BLOCK_DMODEL : Nat} (qk : Fin S → ℝ) (M : ℝ)
    (v : Fin S → Fin BLOCK_DMODEL → ℝ) (d : Fin BLOCK_DMODEL) :
    softmaxReducevWeightedSum qk M v d
      = (∑ n : Fin S, Real.exp (qk n) * v n d) / (∑ n : Fin S, Real.exp (qk n)) := by
  unfold softmaxReducevWeightedSum
  rw [srWeightedSum_acc_shift, srWeightedSum_denom_shift]
  exact mul_div_mul_left _ _ (Real.exp_ne_zero _)

/-- **Running-max invariance of the softmax-weighted sum.** Both legs scale
by `exp(−M)`, which cancels in the quotient. -/
theorem srWeightedSum_shift_invariant {S BLOCK_DMODEL : Nat} (qk : Fin S → ℝ)
    (v : Fin S → Fin BLOCK_DMODEL → ℝ) (d : Fin BLOCK_DMODEL) (M₁ M₂ : ℝ) :
    softmaxReducevWeightedSum qk M₁ v d = softmaxReducevWeightedSum qk M₂ v d := by
  rw [srWeightedSum_eq_unshifted qk M₁ v d, srWeightedSum_eq_unshifted qk M₂ v d]

/-! ## The spec-equals-genuine bridge

The exact stack's genuine closed form names the kernel's own memory-side
data functions (`srQkFG`, and `srVFG` — which *inlines the gather* inside
the `V` read). The io face states the same value over the pinned streams;
the bridge rewrites `srVIndexG ↦ G t jL` through the gather pin, which is
the one real proof-content delta of the gather skin. -/

set_option maxHeartbeats 1600000 in
/-- **Streamed spec = genuine closed form.** Under the slot/gather/data pins
the exact headline's `softmaxReducevWeightedSum` over `srQkFG`/`srVFG` (at
*any* running-max shift `mr`) is the streamed `softmaxReducevIOSpec`. The
`V` leg goes through the gather pin: the kernel's `srVIndexG` page index at
live token `n` *is* `G ⟨n / BLOCK_N⟩ ⟨n % BLOCK_N⟩`. -/
theorem srIOSpec_eq_genuineG
    (Logics V : RegionName) (BLoc : Region .int) (BStartLoc BSeqLen : RegionName)
    (mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N : Nat)
    (hBN : 0 < BLOCK_N) (s₀ : BlockState) (S startloc T : Nat)
    (hT : T = srIOT mil BLOCK_N)
    (hS : srSeqLen s₀ BSeqLen = S) (hstart : srStartLoc s₀ BStartLoc = startloc)
    (hSle : S ≤ mil) (hSmod : S % BLOCK_N = 0)
    (G : Fin T → Fin BLOCK_N → Int) (xs : Fin T → Fin BLOCK_N → ℝ)
    (ys : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (hg : ∀ (t : Fin T) (jL : Fin BLOCK_N), t.val * BLOCK_N + jL.val < S →
      s₀.readMemValue .int (Region.cast BLoc)
          (s₀.pids 0 * sb + (mil - S) * ss + (t.val * BLOCK_N + jL.val) * ss)
        = G t jL)
    (hx : ∀ (t : Fin T) (jL : Fin BLOCK_N), t.val * BLOCK_N + jL.val < S →
      s₀.readMem Logics
          (s₀.pids 1 * slh + (startloc + (t.val * BLOCK_N + jL.val)) * slb)
        = xs t jL)
    (hy : ∀ (t : Fin T) (j : Fin (BLOCK_N * BLOCK_DMODEL)),
      s₀.readMem V ((G t (Lane2D.decode j).1 * svbs).toNat + s₀.pids 1 * svh
          + (Lane2D.decode j).2.1.val * svd) = ys t j)
    (mr : ℝ) (d : Fin BLOCK_DMODEL) :
    softmaxReducevWeightedSum (srQkFG s₀ Logics BStartLoc BSeqLen slh slb) mr
        (srVFG s₀ V BLoc BSeqLen mil sb ss svbs svh svd BLOCK_DMODEL) d
      = softmaxReducevIOSpec BLOCK_N BLOCK_DMODEL T S hBN xs ys d := by
  rw [srWeightedSum_shift_invariant _ _ d mr 0]
  unfold softmaxReducevIOSpec
  have hsplit : ∀ n : Nat, n / BLOCK_N * BLOCK_N + n % BLOCK_N = n := fun n =>
    Nat.div_add_mod' n BLOCK_N
  have hstepT : ∀ n : Fin (srSeqLen s₀ BSeqLen), n.val / BLOCK_N < T := by
    intro n
    rw [hT]
    exact srIO_step_lt n.val S mil BLOCK_N hBN (by rw [← hS]; exact n.isLt) hSle hSmod
  refine softmaxReducevWeightedSum_reindexG hS _ _ 0 _ _ d ?_ ?_
  · -- the logits leg
    intro n
    have hlt : n.val < S := by rw [← hS]; exact n.isLt
    have hm : n.val % BLOCK_N < BLOCK_N := Nat.mod_lt _ hBN
    have hxn : s₀.readMem Logics
        (s₀.pids 1 * slh
          + (startloc + (n.val / BLOCK_N * BLOCK_N + n.val % BLOCK_N)) * slb)
        = xs ⟨n.val / BLOCK_N, hstepT n⟩ ⟨n.val % BLOCK_N, hm⟩ :=
      hx ⟨n.val / BLOCK_N, hstepT n⟩ ⟨n.val % BLOCK_N, hm⟩
        (by rw [hsplit n.val]; exact hlt)
    rw [hsplit n.val] at hxn
    simp only [srIOqk, Fin.val_cast, dif_pos (hstepT n)]
    simp only [srQkFG, srQkG, hstart]
    exact hxn
  · -- the gathered value leg
    intro n e
    have hlt : n.val < S := by rw [← hS]; exact n.isLt
    have hm : n.val % BLOCK_N < BLOCK_N := Nat.mod_lt _ hBN
    have hgn : s₀.readMemValue .int (Region.cast BLoc)
        (s₀.pids 0 * sb + (mil - S) * ss
          + (n.val / BLOCK_N * BLOCK_N + n.val % BLOCK_N) * ss)
        = G ⟨n.val / BLOCK_N, hstepT n⟩ ⟨n.val % BLOCK_N, hm⟩ :=
      hg ⟨n.val / BLOCK_N, hstepT n⟩ ⟨n.val % BLOCK_N, hm⟩
        (by rw [hsplit n.val]; exact hlt)
    rw [hsplit n.val] at hgn
    have hidx : srVIndexG s₀ BLoc BSeqLen mil sb ss n.val
        = G ⟨n.val / BLOCK_N, hstepT n⟩ ⟨n.val % BLOCK_N, hm⟩ := by
      rw [← hS] at hgn
      rw [srVIndexG, srOffBLocG]
      exact hgn
    have hyn : s₀.readMem V
        ((G ⟨n.val / BLOCK_N, hstepT n⟩ ⟨n.val % BLOCK_N, hm⟩ * svbs).toNat
          + s₀.pids 1 * svh + e.val * svd)
        = ys ⟨n.val / BLOCK_N, hstepT n⟩
            (Lane2D.encode (⟨n.val % BLOCK_N, hm⟩, e, PUnit.unit)) := by
      have h := hy ⟨n.val / BLOCK_N, hstepT n⟩
        (Lane2D.encode ((⟨n.val % BLOCK_N, hm⟩ : Fin BLOCK_N), e, PUnit.unit))
      rw [Lane2D.decode_encode] at h
      exact h
    simp only [srIOvv, Fin.val_cast, dif_pos (hstepT n)]
    simp only [srVFG, srVG, hidx]
    exact hyn

/-! ## Flat-bridge coverage and cast-freedom

The port is cast-free in the `R` sense: every load/store is at `.real` /
`.nat` / `.int`, the only casts are the *address*-arithmetic
`castNatToInt` / `castIntToNat` of the paged-`V` gather, `tl.multiple_of` is
a self-assign, and there are no `ptrSub`s or block pointers. So
`stepStmtsR R` collapses verbatim onto the exact stepper on every segment. -/

set_option maxHeartbeats 4000000 in
/-- The surface sits inside the flat-memory bridge's covered fragment. -/
theorem srIO_flattenOk (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat)
    (mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int) :
    ((softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N
      other_kv_index).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [srBody_splitG]
  simp [srPreLoopG, srLoopBodyG, srPostLoopG, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]
  simp [Op.FlattenOk.eq_def]

/-- Per-statement cast-free collapse lifts to statement lists (walks the
actual successor chain; a failing step collapses on both sides). Private
copy of the family helper (bench files never import each other). -/
private theorem srIO_stepStmtsR_castFree_of_stmts (R : RoundingModel) :
    ∀ (l : List Stmt), (∀ st ∈ l, ∀ u, stepStmtR R st u = stepStmt st u) →
      ∀ s, stepStmtsR R l s = stepStmts l s
  | [], _, s => by simp only [stepStmtsR, stepStmts]
  | st :: rest, h, s => by
      simp only [stepStmtsR, stepStmts, h st List.mem_cons_self s]
      cases stepStmt st s with
      | none => rfl
      | some s' =>
          exact srIO_stepStmtsR_castFree_of_stmts R rest
            (fun st' h' u => h st' (List.mem_cons_of_mem _ h') u) s'

set_option maxHeartbeats 4000000 in
/-- Every preLoop statement is cast-free (two `.nat` slot loads,
register-only `.nat` address arithmetic, the `-inf`/`0.0` register seeds). -/
private theorem srIO_preLoop_stmt_castFree (R : RoundingModel)
    (V : RegionName) (BStartLoc BSeqLen : Region .nat)
    (mil svh svd sb ss BLOCK_DMODEL BLOCK_N : Nat) :
    ∀ st ∈ srPreLoopG V BStartLoc BSeqLen mil svh svd sb ss BLOCK_DMODEL BLOCK_N,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [srPreLoopG, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]

set_option maxHeartbeats 4000000 in
/-- Every loop-body statement is cast-free: the two masked loads are at
`.int` (`other = other_kv_index`) and `.real` (`other = -inf`), the paged-`V`
load is an unmasked `.real` pointer load whose address casts
(`castNatToInt` / `castIntToNat`) are exact under `evalOpR`, and the fold
arithmetic is `.real`. -/
private theorem srIO_body_stmt_castFree (R : RoundingModel)
    (Logics : RegionName) (BLoc : Region .int)
    (slh slb svbs ss BLOCK_DMODEL BLOCK_N : Nat) (other_kv_index : Int) :
    ∀ st ∈ srLoopBodyG Logics BLoc slh slb svbs ss BLOCK_DMODEL BLOCK_N other_kv_index,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [srLoopBodyG, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]

/-- The loop body is cast-free as a list. -/
private theorem srIO_body_castFree (R : RoundingModel)
    (Logics : RegionName) (BLoc : Region .int)
    (slh slb svbs ss BLOCK_DMODEL BLOCK_N : Nat) (other_kv_index : Int)
    (t : BlockState) :
    stepStmtsR R (srLoopBodyG Logics BLoc slh slb svbs ss BLOCK_DMODEL BLOCK_N
        other_kv_index) t
      = stepStmts (srLoopBodyG Logics BLoc slh slb svbs ss BLOCK_DMODEL BLOCK_N
          other_kv_index) t :=
  srIO_stepStmtsR_castFree_of_stmts R _
    (srIO_body_stmt_castFree R Logics BLoc slh slb svbs ss BLOCK_DMODEL BLOCK_N
      other_kv_index) t

set_option maxHeartbeats 4000000 in
/-- Every postLoop statement is cast-free (`writeMemTypedR R .real` *is*
`writeMemTyped .real`). -/
private theorem srIO_postLoop_stmt_castFree (R : RoundingModel)
    (Out : RegionName) (sob soh sod BLOCK_DMODEL : Nat) :
    ∀ st ∈ srPostLoopG Out sob soh sod BLOCK_DMODEL,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [srPostLoopG, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl
  all_goals
    simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def,
      BlockState.writeMemTypedR]

/-- `evalOpR` of a `constNat` (R-independent). -/
private theorem srIO_evalOpR_constNat (R : RoundingModel) (n : Nat) (u : BlockState) :
    evalOpR R (Op.constNat n) u = some (Tile.scalar n) := by
  simp [evalOpR]

/-- `evalOpR` of the `forRangeDyn` stop expression (a bare `.nat` register
read — R-independent). -/
private theorem srIO_stopOpR_castFree (R : RoundingModel) (u : BlockState) :
    evalOpR R (Op.ref .nat [] "cur_batch_seq_len") u
      = evalOp (Op.ref .nat [] "cur_batch_seq_len") u := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

set_option maxHeartbeats 1600000 in
/-- The streaming `forRangeDyn` statement is cast-free per-state: its bound
expressions are a literal and a `.nat` register read, and its body is
cast-free, so `stepStmtR R` on the whole loop *is* `stepStmt`. -/
private theorem srIO_dyn_castFree (R : RoundingModel)
    (Logics : RegionName) (BLoc : Region .int)
    (slh slb svbs ss BLOCK_DMODEL BLOCK_N : Nat) (other_kv_index : Int) :
    ∀ u, stepStmtR R (Stmt.forRangeDyn "start_n" (Op.constNat 0)
        (Op.ref .nat [] "cur_batch_seq_len") (Op.constNat BLOCK_N)
        (srLoopBodyG Logics BLoc slh slb svbs ss BLOCK_DMODEL BLOCK_N
          other_kv_index)) u
      = stepStmt (Stmt.forRangeDyn "start_n" (Op.constNat 0)
          (Op.ref .nat [] "cur_batch_seq_len") (Op.constNat BLOCK_N)
          (srLoopBodyG Logics BLoc slh slb svbs ss BLOCK_DMODEL BLOCK_N
            other_kv_index)) u := by
  intro u
  rw [stepForRangeAux.forRangeDyn_unfold]
  simp only [stepStmtR, srIO_evalOpR_constNat, srIO_stopOpR_castFree,
    evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  cases hstop : evalOp (Op.ref .nat [] "cur_batch_seq_len") u with
  | none => rfl
  | some t =>
      simp only [Option.bind_some]
      exact stepForRangeAuxR_castFree R _
        (srIO_body_castFree R Logics BLoc slh slb svbs ss BLOCK_DMODEL BLOCK_N
          other_kv_index) "start_n" _ _ _ u

/-- The whole lowered body is cast-free, statement by statement: `execR R`
on the surface *is* the exact `stepStmts` run. -/
private theorem srIO_execR_collapse (R : RoundingModel)
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat)
    (mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int) (s : BlockState) :
    execR R ((softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
        mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N
        other_kv_index).toAlgKernel) s
      = stepStmts ((softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
          mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N
          other_kv_index).toAlgKernel.body) s := by
  unfold execR
  rw [srBody_splitG]
  refine srIO_stepStmtsR_castFree_of_stmts R _ ?_ s
  intro st hst
  rcases List.mem_append.mp hst with hpre | hrest
  · exact srIO_preLoop_stmt_castFree R V BStartLoc BSeqLen mil svh svd sb ss
      BLOCK_DMODEL BLOCK_N st hpre
  · rcases List.mem_cons.mp hrest with rfl | hpost
    · exact srIO_dyn_castFree R Logics BLoc slh slb svbs ss BLOCK_DMODEL BLOCK_N
        other_kv_index
    · exact srIO_postLoop_stmt_castFree R Out sob soh sod BLOCK_DMODEL st hpost

/-! ## The weak safety stack (`hts`)

The skin's `hts` obligation quantifies over **arbitrary** launch states (no
clean-`undef` pin, no `pre`-legal loop bound), so the exact stack's
`srInvariantG` — whose `e_max`/`e_sum`/`acc` conjuncts pin the *fold values*
of a clean run — is unavailable there. Since every load in this kernel is
`other=`-defaulted or an unmasked `.nat`/pointer read, the whole walk is
`undef`-independent: the weak stack re-runs the register chain with exact
pins for the address-bearing registers and **existential** pins for the
three fold registers. The `pre` hypothesis is what makes the per-step
`Fin T` window bounds citable at all. -/

/-- Combined walk cons: safety of the head at the current state, the R-step
it actually takes, and the pair (safety, run) of the tail from the successor
give the pair for the whole list. Private copy of the family combinator. -/
private theorem srIO_walkCons {R : RoundingModel} {bounds : RegionBounds}
    {P : BlockState → Prop} {st : Stmt} {rest : List Stmt} {s s' : BlockState}
    (h1 : Stmt.TraceSafeR R bounds st s)
    (hstep : stepStmtR R st s = some s')
    (h2 : Stmt.TraceSafeListR R bounds rest s'
      ∧ ∃ sF, stepStmtsR R rest s' = some sF ∧ P sF) :
    Stmt.TraceSafeListR R bounds (st :: rest) s
      ∧ ∃ sF, stepStmtsR R (st :: rest) s = some sF ∧ P sF :=
  ⟨Stmt.TraceSafeListR.cons_intro h1 (fun u hu => by
      rw [hstep] at hu
      exact (Option.some.inj hu) ▸ h2.1),
    by rw [stepStmtsR_cons_some hstep]; exact h2.2⟩

/-- Walk terminator: the empty tail is safe and runs to the current state. -/
private theorem srIO_walkNil {R : RoundingModel} {bounds : RegionBounds}
    {P : BlockState → Prop} (s : BlockState) (h : P s) :
    Stmt.TraceSafeListR R bounds [] s
      ∧ ∃ sF, stepStmtsR R [] s = some sF ∧ P sF :=
  ⟨Stmt.TraceSafeListR.nil_intro, s, by simp only [stepStmtsR], h⟩

/-- R-step of an assign whose op is cast-free: the two collapse into one
walk-ready equation. -/
private theorem srIO_stepR_of_assign {R : RoundingModel} {dt : TileDType}
    {sh : TileShape} {nm : RegName} {e : Op dt sh} {s : BlockState} {v : Tile dt sh}
    (hcf : evalOpR R e s = evalOp e s) (h : evalOp e s = some v) :
    stepStmtR R (.assign dt sh nm e) s = some (s.setReg nm dt sh v) :=
  stepStmtR_assign_eq_some (hcf.trans h)

set_option maxHeartbeats 1600000 in
/-- `evalOpR` of the `B_Loc` gather offsets `off_b_loc + (start_n +
offs_n)·stride_b_loc_s` at pinned registers. -/
private theorem srIO_blocOffsR_eval (R : RoundingModel) {BN : Nat} (u : BlockState)
    (off SN ss : Nat)
    (hoff : u.regs .nat [] "off_b_loc" = some (Tile.scalar off))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : u.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))) :
    evalOpR R (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "off_b_loc")
        (Op.mul .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.ref .nat [BN] "offs_n"))
          (Op.constNat ss))) u
      = some (⟨fun idx : TileIndex [BN] => off + (SN + idx.1.val) * ss⟩
          : Tile .nat [BN]) := by
  rw [show evalOpR R (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "off_b_loc")
        (Op.mul .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.ref .nat [BN] "offs_n"))
          (Op.constNat ss))) u
      = evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "off_b_loc")
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.ref .nat [BN] "offs_n"))
            (Op.constNat ss))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hoff, hsn, hn,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

set_option maxHeartbeats 1600000 in
/-- `evalOpR` of the shared live-token mask `(start_n + offs_n) <
cur_batch_seq_len` (both masked loads carry it) at pinned registers. -/
private theorem srIO_liveMaskR_eval (R : RoundingModel) {BN : Nat} (u : BlockState)
    (SN S : Nat)
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : u.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hseq : u.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar S)) :
    evalOpR R (Op.lt .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.ref .nat [BN] "offs_n"))
        (Op.ref .nat [] "cur_batch_seq_len")) u
      = some (⟨fun idx : TileIndex [BN] => decide (SN + idx.1.val < S)⟩
          : Tile .bool [BN]) := by
  rw [show evalOpR R (Op.lt .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.ref .nat [BN] "offs_n"))
        (Op.ref .nat [] "cur_batch_seq_len")) u
      = evalOp (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.ref .nat [BN] "offs_n"))
          (Op.ref .nat [] "cur_batch_seq_len")) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  simp only [evalOp, hsn, hn, hseq, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec,
    ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
  rfl

set_option maxHeartbeats 1600000 in
/-- `evalOpR` of the `Logics` offsets `cur_head·stride_logic_h +
(cur_batch_start_loc + start_n + offs_n)·stride_logic_bs` at pinned
registers. -/
private theorem srIO_qkOffsR_eval (R : RoundingModel) {BN : Nat} (u : BlockState)
    (ch sl SN slh slb : Nat)
    (hch : u.regs .nat [] "cur_head" = some (Tile.scalar ch))
    (hsl : u.regs .nat [] "cur_batch_start_loc" = some (Tile.scalar sl))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : u.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))) :
    evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat slh))
        (Op.mul .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_start_loc")
              (Op.ref .nat [] "start_n"))
            (Op.ref .nat [BN] "offs_n"))
          (Op.constNat slb))) u
      = some (⟨fun idx : TileIndex [BN] => ch * slh + (sl + SN + idx.1.val) * slb⟩
          : Tile .nat [BN]) := by
  rw [show evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat slh))
        (Op.mul .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_start_loc")
              (Op.ref .nat [] "start_n"))
            (Op.ref .nat [BN] "offs_n"))
          (Op.constNat slb))) u
      = evalOp (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat slh))
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_start_loc")
                (Op.ref .nat [] "start_n"))
              (Op.ref .nat [BN] "offs_n"))
            (Op.constNat slb))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  simp only [evalOp, hch, hsl, hsn, hn, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

set_option maxHeartbeats 1600000 in
/-- `evalOpR` of the paged-`V` **pointer** tree `v_ptrs + v_index[:, None] ·
stride_vbs` (the ℤ multiply, then `Int.toNat`, then the pointer add) at
pinned registers — the address whose in-bounds obligation the skin's
`read2`/`mask2 ≡ True` field carries on *every* lane. -/
private theorem srIO_vPtrR_eval (R : RoundingModel) (BLOCK_N BLOCK_DMODEL : Nat)
    (u : BlockState) (V : RegionName) (svbs : Nat)
    (offFn : Fin BLOCK_DMODEL → Nat) (vIdxFn : Fin BLOCK_N → Int)
    (hvp : u.regs .ptr [1, BLOCK_DMODEL] "v_ptrs" =
      some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] => (V, offFn idx.2.1)⟩
        : Tile .ptr [1, BLOCK_DMODEL]))
    (hvi : u.regs .int [BLOCK_N] "v_index" =
      some (⟨fun idx : TileIndex [BLOCK_N] => vIdxFn idx.1⟩ : Tile .int [BLOCK_N])) :
    evalOpR R (Op.ptrAdd Broadcast.nil.consR.consL
        (Op.ref .ptr [1, BLOCK_DMODEL] "v_ptrs")
        (Op.mul .int Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .int [BLOCK_N] "v_index"))
            (Op.constNat svbs).castNatToInt).castIntToNat) u
      = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
          (V, (vIdxFn idx.1 * svbs).toNat + offFn idx.2.1)⟩
          : Tile .ptr [BLOCK_N, BLOCK_DMODEL]) := by
  rw [show evalOpR R (Op.ptrAdd Broadcast.nil.consR.consL
        (Op.ref .ptr [1, BLOCK_DMODEL] "v_ptrs")
        (Op.mul .int Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .int [BLOCK_N] "v_index"))
            (Op.constNat svbs).castNatToInt).castIntToNat) u
      = evalOp (Op.ptrAdd Broadcast.nil.consR.consL
          (Op.ref .ptr [1, BLOCK_DMODEL] "v_ptrs")
          (Op.mul .int Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .int [BLOCK_N] "v_index"))
              (Op.constNat svbs).castNatToInt).castIntToNat) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  have hexp : @evalOp .int [BLOCK_N, 1]
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .int [BLOCK_N] "v_index")) u
      = some (Tile.expandDim ⟨1, by simp⟩
          (⟨fun idx : TileIndex [BLOCK_N] => vIdxFn idx.1⟩ : Tile .int [BLOCK_N])) :=
    evalOp_expandDim_ref_of_regs .int [BLOCK_N] ⟨1, by simp⟩ "v_index" u _ hvi
  have hc : @evalOp .int [] (Op.constNat svbs).castNatToInt u
      = some (Tile.scalar (svbs : Int)) := by
    simp only [evalOp, evalOp_constNat, Option.bind]; rfl
  have hmulint : @evalOp .int [BLOCK_N, 1]
        (Op.mul .int Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .int [BLOCK_N] "v_index"))
          (Op.constNat svbs).castNatToInt) u
      = some (⟨fun idx : TileIndex [BLOCK_N, 1] => vIdxFn idx.1 * svbs⟩
          : Tile .int [BLOCK_N, 1]) := by
    rw [evalOp_mul, hexp, hc]
    simp only [Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    ext idx
    obtain ⟨j, d, uu⟩ := idx
    simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.expandDim_data,
      TileShape.dropInsertedIndex_succ, TileShape.dropInsertedIndex_nil,
      TileShape.dropInsertedIndex_zero_cons, NumericDType.mul, NumericDType.int_mul,
      Broadcast.leftIndex, Broadcast.rightIndex]
  have hoff : @evalOp .nat [BLOCK_N, 1]
        (Op.mul .int Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .int [BLOCK_N] "v_index"))
            (Op.constNat svbs).castNatToInt).castIntToNat u
      = some (⟨fun idx : TileIndex [BLOCK_N, 1] => (vIdxFn idx.1 * svbs).toNat⟩
          : Tile .nat [BLOCK_N, 1]) := by
    erw [sr_evalOp_castIntToNat, hmulint]
    rfl
  simp only [evalOp, evalOp_ref, hvp, hoff, Option.bind, Option.some.injEq]
  refine congrArg some (Tile.ext (fun idx => ?_))
  obtain ⟨j, d, uu⟩ := idx
  simp only [Tile.ptrAdd_data, Broadcast.leftIndex, Broadcast.rightIndex, Prod.mk.injEq]
  exact ⟨trivial, Nat.add_comm _ _⟩

set_option maxHeartbeats 1600000 in
/-- `evalOpR` of the output offsets `cur_batch·stride_obs + cur_head·stride_oh
+ offs_d·stride_od` at pinned registers. -/
private theorem srIO_offoR_eval (R : RoundingModel) {D : Nat} (u : BlockState)
    (cb ch sob soh sod : Nat)
    (hcb : u.regs .nat [] "cur_batch" = some (Tile.scalar cb))
    (hch : u.regs .nat [] "cur_head" = some (Tile.scalar ch))
    (hd : u.regs .nat [D] "offs_d" = some (Tile.vec (fun e : Fin D => e.val))) :
    evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat sob))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat soh)))
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [D] "offs_d")
          (Op.constNat sod))) u
      = some (⟨fun idx : TileIndex [D] => cb * sob + ch * soh + idx.1.val * sod⟩
          : Tile .nat [D]) := by
  rw [show evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat sob))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat soh)))
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [D] "offs_d")
          (Op.constNat sod))) u
      = evalOp (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat sob))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat soh)))
          (Op.mul .nat Broadcast.scalarR (Op.ref .nat [D] "offs_d")
            (Op.constNat sod))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  rw [evalOp_add, evalOp_add, evalOp_mul, evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hcb, hch, hd, Option.bind_eq_bind,
    Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `evalOpR` of the terminal `out_ptrs` tree at a pinned `off_o`. -/
private theorem srIO_outPtrR_eval (R : RoundingModel) {D : Nat} (u : BlockState)
    (Out : RegionName) (offFn : Fin D → Nat)
    (hoffo : u.regs .nat [D] "off_o" =
      some (⟨fun idx : TileIndex [D] => offFn idx.1⟩ : Tile .nat [D])) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.ref .nat [D] "off_o")) u
      = some (⟨fun idx : TileIndex [D] => (Out, offFn idx.1)⟩ : Tile .ptr [D]) := by
  rw [show evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.ref .nat [D] "off_o")) u
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
          (Op.ref .nat [D] "off_o")) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  simp only [evalOp, evalOp_ref, hoffo, Option.bind]
  refine congrArg some (Tile.ext (fun idx => ?_))
  simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex,
    Broadcast.rightIndex, Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and]

/-- `readMemValue` depends on `mem` only (used to re-anchor the walk's tiles
at the launch state across the `setReg` chain). -/
private theorem srIO_readMemValue_of_mem (u s : BlockState)
    (h : u.mem = s.mem) (r : RegionName) (a : Nat) :
    u.readMemValue .int r a = s.readMemValue .int r a := by
  simp only [BlockState.readMemValue, BlockState.readMemTyped, h]

/-- `readMem` depends on `mem` only. -/
private theorem srIO_readMem_of_mem (u s : BlockState) (h : u.mem = s.mem)
    (r : RegionName) (a : Nat) : u.readMem r a = s.readMem r a := by
  simp only [BlockState.readMem, BlockState.readMemValue, BlockState.readMemTyped, h]

set_option maxHeartbeats 1600000 in
/-- Launch-anchored `v_index` gather recipe: like `sr_vindex_gather_evalG`
but with the window base `off`, the trip value `S` and the gathered raw
values `vRaw` **decoupled** from the running state (so the walk's tiles stay
anchored at the launch state). -/
private theorem srIO_vindex_eval (BLOCK_N : Nat) (u : BlockState) (BLoc : Region .int)
    (off SN S ss : Nat) (other : Int) (vRaw : Nat → Int)
    (hoff : u.regs .nat [] "off_b_loc" = some (Tile.scalar off))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : u.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hseq : u.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar S))
    (hvRaw : ∀ n : Nat,
      u.readMemValue .int (Region.cast BLoc) (off + n * ss) = vRaw n) :
    evalOp (Op.load .int
        (MemAccess.region BLoc
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "off_b_loc")
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.ref .nat [BLOCK_N] "offs_n"))
              (Op.constNat ss))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.ref .nat [BLOCK_N] "offs_n"))
            (Op.ref .nat [] "cur_batch_seq_len"))
          ((Op.constInt other).broadcast [BLOCK_N]))) u
      = some (⟨fun idx : TileIndex [BLOCK_N] =>
          if SN + idx.1.val < S then vRaw (SN + idx.1.val) else other⟩
          : Tile .int [BLOCK_N]) := by
  simp only [evalOp, hoff, hsn, hn, hseq, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, uu⟩ := idx
  simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, Broadcast.leftIndex,
    Broadcast.rightIndex]
  by_cases hlt : SN + j.val < S
  · simp only [hlt, decide_true, if_true, if_pos hlt]
    exact hvRaw (SN + j.val)
  · simp only [hlt, decide_false, if_false, if_neg hlt, Bool.false_eq_true]

set_option maxHeartbeats 1600000 in
/-- Launch-anchored `qk` masked-load recipe (the `-inf`-defaulted logits
load), with the row base `sl`, trip value `S` and raw logits `qRaw`
decoupled from the running state. -/
private theorem srIO_qk_eval (BLOCK_N : Nat) (u : BlockState) (Logics : RegionName)
    (ch sl SN S slh slb : Nat) (qRaw : Nat → ℝ)
    (hch : u.regs .nat [] "cur_head" = some (Tile.scalar ch))
    (hsl : u.regs .nat [] "cur_batch_start_loc" = some (Tile.scalar sl))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : u.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hseq : u.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar S))
    (hqRaw : ∀ n : Nat, u.readMem Logics (ch * slh + (sl + n) * slb) = qRaw n) :
    evalOp (Op.load .real
        (MemAccess.region Logics
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat slh))
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_start_loc")
                  (Op.ref .nat [] "start_n"))
                (Op.ref .nat [BLOCK_N] "offs_n"))
              (Op.constNat slb))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.ref .nat [BLOCK_N] "offs_n"))
            (Op.ref .nat [] "cur_batch_seq_len"))
          (Op.negInf.broadcast [BLOCK_N]))) u
      = some (⟨fun idx : TileIndex [BLOCK_N] =>
          if SN + idx.1.val < S then some (qRaw (SN + idx.1.val))
          else (⊥ : WithBot ℝ)⟩ : Tile .real [BLOCK_N]) := by
  simp only [evalOp, hch, hsl, hsn, hn, hseq, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, uu⟩ := idx
  simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, Broadcast.leftIndex,
    Broadcast.rightIndex, BlockState.readMemValue_real, Region.cast_id]
  have haddr : ch * slh + (sl + SN + j.val) * slb = ch * slh + (sl + (SN + j.val)) * slb := by
    ring
  by_cases hlt : SN + j.val < S
  · simp only [hlt, decide_true, if_true, if_pos hlt]
    rw [haddr, hqRaw (SN + j.val)]
  · simp only [hlt, decide_false, if_false, if_neg hlt, Bool.false_eq_true]
    rfl

/-- Launch-anchored paged-`V` load recipe: `sr_v_gather_evalG` with the
loaded tile re-expressed through a caller-supplied `vFn` (so the walk can
anchor it at the launch state). -/
private theorem srIO_vload_eval (BLOCK_N BLOCK_DMODEL : Nat) (u : BlockState)
    (V : RegionName) (svbs : Nat) (offFn : Fin BLOCK_DMODEL → Nat)
    (vIdxFn : Fin BLOCK_N → Int)
    (vFn : Fin BLOCK_N → Fin BLOCK_DMODEL → WithBot ℝ)
    (hvp : u.regs .ptr [1, BLOCK_DMODEL] "v_ptrs" =
      some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] => (V, offFn idx.2.1)⟩
        : Tile .ptr [1, BLOCK_DMODEL]))
    (hvi : u.regs .int [BLOCK_N] "v_index" =
      some (⟨fun idx : TileIndex [BLOCK_N] => vIdxFn idx.1⟩ : Tile .int [BLOCK_N]))
    (hvFn : ∀ (jL : Fin BLOCK_N) (d : Fin BLOCK_DMODEL),
      (some (u.readMem V ((vIdxFn jL * svbs).toNat + offFn d)) : WithBot ℝ) = vFn jL d) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consR.consL
            (Op.ref .ptr [1, BLOCK_DMODEL] "v_ptrs")
            (Op.mul .int Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .int [BLOCK_N] "v_index"))
                (Op.constNat svbs).castNatToInt).castIntToNat))
        MaskOpt.none) u
      = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] => vFn idx.1 idx.2.1⟩
          : Tile .real [BLOCK_N, BLOCK_DMODEL]) := by
  rw [sr_v_gather_evalG BLOCK_N BLOCK_DMODEL u V svbs offFn vIdxFn hvp hvi]
  refine congrArg some (Tile.ext (fun idx => ?_))
  obtain ⟨j, d, uu⟩ := idx
  exact hvFn j d

/-- Weak (safety-walk) invariant: exact pins for the address-bearing
registers (all anchored to the launch state `s` through `readMemValue`, so
they survive the whole run), **existential** pins for the three fold
registers `e_max` / `e_sum` / `acc` (the walk never needs their values, only
their shapes). -/
private def srIOSafeInvW (V : RegionName) (BStartLoc BSeqLen : RegionName)
    (mil svh svd sb ss BLOCK_DMODEL BLOCK_N : Nat) (s s' : BlockState) : Prop :=
  s'.mem = s.mem
  ∧ s'.pids = s.pids
  ∧ s'.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 0))
  ∧ s'.regs .nat [] "cur_head" = some (Tile.scalar (s.pids 1))
  ∧ s'.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (srSeqLen s BSeqLen))
  ∧ s'.regs .nat [] "cur_batch_start_loc"
      = some (Tile.scalar (srStartLoc s BStartLoc))
  ∧ s'.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
  ∧ s'.regs .nat [BLOCK_DMODEL] "offs_d"
      = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))
  ∧ s'.regs .nat [] "off_b_loc"
      = some (Tile.scalar (srOffBLocG s BSeqLen mil sb ss))
  ∧ s'.regs .ptr [1, BLOCK_DMODEL] "v_ptrs" =
      some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] =>
        (V, s.pids 1 * svh + idx.2.1.val * svd)⟩ : Tile .ptr [1, BLOCK_DMODEL])
  ∧ (∃ em : WithBot ℝ, s'.regs .real [] "e_max" = some (Tile.scalar em))
  ∧ (∃ es : WithBot ℝ, s'.regs .real [] "e_sum" = some (Tile.scalar es))
  ∧ (∃ ac : Fin BLOCK_DMODEL → WithBot ℝ, s'.regs .real [BLOCK_DMODEL] "acc"
      = some (⟨fun idx : TileIndex [BLOCK_DMODEL] => ac idx.1⟩
        : Tile .real [BLOCK_DMODEL]))

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **Weak preLoop walk** (single pass): from an **arbitrary** launch state
the 12 preLoop statements are trace-safe (the two `.nat` slot loads bounded
by their slot windows; everything else is register-only) and step to a state
satisfying `srIOSafeInvW`. -/
private theorem srIO_preLoopW (R : RoundingModel) (bounds : RegionBounds)
    (V : RegionName) (BStartLoc BSeqLen : Region .nat)
    (mil svh svd sb ss BLOCK_DMODEL BLOCK_N : Nat) (s : BlockState)
    (hbSQ : s.pids 0 < bounds (Region.cast BSeqLen))
    (hbSL : s.pids 0 < bounds (Region.cast BStartLoc)) :
    Stmt.TraceSafeListR R bounds
        (srPreLoopG V BStartLoc BSeqLen mil svh svd sb ss BLOCK_DMODEL BLOCK_N) s
      ∧ ∃ s0, stepStmtsR R
            (srPreLoopG V BStartLoc BSeqLen mil svh svd sb ss BLOCK_DMODEL BLOCK_N) s
              = some s0
          ∧ srIOSafeInvW V BStartLoc.cast BSeqLen.cast mil svh svd sb ss
              BLOCK_DMODEL BLOCK_N s s0 := by
  unfold srPreLoopG
  -- stmt 0: cur_batch = programId 0
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.programId 0) s = some (Tile.scalar (s.pids 0)) from
        evalOp_programId 0 s)) ?_
  -- stmt 1: cur_head = programId 1
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.programId 1) _ = some (Tile.scalar (s.pids 1)) from
        evalOp_programId 1 _)) ?_
  -- stmt 2: cur_batch_seq_len = load(BSeqLen + cur_batch)  [slot-0 window]
  refine srIO_walkCons ?_
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.load .nat (MemAccess.region BSeqLen (Op.ref .nat [] "cur_batch"))
            MaskOpt.none) _
          = some (Tile.scalar (srSeqLen s BSeqLen.cast)) from by
        simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
          BlockState.setReg_ne_name, Option.bind, Option.pure_def, srSeqLen]
        rfl)) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR], ?_⟩
    intro offsets hoff idx _
    rw [evalOpR_ref] at hoff
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same] at hoff
    obtain rfl := Option.some.inj hoff
    exact hbSQ
  -- stmt 3: cur_batch_start_loc = load(BStartLoc + cur_batch)  [slot-1 window]
  refine srIO_walkCons ?_
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.load .nat
            (MemAccess.region BStartLoc (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
          = some (Tile.scalar (srStartLoc s BStartLoc.cast)) from by
        simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
          BlockState.setReg_ne_name, Option.bind, Option.pure_def, srStartLoc]
        rfl)) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR], ?_⟩
    intro offsets hoff idx _
    rw [evalOpR_ref] at hoff
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same] at hoff
    obtain rfl := Option.some.inj hoff
    exact hbSL
  -- stmt 4: offs_n = arange BLOCK_N
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.arange BLOCK_N) _
          = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) from
        evalOp_arange BLOCK_N _)) ?_
  -- stmt 5: offs_d = arange BLOCK_DMODEL
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.arange BLOCK_DMODEL) _
          = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val)) from
        evalOp_arange BLOCK_DMODEL _)) ?_
  -- stmt 6: off_v
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat svh))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
              (Op.constNat svd))) _
          = some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] =>
              s.pids 1 * svh + idx.2.1.val * svd⟩ : Tile .nat [1, BLOCK_DMODEL]) from
        sr_offv_evalG BLOCK_DMODEL _ (s.pids 1) svh svd (by simp)
          (by simp [Tile.vec]))) ?_
  -- stmt 7: off_b_loc
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat sb))
            (Op.mul .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil (Op.constNat mil)
                (Op.ref .nat [] "cur_batch_seq_len"))
              (Op.constNat ss))) _
          = some (Tile.scalar (srOffBLocG s BSeqLen.cast mil sb ss)) from by
        rw [evalOp_add, evalOp_mul, evalOp_mul, evalOp_sub]
        simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same,
          BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          BlockState.setReg_pids, Option.bind_eq_bind, Option.bind_some,
          srOffBLocG, srSeqLen]
        refine congrArg some ?_
        ext idx
        simp only [Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex,
          Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
          NumericDType.sub])) ?_
  -- stmt 8: v_ptrs = V + off_v
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V)
            (Op.ref .nat [1, BLOCK_DMODEL] "off_v")) _
          = some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] =>
              (V, s.pids 1 * svh + idx.2.1.val * svd)⟩
              : Tile .ptr [1, BLOCK_DMODEL]) from by
        simp only [evalOp, evalOp_ref, BlockState.setReg_same,
          BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          BlockState.setReg_pids, Option.bind]
        refine congrArg some (Tile.ext (fun idx => ?_))
        obtain ⟨z, e, uu⟩ := idx
        simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex,
          Broadcast.rightIndex, Region.cast_id, Nat.zero_add, Prod.mk.injEq,
          true_and])) ?_
  -- stmt 9: e_max = -inf
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp Op.negInf _ = some (Tile.scalar (⊥ : WithBot ℝ)) from by
        simp [evalOp_negInf]; rfl)) ?_
  -- stmt 10: e_sum = 0.0
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.const 0.0) _ = some (Tile.scalar (some (0 : ℝ))) from by
        simp only [evalOp_const]; norm_num)) ?_
  -- stmt 11: acc = zeros
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.full [BLOCK_DMODEL] (Op.const 0)) _
          = some (⟨fun _ : TileIndex [BLOCK_DMODEL] => some (0 : ℝ)⟩
              : Tile .real [BLOCK_DMODEL]) from by
        simp [evalOp_full, evalOp_const])) ?_
  refine srIO_walkNil _ ?_
  refine ⟨?_, by simp, by simp, by simp, by simp, by simp, by simp [Tile.vec],
    by simp [Tile.vec], by simp, by simp, ⟨⊥, by simp⟩, ⟨some (0 : ℝ), by simp⟩,
    ⟨fun _ => some (0 : ℝ), by simp⟩⟩
  funext rg o
  simp

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **Weak loop-body walk**: one iteration at index `c` is trace-safe (the
`B_Loc` gather and the `Logics` load bounded on their live lanes, the
**unmasked** paged-`V` load bounded on *every* lane — sentinel-substituted
addresses included) and steps to a state that re-establishes
`srIOSafeInvW`. -/
private theorem srIO_bodyW (R : RoundingModel) (bounds : RegionBounds)
    (Logics V : RegionName) (BLoc : Region .int) (BStartLoc BSeqLen : RegionName)
    (mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N : Nat)
    (hN : 0 < BLOCK_N) (other_kv_index : Int)
    (s stt : BlockState) (c : Nat)
    (hP : srIOSafeInvW V BStartLoc BSeqLen mil svh svd sb ss BLOCK_DMODEL BLOCK_N s stt)
    (hbG : ∀ jL : Fin BLOCK_N, c + jL.val < srSeqLen s BSeqLen →
      srOffBLocG s BSeqLen mil sb ss + (c + jL.val) * ss
        < bounds (Region.cast BLoc))
    (hbL : ∀ jL : Fin BLOCK_N, c + jL.val < srSeqLen s BSeqLen →
      s.pids 1 * slh + (srStartLoc s BStartLoc + (c + jL.val)) * slb < bounds Logics)
    (hbV : ∀ (jL : Fin BLOCK_N) (d : Fin BLOCK_DMODEL),
      ((if c + jL.val < srSeqLen s BSeqLen
            then s.readMemValue .int (Region.cast BLoc)
              (srOffBLocG s BSeqLen mil sb ss + (c + jL.val) * ss)
            else other_kv_index) * svbs).toNat + (s.pids 1 * svh + d.val * svd)
        < bounds V) :
    Stmt.TraceSafeListR R bounds
        (srLoopBodyG Logics BLoc slh slb svbs ss BLOCK_DMODEL BLOCK_N other_kv_index)
        (stt.setReg "start_n" .nat [] (Tile.scalar c))
      ∧ ∃ s', stepStmtsR R
            (srLoopBodyG Logics BLoc slh slb svbs ss BLOCK_DMODEL BLOCK_N
              other_kv_index)
            (stt.setReg "start_n" .nat [] (Tile.scalar c)) = some s'
          ∧ srIOSafeInvW V BStartLoc BSeqLen mil svh svd sb ss BLOCK_DMODEL
              BLOCK_N s s' := by
  obtain ⟨hmem, hpids, hcb, hch, hseq, hsl, hn, hd, hoff, hvp, ⟨em0, hemax⟩,
    ⟨es0, hesum⟩, ⟨ac0, hacc⟩⟩ := hP
  haveI hNe : Nonempty (Fin BLOCK_N) := ⟨⟨0, hN⟩⟩
  have hne : (Finset.univ : Finset (Fin BLOCK_N)).Nonempty := Finset.univ_nonempty
  set S := srSeqLen s BSeqLen with hSdef
  set off := srOffBLocG s BSeqLen mil sb ss with hoffdef
  set sl0 := srStartLoc s BStartLoc with hsl0def
  set vIdxFn : Fin BLOCK_N → Int := fun jL =>
    if c + jL.val < S then
      s.readMemValue .int (Region.cast BLoc) (off + (c + jL.val) * ss)
    else other_kv_index with hvIdxFn
  set qkFn : Fin BLOCK_N → WithBot ℝ := fun jL =>
    if c + jL.val < S then
      some (s.readMem Logics (s.pids 1 * slh + (sl0 + (c + jL.val)) * slb))
    else (⊥ : WithBot ℝ) with hqkFn
  set nemaxVal := em0 ⊔ Finset.univ.sup' hne (fun k : Fin BLOCK_N => qkFn k)
    with hnemaxVal
  set αVal := WithBot.realExp (WithBot.realSub em0 nemaxVal) with hαVal
  set pFn : Fin BLOCK_N → WithBot ℝ := fun jL =>
    WithBot.realExp (WithBot.realSub (qkFn jL) nemaxVal) with hpFn
  set esumNew := WithBot.realAdd (WithBot.realMul es0 αVal)
    (∑ j : Fin BLOCK_N, pFn j) with hesumNew
  set vFn : Fin BLOCK_N → Fin BLOCK_DMODEL → WithBot ℝ := fun jL d =>
    some (s.readMem V ((vIdxFn jL * svbs).toNat + (s.pids 1 * svh + d.val * svd)))
    with hvFn
  set accNew : Fin BLOCK_DMODEL → WithBot ℝ := fun d =>
    WithBot.realAdd (WithBot.realMul (ac0 d) αVal)
      (∑ j : Fin BLOCK_N, WithBot.realMul (pFn j) (vFn j d)) with haccNew
  unfold srLoopBodyG
  -- stmt 0: start_n = tl.multiple_of(start_n, BLOCK_N)  (erased: self-assign)
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.ref .nat [] "start_n") _ = some (Tile.scalar c) from by
        rw [evalOp_ref, BlockState.setReg_same])) ?_
  -- stmt 1: v_index = masked B_Loc gather   [gread window]
  refine srIO_walkCons ?_
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (srIO_vindex_eval BLOCK_N _ BLoc off c S ss other_kv_index
        (fun n => s.readMemValue .int (Region.cast BLoc) (off + n * ss))
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hoff)
        (by simp only [BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hn)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hseq)
        (fun n => srIO_readMemValue_of_mem _ s
          (by funext rg o
              simp only [BlockState.setReg_mem]
              exact congrFun (congrFun hmem rg) o)
          (Region.cast BLoc) (off + n * ss)))) ?_
  · -- safety: the gather's live lanes are in bounds
    simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def],
      ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def]⟩, ?_⟩
    intro offsets hoffs idx hactive
    rw [srIO_blocOffsR_eval R _ off c ss
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true]
          exact hoff)
      (by simp only [BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true]
          exact hn)] at hoffs
    obtain rfl := Option.some.inj hoffs
    obtain ⟨masks, hmask, hactl⟩ := hactive
    rw [srIO_liveMaskR_eval R _ c S
      (by simp only [BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true]
          exact hn)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true]
          exact hseq)] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨jL, uu⟩ := idx
    have hlive : c + jL.val < S := by simpa using hactl
    exact hbG jL hlive
  -- stmt 2: qk = masked Logics load   [read1 window]
  refine srIO_walkCons ?_
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (srIO_qk_eval BLOCK_N _ Logics (s.pids 1) sl0 c S slh slb
        (fun n => s.readMem Logics (s.pids 1 * slh + (sl0 + n) * slb))
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hch)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hsl)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true, BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hn)
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hseq)
        (fun n => srIO_readMem_of_mem _ s
          (by funext rg o
              simp only [BlockState.setReg_mem]
              exact congrFun (congrFun hmem rg) o)
          Logics (s.pids 1 * slh + (sl0 + n) * slb)))) ?_
  · -- safety: the logits load's live lanes are in bounds
    simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def],
      ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def]⟩, ?_⟩
    intro offsets hoffs idx hactive
    rw [srIO_qkOffsR_eval R _ (s.pids 1) sl0 c slh slb
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true]
          exact hch)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true]
          exact hsl)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true, BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true]
          exact hn)] at hoffs
    obtain rfl := Option.some.inj hoffs
    obtain ⟨masks, hmask, hactl⟩ := hactive
    rw [srIO_liveMaskR_eval R _ c S
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true, BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true]
          exact hn)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true]
          exact hseq)] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨jL, uu⟩ := idx
    have hlive : c + jL.val < S := by simpa using hactl
    have hb := hbL jL hlive
    rw [show sl0 + (c + jL.val) = sl0 + c + jL.val from by ring] at hb
    exact hb
  -- stmt 3: n_e_max
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (sr_nemax_evalG BLOCK_N hN _ qkFn em0
        (by simp only [hqkFn, BlockState.setReg_same])
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hemax)
        hne)) ?_
  -- stmt 4: old_scale
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (sr_oldscale_eval _ em0 nemaxVal
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hemax)
        (by simp only [hnemaxVal, BlockState.setReg_same]))) ?_
  -- stmt 5: p
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (sr_p_evalG BLOCK_N _ qkFn nemaxVal
        (by simp only [hqkFn, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true, BlockState.setReg_same])
        (by simp only [hnemaxVal, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true, BlockState.setReg_same]))) ?_
  -- stmt 6: e_sum
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (sr_esum_evalG BLOCK_N _ es0 αVal pFn
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hesum)
        (by simp only [hαVal, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true, BlockState.setReg_same])
        (by simp only [hpFn, BlockState.setReg_same]))) ?_
  -- stmt 7: v = unmasked paged gather   [read2 window, EVERY lane]
  refine srIO_walkCons ?_
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (srIO_vload_eval BLOCK_N BLOCK_DMODEL _ V svbs
        (fun d : Fin BLOCK_DMODEL => s.pids 1 * svh + d.val * svd) vIdxFn vFn
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hvp)
        (by simp only [hvIdxFn, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true, BlockState.setReg_same])
        (fun jL d => by
          simp only [hvFn]
          exact congrArg some (srIO_readMem_of_mem _ s
            (by funext rg o
                simp only [BlockState.setReg_mem]
                exact congrFun (congrFun hmem rg) o)
            V ((vIdxFn jL * svbs).toNat + (s.pids 1 * svh + d.val * svd)))))) ?_
  · -- safety: EVERY lane of the unmasked paged load, sentinel lanes included
    simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR], ?_⟩
    intro ptrs hptrs idx _
    rw [srIO_vPtrR_eval R BLOCK_N BLOCK_DMODEL _ V svbs
      (fun d : Fin BLOCK_DMODEL => s.pids 1 * svh + d.val * svd) vIdxFn
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true]
          exact hvp)
      (by simp only [hvIdxFn, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true, BlockState.setReg_same])] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨jL, d, uu⟩ := idx
    simp only [hvIdxFn]
    exact hbV jL d
  -- stmt 8: acc
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (sr_acc_evalG BLOCK_N BLOCK_DMODEL _ αVal ac0 pFn vFn
        (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true]
            exact hacc)
        (by simp only [hαVal, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true, BlockState.setReg_same])
        (by simp only [hpFn, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
              not_false_eq_true, BlockState.setReg_same])
        (by simp only [BlockState.setReg_same]))) ?_
  -- stmt 9: e_max = n_e_max
  refine srIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (srIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.ref .real [] "n_e_max") _ = some (Tile.scalar nemaxVal) from by
        rw [evalOp_ref]
        simp only [hnemaxVal, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, BlockState.setReg_same])) ?_
  refine srIO_walkNil _ ?_
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ⟨nemaxVal, by simp only [BlockState.setReg_same]⟩,
    ⟨esumNew, by simp only [hesumNew, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true, BlockState.setReg_same]⟩,
    ⟨accNew, by simp only [haccNew, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true, BlockState.setReg_same]⟩⟩
  · funext rg o
    simp only [BlockState.setReg_mem]
    exact congrFun (congrFun hmem rg) o
  · simp only [BlockState.setReg_pids]
    exact hpids
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    exact hcb
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    exact hch
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    exact hseq
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    exact hsl
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    exact hn
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    exact hd
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    exact hoff
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    exact hvp

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **The safety walk**: preLoop ++ the dynamic loop (by the
`forRangeTraceSafeR` invariant principle at `srIOSafeInvW`, with the
`pre`-forced budget `T` supplying every live step's window bound) ++ the
postLoop's deterministic 4-statement tail. -/
private theorem srIO_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat)
    (mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N : Nat)
    (hN : 0 < BLOCK_N) (other_kv_index : Int) (s : BlockState)
    (hSle : srSeqLen s BSeqLen.cast ≤ mil)
    (hSmod : srSeqLen s BSeqLen.cast % BLOCK_N = 0)
    (hbSQ : s.pids 0 < bounds (Region.cast BSeqLen))
    (hbSL : s.pids 0 < bounds (Region.cast BStartLoc))
    (hbG : ∀ (t : Fin (srIOT mil BLOCK_N)) (jL : Fin BLOCK_N),
      t.val * BLOCK_N + jL.val < srSeqLen s BSeqLen.cast →
      srOffBLocG s BSeqLen.cast mil sb ss + (t.val * BLOCK_N + jL.val) * ss
        < bounds (Region.cast BLoc))
    (hbL : ∀ (t : Fin (srIOT mil BLOCK_N)) (jL : Fin BLOCK_N),
      t.val * BLOCK_N + jL.val < srSeqLen s BSeqLen.cast →
      s.pids 1 * slh
          + (srStartLoc s BStartLoc.cast + (t.val * BLOCK_N + jL.val)) * slb
        < bounds Logics)
    (hbV : ∀ (t : Fin (srIOT mil BLOCK_N)) (jL : Fin BLOCK_N) (d : Fin BLOCK_DMODEL),
      ((if t.val * BLOCK_N + jL.val < srSeqLen s BSeqLen.cast
            then s.readMemValue .int (Region.cast BLoc)
              (srOffBLocG s BSeqLen.cast mil sb ss
                + (t.val * BLOCK_N + jL.val) * ss)
            else other_kv_index) * svbs).toNat + (s.pids 1 * svh + d.val * svd)
        < bounds V)
    (hbO : ∀ i : Fin BLOCK_DMODEL,
      s.pids 0 * sob + s.pids 1 * soh + i.val * sod < bounds Out) :
    ((softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N
      other_kv_index).toAlgKernel).TraceSafeR R bounds s := by
  -- shared per-iteration body handler: the live step index is `cc / BLOCK_N`
  have hbody : ∀ (cc : Nat) (stt : BlockState), cc < srSeqLen s BSeqLen.cast →
      (srIOSafeInvW V BStartLoc.cast BSeqLen.cast mil svh svd sb ss BLOCK_DMODEL
          BLOCK_N s stt ∧ cc % BLOCK_N = 0) →
      Stmt.TraceSafeListR R bounds
          (srLoopBodyG Logics BLoc slh slb svbs ss BLOCK_DMODEL BLOCK_N
            other_kv_index)
          (stt.setReg "start_n" .nat [] (Tile.scalar cc))
        ∧ ∃ s', stepStmtsR R
              (srLoopBodyG Logics BLoc slh slb svbs ss BLOCK_DMODEL BLOCK_N
                other_kv_index)
              (stt.setReg "start_n" .nat [] (Tile.scalar cc)) = some s'
            ∧ (srIOSafeInvW V BStartLoc.cast BSeqLen.cast mil svh svd sb ss
                BLOCK_DMODEL BLOCK_N s s' ∧ (cc + BLOCK_N) % BLOCK_N = 0) := by
    intro cc stt hcc hPP
    obtain ⟨hPinv, hPmod⟩ := hPP
    have hceq : cc / BLOCK_N * BLOCK_N = cc :=
      Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hPmod)
    have hcT : cc / BLOCK_N < srIOT mil BLOCK_N :=
      srIO_step_lt cc (srSeqLen s BSeqLen.cast) mil BLOCK_N hN hcc hSle hSmod
    obtain ⟨hsafeB, s', hrunB, hInvB⟩ :=
      srIO_bodyW R bounds Logics V BLoc BStartLoc.cast BSeqLen.cast
        mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N hN other_kv_index
        s stt cc hPinv
        (fun jL hlive => by
          have h := hbG ⟨cc / BLOCK_N, hcT⟩ jL (by rw [hceq]; exact hlive)
          rwa [hceq] at h)
        (fun jL hlive => by
          have h := hbL ⟨cc / BLOCK_N, hcT⟩ jL (by rw [hceq]; exact hlive)
          rwa [hceq] at h)
        (fun jL d => by
          have h := hbV ⟨cc / BLOCK_N, hcT⟩ jL d
          rwa [hceq] at h)
    exact ⟨hsafeB, s', hrunB, hInvB, by rw [Nat.add_mod_right]; exact hPmod⟩
  unfold Kernel.TraceSafeR
  rw [srBody_splitG]
  obtain ⟨hsafePre, s0, hrun0, hInv0⟩ :=
    srIO_preLoopW R bounds V BStartLoc BSeqLen mil svh svd sb ss BLOCK_DMODEL
      BLOCK_N s hbSQ hbSL
  have hInv0' := hInv0
  obtain ⟨hmem0, hpids0, hcb0, hch0, hseq0, hsl0, hn0, hd0, hoff0, hvp0, hem0,
    hes0, hac0⟩ := hInv0
  have hstopExact : evalOp (Op.ref .nat [] "cur_batch_seq_len") s0
      = some (Tile.scalar (srSeqLen s BSeqLen.cast)) := by
    rw [evalOp_ref]; exact hseq0
  refine Stmt.TraceSafeListR.append_intro _ _ hsafePre ?_
  intro s1 hs1
  rw [hrun0] at hs1
  obtain rfl := Option.some.inj hs1
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s2 hs2 => ?_)
  · -- trace safety of the `forRangeDyn` itself
    simp only [Stmt.TraceSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
      by simp [Op.SafeAtR.eq_def], ?_⟩
    rw [srIO_evalOpR_constNat, srIO_evalOpR_constNat, srIO_stopOpR_castFree,
      hstopExact]
    refine Stmt.forRangeTraceSafeR_inv R bounds "start_n" _ _
      (srLoopBodyG Logics BLoc slh slb svbs ss BLOCK_DMODEL BLOCK_N other_kv_index)
      (fun i stt => srIOSafeInvW V BStartLoc.cast BSeqLen.cast mil svh svd sb ss
        BLOCK_DMODEL BLOCK_N s stt ∧ i % BLOCK_N = 0)
      ?_ _ s0 ⟨hInv0', Nat.zero_mod BLOCK_N⟩
    intro cc stt hcc hPP
    exact hbody cc stt hcc hPP
  · -- the loop's actual successor, then the postLoop
    obtain ⟨finalC, sL, hLoopExact, hfin, hPL⟩ :=
      forRangeDyn_inv (idx := "start_n") (startOp := Op.constNat 0)
        (stopOp := Op.ref .nat [] "cur_batch_seq_len")
        (stepOp := Op.constNat BLOCK_N)
        (P := fun i stt => srIOSafeInvW V BStartLoc.cast BSeqLen.cast mil svh svd
          sb ss BLOCK_DMODEL BLOCK_N s stt ∧ i % BLOCK_N = 0)
        (s_init := s0)
        (evalOp_constNat 0 s0) hstopExact (evalOp_constNat BLOCK_N s0)
        hN.ne'
        ⟨hInv0', Nat.zero_mod BLOCK_N⟩
        (fun i stt hi hP => by
          obtain ⟨hsafeB, s', hrunB, hInvB⟩ := hbody i stt hi hP
          exact ⟨s', by
            rw [← srIO_body_castFree R Logics BLoc slh slb svbs ss BLOCK_DMODEL
              BLOCK_N other_kv_index]
            exact hrunB, hInvB⟩)
    rw [srIO_dyn_castFree R Logics BLoc slh slb svbs ss BLOCK_DMODEL BLOCK_N
      other_kv_index s0, hLoopExact] at hs2
    obtain rfl := Option.some.inj hs2
    obtain ⟨⟨hmemL, hpidsL, hcbL, hchL, hseqL, hslL, hnL, hdL, hoffL, hvpL, hemL,
      hesL, hacL⟩, hmodL⟩ := hPL
    -- postLoop: `acc /= e_sum`, `off_o`, `out_ptrs`, the unmasked terminal store
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s3 hs3 => ?_)
    obtain ⟨v3, hv3, rfl⟩ := stepStmtR_assign_inv hs3
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s4 hs4 => ?_)
    obtain ⟨v4, hv4, rfl⟩ := stepStmtR_assign_inv hs4
    rw [srIO_offoR_eval R _ (s.pids 0) (s.pids 1) sob soh sod
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true]
          exact hcbL)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true]
          exact hchL)
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true]
          exact hdL)] at hv4
    obtain rfl := Option.some.inj hv4
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s5 hs5 => ?_)
    obtain ⟨v5, hv5, rfl⟩ := stepStmtR_assign_inv hs5
    rw [srIO_outPtrR_eval R _ Out
      (fun i : Fin BLOCK_DMODEL =>
        s.pids 0 * sob + s.pids 1 * soh + i.val * sod)
      (by simp only [BlockState.setReg_same])] at hv5
    obtain rfl := Option.some.inj hv5
    refine Stmt.TraceSafeListR.cons_intro ?_
      (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR,
      MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [MemAccess.SafeAtR, Op.SafeAtR.eq_def],
      by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR], ?_⟩
    intro ptrs hptrs idx _
    rw [evalOpR_ref] at hptrs
    simp only [BlockState.setReg_same] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨i, uu⟩ := idx
    exact hbO i

/-! ## The rounded Hoare triple (`hrun`): framed exact run

`hrun` rides the exact `srPreLoopG_eval` → `forRangeDyn_inv` →
`srPostLoopG_eval` stack unchanged (everything is cast-free, so `execR R`
collapses onto it verbatim); the only new obligation the skin adds over the
existing headline is the **memory frame**, recovered by replaying the
deterministic 4-statement postLoop and framing its scatter. -/

/-- A `writeMem` scatter `foldl` leaves every cell not hit by a lane
untouched (the store is unmasked, so there is no active-lane guard). -/
private theorem srIO_foldl_writeMem_frame {α : Type} (region : RegionName)
    (offFn : α → Nat) (valFn : α → ℝ) :
    ∀ (l : List α) (st : BlockState) (r : RegionName) (o : Nat),
      (r = region → ∀ k ∈ l, offFn k ≠ o) →
      ((l.foldl (fun acc k => acc.writeMem region (offFn k) (valFn k)) st).mem r o
        = st.mem r o)
  | [], _, _, _, _ => rfl
  | k :: rest, st, r, o, h => by
      rw [List.foldl_cons,
        srIO_foldl_writeMem_frame region offFn valFn rest _ r o
          (fun hr k' hk' => h hr k' (List.mem_cons_of_mem _ hk')),
        BlockState.writeMem_mem]
      rw [if_neg (fun hro => h hro.1 k List.mem_cons_self hro.2.symm)]

set_option maxHeartbeats 3200000 in
set_option maxRecDepth 8000 in
/-- **PostLoop frame**: every cell outside the terminal store's footprint is
untouched by the postLoop (replay of `srPostLoopG_eval`'s deterministic
4-statement chain, framed). -/
private theorem srIO_postLoop_frame
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : RegionName)
    (mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N : Nat)
    (hD : 0 < BLOCK_DMODEL) (s0 : BlockState) (c : Nat) (sL sP : BlockState)
    (hinv : srInvariantG Logics V BLoc BStartLoc BSeqLen mil slh slb svbs svh svd
      sb ss BLOCK_DMODEL BLOCK_N s0 c sL)
    (hpost : stepStmts (srPostLoopG Out sob soh sod BLOCK_DMODEL) sL = some sP) :
    ∀ r o, (r = Out → ∀ i : Fin BLOCK_DMODEL,
        o ≠ outOffsetG s0 sob soh sod i) →
      sP.mem r o = sL.mem r o := by
  set qk := srQkFG s0 Logics BStartLoc BSeqLen slh slb with hqkdef
  set vF := srVFG s0 V BLoc BSeqLen mil sb ss svbs svh svd BLOCK_DMODEL with hvFdef
  simp only [srInvariantG, ← hqkdef, ← hvFdef] at hinv
  obtain ⟨hpids, hmem, hundef, hcb, hch, hseq, hsl, hn, hd, hoff, hvp, hemaxA,
    hesumA, hacc⟩ := hinv
  have hesum := hesumA hD
  set denomVal := ((srStateBot qk vF (c * BLOCK_N) (⟨0, hD⟩ : Fin BLOCK_DMODEL)).2.1
    : WithBot ℝ) with hdenomVal
  set accFn : Fin BLOCK_DMODEL → WithBot ℝ := fun d =>
    ((srStateBot qk vF (c * BLOCK_N) d).2.2 : WithBot ℝ) with haccFn
  unfold srPostLoopG at hpost
  -- stmt 0: acc = acc / e_sum
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.div .real Broadcast.scalarR
          (Op.ref .real [BLOCK_DMODEL] "acc") (Op.ref .real [] "e_sum")) sL
        = some (⟨fun idx : TileIndex [BLOCK_DMODEL] =>
            WithBot.realDiv (accFn idx.1) denomVal⟩ : Tile .real [BLOCK_DMODEL]) from by
      rw [evalOp_div]
      simp only [evalOp_ref, hacc, hesum, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.div, haccFn, hdenomVal]))] at hpost
  set s1 := sL.setReg "acc" .real [BLOCK_DMODEL]
    (⟨fun idx : TileIndex [BLOCK_DMODEL] => WithBot.realDiv (accFn idx.1) denomVal⟩
      : Tile .real [BLOCK_DMODEL]) with hs1
  have e1 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc" → sL.regs dt sh nm = some t → s1.regs dt sh nm = some t := by
    intro dt sh nm t hne' h
    rw [hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne']; exact h
  set offoTile : Tile .nat [BLOCK_DMODEL] :=
    ⟨fun idx : TileIndex [BLOCK_DMODEL] => outOffsetG s0 sob soh sod idx.1⟩
    with hoffoTile
  -- stmt 1: off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat sob))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat soh)))
          (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_DMODEL] "offs_d")
            (Op.constNat sod))) s1 = some offoTile from by
      rw [evalOp_add, evalOp_add, evalOp_mul, evalOp_mul, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, e1 (by decide) hcb, e1 (by decide) hch,
        e1 (by decide) hd, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [hoffoTile, Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add,
        NumericDType.mul, outOffsetG]))] at hpost
  set s2 := s1.setReg "off_o" .nat [BLOCK_DMODEL] offoTile with hs2
  -- stmt 2: out_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
          (Op.ref .nat [BLOCK_DMODEL] "off_o")) s2
        = some (⟨fun idx : TileIndex [BLOCK_DMODEL] => (Out, offoTile.data idx)⟩
            : Tile .ptr [BLOCK_DMODEL]) from by
      simp only [evalOp, evalOp_ref, hs2, BlockState.setReg_same, Option.bind]
      refine congrArg some (Tile.ext (fun idx => ?_))
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex,
        Broadcast.rightIndex, Region.cast_id, Nat.zero_add, Prod.mk.injEq,
        true_and]))] at hpost
  set s3 := s2.setReg "out_ptrs" .ptr [BLOCK_DMODEL]
    (⟨fun idx : TileIndex [BLOCK_DMODEL] => (Out, offoTile.data idx)⟩
      : Tile .ptr [BLOCK_DMODEL]) with hs3
  have hs3acc : s3.regs .real [BLOCK_DMODEL] "acc"
      = some (⟨fun idx : TileIndex [BLOCK_DMODEL] =>
          WithBot.realDiv (accFn idx.1) denomVal⟩ : Tile .real [BLOCK_DMODEL]) := by
    rw [hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hs1, BlockState.setReg_same]
  have hs3ptr : s3.regs .ptr [BLOCK_DMODEL] "out_ptrs"
      = some (⟨fun idx : TileIndex [BLOCK_DMODEL] => (Out, offoTile.data idx)⟩
          : Tile .ptr [BLOCK_DMODEL]) := by
    rw [hs3, BlockState.setReg_same]
  have hstore : stepStmt (Stmt.store .real [BLOCK_DMODEL]
      (MemAccess.ptr (Op.ref .ptr [BLOCK_DMODEL] "out_ptrs"))
      (Op.ref .real [BLOCK_DMODEL] "acc") MaskOpt.none) s3
      = some ((TileShape.allIndices [BLOCK_DMODEL]).foldl
          (fun acc i => acc.writeMem Out (offoTile.data i)
            ((WithBot.realDiv (accFn i.1) denomVal).unbotD 0)) s3) := by
    unfold stepStmt
    simp only [evalOp_ref, hs3acc, hs3ptr, Option.bind_eq_bind, Option.bind_some,
      Option.map_some, BlockState.writeMemTyped_real, FloatDType.real_storeValue]
    rfl
  rw [stepStmts.cons_some hstore, stepStmts.nil] at hpost
  obtain rfl := Option.some.inj hpost
  intro r o hno
  refine (srIO_foldl_writeMem_frame Out (fun i => offoTile.data i)
    (fun i => (WithBot.realDiv (accFn i.1) denomVal).unbotD 0) _ s3 r o ?_).trans ?_
  · intro hr k _
    exact fun hoff' => hno hr k.1 hoff'.symm
  · rw [hs3, hs2, hs1]
    simp only [BlockState.setReg_mem]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Framed general execution**: `sr_execG` extended with the per-cell
memory frame (this is the exact run `hrun` rides). -/
private theorem srIO_exec_framed
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat)
    (mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int) (hD : 0 < BLOCK_DMODEL) (hN : 0 < BLOCK_N)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (hseqmod : srSeqLen s BSeqLen.cast % BLOCK_N = 0)
    (hseqpos : 0 < srSeqLen s BSeqLen.cast)
    (hOutInj : ∀ s0 : BlockState,
      Function.Injective (fun i : Fin BLOCK_DMODEL => outOffsetG s0 sob soh sod i))
    (mr : ℝ)
    (hM : srRunningMax (srQkFG s Logics BStartLoc.cast BSeqLen.cast slh slb)
      (srVFG s V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL)
      (srSeqLen s BSeqLen.cast) (⟨0, hD⟩ : Fin BLOCK_DMODEL) = (mr : WithBot ℝ)) :
    ∃ sF, stepStmts ((softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
        mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N
        other_kv_index).toAlgKernel.body) s = some sF
      ∧ (∀ d : Fin BLOCK_DMODEL,
          sF.readMem Out (outOffsetG s sob soh sod d)
            = softmaxReducevWeightedSum
                (srQkFG s Logics BStartLoc.cast BSeqLen.cast slh slb) mr
                (srVFG s V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL) d)
      ∧ (∀ r o, (r ≠ Out ∨ ∀ d : Fin BLOCK_DMODEL,
            o ≠ outOffsetG s sob soh sod d) → sF.mem r o = s.mem r o) := by
  rw [srBody_splitG]
  obtain ⟨s0, hpre, hinv0'⟩ := srPreLoopG_eval s V BLoc BStartLoc BSeqLen
    mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N hundef
  have hinv0 : srInvariantG Logics V BLoc BStartLoc.cast BSeqLen.cast
      mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N s 0 s0 :=
    srInvariantG_zero_logics_irrel BSeqLen.cast Logics V BLoc BLoc BStartLoc.cast
      BSeqLen.cast mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N s s0 hinv0'
  rw [stepStmts.append_some hpre]
  obtain ⟨hpids0, hmem0, hundef0, hcb0, hch0, hseq0, hsl0, hn0, hd0, hoff0, hvp0,
    h12, h13, h14⟩ := hinv0
  set S := srSeqLen s BSeqLen.cast with hSdef
  have hseqS : srSeqLen s0 BSeqLen.cast = S := by
    rw [hSdef]; exact srSeqLen_eq_of_mem_pids s0 s BSeqLen.cast hmem0 hpids0
  have hinv0s0 : srInvariantG Logics V BLoc BStartLoc.cast BSeqLen.cast
      mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N s0 0 s0 := by
    refine srInvariantG_zero_logics_irrel Logics Logics V BLoc BLoc BStartLoc.cast
      BSeqLen.cast mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N s0 s0 ?_
    refine ⟨rfl, rfl, hundef0, ?_, ?_, ?_, ?_, hn0, hd0, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hcb0, hpids0]
    · rw [hch0, hpids0]
    · rw [hseq0, srSeqLen_eq_of_mem_pids s0 s BSeqLen.cast hmem0 hpids0]
    · rw [hsl0, srStartLoc_eq_of_mem_pids s0 s BStartLoc.cast hmem0 hpids0]
    · rw [hoff0]
      simp only [srOffBLocG, hpids0,
        srSeqLen_eq_of_mem_pids s0 s BSeqLen.cast hmem0 hpids0]
    · rw [hvp0, hpids0]
    · intro hD'; rw [h12 hD']; simp only [Nat.zero_mul, srRunningMax_zero]
    · intro hD'; rw [h13 hD']; simp only [Nat.zero_mul, srStateBot_zero]
    · rw [h14]; simp only [Nat.zero_mul, srStateBot_zero]
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    VeriTile.Triton.forRangeDyn_inv (idx := "start_n")
      (startOp := Op.constNat 0) (stopOp := Op.ref .nat [] "cur_batch_seq_len")
      (stepOp := Op.constNat BLOCK_N)
      (P := fun i st => srInvariantG Logics V BLoc BStartLoc.cast BSeqLen.cast
        mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N s0 (i / BLOCK_N) st
        ∧ i % BLOCK_N = 0 ∧ i ≤ srSeqLen s0 BSeqLen.cast)
      (s_init := s0) (start := 0) (stop := srSeqLen s0 BSeqLen.cast)
      (step := BLOCK_N)
      (by rw [evalOp_constNat])
      (by rw [evalOp_ref, hseq0, hseqS])
      (by rw [evalOp_constNat])
      hN.ne'
      ⟨by rw [Nat.zero_div]; exact hinv0s0, Nat.zero_mod _, by omega⟩
      (fun i st hi hP => by
        obtain ⟨hPinv, hPmod, hPle⟩ := hP
        obtain ⟨s', hstep, hinv'⟩ := sr_attn_stepG Logics V BLoc BStartLoc.cast
          BSeqLen.cast mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N hD hN
          other_kv_index s0 i st hi (by rw [hseqS]; exact hseqmod) hPinv
          ((Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hPmod)).symm)
        refine ⟨s', hstep, ?_, by
          rw [Nat.add_mod, hPmod, Nat.mod_self]; simp, by
          have hmod : srSeqLen s0 BSeqLen.cast % BLOCK_N = 0 := by
            rw [hseqS]; exact hseqmod
          have hstep' : (i / BLOCK_N + 1) * BLOCK_N ≤ srSeqLen s0 BSeqLen.cast := by
            have hi2 : i / BLOCK_N * BLOCK_N < srSeqLen s0 BSeqLen.cast := by
              rw [Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hPmod)]; exact hi
            have hclt : i / BLOCK_N < srSeqLen s0 BSeqLen.cast / BLOCK_N := by
              apply Nat.lt_of_mul_lt_mul_right (a := BLOCK_N)
              rw [Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)]; exact hi2
            calc (i / BLOCK_N + 1) * BLOCK_N
                ≤ (srSeqLen s0 BSeqLen.cast / BLOCK_N) * BLOCK_N :=
                  Nat.mul_le_mul_right _ hclt
              _ = srSeqLen s0 BSeqLen.cast :=
                  Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)
          rw [Nat.succ_mul] at hstep'
          have hieq : i / BLOCK_N * BLOCK_N = i :=
            Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hPmod)
          omega⟩
        rw [show (i + BLOCK_N) / BLOCK_N = i / BLOCK_N + 1 from by
          rw [Nat.add_div_right i hN]]
        exact hinv')
  rw [stepStmts.cons_some hloop]
  obtain ⟨hinvLinv, hinvLmod, hinvLle⟩ := hinvL
  have hfinS : final = srSeqLen s0 BSeqLen.cast := by
    have hmod : srSeqLen s0 BSeqLen.cast % BLOCK_N = 0 := by
      rw [hseqS]; exact hseqmod
    omega
  subst hfinS
  have hfull : (srSeqLen s0 BSeqLen.cast / BLOCK_N) * BLOCK_N
      = srSeqLen s0 BSeqLen.cast := by
    have hmod : srSeqLen s0 BSeqLen.cast % BLOCK_N = 0 := by
      rw [hseqS]; exact hseqmod
    rw [Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)]
  have hMs0 : srRunningMax (srQkFG s0 Logics BStartLoc.cast BSeqLen.cast slh slb)
      (srVFG s0 V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL)
      (srSeqLen s0 BSeqLen.cast) (⟨0, hD⟩ : Fin BLOCK_DMODEL) = (mr : WithBot ℝ) := by
    rw [srRunningMax_reindexG (S := srSeqLen s0 BSeqLen.cast)
      (S' := srSeqLen s BSeqLen.cast) (by rw [hseqS])
      (srQkFG s0 Logics BStartLoc.cast BSeqLen.cast slh slb)
      (srQkFG s Logics BStartLoc.cast BSeqLen.cast slh slb)
      (srVFG s0 V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL)
      (srVFG s V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL)
      (srSeqLen s0 BSeqLen.cast) ⟨0, hD⟩
      (fun n => by
        simp only [srQkFG, Fin.val_cast]
        rw [srQkG_eq_of_mem_pids s0 s Logics BStartLoc.cast slh slb hmem0 hpids0])]
    rw [hseqS]; exact hM
  obtain ⟨sP, hpost, hOut⟩ := srPostLoopG_eval Logics V Out BLoc BStartLoc.cast
    BSeqLen.cast mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N hD
    s0 (srSeqLen s0 BSeqLen.cast / BLOCK_N) sL hfull (by rw [hseqS]; exact hseqpos)
    hinvLinv mr hMs0 (hOutInj s0)
  have hframeP := srIO_postLoop_frame Logics V Out BLoc BStartLoc.cast BSeqLen.cast
    mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N hD s0
    (srSeqLen s0 BSeqLen.cast / BLOCK_N) sL sP hinvLinv hpost
  obtain ⟨hpidsL, hmemL, -⟩ := hinvLinv
  rw [hpost]
  have hoeq : ∀ d : Fin BLOCK_DMODEL,
      outOffsetG s0 sob soh sod d = outOffsetG s sob soh sod d := by
    intro d; simp only [outOffsetG, hpids0]
  refine ⟨sP, rfl, ?_, ?_⟩
  · intro d
    rw [← hoeq d, hOut d]
    exact softmaxReducevWeightedSum_reindexG (by rw [hseqS])
      (srQkFG s0 Logics BStartLoc.cast BSeqLen.cast slh slb)
      (srQkFG s Logics BStartLoc.cast BSeqLen.cast slh slb) mr
      (srVFG s0 V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL)
      (srVFG s V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL) d
      (fun n => by
        simp only [srQkFG, Fin.val_cast]
        rw [srQkG_eq_of_mem_pids s0 s Logics BStartLoc.cast slh slb hmem0 hpids0])
      (fun n e => by
        simp only [srVFG, Fin.val_cast]
        rw [srVG_eq_of_mem_pids s0 s V BLoc BSeqLen.cast mil sb ss svbs svh svd
          hmem0 hpids0])
  · intro r o hcond
    have hstep1 : sP.mem r o = sL.mem r o := by
      refine hframeP r o ?_
      intro hr d hoeq'
      rcases hcond with hne | hno
      · exact hne hr
      · exact hno d (hoeq'.trans (hoeq d))
    rw [hstep1]
    have h1 : sL.mem r o = s0.mem r o := congrFun (congrFun hmemL r) o
    have h2 : s0.mem r o = s.mem r o := congrFun (congrFun hmem0 r) o
    rw [h1, h2]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-! ### ════════ ★ MAIN THEOREM (io face) ★ ════════ -/
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
          (m (⟨0, by omega⟩ : Fin 2)) hN xs ys j := by
  refine StreamMetaGatherMasked3DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact srIO_flattenOk Logics V Out BLoc BStartLoc BSeqLen mil slh slb svbs svh
      svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N other_kv_index
  · -- the safety walk
    intro bounds s m G xs ys hpre hm hg hgo hx hy hbm hbrG hbr1 hbr2 hbw
    simp only [softmaxReducevIO] at hpre hm hg hgo hbm hbrG hbr1 hbr2 hbw ⊢
    have hm0 : srSeqLen s BSeqLen.cast = m (⟨0, by omega⟩ : Fin 2) :=
      hm (⟨0, by omega⟩ : Fin 2)
    have hm1 : srStartLoc s BStartLoc.cast = m (⟨1, by omega⟩ : Fin 2) :=
      hm (⟨1, by omega⟩ : Fin 2)
    refine srIO_traceSafeR R bounds Logics V Out BLoc BStartLoc BSeqLen
      mil slh slb svbs svh svd sob soh sod sb ss BLOCK_DMODEL BLOCK_N hN
      other_kv_index s (by rw [hm0]; exact hpre.1) (by rw [hm0]; exact hpre.2.2)
      (hbm (⟨0, by omega⟩ : Fin 2)) (hbm (⟨1, by omega⟩ : Fin 2)) ?_ ?_ ?_ ?_
    · -- the gather channel's live window
      intro t jL hlive
      have h := hbrG t jL (by rw [← hm0]; exact hlive)
      simp only [srOffBLocG, hm0]
      exact h
    · -- the logits stream's live window
      intro t jL hlive
      have h := hbr1 t jL (by rw [← hm0]; exact hlive)
      rw [hm1]
      exact h
    · -- the paged-`V` stream: EVERY lane (sentinel addresses included)
      intro t jL d
      have hGeq : (if t.val * BLOCK_N + jL.val < srSeqLen s BSeqLen.cast
            then s.readMemValue .int (Region.cast BLoc)
              (srOffBLocG s BSeqLen.cast mil sb ss
                + (t.val * BLOCK_N + jL.val) * ss)
            else other_kv_index) = G t jL := by
        by_cases hlive : t.val * BLOCK_N + jL.val < srSeqLen s BSeqLen.cast
        · rw [if_pos hlive]
          have h := hg t jL (by rw [← hm0]; exact hlive)
          simp only [srOffBLocG, hm0]
          exact h
        · rw [if_neg hlive]
          exact (hgo t jL (by rw [← hm0]; exact hlive)).symm
      rw [hGeq, ← Nat.add_assoc]
      have h := hbr2 t (Lane2D.encode
        ((jL, d, PUnit.unit) : TileIndex [BLOCK_N, BLOCK_DMODEL])) trivial
      rw [Lane2D.decode_encode] at h
      exact h
    · -- the terminal store
      intro i
      exact hbw i trivial
  · -- the rounded Hoare triple: framed exact stack + cast-free collapse
    intro s₀ m G xs ys hpre hu hm hg hgo hx hy
    simp only [softmaxReducevIO] at hpre hm hg hgo hx hy ⊢
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hu]
    have hm0 : srSeqLen s₀ BSeqLen.cast = m (⟨0, by omega⟩ : Fin 2) :=
      hm (⟨0, by omega⟩ : Fin 2)
    have hm1 : srStartLoc s₀ BStartLoc.cast = m (⟨1, by omega⟩ : Fin 2) :=
      hm (⟨1, by omega⟩ : Fin 2)
    have hseqmod : srSeqLen s₀ BSeqLen.cast % BLOCK_N = 0 := by
      rw [hm0]; exact hpre.2.2
    have hseqpos : 0 < srSeqLen s₀ BSeqLen.cast := by rw [hm0]; exact hpre.2.1
    have hOutInj : ∀ st : BlockState,
        Function.Injective (fun i : Fin BLOCK_DMODEL => outOffsetG st sob soh sod i) := by
      intro st
      simpa only [outOffsetG] using hOInj (st.pids 0) (st.pids 1)
    obtain ⟨mr, hM⟩ : ∃ mr : ℝ,
        srRunningMax (srQkFG s₀ Logics BStartLoc.cast BSeqLen.cast slh slb)
          (srVFG s₀ V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL)
          (srSeqLen s₀ BSeqLen.cast) (⟨0, hD⟩ : Fin BLOCK_DMODEL)
        = (mr : WithBot ℝ) := by
      have hnb := srRunningMax_ne_botG
        (srQkFG s₀ Logics BStartLoc.cast BSeqLen.cast slh slb)
        (srVFG s₀ V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL)
        (srSeqLen s₀ BSeqLen.cast) (⟨0, hD⟩ : Fin BLOCK_DMODEL) hseqpos
        (le_refl _)
      cases hh : srRunningMax (srQkFG s₀ Logics BStartLoc.cast BSeqLen.cast slh slb)
          (srVFG s₀ V BLoc BSeqLen.cast mil sb ss svbs svh svd BLOCK_DMODEL)
          (srSeqLen s₀ BSeqLen.cast) (⟨0, hD⟩ : Fin BLOCK_DMODEL) with
      | bot => exact absurd hh hnb
      | coe x => exact ⟨x, rfl⟩
    obtain ⟨sF, hstep, hOut, hframe⟩ :=
      srIO_exec_framed Logics V Out BLoc BStartLoc BSeqLen mil slh slb svbs svh svd
        sob soh sod sb ss BLOCK_DMODEL BLOCK_N other_kv_index hD hN s₀ hundef'
        hseqmod hseqpos hOutInj mr hM
    refine ⟨sF, ?_, ?_, ?_⟩
    · rw [srIO_execR_collapse]
      exact hstep
    · -- readback: the genuine closed form = the streamed closed form
      intro j _
      simp only [BlockState.readMemAs_real, R.round_real_apply]
      refine congrArg some ((hOut j).trans ?_)
      exact srIOSpec_eq_genuineG Logics V BLoc BStartLoc.cast BSeqLen.cast
        mil slh slb svbs svh svd sb ss BLOCK_DMODEL BLOCK_N hN s₀
        (m (⟨0, by omega⟩ : Fin 2)) (m (⟨1, by omega⟩ : Fin 2))
        (srIOT mil BLOCK_N) rfl hm0 hm1 hpre.1 hpre.2.2 G xs ys
        (fun t jL h => hg t jL h) (fun t jL h => hx t jL h)
        (fun t j' => hy t j' trivial) mr j
    · -- the frame
      intro r o hcond
      refine hframe r o ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · exact Or.inr (fun d => hno d trivial)

end IOFace

end VeriTile.Bench.TritonBenchG.SoftmaxReducev
