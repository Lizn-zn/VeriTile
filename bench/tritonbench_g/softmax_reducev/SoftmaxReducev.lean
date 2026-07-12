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


end VeriTile.Bench.TritonBenchG.SoftmaxReducev
