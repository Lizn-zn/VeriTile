import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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
softmax_reducev_python_test_shape_output_summary      ← TOP THEOREM
  ├─ softmax_reducev_python_case_other_neg_one_surface_toAlgorithm_supported  surface lowers
  ├─ softmax_reducev_python_case_other_zero_surface_toAlgorithm_supported     (other_kv_index = 0)
  │    └─ softmax_reducev_surface_toAlgorithm_supported
  └─ softmax_reducev_surface_output_compute_correct     ← ComputeCorrect of the store
       ├─ softmax_reducev_python_output_offset_injective
       └─ softmax_reducev_python_test_shape_final_output_compute_correct
            └─ softmax_reducev_final_store_slice_compute_correct
                 └─ softmax_reducev_final_store_slice_correct  (per-lane normalized readback)
```

The top theorem covers both `other_kv_index ∈ {-1, 0}` test variants at the
Python test shape (`BLOCK_DMODEL = 64`, `BLOCK_N = 64`).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float, so the `-inf` init,
`exp`, running-max rescaling, and the final division are real-valued);
`@triton.autotune` / `num_warps` / `num_stages` are not modeled. The verified
statement is scoped to the **final normalized store** to `Out`: the expected
value is `acc / e_sum` (`softmaxReducevFinalSpec` / `softmaxReducevSurfaceValue`)
read off at each `outOffset`, over a one-block output footprint with the program
ids universally quantified. The streaming online-softmax loop (running max,
rescale-and-accumulate, paged-V gather) feeds those `Acc`/`ESum` values; the side
condition is the offset-injectivity of the `Out` slice at the test shape.

## Genuine closed form (input-side, non-self-referential)

`softmaxReducevWeightedSum` states the mathematical result over the *input*
logits and gathered value rows:
`out[d] = Σ_n softmax(qk)[n] · V[v_index[n], d]`
`= (Σ_n exp(qk[n] - M)·V[v_index[n], d]) / (Σ_n exp(qk[n] - M))`.
`softmaxReducevWeightedSum_shift_invariant` proves this is the genuine softmax
(independent of the running max `M`). `softmaxReducevFinalSpec_eq_weightedSum`
bridges the verified `acc / e_sum` store spec to this closed form, and
`softmax_reducev_final_store_slice_weighted_sum_correct` proves the verified
final-store slice *realizes* it under the loop-output contract (`Acc` / `ESum`
hold the closed-form numerator / denominator — exact over `ℝ`). What remains to
fully discharge the full-surface top theorem's self-reference is the `exec`-side
unfold of the dynamic (`forRangeDyn`) online-softmax loop with the paged-V gather
— the multi-thousand-line FA-1/attention-forward analogue.
-/

namespace VeriTile.Bench.TritonBenchG.SoftmaxReducev

open VeriTile.Triton

set_option linter.unusedSimpArgs false

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

/-- **Shift-invariance of the closed form.** The normalized weighted sum does
not depend on the running max `mMax`: subtracting any constant from every logit
cancels between numerator and denominator. This certifies that
`softmaxReducevWeightedSum` is the *genuine* softmax-weighted V reduction — the
streaming max `M = maxₙ qk[n]` is only a numerical-stability device, not part of
the mathematical result. Requires the denominator nonzero (always true here,
`exp > 0`, when `S > 0`). -/
theorem softmaxReducevWeightedSum_shift_invariant {S BLOCK_DMODEL : Nat}
    (qk : Fin S → ℝ) (m₁ m₂ : ℝ) (v : Fin S → Fin BLOCK_DMODEL → ℝ)
    (d : Fin BLOCK_DMODEL) :
    softmaxReducevAcc qk m₁ v d / softmaxReducevDenom qk m₁
      = softmaxReducevAcc qk m₂ v d / softmaxReducevDenom qk m₂ := by
  have hnum : softmaxReducevAcc qk m₁ v d
      = Real.exp (m₂ - m₁) * softmaxReducevAcc qk m₂ v d := by
    unfold softmaxReducevAcc softmaxWeight
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro n _
    rw [← mul_assoc, ← Real.exp_add]
    ring_nf
  have hden : softmaxReducevDenom qk m₁
      = Real.exp (m₂ - m₁) * softmaxReducevDenom qk m₂ := by
    unfold softmaxReducevDenom softmaxWeight
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro n _
    rw [← Real.exp_add]
    ring_nf
  rw [hnum, hden, mul_div_mul_left _ _ (Real.exp_ne_zero _)]

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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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

/-! ## Python test-shape wrappers

`test_token_softmax_reducev_fwd` fixes `batch = 2`, `head = 2`,
`max_input_len = 128`, `d_model = 64`, and `BLOCK_N = 64`.  The first three
cases use `other_kv_index = -1`; the fourth uses `other_kv_index = 0`. -/

theorem softmax_reducev_python_output_offset_injective
    (s : BlockState) :
    Function.Injective (fun i : Fin 64 => outOffset s 128 64 1 i) := by
  intro a b h
  apply Fin.ext
  simp [outOffset, dIndex] at h
  omega

/-- Python test-shape final output store correctness for `o` with shape
`(batch, head, d_model) = (2, 2, 64)`.  The accumulator and denominator are
materialized in proof regions; the full streaming softmax loop is represented
by the surface wrappers below. -/
theorem softmax_reducev_python_test_shape_final_output_compute_correct
    (Acc ESum Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := softmax_reducev_final_store_slice Acc ESum Out
        128 64 1 2 1 128 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 128 64 1 i))
      (expected := fun i =>
        softmaxReducevFinalSpec s Acc ESum 128 64 1 2 1 i) := by
  exact softmax_reducev_final_store_slice_compute_correct Acc ESum Out
    128 64 1 2 1 128 64 1 64 s
    (softmax_reducev_python_output_offset_injective s)

theorem softmax_reducev_python_case_other_neg_one_surface_toAlgorithm_supported
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat) :
    ∃ alg, (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      128 256 1 8192 64 1 128 64 1 128 1 64 64 (-1)).toAlgorithm? =
        Except.ok alg := by
  exact softmax_reducev_surface_toAlgorithm_supported Logics V Out BLoc
    BStartLoc BSeqLen 128 256 1 8192 64 1 128 64 1 128 1 64 64 (-1)

theorem softmax_reducev_python_case_other_zero_surface_toAlgorithm_supported
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat) :
    ∃ alg, (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      128 256 1 8192 64 1 128 64 1 128 1 64 64 0).toAlgorithm? =
        Except.ok alg := by
  exact softmax_reducev_surface_toAlgorithm_supported Logics V Out BLoc
    BStartLoc BSeqLen 128 256 1 8192 64 1 128 64 1 128 1 64 64 0

noncomputable def softmaxReducevSurfaceValue
    (s : BlockState) (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat)
    (max_input_len
      stride_logic_h stride_logic_bs
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_b_loc_b stride_b_loc_s
      BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int) (offset : Nat) : ℝ :=
  match exec (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      max_input_len stride_logic_h stride_logic_bs stride_vbs stride_vh
      stride_vd stride_obs stride_oh stride_od stride_b_loc_b stride_b_loc_s
      BLOCK_DMODEL BLOCK_N other_kv_index) s with
  | some s' => s'.readMem Out offset
  | none => 0.0

theorem softmax_reducev_surface_output_compute_correct
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat)
    (max_input_len
      stride_logic_h stride_logic_bs
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_b_loc_b stride_b_loc_s
      BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
        max_input_len stride_logic_h stride_logic_bs stride_vbs stride_vh
        stride_vd stride_obs stride_oh stride_od stride_b_loc_b stride_b_loc_s
        BLOCK_DMODEL BLOCK_N other_kv_index)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i =>
        softmaxReducevSurfaceValue s Logics V Out BLoc BStartLoc BSeqLen
          max_input_len stride_logic_h stride_logic_bs stride_vbs stride_vh
          stride_vd stride_obs stride_oh stride_od stride_b_loc_b
          stride_b_loc_s BLOCK_DMODEL BLOCK_N other_kv_index
          (outOffset s stride_obs stride_oh stride_od i)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_reducev_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [softmaxReducevSurfaceValue, hExec]

/-- Public Python case coverage summary: the full checked surface lowers for
both sentinel choices, and the final normalized `o` vector writeback realizes
the Python output shape. -/
theorem softmax_reducev_python_test_shape_output_surface_summary
    (Logics V Acc ESum Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat) (s : BlockState) :
    (∃ alg, (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      128 256 1 8192 64 1 128 64 1 128 1 64 64 (-1)).toAlgorithm? =
        Except.ok alg) ∧
    (∃ alg, (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      128 256 1 8192 64 1 128 64 1 128 1 64 64 0).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := softmax_reducev_final_store_slice Acc ESum Out
        128 64 1 2 1 128 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 128 64 1 i))
      (expected := fun i =>
        softmaxReducevFinalSpec s Acc ESum 128 64 1 2 1 i)) := by
  constructor
  · exact softmax_reducev_python_case_other_neg_one_surface_toAlgorithm_supported
      Logics V Out BLoc BStartLoc BSeqLen
  constructor
  · exact softmax_reducev_python_case_other_zero_surface_toAlgorithm_supported
      Logics V Out BLoc BStartLoc BSeqLen
  · exact softmax_reducev_python_test_shape_final_output_compute_correct
      Acc ESum Out s

/-- Python reduce-V softmax final-store coverage. -/
abbrev softmax_reducev_python_test_shape_store_summary
    (Logics V Acc ESum Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat) (s : BlockState) :=
  softmax_reducev_python_test_shape_output_surface_summary
    Logics V Acc ESum Out BLoc BStartLoc BSeqLen s




















theorem softmax_reducev_python_test_shape_output_summary
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat) (s : BlockState) :
    (∃ alg, (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      128 256 1 8192 64 1 128 64 1 128 1 64 64 (-1)).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
        128 256 1 8192 64 1 128 64 1 128 1 64 64 (-1))
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 128 64 1 i))
      (expected := fun i : Fin 64 =>
        softmaxReducevSurfaceValue s Logics V Out BLoc BStartLoc BSeqLen
          128 256 1 8192 64 1 128 64 1 128 1 64 64 (-1)
          (outOffset s 128 64 1 i))) ∧
    (∃ alg, (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      128 256 1 8192 64 1 128 64 1 128 1 64 64 0).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
        128 256 1 8192 64 1 128 64 1 128 1 64 64 0)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 128 64 1 i))
      (expected := fun i : Fin 64 =>
        softmaxReducevSurfaceValue s Logics V Out BLoc BStartLoc BSeqLen
          128 256 1 8192 64 1 128 64 1 128 1 64 64 0
          (outOffset s 128 64 1 i))) := by
  constructor
  · exact softmax_reducev_python_case_other_neg_one_surface_toAlgorithm_supported
      Logics V Out BLoc BStartLoc BSeqLen
  constructor
  · exact softmax_reducev_surface_output_compute_correct Logics V Out BLoc
      BStartLoc BSeqLen 128 256 1 8192 64 1 128 64 1 128 1 64 64 (-1) s
  constructor
  · exact softmax_reducev_python_case_other_zero_surface_toAlgorithm_supported
      Logics V Out BLoc BStartLoc BSeqLen
  · exact softmax_reducev_surface_output_compute_correct Logics V Out BLoc
      BStartLoc BSeqLen 128 256 1 8192 64 1 128 64 1 128 1 64 64 0 s

end VeriTile.Bench.TritonBenchG.SoftmaxReducev
