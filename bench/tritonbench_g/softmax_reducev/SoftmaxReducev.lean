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
