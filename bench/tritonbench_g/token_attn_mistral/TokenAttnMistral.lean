import VeriTile.Triton

/-!
# `token_attn_mistral` — strict per-kernel correctness

`_fwd_kernel_token_att2` is the Mistral token-decode PV-accumulation stage. Each
program `(cur_batch, cur_head)` streams over the attention-window tokens
(`cur_batch_start_index = max(cur_batch_seq_len - sliding_window, 0)`), loads the
per-token probabilities `Prob` and the value rows `V` gathered through
`Req_to_tokens`, accumulates `acc = Σ p_value · v` into a `[BLOCK_DMODEL]`
vector, and stores it to `Out`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_token_att2[grid](...)` over `(batch, head)`,
the scheduling, and how the runtime composes per-program writes into one buffer)
is the *trusted boundary*, not a proof obligation here. Because the program ids
`(cur_batch, cur_head)` are universally quantified, the per-program statement
covers every program of the grid.

## Proof architecture

```
token_attn_mistral_output_summary_general                     ← HEADLINE (symbolic dims / sliding_window / strides)
  ├─ token_attn_mistral_surface_toAlgorithm_supported                  surface lowers to the algorithm layer
  └─ token_attn_mistral_closed_form_compute_correct                    full surface, final store = closed form
       └─ token_attn_mistral_closed_form_correct                       exec readback = tokenAttnMistralClosedForm
            ├─ mistral_preLoop          14 prelude assigns → entry invariant (acc = partialAcc 0)
            ├─ mistral_loop_step        5-stmt body advances partialAcc by one BLOCK_N block (forRangeDyn_inv)
            └─ mistral_postLoop         cast + off_o + out_ptrs + unmasked store readback = closed form
(algebra: partialAcc_block_succ / partialAcc_eq_PVValue; recipes: mistral_*_eval)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. Each per-case `output_summary` shows the surface kernel lowers to the
algorithm layer AND the store to `Out` is compute-correct against a **genuine,
self-reference-free closed form** `tokenAttnMistralClosedForm` (this is stated once
in the dimension-general headline `token_attn_mistral_output_summary_general` over
symbolic `BLOCK_DMODEL`/`BLOCK_N`/`sliding_window`/strides): every lane `d` of
the `[BLOCK_DMODEL]` output holds the sliding-window probability-weighted value
sum `Σ_{n < cur_att_seq_len} p[n]·v[v_loc[n], d]` (with the `start_index` offset
and `Req_to_tokens` gather folded in; out-of-window tokens masked to `0`). This is
proven by fully executing the surface kernel — the `mistral_preLoop` prelude
decode, the `mistral_loop_step` one-block accumulator advance driven through
`forRangeDyn_inv`, and the `mistral_postLoop` unmasked-store readback — never by
re-asserting the kernel's own executed value. The final store is **unmasked** (the
whole `[BLOCK_DMODEL]` vector is written) and includes a `.to(Out.dtype)` cast
that reduces to the identity at the algorithm layer. The headline theorem
`token_attn_mistral_output_summary_general` is dimension-general and symbolic in
`sliding_window`/strides, so it covers arbitrary shapes (under the honest
contiguous-layout side-conditions `stride_pbs = 1` and `stride_req_to_tokens_s = 1`).
-/

namespace VeriTile.Bench.TritonBenchG.TokenAttnMistral

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `token_attn_mistral_output_summary_general` (dimension-general headline). -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-- Faithful transcription of `token_attn_mistral.py`'s
`_fwd_kernel_token_att2`.

Typed-region note: metadata/gather buffers are `Region .nat`, matching their
index role without adding source-level `dtype=` kwargs. -/
def token_attn_mistral_surface
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (_B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  cur_kv_head = cur_head // $(kv_group_num)
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_start_index = tl.maximum(cur_batch_seq_len - $(sliding_window), $(0))
  cur_batch_in_all_start_index = tl.load(B_Att_Start_Loc + cur_batch)
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
  cur_att_seq_len = tl.load(B_Att_Seqlen + cur_batch)
  v_loc_off = cur_batch_req_idx * $(stride_req_to_tokens_b) +
    (cur_batch_start_index + offs_n) * $(stride_req_to_tokens_s)
  p_offs = cur_head * $(stride_ph) +
    (cur_batch_in_all_start_index + offs_n) * $(stride_pbs)
  v_offs = cur_kv_head * $(stride_vh) + offs_d[None, :] * $(stride_vd)
  acc = tl.zeros([$(BLOCK_DMODEL)], dtype=tl.float32)
  for start_n in range($(0), cur_att_seq_len, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    p_value = tl.load(Prob + p_offs + start_n,
      mask=(start_n + offs_n) < cur_att_seq_len, other=0.0)
    v_loc = tl.load(Req_to_tokens + v_loc_off +
      start_n * $(stride_req_to_tokens_s),
      mask=(start_n + offs_n + cur_batch_start_index) < cur_batch_seq_len,
      other=0.0)
    v_value = tl.load(V + v_offs + v_loc[:, None] * $(stride_vbs),
      mask=(start_n + offs_n[:, None] + cur_batch_start_index) < cur_batch_seq_len,
      other=0.0)
    acc += tl.sum(p_value[:, None] * v_value, 0)
  }
  acc = (acc).to(Out.dtype.element_ty)
  off_o = cur_batch * $(stride_obs) + cur_head * $(stride_oh) + offs_d * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc)
}

/-- The full token-attention Mistral reduce-V surface lowers to the algorithm
layer. -/
theorem token_attn_mistral_surface_toAlgorithm_supported
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N : Nat) :
    ∃ alg, (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen stride_req_to_tokens_b
      stride_req_to_tokens_s stride_ph stride_pbs stride_vbs stride_vh
      stride_vd stride_obs stride_oh stride_od kv_group_num sliding_window
      BLOCK_DMODEL BLOCK_N).toAlgorithm? = Except.ok alg := by
  simp [token_attn_mistral_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented final output-store slice of `token_attn_mistral.py`'s
`_fwd_kernel_token_att2`.

The full kernel applies a sliding-window token gather and accumulates
`sum(prob * v)`. This slice starts from a precomputed `Acc` vector and proves the
final `BLOCK_DMODEL` writeback into `Out`. -/
def token_attn_mistral_final_store_slice
    (Acc Out : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  acc = tl.load(Acc + cur_batch * $(stride_acc_bs) + cur_head * $(stride_acc_h) +
      offs_d * $(stride_acc_d))
  tl.store(Out + cur_batch * $(stride_obs) + cur_head * $(stride_oh) +
      offs_d * $(stride_od), (acc).to(Out.dtype.element_ty))
}

def dIndex (_s : BlockState) (i : Fin BLOCK_DMODEL) : Nat :=
  i.val

def accOffset
    (s : BlockState) (stride_acc_bs stride_acc_h stride_acc_d : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_acc_bs + s.pids 1 * stride_acc_h + dIndex s i * stride_acc_d

def outOffset
    (s : BlockState) (stride_obs stride_oh stride_od : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_obs + s.pids 1 * stride_oh + dIndex s i * stride_od

/-- Algorithm-layer correctness for the Mistral token-attention final store. -/
theorem token_attn_mistral_final_store_slice_correct
    (Acc Out : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ∀ i : Fin BLOCK_DMODEL,
      let outAddr := outOffset s stride_obs stride_oh stride_od i
      (exec (token_attn_mistral_final_store_slice Acc Out stride_acc_bs
            stride_acc_h stride_acc_d stride_obs stride_oh stride_od BLOCK_DMODEL)
          s).map (·.readMem Out outAddr)
        = some (s.readMem Acc (accOffset s stride_acc_bs stride_acc_h stride_acc_d i)) := by
  intro i
  simp [exec, token_attn_mistral_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, dIndex, accOffset, outOffset]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_DMODEL] =>
        s.pids 0 * stride_obs + s.pids 1 * stride_oh + idx.1.val * stride_od) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s stride_obs stride_oh stride_od a =
        outOffset s stride_obs stride_oh stride_od b := by
      simpa [outOffset, dIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]

/-- Compute-facing correctness for the Mistral token-attention final store. -/
theorem token_attn_mistral_final_store_slice_compute_correct
    (Acc Out : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := token_attn_mistral_final_store_slice Acc Out stride_acc_bs
        stride_acc_h stride_acc_d stride_obs stride_oh stride_od BLOCK_DMODEL)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i =>
        s.readMem Acc (accOffset s stride_acc_bs stride_acc_h stride_acc_d i)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [token_attn_mistral_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := token_attn_mistral_final_store_slice_correct Acc Out stride_acc_bs
    stride_acc_h stride_acc_d stride_obs stride_oh stride_od BLOCK_DMODEL
    s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-! ## Genuine closed-form PV-reduction spec

The Triton `_fwd_kernel_token_att2` accumulates, per output head-dim `d`, the
sliding-window probability-weighted value sum

```
acc[d] = Σ_{n < cur_att_seq_len}  p[n] · v[v_loc[n], d]
```

where, with `start_index = max(cur_batch_seq_len - sliding_window, 0)` and
`cur_kv_head = cur_head / kv_group_num`:

* the per-token probability
  `p[n] = Prob[cur_head·stride_ph + (att_start_loc + n)·stride_pbs]`, masked by
  `n < cur_att_seq_len` (`other = 0`);
* the gathered KV page index
  `v_loc[n] = Req_to_tokens[req_idx·stride_req_to_tokens_b +
    (start_index + n)·stride_req_to_tokens_s]`, masked by
  `start_index + n < cur_batch_seq_len`;
* the value row
  `v[v_loc[n], d] = V[cur_kv_head·stride_vh + d·stride_vd +
    v_loc[n]·stride_vbs]`, masked by `start_index + n < cur_batch_seq_len`
  (`other = 0`), so an out-of-window token contributes `0` to every `d`.

The final store of the whole `[BLOCK_DMODEL]` accumulator is **unmasked**, with a
`.to(Out.dtype)` cast that is the identity at the algorithm (ℝ) layer.

These definitions are a *genuine closed form* — they never execute the kernel —
and (via `token_attn_mistral_closed_form_correct`) fully replace the former
self-referential surface-value spec.

`attSeqLen`/`batchSeqLen`/`reqIdx`/`attStartLoc` are the metadata loads of the
prelude; `startIndex` is the sliding-window left edge. -/

def attSeqLen (s : BlockState) (B_Att_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Att_Seqlen (s.pids 0)

def batchSeqLen (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0)

def reqIdx (s : BlockState) (B_req_idx : RegionName) : Nat :=
  s.readMemValue .nat B_req_idx (s.pids 0)

def attStartLoc (s : BlockState) (B_Att_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Att_Start_Loc (s.pids 0)

/-- Sliding-window left edge `cur_batch_start_index = max(cur_batch_seq_len -
sliding_window, 0)`. The Nat subtraction already truncates at `0`, so the
`tl.maximum(·, 0)` is the identity. -/
def startIndex (s : BlockState) (B_Seqlen : RegionName) (sliding_window : Nat) : Nat :=
  batchSeqLen s B_Seqlen - sliding_window

/-- A window token `n` whose V-gather is in range: `start_index + n <
cur_batch_seq_len`. Out-of-range tokens read `0` from both the `Req_to_tokens`
gather and the masked `V` load, hence contribute `0`. -/
def vActive
    (s : BlockState) (B_Seqlen : RegionName) (sliding_window : Nat) (n : Nat) : Prop :=
  startIndex s B_Seqlen sliding_window + n < batchSeqLen s B_Seqlen

instance vActiveDecidable
    (s : BlockState) (B_Seqlen : RegionName) (sliding_window n : Nat) :
    Decidable (vActive s B_Seqlen sliding_window n) := by
  unfold vActive; infer_instance

/-- Per-token probability load offset:
`cur_head·stride_ph + (att_start_loc + n)·stride_pbs`. -/
def pOffset
    (s : BlockState) (B_Att_Start_Loc : RegionName)
    (stride_ph stride_pbs : Nat) (n : Nat) : Nat :=
  s.pids 1 * stride_ph + (attStartLoc s B_Att_Start_Loc + n) * stride_pbs

/-- Gathered KV page index for window token `n`:
`Req_to_tokens[req_idx·stride_req_to_tokens_b +
  (start_index + n)·stride_req_to_tokens_s]`. -/
def vLoc
    (s : BlockState) (Req_to_tokens B_req_idx B_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s sliding_window : Nat)
    (n : Nat) : Nat :=
  s.readMemValue .nat Req_to_tokens
    (reqIdx s B_req_idx * stride_req_to_tokens_b +
      (startIndex s B_Seqlen sliding_window + n) * stride_req_to_tokens_s)

/-- Value-row load offset for window token `n`, head-dim `d`:
`v_loc[n]·stride_vbs + cur_kv_head·stride_vh + d·stride_vd`, with
`cur_kv_head = cur_head / kv_group_num`. -/
def vOffset
    (s : BlockState) (Req_to_tokens B_req_idx B_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_vbs stride_vh stride_vd
      kv_group_num sliding_window : Nat) (n d : Nat) : Nat :=
  vLoc s Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
      stride_req_to_tokens_s sliding_window n * stride_vbs +
    (s.pids 1 / kv_group_num) * stride_vh + d * stride_vd

/-- The genuine closed-form accumulator for output head-dim `d`:
`Σ_{n < cur_att_seq_len} p[n] · v[v_loc[n], d]`, where an out-of-window token
(`¬ vActive`) contributes `0` because its masked `V` load reads `0`. -/
noncomputable def tokenAttnMistralPVValue
    (s : BlockState) (Prob V : RegionName)
    (Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd kv_group_num sliding_window : Nat)
    (d : Nat) : ℝ :=
  ∑ n ∈ Finset.range (attSeqLen s B_Att_Seqlen),
    s.readMem Prob (pOffset s B_Att_Start_Loc stride_ph stride_pbs n) *
      (if vActive s B_Seqlen sliding_window n then
        s.readMem V (vOffset s Req_to_tokens B_req_idx B_Seqlen
          stride_req_to_tokens_b stride_req_to_tokens_s stride_vbs stride_vh
          stride_vd kv_group_num sliding_window n d)
      else 0.0)

/-- Genuine closed-form value written to `Out[outOffset d]`. The store is
unmasked over the full `[BLOCK_DMODEL]` vector, so every lane holds the
PV-accumulator `tokenAttnMistralPVValue` for its head-dim `d = dIndex s i`. -/
noncomputable def tokenAttnMistralClosedForm
    (s : BlockState) (Prob V : RegionName)
    (Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd kv_group_num sliding_window BLOCK_DMODEL : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  tokenAttnMistralPVValue s Prob V Req_to_tokens B_req_idx B_Att_Start_Loc
    B_Seqlen B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
    stride_ph stride_pbs stride_vbs stride_vh stride_vd kv_group_num
    sliding_window (dIndex s i)

/-- Masked per-token probability: `p[n]` if the token is in the
`cur_att_seq_len` window, else `0` (the `tl.load(Prob …, mask=(start_n+offs_n) <
cur_att_seq_len, other=0)` masked load). -/
noncomputable def pMasked
    (s : BlockState) (Prob B_Att_Start_Loc B_Att_Seqlen : RegionName)
    (stride_ph stride_pbs : Nat) (n : Nat) : ℝ :=
  if n < attSeqLen s B_Att_Seqlen then
    s.readMem Prob (pOffset s B_Att_Start_Loc stride_ph stride_pbs n)
  else 0.0

/-- Masked value row: `v[v_loc[n], d]` if the gather is in range (`vActive`),
else `0` (the `tl.load(V …, mask=start_index+n < cur_batch_seq_len, other=0)`
masked load). -/
noncomputable def vMasked
    (s : BlockState) (V Req_to_tokens B_req_idx B_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_vbs stride_vh stride_vd
      kv_group_num sliding_window : Nat) (n d : Nat) : ℝ :=
  if vActive s B_Seqlen sliding_window n then
    s.readMem V (vOffset s Req_to_tokens B_req_idx B_Seqlen
      stride_req_to_tokens_b stride_req_to_tokens_s stride_vbs stride_vh
      stride_vd kv_group_num sliding_window n d)
  else 0.0

/-- Partial PV accumulator after the window tokens `n < k` have been folded in,
for output head-dim `d`. The loop carries `partialAcc (c·BLOCK_N)`. -/
noncomputable def partialAcc
    (s : BlockState) (Prob V : RegionName)
    (Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd kv_group_num sliding_window : Nat)
    (k d : Nat) : ℝ :=
  ∑ n ∈ Finset.range k,
    pMasked s Prob B_Att_Start_Loc B_Att_Seqlen stride_ph stride_pbs n *
      vMasked s V Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
        stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num
        sliding_window n d

/-- One block of `BLOCK_N` window tokens advances the partial accumulator.
`partialAcc (c·BLOCK_N + BLOCK_N) = partialAcc (c·BLOCK_N) + Σ_{j<BLOCK_N} …`.
This is the algebraic content of the loop body's `acc += tl.sum(p·v, 0)`. -/
theorem partialAcc_block_succ
    (s : BlockState) (Prob V : RegionName)
    (Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd kv_group_num sliding_window : Nat)
    (start_n BLOCK_N d : Nat) :
    partialAcc s Prob V Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen
        B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
        stride_pbs stride_vbs stride_vh stride_vd kv_group_num sliding_window
        (start_n + BLOCK_N) d
      = partialAcc s Prob V Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen
          B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
          stride_pbs stride_vbs stride_vh stride_vd kv_group_num sliding_window
          start_n d
        + ∑ j ∈ Finset.range BLOCK_N,
            pMasked s Prob B_Att_Start_Loc B_Att_Seqlen stride_ph stride_pbs
                (start_n + j) *
              vMasked s V Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
                stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num
                sliding_window (start_n + j) d := by
  unfold partialAcc
  rw [Finset.sum_range_add]

/-- Once the window loop has run past `cur_att_seq_len` (the loop's existential
final counter `final ≥ attSeqLen`, here `K ≥ attSeqLen`), the partial PV
accumulator over `range K` coincides with the genuine closed form
`tokenAttnMistralPVValue`: tokens `n ≥ attSeqLen` are masked to `p = 0`, so they
contribute nothing, and the range can be cut back to `attSeqLen`. -/
theorem partialAcc_eq_PVValue
    (s : BlockState) (Prob V : RegionName)
    (Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd kv_group_num sliding_window : Nat)
    (K d : Nat) (hK : attSeqLen s B_Att_Seqlen ≤ K) :
    partialAcc s Prob V Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen
        B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
        stride_pbs stride_vbs stride_vh stride_vd kv_group_num sliding_window K d
      = tokenAttnMistralPVValue s Prob V Req_to_tokens B_req_idx B_Att_Start_Loc
          B_Seqlen B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
          stride_ph stride_pbs stride_vbs stride_vh stride_vd kv_group_num
          sliding_window d := by
  unfold partialAcc tokenAttnMistralPVValue
  -- Split range K = range attSeqLen ∪ [attSeqLen, K); the tail is all zero.
  rw [← Finset.sum_range_add_sum_Ico _ hK]
  have htail : ∑ n ∈ Finset.Ico (attSeqLen s B_Att_Seqlen) K,
      pMasked s Prob B_Att_Start_Loc B_Att_Seqlen stride_ph stride_pbs n *
        vMasked s V Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
          stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num
          sliding_window n d = 0 := by
    apply Finset.sum_eq_zero
    intro n hn
    rw [Finset.mem_Ico] at hn
    have : ¬ n < attSeqLen s B_Att_Seqlen := by omega
    simp only [pMasked, this, if_false]
    norm_num
  rw [htail, add_zero]
  apply Finset.sum_congr rfl
  intro n hn
  rw [Finset.mem_range] at hn
  simp only [pMasked, vMasked, if_pos hn]

/-! ### Surface readback bridge (proven sorry-free)

`tokenAttnMistralClosedForm` is the genuine, self-reference-free PV spec.  The
surface readback bridge

```
exec (token_attn_mistral_surface …) s = some s' →
  s'.readMem Out (outOffset … i) =
    tokenAttnMistralClosedForm … i
```

is discharged by `token_attn_mistral_closed_form_correct`, assembled from
`mistral_preLoop` (prelude decode), `mistral_loop_step` (one-block accumulator
advance through `forRangeDyn_inv`), and `mistral_postLoop` (unmasked store
readback).  Routing/decode notes for that bridge:

* The kernel is the **PV reduce / accumulate** route (no online softmax): the
  genuine spec is the direct probability-weighted value sum
  `Σ_n p[n]·v[v_loc[n], d]`, reusing the `reduceSum_some` (`Tile.reduceSum`)
  machinery as in `batched_vecmat_one_row_block_correct` for the per-block
  `tl.sum(p_value[:,None]·v_value, 0)`, with the cross-block accumulator carried
  by the outer `range(0, cur_att_seq_len, BLOCK_N)` loop, followed by the
  unmasked `[BLOCK_DMODEL]` store readback (`scatter_readback_nd`, as in
  `token_attn_mistral_final_store_slice_correct`).
* Exec assembly: `exec` reduces to `stepStmts (…).toAlgKernel.body s` by `rfl`
  (the `ComputeStmt → Stmt` lowering of every `ComputeExpr.alg` body statement
  is definitional).  Decode the prelude assigns and the
  `forRangeDyn "start_n" 0 cur_att_seq_len BLOCK_N …` loop via
  `stepStmts.cons_some (stepStmt_assign_eq_some (…_op_eval …))` and
  `stepForRangeAux.forRangeDyn_unfold`.  Unlike the llama2 sibling's single
  `block_mask`-guarded iteration, this loop runs `⌈cur_att_seq_len / BLOCK_N⌉`
  times, so the bridge needs a loop-invariant induction (`LoopInvariant`) on the
  partial accumulator `acc_k = Σ_{n < k·BLOCK_N} p[n]·v[…]` rather than a single
  `step_one_iter`/`step_ge` split.
* Each loop body needs per-statement `*_op_eval` recipes for the `Prob` masked
  load (`pOffset`, mask `n < cur_att_seq_len`), the `Req_to_tokens` gather
  (`vLoc`), the 2D `v_offs`/`v_value` masked gather (`vOffset`, mask
  `start_index + n < cur_batch_seq_len` ⇒ `vActive`), the
  `tl.sum(p_value[:,None]·v_value, 0)` block reduction (`reduceSum_some`), and the
  `acc += …` accumulation; then the final unmasked store readback.

### Algebraic ingredients of the bridge

The two *algebraic* halves of the loop-invariant induction are proven
sorry-free, together with the generic dynamic-loop principle:

* `VeriTile.Triton.forRangeDyn_inv` — master invariant principle for
  `forRangeDyn` (mirror of `forRange_inv`), the induction engine for the
  `range(0, cur_att_seq_len, BLOCK_N)` window loop.
* `pMasked` / `vMasked` — the algorithm-layer values produced by the masked
  `Prob` / `V` loads (`other = 0` for the out-of-window lanes).
* `partialAcc s … k d = Σ_{n < k} pMasked n · vMasked n d` — the accumulator the
  loop carries (`acc_k = partialAcc (counter)`).
* `partialAcc_block_succ` — one `BLOCK_N` block advances the carry:
  `partialAcc (start_n + BLOCK_N) = partialAcc start_n + Σ_{j<BLOCK_N} …`.
  This is exactly the algebraic content of `acc += tl.sum(p·v, 0)`.
* `partialAcc_eq_PVValue` — at the loop's existential final counter
  `K ≥ attSeqLen`, the carry equals the genuine closed form
  `tokenAttnMistralPVValue` (tail tokens `n ≥ attSeqLen` are `p`-masked to `0`).

### The operational `exec` decode (discharged)

The purely mechanical state-stepping that connects the kernel's `stepStmts` to
these algebraic facts (no further mathematical content) is carried out by the
three step lemmas above:

1. Prelude decode (14 assigns) — `mistral_preLoop`: thread
   `cur_batch`/`cur_head`/`cur_kv_head`,
   `offs_n = arange 128`, `offs_d = arange 64`, the four metadata loads, the
   `tl.maximum(... , 0)` `startIndex`, the `v_loc_off`/`p_offs`/`v_offs` vectors,
   and `acc = full [BLOCK_DMODEL] 0` into a symbolic post-prelude state, mirroring
   `chunk_cumsum_kernel`'s prelude reduction.
2. Loop step lemma — `mistral_loop_step`: the 5-statement body
   (`start_n = multiple_of …` no-op,
   `p_value`/`v_loc`/`v_value` masked region loads, `acc += reduceSum (p[:,None]·v)`)
   advances `acc = partialAcc i` to `acc = partialAcc (i + BLOCK_N)` via
   `partialAcc_block_succ` + the per-lane load/`reduceSum` recipes, fed to
   `forRangeDyn_inv`.
3. Final unmasked `[BLOCK_DMODEL]` store readback — `mistral_postLoop`
   (`scatter_readback_nd`), then
   `partialAcc_eq_PVValue` to land on `tokenAttnMistralClosedForm`.

NOTE: the closed form folds the per-lane offset arithmetic
`pOffset n = … + (att_start_loc + n)·stride_pbs` and the `Req_to_tokens` gather
`(start_index + n)·stride_req_to_tokens_s`, whereas the kernel computes
`p_offs[j] + start_n` (resp. `v_loc_off[j] + start_n·stride_req_to_tokens_s`).
These coincide for the lane index `n = start_n + j` exactly when
`stride_pbs = 1` and `stride_req_to_tokens_s = 1`, which holds for all four
checked Python shapes; the operational decode carries those two stride
equalities as hypotheses (instantiated at the concrete shapes in the per-case
corollaries).
-/

/-! ## Per-statement op-eval recipes (the tedious recipe layer)

Standalone, sorry-free `*_op_eval` recipe lemmas for every body statement of
`token_attn_mistral_surface`, mirroring the recipe-building patterns of
`VeriTile.Examples.AttentionForwardClosedForm`
(`qptrs_eval`/`vmask_eval`/`load_ptr_mask_real`/`mij_op_eval`). Each lemma takes
abstract register-readback hypotheses and a symbolic `BlockState`, and decodes a
single statement's `evalOp` to its closed-form tile. These feed the later
operational `exec` decode of the loop-invariant assembly.

`stride_pbs = 1 ∧ stride_req_to_tokens_s = 1` are carried as hypotheses where the
per-lane address arithmetic needs them (true for all four checked Python shapes). -/

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **Keystone `v_offs` recipe** (the blocked
`v_offs = cur_kv_head·stride_vh + offs_d[None,:]·stride_vd`, shape `[1, BD]`).
`expandDim ⟨0,_⟩` of `offs_d` broadcasts the head-dim arange to a row; the result
lane `(_, e)` holds the address `kvh·stride_vh + e·stride_vd`. Built with the
generic `evalOp_expandDim_ref_of_regs` recipe (mirrors `vptrs_eval`). -/
theorem mistral_voffs_eval (s : BlockState) (BD kvh stride_vh stride_vd : Nat)
    (hkvh : s.regs .nat [] "cur_kv_head" = some (Tile.scalar kvh))
    (hd : s.regs .nat [BD] "offs_d" = some (Tile.vec (fun e : Fin BD => e.val))) :
    evalOp (Op.add .nat (Broadcast.scalarL)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat stride_vh))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d"))
          (Op.constNat stride_vd))) s
      = some (⟨fun idx : TileIndex [1, BD] =>
          kvh * stride_vh + idx.2.1.val * stride_vd⟩ : Tile .nat [1, BD]) := by
  have hexp : @evalOp .nat [1, BD] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) s
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun e : Fin BD => e.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BD] ⟨0, by simp⟩ "offs_d" s _ hd
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hexp,
    hkvh, hd, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp [Tile.bop, Tile.vec, NumericDType.mul, NumericDType.add]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`v_loc_off` recipe** (the `Req_to_tokens` gather base
`v_loc_off = cur_batch_req_idx·stride_b + (cur_batch_start_index + offs_n)·stride_s`,
shape `[BLOCK_N]`). Lane `j` holds `reqIdx·stride_b + (startIdx + j)·stride_s`. -/
theorem mistral_vloc_off_eval (s : BlockState) (BN reqIdx startIdx stride_b stride_s : Nat)
    (hreq : s.regs .nat [] "cur_batch_req_idx" = some (Tile.scalar reqIdx))
    (hsi : s.regs .nat [] "cur_batch_start_index" = some (Tile.scalar startIdx))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch_req_idx") (Op.constNat stride_b))
        (Op.mul .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_start_index")
            (Op.ref .nat [BN] "offs_n"))
          (Op.constNat stride_s))) s
      = some (⟨fun idx : TileIndex [BN] =>
          reqIdx * stride_b + (startIdx + idx.1.val) * stride_s⟩ : Tile .nat [BN]) := by
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hreq, hsi, hn,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp [Tile.bop, Tile.vec, NumericDType.mul, NumericDType.add]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`p_offs` recipe** (the `Prob` load base
`p_offs = cur_head·stride_ph + (cur_batch_in_all_start_index + offs_n)·stride_pbs`,
shape `[BLOCK_N]`). Lane `j` holds `head·stride_ph + (attStart + j)·stride_pbs`. -/
theorem mistral_poffs_eval (s : BlockState) (BN head attStart stride_ph stride_pbs : Nat)
    (hhead : s.regs .nat [] "cur_head" = some (Tile.scalar head))
    (has : s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar attStart))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_ph))
        (Op.mul .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
            (Op.ref .nat [BN] "offs_n"))
          (Op.constNat stride_pbs))) s
      = some (⟨fun idx : TileIndex [BN] =>
          head * stride_ph + (attStart + idx.1.val) * stride_pbs⟩ : Tile .nat [BN]) := by
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hhead, has, hn,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp [Tile.bop, Tile.vec, NumericDType.mul, NumericDType.add]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`p_value` masked-load recipe** (`tl.load(Prob + p_offs + start_n,
mask=(start_n+offs_n) < cur_att_seq_len, other=0.0)`, shape `[BLOCK_N]`).

The `p_offs` register holds the per-lane `pOffset` (its lane `j` is
`pOffset … j`), `start_n = SN`, and `stride_pbs = 1`, so the loaded address at
lane `j` is `pOffset … j + SN = pOffset … (SN + j)` and the boundary mask
`SN + j < attSeqLen` is exactly `pMasked`'s guard. The result lane `j` is
therefore `pMasked … (SN + j)`. -/
theorem mistral_prob_load_eval (s : BlockState) (Prob : RegionName)
    (B_Att_Start_Loc B_Att_Seqlen : RegionName)
    (BN stride_ph stride_pbs SN : Nat) (hpbs : stride_pbs = 1)
    (hpoffs : s.regs .nat [BN] "p_offs" =
      some (Tile.vec (fun j : Fin BN => pOffset s B_Att_Start_Loc stride_ph stride_pbs j.val)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hsl : s.regs .nat [] "cur_att_seq_len" = some (Tile.scalar (attSeqLen s B_Att_Seqlen))) :
    evalOp (Op.load .real
        (MemAccess.region Prob
          (Op.add .nat Broadcast.scalarR (Op.ref .nat [BN] "p_offs") (Op.ref .nat [] "start_n")))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BN] "offs_n"))
            (Op.ref .nat [] "cur_att_seq_len"))
          (Op.broadcast (Op.const 0.0) [BN]))) s
      = some (⟨fun idx : TileIndex [BN] =>
          some (pMasked s Prob B_Att_Start_Loc B_Att_Seqlen stride_ph stride_pbs
            (SN + idx.1.val))⟩ : Tile .real [BN]) := by
  simp only [evalOp, hpoffs, hsn, hn, hsl, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, u⟩ := idx
  have haddr : pOffset s B_Att_Start_Loc stride_ph stride_pbs j.val + SN
      = pOffset s B_Att_Start_Loc stride_ph stride_pbs (SN + j.val) := by
    simp only [pOffset, hpbs]; ring
  simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar, NumericDType.add,
    ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex,
    BlockState.readMemValue_real, pMasked]
  rw [haddr]
  simp only [if_true, Region.cast_id]
  by_cases hlt : SN + j.val < attSeqLen s B_Att_Seqlen
  · simp only [hlt, decide_true, if_true, if_pos hlt]
  · simp only [hlt, decide_false, if_neg hlt, if_false, Bool.false_eq_true]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`v_loc` masked-gather recipe** (`tl.load(Req_to_tokens + v_loc_off +
start_n·stride_s, mask=(start_n+offs_n+cur_batch_start_index) < cur_batch_seq_len,
other=0)`, shape `[BLOCK_N]`, dtype `.nat`).

`v_loc_off` lane `j` is the gather base `reqIdx·stride_b + (startIdx + j)·stride_s`
(the `mistral_vloc_off_eval` form), `start_n = SN`, and
`stride_req_to_tokens_s = 1`, so the loaded address at lane `j` equals the
`vLoc … (SN + j)` address `reqIdx·stride_b + (startIdx + (SN + j))·stride_s`, and
the boundary mask `SN + j + startIdx < batchSeqLen` is exactly `vActive (SN + j)`.
The result lane `j` is `if vActive (SN+j) then vLoc … (SN+j) else 0`. -/
theorem mistral_reqloc_gather_eval (s : BlockState)
    (Req_to_tokens B_req_idx B_Seqlen : RegionName)
    (BN stride_b stride_s sliding_window SN : Nat) (hs : stride_s = 1)
    (hoff : s.regs .nat [BN] "v_loc_off" =
      some (Tile.vec (fun j : Fin BN =>
        reqIdx s B_req_idx * stride_b +
          (startIndex s B_Seqlen sliding_window + j.val) * stride_s)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hsi : s.regs .nat [] "cur_batch_start_index" =
      some (Tile.scalar (startIndex s B_Seqlen sliding_window)))
    (hsl : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (batchSeqLen s B_Seqlen))) :
    evalOp (Op.load .nat
        (MemAccess.region Req_to_tokens
          (Op.add .nat Broadcast.scalarR (Op.ref .nat [BN] "v_loc_off")
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat stride_s))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BN] "offs_n"))
              (Op.ref .nat [] "cur_batch_start_index"))
            (Op.ref .nat [] "cur_batch_seq_len"))
          (Op.broadcast (Op.constNat 0) [BN]))) s
      = some (⟨fun idx : TileIndex [BN] =>
          if vActive s B_Seqlen sliding_window (SN + idx.1.val) then
            vLoc s Req_to_tokens B_req_idx B_Seqlen stride_b stride_s sliding_window
              (SN + idx.1.val)
          else 0⟩ : Tile .nat [BN]) := by
  simp only [evalOp, hoff, hsn, hn, hsi, hsl, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, u⟩ := idx
  have haddr : reqIdx s B_req_idx * stride_b +
        (startIndex s B_Seqlen sliding_window + j.val) * stride_s + SN * stride_s
      = reqIdx s B_req_idx * stride_b +
        (startIndex s B_Seqlen sliding_window + (SN + j.val)) * stride_s := by
    subst hs; ring
  simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar, NumericDType.add,
    NumericDType.mul, ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex,
    Region.cast_cast, Region.cast_id, vLoc]
  rw [haddr]
  simp only [if_true]
  by_cases hlt : SN + j.val + startIndex s B_Seqlen sliding_window < batchSeqLen s B_Seqlen
  · have hv : vActive s B_Seqlen sliding_window (SN + j.val) := by
      simp only [vActive]; omega
    simp only [hlt, decide_true, if_true, if_pos hv]
  · have hv : ¬ vActive s B_Seqlen sliding_window (SN + j.val) := by
      simp only [vActive]; omega
    simp only [hlt, decide_false, if_neg hv, if_false, Bool.false_eq_true]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`v_value` 2D masked-gather recipe** (`tl.load(V + v_offs + v_loc[:,None]·stride_vbs,
mask=(start_n+offs_n[:,None]+cur_batch_start_index) < cur_batch_seq_len, other=0)`,
shape `[BLOCK_N, BLOCK_DMODEL]`). The mask is an `Op.remap` of a `[BN,1]` boundary
test broadcast across the head-dim columns.

`v_offs` lane `(_,d)` is `kvh·stride_vh + d·stride_vd` (the `mistral_voffs_eval`
form), `v_loc` lane `j` is the `mistral_reqloc_gather_eval` output
`if vActive(SN+j) then vLoc(SN+j) else 0`, and `kvh = pids 1 / kv_group_num`. On
an active lane `(j,d)` the loaded address is
`vLoc(SN+j)·stride_vbs + kvh·stride_vh + d·stride_vd = vOffset (SN+j) d`, and the
mask `SN+j+startIdx < batchSeqLen` is `vActive (SN+j)`; on an inactive lane the
`other = 0` fires. The result lane `(j,d)` is therefore `vMasked … (SN+j) d`. -/
theorem mistral_v_gather_eval (s : BlockState)
    (V : RegionName) (Req_to_tokens B_req_idx B_Seqlen : RegionName)
    (BN BD stride_b stride_s stride_vbs stride_vh stride_vd kv_group_num sliding_window SN : Nat)
    (hvoffs : s.regs .nat [1, BD] "v_offs" =
      some (⟨fun idx : TileIndex [1, BD] =>
          (s.pids 1 / kv_group_num) * stride_vh + idx.2.1.val * stride_vd⟩
        : Tile .nat [1, BD]))
    (hvloc : s.regs .nat [BN] "v_loc" =
      some (Tile.vec (fun j : Fin BN =>
        if vActive s B_Seqlen sliding_window (SN + j.val) then
          vLoc s Req_to_tokens B_req_idx B_Seqlen stride_b stride_s sliding_window (SN + j.val)
        else 0)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hsi : s.regs .nat [] "cur_batch_start_index" =
      some (Tile.scalar (startIndex s B_Seqlen sliding_window)))
    (hsl : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (batchSeqLen s B_Seqlen))) :
    evalOp (Op.load .real
        (MemAccess.region V
          (Op.add .nat (Broadcast.consL (Broadcast.consR Broadcast.nil))
            (Op.ref .nat [1, BD] "v_offs")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "v_loc"))
              (Op.constNat stride_vbs))))
        (MaskOpt.maskOther
          (Op.remap [BN, BD] (Broadcast.leftIndex (Broadcast.consSame (Broadcast.consL Broadcast.nil)))
            (Op.lt .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarR
                (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")))
                (Op.ref .nat [] "cur_batch_start_index"))
              (Op.ref .nat [] "cur_batch_seq_len")))
          (Op.broadcast (Op.const 0.0) [BN, BD]))) s
      = some (⟨fun idx : TileIndex [BN, BD] =>
          some (vMasked s V Req_to_tokens B_req_idx B_Seqlen stride_b stride_s stride_vbs
            stride_vh stride_vd kv_group_num sliding_window (SN + idx.1.val) idx.2.1.val)⟩
          : Tile .real [BN, BD]) := by
  have hexp : @evalOp .nat [BN, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "v_loc")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun j : Fin BN =>
          if vActive s B_Seqlen sliding_window (SN + j.val) then
            vLoc s Req_to_tokens B_req_idx B_Seqlen stride_b stride_s sliding_window (SN + j.val)
          else 0))) :=
    evalOp_expandDim_ref_of_regs .nat [BN] ⟨1, by simp⟩ "v_loc" s _ hvloc
  have hexpn : @evalOp .nat [BN, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun j : Fin BN => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BN] ⟨1, by simp⟩ "offs_n" s _ hn
  simp only [evalOp, hvoffs, hexp, hexpn, hsn, hsi, hsl, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, d, u⟩ := idx
  simp only [Tile.remap, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar,
    Tile.expandDim_data, TileShape.dropInsertedIndex_succ, TileShape.dropInsertedIndex_nil,
    TileShape.dropInsertedIndex_zero_cons, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast_cast,
    Region.cast_id, vMasked, vOffset]
  by_cases hlt : SN + j.val + startIndex s B_Seqlen sliding_window < batchSeqLen s B_Seqlen
  · have hv : vActive s B_Seqlen sliding_window (SN + j.val) := by
      simp only [vActive]; omega
    simp only [hlt, decide_true, if_true, if_pos hv, BlockState.readMemValue_real]
    congr 2
    ring
  · have hv : ¬ vActive s B_Seqlen sliding_window (SN + j.val) := by
      simp only [vActive]; omega
    simp only [hlt, decide_false, if_neg hv, if_false, Bool.false_eq_true]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`acc += tl.sum(p_value[:,None]·v_value, 0)` block-accumulation recipe**
(shape `[BLOCK_DMODEL]`). The `expandDim ⟨1,_⟩` lifts `p_value` to a `[BN,1]`
column, the elementwise `·v_value` broadcasts it across the head-dim, and the
`reduceSum` over axis 0 contracts the `BLOCK_N` window-token rows.

Given the accumulator carry `acc` lane `d = accVal d`, `p_value` lane `j =
pMasked (SN+j)`, and `v_value` lane `(j,d) = vMasked (SN+j) d`, the result lane
`d` is `accVal d + Σ_{j<BN} pMasked(SN+j)·vMasked(SN+j) d` — the `acc`-side of the
`partialAcc_block_succ` recurrence. Uses the `reduceSum`/`Finset.sum_congr`
machinery (the axis-0 analogue of `reduceSum_some`). -/
theorem mistral_acc_step_eval (s : BlockState) (BN BD : Nat)
    (accVal : Fin BD → ℝ) (pVal : Fin BN → ℝ) (vVal : Fin BN → Fin BD → ℝ)
    (hacc : s.regs .real [BD] "acc" =
      some (⟨fun idx : TileIndex [BD] => some (accVal idx.1)⟩ : Tile .real [BD]))
    (hp : s.regs .real [BN] "p_value" =
      some (⟨fun idx : TileIndex [BN] => some (pVal idx.1)⟩ : Tile .real [BN]))
    (hv : s.regs .real [BN, BD] "v_value" =
      some (⟨fun idx : TileIndex [BN, BD] => some (vVal idx.1 idx.2.1)⟩ : Tile .real [BN, BD])) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BD] "acc")
        (Op.reduceSum ⟨0, by simp⟩ Bool.false
          (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BN] "p_value"))
            (Op.ref .real [BN, BD] "v_value")))) s
      = some (⟨fun idx : TileIndex [BD] =>
          some (accVal idx.1 +
            ∑ j : Fin BN, pVal j * vVal j idx.1)⟩ : Tile .real [BD]) := by
  have hexp : @evalOp .real [BN, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BN] "p_value")) s
      = some (Tile.expandDim ⟨1, by simp⟩
          (⟨fun idx : TileIndex [BN] => some (pVal idx.1)⟩ : Tile .real [BN])) :=
    evalOp_expandDim_ref_of_regs .real [BN] ⟨1, by simp⟩ "p_value" s _ hp
  have hmul : evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BN] "p_value"))
        (Op.ref .real [BN, BD] "v_value")) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consL Broadcast.nil))
          (Tile.expandDim ⟨1, by simp⟩ (⟨fun idx : TileIndex [BN] => some (pVal idx.1)⟩ : Tile .real [BN]))
          (⟨fun idx : TileIndex [BN, BD] => some (vVal idx.1 idx.2.1)⟩ : Tile .real [BN, BD])) := by
    rw [evalOp_mul, hexp, evalOp_ref, hv]; rfl
  rw [evalOp_add, evalOp_ref, hacc]
  show (evalOp (Op.reduceSum ⟨0, by simp⟩ Bool.false
      (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BN] "p_value"))
        (Op.ref .real [BN, BD] "v_value"))) s).bind _ = _
  rw [evalOp_reduceSum, hmul]
  simp only [Option.bind_eq_bind, Option.bind_some, Tile.reduceSum_false]
  refine congrArg some ?_
  ext idx
  obtain ⟨d, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.reduceSumDrop_data, Tile.expandDim_data,
    TileShape.dropInsertedIndex_succ, TileShape.dropInsertedIndex_nil,
    TileShape.dropInsertedIndex_zero_cons, TileShape.insertAxisIndex, TileShape.axisDim,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]
  have hcoe : ∀ k : Fin BN, WithBot.realMul (some (pVal k)) (some (vVal k d))
      = ((pVal k * vVal k d : ℝ) : WithBot ℝ) := fun k => rfl
  rw [Finset.sum_congr rfl (fun k _ => hcoe k), ← WithBot.coe_sum]
  rfl

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **Unmasked `[BLOCK_DMODEL]` store readback recipe** (`tl.store(out_ptrs, acc)`).
The store is unmasked over the whole head-dim vector through the `out_ptrs`
pointer tile. Given `out_ptrs` reads back as the pointer tile
`(fun i => (Out, outAddr i))` with injective `outAddr`, and `acc` reads back as
the value tile lane `accVal`, every lane reads back its stored accumulator value:
`(stepStmt (store …) s).map (·.readMem Out (outAddr i)) = some (accVal i)`. The
per-lane readback is the `scatter_readback_nd` scatter/gather principle, the same
one used by `token_attn_mistral_final_store_slice_correct`. -/
theorem mistral_store_eval (s : BlockState) (Out : RegionName) (BD : Nat)
    (outAddr : TileIndex [BD] → Nat) (accVal : TileIndex [BD] → ℝ)
    (hout : s.regs .ptr [BD] "out_ptrs" =
      some (⟨fun i : TileIndex [BD] => ((Out : RegionName), outAddr i)⟩ : Tile .ptr [BD]))
    (hacc : s.regs .real [BD] "acc" =
      some (⟨fun i : TileIndex [BD] => some (accVal i)⟩ : Tile .real [BD]))
    (hinj : Function.Injective outAddr) (i : TileIndex [BD]) :
    (stepStmt (Stmt.store .real [BD]
        (MemAccess.ptr (Op.ref .ptr [BD] "out_ptrs"))
        (Op.ref .real [BD] "acc")
        (MaskOpt.none)) s).map (·.readMem Out (outAddr i))
      = some (accVal i) := by
  simp only [stepStmt, evalOp_ref, hout, hacc, Option.bind, Option.map]
  have hrw : ((TileShape.allIndices [BD]).foldl
      (fun acc k => acc.writeMem Out (outAddr k) (accVal k)) s).readMem Out (outAddr i)
      = accVal i := BlockState.scatter_readback_nd s outAddr accVal hinj i
  rw [← hrw]
  rfl

/-! ## Operational `exec` decode: prelude + loop-invariant + store readback

The genuine closed-form bridge. We thread the banked recipes through the prelude
decode, the `forRangeDyn` window loop (carrying `acc = partialAcc (counter)`), and
the final unmasked store readback, landing on `tokenAttnMistralClosedForm`.

`stride_pbs = 1 ∧ stride_req_to_tokens_s = 1` (true for all four checked Python
shapes) are carried as hypotheses for the per-lane address arithmetic. -/

/-- The loop invariant carried across the `range(0, cur_att_seq_len, BLOCK_N)`
window loop: the accumulator holds `partialAcc k`, and every loop-invariant
register holds its prelude-seeded value. -/
def mistralInvariant
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen : RegionName)
    (s0 : BlockState)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N : Nat)
    (k : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids
  ∧ s.mem = s0.mem
  ∧ (∀ rg o, s.undef rg o = 0)
  ∧ s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 0))
  ∧ s.regs .nat [] "cur_head" = some (Tile.scalar (s0.pids 1))
  ∧ s.regs .nat [] "cur_kv_head" = some (Tile.scalar (s0.pids 1 / kv_group_num))
  ∧ s.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
  ∧ s.regs .nat [BLOCK_DMODEL] "offs_d" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))
  ∧ s.regs .nat [] "cur_att_seq_len" = some (Tile.scalar (attSeqLen s0 B_Att_Seqlen))
  ∧ s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (batchSeqLen s0 B_Seqlen))
  ∧ s.regs .nat [] "cur_batch_start_index" =
      some (Tile.scalar (startIndex s0 B_Seqlen sliding_window))
  ∧ s.regs .nat [BLOCK_N] "v_loc_off" =
      some (Tile.vec (fun j : Fin BLOCK_N =>
        reqIdx s0 B_req_idx * stride_req_to_tokens_b +
          (startIndex s0 B_Seqlen sliding_window + j.val) * stride_req_to_tokens_s))
  ∧ s.regs .nat [BLOCK_N] "p_offs" =
      some (Tile.vec (fun j : Fin BLOCK_N =>
        pOffset s0 B_Att_Start_Loc stride_ph stride_pbs j.val))
  ∧ s.regs .nat [1, BLOCK_DMODEL] "v_offs" =
      some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] =>
        (s0.pids 1 / kv_group_num) * stride_vh + idx.2.1.val * stride_vd⟩ : Tile .nat [1, BLOCK_DMODEL])
  ∧ s.regs .real [BLOCK_DMODEL] "acc" =
      some (⟨fun idx : TileIndex [BLOCK_DMODEL] =>
        some (partialAcc s0 Prob V Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen
          B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
          stride_vbs stride_vh stride_vd kv_group_num sliding_window k idx.1.val)⟩
        : Tile .real [BLOCK_DMODEL])

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **preLoop** (14 prelude assigns): from a clean input state (`undef = 0`), the
prologue steps to a state satisfying `mistralInvariant … 0` — the loop-entry base
case (`acc = partialAcc 0 = 0`, all loop-invariant registers seeded). -/
theorem mistral_preLoop
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N : Nat)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts ((token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen stride_req_to_tokens_b
        stride_req_to_tokens_s stride_ph stride_pbs stride_vbs stride_vh stride_vd
        stride_obs stride_oh stride_od kv_group_num sliding_window BLOCK_DMODEL
        BLOCK_N).toAlgKernel.body.take 14) s = some s'
      ∧ mistralInvariant Prob V Out Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen
          B_Att_Seqlen s stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
          stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
          kv_group_num sliding_window BLOCK_DMODEL BLOCK_N 0 s' := by
  rw [show ((token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen stride_req_to_tokens_b
        stride_req_to_tokens_s stride_ph stride_pbs stride_vbs stride_vh stride_vd
        stride_obs stride_oh stride_od kv_group_num sliding_window BLOCK_DMODEL
        BLOCK_N).toAlgKernel.body.take 14)
      = [ Stmt.assign .nat [] "cur_batch" (Op.programId 0),
          Stmt.assign .nat [] "cur_head" (Op.programId 1),
          Stmt.assign .nat [] "cur_kv_head"
            (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat kv_group_num)),
          Stmt.assign .nat [BLOCK_N] "offs_n" (Op.arange BLOCK_N),
          Stmt.assign .nat [BLOCK_DMODEL] "offs_d" (Op.arange BLOCK_DMODEL),
          Stmt.assign .nat [] "cur_batch_seq_len"
            (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none),
          Stmt.assign .nat [] "cur_batch_start_index"
            (Op.where (Op.gt .nat Broadcast.nil
                (Op.sub .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len") (Op.constNat sliding_window))
                (Op.constNat 0))
              (Op.sub .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len") (Op.constNat sliding_window))
              (Op.constNat 0)),
          Stmt.assign .nat [] "cur_batch_in_all_start_index"
            (Op.load .nat (MemAccess.region B_Att_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none),
          Stmt.assign .nat [] "cur_batch_req_idx"
            (Op.load .nat (MemAccess.region B_req_idx (Op.ref .nat [] "cur_batch")) MaskOpt.none),
          Stmt.assign .nat [] "cur_att_seq_len"
            (Op.load .nat (MemAccess.region B_Att_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none),
          Stmt.assign .nat [BLOCK_N] "v_loc_off"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch_req_idx") (Op.constNat stride_req_to_tokens_b))
              (Op.mul .nat Broadcast.scalarR
                (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_start_index") (Op.ref .nat [BLOCK_N] "offs_n"))
                (Op.constNat stride_req_to_tokens_s))),
          Stmt.assign .nat [BLOCK_N] "p_offs"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_ph))
              (Op.mul .nat Broadcast.scalarR
                (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index") (Op.ref .nat [BLOCK_N] "offs_n"))
                (Op.constNat stride_pbs))),
          Stmt.assign .nat [1, BLOCK_DMODEL] "v_offs"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat stride_vh))
              (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d")) (Op.constNat stride_vd))),
          Stmt.assign .real [BLOCK_DMODEL] "acc" (Op.full [BLOCK_DMODEL] (Op.const 0)) ] from rfl]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat kv_group_num)) _
        = some (Tile.scalar (s.pids 1 / kv_group_num)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, BlockState.setReg_pids, Option.bind]
      refine congrArg some ?_
      ext idx
      simp only [Tile.bop, Tile.scalar, BlockState.setReg_pids, IntegralDType.nat_floorDiv]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BLOCK_N) _ = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) from by
      rw [evalOp_arange]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BLOCK_DMODEL) _ = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val)) from by
      rw [evalOp_arange]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
        = some (Tile.scalar (batchSeqLen s B_Seqlen)) from by
      simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind, Option.pure_def]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.where (Op.gt .nat Broadcast.nil
            (Op.sub .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len") (Op.constNat sliding_window))
            (Op.constNat 0))
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len") (Op.constNat sliding_window))
          (Op.constNat 0)) _
        = some (Tile.scalar (startIndex s B_Seqlen sliding_window)) from by
      simp only [evalOp_where, evalOp_gt, evalOp_sub, evalOp_constNat, evalOp_ref,
        BlockState.setReg_same, BlockState.setReg_ne_name, Option.bind]
      refine congrArg some ?_
      ext idx
      simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.scalar,
        Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, NumericDType.sub,
        startIndex, batchSeqLen]
      by_cases h : 0 < s.readMemValue .nat B_Seqlen (s.pids 0) - sliding_window
      · rw [if_pos (by simpa using h)]
      · simp only [Nat.not_lt, Nat.le_zero] at h
        rw [if_neg (by simp [h]), h]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region B_Att_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
        = some (Tile.scalar (attStartLoc s B_Att_Start_Loc)) from by
      simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind, Option.pure_def]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region B_req_idx (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
        = some (Tile.scalar (reqIdx s B_req_idx)) from by
      simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind, Option.pure_def]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region B_Att_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
        = some (Tile.scalar (attSeqLen s B_Att_Seqlen)) from by
      simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind, Option.pure_def]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mistral_vloc_off_eval _ BLOCK_N (reqIdx s B_req_idx) (startIndex s B_Seqlen sliding_window)
      stride_req_to_tokens_b stride_req_to_tokens_s (by simp) (by simp) (by simp [Tile.vec])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_ph))
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index") (Op.ref .nat [BLOCK_N] "offs_n"))
              (Op.constNat stride_pbs))) _
        = some (Tile.vec (fun j : Fin BLOCK_N =>
            pOffset s B_Att_Start_Loc stride_ph stride_pbs j.val)) from by
      rw [mistral_poffs_eval _ BLOCK_N (s.pids 1) (attStartLoc s B_Att_Start_Loc)
        stride_ph stride_pbs (by simp) (by simp) (by simp [Tile.vec])]
      refine congrArg some ?_
      ext idx
      simp [Tile.vec, pOffset, attStartLoc]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mistral_voffs_eval _ BLOCK_DMODEL (s.pids 1 / kv_group_num) stride_vh stride_vd
      (by simp) (by simp [Tile.vec])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BLOCK_DMODEL] (Op.const 0)) _
        = some (⟨fun idx : TileIndex [BLOCK_DMODEL] => some (0 : ℝ)⟩ : Tile .real [BLOCK_DMODEL]) from by
      simp only [evalOp_full, evalOp, Option.bind]
      refine congrArg some ?_
      ext idx
      rfl))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp, ?_, ?_, by simp, by simp, by simp, by simp [Tile.vec], by simp [Tile.vec],
    by simp, by simp, by simp, by simp [Tile.vec], by simp [Tile.vec], by simp, ?_⟩
  · funext rg o; simp
  · intro rg o; simp [hundef]
  · -- acc = partialAcc 0 = 0
    simp only [BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    simp [partialAcc]

/-- The 5-statement window-loop body of `token_attn_mistral_surface`, transcribed
(`start_n` no-op + `p_value`/`v_loc`/`v_value` masked loads + `acc += reduceSum`).
Independent of region names except `Prob`/`Req_to_tokens`/`V`. -/
def mistralLoopBody
    (Prob V Req_to_tokens : RegionName)
    (stride_req_to_tokens_s stride_vbs BLOCK_DMODEL BLOCK_N : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "start_n" (Op.ref .nat [] "start_n"),
    Stmt.assign .real [BLOCK_N] "p_value"
      (Op.load .real
        (MemAccess.region Prob
          (Op.add .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "p_offs") (Op.ref .nat [] "start_n")))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BLOCK_N] "offs_n"))
            (Op.ref .nat [] "cur_att_seq_len"))
          (Op.broadcast (Op.const 0.0) [BLOCK_N]))),
    Stmt.assign .nat [BLOCK_N] "v_loc"
      (Op.load .nat
        (MemAccess.region Req_to_tokens
          (Op.add .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "v_loc_off")
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat stride_req_to_tokens_s))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BLOCK_N] "offs_n"))
              (Op.ref .nat [] "cur_batch_start_index"))
            (Op.ref .nat [] "cur_batch_seq_len"))
          (Op.broadcast (Op.constNat 0) [BLOCK_N]))),
    Stmt.assign .real [BLOCK_N, BLOCK_DMODEL] "v_value"
      (Op.load .real
        (MemAccess.region V
          (Op.add .nat (Broadcast.consL (Broadcast.consR Broadcast.nil))
            (Op.ref .nat [1, BLOCK_DMODEL] "v_offs")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "v_loc"))
              (Op.constNat stride_vbs))))
        (MaskOpt.maskOther
          (Op.remap [BLOCK_N, BLOCK_DMODEL] (Broadcast.leftIndex (Broadcast.consSame (Broadcast.consL Broadcast.nil)))
            (Op.lt .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarR
                (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")))
                (Op.ref .nat [] "cur_batch_start_index"))
              (Op.ref .nat [] "cur_batch_seq_len")))
          (Op.broadcast (Op.const 0.0) [BLOCK_N, BLOCK_DMODEL]))),
    Stmt.assign .real [BLOCK_DMODEL] "acc"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BLOCK_DMODEL] "acc")
        (Op.reduceSum ⟨0, by simp⟩ Bool.false
          (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_N] "p_value"))
            (Op.ref .real [BLOCK_N, BLOCK_DMODEL] "v_value")))) ]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **Loop step**: the 5-statement body advances `acc = partialAcc k` to
`acc = partialAcc (k + BLOCK_N)` (via `partialAcc_block_succ`), preserving every
other loop-invariant register. The masked loads decode to `pMasked`/`vMasked` and
the `acc += reduceSum(p[:,None]·v)` is the `partialAcc_block_succ` recurrence. -/
theorem mistral_loop_step
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen : RegionName)
    (s0 : BlockState)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N : Nat)
    (hpbs : stride_pbs = 1) (hrts : stride_req_to_tokens_s = 1)
    (k : Nat) (st : BlockState)
    (hinv : mistralInvariant Prob V Out Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen
        B_Att_Seqlen s0 stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
        sliding_window BLOCK_DMODEL BLOCK_N k st) :
    ∃ st', stepStmts (mistralLoopBody Prob V Req_to_tokens stride_req_to_tokens_s stride_vbs
        BLOCK_DMODEL BLOCK_N) (st.setReg "start_n" .nat [] (Tile.scalar k)) = some st'
      ∧ mistralInvariant Prob V Out Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen
          B_Att_Seqlen s0 stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
          stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
          sliding_window BLOCK_DMODEL BLOCK_N (k + BLOCK_N) st' := by
  simp only [mistralInvariant] at hinv
  obtain ⟨hpids, hmem, hundef, hcb, hch, hckv, hn, hd, hsl, hbsl, hsi, hvoff, hpoff, hvoffs, hacc⟩ := hinv
  -- abbreviations for the loaded register memory (s0 = st memory)
  set sin := st.setReg "start_n" .nat [] (Tile.scalar k) with hsin
  have hmem' : sin.readMem = s0.readMem := by
    funext rg o; simp only [hsin, BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  have hreadval : ∀ (dt : TileDType) (rg : RegionName) (o : Nat),
      sin.readMemValue dt rg o = st.readMemValue dt rg o := by
    intro dt rg o; simp [hsin]
  -- register readbacks on sin (start_n now set, everything else carried)
  have hsn : sin.regs .nat [] "start_n" = some (Tile.scalar k) := by simp [hsin]
  have hpoffSin : sin.regs .nat [BLOCK_N] "p_offs" =
      some (Tile.vec (fun j : Fin BLOCK_N => pOffset s0 B_Att_Start_Loc stride_ph stride_pbs j.val)) := by
    rw [hsin]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hpoff
  have hnSin : sin.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
    rw [hsin]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hn
  have hslSin : sin.regs .nat [] "cur_att_seq_len" = some (Tile.scalar (attSeqLen s0 B_Att_Seqlen)) := by
    rw [hsin]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hsl
  have hvoffSin : sin.regs .nat [BLOCK_N] "v_loc_off" =
      some (Tile.vec (fun j : Fin BLOCK_N =>
        reqIdx s0 B_req_idx * stride_req_to_tokens_b +
          (startIndex s0 B_Seqlen sliding_window + j.val) * stride_req_to_tokens_s)) := by
    rw [hsin]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hvoff
  have hsiSin : sin.regs .nat [] "cur_batch_start_index" =
      some (Tile.scalar (startIndex s0 B_Seqlen sliding_window)) := by
    rw [hsin]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hsi
  have hbslSin : sin.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (batchSeqLen s0 B_Seqlen)) := by
    rw [hsin]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hbsl
  have hvoffsSin : sin.regs .nat [1, BLOCK_DMODEL] "v_offs" =
      some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] =>
        (s0.pids 1 / kv_group_num) * stride_vh + idx.2.1.val * stride_vd⟩ : Tile .nat [1, BLOCK_DMODEL]) := by
    rw [hsin]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hvoffs
  have haccSin : sin.regs .real [BLOCK_DMODEL] "acc" =
      some (⟨fun idx : TileIndex [BLOCK_DMODEL] =>
        some (partialAcc s0 Prob V Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen
          B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
          stride_vbs stride_vh stride_vd kv_group_num sliding_window k idx.1.val)⟩
        : Tile .real [BLOCK_DMODEL]) := by
    rw [hsin]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hacc
  -- sin agrees with s0 on pids and on all memory reads, so every memory-derived
  -- quantity (pOffset/attSeqLen/startIndex/batchSeqLen/reqIdx/vLoc/vOffset/pMasked/vMasked)
  -- evaluated at sin equals the s0 form carried by the invariant.
  have hpidsSin : sin.pids = s0.pids := by simp [hsin, hpids]
  have hrmvSin : ∀ (dt : TileDType) (rg : RegionName) (o : Nat),
      sin.readMemValue dt rg o = s0.readMemValue dt rg o := by
    intro dt rg o; rw [hreadval]
    cases dt <;>
      simp only [BlockState.readMemValue, BlockState.readMemAs, BlockState.readMemTyped,
        BlockState.readMem, hmem]
  -- state-congruence (general over any state sharing s0's memory and pids)
  have hpMaskedG : ∀ (sx : BlockState), sx.mem = s0.mem → sx.pids = s0.pids →
      ∀ nn, pMasked sx Prob B_Att_Start_Loc B_Att_Seqlen stride_ph stride_pbs nn
        = pMasked s0 Prob B_Att_Start_Loc B_Att_Seqlen stride_ph stride_pbs nn := by
    intro sx hmx hp nn
    have hrm : sx.readMem = s0.readMem := by
      funext rg o; simp only [BlockState.readMem, hmx]
    have hrv : ∀ (rg : RegionName) (o : Nat), sx.readMemValue .nat rg o = s0.readMemValue .nat rg o := by
      intro rg o; simp only [BlockState.readMemValue, BlockState.readMemTyped, hmx]
    simp only [pMasked, attSeqLen, pOffset, attStartLoc, hrv, hp]
    by_cases h : nn < s0.readMemValue .nat B_Att_Seqlen (s0.pids 0)
    · rw [if_pos h, if_pos h]; exact congrFun (congrFun hrm Prob) _
    · rw [if_neg h, if_neg h]
  have hvMaskedG : ∀ (sx : BlockState), sx.mem = s0.mem → sx.pids = s0.pids →
      ∀ nn dd, vMasked sx V Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
          stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num sliding_window nn dd
        = vMasked s0 V Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
          stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num sliding_window nn dd := by
    intro sx hmx hp nn dd
    have hrm : sx.readMem = s0.readMem := by
      funext rg o; simp only [BlockState.readMem, hmx]
    have hrv : ∀ (rg : RegionName) (o : Nat), sx.readMemValue .nat rg o = s0.readMemValue .nat rg o := by
      intro rg o; simp only [BlockState.readMemValue, BlockState.readMemTyped, hmx]
    simp only [vMasked, vActive, startIndex, batchSeqLen, vOffset, vLoc, reqIdx, hrv, hp]
    by_cases h : s0.readMemValue .nat B_Seqlen (s0.pids 0) - sliding_window + nn <
        s0.readMemValue .nat B_Seqlen (s0.pids 0)
    · rw [if_pos h, if_pos h]; exact congrFun (congrFun hrm V) _
    · rw [if_neg h, if_neg h]
  have hvLocActiveG : ∀ (sx : BlockState), sx.mem = s0.mem → sx.pids = s0.pids →
      ∀ nn, (if vActive sx B_Seqlen sliding_window nn then
              vLoc sx Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
                sliding_window nn else 0)
        = (if vActive s0 B_Seqlen sliding_window nn then
              vLoc s0 Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
                sliding_window nn else 0) := by
    intro sx hmx hp nn
    have hrv : ∀ (rg : RegionName) (o : Nat), sx.readMemValue .nat rg o = s0.readMemValue .nat rg o := by
      intro rg o; simp only [BlockState.readMemValue, BlockState.readMemTyped, hmx]
    simp only [vActive, startIndex, batchSeqLen, vLoc, reqIdx, hrv, hp]
  have hmetaG : ∀ (sx : BlockState), sx.mem = s0.mem → sx.pids = s0.pids →
      reqIdx sx B_req_idx = reqIdx s0 B_req_idx
      ∧ startIndex sx B_Seqlen sliding_window = startIndex s0 B_Seqlen sliding_window
      ∧ batchSeqLen sx B_Seqlen = batchSeqLen s0 B_Seqlen := by
    intro sx hmx hp
    have hrv : ∀ (rg : RegionName) (o : Nat), sx.readMemValue .nat rg o = s0.readMemValue .nat rg o := by
      intro rg o; simp only [BlockState.readMemValue, BlockState.readMemTyped, hmx]
    refine ⟨?_, ?_, ?_⟩
    · simp only [reqIdx, hrv, hp]
    · simp only [startIndex, batchSeqLen, hrv, hp]
    · simp only [batchSeqLen, hrv, hp]
  have hsinmem : sin.mem = s0.mem := by rw [hsin]; exact hmem
  have hpMasked := hpMaskedG sin hsinmem hpidsSin
  have hvMasked := hvMaskedG sin hsinmem hpidsSin
  have hattSeqLen : attSeqLen sin B_Att_Seqlen = attSeqLen s0 B_Att_Seqlen := by
    simp only [attSeqLen, hrmvSin, hpidsSin]
  have hbatchSeqLen : batchSeqLen sin B_Seqlen = batchSeqLen s0 B_Seqlen := by
    simp only [batchSeqLen, hrmvSin, hpidsSin]
  have hreqIdx : reqIdx sin B_req_idx = reqIdx s0 B_req_idx := by
    simp only [reqIdx, hrmvSin, hpidsSin]
  have hstartIndex : startIndex sin B_Seqlen sliding_window = startIndex s0 B_Seqlen sliding_window := by
    simp only [startIndex, hbatchSeqLen]
  have hpOffset : ∀ nn, pOffset sin B_Att_Start_Loc stride_ph stride_pbs nn
      = pOffset s0 B_Att_Start_Loc stride_ph stride_pbs nn := by
    intro nn; simp only [pOffset, attStartLoc, hrmvSin, hpidsSin]
  -- rewrite the sin-readbacks into recipe (sin) form
  have hpoffSin' : sin.regs .nat [BLOCK_N] "p_offs" =
      some (Tile.vec (fun j : Fin BLOCK_N => pOffset sin B_Att_Start_Loc stride_ph stride_pbs j.val)) := by
    rw [hpoffSin]; refine congrArg some ?_; ext idx; simp [Tile.vec, hpOffset]
  have hslSin' : sin.regs .nat [] "cur_att_seq_len" = some (Tile.scalar (attSeqLen sin B_Att_Seqlen)) := by
    rw [hslSin, hattSeqLen]
  have hvoffSin' : sin.regs .nat [BLOCK_N] "v_loc_off" =
      some (Tile.vec (fun j : Fin BLOCK_N =>
        reqIdx sin B_req_idx * stride_req_to_tokens_b +
          (startIndex sin B_Seqlen sliding_window + j.val) * stride_req_to_tokens_s)) := by
    rw [hvoffSin]; refine congrArg some ?_; ext idx; simp [Tile.vec, hreqIdx, hstartIndex]
  have hsiSin' : sin.regs .nat [] "cur_batch_start_index" =
      some (Tile.scalar (startIndex sin B_Seqlen sliding_window)) := by rw [hsiSin, hstartIndex]
  have hbslSin' : sin.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (batchSeqLen sin B_Seqlen)) := by
    rw [hbslSin, hbatchSeqLen]
  have hvoffsSin' : sin.regs .nat [1, BLOCK_DMODEL] "v_offs" =
      some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] =>
        (sin.pids 1 / kv_group_num) * stride_vh + idx.2.1.val * stride_vd⟩ : Tile .nat [1, BLOCK_DMODEL]) := by
    rw [hvoffsSin]; refine congrArg some ?_; ext idx; simp [hpidsSin]
  -- the start_n no-op re-sets start_n to k, leaving the state unchanged
  have hidem : sin.setReg "start_n" .nat [] (Tile.scalar k) = sin := by
    refine BlockState.ext (fun r o => rfl) (fun dtype shape name => ?_)
      (fun a => ?_) (fun r o => rfl) (fun a => rfl)
    · conv_rhs => rw [hsin]
      by_cases hname : name = "start_n"
      · subst hname
        by_cases hdt : dtype = .nat
        · subst hdt
          by_cases hsh : shape = ([] : TileShape)
          · subst hsh; simp
          · rw [BlockState.setReg_ne_shape (h := hsh) (hName := rfl) (hDType := rfl),
              BlockState.setReg_ne_shape (h := hsh) (hName := rfl) (hDType := rfl)]
        · rw [BlockState.setReg_ne_dtype (h := hdt) (hName := rfl),
            BlockState.setReg_ne_dtype (h := hdt) (hName := rfl)]
      · rw [BlockState.setReg_ne_name (h := hname)]
    · rw [BlockState.setReg_pids]
  -- helper: peel a non-"start_n"/"p_value"/"v_loc"/"v_value"/"acc" register from any setReg
  -- of these names; expressed via the chained states named below.
  set pvalT : Tile .real [BLOCK_N] :=
    ⟨fun idx : TileIndex [BLOCK_N] =>
      some (pMasked sin Prob B_Att_Start_Loc B_Att_Seqlen stride_ph stride_pbs (k + idx.1.val))⟩
    with hpvalT
  set s2 : BlockState := sin.setReg "p_value" .real [BLOCK_N] pvalT with hs2
  set vlocT : Tile .nat [BLOCK_N] :=
    ⟨fun idx : TileIndex [BLOCK_N] =>
      if vActive s2 B_Seqlen sliding_window (k + idx.1.val) then
        vLoc s2 Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
          sliding_window (k + idx.1.val)
      else 0⟩ with hvlocT
  set s3 : BlockState := s2.setReg "v_loc" .nat [BLOCK_N] vlocT with hs3
  set vvalT : Tile .real [BLOCK_N, BLOCK_DMODEL] :=
    ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
      some (vMasked s3 V Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
        stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num sliding_window
        (k + idx.1.val) idx.2.1.val)⟩ with hvvalT
  set s4 : BlockState := s3.setReg "v_value" .real [BLOCK_N, BLOCK_DMODEL] vvalT with hs4
  have hs2mem : s2.mem = s0.mem := by rw [hs2]; exact hsinmem
  have hs2pids : s2.pids = s0.pids := by rw [hs2, BlockState.setReg_pids]; exact hpidsSin
  have hs3mem : s3.mem = s0.mem := by rw [hs3]; exact hs2mem
  have hs3pids : s3.pids = s0.pids := by rw [hs3, BlockState.setReg_pids]; exact hs2pids
  unfold mistralLoopBody
  -- statement 1: start_n = start_n (no-op)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_n") sin = some (Tile.scalar k) from by rw [evalOp_ref]; exact hsn))]
  rw [hidem]
  -- statement 2: p_value masked load = pMasked (state is sin after hidem)
  rw [stepStmts.cons_some
    (show stepStmt _ sin = some s2 from stepStmt_assign_eq_some
      (mistral_prob_load_eval sin Prob B_Att_Start_Loc B_Att_Seqlen BLOCK_N stride_ph stride_pbs k hpbs
        (by simpa using hpoffSin') (by simp [hsn]) (by simpa using hnSin) hslSin'))]
  -- statement 3: v_loc masked gather (on s2)
  rw [stepStmts.cons_some
    (show stepStmt _ s2 = some s3 from stepStmt_assign_eq_some
      (mistral_reqloc_gather_eval s2 Req_to_tokens B_req_idx B_Seqlen BLOCK_N
        stride_req_to_tokens_b stride_req_to_tokens_s sliding_window k hrts
        (by
          rw [hs2]
          simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
          rw [hvoffSin]
          refine congrArg some ?_
          ext idx
          simp only [Tile.vec]
          rw [(hmetaG s2 hs2mem hs2pids).1, (hmetaG s2 hs2mem hs2pids).2.1])
        (by rw [hs2]; simp [hsn]) (by rw [hs2]; simpa using hnSin)
        (by
          rw [hs2]
          simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
          rw [hsiSin, (hmetaG s2 hs2mem hs2pids).2.1])
        (by
          rw [hs2]
          simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
          rw [hbslSin, (hmetaG s2 hs2mem hs2pids).2.2])))]
  -- statement 4: v_value 2D masked gather = vMasked (on s3)
  rw [stepStmts.cons_some
    (show stepStmt _ s3 = some s4 from stepStmt_assign_eq_some
      (mistral_v_gather_eval s3 V Req_to_tokens B_req_idx B_Seqlen BLOCK_N BLOCK_DMODEL
        stride_req_to_tokens_b stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num
        sliding_window k
        (by
          rw [hs3, hs2]
          simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
          rw [hvoffsSin]
          refine congrArg some ?_
          ext idx
          rw [hs3pids])
        (by
          rw [hs3]
          simp only [BlockState.setReg_same, hvlocT]
          refine congrArg some ?_
          ext idx
          simp only [Tile.vec]
          rw [hvLocActiveG s2 hs2mem hs2pids, hvLocActiveG s3 hs3mem hs3pids])
        (by rw [hs3, hs2]; simp [hsn]) (by rw [hs3, hs2]; simpa using hnSin)
        (by
          rw [hs3, hs2]
          simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
          rw [hsiSin, (hmetaG s3 hs3mem hs3pids).2.1])
        (by
          rw [hs3, hs2]
          simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
          rw [hbslSin, (hmetaG s3 hs3mem hs3pids).2.2])))]
  -- statement 5: acc += reduceSum(p[:,None]·v) = partialAcc (k + BLOCK_N)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mistral_acc_step_eval s4 BLOCK_N BLOCK_DMODEL
      (fun e => partialAcc s0 Prob V Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen
        B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd kv_group_num sliding_window k e.val)
      (fun j => pMasked s0 Prob B_Att_Start_Loc B_Att_Seqlen stride_ph stride_pbs (k + j.val))
      (fun j e => vMasked s0 V Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
        stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num sliding_window
        (k + j.val) e.val)
      (by rw [hs4, hs3, hs2]
          simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
            String.reduceEq]
          exact haccSin)
      (by rw [hs4, hs3, hs2]
          simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
            String.reduceEq, BlockState.setReg_same, hpvalT]
          refine congrArg some ?_
          ext idx
          exact congrArg some (hpMaskedG sin hsinmem hpidsSin _))
      (by rw [hs4]
          simp only [BlockState.setReg_same, hvvalT]
          refine congrArg some ?_
          ext idx
          obtain ⟨j, d, u⟩ := idx
          exact congrArg some (hvMaskedG s3 hs3mem hs3pids _ _))))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  -- the body only changed acc, p_value, v_loc, v_value, start_n; the rest carry
  -- final state = s4.setReg "acc" ...; peel acc/v_value/v_loc/p_value/start_n for carried regs
  have hpeel : ∀ {dt : TileDType} {sh : TileShape} (nm : RegName),
      nm ≠ "acc" → nm ≠ "v_value" → nm ≠ "v_loc" → nm ≠ "p_value" → nm ≠ "start_n" →
      ∀ (acctile : Tile .real [BLOCK_DMODEL]),
        (s4.setReg "acc" .real [BLOCK_DMODEL] acctile).regs dt sh nm = st.regs dt sh nm := by
    intro dt sh nm h1 h2 h3 h4 h5 acctile
    rw [BlockState.setReg_ne_name (h := h1), hs4,
      BlockState.setReg_ne_name (h := h2), hs3,
      BlockState.setReg_ne_name (h := h3), hs2,
      BlockState.setReg_ne_name (h := h4), hsin,
      BlockState.setReg_ne_name (h := h5)]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · funext a
    rw [BlockState.setReg_pids, hs4, BlockState.setReg_pids, hs3, BlockState.setReg_pids, hs2,
      BlockState.setReg_pids, hsin, BlockState.setReg_pids]
    exact congrFun hpids a
  · funext rg o
    rw [BlockState.setReg_mem, hs4, BlockState.setReg_mem, hs3, BlockState.setReg_mem, hs2,
      BlockState.setReg_mem, hsin, BlockState.setReg_mem]
    exact congrFun (congrFun hmem rg) o
  · intro rg o
    rw [BlockState.setReg_undef, hs4, BlockState.setReg_undef, hs3, BlockState.setReg_undef, hs2,
      BlockState.setReg_undef, hsin, BlockState.setReg_undef]
    exact hundef rg o
  · rw [hpeel _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hcb
  · rw [hpeel _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hch
  · rw [hpeel _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hckv
  · rw [hpeel _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hn
  · rw [hpeel _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hd
  · rw [hpeel _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hsl
  · rw [hpeel _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hbsl
  · rw [hpeel _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hsi
  · rw [hpeel _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hvoff
  · rw [hpeel _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hpoff
  · rw [hpeel _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hvoffs
  · -- acc = partialAcc (k + BLOCK_N)
    simp only [BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    simp only
    rw [partialAcc_block_succ]
    congr 1
    rw [Fin.sum_univ_eq_sum_range
      (fun j => pMasked s0 Prob B_Att_Start_Loc B_Att_Seqlen stride_ph stride_pbs (k + j) *
        vMasked s0 V Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
          stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num sliding_window
          (k + j) idx.1.val) BLOCK_N]

/-- The 4-statement postlude of `token_attn_mistral_surface`: the `.to(Out.dtype)`
cast (identity), `off_o`, `out_ptrs`, and the unmasked `[BLOCK_DMODEL]` store. -/
def mistralPostlude (Out : RegionName) (stride_obs stride_oh stride_od BLOCK_DMODEL : Nat) :
    List Stmt :=
  [ Stmt.assign .real [BLOCK_DMODEL] "acc" (Op.ref .real [BLOCK_DMODEL] "acc"),
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

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **Post-loop + store bridge**: after the window loop (invariant at `final ≥
attSeqLen`, so `acc = partialAcc final`), the cast (identity), `off_o`/`out_ptrs`
seeding, and the unmasked store write back `tokenAttnMistralClosedForm` to every
`Out[outOffset … i]` (via `mistral_store_eval` + `partialAcc_eq_PVValue`). -/
theorem mistral_postLoop
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen : RegionName)
    (s0 : BlockState)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N : Nat)
    (final : Nat) (hfinal : attSeqLen s0 B_Att_Seqlen ≤ final)
    (st : BlockState)
    (hinv : mistralInvariant Prob V Out Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen
        B_Att_Seqlen s0 stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
        sliding_window BLOCK_DMODEL BLOCK_N final st)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s0 stride_obs stride_oh stride_od i)) :
    ∃ sfin, stepStmts (mistralPostlude Out stride_obs stride_oh stride_od BLOCK_DMODEL) st = some sfin
      ∧ ∀ i : Fin BLOCK_DMODEL,
          sfin.readMem Out (outOffset s0 stride_obs stride_oh stride_od i)
            = tokenAttnMistralClosedForm s0 Prob V Req_to_tokens B_req_idx B_Att_Start_Loc
                B_Seqlen B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
                stride_pbs stride_vbs stride_vh stride_vd kv_group_num sliding_window
                BLOCK_DMODEL i := by
  simp only [mistralInvariant] at hinv
  obtain ⟨hpids, hmem, hundef, hcb, hch, hckv, hn, hd, hsl, hbsl, hsi, hvoff, hpoff, hvoffs, hacc⟩ := hinv
  set accFn : Fin BLOCK_DMODEL → ℝ := fun e =>
    partialAcc s0 Prob V Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen
      stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs stride_vbs stride_vh
      stride_vd kv_group_num sliding_window final e.val with haccFn
  set offFn : TileIndex [BLOCK_DMODEL] → Nat :=
    fun i => outOffset s0 stride_obs stride_oh stride_od i.1 with hoffFn
  unfold mistralPostlude
  -- statement 1: acc = (acc).to(...) — identity ref, re-sets acc to itself
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BLOCK_DMODEL] "acc") st =
        some (⟨fun idx : TileIndex [BLOCK_DMODEL] => some (accFn idx.1)⟩ : Tile .real [BLOCK_DMODEL])
      from by rw [evalOp_ref]; exact hacc))]
  -- statement 2: off_o = cur_batch*obs + cur_head*oh + offs_d*od = outOffset
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat stride_obs))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_oh)))
          (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_DMODEL] "offs_d") (Op.constNat stride_od))) _
        = some (Tile.vec (fun e : Fin BLOCK_DMODEL => offFn (e, PUnit.unit))) from by
      simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, BlockState.setReg_ne_name,
        ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq, hcb, hch, hd,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_
      ext idx
      simp only [Tile.bop, Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul,
        Broadcast.leftIndex, Broadcast.rightIndex, hoffFn, outOffset, dIndex]
      try ring))]
  -- statement 3: out_ptrs = Out + off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BLOCK_DMODEL] "off_o")) _
        = some (⟨fun i : TileIndex [BLOCK_DMODEL] => ((Out : RegionName), offFn i)⟩ : Tile .ptr [BLOCK_DMODEL])
      from by
      simp only [evalOp, evalOp_ref_setReg_same, Option.bind]
      refine congrArg some ?_
      ext idx
      · simp only [Tile.ptrAdd, Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex,
          Broadcast.rightIndex, Region.cast_id]
      · simp only [Tile.ptrAdd, Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex,
          Broadcast.rightIndex, Nat.zero_add]))]
  -- statement 4: unmasked store readback = accFn = partialAcc final = closed form
  -- the store executes to some sfin (mistral_store_eval gives the readback);
  -- obtain the post-store state explicitly.
  set stStore : BlockState :=
    ((((st.setReg "acc" .real [BLOCK_DMODEL]
        (⟨fun idx : TileIndex [BLOCK_DMODEL] => some (accFn idx.1)⟩ : Tile .real [BLOCK_DMODEL])).setReg
          "off_o" .nat [BLOCK_DMODEL] (Tile.vec (fun e : Fin BLOCK_DMODEL => offFn (e, PUnit.unit)))).setReg
        "out_ptrs" .ptr [BLOCK_DMODEL]
          (⟨fun i : TileIndex [BLOCK_DMODEL] => ((Out : RegionName), offFn i)⟩ : Tile .ptr [BLOCK_DMODEL])))
    with hstStore
  have hinj' : Function.Injective offFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : outOffset s0 stride_obs stride_oh stride_od a = outOffset s0 stride_obs stride_oh stride_od b := by
      simpa [hoffFn] using hab
    obtain rfl := hOutInj this; rfl
  have hout : stStore.regs .ptr [BLOCK_DMODEL] "out_ptrs" =
      some (⟨fun i : TileIndex [BLOCK_DMODEL] => ((Out : RegionName), offFn i)⟩ : Tile .ptr [BLOCK_DMODEL]) := by
    rw [hstStore]; simp
  have haccStore : stStore.regs .real [BLOCK_DMODEL] "acc" =
      some (⟨fun idx : TileIndex [BLOCK_DMODEL] => some (accFn idx.1)⟩ : Tile .real [BLOCK_DMODEL]) := by
    rw [hstStore]
    simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq,
      BlockState.setReg_same]
  -- the unmasked store leaves a definite final state
  obtain ⟨sfin, hstep⟩ : ∃ sfin, stepStmt (Stmt.store .real [BLOCK_DMODEL]
      (MemAccess.ptr (Op.ref .ptr [BLOCK_DMODEL] "out_ptrs"))
      (Op.ref .real [BLOCK_DMODEL] "acc") MaskOpt.none) stStore = some sfin := by
    simp only [stepStmt, evalOp_ref, hout, haccStore, Option.bind, Option.map]
    exact ⟨_, rfl⟩
  refine ⟨sfin, by rw [stepStmts.cons_some hstep, stepStmts.nil], ?_⟩
  intro i
  have hrb := mistral_store_eval stStore Out BLOCK_DMODEL offFn (fun i => accFn i.1) hout haccStore hinj'
    (i, PUnit.unit)
  rw [hstep] at hrb
  simp only [Option.map_some] at hrb
  have hval : sfin.readMem Out (offFn (i, PUnit.unit)) = accFn i := Option.some.inj hrb
  rw [show outOffset s0 stride_obs stride_oh stride_od i = offFn (i, PUnit.unit) from rfl, hval]
  -- accFn = partialAcc final = tokenAttnMistralClosedForm
  rw [haccFn]
  simp only
  rw [partialAcc_eq_PVValue s0 Prob V Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen
    B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs stride_vbs
    stride_vh stride_vd kv_group_num sliding_window final i.val hfinal]
  rfl

/-- Body decomposition: `prelude(14) ++ [forRangeDyn, postlude…]`. By `rfl`. -/
theorem mistral_body_split
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N : Nat) :
    (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
        B_Att_Start_Loc B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
        stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
        sliding_window BLOCK_DMODEL BLOCK_N).toAlgKernel.body
      = (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
          B_Att_Start_Loc B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
          stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
          sliding_window BLOCK_DMODEL BLOCK_N).toAlgKernel.body.take 14
        ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0) (Op.ref .nat [] "cur_att_seq_len") (Op.constNat BLOCK_N)
              (mistralLoopBody Prob V Req_to_tokens stride_req_to_tokens_s stride_vbs BLOCK_DMODEL BLOCK_N)
            :: mistralPostlude Out stride_obs stride_oh stride_od BLOCK_DMODEL) := by
  rfl

set_option maxHeartbeats 1000000 in
/-- **Genuine closed-form correctness for `token_attn_mistral` (general).** With
`stride_pbs = 1 ∧ stride_req_to_tokens_s = 1` and `0 < BLOCK_N` (true for all four
checked Python shapes), every lane of the `[BLOCK_DMODEL]` output store holds the
genuine PV-reduction closed form `tokenAttnMistralClosedForm` — i.e. the
probability-weighted, sliding-window value sum `Σ_n p[n]·v[v_loc[n], d]` — NOT the
kernel's own executed value. Proven sorry-free via `mistral_preLoop` (entry
invariant), `mistral_loop_step` fed to `forRangeDyn_inv` (one-block carry advance),
and `mistral_postLoop` (unmasked store readback = closed form). -/
theorem token_attn_mistral_closed_form_correct
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N : Nat)
    (hpbs : stride_pbs = 1) (hrts : stride_req_to_tokens_s = 1) (hBN : 0 < BLOCK_N)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i))
    (i : Fin BLOCK_DMODEL) :
    (match exec (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx B_Start_Loc
        B_Seqlen B_Att_Start_Loc B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
        stride_ph stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        kv_group_num sliding_window BLOCK_DMODEL BLOCK_N) s with
      | some s' => s'.readMem Out (outOffset s stride_obs stride_oh stride_od i)
      | none => (0.0 : ℝ)) =
      tokenAttnMistralClosedForm s Prob V Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen
        B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd kv_group_num sliding_window BLOCK_DMODEL i := by
  obtain ⟨s', hpre, hinv0⟩ := mistral_preLoop Prob V Out Req_to_tokens B_req_idx B_Start_Loc
    B_Seqlen B_Att_Start_Loc B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
    stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
    sliding_window BLOCK_DMODEL BLOCK_N s hundef
  -- the dynamic loop bound `cur_att_seq_len` reads as attSeqLen on the entry state
  have hslEntry : s'.regs .nat [] "cur_att_seq_len" = some (Tile.scalar (attSeqLen s B_Att_Seqlen)) := by
    simp only [mistralInvariant] at hinv0
    exact hinv0.2.2.2.2.2.2.2.2.1
  have hstop : evalOp (Op.ref .nat [] "cur_att_seq_len") s' = some (Tile.scalar (attSeqLen s B_Att_Seqlen)) := by
    rw [evalOp_ref]; exact hslEntry
  -- drive the window loop
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hinvFinal⟩ :=
    forRangeDyn_inv (idx := "start_n") (startOp := Op.constNat 0)
      (stopOp := Op.ref .nat [] "cur_att_seq_len") (stepOp := Op.constNat BLOCK_N)
      (start := 0) (stop := attSeqLen s B_Att_Seqlen) (step := BLOCK_N)
      (P := mistralInvariant Prob V Out Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen
        B_Att_Seqlen s stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
        sliding_window BLOCK_DMODEL BLOCK_N)
      (by simp) hstop (by simp) (by omega) hinv0
      (fun c st _ hP => mistral_loop_step Prob V Out Req_to_tokens B_req_idx B_Att_Start_Loc
        B_Seqlen B_Att_Seqlen s stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num sliding_window
        BLOCK_DMODEL BLOCK_N hpbs hrts c st hP)
  -- post-loop store readback
  obtain ⟨sfin, hPost, hRead⟩ := mistral_postLoop Prob V Out Req_to_tokens B_req_idx
    B_Att_Start_Loc B_Seqlen B_Att_Seqlen s stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
    stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
    sliding_window BLOCK_DMODEL BLOCK_N final hfinal sLoop hinvFinal hOutInj
  -- assemble exec = prelude ++ loop ++ postlude
  have hexec : exec (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx B_Start_Loc
      B_Seqlen B_Att_Start_Loc B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
      stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
      sliding_window BLOCK_DMODEL BLOCK_N) s = some sfin := by
    rw [exec, mistral_body_split, stepStmts.append_some hpre, stepStmts.cons_some hLoopStmt, hPost]
  rw [hexec]
  exact hRead i

/-- **Compute-facing genuine closed-form correctness for `token_attn_mistral`.**
The full surface kernel realizes the genuine PV-reduction closed form
`tokenAttnMistralClosedForm` at every `[BLOCK_DMODEL]` output lane (under
`stride_pbs = 1 ∧ stride_req_to_tokens_s = 1`, `0 < BLOCK_N`, clean `undef`, and
output-offset injectivity). -/
theorem token_attn_mistral_closed_form_compute_correct
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N : Nat)
    (hpbs : stride_pbs = 1) (hrts : stride_req_to_tokens_s = 1) (hBN : 0 < BLOCK_N)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
        stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        kv_group_num sliding_window BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i =>
        tokenAttnMistralClosedForm s Prob V Req_to_tokens B_req_idx B_Att_Start_Loc
          B_Seqlen B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
          stride_pbs stride_vbs stride_vh stride_vd kv_group_num sliding_window
          BLOCK_DMODEL i) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [token_attn_mistral_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := token_attn_mistral_closed_form_correct Prob V Out Req_to_tokens B_req_idx
    B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen stride_req_to_tokens_b
    stride_req_to_tokens_s stride_ph stride_pbs stride_vbs stride_vh stride_vd stride_obs
    stride_oh stride_od kv_group_num sliding_window BLOCK_DMODEL BLOCK_N hpbs hrts hBN s
    hundef hOutInj i
  rw [show exec _ s = some s' from hExec] at h
  exact h

/-! ### ════════ ★ MAIN THEOREM ★ ════════

**Dimension-general headline.** For *symbolic* output width `BLOCK_DMODEL`, block
size `BLOCK_N`, sliding window `sliding_window`, and *all* strides, the full
sliding-window reduce-V surface kernel (1) lowers to the algorithm layer, and (2)
realizes the genuine, self-reference-free PV-reduction closed form
`tokenAttnMistralClosedForm` (which incorporates `sliding_window` symbolically via
`startIndex`/`vActive`) at every `[BLOCK_DMODEL]` output lane.

Honest side-conditions only: `0 < BLOCK_N`, the contiguous
layout hyps `stride_pbs = 1` / `stride_req_to_tokens_s = 1` (faithful to the
checked test's contiguous `Prob`/`Req_to_tokens`), output-offset injectivity
`hOutInj`, and a clean `undef` state `hundef`. -/
specification token_attn_mistral_output_summary_general
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N : Nat)
    (hpbs : stride_pbs = 1) (hrts : stride_req_to_tokens_s = 1) (hBN : 0 < BLOCK_N)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    (∃ alg, (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
        stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        kv_group_num sliding_window BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i : Fin BLOCK_DMODEL =>
        tokenAttnMistralClosedForm s Prob V Req_to_tokens B_req_idx B_Att_Start_Loc
          B_Seqlen B_Att_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
          stride_ph stride_pbs stride_vbs stride_vh stride_vd kv_group_num
          sliding_window BLOCK_DMODEL i)) := by
  refine ⟨?_, ?_⟩
  · exact token_attn_mistral_surface_toAlgorithm_supported Prob V Out
      Req_to_tokens B_req_idx B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N
  · exact token_attn_mistral_closed_form_compute_correct Prob V Out
      Req_to_tokens B_req_idx B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N hpbs hrts hBN s hundef hOutInj

end Correct_without_Rounding

/-! # ══════════ IO-face (`⊨[R]`) — the gather skin ══════════ -/

section IOFace

open scoped VeriTile.Triton.StreamMetaGatherMasked3DKernelIO₂

/-! ## Slot table, step budget and IO signature

`token_attn_mistral` is a consumer of `StreamMetaGatherMasked3DKernelIO₂`,
the **gather-indexed** two-stream fold skin. The 2-D grid is
`(cur_batch, cur_head) = (pid₀, pid₁)` (`pid₂` unused); the **four** `.nat`
metadata slots are read at cell `cur_batch = pid₀` of their own regions, in
the kernel's own load order — slot `0` = `B_Seqlen[pid₀]`
(`cur_batch_seq_len`, the sliding window's right edge), slot `1` =
`B_Att_Start_Loc[pid₀]` (`cur_batch_in_all_start_index`, the `Prob` row
base), slot `2` = `B_req_idx[pid₀]` (`cur_batch_req_idx`, the page table's
row), slot `3` = `B_Att_Seqlen[pid₀]` (`cur_att_seq_len`, the loop's dynamic
trip count).

The sliding-window left edge is `cur_batch_start_index = m 0 − sliding_window`,
transcribed **verbatim** as the ℕ-truncated subtraction the kernel spells
(the Python's `tl.maximum(·, 0)` is the identity on ℕ, as
`startIndex`'s docstring records).

The **gather channel** is the paged-KV page table `Req_to_tokens` at
`gty = .nat` with `gother = 0` (the kernel's own `other=0`): per step `t`
the kernel loads `BLOCK_N` page indices, masked by
`(m 0 − sliding_window) + (t·BN + jL) < m 0`, and those *values* address the
`V` load. Unlike the softmax_reducev exemplar this port's `V` load carries
**the same mask as the gather**, so a dead lane never dereferences the
`other=` sentinel and no bound on `gother` is needed anywhere.

Stride abbreviations used throughout this section (the exact stack's long
names): `srtb`/`srts` = `stride_req_to_tokens_b`/`_s`, `sph`/`spbs` =
`stride_ph`/`stride_pbs`, `svbs`/`svh`/`svd` = the three `V` strides,
`sobs`/`soh`/`sod` = the three `Out` strides, `kvg` = `kv_group_num`,
`sw` = `sliding_window`. -/

/-- Slot-region table of the four per-batch metadata slots, in the kernel's
own load order (`B_Seqlen`, `B_Att_Start_Loc`, `B_req_idx`, `B_Att_Seqlen`).
A shared def, never an inline `match` in a window position. -/
def mistralIOMetaBuf (B_Seqlen B_Att_Start_Loc B_req_idx B_Att_Seqlen : Region .nat) :
    Fin 4 → RegionName
  | ⟨0, _⟩ => B_Seqlen.cast
  | ⟨1, _⟩ => B_Att_Start_Loc.cast
  | ⟨2, _⟩ => B_req_idx.cast
  | ⟨_ + 3, _⟩ => B_Att_Seqlen.cast

/-- Every live window token's block index is a legal step index. This is the
only use of the `pre` budget: the surface has no `max_input_len` argument, so
`T` is an io-level parameter and `m 3 ≤ T · BLOCK_N` is a *disclosed* launch
restriction rather than a derived fact. -/
theorem mistralIO_step_lt (n S T BLOCK_N : Nat) (hBN : 0 < BLOCK_N) (hn : n < S)
    (hle : S ≤ T * BLOCK_N) : n / BLOCK_N < T :=
  (Nat.div_lt_iff_lt_mul hBN).mpr (by omega)

/-- **Gather-indexed IO signature** of `token_attn_mistral` on the
gather-indexed two-stream fold skin (S1: PV-accumulation fold + terminal
store, 2-D pid grid `(cur_batch, cur_head)`), at fully **symbolic per-axis
strides** and symbolic `sliding_window`.

Windows transcribe the kernel's pointer arithmetic VERBATIM, with the loaded
slot vector `m` in place of the in-state metadata reads:

* `gread` (`Req_to_tokens`, the page table): lane `jL` of step `t` reads
  `m 2 · srtb + ((m 0 − sw) + (t·BN + jL)) · srts`; `gmask` is
  `(m 0 − sw) + (t·BN + jL) < m 0` and `gother = 0`.
* `read1` (`Prob`): lane `jL` of step `t` reads
  `pid₁ · sph + (m 1 + (t·BN + jL)) · spbs`, masked by `t·BN + jL < m 3`
  (the `other = 0.0` lanes are never read by the spec).
* `read2` (`V`, the **gather-addressed** value rows, lane `j = (jL, d)`
  row-major over `[BLOCK_N, BLOCK_DMODEL]`) reads
  `G t jL · svbs + (pid₁ / kvg) · svh + d · svd`. `mask2` repeats `gmask`'s
  predicate on the row coordinate — the Python `V` load carries exactly the
  gather's mask, so dead lanes are never dereferenced.
* `write` (`Out`, the terminal store): lane `i` writes
  `pid₀ · sobs + pid₁ · soh + i · sod`, `writeMask ≡ True` (the store is
  unmasked over the whole `[BLOCK_DMODEL]` vector).

`pre` is the launch-legality field: `m 3 ≤ T · BLOCK_N`. The surface takes no
`max_input_len` argument, so the pid-free step budget `T` cannot be derived
from a host parameter; it is a **new io-level parameter** and `pre` is the
honest disclosure of the launch restriction it imposes.

`outDType` is the `.real` default: the terminal `tl.store` is untyped and the
`.to(Out.dtype.element_ty)` cast lowers to a self-assign, so there is no
quantization event. -/
def mistralIO (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (srtb srts sph spbs svbs svh svd sobs soh sod kvg sw BLOCK_DMODEL BLOCK_N T : Nat) :
    StreamMetaGatherMasked3DKernelIO₂ where
  kernel := token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx B_Start_Loc
    B_Seqlen B_Att_Start_Loc B_Att_Seqlen srtb srts sph spbs svbs svh svd sobs soh sod
    kvg sw BLOCK_DMODEL BLOCK_N
  inp1 := Prob
  inp2 := V
  out := Out
  nMeta := 4
  sty := fun _ => ChanTy.nat
  mbuf := mistralIOMetaBuf B_Seqlen B_Att_Start_Loc B_req_idx B_Att_Seqlen
  mwin := fun _ pid₀ _ _ => pid₀
  gbuf := Req_to_tokens.cast
  gty := ChanTy.nat
  Bg := BLOCK_N
  gother := 0
  T := T
  B1 := BLOCK_N
  B2 := BLOCK_N * BLOCK_DMODEL
  C := BLOCK_DMODEL
  outDType := .real
  pre := fun _ _ _ m => m (⟨3, by omega⟩ : Fin 4) ≤ T * BLOCK_N
  gread := fun _ _ _ m t jL =>
    m (⟨2, by omega⟩ : Fin 4) * srtb
      + ((m (⟨0, by omega⟩ : Fin 4) - sw) + (t.val * BLOCK_N + jL.val)) * srts
  gmask := fun _ _ _ m t jL =>
    (m (⟨0, by omega⟩ : Fin 4) - sw) + (t.val * BLOCK_N + jL.val)
      < m (⟨0, by omega⟩ : Fin 4)
  read1 := fun _ pid₁ _ m t jL =>
    pid₁ * sph + (m (⟨1, by omega⟩ : Fin 4) + (t.val * BLOCK_N + jL.val)) * spbs
  mask1 := fun _ _ _ m t jL => t.val * BLOCK_N + jL.val < m (⟨3, by omega⟩ : Fin 4)
  read2 := fun _ pid₁ _ _ G t j =>
    G t (Lane2D.decode j).1 * svbs + (pid₁ / kvg) * svh
      + (Lane2D.decode j).2.1.val * svd
  mask2 := fun _ _ _ m t j =>
    (m (⟨0, by omega⟩ : Fin 4) - sw) + (t.val * BLOCK_N + (Lane2D.decode j).1.val)
      < m (⟨0, by omega⟩ : Fin 4)
  write := fun pid₀ pid₁ _ _ i => pid₀ * sobs + pid₁ * soh + i.val * sod
  writeMask := fun _ _ _ _ _ => True

/-! ## Stream-indexed tiles and the streamed closed form -/

/-- The probability of window token `n` read off the first stream: step
`n / BLOCK_N`, lane `n % BLOCK_N` (`0` past the step budget — unreachable at
any `pre`-legal launch). -/
noncomputable def mistralIOprob (BLOCK_N T : Nat) (hBN : 0 < BLOCK_N)
    (xs : Fin T → Fin BLOCK_N → ℝ) (n : Nat) : ℝ :=
  if h : n / BLOCK_N < T then
    xs ⟨n / BLOCK_N, h⟩ ⟨n % BLOCK_N, Nat.mod_lt _ hBN⟩
  else 0

/-- The gathered value row of window token `n`, channel `d`, read off the
second stream at lane `(n % BLOCK_N, d)`. The page indirection lives in the
*window* (`read2` eats `G`), so the stream cell is already the gathered
row. -/
noncomputable def mistralIOval (BLOCK_N BLOCK_DMODEL T : Nat) (hBN : 0 < BLOCK_N)
    (ys : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ) (n : Nat)
    (d : Fin BLOCK_DMODEL) : ℝ :=
  if h : n / BLOCK_N < T then
    ys ⟨n / BLOCK_N, h⟩
      (Lane2D.encode (⟨n % BLOCK_N, Nat.mod_lt _ hBN⟩, d, PUnit.unit))
  else 0

/-- **The streamed closed form**: `tokenAttnMistralPVValue` restated over the
two streamed tiles — `out[d] = Σ_{n < attSeq} p[n] · v[n, d]`, where a token
outside the sliding window (`¬ startIdx + n < batchSeq`) contributes `0`
because its masked `V` load reads `0`. -/
noncomputable def mistralIOSpec (BLOCK_N BLOCK_DMODEL T attSeq batchSeq sw : Nat)
    (hBN : 0 < BLOCK_N) (xs : Fin T → Fin BLOCK_N → ℝ)
    (ys : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ) (d : Fin BLOCK_DMODEL) : ℝ :=
  ∑ n ∈ Finset.range attSeq,
    mistralIOprob BLOCK_N T hBN xs n *
      (if batchSeq - sw + n < batchSeq then
        mistralIOval BLOCK_N BLOCK_DMODEL T hBN ys n d
      else 0)

/-! ## The spec-equals-genuine bridge

The exact stack's genuine closed form names the kernel's own memory-side
reads (`pOffset`/`vOffset` — the latter **inlines the gather** `vLoc` inside
the `V` address). The io face states the same value over the pinned streams;
the bridge rewrites `vLoc ↦ G t jL` through the gather pin, which is the one
real proof-content delta of the gather skin. Because `mask2` repeats
`gmask`'s predicate, only the pin's **active** leg is ever used. -/

set_option maxHeartbeats 1600000 in
/-- **Streamed spec = genuine closed form.** Under the slot/gather/data pins
the exact headline's `tokenAttnMistralPVValue` is the streamed
`mistralIOSpec`. -/
theorem mistralIOSpec_eq_genuine
    (Prob V Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen : RegionName)
    (srtb srts sph spbs svbs svh svd kvg sw BLOCK_DMODEL BLOCK_N T : Nat)
    (hBN : 0 < BLOCK_N) (s₀ : BlockState) (S startloc batchS reqI : Nat)
    (hS : attSeqLen s₀ B_Att_Seqlen = S)
    (hstart : attStartLoc s₀ B_Att_Start_Loc = startloc)
    (hbatch : batchSeqLen s₀ B_Seqlen = batchS)
    (hreq : reqIdx s₀ B_req_idx = reqI)
    (hSle : S ≤ T * BLOCK_N)
    (G : Fin T → Fin BLOCK_N → Nat) (xs : Fin T → Fin BLOCK_N → ℝ)
    (ys : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (hg : ∀ (t : Fin T) (jL : Fin BLOCK_N),
      (batchS - sw) + (t.val * BLOCK_N + jL.val) < batchS →
      s₀.readMemValue .nat Req_to_tokens
          (reqI * srtb + ((batchS - sw) + (t.val * BLOCK_N + jL.val)) * srts) = G t jL)
    (hx : ∀ (t : Fin T) (jL : Fin BLOCK_N), t.val * BLOCK_N + jL.val < S →
      s₀.readMem Prob (s₀.pids 1 * sph + (startloc + (t.val * BLOCK_N + jL.val)) * spbs)
        = xs t jL)
    (hy : ∀ (t : Fin T) (j : Fin (BLOCK_N * BLOCK_DMODEL)),
      (batchS - sw) + (t.val * BLOCK_N + (Lane2D.decode j).1.val) < batchS →
      s₀.readMem V (G t (Lane2D.decode j).1 * svbs + (s₀.pids 1 / kvg) * svh
          + (Lane2D.decode j).2.1.val * svd) = ys t j)
    (d : Fin BLOCK_DMODEL) :
    tokenAttnMistralPVValue s₀ Prob V Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen
        B_Att_Seqlen srtb srts sph spbs svbs svh svd kvg sw d.val
      = mistralIOSpec BLOCK_N BLOCK_DMODEL T S batchS sw hBN xs ys d := by
  unfold tokenAttnMistralPVValue mistralIOSpec
  rw [hS]
  refine Finset.sum_congr rfl (fun n hn => ?_)
  rw [Finset.mem_range] at hn
  have hstepT : n / BLOCK_N < T := mistralIO_step_lt n S T BLOCK_N hBN hn hSle
  have hmodlt : n % BLOCK_N < BLOCK_N := Nat.mod_lt _ hBN
  have hsplit : (⟨n / BLOCK_N, hstepT⟩ : Fin T).val * BLOCK_N
      + (⟨n % BLOCK_N, hmodlt⟩ : Fin BLOCK_N).val = n := Nat.div_add_mod' n BLOCK_N
  -- the probability leg
  have hp : s₀.readMem Prob (pOffset s₀ B_Att_Start_Loc sph spbs n)
      = mistralIOprob BLOCK_N T hBN xs n := by
    have h := hx ⟨n / BLOCK_N, hstepT⟩ ⟨n % BLOCK_N, hmodlt⟩ (by rw [hsplit]; exact hn)
    rw [hsplit] at h
    simp only [mistralIOprob, dif_pos hstepT]
    rw [← h, pOffset, hstart]
  -- the gathered value leg
  have hv : (if vActive s₀ B_Seqlen sw n then
        s₀.readMem V (vOffset s₀ Req_to_tokens B_req_idx B_Seqlen srtb srts svbs svh svd
          kvg sw n d.val)
      else (0.0 : ℝ))
      = (if (batchS - sw) + n < batchS then
          mistralIOval BLOCK_N BLOCK_DMODEL T hBN ys n d else 0) := by
    have hact : vActive s₀ B_Seqlen sw n ↔ (batchS - sw) + n < batchS := by
      simp only [vActive, startIndex, hbatch]
    by_cases hlive : (batchS - sw) + n < batchS
    · rw [if_pos (hact.mpr hlive), if_pos hlive]
      have hidx : vLoc s₀ Req_to_tokens B_req_idx B_Seqlen srtb srts sw n
          = G ⟨n / BLOCK_N, hstepT⟩ ⟨n % BLOCK_N, hmodlt⟩ := by
        have h := hg ⟨n / BLOCK_N, hstepT⟩ ⟨n % BLOCK_N, hmodlt⟩ (by rw [hsplit]; exact hlive)
        rw [hsplit] at h
        rw [vLoc, hreq, startIndex, hbatch]
        exact h
      have hyn := hy ⟨n / BLOCK_N, hstepT⟩
        (Lane2D.encode ((⟨n % BLOCK_N, hmodlt⟩, d, PUnit.unit)
          : TileIndex [BLOCK_N, BLOCK_DMODEL]))
      rw [Lane2D.decode_encode] at hyn
      rw [hsplit] at hyn
      simp only [mistralIOval, dif_pos hstepT]
      rw [← hyn hlive, vOffset, hidx]
    · rw [if_neg (fun h => hlive (hact.mp h)), if_neg hlive]
      norm_num
  rw [hp, hv]

/-! ## Flat-bridge coverage and cast-freedom

The port is cast-free in the `R` sense: every load/store is at `.real` or
`.nat`, `tl.multiple_of` and `.to(Out.dtype.element_ty)` are self-assigns,
and there is no `Op.castFloat`, no `ptrSub` and no block pointer anywhere in
the file. So `stepStmtsR R` collapses verbatim onto the exact stepper on
every segment, and no `R.round … = id` boundary hypothesis is needed. -/

/-- The 14-statement prelude of `token_attn_mistral_surface`, transcribed
(the literal list `mistral_preLoop` decodes; `rfl`-equal to
`…toAlgKernel.body.take 14`). -/
private def mistralIOPrelude (B_req_idx : Region .nat)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (srtb srts sph spbs svh svd kvg sw BLOCK_DMODEL BLOCK_N : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "cur_batch" (Op.programId 0),
    Stmt.assign .nat [] "cur_head" (Op.programId 1),
    Stmt.assign .nat [] "cur_kv_head"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat kvg)),
    Stmt.assign .nat [BLOCK_N] "offs_n" (Op.arange BLOCK_N),
    Stmt.assign .nat [BLOCK_DMODEL] "offs_d" (Op.arange BLOCK_DMODEL),
    Stmt.assign .nat [] "cur_batch_seq_len"
      (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "cur_batch_start_index"
      (Op.where (Op.gt .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len") (Op.constNat sw))
          (Op.constNat 0))
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len") (Op.constNat sw))
        (Op.constNat 0)),
    Stmt.assign .nat [] "cur_batch_in_all_start_index"
      (Op.load .nat (MemAccess.region B_Att_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "cur_batch_req_idx"
      (Op.load .nat (MemAccess.region B_req_idx (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "cur_att_seq_len"
      (Op.load .nat (MemAccess.region B_Att_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [BLOCK_N] "v_loc_off"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch_req_idx") (Op.constNat srtb))
        (Op.mul .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_start_index")
            (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.constNat srts))),
    Stmt.assign .nat [BLOCK_N] "p_offs"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat sph))
        (Op.mul .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
            (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.constNat spbs))),
    Stmt.assign .nat [1, BLOCK_DMODEL] "v_offs"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat svh))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
          (Op.constNat svd))),
    Stmt.assign .real [BLOCK_DMODEL] "acc" (Op.full [BLOCK_DMODEL] (Op.const 0)) ]

/-- The prelude list *is* the surface's first 14 statements. By `rfl`. -/
private theorem mistralIO_prelude_eq (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (srtb srts sph spbs svbs svh svd sobs soh sod kvg sw BLOCK_DMODEL BLOCK_N : Nat) :
    (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
        B_Att_Start_Loc B_Att_Seqlen srtb srts sph spbs svbs svh svd sobs soh sod kvg sw
        BLOCK_DMODEL BLOCK_N).toAlgKernel.body.take 14
      = mistralIOPrelude B_req_idx B_Seqlen B_Att_Start_Loc B_Att_Seqlen srtb srts sph spbs
          svh svd kvg sw BLOCK_DMODEL BLOCK_N := rfl

/-- Body decomposition against the literal prelude list. By `rfl`. -/
private theorem mistralIO_body_split (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (srtb srts sph spbs svbs svh svd sobs soh sod kvg sw BLOCK_DMODEL BLOCK_N : Nat) :
    (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
        B_Att_Start_Loc B_Att_Seqlen srtb srts sph spbs svbs svh svd sobs soh sod kvg sw
        BLOCK_DMODEL BLOCK_N).toAlgKernel.body
      = mistralIOPrelude B_req_idx B_Seqlen B_Att_Start_Loc B_Att_Seqlen srtb srts sph spbs
            svh svd kvg sw BLOCK_DMODEL BLOCK_N
        ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0) (Op.ref .nat [] "cur_att_seq_len")
              (Op.constNat BLOCK_N)
              (mistralLoopBody Prob V Req_to_tokens srts svbs BLOCK_DMODEL BLOCK_N)
            :: mistralPostlude Out sobs soh sod BLOCK_DMODEL) := rfl

set_option maxHeartbeats 4000000 in
/-- The surface sits inside the flat-memory bridge's covered fragment. -/
private theorem mistralIO_flattenOk (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (srtb srts sph spbs svbs svh svd sobs soh sod kvg sw BLOCK_DMODEL BLOCK_N : Nat) :
    ((token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
      B_Att_Start_Loc B_Att_Seqlen srtb srts sph spbs svbs svh svd sobs soh sod kvg sw
      BLOCK_DMODEL BLOCK_N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [mistralIO_body_split]
  simp [mistralIOPrelude, mistralLoopBody, mistralPostlude, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]
  simp [Op.FlattenOk.eq_def]

/-- Per-statement cast-free collapse lifts to statement lists (walks the
actual successor chain; a failing step collapses on both sides). Private
copy of the family helper (bench files never import each other). -/
private theorem mistralIO_stepStmtsR_castFree_of_stmts (R : RoundingModel) :
    ∀ (l : List Stmt), (∀ st ∈ l, ∀ u, stepStmtR R st u = stepStmt st u) →
      ∀ s, stepStmtsR R l s = stepStmts l s
  | [], _, s => by simp only [stepStmtsR, stepStmts]
  | st :: rest, h, s => by
      simp only [stepStmtsR, stepStmts, h st List.mem_cons_self s]
      cases stepStmt st s with
      | none => rfl
      | some s' =>
          exact mistralIO_stepStmtsR_castFree_of_stmts R rest
            (fun st' h' u => h st' (List.mem_cons_of_mem _ h') u) s'

set_option maxHeartbeats 4000000 in
/-- Every prelude statement is cast-free (four `.nat` slot loads,
register-only `.nat` address arithmetic, the `0.0` accumulator seed). -/
private theorem mistralIO_prelude_stmt_castFree (R : RoundingModel)
    (B_req_idx B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (srtb srts sph spbs svh svd kvg sw BLOCK_DMODEL BLOCK_N : Nat) :
    ∀ st ∈ mistralIOPrelude B_req_idx B_Seqlen B_Att_Start_Loc B_Att_Seqlen srtb srts sph
      spbs svh svd kvg sw BLOCK_DMODEL BLOCK_N,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [mistralIOPrelude, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl
  all_goals simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]

set_option maxHeartbeats 4000000 in
/-- Every loop-body statement is cast-free: the `start_n` self-assign, the
`.real` `Prob` load (`other = 0.0`), the `.nat` page-table gather
(`other = 0`), the gather-addressed `.real` `V` load, and the `.real`
fold arithmetic. -/
private theorem mistralIO_body_stmt_castFree (R : RoundingModel)
    (Prob V Req_to_tokens : RegionName) (srts svbs BLOCK_DMODEL BLOCK_N : Nat) :
    ∀ st ∈ mistralLoopBody Prob V Req_to_tokens srts svbs BLOCK_DMODEL BLOCK_N,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [mistralLoopBody, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl
  all_goals simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]

/-- The loop body is cast-free as a list. -/
private theorem mistralIO_body_castFree (R : RoundingModel)
    (Prob V Req_to_tokens : RegionName) (srts svbs BLOCK_DMODEL BLOCK_N : Nat)
    (t : BlockState) :
    stepStmtsR R (mistralLoopBody Prob V Req_to_tokens srts svbs BLOCK_DMODEL BLOCK_N) t
      = stepStmts (mistralLoopBody Prob V Req_to_tokens srts svbs BLOCK_DMODEL BLOCK_N) t :=
  mistralIO_stepStmtsR_castFree_of_stmts R _
    (mistralIO_body_stmt_castFree R Prob V Req_to_tokens srts svbs BLOCK_DMODEL BLOCK_N) t

set_option maxHeartbeats 4000000 in
/-- Every postlude statement is cast-free: the `.to(Out.dtype.element_ty)`
cast lowers to a **self-assign** (`Op.ref .real [BD] "acc"`, no
`Op.castFloat`), and `writeMemTypedR R .real` *is* `writeMemTyped .real`. -/
private theorem mistralIO_postlude_stmt_castFree (R : RoundingModel)
    (Out : RegionName) (sobs soh sod BLOCK_DMODEL : Nat) :
    ∀ st ∈ mistralPostlude Out sobs soh sod BLOCK_DMODEL,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [mistralPostlude, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl
  all_goals
    simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def,
      BlockState.writeMemTypedR]

/-- `evalOpR` of a `constNat` (R-independent). -/
private theorem mistralIO_evalOpR_constNat (R : RoundingModel) (n : Nat) (u : BlockState) :
    evalOpR R (Op.constNat n) u = some (Tile.scalar n) := by
  simp [evalOpR]

/-- `evalOpR` of the `forRangeDyn` stop expression (a bare `.nat` register
read — R-independent). -/
private theorem mistralIO_stopOpR_castFree (R : RoundingModel) (u : BlockState) :
    evalOpR R (Op.ref .nat [] "cur_att_seq_len") u
      = evalOp (Op.ref .nat [] "cur_att_seq_len") u := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

set_option maxHeartbeats 1600000 in
/-- The streaming `forRangeDyn` statement is cast-free per-state: its bound
expressions are a literal and a `.nat` register read, and its body is
cast-free, so `stepStmtR R` on the whole loop *is* `stepStmt`. -/
private theorem mistralIO_dyn_castFree (R : RoundingModel)
    (Prob V Req_to_tokens : RegionName) (srts svbs BLOCK_DMODEL BLOCK_N : Nat) :
    ∀ u, stepStmtR R (Stmt.forRangeDyn "start_n" (Op.constNat 0)
        (Op.ref .nat [] "cur_att_seq_len") (Op.constNat BLOCK_N)
        (mistralLoopBody Prob V Req_to_tokens srts svbs BLOCK_DMODEL BLOCK_N)) u
      = stepStmt (Stmt.forRangeDyn "start_n" (Op.constNat 0)
          (Op.ref .nat [] "cur_att_seq_len") (Op.constNat BLOCK_N)
          (mistralLoopBody Prob V Req_to_tokens srts svbs BLOCK_DMODEL BLOCK_N)) u := by
  intro u
  rw [stepForRangeAux.forRangeDyn_unfold]
  simp only [stepStmtR, mistralIO_evalOpR_constNat, mistralIO_stopOpR_castFree,
    evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  cases hstop : evalOp (Op.ref .nat [] "cur_att_seq_len") u with
  | none => rfl
  | some t =>
      simp only [Option.bind_some]
      exact stepForRangeAuxR_castFree R _
        (mistralIO_body_castFree R Prob V Req_to_tokens srts svbs BLOCK_DMODEL BLOCK_N)
        "start_n" _ _ _ u

/-- The whole lowered body is cast-free, statement by statement: `execR R`
on the surface *is* the exact `stepStmts` run. -/
private theorem mistralIO_execR_collapse (R : RoundingModel)
    (Prob V Out : RegionName) (Req_to_tokens B_req_idx : Region .nat)
    (B_Start_Loc : RegionName) (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (srtb srts sph spbs svbs svh svd sobs soh sod kvg sw BLOCK_DMODEL BLOCK_N : Nat)
    (s : BlockState) :
    execR R ((token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx B_Start_Loc
        B_Seqlen B_Att_Start_Loc B_Att_Seqlen srtb srts sph spbs svbs svh svd sobs soh sod
        kvg sw BLOCK_DMODEL BLOCK_N).toAlgKernel) s
      = stepStmts ((token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx B_Start_Loc
          B_Seqlen B_Att_Start_Loc B_Att_Seqlen srtb srts sph spbs svbs svh svd sobs soh sod
          kvg sw BLOCK_DMODEL BLOCK_N).toAlgKernel.body) s := by
  unfold execR
  rw [mistralIO_body_split]
  refine mistralIO_stepStmtsR_castFree_of_stmts R _ ?_ s
  intro st hst
  rcases List.mem_append.mp hst with hpre | hrest
  · exact mistralIO_prelude_stmt_castFree R B_req_idx B_Seqlen B_Att_Start_Loc
      B_Att_Seqlen srtb srts sph spbs svh svd kvg sw BLOCK_DMODEL BLOCK_N st hpre
  · rcases List.mem_cons.mp hrest with rfl | hpost
    · exact mistralIO_dyn_castFree R Prob V Req_to_tokens srts svbs BLOCK_DMODEL BLOCK_N
    · exact mistralIO_postlude_stmt_castFree R Out sobs soh sod BLOCK_DMODEL st hpost

/-! ## The weak safety stack (`hts`)

The skin's `hts` obligation quantifies over **arbitrary** launch states (no
clean-`undef` pin), so the exact stack's `mistralInvariant` — whose `undef`
and `acc = partialAcc k` conjuncts describe a clean run — is unavailable
there. Since every load in this kernel is `other=`-defaulted or an unmasked
`.nat` scalar read, the whole walk is `undef`-independent: the weak stack
re-runs the register chain with exact pins for the address-bearing registers
and an **existential** pin for the fold register `acc`. The `pre` hypothesis
is what makes the per-step `Fin T` window bounds citable at all. -/

/-- Combined walk cons: safety of the head at the current state, the R-step
it actually takes, and the pair (safety, run) of the tail from the successor
give the pair for the whole list. Private copy of the family combinator. -/
private theorem mistralIO_walkCons {R : RoundingModel} {bounds : RegionBounds}
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
private theorem mistralIO_walkNil {R : RoundingModel} {bounds : RegionBounds}
    {P : BlockState → Prop} (s : BlockState) (h : P s) :
    Stmt.TraceSafeListR R bounds [] s
      ∧ ∃ sF, stepStmtsR R [] s = some sF ∧ P sF :=
  ⟨Stmt.TraceSafeListR.nil_intro, s, by simp only [stepStmtsR], h⟩

/-- R-step of an assign whose op is cast-free: the two collapse into one
walk-ready equation. -/
private theorem mistralIO_stepR_of_assign {R : RoundingModel} {dt : TileDType}
    {sh : TileShape} {nm : RegName} {e : Op dt sh} {s : BlockState} {v : Tile dt sh}
    (hcf : evalOpR R e s = evalOp e s) (h : evalOp e s = some v) :
    stepStmtR R (.assign dt sh nm e) s = some (s.setReg nm dt sh v) :=
  stepStmtR_assign_eq_some (hcf.trans h)

/-- `readMem` depends on `mem` only (used to re-anchor the walk's
memory-derived quantities at the launch state across the `setReg` chain). -/
private theorem mistralIO_readMem_of_mem (u s : BlockState) (h : u.mem = s.mem)
    (r : RegionName) (a : Nat) : u.readMem r a = s.readMem r a := by
  simp only [BlockState.readMem, BlockState.readMemValue, BlockState.readMemTyped, h]

/-- `readMemValue` depends on `mem` only, at every channel dtype. -/
private theorem mistralIO_readMemValue_of_mem (u s : BlockState) (h : u.mem = s.mem)
    (dt : TileDType) (r : RegionName) (a : Nat) :
    u.readMemValue dt r a = s.readMemValue dt r a := by
  cases dt <;>
    simp only [BlockState.readMemValue, BlockState.readMemAs, BlockState.readMemTyped,
      BlockState.readMem, h]

/-- Every metadata-derived quantity of the closed form is a function of
`mem` and `pids` only, so it is stable along the whole register walk. -/
private theorem mistralIO_meta_of_mem (u s : BlockState) (hm : u.mem = s.mem)
    (hp : u.pids = s.pids)
    (B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen : RegionName) (sph spbs sw : Nat) :
    attSeqLen u B_Att_Seqlen = attSeqLen s B_Att_Seqlen
    ∧ batchSeqLen u B_Seqlen = batchSeqLen s B_Seqlen
    ∧ reqIdx u B_req_idx = reqIdx s B_req_idx
    ∧ startIndex u B_Seqlen sw = startIndex s B_Seqlen sw
    ∧ (∀ n, pOffset u B_Att_Start_Loc sph spbs n
        = pOffset s B_Att_Start_Loc sph spbs n) := by
  have hrv := mistralIO_readMemValue_of_mem u s hm
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simp only [attSeqLen, hrv, hp]
  · simp only [batchSeqLen, hrv, hp]
  · simp only [reqIdx, hrv, hp]
  · simp only [startIndex, batchSeqLen, hrv, hp]
  · intro n; simp only [pOffset, attStartLoc, hrv, hp]

/-! ### `evalOpR` decoders for the address and mask trees

Each address/mask op of the loop body and the postlude, decoded at pinned
registers. These are what the trace-safety obligations rewrite with before
citing the skin's per-window bounds. -/

set_option maxHeartbeats 1600000 in
/-- The `Prob` load's address tree `p_offs + start_n`. -/
private theorem mistralIO_pAddrR_eval (R : RoundingModel) {BN : Nat} (u : BlockState)
    (poffFn : Fin BN → Nat) (c : Nat)
    (hpo : u.regs .nat [BN] "p_offs" = some (Tile.vec poffFn))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar c)) :
    evalOpR R (Op.add .nat Broadcast.scalarR (Op.ref .nat [BN] "p_offs")
        (Op.ref .nat [] "start_n")) u
      = some (⟨fun idx : TileIndex [BN] => poffFn idx.1 + c⟩ : Tile .nat [BN]) := by
  rw [show evalOpR R (Op.add .nat Broadcast.scalarR (Op.ref .nat [BN] "p_offs")
        (Op.ref .nat [] "start_n")) u
      = evalOp (Op.add .nat Broadcast.scalarR (Op.ref .nat [BN] "p_offs")
          (Op.ref .nat [] "start_n")) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  simp only [evalOp_add, evalOp_ref, hpo, hsn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add]

set_option maxHeartbeats 1600000 in
/-- The `Prob` load's live-token mask `(start_n + offs_n) < cur_att_seq_len`. -/
private theorem mistralIO_pMaskR_eval (R : RoundingModel) {BN : Nat} (u : BlockState)
    (c S : Nat)
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar c))
    (hn : u.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hS : u.regs .nat [] "cur_att_seq_len" = some (Tile.scalar S)) :
    evalOpR R (Op.lt .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.ref .nat [BN] "offs_n"))
        (Op.ref .nat [] "cur_att_seq_len")) u
      = some (⟨fun idx : TileIndex [BN] => decide (c + idx.1.val < S)⟩
          : Tile .bool [BN]) := by
  rw [show evalOpR R (Op.lt .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.ref .nat [BN] "offs_n"))
        (Op.ref .nat [] "cur_att_seq_len")) u
      = evalOp (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.ref .nat [BN] "offs_n"))
          (Op.ref .nat [] "cur_att_seq_len")) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  simp only [evalOp, hsn, hn, hS, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec,
    ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
  rfl

set_option maxHeartbeats 1600000 in
/-- The page-table gather's address tree
`v_loc_off + start_n · stride_req_to_tokens_s`. -/
private theorem mistralIO_gAddrR_eval (R : RoundingModel) {BN : Nat} (u : BlockState)
    (vloFn : Fin BN → Nat) (c srts : Nat)
    (hvo : u.regs .nat [BN] "v_loc_off" = some (Tile.vec vloFn))
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar c)) :
    evalOpR R (Op.add .nat Broadcast.scalarR (Op.ref .nat [BN] "v_loc_off")
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat srts))) u
      = some (⟨fun idx : TileIndex [BN] => vloFn idx.1 + c * srts⟩ : Tile .nat [BN]) := by
  rw [show evalOpR R (Op.add .nat Broadcast.scalarR (Op.ref .nat [BN] "v_loc_off")
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat srts))) u
      = evalOp (Op.add .nat Broadcast.scalarR (Op.ref .nat [BN] "v_loc_off")
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat srts))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, hvo, hsn,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

set_option maxHeartbeats 1600000 in
/-- The gather's window mask
`(start_n + offs_n) + cur_batch_start_index < cur_batch_seq_len`. -/
private theorem mistralIO_gMaskR_eval (R : RoundingModel) {BN : Nat} (u : BlockState)
    (c startIdx batchS : Nat)
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar c))
    (hn : u.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hsi : u.regs .nat [] "cur_batch_start_index" = some (Tile.scalar startIdx))
    (hbs : u.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar batchS)) :
    evalOpR R (Op.lt .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.ref .nat [BN] "offs_n"))
          (Op.ref .nat [] "cur_batch_start_index"))
        (Op.ref .nat [] "cur_batch_seq_len")) u
      = some (⟨fun idx : TileIndex [BN] => decide (c + idx.1.val + startIdx < batchS)⟩
          : Tile .bool [BN]) := by
  rw [show evalOpR R (Op.lt .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.ref .nat [BN] "offs_n"))
          (Op.ref .nat [] "cur_batch_start_index"))
        (Op.ref .nat [] "cur_batch_seq_len")) u
      = evalOp (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.ref .nat [BN] "offs_n"))
            (Op.ref .nat [] "cur_batch_start_index"))
          (Op.ref .nat [] "cur_batch_seq_len")) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  simp only [evalOp, hsn, hn, hsi, hbs, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec,
    ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
  rfl

set_option maxHeartbeats 1600000 in
/-- The gather-addressed `V` load's address tree
`v_offs + v_loc[:, None] · stride_vbs` (shape `[BLOCK_N, BLOCK_DMODEL]`). -/
private theorem mistralIO_vAddrR_eval (R : RoundingModel) (BN BD : Nat) (u : BlockState)
    (voffFn : Fin BD → Nat) (vlocFn : Fin BN → Nat) (svbs : Nat)
    (hvoffs : u.regs .nat [1, BD] "v_offs" =
      some (⟨fun idx : TileIndex [1, BD] => voffFn idx.2.1⟩ : Tile .nat [1, BD]))
    (hvloc : u.regs .nat [BN] "v_loc" = some (Tile.vec vlocFn)) :
    evalOpR R (Op.add .nat (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.ref .nat [1, BD] "v_offs")
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "v_loc"))
          (Op.constNat svbs))) u
      = some (⟨fun idx : TileIndex [BN, BD] => voffFn idx.2.1 + vlocFn idx.1 * svbs⟩
          : Tile .nat [BN, BD]) := by
  rw [show evalOpR R (Op.add .nat (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.ref .nat [1, BD] "v_offs")
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "v_loc"))
          (Op.constNat svbs))) u
      = evalOp (Op.add .nat (Broadcast.consL (Broadcast.consR Broadcast.nil))
          (Op.ref .nat [1, BD] "v_offs")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "v_loc"))
            (Op.constNat svbs))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  have hexp : @evalOp .nat [BN, 1]
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "v_loc")) u
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec vlocFn)) :=
    evalOp_expandDim_ref_of_regs .nat [BN] ⟨1, by simp⟩ "v_loc" u _ hvloc
  simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, hvoffs, hexp,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, d, uu⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar, Tile.expandDim_data,
    TileShape.dropInsertedIndex_succ, TileShape.dropInsertedIndex_nil,
    TileShape.dropInsertedIndex_zero_cons, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul]

set_option maxHeartbeats 1600000 in
/-- The gather-addressed `V` load's mask: the `[BN, 1]` window test remapped
across the head-dim columns. It is the **same predicate** as the gather's
mask, which is why no dead lane ever dereferences a sentinel address. -/
private theorem mistralIO_vMaskR_eval (R : RoundingModel) (BN BD : Nat) (u : BlockState)
    (c startIdx batchS : Nat)
    (hsn : u.regs .nat [] "start_n" = some (Tile.scalar c))
    (hn : u.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hsi : u.regs .nat [] "cur_batch_start_index" = some (Tile.scalar startIdx))
    (hbs : u.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar batchS)) :
    evalOpR R (Op.remap [BN, BD]
        (Broadcast.leftIndex (Broadcast.consSame (Broadcast.consL Broadcast.nil)))
        (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")))
            (Op.ref .nat [] "cur_batch_start_index"))
          (Op.ref .nat [] "cur_batch_seq_len"))) u
      = some (⟨fun idx : TileIndex [BN, BD] => decide (c + idx.1.val + startIdx < batchS)⟩
          : Tile .bool [BN, BD]) := by
  rw [show evalOpR R (Op.remap [BN, BD]
        (Broadcast.leftIndex (Broadcast.consSame (Broadcast.consL Broadcast.nil)))
        (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")))
            (Op.ref .nat [] "cur_batch_start_index"))
          (Op.ref .nat [] "cur_batch_seq_len"))) u
      = evalOp (Op.remap [BN, BD]
          (Broadcast.leftIndex (Broadcast.consSame (Broadcast.consL Broadcast.nil)))
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")))
              (Op.ref .nat [] "cur_batch_start_index"))
            (Op.ref .nat [] "cur_batch_seq_len"))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  have hexpn : @evalOp .nat [BN, 1]
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")) u
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun j : Fin BN => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BN] ⟨1, by simp⟩ "offs_n" u _ hn
  simp only [evalOp, hsn, hexpn, hsi, hbs, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, d, uu⟩ := idx
  simp only [Tile.remap, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar,
    Tile.expandDim_data, TileShape.dropInsertedIndex_succ, TileShape.dropInsertedIndex_nil,
    TileShape.dropInsertedIndex_zero_cons, NumericDType.add, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex]
  rfl

set_option maxHeartbeats 1600000 in
/-- The postlude's output offsets
`cur_batch·stride_obs + cur_head·stride_oh + offs_d·stride_od`. -/
private theorem mistralIO_offoR_eval (R : RoundingModel) {D : Nat} (u : BlockState)
    (cb ch sobs soh sod : Nat)
    (hcb : u.regs .nat [] "cur_batch" = some (Tile.scalar cb))
    (hch : u.regs .nat [] "cur_head" = some (Tile.scalar ch))
    (hd : u.regs .nat [D] "offs_d" = some (Tile.vec (fun e : Fin D => e.val))) :
    evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat sobs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat soh)))
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [D] "offs_d")
          (Op.constNat sod))) u
      = some (⟨fun idx : TileIndex [D] => cb * sobs + ch * soh + idx.1.val * sod⟩
          : Tile .nat [D]) := by
  rw [show evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat sobs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat soh)))
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [D] "offs_d")
          (Op.constNat sod))) u
      = evalOp (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat sobs))
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

/-- The postlude's `out_ptrs` tree at a pinned `off_o`. -/
private theorem mistralIO_outPtrR_eval (R : RoundingModel) {D : Nat} (u : BlockState)
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

/-- Weak (safety-walk) invariant: exact pins for the address-bearing
registers (all anchored to the launch state `s` through `readMemValue`, so
they survive the whole run), an **existential** pin for the fold register
`acc` (the walk never needs its value, only its shape). This is
`mistralInvariant` with the `undef` conjunct dropped — `hts` runs from an
arbitrary launch state — and the accumulator freed. -/
private def mistralIOSafeInvW
    (B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen : RegionName)
    (srtb srts sph spbs svh svd kvg sw BLOCK_DMODEL BLOCK_N : Nat)
    (s s' : BlockState) : Prop :=
  s'.mem = s.mem
  ∧ s'.pids = s.pids
  ∧ s'.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 0))
  ∧ s'.regs .nat [] "cur_head" = some (Tile.scalar (s.pids 1))
  ∧ s'.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
  ∧ s'.regs .nat [BLOCK_DMODEL] "offs_d"
      = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))
  ∧ s'.regs .nat [] "cur_att_seq_len" = some (Tile.scalar (attSeqLen s B_Att_Seqlen))
  ∧ s'.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (batchSeqLen s B_Seqlen))
  ∧ s'.regs .nat [] "cur_batch_start_index"
      = some (Tile.scalar (startIndex s B_Seqlen sw))
  ∧ s'.regs .nat [BLOCK_N] "v_loc_off"
      = some (Tile.vec (fun j : Fin BLOCK_N =>
          reqIdx s B_req_idx * srtb + (startIndex s B_Seqlen sw + j.val) * srts))
  ∧ s'.regs .nat [BLOCK_N] "p_offs"
      = some (Tile.vec (fun j : Fin BLOCK_N =>
          pOffset s B_Att_Start_Loc sph spbs j.val))
  ∧ s'.regs .nat [1, BLOCK_DMODEL] "v_offs"
      = some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] =>
          (s.pids 1 / kvg) * svh + idx.2.1.val * svd⟩ : Tile .nat [1, BLOCK_DMODEL])
  ∧ (∃ ac : Fin BLOCK_DMODEL → ℝ, s'.regs .real [BLOCK_DMODEL] "acc"
      = some (⟨fun idx : TileIndex [BLOCK_DMODEL] => some (ac idx.1)⟩
        : Tile .real [BLOCK_DMODEL]))

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **Weak prelude walk** (single pass): from an **arbitrary** launch state
the 14 prelude statements are trace-safe (the four `.nat` slot loads bounded
by their slot windows at cell `pid₀`; everything else is register-only) and
step to a state satisfying `mistralIOSafeInvW`. -/
private theorem mistralIO_preLoopW (R : RoundingModel) (bounds : RegionBounds)
    (B_req_idx : Region .nat) (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (srtb srts sph spbs svh svd kvg sw BLOCK_DMODEL BLOCK_N : Nat) (s : BlockState)
    (hbSQ : s.pids 0 < bounds (Region.cast B_Seqlen))
    (hbAS : s.pids 0 < bounds (Region.cast B_Att_Start_Loc))
    (hbRQ : s.pids 0 < bounds (Region.cast B_req_idx))
    (hbAQ : s.pids 0 < bounds (Region.cast B_Att_Seqlen)) :
    Stmt.TraceSafeListR R bounds
        (mistralIOPrelude B_req_idx B_Seqlen B_Att_Start_Loc B_Att_Seqlen srtb srts sph
          spbs svh svd kvg sw BLOCK_DMODEL BLOCK_N) s
      ∧ ∃ s0, stepStmtsR R
            (mistralIOPrelude B_req_idx B_Seqlen B_Att_Start_Loc B_Att_Seqlen srtb srts
              sph spbs svh svd kvg sw BLOCK_DMODEL BLOCK_N) s = some s0
          ∧ mistralIOSafeInvW B_req_idx.cast B_Att_Start_Loc.cast B_Seqlen.cast
              B_Att_Seqlen.cast srtb srts sph spbs svh svd kvg sw BLOCK_DMODEL BLOCK_N
              s s0 := by
  unfold mistralIOPrelude
  -- stmt 0: cur_batch = programId 0
  refine mistralIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (evalOp_programId 0 s)) ?_
  -- stmt 1: cur_head = programId 1
  refine mistralIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (evalOp_programId 1 _)) ?_
  -- stmt 2: cur_kv_head = cur_head // kv_group_num
  refine mistralIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head")
            (Op.constNat kvg)) _
          = some (Tile.scalar (s.pids 1 / kvg)) from by
        simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
          BlockState.setReg_ne_name, BlockState.setReg_pids, Option.bind]
        refine congrArg some ?_
        ext idx
        simp only [Tile.bop, Tile.scalar, BlockState.setReg_pids,
          IntegralDType.nat_floorDiv])) ?_
  -- stmt 3: offs_n = arange BLOCK_N
  refine mistralIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.arange BLOCK_N) _
          = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) from evalOp_arange _ _)) ?_
  -- stmt 4: offs_d = arange BLOCK_DMODEL
  refine mistralIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.arange BLOCK_DMODEL) _
          = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val)) from
        evalOp_arange _ _)) ?_
  -- stmt 5: cur_batch_seq_len = load(B_Seqlen + cur_batch)   [slot-0 window]
  refine mistralIO_walkCons ?_
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch"))
            MaskOpt.none) _
          = some (Tile.scalar (batchSeqLen s B_Seqlen.cast)) from by
        simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
          BlockState.setReg_ne_name, Option.bind, Option.pure_def]
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
  -- stmt 6: cur_batch_start_index = max(cur_batch_seq_len - sliding_window, 0)
  refine mistralIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.where (Op.gt .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len")
                (Op.constNat sw))
              (Op.constNat 0))
            (Op.sub .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len")
              (Op.constNat sw))
            (Op.constNat 0)) _
          = some (Tile.scalar (startIndex s B_Seqlen.cast sw)) from by
        simp only [evalOp_where, evalOp_gt, evalOp_sub, evalOp_constNat, evalOp_ref,
          BlockState.setReg_same, BlockState.setReg_ne_name, Option.bind]
        refine congrArg some ?_
        ext idx
        simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.scalar,
          Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt,
          NumericDType.sub, startIndex, batchSeqLen]
        by_cases h : 0 < s.readMemValue .nat B_Seqlen.cast (s.pids 0) - sw
        · rw [if_pos (by simpa using h)]
        · simp only [Nat.not_lt, Nat.le_zero] at h
          rw [if_neg (by simp [h]), h])) ?_
  -- stmt 7: cur_batch_in_all_start_index = load(B_Att_Start_Loc + cur_batch)  [slot 1]
  refine mistralIO_walkCons ?_
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.load .nat
            (MemAccess.region B_Att_Start_Loc (Op.ref .nat [] "cur_batch"))
            MaskOpt.none) _
          = some (Tile.scalar (attStartLoc s B_Att_Start_Loc.cast)) from by
        simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
          BlockState.setReg_ne_name, Option.bind, Option.pure_def]
        rfl)) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR], ?_⟩
    intro offsets hoff idx _
    rw [evalOpR_ref] at hoff
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same] at hoff
    obtain rfl := Option.some.inj hoff
    exact hbAS
  -- stmt 8: cur_batch_req_idx = load(B_req_idx + cur_batch)   [slot 2]
  refine mistralIO_walkCons ?_
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.load .nat (MemAccess.region B_req_idx (Op.ref .nat [] "cur_batch"))
            MaskOpt.none) _
          = some (Tile.scalar (reqIdx s B_req_idx.cast)) from by
        simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
          BlockState.setReg_ne_name, Option.bind, Option.pure_def]
        rfl)) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR], ?_⟩
    intro offsets hoff idx _
    rw [evalOpR_ref] at hoff
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same] at hoff
    obtain rfl := Option.some.inj hoff
    exact hbRQ
  -- stmt 9: cur_att_seq_len = load(B_Att_Seqlen + cur_batch)   [slot 3]
  refine mistralIO_walkCons ?_
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.load .nat
            (MemAccess.region B_Att_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
          = some (Tile.scalar (attSeqLen s B_Att_Seqlen.cast)) from by
        simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
          BlockState.setReg_ne_name, Option.bind, Option.pure_def]
        rfl)) ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR], ?_⟩
    intro offsets hoff idx _
    rw [evalOpR_ref] at hoff
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same] at hoff
    obtain rfl := Option.some.inj hoff
    exact hbAQ
  -- stmt 10: v_loc_off
  refine mistralIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (mistral_vloc_off_eval _ BLOCK_N (reqIdx s B_req_idx.cast)
        (startIndex s B_Seqlen.cast sw) srtb srts (by simp) (by simp)
        (by simp [Tile.vec]))) ?_
  -- stmt 11: p_offs
  refine mistralIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat sph))
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [BLOCK_N] "offs_n"))
              (Op.constNat spbs))) _
          = some (Tile.vec (fun j : Fin BLOCK_N =>
              pOffset s B_Att_Start_Loc.cast sph spbs j.val)) from by
        rw [mistral_poffs_eval _ BLOCK_N (s.pids 1) (attStartLoc s B_Att_Start_Loc.cast)
          sph spbs (by simp) (by simp) (by simp [Tile.vec])]
        refine congrArg some ?_
        ext idx
        simp [Tile.vec, pOffset, attStartLoc])) ?_
  -- stmt 12: v_offs
  refine mistralIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (mistral_voffs_eval _ BLOCK_DMODEL (s.pids 1 / kvg) svh svd (by simp)
        (by simp [Tile.vec]))) ?_
  -- stmt 13: acc = zeros
  refine mistralIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.full [BLOCK_DMODEL] (Op.const 0)) _
          = some (⟨fun _ : TileIndex [BLOCK_DMODEL] => some (0 : ℝ)⟩
              : Tile .real [BLOCK_DMODEL]) from by
        simp only [evalOp_full, evalOp, Option.bind]
        refine congrArg some ?_
        ext idx
        rfl)) ?_
  refine mistralIO_walkNil _ ?_
  refine ⟨?_, by simp, by simp, by simp, by simp [Tile.vec], by simp [Tile.vec],
    by simp, by simp, by simp, by simp [Tile.vec], by simp, by simp,
    ⟨fun _ => 0, by simp⟩⟩
  funext rg o
  simp

/-- The masked `Prob` value is a function of `mem`/`pids` only. -/
private theorem mistralIO_pMasked_of_mem (sx s : BlockState) (hmx : sx.mem = s.mem)
    (hp : sx.pids = s.pids) (Prob B_Att_Start_Loc B_Att_Seqlen : RegionName)
    (sph spbs n : Nat) :
    pMasked sx Prob B_Att_Start_Loc B_Att_Seqlen sph spbs n
      = pMasked s Prob B_Att_Start_Loc B_Att_Seqlen sph spbs n := by
  have hrm := mistralIO_readMem_of_mem sx s hmx
  have hrv := mistralIO_readMemValue_of_mem sx s hmx
  simp only [pMasked, attSeqLen, pOffset, attStartLoc, hrv, hp]
  by_cases h : n < s.readMemValue .nat B_Att_Seqlen (s.pids 0)
  · rw [if_pos h, if_pos h]; exact hrm Prob _
  · rw [if_neg h, if_neg h]

/-- The masked gathered `V` row is a function of `mem`/`pids` only. -/
private theorem mistralIO_vMasked_of_mem (sx s : BlockState) (hmx : sx.mem = s.mem)
    (hp : sx.pids = s.pids) (V Req_to_tokens B_req_idx B_Seqlen : RegionName)
    (srtb srts svbs svh svd kvg sw n d : Nat) :
    vMasked sx V Req_to_tokens B_req_idx B_Seqlen srtb srts svbs svh svd kvg sw n d
      = vMasked s V Req_to_tokens B_req_idx B_Seqlen srtb srts svbs svh svd kvg sw n d := by
  have hrm := mistralIO_readMem_of_mem sx s hmx
  have hrv := mistralIO_readMemValue_of_mem sx s hmx
  simp only [vMasked, vActive, startIndex, batchSeqLen, vOffset, vLoc, reqIdx, hrv, hp]
  by_cases h : s.readMemValue .nat B_Seqlen (s.pids 0) - sw + n
      < s.readMemValue .nat B_Seqlen (s.pids 0)
  · rw [if_pos h, if_pos h]; exact hrm V _
  · rw [if_neg h, if_neg h]

/-- The gathered page index (with the masked-off `0`) is a function of
`mem`/`pids` only. -/
private theorem mistralIO_vLocIf_of_mem (sx s : BlockState) (hmx : sx.mem = s.mem)
    (hp : sx.pids = s.pids) (Req_to_tokens B_req_idx B_Seqlen : RegionName)
    (srtb srts sw n : Nat) :
    (if vActive sx B_Seqlen sw n then
        vLoc sx Req_to_tokens B_req_idx B_Seqlen srtb srts sw n else 0)
      = (if vActive s B_Seqlen sw n then
          vLoc s Req_to_tokens B_req_idx B_Seqlen srtb srts sw n else 0) := by
  have hrv := mistralIO_readMemValue_of_mem sx s hmx
  simp only [vActive, startIndex, batchSeqLen, vLoc, reqIdx, hrv, hp]

/-! ### `setReg`-peeling of the memory-derived quantities

Every quantity the recipes speak of is a function of `mem`/`pids` only, so a
register write is transparent to it. As `simp only` lemmas these peel a walk
state all the way back to the loop-entry state, where the invariant's pins
apply. -/

private theorem mistralIO_attSeqLen_setReg (u : BlockState) {dt : TileDType}
    {sh : TileShape} (nm : RegName) (t : Tile dt sh) (B_Att_Seqlen : RegionName) :
    attSeqLen (u.setReg nm dt sh t) B_Att_Seqlen = attSeqLen u B_Att_Seqlen := by
  unfold attSeqLen
  rw [BlockState.setReg_pids]
  exact mistralIO_readMemValue_of_mem (u.setReg nm dt sh t) u
    rfl _ _ _

private theorem mistralIO_batchSeqLen_setReg (u : BlockState) {dt : TileDType}
    {sh : TileShape} (nm : RegName) (t : Tile dt sh) (B_Seqlen : RegionName) :
    batchSeqLen (u.setReg nm dt sh t) B_Seqlen = batchSeqLen u B_Seqlen := by
  unfold batchSeqLen
  rw [BlockState.setReg_pids]
  exact mistralIO_readMemValue_of_mem (u.setReg nm dt sh t) u
    rfl _ _ _

private theorem mistralIO_reqIdx_setReg (u : BlockState) {dt : TileDType}
    {sh : TileShape} (nm : RegName) (t : Tile dt sh) (B_req_idx : RegionName) :
    reqIdx (u.setReg nm dt sh t) B_req_idx = reqIdx u B_req_idx := by
  unfold reqIdx
  rw [BlockState.setReg_pids]
  exact mistralIO_readMemValue_of_mem (u.setReg nm dt sh t) u
    rfl _ _ _

private theorem mistralIO_startIndex_setReg (u : BlockState) {dt : TileDType}
    {sh : TileShape} (nm : RegName) (t : Tile dt sh) (B_Seqlen : RegionName) (sw : Nat) :
    startIndex (u.setReg nm dt sh t) B_Seqlen sw = startIndex u B_Seqlen sw := by
  unfold startIndex
  rw [mistralIO_batchSeqLen_setReg]

private theorem mistralIO_pOffset_setReg (u : BlockState) {dt : TileDType}
    {sh : TileShape} (nm : RegName) (t : Tile dt sh) (B_Att_Start_Loc : RegionName)
    (sph spbs n : Nat) :
    pOffset (u.setReg nm dt sh t) B_Att_Start_Loc sph spbs n
      = pOffset u B_Att_Start_Loc sph spbs n := by
  unfold pOffset attStartLoc
  rw [BlockState.setReg_pids,
    mistralIO_readMemValue_of_mem (u.setReg nm dt sh t) u
      rfl .nat B_Att_Start_Loc (u.pids 0)]

private theorem mistralIO_pMasked_setReg (u : BlockState) {dt : TileDType}
    {sh : TileShape} (nm : RegName) (t : Tile dt sh)
    (Prob B_Att_Start_Loc B_Att_Seqlen : RegionName) (sph spbs n : Nat) :
    pMasked (u.setReg nm dt sh t) Prob B_Att_Start_Loc B_Att_Seqlen sph spbs n
      = pMasked u Prob B_Att_Start_Loc B_Att_Seqlen sph spbs n :=
  mistralIO_pMasked_of_mem (u.setReg nm dt sh t) u rfl
    rfl _ _ _ _ _ _

private theorem mistralIO_vMasked_setReg (u : BlockState) {dt : TileDType}
    {sh : TileShape} (nm : RegName) (t : Tile dt sh)
    (V Req_to_tokens B_req_idx B_Seqlen : RegionName)
    (srtb srts svbs svh svd kvg sw n d : Nat) :
    vMasked (u.setReg nm dt sh t) V Req_to_tokens B_req_idx B_Seqlen srtb srts svbs svh
        svd kvg sw n d
      = vMasked u V Req_to_tokens B_req_idx B_Seqlen srtb srts svbs svh svd kvg sw n d :=
  mistralIO_vMasked_of_mem (u.setReg nm dt sh t) u rfl
    rfl _ _ _ _ _ _ _ _ _ _ _ _ _

private theorem mistralIO_vLocIf_setReg (u : BlockState) {dt : TileDType}
    {sh : TileShape} (nm : RegName) (t : Tile dt sh)
    (Req_to_tokens B_req_idx B_Seqlen : RegionName) (srtb srts sw n : Nat) :
    (if vActive (u.setReg nm dt sh t) B_Seqlen sw n then
        vLoc (u.setReg nm dt sh t) Req_to_tokens B_req_idx B_Seqlen srtb srts sw n else 0)
      = (if vActive u B_Seqlen sw n then
          vLoc u Req_to_tokens B_req_idx B_Seqlen srtb srts sw n else 0) :=
  mistralIO_vLocIf_of_mem (u.setReg nm dt sh t) u rfl
    rfl _ _ _ _ _ _ _

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **Weak loop-body walk**: one iteration at counter `c` is trace-safe (the
`Prob` load bounded on its live lanes, the page-table gather and the
gather-addressed `V` load bounded on the lanes of their **shared** window
predicate — the mask that makes this port sentinel-free) and steps to a state
that re-establishes `mistralIOSafeInvW`. -/
private theorem mistralIO_bodyW (R : RoundingModel) (bounds : RegionBounds)
    (Prob V Req_to_tokens : RegionName)
    (B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen : RegionName)
    (srtb srts sph spbs svbs svh svd kvg sw BLOCK_DMODEL BLOCK_N : Nat)
    (hpbs : spbs = 1) (hrts : srts = 1) (s stt : BlockState) (c : Nat)
    (hP : mistralIOSafeInvW B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen srtb srts sph
      spbs svh svd kvg sw BLOCK_DMODEL BLOCK_N s stt)
    (hbP : ∀ jL : Fin BLOCK_N, c + jL.val < attSeqLen s B_Att_Seqlen →
      pOffset s B_Att_Start_Loc sph spbs (c + jL.val) < bounds Prob)
    (hbG : ∀ jL : Fin BLOCK_N,
      startIndex s B_Seqlen sw + (c + jL.val) < batchSeqLen s B_Seqlen →
      reqIdx s B_req_idx * srtb + (startIndex s B_Seqlen sw + (c + jL.val)) * srts
        < bounds Req_to_tokens)
    (hbV : ∀ (jL : Fin BLOCK_N) (d : Fin BLOCK_DMODEL),
      startIndex s B_Seqlen sw + (c + jL.val) < batchSeqLen s B_Seqlen →
      vLoc s Req_to_tokens B_req_idx B_Seqlen srtb srts sw (c + jL.val) * svbs
          + (s.pids 1 / kvg) * svh + d.val * svd < bounds V) :
    Stmt.TraceSafeListR R bounds
        (mistralLoopBody Prob V Req_to_tokens srts svbs BLOCK_DMODEL BLOCK_N)
        (stt.setReg "start_n" .nat [] (Tile.scalar c))
      ∧ ∃ s', stepStmtsR R
            (mistralLoopBody Prob V Req_to_tokens srts svbs BLOCK_DMODEL BLOCK_N)
            (stt.setReg "start_n" .nat [] (Tile.scalar c)) = some s'
          ∧ mistralIOSafeInvW B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen srtb srts
              sph spbs svh svd kvg sw BLOCK_DMODEL BLOCK_N s s' := by
  obtain ⟨hmem, hpids, hcb, hch, hn, hd, hsl, hbsl, hsi, hvoff, hpoff, hvoffs,
    ⟨ac0, hacc⟩⟩ := hP
  obtain ⟨hAS, hBS, hRI, hSI, hPO⟩ := mistralIO_meta_of_mem stt s hmem hpids B_req_idx
    B_Att_Start_Loc B_Seqlen B_Att_Seqlen sph spbs sw
  have hVL : ∀ n, (if vActive stt B_Seqlen sw n then
        vLoc stt Req_to_tokens B_req_idx B_Seqlen srtb srts sw n else 0)
      = (if vActive s B_Seqlen sw n then
          vLoc s Req_to_tokens B_req_idx B_Seqlen srtb srts sw n else 0) :=
    fun n => mistralIO_vLocIf_of_mem stt s hmem hpids Req_to_tokens B_req_idx B_Seqlen
      srtb srts sw n
  set S := attSeqLen s B_Att_Seqlen with hSdef
  set batchS := batchSeqLen s B_Seqlen with hbatchdef
  set startIdx := startIndex s B_Seqlen sw with hstartdef
  set poffFn : Fin BLOCK_N → Nat :=
    fun j => pOffset s B_Att_Start_Loc sph spbs j.val with hpoffFn
  set vloFn : Fin BLOCK_N → Nat :=
    fun j => reqIdx s B_req_idx * srtb + (startIdx + j.val) * srts with hvloFn
  set voffFn : Fin BLOCK_DMODEL → Nat :=
    fun d => (s.pids 1 / kvg) * svh + d.val * svd with hvoffFn
  set vlocFn : Fin BLOCK_N → Nat :=
    fun j => if vActive stt B_Seqlen sw (c + j.val) then
        vLoc stt Req_to_tokens B_req_idx B_Seqlen srtb srts sw (c + j.val) else 0
    with hvlocFn
  -- the address-tree arithmetic the two contiguous strides supply
  have hpaddr : ∀ j : Fin BLOCK_N, poffFn j + c
      = pOffset s B_Att_Start_Loc sph spbs (c + j.val) := by
    intro j; simp only [hpoffFn, pOffset, hpbs]; ring
  have hgaddr : ∀ j : Fin BLOCK_N, vloFn j + c * srts
      = reqIdx s B_req_idx * srtb + (startIdx + (c + j.val)) * srts := by
    intro j; simp only [hvloFn]; subst hrts; ring
  unfold mistralLoopBody
  -- stmt 0: start_n = tl.multiple_of(start_n, BLOCK_N)  (erased: self-assign)
  refine mistralIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.ref .nat [] "start_n") _ = some (Tile.scalar c) from by
        rw [evalOp_ref, BlockState.setReg_same])) ?_
  -- stmt 1: p_value = masked Prob load   [read1 window]
  refine mistralIO_walkCons ?_
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (mistral_prob_load_eval _ Prob B_Att_Start_Loc B_Att_Seqlen BLOCK_N sph spbs c hpbs
        (by
          simp only [mistralIO_pOffset_setReg, hPO, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true, reduceCtorEq]
          exact hpoff)
        (by simp only [BlockState.setReg_same])
        (by
          simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true, reduceCtorEq]
          exact hn)
        (by
          simp only [mistralIO_attSeqLen_setReg, hAS, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true, reduceCtorEq]
          exact hsl))) ?_
  · -- safety: the `Prob` load's live lanes are in bounds
    simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def],
      ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def]⟩, ?_⟩
    intro offsets hoffs idx hactive
    rw [mistralIO_pAddrR_eval R _ poffFn c
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, reduceCtorEq]
        exact hpoff)
      (by simp only [BlockState.setReg_same])] at hoffs
    obtain rfl := Option.some.inj hoffs
    obtain ⟨masks, hmask, hactl⟩ := hactive
    rw [mistralIO_pMaskR_eval R _ c S
      (by simp only [BlockState.setReg_same])
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, reduceCtorEq]
        exact hn)
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, reduceCtorEq]
        exact hsl)] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨jL, uu⟩ := idx
    have hlive : c + jL.val < S := by simpa using hactl
    show poffFn jL + c < bounds Prob
    rw [hpaddr jL]
    exact hbP jL hlive
  -- stmt 2: v_loc = masked page-table gather   [gread window]
  refine mistralIO_walkCons ?_
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (mistral_reqloc_gather_eval _ Req_to_tokens B_req_idx B_Seqlen BLOCK_N srtb srts sw
        c hrts
        (by
          simp only [mistralIO_reqIdx_setReg, mistralIO_startIndex_setReg, hRI, hSI,
            BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hvoff)
        (by
          simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true, reduceCtorEq, BlockState.setReg_same])
        (by
          simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true, reduceCtorEq]
          exact hn)
        (by
          simp only [mistralIO_startIndex_setReg, hSI, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true, reduceCtorEq]
          exact hsi)
        (by
          simp only [mistralIO_batchSeqLen_setReg, hBS, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true, reduceCtorEq]
          exact hbsl))) ?_
  · -- safety: the gather's live lanes are in bounds
    simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def],
      ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def]⟩, ?_⟩
    intro offsets hoffs idx hactive
    rw [mistralIO_gAddrR_eval R _ vloFn c srts
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, reduceCtorEq]
        exact hvoff)
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, reduceCtorEq, BlockState.setReg_same])] at hoffs
    obtain rfl := Option.some.inj hoffs
    obtain ⟨masks, hmask, hactl⟩ := hactive
    rw [mistralIO_gMaskR_eval R _ c startIdx batchS
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, reduceCtorEq, BlockState.setReg_same])
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, reduceCtorEq]
        exact hn)
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, reduceCtorEq]
        exact hsi)
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, reduceCtorEq]
        exact hbsl)] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨jL, uu⟩ := idx
    have hlive : c + jL.val + startIdx < batchS := by simpa using hactl
    show vloFn jL + c * srts < bounds Req_to_tokens
    rw [hgaddr jL]
    exact hbG jL (by omega)
  -- stmt 3: v_value = gather-addressed masked V load   [read2 window]
  refine mistralIO_walkCons ?_
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (mistral_v_gather_eval _ V Req_to_tokens B_req_idx B_Seqlen BLOCK_N BLOCK_DMODEL
        srtb srts svbs svh svd kvg sw c
        (by
          simp only [BlockState.setReg_pids, hpids, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true, reduceCtorEq]
          exact hvoffs)
        (by
          simp only [mistralIO_vLocIf_setReg, BlockState.setReg_same]
          rfl)
        (by
          simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true, reduceCtorEq, BlockState.setReg_same])
        (by
          simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true, reduceCtorEq]
          exact hn)
        (by
          simp only [mistralIO_startIndex_setReg, hSI, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true, reduceCtorEq]
          exact hsi)
        (by
          simp only [mistralIO_batchSeqLen_setReg, hBS, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true, reduceCtorEq]
          exact hbsl))) ?_
  · -- safety: the gather-addressed load, on the gather's own window
    simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def],
      ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def]⟩, ?_⟩
    intro offsets hoffs idx hactive
    rw [mistralIO_vAddrR_eval R BLOCK_N BLOCK_DMODEL _ voffFn vlocFn svbs
      (by
        simp only [BlockState.setReg_pids, hpids, BlockState.setReg_ne_name, ne_eq,
          String.reduceEq, not_false_eq_true, reduceCtorEq]
        exact hvoffs)
      (by
        simp only [hvlocFn, mistralIO_vLocIf_setReg, BlockState.setReg_same]
        rfl)] at hoffs
    obtain rfl := Option.some.inj hoffs
    obtain ⟨masks, hmask, hactl⟩ := hactive
    rw [mistralIO_vMaskR_eval R BLOCK_N BLOCK_DMODEL _ c startIdx batchS
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, reduceCtorEq, BlockState.setReg_same])
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, reduceCtorEq]
        exact hn)
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, reduceCtorEq]
        exact hsi)
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, reduceCtorEq]
        exact hbsl)] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨jL, e, uu⟩ := idx
    have hlive : c + jL.val + startIdx < batchS := by simpa using hactl
    have hact : vActive s B_Seqlen sw (c + jL.val) := by
      simp only [vActive, ← hstartdef, ← hbatchdef]; omega
    show voffFn e + vlocFn jL * svbs < bounds V
    simp only [hvlocFn, hvoffFn, hVL, if_pos hact]
    rw [show (s.pids 1 / kvg) * svh + e.val * svd
        + vLoc s Req_to_tokens B_req_idx B_Seqlen srtb srts sw (c + jL.val) * svbs
      = vLoc s Req_to_tokens B_req_idx B_Seqlen srtb srts sw (c + jL.val) * svbs
        + (s.pids 1 / kvg) * svh + e.val * svd from by ring]
    exact hbV jL e (by omega)
  -- stmt 4: acc += tl.sum(p_value[:, None] · v_value, 0)
  refine mistralIO_walkCons (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (mistralIO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (mistral_acc_step_eval _ BLOCK_N BLOCK_DMODEL ac0
        (fun j => pMasked stt Prob B_Att_Start_Loc B_Att_Seqlen sph spbs (c + j.val))
        (fun j e => vMasked stt V Req_to_tokens B_req_idx B_Seqlen srtb srts svbs svh svd
          kvg sw (c + j.val) e.val)
        (by
          simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true, reduceCtorEq]
          exact hacc)
        (by
          simp only [mistralIO_pMasked_setReg, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true, reduceCtorEq, BlockState.setReg_same])
        (by simp only [mistralIO_vMasked_setReg, BlockState.setReg_same]))) ?_
  refine mistralIO_walkNil _ ?_
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ⟨fun e => ac0 e + ∑ j : Fin BLOCK_N,
        pMasked stt Prob B_Att_Start_Loc B_Att_Seqlen sph spbs (c + j.val) *
          vMasked stt V Req_to_tokens B_req_idx B_Seqlen srtb srts svbs svh svd kvg sw
            (c + j.val) e.val,
      by simp only [BlockState.setReg_same]⟩⟩
  · funext rg o
    simp only [BlockState.setReg_mem]
    exact congrFun (congrFun hmem rg) o
  · simp only [BlockState.setReg_pids]
    exact hpids
  · simpa only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      reduceCtorEq] using hcb
  · simpa only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      reduceCtorEq] using hch
  · simpa only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      reduceCtorEq] using hn
  · simpa only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      reduceCtorEq] using hd
  · simpa only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      reduceCtorEq] using hsl
  · simpa only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      reduceCtorEq] using hbsl
  · simpa only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      reduceCtorEq] using hsi
  · simpa only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      reduceCtorEq] using hvoff
  · simpa only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      reduceCtorEq] using hpoff
  · simpa only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      reduceCtorEq] using hvoffs

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **The safety walk**: prelude ++ the dynamic loop (by the
`forRangeTraceSafeR` invariant principle at `mistralIOSafeInvW`, with the
`pre`-forced budget `T` supplying every live step's window bound) ++ the
postlude's deterministic 4-statement tail. -/
private theorem mistralIO_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (Prob V Out : RegionName) (Req_to_tokens B_req_idx : Region .nat)
    (B_Start_Loc : RegionName) (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (srtb srts sph spbs svbs svh svd sobs soh sod kvg sw BLOCK_DMODEL BLOCK_N T : Nat)
    (hpbs : spbs = 1) (hrts : srts = 1) (hBN : 0 < BLOCK_N) (s : BlockState)
    (hSle : attSeqLen s B_Att_Seqlen.cast ≤ T * BLOCK_N)
    (hbSQ : s.pids 0 < bounds (Region.cast B_Seqlen))
    (hbAS : s.pids 0 < bounds (Region.cast B_Att_Start_Loc))
    (hbRQ : s.pids 0 < bounds (Region.cast B_req_idx))
    (hbAQ : s.pids 0 < bounds (Region.cast B_Att_Seqlen))
    (hbP : ∀ (t : Fin T) (jL : Fin BLOCK_N),
      t.val * BLOCK_N + jL.val < attSeqLen s B_Att_Seqlen.cast →
      pOffset s B_Att_Start_Loc.cast sph spbs (t.val * BLOCK_N + jL.val) < bounds Prob)
    (hbG : ∀ (t : Fin T) (jL : Fin BLOCK_N),
      startIndex s B_Seqlen.cast sw + (t.val * BLOCK_N + jL.val)
        < batchSeqLen s B_Seqlen.cast →
      reqIdx s B_req_idx.cast * srtb
          + (startIndex s B_Seqlen.cast sw + (t.val * BLOCK_N + jL.val)) * srts
        < bounds (Region.cast Req_to_tokens))
    (hbV : ∀ (t : Fin T) (jL : Fin BLOCK_N) (d : Fin BLOCK_DMODEL),
      startIndex s B_Seqlen.cast sw + (t.val * BLOCK_N + jL.val)
        < batchSeqLen s B_Seqlen.cast →
      vLoc s (Region.cast Req_to_tokens) B_req_idx.cast B_Seqlen.cast srtb srts sw
            (t.val * BLOCK_N + jL.val) * svbs
          + (s.pids 1 / kvg) * svh + d.val * svd < bounds V)
    (hbO : ∀ i : Fin BLOCK_DMODEL,
      s.pids 0 * sobs + s.pids 1 * soh + i.val * sod < bounds Out) :
    ((token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
      B_Att_Start_Loc B_Att_Seqlen srtb srts sph spbs svbs svh svd sobs soh sod kvg sw
      BLOCK_DMODEL BLOCK_N).toAlgKernel).TraceSafeR R bounds s := by
  -- shared per-iteration body handler: the live step index is `cc / BLOCK_N`
  have hbody : ∀ (cc : Nat) (stt : BlockState), cc < attSeqLen s B_Att_Seqlen.cast →
      (mistralIOSafeInvW B_req_idx.cast B_Att_Start_Loc.cast B_Seqlen.cast
          B_Att_Seqlen.cast srtb srts sph spbs svh svd kvg sw BLOCK_DMODEL BLOCK_N s stt
        ∧ cc % BLOCK_N = 0) →
      Stmt.TraceSafeListR R bounds
          (mistralLoopBody Prob V (Region.cast Req_to_tokens) srts svbs BLOCK_DMODEL
            BLOCK_N)
          (stt.setReg "start_n" .nat [] (Tile.scalar cc))
        ∧ ∃ s', stepStmtsR R
              (mistralLoopBody Prob V (Region.cast Req_to_tokens) srts svbs BLOCK_DMODEL
                BLOCK_N)
              (stt.setReg "start_n" .nat [] (Tile.scalar cc)) = some s'
            ∧ (mistralIOSafeInvW B_req_idx.cast B_Att_Start_Loc.cast B_Seqlen.cast
                B_Att_Seqlen.cast srtb srts sph spbs svh svd kvg sw BLOCK_DMODEL BLOCK_N
                s s' ∧ (cc + BLOCK_N) % BLOCK_N = 0) := by
    intro cc stt hcc hPP
    obtain ⟨hPinv, hPmod⟩ := hPP
    have hceq : cc / BLOCK_N * BLOCK_N = cc :=
      Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hPmod)
    have hcT : cc / BLOCK_N < T :=
      mistralIO_step_lt cc (attSeqLen s B_Att_Seqlen.cast) T BLOCK_N hBN hcc hSle
    obtain ⟨hsafeB, s', hrunB, hInvB⟩ :=
      mistralIO_bodyW R bounds Prob V (Region.cast Req_to_tokens) B_req_idx.cast
        B_Att_Start_Loc.cast B_Seqlen.cast B_Att_Seqlen.cast srtb srts sph spbs svbs svh
        svd kvg sw BLOCK_DMODEL BLOCK_N hpbs hrts s stt cc hPinv
        (fun jL hlive => by
          have h := hbP ⟨cc / BLOCK_N, hcT⟩ jL (by rw [hceq]; exact hlive)
          rwa [hceq] at h)
        (fun jL hlive => by
          have h := hbG ⟨cc / BLOCK_N, hcT⟩ jL (by rw [hceq]; exact hlive)
          rwa [hceq] at h)
        (fun jL d hlive => by
          have h := hbV ⟨cc / BLOCK_N, hcT⟩ jL d (by rw [hceq]; exact hlive)
          rwa [hceq] at h)
    exact ⟨hsafeB, s', hrunB, hInvB, by rw [Nat.add_mod_right]; exact hPmod⟩
  unfold Kernel.TraceSafeR
  rw [mistralIO_body_split]
  obtain ⟨hsafePre, s0, hrun0, hInv0⟩ :=
    mistralIO_preLoopW R bounds B_req_idx B_Seqlen B_Att_Start_Loc B_Att_Seqlen srtb srts
      sph spbs svh svd kvg sw BLOCK_DMODEL BLOCK_N s hbSQ hbAS hbRQ hbAQ
  have hInv0' := hInv0
  obtain ⟨hmem0, hpids0, hcb0, hch0, hn0, hd0, hsl0, hbsl0, hsi0, hvoff0, hpoff0,
    hvoffs0, hac0⟩ := hInv0
  have hstopExact : evalOp (Op.ref .nat [] "cur_att_seq_len") s0
      = some (Tile.scalar (attSeqLen s B_Att_Seqlen.cast)) := by
    rw [evalOp_ref]; exact hsl0
  refine Stmt.TraceSafeListR.append_intro _ _ hsafePre ?_
  intro s1 hs1
  rw [hrun0] at hs1
  obtain rfl := Option.some.inj hs1
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s2 hs2 => ?_)
  · -- trace safety of the `forRangeDyn` itself
    simp only [Stmt.TraceSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
      by simp [Op.SafeAtR.eq_def], ?_⟩
    rw [mistralIO_evalOpR_constNat, mistralIO_evalOpR_constNat,
      mistralIO_stopOpR_castFree, hstopExact]
    refine Stmt.forRangeTraceSafeR_inv R bounds "start_n" _ _
      (mistralLoopBody Prob V (Region.cast Req_to_tokens) srts svbs BLOCK_DMODEL BLOCK_N)
      (fun i stt => mistralIOSafeInvW B_req_idx.cast B_Att_Start_Loc.cast B_Seqlen.cast
        B_Att_Seqlen.cast srtb srts sph spbs svh svd kvg sw BLOCK_DMODEL BLOCK_N s stt
        ∧ i % BLOCK_N = 0)
      ?_ _ s0 ⟨hInv0', Nat.zero_mod BLOCK_N⟩
    intro cc stt hcc hPP
    exact hbody cc stt hcc hPP
  · -- the loop's actual successor, then the postlude
    obtain ⟨finalC, sL, hLoopExact, hfin, hPL⟩ :=
      forRangeDyn_inv (idx := "start_n") (startOp := Op.constNat 0)
        (stopOp := Op.ref .nat [] "cur_att_seq_len")
        (stepOp := Op.constNat BLOCK_N)
        (P := fun i stt => mistralIOSafeInvW B_req_idx.cast B_Att_Start_Loc.cast
          B_Seqlen.cast B_Att_Seqlen.cast srtb srts sph spbs svh svd kvg sw BLOCK_DMODEL
          BLOCK_N s stt ∧ i % BLOCK_N = 0)
        (s_init := s0)
        (evalOp_constNat 0 s0) hstopExact (evalOp_constNat BLOCK_N s0)
        hBN.ne'
        ⟨hInv0', Nat.zero_mod BLOCK_N⟩
        (fun i stt hi hP => by
          obtain ⟨hsafeB, s', hrunB, hInvB⟩ := hbody i stt hi hP
          exact ⟨s', by
            rw [← mistralIO_body_castFree R Prob V (Region.cast Req_to_tokens) srts svbs
              BLOCK_DMODEL BLOCK_N]
            exact hrunB, hInvB⟩)
    rw [mistralIO_dyn_castFree R Prob V (Region.cast Req_to_tokens) srts svbs
      BLOCK_DMODEL BLOCK_N s0, hLoopExact] at hs2
    obtain rfl := Option.some.inj hs2
    obtain ⟨⟨hmemL, hpidsL, hcbL, hchL, hnL, hdL, hslL, hbslL, hsiL, hvoffL, hpoffL,
      hvoffsL, hacL⟩, hmodL⟩ := hPL
    -- postlude: the `.to(...)` self-assign, `off_o`, `out_ptrs`, the unmasked store
    unfold mistralPostlude
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s3 hs3 => ?_)
    obtain ⟨v3, hv3, rfl⟩ := stepStmtR_assign_inv hs3
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s4 hs4 => ?_)
    obtain ⟨v4, hv4, rfl⟩ := stepStmtR_assign_inv hs4
    rw [mistralIO_offoR_eval R _ (s.pids 0) (s.pids 1) sobs soh sod
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, reduceCtorEq]
        exact hcbL)
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, reduceCtorEq]
        exact hchL)
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, reduceCtorEq]
        exact hdL)] at hv4
    obtain rfl := Option.some.inj hv4
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s5 hs5 => ?_)
    obtain ⟨v5, hv5, rfl⟩ := stepStmtR_assign_inv hs5
    rw [mistralIO_outPtrR_eval R _ Out
      (fun i : Fin BLOCK_DMODEL => s.pids 0 * sobs + s.pids 1 * soh + i.val * sod)
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

`hrun` rides the exact `mistral_preLoop` → `forRangeDyn_inv` →
`mistral_postLoop` stack unchanged (everything is cast-free, so `execR R`
collapses onto it verbatim); the only new obligation the skin adds over the
existing headline is the **memory frame**. The prelude and the loop body only
write registers — `mistralInvariant` already carries `mem = s.mem` — so the
frame reduces to replaying the deterministic 4-statement postlude and framing
its scatter. -/

/-- A `writeMem` scatter `foldl` leaves every cell not hit by a lane
untouched (the store is unmasked, so there is no active-lane guard). -/
private theorem mistralIO_foldl_writeMem_frame {α : Type} (region : RegionName)
    (offFn : α → Nat) (valFn : α → ℝ) :
    ∀ (l : List α) (st : BlockState) (r : RegionName) (o : Nat),
      (r = region → ∀ k ∈ l, offFn k ≠ o) →
      ((l.foldl (fun acc k => acc.writeMem region (offFn k) (valFn k)) st).mem r o
        = st.mem r o)
  | [], _, _, _, _ => rfl
  | k :: rest, st, r, o, h => by
      rw [List.foldl_cons,
        mistralIO_foldl_writeMem_frame region offFn valFn rest _ r o
          (fun hr k' hk' => h hr k' (List.mem_cons_of_mem _ hk')),
        BlockState.writeMem_mem]
      rw [if_neg (fun hro => h hro.1 k List.mem_cons_self hro.2.symm)]

set_option maxHeartbeats 3200000 in
set_option maxRecDepth 8000 in
/-- **Postlude frame**: every cell outside the terminal store's footprint is
untouched by the postlude (replay of `mistral_postLoop`'s deterministic
4-statement chain, framed). -/
private theorem mistralIO_postlude_frame
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen : RegionName)
    (s0 : BlockState)
    (srtb srts sph spbs svbs svh svd sobs soh sod kvg sw BLOCK_DMODEL BLOCK_N : Nat)
    (final : Nat) (st sP : BlockState)
    (hinv : mistralInvariant Prob V Out Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen
      B_Att_Seqlen s0 srtb srts sph spbs svbs svh svd sobs soh sod kvg sw BLOCK_DMODEL
      BLOCK_N final st)
    (hpost : stepStmts (mistralPostlude Out sobs soh sod BLOCK_DMODEL) st = some sP) :
    ∀ r o, (r = Out → ∀ i : Fin BLOCK_DMODEL, o ≠ outOffset s0 sobs soh sod i) →
      sP.mem r o = st.mem r o := by
  simp only [mistralInvariant] at hinv
  obtain ⟨hpids, hmem, hundef, hcb, hch, hckv, hn, hd, hsl, hbsl, hsi, hvoff, hpoff,
    hvoffs, hacc⟩ := hinv
  set accFn : Fin BLOCK_DMODEL → ℝ := fun e =>
    partialAcc s0 Prob V Req_to_tokens B_req_idx B_Att_Start_Loc B_Seqlen B_Att_Seqlen
      srtb srts sph spbs svbs svh svd kvg sw final e.val with haccFn
  set offFn : TileIndex [BLOCK_DMODEL] → Nat :=
    fun i => outOffset s0 sobs soh sod i.1 with hoffFn
  unfold mistralPostlude at hpost
  -- stmt 0: acc = (acc).to(Out.dtype.element_ty) — a self-assign
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BLOCK_DMODEL] "acc") st
        = some (⟨fun idx : TileIndex [BLOCK_DMODEL] => some (accFn idx.1)⟩
            : Tile .real [BLOCK_DMODEL])
      from by rw [evalOp_ref]; exact hacc))] at hpost
  set s1 := st.setReg "acc" .real [BLOCK_DMODEL]
    (⟨fun idx : TileIndex [BLOCK_DMODEL] => some (accFn idx.1)⟩
      : Tile .real [BLOCK_DMODEL]) with hs1
  -- stmt 1: off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat sobs))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat soh)))
          (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_DMODEL] "offs_d")
            (Op.constNat sod))) s1
        = some (Tile.vec (fun e : Fin BLOCK_DMODEL => offFn (e, PUnit.unit))) from by
      simp only [hs1, evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
        BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
        String.reduceEq, hcb, hch, hd, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_
      ext idx
      simp only [Tile.bop, Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul,
        Broadcast.leftIndex, Broadcast.rightIndex, hoffFn, outOffset, dIndex]
      try ring))] at hpost
  set s2 := s1.setReg "off_o" .nat [BLOCK_DMODEL]
    (Tile.vec (fun e : Fin BLOCK_DMODEL => offFn (e, PUnit.unit))) with hs2
  -- stmt 2: out_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
          (Op.ref .nat [BLOCK_DMODEL] "off_o")) s2
        = some (⟨fun i : TileIndex [BLOCK_DMODEL] => ((Out : RegionName), offFn i)⟩
            : Tile .ptr [BLOCK_DMODEL]) from by
      simp only [hs2, evalOp, evalOp_ref_setReg_same, Option.bind]
      refine congrArg some ?_
      ext idx
      · simp only [Tile.ptrAdd, Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex,
          Broadcast.rightIndex, Region.cast_id]
      · simp only [Tile.ptrAdd, Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex,
          Broadcast.rightIndex, Nat.zero_add]))] at hpost
  set s3 := s2.setReg "out_ptrs" .ptr [BLOCK_DMODEL]
    (⟨fun i : TileIndex [BLOCK_DMODEL] => ((Out : RegionName), offFn i)⟩
      : Tile .ptr [BLOCK_DMODEL]) with hs3
  have hs3acc : s3.regs .real [BLOCK_DMODEL] "acc"
      = some (⟨fun idx : TileIndex [BLOCK_DMODEL] => some (accFn idx.1)⟩
          : Tile .real [BLOCK_DMODEL]) := by
    rw [hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hs1, BlockState.setReg_same]
  have hs3ptr : s3.regs .ptr [BLOCK_DMODEL] "out_ptrs"
      = some (⟨fun i : TileIndex [BLOCK_DMODEL] => ((Out : RegionName), offFn i)⟩
          : Tile .ptr [BLOCK_DMODEL]) := by
    rw [hs3, BlockState.setReg_same]
  have hstore : stepStmt (Stmt.store .real [BLOCK_DMODEL]
      (MemAccess.ptr (Op.ref .ptr [BLOCK_DMODEL] "out_ptrs"))
      (Op.ref .real [BLOCK_DMODEL] "acc") MaskOpt.none) s3
      = some ((TileShape.allIndices [BLOCK_DMODEL]).foldl
          (fun acc i => acc.writeMem Out (offFn i) (accFn i.1)) s3) := by
    unfold stepStmt
    simp only [evalOp_ref, hs3acc, hs3ptr, Option.bind_eq_bind, Option.bind_some,
      Option.map_some, BlockState.writeMemTyped_real, FloatDType.real_storeValue]
    rfl
  rw [stepStmts.cons_some hstore, stepStmts.nil] at hpost
  obtain rfl := Option.some.inj hpost
  intro r o hno
  refine (mistralIO_foldl_writeMem_frame Out (fun i => offFn i)
    (fun i => accFn i.1) _ s3 r o ?_).trans ?_
  · intro hr k _
    exact fun hoff' => hno hr k.1 hoff'.symm
  · rw [hs3, hs2, hs1]
    simp only [BlockState.setReg_mem]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Framed general execution**: the exact closed-form run of
`token_attn_mistral_closed_form_correct` re-assembled with the per-cell
memory frame (this is the exact run `hrun` rides). -/
private theorem mistralIO_exec_framed (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (srtb srts sph spbs svbs svh svd sobs soh sod kvg sw BLOCK_DMODEL BLOCK_N : Nat)
    (hpbs : spbs = 1) (hrts : srts = 1) (hBN : 0 < BLOCK_N)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s sobs soh sod i)) :
    ∃ sF, stepStmts ((token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen srtb srts sph spbs svbs svh svd
        sobs soh sod kvg sw BLOCK_DMODEL BLOCK_N).toAlgKernel.body) s = some sF
      ∧ (∀ i : Fin BLOCK_DMODEL,
          sF.readMem Out (outOffset s sobs soh sod i)
            = tokenAttnMistralClosedForm s Prob V Req_to_tokens B_req_idx
                B_Att_Start_Loc B_Seqlen B_Att_Seqlen srtb srts sph spbs svbs svh svd kvg
                sw BLOCK_DMODEL i)
      ∧ (∀ r o, (r ≠ Out ∨ ∀ i : Fin BLOCK_DMODEL,
            o ≠ outOffset s sobs soh sod i) → sF.mem r o = s.mem r o) := by
  obtain ⟨s', hpre, hinv0⟩ := mistral_preLoop Prob V Out Req_to_tokens B_req_idx
    B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen srtb srts sph spbs svbs svh svd
    sobs soh sod kvg sw BLOCK_DMODEL BLOCK_N s hundef
  have hslEntry : s'.regs .nat [] "cur_att_seq_len"
      = some (Tile.scalar (attSeqLen s B_Att_Seqlen.cast)) := by
    simp only [mistralInvariant] at hinv0
    exact hinv0.2.2.2.2.2.2.2.2.1
  have hstop : evalOp (Op.ref .nat [] "cur_att_seq_len") s'
      = some (Tile.scalar (attSeqLen s B_Att_Seqlen.cast)) := by
    rw [evalOp_ref]; exact hslEntry
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hinvFinal⟩ :=
    forRangeDyn_inv (idx := "start_n") (startOp := Op.constNat 0)
      (stopOp := Op.ref .nat [] "cur_att_seq_len") (stepOp := Op.constNat BLOCK_N)
      (start := 0) (stop := attSeqLen s B_Att_Seqlen.cast) (step := BLOCK_N)
      (P := mistralInvariant Prob V Out (Region.cast Req_to_tokens) B_req_idx.cast
        B_Att_Start_Loc.cast B_Seqlen.cast B_Att_Seqlen.cast s srtb srts sph spbs svbs
        svh svd sobs soh sod kvg sw BLOCK_DMODEL BLOCK_N)
      (by simp) hstop (by simp) (by omega) hinv0
      (fun c stt _ hP => mistral_loop_step Prob V Out (Region.cast Req_to_tokens)
        B_req_idx.cast B_Att_Start_Loc.cast B_Seqlen.cast B_Att_Seqlen.cast s srtb srts
        sph spbs svbs svh svd sobs soh sod kvg sw BLOCK_DMODEL BLOCK_N hpbs hrts c stt hP)
  obtain ⟨sfin, hPost, hRead⟩ := mistral_postLoop Prob V Out (Region.cast Req_to_tokens)
    B_req_idx.cast B_Att_Start_Loc.cast B_Seqlen.cast B_Att_Seqlen.cast s srtb srts sph
    spbs svbs svh svd sobs soh sod kvg sw BLOCK_DMODEL BLOCK_N final hfinal sLoop
    hinvFinal hOutInj
  have hframeP := mistralIO_postlude_frame Prob V Out (Region.cast Req_to_tokens)
    B_req_idx.cast B_Att_Start_Loc.cast B_Seqlen.cast B_Att_Seqlen.cast s srtb srts sph
    spbs svbs svh svd sobs soh sod kvg sw BLOCK_DMODEL BLOCK_N final sLoop sfin
    hinvFinal hPost
  have hmemL : sLoop.mem = s.mem := by
    simp only [mistralInvariant] at hinvFinal
    exact hinvFinal.2.1
  refine ⟨sfin, ?_, hRead, ?_⟩
  · rw [mistral_body_split, stepStmts.append_some hpre, stepStmts.cons_some hLoopStmt]
    exact hPost
  · intro r o hcond
    refine (hframeP r o ?_).trans (congrFun (congrFun hmemL r) o)
    intro hr i
    rcases hcond with hne | hno
    · exact absurd hr hne
    · exact hno i

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-! ### ════════ ★ MAIN THEOREM (io face) ★ ════════ -/
/-- **The `⊨[R]` gather-skin headline** — `token_attn_mistral` on
`StreamMetaGatherMasked3DKernelIO₂`, at fully symbolic per-axis strides and
symbolic `sliding_window`. For every rounding model `R`, the faithful surface
implements, on its gather-indexed signature, the streamed closed form
`mistralIOSpec`: every output lane `d` holds
`Σ_{n < m 3} p[n] · v[n, d]` over the attention window, read off the two
pinned streams, with out-of-window tokens contributing `0`. The kernel has
**zero rounding events** (four `.nat` slot loads, a `.nat` page-table gather,
`other = 0`-defaulted `.real` loads, `.real` in-loop arithmetic, and a
`.to(Out.dtype.element_ty)` that lowers to a self-assign, not an
`Op.castFloat`), so the skin's boundary quantization degenerates: the
readback's `R.round .real` is the identity by the model's defining
`round_real`.

**The gather channel.** `Req_to_tokens` enters as the skin's index channel
(`gty = .nat`, `gother = 0`), and the `V` window `read2` eats the gathered
tile: `G t jL · stride_vbs + …`. Unlike the softmax_reducev exemplar, the
Python `V` load carries **the gather's own mask**
(`mask2 = gmask` on the row coordinate), so a masked-off lane never
dereferences the substituted `other=` address and this port needs **no
hypothesis at all** on `gother`; only the gather pin's *active* leg is used.

**Launch legality (`pre` = the trusted-launch boundary).** The surface takes
no `max_input_len`-style host argument, so the skin's pid-free step budget
`T` cannot be derived from the kernel's own parameters: `T` is a **new
io-level parameter** and the triple is guarded by
`io.pre = (m 3 ≤ T · BLOCK_N)`, i.e. the host promises that the dynamic trip
count `cur_att_seq_len` fits the budget. This is a **disclosed launch
restriction** — the honest cost of putting a data-dependent trip count on a
fixed-`T` streaming skin — not a derived fact, and the `⊨[R]` triple says
nothing about launches outside it.

**Hypothesis provenance**: `stride_pbs = 1` and `stride_req_to_tokens_s = 1`
are the exact headline's contiguous-layout side conditions (the per-lane
address arithmetic `p_offs + start_n` / `v_loc_off + start_n·stride_s` folds
into the closed form's `(base + n)·stride` only at unit stride); `0 < BLOCK_N`
is the exact headline's nonempty-block condition (and what makes the step
budget citable); `hOutInj` restates the exact headline's **open**
output-offset injectivity side condition in ∀-pids form (per-axis strides are
symbolic, so no contiguity discharge is available). The exact headline's
`hundef` is **not** a hypothesis: the skin's Hoare triple carries the `undef`
pin itself.

Relation to the exact surface: the `Realizes_without_Rounding` headline above
is retained unchanged; this `⊨[R]` face restates the same genuine closed form
on the gather skin, for every `R` at once. -/
specification token_attn_mistral_io_correctness (R : RoundingModel)
    (Prob V Out : RegionName) (Req_to_tokens B_req_idx : Region .nat)
    (B_Start_Loc : RegionName) (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (srtb srts sph spbs svbs svh svd sobs soh sod kvg sw BLOCK_DMODEL BLOCK_N T : Nat)
    (hpbs : spbs = 1) (hrts : srts = 1) (hBN : 0 < BLOCK_N)
    (hOutInj : ∀ pid₀ pid₁ : Nat, Function.Injective
      (fun i : Fin BLOCK_DMODEL => pid₀ * sobs + pid₁ * soh + i.val * sod)) :
    mistralIO Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen B_Att_Start_Loc
        B_Att_Seqlen srtb srts sph spbs svbs svh svd sobs soh sod kvg sw BLOCK_DMODEL
        BLOCK_N T ⊨[R]
      fun _ _ _ m xs ys j =>
        mistralIOSpec BLOCK_N BLOCK_DMODEL T (m (⟨3, by omega⟩ : Fin 4))
          (m (⟨0, by omega⟩ : Fin 4)) sw hBN xs ys j := by
  refine StreamMetaGatherMasked3DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact mistralIO_flattenOk Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
      B_Att_Start_Loc B_Att_Seqlen srtb srts sph spbs svbs svh svd sobs soh sod kvg sw
      BLOCK_DMODEL BLOCK_N
  · -- the safety walk
    intro bounds s m G xs ys hpre hm hg hgo hx hy hbm hbrG hbr1 hbr2 hbw
    simp only [mistralIO] at hpre hm hg hgo hbm hbrG hbr1 hbr2 hbw ⊢
    have hm0 : batchSeqLen s B_Seqlen.cast = m (⟨0, by omega⟩ : Fin 4) :=
      hm (⟨0, by omega⟩ : Fin 4)
    have hm1 : attStartLoc s B_Att_Start_Loc.cast = m (⟨1, by omega⟩ : Fin 4) :=
      hm (⟨1, by omega⟩ : Fin 4)
    have hm2 : reqIdx s B_req_idx.cast = m (⟨2, by omega⟩ : Fin 4) :=
      hm (⟨2, by omega⟩ : Fin 4)
    have hm3 : attSeqLen s B_Att_Seqlen.cast = m (⟨3, by omega⟩ : Fin 4) :=
      hm (⟨3, by omega⟩ : Fin 4)
    rw [← hm3] at hpre hbr1
    rw [← hm1] at hbr1
    rw [← hm0] at hg hbrG hbr2
    rw [← hm2] at hg hbrG
    refine mistralIO_traceSafeR R bounds Prob V Out Req_to_tokens B_req_idx B_Start_Loc
      B_Seqlen B_Att_Start_Loc B_Att_Seqlen srtb srts sph spbs svbs svh svd sobs soh sod
      kvg sw BLOCK_DMODEL BLOCK_N T hpbs hrts hBN s hpre
      (hbm (⟨0, by omega⟩ : Fin 4)) (hbm (⟨1, by omega⟩ : Fin 4))
      (hbm (⟨2, by omega⟩ : Fin 4)) (hbm (⟨3, by omega⟩ : Fin 4)) ?_ ?_ ?_ ?_
    · -- the `Prob` stream's live window
      intro t jL hlive
      simp only [pOffset]
      exact hbr1 t jL hlive
    · -- the gather channel's live window
      intro t jL hlive
      simp only [startIndex] at hlive ⊢
      exact hbrG t jL hlive
    · -- the gather-addressed `V` stream, on the gather's own window
      intro t jL d hlive
      simp only [startIndex] at hlive
      have hGeq : vLoc s (Region.cast Req_to_tokens) B_req_idx.cast B_Seqlen.cast srtb
          srts sw (t.val * BLOCK_N + jL.val) = G t jL := by
        simp only [vLoc, startIndex]
        exact hg t jL hlive
      rw [hGeq]
      have h := hbr2 t (Lane2D.encode
        ((jL, d, PUnit.unit) : TileIndex [BLOCK_N, BLOCK_DMODEL]))
        (by rw [Lane2D.decode_encode]; exact hlive)
      rw [Lane2D.decode_encode] at h
      exact h
    · -- the terminal store
      intro i
      exact hbw i trivial
  · -- the rounded Hoare triple: framed exact stack + cast-free collapse
    intro s₀ m G xs ys hpre hu hm hg hgo hx hy
    simp only [mistralIO] at hpre hm hg hgo hx hy ⊢
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hu]
    have hm0 : batchSeqLen s₀ B_Seqlen.cast = m (⟨0, by omega⟩ : Fin 4) :=
      hm (⟨0, by omega⟩ : Fin 4)
    have hm1 : attStartLoc s₀ B_Att_Start_Loc.cast = m (⟨1, by omega⟩ : Fin 4) :=
      hm (⟨1, by omega⟩ : Fin 4)
    have hm2 : reqIdx s₀ B_req_idx.cast = m (⟨2, by omega⟩ : Fin 4) :=
      hm (⟨2, by omega⟩ : Fin 4)
    have hm3 : attSeqLen s₀ B_Att_Seqlen.cast = m (⟨3, by omega⟩ : Fin 4) :=
      hm (⟨3, by omega⟩ : Fin 4)
    have hOInj : Function.Injective
        (fun i : Fin BLOCK_DMODEL => outOffset s₀ sobs soh sod i) := by
      simpa only [outOffset, dIndex] using hOutInj (s₀.pids 0) (s₀.pids 1)
    obtain ⟨sF, hstep, hOut, hframe⟩ :=
      mistralIO_exec_framed Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
        B_Att_Start_Loc B_Att_Seqlen srtb srts sph spbs svbs svh svd sobs soh sod kvg sw
        BLOCK_DMODEL BLOCK_N hpbs hrts hBN s₀ hundef' hOInj
    refine ⟨sF, ?_, ?_, ?_⟩
    · rw [mistralIO_execR_collapse]
      exact hstep
    · -- readback: the genuine closed form = the streamed closed form
      intro j _
      simp only [BlockState.readMemAs_real, R.round_real_apply]
      refine congrArg some ((hOut j).trans ?_)
      exact mistralIOSpec_eq_genuine Prob V (Region.cast Req_to_tokens) B_req_idx.cast
        B_Att_Start_Loc.cast B_Seqlen.cast B_Att_Seqlen.cast srtb srts sph spbs svbs svh
        svd kvg sw BLOCK_DMODEL BLOCK_N T hBN s₀ (m (⟨3, by omega⟩ : Fin 4))
        (m (⟨1, by omega⟩ : Fin 4)) (m (⟨0, by omega⟩ : Fin 4))
        (m (⟨2, by omega⟩ : Fin 4)) hm3 hm1 hm0 hm2 hpre G xs ys
        (fun t jL h => hg t jL h) (fun t jL h => hx t jL h) (fun t j' h => hy t j' h) j
    · -- the frame
      intro r o hcond
      refine hframe r o ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · exact Or.inr (fun i => hno i trivial)


end IOFace

end VeriTile.Bench.TritonBenchG.TokenAttnMistral

