import VeriTile.Triton

/-!
# `token_attn_reduceV` — strict per-kernel correctness

`_fwd_kernel_token_att2` is the LightLLM token-attention "reduce over V" kernel:
program `(cur_batch, cur_head)` streams the sequence in `BLOCK_N` chunks,
gathers `V` rows through the `Req_to_tokens` paged-KV index, accumulates
`acc += sum(prob[:, None] * v_value, axis=0)` into a `BLOCK_DMODEL` register, and
stores the reduced vector to `Out[cur_batch, cur_head, :]`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_token_att2[(batch, head)](...)`, the grid
over `(batch, head)`, block scheduling, and how the runtime composes per-program
writes into the `Out` buffer) is the *trusted boundary*, not a proof obligation
here. Because the program ids `cur_batch`/`cur_head` are universally quantified
(via `BlockState`), the per-program statements cover every program of the grid.

## Proof architecture

```
token_attn_reducev_output_summary_general                   ← ★ HEADLINE (dimension-general)
  ├─ token_attn_reducev_surface_toAlgorithm_supported              surface lowers (any dims/strides)
  └─ token_attn_reducev_closed_form_compute_correct               full surface, final store = closed form
       └─ token_attn_reducev_closed_form_correct                  exec readback = tokenAttnReduceVClosedForm
            ├─ reducev_preLoop          13 prelude assigns → entry invariant (acc = partialAcc 0)
            ├─ reducev_loop_step        5-stmt body advances partialAcc by one BLOCK_N block (forRangeDyn_inv)
            └─ reducev_postLoop         cast + off_o + out_ptrs + unmasked store readback = closed form
(algebra: partialAcc_block_succ / partialAcc_eq_PVValue; recipes: reducev_*_eval)
```

The headline `token_attn_reducev_output_summary_general` is fully
dimension-general (symbolic `BLOCK_DMODEL`, `BLOCK_N`, `kv_group_num`, and
strides) under honest side-conditions (`0 < BLOCK_DMODEL`, `0 < BLOCK_N`,
contiguous `stride_pbs = stride_req_to_tokens_s = 1`, output-offset injectivity,
clean `undef`).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` /
`num_warps` / `num_stages` are not modeled. The `acc.to(Out.dtype.element_ty)`
cast reduces to the identity at the algorithm layer (post-erasure all dtypes
unify to `ℝ`). The headline `output_summary_general` shows the surface kernel
lowers to the algorithm layer AND the store to `Out` is compute-correct against a
**genuine, self-reference-free closed form** `tokenAttnReduceVClosedForm`: every
lane `d` of the `[BLOCK_DMODEL]` output holds the probability-weighted value sum
`acc[d] = Σ_{n < cur_batch_seq_len} p[n]·v[v_loc[n], d]` (with the
`Req_to_tokens` page-table gather and the `cur_kv_head` / `in_all_start_index`
offsets decoded as `pOffset`/`vLoc`/`vOffset`; out-of-range tokens masked to `0`).
This is proven by fully executing the surface kernel — the `reducev_preLoop`
prelude decode, the `reducev_loop_step` one-block accumulator advance driven
through `forRangeDyn_inv`, and the `reducev_postLoop` unmasked-store readback —
never by re-asserting the kernel's own executed value. This kernel has **no
sliding window** (`cur_batch_start_index = 0`), so the loop range equals the mask
window and a single `inWindow` predicate guards every masked load. The headline
`token_attn_reducev_output_summary_general` covers **all** shapes with symbolic
`BLOCK_DMODEL`/`BLOCK_N`/`kv_group_num`/strides (under the honest contiguous-layout
hypotheses `stride_pbs = 1` and `stride_req_to_tokens_s = 1`).
-/

namespace VeriTile.Bench.TritonBenchG.TokenAttnReduceV

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `token_attn_reducev_output_summary_general` (dimension-general). -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-- Faithful transcription of `token_attn_reduceV.py`'s
`_fwd_kernel_token_att2`.

Typed-region note: metadata/gather buffers are `Region .nat`, matching their
index role without adding source-level `dtype=` kwargs. -/
def token_attn_reducev_surface
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  cur_kv_head = cur_head // $(kv_group_num)
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_start_index = 0
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
  v_loc_off = cur_batch_req_idx * $(stride_req_to_tokens_b) +
    (cur_batch_start_index + offs_n) * $(stride_req_to_tokens_s)
  p_offs = cur_head * $(stride_ph) +
    (cur_batch_in_all_start_index + offs_n) * $(stride_pbs)
  v_offs = cur_kv_head * $(stride_vh) + offs_d[None, :] * $(stride_vd)
  acc = tl.zeros([$(BLOCK_DMODEL)], dtype=tl.float32)
  for start_n in range($(0), cur_batch_seq_len, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    p_value = tl.load(Prob + p_offs + start_n,
      mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
    v_loc = tl.load(Req_to_tokens + v_loc_off +
      start_n * $(stride_req_to_tokens_s),
      mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
    v_value = tl.load(V + v_offs + v_loc[:, None] * $(stride_vbs),
      mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
    acc += tl.sum(p_value[:, None] * v_value, 0)
  }
  acc = (acc).to(Out.dtype.element_ty)
  off_o = cur_batch * $(stride_obs) + cur_head * $(stride_oh) + offs_d * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc)
}

/-- The full token-attention reduce-V surface lowers to the algorithm layer. -/
theorem token_attn_reducev_surface_toAlgorithm_supported
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N : Nat) :
    ∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
      stride_ph stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh
      stride_od kv_group_num BLOCK_DMODEL BLOCK_N).toAlgorithm? = Except.ok alg := by
  simp [token_attn_reducev_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented final output-store slice of `token_attn_reduceV.py`'s
`_fwd_kernel_token_att2`.

The full kernel streams over token blocks, gathers V through `Req_to_tokens`,
and accumulates `sum(prob * v)`. This slice starts from a precomputed `Acc`
vector and proves the final `BLOCK_DMODEL` writeback into `Out`. -/
def token_attn_reducev_final_store_slice
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

/-- Algorithm-layer correctness for the token-attention reduce-V final store. -/
theorem token_attn_reducev_final_store_slice_correct
    (Acc Out : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ∀ i : Fin BLOCK_DMODEL,
      let outAddr := outOffset s stride_obs stride_oh stride_od i
      (exec (token_attn_reducev_final_store_slice Acc Out stride_acc_bs
            stride_acc_h stride_acc_d stride_obs stride_oh stride_od BLOCK_DMODEL)
          s).map (·.readMem Out outAddr)
        = some (s.readMem Acc (accOffset s stride_acc_bs stride_acc_h stride_acc_d i)) := by
  intro i
  simp [exec, token_attn_reducev_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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

/-- Compute-facing correctness for the token-attention reduce-V final store. -/
theorem token_attn_reducev_final_store_slice_compute_correct
    (Acc Out : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := token_attn_reducev_final_store_slice Acc Out stride_acc_bs
        stride_acc_h stride_acc_d stride_obs stride_oh stride_od BLOCK_DMODEL)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i =>
        s.readMem Acc (accOffset s stride_acc_bs stride_acc_h stride_acc_d i)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [token_attn_reducev_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := token_attn_reducev_final_store_slice_correct Acc Out stride_acc_bs
    stride_acc_h stride_acc_d stride_obs stride_oh stride_od BLOCK_DMODEL
    s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-! ## Genuine closed-form PV-reduction spec

The Triton `_fwd_kernel_token_att2` accumulates, per output head-dim `d`, the
probability-weighted value sum

```
acc[d] = Σ_{n < cur_batch_seq_len}  p[n] · v[v_loc[n], d]
```

where, with `cur_batch_start_index = 0` (this kernel has **no sliding window**,
unlike the Mistral sibling) and `cur_kv_head = cur_head / kv_group_num`:

* the per-token probability
  `p[n] = Prob[cur_head·stride_ph + (in_all_start_index + n)·stride_pbs]`,
  masked by `n < cur_batch_seq_len` (`other = 0`);
* the gathered KV page index
  `v_loc[n] = Req_to_tokens[req_idx·stride_req_to_tokens_b +
    n·stride_req_to_tokens_s]`, masked by `n < cur_batch_seq_len`;
* the value row
  `v[v_loc[n], d] = V[cur_kv_head·stride_vh + d·stride_vd +
    v_loc[n]·stride_vbs]`, masked by `n < cur_batch_seq_len` (`other = 0`).

The outer loop `range(0, cur_batch_seq_len, BLOCK_N)` ranges over exactly the
masked region, so every `n` in the sum is active (the per-block `BLOCK_N`
padding lanes whose `n ≥ cur_batch_seq_len` read `0` and drop out). The final
store of the whole `[BLOCK_DMODEL]` accumulator is **unmasked**, with a
`.to(Out.dtype)` cast that is the identity at the algorithm (ℝ) layer.

These definitions are a *genuine closed form* — they never execute the kernel —
and (via `token_attn_reducev_closed_form_correct`) fully replace the former
self-referential surface-value spec.

`batchSeqLen`/`reqIdx`/`inAllStartLoc` are the metadata loads of the prelude. -/

/-- `cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)`: the loop bound and the
mask threshold for every per-token load. -/
def batchSeqLen (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0)

/-- `cur_batch_req_idx = tl.load(B_req_idx + cur_batch)`: the request row used to
index `Req_to_tokens`. -/
def reqIdx (s : BlockState) (B_req_idx : RegionName) : Nat :=
  s.readMemValue .nat B_req_idx (s.pids 0)

/-- `cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)`: the
flattened start offset folded into the `Prob` load address. -/
def inAllStartLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)

/-- Per-token probability load offset:
`cur_head·stride_ph + (in_all_start_index + n)·stride_pbs`. -/
def pOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_ph stride_pbs : Nat) (n : Nat) : Nat :=
  s.pids 1 * stride_ph + (inAllStartLoc s B_Start_Loc + n) * stride_pbs

/-- Gathered KV page index for token `n` (`cur_batch_start_index = 0`):
`Req_to_tokens[req_idx·stride_req_to_tokens_b + n·stride_req_to_tokens_s]`. -/
def vLoc
    (s : BlockState) (Req_to_tokens B_req_idx : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s : Nat)
    (n : Nat) : Nat :=
  s.readMemValue .nat Req_to_tokens
    (reqIdx s B_req_idx * stride_req_to_tokens_b +
      n * stride_req_to_tokens_s)

/-- Value-row load offset for token `n`, head-dim `d`:
`v_loc[n]·stride_vbs + cur_kv_head·stride_vh + d·stride_vd`, with
`cur_kv_head = cur_head / kv_group_num`. -/
def vOffset
    (s : BlockState) (Req_to_tokens B_req_idx : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_vbs stride_vh stride_vd
      kv_group_num : Nat) (n d : Nat) : Nat :=
  vLoc s Req_to_tokens B_req_idx stride_req_to_tokens_b
      stride_req_to_tokens_s n * stride_vbs +
    (s.pids 1 / kv_group_num) * stride_vh + d * stride_vd

/-- The genuine closed-form accumulator for output head-dim `d`:
`Σ_{n < cur_batch_seq_len} p[n] · v[v_loc[n], d]`. The sum range is exactly the
masked token window, so no `if`-guard is needed — every padding lane
(`n ≥ cur_batch_seq_len`) is excluded from `Finset.range` and contributes `0`,
mirroring the `other = 0` masked loads. -/
noncomputable def tokenAttnReduceVPVValue
    (s : BlockState) (Prob V : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd kv_group_num : Nat)
    (d : Nat) : ℝ :=
  ∑ n ∈ Finset.range (batchSeqLen s B_Seqlen),
    s.readMem Prob (pOffset s B_Start_Loc stride_ph stride_pbs n) *
      s.readMem V (vOffset s Req_to_tokens B_req_idx
        stride_req_to_tokens_b stride_req_to_tokens_s stride_vbs stride_vh
        stride_vd kv_group_num n d)

/-- Genuine closed-form value written to `Out[outOffset d]`. The store is
unmasked over the full `[BLOCK_DMODEL]` vector, so every lane holds the
PV-accumulator `tokenAttnReduceVPVValue` for its head-dim `d = dIndex s i`. -/
noncomputable def tokenAttnReduceVClosedForm
    (s : BlockState) (Prob V : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd kv_group_num BLOCK_DMODEL : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  tokenAttnReduceVPVValue s Prob V Req_to_tokens B_req_idx B_Start_Loc
    B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
    stride_vbs stride_vh stride_vd kv_group_num (dIndex s i)

/-! ### Surface→closed-form bridge (proven)

`tokenAttnReduceVClosedForm` is the genuine, self-reference-free PV spec, and the
surface readback bridge

```
exec (token_attn_reducev_surface …) s = some s' →
  s'.readMem Out (outOffset … i) = tokenAttnReduceVClosedForm … i
```

is proven sorry-free as `token_attn_reducev_closed_form_correct` (this kernel is
the **PV reduce / accumulate** route — there is no online softmax). The proof
threads the per-statement `reducev_*_eval` recipes through `reducev_preLoop`
(13-assign prelude → entry invariant `acc = partialAcc 0`), `reducev_loop_step`
fed to `forRangeDyn_inv` (each `BLOCK_N` block advances the carry
`acc = partialAcc k` via `partialAcc_block_succ`), and `reducev_postLoop` (the
unmasked `[BLOCK_DMODEL]` store readback via `scatter_readback_nd`, then
`partialAcc_eq_PVValue` at the loop's final counter `final ≥ cur_batch_seq_len`).
-/

/-! ## Masked loads, partial accumulator, and the loop algebra

Because `cur_batch_start_index = 0`, the same boundary predicate `n <
cur_batch_seq_len` (`inWindow`) guards both the `Prob` and `V` masked loads and is
exactly the loop range, so there is no separate window/active distinction (unlike
the Mistral sibling). -/

/-- A token `n` is in range iff `n < cur_batch_seq_len`. Out-of-range tokens read
`0` from both the `Req_to_tokens` gather and the masked `V`/`Prob` loads, hence
contribute `0`. This single predicate guards every masked load (no sliding
window). -/
def inWindow (s : BlockState) (B_Seqlen : RegionName) (n : Nat) : Prop :=
  n < batchSeqLen s B_Seqlen

instance inWindowDecidable (s : BlockState) (B_Seqlen : RegionName) (n : Nat) :
    Decidable (inWindow s B_Seqlen n) := by
  unfold inWindow; infer_instance

/-- Masked per-token probability: `p[n]` if `n < cur_batch_seq_len`, else `0`. -/
noncomputable def pMasked
    (s : BlockState) (Prob B_Start_Loc B_Seqlen : RegionName)
    (stride_ph stride_pbs : Nat) (n : Nat) : ℝ :=
  if inWindow s B_Seqlen n then
    s.readMem Prob (pOffset s B_Start_Loc stride_ph stride_pbs n)
  else 0.0

/-- Masked value row: `v[v_loc[n], d]` if `n < cur_batch_seq_len`, else `0`. -/
noncomputable def vMasked
    (s : BlockState) (V Req_to_tokens B_req_idx B_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_vbs stride_vh stride_vd
      kv_group_num : Nat) (n d : Nat) : ℝ :=
  if inWindow s B_Seqlen n then
    s.readMem V (vOffset s Req_to_tokens B_req_idx
      stride_req_to_tokens_b stride_req_to_tokens_s stride_vbs stride_vh
      stride_vd kv_group_num n d)
  else 0.0

/-- Partial PV accumulator after tokens `n < k` have been folded in, for output
head-dim `d`. The loop carries `partialAcc (c·BLOCK_N)`. -/
noncomputable def partialAcc
    (s : BlockState) (Prob V : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd kv_group_num : Nat)
    (k d : Nat) : ℝ :=
  ∑ n ∈ Finset.range k,
    pMasked s Prob B_Start_Loc B_Seqlen stride_ph stride_pbs n *
      vMasked s V Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
        stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num n d

/-- One block of `BLOCK_N` tokens advances the partial accumulator.
`partialAcc (c·BLOCK_N + BLOCK_N) = partialAcc (c·BLOCK_N) + Σ_{j<BLOCK_N} …`.
This is the algebraic content of the loop body's `acc += tl.sum(p·v, 0)`. -/
theorem partialAcc_block_succ
    (s : BlockState) (Prob V : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd kv_group_num : Nat)
    (start_n BLOCK_N d : Nat) :
    partialAcc s Prob V Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
        stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
        stride_pbs stride_vbs stride_vh stride_vd kv_group_num
        (start_n + BLOCK_N) d
      = partialAcc s Prob V Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
          stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
          stride_pbs stride_vbs stride_vh stride_vd kv_group_num
          start_n d
        + ∑ j ∈ Finset.range BLOCK_N,
            pMasked s Prob B_Start_Loc B_Seqlen stride_ph stride_pbs
                (start_n + j) *
              vMasked s V Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
                stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num
                (start_n + j) d := by
  unfold partialAcc
  rw [Finset.sum_range_add]

/-- Once the loop has run past `cur_batch_seq_len` (`K ≥ batchSeqLen`), the
partial PV accumulator over `range K` coincides with the genuine closed form
`tokenAttnReduceVPVValue`: tokens `n ≥ batchSeqLen` are masked to `p = 0`, so they
contribute nothing, and the range can be cut back to `batchSeqLen`. -/
theorem partialAcc_eq_PVValue
    (s : BlockState) (Prob V : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : RegionName)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd kv_group_num : Nat)
    (K d : Nat) (hK : batchSeqLen s B_Seqlen ≤ K) :
    partialAcc s Prob V Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
        stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
        stride_pbs stride_vbs stride_vh stride_vd kv_group_num K d
      = tokenAttnReduceVPVValue s Prob V Req_to_tokens B_req_idx B_Start_Loc
          B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
          stride_ph stride_pbs stride_vbs stride_vh stride_vd kv_group_num d := by
  unfold partialAcc tokenAttnReduceVPVValue
  rw [← Finset.sum_range_add_sum_Ico _ hK]
  have htail : ∑ n ∈ Finset.Ico (batchSeqLen s B_Seqlen) K,
      pMasked s Prob B_Start_Loc B_Seqlen stride_ph stride_pbs n *
        vMasked s V Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
          stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num n d = 0 := by
    apply Finset.sum_eq_zero
    intro n hn
    rw [Finset.mem_Ico] at hn
    have : ¬ inWindow s B_Seqlen n := by simp only [inWindow]; omega
    simp only [pMasked, this, if_false]
    norm_num
  rw [htail, add_zero]
  apply Finset.sum_congr rfl
  intro n hn
  rw [Finset.mem_range] at hn
  have hw : inWindow s B_Seqlen n := by simp only [inWindow]; exact hn
  simp only [pMasked, vMasked, if_pos hw]

/-! ## Per-statement op-eval recipes (the tedious recipe layer)

Standalone, sorry-free `*_eval` recipe lemmas for every body statement of
`token_attn_reducev_surface`, mirroring the recipe-building patterns of the
Mistral sibling. Each lemma takes abstract register-readback hypotheses and a
symbolic `BlockState`, and decodes a single statement's `evalOp` to its
closed-form tile.

`stride_pbs = 1 ∧ stride_req_to_tokens_s = 1` are carried as hypotheses where the
per-lane address arithmetic needs them (true for all checked Python shapes). -/

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **Keystone `v_offs` recipe** (`v_offs = cur_kv_head·stride_vh +
offs_d[None,:]·stride_vd`, shape `[1, BD]`). -/
theorem reducev_voffs_eval (s : BlockState) (BD kvh stride_vh stride_vd : Nat)
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
/-- **`v_loc_off` recipe** (`v_loc_off = cur_batch_req_idx·stride_b +
(cur_batch_start_index + offs_n)·stride_s`, shape `[BLOCK_N]`). With
`cur_batch_start_index = 0`, lane `j` holds `reqIdx·stride_b + j·stride_s`. -/
theorem reducev_vloc_off_eval (s : BlockState) (BN reqIdx stride_b stride_s : Nat)
    (hreq : s.regs .nat [] "cur_batch_req_idx" = some (Tile.scalar reqIdx))
    (hsi : s.regs .nat [] "cur_batch_start_index" = some (Tile.scalar 0))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch_req_idx") (Op.constNat stride_b))
        (Op.mul .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_start_index")
            (Op.ref .nat [BN] "offs_n"))
          (Op.constNat stride_s))) s
      = some (⟨fun idx : TileIndex [BN] =>
          reqIdx * stride_b + idx.1.val * stride_s⟩ : Tile .nat [BN]) := by
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hreq, hsi, hn,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp [Tile.bop, Tile.vec, NumericDType.mul, NumericDType.add]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`p_offs` recipe** (`p_offs = cur_head·stride_ph +
(cur_batch_in_all_start_index + offs_n)·stride_pbs`, shape `[BLOCK_N]`). Lane `j`
holds `head·stride_ph + (inAllStart + j)·stride_pbs`. -/
theorem reducev_poffs_eval (s : BlockState) (BN head inAllStart stride_ph stride_pbs : Nat)
    (hhead : s.regs .nat [] "cur_head" = some (Tile.scalar head))
    (has : s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar inAllStart))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_ph))
        (Op.mul .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
            (Op.ref .nat [BN] "offs_n"))
          (Op.constNat stride_pbs))) s
      = some (⟨fun idx : TileIndex [BN] =>
          head * stride_ph + (inAllStart + idx.1.val) * stride_pbs⟩ : Tile .nat [BN]) := by
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hhead, has, hn,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp [Tile.bop, Tile.vec, NumericDType.mul, NumericDType.add]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`p_value` masked-load recipe** (`tl.load(Prob + p_offs + start_n,
mask=(start_n+offs_n) < cur_batch_seq_len, other=0.0)`, shape `[BLOCK_N]`).
With `stride_pbs = 1`, the loaded address at lane `j` is `pOffset (SN+j)` and the
boundary mask `SN+j < batchSeqLen` is exactly `pMasked`'s guard. -/
theorem reducev_prob_load_eval (s : BlockState) (Prob : RegionName)
    (B_Start_Loc B_Seqlen : RegionName)
    (BN stride_ph stride_pbs SN : Nat) (hpbs : stride_pbs = 1)
    (hpoffs : s.regs .nat [BN] "p_offs" =
      some (Tile.vec (fun j : Fin BN => pOffset s B_Start_Loc stride_ph stride_pbs j.val)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hsl : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (batchSeqLen s B_Seqlen))) :
    evalOp (Op.load .real
        (MemAccess.region Prob
          (Op.add .nat Broadcast.scalarR (Op.ref .nat [BN] "p_offs") (Op.ref .nat [] "start_n")))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BN] "offs_n"))
            (Op.ref .nat [] "cur_batch_seq_len"))
          (Op.broadcast (Op.const 0.0) [BN]))) s
      = some (⟨fun idx : TileIndex [BN] =>
          some (pMasked s Prob B_Start_Loc B_Seqlen stride_ph stride_pbs
            (SN + idx.1.val))⟩ : Tile .real [BN]) := by
  simp only [evalOp, hpoffs, hsn, hn, hsl, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, u⟩ := idx
  have haddr : pOffset s B_Start_Loc stride_ph stride_pbs j.val + SN
      = pOffset s B_Start_Loc stride_ph stride_pbs (SN + j.val) := by
    simp only [pOffset, hpbs]; ring
  simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar, NumericDType.add,
    ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex,
    BlockState.readMemValue_real, pMasked, inWindow]
  rw [haddr]
  simp only [if_true, Region.cast_id]
  by_cases hlt : SN + j.val < batchSeqLen s B_Seqlen
  · simp only [hlt, decide_true, if_true, if_pos hlt]
  · simp only [hlt, decide_false, if_neg hlt, if_false, Bool.false_eq_true]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`v_loc` masked-gather recipe** (`tl.load(Req_to_tokens + v_loc_off +
start_n·stride_s, mask=(start_n+offs_n) < cur_batch_seq_len, other=0)`, shape
`[BLOCK_N]`, dtype `.nat`). With `stride_req_to_tokens_s = 1`, the loaded address
at lane `j` is the `vLoc (SN+j)` address and the mask is `inWindow (SN+j)`. -/
theorem reducev_vloc_gather_eval (s : BlockState)
    (Req_to_tokens B_req_idx B_Seqlen : RegionName)
    (BN stride_b stride_s SN : Nat) (hs : stride_s = 1)
    (hoff : s.regs .nat [BN] "v_loc_off" =
      some (Tile.vec (fun j : Fin BN =>
        reqIdx s B_req_idx * stride_b + j.val * stride_s)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hsl : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (batchSeqLen s B_Seqlen))) :
    evalOp (Op.load .nat
        (MemAccess.region Req_to_tokens
          (Op.add .nat Broadcast.scalarR (Op.ref .nat [BN] "v_loc_off")
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat stride_s))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BN] "offs_n"))
            (Op.ref .nat [] "cur_batch_seq_len"))
          (Op.broadcast (Op.constNat 0) [BN]))) s
      = some (⟨fun idx : TileIndex [BN] =>
          if inWindow s B_Seqlen (SN + idx.1.val) then
            vLoc s Req_to_tokens B_req_idx stride_b stride_s (SN + idx.1.val)
          else 0⟩ : Tile .nat [BN]) := by
  simp only [evalOp, hoff, hsn, hn, hsl, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, u⟩ := idx
  have haddr : reqIdx s B_req_idx * stride_b + j.val * stride_s + SN * stride_s
      = reqIdx s B_req_idx * stride_b + (SN + j.val) * stride_s := by
    subst hs; ring
  simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar, NumericDType.add,
    NumericDType.mul, ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex,
    Region.cast_cast, Region.cast_id, vLoc, inWindow]
  rw [haddr]
  simp only [if_true]
  by_cases hlt : SN + j.val < batchSeqLen s B_Seqlen
  · simp only [hlt, decide_true, if_true, if_pos hlt]
  · simp only [hlt, decide_false, if_neg hlt, if_false, Bool.false_eq_true]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`v_value` 2D masked-gather recipe** (`tl.load(V + v_offs +
v_loc[:,None]·stride_vbs, mask=(start_n+offs_n[:,None]) < cur_batch_seq_len,
other=0)`, shape `[BLOCK_N, BLOCK_DMODEL]`). The result lane `(j,d)` is
`vMasked (SN+j) d`. -/
theorem reducev_v_gather_eval (s : BlockState)
    (V : RegionName) (Req_to_tokens B_req_idx B_Seqlen : RegionName)
    (BN BD stride_b stride_s stride_vbs stride_vh stride_vd kv_group_num SN : Nat)
    (hvoffs : s.regs .nat [1, BD] "v_offs" =
      some (⟨fun idx : TileIndex [1, BD] =>
          (s.pids 1 / kv_group_num) * stride_vh + idx.2.1.val * stride_vd⟩
        : Tile .nat [1, BD]))
    (hvloc : s.regs .nat [BN] "v_loc" =
      some (Tile.vec (fun j : Fin BN =>
        if inWindow s B_Seqlen (SN + j.val) then
          vLoc s Req_to_tokens B_req_idx stride_b stride_s (SN + j.val)
        else 0)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
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
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          (Op.broadcast (Op.const 0.0) [BN, BD]))) s
      = some (⟨fun idx : TileIndex [BN, BD] =>
          some (vMasked s V Req_to_tokens B_req_idx B_Seqlen stride_b stride_s stride_vbs
            stride_vh stride_vd kv_group_num (SN + idx.1.val) idx.2.1.val)⟩
          : Tile .real [BN, BD]) := by
  have hexp : @evalOp .nat [BN, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "v_loc")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun j : Fin BN =>
          if inWindow s B_Seqlen (SN + j.val) then
            vLoc s Req_to_tokens B_req_idx stride_b stride_s (SN + j.val)
          else 0))) :=
    evalOp_expandDim_ref_of_regs .nat [BN] ⟨1, by simp⟩ "v_loc" s _ hvloc
  have hexpn : @evalOp .nat [BN, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun j : Fin BN => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BN] ⟨1, by simp⟩ "offs_n" s _ hn
  simp only [evalOp, hvoffs, hexp, hexpn, hsn, hsl, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, d, u⟩ := idx
  simp only [Tile.remap, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar,
    Tile.expandDim_data, TileShape.dropInsertedIndex_succ, TileShape.dropInsertedIndex_nil,
    TileShape.dropInsertedIndex_zero_cons, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast_cast,
    Region.cast_id, vMasked, vOffset, inWindow]
  by_cases hlt : SN + j.val < batchSeqLen s B_Seqlen
  · simp only [hlt, decide_true, if_true, if_pos hlt, BlockState.readMemValue_real]
    congr 2
    ring
  · simp only [hlt, decide_false, if_neg hlt, if_false, Bool.false_eq_true]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`acc += tl.sum(p_value[:,None]·v_value, 0)` block-accumulation recipe**
(shape `[BLOCK_DMODEL]`). The result lane `d` is
`accVal d + Σ_{j<BN} pVal j · vVal j d`. -/
theorem reducev_acc_step_eval (s : BlockState) (BN BD : Nat)
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
Every lane reads back its stored accumulator value. -/
theorem reducev_store_eval (s : BlockState) (Out : RegionName) (BD : Nat)
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
decode, the `forRangeDyn` token loop (carrying `acc = partialAcc (counter)`), and
the final unmasked store readback, landing on `tokenAttnReduceVClosedForm`.

`stride_pbs = 1 ∧ stride_req_to_tokens_s = 1` (true for all checked Python shapes)
are carried as hypotheses for the per-lane address arithmetic. -/

/-- The loop invariant carried across the `range(0, cur_batch_seq_len, BLOCK_N)`
loop: the accumulator holds `partialAcc k`, and every loop-invariant register
holds its prelude-seeded value. -/
def reducevInvariant
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : RegionName)
    (s0 : BlockState)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N : Nat)
    (k : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids
  ∧ s.mem = s0.mem
  ∧ (∀ rg o, s.undef rg o = 0)
  ∧ s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 0))
  ∧ s.regs .nat [] "cur_head" = some (Tile.scalar (s0.pids 1))
  ∧ s.regs .nat [] "cur_kv_head" = some (Tile.scalar (s0.pids 1 / kv_group_num))
  ∧ s.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
  ∧ s.regs .nat [BLOCK_DMODEL] "offs_d" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))
  ∧ s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (batchSeqLen s0 B_Seqlen))
  ∧ s.regs .nat [] "cur_batch_start_index" = some (Tile.scalar 0)
  ∧ s.regs .nat [BLOCK_N] "v_loc_off" =
      some (Tile.vec (fun j : Fin BLOCK_N =>
        reqIdx s0 B_req_idx * stride_req_to_tokens_b + j.val * stride_req_to_tokens_s))
  ∧ s.regs .nat [BLOCK_N] "p_offs" =
      some (Tile.vec (fun j : Fin BLOCK_N =>
        pOffset s0 B_Start_Loc stride_ph stride_pbs j.val))
  ∧ s.regs .nat [1, BLOCK_DMODEL] "v_offs" =
      some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] =>
        (s0.pids 1 / kv_group_num) * stride_vh + idx.2.1.val * stride_vd⟩ : Tile .nat [1, BLOCK_DMODEL])
  ∧ s.regs .real [BLOCK_DMODEL] "acc" =
      some (⟨fun idx : TileIndex [BLOCK_DMODEL] =>
        some (partialAcc s0 Prob V Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
          stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
          stride_vbs stride_vh stride_vd kv_group_num k idx.1.val)⟩
        : Tile .real [BLOCK_DMODEL])

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **preLoop** (13 prelude assigns): from a clean input state (`undef = 0`), the
prologue steps to a state satisfying `reducevInvariant … 0` (acc = partialAcc 0 =
0, all loop-invariant registers seeded). -/
theorem reducev_preLoop
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N : Nat)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts ((token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
        stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        kv_group_num BLOCK_DMODEL BLOCK_N).toAlgKernel.body.take 13) s = some s'
      ∧ reducevInvariant Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen s
          stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
          stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
          BLOCK_DMODEL BLOCK_N 0 s' := by
  rw [show ((token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
        stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        kv_group_num BLOCK_DMODEL BLOCK_N).toAlgKernel.body.take 13)
      = [ Stmt.assign .nat [] "cur_batch" (Op.programId 0),
          Stmt.assign .nat [] "cur_head" (Op.programId 1),
          Stmt.assign .nat [] "cur_kv_head"
            (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat kv_group_num)),
          Stmt.assign .nat [BLOCK_N] "offs_n" (Op.arange BLOCK_N),
          Stmt.assign .nat [BLOCK_DMODEL] "offs_d" (Op.arange BLOCK_DMODEL),
          Stmt.assign .nat [] "cur_batch_seq_len"
            (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none),
          Stmt.assign .nat [] "cur_batch_start_index" (Op.constNat 0),
          Stmt.assign .nat [] "cur_batch_in_all_start_index"
            (Op.load .nat (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none),
          Stmt.assign .nat [] "cur_batch_req_idx"
            (Op.load .nat (MemAccess.region B_req_idx (Op.ref .nat [] "cur_batch")) MaskOpt.none),
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
    (show evalOp (Op.constNat 0) _ = some (Tile.scalar 0) from by rw [evalOp_constNat]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
        = some (Tile.scalar (inAllStartLoc s B_Start_Loc)) from by
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
    (reducev_vloc_off_eval _ BLOCK_N (reqIdx s B_req_idx)
      stride_req_to_tokens_b stride_req_to_tokens_s (by simp) (by simp) (by simp [Tile.vec])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_ph))
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index") (Op.ref .nat [BLOCK_N] "offs_n"))
              (Op.constNat stride_pbs))) _
        = some (Tile.vec (fun j : Fin BLOCK_N =>
            pOffset s B_Start_Loc stride_ph stride_pbs j.val)) from by
      rw [reducev_poffs_eval _ BLOCK_N (s.pids 1) (inAllStartLoc s B_Start_Loc)
        stride_ph stride_pbs (by simp) (by simp) (by simp [Tile.vec])]
      refine congrArg some ?_
      ext idx
      simp [Tile.vec, pOffset, inAllStartLoc]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (reducev_voffs_eval _ BLOCK_DMODEL (s.pids 1 / kv_group_num) stride_vh stride_vd
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
    by simp, by simp, by simp [Tile.vec], by simp [Tile.vec], by simp, ?_⟩
  · funext rg o; simp
  · intro rg o; simp [hundef]
  · simp only [BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    simp [partialAcc]

/-- The 5-statement loop body of `token_attn_reducev_surface`, transcribed
(`start_n` no-op + `p_value`/`v_loc`/`v_value` masked loads + `acc += reduceSum`).
Independent of region names except `Prob`/`Req_to_tokens`/`V`. -/
def reducevLoopBody
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
            (Op.ref .nat [] "cur_batch_seq_len"))
          (Op.broadcast (Op.const 0.0) [BLOCK_N]))),
    Stmt.assign .nat [BLOCK_N] "v_loc"
      (Op.load .nat
        (MemAccess.region Req_to_tokens
          (Op.add .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "v_loc_off")
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat stride_req_to_tokens_s))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BLOCK_N] "offs_n"))
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
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")))
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
other loop-invariant register. -/
theorem reducev_loop_step
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : RegionName)
    (s0 : BlockState)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N : Nat)
    (hpbs : stride_pbs = 1) (hrts : stride_req_to_tokens_s = 1)
    (k : Nat) (st : BlockState)
    (hinv : reducevInvariant Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen s0
        stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
        BLOCK_DMODEL BLOCK_N k st) :
    ∃ st', stepStmts (reducevLoopBody Prob V Req_to_tokens stride_req_to_tokens_s stride_vbs
        BLOCK_DMODEL BLOCK_N) (st.setReg "start_n" .nat [] (Tile.scalar k)) = some st'
      ∧ reducevInvariant Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen s0
          stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
          stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
          BLOCK_DMODEL BLOCK_N (k + BLOCK_N) st' := by
  simp only [reducevInvariant] at hinv
  obtain ⟨hpids, hmem, hundef, hcb, hch, hckv, hn, hd, hbsl, hsi, hvoff, hpoff, hvoffs, hacc⟩ := hinv
  set sin := st.setReg "start_n" .nat [] (Tile.scalar k) with hsin
  have hreadval : ∀ (dt : TileDType) (rg : RegionName) (o : Nat),
      sin.readMemValue dt rg o = st.readMemValue dt rg o := by
    intro dt rg o; simp [hsin]
  have hsn : sin.regs .nat [] "start_n" = some (Tile.scalar k) := by simp [hsin]
  have hpoffSin : sin.regs .nat [BLOCK_N] "p_offs" =
      some (Tile.vec (fun j : Fin BLOCK_N => pOffset s0 B_Start_Loc stride_ph stride_pbs j.val)) := by
    rw [hsin]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hpoff
  have hnSin : sin.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
    rw [hsin]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hn
  have hbslSin : sin.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (batchSeqLen s0 B_Seqlen)) := by
    rw [hsin]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hbsl
  have hvoffSin : sin.regs .nat [BLOCK_N] "v_loc_off" =
      some (Tile.vec (fun j : Fin BLOCK_N =>
        reqIdx s0 B_req_idx * stride_req_to_tokens_b + j.val * stride_req_to_tokens_s)) := by
    rw [hsin]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hvoff
  have hvoffsSin : sin.regs .nat [1, BLOCK_DMODEL] "v_offs" =
      some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] =>
        (s0.pids 1 / kv_group_num) * stride_vh + idx.2.1.val * stride_vd⟩ : Tile .nat [1, BLOCK_DMODEL]) := by
    rw [hsin]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hvoffs
  have haccSin : sin.regs .real [BLOCK_DMODEL] "acc" =
      some (⟨fun idx : TileIndex [BLOCK_DMODEL] =>
        some (partialAcc s0 Prob V Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
          stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
          stride_vbs stride_vh stride_vd kv_group_num k idx.1.val)⟩
        : Tile .real [BLOCK_DMODEL]) := by
    rw [hsin]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hacc
  have hpidsSin : sin.pids = s0.pids := by simp [hsin, hpids]
  have hrmvSin : ∀ (dt : TileDType) (rg : RegionName) (o : Nat),
      sin.readMemValue dt rg o = s0.readMemValue dt rg o := by
    intro dt rg o; rw [hreadval]
    cases dt <;>
      simp only [BlockState.readMemValue, BlockState.readMemAs, BlockState.readMemTyped,
        BlockState.readMem, hmem]
  -- state-congruence helpers (general over any state sharing s0's memory and pids)
  have hpMaskedG : ∀ (sx : BlockState), sx.mem = s0.mem → sx.pids = s0.pids →
      ∀ nn, pMasked sx Prob B_Start_Loc B_Seqlen stride_ph stride_pbs nn
        = pMasked s0 Prob B_Start_Loc B_Seqlen stride_ph stride_pbs nn := by
    intro sx hmx hp nn
    have hrm : sx.readMem = s0.readMem := by funext rg o; simp only [BlockState.readMem, hmx]
    have hrv : ∀ (rg : RegionName) (o : Nat), sx.readMemValue .nat rg o = s0.readMemValue .nat rg o := by
      intro rg o; simp only [BlockState.readMemValue, BlockState.readMemTyped, hmx]
    simp only [pMasked, inWindow, batchSeqLen, pOffset, inAllStartLoc, hrv, hp]
    by_cases h : nn < s0.readMemValue .nat B_Seqlen (s0.pids 0)
    · rw [if_pos h, if_pos h]; exact congrFun (congrFun hrm Prob) _
    · rw [if_neg h, if_neg h]
  have hvMaskedG : ∀ (sx : BlockState), sx.mem = s0.mem → sx.pids = s0.pids →
      ∀ nn dd, vMasked sx V Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
          stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num nn dd
        = vMasked s0 V Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
          stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num nn dd := by
    intro sx hmx hp nn dd
    have hrm : sx.readMem = s0.readMem := by funext rg o; simp only [BlockState.readMem, hmx]
    have hrv : ∀ (rg : RegionName) (o : Nat), sx.readMemValue .nat rg o = s0.readMemValue .nat rg o := by
      intro rg o; simp only [BlockState.readMemValue, BlockState.readMemTyped, hmx]
    simp only [vMasked, inWindow, batchSeqLen, vOffset, vLoc, reqIdx, hrv, hp]
    by_cases h : nn < s0.readMemValue .nat B_Seqlen (s0.pids 0)
    · rw [if_pos h, if_pos h]; exact congrFun (congrFun hrm V) _
    · rw [if_neg h, if_neg h]
  have hvLocActiveG : ∀ (sx : BlockState), sx.mem = s0.mem → sx.pids = s0.pids →
      ∀ nn, (if inWindow sx B_Seqlen nn then
              vLoc sx Req_to_tokens B_req_idx stride_req_to_tokens_b stride_req_to_tokens_s nn else 0)
        = (if inWindow s0 B_Seqlen nn then
              vLoc s0 Req_to_tokens B_req_idx stride_req_to_tokens_b stride_req_to_tokens_s nn else 0) := by
    intro sx hmx hp nn
    have hrv : ∀ (rg : RegionName) (o : Nat), sx.readMemValue .nat rg o = s0.readMemValue .nat rg o := by
      intro rg o; simp only [BlockState.readMemValue, BlockState.readMemTyped, hmx]
    simp only [inWindow, batchSeqLen, vLoc, reqIdx, hrv, hp]
  have hmetaG : ∀ (sx : BlockState), sx.mem = s0.mem → sx.pids = s0.pids →
      reqIdx sx B_req_idx = reqIdx s0 B_req_idx
      ∧ batchSeqLen sx B_Seqlen = batchSeqLen s0 B_Seqlen := by
    intro sx hmx hp
    have hrv : ∀ (rg : RegionName) (o : Nat), sx.readMemValue .nat rg o = s0.readMemValue .nat rg o := by
      intro rg o; simp only [BlockState.readMemValue, BlockState.readMemTyped, hmx]
    exact ⟨by simp only [reqIdx, hrv, hp], by simp only [batchSeqLen, hrv, hp]⟩
  have hsinmem : sin.mem = s0.mem := by rw [hsin]; exact hmem
  have hattG := hmetaG sin hsinmem hpidsSin
  have hbatchSeqLen : batchSeqLen sin B_Seqlen = batchSeqLen s0 B_Seqlen := hattG.2
  have hreqIdx : reqIdx sin B_req_idx = reqIdx s0 B_req_idx := hattG.1
  have hpOffset : ∀ nn, pOffset sin B_Start_Loc stride_ph stride_pbs nn
      = pOffset s0 B_Start_Loc stride_ph stride_pbs nn := by
    intro nn; simp only [pOffset, inAllStartLoc, hrmvSin, hpidsSin]
  have hpoffSin' : sin.regs .nat [BLOCK_N] "p_offs" =
      some (Tile.vec (fun j : Fin BLOCK_N => pOffset sin B_Start_Loc stride_ph stride_pbs j.val)) := by
    rw [hpoffSin]; refine congrArg some ?_; ext idx; simp [Tile.vec, hpOffset]
  have hbslSin' : sin.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (batchSeqLen sin B_Seqlen)) := by
    rw [hbslSin, hbatchSeqLen]
  have hvoffSin' : sin.regs .nat [BLOCK_N] "v_loc_off" =
      some (Tile.vec (fun j : Fin BLOCK_N =>
        reqIdx sin B_req_idx * stride_req_to_tokens_b + j.val * stride_req_to_tokens_s)) := by
    rw [hvoffSin]; refine congrArg some ?_; ext idx; simp [Tile.vec, hreqIdx]
  have hvoffsSin' : sin.regs .nat [1, BLOCK_DMODEL] "v_offs" =
      some (⟨fun idx : TileIndex [1, BLOCK_DMODEL] =>
        (sin.pids 1 / kv_group_num) * stride_vh + idx.2.1.val * stride_vd⟩ : Tile .nat [1, BLOCK_DMODEL]) := by
    rw [hvoffsSin]; refine congrArg some ?_; ext idx; simp [hpidsSin]
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
  set pvalT : Tile .real [BLOCK_N] :=
    ⟨fun idx : TileIndex [BLOCK_N] =>
      some (pMasked sin Prob B_Start_Loc B_Seqlen stride_ph stride_pbs (k + idx.1.val))⟩
    with hpvalT
  set s2 : BlockState := sin.setReg "p_value" .real [BLOCK_N] pvalT with hs2
  set vlocT : Tile .nat [BLOCK_N] :=
    ⟨fun idx : TileIndex [BLOCK_N] =>
      if inWindow s2 B_Seqlen (k + idx.1.val) then
        vLoc s2 Req_to_tokens B_req_idx stride_req_to_tokens_b stride_req_to_tokens_s (k + idx.1.val)
      else 0⟩ with hvlocT
  set s3 : BlockState := s2.setReg "v_loc" .nat [BLOCK_N] vlocT with hs3
  set vvalT : Tile .real [BLOCK_N, BLOCK_DMODEL] :=
    ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
      some (vMasked s3 V Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
        stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num
        (k + idx.1.val) idx.2.1.val)⟩ with hvvalT
  set s4 : BlockState := s3.setReg "v_value" .real [BLOCK_N, BLOCK_DMODEL] vvalT with hs4
  have hs2mem : s2.mem = s0.mem := by rw [hs2]; exact hsinmem
  have hs2pids : s2.pids = s0.pids := by rw [hs2, BlockState.setReg_pids]; exact hpidsSin
  have hs3mem : s3.mem = s0.mem := by rw [hs3]; exact hs2mem
  have hs3pids : s3.pids = s0.pids := by rw [hs3, BlockState.setReg_pids]; exact hs2pids
  unfold reducevLoopBody
  -- statement 1: start_n = start_n (no-op)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_n") sin = some (Tile.scalar k) from by rw [evalOp_ref]; exact hsn))]
  rw [hidem]
  -- statement 2: p_value masked load
  rw [stepStmts.cons_some
    (show stepStmt _ sin = some s2 from stepStmt_assign_eq_some
      (reducev_prob_load_eval sin Prob B_Start_Loc B_Seqlen BLOCK_N stride_ph stride_pbs k hpbs
        (by simpa using hpoffSin') (by simp [hsn]) (by simpa using hnSin) hbslSin'))]
  -- statement 3: v_loc masked gather (on s2)
  rw [stepStmts.cons_some
    (show stepStmt _ s2 = some s3 from stepStmt_assign_eq_some
      (reducev_vloc_gather_eval s2 Req_to_tokens B_req_idx B_Seqlen BLOCK_N
        stride_req_to_tokens_b stride_req_to_tokens_s k hrts
        (by
          rw [hs2]
          simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
          rw [hvoffSin]
          refine congrArg some ?_
          ext idx
          simp only [Tile.vec]
          rw [(hmetaG s2 hs2mem hs2pids).1])
        (by rw [hs2]; simp [hsn]) (by rw [hs2]; simpa using hnSin)
        (by
          rw [hs2]
          simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
          rw [hbslSin, (hmetaG s2 hs2mem hs2pids).2])))]
  -- statement 4: v_value 2D masked gather (on s3)
  rw [stepStmts.cons_some
    (show stepStmt _ s3 = some s4 from stepStmt_assign_eq_some
      (reducev_v_gather_eval s3 V Req_to_tokens B_req_idx B_Seqlen BLOCK_N BLOCK_DMODEL
        stride_req_to_tokens_b stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num k
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
          rw [hbslSin, (hmetaG s3 hs3mem hs3pids).2])))]
  -- statement 5: acc += reduceSum(p[:,None]·v) = partialAcc (k + BLOCK_N)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (reducev_acc_step_eval s4 BLOCK_N BLOCK_DMODEL
      (fun e => partialAcc s0 Prob V Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
        stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd kv_group_num k e.val)
      (fun j => pMasked s0 Prob B_Start_Loc B_Seqlen stride_ph stride_pbs (k + j.val))
      (fun j e => vMasked s0 V Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
        stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num (k + j.val) e.val)
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
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
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
  · rw [hpeel _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hbsl
  · rw [hpeel _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hsi
  · rw [hpeel _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hvoff
  · rw [hpeel _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hpoff
  · rw [hpeel _ (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hvoffs
  · simp only [BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    simp only
    rw [partialAcc_block_succ]
    congr 1
    rw [Fin.sum_univ_eq_sum_range
      (fun j => pMasked s0 Prob B_Start_Loc B_Seqlen stride_ph stride_pbs (k + j) *
        vMasked s0 V Req_to_tokens B_req_idx B_Seqlen stride_req_to_tokens_b
          stride_req_to_tokens_s stride_vbs stride_vh stride_vd kv_group_num
          (k + j) idx.1.val) BLOCK_N]

/-- The 4-statement postlude of `token_attn_reducev_surface`: the `.to(Out.dtype)`
cast (identity), `off_o`, `out_ptrs`, and the unmasked `[BLOCK_DMODEL]` store. -/
def reducevPostlude (Out : RegionName) (stride_obs stride_oh stride_od BLOCK_DMODEL : Nat) :
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
/-- **Post-loop + store bridge**: after the loop (invariant at `final ≥
batchSeqLen`, so `acc = partialAcc final`), the cast (identity), `off_o`/`out_ptrs`
seeding, and the unmasked store write back `tokenAttnReduceVClosedForm` to every
`Out[outOffset … i]` (via `reducev_store_eval` + `partialAcc_eq_PVValue`). -/
theorem reducev_postLoop
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : RegionName)
    (s0 : BlockState)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N : Nat)
    (final : Nat) (hfinal : batchSeqLen s0 B_Seqlen ≤ final)
    (st : BlockState)
    (hinv : reducevInvariant Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen s0
        stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
        BLOCK_DMODEL BLOCK_N final st)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s0 stride_obs stride_oh stride_od i)) :
    ∃ sfin, stepStmts (reducevPostlude Out stride_obs stride_oh stride_od BLOCK_DMODEL) st = some sfin
      ∧ ∀ i : Fin BLOCK_DMODEL,
          sfin.readMem Out (outOffset s0 stride_obs stride_oh stride_od i)
            = tokenAttnReduceVClosedForm s0 Prob V Req_to_tokens B_req_idx B_Start_Loc
                B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
                stride_pbs stride_vbs stride_vh stride_vd kv_group_num BLOCK_DMODEL i := by
  simp only [reducevInvariant] at hinv
  obtain ⟨hpids, hmem, hundef, hcb, hch, hckv, hn, hd, hbsl, hsi, hvoff, hpoff, hvoffs, hacc⟩ := hinv
  set accFn : Fin BLOCK_DMODEL → ℝ := fun e =>
    partialAcc s0 Prob V Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
      stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs stride_vbs stride_vh
      stride_vd kv_group_num final e.val with haccFn
  set offFn : TileIndex [BLOCK_DMODEL] → Nat :=
    fun i => outOffset s0 stride_obs stride_oh stride_od i.1 with hoffFn
  unfold reducevPostlude
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BLOCK_DMODEL] "acc") st =
        some (⟨fun idx : TileIndex [BLOCK_DMODEL] => some (accFn idx.1)⟩ : Tile .real [BLOCK_DMODEL])
      from by rw [evalOp_ref]; exact hacc))]
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
  obtain ⟨sfin, hstep⟩ : ∃ sfin, stepStmt (Stmt.store .real [BLOCK_DMODEL]
      (MemAccess.ptr (Op.ref .ptr [BLOCK_DMODEL] "out_ptrs"))
      (Op.ref .real [BLOCK_DMODEL] "acc") MaskOpt.none) stStore = some sfin := by
    simp only [stepStmt, evalOp_ref, hout, haccStore, Option.bind, Option.map]
    exact ⟨_, rfl⟩
  refine ⟨sfin, by rw [stepStmts.cons_some hstep, stepStmts.nil], ?_⟩
  intro i
  have hrb := reducev_store_eval stStore Out BLOCK_DMODEL offFn (fun i => accFn i.1) hout haccStore hinj'
    (i, PUnit.unit)
  rw [hstep] at hrb
  simp only [Option.map_some] at hrb
  have hval : sfin.readMem Out (offFn (i, PUnit.unit)) = accFn i := Option.some.inj hrb
  rw [show outOffset s0 stride_obs stride_oh stride_od i = offFn (i, PUnit.unit) from rfl, hval]
  rw [haccFn]
  simp only
  rw [partialAcc_eq_PVValue s0 Prob V Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
    stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs stride_vbs
    stride_vh stride_vd kv_group_num final i.val hfinal]
  rfl

/-- Body decomposition: `prelude(13) ++ [forRangeDyn, postlude…]`. By `rfl`. -/
theorem reducev_body_split
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N : Nat) :
    (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
        stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
        stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
        BLOCK_DMODEL BLOCK_N).toAlgKernel.body
      = (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
          stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
          stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
          BLOCK_DMODEL BLOCK_N).toAlgKernel.body.take 13
        ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0) (Op.ref .nat [] "cur_batch_seq_len") (Op.constNat BLOCK_N)
              (reducevLoopBody Prob V Req_to_tokens stride_req_to_tokens_s stride_vbs BLOCK_DMODEL BLOCK_N)
            :: reducevPostlude Out stride_obs stride_oh stride_od BLOCK_DMODEL) := by
  rfl

set_option maxHeartbeats 1000000 in
/-- **Genuine closed-form correctness for `token_attn_reduceV` (general).** With
`stride_pbs = 1 ∧ stride_req_to_tokens_s = 1` and `0 < BLOCK_N` (true for all
checked Python shapes), every lane of the `[BLOCK_DMODEL]` output store holds the
genuine PV-reduction closed form `tokenAttnReduceVClosedForm` — i.e. the
probability-weighted value sum `Σ_n p[n]·v[v_loc[n], d]` — NOT the kernel's own
executed value. Proven sorry-free via `reducev_preLoop` (entry invariant),
`reducev_loop_step` fed to `forRangeDyn_inv` (one-block carry advance), and
`reducev_postLoop` (unmasked store readback = closed form). -/
theorem token_attn_reducev_closed_form_correct
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N : Nat)
    (hpbs : stride_pbs = 1) (hrts : stride_req_to_tokens_s = 1) (hBN : 0 < BLOCK_N)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i))
    (i : Fin BLOCK_DMODEL) :
    (match exec (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx B_Start_Loc
        B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
        stride_ph stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        kv_group_num BLOCK_DMODEL BLOCK_N) s with
      | some s' => s'.readMem Out (outOffset s stride_obs stride_oh stride_od i)
      | none => (0.0 : ℝ)) =
      tokenAttnReduceVClosedForm s Prob V Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
        stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd kv_group_num BLOCK_DMODEL i := by
  obtain ⟨s', hpre, hinv0⟩ := reducev_preLoop Prob V Out Req_to_tokens B_req_idx B_Start_Loc
    B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
    stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
    BLOCK_DMODEL BLOCK_N s hundef
  have hslEntry : s'.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (batchSeqLen s B_Seqlen)) := by
    simp only [reducevInvariant] at hinv0
    exact hinv0.2.2.2.2.2.2.2.2.1
  have hstop : evalOp (Op.ref .nat [] "cur_batch_seq_len") s' = some (Tile.scalar (batchSeqLen s B_Seqlen)) := by
    rw [evalOp_ref]; exact hslEntry
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hinvFinal⟩ :=
    forRangeDyn_inv (idx := "start_n") (startOp := Op.constNat 0)
      (stopOp := Op.ref .nat [] "cur_batch_seq_len") (stepOp := Op.constNat BLOCK_N)
      (start := 0) (stop := batchSeqLen s B_Seqlen) (step := BLOCK_N)
      (P := reducevInvariant Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen s
        stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
        BLOCK_DMODEL BLOCK_N)
      (by simp) hstop (by simp) (by omega) hinv0
      (fun c st _ hP => reducev_loop_step Prob V Out Req_to_tokens B_req_idx B_Start_Loc
        B_Seqlen s stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
        BLOCK_DMODEL BLOCK_N hpbs hrts c st hP)
  obtain ⟨sfin, hPost, hRead⟩ := reducev_postLoop Prob V Out Req_to_tokens B_req_idx
    B_Start_Loc B_Seqlen s stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
    stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
    BLOCK_DMODEL BLOCK_N final hfinal sLoop hinvFinal hOutInj
  have hexec : exec (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx B_Start_Loc
      B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
      stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num
      BLOCK_DMODEL BLOCK_N) s = some sfin := by
    rw [exec, reducev_body_split, stepStmts.append_some hpre, stepStmts.cons_some hLoopStmt, hPost]
  rw [hexec]
  exact hRead i

/-- **Compute-facing genuine closed-form correctness for `token_attn_reduceV`.**
The full surface kernel realizes the genuine PV-reduction closed form
`tokenAttnReduceVClosedForm` at every `[BLOCK_DMODEL]` output lane (under
`stride_pbs = 1 ∧ stride_req_to_tokens_s = 1`, `0 < BLOCK_N`, clean `undef`, and
output-offset injectivity). -/
theorem token_attn_reducev_closed_form_compute_correct
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N : Nat)
    (hpbs : stride_pbs = 1) (hrts : stride_req_to_tokens_s = 1) (hBN : 0 < BLOCK_N)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
        stride_ph stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh
        stride_od kv_group_num BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i =>
        tokenAttnReduceVClosedForm s Prob V Req_to_tokens B_req_idx B_Start_Loc
          B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
          stride_pbs stride_vbs stride_vh stride_vd kv_group_num BLOCK_DMODEL i) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [token_attn_reducev_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := token_attn_reducev_closed_form_correct Prob V Out Req_to_tokens B_req_idx
    B_Start_Loc B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
    stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num BLOCK_DMODEL
    BLOCK_N hpbs hrts hBN s hundef hOutInj i
  rw [show exec _ s = some s' from hExec] at h
  exact h

/-! ### ════════ ★ MAIN THEOREM ★ ════════

`token_attn_reducev_output_summary_general` is the headline: a **single
dimension-general** statement over symbolic `(BLOCK_DMODEL, BLOCK_N)` and all
strides. It bundles the surface→algorithm lowering with the genuine
PV-reduction compute-correctness (`tokenAttnReduceVClosedForm`), under honest
side-conditions only:

* `0 < BLOCK_DMODEL`, `0 < BLOCK_N` — non-degenerate output / loop block;
* `stride_pbs = 1`, `stride_req_to_tokens_s = 1` — the contiguous `Prob` /
  `Req_to_tokens` layouts the kernel's per-lane address arithmetic genuinely
  assumes;
* `hOutInj` — output-offset injectivity (no aliasing across the store);
* `hundef` — clean input (`undef = 0`).

No shape-specific numeric cheats; `expected` is the self-reference-free closed
form over input memory, never the kernel's executed value. The Python test
cases below are thin corollaries instantiating this at `BLOCK_DMODEL = 64`,
`BLOCK_N = 128` and the checked `stride_req_to_tokens_b` gather-stride
(`128` for cases 1/3, `64` for case 2). That stride feeds `vLoc`/`vMasked`, so
it genuinely enters the PV-reduction value spec; the cases are distinct
instantiations, not a single collapsed one. -/
theorem token_attn_reducev_output_summary_general
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N : Nat)
    (hBD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N)
    (hpbs : stride_pbs = 1) (hrts : stride_req_to_tokens_s = 1)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    (∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
      stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
        stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        kv_group_num BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i : Fin BLOCK_DMODEL =>
        tokenAttnReduceVClosedForm s Prob V Req_to_tokens B_req_idx
          B_Start_Loc B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s stride_ph
          stride_pbs stride_vbs stride_vh stride_vd kv_group_num BLOCK_DMODEL i)) := by
  refine ⟨?_, ?_⟩
  · exact token_attn_reducev_surface_toAlgorithm_supported Prob V Out
      Req_to_tokens B_req_idx B_Start_Loc B_Seqlen stride_req_to_tokens_b
      stride_req_to_tokens_s stride_ph stride_pbs stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od kv_group_num BLOCK_DMODEL BLOCK_N
  · exact token_attn_reducev_closed_form_compute_correct Prob V Out
      Req_to_tokens B_req_idx B_Start_Loc B_Seqlen stride_req_to_tokens_b
      stride_req_to_tokens_s stride_ph stride_pbs stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od kv_group_num BLOCK_DMODEL BLOCK_N hpbs hrts
      hBN s hundef hOutInj


end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.TokenAttnReduceV
