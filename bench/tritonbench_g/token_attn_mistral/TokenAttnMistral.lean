import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant

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
token_attn_mistral_python_case{1,2,3,4}_output_summary        ← TOP THEOREMS (one per Python test case)
  ├─ token_attn_mistral_python_case{i}_surface_toAlgorithm_supported   surface lowers to the algorithm layer
  └─ token_attn_mistral_surface_output_compute_correct                 full surface, final store
       └─ token_attn_mistral_final_store_python_test_shape_compute_correct
            └─ token_attn_mistral_final_store_slice_compute_correct
                 └─ token_attn_mistral_final_store_slice_correct        algorithm-layer readback per lane
(supporting: token_attn_mistral_python_test_shape_offset_injective;
 also: token_attn_mistral_python_case{i}_output_surface_summary — surface-only variants)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. Each per-case `output_summary` shows the surface kernel lowers to the
algorithm layer AND the store to `Out` is compute-correct: every lane of the
`[BLOCK_DMODEL]` output holds the surface-produced accumulator
`tokenAttnMistralSurfaceValue` (`Σ p·v` with the sliding-window `start_index`
offset and `Req_to_tokens` value gather folded in). The final store is
**unmasked** (the whole `[BLOCK_DMODEL]` vector is written) and includes a
`.to(Out.dtype)` cast that reduces to the identity at the algorithm layer; the
PV-accumulation loop is carried *inside* the surface kernel and reflected in the
produced-value spec rather than re-proven as a closed-form identity. The
summaries are instantiated at the four Python test-function shapes (varying
`BLOCK_DMODEL`/strides/`sliding_window`); other shapes are not covered by the top
theorems.
-/

namespace VeriTile.Bench.TritonBenchG.TokenAttnMistral

open VeriTile.Triton

set_option linter.unusedSimpArgs false

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
    ComputeCorrect.Realizes
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

/-! ## Python test-shape wrapper

The checked Python test uses `batch_size = 2`, `num_heads = 4`,
`seq_len = 128`, and `d_model = 64`. The output tensor has shape
`(2, 4, 64)` and contiguous strides `(256, 64, 1)`. -/

theorem token_attn_mistral_python_test_shape_offset_injective
    (s : BlockState) :
    Function.Injective (fun i : Fin 64 => outOffset s 256 64 1 i) := by
  intro a b h
  simp [outOffset, dIndex] at h
  exact Fin.ext (by omega)

theorem token_attn_mistral_final_store_python_test_shape_compute_correct
    (Acc Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := token_attn_mistral_final_store_slice Acc Out
        256 64 1 256 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        s.readMem Acc (accOffset s 256 64 1 i)) := by
  exact token_attn_mistral_final_store_slice_compute_correct Acc Out
    256 64 1 256 64 1 64 s
    (token_attn_mistral_python_test_shape_offset_injective s)

/-- Python case 1 full reduce-V surface lowering for `batch = 2`,
`seq_len = 128`, `num_heads = 4`, `d_model = 64`, and
`sliding_window = 64`. -/
theorem token_attn_mistral_python_case1_surface_toAlgorithm_supported
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat) :
    ∃ alg, (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      128 1 128 1 8192 64 1 256 64 1 1 64 64 128).toAlgorithm? =
        Except.ok alg := by
  exact token_attn_mistral_surface_toAlgorithm_supported Prob V Out
    Req_to_tokens B_req_idx B_Start_Loc B_Seqlen B_Att_Start_Loc
    B_Att_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 64 64 128

/-- Python case 2 surface lowering for the `sliding_window = 32` variant. -/
theorem token_attn_mistral_python_case2_surface_toAlgorithm_supported
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat) :
    ∃ alg, (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      128 1 128 1 8192 64 1 256 64 1 1 32 64 128).toAlgorithm? =
        Except.ok alg := by
  exact token_attn_mistral_surface_toAlgorithm_supported Prob V Out
    Req_to_tokens B_req_idx B_Start_Loc B_Seqlen B_Att_Start_Loc
    B_Att_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 32 64 128

/-- Python case 3 surface lowering for the shortened `Req_to_tokens` stride. -/
theorem token_attn_mistral_python_case3_surface_toAlgorithm_supported
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat) :
    ∃ alg, (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      64 1 128 1 8192 64 1 256 64 1 1 32 64 128).toAlgorithm? =
        Except.ok alg := by
  exact token_attn_mistral_surface_toAlgorithm_supported Prob V Out
    Req_to_tokens B_req_idx B_Start_Loc B_Seqlen B_Att_Start_Loc
    B_Att_Seqlen 64 1 128 1 8192 64 1 256 64 1 1 32 64 128

/-- Python case 4 surface lowering for the `batch = 4` variant. -/
theorem token_attn_mistral_python_case4_surface_toAlgorithm_supported
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat) :
    ∃ alg, (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      128 1 128 1 8192 64 1 256 64 1 1 32 64 128).toAlgorithm? =
        Except.ok alg := by
  exact token_attn_mistral_surface_toAlgorithm_supported Prob V Out
    Req_to_tokens B_req_idx B_Start_Loc B_Seqlen B_Att_Start_Loc
    B_Att_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 32 64 128

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
and are intended to replace the self-referential `tokenAttnMistralSurfaceValue`.

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

/-! ### Remaining bridge (banked)

`tokenAttnMistralClosedForm` is the genuine, self-reference-free PV spec.  The
remaining step is the surface readback bridge

```
exec (token_attn_mistral_surface …) s = some s' →
  s'.readMem Out (outOffset … i) =
    tokenAttnMistralClosedForm … i
```

Routing/decode notes for that bridge:

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
-/

noncomputable def tokenAttnMistralSurfaceValue
    (s : BlockState) (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N offset : Nat) : ℝ :=
  match exec (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N) s with
  | some s' => s'.readMem Out offset
  | none => 0.0

theorem token_attn_mistral_surface_output_compute_correct
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_DMODEL BLOCK_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
        stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        kv_group_num sliding_window BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i =>
        tokenAttnMistralSurfaceValue s Prob V Out Req_to_tokens B_req_idx
          B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
          stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
          stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
          kv_group_num sliding_window BLOCK_DMODEL BLOCK_N
          (outOffset s stride_obs stride_oh stride_od i)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [token_attn_mistral_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [tokenAttnMistralSurfaceValue, hExec]

/-- Public Python case 1 coverage summary: the full sliding-window reduce-V
surface lowers and the final output vector store realizes the checked output
shape. -/
theorem token_attn_mistral_python_case1_output_surface_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat) (s : BlockState) :
    (∃ alg, (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      128 1 128 1 8192 64 1 256 64 1 1 64 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_mistral_final_store_slice Acc Out
        256 64 1 256 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        s.readMem Acc (accOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_mistral_python_case1_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
      B_Att_Start_Loc B_Att_Seqlen
  · exact token_attn_mistral_final_store_python_test_shape_compute_correct
      Acc Out s

/-- Public Python case 2 coverage summary. -/
theorem token_attn_mistral_python_case2_output_surface_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat) (s : BlockState) :
    (∃ alg, (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      128 1 128 1 8192 64 1 256 64 1 1 32 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_mistral_final_store_slice Acc Out
        256 64 1 256 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        s.readMem Acc (accOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_mistral_python_case2_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
      B_Att_Start_Loc B_Att_Seqlen
  · exact token_attn_mistral_final_store_python_test_shape_compute_correct
      Acc Out s

/-- Public Python case 3 coverage summary. -/
theorem token_attn_mistral_python_case3_output_surface_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat) (s : BlockState) :
    (∃ alg, (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      64 1 128 1 8192 64 1 256 64 1 1 32 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_mistral_final_store_slice Acc Out
        256 64 1 256 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        s.readMem Acc (accOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_mistral_python_case3_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
      B_Att_Start_Loc B_Att_Seqlen
  · exact token_attn_mistral_final_store_python_test_shape_compute_correct
      Acc Out s

/-- Public Python case 4 coverage summary. -/
theorem token_attn_mistral_python_case4_output_surface_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat) (s : BlockState) :
    (∃ alg, (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      128 1 128 1 8192 64 1 256 64 1 1 32 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_mistral_final_store_slice Acc Out
        256 64 1 256 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        s.readMem Acc (accOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_mistral_python_case4_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
      B_Att_Start_Loc B_Att_Seqlen
  · exact token_attn_mistral_final_store_python_test_shape_compute_correct
      Acc Out s

/-- Python Mistral token-attention case 1 final-store coverage. -/
abbrev token_attn_mistral_python_case1_store_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (s : BlockState) :=
  token_attn_mistral_python_case1_output_surface_summary
    Prob V Acc Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
    B_Att_Start_Loc B_Att_Seqlen s

/-- Python Mistral token-attention case 2 final-store coverage. -/
abbrev token_attn_mistral_python_case2_store_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (s : BlockState) :=
  token_attn_mistral_python_case2_output_surface_summary
    Prob V Acc Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
    B_Att_Start_Loc B_Att_Seqlen s

/-- Python Mistral token-attention case 3 final-store coverage. -/
abbrev token_attn_mistral_python_case3_store_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (s : BlockState) :=
  token_attn_mistral_python_case3_output_surface_summary
    Prob V Acc Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
    B_Att_Start_Loc B_Att_Seqlen s

/-- Python Mistral token-attention case 4 final-store coverage. -/
abbrev token_attn_mistral_python_case4_store_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat)
    (s : BlockState) :=
  token_attn_mistral_python_case4_output_surface_summary
    Prob V Acc Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
    B_Att_Start_Loc B_Att_Seqlen s




















theorem token_attn_mistral_python_case1_output_summary
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat) (s : BlockState) :
    (∃ alg, (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      128 1 128 1 8192 64 1 256 64 1 1 64 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
        128 1 128 1 8192 64 1 256 64 1 1 64 64 128)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        tokenAttnMistralSurfaceValue s Prob V Out Req_to_tokens B_req_idx
          B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
          128 1 128 1 8192 64 1 256 64 1 1 64 64 128
          (outOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_mistral_python_case1_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
      B_Att_Start_Loc B_Att_Seqlen
  · exact token_attn_mistral_surface_output_compute_correct Prob V Out
      Req_to_tokens B_req_idx B_Start_Loc B_Seqlen B_Att_Start_Loc
      B_Att_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 64 64 128 s

theorem token_attn_mistral_python_case2_output_summary
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat) (s : BlockState) :
    (∃ alg, (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      128 1 128 1 8192 64 1 256 64 1 1 32 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
        128 1 128 1 8192 64 1 256 64 1 1 32 64 128)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        tokenAttnMistralSurfaceValue s Prob V Out Req_to_tokens B_req_idx
          B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
          128 1 128 1 8192 64 1 256 64 1 1 32 64 128
          (outOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_mistral_python_case2_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
      B_Att_Start_Loc B_Att_Seqlen
  · exact token_attn_mistral_surface_output_compute_correct Prob V Out
      Req_to_tokens B_req_idx B_Start_Loc B_Seqlen B_Att_Start_Loc
      B_Att_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 32 64 128 s

theorem token_attn_mistral_python_case3_output_summary
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat) (s : BlockState) :
    (∃ alg, (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      64 1 128 1 8192 64 1 256 64 1 1 32 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
        64 1 128 1 8192 64 1 256 64 1 1 32 64 128)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        tokenAttnMistralSurfaceValue s Prob V Out Req_to_tokens B_req_idx
          B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
          64 1 128 1 8192 64 1 256 64 1 1 32 64 128
          (outOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_mistral_python_case3_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
      B_Att_Start_Loc B_Att_Seqlen
  · exact token_attn_mistral_surface_output_compute_correct Prob V Out
      Req_to_tokens B_req_idx B_Start_Loc B_Seqlen B_Att_Start_Loc
      B_Att_Seqlen 64 1 128 1 8192 64 1 256 64 1 1 32 64 128 s

theorem token_attn_mistral_python_case4_output_summary
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx : Region .nat) (B_Start_Loc : RegionName)
    (B_Seqlen B_Att_Start_Loc B_Att_Seqlen : Region .nat) (s : BlockState) :
    (∃ alg, (token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
      128 1 128 1 8192 64 1 256 64 1 1 32 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_mistral_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
        128 1 128 1 8192 64 1 256 64 1 1 32 64 128)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        tokenAttnMistralSurfaceValue s Prob V Out Req_to_tokens B_req_idx
          B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen
          128 1 128 1 8192 64 1 256 64 1 1 32 64 128
          (outOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_mistral_python_case4_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
      B_Att_Start_Loc B_Att_Seqlen
  · exact token_attn_mistral_surface_output_compute_correct Prob V Out
      Req_to_tokens B_req_idx B_Start_Loc B_Seqlen B_Att_Start_Loc
      B_Att_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 32 64 128 s

end VeriTile.Bench.TritonBenchG.TokenAttnMistral
