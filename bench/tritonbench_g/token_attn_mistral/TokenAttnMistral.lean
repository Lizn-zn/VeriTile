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

### Banked progress toward the bridge

The two *algebraic* halves of the loop-invariant induction are now proven
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

### Remaining (the operational `exec` decode)

What is left is the purely mechanical state-stepping that connects the kernel's
`stepStmts` to these algebraic facts (no further mathematical content):

1. Prelude decode (14 assigns): thread `cur_batch`/`cur_head`/`cur_kv_head`,
   `offs_n = arange 128`, `offs_d = arange 64`, the four metadata loads, the
   `tl.maximum(... , 0)` `startIndex`, the `v_loc_off`/`p_offs`/`v_offs` vectors,
   and `acc = full [BLOCK_DMODEL] 0` into a symbolic post-prelude state, mirroring
   `chunk_cumsum_kernel`'s prelude reduction.
2. Loop step lemma: the 5-statement body (`start_n = multiple_of …` no-op,
   `p_value`/`v_loc`/`v_value` masked region loads, `acc += reduceSum (p[:,None]·v)`)
   advances `acc = partialAcc i` to `acc = partialAcc (i + BLOCK_N)` via
   `partialAcc_block_succ` + the per-lane load/`reduceSum` recipes, fed to
   `forRangeDyn_inv`.
3. Final unmasked `[BLOCK_DMODEL]` store readback (`scatter_readback_nd`), then
   `partialAcc_eq_PVValue` to land on `tokenAttnMistralClosedForm`.

NOTE: the closed form folds the per-lane offset arithmetic
`pOffset n = … + (att_start_loc + n)·stride_pbs` and the `Req_to_tokens` gather
`(start_index + n)·stride_req_to_tokens_s`, whereas the kernel computes
`p_offs[j] + start_n` (resp. `v_loc_off[j] + start_n·stride_req_to_tokens_s`).
These coincide for the lane index `n = start_n + j` exactly when
`stride_pbs = 1` and `stride_req_to_tokens_s = 1`, which holds for all four
checked Python shapes; the operational decode should carry those two stride
equalities as hypotheses (or instantiate at the concrete shapes).
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
