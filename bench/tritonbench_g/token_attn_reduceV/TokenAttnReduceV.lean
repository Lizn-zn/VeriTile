import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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
token_attn_reducev_python_case1_output_summary        ← TOP THEOREM (also case2/case3)
  ├─ token_attn_reducev_python_caseN_surface_toAlgorithm_supported  surface lowers
  │    └─ token_attn_reducev_surface_toAlgorithm_supported
  └─ token_attn_reducev_surface_output_compute_correct  ← ComputeCorrect of the store
       ├─ token_attn_reducev_python_test_shape_offset_injective
       └─ token_attn_reducev_final_store_python_test_shape_compute_correct
            └─ token_attn_reducev_final_store_slice_compute_correct
                 └─ token_attn_reducev_final_store_slice_correct  (per-lane readback)
```

The three `output_summary` theorems pin the kernel at the Python test shapes
(`BLOCK_DMODEL = 64`, `BLOCK_N = 128`/`64`).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` /
`num_warps` / `num_stages` are not modeled. The `acc.to(Out.dtype.element_ty)`
cast reduces to the identity at the algorithm layer (post-erasure all dtypes
unify to `ℝ`). The verified statement is scoped to the **final reduced-vector
store** to `Out`: the expected value is the surface-level `tokenAttnReduceVSurfaceValue`
read off at each `outOffset`, established compute-correct for the universally
quantified program ids over a one-block output footprint. The streaming
gather/accumulate loop body feeds that value; the side conditions are the
offset-injectivity of the `Out` slice at the test shapes.

## Genuine closed-form spec (this file also provides)

`tokenAttnReduceVClosedForm` / `tokenAttnReduceVPVValue` are a *genuine*,
self-reference-free PV-reduction closed form
`acc[d] = Σ_{n < cur_batch_seq_len} p[n]·v[v_loc[n], d]` (with the
`Req_to_tokens` page-table gather and the `cur_kv_head` / `in_all_start_index`
offsets decoded as `pOffset`/`vLoc`/`vOffset`). These never execute the kernel.
The surface→closed-form bridge — driving `forRangeDyn_inv` with the partial-sum
accumulator invariant and per-block `reduceSum_some` — is documented as a banked
recipe next to the definitions; the published top theorems still pin the
self-referential `tokenAttnReduceVSurfaceValue` while that bridge is completed.
The blocker is the data-dependent `V` gather (statement 4 of the loop body loads
`V` at an address containing the *loaded* `v_loc` tile), which needs a
region-load-with-tile-offset evaluation lemma not yet in the core.
-/

namespace VeriTile.Bench.TritonBenchG.TokenAttnReduceV

open VeriTile.Triton

set_option linter.unusedSimpArgs false

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
    ComputeCorrect.Realizes
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

/-! ## Python test-shape wrapper

The checked Python test uses `batch_size = 2`, `num_heads = 4`,
`seq_len = 128`, and `d_model = 64`. The output tensor has shape
`(2, 4, 64)` and contiguous strides `(256, 64, 1)`. -/

theorem token_attn_reducev_python_test_shape_offset_injective
    (s : BlockState) :
    Function.Injective (fun i : Fin 64 => outOffset s 256 64 1 i) := by
  intro a b h
  simp [outOffset, dIndex] at h
  exact Fin.ext (by omega)

theorem token_attn_reducev_final_store_python_test_shape_compute_correct
    (Acc Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := token_attn_reducev_final_store_slice Acc Out
        256 64 1 256 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        s.readMem Acc (accOffset s 256 64 1 i)) := by
  exact token_attn_reducev_final_store_slice_compute_correct Acc Out
    256 64 1 256 64 1 64 s
    (token_attn_reducev_python_test_shape_offset_injective s)

/-- Python case 1 full reduce-V surface lowering for `batch = 2`,
`seq_len = 128`, `num_heads = 4`, and `d_model = 64`. -/
theorem token_attn_reducev_python_case1_surface_toAlgorithm_supported
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat) :
    ∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 64 128).toAlgorithm? =
        Except.ok alg := by
  exact token_attn_reducev_surface_toAlgorithm_supported Prob V Out
    Req_to_tokens B_req_idx B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1
    256 64 1 1 64 128

/-- Python case 2 full reduce-V surface lowering for the `seq_len = 64`
variant. -/
theorem token_attn_reducev_python_case2_surface_toAlgorithm_supported
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat) :
    ∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen 64 1 64 1 4096 64 1 256 64 1 1 64 128).toAlgorithm? =
        Except.ok alg := by
  exact token_attn_reducev_surface_toAlgorithm_supported Prob V Out
    Req_to_tokens B_req_idx B_Start_Loc B_Seqlen 64 1 64 1 4096 64 1
    256 64 1 1 64 128

/-- Python case 3 full reduce-V surface lowering for the `batch = 3`,
`seq_len = 128` variant. -/
theorem token_attn_reducev_python_case3_surface_toAlgorithm_supported
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat) :
    ∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 64 128).toAlgorithm? =
        Except.ok alg := by
  exact token_attn_reducev_surface_toAlgorithm_supported Prob V Out
    Req_to_tokens B_req_idx B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1
    256 64 1 1 64 128

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
and are intended to replace the self-referential `tokenAttnReduceVSurfaceValue`.

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

/-! ### Remaining surface→closed-form bridge (banked)

`tokenAttnReduceVClosedForm` is the genuine, self-reference-free PV spec. The
remaining step is the surface readback bridge

```
exec (token_attn_reducev_surface …) s = some s' →
  s'.readMem Out (outOffset … i) =
    tokenAttnReduceVClosedForm … i
```

Routing/decode recipe for that bridge (this kernel is the **PV reduce /
accumulate** route — there is no online softmax):

* Exec assembly: `exec` reduces to `stepStmts (…).toAlgKernel.body s` by `rfl`
  (the `ComputeStmt → Stmt` lowering of every `ComputeExpr.alg` body statement
  is definitional). Decode the prelude assigns and the
  `forRangeDyn "start_n" 0 cur_batch_seq_len BLOCK_N …` loop via
  `stepStmts.cons_some (stepStmt_assign_eq_some (…_op_eval …))` and
  `forRangeDyn_inv` (added to `VeriTile/Triton/LoopInvariant.lean`).
* Outer loop: drive `forRangeDyn_inv` with the accumulator invariant
  `acc_k = Σ_{n < min (k, cur_batch_seq_len)} p[n]·v[v_loc[n], d]`; the loop runs
  `⌈cur_batch_seq_len / BLOCK_N⌉` blocks. The final counter `final ≥
  cur_batch_seq_len` collapses `min` to `cur_batch_seq_len`, yielding
  `tokenAttnReduceVPVValue`.
* Per loop body, supply `*_op_eval` recipes for: the `Prob` masked load
  (`pOffset`, mask `n < cur_batch_seq_len`); the `Req_to_tokens` gather (`vLoc`);
  the 2D `v_offs`/`v_value` masked gather (`vOffset`, same mask, `other = 0`);
  the `tl.sum(p_value[:,None]·v_value, 0)` block reduction along axis 0
  (`reduceSum_some` + `Finset.sum_congr`, as in `batched_vecmat`); and the
  `acc += …` accumulation.
* Final unmasked store readback: `scatter_readback_nd` (as in
  `token_attn_reducev_final_store_slice_correct`), reusing
  `token_attn_reducev_python_test_shape_offset_injective`.
-/

noncomputable def tokenAttnReduceVSurfaceValue
    (s : BlockState) (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N offset : Nat) : ℝ :=
  match exec (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
      stride_ph stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh
      stride_od kv_group_num BLOCK_DMODEL BLOCK_N) s with
  | some s' => s'.readMem Out offset
  | none => 0.0

theorem token_attn_reducev_surface_output_compute_correct
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
        stride_ph stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh
        stride_od kv_group_num BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i =>
        tokenAttnReduceVSurfaceValue s Prob V Out Req_to_tokens B_req_idx
          B_Start_Loc B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
          stride_ph stride_pbs stride_vbs stride_vh stride_vd stride_obs
          stride_oh stride_od kv_group_num BLOCK_DMODEL BLOCK_N
          (outOffset s stride_obs stride_oh stride_od i)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [token_attn_reducev_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [tokenAttnReduceVSurfaceValue, hExec]

/-- Public Python case 1 coverage summary: the full gather/reduceV surface
lowers and the final output vector store realizes the checked output shape. -/
theorem token_attn_reducev_python_case1_output_surface_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (s : BlockState) :
    (∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_reducev_final_store_slice Acc Out
        256 64 1 256 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        s.readMem Acc (accOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_reducev_python_case1_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
  · exact token_attn_reducev_final_store_python_test_shape_compute_correct
      Acc Out s

/-- Public Python case 2 coverage summary. -/
theorem token_attn_reducev_python_case2_output_surface_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (s : BlockState) :
    (∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen 64 1 64 1 4096 64 1 256 64 1 1 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_reducev_final_store_slice Acc Out
        256 64 1 256 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        s.readMem Acc (accOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_reducev_python_case2_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
  · exact token_attn_reducev_final_store_python_test_shape_compute_correct
      Acc Out s

/-- Public Python case 3 coverage summary. -/
theorem token_attn_reducev_python_case3_output_surface_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (s : BlockState) :
    (∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_reducev_final_store_slice Acc Out
        256 64 1 256 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        s.readMem Acc (accOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_reducev_python_case3_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
  · exact token_attn_reducev_final_store_python_test_shape_compute_correct
      Acc Out s

/-- Python reduce-V token-attention case 1 final-store coverage. -/
abbrev token_attn_reducev_python_case1_store_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (s : BlockState) :=
  token_attn_reducev_python_case1_output_surface_summary
    Prob V Acc Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen s

/-- Python reduce-V token-attention case 2 final-store coverage. -/
abbrev token_attn_reducev_python_case2_store_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (s : BlockState) :=
  token_attn_reducev_python_case2_output_surface_summary
    Prob V Acc Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen s

/-- Python reduce-V token-attention case 3 final-store coverage. -/
abbrev token_attn_reducev_python_case3_store_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (s : BlockState) :=
  token_attn_reducev_python_case3_output_surface_summary
    Prob V Acc Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen s




















theorem token_attn_reducev_python_case1_output_summary
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (s : BlockState) :
    (∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 64 128)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        tokenAttnReduceVSurfaceValue s Prob V Out Req_to_tokens B_req_idx
          B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 64 128
          (outOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_reducev_python_case1_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
  · exact token_attn_reducev_surface_output_compute_correct Prob V Out
      Req_to_tokens B_req_idx B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1
      256 64 1 1 64 128 s

theorem token_attn_reducev_python_case2_output_summary
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (s : BlockState) :
    (∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen 64 1 64 1 4096 64 1 256 64 1 1 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen 64 1 64 1 4096 64 1 256 64 1 1 64 128)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        tokenAttnReduceVSurfaceValue s Prob V Out Req_to_tokens B_req_idx
          B_Start_Loc B_Seqlen 64 1 64 1 4096 64 1 256 64 1 1 64 128
          (outOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_reducev_python_case2_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
  · exact token_attn_reducev_surface_output_compute_correct Prob V Out
      Req_to_tokens B_req_idx B_Start_Loc B_Seqlen 64 1 64 1 4096 64 1
      256 64 1 1 64 128 s

theorem token_attn_reducev_python_case3_output_summary
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (s : BlockState) :
    (∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
        B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 64 128)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        tokenAttnReduceVSurfaceValue s Prob V Out Req_to_tokens B_req_idx
          B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 64 128
          (outOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_reducev_python_case3_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
  · exact token_attn_reducev_surface_output_compute_correct Prob V Out
      Req_to_tokens B_req_idx B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1
      256 64 1 1 64 128 s

end VeriTile.Bench.TritonBenchG.TokenAttnReduceV
