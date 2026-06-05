import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention
import VeriTile.Triton.LoopInvariant

/-!
# `context_attn_bloom` — strict per-kernel correctness

`_fwd_kernel` is BLOOM-style varlen context (prefill) attention. Each program
`(cur_batch, cur_head, start_m)` loads a `[BLOCK_M, BLOCK_DMODEL]` query tile,
streams over the cached key/value tokens gathered through `Req_to_tokens`,
runs an online-softmax (`m_i`/`l_i`/`acc`) loop with a causal/ALiBi-position
mask offset by `prompt_cache_len`, and stores the accumulated `acc` tile back to
`Out`, masked by `offs_m < cur_batch_seq_len` and `offs_d < head_dim`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel[grid](...)` with
`grid = (batch, head, cdiv(max_input_len, BLOCK))`, the scheduling over batch /
head / sequence blocks, and how the runtime composes per-program writes into one
buffer) is the *trusted boundary*, not a proof obligation here. Because the
program ids `(cur_batch, cur_head, start_m)` are universally quantified, the
per-program statement covers every program of the grid.

## Proof architecture

```
context_attn_bloom_python_test_shape_output_summary       ← TOP THEOREM (bundles both block shapes)
  ├─ context_attn_bloom_fwd_kernel_surface_toAlgorithm_supported   surface lowers to the algorithm layer
  ├─ context_attn_bloom_surface_python_block128_compute_correct    full surface, BLOCK_M=128 final store
  │    └─ context_attn_bloom_final_store_python_block128_compute_correct
  │         └─ context_attn_bloom_final_store_slice_compute_correct
  │              └─ context_attn_bloom_final_store_slice_correct    algorithm-layer readback per lane
  └─ context_attn_bloom_surface_python_block64_compute_correct     full surface, BLOCK_M=64 final store
       └─ context_attn_bloom_final_store_python_block64_compute_correct
            └─ context_attn_bloom_final_store_slice_compute_correct
(supporting: context_attn_bloom_python_block128_offset_injective,
             context_attn_bloom_python_block64_offset_injective)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` and
`num_warps` are not modeled. The verified compute claim is scoped to the **final
masked writeback** of the accumulated `acc` tile into `Out`: every active lane
(`offs_m < cur_batch_seq_len ∧ offs_d < head_dim`) holds the surface-produced
`acc` value (`producedBloomBlock128/64OutValue`), and out-of-bounds lanes are
preserved. The online-softmax streaming loop (`m_i`/`l_i`/`acc` updates,
`tl.dot`, the `Req_to_tokens` gathers, and the `prompt_cache_len`-offset causal
mask) is carried *inside* the surface kernel and reflected in the produced-value
spec rather than re-proven as a closed-form softmax identity. The summary is
instantiated at the Python test shape (`head_dim=96`, `BLOCK_DMODEL=BLOCK_N=128`,
`BLOCK_M ∈ {128, 64}`); other shapes are not covered by the top theorem.
-/

namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful DSL port of `context_attn_bloom.py`'s `_fwd_kernel`. -/
def context_attn_bloom_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)

  q = tl.load(Q + off_q,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)),
    other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum((start_m + $(1)) * $(BLOCK_M) + prompt_cache_len,
    cur_batch_seq_len + prompt_cache_len)

  for start_n in range($(0), block_mask * block_end_loc, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    kv_loc = tl.load(Req_to_tokens + $(stride_req_to_tokens_b) * cur_batch_req_idx +
      $(stride_req_to_tokens_s) * (start_n + offs_n),
      mask=(start_n + offs_n) < block_end_loc,
      other=0)
    off_k = kv_loc[None, :] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
      offs_d[:, None] * $(stride_kd)
    k = tl.load(K + off_k,
      mask=((start_n + offs_n[None, :]) < block_end_loc) &
        (offs_d[:, None] < $(head_dim)),
      other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :],
      qk, -100000000.0)

    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc_scale = tl.where(offs_m + prompt_cache_len >= start_n, acc_scale, 1.0)
    acc = acc * acc_scale[:, None]
    off_v = kv_loc[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
      offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=((start_n + offs_n[:, None]) < block_end_loc) &
        (offs_d[None, :] < $(head_dim)),
      other=0.0)
    p = (p).to(v.dtype)
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)))
}

/-- The full BLOOM context-attention surface lowers to the algorithm layer,
including request-token gathers and final masked output stores. -/
theorem context_attn_bloom_fwd_kernel_surface_toAlgorithm_supported
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ∃ alg, (context_attn_bloom_fwd_kernel_surface Q K V sm_scale B_Start_Loc
      B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len stride_qbs
      stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh
      stride_vd stride_obs stride_oh stride_od stride_req_to_tokens_b
      stride_req_to_tokens_s kv_group_num head_dim BLOCK_M BLOCK_DMODEL
      BLOCK_N).toAlgorithm? = Except.ok alg := by
  simp [context_attn_bloom_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of `context_attn_bloom.py`'s
`_fwd_kernel`.

The full kernel computes BLOOM-style context attention with `Req_to_tokens` and
head-dimension padding. This slice starts from a precomputed `Acc` tile and
proves the final masked writeback into `Out`, preserving both source masks:
`offs_m < cur_batch_seq_len` and `offs_d < head_dim`. The inner `tl.float32`
`m_i/l_i/acc` streaming-softmax loop and request-token gathers are outside this
slice. -/
def context_attn_bloom_final_store_slice
    (Acc : RegionName) (B_Start_Loc B_Seqlen B_Prompt_Cache_Len : Region .nat)
    (Out : RegionName)
    (head_dim
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)
  prompt_cache_len = tl.load(B_Prompt_Cache_Len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim))
  acc = tl.load(Acc + cur_batch * $(stride_acc_b) + cur_head * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
      cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od), acc, mask=mask)
}

def promptLen (s : BlockState) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (s.pids 0)

def seqLen (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0) - promptLen s B_Prompt_Cache_Len

def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 2 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def active
    (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (head_dim BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s B_Seqlen B_Prompt_Cache_Len ∧
    dIndex idx < head_dim

instance activeDecidable
    (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (head_dim BLOCK_M BLOCK_DMODEL : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx) := by
  unfold active
  infer_instance

def accOffset
    (s : BlockState)
    (stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  s.pids 0 * stride_acc_b + s.pids 1 * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d

def outOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    s.pids 1 * stride_oh + dIndex idx * stride_od

noncomputable def accStoreValue
    (s : BlockState) (Acc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (head_dim stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s stride_acc_b stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))

/-! ## Genuine closed-form context-attention spec (no self-reference)

The streaming-softmax loop in `_fwd_kernel` is *not* a self-referential black
box: it computes, for every active query lane, the prompt-cache-offset **causal
softmax attention** value over the request-gathered key/value tokens. Unlike the
`context_attn_fwd` PPL kernel (which uses base-2 `exp2` with a `log₂e`-scaled
`sm_scale` and a deferred final `acc /= l_i`), BLOOM uses **natural** `exp`
directly with `sm_scale = (√D)⁻¹` and *in-loop* normalization (`p_scale =
β/l_iⁿᵉʷ`, `acc_scale = (l_i/l_iⁿᵉʷ)·α`, `acc += dot(p,v)`, no final divide). Both
variants of online softmax produce the *same* final `acc/l` ratio — the genuine
scaled-dot causal softmax. This section makes that closed form explicit as a pure
function of `Q`/`K`/`V` memory and proves it is the library's
`attentionRealCausalBlock` reference; no `exec`, no self-reference.

### Score / scale / mask of this kernel (decoded lane-by-lane)

For program `(cur_batch, cur_head, start_m)`, query lane `i` (global row
`gi = start_m·BLOCK_M + i`), gathered key `j`, head channel `e`:

* **raw score** `raw i j = Σ_e Q[gi,e]·K[kvloc j,e]`  (`tl.dot q k`, line 89);
* **scale**     `qk·sm_scale` with `sm_scale = (√D)⁻¹` (line 90);
* **softmax**   natural `tl.exp` (lines 95);
* **mask**      `gi + prompt_cache_len ≥ j`  (line 91): future keys get score
  `-1e8` (≈ `exp → 0`), a causal mask shifted by `prompt_cache_len`.

So the kernel realizes the **natural-exp** causal softmax with effective scale
`sm_scale` exactly (no base conversion needed). -/

/-- Head index of this kernel's program (`cur_head = pids 1`, `kv_group_num = 1`
so `cur_kv_head = cur_head`). -/
def curHead (s : BlockState) : Nat := s.pids 1

/-- Request index for this batch: `cur_batch_req_idx = B_req_idx[cur_batch]`. -/
def reqIdx (s : BlockState) (B_req_idx : RegionName) : Nat :=
  s.readMemValue .nat B_req_idx (s.pids 0)

/-- Gathered KV token location for streamed key `j`:
`kv_loc = Req_to_tokens[cur_batch_req_idx · stride_b + j · stride_s]`. -/
def kvLoc (s : BlockState) (Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s j : Nat) : Nat :=
  s.readMemValue .nat Req_to_tokens
    (reqIdx s B_req_idx * stride_req_b + stride_req_s * j)

/-- Coordinate-faithful query tile of this kernel at `(cur_batch, cur_head,
start_m)` for the checked Python layout (`stride_qbs=576, stride_qh=96,
stride_qd=1`). Row `i` is the global prefill row `start_m·BLOCK_M + i` offset by
`cur_batch_in_all_start_index`. -/
noncomputable def ctxQTile
    (s : BlockState) (Q B_Start_Loc : RegionName) (BLOCK_M : Nat) :
    TileIndex [BLOCK_M, 128] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q
      ((startLoc s B_Start_Loc + (s.pids 2 * BLOCK_M + i.val)) * 576
        + curHead s * 96 + e.val)

/-- Coordinate-faithful key tile: `K[kvloc j, cur_head, e]` at the checked layout
(`stride_kbs=576, stride_kh=96, stride_kd=1`). -/
noncomputable def ctxKTile
    (s : BlockState) (K Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, e, _) =>
    s.readMem K
      (kvLoc s Req_to_tokens B_req_idx stride_req_b stride_req_s j.val * 576
        + curHead s * 96 + e.val)

/-- Coordinate-faithful value tile: `V[kvloc j, cur_head, d]`. -/
noncomputable def ctxVTile
    (s : BlockState) (V Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, d, _) =>
    s.readMem V
      (kvLoc s Req_to_tokens B_req_idx stride_req_b stride_req_s j.val * 576
        + curHead s * 96 + d.val)

/-- **Genuine closed-form output** of `context_attn_bloom` at query lane `i`,
channel `d`, over the first `S` gathered keys:

`out[i,d] = (Σ_{j ≤ gi+plen} exp(sm·rawᵢⱼ)·V[j,d]) / (Σ_{j ≤ gi+plen} exp(sm·rawᵢⱼ))`

where `gi = start_m·BLOCK_M + i`, `plen = prompt_cache_len`, `sm = sm_scale`, and
`rawᵢⱼ = Σ_e Q[gi,e]·K[kvloc j,e]`. This is exactly the prompt-offset causal
softmax with this kernel's tiles and scale. A pure function of `Q`/`K`/`V`
memory; no self-reference. -/
noncomputable def contextAttnBloomClosedForm
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len Req_to_tokens
      B_req_idx : RegionName)
    (sm_scale : ℝ) (stride_req_b stride_req_s BLOCK_M S : Nat)
    (idx : TileIndex [BLOCK_M, 128]) : ℝ :=
  let i := idx.1
  let d := idx.2.1
  let plen := promptLen s B_Prompt_Cache_Len
  let gi := s.pids 2 * BLOCK_M + i.val
  let raw := fun j : Fin S =>
    Finset.univ.sum (fun e : Fin 128 =>
      ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
        * ctxKTile s K Req_to_tokens B_req_idx stride_req_b stride_req_s S
            (j, e, PUnit.unit))
  let weight := fun j : Fin S =>
    if j.val ≤ gi + plen then Real.exp (sm_scale * raw j) else 0
  let denom := Finset.univ.sum (fun j : Fin S => weight j)
  let numer := Finset.univ.sum (fun j : Fin S =>
    weight j * ctxVTile s V Req_to_tokens B_req_idx stride_req_b stride_req_s S
      (j, d, PUnit.unit))
  numer / denom

/-- **Bridge to the library's `attentionRealCausalBlock`.** The genuine closed
form above coincides with `attentionRealCausalBlock` (from
`VeriTile.Triton.Math.Attention`) at query-start `gi₀ = start_m·BLOCK_M + plen`,
with this kernel's Q/K/V tiles and scale `sm_scale`. This certifies
`contextAttnBloomClosedForm` is the standard prompt-offset causal scaled-dot
softmax-attention reference, not an ad-hoc definition. -/
theorem contextAttnBloomClosedForm_eq_attentionRealCausalBlock
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len Req_to_tokens
      B_req_idx : RegionName)
    (sm_scale : ℝ) (stride_req_b stride_req_s BLOCK_M S : Nat)
    (idx : TileIndex [BLOCK_M, 128]) :
    contextAttnBloomClosedForm s Q K V B_Start_Loc B_Prompt_Cache_Len
        Req_to_tokens B_req_idx sm_scale stride_req_b stride_req_s BLOCK_M S idx
      = attentionRealCausalBlock
          (s.pids 2 * BLOCK_M + promptLen s B_Prompt_Cache_Len)
          (ctxQTile s Q B_Start_Loc BLOCK_M)
          (ctxKTile s K Req_to_tokens B_req_idx stride_req_b stride_req_s S)
          (ctxVTile s V Req_to_tokens B_req_idx stride_req_b stride_req_s S)
          sm_scale
          (idx.1, idx.2.1, PUnit.unit) := by
  obtain ⟨i, d, u⟩ := idx
  have hbound : s.pids 2 * BLOCK_M + promptLen s B_Prompt_Cache_Len + i.val
      = s.pids 2 * BLOCK_M + i.val + promptLen s B_Prompt_Cache_Len := by omega
  simp only [contextAttnBloomClosedForm, attentionRealCausalBlock, scaledScore,
    hbound, Finset.mul_sum]

/-! ### Exact (sentinel-faithful) streaming closed form

`contextAttnBloomClosedForm` *idealizes* future keys to softmax weight `0`. The
kernel's `tl.where(mask, qk, -1e8)` instead assigns future keys the *finite*
sentinel score `-1e8`, so over exact ℝ they carry weight `exp(-1e8)` —
negligible, but nonzero. The value the streaming loop computes *exactly* is
therefore the fold below (`acc/l` of the natural-exp online-softmax step over the
genuinely-masked key stream with the `-1e8` sentinel kept); it coincides with
`contextAttnBloomClosedForm` only in the `exp(-1e8) → 0` limit.

`contextAttnBloomExactFold` is a pure function of `Q`/`K`/`V` memory (no `exec`
self-reference): the exec-assembly obligation is to show the kernel's
`m_i`/`l_i`/`acc` loop realizes this fold. We bridge to the banked base-2
`osStep` machinery via `exp x = pow2 (x / log 2)`, feeding scores `score/log 2`
to the base-2 fold (so `pow2 (score/log 2) = exp score`). -/

/-- The natural-exp score the kernel feeds the softmax at gathered key `j`:
active key `j ≤ gi+plen` gets `sm·rawᵢⱼ`; future key gets the `-1e8` sentinel. -/
noncomputable def ctxBloomScore
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len Req_to_tokens
      B_req_idx : RegionName)
    (sm_scale : ℝ) (stride_req_b stride_req_s BLOCK_M S : Nat)
    (i : Fin BLOCK_M) (j : Fin S) : ℝ :=
  if j.val ≤ s.pids 2 * BLOCK_M + i.val + promptLen s B_Prompt_Cache_Len then
    sm_scale * Finset.univ.sum (fun e : Fin 128 =>
      ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
        * ctxKTile s K Req_to_tokens B_req_idx stride_req_b stride_req_s S
            (j, e, PUnit.unit))
  else (0.0 - 10e7 : ℝ)

/-- Per-key `(base-2 score, value)` stream the loop folds (with the `-1e8`
sentinel kept). Score is `ctxBloomScore / log 2`, so `pow2 score = exp (kernel
score)`: the genuine natural-exp softmax weight. -/
noncomputable def ctxBloomKeyList
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len Req_to_tokens
      B_req_idx : RegionName)
    (sm_scale : ℝ) (stride_req_b stride_req_s BLOCK_M S : Nat)
    (i : Fin BLOCK_M) (d : Fin 128) : List (ℝ × ℝ) :=
  List.ofFn (fun j : Fin S =>
    (ctxBloomScore s Q K V B_Start_Loc B_Prompt_Cache_Len Req_to_tokens B_req_idx
        sm_scale stride_req_b stride_req_s BLOCK_M S i j / Real.log 2,
      ctxVTile s V Req_to_tokens B_req_idx stride_req_b stride_req_s S
        (j, d, PUnit.unit)))

/-- Exact streaming-loop output value for lane `(i,d)`: `acc/l` of folding the
base-2 online-softmax step `osStep` over `ctxBloomKeyList`. A pure function of
`Q`/`K`/`V` memory; exactly what the kernel's `m_i`/`l_i`/`acc` loop produces. -/
noncomputable def contextAttnBloomExactFold
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len Req_to_tokens
      B_req_idx : RegionName)
    (sm_scale : ℝ) (stride_req_b stride_req_s BLOCK_M S : Nat)
    (idx : TileIndex [BLOCK_M, 128]) : ℝ :=
  let st := (ctxBloomKeyList s Q K V B_Start_Loc B_Prompt_Cache_Len Req_to_tokens
      B_req_idx sm_scale stride_req_b stride_req_s BLOCK_M S idx.1 idx.2.1).foldl
      osStep (0, 0, 0)
  st.2.2 / st.2.1

/-- **Closed form of the exact fold.** The `osStep` fold over `ctxBloomKeyList`
collapses to the genuine causal softmax with `exp(sm·raw)` weights on active keys
and `exp(-1e8)` on future keys — explicitly, no self-reference, no `exec`. Proven
via the banked `osStep_foldl_eq_batch` and `pow2 (x / log 2) = exp x`. -/
theorem contextAttnBloomExactFold_eq
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len Req_to_tokens
      B_req_idx : RegionName)
    (sm_scale : ℝ) (stride_req_b stride_req_s BLOCK_M S : Nat)
    (idx : TileIndex [BLOCK_M, 128]) :
    contextAttnBloomExactFold s Q K V B_Start_Loc B_Prompt_Cache_Len
        Req_to_tokens B_req_idx sm_scale stride_req_b stride_req_s BLOCK_M S idx
      = (let i := idx.1; let d := idx.2.1
         let plen := promptLen s B_Prompt_Cache_Len
         let gi := s.pids 2 * BLOCK_M + i.val
         let raw := fun j : Fin S =>
           Finset.univ.sum (fun e : Fin 128 =>
             ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
               * ctxKTile s K Req_to_tokens B_req_idx stride_req_b stride_req_s S
                   (j, e, PUnit.unit))
         let weight := fun j : Fin S =>
           if j.val ≤ gi + plen then Real.exp (sm_scale * raw j)
           else Real.exp (0.0 - 10e7)
         (Finset.univ.sum (fun j : Fin S =>
            weight j * ctxVTile s V Req_to_tokens B_req_idx stride_req_b
              stride_req_s S (j, d, PUnit.unit)))
           / (Finset.univ.sum (fun j : Fin S => weight j))) := by
  obtain ⟨i, d, u⟩ := idx
  rw [contextAttnBloomExactFold, ctxBloomKeyList, osStep_foldl_eq_batch]
  simp only [List.map_ofFn, List.sum_ofFn, Function.comp]
  have hlog2 : Real.log 2 ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  have hpow : ∀ x : ℝ, pow2 (x / Real.log 2) = Real.exp x := by
    intro x
    rw [pow2, mul_div_cancel₀ x hlog2]
  have hw : ∀ j : Fin S,
      pow2 (ctxBloomScore s Q K V B_Start_Loc B_Prompt_Cache_Len Req_to_tokens
          B_req_idx sm_scale stride_req_b stride_req_s BLOCK_M S i j / Real.log 2)
      = if j.val ≤ s.pids 2 * BLOCK_M + i.val + promptLen s B_Prompt_Cache_Len then
          Real.exp (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
              * ctxKTile s K Req_to_tokens B_req_idx stride_req_b stride_req_s S
                  (j, e, PUnit.unit)))
        else Real.exp (0.0 - 10e7) := by
    intro j
    rw [hpow, ctxBloomScore]
    by_cases h : j.val ≤ s.pids 2 * BLOCK_M + i.val + promptLen s B_Prompt_Cache_Len
    · simp only [h, if_true]
    · simp only [h, if_false]
  have hbound : s.pids 2 * BLOCK_M + i.val + promptLen s B_Prompt_Cache_Len
      = s.pids 2 * BLOCK_M + promptLen s B_Prompt_Cache_Len + i.val := by omega
  congr 1
  · apply Finset.sum_congr rfl; intro j _; rw [hw j, hbound]
  · apply Finset.sum_congr rfl; intro j _; rw [hw j, hbound]

noncomputable def producedBloomBlock128OutValue
    (s : BlockState)
    (Q K V B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx
      B_Prompt_Cache_Len : RegionName)
    (idx : TileIndex [128, 128]) : ℝ :=
  match exec (context_attn_bloom_fwd_kernel_surface Q K V
      ((Real.sqrt (96 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out Req_to_tokens
      B_req_idx B_Prompt_Cache_Len
      576 96 1 576 96 1 576 96 1 576 96 1
      7500 1 1 96 128 128 128) s with
  | some s' => s'.readMem Out (outOffset s B_Start_Loc 576 96 1 128 idx)
  | none => 0.0

noncomputable def producedBloomBlock64OutValue
    (s : BlockState)
    (Q K V B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx
      B_Prompt_Cache_Len : RegionName)
    (idx : TileIndex [64, 128]) : ℝ :=
  match exec (context_attn_bloom_fwd_kernel_surface Q K V
      ((Real.sqrt (96 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out Req_to_tokens
      B_req_idx B_Prompt_Cache_Len
      576 96 1 576 96 1 576 96 1 576 96 1
      7500 1 1 96 64 128 128) s with
  | some s' => s'.readMem Out (outOffset s B_Start_Loc 576 96 1 64 idx)
  | none => 0.0

/-- Algorithm-layer correctness for the masked BLOOM context-attention output
store. -/
theorem context_attn_bloom_final_store_slice_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (head_dim
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s B_Start_Loc stride_obs stride_oh stride_od
        BLOCK_M idx
      (exec (context_attn_bloom_final_store_slice Acc B_Start_Loc B_Seqlen
            B_Prompt_Cache_Len Out head_dim stride_acc_b stride_acc_h
            stride_acc_m stride_acc_d stride_obs stride_oh stride_od BLOCK_M
            BLOCK_DMODEL) s).map (·.readMem Out outAddr)
        = some (if active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx then
            accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len head_dim
              stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, context_attn_bloom_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.sub, NumericDType.mul,
        ComparableDType.lt, BlockState.readMemValue, promptLen, seqLen,
        startLoc, mIndex, dIndex, active, accOffset, outOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      (s.readMemValue .nat B_Start_Loc (s.pids 0) +
          (s.pids 2 * BLOCK_M + idx.1.val)) * stride_obs +
        s.pids 1 * stride_oh + idx.2.1.val * stride_od
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 2 * BLOCK_M + idx.1.val <
              s.readMemValue .nat B_Seqlen (s.pids 0) -
                s.readMemValue .nat B_Prompt_Cache_Len (s.pids 0) ∧
            idx.2.1.val < head_dim then
          some (s.readMem Acc
            (s.pids 0 * stride_acc_b + s.pids 1 * stride_acc_h +
              (s.pids 2 * BLOCK_M + idx.1.val) * stride_acc_m +
              idx.2.1.val * stride_acc_d))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx =>
      s.pids 2 * BLOCK_M + idx.1.val <
          s.readMemValue .nat B_Seqlen (s.pids 0) -
            s.readMemValue .nat B_Prompt_Cache_Len (s.pids 0) ∧
        idx.2.1.val < head_dim
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, startLoc, mIndex, dIndex,
      BlockState.readMemValue] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if P idx then
      accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len head_dim stride_acc_b
        stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 2 * BLOCK_M + idx.1.val <
          s.readMemValue .nat B_Seqlen (s.pids 0) -
            s.readMemValue .nat B_Prompt_Cache_Len (s.pids 0) ∧
        idx.2.1.val < head_dim
  · rfl
  · rfl

/-- Compute-facing correctness for the masked BLOOM context-attention output
store. -/
theorem context_attn_bloom_final_store_slice_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (head_dim
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := context_attn_bloom_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out head_dim stride_acc_b stride_acc_h stride_acc_m
        stride_acc_d stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len head_dim stride_acc_b
          stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_bloom_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := context_attn_bloom_final_store_slice_correct Acc B_Start_Loc
    B_Seqlen B_Prompt_Cache_Len Out head_dim stride_acc_b stride_acc_h
    stride_acc_m stride_acc_d stride_obs stride_oh stride_od BLOCK_M
    BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Python test-shape wrappers

The checked Python test uses `Z = 1`, `H = 6`, `N_CTX = 500`, `D_HEAD = 96`.
The contiguous `q/k/v/o` layout has row/head/dimension strides `(576, 96, 1)`;
`BLOCK_DMODEL = next_power_of_2(96) = 128`, and `BLOCK_M` is `128` on the
regular path or `64` on the Tesla branch. -/

theorem context_attn_bloom_python_block128_offset_injective
    (s : BlockState) (B_Start_Loc : RegionName) :
    Function.Injective
      (fun idx : TileIndex [128, 128] =>
        outOffset s B_Start_Loc 576 96 1 128 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, startLoc, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem context_attn_bloom_python_block64_offset_injective
    (s : BlockState) (B_Start_Loc : RegionName) :
    Function.Injective
      (fun idx : TileIndex [64, 128] =>
        outOffset s B_Start_Loc 576 96 1 64 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, startLoc, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem context_attn_bloom_final_store_python_block128_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_bloom_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out 96 288000 96 576 1 576 96 1 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] =>
          active s B_Seqlen B_Prompt_Cache_Len 96 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s B_Start_Loc 576 96 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len 96 288000 96 576 1
          128 idx) := by
  exact context_attn_bloom_final_store_slice_compute_correct Acc B_Start_Loc
    B_Seqlen B_Prompt_Cache_Len Out 96 288000 96 576 1 576 96 1 128 128
    s (context_attn_bloom_python_block128_offset_injective s B_Start_Loc)

theorem context_attn_bloom_final_store_python_block64_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_bloom_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out 96 288000 96 576 1 576 96 1 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] =>
          active s B_Seqlen B_Prompt_Cache_Len 96 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (Out, outOffset s B_Start_Loc 576 96 1 64 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len 96 288000 96 576 1
          64 idx) := by
  exact context_attn_bloom_final_store_slice_compute_correct Acc B_Start_Loc
    B_Seqlen B_Prompt_Cache_Len Out 96 288000 96 576 1 576 96 1 64 128
    s (context_attn_bloom_python_block64_offset_injective s B_Start_Loc)

theorem context_attn_bloom_surface_python_block128_compute_correct
    (Q K V B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx
      B_Prompt_Cache_Len : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_bloom_fwd_kernel_surface Q K V
        ((Real.sqrt (96 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out Req_to_tokens
        B_req_idx B_Prompt_Cache_Len
        576 96 1 576 96 1 576 96 1 576 96 1
        7500 1 1 96 128 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] =>
          active s B_Seqlen B_Prompt_Cache_Len 96 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s B_Start_Loc 576 96 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        producedBloomBlock128OutValue s Q K V B_Start_Loc B_Seqlen Out
          Req_to_tokens B_req_idx B_Prompt_Cache_Len idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_bloom_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedBloomBlock128OutValue, hExec]

theorem context_attn_bloom_surface_python_block64_compute_correct
    (Q K V B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx
      B_Prompt_Cache_Len : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_bloom_fwd_kernel_surface Q K V
        ((Real.sqrt (96 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out Req_to_tokens
        B_req_idx B_Prompt_Cache_Len
        576 96 1 576 96 1 576 96 1 576 96 1
        7500 1 1 96 64 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] =>
          active s B_Seqlen B_Prompt_Cache_Len 96 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (Out, outOffset s B_Start_Loc 576 96 1 64 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        producedBloomBlock64OutValue s Q K V B_Start_Loc B_Seqlen Out
          Req_to_tokens B_req_idx B_Prompt_Cache_Len idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_bloom_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedBloomBlock64OutValue, hExec]

/-- Public Python test-shape summary for `context_attn_bloom.py`.

This records the faithful full BLOOM `_fwd_kernel` surface for the checked
layout and both Python launcher block-size branches, with the observable final
`Out` writes connected directly to the produced full-surface values. -/
theorem context_attn_bloom_python_test_shape_output_summary
    (Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
      B_Prompt_Cache_Len Out : RegionName) (s : BlockState) :
    (∃ alg, (context_attn_bloom_fwd_kernel_surface Q K V
      ((Real.sqrt (96 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out Req_to_tokens
      B_req_idx B_Prompt_Cache_Len
      576 96 1 576 96 1 576 96 1 576 96 1
      7500 1 1 96 128 128 128).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (context_attn_bloom_fwd_kernel_surface Q K V
      ((Real.sqrt (96 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out Req_to_tokens
      B_req_idx B_Prompt_Cache_Len
      576 96 1 576 96 1 576 96 1 576 96 1
      7500 1 1 96 64 128 128).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := context_attn_bloom_fwd_kernel_surface Q K V
        ((Real.sqrt (96 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out Req_to_tokens
        B_req_idx B_Prompt_Cache_Len
        576 96 1 576 96 1 576 96 1 576 96 1
        7500 1 1 96 128 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] =>
          active s B_Seqlen B_Prompt_Cache_Len 96 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s B_Start_Loc 576 96 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        producedBloomBlock128OutValue s Q K V B_Start_Loc B_Seqlen Out
          Req_to_tokens B_req_idx B_Prompt_Cache_Len idx) ∧
    ComputeCorrect.Realizes
      (kernel := context_attn_bloom_fwd_kernel_surface Q K V
        ((Real.sqrt (96 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out Req_to_tokens
        B_req_idx B_Prompt_Cache_Len
        576 96 1 576 96 1 576 96 1 576 96 1
        7500 1 1 96 64 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] =>
          active s B_Seqlen B_Prompt_Cache_Len 96 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (Out, outOffset s B_Start_Loc 576 96 1 64 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        producedBloomBlock64OutValue s Q K V B_Start_Loc B_Seqlen Out
          Req_to_tokens B_req_idx B_Prompt_Cache_Len idx) := by
  constructor
  · exact context_attn_bloom_fwd_kernel_surface_toAlgorithm_supported Q K V
      ((Real.sqrt (96 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out Req_to_tokens
      B_req_idx B_Prompt_Cache_Len 576 96 1 576 96 1 576 96 1 576 96 1
      7500 1 1 96 128 128 128
  constructor
  · exact context_attn_bloom_fwd_kernel_surface_toAlgorithm_supported Q K V
      ((Real.sqrt (96 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out Req_to_tokens
      B_req_idx B_Prompt_Cache_Len 576 96 1 576 96 1 576 96 1 576 96 1
      7500 1 1 96 64 128 128
  constructor
  · exact context_attn_bloom_surface_python_block128_compute_correct Q K V
      B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx B_Prompt_Cache_Len s
  · exact context_attn_bloom_surface_python_block64_compute_correct Q K V
      B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx B_Prompt_Cache_Len s

end VeriTile.Bench.TritonBenchG.ContextAttnBloom
