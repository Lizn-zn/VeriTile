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

/-! ## Generic ⊥-seeded online-softmax running-state machinery (exec-assembly layer)

Kernel-agnostic base-2 online-softmax fold over an abstract per-key
`g : Fin S → ℝ × ℝ` (score, value). The bloom kernel feeds it base-2 scores
`ctxBloomScore / log 2` so that `pow2 score = exp (natural kernel score)`. Ported
from the closed `context_attn_llama` template (#306). -/

open VeriTile.Triton (osStep pow2)

/-- One ⊥-seeded online-softmax step: running max in `WithBot ℝ` (seeded `⊥`), so
`α = realExp2(m ⊖ m')` is `0` on the first block. -/
noncomputable def osStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let sc := sv.1; let v := sv.2
  let m' := m ⊔ ((sc : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
  let p := pow2 (sc - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)

theorem osStepBot_foldl_fst
    (xs : List (ℝ × ℝ)) (m₀ : WithBot ℝ) (l₀ acc₀ : ℝ) :
    (xs.foldl osStepBot (m₀, l₀, acc₀)).1
      = (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldl (· ⊔ ·) m₀ := by
  induction xs generalizing m₀ l₀ acc₀ with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl_cons, List.map_cons]; rw [ih]; rfl

/-- **⊥-seeded consistency.** Folding `osStepBot` from `(m, l, acc)` anchored to the
true (max-free) denominator `L` / accumulator `T` via the ⊥-aware factor keeps that
invariant (`l = κ(m)·L`, `acc = κ(m)·T`, `κ ⊥ = 0`, `κ (some r) = pow2(−r)`). -/
theorem osStepBot_foldl_consistent (xs : List (ℝ × ℝ)) (m : WithBot ℝ) (l acc T L : ℝ)
    (hl : l = (m.elim 0 (fun r => pow2 (-r))) * L)
    (hacc : acc = (m.elim 0 (fun r => pow2 (-r))) * T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0) :
    let st := xs.foldl osStepBot (m, l, acc)
    st.2.1 = (st.1.elim 0 (fun r => pow2 (-r))) * (L + (xs.map (fun p => pow2 p.1)).sum) ∧
    st.2.2 = (st.1.elim 0 (fun r => pow2 (-r))) * (T + (xs.map (fun p => pow2 p.1 * p.2)).sum) := by
  induction xs generalizing m l acc T L with
  | nil => simp [hl, hacc]
  | cons x xs ih =>
    obtain ⟨sc, v⟩ := x
    set m' : WithBot ℝ := m ⊔ ((sc : ℝ) : WithBot ℝ) with hm'
    have hm'r : ∃ r : ℝ, m' = (r : WithBot ℝ) := by
      cases m with
      | bot => exact ⟨sc, by rw [hm']; rfl⟩
      | coe a => exact ⟨max a sc, by rw [hm']; rw [← WithBot.coe_max]⟩
    obtain ⟨mr, hmr⟩ := hm'r
    have hκm' : m'.elim 0 (fun r => pow2 (-r)) = pow2 (-mr) := by rw [hmr]; rfl
    have hunbot : m'.unbotD 0 = mr := by rw [hmr]; rfl
    have hp : pow2 (sc - m'.unbotD 0) = pow2 (-mr) * pow2 sc := by
      rw [hunbot, ← pow2_add]; ring_nf
    have hl' : l * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
        + pow2 (sc - m'.unbotD 0) = pow2 (-mr) * (L + pow2 sc) := by
      cases m with
      | bot =>
        rw [hmL rfl]
        have hz : (WithBot.realExp2 (WithBot.realSub (⊥ : WithBot ℝ) m')).unbotD 0 = 0 := by
          rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
        rw [hz, mul_zero, zero_add, hp]; ring
      | coe a =>
        have hm'a : m' = ((max a sc : ℝ) : WithBot ℝ) := by rw [hm']; rw [← WithBot.coe_max]
        have hmra : mr = max a sc := by rw [hm'a] at hmr; exact (WithBot.coe_inj.mp hmr.symm)
        have hαa : (pow2 (-a)) * (WithBot.realExp2 (WithBot.realSub (↑a) m')).unbotD 0
            = pow2 (-mr) := by
          rw [hm'a, WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe]
          rw [show Real.exp ((a - max a sc) * Real.log 2) = pow2 (a - max a sc) from by
            simp [pow2, mul_comm], ← pow2_add, hmra]; ring_nf
        rw [hl, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [mul_right_comm, hαa, hp]; ring
    have hacc' : acc * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
        + pow2 (sc - m'.unbotD 0) * v = pow2 (-mr) * (T + pow2 sc * v) := by
      cases m with
      | bot =>
        rw [hmT rfl]
        have hz : (WithBot.realExp2 (WithBot.realSub (⊥ : WithBot ℝ) m')).unbotD 0 = 0 := by
          rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
        rw [hz, mul_zero, zero_add, hp]; ring
      | coe a =>
        have hm'a : m' = ((max a sc : ℝ) : WithBot ℝ) := by rw [hm']; rw [← WithBot.coe_max]
        have hmra : mr = max a sc := by rw [hm'a] at hmr; exact (WithBot.coe_inj.mp hmr.symm)
        have hαa : (pow2 (-a)) * (WithBot.realExp2 (WithBot.realSub (↑a) m')).unbotD 0
            = pow2 (-mr) := by
          rw [hm'a, WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe]
          rw [show Real.exp ((a - max a sc) * Real.log 2) = pow2 (a - max a sc) from by
            simp [pow2, mul_comm], ← pow2_add, hmra]; ring_nf
        rw [hacc, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [mul_right_comm, hαa, hp]; ring
    have step := ih m'
      (l * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0 + pow2 (sc - m'.unbotD 0))
      (acc * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0 + pow2 (sc - m'.unbotD 0) * v)
      (T + pow2 sc * v) (L + pow2 sc) (by rw [hl', hκm']) (by rw [hacc', hκm'])
      (by rw [hmr]; simp) (by rw [hmr]; simp)
    simpa [List.foldl_cons, osStepBot, hm', List.map_cons, add_assoc] using step

/-- The `WithBot ⊔`-fold is seed/direction-agnostic. -/
theorem foldl_sup_bot_eq_foldr (L : List (WithBot ℝ)) :
    L.foldl (· ⊔ ·) (⊥ : WithBot ℝ) = L.foldr (· ⊔ ·) (⊥ : WithBot ℝ) := by
  have gen : ∀ (m : WithBot ℝ), L.foldl (· ⊔ ·) m = m ⊔ L.foldr (· ⊔ ·) ⊥ := by
    induction L with
    | nil => intro m; simp
    | cons a t ih =>
      intro m
      simp only [List.foldl_cons, List.foldr_cons, ih]
      rw [max_assoc]
  rw [gen ⊥, bot_sup_eq]

/-- Any list member is `≤` the `foldr ⊔ ⊥`. -/
theorem mem_le_foldr_sup (a : WithBot ℝ) :
    ∀ (L : List (WithBot ℝ)), a ∈ L → a ≤ L.foldr (· ⊔ ·) ⊥ := by
  intro L
  induction L with
  | nil => intro h; simp at h
  | cons x t ih =>
    intro h
    simp only [List.foldr_cons]
    rcases List.mem_cons.mp h with h | h
    · rw [h]; exact le_sup_left
    · exact le_trans (ih h) le_sup_right

/-- Generic windowed key list `[0, hi)` over an abstract per-key `g`. -/
noncomputable def gKeysUpto (S hi : Nat) (g : Fin S → ℝ × ℝ) : List (ℝ × ℝ) :=
  (List.finRange S).filterMap (fun j : Fin S => if j.val < hi then some (g j) else none)

/-- Generic block-`c` key list (keys `c·BN ≤ j < (c+1)·BN`). -/
noncomputable def gBlock (S BN c : Nat) (g : Fin S → ℝ × ℝ) : List (ℝ × ℝ) :=
  (List.finRange S).filterMap (fun j : Fin S =>
    if c * BN ≤ j.val ∧ j.val < (c + 1) * BN then some (g j) else none)

/-- Generic ⊥-seeded running `(max, l, acc)` after streaming `[0, hi)`. -/
noncomputable def gStateBot (S hi : Nat) (g : Fin S → ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  (gKeysUpto S hi g).foldl osStepBot (⊥, 0, 0)

/-- Generic ⊥-seeded running max after streaming `[0, hi)`. -/
noncomputable def gRunningMax (S hi : Nat) (g : Fin S → ℝ × ℝ) : WithBot ℝ :=
  ((gKeysUpto S hi g).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥

/-- The running max component of `gStateBot` is the `WithBot ⊔`-fold `gRunningMax`. -/
theorem gStateBot_fst_eq_runningMax (S hi : Nat) (g : Fin S → ℝ × ℝ) :
    (gStateBot S hi g).1 = gRunningMax S hi g := by
  rw [gStateBot, osStepBot_foldl_fst, gRunningMax, foldl_sup_bot_eq_foldr]

/-- Threshold-split for a `.val`-ascending `Fin` list. -/
private theorem g_filterMap_window_split {n : Nat} (l : List (Fin n))
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
      rw [if_neg hnb, if_pos (lt_of_lt_of_le hlt hle), if_pos hlt]; rfl
    · have hge : t ≤ a.val := Nat.not_lt.mp hlt
      have htail_prefix : tl.filterMap (fun j => if j.val < t then some (g j) else none) = [] := by
        apply List.filterMap_eq_nil_iff.mpr
        intro b hb
        have hab : a.val < b.val := hahead b hb
        have hbt : ¬ (b.val < t) := by omega
        simp [hbt]
      rw [ih htl, htail_prefix, if_neg hlt]
      by_cases h2 : a.val < hi₂
      · rw [if_pos h2, if_pos (And.intro hge h2 : t ≤ a.val ∧ a.val < hi₂)]; rfl
      · rw [if_neg h2, if_neg (fun h : t ≤ a.val ∧ a.val < hi₂ => h2 h.2)]

/-- **Window split** (`hi = c·BN`) for the generic key list. -/
theorem gKeysUpto_succ (S BN c : Nat) (g : Fin S → ℝ × ℝ) :
    gKeysUpto S ((c + 1) * BN) g = gKeysUpto S (c * BN) g ++ gBlock S BN c g := by
  unfold gKeysUpto gBlock
  rw [g_filterMap_window_split (List.finRange S) (List.pairwise_lt_finRange S)
    (c * BN) ((c + 1) * BN) g (by nlinarith [Nat.zero_le BN])]

/-- **One-block advance** of the generic ⊥-seeded state. -/
theorem gStateBot_succ (S BN c : Nat) (g : Fin S → ℝ × ℝ) :
    gStateBot S ((c + 1) * BN) g
      = (gBlock S BN c g).foldl osStepBot (gStateBot S (c * BN) g) := by
  unfold gStateBot; rw [gKeysUpto_succ, List.foldl_append]

/-- The running max of an `osStepBot` fold over `block` from `(m, l, acc)` is
`m ⊔ (block max)`. -/
theorem osStepBot_block_fst (m : WithBot ℝ) (l acc : ℝ) (block : List (ℝ × ℝ)) :
    (block.foldl osStepBot (m, l, acc)).1
      = m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  rw [osStepBot_foldl_fst]
  induction block generalizing m with
  | nil => simp
  | cons a t ih =>
    simp only [List.map_cons, List.foldl_cons, List.foldr_cons]
    rw [ih]
    rw [show (m ⊔ ((a.1 : ℝ) : WithBot ℝ)) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) t))
          = m ⊔ (((a.1 : ℝ) : WithBot ℝ) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) t))) from by
      rw [sup_assoc]]

/-- **The block-at-once update equals the key-by-key `osStepBot` fold.** -/
theorem osStepBot_block_eq (m : WithBot ℝ) (l acc T L : ℝ) (block : List (ℝ × ℝ))
    (hl : l = (m.elim 0 (fun r => pow2 (-r))) * L)
    (hacc : acc = (m.elim 0 (fun r => pow2 (-r))) * T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0) :
    let M' := m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
    (M',
     l * (WithBot.realExp2 (WithBot.realSub m M')).unbotD 0
       + (block.map (fun p => pow2 (p.1 - M'.unbotD 0))).sum,
     acc * (WithBot.realExp2 (WithBot.realSub m M')).unbotD 0
       + (block.map (fun p => pow2 (p.1 - M'.unbotD 0) * p.2)).sum)
      = block.foldl osStepBot (m, l, acc) := by
  intro M'
  have hfst : (block.foldl osStepBot (m, l, acc)).1 = M' := by
    rw [osStepBot_block_fst]
  obtain ⟨hfold_l, hfold_acc⟩ := osStepBot_foldl_consistent block m l acc T L hl hacc hmL hmT
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
    have hlα : l * (WithBot.realExp2 (WithBot.realSub m (↑Mr : WithBot ℝ))).unbotD 0 = pow2 (-Mr) * L := by
      cases hm : m with
      | bot =>
        rw [hl, hm, show ((⊥ : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = 0 from rfl,
          zero_mul, hmL hm]; ring
      | coe a =>
        rw [hl, hm, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe,
          show Real.exp ((a - Mr) * Real.log 2) = pow2 (a - Mr) from by simp [pow2, mul_comm]]
        rw [mul_right_comm, ← pow2_add]; ring_nf
    have haccα : acc * (WithBot.realExp2 (WithBot.realSub m (↑Mr : WithBot ℝ))).unbotD 0 = pow2 (-Mr) * T := by
      cases hm : m with
      | bot =>
        rw [hacc, hm, show ((⊥ : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = 0 from rfl,
          zero_mul, hmT hm]; ring
      | coe a =>
        rw [hacc, hm, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe,
          show Real.exp ((a - Mr) * Real.log 2) = pow2 (a - Mr) from by simp [pow2, mul_comm]]
        rw [mul_right_comm, ← pow2_add]; ring_nf
    have hsumL : (block.map (fun p => pow2 (p.1 - (↑Mr : WithBot ℝ).unbotD 0))).sum
        = pow2 (-Mr) * (block.map (fun p => pow2 p.1)).sum := by
      have := sum_map_pow2_sub ((↑Mr : WithBot ℝ).unbotD 0) block (fun _ => 1)
      simp only [mul_one] at this
      rw [this, WithBot.unbotD_coe]
    have hsumT : (block.map (fun p => pow2 (p.1 - (↑Mr : WithBot ℝ).unbotD 0) * p.2)).sum
        = pow2 (-Mr) * (block.map (fun p => pow2 p.1 * p.2)).sum := by
      rw [sum_map_pow2_sub ((↑Mr : WithBot ℝ).unbotD 0) block (fun p => p.2), WithBot.unbotD_coe]
    refine Prod.ext hfst.symm (Prod.ext ?_ ?_)
    · rw [hfold_l, hlα, hsumL, show ((↑Mr : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-Mr) from rfl]; ring
    · rw [hfold_acc, haccα, hsumT, show ((↑Mr : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-Mr) from rfl]; ring

/-- filterMap-sum over `Fin n` with a guard collapses into the masked `Finset.sum`. -/
theorem g_filterMap_finRange_sum {α : Type*} (n : Nat)
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

/-- The `WithBot` `foldr` of a filtered score list (coerced) equals the `Finset.sup`. -/
theorem g_filterMap_foldr_sup (n : Nat) (P : Fin n → Prop) [DecidablePred P]
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

/-- Block-local lane `jL : Fin BN` maps to a global key `c·BN + jL < S`. -/
theorem gBlock_idx_lt (S BN c : Nat) (hwin : (c + 1) * BN ≤ S) (jL : Fin BN) :
    c * BN + jL.val < S := by
  have hjlt := jL.isLt
  have heq : (c + 1) * BN = c * BN + BN := by ring
  omega

/-- Reindex a windowed `Finset.sup` over `Fin S` onto `Fin BN`. -/
theorem g_window_sup_reindex (BN c S : Nat) (hwin : (c + 1) * BN ≤ S)
    (F : Nat → WithBot ℝ) :
    Finset.univ.sup (fun j : Fin S =>
        if c * BN ≤ j.val ∧ j.val < (c + 1) * BN then F j.val else ⊥)
      = Finset.univ.sup (fun jL : Fin BN => F (c * BN + jL.val)) := by
  have hmul : (c + 1) * BN = c * BN + BN := by ring
  apply le_antisymm
  · apply Finset.sup_le
    intro j _
    by_cases hj : c * BN ≤ j.val ∧ j.val < (c + 1) * BN
    · rw [if_pos hj]
      have hjL : j.val - c * BN < BN := by omega
      refine le_trans ?_ (Finset.le_sup (f := fun jL : Fin BN => F (c * BN + jL.val))
        (Finset.mem_univ (⟨j.val - c * BN, hjL⟩ : Fin BN)))
      simp only
      rw [show c * BN + (j.val - c * BN) = j.val from by omega]
    · rw [if_neg hj]; exact bot_le
  · apply Finset.sup_le
    intro jL _
    have hb : c * BN + jL.val < S := by have := jL.isLt; omega
    refine le_trans ?_ (Finset.le_sup
      (f := fun j : Fin S => if c * BN ≤ j.val ∧ j.val < (c + 1) * BN then F j.val else ⊥)
      (Finset.mem_univ (⟨c * BN + jL.val, hb⟩ : Fin S)))
    simp only
    rw [if_pos (by have := jL.isLt; exact ⟨by omega, by omega⟩)]

/-- Reindex a masked `Fin S`-window sum onto `Fin BN`. -/
theorem g_window_sum_reindex (BN c S : Nat) (hwin : (c + 1) * BN ≤ S) (g : Nat → ℝ) :
    (∑ j : Fin S, if c * BN ≤ j.val ∧ j.val < (c + 1) * BN then g j.val else 0)
      = ∑ jL : Fin BN, g (c * BN + jL.val) := by
  have hmul : (c + 1) * BN = c * BN + BN := by ring
  rw [← Finset.sum_filter]
  symm
  refine Finset.sum_bij (i := fun jL _ => (⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩ : Fin S))
    ?_ ?_ ?_ ?_
  · intro jL _; simp only [Finset.mem_filter, Finset.mem_univ, true_and]; have := jL.isLt; omega
  · intro a _ b _ hab
    apply Fin.ext
    have : c * BN + a.val = c * BN + b.val := by simpa using congrArg Fin.val hab
    omega
  · intro j hj
    have hj2 : c * BN ≤ j.val ∧ j.val < (c + 1) * BN := (Finset.mem_filter.mp hj).2
    exact ⟨⟨j.val - c * BN, by omega⟩, Finset.mem_univ _, by apply Fin.ext; simp only; omega⟩
  · intro jL _; rfl

/-- The generic block list reindexes onto `Fin BN` (key `c·BN + jL`). -/
theorem gBlock_map_sum (S BN c : Nat) (g : Fin S → ℝ × ℝ)
    (hwin : (c + 1) * BN ≤ S) (h : ℝ × ℝ → ℝ) :
    ((gBlock S BN c g).map h).sum
      = ∑ jL : Fin BN, h (g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩) := by
  rw [gBlock, g_filterMap_finRange_sum S
    (fun j => c * BN ≤ j.val ∧ j.val < (c + 1) * BN) g h]
  rw [show (∑ j : Fin S, if c * BN ≤ j.val ∧ j.val < (c + 1) * BN then h (g j) else 0)
        = ∑ j : Fin S, if c * BN ≤ j.val ∧ j.val < (c + 1) * BN
            then (fun jg => if h' : jg < S then h (g ⟨jg, h'⟩) else 0) j.val else 0 from by
    apply Finset.sum_congr rfl; intro j _
    by_cases hw : c * BN ≤ j.val ∧ j.val < (c + 1) * BN
    · rw [if_pos hw, if_pos hw]; simp only [dif_pos j.isLt]
    · rw [if_neg hw, if_neg hw]]
  rw [g_window_sum_reindex BN c S hwin
    (fun jg => if h' : jg < S then h (g ⟨jg, h'⟩) else 0)]
  apply Finset.sum_congr rfl
  intro jL _
  simp only [dif_pos (gBlock_idx_lt S BN c hwin jL)]

/-- **One-block advance** of the generic ⊥-seeded running max. -/
theorem gRunningMax_succ (S BN c : Nat) (g : Fin S → ℝ × ℝ) (hwin : (c + 1) * BN ≤ S) :
    gRunningMax S ((c + 1) * BN) g
      = gRunningMax S (c * BN) g
        ⊔ Finset.univ.sup (fun jL : Fin BN =>
            ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 : WithBot ℝ)) := by
  unfold gRunningMax
  rw [gKeysUpto_succ, List.map_append, List.foldr_append]
  have hblock : ((gBlock S BN c g).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = Finset.univ.sup (fun jL : Fin BN =>
          ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 : WithBot ℝ)) := by
    rw [show (gBlock S BN c g).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
          = ((List.finRange S).filterMap (fun j : Fin S =>
              if c * BN ≤ j.val ∧ j.val < (c + 1) * BN
              then some ((g j).1) else none)).map (fun x : ℝ => ((x : ℝ) : WithBot ℝ)) from by
      unfold gBlock
      rw [List.map_filterMap, List.map_filterMap]
      apply List.filterMap_congr
      intro j _
      by_cases hj : c * BN ≤ j.val ∧ j.val < (c + 1) * BN <;> simp [hj]]
    rw [g_filterMap_foldr_sup S
      (fun j => c * BN ≤ j.val ∧ j.val < (c + 1) * BN) (fun j => (g j).1)]
    classical
    rw [show (Finset.univ.sup (fun j : Fin S =>
          if c * BN ≤ j.val ∧ j.val < (c + 1) * BN
          then (((g j).1 : ℝ) : WithBot ℝ) else ⊥))
        = Finset.univ.sup (fun j : Fin S =>
            if c * BN ≤ j.val ∧ j.val < (c + 1) * BN
            then (fun jg => if h : jg < S then (((g ⟨jg, h⟩).1 : ℝ) : WithBot ℝ) else ⊥) j.val else ⊥)
        from by
      apply Finset.sup_congr rfl
      intro j _
      by_cases hw : c * BN ≤ j.val ∧ j.val < (c + 1) * BN
      · rw [if_pos hw, if_pos hw]; simp only [dif_pos j.isLt]
      · rw [if_neg hw, if_neg hw]]
    rw [g_window_sup_reindex BN c S hwin
      (fun jg => if h : jg < S then (((g ⟨jg, h⟩).1 : ℝ) : WithBot ℝ) else ⊥)]
    apply Finset.sup_congr rfl
    intro jL _
    have hb : c * BN + jL.val < S := by
      have hjlt := jL.isLt; have heq : (c + 1) * BN = c * BN + BN := by ring
      omega
    simp only [dif_pos hb]
  rw [hblock]
  generalize (gKeysUpto S (c * BN) g).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) = preL
  induction preL with
  | nil => simp
  | cons a t ih => simp only [List.foldr_cons, ih]; rw [sup_assoc]

/-- **Generic ⊥-seeded consistency**: the running `(l, acc)` of `gStateBot` are
`κ(runningMax)` times the max-free reference sums. -/
theorem gStateBot_consistent (S hi : Nat) (g : Fin S → ℝ × ℝ) :
    (gStateBot S hi g).2.1
        = ((gStateBot S hi g).1.elim 0 (fun r => pow2 (-r)))
          * ((gKeysUpto S hi g).map (fun p => pow2 p.1)).sum ∧
    (gStateBot S hi g).2.2
        = ((gStateBot S hi g).1.elim 0 (fun r => pow2 (-r)))
          * ((gKeysUpto S hi g).map (fun p => pow2 p.1 * p.2)).sum := by
  obtain ⟨hL, hT⟩ := osStepBot_foldl_consistent (gKeysUpto S hi g) ⊥ 0 0 0 0
    (by simp) (by simp) (fun _ => rfl) (fun _ => rfl)
  refine ⟨?_, ?_⟩
  · rw [gStateBot]; rw [show (List.foldl osStepBot (⊥, 0, 0) (gKeysUpto S hi g)).2.1 = _ from hL,
      zero_add]
  · rw [gStateBot]; rw [show (List.foldl osStepBot (⊥, 0, 0) (gKeysUpto S hi g)).2.2 = _ from hT,
      zero_add]

theorem gKeysUpto_map_sum_eq_zero_of_bot (S hi : Nat) (g : Fin S → ℝ × ℝ)
    (hbot : (gStateBot S hi g).1 = ⊥) (f : ℝ × ℝ → ℝ) :
    ((gKeysUpto S hi g).map f).sum = 0 := by
  rw [gStateBot_fst_eq_runningMax, gRunningMax] at hbot
  have hnil : gKeysUpto S hi g = [] := by
    rcases hk : gKeysUpto S hi g with _ | ⟨a, t⟩
    · rfl
    · exfalso
      have : ((a.1 : ℝ) : WithBot ℝ) ≤ ((gKeysUpto S hi g).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ :=
        mem_le_foldr_sup _ _ (by rw [hk]; exact List.mem_cons_self ..)
      rw [hbot] at this
      exact absurd (le_bot_iff.mp this) WithBot.coe_ne_bot
  rw [hnil]; simp

/-- After `c+1` blocks the ⊥-seeded running max is finite. -/
theorem gRunningMax_succ_ne_bot (S BN c : Nat) (g : Fin S → ℝ × ℝ)
    (hBN : 0 < BN) (hwin : (c + 1) * BN ≤ S) :
    gRunningMax S ((c + 1) * BN) g ≠ ⊥ := by
  rw [gRunningMax_succ S BN c g hwin]
  intro h
  rw [max_eq_bot] at h
  have hle := Finset.le_sup (f := fun jL : Fin BN =>
      ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 : WithBot ℝ))
      (Finset.mem_univ (⟨0, hBN⟩ : Fin BN))
  rw [h.2] at hle
  exact absurd (le_bot_iff.mp hle) WithBot.coe_ne_bot

/-- The running max / denominator of `gStateBot` depend only on the per-key scores. -/
theorem gStateBot_score_congr (S hi : Nat) (g1 g2 : Fin S → ℝ × ℝ)
    (h : ∀ j : Fin S, (g1 j).1 = (g2 j).1) :
    (gStateBot S hi g1).1 = (gStateBot S hi g2).1
      ∧ (gStateBot S hi g1).2.1 = (gStateBot S hi g2).2.1 := by
  have hkeys : (gKeysUpto S hi g1).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
      = (gKeysUpto S hi g2).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) := by
    unfold gKeysUpto
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr; intro j _
    by_cases hj : j.val < hi <;> simp [hj, h j]
  have hkeys2 : (gKeysUpto S hi g1).map (fun p => pow2 p.1)
      = (gKeysUpto S hi g2).map (fun p => pow2 p.1) := by
    unfold gKeysUpto
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr; intro j _
    by_cases hj : j.val < hi <;> simp [hj, h j]
  have hfst : (gStateBot S hi g1).1 = (gStateBot S hi g2).1 := by
    rw [gStateBot_fst_eq_runningMax, gStateBot_fst_eq_runningMax, gRunningMax, gRunningMax, hkeys]
  refine ⟨hfst, ?_⟩
  rw [(gStateBot_consistent S hi g1).1, (gStateBot_consistent S hi g2).1, hfst, hkeys2]

/-- **Block-step in explicit `Fin BN` form** (BN-parametric). -/
theorem gStateBot_succ_explicit (S BN c : Nat) (g : Fin S → ℝ × ℝ) (hwin : (c + 1) * BN ≤ S) :
    let st := gStateBot S (c * BN) g
    let M' := st.1 ⊔ Finset.univ.sup (fun jL : Fin BN =>
        ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 : WithBot ℝ))
    gStateBot S ((c + 1) * BN) g
      = (M',
         st.2.1 * (WithBot.realExp2 (WithBot.realSub st.1 M')).unbotD 0
           + Finset.univ.sum (fun jL : Fin BN =>
               pow2 ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 - M'.unbotD 0)),
         st.2.2 * (WithBot.realExp2 (WithBot.realSub st.1 M')).unbotD 0
           + Finset.univ.sum (fun jL : Fin BN =>
               pow2 ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 - M'.unbotD 0)
                 * (g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).2)) := by
  intro st M'
  obtain ⟨hLc, hTc⟩ := gStateBot_consistent S (c * BN) g
  have hMblock : M' = st.1 ⊔ ((gBlock S BN c g).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    have hsup : ((gBlock S BN c g).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
        = Finset.univ.sup (fun jL : Fin BN =>
            ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 : WithBot ℝ)) := by
      rw [show (gBlock S BN c g).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
            = ((List.finRange S).filterMap (fun j : Fin S =>
                if c * BN ≤ j.val ∧ j.val < (c + 1) * BN
                then some ((g j).1) else none)).map (fun x : ℝ => ((x : ℝ) : WithBot ℝ)) from by
        unfold gBlock
        rw [List.map_filterMap, List.map_filterMap]
        apply List.filterMap_congr
        intro j _
        by_cases hj : c * BN ≤ j.val ∧ j.val < (c + 1) * BN <;> simp [hj]]
      rw [g_filterMap_foldr_sup S (fun j => c * BN ≤ j.val ∧ j.val < (c + 1) * BN) (fun j => (g j).1)]
      classical
      rw [show (Finset.univ.sup (fun j : Fin S =>
            if c * BN ≤ j.val ∧ j.val < (c + 1) * BN then (((g j).1 : ℝ) : WithBot ℝ) else ⊥))
          = Finset.univ.sup (fun j : Fin S =>
              if c * BN ≤ j.val ∧ j.val < (c + 1) * BN
              then (fun jg => if h : jg < S then (((g ⟨jg, h⟩).1 : ℝ) : WithBot ℝ) else ⊥) j.val else ⊥)
          from by
        apply Finset.sup_congr rfl
        intro j _
        by_cases hw : c * BN ≤ j.val ∧ j.val < (c + 1) * BN
        · rw [if_pos hw, if_pos hw]; simp only [dif_pos j.isLt]
        · rw [if_neg hw, if_neg hw]]
      rw [g_window_sup_reindex BN c S hwin
        (fun jg => if h : jg < S then (((g ⟨jg, h⟩).1 : ℝ) : WithBot ℝ) else ⊥)]
      apply Finset.sup_congr rfl
      intro jL _
      simp only [dif_pos (gBlock_idx_lt S BN c hwin jL)]
    show _ = _
    rw [hsup]
  have hstep := osStepBot_block_eq st.1 st.2.1 st.2.2
    (((gKeysUpto S (c * BN) g).map (fun p => pow2 p.1 * p.2)).sum)
    (((gKeysUpto S (c * BN) g).map (fun p => pow2 p.1)).sum)
    (gBlock S BN c g) hLc hTc
    (fun hb => gKeysUpto_map_sum_eq_zero_of_bot S (c * BN) g hb _)
    (fun hb => gKeysUpto_map_sum_eq_zero_of_bot S (c * BN) g hb _)
  have hfold : (gBlock S BN c g).foldl osStepBot st = gStateBot S ((c + 1) * BN) g :=
    (gStateBot_succ S BN c g).symm
  rw [hfold] at hstep
  simp only [] at hstep
  rw [← hMblock] at hstep
  rw [show ((gBlock S BN c g).map (fun p => pow2 (p.1 - M'.unbotD 0))).sum
        = Finset.univ.sum (fun jL : Fin BN =>
            pow2 ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 - M'.unbotD 0))
      from gBlock_map_sum S BN c g hwin (fun p => pow2 (p.1 - M'.unbotD 0))] at hstep
  rw [show ((gBlock S BN c g).map (fun p => pow2 (p.1 - M'.unbotD 0) * p.2)).sum
        = Finset.univ.sum (fun jL : Fin BN =>
            pow2 ((g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).1 - M'.unbotD 0)
              * (g ⟨c * BN + jL.val, gBlock_idx_lt S BN c hwin jL⟩).2)
      from gBlock_map_sum S BN c g hwin (fun p => pow2 (p.1 - M'.unbotD 0) * p.2)] at hstep
  exact hstep.symm

/-! ### Tile-cell bridges (shape-generic) -/

theorem ctxg_reduceMaxDrop_data_row {M N : Nat} (hN : 0 < N) (qk : Tile .real [M, N])
    (rmaxT : Tile .real [M]) (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [M,N].length) qk = some rmaxT)
    (r : Fin M) (g : Fin N → WithBot ℝ) (hqk : ∀ jL : Fin N, qk.data (r, jL, PUnit.unit) = g jL) :
    rmaxT.data (r, PUnit.unit) = Finset.univ.sup g := by
  unfold Tile.reduceMaxDrop at hrm
  rw [dif_pos (show 0 < TileShape.axisDim [M,N] (⟨1, by simp⟩ : Fin [M,N].length) from hN)] at hrm
  rw [← Option.some.inj hrm]
  simp only [Finset.sup'_eq_sup]
  exact Finset.sup_congr rfl (fun jL _ => hqk jL)

/-- `WithBot.realExp2` of a `some`-cell is `some (pow2 …)`. -/
theorem ctxg_exp2_some {M N : Nat} (h : Fin M → Fin N → ℝ) (x : Tile .real [M, N])
    (r : Fin M) (jL : Fin N) (hx : x.data (r, jL, PUnit.unit) = some (h r jL)) :
    (Tile.uop WithBot.realExp2 x).data (r, jL, PUnit.unit) = some (pow2 (h r jL)) := by
  show WithBot.realExp2 (x.data (r, jL, PUnit.unit)) = _
  rw [hx]; simp [WithBot.realExp2, pow2, mul_comm]

/-- A `WithBot ℝ` sum of `some`-valued cells is `some` of the real sum. -/
theorem ctxg_withBot_sum_some {N : Nat} (g : Fin N → ℝ) :
    @Finset.sum (Fin N) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) :=
  (WithBot.coe_sum Finset.univ g).symm

/-- `realExp2` is total (never `⊥`). -/
theorem ctxg_realExp2_eq_some_unbotD (z : WithBot ℝ) :
    WithBot.realExp2 z = some ((WithBot.realExp2 z).unbotD 0) := by
  cases z <;> rfl

/-- `m_ij = select(m_i > rmax) m_i rmax` collapses to `max` in `WithBot ℝ`. -/
theorem ctxg_mij_max {M : Nat} (m_i rmaxT : Tile .real [M]) (r : Fin M)
    (a b : WithBot ℝ) (hmi : m_i.data (r, PUnit.unit) = a) (hrm : rmaxT.data (r, PUnit.unit) = b) :
    (Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame m_i rmaxT) m_i rmaxT).data
        (r, PUnit.unit) = max a b := by
  rw [Tile.select_data, Tile.cop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hmi, hrm]
  by_cases h : a ≤ b
  · rw [if_neg (by simp [not_lt.mpr h]), max_eq_right h]
  · rw [if_pos (by simpa using not_le.mp h), max_eq_left (le_of_lt (not_le.mp h))]

/-- A `dot` row over all `K` keys when both factors are all-`some`. -/
theorem ctxg_dot_row {M K N : Nat} (p : Tile .real [M, K]) (v : Tile .real [K, N])
    (r : Fin M) (d : Fin N) (fp fv : Fin K → ℝ)
    (hp : ∀ jL : Fin K, p.data (r, jL, PUnit.unit) = some (fp jL))
    (hv : ∀ jL : Fin K, v.data (jL, d, PUnit.unit) = some (fv jL)) :
    (Tile.dot [] p v).data (r, d, PUnit.unit) = some (Finset.univ.sum fun jL : Fin K => fp jL * fv jL) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·) (p.data (r, k, PUnit.unit)) (v.data (k, d, PUnit.unit))))
      = @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ (fun k => (some (fp k * fv k) : WithBot ℝ))
      from Finset.sum_congr rfl (fun k _ => by rw [hp k, hv k]; rfl)]
  exact ctxg_withBot_sum_some _

/-- The `qk -= m_ij[:, None]` cell readback (avoids deep recursion inline). -/
theorem ctx_qk_sub_mij_cell {BM BN : Nat} (qkT : Tile .real [BM, BN]) (mijT : Tile .real [BM])
    (i : Fin BM) (jL : Fin BN) (sc mij : ℝ)
    (hqk : qkT.data (i, jL, PUnit.unit) = some sc)
    (hmij : mijT.data (i, PUnit.unit) = some mij) :
    (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame qkT
        (Tile.expandDim ⟨1, by simp⟩ mijT)).data (i, jL, PUnit.unit)
      = some (sc - mij) := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
    TileShape.dropInsertedIndex, TileShape.insertAxisIndex, hqk, hmij,
    NumericDType.sub, WithBot.realSub, Option.map₂, Option.bind, Option.map]

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
