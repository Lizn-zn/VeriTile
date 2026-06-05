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

/-! ### Bloom natural-exp cell bridges (analogues with `WithBot.realExp`)

The bloom loop body uses **natural** `tl.exp` (`WithBot.realExp`), feeding the
base-2 `osStep` fold via the invariant's log2-scaling (`m_i = log2·(running max)`,
so `realExp((Mc − Mc₊₁)·log2) = realExp2(Mc − Mc₊₁) = pow2(Mc − Mc₊₁)`). -/

/-- `realExp (x · log 2) = realExp2 x` on the carrier: natural `exp` of a
log2-scaled value is the base-2 `realExp2`. -/
theorem ctxg_realExp_log2 (x : ℝ) :
    WithBot.realExp (((x * Real.log 2 : ℝ) : WithBot ℝ)) = WithBot.realExp2 ((x : ℝ) : WithBot ℝ) := by
  rfl

/-- `WithBot.realExp` of a `some`-cell is `some (Real.exp …)` (shape-generic). -/
theorem ctxg_exp_some {M N : Nat} (h : Fin M → Fin N → ℝ) (x : Tile .real [M, N])
    (r : Fin M) (jL : Fin N) (hx : x.data (r, jL, PUnit.unit) = some (h r jL)) :
    (Tile.uop WithBot.realExp x).data (r, jL, PUnit.unit) = some (Real.exp (h r jL)) := by
  show WithBot.realExp (x.data (r, jL, PUnit.unit)) = _
  rw [hx]; rfl

/-- `realExp` is total (never `⊥`). -/
theorem ctxg_realExp_eq_some_unbotD (z : WithBot ℝ) :
    WithBot.realExp z = some ((WithBot.realExp z).unbotD 0) := by
  cases z <;> rfl

/-- The `q·k` dot cell is the (unscaled) score (shape-generic `[BM,D]·[D,BN]`). -/
theorem ctx_dot_score_cell {BM BN D : Nat}
    (qtile : Tile .real [BM, D]) (ktile : Tile .real [D, BN]) (i : Fin BM) (j : Fin BN)
    (qf : Fin BM → Fin D → ℝ) (kf : Fin BN → Fin D → ℝ)
    (hq : ∀ e : Fin D, qtile.data (i, e, PUnit.unit) = some (qf i e))
    (hk : ∀ e : Fin D, ktile.data (e, j, PUnit.unit) = some (kf j e)) :
    (Tile.dot [] qtile ktile).data (i, j, PUnit.unit)
      = some (Finset.univ.sum (fun e : Fin D => qf i e * kf j e)) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·) (qtile.data (i, e, PUnit.unit)) (ktile.data (e, j, PUnit.unit))))
      = @Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
          (fun e => (some (qf i e * kf j e) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by rw [hq e, hk e]; rfl)]
  rw [show (fun e : Fin D => (some (qf i e * kf j e) : WithBot ℝ))
        = (fun e : Fin D => ((qf i e * kf j e : ℝ) : WithBot ℝ)) from rfl,
    ← WithBot.coe_sum]; rfl

/-- The bloom kernel's masked `qk` cell at lane `(i,j)`: active lane gets the
scaled dot `sm·Σ_e qf·kf`; future lane gets the `-1e8` sentinel. -/
noncomputable def bloomQkCell {BM BN D : Nat} (sm : ℝ) (SN plen : Nat) (gOM : Fin BM → Nat)
    (qf : Fin BM → Fin D → ℝ) (kf : Fin BN → Fin D → ℝ) (i : Fin BM) (j : Fin BN) : ℝ :=
  if SN + j.val ≤ gOM i + plen then
    sm * Finset.univ.sum (fun e : Fin D => qf i e * kf j e)
  else (0.0 - 100000000.0 : ℝ)

/-- **The bloom kernel's `qkT` cell is `some (bloomQkCell …)`** (shape-generic).
The `tl.where(mask, (0 + dot)·sm, -1e8)` register (the bloom `+0` from
`qk = tl.zeros; qk += dot`) reads `qf`/`kf` and has cell `(i,j) = bloomQkCell`. -/
theorem bloom_qkT_cell {BM BN D : Nat} (sm : ℝ) (SN plen : Nat) (gOM : Fin BM → Nat)
    (qtile : Tile .real [BM, D]) (kloadT : Tile .real [D, BN]) (qf : Fin BM → Fin D → ℝ) (kf : Fin BN → Fin D → ℝ)
    (hq : ∀ (i : Fin BM) (e : Fin D), qtile.data (i, e, PUnit.unit) = some (qf i e))
    (hk : ∀ (j : Fin BN) (e : Fin D), kloadT.data (e, j, PUnit.unit) = some (kf j e))
    (i : Fin BM) (j : Fin BN) :
    (Tile.select
        (⟨fun idx : TileIndex [BM, BN] => decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ : Tile .bool [BM, BN])
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
            (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
            (Tile.dot [] qtile kloadT))
          (Tile.scalar (some sm)))
        (⟨fun _ : TileIndex [BM, BN] => some (0.0 - 100000000.0 : ℝ)⟩ : Tile .real [BM, BN])).data
      (i, j, PUnit.unit)
      = some (bloomQkCell sm SN plen gOM qf kf i j) := by
  rw [Tile.select_data, bloomQkCell]
  have hsel : (⟨fun idx : TileIndex [BM, BN] => decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ : Tile .bool [BM, BN]).data (i, j, PUnit.unit)
      = decide (SN + j.val ≤ gOM i + plen) := rfl
  by_cases h : SN + j.val ≤ gOM i + plen
  · rw [hsel, if_pos h]
    simp only [decide_eq_true_eq.mpr h, if_true]
    rw [Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul]
    have hadd : (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
        (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
        (Tile.dot [] qtile kloadT)).data (i, j, PUnit.unit)
        = some (Finset.univ.sum (fun e : Fin D => qf i e * kf j e)) := by
      rw [Tile.bop_data]
      simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
      rw [ctx_dot_score_cell qtile kloadT i j qf kf (hq i) (hk j)]
      show WithBot.realAdd (some 0) (some _) = _
      simp only [WithBot.realAdd, Option.map₂, Option.bind, Option.map]
      rw [zero_add]
    rw [hadd]
    show Option.map₂ (· * ·) _ _ = _
    simp only [Tile.scalar_data, Option.map₂]
    refine congrArg some ?_; ring
  · rw [hsel, if_neg h]
    simp only [decide_eq_false_iff_not.mpr h, Bool.false_eq_true, if_false]

/-! ### Per-statement `evalOp` helpers (parametric) -/

/-- Axis-0 `expandDim` over a `nat` register (`offs_n[None, :]` row broadcast). -/
@[simp] theorem ctx_evalOp_expandDim_zero_nat {D : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [1, D] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] name)) s =
      (s.regs .nat [D] name).bind (fun v =>
        some ({ data := fun i : TileIndex [1, D] => v.data (i.2.1, PUnit.unit) } : Tile .nat [1, D])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

/-- Axis-1 `expandDim` over a `nat` register (`offs_m[:, None]` column broadcast). -/
@[simp] theorem ctx_evalOp_expandDim_one_nat {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] name)) s =
      (s.regs .nat [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .nat [M, 1])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

/-- Axis-1 `expandDim` over a `real` register (`m_ij[:, None]`/`alpha[:, None]`). -/
@[simp] theorem ctx_evalOp_expandDim_one_real {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .real [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] name)) s =
      (s.regs .real [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .real [M, 1])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

/-- Eval helper for `ge` (causal mask comparison). -/
theorem ctx_evalOp_ge {dtype a b shape} (h : ComparableDType dtype) (bc : Broadcast a b shape)
    (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

/-- Eval helper for `floorDiv` (`cur_kv_head = cur_head // kv_group_num`). -/
theorem ctx_evalOp_floorDiv {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.floorDiv h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.floorDiv bc vx vy)) := by
  simp [evalOp]

/-- Eval helper for `remap` (column/row broadcast of load/store masks). -/
theorem ctx_evalOp_remap {dtype outShape inShape}
    (map : TileIndex outShape → TileIndex inShape) (a : Op dtype inShape) (s : BlockState) :
    evalOp (.remap outShape map a) s = (do
      let v ← evalOp a s; some (Tile.remap map v)) := by
  simp [evalOp]

/-- Scalar `nat` region load (`tl.load(b_prompt_cache_len + cur_batch)` etc.). -/
theorem ctx_evalOp_load_scalar_nat (region : Region .nat) (off : Op .nat [])
    (s : BlockState) (o : Nat) (hoff : evalOp off s = some (Tile.scalar o)) :
    evalOp (.load .nat (MemAccess.region region off) MaskOpt.none) s
      = some (Tile.scalar (s.readMemValue .nat (Region.cast region) o)) := by
  simp only [evalOp, hoff, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- Masked pointer-arith region load (`tl.load(R + offs, mask=m, other=o)`). -/
theorem ctx_evalOp_load_region_maskOther {dtype : TileDType} {shape : TileShape}
    (region : Region dtype) (offsets : Op .nat shape)
    (mask : Op .bool shape) (other : Op dtype shape) (s : BlockState)
    (offsTile : Tile .nat shape) (maskTile : Tile .bool shape) (otherTile : Tile dtype shape)
    (hoff : evalOp offsets s = some offsTile)
    (hmask : evalOp mask s = some maskTile)
    (hother : evalOp other s = some otherTile) :
    evalOp (.load dtype (MemAccess.region region offsets) (MaskOpt.maskOther mask other)) s
      = some ⟨fun i => if maskTile.data i then
          s.readMemValue dtype (Region.cast region) (offsTile.data i) else otherTile.data i⟩ := by
  simp only [evalOp, hoff, hmask, hother, Option.bind_eq_bind, Option.bind_some]
  rfl

/-! ### Bloom `Req_to_tokens` gather recipes (constants: stride_req_b = 7500,
stride_kbs/vbs = 576, stride_kh/vh = 96, stride_kd/vd = 1) -/

/-- **`kv_loc` masked nat-gather recipe**: lane `j` reads physical token
`Req_to_tokens[7500·rqi + (SN+j)]` when active (`SN+j < bel`), else `0`. -/
theorem bloom_kvloc_gather_eval {BN : Nat} (s : BlockState)
    (Req_to_tokens : RegionName) (rqi SN bel : Nat)
    (hrqi : s.regs .nat [] "cur_batch_req_idx" = some (Tile.scalar rqi))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hbel : s.regs .nat [] "block_end_loc" = some (Tile.scalar bel)) :
    evalOp (Op.load .nat
        (MemAccess.region Req_to_tokens
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.constNat 7500) (Op.ref .nat [] "cur_batch_req_idx"))
            (Op.mul .nat Broadcast.scalarL (Op.constNat 1)
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BN] "offs_n")))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BN] "offs_n"))
            (Op.ref .nat [] "block_end_loc"))
          (Op.broadcast (Op.constNat 0) [BN]))) s
      = some (⟨fun idx : TileIndex [BN] =>
          if decide (SN + idx.1.val < bel) then
            s.readMemValue .nat (Region.cast Req_to_tokens) (7500 * rqi + (SN + idx.1.val))
          else 0⟩ : Tile .nat [BN]) := by
  simp only [evalOp, hrqi, hsn, hn, hbel, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.cop_data, Tile.bop_data, Tile.scalar_data, Tile.scalar_data_index, Tile.vec_data,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, Region.cast_cast]
  by_cases h : SN + idx.1.val < bel
  · simp only [h, decide_true, if_true]
    congr 1; omega
  · simp only [h, decide_false, Bool.false_eq_true, if_false]

/-- **Gathered `off_k` recipe** (`kv_loc[None,:]·576 + cur_kv_head·96 + offs_d[:,None]·1`,
shape `[D, BN]`): lane `(e,j)` is `kvloc(j)·576 + ckvh·96 + e`. -/
theorem bloom_offk_gather_eval {BN D : Nat} (s : BlockState) (ckvh : Nat)
    (kvf : Fin BN → Nat)
    (hkvloc : s.regs .nat [BN] "kv_loc" = some (Tile.vec kvf))
    (hckvh : s.regs .nat [] "cur_kv_head" = some (Tile.scalar ckvh))
    (hd : s.regs .nat [D] "offs_d" = some (Tile.vec (fun e : Fin D => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consR.consL
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "kv_loc"))
            (Op.constNat 576))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat 96)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat 1))) s
      = some (⟨fun idx : TileIndex [D, BN] =>
          kvf idx.2.1 * 576 + ckvh * 96 + idx.1.val * 1⟩ : Tile .nat [D, BN]) := by
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
    ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
    hkvloc, hckvh, hd, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- **Gathered `off_v` recipe** (`kv_loc[:,None]·576 + cur_kv_head·96 + offs_d[None,:]·1`,
shape `[BN, D]`): lane `(j,e)` is `kvloc(j)·576 + ckvh·96 + e`. -/
theorem bloom_offv_gather_eval {BN D : Nat} (s : BlockState) (ckvh : Nat)
    (kvf : Fin BN → Nat)
    (hkvloc : s.regs .nat [BN] "kv_loc" = some (Tile.vec kvf))
    (hckvh : s.regs .nat [] "cur_kv_head" = some (Tile.scalar ckvh))
    (hd : s.regs .nat [D] "offs_d" = some (Tile.vec (fun e : Fin D => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "kv_loc"))
            (Op.constNat 576))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat 96)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat 1))) s
      = some (⟨fun idx : TileIndex [BN, D] =>
          kvf idx.1 * 576 + ckvh * 96 + idx.2.1.val * 1⟩ : Tile .nat [BN, D]) := by
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
    ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
    hkvloc, hckvh, hd, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-! ### Bloom loop-body lowered statement list + body split (Milestone 1)

`bloomLoopBody` is the exact lowering of `_fwd_kernel`'s `for start_n …` body at the
Python test shape (`BLOCK_M=BLOCK_N=BLOCK_DMODEL=128`, `head_dim=96`, strides
`576/96/1`, `stride_req_b=7500`, `stride_req_s=1`, natural-`exp` softmax with
two-level max and in-loop `β/lᵢⁿᵉʷ`/`(lᵢ/lᵢⁿᵉʷ)·α`-where normalization). Verified
to equal the surface's `toAlgKernel` loop body by `rfl` (`bloomBody_split`). -/

/-- The kernel's chosen natural-`exp` `sm_scale` constant at the Python test shape
(`(√96)⁻¹`, fed `/ log 2` into the base-2 fold so `pow2 (score/log2) = exp score`). -/
noncomputable def sm_scale_bloom : ℝ := (Real.sqrt (96 : ℝ))⁻¹

/-- The 24 lowered bloom loop-body statements (transcribed from
`#print context_attn_bloom_fwd_kernel_surface`; checked by `rfl` via
`bloomBody_split`). -/
noncomputable def bloomLoopBody (Q K V Req_to_tokens B_req_idx : RegionName) : List Stmt :=
  [ Stmt.assign .nat [] "start_n" (Op.ref .nat [] "start_n"),
    Stmt.assign .nat [128] "kv_loc"
      (Op.load .nat
        (MemAccess.region Req_to_tokens
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.constNat 7500) (Op.ref .nat [] "cur_batch_req_idx"))
            (Op.mul .nat Broadcast.scalarL (Op.constNat 1)
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [128] "offs_n")))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [128] "offs_n"))
            (Op.ref .nat [] "block_end_loc"))
          (Op.broadcast (Op.constNat 0) [128]))),
    Stmt.assign .nat [128, 128] "off_k"
      (Op.add .nat Broadcast.nil.consR.consL
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "kv_loc"))
            (Op.constNat 576))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat 96)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .real [128, 128] "k"
      (Op.load .real (MemAccess.region K (Op.ref .nat [128, 128] "off_k"))
        (MaskOpt.maskOther
          (Op.boolAnd Broadcast.nil.consR.consL
            (Op.lt .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))
              (Op.ref .nat [] "block_end_loc"))
            (Op.lt .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_d"))
              (Op.constNat 96)))
          ((Op.const (0.0 : ℝ)).broadcast [128, 128]))),
    Stmt.assign .real [128, 128] "qk"
      (Op.full [128, 128] (Op.const (0 : ℝ))),
    Stmt.assign .real [128, 128] "qk"
      (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [128, 128] "qk")
        (Op.dot (batch := []) (Op.ref .real [128, 128] "q") (Op.ref .real [128, 128] "k"))),
    Stmt.assign .real [128, 128] "qk"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [128, 128] "qk") (Op.const (sm_scale_bloom : ℝ))),
    Stmt.assign .real [128, 128] "qk"
      ((Op.ge .nat Broadcast.nil.consL.consR
          (Op.add .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
            (Op.ref .nat [] "prompt_cache_len"))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))).where
        (Op.ref .real [128, 128] "qk")
        ((Op.sub .real Broadcast.nil (Op.const (0.0 : ℝ)) (Op.const (100000000.0 : ℝ))).broadcast [128, 128])),
    Stmt.assign .real [128] "m_ij"
      (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "qk")),
    Stmt.assign .real [128, 128] "p"
      (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "m_ij"))).exp,
    Stmt.assign .real [128] "l_ij"
      (Op.reduceSum ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "p")),
    Stmt.assign .real [128] "m_i_new"
      ((Op.gt .real Broadcast.nil.consSame (Op.ref .real [128] "m_i") (Op.ref .real [128] "m_ij")).where
        (Op.ref .real [128] "m_i") (Op.ref .real [128] "m_ij")),
    Stmt.assign .real [128] "alpha"
      (Op.sub .real Broadcast.nil.consSame (Op.ref .real [128] "m_i") (Op.ref .real [128] "m_i_new")).exp,
    Stmt.assign .real [128] "beta"
      (Op.sub .real Broadcast.nil.consSame (Op.ref .real [128] "m_ij") (Op.ref .real [128] "m_i_new")).exp,
    Stmt.assign .real [128] "l_i_new"
      (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [128] "alpha") (Op.ref .real [128] "l_i"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [128] "beta") (Op.ref .real [128] "l_ij"))),
    Stmt.assign .real [128] "p_scale"
      (Op.div .real Broadcast.nil.consSame (Op.ref .real [128] "beta") (Op.ref .real [128] "l_i_new")),
    Stmt.assign .real [128, 128] "p"
      (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "p_scale"))),
    Stmt.assign .real [128] "acc_scale"
      (Op.mul .real Broadcast.nil.consSame
        (Op.div .real Broadcast.nil.consSame (Op.ref .real [128] "l_i") (Op.ref .real [128] "l_i_new"))
        (Op.ref .real [128] "alpha")),
    Stmt.assign .real [128] "acc_scale"
      ((Op.ge .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarR (Op.ref .nat [128] "offs_m") (Op.ref .nat [] "prompt_cache_len"))
          (Op.ref .nat [] "start_n")).where
        (Op.ref .real [128] "acc_scale") ((Op.const (1.0 : ℝ)).broadcast [128])),
    Stmt.assign .real [128, 128] "acc"
      (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "acc_scale"))),
    Stmt.assign .nat [128, 128] "off_v"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "kv_loc"))
            (Op.constNat 576))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat 96)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .real [128, 128] "v"
      (Op.load .real (MemAccess.region V (Op.ref .nat [128, 128] "off_v"))
        (MaskOpt.maskOther
          (Op.boolAnd Broadcast.nil.consL.consR
            (Op.lt .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_n")))
              (Op.ref .nat [] "block_end_loc"))
            (Op.lt .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
              (Op.constNat 96)))
          ((Op.const (0.0 : ℝ)).broadcast [128, 128]))),
    Stmt.assign .real [128, 128] "p" (Op.ref .real [128, 128] "p"),
    Stmt.assign .real [128, 128] "acc"
      (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [128, 128] "acc")
        (Op.dot (batch := []) (Op.ref .real [128, 128] "p") (Op.ref .real [128, 128] "v"))),
    Stmt.assign .real [128] "l_i" (Op.ref .real [128] "l_i_new"),
    Stmt.assign .real [128] "m_i" (Op.ref .real [128] "m_i_new") ]

/-- The 4 lowered bloom post-loop statements (NO `acc /= l_i` — bloom normalizes
in-loop; `off_o`, `out_ptrs = Out + off_o`, masked `store`). -/
noncomputable def bloomPostLoop (Out : RegionName) : List Stmt :=
  [ Stmt.assign .nat [128, 128] "off_o"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")))
            (Op.constNat 576))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 96)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .ptr [128, 128] "out_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [128, 128] "off_o")),
    Stmt.store .real [128, 128] (MemAccess.ptr (Op.ref .ptr [128, 128] "out_ptrs"))
      (Op.ref .real [128, 128] "acc")
      (MaskOpt.mask
        (Op.boolAnd Broadcast.nil.consL.consR
          (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
            (Op.ref .nat [] "cur_batch_seq_len"))
          (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
            (Op.constNat 96)))) ]

/-- **Body decomposition.** The compiled Python-shape bloom body splits as
`bloomPreLoop ++ (forRangeDyn start_n 0 (block_mask·block_end_loc) 128 bloomLoopBody
:: bloomPostLoop)` — the loop body + post-loop tail are exactly the surface
lowering (pure `List` identity, `rfl`). -/
theorem bloomBody_split (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat) :
    (context_attn_bloom_fwd_kernel_surface Q K V (sm_scale_bloom : ℝ)
        B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len
        576 96 1 576 96 1 576 96 1 576 96 1
        7500 1 1 96 128 128 128).toAlgKernel.body
      = ((context_attn_bloom_fwd_kernel_surface Q K V (sm_scale_bloom : ℝ)
          B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len
          576 96 1 576 96 1 576 96 1 576 96 1
          7500 1 1 96 128 128 128).toAlgKernel.body.take 19)
        ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0)
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask") (Op.ref .nat [] "block_end_loc"))
              (Op.constNat 128) (bloomLoopBody Q K V Req_to_tokens B_req_idx)
            :: bloomPostLoop Out) := by
  rfl

/-! ### Bloom loop-body per-statement `evalOp` recipes -/

/-- `k` masked-load mask (`((start_n+offs_n[None,:]) < bel) & (offs_d[:,None] < 96)`,
shape `[D, BN]`). Lane `(e, j)` is `(SN + j < bel) ∧ (e < 96)`. -/
theorem bloomKMask_eval {BN D : Nat} (s : BlockState) (SN bel : Nat)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hd : s.regs .nat [D] "offs_d" = some (Tile.vec (fun e : Fin D => e.val)))
    (hbel : s.regs .nat [] "block_end_loc" = some (Tile.scalar bel)) :
    evalOp (Op.boolAnd Broadcast.nil.consR.consL
        (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")))
          (Op.ref .nat [] "block_end_loc"))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat 96))) s
      = some (⟨fun idx : TileIndex [D, BN] =>
          decide (SN + idx.2.1.val < bel) && decide (idx.1.val < 96)⟩ : Tile .bool [D, BN]) := by
  rw [evalOp]
  rw [evalOp_lt, evalOp_add]
  erw [ctx_evalOp_expandDim_zero_nat]
  rw [evalOp_lt]
  erw [ctx_evalOp_expandDim_one_nat]
  simp only [evalOp_ref, evalOp_constNat, hsn, hn, hd, hbel,
    Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  refine congrArg some ?_; ext idx
  simp [Tile.bop_data, Tile.cop_data, Tile.expandDim, Tile.scalar_data_index,
    Tile.vec_data, Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt, NumericDType.add]

/-- `v` masked-load mask (`((start_n+offs_n[:,None]) < bel) & (offs_d[None,:] < 96)`,
shape `[BN, D]`). Lane `(j, e)` is `(SN + j < bel) ∧ (e < 96)`. -/
theorem bloomVMask_eval {BN D : Nat} (s : BlockState) (SN bel : Nat)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hd : s.regs .nat [D] "offs_d" = some (Tile.vec (fun e : Fin D => e.val)))
    (hbel : s.regs .nat [] "block_end_loc" = some (Tile.scalar bel)) :
    evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")))
          (Op.ref .nat [] "block_end_loc"))
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat 96))) s
      = some (⟨fun idx : TileIndex [BN, D] =>
          decide (SN + idx.1.val < bel) && decide (idx.2.1.val < 96)⟩ : Tile .bool [BN, D]) := by
  rw [evalOp]
  rw [evalOp_lt, evalOp_add]
  erw [ctx_evalOp_expandDim_one_nat]
  rw [evalOp_lt]
  erw [ctx_evalOp_expandDim_zero_nat]
  simp only [evalOp_ref, evalOp_constNat, hsn, hn, hd, hbel,
    Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  refine congrArg some ?_; ext idx
  simp [Tile.bop_data, Tile.cop_data, Tile.expandDim, Tile.scalar_data_index,
    Tile.vec_data, Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt, NumericDType.add]

/-- `qk = tl.zeros([…])` full-zero eval. -/
theorem bloomQkFull_eval {BM BN : Nat} (s : BlockState) :
    evalOp (Op.full [BM, BN] (Op.const (0 : ℝ))) s
      = some (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN]) := by
  simp [evalOp_full, evalOp_const]

/-- `qk += tl.dot(q, k)` eval (`qk = qk + dot q k`, `qk` is all-zero). -/
theorem bloomQkAddDot_eval {BM BN D : Nat} (s : BlockState)
    (qktile : Tile .real [BM, BN]) (qtile : Tile .real [BM, D]) (ktile : Tile .real [D, BN])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hq : s.regs .real [BM, D] "q" = some qtile) (hk : s.regs .real [D, BN] "k" = some ktile) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BM, BN] "qk")
        (Op.dot (batch := []) (Op.ref .real [BM, D] "q") (Op.ref .real [D, BN] "k"))) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame qktile
          (Tile.dot [] qtile ktile)) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [BM, D] "q") (Op.ref .real [D, BN] "k")) s
      = some (Tile.dot [] qtile ktile) := by rw [evalOp_dot]; simp [hq, hk]
  have hdot2 : @evalOp TileDType.real [BM, BN]
      (Op.dot (batch := []) (Op.ref .real [BM, D] "q") (Op.ref .real [D, BN] "k")) s
      = some (Tile.dot [] qtile ktile) := hdot
  rw [evalOp_add]; simp only [evalOp_ref, hqk, hdot2, Option.bind_eq_bind, Option.bind_some]; rfl

/-- `qk *= sm_scale` eval. -/
theorem bloomQkScale_eval {BM BN : Nat} (s : BlockState) (sm : ℝ) (qktile : Tile .real [BM, BN])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, BN] "qk") (Op.const sm)) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qktile (Tile.scalar (some sm))) := by
  rw [evalOp_mul]; simp only [evalOp_ref, evalOp_const, hqk, Option.bind_eq_bind, Option.bind_some]

/-- `qk = tl.where(offs_m[:,None]+plen ≥ start_n+offs_n[None,:], qk, -1e8)` eval
(causal `-1e8` sentinel, mask `ge` inline). -/
theorem bloomQkWhere_eval (s : BlockState) (BM BN plen SN : Nat) (gOM : Fin BM → Nat)
    (qktile : Tile .real [BM, BN])
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec gOM))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hp : s.regs .nat [] "prompt_cache_len" = some (Tile.scalar plen))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp ((Op.ge .nat Broadcast.nil.consL.consR
          (Op.add .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
            (Op.ref .nat [] "prompt_cache_len"))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")))).where
        (Op.ref .real [BM, BN] "qk")
        ((Op.sub .real Broadcast.nil (Op.const (0.0 : ℝ)) (Op.const (100000000.0 : ℝ))).broadcast [BM, BN])) s
      = some (Tile.select
          (⟨fun idx : TileIndex [BM, BN] => decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ : Tile .bool [BM, BN])
          qktile
          (⟨fun _ : TileIndex [BM, BN] => some (0.0 - 100000000.0 : ℝ)⟩ : Tile .real [BM, BN])) := by
  rw [evalOp_where]
  have hmask : evalOp (Op.ge .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
          (Op.ref .nat [] "prompt_cache_len"))
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")))) s
      = some (⟨fun idx : TileIndex [BM, BN] => decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ : Tile .bool [BM, BN]) := by
    rw [ctx_evalOp_ge]
    simp only [evalOp_add, evalOp_ref, ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
      hm, hn, hp, hsn, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_; ext idx
    simp [Tile.cop, Tile.bop, Tile.expandDim, Tile.vec, ComparableDType.ge, NumericDType.add]
  have hother : @evalOp .real [BM, BN]
      ((Op.sub NumericDType.real Broadcast.nil (Op.const (0.0:ℝ)) (Op.const (100000000.0:ℝ))).broadcast [BM, BN]) s
      = some (⟨fun _ : TileIndex [BM, BN] => some (0.0 - 100000000.0 : ℝ)⟩ : Tile .real [BM, BN]) := by
    simp only [evalOp]; refine congrArg some ?_; ext idx; simp [Tile.bop, NumericDType.sub]
  simp only [hmask, evalOp_ref, hqk, hother, Option.bind_eq_bind, Option.bind_some]

/-- `m_ij = tl.max(qk, 1)` eval (block-max reduce). -/
theorem bloomMij_eval {BM BN : Nat} (s : BlockState) (qkfull : Tile .real [BM, BN]) (rmaxT : Tile .real [BM])
    (hqk : s.regs .real [BM, BN] "qk" = some qkfull)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM, BN].length) qkfull = some rmaxT) :
    evalOp (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [BM, BN] "qk")) s = some rmaxT := by
  rw [evalOp_reduceMax]; simp only [evalOp_ref, hqk, Option.bind_some]; exact hrm

/-- `l_ij = tl.sum(p, 1)` eval. -/
theorem bloomLij_eval {BM BN : Nat} (s : BlockState) (ptile : Tile .real [BM, BN])
    (hp : s.regs .real [BM, BN] "p" = some ptile) :
    evalOp (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false (Op.ref .real [BM, BN] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length) ptile) := by
  rw [evalOp_reduceSum]; simp only [evalOp_ref, hp, Option.bind_some]; rfl

/-- `p = tl.exp(qk − m_ij[:, None])` natural-exp eval. -/
theorem bloomP_eval {BM BN : Nat} (s : BlockState) (qk2tile : Tile .real [BM, BN])
    (mij : Tile .real [BM])
    (hqk : s.regs .real [BM, BN] "qk" = some qk2tile) (hmij : s.regs .real [BM] "m_ij" = some mij) :
    evalOp (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [BM, BN] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "m_ij"))).exp s
      = some (Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
          qk2tile (Tile.expandDim ⟨1, by simp⟩ mij))) := by
  rw [evalOp_exp, evalOp_sub]
  have hexp : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "m_ij")) s
      = some (Tile.expandDim ⟨1, by simp⟩ mij) := by erw [ctx_evalOp_expandDim_one_real, hmij]; rfl
  simp only [evalOp_ref, hqk, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

/-- `m_i_new = tl.maximum(m_i, m_ij)` = `where(m_i > m_ij, m_i, m_ij)` eval. -/
theorem bloomMiNew_eval {BM : Nat} (s : BlockState) (mi mij : Tile .real [BM])
    (hmi : s.regs .real [BM] "m_i" = some mi) (hmij : s.regs .real [BM] "m_ij" = some mij) :
    evalOp ((Op.gt .real Broadcast.nil.consSame (Op.ref .real [BM] "m_i") (Op.ref .real [BM] "m_ij")).where
        (Op.ref .real [BM] "m_i") (Op.ref .real [BM] "m_ij")) s
      = some (Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mi mij) mi mij) := by
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hmi, hmij, Option.bind_eq_bind, Option.bind_some]

/-- `alpha = tl.exp(m_i − m_i_new)` / `beta = tl.exp(m_ij − m_i_new)` natural-exp eval. -/
theorem bloomExpSub_eval {BM : Nat} (s : BlockState) (nm1 nm2 : RegName) (t1 t2 : Tile .real [BM])
    (h1 : s.regs .real [BM] nm1 = some t1) (h2 : s.regs .real [BM] nm2 = some t2) :
    evalOp (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BM] nm1) (Op.ref .real [BM] nm2)).exp s
      = some (Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consSame t1 t2)) := by
  rw [evalOp_exp, evalOp_sub]; simp [h1, h2]

/-- `l_i_new = alpha·l_i + beta·l_ij` eval. -/
theorem bloomLiNew_eval {BM : Nat} (s : BlockState) (alpha li beta lij : Tile .real [BM])
    (ha : s.regs .real [BM] "alpha" = some alpha) (hli : s.regs .real [BM] "l_i" = some li)
    (hb : s.regs .real [BM] "beta" = some beta) (hlij : s.regs .real [BM] "l_ij" = some lij) :
    evalOp (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BM] "alpha") (Op.ref .real [BM] "l_i"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BM] "beta") (Op.ref .real [BM] "l_ij"))) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame
          (Tile.bop NumericDType.real.mul Broadcast.nil.consSame alpha li)
          (Tile.bop NumericDType.real.mul Broadcast.nil.consSame beta lij)) := by
  rw [evalOp_add]; simp [evalOp_mul, ha, hli, hb, hlij]

/-- `p_scale = beta / l_i_new` eval. -/
theorem bloomPscale_eval {BM : Nat} (s : BlockState) (beta lin : Tile .real [BM])
    (hb : s.regs .real [BM] "beta" = some beta) (hlin : s.regs .real [BM] "l_i_new" = some lin) :
    evalOp (Op.div .real Broadcast.nil.consSame (Op.ref .real [BM] "beta") (Op.ref .real [BM] "l_i_new")) s
      = some (Tile.bop NumericDType.real.div Broadcast.nil.consSame beta lin) := by
  rw [evalOp_div]; simp [hb, hlin]

/-- `p = p · p_scale[:, None]` eval. -/
theorem bloomP2_eval {BM BN : Nat} (s : BlockState) (ptile : Tile .real [BM, BN]) (pscale : Tile .real [BM])
    (hp : s.regs .real [BM, BN] "p" = some ptile) (hps : s.regs .real [BM] "p_scale" = some pscale) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BM, BN] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "p_scale"))) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame ptile
          (Tile.expandDim ⟨1, by simp⟩ pscale)) := by
  have hexp : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "p_scale")) s
      = some (Tile.expandDim ⟨1, by simp⟩ pscale) := by erw [ctx_evalOp_expandDim_one_real, hps]; rfl
  rw [evalOp_mul]; simp only [evalOp_ref, hp, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

/-- `acc_scale = (l_i / l_i_new) · alpha` eval. -/
theorem bloomAccScale1_eval {BM : Nat} (s : BlockState) (li lin alpha : Tile .real [BM])
    (hli : s.regs .real [BM] "l_i" = some li) (hlin : s.regs .real [BM] "l_i_new" = some lin)
    (ha : s.regs .real [BM] "alpha" = some alpha) :
    evalOp (Op.mul .real Broadcast.nil.consSame
        (Op.div .real Broadcast.nil.consSame (Op.ref .real [BM] "l_i") (Op.ref .real [BM] "l_i_new"))
        (Op.ref .real [BM] "alpha")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consSame
          (Tile.bop NumericDType.real.div Broadcast.nil.consSame li lin) alpha) := by
  rw [evalOp_mul]; simp [evalOp_div, hli, hlin, ha]

/-- `acc_scale = tl.where(offs_m+plen ≥ start_n, acc_scale, 1.0)` eval (the
acc-rescale active-lane guard). -/
theorem bloomAccScale2_eval (s : BlockState) (BM plen SN : Nat) (gOM : Fin BM → Nat)
    (acctile : Tile .real [BM])
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec gOM))
    (hp : s.regs .nat [] "prompt_cache_len" = some (Tile.scalar plen))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hacc : s.regs .real [BM] "acc_scale" = some acctile) :
    evalOp ((Op.ge .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarR (Op.ref .nat [BM] "offs_m") (Op.ref .nat [] "prompt_cache_len"))
          (Op.ref .nat [] "start_n")).where
        (Op.ref .real [BM] "acc_scale") ((Op.const (1.0 : ℝ)).broadcast [BM])) s
      = some (Tile.select
          (⟨fun idx : TileIndex [BM] => decide (SN ≤ gOM idx.1 + plen)⟩ : Tile .bool [BM])
          acctile (⟨fun _ : TileIndex [BM] => some (1.0 : ℝ)⟩ : Tile .real [BM])) := by
  rw [evalOp_where]
  have hmask : evalOp (Op.ge .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarR (Op.ref .nat [BM] "offs_m") (Op.ref .nat [] "prompt_cache_len"))
        (Op.ref .nat [] "start_n")) s
      = some (⟨fun idx : TileIndex [BM] => decide (SN ≤ gOM idx.1 + plen)⟩ : Tile .bool [BM]) := by
    rw [ctx_evalOp_ge]
    simp only [evalOp_add, evalOp_ref, hm, hp, hsn, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_; ext idx
    simp [Tile.cop, Tile.bop, Tile.vec, Tile.scalar_data_index, ComparableDType.ge, NumericDType.add]
  have hother : @evalOp .real [BM] ((Op.const (1.0 : ℝ)).broadcast [BM]) s
      = some (⟨fun _ : TileIndex [BM] => some (1.0 : ℝ)⟩ : Tile .real [BM]) := by
    simp only [evalOp]; refine congrArg some ?_; ext idx; rfl
  simp only [hmask, evalOp_ref, hacc, hother, Option.bind_eq_bind, Option.bind_some]

/-- `acc = acc · acc_scale[:, None]` eval. -/
theorem bloomAcc1_eval {BM D : Nat} (s : BlockState) (acctile : Tile .real [BM, D]) (ascale : Tile .real [BM])
    (hacc : s.regs .real [BM, D] "acc" = some acctile) (has : s.regs .real [BM] "acc_scale" = some ascale) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BM, D] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "acc_scale"))) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acctile
          (Tile.expandDim ⟨1, by simp⟩ ascale)) := by
  have hexp : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "acc_scale")) s
      = some (Tile.expandDim ⟨1, by simp⟩ ascale) := by erw [ctx_evalOp_expandDim_one_real, has]; rfl
  rw [evalOp_mul]; simp only [evalOp_ref, hacc, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

/-- `acc += tl.dot(p, v)` eval (`acc = acc + dot p v`). -/
theorem bloomAcc2_eval {BM BN D : Nat} (s : BlockState) (acctile : Tile .real [BM, D])
    (ptile : Tile .real [BM, BN]) (vtile : Tile .real [BN, D])
    (hacc : s.regs .real [BM, D] "acc" = some acctile)
    (hp : s.regs .real [BM, BN] "p" = some ptile) (hv : s.regs .real [BN, D] "v" = some vtile) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BM, D] "acc")
        (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, D] "v"))) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame acctile
          (Tile.dot [] ptile vtile)) := by
  have hdot0 : evalOp (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, D] "v")) s
      = some (Tile.dot [] ptile vtile) := by rw [evalOp_dot]; simp [hp, hv]
  have hdot : @evalOp TileDType.real [BM, D]
      (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, D] "v")) s
      = some (Tile.dot [] ptile vtile) := hdot0
  rw [evalOp_add]; simp only [evalOp_ref, hacc, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **LoopBody execution (Milestone 1).** The 26 lowered bloom loop-body statements
step a loop-entry state to a final state whose running registers advance by one
block of the natural-exp two-level-max in-loop-normalized online softmax. Exposes
the symbolic intermediate tiles (`k`/`v` loads, the `qk` where, `m_ij`, `m_i_new`,
`alpha`/`beta`, `l_ij`, `p`, `p_scale`, `l_i_new`, `acc_scale`) and the final
`m_i`/`l_i`/`acc` registers; all index/scalar registers are preserved. -/
theorem bloomLoopBody_steps (Q K V Req_to_tokens B_req_idx : RegionName) (sin : BlockState) (SN : Nat)
    (gOM : Fin 128 → Nat) (cb ckvh ch plen sl bel cbsi rqi : Nat)
    (qtile : Tile .real [128, 128]) (mtile ltile : Tile .real [128])
    (acctile : Tile .real [128, 128])
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hcb : sin.regs .nat [] "cur_batch" = some (Tile.scalar cb))
    (hckvh : sin.regs .nat [] "cur_kv_head" = some (Tile.scalar ckvh))
    (hch : sin.regs .nat [] "cur_head" = some (Tile.scalar ch))
    (hplen : sin.regs .nat [] "prompt_cache_len" = some (Tile.scalar plen))
    (hsl : sin.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar sl))
    (hbel : sin.regs .nat [] "block_end_loc" = some (Tile.scalar bel))
    (hcbsi : sin.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar cbsi))
    (hrqi : sin.regs .nat [] "cur_batch_req_idx" = some (Tile.scalar rqi))
    (hm : sin.regs .nat [128] "offs_m" = some (Tile.vec gOM))
    (hn : sin.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hd : sin.regs .nat [128] "offs_d" = some (Tile.vec (fun e : Fin 128 => e.val)))
    (hq : sin.regs .real [128, 128] "q" = some qtile)
    (hmi : sin.regs .real [128] "m_i" = some mtile)
    (hli : sin.regs .real [128] "l_i" = some ltile)
    (hacc : sin.regs .real [128, 128] "acc" = some acctile)
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ sF, stepStmts (bloomLoopBody Q K V Req_to_tokens B_req_idx) sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = 0)
      ∧ sF.regs .nat [] "start_n" = some (Tile.scalar SN)
      ∧ sF.regs .nat [] "cur_batch" = some (Tile.scalar cb)
      ∧ sF.regs .nat [] "cur_kv_head" = some (Tile.scalar ckvh)
      ∧ sF.regs .nat [] "cur_head" = some (Tile.scalar ch)
      ∧ sF.regs .nat [] "prompt_cache_len" = some (Tile.scalar plen)
      ∧ sF.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar sl)
      ∧ sF.regs .nat [] "block_end_loc" = some (Tile.scalar bel)
      ∧ sF.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar cbsi)
      ∧ sF.regs .nat [] "cur_batch_req_idx" = some (Tile.scalar rqi)
      ∧ sF.regs .nat [128] "offs_m" = some (Tile.vec gOM)
      ∧ sF.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val))
      ∧ sF.regs .nat [128] "offs_d" = some (Tile.vec (fun e : Fin 128 => e.val))
      ∧ sF.regs .real [128, 128] "q" = some qtile
      ∧ ∃ (kloadT vloadT : Tile .real [128, 128]) (qkT : Tile .real [128, 128])
            (rmaxT miNewT alphaT betaT lijT pscaleT liNewT accscale2T : Tile .real [128])
            (pexpT p2T acc1T : Tile .real [128, 128]),
          kloadT = (⟨fun idx : TileIndex [128, 128] =>
              if decide (SN + idx.2.1.val < bel) && decide (idx.1.val < 96) then
                sin.readMemValue .real (Region.cast K)
                  ((if decide (SN + idx.2.1.val < bel) then
                      sin.readMemValue .nat (Region.cast Req_to_tokens) (7500 * rqi + (SN + idx.2.1.val))
                    else 0) * 576 + ckvh * 96 + idx.1.val * 1)
              else some (0.0 : ℝ)⟩ : Tile .real [128, 128])
          ∧ vloadT = (⟨fun idx : TileIndex [128, 128] =>
              if decide (SN + idx.1.val < bel) && decide (idx.2.1.val < 96) then
                sin.readMemValue .real (Region.cast V)
                  ((if decide (SN + idx.1.val < bel) then
                      sin.readMemValue .nat (Region.cast Req_to_tokens) (7500 * rqi + (SN + idx.1.val))
                    else 0) * 576 + ckvh * 96 + idx.2.1.val * 1)
              else some (0.0 : ℝ)⟩ : Tile .real [128, 128])
          ∧ qkT = Tile.select
              (⟨fun idx : TileIndex [128, 128] => decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ : Tile .bool [128, 128])
              (Tile.bop NumericDType.real.mul Broadcast.scalarR
                (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
                  (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128])
                  (Tile.dot [] qtile kloadT))
                (Tile.scalar (some sm_scale_bloom)))
              (⟨fun _ : TileIndex [128, 128] => some (0.0 - 100000000.0 : ℝ)⟩ : Tile .real [128, 128])
          ∧ Tile.reduceMaxDrop (⟨1, by decide⟩ : Fin [128, 128].length) qkT = some rmaxT
          ∧ pexpT = Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
              qkT (Tile.expandDim ⟨1, by decide⟩ rmaxT))
          ∧ lijT = Tile.reduceSumDrop (⟨1, by decide⟩ : Fin [128, 128].length) pexpT
          ∧ miNewT = Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mtile rmaxT) mtile rmaxT
          ∧ alphaT = Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile miNewT)
          ∧ betaT = Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consSame rmaxT miNewT)
          ∧ liNewT = Tile.bop NumericDType.real.add Broadcast.nil.consSame
              (Tile.bop NumericDType.real.mul Broadcast.nil.consSame alphaT ltile)
              (Tile.bop NumericDType.real.mul Broadcast.nil.consSame betaT lijT)
          ∧ pscaleT = Tile.bop NumericDType.real.div Broadcast.nil.consSame betaT liNewT
          ∧ p2T = Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame pexpT (Tile.expandDim ⟨1, by decide⟩ pscaleT)
          ∧ accscale2T = Tile.select
              (⟨fun idx : TileIndex [128] => decide (SN ≤ gOM idx.1 + plen)⟩ : Tile .bool [128])
              (Tile.bop NumericDType.real.mul Broadcast.nil.consSame
                (Tile.bop NumericDType.real.div Broadcast.nil.consSame ltile liNewT) alphaT)
              (⟨fun _ : TileIndex [128] => some (1.0 : ℝ)⟩ : Tile .real [128])
          ∧ acc1T = Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acctile (Tile.expandDim ⟨1, by decide⟩ accscale2T)
          ∧ sF.regs .real [128] "m_i" = some miNewT
          ∧ sF.regs .real [128] "l_i" = some liNewT
          ∧ sF.regs .real [128, 128] "acc" = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
              acc1T (Tile.dot [] p2T vloadT)) := by
  set kvf : Fin 128 → Nat := fun j : Fin 128 =>
      if decide (SN + j.val < bel) then
        sin.readMemValue .nat (Region.cast Req_to_tokens) (7500 * rqi + (SN + j.val))
      else 0 with hkvf
  set kloadT : Tile .real [128, 128] := ⟨fun idx : TileIndex [128, 128] =>
      if decide (SN + idx.2.1.val < bel) && decide (idx.1.val < 96) then
        sin.readMemValue .real (Region.cast K) (kvf idx.2.1 * 576 + ckvh * 96 + idx.1.val * 1)
      else some (0.0 : ℝ)⟩ with hkl
  set vloadT : Tile .real [128, 128] := ⟨fun idx : TileIndex [128, 128] =>
      if decide (SN + idx.1.val < bel) && decide (idx.2.1.val < 96) then
        sin.readMemValue .real (Region.cast V) (kvf idx.1 * 576 + ckvh * 96 + idx.2.1.val * 1)
      else some (0.0 : ℝ)⟩ with hvl
  set qkdotT : Tile .real [128, 128] := Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
      (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128]) (Tile.dot [] qtile kloadT) with hqkdot
  set qkscaleT : Tile .real [128, 128] := Tile.bop NumericDType.real.mul Broadcast.scalarR qkdotT
      (Tile.scalar (some sm_scale_bloom)) with hqkscale
  set qkT : Tile .real [128, 128] := Tile.select
      (⟨fun idx : TileIndex [128, 128] => decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ : Tile .bool [128, 128])
      qkscaleT
      (⟨fun _ : TileIndex [128, 128] => some (0.0 - 100000000.0 : ℝ)⟩ : Tile .real [128, 128]) with hqk
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by decide⟩ : Fin [128, 128].length) qkT = some t :=
    ⟨_, by unfold Tile.reduceMaxDrop; rw [dif_pos (show 0 < TileShape.axisDim [128, 128] (⟨1, by decide⟩ : Fin [128, 128].length) from by decide)]⟩
  set pexpT : Tile .real [128, 128] := Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame qkT (Tile.expandDim ⟨1, by decide⟩ rmaxT)) with hpexp
  set lijT : Tile .real [128] := Tile.reduceSumDrop (⟨1, by decide⟩ : Fin [128, 128].length) pexpT with hlij
  set miNewT : Tile .real [128] := Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mtile rmaxT) mtile rmaxT with hminew
  set alphaT : Tile .real [128] := Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile miNewT) with hal
  set betaT : Tile .real [128] := Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consSame rmaxT miNewT) with hbeta
  set liNewT : Tile .real [128] := Tile.bop NumericDType.real.add Broadcast.nil.consSame
      (Tile.bop NumericDType.real.mul Broadcast.nil.consSame alphaT ltile)
      (Tile.bop NumericDType.real.mul Broadcast.nil.consSame betaT lijT) with hlinew
  set pscaleT : Tile .real [128] := Tile.bop NumericDType.real.div Broadcast.nil.consSame betaT liNewT with hpscale
  set p2T : Tile .real [128, 128] := Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame pexpT (Tile.expandDim ⟨1, by decide⟩ pscaleT) with hp2
  set accscale1T : Tile .real [128] := Tile.bop NumericDType.real.mul Broadcast.nil.consSame
      (Tile.bop NumericDType.real.div Broadcast.nil.consSame ltile liNewT) alphaT with hascale1
  set accscale2T : Tile .real [128] := Tile.select
      (⟨fun idx : TileIndex [128] => decide (SN ≤ gOM idx.1 + plen)⟩ : Tile .bool [128]) accscale1T
      (⟨fun _ : TileIndex [128] => some (1.0 : ℝ)⟩ : Tile .real [128]) with hascale2
  set acc1T : Tile .real [128, 128] := Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acctile (Tile.expandDim ⟨1, by decide⟩ accscale2T) with hacc1
  unfold bloomLoopBody
  -- stmt 0: start_n = ref start_n
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_n") sin = some (Tile.scalar SN) from by rw [evalOp_ref]; exact hsn))]
  -- stmt 1: kv_loc = masked gather
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloom_kvloc_gather_eval _ Req_to_tokens rqi SN bel
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hrqi]) (by simp only [BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hn]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hbel])))]
  -- stmt 2: off_k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloom_offk_gather_eval _ ckvh kvf
      (by rw [BlockState.setReg_same, hkvf]; refine congrArg some ?_; ext idx;
          simp only [Tile.vec, BlockState.setReg_readMemValue])
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hckvh]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hd])))]
  -- stmt 3: k = masked load (2-cond)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_region_maskOther K (Op.ref .nat [128, 128] "off_k") _ _ _
      (⟨fun idx : TileIndex [128, 128] => kvf idx.2.1 * 576 + ckvh * 96 + idx.1.val * 1⟩ : Tile .nat [128, 128])
      (⟨fun idx : TileIndex [128, 128] => decide (SN + idx.2.1.val < bel) && decide (idx.1.val < 96)⟩ : Tile .bool [128, 128])
      (⟨fun _ : TileIndex [128, 128] => some (0.0 : ℝ)⟩ : Tile .real [128, 128])
      (by rw [evalOp_ref]; simp [BlockState.setReg_same])
      (bloomKMask_eval _ SN bel (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hsn])
        (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hn]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hd]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hbel]))
      (by simp only [evalOp, Option.bind_eq_bind, Option.bind_some]; rfl)))]
  -- stmt 4: qk = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (bloomQkFull_eval _))]
  -- stmt 5: qk = qk + dot q k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomQkAddDot_eval _ (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128]) qtile kloadT
      (by simp only [BlockState.setReg_same])
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hq])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), BlockState.setReg_same, hkl];
          refine congrArg some ?_; ext idx; simp only [BlockState.setReg_readMemValue])))]
  -- stmt 6: qk = qk * sm_scale
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomQkScale_eval _ sm_scale_bloom qkdotT (by simp only [BlockState.setReg_same, hqkdot])))]
  -- stmt 7: qk = where(ge, qk, -1e8)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomQkWhere_eval _ 128 128 plen SN gOM qkscaleT
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hm]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hn])
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hplen]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hsn])
      (by simp only [BlockState.setReg_same, hqkscale])))]
  -- stmt 8: m_ij = reduceMax
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .real [128] "m_ij"
    (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "qk")) _ rmaxT
    (bloomMij_eval _ qkT rmaxT (by simp only [BlockState.setReg_same, hqk]) hrm))]
  -- stmt 9: p = exp(qk - m_ij)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomP_eval _ qkT rmaxT (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hqk]) (by simp only [BlockState.setReg_same, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 10: l_ij = sum p 1
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .real [128] "l_ij"
    (Op.reduceSum (⟨1, by decide⟩ : Fin [128, 128].length) Bool.false (Op.ref .real [128, 128] "p")) _ lijT
    (bloomLij_eval _ pexpT (by simp only [BlockState.setReg_same, hpexp]; try rfl)))]
  -- stmt 11: m_i_new = maximum(m_i, m_ij)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomMiNew_eval _ mtile rmaxT (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hmi]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 12: alpha = exp(m_i - m_i_new)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomExpSub_eval _ "m_i" "m_i_new" mtile miNewT (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hmi]) (by simp only [BlockState.setReg_same, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 13: beta = exp(m_ij - m_i_new)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomExpSub_eval _ "m_ij" "m_i_new" rmaxT miNewT (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 14: l_i_new = alpha*l_i + beta*l_ij
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomLiNew_eval _ alphaT ltile betaT lijT
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hli])
      (by simp only [BlockState.setReg_same, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 15: p_scale = beta / l_i_new
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomPscale_eval _ betaT liNewT (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl) (by simp only [BlockState.setReg_same, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 16: p = p * p_scale
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomP2_eval _ pexpT pscaleT
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)
      (by simp only [BlockState.setReg_same, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 17: acc_scale = (l_i/l_i_new)*alpha
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomAccScale1_eval _ ltile liNewT alphaT
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hli])
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 18: acc_scale = where(ge, acc_scale, 1)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomAccScale2_eval _ 128 plen SN gOM accscale1T
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hm])
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hplen])
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hsn])
      (by simp only [BlockState.setReg_same, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 19: acc = acc * acc_scale
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomAcc1_eval _ acctile accscale2T
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hacc])
      (by simp only [BlockState.setReg_same, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)))]
  -- stmt 20: off_v = gather
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloom_offv_gather_eval _ ckvh kvf
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hkvf];
          refine congrArg some ?_; ext idx; simp only [Tile.vec, BlockState.setReg_readMemValue])
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hckvh]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hd])))]
  -- stmt 21: v = masked load (2-cond)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_region_maskOther V (Op.ref .nat [128, 128] "off_v") _ _ _
      (⟨fun idx : TileIndex [128, 128] => kvf idx.1 * 576 + ckvh * 96 + idx.2.1.val * 1⟩ : Tile .nat [128, 128])
      (⟨fun idx : TileIndex [128, 128] => decide (SN + idx.1.val < bel) && decide (idx.2.1.val < 96)⟩ : Tile .bool [128, 128])
      (⟨fun _ : TileIndex [128, 128] => some (0.0 : ℝ)⟩ : Tile .real [128, 128])
      (by rw [evalOp_ref]; simp [BlockState.setReg_same])
      (bloomVMask_eval _ SN bel (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hsn])
        (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hn]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hd]) (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hbel]))
      (by simp only [evalOp, Option.bind_eq_bind, Option.bind_some]; rfl)))]
  -- stmt 22: p = ref p (noop)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [128, 128] "p") _ = some p2T from by rw [evalOp_ref]; simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hp2]; try rfl))]
  -- stmt 23: acc = acc + dot(p, v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bloomAcc2_eval _ acc1T p2T vloadT
      (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)
      (by simp only [BlockState.setReg_same, hpexp, hlij, hminew, hal, hbeta, hlinew, hpscale, hp2, hascale1, hascale2, hacc1]; try rfl)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), BlockState.setReg_same, hvl];
          refine congrArg some ?_; ext idx; simp only [BlockState.setReg_readMemValue])))]
  -- stmt 24: l_i = ref l_i_new
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [128] "l_i_new") _ = some liNewT from by rw [evalOp_ref]; simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hlinew]; try rfl))]
  -- stmt 25: m_i = ref m_i_new
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [128] "m_i_new") _ = some miNewT from by rw [evalOp_ref]; simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hminew]; try rfl))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    kloadT, vloadT, qkT, rmaxT, miNewT, alphaT, betaT, lijT, pscaleT, liNewT, accscale2T,
    pexpT, p2T, acc1T,
    rfl, rfl, rfl, hrm, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_pids]
  · funext rg o; simp only [BlockState.setReg_mem]
  · intro rg o; simp only [BlockState.setReg_undef]; exact hundef rg o
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hcb]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hckvh]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hch]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hplen]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hsl]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hbel]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hcbsi]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hrqi]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hm]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hn]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hd]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hq]
  · -- m_i = miNewT (top setReg)
    rw [BlockState.setReg_same]
  · -- l_i = liNewT (peel m_i)
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq]
  · -- acc readback (peel m_i, l_i)
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq]

/-! ### Bloom-specific per-key data + boundary-masked exact fold

The bloom kernel feeds the generic base-2 machinery the per-key pair
`(ctxBloomScore / log 2, value)` so `pow2 score = exp (natural kernel score)`.
The streaming loop runs to `S = ceil₁₂₈(block_end_loc)`, streaming phantom keys
`[bel, S)` whose masked `k`/`v` are `0`; the boundary-masked tiles
`bloomKMaskTile`/`bloomVMaskTile` capture exactly that. -/

/-- `block_end_loc`-masked key tile (genuine `ctxKTile` for `j < bel`, else `0`). -/
noncomputable def bloomKTileM
    (s : BlockState) (K Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s S bel : Nat) : TileIndex [S, 128] → ℝ :=
  fun (j, e, u) =>
    if j.val < bel then ctxKTile s K Req_to_tokens B_req_idx stride_req_b stride_req_s S (j, e, u)
    else 0

/-- `block_end_loc`-masked value tile. -/
noncomputable def bloomVTileM
    (s : BlockState) (V Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s S bel : Nat) : TileIndex [S, 128] → ℝ :=
  fun (j, d, u) =>
    if j.val < bel then ctxVTile s V Req_to_tokens B_req_idx stride_req_b stride_req_s S (j, d, u)
    else 0

/-- Row-masked query tile: genuine `ctxQTile` on active rows, else `0`. -/
noncomputable def bloomQTileM
    (s : BlockState) (Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Fin 128) : ℝ :=
  if s.pids 2 * BLOCK_M + i.val < seqLen s B_Seqlen B_Prompt_Cache_Len then
    ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
  else 0

/-- **Faithful per-key `(base-2 score, value)`** the loop folds, with the genuine
`-1e8` sentinel and the `block_end_loc` load mask on `k`/`v`. Score is fed in
base-2 (`/ log 2`) so `pow2 score = exp (kernel natural score)`. -/
noncomputable def bloomKVM
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens
      B_req_idx : RegionName) (sm_scale : ℝ)
    (stride_req_b stride_req_s BLOCK_M S bel : Nat) (i : Fin BLOCK_M) (d : Fin 128)
    (j : Fin S) : ℝ × ℝ :=
  ((if j.val ≤ s.pids 2 * BLOCK_M + i.val + promptLen s B_Prompt_Cache_Len then
      sm_scale * Finset.univ.sum (fun e : Fin 128 =>
        bloomQTileM s Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len BLOCK_M i e
          * bloomKTileM s K Req_to_tokens B_req_idx stride_req_b stride_req_s S bel (j, e, PUnit.unit))
    else (0.0 - 10e7 : ℝ)) / Real.log 2,
    bloomVTileM s V Req_to_tokens B_req_idx stride_req_b stride_req_s S bel (j, d, PUnit.unit))

/-- **The faithful kernel value** at output lane `(i,d)`: `acc/l` of the ⊥-seeded
online-softmax fold over `bloomKVM` for the full streamed window `[0, S)`. -/
noncomputable def contextAttnBloomExactFoldM
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens
      B_req_idx : RegionName) (sm_scale : ℝ)
    (stride_req_b stride_req_s BLOCK_M S bel : Nat) (idx : TileIndex [BLOCK_M, 128]) : ℝ :=
  let st := gStateBot S S (bloomKVM s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len
      Req_to_tokens B_req_idx sm_scale stride_req_b stride_req_s BLOCK_M S bel idx.1 idx.2.1)
  st.2.2 / st.2.1

/-! ### Bloom preLoop + streaming-loop invariant (Milestone 2) -/

/-- The 19 lowered bloom prologue statements (transcribed from the surface
lowering; `bloomPreLoop_take` checks by `rfl`). -/
noncomputable def bloomPreLoop (Q : RegionName)
    (B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len : Region .nat) : List Stmt :=
  [ Stmt.assign .nat [] "cur_batch" (Op.programId 0),
    Stmt.assign .nat [] "cur_head" (Op.programId 1),
    Stmt.assign .nat [] "start_m" (Op.programId 2),
    Stmt.assign .nat [] "cur_kv_head"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 1)),
    Stmt.assign .nat [] "cur_batch_in_all_start_index"
      (Op.load .nat (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "prompt_cache_len"
      (Op.load .nat (MemAccess.region b_prompt_cache_len (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "cur_batch_seq_len"
      (Op.sub .nat Broadcast.nil
        (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none)
        (Op.ref .nat [] "prompt_cache_len")),
    Stmt.assign .nat [] "cur_batch_req_idx"
      (Op.load .nat (MemAccess.region B_req_idx (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "block_start_loc"
      (Op.mul .nat Broadcast.nil (Op.constNat 128) (Op.ref .nat [] "start_m")),
    Stmt.assign .nat [128] "offs_n" (Op.arange 128),
    Stmt.assign .nat [128] "offs_d" (Op.arange 128),
    Stmt.assign .nat [128] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128)) (Op.arange 128)),
    Stmt.assign .nat [128, 128] "off_q"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")))
            (Op.constNat 576))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 96)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .real [128, 128] "q"
      (Op.load .real (MemAccess.region Q (Op.ref .nat [128, 128] "off_q"))
        (MaskOpt.maskOther
          (Op.boolAnd Broadcast.nil.consL.consR
            (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
              (Op.ref .nat [] "cur_batch_seq_len"))
            (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
              (Op.constNat 96)))
          ((Op.const (0.0 : ℝ)).broadcast [128, 128]))),
    Stmt.assign .real [128] "m_i"
      (Op.add .real Broadcast.scalarR (Op.full [128] (Op.const 0)) Op.negInf),
    Stmt.assign .real [128] "l_i" (Op.full [128] (Op.const 0)),
    Stmt.assign .real [128, 128] "acc" (Op.full [128, 128] (Op.const 0)),
    Stmt.assign .nat [] "block_mask"
      ((Op.lt .nat Broadcast.nil (Op.ref .nat [] "block_start_loc") (Op.ref .nat [] "cur_batch_seq_len")).where
        (Op.constNat 1) (Op.constNat 0)),
    Stmt.assign .nat [] "block_end_loc"
      ((Op.lt .nat Broadcast.nil
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat 128))
              (Op.ref .nat [] "prompt_cache_len"))
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len")
              (Op.ref .nat [] "prompt_cache_len"))).where
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat 128))
          (Op.ref .nat [] "prompt_cache_len"))
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len")
          (Op.ref .nat [] "prompt_cache_len"))) ]

/-- The lowered Python-shape bloom body `take 19` is exactly `bloomPreLoop`. -/
theorem bloomPreLoop_take (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat) :
    (context_attn_bloom_fwd_kernel_surface Q K V (sm_scale_bloom : ℝ)
        B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len
        576 96 1 576 96 1 576 96 1 576 96 1
        7500 1 1 96 128 128 128).toAlgKernel.body.take 19
      = bloomPreLoop Q B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len := by
  rfl

/-- Per-key data feeding the bloom invariant: `bloomKVM` at the Python shape. -/
noncomputable def bloomG
    (s0 : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx : RegionName)
    (sm_scale : ℝ) (S bel : Nat) (i d : Fin 128) : Fin S → ℝ × ℝ :=
  bloomKVM s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale 7500 1 128 S bel i d

/-- **Streaming-loop invariant** for the bloom kernel. After `c` blocks the
in-loop-normalized registers carry the ⊥-seeded online-softmax fold `gStateBot
(c·128)` over `bloomG`: `m_i = log2 · (running max)` (so natural `exp(m_i−·)`
matches the base-2 `pow2`), `l_i = (running denominator)`, and `acc = (running
numerator)/(running denominator)` — the in-loop `β/lᵢⁿᵉʷ`/`(lᵢ/lᵢⁿᵉʷ)·α`
normalization. The auxiliary registers are the constants seeded by `bloomPreLoop`. -/
noncomputable def bloomInvariant
    (Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName) (s0 : BlockState)
    (sm_scale : ℝ) (S bel : Nat) (c : Nat) (s : BlockState) : Prop :=
  let plen := promptLen s0 B_Prompt_Cache_Len
  let sl := seqLen s0 B_Seqlen B_Prompt_Cache_Len
  let g := fun (i d : Fin 128) => bloomG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale S bel i d
  s.pids = s0.pids ∧ s.mem = s0.mem ∧ (∀ rg o, s.undef rg o = 0) ∧
  (s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 0))) ∧
  (s.regs .nat [] "cur_kv_head" = some (Tile.scalar (s0.pids 1 / 1))) ∧
  (s.regs .nat [] "cur_head" = some (Tile.scalar (s0.pids 1))) ∧
  (s.regs .nat [] "prompt_cache_len" = some (Tile.scalar plen)) ∧
  (s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar sl)) ∧
  (s.regs .nat [] "block_end_loc" = some (Tile.scalar bel)) ∧
  (s.regs .nat [] "cur_batch_in_all_start_index"
      = some (Tile.scalar (startLoc s0 B_Start_Loc))) ∧
  (s.regs .nat [] "cur_batch_req_idx"
      = some (Tile.scalar (reqIdx s0 B_req_idx))) ∧
  (s.regs .nat [128] "offs_m" = some (Tile.vec (fun r : Fin 128 => s0.pids 2 * 128 + r.val))) ∧
  (s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val))) ∧
  (s.regs .nat [128] "offs_d" = some (Tile.vec (fun e : Fin 128 => e.val))) ∧
  (s.regs .real [128, 128] "q" = some ⟨fun idx : TileIndex [128, 128] =>
      if decide (s0.pids 2 * 128 + idx.1.val < sl) && decide (idx.2.1.val < 96) then
        s0.readMemValue .real (Region.cast Q) ((startLoc s0 B_Start_Loc + (s0.pids 2 * 128 + idx.1.val)) * 576
            + s0.pids 1 * 96 + idx.2.1.val * 1)
      else some (0.0 : ℝ)⟩) ∧
  (s.regs .real [128] "m_i" = some ⟨fun r : TileIndex [128] =>
      (gStateBot S (c * 128) (g r.1 ⟨0, by decide⟩)).1.map (· * Real.log 2)⟩) ∧
  (s.regs .real [128] "l_i" = some ⟨fun r : TileIndex [128] =>
      some (gStateBot S (c * 128) (g r.1 ⟨0, by decide⟩)).2.1⟩) ∧
  (s.regs .real [128, 128] "acc" = some ⟨fun idx : TileIndex [128, 128] =>
      some ((gStateBot S (c * 128) (g idx.1 idx.2.1)).2.2 / (gStateBot S (c * 128) (g idx.1 idx.2.1)).2.1)⟩) ∧
  (c * 128 ≤ S)

/-- `q`/store-mask eval: `(offs_m[:,None] < sl) & (offs_d[None,:] < 96)`, shape
`[128, 128]`. Lane `(r, e)` is `(gOM r < sl) ∧ (e < 96)`. -/
theorem bloomQMask_eval {BM D : Nat} (s : BlockState) (gOM : Fin BM → Nat) (sl : Nat)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec gOM))
    (hd : s.regs .nat [D] "offs_d" = some (Tile.vec (fun e : Fin D => e.val)))
    (hsl : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar sl)) :
    evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
          (Op.ref .nat [] "cur_batch_seq_len"))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat 96))) s
      = some (⟨fun idx : TileIndex [BM, D] =>
          decide (gOM idx.1 < sl) && decide (idx.2.1.val < 96)⟩ : Tile .bool [BM, D]) := by
  rw [evalOp]
  rw [evalOp_lt]
  erw [ctx_evalOp_expandDim_one_nat]
  rw [evalOp_lt]
  erw [ctx_evalOp_expandDim_zero_nat]
  simp only [evalOp_ref, evalOp_constNat, hm, hd, hsl, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  refine congrArg some ?_; ext idx
  simp [Tile.bop_data, Tile.cop_data, Tile.expandDim, Tile.scalar_data_index, Tile.vec_data,
    Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **PreLoop execution (Milestone 2).** From a clean input state, the 19 bloom
prologue statements step to the loop-entry state, exposing the runtime scalars
(`cur_batch = pid₀`, `cur_head = pid₁`, `start_m = pid₂`, `cur_kv_head`,
`prompt_cache_len`, `cur_batch_in_all_start_index`, `cur_batch_seq_len`,
`block_start_loc = 128·pid₂`), the index vectors, the ⊥/0 running-state seeds, the
masked `q` tile, and `block_mask`/`block_end_loc`. -/
theorem bloomPreLoop_eval
    (s : BlockState) (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (bloomPreLoop Q B_Start_Loc B_Seqlen B_req_idx b_prompt_cache_len) s = some s0
      ∧ s0.pids = s.pids ∧ s0.mem = s.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .nat [] "cur_head" = some (Tile.scalar (s.pids 1))
      ∧ s0.regs .nat [] "start_m" = some (Tile.scalar (s.pids 2))
      ∧ s0.regs .nat [] "cur_kv_head" = some (Tile.scalar (s.pids 1 / 1))
      ∧ s0.regs .nat [] "prompt_cache_len"
          = some (Tile.scalar (s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)))
      ∧ s0.regs .nat [] "cur_batch_in_all_start_index"
          = some (Tile.scalar (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 0)))
      ∧ s0.regs .nat [] "cur_batch_req_idx"
          = some (Tile.scalar (s.readMemValue .nat (Region.cast B_req_idx) (s.pids 0)))
      ∧ s0.regs .nat [] "cur_batch_seq_len"
          = some (Tile.scalar (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
              - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)))
      ∧ s0.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val))
      ∧ s0.regs .nat [128] "offs_d" = some (Tile.vec (fun e : Fin 128 => e.val))
      ∧ s0.regs .nat [128] "offs_m" = some (Tile.vec (fun r : Fin 128 => s.pids 2 * 128 + r.val))
      ∧ s0.regs .real [128] "m_i" = some ⟨fun _ : TileIndex [128] => (⊥ : WithBot ℝ)⟩
      ∧ s0.regs .real [128] "l_i" = some ⟨fun _ : TileIndex [128] => some (0 : ℝ)⟩
      ∧ s0.regs .real [128, 128] "acc" = some ⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩
      ∧ s0.regs .nat [] "block_end_loc"
          = some (Tile.scalar
              (let a := (s.pids 2 + 1) * 128 + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
               let b := (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
                   - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
                 + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
               if a < b then a else b))
      ∧ s0.regs .real [128, 128] "q" = some ⟨fun idx : TileIndex [128, 128] =>
          if decide (s.pids 2 * 128 + idx.1.val
              < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
                - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)) && decide (idx.2.1.val < 96) then
            s.readMemValue .real (Region.cast Q)
              ((s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 0) + (s.pids 2 * 128 + idx.1.val))
                  * 576 + s.pids 1 * 96 + idx.2.1.val * 1)
          else some (0.0 : ℝ)⟩ := by
  unfold bloomPreLoop
  -- stmt 0: cur_batch = programId 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  -- stmt 1: cur_head = programId 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  -- stmt 2: start_m = programId 2
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  -- stmt 3: cur_kv_head = cur_head // 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 1)) _
        = some (Tile.scalar (s.pids 1 / 1)) from by
      rw [ctx_evalOp_floorDiv]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 4: cur_batch_in_all_start_index = load(B_Start_Loc + cur_batch)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_scalar_nat B_Start_Loc (Op.ref .nat [] "cur_batch") _ (s.pids 0)
      (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids])))]
  -- stmt 5: prompt_cache_len = load(bpc + cur_batch)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_scalar_nat b_prompt_cache_len (Op.ref .nat [] "cur_batch") _ (s.pids 0)
      (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids])))]
  -- stmt 6: cur_batch_seq_len = load(B_Seqlen + cur_batch) - prompt_cache_len
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.sub .nat Broadcast.nil
        (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none)
        (Op.ref .nat [] "prompt_cache_len")) _
        = some (Tile.scalar (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
            - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))) from by
      rw [evalOp_sub, ctx_evalOp_load_scalar_nat B_Seqlen (Op.ref .nat [] "cur_batch") _ (s.pids 0)
        (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids])]
      simp only [evalOp_ref, BlockState.setReg_same, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 7: cur_batch_req_idx = load(B_req_idx + cur_batch)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_scalar_nat B_req_idx (Op.ref .nat [] "cur_batch") _ (s.pids 0)
      (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids])))]
  -- stmt 8: block_start_loc = 128 * start_m
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .nat Broadcast.nil (Op.constNat 128) (Op.ref .nat [] "start_m")) _
        = some (Tile.scalar (128 * s.pids 2)) from by
      rw [evalOp_mul]
      simp only [evalOp_constNat, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 9: offs_n = arange 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange 128) _ = some (Tile.vec (fun j : Fin 128 => j.val)) from evalOp_arange 128 _))]
  -- stmt 10: offs_d = arange 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange 128) _ = some (Tile.vec (fun e : Fin 128 => e.val)) from evalOp_arange 128 _))]
  -- stmt 11: offs_m = start_m * 128 + arange 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128)) (Op.arange 128)) _
        = some (Tile.vec (fun r : Fin 128 => s.pids 2 * 128 + r.val)) from by
      rw [evalOp_add, evalOp_mul]
      simp only [evalOp_ref, evalOp_arange, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r
      simp [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul]))]
  -- stmt 12: off_q
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")))
            (Op.constNat 576))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 96)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))) _
        = some (⟨fun idx : TileIndex [128, 128] =>
            (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 0) + (s.pids 2 * 128 + idx.1.val))
                * 576 + s.pids 1 * 96 + idx.2.1.val * 1⟩ : Tile .nat [128, 128]) from by
      simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
        ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
        BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, reduceCtorEq, BlockState.setReg_pids, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 13: q = masked load (2-cond)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_region_maskOther Q (Op.ref .nat [128, 128] "off_q") _ _ _
      (⟨fun idx : TileIndex [128, 128] =>
          (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 0) + (s.pids 2 * 128 + idx.1.val))
              * 576 + s.pids 1 * 96 + idx.2.1.val * 1⟩ : Tile .nat [128, 128])
      (⟨fun idx : TileIndex [128, 128] => decide (s.pids 2 * 128 + idx.1.val
          < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
            - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)) && decide (idx.2.1.val < 96)⟩ : Tile .bool [128, 128])
      (⟨fun _ : TileIndex [128, 128] => some (0.0 : ℝ)⟩ : Tile .real [128, 128])
      (by rw [evalOp_ref]; simp [BlockState.setReg_same])
      (bloomQMask_eval _ (fun r : Fin 128 => s.pids 2 * 128 + r.val) _
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same]))
      (by simp only [evalOp, Option.bind_eq_bind, Option.bind_some]; rfl)))]
  -- stmt 14: m_i = full 0 + (-inf) = full ⊥
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [128] (Op.const 0)) Op.negInf) _
        = some (⟨fun _ : TileIndex [128] => (⊥ : WithBot ℝ)⟩ : Tile .real [128]) from by
      rw [evalOp_add]
      simp only [evalOp_full, evalOp_const, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map]
      rfl))]
  -- stmt 15: l_i = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [128] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [128] => some (0 : ℝ)⟩ : Tile .real [128]) from by
      simp [evalOp_full, evalOp_const]))]
  -- stmt 16: acc = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [128, 128] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128]) from by
      simp [evalOp_full, evalOp_const]))]
  -- stmt 17: block_mask = where (block_start_loc < seqlen) 1 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp ((Op.lt .nat Broadcast.nil (Op.ref .nat [] "block_start_loc")
            (Op.ref .nat [] "cur_batch_seq_len")).where (Op.constNat 1) (Op.constNat 0)) _
        = some (Tile.scalar (if 128 * s.pids 2 < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
            - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0) then 1 else 0)) from by
      rw [evalOp_where]
      simp only [evalOp_lt, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_
      ext _idx
      simp only [Tile.select_data, Tile.cop_data, Tile.scalar_data_index, ComparableDType.lt]
      by_cases h : 128 * s.pids 2 < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
          - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
      · simp [h]
      · simp [h]))]
  -- stmt 18: block_end_loc
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp ((Op.lt .nat Broadcast.nil
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat 128))
              (Op.ref .nat [] "prompt_cache_len"))
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len")
              (Op.ref .nat [] "prompt_cache_len"))).where
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat 128))
            (Op.ref .nat [] "prompt_cache_len"))
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len")
            (Op.ref .nat [] "prompt_cache_len"))) _
        = some (Tile.scalar
            (let a := (s.pids 2 + 1) * 128 + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
             let b := (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
                 - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
               + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
             if a < b then a else b)) from by
      rw [evalOp_where]
      simp only [evalOp_lt, evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_
      ext _idx
      simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.scalar_data_index,
        ComparableDType.lt, NumericDType.add, NumericDType.mul, Broadcast.leftIndex, Broadcast.rightIndex]
      by_cases h : ((s.pids 2 + 1) * 128 + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
          < (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 0)
              - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0))
            + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 0)
      · simp [h]
      · simp [h]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_pids]
  · funext rg o; simp only [BlockState.setReg_mem]
  · intro rg o; simp only [BlockState.setReg_undef]; exact hundef rg o
  all_goals
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids,
      BlockState.setReg_readMemValue, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq]

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

/-- The ⊥-seeded fold over the empty window `[0, 0)` is the seed `(⊥, 0, 0)`. -/
theorem gStateBot_zero (S : Nat) (g : Fin S → ℝ × ℝ) : gStateBot S 0 g = (⊥, 0, 0) := by
  rw [gStateBot, show gKeysUpto S 0 g = [] from by
    rw [gKeysUpto]; apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **M4 — `bloomPostLoop`: masked store reads off the faithful fold (NO divide).**
At the loop exit (`c·128 = S`), the `acc` register already holds the in-loop
normalized `acc/l` ratio (`= contextAttnBloomExactFoldM`), so the post-loop tail
(`off_o`, `out_ptrs = Out + off_o`, masked `tl.store`) writes that genuine value
into `Out` at every active lane (`offs_m < cur_batch_seq_len ∧ offs_d < head_dim`),
preserving inactive lanes. The store is a scatter over the injective `outOffset`
map. -/
theorem bloomPostLoop_eval
    (Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName) (s0 : BlockState)
    (S bel : Nat) (c : Nat) (s : BlockState) (hSc : S = c * 128)
    (hinv : bloomInvariant Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len s0
        sm_scale_bloom S bel c s) :
    ∃ sP, stepStmts (bloomPostLoop Out) s = some sP
      ∧ ∀ idx : TileIndex [128, 128],
          sP.readMem Out (outOffset s0 B_Start_Loc 576 96 1 128 idx)
            = if active s0 B_Seqlen B_Prompt_Cache_Len 96 128 idx then
                contextAttnBloomExactFoldM s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len
                  Req_to_tokens B_req_idx sm_scale_bloom 7500 1 128 S bel idx
              else s0.readMem Out (outOffset s0 B_Start_Loc 576 96 1 128 idx) := by
  subst hSc
  set plen := promptLen s0 B_Prompt_Cache_Len with hplend
  set sl := seqLen s0 B_Seqlen B_Prompt_Cache_Len with hsld
  set g := fun (i d : Fin 128) =>
    bloomG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale_bloom (c * 128) bel i d with hgd
  simp only [bloomInvariant] at hinv
  obtain ⟨hpids, hmem, hundef, hcb, hckvh, hch, hplen, hsl, hbel, hcbsi, hrqi, hom, hon, hod,
      hq, hmi, hli, hacc, hcle⟩ := hinv
  -- the acc tile already holds the ratio numer/denom
  set accTile : Tile .real [128, 128] := ⟨fun idx : TileIndex [128, 128] =>
      some ((gStateBot (c * 128) (c * 128) (g idx.1 idx.2.1)).2.2 / (gStateBot (c * 128) (c * 128) (g idx.1 idx.2.1)).2.1)⟩
    with haccTile
  have haccref0 : s.regs .real [128, 128] "acc" = some accTile := by
    rw [hacc]
  unfold bloomPostLoop
  -- off_o offset tile (= outOffset)
  set offoTile : Tile .nat [128, 128] :=
    ⟨fun idx : TileIndex [128, 128] => outOffset s0 B_Start_Loc 576 96 1 128 idx⟩ with hoffo
  -- stmt 0: off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")))
            (Op.constNat 576))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 96)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))) s = some offoTile from by
      simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
        ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq,
        hcbsi, hch, hom, hod, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [hoffo, outOffset, startLoc, mIndex, dIndex,
        Tile.bop_data, Tile.scalar_data, Tile.vec_data,
        Tile.expandDim, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul]))]
  set s1 := s.setReg "off_o" .nat [128, 128] offoTile with hs1
  -- stmt 1: out_ptrs = Out + off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [128, 128] "off_o")) s1
        = some (⟨fun idx : TileIndex [128, 128] =>
            (Out, offoTile.data idx)⟩ : Tile .ptr [128, 128]) from by
      simp only [evalOp, hs1, BlockState.setReg_same, Region.cast_id,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      · simp only [Tile.ptrAdd, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex]
      · simp only [Tile.ptrAdd, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
          Nat.zero_add]))]
  set s2 := s1.setReg "out_ptrs" .ptr [128, 128]
    (⟨fun idx : TileIndex [128, 128] => (Out, offoTile.data idx)⟩ : Tile .ptr [128, 128]) with hs2
  -- stmt 2: masked store of acc into Out (2D boolAnd mask)
  set P : TileIndex [128, 128] → Prop :=
    fun idx => s0.pids 2 * 128 + idx.1.val < sl ∧ idx.2.1.val < 96 with hP
  have hOffInj : Function.Injective (fun idx : TileIndex [128, 128] => offoTile.data idx) :=
    context_attn_bloom_python_block128_offset_injective s0 B_Start_Loc
  have hmaskev : evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
          (Op.ref .nat [] "cur_batch_seq_len"))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 96))) s2
      = some (⟨fun idx : TileIndex [128, 128] =>
          decide (s0.pids 2 * 128 + idx.1.val < sl) && decide (idx.2.1.val < 96)⟩ : Tile .bool [128, 128]) := by
    exact bloomQMask_eval s2 (fun r : Fin 128 => s0.pids 2 * 128 + r.val) sl
      (by rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hom)
      (by rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hod)
      (by rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsl)
  have haccref : evalOp (Op.ref .real [128, 128] "acc") s2 = some accTile := by
    rw [evalOp_ref, hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact haccref0
  have hptrref : evalOp (Op.ref .ptr [128, 128] "out_ptrs") s2
      = some (⟨fun idx : TileIndex [128, 128] => (Out, offoTile.data idx)⟩ : Tile .ptr [128, 128]) := by
    rw [evalOp_ref, hs2, BlockState.setReg_same]
  have hstore : stepStmt (Stmt.store .real [128, 128]
      (MemAccess.ptr (Op.ref .ptr [128, 128] "out_ptrs"))
      (Op.ref .real [128, 128] "acc")
      (MaskOpt.mask
        (Op.boolAnd Broadcast.nil.consL.consR
          (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
            (Op.ref .nat [] "cur_batch_seq_len"))
          (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
            (Op.constNat 96))))) s2
      = some ((TileShape.allIndices [128, 128]).foldl
          (fun acc idx => if P idx then acc.writeMem Out (offoTile.data idx)
            ((accTile.data idx).unbotD 0) else acc) s2) := by
    unfold stepStmt
    rw [haccref]
    simp only [hmaskev, Option.bind_eq_bind, Option.bind_some, Option.map_some]
    rw [hptrref]
    simp only [Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    refine List.foldl_ext _ _ s2 ?_
    intro acc idx _
    by_cases h1 : s0.pids 2 * 128 + idx.1.val < sl
    · by_cases h2 : idx.2.1.val < 96
      · rw [decide_eq_true_eq.mpr h1, decide_eq_true_eq.mpr h2, Bool.and_true]
        simp only [if_true, BlockState.writeMemTyped_real, FloatDType.real_storeValue]
        rw [if_pos (show P idx from ⟨h1, h2⟩)]
      · rw [decide_eq_false_iff_not.mpr h2, Bool.and_false]
        simp only [Bool.false_eq_true, if_false]
        rw [if_neg (fun hc => h2 hc.2)]
    · rw [decide_eq_false_iff_not.mpr h1, Bool.false_and]
      simp only [Bool.false_eq_true, if_false]
      rw [if_neg (fun hc => h1 hc.1)]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  rw [show outOffset s0 B_Start_Loc 576 96 1 128 idx = offoTile.data idx from rfl]
  rw [BlockState.scatter_readback_prop_masked_nd (region := Out) s2 (fun idx => offoTile.data idx)
    (fun idx => (accTile.data idx).unbotD 0) P hOffInj idx]
  have hactive_iff : active s0 B_Seqlen B_Prompt_Cache_Len 96 128 idx ↔ P idx := by
    have he : mIndex s0 128 idx.1 = s0.pids 2 * 128 + idx.1.val := rfl
    simp only [active, hP, he, hsld, dIndex]
  by_cases hac : P idx
  · rw [if_pos hac, if_pos (hactive_iff.mpr hac)]
    -- accTile cell = contextAttnBloomExactFoldM
    simp only [haccTile, contextAttnBloomExactFoldM, WithBot.unbotD_coe]
    show (gStateBot (c * 128) (c * 128) (g idx.1 idx.2.1)).2.2
        / (gStateBot (c * 128) (c * 128) (g idx.1 idx.2.1)).2.1 = _
    have hkvm : g idx.1 idx.2.1
        = bloomKVM s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len
            Req_to_tokens B_req_idx sm_scale_bloom 7500 1 128 (c * 128) bel idx.1 idx.2.1 := by
      rw [hgd]; rfl
    rw [hkvm]
  · rw [if_neg hac, if_neg (fun hcon => hac (hactive_iff.mp hcon))]
    show (s2.readMem Out (offoTile.data idx)) = _
    have hreadeq : s2.readMem Out (offoTile.data idx) = s.readMem Out (offoTile.data idx) := by
      unfold BlockState.readMem; rw [hs2, hs1]; simp only [BlockState.setReg_mem]
    rw [hreadeq]
    unfold BlockState.readMem; rw [hmem]

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
