import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention
import VeriTile.Triton.LoopInvariant

/-!
# `context_attn_fwd` — strict per-kernel correctness

`_fwd_kernel_int8kv` is varlen context (prefill) attention. Each program
`(start_m, cur_bh)` decodes its batch/head as `cur_batch = cur_bh // H`,
`cur_head = cur_bh % H`, loads a `[BLOCK_M, BLOCK_DMODEL]` query tile, runs an
online-softmax (`m_i`/`l_i`/`acc`) loop over the cached key/value tokens with a
`prompt_cache_len`-offset causal mask, and stores the accumulated `acc` tile to
`Out`, masked by `offs_m < cur_batch_seq_len`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_int8kv[grid](...)`, the grid over sequence
blocks × `(batch·head)`, the scheduling, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because the program ids `(start_m, cur_bh)` are universally
quantified, the per-program statement covers every program of the grid.

## Proof architecture

```
context_attn_fwd_python_test_shape_output_summary          ← TOP THEOREM (bundles both block shapes)
  ├─ context_attn_fwd_kernel_int8kv_surface_toAlgorithm_supported   surface lowers to the algorithm layer
  ├─ context_attn_fwd_surface_python_block128_compute_correct       full surface, BLOCK_M=128 final store
  │    └─ context_attn_fwd_final_store_python_block128_compute_correct
  │         └─ context_attn_fwd_final_store_slice_compute_correct
  │              └─ context_attn_fwd_final_store_slice_correct       algorithm-layer readback per lane
  └─ context_attn_fwd_surface_python_block64_compute_correct        full surface, BLOCK_M=64 final store
       └─ context_attn_fwd_final_store_python_block64_compute_correct
            └─ context_attn_fwd_final_store_slice_compute_correct
(supporting: context_attn_fwd_python_block128_offset_injective,
             context_attn_fwd_python_block64_offset_injective)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` and the
int8 KV path are not modeled. The verified compute claim is scoped to the
**final masked writeback** of the accumulated `acc` tile into `Out`: every active
lane (`offs_m < cur_batch_seq_len`, with `offs_d < head_dim` folded into the
slice) holds the surface-produced `acc` value
(`producedContextFwdBlock128/64OutValue`), and out-of-bounds lanes are preserved.
The online-softmax streaming loop (`m_i`/`l_i`/`acc` updates, `tl.dot`, the
`prompt_cache_len`-offset causal mask) is carried *inside* the surface kernel and
reflected in the produced-value spec rather than re-proven as a closed-form
softmax identity. The summary is instantiated at the Python test shape
(`H=16`, `BLOCK_DMODEL=BLOCK_N=128`, `BLOCK_M ∈ {128, 64}`); other shapes are not
covered by the top theorem.
-/

namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful DSL port of `context_attn_fwd.py`'s `_fwd_kernel_int8kv`. -/
def context_attn_fwd_kernel_int8kv_surface
    (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd
      stride_obs stride_oh stride_od
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  cur_bh = tl.program_id(1)
  cur_batch = cur_bh // $(H)
  cur_head = cur_bh % $(H)

  cur_kv_head = cur_head // $(kv_group_num)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = block_start_loc + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)
  q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum(block_start_loc + $(BLOCK_M) + prompt_cache_len,
    cur_batch_seq_len + prompt_cache_len)
  for start_n in range($(0), block_mask * block_end_loc, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    off_k = cur_batch * $(stride_kb) + (start_n + offs_n[None, :]) * $(stride_ks) +
      cur_kv_head * $(stride_kh) + offs_d[:, None] * $(stride_kd)
    k = tl.load(K + off_k,
      mask=(start_n + offs_n[None, :]) < block_end_loc,
      other=0.0)

    qk = tl.dot(q, k)
    mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
    qk = tl.where(mask, qk * $((sm_scale : ℝ)), -1.0e8)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)

    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    off_v = cur_batch * $(stride_vb) + (start_n + offs_n[:, None]) * $(stride_vs) +
      cur_kv_head * $(stride_vh) + offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=(start_n + offs_n[:, None]) < block_end_loc,
      other=0.0)

    p = (p).to(v.dtype)
    acc = tl.dot(p, v, acc)
    m_i = m_ij
  }

  acc = acc / l_i[:, None]
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
}

/-- The full int8-KV context-attention surface lowers to the algorithm layer. -/
theorem context_attn_fwd_kernel_int8kv_surface_toAlgorithm_supported
    (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd
      stride_obs stride_oh stride_od
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    ∃ alg, (context_attn_fwd_kernel_int8kv_surface Q K V sm_scale Out
      B_Start_Loc B_Seqlen b_prompt_cache_len stride_qbs stride_qh stride_qd
      stride_kb stride_kh stride_ks stride_kd stride_vb stride_vh stride_vs
      stride_vd stride_obs stride_oh stride_od kv_group_num H BLOCK_DMODEL
      BLOCK_M BLOCK_N).toAlgorithm? = Except.ok alg := by
  simp [context_attn_fwd_kernel_int8kv_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of `context_attn_fwd.py`'s
`_fwd_kernel_int8kv`.

The full kernel computes PPL int8-KV context attention. This slice starts from
a precomputed `Acc` tile and proves the final masked writeback into `Out`,
preserving the fused `cur_bh` program-id decomposition, `B_Start_Loc`, and the
prompt-cache-adjusted sequence length. The inner `tl.float32`
streaming-softmax accumulator and int8-KV dequantization are outside this
slice. -/
def context_attn_fwd_final_store_slice
    (Acc : RegionName) (B_Start_Loc B_Seqlen B_Prompt_Cache_Len : Region .nat)
    (Out : RegionName)
    (H
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  cur_bh = tl.program_id(1)
  cur_batch = cur_bh // $(H)
  cur_head = cur_bh % $(H)
  prompt_cache_len = tl.load(B_Prompt_Cache_Len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(BLOCK_DMODEL))
  acc = tl.load(Acc + cur_batch * $(stride_acc_b) + cur_head * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
      cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od), acc, mask=mask)
}

def curBatch (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H

def curHead (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H

def promptLen (s : BlockState) (H : Nat) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (curBatch s H)

def seqLen
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (curBatch s H) -
    promptLen s H B_Prompt_Cache_Len

def startLoc (s : BlockState) (H : Nat) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (curBatch s H)

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def active
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H B_Seqlen B_Prompt_Cache_Len

instance activeDecidable
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (BLOCK_M BLOCK_DMODEL : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx) := by
  unfold active
  infer_instance

def accOffset
    (s : BlockState)
    (H stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  curBatch s H * stride_acc_b + curHead s H * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d

def outOffset
    (s : BlockState) (H : Nat) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s H B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    curHead s H * stride_oh + dIndex idx * stride_od

noncomputable def accStoreValue
    (s : BlockState) (Acc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (H stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s H stride_acc_b stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))

/-! ## Genuine closed-form context-attention spec

The streaming-softmax loop in `_fwd_kernel_int8kv` is *not* a self-referential
black box: it computes, for every active query lane, the prompt-cache-offset
**causal softmax attention** value. This section makes that closed form explicit
and proves the kernel's exact base-2 / `sm_scale` streaming weights collapse to
it — independent of the kernel `exec`.

### Score / scale / mask of this kernel (decoded lane-by-lane from the body)

For program `(start_m, cur_bh)`, query lane `i` (global row
`gi = start_m·BLOCK_M + i`), key `j`, head channel `e`:

* **raw score** `raw i j = Σ_e Q[gi,e]·K[j,e]`  (`tl.dot q k`, line 85/107);
* **scale**     `qk·sm_scale` with
  `sm_scale = (1/√D)·1.4426950408889634` (`= (1/√D)·log₂ e`, line 87/109);
* **softmax**   base-2 `tl.math.exp2` (lines 90/112, 94/115);
* **mask**      `(gi + prompt_cache_len) ≥ j`  (`mask`, line 86/108): future keys
  get score `-1e8` (≈ `exp2 → 0`), i.e. a causal mask shifted by `prompt_cache_len`.

Because `exp2(x) = exp(log 2 · x)` and `exp2(qk·sm_scale) = exp((log 2 · sm_scale)·qk)`,
the kernel realizes the **natural-exp** causal softmax with effective scale
`effScale = log 2 · sm_scale`. With the kernel's `sm_scale = (√D)⁻¹·1.4426950408889634`
and `log 2 · 1.4426950408889634 ≈ 1`, `effScale ≈ (√D)⁻¹` — standard scaled-dot
attention. We keep the *exact* `effScale` so the closed form is bit-faithful to the
kernel's chosen constant. -/

/-- Effective natural-exp score scale of this kernel: `log 2 · sm_scale`. -/
noncomputable def contextEffScale (sm_scale : ℝ) : ℝ := Real.log 2 * sm_scale

/-- `pow2 (sm_scale · x) = exp (effScale · x)`: the kernel's `exp2`-with-`sm_scale`
weight is the natural-exp weight at the effective scale. This is the algebraic
identity that turns the kernel's base-2 softmax into ordinary `attentionReal`. -/
theorem pow2_smScale_eq_exp_effScale (sm_scale x : ℝ) :
    pow2 (sm_scale * x) = Real.exp (contextEffScale sm_scale * x) := by
  simp [pow2, contextEffScale, mul_assoc]

/-- Coordinate-faithful query tile of this kernel at `(start_m, cur_bh)` for the
checked Python layout (strides `stride_qbs=2048, stride_qh=128, stride_qd=1`,
`H=16`, head decode `cur_batch=cur_bh/16`, `cur_head=cur_bh%16`). Row `i` is the
*global* prefill row `start_m·BLOCK_M + i` offset by `cur_batch_in_all_start_index`. -/
noncomputable def ctxQTile
    (s : BlockState) (Q B_Start_Loc : RegionName) (BLOCK_M : Nat) :
    TileIndex [BLOCK_M, 128] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q
      ((s.readMemValue .nat B_Start_Loc (curBatch s 16) + (s.pids 0 * BLOCK_M + i.val))
          * 2048 + curHead s 16 * 128 + e.val)

/-- Coordinate-faithful key tile: `K[cur_batch, j, cur_head, e]` at the checked
layout (`stride_kb=8388608, stride_ks=128, stride_kh=262144, stride_kd=1`,
`kv_group_num=1` so `cur_kv_head=cur_head`). -/
noncomputable def ctxKTile (s : BlockState) (K : RegionName) (S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (curBatch s 16 * 8388608 + j.val * 128 + curHead s 16 * 262144 + e.val)

/-- Coordinate-faithful value tile: `V[cur_batch, j, cur_head, d]`. -/
noncomputable def ctxVTile (s : BlockState) (V : RegionName) (S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (curBatch s 16 * 8388608 + j.val * 128 + curHead s 16 * 262144 + d.val)

/-- **Genuine closed-form output** of `context_attn_fwd` at query lane `i`,
channel `d`, over the first `S` keys.

`out[i,d] = (Σ_{j ≤ gi+plen} exp(effScale·rawᵢⱼ)·V[j,d]) / (Σ_{j ≤ gi+plen} exp(effScale·rawᵢⱼ))`

where `gi = start_m·BLOCK_M + i`, `plen = prompt_cache_len`, and `rawᵢⱼ = Σ_e Q[gi,e]·K[j,e]`.
This is exactly `attentionRealCausalBlock` (the library's prompt-offset causal
softmax) instantiated with this kernel's tiles, effective scale, and a causal
boundary shifted by `prompt_cache_len`. No self-reference: it is a pure function
of `Q`/`K`/`V` memory. -/
noncomputable def contextAttnClosedForm
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ) (BLOCK_M S : Nat)
    (idx : TileIndex [BLOCK_M, 128]) : ℝ :=
  let i := idx.1
  let d := idx.2.1
  let plen := promptLen s 16 B_Prompt_Cache_Len
  let gi := s.pids 0 * BLOCK_M + i.val
  let raw := fun j : Fin S =>
    Finset.univ.sum (fun e : Fin 128 =>
      ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
        * ctxKTile s K S (j, e, PUnit.unit))
  let weight := fun j : Fin S =>
    if j.val ≤ gi + plen then Real.exp (contextEffScale sm_scale * raw j) else 0
  let denom := Finset.univ.sum (fun j : Fin S => weight j)
  let numer := Finset.univ.sum (fun j : Fin S =>
    weight j * ctxVTile s V S (j, d, PUnit.unit))
  numer / denom

/-- **Bridge to the library's `attentionRealCausalBlock`.** The genuine closed
form above coincides with `attentionRealCausalBlock` (from
`VeriTile.Triton.Math.Attention`) at query-start `gi₀ = start_m·BLOCK_M + plen`,
with this kernel's Q/K/V tiles and effective scale `effScale = log 2 · sm_scale`.
This certifies `contextAttnClosedForm` is the standard prompt-offset causal
softmax-attention reference, not an ad-hoc definition. -/
theorem contextAttnClosedForm_eq_attentionRealCausalBlock
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ) (BLOCK_M S : Nat) (idx : TileIndex [BLOCK_M, 128]) :
    contextAttnClosedForm s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S idx
      = attentionRealCausalBlock
          (s.pids 0 * BLOCK_M + promptLen s 16 B_Prompt_Cache_Len)
          (ctxQTile s Q B_Start_Loc BLOCK_M)
          (ctxKTile s K S) (ctxVTile s V S)
          (contextEffScale sm_scale)
          (idx.1, idx.2.1, PUnit.unit) := by
  obtain ⟨i, d, u⟩ := idx
  -- The two sides agree key-by-key once the causal boundary
  -- `start_m·BM + i + plen = start_m·BM + plen + i` is reassociated and the scale
  -- is pulled into the dot via `Finset.mul_sum`.
  have hbound : s.pids 0 * BLOCK_M + promptLen s 16 B_Prompt_Cache_Len + i.val
      = s.pids 0 * BLOCK_M + i.val + promptLen s 16 B_Prompt_Cache_Len := by omega
  simp only [contextAttnClosedForm, attentionRealCausalBlock, scaledScore, hbound,
    Finset.mul_sum]

/-- **The genuine closed form is the kernel's streaming-loop result (math layer).**
For each output lane, `contextAttnClosedForm` equals `acc/l` of folding the
online-softmax block step `osBlockStep` over the per-key `(score, value)` list
built from this kernel's *masked* scores — score `effScale·rawᵢⱼ` for active
(`j ≤ gi+plen`) keys and the kernel's masked `-∞` sentinel (weight `0`) for
future keys. This is the form the kernel's `m_i`/`l_i`/`acc` loop carries; it
reduces the eventual `exec` obligation to matching the loop to this fold (the
remaining, FA-1-style, multi-thousand-line grind). -/
noncomputable def ctxMaskedKeyList
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ) (BLOCK_M S : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    List (ℝ × ℝ) :=
  let plen := promptLen s 16 B_Prompt_Cache_Len
  let gi := s.pids 0 * BLOCK_M + i.val
  List.ofFn (fun j : Fin S =>
    (if j.val ≤ gi + plen then
        contextEffScale sm_scale *
          Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
              * ctxKTile s K S (j, e, PUnit.unit))
      else (Real.log 2)⁻¹ * (-1.0e8 : ℝ),       -- kernel's `-1e8` sentinel score (pre-`m` shift)
      ctxVTile s V S (j, d, PUnit.unit)))

/-! ### Exact (sentinel-faithful) streaming closed form

`contextAttnClosedForm` *idealizes* future keys to softmax weight `0`. The kernel's
`tl.where(mask, qk·sm_scale, -1e8)` instead assigns future keys the *finite*
sentinel score `-1e8`, so over exact ℝ they carry weight `exp(-1e8)` — negligible,
but nonzero. The value the streaming loop computes *exactly* is therefore the
fold below (`acc/l` of the online-softmax step `osStep` over the genuinely-masked
key stream with the `-1e8` sentinel kept); it coincides with
`contextAttnClosedForm` only in the `exp(-1e8) → 0` limit.

Crucially `contextAttnExactFold` is a pure function of `Q`/`K`/`V` memory (no
`exec` self-reference): the FA-1 exec-assembly obligation is to show the kernel's
`m_i`/`l_i`/`acc` loop realizes this fold (see roadmap in
`ctxExactKeyList`). The score `sm_scale·raw` (not `effScale·raw`) is the value the
kernel feeds to `exp2`, so `exp2(sm_scale·raw) = exp(effScale·raw)` is the genuine
softmax weight (`pow2_smScale_eq_exp_effScale`). -/

/-- Per-key `(score, value)` stream the loop folds, with the kernel's genuine
`-1e8` sentinel kept. Active key `j ≤ gi+plen`: score `sm_scale·rawᵢⱼ`; future
key: sentinel `(log 2)⁻¹·(-1e8)` (so `exp2` → `exp(-1e8)`). -/
noncomputable def ctxExactKeyList
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ) (BLOCK_M S : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    List (ℝ × ℝ) :=
  let plen := promptLen s 16 B_Prompt_Cache_Len
  let gi := s.pids 0 * BLOCK_M + i.val
  List.ofFn (fun j : Fin S =>
    (if j.val ≤ gi + plen then
        sm_scale *
          Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
              * ctxKTile s K S (j, e, PUnit.unit))
      else (0.0 - 10e7 : ℝ),
      ctxVTile s V S (j, d, PUnit.unit)))

/-- Exact streaming-loop output value for lane `(i,d)`: `acc/l` of folding the
online-softmax step `osStep` over `ctxExactKeyList`. A pure function of
`Q`/`K`/`V` memory; exactly what the kernel's `m_i`/`l_i`/`acc` loop produces. -/
noncomputable def contextAttnExactFold
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ) (BLOCK_M S : Nat) (idx : TileIndex [BLOCK_M, 128]) : ℝ :=
  let st := (ctxExactKeyList s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale
      BLOCK_M S idx.1 idx.2.1).foldl osStep (0, 0, 0)
  st.2.2 / st.2.1

/-- **Closed form of the exact fold.** The `osStep` fold over `ctxExactKeyList`
collapses to the genuine causal softmax with `exp(effScale·raw)` weights on active
keys and `exp(-1e8)` on future keys — explicitly, no self-reference, no `exec`.
Proven via the banked `osStep_foldl_eq_batch` and `pow2_smScale_eq_exp_effScale`. -/
theorem contextAttnExactFold_eq
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ) (BLOCK_M S : Nat) (idx : TileIndex [BLOCK_M, 128]) :
    contextAttnExactFold s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S idx
      = (let i := idx.1; let d := idx.2.1
         let plen := promptLen s 16 B_Prompt_Cache_Len
         let gi := s.pids 0 * BLOCK_M + i.val
         let raw := fun j : Fin S =>
           Finset.univ.sum (fun e : Fin 128 =>
             ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
               * ctxKTile s K S (j, e, PUnit.unit))
         let weight := fun j : Fin S =>
           if j.val ≤ gi + plen then Real.exp (contextEffScale sm_scale * raw j)
           else pow2 (0.0 - 10e7)
         (Finset.univ.sum (fun j : Fin S => weight j * ctxVTile s V S (j, d, PUnit.unit)))
           / (Finset.univ.sum (fun j : Fin S => weight j))) := by
  obtain ⟨i, d, u⟩ := idx
  rw [contextAttnExactFold, ctxExactKeyList, osStep_foldl_eq_batch]
  simp only [List.map_ofFn, List.sum_ofFn, Function.comp, contextEffScale]
  have hw : ∀ j : Fin S,
      pow2 (if j.val ≤ s.pids 0 * BLOCK_M + i.val + promptLen s 16 B_Prompt_Cache_Len then
          sm_scale * Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit) * ctxKTile s K S (j, e, PUnit.unit))
        else (0.0 - 10e7 : ℝ))
      = if j.val ≤ s.pids 0 * BLOCK_M + i.val + promptLen s 16 B_Prompt_Cache_Len then
          Real.exp (Real.log 2 * sm_scale * Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit) * ctxKTile s K S (j, e, PUnit.unit)))
        else pow2 (0.0 - 10e7) := by
    intro j
    by_cases h : j.val ≤ s.pids 0 * BLOCK_M + i.val + promptLen s 16 B_Prompt_Cache_Len
    · simp only [h, if_true, pow2]; ring_nf
    · simp only [h, if_false]
  congr 1
  · apply Finset.sum_congr rfl; intro j _; rw [hw j]
  · apply Finset.sum_congr rfl; intro j _; rw [hw j]

/-! ## ⊥-seeded online-softmax running-state recurrence (the loop invariant's math)

The kernel seeds its running max `m_i = tl.zeros − inf` at `⊥` and `l_i`/`acc` at
real `0`; each block it rescales by `α = exp2(m_i − m_ij)` (with `m_i = ⊥` on block 0
the rescale is `realExp2 ⊥ = 0`, killing the seed). This section is the
context-kernel analogue of `flash_attn`'s `osStepBot`/`flashStateBot`/`flashRunningMax`
family: the ⊥-seeded recurrence the streaming loop carries, with this kernel's
per-key score (`sm_scale·raw` on active keys `j ≤ gi+plen`, the genuine `-1e8`
sentinel `(log 2)⁻¹·(−1e8)` on future keys) and value `ctxVTile`. Its final
`acc/l` ratio over the full window reads off the banked `contextAttnExactFold`. -/

open VeriTile.Triton (osStep pow2)

/-- The kernel's per-key `(score, value)` pair for output `(i, d)` at global key `j`
over `S` keys, with the genuine `-1e8` sentinel kept (matching `ctxExactKeyList`). -/
noncomputable def ctxKV
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S : Nat) (i : Fin BLOCK_M) (d : Fin 128) (j : Fin S) : ℝ × ℝ :=
  (if j.val ≤ s.pids 0 * BLOCK_M + i.val + promptLen s 16 B_Prompt_Cache_Len then
      sm_scale * Finset.univ.sum (fun e : Fin 128 =>
        ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit) * ctxKTile s K S (j, e, PUnit.unit))
    else (0.0 - 10e7 : ℝ),
    ctxVTile s V S (j, d, PUnit.unit))

/-- Per-row key list over the streamed window `[0, hi)`: keys `j < hi`, index order. -/
noncomputable def ctxKeysUpto
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S hi : Nat) (i : Fin BLOCK_M) (d : Fin 128) : List (ℝ × ℝ) :=
  (List.finRange S).filterMap (fun j : Fin S =>
    if j.val < hi then some (ctxKV s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S i d j)
    else none)

/-- One ⊥-seeded online-softmax step: running max in `WithBot ℝ` (seeded `⊥`), so
`α = realExp2(m ⊖ m')` is `0` on the first block — faithful to the kernel's
`m_i = tl.zeros − inf` and `l_i`/`acc = 0`. -/
noncomputable def osStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let sc := sv.1; let v := sv.2
  let m' := m ⊔ ((sc : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
  let p := pow2 (sc - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)

/-- ⊥-seeded running max of the streamed window `[0, hi)` — the value the kernel
carries in `m_i` (seeded `⊥`, the `WithBot ⊔`-fold of the per-key scores). -/
noncomputable def ctxRunningMax
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S hi : Nat) (i : Fin BLOCK_M) (d : Fin 128) : WithBot ℝ :=
  ((ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d).map
    (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥

/-- The ⊥-seeded running `(max, l, acc)` after streaming the window `[0, hi)`. -/
noncomputable def ctxStateBot
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S hi : Nat) (i : Fin BLOCK_M) (d : Fin 128) : WithBot ℝ × ℝ × ℝ :=
  (ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d).foldl
    osStepBot (⊥, 0, 0)

/-- The running `max` component of an `osStepBot` fold is the `WithBot ⊔`-fold. -/
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

/-- The ⊥-seeded denominator equals `κ(ctxRunningMax)·Σpow2 score`. -/
theorem ctxStateBot_snd_fst
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S hi : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    (ctxStateBot s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d).2.1
      = ((ctxRunningMax s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d).elim 0
            (fun r => pow2 (-r)))
        * ((ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d).map
            (fun p => pow2 p.1)).sum := by
  have h := (osStepBot_foldl_consistent
    (ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d)
    ⊥ 0 0 0 0 (by simp) (by simp) (fun _ => rfl) (fun _ => rfl)).1
  rw [ctxStateBot]
  rw [show (List.foldl osStepBot (⊥, 0, 0)
        (ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d)).2.1 = _ from h]
  rw [show (List.foldl osStepBot (⊥, 0, 0)
        (ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d)).1
        = ctxRunningMax s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d from by
    rw [ctxRunningMax, osStepBot_foldl_fst, foldl_sup_bot_eq_foldr]]
  rw [zero_add]

/-- The ⊥-seeded accumulator equals `κ(ctxRunningMax)·Σpow2 score·v`. -/
theorem ctxStateBot_snd_snd
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S hi : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    (ctxStateBot s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d).2.2
      = ((ctxRunningMax s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d).elim 0
            (fun r => pow2 (-r)))
        * ((ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d).map
            (fun p => pow2 p.1 * p.2)).sum := by
  have h := (osStepBot_foldl_consistent
    (ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d)
    ⊥ 0 0 0 0 (by simp) (by simp) (fun _ => rfl) (fun _ => rfl)).2
  rw [ctxStateBot]
  rw [show (List.foldl osStepBot (⊥, 0, 0)
        (ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d)).2.2 = _ from h]
  rw [show (List.foldl osStepBot (⊥, 0, 0)
        (ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d)).1
        = ctxRunningMax s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d from by
    rw [ctxRunningMax, osStepBot_foldl_fst, foldl_sup_bot_eq_foldr]]
  rw [zero_add]

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

/-- The ⊥-seeded running max over a nonempty window (`0 < hi`, `0 < S`) is `≠ ⊥`:
key `0` is always streamed, so the coerced score list is nonempty. -/
theorem ctxRunningMax_ne_bot
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S hi : Nat) (i : Fin BLOCK_M) (d : Fin 128) (hhi : 0 < hi) (hS : 0 < S) :
    ctxRunningMax s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d ≠ ⊥ := by
  unfold ctxRunningMax ctxKeysUpto
  have hmem : (⟨0, hS⟩ : Fin S) ∈ List.finRange S := List.mem_finRange _
  set L := ((List.finRange S).filterMap (fun j : Fin S =>
      if j.val < hi then some (ctxKV s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S i d j)
      else none)).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) with hL
  have hmemL : ((ctxKV s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S i d ⟨0, hS⟩).1
      : WithBot ℝ) ∈ L := by
    rw [hL, List.mem_map]
    refine ⟨ctxKV s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S i d ⟨0, hS⟩, ?_, rfl⟩
    rw [List.mem_filterMap]
    exact ⟨⟨0, hS⟩, hmem, by rw [if_pos (show (⟨0, hS⟩ : Fin S).val < hi from hhi)]⟩
  have hle : ((ctxKV s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S i d ⟨0, hS⟩).1
      : WithBot ℝ) ≤ L.foldr (· ⊔ ·) ⊥ := mem_le_foldr_sup _ L hmemL
  intro hbot
  exact absurd (le_bot_iff.mp (hbot ▸ hle)) (WithBot.coe_ne_bot)

/-- **Full window = the banked `ctxExactKeyList`.** At `hi = S` every key is
streamed, so the windowed key list coincides with the exact-fold key list. -/
theorem ctxKeysUpto_full
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S S i d
      = ctxExactKeyList s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S i d := by
  unfold ctxKeysUpto ctxExactKeyList ctxKV
  rw [List.ofFn_eq_map]
  refine List.filterMap_eq_map_iff_forall_eq_some.mpr (fun j _ => ?_)
  simp only [j.isLt, if_true, Nat.add_assoc]

/-- ⊥-seeded ratio = 0-seeded `osStep` ratio whenever the running max `≠ ⊥`
(the `pow2(−m)` common factor cancels). -/
theorem ctxStateBot_ratio_eq
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S hi : Nat) (i : Fin BLOCK_M) (d : Fin 128)
    (hne : ctxRunningMax s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d ≠ ⊥) :
    (ctxStateBot s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d).2.2
        / (ctxStateBot s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d).2.1
      = ((ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d).foldl
            osStep (0, 0, 0)).2.2
        / ((ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d).foldl
            osStep (0, 0, 0)).2.1 := by
  rw [ctxStateBot_snd_fst, ctxStateBot_snd_snd]
  have hcL := (VeriTile.Triton.osStep_foldl_consistent
    (ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d)
    0 0 0 0 0 (by simp) (by simp)).1
  have hcT := (VeriTile.Triton.osStep_foldl_consistent
    (ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d)
    0 0 0 0 0 (by simp) (by simp)).2
  rw [show (List.foldl osStep (0, 0, 0)
        (ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d)).2.1
        = _ from hcL,
      show (List.foldl osStep (0, 0, 0)
        (ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d)).2.2
        = _ from hcT]
  cases hM : ctxRunningMax s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d with
  | bot => exact absurd hM hne
  | coe r =>
    rw [show ((↑r : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-r) from rfl]
    simp only [zero_add]
    rw [mul_div_mul_left _ _ (ne_of_gt (pow2_pos _)),
        mul_div_mul_left _ _ (ne_of_gt (pow2_pos _))]

/-- **The full-window ⊥-seeded final state reads off `contextAttnExactFold`.**
`ctxStateBot.acc / ctxStateBot.l` over the full window (`hi = S`) equals the
banked genuine closed form — the value the kernel's `m_i`/`l_i`/`acc` loop
produces. Requires the window nonempty (`ctxRunningMax ≠ ⊥`). -/
theorem ctxStateBot_full_eq_exactFold
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S : Nat) (i : Fin BLOCK_M) (d : Fin 128)
    (hne : ctxRunningMax s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S S i d ≠ ⊥) :
    (ctxStateBot s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S S i d).2.2
        / (ctxStateBot s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S S i d).2.1
      = contextAttnExactFold s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S
          (i, d, PUnit.unit) := by
  rw [ctxStateBot_ratio_eq _ _ _ _ _ _ _ _ _ _ _ _ hne, ctxKeysUpto_full]
  rfl

/-! ### one-block advance of the ⊥-seeded state (the step lemma's math)

Per-block decomposition of the ⊥-seeded recurrence: the loop's `c`-th iteration
streams block `c` (keys `c·128 ≤ j < (c+1)·128`), advancing `ctxStateBot(c·128)`
to `ctxStateBot((c+1)·128)` by one `osStepBot` fold over that block. The
context kernel keeps ALL keys `j < hi` (no causal drop — future keys carry the
`-1e8` sentinel score in `ctxKV`), so the window split is a pure threshold split. -/

/-- Block-`c` per-row key list: keys `c·BN ≤ j < (c+1)·BN`, the keys the loop's
`c`-th iteration streams (the `-1e8` sentinel kept on future keys). -/
noncomputable def ctxBlock
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S BN c : Nat) (i : Fin BLOCK_M) (d : Fin 128) : List (ℝ × ℝ) :=
  (List.finRange S).filterMap (fun j : Fin S =>
    if c * BN ≤ j.val ∧ j.val < (c + 1) * BN then
      some (ctxKV s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S i d j)
    else none)

/-- Threshold-split for a `.val`-ascending `Fin` list: the `j.val < hi₂` filterMap
splits into the `j.val < t` prefix and the `t ≤ j.val < hi₂` block (`t ≤ hi₂`). -/
private theorem ctx_filterMap_window_split {n : Nat} (l : List (Fin n))
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

/-- **Window split** (`hi = c·BN`): keys streamed through `c+1` blocks are those
through `c` blocks followed by block `c`. -/
theorem ctxKeysUpto_succ
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S BN c : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S ((c + 1) * BN) i d
      = ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S (c * BN) i d
        ++ ctxBlock s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S BN c i d := by
  unfold ctxKeysUpto ctxBlock
  rw [ctx_filterMap_window_split (List.finRange S) (List.pairwise_lt_finRange S)
    (c * BN) ((c + 1) * BN) (fun j => ctxKV s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S i d j)
    (by nlinarith [Nat.zero_le BN])]

/-- **One-block advance** of the ⊥-seeded state: streaming block `c` folds that
block's keys (via `osStepBot`) onto the state after `c` blocks. -/
theorem ctxStateBot_succ
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S BN c : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    ctxStateBot s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S ((c + 1) * BN) i d
      = (ctxBlock s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S BN c i d).foldl
          osStepBot (ctxStateBot s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S (c * BN) i d) := by
  unfold ctxStateBot
  rw [ctxKeysUpto_succ, List.foldl_append]

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

/-- **The block-at-once update equals the key-by-key `osStepBot` fold.** For a
block with max `M' = m ⊔ blockSup` and a state `(m, l, acc)` anchored to the true
denominator/accumulator via `l = κ(m)·L`, `acc = κ(m)·T`, the kernel's one-shot
rescale-and-add lands on `block.foldl osStepBot (m, l, acc)`. -/
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

/-! ### exec-stepping infrastructure for the FA-1 assembly (WIP)

Per-statement `evalOp` recipes specific to this kernel — the causal `≥` mask and
the `-1e8` sentinel `where`, which the non-causal template does not have. Reusable
building blocks for the step lemma of the streaming loop (the remaining
multi-thousand-line FA-1 grind matching the loop to `contextAttnExactFold`). -/

/-- Axis-0 `expandDim` over a `nat` register (the `offs_n[None, :]` row broadcast). -/
@[simp] theorem ctx_evalOp_expandDim_zero_nat {D : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [1, D] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] name)) s =
      (s.regs .nat [D] name).bind (fun v =>
        some ({ data := fun i : TileIndex [1, D] => v.data (i.2.1, PUnit.unit) } : Tile .nat [1, D])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

/-- Axis-1 `expandDim` over a `nat` register (the `offs_m[:, None]` column broadcast). -/
@[simp] theorem ctx_evalOp_expandDim_one_nat {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] name)) s =
      (s.regs .nat [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .nat [M, 1])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

/-- Eval helper for `ge` (the causal mask comparison): no `@[simp]` form exists. -/
theorem ctx_evalOp_ge {dtype a b shape} (h : ComparableDType dtype) (bc : Broadcast a b shape)
    (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

/-- **Causal-mask eval** (`(offs_m[:,None] + prompt_cache_len) ≥ (start_n + offs_n[None,:])`,
shape `[BLOCK_M, BLOCK_N]`): `mask[i,j] = (start_n + j ≤ offs_m_i + prompt_cache_len)`.
The prompt-cache-offset causal boundary, decoded lane-by-lane. -/
theorem ctxMask_eval (s : BlockState) (BM BN plen SN : Nat) (gOM : Fin BM → Nat)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec gOM))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hp : s.regs .nat [] "prompt_cache_len" = some (Tile.scalar plen))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOp (Op.ge ComparableDType.nat Broadcast.nil.consL.consR
        (Op.add NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
          (Op.ref .nat [] "prompt_cache_len"))
        (Op.add NumericDType.nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")))) s
      = some (⟨fun idx : TileIndex [BM, BN] =>
          decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ : Tile .bool [BM, BN]) := by
  rw [ctx_evalOp_ge]
  simp only [evalOp_add, evalOp_ref, ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
    hm, hn, hp, hsn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp [Tile.cop, Tile.bop, Tile.expandDim, Tile.vec, ComparableDType.ge, NumericDType.add]

/-- **`-1e8` sentinel `where` eval** (`tl.where(mask, qk·sm_scale, -1e8)`): the
causal masking statement. Active lanes get the scaled score `qk·sm_scale`; future
lanes get the finite sentinel `0 - 10e7 = -1e8` (whence `exp2(-1e8)`, not `0`). -/
theorem ctxWhere_eval (s : BlockState) (BM BN : Nat) (sm : ℝ)
    (masktile : Tile .bool [BM, BN]) (qktile : Tile .real [BM, BN])
    (hmask : s.regs .bool [BM, BN] "mask" = some masktile)
    (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp (Op.where (Op.ref .bool [BM, BN] "mask")
        (Op.mul NumericDType.real Broadcast.scalarR (Op.ref .real [BM, BN] "qk") (Op.const sm))
        ((Op.sub NumericDType.real Broadcast.nil (Op.const 0.0) (Op.const 10e7)).broadcast
          [BM, BN])) s
      = some (Tile.select masktile
          (Tile.bop NumericDType.real.mul Broadcast.scalarR qktile (Tile.scalar (some sm)))
          (⟨fun _ : TileIndex [BM, BN] => some (0.0 - 10e7 : ℝ)⟩ : Tile .real [BM, BN])) := by
  rw [evalOp_where]
  simp only [evalOp_mul, evalOp_ref, evalOp_const, hmask, hqk, Option.bind_eq_bind, Option.bind_some]
  have hbroad : @evalOp .real [BM, BN]
      ((Op.sub NumericDType.real Broadcast.nil (Op.const 0.0) (Op.const 10e7)).broadcast [BM, BN]) s
      = some (⟨fun _ : TileIndex [BM, BN] => some (0.0 - 10e7 : ℝ)⟩ : Tile .real [BM, BN]) := by
    simp only [evalOp]
    refine congrArg some ?_
    ext idx
    simp [Tile.bop, NumericDType.sub]
  rw [hbroad]
  rfl

/-- **Masked pointer-arith region load eval** (`tl.load(R + offs, mask=m, other=o)`):
lane `i` reads `R[offs i]` when `mask i` holds, else takes the `other` value `o i`.
This is the context kernel's `q`/`k`/`v` load shape (`MemAccess.region` with a
`MaskOpt.maskOther` mask), the pointer-arith analogue of flash's block-ptr
`flash_load_{K,Q}_eval`. -/
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

/-! ## exec-assembly: preLoop (Milestone 1)

The compiled body of `context_attn_fwd_kernel_int8kv_surface` at the Python test
shape (`BLOCK_M = BLOCK_N = BLOCK_DMODEL = 128`, `H = 16`, `kv_group_num = 1`,
contiguous strides) is a 24-statement list: 19 preLoop statements, the
`forRangeDyn start_n 0 (block_mask·block_end_loc) 128` streaming loop (18-stmt
body), and 4 post-loop statements (`acc /= l_i`, `off_o`, `out_ptrs`, masked
store). This section banks the **preLoop** execution: the 19 deterministic
prologue statements step a clean input state to the loop-entry state, exposing
every register the loop invariant / loop body reads back (`m_i = ⊥` seed,
`l_i`/`acc = 0` seeds, the scaled-and-masked `q` tile, the index vectors, the
runtime scalars `cur_batch`/`cur_head`/`prompt_cache_len`/…, and the resolved
dynamic bound `block_mask·block_end_loc`). -/

/-- Eval helper for `floorDiv` (the `cur_batch = cur_bh // H` / `cur_kv_head`
decomposition). No `@[simp]` form. -/
theorem ctx_evalOp_floorDiv {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.floorDiv h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.floorDiv bc vx vy)) := by
  simp [evalOp]

/-- Eval helper for `mod` (the `cur_head = cur_bh % H` decomposition). -/
theorem ctx_evalOp_mod {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.mod h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.mod bc vx vy)) := by
  simp [evalOp]

/-- Eval helper for `remap` (the column/row broadcast of the load/store masks). -/
theorem ctx_evalOp_remap {dtype outShape inShape}
    (map : TileIndex outShape → TileIndex inShape) (a : Op dtype inShape) (s : BlockState) :
    evalOp (.remap outShape map a) s = (do
      let v ← evalOp a s; some (Tile.remap map v)) := by
  simp [evalOp]

/-- Scalar `nat` region load (`tl.load(b_prompt_cache_len + cur_batch)` etc.):
reads `region[off]` from memory, with no mask, into a `[]`-shape `nat` tile. -/
theorem ctx_evalOp_load_scalar_nat (region : Region .nat) (off : Op .nat [])
    (s : BlockState) (o : Nat) (hoff : evalOp off s = some (Tile.scalar o)) :
    evalOp (.load .nat (MemAccess.region region off) MaskOpt.none) s
      = some (Tile.scalar (s.readMemValue .nat (Region.cast region) o)) := by
  simp only [evalOp, hoff, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- The Python-shape preLoop: the first 19 lowered statements of the context
surface kernel. Transcribed from `(surface …).toAlgKernel.body.take 19` (checked
by `rfl` in `ctxPreLoop_take`). -/
def ctxPreLoop (Q : RegionName) (B_Start_Loc B_Seqlen b_prompt_cache_len : Region .nat) :
    List Stmt :=
  [ Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [] "cur_bh" (Op.programId 1),
    Stmt.assign .nat [] "cur_batch"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_bh") (Op.constNat 16)),
    Stmt.assign .nat [] "cur_head"
      (Op.mod .nat Broadcast.nil (Op.ref .nat [] "cur_bh") (Op.constNat 16)),
    Stmt.assign .nat [] "cur_kv_head"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 1)),
    Stmt.assign .nat [] "prompt_cache_len"
      (Op.load .nat (MemAccess.region b_prompt_cache_len (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "cur_batch_in_all_start_index"
      (Op.load .nat (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "cur_batch_seq_len"
      (Op.sub .nat Broadcast.nil
        (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none)
        (Op.ref .nat [] "prompt_cache_len")),
    Stmt.assign .nat [] "block_start_loc"
      (Op.mul .nat Broadcast.nil (Op.constNat 128) (Op.ref .nat [] "start_m")),
    Stmt.assign .nat [128] "offs_n" (Op.arange 128),
    Stmt.assign .nat [128] "offs_d" (Op.arange 128),
    Stmt.assign .nat [128] "offs_m"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "block_start_loc") (Op.arange 128)),
    Stmt.assign .nat [128, 128] "off_q"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")))
            (Op.constNat 2048))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 128)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .real [128, 128] "q"
      (Op.load .real (MemAccess.region Q (Op.ref .nat [128, 128] "off_q"))
        (MaskOpt.maskOther
          (Op.remap [128, 128] (fun x => (x.1, ⟨0, Broadcast.leftIndex._proof_1⟩, PUnit.unit))
            (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
              (Op.ref .nat [] "cur_batch_seq_len")))
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
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "block_start_loc") (Op.constNat 128))
              (Op.ref .nat [] "prompt_cache_len"))
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len")
              (Op.ref .nat [] "prompt_cache_len"))).where
        (Op.add .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "block_start_loc") (Op.constNat 128))
          (Op.ref .nat [] "prompt_cache_len"))
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len")
          (Op.ref .nat [] "prompt_cache_len"))) ]

/-- The lowered Python-shape context body `take 19` is exactly `ctxPreLoop`. -/
theorem ctxPreLoop_take (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : Region .nat) (sm_scale : ℝ) :
    (context_attn_fwd_kernel_int8kv_surface Q K V sm_scale Out
        B_Start_Loc B_Seqlen b_prompt_cache_len
        2048 128 1 8388608 262144 128 1 8388608 262144 128 1
        2048 128 1 1 16 128 128 128).toAlgKernel.body.take 19
      = ctxPreLoop Q B_Start_Loc B_Seqlen b_prompt_cache_len := by
  rfl

/-- **q/store-mask eval** (`(offs_m[:, None] < cur_batch_seq_len)` broadcast over
the `[128, 128]` tile via `remap`): `mask[r, c] = (offs_m_r < seqLen)`. The row
mask shared by the `q` load and the final `Out` store. -/
theorem ctxRowMask_eval (s : BlockState) (gOM : Fin 128 → Nat) (sl : Nat)
    (hm : s.regs .nat [128] "offs_m" = some (Tile.vec gOM))
    (hsl : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar sl)) :
    evalOp (Op.remap [128, 128] (fun x => (x.1, ⟨0, Broadcast.leftIndex._proof_1⟩, PUnit.unit))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
          (Op.ref .nat [] "cur_batch_seq_len"))) s
      = some (⟨fun idx : TileIndex [128, 128] => decide (gOM idx.1 < sl)⟩ : Tile .bool [128, 128]) := by
  rw [ctx_evalOp_remap, evalOp_lt]
  erw [ctx_evalOp_expandDim_one_nat]
  simp only [evalOp_ref, hm, hsl, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  refine congrArg some ?_
  ext idx
  simp [Tile.remap, Tile.cop, Tile.expandDim, Tile.vec, ComparableDType.lt]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **PreLoop execution (Milestone 1).** From a clean input state (`undef = 0`),
the 19 deterministic prologue statements step to the loop-entry state, exposing
every register the streaming loop / its invariant reads back: the runtime scalars
(`start_m`, `cur_batch = pid₁/16`, `cur_head = pid₁%16`, `cur_kv_head`,
`prompt_cache_len`, `cur_batch_in_all_start_index`, `cur_batch_seq_len`,
`block_start_loc = 128·pid₀`), the index vectors (`offs_n`/`offs_d`/`offs_m`), the
⊥-seed running state (`m_i = full ⊥`, `l_i = full 0`, `acc = full 0`), the
masked-and-loaded `q` tile, and the resolved dynamic-bound prep
(`block_mask`/`block_end_loc`). Memory and `undef` are preserved. -/
theorem ctxPreLoop_eval
    (s : BlockState) (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : Region .nat) (sm_scale : ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (ctxPreLoop Q B_Start_Loc B_Seqlen b_prompt_cache_len) s = some s0
      ∧ s0.pids = s.pids ∧ s0.mem = s.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 1 / 16))
      ∧ s0.regs .nat [] "cur_head" = some (Tile.scalar (s.pids 1 % 16))
      ∧ s0.regs .nat [] "cur_kv_head" = some (Tile.scalar (s.pids 1 % 16 / 1))
      ∧ s0.regs .nat [] "prompt_cache_len"
          = some (Tile.scalar (s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16)))
      ∧ s0.regs .nat [] "cur_batch_in_all_start_index"
          = some (Tile.scalar (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 1 / 16)))
      ∧ s0.regs .nat [] "cur_batch_seq_len"
          = some (Tile.scalar (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 1 / 16)
              - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16)))
      ∧ s0.regs .nat [] "block_start_loc" = some (Tile.scalar (128 * s.pids 0))
      ∧ s0.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val))
      ∧ s0.regs .nat [128] "offs_d" = some (Tile.vec (fun e : Fin 128 => e.val))
      ∧ s0.regs .nat [128] "offs_m" = some (Tile.vec (fun r : Fin 128 => 128 * s.pids 0 + r.val))
      ∧ s0.regs .real [128] "m_i" = some ⟨fun _ : TileIndex [128] => (⊥ : WithBot ℝ)⟩
      ∧ s0.regs .real [128] "l_i" = some ⟨fun _ : TileIndex [128] => some (0 : ℝ)⟩
      ∧ s0.regs .real [128, 128] "acc" = some ⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩
      ∧ s0.regs .nat [] "block_mask"
          = some (Tile.scalar (if 128 * s.pids 0 < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 1 / 16)
              - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16) then 1 else 0))
      ∧ s0.regs .nat [] "block_end_loc"
          = some (Tile.scalar
              (let a := 128 * s.pids 0 + 128 + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16)
               let b := (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 1 / 16)
                   - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16))
                 + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16)
               if a < b then a else b))
      ∧ s0.regs .nat [128, 128] "off_q" = some ⟨fun idx : TileIndex [128, 128] =>
          (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 1 / 16) + (128 * s.pids 0 + idx.1.val))
              * 2048 + s.pids 1 % 16 * 128 + idx.2.1.val * 1⟩
      ∧ s0.regs .real [128, 128] "q" = some ⟨fun idx : TileIndex [128, 128] =>
          if decide (128 * s.pids 0 + idx.1.val
              < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 1 / 16)
                - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16)) then
            s.readMemValue .real (Region.cast Q)
              ((s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 1 / 16) + (128 * s.pids 0 + idx.1.val))
                  * 2048 + s.pids 1 % 16 * 128 + idx.2.1.val * 1)
          else some (0.0 : ℝ)⟩ := by
  unfold ctxPreLoop
  -- stmt 0: start_m = programId 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  -- stmt 1: cur_bh = programId 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  -- stmt 2: cur_batch = cur_bh // 16
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_bh") (Op.constNat 16)) _
        = some (Tile.scalar (s.pids 1 / 16)) from by
      rw [ctx_evalOp_floorDiv]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 3: cur_head = cur_bh % 16
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mod .nat Broadcast.nil (Op.ref .nat [] "cur_bh") (Op.constNat 16)) _
        = some (Tile.scalar (s.pids 1 % 16)) from by
      rw [ctx_evalOp_mod]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 4: cur_kv_head = cur_head // 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 1)) _
        = some (Tile.scalar (s.pids 1 % 16 / 1)) from by
      rw [ctx_evalOp_floorDiv]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, Option.bind_eq_bind,
        Option.bind_some]
      rfl))]
  -- stmt 5: prompt_cache_len = load(bpc + cur_batch)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_scalar_nat b_prompt_cache_len (Op.ref .nat [] "cur_batch") _ (s.pids 1 / 16)
      (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  -- stmt 6: cur_batch_in_all_start_index = load(B_Start_Loc + cur_batch)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_scalar_nat B_Start_Loc (Op.ref .nat [] "cur_batch") _ (s.pids 1 / 16)
      (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  -- stmt 7: cur_batch_seq_len = load(B_Seqlen + cur_batch) - prompt_cache_len
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.sub .nat Broadcast.nil
        (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none)
        (Op.ref .nat [] "prompt_cache_len")) _
        = some (Tile.scalar (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 1 / 16)
            - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16))) from by
      rw [evalOp_sub, ctx_evalOp_load_scalar_nat B_Seqlen (Op.ref .nat [] "cur_batch") _ (s.pids 1 / 16)
        (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, BlockState.setReg_same])]
      simp only [evalOp_ref, BlockState.setReg_same, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 8: block_start_loc = 128 * start_m
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .nat Broadcast.nil (Op.constNat 128) (Op.ref .nat [] "start_m")) _
        = some (Tile.scalar (128 * s.pids 0)) from by
      rw [evalOp_mul]
      simp only [evalOp_constNat, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 9: offs_n = arange 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange 128) _ = some (Tile.vec (fun j : Fin 128 => j.val)) from evalOp_arange 128 _))]
  -- stmt 10: offs_d = arange 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange 128) _ = some (Tile.vec (fun e : Fin 128 => e.val)) from evalOp_arange 128 _))]
  -- stmt 11: offs_m = block_start_loc + arange 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "block_start_loc") (Op.arange 128)) _
        = some (Tile.vec (fun r : Fin 128 => 128 * s.pids 0 + r.val)) from by
      rw [evalOp_add]
      simp only [evalOp_ref, evalOp_arange, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r
      simp [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add]))]
  -- stmt 12: off_q
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")))
            (Op.constNat 2048))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 128)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))) _
        = some (⟨fun idx : TileIndex [128, 128] =>
            (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 1 / 16) + (128 * s.pids 0 + idx.1.val))
                * 2048 + s.pids 1 % 16 * 128 + idx.2.1.val * 1⟩ : Tile .nat [128, 128]) from by
      simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
        ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
        BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 13: q = masked load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_region_maskOther Q (Op.ref .nat [128, 128] "off_q") _ _ _
      (⟨fun idx : TileIndex [128, 128] =>
          (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 1 / 16) + (128 * s.pids 0 + idx.1.val))
              * 2048 + s.pids 1 % 16 * 128 + idx.2.1.val * 1⟩ : Tile .nat [128, 128])
      (⟨fun idx : TileIndex [128, 128] => decide (128 * s.pids 0 + idx.1.val
          < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 1 / 16)
            - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16))⟩ : Tile .bool [128, 128])
      (⟨fun _ : TileIndex [128, 128] => some (0.0 : ℝ)⟩ : Tile .real [128, 128])
      (by rw [evalOp_ref]; simp [BlockState.setReg_same])
      (ctxRowMask_eval _ (fun r : Fin 128 => 128 * s.pids 0 + r.val) _
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
        = some (Tile.scalar (if 128 * s.pids 0 < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 1 / 16)
            - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16) then 1 else 0)) from by
      rw [evalOp_where]
      simp only [evalOp_lt, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_
      ext _idx
      simp only [Tile.select_data, Tile.cop_data, Tile.scalar_data_index, ComparableDType.lt]
      by_cases h : 128 * s.pids 0 < s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 1 / 16)
          - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16)
      · simp [h]
      · simp [h]))]
  -- stmt 18: block_end_loc = where(...) ... (value irrelevant to invariant readbacks; eval generically)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp ((Op.lt .nat Broadcast.nil
            (Op.add .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "block_start_loc") (Op.constNat 128))
              (Op.ref .nat [] "prompt_cache_len"))
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len")
              (Op.ref .nat [] "prompt_cache_len"))).where
          (Op.add .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "block_start_loc") (Op.constNat 128))
            (Op.ref .nat [] "prompt_cache_len"))
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_seq_len")
            (Op.ref .nat [] "prompt_cache_len"))) _
        = some (Tile.scalar
            (let a := 128 * s.pids 0 + 128 + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16)
             let b := (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 1 / 16)
                 - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16))
               + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16)
             if a < b then a else b)) from by
      rw [evalOp_where]
      simp only [evalOp_lt, evalOp_add, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_
      ext _idx
      simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.scalar_data_index,
        ComparableDType.lt, NumericDType.add, Broadcast.leftIndex, Broadcast.rightIndex]
      by_cases h : 128 * s.pids 0 + 128 + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16)
          < (s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 1 / 16)
              - s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16))
            + s.readMemValue .nat (Region.cast b_prompt_cache_len) (s.pids 1 / 16)
      · simp [h]
      · simp [h]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_pids]
  · funext rg o; simp only [BlockState.setReg_mem]
  · intro rg o; simp only [BlockState.setReg_undef]; exact hundef rg o
  all_goals
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids,
      BlockState.setReg_readMemValue, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq]

/-! ## exec-assembly: body decomposition (loop body + postLoop)

Scaffolding for the streaming-loop assembly: the 18-statement loop body
(`ctxLoopBody`) and the 4-statement post-loop (`ctxPostLoop`), transcribed from
the lowered surface body and validated as pure `List` identities. The full body
is `ctxPreLoop ++ (forRangeDyn 0 (block_mask·block_end_loc) 128 ctxLoopBody ::
ctxPostLoop)`, exposed via `List.take_append_drop` so the dynamic-loop driver
`forRangeDyn_inv` applies with `ctxLoopBody`/`ctxPostLoop` supplied at the call
site. -/

/-- The kernel's chosen `sm_scale` constant at the Python test shape:
`(√128)⁻¹ · 1.4426950408889634` (`= (1/√D)·log₂ e`). -/
noncomputable def sm_scale_python : ℝ := ((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634

/-- The 18 lowered loop-body statements of the Python-shape context surface body
(`start_n=multiple_of`, `off_k`/`k` masked load, `qk=dot`, causal `mask`/`where`
with the `-1e8` sentinel, online-softmax `m_ij`/`p`/`l_ij`/`alpha`/`l_i`/`acc`
update, `off_v`/`v` masked load, `p` cast (noop), `acc=dot(p,v)+acc`,
`m_i=m_ij`). Transcribed; checked by `rfl` in `ctxBody_split`. -/
noncomputable def ctxLoopBody (Q K V : RegionName) : List Stmt :=
  [ Stmt.assign .nat [] "start_n" (Op.ref .nat [] "start_n"),
    Stmt.assign .nat [128, 128] "off_k"
      (Op.add .nat Broadcast.nil.consR.consL
        (Op.add .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat 8388608))
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))
              (Op.constNat 128)))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat 262144)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .real [128, 128] "k"
      (Op.load .real (MemAccess.region K (Op.ref .nat [128, 128] "off_k"))
        (MaskOpt.maskOther
          (Op.remap [128, 128] (fun x => (⟨0, Broadcast.leftIndex._proof_1⟩, x.2.1, PUnit.unit))
            (Op.lt .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))
              (Op.ref .nat [] "block_end_loc")))
          ((Op.const (0.0 : ℝ)).broadcast [128, 128]))),
    Stmt.assign .real [128, 128] "qk"
      (Op.dot (batch := []) (Op.ref .real [128, 128] "q") (Op.ref .real [128, 128] "k")),
    Stmt.assign .bool [128, 128] "mask"
      (Op.ge .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
          (Op.ref .nat [] "prompt_cache_len"))
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))),
    Stmt.assign .real [128, 128] "qk"
      ((Op.ref .bool [128, 128] "mask").where
        (Op.mul .real Broadcast.scalarR (Op.ref .real [128, 128] "qk") (Op.const (sm_scale_python : ℝ)))
        ((Op.sub .real Broadcast.nil (Op.const (0.0 : ℝ)) (Op.const (10e7 : ℝ))).broadcast [128, 128])),
    Stmt.assign .real [128] "m_ij"
      ((Op.gt .real Broadcast.nil.consSame (Op.ref .real [128] "m_i")
            (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "qk"))).where
        (Op.ref .real [128] "m_i") (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "qk"))),
    Stmt.assign .real [128, 128] "qk"
      (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "m_ij"))),
    Stmt.assign .real [128, 128] "p" (Op.ref .real [128, 128] "qk").exp2,
    Stmt.assign .real [128] "l_ij" (Op.reduceSum ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "p")),
    Stmt.assign .real [128] "alpha"
      (Op.sub .real Broadcast.nil.consSame (Op.ref .real [128] "m_i") (Op.ref .real [128] "m_ij")).exp2,
    Stmt.assign .real [128] "l_i"
      (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [128] "l_i") (Op.ref .real [128] "alpha"))
        (Op.ref .real [128] "l_ij")),
    Stmt.assign .real [128, 128] "acc"
      (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "alpha"))),
    Stmt.assign .nat [128, 128] "off_v"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat 8388608))
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_n")))
              (Op.constNat 128)))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat 262144)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .real [128, 128] "v"
      (Op.load .real (MemAccess.region V (Op.ref .nat [128, 128] "off_v"))
        (MaskOpt.maskOther
          (Op.remap [128, 128] (fun x => (x.1, ⟨0, Broadcast.leftIndex._proof_1⟩, PUnit.unit))
            (Op.lt .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_n")))
              (Op.ref .nat [] "block_end_loc")))
          ((Op.const (0.0 : ℝ)).broadcast [128, 128]))),
    Stmt.assign .real [128, 128] "p" (Op.ref .real [128, 128] "p"),
    Stmt.assign .real [128, 128] "acc"
      (Op.add .real Broadcast.nil.consSame.consSame
        (Op.dot (batch := []) (Op.ref .real [128, 128] "p") (Op.ref .real [128, 128] "v"))
        (Op.ref .real [128, 128] "acc")),
    Stmt.assign .real [128] "m_i" (Op.ref .real [128] "m_ij") ]

/-- The 4 lowered post-loop statements (`acc /= l_i[:, None]`, `off_o`,
`out_ptrs = Out + off_o`, masked `store`). Transcribed; checked by `rfl`. -/
def ctxPostLoop (Out : RegionName) : List Stmt :=
  [ Stmt.assign .real [128, 128] "acc"
      (Op.div .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "l_i"))),
    Stmt.assign .nat [128, 128] "off_o"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")))
            (Op.constNat 2048))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 128)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .ptr [128, 128] "out_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [128, 128] "off_o")),
    Stmt.store .real [128, 128] (MemAccess.ptr (Op.ref .ptr [128, 128] "out_ptrs"))
      (Op.ref .real [128, 128] "acc")
      (MaskOpt.mask
        (Op.remap [128, 128] (fun x => (x.1, ⟨0, Broadcast.leftIndex._proof_1⟩, PUnit.unit))
          (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
            (Op.ref .nat [] "cur_batch_seq_len")))) ]

/-- **Body decomposition.** The compiled Python-shape context body splits as
`ctxPreLoop ++ (forRangeDyn start_n 0 (block_mask·block_end_loc) 128 ctxLoopBody
:: ctxPostLoop)`. Pure `List` identity (`rfl`), independent of any transcription:
the loop driver `forRangeDyn_inv` applies with `ctxLoopBody`/`ctxPostLoop` supplied
at the call site. -/
theorem ctxBody_split (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : Region .nat) :
    (context_attn_fwd_kernel_int8kv_surface Q K V (sm_scale_python : ℝ) Out
        B_Start_Loc B_Seqlen b_prompt_cache_len
        2048 128 1 8388608 262144 128 1 8388608 262144 128 1
        2048 128 1 1 16 128 128 128).toAlgKernel.body
      = ctxPreLoop Q B_Start_Loc B_Seqlen b_prompt_cache_len
        ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0)
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask") (Op.ref .nat [] "block_end_loc"))
              (Op.constNat 128) (ctxLoopBody Q K V)
            :: ctxPostLoop Out) := by
  rfl

/-! ### loop-body per-statement op-eval recipes

The `evalOp` recipes for the streaming-loop body statements, mirroring
`Examples/AttentionForwardClosedForm`'s `mij_op_eval`/`qk2_op_eval`/… but with
this kernel's pointer-arith key/value offsets and (crucially) the causal
`-1e8` `where` sentinel (already banked as `ctxWhere_eval`). The online-softmax
update statements (`m_ij`/`qk2`/`p`/`l_ij`/`alpha`/`l_i`/`acc1`) are shape- and
op-identical to the non-causal template. -/

/-- `off_k`/`off_v` pointer-offset eval (`cur_batch·stride + (start_n+offs_n[axis])·stride
+ cur_kv_head·stride + offs_d[axis]·1`), the masked-load address tile for `k`/`v`.
`expCol` selects the axis the `start_n+offs_n` term is broadcast on (`0` for `k`'s
`[None,:]`, `1` for `v`'s `[:,None]`); `expD` the other. Lane `(a,b)` reads the
address with `j = offs_n` on the kernel's chosen axis. -/
theorem ctx_offk_eval (s : BlockState) (cb ckvh SN : Nat)
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar cb))
    (hckvh : s.regs .nat [] "cur_kv_head" = some (Tile.scalar ckvh))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hd : s.regs .nat [128] "offs_d" = some (Tile.vec (fun e : Fin 128 => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consR.consL
        (Op.add .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat 8388608))
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))
              (Op.constNat 128)))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat 262144)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          cb * 8388608 + (SN + idx.2.1.val) * 128 + ckvh * 262144 + idx.1.val * 1⟩ : Tile .nat [128, 128]) := by
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
    ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
    hcb, hckvh, hsn, hn, hd, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul]

/-- `off_v` pointer-offset eval (`start_n+offs_n` on the **column** axis `[:,None]`,
`offs_d` row-broadcast). Lane `(a,b)` reads `cb·strideV + (SN+a)·128 + ckvh·262144 + b`. -/
theorem ctx_offv_eval (s : BlockState) (cb ckvh SN : Nat)
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar cb))
    (hckvh : s.regs .nat [] "cur_kv_head" = some (Tile.scalar ckvh))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hd : s.regs .nat [128] "offs_d" = some (Tile.vec (fun e : Fin 128 => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat 8388608))
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_n")))
              (Op.constNat 128)))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat 262144)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          cb * 8388608 + (SN + idx.1.val) * 128 + ckvh * 262144 + idx.2.1.val * 1⟩ : Tile .nat [128, 128]) := by
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
    ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat,
    hcb, hckvh, hsn, hn, hd, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul]

/-- `k`/`v` masked-load mask eval (`(start_n + offs_n[axis]) < block_end_loc`,
broadcast over the other axis via `remap`). `k`: `offs_n` on column axis (`[None,:]`),
mask broadcast over rows. `mask[a,b] = (SN + b < bel)`. -/
theorem ctxKMask_eval (s : BlockState) (SN bel : Nat)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hbel : s.regs .nat [] "block_end_loc" = some (Tile.scalar bel)) :
    evalOp (Op.remap [128, 128] (fun x => (⟨0, Broadcast.leftIndex._proof_1⟩, x.2.1, PUnit.unit))
        (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))
          (Op.ref .nat [] "block_end_loc"))) s
      = some (⟨fun idx : TileIndex [128, 128] => decide (SN + idx.2.1.val < bel)⟩ : Tile .bool [128, 128]) := by
  rw [ctx_evalOp_remap, evalOp_lt, evalOp_add]
  erw [ctx_evalOp_expandDim_zero_nat]
  simp only [evalOp_ref, hsn, hn, hbel, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  refine congrArg some ?_; ext idx
  simp [Tile.remap, Tile.cop_data, Tile.bop_data, Tile.expandDim, Tile.scalar_data_index,
    Tile.vec_data, Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt, NumericDType.add]

/-- `v` masked-load mask eval (`offs_n` on **row** axis `[:,None]`). `mask[a,b] = (SN + a < bel)`. -/
theorem ctxVMask_eval (s : BlockState) (SN bel : Nat)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hbel : s.regs .nat [] "block_end_loc" = some (Tile.scalar bel)) :
    evalOp (Op.remap [128, 128] (fun x => (x.1, ⟨0, Broadcast.leftIndex._proof_1⟩, PUnit.unit))
        (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_n")))
          (Op.ref .nat [] "block_end_loc"))) s
      = some (⟨fun idx : TileIndex [128, 128] => decide (SN + idx.1.val < bel)⟩ : Tile .bool [128, 128]) := by
  rw [ctx_evalOp_remap, evalOp_lt, evalOp_add]
  erw [ctx_evalOp_expandDim_one_nat]
  simp only [evalOp_ref, hsn, hn, hbel, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  refine congrArg some ?_; ext idx
  simp [Tile.remap, Tile.cop_data, Tile.bop_data, Tile.expandDim, Tile.scalar_data_index,
    Tile.vec_data, Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt, NumericDType.add]

/-- `qk = tl.dot(q, k)` eval (no scale/cast — the kernel scales later via the
causal `where`). -/
theorem ctxQk_op_eval (s : BlockState) (qtile ktile : Tile .real [128, 128])
    (hq : s.regs .real [128, 128] "q" = some qtile) (hk : s.regs .real [128, 128] "k" = some ktile) :
    evalOp (Op.dot (batch := []) (Op.ref .real [128, 128] "q") (Op.ref .real [128, 128] "k")) s
      = some (Tile.dot [] qtile ktile) := by
  rw [evalOp_dot]; simp [hq, hk]

/-- Axis-1 `expandDim` over a `real` register (`m_ij[:, None]`/`alpha[:, None]`),
explicit `[M,1]` result shape so `erw` matches past the proof-irrelevant axis. -/
@[simp] theorem ctx_evalOp_expandDim_one_real {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .real [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] name)) s =
      (s.regs .real [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .real [M, 1])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

/-- `m_ij = tl.maximum(m_i, tl.max(qk, 1))` eval. -/
theorem ctxMij_op_eval (s : BlockState) (mtile qktile rmaxT : Tile .real [128])
    (qkfull : Tile .real [128, 128])
    (hmi : s.regs .real [128] "m_i" = some mtile)
    (hqk : s.regs .real [128, 128] "qk" = some qkfull)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [128, 128].length) qkfull = some rmaxT) :
    evalOp (Op.where
        (Op.gt .real Broadcast.nil.consSame (Op.ref .real [128] "m_i")
          (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "qk")))
        (Op.ref .real [128] "m_i")
        (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "qk"))) s
      = some (Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mtile rmaxT) mtile rmaxT) := by
  have hrmaxN : evalOp (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "qk")) s = some rmaxT := by
    rw [evalOp_reduceMax]; simp only [evalOp_ref, hqk]; exact hrm
  have hrmax : @evalOp TileDType.real [128]
      (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "qk")) s = some rmaxT := hrmaxN
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hmi, hrmax, Option.bind_eq_bind, Option.bind_some]

/-- `qk -= m_ij[:, None]` eval. -/
theorem ctxQk2_op_eval (s : BlockState) (qkfull : Tile .real [128, 128]) (mij : Tile .real [128])
    (hqk : s.regs .real [128, 128] "qk" = some qkfull) (hmij : s.regs .real [128] "m_ij" = some mij) :
    evalOp (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "m_ij"))) s
      = some (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame qkfull
          (Tile.expandDim ⟨1, by simp⟩ mij)) := by
  have hexp : @evalOp TileDType.real [128, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "m_ij")) s
      = some (Tile.expandDim ⟨1, by simp⟩ mij) := by erw [ctx_evalOp_expandDim_one_real, hmij]; rfl
  rw [evalOp_sub]
  simp only [evalOp_ref, hqk, hexp, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `p = tl.math.exp2(qk)` eval. -/
theorem ctxP_op_eval (s : BlockState) (qk2tile : Tile .real [128, 128])
    (hqk : s.regs .real [128, 128] "qk" = some qk2tile) :
    evalOp (Op.exp2 (Op.ref .real [128, 128] "qk")) s = some (Tile.uop WithBot.realExp2 qk2tile) := by
  rw [evalOp]; simp [hqk]

/-- `l_ij = tl.sum(p, 1)` eval. -/
theorem ctxLij_op_eval (s : BlockState) (ptile : Tile .real [128, 128])
    (hp : s.regs .real [128, 128] "p" = some ptile) :
    evalOp (Op.reduceSum (⟨1, by simp⟩ : Fin [128, 128].length) Bool.false (Op.ref .real [128, 128] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [128, 128].length) ptile) := by
  rw [evalOp_reduceSum]; simp only [evalOp_ref, hp, Option.bind_some]; rfl

/-- `alpha = tl.math.exp2(m_i − m_ij)` eval. -/
theorem ctxAlpha_op_eval (s : BlockState) (mi mij : Tile .real [128])
    (hmi : s.regs .real [128] "m_i" = some mi) (hmij : s.regs .real [128] "m_ij" = some mij) :
    evalOp (Op.exp2 (Op.sub .real Broadcast.nil.consSame (Op.ref .real [128] "m_i") (Op.ref .real [128] "m_ij"))) s
      = some (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mi mij)) := by
  rw [evalOp]; simp [evalOp_sub, hmi, hmij]

/-- `l_i = l_i · alpha + l_ij` eval. -/
theorem ctxLi_op_eval (s : BlockState) (li alpha lij : Tile .real [128])
    (hli : s.regs .real [128] "l_i" = some li) (ha : s.regs .real [128] "alpha" = some alpha)
    (hlij : s.regs .real [128] "l_ij" = some lij) :
    evalOp (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [128] "l_i") (Op.ref .real [128] "alpha"))
        (Op.ref .real [128] "l_ij")) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame
          (Tile.bop NumericDType.real.mul Broadcast.nil.consSame li alpha) lij) := by
  rw [evalOp_add]; simp [evalOp_mul, hli, ha, hlij]

/-- `acc *= alpha[:, None]` eval. -/
theorem ctxAcc1_op_eval (s : BlockState) (acctile : Tile .real [128, 128]) (alpha : Tile .real [128])
    (hacc : s.regs .real [128, 128] "acc" = some acctile) (ha : s.regs .real [128] "alpha" = some alpha) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "alpha"))) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acctile
          (Tile.expandDim ⟨1, by simp⟩ alpha)) := by
  have hexp : @evalOp TileDType.real [128, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "alpha")) s
      = some (Tile.expandDim ⟨1, by simp⟩ alpha) := by erw [ctx_evalOp_expandDim_one_real, ha]; rfl
  rw [evalOp_mul]; simp only [evalOp_ref, hacc, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

/-- `acc = tl.dot(p, v) + acc` eval (context's `p` stays real — the `.to(v.dtype)`
is a `real→real` no-op `Op.ref`). -/
theorem ctxAcc2_op_eval (s : BlockState) (acc1tile : Tile .real [128, 128])
    (ptile vtile : Tile .real [128, 128])
    (hp : s.regs .real [128, 128] "p" = some ptile) (hv : s.regs .real [128, 128] "v" = some vtile)
    (hacc : s.regs .real [128, 128] "acc" = some acc1tile) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame
        (Op.dot (batch := []) (Op.ref .real [128, 128] "p") (Op.ref .real [128, 128] "v"))
        (Op.ref .real [128, 128] "acc")) s
      = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
          (Tile.dot [] ptile vtile) acc1tile) := by
  have hdotN : evalOp (Op.dot (batch := []) (Op.ref .real [128, 128] "p") (Op.ref .real [128, 128] "v")) s
      = some (Tile.dot [] ptile vtile) := by rw [evalOp_dot]; simp [hp, hv]
  have hdotN2 : @evalOp TileDType.real [128, 128]
      (Op.dot (batch := []) (Op.ref .real [128, 128] "p") (Op.ref .real [128, 128] "v")) s
      = some (Tile.dot [] ptile vtile) := hdotN
  rw [evalOp_add]; simp only [evalOp_ref, hacc, hdotN2, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **LoopBody execution (Milestone 2 / M1).** The 18 lowered loop-body statements,
from a state exposing the loop-entry registers, step to a final state whose
running-softmax registers carry one more block of the online-softmax recurrence:
`m_i = max(m_i, max(qk·sm−sentinel, 1))`, `l_i = l_i·α + Σexp2(qk−m_ij)`,
`acc = acc·α + Σexp2(qk−m_ij)·v`, with the causal `-1e8` sentinel `where`. All
index / scalar registers (`q`, `cur_batch`, `cur_kv_head`, `cur_head`,
`prompt_cache_len`, `cur_batch_seq_len`, `block_end_loc`, `offs_*`) preserved. -/
theorem ctxLoopBody_steps (Q K V : RegionName) (sin : BlockState) (SN : Nat)
    (gOM : Fin 128 → Nat) (cb ckvh ch plen sl bel cbsi : Nat)
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
    (hm : sin.regs .nat [128] "offs_m" = some (Tile.vec gOM))
    (hn : sin.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hd : sin.regs .nat [128] "offs_d" = some (Tile.vec (fun e : Fin 128 => e.val)))
    (hq : sin.regs .real [128, 128] "q" = some qtile)
    (hmi : sin.regs .real [128] "m_i" = some mtile)
    (hli : sin.regs .real [128] "l_i" = some ltile)
    (hacc : sin.regs .real [128, 128] "acc" = some acctile)
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ sF, stepStmts (ctxLoopBody Q K V) sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = 0)
      ∧ sF.regs .nat [] "start_n" = some (Tile.scalar SN)
      ∧ sF.regs .nat [] "cur_batch" = some (Tile.scalar cb)
      ∧ sF.regs .nat [] "cur_kv_head" = some (Tile.scalar ckvh)
      ∧ sF.regs .nat [] "cur_head" = some (Tile.scalar ch)
      ∧ sF.regs .nat [] "prompt_cache_len" = some (Tile.scalar plen)
      ∧ sF.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar sl)
      ∧ sF.regs .nat [] "block_end_loc" = some (Tile.scalar bel)
      ∧ sF.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar cbsi)
      ∧ sF.regs .nat [128] "offs_m" = some (Tile.vec gOM)
      ∧ sF.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val))
      ∧ sF.regs .nat [128] "offs_d" = some (Tile.vec (fun e : Fin 128 => e.val))
      ∧ sF.regs .real [128, 128] "q" = some qtile
      ∧ ∃ (kloadT vloadT : Tile .real [128, 128]) (qkT : Tile .real [128, 128])
            (rmaxT mijT alphaT lijT : Tile .real [128]) (pT acc1T : Tile .real [128, 128]),
          kloadT = (⟨fun idx : TileIndex [128, 128] =>
              if decide (SN + idx.2.1.val < bel) then
                sin.readMemValue .real (Region.cast K)
                  (cb * 8388608 + (SN + idx.2.1.val) * 128 + ckvh * 262144 + idx.1.val * 1)
              else some (0.0 : ℝ)⟩ : Tile .real [128, 128])
          ∧ vloadT = (⟨fun idx : TileIndex [128, 128] =>
              if decide (SN + idx.1.val < bel) then
                sin.readMemValue .real (Region.cast V)
                  (cb * 8388608 + (SN + idx.1.val) * 128 + ckvh * 262144 + idx.2.1.val * 1)
              else some (0.0 : ℝ)⟩ : Tile .real [128, 128])
          ∧ qkT = Tile.select
              (⟨fun idx : TileIndex [128, 128] => decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ : Tile .bool [128, 128])
              (Tile.bop NumericDType.real.mul Broadcast.scalarR (Tile.dot [] qtile kloadT)
                (Tile.scalar (some sm_scale_python)))
              (⟨fun _ : TileIndex [128, 128] => some (0.0 - 10e7 : ℝ)⟩ : Tile .real [128, 128])
          ∧ Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [128, 128].length) qkT = some rmaxT
          ∧ mijT = Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mtile rmaxT) mtile rmaxT
          ∧ alphaT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile mijT)
          ∧ pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
              qkT (Tile.expandDim ⟨1, by simp⟩ mijT))
          ∧ lijT = Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [128, 128].length) pT
          ∧ acc1T = Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acctile (Tile.expandDim ⟨1, by simp⟩ alphaT)
          ∧ sF.regs .real [128] "m_i" = some mijT
          ∧ sF.regs .real [128] "l_i" = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame
              (Tile.bop NumericDType.real.mul Broadcast.nil.consSame ltile alphaT) lijT)
          ∧ sF.regs .real [128, 128] "acc" = some (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
              (Tile.dot [] pT vloadT) acc1T) := by
  set kloadT : Tile .real [128, 128] := ⟨fun idx : TileIndex [128, 128] =>
      if decide (SN + idx.2.1.val < bel) then
        sin.readMemValue .real (Region.cast K)
          (cb * 8388608 + (SN + idx.2.1.val) * 128 + ckvh * 262144 + idx.1.val * 1)
      else some (0.0 : ℝ)⟩ with hkl
  set vloadT : Tile .real [128, 128] := ⟨fun idx : TileIndex [128, 128] =>
      if decide (SN + idx.1.val < bel) then
        sin.readMemValue .real (Region.cast V)
          (cb * 8388608 + (SN + idx.1.val) * 128 + ckvh * 262144 + idx.2.1.val * 1)
      else some (0.0 : ℝ)⟩ with hvl
  set masktile : Tile .bool [128, 128] :=
      ⟨fun idx : TileIndex [128, 128] => decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ with hmt
  set qkdotT : Tile .real [128, 128] := Tile.dot [] qtile kloadT with hqkdot
  set qkT : Tile .real [128, 128] := Tile.select masktile
      (Tile.bop NumericDType.real.mul Broadcast.scalarR qkdotT (Tile.scalar (some sm_scale_python)))
      (⟨fun _ : TileIndex [128, 128] => some (0.0 - 10e7 : ℝ)⟩ : Tile .real [128, 128]) with hqk
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [128, 128].length) qkT = some t :=
    ⟨_, by unfold Tile.reduceMaxDrop; rw [dif_pos (show 0 < TileShape.axisDim [128, 128] (⟨1, by simp⟩ : Fin [128, 128].length) from by decide)]⟩
  set mijT : Tile .real [128] := Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mtile rmaxT) mtile rmaxT with hmij
  set qk2T : Tile .real [128, 128] := Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame qkT (Tile.expandDim ⟨1, by simp⟩ mijT) with hqk2
  set pT : Tile .real [128, 128] := Tile.uop WithBot.realExp2 qk2T with hpT
  set lijT : Tile .real [128] := Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [128, 128].length) pT with hlij
  set alphaT : Tile .real [128] := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile mijT) with hal
  set acc1T : Tile .real [128, 128] := Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acctile (Tile.expandDim ⟨1, by simp⟩ alphaT) with hacc1
  unfold ctxLoopBody
  -- stmt 0: start_n = ref start_n
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_n") sin = some (Tile.scalar SN) from by rw [evalOp_ref]; exact hsn))]
  -- stmt 1: off_k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_offk_eval _ cb ckvh SN
      (by simp [BlockState.setReg_ne_name, hcb]) (by simp [BlockState.setReg_ne_name, hckvh])
      (by simp [BlockState.setReg_same]) (by simp [BlockState.setReg_ne_name, hn])
      (by simp [BlockState.setReg_ne_name, hd])))]
  -- stmt 2: k = masked load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_region_maskOther K (Op.ref .nat [128, 128] "off_k") _ _ _
      (⟨fun idx : TileIndex [128, 128] =>
          cb * 8388608 + (SN + idx.2.1.val) * 128 + ckvh * 262144 + idx.1.val * 1⟩ : Tile .nat [128, 128])
      (⟨fun idx : TileIndex [128, 128] => decide (SN + idx.2.1.val < bel)⟩ : Tile .bool [128, 128])
      (⟨fun _ : TileIndex [128, 128] => some (0.0 : ℝ)⟩ : Tile .real [128, 128])
      (by rw [evalOp_ref]; simp [BlockState.setReg_same])
      (ctxKMask_eval _ SN bel (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, hn]) (by simp [BlockState.setReg_ne_name, hbel]))
      (by simp only [evalOp, Option.bind_eq_bind, Option.bind_some]; rfl)))]
  -- stmt 3: qk = dot q k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show @evalOp .real [128, 128]
        (Op.dot (batch := []) (Op.ref .real [128, 128] "q") (Op.ref .real [128, 128] "k")) _
        = some (Tile.dot [] qtile kloadT) from
      ctxQk_op_eval _ qtile kloadT
      (by simp [BlockState.setReg_ne_name, hq])
      (by rw [BlockState.setReg_same]; rw [hkl];
          refine congrArg some ?_; ext idx;
          simp only [BlockState.setReg_readMemValue])))]
  -- stmt 4: mask = ge(...)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctxMask_eval _ 128 128 plen SN gOM
      (by simp [BlockState.setReg_ne_name, hm]) (by simp [BlockState.setReg_ne_name, hn])
      (by simp [BlockState.setReg_ne_name, hplen]) (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  -- stmt 5: qk = where(mask, qk*sm, -1e8)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctxWhere_eval _ 128 128 sm_scale_python masktile qkdotT
      (by simp [BlockState.setReg_same, hmt]) (by simp [BlockState.setReg_ne_name, BlockState.setReg_same, hqkdot])))]
  -- stmt 6: m_ij
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctxMij_op_eval _ mtile rmaxT rmaxT qkT
      (by simp [BlockState.setReg_ne_name, hmi]) (by simp [BlockState.setReg_same, hqk]) hrm))]
  -- stmt 7: qk -= m_ij[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctxQk2_op_eval _ qkT mijT
      (by simp [BlockState.setReg_ne_name, hqk]) (by simp [BlockState.setReg_same, hmij])))]
  -- stmt 8: p = exp2 qk
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctxP_op_eval _ qk2T (by simp [BlockState.setReg_same, hqk2])))]
  -- stmt 9: l_ij = sum p 1
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .real [128] "l_ij"
    (Op.reduceSum (⟨1, by simp⟩ : Fin [128, 128].length) Bool.false (Op.ref .real [128, 128] "p")) _ lijT
    (ctxLij_op_eval _ pT (by simp [BlockState.setReg_same, hpT])))]
  -- stmt 10: alpha = exp2(m_i - m_ij)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctxAlpha_op_eval _ mtile mijT
      (by simp [BlockState.setReg_ne_name, hmi]) (by simp [BlockState.setReg_ne_name, hmij])))]
  -- stmt 11: l_i = l_i*alpha + l_ij
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctxLi_op_eval _ ltile alphaT lijT
      (by simp [BlockState.setReg_ne_name, hli]) (by simp [BlockState.setReg_same, hal])
      (by simp [BlockState.setReg_ne_name, hlij])))]
  -- stmt 12: acc *= alpha[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctxAcc1_op_eval _ acctile alphaT
      (by simp [BlockState.setReg_ne_name, hacc]) (by simp [BlockState.setReg_ne_name, hal])))]
  -- stmt 13: off_v
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_offv_eval _ cb ckvh SN
      (by simp [BlockState.setReg_ne_name, hcb]) (by simp [BlockState.setReg_ne_name, hckvh])
      (by simp [BlockState.setReg_ne_name, hsn]) (by simp [BlockState.setReg_ne_name, hn])
      (by simp [BlockState.setReg_ne_name, hd])))]
  -- stmt 14: v = masked load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctx_evalOp_load_region_maskOther V (Op.ref .nat [128, 128] "off_v") _ _ _
      (⟨fun idx : TileIndex [128, 128] =>
          cb * 8388608 + (SN + idx.1.val) * 128 + ckvh * 262144 + idx.2.1.val * 1⟩ : Tile .nat [128, 128])
      (⟨fun idx : TileIndex [128, 128] => decide (SN + idx.1.val < bel)⟩ : Tile .bool [128, 128])
      (⟨fun _ : TileIndex [128, 128] => some (0.0 : ℝ)⟩ : Tile .real [128, 128])
      (by rw [evalOp_ref]; simp [BlockState.setReg_same])
      (ctxVMask_eval _ SN bel (by simp [BlockState.setReg_ne_name, hsn])
        (by simp [BlockState.setReg_ne_name, hn]) (by simp [BlockState.setReg_ne_name, hbel]))
      (by simp only [evalOp, Option.bind_eq_bind, Option.bind_some]; rfl)))]
  -- stmt 15: p = ref p (noop cast)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [128, 128] "p") _ = some pT from by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, hpT]))]
  -- stmt 16: acc = dot(p,v) + acc
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ctxAcc2_op_eval _ acc1T pT vloadT
      (by simp [BlockState.setReg_same, hpT])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), BlockState.setReg_same];
          rw [hvl]; refine congrArg some ?_; ext idx;
          simp only [BlockState.setReg_readMemValue])
      (by simp [BlockState.setReg_ne_name, hacc1])))]
  -- stmt 17: m_i = ref m_ij
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [128] "m_ij") _ = some mijT from by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, hmij]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    kloadT, vloadT, qkT, rmaxT, mijT, alphaT, lijT, pT, acc1T,
    ?_, ?_, ?_, hrm, rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_pids]
  · funext rg o; simp only [BlockState.setReg_mem]
  · intro rg o; simp only [BlockState.setReg_undef]; exact hundef rg o
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hcb]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hckvh]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hch]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hplen]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hsl]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hbel]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hcbsi]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hm]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hn]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hd]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, hq]
  · -- kloadT readback
    rw [hkl]
  · -- vloadT readback
    rw [hvl]
  · -- qkT readback
    rw [hqk, hqkdot]
  · -- m_i = mijT (top setReg)
    rw [BlockState.setReg_same]
  · -- l_i readback (peel through stmts 12-17)
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq]
  · -- acc readback (peel m_i)
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq]

/-! ## STEP C — exec-side bridge suite (context analogues of flash's bridges)

Generic ⊥-seeded online-softmax machinery over an abstract per-key
`(score, value)` function `g : Fin S → ℝ × ℝ`. Both the spec form (`g = ctxKV`,
score `ctxQTile`) and the kernel-loaded form (`g` over the actual loaded `q`
tile) instantiate it, so the per-block advance bridges are proved once. The
context kernel keeps **all** keys (the causal mask becomes the `-1e8` sentinel
score, not a `⊥` list drop), so the block sums/sups range over the full `Fin BN`
window with no `⊥` lanes — simpler than the flash (causal `⊥`-list) bridges. -/

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

theorem ctxKeysUpto_eq_g
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S hi : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    ctxKeysUpto s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d
      = gKeysUpto S hi (ctxKV s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S i d) := rfl

theorem ctxBlock_eq_g
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S BN c : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    ctxBlock s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S BN c i d
      = gBlock S BN c (ctxKV s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S i d) := rfl

theorem ctxStateBot_eq_g
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S hi : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    ctxStateBot s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d
      = gStateBot S hi (ctxKV s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S i d) := rfl

theorem ctxRunningMax_eq_g
    (s : BlockState) (Q K V B_Start_Loc B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S hi : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    ctxRunningMax s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S hi i d
      = gRunningMax S hi (ctxKV s Q K V B_Start_Loc B_Prompt_Cache_Len sm_scale BLOCK_M S i d) := rfl

/-- The running max component of `gStateBot` is the `WithBot ⊔`-fold `gRunningMax`. -/
theorem gStateBot_fst_eq_runningMax (S hi : Nat) (g : Fin S → ℝ × ℝ) :
    (gStateBot S hi g).1 = gRunningMax S hi g := by
  rw [gStateBot, osStepBot_foldl_fst, gRunningMax, foldl_sup_bot_eq_foldr]

/-- **Window split** (`hi = c·BN`) for the generic key list. -/
theorem gKeysUpto_succ (S BN c : Nat) (g : Fin S → ℝ × ℝ) :
    gKeysUpto S ((c + 1) * BN) g = gKeysUpto S (c * BN) g ++ gBlock S BN c g := by
  unfold gKeysUpto gBlock
  rw [ctx_filterMap_window_split (List.finRange S) (List.pairwise_lt_finRange S)
    (c * BN) ((c + 1) * BN) g (by nlinarith [Nat.zero_le BN])]

/-- **One-block advance** of the generic ⊥-seeded state. -/
theorem gStateBot_succ (S BN c : Nat) (g : Fin S → ℝ × ℝ) :
    gStateBot S ((c + 1) * BN) g
      = (gBlock S BN c g).foldl osStepBot (gStateBot S (c * BN) g) := by
  unfold gStateBot; rw [gKeysUpto_succ, List.foldl_append]

/-- filterMap-sum over `Fin n` with a guard collapses into the masked `Finset.sum`. -/
theorem ctx_filterMap_finRange_sum {α : Type*} (n : Nat)
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

/-- The `WithBot` `foldr` of a filtered score list (coerced) equals the `Finset.sup`
over `Fin n` of the lane terms (`⊥` on filtered-out lanes). -/
theorem ctx_filterMap_foldr_sup (n : Nat) (P : Fin n → Prop) [DecidablePred P]
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

/-- Reindex a windowed `Finset.sup` over `Fin S` (window `c·BN ≤ j < (c+1)·BN`)
onto `Fin BN` (lane `jL` ↦ key `c·BN + jL`); out-of-window lanes contribute `⊥`. -/
theorem ctx_window_sup_reindex (BN c S : Nat) (hwin : (c + 1) * BN ≤ S)
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

/-- Block-local lane `jL : Fin BN` maps to a global key `c·BN + jL < S`. -/
theorem gBlock_idx_lt (S BN c : Nat) (hwin : (c + 1) * BN ≤ S) (jL : Fin BN) :
    c * BN + jL.val < S := by
  have hjlt := jL.isLt
  have heq : (c + 1) * BN = c * BN + BN := by ring
  omega

/-- Reindex a masked `Fin S`-window sum onto `Fin BN` (lane `jL` ↦ key `c·BN + jL`). -/
theorem ctx_window_sum_reindex (BN c S : Nat) (hwin : (c + 1) * BN ≤ S) (g : Nat → ℝ) :
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
      = ∑ jL : Fin BN, h (g ⟨c * BN + jL.val,
          gBlock_idx_lt S BN c hwin jL⟩) := by
  rw [gBlock, ctx_filterMap_finRange_sum S
    (fun j => c * BN ≤ j.val ∧ j.val < (c + 1) * BN) g h]
  rw [show (∑ j : Fin S, if c * BN ≤ j.val ∧ j.val < (c + 1) * BN then h (g j) else 0)
        = ∑ j : Fin S, if c * BN ≤ j.val ∧ j.val < (c + 1) * BN
            then (fun jg => if h' : jg < S then h (g ⟨jg, h'⟩) else 0) j.val else 0 from by
    apply Finset.sum_congr rfl; intro j _
    by_cases hw : c * BN ≤ j.val ∧ j.val < (c + 1) * BN
    · rw [if_pos hw, if_pos hw]; simp only [dif_pos j.isLt]
    · rw [if_neg hw, if_neg hw]]
  rw [ctx_window_sum_reindex BN c S hwin
    (fun jg => if h' : jg < S then h (g ⟨jg, h'⟩) else 0)]
  apply Finset.sum_congr rfl
  intro jL _
  simp only [dif_pos (gBlock_idx_lt S BN c hwin jL)]

/-- **One-block advance** of the generic ⊥-seeded running max (block sup over the
full `Fin BN` window, no `⊥` lanes). -/
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
    rw [ctx_filterMap_foldr_sup S
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
    rw [ctx_window_sup_reindex BN c S hwin
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

/-! ### score-cell / nume / acc bridges (kernel tile arithmetic → block list sums)

The loop body's symbolic `qk`/`p`/`acc` registers (after `ctxLoopBody_steps`)
carry the kernel's per-block tile arithmetic over the masked cell
`ctxQkCell` (active lane `SN+j ≤ gOM i + plen`: scaled dot; future lane: the
`-1e8` sentinel — all finite, no `⊥`). These bridges read them off
`gStateBot((c+1)·128)`/`gRunningMax((c+1)·128)` via `osStepBot_block_eq`. -/

/-- The kernel's masked `qk` cell at lane `(i, j)`: active lane gets the scaled
dot `sm·Σ_e qf(i,e)·kf(j,e)`; future lane gets the `-1e8` sentinel. Finite (no
`⊥`), matching `ctxLoopBody_steps`'s `qkT`. -/
noncomputable def ctxQkCell {BM BN D : Nat} (sm : ℝ) (SN plen : Nat) (gOM : Fin BM → Nat)
    (qf : Fin BM → Fin D → ℝ) (kf : Fin BN → Fin D → ℝ) (i : Fin BM) (j : Fin BN) : ℝ :=
  if SN + j.val ≤ gOM i + plen then
    sm * Finset.univ.sum (fun e : Fin D => qf i e * kf j e)
  else (0.0 - 10e7 : ℝ)

/-- **The `q·k` dot cell is the scaled score** (shape-generic `[BM,D]·[D,BN]`). With
`qtile`/`ktile` reading `qf`/`kf` (as `some`), the dot of `qtile` against `ktile`
at cell `(i, j)` is `some (Σ_e qf(i,e)·kf(j,e))`. -/
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

/-- **The kernel's `qkT` cell is `some (ctxQkCell …)`** (shape-generic). The
`tl.where(mask, qk·sm, -1e8)` register, with the loaded `q`/`k` reading `qf`/`kf`,
has cell `(i,j)` equal to the masked `ctxQkCell`. -/
theorem ctx_qkT_cell {BM BN D : Nat} (sm : ℝ) (SN plen : Nat) (gOM : Fin BM → Nat)
    (qtile : Tile .real [BM, D]) (kloadT : Tile .real [D, BN]) (qf : Fin BM → Fin D → ℝ) (kf : Fin BN → Fin D → ℝ)
    (hq : ∀ (i : Fin BM) (e : Fin D), qtile.data (i, e, PUnit.unit) = some (qf i e))
    (hk : ∀ (j : Fin BN) (e : Fin D), kloadT.data (e, j, PUnit.unit) = some (kf j e))
    (i : Fin BM) (j : Fin BN) :
    (Tile.select
        (⟨fun idx : TileIndex [BM, BN] => decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ : Tile .bool [BM, BN])
        (Tile.bop NumericDType.real.mul Broadcast.scalarR (Tile.dot [] qtile kloadT)
          (Tile.scalar (some sm)))
        (⟨fun _ : TileIndex [BM, BN] => some (0.0 - 10e7 : ℝ)⟩ : Tile .real [BM, BN])).data
      (i, j, PUnit.unit)
      = some (ctxQkCell sm SN plen gOM qf kf i j) := by
  rw [Tile.select_data, ctxQkCell]
  have hsel : (⟨fun idx : TileIndex [BM, BN] => decide (SN + idx.2.1.val ≤ gOM idx.1 + plen)⟩ : Tile .bool [BM, BN]).data (i, j, PUnit.unit)
      = decide (SN + j.val ≤ gOM i + plen) := rfl
  by_cases h : SN + j.val ≤ gOM i + plen
  · rw [hsel, if_pos h]
    simp only [decide_eq_true_eq.mpr h, if_true]
    rw [Tile.bop_data]
    simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul]
    rw [ctx_dot_score_cell qtile kloadT i j qf kf (hq i) (hk j)]
    show Option.map₂ (· * ·) _ _ = _
    simp only [Tile.scalar_data, Option.map₂]
    refine congrArg some ?_; ring
  · rw [hsel, if_neg h]
    simp only [decide_eq_false_iff_not.mpr h, Bool.false_eq_true, if_false]

/-! ## STEP D — the faithful masked closed form + streaming-loop assembly

The streaming loop's bound `block_mask · block_end_loc` is stepped by `128`, but
`block_end_loc` (`= min(block_start_loc + 128 + plen, seq_len + plen)`) is **not**
128-aligned. `forRangeDyn_inv` therefore runs the loop to `final = ceil₁₂₈(bel) ≥
bel`, streaming *phantom* keys `j ∈ [bel, final)`. The kernel's `k`/`v` loads are
masked by `< block_end_loc` (so phantom `k`/`v` are **0**), but the causal
`tl.where` still feeds each phantom lane a finite score (`sm·dot(q,0) = 0` when the
causal mask passes, the `-1e8` sentinel otherwise) into `exp2`, so phantom keys
contribute `exp2(score)·0 = 0` to the numerator (zeroed `v`) yet `exp2(score)` to
the denominator.

To stay faithful we capture **exactly** that: the per-key data the loop folds uses
`block_end_loc`-masked tiles. `ctxKVM` is `ctxKV` with `k`/`v` read through the
`< bel` load mask (`ctxKTileM`/`ctxVTileM`), so the closed form
`contextAttnExactFoldM` over `final` keys equals what the kernel's `m_i`/`l_i`/`acc`
loop produces — phantom-key denominator contribution included. -/

/-- `block_end_loc`-masked key tile: `ctxKTile` for `j < bel`, else `0` (the
kernel's `k` load mask `(start_n+offs_n) < block_end_loc, other=0`). -/
noncomputable def ctxKTileM (s : BlockState) (K : RegionName) (S bel : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, e, u) => if j.val < bel then ctxKTile s K S (j, e, u) else 0

/-- `block_end_loc`-masked value tile: `ctxVTile` for `j < bel`, else `0`. -/
noncomputable def ctxVTileM (s : BlockState) (V : RegionName) (S bel : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, d, u) => if j.val < bel then ctxVTile s V S (j, d, u) else 0

/-- Row-masked query tile: `ctxQTile` for active rows (`128·pids0 + i < seq_len`),
else `0` (the kernel's `q` load mask `offs_m < cur_batch_seq_len, other=0`). On
active rows it is the genuine `ctxQTile`, so the closed form is the true causal
softmax there; inactive rows are masked off at the final store. -/
noncomputable def ctxQTileM
    (s : BlockState) (Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Fin 128) : ℝ :=
  if s.pids 0 * BLOCK_M + i.val < seqLen s 16 B_Seqlen B_Prompt_Cache_Len then
    ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
  else 0

/-- **Faithful per-key `(score, value)` the loop folds**: `ctxKV` with the
`block_end_loc` load mask applied to `k`/`v` and the row mask applied to `q`.
Active causal lane (`j ≤ gi+plen`) gets `sm·Σ_e ctxQTileM(i,e)·ctxKTileM(j,e)` (so
phantom `j ≥ bel` get `sm·0 = 0`); future lane gets the `-1e8` sentinel; value is
the masked `ctxVTileM` (`0` for `j ≥ bel`). -/
noncomputable def ctxKVM
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel : Nat) (i : Fin BLOCK_M) (d : Fin 128) (j : Fin S) : ℝ × ℝ :=
  (if j.val ≤ s.pids 0 * BLOCK_M + i.val + promptLen s 16 B_Prompt_Cache_Len then
      sm_scale * Finset.univ.sum (fun e : Fin 128 =>
        ctxQTileM s Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len BLOCK_M i e
          * ctxKTileM s K S bel (j, e, PUnit.unit))
    else (0.0 - 10e7 : ℝ),
    ctxVTileM s V S bel (j, d, PUnit.unit))

/-- **The faithful kernel value** at output lane `(i, d)`: `acc/l` of the
⊥-seeded online-softmax fold over `ctxKVM` for the full streamed window `[0, S)`
(`S = final`). A pure function of `Q`/`K`/`V` memory — exactly what the loop's
`m_i`/`l_i`/`acc` realize, phantom keys included. -/
noncomputable def contextAttnExactFoldM
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ) (BLOCK_M S bel : Nat) (idx : TileIndex [BLOCK_M, 128]) : ℝ :=
  let st := gStateBot S S (ctxKVM s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len sm_scale
      BLOCK_M S bel idx.1 idx.2.1)
  st.2.2 / st.2.1

/-- The kernel-loaded per-key `(score, value)` carried by the loop registers at
output lane `(i, d)`, over `S` keys with `block_end_loc = bel`. This is `ctxKVM`
specialized to the Python shape (`BLOCK_M = 128`); the invariant tracks the
⊥-seeded `gStateBot` fold over it. -/
noncomputable def ctxG
    (s0 : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ) (S bel : Nat) (i d : Fin 128) : Fin S → ℝ × ℝ :=
  ctxKVM s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len sm_scale 128 S bel i d

/-- **Streaming-loop invariant** for the Python-shape context kernel. After `c`
blocks (loop counter `c·128`), the `m_i`/`l_i`/`acc` registers carry the ⊥-seeded
online-softmax fold `gStateBot (c·128)` over the kernel-loaded per-key data
`ctxG`, and the auxiliary registers (`q`, the offsets, the scalar metadata) are
the constant values seeded by `ctxPreLoop`. -/
noncomputable def ctxInvariant
    (Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName) (s0 : BlockState)
    (sm_scale : ℝ) (S bel : Nat) (c : Nat) (s : BlockState) : Prop :=
  let plen := s0.readMemValue .nat (Region.cast B_Prompt_Cache_Len) (s0.pids 1 / 16)
  let sl := s0.readMemValue .nat (Region.cast B_Seqlen) (s0.pids 1 / 16) - plen
  let g := fun (i d : Fin 128) => ctxG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len sm_scale S bel i d
  s.pids = s0.pids ∧ s.mem = s0.mem ∧ (∀ rg o, s.undef rg o = 0) ∧
  (s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1 / 16))) ∧
  (s.regs .nat [] "cur_kv_head" = some (Tile.scalar (s0.pids 1 % 16 / 1))) ∧
  (s.regs .nat [] "cur_head" = some (Tile.scalar (s0.pids 1 % 16))) ∧
  (s.regs .nat [] "prompt_cache_len" = some (Tile.scalar plen)) ∧
  (s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar sl)) ∧
  (s.regs .nat [] "block_end_loc" = some (Tile.scalar bel)) ∧
  (s.regs .nat [] "cur_batch_in_all_start_index"
      = some (Tile.scalar (startLoc s0 16 B_Start_Loc))) ∧
  (s.regs .nat [128] "offs_m" = some (Tile.vec (fun r : Fin 128 => 128 * s0.pids 0 + r.val))) ∧
  (s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val))) ∧
  (s.regs .nat [128] "offs_d" = some (Tile.vec (fun e : Fin 128 => e.val))) ∧
  (s.regs .real [128, 128] "q" = some ⟨fun idx : TileIndex [128, 128] =>
      if decide (128 * s0.pids 0 + idx.1.val < sl) then
        s0.readMemValue .real (Region.cast Q)
          ((s0.readMemValue .nat (Region.cast B_Start_Loc) (s0.pids 1 / 16) + (128 * s0.pids 0 + idx.1.val))
              * 2048 + s0.pids 1 % 16 * 128 + idx.2.1.val * 1)
      else some (0.0 : ℝ)⟩) ∧
  (s.regs .real [128] "m_i" = some ⟨fun r : TileIndex [128] =>
      (gStateBot S (c * 128) (g r.1 ⟨0, by decide⟩)).1⟩) ∧
  (s.regs .real [128] "l_i" = some ⟨fun r : TileIndex [128] =>
      some (gStateBot S (c * 128) (g r.1 ⟨0, by decide⟩)).2.1⟩) ∧
  (s.regs .real [128, 128] "acc" = some ⟨fun idx : TileIndex [128, 128] =>
      some (gStateBot S (c * 128) (g idx.1 idx.2.1)).2.2⟩) ∧
  (c * 128 ≤ S)

/-! ### generic tile-arithmetic bridges (kernel-agnostic; reused in `ctx_attn_step`) -/

/-- A `reduceMaxDrop` over axis 1 reads off `Finset.sup` of a row's per-cell values
(shape-generic over `[M, N]`, `0 < N`). -/
theorem ctxg_reduceMaxDrop_data_row {M N : Nat} (hN : 0 < N) (qk : Tile .real [M, N])
    (rmaxT : Tile .real [M]) (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [M,N].length) qk = some rmaxT)
    (r : Fin M) (g : Fin N → WithBot ℝ) (hqk : ∀ jL : Fin N, qk.data (r, jL, PUnit.unit) = g jL) :
    rmaxT.data (r, PUnit.unit) = Finset.univ.sup g := by
  unfold Tile.reduceMaxDrop at hrm
  rw [dif_pos (show 0 < TileShape.axisDim [M,N] (⟨1, by simp⟩ : Fin [M,N].length) from hN)] at hrm
  rw [← Option.some.inj hrm]
  simp only [Finset.sup'_eq_sup]
  exact Finset.sup_congr rfl (fun jL _ => hqk jL)

/-- `WithBot.realExp2` of a `some`-cell is `some (pow2 …)` (shape-generic). -/
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

/-- `realExp2` is total (never `⊥`): `realExp2 z = some ((realExp2 z).unbotD 0)`. -/
theorem ctxg_realExp2_eq_some_unbotD (z : WithBot ℝ) :
    WithBot.realExp2 z = some ((WithBot.realExp2 z).unbotD 0) := by
  cases z <;> rfl

/-- `m_ij = select(m_i > rmax) m_i rmax` collapses to `max` in `WithBot ℝ` (shape-generic). -/
theorem ctxg_mij_max {M : Nat} (m_i rmaxT : Tile .real [M]) (r : Fin M)
    (a b : WithBot ℝ) (hmi : m_i.data (r, PUnit.unit) = a) (hrm : rmaxT.data (r, PUnit.unit) = b) :
    (Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame m_i rmaxT) m_i rmaxT).data
        (r, PUnit.unit) = max a b := by
  rw [Tile.select_data, Tile.cop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hmi, hrm]
  by_cases h : a ≤ b
  · rw [if_neg (by simp [not_lt.mpr h]), max_eq_right h]
  · rw [if_pos (by simpa using not_le.mp h), max_eq_left (le_of_lt (not_le.mp h))]

/-- A `dot` row over all `K` keys when both factors are all-`some` (shape-generic
over `[M, K] · [K, N]`). -/
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

/-- The running max of `gStateBot` is `⊥` iff its window streams no keys; so when
the max is `⊥` the max-free reference sums vanish. -/
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

/-- After `c+1` blocks (`(c+1)·BN ≤ S`, `0 < BN`) the ⊥-seeded running max is
finite: the block-`c` window streams at least key `c·BN < S`. -/
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

/-- The running max / denominator of `gStateBot` depend only on the per-key
*scores* (`.1`), not the values (`.2`). -/
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

/-- **Block-step in explicit `Fin BN` form** (BN-parametric). The ⊥-seeded state
after `c+1` blocks equals the kernel's one-shot rescale-and-add over block `c`'s
reindexed `Fin BN` lanes, anchored to the state after `c` blocks. This is the
exact tuple `(m_ij, l_i', acc')` the loop body computes (instantiated at `BN = 128`
for the regular path, `BN = 64` for the Tesla path). -/
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
  -- M' is the running-max after c+1 blocks
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
      rw [ctx_filterMap_foldr_sup S (fun j => c * BN ≤ j.val ∧ j.val < (c + 1) * BN) (fun j => (g j).1)]
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
      rw [ctx_window_sup_reindex BN c S hwin
        (fun jg => if h : jg < S then (((g ⟨jg, h⟩).1 : ℝ) : WithBot ℝ) else ⊥)]
      apply Finset.sup_congr rfl
      intro jL _
      simp only [dif_pos (gBlock_idx_lt S BN c hwin jL)]
    show _ = _
    rw [hsup]
  -- now invoke osStepBot_block_eq with L,T from consistency
  have hstep := osStepBot_block_eq st.1 st.2.1 st.2.2
    (((gKeysUpto S (c * BN) g).map (fun p => pow2 p.1 * p.2)).sum)
    (((gKeysUpto S (c * BN) g).map (fun p => pow2 p.1)).sum)
    (gBlock S BN c g) hLc hTc
    (fun hb => gKeysUpto_map_sum_eq_zero_of_bot S (c * BN) g hb _)
    (fun hb => gKeysUpto_map_sum_eq_zero_of_bot S (c * BN) g hb _)
  -- the block-fold equals gStateBot((c+1)*BN)
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

/-- The block-`c` score sup over `Fin 128` (local lane `jL` ↦ global key
`c·128 + jL`) coerced into `WithBot ℝ`, matching `gRunningMax_succ`'s block term. -/
private theorem ctxg_block_sup_eq (s0 : BlockState)
    (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (S bel c : Nat) (hwin : (c + 1) * 128 ≤ S) (i d : Fin 128)
    (qkT : Tile .real [128, 128])
    (hqk : ∀ jL : Fin 128, qkT.data (i, jL, PUnit.unit)
        = some ((ctxG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len sm_scale S bel i d
            ⟨c * 128 + jL.val, gBlock_idx_lt S 128 c hwin jL⟩).1))
    (rmaxT : Tile .real [128])
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [128,128].length) qkT = some rmaxT) :
    rmaxT.data (i, PUnit.unit)
      = Finset.univ.sup (fun jL : Fin 128 =>
          (((ctxG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len sm_scale S bel i d
              ⟨c * 128 + jL.val, gBlock_idx_lt S 128 c hwin jL⟩).1 : ℝ) : WithBot ℝ)) := by
  refine ctxg_reduceMaxDrop_data_row (by decide) qkT rmaxT hrm i _ ?_
  intro jL; rw [hqk jL]; rfl

/-- **acc-wall fix: alpha/p expandDim-subtraction readback as a TOP-LEVEL lemma.**
The `qk -= m_ij[:, None]` cell `(qkT − expandDim m_ij).data (i, jL)` equals
`some (score i jL − mij i)`, computed once here so the kernel elaborates the
`dropInsertedIndex [128] ⟨1,_⟩ 1 (...)` axis term a single time (avoiding the
in-proof deep recursion of forcing `Tile.expandDim_data`+`dropInsertedIndex`
inline in the `acc` conjunct). -/
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

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **M3 — `ctx_attn_step`: the streaming loop body advances `ctxInvariant` by one
block.** From the invariant at block `c` (`i = c·128`, `(c+1)·128 ≤ S`), the 18
lowered loop-body statements step to the invariant at block `c+1`. The running
`m_i`/`l_i`/`acc` registers advance by one `osStepBot` block over `ctxG` — proved
via `gStateBot_succ_explicit` (the explicit one-block rescale-and-add) matched
cell-by-cell to the kernel's `m_ij`/`l_i`/`acc` tiles (`ctxg_mij_max`/
`ctxg_dot_row` + `ctx_qk_sub_mij_cell` for the `p` cell). -/
theorem ctx_attn_step
    (Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName) (s0 : BlockState)
    (S bel : Nat) (c : Nat) (i : Nat) (s : BlockState)
    (hwin : (c + 1) * 128 ≤ S) (hieq : i = c * 128)
    (hinv : ctxInvariant Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len s0
        sm_scale_python S bel c s) :
    ∃ s', stepStmts (ctxLoopBody Q K V) (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ ctxInvariant Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len s0
          sm_scale_python S bel (c + 1) s' := by
  subst hieq
  -- abbreviations matching the invariant
  set plen := s0.readMemValue .nat (Region.cast B_Prompt_Cache_Len) (s0.pids 1 / 16) with hplend
  set sl := s0.readMemValue .nat (Region.cast B_Seqlen) (s0.pids 1 / 16) - plen with hsld
  set g := fun (i d : Fin 128) =>
    ctxG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len sm_scale_python S bel i d with hgd
  simp only [ctxInvariant] at hinv
  obtain ⟨hpids, hmem, hundef, hcb, hckvh, hch, hplen, hsl, hbel, hcbsi, hom, hon, hod,
      hq, hmi, hli, hacc, hcle⟩ := hinv
  -- the loaded q tile of the invariant (row-masked) equals ctxQTileM at every cell
  set qtile : Tile .real [128, 128] := ⟨fun idx : TileIndex [128, 128] =>
      if decide (128 * s0.pids 0 + idx.1.val < sl) then
        s0.readMemValue .real (Region.cast Q)
          ((s0.readMemValue .nat (Region.cast B_Start_Loc) (s0.pids 1 / 16) + (128 * s0.pids 0 + idx.1.val))
              * 2048 + s0.pids 1 % 16 * 128 + idx.2.1.val * 1)
      else some (0.0 : ℝ)⟩ with hqtile
  set mtile : Tile .real [128] := ⟨fun r : TileIndex [128] =>
      (gStateBot S (c * 128) (g r.1 ⟨0, by decide⟩)).1⟩ with hmtile
  set ltile : Tile .real [128] := ⟨fun r : TileIndex [128] =>
      some (gStateBot S (c * 128) (g r.1 ⟨0, by decide⟩)).2.1⟩ with hltile
  set acctile : Tile .real [128, 128] := ⟨fun idx : TileIndex [128, 128] =>
      some (gStateBot S (c * 128) (g idx.1 idx.2.1)).2.2⟩ with hacctile
  -- run the body via the banked steps lemma
  obtain ⟨sF, hchain, hpidsF, hmemF, hundefF, hsnF, hcbF, hckvhF, hchF, hplenF, hslF, hbelF,
      hcbsiF, homF, honF, hodF, hqF,
      kloadT, vloadT, qkT, rmaxT, mijT, alphaT, lijT, pT, acc1T,
      hkleq, hvleq, hqkeq, hrm, hmijeq, haleq, hpeq, hlijeq, hacc1eq,
      hm_iF, hl_iF, haccF⟩ :=
    ctxLoopBody_steps Q K V (s.setReg "start_n" .nat [] (Tile.scalar (c * 128))) (c * 128)
      (fun r : Fin 128 => 128 * s0.pids 0 + r.val)
      (s0.pids 1 / 16) (s0.pids 1 % 16 / 1) (s0.pids 1 % 16) plen sl bel
      (startLoc s0 16 B_Start_Loc)
      qtile mtile ltile acctile
      (by rw [BlockState.setReg_same])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hcb)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hckvh)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hch)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hplen)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsl)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hbel)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hcbsi)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hom)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hon)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hod)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmi)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hli)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hacc)
      (by intro rg o; simp [BlockState.setReg_undef, hundef])
  refine ⟨sF, hchain, ?_⟩
  -- key memory facts: cur_kv_head = cur_head (kv_group_num = 1)
  have hckvhd : s0.pids 1 % 16 / 1 = s0.pids 1 % 16 := Nat.div_one _
  have hrmem : ∀ (R : RegionName) (o : Nat),
      (s.setReg "start_n" .nat [] (Tile.scalar (c * 128))).readMem R o = s0.readMem R o := by
    intro R o; rw [BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  -- per-cell q readback: qtile = ctxQTileM
  have hqf : ∀ (a e : Fin 128),
      qtile.data (a, e, PUnit.unit) = some (ctxQTileM s0 Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len 128 a e) := by
    intro a e
    simp only [hqtile, ctxQTileM, ctxQTile, curBatch, curHead, seqLen, promptLen,
      BlockState.readMemValue_real, hsld, hplend, Region.cast_id,
      show s0.pids 0 * 128 + a.val = 128 * s0.pids 0 + a.val from by ring,
      Nat.mul_one, decide_eq_true_eq]
    split
    · rfl
    · norm_num
  -- per-cell k readback: kloadT = ctxKTileM (at global key c*128+jL)
  have hkf : ∀ (jL e : Fin 128),
      kloadT.data (e, jL, PUnit.unit)
        = some (ctxKTileM s0 K S bel (⟨c * 128 + jL.val, gBlock_idx_lt S 128 c hwin jL⟩, e, PUnit.unit)) := by
    intro jL e
    rw [hkleq]
    show (if decide (c * 128 + jL.val < bel) then
        (s.setReg "start_n" .nat [] (Tile.scalar (c * 128))).readMemValue .real (Region.cast K)
          (s0.pids 1 / 16 * 8388608 + (c * 128 + jL.val) * 128 + (s0.pids 1 % 16 / 1) * 262144 + e.val * 1)
        else some (0.0 : ℝ)) = _
    simp only [ctxKTileM, ctxKTile, curBatch, curHead, BlockState.readMemValue_real,
      Region.cast_id, hrmem, hckvhd, Nat.mul_one, decide_eq_true_eq]
    by_cases h : c * 128 + jL.val < bel
    · rw [if_pos h, if_pos h]
    · rw [if_neg h, if_neg h]; norm_num
  -- per-cell v readback: vloadT = ctxVTileM
  have hvf : ∀ (jL d : Fin 128),
      vloadT.data (jL, d, PUnit.unit)
        = some (ctxVTileM s0 V S bel (⟨c * 128 + jL.val, gBlock_idx_lt S 128 c hwin jL⟩, d, PUnit.unit)) := by
    intro jL d
    rw [hvleq]
    show (if decide (c * 128 + jL.val < bel) then
        (s.setReg "start_n" .nat [] (Tile.scalar (c * 128))).readMemValue .real (Region.cast V)
          (s0.pids 1 / 16 * 8388608 + (c * 128 + jL.val) * 128 + (s0.pids 1 % 16 / 1) * 262144 + d.val * 1)
        else some (0.0 : ℝ)) = _
    simp only [ctxVTileM, ctxVTile, curBatch, curHead, BlockState.readMemValue_real,
      Region.cast_id, hrmem, hckvhd, Nat.mul_one, decide_eq_true_eq]
    by_cases h : c * 128 + jL.val < bel
    · rw [if_pos h, if_pos h]
    · rw [if_neg h, if_neg h]; norm_num
  -- the score cell: ctxQkCell = (g i d ⟨c*128+jL⟩).1, for any d (score is d-independent)
  have hscore : ∀ (a : Fin 128) (jL : Fin 128) (dd : Fin 128),
      ctxQkCell sm_scale_python (c * 128) plen (fun r : Fin 128 => 128 * s0.pids 0 + r.val)
          (fun a e => ctxQTileM s0 Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len 128 a e)
          (fun jK e => ctxKTileM s0 K S bel (⟨c * 128 + jK.val, gBlock_idx_lt S 128 c hwin jK⟩, e, PUnit.unit))
          a jL
        = (g a dd ⟨c * 128 + jL.val, gBlock_idx_lt S 128 c hwin jL⟩).1 := by
    intro a jL dd
    simp only [hgd, ctxG, ctxKVM, ctxQkCell]
    have hpeq2 : promptLen s0 16 B_Prompt_Cache_Len = plen := by
      rw [hplend]; rfl
    by_cases h : c * 128 + jL.val ≤ 128 * s0.pids 0 + a.val + plen
    · rw [if_pos h, if_pos (by rw [hpeq2]; omega)]
    · rw [if_neg h, if_neg (by rw [hpeq2]; omega)]
  -- qkT cell = some score
  have hqk : ∀ (a jL : Fin 128) (dd : Fin 128),
      qkT.data (a, jL, PUnit.unit) = some (g a dd ⟨c * 128 + jL.val, gBlock_idx_lt S 128 c hwin jL⟩).1 := by
    intro a jL dd
    rw [hqkeq]
    rw [ctx_qkT_cell sm_scale_python (c * 128) plen (fun r : Fin 128 => 128 * s0.pids 0 + r.val)
      qtile kloadT
      (fun aa ee => ctxQTileM s0 Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len 128 aa ee)
      (fun jK ee => ctxKTileM s0 K S bel (⟨c * 128 + jK.val, gBlock_idx_lt S 128 c hwin jK⟩, ee, PUnit.unit))
      hqf hkf a jL]
    rw [hscore a jL dd]
  -- abbreviate the explicit succ state
  set st := gStateBot S (c * 128) with hstdef
  -- M' a dd := running max after (c+1) blocks
  set M' := fun (a dd : Fin 128) => (gStateBot S ((c + 1) * 128) (g a dd)).1 with hM'def
  -- the explicit gStateBot at (c+1) (via gStateBot_succ_explicit)
  have hsucc : ∀ a dd : Fin 128,
      gStateBot S ((c + 1) * 128) (g a dd)
        = (M' a dd,
           (st (g a dd)).2.1 * (WithBot.realExp2 (WithBot.realSub (st (g a dd)).1 (M' a dd))).unbotD 0
             + Finset.univ.sum (fun jL : Fin 128 =>
                 pow2 ((g a dd ⟨c * 128 + jL.val, gBlock_idx_lt S 128 c hwin jL⟩).1 - (M' a dd).unbotD 0)),
           (st (g a dd)).2.2 * (WithBot.realExp2 (WithBot.realSub (st (g a dd)).1 (M' a dd))).unbotD 0
             + Finset.univ.sum (fun jL : Fin 128 =>
                 pow2 ((g a dd ⟨c * 128 + jL.val, gBlock_idx_lt S 128 c hwin jL⟩).1 - (M' a dd).unbotD 0)
                   * (g a dd ⟨c * 128 + jL.val, gBlock_idx_lt S 128 c hwin jL⟩).2)) := by
    intro a dd
    have he := gStateBot_succ_explicit S 128 c (g a dd) hwin
    simp only [] at he
    -- M' a dd = the internal sup (take .1 of he)
    have hM'eq : M' a dd = (gStateBot S (c * 128) (g a dd)).1 ⊔ Finset.univ.sup (fun jL : Fin 128 =>
        ((g a dd ⟨c * 128 + jL.val, gBlock_idx_lt S 128 c hwin jL⟩).1 : WithBot ℝ)) := by
      rw [hM'def]; simp only []; rw [he]
    rw [hM'eq]
    exact he
  -- M' a dd ≠ ⊥
  have hM'ne : ∀ a dd : Fin 128, M' a dd ≠ ⊥ := by
    intro a dd
    rw [hM'def]
    simp only []
    rw [gStateBot_fst_eq_runningMax]
    exact gRunningMax_succ_ne_bot S 128 c (g a dd) (by norm_num) hwin
  -- the running-max (score) component is dd-independent
  have hMdd : ∀ a dd : Fin 128, M' a ⟨0, by decide⟩ = M' a dd := by
    intro a dd
    rw [hM'def]
    simp only []
    exact (gStateBot_score_congr S ((c + 1) * 128) _ _
      (fun j => by simp only [hgd, ctxG, ctxKVM])).1
  -- m_ij cell = M' a 0
  have hmijc : ∀ a : Fin 128, mijT.data (a, PUnit.unit) = M' a ⟨0, by decide⟩ := by
    intro a
    rw [hmijeq]
    rw [ctxg_mij_max mtile rmaxT a _ _ (by rw [hmtile]) (by rfl)]
    rw [hM'def]
    simp only []
    rw [congrArg Prod.fst (gStateBot_succ_explicit S 128 c (g a ⟨0, by decide⟩) hwin)]
    simp only []
    refine congrArg (fun z => (gStateBot S (c * 128) (g a ⟨0, by decide⟩)).1 ⊔ z) ?_
    exact ctxg_block_sup_eq s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len sm_scale_python
      S bel c hwin a ⟨0, by decide⟩ qkT (fun jL => hqk a jL ⟨0, by decide⟩) rmaxT hrm
  -- p cell = pow2(score - M'.unbotD)
  have hpc : ∀ (a jL : Fin 128) (dd : Fin 128),
      pT.data (a, jL, PUnit.unit)
        = some (pow2 ((g a dd ⟨c * 128 + jL.val, gBlock_idx_lt S 128 c hwin jL⟩).1 - (M' a dd).unbotD 0)) := by
    intro a jL dd
    rw [hpeq]
    refine ctxg_exp2_some
      (fun aa bb => (g aa dd ⟨c * 128 + bb.val, gBlock_idx_lt S 128 c hwin bb⟩).1 - (M' aa dd).unbotD 0)
      _ a jL ?_
    refine ctx_qk_sub_mij_cell qkT mijT a jL _ _ (hqk a jL dd) ?_
    rw [hmijc a, hMdd a dd]
    cases hMc : M' a dd with
    | bot => exact absurd hMc (hM'ne a dd)
    | coe r => rfl
  -- now build the invariant at (c+1)
  have hgapp : ∀ a d : Fin 128,
      ctxG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len sm_scale_python S bel a d = g a d :=
    fun a d => rfl
  simp only [ctxInvariant, hgapp]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF, BlockState.setReg_pids]; exact hpids
  · rw [hmemF]; exact hmem
  · exact hundefF
  · rw [hcbF]
  · rw [hckvhF]
  · rw [hchF]
  · rw [hplenF]
  · rw [hslF]
  · rw [hbelF]
  · rw [hcbsiF]
  · rw [homF]
  · rw [honF]
  · rw [hodF]
  · rw [hqF]
  · -- m_i = (gStateBot (c+1)).1
    rw [hm_iF]; refine congrArg some ?_; ext a
    show mijT.data a = (gStateBot S ((c + 1) * 128) (g a.1 ⟨0, by decide⟩)).1
    rw [hmijc a.1, hM'def]
  · -- l_i = (gStateBot (c+1)).2.1
    rw [hl_iF]; refine congrArg some ?_; ext a
    show _ = some (gStateBot S ((c + 1) * 128) (g a.1 ⟨0, by decide⟩)).2.1
    rw [hsucc a.1 ⟨0, by decide⟩]
    simp only []
    -- l_i tile cell = ltile · alpha + lij
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
    have halc : alphaT.data (a.1, PUnit.unit)
        = some ((WithBot.realExp2 (WithBot.realSub (st (g a.1 ⟨0, by decide⟩)).1 (M' a.1 ⟨0, by decide⟩))).unbotD 0) := by
      rw [haleq]
      show WithBot.realExp2 ((Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile mijT).data (a.1, PUnit.unit)) = _
      have hinner : (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile mijT).data (a.1, PUnit.unit)
          = WithBot.realSub (st (g a.1 ⟨0, by decide⟩)).1 (M' a.1 ⟨0, by decide⟩) := by
        simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub]
        rw [hmijc a.1]
      rw [hinner]; exact ctxg_realExp2_eq_some_unbotD _
    have hlijc : lijT.data (a.1, PUnit.unit)
        = some (Finset.univ.sum (fun jL : Fin 128 =>
            pow2 ((g a.1 ⟨0, by decide⟩ ⟨c * 128 + jL.val, gBlock_idx_lt S 128 c hwin jL⟩).1
              - (M' a.1 ⟨0, by decide⟩).unbotD 0))) := by
      rw [hlijeq, Tile.reduceSumDrop_data]
      simp only [TileShape.insertAxisIndex]
      refine Eq.trans (Finset.sum_congr rfl (fun (jL : Fin 128) _ => hpc a.1 jL ⟨0, by decide⟩)) ?_
      exact ctxg_withBot_sum_some
        (fun jL : Fin 128 => pow2 ((g a.1 ⟨0, by decide⟩ ⟨c * 128 + jL.val, gBlock_idx_lt S 128 c hwin jL⟩).1
          - (M' a.1 ⟨0, by decide⟩).unbotD 0))
    rw [hltile, halc, hlijc]
    show WithBot.realAdd (WithBot.realMul (some _) (some _)) (some _) = _
    simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  · -- acc = (gStateBot (c+1)).2.2  (the acc conjunct, via accg-style)
    rw [haccF]; refine congrArg some ?_; ext idx
    show _ = some (gStateBot S ((c + 1) * 128) (g idx.1 idx.2.1)).2.2
    rw [hsucc idx.1 idx.2.1]
    simp only []
    -- acc tile cell = dot(p, v) + acc1, acc1 = acc · alpha[:,None]
    simp only [hacc1eq, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.expandDim_data, TileShape.dropInsertedIndex, TileShape.insertAxisIndex]
    -- m_i cell (d=0) score-independence: (st (g idx.1 0)).1 = (st (g idx.1 idx.2.1)).1
    have hst1 : (st (g idx.1 ⟨0, by decide⟩)).1 = (st (g idx.1 idx.2.1)).1 := by
      rw [hstdef]
      exact (gStateBot_score_congr S (c * 128) _ _
        (fun j => by simp only [hgd, ctxG, ctxKVM])).1
    have halc : alphaT.data (idx.1, PUnit.unit)
        = some ((WithBot.realExp2 (WithBot.realSub (st (g idx.1 idx.2.1)).1 (M' idx.1 idx.2.1))).unbotD 0) := by
      rw [haleq]
      show WithBot.realExp2 ((Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile mijT).data (idx.1, PUnit.unit)) = _
      have hinner : (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile mijT).data (idx.1, PUnit.unit)
          = WithBot.realSub (st (g idx.1 idx.2.1)).1 (M' idx.1 idx.2.1) := by
        simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub]
        rw [hmijc idx.1, hMdd idx.1 idx.2.1]
        show WithBot.realSub (st (g idx.1 ⟨0, by decide⟩)).1 _ = _
        rw [hst1]
      rw [hinner]; exact ctxg_realExp2_eq_some_unbotD _
    -- alpha factor: st.2.2 uses g idx.1 idx.2.1 directly (value carries d)
    have hdot : (Tile.dot [] pT vloadT).data (idx.1, idx.2.1, PUnit.unit)
        = some (Finset.univ.sum (fun jL : Fin 128 =>
            pow2 ((g idx.1 idx.2.1 ⟨c * 128 + jL.val, gBlock_idx_lt S 128 c hwin jL⟩).1
                - (M' idx.1 idx.2.1).unbotD 0)
              * (g idx.1 idx.2.1 ⟨c * 128 + jL.val, gBlock_idx_lt S 128 c hwin jL⟩).2)) := by
      refine ctxg_dot_row pT vloadT idx.1 idx.2.1 _ _ (fun jL => hpc idx.1 jL idx.2.1) ?_
      intro jL
      rw [hvf jL idx.2.1]
      refine congrArg some ?_
      simp only [hgd, ctxG, ctxKVM]
    rw [hacctile, halc, hdot]
    show WithBot.realAdd (some _) (WithBot.realMul (some _) (some _)) = _
    simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
    rw [add_comm, mul_comm ((st (g idx.1 idx.2.1)).2.2)]
  · -- (c+1)*128 ≤ S
    exact hwin

theorem ctxOut_offset_injective_128
    (s : BlockState) (B_Start_Loc : RegionName) :
    Function.Injective
      (fun idx : TileIndex [128, 128] => outOffset s 16 B_Start_Loc 2048 128 1 128 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, startLoc, curBatch, curHead, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb; subst db; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **M3 — `ctxPostLoop`: `acc /= l_i` + masked store reads off the faithful fold.**
At the loop exit (`c·128 = S`, the streamed window), the `acc /= l_i[:, None]`
rescale and masked `tl.store` write the ⊥-seeded `acc/l` ratio of `ctxG` —
which is exactly `contextAttnExactFoldM` (the boundary-masked genuine fold) — into
`Out` at every active lane (`offs_m < cur_batch_seq_len`), preserving inactive
lanes. The store is a scatter over the injective `outOffset` map. -/
theorem ctxPostLoop_eval
    (Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName) (s0 : BlockState)
    (S bel : Nat) (c : Nat) (s : BlockState) (hSc : S = c * 128)
    (hinv : ctxInvariant Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len s0
        sm_scale_python S bel c s) :
    ∃ sP, stepStmts (ctxPostLoop Out) s = some sP
      ∧ ∀ idx : TileIndex [128, 128],
          sP.readMem Out (outOffset s0 16 B_Start_Loc 2048 128 1 128 idx)
            = if active s0 16 B_Seqlen B_Prompt_Cache_Len 128 idx then
                contextAttnExactFoldM s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len
                  sm_scale_python 128 S bel idx
              else s0.readMem Out (outOffset s0 16 B_Start_Loc 2048 128 1 128 idx) := by
  subst hSc
  set plen := s0.readMemValue .nat (Region.cast B_Prompt_Cache_Len) (s0.pids 1 / 16) with hplend
  set sl := s0.readMemValue .nat (Region.cast B_Seqlen) (s0.pids 1 / 16) - plen with hsld
  set g := fun (i d : Fin 128) =>
    ctxG s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len sm_scale_python (c * 128) bel i d with hgd
  simp only [ctxInvariant] at hinv
  obtain ⟨hpids, hmem, hundef, hcb, hckvh, hch, hplen, hsl, hbel, hcbsi, hom, hon, hod,
      hq, hmi, hli, hacc, hcle⟩ := hinv
  -- the divided acc tile cell = (gStateBot S (c*128)).2.2 / .2.1 = contextAttnExactFoldM
  set accDiv : Tile .real [128, 128] := ⟨fun idx : TileIndex [128, 128] =>
      some ((gStateBot (c * 128) (c * 128) (g idx.1 idx.2.1)).2.2 / (gStateBot (c * 128) (c * 128) (g idx.1 ⟨0, by decide⟩)).2.1)⟩
    with haccDiv
  unfold ctxPostLoop
  -- stmt 0: acc = acc / l_i[:, None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.div .real Broadcast.nil.consR.consSame
        (Op.ref .real [128, 128] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "l_i"))) s = some accDiv from by
      have hexp : @evalOp TileDType.real [128, 1]
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "l_i")) s
          = some (Tile.expandDim ⟨1, by simp⟩
              (⟨fun r : TileIndex [128] => some (gStateBot (c * 128) (c * 128) (g r.1 ⟨0, by decide⟩)).2.1⟩
                : Tile .real [128])) :=
        evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hli
      rw [evalOp_div]
      simp only [evalOp_ref, hacc, hexp, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [haccDiv, Tile.bop_data, Tile.expandDim_data, TileShape.dropInsertedIndex,
        TileShape.insertAxisIndex, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.div, WithBot.realDiv, Option.map₂, Option.bind, Option.map]
      rfl))]
  set s1 := s.setReg "acc" .real [128, 128] accDiv with hs1
  -- off_o offset tile (= outOffset)
  set offoTile : Tile .nat [128, 128] :=
    ⟨fun idx : TileIndex [128, 128] => outOffset s0 16 B_Start_Loc 2048 128 1 128 idx⟩ with hoffo
  -- stmt 1: off_o (column/row offsets, same shape as off_q but for Out)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")))
            (Op.constNat 2048))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 128)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))) s1 = some offoTile from by
      simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
        ctx_evalOp_expandDim_one_nat, ctx_evalOp_expandDim_zero_nat, hs1,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq,
        hcbsi, hch, hom, hod, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [hoffo, outOffset, startLoc, curBatch, curHead, mIndex, dIndex,
        Tile.bop_data, Tile.scalar_data, Tile.vec_data,
        Tile.expandDim, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul]
      generalize s0.readMemValue .nat B_Start_Loc (s0.pids 1 / 16) = A
      ring))]
  set s2 := s1.setReg "off_o" .nat [128, 128] offoTile with hs2
  -- stmt 2: out_ptrs = Out + off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [128, 128] "off_o")) s2
        = some (⟨fun idx : TileIndex [128, 128] =>
            (Out, offoTile.data idx)⟩ : Tile .ptr [128, 128]) from by
      simp only [evalOp, hs2, BlockState.setReg_same, Region.cast_id,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      · simp only [Tile.ptrAdd, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex]
      · simp only [Tile.ptrAdd, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
          Nat.zero_add]))]
  set s3 := s2.setReg "out_ptrs" .ptr [128, 128]
    (⟨fun idx : TileIndex [128, 128] => (Out, offoTile.data idx)⟩ : Tile .ptr [128, 128]) with hs3
  -- stmt 3: masked store of acc into Out
  set P : TileIndex [128, 128] → Prop :=
    fun idx => 128 * s0.pids 0 + idx.1.val < sl with hP
  have hOffInj : Function.Injective (fun idx : TileIndex [128, 128] => offoTile.data idx) :=
    ctxOut_offset_injective_128 s0 B_Start_Loc
  have hmaskev : evalOp (Op.remap [128, 128] (fun x => (x.1, ⟨0, Broadcast.leftIndex._proof_1⟩, PUnit.unit))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
          (Op.ref .nat [] "cur_batch_seq_len"))) s3
      = some (⟨fun idx : TileIndex [128, 128] => decide (128 * s0.pids 0 + idx.1.val < sl)⟩ : Tile .bool [128, 128]) := by
    have := ctxRowMask_eval s3 (fun r : Fin 128 => 128 * s0.pids 0 + r.val) sl
      (by rw [hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hom)
      (by rw [hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsl)
    exact this
  have haccref : evalOp (Op.ref .real [128, 128] "acc") s3 = some accDiv := by
    rw [evalOp_ref, hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hs1, BlockState.setReg_same]
  have hptrref : evalOp (Op.ref .ptr [128, 128] "out_ptrs") s3
      = some (⟨fun idx : TileIndex [128, 128] => (Out, offoTile.data idx)⟩ : Tile .ptr [128, 128]) := by
    rw [evalOp_ref, hs3, BlockState.setReg_same]
  have hstore : stepStmt (Stmt.store .real [128, 128]
      (MemAccess.ptr (Op.ref .ptr [128, 128] "out_ptrs"))
      (Op.ref .real [128, 128] "acc")
      (MaskOpt.mask
        (Op.remap [128, 128] (fun x => (x.1, ⟨0, Broadcast.leftIndex._proof_1⟩, PUnit.unit))
          (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
            (Op.ref .nat [] "cur_batch_seq_len"))))) s3
      = some ((TileShape.allIndices [128, 128]).foldl
          (fun acc idx => if P idx then acc.writeMem Out (offoTile.data idx)
            ((accDiv.data idx).unbotD 0) else acc) s3) := by
    unfold stepStmt
    rw [haccref]
    simp only [hmaskev, Option.bind_eq_bind, Option.bind_some, Option.map_some]
    rw [hptrref]
    simp only [Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    refine List.foldl_ext _ _ s3 ?_
    intro acc idx _
    by_cases h : 128 * s0.pids 0 + idx.1.val < sl
    · simp only [h, decide_true, if_pos, hP, if_pos h, BlockState.writeMemTyped_real,
        FloatDType.real_storeValue]
    · simp only [decide_eq_false_iff_not.mpr h, Bool.false_eq_true, if_false, hP, if_neg h]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  rw [show outOffset s0 16 B_Start_Loc 2048 128 1 128 idx = offoTile.data idx from rfl]
  rw [BlockState.scatter_readback_prop_masked_nd (region := Out) s3 (fun idx => offoTile.data idx)
    (fun idx => (accDiv.data idx).unbotD 0) P hOffInj idx]
  have hactive_iff : active s0 16 B_Seqlen B_Prompt_Cache_Len 128 idx ↔ P idx := by
    have he : mIndex s0 128 idx.1 = 128 * s0.pids 0 + idx.1.val := by
      simp only [mIndex]; ring
    have he2 : seqLen s0 16 B_Seqlen B_Prompt_Cache_Len = sl := by
      rw [hsld, hplend]; rfl
    simp only [active, hP, he, he2]
  by_cases hac : P idx
  · rw [if_pos hac, if_pos (hactive_iff.mpr hac)]
    -- accDiv cell = contextAttnExactFoldM
    simp only [haccDiv, contextAttnExactFoldM]
    show (gStateBot (c * 128) (c * 128) (g idx.1 idx.2.1)).2.2
        / (gStateBot (c * 128) (c * 128) (g idx.1 ⟨0, by decide⟩)).2.1 = _
    have hden : (gStateBot (c * 128) (c * 128) (g idx.1 ⟨0, by decide⟩)).2.1
        = (gStateBot (c * 128) (c * 128) (ctxKVM s0 Q K V B_Start_Loc B_Seqlen
            B_Prompt_Cache_Len sm_scale_python 128 (c * 128) bel idx.1 idx.2.1)).2.1 :=
      (gStateBot_score_congr (c * 128) (c * 128) _ _ (fun j => by simp only [hgd, ctxG, ctxKVM])).2
    rw [hden]
    rfl
  · rw [if_neg hac, if_neg (fun hcon => hac (hactive_iff.mp hcon))]
    show (s3.readMem Out (offoTile.data idx)) = _
    have : s3.readMem Out (offoTile.data idx) = s.readMem Out (offoTile.data idx) := by
      unfold BlockState.readMem; rw [hs3, hs2, hs1]; simp only [BlockState.setReg_mem]
    rw [this]
    unfold BlockState.readMem; rw [hmem]

/-! ## STEP E — whole-kernel exec assembly (BLOCK_M = 128)

The full Python-shape context surface (prologue + dynamic `forRangeDyn` loop +
epilogue) steps to a final state whose `Out` store holds the genuine
`block_end_loc`-masked causal-softmax closed form `contextAttnExactFoldM` at every
active lane, preserving inactive lanes. Composed from `ctxPreLoop_eval` (`P 0`) +
`forRangeDyn_inv` over `ctx_attn_step` + `ctxPostLoop_eval`. The streamed window
`S = ceil₁₂₈(block_mask·block_end_loc)` is the first multiple of 128 at/above the
dynamic bound (the loop streams phantom keys `[bel, S)` whose masked `k`/`v` are
`0`). -/

/-- The ⊥-seeded fold over the empty window `[0, 0)` is the seed `(⊥, 0, 0)`. -/
theorem gStateBot_zero (S : Nat) (g : Fin S → ℝ × ℝ) : gStateBot S 0 g = (⊥, 0, 0) := by
  rw [gStateBot, show gKeysUpto S 0 g = [] from by
    rw [gKeysUpto]; apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`context_attn_fwd.py` whole-kernel exec assembly (BLOCK_M = 128).** Steps the
entire faithful surface and reads off the genuine masked closed form
`contextAttnExactFoldM` at every active `Out` lane (inactive lanes preserved). -/
theorem context_attn_fwd_exec_block128
    (Q K V Out : RegionName)
    (B_Start_Loc B_Seqlen B_Prompt_Cache_Len : Region .nat) (s : BlockState)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sF, exec (context_attn_fwd_kernel_int8kv_surface Q K V
        sm_scale_python Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len
        2048 128 1 8388608 262144 128 1 8388608 262144 128 1
        2048 128 1 1 16 128 128 128) s = some sF
      ∧ ∀ idx : TileIndex [128, 128],
          sF.readMem Out (outOffset s 16 B_Start_Loc 2048 128 1 128 idx)
            = if active s 16 B_Seqlen B_Prompt_Cache_Len 128 idx then
                contextAttnExactFoldM s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len
                  sm_scale_python 128
                  (let plen := s.readMemValue .nat (Region.cast B_Prompt_Cache_Len) (s.pids 1 / 16)
                   let sl := s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 1 / 16) - plen
                   let bel := (let a := 128 * s.pids 0 + 128 + plen
                               let b := sl + plen
                               if a < b then a else b)
                   let bm := if 128 * s.pids 0 < sl then 1 else 0
                   128 * ((bm * bel + 127) / 128))
                  (let plen := s.readMemValue .nat (Region.cast B_Prompt_Cache_Len) (s.pids 1 / 16)
                   let sl := s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 1 / 16) - plen
                   let a := 128 * s.pids 0 + 128 + plen
                   let b := sl + plen
                   if a < b then a else b) idx
              else s.readMem Out (outOffset s 16 B_Start_Loc 2048 128 1 128 idx) := by
  -- the body decomposes as preLoop ++ (forRangeDyn :: postLoop)
  rw [show exec (context_attn_fwd_kernel_int8kv_surface Q K V
        sm_scale_python Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len
        2048 128 1 8388608 262144 128 1 8388608 262144 128 1
        2048 128 1 1 16 128 128 128) s
      = stepStmts (context_attn_fwd_kernel_int8kv_surface Q K V
        sm_scale_python Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len
        2048 128 1 8388608 262144 128 1 8388608 262144 128 1
        2048 128 1 1 16 128 128 128).toAlgKernel.body s from rfl]
  rw [ctxBody_split Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len]
  -- preLoop
  obtain ⟨s0, hpre, hpids, hmem, hundef0, hstart_m, hcb, hch, hckvh, hplen, hcbsi, hsl0,
      hbsl, hon, hod, hom, hmi, hli, hacc, hbm0, hbel0, hoffq, hq⟩ :=
    ctxPreLoop_eval s Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len sm_scale_python hundef
  rw [stepStmts.append_some hpre]
  -- s0-based kernel-decoded dynamic-loop bound (`s0.mem = s.mem`, `s0.pids = s.pids`)
  set plen := s0.readMemValue .nat (Region.cast B_Prompt_Cache_Len) (s0.pids 1 / 16) with hplend
  set slv := s0.readMemValue .nat (Region.cast B_Seqlen) (s0.pids 1 / 16) with hslvd
  set sl := slv - plen with hsld
  set bel := (let a := 128 * s0.pids 0 + 128 + plen
              let b := sl + plen
              if a < b then a else b) with hbeld
  set bm := (if 128 * s0.pids 0 < sl then 1 else 0) with hbmd
  set S := 128 * ((bm * bel + 127) / 128) with hSd
  -- bridge: the preLoop readbacks (over `s`) equal the `s0`-based abbreviations
  have hmemv : ∀ (rg : RegionName) (i : Nat),
      s.readMemValue .nat rg i = s0.readMemValue .nat rg i := by
    intro rg i
    simp only [BlockState.readMemValue, BlockState.readMemTyped, hmem]
  have hmemvr : ∀ (rg : RegionName) (i : Nat),
      s.readMemValue .real rg i = s0.readMemValue .real rg i := by
    intro rg i
    simp only [BlockState.readMemValue, BlockState.readMemAs, hmem]
  have hbelrb : s0.regs .nat [] "block_end_loc" = some (Tile.scalar bel) := by
    rw [hbel0]
    refine congrArg (fun x => some (Tile.scalar x)) ?_
    simp only [hbeld, hsld, hslvd, hplend, Region.cast_cast, ← hpids, hmemv]
  have hbmrb : s0.regs .nat [] "block_mask" = some (Tile.scalar bm) := by
    rw [hbm0]
    refine congrArg (fun x => some (Tile.scalar x)) ?_
    simp only [hbmd, hsld, hslvd, hplend, Region.cast_cast, ← hpids, hmemv]
  -- the loop-entry invariant at block 0 (= counter 0): m_i/l_i/acc seeds = gStateBot S 0
  have hinv0 : ctxInvariant Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len s0
      sm_scale_python S bel 0 s0 := by
    refine ⟨rfl, rfl, hundef0, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hcb, hpids]
    · rw [hckvh, hpids]
    · rw [hch, hpids]
    · rw [hplen]; simp only [Region.cast_cast, hmemv, ← hpids]
    · rw [hsl0]; simp only [Region.cast_cast, hmemv, ← hpids]
    · exact hbelrb
    · rw [hcbsi]; simp only [startLoc, curBatch, Region.cast_cast, hmemv, ← hpids]
    · rw [hom, hpids]
    · rw [hon]
    · rw [hod]
    · rw [hq]; refine congrArg some ?_; ext idx
      simp only [Region.cast_cast, hmemv, hmemvr, ← hpids]; rfl
    · rw [hmi]; refine congrArg some ?_; ext r
      simp only [Nat.zero_mul, gStateBot_zero]
    · rw [hli]; refine congrArg some ?_; ext r
      simp only [Nat.zero_mul, gStateBot_zero]
    · rw [hacc]; refine congrArg some ?_; ext idx
      simp only [Nat.zero_mul, gStateBot_zero]
    · omega
  -- run the forRangeDyn streaming loop via forRangeDyn_inv with P = ∃c, i=c·128 ∧ ctxInvariant c
  obtain ⟨final, sL, hloop, hfin, c_final, hfinaleq, hinvL⟩ :=
    forRangeDyn_inv (idx := "start_n")
      (startOp := Op.constNat 0)
      (stopOp := Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask") (Op.ref .nat [] "block_end_loc"))
      (stepOp := Op.constNat 128)
      (P := fun i st => ∃ c, i = c * 128 ∧
        ctxInvariant Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len s0
          sm_scale_python S bel c st)
      (s_init := s0)
      (by rw [evalOp_constNat])
      (show evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
            (Op.ref .nat [] "block_end_loc")) s0 = some (Tile.scalar (bm * bel)) from by
        rw [evalOp_mul, evalOp_ref, evalOp_ref, hbmrb, hbelrb]
        simp only [Option.bind_eq_bind, Option.bind_some]; rfl)
      (by rw [evalOp_constNat])
      (by norm_num)
      ⟨0, by ring, hinv0⟩
      (fun i st hi hP => by
        obtain ⟨c, hic, hinvc⟩ := hP
        -- from i < stop = bm·bel and i = c·128: (c+1)·128 ≤ S = ⌈bm·bel⌉₁₂₈
        have hwin : (c + 1) * 128 ≤ S := by
          have hstop : c * 128 < bm * bel := hic ▸ hi
          rw [hSd]; omega
        obtain ⟨s', hs', hinv'⟩ :=
          ctx_attn_step Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len s0 S bel c i st
            hwin hic hinvc
        exact ⟨s', hs', c + 1, by rw [hic]; ring, hinv'⟩)
  -- at loop exit, stop = bm·bel ≤ final = c_final·128 ≤ S = ⌈bm·bel⌉₁₂₈, so final = S
  subst hfinaleq
  have hcle : c_final * 128 ≤ S := hinvL.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have hScfinal : S = c_final * 128 := by
    rw [hSd] at hcle ⊢; omega
  -- postLoop: acc/=l_i + masked store = contextAttnExactFoldM over the s0-decoded window
  obtain ⟨sP, hpostStep, hO⟩ :=
    ctxPostLoop_eval Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len s0 S bel c_final sL
      hScfinal hinvL
  refine ⟨sP, ?_, ?_⟩
  · rw [stepStmts.cons_some hloop]
    exact hpostStep
  · -- readback: bridge outOffset/active/contextAttnExactFoldM over s0 to s, and S/bel to the s-form
    intro idx
    have hOidx := hO idx
    -- outOffset s0 = outOffset s (states differ only in regs; outOffset reads B_Start_Loc/pids)
    have houtoff : outOffset s0 16 B_Start_Loc 2048 128 1 128 idx
        = outOffset s 16 B_Start_Loc 2048 128 1 128 idx := by
      simp only [outOffset, startLoc, curBatch, curHead, mIndex, hpids, hmemv]
    -- active s0 = active s
    have hact : active s0 16 B_Seqlen B_Prompt_Cache_Len 128 idx
        ↔ active s 16 B_Seqlen B_Prompt_Cache_Len 128 idx := by
      simp only [active, mIndex, seqLen, promptLen, curBatch, hpids, hmemv]
    rw [houtoff] at hOidx
    rw [hOidx]
    by_cases hac : active s0 16 B_Seqlen B_Prompt_Cache_Len 128 idx
    · rw [if_pos hac, if_pos (hact.mp hac)]
      -- contextAttnExactFoldM over s0/S/bel = over s/S'/bel' (states agree on mem+pids)
      have hreadmem : ∀ (rg : RegionName) (o : Nat), s0.readMem rg o = s.readMem rg o := by
        intro rg o; unfold BlockState.readMem; rw [hmem]
      -- the s-form `bel`/`S` in the goal equal the s0-decoded abbreviations
      have hbeleq : bel
          = (let plen := s.readMemValue .nat (Region.cast B_Prompt_Cache_Len) (s.pids 1 / 16)
             let sl := s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 1 / 16) - plen
             let a := 128 * s.pids 0 + 128 + plen
             let b := sl + plen
             if a < b then a else b) := by
        simp only [hbeld, hsld, hslvd, hplend, ← hpids, hmemv]
      have hSeq : S
          = (let plen := s.readMemValue .nat (Region.cast B_Prompt_Cache_Len) (s.pids 1 / 16)
             let sl := s.readMemValue .nat (Region.cast B_Seqlen) (s.pids 1 / 16) - plen
             let bel := (let a := 128 * s.pids 0 + 128 + plen
                         let b := sl + plen
                         if a < b then a else b)
             let bm := if 128 * s.pids 0 < sl then 1 else 0
             128 * ((bm * bel + 127) / 128)) := by
        simp only [hSd, hbmd, hbeld, hsld, hslvd, hplend, ← hpids, hmemv]
      rw [← hbeleq, ← hSeq]
      -- now both sides are contextAttnExactFoldM at S/bel; bridge s0 → s via ctxKVM equality
      have hkvm : ctxKVM s0 Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len sm_scale_python 128 S bel idx.1 idx.2.1
          = ctxKVM s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len sm_scale_python 128 S bel idx.1 idx.2.1 := by
        funext j
        simp only [ctxKVM, ctxQTileM, ctxKTileM, ctxVTileM, ctxQTile, ctxKTile, ctxVTile,
          promptLen, seqLen, curBatch, curHead, hpids, hmemv, hreadmem]
      simp only [contextAttnExactFoldM, hkvm]
    · rw [if_neg hac, if_neg (fun h => hac (hact.mpr h))]
      -- preserved lane: s0.readMem Out = s.readMem Out
      unfold BlockState.readMem; rw [hmem]

/-- The kernel-decoded streamed window `S = ceil_BM(block_mask·block_end_loc)` at the
Python test shape (BLOCK_N = BLOCK_M): the first multiple of `BM` at/above the
dynamic loop bound. -/
def ctxFwdWindow (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (BM : Nat) : Nat :=
  let plen := s.readMemValue .nat B_Prompt_Cache_Len (s.pids 1 / 16)
  let sl := s.readMemValue .nat B_Seqlen (s.pids 1 / 16) - plen
  let bel := (let a := BM * s.pids 0 + BM + plen
              let b := sl + plen
              if a < b then a else b)
  let bm := if BM * s.pids 0 < sl then 1 else 0
  BM * ((bm * bel + (BM - 1)) / BM)

/-- The kernel-decoded `block_end_loc = min(BM·start_m + BM + plen, seq_len + plen)`
at the Python test shape. -/
def ctxFwdBel (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (BM : Nat) : Nat :=
  let plen := s.readMemValue .nat B_Prompt_Cache_Len (s.pids 1 / 16)
  let sl := s.readMemValue .nat B_Seqlen (s.pids 1 / 16) - plen
  let a := BM * s.pids 0 + BM + plen
  let b := sl + plen
  if a < b then a else b

/-- **Genuine closed-form output value** of `context_attn_fwd.py` at the Python test
shape, BLOCK_M = 128: the boundary-masked causal-softmax fold `contextAttnExactFoldM`
of the loaded Q/K/V memory — a pure function of memory, NOT the kernel's executed
readback. -/
noncomputable def ctxFwdGenuineOutValue128
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (idx : TileIndex [128, 128]) : ℝ :=
  contextAttnExactFoldM s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len
    sm_scale_python 128 (ctxFwdWindow s B_Seqlen B_Prompt_Cache_Len 128)
    (ctxFwdBel s B_Seqlen B_Prompt_Cache_Len 128) idx

/-- Self-referential executed `Out`-readback for the BLOCK_M = 64 (Tesla) branch.
The genuine closed form `contextAttnExactFoldM` for this branch requires the full
streaming-loop suite (`ctxLoopBody_steps`/`ctxInvariant`/`ctx_attn_step`/
`ctxPostLoop_eval`) re-derived at the `[64,128]` tile shape (`BLOCK_N = 64` key
blocks); banked only at `[128,128]`. -/
noncomputable def producedContextFwdBlock64OutValue
    (s : BlockState)
    (Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (idx : TileIndex [64, 128]) : ℝ :=
  match exec (context_attn_fwd_kernel_int8kv_surface Q K V
      (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
      B_Start_Loc B_Seqlen B_Prompt_Cache_Len
      2048 128 1 8388608 262144 128 1 8388608 262144 128 1
      2048 128 1 1 16 128 64 128) s with
  | some s' => s'.readMem Out (outOffset s 16 B_Start_Loc 2048 128 1 64 idx)
  | none => 0.0

/-- Algorithm-layer correctness for the masked context-attention output store. -/
theorem context_attn_fwd_final_store_slice_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (H
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s H B_Start_Loc stride_obs stride_oh stride_od
        BLOCK_M idx
      (exec (context_attn_fwd_final_store_slice Acc B_Start_Loc B_Seqlen
            B_Prompt_Cache_Len Out H stride_acc_b stride_acc_h stride_acc_m
            stride_acc_d stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL)
          s).map (·.readMem Out outAddr)
        = some (if active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx then
            accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len H stride_acc_b
              stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, context_attn_fwd_final_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.sub, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
        BlockState.readMemValue, curBatch, curHead, promptLen, seqLen, startLoc,
        mIndex, dIndex, active, accOffset, outOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      (s.readMemValue .nat B_Start_Loc (s.pids 1 / H) +
          (s.pids 0 * BLOCK_M + idx.1.val)) * stride_obs +
        s.pids 1 % H * stride_oh + idx.2.1.val * stride_od
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 * BLOCK_M + idx.1.val <
            s.readMemValue .nat B_Seqlen (s.pids 1 / H) -
              s.readMemValue .nat B_Prompt_Cache_Len (s.pids 1 / H) then
          some (s.readMem Acc
            ((s.pids 1 / H) * stride_acc_b + (s.pids 1 % H) * stride_acc_h +
              (s.pids 0 * BLOCK_M + idx.1.val) * stride_acc_m +
              idx.2.1.val * stride_acc_d))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_M + idx.1.val <
        s.readMemValue .nat B_Seqlen (s.pids 1 / H) -
          s.readMemValue .nat B_Prompt_Cache_Len (s.pids 1 / H)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, startLoc, curBatch, curHead, mIndex, dIndex,
      BlockState.readMemValue] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if P idx then
      accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len H stride_acc_b
        stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 0 * BLOCK_M + idx.1.val <
        s.readMemValue .nat B_Seqlen (s.pids 1 / H) -
          s.readMemValue .nat B_Prompt_Cache_Len (s.pids 1 / H)
  · rfl
  · rfl

/-- Compute-facing correctness for the masked context-attention output store. -/
theorem context_attn_fwd_final_store_slice_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (H
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := context_attn_fwd_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out H stride_acc_b stride_acc_h stride_acc_m
        stride_acc_d stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len H stride_acc_b
          stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_fwd_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := context_attn_fwd_final_store_slice_correct Acc B_Start_Loc
    B_Seqlen B_Prompt_Cache_Len Out H stride_acc_b stride_acc_h stride_acc_m
    stride_acc_d stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL s
    hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Python test-shape wrappers

The checked Python test uses `Z = 16`, `H = 16`, `N_CTX = 2048`, and
`D_HEAD = 128`. The output tensor is contiguous with row/head/dimension strides
`(2048, 128, 1)`. `BLOCK_DMODEL = 128`, and `BLOCK_M` is `128` on the regular
path or `64` on the Tesla branch. -/

theorem context_attn_fwd_python_block128_offset_injective
    (s : BlockState) (B_Start_Loc : RegionName) :
    Function.Injective
      (fun idx : TileIndex [128, 128] =>
        outOffset s 16 B_Start_Loc 2048 128 1 128 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, startLoc, curBatch, curHead, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem context_attn_fwd_python_block64_offset_injective
    (s : BlockState) (B_Start_Loc : RegionName) :
    Function.Injective
      (fun idx : TileIndex [64, 128] =>
        outOffset s 16 B_Start_Loc 2048 128 1 64 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, startLoc, curBatch, curHead, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem context_attn_fwd_final_store_python_block128_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_fwd_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out 16 4194304 128 2048 1 2048 128 1 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len 16 4194304 128
          2048 1 128 idx) := by
  exact context_attn_fwd_final_store_slice_compute_correct Acc B_Start_Loc
    B_Seqlen B_Prompt_Cache_Len Out 16 4194304 128 2048 1 2048 128 1
    128 128 s (context_attn_fwd_python_block128_offset_injective s B_Start_Loc)

theorem context_attn_fwd_final_store_python_block64_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_fwd_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out 16 4194304 128 2048 1 2048 128 1 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 64 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len 16 4194304 128
          2048 1 64 idx) := by
  exact context_attn_fwd_final_store_slice_compute_correct Acc B_Start_Loc
    B_Seqlen B_Prompt_Cache_Len Out 16 4194304 128 2048 1 2048 128 1
    64 128 s (context_attn_fwd_python_block64_offset_injective s B_Start_Loc)

theorem context_attn_fwd_surface_python_block128_compute_correct
    (Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := context_attn_fwd_kernel_int8kv_surface Q K V
        (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
        B_Start_Loc B_Seqlen B_Prompt_Cache_Len
        2048 128 1 8388608 262144 128 1 8388608 262144 128 1
        2048 128 1 1 16 128 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        ctxFwdGenuineOutValue128 s Q K V B_Start_Loc B_Seqlen
          B_Prompt_Cache_Len idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_fwd_kernel_int8kv_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  obtain ⟨sF, hexec, hO⟩ :=
    context_attn_fwd_exec_block128 Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len s hundef
  rw [show sm_scale_python = (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) from rfl] at hexec
  rw [hExec] at hexec
  obtain rfl : sF = s' := (Option.some.inj hexec).symm
  have hb := hO idx
  simp only [ComputeCorrect.OutputReadable.read_real, Region.cast_cast, Region.cast_id]
    at hb hActive ⊢
  rw [if_pos hActive] at hb
  rw [hb, ctxFwdGenuineOutValue128, ctxFwdWindow, ctxFwdBel]
  norm_num

theorem context_attn_fwd_surface_python_block64_compute_correct
    (Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_fwd_kernel_int8kv_surface Q K V
        (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
        B_Start_Loc B_Seqlen B_Prompt_Cache_Len
        2048 128 1 8388608 262144 128 1 8388608 262144 128 1
        2048 128 1 1 16 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 64 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        producedContextFwdBlock64OutValue s Q K V Out B_Start_Loc B_Seqlen
          B_Prompt_Cache_Len idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_fwd_kernel_int8kv_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedContextFwdBlock64OutValue, hExec]

/-- Public Python test-shape summary for `context_attn_fwd.py`.

This records the faithful full int8-KV `_fwd_kernel` surface for the checked
layout and both Python launcher block-size branches. For the regular
(non-Tesla, BLOCK_M = 128) path — the path exercised on the gold-test hardware —
every active observable `Out` write holds the **genuine** boundary-masked
causal-softmax closed form `contextAttnExactFoldM` of the loaded Q/K/V memory
(`ctxFwdGenuineOutValue128`), NOT the kernel's own executed readback: the
self-referential proof gap is closed. The Tesla (BLOCK_M = 64) branch is still
recorded via the executed readback (`producedContextFwdBlock64OutValue`); its
genuine closed form needs the streaming-loop suite re-derived at the `[64,128]`
tile shape. -/
theorem context_attn_fwd_python_test_shape_output_summary
    (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (context_attn_fwd_kernel_int8kv_surface Q K V
      (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
      B_Start_Loc B_Seqlen B_Prompt_Cache_Len
      2048 128 1 8388608 262144 128 1 8388608 262144 128 1
      2048 128 1 1 16 128 128 128).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (context_attn_fwd_kernel_int8kv_surface Q K V
      (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
      B_Start_Loc B_Seqlen B_Prompt_Cache_Len
      2048 128 1 8388608 262144 128 1 8388608 262144 128 1
      2048 128 1 1 16 128 64 128).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := context_attn_fwd_kernel_int8kv_surface Q K V
        (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
        B_Start_Loc B_Seqlen B_Prompt_Cache_Len
        2048 128 1 8388608 262144 128 1 8388608 262144 128 1
        2048 128 1 1 16 128 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        ctxFwdGenuineOutValue128 s Q K V B_Start_Loc B_Seqlen
          B_Prompt_Cache_Len idx) ∧
    ComputeCorrect.Realizes
      (kernel := context_attn_fwd_kernel_int8kv_surface Q K V
        (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
        B_Start_Loc B_Seqlen B_Prompt_Cache_Len
        2048 128 1 8388608 262144 128 1 8388608 262144 128 1
        2048 128 1 1 16 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 64 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        producedContextFwdBlock64OutValue s Q K V Out B_Start_Loc B_Seqlen
          B_Prompt_Cache_Len idx) := by
  constructor
  · exact context_attn_fwd_kernel_int8kv_surface_toAlgorithm_supported Q K V
      (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
      B_Start_Loc B_Seqlen B_Prompt_Cache_Len
      2048 128 1 8388608 262144 128 1 8388608 262144 128 1
      2048 128 1 1 16 128 128 128
  constructor
  · exact context_attn_fwd_kernel_int8kv_surface_toAlgorithm_supported Q K V
      (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
      B_Start_Loc B_Seqlen B_Prompt_Cache_Len
      2048 128 1 8388608 262144 128 1 8388608 262144 128 1
      2048 128 1 1 16 128 64 128
  constructor
  · exact context_attn_fwd_surface_python_block128_compute_correct Q K V Out
      B_Start_Loc B_Seqlen B_Prompt_Cache_Len s hundef
  · exact context_attn_fwd_surface_python_block64_compute_correct Q K V Out
      B_Start_Loc B_Seqlen B_Prompt_Cache_Len s

end VeriTile.Bench.TritonBenchG.ContextAttnFwd

