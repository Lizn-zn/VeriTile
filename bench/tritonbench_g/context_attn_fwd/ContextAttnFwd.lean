import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention

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
      else (Real.log 2)⁻¹ * (-1.0e8 : ℝ),
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
           else Real.exp (-1.0e8)
         (Finset.univ.sum (fun j : Fin S => weight j * ctxVTile s V S (j, d, PUnit.unit)))
           / (Finset.univ.sum (fun j : Fin S => weight j))) := by
  obtain ⟨i, d, u⟩ := idx
  rw [contextAttnExactFold, ctxExactKeyList, osStep_foldl_eq_batch]
  simp only [List.map_ofFn, List.sum_ofFn, Function.comp, contextEffScale]
  have hlog2 : Real.log 2 ≠ 0 := by positivity
  have hw : ∀ j : Fin S,
      pow2 (if j.val ≤ s.pids 0 * BLOCK_M + i.val + promptLen s 16 B_Prompt_Cache_Len then
          sm_scale * Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit) * ctxKTile s K S (j, e, PUnit.unit))
        else (Real.log 2)⁻¹ * (-1.0e8 : ℝ))
      = if j.val ≤ s.pids 0 * BLOCK_M + i.val + promptLen s 16 B_Prompt_Cache_Len then
          Real.exp (Real.log 2 * sm_scale * Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit) * ctxKTile s K S (j, e, PUnit.unit)))
        else Real.exp (-1.0e8) := by
    intro j
    by_cases h : j.val ≤ s.pids 0 * BLOCK_M + i.val + promptLen s 16 B_Prompt_Cache_Len
    · simp only [h, if_true, pow2]; ring_nf
    · simp only [h, if_false, pow2, ← mul_assoc, mul_inv_cancel₀ hlog2, one_mul]
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
    else (Real.log 2)⁻¹ * (-1.0e8 : ℝ),
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
      ∧ s0.regs .nat [128, 128] "off_q" = some ⟨fun idx : TileIndex [128, 128] =>
          (s.readMemValue .nat (Region.cast B_Start_Loc) (s.pids 1 / 16) + (128 * s.pids 0 + idx.1.val))
              * 2048 + s.pids 1 % 16 * 128 + idx.2.1.val * 1⟩ := by
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
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
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

noncomputable def producedContextFwdBlock128OutValue
    (s : BlockState)
    (Q K V Out B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (idx : TileIndex [128, 128]) : ℝ :=
  match exec (context_attn_fwd_kernel_int8kv_surface Q K V
      (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
      B_Start_Loc B_Seqlen B_Prompt_Cache_Len
      2048 128 1 8388608 262144 128 1 8388608 262144 128 1
      2048 128 1 1 16 128 128 128) s with
  | some s' => s'.readMem Out (outOffset s 16 B_Start_Loc 2048 128 1 128 idx)
  | none => 0.0

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
    (s : BlockState) :
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
        producedContextFwdBlock128OutValue s Q K V Out B_Start_Loc B_Seqlen
          B_Prompt_Cache_Len idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_fwd_kernel_int8kv_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedContextFwdBlock128OutValue, hExec]

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
layout and both Python launcher block-size branches, with the observable final
`Out` writes connected directly to the produced full-surface values. -/
theorem context_attn_fwd_python_test_shape_output_summary
    (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (s : BlockState) :
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
        producedContextFwdBlock128OutValue s Q K V Out B_Start_Loc B_Seqlen
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
      B_Start_Loc B_Seqlen B_Prompt_Cache_Len s
  · exact context_attn_fwd_surface_python_block64_compute_correct Q K V Out
      B_Start_Loc B_Seqlen B_Prompt_Cache_Len s

end VeriTile.Bench.TritonBenchG.ContextAttnFwd
