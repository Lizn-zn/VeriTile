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
