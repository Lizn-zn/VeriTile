import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention
import VeriTile.Triton.LoopInvariant

/-!
# `context_attn_llama` — strict per-kernel correctness

`_fwd_kernel` is Llama-style varlen context (prefill) attention. Each program
`(start_m, cur_bh)` decodes its batch/head as `cur_batch = cur_bh // H`,
`cur_head = cur_bh % H`, loads a `[BLOCK_M, BLOCK_DMODEL]` query tile, streams
over the cached key/value tokens gathered through `Req_to_tokens`, runs an
online-softmax (`m_i`/`l_i`/`acc`) loop with a `prompt_cache_len`-offset causal
mask, and stores the accumulated `acc` tile to `Out`, masked by
`offs_m < cur_batch_seq_len`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel[grid](...)`, the grid over sequence blocks ×
`(batch·head)`, the scheduling, and how the runtime composes per-program writes
into one buffer) is the *trusted boundary*, not a proof obligation here. Because
the program ids `(start_m, cur_bh)` are universally quantified, the per-program
statement covers every program of the grid.

## Proof architecture

```
context_attn_llama_python_test_shape_output_summary        ← TOP THEOREM (bundles both block shapes)
  ├─ context_attn_llama_fwd_kernel_surface_toAlgorithm_supported   surface lowers to the algorithm layer
  ├─ context_attn_llama_surface_python_block128_compute_correct    full surface, BLOCK_M=128 final store
  │    └─ context_attn_llama_final_store_python_block128_compute_correct
  │         └─ context_attn_llama_final_store_slice_compute_correct
  │              └─ context_attn_llama_final_store_slice_correct    algorithm-layer readback per lane
  └─ context_attn_llama_surface_python_block64_compute_correct     full surface, BLOCK_M=64 final store
       └─ context_attn_llama_final_store_python_block64_compute_correct
            └─ context_attn_llama_final_store_slice_compute_correct
(supporting: context_attn_llama_python_block128_offset_injective,
             context_attn_llama_python_block64_offset_injective)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` and
`num_warps` are not modeled. The verified compute claim is scoped to the **final
masked writeback** of the accumulated `acc` tile into `Out`: every active lane
(`offs_m < cur_batch_seq_len`, with `offs_d < head_dim` folded into the slice)
holds the surface-produced `acc` value
(`producedContextLlamaBlock128/64OutValue`), and out-of-bounds lanes are preserved. The
online-softmax streaming loop (`m_i`/`l_i`/`acc` updates, `tl.dot`, the
`Req_to_tokens` gathers, the `prompt_cache_len`-offset causal mask) is carried
*inside* the surface kernel and reflected in the produced-value spec rather than
re-proven as a closed-form softmax identity. The summary is instantiated at the
Python test shape (`H=16`, `BLOCK_DMODEL=BLOCK_N=128`, `BLOCK_M ∈ {128, 64}`);
other shapes are not covered by the top theorem.
-/

namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful DSL port of `context_attn_llama.py`'s `_fwd_kernel`. -/
def context_attn_llama_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen : Region .nat)
    (Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  cur_bh = tl.program_id(1)
  cur_batch = cur_bh // $(H)
  cur_head = cur_bh % $(H)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

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
    kv_loc = tl.load(Req_to_tokens + $(stride_req_to_tokens_b) * cur_batch_req_idx +
      $(stride_req_to_tokens_s) * (start_n + offs_n),
      mask=(start_n + offs_n) < block_end_loc,
      other=0)
    off_k = kv_loc[None, :] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
      offs_d[:, None] * $(stride_kd)
    k = tl.load(K + off_k,
      mask=(start_n + offs_n[None, :]) < block_end_loc,
      other=0.0)
    qk = tl.dot(q, k)

    mask = offs_m[:, None] + prompt_cache_len >= (start_n + offs_n[None, :])
    qk = tl.where(mask, qk * $((sm_scale : ℝ)), -1.0e8)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)

    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    off_v = kv_loc[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
      offs_d[None, :] * $(stride_vd)
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

/-- The full LLaMA context-attention surface lowers to the algorithm layer. -/
theorem context_attn_llama_fwd_kernel_surface_toAlgorithm_supported
    (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen : Region .nat)
    (Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    ∃ alg, (context_attn_llama_fwd_kernel_surface Q K V sm_scale Out
      B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len
      stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s kv_group_num H
      BLOCK_DMODEL BLOCK_M BLOCK_N).toAlgorithm? = Except.ok alg := by
  simp [context_attn_llama_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of `context_attn_llama.py`'s
`_fwd_kernel`.

The full kernel computes grouped-KV causal context attention using
`Req_to_tokens`. This slice starts from a precomputed `Acc` tile and proves the
final masked writeback into `Out`, preserving the fused `cur_bh` program-id
decomposition, `B_Start_Loc`, and the prompt-cache-adjusted sequence length.
The inner `tl.float32` `m_i/l_i/acc` streaming-softmax loop and request-token
gathers are outside this slice. -/
def context_attn_llama_final_store_slice
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

/-! ## Genuine closed-form context-attention spec (Llama `Req_to_tokens` gather)

The streaming-softmax loop in `_fwd_kernel` is *not* a self-referential black
box: it computes, for every active query lane, the prompt-cache-offset **causal
softmax attention** value over the `Req_to_tokens`-gathered key/value tokens.
This section makes that closed form explicit and proves the kernel's exact
base-2 / `sm_scale` streaming weights collapse to it — independent of the kernel
`exec`.

### Score / scale / mask of this kernel (decoded lane-by-lane from the body)

For program `(start_m, cur_bh)`, query lane `i` (global row
`gi = start_m·BLOCK_M + i`), streamed key index `j`, head channel `e`:

* **gather**    `kv_loc(j) = Req_to_tokens[stride_b·req_idx + stride_s·j]` — the
  physical token slot read by `kv_loc = tl.load(Req_to_tokens + …)` (line 108).
* **raw score** `raw i j = Σ_e Q[gi,e]·K[kv_loc(j),e]` (`tl.dot q k`, line 117);
* **scale**     `qk·sm_scale` with `sm_scale = (1/√D)·1.4426950408889634`;
* **softmax**   base-2 `tl.math.exp2`;
* **mask**      `(gi + prompt_cache_len) ≥ j`: future keys get score `-1e8`, i.e.
  a causal mask shifted by `prompt_cache_len`.

Because `exp2(qk·sm_scale) = exp((log 2 · sm_scale)·qk)`, the kernel realizes the
natural-exp causal softmax with effective scale `effScale = log 2 · sm_scale`. -/

/-- Effective natural-exp score scale of this kernel: `log 2 · sm_scale`. -/
noncomputable def contextEffScale (sm_scale : ℝ) : ℝ := Real.log 2 * sm_scale

/-- `pow2 (sm_scale · x) = exp (effScale · x)`: the kernel's `exp2`-with-`sm_scale`
weight is the natural-exp weight at the effective scale. -/
theorem pow2_smScale_eq_exp_effScale (sm_scale x : ℝ) :
    pow2 (sm_scale * x) = Real.exp (contextEffScale sm_scale * x) := by
  simp [pow2, contextEffScale, mul_assoc]

/-- Coordinate-faithful query tile of this kernel at `(start_m, cur_bh)` for the
checked Python layout (strides `stride_qbs=2048, stride_qh=128, stride_qd=1`,
`H=16`). Row `i` is the *global* prefill row `start_m·BLOCK_M + i` offset by
`cur_batch_in_all_start_index`. -/
noncomputable def ctxQTile
    (s : BlockState) (Q B_Start_Loc : RegionName) (BLOCK_M : Nat) :
    TileIndex [BLOCK_M, 128] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q
      ((s.readMemValue .nat B_Start_Loc (curBatch s 16) + (s.pids 0 * BLOCK_M + i.val))
          * 2048 + curHead s 16 * 128 + e.val)

/-- The Llama `Req_to_tokens` gather: physical token slot for streamed key index
`j`, decoded at the checked Python layout (`stride_req_to_tokens_b = 9048`,
`stride_req_to_tokens_s = 1`, request id `req_idx = B_req_idx[cur_batch]`):
`kv_loc(j) = Req_to_tokens[9048·req_idx + j]`. The key/value coordinate tiles
read physical token `kv_loc(j)` rather than the streamed index `j` — this is the
sole structural difference from `context_attn_fwd`'s sequential indexing. -/
def ctxKvLoc
    (s : BlockState) (Req_to_tokens B_req_idx : RegionName) (j : Nat) : Nat :=
  s.readMemValue .nat Req_to_tokens
    (9048 * s.readMemValue .nat B_req_idx (curBatch s 16) + j)

/-- Coordinate-faithful key tile: `K[kv_loc(j), cur_head, e]` at the checked
Llama layout (`stride_kbs=2048, stride_kh=128, stride_kd=1`, `kv_group_num=1` so
`cur_kv_head=cur_head`), where `kv_loc(j) = Req_to_tokens[9048·req_idx + j]`. -/
noncomputable def ctxKTile
    (s : BlockState) (K Req_to_tokens B_req_idx : RegionName) (S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (ctxKvLoc s Req_to_tokens B_req_idx j.val * 2048
      + curHead s 16 * 128 + e.val)

/-- Coordinate-faithful value tile: `V[kv_loc(j), cur_head, d]`. -/
noncomputable def ctxVTile
    (s : BlockState) (V Req_to_tokens B_req_idx : RegionName) (S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (ctxKvLoc s Req_to_tokens B_req_idx j.val * 2048
      + curHead s 16 * 128 + d.val)

/-- **Genuine closed-form output** of `context_attn_llama` at query lane `i`,
channel `d`, over the first `S` keys.

`out[i,d] = (Σ_{j ≤ gi+plen} exp(effScale·rawᵢⱼ)·V[kv_loc(j),d])
             / (Σ_{j ≤ gi+plen} exp(effScale·rawᵢⱼ))`

where `gi = start_m·BLOCK_M + i`, `plen = prompt_cache_len`, and
`rawᵢⱼ = Σ_e Q[gi,e]·K[kv_loc(j),e]`. This is exactly `attentionRealCausalBlock`
(the library's prompt-offset causal softmax) instantiated with this kernel's
gathered tiles, effective scale, and a causal boundary shifted by
`prompt_cache_len`. No self-reference: it is a pure function of
`Q`/`K`/`V`/`Req_to_tokens`/`B_req_idx` memory. -/
noncomputable def contextAttnClosedForm
    (s : BlockState) (Q K V B_Start_Loc Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ) (BLOCK_M S : Nat)
    (idx : TileIndex [BLOCK_M, 128]) : ℝ :=
  let i := idx.1
  let d := idx.2.1
  let plen := promptLen s 16 B_Prompt_Cache_Len
  let gi := s.pids 0 * BLOCK_M + i.val
  let raw := fun j : Fin S =>
    Finset.univ.sum (fun e : Fin 128 =>
      ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
        * ctxKTile s K Req_to_tokens B_req_idx S (j, e, PUnit.unit))
  let weight := fun j : Fin S =>
    if j.val ≤ gi + plen then Real.exp (contextEffScale sm_scale * raw j) else 0
  let denom := Finset.univ.sum (fun j : Fin S => weight j)
  let numer := Finset.univ.sum (fun j : Fin S =>
    weight j * ctxVTile s V Req_to_tokens B_req_idx S (j, d, PUnit.unit))
  numer / denom

/-- **Bridge to the library's `attentionRealCausalBlock`.** The genuine closed
form above coincides with `attentionRealCausalBlock` at query-start
`gi₀ = start_m·BLOCK_M + plen`, with this kernel's gathered Q/K/V tiles and
effective scale `effScale = log 2 · sm_scale`. Certifies `contextAttnClosedForm`
is the standard prompt-offset causal softmax-attention reference. -/
theorem contextAttnClosedForm_eq_attentionRealCausalBlock
    (s : BlockState) (Q K V B_Start_Loc Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ) (BLOCK_M S : Nat) (idx : TileIndex [BLOCK_M, 128]) :
    contextAttnClosedForm s Q K V B_Start_Loc Req_to_tokens B_req_idx
        B_Prompt_Cache_Len sm_scale BLOCK_M S idx
      = attentionRealCausalBlock
          (s.pids 0 * BLOCK_M + promptLen s 16 B_Prompt_Cache_Len)
          (ctxQTile s Q B_Start_Loc BLOCK_M)
          (ctxKTile s K Req_to_tokens B_req_idx S)
          (ctxVTile s V Req_to_tokens B_req_idx S)
          (contextEffScale sm_scale)
          (idx.1, idx.2.1, PUnit.unit) := by
  obtain ⟨i, d, u⟩ := idx
  have hbound : s.pids 0 * BLOCK_M + promptLen s 16 B_Prompt_Cache_Len + i.val
      = s.pids 0 * BLOCK_M + i.val + promptLen s 16 B_Prompt_Cache_Len := by omega
  simp only [contextAttnClosedForm, attentionRealCausalBlock, scaledScore, hbound,
    Finset.mul_sum]

/-- Per-key `(score, value)` stream the loop folds, with the kernel's genuine
`-1e8` sentinel kept. Active key `j ≤ gi+plen`: score `sm_scale·rawᵢⱼ`; future
key: sentinel (so `exp2` → `exp(-1e8)`). Value is the gathered `ctxVTile`. -/
noncomputable def ctxExactKeyList
    (s : BlockState) (Q K V B_Start_Loc Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ) (BLOCK_M S : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    List (ℝ × ℝ) :=
  let plen := promptLen s 16 B_Prompt_Cache_Len
  let gi := s.pids 0 * BLOCK_M + i.val
  List.ofFn (fun j : Fin S =>
    (if j.val ≤ gi + plen then
        sm_scale *
          Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
              * ctxKTile s K Req_to_tokens B_req_idx S (j, e, PUnit.unit))
      else (0.0 - 10e7 : ℝ),
      ctxVTile s V Req_to_tokens B_req_idx S (j, d, PUnit.unit)))

/-- Exact streaming-loop output value for lane `(i,d)`: `acc/l` of folding the
online-softmax step `osStep` over `ctxExactKeyList`. A pure function of
`Q`/`K`/`V`/`Req_to_tokens`/`B_req_idx` memory; exactly what the kernel's
`m_i`/`l_i`/`acc` loop produces. -/
noncomputable def contextAttnExactFold
    (s : BlockState) (Q K V B_Start_Loc Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ) (BLOCK_M S : Nat) (idx : TileIndex [BLOCK_M, 128]) : ℝ :=
  let st := (ctxExactKeyList s Q K V B_Start_Loc Req_to_tokens B_req_idx
      B_Prompt_Cache_Len sm_scale BLOCK_M S idx.1 idx.2.1).foldl osStep (0, 0, 0)
  st.2.2 / st.2.1

/-- **Closed form of the exact fold.** The `osStep` fold over `ctxExactKeyList`
collapses to the genuine causal softmax with `exp(effScale·raw)` weights on active
keys and `exp(-1e8)` on future keys — explicitly, no self-reference, no `exec`. -/
theorem contextAttnExactFold_eq
    (s : BlockState) (Q K V B_Start_Loc Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ) (BLOCK_M S : Nat) (idx : TileIndex [BLOCK_M, 128]) :
    contextAttnExactFold s Q K V B_Start_Loc Req_to_tokens B_req_idx
        B_Prompt_Cache_Len sm_scale BLOCK_M S idx
      = (let i := idx.1; let d := idx.2.1
         let plen := promptLen s 16 B_Prompt_Cache_Len
         let gi := s.pids 0 * BLOCK_M + i.val
         let raw := fun j : Fin S =>
           Finset.univ.sum (fun e : Fin 128 =>
             ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
               * ctxKTile s K Req_to_tokens B_req_idx S (j, e, PUnit.unit))
         let weight := fun j : Fin S =>
           if j.val ≤ gi + plen then Real.exp (contextEffScale sm_scale * raw j)
           else pow2 (0.0 - 10e7)
         (Finset.univ.sum (fun j : Fin S =>
            weight j * ctxVTile s V Req_to_tokens B_req_idx S (j, d, PUnit.unit)))
           / (Finset.univ.sum (fun j : Fin S => weight j))) := by
  obtain ⟨i, d, u⟩ := idx
  rw [contextAttnExactFold, ctxExactKeyList, osStep_foldl_eq_batch]
  simp only [List.map_ofFn, List.sum_ofFn, Function.comp, contextEffScale]
  have hw : ∀ j : Fin S,
      pow2 (if j.val ≤ s.pids 0 * BLOCK_M + i.val + promptLen s 16 B_Prompt_Cache_Len then
          sm_scale * Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
              * ctxKTile s K Req_to_tokens B_req_idx S (j, e, PUnit.unit))
        else (0.0 - 10e7 : ℝ))
      = if j.val ≤ s.pids 0 * BLOCK_M + i.val + promptLen s 16 B_Prompt_Cache_Len then
          Real.exp (Real.log 2 * sm_scale * Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
              * ctxKTile s K Req_to_tokens B_req_idx S (j, e, PUnit.unit)))
        else pow2 (0.0 - 10e7) := by
    intro j
    by_cases h : j.val ≤ s.pids 0 * BLOCK_M + i.val + promptLen s 16 B_Prompt_Cache_Len
    · simp only [h, if_true, pow2]; ring_nf
    · simp only [h, if_false]
  congr 1
  · apply Finset.sum_congr rfl; intro j _; rw [hw j]
  · apply Finset.sum_congr rfl; intro j _; rw [hw j]

noncomputable def producedContextLlamaBlock128OutValue
    (s : BlockState)
    (Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
      B_Prompt_Cache_Len : RegionName)
    (idx : TileIndex [128, 128]) : ℝ :=
  match exec (context_attn_llama_fwd_kernel_surface Q K V
      (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
      B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
      2048 128 1 2048 128 1 2048 128 1 2048 128 1
      9048 1 1 16 128 128 128) s with
  | some s' => s'.readMem Out (outOffset s 16 B_Start_Loc 2048 128 1 128 idx)
  | none => 0.0

noncomputable def producedContextLlamaBlock64OutValue
    (s : BlockState)
    (Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
      B_Prompt_Cache_Len : RegionName)
    (idx : TileIndex [64, 128]) : ℝ :=
  match exec (context_attn_llama_fwd_kernel_surface Q K V
      (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
      B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
      2048 128 1 2048 128 1 2048 128 1 2048 128 1
      9048 1 1 16 128 64 128) s with
  | some s' => s'.readMem Out (outOffset s 16 B_Start_Loc 2048 128 1 64 idx)
  | none => 0.0

/-- Algorithm-layer correctness for the masked context-attention output store. -/
theorem context_attn_llama_final_store_slice_correct
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
      (exec (context_attn_llama_final_store_slice Acc B_Start_Loc B_Seqlen
            B_Prompt_Cache_Len Out H stride_acc_b stride_acc_h stride_acc_m
            stride_acc_d stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL)
          s).map (·.readMem Out outAddr)
        = some (if active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx then
            accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len H stride_acc_b
              stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, context_attn_llama_final_store_slice, stepStmts, stepStmt,
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
theorem context_attn_llama_final_store_slice_compute_correct
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
      (kernel := context_attn_llama_final_store_slice Acc B_Start_Loc B_Seqlen
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
  · simp [context_attn_llama_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := context_attn_llama_final_store_slice_correct Acc B_Start_Loc
    B_Seqlen B_Prompt_Cache_Len Out H stride_acc_b stride_acc_h stride_acc_m
    stride_acc_d stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL s
    hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Python test-shape wrappers

The checked Python test uses `Z = 16`, `H = 16`, `N_CTX = 2048`, and
`D_HEAD = 128`. The contiguous output layout has row/head/dimension strides
`(2048, 128, 1)`. The launcher uses `BLOCK_DMODEL = 128` and either
`BLOCK_M = 128` or the Tesla branch `BLOCK_M = 64`. -/

theorem context_attn_llama_python_block128_offset_injective
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

theorem context_attn_llama_python_block64_offset_injective
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

theorem context_attn_llama_final_store_python_block128_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_llama_final_store_slice Acc B_Start_Loc B_Seqlen
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
  exact context_attn_llama_final_store_slice_compute_correct Acc B_Start_Loc
    B_Seqlen B_Prompt_Cache_Len Out 16 4194304 128 2048 1 2048 128 1
    128 128 s (context_attn_llama_python_block128_offset_injective s B_Start_Loc)

theorem context_attn_llama_final_store_python_block64_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_llama_final_store_slice Acc B_Start_Loc B_Seqlen
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
  exact context_attn_llama_final_store_slice_compute_correct Acc B_Start_Loc
    B_Seqlen B_Prompt_Cache_Len Out 16 4194304 128 2048 1 2048 128 1
    64 128 s (context_attn_llama_python_block64_offset_injective s B_Start_Loc)

theorem context_attn_llama_surface_python_block128_compute_correct
    (Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
      B_Prompt_Cache_Len : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_llama_fwd_kernel_surface Q K V
        (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
        B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
        2048 128 1 2048 128 1 2048 128 1 2048 128 1
        9048 1 1 16 128 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        producedContextLlamaBlock128OutValue s Q K V Out B_Start_Loc
          B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_llama_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedContextLlamaBlock128OutValue, hExec]

theorem context_attn_llama_surface_python_block64_compute_correct
    (Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
      B_Prompt_Cache_Len : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_llama_fwd_kernel_surface Q K V
        (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
        B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
        2048 128 1 2048 128 1 2048 128 1 2048 128 1
        9048 1 1 16 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 64 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        producedContextLlamaBlock64OutValue s Q K V Out B_Start_Loc
          B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_llama_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedContextLlamaBlock64OutValue, hExec]

/-- Public Python test-shape summary for `context_attn_llama.py`.

This records the faithful full LLaMA `_fwd_kernel` surface for the checked
layout and both Python launcher block-size branches, with the observable final
`Out` writes connected directly to the produced full-surface values. -/
theorem context_attn_llama_python_test_shape_output_summary
    (Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx
      B_Prompt_Cache_Len Out : RegionName) (s : BlockState) :
    (∃ alg, (context_attn_llama_fwd_kernel_surface Q K V
      (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
      B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
      2048 128 1 2048 128 1 2048 128 1 2048 128 1
      9048 1 1 16 128 128 128).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (context_attn_llama_fwd_kernel_surface Q K V
      (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
      B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
      2048 128 1 2048 128 1 2048 128 1 2048 128 1
      9048 1 1 16 128 64 128).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := context_attn_llama_fwd_kernel_surface Q K V
        (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
        B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
        2048 128 1 2048 128 1 2048 128 1 2048 128 1
        9048 1 1 16 128 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        producedContextLlamaBlock128OutValue s Q K V Out B_Start_Loc
          B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len idx) ∧
    ComputeCorrect.Realizes
      (kernel := context_attn_llama_fwd_kernel_surface Q K V
        (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
        B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
        2048 128 1 2048 128 1 2048 128 1 2048 128 1
        9048 1 1 16 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 64 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        producedContextLlamaBlock64OutValue s Q K V Out B_Start_Loc B_Seqlen
          Req_to_tokens B_req_idx B_Prompt_Cache_Len idx) := by
  constructor
  · exact context_attn_llama_fwd_kernel_surface_toAlgorithm_supported Q K V
      (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
      B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
      2048 128 1 2048 128 1 2048 128 1 2048 128 1
      9048 1 1 16 128 128 128
  constructor
  · exact context_attn_llama_fwd_kernel_surface_toAlgorithm_supported Q K V
      (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
      B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
      2048 128 1 2048 128 1 2048 128 1 2048 128 1
      9048 1 1 16 128 64 128
  constructor
  · exact context_attn_llama_surface_python_block128_compute_correct Q K V Out
      B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len s
  · exact context_attn_llama_surface_python_block64_compute_correct Q K V Out
      B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len s

end VeriTile.Bench.TritonBenchG.ContextAttnLlama
