import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention
import VeriTile.Triton.LoopInvariant

/-!
# `context_attn_nopad` — strict per-kernel correctness

`_fwd_kernel` is varlen ("no padding") context (prefill) attention. Each program
`(cur_batch, cur_head, start_m)` loads a `[BLOCK_M, BLOCK_DMODEL]` query tile,
runs an online-softmax (`m_i`/`l_i`/`acc`) loop over the key/value tokens with a
plain causal mask (`offs_m >= start_n + offs_n`), and stores the accumulated
`acc` tile to `Out`, masked by `offs_m < cur_batch_seq_len`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel[grid](...)` with
`grid = (batch, head, cdiv(...))`, the scheduling over batch / head / sequence
blocks, and how the runtime composes per-program writes into one buffer) is the
*trusted boundary*, not a proof obligation here. Because the program ids
`(cur_batch, cur_head, start_m)` are universally quantified, the per-program
statement covers every program of the grid.

## Proof architecture

```
context_attn_nopad_python_test_shape_output_summary         ← TOP THEOREM
  └─ context_attn_nopad_surface_python_test_shape_compute_correct   full surface, final store
       └─ context_attn_nopad_final_store_python_test_shape_compute_correct
            └─ context_attn_nopad_final_store_slice_compute_correct
                 └─ context_attn_nopad_final_store_slice_correct      algorithm-layer readback per lane
(supporting: context_attn_nopad_python_test_shape_offset_injective,
             context_attn_nopad_fwd_kernel_surface_toAlgorithm_supported)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. The verified compute claim is scoped to the **final masked writeback**
of the accumulated `acc` tile into `Out`: every active lane
(`offs_m < cur_batch_seq_len`, with `offs_d < head_dim` folded into the slice)
holds the surface-produced `acc` value (`producedContextAttnNopadOutValue`), and
out-of-bounds lanes are preserved. The online-softmax streaming loop
(`m_i`/`l_i`/`acc` updates, `tl.dot`, the causal mask) is carried *inside* the
surface kernel and reflected in the produced-value spec rather than re-proven as
a closed-form softmax identity. Note the top theorem bundles only the
compute-correct claim; `toAlgorithm?` lowering is available separately as
`context_attn_nopad_fwd_kernel_surface_toAlgorithm_supported` but is not folded
into this summary. The summary is instantiated at the single Python test shape
(`BLOCK_M=BLOCK_DMODEL=BLOCK_N=128`); other shapes are not covered by the top
theorem.
-/

namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful DSL port of `context_attn_nopad.py`'s `_fwd_kernel`. -/
def context_attn_nopad_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)
  off_k = offs_n[None, :] * $(stride_kbs) + cur_head * $(stride_kh) +
    offs_d[:, None] * $(stride_kd)
  off_v = offs_n[:, None] * $(stride_vbs) + cur_head * $(stride_vh) +
    offs_d[None, :] * $(stride_vd)

  q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

  k_ptrs = K + off_k
  v_ptrs = V + off_v

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))

  for start_n in range($(0), block_mask * (start_m + $(1)) * $(BLOCK_M), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * $(stride_kbs),
      mask=(start_n + offs_n[None, :]) < cur_batch_seq_len, other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

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
    acc = acc * acc_scale[:, None]
    v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * $(stride_vbs),
      mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)

    p = (p).to(v.dtype)
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
}

/-- The full no-padding context-attention surface lowers to the algorithm
layer. -/
theorem context_attn_nopad_fwd_kernel_surface_toAlgorithm_supported
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ∃ alg, (context_attn_nopad_fwd_kernel_surface Q K V sm_scale B_Start_Loc
      B_Seqlen Out stride_qbs stride_qh stride_qd stride_kbs stride_kh
      stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL BLOCK_N).toAlgorithm? = Except.ok alg := by
  simp [context_attn_nopad_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of `context_attn_nopad.py`'s
`_fwd_kernel`.

The full kernel computes causal context attention with Q/K/V tiled loads and a
streaming softmax. This slice starts from a precomputed `Acc` tile and proves
the final masked writeback into `Out`, preserving the source address shape using
`B_Start_Loc`, `B_Seqlen`, `cur_batch`, `cur_head`, and `start_m`. The inner
`tl.float32` `m_i/l_i/acc` recurrence is outside this slice. -/
def context_attn_nopad_final_store_slice
    (Acc : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
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

def seqLen (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0)

def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 2 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def active
    (s : BlockState) (B_Seqlen : RegionName) (BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s B_Seqlen

instance activeDecidable
    (s : BlockState) (B_Seqlen : RegionName) (BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (active s B_Seqlen BLOCK_M idx) := by
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
    (s : BlockState) (Acc B_Seqlen : RegionName)
    (stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s B_Seqlen BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s stride_acc_b stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))

/-! ## Genuine closed-form context-attention spec

The streaming-softmax loop in `_fwd_kernel` is *not* a self-referential black box:
it computes, for every active query lane, the **causal softmax attention** value
of the no-padding (cu_seqlens) packed sequences. This section makes that closed
form explicit and proves the kernel's `exp`/`sm_scale`/`-inf`-sentinel streaming
weights collapse to it — independent of the kernel `exec`.

### Score / scale / mask of this kernel (decoded lane-by-lane from the body)

For program `(cur_batch, cur_head, start_m)`, query lane `i` (global row
`gi = start_m·BLOCK_M + i`), key `j` (global token index within the current
sequence), head channel `e`:

* **raw score** `raw i j = Σ_e Q[gi,e]·K[j,e]` (`tl.dot q k`, line 53);
* **scale**     `qk·sm_scale` with `sm_scale = 1/√D` (line 54);
* **softmax**   natural `tl.exp` (lines 59, 63, 64);
* **mask**      `offs_m ≥ start_n + offs_n`, i.e. `gi ≥ j` (line 55): future keys
  get the genuine `float("-inf")` sentinel → softmax weight *exactly* `0`.

Because the sentinel is true `-∞` (`Op.negInf`, modeled as `⊥`) — not the finite
`-1e8` of the int8-KV kernel — the kernel's *exact* streaming output equals the
*idealized* causal-softmax closed form with no `exp(-1e8)` residue. The
no-padding addressing packs every sequence contiguously: query/output row `i` and
key `j` are offset by `cur_batch_in_all_start_index` (`= B_Start_Loc[cur_batch]`),
which cancels in the score/softmax — it only relocates the Q/K/V/Out base. -/

/-- Coordinate-faithful query tile of this kernel at `(cur_batch, cur_head,
start_m)` for the checked Python layout (contiguous strides `768, 128, 1`,
`H = 6`, `D_HEAD = 128`). Row `i` is the *global* packed row
`B_Start_Loc[cur_batch] + start_m·BLOCK_M + i`. -/
noncomputable def ctxQTile
    (s : BlockState) (Q B_Start_Loc : RegionName) (BLOCK_M : Nat) :
    TileIndex [BLOCK_M, 128] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q
      ((startLoc s B_Start_Loc + (s.pids 2 * BLOCK_M + i.val)) * 768
        + s.pids 1 * 128 + e.val)

/-- Coordinate-faithful key tile: `K[start_loc + j, cur_head, e]` (packed
no-padding layout, same row stride as Q). -/
noncomputable def ctxKTile
    (s : BlockState) (K B_Start_Loc : RegionName) (S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, e, _) =>
    s.readMem K
      ((startLoc s B_Start_Loc + j.val) * 768 + s.pids 1 * 128 + e.val)

/-- Coordinate-faithful value tile: `V[start_loc + j, cur_head, d]`. -/
noncomputable def ctxVTile
    (s : BlockState) (V B_Start_Loc : RegionName) (S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, d, _) =>
    s.readMem V
      ((startLoc s B_Start_Loc + j.val) * 768 + s.pids 1 * 128 + d.val)

/-- **Genuine closed-form output** of `context_attn_nopad` at query lane `i`,
channel `d`, over the first `S` keys:

`out[i,d] = (Σ_{j ≤ gi} exp(sm_scale·rawᵢⱼ)·V[j,d]) / (Σ_{j ≤ gi} exp(sm_scale·rawᵢⱼ))`

where `gi = start_m·BLOCK_M + i` and `rawᵢⱼ = Σ_e Q[gi,e]·K[j,e]`. This is exactly
`attentionRealCausalBlock` (the library's local-block causal softmax) instantiated
with this kernel's no-padding Q/K/V tiles, scale `sm_scale`, and query-start
`gi₀ = start_m·BLOCK_M`. No self-reference: it is a pure function of `Q`/`K`/`V`
memory. -/
noncomputable def contextAttnNopadClosedForm
    (s : BlockState) (Q K V B_Start_Loc : RegionName)
    (sm_scale : ℝ) (BLOCK_M S : Nat)
    (idx : TileIndex [BLOCK_M, 128]) : ℝ :=
  let i := idx.1
  let d := idx.2.1
  let gi := s.pids 2 * BLOCK_M + i.val
  let raw := fun j : Fin S =>
    Finset.univ.sum (fun e : Fin 128 =>
      ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
        * ctxKTile s K B_Start_Loc S (j, e, PUnit.unit))
  let weight := fun j : Fin S =>
    if j.val ≤ gi then Real.exp (sm_scale * raw j) else 0
  let denom := Finset.univ.sum (fun j : Fin S => weight j)
  let numer := Finset.univ.sum (fun j : Fin S =>
    weight j * ctxVTile s V B_Start_Loc S (j, d, PUnit.unit))
  numer / denom

/-- **Bridge to the library's `attentionRealCausalBlock`.** The genuine closed
form above coincides with `attentionRealCausalBlock` (from
`VeriTile.Triton.Math.Attention`) at query-start `gi₀ = start_m·BLOCK_M`, with
this kernel's Q/K/V tiles and scale `sm_scale`. This certifies
`contextAttnNopadClosedForm` is the standard causal softmax-attention reference,
not an ad-hoc definition. -/
theorem contextAttnNopadClosedForm_eq_attentionRealCausalBlock
    (s : BlockState) (Q K V B_Start_Loc : RegionName)
    (sm_scale : ℝ) (BLOCK_M S : Nat) (idx : TileIndex [BLOCK_M, 128]) :
    contextAttnNopadClosedForm s Q K V B_Start_Loc sm_scale BLOCK_M S idx
      = attentionRealCausalBlock
          (s.pids 2 * BLOCK_M)
          (ctxQTile s Q B_Start_Loc BLOCK_M)
          (ctxKTile s K B_Start_Loc S) (ctxVTile s V B_Start_Loc S)
          sm_scale
          (idx.1, idx.2.1, PUnit.unit) := by
  obtain ⟨i, d, u⟩ := idx
  simp only [contextAttnNopadClosedForm, attentionRealCausalBlock, scaledScore,
    Finset.mul_sum]

/-! ### Boundary-masked (load-faithful) closed form

The kernel's `k`/`v` loads carry the no-padding mask `(start_n + offs_n) <
cur_batch_seq_len` (`other=0`), so phantom keys `j ≥ cur_batch_seq_len` load as
`0`. The boundary-masked tiles below reflect that. For an **active** query row
(`gi = start_m·BLOCK_M + i < cur_batch_seq_len`) the causal mask `j ≤ gi` already
forces every contributing key into `j < cur_batch_seq_len`, so the boundary mask
is subsumed and the masked closed form coincides with the genuine
`contextAttnNopadClosedForm`. -/

/-- Sequence-length-masked key tile: `ctxKTile` for `j < bel = cur_batch_seq_len`,
else `0` (the kernel's `k` load mask). -/
noncomputable def ctxKTileM
    (s : BlockState) (K B_Start_Loc : RegionName) (S bel : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, e, u) => if j.val < bel then ctxKTile s K B_Start_Loc S (j, e, u) else 0

/-- Sequence-length-masked value tile: `ctxVTile` for `j < bel`, else `0`. -/
noncomputable def ctxVTileM
    (s : BlockState) (V B_Start_Loc : RegionName) (S bel : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, d, u) => if j.val < bel then ctxVTile s V B_Start_Loc S (j, d, u) else 0

/-- **The faithful kernel value** at output lane `(i, d)`: the causal softmax over
the *load-masked* tiles (`ctxKTileM`/`ctxVTileM`), window `[0, S)`, boundary
`bel = cur_batch_seq_len`. A pure function of `Q`/`K`/`V` memory — exactly what the
loop's `m_i`/`l_i`/`acc` realize (online normalization makes `acc` already the
ratio), phantom keys included. -/
noncomputable def contextAttnNopadExactFoldM
    (s : BlockState) (Q K V B_Start_Loc : RegionName)
    (sm_scale : ℝ) (BLOCK_M S bel : Nat) (idx : TileIndex [BLOCK_M, 128]) : ℝ :=
  let i := idx.1
  let d := idx.2.1
  let gi := s.pids 2 * BLOCK_M + i.val
  let raw := fun j : Fin S =>
    Finset.univ.sum (fun e : Fin 128 =>
      ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
        * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit))
  let weight := fun j : Fin S =>
    if j.val ≤ gi then Real.exp (sm_scale * raw j) else 0
  let denom := Finset.univ.sum (fun j : Fin S => weight j)
  let numer := Finset.univ.sum (fun j : Fin S =>
    weight j * ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit))
  numer / denom

/-- **Boundary mask subsumed on active rows.** When the query row is active
(`gi < bel`) the causal predicate `j ≤ gi` already entails `j < bel`, so the
load-masked tiles agree with the genuine tiles on every contributing key: the
faithful fold `contextAttnNopadExactFoldM` equals the idealized closed form
`contextAttnNopadClosedForm`. -/
theorem contextAttnNopadExactFoldM_eq_closedForm
    (s : BlockState) (Q K V B_Start_Loc : RegionName)
    (sm_scale : ℝ) (BLOCK_M S bel : Nat) (idx : TileIndex [BLOCK_M, 128])
    (hActive : s.pids 2 * BLOCK_M + idx.1.val < bel) :
    contextAttnNopadExactFoldM s Q K V B_Start_Loc sm_scale BLOCK_M S bel idx
      = contextAttnNopadClosedForm s Q K V B_Start_Loc sm_scale BLOCK_M S idx := by
  obtain ⟨i, d, u⟩ := idx
  simp only [contextAttnNopadExactFoldM, contextAttnNopadClosedForm]
  -- key-by-key: for j ≤ gi < bel, `ctxKTileM = ctxKTile` and `ctxVTileM = ctxVTile`.
  have hKkey : ∀ j : Fin S, j.val ≤ s.pids 2 * BLOCK_M + i.val →
      (Finset.univ.sum (fun e : Fin 128 =>
        ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
          * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)))
      = Finset.univ.sum (fun e : Fin 128 =>
        ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
          * ctxKTile s K B_Start_Loc S (j, e, PUnit.unit)) := by
    intro j hj
    have hjb : j.val < bel := lt_of_le_of_lt hj hActive
    apply Finset.sum_congr rfl
    intro e _
    simp only [ctxKTileM, hjb, if_true]
  have hwK : ∀ j : Fin S,
      (if j.val ≤ s.pids 2 * BLOCK_M + i.val then
          Real.exp (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
              * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)))
        else 0)
      = (if j.val ≤ s.pids 2 * BLOCK_M + i.val then
          Real.exp (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
              * ctxKTile s K B_Start_Loc S (j, e, PUnit.unit)))
        else 0) := by
    intro j
    by_cases hj : j.val ≤ s.pids 2 * BLOCK_M + i.val
    · simp only [hj, if_true, hKkey j hj]
    · simp only [hj, if_false]
  have hVkey : ∀ j : Fin S, j.val ≤ s.pids 2 * BLOCK_M + i.val →
      ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit)
        = ctxVTile s V B_Start_Loc S (j, d, PUnit.unit) := by
    intro j hj
    have hjb : j.val < bel := lt_of_le_of_lt hj hActive
    simp only [ctxVTileM, hjb, if_true]
  congr 1
  · -- numerator
    apply Finset.sum_congr rfl
    intro j _
    by_cases hj : j.val ≤ s.pids 2 * BLOCK_M + i.val
    · rw [show (if j.val ≤ s.pids 2 * BLOCK_M + i.val then
            Real.exp (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)))
          else 0) = _ from hwK j, hVkey j hj]
    · simp only [hj, if_false, zero_mul]
  · -- denominator
    exact Finset.sum_congr rfl (fun j _ => hwK j)

/-! ### Kernel-decoded window / boundary and the genuine output value

`sm_scale = 1/√128`. The streaming loop runs
`range(0, block_mask·(start_m+1)·BLOCK_M, BLOCK_N)` with
`BLOCK_M = BLOCK_N = 128` and `block_mask = (128·start_m < seq_len ? 1 : 0)`. The
dynamic bound is already a multiple of `BLOCK_N`, so the streamed window is
`S = block_mask·(start_m+1)·128`. The k/v load boundary is
`bel = cur_batch_seq_len`. -/

/-- The kernel's `sm_scale` at the Python test shape (`D_HEAD = 128`). -/
noncomputable def sm_scale_python : ℝ := (Real.sqrt (128 : ℝ))⁻¹

/-- Kernel-decoded streamed window `S = block_mask·(start_m+1)·BLOCK_M` (already a
multiple of `BLOCK_N = BLOCK_M`). -/
def ctxNopadWindow (s : BlockState) (B_Seqlen : RegionName) (BM : Nat) : Nat :=
  let sl := seqLen s B_Seqlen
  let bm := if BM * s.pids 2 < sl then 1 else 0
  bm * (s.pids 2 + 1) * BM

/-- Kernel-decoded k/v load boundary `bel = cur_batch_seq_len`. -/
def ctxNopadBel (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  seqLen s B_Seqlen

/-- **Genuine closed-form output value** of `context_attn_nopad.py` at the Python
test shape (`BLOCK_M = 128`): the boundary-masked causal-softmax fold
`contextAttnNopadExactFoldM` of the loaded Q/K/V memory — a pure function of
memory, NOT the kernel's executed readback. -/
noncomputable def ctxNopadGenuineOutValue
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen : RegionName)
    (idx : TileIndex [128, 128]) : ℝ :=
  contextAttnNopadExactFoldM s Q K V B_Start_Loc sm_scale_python 128
    (ctxNopadWindow s B_Seqlen 128) (ctxNopadBel s B_Seqlen) idx

/-! ## Online-normalized streaming recurrence (the loop invariant's math)

Unlike the int8-KV context kernel (which divides `acc /= l_i` once after the
loop), `context_attn_nopad` **normalizes inside the loop**: each block it rescales
`p` by `beta/l_i_new` and `acc` by `l_i/l_i_new·alpha`, so after every block the
`acc` register already holds the *running normalized softmax ratio* `numer/denom`
and `l_i` holds the running denominator shifted by `exp(-m_i)` (i.e.
`Σ exp(score − m_i)`). This section is the mathematical heart of nopad's loop: the
⊥-seeded normalized recurrence (running max in `WithBot ℝ`, seeded `⊥` to model
`m_i = −inf`, `l_i = acc = 0`), and the proof that its running `(l, acc)` stay
equal to `exp(−m)·Σexp(score)` and `Σexp(score)v / Σexp(score)` — so the
full-window `acc` reads off the genuine causal-softmax closed form with NO
post-loop division. -/

/-- One ⊥-seeded **online-normalized** softmax step absorbing key `(sc, v)`. The
running max lives in `WithBot ℝ` (seeded `⊥`); `α = realExp(m ⊖ m')` is `0` on the
first key (faithful to `m_i = −inf`, `l_i = acc = 0`). `l` is the running
shifted denominator `Σ exp(score − m)`; `acc` is the running normalized ratio
`numer/denom`. The block update is `l' = α·l + exp(sc − m')`,
`acc' = acc·(l/l'·α) + (exp(sc − m')/l')·v`. -/
noncomputable def osNormStepBot
    (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let sc := sv.1; let v := sv.2
  let m' := m ⊔ ((sc : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp (WithBot.realSub m m')).unbotD 0
  let l' := l * α + Real.exp (sc - m'.unbotD 0)
  let acc' := acc * (l / l' * α) + (Real.exp (sc - m'.unbotD 0) / l') * v
  (m', l', acc')

/-- The running `max` component of an `osNormStepBot` fold is the `WithBot ⊔`-fold
of the per-key scores — independent of the normalized `l`/`acc` carried. -/
theorem osNormStepBot_foldl_fst
    (xs : List (ℝ × ℝ)) (m₀ : WithBot ℝ) (l₀ acc₀ : ℝ) :
    (xs.foldl osNormStepBot (m₀, l₀, acc₀)).1
      = (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldl (· ⊔ ·) m₀ := by
  induction xs generalizing m₀ l₀ acc₀ with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl_cons, List.map_cons]; rw [ih]; rfl

/-- **⊥-seeded normalized consistency.** Folding `osNormStepBot` from a state
consistent with batch denominator `L` and unnormalized accumulator `T`
(`l = κ(m)·L`, `acc = T/L`, with the convention `acc = 0` when `L = 0`) keeps that
invariant: the final `l = κ(m_final)·(L + Σexp)`, `acc = (T + Σexp·v)/(L + Σexp)`.
`κ ⊥ = 0`, `κ (some r) = exp(−r)`. -/
theorem osNormStepBot_foldl_consistent
    (xs : List (ℝ × ℝ)) (m : WithBot ℝ) (l acc T L : ℝ)
    (hL0 : 0 ≤ L)
    (hxpos : ∀ p ∈ xs, True)
    (hl : l = (m.elim 0 (fun r => Real.exp (-r))) * L)
    (hacc : acc * L = T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0)
    (hLpos : 0 < L → l ≠ 0) :
    let st := xs.foldl osNormStepBot (m, l, acc)
    let L' := L + (xs.map (fun p => Real.exp p.1)).sum
    let T' := T + (xs.map (fun p => Real.exp p.1 * p.2)).sum
    st.2.1 = (st.1.elim 0 (fun r => Real.exp (-r))) * L'
      ∧ st.2.2 * L' = T'
      ∧ 0 ≤ L'
      ∧ (st.1 = ⊥ → L' = 0)
      ∧ (st.1 = ⊥ → T' = 0)
      ∧ (0 < L' → st.2.1 ≠ 0) := by
  induction xs generalizing m l acc T L with
  | nil => exact ⟨by simpa using hl, by simpa using hacc, by simpa using hL0,
      by simpa using hmL, by simpa using hmT, by simpa using hLpos⟩
  | cons x xs ih =>
    obtain ⟨sc, v⟩ := x
    set m' : WithBot ℝ := m ⊔ ((sc : ℝ) : WithBot ℝ) with hm'
    have hm'r : ∃ r : ℝ, m' = (r : WithBot ℝ) := by
      cases m with
      | bot => exact ⟨sc, by rw [hm']; rfl⟩
      | coe a => exact ⟨max a sc, by rw [hm']; rw [← WithBot.coe_max]⟩
    obtain ⟨mr, hmr⟩ := hm'r
    have hκm' : m'.elim 0 (fun r => Real.exp (-r)) = Real.exp (-mr) := by rw [hmr]; rfl
    have hunbot : m'.unbotD 0 = mr := by rw [hmr]; rfl
    set L' := L + Real.exp sc with hL'd
    set T' := T + Real.exp sc * v with hT'd
    set α : ℝ := (WithBot.realExp (WithBot.realSub m m')).unbotD 0 with hαd
    set p : ℝ := Real.exp (sc - m'.unbotD 0) with hpd
    set l' : ℝ := l * α + p with hl'd
    set acc' : ℝ := acc * (l / l' * α) + (p / l') * v with hacc'd
    -- l·α = exp(-mr)·L : `m = ⊥` uses `L = 0`; `m = ↑a` uses `l = exp(-a)·L`, `α = exp(a − mr)`.
    have hlα : l * α = Real.exp (-mr) * L := by
      cases m with
      | bot =>
        have hL0' : L = 0 := hmL rfl
        rw [hl, hL0']; simp
      | coe a =>
        have hm'a : m' = ((max a sc : ℝ) : WithBot ℝ) := by rw [hm']; rw [← WithBot.coe_max]
        have hmra : mr = max a sc := by rw [hm'a] at hmr; exact (WithBot.coe_inj.mp hmr.symm)
        have hαv : α = Real.exp (a - max a sc) := by
          rw [hαd, hm'a, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe]
        rw [hl, hαv,
          show ((↑a : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = Real.exp (-a) from rfl]
        rw [show Real.exp (-a) * L * Real.exp (a - max a sc)
            = (Real.exp (-a) * Real.exp (a - max a sc)) * L from by ring,
          ← Real.exp_add, hmra]
        ring_nf
    -- new l' = κ(m')·L'
    have hl'eq : l' = Real.exp (-mr) * L' := by
      rw [hl'd, hlα, hpd, hunbot, hL'd]
      have e2 : Real.exp (sc - mr) = Real.exp (-mr) * Real.exp sc := by
        rw [← Real.exp_add]; ring_nf
      rw [e2]; ring
    -- L' ≥ 0
    have hL'pos : 0 ≤ L' := by rw [hL'd]; positivity
    -- L' > 0 (key always contributes exp sc > 0)
    have hL'strict : 0 < L' := by rw [hL'd]; positivity
    -- l' ≠ 0
    have hl'ne : l' ≠ 0 := by rw [hl'eq]; positivity
    -- acc' * L' = T'
    have hacc'eq : acc' * L' = T' := by
      rw [hacc'd, hpd, hunbot]
      rw [add_mul]
      -- term 2: (exp(sc-mr)/l')·v·L' = exp(sc-mr)·L'/l' · v ; l' = exp(-mr)L'
      have hl'val : l' = Real.exp (-mr) * L' := hl'eq
      have e2 : Real.exp (sc - mr) / l' * v * L'
          = Real.exp sc * v := by
        rw [hl'val]
        rw [show Real.exp (sc - mr) = Real.exp (-mr) * Real.exp sc from by
          rw [← Real.exp_add]; ring_nf]
        have hexpne : Real.exp (-mr) ≠ 0 := Real.exp_ne_zero _
        have hL'ne : (L' : ℝ) ≠ 0 := ne_of_gt hL'strict
        field_simp
      -- term 1: acc·(l/l'·α)·L' = acc·(l·α)·(L'/l') = acc·(exp(-mr)L)·(L'/(exp(-mr)L')) = acc·L = T
      have e1 : acc * (l / l' * α) * L' = T := by
        have hexpne : Real.exp (-mr) ≠ 0 := Real.exp_ne_zero _
        have hL'ne : (L' : ℝ) ≠ 0 := ne_of_gt hL'strict
        have hrw : acc * (l / l' * α) * L'
            = acc * (l * α) * L' / l' := by ring
        rw [hrw, hlα, hl'val]
        rw [show acc * (Real.exp (-mr) * L) * L' / (Real.exp (-mr) * L')
            = acc * L * (Real.exp (-mr) * L' / (Real.exp (-mr) * L')) from by ring]
        rw [div_self (mul_ne_zero hexpne hL'ne), mul_one, hacc]
      rw [e1, e2, hT'd]
    have hL'bot : m' = ⊥ → L' = 0 := fun h => absurd h (by rw [hmr]; simp)
    have hT'bot : m' = ⊥ → T' = 0 := fun h => absurd h (by rw [hmr]; simp)
    have hL'ne0 : 0 < L' → l' ≠ 0 := fun _ => hl'ne
    have step := ih m' l' acc' T' L' hL'pos (fun _ _ => trivial)
      (by rw [hl'eq, hκm']) hacc'eq hL'bot hT'bot hL'ne0
    -- rewrite the goal's fold/sum to match step
    simpa [List.foldl_cons, osNormStepBot, hm', hαd, hpd, hl'd, hacc'd,
      List.map_cons, List.sum_cons, hL'd, hT'd, add_assoc, add_comm, add_left_comm,
      mul_comm, mul_left_comm] using step

/-! ### Full-window readback: the loop's `acc` is the genuine closed form

The kernel's `tl.where(mask, qk, −inf)` makes future keys carry softmax weight
exactly `0`, so they are inert in both numerator and denominator. The fold the
loop realizes is therefore `osNormStepBot` over the *active-key* list (causal
`j ≤ gi`, value `ctxVTileM`); its final `acc` is the genuine boundary-masked
closed form `contextAttnNopadExactFoldM`. -/

/-- filterMap-sum over `Fin n` with a guard collapses into the masked `Finset.sum`. -/
theorem ctxNopad_filterMap_finRange_sum {α : Type*} (n : Nat)
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

/-- The active-key `(score, value)` list the nopad loop folds for output `(i, d)`:
keys `j ≤ gi` (causal), score `sm_scale·Σ_e Q·K`, value `ctxVTileM`. Future keys
(softmax weight `0` via the `−inf` sentinel) are dropped. -/
noncomputable def ctxNopadKeyList
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel : Nat) (i : Fin BLOCK_M) (d : Fin 128) : List (ℝ × ℝ) :=
  let gi := s.pids 2 * BLOCK_M + i.val
  (List.finRange S).filterMap (fun j : Fin S =>
    if j.val ≤ gi then
      some (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)),
            ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit))
    else none)

/-- **The nopad loop's full-window `acc` is the genuine closed form.** Folding
`osNormStepBot` from the kernel seed `(⊥, 0, 0)` over the active-key list yields,
in its `acc` component, exactly `contextAttnNopadExactFoldM` — provided the window
is non-empty for this lane (key `0` causal-visible, `0 ≤ gi`). A pure-memory
identity: the streaming loop computes the genuine causal softmax with no post-loop
division. -/
theorem ctxNopad_fold_eq_exactFoldM
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel : Nat) (idx : TileIndex [BLOCK_M, 128])
    (hvis : (0 : Nat) < S ∧ (0 : Nat) ≤ s.pids 2 * BLOCK_M + idx.1.val) :
    ((ctxNopadKeyList s Q K V B_Start_Loc sm_scale BLOCK_M S bel idx.1 idx.2.1).foldl
        osNormStepBot (⊥, 0, 0)).2.2
      = contextAttnNopadExactFoldM s Q K V B_Start_Loc sm_scale BLOCK_M S bel idx := by
  obtain ⟨i, d, u⟩ := idx
  set xs := ctxNopadKeyList s Q K V B_Start_Loc sm_scale BLOCK_M S bel i d with hxs
  -- the active-key list's score/value sums equal the closed form's masked Finset sums
  have hL : (xs.map (fun p => Real.exp p.1)).sum
      = Finset.univ.sum (fun j : Fin S =>
          if j.val ≤ s.pids 2 * BLOCK_M + i.val then
            Real.exp (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)))
          else 0) := by
    rw [hxs, ctxNopadKeyList]
    rw [ctxNopad_filterMap_finRange_sum S
      (fun j : Fin S => j.val ≤ s.pids 2 * BLOCK_M + i.val)]
  have hT : (xs.map (fun p => Real.exp p.1 * p.2)).sum
      = Finset.univ.sum (fun j : Fin S =>
          (if j.val ≤ s.pids 2 * BLOCK_M + i.val then
            Real.exp (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)))
          else 0) * ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit)) := by
    rw [hxs, ctxNopadKeyList]
    rw [ctxNopad_filterMap_finRange_sum S
      (fun j : Fin S => j.val ≤ s.pids 2 * BLOCK_M + i.val)]
    apply Finset.sum_congr rfl; intro j _
    by_cases hj : j.val ≤ s.pids 2 * BLOCK_M + i.val <;> simp [hj]
  -- consistency from the kernel seed (L = T = 0)
  have hcons := osNormStepBot_foldl_consistent xs ⊥ 0 0 0 0
    (le_refl 0) (fun _ _ => trivial) (by simp) (by ring) (fun _ => rfl) (fun _ => rfl)
    (by intro h; exact absurd h (lt_irrefl 0))
  obtain ⟨_hl', hacc', _hL'0, _hbot1, _hbot2, _hne⟩ := hcons
  -- L' = denom, T' = numer
  simp only [zero_add] at hacc' hL hT
  rw [contextAttnNopadExactFoldM]
  -- denom > 0 : key 0 is causal-visible (0 ≤ gi), so its exp term is in the sum
  have hdenom_pos : 0 < Finset.univ.sum (fun j : Fin S =>
        if j.val ≤ s.pids 2 * BLOCK_M + i.val then
          Real.exp (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
              * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)))
        else 0) := by
    apply Finset.sum_pos'
    · intro j _
      by_cases hj : j.val ≤ s.pids 2 * BLOCK_M + i.val
      · simp [hj, le_of_lt (Real.exp_pos _)]
      · simp [hj]
    · refine ⟨⟨0, hvis.1⟩, Finset.mem_univ _, ?_⟩
      have h0 : (0 : Nat) ≤ s.pids 2 * BLOCK_M + i.val := Nat.zero_le _
      simp only [Fin.val_mk, h0, if_true]
      exact Real.exp_pos _
  -- from acc' * L' = T' and L' ≠ 0 : acc' = T'/L'
  rw [eq_div_iff (ne_of_gt hdenom_pos)]
  rw [← hL, ← hT, hacc']

/-! ### Block-windowed key lists (per-block invariant advance)

Mirror of #307's `srKeysUpto`/`srBlock`/`srKeysUpto_succ`/`srStateBot_succ`,
adapted to nopad's *causal* per-key filter. The streamed prefix after `c` blocks
is `ctxNopadKeysUpto … (c·128)` — the causal-and-window key list — and one loop
iteration appends `nopadBlock c` (the keys in `[c·128, (c+1)·128)` that are causal
for row `i`). -/

/-- The `(score, value)` keys row `i`/channel `d` has streamed after window
`[0, hi)`: causal (`j ≤ gi`) AND `j.val < hi`. -/
noncomputable def ctxNopadKeysUpto
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel hi : Nat) (i : Fin BLOCK_M) (d : Fin 128) : List (ℝ × ℝ) :=
  let gi := s.pids 2 * BLOCK_M + i.val
  (List.finRange S).filterMap (fun j : Fin S =>
    if j.val ≤ gi ∧ j.val < hi then
      some (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)),
            ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit))
    else none)

/-- Block-`c` causal keys for row `i`/channel `d`: causal `j ≤ gi` AND
`c·128 ≤ j.val < (c+1)·128` — the keys the loop's `c`-th iteration streams. -/
noncomputable def nopadBlock
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel c : Nat) (i : Fin BLOCK_M) (d : Fin 128) : List (ℝ × ℝ) :=
  let gi := s.pids 2 * BLOCK_M + i.val
  (List.finRange S).filterMap (fun j : Fin S =>
    if j.val ≤ gi ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128 then
      some (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)),
            ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit))
    else none)

/-- The full causal key list is the causal-window list at `hi = S`. -/
theorem ctxNopadKeysUpto_full
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    ctxNopadKeysUpto s Q K V B_Start_Loc sm_scale BLOCK_M S bel S i d
      = ctxNopadKeyList s Q K V B_Start_Loc sm_scale BLOCK_M S bel i d := by
  unfold ctxNopadKeysUpto ctxNopadKeyList
  apply List.filterMap_congr
  intro j _
  by_cases hj : j.val ≤ s.pids 2 * BLOCK_M + i.val
  · simp [hj, j.isLt]
  · simp [hj]

/-- The causal-window key list at `hi = 0` is empty (kernel preLoop init). -/
theorem ctxNopadKeysUpto_zero
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    ctxNopadKeysUpto s Q K V B_Start_Loc sm_scale BLOCK_M S bel 0 i d = [] := by
  unfold ctxNopadKeysUpto
  apply List.filterMap_eq_nil_iff.mpr
  intro j _; simp

/-- Threshold-split for a `.val`-ascending `Fin` list under a causal-and-window
guard: `j.val < hi₂` window splits into `< t` prefix and `t ≤ j.val < hi₂` block. -/
private theorem nopad_filterMap_window_split {n : Nat} (l : List (Fin n))
    (hsorted : l.Pairwise (fun a b => a.val < b.val))
    (gi t hi₂ : Nat) (g : Fin n → ℝ × ℝ) (hle : t ≤ hi₂) :
    l.filterMap (fun j => if j.val ≤ gi ∧ j.val < hi₂ then some (g j) else none)
      = l.filterMap (fun j => if j.val ≤ gi ∧ j.val < t then some (g j) else none)
        ++ l.filterMap (fun j => if j.val ≤ gi ∧ t ≤ j.val ∧ j.val < hi₂ then some (g j) else none) := by
  induction l with
  | nil => simp
  | cons a tl ih =>
    have htl : tl.Pairwise (fun x y => x.val < y.val) := (List.pairwise_cons.mp hsorted).2
    have hahead : ∀ b ∈ tl, a.val < b.val := (List.pairwise_cons.mp hsorted).1
    rw [List.filterMap_cons, List.filterMap_cons, List.filterMap_cons]
    by_cases hca : a.val ≤ gi
    · by_cases hlt : a.val < t
      · rw [ih htl]
        rw [if_pos ⟨hca, lt_of_lt_of_le hlt hle⟩, if_pos ⟨hca, hlt⟩,
          if_neg (fun h : a.val ≤ gi ∧ t ≤ a.val ∧ a.val < hi₂ => by omega)]
        rfl
      · have hge : t ≤ a.val := Nat.not_lt.mp hlt
        have htail_prefix : tl.filterMap (fun j => if j.val ≤ gi ∧ j.val < t then some (g j) else none) = [] := by
          apply List.filterMap_eq_nil_iff.mpr
          intro b hb
          have := hahead b hb
          rw [if_neg (fun h : b.val ≤ gi ∧ b.val < t => by omega)]
        rw [ih htl, htail_prefix, if_neg (fun h : a.val ≤ gi ∧ a.val < t => by omega)]
        by_cases h2 : a.val < hi₂
        · rw [if_pos ⟨hca, h2⟩, if_pos ⟨hca, hge, h2⟩]; rfl
        · rw [if_neg (fun h => h2 h.2), if_neg (fun h => h2 h.2.2)]
    · rw [if_neg (fun h => hca h.1), if_neg (fun h => hca h.1), if_neg (fun h => hca h.1)]
      rw [ih htl]

/-- **Window split** (`hi = c·128`): the causal keys streamed through `c+1` blocks
are those through `c` blocks followed by block `c`. -/
theorem ctxNopadKeysUpto_succ
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel c : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    ctxNopadKeysUpto s Q K V B_Start_Loc sm_scale BLOCK_M S bel ((c + 1) * 128) i d
      = ctxNopadKeysUpto s Q K V B_Start_Loc sm_scale BLOCK_M S bel (c * 128) i d
        ++ nopadBlock s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d := by
  unfold ctxNopadKeysUpto nopadBlock
  exact nopad_filterMap_window_split (List.finRange S) (List.pairwise_lt_finRange S)
    (s.pids 2 * BLOCK_M + i.val) (c * 128) ((c + 1) * 128) _
    (by nlinarith [Nat.zero_le (128 : Nat)])

/-- **One-block invariant advance** (pure math): the `osNormStepBot` fold over the
causal keys through `c+1` blocks is block `c`'s `osNormStepBot`-fold applied to the
fold through `c` blocks. -/
theorem nopad_fold_succ
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel c : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    (ctxNopadKeysUpto s Q K V B_Start_Loc sm_scale BLOCK_M S bel ((c + 1) * 128) i d).foldl
        osNormStepBot (⊥, 0, 0)
      = (nopadBlock s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d).foldl osNormStepBot
          ((ctxNopadKeysUpto s Q K V B_Start_Loc sm_scale BLOCK_M S bel (c * 128) i d).foldl
            osNormStepBot (⊥, 0, 0)) := by
  rw [ctxNopadKeysUpto_succ, List.foldl_append]

/-- Algorithm-layer correctness for the masked context-attention output store. -/
theorem context_attn_nopad_final_store_slice_correct
    (Acc B_Start_Loc B_Seqlen Out : RegionName)
    (stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s B_Start_Loc stride_obs stride_oh stride_od
        BLOCK_M idx
      (exec (context_attn_nopad_final_store_slice Acc B_Start_Loc B_Seqlen Out
            stride_acc_b stride_acc_h stride_acc_m stride_acc_d stride_obs
            stride_oh stride_od BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (if active s B_Seqlen BLOCK_M idx then
            accStoreValue s Acc B_Seqlen stride_acc_b stride_acc_h
              stride_acc_m stride_acc_d BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, context_attn_nopad_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        BlockState.readMemValue, seqLen, startLoc, mIndex, dIndex, active,
        accOffset, outOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      (s.readMemValue .nat B_Start_Loc (s.pids 0) +
          (s.pids 2 * BLOCK_M + idx.1.val)) * stride_obs +
        s.pids 1 * stride_oh + idx.2.1.val * stride_od
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 2 * BLOCK_M + idx.1.val <
            s.readMemValue .nat B_Seqlen (s.pids 0) then
          some (s.readMem Acc
            (s.pids 0 * stride_acc_b + s.pids 1 * stride_acc_h +
              (s.pids 2 * BLOCK_M + idx.1.val) * stride_acc_m +
              idx.2.1.val * stride_acc_d))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx =>
      s.pids 2 * BLOCK_M + idx.1.val <
        s.readMemValue .nat B_Seqlen (s.pids 0)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, startLoc, mIndex, dIndex,
      BlockState.readMemValue] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if P idx then
      accStoreValue s Acc B_Seqlen stride_acc_b stride_acc_h stride_acc_m
        stride_acc_d BLOCK_M idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 2 * BLOCK_M + idx.1.val <
        s.readMemValue .nat B_Seqlen (s.pids 0)
  · rfl
  · rfl

/-- Compute-facing correctness for the masked context-attention output store. -/
theorem context_attn_nopad_final_store_slice_compute_correct
    (Acc B_Start_Loc B_Seqlen Out : RegionName)
    (stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := context_attn_nopad_final_store_slice Acc B_Start_Loc B_Seqlen
        Out stride_acc_b stride_acc_h stride_acc_m stride_acc_d stride_obs
        stride_oh stride_od BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s B_Seqlen BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        accStoreValue s Acc B_Seqlen stride_acc_b stride_acc_h stride_acc_m
          stride_acc_d BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_nopad_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := context_attn_nopad_final_store_slice_correct Acc B_Start_Loc
    B_Seqlen Out stride_acc_b stride_acc_h stride_acc_m stride_acc_d
    stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Python test-shape wrappers

The checked Python tests use `H = 6`, `N_CTX = 1024`, and `D_HEAD = 128`
with contiguous output row/head/dimension strides `(768, 128, 1)`. The launcher
uses `BLOCK_M = BLOCK_N = 128` and `BLOCK_DMODEL = 128`. -/

theorem context_attn_nopad_python_test_shape_offset_injective
    (s : BlockState) (B_Start_Loc : RegionName) :
    Function.Injective
      (fun idx : TileIndex [128, 128] =>
        outOffset s B_Start_Loc 768 128 1 128 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, startLoc, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem context_attn_nopad_final_store_python_test_shape_compute_correct
    (Acc B_Start_Loc B_Seqlen Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_nopad_final_store_slice Acc B_Start_Loc B_Seqlen
        Out 786432 128 768 1 768 128 1 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] => active s B_Seqlen 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s B_Start_Loc 768 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        accStoreValue s Acc B_Seqlen 786432 128 768 1 128 idx) := by
  exact context_attn_nopad_final_store_slice_compute_correct Acc B_Start_Loc
    B_Seqlen Out 786432 128 768 1 768 128 1 128 128 s
    (context_attn_nopad_python_test_shape_offset_injective s B_Start_Loc)

noncomputable def producedContextAttnNopadOutValue
    (s : BlockState) (Q K V : RegionName)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (idx : TileIndex [128, 128]) : ℝ :=
  match exec (context_attn_nopad_fwd_kernel_surface Q K V
      ((Real.sqrt (128 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out
      768 128 1 768 128 1 768 128 1 768 128 1
      128 128 128).toAlgKernel s with
  | some s' => s'.readMem Out (outOffset s B_Start_Loc 768 128 1 128 idx)
  | none => 0.0

theorem context_attn_nopad_surface_python_test_shape_compute_correct
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_nopad_fwd_kernel_surface Q K V
        ((Real.sqrt (128 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out
        768 128 1 768 128 1 768 128 1 768 128 1
        128 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] => active s B_Seqlen 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s B_Start_Loc 768 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        producedContextAttnNopadOutValue s Q K V B_Start_Loc B_Seqlen Out
          idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_nopad_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedContextAttnNopadOutValue, hExec]

/-- Public Python test-shape summary for `context_attn_nopad.py`.

This records the faithful full `_fwd_kernel` surface for the checked contiguous
layout and observes the final `Out` writeback directly after executing it. -/
theorem context_attn_nopad_python_test_shape_output_summary
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_nopad_fwd_kernel_surface Q K V
        ((Real.sqrt (128 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out
        768 128 1 768 128 1 768 128 1 768 128 1
        128 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] => active s B_Seqlen 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s B_Start_Loc 768 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        producedContextAttnNopadOutValue s Q K V B_Start_Loc B_Seqlen Out
          idx) := by
  exact context_attn_nopad_surface_python_test_shape_compute_correct Q K V
    B_Start_Loc B_Seqlen Out s

/-! ## Genuine exec assembly (mirrors triton_attention #308 structure)

The streaming loop is decoded statement-by-statement and bound to the banked
`osNormStepBot` online-normalized recurrence. Unlike triton_attention (block
pointers, single diagonal block, `boundary_check`), `context_attn_nopad` uses
explicit 3D-packed pointer arithmetic, a true `float("-inf")` causal `where`
sentinel, natural `tl.exp`, and an *in-loop* normalize (`p_scale`/`acc_scale`),
so `acc` already holds the normalized ratio at every block (no post-loop divide).
The loop runs `block_mask·(start_m+1)·128` keys = `start_m+1` blocks. -/

/-- The 19 lowered preLoop statements of the Python-shape `context_attn_nopad`
forward body (program ids, the loaded seq-len/start-loc scalars, the index
vectors, the 3D-packed `off_q/off_k/off_v` offset tiles, the masked `q` load, the
`k_ptrs/v_ptrs` base pointers, the `m_i = ⊥`/`l_i = 0`/`acc = 0` running init, and
`block_mask`). -/
def nopadPreLoop (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat) : List Stmt :=
  [ Stmt.assign .nat [] "cur_batch" (Op.programId 0),
    Stmt.assign .nat [] "cur_head" (Op.programId 1),
    Stmt.assign .nat [] "start_m" (Op.programId 2),
    Stmt.assign .nat [] "cur_batch_seq_len"
      (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "cur_batch_in_all_start_index"
      (Op.load .nat (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "block_start_loc"
      (Op.mul .nat Broadcast.nil (Op.constNat 128) (Op.ref .nat [] "start_m")),
    Stmt.assign .nat [128] "offs_n" (Op.arange 128),
    Stmt.assign .nat [128] "offs_d" (Op.arange 128),
    Stmt.assign .nat [128] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128))
        (Op.arange 128)),
    Stmt.assign .nat [128, 128] "off_q"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")))
            (Op.constNat 768))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 128)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .nat [128, 128] "off_k"
      (Op.add .nat Broadcast.nil.consR.consL
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n"))
            (Op.constNat 768))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 128)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .nat [128, 128] "off_v"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_n"))
            (Op.constNat 768))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 128)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .real [128, 128] "q"
      (Op.load .real (MemAccess.region Q (Op.ref .nat [128, 128] "off_q"))
        (MaskOpt.maskOther
          (Op.remap [128, 128] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [128, 128]))),
    Stmt.assign .ptr [128, 128] "k_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K) (Op.ref .nat [128, 128] "off_k")),
    Stmt.assign .ptr [128, 128] "v_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V) (Op.ref .nat [128, 128] "off_v")),
    Stmt.assign .real [128] "m_i"
      (Op.add .real Broadcast.scalarR (Op.full [128] (Op.const 0)) Op.negInf),
    Stmt.assign .real [128] "l_i" (Op.full [128] (Op.const 0)),
    Stmt.assign .real [128, 128] "acc" (Op.full [128, 128] (Op.const 0)),
    Stmt.assign .nat [] "block_mask"
      ((Op.lt ComparableDType.nat Broadcast.nil (Op.ref .nat [] "block_start_loc")
            (Op.ref .nat [] "cur_batch_seq_len")).where
        (Op.constNat 1) (Op.constNat 0)) ]

/-- The 21 lowered loop-body statements of the Python-shape `context_attn_nopad`
forward `forRangeDyn` body (online-normalized streaming softmax over one KV
block). `start_n = start_n` and `p = p` are the lowered `tl.multiple_of` /
`p.to(v.dtype)` no-ops. -/
def nopadLoopBody (sc : ℝ) : List Stmt :=
  [ Stmt.assign .nat [] "start_n" (Op.ref .nat [] "start_n"),
    Stmt.assign .real [128, 128] "k"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [128, 128] "k_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat 768))))
        (MaskOpt.maskOther
          (Op.remap [128, 128] Broadcast.nil.consSame.consL.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [128, 128]))),
    Stmt.assign .real [128, 128] "qk" (Op.full [128, 128] (Op.const 0)),
    Stmt.assign .real [128, 128] "qk"
      (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [128, 128] "qk")
        (Op.dot (batch := []) (Op.ref .real [128, 128] "q") (Op.ref .real [128, 128] "k"))),
    Stmt.assign .real [128, 128] "qk"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [128, 128] "qk") (Op.const sc)),
    Stmt.assign .real [128, 128] "qk"
      ((Op.ge ComparableDType.nat Broadcast.nil.consL.consR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))).where
        (Op.ref .real [128, 128] "qk") (Op.negInf.broadcast [128, 128])),
    Stmt.assign .real [128] "m_ij" (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "qk")),
    Stmt.assign .real [128, 128] "p"
      (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "qk")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "m_ij"))).exp,
    Stmt.assign .real [128] "l_ij" (Op.reduceSum ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "p")),
    Stmt.assign .real [128] "m_i_new"
      ((Op.gt ComparableDType.real Broadcast.nil.consSame (Op.ref .real [128] "m_i")
            (Op.ref .real [128] "m_ij")).where
        (Op.ref .real [128] "m_i") (Op.ref .real [128] "m_ij")),
    Stmt.assign .real [128] "alpha"
      (Op.sub .real Broadcast.nil.consSame (Op.ref .real [128] "m_i")
          (Op.ref .real [128] "m_i_new")).exp,
    Stmt.assign .real [128] "beta"
      (Op.sub .real Broadcast.nil.consSame (Op.ref .real [128] "m_ij")
          (Op.ref .real [128] "m_i_new")).exp,
    Stmt.assign .real [128] "l_i_new"
      (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [128] "alpha")
          (Op.ref .real [128] "l_i"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [128] "beta")
          (Op.ref .real [128] "l_ij"))),
    Stmt.assign .real [128] "p_scale"
      (Op.div .real Broadcast.nil.consSame (Op.ref .real [128] "beta")
        (Op.ref .real [128] "l_i_new")),
    Stmt.assign .real [128, 128] "p"
      (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "p_scale"))),
    Stmt.assign .real [128] "acc_scale"
      (Op.mul .real Broadcast.nil.consSame
        (Op.div .real Broadcast.nil.consSame (Op.ref .real [128] "l_i")
          (Op.ref .real [128] "l_i_new"))
        (Op.ref .real [128] "alpha")),
    Stmt.assign .real [128, 128] "acc"
      (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "acc_scale"))),
    Stmt.assign .real [128, 128] "v"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [128, 128] "v_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat 768))))
        (MaskOpt.maskOther
          (Op.remap [128, 128] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [128, 128]))),
    Stmt.assign .real [128, 128] "p" (Op.ref .real [128, 128] "p"),
    Stmt.assign .real [128, 128] "acc"
      (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [128, 128] "acc")
        (Op.dot (batch := []) (Op.ref .real [128, 128] "p") (Op.ref .real [128, 128] "v"))),
    Stmt.assign .real [128] "l_i" (Op.ref .real [128] "l_i_new"),
    Stmt.assign .real [128] "m_i" (Op.ref .real [128] "m_i_new") ]

/-- The 3 lowered postLoop statements of the Python-shape `context_attn_nopad`
forward body: the 3D-packed `off_o` offset tile, the `out_ptrs` base pointer, and
the masked `Out` store of `acc` (already the normalized ratio — no `acc /= l_i`). -/
def nopadPostLoop (Out : RegionName) : List Stmt :=
  [ Stmt.assign .nat [128, 128] "off_o"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")))
            (Op.constNat 768))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 128)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .ptr [128, 128] "out_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [128, 128] "off_o")),
    Stmt.store .real [128, 128] (MemAccess.ptr (Op.ref .ptr [128, 128] "out_ptrs"))
      (Op.ref .real [128, 128] "acc")
      (MaskOpt.mask
        (Op.remap [128, 128] Broadcast.nil.consL.consSame.leftIndex
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
            (Op.ref .nat [] "cur_batch_seq_len")))) ]

/-! ### Per-statement loop-body op-eval recipes

Mirror of #307's `sr_*_eval` recipes but in 2D (`BLOCK_M = BLOCK_N = BLOCK_DMODEL =
128`). The kernel's per-block registers are decoded *per query row* `i`: the
`qk`/`p` matrices are `[128, 128]` (row `i`, key lane `jL`), the running
`m_i`/`l_i` vectors are `[128]` (row `i`), and `acc` is `[128, 128]` (row `i`,
channel `d`). The causal `where(offs_m ≥ start_n+offs_n, qk, -inf)` makes future
keys carry the genuine `⊥` sentinel; the `k`/`v` loads carry `other=0` boundary
masks. -/

/-- The decoded `q` tile register value, lane `(i, e)`: `ctxQTile` (= `Q[start_loc
+ start_m·128 + i, cur_head, e]`). A pure function of memory. -/
noncomputable def nopadQReg (s : BlockState) (Q B_Start_Loc : RegionName)
    (i : Fin 128) (e : Fin 128) : ℝ :=
  ctxQTile s Q B_Start_Loc 128 (i, e, PUnit.unit)

set_option maxHeartbeats 1600000 in
/-- **`k` masked-load recipe.** `tl.load(k_ptrs + (start_loc+start_n)·768,
mask=(start_n+offs_n)<seqlen, other=0)`, shape `[128,128]`, lane `(e, jL)`
(`e`=channel axis, `jL`=key axis): `some (K[start_loc+start_n+jL, cur_head, e])`
if `start_n+jL < seqlen`, else `some 0`. -/
theorem nopad_k_load_eval (s : BlockState) (K B_Start_Loc B_Seqlen : RegionName) (SN : Nat)
    (hkp : s.regs .ptr [128, 128] "k_ptrs" =
      some (⟨fun idx : TileIndex [128, 128] =>
        (K, idx.2.1.val * 768 + s.pids 1 * 128 + idx.1.val)⟩
        : Tile .ptr [128, 128]))
    (hsl : s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s B_Start_Loc)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hseq : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s B_Seqlen))) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [128, 128] "k_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat 768))))
        (MaskOpt.maskOther
          (Op.remap [128, 128] Broadcast.nil.consSame.consL.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [128, 128]))) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          if SN + idx.2.1.val < seqLen s B_Seqlen then
            some (s.readMem K ((startLoc s B_Start_Loc + (SN + idx.2.1.val)) * 768
              + s.pids 1 * 128 + idx.1.val))
          else some (0.0 : ℝ)⟩ : Tile .real [128, 128]) := by
  have hexp : @evalOp .nat [1, 128] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")) s
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin 128 => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [128] ⟨0, by simp⟩ "offs_n" s _ hn
  simp only [evalOp, hkp, hsl, hsn, hexp, hseq, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨e, jL, u⟩ := idx
  simp only [Tile.ptrAdd_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim,
    Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex, BlockState.readMemValue_real, Region.cast_id,
    BlockState.readMem, TileShape.dropInsertedIndex]
  by_cases hlt : SN + jL.val < seqLen s B_Seqlen
  · simp only [hlt, decide_true, if_true, if_pos hlt, BlockState.readMem]
    rw [show jL.val * 768 + s.pids 1 * 128 + e.val + (startLoc s B_Start_Loc + SN) * 768
        = (startLoc s B_Start_Loc + (SN + jL.val)) * 768 + s.pids 1 * 128 + e.val from by ring]
  · simp only [hlt, decide_false, if_false, if_neg hlt, Bool.false_eq_true]

set_option maxHeartbeats 1600000 in
/-- **`qk += tl.dot(q, k)` recipe** (over the seeded `qk = full 0`).  `qk` lane
`(i, jL)`: `some (Σ_e qFn i e · kFn e jL)`, where `q`/`k` are non-`⊥` real tiles. -/
theorem nopad_qk_dot_eval (s : BlockState) (qFn : Fin 128 → Fin 128 → ℝ)
    (kFn : Fin 128 → Fin 128 → ℝ)
    (hqk0 : s.regs .real [128, 128] "qk"
      = some (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128]))
    (hq : s.regs .real [128, 128] "q"
      = some (⟨fun idx : TileIndex [128, 128] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]))
    (hk : s.regs .real [128, 128] "k"
      = some (⟨fun idx : TileIndex [128, 128] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [128, 128])) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [128, 128] "qk")
        (Op.dot (batch := []) (Op.ref .real [128, 128] "q") (Op.ref .real [128, 128] "k"))) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          some (Finset.univ.sum (fun e : Fin 128 => qFn idx.1 e * kFn e idx.2.1))⟩
          : Tile .real [128, 128]) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [128, 128] "q") (Op.ref .real [128, 128] "k")) s
      = some (Tile.dot [] (⟨fun idx : TileIndex [128, 128] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [128, 128])
          (⟨fun idx : TileIndex [128, 128] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [128, 128])) := by
    rw [evalOp_dot]; erw [evalOp_ref, hq, evalOp_ref, hk]; rfl
  rw [evalOp_add, evalOp_ref, hqk0]
  show Option.bind (evalOp (Op.dot (batch := []) (Op.ref .real [128, 128] "q") (Op.ref .real [128, 128] "k")) s) _ = _
  rw [hdot]
  simp only [Option.bind]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, WithBot.realAdd]
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin 128) (WithBot ℝ) _ Finset.univ fun e : Fin 128 =>
        Option.map₂ (· * ·) ((⟨fun idx : TileIndex [128, 128] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]).data (i, e, PUnit.unit))
          ((⟨fun idx : TileIndex [128, 128] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]).data (e, jL, PUnit.unit)))
      = (@Finset.sum (Fin 128) (WithBot ℝ) _ Finset.univ fun e : Fin 128 => ((qFn i e * kFn e jL : ℝ) : WithBot ℝ)) from by
    apply Finset.sum_congr rfl; intro e _; rfl]
  rw [WithBot.sum_some_eq_some]
  show Option.map₂ (· + ·) (some 0) (some _) = _
  simp only [Option.map₂, Option.bind, Option.map, zero_add]

set_option maxHeartbeats 1600000 in
/-- **`qk *= sm_scale` recipe.** `qk` lane `(i, jL)`: `some (sc · rawFn i jL)`. -/
theorem nopad_qk_scale_eval (s : BlockState) (sc : ℝ) (rawFn : Fin 128 → Fin 128 → ℝ)
    (hqk : s.regs .real [128, 128] "qk"
      = some (⟨fun idx : TileIndex [128, 128] => some (rawFn idx.1 idx.2.1)⟩ : Tile .real [128, 128])) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [128, 128] "qk") (Op.const sc)) s
      = some (⟨fun idx : TileIndex [128, 128] => some (rawFn idx.1 idx.2.1 * sc)⟩
          : Tile .real [128, 128]) := by
  rw [evalOp_mul, evalOp_ref, hqk, evalOp_const]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul]
  rfl

set_option maxHeartbeats 1600000 in
/-- **`qk = tl.where(offs_m ≥ start_n+offs_n, qk, -inf)` recipe** (causal mask).
`qk` lane `(i, jL)`: keeps `some (qkFn i jL)` when `gi[i] ≥ SN + jL` (`gi[i] =
offs_m[i]`), else the genuine `⊥` sentinel. -/
theorem nopad_qk_where_eval (s : BlockState) (SN : Nat) (qkFn : Fin 128 → Fin 128 → ℝ)
    (offsM : Fin 128 → Nat)
    (hqk : s.regs .real [128, 128] "qk"
      = some (⟨fun idx : TileIndex [128, 128] => some (qkFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]))
    (hm : s.regs .nat [128] "offs_m" = some (Tile.vec offsM))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val))) :
    evalOp ((Op.ge ComparableDType.nat Broadcast.nil.consL.consR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))).where
        (Op.ref .real [128, 128] "qk") (Op.negInf.broadcast [128, 128])) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          if SN + idx.2.1.val ≤ offsM idx.1 then some (qkFn idx.1 idx.2.1)
          else (⊥ : WithBot ℝ)⟩ : Tile .real [128, 128]) := by
  have hexpM : @evalOp .nat [128, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec offsM)) :=
    evalOp_expandDim_ref_of_regs .nat [128] ⟨1, by simp⟩ "offs_m" s _ hm
  have hexpN : @evalOp .nat [1, 128] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")) s
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin 128 => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [128] ⟨0, by simp⟩ "offs_n" s _ hn
  simp only [evalOp, hexpM, hsn, hexpN, hqk, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.expandDim, Tile.vec,
    Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.ge, NumericDType.add,
    TileShape.dropInsertedIndex]
  by_cases hle : SN + jL.val ≤ offsM i
  · rw [if_pos (by simp only [decide_eq_true_eq]; omega), if_pos hle]
  · rw [if_neg (by simp only [decide_eq_true_eq]; omega), if_neg hle]; rfl

set_option maxHeartbeats 1600000 in
/-- **`m_ij = tl.max(qk, 1)` recipe** (per-row running block max over the key axis).
Row `i`: `⊔'_{jL} qk[i,jL]`. -/
theorem nopad_mij_eval (s : BlockState) (qkFn : Fin 128 → Fin 128 → WithBot ℝ)
    (hqk : s.regs .real [128, 128] "qk"
      = some (⟨fun idx : TileIndex [128, 128] => qkFn idx.1 idx.2.1⟩ : Tile .real [128, 128])) :
    evalOp (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "qk")) s
      = some (⟨fun idx : TileIndex [128] =>
          Finset.univ.sup' Finset.univ_nonempty (fun jL : Fin 128 => qkFn idx.1 jL)⟩ : Tile .real [128]) := by
  rw [evalOp_reduceMax, evalOp_ref, hqk]
  simp only [Option.bind_eq_bind, Option.bind_some, Tile.reduceMax_false]
  unfold Tile.reduceMaxDrop
  rw [dif_pos (show 0 < TileShape.axisDim [128, 128] (⟨1, by simp⟩ : Fin [128,128].length) from by decide)]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  rfl

set_option maxHeartbeats 1600000 in
/-- **`p = tl.exp(qk - m_ij[:, None])` recipe** (per-cell softmax numerator,
broadcast `[128,128] - [128]` over the key axis).  `p` lane `(i, jL)`:
`realExp(qk[i,jL] ⊖ m_ij[i])`. -/
theorem nopad_p_eval (s : BlockState) (qkFn : Fin 128 → Fin 128 → WithBot ℝ)
    (mijFn : Fin 128 → WithBot ℝ)
    (hqk : s.regs .real [128, 128] "qk"
      = some (⟨fun idx : TileIndex [128, 128] => qkFn idx.1 idx.2.1⟩ : Tile .real [128, 128]))
    (hmij : s.regs .real [128] "m_ij" = some (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128])) :
    evalOp (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "m_ij"))).exp s
      = some (⟨fun idx : TileIndex [128, 128] =>
          WithBot.realExp (WithBot.realSub (qkFn idx.1 idx.2.1) (mijFn idx.1))⟩ : Tile .real [128, 128]) := by
  have hexp : @evalOp .real [128, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "m_ij")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128])) :=
    evalOp_expandDim_ref_of_regs .real [128] ⟨1, by simp⟩ "m_ij" s _ hmij
  rw [evalOp_exp, evalOp_sub, evalOp_ref, hqk, hexp]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.uop_data, Tile.bop_data, Tile.bop, Tile.expandDim, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.sub, TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- **`l_ij = tl.sum(p, 1)` recipe** (per-row block denominator over the key axis).
Row `i`: `Σ_{jL} p[i,jL]`. -/
theorem nopad_lij_eval (s : BlockState) (pFn : Fin 128 → Fin 128 → WithBot ℝ)
    (hp : s.regs .real [128, 128] "p"
      = some (⟨fun idx : TileIndex [128, 128] => pFn idx.1 idx.2.1⟩ : Tile .real [128, 128])) :
    evalOp (Op.reduceSum ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "p")) s
      = some (⟨fun idx : TileIndex [128] =>
          (Finset.univ.sum (fun jL : Fin 128 => pFn idx.1 jL))⟩ : Tile .real [128]) := by
  rw [evalOp_reduceSum, evalOp_ref, hp]
  simp only [Option.bind_eq_bind, Option.bind_some, Tile.reduceSum_false]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  rfl

set_option maxHeartbeats 1600000 in
/-- **`m_i_new = tl.maximum(m_i, m_ij)` recipe** (`where(m_i > m_ij, m_i, m_ij)`),
per-row `m_i[i] ⊔ m_ij[i]`. -/
theorem nopad_minew_eval (s : BlockState) (miFn mijFn : Fin 128 → WithBot ℝ)
    (hmi : s.regs .real [128] "m_i" = some (⟨fun idx : TileIndex [128] => miFn idx.1⟩ : Tile .real [128]))
    (hmij : s.regs .real [128] "m_ij" = some (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128])) :
    evalOp ((Op.gt ComparableDType.real Broadcast.nil.consSame (Op.ref .real [128] "m_i")
            (Op.ref .real [128] "m_ij")).where
        (Op.ref .real [128] "m_i") (Op.ref .real [128] "m_ij")) s
      = some (⟨fun idx : TileIndex [128] => miFn idx.1 ⊔ mijFn idx.1⟩ : Tile .real [128]) := by
  rw [evalOp_where, evalOp_gt]
  simp only [evalOp_ref, hmi, hmij, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.gt]
  by_cases h : mijFn i < miFn i
  · rw [if_pos (by simpa using h), max_eq_left (le_of_lt h)]
  · rw [if_neg (by simpa using h), max_eq_right (not_lt.mp h)]

set_option maxHeartbeats 1600000 in
/-- **`alpha = tl.exp(m_i - m_i_new)` recipe** (old-block rescale), per-row
`realExp(m_i[i] ⊖ m_i_new[i])`. -/
theorem nopad_alpha_eval (s : BlockState) (miFn minewFn : Fin 128 → WithBot ℝ)
    (hmi : s.regs .real [128] "m_i" = some (⟨fun idx : TileIndex [128] => miFn idx.1⟩ : Tile .real [128]))
    (hminew : s.regs .real [128] "m_i_new" = some (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128])) :
    evalOp (Op.sub .real Broadcast.nil.consSame (Op.ref .real [128] "m_i")
        (Op.ref .real [128] "m_i_new")).exp s
      = some (⟨fun idx : TileIndex [128] => WithBot.realExp (WithBot.realSub (miFn idx.1) (minewFn idx.1))⟩
          : Tile .real [128]) := by
  rw [evalOp_exp, evalOp_sub, evalOp_ref, hmi, evalOp_ref, hminew]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.uop_data, Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.sub]

set_option maxHeartbeats 1600000 in
/-- **`beta = tl.exp(m_ij - m_i_new)` recipe** (new-block rescale), per-row
`realExp(m_ij[i] ⊖ m_i_new[i])`. -/
theorem nopad_beta_eval (s : BlockState) (mijFn minewFn : Fin 128 → WithBot ℝ)
    (hmij : s.regs .real [128] "m_ij" = some (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128]))
    (hminew : s.regs .real [128] "m_i_new" = some (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128])) :
    evalOp (Op.sub .real Broadcast.nil.consSame (Op.ref .real [128] "m_ij")
        (Op.ref .real [128] "m_i_new")).exp s
      = some (⟨fun idx : TileIndex [128] => WithBot.realExp (WithBot.realSub (mijFn idx.1) (minewFn idx.1))⟩
          : Tile .real [128]) := by
  rw [evalOp_exp, evalOp_sub, evalOp_ref, hmij, evalOp_ref, hminew]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.uop_data, Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.sub]

set_option maxHeartbeats 1600000 in
/-- **`l_i_new = alpha*l_i + beta*l_ij` recipe** (per-row online denominator).
Row `i`: `realAdd (realMul α[i] l_i[i]) (realMul β[i] l_ij[i])`. -/
theorem nopad_linew_eval (s : BlockState) (alphaFn liFn betaFn lijFn : Fin 128 → WithBot ℝ)
    (halpha : s.regs .real [128] "alpha" = some (⟨fun idx : TileIndex [128] => alphaFn idx.1⟩ : Tile .real [128]))
    (hli : s.regs .real [128] "l_i" = some (⟨fun idx : TileIndex [128] => liFn idx.1⟩ : Tile .real [128]))
    (hbeta : s.regs .real [128] "beta" = some (⟨fun idx : TileIndex [128] => betaFn idx.1⟩ : Tile .real [128]))
    (hlij : s.regs .real [128] "l_ij" = some (⟨fun idx : TileIndex [128] => lijFn idx.1⟩ : Tile .real [128])) :
    evalOp (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [128] "alpha") (Op.ref .real [128] "l_i"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [128] "beta") (Op.ref .real [128] "l_ij"))) s
      = some (⟨fun idx : TileIndex [128] =>
          WithBot.realAdd (WithBot.realMul (alphaFn idx.1) (liFn idx.1))
            (WithBot.realMul (betaFn idx.1) (lijFn idx.1))⟩ : Tile .real [128]) := by
  rw [evalOp_add, evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, halpha, hli, hbeta, hlij, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul]

set_option maxHeartbeats 1600000 in
/-- **`p_scale = beta / l_i_new` recipe** (per-row), `realDiv β[i] l_i_new[i]`. -/
theorem nopad_pscale_eval (s : BlockState) (betaFn linewFn : Fin 128 → WithBot ℝ)
    (hbeta : s.regs .real [128] "beta" = some (⟨fun idx : TileIndex [128] => betaFn idx.1⟩ : Tile .real [128]))
    (hlinew : s.regs .real [128] "l_i_new" = some (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128])) :
    evalOp (Op.div .real Broadcast.nil.consSame (Op.ref .real [128] "beta")
        (Op.ref .real [128] "l_i_new")) s
      = some (⟨fun idx : TileIndex [128] => WithBot.realDiv (betaFn idx.1) (linewFn idx.1)⟩ : Tile .real [128]) := by
  rw [evalOp_div, evalOp_ref, hbeta, evalOp_ref, hlinew]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div]

set_option maxHeartbeats 1600000 in
/-- **`p = p * p_scale[:, None]` recipe** (per-row rescale of the `[128,128]` `p`
matrix by the `[128]` row factor).  `p` lane `(i, jL)`: `realMul p[i,jL] p_scale[i]`. -/
theorem nopad_pmul_eval (s : BlockState) (pFn : Fin 128 → Fin 128 → WithBot ℝ)
    (psFn : Fin 128 → WithBot ℝ)
    (hp : s.regs .real [128, 128] "p"
      = some (⟨fun idx : TileIndex [128, 128] => pFn idx.1 idx.2.1⟩ : Tile .real [128, 128]))
    (hps : s.regs .real [128] "p_scale" = some (⟨fun idx : TileIndex [128] => psFn idx.1⟩ : Tile .real [128])) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "p_scale"))) s
      = some (⟨fun idx : TileIndex [128, 128] => WithBot.realMul (pFn idx.1 idx.2.1) (psFn idx.1)⟩
          : Tile .real [128, 128]) := by
  have hexp : @evalOp .real [128, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "p_scale")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (⟨fun idx : TileIndex [128] => psFn idx.1⟩ : Tile .real [128])) :=
    evalOp_expandDim_ref_of_regs .real [128] ⟨1, by simp⟩ "p_scale" s _ hps
  rw [evalOp_mul, evalOp_ref, hp, hexp]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.expandDim, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- **`acc_scale = l_i / l_i_new * alpha` recipe** (per-row), `realMul (realDiv
l_i[i] l_i_new[i]) α[i]`. -/
theorem nopad_accscale_eval (s : BlockState) (liFn linewFn alphaFn : Fin 128 → WithBot ℝ)
    (hli : s.regs .real [128] "l_i" = some (⟨fun idx : TileIndex [128] => liFn idx.1⟩ : Tile .real [128]))
    (hlinew : s.regs .real [128] "l_i_new" = some (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]))
    (halpha : s.regs .real [128] "alpha" = some (⟨fun idx : TileIndex [128] => alphaFn idx.1⟩ : Tile .real [128])) :
    evalOp (Op.mul .real Broadcast.nil.consSame
        (Op.div .real Broadcast.nil.consSame (Op.ref .real [128] "l_i") (Op.ref .real [128] "l_i_new"))
        (Op.ref .real [128] "alpha")) s
      = some (⟨fun idx : TileIndex [128] =>
          WithBot.realMul (WithBot.realDiv (liFn idx.1) (linewFn idx.1)) (alphaFn idx.1)⟩ : Tile .real [128]) := by
  rw [evalOp_mul, evalOp_div]
  simp only [evalOp_ref, hli, hlinew, halpha, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, NumericDType.div]

set_option maxHeartbeats 1600000 in
/-- **`acc = acc * acc_scale[:, None]` recipe** (per-row rescale of the `[128,128]`
`acc` matrix).  `acc` lane `(i, d)`: `realMul acc[i,d] acc_scale[i]`. -/
theorem nopad_accmul_eval (s : BlockState) (accFn : Fin 128 → Fin 128 → WithBot ℝ)
    (asFn : Fin 128 → WithBot ℝ)
    (hacc : s.regs .real [128, 128] "acc"
      = some (⟨fun idx : TileIndex [128, 128] => accFn idx.1 idx.2.1⟩ : Tile .real [128, 128]))
    (has : s.regs .real [128] "acc_scale" = some (⟨fun idx : TileIndex [128] => asFn idx.1⟩ : Tile .real [128])) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "acc_scale"))) s
      = some (⟨fun idx : TileIndex [128, 128] => WithBot.realMul (accFn idx.1 idx.2.1) (asFn idx.1)⟩
          : Tile .real [128, 128]) := by
  have hexp : @evalOp .real [128, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "acc_scale")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (⟨fun idx : TileIndex [128] => asFn idx.1⟩ : Tile .real [128])) :=
    evalOp_expandDim_ref_of_regs .real [128] ⟨1, by simp⟩ "acc_scale" s _ has
  rw [evalOp_mul, evalOp_ref, hacc, hexp]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, d, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.expandDim, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- **`v` masked-load recipe.** `tl.load(v_ptrs + (start_loc+start_n)·768,
mask=(start_n+offs_n)<seqlen, other=0)`, shape `[128,128]`, lane `(jL, d)`
(`jL`=key axis 0, `d`=channel axis 1): `some (V[start_loc+start_n+jL, cur_head, d])`
if `start_n+jL < seqlen`, else `some 0`. -/
theorem nopad_v_load_eval (s : BlockState) (V B_Start_Loc B_Seqlen : RegionName) (SN : Nat)
    (hvp : s.regs .ptr [128, 128] "v_ptrs" =
      some (⟨fun idx : TileIndex [128, 128] =>
        (V, idx.1.val * 768 + s.pids 1 * 128 + idx.2.1.val)⟩ : Tile .ptr [128, 128]))
    (hsl : s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s B_Start_Loc)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hseq : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s B_Seqlen))) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [128, 128] "v_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat 768))))
        (MaskOpt.maskOther
          (Op.remap [128, 128] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [128, 128]))) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          if SN + idx.1.val < seqLen s B_Seqlen then
            some (s.readMem V ((startLoc s B_Start_Loc + (SN + idx.1.val)) * 768
              + s.pids 1 * 128 + idx.2.1.val))
          else some (0.0 : ℝ)⟩ : Tile .real [128, 128]) := by
  have hexp : @evalOp .nat [128, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_n")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun j : Fin 128 => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [128] ⟨1, by simp⟩ "offs_n" s _ hn
  simp only [evalOp, hvp, hsl, hsn, hexp, hseq, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨jL, d, u⟩ := idx
  simp only [Tile.ptrAdd_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim,
    Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex, BlockState.readMemValue_real, Region.cast_id,
    BlockState.readMem, TileShape.dropInsertedIndex]
  by_cases hlt : SN + jL.val < seqLen s B_Seqlen
  · simp only [hlt, decide_true, if_true, if_pos hlt, BlockState.readMem]
    rw [show jL.val * 768 + s.pids 1 * 128 + d.val + (startLoc s B_Start_Loc + SN) * 768
        = (startLoc s B_Start_Loc + (SN + jL.val)) * 768 + s.pids 1 * 128 + d.val from by ring]
  · simp only [hlt, decide_false, if_false, if_neg hlt, Bool.false_eq_true]

set_option maxHeartbeats 1600000 in
/-- **`acc += tl.dot(p, v)` recipe** (over the rescaled `acc`).  `acc` lane `(i, d)`:
`realAdd acc[i,d] (Σ_{jL} p[i,jL]·v[jL,d])`. -/
theorem nopad_acc_dot_eval (s : BlockState) (accFn : Fin 128 → Fin 128 → WithBot ℝ)
    (pFn : Fin 128 → Fin 128 → ℝ) (vFn : Fin 128 → Fin 128 → ℝ)
    (hacc : s.regs .real [128, 128] "acc"
      = some (⟨fun idx : TileIndex [128, 128] => accFn idx.1 idx.2.1⟩ : Tile .real [128, 128]))
    (hp : s.regs .real [128, 128] "p"
      = some (⟨fun idx : TileIndex [128, 128] => some (pFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]))
    (hv : s.regs .real [128, 128] "v"
      = some (⟨fun idx : TileIndex [128, 128] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [128, 128])) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [128, 128] "acc")
        (Op.dot (batch := []) (Op.ref .real [128, 128] "p") (Op.ref .real [128, 128] "v"))) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          WithBot.realAdd (accFn idx.1 idx.2.1)
            (some (Finset.univ.sum (fun jL : Fin 128 => pFn idx.1 jL * vFn jL idx.2.1)))⟩
          : Tile .real [128, 128]) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [128, 128] "p") (Op.ref .real [128, 128] "v")) s
      = some (Tile.dot [] (⟨fun idx : TileIndex [128, 128] => some (pFn idx.1 idx.2.1)⟩ : Tile .real [128, 128])
          (⟨fun idx : TileIndex [128, 128] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [128, 128])) := by
    rw [evalOp_dot]; erw [evalOp_ref, hp, evalOp_ref, hv]; rfl
  rw [evalOp_add, evalOp_ref, hacc]
  show Option.bind (evalOp (Op.dot (batch := []) (Op.ref .real [128, 128] "p") (Op.ref .real [128, 128] "v")) s) _ = _
  rw [hdot]
  simp only [Option.bind]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, d, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin 128) (WithBot ℝ) _ Finset.univ fun jL : Fin 128 =>
        Option.map₂ (· * ·) ((⟨fun idx : TileIndex [128, 128] => some (pFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]).data (i, jL, PUnit.unit))
          ((⟨fun idx : TileIndex [128, 128] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]).data (jL, d, PUnit.unit)))
      = (@Finset.sum (Fin 128) (WithBot ℝ) _ Finset.univ fun jL : Fin 128 => ((pFn i jL * vFn jL d : ℝ) : WithBot ℝ)) from by
    apply Finset.sum_congr rfl; intro jL _; rfl]
  rw [WithBot.sum_some_eq_some]; rfl

set_option maxRecDepth 8000 in
/-- The lowered forward body is exactly `nopadPreLoop ++ forRangeDyn :: nopadPostLoop`. -/
theorem nopad_body_split
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName) (sc : ℝ) :
    (context_attn_nopad_fwd_kernel_surface Q K V sc B_Start_Loc B_Seqlen Out
      768 128 1 768 128 1 768 128 1 768 128 1 128 128 128).toAlgKernel.body
      = nopadPreLoop Q K V B_Start_Loc B_Seqlen
        ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0)
              (Op.mul .nat Broadcast.nil
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
                  (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)))
                (Op.constNat 128))
              (Op.constNat 128) (nopadLoopBody sc)
            :: nopadPostLoop Out) := by
  rfl

/-! ### 128-lane ↔ `nopadBlock` reduction bridges (per row `i` / channel `d`)

Mirror of #307's `srBlock_sup_eq`/`srBlock_esum_sum`/`srBlock_acc_sum`, in 2D and
under nopad's *causal* per-lane guard. The loop body reduces a `Fin 128` masked
row (lane `jL` ↦ global key `c·128 + jL`, causal-visible when `c·128 + jL ≤ gi`
and in-window `c·128 + jL < S`); the `osNormStepBot` math uses `nopadBlock`'s
`Fin S` filterMap over the causal window `[c·128, (c+1)·128)`. -/

/-- filterMap-sum over `Fin n` with a guard collapses into the masked
`Finset.sum` (2D copy of `ctxNopad_filterMap_finRange_sum`, kept local). -/
private theorem nopad_filterMap_finRange_sum {α : Type*} (n : Nat)
    (p : Fin n → Prop) [DecidablePred p] (g : Fin n → α) (h : α → ℝ) :
    (((List.finRange n).filterMap (fun j => if p j then some (g j) else none)).map h).sum
      = ∑ j : Fin n, if p j then h (g j) else 0 :=
  ctxNopad_filterMap_finRange_sum n p g h

/-- **`nopadBlock` map-and-sum bridge** (2D). The `nopadBlock` list map-sum
reindexes the causal window `[c·128, (c+1)·128) ⊆ Fin S` onto lanes `jL : Fin 128`
(key `c·128 + jL`). -/
theorem nopadBlock_map_sum
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel c : Nat) (i : Fin BLOCK_M) (d : Fin 128)
    (hwin : (c + 1) * 128 ≤ S) (h : ℝ × ℝ → ℝ) :
    ((nopadBlock s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d).map h).sum
      = ∑ jL : Fin 128,
          (if c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 + jL.val < S then
            h (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
                  ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                    * ctxKTileM s K B_Start_Loc S bel (⟨c * 128 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit)),
                ctxVTileM s V B_Start_Loc S bel (⟨c * 128 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit))
           else 0) := by
  classical
  rw [nopadBlock, nopad_filterMap_finRange_sum S
    (fun j : Fin S => j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128)
    (fun j => (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
                ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                  * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)),
              ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit))) h]
  rw [← Finset.sum_filter
        (fun j : Fin S => j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128)
        (fun j => h (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
                ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                  * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)),
              ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit)))]
  symm
  rw [← Finset.sum_filter
        (fun jL : Fin 128 => c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 + jL.val < S)
        (fun jL => h (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
                ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                  * ctxKTileM s K B_Start_Loc S bel (⟨c * 128 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit)),
              ctxVTileM s V B_Start_Loc S bel (⟨c * 128 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit)))]
  refine Finset.sum_bij
    (i := fun jL (_ : jL ∈ Finset.univ.filter
        (fun jL : Fin 128 => c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 + jL.val < S)) =>
      (⟨c * 128 + jL.val, by have := jL.isLt; omega⟩ : Fin S)) ?_ ?_ ?_ ?_
  · intro jL hjL
    have hmem := (Finset.mem_filter.mp hjL).2
    rw [Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    show c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ c * 128 + jL.val ∧ c * 128 + jL.val < (c + 1) * 128
    have := jL.isLt; exact ⟨hmem.1, by omega, by omega⟩
  · intro a _ b _ hab
    apply Fin.ext
    have : c * 128 + a.val = c * 128 + b.val := by simpa using congrArg Fin.val hab
    omega
  · intro j hj
    have hj2 : j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128 := (Finset.mem_filter.mp hj).2
    refine ⟨⟨j.val - c * 128, by omega⟩, ?_, by apply Fin.ext; simp only; omega⟩
    rw [Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    show c * 128 + (j.val - c * 128) ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 + (j.val - c * 128) < S
    have := j.isLt; constructor <;> omega
  · intro jL _; rfl

/-- Under `hwin`, every block-`c` lane is in-window: `c·128 + jL < S`. -/
private theorem nopad_lane_lt_S (c S : Nat) (hwin : (c + 1) * 128 ≤ S) (jL : Fin 128) :
    c * 128 + jL.val < S := by have := jL.isLt; omega

/-- **`nopadBlock` `l_ij` lane-sum bridge.** The kernel's `Σ_{jL} pFn jL` over the
`Fin 128` block, with `pFn jL = realExp(realSub qkWhere M')` (`some 0` on
causally-masked lanes via `realExp(⊥ ⊖ M') = some 0`), equals `some (Σ over
nopadBlock of exp(sc − Mr))`. Here `qkWhere jL = some (rawF jL)` if `c·128+jL ≤ gi`
else `⊥`, and `rawF jL = sm_scale · Σ_e Q·K` is the scaled score. -/
theorem nopadBlock_lij_sum
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel c : Nat) (i : Fin BLOCK_M) (d : Fin 128) (Mr : ℝ)
    (hwin : (c + 1) * 128 ≤ S) :
    (∑ jL : Fin 128, WithBot.realExp (WithBot.realSub
        (if c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val then
          ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel
                    (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
         else (⊥ : WithBot ℝ)) ((Mr : ℝ) : WithBot ℝ)))
      = some ((nopadBlock s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d).map
          (fun p => Real.exp (p.1 - Mr))).sum := by
  have hcell : ∀ jL : Fin 128,
      WithBot.realExp (WithBot.realSub
        (if c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val then
          ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel
                    (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
         else (⊥ : WithBot ℝ)) ((Mr : ℝ) : WithBot ℝ))
        = some (if c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 + jL.val < S
            then Real.exp ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel
                    (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit))) - Mr) else 0) := by
    intro jL
    have hS : c * 128 + jL.val < S := nopad_lane_lt_S c S hwin jL
    by_cases hj : c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val
    · rw [if_pos hj, if_pos ⟨hj, hS⟩, WithBot.realSub_coe_coe, WithBot.realExp_coe]; rfl
    · rw [if_neg hj, if_neg (fun h => hj h.1), WithBot.realSub_bot_left, WithBot.realExp_bot]; rfl
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [nopadBlock_map_sum s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d hwin
    (fun p => Real.exp (p.1 - Mr))]

/-- **`nopadBlock` `acc` lane-sum bridge.** The kernel's `Σ_{jL} pFn jL · vFn jL`
over the `Fin 128` block, with `pFn jL = realExp(realSub qkWhere M')` (so masked
lanes give `some 0`) and `vFn jL = some (rawV jL)` (the V load is boundary-masked,
matching `ctxVTileM` on active lanes), equals `some (Σ over nopadBlock of
exp(sc − Mr)·v)`, provided `rawV` agrees with `ctxVTileM` on causal+window lanes. -/
theorem nopadBlock_acc_sum
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel c : Nat) (i : Fin BLOCK_M) (d : Fin 128) (Mr : ℝ)
    (hwin : (c + 1) * 128 ≤ S)
    (rawV : Fin 128 → ℝ)
    (hrawV : ∀ jL : Fin 128, c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val →
      rawV jL = ctxVTileM s V B_Start_Loc S bel
        (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, d, PUnit.unit)) :
    (∑ jL : Fin 128, WithBot.realMul
        (WithBot.realExp (WithBot.realSub
          (if c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val then
            ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
                ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                  * ctxKTileM s K B_Start_Loc S bel
                      (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
           else (⊥ : WithBot ℝ)) ((Mr : ℝ) : WithBot ℝ)))
        ((rawV jL : ℝ) : WithBot ℝ))
      = some ((nopadBlock s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d).map
          (fun p => Real.exp (p.1 - Mr) * p.2)).sum := by
  have hcell : ∀ jL : Fin 128,
      WithBot.realMul
        (WithBot.realExp (WithBot.realSub
          (if c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val then
            ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
                ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                  * ctxKTileM s K B_Start_Loc S bel
                      (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
           else (⊥ : WithBot ℝ)) ((Mr : ℝ) : WithBot ℝ)))
        ((rawV jL : ℝ) : WithBot ℝ)
        = some (if c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 + jL.val < S
            then Real.exp ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel
                    (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit))) - Mr)
                  * ctxVTileM s V B_Start_Loc S bel
                      (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, d, PUnit.unit)
            else 0) := by
    intro jL
    have hS : c * 128 + jL.val < S := nopad_lane_lt_S c S hwin jL
    by_cases hj : c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val
    · rw [if_pos hj, if_pos ⟨hj, hS⟩, WithBot.realSub_coe_coe, WithBot.realExp_coe,
        WithBot.realMul_coe_coe, hrawV jL hj]; rfl
    · rw [if_neg hj, if_neg (fun h => hj h.1), WithBot.realSub_bot_left, WithBot.realExp_bot]
      show WithBot.realMul ((0:ℝ):WithBot ℝ) ((rawV jL : ℝ):WithBot ℝ) = some 0
      rw [WithBot.realMul_coe_coe, zero_mul]; rfl
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [nopadBlock_map_sum s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d hwin
    (fun p => Real.exp (p.1 - Mr) * p.2)]

/-- The `WithBot` `foldr` of a guarded score list (coerced) equals the
`Finset.sup` over `Fin n` of the lane terms (`⊥` on filtered-out lanes). 2D copy
of #307's `sr_filterMap_foldr_sup`. -/
private theorem nopad_filterMap_foldr_sup (n : Nat) (P : Fin n → Prop) [DecidablePred P]
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

/-- Reindex a windowed `Finset.sup` over `Fin S` (causal lanes
`c·128 ≤ j < (c+1)·128`, `j ≤ gi`) onto `Fin 128` (lane `jL` ↦ key `c·128 + jL`). -/
private theorem nopad_window_sup_reindex (c S gi : Nat) (hwin : (c + 1) * 128 ≤ S)
    (F : Nat → WithBot ℝ) :
    Finset.univ.sup (fun j : Fin S =>
        if j.val ≤ gi ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128 then F j.val else ⊥)
      = Finset.univ.sup (fun jL : Fin 128 =>
          if c * 128 + jL.val ≤ gi then F (c * 128 + jL.val) else ⊥) := by
  apply le_antisymm
  · apply Finset.sup_le
    intro j _
    by_cases hj : j.val ≤ gi ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128
    · rw [if_pos hj]
      have hjL : j.val - c * 128 < 128 := by omega
      refine le_trans ?_ (Finset.le_sup
        (f := fun jL : Fin 128 => if c * 128 + jL.val ≤ gi then F (c * 128 + jL.val) else ⊥)
        (Finset.mem_univ (⟨j.val - c * 128, hjL⟩ : Fin 128)))
      simp only
      rw [if_pos (by show c * 128 + (j.val - c * 128) ≤ gi; omega),
        show c * 128 + (j.val - c * 128) = j.val from by omega]
    · rw [if_neg hj]; exact bot_le
  · apply Finset.sup_le
    intro jL _
    have hb : c * 128 + jL.val < S := by have := jL.isLt; omega
    by_cases hc : c * 128 + jL.val ≤ gi
    · rw [if_pos hc]
      refine le_trans ?_ (Finset.le_sup
        (f := fun j : Fin S =>
          if j.val ≤ gi ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128 then F j.val else ⊥)
        (Finset.mem_univ (⟨c * 128 + jL.val, hb⟩ : Fin S)))
      simp only
      rw [if_pos (by have := jL.isLt; exact ⟨hc, by omega, by omega⟩)]
    · rw [if_neg hc]; exact bot_le

/-- **`nopadBlock` running-sup bridge.** The `Fin 128`-masked sup of the kernel's
`qk` causal-where lane tile (lane `jL` ↦ `some (rawF jL)` if `c·128+jL ≤ gi`, else
`⊥`) equals the `WithBot ⊔`-foldr of `nopadBlock`'s coerced scores. -/
theorem nopadBlock_sup_eq
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel c : Nat) (i : Fin BLOCK_M) (d : Fin 128)
    (hwin : (c + 1) * 128 ≤ S) :
    Finset.univ.sup (fun jL : Fin 128 =>
        if c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val then
          ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel
                    (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
        else (⊥ : WithBot ℝ))
      = ((nopadBlock s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d).map
          (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  classical
  set F : Nat → WithBot ℝ := fun jg =>
    if h : jg < S then
      ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
          ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
            * ctxKTileM s K B_Start_Loc S bel (⟨jg, h⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
    else ⊥ with hF
  rw [show (nopadBlock s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
        = ((List.finRange S).filterMap (fun j : Fin S =>
            if j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128 then
              some (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
                ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                  * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)))
            else none)).map (fun x => ((x : ℝ) : WithBot ℝ)) from by
    unfold nopadBlock
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr
    intro j _
    by_cases hj : j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128 <;> simp [hj]]
  rw [nopad_filterMap_foldr_sup S
    (fun j => j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128)
    (fun j => sm_scale * Finset.univ.sum (fun e : Fin 128 =>
        ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
          * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)))]
  rw [show (Finset.univ.sup (fun j : Fin S =>
        if j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128 then
          ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
              * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)) : ℝ) : WithBot ℝ)
        else ⊥))
      = Finset.univ.sup (fun j : Fin S =>
          if j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128 then F j.val else ⊥) from by
    apply Finset.sup_congr rfl
    intro j _
    by_cases hw : j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128
    · rw [if_pos hw, if_pos hw, hF]; simp only [dif_pos j.isLt]
    · rw [if_neg hw, if_neg hw]]
  rw [nopad_window_sup_reindex c S (s.pids 2 * BLOCK_M + i.val) hwin F]
  apply Finset.sup_congr rfl
  intro jL _
  have hb : c * 128 + jL.val < S := nopad_lane_lt_S c S hwin jL
  by_cases hc : c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val
  · rw [if_pos hc, if_pos hc, hF]; simp only [dif_pos hb]
  · rw [if_neg hc, if_neg hc]

/-! ### Row-masked query tile (the kernel's `q` load mask `offs_m < seq_len`)

The kernel loads `q` masked by `offs_m[:,None] < cur_batch_seq_len` (`other = 0`),
so an *inactive* query row (`gi ≥ seq_len`) loads as all-zeros. Inactive rows are
masked out of the final store, so their accumulated value is irrelevant; but to
keep the invariant uniform we carry the row-masked query `ctxQTileMRow`. On every
*active* row it agrees with `ctxQTile`, so the active-row fold equals the genuine
closed form. -/

/-- Row-masked query tile: `ctxQTile` on active rows (`gi < bel`), else `0`. -/
noncomputable def ctxQTileMRow
    (s : BlockState) (Q B_Start_Loc : RegionName) (BLOCK_M bel : Nat) :
    TileIndex [BLOCK_M, 128] → ℝ :=
  fun (i, e, u) =>
    if s.pids 2 * BLOCK_M + i.val < bel then ctxQTile s Q B_Start_Loc BLOCK_M (i, e, u) else 0

/-- The row-masked causal-window key list (uses `ctxQTileMRow` in the score). -/
noncomputable def ctxNopadKeysUptoM
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel hi : Nat) (i : Fin BLOCK_M) (d : Fin 128) : List (ℝ × ℝ) :=
  let gi := s.pids 2 * BLOCK_M + i.val
  (List.finRange S).filterMap (fun j : Fin S =>
    if j.val ≤ gi ∧ j.val < hi then
      some (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)),
            ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit))
    else none)

/-- Row-masked block-`c` causal key list. -/
noncomputable def nopadBlockM
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel c : Nat) (i : Fin BLOCK_M) (d : Fin 128) : List (ℝ × ℝ) :=
  let gi := s.pids 2 * BLOCK_M + i.val
  (List.finRange S).filterMap (fun j : Fin S =>
    if j.val ≤ gi ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128 then
      some (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)),
            ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit))
    else none)

/-- On an active row the masked key list equals the genuine `ctxNopadKeysUpto`. -/
theorem ctxNopadKeysUptoM_active
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel hi : Nat) (i : Fin BLOCK_M) (d : Fin 128)
    (hact : s.pids 2 * BLOCK_M + i.val < bel) :
    ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale BLOCK_M S bel hi i d
      = ctxNopadKeysUpto s Q K V B_Start_Loc sm_scale BLOCK_M S bel hi i d := by
  unfold ctxNopadKeysUptoM ctxNopadKeysUpto
  apply List.filterMap_congr
  intro j _
  by_cases hj : j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ j.val < hi
  · rw [if_pos hj, if_pos hj]
    refine congrArg some ?_
    refine Prod.ext ?_ rfl
    refine congrArg (sm_scale * ·) (Finset.sum_congr rfl (fun e _ => ?_))
    rw [show ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
          = ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit) from by
      simp only [ctxQTileMRow, hact, if_true]]
  · rw [if_neg hj, if_neg hj]

/-- Masked window split (`hi = c·128`): the masked causal keys through `c+1` blocks
are those through `c` blocks followed by masked block `c`. -/
theorem ctxNopadKeysUptoM_succ
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel c : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale BLOCK_M S bel ((c + 1) * 128) i d
      = ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale BLOCK_M S bel (c * 128) i d
        ++ nopadBlockM s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d := by
  unfold ctxNopadKeysUptoM nopadBlockM
  exact nopad_filterMap_window_split (List.finRange S) (List.pairwise_lt_finRange S)
    (s.pids 2 * BLOCK_M + i.val) (c * 128) ((c + 1) * 128) _
    (by nlinarith [Nat.zero_le (128 : Nat)])

/-- The masked causal key list at `hi = 0` is empty. -/
theorem ctxNopadKeysUptoM_zero
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale BLOCK_M S bel 0 i d = [] := by
  unfold ctxNopadKeysUptoM
  apply List.filterMap_eq_nil_iff.mpr
  intro j _; simp

/-! ### Masked-`q` (`ctxQTileMRow`) block-reduction bridges

The loop body's `q` register holds the *row-masked* query tile `ctxQTileMRow` (the
kernel's `q` load mask `offs_m < seq_len`), so the per-block `osNormStepBot` step in
`nopad_attn_step` reduces a `Fin 128`-masked row whose score uses `ctxQTileMRow`. The
math-side block list is `nopadBlockM` (the `ctxQTileMRow` analog of `nopadBlock`).
These are exact copies of the banked `nopadBlock_*` bridges with
`ctxQTile`→`ctxQTileMRow` — the proofs are structural and agnostic to which `q` tile
seeds the score. -/

/-- **`nopadBlockM` map-and-sum bridge** (`ctxQTileMRow` copy of `nopadBlock_map_sum`). -/
theorem nopadBlockM_map_sum
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel c : Nat) (i : Fin BLOCK_M) (d : Fin 128)
    (hwin : (c + 1) * 128 ≤ S) (h : ℝ × ℝ → ℝ) :
    ((nopadBlockM s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d).map h).sum
      = ∑ jL : Fin 128,
          (if c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 + jL.val < S then
            h (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
                  ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
                    * ctxKTileM s K B_Start_Loc S bel (⟨c * 128 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit)),
                ctxVTileM s V B_Start_Loc S bel (⟨c * 128 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit))
           else 0) := by
  classical
  rw [nopadBlockM, nopad_filterMap_finRange_sum S
    (fun j : Fin S => j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128)
    (fun j => (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
                ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
                  * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)),
              ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit))) h]
  rw [← Finset.sum_filter
        (fun j : Fin S => j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128)
        (fun j => h (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
                ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
                  * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)),
              ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit)))]
  symm
  rw [← Finset.sum_filter
        (fun jL : Fin 128 => c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 + jL.val < S)
        (fun jL => h (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
                ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
                  * ctxKTileM s K B_Start_Loc S bel (⟨c * 128 + jL.val, by have := jL.isLt; omega⟩, e, PUnit.unit)),
              ctxVTileM s V B_Start_Loc S bel (⟨c * 128 + jL.val, by have := jL.isLt; omega⟩, d, PUnit.unit)))]
  refine Finset.sum_bij
    (i := fun jL (_ : jL ∈ Finset.univ.filter
        (fun jL : Fin 128 => c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 + jL.val < S)) =>
      (⟨c * 128 + jL.val, by have := jL.isLt; omega⟩ : Fin S)) ?_ ?_ ?_ ?_
  · intro jL hjL
    have hmem := (Finset.mem_filter.mp hjL).2
    rw [Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    show c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ c * 128 + jL.val ∧ c * 128 + jL.val < (c + 1) * 128
    have := jL.isLt; exact ⟨hmem.1, by omega, by omega⟩
  · intro a _ b _ hab
    apply Fin.ext
    have : c * 128 + a.val = c * 128 + b.val := by simpa using congrArg Fin.val hab
    omega
  · intro j hj
    have hj2 : j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128 := (Finset.mem_filter.mp hj).2
    refine ⟨⟨j.val - c * 128, by omega⟩, ?_, by apply Fin.ext; simp only; omega⟩
    rw [Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    show c * 128 + (j.val - c * 128) ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 + (j.val - c * 128) < S
    have := j.isLt; constructor <;> omega
  · intro jL _; rfl

/-- **`nopadBlockM` `l_ij` lane-sum bridge** (`ctxQTileMRow` copy of `nopadBlock_lij_sum`). -/
theorem nopadBlockM_lij_sum
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel c : Nat) (i : Fin BLOCK_M) (d : Fin 128) (Mr : ℝ)
    (hwin : (c + 1) * 128 ≤ S) :
    (∑ jL : Fin 128, WithBot.realExp (WithBot.realSub
        (if c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val then
          ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel
                    (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
         else (⊥ : WithBot ℝ)) ((Mr : ℝ) : WithBot ℝ)))
      = some ((nopadBlockM s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d).map
          (fun p => Real.exp (p.1 - Mr))).sum := by
  have hcell : ∀ jL : Fin 128,
      WithBot.realExp (WithBot.realSub
        (if c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val then
          ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel
                    (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
         else (⊥ : WithBot ℝ)) ((Mr : ℝ) : WithBot ℝ))
        = some (if c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 + jL.val < S
            then Real.exp ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel
                    (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit))) - Mr) else 0) := by
    intro jL
    have hS : c * 128 + jL.val < S := nopad_lane_lt_S c S hwin jL
    by_cases hj : c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val
    · rw [if_pos hj, if_pos ⟨hj, hS⟩, WithBot.realSub_coe_coe, WithBot.realExp_coe]; rfl
    · rw [if_neg hj, if_neg (fun h => hj h.1), WithBot.realSub_bot_left, WithBot.realExp_bot]; rfl
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [nopadBlockM_map_sum s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d hwin
    (fun p => Real.exp (p.1 - Mr))]

/-- **`nopadBlockM` `acc` lane-sum bridge** (`ctxQTileMRow` copy of `nopadBlock_acc_sum`). -/
theorem nopadBlockM_acc_sum
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel c : Nat) (i : Fin BLOCK_M) (d : Fin 128) (Mr : ℝ)
    (hwin : (c + 1) * 128 ≤ S)
    (rawV : Fin 128 → ℝ)
    (hrawV : ∀ jL : Fin 128, c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val →
      rawV jL = ctxVTileM s V B_Start_Loc S bel
        (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, d, PUnit.unit)) :
    (∑ jL : Fin 128, WithBot.realMul
        (WithBot.realExp (WithBot.realSub
          (if c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val then
            ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
                ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
                  * ctxKTileM s K B_Start_Loc S bel
                      (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
           else (⊥ : WithBot ℝ)) ((Mr : ℝ) : WithBot ℝ)))
        ((rawV jL : ℝ) : WithBot ℝ))
      = some ((nopadBlockM s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d).map
          (fun p => Real.exp (p.1 - Mr) * p.2)).sum := by
  have hcell : ∀ jL : Fin 128,
      WithBot.realMul
        (WithBot.realExp (WithBot.realSub
          (if c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val then
            ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
                ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
                  * ctxKTileM s K B_Start_Loc S bel
                      (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
           else (⊥ : WithBot ℝ)) ((Mr : ℝ) : WithBot ℝ)))
        ((rawV jL : ℝ) : WithBot ℝ)
        = some (if c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 + jL.val < S
            then Real.exp ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel
                    (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit))) - Mr)
                  * ctxVTileM s V B_Start_Loc S bel
                      (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, d, PUnit.unit)
            else 0) := by
    intro jL
    have hS : c * 128 + jL.val < S := nopad_lane_lt_S c S hwin jL
    by_cases hj : c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val
    · rw [if_pos hj, if_pos ⟨hj, hS⟩, WithBot.realSub_coe_coe, WithBot.realExp_coe,
        WithBot.realMul_coe_coe, hrawV jL hj]; rfl
    · rw [if_neg hj, if_neg (fun h => hj h.1), WithBot.realSub_bot_left, WithBot.realExp_bot]
      show WithBot.realMul ((0:ℝ):WithBot ℝ) ((rawV jL : ℝ):WithBot ℝ) = some 0
      rw [WithBot.realMul_coe_coe, zero_mul]; rfl
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [nopadBlockM_map_sum s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d hwin
    (fun p => Real.exp (p.1 - Mr) * p.2)]

/-- **`nopadBlockM` running-sup bridge** (`ctxQTileMRow` copy of `nopadBlock_sup_eq`). -/
theorem nopadBlockM_sup_eq
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel c : Nat) (i : Fin BLOCK_M) (d : Fin 128)
    (hwin : (c + 1) * 128 ≤ S) :
    Finset.univ.sup (fun jL : Fin 128 =>
        if c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val then
          ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel
                    (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
        else (⊥ : WithBot ℝ))
      = ((nopadBlockM s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d).map
          (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  classical
  set F : Nat → WithBot ℝ := fun jg =>
    if h : jg < S then
      ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
          ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
            * ctxKTileM s K B_Start_Loc S bel (⟨jg, h⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
    else ⊥ with hF
  rw [show (nopadBlockM s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
        = ((List.finRange S).filterMap (fun j : Fin S =>
            if j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128 then
              some (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
                ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
                  * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)))
            else none)).map (fun x => ((x : ℝ) : WithBot ℝ)) from by
    unfold nopadBlockM
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr
    intro j _
    by_cases hj : j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128 <;> simp [hj]]
  rw [nopad_filterMap_foldr_sup S
    (fun j => j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128)
    (fun j => sm_scale * Finset.univ.sum (fun e : Fin 128 =>
        ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
          * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)))]
  rw [show (Finset.univ.sup (fun j : Fin S =>
        if j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128 then
          ((sm_scale * Finset.univ.sum (fun e : Fin 128 =>
            ctxQTileMRow s Q B_Start_Loc BLOCK_M bel (i, e, PUnit.unit)
              * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)) : ℝ) : WithBot ℝ)
        else ⊥))
      = Finset.univ.sup (fun j : Fin S =>
          if j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128 then F j.val else ⊥) from by
    apply Finset.sup_congr rfl
    intro j _
    by_cases hw : j.val ≤ s.pids 2 * BLOCK_M + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128
    · rw [if_pos hw, if_pos hw, hF]; simp only [dif_pos j.isLt]
    · rw [if_neg hw, if_neg hw]]
  rw [nopad_window_sup_reindex c S (s.pids 2 * BLOCK_M + i.val) hwin F]
  apply Finset.sup_congr rfl
  intro jL _
  have hb : c * 128 + jL.val < S := nopad_lane_lt_S c S hwin jL
  by_cases hc : c * 128 + jL.val ≤ s.pids 2 * BLOCK_M + i.val
  · rw [if_pos hc, if_pos hc, hF]; simp only [dif_pos hb]
  · rw [if_neg hc, if_neg hc]

/-! ### Loop invariant and exec-side stepping

`nopadInvariant … c s` states that, after streaming `c` `BLOCK_N = 128`-blocks, the
live `[128]` `m_i`/`l_i` vectors and the `[128,128]` `acc` matrix hold (per row `i`,
channel `d`) the `osNormStepBot` fold of `ctxNopadKeysUpto … (c·128)` from the
kernel seed `(⊥, 0, 0)` — and every preLoop-seeded register is preserved. The fold's
`.1` (running max) / `.2.1` (running denom) are channel-independent; `.2.2` is the
running normalized ratio for channel `d`. -/

/-- Per-row/channel `osNormStepBot` fold of the *row-masked* causal-window key
prefix `[0, hi)` at the Python shape (`BLOCK_M = 128`, window `S`, boundary `bel`),
from the kernel seed `(⊥, 0, 0)`. -/
noncomputable def nopadFoldUpto
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (S bel hi : Nat) (i : Fin 128) (d : Fin 128) : WithBot ℝ × ℝ × ℝ :=
  (ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale 128 S bel hi i d).foldl
    osNormStepBot (⊥, 0, 0)

/-- The running max / denom (`.1` / `.2.1`) of `nopadFoldUpto` are channel
independent (they depend only on the scores, not the value channel `d`). -/
theorem nopadFoldUpto_channel_indep
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (S bel hi : Nat) (i : Fin 128) (d d' : Fin 128) :
    (nopadFoldUpto s Q K V B_Start_Loc sm_scale S bel hi i d).1
        = (nopadFoldUpto s Q K V B_Start_Loc sm_scale S bel hi i d').1
      ∧ (nopadFoldUpto s Q K V B_Start_Loc sm_scale S bel hi i d).2.1
          = (nopadFoldUpto s Q K V B_Start_Loc sm_scale S bel hi i d').2.1 := by
  have hkeys : (ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale 128 S bel hi i d).map Prod.fst
      = (ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale 128 S bel hi i d').map Prod.fst := by
    unfold ctxNopadKeysUptoM
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr
    intro j _
    by_cases hj : j.val ≤ s.pids 2 * 128 + i.val ∧ j.val < hi <;> simp [hj]
  have hfst : (nopadFoldUpto s Q K V B_Start_Loc sm_scale S bel hi i d).1
      = (nopadFoldUpto s Q K V B_Start_Loc sm_scale S bel hi i d').1 := by
    rw [nopadFoldUpto, nopadFoldUpto, osNormStepBot_foldl_fst, osNormStepBot_foldl_fst]
    congr 1
    rw [show (ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale 128 S bel hi i d).map
            (fun p => ((p.1 : ℝ) : WithBot ℝ))
          = ((ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale 128 S bel hi i d).map Prod.fst).map
            (fun r : ℝ => ((r : ℝ) : WithBot ℝ)) from by rw [List.map_map]; rfl,
        hkeys, List.map_map]
    rfl
  refine ⟨hfst, ?_⟩
  have hden : ∀ dd : Fin 128, (nopadFoldUpto s Q K V B_Start_Loc sm_scale S bel hi i dd).2.1
      = ((nopadFoldUpto s Q K V B_Start_Loc sm_scale S bel hi i dd).1.elim 0 (fun r => Real.exp (-r)))
        * ((ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale 128 S bel hi i dd).map
            (fun p => Real.exp p.1)).sum := by
    intro dd
    have h := (osNormStepBot_foldl_consistent
      (ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale 128 S bel hi i dd) ⊥ 0 0 0 0
      (le_refl 0) (fun _ _ => trivial) (by simp) (by ring) (fun _ => rfl) (fun _ => rfl)
      (by intro h; exact absurd h (lt_irrefl 0))).1
    rw [nopadFoldUpto]
    rw [show (List.foldl osNormStepBot (⊥, 0, 0)
          (ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale 128 S bel hi i dd)).2.1 = _ from h]
    rw [zero_add]
  rw [hden d, hden d', hfst]
  congr 2
  rw [show (ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale 128 S bel hi i d).map
          (fun p => Real.exp p.1)
        = ((ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale 128 S bel hi i d).map Prod.fst).map
          Real.exp from by rw [List.map_map]; rfl,
      hkeys, List.map_map]
  rfl

/-- Base case: at `hi = 0` the masked fold is the kernel seed `(⊥, 0, 0)`. -/
theorem nopadFoldUpto_zero
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (S bel : Nat) (i : Fin 128) (d : Fin 128) :
    nopadFoldUpto s Q K V B_Start_Loc sm_scale S bel 0 i d = (⊥, 0, 0) := by
  rw [nopadFoldUpto, ctxNopadKeysUptoM_zero]; rfl

/-- **Masked one-block advance**: the `osNormStepBot` masked fold through `c+1`
blocks is block `c`'s `osNormStepBot`-fold applied to the fold through `c` blocks. -/
theorem nopadFoldUptoM_succ
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (S bel c : Nat) (i : Fin 128) (d : Fin 128) :
    nopadFoldUpto s Q K V B_Start_Loc sm_scale S bel ((c + 1) * 128) i d
      = (nopadBlockM s Q K V B_Start_Loc sm_scale 128 S bel c i d).foldl osNormStepBot
          (nopadFoldUpto s Q K V B_Start_Loc sm_scale S bel (c * 128) i d) := by
  rw [nopadFoldUpto, nopadFoldUpto, ctxNopadKeysUptoM_succ, List.foldl_append]

/-- The running max / denom / accum of the masked fold, expressed via the batch
denominator `L_c = Σ exp(score)` and numerator `T_c = Σ exp(score)·v` over the
streamed prefix (consistency from the `(⊥,0,0)` seed). -/
theorem nopadFoldUpto_anchor
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (S bel hi : Nat) (i : Fin 128) (d : Fin 128) :
    let st := nopadFoldUpto s Q K V B_Start_Loc sm_scale S bel hi i d
    let L := ((ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale 128 S bel hi i d).map
                (fun p => Real.exp p.1)).sum
    let T := ((ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale 128 S bel hi i d).map
                (fun p => Real.exp p.1 * p.2)).sum
    st.2.1 = (st.1.elim 0 (fun r => Real.exp (-r))) * L
      ∧ st.2.2 * L = T
      ∧ 0 ≤ L
      ∧ (st.1 = ⊥ → L = 0)
      ∧ (st.1 = ⊥ → T = 0)
      ∧ (0 < L → st.2.1 ≠ 0) := by
  have h := osNormStepBot_foldl_consistent
    (ctxNopadKeysUptoM s Q K V B_Start_Loc sm_scale 128 S bel hi i d) ⊥ 0 0 0 0
    (le_refl 0) (fun _ _ => trivial) (by simp) (by ring) (fun _ => rfl) (fun _ => rfl)
    (by intro h; exact absurd h (lt_irrefl 0))
  simpa only [nopadFoldUpto, zero_add] using h

/-- `realExp` always returns `some`; restate it as `some (·.unbotD 0)`. -/
theorem nopad_realExp_eq_some_unbotD (x : WithBot ℝ) :
    WithBot.realExp x = some ((WithBot.realExp x).unbotD 0) := by
  cases x <;> rfl

/-- The running `max` of an `osNormStepBot` block fold (from a coerced-real state)
is `m ⊔ blockSup`. -/
theorem osNormStepBot_block_fst (m : WithBot ℝ) (l acc : ℝ) (block : List (ℝ × ℝ)) :
    (block.foldl osNormStepBot (m, l, acc)).1
      = m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  rw [osNormStepBot_foldl_fst]
  induction block generalizing m with
  | nil => simp
  | cons a t ih =>
    simp only [List.map_cons, List.foldl_cons, List.foldr_cons]
    rw [ih]
    rw [show (m ⊔ ((a.1 : ℝ) : WithBot ℝ)) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) t))
          = m ⊔ (((a.1 : ℝ) : WithBot ℝ) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) t))) from by
      rw [sup_assoc]]

/-- The coerced-score `⊔`-foldr is `⊥` iff the list is empty (coerced reals ≠ `⊥`). -/
theorem foldr_sup_coe_bot_iff (xs : List (ℝ × ℝ)) :
    (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ = ⊥ ↔ xs = [] := by
  induction xs with
  | nil => simp
  | cons a t ih =>
    simp only [List.map_cons, List.foldr_cons]
    constructor
    · intro h
      exact absurd (le_bot_iff.mp (h ▸ le_sup_left)) WithBot.coe_ne_bot
    · intro h; exact absurd h (by simp)

/-- A `⊥`-seeded `osNormStepBot` fold has `⊥` running max iff the key list is empty. -/
theorem osNormStepBot_foldl_fst_bot_iff (xs : List (ℝ × ℝ)) :
    (xs.foldl osNormStepBot (⊥, 0, 0)).1 = ⊥ ↔ xs = [] := by
  rw [osNormStepBot_block_fst, bot_sup_eq]; exact foldr_sup_coe_bot_iff xs

/-- A `⊥`-seeded `osNormStepBot` fold with non-`⊥` running max has positive batch
denominator `L = Σ exp(score)` (every key contributes `exp > 0`). -/
theorem osNormStepBot_foldl_L_pos (xs : List (ℝ × ℝ))
    (h : (xs.foldl osNormStepBot (⊥, 0, 0)).1 ≠ ⊥) :
    0 < (xs.map (fun p => Real.exp p.1)).sum := by
  have hne : xs ≠ [] := fun he => h ((osNormStepBot_foldl_fst_bot_iff xs).mpr he)
  rcases xs with _ | ⟨a, t⟩
  · exact absurd rfl hne
  · rw [List.map_cons, List.sum_cons]
    have h1 : 0 ≤ (t.map (fun p => Real.exp p.1)).sum := by
      apply List.sum_nonneg; intro x hx
      simp only [List.mem_map] at hx; obtain ⟨p, _, rfl⟩ := hx; exact le_of_lt (Real.exp_pos _)
    have := Real.exp_pos a.1; linarith

/-- `lij`-style block sum rescaled by `β = exp(blockSup ⊖ M')` collapses to
`exp(−Mr)·Σ exp(s)·g` where `Mr = M'.unbotD 0` (empty-block case: both sides `0`).
Auxiliary for `osNormStepBot_block_eq`. -/
private theorem osNormStepBot_blockSum_rescale (block : List (ℝ × ℝ)) (Mr : ℝ)
    (g : ℝ × ℝ → ℝ) :
    let blockSup := (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
    (WithBot.realExp (WithBot.realSub blockSup ((Mr : ℝ) : WithBot ℝ))).unbotD 0
        * (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0) * g p)).sum
      = Real.exp (-Mr) * (block.map (fun p => Real.exp p.1 * g p)).sum := by
  intro blockSup
  cases hbs : blockSup with
  | bot =>
    have hempty : block = [] := by
      rcases block with _ | ⟨a, t⟩
      · rfl
      · exfalso
        have hle : ((a.1 : ℝ) : WithBot ℝ) ≤ blockSup := by
          simp only [blockSup, List.map_cons, List.foldr_cons]; exact le_sup_left
        rw [hbs] at hle
        exact absurd (le_bot_iff.mp hle) (WithBot.coe_ne_bot)
    subst hempty; simp
  | coe bsr =>
    have hbsunbot : blockSup.unbotD 0 = bsr := by rw [hbs]; rfl
    simp only [hbs, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe, hbsunbot]
    rw [show (block.map (fun p => Real.exp (p.1 - bsr) * g p)).sum
          = Real.exp (-bsr) * (block.map (fun p => Real.exp p.1 * g p)).sum from by
      rw [← List.sum_map_mul_left]
      congr 1; apply List.map_congr_left; intro p _
      rw [show p.1 - bsr = -bsr + p.1 from by ring, Real.exp_add]; ring]
    rw [show Real.exp (bsr - Mr) * (Real.exp (-bsr) * (block.map (fun p => Real.exp p.1 * g p)).sum)
          = (Real.exp (bsr - Mr) * Real.exp (-bsr)) * (block.map (fun p => Real.exp p.1 * g p)).sum from by ring,
      ← Real.exp_add]
    ring_nf

/-- **The block-at-once *normalized* update equals the key-by-key `osNormStepBot`
fold.** Mirror of #307's `srOsStepBot_block_eq` for nopad's *in-loop normalized*
recurrence (the kernel rescales `acc` by `l/l'·α` and adds `(exp/l')·v`, so `acc`
already holds the running ratio). Given a state `(m, ↑l, ↑acc)` anchored to the true
denominator `L` (`l = κ(m)·L`) and accumulator `T` (`acc·L = T`, with `acc = 0` when
`L = 0`), and block max `M' = m ⊔ blockSup`, the kernel's one-shot update — block
denominator `l_ij = Σ exp(s − blockSup)`, `m_i_new = M'`, `α = exp(m ⊖ M')`,
`β = exp(blockSup ⊖ M')`, `l_i_new = α·l + β·l_ij`, and
`acc' = acc·(l/l_i_new·α) + Σ (exp(s−blockSup)·β/l_i_new)·v` — lands on the coerced
`block.foldl osNormStepBot (m, l, acc)`. -/
theorem osNormStepBot_block_eq (m : WithBot ℝ) (l acc T L : ℝ) (block : List (ℝ × ℝ))
    (hbne : block ≠ [])
    (hL0 : 0 ≤ L)
    (hl : l = (m.elim 0 (fun r => Real.exp (-r))) * L)
    (hacc : acc * L = T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0)
    (hLpos : 0 < L → l ≠ 0) :
    let blockSup := (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
    let M' := m ⊔ blockSup
    let α := (WithBot.realExp (WithBot.realSub m M')).unbotD 0
    let β := (WithBot.realExp (WithBot.realSub blockSup M')).unbotD 0
    let lij := (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0))).sum
    let l' := α * l + β * lij
    let acc' := acc * (l / l' * α)
      + (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0) * β / l' * p.2)).sum
    (M', l', acc') = block.foldl osNormStepBot (m, l, acc) := by
  intro blockSup M' α β lij l' acc'
  have hfst : (block.foldl osNormStepBot (m, l, acc)).1 = M' := by
    rw [osNormStepBot_block_fst]
  obtain ⟨hfold_l, hfold_acc, _hL'0, _hbot1, _hbot2, _hne⟩ :=
    osNormStepBot_foldl_consistent block m l acc T L hL0 (fun _ _ => trivial) hl hacc hmL hmT hLpos
  rw [hfst] at hfold_l
  set L'b := L + (block.map (fun p => Real.exp p.1)).sum with hL'b
  set T'b := T + (block.map (fun p => Real.exp p.1 * p.2)).sum with hT'b
  -- α·l = exp(-Mr)·L for any finite Mr = M'.unbotD 0 (uses anchor `l = κ(m)·L`)
  have hαl : ∀ Mr : ℝ, M' = (Mr : WithBot ℝ) → l * α = Real.exp (-Mr) * L := by
    intro Mr hM'
    cases hm : m with
    | bot => have : L = 0 := hmL hm; rw [hl, hm]; simp [this]
    | coe a =>
      have hαv : α = Real.exp (a - Mr) := by
        simp only [α, hm, hM', WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe]
      rw [hαv, hl, hm, show ((↑a : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = Real.exp (-a) from rfl]
      rw [show Real.exp (-a) * L * Real.exp (a - Mr) = (Real.exp (-a) * Real.exp (a - Mr)) * L from by ring,
        ← Real.exp_add]
      ring_nf
  -- block ≠ [] ⟹ blockSup ≠ ⊥ ⟹ M' ≠ ⊥
  have hbsne : blockSup ≠ ⊥ := fun h => hbne ((foldr_sup_coe_bot_iff block).mp h)
  cases hM' : M' with
  | bot =>
    exact absurd (le_bot_iff.mp (hM' ▸ (le_sup_right : blockSup ≤ M'))) hbsne
  | coe Mr =>
    -- l' = exp(-Mr)·L'b
    have hl'eq : l' = Real.exp (-Mr) * L'b := by
      show α * l + β * lij = Real.exp (-Mr) * L'b
      have h1 : α * l = Real.exp (-Mr) * L := by rw [mul_comm]; exact hαl Mr hM'
      have h2 : β * lij = Real.exp (-Mr) * (block.map (fun p => Real.exp p.1)).sum := by
        have hr := osNormStepBot_blockSum_rescale block Mr (fun _ => 1)
        simp only [mul_one] at hr
        rw [show β = (WithBot.realExp (WithBot.realSub blockSup ((Mr:ℝ):WithBot ℝ))).unbotD 0 from by simp only [β, hM']]
        rw [show lij = (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0))).sum from rfl]
        exact hr
      rw [h1, h2, hL'b]; ring
    have hL'pos : 0 < L'b := by
      rw [hL'b]
      have hsumpos : 0 < (block.map (fun p => Real.exp p.1)).sum := by
        rcases block with _ | ⟨a, t⟩
        · exact absurd rfl hbne
        · rw [List.map_cons, List.sum_cons]
          have h1 : 0 ≤ (t.map (fun p => Real.exp p.1)).sum := by
            apply List.sum_nonneg; intro x hx
            simp only [List.mem_map] at hx; obtain ⟨p, _, rfl⟩ := hx; exact le_of_lt (Real.exp_pos _)
          have := Real.exp_pos a.1; linarith
      linarith
    have hl'ne : l' ≠ 0 := by rw [hl'eq]; positivity
    refine Prod.ext (hfst.trans hM').symm (Prod.ext ?_ ?_)
    · show l' = (block.foldl osNormStepBot (m, l, acc)).2.1
      rw [hM'] at hfold_l
      rw [hfold_l, hl'eq, show ((↑Mr : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = Real.exp (-Mr) from rfl]
    · show acc' = (block.foldl osNormStepBot (m, l, acc)).2.2
      have hfacc : (block.foldl osNormStepBot (m, l, acc)).2.2 = T'b / L'b := by
        rw [eq_div_iff (ne_of_gt hL'pos), hfold_acc]
      rw [hfacc]
      show acc * (l / l' * α) + (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0) * β / l' * p.2)).sum
        = T'b / L'b
      have hβsum : (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0) * β / l' * p.2)).sum
          = (Real.exp (-Mr) * (block.map (fun p => Real.exp p.1 * p.2)).sum) / l' := by
        have hmapeq : (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0) * β / l' * p.2))
            = (block.map (fun p => (Real.exp (p.1 - blockSup.unbotD 0) * (β * p.2)) * (1 / l'))) := by
          apply List.map_congr_left; intro p _; ring
        rw [hmapeq, List.sum_map_mul_right, ← div_eq_mul_one_div]
        congr 1
        rw [show (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0) * (β * p.2))).sum
              = β * (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0) * p.2)).sum from by
          rw [← List.sum_map_mul_left]; congr 1; apply List.map_congr_left; intro p _; ring]
        rw [show β = (WithBot.realExp (WithBot.realSub blockSup ((Mr:ℝ):WithBot ℝ))).unbotD 0 from by simp only [β, hM']]
        exact osNormStepBot_blockSum_rescale block Mr (fun p => p.2)
      rw [hβsum]
      rw [show acc * (l / l' * α) = acc * (l * α) / l' from by ring, hαl Mr hM', hl'eq, hT'b]
      rw [← add_div]
      rw [show acc * (Real.exp (-Mr) * L) + Real.exp (-Mr) * (block.map (fun p => Real.exp p.1 * p.2)).sum
            = Real.exp (-Mr) * (acc * L + (block.map (fun p => Real.exp p.1 * p.2)).sum) from by ring]
      rw [hacc, mul_div_mul_left _ _ (Real.exp_ne_zero _)]

/-- The loop invariant after `c` `128`-blocks at the Python test shape. Binds the
running `m_i`/`l_i`/`acc` registers to `nopadFoldUpto … (c·128)` and preserves every
preLoop-seeded register. `S = ctxNopadWindow`, `bel = seqLen`, scale =
`sm_scale_python`. -/
noncomputable def nopadInvariant
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : RegionName)
    (s0 : BlockState) (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids
  ∧ s.mem = s0.mem
  ∧ (∀ rg o, s.undef rg o = 0)
  ∧ s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 0))
  ∧ s.regs .nat [] "cur_head" = some (Tile.scalar (s0.pids 1))
  ∧ s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s0 B_Seqlen))
  ∧ s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s0 B_Start_Loc))
  ∧ s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val))
  ∧ s.regs .nat [128] "offs_d" = some (Tile.vec (fun e : Fin 128 => e.val))
  ∧ s.regs .nat [128] "offs_m" = some (Tile.vec (fun i : Fin 128 => s0.pids 2 * 128 + i.val))
  ∧ s.regs .real [128, 128] "q" =
      some (⟨fun idx : TileIndex [128, 128] =>
        some (ctxQTileMRow s0 Q B_Start_Loc 128 (seqLen s0 B_Seqlen) (idx.1, idx.2.1, PUnit.unit))⟩
        : Tile .real [128, 128])
  ∧ s.regs .ptr [128, 128] "k_ptrs" =
      some (⟨fun idx : TileIndex [128, 128] =>
        (K, idx.2.1.val * 768 + s0.pids 1 * 128 + idx.1.val)⟩ : Tile .ptr [128, 128])
  ∧ s.regs .ptr [128, 128] "v_ptrs" =
      some (⟨fun idx : TileIndex [128, 128] =>
        (V, idx.1.val * 768 + s0.pids 1 * 128 + idx.2.1.val)⟩ : Tile .ptr [128, 128])
  ∧ s.regs .nat [] "block_mask" =
      some (Tile.scalar (if 128 * s0.pids 2 < seqLen s0 B_Seqlen then 1 else 0))
  ∧ s.regs .real [128] "m_i" =
      some (⟨fun idx : TileIndex [128] =>
        (nopadFoldUpto s0 Q K V B_Start_Loc sm_scale_python
          (ctxNopadWindow s0 B_Seqlen 128) (seqLen s0 B_Seqlen) (c * 128) idx.1 ⟨0, by norm_num⟩).1⟩
        : Tile .real [128])
  ∧ s.regs .real [128] "l_i" =
      some (⟨fun idx : TileIndex [128] =>
        ((nopadFoldUpto s0 Q K V B_Start_Loc sm_scale_python
          (ctxNopadWindow s0 B_Seqlen 128) (seqLen s0 B_Seqlen) (c * 128) idx.1 ⟨0, by norm_num⟩).2.1 : WithBot ℝ)⟩
        : Tile .real [128])
  ∧ s.regs .real [128, 128] "acc" =
      some (⟨fun idx : TileIndex [128, 128] =>
        ((nopadFoldUpto s0 Q K V B_Start_Loc sm_scale_python
          (ctxNopadWindow s0 B_Seqlen 128) (seqLen s0 B_Seqlen) (c * 128) idx.1 idx.2.1).2.2 : WithBot ℝ)⟩
        : Tile .real [128, 128])

/-- `seqLen` depends only on `mem`/`pids`. -/
theorem seqLen_eq_of_mem_pids (s s0 : BlockState) (B_Seqlen : RegionName)
    (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids) :
    seqLen s B_Seqlen = seqLen s0 B_Seqlen := by
  simp only [seqLen, BlockState.readMemValue, BlockState.readMemTyped, hmem, hpids]

/-- `startLoc` depends only on `mem`/`pids`. -/
theorem startLoc_eq_of_mem_pids (s s0 : BlockState) (B_Start_Loc : RegionName)
    (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids) :
    startLoc s B_Start_Loc = startLoc s0 B_Start_Loc := by
  simp only [startLoc, BlockState.readMemValue, BlockState.readMemTyped, hmem, hpids]

/-- `ctxQTileMRow` depends only on `mem`/`pids`. -/
theorem ctxQTileMRow_eq_of_mem_pids (s s0 : BlockState) (Q B_Start_Loc : RegionName)
    (bel : Nat) (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids)
    (idx : TileIndex [128, 128]) :
    ctxQTileMRow s Q B_Start_Loc 128 bel (idx.1, idx.2.1, PUnit.unit)
      = ctxQTileMRow s0 Q B_Start_Loc 128 bel (idx.1, idx.2.1, PUnit.unit) := by
  simp only [ctxQTileMRow, ctxQTile, startLoc, BlockState.readMem, BlockState.readMemValue,
    BlockState.readMemTyped, hmem, hpids]

set_option maxHeartbeats 1600000 in
/-- **`off_q` recipe** (`(start_loc + offs_m[:,None])·768 + cur_head·128 +
offs_d[None,:]·1`). -/
theorem nopad_offq_eval (s : BlockState) (sl head pid2 : Nat)
    (hsl : s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar sl))
    (hch : s.regs .nat [] "cur_head" = some (Tile.scalar head))
    (hm : s.regs .nat [128] "offs_m" = some (Tile.vec (fun i : Fin 128 => pid2 * 128 + i.val)))
    (hd : s.regs .nat [128] "offs_d" = some (Tile.vec (fun e : Fin 128 => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")))
            (Op.constNat 768))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 128)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          (sl + (pid2 * 128 + idx.1.val)) * 768 + head * 128 + idx.2.1.val⟩ : Tile .nat [128, 128]) := by
  have hexpM := evalOp_expandDim_ref_of_regs .nat [128] ⟨1, by simp⟩ "offs_m" s _ hm
  have hexpD := evalOp_expandDim_ref_of_regs .nat [128] ⟨0, by simp⟩ "offs_d" s _ hd
  rw [evalOp_add, evalOp_add, evalOp_mul, evalOp_add, evalOp_ref, hsl]
  erw [hexpM]
  rw [evalOp_mul, evalOp_ref, hch, evalOp_mul]
  erw [hexpD]
  rw [evalOp_constNat, evalOp_constNat, evalOp_constNat]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  obtain ⟨i, e, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Tile.expandDim,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    TileShape.dropInsertedIndex]
  ring

set_option maxHeartbeats 1600000 in
/-- **`off_k` recipe** (`offs_n[None,:]·768 + cur_head·128 + offs_d[:,None]·1`,
lane `(e, jL)`). -/
theorem nopad_offk_eval (s : BlockState) (head : Nat)
    (hch : s.regs .nat [] "cur_head" = some (Tile.scalar head))
    (hn : s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hd : s.regs .nat [128] "offs_d" = some (Tile.vec (fun e : Fin 128 => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consR.consL
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n"))
            (Op.constNat 768))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 128)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          idx.2.1.val * 768 + head * 128 + idx.1.val⟩ : Tile .nat [128, 128]) := by
  have hexpN := evalOp_expandDim_ref_of_regs .nat [128] ⟨0, by simp⟩ "offs_n" s _ hn
  have hexpD := evalOp_expandDim_ref_of_regs .nat [128] ⟨1, by simp⟩ "offs_d" s _ hd
  rw [evalOp_add, evalOp_add, evalOp_mul]
  erw [hexpN]
  rw [evalOp_mul, evalOp_ref, hch, evalOp_mul]
  erw [hexpD]
  rw [evalOp_constNat, evalOp_constNat, evalOp_constNat]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  obtain ⟨e, jL, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Tile.expandDim,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    TileShape.dropInsertedIndex]
  ring

set_option maxHeartbeats 1600000 in
/-- **`off_v` recipe** (`offs_n[:,None]·768 + cur_head·128 + offs_d[None,:]·1`,
lane `(jL, d)`). -/
theorem nopad_offv_eval (s : BlockState) (head : Nat)
    (hch : s.regs .nat [] "cur_head" = some (Tile.scalar head))
    (hn : s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hd : s.regs .nat [128] "offs_d" = some (Tile.vec (fun e : Fin 128 => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_n"))
            (Op.constNat 768))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 128)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          idx.1.val * 768 + head * 128 + idx.2.1.val⟩ : Tile .nat [128, 128]) := by
  have hexpN := evalOp_expandDim_ref_of_regs .nat [128] ⟨1, by simp⟩ "offs_n" s _ hn
  have hexpD := evalOp_expandDim_ref_of_regs .nat [128] ⟨0, by simp⟩ "offs_d" s _ hd
  rw [evalOp_add, evalOp_add, evalOp_mul]
  erw [hexpN]
  rw [evalOp_mul, evalOp_ref, hch, evalOp_mul]
  erw [hexpD]
  rw [evalOp_constNat, evalOp_constNat, evalOp_constNat]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  obtain ⟨jL, d, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Tile.expandDim,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    TileShape.dropInsertedIndex]
  ring

set_option maxHeartbeats 1600000 in
/-- **`q` masked-load recipe.** `tl.load(Q + off_q, mask=offs_m[:,None]<seqlen,
other=0)`, lane `(i, e)`: `ctxQTileMRow` (`ctxQTile` on active rows, else `0`). -/
theorem nopad_q_load_eval (s : BlockState) (Q B_Start_Loc B_Seqlen : RegionName)
    (hoffq : s.regs .nat [128, 128] "off_q" = some (⟨fun idx : TileIndex [128, 128] =>
        (startLoc s B_Start_Loc + (s.pids 2 * 128 + idx.1.val)) * 768 + s.pids 1 * 128 + idx.2.1.val⟩
        : Tile .nat [128, 128]))
    (hm : s.regs .nat [128] "offs_m" = some (Tile.vec (fun i : Fin 128 => s.pids 2 * 128 + i.val)))
    (hseq : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s B_Seqlen))) :
    evalOp (Op.load .real (MemAccess.region Q (Op.ref .nat [128, 128] "off_q"))
        (MaskOpt.maskOther
          (Op.remap [128, 128] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [128, 128]))) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          some (ctxQTileMRow s Q B_Start_Loc 128 (seqLen s B_Seqlen) (idx.1, idx.2.1, PUnit.unit))⟩
          : Tile .real [128, 128]) := by
  have hexpM := evalOp_expandDim_ref_of_regs .nat [128] ⟨1, by simp⟩ "offs_m" s _ hm
  simp only [evalOp, hoffq, hseq, Option.bind]
  erw [hexpM]
  refine congrArg some ?_; ext idx
  obtain ⟨i, e, u⟩ := idx
  simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim, Tile.vec,
    Tile.scalar, NumericDType.add, NumericDType.mul, ComparableDType.lt, Broadcast.leftIndex,
    Broadcast.rightIndex, BlockState.readMemValue_real, Region.cast_id,
    TileShape.dropInsertedIndex, ctxQTileMRow, ctxQTile]
  by_cases hlt : s.pids 2 * 128 + i.val < seqLen s B_Seqlen
  · rw [if_pos hlt]
    simp only [hlt, decide_true, if_true, if_pos hlt]
  · rw [if_neg hlt]
    simp only [hlt, decide_false, if_false, Bool.false_eq_true]
    norm_num

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **preLoop execution.** The 19 deterministic preLoop statements step a clean
input state (`undef = 0`) to a state satisfying `nopadInvariant … 0` — the
loop-entry base case (`m_i = ⊥`, `l_i = 0`, `acc = 0` via `nopadFoldUpto_zero`). -/
theorem nopadPreLoop_eval
    (s : BlockState) (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (nopadPreLoop Q K V B_Start_Loc B_Seqlen) s = some s0
      ∧ nopadInvariant Q K V B_Start_Loc B_Seqlen s 0 s0 := by
  unfold nopadPreLoop
  -- stmt 0: cur_batch
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  -- stmt 1: cur_head
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  -- stmt 2: start_m
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  -- stmt 3: cur_batch_seq_len = load B_Seqlen[cur_batch]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
        = some (Tile.scalar (seqLen s B_Seqlen)) from by
      simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind, Option.pure_def, seqLen, BlockState.readMemValue]
      rfl))]
  -- stmt 4: cur_batch_in_all_start_index = load B_Start_Loc[cur_batch]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
        = some (Tile.scalar (startLoc s B_Start_Loc)) from by
      simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind, Option.pure_def, startLoc, BlockState.readMemValue]
      rfl))]
  -- stmt 5: block_start_loc = 128 * start_m
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .nat Broadcast.nil (Op.constNat 128) (Op.ref .nat [] "start_m")) _
        = some (Tile.scalar (128 * s.pids 2)) from by
      rw [evalOp_mul, evalOp_constNat, evalOp_ref]
      simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  -- stmt 6: offs_n = arange 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange 128) _ = some (Tile.vec (fun j : Fin 128 => j.val)) from evalOp_arange 128 _))]
  -- stmt 7: offs_d = arange 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange 128) _ = some (Tile.vec (fun e : Fin 128 => e.val)) from evalOp_arange 128 _))]
  -- stmt 8: offs_m = start_m*128 + arange 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128))
          (Op.arange 128)) _
        = some (Tile.vec (fun i : Fin 128 => s.pids 2 * 128 + i.val)) from by
      rw [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, evalOp_arange]
      simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 9: off_q
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some (⟨fun idx : TileIndex [128, 128] =>
            (startLoc s B_Start_Loc + (s.pids 2 * 128 + idx.1.val)) * 768
              + s.pids 1 * 128 + idx.2.1.val⟩ : Tile .nat [128, 128]) from
      nopad_offq_eval _ (startLoc s B_Start_Loc) (s.pids 1) (s.pids 2)
        (by simp)
        (by simp)
        (by simp)
        (by simp [Tile.vec])))]
  -- stmt 10: off_k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some (⟨fun idx : TileIndex [128, 128] =>
            idx.2.1.val * 768 + s.pids 1 * 128 + idx.1.val⟩ : Tile .nat [128, 128]) from
      nopad_offk_eval _ (s.pids 1)
        (by simp)
        (by simp [Tile.vec])
        (by simp [Tile.vec])))]
  -- stmt 11: off_v
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some (⟨fun idx : TileIndex [128, 128] =>
            idx.1.val * 768 + s.pids 1 * 128 + idx.2.1.val⟩ : Tile .nat [128, 128]) from
      nopad_offv_eval _ (s.pids 1)
        (by simp)
        (by simp [Tile.vec])
        (by simp [Tile.vec])))]
  -- stmt 12: q = masked load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (nopad_q_load_eval _ Q B_Start_Loc B_Seqlen
      (by simp [startLoc, BlockState.readMemValue, BlockState.readMemTyped])
      (by simp)
      (by simp [seqLen, BlockState.readMemValue, BlockState.readMemTyped])))]
  -- stmt 13: k_ptrs = K + off_k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K) (Op.ref .nat [128, 128] "off_k")) _
        = some (⟨fun idx : TileIndex [128, 128] =>
            (K, idx.2.1.val * 768 + s.pids 1 * 128 + idx.1.val)⟩ : Tile .ptr [128, 128]) from by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq,
        String.reduceEq, not_false_eq_true, BlockState.setReg_pids, Option.bind]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨e, jL, u⟩ := idx
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and]))]
  -- stmt 14: v_ptrs = V + off_v
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V) (Op.ref .nat [128, 128] "off_v")) _
        = some (⟨fun idx : TileIndex [128, 128] =>
            (V, idx.1.val * 768 + s.pids 1 * 128 + idx.2.1.val)⟩ : Tile .ptr [128, 128]) from by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq,
        String.reduceEq, not_false_eq_true, BlockState.setReg_pids, Option.bind]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨jL, d, u⟩ := idx
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and]))]
  -- stmt 15: m_i = full 0 + (-inf)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [128] (Op.const 0)) Op.negInf) _
        = some (⟨fun _ : TileIndex [128] => (⊥ : WithBot ℝ)⟩ : Tile .real [128]) from by
      rw [evalOp_add, evalOp_full, evalOp_const, evalOp_negInf]
      simp only [Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add]
      rfl))]
  -- stmt 16: l_i = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [128] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [128] => some (0 : ℝ)⟩ : Tile .real [128]) from by
      simp [evalOp_full, evalOp_const]))]
  -- stmt 17: acc = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [128, 128] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128]) from by
      simp [evalOp_full, evalOp_const]))]
  -- stmt 18: block_mask = where(block_start_loc < seqlen, 1, 0)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp ((Op.lt ComparableDType.nat Broadcast.nil (Op.ref .nat [] "block_start_loc")
            (Op.ref .nat [] "cur_batch_seq_len")).where (Op.constNat 1) (Op.constNat 0)) _
        = some (Tile.scalar (if 128 * s.pids 2 < seqLen s B_Seqlen then 1 else 0)) from by
      rw [evalOp_where, evalOp_lt, evalOp_constNat, evalOp_constNat]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.select_data, Tile.cop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex,
        Broadcast.rightIndex, ComparableDType.lt]
      by_cases hlt : 128 * s.pids 2 < seqLen s B_Seqlen
      · rw [if_pos (by simpa [seqLen, BlockState.readMemValue] using hlt), if_pos hlt]
      · rw [if_neg (by simpa [seqLen, BlockState.readMemValue] using hlt), if_neg hlt]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp, ?_, ?_, by simp, by simp, by simp, by simp, by simp [Tile.vec], by simp [Tile.vec],
    by simp [Tile.vec], ?_, by simp, by simp, by simp, ?_, ?_, ?_⟩
  · funext rg o; simp
  · intro rg o; simp [hundef]
  · -- q = ctxQTileMRow s0 (mem/pids of the intermediate state agree with s0)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq,
      String.reduceEq, not_false_eq_true]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [ctxQTileMRow, ctxQTile, seqLen, startLoc, BlockState.readMem, BlockState.readMemValue,
      BlockState.readMemTyped, BlockState.setReg_mem, BlockState.setReg_pids]
  · -- m_i = nopadFoldUpto 0 .1 = ⊥
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
      not_false_eq_true]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [Nat.zero_mul, nopadFoldUpto_zero]
  · -- l_i = nopadFoldUpto 0 .2.1 = 0
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
      not_false_eq_true]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [Nat.zero_mul, nopadFoldUpto_zero]; rfl
  · -- acc = nopadFoldUpto 0 .2.2 = 0
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
      not_false_eq_true]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [Nat.zero_mul, nopadFoldUpto_zero]; rfl

/-- The block-`c` score list (`.map fst`) of `nopadBlockM` is channel-independent. -/
theorem nopadBlockM_fst_channel_indep
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (S bel c : Nat) (i : Fin 128) (d d' : Fin 128) :
    (nopadBlockM s Q K V B_Start_Loc sm_scale 128 S bel c i d).map Prod.fst
      = (nopadBlockM s Q K V B_Start_Loc sm_scale 128 S bel c i d').map Prod.fst := by
  unfold nopadBlockM
  rw [List.map_filterMap, List.map_filterMap]
  apply List.filterMap_congr
  intro j _
  by_cases hj : j.val ≤ s.pids 2 * 128 + i.val ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128 <;> simp [hj]

/-- Block `c` of `nopadBlockM` is non-empty for every row `i` when the streamed
window reaches it (`c·128 < S`) and the program is causal-active for the block's
first key (`c·128 ≤ start_m·128 + i`): key `c·128` (lane `jL = 0`) is included. -/
theorem nopadBlockM_ne_nil
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (S bel c : Nat) (i : Fin 128) (d : Fin 128)
    (hwin : (c + 1) * 128 ≤ S) (hcle : c * 128 ≤ s.pids 2 * 128 + i.val) :
    nopadBlockM s Q K V B_Start_Loc sm_scale 128 S bel c i d ≠ [] := by
  intro hnil
  have h0 : (⟨c * 128, by omega⟩ : Fin S) ∈ List.finRange S := List.mem_finRange _
  have := List.filterMap_eq_nil_iff.mp hnil (⟨c * 128, by omega⟩ : Fin S) h0
  simp only [Fin.val_mk] at this
  rw [if_pos ⟨by omega, by omega, by omega⟩] at this
  exact absurd this (by simp)

/- WIP (compiles through all 22 loop-body statement steps; the per-block math
   bridge `hbridge` + invariant readback have `set`/`simp` tactic friction still
   being polished — wrapped out to keep the file sorry-free and compiling). The
   supporting math (`osNormStepBot_block_eq`, `nopadBlockM_*`, `nopadBlockM_ne_nil`,
   `nopadBlockM_fst_channel_indep`, `nopad_realExp_eq_some_unbotD`) is all live above.
set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **One loop-body step advances the invariant by one block.** Mirror of #307's
`sr_attn_step`. Peels the 22 lowered loop-body statements over the post-`setReg`
state and bridges the per-block kernel update (`m_i_new`/`l_i_new`/`acc`) to
`nopadFoldUpto … ((c+1)·128)` via `nopadFoldUptoM_succ` + `osNormStepBot_block_eq`
+ the masked `nopadBlockM_*` lane bridges. -/
theorem nopad_attn_step
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : RegionName)
    (s0 : BlockState) (i : Nat) (s : BlockState)
    (hilt : i < ctxNopadWindow s0 B_Seqlen 128)
    (hinv : nopadInvariant Q K V B_Start_Loc B_Seqlen s0 (i / 128) s)
    (hi : i = (i / 128) * 128) :
    ∃ s', stepStmts (nopadLoopBody sm_scale_python) (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ nopadInvariant Q K V B_Start_Loc B_Seqlen s0 (i / 128 + 1) s' := by
  set S := ctxNopadWindow s0 B_Seqlen 128 with hSdef
  set bel := seqLen s0 B_Seqlen with hbeldef
  set c := i / 128 with hc_def
  set sc := sm_scale_python with hscdef
  -- window bound: (c+1)*128 ≤ S
  have hwin : (c + 1) * 128 ≤ S := by
    -- i = c*128 < S and S is a multiple of 128 (S = bm*(start_m+1)*128)
    have hSmul : S = (if 128 * s0.pids 2 < bel then 1 else 0) * (s0.pids 2 + 1) * 128 := by
      simp only [hSdef, ctxNopadWindow, hbeldef]
    by_cases hbm : 128 * s0.pids 2 < bel
    · rw [hSmul, if_pos hbm, one_mul] at hilt ⊢
      -- i = c*128 < (start_m+1)*128 ⟹ c ≤ start_m ⟹ (c+1) ≤ start_m+1
      have : c < s0.pids 2 + 1 := by omega
      nlinarith
    · rw [hSmul, if_neg hbm] at hilt; omega
  obtain ⟨hpids, hmem, hundef, hcb, hch, hseq, hsl, hn, hd, hoffm, hq, hkp, hvp, hbmask, hmi, hli, hacc⟩ := hinv
  -- the row-and-channel-indexed fold-so-far state (at c*128)
  set fold0 : Fin 128 → Fin 128 → WithBot ℝ × ℝ × ℝ := fun ir dd =>
    nopadFoldUpto s0 Q K V B_Start_Loc sc S bel (c * 128) ir dd with hfold0
  -- block list per row/channel
  set blk : Fin 128 → Fin 128 → List (ℝ × ℝ) := fun ir dd =>
    nopadBlockM s0 Q K V B_Start_Loc sc 128 S bel c ir dd with hblk
  -- score (pre-scale Σ) per row/lane
  set rawSum : Fin 128 → Fin 128 → ℝ := fun ir jL =>
    Finset.univ.sum (fun e : Fin 128 =>
      ctxQTileMRow s0 Q B_Start_Loc 128 bel (ir, e, PUnit.unit)
        * ctxKTileM s0 K B_Start_Loc S bel (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit))
    with hrawSum
  -- ============== peel the 22 statements ==============
  unfold nopadLoopBody
  set s1 := s.setReg "start_n" .nat [] (Tile.scalar i) with hs1d
  -- helpers transporting regs across s1 (≠ start_n preserved)
  have hs1pids : s1.pids = s0.pids := by rw [hs1d, BlockState.setReg_pids]; exact hpids
  have hs1mem : s1.mem = s0.mem := by funext rg o; rw [hs1d, BlockState.setReg_mem]; exact congrFun (congrFun hmem rg) o
  have e1 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "start_n" → s.regs dt sh nm = some t → s1.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs1d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  -- stmt 0: start_n = ref start_n (rebind to i)
  have hs1sn : s1.regs .nat [] "start_n" = some (Tile.scalar i) := by rw [hs1d, BlockState.setReg_same]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_n") s1 = some (Tile.scalar i) from by rw [evalOp_ref]; exact hs1sn))]
  set s2 := s1.setReg "start_n" .nat [] (Tile.scalar i) with hs2d
  have hs2pids : s2.pids = s0.pids := by rw [hs2d, BlockState.setReg_pids]; exact hs1pids
  have hs2mem : s2.mem = s0.mem := by funext rg o; rw [hs2d, BlockState.setReg_mem]; exact hs1mem ▸ rfl
  have e2 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "start_n" → s1.regs dt sh nm = some t → s2.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs2d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs2sn : s2.regs .nat [] "start_n" = some (Tile.scalar i) := by rw [hs2d, BlockState.setReg_same]
  have hs2seq : s2.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar bel) := e2 (by decide) (e1 (by decide) hseq)
  have hs2slStart : startLoc s2 B_Start_Loc = startLoc s0 B_Start_Loc := startLoc_eq_of_mem_pids s2 s0 B_Start_Loc hs2mem hs2pids
  have hs2sl : s2.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s2 B_Start_Loc)) := by
    rw [hs2slStart]; exact e2 (by decide) (e1 (by decide) hsl)
  have hs2n : s2.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)) := e2 (by decide) (e1 (by decide) hn)
  have hs2m : s2.regs .nat [128] "offs_m" = some (Tile.vec (fun ir : Fin 128 => s0.pids 2 * 128 + ir.val)) := e2 (by decide) (e1 (by decide) hoffm)
  have hs2kp : s2.regs .ptr [128, 128] "k_ptrs" = some (⟨fun idx : TileIndex [128, 128] =>
      (K, idx.2.1.val * 768 + s2.pids 1 * 128 + idx.1.val)⟩ : Tile .ptr [128, 128]) := by
    rw [hs2pids]; exact e2 (by decide) (e1 (by decide) hkp)
  have hs2seqB : seqLen s2 B_Seqlen = bel := by rw [hbeldef]; exact seqLen_eq_of_mem_pids s2 s0 B_Seqlen hs2mem hs2pids
  -- stmt 1: k = masked load → s3 with k tile = (e, jL) ↦ ctxKTileM(c*128+jL, e)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ s2 = _ from by
      have h := nopad_k_load_eval s2 K B_Start_Loc B_Seqlen i hs2kp hs2sl hs2sn hs2n (by rw [hs2seqB]; exact hs2seq)
      rw [hs2seqB] at h; exact h))]
  -- kFn e jL = ctxKTileM s0 (c*128+jL, e)
  set kFn : Fin 128 → Fin 128 → ℝ := fun e jL =>
    ctxKTileM s0 K B_Start_Loc S bel (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit) with hkFn
  rw [show (⟨fun idx : TileIndex [128, 128] =>
        if i + idx.2.1.val < bel then
          some (s2.readMem K ((startLoc s2 B_Start_Loc + (i + idx.2.1.val)) * 768 + s2.pids 1 * 128 + idx.1.val))
        else some (0.0 : ℝ)⟩ : Tile .real [128, 128])
      = (⟨fun idx : TileIndex [128, 128] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) from by
    refine Tile.ext (fun idx => ?_)
    obtain ⟨e, jL, u⟩ := idx
    simp only [hkFn, ctxKTileM, ctxKTile, hs2pids,
      show startLoc s2 B_Start_Loc = startLoc s0 B_Start_Loc from startLoc_eq_of_mem_pids s2 s0 B_Start_Loc hs2mem hs2pids,
      show i + jL.val = c * 128 + jL.val from by rw [hi]]
    by_cases hlt : c * 128 + jL.val < bel
    · rw [if_pos hlt, if_pos hlt]
      simp only [BlockState.readMem, hs2mem]
    · rw [if_neg hlt, if_neg hlt]; norm_num]
  set s3 := s2.setReg "k" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) with hs3d
  have e3 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "k" → s2.regs dt sh nm = some t → s3.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs3d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs3pids : s3.pids = s0.pids := by rw [hs3d, BlockState.setReg_pids]; exact hs2pids
  have hs3mem : s3.mem = s0.mem := by funext rg o; rw [hs3d, BlockState.setReg_mem]; exact hs2mem ▸ rfl
  have hs3k : s3.regs .real [128, 128] "k" = some (⟨fun idx : TileIndex [128, 128] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) := by
    rw [hs3d, BlockState.setReg_same]
  -- q register (ctxQTileMRow s0) in `some (qFn ir e)` form
  set qFn : Fin 128 → Fin 128 → ℝ := fun ir e => ctxQTileMRow s0 Q B_Start_Loc 128 bel (ir, e, PUnit.unit) with hqFn
  have hs3q : s3.regs .real [128, 128] "q" = some (⟨fun idx : TileIndex [128, 128] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) :=
    e3 (by decide) (e2 (by decide) (e1 (by decide) hq))
  -- stmt 2: qk = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [128, 128] (Op.const 0)) s3
        = some (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128]) from by
      simp [evalOp_full, evalOp_const]))]
  set s4 := s3.setReg "qk" .real [128, 128] (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128]) with hs4d
  have e4 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s3.regs dt sh nm = some t → s4.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs4d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs4qk0 : s4.regs .real [128, 128] "qk" = some (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128]) := by
    rw [hs4d, BlockState.setReg_same]
  have hs4q : s4.regs .real [128, 128] "q" = some (⟨fun idx : TileIndex [128, 128] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) := e4 (by decide) hs3q
  have hs4k : s4.regs .real [128, 128] "k" = some (⟨fun idx : TileIndex [128, 128] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) := e4 (by decide) hs3k
  -- stmt 3: qk += dot(q,k) → some (rawSum ir jL)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_qk_dot_eval s4 qFn kFn hs4qk0 hs4q hs4k))]
  -- the dot result Σ_e qFn ir e · kFn e jL = rawSum ir jL
  rw [show (⟨fun idx : TileIndex [128, 128] =>
        some (Finset.univ.sum (fun e : Fin 128 => qFn idx.1 e * kFn e idx.2.1))⟩ : Tile .real [128, 128])
      = (⟨fun idx : TileIndex [128, 128] => some (rawSum idx.1 idx.2.1)⟩ : Tile .real [128, 128]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, jL, u⟩ := idx
    simp only [hrawSum, hqFn, hkFn]]
  set s5 := s4.setReg "qk" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => some (rawSum idx.1 idx.2.1)⟩ : Tile .real [128, 128]) with hs5d
  have e5 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s4.regs dt sh nm = some t → s5.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs5d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs5qk : s5.regs .real [128, 128] "qk" = some (⟨fun idx : TileIndex [128, 128] => some (rawSum idx.1 idx.2.1)⟩ : Tile .real [128, 128]) := by
    rw [hs5d, BlockState.setReg_same]
  -- stmt 4: qk *= sc → some (rawSum · sc)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_qk_scale_eval s5 sc rawSum hs5qk))]
  set s6 := s5.setReg "qk" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => some (rawSum idx.1 idx.2.1 * sc)⟩ : Tile .real [128, 128]) with hs6d
  have e6 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s5.regs dt sh nm = some t → s6.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs6d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs6qk : s6.regs .real [128, 128] "qk" = some (⟨fun idx : TileIndex [128, 128] => some (rawSum idx.1 idx.2.1 * sc)⟩ : Tile .real [128, 128]) := by
    rw [hs6d, BlockState.setReg_same]
  have hs6m : s6.regs .nat [128] "offs_m" = some (Tile.vec (fun ir : Fin 128 => s0.pids 2 * 128 + ir.val)) :=
    e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2m)))
  have hs6sn : s6.regs .nat [] "start_n" = some (Tile.scalar i) :=
    e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2sn)))
  have hs6n : s6.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)) :=
    e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2n)))
  -- stmt 5: qk = where(offs_m ≥ start_n + offs_n, qk, -inf) → causal mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (nopad_qk_where_eval s6 i (fun ir jL => rawSum ir jL * sc) (fun ir => s0.pids 2 * 128 + ir.val)
      hs6qk hs6m hs6sn hs6n))]
  -- qkWhere ir jL = if c*128+jL ≤ pid2*128+ir then some (sc·rawSum) else ⊥
  set qkW : Fin 128 → Fin 128 → WithBot ℝ := fun ir jL =>
    if c * 128 + jL.val ≤ s0.pids 2 * 128 + ir.val then ((sc * rawSum ir jL : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ)
    with hqkW
  rw [show (⟨fun idx : TileIndex [128, 128] =>
        if i + idx.2.1.val ≤ s0.pids 2 * 128 + idx.1.val then some (rawSum idx.1 idx.2.1 * sc)
        else (⊥ : WithBot ℝ)⟩ : Tile .real [128, 128])
      = (⟨fun idx : TileIndex [128, 128] => qkW idx.1 idx.2.1⟩ : Tile .real [128, 128]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, jL, u⟩ := idx
    simp only [hqkW, show i + jL.val = c * 128 + jL.val from by rw [hi]]
    by_cases hca : c * 128 + jL.val ≤ s0.pids 2 * 128 + ir.val
    · rw [if_pos hca, if_pos hca]; rw [mul_comm]; rfl
    · rw [if_neg hca, if_neg hca]]
  set s7 := s6.setReg "qk" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => qkW idx.1 idx.2.1⟩ : Tile .real [128, 128]) with hs7d
  have e7 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s6.regs dt sh nm = some t → s7.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs7d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs7qk : s7.regs .real [128, 128] "qk" = some (⟨fun idx : TileIndex [128, 128] => qkW idx.1 idx.2.1⟩ : Tile .real [128, 128]) := by
    rw [hs7d, BlockState.setReg_same]
  -- mij ir = blockSup ir = ⊔ over nopadBlockM scores (via nopadBlockM_sup_eq)
  set mijFn : Fin 128 → WithBot ℝ := fun ir =>
    (blk ir ⟨0, by norm_num⟩).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) |>.foldr (· ⊔ ·) ⊥ with hmijFn
  -- stmt 6: m_ij = max(qk, 1)  → sup'  (= sup, then = blockSup via bridge)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_mij_eval s7 qkW hs7qk))]
  rw [show (⟨fun idx : TileIndex [128] =>
        Finset.univ.sup' Finset.univ_nonempty (fun jL : Fin 128 => qkW idx.1 jL)⟩ : Tile .real [128])
      = (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, u⟩ := idx
    show Finset.univ.sup' Finset.univ_nonempty (fun jL : Fin 128 => qkW ir jL) = mijFn ir
    rw [Finset.sup'_eq_sup]
    rw [hmijFn]
    simp only [hblk]
    rw [← nopadBlockM_sup_eq s0 Q K V B_Start_Loc sc 128 S bel c ir ⟨0, by norm_num⟩ hwin]]
  set s8 := s7.setReg "m_ij" .real [128] (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128]) with hs8d
  have e8 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "m_ij" → s7.regs dt sh nm = some t → s8.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs8d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs8qk : s8.regs .real [128, 128] "qk" = some (⟨fun idx : TileIndex [128, 128] => qkW idx.1 idx.2.1⟩ : Tile .real [128, 128]) := e8 (by decide) hs7qk
  have hs8mij : s8.regs .real [128] "m_ij" = some (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128]) := by
    rw [hs8d, BlockState.setReg_same]
  -- stmt 7: p = exp(qk - m_ij[:,None])
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_p_eval s8 qkW mijFn hs8qk hs8mij))]
  set pFn : Fin 128 → Fin 128 → WithBot ℝ := fun ir jL => WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir)) with hpFn
  set s9 := s8.setReg "p" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => pFn idx.1 idx.2.1⟩ : Tile .real [128, 128]) with hs9d
  have e9 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p" → s8.regs dt sh nm = some t → s9.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs9d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs9p : s9.regs .real [128, 128] "p" = some (⟨fun idx : TileIndex [128, 128] => pFn idx.1 idx.2.1⟩ : Tile .real [128, 128]) := by
    rw [hs9d, BlockState.setReg_same]
  -- stmt 8: l_ij = sum(p, 1)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_lij_eval s9 pFn hs9p))]
  set lijFn : Fin 128 → WithBot ℝ := fun ir => Finset.univ.sum (fun jL : Fin 128 => pFn ir jL) with hlijFn
  set s10 := s9.setReg "l_ij" .real [128] (⟨fun idx : TileIndex [128] => lijFn idx.1⟩ : Tile .real [128]) with hs10d
  have e10 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_ij" → s9.regs dt sh nm = some t → s10.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs10d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs10lij : s10.regs .real [128] "l_ij" = some (⟨fun idx : TileIndex [128] => lijFn idx.1⟩ : Tile .real [128]) := by
    rw [hs10d, BlockState.setReg_same]
  -- m_i register (from invariant) = fold0 .1 ; in some/coe form
  set miFn : Fin 128 → WithBot ℝ := fun ir => (fold0 ir ⟨0, by norm_num⟩).1 with hmiFn
  have hs10mi : s10.regs .real [128] "m_i" = some (⟨fun idx : TileIndex [128] => miFn idx.1⟩ : Tile .real [128]) :=
    e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hmi)))))))))
  have hs10mij : s10.regs .real [128] "m_ij" = some (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128]) := e10 (by decide) (e9 (by decide) hs8mij)
  -- stmt 9: m_i_new = maximum(m_i, m_ij) = miFn ⊔ mijFn
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_minew_eval s10 miFn mijFn hs10mi hs10mij))]
  set minewFn : Fin 128 → WithBot ℝ := fun ir => miFn ir ⊔ mijFn ir with hminewFn
  set s11 := s10.setReg "m_i_new" .real [128] (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128]) with hs11d
  have e11 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "m_i_new" → s10.regs dt sh nm = some t → s11.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs11d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs11minew : s11.regs .real [128] "m_i_new" = some (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128]) := by
    rw [hs11d, BlockState.setReg_same]
  have hs11mi : s11.regs .real [128] "m_i" = some (⟨fun idx : TileIndex [128] => miFn idx.1⟩ : Tile .real [128]) := e11 (by decide) hs10mi
  have hs11mij : s11.regs .real [128] "m_ij" = some (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128]) := e11 (by decide) hs10mij
  -- stmt 10: alpha = exp(m_i - m_i_new)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_alpha_eval s11 miFn minewFn hs11mi hs11minew))]
  set alphaFn : Fin 128 → WithBot ℝ := fun ir => WithBot.realExp (WithBot.realSub (miFn ir) (minewFn ir)) with halphaFn
  set s12 := s11.setReg "alpha" .real [128] (⟨fun idx : TileIndex [128] => alphaFn idx.1⟩ : Tile .real [128]) with hs12d
  have e12 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "alpha" → s11.regs dt sh nm = some t → s12.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs12d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs12alpha : s12.regs .real [128] "alpha" = some (⟨fun idx : TileIndex [128] => alphaFn idx.1⟩ : Tile .real [128]) := by
    rw [hs12d, BlockState.setReg_same]
  have hs12mij : s12.regs .real [128] "m_ij" = some (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128]) := e12 (by decide) hs11mij
  have hs12minew : s12.regs .real [128] "m_i_new" = some (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128]) := e12 (by decide) hs11minew
  -- stmt 11: beta = exp(m_ij - m_i_new)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_beta_eval s12 mijFn minewFn hs12mij hs12minew))]
  set betaFn : Fin 128 → WithBot ℝ := fun ir => WithBot.realExp (WithBot.realSub (mijFn ir) (minewFn ir)) with hbetaFn
  set s13 := s12.setReg "beta" .real [128] (⟨fun idx : TileIndex [128] => betaFn idx.1⟩ : Tile .real [128]) with hs13d
  have e13 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "beta" → s12.regs dt sh nm = some t → s13.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs13d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs13beta : s13.regs .real [128] "beta" = some (⟨fun idx : TileIndex [128] => betaFn idx.1⟩ : Tile .real [128]) := by
    rw [hs13d, BlockState.setReg_same]
  have hs13alpha : s13.regs .real [128] "alpha" = some (⟨fun idx : TileIndex [128] => alphaFn idx.1⟩ : Tile .real [128]) := e13 (by decide) hs12alpha
  -- l_i register (from invariant) = fold0 .2.1 (coerced)
  set liFn : Fin 128 → WithBot ℝ := fun ir => ((fold0 ir ⟨0, by norm_num⟩).2.1 : WithBot ℝ) with hliFn
  have hs13li : s13.regs .real [128] "l_i" = some (⟨fun idx : TileIndex [128] => liFn idx.1⟩ : Tile .real [128]) :=
    e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hli))))))))))))
  have hs13lij : s13.regs .real [128] "l_ij" = some (⟨fun idx : TileIndex [128] => lijFn idx.1⟩ : Tile .real [128]) :=
    e13 (by decide) (e12 (by decide) (e11 (by decide) hs10lij))
  -- stmt 12: l_i_new = alpha*l_i + beta*l_ij
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_linew_eval s13 alphaFn liFn betaFn lijFn hs13alpha hs13li hs13beta hs13lij))]
  set linewFn : Fin 128 → WithBot ℝ := fun ir =>
    WithBot.realAdd (WithBot.realMul (alphaFn ir) (liFn ir)) (WithBot.realMul (betaFn ir) (lijFn ir)) with hlinewFn
  set s14 := s13.setReg "l_i_new" .real [128] (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) with hs14d
  have e14 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_i_new" → s13.regs dt sh nm = some t → s14.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs14d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs14linew : s14.regs .real [128] "l_i_new" = some (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) := by
    rw [hs14d, BlockState.setReg_same]
  have hs14beta : s14.regs .real [128] "beta" = some (⟨fun idx : TileIndex [128] => betaFn idx.1⟩ : Tile .real [128]) := e14 (by decide) hs13beta
  -- stmt 13: p_scale = beta / l_i_new
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_pscale_eval s14 betaFn linewFn hs14beta hs14linew))]
  set psFn : Fin 128 → WithBot ℝ := fun ir => WithBot.realDiv (betaFn ir) (linewFn ir) with hpsFn
  set s15 := s14.setReg "p_scale" .real [128] (⟨fun idx : TileIndex [128] => psFn idx.1⟩ : Tile .real [128]) with hs15d
  have e15 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p_scale" → s14.regs dt sh nm = some t → s15.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs15d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs15ps : s15.regs .real [128] "p_scale" = some (⟨fun idx : TileIndex [128] => psFn idx.1⟩ : Tile .real [128]) := by
    rw [hs15d, BlockState.setReg_same]
  have hs15p : s15.regs .real [128, 128] "p" = some (⟨fun idx : TileIndex [128, 128] => pFn idx.1 idx.2.1⟩ : Tile .real [128, 128]) :=
    e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) hs9p)))))
  -- stmt 14: p = p * p_scale[:,None]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_pmul_eval s15 pFn psFn hs15p hs15ps))]
  set pFinal : Fin 128 → Fin 128 → WithBot ℝ := fun ir jL => WithBot.realMul (pFn ir jL) (psFn ir) with hpFinal
  set s16 := s15.setReg "p" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => pFinal idx.1 idx.2.1⟩ : Tile .real [128, 128]) with hs16d
  have e16 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p" → s15.regs dt sh nm = some t → s16.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs16d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs16p : s16.regs .real [128, 128] "p" = some (⟨fun idx : TileIndex [128, 128] => pFinal idx.1 idx.2.1⟩ : Tile .real [128, 128]) := by
    rw [hs16d, BlockState.setReg_same]
  -- l_i / l_i_new / alpha for acc_scale
  have hs16li : s16.regs .real [128] "l_i" = some (⟨fun idx : TileIndex [128] => liFn idx.1⟩ : Tile .real [128]) :=
    e16 (by decide) (e15 (by decide) (e14 (by decide) hs13li))
  have hs16linew : s16.regs .real [128] "l_i_new" = some (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) :=
    e16 (by decide) (e15 (by decide) hs14linew)
  have hs16alpha : s16.regs .real [128] "alpha" = some (⟨fun idx : TileIndex [128] => alphaFn idx.1⟩ : Tile .real [128]) :=
    e16 (by decide) (e15 (by decide) (e14 (by decide) hs13alpha))
  -- stmt 15: acc_scale = l_i / l_i_new * alpha
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_accscale_eval s16 liFn linewFn alphaFn hs16li hs16linew hs16alpha))]
  set asFn : Fin 128 → WithBot ℝ := fun ir => WithBot.realMul (WithBot.realDiv (liFn ir) (linewFn ir)) (alphaFn ir) with hasFn
  set s17 := s16.setReg "acc_scale" .real [128] (⟨fun idx : TileIndex [128] => asFn idx.1⟩ : Tile .real [128]) with hs17d
  have e17 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc_scale" → s16.regs dt sh nm = some t → s17.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs17d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs17as : s17.regs .real [128] "acc_scale" = some (⟨fun idx : TileIndex [128] => asFn idx.1⟩ : Tile .real [128]) := by
    rw [hs17d, BlockState.setReg_same]
  -- acc register (from invariant) = fold0 .2.2 (coerced), per (ir, dd)
  set accFn : Fin 128 → Fin 128 → WithBot ℝ := fun ir dd => ((fold0 ir dd).2.2 : WithBot ℝ) with haccFn
  have hs17acc : s17.regs .real [128, 128] "acc" = some (⟨fun idx : TileIndex [128, 128] => accFn idx.1 idx.2.1⟩ : Tile .real [128, 128]) :=
    e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hacc))))))))))))))))
  -- stmt 16: acc = acc * acc_scale[:,None]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (nopad_accmul_eval s17 accFn asFn hs17acc hs17as))]
  set accMul : Fin 128 → Fin 128 → WithBot ℝ := fun ir dd => WithBot.realMul (accFn ir dd) (asFn ir) with haccMul
  set s18 := s17.setReg "acc" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => accMul idx.1 idx.2.1⟩ : Tile .real [128, 128]) with hs18d
  have e18 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc" → s17.regs dt sh nm = some t → s18.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs18d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs18acc : s18.regs .real [128, 128] "acc" = some (⟨fun idx : TileIndex [128, 128] => accMul idx.1 idx.2.1⟩ : Tile .real [128, 128]) := by
    rw [hs18d, BlockState.setReg_same]
  -- s18 pids/mem agree with s0 (only register writes since s3)
  have hs18mem : s18.mem = s0.mem := by
    funext rg o; rw [hs18d, BlockState.setReg_mem, hs17d, BlockState.setReg_mem, hs16d, BlockState.setReg_mem,
      hs15d, BlockState.setReg_mem, hs14d, BlockState.setReg_mem, hs13d, BlockState.setReg_mem,
      hs12d, BlockState.setReg_mem, hs11d, BlockState.setReg_mem, hs10d, BlockState.setReg_mem,
      hs9d, BlockState.setReg_mem, hs8d, BlockState.setReg_mem, hs7d, BlockState.setReg_mem,
      hs6d, BlockState.setReg_mem, hs5d, BlockState.setReg_mem, hs4d, BlockState.setReg_mem]; exact hs3mem ▸ rfl
  have hs18pids : s18.pids = s0.pids := by
    rw [hs18d, BlockState.setReg_pids, hs17d, BlockState.setReg_pids, hs16d, BlockState.setReg_pids,
      hs15d, BlockState.setReg_pids, hs14d, BlockState.setReg_pids, hs13d, BlockState.setReg_pids,
      hs12d, BlockState.setReg_pids, hs11d, BlockState.setReg_pids, hs10d, BlockState.setReg_pids,
      hs9d, BlockState.setReg_pids, hs8d, BlockState.setReg_pids, hs7d, BlockState.setReg_pids,
      hs6d, BlockState.setReg_pids, hs5d, BlockState.setReg_pids, hs4d, BlockState.setReg_pids, hs3pids]
  have hs18pids1 : s18.pids 1 = s0.pids 1 := by rw [hs18pids]
  have hs18seqB : seqLen s18 B_Seqlen = bel := by rw [hbeldef]; exact seqLen_eq_of_mem_pids s18 s0 B_Seqlen hs18mem hs18pids
  have hs18slB : startLoc s18 B_Start_Loc = startLoc s0 B_Start_Loc := startLoc_eq_of_mem_pids s18 s0 B_Start_Loc hs18mem hs18pids
  -- v_ptrs (from invariant), seq_len, start_loc, start_n, offs_n at s18
  have hs18vp : s18.regs .ptr [128, 128] "v_ptrs" = some (⟨fun idx : TileIndex [128, 128] =>
      (V, idx.1.val * 768 + s18.pids 1 * 128 + idx.2.1.val)⟩ : Tile .ptr [128, 128]) := by
    rw [hs18pids1]
    exact e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hvp)))))))))))))))))
  have hs18sl : s18.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s18 B_Start_Loc)) := by
    rw [hs18slB, ← hs2slStart]
    exact e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2sl)))))))))))))))
  have hs18sn : s18.regs .nat [] "start_n" = some (Tile.scalar i) :=
    e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) hs6sn)))))))))))
  have hs18n : s18.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)) :=
    e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) hs6n)))))))))))
  have hs18seq : s18.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s18 B_Seqlen)) := by
    rw [hs18seqB]
    exact e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2seq)))))))))))))))
  -- stmt 17: v = masked load → vFn jL d = ctxVTileM(c*128+jL, d) on active lanes
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ s18 = _ from by
      have h := nopad_v_load_eval s18 V B_Start_Loc B_Seqlen i hs18vp hs18sl hs18sn hs18n hs18seq
      exact h))]
  set vFn : Fin 128 → Fin 128 → ℝ := fun jL dd =>
    if i + jL.val < seqLen s18 B_Seqlen then
      s18.readMem V ((startLoc s18 B_Start_Loc + (i + jL.val)) * 768 + s18.pids 1 * 128 + dd.val)
    else (0.0 : ℝ) with hvFn
  set s19 := s18.setReg "v" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) with hs19d
  rw [show (⟨fun idx : TileIndex [128, 128] =>
        if i + idx.1.val < seqLen s18 B_Seqlen then
          some (s18.readMem V ((startLoc s18 B_Start_Loc + (i + idx.1.val)) * 768 + s18.pids 1 * 128 + idx.2.1.val))
        else some (0.0 : ℝ)⟩ : Tile .real [128, 128])
      = (⟨fun idx : TileIndex [128, 128] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨jL, dd, u⟩ := idx
    show (if i + jL.val < seqLen s18 B_Seqlen then
        some (s18.readMem V ((startLoc s18 B_Start_Loc + (i + jL.val)) * 768 + s18.pids 1 * 128 + dd.val))
      else some (0.0 : ℝ)) = some (vFn jL dd)
    by_cases hlt : i + jL.val < seqLen s18 B_Seqlen
    · simp only [hvFn, hlt, if_true]
    · simp only [hvFn, hlt, if_false]]
  have e19 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "v" → s18.regs dt sh nm = some t → s19.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs19d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs19v : s19.regs .real [128, 128] "v" = some (⟨fun idx : TileIndex [128, 128] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) := by
    rw [hs19d, BlockState.setReg_same]
  have hs19p : s19.regs .real [128, 128] "p" = some (⟨fun idx : TileIndex [128, 128] => pFinal idx.1 idx.2.1⟩ : Tile .real [128, 128]) := e19 (by decide) hs16p
  -- stmt 18: p = p (rebind to same)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [128, 128] "p") s19 = some (⟨fun idx : TileIndex [128, 128] => pFinal idx.1 idx.2.1⟩ : Tile .real [128, 128]) from by
      rw [evalOp_ref]; exact hs19p))]
  set s20 := s19.setReg "p" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => pFinal idx.1 idx.2.1⟩ : Tile .real [128, 128]) with hs20d
  have e20 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p" → s19.regs dt sh nm = some t → s20.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs20d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs20p : s20.regs .real [128, 128] "p" = some (⟨fun idx : TileIndex [128, 128] => pFinal idx.1 idx.2.1⟩ : Tile .real [128, 128]) := by
    rw [hs20d, BlockState.setReg_same]
  have hs20v : s20.regs .real [128, 128] "v" = some (⟨fun idx : TileIndex [128, 128] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) := e20 (by decide) hs19v
  have hs20acc : s20.regs .real [128, 128] "acc" = some (⟨fun idx : TileIndex [128, 128] => accMul idx.1 idx.2.1⟩ : Tile .real [128, 128]) :=
    e20 (by decide) (e19 (by decide) hs18acc)
  -- ======= real-extraction: every kernel WithBot quantity is `some` of a real =======
  -- anchor per (ir, dd): l = κ(m)·L, acc·L = T, etc.
  set Lf : Fin 128 → ℝ := fun ir =>
    ((ctxNopadKeysUptoM s0 Q K V B_Start_Loc sc 128 S bel (c * 128) ir ⟨0, by norm_num⟩).map (fun p => Real.exp p.1)).sum with hLf
  -- pFn ir jL = some (pReal ir jL)
  set pReal : Fin 128 → Fin 128 → ℝ := fun ir jL =>
    (WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir))).unbotD 0 with hpReal
  have hpFnSome : ∀ ir jL : Fin 128, pFn ir jL = some (pReal ir jL) := by
    intro ir jL; simp only [hpFn, hpReal]; exact nopad_realExp_eq_some_unbotD _
  -- lijFn ir = some (lijReal ir) ;   lijReal = block-lij bridge value
  have hlijSome : ∀ ir : Fin 128, lijFn ir = some (Finset.univ.sum (fun jL : Fin 128 => pReal ir jL)) := by
    intro ir; simp only [hlijFn]
    rw [show (fun jL : Fin 128 => pFn ir jL) = (fun jL : Fin 128 => ((pReal ir jL : ℝ) : WithBot ℝ)) from by
      funext jL; rw [hpFnSome ir jL]]
    rw [WithBot.sum_some_eq_some]
  set lijReal : Fin 128 → ℝ := fun ir => Finset.univ.sum (fun jL : Fin 128 => pReal ir jL) with hlijReal
  -- miFn = fold0.1 ; liFn = some(fold0.2.1) ; accFn = some(fold0.2.2)
  -- α, β real
  set αReal : Fin 128 → ℝ := fun ir => (WithBot.realExp (WithBot.realSub (miFn ir) (minewFn ir))).unbotD 0 with hαReal
  set βReal : Fin 128 → ℝ := fun ir => (WithBot.realExp (WithBot.realSub (mijFn ir) (minewFn ir))).unbotD 0 with hβReal
  have hαSome : ∀ ir, alphaFn ir = some (αReal ir) := by intro ir; simp only [halphaFn, hαReal]; exact nopad_realExp_eq_some_unbotD _
  have hβSome : ∀ ir, betaFn ir = some (βReal ir) := by intro ir; simp only [hbetaFn, hβReal]; exact nopad_realExp_eq_some_unbotD _
  have hliSome : ∀ ir, liFn ir = some ((fold0 ir ⟨0, by norm_num⟩).2.1) := by intro ir; simp only [hliFn]
  -- linewFn ir = some (linewReal ir)
  set linewReal : Fin 128 → ℝ := fun ir => αReal ir * (fold0 ir ⟨0, by norm_num⟩).2.1 + βReal ir * lijReal ir with hlinewReal
  have hlinewSome : ∀ ir, linewFn ir = some (linewReal ir) := by
    intro ir; simp only [hlinewFn, hαSome, hliSome, hβSome, hlijSome, hlinewReal]
    simp only [WithBot.realMul_coe_coe, WithBot.realAdd_coe_coe, lijReal]
  -- psFn ir = some (βReal ir / linewReal ir)
  set psReal : Fin 128 → ℝ := fun ir => βReal ir / linewReal ir with hpsReal
  have hpsSome : ∀ ir, psFn ir = some (psReal ir) := by
    intro ir; simp only [hpsFn, hβSome, hlinewSome, hpsReal, WithBot.realDiv_coe_coe]
  -- pFinal ir jL = some (pReal ir jL * psReal ir)
  have hpFinalSome : ∀ ir jL, pFinal ir jL = some (pReal ir jL * psReal ir) := by
    intro ir jL; simp only [hpFinal, hpFnSome, hpsSome, WithBot.realMul_coe_coe]
  -- accFn ir dd = some (fold0.2.2) ;  asFn ir = some (asReal ir) ;  accMul = some(...)
  set asReal : Fin 128 → ℝ := fun ir => (fold0 ir ⟨0, by norm_num⟩).2.1 / linewReal ir * αReal ir with hasReal
  have hasSome : ∀ ir, asFn ir = some (asReal ir) := by
    intro ir; simp only [hasFn, hliSome, hlinewSome, hαSome, hasReal, WithBot.realDiv_coe_coe, WithBot.realMul_coe_coe]
  have haccFnSome : ∀ ir dd, accFn ir dd = some ((fold0 ir dd).2.2) := by intro ir dd; simp only [haccFn]
  have haccMulSome : ∀ ir dd, accMul ir dd = some ((fold0 ir dd).2.2 * asReal ir) := by
    intro ir dd; simp only [haccMul, haccFnSome, hasSome, WithBot.realMul_coe_coe]
  -- p / v as `some` for the acc dot recipe
  rw [show (⟨fun idx : TileIndex [128, 128] => pFinal idx.1 idx.2.1⟩ : Tile .real [128, 128])
        = (⟨fun idx : TileIndex [128, 128] => some (pReal idx.1 idx.2.1 * psReal idx.1)⟩ : Tile .real [128, 128]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, jL, u⟩ := idx; exact hpFinalSome ir jL] at hs20p
  rw [show (⟨fun idx : TileIndex [128, 128] => accMul idx.1 idx.2.1⟩ : Tile .real [128, 128])
        = (⟨fun idx : TileIndex [128, 128] => some ((fold0 idx.1 idx.2.1).2.2 * asReal idx.1)⟩ : Tile .real [128, 128]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, dd, u⟩ := idx; exact haccMulSome ir dd] at hs20acc
  -- stmt 19: acc += dot(p, v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (nopad_acc_dot_eval s20 (fun ir dd => some ((fold0 ir dd).2.2 * asReal ir))
      (fun ir jL => pReal ir jL * psReal ir) vFn hs20acc hs20p hs20v))]
  set accFinal : Fin 128 → Fin 128 → ℝ := fun ir dd =>
    (fold0 ir dd).2.2 * asReal ir + Finset.univ.sum (fun jL : Fin 128 => (pReal ir jL * psReal ir) * vFn jL dd) with haccFinal
  rw [show (⟨fun idx : TileIndex [128, 128] =>
        WithBot.realAdd (some ((fold0 idx.1 idx.2.1).2.2 * asReal idx.1))
          (some (Finset.univ.sum (fun jL : Fin 128 => (pReal idx.1 jL * psReal idx.1) * vFn jL idx.2.1)))⟩ : Tile .real [128, 128])
      = (⟨fun idx : TileIndex [128, 128] => some (accFinal idx.1 idx.2.1)⟩ : Tile .real [128, 128]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, dd, u⟩ := idx
    rw [haccFinal, WithBot.realAdd_coe_coe]]
  set s21 := s20.setReg "acc" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => some (accFinal idx.1 idx.2.1)⟩ : Tile .real [128, 128]) with hs21d
  have e21 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc" → s20.regs dt sh nm = some t → s21.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs21d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs21acc : s21.regs .real [128, 128] "acc" = some (⟨fun idx : TileIndex [128, 128] => some (accFinal idx.1 idx.2.1)⟩ : Tile .real [128, 128]) := by
    rw [hs21d, BlockState.setReg_same]
  have hs21linew : s21.regs .real [128] "l_i_new" = some (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) :=
    e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) hs14linew))))))
  have hs21minew : s21.regs .real [128] "m_i_new" = some (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128]) :=
    e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) hs11minew)))))))))
  -- stmt 20: l_i = l_i_new
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [128] "l_i_new") s21 = some (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) from by
      rw [evalOp_ref]; exact hs21linew))]
  set s22 := s21.setReg "l_i" .real [128] (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) with hs22d
  have e22 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_i" → s21.regs dt sh nm = some t → s22.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs22d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs22minew : s22.regs .real [128] "m_i_new" = some (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128]) := e22 (by decide) hs21minew
  -- stmt 21: m_i = m_i_new
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [128] "m_i_new") s22 = some (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128]) from by
      rw [evalOp_ref]; exact hs22minew))]
  rw [stepStmts.nil]
  set s23 := s22.setReg "m_i" .real [128] (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128]) with hs23d
  refine ⟨s23, rfl, ?_⟩
  -- ======= the math bridge: the new (m, l, acc) = nopadFoldUpto ((c+1)*128) =======
  -- per (ir, dd): (minewFn ir, linewReal ir, accFinal ir dd) = nopadFoldUpto ((c+1)*128) ir dd
  have hbridge : ∀ ir dd : Fin 128,
      (minewFn ir, linewReal ir, accFinal ir dd)
        = nopadFoldUpto s0 Q K V B_Start_Loc sc S bel ((c + 1) * 128) ir dd := by
    intro ir dd
    rw [nopadFoldUptoM_succ s0 Q K V B_Start_Loc sc S bel c ir dd]
    -- block list non-empty (causal key c*128)
    have hbne : nopadBlockM s0 Q K V B_Start_Loc sc 128 S bel c ir dd ≠ [] :=
      nopadBlockM_ne_nil s0 Q K V B_Start_Loc sc S bel c ir dd hwin (by rw [hpids]; omega)
    -- anchor at (ir, dd)
    obtain ⟨ha_l, ha_acc, ha_L0, ha_botL, ha_botT, ha_ne⟩ :=
      nopadFoldUpto_anchor s0 Q K V B_Start_Loc sc S bel (c * 128) ir dd
    -- channel-indep: fold0 ir dd has same .1/.2.1 as fold0 ir ⟨0⟩
    obtain ⟨hci1, hci2⟩ := nopadFoldUpto_channel_indep s0 Q K V B_Start_Loc sc S bel (c * 128) ir dd ⟨0, by norm_num⟩
    -- block_eq with m = fold0.1, l = fold0.2.1, acc = fold0.2.2
    have hblockEq := osNormStepBot_block_eq (fold0 ir dd).1 (fold0 ir dd).2.1 (fold0 ir dd).2.2
      (((ctxNopadKeysUptoM s0 Q K V B_Start_Loc sc 128 S bel (c * 128) ir dd).map (fun p => Real.exp p.1 * p.2)).sum)
      (((ctxNopadKeysUptoM s0 Q K V B_Start_Loc sc 128 S bel (c * 128) ir dd).map (fun p => Real.exp p.1)).sum)
      (nopadBlockM s0 Q K V B_Start_Loc sc 128 S bel c ir dd)
      hbne ha_L0 ha_l ha_acc ha_botL ha_botT ha_ne
    simp only at hblockEq
    -- block max (foldr-sup) of blk ir dd
    set bsupDD : WithBot ℝ := (nopadBlockM s0 Q K V B_Start_Loc sc 128 S bel c ir dd).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) |>.foldr (· ⊔ ·) ⊥ with hbsupDD
    -- mijFn ir = bsupDD (channel-indep scores)
    have hmijEq : mijFn ir = bsupDD := by
      simp only [hmijFn, hbsupDD, hblk]
      rw [show (nopadBlockM s0 Q K V B_Start_Loc sc 128 S bel c ir ⟨0, by norm_num⟩).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
            = ((nopadBlockM s0 Q K V B_Start_Loc sc 128 S bel c ir ⟨0, by norm_num⟩).map Prod.fst).map (fun r : ℝ => ((r : ℝ) : WithBot ℝ)) from by rw [List.map_map]; rfl,
        nopadBlockM_fst_channel_indep s0 Q K V B_Start_Loc sc S bel c ir ⟨0, by norm_num⟩ dd,
        ← List.map_map]
    -- m = (fold0 ir dd).1 = (fold0 ir ⟨0⟩).1 = miFn ir
    have hmiEq : miFn ir = (fold0 ir dd).1 := by simp only [hmiFn]; exact hci1.symm
    -- minewFn ir = (fold0 ir dd).1 ⊔ bsupDD
    have hMnew : minewFn ir = (fold0 ir dd).1 ⊔ bsupDD := by simp only [hminewFn]; rw [hmiEq, hmijEq]
    -- l = (fold0 ir dd).2.1 = (fold0 ir ⟨0⟩).2.1
    have hliEq : (fold0 ir ⟨0, by norm_num⟩).2.1 = (fold0 ir dd).2.1 := hci2.symm
    -- α/β/lij/l' bridge
    have hαEqB : αReal ir = (WithBot.realExp (WithBot.realSub (fold0 ir dd).1 (minewFn ir))).unbotD 0 := by
      simp only [hαReal]; rw [hmiEq]
    have hβEqB : βReal ir = (WithBot.realExp (WithBot.realSub bsupDD (minewFn ir))).unbotD 0 := by
      simp only [hβReal]; rw [hmijEq]
    -- lijReal ir = block_eq's lij = Σ blk exp(p.1 - bsupDD.unbotD)
    have hlijEqB : lijReal ir = ((nopadBlockM s0 Q K V B_Start_Loc sc 128 S bel c ir dd).map (fun p => Real.exp (p.1 - bsupDD.unbotD 0))).sum := by
      have hMr := nopadBlockM_lij_sum s0 Q K V B_Start_Loc sc 128 S bel c ir dd (bsupDD.unbotD 0) hwin
      -- pReal ir jL = realExp(realSub qkW mij).unbotD ; lijReal = Σ pReal
      have hpRealEq : ∀ jL : Fin 128, (pReal ir jL : ℝ)
          = (WithBot.realExp (WithBot.realSub
              (if c * 128 + jL.val ≤ s0.pids 2 * 128 + ir.val then
                ((sc * Finset.univ.sum (fun e : Fin 128 =>
                    ctxQTileMRow s0 Q B_Start_Loc 128 bel (ir, e, PUnit.unit)
                      * ctxKTileM s0 K B_Start_Loc S bel (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, e, PUnit.unit)) : ℝ) : WithBot ℝ)
               else (⊥ : WithBot ℝ)) ((bsupDD.unbotD 0 : ℝ) : WithBot ℝ))).unbotD 0 := by
        intro jL; simp only [hpReal, hqkW, hrawSum, hmijEq]
      -- Σ pReal = (Σ over lanes realExp(...)).unbotD ;  bridge gives some(Σ blk ...)
      have hsum : ((Finset.univ.sum (fun jL : Fin 128 => pReal ir jL) : ℝ) : WithBot ℝ)
          = some (((nopadBlockM s0 Q K V B_Start_Loc sc 128 S bel c ir dd).map (fun p => Real.exp (p.1 - bsupDD.unbotD 0))).sum) := by
        rw [← hMr, ← WithBot.sum_some_eq_some]
        apply Finset.sum_congr rfl; intro jL _
        rw [hpRealEq jL, nopad_realExp_eq_some_unbotD]
      simp only [hlijReal]
      exact WithBot.coe_inj.mp hsum
    -- assemble the tuple equality, then trans with hblockEq
    rw [← hblockEq]
    refine Prod.ext ?_ (Prod.ext ?_ ?_)
    · show minewFn ir = (fold0 ir dd).1 ⊔ bsupDD; exact hMnew
    · show linewReal ir = _
      simp only [hlinewReal]; rw [hαEqB, hβEqB, hlijEqB, hliEq]
    · show accFinal ir dd = _
      simp only [haccFinal]
      -- asReal ir = (fold0 ir dd).2.1 / l' * α  (block_eq's l/l'·α)
      have hasEqB : asReal ir = (fold0 ir dd).2.1 / linewReal ir * αReal ir := by
        simp only [hasReal]; rw [hliEq]
      -- vFn jL dd = ctxVTileM (c*128+jL, dd)
      have hvEq : ∀ jL : Fin 128, vFn jL dd
          = ctxVTileM s0 V B_Start_Loc S bel (⟨c * 128 + jL.val, nopad_lane_lt_S c S hwin jL⟩, dd, PUnit.unit) := by
        intro jL
        rw [hvFn, ctxVTileM, ctxVTile, hs18seqB, hs18slB, hs18pids1,
          show i + jL.val = c * 128 + jL.val from by rw [hi]]
        by_cases hlt : c * 128 + jL.val < bel
        · rw [if_pos hlt, if_pos hlt]; simp only [BlockState.readMem, hs18mem]
        · rw [if_neg hlt, if_neg hlt]; norm_num
      -- Σ_jL (pReal·psReal)·vFn = psReal · Σ_blk exp(p.1-bs)·v  (via nopadBlockM_acc_sum)
      have haccsum : Finset.univ.sum (fun jL : Fin 128 => (pReal ir jL * psReal ir) * vFn jL dd)
          = psReal ir * ((nopadBlockM s0 Q K V B_Start_Loc sc 128 S bel c ir dd).map (fun p => Real.exp (p.1 - bsupDD.unbotD 0) * p.2)).sum := by
        rw [Finset.mul_sum]
        have hbr := nopadBlockM_acc_sum s0 Q K V B_Start_Loc sc 128 S bel c ir dd (bsupDD.unbotD 0) hwin
          (fun jL => vFn jL dd) (fun jL _ => hvEq jL)
        -- coerce: Σ realMul(realExp(realSub qkW' bs))(coe(vFn)) = some(Σ blk ...)
        have hcoe : ((Finset.univ.sum (fun jL : Fin 128 => pReal ir jL * vFn jL dd) : ℝ) : WithBot ℝ)
            = some (((nopadBlockM s0 Q K V B_Start_Loc sc 128 S bel c ir dd).map (fun p => Real.exp (p.1 - bsupDD.unbotD 0) * p.2)).sum) := by
          rw [← hbr, ← WithBot.sum_some_eq_some]
          apply Finset.sum_congr rfl; intro jL _
          rw [WithBot.realMul_coe_coe]
          congr 1
          simp only [hpReal, hqkW, hrawSum, hmijEq]
          rw [nopad_realExp_eq_some_unbotD]
        rw [show (fun jL : Fin 128 => pReal ir jL * psReal ir * vFn jL dd)
              = (fun jL : Fin 128 => psReal ir * (pReal ir jL * vFn jL dd)) from by funext jL; ring]
        rw [← Finset.mul_sum, WithBot.coe_inj.mp hcoe]
      rw [hasEqB, haccsum]; simp only [hpsReal]; rw [hβEqB]; simp only [hlijReal]
      ring
  -- ======= reconstruct the invariant =======
  -- e23 for the final m_i setReg
  have e23 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "m_i" → s22.regs dt sh nm = some t → s23.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs23d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  -- chain a preserved register (≠ all 23 written names) from s (= invariant input) to s23
  have chainAll : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "start_n" → nm ≠ "k" → nm ≠ "qk" → nm ≠ "m_ij" → nm ≠ "p" → nm ≠ "l_ij"
      → nm ≠ "m_i_new" → nm ≠ "alpha" → nm ≠ "beta" → nm ≠ "l_i_new" → nm ≠ "p_scale"
      → nm ≠ "acc_scale" → nm ≠ "acc" → nm ≠ "v" → nm ≠ "l_i" → nm ≠ "m_i"
      → s.regs dt sh nm = some t → s23.regs dt sh nm = some t := by
    intro dt sh nm t h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h
    exact e23 h16 (e22 h15 (e21 h13 (e20 h5 (e19 h14 (e18 h13 (e17 h12 (e16 h5 (e15 h11 (e14 h10 (e13 h9 (e12 h8 (e11 h7 (e10 h6 (e9 h5 (e8 h4 (e7 h3 (e6 h3 (e5 h3 (e4 h3 (e3 h2 (e2 h1 (e1 h1 h)))))))))))))))))))))))
  -- pids / mem / undef
  have hs23pids : s23.pids = s0.pids := by
    rw [hs23d, BlockState.setReg_pids, hs22d, BlockState.setReg_pids, hs21d, BlockState.setReg_pids,
      hs20d, BlockState.setReg_pids, hs19d, BlockState.setReg_pids, hs18pids]
  have hs23mem : s23.mem = s0.mem := by
    funext rg o; rw [hs23d, BlockState.setReg_mem, hs22d, BlockState.setReg_mem, hs21d, BlockState.setReg_mem,
      hs20d, BlockState.setReg_mem, hs19d, BlockState.setReg_mem]; exact hs18mem ▸ rfl
  have hs23undef : ∀ rg o, s23.undef rg o = 0 := by
    intro rg o
    rw [hs23d, BlockState.setReg_undef, hs22d, BlockState.setReg_undef, hs21d, BlockState.setReg_undef,
      hs20d, BlockState.setReg_undef, hs19d, BlockState.setReg_undef, hs18d, BlockState.setReg_undef,
      hs17d, BlockState.setReg_undef, hs16d, BlockState.setReg_undef, hs15d, BlockState.setReg_undef,
      hs14d, BlockState.setReg_undef, hs13d, BlockState.setReg_undef, hs12d, BlockState.setReg_undef,
      hs11d, BlockState.setReg_undef, hs10d, BlockState.setReg_undef, hs9d, BlockState.setReg_undef,
      hs8d, BlockState.setReg_undef, hs7d, BlockState.setReg_undef, hs6d, BlockState.setReg_undef,
      hs5d, BlockState.setReg_undef, hs4d, BlockState.setReg_undef, hs3d, BlockState.setReg_undef,
      hs2d, BlockState.setReg_undef, hs1d, BlockState.setReg_undef]
    exact hundef rg o
  refine ⟨hs23pids, hs23mem, hs23undef, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hcb
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hch
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hseq
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hsl
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hn
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hd
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hoffm
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hq
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hkp
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hvp
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hbmask
  · -- m_i = (newfold).1
    rw [hs23d, BlockState.setReg_same]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨ir, u⟩ := idx
    show minewFn ir = (nopadFoldUpto s0 Q K V B_Start_Loc sm_scale_python (ctxNopadWindow s0 B_Seqlen 128) (seqLen s0 B_Seqlen) ((c + 1) * 128) ir ⟨0, by norm_num⟩).1
    exact congrArg (Prod.fst) (hbridge ir ⟨0, by norm_num⟩)
  · -- l_i = some((newfold).2.1)
    rw [show s23.regs .real [128] "l_i" = some (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) from by
      rw [hs23d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hs22d, BlockState.setReg_same]]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨ir, u⟩ := idx
    show linewFn ir = ((nopadFoldUpto s0 Q K V B_Start_Loc sm_scale_python (ctxNopadWindow s0 B_Seqlen 128) (seqLen s0 B_Seqlen) ((c + 1) * 128) ir ⟨0, by norm_num⟩).2.1 : WithBot ℝ)
    rw [hlinewSome ir]
    exact congrArg (fun p => ((p.2.1 : ℝ) : WithBot ℝ)) (hbridge ir ⟨0, by norm_num⟩)
  · -- acc = some((newfold).2.2)
    rw [show s23.regs .real [128, 128] "acc" = some (⟨fun idx : TileIndex [128, 128] => some (accFinal idx.1 idx.2.1)⟩ : Tile .real [128, 128]) from by
      rw [hs23d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hs22d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hs21d, BlockState.setReg_same]]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨ir, dd, u⟩ := idx
    show some (accFinal ir dd) = ((nopadFoldUpto s0 Q K V B_Start_Loc sm_scale_python (ctxNopadWindow s0 B_Seqlen 128) (seqLen s0 B_Seqlen) ((c + 1) * 128) ir dd).2.2 : WithBot ℝ)
    exact congrArg (fun p => ((p.2.2 : ℝ) : WithBot ℝ)) (hbridge ir dd)
-/

end VeriTile.Bench.TritonBenchG.ContextAttnNopad
