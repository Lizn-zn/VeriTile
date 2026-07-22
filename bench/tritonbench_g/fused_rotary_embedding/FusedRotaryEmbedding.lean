import VeriTile.Triton

/-!
# `fused_rotary_embedding` — strict per-kernel correctness

`decoding_fused_rotary_embedding_kernel` fuses rotary position embedding with
a paged KV-cache fill for decoding: each program unconditionally rotates the Q
row (`out0 = q0 * cos - q1 * sin`, `out1 = q1 * cos + q0 * sin`), and — guarded
by `cur_head_idx % KV_GROUP_NUM == 0` — rotates the K row and scatters the
rotated K and the V row into the paged `k_cache` / `v_cache` at offsets driven
by `BLOCK_TABLES` and `context_lengths`.

## Scope

This file verifies **the Triton kernel itself** — the per-program
`@triton.jit` body. The host launch (decoding grid, page-table layout, and how
the runtime composes per-program cache writes) is the *trusted boundary*.
Because the program ids are universally quantified, the per-program statement
covers every program. Both the old and new cache layouts are covered.

The headline is stated on the grouped vector-channel IO skin
(`GroupedMasked2DKernelIO.Implements`, notation `⊨`): a full Hoare triple over
**flat pointer memory** for the unconditional Q rotary face that every program
instance performs — ∀ disjoint base-pointer placements of `q`/`cos`/`sin`, ∀
program ids whose windows are in bounds, ∀ launch states whose four input
windows hold `q0`/`q1`/`cos`/`sin`, the translated pointer kernel terminates,
both Q half-windows hold their rotary closed forms, and every other flat cell
is untouched. The paged K/V cache faces now have **honest `⊨` headlines**
(`decoding_fused_rotary_embedding_kcache_chain_correctness` /
`..._vcache_chain_correctness`) on the `ChainMetaGroupedMasked2DKernelIO` skin:
their in-kernel `context_lengths → BLOCK_TABLES` chained loads are modelled as
two chained `.nat` slots, so the store cell's block id is *loaded*, not pinned.
The older `*_store_slice` faces (which pin `block_id` as a host `Nat` and read a
pre-rotated scratch value) are kept as independently audited `Realizes` slices,
but the cache headlines no longer NEED the pin.

## Proof architecture

```
decoding_fused_rotary_embedding_q_correctness    ← TOP SPECIFICATION
                                                   (decodingRotaryQIO ⊨ rotary)
  ├─ decoding_fused_rotary_embedding_q_surface_flattenOk
  │      bridge fragment membership
  ├─ decoding_fused_rotary_embedding_q_surface_traceSafe
  │      per-execution lane-wise safety walk
  └─ decoding_fused_rotary_embedding_q_surface_region_run
         region-model Hoare triple (both in-place Q stores + frame)

decoding_fused_rotary_embedding_all_outputs_compute_correct_general  ← ★ (K/V faces)
  ├─ q_{first,second}_half_compute_correct ⊳ q_{first,second}_half_correct
  ├─ k_{first,second}_half_compute_correct ⊳ k_{first,second}_half_correct
  ├─ context_k_cache_{first,second}_half_guarded_store_slice_compute_correct
  │      ⊳ *_store_slice_correct
  └─ context_v_cache_guarded_store_slice_compute_correct ⊳ v_cache_guarded_store_slice_correct

decoding_fused_rotary_embedding_kernel_surface_toAlgorithm_supported
       the full Python-shaped surface lowers to the algorithm layer
```

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; dtype `.to(...)` casts
erase to the identity. The rotary `cos` / `sin` factors are precomputed inputs
loaded per lane; rotation is proved as masked first-half and second-half faces
for both Q and K. The `cur_head_idx % KV_GROUP_NUM == 0` guard on the K-rotary and
K/V cache-fill path is modeled (the guard predicate `handleKv` gates the
writeback). The paged-cache offsets read `BLOCK_TABLES` and `context_lengths`
as natural-number index regions. Side
conditions: `0 < head_dim_stride` for the `⊨` headline (it is what makes the
two in-place Q half-windows disjoint and each of them injective), and
store-offset injectivity hypotheses for the `Realizes` K/V faces.
-/

namespace VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding

open VeriTile.Triton
open scoped VeriTile.Triton.GroupedMasked2DKernelIO

set_option linter.unusedSimpArgs false

/-! **★ Main theorems:** `decoding_fused_rotary_embedding_q_correctness` (top
`⊨` Q rotary specification), `decoding_fused_rotary_embedding_kcache_chain_correctness`
and `decoding_fused_rotary_embedding_vcache_chain_correctness` (the honest chained
`⊨` cache headlines that redeem the pinned `block_id`), and
`decoding_fused_rotary_embedding_all_outputs_compute_correct_general` -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-- Faithful transcription of `fused_rotary_embedding.py`'s
`decoding_fused_rotary_embedding_kernel`.

This keeps the unconditional Q rotary writes plus the conditional K rotary and
K/V cache-fill path guarded by `cur_head_idx % KV_GROUP_NUM == 0`. -/
def decoding_fused_rotary_embedding_kernel_surface
    (q k v cos sin k_cache v_cache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (x q_token_stride q_head_stride k_token_stride k_head_stride
      head_dim_stride cos_token_stride cos_stride kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride vcb_stride vch_stride vcs_stride
      vcd_stride bts_stride btb_stride block_size KV_GROUP_NUM HEAD_DIM
      : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)

  dim_range = tl.arange(0, $(HEAD_DIM))
  dim_range0 = tl.arange(0, $(HEAD_DIM) // $(2))
  dim_range1 = tl.arange($(HEAD_DIM) // $(2), $(HEAD_DIM))

  off_q = cur_token_idx * $(q_token_stride) + cur_head_idx * $(q_head_stride)
  off_q0 = off_q + dim_range0 * $(head_dim_stride)
  off_q1 = off_q + dim_range1 * $(head_dim_stride)

  loaded_q0 = tl.load(q + off_q0)
  loaded_q1 = tl.load(q + off_q1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin)
  loaded_sin = tl.load(sin + off_cos_sin)

  out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
  out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
  tl.store(q + off_q0, out_q0)
  tl.store(q + off_q1, out_q1)

  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    cur_k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    off_kv = cur_token_idx * $(k_token_stride) + cur_k_head_idx * $(k_head_stride)
    off_k0 = off_kv + dim_range0 * $(head_dim_stride)
    off_k1 = off_kv + dim_range1 * $(head_dim_stride)
    loaded_k0 = tl.load(k + off_k0)
    loaded_k1 = tl.load(k + off_k1)

    out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
    out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos

    past_kv_seq_len = tl.load(context_lengths + cur_token_idx) - $(1)

    last_block_idx = past_kv_seq_len // $(block_size)
    block_ids = tl.load(BLOCK_TABLES + cur_token_idx * $(bts_stride) +
      last_block_idx * $(btb_stride))
    offsets_in_last_block = past_kv_seq_len % $(block_size)
    offsets_cache_base = block_ids * $(kcb_stride) +
      cur_k_head_idx * $(kch_stride)
    k_range0 = offsets_cache_base +
      offsets_in_last_block * $(kcs_stride) +
      (dim_range0 // $(x)) * $(kcsplit_x_stride) +
      (dim_range0 % $(x)) * $(kcd_stride)
    k_range1 = offsets_cache_base +
      offsets_in_last_block * $(kcs_stride) +
      (dim_range1 // $(x)) * $(kcsplit_x_stride) +
      (dim_range1 % $(x)) * $(kcd_stride)
    tl.store(k_cache + k_range0, out_k0)
    tl.store(k_cache + k_range1, out_k1)

    off_v = off_kv + dim_range * $(head_dim_stride)
    loaded_v = tl.load(v + off_v)
    v_range = block_ids * $(vcb_stride) +
      cur_k_head_idx * $(vch_stride) +
      offsets_in_last_block * $(vcs_stride) +
      dim_range * $(vcd_stride)
    tl.store(v_cache + v_range, loaded_v)
  }
}

/-- The full Python-shaped fused rotary decoding surface lowers to the
algorithm layer, including both Q halves, conditional K rotation, and K/V cache
stores. -/
theorem decoding_fused_rotary_embedding_kernel_surface_toAlgorithm_supported
    (q k v cos sin k_cache v_cache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (x q_token_stride q_head_stride k_token_stride k_head_stride
      head_dim_stride cos_token_stride cos_stride kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride vcb_stride vch_stride vcs_stride
      vcd_stride bts_stride btb_stride block_size KV_GROUP_NUM HEAD_DIM
      : Nat) :
    ∃ alg, (decoding_fused_rotary_embedding_kernel_surface q k v cos sin
      k_cache v_cache BLOCK_TABLES context_lengths x q_token_stride
      q_head_stride k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride vcb_stride vch_stride vcs_stride vcd_stride
      bts_stride btb_stride block_size KV_GROUP_NUM HEAD_DIM).toAlgorithm?
        = Except.ok alg := by
  simp [decoding_fused_rotary_embedding_kernel_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription of the unconditional Q rotary part of
`fused_rotary_embedding.py`'s `decoding_fused_rotary_embedding_kernel`.

The full Python kernel also conditionally rotates K and fills K/V caches when
`cur_head_idx % KV_GROUP_NUM == 0`. That branch depends on context-length
metadata and cache block tables, so this surface covers the unconditional Q
updates that every program instance performs: both the first and second rotary
halves are written. -/
def decoding_fused_rotary_embedding_q_surface
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      _HEAD_DIM HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_q = cur_token_idx * $(q_token_stride) + cur_head_idx * $(q_head_stride)
  off_q0 = off_q + dim_range0 * $(head_dim_stride)
  off_q1 = off_q + dim_range1 * $(head_dim_stride)
  loaded_q0 = tl.load(Q + off_q0)
  loaded_q1 = tl.load(Q + off_q1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(Cos + off_cos_sin)
  loaded_sin = tl.load(Sin + off_cos_sin)
  out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
  out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
  tl.store(Q + off_q0, out_q0)
  tl.store(Q + off_q1, out_q1)
}

/-- The standalone Q rotary decoding surface lowers to the algorithm layer. -/
theorem decoding_fused_rotary_embedding_q_surface_toAlgorithm_supported
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      _HEAD_DIM HALF_DIM : Nat) :
    ∃ alg, (decoding_fused_rotary_embedding_q_surface Q Cos Sin q_token_stride
      q_head_stride head_dim_stride cos_token_stride cos_stride _HEAD_DIM
      HALF_DIM).toAlgorithm? = Except.ok alg := by
  simp [decoding_fused_rotary_embedding_q_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented Q first-half slice of `fused_rotary_embedding.py`'s
`decoding_fused_rotary_embedding_kernel`.

The full kernel updates Q, conditionally rotates K, and fills K/V caches. This
slice captures the unconditional Q first-half rotary writeback:
`q0 * cos - q1 * sin`. -/
def decoding_fused_rotary_embedding_q_first_half
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_q = cur_token_idx * $(q_token_stride) + cur_head_idx * $(q_head_stride)
  off_q0 = off_q + dim_range0 * $(head_dim_stride)
  off_q1 = off_q + dim_range1 * $(head_dim_stride)
  loaded_q0 = tl.load(q + off_q0)
  loaded_q1 = tl.load(q + off_q1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin)
  loaded_sin = tl.load(sin + off_cos_sin)
  out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
  tl.store(q + off_q0, out_q0)
}

def dimIndex (i : Fin HALF_DIM) : Nat :=
  i.val

def qBase (s : BlockState) (q_token_stride q_head_stride : Nat) : Nat :=
  s.pids 1 * q_token_stride + s.pids 0 * q_head_stride

def qFirstOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride : Nat)
    (i : Fin HALF_DIM) : Nat :=
  qBase s q_token_stride q_head_stride + dimIndex i * head_dim_stride

def qSecondOffset
    (s : BlockState)
    (q_token_stride q_head_stride head_dim_stride HALF_DIM : Nat)
    (i : Fin HALF_DIM) : Nat :=
  qBase s q_token_stride q_head_stride + (dimIndex i + HALF_DIM) * head_dim_stride

def cosOffset
    (s : BlockState) (cos_token_stride cos_stride : Nat) (i : Fin HALF_DIM) : Nat :=
  s.pids 1 * cos_token_stride + dimIndex i * cos_stride

def sinOffset
    (s : BlockState) (cos_token_stride cos_stride : Nat) (i : Fin HALF_DIM) : Nat :=
  s.pids 1 * cos_token_stride + dimIndex i * cos_stride

noncomputable def qFirstSpec
    (s : BlockState) (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (i : Fin HALF_DIM) : ℝ :=
  s.readMem q (qFirstOffset s q_token_stride q_head_stride head_dim_stride i) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride i) -
  s.readMem q
      (qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i) *
    s.readMem sin (sinOffset s cos_token_stride cos_stride i)

/-- Algorithm-layer correctness for the Q first-half rotary writeback. -/
theorem decoding_fused_rotary_embedding_q_first_half_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        qFirstOffset s q_token_stride q_head_stride head_dim_stride i))
    (hExec : exec (decoding_fused_rotary_embedding_q_first_half q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem q (qFirstOffset s q_token_stride q_head_stride head_dim_stride i) =
        qFirstSpec s q cos sin q_token_stride q_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
          idx.1.val * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [qFirstOffset, qBase, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hHalf : 0 < HALF_DIM
  · simp [exec, decoding_fused_rotary_embedding_q_first_half, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
          NumericDType.add, NumericDType.mul, NumericDType.sub, hHalf] at hExec
    rw [← hExec]
    simp only [qFirstOffset, qBase, dimIndex]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := q)
        (shape := [HALF_DIM])
        (s := (s.setReg "cur_head_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
          |>.setReg "cur_token_idx" TileDType.nat [] (Tile.scalar (s.pids 1))
          |>.setReg "dim_range0" TileDType.nat [HALF_DIM] (Tile.vec fun i => i.val)
          |>.setReg "dim_range1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => i.val + HALF_DIM)
          |>.setReg "off_q" TileDType.nat []
            (Tile.scalar (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride))
          |>.setReg "off_q0" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                i.val * head_dim_stride)
          |>.setReg "off_q1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                (i.val + HALF_DIM) * head_dim_stride)
          |>.setReg "loaded_q0" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem q
                (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                  i.1.val * head_dim_stride)) }
          |>.setReg "loaded_q1" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem q
                (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                  (i.1.val + HALF_DIM) * head_dim_stride)) }
          |>.setReg "off_cos_sin" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => s.pids 1 * cos_token_stride + i.val * cos_stride)
          |>.setReg "loaded_cos" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "loaded_sin" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "out_q0" TileDType.real [HALF_DIM]
            { data := fun i =>
              some
                (s.readMem q
                    (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                      i.1.val * head_dim_stride) *
                  s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride) -
                s.readMem q
                    (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                      (i.1.val + HALF_DIM) * head_dim_stride) *
                  s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }))
        (offsetFn := fun idx : TileIndex [HALF_DIM] =>
          s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
            idx.1.val * head_dim_stride)
        (valueFn := fun idx : TileIndex [HALF_DIM] =>
          s.readMem q
              (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                idx.1.val * head_dim_stride) *
            s.readMem cos (s.pids 1 * cos_token_stride + idx.1.val * cos_stride) -
          s.readMem q
              (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                (idx.1.val + HALF_DIM) * head_dim_stride) *
            s.readMem sin (s.pids 1 * cos_token_stride + idx.1.val * cos_stride))
        (P := fun _idx : TileIndex [HALF_DIM] => True)
        hRawInj (i, PUnit.unit))
    simpa [qFirstSpec, qFirstOffset, qSecondOffset, cosOffset, sinOffset, qBase,
      dimIndex, Tile.vec] using hScatter
  · exact False.elim (hHalf (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the Q first-half rotary writeback. -/
theorem decoding_fused_rotary_embedding_q_first_half_compute_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        qFirstOffset s q_token_stride q_head_stride head_dim_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_fused_rotary_embedding_q_first_half q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin HALF_DIM => True)
        (fun i => (q, qFirstOffset s q_token_stride q_head_stride head_dim_stride i)))
      (expected := fun i =>
        qFirstSpec s q cos sin q_token_stride q_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_q_first_half]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  exact decoding_fused_rotary_embedding_q_first_half_correct q cos sin
    q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
    HALF_DIM s s' hOutInj hExec i

/-! ## Q second-half rotary writeback (`out_q1 = q1 * cos + q0 * sin`) -/

/-- Proof-oriented Q second-half slice of `fused_rotary_embedding.py`'s
`decoding_fused_rotary_embedding_kernel`. Captures the unconditional Q
second-half rotary writeback: `q1' = q1 * cos + q0 * sin`. -/
def decoding_fused_rotary_embedding_q_second_half
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_q = cur_token_idx * $(q_token_stride) + cur_head_idx * $(q_head_stride)
  off_q0 = off_q + dim_range0 * $(head_dim_stride)
  off_q1 = off_q + dim_range1 * $(head_dim_stride)
  loaded_q0 = tl.load(q + off_q0)
  loaded_q1 = tl.load(q + off_q1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin)
  loaded_sin = tl.load(sin + off_cos_sin)
  out_q1 = loaded_q1 * loaded_cos + loaded_q0 * loaded_sin
  tl.store(q + off_q1, out_q1)
}

noncomputable def qSecondSpec
    (s : BlockState) (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (i : Fin HALF_DIM) : ℝ :=
  s.readMem q
      (qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride i) +
  s.readMem q (qFirstOffset s q_token_stride q_head_stride head_dim_stride i) *
    s.readMem sin (sinOffset s cos_token_stride cos_stride i)

/-- Algorithm-layer correctness for the Q second-half rotary writeback. -/
theorem decoding_fused_rotary_embedding_q_second_half_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i))
    (hExec : exec (decoding_fused_rotary_embedding_q_second_half q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem q
          (qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i) =
        qSecondSpec s q cos sin q_token_stride q_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
          (idx.1.val + HALF_DIM) * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [qSecondOffset, qBase, dimIndex] using h
    cases a; cases b; simp only at hab; cases hab; rfl
  by_cases hHalf : 0 < HALF_DIM
  · simp [exec, decoding_fused_rotary_embedding_q_second_half, stepStmts,
          stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
          NumericDType.add, NumericDType.mul, NumericDType.sub, hHalf] at hExec
    rw [← hExec]
    simp only [qSecondOffset, qBase, dimIndex]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := q)
        (shape := [HALF_DIM])
        (s := (s.setReg "cur_head_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
          |>.setReg "cur_token_idx" TileDType.nat [] (Tile.scalar (s.pids 1))
          |>.setReg "dim_range0" TileDType.nat [HALF_DIM] (Tile.vec fun i => i.val)
          |>.setReg "dim_range1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => i.val + HALF_DIM)
          |>.setReg "off_q" TileDType.nat []
            (Tile.scalar (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride))
          |>.setReg "off_q0" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                i.val * head_dim_stride)
          |>.setReg "off_q1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                (i.val + HALF_DIM) * head_dim_stride)
          |>.setReg "loaded_q0" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem q
                (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                  i.1.val * head_dim_stride)) }
          |>.setReg "loaded_q1" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem q
                (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                  (i.1.val + HALF_DIM) * head_dim_stride)) }
          |>.setReg "off_cos_sin" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => s.pids 1 * cos_token_stride + i.val * cos_stride)
          |>.setReg "loaded_cos" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "loaded_sin" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "out_q1" TileDType.real [HALF_DIM]
            { data := fun i =>
              some
                (s.readMem q
                    (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                      (i.1.val + HALF_DIM) * head_dim_stride) *
                  s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride) +
                s.readMem q
                    (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                      i.1.val * head_dim_stride) *
                  s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }))
        (offsetFn := fun idx : TileIndex [HALF_DIM] =>
          s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
            (idx.1.val + HALF_DIM) * head_dim_stride)
        (valueFn := fun idx : TileIndex [HALF_DIM] =>
          s.readMem q
              (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                (idx.1.val + HALF_DIM) * head_dim_stride) *
            s.readMem cos (s.pids 1 * cos_token_stride + idx.1.val * cos_stride) +
          s.readMem q
              (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                idx.1.val * head_dim_stride) *
            s.readMem sin (s.pids 1 * cos_token_stride + idx.1.val * cos_stride))
        (P := fun _idx : TileIndex [HALF_DIM] => True)
        hRawInj (i, PUnit.unit))
    simpa [qSecondSpec, qFirstOffset, qSecondOffset, cosOffset, sinOffset, qBase,
      dimIndex, Tile.vec] using hScatter
  · exact False.elim (hHalf (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the Q second-half rotary writeback. -/
theorem decoding_fused_rotary_embedding_q_second_half_compute_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_fused_rotary_embedding_q_second_half q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin HALF_DIM => True)
        (fun i => (q,
          qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i)))
      (expected := fun i =>
        qSecondSpec s q cos sin q_token_stride q_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_q_second_half]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  exact decoding_fused_rotary_embedding_q_second_half_correct q cos sin
    q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
    HALF_DIM s s' hOutInj hExec i

/-- Proof-oriented K first-half slice of
`decoding_fused_rotary_embedding_kernel`. K analog of the Q first-half slice
writing `out_k0 = k0 * cos - k1 * sin` to `k + off_k0`. The KV_GROUP gating
that ungates the K branch in the full kernel is omitted here; the slice models
the K rotary stores assuming the gate is active. -/
def decoding_fused_rotary_embedding_k_first_half
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_k_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_kv = cur_token_idx * $(k_token_stride) + cur_k_head_idx * $(k_head_stride)
  off_k0 = off_kv + dim_range0 * $(head_dim_stride)
  off_k1 = off_kv + dim_range1 * $(head_dim_stride)
  loaded_k0 = tl.load(k + off_k0)
  loaded_k1 = tl.load(k + off_k1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin)
  loaded_sin = tl.load(sin + off_cos_sin)
  out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
  tl.store(k + off_k0, out_k0)
}

def kBase (s : BlockState) (k_token_stride k_head_stride : Nat) : Nat :=
  s.pids 1 * k_token_stride + s.pids 0 * k_head_stride

def kFirstOffset
    (s : BlockState) (k_token_stride k_head_stride head_dim_stride : Nat)
    (i : Fin HALF_DIM) : Nat :=
  kBase s k_token_stride k_head_stride + dimIndex i * head_dim_stride

def kSecondOffset
    (s : BlockState)
    (k_token_stride k_head_stride head_dim_stride HALF_DIM : Nat)
    (i : Fin HALF_DIM) : Nat :=
  kBase s k_token_stride k_head_stride + (dimIndex i + HALF_DIM) * head_dim_stride

noncomputable def kFirstSpec
    (s : BlockState) (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (i : Fin HALF_DIM) : ℝ :=
  s.readMem k (kFirstOffset s k_token_stride k_head_stride head_dim_stride i) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride i) -
  s.readMem k
      (kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i) *
    s.readMem sin (sinOffset s cos_token_stride cos_stride i)

theorem decoding_fused_rotary_embedding_k_first_half_correct
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kFirstOffset s k_token_stride k_head_stride head_dim_stride i))
    (hExec : exec (decoding_fused_rotary_embedding_k_first_half k cos sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride
        cos_stride HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem k
          (kFirstOffset s k_token_stride k_head_stride head_dim_stride i) =
        kFirstSpec s k cos sin k_token_stride k_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
          idx.1.val * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kFirstOffset, kBase, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hHalf : 0 < HALF_DIM
  · simp [exec, decoding_fused_rotary_embedding_k_first_half, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
          NumericDType.add, NumericDType.mul, NumericDType.sub, hHalf] at hExec
    rw [← hExec]
    simp only [kFirstOffset, kBase, dimIndex]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := k)
        (shape := [HALF_DIM])
        (s := (s.setReg "cur_k_head_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
          |>.setReg "cur_token_idx" TileDType.nat [] (Tile.scalar (s.pids 1))
          |>.setReg "dim_range0" TileDType.nat [HALF_DIM] (Tile.vec fun i => i.val)
          |>.setReg "dim_range1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => i.val + HALF_DIM)
          |>.setReg "off_kv" TileDType.nat []
            (Tile.scalar (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride))
          |>.setReg "off_k0" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                i.val * head_dim_stride)
          |>.setReg "off_k1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                (i.val + HALF_DIM) * head_dim_stride)
          |>.setReg "loaded_k0" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem k
                (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                  i.1.val * head_dim_stride)) }
          |>.setReg "loaded_k1" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem k
                (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                  (i.1.val + HALF_DIM) * head_dim_stride)) }
          |>.setReg "off_cos_sin" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => s.pids 1 * cos_token_stride + i.val * cos_stride)
          |>.setReg "loaded_cos" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "loaded_sin" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "out_k0" TileDType.real [HALF_DIM]
            { data := fun i =>
              some
                (s.readMem k
                    (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                      i.1.val * head_dim_stride) *
                  s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride) -
                s.readMem k
                    (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                      (i.1.val + HALF_DIM) * head_dim_stride) *
                  s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }))
        (offsetFn := fun idx : TileIndex [HALF_DIM] =>
          s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
            idx.1.val * head_dim_stride)
        (valueFn := fun idx : TileIndex [HALF_DIM] =>
          s.readMem k
              (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                idx.1.val * head_dim_stride) *
            s.readMem cos (s.pids 1 * cos_token_stride + idx.1.val * cos_stride) -
          s.readMem k
              (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                (idx.1.val + HALF_DIM) * head_dim_stride) *
            s.readMem sin (s.pids 1 * cos_token_stride + idx.1.val * cos_stride))
        (P := fun _idx : TileIndex [HALF_DIM] => True)
        hRawInj (i, PUnit.unit))
    simpa [kFirstSpec, kFirstOffset, kSecondOffset, cosOffset, sinOffset, kBase,
      dimIndex, Tile.vec] using hScatter
  · exact False.elim (hHalf (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem decoding_fused_rotary_embedding_k_first_half_compute_correct
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kFirstOffset s k_token_stride k_head_stride head_dim_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_fused_rotary_embedding_k_first_half k cos sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride
        cos_stride HALF_DIM)
      (initialState := s)
      (write := fun i : Fin HALF_DIM => some (k,
        kFirstOffset s k_token_stride k_head_stride head_dim_stride i))
      (expected := fun i =>
        kFirstSpec s k cos sin k_token_stride k_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i) := by
  unfold ComputeCorrect.Realizes_without_Rounding
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_k_first_half]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact decoding_fused_rotary_embedding_k_first_half_correct k cos sin
    k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
    HALF_DIM s s' hOutInj hExec i

/-- Proof-oriented K second-half slice of
`decoding_fused_rotary_embedding_kernel`. Writes `out_k1 = k0 * sin + k1 * cos`
to `k + off_k1`. -/
def decoding_fused_rotary_embedding_k_second_half
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_k_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_kv = cur_token_idx * $(k_token_stride) + cur_k_head_idx * $(k_head_stride)
  off_k0 = off_kv + dim_range0 * $(head_dim_stride)
  off_k1 = off_kv + dim_range1 * $(head_dim_stride)
  loaded_k0 = tl.load(k + off_k0)
  loaded_k1 = tl.load(k + off_k1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin)
  loaded_sin = tl.load(sin + off_cos_sin)
  out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos
  tl.store(k + off_k1, out_k1)
}

noncomputable def kSecondSpec
    (s : BlockState) (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (i : Fin HALF_DIM) : ℝ :=
  s.readMem k (kFirstOffset s k_token_stride k_head_stride head_dim_stride i) *
    s.readMem sin (sinOffset s cos_token_stride cos_stride i) +
  s.readMem k
      (kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride i)

theorem decoding_fused_rotary_embedding_k_second_half_correct
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i))
    (hExec : exec (decoding_fused_rotary_embedding_k_second_half k cos sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride
        cos_stride HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem k
          (kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i) =
        kSecondSpec s k cos sin k_token_stride k_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
          (idx.1.val + HALF_DIM) * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kSecondOffset, kBase, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hHalf : 0 < HALF_DIM
  · simp [exec, decoding_fused_rotary_embedding_k_second_half, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
          NumericDType.add, NumericDType.mul, NumericDType.sub, hHalf] at hExec
    rw [← hExec]
    simp only [kSecondOffset, kBase, dimIndex]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := k)
        (shape := [HALF_DIM])
        (s := (s.setReg "cur_k_head_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
          |>.setReg "cur_token_idx" TileDType.nat [] (Tile.scalar (s.pids 1))
          |>.setReg "dim_range0" TileDType.nat [HALF_DIM] (Tile.vec fun i => i.val)
          |>.setReg "dim_range1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => i.val + HALF_DIM)
          |>.setReg "off_kv" TileDType.nat []
            (Tile.scalar (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride))
          |>.setReg "off_k0" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                i.val * head_dim_stride)
          |>.setReg "off_k1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                (i.val + HALF_DIM) * head_dim_stride)
          |>.setReg "loaded_k0" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem k
                (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                  i.1.val * head_dim_stride)) }
          |>.setReg "loaded_k1" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem k
                (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                  (i.1.val + HALF_DIM) * head_dim_stride)) }
          |>.setReg "off_cos_sin" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => s.pids 1 * cos_token_stride + i.val * cos_stride)
          |>.setReg "loaded_cos" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "loaded_sin" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "out_k1" TileDType.real [HALF_DIM]
            { data := fun i =>
              some
                (s.readMem k
                    (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                      i.1.val * head_dim_stride) *
                  s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride) +
                s.readMem k
                    (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                      (i.1.val + HALF_DIM) * head_dim_stride) *
                  s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }))
        (offsetFn := fun idx : TileIndex [HALF_DIM] =>
          s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
            (idx.1.val + HALF_DIM) * head_dim_stride)
        (valueFn := fun idx : TileIndex [HALF_DIM] =>
          s.readMem k
              (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                idx.1.val * head_dim_stride) *
            s.readMem sin (s.pids 1 * cos_token_stride + idx.1.val * cos_stride) +
          s.readMem k
              (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                (idx.1.val + HALF_DIM) * head_dim_stride) *
            s.readMem cos (s.pids 1 * cos_token_stride + idx.1.val * cos_stride))
        (P := fun _idx : TileIndex [HALF_DIM] => True)
        hRawInj (i, PUnit.unit))
    simpa [kSecondSpec, kFirstOffset, kSecondOffset, cosOffset, sinOffset, kBase,
      dimIndex, Tile.vec] using hScatter
  · exact False.elim (hHalf (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem decoding_fused_rotary_embedding_k_second_half_compute_correct
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_fused_rotary_embedding_k_second_half k cos sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride
        cos_stride HALF_DIM)
      (initialState := s)
      (write := fun i : Fin HALF_DIM => some (k,
        kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i))
      (expected := fun i =>
        kSecondSpec s k cos sin k_token_stride k_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i) := by
  unfold ComputeCorrect.Realizes_without_Rounding
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_k_second_half]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact decoding_fused_rotary_embedding_k_second_half_correct k cos sin
    k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
    HALF_DIM s s' hOutInj hExec i

/-- Proof-oriented v_cache store slice of `fused_rotary_embedding.py`'s
`decoding_fused_rotary_embedding_kernel`. Takes a precomputed `LoadedV` tile
and proves the writeback into `v_cache` at the canonical cache-layout offset
parameterized on `(block_id, offsets_in_last_block)`. -/
def decoding_fused_rotary_embedding_v_cache_store_slice
    (LoadedV v_cache : RegionName)
    (block_id offsets_in_last_block vcb_stride vch_stride vcs_stride vcd_stride
      HEAD_DIM : Nat) :
    ComputeKernel := triton {
  cur_k_head_idx = tl.program_id(0)
  dim_range = tl.arange(0, $(HEAD_DIM))
  loaded_v = tl.load(LoadedV + dim_range)
  v_range = $(block_id) * $(vcb_stride) +
    cur_k_head_idx * $(vch_stride) +
    $(offsets_in_last_block) * $(vcs_stride) +
    dim_range * $(vcd_stride)
  tl.store(v_cache + v_range, loaded_v)
}

def vCacheOffset
    (s : BlockState)
    (block_id offsets_in_last_block vcb_stride vch_stride vcs_stride vcd_stride : Nat)
    (i : Fin HEAD_DIM) : Nat :=
  block_id * vcb_stride + s.pids 0 * vch_stride +
    offsets_in_last_block * vcs_stride + i.val * vcd_stride

noncomputable def vCacheStoreSpec
    (s : BlockState) (LoadedV : RegionName) (i : Fin HEAD_DIM) : ℝ :=
  s.readMem LoadedV i.val

theorem decoding_fused_rotary_embedding_v_cache_store_slice_correct
    (LoadedV v_cache : RegionName)
    (block_id offsets_in_last_block vcb_stride vch_stride vcs_stride vcd_stride
      HEAD_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        vCacheOffset s block_id offsets_in_last_block vcb_stride vch_stride
          vcs_stride vcd_stride i))
    (hExec : exec (decoding_fused_rotary_embedding_v_cache_store_slice LoadedV
        v_cache block_id offsets_in_last_block vcb_stride vch_stride vcs_stride
        vcd_stride HEAD_DIM) s = some s') :
    ∀ i : Fin HEAD_DIM,
      s'.readMem v_cache
          (vCacheOffset s block_id offsets_in_last_block vcb_stride vch_stride
            vcs_stride vcd_stride i) =
        vCacheStoreSpec s LoadedV i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HEAD_DIM] =>
        block_id * vcb_stride + s.pids 0 * vch_stride +
          offsets_in_last_block * vcs_stride + idx.1.val * vcd_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [vCacheOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, decoding_fused_rotary_embedding_v_cache_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul] at hExec
  rw [← hExec]
  simp only [vCacheOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [vCacheStoreSpec]

theorem decoding_fused_rotary_embedding_v_cache_store_slice_compute_correct
    (LoadedV v_cache : RegionName)
    (block_id offsets_in_last_block vcb_stride vch_stride vcs_stride vcd_stride
      HEAD_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        vCacheOffset s block_id offsets_in_last_block vcb_stride vch_stride
          vcs_stride vcd_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_fused_rotary_embedding_v_cache_store_slice LoadedV
        v_cache block_id offsets_in_last_block vcb_stride vch_stride vcs_stride
        vcd_stride HEAD_DIM)
      (initialState := s)
      (write := fun i : Fin HEAD_DIM => some (v_cache,
        vCacheOffset s block_id offsets_in_last_block vcb_stride vch_stride
          vcs_stride vcd_stride i))
      (expected := fun i => vCacheStoreSpec s LoadedV i) := by
  unfold ComputeCorrect.Realizes_without_Rounding
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_v_cache_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact decoding_fused_rotary_embedding_v_cache_store_slice_correct LoadedV v_cache
    block_id offsets_in_last_block vcb_stride vch_stride vcs_stride vcd_stride
    HEAD_DIM s s' hOutInj hExec i

/-- Guarded V-cache store slice for the `handle_kv` branch of
`decoding_fused_rotary_embedding_kernel`.

This keeps the Python branch guard `cur_head_idx % KV_GROUP_NUM == 0` and uses
the derived `cur_k_head_idx = cur_head_idx // KV_GROUP_NUM` in the cache
address. `block_id` and `offsets_in_last_block` remain parameters here; the
companion full surface above records their origin from `context_lengths` and
`BLOCK_TABLES`. -/
def decoding_fused_rotary_embedding_v_cache_guarded_store_slice
    (LoadedV v_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM vcb_stride vch_stride
      vcs_stride vcd_stride HEAD_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    cur_k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    dim_range = tl.arange(0, $(HEAD_DIM))
    loaded_v = tl.load(LoadedV + dim_range)
    v_range = $(block_id) * $(vcb_stride) +
      cur_k_head_idx * $(vch_stride) +
      $(offsets_in_last_block) * $(vcs_stride) +
      dim_range * $(vcd_stride)
    tl.store(v_cache + v_range, loaded_v)
  }
}

def handleKv (s : BlockState) (KV_GROUP_NUM : Nat) : Prop :=
  s.pids 0 % KV_GROUP_NUM = 0

instance handleKvDecidable (s : BlockState) (KV_GROUP_NUM : Nat) :
    Decidable (handleKv s KV_GROUP_NUM) := by
  unfold handleKv
  infer_instance

def vCacheGuardedOffset
    (s : BlockState)
    (block_id offsets_in_last_block KV_GROUP_NUM vcb_stride vch_stride
      vcs_stride vcd_stride : Nat)
    (i : Fin HEAD_DIM) : Nat :=
  block_id * vcb_stride + (s.pids 0 / KV_GROUP_NUM) * vch_stride +
    offsets_in_last_block * vcs_stride + i.val * vcd_stride

theorem decoding_fused_rotary_embedding_v_cache_guarded_store_slice_correct
    (LoadedV v_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM vcb_stride vch_stride
      vcs_stride vcd_stride HEAD_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        vCacheGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
          vcb_stride vch_stride vcs_stride vcd_stride i))
    (hExec : exec (decoding_fused_rotary_embedding_v_cache_guarded_store_slice
        LoadedV v_cache block_id offsets_in_last_block KV_GROUP_NUM vcb_stride
        vch_stride vcs_stride vcd_stride HEAD_DIM) s = some s') :
    ∀ i : Fin HEAD_DIM,
      s'.readMem v_cache
          (vCacheGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
            vcb_stride vch_stride vcs_stride vcd_stride i) =
        if handleKv s KV_GROUP_NUM then
          vCacheStoreSpec s LoadedV i
        else
          s.readMem v_cache
            (vCacheGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
              vcb_stride vch_stride vcs_stride vcd_stride i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HEAD_DIM] =>
        block_id * vcb_stride + (s.pids 0 / KV_GROUP_NUM) * vch_stride +
          offsets_in_last_block * vcs_stride + idx.1.val * vcd_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [vCacheGuardedOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, decoding_fused_rotary_embedding_v_cache_guarded_store_slice,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop,
        Tile.cop, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.eq] at hExec
  by_cases hHandle : s.pids 0 % KV_GROUP_NUM = 0
  · simp [hHandle] at hExec
    rw [← hExec]
    simp [hHandle, handleKv, vCacheGuardedOffset]
    rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
    simp [vCacheStoreSpec]
  · simp [hHandle] at hExec
    rw [← hExec]
    simp [hHandle, handleKv, vCacheGuardedOffset]

theorem decoding_fused_rotary_embedding_v_cache_guarded_store_slice_compute_correct
    (LoadedV v_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM vcb_stride vch_stride
      vcs_stride vcd_stride HEAD_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        vCacheGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
          vcb_stride vch_stride vcs_stride vcd_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_fused_rotary_embedding_v_cache_guarded_store_slice
        LoadedV v_cache block_id offsets_in_last_block KV_GROUP_NUM vcb_stride
        vch_stride vcs_stride vcd_stride HEAD_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HEAD_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (v_cache,
          vCacheGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
            vcb_stride vch_stride vcs_stride vcd_stride i)))
      (expected := fun i => vCacheStoreSpec s LoadedV i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_v_cache_guarded_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := decoding_fused_rotary_embedding_v_cache_guarded_store_slice_correct
    LoadedV v_cache block_id offsets_in_last_block KV_GROUP_NUM vcb_stride
    vch_stride vcs_stride vcd_stride HEAD_DIM s s' hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented k_cache first-half store slice of
`fused_rotary_embedding.py`'s `decoding_fused_rotary_embedding_kernel`.
Takes a precomputed `OutK0Pre` tile (the post-rotary K first-half values)
and proves the writeback into `k_cache` at the cache-layout first-half
offsets. Parameterized over `(block_id, offsets_in_last_block, x)`, covering
both the legacy 4D cache layout (`kcsplit_x_stride = 0`, `x = head_dim`) and
the new split layout `[num_blocks, heads, head_dim // x, block_size, x]`. -/
def decoding_fused_rotary_embedding_k_cache_first_half_store_slice
    (OutK0Pre k_cache : RegionName)
    (block_id offsets_in_last_block x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_k_head_idx = tl.program_id(0)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  out_k0 = tl.load(OutK0Pre + dim_range0)
  k_range0 = $(block_id) * $(kcb_stride) +
    cur_k_head_idx * $(kch_stride) +
    $(offsets_in_last_block) * $(kcs_stride) +
    (dim_range0 // $(x)) * $(kcsplit_x_stride) +
    (dim_range0 % $(x)) * $(kcd_stride)
  tl.store(k_cache + k_range0, out_k0)
}

def kCacheFirstOffset
    (s : BlockState)
    (block_id offsets_in_last_block x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride : Nat)
    (i : Fin HALF_DIM) : Nat :=
  block_id * kcb_stride + s.pids 0 * kch_stride +
    offsets_in_last_block * kcs_stride +
    (i.val / x) * kcsplit_x_stride + (i.val % x) * kcd_stride

noncomputable def kCacheFirstStoreSpec
    (s : BlockState) (OutK0Pre : RegionName) (i : Fin HALF_DIM) : ℝ :=
  s.readMem OutK0Pre i.val

theorem decoding_fused_rotary_embedding_k_cache_first_half_store_slice_correct
    (OutK0Pre k_cache : RegionName)
    (block_id offsets_in_last_block x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kCacheFirstOffset s block_id offsets_in_last_block x kcb_stride
          kch_stride kcsplit_x_stride kcs_stride kcd_stride i))
    (hExec : exec (decoding_fused_rotary_embedding_k_cache_first_half_store_slice
        OutK0Pre k_cache block_id offsets_in_last_block x kcb_stride kch_stride
        kcsplit_x_stride kcs_stride kcd_stride HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem k_cache
          (kCacheFirstOffset s block_id offsets_in_last_block x kcb_stride
            kch_stride kcsplit_x_stride kcs_stride kcd_stride i) =
        kCacheFirstStoreSpec s OutK0Pre i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        block_id * kcb_stride + s.pids 0 * kch_stride +
          offsets_in_last_block * kcs_stride +
          (idx.1.val / x) * kcsplit_x_stride + (idx.1.val % x) * kcd_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kCacheFirstOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, decoding_fused_rotary_embedding_k_cache_first_half_store_slice,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul] at hExec
  rw [← hExec]
  simp only [kCacheFirstOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [kCacheFirstStoreSpec]

theorem decoding_fused_rotary_embedding_k_cache_first_half_store_slice_compute_correct
    (OutK0Pre k_cache : RegionName)
    (block_id offsets_in_last_block x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kCacheFirstOffset s block_id offsets_in_last_block x kcb_stride
          kch_stride kcsplit_x_stride kcs_stride kcd_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_fused_rotary_embedding_k_cache_first_half_store_slice
        OutK0Pre k_cache block_id offsets_in_last_block x kcb_stride kch_stride
        kcsplit_x_stride kcs_stride kcd_stride HALF_DIM)
      (initialState := s)
      (write := fun i : Fin HALF_DIM => some (k_cache,
        kCacheFirstOffset s block_id offsets_in_last_block x kcb_stride
          kch_stride kcsplit_x_stride kcs_stride kcd_stride i))
      (expected := fun i => kCacheFirstStoreSpec s OutK0Pre i) := by
  unfold ComputeCorrect.Realizes_without_Rounding
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_k_cache_first_half_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact decoding_fused_rotary_embedding_k_cache_first_half_store_slice_correct
    OutK0Pre k_cache block_id offsets_in_last_block x kcb_stride kch_stride
    kcsplit_x_stride kcs_stride kcd_stride HALF_DIM s s' hOutInj hExec i

/-- Proof-oriented k_cache second-half store slice. K-side analog of the
first-half slice. -/
def decoding_fused_rotary_embedding_k_cache_second_half_store_slice
    (OutK1Pre k_cache : RegionName)
    (block_id offsets_in_last_block x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_k_head_idx = tl.program_id(0)
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  out_k1 = tl.load(OutK1Pre + dim_range1)
  k_range1 = $(block_id) * $(kcb_stride) +
    cur_k_head_idx * $(kch_stride) +
    $(offsets_in_last_block) * $(kcs_stride) +
    (dim_range1 // $(x)) * $(kcsplit_x_stride) +
    (dim_range1 % $(x)) * $(kcd_stride)
  tl.store(k_cache + k_range1, out_k1)
}

def kCacheSecondOffset
    (s : BlockState)
    (block_id offsets_in_last_block x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride HALF_DIM : Nat) (i : Fin HALF_DIM) : Nat :=
  block_id * kcb_stride + s.pids 0 * kch_stride +
    offsets_in_last_block * kcs_stride +
    ((i.val + HALF_DIM) / x) * kcsplit_x_stride +
    ((i.val + HALF_DIM) % x) * kcd_stride

noncomputable def kCacheSecondStoreSpec
    (s : BlockState) (OutK1Pre : RegionName) (HALF_DIM : Nat) (i : Fin HALF_DIM) : ℝ :=
  s.readMem OutK1Pre (i.val + HALF_DIM)

theorem decoding_fused_rotary_embedding_k_cache_second_half_store_slice_correct
    (OutK1Pre k_cache : RegionName)
    (block_id offsets_in_last_block x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kCacheSecondOffset s block_id offsets_in_last_block x kcb_stride
          kch_stride kcsplit_x_stride kcs_stride kcd_stride HALF_DIM i))
    (hExec : exec (decoding_fused_rotary_embedding_k_cache_second_half_store_slice
        OutK1Pre k_cache block_id offsets_in_last_block x kcb_stride kch_stride
        kcsplit_x_stride kcs_stride kcd_stride HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem k_cache
          (kCacheSecondOffset s block_id offsets_in_last_block x kcb_stride
            kch_stride kcsplit_x_stride kcs_stride kcd_stride HALF_DIM i) =
        kCacheSecondStoreSpec s OutK1Pre HALF_DIM i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        block_id * kcb_stride + s.pids 0 * kch_stride +
          offsets_in_last_block * kcs_stride +
          ((idx.1.val + HALF_DIM) / x) * kcsplit_x_stride +
          ((idx.1.val + HALF_DIM) % x) * kcd_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kCacheSecondOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, decoding_fused_rotary_embedding_k_cache_second_half_store_slice,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul] at hExec
  rw [← hExec]
  simp only [kCacheSecondOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [kCacheSecondStoreSpec]

theorem decoding_fused_rotary_embedding_k_cache_second_half_store_slice_compute_correct
    (OutK1Pre k_cache : RegionName)
    (block_id offsets_in_last_block x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kCacheSecondOffset s block_id offsets_in_last_block x kcb_stride
          kch_stride kcsplit_x_stride kcs_stride kcd_stride HALF_DIM i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_fused_rotary_embedding_k_cache_second_half_store_slice
        OutK1Pre k_cache block_id offsets_in_last_block x kcb_stride kch_stride
        kcsplit_x_stride kcs_stride kcd_stride HALF_DIM)
      (initialState := s)
      (write := fun i : Fin HALF_DIM => some (k_cache,
        kCacheSecondOffset s block_id offsets_in_last_block x kcb_stride
          kch_stride kcsplit_x_stride kcs_stride kcd_stride HALF_DIM i))
      (expected := fun i => kCacheSecondStoreSpec s OutK1Pre HALF_DIM i) := by
  unfold ComputeCorrect.Realizes_without_Rounding
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_k_cache_second_half_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact decoding_fused_rotary_embedding_k_cache_second_half_store_slice_correct
    OutK1Pre k_cache block_id offsets_in_last_block x kcb_stride kch_stride
    kcsplit_x_stride kcs_stride kcd_stride HALF_DIM s s' hOutInj hExec i

/-- Guarded K-cache first-half store slice for the `handle_kv` branch. -/
def decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
    (OutK0Pre k_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    cur_k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    dim_range0 = tl.arange(0, $(HALF_DIM))
    out_k0 = tl.load(OutK0Pre + dim_range0)
    k_range0 = $(block_id) * $(kcb_stride) +
      cur_k_head_idx * $(kch_stride) +
      $(offsets_in_last_block) * $(kcs_stride) +
      (dim_range0 // $(x)) * $(kcsplit_x_stride) +
      (dim_range0 % $(x)) * $(kcd_stride)
    tl.store(k_cache + k_range0, out_k0)
  }
}

def kCacheFirstGuardedOffset
    (s : BlockState)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride : Nat)
    (i : Fin HALF_DIM) : Nat :=
  block_id * kcb_stride + (s.pids 0 / KV_GROUP_NUM) * kch_stride +
    offsets_in_last_block * kcs_stride +
    (i.val / x) * kcsplit_x_stride + (i.val % x) * kcd_stride

theorem decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice_correct
    (OutK0Pre k_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kCacheFirstGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
          x kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride i))
    (hExec : exec
        (decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
          OutK0Pre k_cache block_id offsets_in_last_block KV_GROUP_NUM x
          kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
          HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem k_cache
          (kCacheFirstGuardedOffset s block_id offsets_in_last_block
            KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
            kcd_stride i) =
        if handleKv s KV_GROUP_NUM then
          kCacheFirstStoreSpec s OutK0Pre i
        else
          s.readMem k_cache
            (kCacheFirstGuardedOffset s block_id offsets_in_last_block
              KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
              kcd_stride i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        block_id * kcb_stride + (s.pids 0 / KV_GROUP_NUM) * kch_stride +
          offsets_in_last_block * kcs_stride +
          (idx.1.val / x) * kcsplit_x_stride + (idx.1.val % x) * kcd_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kCacheFirstGuardedOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop,
        Tile.cop, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.eq] at hExec
  by_cases hHandle : s.pids 0 % KV_GROUP_NUM = 0
  · simp [hHandle] at hExec
    rw [← hExec]
    simp [hHandle, handleKv, kCacheFirstGuardedOffset]
    rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
    simp [kCacheFirstStoreSpec]
  · simp [hHandle] at hExec
    rw [← hExec]
    simp [hHandle, handleKv, kCacheFirstGuardedOffset]

theorem decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice_compute_correct
    (OutK0Pre k_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kCacheFirstGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
          x kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
          OutK0Pre k_cache block_id offsets_in_last_block KV_GROUP_NUM x
          kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
          HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HALF_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (k_cache,
          kCacheFirstGuardedOffset s block_id offsets_in_last_block
            KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
            kcd_stride i)))
      (expected := fun i => kCacheFirstStoreSpec s OutK0Pre i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h :=
    decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice_correct
      OutK0Pre k_cache block_id offsets_in_last_block KV_GROUP_NUM x
      kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride HALF_DIM
      s s' hOutInj hExec i
  simpa [hActive] using h

/-- Guarded K-cache second-half store slice for the `handle_kv` branch. -/
def decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
    (OutK1Pre k_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    cur_k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
    out_k1 = tl.load(OutK1Pre + dim_range1)
    k_range1 = $(block_id) * $(kcb_stride) +
      cur_k_head_idx * $(kch_stride) +
      $(offsets_in_last_block) * $(kcs_stride) +
      (dim_range1 // $(x)) * $(kcsplit_x_stride) +
      (dim_range1 % $(x)) * $(kcd_stride)
    tl.store(k_cache + k_range1, out_k1)
  }
}

def kCacheSecondGuardedOffset
    (s : BlockState)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat)
    (i : Fin HALF_DIM) : Nat :=
  block_id * kcb_stride + (s.pids 0 / KV_GROUP_NUM) * kch_stride +
    offsets_in_last_block * kcs_stride +
    ((i.val + HALF_DIM) / x) * kcsplit_x_stride +
    ((i.val + HALF_DIM) % x) * kcd_stride

theorem decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice_correct
    (OutK1Pre k_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kCacheSecondGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
          x kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
          HALF_DIM i))
    (hExec : exec
        (decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
          OutK1Pre k_cache block_id offsets_in_last_block KV_GROUP_NUM x
          kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
          HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem k_cache
          (kCacheSecondGuardedOffset s block_id offsets_in_last_block
            KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
            kcd_stride HALF_DIM i) =
        if handleKv s KV_GROUP_NUM then
          kCacheSecondStoreSpec s OutK1Pre HALF_DIM i
        else
          s.readMem k_cache
            (kCacheSecondGuardedOffset s block_id offsets_in_last_block
              KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
              kcd_stride HALF_DIM i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        block_id * kcb_stride + (s.pids 0 / KV_GROUP_NUM) * kch_stride +
          offsets_in_last_block * kcs_stride +
          ((idx.1.val + HALF_DIM) / x) * kcsplit_x_stride +
          ((idx.1.val + HALF_DIM) % x) * kcd_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kCacheSecondGuardedOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop,
        Tile.cop, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.eq] at hExec
  by_cases hHandle : s.pids 0 % KV_GROUP_NUM = 0
  · simp [hHandle] at hExec
    rw [← hExec]
    simp [hHandle, handleKv, kCacheSecondGuardedOffset]
    rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
    simp [kCacheSecondStoreSpec]
  · simp [hHandle] at hExec
    rw [← hExec]
    simp [hHandle, handleKv, kCacheSecondGuardedOffset]

theorem decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice_compute_correct
    (OutK1Pre k_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kCacheSecondGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
          x kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
          HALF_DIM i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
          OutK1Pre k_cache block_id offsets_in_last_block KV_GROUP_NUM x
          kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
          HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HALF_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (k_cache,
          kCacheSecondGuardedOffset s block_id offsets_in_last_block
            KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
            kcd_stride HALF_DIM i)))
      (expected := fun i => kCacheSecondStoreSpec s OutK1Pre HALF_DIM i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h :=
    decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice_correct
      OutK1Pre k_cache block_id offsets_in_last_block KV_GROUP_NUM x
      kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride HALF_DIM
      s s' hOutInj hExec i
  simpa [hActive] using h

/-! ## Python metadata-specialized guarded cache writebacks -/

/-- Python decode metadata:
`past_kv_seq_len = context_lengths[cur_token_idx] - 1`. -/
def decodingPastKvSeqLen (s : BlockState) (context_lengths : RegionName) : Nat :=
  s.readMemValue .nat context_lengths (s.pids 1) - 1

/-- Python decode metadata: `last_block_idx = past_kv_seq_len // block_size`. -/
def decodingLastBlockIdx
    (s : BlockState) (context_lengths : RegionName) (block_size : Nat) : Nat :=
  decodingPastKvSeqLen s context_lengths / block_size

/-- Python decode metadata:
`block_ids = BLOCK_TABLES[cur_token_idx, last_block_idx]`. -/
def decodingBlockId
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (bts_stride btb_stride block_size : Nat) : Nat :=
  s.readMemValue .nat BLOCK_TABLES
    (s.pids 1 * bts_stride +
      decodingLastBlockIdx s context_lengths block_size * btb_stride)

/-- Python decode metadata:
`offsets_in_last_block = past_kv_seq_len % block_size`. -/
def decodingOffsetsInLastBlock
    (s : BlockState) (context_lengths : RegionName) (block_size : Nat) : Nat :=
  decodingPastKvSeqLen s context_lengths % block_size

def decodingKCacheFirstGuardedOffset
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
      kcd_stride bts_stride btb_stride block_size : Nat)
    (i : Fin HALF_DIM) : Nat :=
  kCacheFirstGuardedOffset s
    (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
      block_size)
    (decodingOffsetsInLastBlock s context_lengths block_size)
    KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
    kcd_stride i

def decodingKCacheSecondGuardedOffset
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
      kcd_stride bts_stride btb_stride block_size HALF_DIM : Nat)
    (i : Fin HALF_DIM) : Nat :=
  kCacheSecondGuardedOffset s
    (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
      block_size)
    (decodingOffsetsInLastBlock s context_lengths block_size)
    KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
    kcd_stride HALF_DIM i

def decodingVCacheGuardedOffset
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride bts_stride
      btb_stride block_size : Nat)
    (i : Fin HEAD_DIM) : Nat :=
  vCacheGuardedOffset s
    (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
      block_size)
    (decodingOffsetsInLastBlock s context_lengths block_size)
    KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride i

/-- Public guarded K-cache first-half theorem specialized to the Python
`context_lengths` and `BLOCK_TABLES` metadata path. -/
theorem decoding_fused_rotary_embedding_context_k_cache_first_half_guarded_store_slice_compute_correct
    (OutK0Pre k_cache BLOCK_TABLES context_lengths : RegionName)
    (KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
      kcd_stride bts_stride btb_stride block_size HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        decodingKCacheFirstGuardedOffset s BLOCK_TABLES context_lengths
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride bts_stride btb_stride block_size i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
          OutK0Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
            block_size)
          (decodingOffsetsInLastBlock s context_lengths block_size)
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HALF_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (k_cache,
          decodingKCacheFirstGuardedOffset s BLOCK_TABLES context_lengths
            KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
            kcd_stride bts_stride btb_stride block_size i)))
      (expected := fun i => kCacheFirstStoreSpec s OutK0Pre i) := by
  simpa [decodingKCacheFirstGuardedOffset]
    using
      decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice_compute_correct
        OutK0Pre k_cache
        (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
          block_size)
        (decodingOffsetsInLastBlock s context_lengths block_size)
        KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
        kcd_stride HALF_DIM s hOutInj

/-- Public guarded K-cache second-half theorem specialized to the Python
`context_lengths` and `BLOCK_TABLES` metadata path. -/
theorem decoding_fused_rotary_embedding_context_k_cache_second_half_guarded_store_slice_compute_correct
    (OutK1Pre k_cache BLOCK_TABLES context_lengths : RegionName)
    (KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
      kcd_stride bts_stride btb_stride block_size HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        decodingKCacheSecondGuardedOffset s BLOCK_TABLES context_lengths
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride bts_stride btb_stride block_size HALF_DIM i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
          OutK1Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
            block_size)
          (decodingOffsetsInLastBlock s context_lengths block_size)
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HALF_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (k_cache,
          decodingKCacheSecondGuardedOffset s BLOCK_TABLES context_lengths
            KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
            kcd_stride bts_stride btb_stride block_size HALF_DIM i)))
      (expected := fun i => kCacheSecondStoreSpec s OutK1Pre HALF_DIM i) := by
  simpa [decodingKCacheSecondGuardedOffset]
    using
      decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice_compute_correct
        OutK1Pre k_cache
        (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
          block_size)
        (decodingOffsetsInLastBlock s context_lengths block_size)
        KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
        kcd_stride HALF_DIM s hOutInj

/-- Public guarded V-cache theorem specialized to the Python `context_lengths`
and `BLOCK_TABLES` metadata path. -/
theorem decoding_fused_rotary_embedding_context_v_cache_guarded_store_slice_compute_correct
    (LoadedV v_cache BLOCK_TABLES context_lengths : RegionName)
    (KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride bts_stride
      btb_stride block_size HEAD_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        decodingVCacheGuardedOffset s BLOCK_TABLES context_lengths KV_GROUP_NUM
          vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride
          block_size i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_fused_rotary_embedding_v_cache_guarded_store_slice
        LoadedV v_cache
        (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
          block_size)
        (decodingOffsetsInLastBlock s context_lengths block_size)
        KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride HEAD_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HEAD_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (v_cache,
          decodingVCacheGuardedOffset s BLOCK_TABLES context_lengths
            KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride
            bts_stride btb_stride block_size i)))
      (expected := fun i => vCacheStoreSpec s LoadedV i) := by
  simpa [decodingVCacheGuardedOffset]
    using decoding_fused_rotary_embedding_v_cache_guarded_store_slice_compute_correct
      LoadedV v_cache
      (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
        block_size)
      (decodingOffsetsInLastBlock s context_lengths block_size)
      KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride HEAD_DIM s
      hOutInj

/-! ## All-outputs correctness (dimension-general)

The two main theorems below are symbolic in every stride and dimension (rotary
half width `HALF_DIM`, V head dim `HEAD_DIM`, `block_size`, `KV_GROUP_NUM`, …);
no test-shape literals are hardcoded. For background, the checked
`fused_rotary_embedding.py` test uses `total_tokens = 16`, `q_head_num = 8`,
`kv_head_num = 4`, `head_dim = 64`, `block_size = 4`, and the default 4D K/V
cache layout in its first case — just one instance of the general statements,
which cover both the old and new cache layouts. -/


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **General all-outputs correctness (genuine closed form).** Fully
dimension-parameterized over all Q/K/cos/sin/cache strides, the rotary half
width `HALF_DIM`, the V head dim `HEAD_DIM`, the cache split width `x`, and
`KV_GROUP_NUM`: the unconditional Q rotary writebacks, the K rotary writebacks,
and the `handle_kv`-guarded paged K/V cache stores (driven by `context_lengths`
/ `BLOCK_TABLES` metadata) all realize their genuine rotary closed forms reading
input memory. No hardcoded `512`/`64`/`32`/`2`/`4`/`16` literals — every stride
and dimension is a free `Nat` parameter, with only the per-region store-offset
injectivity hypotheses as side conditions. -/
theorem decoding_fused_rotary_embedding_all_outputs_compute_correct_general
    (q k cos sin OutK0Pre OutK1Pre LoadedV k_cache v_cache BLOCK_TABLES
      context_lengths : RegionName)
    (q_token_stride q_head_stride k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride vcb_stride vch_stride vcs_stride vcd_stride
      bts_stride btb_stride block_size KV_GROUP_NUM HALF_DIM HEAD_DIM : Nat)
    (s : BlockState)
    (hQ1Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        qFirstOffset s q_token_stride q_head_stride head_dim_stride i))
    (hQ2Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i))
    (hK1Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kFirstOffset s k_token_stride k_head_stride head_dim_stride i))
    (hK2Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i))
    (hKC1Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        decodingKCacheFirstGuardedOffset s BLOCK_TABLES context_lengths
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride bts_stride btb_stride block_size i))
    (hKC2Inj : Function.Injective
      (fun i : Fin HALF_DIM =>
        decodingKCacheSecondGuardedOffset s BLOCK_TABLES context_lengths
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride bts_stride btb_stride block_size HALF_DIM i))
    (hVCInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        decodingVCacheGuardedOffset s BLOCK_TABLES context_lengths KV_GROUP_NUM
          vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride
          block_size i)) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_fused_rotary_embedding_q_first_half q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin HALF_DIM => True)
        (fun i => (q, qFirstOffset s q_token_stride q_head_stride head_dim_stride i)))
      (expected := fun i =>
        qFirstSpec s q cos sin q_token_stride q_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_fused_rotary_embedding_q_second_half q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin HALF_DIM => True)
        (fun i => (q,
          qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i)))
      (expected := fun i =>
        qSecondSpec s q cos sin q_token_stride q_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_fused_rotary_embedding_k_first_half k cos sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM)
      (initialState := s)
      (write := fun i : Fin HALF_DIM => some (k,
        kFirstOffset s k_token_stride k_head_stride head_dim_stride i))
      (expected := fun i =>
        kFirstSpec s k cos sin k_token_stride k_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_fused_rotary_embedding_k_second_half k cos sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM)
      (initialState := s)
      (write := fun i : Fin HALF_DIM => some (k,
        kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i))
      (expected := fun i =>
        kSecondSpec s k cos sin k_token_stride k_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
          OutK0Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
            block_size)
          (decodingOffsetsInLastBlock s context_lengths block_size)
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HALF_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (k_cache,
          decodingKCacheFirstGuardedOffset s BLOCK_TABLES context_lengths
            KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
            kcd_stride bts_stride btb_stride block_size i)))
      (expected := fun i => kCacheFirstStoreSpec s OutK0Pre i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
          OutK1Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
            block_size)
          (decodingOffsetsInLastBlock s context_lengths block_size)
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HALF_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (k_cache,
          decodingKCacheSecondGuardedOffset s BLOCK_TABLES context_lengths
            KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
            kcd_stride bts_stride btb_stride block_size HALF_DIM i)))
      (expected := fun i => kCacheSecondStoreSpec s OutK1Pre HALF_DIM i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_fused_rotary_embedding_v_cache_guarded_store_slice
        LoadedV v_cache
        (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
          block_size)
        (decodingOffsetsInLastBlock s context_lengths block_size)
        KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride HEAD_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HEAD_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (v_cache,
          decodingVCacheGuardedOffset s BLOCK_TABLES context_lengths
            KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride bts_stride
            btb_stride block_size i)))
      (expected := fun i => vCacheStoreSpec s LoadedV i)) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact decoding_fused_rotary_embedding_q_first_half_compute_correct q cos sin
      q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM s hQ1Inj
  · exact decoding_fused_rotary_embedding_q_second_half_compute_correct q cos sin
      q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM s hQ2Inj
  · exact decoding_fused_rotary_embedding_k_first_half_compute_correct k cos sin
      k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM s hK1Inj
  · exact decoding_fused_rotary_embedding_k_second_half_compute_correct k cos sin
      k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM s hK2Inj
  · exact
      decoding_fused_rotary_embedding_context_k_cache_first_half_guarded_store_slice_compute_correct
        OutK0Pre k_cache BLOCK_TABLES context_lengths KV_GROUP_NUM x kcb_stride
        kch_stride kcsplit_x_stride kcs_stride kcd_stride bts_stride btb_stride
        block_size HALF_DIM s hKC1Inj
  · exact
      decoding_fused_rotary_embedding_context_k_cache_second_half_guarded_store_slice_compute_correct
        OutK1Pre k_cache BLOCK_TABLES context_lengths KV_GROUP_NUM x kcb_stride
        kch_stride kcsplit_x_stride kcs_stride kcd_stride bts_stride btb_stride
        block_size HALF_DIM s hKC2Inj
  · exact
      decoding_fused_rotary_embedding_context_v_cache_guarded_store_slice_compute_correct
        LoadedV v_cache BLOCK_TABLES context_lengths KV_GROUP_NUM vcb_stride
        vch_stride vcs_stride vcd_stride bts_stride btb_stride block_size
        HEAD_DIM s hVCInj


/-! ## ════════ The `⊨` specification (unconditional Q rotary) ════════

The headline below is stated on the grouped vector-channel IO skin
`GroupedMasked2DKernelIO`: a full Hoare triple over **flat pointer memory**
for `decoding_fused_rotary_embedding_q_surface`, the unconditional Q rotary
face that *every* program instance performs. Four input channels
(`q` first half, `q` second half, `cos`, `sin`) and two output channels
(both the *same* `q` buffer — the rotation is in place, which is why the
allocation list `bufs` is decoupled from the channel tables). -/

/-- Program `(pid₀, pid₁)`'s Q first-half lane address — the pid-level twin of
`qFirstOffset` (which reads the ids out of a `BlockState`). -/
def qFirstAddr
    (pid₀ pid₁ q_token_stride q_head_stride head_dim_stride HALF_DIM : Nat)
    (j : Fin HALF_DIM) : Nat :=
  pid₁ * q_token_stride + pid₀ * q_head_stride + j.val * head_dim_stride

/-- Program `(pid₀, pid₁)`'s Q second-half lane address — the pid-level twin of
`qSecondOffset`. -/
def qSecondAddr
    (pid₀ pid₁ q_token_stride q_head_stride head_dim_stride HALF_DIM : Nat)
    (j : Fin HALF_DIM) : Nat :=
  pid₁ * q_token_stride + pid₀ * q_head_stride +
    (j.val + HALF_DIM) * head_dim_stride

/-- Program `pid₁`'s rotary-factor lane address — the pid-level twin of
`cosOffset` / `sinOffset` (`cos` and `sin` share one addressing scheme). -/
def cosSinAddr (pid₁ cos_token_stride cos_stride HALF_DIM : Nat)
    (j : Fin HALF_DIM) : Nat :=
  pid₁ * cos_token_stride + j.val * cos_stride

/-- An unmasked scatter-store `foldl` leaves every address it does not hit
unchanged (value-level frame; used to push the second-half store past the
first-half readback, since both stores target the same `q` buffer). -/
private theorem foldl_store_preserve_readMem {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (o : Nat) (l : List α)
    (s : BlockState) (hnot : ∀ k ∈ l, offsetFn k ≠ o) :
    (l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k))
        s).readMem region o = s.readMem region o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons,
        ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
        BlockState.writeMem_readMem]
      show (if region = region ∧ o = offsetFn hd then valueFn hd
        else s.readMem region o) = s.readMem region o
      exact if_neg (fun hc => hnot hd List.mem_cons_self hc.2.symm)

/-- Cell-level frame for an unmasked scatter-store `foldl`: every memory cell
it does not hit is preserved. -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ)
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k))
        s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons,
        ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
        BlockState.writeMem_mem]
      exact if_neg (fun hc =>
        hnot hd List.mem_cons_self ⟨hc.1.symm, hc.2.symm⟩)

/-- **The region-model Hoare triple** for the unconditional Q rotary surface —
termination, the values of both output windows, and the frame off them, from
any launch state whose four input windows hold `q0s`/`q1s`/`cs`/`sns`.

This is the `hrun` obligation of the `⊨` headline. The two stores hit the
*same* `q` buffer, so the first-half readback has to survive the second-half
scatter: that is exactly what `0 < head_dim_stride` buys (lane `j`'s first-half
address `j · hds` is below every second-half address `(k + HALF_DIM) · hds`),
and the same hypothesis gives both scatters their injectivity. -/
theorem decoding_fused_rotary_embedding_q_surface_region_run
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (hstride : 0 < head_dim_stride)
    (s₀ : BlockState) (q0s q1s cs sns : Fin HALF_DIM → ℝ)
    (hq0 : ∀ j : Fin HALF_DIM, s₀.readMem q (qFirstAddr (s₀.pids 0) (s₀.pids 1)
      q_token_stride q_head_stride head_dim_stride HALF_DIM j) = q0s j)
    (hq1 : ∀ j : Fin HALF_DIM, s₀.readMem q (qSecondAddr (s₀.pids 0) (s₀.pids 1)
      q_token_stride q_head_stride head_dim_stride HALF_DIM j) = q1s j)
    (hc : ∀ j : Fin HALF_DIM, s₀.readMem cos
      (cosSinAddr (s₀.pids 1) cos_token_stride cos_stride HALF_DIM j) = cs j)
    (hs : ∀ j : Fin HALF_DIM, s₀.readMem sin
      (cosSinAddr (s₀.pids 1) cos_token_stride cos_stride HALF_DIM j) = sns j) :
    ∃ s1, exec ((decoding_fused_rotary_embedding_q_surface q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        (HALF_DIM * 2) HALF_DIM).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin HALF_DIM, s1.readMem q (qFirstAddr (s₀.pids 0) (s₀.pids 1)
            q_token_stride q_head_stride head_dim_stride HALF_DIM j)
          = q0s j * cs j - q1s j * sns j)
      ∧ (∀ j : Fin HALF_DIM, s1.readMem q (qSecondAddr (s₀.pids 0) (s₀.pids 1)
            q_token_stride q_head_stride head_dim_stride HALF_DIM j)
          = q0s j * sns j + q1s j * cs j)
      ∧ (∀ r o,
          (r ≠ q ∨ ∀ j : Fin HALF_DIM, o ≠ qFirstAddr (s₀.pids 0) (s₀.pids 1)
            q_token_stride q_head_stride head_dim_stride HALF_DIM j) →
          (r ≠ q ∨ ∀ j : Fin HALF_DIM, o ≠ qSecondAddr (s₀.pids 0) (s₀.pids 1)
            q_token_stride q_head_stride head_dim_stride HALF_DIM j) →
          s1.mem r o = s₀.mem r o) := by
  have hexec : ∃ s1, exec ((decoding_fused_rotary_embedding_q_surface q cos sin
      q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      (HALF_DIM * 2) HALF_DIM).toAlgKernel) s₀ = some s1 := by
    simp [exec, decoding_fused_rotary_embedding_q_surface,
      ComputeKernel.toAlgKernel, stepStmts, stepStmt,
      evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  obtain ⟨s1, hs1⟩ := hexec
  have hs1' := hs1
  simp [exec, decoding_fused_rotary_embedding_q_surface,
    ComputeKernel.toAlgKernel, stepStmts, stepStmt,
    evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hs1'
  subst hs1'
  simp only [qFirstAddr, qSecondAddr, cosSinAddr] at hq0 hq1 hc hs ⊢
  have hInj1 : Function.Injective (fun idx : TileIndex [HALF_DIM] =>
      s₀.pids 1 * q_token_stride + s₀.pids 0 * q_head_stride +
        idx.1.val * head_dim_stride) := by
    intro a b h
    dsimp only at h
    have h1 : a.1 = b.1 :=
      Fin.ext (Nat.eq_of_mul_eq_mul_right hstride (Nat.add_left_cancel h))
    exact Prod.ext h1 (Subsingleton.elim _ _)
  have hInj2 : Function.Injective (fun idx : TileIndex [HALF_DIM] =>
      s₀.pids 1 * q_token_stride + s₀.pids 0 * q_head_stride +
        (idx.1.val + HALF_DIM) * head_dim_stride) := by
    intro a b h
    dsimp only at h
    have h1 : a.1 = b.1 :=
      Fin.ext (by
        have := Nat.eq_of_mul_eq_mul_right hstride (Nat.add_left_cancel h)
        omega)
    exact Prod.ext h1 (Subsingleton.elim _ _)
  refine ⟨_, hs1, ?_, ?_, ?_⟩
  · -- first-half channel: read past the second-half scatter, then read back
    intro j
    have hmiss : ∀ k : TileIndex [HALF_DIM],
        k ∈ TileShape.allIndices [HALF_DIM] →
        s₀.pids 1 * q_token_stride + s₀.pids 0 * q_head_stride +
            (k.1.val + HALF_DIM) * head_dim_stride
          ≠ s₀.pids 1 * q_token_stride + s₀.pids 0 * q_head_stride +
            j.val * head_dim_stride := by
      intro k _ hcon
      have h2 := Nat.add_left_cancel hcon
      have h3 : j.val * head_dim_stride
          < (k.1.val + HALF_DIM) * head_dim_stride :=
        Nat.mul_lt_mul_of_pos_right (by have := j.isLt; omega) hstride
      omega
    rw [foldl_store_preserve_readMem _ _ _ _ _ hmiss,
      BlockState.scatter_readback_nd _ _ _ hInj1 (j, PUnit.unit)]
    rw [hq0 j, hc j, hq1 j, hs j]
  · -- second-half channel: the outermost scatter reads back directly
    intro j
    rw [BlockState.scatter_readback_nd _ _ _ hInj2 (j, PUnit.unit)]
    rw [hq0 j, hs j, hq1 j, hc j]
  · -- frame: neither scatter touches a cell outside its own window
    intro r o h1 h2
    refine Eq.trans (foldl_store_preserve_cell _ _ r o _ _ ?_)
      (Eq.trans (foldl_store_preserve_cell _ _ r o _ _ ?_) rfl)
    · rintro k _ ⟨hqr, ho⟩
      rcases h2 with hne | hno
      · exact hne hqr.symm
      · exact hno k.1 ho.symm
    · rintro k _ ⟨hqr, ho⟩
      rcases h1 with hne | hno
      · exact hne hqr.symm
      · exact hno k.1 ho.symm

/-- The Q rotary surface sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, unmasked loads/stores, elementwise multiply/add/sub). -/
theorem decoding_fused_rotary_embedding_q_surface_flattenOk
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat) :
    ((decoding_fused_rotary_embedding_q_surface q cos sin q_token_stride
      q_head_stride head_dim_stride cos_token_stride cos_stride (HALF_DIM * 2)
      HALF_DIM).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [decoding_fused_rotary_embedding_q_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Per-execution safety walk: the surface's four loads and two stores address
the two Q half-windows and the shared `cos`/`sin` window, all unmasked, so the
bounds contract is lane-wise over *every* lane of each window. -/
theorem decoding_fused_rotary_embedding_q_surface_traceSafe
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h0 : ∀ j : Fin HALF_DIM, qFirstAddr (s.pids 0) (s.pids 1) q_token_stride
      q_head_stride head_dim_stride HALF_DIM j < bounds q)
    (h1 : ∀ j : Fin HALF_DIM, qSecondAddr (s.pids 0) (s.pids 1) q_token_stride
      q_head_stride head_dim_stride HALF_DIM j < bounds q)
    (h2 : ∀ j : Fin HALF_DIM, cosSinAddr (s.pids 1) cos_token_stride cos_stride
      HALF_DIM j < bounds cos)
    (h3 : ∀ j : Fin HALF_DIM, cosSinAddr (s.pids 1) cos_token_stride cos_stride
      HALF_DIM j < bounds sin) :
    Kernel.TraceSafe bounds ((decoding_fused_rotary_embedding_q_surface q cos
      sin q_token_stride q_head_stride head_dim_stride cos_token_stride
      cos_stride (HALF_DIM * 2) HALF_DIM).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp only [qFirstAddr, qSecondAddr, cosSinAddr] at h0 h1 h2 h3
  simp [decoding_fused_rotary_embedding_q_surface, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
    stepStmt, stepStmts, evalOp.eq_def,
    Tile.bop, Tile.ptrAdd, NumericDType.add, NumericDType.mul, NumericDType.sub,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]
  exact ⟨h0, h1, h2, h3, h0, h1⟩

/-- The Q rotary surface's **IO signature** — the whole kernel-specific audit
surface of the `⊨` headline: which buffer is which channel, where program
`(pid₀, pid₁)` reads and writes, and the (here total) activity predicates.

Four input channels — `0` the Q first half, `1` the Q second half, `2` `cos`,
`3` `sin` — and two output channels, `0` the Q first half and `1` the Q second
half. Both output channels name the **same** `q` buffer as input channels `0`
and `1`: the rotation is in place, which is why `bufs` (the allocation list,
`[q, cos, sin]`) is decoupled from the channel tables. Every lane is active:
the Python kernel carries no mask on this face. -/
def decodingRotaryQIO (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat) : GroupedMasked2DKernelIO where
  kernel := decoding_fused_rotary_embedding_q_surface q cos sin q_token_stride
    q_head_stride head_dim_stride cos_token_stride cos_stride (HALF_DIM * 2)
    HALF_DIM
  nIn := 4
  nOut := 2
  bufs := [q, cos, sin]
  inp := fun
    | ⟨0, _⟩ => q
    | ⟨1, _⟩ => q
    | ⟨2, _⟩ => cos
    | ⟨_ + 3, _⟩ => sin
  out := fun
    | ⟨0, _⟩ => q
    | ⟨_ + 1, _⟩ => q
  B := HALF_DIM
  read := fun
    | ⟨0, _⟩ => fun p₀ p₁ j =>
        qFirstAddr p₀ p₁ q_token_stride q_head_stride head_dim_stride HALF_DIM j
    | ⟨1, _⟩ => fun p₀ p₁ j =>
        qSecondAddr p₀ p₁ q_token_stride q_head_stride head_dim_stride HALF_DIM j
    | ⟨2, _⟩ => fun _p₀ p₁ j =>
        cosSinAddr p₁ cos_token_stride cos_stride HALF_DIM j
    | ⟨_ + 3, _⟩ => fun _p₀ p₁ j =>
        cosSinAddr p₁ cos_token_stride cos_stride HALF_DIM j
  readMask := fun _ _ _ _ => True
  write := fun
    | ⟨0, _⟩ => fun p₀ p₁ j =>
        qFirstAddr p₀ p₁ q_token_stride q_head_stride head_dim_stride HALF_DIM j
    | ⟨_ + 1, _⟩ => fun p₀ p₁ j =>
        qSecondAddr p₀ p₁ q_token_stride q_head_stride head_dim_stride HALF_DIM j
  writeMask := fun _ _ _ _ => True

/-! ### ════════ ★ TOP SPECIFICATION ★ ════════ -/
/-- **`decodingRotaryQIO ⊨ rotary`** — the unconditional Q rotary face of
`decoding_fused_rotary_embedding_kernel` as one Hoare triple over flat pointer
memory: for every disjoint base-pointer placement of `q`/`cos`/`sin`, every
program `(pid₀, pid₁)` whose windows are in bounds, and every launch state
whose four input windows hold `q0`, `q1`, `cos`, `sin`, the translated pointer
kernel terminates, the Q first half ends up holding `q0 · cos - q1 · sin`, the
Q second half `q0 · sin + q1 · cos`, and every flat cell outside the two Q
half-windows is untouched.

Dimension-general: the rotary half width `HALF_DIM` and all four strides are
free `Nat` parameters. The single side condition `0 < head_dim_stride` is what
makes the two in-place half-windows disjoint (and each of them injective) —
without it the two stores would alias and no closed form could hold. -/
specification decoding_fused_rotary_embedding_q_correctness
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (hstride : 0 < head_dim_stride) :
    decodingRotaryQIO q cos sin q_token_stride q_head_stride head_dim_stride
        cos_token_stride cos_stride HALF_DIM ⊨
      fun _pid₀ _pid₁ xs o j =>
        let q0 := xs (⟨0, by decide⟩ : Fin 4) j
        let q1 := xs (⟨1, by decide⟩ : Fin 4) j
        let c := xs (⟨2, by decide⟩ : Fin 4) j
        let sn := xs (⟨3, by decide⟩ : Fin 4) j
        match o with
        | ⟨0, _⟩ => q0 * c - q1 * sn
        | ⟨_ + 1, _⟩ => q0 * sn + q1 * c := by
  refine GroupedMasked2DKernelIO.Implements.intro _ ?_ ?_ ?_ ?_
  · -- both output channels live in the declared allocation list
    intro o
    fin_cases o <;> simp [decodingRotaryQIO]
  · exact decoding_fused_rotary_embedding_q_surface_flattenOk q cos sin
      q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM
  · intro bounds s hin _hout
    exact decoding_fused_rotary_embedding_q_surface_traceSafe q cos sin
      q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM bounds s
      (fun j => hin (⟨0, by decide⟩ : Fin 4) j trivial)
      (fun j => hin (⟨1, by decide⟩ : Fin 4) j trivial)
      (fun j => hin (⟨2, by decide⟩ : Fin 4) j trivial)
      (fun j => hin (⟨3, by decide⟩ : Fin 4) j trivial)
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hv0, hv1, hframe⟩ :=
      decoding_fused_rotary_embedding_q_surface_region_run q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM hstride s₀
        (xs (⟨0, by decide⟩ : Fin 4)) (xs (⟨1, by decide⟩ : Fin 4))
        (xs (⟨2, by decide⟩ : Fin 4)) (xs (⟨3, by decide⟩ : Fin 4))
        (fun j => hx (⟨0, by decide⟩ : Fin 4) j trivial)
        (fun j => hx (⟨1, by decide⟩ : Fin 4) j trivial)
        (fun j => hx (⟨2, by decide⟩ : Fin 4) j trivial)
        (fun j => hx (⟨3, by decide⟩ : Fin 4) j trivial)
    refine ⟨s1, hexec, ?_, ?_⟩
    · rintro ⟨o, ho⟩ j _
      match o, ho with
      | 0, _ => exact hv0 j
      | _ + 1, _ => exact hv1 j
    · intro r o' hcond
      refine hframe r o' ?_ ?_
      · by_cases hr : r = q
        · refine Or.inr (fun j => ?_)
          rcases hcond (⟨0, by decide⟩ : Fin 2) j trivial with h | h
          · exact absurd hr h
          · exact h
        · exact Or.inl hr
      · by_cases hr : r = q
        · refine Or.inr (fun j => ?_)
          rcases hcond (⟨1, by decide⟩ : Fin 2) j trivial with h | h
          · exact absurd hr h
          · exact h
        · exact Or.inl hr



/-! ## ════════ ★ Honest chained-metadata cache faces (block_id redeemed) ★ ════════

The K-cache and V-cache paged stores below are stated on the brand-new
`ChainMetaGroupedMasked2DKernelIO` skin (two chained `.nat` slots: slot 1 =
`context_lengths[cur_token_idx]`, slot 2 = the `BLOCK_TABLES` cell whose address
eats slot 1's loaded value). Unlike the `*_store_slice` faces above — which pin
`block_id` as a host `Nat` and read a pre-rotated value from a scratch buffer —
these kernels perform the genuine in-kernel `context_lengths → BLOCK_TABLES`
chained loads and the genuine rotary combination, so the headlines below no
longer NEED the pinned `block_id` parameter. The `handle_kv` guard lives in every
window/mask; block-id aliasing enters only through the skin's per-context
`WriteInj` readback gates (never as a `∀` hypothesis), plus — for the two K-cache
half stores sharing `k_cache` — a `hCross` half-window disjointness side
condition (the honest analogue of `kv_cache_copy`'s `KCache ≠ VCache`). -/

set_option maxHeartbeats 5000000

/-- Honest chained-metadata K-cache slice: unconditional metadata + data loads,
in-kernel rotation, both K-cache half stores, all guarded by `handle_kv`. -/
def decoding_fused_rotary_embedding_kcache_chain
    (k cos sin k_cache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM x k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride bts_stride btb_stride block_size HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    cur_k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    dim_range0 = tl.arange(0, $(HALF_DIM))
    dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
    off_kv = cur_token_idx * $(k_token_stride) + cur_k_head_idx * $(k_head_stride)
    loaded_k0 = tl.load(k + off_kv + dim_range0 * $(head_dim_stride))
    loaded_k1 = tl.load(k + off_kv + dim_range1 * $(head_dim_stride))
    off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
    loaded_cos = tl.load(cos + off_cos_sin)
    loaded_sin = tl.load(sin + off_cos_sin)
    out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
    out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos
    past_kv_seq_len = tl.load(context_lengths + cur_token_idx) - $(1)
    last_block_idx = past_kv_seq_len // $(block_size)
    block_ids = tl.load(BLOCK_TABLES + cur_token_idx * $(bts_stride) +
      last_block_idx * $(btb_stride))
    offsets_in_last_block = past_kv_seq_len % $(block_size)
    offsets_cache_base = block_ids * $(kcb_stride) + cur_k_head_idx * $(kch_stride)
    k_range0 = offsets_cache_base + offsets_in_last_block * $(kcs_stride) +
      (dim_range0 // $(x)) * $(kcsplit_x_stride) + (dim_range0 % $(x)) * $(kcd_stride)
    k_range1 = offsets_cache_base + offsets_in_last_block * $(kcs_stride) +
      (dim_range1 // $(x)) * $(kcsplit_x_stride) + (dim_range1 % $(x)) * $(kcd_stride)
    tl.store(k_cache + k_range0, out_k0)
    tl.store(k_cache + k_range1, out_k1)
  }
}

theorem decoding_fused_rotary_embedding_kcache_chain_flattenOk
    (k cos sin k_cache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM x k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride bts_stride btb_stride block_size HALF_DIM : Nat) :
    ((decoding_fused_rotary_embedding_kcache_chain k cos sin k_cache BLOCK_TABLES context_lengths KV_GROUP_NUM x
      k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride bts_stride
      btb_stride block_size HALF_DIM).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [decoding_fused_rotary_embedding_kcache_chain, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Unmasked intra-region disjoint-offset frame (companion for seeing a cell
through a later same-region store whose offsets all differ). -/
private theorem foldl_writeMem_readMem_disjoint {shape : TileShape}
    (region : RegionName) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → ℝ) (l : List (TileIndex shape)) (s : BlockState)
    (o : Nat) (h : ∀ k ∈ l, offsetFn k ≠ o) :
    (l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).readMem
        region o = s.readMem region o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih _ (fun k hk => h k (List.mem_cons_of_mem hd hk)),
        BlockState.writeMem_readMem, if_neg]
      rintro ⟨_, hoff⟩
      exact (h hd List.mem_cons_self) hoff.symm

private theorem foldl_writeMem_mem_disjoint {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (r : RegionName) (o : Nat)
    (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).mem r o
      = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
        BlockState.writeMem_mem, if_neg]
      exact fun hc => hnot hd List.mem_cons_self ⟨hc.1.symm, hc.2.symm⟩

theorem decoding_fused_rotary_embedding_kcache_chain_traceSafe
    (k cos sin k_cache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM x k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride bts_stride btb_stride block_size HALF_DIM : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hb₁ : s.pids 1 < bounds context_lengths)
    (hb₂ : s.pids 1 * bts_stride +
      ((s.readMemValue .nat context_lengths (s.pids 1) - 1) / block_size) *
        btb_stride < bounds BLOCK_TABLES)
    (hrk0 : ∀ j : Fin HALF_DIM, s.pids 0 % KV_GROUP_NUM = 0 →
      s.pids 1 * k_token_stride + (s.pids 0 / KV_GROUP_NUM) * k_head_stride +
        j.val * head_dim_stride < bounds k)
    (hrk1 : ∀ j : Fin HALF_DIM, s.pids 0 % KV_GROUP_NUM = 0 →
      s.pids 1 * k_token_stride + (s.pids 0 / KV_GROUP_NUM) * k_head_stride +
        (j.val + HALF_DIM) * head_dim_stride < bounds k)
    (hrcos : ∀ j : Fin HALF_DIM, s.pids 0 % KV_GROUP_NUM = 0 →
      s.pids 1 * cos_token_stride + j.val * cos_stride < bounds cos)
    (hrsin : ∀ j : Fin HALF_DIM, s.pids 0 % KV_GROUP_NUM = 0 →
      s.pids 1 * cos_token_stride + j.val * cos_stride < bounds sin)
    (hwk0 : ∀ j : Fin HALF_DIM, s.pids 0 % KV_GROUP_NUM = 0 →
      s.readMemValue .nat BLOCK_TABLES
          (s.pids 1 * bts_stride +
            ((s.readMemValue .nat context_lengths (s.pids 1) - 1) / block_size) *
              btb_stride) * kcb_stride +
        (s.pids 0 / KV_GROUP_NUM) * kch_stride +
        ((s.readMemValue .nat context_lengths (s.pids 1) - 1) % block_size) *
          kcs_stride +
        (j.val / x) * kcsplit_x_stride + (j.val % x) * kcd_stride < bounds k_cache)
    (hwk1 : ∀ j : Fin HALF_DIM, s.pids 0 % KV_GROUP_NUM = 0 →
      s.readMemValue .nat BLOCK_TABLES
          (s.pids 1 * bts_stride +
            ((s.readMemValue .nat context_lengths (s.pids 1) - 1) / block_size) *
              btb_stride) * kcb_stride +
        (s.pids 0 / KV_GROUP_NUM) * kch_stride +
        ((s.readMemValue .nat context_lengths (s.pids 1) - 1) % block_size) *
          kcs_stride +
        ((j.val + HALF_DIM) / x) * kcsplit_x_stride +
        ((j.val + HALF_DIM) % x) * kcd_stride < bounds k_cache) :
    Kernel.TraceSafe bounds
      ((decoding_fused_rotary_embedding_kcache_chain k cos sin k_cache BLOCK_TABLES context_lengths KV_GROUP_NUM x
        k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
        kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride bts_stride
        btb_stride block_size HALF_DIM).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  by_cases hHandle : s.pids 0 % KV_GROUP_NUM = 0
  · simp [decoding_fused_rotary_embedding_kcache_chain, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
      MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
      MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
      tile_elementwise, Bool.and_eq_true,
      Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
      ComparableDType.eq, BlockState.readMemValue, hHandle, hb₁]
    refine ⟨fun a => hrk0 a hHandle, fun a => hrk1 a hHandle, fun a => hrcos a hHandle,
      fun a => hrsin a hHandle, ?_, fun a => ?_, fun a => ?_⟩
    · simpa only [BlockState.readMemValue] using hb₂
    · simpa only [BlockState.readMemValue] using hwk0 a hHandle
    · simpa only [BlockState.readMemValue] using hwk1 a hHandle
  · simp [decoding_fused_rotary_embedding_kcache_chain, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
      MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
      MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
      tile_elementwise, Bool.and_eq_true,
      Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
      ComparableDType.eq, BlockState.readMemValue, hHandle]

/-- Prototype: existence of exec (guard handling). -/
theorem decoding_fused_rotary_embedding_kcache_chain_exec_exists
    (k cos sin k_cache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM x k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride bts_stride btb_stride block_size HALF_DIM : Nat)
    (s₀ : BlockState) :
    ∃ s1, exec ((decoding_fused_rotary_embedding_kcache_chain k cos sin k_cache BLOCK_TABLES context_lengths
      KV_GROUP_NUM x k_token_stride k_head_stride head_dim_stride cos_token_stride
      cos_stride kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
      bts_stride btb_stride block_size HALF_DIM).toAlgKernel) s₀ = some s1 := by
  by_cases hHandle : s₀.pids 0 % KV_GROUP_NUM = 0 <;>
  simp [exec, decoding_fused_rotary_embedding_kcache_chain, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
    Tile.bop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.eq,
    BlockState.readMemValue, hHandle]

set_option maxHeartbeats 20000000 in
theorem decoding_fused_rotary_embedding_kcache_chain_region_run
    (k cos sin k_cache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM x k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride bts_stride btb_stride block_size HALF_DIM : Nat)
    (s₀ : BlockState) (m₁ m₂ : Nat) (xs : Fin 4 → Fin HALF_DIM → ℝ)
    (hCross : ∀ j l : Fin HALF_DIM,
      m₂ * kcb_stride + (s₀.pids 0 / KV_GROUP_NUM) * kch_stride +
          ((m₁ - 1) % block_size) * kcs_stride +
          (j.val / x) * kcsplit_x_stride + (j.val % x) * kcd_stride
        ≠ m₂ * kcb_stride + (s₀.pids 0 / KV_GROUP_NUM) * kch_stride +
          ((m₁ - 1) % block_size) * kcs_stride +
          ((l.val + HALF_DIM) / x) * kcsplit_x_stride +
          ((l.val + HALF_DIM) % x) * kcd_stride)
    (hm₁ : s₀.readMemValue .nat context_lengths (s₀.pids 1) = m₁)
    (hm₂ : s₀.readMemValue .nat BLOCK_TABLES
      (s₀.pids 1 * bts_stride + ((m₁ - 1) / block_size) * btb_stride) = m₂)
    (hk0 : ∀ j : Fin HALF_DIM, s₀.pids 0 % KV_GROUP_NUM = 0 →
      s₀.readMem k (s₀.pids 1 * k_token_stride +
        (s₀.pids 0 / KV_GROUP_NUM) * k_head_stride + j.val * head_dim_stride)
        = xs 0 j)
    (hk1 : ∀ j : Fin HALF_DIM, s₀.pids 0 % KV_GROUP_NUM = 0 →
      s₀.readMem k (s₀.pids 1 * k_token_stride +
        (s₀.pids 0 / KV_GROUP_NUM) * k_head_stride + (j.val + HALF_DIM) * head_dim_stride)
        = xs 1 j)
    (hcos : ∀ j : Fin HALF_DIM, s₀.pids 0 % KV_GROUP_NUM = 0 →
      s₀.readMem cos (s₀.pids 1 * cos_token_stride + j.val * cos_stride) = xs 2 j)
    (hsin : ∀ j : Fin HALF_DIM, s₀.pids 0 % KV_GROUP_NUM = 0 →
      s₀.readMem sin (s₀.pids 1 * cos_token_stride + j.val * cos_stride) = xs 3 j) :
    ∃ s1, exec ((decoding_fused_rotary_embedding_kcache_chain k cos sin k_cache BLOCK_TABLES context_lengths
        KV_GROUP_NUM x k_token_stride k_head_stride head_dim_stride cos_token_stride
        cos_stride kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
        bts_stride btb_stride block_size HALF_DIM).toAlgKernel) s₀ = some s1
      ∧ ((∀ j l : Fin HALF_DIM, s₀.pids 0 % KV_GROUP_NUM = 0 →
            s₀.pids 0 % KV_GROUP_NUM = 0 →
            m₂ * kcb_stride + (s₀.pids 0 / KV_GROUP_NUM) * kch_stride +
                ((m₁ - 1) % block_size) * kcs_stride +
                (j.val / x) * kcsplit_x_stride + (j.val % x) * kcd_stride
              = m₂ * kcb_stride + (s₀.pids 0 / KV_GROUP_NUM) * kch_stride +
                ((m₁ - 1) % block_size) * kcs_stride +
                (l.val / x) * kcsplit_x_stride + (l.val % x) * kcd_stride → j = l) →
          ∀ j : Fin HALF_DIM, s₀.pids 0 % KV_GROUP_NUM = 0 →
            s1.readMem k_cache
              (m₂ * kcb_stride + (s₀.pids 0 / KV_GROUP_NUM) * kch_stride +
                ((m₁ - 1) % block_size) * kcs_stride +
                (j.val / x) * kcsplit_x_stride + (j.val % x) * kcd_stride)
              = xs 0 j * xs 2 j - xs 1 j * xs 3 j)
      ∧ ((∀ j l : Fin HALF_DIM, s₀.pids 0 % KV_GROUP_NUM = 0 →
            s₀.pids 0 % KV_GROUP_NUM = 0 →
            m₂ * kcb_stride + (s₀.pids 0 / KV_GROUP_NUM) * kch_stride +
                ((m₁ - 1) % block_size) * kcs_stride +
                ((j.val + HALF_DIM) / x) * kcsplit_x_stride +
                ((j.val + HALF_DIM) % x) * kcd_stride
              = m₂ * kcb_stride + (s₀.pids 0 / KV_GROUP_NUM) * kch_stride +
                ((m₁ - 1) % block_size) * kcs_stride +
                ((l.val + HALF_DIM) / x) * kcsplit_x_stride +
                ((l.val + HALF_DIM) % x) * kcd_stride → j = l) →
          ∀ j : Fin HALF_DIM, s₀.pids 0 % KV_GROUP_NUM = 0 →
            s1.readMem k_cache
              (m₂ * kcb_stride + (s₀.pids 0 / KV_GROUP_NUM) * kch_stride +
                ((m₁ - 1) % block_size) * kcs_stride +
                ((j.val + HALF_DIM) / x) * kcsplit_x_stride +
                ((j.val + HALF_DIM) % x) * kcd_stride)
              = xs 0 j * xs 3 j + xs 1 j * xs 2 j)
      ∧ (∀ r o,
          (r ≠ k_cache ∨ ∀ j : Fin HALF_DIM, s₀.pids 0 % KV_GROUP_NUM = 0 →
            o ≠ m₂ * kcb_stride + (s₀.pids 0 / KV_GROUP_NUM) * kch_stride +
              ((m₁ - 1) % block_size) * kcs_stride +
              (j.val / x) * kcsplit_x_stride + (j.val % x) * kcd_stride) →
          (r ≠ k_cache ∨ ∀ j : Fin HALF_DIM, s₀.pids 0 % KV_GROUP_NUM = 0 →
            o ≠ m₂ * kcb_stride + (s₀.pids 0 / KV_GROUP_NUM) * kch_stride +
              ((m₁ - 1) % block_size) * kcs_stride +
              ((j.val + HALF_DIM) / x) * kcsplit_x_stride +
              ((j.val + HALF_DIM) % x) * kcd_stride) →
          s1.mem r o = s₀.mem r o) := by
  subst hm₂; subst hm₁
  by_cases hHandle : s₀.pids 0 % KV_GROUP_NUM = 0
  · -- handle_kv true: full content
    obtain ⟨s1, hs1⟩ : ∃ s1, exec ((decoding_fused_rotary_embedding_kcache_chain k cos sin k_cache BLOCK_TABLES
        context_lengths KV_GROUP_NUM x k_token_stride k_head_stride head_dim_stride
        cos_token_stride cos_stride kcb_stride kch_stride kcsplit_x_stride kcs_stride
        kcd_stride bts_stride btb_stride block_size HALF_DIM).toAlgKernel) s₀ = some s1 :=
      decoding_fused_rotary_embedding_kcache_chain_exec_exists k cos sin k_cache BLOCK_TABLES context_lengths
        KV_GROUP_NUM x k_token_stride k_head_stride head_dim_stride cos_token_stride
        cos_stride kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
        bts_stride btb_stride block_size HALF_DIM s₀
    have hs1' := hs1
    simp [exec, decoding_fused_rotary_embedding_kcache_chain, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
      Tile.bop, Tile.ptrAdd, Tile.uop,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.eq,
      BlockState.readMemValue, hHandle] at hs1'
    refine ⟨s1, hs1, ?_, ?_, ?_⟩
    · -- first-half readback (see through the later second-half store)
      intro hInj j _
      rw [← hs1']
      simp only [BlockState.readMemValue]
      rw [foldl_writeMem_readMem_disjoint (shape := [HALF_DIM]) k_cache _ _ _ _ _
        (fun kidx _ => by
          simpa only [BlockState.readMemValue] using
            (hCross j ⟨kidx.1.val, kidx.1.isLt⟩).symm)]
      rw [BlockState.scatter_readback_nd (shape := [HALF_DIM]) _ _ _ ?_ (j, PUnit.unit)]
      · simp only [BlockState.readMemValue]
        rw [hk0 j hHandle, hk1 j hHandle, hcos j hHandle, hsin j hHandle]
      · rintro ⟨a, ua⟩ ⟨b, ub⟩ hab
        cases ua; cases ub
        simp only [BlockState.readMemValue] at hab
        have := hInj ⟨a.val, a.isLt⟩ ⟨b.val, b.isLt⟩ hHandle hHandle (by simpa using hab)
        simpa using this
    · -- second-half readback (outer store, direct)
      intro hInj j _
      rw [← hs1']
      simp only [BlockState.readMemValue]
      rw [BlockState.scatter_readback_nd (shape := [HALF_DIM]) _ _ _ ?_ (j, PUnit.unit)]
      · simp only [BlockState.readMemValue]
        rw [hk0 j hHandle, hk1 j hHandle, hcos j hHandle, hsin j hHandle]
      · rintro ⟨a, ua⟩ ⟨b, ub⟩ hab
        cases ua; cases ub
        simp only [BlockState.readMemValue] at hab
        have := hInj ⟨a.val, a.isLt⟩ ⟨b.val, b.isLt⟩ hHandle hHandle (by simpa using hab)
        simpa using this
    · -- frame
      intro r o hc1 hc2
      rw [← hs1']
      rw [foldl_writeMem_mem_disjoint (region := k_cache) _ _ r o _ _ (fun kidx _ hc => ?_)]
      rw [foldl_writeMem_mem_disjoint (region := k_cache) _ _ r o _ _ (fun kidx _ hc => ?_)]
      · rfl
      · -- inner (first-half) store misses (r,o)
        rcases hc1 with hne | hno
        · exact hne hc.1.symm
        · exact hno ⟨kidx.1.val, kidx.1.isLt⟩ hHandle
            (by simpa only [BlockState.readMemValue] using hc.2.symm)
      · -- outer (second-half) store misses (r,o)
        rcases hc2 with hne | hno
        · exact hne hc.1.symm
        · exact hno ⟨kidx.1.val, kidx.1.isLt⟩ hHandle
            (by simpa only [BlockState.readMemValue] using hc.2.symm)
  · -- handle_kv false: stores skipped, value legs vacuous, frame trivial
    obtain ⟨s1, hs1⟩ : ∃ s1, exec ((decoding_fused_rotary_embedding_kcache_chain k cos sin k_cache BLOCK_TABLES
        context_lengths KV_GROUP_NUM x k_token_stride k_head_stride head_dim_stride
        cos_token_stride cos_stride kcb_stride kch_stride kcsplit_x_stride kcs_stride
        kcd_stride bts_stride btb_stride block_size HALF_DIM).toAlgKernel) s₀ = some s1 :=
      decoding_fused_rotary_embedding_kcache_chain_exec_exists k cos sin k_cache BLOCK_TABLES context_lengths
        KV_GROUP_NUM x k_token_stride k_head_stride head_dim_stride cos_token_stride
        cos_stride kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
        bts_stride btb_stride block_size HALF_DIM s₀
    have hs1' := hs1
    simp [exec, decoding_fused_rotary_embedding_kcache_chain, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
      Tile.bop, Tile.ptrAdd, Tile.uop,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.eq,
      BlockState.readMemValue, hHandle] at hs1'
    refine ⟨s1, hs1, ?_, ?_, ?_⟩
    · exact fun _ j hj => absurd hj hHandle
    · exact fun _ j hj => absurd hj hHandle
    · intro r o _ _
      rw [← hs1']
      rfl

/-- Factored value function: the two rotary K-cache half closed forms over the
loaded input channels (0=k0, 1=k1, 2=cos, 3=sin; output 0=first half, 1=second). -/
def decodingKCacheChainSpec : Nat → Nat → Nat → Nat → (Fin 4 → Fin HALF_DIM → ℝ) → Fin 2 →
    Fin HALF_DIM → ℝ :=
  fun _ _ _ _ xs o j =>
    match o with
    | ⟨0, _⟩ => xs 0 j * xs 2 j - xs 1 j * xs 3 j
    | ⟨_ + 1, _⟩ => xs 0 j * xs 3 j + xs 1 j * xs 2 j

/-- The chained-metadata grouped IO signature for the honest K-cache slice. -/
def decodingKCacheChainIO
    (k cos sin k_cache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM x k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride bts_stride btb_stride block_size HALF_DIM : Nat) :
    ChainMetaGroupedMasked2DKernelIO where
  kernel := decoding_fused_rotary_embedding_kcache_chain k cos sin k_cache BLOCK_TABLES context_lengths KV_GROUP_NUM
    x k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
    kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride bts_stride
    btb_stride block_size HALF_DIM
  nIn := 4
  nOut := 2
  bufs := [k, cos, sin, k_cache, context_lengths, BLOCK_TABLES]
  mbuf1 := context_lengths
  mbuf2 := BLOCK_TABLES
  inp := fun i => match i with
    | ⟨0, _⟩ => k
    | ⟨1, _⟩ => k
    | ⟨2, _⟩ => cos
    | ⟨_ + 3, _⟩ => sin
  out := fun _ => k_cache
  B := HALF_DIM
  mwin1 := fun _ pid₁ => pid₁
  mwin2 := fun _ pid₁ m₁ => pid₁ * bts_stride + ((m₁ - 1) / block_size) * btb_stride
  read := fun i pid₀ pid₁ _ _ j => match i with
    | ⟨0, _⟩ => pid₁ * k_token_stride + (pid₀ / KV_GROUP_NUM) * k_head_stride +
        j.val * head_dim_stride
    | ⟨1, _⟩ => pid₁ * k_token_stride + (pid₀ / KV_GROUP_NUM) * k_head_stride +
        (j.val + HALF_DIM) * head_dim_stride
    | ⟨2, _⟩ => pid₁ * cos_token_stride + j.val * cos_stride
    | ⟨_ + 3, _⟩ => pid₁ * cos_token_stride + j.val * cos_stride
  readMask := fun _ pid₀ _ _ _ _ => pid₀ % KV_GROUP_NUM = 0
  write := fun o pid₀ _ m₁ m₂ j => match o with
    | ⟨0, _⟩ => m₂ * kcb_stride + (pid₀ / KV_GROUP_NUM) * kch_stride +
        ((m₁ - 1) % block_size) * kcs_stride +
        (j.val / x) * kcsplit_x_stride + (j.val % x) * kcd_stride
    | ⟨_ + 1, _⟩ => m₂ * kcb_stride + (pid₀ / KV_GROUP_NUM) * kch_stride +
        ((m₁ - 1) % block_size) * kcs_stride +
        ((j.val + HALF_DIM) / x) * kcsplit_x_stride +
        ((j.val + HALF_DIM) % x) * kcd_stride
  writeMask := fun _ pid₀ _ _ _ _ => pid₀ % KV_GROUP_NUM = 0

open scoped VeriTile.Triton.ChainMetaGroupedMasked2DKernelIO in
theorem decoding_fused_rotary_embedding_kcache_chain_correctness
    (k cos sin k_cache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM x k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride bts_stride btb_stride block_size HALF_DIM : Nat)
    (hCross : ∀ (pid₀ _pid₁ m₁ m₂ : Nat) (j l : Fin HALF_DIM),
      m₂ * kcb_stride + (pid₀ / KV_GROUP_NUM) * kch_stride +
          ((m₁ - 1) % block_size) * kcs_stride +
          (j.val / x) * kcsplit_x_stride + (j.val % x) * kcd_stride
        ≠ m₂ * kcb_stride + (pid₀ / KV_GROUP_NUM) * kch_stride +
          ((m₁ - 1) % block_size) * kcs_stride +
          ((l.val + HALF_DIM) / x) * kcsplit_x_stride +
          ((l.val + HALF_DIM) % x) * kcd_stride) :
    decodingKCacheChainIO k cos sin k_cache BLOCK_TABLES context_lengths KV_GROUP_NUM x
        k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
        kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride bts_stride
        btb_stride block_size HALF_DIM
      ⊨ decodingKCacheChainSpec := by
  refine ChainMetaGroupedMasked2DKernelIO.Implements.intro _ ?_ ?_ ?_ ?_
  · -- hout
    intro o
    simp only [decodingKCacheChainIO, List.mem_cons]
    tauto
  · -- hok
    exact decoding_fused_rotary_embedding_kcache_chain_flattenOk k cos sin k_cache BLOCK_TABLES context_lengths
      KV_GROUP_NUM x k_token_stride k_head_stride head_dim_stride cos_token_stride
      cos_stride kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
      bts_stride btb_stride block_size HALF_DIM
  · -- hts
    intro bounds s s1 s2 hm₁ hm₂ hb1 hb2 hbr hbw
    simp only [decodingKCacheChainIO] at hm₁ hm₂ hb1 hb2 hbr hbw
    refine decoding_fused_rotary_embedding_kcache_chain_traceSafe k cos sin k_cache BLOCK_TABLES context_lengths
      KV_GROUP_NUM x k_token_stride k_head_stride head_dim_stride cos_token_stride
      cos_stride kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
      bts_stride btb_stride block_size HALF_DIM bounds s hb1 ?_ ?_ ?_ ?_ ?_ ?_ ?_
    · rw [← hm₁] at hb2; exact hb2
    · exact fun j hj => hbr ⟨0, by omega⟩ j hj
    · exact fun j hj => hbr ⟨1, by omega⟩ j hj
    · exact fun j hj => hbr ⟨2, by omega⟩ j hj
    · exact fun j hj => hbr ⟨3, by omega⟩ j hj
    · intro j hj
      have h := hbw ⟨0, by omega⟩ j hj
      rw [← hm₂, ← hm₁] at h; exact h
    · intro j hj
      have h := hbw ⟨1, by omega⟩ j hj
      rw [← hm₂, ← hm₁] at h; exact h
  · -- hrun
    intro s₀ s1 s2 xs hm₁ hm₂ hx
    simp only [decodingKCacheChainIO] at hm₁ hm₂ hx
    obtain ⟨s1', hexec, hval0, hval1, hframe⟩ :=
      decoding_fused_rotary_embedding_kcache_chain_region_run k cos sin k_cache BLOCK_TABLES context_lengths
        KV_GROUP_NUM x k_token_stride k_head_stride head_dim_stride cos_token_stride
        cos_stride kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
        bts_stride btb_stride block_size HALF_DIM s₀ s1 s2 xs
        (fun j l => hCross (s₀.pids 0) (s₀.pids 1) s1 s2 j l)
        hm₁ hm₂
        (fun j hj => hx ⟨0, by omega⟩ j hj)
        (fun j hj => hx ⟨1, by omega⟩ j hj)
        (fun j hj => hx ⟨2, by omega⟩ j hj)
        (fun j hj => hx ⟨3, by omega⟩ j hj)
    refine ⟨s1', hexec, ?_, ?_⟩
    · intro o hInj j hmask
      simp only [decodingKCacheChainIO, decodingKCacheChainSpec] at hInj hmask ⊢
      match o with
      | ⟨0, _⟩ => exact hval0 hInj j hmask
      | ⟨1, _⟩ => exact hval1 hInj j hmask
    · intro r o' hc
      simp only [decodingKCacheChainIO] at hc
      refine hframe r o' ?_ ?_
      · by_cases hr : r = k_cache
        · right; intro j hjh
          rcases hc ⟨0, by omega⟩ j hjh with h | h
          · exact absurd hr h
          · exact h
        · left; exact hr
      · by_cases hr : r = k_cache
        · right; intro j hjh
          rcases hc ⟨1, by omega⟩ j hjh with h | h
          · exact absurd hr h
          · exact h
        · left; exact hr

/-! ## V-cache honest chained face (single store, copy, B = HEAD_DIM) -/

def decoding_fused_rotary_embedding_vcache_chain
    (v v_cache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM k_token_stride k_head_stride head_dim_stride
      vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride
      block_size HEAD_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    cur_k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    dim_range = tl.arange(0, $(HEAD_DIM))
    off_kv = cur_token_idx * $(k_token_stride) + cur_k_head_idx * $(k_head_stride)
    loaded_v = tl.load(v + off_kv + dim_range * $(head_dim_stride))
    past_kv_seq_len = tl.load(context_lengths + cur_token_idx) - $(1)
    last_block_idx = past_kv_seq_len // $(block_size)
    block_ids = tl.load(BLOCK_TABLES + cur_token_idx * $(bts_stride) +
      last_block_idx * $(btb_stride))
    offsets_in_last_block = past_kv_seq_len % $(block_size)
    v_range = block_ids * $(vcb_stride) + cur_k_head_idx * $(vch_stride) +
      offsets_in_last_block * $(vcs_stride) + dim_range * $(vcd_stride)
    tl.store(v_cache + v_range, loaded_v)
  }
}

theorem decoding_fused_rotary_embedding_vcache_chain_flattenOk
    (v v_cache : RegionName) (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM k_token_stride k_head_stride head_dim_stride
      vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride
      block_size HEAD_DIM : Nat) :
    ((decoding_fused_rotary_embedding_vcache_chain v v_cache BLOCK_TABLES context_lengths KV_GROUP_NUM
      k_token_stride k_head_stride head_dim_stride vcb_stride vch_stride vcs_stride
      vcd_stride bts_stride btb_stride block_size HEAD_DIM).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [decoding_fused_rotary_embedding_vcache_chain, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem decoding_fused_rotary_embedding_vcache_chain_exec_exists
    (v v_cache : RegionName) (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM k_token_stride k_head_stride head_dim_stride
      vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride
      block_size HEAD_DIM : Nat) (s₀ : BlockState) :
    ∃ s1, exec ((decoding_fused_rotary_embedding_vcache_chain v v_cache BLOCK_TABLES context_lengths KV_GROUP_NUM
      k_token_stride k_head_stride head_dim_stride vcb_stride vch_stride vcs_stride
      vcd_stride bts_stride btb_stride block_size HEAD_DIM).toAlgKernel) s₀
      = some s1 := by
  by_cases hHandle : s₀.pids 0 % KV_GROUP_NUM = 0 <;>
  simp [exec, decoding_fused_rotary_embedding_vcache_chain, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
    Tile.bop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.eq,
    BlockState.readMemValue, hHandle]

theorem decoding_fused_rotary_embedding_vcache_chain_traceSafe
    (v v_cache : RegionName) (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM k_token_stride k_head_stride head_dim_stride
      vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride
      block_size HEAD_DIM : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hb₁ : s.pids 1 < bounds context_lengths)
    (hb₂ : s.pids 1 * bts_stride +
      ((s.readMemValue .nat context_lengths (s.pids 1) - 1) / block_size) *
        btb_stride < bounds BLOCK_TABLES)
    (hrv : ∀ j : Fin HEAD_DIM, s.pids 0 % KV_GROUP_NUM = 0 →
      s.pids 1 * k_token_stride + (s.pids 0 / KV_GROUP_NUM) * k_head_stride +
        j.val * head_dim_stride < bounds v)
    (hwv : ∀ j : Fin HEAD_DIM, s.pids 0 % KV_GROUP_NUM = 0 →
      s.readMemValue .nat BLOCK_TABLES
          (s.pids 1 * bts_stride +
            ((s.readMemValue .nat context_lengths (s.pids 1) - 1) / block_size) *
              btb_stride) * vcb_stride +
        (s.pids 0 / KV_GROUP_NUM) * vch_stride +
        ((s.readMemValue .nat context_lengths (s.pids 1) - 1) % block_size) *
          vcs_stride + j.val * vcd_stride < bounds v_cache) :
    Kernel.TraceSafe bounds
      ((decoding_fused_rotary_embedding_vcache_chain v v_cache BLOCK_TABLES context_lengths KV_GROUP_NUM
        k_token_stride k_head_stride head_dim_stride vcb_stride vch_stride vcs_stride
        vcd_stride bts_stride btb_stride block_size HEAD_DIM).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  by_cases hHandle : s.pids 0 % KV_GROUP_NUM = 0
  · simp [decoding_fused_rotary_embedding_vcache_chain, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
      MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
      MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
      tile_elementwise, Bool.and_eq_true,
      Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
      ComparableDType.eq, BlockState.readMemValue, hHandle, hb₁]
    refine ⟨fun a => hrv a hHandle, ?_, fun a => ?_⟩
    · simpa only [BlockState.readMemValue] using hb₂
    · simpa only [BlockState.readMemValue] using hwv a hHandle
  · simp [decoding_fused_rotary_embedding_vcache_chain, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
      MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
      MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
      tile_elementwise, Bool.and_eq_true,
      Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
      ComparableDType.eq, BlockState.readMemValue, hHandle]

set_option maxHeartbeats 20000000 in
theorem decoding_fused_rotary_embedding_vcache_chain_region_run
    (v v_cache : RegionName) (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM k_token_stride k_head_stride head_dim_stride
      vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride
      block_size HEAD_DIM : Nat)
    (s₀ : BlockState) (m₁ m₂ : Nat) (xs : Fin 1 → Fin HEAD_DIM → ℝ)
    (hm₁ : s₀.readMemValue .nat context_lengths (s₀.pids 1) = m₁)
    (hm₂ : s₀.readMemValue .nat BLOCK_TABLES
      (s₀.pids 1 * bts_stride + ((m₁ - 1) / block_size) * btb_stride) = m₂)
    (hv : ∀ j : Fin HEAD_DIM, s₀.pids 0 % KV_GROUP_NUM = 0 →
      s₀.readMem v (s₀.pids 1 * k_token_stride +
        (s₀.pids 0 / KV_GROUP_NUM) * k_head_stride + j.val * head_dim_stride)
        = xs 0 j) :
    ∃ s1, exec ((decoding_fused_rotary_embedding_vcache_chain v v_cache BLOCK_TABLES context_lengths KV_GROUP_NUM
        k_token_stride k_head_stride head_dim_stride vcb_stride vch_stride vcs_stride
        vcd_stride bts_stride btb_stride block_size HEAD_DIM).toAlgKernel) s₀
        = some s1
      ∧ ((∀ j l : Fin HEAD_DIM, s₀.pids 0 % KV_GROUP_NUM = 0 →
            s₀.pids 0 % KV_GROUP_NUM = 0 →
            m₂ * vcb_stride + (s₀.pids 0 / KV_GROUP_NUM) * vch_stride +
                ((m₁ - 1) % block_size) * vcs_stride + j.val * vcd_stride
              = m₂ * vcb_stride + (s₀.pids 0 / KV_GROUP_NUM) * vch_stride +
                ((m₁ - 1) % block_size) * vcs_stride + l.val * vcd_stride → j = l) →
          ∀ j : Fin HEAD_DIM, s₀.pids 0 % KV_GROUP_NUM = 0 →
            s1.readMem v_cache
              (m₂ * vcb_stride + (s₀.pids 0 / KV_GROUP_NUM) * vch_stride +
                ((m₁ - 1) % block_size) * vcs_stride + j.val * vcd_stride)
              = xs 0 j)
      ∧ (∀ r o,
          (r ≠ v_cache ∨ ∀ j : Fin HEAD_DIM, s₀.pids 0 % KV_GROUP_NUM = 0 →
            o ≠ m₂ * vcb_stride + (s₀.pids 0 / KV_GROUP_NUM) * vch_stride +
              ((m₁ - 1) % block_size) * vcs_stride + j.val * vcd_stride) →
          s1.mem r o = s₀.mem r o) := by
  subst hm₂; subst hm₁
  by_cases hHandle : s₀.pids 0 % KV_GROUP_NUM = 0
  · obtain ⟨s1, hs1⟩ := decoding_fused_rotary_embedding_vcache_chain_exec_exists v v_cache BLOCK_TABLES
      context_lengths KV_GROUP_NUM k_token_stride k_head_stride head_dim_stride
      vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride block_size
      HEAD_DIM s₀
    have hs1' := hs1
    simp [exec, decoding_fused_rotary_embedding_vcache_chain, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
      Tile.bop, Tile.ptrAdd, Tile.uop,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.eq,
      BlockState.readMemValue, hHandle] at hs1'
    refine ⟨s1, hs1, ?_, ?_⟩
    · intro hInj j _
      rw [← hs1']
      simp only [BlockState.readMemValue]
      rw [BlockState.scatter_readback_nd (shape := [HEAD_DIM]) _ _ _ ?_ (j, PUnit.unit)]
      · simp only [BlockState.readMemValue]
        rw [hv j hHandle]
      · rintro ⟨a, ua⟩ ⟨b, ub⟩ hab
        cases ua; cases ub
        simp only [BlockState.readMemValue] at hab
        have := hInj ⟨a.val, a.isLt⟩ ⟨b.val, b.isLt⟩ hHandle hHandle (by simpa using hab)
        simpa using this
    · intro r o hc
      rw [← hs1']
      rw [foldl_writeMem_mem_disjoint (region := v_cache) _ _ r o _ _ (fun kidx _ hcc => ?_)]
      · rfl
      · rcases hc with hne | hno
        · exact hne hcc.1.symm
        · exact hno ⟨kidx.1.val, kidx.1.isLt⟩ hHandle
            (by simpa only [BlockState.readMemValue] using hcc.2.symm)
  · obtain ⟨s1, hs1⟩ := decoding_fused_rotary_embedding_vcache_chain_exec_exists v v_cache BLOCK_TABLES
      context_lengths KV_GROUP_NUM k_token_stride k_head_stride head_dim_stride
      vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride block_size
      HEAD_DIM s₀
    have hs1' := hs1
    simp [exec, decoding_fused_rotary_embedding_vcache_chain, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
      Tile.bop, Tile.ptrAdd, Tile.uop,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.eq,
      BlockState.readMemValue, hHandle] at hs1'
    refine ⟨s1, hs1, ?_, ?_⟩
    · exact fun _ j hj => absurd hj hHandle
    · intro r o _
      rw [← hs1']; rfl

def decodingVCacheChainSpec : Nat → Nat → Nat → Nat → (Fin 1 → Fin HEAD_DIM → ℝ) → Fin 1 →
    Fin HEAD_DIM → ℝ :=
  fun _ _ _ _ xs _ j => xs 0 j

def decodingVCacheChainIO
    (v v_cache : RegionName) (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM k_token_stride k_head_stride head_dim_stride
      vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride
      block_size HEAD_DIM : Nat) :
    ChainMetaGroupedMasked2DKernelIO where
  kernel := decoding_fused_rotary_embedding_vcache_chain v v_cache BLOCK_TABLES context_lengths KV_GROUP_NUM
    k_token_stride k_head_stride head_dim_stride vcb_stride vch_stride vcs_stride
    vcd_stride bts_stride btb_stride block_size HEAD_DIM
  nIn := 1
  nOut := 1
  bufs := [v, v_cache, context_lengths, BLOCK_TABLES]
  mbuf1 := context_lengths
  mbuf2 := BLOCK_TABLES
  inp := fun _ => v
  out := fun _ => v_cache
  B := HEAD_DIM
  mwin1 := fun _ pid₁ => pid₁
  mwin2 := fun _ pid₁ m₁ => pid₁ * bts_stride + ((m₁ - 1) / block_size) * btb_stride
  read := fun _ pid₀ pid₁ _ _ j =>
    pid₁ * k_token_stride + (pid₀ / KV_GROUP_NUM) * k_head_stride +
      j.val * head_dim_stride
  readMask := fun _ pid₀ _ _ _ _ => pid₀ % KV_GROUP_NUM = 0
  write := fun _ pid₀ _ m₁ m₂ j =>
    m₂ * vcb_stride + (pid₀ / KV_GROUP_NUM) * vch_stride +
      ((m₁ - 1) % block_size) * vcs_stride + j.val * vcd_stride
  writeMask := fun _ pid₀ _ _ _ _ => pid₀ % KV_GROUP_NUM = 0

open scoped VeriTile.Triton.ChainMetaGroupedMasked2DKernelIO in
theorem decoding_fused_rotary_embedding_vcache_chain_correctness
    (v v_cache : RegionName) (BLOCK_TABLES context_lengths : Region .nat)
    (KV_GROUP_NUM k_token_stride k_head_stride head_dim_stride
      vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride
      block_size HEAD_DIM : Nat) :
    decodingVCacheChainIO v v_cache BLOCK_TABLES context_lengths KV_GROUP_NUM
        k_token_stride k_head_stride head_dim_stride vcb_stride vch_stride vcs_stride
        vcd_stride bts_stride btb_stride block_size HEAD_DIM
      ⊨ decodingVCacheChainSpec := by
  refine ChainMetaGroupedMasked2DKernelIO.Implements.intro _ ?_ ?_ ?_ ?_
  · intro o
    simp only [decodingVCacheChainIO, List.mem_cons]
    tauto
  · exact decoding_fused_rotary_embedding_vcache_chain_flattenOk v v_cache BLOCK_TABLES context_lengths KV_GROUP_NUM
      k_token_stride k_head_stride head_dim_stride vcb_stride vch_stride vcs_stride
      vcd_stride bts_stride btb_stride block_size HEAD_DIM
  · intro bounds s s1 s2 hm₁ hm₂ hb1 hb2 hbr hbw
    simp only [decodingVCacheChainIO] at hm₁ hm₂ hb1 hb2 hbr hbw
    refine decoding_fused_rotary_embedding_vcache_chain_traceSafe v v_cache BLOCK_TABLES context_lengths KV_GROUP_NUM
      k_token_stride k_head_stride head_dim_stride vcb_stride vch_stride vcs_stride
      vcd_stride bts_stride btb_stride block_size HEAD_DIM bounds s hb1 ?_ ?_ ?_
    · rw [← hm₁] at hb2; exact hb2
    · exact fun j hj => hbr ⟨0, by omega⟩ j hj
    · intro j hj
      have h := hbw ⟨0, by omega⟩ j hj
      rw [← hm₂, ← hm₁] at h; exact h
  · intro s₀ s1 s2 xs hm₁ hm₂ hx
    simp only [decodingVCacheChainIO] at hm₁ hm₂ hx
    obtain ⟨s1', hexec, hval, hframe⟩ :=
      decoding_fused_rotary_embedding_vcache_chain_region_run v v_cache BLOCK_TABLES context_lengths KV_GROUP_NUM
        k_token_stride k_head_stride head_dim_stride vcb_stride vch_stride vcs_stride
        vcd_stride bts_stride btb_stride block_size HEAD_DIM s₀ s1 s2 xs hm₁ hm₂
        (fun j hj => hx ⟨0, by omega⟩ j hj)
    refine ⟨s1', hexec, ?_, ?_⟩
    · intro o hInj j hmask
      simp only [decodingVCacheChainIO, decodingVCacheChainSpec] at hInj hmask ⊢
      exact hval hInj j hmask
    · intro r o' hc
      simp only [decodingVCacheChainIO] at hc
      refine hframe r o' ?_
      by_cases hr : r = v_cache
      · right; intro j hjh
        rcases hc ⟨0, by omega⟩ j hjh with h | h
        · exact absurd hr h
        · exact h
      · left; exact hr



end Correct_without_Rounding


end VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding
