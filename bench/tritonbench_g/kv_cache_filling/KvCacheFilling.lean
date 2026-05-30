import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `kv_cache_filling` — strict per-kernel correctness

`_fill_kv_cache_kernel` is a prefill/decode paged KV-cache fill: program
`(batch_id, block_id)` computes sequence/block positions from `QStartLoc`,
`QSeqLens`, `KVSeqLens` and the `BlockOffsets` table, then loops over the
cache-block token slots, gathering `BLOCK_H × BLOCK_D` K and V tiles from
`KStates` / `VStates` and scattering them into `KCaches` / `VCaches`, masked by
`h_off < num_heads` and `d_off < head_dim`. The Python wrapper also has int8 /
int4 quantized paths that additionally write rounded values plus per-head
scale/zero metadata.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (grid over `(batch, blocks)`, the start-loc / seqlen /
block-offset inputs, and how the runtime composes per-program scatters into the
paged cache buffers) is the *trusted boundary*, not a proof obligation here.
Because the program ids are universally quantified, the per-program statement
covers every program of the grid.

## Proof architecture

```
fill_kv_cache_python_test_layout_output_summary        ← TOP THEOREM (quant_policy 0)
fill_quant_int8_kv_cache_python_test_layout_summary     ← TOP THEOREM (int8 path)
fill_quant_int4_kv_cache_python_test_layout_summary     ← TOP THEOREM (int4 path)
  ├─ fill_kv_cache_kernel_surface_toAlgorithm_supported              surface lowers
  ├─ fill_kv_cache_python_test_layout_all_outputs_compute_correct
  │    ├─ fill_k_cache_tile_python_test_layout_compute_correct → _compute_correct → _correct
  │    └─ fill_v_cache_tile_python_test_layout_compute_correct → _compute_correct → _correct
  ├─ fill_quant_kv_cache_python_test_layout_all_outputs_compute_correct  (quant value stores)
  └─ fill_quant_meta_store_slice_compute_correct → _correct                (scale/zero metadata)
       (k_scale / k_zero / v_scale / v_zero specializations)
```
Single-tile lemmas (`fill_k_cache_tile_*`, `fill_v_cache_tile_*`,
`fill_quant_meta_store_slice_*`) and Python test-shape / test-layout wrappers
(`*_test_h4_d16_*`, `*_python_test_layout_*`, offset-injectivity lemmas) feed
the three top summaries.

## Modeling boundary

Arithmetic/values are over `ℝ` (not bit-accurate IEEE float); dtype casts are
erased (post-erasure all dtypes unify to `ℝ`). This is a **partial / blocked**
verification: the cache stores are proved per masked K/V tile and per
scale/zero metadata slice at the **Python test shapes** (`num_heads = 4`,
`head_dim = head_dim_v = 16`, `BLOCK = 4`, contiguous strides) for selected
`SIDX` / `BIDX` / `KV_BLOCK_IDX` source/cache block positions — not the full
data-dependent loop bounds. The destination block offset is gathered from
`BlockOffsets`; offset-injectivity (no two cells of one program alias) is
discharged by explicit lemmas for those shapes. The int8 quant value stores are
modeled as a copy of the (rounded) source value at the algorithm layer (the
rounding is `ℝ`-erased); int4 packing is represented at its store surface.
`@triton.autotune` is not modeled.
-/

namespace VeriTile.Bench.TritonBenchG.KvCacheFilling

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Lean port of `kv_cache_filling.py`'s `_fill_kv_cache_kernel`.

This covers the Python wrapper's `quant_policy = 0` path. Correctness below is
still proved on smaller K/V tile lemmas, while this definition records the
complete non-quant kernel body used by the unquantized path. -/
def fill_kv_cache_kernel_surface
    (KStates VStates KCaches VCaches : RegionName)
    (QStartLoc QSeqLens KVSeqLens BlockOffsets : Region .nat)
    (num_heads head_dim head_dim_v stride_kss stride_ksh stride_ksd
      stride_vss stride_vsh stride_vsd stride_kcn stride_kcb stride_kch
      stride_kcd stride_vcn stride_vcb stride_vch stride_vcd stride_boff
      BLOCK BLOCK_D BLOCK_DV BLOCK_H : Nat) :
    ComputeKernel := triton {
  batch_id = tl.program_id(0)
  block_id = tl.program_id(1)
  h_off = tl.arange(0, $(BLOCK_H))
  d_off = tl.arange(0, $(BLOCK_D))
  q_startloc = tl.load(QStartLoc + batch_id)
  q_seqlen = tl.load(QSeqLens + batch_id)
  kv_seqlen = tl.load(KVSeqLens + batch_id)
  history_seqlen = kv_seqlen - q_seqlen
  block0_first_tokenloc = history_seqlen % $(BLOCK)
  state_token_offset = tl.maximum(block_id * $(BLOCK) - block0_first_tokenloc, $(0))
  kv_block_id = _div_up(history_seqlen + $(1), $(BLOCK)) - $(1) + block_id
  kv_block_id = min(kv_block_id, $((stride_boff - 1 : Nat)))
  block_off = tl.load(BlockOffsets +
    batch_id * $(stride_boff) + kv_block_id)
  cur_startloc = q_startloc + state_token_offset
  ks_ptr = KStates + cur_startloc * $(stride_kss)
  vs_ptr = VStates + cur_startloc * $(stride_vss)
  kc_ptr = KCaches + block_off * $(stride_kcn)
  vc_ptr = VCaches + block_off * $(stride_vcn)
  c_first_tokenloc = block0_first_tokenloc
  if block_id != $(0) {
    c_first_tokenloc *= $(0)
  }
  c_last_tokenloc = tl.minimum($(BLOCK),
    q_seqlen + block0_first_tokenloc - block_id * $(BLOCK))
  for bidx in range(c_first_tokenloc, c_last_tokenloc) {
    sidx = bidx - c_first_tokenloc
    mask = (h_off[:, None] < $(num_heads)) & (d_off[None, :] < $(head_dim))
    k = tl.load(ks_ptr + sidx * $(stride_kss) + h_off[:, None] * $(stride_ksh) +
        d_off[None, :] * $(stride_ksd), mask=mask)
    tl.store(kc_ptr + bidx * $(stride_kcb) + h_off[:, None] * $(stride_kch) +
        d_off[None, :] * $(stride_kcd), k, mask=mask)
    if $((BLOCK_DV > 0 : Bool)) {
      dv_off = tl.arange(0, $(BLOCK_DV))
      maskv = (h_off[:, None] < $(num_heads)) & (dv_off[None, :] < $(head_dim_v))
      v = tl.load(vs_ptr + sidx * $(stride_vss) +
          h_off[:, None] * $(stride_vsh) + dv_off[None, :] * $(stride_vsd),
        mask=maskv)
      tl.store(vc_ptr + bidx * $(stride_vcb) + h_off[:, None] * $(stride_vch) +
          dv_off[None, :] * $(stride_vcd), v, mask=maskv)
    }
  }
}

/-- The full non-quantized KV-cache filling surface lowers to the algorithm
layer, including sequence/block arithmetic and K/V cache stores. -/
theorem fill_kv_cache_kernel_surface_toAlgorithm_supported
    (KStates VStates KCaches VCaches : RegionName)
    (QStartLoc QSeqLens KVSeqLens BlockOffsets : Region .nat)
    (num_heads head_dim head_dim_v stride_kss stride_ksh stride_ksd
      stride_vss stride_vsh stride_vsd stride_kcn stride_kcb stride_kch
      stride_kcd stride_vcn stride_vcb stride_vch stride_vcd stride_boff
      BLOCK BLOCK_D BLOCK_DV BLOCK_H : Nat) :
    ∃ alg, (fill_kv_cache_kernel_surface KStates VStates KCaches VCaches
      QStartLoc QSeqLens KVSeqLens BlockOffsets num_heads head_dim head_dim_v
      stride_kss stride_ksh stride_ksd stride_vss stride_vsh stride_vsd
      stride_kcn stride_kcb stride_kch stride_kcd stride_vcn stride_vcb
      stride_vch stride_vcd stride_boff BLOCK BLOCK_D BLOCK_DV
      BLOCK_H).toAlgorithm? = Except.ok alg := by
  simp [fill_kv_cache_kernel_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Surface transcription and proof-oriented K-cache fill slice of `kv_cache_filling.py`'s
`_fill_kv_cache_kernel`.

The full kernel computes sequence/block positions and loops over cache block
slots. This slice starts after that arithmetic has selected `SIDX`, `BIDX`, and
`KV_BLOCK_IDX`: load a block offset from `BlockOffsets`, load a
`BLOCK_H × BLOCK_D` K tile from `KStates`, and store it into `KCaches` under the
original `num_heads/head_dim` mask.

This tile is the `quant_policy = 0` value path. The quantized value and
metadata writebacks, plus the `_quant_int8`/`_quant_int4` helper surfaces for
rounding, scale/zero metadata, and int4 packing, are represented separately
below so each Python-tested branch has an explicit DSL artifact. -/
def fill_k_cache_tile
    (KStates KCaches : RegionName) (BlockOffsets : Region .nat)
    (SIDX BIDX KV_BLOCK_IDX
      stride_kss stride_ksh stride_ksd
      stride_kcn stride_kcb stride_kch stride_kcd
      stride_boff num_heads head_dim BLOCK_H BLOCK_D : Nat) :
    ComputeKernel := triton {
  batch_id = tl.program_id(0)
  h_off = tl.arange(0, $(BLOCK_H))
  d_off = tl.arange(0, $(BLOCK_D))
  block_off = tl.load(BlockOffsets + batch_id * $(stride_boff) + $(KV_BLOCK_IDX))
  mask = (h_off[:, None] < $(num_heads)) & (d_off[None, :] < $(head_dim))
  k = tl.load(KStates + $(SIDX) * $(stride_kss) +
      h_off[:, None] * $(stride_ksh) + d_off[None, :] * $(stride_ksd),
    mask=mask, other=0.0)
  tl.store(KCaches + block_off * $(stride_kcn) + $(BIDX) * $(stride_kcb) +
      h_off[:, None] * $(stride_kch) + d_off[None, :] * $(stride_kcd),
    k, mask=mask)
}

/-- Surface transcription of the V-cache fill side of `kv_cache_filling.py`'s
`_fill_kv_cache_kernel`.

This is the `BLOCK_DV > 0` branch paired with `fill_k_cache_tile`: after the
sequence/block arithmetic has selected `SIDX`, `BIDX`, and `KV_BLOCK_IDX`, it
loads one `BLOCK_H × BLOCK_DV` tile from `VStates` and stores it into
`VCaches` under the original `num_heads/head_dim_v` mask. -/
def fill_v_cache_tile
    (VStates VCaches : RegionName) (BlockOffsets : Region .nat)
    (SIDX BIDX KV_BLOCK_IDX
      stride_vss stride_vsh stride_vsd
      stride_vcn stride_vcb stride_vch stride_vcd
      stride_boff num_heads head_dim_v BLOCK_H BLOCK_DV : Nat) :
    ComputeKernel := triton {
  batch_id = tl.program_id(0)
  h_off = tl.arange(0, $(BLOCK_H))
  dv_off = tl.arange(0, $(BLOCK_DV))
  block_off = tl.load(BlockOffsets + batch_id * $(stride_boff) + $(KV_BLOCK_IDX))
  maskv = (h_off[:, None] < $(num_heads)) & (dv_off[None, :] < $(head_dim_v))
  v = tl.load(VStates + $(SIDX) * $(stride_vss) +
      h_off[:, None] * $(stride_vsh) + dv_off[None, :] * $(stride_vsd),
    mask=maskv, other=0.0)
  tl.store(VCaches + block_off * $(stride_vcn) + $(BIDX) * $(stride_vcb) +
      h_off[:, None] * $(stride_vch) + dv_off[None, :] * $(stride_vcd),
    v, mask=maskv)
}

def headIndex (_s : BlockState) (i : Fin BLOCK_H) : Nat :=
  i.val

def dimIndex (_s : BlockState) (j : Fin BLOCK_D) : Nat :=
  j.val

def blockOff (s : BlockState) (BlockOffsets : RegionName)
    (KV_BLOCK_IDX stride_boff : Nat) : Nat :=
  s.readMemValue .nat BlockOffsets (s.pids 0 * stride_boff + KV_BLOCK_IDX)

def active
    (s : BlockState) (num_heads head_dim BLOCK_H BLOCK_D : Nat)
    (idx : TileIndex [BLOCK_H, BLOCK_D]) : Prop :=
  headIndex s idx.1 < num_heads ∧ dimIndex s idx.2.1 < head_dim

instance activeDecidable
    (s : BlockState) (num_heads head_dim BLOCK_H BLOCK_D : Nat)
    (idx : TileIndex [BLOCK_H, BLOCK_D]) :
    Decidable (active s num_heads head_dim BLOCK_H BLOCK_D idx) := by
  unfold active
  infer_instance

def kSourceOffset
    (s : BlockState) (SIDX stride_kss stride_ksh stride_ksd : Nat)
    (idx : TileIndex [BLOCK_H, BLOCK_D]) : Nat :=
  SIDX * stride_kss + headIndex s idx.1 * stride_ksh +
    dimIndex s idx.2.1 * stride_ksd

def kCacheOffset
    (s : BlockState) (BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX stride_kcn stride_kcb stride_kch stride_kcd stride_boff : Nat)
    (idx : TileIndex [BLOCK_H, BLOCK_D]) : Nat :=
  blockOff s BlockOffsets KV_BLOCK_IDX stride_boff * stride_kcn +
    BIDX * stride_kcb + headIndex s idx.1 * stride_kch +
    dimIndex s idx.2.1 * stride_kcd

private noncomputable def kRegisterValue
    (s : BlockState) (KStates : RegionName)
    (SIDX stride_kss stride_ksh stride_ksd num_heads head_dim : Nat)
    (idx : TileIndex [BLOCK_H, BLOCK_D]) : ℝ :=
  WithBot.unbotD 0
    (if idx.1.val < num_heads ∧ idx.2.1.val < head_dim then
      some (s.readMem KStates
        (SIDX * stride_kss + idx.1.val * stride_ksh +
          idx.2.1.val * stride_ksd))
    else some (0.0 : ℝ))

private noncomputable def preStoreState
    (s : BlockState) (KStates BlockOffsets : RegionName)
    (SIDX KV_BLOCK_IDX stride_kss stride_ksh stride_ksd stride_boff
      num_heads head_dim BLOCK_H BLOCK_D : Nat) : BlockState :=
  s.setReg "batch_id" TileDType.nat [] (Tile.scalar (s.pids 0))
    |>.setReg "h_off" TileDType.nat [BLOCK_H] (Tile.vec fun i => i.val)
    |>.setReg "d_off" TileDType.nat [BLOCK_D] (Tile.vec fun j => j.val)
    |>.setReg "block_off" TileDType.nat []
      (Tile.scalar (s.readMemValue .nat BlockOffsets
        (s.pids 0 * stride_boff + KV_BLOCK_IDX)))
    |>.setReg "mask" TileDType.bool [BLOCK_H, BLOCK_D]
      { data := fun idx =>
        decide (idx.1.val < num_heads ∧ idx.2.1.val < head_dim) }
    |>.setReg "k" TileDType.real [BLOCK_H, BLOCK_D]
      { data := fun idx =>
        if idx.1.val < num_heads ∧ idx.2.1.val < head_dim then
          some (s.readMem KStates
            (SIDX * stride_kss + idx.1.val * stride_ksh +
              idx.2.1.val * stride_ksd))
        else some (0.0 : ℝ) }

/-- Algorithm-layer correctness for the K-cache tile fill. -/
theorem fill_k_cache_tile_correct
    (KStates KCaches BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX
      stride_kss stride_ksh stride_ksd
      stride_kcn stride_kcb stride_kch stride_kcd
      stride_boff num_heads head_dim BLOCK_H BLOCK_D : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_H, BLOCK_D] =>
        kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_kcn stride_kcb
          stride_kch stride_kcd stride_boff idx))
    (hExec : exec (fill_k_cache_tile KStates KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX stride_kss stride_ksh stride_ksd
        stride_kcn stride_kcb stride_kch stride_kcd stride_boff
        num_heads head_dim BLOCK_H BLOCK_D) s = some s') :
    ∀ idx : TileIndex [BLOCK_H, BLOCK_D],
      s'.readMem KCaches
          (kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_kcn stride_kcb
            stride_kch stride_kcd stride_boff idx) =
        if active s num_heads head_dim BLOCK_H BLOCK_D idx then
          s.readMem KStates
            (kSourceOffset s SIDX stride_kss stride_ksh stride_ksd idx)
        else
          s.readMem KCaches
            (kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_kcn stride_kcb
              stride_kch stride_kcd stride_boff idx) := by
  intro idx
  simp [exec, fill_k_cache_tile, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        BlockState.readMemValue, TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  let offsetFn : TileIndex [BLOCK_H, BLOCK_D] → Nat :=
    fun idx =>
      s.readMemValue .nat BlockOffsets (s.pids 0 * stride_boff + KV_BLOCK_IDX) *
          stride_kcn +
        BIDX * stride_kcb + idx.1.val * stride_kch + idx.2.1.val * stride_kcd
  let valueFn : TileIndex [BLOCK_H, BLOCK_D] → ℝ :=
    fun idx =>
      kRegisterValue s KStates SIDX stride_kss stride_ksh stride_ksd
        num_heads head_dim idx
  let P : TileIndex [BLOCK_H, BLOCK_D] → Prop :=
    fun idx => idx.1.val < num_heads ∧ idx.2.1.val < head_dim
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, kCacheOffset, blockOff, headIndex, dimIndex,
      BlockState.readMemValue] using hOutInj
  have hscatter := BlockState.scatter_readback_prop_masked_nd
    (region := KCaches)
    (s := preStoreState s KStates BlockOffsets SIDX KV_BLOCK_IDX
      stride_kss stride_ksh stride_ksd stride_boff num_heads head_dim
      BLOCK_H BLOCK_D)
    (offsetFn := offsetFn) (valueFn := valueFn) (P := P)
    hOffsetInj idx
  by_cases hActive : P idx
  · simpa [offsetFn, valueFn, P, active, kSourceOffset, kCacheOffset, blockOff,
      headIndex, dimIndex, kRegisterValue, preStoreState, BlockState.readMemValue,
      TileShape.dropInsertedIndex, hActive] using hscatter
  · simpa [offsetFn, valueFn, P, active, kSourceOffset, kCacheOffset, blockOff,
      headIndex, dimIndex, kRegisterValue, preStoreState, BlockState.readMemValue,
      TileShape.dropInsertedIndex, hActive] using hscatter

/-- Compute-facing correctness for the K-cache tile fill. -/
theorem fill_k_cache_tile_compute_correct
    (KStates KCaches BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX
      stride_kss stride_ksh stride_ksd
      stride_kcn stride_kcb stride_kch stride_kcd
      stride_boff num_heads head_dim BLOCK_H BLOCK_D : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_H, BLOCK_D] =>
        kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_kcn stride_kcb
          stride_kch stride_kcd stride_boff idx)) :
    ComputeCorrect.Realizes
      (kernel := fill_k_cache_tile KStates KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX stride_kss stride_ksh stride_ksd
        stride_kcn stride_kcb stride_kch stride_kcd stride_boff
        num_heads head_dim BLOCK_H BLOCK_D)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_heads head_dim BLOCK_H BLOCK_D)
        (fun idx => (KCaches,
          kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_kcn stride_kcb
            stride_kch stride_kcd stride_boff idx)))
      (expected := fun idx =>
        s.readMem KStates
          (kSourceOffset s SIDX stride_kss stride_ksh stride_ksd idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fill_k_cache_tile]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := fill_k_cache_tile_correct KStates KCaches BlockOffsets
    SIDX BIDX KV_BLOCK_IDX stride_kss stride_ksh stride_ksd stride_kcn
    stride_kcb stride_kch stride_kcd stride_boff num_heads head_dim
    BLOCK_H BLOCK_D s s' hOutInj hExec idx
  simpa [hActive] using h

/-! ## V-cache tile fill correctness -/

def vSourceOffset
    (s : BlockState) (SIDX stride_vss stride_vsh stride_vsd : Nat)
    (idx : TileIndex [BLOCK_H, BLOCK_DV]) : Nat :=
  SIDX * stride_vss + headIndex s idx.1 * stride_vsh +
    dimIndex s idx.2.1 * stride_vsd

def vCacheOffset
    (s : BlockState) (BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX stride_vcn stride_vcb stride_vch stride_vcd stride_boff : Nat)
    (idx : TileIndex [BLOCK_H, BLOCK_DV]) : Nat :=
  blockOff s BlockOffsets KV_BLOCK_IDX stride_boff * stride_vcn +
    BIDX * stride_vcb + headIndex s idx.1 * stride_vch +
    dimIndex s idx.2.1 * stride_vcd

private noncomputable def vRegisterValue
    (s : BlockState) (VStates : RegionName)
    (SIDX stride_vss stride_vsh stride_vsd num_heads head_dim_v : Nat)
    (idx : TileIndex [BLOCK_H, BLOCK_DV]) : ℝ :=
  WithBot.unbotD 0
    (if idx.1.val < num_heads ∧ idx.2.1.val < head_dim_v then
      some (s.readMem VStates
        (SIDX * stride_vss + idx.1.val * stride_vsh +
          idx.2.1.val * stride_vsd))
    else some (0.0 : ℝ))

private noncomputable def preStoreStateV
    (s : BlockState) (VStates BlockOffsets : RegionName)
    (SIDX KV_BLOCK_IDX
      stride_vss stride_vsh stride_vsd stride_boff
      num_heads head_dim_v BLOCK_H BLOCK_DV : Nat) : BlockState :=
  (s.setReg "batch_id" TileDType.nat [] (Tile.scalar (s.pids 0))
    |>.setReg "h_off" TileDType.nat [BLOCK_H] (Tile.vec fun i => i.val)
    |>.setReg "dv_off" TileDType.nat [BLOCK_DV] (Tile.vec fun j => j.val)
    |>.setReg "block_off" TileDType.nat []
        (Tile.scalar
          (s.readMemValue .nat BlockOffsets (s.pids 0 * stride_boff + KV_BLOCK_IDX)))
    |>.setReg "maskv" TileDType.bool [BLOCK_H, BLOCK_DV]
        { data := fun idx =>
            decide (idx.1.val < num_heads) && decide (idx.2.1.val < head_dim_v) }
    |>.setReg "v" TileDType.real [BLOCK_H, BLOCK_DV]
        { data := fun idx =>
          if idx.1.val < num_heads ∧ idx.2.1.val < head_dim_v then
            some (s.readMem VStates
              (SIDX * stride_vss + idx.1.val * stride_vsh +
                idx.2.1.val * stride_vsd))
          else some (0.0 : ℝ) })

/-- Algorithm-layer correctness for the V-cache tile fill (analogous to
`fill_k_cache_tile_correct`, with V-specific strides and bound). -/
theorem fill_v_cache_tile_correct
    (VStates VCaches BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX
      stride_vss stride_vsh stride_vsd
      stride_vcn stride_vcb stride_vch stride_vcd
      stride_boff num_heads head_dim_v BLOCK_H BLOCK_DV : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_H, BLOCK_DV] =>
        vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_vcn stride_vcb
          stride_vch stride_vcd stride_boff idx))
    (hExec : exec (fill_v_cache_tile VStates VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX stride_vss stride_vsh stride_vsd
        stride_vcn stride_vcb stride_vch stride_vcd stride_boff
        num_heads head_dim_v BLOCK_H BLOCK_DV) s = some s') :
    ∀ idx : TileIndex [BLOCK_H, BLOCK_DV],
      s'.readMem VCaches
          (vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_vcn stride_vcb
            stride_vch stride_vcd stride_boff idx) =
        if active s num_heads head_dim_v BLOCK_H BLOCK_DV idx then
          s.readMem VStates
            (vSourceOffset s SIDX stride_vss stride_vsh stride_vsd idx)
        else
          s.readMem VCaches
            (vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_vcn stride_vcb
              stride_vch stride_vcd stride_boff idx) := by
  intro idx
  simp [exec, fill_v_cache_tile, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        BlockState.readMemValue, TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  let offsetFn : TileIndex [BLOCK_H, BLOCK_DV] → Nat :=
    fun idx =>
      s.readMemValue .nat BlockOffsets (s.pids 0 * stride_boff + KV_BLOCK_IDX) *
          stride_vcn +
        BIDX * stride_vcb + idx.1.val * stride_vch + idx.2.1.val * stride_vcd
  let valueFn : TileIndex [BLOCK_H, BLOCK_DV] → ℝ :=
    fun idx =>
      vRegisterValue s VStates SIDX stride_vss stride_vsh stride_vsd
        num_heads head_dim_v idx
  let P : TileIndex [BLOCK_H, BLOCK_DV] → Prop :=
    fun idx => idx.1.val < num_heads ∧ idx.2.1.val < head_dim_v
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, vCacheOffset, blockOff, headIndex, dimIndex,
      BlockState.readMemValue] using hOutInj
  have hscatter := BlockState.scatter_readback_prop_masked_nd
    (region := VCaches)
    (s := preStoreStateV s VStates BlockOffsets SIDX KV_BLOCK_IDX
      stride_vss stride_vsh stride_vsd stride_boff num_heads head_dim_v
      BLOCK_H BLOCK_DV)
    (offsetFn := offsetFn) (valueFn := valueFn) (P := P)
    hOffsetInj idx
  by_cases hActive : P idx
  · simpa [offsetFn, valueFn, P, active, vSourceOffset, vCacheOffset, blockOff,
      headIndex, dimIndex, vRegisterValue, preStoreStateV,
      BlockState.readMemValue, TileShape.dropInsertedIndex, hActive] using hscatter
  · simpa [offsetFn, valueFn, P, active, vSourceOffset, vCacheOffset, blockOff,
      headIndex, dimIndex, vRegisterValue, preStoreStateV,
      BlockState.readMemValue, TileShape.dropInsertedIndex, hActive] using hscatter

/-- Compute-facing correctness for the V-cache tile fill. -/
theorem fill_v_cache_tile_compute_correct
    (VStates VCaches BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX
      stride_vss stride_vsh stride_vsd
      stride_vcn stride_vcb stride_vch stride_vcd
      stride_boff num_heads head_dim_v BLOCK_H BLOCK_DV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_H, BLOCK_DV] =>
        vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_vcn stride_vcb
          stride_vch stride_vcd stride_boff idx)) :
    ComputeCorrect.Realizes
      (kernel := fill_v_cache_tile VStates VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX stride_vss stride_vsh stride_vsd
        stride_vcn stride_vcb stride_vch stride_vcd stride_boff
        num_heads head_dim_v BLOCK_H BLOCK_DV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_heads head_dim_v BLOCK_H BLOCK_DV)
        (fun idx => (VCaches,
          vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_vcn stride_vcb
            stride_vch stride_vcd stride_boff idx)))
      (expected := fun idx =>
        s.readMem VStates
          (vSourceOffset s SIDX stride_vss stride_vsh stride_vsd idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fill_v_cache_tile]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := fill_v_cache_tile_correct VStates VCaches BlockOffsets
    SIDX BIDX KV_BLOCK_IDX stride_vss stride_vsh stride_vsd stride_vcn
    stride_vcb stride_vch stride_vcd stride_boff num_heads head_dim_v
    BLOCK_H BLOCK_DV s s' hOutInj hExec idx
  simpa [hActive] using h

/-! ## Quantized cache value stores -/

/-- Named K-cache value writeback for `_fill_kv_cache_quant_kernel`.

The quantized kernel computes `q_k` via `_quant_int8` or `_quant_int4` before
the store. This theorem exposes the cache writeback from a precomputed
`QKPre` tile while leaving the integer quantization/packing arithmetic as the
remaining proof obligation. -/
theorem fill_quant_k_cache_value_store_slice_compute_correct
    (QKPre KCaches BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX
      stride_qss stride_qsh stride_qsd
      stride_kcn stride_kcb stride_kch stride_kcd
      stride_boff num_heads head_dim BLOCK_H BLOCK_D : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_H, BLOCK_D] =>
        kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_kcn stride_kcb
          stride_kch stride_kcd stride_boff idx)) :
    ComputeCorrect.Realizes
      (kernel := fill_k_cache_tile QKPre KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX stride_qss stride_qsh stride_qsd
        stride_kcn stride_kcb stride_kch stride_kcd stride_boff
        num_heads head_dim BLOCK_H BLOCK_D)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_heads head_dim BLOCK_H BLOCK_D)
        (fun idx => (KCaches,
          kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_kcn stride_kcb
            stride_kch stride_kcd stride_boff idx)))
      (expected := fun idx =>
        s.readMem QKPre
          (kSourceOffset s SIDX stride_qss stride_qsh stride_qsd idx)) := by
  exact fill_k_cache_tile_compute_correct QKPre KCaches BlockOffsets
    SIDX BIDX KV_BLOCK_IDX stride_qss stride_qsh stride_qsd stride_kcn
    stride_kcb stride_kch stride_kcd stride_boff num_heads head_dim BLOCK_H
    BLOCK_D s hOutInj

/-- Named V-cache value writeback for `_fill_kv_cache_quant_kernel`.

For `quant_policy = 4`, callers instantiate `BLOCK_DV`/`head_dim_v` with the
packed width; for `quant_policy = 8`, they use the full value head dimension.
The theorem covers the masked cache writeback from a precomputed `QVPre` tile. -/
theorem fill_quant_v_cache_value_store_slice_compute_correct
    (QVPre VCaches BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX
      stride_qss stride_qsh stride_qsd
      stride_vcn stride_vcb stride_vch stride_vcd
      stride_boff num_heads head_dim_v BLOCK_H BLOCK_DV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_H, BLOCK_DV] =>
        vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_vcn stride_vcb
          stride_vch stride_vcd stride_boff idx)) :
    ComputeCorrect.Realizes
      (kernel := fill_v_cache_tile QVPre VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX stride_qss stride_qsh stride_qsd
        stride_vcn stride_vcb stride_vch stride_vcd stride_boff
        num_heads head_dim_v BLOCK_H BLOCK_DV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_heads head_dim_v BLOCK_H BLOCK_DV)
        (fun idx => (VCaches,
          vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_vcn stride_vcb
            stride_vch stride_vcd stride_boff idx)))
      (expected := fun idx =>
        s.readMem QVPre
          (vSourceOffset s SIDX stride_qss stride_qsh stride_qsd idx)) := by
  exact fill_v_cache_tile_compute_correct QVPre VCaches BlockOffsets
    SIDX BIDX KV_BLOCK_IDX stride_qss stride_qsh stride_qsd stride_vcn
    stride_vcb stride_vch stride_vcd stride_boff num_heads head_dim_v BLOCK_H
    BLOCK_DV s hOutInj

/-! ## Quantized helper compute surfaces -/

/-- Proof-oriented int8 quantization compute slice for `_quant_int8`.

The Python helper computes per-head `val_min`, `val_max`, `scales`, `zeros`,
then forms `val / scales[:, None] + zeros[:, None] + 0.5` before the
`tl.uint8` cast. This slice keeps that operator chain explicit and writes the
uint8-cast tile plus scale/zero vectors to proof regions. -/
def quant_int8_compute_store_slice
    (Val QVal ScaleOut ZeroOut : RegionName)
    (stride_vh stride_vd stride_qh stride_qd
      num_heads head_dim BLOCK_H BLOCK_D : Nat) :
    ComputeKernel := triton {
  h_off = tl.arange(0, $(BLOCK_H))
  d_off = tl.arange(0, $(BLOCK_D))
  mask = (h_off[:, None] < $(num_heads)) & (d_off[None, :] < $(head_dim))
  val = tl.load(Val + h_off[:, None] * $(stride_vh) +
      d_off[None, :] * $(stride_vd), mask=mask, other=0.0).to(tl.float32)
  val_min = -tl.max(-val, 1)
  val_max = tl.max(val, 1)
  scales = (val_max - val_min) / 255.0
  zeros = -val_min / scales
  q_val = (val / scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
  tl.store(QVal + h_off[:, None] * $(stride_qh) + d_off[None, :] * $(stride_qd),
    q_val, mask=mask)
  tl.store(ScaleOut + h_off, scales, mask=h_off < $(num_heads))
  tl.store(ZeroOut + h_off, zeros, mask=h_off < $(num_heads))
}

/-- The `_quant_int8` compute/store slice lowers with min/max reductions,
scale/zero arithmetic, rounding offset, and `tl.uint8` cast preserved. -/
theorem quant_int8_compute_store_slice_toAlgorithm_supported
    (Val QVal ScaleOut ZeroOut : RegionName)
    (stride_vh stride_vd stride_qh stride_qd
      num_heads head_dim BLOCK_H BLOCK_D : Nat) :
    ∃ alg, (quant_int8_compute_store_slice Val QVal ScaleOut ZeroOut
      stride_vh stride_vd stride_qh stride_qd num_heads head_dim BLOCK_H
      BLOCK_D).toAlgorithm? = Except.ok alg := by
  simp [quant_int8_compute_store_slice, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented int4 quantization compute slice for `_quant_int4`.

The Python helper casts both halves to float32, computes min/max across the
paired halves, forms scale/zero vectors, rounds both halves with `+ 0.5`, and
packs them as `q_val1 + q_val2 * 16` after uint8 casts. This slice preserves
that full helper chain. -/
def quant_int4_compute_store_slice
    (Val1 Val2 QVal ScaleOut ZeroOut : RegionName)
    (stride_v1h stride_v1d stride_v2h stride_v2d stride_qh stride_qd
      num_heads packed_head_dim BLOCK_H BLOCK_D : Nat) :
    ComputeKernel := triton {
  h_off = tl.arange(0, $(BLOCK_H))
  d_off = tl.arange(0, $(BLOCK_D))
  mask = (h_off[:, None] < $(num_heads)) &
    (d_off[None, :] < $(packed_head_dim))
  val1 = tl.load(Val1 + h_off[:, None] * $(stride_v1h) +
      d_off[None, :] * $(stride_v1d), mask=mask, other=0.0).to(tl.float32)
  val2 = tl.load(Val2 + h_off[:, None] * $(stride_v2h) +
      d_off[None, :] * $(stride_v2d), mask=mask, other=0.0).to(tl.float32)
  val_min = -tl.max(-tl.minimum(val1, val2), 1)
  val_max = tl.max(tl.maximum(val1, val2), 1)
  scales = (val_max - val_min) / 15.0
  zeros = -val_min / scales
  q_val1 = (val1 / scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
  q_val2 = (val2 / scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
  q_val = q_val1 + q_val2 * 16
  tl.store(QVal + h_off[:, None] * $(stride_qh) + d_off[None, :] * $(stride_qd),
    q_val, mask=mask)
  tl.store(ScaleOut + h_off, scales, mask=h_off < $(num_heads))
  tl.store(ZeroOut + h_off, zeros, mask=h_off < $(num_heads))
}

/-- The `_quant_int4` compute/store slice lowers with paired-half min/max
reductions, scale/zero arithmetic, rounding offsets, uint8 casts, and pack
formula preserved. -/
theorem quant_int4_compute_store_slice_toAlgorithm_supported
    (Val1 Val2 QVal ScaleOut ZeroOut : RegionName)
    (stride_v1h stride_v1d stride_v2h stride_v2d stride_qh stride_qd
      num_heads packed_head_dim BLOCK_H BLOCK_D : Nat) :
    ∃ alg, (quant_int4_compute_store_slice Val1 Val2 QVal ScaleOut
      ZeroOut stride_v1h stride_v1d stride_v2h stride_v2d stride_qh stride_qd
      num_heads packed_head_dim BLOCK_H BLOCK_D).toAlgorithm? = Except.ok alg := by
  simp [quant_int4_compute_store_slice, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Python test-layout quant helper coverage for `quant_policy = 8`.

The checked KV-cache tests use four heads and sixteen value lanes for each
K/V tile. This theorem records that the `_quant_int8` helper surface lowers
with min/max reductions, scale/zero arithmetic, `+ 0.5` rounding, and uint8
casts at that layout. -/
theorem quant_int8_python_test_layout_surface_toAlgorithm_supported
    (Val QVal ScaleOut ZeroOut : RegionName) :
    ∃ alg, (quant_int8_compute_store_slice Val QVal ScaleOut ZeroOut
      16 1 16 1 4 16 4 16).toAlgorithm? = Except.ok alg := by
  exact quant_int8_compute_store_slice_toAlgorithm_supported Val QVal
    ScaleOut ZeroOut 16 1 16 1 4 16 4 16

/-- Python test-layout quant helper coverage for `quant_policy = 4`.

The int4 helper uses paired eight-lane halves for the sixteen-lane K/V tile and
packs them as `q_val1 + q_val2 * 16`, matching the Python helper surface. -/
theorem quant_int4_python_test_layout_surface_toAlgorithm_supported
    (Val1 Val2 QVal ScaleOut ZeroOut : RegionName) :
    ∃ alg, (quant_int4_compute_store_slice Val1 Val2 QVal ScaleOut ZeroOut
      16 1 16 1 8 1 4 8 4 8).toAlgorithm? = Except.ok alg := by
  exact quant_int4_compute_store_slice_toAlgorithm_supported Val1 Val2 QVal
    ScaleOut ZeroOut 16 1 16 1 8 1 4 8 4 8

/-! ## Quantized scale/zero metadata stores -/

/-- Proof-oriented metadata store slice for `_fill_kv_cache_quant_kernel`.

The Python quantized path writes per-head scale values at `szd = 0` and zero
points at `szd = 1` for both K and V metadata regions. This generic slice fixes
one slot (`SZD = 0` or `SZD = 1`) and proves the masked writeback from a
precomputed per-head metadata vector. -/
def fill_quant_meta_store_slice
    (MetaPre MetaOut : RegionName) (BlockOffsets : Region .nat)
    (BIDX KV_BLOCK_IDX SZD
      stride_mn stride_mb stride_mh stride_md stride_boff
      num_heads BLOCK_H : Nat) :
    ComputeKernel := triton {
  batch_id = tl.program_id(0)
  h_off = tl.arange(0, $(BLOCK_H))
  block_off = tl.load(BlockOffsets + batch_id * $(stride_boff) + $(KV_BLOCK_IDX))
  mask = h_off < $(num_heads)
  meta_val = tl.load(MetaPre + h_off, mask=mask, other=0.0)
  tl.store(MetaOut + block_off * $(stride_mn) + $(BIDX) * $(stride_mb) +
      h_off * $(stride_mh) + $(SZD) * $(stride_md), meta_val, mask=mask)
}

def metaActive (_s : BlockState) (num_heads BLOCK_H : Nat) (i : Fin BLOCK_H) : Prop :=
  i.val < num_heads

instance metaActiveDecidable (s : BlockState) (num_heads BLOCK_H : Nat)
    (i : Fin BLOCK_H) :
    Decidable (metaActive s num_heads BLOCK_H i) := by
  unfold metaActive
  infer_instance

def metaOffset
    (s : BlockState) (BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX SZD stride_mn stride_mb stride_mh stride_md stride_boff : Nat)
    (i : Fin BLOCK_H) : Nat :=
  blockOff s BlockOffsets KV_BLOCK_IDX stride_boff * stride_mn +
    BIDX * stride_mb + i.val * stride_mh + SZD * stride_md

noncomputable def metaStoreSpec
    (s : BlockState) (MetaPre : RegionName) (i : Fin BLOCK_H) : ℝ :=
  s.readMem MetaPre i.val

private noncomputable def preStoreStateMeta
    (s : BlockState) (MetaPre BlockOffsets : RegionName)
    (KV_BLOCK_IDX stride_boff num_heads BLOCK_H : Nat) : BlockState :=
  (s.setReg "batch_id" TileDType.nat [] (Tile.scalar (s.pids 0))
    |>.setReg "h_off" TileDType.nat [BLOCK_H] (Tile.vec fun i => i.val)
    |>.setReg "block_off" TileDType.nat []
        (Tile.scalar
          (s.readMemValue .nat BlockOffsets (s.pids 0 * stride_boff + KV_BLOCK_IDX)))
    |>.setReg "mask" TileDType.bool [BLOCK_H]
        { data := fun i => decide (i.1.val < num_heads) }
    |>.setReg "meta_val" TileDType.real [BLOCK_H]
        { data := fun i =>
          if i.1.val < num_heads then some (s.readMem MetaPre i.1.val)
          else some (0.0 : ℝ) })

theorem fill_quant_meta_store_slice_correct
    (MetaPre MetaOut BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX SZD
      stride_mn stride_mb stride_mh stride_md stride_boff
      num_heads BLOCK_H : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_H =>
        metaOffset s BlockOffsets BIDX KV_BLOCK_IDX SZD stride_mn stride_mb
          stride_mh stride_md stride_boff i))
    (hExec : exec (fill_quant_meta_store_slice MetaPre MetaOut BlockOffsets
        BIDX KV_BLOCK_IDX SZD stride_mn stride_mb stride_mh stride_md
        stride_boff num_heads BLOCK_H) s = some s') :
    ∀ i : Fin BLOCK_H,
      s'.readMem MetaOut
          (metaOffset s BlockOffsets BIDX KV_BLOCK_IDX SZD stride_mn stride_mb
            stride_mh stride_md stride_boff i) =
        if metaActive s num_heads BLOCK_H i then
          metaStoreSpec s MetaPre i
        else
          s.readMem MetaOut
            (metaOffset s BlockOffsets BIDX KV_BLOCK_IDX SZD stride_mn stride_mb
              stride_mh stride_md stride_boff i) := by
  intro i
  simp [exec, fill_quant_meta_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        BlockState.readMemValue, TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  let offsetFn : TileIndex [BLOCK_H] → Nat :=
    fun idx =>
      s.readMemValue .nat BlockOffsets (s.pids 0 * stride_boff + KV_BLOCK_IDX) *
          stride_mn +
        BIDX * stride_mb + idx.1.val * stride_mh + SZD * stride_md
  let valueFn : TileIndex [BLOCK_H] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if idx.1.val < num_heads then some (s.readMem MetaPre idx.1.val)
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_H] → Prop := fun idx => idx.1.val < num_heads
  have hOffsetInj : Function.Injective offsetFn := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [offsetFn, metaOffset, blockOff, BlockState.readMemValue] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  have hscatter := BlockState.scatter_readback_prop_masked_nd
    (region := MetaOut)
    (s := preStoreStateMeta s MetaPre BlockOffsets KV_BLOCK_IDX stride_boff
      num_heads BLOCK_H)
    (offsetFn := offsetFn) (valueFn := valueFn) (P := P)
    hOffsetInj (i, PUnit.unit)
  by_cases hActive : P (i, PUnit.unit)
  · simpa [offsetFn, valueFn, P, metaActive, metaOffset, blockOff, metaStoreSpec,
      preStoreStateMeta, BlockState.readMemValue, hActive] using hscatter
  · simpa [offsetFn, valueFn, P, metaActive, metaOffset, blockOff, metaStoreSpec,
      preStoreStateMeta, BlockState.readMemValue, hActive] using hscatter

theorem fill_quant_meta_store_slice_compute_correct
    (MetaPre MetaOut BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX SZD
      stride_mn stride_mb stride_mh stride_md stride_boff
      num_heads BLOCK_H : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_H =>
        metaOffset s BlockOffsets BIDX KV_BLOCK_IDX SZD stride_mn stride_mb
          stride_mh stride_md stride_boff i)) :
    ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice MetaPre MetaOut BlockOffsets
        BIDX KV_BLOCK_IDX SZD stride_mn stride_mb stride_mh stride_md
        stride_boff num_heads BLOCK_H)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_H => metaActive s num_heads BLOCK_H i)
        (fun i : Fin BLOCK_H => (MetaOut,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX SZD stride_mn stride_mb
            stride_mh stride_md stride_boff i)))
      (expected := fun i : Fin BLOCK_H => metaStoreSpec s MetaPre i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fill_quant_meta_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fill_quant_meta_store_slice_correct MetaPre MetaOut BlockOffsets
    BIDX KV_BLOCK_IDX SZD stride_mn stride_mb stride_mh stride_md stride_boff
    num_heads BLOCK_H s s' hOutInj hExec i
  simpa [hActive] using h

/-- Named K-scale writeback for `_fill_kv_cache_quant_kernel`.

This is the Python store with `ksz_ptr + ... + szd_off * stride_kszd` and
`szd_off = 0`, using a precomputed per-head scale vector. -/
theorem fill_quant_k_scale_store_slice_compute_correct
    (KScalePre KScalesZeros BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX
      stride_kszn stride_kszb stride_kszh stride_kszd stride_boff
      num_heads BLOCK_H : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_H =>
        metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 stride_kszn stride_kszb
          stride_kszh stride_kszd stride_boff i)) :
    ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice KScalePre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 stride_kszn stride_kszb stride_kszh stride_kszd
        stride_boff num_heads BLOCK_H)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_H => metaActive s num_heads BLOCK_H i)
        (fun i : Fin BLOCK_H => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 stride_kszn
            stride_kszb stride_kszh stride_kszd stride_boff i)))
      (expected := fun i : Fin BLOCK_H => metaStoreSpec s KScalePre i) := by
  exact fill_quant_meta_store_slice_compute_correct KScalePre KScalesZeros
    BlockOffsets BIDX KV_BLOCK_IDX 0 stride_kszn stride_kszb stride_kszh
    stride_kszd stride_boff num_heads BLOCK_H s hOutInj

/-- Named K-zero-point writeback for `_fill_kv_cache_quant_kernel`, fixing
`szd_off = 1`. -/
theorem fill_quant_k_zero_store_slice_compute_correct
    (KZeroPre KScalesZeros BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX
      stride_kszn stride_kszb stride_kszh stride_kszd stride_boff
      num_heads BLOCK_H : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_H =>
        metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 stride_kszn stride_kszb
          stride_kszh stride_kszd stride_boff i)) :
    ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice KZeroPre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 stride_kszn stride_kszb stride_kszh stride_kszd
        stride_boff num_heads BLOCK_H)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_H => metaActive s num_heads BLOCK_H i)
        (fun i : Fin BLOCK_H => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 stride_kszn
            stride_kszb stride_kszh stride_kszd stride_boff i)))
      (expected := fun i : Fin BLOCK_H => metaStoreSpec s KZeroPre i) := by
  exact fill_quant_meta_store_slice_compute_correct KZeroPre KScalesZeros
    BlockOffsets BIDX KV_BLOCK_IDX 1 stride_kszn stride_kszb stride_kszh
    stride_kszd stride_boff num_heads BLOCK_H s hOutInj

/-- Named V-scale writeback for `_fill_kv_cache_quant_kernel`, fixing
`szd_off = 0`. -/
theorem fill_quant_v_scale_store_slice_compute_correct
    (VScalePre VScalesZeros BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX
      stride_vszn stride_vszb stride_vszh stride_vszd stride_boff
      num_heads BLOCK_H : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_H =>
        metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 stride_vszn stride_vszb
          stride_vszh stride_vszd stride_boff i)) :
    ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice VScalePre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 stride_vszn stride_vszb stride_vszh stride_vszd
        stride_boff num_heads BLOCK_H)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_H => metaActive s num_heads BLOCK_H i)
        (fun i : Fin BLOCK_H => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 stride_vszn
            stride_vszb stride_vszh stride_vszd stride_boff i)))
      (expected := fun i : Fin BLOCK_H => metaStoreSpec s VScalePre i) := by
  exact fill_quant_meta_store_slice_compute_correct VScalePre VScalesZeros
    BlockOffsets BIDX KV_BLOCK_IDX 0 stride_vszn stride_vszb stride_vszh
    stride_vszd stride_boff num_heads BLOCK_H s hOutInj

/-- Named V-zero-point writeback for `_fill_kv_cache_quant_kernel`, fixing
`szd_off = 1`. -/
theorem fill_quant_v_zero_store_slice_compute_correct
    (VZeroPre VScalesZeros BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX
      stride_vszn stride_vszb stride_vszh stride_vszd stride_boff
      num_heads BLOCK_H : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_H =>
        metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 stride_vszn stride_vszb
          stride_vszh stride_vszd stride_boff i)) :
    ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice VZeroPre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 stride_vszn stride_vszb stride_vszh stride_vszd
        stride_boff num_heads BLOCK_H)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_H => metaActive s num_heads BLOCK_H i)
        (fun i : Fin BLOCK_H => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 stride_vszn
            stride_vszb stride_vszh stride_vszd stride_boff i)))
      (expected := fun i : Fin BLOCK_H => metaStoreSpec s VZeroPre i) := by
  exact fill_quant_meta_store_slice_compute_correct VZeroPre VScalesZeros
    BlockOffsets BIDX KV_BLOCK_IDX 1 stride_vszn stride_vszb stride_vszh
    stride_vszd stride_boff num_heads BLOCK_H s hOutInj

/-! ## Python test-shape wrappers

The checked Python test uses `num_heads = 4`, `head_dim = 16`,
`head_dim_v = 16`, and `block_size = 8`, so the next-power-of-two block shapes
are exactly `BLOCK_H = 4`, `BLOCK_D = 16`, and `BLOCK_DV = 16`.  The following
wrappers expose the generic K/V and metadata store proofs at those observed
dimensions. -/

theorem fill_k_cache_tile_test_h4_d16_compute_correct
    (KStates KCaches BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX
      stride_kss stride_ksh stride_ksd
      stride_kcn stride_kcb stride_kch stride_kcd stride_boff : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [4, 16] =>
        kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_kcn stride_kcb
          stride_kch stride_kcd stride_boff idx)) :
    ComputeCorrect.Realizes
      (kernel := fill_k_cache_tile KStates KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX stride_kss stride_ksh stride_ksd
        stride_kcn stride_kcb stride_kch stride_kcd stride_boff 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (KCaches,
          kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_kcn stride_kcb
            stride_kch stride_kcd stride_boff idx)))
      (expected := fun idx =>
        s.readMem KStates
          (kSourceOffset s SIDX stride_kss stride_ksh stride_ksd idx)) := by
  exact fill_k_cache_tile_compute_correct KStates KCaches BlockOffsets
    SIDX BIDX KV_BLOCK_IDX stride_kss stride_ksh stride_ksd stride_kcn
    stride_kcb stride_kch stride_kcd stride_boff 4 16 4 16 s hOutInj

theorem fill_v_cache_tile_test_h4_dv16_compute_correct
    (VStates VCaches BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX
      stride_vss stride_vsh stride_vsd
      stride_vcn stride_vcb stride_vch stride_vcd stride_boff : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [4, 16] =>
        vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_vcn stride_vcb
          stride_vch stride_vcd stride_boff idx)) :
    ComputeCorrect.Realizes
      (kernel := fill_v_cache_tile VStates VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX stride_vss stride_vsh stride_vsd
        stride_vcn stride_vcb stride_vch stride_vcd stride_boff 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (VCaches,
          vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_vcn stride_vcb
            stride_vch stride_vcd stride_boff idx)))
      (expected := fun idx =>
        s.readMem VStates
          (vSourceOffset s SIDX stride_vss stride_vsh stride_vsd idx)) := by
  exact fill_v_cache_tile_compute_correct VStates VCaches BlockOffsets
    SIDX BIDX KV_BLOCK_IDX stride_vss stride_vsh stride_vsd stride_vcn
    stride_vcb stride_vch stride_vcd stride_boff 4 16 4 16 s hOutInj

theorem fill_quant_k_cache_value_store_test_h4_d16_compute_correct
    (QKPre KCaches BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX
      stride_qss stride_qsh stride_qsd
      stride_kcn stride_kcb stride_kch stride_kcd stride_boff : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [4, 16] =>
        kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_kcn stride_kcb
          stride_kch stride_kcd stride_boff idx)) :
    ComputeCorrect.Realizes
      (kernel := fill_k_cache_tile QKPre KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX stride_qss stride_qsh stride_qsd
        stride_kcn stride_kcb stride_kch stride_kcd stride_boff 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (KCaches,
          kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_kcn stride_kcb
            stride_kch stride_kcd stride_boff idx)))
      (expected := fun idx =>
        s.readMem QKPre
          (kSourceOffset s SIDX stride_qss stride_qsh stride_qsd idx)) := by
  exact fill_quant_k_cache_value_store_slice_compute_correct QKPre KCaches
    BlockOffsets SIDX BIDX KV_BLOCK_IDX stride_qss stride_qsh stride_qsd
    stride_kcn stride_kcb stride_kch stride_kcd stride_boff 4 16 4 16
    s hOutInj

theorem fill_quant_v_cache_value_store_test_h4_dv16_compute_correct
    (QVPre VCaches BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX
      stride_qss stride_qsh stride_qsd
      stride_vcn stride_vcb stride_vch stride_vcd stride_boff : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [4, 16] =>
        vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_vcn stride_vcb
          stride_vch stride_vcd stride_boff idx)) :
    ComputeCorrect.Realizes
      (kernel := fill_v_cache_tile QVPre VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX stride_qss stride_qsh stride_qsd
        stride_vcn stride_vcb stride_vch stride_vcd stride_boff 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (VCaches,
          vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX stride_vcn stride_vcb
            stride_vch stride_vcd stride_boff idx)))
      (expected := fun idx =>
        s.readMem QVPre
          (vSourceOffset s SIDX stride_qss stride_qsh stride_qsd idx)) := by
  exact fill_quant_v_cache_value_store_slice_compute_correct QVPre VCaches
    BlockOffsets SIDX BIDX KV_BLOCK_IDX stride_qss stride_qsh stride_qsd
    stride_vcn stride_vcb stride_vch stride_vcd stride_boff 4 16 4 16
    s hOutInj

theorem fill_quant_k_scale_store_test_h4_compute_correct
    (KScalePre KScalesZeros BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX
      stride_kszn stride_kszb stride_kszh stride_kszd stride_boff : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin 4 =>
        metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 stride_kszn stride_kszb
          stride_kszh stride_kszd stride_boff i)) :
    ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice KScalePre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 stride_kszn stride_kszb stride_kszh stride_kszd
        stride_boff 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 stride_kszn
            stride_kszb stride_kszh stride_kszd stride_boff i)))
      (expected := fun i : Fin 4 => metaStoreSpec s KScalePre i) := by
  exact fill_quant_k_scale_store_slice_compute_correct KScalePre KScalesZeros
    BlockOffsets BIDX KV_BLOCK_IDX stride_kszn stride_kszb stride_kszh
    stride_kszd stride_boff 4 4 s hOutInj

theorem fill_quant_k_zero_store_test_h4_compute_correct
    (KZeroPre KScalesZeros BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX
      stride_kszn stride_kszb stride_kszh stride_kszd stride_boff : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin 4 =>
        metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 stride_kszn stride_kszb
          stride_kszh stride_kszd stride_boff i)) :
    ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice KZeroPre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 stride_kszn stride_kszb stride_kszh stride_kszd
        stride_boff 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 stride_kszn
            stride_kszb stride_kszh stride_kszd stride_boff i)))
      (expected := fun i : Fin 4 => metaStoreSpec s KZeroPre i) := by
  exact fill_quant_k_zero_store_slice_compute_correct KZeroPre KScalesZeros
    BlockOffsets BIDX KV_BLOCK_IDX stride_kszn stride_kszb stride_kszh
    stride_kszd stride_boff 4 4 s hOutInj

theorem fill_quant_v_scale_store_test_h4_compute_correct
    (VScalePre VScalesZeros BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX
      stride_vszn stride_vszb stride_vszh stride_vszd stride_boff : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin 4 =>
        metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 stride_vszn stride_vszb
          stride_vszh stride_vszd stride_boff i)) :
    ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice VScalePre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 stride_vszn stride_vszb stride_vszh stride_vszd
        stride_boff 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 stride_vszn
            stride_vszb stride_vszh stride_vszd stride_boff i)))
      (expected := fun i : Fin 4 => metaStoreSpec s VScalePre i) := by
  exact fill_quant_v_scale_store_slice_compute_correct VScalePre VScalesZeros
    BlockOffsets BIDX KV_BLOCK_IDX stride_vszn stride_vszb stride_vszh
    stride_vszd stride_boff 4 4 s hOutInj

theorem fill_quant_v_zero_store_test_h4_compute_correct
    (VZeroPre VScalesZeros BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX
      stride_vszn stride_vszb stride_vszh stride_vszd stride_boff : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin 4 =>
        metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 stride_vszn stride_vszb
          stride_vszh stride_vszd stride_boff i)) :
    ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice VZeroPre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 stride_vszn stride_vszb stride_vszh stride_vszd
        stride_boff 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 stride_vszn
            stride_vszb stride_vszh stride_vszd stride_boff i)))
      (expected := fun i : Fin 4 => metaStoreSpec s VZeroPre i) := by
  exact fill_quant_v_zero_store_slice_compute_correct VZeroPre VScalesZeros
    BlockOffsets BIDX KV_BLOCK_IDX stride_vszn stride_vszb stride_vszh
    stride_vszd stride_boff 4 4 s hOutInj

/-! ## Python test-layout wrappers

The checked Python test instantiates contiguous tensors with:
* `k_states/v_states.shape = (2, 32, 4, 16)`, inner strides `(64, 16, 1)`;
* `k_caches/v_caches.shape = (2, 8, 4, 16)`, strides `(512, 64, 16, 1)`;
* `k_scales_zeros/v_scales_zeros.shape = (2, 8, 4, 2)`, strides `(64, 8, 2, 1)`;
* `block_offsets.shape = (2, 5)`, stride `5`.

These wrappers discharge the concrete no-collision obligations for those
Python layouts, leaving only the already documented uint8 rounding/int4
packing semantics outside this file's real-valued store model. -/

theorem fill_cache_python_test_h4_d16_offset_injective
    (s : BlockState) (BlockOffsets : RegionName) (BIDX KV_BLOCK_IDX : Nat) :
    Function.Injective
      (fun idx : TileIndex [4, 16] =>
        kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx) := by
  intro a b h
  rcases a with ⟨ha, da, hta⟩
  rcases b with ⟨hb, db, htb⟩
  rcases da with ⟨da, hda⟩
  rcases db with ⟨db, hdb⟩
  simp [kCacheOffset, blockOff, headIndex, dimIndex, BlockState.readMemValue] at h
  have hh : ha = hb := by omega
  have hd : da = db := by omega
  subst hb
  subst db
  rfl

theorem fill_cache_python_test_v_h4_d16_offset_injective
    (s : BlockState) (BlockOffsets : RegionName) (BIDX KV_BLOCK_IDX : Nat) :
    Function.Injective
      (fun idx : TileIndex [4, 16] =>
        vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx) := by
  intro a b h
  rcases a with ⟨ha, da, hta⟩
  rcases b with ⟨hb, db, htb⟩
  rcases da with ⟨da, hda⟩
  rcases db with ⟨db, hdb⟩
  simp [vCacheOffset, blockOff, headIndex, dimIndex, BlockState.readMemValue] at h
  have hh : ha = hb := by omega
  have hd : da = db := by omega
  subst hb
  subst db
  rfl

theorem fill_cache_python_test_h4_meta_offset_injective
    (s : BlockState) (BlockOffsets : RegionName) (BIDX KV_BLOCK_IDX SZD : Nat) :
    Function.Injective
      (fun i : Fin 4 =>
        metaOffset s BlockOffsets BIDX KV_BLOCK_IDX SZD 64 8 2 1 5 i) := by
  intro a b h
  simp [metaOffset, blockOff, BlockState.readMemValue] at h
  apply Fin.ext
  omega

theorem fill_k_cache_tile_python_test_layout_compute_correct
    (KStates KCaches BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fill_k_cache_tile KStates KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (KCaches,
          kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem KStates (kSourceOffset s SIDX 64 16 1 idx)) := by
  exact fill_k_cache_tile_test_h4_d16_compute_correct KStates KCaches
    BlockOffsets SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 s
    (fill_cache_python_test_h4_d16_offset_injective s BlockOffsets BIDX
      KV_BLOCK_IDX)

theorem fill_v_cache_tile_python_test_layout_compute_correct
    (VStates VCaches BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fill_v_cache_tile VStates VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (VCaches,
          vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem VStates (vSourceOffset s SIDX 64 16 1 idx)) := by
  exact fill_v_cache_tile_test_h4_dv16_compute_correct VStates VCaches
    BlockOffsets SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 s
    (fill_cache_python_test_v_h4_d16_offset_injective s BlockOffsets BIDX
      KV_BLOCK_IDX)

theorem fill_quant_k_cache_value_store_python_test_layout_compute_correct
    (QKPre KCaches BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fill_k_cache_tile QKPre KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (KCaches,
          kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem QKPre (kSourceOffset s SIDX 64 16 1 idx)) := by
  exact fill_quant_k_cache_value_store_test_h4_d16_compute_correct QKPre
    KCaches BlockOffsets SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 s
    (fill_cache_python_test_h4_d16_offset_injective s BlockOffsets BIDX
      KV_BLOCK_IDX)

theorem fill_quant_v_cache_value_store_python_test_layout_compute_correct
    (QVPre VCaches BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fill_v_cache_tile QVPre VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (VCaches,
          vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem QVPre (vSourceOffset s SIDX 64 16 1 idx)) := by
  exact fill_quant_v_cache_value_store_test_h4_dv16_compute_correct QVPre
    VCaches BlockOffsets SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 s
    (fill_cache_python_test_v_h4_d16_offset_injective s BlockOffsets BIDX
      KV_BLOCK_IDX)

theorem fill_quant_k_scale_store_python_test_layout_compute_correct
    (KScalePre KScalesZeros BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice KScalePre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s KScalePre i) := by
  exact fill_quant_k_scale_store_test_h4_compute_correct KScalePre KScalesZeros
    BlockOffsets BIDX KV_BLOCK_IDX 64 8 2 1 5 s
    (fill_cache_python_test_h4_meta_offset_injective s BlockOffsets BIDX
      KV_BLOCK_IDX 0)

theorem fill_quant_k_zero_store_python_test_layout_compute_correct
    (KZeroPre KScalesZeros BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice KZeroPre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s KZeroPre i) := by
  exact fill_quant_k_zero_store_test_h4_compute_correct KZeroPre KScalesZeros
    BlockOffsets BIDX KV_BLOCK_IDX 64 8 2 1 5 s
    (fill_cache_python_test_h4_meta_offset_injective s BlockOffsets BIDX
      KV_BLOCK_IDX 1)

theorem fill_quant_v_scale_store_python_test_layout_compute_correct
    (VScalePre VScalesZeros BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice VScalePre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s VScalePre i) := by
  exact fill_quant_v_scale_store_test_h4_compute_correct VScalePre VScalesZeros
    BlockOffsets BIDX KV_BLOCK_IDX 64 8 2 1 5 s
    (fill_cache_python_test_h4_meta_offset_injective s BlockOffsets BIDX
      KV_BLOCK_IDX 0)

theorem fill_quant_v_zero_store_python_test_layout_compute_correct
    (VZeroPre VScalesZeros BlockOffsets : RegionName)
    (BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice VZeroPre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s VZeroPre i) := by
  exact fill_quant_v_zero_store_test_h4_compute_correct VZeroPre VScalesZeros
    BlockOffsets BIDX KV_BLOCK_IDX 64 8 2 1 5 s
    (fill_cache_python_test_h4_meta_offset_injective s BlockOffsets BIDX
      KV_BLOCK_IDX 1)

/-- Python non-quantized cache-fill layout: K-cache and V-cache tile writebacks
are both compute-correct for the observed `H = 4`, `D = 16` shape. -/
theorem fill_kv_cache_python_test_layout_all_outputs_compute_correct
    (KStates VStates KCaches VCaches BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := fill_k_cache_tile KStates KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (KCaches,
          kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem KStates (kSourceOffset s SIDX 64 16 1 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_v_cache_tile VStates VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (VCaches,
          vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem VStates (vSourceOffset s SIDX 64 16 1 idx))) := by
  constructor
  · exact fill_k_cache_tile_python_test_layout_compute_correct KStates
      KCaches BlockOffsets SIDX BIDX KV_BLOCK_IDX s
  · exact fill_v_cache_tile_python_test_layout_compute_correct VStates
      VCaches BlockOffsets SIDX BIDX KV_BLOCK_IDX s

/-- Public Python non-quantized cache-fill summary for the checked layout.

The surface conjunct pins the faithful `_fill_kv_cache_kernel` launch for
`num_heads = 4`, `head_dim = head_dim_v = 16`, `BLOCK = 4`, and contiguous
K/V state/cache strides used by the benchmark. The output conjunct exposes the
checked K-cache and V-cache tile writebacks for the selected source/cache
block positions. -/
theorem fill_kv_cache_python_test_layout_output_summary
    (KStates VStates KCaches VCaches QStartLoc QSeqLens KVSeqLens
      BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :
    (∃ alg, (fill_kv_cache_kernel_surface KStates VStates KCaches VCaches
      QStartLoc QSeqLens KVSeqLens BlockOffsets
      4 16 16 64 16 1 64 16 1 512 64 16 1 512 64 16 1 5 4 16 16 4
      ).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_k_cache_tile KStates KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (KCaches,
          kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem KStates (kSourceOffset s SIDX 64 16 1 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_v_cache_tile VStates VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (VCaches,
          vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem VStates (vSourceOffset s SIDX 64 16 1 idx))) := by
  constructor
  · exact fill_kv_cache_kernel_surface_toAlgorithm_supported KStates VStates
      KCaches VCaches QStartLoc QSeqLens KVSeqLens BlockOffsets
      4 16 16 64 16 1 64 16 1 512 64 16 1 512 64 16 1 5 4 16 16 4
  · exact fill_kv_cache_python_test_layout_all_outputs_compute_correct
      KStates VStates KCaches VCaches BlockOffsets SIDX BIDX KV_BLOCK_IDX s

/-- Python quantized cache-fill layout: K/V cache values plus K/V scale and
zero-point metadata stores are compute-correct for the observed shape, assuming
the quantized value tiles and metadata tiles have already been computed. -/
theorem fill_quant_kv_cache_python_test_layout_all_outputs_compute_correct
    (QKPre QVPre KScalePre KZeroPre VScalePre VZeroPre KCaches VCaches
      KScalesZeros VScalesZeros BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := fill_k_cache_tile QKPre KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (KCaches,
          kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem QKPre (kSourceOffset s SIDX 64 16 1 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_v_cache_tile QVPre VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (VCaches,
          vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem QVPre (vSourceOffset s SIDX 64 16 1 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice KScalePre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s KScalePre i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice KZeroPre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s KZeroPre i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice VScalePre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s VScalePre i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice VZeroPre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s VZeroPre i)) := by
  constructor
  · exact fill_quant_k_cache_value_store_python_test_layout_compute_correct
      QKPre KCaches BlockOffsets SIDX BIDX KV_BLOCK_IDX s
  constructor
  · exact fill_quant_v_cache_value_store_python_test_layout_compute_correct
      QVPre VCaches BlockOffsets SIDX BIDX KV_BLOCK_IDX s
  constructor
  · exact fill_quant_k_scale_store_python_test_layout_compute_correct
      KScalePre KScalesZeros BlockOffsets BIDX KV_BLOCK_IDX s
  constructor
  · exact fill_quant_k_zero_store_python_test_layout_compute_correct
      KZeroPre KScalesZeros BlockOffsets BIDX KV_BLOCK_IDX s
  constructor
  · exact fill_quant_v_scale_store_python_test_layout_compute_correct
      VScalePre VScalesZeros BlockOffsets BIDX KV_BLOCK_IDX s
  · exact fill_quant_v_zero_store_python_test_layout_compute_correct
      VZeroPre VScalesZeros BlockOffsets BIDX KV_BLOCK_IDX s

/-- Python `quant_policy = 8` summary for the checked cache-fill layout.

This combines the `_quant_int8` helper surface, including min/max scale and
zero computation plus the uint8 cast, with the checked K/V cache and metadata
writeback obligations at the observed `H = 4`, `D = 16` test shape. -/
theorem fill_quant_int8_kv_cache_python_test_layout_summary
    (KStates VStates QKPre QVPre KScalePre KZeroPre VScalePre VZeroPre KCaches
      VCaches KScalesZeros VScalesZeros BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :
    (∃ alg, (quant_int8_compute_store_slice KStates QKPre KScalePre KZeroPre
      16 1 16 1 4 16 4 16).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (quant_int8_compute_store_slice VStates QVPre VScalePre VZeroPre
      16 1 16 1 4 16 4 16).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_k_cache_tile QKPre KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (KCaches,
          kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem QKPre (kSourceOffset s SIDX 64 16 1 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_v_cache_tile QVPre VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (VCaches,
          vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem QVPre (vSourceOffset s SIDX 64 16 1 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice KScalePre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s KScalePre i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice KZeroPre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s KZeroPre i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice VScalePre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s VScalePre i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice VZeroPre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s VZeroPre i)) := by
  constructor
  · exact quant_int8_python_test_layout_surface_toAlgorithm_supported KStates
      QKPre KScalePre KZeroPre
  constructor
  · exact quant_int8_python_test_layout_surface_toAlgorithm_supported VStates
      QVPre VScalePre VZeroPre
  · exact fill_quant_kv_cache_python_test_layout_all_outputs_compute_correct
      QKPre QVPre KScalePre KZeroPre VScalePre VZeroPre KCaches VCaches
      KScalesZeros VScalesZeros BlockOffsets SIDX BIDX KV_BLOCK_IDX s

/-- Python `quant_policy = 4` summary for the checked cache-fill layout.

The int4 helper proof keeps both half-lane uint8 casts and the
`q_val1 + q_val2 * 16` packing formula in the lowering surface, then reuses
the same checked K/V cache and metadata writeback obligations. -/
theorem fill_quant_int4_kv_cache_python_test_layout_summary
    (KStatesLo KStatesHi VStatesLo VStatesHi QKPre QVPre KScalePre KZeroPre
      VScalePre VZeroPre KCaches VCaches KScalesZeros VScalesZeros
      BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :
    (∃ alg, (quant_int4_compute_store_slice KStatesLo KStatesHi QKPre
      KScalePre KZeroPre 16 1 16 1 8 1 4 8 4 8).toAlgorithm? =
        Except.ok alg) ∧
    (∃ alg, (quant_int4_compute_store_slice VStatesLo VStatesHi QVPre
      VScalePre VZeroPre 16 1 16 1 8 1 4 8 4 8).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_k_cache_tile QKPre KCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (KCaches,
          kCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem QKPre (kSourceOffset s SIDX 64 16 1 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_v_cache_tile QVPre VCaches BlockOffsets
        SIDX BIDX KV_BLOCK_IDX 64 16 1 512 64 16 1 5 4 16 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 16 4 16)
        (fun idx => (VCaches,
          vCacheOffset s BlockOffsets BIDX KV_BLOCK_IDX 512 64 16 1 5 idx)))
      (expected := fun idx =>
        s.readMem QVPre (vSourceOffset s SIDX 64 16 1 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice KScalePre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s KScalePre i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice KZeroPre KScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (KScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s KZeroPre i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice VScalePre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 0 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 0 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s VScalePre i)) ∧
    (ComputeCorrect.Realizes
      (kernel := fill_quant_meta_store_slice VZeroPre VScalesZeros BlockOffsets
        BIDX KV_BLOCK_IDX 1 64 8 2 1 5 4 4)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 4 => metaActive s 4 4 i)
        (fun i : Fin 4 => (VScalesZeros,
          metaOffset s BlockOffsets BIDX KV_BLOCK_IDX 1 64 8 2 1 5 i)))
      (expected := fun i : Fin 4 => metaStoreSpec s VZeroPre i)) := by
  constructor
  · exact quant_int4_python_test_layout_surface_toAlgorithm_supported KStatesLo
      KStatesHi QKPre KScalePre KZeroPre
  constructor
  · exact quant_int4_python_test_layout_surface_toAlgorithm_supported VStatesLo
      VStatesHi QVPre VScalePre VZeroPre
  · exact fill_quant_kv_cache_python_test_layout_all_outputs_compute_correct
      QKPre QVPre KScalePre KZeroPre VScalePre VZeroPre KCaches VCaches
      KScalesZeros VScalesZeros BlockOffsets SIDX BIDX KV_BLOCK_IDX s

/-- `output_summary` alias for the checked int8 quantized KV cache-fill path. -/
abbrev fill_quant_int8_kv_cache_python_test_layout_output_summary
    (KStates VStates QKPre QVPre KScalePre KZeroPre VScalePre VZeroPre KCaches
      VCaches KScalesZeros VScalesZeros BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :=
  fill_quant_int8_kv_cache_python_test_layout_summary KStates VStates QKPre
    QVPre KScalePre KZeroPre VScalePre VZeroPre KCaches VCaches KScalesZeros
    VScalesZeros BlockOffsets SIDX BIDX KV_BLOCK_IDX s

/-- `output_summary` alias for the checked int4 quantized KV cache-fill path. -/
abbrev fill_quant_int4_kv_cache_python_test_layout_output_summary
    (KStatesLo KStatesHi VStatesLo VStatesHi QKPre QVPre KScalePre KZeroPre
      VScalePre VZeroPre KCaches VCaches KScalesZeros VScalesZeros
      BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :=
  fill_quant_int4_kv_cache_python_test_layout_summary KStatesLo KStatesHi
    VStatesLo VStatesHi QKPre QVPre KScalePre KZeroPre VScalePre VZeroPre
    KCaches VCaches KScalesZeros VScalesZeros BlockOffsets SIDX BIDX
    KV_BLOCK_IDX s

/-- Complete checked-layout target for `kv_cache_filling.py`.

This gathers the non-quantized cache fill, the `quant_policy = 8` uint8 path,
and the `quant_policy = 4` packed-int4 path. The summaries expose K and V cache
writebacks, scale/zero metadata stores, quant helper surfaces, and the
sequence/block-addressed cache layout used by the Python tests. -/
abbrev fill_kv_cache_python_test_layout_complete_summary
    (KStates VStates KStatesLo KStatesHi VStatesLo VStatesHi QKPre QVPre
      KScalePre KZeroPre VScalePre VZeroPre KCaches VCaches KScalesZeros
      VScalesZeros QStartLoc QSeqLens KVSeqLens BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX : Nat) (s : BlockState) :=
  And.intro
    (fill_kv_cache_python_test_layout_output_summary KStates VStates KCaches
      VCaches QStartLoc QSeqLens KVSeqLens BlockOffsets SIDX BIDX KV_BLOCK_IDX s)
    (And.intro
      (fill_quant_int8_kv_cache_python_test_layout_summary KStates VStates
        QKPre QVPre KScalePre KZeroPre VScalePre VZeroPre KCaches VCaches
        KScalesZeros VScalesZeros BlockOffsets SIDX BIDX KV_BLOCK_IDX s)
      (fill_quant_int4_kv_cache_python_test_layout_summary KStatesLo KStatesHi
        VStatesLo VStatesHi QKPre QVPre KScalePre KZeroPre VScalePre VZeroPre
        KCaches VCaches KScalesZeros VScalesZeros BlockOffsets SIDX BIDX
        KV_BLOCK_IDX s))

end VeriTile.Bench.TritonBenchG.KvCacheFilling
