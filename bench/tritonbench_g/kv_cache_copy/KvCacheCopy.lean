import VeriTile.Triton

/-!
# `kv_cache_copy` — strict per-kernel correctness

`_copy_to_kvcache_seqlen1_kernel` is a decode-stage paged KV-cache scatter:
program `(cur_seq_idx, cur_kv_head_idx)` reads `context_lengths[cur_seq_idx]`
and the per-sequence block table to locate the destination block / slot, then
gathers the seq-len-1 K/V rows from `K` / `V` and scatters them into the paged
`KCache` / `VCache` for that head. It supports two cache layouts: the legacy
`[num_blocks, num_kv_heads, block_size, head_dim]` and the split-x
`[num_blocks, num_kv_heads, head_dim // x, block_size, x]` layout, the latter
walked over a `static_range(HEAD_DIM // KCACHE_X)` of x-blocks.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (grid over `(num_seqs, num_kv_heads)`, the block-table /
context-length inputs, and how the runtime composes per-program scatters into
the paged cache buffers) is the *trusted boundary*, not a proof obligation here.
Because the program ids are universally quantified, the per-program statement
covers every program of the grid.

## Proof architecture

```
kv_cache_copy_correctness                       ← TOP SPECIFICATION
  · kvCacheCopyIO ⊨ (xs, ys): the chained-metadata masked copy triple
    (ChainMetaMasked2DKernelIO₂ₓ₂ — slot 1 = context_lengths[pid₀], slot 2 =
     the block-table cell whose ADDRESS eats slot 1's loaded value)
  ├─ copy_to_kvcache_seqlen1_xblock_flattenOk    bridge fragment membership
  ├─ copy_to_kvcache_seqlen1_xblock_traceSafe    per-execution safety walk
  └─ copy_to_kvcache_seqlen1_xblock_region_run   region-model chained triple
copy_to_kvcache_seqlen1_kernel_toAlgorithm_supported  faithful full-loop kernel lowers
copy_to_kcache_seqlen1_xblock_correct /               per-channel audit slices
copy_to_vcache_seqlen1_dblock_correct                 (single-store exec value lemmas)
```

The headline kernel `copy_to_kvcache_seqlen1_xblock` is the decode prologue
plus ONE `split_x` iteration of the Python static loop, with BOTH stores (K
and V) in a single program. Instantiating `SPLIT_X` at every
`split_x < HEAD_DIM / KCACHE_X` covers the whole loop; the legacy layout is
the instance `SPLIT_X = 0`, `KCACHE_X = HEAD_DIM`, `stride_kcsplit_x = 0`.
The single-store slices are kept as independently audited per-channel value
lemmas.

## Modeling boundary

Arithmetic/values are over `ℝ` (not bit-accurate IEEE float); dtype casts are
erased (post-erasure all dtypes unify to `ℝ`). The destination block id and
in-block slot are data-dependent: `block_id` is gathered from `BLOCK_TABLES`
at an address computed from `context_lengths[cur_seq_idx] - 1` (Nat-truncated
subtraction, exactly as the DSL executes it) — the `⊨` headline pins the two
**raw** loaded scalars as named binders `m₁` (context length) and `m₂` (block
id); the `- 1` / `/ block_size` / `% block_size` arithmetic lives in the
declared windows. Everything is dimension-general; no-aliasing of the
block-id-dependent cache windows enters through the skin's per-pinned-context
`WriteInj` readback gates (not as `∀`-quantified hypotheses), plus the
`KCache ≠ VCache` side condition (the V store must not clobber the K output).
-/

namespace VeriTile.Bench.TritonBenchG.KvCacheCopy

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `kv_cache_copy.py`'s
`_copy_to_kvcache_seqlen1_kernel`. -/
def copy_to_kvcache_seqlen1_kernel
    (K V KCache VCache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (stride_kt stride_kh stride_kd
      stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs _stride_kcd
      stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat) :
    ComputeKernel := triton {
  cur_seq_idx = tl.program_id(0)
  cur_kv_head_idx = tl.program_id(1)
  past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - $(1)
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  block_table_ptr = BLOCK_TABLES + cur_seq_idx * $(stride_bts)
  block_id = tl.load(block_table_ptr + last_bt_block_idx * $(stride_btb))
  offsets_in_last_block = past_kv_seq_len % $(block_size)
  range_x = tl.arange(0, $(KCACHE_X))
  offsets_dmodel_x_partition = tl.arange(0, $(KCACHE_X))
  for split_x in tl.static_range($((HEAD_DIM / KCACHE_X : Nat))) {
    offsets_dmodel_x_partition =
      tl.arange(split_x * $(KCACHE_X), (split_x + $(1)) * $(KCACHE_X))
    offsets_k = cur_seq_idx * $(stride_kt) + cur_kv_head_idx * $(stride_kh) +
      offsets_dmodel_x_partition * $(stride_kd)
    k = tl.load(K + offsets_k)
    offsets_v = cur_seq_idx * $(stride_vt) + cur_kv_head_idx * $(stride_vh) +
      offsets_dmodel_x_partition * $(stride_vd)
    v = tl.load(V + offsets_v)
    offsets_kcache = block_id * $(stride_kcb) +
      cur_kv_head_idx * $(stride_kch) +
      split_x * $(stride_kcsplit_x) +
      offsets_in_last_block * $(stride_kcs) +
      range_x
    tl.store(KCache + offsets_kcache, k)
    offsets_vcache = block_id * $(stride_vcb) +
      cur_kv_head_idx * $(stride_vch) +
      offsets_in_last_block * $(stride_vcs) +
      offsets_dmodel_x_partition * $(stride_vcd)
    tl.store(VCache + offsets_vcache, v)
  }
}

/-- The full `kv_cache_copy.py` seqlen=1 copy surface lowers to the algorithm
layer, including block-table/context-length address arithmetic and both K/V
cache stores. -/
theorem copy_to_kvcache_seqlen1_kernel_toAlgorithm_supported
    (K V KCache VCache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (stride_kt stride_kh stride_kd
      stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_kcd
      stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat) :
    ∃ alg, (copy_to_kvcache_seqlen1_kernel K V KCache VCache BLOCK_TABLES
      context_lengths stride_kt stride_kh stride_kd stride_vt stride_vh
      stride_vd stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_kcd
      stride_vcb stride_vch stride_vcs stride_vcd stride_bts stride_btb
      block_size HEAD_DIM KCACHE_X).toAlgorithm? = Except.ok alg := by
  simp [copy_to_kvcache_seqlen1_kernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Per-channel audit slice: the K-cache store of `kv_cache_copy.py`'s
`_copy_to_kvcache_seqlen1_kernel`, keeping the decode-path arithmetic
(`context_lengths[cur_seq_idx] - 1`, block-table lookup, last-block offset),
one K load, and the K-cache store for one `split_x` partition. -/
def copy_to_kcache_seqlen1_xblock
    (K KCache : RegionName) (BLOCK_TABLES context_lengths : Region .nat)
    (SPLIT_X stride_kt stride_kh stride_kd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat) :
    ComputeKernel := triton {
  cur_seq_idx = tl.program_id(0)
  cur_kv_head_idx = tl.program_id(1)
  past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - $(1)
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    last_bt_block_idx * $(stride_btb))
  offsets_in_last_block = past_kv_seq_len % $(block_size)
  range_x = tl.arange(0, $(KCACHE_X))
  offsets_dmodel_x_partition = $(SPLIT_X) * $(KCACHE_X) + range_x
  k = tl.load(K + cur_seq_idx * $(stride_kt) +
      cur_kv_head_idx * $(stride_kh) +
      offsets_dmodel_x_partition * $(stride_kd),
    mask=offsets_dmodel_x_partition < $(HEAD_DIM), other=0.0)
  tl.store(KCache + block_id * $(stride_kcb) +
      cur_kv_head_idx * $(stride_kch) +
      $(SPLIT_X) * $(stride_kcsplit_x) +
      offsets_in_last_block * $(stride_kcs) + range_x,
    k, mask=offsets_dmodel_x_partition < $(HEAD_DIM))
}

/-- Surface transcription of the V-cache store in `kv_cache_copy.py`'s
`_copy_to_kvcache_seqlen1_kernel`.

This keeps the Python decode path's `context_lengths[cur_seq_idx] - 1`, block
division/modulo, block-table lookup, V load, and V-cache store for one
dimension block. -/
def copy_to_vcache_seqlen1_dblock
    (V VCache : RegionName) (BLOCK_TABLES context_lengths : Region .nat)
    (stride_vt stride_vh stride_vd stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM BLOCK_D : Nat) :
    ComputeKernel := triton {
  cur_seq_idx = tl.program_id(0)
  cur_kv_head_idx = tl.program_id(1)
  d = tl.arange(0, $(BLOCK_D))
  past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - $(1)
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    last_bt_block_idx * $(stride_btb))
  offset_last_block = past_kv_seq_len % $(block_size)
  v = tl.load(V + cur_seq_idx * $(stride_vt) +
      cur_kv_head_idx * $(stride_vh) + d * $(stride_vd),
    mask=d < $(HEAD_DIM), other=0.0)
  tl.store(VCache + block_id * $(stride_vcb) +
      cur_kv_head_idx * $(stride_vch) + offset_last_block * $(stride_vcs) +
      d * $(stride_vcd),
    v, mask=d < $(HEAD_DIM))
}

def dimIndex (i : Fin BLOCK_D) : Nat :=
  i.val

def active (HEAD_DIM : Nat) (i : Fin BLOCK_D) : Prop :=
  dimIndex i < HEAD_DIM

instance activeDecidable (HEAD_DIM : Nat) (i : Fin BLOCK_D) :
    Decidable (active HEAD_DIM i) := by
  unfold active
  infer_instance

def kDimIndex (SPLIT_X KCACHE_X : Nat) (i : Fin KCACHE_X) : Nat :=
  SPLIT_X * KCACHE_X + i.val

def kActive (SPLIT_X HEAD_DIM KCACHE_X : Nat) (i : Fin KCACHE_X) : Prop :=
  kDimIndex SPLIT_X KCACHE_X i < HEAD_DIM

instance kActiveDecidable (SPLIT_X HEAD_DIM KCACHE_X : Nat) (i : Fin KCACHE_X) :
    Decidable (kActive SPLIT_X HEAD_DIM KCACHE_X i) := by
  unfold kActive
  infer_instance

def kSourceOffset
    (s : BlockState) (SPLIT_X stride_kt stride_kh stride_kd KCACHE_X : Nat)
    (i : Fin KCACHE_X) : Nat :=
  s.pids 0 * stride_kt + s.pids 1 * stride_kh +
    kDimIndex SPLIT_X KCACHE_X i * stride_kd

def seqlen1PastKvSeqLen (s : BlockState) (context_lengths : RegionName) : Nat :=
  s.readMemValue .nat context_lengths (s.pids 0) - 1

def seqlen1LastBlockIdx (s : BlockState) (context_lengths : RegionName)
    (block_size : Nat) : Nat :=
  seqlen1PastKvSeqLen s context_lengths / block_size

def seqlen1OffsetLastBlock (s : BlockState) (context_lengths : RegionName)
    (block_size : Nat) : Nat :=
  seqlen1PastKvSeqLen s context_lengths % block_size

def seqlen1BlockId (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (stride_bts stride_btb block_size : Nat) : Nat :=
  s.readMemValue .nat BLOCK_TABLES
    (s.pids 0 * stride_bts +
      seqlen1LastBlockIdx s context_lengths block_size * stride_btb)

def seqlen1KCacheOffset
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (SPLIT_X stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
      stride_btb block_size KCACHE_X : Nat)
    (i : Fin KCACHE_X) : Nat :=
  seqlen1BlockId s BLOCK_TABLES context_lengths stride_bts stride_btb block_size *
      stride_kcb +
    s.pids 1 * stride_kch + SPLIT_X * stride_kcsplit_x +
    seqlen1OffsetLastBlock s context_lengths block_size * stride_kcs + i.val

/-- Algorithm-layer correctness for the K-cache seqlen=1 copy surface. -/
theorem copy_to_kcache_seqlen1_xblock_correct
    (K KCache BLOCK_TABLES context_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        seqlen1KCacheOffset s BLOCK_TABLES context_lengths SPLIT_X stride_kcb
          stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb
          block_size KCACHE_X i))
    (hExec : exec (copy_to_kcache_seqlen1_xblock K KCache BLOCK_TABLES
        context_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb
        stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
        HEAD_DIM KCACHE_X) s = some s') :
    ∀ i : Fin KCACHE_X,
      s'.readMem KCache
          (seqlen1KCacheOffset s BLOCK_TABLES context_lengths SPLIT_X
            stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
            stride_btb block_size KCACHE_X i) =
        if kActive SPLIT_X HEAD_DIM KCACHE_X i then
          s.readMem K (kSourceOffset s SPLIT_X stride_kt stride_kh stride_kd
            KCACHE_X i)
        else
          s.readMem KCache
            (seqlen1KCacheOffset s BLOCK_TABLES context_lengths SPLIT_X
              stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
              stride_btb block_size KCACHE_X i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [KCACHE_X] =>
        s.readMemValue .nat BLOCK_TABLES
            (s.pids 0 * stride_bts +
              ((s.readMemValue .nat context_lengths (s.pids 0) - 1) / block_size) *
                stride_btb) * stride_kcb +
          s.pids 1 * stride_kch + SPLIT_X * stride_kcsplit_x +
          ((s.readMemValue .nat context_lengths (s.pids 0) - 1) % block_size) *
            stride_kcs + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [seqlen1KCacheOffset, seqlen1BlockId, seqlen1LastBlockIdx,
        seqlen1OffsetLastBlock, seqlen1PastKvSeqLen, BlockState.readMemValue]
        using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBX : 0 < KCACHE_X
  · simp [exec, copy_to_kcache_seqlen1_xblock, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
          BlockState.readMemValue, hBX] at hExec
    rw [← hExec]
    let st : BlockState :=
      s.setReg "cur_seq_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
        |>.setReg "cur_kv_head_idx" TileDType.nat [] (Tile.scalar (s.pids 1))
        |>.setReg "past_kv_seq_len" TileDType.nat []
          (Tile.scalar (s.readMemValue .nat context_lengths (s.pids 0) - 1))
        |>.setReg "last_bt_block_idx" TileDType.nat []
          (Tile.scalar ((s.readMemValue .nat context_lengths (s.pids 0) - 1) /
            block_size))
        |>.setReg "block_id" TileDType.nat []
          (Tile.scalar (s.readMemValue .nat BLOCK_TABLES
            (s.pids 0 * stride_bts +
              ((s.readMemValue .nat context_lengths (s.pids 0) - 1) /
                block_size) * stride_btb)))
        |>.setReg "offsets_in_last_block" TileDType.nat []
          (Tile.scalar ((s.readMemValue .nat context_lengths (s.pids 0) - 1) %
            block_size))
        |>.setReg "range_x" TileDType.nat [KCACHE_X] (Tile.vec fun i => i.val)
        |>.setReg "offsets_dmodel_x_partition" TileDType.nat [KCACHE_X]
          { data := fun i => SPLIT_X * KCACHE_X + i.1.val }
        |>.setReg "k" TileDType.real [KCACHE_X]
          { data := fun i =>
            if SPLIT_X * KCACHE_X + i.1.val < HEAD_DIM then
              some (s.readMem K
                (s.pids 0 * stride_kt + s.pids 1 * stride_kh +
                  (SPLIT_X * KCACHE_X + i.1.val) * stride_kd))
            else some (0.0 : ℝ) }
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := KCache)
        (shape := [KCACHE_X])
        (s := st)
        (offsetFn := fun idx : TileIndex [KCACHE_X] =>
          s.readMemValue .nat BLOCK_TABLES
              (s.pids 0 * stride_bts +
                ((s.readMemValue .nat context_lengths (s.pids 0) - 1) / block_size) *
                  stride_btb) * stride_kcb +
            s.pids 1 * stride_kch + SPLIT_X * stride_kcsplit_x +
            ((s.readMemValue .nat context_lengths (s.pids 0) - 1) % block_size) *
              stride_kcs + idx.1.val)
        (valueFn := fun idx : TileIndex [KCACHE_X] =>
          WithBot.unbotD 0
            (if SPLIT_X * KCACHE_X + idx.1.val < HEAD_DIM then
              some (s.readMem K
                (s.pids 0 * stride_kt + s.pids 1 * stride_kh +
                  (SPLIT_X * KCACHE_X + idx.1.val) * stride_kd))
            else some (0.0 : ℝ)))
        (P := fun idx : TileIndex [KCACHE_X] =>
          SPLIT_X * KCACHE_X + idx.1.val < HEAD_DIM)
        hRawInj (i, PUnit.unit))
    simp [BlockState.readMemValue] at hScatter
    have hScatter' := by
      simpa [st] using hScatter
    simp only [kActive, kSourceOffset, seqlen1KCacheOffset, seqlen1BlockId,
      seqlen1LastBlockIdx, seqlen1OffsetLastBlock, seqlen1PastKvSeqLen,
      kDimIndex, BlockState.readMemValue]
    by_cases hi : SPLIT_X * KCACHE_X + i.val < HEAD_DIM
    · simpa [hi] using hScatter'
    · simpa [hi] using hScatter'
  · exact False.elim (hBX (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

def vSourceOffset
    (s : BlockState) (stride_vt stride_vh stride_vd : Nat)
    (i : Fin BLOCK_D) : Nat :=
  s.pids 0 * stride_vt + s.pids 1 * stride_vh + dimIndex i * stride_vd

def seqlen1VCacheOffset
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (stride_vcb stride_vch stride_vcs stride_vcd stride_bts stride_btb
      block_size : Nat)
    (i : Fin BLOCK_D) : Nat :=
  seqlen1BlockId s BLOCK_TABLES context_lengths stride_bts stride_btb block_size *
      stride_vcb +
    s.pids 1 * stride_vch +
    seqlen1OffsetLastBlock s context_lengths block_size * stride_vcs +
    dimIndex i * stride_vcd

/-- Algorithm-layer correctness for the V-cache seqlen=1 copy surface. -/
theorem copy_to_vcache_seqlen1_dblock_correct
    (V VCache BLOCK_TABLES context_lengths : RegionName)
    (stride_vt stride_vh stride_vd stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM BLOCK_D : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_D =>
        seqlen1VCacheOffset s BLOCK_TABLES context_lengths stride_vcb stride_vch
          stride_vcs stride_vcd stride_bts stride_btb block_size i))
    (hExec : exec (copy_to_vcache_seqlen1_dblock V VCache BLOCK_TABLES
        context_lengths stride_vt stride_vh stride_vd stride_vcb stride_vch
        stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM BLOCK_D)
        s = some s') :
    ∀ i : Fin BLOCK_D,
      s'.readMem VCache
          (seqlen1VCacheOffset s BLOCK_TABLES context_lengths stride_vcb
            stride_vch stride_vcs stride_vcd stride_bts stride_btb block_size i) =
        if active HEAD_DIM i then
          s.readMem V (vSourceOffset s stride_vt stride_vh stride_vd i)
        else
          s.readMem VCache
            (seqlen1VCacheOffset s BLOCK_TABLES context_lengths stride_vcb
              stride_vch stride_vcs stride_vcd stride_bts stride_btb block_size i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_D] =>
        s.readMemValue .nat BLOCK_TABLES
            (s.pids 0 * stride_bts +
              ((s.readMemValue .nat context_lengths (s.pids 0) - 1) / block_size) *
                stride_btb) * stride_vcb +
          s.pids 1 * stride_vch +
          ((s.readMemValue .nat context_lengths (s.pids 0) - 1) % block_size) *
            stride_vcs + idx.1.val * stride_vcd) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [seqlen1VCacheOffset, seqlen1BlockId, seqlen1LastBlockIdx,
        seqlen1OffsetLastBlock, seqlen1PastKvSeqLen, dimIndex,
        BlockState.readMemValue] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBD : 0 < BLOCK_D
  · simp [exec, copy_to_vcache_seqlen1_dblock, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
          BlockState.readMemValue, hBD] at hExec
    rw [← hExec]
    let st : BlockState :=
      s.setReg "cur_seq_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
        |>.setReg "cur_kv_head_idx" TileDType.nat [] (Tile.scalar (s.pids 1))
        |>.setReg "d" TileDType.nat [BLOCK_D] (Tile.vec fun i => i.val)
        |>.setReg "past_kv_seq_len" TileDType.nat []
          (Tile.scalar (s.readMemValue .nat context_lengths (s.pids 0) - 1))
        |>.setReg "last_bt_block_idx" TileDType.nat []
          (Tile.scalar ((s.readMemValue .nat context_lengths (s.pids 0) - 1) /
            block_size))
        |>.setReg "block_id" TileDType.nat []
          (Tile.scalar (s.readMemValue .nat BLOCK_TABLES
            (s.pids 0 * stride_bts +
              ((s.readMemValue .nat context_lengths (s.pids 0) - 1) /
                block_size) * stride_btb)))
        |>.setReg "offset_last_block" TileDType.nat []
          (Tile.scalar ((s.readMemValue .nat context_lengths (s.pids 0) - 1) %
            block_size))
        |>.setReg "v" TileDType.real [BLOCK_D]
          { data := fun i =>
            if i.1.val < HEAD_DIM then
              some (s.readMem V
                (s.pids 0 * stride_vt + s.pids 1 * stride_vh +
                  i.1.val * stride_vd))
            else some (0.0 : ℝ) }
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := VCache)
        (shape := [BLOCK_D])
        (s := st)
        (offsetFn := fun idx : TileIndex [BLOCK_D] =>
          s.readMemValue .nat BLOCK_TABLES
              (s.pids 0 * stride_bts +
                ((s.readMemValue .nat context_lengths (s.pids 0) - 1) / block_size) *
                  stride_btb) * stride_vcb +
            s.pids 1 * stride_vch +
            ((s.readMemValue .nat context_lengths (s.pids 0) - 1) % block_size) *
              stride_vcs + idx.1.val * stride_vcd)
        (valueFn := fun idx : TileIndex [BLOCK_D] =>
          WithBot.unbotD 0
            (if idx.1.val < HEAD_DIM then
              some (s.readMem V
                (s.pids 0 * stride_vt + s.pids 1 * stride_vh +
                  idx.1.val * stride_vd))
            else some (0.0 : ℝ)))
        (P := fun idx : TileIndex [BLOCK_D] => idx.1.val < HEAD_DIM)
        hRawInj (i, PUnit.unit))
    simp [BlockState.readMemValue] at hScatter
    have hScatter' := by
      simpa [st] using hScatter
    simp only [active, vSourceOffset, seqlen1VCacheOffset, seqlen1BlockId,
      seqlen1LastBlockIdx, seqlen1OffsetLastBlock, seqlen1PastKvSeqLen,
      dimIndex, BlockState.readMemValue]
    by_cases hi : i.val < HEAD_DIM
    · simpa [hi] using hScatter'
    · simpa [hi] using hScatter'
  · exact False.elim (hBD (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-! ## The combined per-split surface, chained-metadata triple, and headline -/

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for one masked store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

/-- **The headline surface**: the decode prologue plus ONE `split_x` iteration
of `_copy_to_kvcache_seqlen1_kernel`'s static loop, with BOTH cache stores in
a single program — `context_lengths[cur_seq_idx] - 1`, block division/modulo,
the chained block-table lookup, the masked K load / K-cache store and the
masked V load / V-cache store for the `SPLIT_X` x-partition. Instantiating
`SPLIT_X` at every `split_x < HEAD_DIM / KCACHE_X` covers the whole loop; the
legacy cache layout is the instance `SPLIT_X = 0`, `KCACHE_X = HEAD_DIM`,
`stride_kcsplit_x = 0`. -/
def copy_to_kvcache_seqlen1_xblock
    (K V KCache VCache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (SPLIT_X stride_kt stride_kh stride_kd
      stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat) :
    ComputeKernel := triton {
  cur_seq_idx = tl.program_id(0)
  cur_kv_head_idx = tl.program_id(1)
  past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - $(1)
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    last_bt_block_idx * $(stride_btb))
  offsets_in_last_block = past_kv_seq_len % $(block_size)
  range_x = tl.arange(0, $(KCACHE_X))
  offsets_dmodel_x_partition = $(SPLIT_X) * $(KCACHE_X) + range_x
  k = tl.load(K + cur_seq_idx * $(stride_kt) +
      cur_kv_head_idx * $(stride_kh) +
      offsets_dmodel_x_partition * $(stride_kd),
    mask=offsets_dmodel_x_partition < $(HEAD_DIM), other=0.0)
  v = tl.load(V + cur_seq_idx * $(stride_vt) +
      cur_kv_head_idx * $(stride_vh) +
      offsets_dmodel_x_partition * $(stride_vd),
    mask=offsets_dmodel_x_partition < $(HEAD_DIM), other=0.0)
  tl.store(KCache + block_id * $(stride_kcb) +
      cur_kv_head_idx * $(stride_kch) +
      $(SPLIT_X) * $(stride_kcsplit_x) +
      offsets_in_last_block * $(stride_kcs) + range_x,
    k, mask=offsets_dmodel_x_partition < $(HEAD_DIM))
  tl.store(VCache + block_id * $(stride_vcb) +
      cur_kv_head_idx * $(stride_vch) +
      offsets_in_last_block * $(stride_vcs) +
      offsets_dmodel_x_partition * $(stride_vcd),
    v, mask=offsets_dmodel_x_partition < $(HEAD_DIM))
}

/-- The combined per-split surface sits inside the flat-memory bridge's
covered fragment (scalar `.nat` slot loads, Nat sub/div/mod register
arithmetic, pointer arithmetic, masked loads with `other`, masked stores). -/
theorem copy_to_kvcache_seqlen1_xblock_flattenOk
    (K V KCache VCache BLOCK_TABLES context_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd
      stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat) :
    ((copy_to_kvcache_seqlen1_xblock K V KCache VCache BLOCK_TABLES
      context_lengths SPLIT_X stride_kt stride_kh stride_kd stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_vcb stride_vch
      stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM KCACHE_X).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [copy_to_kvcache_seqlen1_xblock, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Per-execution safety walk: one computational unfold walks all eleven
statements — the register arithmetic is memory-silent — and reduces the six
memory accesses to the skin-shaped hypotheses: the two **unmasked scalar-slot
loads** need their cells in bounds (`context_lengths[pid₀]`, then the
**chained** block-table cell whose address eats the first loaded value), and
the masked K/V loads and K/V-cache stores need their windows in bounds at the
active lanes `SPLIT_X·KCACHE_X + j < HEAD_DIM` (kernel-level spelling:
`s.readMemValue .nat …` for the loaded scalars). -/
theorem copy_to_kvcache_seqlen1_xblock_traceSafe
    (K V KCache VCache BLOCK_TABLES context_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd
      stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hb₁ : s.pids 0 < bounds context_lengths)
    (hb₂ : s.pids 0 * stride_bts +
      ((s.readMemValue .nat context_lengths (s.pids 0) - 1) / block_size) *
        stride_btb < bounds BLOCK_TABLES)
    (hrk : ∀ j : Fin KCACHE_X, SPLIT_X * KCACHE_X + j.val < HEAD_DIM →
      s.pids 0 * stride_kt + s.pids 1 * stride_kh +
        (SPLIT_X * KCACHE_X + j.val) * stride_kd < bounds K)
    (hrv : ∀ j : Fin KCACHE_X, SPLIT_X * KCACHE_X + j.val < HEAD_DIM →
      s.pids 0 * stride_vt + s.pids 1 * stride_vh +
        (SPLIT_X * KCACHE_X + j.val) * stride_vd < bounds V)
    (hwk : ∀ j : Fin KCACHE_X, SPLIT_X * KCACHE_X + j.val < HEAD_DIM →
      s.readMemValue .nat BLOCK_TABLES
          (s.pids 0 * stride_bts +
            ((s.readMemValue .nat context_lengths (s.pids 0) - 1) / block_size) *
              stride_btb) * stride_kcb +
        s.pids 1 * stride_kch + SPLIT_X * stride_kcsplit_x +
        ((s.readMemValue .nat context_lengths (s.pids 0) - 1) % block_size) *
          stride_kcs + j.val < bounds KCache)
    (hwv : ∀ j : Fin KCACHE_X, SPLIT_X * KCACHE_X + j.val < HEAD_DIM →
      s.readMemValue .nat BLOCK_TABLES
          (s.pids 0 * stride_bts +
            ((s.readMemValue .nat context_lengths (s.pids 0) - 1) / block_size) *
              stride_btb) * stride_vcb +
        s.pids 1 * stride_vch +
        ((s.readMemValue .nat context_lengths (s.pids 0) - 1) % block_size) *
          stride_vcs + (SPLIT_X * KCACHE_X + j.val) * stride_vcd
          < bounds VCache) :
    Kernel.TraceSafe bounds
      ((copy_to_kvcache_seqlen1_xblock K V KCache VCache BLOCK_TABLES
        context_lengths SPLIT_X stride_kt stride_kh stride_kd stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_vcb stride_vch
      stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM KCACHE_X).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [copy_to_kvcache_seqlen1_xblock, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
    BlockState.readMemValue, hb₁, hb₂]
  exact ⟨hb₂, fun a ha => hrk a ha, fun a ha => hrv a ha,
    fun a ha => hwk a ha, fun a ha => hwv a ha⟩

/-- **The region-model chained-metadata masked triple** — termination, the
`WriteInj`-gated active-lane values of both cache stores, and frame off the
two active store windows, from any launch state whose slot cells hold the
pinned raw scalars `m₁` (the context length) / `m₂` (the block id, sitting at
the `m₁`-dependent block-table cell) and whose active K/V windows hold
`xs`/`ys`. This is the `hrun` obligation of the `⊨` headline. `KCache ≠
VCache` lets the K readback see through the later V store. -/
theorem copy_to_kvcache_seqlen1_xblock_region_run
    (K V KCache VCache BLOCK_TABLES context_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd
      stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat)
    (hKV : KCache ≠ VCache)
    (s₀ : BlockState) (m₁ m₂ : Nat) (xs ys : Fin KCACHE_X → ℝ)
    (hm₁ : s₀.readMemValue .nat context_lengths (s₀.pids 0) = m₁)
    (hm₂ : s₀.readMemValue .nat BLOCK_TABLES
      (s₀.pids 0 * stride_bts + ((m₁ - 1) / block_size) * stride_btb) = m₂)
    (hx : ∀ j : Fin KCACHE_X, SPLIT_X * KCACHE_X + j.val < HEAD_DIM →
      s₀.readMem K (s₀.pids 0 * stride_kt + s₀.pids 1 * stride_kh +
        (SPLIT_X * KCACHE_X + j.val) * stride_kd) = xs j)
    (hy : ∀ j : Fin KCACHE_X, SPLIT_X * KCACHE_X + j.val < HEAD_DIM →
      s₀.readMem V (s₀.pids 0 * stride_vt + s₀.pids 1 * stride_vh +
        (SPLIT_X * KCACHE_X + j.val) * stride_vd) = ys j) :
    ∃ s1, exec ((copy_to_kvcache_seqlen1_xblock K V KCache VCache BLOCK_TABLES
        context_lengths SPLIT_X stride_kt stride_kh stride_kd stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_vcb stride_vch
      stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM KCACHE_X).toAlgKernel) s₀ = some s1
      ∧ ((∀ j k : Fin KCACHE_X,
            SPLIT_X * KCACHE_X + j.val < HEAD_DIM →
            SPLIT_X * KCACHE_X + k.val < HEAD_DIM →
            m₂ * stride_kcb + s₀.pids 1 * stride_kch +
                SPLIT_X * stride_kcsplit_x +
                ((m₁ - 1) % block_size) * stride_kcs + j.val
              = m₂ * stride_kcb + s₀.pids 1 * stride_kch +
                SPLIT_X * stride_kcsplit_x +
                ((m₁ - 1) % block_size) * stride_kcs + k.val → j = k) →
          ∀ j : Fin KCACHE_X, SPLIT_X * KCACHE_X + j.val < HEAD_DIM →
            s1.readMem KCache
              (m₂ * stride_kcb + s₀.pids 1 * stride_kch +
                SPLIT_X * stride_kcsplit_x +
                ((m₁ - 1) % block_size) * stride_kcs + j.val) = xs j)
      ∧ ((∀ j k : Fin KCACHE_X,
            SPLIT_X * KCACHE_X + j.val < HEAD_DIM →
            SPLIT_X * KCACHE_X + k.val < HEAD_DIM →
            m₂ * stride_vcb + s₀.pids 1 * stride_vch +
                ((m₁ - 1) % block_size) * stride_vcs +
                (SPLIT_X * KCACHE_X + j.val) * stride_vcd
              = m₂ * stride_vcb + s₀.pids 1 * stride_vch +
                ((m₁ - 1) % block_size) * stride_vcs +
                (SPLIT_X * KCACHE_X + k.val) * stride_vcd → j = k) →
          ∀ j : Fin KCACHE_X, SPLIT_X * KCACHE_X + j.val < HEAD_DIM →
            s1.readMem VCache
              (m₂ * stride_vcb + s₀.pids 1 * stride_vch +
                ((m₁ - 1) % block_size) * stride_vcs +
                (SPLIT_X * KCACHE_X + j.val) * stride_vcd) = ys j)
      ∧ (∀ r o,
          (r ≠ KCache ∨ ∀ j : Fin KCACHE_X,
            SPLIT_X * KCACHE_X + j.val < HEAD_DIM →
              o ≠ m₂ * stride_kcb + s₀.pids 1 * stride_kch +
                SPLIT_X * stride_kcsplit_x +
                ((m₁ - 1) % block_size) * stride_kcs + j.val) →
          (r ≠ VCache ∨ ∀ j : Fin KCACHE_X,
            SPLIT_X * KCACHE_X + j.val < HEAD_DIM →
              o ≠ m₂ * stride_vcb + s₀.pids 1 * stride_vch +
                ((m₁ - 1) % block_size) * stride_vcs +
                (SPLIT_X * KCACHE_X + j.val) * stride_vcd) →
          s1.mem r o = s₀.mem r o) := by
  subst hm₂
  subst hm₁
  obtain ⟨s1, hs1⟩ : ∃ s1,
      exec ((copy_to_kvcache_seqlen1_xblock K V KCache VCache BLOCK_TABLES
        context_lengths SPLIT_X stride_kt stride_kh stride_kd stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_vcb stride_vch
      stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM KCACHE_X).toAlgKernel) s₀ = some s1 := by
    simp [exec, copy_to_kvcache_seqlen1_xblock, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
      Tile.bop, Tile.ptrAdd, Tile.uop,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
      BlockState.readMemValue]
  have hs1' := hs1
  simp [exec, copy_to_kvcache_seqlen1_xblock, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
    Tile.bop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
    BlockState.readMemValue] at hs1'
  refine ⟨s1, hs1, ?_, ?_, ?_⟩
  · -- K-cache value leg (`WriteInj₁`-gated readback).
    intro hKInj j hj
    rw [← hs1']
    simp only [BlockState.readMemValue]
    refine ((BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      VCache _ _ _ _ _ _ _ hKV).trans ?_)
    refine ((BlockState.scatter_readback_prop_masked_nd_of_true
      (region := KCache) (shape := [KCACHE_X]) _ _ _ _
      (j, PUnit.unit) ?_ ?_).trans ?_)
    · simpa using hj
    · rintro ⟨a, u⟩ ha heq
      cases u
      have hval : a = j := hKInj a j (by simpa using ha) hj (by simpa using heq)
      subst hval
      rfl
    · simpa [hj] using hx j hj
  · -- V-cache value leg (`WriteInj₂`-gated readback).
    intro hVInj j hj
    rw [← hs1']
    simp only [BlockState.readMemValue]
    refine ((BlockState.scatter_readback_prop_masked_nd_of_true
      (region := VCache) (shape := [KCACHE_X]) _ _ _ _
      (j, PUnit.unit) ?_ ?_).trans ?_)
    · simpa using hj
    · rintro ⟨a, u⟩ ha heq
      cases u
      have hval : a = j := hVInj a j (by simpa using ha) hj (by simpa using heq)
      subst hval
      rfl
    · simpa [hj] using hy j hj
  · -- frame off the two active store windows.
    intro r o hc1 hc2
    rw [← hs1']
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
      (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl)
    · -- the outer (V-cache) store misses `(r, o)`
      rintro k - hPk ⟨hreg, hoff⟩
      rcases hc2 with hne | hno
      · exact hne hreg.symm
      · exact hno k.1 (by simpa using hPk)
          (by simp only [BlockState.readMemValue]; exact hoff.symm)
    · -- the inner (K-cache) store misses `(r, o)`
      rintro k - hPk ⟨hreg, hoff⟩
      rcases hc1 with hne | hno
      · exact hne hreg.symm
      · exact hno k.1 (by simpa using hPk)
          (by simp only [BlockState.readMemValue]; exact hoff.symm)

/-- `copy_to_kvcache_seqlen1_xblock`'s chained-metadata **IO signature** — the
whole kernel-specific audit surface of the `⊨` headline
(`ChainMetaMasked2DKernelIO₂ₓ₂`, the chained-metadata genre this kernel was
designed from):

* `mbuf1 = context_lengths` — slot 1: program `(cur_seq_idx, cur_kv_head_idx)
  = (pid₀, pid₁)` reads cell `pid₀` (`mwin1`), yielding the **raw** context
  length `m₁`;
* `mbuf2 = BLOCK_TABLES` — slot 2, **chained**: its cell
  `pid₀·stride_bts + ((m₁ − 1) / block_size)·stride_btb` (`mwin2`) eats
  slot 1's loaded value (`- 1` is the DSL's Nat-truncated subtraction),
  yielding the raw block id `m₂ = block_id`;
* `in1 = K → out1 = KCache`, `in2 = V → out2 = VCache` — the two data
  channels, `B = KCACHE_X` lanes at x-partition `SPLIT_X`;
* `read1`/`read2` — lane `j` reads
  `pid₀·stride_*t + pid₁·stride_*h + (SPLIT_X·KCACHE_X + j)·stride_*d`;
* `write1` — the K-cache cell `m₂·stride_kcb + pid₁·stride_kch +
  SPLIT_X·stride_kcsplit_x + ((m₁ − 1) % block_size)·stride_kcs + j`;
* `write2` — the V-cache cell `m₂·stride_vcb + pid₁·stride_vch +
  ((m₁ − 1) % block_size)·stride_vcs + (SPLIT_X·KCACHE_X + j)·stride_vcd`;
* `mask1 = mask2` — the active lanes `SPLIT_X·KCACHE_X + j < HEAD_DIM`
  (`writeMask`s default to the masks).

The slot cells, windows, and masks are declared, not parsed from the kernel;
the headline **proves** the kernel's actual chained slot loads, addressing,
and masking match them. Buffer sizes are not signature content: the headline
quantifies over every allocation whose extents cover the declared cells. -/
def kvCacheCopyIO
    (K V KCache VCache BLOCK_TABLES context_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd
      stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat) : ChainMetaMasked2DKernelIO₂ₓ₂ where
  kernel := copy_to_kvcache_seqlen1_xblock K V KCache VCache BLOCK_TABLES
    context_lengths SPLIT_X stride_kt stride_kh stride_kd stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_vcb stride_vch
      stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM KCACHE_X
  mbuf1 := context_lengths
  mbuf2 := BLOCK_TABLES
  in1 := K
  in2 := V
  out1 := KCache
  out2 := VCache
  B := KCACHE_X
  mwin1 := fun pid₀ _ => pid₀
  mwin2 := fun pid₀ _ m₁ =>
    pid₀ * stride_bts + ((m₁ - 1) / block_size) * stride_btb
  read1 := fun pid₀ pid₁ _ _ j =>
    pid₀ * stride_kt + pid₁ * stride_kh +
      (SPLIT_X * KCACHE_X + j.val) * stride_kd
  read2 := fun pid₀ pid₁ _ _ j =>
    pid₀ * stride_vt + pid₁ * stride_vh +
      (SPLIT_X * KCACHE_X + j.val) * stride_vd
  mask1 := fun _ _ _ _ j => SPLIT_X * KCACHE_X + j.val < HEAD_DIM
  mask2 := fun _ _ _ _ j => SPLIT_X * KCACHE_X + j.val < HEAD_DIM
  write1 := fun _ pid₁ m₁ m₂ j =>
    m₂ * stride_kcb + pid₁ * stride_kch + SPLIT_X * stride_kcsplit_x +
      ((m₁ - 1) % block_size) * stride_kcs + j.val
  write2 := fun _ pid₁ m₁ m₂ j =>
    m₂ * stride_vcb + pid₁ * stride_vch +
      ((m₁ - 1) % block_size) * stride_vcs +
      (SPLIT_X * KCACHE_X + j.val) * stride_vcd

open scoped VeriTile.Triton.ChainMetaMasked2DKernelIO₂ₓ₂ in
/-- **The headline**: the per-split KV-cache copy surface implements the pure
masked copy `(xs, ys)` on its chained-metadata IO signature — for every
disjoint flat placement of the six buffers, every program
`(cur_seq_idx, cur_kv_head_idx)` whose declared cells/lanes are in bounds,
and every launch state pinning the **raw** loaded scalars — `m₁` at the
`context_lengths` slot cell and `m₂` at the **`m₁`-dependent** block-table
cell (the chain) — and the active K/V windows to `xs`/`ys`, the translated
pointer kernel terminates, every active (`SPLIT_X·KCACHE_X + j < HEAD_DIM`)
K-cache window lane holds `xs j` and V-cache window lane `ys j` — each
readback gated by its per-pinned-context `WriteInj` no-aliasing side
condition on the block-id-dependent cache window — and every other memory
cell is unchanged. All `- 1` / `/` / `%` decode arithmetic is Nat-truncated,
exactly as the DSL executes it, and lives in the declared windows, not in the
pins. Side condition: `KCache ≠ VCache` (the V store must not clobber the K
output). Instantiating
`SPLIT_X` at every `split_x < HEAD_DIM / KCACHE_X` covers the full static
loop; the legacy cache layout is `SPLIT_X = 0`, `KCACHE_X = HEAD_DIM`,
`stride_kcsplit_x = 0`. Proof:
`ChainMetaMasked2DKernelIO₂ₓ₂.Implements.intro` assembles the region-model
chained triple with the flat-memory bridge side conditions. -/
specification kv_cache_copy_correctness
    (K V KCache VCache BLOCK_TABLES context_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd
      stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat)
    (hKV : KCache ≠ VCache) :
    kvCacheCopyIO K V KCache VCache BLOCK_TABLES context_lengths
        SPLIT_X stride_kt stride_kh stride_kd stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_vcb stride_vch
      stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM KCACHE_X
      ⊨ fun _ _ _ _ xs ys => (xs, ys) := by
  refine ChainMetaMasked2DKernelIO₂ₓ₂.Implements.intro _ ?_ ?_ ?_
  · exact copy_to_kvcache_seqlen1_xblock_flattenOk K V KCache VCache
      BLOCK_TABLES context_lengths SPLIT_X stride_kt stride_kh stride_kd stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_vcb stride_vch
      stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM KCACHE_X
  · intro bounds s m₁ m₂ hm₁ hm₂ hb₁ hb₂ hbr₁ hbr₂ hbw₁ hbw₂
    subst hm₂
    subst hm₁
    exact copy_to_kvcache_seqlen1_xblock_traceSafe K V KCache VCache
      BLOCK_TABLES context_lengths SPLIT_X stride_kt stride_kh stride_kd stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_vcb stride_vch
      stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM KCACHE_X bounds s hb₁ hb₂
      (fun j hj => hbr₁ j hj) (fun j hj => hbr₂ j hj)
      (fun j hj => hbw₁ j hj) (fun j hj => hbw₂ j hj)
  · intro s₀ m₁ m₂ xs ys hm₁ hm₂ hx hy
    exact copy_to_kvcache_seqlen1_xblock_region_run K V KCache VCache
      BLOCK_TABLES context_lengths SPLIT_X stride_kt stride_kh stride_kd stride_vt stride_vh stride_vd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_vcb stride_vch
      stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM KCACHE_X hKV s₀ m₁ m₂ xs ys hm₁ hm₂
      (fun j hj => hx j hj) (fun j hj => hy j hj)


end VeriTile.Bench.TritonBenchG.KvCacheCopy
