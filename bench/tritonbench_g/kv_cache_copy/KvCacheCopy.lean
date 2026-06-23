import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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
kv_cache_copy_python_case1_all_outputs_summary         ← TOP THEOREM (old layout)
kv_cache_copy_python_case2_all_outputs_summary         ← TOP THEOREM (split-x layout)
  ├─ kv_cache_copy_python_case{1,2}_surface_toAlgorithm_supported   surface lowers
  │    └─ copy_to_kvcache_seqlen1_kernel_toAlgorithm_supported
  ├─ K-cache scatter ComputeCorrect
  │    ├─ copy_to_kcache_seqlen1_old_layout_block_compute_correct   (case 1)
  │    │    └─ copy_to_kcache_seqlen1_xblock_compute_correct → _xblock_correct
  │    └─ copy_to_kcache_seqlen1_new_layout_xblock_compute_correct  (case 2, ∀ split_x)
  ├─ V-cache scatter ComputeCorrect
  │    └─ copy_to_vcache_seqlen1_dblock_compute_correct → _dblock_correct
  └─ offset-injectivity lemmas
       kv_cache_copy_python_{old,new}_kcache_offset_injective,
       kv_cache_copy_python_vcache_offset_injective
```
(`kv_cache_copy_python_case{1,2}_output_summary` are abbrev aliases of the
two top theorems.) Single-x-block lemmas `copy_to_kcache_one_xblock_*` /
`copy_to_vcache_one_dblock_*` underlie the seqlen1 versions.

## Modeling boundary

Arithmetic/values are over `ℝ` (not bit-accurate IEEE float); dtype casts are
erased (post-erasure all dtypes unify to `ℝ`). The destination block id and
in-block slot are data-dependent: `block_id` is gathered from `BLOCK_TABLES`
and `offsets_in_last_block` from `context_lengths[cur_seq_idx] - 1`. The
top theorems are stated at the **Python test shapes** (`HEAD_DIM = 64`,
`KCACHE_X ∈ {64, 8}`, block_size 16, etc.), with explicit offset-injectivity
lemmas (no two cells of one program alias) proved by `omega` for those shapes.
The split-x case quantifies over every `split_x : Fin (HEAD_DIM // KCACHE_X)`.
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

/-- Surface transcription of the K-cache store in `kv_cache_copy.py`'s
`_copy_to_kvcache_seqlen1_kernel` for one `split_x` partition.

The Python kernel iterates `split_x in tl.static_range(HEAD_DIM // KCACHE_X)`.
This surface fixes that partition as `SPLIT_X`, loads the corresponding K
dimension block, and stores it into either the old layout (`SPLIT_X=0`,
`KCACHE_X=HEAD_DIM`) or the new split layout. -/
def copy_to_kcache_one_xblock
    (K KCache : RegionName) (BLOCK_TABLES : Region .nat)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X
      stride_kt stride_kh stride_kd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_bts stride_btb HEAD_DIM KCACHE_X : Nat) :
    ComputeKernel := triton {
  cur_seq_idx = tl.program_id(0)
  cur_kv_head_idx = tl.program_id(1)
  range_x = tl.arange(0, $(KCACHE_X))
  offsets_dmodel_x_partition = $(SPLIT_X) * $(KCACHE_X) + range_x
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    $(LAST_BLOCK_IDX) * $(stride_btb))
  k = tl.load(K + cur_seq_idx * $(stride_kt) +
      cur_kv_head_idx * $(stride_kh) +
      offsets_dmodel_x_partition * $(stride_kd),
    mask=offsets_dmodel_x_partition < $(HEAD_DIM), other=0.0)
  tl.store(KCache + block_id * $(stride_kcb) +
      cur_kv_head_idx * $(stride_kch) +
      $(SPLIT_X) * $(stride_kcsplit_x) +
      $(OFFSET_LAST_BLOCK) * $(stride_kcs) + range_x,
    k, mask=offsets_dmodel_x_partition < $(HEAD_DIM))
}

/-- Surface transcription of the K-cache store in `kv_cache_copy.py`'s
`_copy_to_kvcache_seqlen1_kernel`.

Unlike `copy_to_kcache_one_xblock`, this keeps the decode-path arithmetic from
the Python kernel: `context_lengths[cur_seq_idx] - 1`, block-table lookup,
last-block offset, one K load, and the K-cache store for one `split_x`
partition. -/
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

/-- Proof-oriented V-cache one-dimension-block slice of `kv_cache_copy.py`'s
`_copy_to_kvcache_seqlen1_kernel`.

This captures the V side after sequence/block arithmetic has selected the cache
slot: load the block id from `BLOCK_TABLES`, load a V head block, and store it
into `VCache`. -/
def copy_to_vcache_one_dblock
    (V VCache : RegionName) (BLOCK_TABLES : Region .nat)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK
      stride_vt stride_vh stride_vd stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb HEAD_DIM BLOCK_D : Nat) :
    ComputeKernel := triton {
  cur_seq_idx = tl.program_id(0)
  cur_kv_head_idx = tl.program_id(1)
  d = tl.arange(0, $(BLOCK_D))
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    $(LAST_BLOCK_IDX) * $(stride_btb))
  v = tl.load(V + cur_seq_idx * $(stride_vt) +
      cur_kv_head_idx * $(stride_vh) + d * $(stride_vd),
    mask=d < $(HEAD_DIM), other=0.0)
  tl.store(VCache + block_id * $(stride_vcb) +
      cur_kv_head_idx * $(stride_vch) + $(OFFSET_LAST_BLOCK) * $(stride_vcs) +
      d * $(stride_vcd),
    v, mask=d < $(HEAD_DIM))
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

def blockId (s : BlockState) (BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX stride_bts stride_btb : Nat) : Nat :=
  s.readMemValue .nat BLOCK_TABLES
    (s.pids 0 * stride_bts + LAST_BLOCK_IDX * stride_btb)

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

def kCacheOffset
    (s : BlockState) (BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X stride_kcb stride_kch
      stride_kcsplit_x stride_kcs stride_bts stride_btb KCACHE_X : Nat)
    (i : Fin KCACHE_X) : Nat :=
  blockId s BLOCK_TABLES LAST_BLOCK_IDX stride_bts stride_btb * stride_kcb +
    s.pids 1 * stride_kch + SPLIT_X * stride_kcsplit_x +
    OFFSET_LAST_BLOCK * stride_kcs + i.val

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

/-- Algorithm-layer correctness for the K-cache split-x copy slice. -/
theorem copy_to_kcache_one_xblock_correct
    (K KCache BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X
      stride_kt stride_kh stride_kd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_bts stride_btb HEAD_DIM KCACHE_X : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X
          stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
          stride_btb KCACHE_X i))
    (hExec : exec (copy_to_kcache_one_xblock K KCache BLOCK_TABLES
        LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X stride_kt stride_kh stride_kd
        stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb
        HEAD_DIM KCACHE_X) s = some s') :
    ∀ i : Fin KCACHE_X,
      s'.readMem KCache
          (kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X
            stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
            stride_btb KCACHE_X i) =
        if kActive SPLIT_X HEAD_DIM KCACHE_X i then
          s.readMem K (kSourceOffset s SPLIT_X stride_kt stride_kh stride_kd
            KCACHE_X i)
        else
          s.readMem KCache
            (kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X
              stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
              stride_btb KCACHE_X i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [KCACHE_X] =>
        s.readMemValue .nat BLOCK_TABLES
            (s.pids 0 * stride_bts + LAST_BLOCK_IDX * stride_btb) * stride_kcb +
          s.pids 1 * stride_kch + SPLIT_X * stride_kcsplit_x +
          OFFSET_LAST_BLOCK * stride_kcs + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kCacheOffset, blockId, BlockState.readMemValue] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBX : 0 < KCACHE_X
  · simp [exec, copy_to_kcache_one_xblock, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt,
          BlockState.readMemValue, hBX] at hExec
    rw [← hExec]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := KCache)
        (shape := [KCACHE_X])
        (s := (s.setReg "cur_seq_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
          |>.setReg "cur_kv_head_idx" TileDType.nat [] (Tile.scalar (s.pids 1))
          |>.setReg "range_x" TileDType.nat [KCACHE_X] (Tile.vec fun i => i.val)
          |>.setReg "offsets_dmodel_x_partition" TileDType.nat [KCACHE_X]
            { data := fun i => SPLIT_X * KCACHE_X + i.1.val }
          |>.setReg "block_id" TileDType.nat []
            (Tile.scalar (s.readMemValue .nat BLOCK_TABLES
              (s.pids 0 * stride_bts + LAST_BLOCK_IDX * stride_btb)))
          |>.setReg "k" TileDType.real [KCACHE_X]
            { data := fun i =>
              if SPLIT_X * KCACHE_X + i.1.val < HEAD_DIM then
                some (s.readMem K
                  (s.pids 0 * stride_kt + s.pids 1 * stride_kh +
                    (SPLIT_X * KCACHE_X + i.1.val) * stride_kd))
              else some (0.0 : ℝ) }))
        (offsetFn := fun idx : TileIndex [KCACHE_X] =>
          s.readMemValue .nat BLOCK_TABLES
              (s.pids 0 * stride_bts + LAST_BLOCK_IDX * stride_btb) * stride_kcb +
            s.pids 1 * stride_kch + SPLIT_X * stride_kcsplit_x +
            OFFSET_LAST_BLOCK * stride_kcs + idx.1.val)
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
    simp only [kActive, kSourceOffset, kCacheOffset, kDimIndex, blockId,
      BlockState.readMemValue]
    rw [hScatter]
    split <;> simp_all
  · exact False.elim (hBX (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the K-cache split-x copy slice. -/
theorem copy_to_kcache_one_xblock_compute_correct
    (K KCache BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X
      stride_kt stride_kh stride_kd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_bts stride_btb HEAD_DIM KCACHE_X : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X
          stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
          stride_btb KCACHE_X i)) :
    ComputeCorrect.Realizes
      (kernel := copy_to_kcache_one_xblock K KCache BLOCK_TABLES
        LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X stride_kt stride_kh stride_kd
        stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb
        HEAD_DIM KCACHE_X)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin KCACHE_X => kActive SPLIT_X HEAD_DIM KCACHE_X i)
        (fun i => (KCache,
          kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X
            stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
            stride_btb KCACHE_X i)))
      (expected := fun i =>
        s.readMem K (kSourceOffset s SPLIT_X stride_kt stride_kh stride_kd
          KCACHE_X i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [copy_to_kcache_one_xblock]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := copy_to_kcache_one_xblock_correct K KCache BLOCK_TABLES
    LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X stride_kt stride_kh stride_kd
    stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb
    HEAD_DIM KCACHE_X s s' hOutInj hExec i
  simpa [hActive] using h

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

/-- Compute-facing correctness for the K-cache seqlen=1 copy surface. -/
theorem copy_to_kcache_seqlen1_xblock_compute_correct
    (K KCache BLOCK_TABLES context_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        seqlen1KCacheOffset s BLOCK_TABLES context_lengths SPLIT_X stride_kcb
          stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb
          block_size KCACHE_X i)) :
    ComputeCorrect.Realizes
      (kernel := copy_to_kcache_seqlen1_xblock K KCache BLOCK_TABLES
        context_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb
        stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
        HEAD_DIM KCACHE_X)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin KCACHE_X => kActive SPLIT_X HEAD_DIM KCACHE_X i)
        (fun i => (KCache,
          seqlen1KCacheOffset s BLOCK_TABLES context_lengths SPLIT_X
            stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
            stride_btb block_size KCACHE_X i)))
      (expected := fun i =>
        s.readMem K (kSourceOffset s SPLIT_X stride_kt stride_kh stride_kd
          KCACHE_X i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [copy_to_kcache_seqlen1_xblock]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := copy_to_kcache_seqlen1_xblock_correct K KCache BLOCK_TABLES
    context_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
    stride_kcsplit_x stride_kcs stride_bts stride_btb block_size HEAD_DIM
    KCACHE_X s s' hOutInj hExec i
  simpa [hActive] using h

/-- Named K-cache seqlen=1 writeback for the legacy Python layout
`[num_blocks, num_kv_heads, block_size, head_dim]`.

In this branch Python sets `KCACHE_X = HEAD_DIM` and `stride_kcsplit_x = 0`,
so one x-block covers the complete K head for the selected sequence/head. -/
theorem copy_to_kcache_seqlen1_old_layout_block_compute_correct
    (K KCache BLOCK_TABLES context_lengths : RegionName)
    (stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcs
      stride_bts stride_btb block_size HEAD_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        seqlen1KCacheOffset s BLOCK_TABLES context_lengths 0 stride_kcb
          stride_kch 0 stride_kcs stride_bts stride_btb block_size HEAD_DIM i)) :
    ComputeCorrect.Realizes
      (kernel := copy_to_kcache_seqlen1_xblock K KCache BLOCK_TABLES
        context_lengths 0 stride_kt stride_kh stride_kd stride_kcb stride_kch
        0 stride_kcs stride_bts stride_btb block_size HEAD_DIM HEAD_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin HEAD_DIM => kActive 0 HEAD_DIM HEAD_DIM i)
        (fun i => (KCache,
          seqlen1KCacheOffset s BLOCK_TABLES context_lengths 0 stride_kcb
            stride_kch 0 stride_kcs stride_bts stride_btb block_size
            HEAD_DIM i)))
      (expected := fun i =>
        s.readMem K (kSourceOffset s 0 stride_kt stride_kh stride_kd
          HEAD_DIM i)) := by
  exact copy_to_kcache_seqlen1_xblock_compute_correct K KCache BLOCK_TABLES
    context_lengths 0 stride_kt stride_kh stride_kd stride_kcb stride_kch 0
    stride_kcs stride_bts stride_btb block_size HEAD_DIM HEAD_DIM s hOutInj

/-- Named K-cache seqlen=1 writeback for the new split-x Python layout
`[num_blocks, num_kv_heads, head_dim // x, block_size, x]`. -/
theorem copy_to_kcache_seqlen1_new_layout_xblock_compute_correct
    (K KCache BLOCK_TABLES context_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        seqlen1KCacheOffset s BLOCK_TABLES context_lengths SPLIT_X stride_kcb
          stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb
          block_size KCACHE_X i)) :
    ComputeCorrect.Realizes
      (kernel := copy_to_kcache_seqlen1_xblock K KCache BLOCK_TABLES
        context_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb
        stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
        HEAD_DIM KCACHE_X)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin KCACHE_X => kActive SPLIT_X HEAD_DIM KCACHE_X i)
        (fun i => (KCache,
          seqlen1KCacheOffset s BLOCK_TABLES context_lengths SPLIT_X
            stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
            stride_btb block_size KCACHE_X i)))
      (expected := fun i =>
        s.readMem K (kSourceOffset s SPLIT_X stride_kt stride_kh stride_kd
          KCACHE_X i)) := by
  exact copy_to_kcache_seqlen1_xblock_compute_correct K KCache BLOCK_TABLES
    context_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
    stride_kcsplit_x stride_kcs stride_bts stride_btb block_size HEAD_DIM
    KCACHE_X s hOutInj

/-- Named K-cache writeback for the legacy Python layout
`[num_blocks, num_kv_heads, block_size, head_dim]`.

In this mode Python sets `KCACHE_X = HEAD_DIM`, `SPLIT_X = 0`, and
`stride_kcsplit_x = 0`, so a single x-block covers the whole head dimension. -/
theorem copy_to_kcache_old_layout_block_compute_correct
    (K KCache BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK
      stride_kt stride_kh stride_kd
      stride_kcb stride_kch stride_kcs
      stride_bts stride_btb HEAD_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK 0
          stride_kcb stride_kch 0 stride_kcs stride_bts stride_btb HEAD_DIM i)) :
    ComputeCorrect.Realizes
      (kernel := copy_to_kcache_one_xblock K KCache BLOCK_TABLES
        LAST_BLOCK_IDX OFFSET_LAST_BLOCK 0 stride_kt stride_kh stride_kd
        stride_kcb stride_kch 0 stride_kcs stride_bts stride_btb
        HEAD_DIM HEAD_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin HEAD_DIM => kActive 0 HEAD_DIM HEAD_DIM i)
        (fun i => (KCache,
          kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK 0
            stride_kcb stride_kch 0 stride_kcs stride_bts stride_btb
            HEAD_DIM i)))
      (expected := fun i =>
        s.readMem K (kSourceOffset s 0 stride_kt stride_kh stride_kd
          HEAD_DIM i)) := by
  exact copy_to_kcache_one_xblock_compute_correct K KCache BLOCK_TABLES
    LAST_BLOCK_IDX OFFSET_LAST_BLOCK 0 stride_kt stride_kh stride_kd
    stride_kcb stride_kch 0 stride_kcs stride_bts stride_btb HEAD_DIM HEAD_DIM
    s hOutInj

/-- Named K-cache writeback for the new split-x Python layout
`[num_blocks, num_kv_heads, head_dim // x, block_size, x]`.

This exposes one `split_x` iteration of the Python static loop. Covering the
whole new-layout K-cache path instantiates this theorem for every
`split_x < HEAD_DIM / KCACHE_X`. -/
theorem copy_to_kcache_new_layout_xblock_compute_correct
    (K KCache BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X
      stride_kt stride_kh stride_kd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_bts stride_btb HEAD_DIM KCACHE_X : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X
          stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
          stride_btb KCACHE_X i)) :
    ComputeCorrect.Realizes
      (kernel := copy_to_kcache_one_xblock K KCache BLOCK_TABLES
        LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X stride_kt stride_kh stride_kd
        stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb
        HEAD_DIM KCACHE_X)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin KCACHE_X => kActive SPLIT_X HEAD_DIM KCACHE_X i)
        (fun i => (KCache,
          kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X
            stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
            stride_btb KCACHE_X i)))
      (expected := fun i =>
        s.readMem K (kSourceOffset s SPLIT_X stride_kt stride_kh stride_kd
          KCACHE_X i)) := by
  exact copy_to_kcache_one_xblock_compute_correct K KCache BLOCK_TABLES
    LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X stride_kt stride_kh stride_kd
    stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb
    HEAD_DIM KCACHE_X s hOutInj

def vSourceOffset
    (s : BlockState) (stride_vt stride_vh stride_vd : Nat)
    (i : Fin BLOCK_D) : Nat :=
  s.pids 0 * stride_vt + s.pids 1 * stride_vh + dimIndex i * stride_vd

def vCacheOffset
    (s : BlockState) (BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK stride_vcb stride_vch stride_vcs
      stride_vcd stride_bts stride_btb : Nat)
    (i : Fin BLOCK_D) : Nat :=
  blockId s BLOCK_TABLES LAST_BLOCK_IDX stride_bts stride_btb * stride_vcb +
    s.pids 1 * stride_vch + OFFSET_LAST_BLOCK * stride_vcs +
    dimIndex i * stride_vcd

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

/-- Compute-facing correctness for the V-cache seqlen=1 copy surface. -/
theorem copy_to_vcache_seqlen1_dblock_compute_correct
    (V VCache BLOCK_TABLES context_lengths : RegionName)
    (stride_vt stride_vh stride_vd stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM BLOCK_D : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_D =>
        seqlen1VCacheOffset s BLOCK_TABLES context_lengths stride_vcb stride_vch
          stride_vcs stride_vcd stride_bts stride_btb block_size i)) :
    ComputeCorrect.Realizes
      (kernel := copy_to_vcache_seqlen1_dblock V VCache BLOCK_TABLES
        context_lengths stride_vt stride_vh stride_vd stride_vcb stride_vch
        stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM BLOCK_D)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_D => active HEAD_DIM i)
        (fun i => (VCache,
          seqlen1VCacheOffset s BLOCK_TABLES context_lengths stride_vcb
            stride_vch stride_vcs stride_vcd stride_bts stride_btb block_size i)))
      (expected := fun i =>
        s.readMem V (vSourceOffset s stride_vt stride_vh stride_vd i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [copy_to_vcache_seqlen1_dblock]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := copy_to_vcache_seqlen1_dblock_correct V VCache BLOCK_TABLES
    context_lengths stride_vt stride_vh stride_vd stride_vcb stride_vch
    stride_vcs stride_vcd stride_bts stride_btb block_size HEAD_DIM BLOCK_D
    s s' hOutInj hExec i
  simpa [hActive] using h

/-- Algorithm-layer correctness for the V-cache copy slice. -/
theorem copy_to_vcache_one_dblock_correct
    (V VCache BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK
      stride_vt stride_vh stride_vd stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb HEAD_DIM BLOCK_D : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_D =>
        vCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK
          stride_vcb stride_vch stride_vcs stride_vcd stride_bts stride_btb i))
    (hExec : exec (copy_to_vcache_one_dblock V VCache BLOCK_TABLES
        LAST_BLOCK_IDX OFFSET_LAST_BLOCK stride_vt stride_vh stride_vd
        stride_vcb stride_vch stride_vcs stride_vcd stride_bts stride_btb
        HEAD_DIM BLOCK_D) s = some s') :
    ∀ i : Fin BLOCK_D,
      s'.readMem VCache
          (vCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK
            stride_vcb stride_vch stride_vcs stride_vcd stride_bts stride_btb i) =
        if active HEAD_DIM i then
          s.readMem V (vSourceOffset s stride_vt stride_vh stride_vd i)
        else
          s.readMem VCache
            (vCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK
              stride_vcb stride_vch stride_vcs stride_vcd stride_bts stride_btb i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_D] =>
        s.readMemValue .nat BLOCK_TABLES
            (s.pids 0 * stride_bts + LAST_BLOCK_IDX * stride_btb) * stride_vcb +
          s.pids 1 * stride_vch + OFFSET_LAST_BLOCK * stride_vcs +
          idx.1.val * stride_vcd) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [vCacheOffset, blockId, dimIndex, BlockState.readMemValue] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBD : 0 < BLOCK_D
  · simp [exec, copy_to_vcache_one_dblock, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt,
          BlockState.readMemValue, hBD] at hExec
    rw [← hExec]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := VCache)
        (shape := [BLOCK_D])
        (s := (s.setReg "cur_seq_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
          |>.setReg "cur_kv_head_idx" TileDType.nat [] (Tile.scalar (s.pids 1))
          |>.setReg "d" TileDType.nat [BLOCK_D] (Tile.vec fun i => i.val)
          |>.setReg "block_id" TileDType.nat []
            (Tile.scalar (s.readMemValue .nat BLOCK_TABLES
              (s.pids 0 * stride_bts + LAST_BLOCK_IDX * stride_btb)))
          |>.setReg "v" TileDType.real [BLOCK_D]
            { data := fun i =>
              if i.1.val < HEAD_DIM then
                some (s.readMem V
                  (s.pids 0 * stride_vt + s.pids 1 * stride_vh +
                    i.1.val * stride_vd))
              else some (0.0 : ℝ) }))
        (offsetFn := fun idx : TileIndex [BLOCK_D] =>
          s.readMemValue .nat BLOCK_TABLES
              (s.pids 0 * stride_bts + LAST_BLOCK_IDX * stride_btb) * stride_vcb +
            s.pids 1 * stride_vch + OFFSET_LAST_BLOCK * stride_vcs +
            idx.1.val * stride_vcd)
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
    simp only [active, vSourceOffset, vCacheOffset, blockId, dimIndex,
      BlockState.readMemValue]
    rw [hScatter]
    split <;> simp_all
  · exact False.elim (hBD (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the V-cache copy slice. -/
theorem copy_to_vcache_one_dblock_compute_correct
    (V VCache BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK
      stride_vt stride_vh stride_vd stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb HEAD_DIM BLOCK_D : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_D =>
        vCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK
          stride_vcb stride_vch stride_vcs stride_vcd stride_bts stride_btb i)) :
    ComputeCorrect.Realizes
      (kernel := copy_to_vcache_one_dblock V VCache BLOCK_TABLES
        LAST_BLOCK_IDX OFFSET_LAST_BLOCK stride_vt stride_vh stride_vd
        stride_vcb stride_vch stride_vcs stride_vcd stride_bts stride_btb
        HEAD_DIM BLOCK_D)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_D => active HEAD_DIM i)
        (fun i => (VCache,
          vCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK
            stride_vcb stride_vch stride_vcs stride_vcd stride_bts stride_btb i)))
      (expected := fun i =>
        s.readMem V (vSourceOffset s stride_vt stride_vh stride_vd i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [copy_to_vcache_one_dblock]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := copy_to_vcache_one_dblock_correct V VCache BLOCK_TABLES
    LAST_BLOCK_IDX OFFSET_LAST_BLOCK stride_vt stride_vh stride_vd stride_vcb
    stride_vch stride_vcs stride_vcd stride_bts stride_btb HEAD_DIM BLOCK_D
    s s' hOutInj hExec i
  simpa [hActive] using h

/-! ## Python test-shape wrappers -/

/-- Python case 1 full surface lowering for the legacy K-cache layout
`[num_blocks, num_kv_heads, block_size, head_dim]`. -/
theorem kv_cache_copy_python_case1_surface_toAlgorithm_supported
    (K V KCache VCache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat) :
    ∃ alg, (copy_to_kvcache_seqlen1_kernel K V KCache VCache BLOCK_TABLES
      context_lengths 256 64 1 256 64 1 4096 1024 0 64 1
      4096 1024 64 1 10 1 16 64 64).toAlgorithm? = Except.ok alg := by
  exact copy_to_kvcache_seqlen1_kernel_toAlgorithm_supported K V KCache
    VCache BLOCK_TABLES context_lengths 256 64 1 256 64 1 4096 1024
    0 64 1 4096 1024 64 1 10 1 16 64 64

/-- Python case 2 full surface lowering for the new split-x K-cache layout
`[num_blocks, num_kv_heads, head_dim // 8, block_size, 8]`. -/
theorem kv_cache_copy_python_case2_surface_toAlgorithm_supported
    (K V KCache VCache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat) :
    ∃ alg, (copy_to_kvcache_seqlen1_kernel K V KCache VCache BLOCK_TABLES
      context_lengths 256 64 1 256 64 1 4096 1024 128 8 1
      4096 1024 64 1 10 1 16 64 8).toAlgorithm? = Except.ok alg := by
  exact copy_to_kvcache_seqlen1_kernel_toAlgorithm_supported K V KCache
    VCache BLOCK_TABLES context_lengths 256 64 1 256 64 1 4096 1024
    128 8 1 4096 1024 64 1 10 1 16 64 8

theorem kv_cache_copy_python_old_kcache_offset_injective
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName) :
    Function.Injective
      (fun i : Fin 64 =>
        seqlen1KCacheOffset s BLOCK_TABLES context_lengths 0
          4096 1024 0 64 10 1 16 64 i) := by
  intro a b h
  simp [seqlen1KCacheOffset, seqlen1BlockId, seqlen1LastBlockIdx,
    seqlen1OffsetLastBlock, seqlen1PastKvSeqLen] at h
  exact Fin.ext (by omega)

theorem kv_cache_copy_python_new_kcache_offset_injective
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName) (split_x : Nat) :
    Function.Injective
      (fun i : Fin 8 =>
        seqlen1KCacheOffset s BLOCK_TABLES context_lengths split_x
          4096 1024 128 8 10 1 16 8 i) := by
  intro a b h
  simp [seqlen1KCacheOffset, seqlen1BlockId, seqlen1LastBlockIdx,
    seqlen1OffsetLastBlock, seqlen1PastKvSeqLen] at h
  exact Fin.ext (by omega)

theorem kv_cache_copy_python_vcache_offset_injective
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName) :
    Function.Injective
      (fun i : Fin 64 =>
        seqlen1VCacheOffset s BLOCK_TABLES context_lengths
          4096 1024 64 1 10 1 16 i) := by
  intro a b h
  simp [seqlen1VCacheOffset, seqlen1BlockId, seqlen1LastBlockIdx,
    seqlen1OffsetLastBlock, seqlen1PastKvSeqLen, dimIndex] at h
  exact Fin.ext (by omega)

/-- Public Python case 1 coverage summary: full old-layout K/V cache surface
lowers, and the K-cache plus V-cache writebacks realize the checked output
strides. -/
theorem kv_cache_copy_python_case1_all_outputs_summary
    (K V KCache VCache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat) (s : BlockState) :
    (∃ alg, (copy_to_kvcache_seqlen1_kernel K V KCache VCache BLOCK_TABLES
      context_lengths 256 64 1 256 64 1 4096 1024 0 64 1
      4096 1024 64 1 10 1 16 64 64).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := copy_to_kcache_seqlen1_xblock K KCache BLOCK_TABLES
        context_lengths 0 256 64 1 4096 1024 0 64 10 1 16 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => kActive 0 64 64 i)
        (fun i => (KCache,
          seqlen1KCacheOffset s BLOCK_TABLES context_lengths 0
            4096 1024 0 64 10 1 16 64 i)))
      (expected := fun i =>
        s.readMem K (kSourceOffset s 0 256 64 1 64 i))) ∧
    (ComputeCorrect.Realizes
      (kernel := copy_to_vcache_seqlen1_dblock V VCache BLOCK_TABLES
        context_lengths 256 64 1 4096 1024 64 1 10 1 16 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active 64 i)
        (fun i => (VCache,
          seqlen1VCacheOffset s BLOCK_TABLES context_lengths
            4096 1024 64 1 10 1 16 i)))
      (expected := fun i =>
        s.readMem V (vSourceOffset s 256 64 1 i))) := by
  constructor
  · exact kv_cache_copy_python_case1_surface_toAlgorithm_supported
      K V KCache VCache BLOCK_TABLES context_lengths
  constructor
  · exact copy_to_kcache_seqlen1_old_layout_block_compute_correct K KCache
      BLOCK_TABLES context_lengths 256 64 1 4096 1024 64 10 1 16 64 s
      (kv_cache_copy_python_old_kcache_offset_injective s BLOCK_TABLES
        context_lengths)
  · exact copy_to_vcache_seqlen1_dblock_compute_correct V VCache BLOCK_TABLES
      context_lengths 256 64 1 4096 1024 64 1 10 1 16 64 64 s
      (kv_cache_copy_python_vcache_offset_injective s BLOCK_TABLES
        context_lengths)

/-- Public Python case 2 coverage summary: full new-layout K/V cache surface
lowers, every `split_x : Fin 8` K-cache writeback realizes its x-block, and the
V-cache writeback realizes the checked output strides. -/
theorem kv_cache_copy_python_case2_all_outputs_summary
    (K V KCache VCache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat) (s : BlockState) :
    (∃ alg, (copy_to_kvcache_seqlen1_kernel K V KCache VCache BLOCK_TABLES
      context_lengths 256 64 1 256 64 1 4096 1024 128 8 1
      4096 1024 64 1 10 1 16 64 8).toAlgorithm? = Except.ok alg) ∧
    (∀ split_x : Fin 8,
      ComputeCorrect.Realizes
        (kernel := copy_to_kcache_seqlen1_xblock K KCache BLOCK_TABLES
          context_lengths split_x.val 256 64 1 4096 1024 128 8 10 1 16 64 8)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin 8 => kActive split_x.val 64 8 i)
          (fun i => (KCache,
            seqlen1KCacheOffset s BLOCK_TABLES context_lengths split_x.val
              4096 1024 128 8 10 1 16 8 i)))
        (expected := fun i =>
          s.readMem K (kSourceOffset s split_x.val 256 64 1 8 i))) ∧
    (ComputeCorrect.Realizes
      (kernel := copy_to_vcache_seqlen1_dblock V VCache BLOCK_TABLES
        context_lengths 256 64 1 4096 1024 64 1 10 1 16 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active 64 i)
        (fun i => (VCache,
          seqlen1VCacheOffset s BLOCK_TABLES context_lengths
            4096 1024 64 1 10 1 16 i)))
      (expected := fun i =>
        s.readMem V (vSourceOffset s 256 64 1 i))) := by
  constructor
  · exact kv_cache_copy_python_case2_surface_toAlgorithm_supported
      K V KCache VCache BLOCK_TABLES context_lengths
  constructor
  · intro split_x
    exact copy_to_kcache_seqlen1_new_layout_xblock_compute_correct K KCache
      BLOCK_TABLES context_lengths split_x.val 256 64 1 4096 1024 128 8
      10 1 16 64 8 s
      (kv_cache_copy_python_new_kcache_offset_injective s BLOCK_TABLES
        context_lengths split_x.val)
  · exact copy_to_vcache_seqlen1_dblock_compute_correct V VCache BLOCK_TABLES
      context_lengths 256 64 1 4096 1024 64 1 10 1 16 64 64 s
      (kv_cache_copy_python_vcache_offset_injective s BLOCK_TABLES
        context_lengths)

/-- `output_summary` alias for Python case 1, old K-cache layout. -/
abbrev kv_cache_copy_python_case1_output_summary
    (K V KCache VCache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat) (s : BlockState) :=
  kv_cache_copy_python_case1_all_outputs_summary K V KCache VCache
    BLOCK_TABLES context_lengths s

/-- `output_summary` alias for Python case 2, new split-x K-cache layout. -/
abbrev kv_cache_copy_python_case2_output_summary
    (K V KCache VCache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat) (s : BlockState) :=
  kv_cache_copy_python_case2_all_outputs_summary K V KCache VCache
    BLOCK_TABLES context_lengths s

end VeriTile.Bench.TritonBenchG.KvCacheCopy
