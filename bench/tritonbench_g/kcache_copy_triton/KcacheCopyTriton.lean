import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `kcache_copy_triton.py`'s
`_copy_to_kcache_seqlen_n_kernel`.

The signed sequence-length path is represented with typed int regions for
`BLOCK_TABLES` and `seq_lengths`, so `cur_token_shift` is inferred as signed
when it is added to the signed `seq_lengths` load. -/
def copy_to_kcache_seqlen_n_kernel
    (K KCache : RegionName) (BLOCK_TABLES seq_lengths : Region .int)
    (stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs _stride_kcx stride_bts stride_btb block_size n_tokens
      _HEAD_DIM KCACHE_X : Nat) :
    ComputeKernel := triton {
  cur_token_idx = tl.program_id(0)
  cur_seq_idx = cur_token_idx // $(n_tokens)
  cur_token_shift = cur_token_idx - $(n_tokens) * (cur_seq_idx + $(1))
  cur_kv_head_idx = tl.program_id(1)
  split_x_idx = tl.program_id(2)
  past_kv_seq_len = tl.load(seq_lengths + cur_seq_idx) + cur_token_shift
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  block_table_ptr = BLOCK_TABLES + cur_seq_idx * $(stride_bts)
  block_id = tl.load(block_table_ptr + last_bt_block_idx * $(stride_btb))
  offset_last_block = past_kv_seq_len % $(block_size)
  offsets_dmodel = split_x_idx * $(KCACHE_X) + tl.arange(0, $(KCACHE_X))
  offsets_k = cur_token_idx * $(stride_kt) +
    cur_kv_head_idx * $(stride_kh) + offsets_dmodel * $(stride_kd)
  k = tl.load(K + offsets_k)
  offsets_kcache = block_id * $(stride_kcb) +
    cur_kv_head_idx * $(stride_kch) +
    split_x_idx * $(stride_kcsplit_x) +
    offset_last_block * $(stride_kcs) + tl.arange(0, $(KCACHE_X))
  tl.store(KCache + offsets_kcache, k)
}

/-- The complete signed `n_tokens` K-cache copy kernel lowers to the algorithm
layer.

This is the faithful surface for Python's `n_tokens > 1` branch: `BLOCK_TABLES`
and `seq_lengths` are typed integer regions, so `cur_token_shift` and
`past_kv_seq_len` stay on the signed path instead of being approximated by a
Nat-only rewrite. -/
theorem copy_to_kcache_seqlen_n_kernel_toAlgorithm_supported
    (K KCache : RegionName) (BLOCK_TABLES seq_lengths : Region .int)
    (stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs _stride_kcx stride_bts stride_btb block_size n_tokens
      _HEAD_DIM KCACHE_X : Nat) :
    ∃ alg, (copy_to_kcache_seqlen_n_kernel K KCache BLOCK_TABLES seq_lengths
      stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs _stride_kcx stride_bts stride_btb block_size n_tokens
      _HEAD_DIM KCACHE_X).toAlgorithm? = Except.ok alg := by
  simp [copy_to_kcache_seqlen_n_kernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Python test `n = 2`, legacy cache layout
`[num_blocks, num_kv_heads, block_size, head_dim]`.

For the checked test:
* `k.shape = (bsz*n, num_kv_heads, head_dim) = (4, 4, 64)`,
  so `k.stride = (256, 64, 1)`;
* `k_cache.shape = (10, 4, 16, 64)`,
  so `k_cache.stride = (4096, 1024, 64, 1)`;
* `block_tables.shape = (2, 10)`, stride `(10, 1)`;
* `block_size = 16`, `n_tokens = 2`, `HEAD_DIM = KCACHE_X = 64`.

The theorem pins those parameters on the complete signed kernel. -/
theorem copy_to_kcache_seqlen_n2_old_layout_python_toAlgorithm_supported
    (K KCache : RegionName) (BLOCK_TABLES seq_lengths : Region .int) :
    ∃ alg, (copy_to_kcache_seqlen_n_kernel K KCache BLOCK_TABLES seq_lengths
      256 64 1 4096 1024 0 64 1 10 1 16 2 64 64).toAlgorithm? =
        Except.ok alg := by
  exact copy_to_kcache_seqlen_n_kernel_toAlgorithm_supported K KCache
    BLOCK_TABLES seq_lengths 256 64 1 4096 1024 0 64 1 10 1 16 2 64 64

/-- Python test `n = 2`, new split-x cache layout
`[num_blocks, num_kv_heads, head_dim // x, block_size, x]`.

For the checked test:
* `k.shape = (4, 4, 64)`, so `k.stride = (256, 64, 1)`;
* `k_cache.shape = (10, 4, 8, 16, 8)`,
  so `k_cache.stride = (4096, 1024, 128, 8, 1)`;
* `block_tables.shape = (2, 10)`, stride `(10, 1)`;
* `block_size = 16`, `n_tokens = 2`, `HEAD_DIM = 64`, `KCACHE_X = 8`.

This keeps the signed `cur_token_shift`/`past_kv_seq_len` path intact. -/
theorem copy_to_kcache_seqlen_n2_new_layout_python_toAlgorithm_supported
    (K KCache : RegionName) (BLOCK_TABLES seq_lengths : Region .int) :
    ∃ alg, (copy_to_kcache_seqlen_n_kernel K KCache BLOCK_TABLES seq_lengths
      256 64 1 4096 1024 128 8 1 10 1 16 2 64 8).toAlgorithm? =
        Except.ok alg := by
  exact copy_to_kcache_seqlen_n_kernel_toAlgorithm_supported K KCache
    BLOCK_TABLES seq_lengths 256 64 1 4096 1024 128 8 1 10 1 16 2 64 8

def signedCurSeqIdx (cur_token_idx n_tokens : Nat) : Nat :=
  cur_token_idx / n_tokens

def signedCurTokenShift (cur_token_idx n_tokens : Nat) : Int :=
  (cur_token_idx : Int) -
    (n_tokens : Int) * ((signedCurSeqIdx cur_token_idx n_tokens : Int) + 1)

def signedPastKvSeqLen (seq_len : Int) (cur_token_idx n_tokens : Nat) : Int :=
  seq_len + signedCurTokenShift cur_token_idx n_tokens

/-- Sanity check for the checked Python `n = 2` path.

With `kv_lengths = [5, 10]`, Python's signed formula copies flat token ids
`0,1,2,3` to sequence positions `3,4,8,9`: for each sequence, the provided
length already includes the two new tokens, and `cur_token_shift` backs up to
the first token being copied. -/
theorem signedPastKvSeqLen_n2_python_test_positions :
    [ signedPastKvSeqLen 5 0 2
    , signedPastKvSeqLen 5 1 2
    , signedPastKvSeqLen 10 2 2
    , signedPastKvSeqLen 10 3 2 ] = [3, 4, 8, 9] := by
  decide

/-- Surface transcription of `kcache_copy_triton.py`'s
`_copy_to_kcache_seqlen_n_kernel` for the `n_tokens = 1` decode path.

For `n_tokens = 1`, Python treats `cur_token_idx` as the sequence id and uses
`seq_lengths[cur_seq_idx] - 1` as the position being copied. This surface keeps
that block-table lookup, offset-within-block computation, split-x K load, and
K-cache store. The `n_tokens > 1` path needs negative `cur_token_shift`
arithmetic before the copy and remains outside the current Nat-only pointer
surface. Python's unused `stride_kcx` and `HEAD_DIM` arguments are retained as
ignored parameters; `n_tokens` is fixed to the documented decode value `1` in
the proofs. -/
def copy_to_kcache_seqlen_n1_surface
    (K KCache : RegionName) (BLOCK_TABLES seq_lengths : Region .nat)
    (stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs _stride_kcx stride_bts stride_btb block_size _n_tokens
      _HEAD_DIM KCACHE_X : Nat) :
    ComputeKernel := triton {
  cur_token_idx = tl.program_id(0)
  cur_seq_idx = cur_token_idx
  cur_kv_head_idx = tl.program_id(1)
  split_x_idx = tl.program_id(2)
  past_kv_seq_len = tl.load(seq_lengths + cur_seq_idx) - $(1)
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    last_bt_block_idx * $(stride_btb))
  offset_last_block = past_kv_seq_len % $(block_size)
  offsets_dmodel = split_x_idx * $(KCACHE_X) + tl.arange(0, $(KCACHE_X))
  k = tl.load(K + cur_token_idx * $(stride_kt) +
    cur_kv_head_idx * $(stride_kh) + offsets_dmodel * $(stride_kd))
  tl.store(KCache + block_id * $(stride_kcb) +
    cur_kv_head_idx * $(stride_kch) +
    split_x_idx * $(stride_kcsplit_x) +
    offset_last_block * $(stride_kcs) + tl.arange(0, $(KCACHE_X)), k)
}

/-- The decode-path K-cache copy surface lowers to the algorithm layer,
including sequence-length lookup, block-table lookup, split-x load, and K-cache
store. -/
theorem copy_to_kcache_seqlen_n1_surface_toAlgorithm_supported
    (K KCache : RegionName) (BLOCK_TABLES seq_lengths : Region .nat)
    (stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs _stride_kcx stride_bts stride_btb block_size _n_tokens
      _HEAD_DIM KCACHE_X : Nat) :
    ∃ alg,
      (copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES seq_lengths
        stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
        stride_kcs _stride_kcx stride_bts stride_btb block_size _n_tokens
        _HEAD_DIM KCACHE_X).toAlgorithm? = Except.ok alg := by
  simp [copy_to_kcache_seqlen_n1_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented split-x slice of `kcache_copy_triton.py`'s
`_copy_to_kcache_seqlen_n_kernel`.

The full kernel computes sequence-local block arithmetic with division and
modulo. This slice starts after that arithmetic has selected the cache slot:
load the block id from `BLOCK_TABLES`, load one contiguous K split-x block, and
store it into `KCache`. -/
def copy_to_kcache_split_x_block
    (K KCache : RegionName) (BLOCK_TABLES : Region .nat)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK
      stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs stride_bts stride_btb KCACHE_X : Nat) :
    ComputeKernel := triton {
  cur_token_idx = tl.program_id(0)
  cur_seq_idx = tl.program_id(0)
  cur_kv_head_idx = tl.program_id(1)
  split_x_idx = tl.program_id(2)
  offsets_dmodel = split_x_idx * $(KCACHE_X) + tl.arange(0, $(KCACHE_X))
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    $(LAST_BLOCK_IDX) * $(stride_btb))
  k = tl.load(K + cur_token_idx * $(stride_kt) +
    cur_kv_head_idx * $(stride_kh) + offsets_dmodel * $(stride_kd))
  tl.store(KCache + block_id * $(stride_kcb) +
    cur_kv_head_idx * $(stride_kch) +
    split_x_idx * $(stride_kcsplit_x) +
    $(OFFSET_LAST_BLOCK) * $(stride_kcs) + tl.arange(0, $(KCACHE_X)), k)
}

def dimIndex (i : Fin KCACHE_X) : Nat :=
  i.val

def blockId (s : BlockState) (BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX stride_bts stride_btb : Nat) : Nat :=
  s.readMemValue .nat BLOCK_TABLES
    (s.pids 0 * stride_bts + LAST_BLOCK_IDX * stride_btb)

def kSourceOffset
    (s : BlockState) (stride_kt stride_kh stride_kd KCACHE_X : Nat)
    (i : Fin KCACHE_X) : Nat :=
  s.pids 0 * stride_kt + s.pids 1 * stride_kh +
    (s.pids 2 * KCACHE_X + dimIndex i) * stride_kd

def kCacheOffset
    (s : BlockState) (BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK stride_kcb stride_kch stride_kcsplit_x
      stride_kcs stride_bts stride_btb : Nat)
    (i : Fin KCACHE_X) : Nat :=
  blockId s BLOCK_TABLES LAST_BLOCK_IDX stride_bts stride_btb * stride_kcb +
    s.pids 1 * stride_kch + s.pids 2 * stride_kcsplit_x +
    OFFSET_LAST_BLOCK * stride_kcs + dimIndex i

def n1PastKvSeqLen (s : BlockState) (seq_lengths : RegionName) : Nat :=
  s.readMemValue .nat seq_lengths (s.pids 0) - 1

def n1LastBlockIdx (s : BlockState) (seq_lengths : RegionName)
    (block_size : Nat) : Nat :=
  n1PastKvSeqLen s seq_lengths / block_size

def n1OffsetLastBlock (s : BlockState) (seq_lengths : RegionName)
    (block_size : Nat) : Nat :=
  n1PastKvSeqLen s seq_lengths % block_size

def n1BlockId (s : BlockState) (BLOCK_TABLES seq_lengths : RegionName)
    (stride_bts stride_btb block_size : Nat) : Nat :=
  s.readMemValue .nat BLOCK_TABLES
    (s.pids 0 * stride_bts + n1LastBlockIdx s seq_lengths block_size * stride_btb)

def n1KCacheOffset
    (s : BlockState) (BLOCK_TABLES seq_lengths : RegionName)
    (stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_bts stride_btb block_size : Nat)
    (i : Fin KCACHE_X) : Nat :=
  n1BlockId s BLOCK_TABLES seq_lengths stride_bts stride_btb block_size * stride_kcb +
    s.pids 1 * stride_kch + s.pids 2 * stride_kcsplit_x +
    n1OffsetLastBlock s seq_lengths block_size * stride_kcs + dimIndex i

/-- Algorithm-layer correctness for the `n_tokens = 1` K-cache copy surface.

This theorem covers the Python decode path's dynamic `seq_lengths` read, block
index division, offset modulo, block-table lookup, and split-x K-cache store. -/
theorem copy_to_kcache_seqlen_n1_surface_correct
    (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs stride_bts stride_btb block_size KCACHE_X : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        n1KCacheOffset s BLOCK_TABLES seq_lengths stride_kcb stride_kch
          stride_kcsplit_x stride_kcs stride_bts stride_btb block_size i))
    (hExec : exec (copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES
        seq_lengths stride_kt stride_kh stride_kd stride_kcb stride_kch
        stride_kcsplit_x stride_kcs 0 stride_bts stride_btb block_size 1 0
        KCACHE_X)
        s = some s') :
    ∀ i : Fin KCACHE_X,
      s'.readMem KCache
          (n1KCacheOffset s BLOCK_TABLES seq_lengths stride_kcb stride_kch
            stride_kcsplit_x stride_kcs stride_bts stride_btb block_size i) =
        s.readMem K (kSourceOffset s stride_kt stride_kh stride_kd KCACHE_X i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [KCACHE_X] =>
        s.readMemValue .nat BLOCK_TABLES
            (s.pids 0 * stride_bts +
              ((s.readMemValue .nat seq_lengths (s.pids 0) - 1) / block_size) *
                stride_btb) * stride_kcb +
          s.pids 1 * stride_kch + s.pids 2 * stride_kcsplit_x +
          ((s.readMemValue .nat seq_lengths (s.pids 0) - 1) % block_size) *
            stride_kcs + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [n1KCacheOffset, n1BlockId, n1LastBlockIdx, n1OffsetLastBlock,
        n1PastKvSeqLen, blockId, dimIndex, BlockState.readMemValue] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hX : 0 < KCACHE_X
  · simp [exec, copy_to_kcache_seqlen_n1_surface, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          IntegralDType.floorDiv, IntegralDType.mod, BlockState.readMemValue, hX]
        at hExec
    rw [← hExec]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := KCache)
        (shape := [KCACHE_X])
        (s := (((((((((s.setReg "cur_token_idx" TileDType.nat []
              (Tile.scalar (s.pids 0)))
          |>.setReg "cur_seq_idx" TileDType.nat [] (Tile.scalar (s.pids 0)))
          |>.setReg "cur_kv_head_idx" TileDType.nat [] (Tile.scalar (s.pids 1)))
          |>.setReg "split_x_idx" TileDType.nat [] (Tile.scalar (s.pids 2)))
          |>.setReg "past_kv_seq_len" TileDType.nat []
            (Tile.scalar (s.readMemValue .nat seq_lengths (s.pids 0) - 1)))
          |>.setReg "last_bt_block_idx" TileDType.nat []
            (Tile.scalar ((s.readMemValue .nat seq_lengths (s.pids 0) - 1) /
              block_size)))
          |>.setReg "block_id" TileDType.nat []
            (Tile.scalar (s.readMemValue .nat BLOCK_TABLES
              (s.pids 0 * stride_bts +
                ((s.readMemValue .nat seq_lengths (s.pids 0) - 1) / block_size) *
                  stride_btb))))
          |>.setReg "offset_last_block" TileDType.nat []
            (Tile.scalar ((s.readMemValue .nat seq_lengths (s.pids 0) - 1) %
              block_size)))
          |>.setReg "offsets_dmodel" TileDType.nat [KCACHE_X]
            { data := fun i => s.pids 2 * KCACHE_X + i.1.val })
          |>.setReg "k" TileDType.real [KCACHE_X]
            { data := fun i =>
              some (s.readMem K
                (s.pids 0 * stride_kt + s.pids 1 * stride_kh +
                  (s.pids 2 * KCACHE_X + i.1.val) * stride_kd)) })
        (offsetFn := fun idx : TileIndex [KCACHE_X] =>
          s.readMemValue .nat BLOCK_TABLES
              (s.pids 0 * stride_bts +
                ((s.readMemValue .nat seq_lengths (s.pids 0) - 1) / block_size) *
                  stride_btb) * stride_kcb +
            s.pids 1 * stride_kch + s.pids 2 * stride_kcsplit_x +
            ((s.readMemValue .nat seq_lengths (s.pids 0) - 1) % block_size) *
              stride_kcs + idx.1.val)
        (valueFn := fun idx : TileIndex [KCACHE_X] =>
          WithBot.unbotD 0
            (some (s.readMem K
              (s.pids 0 * stride_kt + s.pids 1 * stride_kh +
                (s.pids 2 * KCACHE_X + idx.1.val) * stride_kd))))
        (P := fun _idx : TileIndex [KCACHE_X] => True)
        hRawInj (i, PUnit.unit))
    simp [BlockState.readMemValue] at hScatter
    simpa [n1KCacheOffset, n1BlockId, n1LastBlockIdx, n1OffsetLastBlock,
      n1PastKvSeqLen, kSourceOffset, dimIndex, BlockState.readMemValue] using hScatter
  · exact False.elim (hX (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the `n_tokens = 1` K-cache copy surface. -/
theorem copy_to_kcache_seqlen_n1_surface_compute_correct
    (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs stride_bts stride_btb block_size KCACHE_X : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        n1KCacheOffset s BLOCK_TABLES seq_lengths stride_kcb stride_kch
          stride_kcsplit_x stride_kcs stride_bts stride_btb block_size i)) :
    ComputeCorrect.Realizes
      (kernel := copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES
        seq_lengths stride_kt stride_kh stride_kd stride_kcb stride_kch
        stride_kcsplit_x stride_kcs 0 stride_bts stride_btb block_size 1 0
        KCACHE_X)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin KCACHE_X => True)
        (fun i => (KCache,
          n1KCacheOffset s BLOCK_TABLES seq_lengths stride_kcb stride_kch
            stride_kcsplit_x stride_kcs stride_bts stride_btb block_size i)))
      (expected := fun i =>
        s.readMem K (kSourceOffset s stride_kt stride_kh stride_kd KCACHE_X i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [copy_to_kcache_seqlen_n1_surface]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  exact copy_to_kcache_seqlen_n1_surface_correct K KCache BLOCK_TABLES
    seq_lengths stride_kt stride_kh stride_kd stride_kcb stride_kch
    stride_kcsplit_x stride_kcs stride_bts stride_btb block_size KCACHE_X
    s s' hOutInj hExec i

/-- Named `n_tokens = 1` K-cache writeback for the legacy Python layout
`[num_blocks, num_kv_heads, block_size, head_dim]`.

The Python wrapper sets `KCACHE_X = HEAD_DIM` and `stride_kcsplit_x = 0` for
this branch, so a single split-x program covers the full K head dimension. -/
theorem copy_to_kcache_seqlen_n1_old_layout_block_compute_correct
    (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcs
      stride_bts stride_btb block_size HEAD_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        n1KCacheOffset s BLOCK_TABLES seq_lengths stride_kcb stride_kch 0
          stride_kcs stride_bts stride_btb block_size i)) :
    ComputeCorrect.Realizes
      (kernel := copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES
        seq_lengths stride_kt stride_kh stride_kd stride_kcb stride_kch
        0 stride_kcs 0 stride_bts stride_btb block_size 1 HEAD_DIM HEAD_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin HEAD_DIM => True)
        (fun i => (KCache,
          n1KCacheOffset s BLOCK_TABLES seq_lengths stride_kcb stride_kch 0
            stride_kcs stride_bts stride_btb block_size i)))
      (expected := fun i =>
        s.readMem K (kSourceOffset s stride_kt stride_kh stride_kd HEAD_DIM i)) := by
  exact copy_to_kcache_seqlen_n1_surface_compute_correct K KCache BLOCK_TABLES
    seq_lengths stride_kt stride_kh stride_kd stride_kcb stride_kch 0
    stride_kcs stride_bts stride_btb block_size HEAD_DIM s hOutInj

/-- Named `n_tokens = 1` K-cache writeback for the new Python split-x layout
`[num_blocks, num_kv_heads, head_dim // x, block_size, x]`.

In the checked Python test this is instantiated with `HEAD_DIM = 64` and
`KCACHE_X = 8`, once for each `split_x_idx < 8`. -/
theorem copy_to_kcache_seqlen_n1_new_layout_xblock_compute_correct
    (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs stride_bts stride_btb block_size HEAD_DIM KCACHE_X : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        n1KCacheOffset s BLOCK_TABLES seq_lengths stride_kcb stride_kch
          stride_kcsplit_x stride_kcs stride_bts stride_btb block_size i)) :
    ComputeCorrect.Realizes
      (kernel := copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES
        seq_lengths stride_kt stride_kh stride_kd stride_kcb stride_kch
        stride_kcsplit_x stride_kcs 0 stride_bts stride_btb block_size 1
        HEAD_DIM KCACHE_X)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin KCACHE_X => True)
        (fun i => (KCache,
          n1KCacheOffset s BLOCK_TABLES seq_lengths stride_kcb stride_kch
            stride_kcsplit_x stride_kcs stride_bts stride_btb block_size i)))
      (expected := fun i =>
        s.readMem K (kSourceOffset s stride_kt stride_kh stride_kd KCACHE_X i)) := by
  exact copy_to_kcache_seqlen_n1_surface_compute_correct K KCache BLOCK_TABLES
    seq_lengths stride_kt stride_kh stride_kd stride_kcb stride_kch
    stride_kcsplit_x stride_kcs stride_bts stride_btb block_size KCACHE_X
    s hOutInj

/-- Algorithm-layer correctness for the K-cache split-x copy slice. -/
theorem copy_to_kcache_split_x_block_correct
    (K KCache BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK
      stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs stride_bts stride_btb KCACHE_X : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK
          stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
          stride_btb i))
    (hExec : exec (copy_to_kcache_split_x_block K KCache BLOCK_TABLES
        LAST_BLOCK_IDX OFFSET_LAST_BLOCK stride_kt stride_kh stride_kd
        stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
        stride_btb KCACHE_X) s = some s') :
    ∀ i : Fin KCACHE_X,
      s'.readMem KCache
          (kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK
            stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
            stride_btb i) =
        s.readMem K (kSourceOffset s stride_kt stride_kh stride_kd KCACHE_X i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [KCACHE_X] =>
        s.readMemValue .nat BLOCK_TABLES
            (s.pids 0 * stride_bts + LAST_BLOCK_IDX * stride_btb) * stride_kcb +
          s.pids 1 * stride_kch + s.pids 2 * stride_kcsplit_x +
          OFFSET_LAST_BLOCK * stride_kcs + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kCacheOffset, blockId, dimIndex, BlockState.readMemValue] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hX : 0 < KCACHE_X
  · simp [exec, copy_to_kcache_split_x_block, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, BlockState.readMemValue, hX]
        at hExec
    rw [← hExec]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := KCache)
        (shape := [KCACHE_X])
        (s := (s.setReg "cur_token_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
          |>.setReg "cur_seq_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
          |>.setReg "cur_kv_head_idx" TileDType.nat [] (Tile.scalar (s.pids 1))
          |>.setReg "split_x_idx" TileDType.nat [] (Tile.scalar (s.pids 2))
          |>.setReg "offsets_dmodel" TileDType.nat [KCACHE_X]
            (Tile.vec fun i => s.pids 2 * KCACHE_X + i.val)
          |>.setReg "block_id" TileDType.nat []
            (Tile.scalar (s.readMemValue .nat BLOCK_TABLES
              (s.pids 0 * stride_bts + LAST_BLOCK_IDX * stride_btb)))
          |>.setReg "k" TileDType.real [KCACHE_X]
            { data := fun i =>
              some (s.readMem K
                (s.pids 0 * stride_kt + s.pids 1 * stride_kh +
                  (s.pids 2 * KCACHE_X + i.1.val) * stride_kd)) }))
        (offsetFn := fun idx : TileIndex [KCACHE_X] =>
          s.readMemValue .nat BLOCK_TABLES
              (s.pids 0 * stride_bts + LAST_BLOCK_IDX * stride_btb) * stride_kcb +
            s.pids 1 * stride_kch + s.pids 2 * stride_kcsplit_x +
            OFFSET_LAST_BLOCK * stride_kcs + idx.1.val)
        (valueFn := fun idx : TileIndex [KCACHE_X] =>
          WithBot.unbotD 0
            (some (s.readMem K
              (s.pids 0 * stride_kt + s.pids 1 * stride_kh +
                (s.pids 2 * KCACHE_X + idx.1.val) * stride_kd))))
        (P := fun _idx : TileIndex [KCACHE_X] => True)
        hRawInj (i, PUnit.unit))
    simp [BlockState.readMemValue] at hScatter
    simpa [kSourceOffset, kCacheOffset, blockId, dimIndex,
      BlockState.readMemValue] using hScatter
  · exact False.elim (hX (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the K-cache split-x copy slice. -/
theorem copy_to_kcache_split_x_block_compute_correct
    (K KCache BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK
      stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs stride_bts stride_btb KCACHE_X : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK
          stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
          stride_btb i)) :
    ComputeCorrect.Realizes
      (kernel := copy_to_kcache_split_x_block K KCache BLOCK_TABLES
        LAST_BLOCK_IDX OFFSET_LAST_BLOCK stride_kt stride_kh stride_kd
        stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
        stride_btb KCACHE_X)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin KCACHE_X => True)
        (fun i => (KCache,
          kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK
            stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
            stride_btb i)))
      (expected := fun i =>
        s.readMem K (kSourceOffset s stride_kt stride_kh stride_kd KCACHE_X i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [copy_to_kcache_split_x_block]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  exact copy_to_kcache_split_x_block_correct K KCache BLOCK_TABLES
    LAST_BLOCK_IDX OFFSET_LAST_BLOCK stride_kt stride_kh stride_kd stride_kcb
    stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb KCACHE_X
    s s' hOutInj hExec i

end VeriTile.Bench.TritonBenchG.KcacheCopyTriton
