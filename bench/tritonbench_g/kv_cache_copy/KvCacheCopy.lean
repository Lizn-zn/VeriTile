import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.KvCacheCopy

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Surface transcription of the K-cache store in `kv_cache_copy.py`'s
`_copy_to_kvcache_seqlen1_kernel` for one `split_x` partition.

The Python kernel iterates `split_x in tl.static_range(HEAD_DIM // KCACHE_X)`.
This surface fixes that partition as `SPLIT_X`, loads the corresponding K
dimension block, and stores it into either the old layout (`SPLIT_X=0`,
`KCACHE_X=HEAD_DIM`) or the new split layout. -/
def copy_to_kcache_one_xblock
    (K KCache BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK SPLIT_X
      stride_kt stride_kh stride_kd
      stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_bts stride_btb HEAD_DIM KCACHE_X : Nat) :
    ComputeKernel := triton {
  cur_seq_idx = tl.program_id(axis=0)
  cur_kv_head_idx = tl.program_id(axis=1)
  range_x = tl.arange(0, $(KCACHE_X))
  offsets_dmodel_x_partition = $(SPLIT_X) * $(KCACHE_X) + range_x
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    $(LAST_BLOCK_IDX) * $(stride_btb), dtype=tl.uint64)
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

/-- Proof-oriented V-cache one-dimension-block slice of `kv_cache_copy.py`'s
`_copy_to_kvcache_seqlen1_kernel`.

This captures the V side after sequence/block arithmetic has selected the cache
slot: load the block id from `BLOCK_TABLES`, load a V head block, and store it
into `VCache`. -/
def copy_to_vcache_one_dblock
    (V VCache BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK
      stride_vt stride_vh stride_vd stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb HEAD_DIM BLOCK_D : Nat) :
    ComputeKernel := triton {
  cur_seq_idx = tl.program_id(0)
  cur_kv_head_idx = tl.program_id(1)
  d = tl.arange(0, $(BLOCK_D))
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    $(LAST_BLOCK_IDX) * $(stride_btb), dtype=tl.uint64)
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
    (V VCache BLOCK_TABLES context_lengths : RegionName)
    (stride_vt stride_vh stride_vd stride_vcb stride_vch stride_vcs stride_vcd
      stride_bts stride_btb block_size HEAD_DIM BLOCK_D : Nat) :
    ComputeKernel := triton {
  cur_seq_idx = tl.program_id(0)
  cur_kv_head_idx = tl.program_id(1)
  d = tl.arange(0, $(BLOCK_D))
  past_kv_seq_len = tl.load(context_lengths + cur_seq_idx, dtype=tl.uint64) - $(1)
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    last_bt_block_idx * $(stride_btb), dtype=tl.uint64)
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
  · simp [exec, copy_to_kcache_one_xblock, stepStmts, stepStmt, evalOp,
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
  · simp [exec, copy_to_vcache_seqlen1_dblock, stepStmts, stepStmt, evalOp,
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
  · simp [exec, copy_to_vcache_one_dblock, stepStmts, stepStmt, evalOp,
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

end VeriTile.Bench.TritonBenchG.KvCacheCopy
