import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Proof-oriented split-x slice of `kcache_copy_triton.py`'s
`_copy_to_kcache_seqlen_n_kernel`.

The full kernel computes sequence-local block arithmetic with division and
modulo. This slice starts after that arithmetic has selected the cache slot:
load the block id from `BLOCK_TABLES`, load one contiguous K split-x block, and
store it into `KCache`. -/
def copy_to_kcache_split_x_block
    (K KCache BLOCK_TABLES : RegionName)
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
    $(LAST_BLOCK_IDX) * $(stride_btb), dtype=tl.uint64)
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
