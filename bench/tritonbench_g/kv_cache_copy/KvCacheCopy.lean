import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.KvCacheCopy

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

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
