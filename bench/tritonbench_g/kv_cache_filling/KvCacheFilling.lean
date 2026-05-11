import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.KvCacheFilling

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Proof-oriented K-cache fill slice of `kv_cache_filling.py`'s
`_fill_kv_cache_kernel`.

The full kernel computes sequence/block positions and loops over cache block
slots. This slice starts after that arithmetic has selected `SIDX`, `BIDX`, and
`KV_BLOCK_IDX`: load a block offset from `BlockOffsets`, load a
`BLOCK_H × BLOCK_D` K tile from `KStates`, and store it into `KCaches` under the
original `num_heads/head_dim` mask. -/
def fill_k_cache_tile
    (KStates KCaches BlockOffsets : RegionName)
    (SIDX BIDX KV_BLOCK_IDX
      stride_kss stride_ksh stride_ksd
      stride_kcn stride_kcb stride_kch stride_kcd
      stride_boff num_heads head_dim BLOCK_H BLOCK_D : Nat) :
    ComputeKernel := triton {
  batch_id = tl.program_id(axis=0)
  h_off = tl.arange(0, $(BLOCK_H))
  d_off = tl.arange(0, $(BLOCK_D))
  block_off = tl.load(BlockOffsets + batch_id * $(stride_boff) + $(KV_BLOCK_IDX),
    dtype=tl.uint64)
  mask = (h_off[:, None] < $(num_heads)) & (d_off[None, :] < $(head_dim))
  k = tl.load(KStates + $(SIDX) * $(stride_kss) +
      h_off[:, None] * $(stride_ksh) + d_off[None, :] * $(stride_ksd),
    mask=mask, other=0.0)
  tl.store(KCaches + block_off * $(stride_kcn) + $(BIDX) * $(stride_kcb) +
      h_off[:, None] * $(stride_kch) + d_off[None, :] * $(stride_kcd),
    k, mask=mask)
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
  simp [exec, fill_k_cache_tile, stepStmts, stepStmt, evalOp,
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

end VeriTile.Bench.TritonBenchG.KvCacheFilling
