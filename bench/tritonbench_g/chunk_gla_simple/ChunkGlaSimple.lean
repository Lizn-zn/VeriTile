import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription of `chunk_gla_simple.py`'s
`chunk_simple_gla_fwd_kernel_o`.

The Triton `order` metadata on block pointers is scheduling-only; the DSL
accepts it at the surface and erases it into the same block-pointer AST. -/
def chunk_gla_simple_fwd_surface
    (Q K V H G O : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t T KSize VSize BT BK BV : Nat)
    (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(axis=0)
  i_t = tl.program_id(axis=1)
  i_bh = tl.program_id(axis=2)
  o_i = tl.arange(0, $(BT))
  m_s = o_i[:, None] >= o_i[None, :]
  b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
  b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
  for i_k in range($(0), tl.cdiv($(KSize), $(BK)), $(1)) {
    p_q = tl.make_block_ptr(base=Q + i_bh * $(s_k_h),
      shape=($(T), $(KSize)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
    p_k = tl.make_block_ptr(base=K + i_bh * $(s_k_h),
      shape=($(KSize), $(T)), strides=($(1), $(s_k_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_h = tl.make_block_ptr(base=H + i_bh * $(s_h_h) + i_t * $(KSize) * $(VSize),
      shape=($(KSize), $(VSize)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
    b_o += tl.dot(b_q, b_h, allow_tf32=false)
    b_s += tl.dot(b_q, b_k, allow_tf32=false)
  }
  p_g = tl.make_block_ptr(base=G + i_bh * $(T), shape=($(T)),
    strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
  b_g = tl.load(p_g, boundary_check=([0] : List Nat))
  b_o = b_o * tl.exp(b_g)[:, None]
  b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
  b_s = tl.where(m_s, b_s, 0.0)
  p_v = tl.make_block_ptr(base=V + i_bh * $(s_v_h),
    shape=($(T), $(VSize)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
  b_o = (b_o + tl.dot((b_s).to(V.dtype.element_ty), b_v, allow_tf32=false)) * $(scale)
  p_o = tl.make_block_ptr(base=O + i_bh * $(s_v_h),
    shape=($(T), $(VSize)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  tl.store(p_o, (b_o).to(O.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

/-- Proof-oriented final output-store slice of `chunk_gla_simple.py`'s
`chunk_simple_gla_fwd_kernel_o`.

The full kernel computes a GLA output tile from Q/K/V/H/G. This slice starts
from a precomputed `BO` tile and proves the final boundary-checked writeback
into `O`. -/
def chunk_gla_simple_output_store_slice
    (BO O : RegionName) (s_v_h s_v_t T V BT BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(axis=0)
  i_t = tl.program_id(axis=1)
  i_bh = tl.program_id(axis=2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_t[:, None] < $(T)) & (offs_v[None, :] < $(V))
  b_o = tl.load(BO + i_bh * $(s_v_h) + offs_t[:, None] * $(s_v_t) +
      offs_v[None, :], mask=mask, other=0.0)
  tl.store(O + i_bh * $(s_v_h) + offs_t[:, None] * $(s_v_t) +
      offs_v[None, :], b_o, mask=mask)
}

def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 1 * BT + i.val

def vIndex (s : BlockState) (BV : Nat) (j : Fin BV) : Nat :=
  s.pids 0 * BV + j.val

def active (s : BlockState) (T V BT BV : Nat) (idx : TileIndex [BT, BV]) : Prop :=
  tIndex s BT idx.1 < T ∧ vIndex s BV idx.2.1 < V

instance activeDecidable (s : BlockState) (T V BT BV : Nat)
    (idx : TileIndex [BT, BV]) : Decidable (active s T V BT BV idx) := by
  unfold active
  infer_instance

def tileOffset (s : BlockState) (s_v_h s_v_t BT BV : Nat)
    (idx : TileIndex [BT, BV]) : Nat :=
  s.pids 2 * s_v_h + tIndex s BT idx.1 * s_v_t + vIndex s BV idx.2.1

noncomputable def storeValue (s : BlockState) (BO : RegionName)
    (s_v_h s_v_t T V BT BV : Nat) (idx : TileIndex [BT, BV]) : ℝ :=
  WithBot.unbotD 0
    (if active s T V BT BV idx then
      some (s.readMem BO (tileOffset s s_v_h s_v_t BT BV idx))
    else some (0.0 : ℝ))

theorem chunk_gla_simple_output_store_slice_correct
    (BO O : RegionName) (s_v_h s_v_t T V BT BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BV] => tileOffset s s_v_h s_v_t BT BV idx)) :
    ∀ idx : TileIndex [BT, BV],
      let outAddr := tileOffset s s_v_h s_v_t BT BV idx
      (exec (chunk_gla_simple_output_store_slice BO O s_v_h s_v_t T V BT BV)
          s).map (·.readMem O outAddr)
        = some (if active s T V BT BV idx then
            storeValue s BO s_v_h s_v_t T V BT BV idx
          else s.readMem O outAddr) := by
  intro idx
  simp [exec, chunk_gla_simple_output_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        tIndex, vIndex, active, tileOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BT, BV] → Nat :=
    fun idx => s.pids 2 * s_v_h + (s.pids 1 * BT + idx.1.val) * s_v_t +
      (s.pids 0 * BV + idx.2.1.val)
  let valueFn : TileIndex [BT, BV] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 1 * BT + idx.1.val < T ∧
          s.pids 0 * BV + idx.2.1.val < V then
        some (s.readMem BO (offsetFn idx))
      else some (0.0 : ℝ))
  let P : TileIndex [BT, BV] → Prop :=
    fun idx => s.pids 1 * BT + idx.1.val < T ∧
      s.pids 0 * BV + idx.2.1.val < V
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, tileOffset, tIndex, vIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem O (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BT, BV])).readMem O (offsetFn idx) =
    if P idx then storeValue s BO s_v_h s_v_t T V BT BV idx
    else s.readMem O (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 1 * BT + idx.1.val < T ∧ s.pids 0 * BV + idx.2.1.val < V
  · rfl
  · rfl

theorem chunk_gla_simple_output_store_slice_compute_correct
    (BO O : RegionName) (s_v_h s_v_t T V BT BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BV] => tileOffset s s_v_h s_v_t BT BV idx)) :
    ComputeCorrect.Realizes
      (kernel := chunk_gla_simple_output_store_slice BO O s_v_h s_v_t T V BT BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BV] => active s T V BT BV idx)
        (fun idx : TileIndex [BT, BV] => (O, tileOffset s s_v_h s_v_t BT BV idx)))
      (expected := fun idx : TileIndex [BT, BV] =>
        storeValue s BO s_v_h s_v_t T V BT BV idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gla_simple_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_gla_simple_output_store_slice_correct BO O s_v_h s_v_t T V BT BV
    s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.ChunkGlaSimple
