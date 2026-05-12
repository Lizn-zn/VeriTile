import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription of `chunk_gated_attention.py`'s
`chunk_gated_abc_fwd_kernel_cum`.

This is the forward-preprocessing cumsum kernel used by `fwd_pre`: it builds the
lower-triangular accumulation mask, loads one `[BT, BS]` tile, computes the
chunk-local cumulative sum as a matrix product, and writes the result through a
boundary-checked block pointer. The Triton block-pointer `order` metadata is
scheduling-only; the DSL accepts it at the surface and erases it into the same
block-pointer AST. -/
def chunk_gated_attention_cum_surface
    (S O : RegionName) (s_s_h s_s_t s_s_d T SSize BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(axis=0)
  i_t = tl.program_id(axis=1)
  i_bh = tl.program_id(axis=2)
  o_i = tl.arange(0, $(BT))
  m_s_bool = o_i[:, None] >= o_i[None, :]
  m_s = tl.where(m_s_bool, 1.0, 0.0).to(tl.float32)
  p_s = tl.make_block_ptr(base=S + i_bh * $(s_s_h),
    shape=($(T), $(SSize)), strides=($(s_s_t), $(s_s_d)),
    offsets=(i_t * $(BT), i_s * $(BS)), block_shape=($(BT), $(BS)), order=(1, 0))
  p_o = tl.make_block_ptr(base=O + i_bh * $(s_s_h),
    shape=($(T), $(SSize)), strides=($(s_s_t), $(s_s_d)),
    offsets=(i_t * $(BT), i_s * $(BS)), block_shape=($(BT), $(BS)), order=(1, 0))
  b_s = tl.load(p_s, boundary_check=([0, 1] : List Nat)).to(tl.float32)
  b_o = tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

/-- Surface transcription of `chunk_gated_attention.py`'s
`chunk_gated_abc_fwd_kernel_h`.

The `GATEK`, `USE_INITIAL_STATE`, and `STORE_FINAL_STATE` constexpr branches
are preserved, including the local-tile dtype casts on gated `b_k`/`b_v` and
the block-pointer element dtype casts on state stores. -/
def chunk_gated_attention_h_surface
    (K V G H H0 HT : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
      T KSize VSize TK TV BT BK BV NT : Nat)
    (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_v = tl.program_id(axis=0)
  i_k = tl.program_id(axis=1)
  i_bh = tl.program_id(axis=2)
  b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = tl.make_block_ptr(base=H0 + i_bh * $(KSize) * $(VSize),
      shape=($(KSize), $(VSize)), strides=($(VSize), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_h += tl.load(p_h0, boundary_check=([0, 1] : List Nat)).to(tl.float32)
  }
  for i_t in range($(0), $(NT), $(1)) {
    p_k = tl.make_block_ptr(base=K + i_bh * $(s_k_h),
      shape=($(KSize), $(T)), strides=($(s_k_d), $(s_k_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_v = tl.make_block_ptr(base=V + i_bh * $(s_v_h),
      shape=($(T), $(VSize)), strides=($(s_v_t), $(s_v_d)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_h = tl.make_block_ptr(base=H + i_bh * $(s_h_h) + i_t * $(KSize) * $(VSize),
      shape=($(KSize), $(VSize)), strides=($(s_h_t), $(s_h_d)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_h, (b_h).to(p_h.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    if GATEK {
      p_g = tl.make_block_ptr(base=G + i_bh * $(s_k_h),
        shape=($(KSize), $(T)), strides=($(s_k_d), $(s_k_t)),
        offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
      p_gn = tl.make_block_ptr(base=G + i_bh * $(s_k_h),
        shape=($(TK)), strides=($(s_k_d)),
        offsets=((i_t * $(BT) + $(BT) - $(1)) * $(KSize) + i_k * $(BK)),
        block_shape=($(BK)), order=(0))
      b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
      b_h = b_h * tl.exp(b_gn)[:, None]
      b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
      b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(b_k.dtype)
    } else {
      p_g = tl.make_block_ptr(base=G + i_bh * $(s_v_h),
        shape=($(T), $(VSize)), strides=($(s_v_t), $(s_v_d)),
        offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
      p_gn = tl.make_block_ptr(base=G + i_bh * $(s_v_h),
        shape=($(TV)), strides=($(s_v_d)),
        offsets=((i_t * $(BT) + $(BT) - $(1)) * $(VSize) + i_v * $(BV)),
        block_shape=($(BV)), order=(0))
      b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
      b_h = b_h * tl.exp(b_gn)[None, :]
      b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
      b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(b_v.dtype)
    }
    b_h += tl.dot(b_k, b_v, allow_tf32=false)
  }
  if STORE_FINAL_STATE {
    p_h = tl.make_block_ptr(base=HT + i_bh * $(KSize) * $(VSize),
      shape=($(KSize), $(VSize)), strides=($(VSize), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_h, (b_h).to(p_h.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  }
}

/-- Proof-oriented block store slice of `chunk_gated_attention.py`'s
`chunk_gated_abc_fwd_kernel_cum`.

The full kernel computes a per-feature gated-ABC cumsum tile. This slice starts from
a precomputed `BC` tile for one `(i_s, i_bh, i_t)` block and proves the
boundary-checked writeback into `Z`. -/
def chunk_gated_attention_store_slice
    (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(axis=0)
  i_bh = tl.program_id(axis=1)
  i_t = tl.program_id(axis=2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_c = tl.load(BC + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), b_c, mask=mask)
}

def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 2 * BT + i.val

def sIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val

def active (s : BlockState) (T S BT BS : Nat) (idx : TileIndex [BT, BS]) : Prop :=
  tIndex s BT idx.1 < T ∧ sIndex s BS idx.2.1 < S

instance activeDecidable (s : BlockState) (T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : Decidable (active s T S BT BS idx) := by
  unfold active
  infer_instance

def tileOffset (s : BlockState) (s_s_h s_s_t s_s_d BT BS : Nat)
    (idx : TileIndex [BT, BS]) : Nat :=
  s.pids 1 * s_s_h + tIndex s BT idx.1 * s_s_t +
    sIndex s BS idx.2.1 * s_s_d

noncomputable def storeValue (s : BlockState) (BC : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    (if active s T S BT BS idx then
      some (s.readMem BC (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
    else some (0.0 : ℝ))

theorem chunk_gated_attention_store_slice_correct
    (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)) :
    ∀ idx : TileIndex [BT, BS],
      let outAddr := tileOffset s s_s_h s_s_t s_s_d BT BS idx
      (exec (chunk_gated_attention_store_slice BC Z s_s_h s_s_t s_s_d T S BT BS)
          s).map (·.readMem Z outAddr)
        = some (if active s T S BT BS idx then
            storeValue s BC s_s_h s_s_t s_s_d T S BT BS idx
          else s.readMem Z outAddr) := by
  intro idx
  simp [exec, chunk_gated_attention_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        tIndex, sIndex, active, tileOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BT, BS] → Nat :=
    fun idx => s.pids 1 * s_s_h + (s.pids 2 * BT + idx.1.val) * s_s_t +
      (s.pids 0 * BS + idx.2.1.val) * s_s_d
  let valueFn : TileIndex [BT, BS] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 2 * BT + idx.1.val < T ∧
          s.pids 0 * BS + idx.2.1.val < S then
        some (s.readMem BC (offsetFn idx))
      else some (0.0 : ℝ))
  let P : TileIndex [BT, BS] → Prop :=
    fun idx => s.pids 2 * BT + idx.1.val < T ∧
      s.pids 0 * BS + idx.2.1.val < S
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, tileOffset, tIndex, sIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Z (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BT, BS])).readMem Z (offsetFn idx) =
    if P idx then storeValue s BC s_s_h s_s_t s_s_d T S BT BS idx
    else s.readMem Z (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 2 * BT + idx.1.val < T ∧ s.pids 0 * BS + idx.2.1.val < S
  · rfl
  · rfl

theorem chunk_gated_attention_store_slice_compute_correct
    (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)) :
    ComputeCorrect.Realizes
      (kernel := chunk_gated_attention_store_slice BC Z s_s_h s_s_t s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => active s T S BT BS idx)
        (fun idx : TileIndex [BT, BS] => (Z, tileOffset s s_s_h s_s_t s_s_d BT BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        storeValue s BC s_s_h s_s_t s_s_d T S BT BS idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gated_attention_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_gated_attention_store_slice_correct BC Z s_s_h s_s_t s_s_d T S BT BS
    s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.ChunkGatedAttention
