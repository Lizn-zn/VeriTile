import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `chunk_gated_attention.py`'s
`chunk_gated_abc_fwd_kernel_cum`.

This is the forward-preprocessing cumsum kernel used by `fwd_pre`: it builds the
lower-triangular accumulation mask, loads one `[BT, BS]` tile, computes the
chunk-local cumulative sum as a matrix product, and writes the result through a
boundary-checked block pointer. The Triton block-pointer `order` metadata is
scheduling-only; the DSL accepts it at the surface and erases it into the same
block-pointer AST. -/
def chunk_gated_attention_cum_surface
    (s o : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_t = tl.program_id(1)
  i_bh = tl.program_id(2)
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0).to(tl.float32)
  p_s = tl.make_block_ptr(base=s + i_bh * $(s_s_h),
    shape=($(T), $(S)), strides=($(s_s_t), $(s_s_d)),
    offsets=(i_t * $(BT), i_s * $(BS)), block_shape=($(BT), $(BS)), order=(1, 0))
  p_o = tl.make_block_ptr(base=o + i_bh * $(s_s_h),
    shape=($(T), $(S)), strides=($(s_s_t), $(s_s_d)),
    offsets=(i_t * $(BT), i_s * $(BS)), block_shape=($(BT), $(BS)), order=(1, 0))
  b_s = tl.load(p_s, boundary_check=([0, 1] : List Nat)).to(tl.float32)
  b_o = tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

/-- The cumulative preprocessing surface lowers to the algorithm layer. -/
theorem chunk_gated_attention_cum_surface_toAlgorithm_supported
    (s o : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ∃ alg, (chunk_gated_attention_cum_surface s o s_s_h s_s_t s_s_d T S BT
      BS).toAlgorithm? = Except.ok alg := by
  simp [chunk_gated_attention_cum_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

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
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
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

/-- The gated H-state forward surface lowers to the algorithm layer, including
initial-state, gate-K/gate-V, recurrent state stores, and final-state branches. -/
theorem chunk_gated_attention_h_surface_toAlgorithm_supported
    (K V G H H0 HT : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
      T KSize VSize TK TV BT BK BV NT : Nat)
    (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ∃ alg, (chunk_gated_attention_h_surface K V G H H0 HT s_k_h s_k_t
      s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T KSize VSize TK TV
      BT BK BV NT GATEK USE_INITIAL_STATE STORE_FINAL_STATE).toAlgorithm?
        = Except.ok alg := by
  simp [chunk_gated_attention_h_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented block store slice of `chunk_gated_attention.py`'s
`chunk_gated_abc_fwd_kernel_cum`.

The full kernel computes a per-feature gated-ABC cumsum tile. This slice starts from
a precomputed `BC` tile for one `(i_s, i_bh, i_t)` block and proves the
boundary-checked writeback into `Z`. -/
def chunk_gated_attention_store_slice
    (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  i_t = tl.program_id(2)
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
  simp [exec, chunk_gated_attention_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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

/-! ## Computed cumulative-normalizer slice

The `fwd_pre` Python path computes `b_o = tl.dot(m_s, b_s)` with a
lower-triangular mask before storing. This slice proves that computation and
writeback directly, rather than starting from a precomputed `BC` tile. -/

def chunk_gated_attention_cum_compute_slice
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  i_t = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_s = tl.load(SReg + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  b_o = tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), b_o, mask=mask)
}

noncomputable def lowerTriTile (BT : Nat) : Tile .real [BT, BT] :=
  { data := fun idx =>
      if idx.1.val >= idx.2.1.val then some (1.0 : ℝ) else some (0.0 : ℝ) }

noncomputable def sourceTile
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Tile .real [BT, BS] :=
  { data := fun idx =>
      if active s T S BT BS idx then
        some (s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
      else some (0.0 : ℝ) }

noncomputable def cumComputeStoreValue
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    ((Tile.dot [] (lowerTriTile BT)
      (sourceTile s SReg s_s_h s_s_t s_s_d T S BT BS)).data
        (idx.1, idx.2.1, PUnit.unit))

theorem chunk_gated_attention_cum_compute_slice_correct
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)) :
    ∀ idx : TileIndex [BT, BS],
      let outAddr := tileOffset s s_s_h s_s_t s_s_d BT BS idx
      (exec (chunk_gated_attention_cum_compute_slice SReg Z s_s_h s_s_t
            s_s_d T S BT BS) s).map (·.readMem Z outAddr)
        = some (if active s T S BT BS idx then
            cumComputeStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx
          else s.readMem Z outAddr) := by
  intro idx
  simp [exec, chunk_gated_attention_cum_compute_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, Tile.dot, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, ComparableDType.ge, tIndex, sIndex, active,
        tileOffset, sourceTile, lowerTriTile, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BT, BS] → Nat :=
    fun idx => s.pids 1 * s_s_h + (s.pids 2 * BT + idx.1.val) * s_s_t +
      (s.pids 0 * BS + idx.2.1.val) * s_s_d
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, tileOffset, tIndex, sIndex] using hOutInj
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 2 * BT + idx.1.val < T ∧ s.pids 0 * BS + idx.2.1.val < S
  · simp [offsetFn, active, tIndex, sIndex, tileOffset, cumComputeStoreValue,
      sourceTile, lowerTriTile, Tile.dot, hActive]
  · simp [offsetFn, active, tIndex, sIndex, tileOffset, hActive]

theorem chunk_gated_attention_cum_compute_slice_compute_correct
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)) :
    ComputeCorrect.Realizes
      (kernel := chunk_gated_attention_cum_compute_slice SReg Z s_s_h s_s_t
        s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => active s T S BT BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (Z, tileOffset s s_s_h s_s_t s_s_d BT BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        cumComputeStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gated_attention_cum_compute_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_gated_attention_cum_compute_slice_correct SReg Z s_s_h
    s_s_t s_s_d T S BT BS s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented intermediate-state store slice of
`chunk_gated_abc_fwd_kernel_h`.

At each `i_t`, the Python kernel stores the current recurrent state `b_h` into
`H + i_bh * s_h_h + i_t * KSize * VSize` before applying the chunk update.
This slice starts from a precomputed `BH` tile and proves that masked writeback. -/
def chunk_gated_attention_h_state_store_slice
    (BH H : RegionName) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_k[:, None] < $(KSize)) & (offs_v[None, :] < $(VSize))
  b_h = tl.load(BH + i_bh * $(s_h_h) + $(i_t) * $(KSize) * $(VSize) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :] * $(s_h_d),
    mask=mask, other=0.0)
  tl.store(H + i_bh * $(s_h_h) + $(i_t) * $(KSize) * $(VSize) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :] * $(s_h_d),
    b_h, mask=mask)
}

def kIndexState (s : BlockState) (BK : Nat) (i : Fin BK) : Nat :=
  s.pids 1 * BK + i.val

def vIndexState (s : BlockState) (BV : Nat) (j : Fin BV) : Nat :=
  s.pids 0 * BV + j.val

def stateActive (s : BlockState) (KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Prop :=
  kIndexState s BK idx.1 < KSize ∧ vIndexState s BV idx.2.1 < VSize

instance stateActiveDecidable (s : BlockState) (KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Decidable (stateActive s KSize VSize BK BV idx) := by
  unfold stateActive
  infer_instance

def hStateOffset (s : BlockState) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + i_t * KSize * VSize +
    kIndexState s BK idx.1 * s_h_t + vIndexState s BV idx.2.1 * s_h_d

noncomputable def hStateStoreValue (s : BlockState) (BH : RegionName)
    (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : ℝ :=
  WithBot.unbotD 0
    (if stateActive s KSize VSize BK BV idx then
      some (s.readMem BH (hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx))
    else some (0.0 : ℝ))

theorem chunk_gated_attention_h_state_store_slice_correct
    (BH H : RegionName) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx)) :
    ∀ idx : TileIndex [BK, BV],
      let outAddr := hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx
      (exec (chunk_gated_attention_h_state_store_slice BH H i_t s_h_h s_h_t
            s_h_d KSize VSize BK BV) s).map (·.readMem H outAddr)
        = some (if stateActive s KSize VSize BK BV idx then
            hStateStoreValue s BH i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx
          else s.readMem H outAddr) := by
  intro idx
  simp [exec, chunk_gated_attention_h_state_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        kIndexState, vIndexState, stateActive, hStateOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BK, BV] → Nat :=
    fun idx => s.pids 2 * s_h_h + i_t * KSize * VSize +
      (s.pids 1 * BK + idx.1.val) * s_h_t +
      (s.pids 0 * BV + idx.2.1.val) * s_h_d
  let valueFn : TileIndex [BK, BV] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 1 * BK + idx.1.val < KSize ∧
          s.pids 0 * BV + idx.2.1.val < VSize then
        some (s.readMem BH (offsetFn idx))
      else some (0.0 : ℝ))
  let P : TileIndex [BK, BV] → Prop :=
    fun idx => s.pids 1 * BK + idx.1.val < KSize ∧
      s.pids 0 * BV + idx.2.1.val < VSize
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, hStateOffset, kIndexState, vIndexState] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem H (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BK, BV])).readMem H (offsetFn idx) =
    if P idx then
      hStateStoreValue s BH i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx
    else s.readMem H (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 1 * BK + idx.1.val < KSize ∧
      s.pids 0 * BV + idx.2.1.val < VSize
  · rfl
  · rfl

theorem chunk_gated_attention_h_state_store_slice_compute_correct
    (BH H : RegionName) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx)) :
    ComputeCorrect.Realizes
      (kernel := chunk_gated_attention_h_state_store_slice BH H i_t s_h_h
        s_h_t s_h_d KSize VSize BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BK, BV] => stateActive s KSize VSize BK BV idx)
        (fun idx : TileIndex [BK, BV] =>
          (H, hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx)))
      (expected := fun idx : TileIndex [BK, BV] =>
        hStateStoreValue s BH i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gated_attention_h_state_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_gated_attention_h_state_store_slice_correct BH H i_t s_h_h
    s_h_t s_h_d KSize VSize BK BV s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented final-state store slice of `chunk_gated_attention.py`'s
`chunk_gated_abc_fwd_kernel_h`. Writes a precomputed final-state `BHFinal`
[BK, BV] tile into `Ht` after the NT-iteration chunk loop completes
(STORE_FINAL_STATE=True branch). -/
def chunk_gated_attention_final_state_store_slice
    (BHFinal Ht : RegionName) (KSize VSize BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_k[:, None] < $(KSize)) & (offs_v[None, :] < $(VSize))
  b_h = tl.load(BHFinal + i_bh * $(KSize) * $(VSize) +
      offs_k[:, None] * $(VSize) + offs_v[None, :],
    mask=mask, other=0.0)
  tl.store(Ht + i_bh * $(KSize) * $(VSize) +
      offs_k[:, None] * $(VSize) + offs_v[None, :], b_h, mask=mask)
}

def kIndexFinal (s : BlockState) (BK : Nat) (i : Fin BK) : Nat :=
  s.pids 1 * BK + i.val

def vIndexFinal (s : BlockState) (BV : Nat) (j : Fin BV) : Nat :=
  s.pids 0 * BV + j.val

def finalActive (s : BlockState) (KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Prop :=
  kIndexFinal s BK idx.1 < KSize ∧ vIndexFinal s BV idx.2.1 < VSize

instance finalActiveDecidable (s : BlockState) (KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Decidable (finalActive s KSize VSize BK BV idx) := by
  unfold finalActive
  infer_instance

def finalStateOffset (s : BlockState) (KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * KSize * VSize +
    kIndexFinal s BK idx.1 * VSize + vIndexFinal s BV idx.2.1

noncomputable def finalStateStoreValue (s : BlockState) (BHFinal : RegionName)
    (KSize VSize BK BV : Nat) (idx : TileIndex [BK, BV]) : ℝ :=
  WithBot.unbotD 0
    (if finalActive s KSize VSize BK BV idx then
      some (s.readMem BHFinal (finalStateOffset s KSize VSize BK BV idx))
    else some (0.0 : ℝ))

theorem chunk_gated_attention_final_state_store_slice_correct
    (BHFinal Ht : RegionName) (KSize VSize BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        finalStateOffset s KSize VSize BK BV idx)) :
    ∀ idx : TileIndex [BK, BV],
      let outAddr := finalStateOffset s KSize VSize BK BV idx
      (exec (chunk_gated_attention_final_state_store_slice BHFinal Ht
            KSize VSize BK BV) s).map (·.readMem Ht outAddr)
        = some (if finalActive s KSize VSize BK BV idx then
            finalStateStoreValue s BHFinal KSize VSize BK BV idx
          else s.readMem Ht outAddr) := by
  intro idx
  simp [exec, chunk_gated_attention_final_state_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        kIndexFinal, vIndexFinal, finalActive, finalStateOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BK, BV] → Nat :=
    fun idx => s.pids 2 * KSize * VSize +
      (s.pids 1 * BK + idx.1.val) * VSize +
      (s.pids 0 * BV + idx.2.1.val)
  let valueFn : TileIndex [BK, BV] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 1 * BK + idx.1.val < KSize ∧
          s.pids 0 * BV + idx.2.1.val < VSize then
        some (s.readMem BHFinal (offsetFn idx))
      else some (0.0 : ℝ))
  let P : TileIndex [BK, BV] → Prop :=
    fun idx => s.pids 1 * BK + idx.1.val < KSize ∧
      s.pids 0 * BV + idx.2.1.val < VSize
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, finalStateOffset, kIndexFinal, vIndexFinal] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Ht (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BK, BV])).readMem Ht (offsetFn idx) =
    if P idx then finalStateStoreValue s BHFinal KSize VSize BK BV idx
    else s.readMem Ht (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 1 * BK + idx.1.val < KSize ∧ s.pids 0 * BV + idx.2.1.val < VSize
  · rfl
  · rfl

theorem chunk_gated_attention_final_state_store_slice_compute_correct
    (BHFinal Ht : RegionName) (KSize VSize BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        finalStateOffset s KSize VSize BK BV idx)) :
    ComputeCorrect.Realizes
      (kernel := chunk_gated_attention_final_state_store_slice BHFinal Ht
        KSize VSize BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BK, BV] => finalActive s KSize VSize BK BV idx)
        (fun idx : TileIndex [BK, BV] => (Ht, finalStateOffset s KSize VSize BK BV idx)))
      (expected := fun idx : TileIndex [BK, BV] =>
        finalStateStoreValue s BHFinal KSize VSize BK BV idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gated_attention_final_state_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_gated_attention_final_state_store_slice_correct BHFinal Ht
    KSize VSize BK BV s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

theorem chunk_gated_attention_python_pre_bs16_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [32, 16] => tileOffset s 8192 64 1 32 16 idx) := by
  rintro ⟨⟨ta, hta⟩, ⟨sa, hsa⟩, _⟩ ⟨⟨tb, htb⟩, ⟨sb, hsb⟩, _⟩ h
  simp [tileOffset, tIndex, sIndex] at h
  have ht : ta = tb := by omega
  have hs : sa = sb := by omega
  subst tb
  subst sb
  rfl

theorem chunk_gated_attention_python_pre_bs32_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [32, 32] => tileOffset s 8192 64 1 32 32 idx) := by
  rintro ⟨⟨ta, hta⟩, ⟨sa, hsa⟩, _⟩ ⟨⟨tb, htb⟩, ⟨sb, hsb⟩, _⟩ h
  simp [tileOffset, tIndex, sIndex] at h
  have ht : ta = tb := by omega
  have hs : sa = sb := by omega
  subst tb
  subst sb
  rfl

theorem chunk_gated_attention_python_pre_bs64_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [32, 64] => tileOffset s 8192 64 1 32 64 idx) := by
  rintro ⟨⟨ta, hta⟩, ⟨sa, hsa⟩, _⟩ ⟨⟨tb, htb⟩, ⟨sb, hsb⟩, _⟩ h
  simp [tileOffset, tIndex, sIndex] at h
  have ht : ta = tb := by omega
  have hs : sa = sb := by omega
  subst tb
  subst sb
  rfl

theorem chunk_gated_attention_python_h_state_offset_injective
    (s : BlockState) (i_t : Fin 4) :
    Function.Injective
      (fun idx : TileIndex [16, 16] =>
        hStateOffset s i_t.val 4096 32 1 32 32 16 16 idx) := by
  rintro ⟨⟨ka, hka⟩, ⟨va, hva⟩, _⟩ ⟨⟨kb, hkb⟩, ⟨vb, hvb⟩, _⟩ h
  simp [hStateOffset, kIndexState, vIndexState] at h
  have hk : ka = kb := by omega
  have hv : va = vb := by omega
  subst kb
  subst vb
  rfl

theorem chunk_gated_attention_python_final_state_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [16, 16] => finalStateOffset s 32 32 16 16 idx) := by
  rintro ⟨⟨ka, hka⟩, ⟨va, hva⟩, _⟩ ⟨⟨kb, hkb⟩, ⟨vb, hvb⟩, _⟩ h
  simp [finalStateOffset, kIndexFinal, vIndexFinal] at h
  have hk : ka = kb := by omega
  have hv : va = vb := by omega
  subst kb
  subst vb
  rfl

theorem chunk_gated_attention_cum_python_bs16_compute_correct
    (SReg Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gated_attention_cum_compute_slice SReg Z 8192 64 1
        128 64 32 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [32, 16] => active s 128 64 32 16 idx)
        (fun idx : TileIndex [32, 16] => (Z, tileOffset s 8192 64 1 32 16 idx)))
      (expected := fun idx : TileIndex [32, 16] =>
        cumComputeStoreValue s SReg 8192 64 1 128 64 32 16 idx) := by
  exact chunk_gated_attention_cum_compute_slice_compute_correct SReg Z
    8192 64 1 128 64 32 16 s
    (chunk_gated_attention_python_pre_bs16_offset_injective s)

theorem chunk_gated_attention_cum_python_bs32_compute_correct
    (SReg Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gated_attention_cum_compute_slice SReg Z 8192 64 1
        128 64 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [32, 32] => active s 128 64 32 32 idx)
        (fun idx : TileIndex [32, 32] => (Z, tileOffset s 8192 64 1 32 32 idx)))
      (expected := fun idx : TileIndex [32, 32] =>
        cumComputeStoreValue s SReg 8192 64 1 128 64 32 32 idx) := by
  exact chunk_gated_attention_cum_compute_slice_compute_correct SReg Z
    8192 64 1 128 64 32 32 s
    (chunk_gated_attention_python_pre_bs32_offset_injective s)

theorem chunk_gated_attention_cum_python_bs64_compute_correct
    (SReg Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gated_attention_cum_compute_slice SReg Z 8192 64 1
        128 64 32 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [32, 64] => active s 128 64 32 64 idx)
        (fun idx : TileIndex [32, 64] => (Z, tileOffset s 8192 64 1 32 64 idx)))
      (expected := fun idx : TileIndex [32, 64] =>
        cumComputeStoreValue s SReg 8192 64 1 128 64 32 64 idx) := by
  exact chunk_gated_attention_cum_compute_slice_compute_correct SReg Z
    8192 64 1 128 64 32 64 s
    (chunk_gated_attention_python_pre_bs64_offset_injective s)

theorem chunk_gated_attention_h_state_python_test_shape_compute_correct
    (BH H : RegionName) (i_t : Fin 4) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gated_attention_h_state_store_slice BH H i_t.val
        4096 32 1 32 32 16 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => stateActive s 32 32 16 16 idx)
        (fun idx : TileIndex [16, 16] =>
          (H, hStateOffset s i_t.val 4096 32 1 32 32 16 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        hStateStoreValue s BH i_t.val 4096 32 1 32 32 16 16 idx) := by
  exact chunk_gated_attention_h_state_store_slice_compute_correct BH H i_t.val
    4096 32 1 32 32 16 16 s
    (chunk_gated_attention_python_h_state_offset_injective s i_t)

theorem chunk_gated_attention_final_state_python_test_shape_compute_correct
    (BHFinal Ht : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_gated_attention_final_state_store_slice BHFinal Ht
        32 32 16 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => finalActive s 32 32 16 16 idx)
        (fun idx : TileIndex [16, 16] => (Ht, finalStateOffset s 32 32 16 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        finalStateStoreValue s BHFinal 32 32 16 16 idx) := by
  exact chunk_gated_attention_final_state_store_slice_compute_correct BHFinal Ht
    32 32 16 16 s
    (chunk_gated_attention_python_final_state_offset_injective s)

/-- Python test-shape output coverage for chunk gated attention: every checked
state-output surface (`h` at each loop row and final state) realizes its masked
store shape. -/
theorem chunk_gated_attention_python_test_shape_all_outputs_compute_correct
    (BH H BHFinal Ht : RegionName) (i_t : Fin 4) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := chunk_gated_attention_h_state_store_slice BH H i_t.val
        4096 32 1 32 32 16 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => stateActive s 32 32 16 16 idx)
        (fun idx : TileIndex [16, 16] =>
          (H, hStateOffset s i_t.val 4096 32 1 32 32 16 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        hStateStoreValue s BH i_t.val 4096 32 1 32 32 16 16 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gated_attention_final_state_store_slice BHFinal Ht
        32 32 16 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => finalActive s 32 32 16 16 idx)
        (fun idx : TileIndex [16, 16] => (Ht, finalStateOffset s 32 32 16 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        finalStateStoreValue s BHFinal 32 32 16 16 idx)) := by
  constructor
  · exact chunk_gated_attention_h_state_python_test_shape_compute_correct
      BH H i_t s
  · exact chunk_gated_attention_final_state_python_test_shape_compute_correct
      BHFinal Ht s

/-- Public Python test-shape summary for `chunk_gated_attention.py`.

This packages the checked `fwd_pre` cumulative surface, the four Python
`fwd_inner` branch surfaces, and the observable state outputs (`h` loop rows and
optional `ht`) for the benchmark shape
`B=2, H=4, T=128, S=64, K=32, V=32, BT=32, BK=BV=16`. -/
theorem chunk_gated_attention_python_test_shape_output_summary
    (G GCum K V H H0 Ht BH BHFinal : RegionName) (i_t : Fin 4)
    (s : BlockState) :
    (∃ alg, (chunk_gated_attention_cum_surface G GCum 8192 64 1
      128 64 32 16).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (chunk_gated_attention_h_surface K V GCum H H0 Ht
      4096 1 128 4096 32 1 4096 32 1
      128 32 32 4096 4096 32 16 16 4
      Bool.false Bool.false Bool.false).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (chunk_gated_attention_h_surface K V GCum H H0 Ht
      4096 1 128 4096 32 1 4096 32 1
      128 32 32 4096 4096 32 16 16 4
      Bool.true Bool.true Bool.false).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (chunk_gated_attention_h_surface K V GCum H H0 Ht
      4096 1 128 4096 32 1 4096 32 1
      128 32 32 4096 4096 32 16 16 4
      Bool.false Bool.true Bool.true).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (chunk_gated_attention_h_surface K V GCum H H0 Ht
      4096 1 128 4096 32 1 4096 32 1
      128 32 32 4096 4096 32 16 16 4
      Bool.true Bool.false Bool.true).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gated_attention_h_state_store_slice BH H i_t.val
        4096 32 1 32 32 16 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => stateActive s 32 32 16 16 idx)
        (fun idx : TileIndex [16, 16] =>
          (H, hStateOffset s i_t.val 4096 32 1 32 32 16 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        hStateStoreValue s BH i_t.val 4096 32 1 32 32 16 16 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gated_attention_final_state_store_slice BHFinal Ht
        32 32 16 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => finalActive s 32 32 16 16 idx)
        (fun idx : TileIndex [16, 16] =>
          (Ht, finalStateOffset s 32 32 16 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        finalStateStoreValue s BHFinal 32 32 16 16 idx)) := by
  constructor
  · exact chunk_gated_attention_cum_surface_toAlgorithm_supported G GCum
      8192 64 1 128 64 32 16
  constructor
  · exact chunk_gated_attention_h_surface_toAlgorithm_supported K V GCum H H0
      Ht 4096 1 128 4096 32 1 4096 32 1
      128 32 32 4096 4096 32 16 16 4
      Bool.false Bool.false Bool.false
  constructor
  · exact chunk_gated_attention_h_surface_toAlgorithm_supported K V GCum H H0
      Ht 4096 1 128 4096 32 1 4096 32 1
      128 32 32 4096 4096 32 16 16 4
      Bool.true Bool.true Bool.false
  constructor
  · exact chunk_gated_attention_h_surface_toAlgorithm_supported K V GCum H H0
      Ht 4096 1 128 4096 32 1 4096 32 1
      128 32 32 4096 4096 32 16 16 4
      Bool.false Bool.true Bool.true
  constructor
  · exact chunk_gated_attention_h_surface_toAlgorithm_supported K V GCum H H0
      Ht 4096 1 128 4096 32 1 4096 32 1
      128 32 32 4096 4096 32 16 16 4
      Bool.true Bool.false Bool.true
  · exact chunk_gated_attention_python_test_shape_all_outputs_compute_correct
      BH H BHFinal Ht i_t s

end VeriTile.Bench.TritonBenchG.ChunkGatedAttention
