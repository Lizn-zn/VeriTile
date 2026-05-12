import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription of `chunk_delta_fwd.py`'s
`chunk_delta_rule_fwd_kernel_h`.

The source uses dynamic tile-dtype casts around the two dot products; this
surface represents them with `KReg.dtype.element_ty`, the dtype of loaded
`b_k`, while preserving the nested `NT`/`ceil(BT/BC)` loop structure, state
stores, `v_new` stores, and optional initial/final state branches. -/
def chunk_delta_rule_fwd_h_surface
    (KReg VReg DReg VNew HOut InitialState FinalState : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      _H T K V BT BC BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_k = tl.program_id(axis=0)
  i_v = tl.program_id(axis=1)
  i_bh = tl.program_id(axis=2)
  b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = tl.make_block_ptr(base=InitialState + i_bh * $(K) * $(V),
      shape=($(K), $(V)), strides=($(V), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_h = tl.load(p_h0, boundary_check=([0, 1] : List Nat)).to(tl.float32)
  }
  for i_t in range($(0), $(NT), $(1)) {
    p_h = tl.make_block_ptr(base=HOut + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_h, (b_h).to(HOut.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    b_h_cumsum = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
    for i_c in range($(0), tl.cdiv($(BT), $(BC)), $(1)) {
      p_k = tl.make_block_ptr(base=KReg + i_bh * $(s_qk_h),
        shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
        offsets=(i_k * $(BK), i_t * $(BT) + i_c * $(BC)),
        block_shape=($(BK), $(BC)), order=(0, 1))
      p_d = tl.make_block_ptr(base=DReg + i_bh * $(s_qk_h),
        shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)),
        offsets=(i_t * $(BT) + i_c * $(BC), i_k * $(BK)),
        block_shape=($(BC), $(BK)), order=(1, 0))
      p_v = tl.make_block_ptr(base=VReg + i_bh * $(s_vo_h),
        shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
        offsets=(i_t * $(BT) + i_c * $(BC), i_v * $(BV)),
        block_shape=($(BC), $(BV)), order=(1, 0))
      p_v_new = tl.make_block_ptr(base=VNew + i_bh * $(s_vo_h),
        shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
        offsets=(i_t * $(BT) + i_c * $(BC), i_v * $(BV)),
        block_shape=($(BC), $(BV)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
      b_d = tl.load(p_d, boundary_check=([0, 1] : List Nat))
      b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
      b_v = b_v - tl.dot(b_d, (b_h).to(KReg.dtype.element_ty), allow_tf32=false)
      tl.store(p_v_new, (b_v).to(VNew.dtype.element_ty),
        boundary_check=([0, 1] : List Nat))
      b_h_cumsum += tl.dot(b_k, (b_v).to(KReg.dtype.element_ty), allow_tf32=false)
    }
    b_h += b_h_cumsum
  }
  if STORE_FINAL_STATE {
    p_ht = tl.make_block_ptr(base=FinalState + i_bh * $(K) * $(V),
      shape=($(K), $(V)), strides=($(V), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_ht, (b_h).to(FinalState.dtype.element_ty),
      boundary_check=([0, 1] : List Nat))
  }
}

/-- Proof-oriented state-store slice of `chunk_delta_fwd.py`'s
`chunk_delta_rule_fwd_kernel_h`.

The full kernel updates the delta-rule recurrent state and writes intermediate
`h` tiles at each time chunk. This slice models one `i_t` store from a
precomputed `BH` tile into `HOut`, preserving the source K/V block offsets and
boundary checks. -/
def chunk_delta_fwd_h_store_slice
    (BH HOut : RegionName)
    (i_t s_h_h s_h_t K V BK BV : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(axis=0)
  i_v = tl.program_id(axis=1)
  i_bh = tl.program_id(axis=2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_k[:, None] < $(K)) & (offs_v[None, :] < $(V))
  b_h = tl.load(BH + i_bh * $(s_h_h) + $(i_t) * $(K) * $(V) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :], mask=mask, other=0.0)
  tl.store(HOut + i_bh * $(s_h_h) + $(i_t) * $(K) * $(V) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :], b_h, mask=mask)
}

def kIndex (s : BlockState) (BK : Nat) (i : Fin BK) : Nat :=
  s.pids 0 * BK + i.val

def vIndex (s : BlockState) (BV : Nat) (j : Fin BV) : Nat :=
  s.pids 1 * BV + j.val

def active (s : BlockState) (K V BK BV : Nat) (idx : TileIndex [BK, BV]) : Prop :=
  kIndex s BK idx.1 < K ∧ vIndex s BV idx.2.1 < V

instance activeDecidable (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Decidable (active s K V BK BV idx) := by
  unfold active
  infer_instance

def hOffset (s : BlockState) (i_t s_h_h s_h_t K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + i_t * K * V +
    kIndex s BK idx.1 * s_h_t + vIndex s BV idx.2.1

noncomputable def storeValue (s : BlockState) (BH : RegionName)
    (i_t s_h_h s_h_t K V BK BV : Nat) (idx : TileIndex [BK, BV]) : ℝ :=
  WithBot.unbotD 0
    (if active s K V BK BV idx then
      some (s.readMem BH (hOffset s i_t s_h_h s_h_t K V BK BV idx))
    else some (0.0 : ℝ))

theorem chunk_delta_fwd_h_store_slice_correct
    (BH HOut : RegionName) (i_t s_h_h s_h_t K V BK BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => hOffset s i_t s_h_h s_h_t K V BK BV idx)) :
    ∀ idx : TileIndex [BK, BV],
      let outAddr := hOffset s i_t s_h_h s_h_t K V BK BV idx
      (exec (chunk_delta_fwd_h_store_slice BH HOut i_t s_h_h s_h_t K V BK BV)
          s).map (·.readMem HOut outAddr)
        = some (if active s K V BK BV idx then
            storeValue s BH i_t s_h_h s_h_t K V BK BV idx
          else s.readMem HOut outAddr) := by
  intro idx
  simp [exec, chunk_delta_fwd_h_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        kIndex, vIndex, active, hOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BK, BV] → Nat :=
    fun idx => s.pids 2 * s_h_h + i_t * K * V +
      (s.pids 0 * BK + idx.1.val) * s_h_t +
      (s.pids 1 * BV + idx.2.1.val)
  let valueFn : TileIndex [BK, BV] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 0 * BK + idx.1.val < K ∧
          s.pids 1 * BV + idx.2.1.val < V then
        some (s.readMem BH (offsetFn idx))
      else some (0.0 : ℝ))
  let P : TileIndex [BK, BV] → Prop :=
    fun idx => s.pids 0 * BK + idx.1.val < K ∧
      s.pids 1 * BV + idx.2.1.val < V
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, hOffset, kIndex, vIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem HOut (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BK, BV])).readMem HOut (offsetFn idx) =
    if P idx then storeValue s BH i_t s_h_h s_h_t K V BK BV idx
    else s.readMem HOut (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BV + idx.2.1.val < V
  · rfl
  · rfl

theorem chunk_delta_fwd_h_store_slice_compute_correct
    (BH HOut : RegionName) (i_t s_h_h s_h_t K V BK BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => hOffset s i_t s_h_h s_h_t K V BK BV idx)) :
    ComputeCorrect.Realizes
      (kernel := chunk_delta_fwd_h_store_slice BH HOut i_t s_h_h s_h_t K V BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BK, BV] => active s K V BK BV idx)
        (fun idx : TileIndex [BK, BV] => (HOut, hOffset s i_t s_h_h s_h_t K V BK BV idx)))
      (expected := fun idx : TileIndex [BK, BV] =>
        storeValue s BH i_t s_h_h s_h_t K V BK BV idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_delta_fwd_h_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_delta_fwd_h_store_slice_correct BH HOut i_t s_h_h s_h_t K V BK BV
    s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd
