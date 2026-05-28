import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Full Lean port of `attention_fwd_triton3.py`'s `_attn_fwd`. -/
def attention_fwd_triton3_surface
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      _sliding_window_offset _sliding_window_size
      IS_EVEN_M _IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      _SLIDING_WINDOW _COMPLEMENT_SLIDING_WINDOW : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  off_hkv = off_h // ($(H) // $(H_KV))
  q_offset = off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh)
  k_offset = off_z.to(tl.int64) * $(stride_kz) + off_hkv.to(tl.int64) * $(stride_kh)
  v_offset = off_z.to(tl.int64) * $(stride_vz) + off_hkv.to(tl.int64) * $(stride_vh)
  o_offset = off_z.to(tl.int64) * $(stride_oz) + off_h.to(tl.int64) * $(stride_oh)

  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)), offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)), order=(1, 0))
  V_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=($(NKV_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)), offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)), order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=($(BLOCK_DMODEL), $(NKV_CTX)),
    strides=($(stride_kk), $(stride_kn)), offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)), order=(0, 1))
  O_block_ptr = tl.make_block_ptr(base=Out + o_offset, shape=($(ROUND_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)), offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)), order=(1, 0))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  m_ptrs = M + off_hz * $(ROUND_CTX) + offs_m
  l_ptrs = L + off_hz * $(ROUND_CTX) + offs_m
  if $(INIT) != $(0) {
    m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
    acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  } else {
    m_i = tl.load(m_ptrs).to(tl.float32)
    l_i = tl.load(l_ptrs).to(tl.float32)
    acc = tl.load(O_block_ptr).to(tl.float32)
  }
  qk_scale = $(sm_scale) * 1.0
  qk_scale *= 1.4426950408889634
  if $(IS_EVEN_M) != $(0) {
    q = tl.load(Q_block_ptr)
  } else {
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
    start_m, qk_scale, $(NKV_CTX), $(_sliding_window_offset), $(_sliding_window_size),
    $(BLOCK_M), $(BLOCK_DMODEL), $(BLOCK_N), $(_SLIDING_WINDOW), $(IS_EVEN_M),
    $(_IS_EVEN_N), $(_COMPLEMENT_SLIDING_WINDOW))
  if $(END) != $(0) {
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
  } else {
    tl.store(l_ptrs, l_i)
  }
  tl.store(m_ptrs, m_i)
  tl.store(O_block_ptr, (acc).to(Out.type.element_ty))
}

/-- The full Python-shaped `_attn_fwd` surface lowers through the algorithm
translation, covering the staged accumulator load/update path and the final
M/O stores. -/
theorem attention_fwd_triton3_surface_toAlgorithm_supported
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      _sliding_window_offset _sliding_window_size
      IS_EVEN_M _IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      _SLIDING_WINDOW _COMPLEMENT_SLIDING_WINDOW : Nat) :
    ∃ alg, (attention_fwd_triton3_surface Q K V M Out L sm_scale stride_qz
      stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om
      stride_on _Z H H_KV N_CTX ROUND_CTX NKV_CTX _sliding_window_offset
      _sliding_window_size IS_EVEN_M _IS_EVEN_N BLOCK_M BLOCK_DMODEL
      BLOCK_N END INIT _SLIDING_WINDOW _COMPLEMENT_SLIDING_WINDOW).toAlgorithm?
        = Except.ok alg := by
  simp [attention_fwd_triton3_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of `attention_fwd_triton3.py`'s
`_attn_fwd`.

The full kernel runs separate streaming attention stages, including the causal
stage when requested. This slice starts after those stages have produced a
precomputed normalized `Acc` tile and proves the final masked writeback into
`Out`, preserving the source store address and mask
`(offs_m < N_CTX) & (offs_k < 96)`. The inner `tl.float32` accumulator is
outside this slice. -/
def attention_fwd_triton3_final_store_slice
    (Acc Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < $(N_CTX)) & (offs_k[None, :] < $(HEAD_ACTIVE))
  acc = tl.load(Acc + off_z * $(stride_acc_z) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_k[None, :] * $(stride_acc_k),
      mask=mask, other=0.0)
  tl.store(Out + off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk),
      (acc).to(Out.dtype.element_ty), mask=mask)
}

def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H

def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def kIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def active
    (s : BlockState) (N_CTX HEAD_ACTIVE BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < N_CTX ∧ kIndex idx < HEAD_ACTIVE

instance activeDecidable
    (s : BlockState) (N_CTX HEAD_ACTIVE BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (active s N_CTX HEAD_ACTIVE BLOCK_M idx) := by
  unfold active
  infer_instance

def accOffset
    (s : BlockState)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_k BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_acc_z + offH s H * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + kIndex idx * stride_acc_k

def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_qm + kIndex idx * stride_qk

/-- Algorithm-layer correctness for the final output store. -/
theorem attention_fwd_triton3_final_store_slice_correct
    (Acc Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s H stride_qz stride_qh stride_qm stride_qk
        BLOCK_M idx
      (exec (attention_fwd_triton3_final_store_slice Acc Out H N_CTX
            HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k
            stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (if active s N_CTX HEAD_ACTIVE BLOCK_M idx then
            s.readMem Acc
              (accOffset s H stride_acc_z stride_acc_h stride_acc_m
                stride_acc_k BLOCK_M idx)
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, attention_fwd_triton3_final_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
        offZ, offH, mIndex, kIndex, active, accOffset, outOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh +
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧
            idx.2.1.val < HEAD_ACTIVE then
          some (s.readMem Acc
            (s.pids 1 / H * stride_acc_z + s.pids 1 % H * stride_acc_h +
              (s.pids 0 * BLOCK_M + idx.1.val) * stride_acc_m +
              idx.2.1.val * stride_acc_k))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧
        idx.2.1.val < HEAD_ACTIVE
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, offZ, offH, mIndex, kIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if P idx then
      s.readMem Acc
        (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_k
          BLOCK_M idx)
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE
  · simp [offsetFn, valueFn, P, active, accOffset, outOffset, offZ, offH,
      mIndex, kIndex, hActive]
  · simp [offsetFn, valueFn, P, active, accOffset, outOffset, offZ, offH,
      mIndex, kIndex, hActive]

/-- Compute-facing correctness for the final output store. -/
theorem attention_fwd_triton3_final_store_slice_compute_correct
    (Acc Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_final_store_slice Acc Out H N_CTX
        HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k
        stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s N_CTX HEAD_ACTIVE BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        s.readMem Acc
          (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_k
            BLOCK_M idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := attention_fwd_triton3_final_store_slice_correct Acc Out H N_CTX
    HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k stride_qz
    stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Formula-level `END=True` output normalization and final store:
`acc = acc / l_i[:, None]` before the masked `Out` writeback. This proves the
observable active output cells against the Python epilogue arithmetic instead
of treating the normalized accumulator as precomputed. -/
def attention_fwd_triton3_end_output_formula_store_slice
    (Acc LPre Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      ROUND_CTX BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < $(N_CTX)) & (offs_k[None, :] < $(HEAD_ACTIVE))
  acc = tl.load(Acc + off_z * $(stride_acc_z) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_k[None, :] * $(stride_acc_k),
      mask=mask, other=0.0)
  l_i = tl.load(LPre + off_hz * $(ROUND_CTX) + offs_m)
  acc = acc / l_i[:, None]
  tl.store(Out + off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk),
      (acc).to(Out.dtype.element_ty), mask=mask)
}

noncomputable def endOutputStoreSpec
    (s : BlockState) (Acc LPre : RegionName)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_k ROUND_CTX
      BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  s.readMem Acc
      (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_k
        BLOCK_M idx) /
    s.readMem LPre (s.pids 1 * ROUND_CTX + mIndex s BLOCK_M idx.1)

theorem attention_fwd_triton3_end_output_formula_store_slice_correct
    (Acc LPre Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      ROUND_CTX BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s H stride_qz stride_qh stride_qm stride_qk
        BLOCK_M idx
      (exec (attention_fwd_triton3_end_output_formula_store_slice Acc LPre Out
            H N_CTX HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m
            stride_acc_k stride_qz stride_qh stride_qm stride_qk ROUND_CTX
            BLOCK_M BLOCK_DMODEL) s).map (·.readMem Out outAddr)
        = some (if active s N_CTX HEAD_ACTIVE BLOCK_M idx then
            endOutputStoreSpec s Acc LPre H stride_acc_z stride_acc_h
              stride_acc_m stride_acc_k ROUND_CTX BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, attention_fwd_triton3_end_output_formula_store_slice,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, NumericDType.div, IntegralDType.floorDiv,
        IntegralDType.mod, ComparableDType.lt, offZ, offH, mIndex, kIndex,
        active, accOffset, outOffset, endOutputStoreSpec,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh +
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧
        idx.2.1.val < HEAD_ACTIVE
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, offZ, offH, mIndex, kIndex] using hOutInj
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE
  · simp [offsetFn, P, active, accOffset, outOffset,
      endOutputStoreSpec, offZ, offH, mIndex, kIndex, hActive]
  · simp [offsetFn, P, active, accOffset, outOffset,
      endOutputStoreSpec, offZ, offH, mIndex, kIndex, hActive]

theorem attention_fwd_triton3_end_output_formula_store_slice_compute_correct
    (Acc LPre Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      ROUND_CTX BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_output_formula_store_slice Acc LPre
        Out H N_CTX HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m
        stride_acc_k stride_qz stride_qh stride_qm stride_qk ROUND_CTX
        BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s N_CTX HEAD_ACTIVE BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        endOutputStoreSpec s Acc LPre H stride_acc_z stride_acc_h stride_acc_m
          stride_acc_k ROUND_CTX BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_end_output_formula_store_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := attention_fwd_triton3_end_output_formula_store_slice_correct
    Acc LPre Out H N_CTX HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m
    stride_acc_k stride_qz stride_qh stride_qm stride_qk ROUND_CTX BLOCK_M
    BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented L (log-sum-exp) row store slice of `attention_fwd_triton3.py`.
Takes a precomputed `LPre` vector and proves the row writeback into `L` at
offset `off_hz * ROUND_CTX + offs_m`. -/
def attention_fwd_triton3_l_store_slice
    (LPre L : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  l_i = tl.load(LPre + $(off_hz) * $(ROUND_CTX) + offs_m)
  tl.store(L + $(off_hz) * $(ROUND_CTX) + offs_m, l_i)
}

def lRowOffset (s : BlockState) (off_hz ROUND_CTX BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  off_hz * ROUND_CTX + (s.pids 0 * BLOCK_M + i.val)

noncomputable def lStoreSpec (s : BlockState) (LPre : RegionName)
    (off_hz ROUND_CTX BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem LPre (lRowOffset s off_hz ROUND_CTX BLOCK_M i)

theorem attention_fwd_triton3_l_store_slice_correct
    (LPre L : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz ROUND_CTX BLOCK_M i)) :
    ∀ i : Fin BLOCK_M,
      let outAddr := lRowOffset s off_hz ROUND_CTX BLOCK_M i
      (exec (attention_fwd_triton3_l_store_slice LPre L off_hz ROUND_CTX BLOCK_M)
          s).map (·.readMem L outAddr)
        = some (lStoreSpec s LPre off_hz ROUND_CTX BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] =>
        off_hz * ROUND_CTX + (s.pids 0 * BLOCK_M + idx.1.val)) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [lRowOffset, Nat.add_assoc] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, attention_fwd_triton3_l_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul]
  simp only [lRowOffset, Nat.add_assoc]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [lStoreSpec, lRowOffset, Nat.add_assoc]

theorem attention_fwd_triton3_l_store_slice_compute_correct
    (LPre L : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz ROUND_CTX BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_l_store_slice LPre L off_hz ROUND_CTX BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lRowOffset s off_hz ROUND_CTX BLOCK_M i))
      (expected := fun i => lStoreSpec s LPre off_hz ROUND_CTX BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_l_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := attention_fwd_triton3_l_store_slice_correct LPre L off_hz ROUND_CTX BLOCK_M
    s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-- Proof-oriented M (max) row store slice of `attention_fwd_triton3.py`.
Mirrors the L-row store slice. -/
def attention_fwd_triton3_m_store_slice
    (MPre M : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  m_i = tl.load(MPre + $(off_hz) * $(ROUND_CTX) + offs_m)
  tl.store(M + $(off_hz) * $(ROUND_CTX) + offs_m, m_i)
}

noncomputable def mStoreSpec (s : BlockState) (MPre : RegionName)
    (off_hz ROUND_CTX BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem MPre (lRowOffset s off_hz ROUND_CTX BLOCK_M i)

theorem attention_fwd_triton3_m_store_slice_correct
    (MPre M : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz ROUND_CTX BLOCK_M i)) :
    ∀ i : Fin BLOCK_M,
      let outAddr := lRowOffset s off_hz ROUND_CTX BLOCK_M i
      (exec (attention_fwd_triton3_m_store_slice MPre M off_hz ROUND_CTX BLOCK_M)
          s).map (·.readMem M outAddr)
        = some (mStoreSpec s MPre off_hz ROUND_CTX BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] =>
        off_hz * ROUND_CTX + (s.pids 0 * BLOCK_M + idx.1.val)) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [lRowOffset, Nat.add_assoc] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, attention_fwd_triton3_m_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul]
  simp only [lRowOffset, Nat.add_assoc]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [mStoreSpec, lRowOffset, Nat.add_assoc]

theorem attention_fwd_triton3_m_store_slice_compute_correct
    (MPre M : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz ROUND_CTX BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_m_store_slice MPre M off_hz ROUND_CTX BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (M, lRowOffset s off_hz ROUND_CTX BLOCK_M i))
      (expected := fun i => mStoreSpec s MPre off_hz ROUND_CTX BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_m_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := attention_fwd_triton3_m_store_slice_correct MPre M off_hz ROUND_CTX BLOCK_M
    s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-- Formula-level `END=True` M-row epilogue of `attention_fwd_triton3.py`:
`m_i += tl.math.log2(l_i)`, then store to `M`. This starts from the row values
computed by the streaming inner loop and proves the Python epilogue arithmetic,
not only a precomputed M readback. -/
def attention_fwd_triton3_end_m_formula_store_slice
    (MPre LPre M : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  m_i = tl.load(MPre + $(off_hz) * $(ROUND_CTX) + offs_m)
  l_i = tl.load(LPre + $(off_hz) * $(ROUND_CTX) + offs_m)
  m_i += tl.math.log2(l_i)
  tl.store(M + $(off_hz) * $(ROUND_CTX) + offs_m, m_i)
}

noncomputable def endMStoreSpec
    (s : BlockState) (MPre LPre : RegionName)
    (off_hz ROUND_CTX BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem MPre (lRowOffset s off_hz ROUND_CTX BLOCK_M i) +
    Real.log (s.readMem LPre (lRowOffset s off_hz ROUND_CTX BLOCK_M i)) /
      Real.log 2

theorem attention_fwd_triton3_end_m_formula_store_slice_correct
    (MPre LPre M : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz ROUND_CTX BLOCK_M i)) :
    ∀ i : Fin BLOCK_M,
      let outAddr := lRowOffset s off_hz ROUND_CTX BLOCK_M i
      (exec (attention_fwd_triton3_end_m_formula_store_slice MPre LPre M
          off_hz ROUND_CTX BLOCK_M) s).map (·.readMem M outAddr)
        = some (endMStoreSpec s MPre LPre off_hz ROUND_CTX BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] =>
        off_hz * ROUND_CTX + (s.pids 0 * BLOCK_M + idx.1.val)) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [lRowOffset, Nat.add_assoc] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, attention_fwd_triton3_end_m_formula_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.uop, Tile.ptrAdd, NumericDType.add, NumericDType.mul]
  simp only [lRowOffset, Nat.add_assoc]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [endMStoreSpec, lRowOffset, Nat.add_assoc]

theorem attention_fwd_triton3_end_m_formula_store_slice_compute_correct
    (MPre LPre M : RegionName) (off_hz ROUND_CTX BLOCK_M : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz ROUND_CTX BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_m_formula_store_slice MPre LPre M
        off_hz ROUND_CTX BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (M,
        lRowOffset s off_hz ROUND_CTX BLOCK_M i))
      (expected := fun i =>
        endMStoreSpec s MPre LPre off_hz ROUND_CTX BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_end_m_formula_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := attention_fwd_triton3_end_m_formula_store_slice_correct MPre LPre
    M off_hz ROUND_CTX BLOCK_M s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-! ## Python test-shape summaries

`attention_fwd_triton3.py`'s checked tests use
`q/k/v/o.shape = (2, 4, 128, 64)`, `BLOCK_M = BLOCK_N = 64`,
`ROUND_CTX = 128`, `H_KV = H = 4`, and `sm_scale = 1 / sqrt(64) = 1/8`.
All four test cases run with `END = true`; the observed `M` row therefore
comes from the formula epilogue `m_i += log2(l_i)`. -/

theorem attention_fwd_triton3_final_store_python_test_shape_compute_correct
    (Acc Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_final_store_slice Acc Out
        4 128 64 32768 8192 64 1 32768 8192 64 1 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        s.readMem Acc (accOffset s 4 32768 8192 64 1 64 idx)) := by
  apply attention_fwd_triton3_final_store_slice_compute_correct
  rintro ⟨⟨ma, hma⟩, ⟨ka, hka⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨kb, hkb⟩, _⟩ h
  simp [outOffset, offZ, offH, mIndex, kIndex] at h
  have hm : ma = mb := by omega
  have hk : ka = kb := by omega
  subst mb
  subst kb
  rfl

theorem attention_fwd_triton3_end_output_formula_python_test_shape_compute_correct
    (Acc LPre Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_output_formula_store_slice Acc LPre
        Out 4 128 64 32768 8192 64 1 32768 8192 64 1 128 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        endOutputStoreSpec s Acc LPre 4 32768 8192 64 1 128 64 idx) := by
  apply attention_fwd_triton3_end_output_formula_store_slice_compute_correct
  rintro ⟨⟨ma, hma⟩, ⟨ka, hka⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨kb, hkb⟩, _⟩ h
  simp [outOffset, offZ, offH, mIndex, kIndex] at h
  have hm : ma = mb := by omega
  have hk : ka = kb := by omega
  subst mb
  subst kb
  rfl

theorem attention_fwd_triton3_end_m_formula_python_test_shape_compute_correct
    (MPre LPre M : RegionName) (off_hz : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_m_formula_store_slice MPre LPre M
        off_hz 128 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s off_hz 128 64 i))
      (expected := fun i : Fin 64 =>
        endMStoreSpec s MPre LPre off_hz 128 64 i) := by
  apply attention_fwd_triton3_end_m_formula_store_slice_compute_correct
  intro a b h
  simp [lRowOffset] at h
  apply Fin.ext
  omega

theorem attention_fwd_triton3_python_case1_surface_toAlgorithm_supported
    (Q K V M Out L : RegionName) :
    ∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0).toAlgorithm? =
        Except.ok alg := by
  exact attention_fwd_triton3_surface_toAlgorithm_supported Q K V M Out L
    (1 / 8 : ℝ) 32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
    32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0

theorem attention_fwd_triton3_python_case2_surface_toAlgorithm_supported
    (Q K V M Out L : RegionName) :
    ∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 1).toAlgorithm? =
        Except.ok alg := by
  exact attention_fwd_triton3_surface_toAlgorithm_supported Q K V M Out L
    (1 / 8 : ℝ) 32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
    32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 1

theorem attention_fwd_triton3_python_case3_surface_toAlgorithm_supported
    (Q K V M Out L : RegionName) :
    ∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 0 1 1 64 64 64 1 1 0 0).toAlgorithm? =
        Except.ok alg := by
  exact attention_fwd_triton3_surface_toAlgorithm_supported Q K V M Out L
    (1 / 8 : ℝ) 32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
    32768 8192 64 1 2 4 4 128 128 128 0 0 1 1 64 64 64 1 1 0 0

theorem attention_fwd_triton3_python_case4_surface_toAlgorithm_supported
    (Q K V M Out L : RegionName) :
    ∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 64 1 1 64 64 64 1 0 1 0).toAlgorithm? =
        Except.ok alg := by
  exact attention_fwd_triton3_surface_toAlgorithm_supported Q K V M Out L
    (1 / 8 : ℝ) 32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
    32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 0 1 0

noncomputable def producedAttentionFwdTriton3Case1OutValue
    (s : BlockState) (Q K V M Out L : RegionName)
    (idx : TileIndex [64, 64]) : ℝ :=
  match exec (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0).toAlgKernel s with
  | some s' => s'.readMem Out (outOffset s 4 32768 8192 64 1 64 idx)
  | none => 0.0

noncomputable def producedAttentionFwdTriton3Case1MValue
    (s : BlockState) (Q K V M Out L : RegionName) (i : Fin 64) : ℝ :=
  match exec (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0).toAlgKernel s with
  | some s' => s'.readMem M (lRowOffset s (s.pids 1) 128 64 i)
  | none => 0.0

theorem attention_fwd_triton3_case1_surface_out_compute_correct
    (Q K V M Out L : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        producedAttentionFwdTriton3Case1OutValue s Q K V M Out L idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedAttentionFwdTriton3Case1OutValue]
  rw [inv_eq_one_div, hExec]

theorem attention_fwd_triton3_case1_surface_m_compute_correct
    (Q K V M Out L : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s (s.pids 1) 128 64 i))
      (expected := fun i : Fin 64 =>
        producedAttentionFwdTriton3Case1MValue s Q K V M Out L i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton3_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [producedAttentionFwdTriton3Case1MValue]
  rw [inv_eq_one_div, hExec]

theorem attention_fwd_triton3_python_case1_output_summary
    (Q K V M Out L : RegionName) (s : BlockState) :
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0
      ).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        producedAttentionFwdTriton3Case1OutValue s Q K V M Out L idx) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s (s.pids 1) 128 64 i))
      (expected := fun i : Fin 64 =>
        producedAttentionFwdTriton3Case1MValue s Q K V M Out L i) := by
  constructor
  · exact attention_fwd_triton3_python_case1_surface_toAlgorithm_supported
      Q K V M Out L
  constructor
  · exact attention_fwd_triton3_case1_surface_out_compute_correct Q K V M Out
      L s
  · exact attention_fwd_triton3_case1_surface_m_compute_correct Q K V M Out
      L s

theorem attention_fwd_triton3_python_case2_output_summary
    (Q K V M Out L Acc MPre LPre : RegionName) (off_hz : Nat) (s : BlockState) :
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 1
      ).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_final_store_slice Acc Out
        4 128 64 32768 8192 64 1 32768 8192 64 1 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        s.readMem Acc (accOffset s 4 32768 8192 64 1 64 idx)) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_m_formula_store_slice MPre LPre M
        off_hz 128 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s off_hz 128 64 i))
      (expected := fun i : Fin 64 =>
        endMStoreSpec s MPre LPre off_hz 128 64 i) := by
  constructor
  · exact attention_fwd_triton3_python_case2_surface_toAlgorithm_supported
      Q K V M Out L
  constructor
  · exact attention_fwd_triton3_final_store_python_test_shape_compute_correct
      Acc Out s
  · exact attention_fwd_triton3_end_m_formula_python_test_shape_compute_correct
      MPre LPre M off_hz s

theorem attention_fwd_triton3_python_case3_output_summary
    (Q K V M Out L Acc MPre LPre : RegionName) (off_hz : Nat) (s : BlockState) :
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 0 1 1 64 64 64 1 1 0 0
      ).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_final_store_slice Acc Out
        4 128 64 32768 8192 64 1 32768 8192 64 1 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        s.readMem Acc (accOffset s 4 32768 8192 64 1 64 idx)) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_m_formula_store_slice MPre LPre M
        off_hz 128 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s off_hz 128 64 i))
      (expected := fun i : Fin 64 =>
        endMStoreSpec s MPre LPre off_hz 128 64 i) := by
  constructor
  · exact attention_fwd_triton3_python_case3_surface_toAlgorithm_supported
      Q K V M Out L
  constructor
  · exact attention_fwd_triton3_final_store_python_test_shape_compute_correct
      Acc Out s
  · exact attention_fwd_triton3_end_m_formula_python_test_shape_compute_correct
      MPre LPre M off_hz s

theorem attention_fwd_triton3_python_case4_output_summary
    (Q K V M Out L Acc MPre LPre : RegionName) (off_hz : Nat) (s : BlockState) :
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 0 1 0
      ).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_final_store_slice Acc Out
        4 128 64 32768 8192 64 1 32768 8192 64 1 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        s.readMem Acc (accOffset s 4 32768 8192 64 1 64 idx)) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_m_formula_store_slice MPre LPre M
        off_hz 128 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s off_hz 128 64 i))
      (expected := fun i : Fin 64 =>
        endMStoreSpec s MPre LPre off_hz 128 64 i) := by
  constructor
  · exact attention_fwd_triton3_python_case4_surface_toAlgorithm_supported
      Q K V M Out L
  constructor
  · exact attention_fwd_triton3_final_store_python_test_shape_compute_correct
      Acc Out s
  · exact attention_fwd_triton3_end_m_formula_python_test_shape_compute_correct
      MPre LPre M off_hz s

/-- Strengthened checked-shape END epilogue summary: in addition to the
existing case summaries, expose the compute-correct `acc / l_i[:, None]`
producer for the final `Out` writeback used when `END=True`. -/
theorem attention_fwd_triton3_python_end_output_formula_summary
    (Acc LPre Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_output_formula_store_slice Acc LPre
        Out 4 128 64 32768 8192 64 1 32768 8192 64 1 128 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        endOutputStoreSpec s Acc LPre 4 32768 8192 64 1 128 64 idx) := by
  exact attention_fwd_triton3_end_output_formula_python_test_shape_compute_correct
    Acc LPre Out s

/-- Combined checked-shape summary for `test_forward` in `attention_fwd_triton3.py`.

This pins all four Python branch launches (`sliding_window`, complement window,
plain full-window, and `INIT=False`) and exposes the END epilogue arithmetic
that mutates the observable `Out` and `M` tensors. -/
theorem attention_fwd_triton3_python_test_shape_complete_summary
    (Q K V M Out L Acc MPre LPre : RegionName) (off_hz : Nat)
    (s : BlockState) :
    ((∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 0
      ).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 1 1 1
      ).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 0 1 1 64 64 64 1 1 0 0
      ).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (attention_fwd_triton3_surface Q K V M Out L (1 / 8 : ℝ)
      32768 8192 64 1 32768 8192 64 1 32768 8192 64 1
      32768 8192 64 1 2 4 4 128 128 128 0 64 1 1 64 64 64 1 0 1 0
      ).toAlgorithm? = Except.ok alg)) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_output_formula_store_slice Acc LPre
        Out 4 128 64 32768 8192 64 1 32768 8192 64 1 128 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 idx)
        (fun idx : TileIndex [64, 64] => (Out,
          outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        endOutputStoreSpec s Acc LPre 4 32768 8192 64 1 128 64 idx) ∧
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton3_end_m_formula_store_slice MPre LPre M
        off_hz 128 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (M, lRowOffset s off_hz 128 64 i))
      (expected := fun i : Fin 64 =>
        endMStoreSpec s MPre LPre off_hz 128 64 i) := by
  constructor
  · constructor
    · exact attention_fwd_triton3_python_case1_surface_toAlgorithm_supported
        Q K V M Out L
    constructor
    · exact attention_fwd_triton3_python_case2_surface_toAlgorithm_supported
        Q K V M Out L
    constructor
    · exact attention_fwd_triton3_python_case3_surface_toAlgorithm_supported
        Q K V M Out L
    · exact attention_fwd_triton3_python_case4_surface_toAlgorithm_supported
        Q K V M Out L
  constructor
  · exact attention_fwd_triton3_python_end_output_formula_summary Acc LPre Out s
  · exact attention_fwd_triton3_end_m_formula_python_test_shape_compute_correct
      MPre LPre M off_hz s

end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
