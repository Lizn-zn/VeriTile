import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.TritonAttention

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- DSL port of `triton_attention.py`'s `_fwd_kernel`. -/
def triton_attention_fwd_kernel
    (Q K V L M Out : RegionName)
    (sm_scale : ℝ)
    (_stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn stride_kk
      _stride_vz _stride_vh stride_vk stride_vn
      _stride_oz _stride_oh stride_om stride_on
      _Z _H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  stride_qh_2d = $(stride_qh) // $(stride_qm) // $(stride_qk)

  q_tile_ptr = tl.make_block_ptr(base=Q,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(off_hz * stride_qh_2d + start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  k_tile_ptr = tl.make_block_ptr(base=K,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_kn), $(stride_kk)),
    offsets=(off_hz * stride_qh_2d, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))
  v_tile_ptr = tl.make_block_ptr(base=V,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(off_hz * stride_qh_2d, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))
  out_tile_ptr = tl.make_block_ptr(base=Out,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)),
    offsets=(off_hz * stride_qh_2d + start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  q = tl.load(q_tile_ptr)

  for start_n in range($(0), (start_m + $(1)) * $(BLOCK_M), $(BLOCK_N)) {
    k = tl.load(k_tile_ptr, boundary_check=(0, 1))
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_tile_ptr, boundary_check=(0, 1))
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_tile_ptr = tl.advance(k_tile_ptr, [$(BLOCK_N), $(0)])
    v_tile_ptr = tl.advance(v_tile_ptr, [$(BLOCK_N), $(0)])
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  l_ptrs = L + off_hz * $(N_CTX) + offs_m
  m_ptrs = M + off_hz * $(N_CTX) + offs_m
  tl.store(l_ptrs, l_prev)
  tl.store(m_ptrs, m_prev)

  acc = (acc).to(tl.float16)
  tl.store(out_tile_ptr, acc, boundary_check=(0, 1))
}

/-- The full Python-shaped forward attention surface lowers to the algorithm
layer, including the streaming softmax loop and final L/M/O stores. -/
theorem triton_attention_fwd_kernel_toAlgorithm_supported
    (Q K V L M Out : RegionName)
    (sm_scale : ℝ)
    (_stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn stride_kk
      _stride_vz _stride_vh stride_vk stride_vn
      _stride_oz _stride_oh stride_om stride_on
      _Z _H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ∃ alg, (triton_attention_fwd_kernel Q K V L M Out sm_scale _stride_qz
      stride_qh stride_qm stride_qk _stride_kz _stride_kh stride_kn
      stride_kk _stride_vz _stride_vh stride_vk stride_vn _stride_oz
      _stride_oh stride_om stride_on _Z _H N_CTX D0 BLOCK_M BLOCK_DMODEL
      BLOCK_N).toAlgorithm? = Except.ok alg := by
  simp [triton_attention_fwd_kernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented forward output-store slice of
`triton_attention.py`'s `_fwd_kernel`.

The Python kernel writes `acc` through a block pointer with
`boundary_check=(0, 1)`. This slice spells the same write as explicit pointer
arithmetic and an explicit two-axis boundary mask. The inner `tl.float32`
streaming-softmax accumulator is outside this slice. -/
def triton_attention_forward_output_store_slice
    (Acc Out : RegionName) (hzRowOffset D0 stride_om stride_on BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] + $(hzRowOffset) < $(D0)) &
    (offs_d[None, :] < $(BLOCK_DMODEL))
  acc = tl.load(Acc + offs_m[:, None] * $(BLOCK_DMODEL) + offs_d[None, :],
    mask=mask, other=0.0)
  tl.store(Out + (offs_m[:, None] + $(hzRowOffset)) * $(stride_om) +
      offs_d[None, :] * $(stride_on), (acc).to(tl.float16), mask=mask)
}

def rowIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def active (s : BlockState) (hzRowOffset D0 BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  rowIndex s BLOCK_M idx.1 + hzRowOffset < D0

instance activeDecidable (s : BlockState) (hzRowOffset D0 BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (active s hzRowOffset D0 BLOCK_M idx) := by
  unfold active
  infer_instance

def accOffset (s : BlockState) (BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  rowIndex s BLOCK_M idx.1 * BLOCK_DMODEL + dIndex idx

def outOffset (s : BlockState) (hzRowOffset stride_om stride_on BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (rowIndex s BLOCK_M idx.1 + hzRowOffset) * stride_om + dIndex idx * stride_on

noncomputable def storeValue (s : BlockState) (Acc : RegionName)
    (hzRowOffset D0 BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s hzRowOffset D0 BLOCK_M idx then
      some (s.readMem Acc (accOffset s BLOCK_M BLOCK_DMODEL idx))
    else some (0.0 : ℝ))

private theorem foldl_writeMemTyped_fp16_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier TileDType.fp16)
    (mask : α → Bool) (o : Nat) (l : List α) :
    ∀ s : BlockState,
      (∀ k ∈ l, mask k = Bool.true → offsetFn k ≠ o) →
        ((l.foldl
          (fun acc k =>
            if mask k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc)
          s).mem region o) = s.mem region o := by
  induction l with
  | nil =>
      intro s _h
      rfl
  | cons hd tl ih =>
      intro s h
      rw [List.foldl_cons]
      have htl : ∀ k ∈ tl, mask k = Bool.true → offsetFn k ≠ o :=
        fun k hk hmk => h k (List.mem_cons_of_mem hd hk) hmk
      by_cases hmaskhd : mask hd = Bool.true
      · have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self) hmaskhd
        simp only [hmaskhd, if_true]
        rw [ih _ htl]
        unfold BlockState.writeMemTyped BlockState.writeMemAs
        change
          (if region = region ∧ o = offsetFn hd then
            MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn hd)))
          else
            s.mem region o) = s.mem region o
        rw [if_neg (by
          intro hsame
          exact hhd hsame.2.symm)]
      · have hmaskhd' : mask hd = Bool.false := by
          cases hm : mask hd
          · rfl
          · exact False.elim (hmaskhd hm)
        simp only [hmaskhd', if_false, Bool.false_eq_true]
        exact ih _ htl

private theorem scatter_memcell_fp16_prop_masked_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier TileDType.fp16)
    (P : TileIndex shape → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc)
       s).mem region (offsetFn i)
    = if P i then
        MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
      else
        s.mem region (offsetFn i) := by
  let l := TileShape.allIndices shape
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  change ((l.foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc)
       s).mem region (offsetFn i))
    = if P i then
        MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
      else
        s.mem region (offsetFn i)
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  have hl' : l = l₁ ++ i :: l₂ := by
    simpa [l] using hl
  rw [hl', List.foldl_append, List.foldl_cons]
  have h_l1_not_in : ∀ k ∈ l₁, decide (P k) = Bool.true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    rw [hki] at hk
    exact (hl1_disj i hk i (List.mem_cons_self)) rfl
  have h_l2_not_in : ∀ k ∈ l₂, decide (P k) = Bool.true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    subst hki
    exact hi_notin_l2 hk
  have hstep :
      (fun (acc : BlockState) k =>
        if P k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc)
        =
      (fun (acc : BlockState) k =>
        if decide (P k) then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc) := by
    funext acc k
    by_cases hk : P k <;> simp [hk]
  rw [hstep]
  rw [foldl_writeMemTyped_fp16_preserves offsetFn valueFn (fun k => decide (P k))
    (offsetFn i) l₂ _ h_l2_not_in]
  by_cases hPi : P i
  · simp only [hPi, if_true]
    unfold BlockState.writeMemTyped BlockState.writeMemAs
    simp
  · simp only [hPi, if_false]
    rw [foldl_writeMemTyped_fp16_preserves offsetFn valueFn (fun k => decide (P k))
      (offsetFn i) l₁]
    exact h_l1_not_in

theorem triton_attention_forward_output_store_slice_correct
    (Acc Out : RegionName) (hzRowOffset D0 stride_om stride_on BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s hzRowOffset stride_om stride_on BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s hzRowOffset stride_om stride_on BLOCK_M idx
      (exec (triton_attention_forward_output_store_slice Acc Out hzRowOffset D0
            stride_om stride_on BLOCK_M BLOCK_DMODEL) s).map
          (·.mem Out outAddr)
        = some (if active s hzRowOffset D0 BLOCK_M idx then
            MemCell.of .fp16
              (FloatDType.real.cast FloatDType.fp16
                (some (storeValue s Acc hzRowOffset D0 BLOCK_M BLOCK_DMODEL idx)))
          else s.mem Out outAddr) := by
  intro idx
  simp [exec, triton_attention_forward_output_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.expandDim, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        rowIndex, dIndex, active, accOffset, outOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      (s.pids 0 * BLOCK_M + idx.1.val + hzRowOffset) * stride_om +
        idx.2.1.val * stride_on
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → TileCarrier TileDType.fp16 :=
    fun idx =>
      FloatDType.real.cast FloatDType.fp16
        (some (WithBot.unbotD 0
          (if s.pids 0 * BLOCK_M + idx.1.val + hzRowOffset < D0 then
            some (s.readMem Acc
              ((s.pids 0 * BLOCK_M + idx.1.val) * BLOCK_DMODEL + idx.2.1.val))
          else some (0.0 : ℝ))))
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx => s.pids 0 * BLOCK_M + idx.1.val + hzRowOffset < D0
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, rowIndex, dIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMemTyped .fp16 Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).mem Out
        (offsetFn idx) =
    if P idx then
      MemCell.of .fp16
        (FloatDType.real.cast FloatDType.fp16
          (some (storeValue s Acc hzRowOffset D0 BLOCK_M BLOCK_DMODEL idx)))
    else s.mem Out (offsetFn idx)
  rw [scatter_memcell_fp16_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 0 * BLOCK_M + idx.1.val + hzRowOffset < D0
  · rfl
  · rfl

theorem triton_attention_forward_output_store_slice_compute_correct
    (Acc Out : RegionName) (hzRowOffset D0 stride_om stride_on BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s hzRowOffset stride_om stride_on BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_output_store_slice Acc Out hzRowOffset D0
        stride_om stride_on BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s hzRowOffset D0 BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, outOffset s hzRowOffset stride_om stride_on BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (storeValue s Acc hzRowOffset D0 BLOCK_M BLOCK_DMODEL idx)))) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_forward_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := triton_attention_forward_output_store_slice_correct Acc Out
    hzRowOffset D0 stride_om stride_on BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented L (log-sum-exp) row store slice of `triton_attention.py`'s
forward kernel. Writes a precomputed `LPrev` vector into `L` at the per-row
`off_hz * N_CTX + offs_m` strided offset. Companion to the output store
slice. -/
def triton_attention_forward_l_store_slice
    (LPrev L : RegionName) (off_hz N_CTX BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  l_prev = tl.load(LPrev + $(off_hz) * $(N_CTX) + offs_m)
  tl.store(L + $(off_hz) * $(N_CTX) + offs_m, l_prev)
}

def lRowOffset (s : BlockState) (off_hz N_CTX BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  off_hz * N_CTX + (s.pids 0 * BLOCK_M + i.val)

noncomputable def lStoreSpec (s : BlockState) (LPrev : RegionName)
    (off_hz N_CTX BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem LPrev (lRowOffset s off_hz N_CTX BLOCK_M i)

theorem triton_attention_forward_l_store_slice_correct
    (LPrev L : RegionName) (off_hz N_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz N_CTX BLOCK_M i)) :
    ∀ i : Fin BLOCK_M,
      let outAddr := lRowOffset s off_hz N_CTX BLOCK_M i
      (exec (triton_attention_forward_l_store_slice LPrev L off_hz N_CTX BLOCK_M)
          s).map (·.readMem L outAddr)
        = some (lStoreSpec s LPrev off_hz N_CTX BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] =>
        off_hz * N_CTX + (s.pids 0 * BLOCK_M + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : lRowOffset s off_hz N_CTX BLOCK_M a =
        lRowOffset s off_hz N_CTX BLOCK_M b := by
      simpa [lRowOffset, Nat.add_assoc] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  simp [exec, triton_attention_forward_l_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul]
  simp only [lRowOffset, Nat.add_assoc]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [lStoreSpec, lRowOffset, Nat.add_assoc]

theorem triton_attention_forward_l_store_slice_compute_correct
    (LPrev L : RegionName) (off_hz N_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz N_CTX BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_l_store_slice LPrev L off_hz N_CTX BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lRowOffset s off_hz N_CTX BLOCK_M i))
      (expected := fun i => lStoreSpec s LPrev off_hz N_CTX BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_forward_l_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := triton_attention_forward_l_store_slice_correct LPrev L
    off_hz N_CTX BLOCK_M s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-- Proof-oriented M (max) row store slice of `triton_attention.py`'s forward
kernel. Mirrors the L-row store slice. -/
def triton_attention_forward_m_store_slice
    (MPrev M : RegionName) (off_hz N_CTX BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  m_prev = tl.load(MPrev + $(off_hz) * $(N_CTX) + offs_m)
  tl.store(M + $(off_hz) * $(N_CTX) + offs_m, m_prev)
}

noncomputable def mStoreSpec (s : BlockState) (MPrev : RegionName)
    (off_hz N_CTX BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem MPrev (lRowOffset s off_hz N_CTX BLOCK_M i)

theorem triton_attention_forward_m_store_slice_correct
    (MPrev M : RegionName) (off_hz N_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz N_CTX BLOCK_M i)) :
    ∀ i : Fin BLOCK_M,
      let outAddr := lRowOffset s off_hz N_CTX BLOCK_M i
      (exec (triton_attention_forward_m_store_slice MPrev M off_hz N_CTX BLOCK_M)
          s).map (·.readMem M outAddr)
        = some (mStoreSpec s MPrev off_hz N_CTX BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] =>
        off_hz * N_CTX + (s.pids 0 * BLOCK_M + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : lRowOffset s off_hz N_CTX BLOCK_M a =
        lRowOffset s off_hz N_CTX BLOCK_M b := by
      simpa [lRowOffset, Nat.add_assoc] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  simp [exec, triton_attention_forward_m_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul]
  simp only [lRowOffset, Nat.add_assoc]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [mStoreSpec, lRowOffset, Nat.add_assoc]

theorem triton_attention_forward_m_store_slice_compute_correct
    (MPrev M : RegionName) (off_hz N_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz N_CTX BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_m_store_slice MPrev M off_hz N_CTX BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (M, lRowOffset s off_hz N_CTX BLOCK_M i))
      (expected := fun i => mStoreSpec s MPrev off_hz N_CTX BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_forward_m_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := triton_attention_forward_m_store_slice_correct MPrev M
    off_hz N_CTX BLOCK_M s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-! ### Auxiliary backward-preprocess slices

The `_bwd_preprocess` kernel in `triton_attention.py` is much simpler than the
forward / backward main kernels: it loads `O`, `DO`, `L`, computes
`do = do / L[:, None]`, then writes `do` to `NewDO` and `sum(o * do, axis=1)`
to `Delta`. The streaming softmax / tl.dot / make_block_ptr / advance pieces
that block the main attention loop are absent. The two store-back regions
support clean proof-oriented slices analogous to the forward `L`/`M` row store
slices already in this file.
-/

/-- DSL port of `triton_attention.py`'s `_bwd_preprocess`. -/
def triton_attention_bwd_preprocess
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat) :
    ComputeKernel := triton {
  off_m = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(D_HEAD))
  o = (tl.load(Out + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  do_val = (tl.load(DO + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  denom = (tl.load(L + off_m)).to(tl.float32)
  do_val = do_val / denom[:, None]
  delta = tl.sum(o * do_val, axis=1)
  tl.store(NewDO + off_m[:, None] * $(D_HEAD) + off_n[None, :], do_val)
  tl.store(Delta + off_m, delta)
}

theorem triton_attention_bwd_preprocess_toAlgorithm_supported
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat) :
    ∃ alg, (triton_attention_bwd_preprocess Out DO L NewDO Delta
      BLOCK_M D_HEAD).toAlgorithm? = Except.ok alg := by
  simp [triton_attention_bwd_preprocess, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented `NewDO` 2D store slice of `triton_attention.py`'s
`_bwd_preprocess`. The kernel stores a (precomputed) `NewDOAcc` tile to
`NewDO` at strided offset `off_m[:, None] * D_HEAD + off_n[None, :]`. -/
def triton_attention_bwd_preprocess_newdo_store_slice
    (NewDOAcc NewDO : RegionName) (BLOCK_M D_HEAD : Nat) :
    ComputeKernel := triton {
  off_m = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(D_HEAD))
  do_val = tl.load(NewDOAcc + off_m[:, None] * $(D_HEAD) + off_n[None, :])
  tl.store(NewDO + off_m[:, None] * $(D_HEAD) + off_n[None, :], do_val)
}

def newdoMIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def newdoNIndex (idx : TileIndex [BLOCK_M, D_HEAD]) : Nat :=
  idx.2.1.val

def newdoOffset (s : BlockState) (BLOCK_M D_HEAD : Nat)
    (idx : TileIndex [BLOCK_M, D_HEAD]) : Nat :=
  newdoMIndex s BLOCK_M idx.1 * D_HEAD + newdoNIndex idx

noncomputable def newdoStoreSpec (s : BlockState) (NewDOAcc : RegionName)
    (BLOCK_M D_HEAD : Nat) (idx : TileIndex [BLOCK_M, D_HEAD]) : ℝ :=
  s.readMem NewDOAcc (newdoOffset s BLOCK_M D_HEAD idx)

theorem triton_attention_bwd_preprocess_newdo_store_slice_correct
    (NewDOAcc NewDO : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    ∀ idx : TileIndex [BLOCK_M, D_HEAD],
      let outAddr := newdoOffset s BLOCK_M D_HEAD idx
      (exec (triton_attention_bwd_preprocess_newdo_store_slice
            NewDOAcc NewDO BLOCK_M D_HEAD) s).map (·.readMem NewDO outAddr)
        = some (newdoStoreSpec s NewDOAcc BLOCK_M D_HEAD idx) := by
  intro idx
  simp [exec, triton_attention_bwd_preprocess_newdo_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        newdoOffset, newdoMIndex, newdoNIndex, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, D_HEAD] → Nat :=
    fun idx => (s.pids 0 * BLOCK_M + idx.1.val) * D_HEAD + idx.2.1.val
  let valueFn : TileIndex [BLOCK_M, D_HEAD] → ℝ :=
    fun idx => s.readMem NewDOAcc
      ((s.pids 0 * BLOCK_M + idx.1.val) * D_HEAD + idx.2.1.val)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, newdoOffset, newdoMIndex, newdoNIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        acc.writeMem NewDO (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, D_HEAD])).readMem NewDO
        (offsetFn idx) =
    newdoStoreSpec s NewDOAcc BLOCK_M D_HEAD idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [newdoStoreSpec, newdoOffset, newdoMIndex, newdoNIndex,
    offsetFn, valueFn]

theorem triton_attention_bwd_preprocess_newdo_store_slice_compute_correct
    (NewDOAcc NewDO : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_newdo_store_slice
        NewDOAcc NewDO BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        some (NewDO, newdoOffset s BLOCK_M D_HEAD idx))
      (expected := fun idx => newdoStoreSpec s NewDOAcc BLOCK_M D_HEAD idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess_newdo_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := triton_attention_bwd_preprocess_newdo_store_slice_correct
    NewDOAcc NewDO BLOCK_M D_HEAD s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Formula-level `NewDO` slice of `triton_attention.py`'s `_bwd_preprocess`.
It covers the Python arithmetic

`do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)`
`denom = tl.load(L + off_m).to(tl.float32)`
`do = do / denom[:, None]`
`tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)`.
-/
def triton_attention_bwd_preprocess_newdo_formula_slice
    (DO L NewDO : RegionName) (BLOCK_M D_HEAD : Nat) :
    ComputeKernel := triton {
  off_m = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(D_HEAD))
  do_val = (tl.load(DO + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  denom = (tl.load(L + off_m)).to(tl.float32)
  new_do = do_val / denom[:, None]
  tl.store(NewDO + off_m[:, None] * $(D_HEAD) + off_n[None, :], new_do)
}

noncomputable def newdoFormulaSpec (s : BlockState) (DO L : RegionName)
    (BLOCK_M D_HEAD : Nat) (idx : TileIndex [BLOCK_M, D_HEAD]) : ℝ :=
  s.readMem DO (newdoOffset s BLOCK_M D_HEAD idx) /
    s.readMem L (newdoMIndex s BLOCK_M idx.1)

theorem triton_attention_bwd_preprocess_newdo_formula_slice_correct
    (DO L NewDO : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    ∀ idx : TileIndex [BLOCK_M, D_HEAD],
      let outAddr := newdoOffset s BLOCK_M D_HEAD idx
      (exec (triton_attention_bwd_preprocess_newdo_formula_slice
            DO L NewDO BLOCK_M D_HEAD) s).map (·.readMem NewDO outAddr)
        = some (newdoFormulaSpec s DO L BLOCK_M D_HEAD idx) := by
  intro idx
  simp [exec, triton_attention_bwd_preprocess_newdo_formula_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        NumericDType.div, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, newdoOffset, newdoMIndex, newdoNIndex,
        TileShape.dropInsertedIndex, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BLOCK_M, D_HEAD] → Nat :=
    fun idx => (s.pids 0 * BLOCK_M + idx.1.val) * D_HEAD + idx.2.1.val
  let valueFn : TileIndex [BLOCK_M, D_HEAD] → ℝ :=
    fun idx => s.readMem DO (offsetFn idx) /
      s.readMem L (s.pids 0 * BLOCK_M + idx.1.val)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, newdoOffset, newdoMIndex, newdoNIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        acc.writeMem NewDO (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, D_HEAD])).readMem NewDO
        (offsetFn idx) =
    newdoFormulaSpec s DO L BLOCK_M D_HEAD idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [newdoFormulaSpec, newdoOffset, newdoMIndex, newdoNIndex,
    offsetFn, valueFn]

theorem triton_attention_bwd_preprocess_newdo_formula_slice_compute_correct
    (DO L NewDO : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_newdo_formula_slice
        DO L NewDO BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        some (NewDO, newdoOffset s BLOCK_M D_HEAD idx))
      (expected := fun idx => newdoFormulaSpec s DO L BLOCK_M D_HEAD idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess_newdo_formula_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := triton_attention_bwd_preprocess_newdo_formula_slice_correct
    DO L NewDO BLOCK_M D_HEAD s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Proof-oriented `Delta` 1D row store slice of `triton_attention.py`'s
`_bwd_preprocess`. Mirrors the L-row store slice of the forward kernel:
load a (precomputed) `DeltaAcc` row vector and write it to `Delta` at
`off_m`. -/
def triton_attention_bwd_preprocess_delta_store_slice
    (DeltaAcc Delta : RegionName) (BLOCK_M : Nat) :
    ComputeKernel := triton {
  off_m = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  delta_val = tl.load(DeltaAcc + off_m)
  tl.store(Delta + off_m, delta_val)
}

def deltaOffset (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

noncomputable def deltaStoreSpec (s : BlockState) (DeltaAcc : RegionName)
    (BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem DeltaAcc (deltaOffset s BLOCK_M i)

/-- Formula-level `Delta` slice of `triton_attention.py`'s `_bwd_preprocess`.
It covers `do = do / L[:, None]` followed by
`delta = tl.sum(o * do, axis=1)` and the row store to `Delta`. -/
def triton_attention_bwd_preprocess_delta_formula_slice
    (Out DO L Delta : RegionName) (BLOCK_M D_HEAD : Nat) :
    ComputeKernel := triton {
  off_m = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(D_HEAD))
  o = (tl.load(Out + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  do_val = (tl.load(DO + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  denom = (tl.load(L + off_m)).to(tl.float32)
  new_do = do_val / denom[:, None]
  delta = tl.sum(o * new_do, axis=1)
  tl.store(Delta + off_m, delta)
}

noncomputable def deltaFormulaSpec (s : BlockState) (Out DO L : RegionName)
    (BLOCK_M D_HEAD : Nat) (i : Fin BLOCK_M) : ℝ :=
  ∑ j : Fin D_HEAD,
    let idx : TileIndex [BLOCK_M, D_HEAD] :=
      TileShape.insertAxisIndex [BLOCK_M, D_HEAD] 1
        (TileShape.insertAxisIndex [BLOCK_M] 0 PUnit.unit i) j
    s.readMem Out (newdoOffset s BLOCK_M D_HEAD idx) *
      newdoFormulaSpec s DO L BLOCK_M D_HEAD idx

theorem triton_attention_bwd_preprocess_delta_formula_slice_correct
    (Out DO L Delta : RegionName) (BLOCK_M D_HEAD : Nat)
    (s s' : BlockState)
    (hExec : exec (triton_attention_bwd_preprocess_delta_formula_slice
        Out DO L Delta BLOCK_M D_HEAD) s = some s') :
    ∀ i : Fin BLOCK_M,
      s'.readMem Delta (deltaOffset s BLOCK_M i) =
        deltaFormulaSpec s Out DO L BLOCK_M D_HEAD i := by
  intro i
  simp [exec, triton_attention_bwd_preprocess_delta_formula_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add,
        NumericDType.mul, NumericDType.div, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  rw [← hExec]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] => s.pids 0 * BLOCK_M + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [deltaOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [deltaFormulaSpec, newdoFormulaSpec, newdoOffset, newdoMIndex,
    newdoNIndex, deltaOffset, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, NumericDType.mul, NumericDType.div]
  congr

theorem triton_attention_bwd_preprocess_delta_formula_slice_compute_correct
    (Out DO L Delta : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_delta_formula_slice
        Out DO L Delta BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (Delta, deltaOffset s BLOCK_M i))
      (expected := fun i => deltaFormulaSpec s Out DO L BLOCK_M D_HEAD i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess_delta_formula_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact triton_attention_bwd_preprocess_delta_formula_slice_correct Out DO L
    Delta BLOCK_M D_HEAD s s' hExec i

theorem triton_attention_bwd_preprocess_delta_store_slice_correct
    (DeltaAcc Delta : RegionName) (BLOCK_M : Nat) (s : BlockState) :
    ∀ i : Fin BLOCK_M,
      let outAddr := deltaOffset s BLOCK_M i
      (exec (triton_attention_bwd_preprocess_delta_store_slice
            DeltaAcc Delta BLOCK_M) s).map (·.readMem Delta outAddr)
        = some (deltaStoreSpec s DeltaAcc BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] => s.pids 0 * BLOCK_M + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, triton_attention_bwd_preprocess_delta_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul]
  simp only [deltaOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [deltaStoreSpec, deltaOffset]

theorem triton_attention_bwd_preprocess_delta_store_slice_compute_correct
    (DeltaAcc Delta : RegionName) (BLOCK_M : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_delta_store_slice
        DeltaAcc Delta BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (Delta, deltaOffset s BLOCK_M i))
      (expected := fun i => deltaStoreSpec s DeltaAcc BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess_delta_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := triton_attention_bwd_preprocess_delta_store_slice_correct
    DeltaAcc Delta BLOCK_M s i
  rw [hExec] at h
  exact Option.some.inj h

/-! ### Main backward gradient store slices

The main `_bwd_kernel` accumulates `dq`, `dk`, and `dv` through nested dot
loops. The slices below start after those accumulators have been materialized
and cover the Python-observed gradient writebacks. `DQ` is stored without a
boundary check in the source kernel; `DK` and `DV` use block-pointer
`boundary_check=(0, 1)`.
-/

def triton_attention_bwd_dq_store_slice
    (DQPre DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  off_hz = tl.program_id(0)
  block = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_m = block * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  dq = tl.load(DQPre + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk))
  tl.store(DQ + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk), dq)
}

def triton_attention_bwd_dkdv_store_slice
    (GradPre Out : RegionName)
    (H D0 stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  off_hz = tl.program_id(0)
  block = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_m = block * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < $(D0)) & (offs_k[None, :] < $(BLOCK_DMODEL))
  grad = tl.load(GradPre + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk))
  tl.store(Out + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk),
      (grad).to(Out.dtype.element_ty), mask=mask)
}

def bwdOffZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 0 / H

def bwdOffH (s : BlockState) (H : Nat) : Nat :=
  s.pids 0 % H

def bwdRowIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * BLOCK_M + i.val

def bwdColIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def bwdGradOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  bwdOffZ s H * stride_qz + bwdOffH s H * stride_qh +
    bwdRowIndex s BLOCK_M idx.1 * stride_qm + bwdColIndex idx * stride_qk

def bwdGradActive (s : BlockState) (D0 BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  bwdRowIndex s BLOCK_M idx.1 < D0

instance bwdGradActiveDecidable
    (s : BlockState) (D0 BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (bwdGradActive s D0 BLOCK_M idx) := by
  unfold bwdGradActive
  infer_instance

noncomputable def bwdGradStoreSpec
    (s : BlockState) (GradPre : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  s.readMem GradPre
    (bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)

/-- One inner-loop DQ update slice from `_bwd_kernel`:
`dq += tl.dot(ds.to(tl.float16), k)` followed by the DQ tile store. The
precomputed `DS` and `KTile` regions stand for the source kernel's `ds` tile
and loaded `k` tile at one loop step. -/
def triton_attention_bwd_dq_dot_step_slice
    (DQPrev DS KTile DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  off_hz = tl.program_id(0)
  block = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_m = block * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  dq = tl.load(DQPrev + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk))
  ds = (tl.load(DS + offs_m[:, None] * $(BLOCK_M) + offs_n[None, :])).to(tl.float16)
  k = tl.load(KTile + offs_n[:, None] * $(BLOCK_DMODEL) + offs_k[None, :])
  dq += tl.dot(ds, k)
  tl.store(DQ + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk), dq)
}

def bwdDsOffset (s : BlockState) (BLOCK_M : Nat)
    (row : Fin BLOCK_M) (k : Fin BLOCK_M) : Nat :=
  bwdRowIndex s BLOCK_M row * BLOCK_M + k.val

def bwdKTileOffset (BLOCK_DMODEL : Nat) (k : Fin BLOCK_M)
    (col : Fin BLOCK_DMODEL) : Nat :=
  k.val * BLOCK_DMODEL + col.val

noncomputable def bwdDqDotStepSpec
    (s : BlockState) (DQPrev DS KTile : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  bwdGradStoreSpec s DQPrev H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M idx +
    ∑ k : Fin BLOCK_M,
      FloatDType.fp16.storeValue
        (FloatDType.real.cast FloatDType.fp16
          (some (s.readMem DS (bwdDsOffset s BLOCK_M idx.1 k)))) *
        s.readMem KTile (bwdKTileOffset BLOCK_DMODEL k idx.2.1)

theorem triton_attention_bwd_dq_dot_step_slice_correct
    (DQPrev DS KTile DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr :=
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx
      (exec (triton_attention_bwd_dq_dot_step_slice DQPrev DS KTile DQ H
            stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
          s).map (·.readMem DQ outAddr)
        = some (bwdDqDotStepSpec s DQPrev DS KTile H stride_qz stride_qh
            stride_qm stride_qk BLOCK_M idx) := by
  intro idx
  simp [exec, triton_attention_bwd_dq_dot_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, Tile.dot, NumericDType.add, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot, bwdOffZ, bwdOffH,
        bwdRowIndex, bwdColIndex, bwdGradOffset, bwdGradStoreSpec,
        bwdDsOffset, bwdKTileOffset, TileShape.dropInsertedIndex,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 0 / H * stride_qz + s.pids 0 % H * stride_qh +
        (s.pids 1 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      s.readMem DQPrev (offsetFn idx) +
        ∑ k : Fin BLOCK_M,
          FloatDType.fp16.storeValue
            (FloatDType.real.cast FloatDType.fp16
              (some (s.readMem DS
                ((s.pids 1 * BLOCK_M + idx.1.val) * BLOCK_M + k.val)))) *
            s.readMem KTile (k.val * BLOCK_DMODEL + idx.2.1.val)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem DQ (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem DQ
        (offsetFn idx) =
    bwdDqDotStepSpec s DQPrev DS KTile H stride_qz stride_qh stride_qm
      stride_qk BLOCK_M idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [bwdDqDotStepSpec, bwdGradStoreSpec, bwdGradOffset, bwdOffZ, bwdOffH,
    bwdRowIndex, bwdColIndex, bwdDsOffset, bwdKTileOffset, offsetFn, valueFn]

theorem triton_attention_bwd_dq_dot_step_slice_compute_correct
    (DQPrev DS KTile DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dq_dot_step_slice DQPrev DS KTile DQ H
        stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (DQ, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdDqDotStepSpec s DQPrev DS KTile H stride_qz stride_qh stride_qm
          stride_qk BLOCK_M idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_dq_dot_step_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := triton_attention_bwd_dq_dot_step_slice_correct DQPrev DS KTile DQ
    H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s
    hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Shared inner-loop transpose-dot update slice for `_bwd_kernel`'s DK/DV
accumulators. It covers both Python paths
`dv += tl.dot(tl.trans(p.to(tl.float16)), do)` and
`dk += tl.dot(tl.trans(ds.to(tl.float16)), q)` by parameterizing the left and
right tiles and the query-row block participating in this loop step. -/
def triton_attention_bwd_trans_dot_step_slice
    (AccPrev LeftTile RightTile Out : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  off_hz = tl.program_id(0)
  block = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_out = block * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_query = $(queryBlock) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  acc = tl.load(AccPrev + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_out[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk))
  left = (tl.load(LeftTile +
      offs_query[:, None] * $(BLOCK_M) + offs_out[None, :])).to(tl.float16)
  right = tl.load(RightTile +
      offs_query[:, None] * $(BLOCK_DMODEL) + offs_k[None, :])
  acc += tl.dot(tl.trans(left), right)
  tl.store(Out + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_out[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk), acc)
}

def bwdQueryIndex (queryBlock BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  queryBlock * BLOCK_M + i.val

def bwdLeftTileOffset (s : BlockState) (queryBlock BLOCK_M : Nat)
    (query : Fin BLOCK_M) (outRow : Fin BLOCK_M) : Nat :=
  bwdQueryIndex queryBlock BLOCK_M query * BLOCK_M +
    bwdRowIndex s BLOCK_M outRow

def bwdRightTileOffset (queryBlock BLOCK_M BLOCK_DMODEL : Nat)
    (query : Fin BLOCK_M) (col : Fin BLOCK_DMODEL) : Nat :=
  bwdQueryIndex queryBlock BLOCK_M query * BLOCK_DMODEL + col.val

noncomputable def bwdTransDotStepSpec
    (s : BlockState) (AccPrev LeftTile RightTile : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  bwdGradStoreSpec s AccPrev H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M idx +
    ∑ query : Fin BLOCK_M,
      s.readMem LeftTile
          (bwdLeftTileOffset s queryBlock BLOCK_M query idx.1) *
        s.readMem RightTile
          (bwdRightTileOffset queryBlock BLOCK_M BLOCK_DMODEL query idx.2.1)

theorem triton_attention_bwd_trans_dot_step_slice_correct
    (AccPrev LeftTile RightTile Out : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr :=
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx
      (exec (triton_attention_bwd_trans_dot_step_slice AccPrev LeftTile
            RightTile Out queryBlock H stride_qz stride_qh stride_qm stride_qk
            BLOCK_M BLOCK_DMODEL) s).map (·.readMem Out outAddr)
        = some (bwdTransDotStepSpec s AccPrev LeftTile RightTile queryBlock H
            stride_qz stride_qh stride_qm stride_qk BLOCK_M idx) := by
  intro idx
  simp [exec, triton_attention_bwd_trans_dot_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, Tile.dot, Tile.transpose, NumericDType.add,
        NumericDType.mul, IntegralDType.floorDiv, IntegralDType.mod,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        bwdOffZ, bwdOffH, bwdRowIndex, bwdColIndex, bwdGradOffset,
        bwdGradStoreSpec, bwdQueryIndex, bwdLeftTileOffset,
        bwdRightTileOffset, TileShape.dropInsertedIndex,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 0 / H * stride_qz + s.pids 0 % H * stride_qh +
        (s.pids 1 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      s.readMem AccPrev (offsetFn idx) +
        ∑ query : Fin BLOCK_M,
          s.readMem LeftTile
              ((queryBlock * BLOCK_M + query.val) * BLOCK_M +
                (s.pids 1 * BLOCK_M + idx.1.val)) *
            s.readMem RightTile
              ((queryBlock * BLOCK_M + query.val) * BLOCK_DMODEL + idx.2.1.val)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem Out (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    bwdTransDotStepSpec s AccPrev LeftTile RightTile queryBlock H stride_qz
      stride_qh stride_qm stride_qk BLOCK_M idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [bwdTransDotStepSpec, bwdGradStoreSpec, bwdGradOffset, bwdOffZ,
    bwdOffH, bwdRowIndex, bwdColIndex, bwdQueryIndex, bwdLeftTileOffset,
    bwdRightTileOffset, offsetFn, valueFn]

theorem triton_attention_bwd_trans_dot_step_slice_compute_correct
    (AccPrev LeftTile RightTile Out : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_trans_dot_step_slice AccPrev LeftTile
        RightTile Out queryBlock H stride_qz stride_qh stride_qm stride_qk
        BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (Out, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdTransDotStepSpec s AccPrev LeftTile RightTile queryBlock H stride_qz
          stride_qh stride_qm stride_qk BLOCK_M idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_trans_dot_step_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := triton_attention_bwd_trans_dot_step_slice_correct AccPrev LeftTile
    RightTile Out queryBlock H stride_qz stride_qh stride_qm stride_qk
    BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

theorem triton_attention_bwd_dv_dot_step_slice_compute_correct
    (DVPrev PTile DOTile DV : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_trans_dot_step_slice DVPrev PTile DOTile
        DV queryBlock H stride_qz stride_qh stride_qm stride_qk BLOCK_M
        BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (DV, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdTransDotStepSpec s DVPrev PTile DOTile queryBlock H stride_qz
          stride_qh stride_qm stride_qk BLOCK_M idx) := by
  exact triton_attention_bwd_trans_dot_step_slice_compute_correct DVPrev PTile
    DOTile DV queryBlock H stride_qz stride_qh stride_qm stride_qk BLOCK_M
    BLOCK_DMODEL s hOutInj

theorem triton_attention_bwd_dk_dot_step_slice_compute_correct
    (DKPrev DSTile QTile DK : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_trans_dot_step_slice DKPrev DSTile QTile
        DK queryBlock H stride_qz stride_qh stride_qm stride_qk BLOCK_M
        BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (DK, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdTransDotStepSpec s DKPrev DSTile QTile queryBlock H stride_qz
          stride_qh stride_qm stride_qk BLOCK_M idx) := by
  exact triton_attention_bwd_trans_dot_step_slice_compute_correct DKPrev DSTile
    QTile DK queryBlock H stride_qz stride_qh stride_qm stride_qk BLOCK_M
    BLOCK_DMODEL s hOutInj

theorem triton_attention_bwd_dq_store_slice_correct
    (DQPre DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr :=
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx
      (exec (triton_attention_bwd_dq_store_slice DQPre DQ H stride_qz
            stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem DQ outAddr)
        = some (bwdGradStoreSpec s DQPre H stride_qz stride_qh stride_qm
            stride_qk BLOCK_M idx) := by
  intro idx
  simp [exec, triton_attention_bwd_dq_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, bwdOffZ, bwdOffH,
        bwdRowIndex, bwdColIndex, bwdGradOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 0 / H * stride_qz + s.pids 0 % H * stride_qh +
        (s.pids 1 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx => s.readMem DQPre (offsetFn idx)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem DQ (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem DQ
        (offsetFn idx) =
    bwdGradStoreSpec s DQPre H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [bwdGradStoreSpec, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
    bwdColIndex, offsetFn, valueFn]

theorem triton_attention_bwd_dq_store_slice_compute_correct
    (DQPre DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dq_store_slice DQPre DQ H stride_qz
        stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (DQ, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradStoreSpec s DQPre H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_dq_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := triton_attention_bwd_dq_store_slice_correct DQPre DQ H stride_qz
    stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

theorem triton_attention_bwd_dkdv_store_slice_correct
    (GradPre Out : RegionName)
    (H D0 stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr :=
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx
      (exec (triton_attention_bwd_dkdv_store_slice GradPre Out H D0 stride_qz
            stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (if bwdGradActive s D0 BLOCK_M idx then
            bwdGradStoreSpec s GradPre H stride_qz stride_qh stride_qm
              stride_qk BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, triton_attention_bwd_dkdv_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
        bwdOffZ, bwdOffH, bwdRowIndex, bwdColIndex, bwdGradOffset,
        bwdGradActive, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 0 / H * stride_qz + s.pids 0 % H * stride_qh +
        (s.pids 1 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx => s.readMem GradPre (offsetFn idx)
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx => s.pids 1 * BLOCK_M + idx.1.val < D0
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if P idx then
      bwdGradStoreSpec s GradPre H stride_qz stride_qh stride_qm stride_qk
        BLOCK_M idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 1 * BLOCK_M + idx.1.val < D0
  · simp [P, bwdGradStoreSpec, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex, offsetFn, valueFn, hActive]
  · simp [P, bwdGradStoreSpec, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex, offsetFn, valueFn, hActive]

theorem triton_attention_bwd_dkdv_store_slice_compute_correct
    (GradPre Out : RegionName)
    (H D0 stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice GradPre Out H D0 stride_qz
        stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          bwdGradActive s D0 BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
            BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradStoreSpec s GradPre H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_dkdv_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := triton_attention_bwd_dkdv_store_slice_correct GradPre Out H D0
    stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

theorem triton_attention_bwd_dk_store_slice_compute_correct
    (DKPre DK : RegionName)
    (H D0 stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice DKPre DK H D0 stride_qz
        stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          bwdGradActive s D0 BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (DK, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
            BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradStoreSpec s DKPre H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx) := by
  exact triton_attention_bwd_dkdv_store_slice_compute_correct DKPre DK H D0
    stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj

theorem triton_attention_bwd_dv_store_slice_compute_correct
    (DVPre DV : RegionName)
    (H D0 stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice DVPre DV H D0 stride_qz
        stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          bwdGradActive s D0 BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (DV, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
            BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradStoreSpec s DVPre H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx) := by
  exact triton_attention_bwd_dkdv_store_slice_compute_correct DVPre DV H D0
    stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj

/-! ## Python test-shape wrappers

The checked Python test uses `q/k/v/o` with shape `(2, 4, 128, 64)`, so
the contiguous tensor strides are `(32768, 8192, 64, 1)`. The forward and
backward launchers use `BLOCK_M = BLOCK_N = 128`, `BLOCK_DMODEL = 64`,
`H = 4`, and `D0 = batch * heads * seq_len = 1024`. -/

theorem triton_attention_python_output_offset_injective
    (s : BlockState) (hzRowOffset : Nat) :
    Function.Injective
      (fun idx : TileIndex [128, 64] =>
        outOffset s hzRowOffset 64 1 128 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, rowIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem triton_attention_python_row_offset_injective
    (s : BlockState) (off_hz : Nat) :
    Function.Injective
      (fun i : Fin 128 => lRowOffset s off_hz 128 128 i) := by
  intro a b h
  simp [lRowOffset] at h
  exact Fin.ext (by omega)

theorem triton_attention_python_newdo_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [128, 64] => newdoOffset s 128 64 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [newdoOffset, newdoMIndex, newdoNIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem triton_attention_python_bwd_grad_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [128, 64] =>
        bwdGradOffset s 4 32768 8192 64 1 128 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex, bwdColIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem triton_attention_forward_output_store_python_test_shape_compute_correct
    (Acc Out : RegionName) (hzRowOffset : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_output_store_slice Acc Out
        hzRowOffset 1024 64 1 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => active s hzRowOffset 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (Out, outOffset s hzRowOffset 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (storeValue s Acc hzRowOffset 1024 128 64 idx)))) := by
  exact triton_attention_forward_output_store_slice_compute_correct Acc Out
    hzRowOffset 1024 64 1 128 64 s
    (triton_attention_python_output_offset_injective s hzRowOffset)

theorem triton_attention_forward_l_store_python_test_shape_compute_correct
    (LPrev L : RegionName) (off_hz : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_l_store_slice LPrev L off_hz 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lRowOffset s off_hz 128 128 i))
      (expected := fun i : Fin 128 => lStoreSpec s LPrev off_hz 128 128 i) := by
  exact triton_attention_forward_l_store_slice_compute_correct LPrev L
    off_hz 128 128 s (triton_attention_python_row_offset_injective s off_hz)

theorem triton_attention_forward_m_store_python_test_shape_compute_correct
    (MPrev M : RegionName) (off_hz : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_m_store_slice MPrev M off_hz 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (M, lRowOffset s off_hz 128 128 i))
      (expected := fun i : Fin 128 => mStoreSpec s MPrev off_hz 128 128 i) := by
  exact triton_attention_forward_m_store_slice_compute_correct MPrev M
    off_hz 128 128 s (triton_attention_python_row_offset_injective s off_hz)

theorem triton_attention_bwd_preprocess_newdo_store_python_test_shape_compute_correct
    (NewDOAcc NewDO : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_newdo_store_slice
        NewDOAcc NewDO 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (NewDO, newdoOffset s 128 64 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        newdoStoreSpec s NewDOAcc 128 64 idx) := by
  exact triton_attention_bwd_preprocess_newdo_store_slice_compute_correct
    NewDOAcc NewDO 128 64 s (triton_attention_python_newdo_offset_injective s)

theorem triton_attention_bwd_preprocess_newdo_formula_python_test_shape_compute_correct
    (DO L NewDO : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_newdo_formula_slice
        DO L NewDO 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (NewDO, newdoOffset s 128 64 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        newdoFormulaSpec s DO L 128 64 idx) := by
  exact triton_attention_bwd_preprocess_newdo_formula_slice_compute_correct
    DO L NewDO 128 64 s (triton_attention_python_newdo_offset_injective s)

theorem triton_attention_bwd_preprocess_delta_formula_python_test_shape_compute_correct
    (Out DO L Delta : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_delta_formula_slice
        Out DO L Delta 128 64)
      (initialState := s)
      (write := fun i : Fin 128 => some (Delta, deltaOffset s 128 i))
      (expected := fun i : Fin 128 => deltaFormulaSpec s Out DO L 128 64 i) := by
  exact triton_attention_bwd_preprocess_delta_formula_slice_compute_correct
    Out DO L Delta 128 64 s

theorem triton_attention_bwd_preprocess_delta_store_python_test_shape_compute_correct
    (DeltaAcc Delta : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_delta_store_slice
        DeltaAcc Delta 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (Delta, deltaOffset s 128 i))
      (expected := fun i : Fin 128 => deltaStoreSpec s DeltaAcc 128 i) := by
  exact triton_attention_bwd_preprocess_delta_store_slice_compute_correct
    DeltaAcc Delta 128 s

theorem triton_attention_bwd_dq_store_python_test_shape_compute_correct
    (DQPre DQ : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dq_store_slice DQPre DQ 4
        32768 8192 64 1 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (DQ, bwdGradOffset s 4 32768 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        bwdGradStoreSpec s DQPre 4 32768 8192 64 1 128 idx) := by
  exact triton_attention_bwd_dq_store_slice_compute_correct DQPre DQ 4
    32768 8192 64 1 128 64 s
    (triton_attention_python_bwd_grad_offset_injective s)

theorem triton_attention_bwd_dk_store_python_test_shape_compute_correct
    (DKPre DK : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice DKPre DK 4 1024
        32768 8192 64 1 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DK, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        bwdGradStoreSpec s DKPre 4 32768 8192 64 1 128 idx) := by
  exact triton_attention_bwd_dk_store_slice_compute_correct DKPre DK 4 1024
    32768 8192 64 1 128 64 s
    (triton_attention_python_bwd_grad_offset_injective s)

theorem triton_attention_bwd_dv_store_python_test_shape_compute_correct
    (DVPre DV : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice DVPre DV 4 1024
        32768 8192 64 1 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DV, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        bwdGradStoreSpec s DVPre 4 32768 8192 64 1 128 idx) := by
  exact triton_attention_bwd_dv_store_slice_compute_correct DVPre DV 4 1024
    32768 8192 64 1 128 64 s
    (triton_attention_python_bwd_grad_offset_injective s)

/-- Python forward shape summary: final output plus the row-wise `L` and `M`
side stores are compute-correct for the tested block shape. -/
theorem triton_attention_forward_python_test_shape_all_outputs_compute_correct
    (Acc LPrev MPrev Out L M : RegionName) (hzRowOffset off_hz : Nat)
    (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := triton_attention_forward_output_store_slice Acc Out
        hzRowOffset 1024 64 1 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => active s hzRowOffset 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (Out, outOffset s hzRowOffset 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (storeValue s Acc hzRowOffset 1024 128 64 idx))))) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_forward_l_store_slice LPrev L off_hz 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lRowOffset s off_hz 128 128 i))
      (expected := fun i : Fin 128 => lStoreSpec s LPrev off_hz 128 128 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_forward_m_store_slice MPrev M off_hz 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (M, lRowOffset s off_hz 128 128 i))
      (expected := fun i : Fin 128 => mStoreSpec s MPrev off_hz 128 128 i)) := by
  constructor
  · exact triton_attention_forward_output_store_python_test_shape_compute_correct
      Acc Out hzRowOffset s
  constructor
  · exact triton_attention_forward_l_store_python_test_shape_compute_correct
      LPrev L off_hz s
  · exact triton_attention_forward_m_store_python_test_shape_compute_correct
      MPrev M off_hz s

/-- Python backward-preprocess shape summary: the `NewDO` formula/store and
`Delta` formula/store outputs are compute-correct for the tested block shape. -/
theorem triton_attention_bwd_preprocess_python_test_shape_all_outputs_compute_correct
    (DO L Out NewDOAcc DeltaAcc NewDO Delta : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_newdo_formula_slice
        DO L NewDO 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (NewDO, newdoOffset s 128 64 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        newdoFormulaSpec s DO L 128 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_newdo_store_slice
        NewDOAcc NewDO 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (NewDO, newdoOffset s 128 64 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        newdoStoreSpec s NewDOAcc 128 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_delta_formula_slice
        Out DO L Delta 128 64)
      (initialState := s)
      (write := fun i : Fin 128 => some (Delta, deltaOffset s 128 i))
      (expected := fun i : Fin 128 => deltaFormulaSpec s Out DO L 128 64 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_delta_store_slice
        DeltaAcc Delta 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (Delta, deltaOffset s 128 i))
      (expected := fun i : Fin 128 => deltaStoreSpec s DeltaAcc 128 i)) := by
  constructor
  · exact triton_attention_bwd_preprocess_newdo_formula_python_test_shape_compute_correct
      DO L NewDO s
  constructor
  · exact triton_attention_bwd_preprocess_newdo_store_python_test_shape_compute_correct
      NewDOAcc NewDO s
  constructor
  · exact triton_attention_bwd_preprocess_delta_formula_python_test_shape_compute_correct
      Out DO L Delta s
  · exact triton_attention_bwd_preprocess_delta_store_python_test_shape_compute_correct
      DeltaAcc Delta s

/-- Python backward gradient shape summary: the final `DQ`, `DK`, and `DV`
gradient stores are compute-correct for the tested block shape. -/
theorem triton_attention_bwd_grads_python_test_shape_all_outputs_compute_correct
    (DQPre DKPre DVPre DQ DK DV : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dq_store_slice DQPre DQ 4
        32768 8192 64 1 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (DQ, bwdGradOffset s 4 32768 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        bwdGradStoreSpec s DQPre 4 32768 8192 64 1 128 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice DKPre DK 4 1024
        32768 8192 64 1 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DK, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        bwdGradStoreSpec s DKPre 4 32768 8192 64 1 128 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice DVPre DV 4 1024
        32768 8192 64 1 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DV, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        bwdGradStoreSpec s DVPre 4 32768 8192 64 1 128 idx)) := by
  constructor
  · exact triton_attention_bwd_dq_store_python_test_shape_compute_correct
      DQPre DQ s
  constructor
  · exact triton_attention_bwd_dk_store_python_test_shape_compute_correct
      DKPre DK s
  · exact triton_attention_bwd_dv_store_python_test_shape_compute_correct
      DVPre DV s

end VeriTile.Bench.TritonBenchG.TritonAttention
