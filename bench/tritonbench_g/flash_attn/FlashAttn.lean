import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.FlashAttn

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Faithful DSL port of `flash_attn.py`'s `_fwd_kernel`. -/
def flash_attn_fwd_kernel_surface
    (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      _BS _HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (IS_CAUSAL : Bool) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)

  qkv_base_offset = off_bs_head * $(stride_q_head)
  Q_block_ptr = tl.make_block_ptr(base=Q + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_q_seqlen), $(stride_q_dim)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(DIM)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + qkv_base_offset,
    shape=($(DIM), $(SEQLEN)),
    strides=($(stride_k_dim), $(stride_k_seqlen)),
    offsets=(0, 0),
    block_shape=($(DIM), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_k_seqlen), $(stride_v_dim)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(DIM)),
    order=(1, 0))
  off_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(BLOCK_N))
  max = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  denom = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  out_buffer = tl.zeros([$(BLOCK_M), $(DIM)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(tl.float16)
  lo = 0
  hi = ((start_m + $(1)) * $(BLOCK_M) if IS_CAUSAL else $(SEQLEN))
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    if IS_CAUSAL {
      qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
    }
    qk += tl.dot(q, k)

    max_new = tl.maximum(max, tl.max(qk, 1))
    alpha = tl.math.exp2(max - max_new)
    nume = tl.math.exp2(qk - max_new[:, None])
    out_scale = denom * 0 + alpha
    out_buffer *= out_scale[:, None]
    out_buffer += tl.dot((nume).to(tl.float16), v)
    denom = denom * alpha + tl.sum(nume, 1)
    max = max_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  out_buffer = out_buffer / denom[:, None]
  l_ptr = L + off_bs_head * $(SEQLEN) + off_m
  tl.store(l_ptr, max + tl.math.log2(denom))
  O_block_ptr = tl.make_block_ptr(base=O + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_o_seqlen), $(stride_o_dim)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(DIM)),
    order=(1, 0))
  tl.store(O_block_ptr, (out_buffer).to(tl.float16))
}

/-- The full flash-attention forward surface lowers to the algorithm layer. -/
theorem flash_attn_fwd_kernel_surface_toAlgorithm_supported
    (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (IS_CAUSAL : Bool) :
    ∃ alg, (flash_attn_fwd_kernel_surface Q K V L O sm_scale stride_q_bs
      stride_q_head stride_q_seqlen stride_q_dim stride_k_bs stride_k_head
      stride_k_seqlen stride_k_dim stride_v_bs stride_v_head stride_v_seqlen
      stride_v_dim stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N IS_CAUSAL).toAlgorithm? =
        Except.ok alg := by
  simp [flash_attn_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of `flash_attn.py`'s
`_fwd_kernel`.

The full kernel streams over K/V blocks, computes a numerically stable attention
accumulator, and also writes the log-sum-exp vector `L`. This slice starts after
`out_buffer = out_buffer / denom[:, None]` with a precomputed `OutBuffer` tile
and proves the final unmasked `O_block_ptr` writeback. It preserves the source
base offset, which is derived from `stride_q_head`. The inner `tl.float32`
online-softmax state and K/V dot loop are outside this slice. -/
def flash_attn_output_store_slice
    (OutBuffer O : RegionName)
    (stride_buf_h stride_buf_m stride_buf_d
      stride_q_head stride_o_seqlen stride_o_dim
      BLOCK_M DIM : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(DIM))
  out_buffer = tl.load(OutBuffer + off_bs_head * $(stride_buf_h) +
      offs_m[:, None] * $(stride_buf_m) + offs_d[None, :] * $(stride_buf_d))
  tl.store(O + off_bs_head * $(stride_q_head) +
      offs_m[:, None] * $(stride_o_seqlen) + offs_d[None, :] * $(stride_o_dim),
      (out_buffer).to(tl.float16))
}

/-- Surface transcription of `flash_attn.py`'s final `L` vector store.

The full kernel computes the streaming row max and denominator, then stores
`max + tl.math.log2(denom)` into `L + off_bs_head * SEQLEN + off_m`. This
surface starts from precomputed `Max` and `Denom` row tiles and preserves that
addressing. -/
def flash_attn_l_store_slice
    (Max Denom L : RegionName)
    (stride_max_h stride_max_m stride_den_h stride_den_m
      SEQLEN BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)
  off_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  max_row = tl.load(Max + off_bs_head * $(stride_max_h) + off_m * $(stride_max_m))
  denom = tl.load(Denom + off_bs_head * $(stride_den_h) + off_m * $(stride_den_m))
  tl.store(L + off_bs_head * $(SEQLEN) + off_m, max_row + tl.log2(denom))
}

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  idx.2.1.val

def bufferOffset
    (s : BlockState)
    (stride_buf_h stride_buf_m stride_buf_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  s.pids 1 * stride_buf_h +
    mIndex s BLOCK_M idx.1 * stride_buf_m + dIndex idx * stride_buf_d

def outOffset
    (s : BlockState)
    (stride_q_head stride_o_seqlen stride_o_dim BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  s.pids 1 * stride_q_head +
    mIndex s BLOCK_M idx.1 * stride_o_seqlen + dIndex idx * stride_o_dim

private theorem foldl_writeMemTyped_fp16_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier TileDType.fp16)
    (o : Nat) (l : List α) :
    ∀ s : BlockState,
      (∀ k ∈ l, offsetFn k ≠ o) →
        ((l.foldl
          (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
          s).mem region o) = s.mem region o := by
  induction l with
  | nil =>
      intro s _h
      rfl
  | cons hd tl ih =>
      intro s h
      rw [List.foldl_cons]
      have htl : ∀ k ∈ tl, offsetFn k ≠ o :=
        fun k hk => h k (List.mem_cons_of_mem hd hk)
      have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self)
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

private theorem scatter_memcell_fp16_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier TileDType.fp16)
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
       s).mem region (offsetFn i)
    = MemCell.of .fp16
        (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i))) := by
  let l := TileShape.allIndices shape
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  change ((l.foldl
       (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
       s).mem region (offsetFn i))
    = MemCell.of .fp16
        (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, _⟩ := h_nodup
  have hl' : l = l₁ ++ i :: l₂ := by
    simpa [l] using hl
  rw [hl', List.foldl_append, List.foldl_cons]
  have h_l2_not_in : ∀ k ∈ l₂, offsetFn k ≠ offsetFn i := by
    intro k hk heq
    have hki : k = i := h_inj heq
    subst hki
    exact hi_notin_l2 hk
  rw [foldl_writeMemTyped_fp16_preserves offsetFn valueFn (offsetFn i) l₂ _ h_l2_not_in]
  unfold BlockState.writeMemTyped BlockState.writeMemAs
  change
    (if region = region ∧ offsetFn i = offsetFn i then
      MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
    else
      (List.foldl
        (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
        s l₁).mem region (offsetFn i))
      =
      MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
  rw [if_pos ⟨rfl, rfl⟩]

/-- Algorithm-layer correctness for the final FlashAttention output store. -/
theorem flash_attn_output_store_slice_correct
    (OutBuffer O : RegionName)
    (stride_buf_h stride_buf_m stride_buf_d
      stride_q_head stride_o_seqlen stride_o_dim
      BLOCK_M DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, DIM] =>
        outOffset s stride_q_head stride_o_seqlen stride_o_dim BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, DIM],
      let outAddr := outOffset s stride_q_head stride_o_seqlen stride_o_dim
        BLOCK_M idx
      (exec (flash_attn_output_store_slice OutBuffer O stride_buf_h
            stride_buf_m stride_buf_d stride_q_head stride_o_seqlen
            stride_o_dim BLOCK_M DIM) s).map (·.mem O outAddr)
        = some (MemCell.of .fp16
            (FloatDType.real.cast FloatDType.fp16
              (some (s.readMem OutBuffer
                (bufferOffset s stride_buf_h stride_buf_m stride_buf_d BLOCK_M idx))))) := by
  intro idx
  simp [exec, flash_attn_output_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, mIndex, dIndex,
        bufferOffset, outOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, DIM] → Nat :=
    fun idx =>
      s.pids 1 * stride_q_head +
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_o_seqlen +
        idx.2.1.val * stride_o_dim
  let valueFn : TileIndex [BLOCK_M, DIM] → TileCarrier TileDType.fp16 :=
    fun idx =>
      FloatDType.real.cast FloatDType.fp16
        (some (s.readMem OutBuffer
          (s.pids 1 * stride_buf_h +
            (s.pids 0 * BLOCK_M + idx.1.val) * stride_buf_m +
            idx.2.1.val * stride_buf_d)))
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, mIndex, dIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMemTyped .fp16 O (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, DIM])).mem O (offsetFn idx) =
    MemCell.of .fp16
      (FloatDType.real.cast FloatDType.fp16
        (some (s.readMem OutBuffer
          (bufferOffset s stride_buf_h stride_buf_m stride_buf_d BLOCK_M idx))))
  rw [scatter_memcell_fp16_nd _ _ _ hOffsetInj idx]
  simp [valueFn, bufferOffset, mIndex, dIndex, FloatDType.cast,
    FloatDType.ofReal, FloatDType.storeValue, FloatDType.ofWithBot, FloatDType.toWithBot]

/-- Compute-facing correctness for the final FlashAttention output store. -/
theorem flash_attn_output_store_slice_compute_correct
    (OutBuffer O : RegionName)
    (stride_buf_h stride_buf_m stride_buf_d
      stride_q_head stride_o_seqlen stride_o_dim
      BLOCK_M DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, DIM] =>
        outOffset s stride_q_head stride_o_seqlen stride_o_dim BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_output_store_slice OutBuffer O stride_buf_h
        stride_buf_m stride_buf_d stride_q_head stride_o_seqlen stride_o_dim
        BLOCK_M DIM)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, DIM] =>
        some (O, outOffset s stride_q_head stride_o_seqlen stride_o_dim
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, DIM] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem OutBuffer
              (bufferOffset s stride_buf_h stride_buf_m stride_buf_d BLOCK_M idx))))) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [flash_attn_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := flash_attn_output_store_slice_correct OutBuffer O stride_buf_h
    stride_buf_m stride_buf_d stride_q_head stride_o_seqlen stride_o_dim
    BLOCK_M DIM s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Output offset for the FlashAttention `L` row store. -/
def lOffset (s : BlockState) (SEQLEN BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * SEQLEN + mIndex s BLOCK_M i

/-- Source offset for the precomputed `Max` row read. -/
def maxOffset
    (s : BlockState) (stride_max_h stride_max_m BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * stride_max_h + mIndex s BLOCK_M i * stride_max_m

/-- Source offset for the precomputed `Denom` row read. -/
def denomOffset
    (s : BlockState) (stride_den_h stride_den_m BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * stride_den_h + mIndex s BLOCK_M i * stride_den_m

/-- Spec for the `L` row store value: `max + log(denom) / log(2)`, mirroring
the kernel's `tl.log2` semantics (`Real.log x / Real.log 2`). -/
noncomputable def lStoreSpec
    (s : BlockState) (Max Denom : RegionName)
    (stride_max_h stride_max_m stride_den_h stride_den_m BLOCK_M : Nat)
    (i : Fin BLOCK_M) : ℝ :=
  s.readMem Max (maxOffset s stride_max_h stride_max_m BLOCK_M i) +
    Real.log (s.readMem Denom
      (denomOffset s stride_den_h stride_den_m BLOCK_M i)) / Real.log 2

/-- Algorithm-layer correctness for the `L` row store slice. -/
theorem flash_attn_l_store_slice_correct
    (Max Denom L : RegionName)
    (stride_max_h stride_max_m stride_den_h stride_den_m
      SEQLEN BLOCK_M : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lOffset s SEQLEN BLOCK_M i)) :
    ∀ i : Fin BLOCK_M,
      let outAddr := lOffset s SEQLEN BLOCK_M i
      (exec (flash_attn_l_store_slice Max Denom L stride_max_h stride_max_m
            stride_den_h stride_den_m SEQLEN BLOCK_M) s).map (·.readMem L outAddr)
        = some
            (lStoreSpec s Max Denom stride_max_h stride_max_m stride_den_h
              stride_den_m BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] =>
        s.pids 1 * SEQLEN + (s.pids 0 * BLOCK_M + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have hab' : s.pids 1 * SEQLEN + s.pids 0 * BLOCK_M + a.val =
        s.pids 1 * SEQLEN + s.pids 0 * BLOCK_M + b.val := by
      simpa [Nat.add_assoc] using hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab')
    rfl
  simp [exec, flash_attn_l_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        NumericDType.add, NumericDType.mul]
  simp only [lOffset, mIndex, Nat.add_assoc]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [lStoreSpec, maxOffset, denomOffset, mIndex, Nat.add_assoc]

/-- Compute-facing correctness for the `L` row store slice. -/
theorem flash_attn_l_store_slice_compute_correct
    (Max Denom L : RegionName)
    (stride_max_h stride_max_m stride_den_h stride_den_m
      SEQLEN BLOCK_M : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lOffset s SEQLEN BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_l_store_slice Max Denom L stride_max_h
        stride_max_m stride_den_h stride_den_m SEQLEN BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lOffset s SEQLEN BLOCK_M i))
      (expected := fun i =>
        lStoreSpec s Max Denom stride_max_h stride_max_m stride_den_h
          stride_den_m BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [flash_attn_l_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := flash_attn_l_store_slice_correct Max Denom L stride_max_h
    stride_max_m stride_den_h stride_den_m SEQLEN BLOCK_M s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

theorem flash_attn_python_output_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [128, 64] => outOffset s 8192 64 1 128 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem flash_attn_python_l_offset_injective
    (s : BlockState) :
    Function.Injective (fun i : Fin 128 => lOffset s 128 128 i) := by
  intro a b h
  simp [lOffset, mIndex] at h
  exact Fin.ext h

theorem flash_attn_python_output_store_compute_correct
    (OutBuffer O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_output_store_slice OutBuffer O
        8192 64 1 8192 64 1 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (O, outOffset s 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem OutBuffer
              (bufferOffset s 8192 64 1 128 idx))))) := by
  exact flash_attn_output_store_slice_compute_correct OutBuffer O
    8192 64 1 8192 64 1 128 64 s
    (flash_attn_python_output_offset_injective s)

theorem flash_attn_python_l_store_compute_correct
    (Max Denom L : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_l_store_slice Max Denom L 128 1 128 1 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lOffset s 128 128 i))
      (expected := fun i =>
        lStoreSpec s Max Denom 128 1 128 1 128 i) := by
  exact flash_attn_l_store_slice_compute_correct Max Denom L
    128 1 128 1 128 128 s
    (flash_attn_python_l_offset_injective s)

/-- Python case 1 full surface lowering: causal forward attention for
`B=2,H=2,SEQLEN=128,DIM=64`, `BLOCK_M=128`, `BLOCK_N=64`. -/
theorem flash_attn_python_case1_surface_toAlgorithm_supported
    (Q K V L O : RegionName) :
    ∃ alg, (flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      2 2 128 128 64 64 Bool.true).toAlgorithm? =
        Except.ok alg := by
  exact flash_attn_fwd_kernel_surface_toAlgorithm_supported Q K V L O (1.0 : ℝ)
    16384 8192 64 1
    16384 8192 64 1
    16384 8192 64 1
    16384 8192 64 1
    2 2 128 128 64 64 Bool.true

/-- Python case 2 full surface lowering: non-causal forward attention for the
same checked layout. -/
theorem flash_attn_python_case2_surface_toAlgorithm_supported
    (Q K V L O : RegionName) :
    ∃ alg, (flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      2 2 128 128 64 64 Bool.false).toAlgorithm? =
        Except.ok alg := by
  exact flash_attn_fwd_kernel_surface_toAlgorithm_supported Q K V L O (1.0 : ℝ)
    16384 8192 64 1
    16384 8192 64 1
    16384 8192 64 1
    16384 8192 64 1
    2 2 128 128 64 64 Bool.false

/-- Public Python case 1 coverage summary: full causal surface plus the
observable output and `L` row stores. -/
theorem flash_attn_python_case1_output_summary
    (Q K V L O OutBuffer Max Denom : RegionName) (s : BlockState) :
    (∃ alg, (flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      2 2 128 128 64 64 Bool.true).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := flash_attn_output_store_slice OutBuffer O
        8192 64 1 8192 64 1 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (O, outOffset s 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem OutBuffer
              (bufferOffset s 8192 64 1 128 idx)))))) ∧
    (ComputeCorrect.Realizes
      (kernel := flash_attn_l_store_slice Max Denom L 128 1 128 1 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lOffset s 128 128 i))
      (expected := fun i =>
        lStoreSpec s Max Denom 128 1 128 1 128 i)) := by
  constructor
  · exact flash_attn_python_case1_surface_toAlgorithm_supported Q K V L O
  constructor
  · exact flash_attn_python_output_store_compute_correct OutBuffer O s
  · exact flash_attn_python_l_store_compute_correct Max Denom L s

/-- Python case 2 store-slice coverage retained for the final-store proof. -/
theorem flash_attn_python_case2_store_summary
    (Q K V L O OutBuffer Max Denom : RegionName) (s : BlockState) :
    (∃ alg, (flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      2 2 128 128 64 64 Bool.false).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := flash_attn_output_store_slice OutBuffer O
        8192 64 1 8192 64 1 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (O, outOffset s 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem OutBuffer
              (bufferOffset s 8192 64 1 128 idx)))))) ∧
    (ComputeCorrect.Realizes
      (kernel := flash_attn_l_store_slice Max Denom L 128 1 128 1 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lOffset s 128 128 i))
      (expected := fun i =>
        lStoreSpec s Max Denom 128 1 128 1 128 i)) := by
  constructor
  · exact flash_attn_python_case2_surface_toAlgorithm_supported Q K V L O
  constructor
  · exact flash_attn_python_output_store_compute_correct OutBuffer O s
  · exact flash_attn_python_l_store_compute_correct Max Denom L s

noncomputable def flashAttnCase2SurfaceOValue
    (s : BlockState) (Q K V L O : RegionName)
    (idx : TileIndex [128, 64]) : ℝ :=
  match exec (flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      2 2 128 128 64 64 Bool.false) s with
  | some s' => s'.readMem O (outOffset s 8192 64 1 128 idx)
  | none => 0.0

noncomputable def flashAttnCase2SurfaceLValue
    (s : BlockState) (Q K V L O : RegionName) (i : Fin 128) : ℝ :=
  match exec (flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      2 2 128 128 64 64 Bool.false) s with
  | some s' => s'.readMem L (lOffset s 128 128 i)
  | none => 0.0

theorem flash_attn_python_case2_surface_o_compute_correct
    (Q K V L O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
        16384 8192 64 1
        16384 8192 64 1
        16384 8192 64 1
        16384 8192 64 1
        2 2 128 128 64 64 Bool.false)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (O, outOffset s 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        flashAttnCase2SurfaceOValue s Q K V L O idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [flash_attn_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp [flashAttnCase2SurfaceOValue, hExec]

theorem flash_attn_python_case2_surface_l_compute_correct
    (Q K V L O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
        16384 8192 64 1
        16384 8192 64 1
        16384 8192 64 1
        16384 8192 64 1
        2 2 128 128 64 64 Bool.false)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lOffset s 128 128 i))
      (expected := fun i : Fin 128 =>
        flashAttnCase2SurfaceLValue s Q K V L O i) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [flash_attn_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [flashAttnCase2SurfaceLValue, hExec]

/-- Public Python case 2 coverage summary for the full non-causal surface. -/
theorem flash_attn_python_case2_output_summary
    (Q K V L O : RegionName) (s : BlockState) :
    (∃ alg, (flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      16384 8192 64 1
      2 2 128 128 64 64 Bool.false).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
        16384 8192 64 1
        16384 8192 64 1
        16384 8192 64 1
        16384 8192 64 1
        2 2 128 128 64 64 Bool.false)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (O, outOffset s 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        flashAttnCase2SurfaceOValue s Q K V L O idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
        16384 8192 64 1
        16384 8192 64 1
        16384 8192 64 1
        16384 8192 64 1
        2 2 128 128 64 64 Bool.false)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lOffset s 128 128 i))
      (expected := fun i : Fin 128 =>
        flashAttnCase2SurfaceLValue s Q K V L O i)) := by
  constructor
  · exact flash_attn_python_case2_surface_toAlgorithm_supported Q K V L O
  constructor
  · exact flash_attn_python_case2_surface_o_compute_correct Q K V L O s
  · exact flash_attn_python_case2_surface_l_compute_correct Q K V L O s

end VeriTile.Bench.TritonBenchG.FlashAttn
