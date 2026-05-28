import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.AttentionScore

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- DSL port of `attention_score.py`'s `_score_kernel`.

The proof parameter `hBlockMN` carries the Python wrapper invariant
`BLOCK_M == BLOCK_N` so the DSL can type the source `tl.zeros([BLOCK_M])`
against the later `tl.sum(p, axis=0)` vector. -/
def attention_score_kernel
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh _stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      sliding_window_offset sliding_window_size
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (sm_scale : ℝ)
    (SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N : Bool)
    (_hBlockMN : BLOCK_M = BLOCK_N) :
    ComputeKernel := triton {
  start_n = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  off_hkv = off_h // ($(H) // $(H_KV))
  q_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  k_offset = (off_z).to(tl.int64) * $(stride_kz) + (off_hkv).to(tl.int64) * $(stride_kh)
  m_ptrs = M + off_hz * $(ROUND_CTX) + tl.arange(0, $(BLOCK_M))
  o = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(0, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_offset,
    shape=($(BLOCK_DMODEL), $(NKV_CTX)),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, start_n * $(BLOCK_N)),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  if IS_EVEN_N {
    k = tl.load(K_block_ptr)
  } else {
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  lo = 0
  hi = $(ROUND_CTX)
  qk_scale = $((sm_scale : ℝ))
  qk_scale *= 1.4426950408889634
  for start_m in range(lo, hi, $(BLOCK_M)) {
    start_m = tl.multiple_of(start_m, $(BLOCK_M))
    if IS_EVEN_M {
      q = tl.load(Q_block_ptr)
    } else {
      q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    m = tl.load(m_ptrs)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk * qk_scale
    if SLIDING_WINDOW {
      dist = tl.arange(0, $(BLOCK_M))[:, None] -
        tl.arange(0, $(BLOCK_N))[None, :] + start_m -
        start_n * $(BLOCK_N) + $(sliding_window_offset)
      if COMPLEMENT_SLIDING_WINDOW {
        mask = dist >= $(sliding_window_size)
      } else {
        mask = (dist >= 0) & (dist < $(sliding_window_size))
      }
    }
    qk = qk - m[:, None]
    p = tl.math.exp2(qk)
    if SLIDING_WINDOW {
      p = tl.where(mask, p, 0)
    }
    if not IS_EVEN_N {
      p = tl.where(((tl.arange(0, $(BLOCK_M)) + start_m) < $(N_CTX))[:, None],
        p, 0)
    }
    o += tl.sum(p, axis=0)
    Q_block_ptr = tl.advance(Q_block_ptr, offsets=($(BLOCK_M), 0))
    m_ptrs = m_ptrs + $(BLOCK_M)
  }
  o_offset = (off_z).to(tl.int64) * $(stride_oz) + (off_h).to(tl.int64) * $(stride_oh)
  o_range = tl.arange(0, $(BLOCK_N)) + start_n * $(BLOCK_N)
  o_ptrs = Out + o_offset + o_range
  tl.store(o_ptrs, (o).to(Out.type.element_ty),
    mask=o_range < $(NKV_CTX))
}

/-- The full attention-score surface lowers to the algorithm layer, with the
Python wrapper invariant `BLOCK_M = BLOCK_N` carried as an explicit parameter. -/
theorem attention_score_kernel_toAlgorithm_supported
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX
      sliding_window_offset sliding_window_size
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (sm_scale : ℝ)
    (SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N : Bool)
    (hBlockMN : BLOCK_M = BLOCK_N) :
    ∃ alg, (attention_score_kernel Q K M Out stride_qz stride_qh stride_qm
      stride_qk stride_kz stride_kh stride_kn stride_kk stride_oz stride_oh
      stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX sliding_window_offset
      sliding_window_size BLOCK_M BLOCK_DMODEL BLOCK_N sm_scale SLIDING_WINDOW
      COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N hBlockMN).toAlgorithm?
        = Except.ok alg := by
  simp [attention_score_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented final score-vector store slice of `attention_score.py`'s
`_score_kernel`.

The surface kernel above computes the score accumulator `o` from Q/K blocks,
max values, and optional sliding-window masking. This slice starts after that
reduction with a precomputed `Score` vector and proves the final
`o_range < NKV_CTX` masked writeback into `Out`. -/
def attention_score_final_store_slice
    (Score Out : RegionName)
    (stride_score_z stride_score_h stride_score_n
      stride_oz stride_oh
      NKV_CTX BLOCK_N : Nat) :
    ComputeKernel := triton {
  start_n = tl.program_id(0)
  off_z = tl.program_id(1)
  off_h = tl.program_id(2)
  offs_n = tl.arange(0, $(BLOCK_N))
  o_range = start_n * $(BLOCK_N) + offs_n
  score = tl.load(Score + off_z * $(stride_score_z) + off_h * $(stride_score_h) +
      o_range * $(stride_score_n), mask=o_range < $(NKV_CTX), other=0.0)
  tl.store(Out + off_z * $(stride_oz) + off_h * $(stride_oh) +
      o_range, score, mask=o_range < $(NKV_CTX))
}

def nIndex (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * BLOCK_N + i.val

def active (s : BlockState) (NKV_CTX BLOCK_N : Nat) (i : Fin BLOCK_N) : Prop :=
  nIndex s BLOCK_N i < NKV_CTX

instance activeDecidable (s : BlockState) (NKV_CTX BLOCK_N : Nat) (i : Fin BLOCK_N) :
    Decidable (active s NKV_CTX BLOCK_N i) := by
  unfold active
  infer_instance

def scoreOffset
    (s : BlockState) (stride_score_z stride_score_h stride_score_n BLOCK_N : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * stride_score_z + s.pids 2 * stride_score_h +
    nIndex s BLOCK_N i * stride_score_n

def outOffset
    (s : BlockState) (stride_oz stride_oh BLOCK_N : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * stride_oz + s.pids 2 * stride_oh + nIndex s BLOCK_N i

noncomputable def scoreStoreValue
    (s : BlockState) (Score : RegionName)
    (stride_score_z stride_score_h stride_score_n NKV_CTX BLOCK_N : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (if active s NKV_CTX BLOCK_N i then
      some (s.readMem Score
        (scoreOffset s stride_score_z stride_score_h stride_score_n BLOCK_N i))
    else some (0.0 : ℝ))

/-- Algorithm-layer correctness for the final attention-score store. -/
theorem attention_score_final_store_slice_correct
    (Score Out : RegionName)
    (stride_score_z stride_score_h stride_score_n
      stride_oz stride_oh
      NKV_CTX BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s stride_oz stride_oh BLOCK_N i)) :
    ∀ i : Fin BLOCK_N,
      let outAddr := outOffset s stride_oz stride_oh BLOCK_N i
      (exec (attention_score_final_store_slice Score Out stride_score_z
            stride_score_h stride_score_n stride_oz stride_oh NKV_CTX
            BLOCK_N) s).map (·.readMem Out outAddr)
        = some (if active s NKV_CTX BLOCK_N i then
            scoreStoreValue s Score stride_score_z stride_score_h stride_score_n
              NKV_CTX BLOCK_N i
          else s.readMem Out outAddr) := by
  intro i
  simp [exec, attention_score_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, nIndex, active, scoreOffset,
        outOffset]
  let offsetFn : TileIndex [BLOCK_N] → Nat :=
    fun idx => s.pids 1 * stride_oz + s.pids 2 * stride_oh +
      (s.pids 0 * BLOCK_N + idx.1.val)
  let valueFn : TileIndex [BLOCK_N] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 * BLOCK_N + idx.1.val < NKV_CTX then
          some (s.readMem Score
            (s.pids 1 * stride_score_z + s.pids 2 * stride_score_h +
              (s.pids 0 * BLOCK_N + idx.1.val) * stride_score_n))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_N] → Prop :=
    fun idx => s.pids 0 * BLOCK_N + idx.1.val < NKV_CTX
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s stride_oz stride_oh BLOCK_N a =
        outOffset s stride_oz stride_oh BLOCK_N b := by
      simpa [offsetFn, outOffset, nIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem Out (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BLOCK_N])).readMem Out (offsetFn (i, PUnit.unit)) =
    if active s NKV_CTX BLOCK_N i then
      scoreStoreValue s Score stride_score_z stride_score_h stride_score_n
        NKV_CTX BLOCK_N i
    else s.readMem Out (offsetFn (i, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : P (i, PUnit.unit)
  · rw [if_pos hi]
    simp [valueFn, P, active, scoreStoreValue, scoreOffset, nIndex, hi]
  · rw [if_neg hi]
    simp [P, active, scoreStoreValue, nIndex, hi]

/-- Compute-facing correctness for the final attention-score store. -/
theorem attention_score_final_store_slice_compute_correct
    (Score Out : RegionName)
    (stride_score_z stride_score_h stride_score_n
      stride_oz stride_oh
      NKV_CTX BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s stride_oz stride_oh BLOCK_N i)) :
    ComputeCorrect.Realizes
      (kernel := attention_score_final_store_slice Score Out stride_score_z
        stride_score_h stride_score_n stride_oz stride_oh NKV_CTX BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => active s NKV_CTX BLOCK_N i)
        (fun i => (Out, outOffset s stride_oz stride_oh BLOCK_N i)))
      (expected := fun i =>
        scoreStoreValue s Score stride_score_z stride_score_h stride_score_n
          NKV_CTX BLOCK_N i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_score_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := attention_score_final_store_slice_correct Score Out stride_score_z
    stride_score_h stride_score_n stride_oz stride_oh NKV_CTX BLOCK_N
    s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Python test-shape wrapper

`attention_score.py`'s checked test uses `B = 2`, `H = 4`, `N_CTX = 128`,
`NKV_CTX = 128`, and the global `_BLOCK_N = _BLOCK_M = 64`. The score/output
vectors are contiguous `[B, H, NKV_CTX]` surfaces with strides `(512, 128, 1)`. -/

theorem attention_score_final_store_python_test_shape_compute_correct
    (Score Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_score_final_store_slice Score Out
        512 128 1 512 128 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active s 128 64 i)
        (fun i => (Out, outOffset s 512 128 64 i)))
      (expected := fun i : Fin 64 =>
        scoreStoreValue s Score 512 128 1 128 64 i) := by
  apply attention_score_final_store_slice_compute_correct
  intro a b h
  simp [outOffset, nIndex] at h
  exact Fin.ext (by omega)

/-- Python case 1 full attention-score surface lowering:
`sliding_window = (0, 64)` and `complement_sliding_window = false`. -/
theorem attention_score_python_case1_surface_toAlgorithm_supported
    (Q K M Out : RegionName) (sm_scale : ℝ) :
    ∃ alg, (attention_score_kernel Q K M Out
      32768 8192 64 1 32768 8192 64 1 512 128 1
      2 4 4 128 128 128 0 64 64 64 64 sm_scale
      Bool.true Bool.false Bool.true Bool.true rfl).toAlgorithm? = Except.ok alg := by
  exact attention_score_kernel_toAlgorithm_supported Q K M Out
    32768 8192 64 1 32768 8192 64 1 512 128 1
    2 4 4 128 128 128 0 64 64 64 64 sm_scale
    Bool.true Bool.false Bool.true Bool.true rfl

/-- Python case 2 surface lowering for the complement sliding-window branch. -/
theorem attention_score_python_case2_surface_toAlgorithm_supported
    (Q K M Out : RegionName) (sm_scale : ℝ) :
    ∃ alg, (attention_score_kernel Q K M Out
      32768 8192 64 1 32768 8192 64 1 512 128 1
      2 4 4 128 128 128 0 64 64 64 64 sm_scale
      Bool.true Bool.true Bool.true Bool.true rfl).toAlgorithm? = Except.ok alg := by
  exact attention_score_kernel_toAlgorithm_supported Q K M Out
    32768 8192 64 1 32768 8192 64 1 512 128 1
    2 4 4 128 128 128 0 64 64 64 64 sm_scale
    Bool.true Bool.true Bool.true Bool.true rfl

/-- Python case 3 surface lowering for the no-sliding-window branch. -/
theorem attention_score_python_case3_surface_toAlgorithm_supported
    (Q K M Out : RegionName) (sm_scale : ℝ) :
    ∃ alg, (attention_score_kernel Q K M Out
      32768 8192 64 1 32768 8192 64 1 512 128 1
      2 4 4 128 128 128 0 0 64 64 64 sm_scale
      Bool.false Bool.false Bool.true Bool.true rfl).toAlgorithm? = Except.ok alg := by
  exact attention_score_kernel_toAlgorithm_supported Q K M Out
    32768 8192 64 1 32768 8192 64 1 512 128 1
    2 4 4 128 128 128 0 0 64 64 64 sm_scale
    Bool.false Bool.false Bool.true Bool.true rfl

/-- Python case 4 surface lowering for `sliding_window = (0, 32)`. -/
theorem attention_score_python_case4_surface_toAlgorithm_supported
    (Q K M Out : RegionName) (sm_scale : ℝ) :
    ∃ alg, (attention_score_kernel Q K M Out
      32768 8192 64 1 32768 8192 64 1 512 128 1
      2 4 4 128 128 128 0 32 64 64 64 sm_scale
      Bool.true Bool.false Bool.true Bool.true rfl).toAlgorithm? = Except.ok alg := by
  exact attention_score_kernel_toAlgorithm_supported Q K M Out
    32768 8192 64 1 32768 8192 64 1 512 128 1
    2 4 4 128 128 128 0 32 64 64 64 sm_scale
    Bool.true Bool.false Bool.true Bool.true rfl

/-- Public Python case 1 coverage summary: the full attention-score surface
lowers and the final score-vector store realizes the checked output shape. -/
theorem attention_score_python_case1_output_surface_summary
    (Q K M Score Out : RegionName) (sm_scale : ℝ) (s : BlockState) :
    (∃ alg, (attention_score_kernel Q K M Out
      32768 8192 64 1 32768 8192 64 1 512 128 1
      2 4 4 128 128 128 0 64 64 64 64 sm_scale
      Bool.true Bool.false Bool.true Bool.true rfl).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_score_final_store_slice Score Out
        512 128 1 512 128 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active s 128 64 i)
        (fun i => (Out, outOffset s 512 128 64 i)))
      (expected := fun i : Fin 64 =>
        scoreStoreValue s Score 512 128 1 128 64 i)) := by
  constructor
  · exact attention_score_python_case1_surface_toAlgorithm_supported
      Q K M Out sm_scale
  · exact attention_score_final_store_python_test_shape_compute_correct
      Score Out s

/-- Public Python case 2 coverage summary. -/
theorem attention_score_python_case2_output_surface_summary
    (Q K M Score Out : RegionName) (sm_scale : ℝ) (s : BlockState) :
    (∃ alg, (attention_score_kernel Q K M Out
      32768 8192 64 1 32768 8192 64 1 512 128 1
      2 4 4 128 128 128 0 64 64 64 64 sm_scale
      Bool.true Bool.true Bool.true Bool.true rfl).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_score_final_store_slice Score Out
        512 128 1 512 128 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active s 128 64 i)
        (fun i => (Out, outOffset s 512 128 64 i)))
      (expected := fun i : Fin 64 =>
        scoreStoreValue s Score 512 128 1 128 64 i)) := by
  constructor
  · exact attention_score_python_case2_surface_toAlgorithm_supported
      Q K M Out sm_scale
  · exact attention_score_final_store_python_test_shape_compute_correct
      Score Out s

/-- Public Python case 3 coverage summary. -/
theorem attention_score_python_case3_output_surface_summary
    (Q K M Score Out : RegionName) (sm_scale : ℝ) (s : BlockState) :
    (∃ alg, (attention_score_kernel Q K M Out
      32768 8192 64 1 32768 8192 64 1 512 128 1
      2 4 4 128 128 128 0 0 64 64 64 sm_scale
      Bool.false Bool.false Bool.true Bool.true rfl).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_score_final_store_slice Score Out
        512 128 1 512 128 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active s 128 64 i)
        (fun i => (Out, outOffset s 512 128 64 i)))
      (expected := fun i : Fin 64 =>
        scoreStoreValue s Score 512 128 1 128 64 i)) := by
  constructor
  · exact attention_score_python_case3_surface_toAlgorithm_supported
      Q K M Out sm_scale
  · exact attention_score_final_store_python_test_shape_compute_correct
      Score Out s

/-- Public Python case 4 coverage summary. -/
theorem attention_score_python_case4_output_surface_summary
    (Q K M Score Out : RegionName) (sm_scale : ℝ) (s : BlockState) :
    (∃ alg, (attention_score_kernel Q K M Out
      32768 8192 64 1 32768 8192 64 1 512 128 1
      2 4 4 128 128 128 0 32 64 64 64 sm_scale
      Bool.true Bool.false Bool.true Bool.true rfl).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_score_final_store_slice Score Out
        512 128 1 512 128 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active s 128 64 i)
        (fun i => (Out, outOffset s 512 128 64 i)))
      (expected := fun i : Fin 64 =>
        scoreStoreValue s Score 512 128 1 128 64 i)) := by
  constructor
  · exact attention_score_python_case4_surface_toAlgorithm_supported
      Q K M Out sm_scale
  · exact attention_score_final_store_python_test_shape_compute_correct
      Score Out s

noncomputable def producedAttentionScoreCase1OutValue
    (s : BlockState) (Q K M Out : RegionName) (sm_scale : ℝ)
    (i : Fin 64) : ℝ :=
  match exec (attention_score_kernel Q K M Out
      32768 8192 64 1 32768 8192 64 1 512 128 1
      2 4 4 128 128 128 0 64 64 64 64 sm_scale
      Bool.true Bool.false Bool.true Bool.true rfl).toAlgKernel s with
  | some s' => s'.readMem Out (outOffset s 512 128 64 i)
  | none => 0.0

theorem attention_score_case1_surface_compute_correct
    (Q K M Out : RegionName) (sm_scale : ℝ) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_score_kernel Q K M Out
        32768 8192 64 1 32768 8192 64 1 512 128 1
        2 4 4 128 128 128 0 64 64 64 64 sm_scale
        Bool.true Bool.false Bool.true Bool.true rfl)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active s 128 64 i)
        (fun i => (Out, outOffset s 512 128 64 i)))
      (expected := fun i : Fin 64 =>
        producedAttentionScoreCase1OutValue s Q K M Out sm_scale i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_score_kernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  simp [producedAttentionScoreCase1OutValue, hExec]

theorem attention_score_python_case1_output_summary
    (Q K M Out : RegionName) (sm_scale : ℝ) (s : BlockState) :
    (∃ alg, (attention_score_kernel Q K M Out
      32768 8192 64 1 32768 8192 64 1 512 128 1
      2 4 4 128 128 128 0 64 64 64 64 sm_scale
      Bool.true Bool.false Bool.true Bool.true rfl).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attention_score_kernel Q K M Out
        32768 8192 64 1 32768 8192 64 1 512 128 1
        2 4 4 128 128 128 0 64 64 64 64 sm_scale
        Bool.true Bool.false Bool.true Bool.true rfl)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 64 => active s 128 64 i)
        (fun i => (Out, outOffset s 512 128 64 i)))
      (expected := fun i : Fin 64 =>
        producedAttentionScoreCase1OutValue s Q K M Out sm_scale i) := by
  constructor
  · exact attention_score_python_case1_surface_toAlgorithm_supported
      Q K M Out sm_scale
  · exact attention_score_case1_surface_compute_correct Q K M Out sm_scale s

/-- `output_summary` alias for Python attention-score case 2. -/
abbrev attention_score_python_case2_output_summary
    (Q K M Score Out : RegionName) (sm_scale : ℝ) (s : BlockState) :=
  attention_score_python_case2_output_surface_summary Q K M Score Out sm_scale s

/-- `output_summary` alias for Python attention-score case 3. -/
abbrev attention_score_python_case3_output_summary
    (Q K M Score Out : RegionName) (sm_scale : ℝ) (s : BlockState) :=
  attention_score_python_case3_output_surface_summary Q K M Score Out sm_scale s

/-- `output_summary` alias for Python attention-score case 4. -/
abbrev attention_score_python_case4_output_summary
    (Q K M Score Out : RegionName) (sm_scale : ℝ) (s : BlockState) :=
  attention_score_python_case4_output_surface_summary Q K M Score Out sm_scale s

end VeriTile.Bench.TritonBenchG.AttentionScore
