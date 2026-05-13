import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.AttentionScore

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Documented `BLOCK_M == BLOCK_N` specialization of
`attention_score.py`'s `_score_kernel`.

This surface keeps the wrapper invariant `BLOCK_M == BLOCK_N` explicit by
shaping `o` as `[BLOCK_N]`, matching `tl.sum(p, axis=0)`. -/
def attention_score_kernel
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh _stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      sliding_window_offset sliding_window_size
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (sm_scale : ℝ)
    (SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N : Bool) :
    ComputeKernel := triton {
  start_n = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  off_hkv = off_h // ($(H) // $(H_KV))
  q_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  k_offset = (off_z).to(tl.int64) * $(stride_kz) + (off_hkv).to(tl.int64) * $(stride_kh)
  m_ptrs = M + off_hz * $(ROUND_CTX) + tl.arange(0, $(BLOCK_M))
  o = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
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
  simp [exec, attention_score_final_store_slice, stepStmts, stepStmt, evalOp,
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

end VeriTile.Bench.TritonBenchG.AttentionScore
