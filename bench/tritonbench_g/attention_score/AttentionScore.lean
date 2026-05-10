import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.AttentionScore

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Proof-oriented final score-vector store slice of `attention_score.py`'s
`_score_kernel`.

The full kernel computes the score accumulator `o` from Q/K blocks, max values,
and optional sliding-window masking. This slice starts after that reduction with
a precomputed `Score` vector. It uses program axes `(start_n, off_z, off_h)` and
proves the final `o_range < NKV_CTX` masked writeback into `Out`. -/
def attention_score_final_store_slice
    (Score Out : RegionName)
    (stride_score_z stride_score_h stride_score_n
      stride_oz stride_oh stride_on
      NKV_CTX BLOCK_N : Nat) :
    ComputeKernel := triton {
  start_n = tl.program_id(axis=0)
  off_z = tl.program_id(axis=1)
  off_h = tl.program_id(axis=2)
  offs_n = tl.arange(0, $(BLOCK_N))
  o_range = start_n * $(BLOCK_N) + offs_n
  score = tl.load(Score + off_z * $(stride_score_z) + off_h * $(stride_score_h) +
      o_range * $(stride_score_n), mask=o_range < $(NKV_CTX), other=0.0)
  tl.store(Out + off_z * $(stride_oz) + off_h * $(stride_oh) +
      o_range * $(stride_on), score, mask=o_range < $(NKV_CTX))
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
    (s : BlockState) (stride_oz stride_oh stride_on BLOCK_N : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * stride_oz + s.pids 2 * stride_oh + nIndex s BLOCK_N i * stride_on

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
      stride_oz stride_oh stride_on
      NKV_CTX BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s stride_oz stride_oh stride_on BLOCK_N i)) :
    ∀ i : Fin BLOCK_N,
      let outAddr := outOffset s stride_oz stride_oh stride_on BLOCK_N i
      (exec (attention_score_final_store_slice Score Out stride_score_z
            stride_score_h stride_score_n stride_oz stride_oh stride_on
            NKV_CTX BLOCK_N) s).map (·.readMem Out outAddr)
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
      (s.pids 0 * BLOCK_N + idx.1.val) * stride_on
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
    have habFin : outOffset s stride_oz stride_oh stride_on BLOCK_N a =
        outOffset s stride_oz stride_oh stride_on BLOCK_N b := by
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
      stride_oz stride_oh stride_on
      NKV_CTX BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s stride_oz stride_oh stride_on BLOCK_N i)) :
    ComputeCorrect.Realizes
      (kernel := attention_score_final_store_slice Score Out stride_score_z
        stride_score_h stride_score_n stride_oz stride_oh stride_on NKV_CTX
        BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => active s NKV_CTX BLOCK_N i)
        (fun i => (Out, outOffset s stride_oz stride_oh stride_on BLOCK_N i)))
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
    stride_score_h stride_score_n stride_oz stride_oh stride_on NKV_CTX BLOCK_N
    s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.AttentionScore
