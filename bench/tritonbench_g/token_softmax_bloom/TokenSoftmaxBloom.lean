import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `token_softmax_bloom.py`'s
`_fwd_kernel_token_softmax`.

The metadata buffers are typed Nat regions so their `tl.load` calls do not need
extra `dtype=` kwargs. -/
def token_softmax_surface
    (Logics : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  row = (tl.load(Logics + cur_head * $(stride_logic_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_logic_bs),
    mask=col_offsets < cur_batch_seq_len, other=-float("inf"))).to(tl.float32)
  row_minus_max = row - tl.max(row, axis=0)
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  tl.store(Prob_Out + cur_head * $(stride_prob_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_prob_bs),
    softmax_output, mask=col_offsets < cur_batch_seq_len)
}

/-- Proof-oriented final probability store slice of `token_softmax_bloom.py`'s
`_fwd_kernel_token_softmax`.

The full kernel computes the row softmax with max/exp/sum reductions. This
slice starts from a precomputed `Softmax` region and proves the masked
destination writeback using `B_Start_Loc` and `B_Seqlen`. -/
def token_softmax_final_store_slice
    (Softmax : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Prob_Out : RegionName)
    (stride_softmax_h stride_softmax_bs stride_prob_h stride_prob_bs
      BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  softmax_output = tl.load(Softmax + cur_head * $(stride_softmax_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_softmax_bs),
    mask=col_offsets < cur_batch_seq_len, other=0.0)
  tl.store(Prob_Out + cur_head * $(stride_prob_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_prob_bs),
    softmax_output, mask=col_offsets < cur_batch_seq_len)
}

def seqLen (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0)

def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)

def tokenIndex (s : BlockState) (B_Start_Loc : RegionName) (i : Fin BLOCK_SIZE) : Nat :=
  startLoc s B_Start_Loc + i.val

def active
    (s : BlockState) (B_Seqlen : RegionName) (i : Fin BLOCK_SIZE) : Prop :=
  i.val < seqLen s B_Seqlen

instance activeDecidable
    (s : BlockState) (B_Seqlen : RegionName) (i : Fin BLOCK_SIZE) :
    Decidable (active s B_Seqlen i) := by
  unfold active
  infer_instance

def softmaxOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_softmax_h stride_softmax_bs : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * stride_softmax_h + tokenIndex s B_Start_Loc i * stride_softmax_bs

def probOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_prob_h stride_prob_bs : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * stride_prob_h + tokenIndex s B_Start_Loc i * stride_prob_bs

noncomputable def softmaxStoreValue
    (s : BlockState) (Softmax B_Start_Loc B_Seqlen : RegionName)
    (stride_softmax_h stride_softmax_bs : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (if i.val < seqLen s B_Seqlen then
      some (s.readMem Softmax
        (softmaxOffset s B_Start_Loc stride_softmax_h stride_softmax_bs i))
    else some (0.0 : ℝ))

/-- Algorithm-layer correctness for the masked token-softmax probability store. -/
theorem token_softmax_final_store_slice_correct
    (Softmax B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_softmax_h stride_softmax_bs stride_prob_h stride_prob_bs
      BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)) :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := probOffset s B_Start_Loc stride_prob_h stride_prob_bs i
      (exec (token_softmax_final_store_slice Softmax B_Start_Loc B_Seqlen Prob_Out
            stride_softmax_h stride_softmax_bs stride_prob_h stride_prob_bs
            BLOCK_SIZE) s).map (·.readMem Prob_Out outAddr)
        = some (if active s B_Seqlen i then
            softmaxStoreValue s Softmax B_Start_Loc B_Seqlen
              stride_softmax_h stride_softmax_bs i
          else s.readMem Prob_Out outAddr) := by
  intro i
  simp [exec, token_softmax_final_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, BlockState.readMemValue,
        seqLen, startLoc, tokenIndex, softmaxOffset, probOffset]
  let offsetFn : TileIndex [BLOCK_SIZE] → Nat :=
    fun idx =>
      s.pids 1 * stride_prob_h +
        (s.readMemValue .nat B_Start_Loc (s.pids 0) + idx.1.val) * stride_prob_bs
  let valueFn : TileIndex [BLOCK_SIZE] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if idx.1.val < s.readMemValue .nat B_Seqlen (s.pids 0) then
          some (s.readMem Softmax
            (s.pids 1 * stride_softmax_h +
              (s.readMemValue .nat B_Start_Loc (s.pids 0) + idx.1.val) *
                stride_softmax_bs))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_SIZE] → Prop :=
    fun idx => idx.1.val < s.readMemValue .nat B_Seqlen (s.pids 0)
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : probOffset s B_Start_Loc stride_prob_h stride_prob_bs a =
        probOffset s B_Start_Loc stride_prob_h stride_prob_bs b := by
      simpa [offsetFn, probOffset, tokenIndex, startLoc, BlockState.readMemValue] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem Prob_Out (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BLOCK_SIZE])).readMem Prob_Out
        (offsetFn (i, PUnit.unit)) =
    if active s B_Seqlen i then
      softmaxStoreValue s Softmax B_Start_Loc B_Seqlen stride_softmax_h
        stride_softmax_bs i
    else s.readMem Prob_Out (offsetFn (i, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : P (i, PUnit.unit)
  · rw [if_pos hi]
    have hraw :
        i.val <
          (match s.readMemTyped TileDType.nat B_Seqlen (s.pids 0) with
          | some value => value
          | none => BlockState.defaultCarrier TileDType.nat) := by
      simpa [P, BlockState.readMemValue] using hi
    simp [valueFn, P, active, softmaxStoreValue, seqLen, startLoc, tokenIndex,
      softmaxOffset, BlockState.readMemValue, hi, hraw]
    intro hle
    exact False.elim ((not_lt_of_ge hle) hraw)
  · rw [if_neg hi]
    have hraw :
        ¬ i.val <
          (match s.readMemTyped TileDType.nat B_Seqlen (s.pids 0) with
          | some value => value
          | none => BlockState.defaultCarrier TileDType.nat) := by
      simpa [P, BlockState.readMemValue] using hi
    simp [P, active, softmaxStoreValue, seqLen, BlockState.readMemValue, hi, hraw]
    intro hcontr
    exact False.elim (hraw hcontr)

/-- Compute-facing correctness for the masked token-softmax probability store. -/
theorem token_softmax_final_store_slice_compute_correct
    (Softmax B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_softmax_h stride_softmax_bs stride_prob_h stride_prob_bs
      BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)) :
    ComputeCorrect.Realizes
      (kernel := token_softmax_final_store_slice Softmax B_Start_Loc B_Seqlen Prob_Out
        stride_softmax_h stride_softmax_bs stride_prob_h stride_prob_bs
        BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s B_Seqlen i)
        (fun i => (Prob_Out, probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)))
      (expected := fun i =>
        softmaxStoreValue s Softmax B_Start_Loc B_Seqlen stride_softmax_h
          stride_softmax_bs i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [token_softmax_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := token_softmax_final_store_slice_correct Softmax B_Start_Loc
    B_Seqlen Prob_Out stride_softmax_h stride_softmax_bs stride_prob_h
    stride_prob_bs BLOCK_SIZE s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom
