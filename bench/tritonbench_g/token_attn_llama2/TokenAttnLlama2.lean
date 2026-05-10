import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.TokenAttnLlama2

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Proof-oriented attention-score store slice of `token_attn_llama2.py`'s
`_fwd_kernel_token_att1`.

The full kernel gathers K, computes `sum(q * k) * sm_scale`, and stores a block
of attention scores. This slice starts from a precomputed `AttValue` vector and
proves the masked writeback into `Att_Out`, preserving the source sequence
window mask. -/
def token_attn_llama2_score_store_slice
    (AttValue B_Start_Loc B_Seqlen Att_Out : RegionName)
    (max_input_len att_value_stride_h att_value_stride_bs
      att_stride_h att_stride_bs BLOCK_N : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(axis=0)
  cur_head = tl.program_id(axis=1)
  start_n = tl.program_id(axis=2)
  offs_n = start_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch, dtype=tl.uint64)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch, dtype=tl.uint64)
  cur_batch_start_index = $(max_input_len) - cur_batch_seq_len
  cur_batch_end_index = $(max_input_len)
  offs_n_new = cur_batch_start_index + offs_n
  att_value = tl.load(AttValue + cur_head * $(att_value_stride_h) +
      (cur_batch_in_all_start_index + offs_n) * $(att_value_stride_bs),
    mask=offs_n_new < cur_batch_end_index, other=0.0)
  tl.store(Att_Out + cur_head * $(att_stride_h) +
      (cur_batch_in_all_start_index + offs_n) * $(att_stride_bs),
    att_value, mask=offs_n_new < cur_batch_end_index)
}

def seqLen (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0)

def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)

def blockOffset (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 2 * BLOCK_N + i.val

def active
    (s : BlockState) (B_Seqlen : RegionName) (max_input_len BLOCK_N : Nat)
    (i : Fin BLOCK_N) : Prop :=
  max_input_len - seqLen s B_Seqlen + blockOffset s BLOCK_N i < max_input_len

instance activeDecidable
    (s : BlockState) (B_Seqlen : RegionName) (max_input_len BLOCK_N : Nat)
    (i : Fin BLOCK_N) :
    Decidable (active s B_Seqlen max_input_len BLOCK_N i) := by
  unfold active
  infer_instance

def attValueOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (att_value_stride_h att_value_stride_bs BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * att_value_stride_h +
    (startLoc s B_Start_Loc + blockOffset s BLOCK_N i) * att_value_stride_bs

def outOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (att_stride_h att_stride_bs BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * att_stride_h +
    (startLoc s B_Start_Loc + blockOffset s BLOCK_N i) * att_stride_bs

noncomputable def attStoreValue
    (s : BlockState) (AttValue B_Start_Loc B_Seqlen : RegionName)
    (max_input_len att_value_stride_h att_value_stride_bs BLOCK_N : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (if active s B_Seqlen max_input_len BLOCK_N i then
      some (s.readMem AttValue
        (attValueOffset s B_Start_Loc att_value_stride_h att_value_stride_bs
          BLOCK_N i))
    else some (0.0 : ℝ))

/-- Algorithm-layer correctness for the masked Llama token-attention score store. -/
theorem token_attn_llama2_score_store_slice_correct
    (AttValue B_Start_Loc B_Seqlen Att_Out : RegionName)
    (max_input_len att_value_stride_h att_value_stride_bs
      att_stride_h att_stride_bs BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)) :
    ∀ i : Fin BLOCK_N,
      let outAddr := outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i
      (exec (token_attn_llama2_score_store_slice AttValue B_Start_Loc B_Seqlen
            Att_Out max_input_len att_value_stride_h att_value_stride_bs
            att_stride_h att_stride_bs BLOCK_N) s).map (·.readMem Att_Out outAddr)
        = some (if active s B_Seqlen max_input_len BLOCK_N i then
            attStoreValue s AttValue B_Start_Loc B_Seqlen max_input_len
              att_value_stride_h att_value_stride_bs BLOCK_N i
          else s.readMem Att_Out outAddr) := by
  intro i
  simp [exec, token_attn_llama2_score_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.sub, NumericDType.mul, ComparableDType.lt,
        BlockState.readMemValue, seqLen, startLoc, blockOffset, active,
        attValueOffset, outOffset]
  let offsetFn : TileIndex [BLOCK_N] → Nat :=
    fun idx =>
      s.pids 1 * att_stride_h +
        (s.readMemValue .nat B_Start_Loc (s.pids 0) +
          (s.pids 2 * BLOCK_N + idx.1.val)) * att_stride_bs
  let valueFn : TileIndex [BLOCK_N] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if max_input_len - s.readMemValue .nat B_Seqlen (s.pids 0) +
              (s.pids 2 * BLOCK_N + idx.1.val) < max_input_len then
          some (s.readMem AttValue
            (s.pids 1 * att_value_stride_h +
              (s.readMemValue .nat B_Start_Loc (s.pids 0) +
                (s.pids 2 * BLOCK_N + idx.1.val)) * att_value_stride_bs))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_N] → Prop :=
    fun idx =>
      max_input_len - s.readMemValue .nat B_Seqlen (s.pids 0) +
        (s.pids 2 * BLOCK_N + idx.1.val) < max_input_len
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N a =
        outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N b := by
      simpa [offsetFn, outOffset, startLoc, blockOffset, BlockState.readMemValue] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem Att_Out (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BLOCK_N])).readMem Att_Out
        (offsetFn (i, PUnit.unit)) =
    if P (i, PUnit.unit) then
      attStoreValue s AttValue B_Start_Loc B_Seqlen max_input_len
        att_value_stride_h att_value_stride_bs BLOCK_N i
    else s.readMem Att_Out (offsetFn (i, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : P (i, PUnit.unit)
  · rw [if_pos hi]
    have hraw :
        max_input_len -
            (match s.readMemTyped TileDType.nat B_Seqlen (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.nat) +
          (s.pids 2 * BLOCK_N + i.val) < max_input_len := by
      simpa [P, BlockState.readMemValue] using hi
    simp [valueFn, P, active, attStoreValue, seqLen, startLoc, blockOffset,
      attValueOffset, outOffset, BlockState.readMemValue, hi, hraw]
    intro hle
    exact False.elim ((not_lt_of_ge hle) hraw)
  · rw [if_neg hi]
    have hraw :
        ¬ max_input_len -
            (match s.readMemTyped TileDType.nat B_Seqlen (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.nat) +
          (s.pids 2 * BLOCK_N + i.val) < max_input_len := by
      simpa [P, BlockState.readMemValue] using hi
    simp [P, active, attStoreValue, seqLen, startLoc, blockOffset,
      BlockState.readMemValue, hi, hraw]
    intro hcontr
    exact False.elim (hraw hcontr)

/-- Compute-facing correctness for the masked Llama token-attention score store. -/
theorem token_attn_llama2_score_store_slice_compute_correct
    (AttValue B_Start_Loc B_Seqlen Att_Out : RegionName)
    (max_input_len att_value_stride_h att_value_stride_bs
      att_stride_h att_stride_bs BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)) :
    ComputeCorrect.Realizes
      (kernel := token_attn_llama2_score_store_slice AttValue B_Start_Loc B_Seqlen
        Att_Out max_input_len att_value_stride_h att_value_stride_bs
        att_stride_h att_stride_bs BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => active s B_Seqlen max_input_len BLOCK_N i)
        (fun i => (Att_Out, outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)))
      (expected := fun i =>
        attStoreValue s AttValue B_Start_Loc B_Seqlen max_input_len
          att_value_stride_h att_value_stride_bs BLOCK_N i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [token_attn_llama2_score_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := token_attn_llama2_score_store_slice_correct AttValue B_Start_Loc
    B_Seqlen Att_Out max_input_len att_value_stride_h att_value_stride_bs
    att_stride_h att_stride_bs BLOCK_N s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.TokenAttnLlama2
