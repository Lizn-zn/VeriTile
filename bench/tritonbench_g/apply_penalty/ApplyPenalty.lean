import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ApplyPenalty

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Proof-oriented scatter-back slice of `apply_penalty.py`'s
`_fwd_kernel_apply_penalty`.

The full kernel computes `pre_logits` from presence/frequency/repetition
penalties before the final masked scatter. This slice starts after that
arithmetic has been materialized in `Adjusted`, and proves the dynamic
token-id/cumsum writeback into `Logits`. -/
def apply_penalty_scatter_back
    (Logits Adjusted p_token_ids p_cumsum_seq_len : RegionName)
    (stride_logit_b BLOCK_P : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch, dtype=tl.uint64)
  cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + $(1), dtype=tl.uint64)
  cur_batch_id_offset = cur_batch_start_index + tl.arange(0, $(BLOCK_P))
  batch_ids = tl.load(p_token_ids + cur_batch_id_offset,
    mask=cur_batch_id_offset < cur_batch_end_index, other=$(0), dtype=tl.uint64)
  pre_logits = tl.load(Adjusted + cur_batch_id_offset,
    mask=cur_batch_id_offset < cur_batch_end_index, other=0.0)
  tl.store(Logits + cur_batch * $(stride_logit_b) + batch_ids, pre_logits,
    mask=cur_batch_id_offset < cur_batch_end_index)
}

def batchStart (s : BlockState) (p_cumsum_seq_len : RegionName) : Nat :=
  s.readMemValue .nat p_cumsum_seq_len (s.pids 0)

def batchEnd (s : BlockState) (p_cumsum_seq_len : RegionName) : Nat :=
  s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1)

def tokenOffset (s : BlockState) (p_cumsum_seq_len : RegionName)
    (i : Fin BLOCK_P) : Nat :=
  batchStart s p_cumsum_seq_len + i.val

def active (s : BlockState) (p_cumsum_seq_len : RegionName)
    (i : Fin BLOCK_P) : Prop :=
  tokenOffset s p_cumsum_seq_len i < batchEnd s p_cumsum_seq_len

instance activeDecidable (s : BlockState) (p_cumsum_seq_len : RegionName)
    (i : Fin BLOCK_P) :
    Decidable (active s p_cumsum_seq_len i) := by
  unfold active
  infer_instance

def tokenId (s : BlockState) (p_token_ids p_cumsum_seq_len : RegionName)
    (i : Fin BLOCK_P) : Nat :=
  s.readMemValue .nat p_token_ids (tokenOffset s p_cumsum_seq_len i)

def storeOffset
    (s : BlockState) (p_token_ids p_cumsum_seq_len : RegionName)
    (stride_logit_b : Nat) (i : Fin BLOCK_P) : Nat :=
  s.pids 0 * stride_logit_b +
    if active s p_cumsum_seq_len i then
      tokenId s p_token_ids p_cumsum_seq_len i
    else
      0

def adjustedOffset (s : BlockState) (p_cumsum_seq_len : RegionName)
    (i : Fin BLOCK_P) : Nat :=
  tokenOffset s p_cumsum_seq_len i

def preStoreState
    (Adjusted p_token_ids p_cumsum_seq_len : RegionName)
    (BLOCK_P : Nat) (s : BlockState) : BlockState :=
  s.setReg "cur_batch" TileDType.nat [] (Tile.scalar (s.pids 0))
    |>.setReg "cur_batch_start_index" TileDType.nat []
      (Tile.scalar (s.readMemValue .nat p_cumsum_seq_len (s.pids 0)))
    |>.setReg "cur_batch_end_index" TileDType.nat []
      (Tile.scalar (s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1)))
    |>.setReg "cur_batch_id_offset" TileDType.nat [BLOCK_P]
      (Tile.vec fun i => s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + i.val)
    |>.setReg "batch_ids" TileDType.nat [BLOCK_P]
      { data := fun i =>
        if s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + i.1.val <
            s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1) then
          s.readMemValue .nat p_token_ids
            (s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + i.1.val)
        else
          0 }
    |>.setReg "pre_logits" TileDType.real [BLOCK_P]
      { data := fun i =>
        if s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + i.1.val <
            s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1) then
          some (s.readMem Adjusted
            (s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + i.1.val))
        else
          some (0.0 : ℝ) }

noncomputable def rawStoreValue
    (s : BlockState) (Adjusted p_cumsum_seq_len : RegionName)
    (idx : TileIndex [BLOCK_P]) : ℝ :=
  WithBot.unbotD 0
    (if s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + idx.1.val <
        s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1) then
      some (s.readMem Adjusted
        (s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + idx.1.val))
    else
      some (0.0 : ℝ))

/-- Algorithm-layer correctness for the masked token-id scatter-back slice. -/
theorem apply_penalty_scatter_back_correct
    (Logits Adjusted p_token_ids p_cumsum_seq_len : RegionName)
    (stride_logit_b BLOCK_P : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_P =>
        storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b i))
    (hExec : exec (apply_penalty_scatter_back Logits Adjusted p_token_ids
        p_cumsum_seq_len stride_logit_b BLOCK_P) s = some s') :
    ∀ i : Fin BLOCK_P,
      s'.readMem Logits
          (storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b i) =
        if active s p_cumsum_seq_len i then
          rawStoreValue s Adjusted p_cumsum_seq_len (i, PUnit.unit)
        else
          s.readMem Logits
            (storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_P] =>
        s.pids 0 * stride_logit_b +
          if (match s.readMemTyped TileDType.nat p_cumsum_seq_len (s.pids 0) with
                | some value => value
                | none => BlockState.defaultCarrier TileDType.nat) + idx.1.val <
              (match s.readMemTyped TileDType.nat p_cumsum_seq_len (s.pids 0 + 1) with
                | some value => value
                | none => BlockState.defaultCarrier TileDType.nat) then
            match s.readMemTyped TileDType.nat p_token_ids
                ((match s.readMemTyped TileDType.nat p_cumsum_seq_len (s.pids 0) with
                  | some value => value
                  | none => BlockState.defaultCarrier TileDType.nat) + idx.1.val) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.nat
          else
            0) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [storeOffset, active, tokenId, tokenOffset, batchStart, batchEnd,
        BlockState.readMemValue] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBlock : 0 < BLOCK_P
  · simp [exec, apply_penalty_scatter_back, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt,
          BlockState.readMemValue, hBlock] at hExec
    rw [← hExec]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := Logits)
        (shape := [BLOCK_P])
        (s := preStoreState Adjusted p_token_ids p_cumsum_seq_len BLOCK_P s)
        (offsetFn := fun idx : TileIndex [BLOCK_P] =>
          s.pids 0 * stride_logit_b +
            if s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + idx.1.val <
                s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1) then
              s.readMemValue .nat p_token_ids
                (s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + idx.1.val)
            else
              0)
        (valueFn := rawStoreValue s Adjusted p_cumsum_seq_len)
        (P := fun idx : TileIndex [BLOCK_P] =>
          s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + idx.1.val <
            s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1))
        hRawInj (i, PUnit.unit))
    simpa [preStoreState, rawStoreValue, storeOffset, active, tokenId,
      tokenOffset, batchStart, batchEnd, adjustedOffset, BlockState.readMemValue]
      using hScatter
  · exact False.elim (hBlock (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the masked token-id scatter-back slice. -/
theorem apply_penalty_scatter_back_compute_correct
    (Logits Adjusted p_token_ids p_cumsum_seq_len : RegionName)
    (stride_logit_b BLOCK_P : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_P =>
        storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b i)) :
    ComputeCorrect.Realizes
      (kernel := apply_penalty_scatter_back Logits Adjusted p_token_ids
        p_cumsum_seq_len stride_logit_b BLOCK_P)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_P => active s p_cumsum_seq_len i)
        (fun i : Fin BLOCK_P => (Logits,
          storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b i)))
      (expected := fun i : Fin BLOCK_P =>
        rawStoreValue s Adjusted p_cumsum_seq_len (i, PUnit.unit)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [apply_penalty_scatter_back]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := apply_penalty_scatter_back_correct Logits Adjusted p_token_ids
    p_cumsum_seq_len stride_logit_b BLOCK_P s s' hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.ApplyPenalty
