import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Surface transcription of the active-LoRA path in `sgmv_expand_slice.py`'s
`_sgmv_expand_slice_kernel`.

This keeps the CTA decomposition, sequence metadata loads, LoRA index gather,
K-block dot-product loop for the `EVEN_K=true` path used by the benchmark,
optional `CAST_TYPE`, optional `ADD_INPUTS`, and final masked store.
`tl.max_contiguous` is a layout hint and is omitted; the loads use the same
modulo offsets directly. The Python early returns for
`pid_m * BLOCK_M > M` and `lora_index == -1` are not represented because the
current DSL lacks kernel-level `return` and the LoRA sentinel is signed. -/
def sgmv_expand_slice_active_surface
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K : Nat)
    (ADD_INPUTS CAST_TYPE : Bool) :
    ComputeKernel := triton {
  tl.region seq_lens = tl.uint64, b_seq_start_loc = tl.uint64, lora_indices = tl.uint64
  pid = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  cta_n_num = tl.cdiv($(N), $(BLOCK_N))
  pid_m = pid // cta_n_num
  pid_n = pid % cta_n_num
  m_len = tl.load(seq_lens + cur_batch, dtype=tl.uint64)
  lora_index = tl.load(lora_indices + cur_batch, dtype=tl.uint64)
  cur_seq_start = tl.load(b_seq_start_loc + cur_batch, dtype=tl.uint64)
  offset_m = tl.arange(0, $(BLOCK_M)) + pid_m * $(BLOCK_M)
  offset_n = tl.arange(0, $(BLOCK_N)) + pid_n * $(BLOCK_N)
  offset_k = tl.arange(0, $(BLOCK_K))
  ram = tl.multiple_of(offset_m % m_len, $(BLOCK_M))
  rbn = tl.multiple_of(offset_n % $(N), $(BLOCK_N))
  a_ptr = input_ptr + cur_seq_start * $(xm_stride) +
    ram[:, None] * $(xm_stride) + offset_k[None, :] * $(xk_stride)
  b_ptr = lora_ptr + $(l0_stride) * lora_index +
    offset_k[:, None] * $(lora_n_stride) + rbn[None, :] * $(lora_k_stride)
  accumulator = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  for k_iter in range($(0), tl.cdiv($(K), $(BLOCK_K)), $(1)) {
    tiled_a = tl.load(a_ptr)
    tiled_b = tl.load(b_ptr)
    if CAST_TYPE {
      tiled_a = (tiled_a).to(lora_ptr.dtype.element_ty)
    }
    accumulator += tl.dot(tiled_a, tiled_b)
    a_ptr += $(BLOCK_K) * $(xk_stride)
    b_ptr += $(BLOCK_K) * $(lora_n_stride)
  }
  tiled_c = (accumulator).to(lora_ptr.dtype.element_ty)
  offset_cm = cur_seq_start + tl.arange(0, $(BLOCK_M)) + pid_m * $(BLOCK_M)
  offset_cn = tl.arange(0, $(BLOCK_N)) + pid_n * $(BLOCK_N) + $(slice_offset)
  c_ptr = out_ptr + offset_cm[:, None] * $(cm_stride) +
    offset_cn[None, :] * $(cn_stride)
  seq_limit = cur_seq_start + m_len
  c_mask = (offset_cm[:, None] < seq_limit) and
    ((offset_cn[None, :] - $(slice_offset)) < $(N))
  if ADD_INPUTS {
    tiled_out = tl.load(c_ptr, mask=c_mask)
    tiled_c += tiled_out
  }
  tl.store(c_ptr, tiled_c, mask=c_mask)
}

/-- Proof-oriented one-row, one-output-block slice of `sgmv_expand_slice.py`'s
`_sgmv_expand_slice_kernel`.

This captures the ADD_INPUTS=false, no-cast path after the CTA has selected a
sequence batch, a row block, and an `n` block: load the row from the flattened
sequence input, load the selected LoRA-B tile, reduce over K, and store the
output slice at `cur_seq_start + row`.

The in-body `tl.region` directive declares each int64 metadata buffer
(`seq_lens`, `b_seq_start_loc`, `lora_indices`) so loads from them
default to the `.nat` channel without explicit `dtype=` kwargs. -/
def sgmv_expand_slice_one_row_block
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  tl.region seq_lens = tl.uint64, b_seq_start_loc = tl.uint64, lora_indices = tl.uint64
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  cur_batch = tl.program_id(2)
  m_len = tl.load(seq_lens + cur_batch)
  cur_seq_start = tl.load(b_seq_start_loc + cur_batch)
  lora_index = tl.load(lora_indices + cur_batch)
  row = pid_m * $(BLOCK_M)
  offset_k = tl.arange(0, $(BLOCK_K))
  offset_n = tl.arange(0, $(BLOCK_N))
  tiled_a = tl.load(input_ptr + cur_seq_start * $(xm_stride) +
      row * $(xm_stride) + offset_k * $(xk_stride),
    mask=(row < m_len) and (offset_k < $(K)), other=0.0)
  tiled_b = tl.load(lora_ptr + $(l0_stride) * lora_index +
      offset_k[:, None] * $(lora_n_stride) +
      (pid_n * $(BLOCK_N) + offset_n)[None, :] * $(lora_k_stride),
    mask=(offset_k[:, None] < $(K)) and
      ((pid_n * $(BLOCK_N) + offset_n)[None, :] < $(N)),
    other=0.0)
  accumulator = tl.sum(tiled_a[:, None] * tiled_b, axis=0)
  tiled_c = (accumulator).to(lora_ptr.dtype.element_ty)
  out_n = pid_n * $(BLOCK_N) + offset_n + $(slice_offset)
  tl.store(out_ptr + (cur_seq_start + row) * $(cm_stride) +
      out_n * $(cn_stride),
    tiled_c, mask=(row < m_len) and
      (pid_n * $(BLOCK_N) + offset_n < $(N)))
}

def rowIndex (s : BlockState) (BLOCK_M : Nat) : Nat :=
  s.pids 0 * BLOCK_M

def nIndex (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + i.val

def seqLen (s : BlockState) (seq_lens : RegionName) : Nat :=
  s.readMemValue .nat seq_lens (s.pids 2)

def seqStart (s : BlockState) (b_seq_start_loc : RegionName) : Nat :=
  s.readMemValue .nat b_seq_start_loc (s.pids 2)

def loraIndex (s : BlockState) (lora_indices : RegionName) : Nat :=
  s.readMemValue .nat lora_indices (s.pids 2)

def active (s : BlockState) (seq_lens : RegionName) (N BLOCK_M BLOCK_N : Nat)
    (i : Fin BLOCK_N) : Prop :=
  rowIndex s BLOCK_M < seqLen s seq_lens ∧ nIndex s BLOCK_N i < N

instance activeDecidable
    (s : BlockState) (seq_lens : RegionName) (N BLOCK_M BLOCK_N : Nat)
    (i : Fin BLOCK_N) :
    Decidable (active s seq_lens N BLOCK_M BLOCK_N i) := by
  unfold active
  infer_instance

def outOffset
    (s : BlockState) (b_seq_start_loc : RegionName)
    (cm_stride cn_stride slice_offset BLOCK_M BLOCK_N : Nat)
    (i : Fin BLOCK_N) : Nat :=
  (seqStart s b_seq_start_loc + rowIndex s BLOCK_M) * cm_stride +
    (nIndex s BLOCK_N i + slice_offset) * cn_stride

def inputOffset
    (s : BlockState) (b_seq_start_loc : RegionName)
    (xm_stride xk_stride BLOCK_M : Nat) (j : Fin BLOCK_K) : Nat :=
  seqStart s b_seq_start_loc * xm_stride +
    rowIndex s BLOCK_M * xm_stride + j.val * xk_stride

def loraOffset
    (s : BlockState) (lora_indices : RegionName)
    (l0_stride lora_k_stride lora_n_stride BLOCK_N : Nat)
    (i : Fin BLOCK_N) (j : Fin BLOCK_K) : Nat :=
  l0_stride * loraIndex s lora_indices +
    j.val * lora_n_stride + nIndex s BLOCK_N i * lora_k_stride

noncomputable def sgmvProdTile
    (s : BlockState)
    (input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_M BLOCK_N BLOCK_K : Nat) :
    Tile .real [BLOCK_K, BLOCK_N] :=
  { data := fun idx =>
      let kj := (TileShape.dropInsertedIndex [BLOCK_K] 1 1 (idx.1, 0, PUnit.unit)).1
      let ni := (TileShape.dropInsertedIndex [BLOCK_N] 0 1 (0, idx.2.1, PUnit.unit)).1
      Option.map₂ (fun a b => a * b)
        (if rowIndex s BLOCK_M < seqLen s seq_lens ∧ kj.val < K then
          some (s.readMem input_ptr
            (inputOffset s b_seq_start_loc xm_stride xk_stride BLOCK_M kj))
        else some (0.0 : ℝ))
        (if kj.val < K ∧ nIndex s BLOCK_N ni < N then
          some (s.readMem lora_ptr
            (loraOffset s lora_indices l0_stride lora_k_stride
              lora_n_stride BLOCK_N ni kj))
        else some (0.0 : ℝ)) }

noncomputable def sgmvSpec
    (s : BlockState)
    (input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    ((Tile.reduceSum (shape := [BLOCK_K, BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (sgmvProdTile s input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
        N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        BLOCK_M BLOCK_N BLOCK_K)).data (i, PUnit.unit))

/-- Algorithm-layer correctness for the one-row SGMV expand slice. -/
theorem sgmv_expand_slice_one_row_block_correct
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N =>
        outOffset s b_seq_start_loc cm_stride cn_stride slice_offset
          BLOCK_M BLOCK_N i))
    (hExec : exec (sgmv_expand_slice_one_row_block input_ptr lora_ptr out_ptr
        b_seq_start_loc seq_lens lora_indices N K xm_stride xk_stride l0_stride
        lora_k_stride lora_n_stride cm_stride cn_stride slice_offset
        BLOCK_M BLOCK_N BLOCK_K) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem out_ptr
          (outOffset s b_seq_start_loc cm_stride cn_stride slice_offset
            BLOCK_M BLOCK_N i) =
        if active s seq_lens N BLOCK_M BLOCK_N i then
          sgmvSpec s input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
            N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
            BLOCK_M BLOCK_N BLOCK_K i
        else
          s.readMem out_ptr
            (outOffset s b_seq_start_loc cm_stride cn_stride slice_offset
              BLOCK_M BLOCK_N i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        (s.readMemValue .nat b_seq_start_loc (s.pids 2) +
            s.pids 0 * BLOCK_M) * cm_stride +
          (s.pids 1 * BLOCK_N + idx.1.val + slice_offset) * cn_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset, rowIndex, nIndex, seqStart] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBN : 0 < BLOCK_N
  · simp [exec, sgmv_expand_slice_one_row_block, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, ComparableDType.lt,
          BlockState.readMemValue, hBN] at hExec
    rw [← hExec]
    simpa [active, sgmvSpec, sgmvProdTile, inputOffset, loraOffset,
            loraIndex, seqLen, seqStart, outOffset, rowIndex, nIndex,
            Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex,
            BlockState.readMemValue] using
      (BlockState.scatter_readback_prop_masked_nd
        (region := out_ptr)
        (shape := [BLOCK_N])
        (offsetFn := fun idx : TileIndex [BLOCK_N] =>
          (s.readMemValue .nat b_seq_start_loc (s.pids 2) +
              s.pids 0 * BLOCK_M) * cm_stride +
            (s.pids 1 * BLOCK_N + idx.1.val + slice_offset) * cn_stride)
        (valueFn := fun idx : TileIndex [BLOCK_N] =>
          WithBot.unbotD 0
            ((Tile.reduceSum (shape := [BLOCK_K, BLOCK_N]) ⟨0, by simp⟩
              Bool.false
              (sgmvProdTile s input_ptr lora_ptr b_seq_start_loc seq_lens
                lora_indices N K xm_stride xk_stride l0_stride lora_k_stride
                lora_n_stride BLOCK_M BLOCK_N BLOCK_K)).data
              (idx.1, PUnit.unit)))
        (P := fun idx : TileIndex [BLOCK_N] =>
          s.pids 0 * BLOCK_M < s.readMemValue .nat seq_lens (s.pids 2) ∧
            s.pids 1 * BLOCK_N + idx.1.val < N)
        _ hRawInj (i, PUnit.unit))
  · exact False.elim (hBN (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-row SGMV expand slice. -/
theorem sgmv_expand_slice_one_row_block_compute_correct
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N =>
        outOffset s b_seq_start_loc cm_stride cn_stride slice_offset
          BLOCK_M BLOCK_N i)) :
    ComputeCorrect.Realizes
      (kernel := sgmv_expand_slice_one_row_block input_ptr lora_ptr out_ptr
        b_seq_start_loc seq_lens lora_indices N K xm_stride xk_stride l0_stride
        lora_k_stride lora_n_stride cm_stride cn_stride slice_offset
        BLOCK_M BLOCK_N BLOCK_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => active s seq_lens N BLOCK_M BLOCK_N i)
        (fun i => (out_ptr,
          outOffset s b_seq_start_loc cm_stride cn_stride slice_offset
            BLOCK_M BLOCK_N i)))
      (expected := fun i =>
        sgmvSpec s input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
          N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
          BLOCK_M BLOCK_N BLOCK_K i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [sgmv_expand_slice_one_row_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := sgmv_expand_slice_one_row_block_correct input_ptr lora_ptr out_ptr
    b_seq_start_loc seq_lens lora_indices N K xm_stride xk_stride l0_stride
    lora_k_stride lora_n_stride cm_stride cn_stride slice_offset
    BLOCK_M BLOCK_N BLOCK_K s s' hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.SgmvExpandSlice
