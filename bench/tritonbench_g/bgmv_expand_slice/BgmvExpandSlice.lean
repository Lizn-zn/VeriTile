import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of
`bgmv_expand_slice.py`'s `_bgmv_expand_slice_kernel`.

This covers the `EVEN_K` load path, `tl.cdiv` split length, output-block
loop, optional `CAST_TYPE` conversion, and `ADD_INPUTS` accumulation path.
Python's signed `lora_index == -1` early return is represented as a guard
around the active body. -/
def bgmv_expand_slice_surface
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .int)
    (xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
      cn_stride slice_offset BLOCK_N BLOCK_K SPLIT_N : Nat)
    (EVEN_K ADD_INPUTS CAST_TYPE : Bool) :
    ComputeKernel := triton {
  pid_sn = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  if lora_index != $((-1 : Int)) {
    offset_k = tl.arange(0, $(BLOCK_K))
    offset_n = tl.arange(0, $(BLOCK_N))
    if EVEN_K {
      tiled_a = tl.load(input_ptr + cur_batch * $(xm_stride) + offset_k * $(xk_stride))
    } else {
      tiled_a = tl.load(input_ptr + cur_batch * $(xm_stride) + offset_k * $(xk_stride),
        mask=offset_k < $(K), other=0)
    }
    split_n_length = tl.cdiv($(N), $(SPLIT_N))
    if CAST_TYPE {
      tiled_a = (tiled_a).to(lora_ptr.dtype.element_ty)
    }
    b_ptr = lora_ptr + $(l0_stride) * lora_index +
      pid_sn * split_n_length * $(lora_k_stride)
    c_ptr = out_ptr + cur_batch * $(cm_stride) + pid_sn * split_n_length +
      $(slice_offset) * $(cn_stride)
    for n in range($(0), split_n_length, $(BLOCK_N)) {
      current_n = n + offset_n
      b_ptr_mask = (current_n[:, None] < split_n_length) & (offset_k[None, :] < $(K))
      c_mask = current_n < split_n_length
      tiled_b = tl.load(
        b_ptr + current_n[:, None] * $(lora_k_stride) +
          offset_k[None, :] * $(lora_n_stride),
        mask=b_ptr_mask, other=0.0)
      if ADD_INPUTS {
        tiled_out = tl.load(c_ptr + current_n * $(cn_stride), mask=c_mask)
        accumulator = tl.sum(tiled_a * tiled_b, 1) + tiled_out
      } else {
        accumulator = tl.sum(tiled_a * tiled_b, 1)
      }
      tl.store(c_ptr + current_n * $(cn_stride), accumulator, mask=c_mask)
    }
  }
}

/-- The full BGMV expand-slice surface lowers to the algorithm layer, including
the signed sentinel guard, split loop, optional cast, add-inputs branch, and
slice offset in the output pointer. -/
theorem bgmv_expand_slice_surface_toAlgorithm_supported
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .int)
    (xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
      cn_stride slice_offset BLOCK_N BLOCK_K SPLIT_N : Nat)
    (EVEN_K ADD_INPUTS CAST_TYPE : Bool) :
    ∃ alg,
      (bgmv_expand_slice_surface input_ptr lora_ptr out_ptr N K lora_indices
        xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride slice_offset BLOCK_N BLOCK_K SPLIT_N EVEN_K ADD_INPUTS
        CAST_TYPE).toAlgorithm? = Except.ok alg := by
  simp [bgmv_expand_slice_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented one-output-block slice of `bgmv_expand_slice.py`'s
`_bgmv_expand_slice_kernel`.

This captures the ADD_INPUTS=false, no-cast path for one `n` block: load one
input vector, load one LoRA-B tile selected by the batch's LoRA index, reduce
over K, and store the resulting output slice. -/
def bgmv_expand_slice_one_block
    (input_ptr lora_ptr out_ptr : RegionName) (lora_indices : Region .nat)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  pid_sn = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  offset_k = tl.arange(0, $(BLOCK_K))
  offset_n = tl.arange(0, $(BLOCK_N))
  tiled_a = tl.load(input_ptr + cur_batch * $(xm_stride) + offset_k * $(xk_stride),
    mask=offset_k < $(K), other=0.0)
  b_ptr = lora_ptr + $(l0_stride) * lora_index +
    pid_sn * $(split_n_length) * $(lora_k_stride)
  c_ptr = out_ptr + cur_batch * $(cm_stride) + pid_sn * $(split_n_length) +
    $(slice_offset) * $(cn_stride)
  current_n = offset_n
  tiled_b = tl.load(
    b_ptr + current_n[:, None] * $(lora_k_stride) +
      offset_k[None, :] * $(lora_n_stride),
    mask=(current_n[:, None] < $(split_n_length)) and (offset_k[None, :] < $(K)),
    other=0.0)
  accumulator = tl.sum(tiled_a * tiled_b, 1)
  tl.store(c_ptr + current_n * $(cn_stride), accumulator,
    mask=current_n < $(split_n_length))
}

def loraIndex (s : BlockState) (lora_indices : RegionName) : Nat :=
  s.readMemValue .nat lora_indices (s.pids 1)

def nIndex (i : Fin BLOCK_N) : Nat :=
  i.val

def outOffset
    (s : BlockState) (split_n_length cm_stride cn_stride slice_offset BLOCK_N : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * cm_stride + s.pids 0 * split_n_length + slice_offset * cn_stride +
    nIndex i * cn_stride

def inputOffset
    (s : BlockState) (xm_stride xk_stride : Nat) (j : Fin BLOCK_K) : Nat :=
  s.pids 1 * xm_stride + j.val * xk_stride

def loraOffset
    (s : BlockState) (lora_indices : RegionName)
    (split_n_length l0_stride lora_k_stride lora_n_stride : Nat)
    (i : Fin BLOCK_N) (j : Fin BLOCK_K) : Nat :=
  l0_stride * loraIndex s lora_indices +
    s.pids 0 * split_n_length * lora_k_stride +
    i.val * lora_k_stride + j.val * lora_n_stride

noncomputable def bgmvProdTile
    (s : BlockState) (input_ptr lora_ptr lora_indices : RegionName)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride BLOCK_N BLOCK_K : Nat) :
    Tile .real [BLOCK_N, BLOCK_K] :=
  { data := fun idx =>
      let ni := (TileShape.dropInsertedIndex [BLOCK_N] 1 1 (idx.1, 0, PUnit.unit)).1
      let kj := (TileShape.dropInsertedIndex [BLOCK_K] 0 1 (0, idx.2.1, PUnit.unit)).1
      Option.map₂ (fun a b => a * b)
        (if kj.val < K then
          some (s.readMem input_ptr (inputOffset s xm_stride xk_stride kj))
        else some (0.0 : ℝ))
        (if ni.val < split_n_length ∧ kj.val < K then
          some (s.readMem lora_ptr
            (loraOffset s lora_indices split_n_length l0_stride lora_k_stride
              lora_n_stride ni kj))
        else some (0.0 : ℝ)) }

noncomputable def bgmvSpec
    (s : BlockState) (input_ptr lora_ptr lora_indices : RegionName)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
      (bgmvProdTile s input_ptr lora_ptr lora_indices K split_n_length
        xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        BLOCK_N BLOCK_K)).data (i, PUnit.unit))

/-- Algorithm-layer correctness for the one-block BGMV expand slice. -/
theorem bgmv_expand_slice_one_block_correct
    (input_ptr lora_ptr out_ptr lora_indices : RegionName)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N =>
        outOffset s split_n_length cm_stride cn_stride slice_offset BLOCK_N i))
    (hExec : exec (bgmv_expand_slice_one_block input_ptr lora_ptr out_ptr
        lora_indices K split_n_length xm_stride xk_stride l0_stride
        lora_k_stride lora_n_stride cm_stride cn_stride slice_offset
        BLOCK_N BLOCK_K) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem out_ptr
          (outOffset s split_n_length cm_stride cn_stride slice_offset BLOCK_N i) =
        if nIndex i < split_n_length then
          bgmvSpec s input_ptr lora_ptr lora_indices K split_n_length
            xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
            BLOCK_N BLOCK_K i
        else
          s.readMem out_ptr
            (outOffset s split_n_length cm_stride cn_stride slice_offset BLOCK_N i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        s.pids 1 * cm_stride + s.pids 0 * split_n_length +
          slice_offset * cn_stride + idx.1.val * cn_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset, nIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBN : 0 < BLOCK_N
  · simp [exec, bgmv_expand_slice_one_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, ComparableDType.lt,
          BlockState.readMemValue, hBN] at hExec
    rw [← hExec]
    simp only [outOffset, nIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
    by_cases hi : i.val < split_n_length
    · simp [hi, bgmvSpec, bgmvProdTile, inputOffset, loraOffset, loraIndex,
            outOffset, nIndex, Tile.reduceSum, Tile.reduceSumDrop,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            BlockState.readMemValue]
      congr 1
      apply Finset.sum_congr rfl
      intro x _
      change Option.map₂ (fun x1 x2 => x1 * x2)
          (if x.val < K then
            some (s.readMem input_ptr
              (s.pids 1 * xm_stride + x.val * xk_stride))
          else some 0.0)
          (if x.val < K then
            some (s.readMem lora_ptr
              ((l0_stride * loraIndex s lora_indices) +
                s.pids 0 * split_n_length * lora_k_stride +
                i.val * lora_k_stride + x.val * lora_n_stride))
          else some 0.0) =
        Option.map₂ (fun x1 x2 => x1 * x2)
          (if x.val < K then
            some (s.readMem input_ptr
              (s.pids 1 * xm_stride + x.val * xk_stride))
          else some 0.0)
          (if i.val < split_n_length ∧ x.val < K then
            some (s.readMem lora_ptr
              ((l0_stride * loraIndex s lora_indices) +
                s.pids 0 * split_n_length * lora_k_stride +
                i.val * lora_k_stride + x.val * lora_n_stride))
          else some 0.0)
      by_cases hxK : x.val < K
      · simp [hxK, hi]
      · simp [hxK]
    · simp [hi]
  · exact False.elim (hBN (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-block BGMV expand slice. -/
theorem bgmv_expand_slice_one_block_compute_correct
    (input_ptr lora_ptr out_ptr lora_indices : RegionName)
    (K split_n_length xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N =>
        outOffset s split_n_length cm_stride cn_stride slice_offset BLOCK_N i)) :
    ComputeCorrect.Realizes
      (kernel := bgmv_expand_slice_one_block input_ptr lora_ptr out_ptr
        lora_indices K split_n_length xm_stride xk_stride l0_stride
        lora_k_stride lora_n_stride cm_stride cn_stride slice_offset
        BLOCK_N BLOCK_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => nIndex i < split_n_length)
        (fun i => (out_ptr,
          outOffset s split_n_length cm_stride cn_stride slice_offset BLOCK_N i)))
      (expected := fun i =>
        bgmvSpec s input_ptr lora_ptr lora_indices K split_n_length
          xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
          BLOCK_N BLOCK_K i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bgmv_expand_slice_one_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := bgmv_expand_slice_one_block_correct input_ptr lora_ptr out_ptr
    lora_indices K split_n_length xm_stride xk_stride l0_stride lora_k_stride
    lora_n_stride cm_stride cn_stride slice_offset BLOCK_N BLOCK_K
    s s' hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.BgmvExpandSlice
