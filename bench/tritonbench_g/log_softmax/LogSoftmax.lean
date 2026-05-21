import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.LogSoftmax

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `log_softmax.py`'s `log_softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_M: tl.constexpr` / `BLOCK_N: tl.constexpr` -> Lean `Nat`
  parameters. -/
def log_softmax_kernel
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  input_ptrs = input_ptr + offset
  inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(tl.float32)
  row_minus_max = inp - tl.max(inp, axis=1)[:, None]
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=1)[:, None]
  softmax_output = tl.log(numerator / denominator)
  output_ptrs = output_ptr + offset
  tl.store(output_ptrs, softmax_output, mask=mask)
}

def mIndex (s : BlockState) (BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  s.pids 0 * BLOCK_M + idx.1.val

def nIndex (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  idx.2.1.val

def outOffset
    (s : BlockState) (N K BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  mIndex s BLOCK_M idx * N * K + nIndex idx * K + s.pids 1

def active
    (s : BlockState) (M N BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Prop :=
  mIndex s BLOCK_M idx < M ∧ nIndex idx < N

instance activeDecidable
    (s : BlockState) (M N BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) :
    Decidable (active s M N BLOCK_M idx) := by
  unfold active
  infer_instance

noncomputable def logSoftmaxInputTile
    (s : BlockState) (input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      if active s M N BLOCK_M idx then
        some (s.readMem input_ptr (outOffset s N K BLOCK_M idx))
      else none }

noncomputable def logSoftmaxSpec
    (s : BlockState) (input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  let inp := logSoftmaxInputTile s input_ptr M N K BLOCK_M BLOCK_N
  match Tile.reduceMax (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.true inp with
  | some rowMax =>
      let rowBroadcast : Broadcast [BLOCK_M, BLOCK_N] [BLOCK_M, 1] [BLOCK_M, BLOCK_N] :=
        Broadcast.consSame (Broadcast.consR Broadcast.nil)
      let shifted := Tile.bop (NumericDType.sub .real) rowBroadcast inp rowMax
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.true numerator
      WithBot.unbotD 0
        ((Tile.uop WithBot.realLog
          (Tile.bop (NumericDType.div .real) rowBroadcast numerator denominator)).data idx)
  | none => 0

/-- Algorithm-layer cellwise correctness for `log_softmax_kernel`. -/
theorem log_softmax_kernel_correct
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] => outOffset s N K BLOCK_M idx))
    (hExec : exec (log_softmax_kernel output_ptr input_ptr M N K BLOCK_M BLOCK_N) s = some s') :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      s'.readMem output_ptr (outOffset s N K BLOCK_M idx) =
        if active s M N BLOCK_M idx then
          logSoftmaxSpec s input_ptr M N K BLOCK_M BLOCK_N idx
        else s.readMem output_ptr (outOffset s N K BLOCK_M idx) := by
  intro idx
  by_cases hBM : 0 < BLOCK_M
  · by_cases hBN : 0 < BLOCK_N
    · simp [exec, log_softmax_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map,
            Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
            Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceMaxKeep,
            Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
            NumericDType.sub, NumericDType.div, ComparableDType.lt,
            FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
            ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, hBN] at hExec
      rw [← hExec]
      have hOffsetInj : Function.Injective
          (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            (s.pids 0 * BLOCK_M + idx.1.val) * N * K + idx.2.1.val * K + s.pids 1) := by
        intro a b h
        apply hOutInj
        simpa [outOffset, mIndex, nIndex] using h
      simp only [outOffset, mIndex, nIndex]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
      by_cases hActive : active s M N BLOCK_M idx
      · rcases hActive with ⟨hM, hN⟩
        simp [active, mIndex, nIndex, hM, hN, logSoftmaxSpec,
              logSoftmaxInputTile, Tile.reduceMax, Tile.reduceMaxDrop,
              Tile.reduceMaxKeep, Tile.reduceSum, Tile.reduceSumDrop,
              Tile.reduceSumKeep, TileShape.axisDim,
              TileShape.eraseAxis, TileShape.insertAxisIndex,
              TileShape.dropInsertedIndex, NumericDType.sub, hBN]
        congr
      · have hInactive :
            ¬ (s.pids 0 * BLOCK_M + idx.1.val < M ∧ idx.2.1.val < N) := by
          simpa [active, mIndex, nIndex] using hActive
        simp [hInactive]
        intro h
        exact False.elim (hActive h)
    · simp [exec, log_softmax_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map,
            Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
            Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceMaxKeep,
            Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.dropInsertedIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            NumericDType.div, ComparableDType.lt,
            ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, hBN] at hExec
  · exact False.elim (hBM (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.1.isLt))

/-- Compute-facing correctness for `log_softmax_kernel`. -/
theorem log_softmax_kernel_compute_correct
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] => outOffset s N K BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := log_softmax_kernel output_ptr input_ptr M N K BLOCK_M BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_N] => active s M N BLOCK_M idx)
        (fun idx => (output_ptr, outOffset s N K BLOCK_M idx)))
      (expected := fun idx => logSoftmaxSpec s input_ptr M N K BLOCK_M BLOCK_N idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [log_softmax_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := log_softmax_kernel_correct output_ptr input_ptr M N K BLOCK_M BLOCK_N
    s s' hOutInj hExec idx
  simpa [hActive] using h

/-- Faithful transcription of `log_softmax.py`'s
`log_softmax_backward_kernel`. -/
def log_softmax_backward_kernel
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offsets = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  out_ptrs = out_ptr + offsets
  out = tl.load(out_ptrs, mask=mask).to(tl.float32)
  out_grad_ptrs = out_grad_ptr + offsets
  out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
  scale = tl.sum(out_grad, 1)
  in_grad = out_grad - tl.exp((out).to(tl.float32)) * scale[:, None]
  in_grad_ptrs = in_grad_ptr + offsets
  tl.store(in_grad_ptrs, in_grad, mask=mask)
}

noncomputable def logSoftmaxBackwardOutTile
    (s : BlockState) (out_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      if active s M N BLOCK_M idx then
        some (s.readMem out_ptr (outOffset s N K BLOCK_M idx))
      else some (s.undef out_ptr (outOffset s N K BLOCK_M idx)) }

noncomputable def logSoftmaxBackwardGradTile
    (s : BlockState) (out_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      if active s M N BLOCK_M idx then
        some (s.readMem out_grad_ptr (outOffset s N K BLOCK_M idx))
      else some (s.undef out_grad_ptr (outOffset s N K BLOCK_M idx)) }

noncomputable def logSoftmaxBackwardSpec
    (s : BlockState) (out_ptr out_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  let out := logSoftmaxBackwardOutTile s out_ptr M N K BLOCK_M BLOCK_N
  let outGrad := logSoftmaxBackwardGradTile s out_grad_ptr M N K BLOCK_M BLOCK_N
  let scale := Tile.reduceSum (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.true outGrad
  let rowBroadcast : Broadcast [BLOCK_M, BLOCK_N] [BLOCK_M, 1] [BLOCK_M, BLOCK_N] :=
    Broadcast.consSame (Broadcast.consR Broadcast.nil)
  let sameBroadcast : Broadcast [BLOCK_M, BLOCK_N] [BLOCK_M, BLOCK_N] [BLOCK_M, BLOCK_N] :=
    Broadcast.consSame (Broadcast.consSame Broadcast.nil)
  WithBot.unbotD 0
    ((Tile.bop (NumericDType.sub .real) sameBroadcast
      outGrad
      (Tile.bop (NumericDType.mul .real) rowBroadcast
        (Tile.uop WithBot.realExp out) scale)).data idx)

/-- Algorithm-layer cellwise correctness for `log_softmax_backward_kernel`. -/
theorem log_softmax_backward_kernel_correct
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] => outOffset s N K BLOCK_M idx))
    (hExec : exec (log_softmax_backward_kernel out_ptr out_grad_ptr in_grad_ptr
        M N K BLOCK_M BLOCK_N) s = some s') :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      s'.readMem in_grad_ptr (outOffset s N K BLOCK_M idx) =
        if active s M N BLOCK_M idx then
          logSoftmaxBackwardSpec s out_ptr out_grad_ptr M N K BLOCK_M BLOCK_N idx
        else s.readMem in_grad_ptr (outOffset s N K BLOCK_M idx) := by
  intro idx
  by_cases hBM : 0 < BLOCK_M
  · by_cases hBN : 0 < BLOCK_N
    · simp [exec, log_softmax_backward_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map,
            Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
            Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
            NumericDType.sub, ComparableDType.lt, FloatDType.cast,
            FloatDType.ofWithBot, FloatDType.toWithBot,
            ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, hBN] at hExec
      rw [← hExec]
      have hOffsetInj : Function.Injective
          (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            (s.pids 0 * BLOCK_M + idx.1.val) * N * K + idx.2.1.val * K + s.pids 1) := by
        intro a b h
        apply hOutInj
        simpa [outOffset, mIndex, nIndex] using h
      simp only [outOffset, mIndex, nIndex]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
      by_cases hActive : active s M N BLOCK_M idx
      · rcases hActive with ⟨hM, hN⟩
        simp [active, mIndex, nIndex, hM, hN, logSoftmaxBackwardSpec,
              logSoftmaxBackwardOutTile, logSoftmaxBackwardGradTile,
              Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
              TileShape.axisDim, TileShape.eraseAxis,
              TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
              FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
              NumericDType.sub, NumericDType.mul, hBN]
        congr
      · have hInactive :
            ¬ (s.pids 0 * BLOCK_M + idx.1.val < M ∧ idx.2.1.val < N) := by
          simpa [active, mIndex, nIndex] using hActive
        simp [hInactive]
        intro h
        exact False.elim (hActive h)
    · exact False.elim (hBN (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.2.1.isLt))
  · exact False.elim (hBM (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.1.isLt))

/-- Compute-facing correctness for `log_softmax_backward_kernel`. -/
theorem log_softmax_backward_kernel_compute_correct
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] => outOffset s N K BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := log_softmax_backward_kernel out_ptr out_grad_ptr in_grad_ptr
        M N K BLOCK_M BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_N] => active s M N BLOCK_M idx)
        (fun idx => (in_grad_ptr, outOffset s N K BLOCK_M idx)))
      (expected := fun idx =>
        logSoftmaxBackwardSpec s out_ptr out_grad_ptr M N K BLOCK_M BLOCK_N idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [log_softmax_backward_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := log_softmax_backward_kernel_correct out_ptr out_grad_ptr in_grad_ptr
    M N K BLOCK_M BLOCK_N s s' hOutInj hExec idx
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.LogSoftmax
