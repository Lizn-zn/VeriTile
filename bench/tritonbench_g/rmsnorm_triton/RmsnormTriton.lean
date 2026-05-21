import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.RmsnormTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `rmsnorm_triton.py`'s `rmsnorm_triton`.

Allowed mechanical Lean-syntax-only changes:
- Python `N_SIZE: tl.constexpr` / `eps: tl.constexpr` / `BLOCK_N_SIZE: tl.constexpr`
  -> Lean `Nat` / `ℝ` parameters. -/
def rmsnorm_triton
    (x_ptr rms_w_ptr output_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k : Nat)
    (N_SIZE BLOCK_N_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(0)
  pid_m = tl.program_id(1)
  offs_m = pid_batch * $(stride_x_batch) + pid_m * $(stride_x_m)
  block_N = tl.arange(0, $(BLOCK_N_SIZE))
  var = tl.zeros([$(BLOCK_N_SIZE)], tl.float32)
  for block_n_start_idx in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offs_n = block_n_start_idx + block_N
    x_ptr_mask = offs_n < $(N_SIZE)
    x = tl.load(x_ptr + offs_m + offs_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    var += tl.extra.cuda.libdevice.pow((x).to(tl.float32), 2)
  }
  var = tl.sum(var, axis=0) / $(N_SIZE)
  rstd = tl.math.rsqrt(var + $(eps))
  for block_n_start_idx in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offs_n = block_n_start_idx + block_N
    x_ptr_mask = offs_n < $(N_SIZE)
    rms_w = tl.load(rms_w_ptr + offs_n * $(stride_rms_w), mask=x_ptr_mask)
    x = tl.load(x_ptr + offs_m + offs_n * $(stride_x_k), mask=x_ptr_mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    out = x_hat * rms_w
    out_off = pid_batch * $(stride_out_batch) + pid_m * $(stride_out_m) +
      offs_n * $(stride_out_k)
    tl.store(output_ptr + out_off, out, mask=x_ptr_mask)
  }
}

def xOffset
    (s : BlockState) (stride_x_batch stride_x_m stride_x_k : Nat)
    (i : Fin BLOCK_N_SIZE) : Nat :=
  s.pids 0 * stride_x_batch + s.pids 1 * stride_x_m + i.val * stride_x_k

def wOffset (stride_rms_w : Nat) (i : Fin BLOCK_N_SIZE) : Nat :=
  i.val * stride_rms_w

def outOffset
    (s : BlockState) (stride_out_batch stride_out_m stride_out_k : Nat)
    (i : Fin BLOCK_N_SIZE) : Nat :=
  s.pids 0 * stride_out_batch + s.pids 1 * stride_out_m + i.val * stride_out_k

noncomputable def rmsInputTile
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat) :
    Tile .real [BLOCK_N_SIZE] :=
  { data := fun idx =>
      if idx.1.val < N_SIZE then
        some (s.readMem x_ptr
          (xOffset s stride_x_batch stride_x_m stride_x_k idx.1))
      else some (0.0 : ℝ) }

noncomputable def rmsVarCarrier
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat) :
    WithBot ℝ :=
  Option.map₂ (fun a n => a / n)
    ((Tile.reduceSum (shape := [BLOCK_N_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s x_ptr stride_x_batch stride_x_m stride_x_k
          N_SIZE BLOCK_N_SIZE)
        (rmsInputTile s x_ptr stride_x_batch stride_x_m stride_x_k
          N_SIZE BLOCK_N_SIZE))).data PUnit.unit)
    ((Tile.scalar (dtype := .real) (some (N_SIZE : ℝ) : WithBot ℝ)).data PUnit.unit)

noncomputable def rmsInvCarrier
    (s : BlockState) (x_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  WithBot.realRsqrt
    (Option.map (fun a => a + eps)
      (rmsVarCarrier s x_ptr stride_x_batch stride_x_m stride_x_k
        N_SIZE BLOCK_N_SIZE))

noncomputable def rmsnormSpec
    (s : BlockState) (x_ptr rms_w_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (i : Fin BLOCK_N_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun x inv => x * inv)
        (some (s.readMem x_ptr
          (xOffset s stride_x_batch stride_x_m stride_x_k i)))
        (rmsInvCarrier s x_ptr stride_x_batch stride_x_m stride_x_k
          N_SIZE BLOCK_N_SIZE eps))
      (some (s.readMem rms_w_ptr (wOffset stride_rms_w i))))

/-- Algorithm-layer correctness under the wrapper's one-block launch shape
(`N_SIZE <= BLOCK_N_SIZE`). -/
theorem rmsnorm_triton_correct
    (x_ptr rms_w_ptr output_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hNpos : 0 < N_SIZE) (hNle : N_SIZE ≤ BLOCK_N_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N_SIZE =>
        outOffset s stride_out_batch stride_out_m stride_out_k i))
    (hExec : exec (rmsnorm_triton x_ptr rms_w_ptr output_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps) s =
        some s') :
    ∀ i : Fin BLOCK_N_SIZE,
      s'.readMem output_ptr
          (outOffset s stride_out_batch stride_out_m stride_out_k i) =
        if i.val < N_SIZE then
          rmsnormSpec s x_ptr rms_w_ptr stride_x_batch stride_x_m stride_x_k
            stride_rms_w N_SIZE BLOCK_N_SIZE eps i
        else s.readMem output_ptr
          (outOffset s stride_out_batch stride_out_m stride_out_k i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_N_SIZE] =>
        s.pids 0 * stride_out_batch + s.pids 1 * stride_out_m +
          idx.1.val * stride_out_k) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_N_SIZE
  · have hStep : BLOCK_N_SIZE ≠ 0 := Nat.ne_of_gt hB
    simp [exec, rmsnorm_triton, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          stepForRangeAux.step_lt, stepForRangeAux.step_ge,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, hNpos, hNle,
          Nat.not_lt.mpr hNle, hStep] at hExec
    subst s'
    simp only [outOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < N_SIZE
    · simp only [hi, ↓reduceIte]
      simp [hi, rmsnormSpec, rmsInvCarrier, rmsVarCarrier, rmsInputTile,
            xOffset, wOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realRsqrt, NumericDType.mul, FloatDType.cast]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness under the wrapper's one-block launch shape
(`N_SIZE <= BLOCK_N_SIZE`). -/
theorem rmsnorm_triton_compute_correct
    (x_ptr rms_w_ptr output_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hNpos : 0 < N_SIZE) (hNle : N_SIZE ≤ BLOCK_N_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N_SIZE =>
        outOffset s stride_out_batch stride_out_m stride_out_k i)) :
    ComputeCorrect.Realizes
      (kernel := rmsnorm_triton x_ptr rms_w_ptr output_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N_SIZE => i.val < N_SIZE)
        (fun i => (output_ptr,
          outOffset s stride_out_batch stride_out_m stride_out_k i)))
      (expected := fun i =>
        rmsnormSpec s x_ptr rms_w_ptr stride_x_batch stride_x_m stride_x_k
          stride_rms_w N_SIZE BLOCK_N_SIZE eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rmsnorm_triton, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rmsnorm_triton_correct x_ptr rms_w_ptr output_ptr
    stride_x_batch stride_x_m stride_x_k stride_rms_w
    stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps
    s s' hNpos hNle hOutInj hExec i
  simpa [hActive] using h
end VeriTile.Bench.TritonBenchG.RmsnormTriton
