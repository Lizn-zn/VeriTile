import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton

open VeriTile.Triton

/-- Faithful transcription of `rms_norm_triton.py`'s `rms_norm_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- Python pointer mutation `Y += ...` / `X += ...` → explicit base pointer
  registers.
- Python `to(Y.dtype.element_ty)` is omitted at the algorithm layer; VeriTile's
  AlgorithmCorrect path models floating arithmetic over `ℝ`, with dtype
  roundoff deferred to ComputeCorrect gap testing. -/
def rms_norm_kernel
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  Y_base = Y + pid * $(y_stride_r)
  X_base = X + pid * $(x_stride_r)
  cols = tl.arange(0, $(BLOCK_SIZE))
  mask = cols < $(N)
  x = tl.load(X_base + cols * $(x_stride_c), mask=mask, other=0.0).to(tl.float32)
  var = tl.sum(x * x, axis=0) / $(N)
  rrms = 1 / tl.sqrt(var + $(eps))
  w = tl.load(W + cols, mask=mask, other=0.0)
  y = x * rrms * w
  tl.store(Y_base + cols * $(y_stride_c), y, mask=mask)
}

noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (x_stride_r x_stride_c N BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pids 0 * x_stride_r + idx.1.val * x_stride_c
      if idx.1.val < N then some (s.readMem X off) else some (0.0 : ℝ) }

noncomputable def rmsSumCarrier
    (s : BlockState) (X : RegionName) (x_stride_r x_stride_c N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (rmsInputTile s X x_stride_r x_stride_c N BLOCK_SIZE)
      (rmsInputTile s X x_stride_r x_stride_c N BLOCK_SIZE))).data PUnit.unit

noncomputable def rmsVarCarrier
    (s : BlockState) (X : RegionName) (x_stride_r x_stride_c N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map (fun x => x / (N : ℝ))
    (rmsSumCarrier s X x_stride_r x_stride_c N BLOCK_SIZE)

noncomputable def rmsRrmsCarrier
    (s : BlockState) (X : RegionName) (x_stride_r x_stride_c N BLOCK_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt (Option.map ((fun a => a + eps) ∘ fun a => a / (N : ℝ))
      (rmsSumCarrier s X x_stride_r x_stride_c N BLOCK_SIZE)))

noncomputable def rmsNormSpec
    (s : BlockState) (X W : RegionName)
    (x_stride_r x_stride_c N BLOCK_SIZE : Nat) (eps : ℝ) (idx : Fin BLOCK_SIZE) :
    ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun x w => x * w)
      (Option.map₂ (fun x rrms => x * rrms)
        (some (s.readMem X (s.pids 0 * x_stride_r + idx.val * x_stride_c)))
        (rmsRrmsCarrier s X x_stride_r x_stride_c N BLOCK_SIZE eps))
      (some (s.readMem W idx.val)))

def rmsOutOffset (s : BlockState) (y_stride_r y_stride_c : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * y_stride_r + i.val * y_stride_c

/-- Algorithm-layer correctness for `rms_norm_kernel`. -/
theorem rms_norm_kernel_correct
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => rmsOutOffset s y_stride_r y_stride_c i))
    (hExec : exec (rms_norm_kernel Y X W y_stride_r y_stride_c x_stride_r x_stride_c
          N eps BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := rmsOutOffset s y_stride_r y_stride_c i
      s'.readMem Y outAddr =
        if i.val < N then
          rmsNormSpec s X W x_stride_r x_stride_c N BLOCK_SIZE eps i
        else s.readMem Y outAddr := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] =>
        s.pids 0 * y_stride_r + idx.1.val * y_stride_c) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [rmsOutOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, rms_norm_kernel, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot] at hExec
    subst s'
    simp only [rmsOutOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, rmsNormSpec, rmsRrmsCarrier, rmsSumCarrier,
            rmsInputTile, Tile.reduceSum, Tile.reduceSumDrop,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for `rms_norm_kernel`. -/
theorem rms_norm_kernel_compute_correct
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => rmsOutOffset s y_stride_r y_stride_c i)) :
    ComputeCorrect.Realizes
      (kernel := rms_norm_kernel Y X W y_stride_r y_stride_c x_stride_r x_stride_c
        N eps BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, rmsOutOffset s y_stride_r y_stride_c i)))
      (expected := fun i => rmsNormSpec s X W x_stride_r x_stride_c N BLOCK_SIZE eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rms_norm_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rms_norm_kernel_correct Y X W y_stride_r y_stride_c x_stride_r
    x_stride_c N BLOCK_SIZE eps s s' hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.RmsNormTriton
