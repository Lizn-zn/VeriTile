import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Proof-oriented one-block slice of `rmsnorm_fused_llama.py`'s
`_rms_norm_fwd_fused`.

The upstream kernel loops over `range(0, N, BLOCK_SIZE)` twice. This slice
specializes to the single-block case and keeps the row program id, row stride
pointer arithmetic, masked X/W loads, RMS variance, inverse stddev, and masked
Y store.

Allowed mechanical Lean-syntax-only changes:
- Python `.to(tl.float32)` / `.to(tl.float16)` casts are omitted at the
  algorithm layer.
- Python `N: tl.constexpr` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat`
  parameters. -/
def rms_norm_fwd_fused_llama_one_block
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  Y_base = Y + row * $(stride)
  X_base = X + row * $(stride)
  cols = tl.arange(0, $(BLOCK_SIZE))
  x_for_var = tl.load(X_base + cols, mask=cols < $(N), other=0.0)
  var = tl.sum(x_for_var * x_for_var, axis=0) / tl.toReal($(N))
  rstd = 1 / tl.sqrt(var + $(eps))
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask)
  x = tl.load(X_base + cols, mask=mask, other=0.0)
  x_hat = x * rstd
  y = x_hat * w
  tl.store(Y_base + cols, y, mask=mask)
}

def xOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val

def yOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val

noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (xOffset s stride idx.1))
      else some (0.0 : ℝ) }

noncomputable def rmsVarCarrier
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map₂ (fun a n => a / n)
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s X stride N BLOCK_SIZE)
        (rmsInputTile s X stride N BLOCK_SIZE))).data PUnit.unit)
    ((Tile.scalar N).natToReal.data PUnit.unit)

noncomputable def rmsInvCarrier
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsVarCarrier s X stride N BLOCK_SIZE)))

noncomputable def rmsnormSpec
    (s : BlockState) (X W : RegionName)
    (stride N BLOCK_SIZE : Nat) (eps : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun x inv => x * inv)
        (some (s.readMem X (xOffset s stride i)))
        (rmsInvCarrier s X stride N BLOCK_SIZE eps))
      (some (s.readMem W i.val)))

/-- Algorithm-layer correctness for the one-block Llama RMSNorm slice. -/
theorem rms_norm_fwd_fused_llama_one_block_correct
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i))
    (hExec : exec (rms_norm_fwd_fused_llama_one_block X Y W stride N BLOCK_SIZE eps)
        s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Y (yOffset s stride i) =
        if i.val < N then
          rmsnormSpec s X W stride N BLOCK_SIZE eps i
        else s.readMem Y (yOffset s stride i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * stride + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [yOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, rms_norm_fwd_fused_llama_one_block, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, rmsnormSpec, rmsInvCarrier, rmsVarCarrier, rmsInputTile,
            xOffset, Tile.reduceSum, Tile.reduceSumDrop,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-block Llama RMSNorm slice. -/
theorem rms_norm_fwd_fused_llama_one_block_compute_correct
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)) :
    ComputeCorrect.Realizes
      (kernel := rms_norm_fwd_fused_llama_one_block X Y W stride N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride i)))
      (expected := fun i => rmsnormSpec s X W stride N BLOCK_SIZE eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rms_norm_fwd_fused_llama_one_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rms_norm_fwd_fused_llama_one_block_correct X Y W stride N BLOCK_SIZE
    eps s s' hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama
