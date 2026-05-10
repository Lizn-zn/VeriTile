import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Proof-oriented `length = 1` slice of `diag_ssm_triton.py`'s
`diag_ssm_forward_kernel`.

The full Python kernel iterates over sequence length and updates the recurrent
state. This first slice captures one forward step over a column block:
`y = s * lambda[col % dim] + x`. -/
def diag_ssm_forward_one_step
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  col_idx = tl.program_id(0) * $(BLOCK_SIZE)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size * dim)
  s = tl.load(s_ptr + col_offsets, mask=mask, other=0.0)
  Lambda = tl.load(lambda_ptr + col_offsets % $(dim), mask=mask, other=0.0)
  x = tl.load(x_ptr + col_offsets, mask=mask, other=0.0)
  out = s * Lambda + x
  tl.store(y_ptr + col_offsets, out, mask=mask)
}

def colOffset (st : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  st.pids 0 * BLOCK_SIZE + i.val

def active (st : BlockState) (batch_size dim BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  colOffset st BLOCK_SIZE i < batch_size * dim

instance activeDecidable (st : BlockState) (batch_size dim BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) :
    Decidable (active st batch_size dim BLOCK_SIZE i) := by
  unfold active
  infer_instance

noncomputable def diagSsmSpec
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (dim BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  let off := colOffset st BLOCK_SIZE i
  st.readMem s_ptr off * st.readMem lambda_ptr (IntegralDType.nat.mod off dim) +
    st.readMem x_ptr off

/-- Algorithm-layer correctness for one forward SSM step. -/
theorem diag_ssm_forward_one_step_correct
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => colOffset s BLOCK_SIZE i))
    (hExec : exec (diag_ssm_forward_one_step s_ptr x_ptr lambda_ptr y_ptr
        batch_size dim BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem y_ptr (colOffset s BLOCK_SIZE i) =
        if active s batch_size dim BLOCK_SIZE i then
          diagSsmSpec s s_ptr x_ptr lambda_ptr dim BLOCK_SIZE i
        else
          s.readMem y_ptr (colOffset s BLOCK_SIZE i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * BLOCK_SIZE + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [colOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBS : 0 < BLOCK_SIZE
  · simp [exec, diag_ssm_forward_one_step, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul,
          ComparableDType.lt, hBS] at hExec
    rw [← hExec]
    simp only [colOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
    by_cases hi : colOffset s BLOCK_SIZE i < batch_size * dim
    · have hi' : s.pids 0 * BLOCK_SIZE + i.val < batch_size * dim := by
        simpa [colOffset] using hi
      simp [active, colOffset, diagSsmSpec, hi', Option.map₂, Option.bind,
        Option.map]
    · have hi' : ¬ s.pids 0 * BLOCK_SIZE + i.val < batch_size * dim := by
        simpa [colOffset] using hi
      simp [active, colOffset, hi']
  · exact False.elim (hBS (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for one forward SSM step. -/
theorem diag_ssm_forward_one_step_compute_correct
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => colOffset s BLOCK_SIZE i)) :
    ComputeCorrect.Realizes
      (kernel := diag_ssm_forward_one_step s_ptr x_ptr lambda_ptr y_ptr
        batch_size dim BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s batch_size dim BLOCK_SIZE i)
        (fun i => (y_ptr, colOffset s BLOCK_SIZE i)))
      (expected := fun i => diagSsmSpec s s_ptr x_ptr lambda_ptr dim BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [diag_ssm_forward_one_step]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := diag_ssm_forward_one_step_correct s_ptr x_ptr lambda_ptr y_ptr
    batch_size dim BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.DiagSsmTriton
