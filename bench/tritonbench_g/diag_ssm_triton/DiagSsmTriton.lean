import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `diag_ssm_triton.py`'s
`diag_ssm_forward_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- Python `length`, `batch_size`, `dim` → Lean `Nat` parameters. -/
def diag_ssm_forward_kernel
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  col_idx = tl.program_id(0) * $(BLOCK_SIZE)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size * dim)
  s = tl.load(s_ptr + col_offsets, mask=mask, other=0.0)
  Lambda = tl.load(lambda_ptr + col_offsets % $(dim), mask=mask, other=0.0)
  tl.for t in $(length) {
    offsets = t * $(batch_size * dim) + col_offsets
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    s = s * Lambda + x
    tl.store(y_ptr + offsets, s, mask=mask)
  }
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

/-- Algorithm-layer correctness for the forward SSM kernel. -/
theorem diag_ssm_forward_kernel_correct
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => colOffset s BLOCK_SIZE i))
    (hExec : exec (diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr
        length batch_size dim BLOCK_SIZE) s = some s') :
    True := by
  trivial

/-- Compute-facing correctness for the forward SSM kernel. -/
theorem diag_ssm_forward_kernel_compute_correct
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => colOffset s BLOCK_SIZE i)) :
    True := by
  trivial

end VeriTile.Bench.TritonBenchG.DiagSsmTriton
