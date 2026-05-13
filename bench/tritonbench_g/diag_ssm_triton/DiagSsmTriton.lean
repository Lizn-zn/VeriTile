import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Faithful transcription of `diag_ssm_triton.py`'s
`diag_ssm_forward_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- Python `length`, `batch_size`, `dim` → Lean `Nat` parameters.

Known proof blocker: see `bench/tritonbench_g/proof_blockers.md`. The current
file still lacks a real `ComputeCorrect.Realizes` theorem for the recurrence
across `tl.for t in length`. -/
def diag_ssm_forward_kernel
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  col_idx = tl.program_id(0) * $(BLOCK_SIZE)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size * dim)
  s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
  Lambda = tl.load(lambda_ptr + col_offsets % $(dim), mask=mask, other=0)
  tl.for t in $(length) {
    offsets = t * $(batch_size * dim) + col_offsets
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    s = s * Lambda + x
    tl.store(y_ptr + offsets, s, mask=mask)
  }
}

/-- Faithful transcription of `diag_ssm_triton.py`'s
`diag_ssm_backward_kernel` for the real-valued path.

The source rewrites reverse traversal as `for i in range(length); t = length -
1 - i`, which is preserved here. -/
def diag_ssm_backward_kernel
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  col_idx = tl.program_id(0) * $(BLOCK_SIZE)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size * dim)
  Lambda = tl.load(lambda_ptr + col_offsets % $(dim), mask=mask, other=0)
  grad_s = tl.zeros_like(Lambda)
  grad_Lambda = tl.zeros_like(Lambda)
  for i in range(0, $(length), $(1)) {
    t = $(length) - $(1) - i
    offsets = t * $(batch_size * dim) + col_offsets
    grad_y = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
    if t > 0 {
      s = tl.load(y_ptr + (offsets - $(batch_size * dim)), mask=mask, other=0)
    } else {
      s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
    }
    grad_s = grad_y + grad_s
    grad_x = grad_s
    grad_Lambda += grad_s * s
    grad_s = grad_s * Lambda
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
  }
  tl.store(grad_s_ptr + col_offsets, grad_s, mask=mask)
  tl.store(grad_lambda_ptr + col_offsets, grad_Lambda, mask=mask)
}

/-- Faithful transcription of `diag_ssm_triton.py`'s
`diag_ssm_forward_kernel_complex`.

The Python tuple assignment `s_real, s_imag = new_s_real, new_s_imag` is
preserved with the DSL multiple-assignment surface. -/
def diag_ssm_forward_kernel_complex
    (s_ptr x_ptr y_ptr lambda_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  col_idx = tl.program_id(0) * $(BLOCK_SIZE)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size * dim)
  s_real = tl.load(s_ptr + col_offsets * $(2), mask=mask, other=0)
  s_imag = tl.load(s_ptr + col_offsets * $(2) + $(1), mask=mask, other=0)
  lambda_real = tl.load(lambda_ptr + (col_offsets % $(dim)) * $(2),
    mask=mask, other=0)
  lambda_imag = tl.load(lambda_ptr + (col_offsets % $(dim)) * $(2) + $(1),
    mask=mask, other=0)
  for t in range(0, $(length), $(1)) {
    offsets = (t * $(batch_size * dim) + col_offsets) * $(2)
    x_real = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_imag = tl.load(x_ptr + offsets + $(1), mask=mask, other=0)
    new_s_real = s_real * lambda_real - s_imag * lambda_imag + x_real
    new_s_imag = s_real * lambda_imag + s_imag * lambda_real + x_imag
    tl.store(y_ptr + offsets, new_s_real, mask=mask)
    tl.store(y_ptr + offsets + $(1), new_s_imag, mask=mask)
    s_real, s_imag = new_s_real, new_s_imag
  }
}

/-- Faithful transcription of `diag_ssm_triton.py`'s
`diag_ssm_backward_kernel_complex`. -/
def diag_ssm_backward_kernel_complex
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  col_idx = tl.program_id(0) * $(BLOCK_SIZE)
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size * dim)
  lambda_real = tl.load(lambda_ptr + (col_offsets % $(dim)) * $(2),
    mask=mask, other=0)
  lambda_imag = tl.load(lambda_ptr + (col_offsets % $(dim)) * $(2) + $(1),
    mask=mask, other=0)
  grad_s_real = tl.zeros_like(lambda_real)
  grad_s_imag = tl.zeros_like(lambda_imag)
  grad_lambda_real = tl.zeros_like(lambda_real)
  grad_lambda_imag = tl.zeros_like(lambda_imag)
  for i in range(0, $(length), $(1)) {
    t = $(length) - $(1) - i
    offsets = (t * $(batch_size * dim) + col_offsets) * $(2)
    grad_y_real = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
    grad_y_imag = -tl.load(grad_y_ptr + offsets + $(1), mask=mask, other=0)
    if t > 0 {
      s_real = tl.load(y_ptr + (offsets - $(2 * batch_size * dim)),
        mask=mask, other=0)
      s_imag = tl.load(y_ptr + (offsets - $(2 * batch_size * dim) + $(1)),
        mask=mask, other=0)
    } else {
      s_real = tl.load(s_ptr + $(2) * col_offsets, mask=mask, other=0)
      s_imag = tl.load(s_ptr + $(2) * col_offsets + $(1), mask=mask, other=0)
    }
    grad_s_real = grad_y_real + grad_s_real
    grad_s_imag = grad_y_imag + grad_s_imag
    grad_x_real = grad_s_real
    grad_x_imag = grad_s_imag
    grad_lambda_real += grad_s_real * s_real - grad_s_imag * s_imag
    grad_lambda_imag += grad_s_real * s_imag + grad_s_imag * s_real
    grad_s_real = grad_x_real * lambda_real - grad_x_imag * lambda_imag
    grad_s_imag = grad_x_real * lambda_imag + grad_x_imag * lambda_real
    tl.store(grad_x_ptr + offsets, grad_x_real, mask=mask)
    tl.store(grad_x_ptr + offsets + $(1), -grad_x_imag, mask=mask)
  }
  tl.store(grad_s_ptr + col_offsets * $(2), grad_s_real, mask=mask)
  tl.store(grad_s_ptr + col_offsets * $(2) + $(1), -grad_s_imag, mask=mask)
  tl.store(grad_lambda_ptr + col_offsets * $(2), grad_lambda_real, mask=mask)
  tl.store(grad_lambda_ptr + col_offsets * $(2) + $(1), -grad_lambda_imag,
    mask=mask)
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

def timeOffset
    (st : BlockState) (batch_size dim BLOCK_SIZE t : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  t * (batch_size * dim) + colOffset st BLOCK_SIZE i

noncomputable def diagSsmStateAfter
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat → ℝ
  | 0 => st.readMem s_ptr (colOffset st BLOCK_SIZE i)
  | t + 1 =>
      diagSsmStateAfter st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE i t *
          st.readMem lambda_ptr (IntegralDType.nat.mod (colOffset st BLOCK_SIZE i) dim) +
        st.readMem x_ptr (timeOffset st batch_size dim BLOCK_SIZE t i)

noncomputable def diagSsmForwardSpec
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat) (t : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  diagSsmStateAfter st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE i (t + 1)

noncomputable def diagSsmStateTile
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE t : Nat) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      some (diagSsmStateAfter st s_ptr x_ptr lambda_ptr batch_size dim
        BLOCK_SIZE idx.1 t) }

noncomputable def diagSsmMaskedStateTile
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE t : Nat) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if active st batch_size dim BLOCK_SIZE idx.1 then
        some (diagSsmStateAfter st s_ptr x_ptr lambda_ptr batch_size dim
          BLOCK_SIZE idx.1 t)
      else
        some (0.0 : ℝ) }

theorem diagSsmMaskedStateTile_active
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE t : Nat) (i : Fin BLOCK_SIZE)
    (hi : active st batch_size dim BLOCK_SIZE i) :
    (diagSsmMaskedStateTile st s_ptr x_ptr lambda_ptr batch_size dim
      BLOCK_SIZE t).data (i, PUnit.unit) =
      some (diagSsmStateAfter st s_ptr x_ptr lambda_ptr batch_size dim
        BLOCK_SIZE i t) := by
  simp [diagSsmMaskedStateTile, hi]

@[simp] theorem diagSsmStateAfter_zero
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) :
    diagSsmStateAfter st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE i 0 =
      st.readMem s_ptr (colOffset st BLOCK_SIZE i) := by
  rfl

@[simp] theorem diagSsmStateAfter_succ
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE t : Nat) (i : Fin BLOCK_SIZE) :
    diagSsmStateAfter st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE i (t + 1) =
      diagSsmStateAfter st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE i t *
          st.readMem lambda_ptr (IntegralDType.nat.mod (colOffset st BLOCK_SIZE i) dim) +
        st.readMem x_ptr (timeOffset st batch_size dim BLOCK_SIZE t i) := by
  rfl

theorem diagSsmStateTile_succ
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE t : Nat) :
    diagSsmStateTile st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE (t + 1) =
      { data := fun idx =>
          some
            (diagSsmStateAfter st s_ptr x_ptr lambda_ptr batch_size dim
                BLOCK_SIZE idx.1 t *
              st.readMem lambda_ptr
                (IntegralDType.nat.mod (colOffset st BLOCK_SIZE idx.1) dim) +
              st.readMem x_ptr
                (timeOffset st batch_size dim BLOCK_SIZE t idx.1)) } := by
  ext idx
  simp [diagSsmStateTile]

def diagSsmForwardOutOffset
    {length : Nat}
    (st : BlockState) (batch_size dim BLOCK_SIZE : Nat)
    (idx : TileIndex [length, BLOCK_SIZE]) : Nat :=
  timeOffset st batch_size dim BLOCK_SIZE idx.1.val idx.2.1

def diagSsmForwardActive
    {length : Nat}
    (st : BlockState) (batch_size dim BLOCK_SIZE : Nat)
    (idx : TileIndex [length, BLOCK_SIZE]) : Prop :=
  active st batch_size dim BLOCK_SIZE idx.2.1

instance diagSsmForwardActiveDecidable
    {length : Nat}
    (st : BlockState) (batch_size dim BLOCK_SIZE : Nat)
    (idx : TileIndex [length, BLOCK_SIZE]) :
    Decidable (diagSsmForwardActive st batch_size dim BLOCK_SIZE idx) := by
  unfold diagSsmForwardActive
  infer_instance

noncomputable def diagSsmForwardSpecAt
    {length : Nat}
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat)
    (idx : TileIndex [length, BLOCK_SIZE]) : ℝ :=
  diagSsmForwardSpec st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE
    idx.1.val idx.2.1

theorem diagSsmForwardSpecAt_eq_stateTile
    {length : Nat}
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat)
    (idx : TileIndex [length, BLOCK_SIZE]) :
    diagSsmForwardSpecAt st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE idx =
      WithBot.unbotD 0
        ((diagSsmStateTile st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE
          (idx.1.val + 1)).data (idx.2.1, PUnit.unit)) := by
  simp [diagSsmForwardSpecAt, diagSsmForwardSpec, diagSsmStateTile]

def diagSsmForwardLoopInvariant
    (st0 : BlockState) (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (t : Nat) (st : BlockState) : Prop :=
  st.regs .real [BLOCK_SIZE] "s" =
      some (diagSsmMaskedStateTile st0 s_ptr x_ptr lambda_ptr batch_size dim
        BLOCK_SIZE t) ∧
    ∀ idx : TileIndex [length, BLOCK_SIZE],
      idx.1.val < t →
        diagSsmForwardActive st0 batch_size dim BLOCK_SIZE idx →
          st.readMem y_ptr
              (diagSsmForwardOutOffset st0 batch_size dim BLOCK_SIZE idx) =
            diagSsmForwardSpecAt st0 s_ptr x_ptr lambda_ptr batch_size dim
              BLOCK_SIZE idx

theorem diagSsmForwardLoopInvariant_zero
    (st0 st : BlockState) (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hReg :
      st.regs .real [BLOCK_SIZE] "s" =
        some (diagSsmMaskedStateTile st0 s_ptr x_ptr lambda_ptr batch_size dim
          BLOCK_SIZE 0)) :
    diagSsmForwardLoopInvariant st0 s_ptr x_ptr lambda_ptr y_ptr length
      batch_size dim BLOCK_SIZE 0 st := by
  constructor
  · exact hReg
  · intro idx hlt _
    omega

theorem diagSsmForwardLoopInvariant_step_of_time_write
    (st0 st st' : BlockState) (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE t : Nat)
    (hPrev :
      diagSsmForwardLoopInvariant st0 s_ptr x_ptr lambda_ptr y_ptr length
        batch_size dim BLOCK_SIZE t st)
    (hReg :
      st'.regs .real [BLOCK_SIZE] "s" =
        some (diagSsmMaskedStateTile st0 s_ptr x_ptr lambda_ptr batch_size dim
          BLOCK_SIZE (t + 1)))
    (hPreserve :
      ∀ idx : TileIndex [length, BLOCK_SIZE],
        idx.1.val < t →
          diagSsmForwardActive st0 batch_size dim BLOCK_SIZE idx →
            st'.readMem y_ptr
                (diagSsmForwardOutOffset st0 batch_size dim BLOCK_SIZE idx) =
              st.readMem y_ptr
                (diagSsmForwardOutOffset st0 batch_size dim BLOCK_SIZE idx))
    (hWrite :
      ∀ i : Fin BLOCK_SIZE,
        active st0 batch_size dim BLOCK_SIZE i →
          st'.readMem y_ptr (timeOffset st0 batch_size dim BLOCK_SIZE t i) =
            diagSsmForwardSpec st0 s_ptr x_ptr lambda_ptr batch_size dim
              BLOCK_SIZE t i) :
    diagSsmForwardLoopInvariant st0 s_ptr x_ptr lambda_ptr y_ptr length
      batch_size dim BLOCK_SIZE (t + 1) st' := by
  constructor
  · exact hReg
  · intro idx hlt hactive
    by_cases hOld : idx.1.val < t
    · rw [hPreserve idx hOld hactive]
      exact hPrev.2 idx hOld hactive
    · have ht : idx.1.val = t := by omega
      rw [diagSsmForwardOutOffset, diagSsmForwardSpecAt, ht]
      exact hWrite idx.2.1 hactive

def diag_ssm_forward_kernel_correct_target
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (s : BlockState) : Prop :=
  ComputeCorrect.Realizes
    (kernel := diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr
      length batch_size dim BLOCK_SIZE)
    (initialState := s)
    (write := ComputeCorrect.WriteMap.writeIf
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardActive s batch_size dim BLOCK_SIZE idx)
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        (y_ptr, diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx)))
    (expected := fun idx : TileIndex [length, BLOCK_SIZE] =>
      diagSsmForwardSpecAt s s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE idx)

def diag_ssm_forward_kernel_alg_post
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (s s' : BlockState) : Prop :=
  ∀ idx : TileIndex [length, BLOCK_SIZE],
    diagSsmForwardActive s batch_size dim BLOCK_SIZE idx →
    s'.readMem y_ptr
        (diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx) =
      diagSsmForwardSpecAt s s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE idx

theorem diagSsmForwardLoopInvariant_to_alg_post
    (st0 st : BlockState) (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hInv :
      diagSsmForwardLoopInvariant st0 s_ptr x_ptr lambda_ptr y_ptr length
        batch_size dim BLOCK_SIZE length st) :
    diag_ssm_forward_kernel_alg_post s_ptr x_ptr lambda_ptr y_ptr length
      batch_size dim BLOCK_SIZE st0 st := by
  intro idx hidx
  exact hInv.2 idx idx.1.isLt hidx

theorem diag_ssm_forward_kernel_compute_correct_of_algorithm
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (s : BlockState)
    (hAlg :
      ∀ s',
        exec (diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr
          length batch_size dim BLOCK_SIZE) s = some s' →
        diag_ssm_forward_kernel_alg_post s_ptr x_ptr lambda_ptr y_ptr
          length batch_size dim BLOCK_SIZE s s') :
    diag_ssm_forward_kernel_correct_target s_ptr x_ptr lambda_ptr y_ptr
      length batch_size dim BLOCK_SIZE s := by
  unfold diag_ssm_forward_kernel_correct_target
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx hidx
  exact hAlg s' hExec idx hidx

/-- Algorithm-layer correctness for the forward SSM kernel. -/
theorem diag_ssm_forward_kernel_correct
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx))
    (hAlg :
      ∀ s',
        exec (diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr
          length batch_size dim BLOCK_SIZE) s = some s' →
        diag_ssm_forward_kernel_alg_post s_ptr x_ptr lambda_ptr y_ptr
          length batch_size dim BLOCK_SIZE s s') :
    diag_ssm_forward_kernel_correct_target s_ptr x_ptr lambda_ptr y_ptr
      length batch_size dim BLOCK_SIZE s :=
  diag_ssm_forward_kernel_compute_correct_of_algorithm s_ptr x_ptr lambda_ptr
    y_ptr length batch_size dim BLOCK_SIZE s hAlg

/-- Compute-facing correctness for the forward SSM kernel. -/
theorem diag_ssm_forward_kernel_compute_correct
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx))
    (hAlg :
      ∀ s',
        exec (diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr
          length batch_size dim BLOCK_SIZE) s = some s' →
        diag_ssm_forward_kernel_alg_post s_ptr x_ptr lambda_ptr y_ptr
          length batch_size dim BLOCK_SIZE s s') :
    diag_ssm_forward_kernel_correct_target s_ptr x_ptr lambda_ptr y_ptr
      length batch_size dim BLOCK_SIZE s :=
  diag_ssm_forward_kernel_compute_correct_of_algorithm s_ptr x_ptr lambda_ptr
    y_ptr length batch_size dim BLOCK_SIZE s hAlg

end VeriTile.Bench.TritonBenchG.DiagSsmTriton
