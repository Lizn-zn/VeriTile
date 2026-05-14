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

theorem diagSsmMaskedStateTile_succ
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE t : Nat) :
    diagSsmMaskedStateTile st s_ptr x_ptr lambda_ptr batch_size dim
        BLOCK_SIZE (t + 1) =
      { data := fun idx =>
          some
            (if active st batch_size dim BLOCK_SIZE idx.1 then
              diagSsmStateAfter st s_ptr x_ptr lambda_ptr batch_size dim
                  BLOCK_SIZE idx.1 t *
                st.readMem lambda_ptr
                  (IntegralDType.nat.mod (colOffset st BLOCK_SIZE idx.1) dim) +
                st.readMem x_ptr
                  (timeOffset st batch_size dim BLOCK_SIZE t idx.1)
            else
              0.0) } := by
  ext idx
  by_cases hactive : active st batch_size dim BLOCK_SIZE idx.1
  · simp [diagSsmMaskedStateTile, hactive]
  · simp [diagSsmMaskedStateTile, hactive]

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

@[simp] theorem diagSsmForwardOutOffset_currentTime
    {length : Nat}
    (st : BlockState) (batch_size dim BLOCK_SIZE t : Nat)
    (i : Fin BLOCK_SIZE) (ht : t < length) :
    diagSsmForwardOutOffset st batch_size dim BLOCK_SIZE
        ((⟨t, ht⟩ : Fin length), i, PUnit.unit) =
      timeOffset st batch_size dim BLOCK_SIZE t i := by
  rfl

@[simp] theorem diagSsmForwardSpecAt_currentTime
    {length : Nat}
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE t : Nat)
    (i : Fin BLOCK_SIZE) (ht : t < length) :
    diagSsmForwardSpecAt st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE
        ((⟨t, ht⟩ : Fin length), i, PUnit.unit) =
      diagSsmForwardSpec st s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE t i := by
  rfl

@[simp] theorem diagSsmForwardActive_currentTime
    {length : Nat}
    (st : BlockState) (batch_size dim BLOCK_SIZE t : Nat)
    (i : Fin BLOCK_SIZE) (ht : t < length) :
    diagSsmForwardActive st batch_size dim BLOCK_SIZE
        ((⟨t, ht⟩ : Fin length), i, PUnit.unit) =
      active st batch_size dim BLOCK_SIZE i := by
  rfl

theorem diagSsmForwardIndex_ne_currentTime
    {length : Nat} (idx : TileIndex [length, BLOCK_SIZE])
    (t : Nat) (i : Fin BLOCK_SIZE)
    (ht : t < length) (hOld : idx.1.val < t) :
    idx ≠ ((⟨t, ht⟩ : Fin length), i, PUnit.unit) := by
  intro hEq
  have htime : idx.1.val = t := by
    have h := congrArg (fun idx : TileIndex [length, BLOCK_SIZE] => idx.1.val) hEq
    simpa using h
  omega

theorem diagSsmForwardOutOffset_ne_currentTime
    {length : Nat}
    (st : BlockState) (batch_size dim BLOCK_SIZE t : Nat)
    (idx : TileIndex [length, BLOCK_SIZE]) (i : Fin BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset st batch_size dim BLOCK_SIZE idx))
    (ht : t < length) (hOld : idx.1.val < t) :
    diagSsmForwardOutOffset st batch_size dim BLOCK_SIZE idx ≠
      timeOffset st batch_size dim BLOCK_SIZE t i := by
  intro hEq
  have hCurrent :
      diagSsmForwardOutOffset st batch_size dim BLOCK_SIZE
          ((⟨t, ht⟩ : Fin length), i, PUnit.unit) =
        timeOffset st batch_size dim BLOCK_SIZE t i := by
    rfl
  have hEqIdx :
      idx = ((⟨t, ht⟩ : Fin length), i, PUnit.unit) := by
    apply hOutInj
    simpa [hCurrent] using hEq
  exact diagSsmForwardIndex_ne_currentTime idx t i ht hOld hEqIdx

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

theorem diagSsmForwardCurrentTimeScatter_write
    (st0 st : BlockState) (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE t : Nat)
    (i : Fin BLOCK_SIZE)
    (hactive : active st0 batch_size dim BLOCK_SIZE i)
    (hNoCollision :
      ∀ lane : TileIndex [BLOCK_SIZE],
        active st0 batch_size dim BLOCK_SIZE lane.1 →
          timeOffset st0 batch_size dim BLOCK_SIZE t lane.1 =
            timeOffset st0 batch_size dim BLOCK_SIZE t i →
          lane = ((i, PUnit.unit) : TileIndex [BLOCK_SIZE])) :
    ((TileShape.allIndices [BLOCK_SIZE]).foldl
        (fun acc lane =>
          if active st0 batch_size dim BLOCK_SIZE lane.1 then
            acc.writeMem y_ptr
              (timeOffset st0 batch_size dim BLOCK_SIZE t lane.1)
              (diagSsmForwardSpec st0 s_ptr x_ptr lambda_ptr batch_size dim
                BLOCK_SIZE t lane.1)
          else
            acc)
        st).readMem y_ptr (timeOffset st0 batch_size dim BLOCK_SIZE t i) =
      diagSsmForwardSpec st0 s_ptr x_ptr lambda_ptr batch_size dim
        BLOCK_SIZE t i := by
  exact
    BlockState.scatter_readback_prop_masked_nd_of_true
      (region := y_ptr)
      (s := st)
      (offsetFn := fun lane : TileIndex [BLOCK_SIZE] =>
        timeOffset st0 batch_size dim BLOCK_SIZE t lane.1)
      (valueFn := fun lane : TileIndex [BLOCK_SIZE] =>
        diagSsmForwardSpec st0 s_ptr x_ptr lambda_ptr batch_size dim
          BLOCK_SIZE t lane.1)
      (P := fun lane : TileIndex [BLOCK_SIZE] =>
        active st0 batch_size dim BLOCK_SIZE lane.1)
      ((i, PUnit.unit) : TileIndex [BLOCK_SIZE])
      hactive
      (fun lane hlane heq => hNoCollision lane hlane heq)

theorem diagSsmForwardCurrentTimeNoCollision_of_out_injective
    {length : Nat}
    (st0 : BlockState) (batch_size dim BLOCK_SIZE t : Nat)
    (i : Fin BLOCK_SIZE) (ht : t < length)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset st0 batch_size dim BLOCK_SIZE idx)) :
    ∀ lane : TileIndex [BLOCK_SIZE],
      active st0 batch_size dim BLOCK_SIZE lane.1 →
        timeOffset st0 batch_size dim BLOCK_SIZE t lane.1 =
          timeOffset st0 batch_size dim BLOCK_SIZE t i →
        lane = ((i, PUnit.unit) : TileIndex [BLOCK_SIZE]) := by
  intro lane _hactive heq
  have hFull :
      ((⟨t, ht⟩ : Fin length), lane.1, PUnit.unit) =
        ((⟨t, ht⟩ : Fin length), i, PUnit.unit) := by
    apply hOutInj
    simpa [diagSsmForwardOutOffset_currentTime] using heq
  have hLane : lane.1 = i := by
    have h := congrArg
      (fun idx : TileIndex [length, BLOCK_SIZE] => idx.2.1) hFull
    simpa using h
  cases lane with
  | mk laneHead laneTail =>
      cases laneTail
      cases hLane
      rfl

theorem diagSsmForwardCurrentTimeScatter_preserve_old
    {length : Nat}
    (st0 st : BlockState) (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE t : Nat)
    (idx : TileIndex [length, BLOCK_SIZE])
    (ht : t < length) (hOld : idx.1.val < t)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset st0 batch_size dim BLOCK_SIZE idx)) :
    ((TileShape.allIndices [BLOCK_SIZE]).foldl
        (fun acc lane =>
          if active st0 batch_size dim BLOCK_SIZE lane.1 then
            acc.writeMem y_ptr
              (timeOffset st0 batch_size dim BLOCK_SIZE t lane.1)
              (diagSsmForwardSpec st0 s_ptr x_ptr lambda_ptr batch_size dim
                BLOCK_SIZE t lane.1)
          else
            acc)
        st).readMem y_ptr
          (diagSsmForwardOutOffset st0 batch_size dim BLOCK_SIZE idx) =
      st.readMem y_ptr
        (diagSsmForwardOutOffset st0 batch_size dim BLOCK_SIZE idx) := by
  exact
    BlockState.scatter_prop_masked_preserves_other_offset
      y_ptr
      (fun lane : TileIndex [BLOCK_SIZE] =>
        timeOffset st0 batch_size dim BLOCK_SIZE t lane.1)
      (fun lane : TileIndex [BLOCK_SIZE] =>
        diagSsmForwardSpec st0 s_ptr x_ptr lambda_ptr batch_size dim
          BLOCK_SIZE t lane.1)
      (fun lane : TileIndex [BLOCK_SIZE] =>
        active st0 batch_size dim BLOCK_SIZE lane.1)
      (diagSsmForwardOutOffset st0 batch_size dim BLOCK_SIZE idx)
      (fun lane _hactive heq =>
        diagSsmForwardOutOffset_ne_currentTime st0 batch_size dim BLOCK_SIZE t
          idx lane.1 hOutInj ht hOld heq.symm)
      (TileShape.allIndices [BLOCK_SIZE]) st

theorem diagSsmForwardLoopInvariant_step_of_current_time_scatter
    {length : Nat}
    (st0 stPrev stReg : BlockState)
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE t : Nat)
    (ht : t < length)
    (hPrev :
      diagSsmForwardLoopInvariant st0 s_ptr x_ptr lambda_ptr y_ptr length
        batch_size dim BLOCK_SIZE t stPrev)
    (hReg :
      stReg.regs .real [BLOCK_SIZE] "s" =
        some (diagSsmMaskedStateTile st0 s_ptr x_ptr lambda_ptr batch_size dim
          BLOCK_SIZE (t + 1)))
    (hMem :
      ∀ offset,
        stReg.readMem y_ptr offset = stPrev.readMem y_ptr offset)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset st0 batch_size dim BLOCK_SIZE idx)) :
    diagSsmForwardLoopInvariant st0 s_ptr x_ptr lambda_ptr y_ptr length
      batch_size dim BLOCK_SIZE (t + 1)
      ((TileShape.allIndices [BLOCK_SIZE]).foldl
        (fun acc lane =>
          if active st0 batch_size dim BLOCK_SIZE lane.1 then
            acc.writeMem y_ptr
              (timeOffset st0 batch_size dim BLOCK_SIZE t lane.1)
              (diagSsmForwardSpec st0 s_ptr x_ptr lambda_ptr batch_size dim
                BLOCK_SIZE t lane.1)
          else
            acc)
        stReg) := by
  apply diagSsmForwardLoopInvariant_step_of_time_write st0 stPrev
  · exact hPrev
  · simpa using hReg
  · intro idx hOld _hactive
    rw [diagSsmForwardCurrentTimeScatter_preserve_old st0 stReg s_ptr x_ptr
      lambda_ptr y_ptr batch_size dim BLOCK_SIZE t idx ht hOld hOutInj]
    exact hMem (diagSsmForwardOutOffset st0 batch_size dim BLOCK_SIZE idx)
  · intro i hactive
    exact diagSsmForwardCurrentTimeScatter_write st0 stReg s_ptr x_ptr
      lambda_ptr y_ptr batch_size dim BLOCK_SIZE t i hactive
      (diagSsmForwardCurrentTimeNoCollision_of_out_injective st0 batch_size dim
        BLOCK_SIZE t i ht hOutInj)

def diagSsmForwardPreLoop
    (s_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat) : List Stmt :=
  [ .assign .nat [] "col_idx"
      (.mul NumericDType.nat Broadcast.nil (.programId 0) (.constNat BLOCK_SIZE))
  , .assign .nat [BLOCK_SIZE] "col_offsets"
      (.add NumericDType.nat Broadcast.scalarL
        (.ref .nat [] "col_idx") (.arange BLOCK_SIZE))
  , .assign .bool [BLOCK_SIZE] "mask"
      (.lt ComparableDType.nat Broadcast.scalarR
        (.ref .nat [BLOCK_SIZE] "col_offsets")
        (.constNat (batch_size * dim)))
  , .assign .real [BLOCK_SIZE] "s"
      (.load .real
        (.region s_ptr (.ref .nat [BLOCK_SIZE] "col_offsets"))
        (.maskOther
          (.ref .bool [BLOCK_SIZE] "mask")
          ((Op.const 0).broadcast [BLOCK_SIZE])))
  , .assign .real [BLOCK_SIZE] "Lambda"
      (.load .real
        (.region lambda_ptr
          (.mod IntegralDType.nat Broadcast.scalarR
            (.ref .nat [BLOCK_SIZE] "col_offsets") (.constNat dim)))
        (.maskOther
          (.ref .bool [BLOCK_SIZE] "mask")
          ((Op.const 0).broadcast [BLOCK_SIZE])))
  ]

def diagSsmForwardLoopBody
    (x_ptr y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat) : List Stmt :=
  [ .assign .nat [BLOCK_SIZE] "offsets"
      (.add NumericDType.nat Broadcast.scalarL
        (.mul NumericDType.nat Broadcast.nil
          (.ref .nat [] "t") (.constNat (batch_size * dim)))
        (.ref .nat [BLOCK_SIZE] "col_offsets"))
  , .assign .real [BLOCK_SIZE] "x"
      (.load .real
        (.region x_ptr (.ref .nat [BLOCK_SIZE] "offsets"))
        (.maskOther
          (.ref .bool [BLOCK_SIZE] "mask")
          ((Op.const 0).broadcast [BLOCK_SIZE])))
  , .assign .real [BLOCK_SIZE] "s"
      (.add NumericDType.real Broadcast.nil.consSame
        (.mul NumericDType.real Broadcast.nil.consSame
          (.ref .real [BLOCK_SIZE] "s")
          (.ref .real [BLOCK_SIZE] "Lambda"))
        (.ref .real [BLOCK_SIZE] "x"))
  , .store .real [BLOCK_SIZE]
      (.region y_ptr (.ref .nat [BLOCK_SIZE] "offsets"))
      (.ref .real [BLOCK_SIZE] "s")
      (.mask (.ref .bool [BLOCK_SIZE] "mask"))
  ]

def diagSsmForwardProjectedBody
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) : List Stmt :=
  diagSsmForwardPreLoop s_ptr lambda_ptr batch_size dim BLOCK_SIZE ++
    [.forLoop "t" length
      (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim BLOCK_SIZE)]

theorem diag_ssm_forward_kernel_toAlg_body
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    (diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr
      length batch_size dim BLOCK_SIZE).toAlgKernel.body =
      diagSsmForwardProjectedBody s_ptr x_ptr lambda_ptr y_ptr
        length batch_size dim BLOCK_SIZE := by
  rfl

theorem diagSsmForwardPreLoop_step_regs
    (st0 st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat)
    (hStep :
      stepStmts (diagSsmForwardPreLoop s_ptr lambda_ptr batch_size dim
        BLOCK_SIZE) st0 = some st) :
    st.regs .real [BLOCK_SIZE] "s" =
        some (diagSsmMaskedStateTile st0 s_ptr x_ptr lambda_ptr batch_size dim
          BLOCK_SIZE 0) ∧
      st.regs .real [BLOCK_SIZE] "Lambda" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          if active st0 batch_size dim BLOCK_SIZE idx.1 then
            some (st0.readMem lambda_ptr
              (IntegralDType.nat.mod (colOffset st0 BLOCK_SIZE idx.1) dim))
          else
            some 0 } ∧
      st.regs .nat [BLOCK_SIZE] "col_offsets" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset st0 BLOCK_SIZE idx.1 } ∧
      st.regs .bool [BLOCK_SIZE] "mask" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active st0 batch_size dim BLOCK_SIZE idx.1 } := by
  unfold diagSsmForwardPreLoop at hStep
  simp [stepStmts, stepStmt, evalOp, Tile.bop, Tile.cop,
    NumericDType.mul, NumericDType.add, IntegralDType.mod,
    ComparableDType.lt, Option.bind, colOffset, active,
    diagSsmMaskedStateTile, diagSsmStateAfter] at hStep
  subst st
  constructor
  · simp [BlockState.setReg, diagSsmMaskedStateTile, diagSsmStateAfter,
      active, colOffset]
    funext idx
    split <;> norm_num
  · constructor
    · simp [BlockState.setReg, active, colOffset, IntegralDType.mod]
    · constructor
      · simp [BlockState.setReg, colOffset]
      · simp [BlockState.setReg, active, colOffset]
        funext idx
        rfl

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
