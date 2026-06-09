import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Kernel

/-!
# `diag_ssm_triton` — strict per-kernel correctness

`diag_ssm_forward_kernel` is a diagonal state-space model (SSM) scan: each
program owns a `BLOCK_SIZE` column slice and carries the state `s` across a
`0..length` time loop, updating `s = s * Lambda + x_t` and writing `s` to the
output at each step (`Lambda` is the per-dim diagonal transition broadcast by
`col_offsets % dim`). `diag_ssm_backward_kernel` runs the matching
reverse-time gradient scan.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies (forward and backward). The host launch (1-D grid over
column blocks and how the runtime composes per-program writes) is the
*trusted boundary*. Because `tl.program_id(0)` is universally quantified, the
per-program statement covers every program of the grid.

## Proof architecture

```
diag_ssm_forward_kernel_output_summary           ← TOP THEOREM (forward)
  ├─ (toAlgorithm? = Except.ok _)                 surface lowers to the algorithm layer
  └─ diag_ssm_forward_kernel_compute_correct      ComputeCorrect over the time-step stores
       └─ diag_ssm_forward_kernel_correct
            └─ diag_ssm_forward_kernel_compute_correct_of_algorithm
                 └─ diag_ssm_forward_kernel_alg_post_of_exec
                      └─ diagSsmForwardForLoop_context_of_preloop   (loop-invariant fold)

diag_ssm_backward_kernel_compute_correct          ← TOP THEOREM (backward)
  └─ diag_ssm_backward_kernel_compute_correct_of_algorithm
       └─ diag_ssm_backward_kernel_alg_post_of_exec
            └─ diagSsmBackwardForLoop_context_of_preloop            (loop-invariant fold)
```

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; the real-valued kernel is
modeled (the complex variants `diag_ssm_*_kernel_complex` are transcribed but
not the proof target). The recurrent state-carry is verified in full here: a
`LoopInvariant` argument folds the `0..length` time loop, proving that after
execution every active output offset holds the SSM spec value
`diagSsmForwardSpecAt` (and the backward gradient specs) — i.e. the cross-step
state recurrence is proved, not trusted. Dtype casts erase to the identity.
`@triton.autotune` is not modeled. Side conditions: output-store-offset
injectivity (`hOutInj`) and output/input region-distinctness hypotheses
(e.g. `x_ptr ≠ y_ptr`, the gradient-region distinctness chain) are explicit.
-/

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

The proof below connects the recurrence invariant across `tl.for t in length`
to `ComputeCorrect.Realizes` under the stated no-collision/no-alias
hypotheses. -/
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

/-
Complex correctness blocker (#119): the complex forward/backward surfaces above
are faithful transcriptions and pass the port checker, but this file only proves
`ComputeCorrect.Realizes` for the real-valued kernels. The complex kernels encode
complex tensors as paired real memory lanes, including conjugated gradient signs;
their readback theorem should use a paired-lane output map/spec rather than the
single-real-lane recurrence used below.
-/

def colOffset (st : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  st.pids 0 * BLOCK_SIZE + i.val

theorem colOffset_injective (st : BlockState) (BLOCK_SIZE : Nat) :
    Function.Injective (fun i : Fin BLOCK_SIZE => colOffset st BLOCK_SIZE i) := by
  intro i j h
  apply Fin.ext
  exact Nat.add_left_cancel h

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

def reverseTime (length k : Nat) : Nat :=
  length - 1 - k

theorem reverseTime_pos_iff (length k : Nat) (hk : k < length) :
    0 < reverseTime length k ↔ k < length - 1 := by
  unfold reverseTime
  omega

theorem timeOffset_reverse_prev
    (st : BlockState) (batch_size dim BLOCK_SIZE length k : Nat)
    (i : Fin BLOCK_SIZE) (hkprev : k < length - 1) :
    timeOffset st batch_size dim BLOCK_SIZE (reverseTime length k) i -
        batch_size * dim =
      timeOffset st batch_size dim BLOCK_SIZE (reverseTime length k - 1) i := by
  have ht : 0 < reverseTime length k := by
    exact (reverseTime_pos_iff length k (by omega)).2 hkprev
  set t := reverseTime length k with htdef
  obtain ⟨t', ht'⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt ht)
  rw [ht']
  simp [timeOffset, Nat.succ_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

theorem reverseTime_inverse
    (length k : Nat) (hk : k < length) :
    length - 1 - reverseTime length k = k := by
  unfold reverseTime
  omega

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

noncomputable def diagSsmBackwardPrevState
    (st : BlockState) (s_ptr y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length k : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  let t := reverseTime length k
  if t > 0 then
    st.readMem y_ptr (timeOffset st batch_size dim BLOCK_SIZE (t - 1) i)
  else
    st.readMem s_ptr (colOffset st BLOCK_SIZE i)

theorem diagSsmBackwardPrevState_of_prev
    (st : BlockState) (s_ptr y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length k : Nat) (i : Fin BLOCK_SIZE)
    (hk : k < length) (hkprev : k < length - 1) :
    diagSsmBackwardPrevState st s_ptr y_ptr batch_size dim BLOCK_SIZE
        length k i =
      st.readMem y_ptr
        (timeOffset st batch_size dim BLOCK_SIZE (reverseTime length k) i -
          batch_size * dim) := by
  have hpos : 0 < reverseTime length k :=
    (reverseTime_pos_iff length k hk).2 hkprev
  simp [diagSsmBackwardPrevState, hpos]
  rw [timeOffset_reverse_prev st batch_size dim BLOCK_SIZE length k i hkprev]

theorem reverseTime_eq_zero_of_last
    (length k : Nat) (hk : k < length) (hnot : ¬ k < length - 1) :
    reverseTime length k = 0 := by
  unfold reverseTime
  omega

theorem diagSsmBackwardPrevState_of_last
    (st : BlockState) (s_ptr y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length k : Nat) (i : Fin BLOCK_SIZE)
    (hk : k < length) (hnot : ¬ k < length - 1) :
    diagSsmBackwardPrevState st s_ptr y_ptr batch_size dim BLOCK_SIZE
        length k i =
      st.readMem s_ptr (colOffset st BLOCK_SIZE i) := by
  have hzero : reverseTime length k = 0 :=
    reverseTime_eq_zero_of_last length k hk hnot
  simp [diagSsmBackwardPrevState, hzero]

noncomputable def diagSsmBackwardGradSAfter
    (st : BlockState) (lambda_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length : Nat) (i : Fin BLOCK_SIZE) : Nat → ℝ
  | 0 => 0
  | k + 1 =>
      (st.readMem grad_y_ptr
          (timeOffset st batch_size dim BLOCK_SIZE (reverseTime length k) i) +
        diagSsmBackwardGradSAfter st lambda_ptr grad_y_ptr batch_size dim
          BLOCK_SIZE length i k) *
        st.readMem lambda_ptr
          (IntegralDType.nat.mod (colOffset st BLOCK_SIZE i) dim)

noncomputable def diagSsmBackwardGradLambdaAfter
    (st : BlockState) (s_ptr lambda_ptr y_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length : Nat) (i : Fin BLOCK_SIZE) : Nat → ℝ
  | 0 => 0
  | k + 1 =>
      diagSsmBackwardGradLambdaAfter st s_ptr lambda_ptr y_ptr grad_y_ptr
          batch_size dim BLOCK_SIZE length i k +
        (st.readMem grad_y_ptr
            (timeOffset st batch_size dim BLOCK_SIZE (reverseTime length k) i) +
          diagSsmBackwardGradSAfter st lambda_ptr grad_y_ptr batch_size dim
            BLOCK_SIZE length i k) *
          diagSsmBackwardPrevState st s_ptr y_ptr batch_size dim BLOCK_SIZE
            length k i

noncomputable def diagSsmBackwardGradXSpec
    (st : BlockState) (lambda_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length t : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  st.readMem grad_y_ptr (timeOffset st batch_size dim BLOCK_SIZE t i) +
    diagSsmBackwardGradSAfter st lambda_ptr grad_y_ptr batch_size dim
      BLOCK_SIZE length i (length - 1 - t)

theorem diagSsmBackwardGradXSpec_reverseTime
    (st : BlockState) (lambda_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length k : Nat) (i : Fin BLOCK_SIZE)
    (hk : k < length) :
    diagSsmBackwardGradXSpec st lambda_ptr grad_y_ptr batch_size dim
        BLOCK_SIZE length (reverseTime length k) i =
      st.readMem grad_y_ptr
          (timeOffset st batch_size dim BLOCK_SIZE (reverseTime length k) i) +
        diagSsmBackwardGradSAfter st lambda_ptr grad_y_ptr batch_size dim
          BLOCK_SIZE length i k := by
  simp [diagSsmBackwardGradXSpec, reverseTime_inverse length k hk]

noncomputable def diagSsmBackwardGradSTile
    (st : BlockState) (lambda_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length k : Nat) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if active st batch_size dim BLOCK_SIZE idx.1 then
        some (diagSsmBackwardGradSAfter st lambda_ptr grad_y_ptr batch_size dim
          BLOCK_SIZE length idx.1 k)
      else
        some (0.0 : ℝ) }

noncomputable def diagSsmBackwardGradLambdaTile
    (st : BlockState) (s_ptr lambda_ptr y_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length k : Nat) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if active st batch_size dim BLOCK_SIZE idx.1 then
        some (diagSsmBackwardGradLambdaAfter st s_ptr lambda_ptr y_ptr
          grad_y_ptr batch_size dim BLOCK_SIZE length idx.1 k)
      else
        some (0.0 : ℝ) }

theorem diagSsmBackwardGradSTile_zero
    (st : BlockState) (lambda_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length : Nat) :
    diagSsmBackwardGradSTile st lambda_ptr grad_y_ptr batch_size dim
        BLOCK_SIZE length 0 =
      { data := fun _idx : TileIndex [BLOCK_SIZE] => some 0 } := by
  ext idx
  by_cases hactive : active st batch_size dim BLOCK_SIZE idx.1
  · simp [diagSsmBackwardGradSTile, diagSsmBackwardGradSAfter, hactive]
  · simp [diagSsmBackwardGradSTile, diagSsmBackwardGradSAfter, hactive]
    norm_num

theorem diagSsmBackwardGradLambdaTile_zero
    (st : BlockState) (s_ptr lambda_ptr y_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length : Nat) :
    diagSsmBackwardGradLambdaTile st s_ptr lambda_ptr y_ptr grad_y_ptr
        batch_size dim BLOCK_SIZE length 0 =
      { data := fun _idx : TileIndex [BLOCK_SIZE] => some 0 } := by
  ext idx
  by_cases hactive : active st batch_size dim BLOCK_SIZE idx.1
  · simp [diagSsmBackwardGradLambdaTile, diagSsmBackwardGradLambdaAfter, hactive]
  · simp [diagSsmBackwardGradLambdaTile, diagSsmBackwardGradLambdaAfter, hactive]
    norm_num

theorem diagSsmBackwardGradSTile_succ
    (st : BlockState) (lambda_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length k : Nat) :
    diagSsmBackwardGradSTile st lambda_ptr grad_y_ptr batch_size dim
        BLOCK_SIZE length (k + 1) =
      { data := fun idx : TileIndex [BLOCK_SIZE] =>
          some
            (if active st batch_size dim BLOCK_SIZE idx.1 then
              (st.readMem grad_y_ptr
                  (timeOffset st batch_size dim BLOCK_SIZE
                    (reverseTime length k) idx.1) +
                diagSsmBackwardGradSAfter st lambda_ptr grad_y_ptr batch_size
                  dim BLOCK_SIZE length idx.1 k) *
                st.readMem lambda_ptr
                  (IntegralDType.nat.mod (colOffset st BLOCK_SIZE idx.1) dim)
            else
              0.0) } := by
  ext idx
  by_cases hactive : active st batch_size dim BLOCK_SIZE idx.1
  · simp [diagSsmBackwardGradSTile, diagSsmBackwardGradSAfter, hactive]
  · simp [diagSsmBackwardGradSTile, diagSsmBackwardGradSAfter, hactive]

theorem diagSsmBackwardGradLambdaTile_succ
    (st : BlockState) (s_ptr lambda_ptr y_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length k : Nat) :
    diagSsmBackwardGradLambdaTile st s_ptr lambda_ptr y_ptr grad_y_ptr
        batch_size dim BLOCK_SIZE length (k + 1) =
      { data := fun idx : TileIndex [BLOCK_SIZE] =>
          some
            (if active st batch_size dim BLOCK_SIZE idx.1 then
              diagSsmBackwardGradLambdaAfter st s_ptr lambda_ptr y_ptr
                  grad_y_ptr batch_size dim BLOCK_SIZE length idx.1 k +
                (st.readMem grad_y_ptr
                    (timeOffset st batch_size dim BLOCK_SIZE
                      (reverseTime length k) idx.1) +
                  diagSsmBackwardGradSAfter st lambda_ptr grad_y_ptr
                    batch_size dim BLOCK_SIZE length idx.1 k) *
                  diagSsmBackwardPrevState st s_ptr y_ptr batch_size dim
                    BLOCK_SIZE length k idx.1
            else
              0.0) } := by
  ext idx
  by_cases hactive : active st batch_size dim BLOCK_SIZE idx.1
  · simp [diagSsmBackwardGradLambdaTile, diagSsmBackwardGradLambdaAfter,
      hactive]
  · simp [diagSsmBackwardGradLambdaTile, diagSsmBackwardGradLambdaAfter,
      hactive]

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

def diagSsmBackwardGradXOffset
    {length : Nat}
    (st : BlockState) (batch_size dim BLOCK_SIZE : Nat)
    (idx : TileIndex [length, BLOCK_SIZE]) : Nat :=
  timeOffset st batch_size dim BLOCK_SIZE (reverseTime length idx.1.val)
    idx.2.1

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

inductive DiagSsmBackwardOutput (length BLOCK_SIZE : Nat) where
  | gradX : TileIndex [length, BLOCK_SIZE] → DiagSsmBackwardOutput length BLOCK_SIZE
  | gradS : Fin BLOCK_SIZE → DiagSsmBackwardOutput length BLOCK_SIZE
  | gradLambda : Fin BLOCK_SIZE → DiagSsmBackwardOutput length BLOCK_SIZE

def diagSsmBackwardOutputActive
    {length : Nat}
    (st : BlockState) (batch_size dim BLOCK_SIZE : Nat) :
    DiagSsmBackwardOutput length BLOCK_SIZE → Prop
  | .gradX idx => active st batch_size dim BLOCK_SIZE idx.2.1
  | .gradS i => active st batch_size dim BLOCK_SIZE i
  | .gradLambda i => active st batch_size dim BLOCK_SIZE i

def diagSsmBackwardOutputAddr
    {length : Nat}
    (st : BlockState) (grad_s_ptr grad_x_ptr grad_lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat) :
    DiagSsmBackwardOutput length BLOCK_SIZE → MemCellAddr
  | .gradX idx =>
      (grad_x_ptr,
        timeOffset st batch_size dim BLOCK_SIZE (reverseTime length idx.1.val)
          idx.2.1)
  | .gradS i => (grad_s_ptr, colOffset st BLOCK_SIZE i)
  | .gradLambda i => (grad_lambda_ptr, colOffset st BLOCK_SIZE i)

noncomputable def diagSsmBackwardWriteMap
    {length : Nat}
    (st : BlockState) (grad_s_ptr grad_x_ptr grad_lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat) :
    ComputeCorrect.WriteMap (DiagSsmBackwardOutput length BLOCK_SIZE) := by
  classical
  exact fun out =>
    if diagSsmBackwardOutputActive st batch_size dim BLOCK_SIZE out then
      some (diagSsmBackwardOutputAddr st grad_s_ptr grad_x_ptr
        grad_lambda_ptr batch_size dim BLOCK_SIZE out)
    else
      none

noncomputable def diagSsmBackwardExpected
    {length : Nat}
    (st : BlockState)
    (s_ptr lambda_ptr y_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat) :
    DiagSsmBackwardOutput length BLOCK_SIZE → ℝ
  | .gradX idx =>
      diagSsmBackwardGradXSpec st lambda_ptr grad_y_ptr batch_size dim
        BLOCK_SIZE length (reverseTime length idx.1.val) idx.2.1
  | .gradS i =>
      diagSsmBackwardGradSAfter st lambda_ptr grad_y_ptr batch_size dim
        BLOCK_SIZE length i length
  | .gradLambda i =>
      diagSsmBackwardGradLambdaAfter st s_ptr lambda_ptr y_ptr grad_y_ptr
        batch_size dim BLOCK_SIZE length i length

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

@[simp] theorem diagSsmBackwardGradXOffset_currentIteration
    {length : Nat}
    (st : BlockState) (batch_size dim BLOCK_SIZE k : Nat)
    (i : Fin BLOCK_SIZE) (hk : k < length) :
    diagSsmBackwardGradXOffset st batch_size dim BLOCK_SIZE
        ((⟨k, hk⟩ : Fin length), i, PUnit.unit) =
      timeOffset st batch_size dim BLOCK_SIZE (reverseTime length k) i := by
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

theorem diagSsmBackwardIndex_ne_currentIteration
    {length : Nat} (idx : TileIndex [length, BLOCK_SIZE])
    (k : Nat) (i : Fin BLOCK_SIZE)
    (hk : k < length) (hOld : idx.1.val < k) :
    idx ≠ ((⟨k, hk⟩ : Fin length), i, PUnit.unit) := by
  intro hEq
  have htime : idx.1.val = k := by
    have h := congrArg (fun idx : TileIndex [length, BLOCK_SIZE] => idx.1.val) hEq
    simpa using h
  omega

theorem diagSsmBackwardGradXOffset_ne_currentIteration
    {length : Nat}
    (st : BlockState) (batch_size dim BLOCK_SIZE k : Nat)
    (idx : TileIndex [length, BLOCK_SIZE]) (i : Fin BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmBackwardGradXOffset st batch_size dim BLOCK_SIZE idx))
    (hk : k < length) (hOld : idx.1.val < k) :
    diagSsmBackwardGradXOffset st batch_size dim BLOCK_SIZE idx ≠
      timeOffset st batch_size dim BLOCK_SIZE (reverseTime length k) i := by
  intro hEq
  have hCurrent :
      diagSsmBackwardGradXOffset st batch_size dim BLOCK_SIZE
          ((⟨k, hk⟩ : Fin length), i, PUnit.unit) =
        timeOffset st batch_size dim BLOCK_SIZE (reverseTime length k) i := by
    rfl
  have hEqIdx :
      idx = ((⟨k, hk⟩ : Fin length), i, PUnit.unit) := by
    apply hOutInj
    simpa [hCurrent] using hEq
  exact diagSsmBackwardIndex_ne_currentIteration idx k i hk hOld hEqIdx

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

def diagSsmBackwardPreLoop
    (lambda_ptr : RegionName)
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
  , .assign .real [BLOCK_SIZE] "Lambda"
      (.load .real
        (.region lambda_ptr
          (.mod IntegralDType.nat Broadcast.scalarR
            (.ref .nat [BLOCK_SIZE] "col_offsets") (.constNat dim)))
        (.maskOther
          (.ref .bool [BLOCK_SIZE] "mask")
          ((Op.const 0).broadcast [BLOCK_SIZE])))
  , .assign .real [BLOCK_SIZE] "grad_s"
      (.full [BLOCK_SIZE] (.const (0.0 : ℝ)))
  , .assign .real [BLOCK_SIZE] "grad_Lambda"
      (.full [BLOCK_SIZE] (.const (0.0 : ℝ)))
  ]

def diagSsmBackwardLoopBody
    (s_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) : List Stmt :=
  [ .assign .nat [] "t"
      (.sub NumericDType.nat Broadcast.nil
        (.sub NumericDType.nat Broadcast.nil
          (.constNat length) (.constNat 1))
        (.ref .nat [] "i"))
  , .assign .nat [BLOCK_SIZE] "offsets"
      (.add NumericDType.nat Broadcast.scalarL
        (.mul NumericDType.nat Broadcast.nil
          (.ref .nat [] "t") (.constNat (batch_size * dim)))
        (.ref .nat [BLOCK_SIZE] "col_offsets"))
  , .assign .real [BLOCK_SIZE] "grad_y"
      (.load .real
        (.region grad_y_ptr (.ref .nat [BLOCK_SIZE] "offsets"))
        (.maskOther
          (.ref .bool [BLOCK_SIZE] "mask")
          ((Op.const 0).broadcast [BLOCK_SIZE])))
  , .ifThenElse
      (.gt ComparableDType.nat Broadcast.nil
        (.ref .nat [] "t") (.constNat 0))
      [ .assign .real [BLOCK_SIZE] "s"
          (.load .real
            (.region y_ptr
              (.sub NumericDType.nat Broadcast.scalarR
                (.ref .nat [BLOCK_SIZE] "offsets")
                (.constNat (batch_size * dim))))
            (.maskOther
              (.ref .bool [BLOCK_SIZE] "mask")
              ((Op.const 0).broadcast [BLOCK_SIZE])))
      ]
      [ .assign .real [BLOCK_SIZE] "s"
          (.load .real
            (.region s_ptr (.ref .nat [BLOCK_SIZE] "col_offsets"))
            (.maskOther
              (.ref .bool [BLOCK_SIZE] "mask")
              ((Op.const 0).broadcast [BLOCK_SIZE])))
      ]
  , .assign .real [BLOCK_SIZE] "grad_s"
      (.add NumericDType.real Broadcast.nil.consSame
        (.ref .real [BLOCK_SIZE] "grad_y")
        (.ref .real [BLOCK_SIZE] "grad_s"))
  , .assign .real [BLOCK_SIZE] "grad_x"
      (.ref .real [BLOCK_SIZE] "grad_s")
  , .assign .real [BLOCK_SIZE] "grad_Lambda"
      (.add NumericDType.real Broadcast.nil.consSame
        (.ref .real [BLOCK_SIZE] "grad_Lambda")
        (.mul NumericDType.real Broadcast.nil.consSame
          (.ref .real [BLOCK_SIZE] "grad_s")
          (.ref .real [BLOCK_SIZE] "s")))
  , .assign .real [BLOCK_SIZE] "grad_s"
      (.mul NumericDType.real Broadcast.nil.consSame
        (.ref .real [BLOCK_SIZE] "grad_s")
        (.ref .real [BLOCK_SIZE] "Lambda"))
  , .store .real [BLOCK_SIZE]
      (.region grad_x_ptr (.ref .nat [BLOCK_SIZE] "offsets"))
      (.ref .real [BLOCK_SIZE] "grad_x")
      (.mask (.ref .bool [BLOCK_SIZE] "mask"))
  ]

def diagSsmBackwardPostLoop
    (grad_s_ptr grad_lambda_ptr : RegionName) (BLOCK_SIZE : Nat) :
    List Stmt :=
  [ .store .real [BLOCK_SIZE]
      (.region grad_s_ptr (.ref .nat [BLOCK_SIZE] "col_offsets"))
      (.ref .real [BLOCK_SIZE] "grad_s")
      (.mask (.ref .bool [BLOCK_SIZE] "mask"))
  , .store .real [BLOCK_SIZE]
      (.region grad_lambda_ptr (.ref .nat [BLOCK_SIZE] "col_offsets"))
      (.ref .real [BLOCK_SIZE] "grad_Lambda")
      (.mask (.ref .bool [BLOCK_SIZE] "mask"))
  ]

def diagSsmBackwardProjectedBody
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) : List Stmt :=
  diagSsmBackwardPreLoop lambda_ptr batch_size dim BLOCK_SIZE ++
    [.forRange "i" 0 length 1
      (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr length
        batch_size dim BLOCK_SIZE)] ++
    diagSsmBackwardPostLoop grad_s_ptr grad_lambda_ptr BLOCK_SIZE

theorem diag_ssm_backward_kernel_toAlg_body
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    (diag_ssm_backward_kernel s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr
      grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body =
      diagSsmBackwardProjectedBody s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr
        grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE := by
  rfl

theorem diagSsmBackwardPreLoop_step_regs
    (st0 st : BlockState) (lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat)
    (hStep :
      stepStmts (diagSsmBackwardPreLoop lambda_ptr batch_size dim BLOCK_SIZE)
        st0 = some st) :
    st.regs .real [BLOCK_SIZE] "Lambda" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          if active st0 batch_size dim BLOCK_SIZE idx.1 then
            some (st0.readMem lambda_ptr
              (IntegralDType.nat.mod (colOffset st0 BLOCK_SIZE idx.1) dim))
          else
            some 0 } ∧
      st.regs .real [BLOCK_SIZE] "grad_s" =
        some { data := fun _idx : TileIndex [BLOCK_SIZE] => some 0 } ∧
      st.regs .real [BLOCK_SIZE] "grad_Lambda" =
        some { data := fun _idx : TileIndex [BLOCK_SIZE] => some 0 } ∧
      st.regs .nat [BLOCK_SIZE] "col_offsets" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset st0 BLOCK_SIZE idx.1 } ∧
      st.regs .bool [BLOCK_SIZE] "mask" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active st0 batch_size dim BLOCK_SIZE idx.1 } := by
  unfold diagSsmBackwardPreLoop at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.cop,
    NumericDType.mul, NumericDType.add, IntegralDType.mod,
    ComparableDType.lt, Option.bind, colOffset, active] at hStep
  subst st
  constructor
  · simp [BlockState.setReg, active, colOffset, IntegralDType.mod]
  · constructor
    · simp [BlockState.setReg]
      funext idx
      norm_num
    · constructor
      · simp [BlockState.setReg]
        funext idx
        norm_num
      · constructor
        · simp [BlockState.setReg, colOffset]
        · simp [BlockState.setReg, active, colOffset]
          funext idx
          rfl

def diagSsmBackwardLoopInvariant
    (st0 : BlockState)
    (s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (k : Nat)
    (st : BlockState) : Prop :=
  st.regs .real [BLOCK_SIZE] "grad_s" =
      some (diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr batch_size dim
        BLOCK_SIZE length k) ∧
    st.regs .real [BLOCK_SIZE] "grad_Lambda" =
      some (diagSsmBackwardGradLambdaTile st0 s_ptr lambda_ptr y_ptr
        grad_y_ptr batch_size dim BLOCK_SIZE length k) ∧
    ∀ idx : TileIndex [length, BLOCK_SIZE],
      idx.1.val < k →
        active st0 batch_size dim BLOCK_SIZE idx.2.1 →
          st.readMem grad_x_ptr
              (timeOffset st0 batch_size dim BLOCK_SIZE
                (reverseTime length idx.1.val) idx.2.1) =
            diagSsmBackwardGradXSpec st0 lambda_ptr grad_y_ptr batch_size dim
              BLOCK_SIZE length (reverseTime length idx.1.val) idx.2.1

theorem diagSsmBackwardLoopInvariant_zero
    (st0 st : BlockState)
    (s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hGradS :
      st.regs .real [BLOCK_SIZE] "grad_s" =
        some { data := fun _idx : TileIndex [BLOCK_SIZE] => some 0 })
    (hGradLambda :
      st.regs .real [BLOCK_SIZE] "grad_Lambda" =
        some { data := fun _idx : TileIndex [BLOCK_SIZE] => some 0 }) :
    diagSsmBackwardLoopInvariant st0 s_ptr lambda_ptr y_ptr grad_x_ptr
      grad_y_ptr length batch_size dim BLOCK_SIZE 0 st := by
  constructor
  · rw [hGradS, diagSsmBackwardGradSTile_zero]
  · constructor
    · rw [hGradLambda, diagSsmBackwardGradLambdaTile_zero]
    · intro idx hlt _
      omega

theorem diagSsmBackwardCurrentIterationScatter_write
    {length : Nat}
    (st0 st : BlockState)
    (lambda_ptr grad_x_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE k : Nat)
    (i : Fin BLOCK_SIZE)
    (hactive : active st0 batch_size dim BLOCK_SIZE i)
    (hNoCollision :
      ∀ lane : TileIndex [BLOCK_SIZE],
        active st0 batch_size dim BLOCK_SIZE lane.1 →
          timeOffset st0 batch_size dim BLOCK_SIZE (reverseTime length k)
              lane.1 =
            timeOffset st0 batch_size dim BLOCK_SIZE (reverseTime length k) i →
          lane = ((i, PUnit.unit) : TileIndex [BLOCK_SIZE])) :
    ((TileShape.allIndices [BLOCK_SIZE]).foldl
        (fun acc lane =>
          if active st0 batch_size dim BLOCK_SIZE lane.1 then
            acc.writeMem grad_x_ptr
              (timeOffset st0 batch_size dim BLOCK_SIZE
                (reverseTime length k) lane.1)
              (diagSsmBackwardGradXSpec st0 lambda_ptr grad_y_ptr batch_size
                dim BLOCK_SIZE length (reverseTime length k) lane.1)
          else
            acc)
        st).readMem grad_x_ptr
          (timeOffset st0 batch_size dim BLOCK_SIZE (reverseTime length k) i) =
      diagSsmBackwardGradXSpec st0 lambda_ptr grad_y_ptr batch_size dim
        BLOCK_SIZE length (reverseTime length k) i := by
  exact
    BlockState.scatter_readback_prop_masked_nd_of_true
      (region := grad_x_ptr)
      (s := st)
      (offsetFn := fun lane : TileIndex [BLOCK_SIZE] =>
        timeOffset st0 batch_size dim BLOCK_SIZE (reverseTime length k)
          lane.1)
      (valueFn := fun lane : TileIndex [BLOCK_SIZE] =>
        diagSsmBackwardGradXSpec st0 lambda_ptr grad_y_ptr batch_size dim
          BLOCK_SIZE length (reverseTime length k) lane.1)
      (P := fun lane : TileIndex [BLOCK_SIZE] =>
        active st0 batch_size dim BLOCK_SIZE lane.1)
      ((i, PUnit.unit) : TileIndex [BLOCK_SIZE])
      hactive
      (fun lane hlane heq => hNoCollision lane hlane heq)

theorem diagSsmBackwardCurrentIterationNoCollision_of_out_injective
    {length : Nat}
    (st0 : BlockState) (batch_size dim BLOCK_SIZE k : Nat)
    (i : Fin BLOCK_SIZE) (hk : k < length)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx)) :
    ∀ lane : TileIndex [BLOCK_SIZE],
      active st0 batch_size dim BLOCK_SIZE lane.1 →
        timeOffset st0 batch_size dim BLOCK_SIZE (reverseTime length k)
            lane.1 =
          timeOffset st0 batch_size dim BLOCK_SIZE (reverseTime length k) i →
        lane = ((i, PUnit.unit) : TileIndex [BLOCK_SIZE]) := by
  intro lane _hactive heq
  have hFull :
      ((⟨k, hk⟩ : Fin length), lane.1, PUnit.unit) =
        ((⟨k, hk⟩ : Fin length), i, PUnit.unit) := by
    apply hOutInj
    simpa [diagSsmBackwardGradXOffset_currentIteration] using heq
  have hLane : lane.1 = i := by
    have h := congrArg
      (fun idx : TileIndex [length, BLOCK_SIZE] => idx.2.1) hFull
    simpa using h
  cases lane with
  | mk laneHead laneTail =>
      cases laneTail
      cases hLane
      rfl

theorem diagSsmBackwardCurrentIterationScatter_preserve_old
    {length : Nat}
    (st0 st : BlockState)
    (lambda_ptr grad_x_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE k : Nat)
    (idx : TileIndex [length, BLOCK_SIZE])
    (hk : k < length) (hOld : idx.1.val < k)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx)) :
    ((TileShape.allIndices [BLOCK_SIZE]).foldl
        (fun acc lane =>
          if active st0 batch_size dim BLOCK_SIZE lane.1 then
            acc.writeMem grad_x_ptr
              (timeOffset st0 batch_size dim BLOCK_SIZE
                (reverseTime length k) lane.1)
              (diagSsmBackwardGradXSpec st0 lambda_ptr grad_y_ptr batch_size
                dim BLOCK_SIZE length (reverseTime length k) lane.1)
          else
            acc)
        st).readMem grad_x_ptr
          (diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx) =
      st.readMem grad_x_ptr
        (diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx) := by
  exact
    BlockState.scatter_prop_masked_preserves_other_offset
      grad_x_ptr
      (fun lane : TileIndex [BLOCK_SIZE] =>
        timeOffset st0 batch_size dim BLOCK_SIZE (reverseTime length k)
          lane.1)
      (fun lane : TileIndex [BLOCK_SIZE] =>
        diagSsmBackwardGradXSpec st0 lambda_ptr grad_y_ptr batch_size dim
          BLOCK_SIZE length (reverseTime length k) lane.1)
      (fun lane : TileIndex [BLOCK_SIZE] =>
        active st0 batch_size dim BLOCK_SIZE lane.1)
      (diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx)
      (fun lane _hactive heq =>
        diagSsmBackwardGradXOffset_ne_currentIteration st0 batch_size dim
          BLOCK_SIZE k idx lane.1 hOutInj hk hOld heq.symm)
      (TileShape.allIndices [BLOCK_SIZE]) st

theorem diagSsmBackwardCurrentIterationScatter_preserve_region
    {length : Nat}
    (st0 st : BlockState)
    (lambda_ptr grad_x_ptr grad_y_ptr region : RegionName)
    (batch_size dim BLOCK_SIZE k offset : Nat)
    (hRegion : grad_x_ptr ≠ region) :
    ((TileShape.allIndices [BLOCK_SIZE]).foldl
        (fun acc lane =>
          if active st0 batch_size dim BLOCK_SIZE lane.1 then
            acc.writeMem grad_x_ptr
              (timeOffset st0 batch_size dim BLOCK_SIZE
                (reverseTime length k) lane.1)
              (diagSsmBackwardGradXSpec st0 lambda_ptr grad_y_ptr batch_size
                dim BLOCK_SIZE length (reverseTime length k) lane.1)
          else
            acc)
        st).readMem region offset =
      st.readMem region offset := by
  have hRegion' : region ≠ grad_x_ptr := by
    exact fun h => hRegion h.symm
  simpa using
    BlockState.scatter_prop_masked_preserves_other_region
      (region := grad_x_ptr)
      (offsetFn := fun lane : TileIndex [BLOCK_SIZE] =>
        timeOffset st0 batch_size dim BLOCK_SIZE (reverseTime length k)
          lane.1)
      (valueFn := fun lane : TileIndex [BLOCK_SIZE] =>
        diagSsmBackwardGradXSpec st0 lambda_ptr grad_y_ptr batch_size dim
          BLOCK_SIZE length (reverseTime length k) lane.1)
      (P := fun lane : TileIndex [BLOCK_SIZE] =>
        active st0 batch_size dim BLOCK_SIZE lane.1)
      (R := region) hRegion' offset (TileShape.allIndices [BLOCK_SIZE]) st

theorem diagSsmBackwardCurrentIterationScatter_preserve_region_of_value
    {length : Nat}
    (st0 st : BlockState)
    (grad_x_ptr region : RegionName)
    (batch_size dim BLOCK_SIZE k offset : Nat)
    (valueFn : TileIndex [BLOCK_SIZE] → ℝ)
    (hRegion : grad_x_ptr ≠ region) :
    ((TileShape.allIndices [BLOCK_SIZE]).foldl
        (fun acc lane =>
          if active st0 batch_size dim BLOCK_SIZE lane.1 then
            acc.writeMem grad_x_ptr
              (timeOffset st0 batch_size dim BLOCK_SIZE
                (reverseTime length k) lane.1)
              (valueFn lane)
          else
            acc)
        st).readMem region offset =
      st.readMem region offset := by
  have hRegion' : region ≠ grad_x_ptr := by
    exact fun h => hRegion h.symm
  simpa using
    BlockState.scatter_prop_masked_preserves_other_region
      (region := grad_x_ptr)
      (offsetFn := fun lane : TileIndex [BLOCK_SIZE] =>
        timeOffset st0 batch_size dim BLOCK_SIZE (reverseTime length k)
          lane.1)
      (valueFn := valueFn)
      (P := fun lane : TileIndex [BLOCK_SIZE] =>
        active st0 batch_size dim BLOCK_SIZE lane.1)
      (R := region) hRegion' offset (TileShape.allIndices [BLOCK_SIZE]) st

theorem diagSsmBackwardFinalGradSScatter_write
    (st0 st : BlockState)
    (lambda_ptr grad_s_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length : Nat)
    (i : Fin BLOCK_SIZE)
    (hactive : active st0 batch_size dim BLOCK_SIZE i) :
    ((TileShape.allIndices [BLOCK_SIZE]).foldl
        (fun acc lane =>
          if active st0 batch_size dim BLOCK_SIZE lane.1 then
            acc.writeMem grad_s_ptr
              (colOffset st0 BLOCK_SIZE lane.1)
              (diagSsmBackwardGradSAfter st0 lambda_ptr grad_y_ptr batch_size
                dim BLOCK_SIZE length lane.1 length)
          else
            acc)
        st).readMem grad_s_ptr (colOffset st0 BLOCK_SIZE i) =
      diagSsmBackwardGradSAfter st0 lambda_ptr grad_y_ptr batch_size dim
        BLOCK_SIZE length i length := by
  exact
    BlockState.scatter_readback_prop_masked_nd_of_true
      (region := grad_s_ptr)
      (s := st)
      (offsetFn := fun lane : TileIndex [BLOCK_SIZE] =>
        colOffset st0 BLOCK_SIZE lane.1)
      (valueFn := fun lane : TileIndex [BLOCK_SIZE] =>
        diagSsmBackwardGradSAfter st0 lambda_ptr grad_y_ptr batch_size dim
          BLOCK_SIZE length lane.1 length)
      (P := fun lane : TileIndex [BLOCK_SIZE] =>
        active st0 batch_size dim BLOCK_SIZE lane.1)
      ((i, PUnit.unit) : TileIndex [BLOCK_SIZE])
      hactive
      (by
        intro lane _hactive heq
        have hlane : lane.1 = i := colOffset_injective st0 BLOCK_SIZE heq
        cases lane with
        | mk laneHead laneTail =>
            cases laneTail
            cases hlane
            rfl)

theorem diagSsmBackwardFinalGradSScatter_write_tile
    (st0 st : BlockState)
    (lambda_ptr grad_s_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length : Nat)
    (i : Fin BLOCK_SIZE)
    (hactive : active st0 batch_size dim BLOCK_SIZE i) :
    ((TileShape.allIndices [BLOCK_SIZE]).foldl
        (fun acc lane =>
          if active st0 batch_size dim BLOCK_SIZE lane.1 then
            acc.writeMem grad_s_ptr
              (colOffset st0 BLOCK_SIZE lane.1)
              (WithBot.unbotD 0
                ((diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr
                  batch_size dim BLOCK_SIZE length length).data lane))
          else
            acc)
        st).readMem grad_s_ptr (colOffset st0 BLOCK_SIZE i) =
      diagSsmBackwardGradSAfter st0 lambda_ptr grad_y_ptr batch_size dim
        BLOCK_SIZE length i length := by
  have hRead :=
    BlockState.scatter_readback_prop_masked_nd_of_true
      (region := grad_s_ptr)
      (s := st)
      (offsetFn := fun lane : TileIndex [BLOCK_SIZE] =>
        colOffset st0 BLOCK_SIZE lane.1)
      (valueFn := fun lane : TileIndex [BLOCK_SIZE] =>
        WithBot.unbotD 0
          ((diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr batch_size dim
            BLOCK_SIZE length length).data lane))
      (P := fun lane : TileIndex [BLOCK_SIZE] =>
        active st0 batch_size dim BLOCK_SIZE lane.1)
      ((i, PUnit.unit) : TileIndex [BLOCK_SIZE])
      hactive
      (by
        intro lane _hactive heq
        have hlane : lane.1 = i := colOffset_injective st0 BLOCK_SIZE heq
        cases lane with
        | mk laneHead laneTail =>
            cases laneTail
            cases hlane
            rfl)
  simpa [diagSsmBackwardGradSTile, hactive] using hRead

theorem diagSsmBackwardFinalGradSScatter_preserve_region_tile
    (st0 st : BlockState)
    (lambda_ptr grad_s_ptr grad_y_ptr region : RegionName)
    (batch_size dim BLOCK_SIZE length offset : Nat)
    (hRegion : grad_s_ptr ≠ region) :
    ((TileShape.allIndices [BLOCK_SIZE]).foldl
        (fun acc lane =>
          if active st0 batch_size dim BLOCK_SIZE lane.1 then
            acc.writeMem grad_s_ptr
              (colOffset st0 BLOCK_SIZE lane.1)
              (WithBot.unbotD 0
                ((diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr
                  batch_size dim BLOCK_SIZE length length).data lane))
          else
            acc)
        st).readMem region offset =
      st.readMem region offset := by
  have hRegion' : region ≠ grad_s_ptr := by
    exact fun h => hRegion h.symm
  simpa using
    BlockState.scatter_prop_masked_preserves_other_region
      (region := grad_s_ptr)
      (offsetFn := fun lane : TileIndex [BLOCK_SIZE] =>
        colOffset st0 BLOCK_SIZE lane.1)
      (valueFn := fun lane : TileIndex [BLOCK_SIZE] =>
        WithBot.unbotD 0
          ((diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr batch_size dim
            BLOCK_SIZE length length).data lane))
      (P := fun lane : TileIndex [BLOCK_SIZE] =>
        active st0 batch_size dim BLOCK_SIZE lane.1)
      (R := region) hRegion' offset (TileShape.allIndices [BLOCK_SIZE]) st

theorem diagSsmBackwardFinalGradLambdaScatter_write
    (st0 st : BlockState)
    (s_ptr lambda_ptr y_ptr grad_lambda_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length : Nat)
    (i : Fin BLOCK_SIZE)
    (hactive : active st0 batch_size dim BLOCK_SIZE i) :
    ((TileShape.allIndices [BLOCK_SIZE]).foldl
        (fun acc lane =>
          if active st0 batch_size dim BLOCK_SIZE lane.1 then
            acc.writeMem grad_lambda_ptr
              (colOffset st0 BLOCK_SIZE lane.1)
              (diagSsmBackwardGradLambdaAfter st0 s_ptr lambda_ptr y_ptr
                grad_y_ptr batch_size dim BLOCK_SIZE length lane.1 length)
          else
            acc)
        st).readMem grad_lambda_ptr (colOffset st0 BLOCK_SIZE i) =
      diagSsmBackwardGradLambdaAfter st0 s_ptr lambda_ptr y_ptr grad_y_ptr
        batch_size dim BLOCK_SIZE length i length := by
  exact
    BlockState.scatter_readback_prop_masked_nd_of_true
      (region := grad_lambda_ptr)
      (s := st)
      (offsetFn := fun lane : TileIndex [BLOCK_SIZE] =>
        colOffset st0 BLOCK_SIZE lane.1)
      (valueFn := fun lane : TileIndex [BLOCK_SIZE] =>
        diagSsmBackwardGradLambdaAfter st0 s_ptr lambda_ptr y_ptr grad_y_ptr
          batch_size dim BLOCK_SIZE length lane.1 length)
      (P := fun lane : TileIndex [BLOCK_SIZE] =>
        active st0 batch_size dim BLOCK_SIZE lane.1)
      ((i, PUnit.unit) : TileIndex [BLOCK_SIZE])
      hactive
      (by
        intro lane _hactive heq
        have hlane : lane.1 = i := colOffset_injective st0 BLOCK_SIZE heq
        cases lane with
        | mk laneHead laneTail =>
            cases laneTail
            cases hlane
            rfl)

theorem diagSsmBackwardFinalGradLambdaScatter_write_tile
    (st0 st : BlockState)
    (s_ptr lambda_ptr y_ptr grad_lambda_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE length : Nat)
    (i : Fin BLOCK_SIZE)
    (hactive : active st0 batch_size dim BLOCK_SIZE i) :
    ((TileShape.allIndices [BLOCK_SIZE]).foldl
        (fun acc lane =>
          if active st0 batch_size dim BLOCK_SIZE lane.1 then
            acc.writeMem grad_lambda_ptr
              (colOffset st0 BLOCK_SIZE lane.1)
              (WithBot.unbotD 0
                ((diagSsmBackwardGradLambdaTile st0 s_ptr lambda_ptr y_ptr
                  grad_y_ptr batch_size dim BLOCK_SIZE length length).data
                  lane))
          else
            acc)
        st).readMem grad_lambda_ptr (colOffset st0 BLOCK_SIZE i) =
      diagSsmBackwardGradLambdaAfter st0 s_ptr lambda_ptr y_ptr grad_y_ptr
        batch_size dim BLOCK_SIZE length i length := by
  have hRead :=
    BlockState.scatter_readback_prop_masked_nd_of_true
      (region := grad_lambda_ptr)
      (s := st)
      (offsetFn := fun lane : TileIndex [BLOCK_SIZE] =>
        colOffset st0 BLOCK_SIZE lane.1)
      (valueFn := fun lane : TileIndex [BLOCK_SIZE] =>
        WithBot.unbotD 0
          ((diagSsmBackwardGradLambdaTile st0 s_ptr lambda_ptr y_ptr
            grad_y_ptr batch_size dim BLOCK_SIZE length length).data lane))
      (P := fun lane : TileIndex [BLOCK_SIZE] =>
        active st0 batch_size dim BLOCK_SIZE lane.1)
      ((i, PUnit.unit) : TileIndex [BLOCK_SIZE])
      hactive
      (by
        intro lane _hactive heq
        have hlane : lane.1 = i := colOffset_injective st0 BLOCK_SIZE heq
        cases lane with
        | mk laneHead laneTail =>
            cases laneTail
            cases hlane
            rfl)
  simpa [diagSsmBackwardGradLambdaTile, hactive] using hRead

theorem diagSsmBackwardFinalGradLambdaScatter_preserve_region
    (st0 st : BlockState)
    (s_ptr lambda_ptr y_ptr grad_lambda_ptr grad_y_ptr region : RegionName)
    (batch_size dim BLOCK_SIZE length offset : Nat)
    (hRegion : grad_lambda_ptr ≠ region) :
    ((TileShape.allIndices [BLOCK_SIZE]).foldl
        (fun acc lane =>
          if active st0 batch_size dim BLOCK_SIZE lane.1 then
            acc.writeMem grad_lambda_ptr
              (colOffset st0 BLOCK_SIZE lane.1)
              (diagSsmBackwardGradLambdaAfter st0 s_ptr lambda_ptr y_ptr
                grad_y_ptr batch_size dim BLOCK_SIZE length lane.1 length)
          else
            acc)
        st).readMem region offset =
      st.readMem region offset := by
  have hRegion' : region ≠ grad_lambda_ptr := by
    exact fun h => hRegion h.symm
  simpa using
    BlockState.scatter_prop_masked_preserves_other_region
      (region := grad_lambda_ptr)
      (offsetFn := fun lane : TileIndex [BLOCK_SIZE] =>
        colOffset st0 BLOCK_SIZE lane.1)
      (valueFn := fun lane : TileIndex [BLOCK_SIZE] =>
        diagSsmBackwardGradLambdaAfter st0 s_ptr lambda_ptr y_ptr grad_y_ptr
          batch_size dim BLOCK_SIZE length lane.1 length)
      (P := fun lane : TileIndex [BLOCK_SIZE] =>
        active st0 batch_size dim BLOCK_SIZE lane.1)
      (R := region) hRegion' offset (TileShape.allIndices [BLOCK_SIZE]) st

theorem diagSsmBackwardFinalGradLambdaScatter_preserve_region_tile
    (st0 st : BlockState)
    (s_ptr lambda_ptr y_ptr grad_lambda_ptr grad_y_ptr region : RegionName)
    (batch_size dim BLOCK_SIZE length offset : Nat)
    (hRegion : grad_lambda_ptr ≠ region) :
    ((TileShape.allIndices [BLOCK_SIZE]).foldl
        (fun acc lane =>
          if active st0 batch_size dim BLOCK_SIZE lane.1 then
            acc.writeMem grad_lambda_ptr
              (colOffset st0 BLOCK_SIZE lane.1)
              (WithBot.unbotD 0
                ((diagSsmBackwardGradLambdaTile st0 s_ptr lambda_ptr y_ptr
                  grad_y_ptr batch_size dim BLOCK_SIZE length length).data
                  lane))
          else
            acc)
        st).readMem region offset =
      st.readMem region offset := by
  have hRegion' : region ≠ grad_lambda_ptr := by
    exact fun h => hRegion h.symm
  simpa using
    BlockState.scatter_prop_masked_preserves_other_region
      (region := grad_lambda_ptr)
      (offsetFn := fun lane : TileIndex [BLOCK_SIZE] =>
        colOffset st0 BLOCK_SIZE lane.1)
      (valueFn := fun lane : TileIndex [BLOCK_SIZE] =>
        WithBot.unbotD 0
          ((diagSsmBackwardGradLambdaTile st0 s_ptr lambda_ptr y_ptr
            grad_y_ptr batch_size dim BLOCK_SIZE length length).data lane))
      (P := fun lane : TileIndex [BLOCK_SIZE] =>
        active st0 batch_size dim BLOCK_SIZE lane.1)
      (R := region) hRegion' offset (TileShape.allIndices [BLOCK_SIZE]) st

theorem diagSsmBackwardPostLoop_readback
    (st0 st st' : BlockState)
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (batch_size dim BLOCK_SIZE length : Nat)
    (hGradS :
      st.regs .real [BLOCK_SIZE] "grad_s" =
        some (diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr batch_size
          dim BLOCK_SIZE length length))
    (hGradLambda :
      st.regs .real [BLOCK_SIZE] "grad_Lambda" =
        some (diagSsmBackwardGradLambdaTile st0 s_ptr lambda_ptr y_ptr
          grad_y_ptr batch_size dim BLOCK_SIZE length length))
    (hCol :
      st.regs .nat [BLOCK_SIZE] "col_offsets" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset st0 BLOCK_SIZE idx.1 })
    (hMask :
      st.regs .bool [BLOCK_SIZE] "mask" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active st0 batch_size dim BLOCK_SIZE idx.1 })
    (hGradLambdaGradSNe : grad_lambda_ptr ≠ grad_s_ptr)
    (hStep :
      stepStmts (diagSsmBackwardPostLoop grad_s_ptr grad_lambda_ptr
        BLOCK_SIZE) st = some st') :
    (∀ i : Fin BLOCK_SIZE,
      active st0 batch_size dim BLOCK_SIZE i →
        st'.readMem grad_s_ptr (colOffset st0 BLOCK_SIZE i) =
          diagSsmBackwardGradSAfter st0 lambda_ptr grad_y_ptr batch_size dim
            BLOCK_SIZE length i length) ∧
    (∀ i : Fin BLOCK_SIZE,
      active st0 batch_size dim BLOCK_SIZE i →
        st'.readMem grad_lambda_ptr (colOffset st0 BLOCK_SIZE i) =
          diagSsmBackwardGradLambdaAfter st0 s_ptr lambda_ptr y_ptr grad_y_ptr
            batch_size dim BLOCK_SIZE length i length) := by
  unfold diagSsmBackwardPostLoop at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hGradS, hGradLambda, hCol, hMask,
    Option.bind] at hStep
  subst st'
  constructor
  · intro i hactive
    rw [diagSsmBackwardFinalGradLambdaScatter_preserve_region_tile st0 _ s_ptr
      lambda_ptr y_ptr grad_lambda_ptr grad_y_ptr grad_s_ptr batch_size dim
      BLOCK_SIZE length (colOffset st0 BLOCK_SIZE i) hGradLambdaGradSNe]
    exact diagSsmBackwardFinalGradSScatter_write_tile st0 st lambda_ptr
      grad_s_ptr grad_y_ptr batch_size dim BLOCK_SIZE length i hactive
  · intro i hactive
    exact diagSsmBackwardFinalGradLambdaScatter_write_tile st0 _ s_ptr
      lambda_ptr y_ptr grad_lambda_ptr grad_y_ptr batch_size dim BLOCK_SIZE
      length i hactive

theorem diagSsmBackwardPostLoop_preserve_gradX
    (st0 st st' : BlockState)
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (batch_size dim BLOCK_SIZE length offset : Nat)
    (hGradS :
      st.regs .real [BLOCK_SIZE] "grad_s" =
        some (diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr batch_size
          dim BLOCK_SIZE length length))
    (hGradLambda :
      st.regs .real [BLOCK_SIZE] "grad_Lambda" =
        some (diagSsmBackwardGradLambdaTile st0 s_ptr lambda_ptr y_ptr
          grad_y_ptr batch_size dim BLOCK_SIZE length length))
    (hCol :
      st.regs .nat [BLOCK_SIZE] "col_offsets" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset st0 BLOCK_SIZE idx.1 })
    (hMask :
      st.regs .bool [BLOCK_SIZE] "mask" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active st0 batch_size dim BLOCK_SIZE idx.1 })
    (hGradSGradXNe : grad_s_ptr ≠ grad_x_ptr)
    (hGradLambdaGradXNe : grad_lambda_ptr ≠ grad_x_ptr)
    (hStep :
      stepStmts (diagSsmBackwardPostLoop grad_s_ptr grad_lambda_ptr
        BLOCK_SIZE) st = some st') :
    st'.readMem grad_x_ptr offset = st.readMem grad_x_ptr offset := by
  unfold diagSsmBackwardPostLoop at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hGradS, hGradLambda, hCol, hMask,
    Option.bind] at hStep
  subst st'
  rw [diagSsmBackwardFinalGradLambdaScatter_preserve_region_tile st0 _ s_ptr
    lambda_ptr y_ptr grad_lambda_ptr grad_y_ptr grad_x_ptr batch_size dim
    BLOCK_SIZE length offset hGradLambdaGradXNe]
  exact diagSsmBackwardFinalGradSScatter_preserve_region_tile st0 st
    lambda_ptr grad_s_ptr grad_y_ptr grad_x_ptr batch_size dim BLOCK_SIZE
    length offset hGradSGradXNe

def diag_ssm_backward_kernel_alg_post
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (s s' : BlockState) : Prop :=
  ∀ out : DiagSsmBackwardOutput length BLOCK_SIZE,
    diagSsmBackwardOutputActive s batch_size dim BLOCK_SIZE out →
      s'.readMem
          (diagSsmBackwardOutputAddr s grad_s_ptr grad_x_ptr grad_lambda_ptr
            batch_size dim BLOCK_SIZE out).1
          (diagSsmBackwardOutputAddr s grad_s_ptr grad_x_ptr grad_lambda_ptr
            batch_size dim BLOCK_SIZE out).2 =
        diagSsmBackwardExpected (length := length) s s_ptr lambda_ptr y_ptr
          grad_y_ptr batch_size dim BLOCK_SIZE out

theorem diagSsmBackwardLoopInvariant_postLoop_alg_post
    (st0 stLoop stPost : BlockState)
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hInv :
      diagSsmBackwardLoopInvariant st0 s_ptr lambda_ptr y_ptr grad_x_ptr
        grad_y_ptr length batch_size dim BLOCK_SIZE length stLoop)
    (hCol :
      stLoop.regs .nat [BLOCK_SIZE] "col_offsets" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset st0 BLOCK_SIZE idx.1 })
    (hMask :
      stLoop.regs .bool [BLOCK_SIZE] "mask" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active st0 batch_size dim BLOCK_SIZE idx.1 })
    (hGradSGradXNe : grad_s_ptr ≠ grad_x_ptr)
    (hGradLambdaGradXNe : grad_lambda_ptr ≠ grad_x_ptr)
    (hGradLambdaGradSNe : grad_lambda_ptr ≠ grad_s_ptr)
    (hPost :
      stepStmts (diagSsmBackwardPostLoop grad_s_ptr grad_lambda_ptr
        BLOCK_SIZE) stLoop = some stPost) :
    diag_ssm_backward_kernel_alg_post s_ptr lambda_ptr y_ptr grad_s_ptr
      grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
      st0 stPost := by
  have hReadPost :=
    diagSsmBackwardPostLoop_readback st0 stLoop stPost s_ptr lambda_ptr y_ptr
      grad_s_ptr grad_lambda_ptr grad_y_ptr batch_size dim BLOCK_SIZE length
      hInv.1 hInv.2.1 hCol hMask hGradLambdaGradSNe hPost
  intro out hactive
  cases out with
  | gradX idx =>
      simp [diagSsmBackwardOutputActive, diagSsmBackwardOutputAddr,
        diagSsmBackwardExpected] at hactive ⊢
      rw [diagSsmBackwardPostLoop_preserve_gradX st0 stLoop stPost s_ptr
        lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr
        batch_size dim BLOCK_SIZE length
        (timeOffset st0 batch_size dim BLOCK_SIZE (reverseTime length idx.1.val)
          idx.2.1)
        hInv.1 hInv.2.1 hCol hMask hGradSGradXNe hGradLambdaGradXNe hPost]
      exact hInv.2.2 idx idx.1.isLt hactive
  | gradS i =>
      simp [diagSsmBackwardOutputActive, diagSsmBackwardOutputAddr,
        diagSsmBackwardExpected] at hactive ⊢
      exact hReadPost.1 i hactive
  | gradLambda i =>
      simp [diagSsmBackwardOutputActive, diagSsmBackwardOutputAddr,
        diagSsmBackwardExpected] at hactive ⊢
      exact hReadPost.2 i hactive

theorem diagSsmBackwardLoopInvariant_step_of_iteration_write
    {length : Nat}
    (st0 stPrev st' : BlockState)
    (s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE k : Nat)
    (hPrev :
      diagSsmBackwardLoopInvariant st0 s_ptr lambda_ptr y_ptr grad_x_ptr
        grad_y_ptr length batch_size dim BLOCK_SIZE k stPrev)
    (hGradS :
      st'.regs .real [BLOCK_SIZE] "grad_s" =
        some (diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr batch_size
          dim BLOCK_SIZE length (k + 1)))
    (hGradLambda :
      st'.regs .real [BLOCK_SIZE] "grad_Lambda" =
        some (diagSsmBackwardGradLambdaTile st0 s_ptr lambda_ptr y_ptr
          grad_y_ptr batch_size dim BLOCK_SIZE length (k + 1)))
    (hPreserve :
      ∀ idx : TileIndex [length, BLOCK_SIZE],
        idx.1.val < k →
          active st0 batch_size dim BLOCK_SIZE idx.2.1 →
            st'.readMem grad_x_ptr
                (diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx) =
              stPrev.readMem grad_x_ptr
                (diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx))
    (hWrite :
      ∀ i : Fin BLOCK_SIZE,
        active st0 batch_size dim BLOCK_SIZE i →
          st'.readMem grad_x_ptr
              (timeOffset st0 batch_size dim BLOCK_SIZE
                (reverseTime length k) i) =
            diagSsmBackwardGradXSpec st0 lambda_ptr grad_y_ptr batch_size dim
              BLOCK_SIZE length (reverseTime length k) i) :
    diagSsmBackwardLoopInvariant st0 s_ptr lambda_ptr y_ptr grad_x_ptr
      grad_y_ptr length batch_size dim BLOCK_SIZE (k + 1) st' := by
  constructor
  · exact hGradS
  · constructor
    · exact hGradLambda
    · intro idx hlt hactive
      by_cases hOld : idx.1.val < k
      · change st'.readMem grad_x_ptr
              (diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx) =
            diagSsmBackwardGradXSpec st0 lambda_ptr grad_y_ptr batch_size dim
              BLOCK_SIZE length (reverseTime length idx.1.val) idx.2.1
        rw [hPreserve idx hOld hactive]
        exact hPrev.2.2 idx hOld hactive
      · have hk : idx.1.val = k := by omega
        simpa [diagSsmBackwardGradXOffset, hk] using hWrite idx.2.1 hactive

theorem diagSsmBackwardLoopInvariant_step_of_current_iteration_scatter
    {length : Nat}
    (st0 stPrev stReg : BlockState)
    (s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE k : Nat)
    (hk : k < length)
    (hPrev :
      diagSsmBackwardLoopInvariant st0 s_ptr lambda_ptr y_ptr grad_x_ptr
        grad_y_ptr length batch_size dim BLOCK_SIZE k stPrev)
    (hGradS :
      stReg.regs .real [BLOCK_SIZE] "grad_s" =
        some (diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr batch_size
          dim BLOCK_SIZE length (k + 1)))
    (hGradLambda :
      stReg.regs .real [BLOCK_SIZE] "grad_Lambda" =
        some (diagSsmBackwardGradLambdaTile st0 s_ptr lambda_ptr y_ptr
          grad_y_ptr batch_size dim BLOCK_SIZE length (k + 1)))
    (hMem :
      ∀ offset, stReg.readMem grad_x_ptr offset =
        stPrev.readMem grad_x_ptr offset)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx)) :
    diagSsmBackwardLoopInvariant st0 s_ptr lambda_ptr y_ptr grad_x_ptr
      grad_y_ptr length batch_size dim BLOCK_SIZE (k + 1)
      ((TileShape.allIndices [BLOCK_SIZE]).foldl
        (fun acc lane =>
          if active st0 batch_size dim BLOCK_SIZE lane.1 then
            acc.writeMem grad_x_ptr
              (timeOffset st0 batch_size dim BLOCK_SIZE
                (reverseTime length k) lane.1)
              (diagSsmBackwardGradXSpec st0 lambda_ptr grad_y_ptr batch_size
                dim BLOCK_SIZE length (reverseTime length k) lane.1)
          else
            acc)
        stReg) := by
  apply diagSsmBackwardLoopInvariant_step_of_iteration_write st0 stPrev
  · exact hPrev
  · simpa using hGradS
  · simpa using hGradLambda
  · intro idx hOld _hactive
    rw [diagSsmBackwardCurrentIterationScatter_preserve_old st0 stReg
      lambda_ptr grad_x_ptr grad_y_ptr batch_size dim BLOCK_SIZE k idx hk
      hOld hOutInj]
    exact hMem (diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx)
  · intro i hactive
    exact diagSsmBackwardCurrentIterationScatter_write st0 stReg lambda_ptr
      grad_x_ptr grad_y_ptr batch_size dim BLOCK_SIZE k i hactive
      (diagSsmBackwardCurrentIterationNoCollision_of_out_injective st0
        batch_size dim BLOCK_SIZE k i hk hOutInj)

def diagSsmBackwardLoopContextInvariant
    (st0 : BlockState)
    (s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (k : Nat)
    (st : BlockState) : Prop :=
  diagSsmBackwardLoopInvariant st0 s_ptr lambda_ptr y_ptr grad_x_ptr
      grad_y_ptr length batch_size dim BLOCK_SIZE k st ∧
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
        active st0 batch_size dim BLOCK_SIZE idx.1 } ∧
    (∀ offset, st.readMem s_ptr offset = st0.readMem s_ptr offset) ∧
    (∀ offset, st.readMem y_ptr offset = st0.readMem y_ptr offset) ∧
    (∀ offset, st.readMem grad_y_ptr offset = st0.readMem grad_y_ptr offset) ∧
    (∀ offset, st.readMem lambda_ptr offset = st0.readMem lambda_ptr offset)

theorem diagSsmBackwardLoopContextInvariant_of_current_iteration_scatter
    {length : Nat}
    (st0 stPrev stReg : BlockState)
    (s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE k : Nat)
    (hk : k < length)
    (hPrev :
      diagSsmBackwardLoopInvariant st0 s_ptr lambda_ptr y_ptr grad_x_ptr
        grad_y_ptr length batch_size dim BLOCK_SIZE k stPrev)
    (hGradS :
      stReg.regs .real [BLOCK_SIZE] "grad_s" =
        some (diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr batch_size
          dim BLOCK_SIZE length (k + 1)))
    (hGradLambda :
      stReg.regs .real [BLOCK_SIZE] "grad_Lambda" =
        some (diagSsmBackwardGradLambdaTile st0 s_ptr lambda_ptr y_ptr
          grad_y_ptr batch_size dim BLOCK_SIZE length (k + 1)))
    (hLambda :
      stReg.regs .real [BLOCK_SIZE] "Lambda" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          if active st0 batch_size dim BLOCK_SIZE idx.1 then
            some (st0.readMem lambda_ptr
              (IntegralDType.nat.mod (colOffset st0 BLOCK_SIZE idx.1) dim))
          else
            some 0 })
    (hCol :
      stReg.regs .nat [BLOCK_SIZE] "col_offsets" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset st0 BLOCK_SIZE idx.1 })
    (hMask :
      stReg.regs .bool [BLOCK_SIZE] "mask" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active st0 batch_size dim BLOCK_SIZE idx.1 })
    (hMemGradX :
      ∀ offset, stReg.readMem grad_x_ptr offset =
        stPrev.readMem grad_x_ptr offset)
    (hSRead : ∀ offset, stReg.readMem s_ptr offset = st0.readMem s_ptr offset)
    (hYRead : ∀ offset, stReg.readMem y_ptr offset = st0.readMem y_ptr offset)
    (hGradYRead :
      ∀ offset, stReg.readMem grad_y_ptr offset = st0.readMem grad_y_ptr offset)
    (hLambdaRead :
      ∀ offset, stReg.readMem lambda_ptr offset = st0.readMem lambda_ptr offset)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx))
    (hGradXSNe : grad_x_ptr ≠ s_ptr)
    (hGradXYNe : grad_x_ptr ≠ y_ptr)
    (hGradXGradYNe : grad_x_ptr ≠ grad_y_ptr)
    (hGradXLambdaNe : grad_x_ptr ≠ lambda_ptr) :
    diagSsmBackwardLoopContextInvariant st0 s_ptr lambda_ptr y_ptr grad_x_ptr
      grad_y_ptr length batch_size dim BLOCK_SIZE (k + 1)
      ((TileShape.allIndices [BLOCK_SIZE]).foldl
        (fun acc lane =>
          if active st0 batch_size dim BLOCK_SIZE lane.1 then
            acc.writeMem grad_x_ptr
              (timeOffset st0 batch_size dim BLOCK_SIZE
                (reverseTime length k) lane.1)
              (diagSsmBackwardGradXSpec st0 lambda_ptr grad_y_ptr batch_size
                dim BLOCK_SIZE length (reverseTime length k) lane.1)
          else
            acc)
        stReg) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact diagSsmBackwardLoopInvariant_step_of_current_iteration_scatter st0
      stPrev stReg s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr batch_size dim
      BLOCK_SIZE k hk hPrev hGradS hGradLambda hMemGradX hOutInj
  · simpa using hLambda
  · simpa using hCol
  · simpa using hMask
  · intro offset
    rw [diagSsmBackwardCurrentIterationScatter_preserve_region st0 stReg
      lambda_ptr grad_x_ptr grad_y_ptr s_ptr batch_size dim BLOCK_SIZE k
      offset hGradXSNe]
    exact hSRead offset
  · intro offset
    rw [diagSsmBackwardCurrentIterationScatter_preserve_region st0 stReg
      lambda_ptr grad_x_ptr grad_y_ptr y_ptr batch_size dim BLOCK_SIZE k
      offset hGradXYNe]
    exact hYRead offset
  · intro offset
    rw [diagSsmBackwardCurrentIterationScatter_preserve_region st0 stReg
      lambda_ptr grad_x_ptr grad_y_ptr grad_y_ptr batch_size dim BLOCK_SIZE k
      offset hGradXGradYNe]
    exact hGradYRead offset
  · intro offset
    rw [diagSsmBackwardCurrentIterationScatter_preserve_region st0 stReg
      lambda_ptr grad_x_ptr grad_y_ptr lambda_ptr batch_size dim BLOCK_SIZE k
      offset hGradXLambdaNe]
    exact hLambdaRead offset

theorem diagSsmBackwardLoopContextInvariant_init_of_preloop
    (st0 st : BlockState)
    (s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hStep :
      stepStmts (diagSsmBackwardPreLoop lambda_ptr batch_size dim BLOCK_SIZE)
        st0 = some st) :
    diagSsmBackwardLoopContextInvariant st0 s_ptr lambda_ptr y_ptr grad_x_ptr
      grad_y_ptr length batch_size dim BLOCK_SIZE 0 st := by
  rcases diagSsmBackwardPreLoop_step_regs st0 st lambda_ptr batch_size dim
      BLOCK_SIZE hStep with
    ⟨hLambda, hGradS, hGradLambda, hCol, hMask⟩
  refine ⟨diagSsmBackwardLoopInvariant_zero st0 st s_ptr lambda_ptr y_ptr
      grad_x_ptr grad_y_ptr length batch_size dim BLOCK_SIZE hGradS
      hGradLambda, hLambda, hCol, hMask, ?_, ?_, ?_, ?_⟩
  · intro offset
    unfold diagSsmBackwardPreLoop at hStep
    simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.cop,
      NumericDType.mul, NumericDType.add, IntegralDType.mod,
      ComparableDType.lt, Option.bind, colOffset, active] at hStep
    subst st
    rfl
  · intro offset
    unfold diagSsmBackwardPreLoop at hStep
    simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.cop,
      NumericDType.mul, NumericDType.add, IntegralDType.mod,
      ComparableDType.lt, Option.bind, colOffset, active] at hStep
    subst st
    rfl
  · intro offset
    unfold diagSsmBackwardPreLoop at hStep
    simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.cop,
      NumericDType.mul, NumericDType.add, IntegralDType.mod,
      ComparableDType.lt, Option.bind, colOffset, active] at hStep
    subst st
    rfl
  · intro offset
    unfold diagSsmBackwardPreLoop at hStep
    simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.cop,
      NumericDType.mul, NumericDType.add, IntegralDType.mod,
      ComparableDType.lt, Option.bind, colOffset, active] at hStep
    subst st
    rfl

theorem diagSsmBackwardLoopBody_step_preserves_context
    (st0 st st' : BlockState)
    (s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE k : Nat)
    (hLambda :
      st.regs .real [BLOCK_SIZE] "Lambda" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          if active st0 batch_size dim BLOCK_SIZE idx.1 then
            some (st0.readMem lambda_ptr
              (IntegralDType.nat.mod (colOffset st0 BLOCK_SIZE idx.1) dim))
          else
            some 0 })
    (hGradS :
      st.regs .real [BLOCK_SIZE] "grad_s" =
        some (diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr batch_size
          dim BLOCK_SIZE length k))
    (hGradLambda :
      st.regs .real [BLOCK_SIZE] "grad_Lambda" =
        some (diagSsmBackwardGradLambdaTile st0 s_ptr lambda_ptr y_ptr
          grad_y_ptr batch_size dim BLOCK_SIZE length k))
    (hCol :
      st.regs .nat [BLOCK_SIZE] "col_offsets" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset st0 BLOCK_SIZE idx.1 })
    (hMask :
      st.regs .bool [BLOCK_SIZE] "mask" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active st0 batch_size dim BLOCK_SIZE idx.1 })
    (hSRead : ∀ offset, st.readMem s_ptr offset = st0.readMem s_ptr offset)
    (hYRead : ∀ offset, st.readMem y_ptr offset = st0.readMem y_ptr offset)
    (hGradYRead :
      ∀ offset, st.readMem grad_y_ptr offset = st0.readMem grad_y_ptr offset)
    (hLambdaRead :
      ∀ offset, st.readMem lambda_ptr offset = st0.readMem lambda_ptr offset)
    (hGradXSNe : grad_x_ptr ≠ s_ptr)
    (hGradXYNe : grad_x_ptr ≠ y_ptr)
    (hGradXGradYNe : grad_x_ptr ≠ grad_y_ptr)
    (hGradXLambdaNe : grad_x_ptr ≠ lambda_ptr)
    (hStep :
      stepStmts (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr
        length batch_size dim BLOCK_SIZE)
        (st.setReg "i" .nat [] (Tile.scalar k)) = some st') :
    st'.regs .real [BLOCK_SIZE] "Lambda" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          if active st0 batch_size dim BLOCK_SIZE idx.1 then
            some (st0.readMem lambda_ptr
              (IntegralDType.nat.mod (colOffset st0 BLOCK_SIZE idx.1) dim))
          else
            some 0 } ∧
      st'.regs .nat [BLOCK_SIZE] "col_offsets" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset st0 BLOCK_SIZE idx.1 } ∧
      st'.regs .bool [BLOCK_SIZE] "mask" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active st0 batch_size dim BLOCK_SIZE idx.1 } ∧
      (∀ offset, st'.readMem s_ptr offset = st0.readMem s_ptr offset) ∧
      (∀ offset, st'.readMem y_ptr offset = st0.readMem y_ptr offset) ∧
      (∀ offset, st'.readMem grad_y_ptr offset = st0.readMem grad_y_ptr offset) ∧
      (∀ offset, st'.readMem lambda_ptr offset = st0.readMem lambda_ptr offset) := by
  unfold diagSsmBackwardLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hLambda, hGradS, hGradLambda, hCol,
    hMask, hSRead, hYRead, hGradYRead, hLambdaRead, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, IntegralDType.mod,
    ComparableDType.gt, Option.bind, timeOffset, active, colOffset,
    diagSsmBackwardGradSTile_succ, diagSsmBackwardGradLambdaTile_succ,
    diagSsmBackwardPrevState_of_prev, diagSsmBackwardPrevState_of_last] at hStep
  by_cases hkprev : k < length - 1
  <;> simp [hkprev] at hStep
  <;> rw [hGradS] at hStep
  <;> simp [BlockState.setReg, hGradLambda, hLambda, hCol, hMask, hkprev] at hStep
  <;> symm at hStep
  <;> subst st'
  <;> refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  all_goals
    try simp [BlockState.setReg, hLambda, hCol, hMask]
  all_goals
    intro offset
    first
    | trans st.readMem s_ptr offset
      · simpa [BlockState.setReg, timeOffset, active, colOffset] using
          diagSsmBackwardCurrentIterationScatter_preserve_region_of_value
            (length := length) st0 _ grad_x_ptr s_ptr batch_size dim BLOCK_SIZE
            k offset
            (fun lane : TileIndex [BLOCK_SIZE] =>
              let valueTile : Tile .real [BLOCK_SIZE] :=
                { data := fun idx : TileIndex [BLOCK_SIZE] =>
                    Option.map₂ (fun x1 x2 => x1 + x2)
                      (if active st0 batch_size dim BLOCK_SIZE idx.1 then
                        some (st0.readMem grad_y_ptr
                          (timeOffset st0 batch_size dim BLOCK_SIZE
                            (reverseTime length k) idx.1))
                      else
                        some 0)
                      ((diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr
                        batch_size dim BLOCK_SIZE length k).data idx) }
              WithBot.unbotD 0 (valueTile.data lane))
            hGradXSNe
      · exact hSRead offset
    | trans st.readMem y_ptr offset
      · simpa [BlockState.setReg, timeOffset, active, colOffset] using
          diagSsmBackwardCurrentIterationScatter_preserve_region_of_value
            (length := length) st0 _ grad_x_ptr y_ptr batch_size dim BLOCK_SIZE
            k offset
            (fun lane : TileIndex [BLOCK_SIZE] =>
              let valueTile : Tile .real [BLOCK_SIZE] :=
                { data := fun idx : TileIndex [BLOCK_SIZE] =>
                    Option.map₂ (fun x1 x2 => x1 + x2)
                      (if active st0 batch_size dim BLOCK_SIZE idx.1 then
                        some (st0.readMem grad_y_ptr
                          (timeOffset st0 batch_size dim BLOCK_SIZE
                            (reverseTime length k) idx.1))
                      else
                        some 0)
                      ((diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr
                        batch_size dim BLOCK_SIZE length k).data idx) }
              WithBot.unbotD 0 (valueTile.data lane))
            hGradXYNe
      · exact hYRead offset
    | trans st.readMem grad_y_ptr offset
      · simpa [BlockState.setReg, timeOffset, active, colOffset] using
          diagSsmBackwardCurrentIterationScatter_preserve_region_of_value
            (length := length) st0 _ grad_x_ptr grad_y_ptr batch_size dim
            BLOCK_SIZE k offset
            (fun lane : TileIndex [BLOCK_SIZE] =>
              let valueTile : Tile .real [BLOCK_SIZE] :=
                { data := fun idx : TileIndex [BLOCK_SIZE] =>
                    Option.map₂ (fun x1 x2 => x1 + x2)
                      (if active st0 batch_size dim BLOCK_SIZE idx.1 then
                        some (st0.readMem grad_y_ptr
                          (timeOffset st0 batch_size dim BLOCK_SIZE
                            (reverseTime length k) idx.1))
                      else
                        some 0)
                      ((diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr
                        batch_size dim BLOCK_SIZE length k).data idx) }
              WithBot.unbotD 0 (valueTile.data lane))
            hGradXGradYNe
      · exact hGradYRead offset
    | trans st.readMem lambda_ptr offset
      · simpa [BlockState.setReg, timeOffset, active, colOffset] using
          diagSsmBackwardCurrentIterationScatter_preserve_region_of_value
            (length := length) st0 _ grad_x_ptr lambda_ptr batch_size dim
            BLOCK_SIZE k offset
            (fun lane : TileIndex [BLOCK_SIZE] =>
              let valueTile : Tile .real [BLOCK_SIZE] :=
                { data := fun idx : TileIndex [BLOCK_SIZE] =>
                    Option.map₂ (fun x1 x2 => x1 + x2)
                      (if active st0 batch_size dim BLOCK_SIZE idx.1 then
                        some (st0.readMem grad_y_ptr
                          (timeOffset st0 batch_size dim BLOCK_SIZE
                            (reverseTime length k) idx.1))
                      else
                        some 0)
                      ((diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr
                        batch_size dim BLOCK_SIZE length k).data idx) }
              WithBot.unbotD 0 (valueTile.data lane))
            hGradXLambdaNe
      · exact hLambdaRead offset

theorem diagSsmBackwardLoopBody_step_regs
    {length : Nat}
    (st0 st st' : BlockState)
    (s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE k : Nat)
    (hk : k < length)
    (hGradS :
      st.regs .real [BLOCK_SIZE] "grad_s" =
        some (diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr batch_size
          dim BLOCK_SIZE length k))
    (hGradLambda :
      st.regs .real [BLOCK_SIZE] "grad_Lambda" =
        some (diagSsmBackwardGradLambdaTile st0 s_ptr lambda_ptr y_ptr
          grad_y_ptr batch_size dim BLOCK_SIZE length k))
    (hLambda :
      st.regs .real [BLOCK_SIZE] "Lambda" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          if active st0 batch_size dim BLOCK_SIZE idx.1 then
            some (st0.readMem lambda_ptr
              (IntegralDType.nat.mod (colOffset st0 BLOCK_SIZE idx.1) dim))
          else
            some 0 })
    (hCol :
      st.regs .nat [BLOCK_SIZE] "col_offsets" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset st0 BLOCK_SIZE idx.1 })
    (hMask :
      st.regs .bool [BLOCK_SIZE] "mask" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active st0 batch_size dim BLOCK_SIZE idx.1 })
    (hGradYRead :
      ∀ offset, st.readMem grad_y_ptr offset = st0.readMem grad_y_ptr offset)
    (hSRead : ∀ offset, st.readMem s_ptr offset = st0.readMem s_ptr offset)
    (hYRead : ∀ offset, st.readMem y_ptr offset = st0.readMem y_ptr offset)
    (hStep :
      stepStmts (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr
        length batch_size dim BLOCK_SIZE)
        (st.setReg "i" .nat [] (Tile.scalar k)) = some st') :
    st'.regs .real [BLOCK_SIZE] "grad_s" =
        some (diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr batch_size
          dim BLOCK_SIZE length (k + 1)) ∧
      st'.regs .real [BLOCK_SIZE] "grad_Lambda" =
        some (diagSsmBackwardGradLambdaTile st0 s_ptr lambda_ptr y_ptr
          grad_y_ptr batch_size dim BLOCK_SIZE length (k + 1)) := by
  unfold diagSsmBackwardLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hGradS, hGradLambda, hLambda, hCol,
    hMask, hGradYRead, hSRead, hYRead, Tile.bop, Tile.cop, NumericDType.add,
    NumericDType.mul, NumericDType.sub, IntegralDType.mod, ComparableDType.gt,
    Option.bind, timeOffset, active, colOffset, diagSsmBackwardGradSTile_succ,
    diagSsmBackwardGradLambdaTile_succ, diagSsmBackwardPrevState_of_prev,
    diagSsmBackwardPrevState_of_last] at hStep
  by_cases hkprev : k < length - 1
  <;> simp [hkprev] at hStep
  <;> rw [hGradS] at hStep
  <;> simp [BlockState.setReg, hGradLambda, hLambda, hCol, hMask, hkprev] at hStep
  <;> symm at hStep
  <;> subst st'
  <;> constructor
  all_goals
    simp [BlockState.foldl_writeMem_prop_masked_regs, BlockState.setReg]
  all_goals
    ext idx
    by_cases hactive : active st0 batch_size dim BLOCK_SIZE idx.1
    · have hlt :
          st0.pids 0 * BLOCK_SIZE + idx.1.val < batch_size * dim := by
        simpa [active, colOffset] using hactive
      simp [diagSsmBackwardGradSTile_succ, diagSsmBackwardGradSTile,
        diagSsmBackwardGradLambdaTile_succ, diagSsmBackwardGradLambdaTile,
        diagSsmBackwardGradSAfter, diagSsmBackwardGradLambdaAfter, hGradLambda,
        hLambda, hMask, hactive, hlt, active, colOffset, timeOffset,
        IntegralDType.mod, diagSsmBackwardPrevState_of_prev,
        diagSsmBackwardPrevState_of_last, hk, hkprev, reverseTime]
    · have hlt :
          ¬ st0.pids 0 * BLOCK_SIZE + idx.1.val < batch_size * dim := by
        simpa [active, colOffset] using hactive
      simp [diagSsmBackwardGradSTile_succ, diagSsmBackwardGradSTile,
        diagSsmBackwardGradLambdaTile_succ, diagSsmBackwardGradLambdaTile,
        hGradLambda, hLambda, hMask, hactive, hlt, active, colOffset,
        timeOffset, IntegralDType.mod, diagSsmBackwardPrevState_of_prev,
        diagSsmBackwardPrevState_of_last, hk, hkprev]
      try norm_num

theorem diagSsmBackwardLoopInvariant_step_of_concrete_body
    {length : Nat}
    (st0 st st' : BlockState)
    (s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE k : Nat)
    (hk : k < length)
    (hPrev :
      diagSsmBackwardLoopInvariant st0 s_ptr lambda_ptr y_ptr grad_x_ptr
        grad_y_ptr length batch_size dim BLOCK_SIZE k st)
    (hLambda :
      st.regs .real [BLOCK_SIZE] "Lambda" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          if active st0 batch_size dim BLOCK_SIZE idx.1 then
            some (st0.readMem lambda_ptr
              (IntegralDType.nat.mod (colOffset st0 BLOCK_SIZE idx.1) dim))
          else
            some 0 })
    (hCol :
      st.regs .nat [BLOCK_SIZE] "col_offsets" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset st0 BLOCK_SIZE idx.1 })
    (hMask :
      st.regs .bool [BLOCK_SIZE] "mask" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active st0 batch_size dim BLOCK_SIZE idx.1 })
    (hGradYRead :
      ∀ offset, st.readMem grad_y_ptr offset = st0.readMem grad_y_ptr offset)
    (hSRead : ∀ offset, st.readMem s_ptr offset = st0.readMem s_ptr offset)
    (hYRead : ∀ offset, st.readMem y_ptr offset = st0.readMem y_ptr offset)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx))
    (hStep :
      stepStmts (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr
        length batch_size dim BLOCK_SIZE)
        (st.setReg "i" .nat [] (Tile.scalar k)) = some st') :
    diagSsmBackwardLoopInvariant st0 s_ptr lambda_ptr y_ptr grad_x_ptr
      grad_y_ptr length batch_size dim BLOCK_SIZE (k + 1) st' := by
  have hRegs :=
    diagSsmBackwardLoopBody_step_regs st0 st st' s_ptr lambda_ptr y_ptr
      grad_x_ptr grad_y_ptr batch_size dim BLOCK_SIZE k hk hPrev.1
      hPrev.2.1 hLambda hCol hMask hGradYRead hSRead hYRead hStep
  apply diagSsmBackwardLoopInvariant_step_of_iteration_write st0 st
  · exact hPrev
  · exact hRegs.1
  · exact hRegs.2
  · intro idx hOld _hactive
    unfold diagSsmBackwardLoopBody at hStep
    simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hPrev.1, hPrev.2.1, hLambda, hCol,
      hMask, hGradYRead, hSRead, hYRead, Tile.bop, Tile.cop,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      IntegralDType.mod, ComparableDType.gt, Option.bind, timeOffset, active,
      colOffset, diagSsmBackwardGradSTile_succ,
      diagSsmBackwardGradLambdaTile_succ, diagSsmBackwardPrevState_of_prev,
      diagSsmBackwardPrevState_of_last] at hStep
    by_cases hkprev : k < length - 1
    <;> simp [hkprev] at hStep
    <;> rw [hPrev.1] at hStep
    <;> simp [BlockState.setReg, hPrev.2.1, hLambda, hCol, hMask,
      hkprev] at hStep
    <;> symm at hStep
    <;> subst st'
    all_goals
      trans st.readMem grad_x_ptr
          (diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx)
      · simpa [timeOffset, reverseTime, diagSsmBackwardGradXOffset, active,
          colOffset] using
          BlockState.scatter_prop_masked_preserves_other_offset
            grad_x_ptr
            (fun lane : TileIndex [BLOCK_SIZE] =>
              timeOffset st0 batch_size dim BLOCK_SIZE (reverseTime length k)
                lane.1)
            (fun lane : TileIndex [BLOCK_SIZE] =>
              WithBot.unbotD 0
                (Option.map₂ (fun x1 x2 => x1 + x2)
                  (if active st0 batch_size dim BLOCK_SIZE lane.1 then
                    some (st0.readMem grad_y_ptr
                      (timeOffset st0 batch_size dim BLOCK_SIZE
                        (reverseTime length k) lane.1))
                  else
                    some 0)
                  ((diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr
                    batch_size dim BLOCK_SIZE length k).data
                      (lane.1, PUnit.unit))))
            (fun lane : TileIndex [BLOCK_SIZE] =>
              active st0 batch_size dim BLOCK_SIZE lane.1)
            (diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx)
            (fun lane _hactive heq =>
              diagSsmBackwardGradXOffset_ne_currentIteration st0 batch_size dim
                BLOCK_SIZE k idx lane.1 hOutInj hk hOld heq.symm)
            (TileShape.allIndices [BLOCK_SIZE]) _
      · rfl
  · intro i hactive
    unfold diagSsmBackwardLoopBody at hStep
    simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hPrev.1, hPrev.2.1, hLambda, hCol,
      hMask, hGradYRead, hSRead, hYRead, Tile.bop, Tile.cop,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      IntegralDType.mod, ComparableDType.gt, Option.bind, timeOffset, active,
      colOffset, diagSsmBackwardGradSTile_succ,
      diagSsmBackwardGradLambdaTile_succ, diagSsmBackwardPrevState_of_prev,
      diagSsmBackwardPrevState_of_last] at hStep
    by_cases hkprev : k < length - 1
    <;> simp [hkprev] at hStep
    <;> rw [hPrev.1] at hStep
    <;> simp [BlockState.setReg, hPrev.2.1, hLambda, hCol, hMask,
      hkprev] at hStep
    <;> symm at hStep
    <;> subst st'
    all_goals
      have hNoCollision :=
        diagSsmBackwardCurrentIterationNoCollision_of_out_injective st0
          batch_size dim BLOCK_SIZE k i hk hOutInj
      simpa [diagSsmBackwardGradXSpec_reverseTime st0 lambda_ptr grad_y_ptr
        batch_size dim BLOCK_SIZE length k i hk, diagSsmBackwardGradSTile,
        hactive] using
        (BlockState.scatter_readback_prop_masked_nd_of_true
          (region := grad_x_ptr)
          (s := _)
          (offsetFn := fun lane : TileIndex [BLOCK_SIZE] =>
            timeOffset st0 batch_size dim BLOCK_SIZE (reverseTime length k)
              lane.1)
          (valueFn := fun lane : TileIndex [BLOCK_SIZE] =>
            WithBot.unbotD 0
              (Option.map₂ (fun x1 x2 => x1 + x2)
                (if active st0 batch_size dim BLOCK_SIZE lane.1 then
                  some (st0.readMem grad_y_ptr
                    (timeOffset st0 batch_size dim BLOCK_SIZE
                      (reverseTime length k) lane.1))
                else
                  some 0)
                ((diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr
                  batch_size dim BLOCK_SIZE length k).data
                    (lane.1, PUnit.unit))))
          (P := fun lane : TileIndex [BLOCK_SIZE] =>
            active st0 batch_size dim BLOCK_SIZE lane.1)
          ((i, PUnit.unit) : TileIndex [BLOCK_SIZE])
          hactive
          (fun lane hlane heq => hNoCollision lane hlane heq))

theorem diagSsmBackwardLoopContextInvariant_step_of_body
    {length : Nat}
    (st0 st st' : BlockState)
    (s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE k : Nat)
    (hk : k < length)
    (hCtx :
      diagSsmBackwardLoopContextInvariant st0 s_ptr lambda_ptr y_ptr
        grad_x_ptr grad_y_ptr length batch_size dim BLOCK_SIZE k st)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx))
    (hGradXSNe : grad_x_ptr ≠ s_ptr)
    (hGradXYNe : grad_x_ptr ≠ y_ptr)
    (hGradXGradYNe : grad_x_ptr ≠ grad_y_ptr)
    (hGradXLambdaNe : grad_x_ptr ≠ lambda_ptr)
    (hStep :
      stepStmts (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr
        length batch_size dim BLOCK_SIZE)
        (st.setReg "i" .nat [] (Tile.scalar k)) = some st') :
    diagSsmBackwardLoopContextInvariant st0 s_ptr lambda_ptr y_ptr grad_x_ptr
      grad_y_ptr length batch_size dim BLOCK_SIZE (k + 1) st' := by
  rcases hCtx with
    ⟨hInv, hLambda, hCol, hMask, hSRead, hYRead, hGradYRead, hLambdaRead⟩
  have hInv' :
      diagSsmBackwardLoopInvariant st0 s_ptr lambda_ptr y_ptr grad_x_ptr
        grad_y_ptr length batch_size dim BLOCK_SIZE (k + 1) st' :=
    diagSsmBackwardLoopInvariant_step_of_concrete_body st0 st st' s_ptr
      lambda_ptr y_ptr grad_x_ptr grad_y_ptr batch_size dim BLOCK_SIZE k hk
      hInv hLambda hCol hMask hGradYRead hSRead hYRead hOutInj hStep
  rcases diagSsmBackwardLoopBody_step_preserves_context st0 st st' s_ptr
      lambda_ptr y_ptr grad_x_ptr grad_y_ptr length batch_size dim BLOCK_SIZE k
      hLambda hInv.1 hInv.2.1 hCol hMask hSRead hYRead hGradYRead
      hLambdaRead hGradXSNe hGradXYNe hGradXGradYNe hGradXLambdaNe hStep with
    ⟨hLambda', hCol', hMask', hSRead', hYRead', hGradYRead',
      hLambdaRead'⟩
  exact
    ⟨hInv', hLambda', hCol', hMask', hSRead', hYRead', hGradYRead',
      hLambdaRead'⟩

theorem diagSsmBackwardLoopContextInvariant_body_step_exists
    {length : Nat}
    (st0 st : BlockState)
    (s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE k : Nat)
    (hk : k < length)
    (hCtx :
      diagSsmBackwardLoopContextInvariant st0 s_ptr lambda_ptr y_ptr
        grad_x_ptr grad_y_ptr length batch_size dim BLOCK_SIZE k st)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx))
    (hGradXSNe : grad_x_ptr ≠ s_ptr)
    (hGradXYNe : grad_x_ptr ≠ y_ptr)
    (hGradXGradYNe : grad_x_ptr ≠ grad_y_ptr)
    (hGradXLambdaNe : grad_x_ptr ≠ lambda_ptr) :
    ∃ st',
      stepStmts (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr
        length batch_size dim BLOCK_SIZE)
        (st.setReg "i" .nat [] (Tile.scalar k)) = some st' ∧
      diagSsmBackwardLoopContextInvariant st0 s_ptr lambda_ptr y_ptr
        grad_x_ptr grad_y_ptr length batch_size dim BLOCK_SIZE (k + 1) st' := by
  rcases hCtx with
    ⟨hInv, hLambda, hCol, hMask, hSRead, hYRead, hGradYRead, hLambdaRead⟩
  cases hStep :
      stepStmts (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr
        length batch_size dim BLOCK_SIZE)
        (st.setReg "i" .nat [] (Tile.scalar k)) with
  | none =>
      unfold diagSsmBackwardLoopBody at hStep
      simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hInv.1, hInv.2.1, hLambda, hCol,
        hMask, hGradYRead, hSRead, hYRead, Tile.bop, Tile.cop,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        IntegralDType.mod, ComparableDType.gt, Option.bind, timeOffset, active,
        colOffset, diagSsmBackwardGradSTile_succ,
        diagSsmBackwardGradLambdaTile_succ, diagSsmBackwardPrevState_of_prev,
        diagSsmBackwardPrevState_of_last] at hStep
      by_cases hkprev : k < length - 1
      <;> simp [hkprev] at hStep
      <;> rw [hInv.1] at hStep
      <;> simp [BlockState.setReg, hInv.2.1, hLambda, hCol, hMask,
        hkprev] at hStep
  | some st' =>
      refine ⟨st', rfl, ?_⟩
      exact diagSsmBackwardLoopContextInvariant_step_of_body st0 st st' s_ptr
        lambda_ptr y_ptr grad_x_ptr grad_y_ptr batch_size dim BLOCK_SIZE k hk
        ⟨hInv, hLambda, hCol, hMask, hSRead, hYRead, hGradYRead,
          hLambdaRead⟩ hOutInj hGradXSNe hGradXYNe hGradXGradYNe
        hGradXLambdaNe hStep

theorem diagSsmBackwardForLoop_context_of_preloop
    (st0 stPre stLoop : BlockState)
    (s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmBackwardGradXOffset st0 batch_size dim BLOCK_SIZE idx))
    (hGradXSNe : grad_x_ptr ≠ s_ptr)
    (hGradXYNe : grad_x_ptr ≠ y_ptr)
    (hGradXGradYNe : grad_x_ptr ≠ grad_y_ptr)
    (hGradXLambdaNe : grad_x_ptr ≠ lambda_ptr)
    (hPre :
      stepStmts (diagSsmBackwardPreLoop lambda_ptr batch_size dim BLOCK_SIZE)
        st0 = some stPre)
    (hLoop :
      stepStmt (.forRange "i" 0 length 1
        (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr length
          batch_size dim BLOCK_SIZE)) stPre = some stLoop) :
    diagSsmBackwardLoopContextInvariant st0 s_ptr lambda_ptr y_ptr grad_x_ptr
      grad_y_ptr length batch_size dim BLOCK_SIZE length stLoop := by
  have hInit :
      diagSsmBackwardLoopContextInvariant st0 s_ptr lambda_ptr y_ptr
        grad_x_ptr grad_y_ptr length batch_size dim BLOCK_SIZE 0 stPre :=
    diagSsmBackwardLoopContextInvariant_init_of_preloop st0 stPre s_ptr
      lambda_ptr y_ptr grad_x_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
      hPre
  let P : Nat → BlockState → Prop := fun k st =>
    if k < length then
      diagSsmBackwardLoopContextInvariant st0 s_ptr lambda_ptr y_ptr
        grad_x_ptr grad_y_ptr length batch_size dim BLOCK_SIZE k st
    else
      diagSsmBackwardLoopContextInvariant st0 s_ptr lambda_ptr y_ptr
        grad_x_ptr grad_y_ptr length batch_size dim BLOCK_SIZE length st
  have hInitP : P 0 stPre := by
    by_cases hlen : 0 < length
    · simpa [P, hlen] using hInit
    · have hzero : length = 0 := by omega
      simpa [P, hlen, hzero] using hInit
  obtain ⟨final, stFinal, hFor, hFinal, hCtx⟩ :=
    forRange_inv
      (idx := "i")
      (start := 0) (stop := length) (step := 1)
      (body := diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr
        length batch_size dim BLOCK_SIZE)
      (P := P)
      (s_init := stPre)
      (by norm_num)
      hInitP
      (by
        intro k st hk hCtx
        have hCtxK :
            diagSsmBackwardLoopContextInvariant st0 s_ptr lambda_ptr y_ptr
              grad_x_ptr grad_y_ptr length batch_size dim BLOCK_SIZE k st := by
          simpa [P, hk] using hCtx
        obtain ⟨st', hStep, hCtx'⟩ :=
          diagSsmBackwardLoopContextInvariant_body_step_exists st0 st s_ptr
            lambda_ptr y_ptr grad_x_ptr grad_y_ptr batch_size dim BLOCK_SIZE k
            hk hCtxK hOutInj hGradXSNe hGradXYNe hGradXGradYNe hGradXLambdaNe
        refine ⟨st', hStep, ?_⟩
        by_cases hnext : k + 1 < length
        · simpa [P, hnext] using hCtx'
        · have hlast : k + 1 = length := by omega
          simpa [P, hnext, hlast] using hCtx')
  have hEq : stFinal = stLoop := by
    rw [hLoop] at hFor
    injection hFor with h
    exact h.symm
  subst hEq
  have hnot : ¬ final < length := by omega
  simpa [P, hnot] using hCtx

theorem diagSsmBackwardProjectedBody_alg_post
    (s s' : BlockState)
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmBackwardGradXOffset s batch_size dim BLOCK_SIZE idx))
    (hGradXSNe : grad_x_ptr ≠ s_ptr)
    (hGradXYNe : grad_x_ptr ≠ y_ptr)
    (hGradXGradYNe : grad_x_ptr ≠ grad_y_ptr)
    (hGradXLambdaNe : grad_x_ptr ≠ lambda_ptr)
    (hGradSGradXNe : grad_s_ptr ≠ grad_x_ptr)
    (hGradLambdaGradXNe : grad_lambda_ptr ≠ grad_x_ptr)
    (hGradLambdaGradSNe : grad_lambda_ptr ≠ grad_s_ptr)
    (hExec :
      stepStmts (diagSsmBackwardProjectedBody s_ptr lambda_ptr y_ptr
        grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim
        BLOCK_SIZE) s = some s') :
    diag_ssm_backward_kernel_alg_post s_ptr lambda_ptr y_ptr grad_s_ptr
      grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
      s s' := by
  unfold diagSsmBackwardProjectedBody at hExec
  rcases (stepStmts.append_some_iff).mp hExec with
    ⟨stLoop, hPreLoop, hPost⟩
  rcases (stepStmts.append_some_iff).mp hPreLoop with
    ⟨stPre, hPre, hLoopStmts⟩
  have hLoop :
      stepStmt (.forRange "i" 0 length 1
        (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr length
          batch_size dim BLOCK_SIZE)) stPre = some stLoop := by
    cases hAux :
        stepForRangeAux "i" 0 length 1
          (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr length
            batch_size dim BLOCK_SIZE) stPre with
    | none =>
        simp [stepStmts, hAux] at hLoopStmts
    | some st =>
        simp [stepStmts, hAux] at hLoopStmts
        subst hLoopStmts
        simp [stepForRangeAux.forRange_unfold, hAux]
  have hCtx :=
    diagSsmBackwardForLoop_context_of_preloop s stPre stLoop s_ptr
      lambda_ptr y_ptr grad_x_ptr grad_y_ptr length batch_size dim
      BLOCK_SIZE hOutInj hGradXSNe hGradXYNe hGradXGradYNe hGradXLambdaNe
      hPre hLoop
  rcases hCtx with
    ⟨hInv, _hLambda, hCol, hMask, _hSRead, _hYRead, _hGradYRead,
      _hLambdaRead⟩
  exact diagSsmBackwardLoopInvariant_postLoop_alg_post s stLoop s' s_ptr
    lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr
    length batch_size dim BLOCK_SIZE hInv hCol hMask hGradSGradXNe
    hGradLambdaGradXNe hGradLambdaGradSNe hPost

theorem diag_ssm_backward_kernel_alg_post_of_exec
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmBackwardGradXOffset s batch_size dim BLOCK_SIZE idx))
    (hGradXSNe : grad_x_ptr ≠ s_ptr)
    (hGradXYNe : grad_x_ptr ≠ y_ptr)
    (hGradXGradYNe : grad_x_ptr ≠ grad_y_ptr)
    (hGradXLambdaNe : grad_x_ptr ≠ lambda_ptr)
    (hGradSGradXNe : grad_s_ptr ≠ grad_x_ptr)
    (hGradLambdaGradXNe : grad_lambda_ptr ≠ grad_x_ptr)
    (hGradLambdaGradSNe : grad_lambda_ptr ≠ grad_s_ptr)
    (hExec :
      exec (diag_ssm_backward_kernel s_ptr lambda_ptr y_ptr grad_s_ptr
        grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim
        BLOCK_SIZE) s = some s') :
    diag_ssm_backward_kernel_alg_post s_ptr lambda_ptr y_ptr grad_s_ptr
      grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
      s s' := by
  unfold exec at hExec
  rw [diag_ssm_backward_kernel_toAlg_body s_ptr lambda_ptr y_ptr grad_s_ptr
    grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE] at hExec
  exact diagSsmBackwardProjectedBody_alg_post s s' s_ptr lambda_ptr y_ptr
    grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim
    BLOCK_SIZE hOutInj hGradXSNe hGradXYNe hGradXGradYNe hGradXLambdaNe
    hGradSGradXNe hGradLambdaGradXNe hGradLambdaGradSNe hExec

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
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.cop,
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

theorem diagSsmForwardLoopInvariant_step_of_concrete_body
    {length : Nat}
    (st0 st st' : BlockState)
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE t : Nat)
    (ht : t < length)
    (hPrev :
      diagSsmForwardLoopInvariant st0 s_ptr x_ptr lambda_ptr y_ptr length
        batch_size dim BLOCK_SIZE t st)
    (hS :
      st.regs .real [BLOCK_SIZE] "s" =
        some (diagSsmMaskedStateTile st0 s_ptr x_ptr lambda_ptr batch_size dim
          BLOCK_SIZE t))
    (hLambda :
      st.regs .real [BLOCK_SIZE] "Lambda" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          if active st0 batch_size dim BLOCK_SIZE idx.1 then
            some (st0.readMem lambda_ptr
              (IntegralDType.nat.mod (colOffset st0 BLOCK_SIZE idx.1) dim))
          else
            some 0 })
    (hCol :
      st.regs .nat [BLOCK_SIZE] "col_offsets" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset st0 BLOCK_SIZE idx.1 })
    (hMask :
      st.regs .bool [BLOCK_SIZE] "mask" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active st0 batch_size dim BLOCK_SIZE idx.1 })
    (hXRead : ∀ offset, st.readMem x_ptr offset = st0.readMem x_ptr offset)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset st0 batch_size dim BLOCK_SIZE idx))
    (hStep :
      stepStmts (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim BLOCK_SIZE)
        (st.setReg "t" .nat [] (Tile.scalar t)) = some st') :
    diagSsmForwardLoopInvariant st0 s_ptr x_ptr lambda_ptr y_ptr length
      batch_size dim BLOCK_SIZE (t + 1) st' := by
  unfold diagSsmForwardLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hS, hLambda, hCol, hMask, hXRead,
    Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul, Option.bind,
    timeOffset, active, colOffset, IntegralDType.mod,
    diagSsmMaskedStateTile_succ] at hStep
  subst st'
  let actualS : Tile .real [BLOCK_SIZE] :=
    { data := fun i : TileIndex [BLOCK_SIZE] =>
        Option.map₂ (fun x1 x2 => x1 + x2)
          (Option.map₂ (fun x1 x2 => x1 * x2)
            ((diagSsmMaskedStateTile st0 s_ptr x_ptr lambda_ptr batch_size dim
              BLOCK_SIZE t).data (i.1, PUnit.unit))
            (if active st0 batch_size dim BLOCK_SIZE i.1 then
              some (st0.readMem lambda_ptr
                (IntegralDType.nat.mod (colOffset st0 BLOCK_SIZE i.1) dim))
            else
              some 0))
          (if active st0 batch_size dim BLOCK_SIZE i.1 then
            some (st0.readMem x_ptr
              (timeOffset st0 batch_size dim BLOCK_SIZE t i.1))
          else
            some 0) }
  have hActualS :
      actualS =
        diagSsmMaskedStateTile st0 s_ptr x_ptr lambda_ptr batch_size dim
          BLOCK_SIZE (t + 1) := by
    ext i
    by_cases hactive : active st0 batch_size dim BLOCK_SIZE i.1
    · have hlt : st0.pids 0 * BLOCK_SIZE + i.1.val < batch_size * dim := by
        simpa [active, colOffset] using hactive
      simp [actualS, diagSsmMaskedStateTile, diagSsmStateAfter, hactive, hlt,
        active, colOffset, timeOffset, IntegralDType.mod]
    · have hlt : ¬ st0.pids 0 * BLOCK_SIZE + i.1.val < batch_size * dim := by
        simpa [active, colOffset] using hactive
      simp [actualS, diagSsmMaskedStateTile, diagSsmStateAfter, hactive, hlt,
        active, colOffset, timeOffset, IntegralDType.mod]
      norm_num
  let stReg : BlockState :=
    ((((st.setReg "t" TileDType.nat [] (Tile.scalar t)).setReg
          "offsets" TileDType.nat [BLOCK_SIZE]
          { data := fun i : TileIndex [BLOCK_SIZE] =>
              timeOffset st0 batch_size dim BLOCK_SIZE t i.1 }).setReg
        "x" TileDType.real [BLOCK_SIZE]
          { data := fun i : TileIndex [BLOCK_SIZE] =>
              if active st0 batch_size dim BLOCK_SIZE i.1 then
                some (st0.readMem x_ptr
                  (timeOffset st0 batch_size dim BLOCK_SIZE t i.1))
              else
                some 0 }).setReg
        "s" TileDType.real [BLOCK_SIZE] actualS)
  have hReg :
      stReg.regs .real [BLOCK_SIZE] "s" =
        some (diagSsmMaskedStateTile st0 s_ptr x_ptr lambda_ptr batch_size dim
          BLOCK_SIZE (t + 1)) := by
    simp [stReg, BlockState.setReg, hActualS]
  have hMem : ∀ offset, stReg.readMem y_ptr offset = st.readMem y_ptr offset := by
    intro offset
    rfl
  have hAbs :=
    diagSsmForwardLoopInvariant_step_of_current_time_scatter st0 st stReg
      s_ptr x_ptr lambda_ptr y_ptr batch_size dim BLOCK_SIZE t ht hPrev
      hReg hMem hOutInj
  have hActualS_active :
      ∀ i : Fin BLOCK_SIZE,
        active st0 batch_size dim BLOCK_SIZE i →
          WithBot.unbotD 0 (actualS.data (i, PUnit.unit)) =
            diagSsmForwardSpec st0 s_ptr x_ptr lambda_ptr batch_size dim
              BLOCK_SIZE t i := by
    intro i hi
    rw [hActualS]
    simp [diagSsmMaskedStateTile, diagSsmForwardSpec, hi]
  convert hAbs using 6
  ; try simp [stReg, timeOffset, active, colOffset]
  all_goals first
  | change WithBot.unbotD 0 (actualS.data (_, PUnit.unit)) = _
    apply hActualS_active
    assumption

def diagSsmForwardLoopContextInvariant
    (st0 : BlockState) (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (t : Nat) (st : BlockState) : Prop :=
  diagSsmForwardLoopInvariant st0 s_ptr x_ptr lambda_ptr y_ptr length
      batch_size dim BLOCK_SIZE t st ∧
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
        active st0 batch_size dim BLOCK_SIZE idx.1 } ∧
    ∀ offset, st.readMem x_ptr offset = st0.readMem x_ptr offset

theorem diagSsmForwardLoopContextInvariant_init_of_preloop
    (st0 st : BlockState) (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hStep :
      stepStmts (diagSsmForwardPreLoop s_ptr lambda_ptr batch_size dim
        BLOCK_SIZE) st0 = some st) :
    diagSsmForwardLoopContextInvariant st0 s_ptr x_ptr lambda_ptr y_ptr length
      batch_size dim BLOCK_SIZE 0 st := by
  rcases diagSsmForwardPreLoop_step_regs st0 st s_ptr x_ptr lambda_ptr
      batch_size dim BLOCK_SIZE hStep with
    ⟨hS, hLambda, hCol, hMask⟩
  refine ⟨diagSsmForwardLoopInvariant_zero st0 st s_ptr x_ptr lambda_ptr y_ptr
      length batch_size dim BLOCK_SIZE hS, hLambda, hCol, hMask, ?_⟩
  intro offset
  unfold diagSsmForwardPreLoop at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.cop,
    NumericDType.mul, NumericDType.add, IntegralDType.mod,
    ComparableDType.lt, Option.bind, colOffset, active,
    diagSsmMaskedStateTile, diagSsmStateAfter] at hStep
  subst st
  rfl

theorem diagSsmForwardLoopBody_step_preserves_context
    (st0 st st' : BlockState) (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE t : Nat)
    (hS :
      st.regs .real [BLOCK_SIZE] "s" =
        some (diagSsmMaskedStateTile st0 s_ptr x_ptr lambda_ptr batch_size dim
          BLOCK_SIZE t))
    (hLambda :
      st.regs .real [BLOCK_SIZE] "Lambda" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          if active st0 batch_size dim BLOCK_SIZE idx.1 then
            some (st0.readMem lambda_ptr
              (IntegralDType.nat.mod (colOffset st0 BLOCK_SIZE idx.1) dim))
          else
            some 0 })
    (hCol :
      st.regs .nat [BLOCK_SIZE] "col_offsets" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset st0 BLOCK_SIZE idx.1 })
    (hMask :
      st.regs .bool [BLOCK_SIZE] "mask" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active st0 batch_size dim BLOCK_SIZE idx.1 })
    (hXRead : ∀ offset, st.readMem x_ptr offset = st0.readMem x_ptr offset)
    (hXOutNe : x_ptr ≠ y_ptr)
    (hStep :
      stepStmts (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim BLOCK_SIZE)
        (st.setReg "t" .nat [] (Tile.scalar t)) = some st') :
    st'.regs .real [BLOCK_SIZE] "Lambda" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          if active st0 batch_size dim BLOCK_SIZE idx.1 then
            some (st0.readMem lambda_ptr
              (IntegralDType.nat.mod (colOffset st0 BLOCK_SIZE idx.1) dim))
          else
            some 0 } ∧
      st'.regs .nat [BLOCK_SIZE] "col_offsets" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset st0 BLOCK_SIZE idx.1 } ∧
      st'.regs .bool [BLOCK_SIZE] "mask" =
        some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active st0 batch_size dim BLOCK_SIZE idx.1 } ∧
      (∀ offset, st'.readMem x_ptr offset = st0.readMem x_ptr offset) := by
  unfold diagSsmForwardLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hS, hLambda, hCol, hMask, hXRead,
    Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul, Option.bind,
    timeOffset, active, colOffset, IntegralDType.mod,
    diagSsmMaskedStateTile_succ] at hStep
  subst st'
  constructor
  · simp [BlockState.setReg, hLambda]
  · constructor
    · simp [BlockState.setReg, hCol]
    · constructor
      · simp [BlockState.setReg, hMask]
      · intro offset
        trans st.readMem x_ptr offset
        · simpa [BlockState.setReg, timeOffset, active, colOffset,
            diagSsmForwardSpec, diagSsmStateAfter, IntegralDType.mod] using
            BlockState.scatter_prop_masked_preserves_other_region
              (region := y_ptr)
              (offsetFn := fun i : TileIndex [BLOCK_SIZE] =>
                timeOffset st0 batch_size dim BLOCK_SIZE t i.1)
              (valueFn := fun i : TileIndex [BLOCK_SIZE] =>
                WithBot.unbotD 0
                  (Option.map₂ (fun x1 x2 => x1 + x2)
                    (Option.map₂ (fun x1 x2 => x1 * x2)
                      ((diagSsmMaskedStateTile st0 s_ptr x_ptr lambda_ptr
                        batch_size dim BLOCK_SIZE t).data (i.1, PUnit.unit))
                      (if active st0 batch_size dim BLOCK_SIZE i.1 then
                        some (st0.readMem lambda_ptr
                          (IntegralDType.nat.mod
                            (colOffset st0 BLOCK_SIZE i.1) dim))
                      else
                        some 0))
                    (if active st0 batch_size dim BLOCK_SIZE i.1 then
                      some (st0.readMem x_ptr
                        (timeOffset st0 batch_size dim BLOCK_SIZE t i.1))
                    else
                      some 0)))
              (P := fun i : TileIndex [BLOCK_SIZE] =>
                active st0 batch_size dim BLOCK_SIZE i.1)
              (R := x_ptr) hXOutNe offset (TileShape.allIndices [BLOCK_SIZE]) _
        · exact hXRead offset

theorem diagSsmForwardLoopContextInvariant_step_of_body
    {length : Nat}
    (st0 st st' : BlockState)
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE t : Nat)
    (ht : t < length)
    (hCtx :
      diagSsmForwardLoopContextInvariant st0 s_ptr x_ptr lambda_ptr y_ptr
        length batch_size dim BLOCK_SIZE t st)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset st0 batch_size dim BLOCK_SIZE idx))
    (hXOutNe : x_ptr ≠ y_ptr)
    (hStep :
      stepStmts (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim BLOCK_SIZE)
        (st.setReg "t" .nat [] (Tile.scalar t)) = some st') :
    diagSsmForwardLoopContextInvariant st0 s_ptr x_ptr lambda_ptr y_ptr length
      batch_size dim BLOCK_SIZE (t + 1) st' := by
  rcases hCtx with ⟨hInv, hLambda, hCol, hMask, hXRead⟩
  have hInv' :
      diagSsmForwardLoopInvariant st0 s_ptr x_ptr lambda_ptr y_ptr length
        batch_size dim BLOCK_SIZE (t + 1) st' :=
    diagSsmForwardLoopInvariant_step_of_concrete_body st0 st st' s_ptr x_ptr
      lambda_ptr y_ptr batch_size dim BLOCK_SIZE t ht hInv hInv.1 hLambda
      hCol hMask hXRead hOutInj hStep
  rcases diagSsmForwardLoopBody_step_preserves_context st0 st st' s_ptr x_ptr
      lambda_ptr y_ptr batch_size dim BLOCK_SIZE t hInv.1 hLambda hCol hMask
      hXRead hXOutNe hStep with
    ⟨hLambda', hCol', hMask', hXRead'⟩
  exact ⟨hInv', hLambda', hCol', hMask', hXRead'⟩

theorem diagSsmForwardLoopContextInvariant_body_step_exists
    {length : Nat}
    (st0 st : BlockState) (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE t : Nat)
    (ht : t < length)
    (hCtx :
      diagSsmForwardLoopContextInvariant st0 s_ptr x_ptr lambda_ptr y_ptr
        length batch_size dim BLOCK_SIZE t st)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset st0 batch_size dim BLOCK_SIZE idx))
    (hXOutNe : x_ptr ≠ y_ptr) :
    ∃ st',
      stepStmts (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim BLOCK_SIZE)
        (st.setReg "t" .nat [] (Tile.scalar t)) = some st' ∧
      diagSsmForwardLoopContextInvariant st0 s_ptr x_ptr lambda_ptr y_ptr
        length batch_size dim BLOCK_SIZE (t + 1) st' := by
  rcases hCtx with ⟨hInv, hLambda, hCol, hMask, hXRead⟩
  cases hStep :
      stepStmts (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim BLOCK_SIZE)
        (st.setReg "t" .nat [] (Tile.scalar t)) with
  | none =>
      unfold diagSsmForwardLoopBody at hStep
      simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hInv.1, hLambda, hCol, hMask,
        hXRead, Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        Option.bind, timeOffset, active, colOffset, IntegralDType.mod,
        diagSsmMaskedStateTile_succ] at hStep
  | some st' =>
      refine ⟨st', rfl, ?_⟩
      exact diagSsmForwardLoopContextInvariant_step_of_body st0 st st'
        s_ptr x_ptr lambda_ptr y_ptr batch_size dim BLOCK_SIZE t ht
        ⟨hInv, hLambda, hCol, hMask, hXRead⟩ hOutInj hXOutNe hStep

theorem diagSsmForwardForLoop_context_of_preloop
    (st0 stPre stLoop : BlockState)
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset st0 batch_size dim BLOCK_SIZE idx))
    (hXOutNe : x_ptr ≠ y_ptr)
    (hPre :
      stepStmts (diagSsmForwardPreLoop s_ptr lambda_ptr batch_size dim
        BLOCK_SIZE) st0 = some stPre)
    (hLoop :
      stepStmt (.forLoop "t" length
        (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim BLOCK_SIZE))
        stPre = some stLoop) :
    diagSsmForwardLoopContextInvariant st0 s_ptr x_ptr lambda_ptr y_ptr length
      batch_size dim BLOCK_SIZE length stLoop := by
  have hInit :
      diagSsmForwardLoopContextInvariant st0 s_ptr x_ptr lambda_ptr y_ptr
        length batch_size dim BLOCK_SIZE 0 stPre :=
    diagSsmForwardLoopContextInvariant_init_of_preloop st0 stPre s_ptr x_ptr
      lambda_ptr y_ptr length batch_size dim BLOCK_SIZE hPre
  obtain ⟨stFinal, hFor, hCtx⟩ :=
    forLoop_inv
      (idx := "t") (n := length)
      (body := diagSsmForwardLoopBody x_ptr y_ptr batch_size dim BLOCK_SIZE)
      (P := diagSsmForwardLoopContextInvariant st0 s_ptr x_ptr lambda_ptr
        y_ptr length batch_size dim BLOCK_SIZE)
      (s_init := stPre)
      hInit
      (by
        intro t st ht hCtx
        exact diagSsmForwardLoopContextInvariant_body_step_exists st0 st s_ptr
          x_ptr lambda_ptr y_ptr batch_size dim BLOCK_SIZE t ht hCtx hOutInj
          hXOutNe)
  have hEq : stFinal = stLoop := by
    rw [hLoop] at hFor
    injection hFor with h
    exact h.symm
  subst hEq
  exact hCtx

def diag_ssm_forward_kernel_alg_post
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (s s' : BlockState) : Prop :=
  ∀ idx : TileIndex [length, BLOCK_SIZE],
    diagSsmForwardActive s batch_size dim BLOCK_SIZE idx →
    s'.readMem y_ptr
        (diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx) =
      diagSsmForwardSpecAt s s_ptr x_ptr lambda_ptr batch_size dim BLOCK_SIZE idx

theorem diagSsmForwardProjectedBody_alg_post
    (s s' : BlockState) (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx))
    (hXOutNe : x_ptr ≠ y_ptr)
    (hExec :
      stepStmts (diagSsmForwardProjectedBody s_ptr x_ptr lambda_ptr y_ptr
        length batch_size dim BLOCK_SIZE) s = some s') :
    diag_ssm_forward_kernel_alg_post s_ptr x_ptr lambda_ptr y_ptr length
      batch_size dim BLOCK_SIZE s s' := by
  unfold diagSsmForwardProjectedBody at hExec
  rcases (stepStmts.append_some_iff).mp hExec with ⟨stPre, hPre, hLoopStmt⟩
  simp [stepStmts] at hLoopStmt
  have hLoop :
      stepStmt (.forLoop "t" length
        (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim BLOCK_SIZE))
        stPre = some s' := by
    cases hAux :
        stepForLoopAux "t" 0 length
          (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim BLOCK_SIZE)
          stPre with
    | none =>
        simp [hAux] at hLoopStmt
    | some stLoop =>
        simp [hAux] at hLoopStmt
        subst hLoopStmt
        simp [hAux]
  have hCtx :=
    diagSsmForwardForLoop_context_of_preloop s stPre s' s_ptr x_ptr
      lambda_ptr y_ptr length batch_size dim BLOCK_SIZE hOutInj hXOutNe hPre
      hLoop
  intro idx hidx
  exact hCtx.1.2 idx idx.1.isLt hidx

theorem diag_ssm_forward_kernel_alg_post_of_exec
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx))
    (hXOutNe : x_ptr ≠ y_ptr)
    (hExec :
      exec (diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr
        length batch_size dim BLOCK_SIZE) s = some s') :
    diag_ssm_forward_kernel_alg_post s_ptr x_ptr lambda_ptr y_ptr length
      batch_size dim BLOCK_SIZE s s' := by
  unfold exec at hExec
  rw [diag_ssm_forward_kernel_toAlg_body s_ptr x_ptr lambda_ptr y_ptr length
    batch_size dim BLOCK_SIZE] at hExec
  exact diagSsmForwardProjectedBody_alg_post s s' s_ptr x_ptr lambda_ptr y_ptr
    length batch_size dim BLOCK_SIZE hOutInj hXOutNe hExec

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

def diag_ssm_backward_kernel_correct_target
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (s : BlockState) : Prop :=
  ComputeCorrect.Realizes
    (kernel := diag_ssm_backward_kernel s_ptr lambda_ptr y_ptr grad_s_ptr
      grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE)
    (initialState := s)
    (write := diagSsmBackwardWriteMap (length := length) s grad_s_ptr
      grad_x_ptr grad_lambda_ptr batch_size dim BLOCK_SIZE)
    (expected := diagSsmBackwardExpected (length := length) s s_ptr
      lambda_ptr y_ptr grad_y_ptr batch_size dim BLOCK_SIZE)

theorem diag_ssm_backward_kernel_compute_correct_of_algorithm
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmBackwardGradXOffset s batch_size dim BLOCK_SIZE idx))
    (hGradXSNe : grad_x_ptr ≠ s_ptr)
    (hGradXYNe : grad_x_ptr ≠ y_ptr)
    (hGradXGradYNe : grad_x_ptr ≠ grad_y_ptr)
    (hGradXLambdaNe : grad_x_ptr ≠ lambda_ptr)
    (hGradSGradXNe : grad_s_ptr ≠ grad_x_ptr)
    (hGradLambdaGradXNe : grad_lambda_ptr ≠ grad_x_ptr)
    (hGradLambdaGradSNe : grad_lambda_ptr ≠ grad_s_ptr) :
    diag_ssm_backward_kernel_correct_target s_ptr lambda_ptr y_ptr grad_s_ptr
      grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
      s := by
  unfold diag_ssm_backward_kernel_correct_target
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0 out
  subst s0
  classical
  unfold diagSsmBackwardWriteMap
  by_cases hactive :
      diagSsmBackwardOutputActive s batch_size dim BLOCK_SIZE out
  · simp [hactive, diagSsmBackwardOutputAddr]
    exact diag_ssm_backward_kernel_alg_post_of_exec s_ptr lambda_ptr y_ptr
      grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim
      BLOCK_SIZE s s' hOutInj hGradXSNe hGradXYNe hGradXGradYNe
      hGradXLambdaNe hGradSGradXNe hGradLambdaGradXNe hGradLambdaGradSNe
      hExec out hactive
  · simp [hactive]

/-- Algorithm-layer correctness for the real-valued backward SSM kernel. -/
theorem diag_ssm_backward_kernel_correct
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmBackwardGradXOffset s batch_size dim BLOCK_SIZE idx))
    (hGradXSNe : grad_x_ptr ≠ s_ptr)
    (hGradXYNe : grad_x_ptr ≠ y_ptr)
    (hGradXGradYNe : grad_x_ptr ≠ grad_y_ptr)
    (hGradXLambdaNe : grad_x_ptr ≠ lambda_ptr)
    (hGradSGradXNe : grad_s_ptr ≠ grad_x_ptr)
    (hGradLambdaGradXNe : grad_lambda_ptr ≠ grad_x_ptr)
    (hGradLambdaGradSNe : grad_lambda_ptr ≠ grad_s_ptr) :
    diag_ssm_backward_kernel_correct_target s_ptr lambda_ptr y_ptr grad_s_ptr
      grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
      s :=
  diag_ssm_backward_kernel_compute_correct_of_algorithm s_ptr lambda_ptr y_ptr
    grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim
    BLOCK_SIZE s hOutInj hGradXSNe hGradXYNe hGradXGradYNe hGradXLambdaNe
    hGradSGradXNe hGradLambdaGradXNe hGradLambdaGradSNe

/-- Compute-facing correctness for the real-valued backward SSM kernel. -/
theorem diag_ssm_backward_kernel_compute_correct
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmBackwardGradXOffset s batch_size dim BLOCK_SIZE idx))
    (hGradXSNe : grad_x_ptr ≠ s_ptr)
    (hGradXYNe : grad_x_ptr ≠ y_ptr)
    (hGradXGradYNe : grad_x_ptr ≠ grad_y_ptr)
    (hGradXLambdaNe : grad_x_ptr ≠ lambda_ptr)
    (hGradSGradXNe : grad_s_ptr ≠ grad_x_ptr)
    (hGradLambdaGradXNe : grad_lambda_ptr ≠ grad_x_ptr)
    (hGradLambdaGradSNe : grad_lambda_ptr ≠ grad_s_ptr) :
    diag_ssm_backward_kernel_correct_target s_ptr lambda_ptr y_ptr grad_s_ptr
      grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
      s :=
  diag_ssm_backward_kernel_compute_correct_of_algorithm s_ptr lambda_ptr y_ptr
    grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim
    BLOCK_SIZE s hOutInj hGradXSNe hGradXYNe hGradXGradYNe hGradXLambdaNe
    hGradSGradXNe hGradLambdaGradXNe hGradLambdaGradSNe

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
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx))
    (hXOutNe : x_ptr ≠ y_ptr) :
    diag_ssm_forward_kernel_correct_target s_ptr x_ptr lambda_ptr y_ptr
      length batch_size dim BLOCK_SIZE s := by
  unfold diag_ssm_forward_kernel_correct_target
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx hidx
  exact diag_ssm_forward_kernel_alg_post_of_exec s_ptr x_ptr lambda_ptr y_ptr
    length batch_size dim BLOCK_SIZE s s' hOutInj hXOutNe hExec idx hidx

/-- Algorithm-layer correctness for the forward SSM kernel. -/
theorem diag_ssm_forward_kernel_correct
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx))
    (hXOutNe : x_ptr ≠ y_ptr) :
    diag_ssm_forward_kernel_correct_target s_ptr x_ptr lambda_ptr y_ptr
      length batch_size dim BLOCK_SIZE s :=
  diag_ssm_forward_kernel_compute_correct_of_algorithm s_ptr x_ptr lambda_ptr
    y_ptr length batch_size dim BLOCK_SIZE s hOutInj hXOutNe

/-- Compute-facing correctness for the forward SSM kernel. -/
theorem diag_ssm_forward_kernel_compute_correct
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx))
    (hXOutNe : x_ptr ≠ y_ptr) :
    diag_ssm_forward_kernel_correct_target s_ptr x_ptr lambda_ptr y_ptr
      length batch_size dim BLOCK_SIZE s :=
  diag_ssm_forward_kernel_compute_correct_of_algorithm s_ptr x_ptr lambda_ptr
    y_ptr length batch_size dim BLOCK_SIZE s hOutInj hXOutNe

/-- Per-kernel output summary for `diag_ssm_forward_kernel`: the DSL surface
lowers to the algorithm layer, and the time-step stores to `y_ptr` are
compute-correct — after the `0..length` recurrent scan every active output
offset holds the diagonal-SSM spec value `diagSsmForwardSpecAt`, and inactive
lanes are preserved. Mirrors `add_kernel_output_summary`. -/
theorem diag_ssm_forward_kernel_output_summary
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [length, BLOCK_SIZE] =>
        diagSsmForwardOutOffset s batch_size dim BLOCK_SIZE idx))
    (hXOutNe : x_ptr ≠ y_ptr) :
    (∃ alg, (diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr
        length batch_size dim BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    diag_ssm_forward_kernel_correct_target s_ptr x_ptr lambda_ptr y_ptr
      length batch_size dim BLOCK_SIZE s := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact diag_ssm_forward_kernel_compute_correct s_ptr x_ptr lambda_ptr y_ptr
    length batch_size dim BLOCK_SIZE s hOutInj hXOutNe

end VeriTile.Bench.TritonBenchG.DiagSsmTriton
