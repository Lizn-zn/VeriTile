import VeriTile.Triton

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
Side conditions: output-store-offset
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
to `ComputeCorrect.Realizes_without_Rounding` under the stated no-collision/no-alias
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

noncomputable def diagSsmMaskedStateTile
    (st : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE t : Nat) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if active st batch_size dim BLOCK_SIZE idx.1 then
        some (diagSsmStateAfter st s_ptr x_ptr lambda_ptr batch_size dim
          BLOCK_SIZE idx.1 t)
      else
        some (0.0 : ℝ) }

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
  ComputeCorrect.Realizes_without_Rounding
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
  ComputeCorrect.Realizes_without_Rounding
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
specification diag_ssm_forward_kernel_output_summary
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

/-! ## The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre)

Everything below is purely additive; the exact surface above is untouched.
This is the first consumer of the **three-stream** per-step emit skin
`StreamEmitMasked2DKernelIO₃` (streaming genre, style S3): the store sits
**inside** the `tl.for t in length` scan loop, so the output is a per-step
`BLOCK_SIZE`-lane window family, and the kernel's spec `f t j` is the
genre's *scan* shape — a prefix fold of the streamed `x` tiles seeded by
the initial-state tile and folded with the decay tile. The `s`/`Λ`
channels are the skin's degenerate **static** streams: their read windows
ignore the step index (a pre-loop tile, re-pinned at the same cell for
every `t`), exactly the design point recorded in the skin's docstring.

Structure of the `execR R` story: this kernel has **zero rounding
events** — every load and store is at `.real` and there is no `castFloat`
at all, so the whole surface collapses verbatim onto the exact stepper and
the proven `preLoop` / `diagSsmForwardLoopContextInvariant` stack above is
reused unchanged. The `⊨[R]` face adds only the `TraceSafeR` walk (the
first over a `forLoop`, via a private `forLoopTraceSafeR` invariant
principle mirroring the library's `forRangeTraceSafeR_inv`), the per-cell
memory frame, and the stream-lane spec bridge. -/

section IOFace

open scoped VeriTile.Triton.StreamEmitMasked2DKernelIO₃

/-! ### IO signature -/

/-- **Streaming IO signature** of `diag_ssm_forward_kernel` on the
three-stream per-step emit skin (S3: in-loop store). Step `t`, lane `j`
of program `pid₀` (the grid is 1-D; `pid₁` is unused and every window
ignores it) transcribes the kernel's pointer arithmetic verbatim
(`col_offsets = pid₀·BLOCK_SIZE + j`):

* `read1` (the `s_ptr` initial-state tile, **static**: ignores `t`):
  `pid₀·BLOCK_SIZE + j` — the kernel's pre-loop `s_ptr + col_offsets`.
* `read2` (the `lambda_ptr` decay tile, **static**: ignores `t`):
  `(pid₀·BLOCK_SIZE + j) % dim` — the kernel's pre-loop
  `lambda_ptr + col_offsets % dim` (the `%` is just the window function;
  reads need no injectivity).
* `read3` (the `x_ptr` input, **streamed**):
  `t·(batch_size·dim) + pid₀·BLOCK_SIZE + j` — the kernel's in-loop
  `offsets = t * batch_size * dim + col_offsets`.
* `write` (the `y_ptr` output): the same in-loop `offsets` window.

All four masks are the kernel's single `mask`:
`pid₀·BLOCK_SIZE + j < batch_size·dim` (`t`-independent). The store is the
raw `tl.store(y_ptr + offsets, s, mask)` at `.real`, so `outDType` keeps
the skin's `.real` default — no quantization event. -/
def diagSsmForwardKernelIO (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    StreamEmitMasked2DKernelIO₃ where
  kernel := diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr length
    batch_size dim BLOCK_SIZE
  inp1 := s_ptr
  inp2 := lambda_ptr
  inp3 := x_ptr
  out := y_ptr
  T := length
  B1 := BLOCK_SIZE
  B2 := BLOCK_SIZE
  B3 := BLOCK_SIZE
  C := BLOCK_SIZE
  read1 := fun p₀ _ _ j => p₀ * BLOCK_SIZE + j.val
  read2 := fun p₀ _ _ j => (p₀ * BLOCK_SIZE + j.val) % dim
  read3 := fun p₀ _ t j => t.val * (batch_size * dim) + (p₀ * BLOCK_SIZE + j.val)
  write := fun p₀ _ t j => t.val * (batch_size * dim) + (p₀ * BLOCK_SIZE + j.val)
  mask1 := fun p₀ _ _ j => p₀ * BLOCK_SIZE + j.val < batch_size * dim
  mask2 := fun p₀ _ _ j => p₀ * BLOCK_SIZE + j.val < batch_size * dim
  mask3 := fun p₀ _ _ j => p₀ * BLOCK_SIZE + j.val < batch_size * dim
  writeMask := fun p₀ _ _ j => p₀ * BLOCK_SIZE + j.val < batch_size * dim

/-! ### The stream-level spec -/

/-- The stream-side diagonal-SSM scan: state after `u` steps of
`s ← s·lam + xs u`, seeded with `init`. The step input is guarded by the
stream length (`u < T`); the guard is always live on the bridge below
(only prefixes `u ≤ T` are ever evaluated). Mirrors the memory-side
recurrence `diagSsmStateAfter` with the three memory channels replaced by
the skin's per-lane stream values. -/
noncomputable def diagSsmStreamState {T : Nat} (init lam : ℝ)
    (xs : Fin T → ℝ) : Nat → ℝ
  | 0 => init
  | u + 1 =>
      diagSsmStreamState init lam xs u * lam +
        (if h : u < T then xs ⟨u, h⟩ else 0)

/-! ### The stream-lane spec bridge -/

/-- Per-lane spec bridge: under the skin's stream pins (the static `s`/`Λ`
windows pinned at the same cell for every step — instantiated at the
given `t` — and the streamed `x` window pinned per step), the stream-side
scan **is** the memory-side recurrence `diagSsmStateAfter` on every prefix
`u ≤ length`. The recurrence at lane `j` only touches lane `j` of all
three streams, so no unmasked-lane values enter. -/
private theorem dssm_streamState_eq_stateAfter
    (s₀ : BlockState) (s_ptr x_ptr lambda_ptr : RegionName)
    {length : Nat} (batch_size dim BLOCK_SIZE : Nat)
    (xs ys zs : Fin length → Fin BLOCK_SIZE → ℝ)
    (t : Fin length) (j : Fin BLOCK_SIZE)
    (hss : s₀.readMem s_ptr (colOffset s₀ BLOCK_SIZE j) = xs t j)
    (hls : s₀.readMem lambda_ptr (colOffset s₀ BLOCK_SIZE j % dim) = ys t j)
    (hzs : ∀ u : Fin length,
      s₀.readMem x_ptr (timeOffset s₀ batch_size dim BLOCK_SIZE u.val j)
        = zs u j)
    (u : Nat) :
    u ≤ length →
      diagSsmStreamState (xs t j) (ys t j) (fun v => zs v j) u
        = diagSsmStateAfter s₀ s_ptr x_ptr lambda_ptr batch_size dim
            BLOCK_SIZE j u := by
  induction u with
  | zero =>
      intro _
      simpa [diagSsmStreamState] using hss.symm
  | succ v ih =>
      intro hu
      have hv : v < length := hu
      have ihv := ih (Nat.le_of_lt hv)
      calc diagSsmStreamState (xs t j) (ys t j) (fun w => zs w j) (v + 1)
          = diagSsmStreamState (xs t j) (ys t j) (fun w => zs w j) v
              * ys t j + zs ⟨v, hv⟩ j := by
            simp [diagSsmStreamState, dif_pos hv]
        _ = diagSsmStateAfter s₀ s_ptr x_ptr lambda_ptr batch_size dim
              BLOCK_SIZE j v * ys t j + zs ⟨v, hv⟩ j := by rw [ihv]
        _ = diagSsmStateAfter s₀ s_ptr x_ptr lambda_ptr batch_size dim
              BLOCK_SIZE j (v + 1) := by
            rw [diagSsmStateAfter_succ, ← hls, ← hzs ⟨v, hv⟩]
            rfl

/-! ### Window injectivity from the tiling geometry -/

/-- The step-window strict growth core: with lane offsets below the
timestep pitch `D`, an earlier step's window sits strictly below a later
step's. -/
private theorem dssm_window_lt (D C t t' j j' : Nat)
    (hj : j < D) (ht : t < t') :
    t * D + (C + j) < t' * D + (C + j') := by
  have h2 : (t + 1) * D ≤ t' * D := Nat.mul_le_mul_right _ ht
  have h3 : (t + 1) * D = t * D + D := by ring
  omega

/-- The full-grid output-window injectivity the exact invariant stack rides
on (`hOutInj` of `diag_ssm_forward_kernel_output_summary`), derived from
the tiling-geometry hypothesis `BLOCK_SIZE ≤ batch_size·dim`: distinct
`(t, j)` windows never collide because the lane offset stays below the
timestep pitch. -/
private theorem dssm_outOffset_injective (st : BlockState)
    (batch_size dim BLOCK_SIZE length : Nat)
    (hBS : BLOCK_SIZE ≤ batch_size * dim) :
    Function.Injective (fun idx : TileIndex [length, BLOCK_SIZE] =>
      diagSsmForwardOutOffset st batch_size dim BLOCK_SIZE idx) := by
  intro a b h
  obtain ⟨ta, ja, ua⟩ := a
  obtain ⟨tb, jb, ub⟩ := b
  cases ua
  cases ub
  simp only [diagSsmForwardOutOffset, timeOffset, colOffset] at h
  have hja : ja.val < batch_size * dim := lt_of_lt_of_le ja.isLt hBS
  have hjb : jb.val < batch_size * dim := lt_of_lt_of_le jb.isLt hBS
  have ht : ta.val = tb.val := by
    rcases Nat.lt_trichotomy ta.val tb.val with hlt | heq | hgt
    · exact absurd h (Nat.ne_of_lt (dssm_window_lt (batch_size * dim)
        (st.pids 0 * BLOCK_SIZE) ta.val tb.val ja.val jb.val hja hlt))
    · exact heq
    · exact absurd h.symm (Nat.ne_of_lt (dssm_window_lt (batch_size * dim)
        (st.pids 0 * BLOCK_SIZE) tb.val ta.val jb.val ja.val hjb hgt))
  have hta : ta = tb := Fin.ext ht
  subst hta
  have hj : ja = jb := Fin.ext (by omega)
  subst hj
  rfl

/-! ### Cast-free collapses and the covered fragment -/

/-- The pre-loop is cast-free (nat index arithmetic and two masked `.real`
loads, no `castFloat` anywhere): it steps identically under
`stepStmtsR R`. -/
private theorem dssm_preLoop_castFree (R : RoundingModel)
    (s_ptr lambda_ptr : RegionName) (batch_size dim BLOCK_SIZE : Nat)
    (t : BlockState) :
    stepStmtsR R (diagSsmForwardPreLoop s_ptr lambda_ptr batch_size dim
        BLOCK_SIZE) t
      = stepStmts (diagSsmForwardPreLoop s_ptr lambda_ptr batch_size dim
        BLOCK_SIZE) t := by
  simp only [diagSsmForwardPreLoop, stepStmtsR, stepStmts, stepStmtR,
    stepStmt, evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The loop body is cast-free **including its in-loop masked `.real`
store** (`stepStmtR` delegates a `.real`-typed store to the exact write),
so the whole scan loop steps identically under `stepStmtsR R` and the
exact context-invariant stack transports to `execR`. -/
private theorem dssm_body_castFree (R : RoundingModel)
    (x_ptr y_ptr : RegionName) (batch_size dim BLOCK_SIZE : Nat)
    (t : BlockState) :
    stepStmtsR R (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim
        BLOCK_SIZE) t
      = stepStmts (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim
        BLOCK_SIZE) t := by
  simp only [diagSsmForwardLoopBody, stepStmtsR, stepStmts, stepStmtR,
    stepStmt, evalOpR.eq_def, evalOp.eq_def, BlockState.writeMemTypedR]
  rfl

/-- The forward surface sits inside the flat-memory bridge's covered
fragment (`FlattenOk`; the `forLoop` clause recurses into the cast-free
body). -/
private theorem dssm_forward_flattenOk
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    ((diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr length batch_size
      dim BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [diag_ssm_forward_kernel_toAlg_body]
  simp [diagSsmForwardProjectedBody, diagSsmForwardPreLoop,
    diagSsmForwardLoopBody, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

/-! ### Cell-level memory frames -/

/-- A run of `assign` statements never touches memory (private copy of the
S3 template's helper — bench files never import each other). -/
private theorem dssm_stepStmts_assigns_mem :
    ∀ (l : List Stmt),
      (∀ stmt ∈ l, ∃ dt sh nm, ∃ e : Op dt sh, stmt = Stmt.assign dt sh nm e) →
      ∀ {s s' : BlockState}, stepStmts l s = some s' → s'.mem = s.mem
  | [], _, s, s', h => by
      rw [stepStmts.nil] at h
      obtain rfl := Option.some.inj h
      rfl
  | stmt :: rest, hall, s, s', h => by
      obtain ⟨dt, sh, nm, e, rfl⟩ := hall _ List.mem_cons_self
      cases hv : evalOp e s with
      | none => simp [stepStmts, stepStmt, hv] at h
      | some v =>
          rw [stepStmts.cons_some (stepStmt_assign_eq_some hv)] at h
          rw [dssm_stepStmts_assigns_mem rest
            (fun st' hst' => hall st' (List.mem_cons_of_mem _ hst')) h]
          rfl

/-- Cell-level frame of a `Prop`-masked exact `writeMem` scatter `foldl`:
every cell not hit by an active lane is untouched (the cell-level sibling
of `BlockState.scatter_prop_masked_preserves_other_offset`). -/
private theorem dssm_foldl_writeMem_preserve_cell {α : Type}
    {region : RegionName} (ofn : α → Nat) (vfn : α → ℝ) (P : α → Prop)
    [DecidablePred P]
    (r : RegionName) (oo : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(r = region ∧ oo = ofn k)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (ofn k) (vfn k) else acc) s).mem r oo
      = s.mem r oo := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hm : P hd
      · rw [if_pos hm,
          ih _ fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk,
          BlockState.writeMem_mem]
        exact if_neg (hnot hd List.mem_cons_self hm)
      · rw [if_neg hm]
        exact ih _ fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk

/-- **Cell-level frame of one scan iteration** (the `mem` twin of the
concrete-body step): from the context-invariant register pins, one body
iteration leaves every cell off the step-`t` active write window
untouched. -/
private theorem dssm_body_step_frame {length : Nat}
    (st0 st st' : BlockState)
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE t : Nat) (ht : t < length)
    (hS : st.regs .real [BLOCK_SIZE] "s" =
      some (diagSsmMaskedStateTile st0 s_ptr x_ptr lambda_ptr batch_size dim
        BLOCK_SIZE t))
    (hLambda : st.regs .real [BLOCK_SIZE] "Lambda" =
      some { data := fun idx : TileIndex [BLOCK_SIZE] =>
        if active st0 batch_size dim BLOCK_SIZE idx.1 then
          some (st0.readMem lambda_ptr
            (IntegralDType.nat.mod (colOffset st0 BLOCK_SIZE idx.1) dim))
        else
          some 0 })
    (hCol : st.regs .nat [BLOCK_SIZE] "col_offsets" =
      some { data := fun idx : TileIndex [BLOCK_SIZE] =>
        colOffset st0 BLOCK_SIZE idx.1 })
    (hMask : st.regs .bool [BLOCK_SIZE] "mask" =
      some { data := fun idx : TileIndex [BLOCK_SIZE] =>
        active st0 batch_size dim BLOCK_SIZE idx.1 })
    (hXRead : ∀ offset, st.readMem x_ptr offset = st0.readMem x_ptr offset)
    (hStep :
      stepStmts (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim BLOCK_SIZE)
        (st.setReg "t" .nat [] (Tile.scalar t)) = some st')
    (r : RegionName) (oo : Nat)
    (hcond : r ≠ y_ptr ∨ ∀ (tf : Fin length) (j : Fin BLOCK_SIZE),
      active st0 batch_size dim BLOCK_SIZE j →
      oo ≠ timeOffset st0 batch_size dim BLOCK_SIZE tf.val j) :
    st'.mem r oo = st.mem r oo := by
  unfold diagSsmForwardLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hS, hLambda, hCol, hMask,
    hXRead, Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
    Option.bind, timeOffset, active, colOffset, IntegralDType.mod,
    diagSsmMaskedStateTile_succ] at hStep
  subst st'
  refine Eq.trans (dssm_foldl_writeMem_preserve_cell _ _ _ r oo _ _ ?_) rfl
  intro k _ hPk hbad
  rcases hcond with hne | hno
  · exact hne hbad.1
  · exact hno ⟨t, ht⟩ k.1 hPk hbad.2

/-! ### The `TraceSafeR` walk

The first bench `TraceSafeR` walk over a `forLoop` (the library principle
`Stmt.forRangeTraceSafeR_inv` covers only `forRange`), so the invariant
principle is built privately below, mirroring the library proof verbatim
with the counter stepping by `1`. -/

/-- `forLoop` mirror of `Stmt.forRangeTraceSafeR_inv`: an invariant `P`
whose every step yields the body's `TraceSafeListR` plus a successor
carrying `P` makes the whole loop unrolling trace-safe. -/
private theorem dssm_forLoopTraceSafeR_inv (R : RoundingModel)
    (bounds : RegionBounds) (idx : RegName) (n : Nat) (body : List Stmt)
    (P : Nat → BlockState → Prop)
    (hstep : ∀ c s, c < n → P c s →
      Stmt.TraceSafeListR R bounds body
        (s.setReg idx .nat [] (Tile.scalar c)) ∧
      ∃ s', stepStmtsR R body (s.setReg idx .nat [] (Tile.scalar c))
          = some s' ∧
        P (c + 1) s') :
    ∀ cur s, P cur s → Stmt.forLoopTraceSafeR R bounds idx cur n body s
  | cur, s, hP => by
      rw [Stmt.forLoopTraceSafeR]
      split
      · obtain ⟨hsafe, s', hrun, hP'⟩ := hstep cur s ‹cur < n› hP
        refine ⟨hsafe, ?_⟩
        rw [hrun]
        exact dssm_forLoopTraceSafeR_inv R bounds idx n body P hstep
          (cur + 1) s' hP'
      · trivial
  termination_by cur _ _ => n - cur
  decreasing_by omega

/-- `evalOpR` = `evalOp` + concrete value of the `col_idx` op. -/
private theorem dssm_colIdx_evalR (R : RoundingModel) (BLOCK_SIZE : Nat)
    (s : BlockState) :
    evalOpR R (Op.mul NumericDType.nat Broadcast.nil (Op.programId 0)
        (Op.constNat BLOCK_SIZE)) s
      = some (Tile.scalar (s.pids 0 * BLOCK_SIZE)) := by
  have h : evalOpR R (Op.mul NumericDType.nat Broadcast.nil (Op.programId 0)
        (Op.constNat BLOCK_SIZE)) s
      = evalOp (Op.mul NumericDType.nat Broadcast.nil (Op.programId 0)
        (Op.constNat BLOCK_SIZE)) s := by
    simp only [evalOpR.eq_def, evalOp.eq_def]
  rw [h, evalOp_mul, evalOp_programId, evalOp_constNat]
  rfl

/-- `evalOpR` = `evalOp` + concrete value of the `col_offsets` op. -/
private theorem dssm_colOffsets_evalR (R : RoundingModel) (BLOCK_SIZE : Nat)
    (s : BlockState) (c : Nat)
    (hci : s.regs .nat [] "col_idx" = some (Tile.scalar c)) :
    evalOpR R (Op.add NumericDType.nat Broadcast.scalarL
        (Op.ref .nat [] "col_idx") (Op.arange BLOCK_SIZE)) s
      = some { data := fun idx : TileIndex [BLOCK_SIZE] => c + idx.1.val } := by
  have h : evalOpR R (Op.add NumericDType.nat Broadcast.scalarL
        (Op.ref .nat [] "col_idx") (Op.arange BLOCK_SIZE)) s
      = evalOp (Op.add NumericDType.nat Broadcast.scalarL
        (Op.ref .nat [] "col_idx") (Op.arange BLOCK_SIZE)) s := by
    simp only [evalOpR.eq_def, evalOp.eq_def]
  rw [h, evalOp_add, evalOp_ref, hci, evalOp_arange]
  rfl

/-- `evalOpR` = `evalOp` + concrete value of the `mask` op. -/
private theorem dssm_mask_evalR (R : RoundingModel)
    (batch_size dim BLOCK_SIZE : Nat) (s : BlockState)
    (co : Fin BLOCK_SIZE → Nat)
    (hco : s.regs .nat [BLOCK_SIZE] "col_offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] => co idx.1 }) :
    evalOpR R (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref .nat [BLOCK_SIZE] "col_offsets")
        (Op.constNat (batch_size * dim))) s
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          decide (co idx.1 < batch_size * dim) } := by
  have h : evalOpR R (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref .nat [BLOCK_SIZE] "col_offsets")
        (Op.constNat (batch_size * dim))) s
      = evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref .nat [BLOCK_SIZE] "col_offsets")
        (Op.constNat (batch_size * dim))) s := by
    simp only [evalOpR.eq_def, evalOp.eq_def]
  rw [h, evalOp_lt, evalOp_ref, hco, evalOp_constNat]
  rfl

/-- `evalOpR` = `evalOp` + concrete value of the `Λ` load's address op. -/
private theorem dssm_lambdaAddr_evalR (R : RoundingModel)
    (dim BLOCK_SIZE : Nat) (s : BlockState) (co : Fin BLOCK_SIZE → Nat)
    (hco : s.regs .nat [BLOCK_SIZE] "col_offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] => co idx.1 }) :
    evalOpR R (Op.mod IntegralDType.nat Broadcast.scalarR
        (Op.ref .nat [BLOCK_SIZE] "col_offsets") (Op.constNat dim)) s
      = some { data := fun idx : TileIndex [BLOCK_SIZE] => co idx.1 % dim }
      := by
  have h : evalOpR R (Op.mod IntegralDType.nat Broadcast.scalarR
        (Op.ref .nat [BLOCK_SIZE] "col_offsets") (Op.constNat dim)) s
      = evalOp (Op.mod IntegralDType.nat Broadcast.scalarR
        (Op.ref .nat [BLOCK_SIZE] "col_offsets") (Op.constNat dim)) s := by
    simp only [evalOpR.eq_def, evalOp.eq_def]
  rw [h, evalOp_mod, evalOp_ref, hco, evalOp_constNat]
  rfl

/-- `evalOpR` = `evalOp` + concrete value of the in-loop `offsets` op. -/
private theorem dssm_offsets_evalR (R : RoundingModel)
    (batch_size dim BLOCK_SIZE : Nat) (s : BlockState) (t : Nat)
    (co : Fin BLOCK_SIZE → Nat)
    (ht : s.regs .nat [] "t" = some (Tile.scalar t))
    (hco : s.regs .nat [BLOCK_SIZE] "col_offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] => co idx.1 }) :
    evalOpR R (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "t")
          (Op.constNat (batch_size * dim)))
        (Op.ref .nat [BLOCK_SIZE] "col_offsets")) s
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          t * (batch_size * dim) + co idx.1 } := by
  have h : evalOpR R (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "t")
          (Op.constNat (batch_size * dim)))
        (Op.ref .nat [BLOCK_SIZE] "col_offsets")) s
      = evalOp (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "t")
          (Op.constNat (batch_size * dim)))
        (Op.ref .nat [BLOCK_SIZE] "col_offsets")) s := by
    simp only [evalOpR.eq_def, evalOp.eq_def]
  rw [h, evalOp_add, evalOp_mul, evalOp_ref, ht, evalOp_constNat,
    evalOp_ref, hco]
  rfl

/-- `TraceSafeListR` of the pre-loop: three register-only index assigns,
then the two masked static-tile loads, whose active lanes are exactly the
skin's (`t`-independent) `mask1`/`mask2` windows — in bounds by the
`read1`/`read2` window bounds. -/
private theorem dssm_preLoopSafeR (R : RoundingModel) (bounds : RegionBounds)
    (s_ptr lambda_ptr : RegionName) (batch_size dim BLOCK_SIZE : Nat)
    (s : BlockState)
    (hbs : ∀ j : Fin BLOCK_SIZE,
      s.pids 0 * BLOCK_SIZE + j.val < batch_size * dim →
      s.pids 0 * BLOCK_SIZE + j.val < bounds s_ptr)
    (hbl : ∀ j : Fin BLOCK_SIZE,
      s.pids 0 * BLOCK_SIZE + j.val < batch_size * dim →
      (s.pids 0 * BLOCK_SIZE + j.val) % dim < bounds lambda_ptr) :
    Stmt.TraceSafeListR R bounds
      (diagSsmForwardPreLoop s_ptr lambda_ptr batch_size dim BLOCK_SIZE) s := by
  unfold diagSsmForwardPreLoop
  -- col_idx
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t1 ht1 => ?_)
  rw [stepStmtR_assign_eq_some (dssm_colIdx_evalR R BLOCK_SIZE s)] at ht1
  obtain rfl := Option.some.inj ht1
  set q1 := s.setReg "col_idx" .nat [] (Tile.scalar (s.pids 0 * BLOCK_SIZE))
    with hq1
  have hci1 : q1.regs .nat [] "col_idx"
      = some (Tile.scalar (s.pids 0 * BLOCK_SIZE)) := by simp [hq1]
  -- col_offsets
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t2 ht2 => ?_)
  rw [stepStmtR_assign_eq_some
    (dssm_colOffsets_evalR R BLOCK_SIZE q1 (s.pids 0 * BLOCK_SIZE) hci1)] at ht2
  obtain rfl := Option.some.inj ht2
  set q2 := q1.setReg "col_offsets" .nat [BLOCK_SIZE]
    { data := fun idx : TileIndex [BLOCK_SIZE] =>
        s.pids 0 * BLOCK_SIZE + idx.1.val } with hq2
  have hco2 : q2.regs .nat [BLOCK_SIZE] "col_offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          s.pids 0 * BLOCK_SIZE + idx.1.val } := by simp [hq2]
  -- mask
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t3 ht3 => ?_)
  rw [stepStmtR_assign_eq_some (dssm_mask_evalR R batch_size dim BLOCK_SIZE q2
    (fun i => s.pids 0 * BLOCK_SIZE + i.val) hco2)] at ht3
  obtain rfl := Option.some.inj ht3
  set q3 := q2.setReg "mask" .bool [BLOCK_SIZE]
    { data := fun idx : TileIndex [BLOCK_SIZE] =>
        decide (s.pids 0 * BLOCK_SIZE + idx.1.val < batch_size * dim) }
    with hq3
  have hco3 : q3.regs .nat [BLOCK_SIZE] "col_offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          s.pids 0 * BLOCK_SIZE + idx.1.val } := by
    rw [hq3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("col_offsets" : RegName) ≠ "mask" by decide)]
    exact hco2
  have hmask3 : q3.regs .bool [BLOCK_SIZE] "mask"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          decide (s.pids 0 * BLOCK_SIZE + idx.1.val < batch_size * dim) } := by
    simp [hq3]
  -- the masked s load
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t4 ht4 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
      and_true, true_and, and_self]
    intro offsets hoffsets idx hactive
    rw [evalOpR_ref, hco3] at hoffsets
    obtain rfl := Option.some.inj hoffsets
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hmask3] at hm
    obtain rfl := Option.some.inj hm
    have hlt : s.pids 0 * BLOCK_SIZE + idx.1.val < batch_size * dim := by
      simpa using hmi
    simpa [Region.cast_id] using hbs idx.1 hlt
  · obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv ht4
    set q4 := q3.setReg "s" .real [BLOCK_SIZE] v4 with hq4
    have hco4 : q4.regs .nat [BLOCK_SIZE] "col_offsets"
        = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
            s.pids 0 * BLOCK_SIZE + idx.1.val } := by
      rw [hq4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("col_offsets" : RegName) ≠ "s" by decide)]
      exact hco3
    have hmask4 : q4.regs .bool [BLOCK_SIZE] "mask"
        = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
            decide (s.pids 0 * BLOCK_SIZE + idx.1.val < batch_size * dim) } := by
      rw [hq4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("mask" : RegName) ≠ "s" by decide)]
      exact hmask3
    -- the masked Λ load
    refine Stmt.TraceSafeListR.cons_intro ?_
      (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
      and_true, true_and, and_self]
    intro offsets hoffsets idx hactive
    rw [dssm_lambdaAddr_evalR R dim BLOCK_SIZE q4
      (fun i => s.pids 0 * BLOCK_SIZE + i.val) hco4] at hoffsets
    obtain rfl := Option.some.inj hoffsets
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hmask4] at hm
    obtain rfl := Option.some.inj hm
    have hlt : s.pids 0 * BLOCK_SIZE + idx.1.val < batch_size * dim := by
      simpa using hmi
    simpa [Region.cast_id] using hbl idx.1 hlt

/-- Per-iteration `TraceSafeListR` for the scan body: the `offsets` and
`s`-update assigns are register-only; the masked `x` load's and the masked
store's **active** lanes are the skin's `mask3`/`writeMask` windows at
step `t`, in bounds by the `read3`/`write` window bounds. -/
private theorem dssm_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (x_ptr y_ptr : RegionName) (batch_size dim BLOCK_SIZE : Nat)
    (s0 st : BlockState) (t : Nat)
    (hCol : st.regs .nat [BLOCK_SIZE] "col_offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset s0 BLOCK_SIZE idx.1 })
    (hMask : st.regs .bool [BLOCK_SIZE] "mask"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active s0 batch_size dim BLOCK_SIZE idx.1 })
    (hbx : ∀ j : Fin BLOCK_SIZE, active s0 batch_size dim BLOCK_SIZE j →
      t * (batch_size * dim) + colOffset s0 BLOCK_SIZE j < bounds x_ptr)
    (hby : ∀ j : Fin BLOCK_SIZE, active s0 batch_size dim BLOCK_SIZE j →
      t * (batch_size * dim) + colOffset s0 BLOCK_SIZE j < bounds y_ptr) :
    Stmt.TraceSafeListR R bounds
      (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim BLOCK_SIZE)
      (st.setReg "t" .nat [] (Tile.scalar t)) := by
  unfold diagSsmForwardLoopBody
  have hT0 : (st.setReg "t" .nat [] (Tile.scalar t)).regs .nat [] "t"
      = some (Tile.scalar t) := BlockState.setReg_same _ _ _ _ _
  have hco0 : (st.setReg "t" .nat [] (Tile.scalar t)).regs .nat
      [BLOCK_SIZE] "col_offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset s0 BLOCK_SIZE idx.1 } := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("col_offsets" : RegName) ≠ "t" by decide)]
    exact hCol
  have hmask0 : (st.setReg "t" .nat [] (Tile.scalar t)).regs .bool
      [BLOCK_SIZE] "mask"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active s0 batch_size dim BLOCK_SIZE idx.1 } := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("mask" : RegName) ≠ "t" by decide)]
    exact hMask
  -- offsets
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t1 ht1 => ?_)
  rw [stepStmtR_assign_eq_some (dssm_offsets_evalR R batch_size dim BLOCK_SIZE
    _ t (fun i => colOffset s0 BLOCK_SIZE i) hT0 hco0)] at ht1
  obtain rfl := Option.some.inj ht1
  set q1 := (st.setReg "t" .nat [] (Tile.scalar t)).setReg "offsets" .nat
    [BLOCK_SIZE]
    { data := fun idx : TileIndex [BLOCK_SIZE] =>
        t * (batch_size * dim) + colOffset s0 BLOCK_SIZE idx.1 } with hq1
  have hoffs1 : q1.regs .nat [BLOCK_SIZE] "offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          t * (batch_size * dim) + colOffset s0 BLOCK_SIZE idx.1 } := by
    simp [hq1]
  have hmask1 : q1.regs .bool [BLOCK_SIZE] "mask"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active s0 batch_size dim BLOCK_SIZE idx.1 } := by
    rw [hq1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("mask" : RegName) ≠ "offsets" by decide)]
    exact hmask0
  -- the masked x load
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t2 ht2 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
      and_true, true_and, and_self]
    intro offsets hoffsets idx hactive
    rw [evalOpR_ref, hoffs1] at hoffsets
    obtain rfl := Option.some.inj hoffsets
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hmask1] at hm
    obtain rfl := Option.some.inj hm
    have hact : active s0 batch_size dim BLOCK_SIZE idx.1 := by
      simpa using hmi
    simpa [Region.cast_id] using hbx idx.1 hact
  · obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv ht2
    set q2 := q1.setReg "x" .real [BLOCK_SIZE] v2 with hq2
    -- s update (register-only)
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t3 ht3 => ?_)
    obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv ht3
    set q3 := q2.setReg "s" .real [BLOCK_SIZE] v3 with hq3
    have hoffs3 : q3.regs .nat [BLOCK_SIZE] "offsets"
        = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
            t * (batch_size * dim) + colOffset s0 BLOCK_SIZE idx.1 } := by
      rw [hq3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("offsets" : RegName) ≠ "s" by decide),
        hq2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("offsets" : RegName) ≠ "x" by decide)]
      exact hoffs1
    have hmask3 : q3.regs .bool [BLOCK_SIZE] "mask"
        = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
            active s0 batch_size dim BLOCK_SIZE idx.1 } := by
      rw [hq3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("mask" : RegName) ≠ "s" by decide),
        hq2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("mask" : RegName) ≠ "x" by decide)]
      exact hmask1
    -- the masked store
    refine Stmt.TraceSafeListR.cons_intro ?_
      (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
      MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR, and_true, true_and, and_self]
    intro offsets hoffsets idx hactive
    rw [evalOpR_ref, hoffs3] at hoffsets
    obtain rfl := Option.some.inj hoffsets
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hmask3] at hm
    obtain rfl := Option.some.inj hm
    have hact : active s0 batch_size dim BLOCK_SIZE idx.1 := by
      simpa using hmi
    simpa [Region.cast_id] using hby idx.1 hact

/-- **The `TraceSafeR` walk for the whole forward kernel** — the pre-loop
by the manual cons walk, the scan loop by the private
`forLoopTraceSafeR` invariant principle over the proven (launch-state-
robust) `diagSsmForwardLoopContextInvariant`, whose register pins feed the
per-iteration body walk. The four bound groups are the skin's
`read1`/`read2`/`read3`/`write` windows; `hT` instantiates the static
windows' step index (a `Fin length` needs a step to exist), and
`hBS`/`hXOutNe` feed the context-invariant step. -/
private theorem dssm_forward_traceSafeR (R : RoundingModel)
    (bounds : RegionBounds)
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hT : 0 < length) (hBS : BLOCK_SIZE ≤ batch_size * dim)
    (hXOutNe : x_ptr ≠ y_ptr) (s : BlockState)
    (hbs : ∀ (t : Fin length) (j : Fin BLOCK_SIZE),
      s.pids 0 * BLOCK_SIZE + j.val < batch_size * dim →
      s.pids 0 * BLOCK_SIZE + j.val < bounds s_ptr)
    (hbl : ∀ (t : Fin length) (j : Fin BLOCK_SIZE),
      s.pids 0 * BLOCK_SIZE + j.val < batch_size * dim →
      (s.pids 0 * BLOCK_SIZE + j.val) % dim < bounds lambda_ptr)
    (hbx : ∀ (t : Fin length) (j : Fin BLOCK_SIZE),
      s.pids 0 * BLOCK_SIZE + j.val < batch_size * dim →
      t.val * (batch_size * dim) + (s.pids 0 * BLOCK_SIZE + j.val)
        < bounds x_ptr)
    (hby : ∀ (t : Fin length) (j : Fin BLOCK_SIZE),
      s.pids 0 * BLOCK_SIZE + j.val < batch_size * dim →
      t.val * (batch_size * dim) + (s.pids 0 * BLOCK_SIZE + j.val)
        < bounds y_ptr) :
    ((diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr length batch_size
      dim BLOCK_SIZE).toAlgKernel).TraceSafeR R bounds s := by
  unfold Kernel.TraceSafeR
  rw [diag_ssm_forward_kernel_toAlg_body]
  unfold diagSsmForwardProjectedBody
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · exact dssm_preLoopSafeR R bounds s_ptr lambda_ptr batch_size dim
      BLOCK_SIZE s (fun j hj => hbs ⟨0, hT⟩ j hj)
      (fun j hj => hbl ⟨0, hT⟩ j hj)
  · intro s5 hs5
    rw [dssm_preLoop_castFree R s_ptr lambda_ptr batch_size dim BLOCK_SIZE
      s] at hs5
    have hCtx0 := diagSsmForwardLoopContextInvariant_init_of_preloop s s5
      s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE hs5
    refine Stmt.TraceSafeListR.cons_intro ?_
      (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR]
    refine dssm_forLoopTraceSafeR_inv R bounds "t" length
      (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim BLOCK_SIZE)
      (diagSsmForwardLoopContextInvariant s s_ptr x_ptr lambda_ptr y_ptr
        length batch_size dim BLOCK_SIZE) ?_ 0 s5 hCtx0
    intro c stt hc hP
    obtain ⟨hInv, hLambda, hCol, hMask, hXRead⟩ := hP
    refine ⟨dssm_bodySafeR R bounds x_ptr y_ptr batch_size dim BLOCK_SIZE s
      stt c hCol hMask (fun j hj => hbx ⟨c, hc⟩ j hj)
      (fun j hj => hby ⟨c, hc⟩ j hj), ?_⟩
    obtain ⟨st', hstep, hP'⟩ :=
      diagSsmForwardLoopContextInvariant_body_step_exists s stt s_ptr x_ptr
        lambda_ptr y_ptr batch_size dim BLOCK_SIZE c hc
        ⟨hInv, hLambda, hCol, hMask, hXRead⟩
        (dssm_outOffset_injective s batch_size dim BLOCK_SIZE length hBS)
        hXOutNe
    exact ⟨st', by
      rw [dssm_body_castFree R x_ptr y_ptr batch_size dim BLOCK_SIZE]
      exact hstep, hP'⟩

/-! ### The rounded Hoare triple (`hrun`) -/

/-- Termination, per-window values and the per-cell frame of the whole
scan under `execR R`, from an **arbitrary** launch state: the exact
`diagSsmForwardLoopContextInvariant` stack runs verbatim (the surface is
cast-free, so `execR R` collapses onto the exact stepper), extended with
the per-segment memory frames. -/
private theorem dssm_runR (R : RoundingModel)
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hBS : BLOCK_SIZE ≤ batch_size * dim) (hXOutNe : x_ptr ≠ y_ptr)
    (s₀ : BlockState) :
    ∃ sfin,
      execR R (diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr length
          batch_size dim BLOCK_SIZE).toAlgKernel s₀ = some sfin
      ∧ (∀ (t : Fin length) (j : Fin BLOCK_SIZE),
          active s₀ batch_size dim BLOCK_SIZE j →
          sfin.readMem y_ptr (timeOffset s₀ batch_size dim BLOCK_SIZE t.val j)
            = diagSsmForwardSpec s₀ s_ptr x_ptr lambda_ptr batch_size dim
                BLOCK_SIZE t.val j)
      ∧ (∀ r oo,
          (r ≠ y_ptr ∨ ∀ (t : Fin length) (j : Fin BLOCK_SIZE),
            active s₀ batch_size dim BLOCK_SIZE j →
            oo ≠ timeOffset s₀ batch_size dim BLOCK_SIZE t.val j) →
          sfin.mem r oo = s₀.mem r oo) := by
  have hInj : Function.Injective (fun idx : TileIndex [length, BLOCK_SIZE] =>
      diagSsmForwardOutOffset s₀ batch_size dim BLOCK_SIZE idx) :=
    dssm_outOffset_injective s₀ batch_size dim BLOCK_SIZE length hBS
  cases hPre : stepStmts (diagSsmForwardPreLoop s_ptr lambda_ptr batch_size
      dim BLOCK_SIZE) s₀ with
  | none =>
      unfold diagSsmForwardPreLoop at hPre
      simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.cop,
        NumericDType.mul, NumericDType.add, IntegralDType.mod,
        ComparableDType.lt, Option.bind] at hPre
  | some stPre =>
      have hCtx0 := diagSsmForwardLoopContextInvariant_init_of_preloop s₀
        stPre s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE
        hPre
      have hPreMem : stPre.mem = s₀.mem :=
        dssm_stepStmts_assigns_mem
          (diagSsmForwardPreLoop s_ptr lambda_ptr batch_size dim BLOCK_SIZE)
          (by
            intro stmt hst
            simp only [diagSsmForwardPreLoop, List.mem_cons,
              List.not_mem_nil, or_false] at hst
            rcases hst with rfl | rfl | rfl | rfl | rfl <;>
              exact ⟨_, _, _, _, rfl⟩)
          hPre
      obtain ⟨stFin, hFor, hP⟩ :=
        forLoop_inv (idx := "t") (n := length)
          (body := diagSsmForwardLoopBody x_ptr y_ptr batch_size dim
            BLOCK_SIZE)
          (P := fun t st =>
            diagSsmForwardLoopContextInvariant s₀ s_ptr x_ptr lambda_ptr
              y_ptr length batch_size dim BLOCK_SIZE t st ∧
            ∀ r oo,
              (r ≠ y_ptr ∨ ∀ (tf : Fin length) (j : Fin BLOCK_SIZE),
                active s₀ batch_size dim BLOCK_SIZE j →
                oo ≠ timeOffset s₀ batch_size dim BLOCK_SIZE tf.val j) →
              st.mem r oo = s₀.mem r oo)
          (s_init := stPre)
          ⟨hCtx0, fun r oo _ => by rw [hPreMem]⟩
          (fun t st ht hP => by
            obtain ⟨hCtx, hFrame⟩ := hP
            obtain ⟨st', hstep, hCtx'⟩ :=
              diagSsmForwardLoopContextInvariant_body_step_exists s₀ st
                s_ptr x_ptr lambda_ptr y_ptr batch_size dim BLOCK_SIZE t ht
                hCtx hInj hXOutNe
            refine ⟨st', hstep, hCtx', ?_⟩
            intro r oo hcond
            obtain ⟨hInv, hLambda, hCol, hMask, hXRead⟩ := hCtx
            rw [dssm_body_step_frame s₀ st st' s_ptr x_ptr lambda_ptr y_ptr
              batch_size dim BLOCK_SIZE t ht hInv.1 hLambda hCol hMask hXRead
              hstep r oo hcond]
            exact hFrame r oo hcond)
      obtain ⟨⟨hInvF, -, -, -, -⟩, hFrameF⟩ := hP
      refine ⟨stFin, ?_, ?_, ?_⟩
      · -- assemble the `execR` run through the cast-free collapses
        have hForR : stepStmtR R (Stmt.forLoop "t" length
            (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim BLOCK_SIZE))
            stPre = some stFin := by
          have h1 : stepStmtR R (Stmt.forLoop "t" length
              (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim BLOCK_SIZE))
              stPre
              = stepForLoopAuxR R "t" 0 length
                  (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim
                    BLOCK_SIZE) stPre := by
            simp only [stepStmtR]
          rw [h1, stepForLoopAuxR_castFree R _
              (dssm_body_castFree R x_ptr y_ptr batch_size dim BLOCK_SIZE)
              "t",
            ← stepForLoopAux.forLoop_unfold]
          exact hFor
        show execR R (diag_ssm_forward_kernel s_ptr x_ptr lambda_ptr y_ptr
          length batch_size dim BLOCK_SIZE).toAlgKernel s₀ = some stFin
        unfold execR
        rw [diag_ssm_forward_kernel_toAlg_body]
        unfold diagSsmForwardProjectedBody
        rw [stepStmtsR_append R
            (diagSsmForwardPreLoop s_ptr lambda_ptr batch_size dim BLOCK_SIZE)
            [Stmt.forLoop "t" length
              (diagSsmForwardLoopBody x_ptr y_ptr batch_size dim BLOCK_SIZE)]
            s₀,
          dssm_preLoop_castFree R s_ptr lambda_ptr batch_size dim BLOCK_SIZE
            s₀,
          hPre, Option.bind_some, stepStmtsR_cons_some hForR, stepStmtsR_nil]
      · intro t j hact
        exact hInvF.2 ((t, j, PUnit.unit) : TileIndex [length, BLOCK_SIZE])
          t.isLt hact
      · exact hFrameF

/-! ### The headline -/

/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `diag_ssm_forward_kernel` surface
implements, on its `StreamEmitMasked2DKernelIO₃` signature, the **ideal ℝ
diagonal-SSM scan** over the streamed tiles: emitted window `(t, j)` holds
the state after `t + 1` steps of `s ← s·Λ + x_u` (`u ≤ t`), seeded with
the static initial-state tile and folded with the static decay tile —
`diagSsmStreamState` is exact real arithmetic. The `s`/`Λ` channels are
the skin's degenerate static streams (windows ignore `t`); the grid is
1-D, so every window also ignores `pid₁` and the headline is ∀-`pid₁`
(the `bgmv_shrink` precedent, one grid rank down). The kernel has **zero
rounding events** (all loads, the scan arithmetic and the per-step raw
stores are at `.real`; there is no `castFloat`), so the skin's boundary
quantization degenerates: the readback's `R.round .real` is the identity
by the model's defining `round_real` — the ∀-`R` face holds via the
`RoundingModel` `.real` identity fields, not as a `.triv` special case.

Layer map: the surface is cast-free, so under `execR R` it collapses
verbatim onto the exact stepper and the proven
`diagSsmForwardLoopContextInvariant` stack above is reused unchanged; the
`⊨[R]` face adds the first bench `TraceSafeR` walk over a `forLoop` (a
private mirror of the library's `forRangeTraceSafeR_inv`), the per-cell
memory frame (`dssm_body_step_frame`), and the stream-lane spec bridge
(`dssm_streamState_eq_stateAfter`).

All three hypotheses are truth-forced, with provenance:

* `hT : 0 < length` — the static `s`/`Λ` windows are step-indexed
  (`Fin length`), so the skin's `read1`/`read2` bound groups are
  non-vacuous only when a step exists; at `length = 0` the kernel still
  issues the two pre-loop loads but the contract would supply no bounds
  for them. A 0-step scan has an empty output anyway.
* `hBS : BLOCK_SIZE ≤ batch_size·dim` — the tiling-geometry form of the
  exact headline `diag_ssm_forward_kernel_output_summary`'s `hOutInj`
  (full-grid output-window injectivity, which the loop invariant carries
  every previously-emitted window through later scatters with): with the
  lane offset below the timestep pitch, distinct `(t, j)` windows never
  collide. It holds for every real launch (the grid is
  `⌈batch_size·dim / BLOCK_SIZE⌉` programs over a `batch_size·dim`-wide
  lane space).
* `hXOutNe : x_ptr ≠ y_ptr` — inherited verbatim from the exact headline:
  step `t` stores into `y_ptr` **before** step `t+1` re-reads `x_ptr`; if
  the two aliased, later steps would stream already-overwritten inputs
  and the closed form would be false.

No `s_ptr`/`lambda_ptr` disalias is needed: both tiles are loaded into
registers before the loop, and the spec is pinned on the initial state.

Relation to the exact surface: the exact headline
`diag_ssm_forward_kernel_output_summary` (`Realizes_without_Rounding`)
above is retained unchanged; this `⊨[R]` face restates the same scan
content on the streaming emit skin, for every `R` at once (at the `.real`
grid the two faces carry the same exact cell). Both faces are kept per
the rounding-as-default doctrine. -/
specification diag_ssm_forward_io_correctness (R : RoundingModel)
    (s_ptr x_ptr lambda_ptr y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hT : 0 < length) (hBS : BLOCK_SIZE ≤ batch_size * dim)
    (hXOutNe : x_ptr ≠ y_ptr) :
    diagSsmForwardKernelIO s_ptr x_ptr lambda_ptr y_ptr length batch_size dim
        BLOCK_SIZE ⊨[R]
      fun _ _ ss ls xs t j =>
        diagSsmStreamState (ss t j) (ls t j) (fun u => xs u j) (t.val + 1) := by
  refine StreamEmitMasked2DKernelIO₃.ImplementsR.intro _ ?_ ?_ ?_
  · exact dssm_forward_flattenOk s_ptr x_ptr lambda_ptr y_ptr length
      batch_size dim BLOCK_SIZE
  · -- safety walk
    intro bounds s xs ys zs _hx _hy _hz hbr1 hbr2 hbr3 hbw
    simp only [diagSsmForwardKernelIO] at hbr1 hbr2 hbr3 hbw ⊢
    exact dssm_forward_traceSafeR R bounds s_ptr x_ptr lambda_ptr y_ptr
      length batch_size dim BLOCK_SIZE hT hBS hXOutNe s hbr1 hbr2 hbr3 hbw
  · -- the rounded Hoare triple
    intro s₀ xs ys zs _hundef hx hy hz
    simp only [diagSsmForwardKernelIO] at hx hy hz ⊢
    obtain ⟨sfin, hexec, hval, hframe⟩ :=
      dssm_runR R s_ptr x_ptr lambda_ptr y_ptr length batch_size dim
        BLOCK_SIZE hBS hXOutNe s₀
    refine ⟨sfin, hexec, ?_, ?_⟩
    · intro t j hj
      have hact : active s₀ batch_size dim BLOCK_SIZE j := hj
      have hss : s₀.readMem s_ptr (colOffset s₀ BLOCK_SIZE j) = xs t j :=
        hx t j hj
      have hls : s₀.readMem lambda_ptr (colOffset s₀ BLOCK_SIZE j % dim)
          = ys t j := hy t j hj
      have hzs : ∀ u : Fin length,
          s₀.readMem x_ptr (timeOffset s₀ batch_size dim BLOCK_SIZE u.val j)
            = zs u j := fun u => hz u j hj
      have hbridge := dssm_streamState_eq_stateAfter s₀ s_ptr x_ptr
        lambda_ptr batch_size dim BLOCK_SIZE xs ys zs t j hss hls hzs
        (t.val + 1) t.isLt
      have hv' : sfin.readMem y_ptr
          (t.val * (batch_size * dim) + (s₀.pids 0 * BLOCK_SIZE + j.val))
          = diagSsmStateAfter s₀ s_ptr x_ptr lambda_ptr batch_size dim
              BLOCK_SIZE j (t.val + 1) := hval t j hact
      rw [BlockState.readMemAs_real, hv', ← hbridge]
      simp only [RoundingModel.round_real_apply, FloatDType.ofReal]
      rfl
    · intro r oo hcond
      refine hframe r oo ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · exact Or.inr fun t j hact => hno t j hact

end IOFace

/-! ## The backward `⊨[R]` streaming headline (wave-5 S3 grouped emit genre)

Everything below is purely additive; the exact backward stack above and the
forward `IOFace` section are untouched. This is the first consumer of the
**grouped** per-step emit skin `StreamGroupedEmitMasked3DKernelIO`
(streaming genre, style S3): the reverse-time gradient scan owns one
in-loop store (`grad_x`, a genuine per-step window family) **plus two
post-loop terminal stores** (`grad_s`/`grad_Λ`), so the output side is a
`Fin 3`-indexed window-family group — a named-field emit skin would need
one field and one intro leg per channel.

Modeling notes:

* **The `if t > 0` branch.** The kernel's data-dependent previous-state
  load (`y[t-1]` when `t > 0`, else the initial state `s`) is a runtime
  `Stmt.ifThenElse` on the (register-computed) reverse counter. On the io
  face it becomes **two read channels with complementary step-indexed
  read masks**: the `y` channel reads window
  `(t-1)·(batch_size·dim)+lane` under `0 < t`, the `s` channel reads
  window `lane` under `t = 0` — together they pin exactly the cells the
  branch actually reads. On the proof side the `TraceSafeR` walk and the
  `execR` collapse resolve the guard per iteration from the concrete
  counter value (both arms are taken, at different iterations).
* **Terminal stores as a degenerate window family.** The two post-loop
  stores write step-independent windows; per the skin's docstring they
  are declared as window families whose windows ignore `t` and whose
  `writeMask` selects the single **designated step `t = 0`** (the reverse
  scan's last processed time step; any designated step would declare the
  same cell set, `0` is the convention because it exists for every
  nonempty stream). -/

section BackwardIOFace

open scoped VeriTile.Triton.StreamGroupedEmitMasked3DKernelIO

/-! ### IO signature -/

/-- **Streaming IO signature** of `diag_ssm_backward_kernel` on the grouped
per-step emit skin (S3; the grid is 1-D, so every window ignores
`pid₁`/`pid₂` and the headline is ∀-both). Step `t`, lane `j` of program
`pid₀` transcribes the kernel's pointer arithmetic verbatim
(`col_offsets = pid₀·BLOCK_SIZE + j`,
`offsets = t·(batch_size·dim) + col_offsets`):

Input channels (`Fin 4`):

* `0` — `grad_y_ptr` (streamed): window `offsets`, the per-iteration
  upstream-gradient load.
* `1` — `y_ptr` (the `t > 0` arm of the previous-state branch): window
  `(t-1)·(batch_size·dim) + col_offsets` (the kernel's
  `offsets - batch_size·dim`), read-active only when `0 < t`.
* `2` — `s_ptr` (the `t = 0` arm): window `col_offsets`, read-active only
  at `t = 0`.
* `3` — `lambda_ptr` (**static**: window ignores `t`): the pre-loop
  `lambda_ptr + col_offsets % dim` register-cached decay tile.

Output channels (`Fin 3`):

* `0` — `grad_x_ptr` (per-step): window `offsets`, the in-loop store.
* `1` — `grad_s_ptr` (**terminal**): window `col_offsets` (ignores `t`),
  write-active only at the designated step `t = 0`.
* `2` — `grad_lambda_ptr` (**terminal**): same designated-step
  convention, window `col_offsets`.

Every step-independent mask conjunct is the kernel's single `mask`:
`pid₀·BLOCK_SIZE + j < batch_size·dim`. All three stores are raw `.real`
stores, so `outDType` keeps the skin's `.real` default — no quantization
event. `bufs` lists every region the kernel touches exactly once, in
kernel-argument order. -/
def diagSsmBackwardKernelIO
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr
      grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    StreamGroupedEmitMasked3DKernelIO where
  kernel := diag_ssm_backward_kernel s_ptr lambda_ptr y_ptr grad_s_ptr
    grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
  nIn := 4
  nOut := 3
  bufs := [s_ptr, lambda_ptr, y_ptr, grad_s_ptr, grad_x_ptr,
    grad_lambda_ptr, grad_y_ptr]
  inp := fun i => match i with
    | ⟨0, _⟩ => grad_y_ptr
    | ⟨1, _⟩ => y_ptr
    | ⟨2, _⟩ => s_ptr
    | ⟨_ + 3, _⟩ => lambda_ptr
  out := fun o => match o with
    | ⟨0, _⟩ => grad_x_ptr
    | ⟨1, _⟩ => grad_s_ptr
    | ⟨_ + 2, _⟩ => grad_lambda_ptr
  T := length
  B := BLOCK_SIZE
  read := fun i p₀ _ _ t j => match i with
    | ⟨0, _⟩ => t.val * (batch_size * dim) + (p₀ * BLOCK_SIZE + j.val)
    | ⟨1, _⟩ => (t.val - 1) * (batch_size * dim) + (p₀ * BLOCK_SIZE + j.val)
    | ⟨2, _⟩ => p₀ * BLOCK_SIZE + j.val
    | ⟨_ + 3, _⟩ => (p₀ * BLOCK_SIZE + j.val) % dim
  readMask := fun i p₀ _ _ t j => match i with
    | ⟨0, _⟩ => p₀ * BLOCK_SIZE + j.val < batch_size * dim
    | ⟨1, _⟩ => 0 < t.val ∧ p₀ * BLOCK_SIZE + j.val < batch_size * dim
    | ⟨2, _⟩ => t.val = 0 ∧ p₀ * BLOCK_SIZE + j.val < batch_size * dim
    | ⟨_ + 3, _⟩ => p₀ * BLOCK_SIZE + j.val < batch_size * dim
  write := fun o p₀ _ _ t j => match o with
    | ⟨0, _⟩ => t.val * (batch_size * dim) + (p₀ * BLOCK_SIZE + j.val)
    | ⟨1, _⟩ => p₀ * BLOCK_SIZE + j.val
    | ⟨_ + 2, _⟩ => p₀ * BLOCK_SIZE + j.val
  writeMask := fun o p₀ _ _ t j => match o with
    | ⟨0, _⟩ => p₀ * BLOCK_SIZE + j.val < batch_size * dim
    | ⟨1, _⟩ => t.val = 0 ∧ p₀ * BLOCK_SIZE + j.val < batch_size * dim
    | ⟨_ + 2, _⟩ => t.val = 0 ∧ p₀ * BLOCK_SIZE + j.val < batch_size * dim

/-! ### The stream-level spec -/

/-- Stream-side reverse-scan gradient carry: `grad_s` after `k` reverse
iterations (iteration `u` processes time `reverseTime T u = T-1-u`),
folded over the `grad_y` stream with the static decay value `lam`
(`gs ← (grad_y[T-1-u] + gs)·lam`, seeded with `0`). The step input is
guarded by the stream length; the guard is always live on the bridge
below (only prefixes `k ≤ T` are ever evaluated). Mirrors the memory-side
recurrence `diagSsmBackwardGradSAfter` with the memory channels replaced
by the skin's per-lane stream values. -/
private noncomputable def dssmbStreamGradS {T : Nat}
    (gys : Fin T → ℝ) (lam : ℝ) : Nat → ℝ
  | 0 => 0
  | k + 1 =>
      ((if h : reverseTime T k < T then gys ⟨reverseTime T k, h⟩ else 0) +
        dssmbStreamGradS gys lam k) * lam

/-- Stream-side previous forward state consumed by reverse iteration `k`
(time `t = reverseTime T k`): the `y` stream at step `t` when `0 < t`
(the kernel's `y[t-1]` window is pinned at step `t`), else the `s` stream
at the designated step `t = 0` — the stream mirror of
`diagSsmBackwardPrevState`, resolving the kernel's `if t > 0` branch. -/
private noncomputable def dssmbStreamPrev {T : Nat}
    (ys ss : Fin T → ℝ) (k : Nat) : ℝ :=
  if h : reverseTime T k < T then
    (if 0 < reverseTime T k then ys ⟨reverseTime T k, h⟩
     else ss ⟨reverseTime T k, h⟩)
  else 0

/-- Stream-side `grad_Λ` accumulator after `k` reverse iterations
(`gΛ ← gΛ + (grad_y[T-1-u] + gs u)·prev u`) — the stream mirror of
`diagSsmBackwardGradLambdaAfter`. -/
private noncomputable def dssmbStreamGradLambda {T : Nat}
    (gys ys ss : Fin T → ℝ) (lam : ℝ) : Nat → ℝ
  | 0 => 0
  | k + 1 =>
      dssmbStreamGradLambda gys ys ss lam k +
        ((if h : reverseTime T k < T then gys ⟨reverseTime T k, h⟩ else 0) +
          dssmbStreamGradS gys lam k) * dssmbStreamPrev ys ss k

/-- The shared three-channel spec matcher of the backward headline (the
grouped skin's `f`, one constructor-head arm per output channel — the
single match point, no inline per-site matchers):

* channel `0` (`grad_x`, step `t`): the running `grad_s` after absorbing
  `grad_y[t]` — the upstream gradient at `t` plus the reverse **suffix**
  fold over times `> t` (`T-1-t` reverse iterations);
* channel `1` (`grad_s`, terminal): the full `T`-iteration reverse fold;
* channel `2` (`grad_Λ`, terminal): the full `T`-iteration accumulator
  over the `grad_y`/`y`/`s` streams.

The static decay channel is read at the ambient step (`xs 3 t j`; its
window ignores `t`, so every pinned step holds the same cell). -/
private noncomputable def diagSsmBackwardStreamSpec {T B : Nat}
    (xs : Fin 4 → Fin T → Fin B → ℝ) (o : Fin 3) (t : Fin T) (j : Fin B) :
    ℝ :=
  match o with
  | ⟨0, _⟩ =>
      xs 0 t j +
        dssmbStreamGradS (fun u => xs 0 u j) (xs 3 t j) (T - 1 - t.val)
  | ⟨1, _⟩ => dssmbStreamGradS (fun u => xs 0 u j) (xs 3 t j) T
  | ⟨_ + 2, _⟩ =>
      dssmbStreamGradLambda (fun u => xs 0 u j) (fun u => xs 1 u j)
        (fun u => xs 2 u j) (xs 3 t j) T

/-! ### The stream-lane spec bridges -/

/-- Per-lane bridge for the `grad_s` carry: under the skin's stream pins
(the static `Λ` window and the streamed `grad_y` windows), the
stream-side reverse fold **is** the memory-side recurrence
`diagSsmBackwardGradSAfter` on every prefix `u ≤ length`. -/
private theorem dssmb_streamGradS_eq (s₀ : BlockState)
    (lambda_ptr grad_y_ptr : RegionName) {length : Nat}
    (batch_size dim BLOCK_SIZE : Nat) (j : Fin BLOCK_SIZE)
    (lam : ℝ) (gys : Fin length → ℝ)
    (hlam : s₀.readMem lambda_ptr (colOffset s₀ BLOCK_SIZE j % dim) = lam)
    (hgy : ∀ u : Fin length,
      s₀.readMem grad_y_ptr
        (timeOffset s₀ batch_size dim BLOCK_SIZE u.val j) = gys u) :
    ∀ u : Nat, u ≤ length →
      dssmbStreamGradS gys lam u
        = diagSsmBackwardGradSAfter s₀ lambda_ptr grad_y_ptr batch_size dim
            BLOCK_SIZE length j u := by
  intro u
  induction u with
  | zero =>
      intro _
      simp [dssmbStreamGradS, diagSsmBackwardGradSAfter]
  | succ v ih =>
      intro hu
      have hv : v < length := hu
      have hrt : reverseTime length v < length := by
        unfold reverseTime
        omega
      have ihv := ih (Nat.le_of_lt hv)
      calc dssmbStreamGradS gys lam (v + 1)
          = (gys ⟨reverseTime length v, hrt⟩ +
              dssmbStreamGradS gys lam v) * lam := by
            simp [dssmbStreamGradS, hrt]
        _ = (gys ⟨reverseTime length v, hrt⟩ +
              diagSsmBackwardGradSAfter s₀ lambda_ptr grad_y_ptr batch_size
                dim BLOCK_SIZE length j v) * lam := by rw [ihv]
        _ = diagSsmBackwardGradSAfter s₀ lambda_ptr grad_y_ptr batch_size
              dim BLOCK_SIZE length j (v + 1) := by
            rw [← hgy ⟨reverseTime length v, hrt⟩, ← hlam]
            rfl

/-- Per-lane bridge for the previous-state select: under the branch pins
(`y` at steps `0 < u`, `s` at the designated step `u = 0`), the stream
select **is** the memory-side `diagSsmBackwardPrevState` at every
in-range iteration. -/
private theorem dssmb_streamPrev_eq (s₀ : BlockState)
    (s_ptr y_ptr : RegionName) {length : Nat}
    (batch_size dim BLOCK_SIZE : Nat) (j : Fin BLOCK_SIZE)
    (ys ss : Fin length → ℝ)
    (hy : ∀ u : Fin length, 0 < u.val →
      s₀.readMem y_ptr
          ((u.val - 1) * (batch_size * dim) + colOffset s₀ BLOCK_SIZE j)
        = ys u)
    (hs : ∀ u : Fin length, u.val = 0 →
      s₀.readMem s_ptr (colOffset s₀ BLOCK_SIZE j) = ss u)
    (v : Nat) (hv : v < length) :
    dssmbStreamPrev ys ss v
      = diagSsmBackwardPrevState s₀ s_ptr y_ptr batch_size dim BLOCK_SIZE
          length v j := by
  have hrt : reverseTime length v < length := by
    unfold reverseTime
    omega
  unfold dssmbStreamPrev
  rw [dif_pos hrt]
  by_cases hkprev : v < length - 1
  · have h0 : 0 < reverseTime length v :=
      (reverseTime_pos_iff length v hv).2 hkprev
    rw [if_pos h0,
      diagSsmBackwardPrevState_of_prev s₀ s_ptr y_ptr batch_size dim
        BLOCK_SIZE length v j hv hkprev,
      timeOffset_reverse_prev s₀ batch_size dim BLOCK_SIZE length v j
        hkprev,
      ← hy ⟨reverseTime length v, hrt⟩ h0]
    rfl
  · have h0 : reverseTime length v = 0 :=
      reverseTime_eq_zero_of_last length v hv hkprev
    rw [if_neg (show ¬ 0 < reverseTime length v by omega),
      diagSsmBackwardPrevState_of_last s₀ s_ptr y_ptr batch_size dim
        BLOCK_SIZE length v j hv hkprev,
      ← hs ⟨reverseTime length v, hrt⟩
        (show (⟨reverseTime length v, hrt⟩ : Fin length).val = 0 from h0)]

/-- Per-lane bridge for the `grad_Λ` accumulator: under all four channel
pins, the stream-side accumulator **is** the memory-side recurrence
`diagSsmBackwardGradLambdaAfter` on every prefix `u ≤ length`. -/
private theorem dssmb_streamGradLambda_eq (s₀ : BlockState)
    (s_ptr lambda_ptr y_ptr grad_y_ptr : RegionName) {length : Nat}
    (batch_size dim BLOCK_SIZE : Nat) (j : Fin BLOCK_SIZE)
    (lam : ℝ) (gys ys ss : Fin length → ℝ)
    (hlam : s₀.readMem lambda_ptr (colOffset s₀ BLOCK_SIZE j % dim) = lam)
    (hgy : ∀ u : Fin length,
      s₀.readMem grad_y_ptr
        (timeOffset s₀ batch_size dim BLOCK_SIZE u.val j) = gys u)
    (hy : ∀ u : Fin length, 0 < u.val →
      s₀.readMem y_ptr
          ((u.val - 1) * (batch_size * dim) + colOffset s₀ BLOCK_SIZE j)
        = ys u)
    (hs : ∀ u : Fin length, u.val = 0 →
      s₀.readMem s_ptr (colOffset s₀ BLOCK_SIZE j) = ss u) :
    ∀ u : Nat, u ≤ length →
      dssmbStreamGradLambda gys ys ss lam u
        = diagSsmBackwardGradLambdaAfter s₀ s_ptr lambda_ptr y_ptr
            grad_y_ptr batch_size dim BLOCK_SIZE length j u := by
  intro u
  induction u with
  | zero =>
      intro _
      simp [dssmbStreamGradLambda, diagSsmBackwardGradLambdaAfter]
  | succ v ih =>
      intro hu
      have hv : v < length := hu
      have hrt : reverseTime length v < length := by
        unfold reverseTime
        omega
      have ihv := ih (Nat.le_of_lt hv)
      have hgs := dssmb_streamGradS_eq s₀ lambda_ptr grad_y_ptr batch_size
        dim BLOCK_SIZE j lam gys hlam hgy v (Nat.le_of_lt hv)
      have hprev := dssmb_streamPrev_eq s₀ s_ptr y_ptr batch_size dim
        BLOCK_SIZE j ys ss hy hs v hv
      calc dssmbStreamGradLambda gys ys ss lam (v + 1)
          = dssmbStreamGradLambda gys ys ss lam v +
              (gys ⟨reverseTime length v, hrt⟩ +
                dssmbStreamGradS gys lam v) * dssmbStreamPrev ys ss v := by
            simp [dssmbStreamGradLambda, hrt]
        _ = diagSsmBackwardGradLambdaAfter s₀ s_ptr lambda_ptr y_ptr
              grad_y_ptr batch_size dim BLOCK_SIZE length j v +
              (gys ⟨reverseTime length v, hrt⟩ +
                diagSsmBackwardGradSAfter s₀ lambda_ptr grad_y_ptr
                  batch_size dim BLOCK_SIZE length j v) *
                diagSsmBackwardPrevState s₀ s_ptr y_ptr batch_size dim
                  BLOCK_SIZE length v j := by
            rw [ihv, hgs, hprev]
        _ = diagSsmBackwardGradLambdaAfter s₀ s_ptr lambda_ptr y_ptr
              grad_y_ptr batch_size dim BLOCK_SIZE length j (v + 1) := by
            rw [← hgy ⟨reverseTime length v, hrt⟩]
            rfl

/-! ### Window injectivity from the tiling geometry -/

/-- The backward twin of `dssm_outOffset_injective`: the full-grid
`grad_x`-window injectivity the exact backward invariant stack rides on
(`hOutInj` of `diag_ssm_backward_kernel_compute_correct`), derived from
the tiling-geometry hypothesis `BLOCK_SIZE ≤ batch_size·dim`. The window
map factors through the (injective on `Fin length`) reversal
`k ↦ length-1-k`, so distinct `(k, j)` never collide. -/
private theorem dssmb_gradXOffset_injective (st : BlockState)
    (batch_size dim BLOCK_SIZE length : Nat)
    (hBS : BLOCK_SIZE ≤ batch_size * dim) :
    Function.Injective (fun idx : TileIndex [length, BLOCK_SIZE] =>
      diagSsmBackwardGradXOffset st batch_size dim BLOCK_SIZE idx) := by
  intro a b h
  obtain ⟨ta, ja, ua⟩ := a
  obtain ⟨tb, jb, ub⟩ := b
  cases ua
  cases ub
  simp only [diagSsmBackwardGradXOffset, timeOffset, colOffset] at h
  have hja : ja.val < batch_size * dim := lt_of_lt_of_le ja.isLt hBS
  have hjb : jb.val < batch_size * dim := lt_of_lt_of_le jb.isLt hBS
  have hrt : reverseTime length ta.val = reverseTime length tb.val := by
    rcases Nat.lt_trichotomy (reverseTime length ta.val)
        (reverseTime length tb.val) with hlt | heq | hgt
    · exact absurd h (Nat.ne_of_lt (dssm_window_lt (batch_size * dim)
        (st.pids 0 * BLOCK_SIZE) _ _ ja.val jb.val hja hlt))
    · exact heq
    · exact absurd h.symm (Nat.ne_of_lt (dssm_window_lt (batch_size * dim)
        (st.pids 0 * BLOCK_SIZE) _ _ jb.val ja.val hjb hgt))
  have hta : ta = tb := by
    have h1 := ta.isLt
    have h2 := tb.isLt
    unfold reverseTime at hrt
    exact Fin.ext (by omega)
  subst hta
  have hj : ja = jb := Fin.ext (by omega)
  subst hj
  rfl

/-! ### Cast-free collapses and the covered fragment -/

/-- The backward pre-loop is cast-free (index arithmetic, one masked
`.real` load and the two `tl.zeros_like` register fills): it steps
identically under `stepStmtsR R`. -/
private theorem dssmb_preLoop_castFree (R : RoundingModel)
    (lambda_ptr : RegionName) (batch_size dim BLOCK_SIZE : Nat)
    (t : BlockState) :
    stepStmtsR R (diagSsmBackwardPreLoop lambda_ptr batch_size dim
        BLOCK_SIZE) t
      = stepStmts (diagSsmBackwardPreLoop lambda_ptr batch_size dim
        BLOCK_SIZE) t := by
  simp only [diagSsmBackwardPreLoop, stepStmtsR, stepStmts, stepStmtR,
    stepStmt, evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The backward loop body is cast-free **including the runtime
`if t > 0` branch and the in-loop masked `.real` store**: both `stepStmtR`
and `stepStmt` evaluate the same guard and recurse into the same
cast-free branch lists, and a `.real`-typed store delegates to the exact
write, so the whole body steps identically under `stepStmtsR R`. -/
private theorem dssmb_body_castFree (R : RoundingModel)
    (s_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (t : BlockState) :
    stepStmtsR R (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr
        length batch_size dim BLOCK_SIZE) t
      = stepStmts (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr
        grad_y_ptr length batch_size dim BLOCK_SIZE) t := by
  simp only [diagSsmBackwardLoopBody, stepStmtsR, stepStmts, stepStmtR,
    stepStmt, evalOpR.eq_def, evalOp.eq_def, BlockState.writeMemTypedR]
  rfl

/-- The two terminal stores are cast-free masked `.real` stores: the
post-loop steps identically under `stepStmtsR R`. -/
private theorem dssmb_postLoop_castFree (R : RoundingModel)
    (grad_s_ptr grad_lambda_ptr : RegionName) (BLOCK_SIZE : Nat)
    (t : BlockState) :
    stepStmtsR R (diagSsmBackwardPostLoop grad_s_ptr grad_lambda_ptr
        BLOCK_SIZE) t
      = stepStmts (diagSsmBackwardPostLoop grad_s_ptr grad_lambda_ptr
        BLOCK_SIZE) t := by
  simp only [diagSsmBackwardPostLoop, stepStmtsR, stepStmts, stepStmtR,
    stepStmt, evalOpR.eq_def, evalOp.eq_def, BlockState.writeMemTypedR]
  rfl

/-- The backward surface sits inside the flat-memory bridge's covered
fragment (`FlattenOk`; the `forRange` clause recurses into the cast-free
body, the `ifThenElse` clause into both branch lists). -/
private theorem dssmb_backward_flattenOk
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr
      grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    ((diag_ssm_backward_kernel s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr
      grad_lambda_ptr grad_y_ptr length batch_size dim
      BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [diag_ssm_backward_kernel_toAlg_body]
  simp [diagSsmBackwardProjectedBody, diagSsmBackwardPreLoop,
    diagSsmBackwardLoopBody, diagSsmBackwardPostLoop, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### Cell-level memory frames -/

/-- **Cell-level frame of one reverse-scan iteration** (the `mem` twin of
`diagSsmBackwardLoopBody_step_preserves_context`): from the
context-invariant register pins, one body iteration leaves every cell off
the step-`t` active `grad_x` write window untouched (both branch arms —
the branch only chooses which *load* feeds the registers). -/
private theorem dssmb_body_step_frame {length : Nat}
    (st0 st st' : BlockState)
    (s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    (batch_size dim BLOCK_SIZE k : Nat) (hk : k < length)
    (hLambda : st.regs .real [BLOCK_SIZE] "Lambda" =
      some { data := fun idx : TileIndex [BLOCK_SIZE] =>
        if active st0 batch_size dim BLOCK_SIZE idx.1 then
          some (st0.readMem lambda_ptr
            (IntegralDType.nat.mod (colOffset st0 BLOCK_SIZE idx.1) dim))
        else
          some 0 })
    (hGradS : st.regs .real [BLOCK_SIZE] "grad_s" =
      some (diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr batch_size
        dim BLOCK_SIZE length k))
    (hGradLambda : st.regs .real [BLOCK_SIZE] "grad_Lambda" =
      some (diagSsmBackwardGradLambdaTile st0 s_ptr lambda_ptr y_ptr
        grad_y_ptr batch_size dim BLOCK_SIZE length k))
    (hCol : st.regs .nat [BLOCK_SIZE] "col_offsets" =
      some { data := fun idx : TileIndex [BLOCK_SIZE] =>
        colOffset st0 BLOCK_SIZE idx.1 })
    (hMask : st.regs .bool [BLOCK_SIZE] "mask" =
      some { data := fun idx : TileIndex [BLOCK_SIZE] =>
        active st0 batch_size dim BLOCK_SIZE idx.1 })
    (hSRead : ∀ offset, st.readMem s_ptr offset = st0.readMem s_ptr offset)
    (hYRead : ∀ offset, st.readMem y_ptr offset = st0.readMem y_ptr offset)
    (hGradYRead :
      ∀ offset, st.readMem grad_y_ptr offset = st0.readMem grad_y_ptr offset)
    (hLambdaRead :
      ∀ offset, st.readMem lambda_ptr offset = st0.readMem lambda_ptr offset)
    (hStep :
      stepStmts (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr
        length batch_size dim BLOCK_SIZE)
        (st.setReg "i" .nat [] (Tile.scalar k)) = some st')
    (r : RegionName) (oo : Nat)
    (hcond : r ≠ grad_x_ptr ∨ ∀ (tf : Fin length) (j : Fin BLOCK_SIZE),
      active st0 batch_size dim BLOCK_SIZE j →
      oo ≠ timeOffset st0 batch_size dim BLOCK_SIZE tf.val j) :
    st'.mem r oo = st.mem r oo := by
  unfold diagSsmBackwardLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hLambda, hGradS,
    hGradLambda, hCol, hMask, hSRead, hYRead, hGradYRead, hLambdaRead,
    Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
    NumericDType.sub, IntegralDType.mod, ComparableDType.gt, Option.bind,
    timeOffset, active, colOffset, diagSsmBackwardGradSTile_succ,
    diagSsmBackwardGradLambdaTile_succ, diagSsmBackwardPrevState_of_prev,
    diagSsmBackwardPrevState_of_last] at hStep
  by_cases hkprev : k < length - 1
  <;> simp [hkprev] at hStep
  <;> rw [hGradS] at hStep
  <;> simp [BlockState.setReg, hGradLambda, hLambda, hCol, hMask,
    hkprev] at hStep
  <;> symm at hStep
  <;> subst st'
  all_goals
    refine Eq.trans (dssm_foldl_writeMem_preserve_cell _ _ _ r oo _ _ ?_) rfl
    intro lane _ hPk hbad
    rcases hcond with hne | hno
    · exact hne hbad.1
    · exact hno ⟨length - 1 - k, by omega⟩ lane.1 hPk hbad.2

/-- **Cell-level frame of the two terminal scatters**: from the loop-exit
register pins, the post-loop leaves every cell off the two active
`col_offsets` windows untouched. -/
private theorem dssmb_postLoop_frame
    (st0 st st' : BlockState)
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_lambda_ptr grad_y_ptr :
      RegionName)
    (batch_size dim BLOCK_SIZE length : Nat)
    (hGradS : st.regs .real [BLOCK_SIZE] "grad_s" =
      some (diagSsmBackwardGradSTile st0 lambda_ptr grad_y_ptr batch_size
        dim BLOCK_SIZE length length))
    (hGradLambda : st.regs .real [BLOCK_SIZE] "grad_Lambda" =
      some (diagSsmBackwardGradLambdaTile st0 s_ptr lambda_ptr y_ptr
        grad_y_ptr batch_size dim BLOCK_SIZE length length))
    (hCol : st.regs .nat [BLOCK_SIZE] "col_offsets" =
      some { data := fun idx : TileIndex [BLOCK_SIZE] =>
        colOffset st0 BLOCK_SIZE idx.1 })
    (hMask : st.regs .bool [BLOCK_SIZE] "mask" =
      some { data := fun idx : TileIndex [BLOCK_SIZE] =>
        active st0 batch_size dim BLOCK_SIZE idx.1 })
    (hStep :
      stepStmts (diagSsmBackwardPostLoop grad_s_ptr grad_lambda_ptr
        BLOCK_SIZE) st = some st')
    (r : RegionName) (oo : Nat)
    (hcs : r ≠ grad_s_ptr ∨ ∀ j : Fin BLOCK_SIZE,
      active st0 batch_size dim BLOCK_SIZE j →
      oo ≠ colOffset st0 BLOCK_SIZE j)
    (hcl : r ≠ grad_lambda_ptr ∨ ∀ j : Fin BLOCK_SIZE,
      active st0 batch_size dim BLOCK_SIZE j →
      oo ≠ colOffset st0 BLOCK_SIZE j) :
    st'.mem r oo = st.mem r oo := by
  unfold diagSsmBackwardPostLoop at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hGradS, hGradLambda,
    hCol, hMask, Option.bind] at hStep
  subst st'
  refine Eq.trans (dssm_foldl_writeMem_preserve_cell _ _ _ r oo _ _ ?_)
    (Eq.trans (dssm_foldl_writeMem_preserve_cell _ _ _ r oo _ _ ?_) rfl)
  · intro lane _ hPk hbad
    rcases hcl with hne | hno
    · exact hne hbad.1
    · exact hno lane.1 hPk hbad.2
  · intro lane _ hPk hbad
    rcases hcs with hne | hno
    · exact hne hbad.1
    · exact hno lane.1 hPk hbad.2

/-! ### The `TraceSafeR` walk

The loop is a `forRange`, so the library's
`Stmt.forRangeTraceSafeR_inv` drives it directly (no private mirror
needed — the forward section's `dssm_forLoopTraceSafeR_inv` was the
`forLoop` sibling). The new ingredient is the runtime `ifThenElse`: the
guard is evaluated concretely from the iteration counter and the walk
proceeds per branch. -/

/-- `evalOpR` = `evalOp` + concrete value of the reverse counter op
`t = length - 1 - i`. -/
private theorem dssmb_t_evalR (R : RoundingModel) (length k : Nat)
    (s : BlockState)
    (hi : s.regs .nat [] "i" = some (Tile.scalar k)) :
    evalOpR R (Op.sub NumericDType.nat Broadcast.nil
        (Op.sub NumericDType.nat Broadcast.nil (Op.constNat length)
          (Op.constNat 1))
        (Op.ref .nat [] "i")) s
      = some (Tile.scalar (length - 1 - k)) := by
  have h : evalOpR R (Op.sub NumericDType.nat Broadcast.nil
        (Op.sub NumericDType.nat Broadcast.nil (Op.constNat length)
          (Op.constNat 1))
        (Op.ref .nat [] "i")) s
      = evalOp (Op.sub NumericDType.nat Broadcast.nil
        (Op.sub NumericDType.nat Broadcast.nil (Op.constNat length)
          (Op.constNat 1))
        (Op.ref .nat [] "i")) s := by
    simp only [evalOpR.eq_def, evalOp.eq_def]
  rw [h, evalOp_sub, evalOp_sub, evalOp_constNat, evalOp_constNat,
    evalOp_ref, hi]
  rfl

/-- `evalOpR` = `evalOp` + concrete value of the branch guard `t > 0`. -/
private theorem dssmb_guard_evalR (R : RoundingModel) (s : BlockState)
    (tv : Nat)
    (ht : s.regs .nat [] "t" = some (Tile.scalar tv)) :
    evalOpR R (Op.gt ComparableDType.nat Broadcast.nil
        (Op.ref .nat [] "t") (Op.constNat 0)) s
      = some (Tile.scalar (decide (0 < tv))) := by
  have h : evalOpR R (Op.gt ComparableDType.nat Broadcast.nil
        (Op.ref .nat [] "t") (Op.constNat 0)) s
      = evalOp (Op.gt ComparableDType.nat Broadcast.nil
        (Op.ref .nat [] "t") (Op.constNat 0)) s := by
    simp only [evalOpR.eq_def, evalOp.eq_def]
  rw [h, evalOp_gt, evalOp_ref, ht, evalOp_constNat]
  rfl

/-- `evalOpR` = `evalOp` + concrete value of the previous-state address
op `offsets - batch_size·dim`. -/
private theorem dssmb_prevAddr_evalR (R : RoundingModel)
    (D BLOCK_SIZE : Nat) (s : BlockState) (co : Fin BLOCK_SIZE → Nat)
    (hco : s.regs .nat [BLOCK_SIZE] "offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] => co idx.1 }) :
    evalOpR R (Op.sub NumericDType.nat Broadcast.scalarR
        (Op.ref .nat [BLOCK_SIZE] "offsets") (Op.constNat D)) s
      = some
        { data := fun idx : TileIndex [BLOCK_SIZE] => co idx.1 - D } := by
  have h : evalOpR R (Op.sub NumericDType.nat Broadcast.scalarR
        (Op.ref .nat [BLOCK_SIZE] "offsets") (Op.constNat D)) s
      = evalOp (Op.sub NumericDType.nat Broadcast.scalarR
        (Op.ref .nat [BLOCK_SIZE] "offsets") (Op.constNat D)) s := by
    simp only [evalOpR.eq_def, evalOp.eq_def]
  rw [h, evalOp_sub, evalOp_ref, hco, evalOp_constNat]
  rfl

/-- `TraceSafeR` introduction for a scalar-guarded `ifThenElse` whose
guard evaluates to a known Boolean: prove the taken branch safe per guard
value. -/
private theorem dssmb_ifte_safeR {R : RoundingModel}
    {bounds : RegionBounds} {c : Op .bool []} {tb eb : List Stmt}
    {s : BlockState} {b : Bool}
    (hc : evalOpR R c s = some (Tile.scalar b))
    (hcsafe : c.SafeAtR R bounds s)
    (ht : b = Bool.true → Stmt.TraceSafeListR R bounds tb s)
    (he : b = Bool.false → Stmt.TraceSafeListR R bounds eb s) :
    Stmt.TraceSafeR R bounds (.ifThenElse c tb eb) s := by
  simp only [Stmt.TraceSafeR]
  refine ⟨hcsafe, ?_⟩
  rw [hc]
  cases b
  · simpa using he rfl
  · simpa using ht rfl

/-- Successor inversion for a scalar-guarded `ifThenElse` under
`stepStmtR R`: the successor is the run of the taken branch. -/
private theorem dssmb_ifte_stepR_inv {R : RoundingModel}
    {c : Op .bool []} {tb eb : List Stmt} {s s' : BlockState} {b : Bool}
    (hc : evalOpR R c s = some (Tile.scalar b))
    (h : stepStmtR R (.ifThenElse c tb eb) s = some s') :
    (b = Bool.true ∧ stepStmtsR R tb s = some s') ∨
      (b = Bool.false ∧ stepStmtsR R eb s = some s') := by
  simp only [stepStmtR, hc, Option.bind] at h
  cases b
  · exact Or.inr ⟨rfl, by simpa using h⟩
  · exact Or.inl ⟨rfl, by simpa using h⟩

/-- Successor inversion for a singleton assign list under `stepStmtsR R`
(the branch bodies of the previous-state `ifThenElse`): the successor is
a `setReg` of some evaluated value. -/
private theorem dssmb_stepsR_singleton_assign_inv {R : RoundingModel}
    {dtype : TileDType} {shape : TileShape} {name : RegName}
    {e : Op dtype shape} {s s' : BlockState}
    (h : stepStmtsR R [.assign dtype shape name e] s = some s') :
    ∃ v, s' = s.setReg name dtype shape v := by
  cases hv : stepStmtR R (.assign dtype shape name e) s with
  | none =>
      simp only [stepStmtsR, hv] at h
      simp at h
  | some s1 =>
      rw [stepStmtsR_cons_some hv, stepStmtsR_nil] at h
      obtain rfl := Option.some.inj h
      obtain ⟨v, -, rfl⟩ := stepStmtR_assign_inv hv
      exact ⟨v, rfl⟩

/-- `TraceSafeListR` of the backward pre-loop: three register-only index
assigns, the masked static `Λ` load (active lanes are the skin's
`t`-independent channel-3 window — in bounds by its read-window bound),
and the two `tl.zeros_like` register fills. -/
private theorem dssmb_preLoopSafeR (R : RoundingModel)
    (bounds : RegionBounds) (lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat) (s : BlockState)
    (hbl : ∀ j : Fin BLOCK_SIZE,
      s.pids 0 * BLOCK_SIZE + j.val < batch_size * dim →
      (s.pids 0 * BLOCK_SIZE + j.val) % dim < bounds lambda_ptr) :
    Stmt.TraceSafeListR R bounds
      (diagSsmBackwardPreLoop lambda_ptr batch_size dim BLOCK_SIZE) s := by
  unfold diagSsmBackwardPreLoop
  -- col_idx
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t1 ht1 => ?_)
  rw [stepStmtR_assign_eq_some (dssm_colIdx_evalR R BLOCK_SIZE s)] at ht1
  obtain rfl := Option.some.inj ht1
  set q1 := s.setReg "col_idx" .nat [] (Tile.scalar (s.pids 0 * BLOCK_SIZE))
    with hq1
  have hci1 : q1.regs .nat [] "col_idx"
      = some (Tile.scalar (s.pids 0 * BLOCK_SIZE)) := by simp [hq1]
  -- col_offsets
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t2 ht2 => ?_)
  rw [stepStmtR_assign_eq_some
    (dssm_colOffsets_evalR R BLOCK_SIZE q1 (s.pids 0 * BLOCK_SIZE)
      hci1)] at ht2
  obtain rfl := Option.some.inj ht2
  set q2 := q1.setReg "col_offsets" .nat [BLOCK_SIZE]
    { data := fun idx : TileIndex [BLOCK_SIZE] =>
        s.pids 0 * BLOCK_SIZE + idx.1.val } with hq2
  have hco2 : q2.regs .nat [BLOCK_SIZE] "col_offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          s.pids 0 * BLOCK_SIZE + idx.1.val } := by simp [hq2]
  -- mask
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t3 ht3 => ?_)
  rw [stepStmtR_assign_eq_some (dssm_mask_evalR R batch_size dim BLOCK_SIZE
    q2 (fun i => s.pids 0 * BLOCK_SIZE + i.val) hco2)] at ht3
  obtain rfl := Option.some.inj ht3
  set q3 := q2.setReg "mask" .bool [BLOCK_SIZE]
    { data := fun idx : TileIndex [BLOCK_SIZE] =>
        decide (s.pids 0 * BLOCK_SIZE + idx.1.val < batch_size * dim) }
    with hq3
  have hco3 : q3.regs .nat [BLOCK_SIZE] "col_offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          s.pids 0 * BLOCK_SIZE + idx.1.val } := by
    rw [hq3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("col_offsets" : RegName) ≠ "mask" by decide)]
    exact hco2
  have hmask3 : q3.regs .bool [BLOCK_SIZE] "mask"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          decide (s.pids 0 * BLOCK_SIZE + idx.1.val < batch_size * dim) } := by
    simp [hq3]
  -- the masked Λ load
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t4 ht4 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
      and_true, true_and, and_self]
    intro offsets hoffsets idx hactive
    rw [dssm_lambdaAddr_evalR R dim BLOCK_SIZE q3
      (fun i => s.pids 0 * BLOCK_SIZE + i.val) hco3] at hoffsets
    obtain rfl := Option.some.inj hoffsets
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hmask3] at hm
    obtain rfl := Option.some.inj hm
    have hlt : s.pids 0 * BLOCK_SIZE + idx.1.val < batch_size * dim := by
      simpa using hmi
    simpa [Region.cast_id] using hbl idx.1 hlt
  · -- the two zeros_like register fills
    exact Stmt.TraceSafeListR.of_forall _ _ (fun stmt hst s' => by
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
      rcases hst with rfl | rfl <;>
        simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])

/-- Per-iteration `TraceSafeListR` for the reverse-scan body: the `t` and
`offsets` assigns are register-only; the masked `grad_y` load, the
branch-taken previous-state load (`y` when `t > 0`, `s` when `t = 0` —
the two `ifThenElse` arms, resolved from the concrete counter) and the
masked `grad_x` store have active lanes bounded by the corresponding skin
windows. -/
private theorem dssmb_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (s_ptr y_ptr grad_x_ptr grad_y_ptr : RegionName)
    {length : Nat} (batch_size dim BLOCK_SIZE : Nat)
    (s0 st : BlockState) (k : Nat) (hk : k < length)
    (hCol : st.regs .nat [BLOCK_SIZE] "col_offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset s0 BLOCK_SIZE idx.1 })
    (hMask : st.regs .bool [BLOCK_SIZE] "mask"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active s0 batch_size dim BLOCK_SIZE idx.1 })
    (hbgy : ∀ j : Fin BLOCK_SIZE, active s0 batch_size dim BLOCK_SIZE j →
      timeOffset s0 batch_size dim BLOCK_SIZE (reverseTime length k) j
        < bounds grad_y_ptr)
    (hby : k < length - 1 → ∀ j : Fin BLOCK_SIZE,
      active s0 batch_size dim BLOCK_SIZE j →
      timeOffset s0 batch_size dim BLOCK_SIZE (reverseTime length k - 1) j
        < bounds y_ptr)
    (hbs : ¬ k < length - 1 → ∀ j : Fin BLOCK_SIZE,
      active s0 batch_size dim BLOCK_SIZE j →
      colOffset s0 BLOCK_SIZE j < bounds s_ptr)
    (hbgx : ∀ j : Fin BLOCK_SIZE, active s0 batch_size dim BLOCK_SIZE j →
      timeOffset s0 batch_size dim BLOCK_SIZE (reverseTime length k) j
        < bounds grad_x_ptr) :
    Stmt.TraceSafeListR R bounds
      (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr length
        batch_size dim BLOCK_SIZE)
      (st.setReg "i" .nat [] (Tile.scalar k)) := by
  unfold diagSsmBackwardLoopBody
  have hi0 : (st.setReg "i" .nat [] (Tile.scalar k)).regs .nat [] "i"
      = some (Tile.scalar k) := BlockState.setReg_same _ _ _ _ _
  have hco0 : (st.setReg "i" .nat [] (Tile.scalar k)).regs .nat
      [BLOCK_SIZE] "col_offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset s0 BLOCK_SIZE idx.1 } := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("col_offsets" : RegName) ≠ "i" by decide)]
    exact hCol
  have hmask0 : (st.setReg "i" .nat [] (Tile.scalar k)).regs .bool
      [BLOCK_SIZE] "mask"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active s0 batch_size dim BLOCK_SIZE idx.1 } := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("mask" : RegName) ≠ "i" by decide)]
    exact hMask
  -- t
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t1 ht1 => ?_)
  rw [stepStmtR_assign_eq_some (dssmb_t_evalR R length k _ hi0)] at ht1
  obtain rfl := Option.some.inj ht1
  set q1 := (st.setReg "i" .nat [] (Tile.scalar k)).setReg "t" .nat []
    (Tile.scalar (length - 1 - k)) with hq1
  have ht1' : q1.regs .nat [] "t"
      = some (Tile.scalar (length - 1 - k)) := by simp [hq1]
  have hco1 : q1.regs .nat [BLOCK_SIZE] "col_offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset s0 BLOCK_SIZE idx.1 } := by
    rw [hq1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("col_offsets" : RegName) ≠ "t" by decide)]
    exact hco0
  have hmask1 : q1.regs .bool [BLOCK_SIZE] "mask"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active s0 batch_size dim BLOCK_SIZE idx.1 } := by
    rw [hq1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("mask" : RegName) ≠ "t" by decide)]
    exact hmask0
  -- offsets
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t2 ht2 => ?_)
  rw [stepStmtR_assign_eq_some (dssm_offsets_evalR R batch_size dim
    BLOCK_SIZE q1 (length - 1 - k) (fun i => colOffset s0 BLOCK_SIZE i)
    ht1' hco1)] at ht2
  obtain rfl := Option.some.inj ht2
  set q2 := q1.setReg "offsets" .nat [BLOCK_SIZE]
    { data := fun idx : TileIndex [BLOCK_SIZE] =>
        (length - 1 - k) * (batch_size * dim) +
          colOffset s0 BLOCK_SIZE idx.1 } with hq2
  have hoffs2 : q2.regs .nat [BLOCK_SIZE] "offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          (length - 1 - k) * (batch_size * dim) +
            colOffset s0 BLOCK_SIZE idx.1 } := by simp [hq2]
  have ht2' : q2.regs .nat [] "t"
      = some (Tile.scalar (length - 1 - k)) := by
    rw [hq2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("t" : RegName) ≠ "offsets" by decide)]
    exact ht1'
  have hco2 : q2.regs .nat [BLOCK_SIZE] "col_offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset s0 BLOCK_SIZE idx.1 } := by
    rw [hq2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("col_offsets" : RegName) ≠ "offsets" by decide)]
    exact hco1
  have hmask2 : q2.regs .bool [BLOCK_SIZE] "mask"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active s0 batch_size dim BLOCK_SIZE idx.1 } := by
    rw [hq2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("mask" : RegName) ≠ "offsets" by decide)]
    exact hmask1
  -- the masked grad_y load
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t3 ht3 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
      and_true, true_and, and_self]
    intro offsets hoffsets idx hactive
    rw [evalOpR_ref, hoffs2] at hoffsets
    obtain rfl := Option.some.inj hoffsets
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hmask2] at hm
    obtain rfl := Option.some.inj hm
    have hact : active s0 batch_size dim BLOCK_SIZE idx.1 := by
      simpa using hmi
    simpa [Region.cast_id, timeOffset, reverseTime, colOffset]
      using hbgy idx.1 hact
  · obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv ht3
    set q3 := q2.setReg "grad_y" .real [BLOCK_SIZE] v3 with hq3
    have ht3' : q3.regs .nat [] "t"
        = some (Tile.scalar (length - 1 - k)) := by
      rw [hq3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("t" : RegName) ≠ "grad_y" by decide)]
      exact ht2'
    have hoffs3 : q3.regs .nat [BLOCK_SIZE] "offsets"
        = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
            (length - 1 - k) * (batch_size * dim) +
              colOffset s0 BLOCK_SIZE idx.1 } := by
      rw [hq3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offsets" : RegName) ≠ "grad_y" by decide)]
      exact hoffs2
    have hco3 : q3.regs .nat [BLOCK_SIZE] "col_offsets"
        = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
            colOffset s0 BLOCK_SIZE idx.1 } := by
      rw [hq3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("col_offsets" : RegName) ≠ "grad_y" by decide)]
      exact hco2
    have hmask3 : q3.regs .bool [BLOCK_SIZE] "mask"
        = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
            active s0 batch_size dim BLOCK_SIZE idx.1 } := by
      rw [hq3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("mask" : RegName) ≠ "grad_y" by decide)]
      exact hmask2
    -- the shared post-branch tail (register updates + the grad_x store),
    -- proved once for whatever value the taken branch loaded into `s`
    have htail : ∀ v : Tile .real [BLOCK_SIZE],
        Stmt.TraceSafeListR R bounds
          [ .assign .real [BLOCK_SIZE] "grad_s"
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
          (q3.setReg "s" .real [BLOCK_SIZE] v) := by
      intro v
      have hoffsP : (q3.setReg "s" .real [BLOCK_SIZE] v).regs .nat
          [BLOCK_SIZE] "offsets"
          = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
              (length - 1 - k) * (batch_size * dim) +
                colOffset s0 BLOCK_SIZE idx.1 } := by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("offsets" : RegName) ≠ "s" by decide)]
        exact hoffs3
      have hmaskP : (q3.setReg "s" .real [BLOCK_SIZE] v).regs .bool
          [BLOCK_SIZE] "mask"
          = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
              active s0 batch_size dim BLOCK_SIZE idx.1 } := by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
          (show ("mask" : RegName) ≠ "s" by decide)]
        exact hmask3
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun p1 hp1 => ?_)
      obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv hp1
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun p2 hp2 => ?_)
      obtain ⟨v6, -, rfl⟩ := stepStmtR_assign_inv hp2
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun p3 hp3 => ?_)
      obtain ⟨v7, -, rfl⟩ := stepStmtR_assign_inv hp3
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun p4 hp4 => ?_)
      obtain ⟨v8, -, rfl⟩ := stepStmtR_assign_inv hp4
      have hoffsF : (((((q3.setReg "s" .real [BLOCK_SIZE] v).setReg
          "grad_s" .real [BLOCK_SIZE] v5).setReg
          "grad_x" .real [BLOCK_SIZE] v6).setReg
          "grad_Lambda" .real [BLOCK_SIZE] v7).setReg
          "grad_s" .real [BLOCK_SIZE] v8).regs .nat [BLOCK_SIZE] "offsets"
          = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
              (length - 1 - k) * (batch_size * dim) +
                colOffset s0 BLOCK_SIZE idx.1 } := by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("offsets" : RegName) ≠ "grad_s" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("offsets" : RegName) ≠ "grad_Lambda" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("offsets" : RegName) ≠ "grad_x" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("offsets" : RegName) ≠ "grad_s" by decide)]
        exact hoffsP
      have hmaskF : (((((q3.setReg "s" .real [BLOCK_SIZE] v).setReg
          "grad_s" .real [BLOCK_SIZE] v5).setReg
          "grad_x" .real [BLOCK_SIZE] v6).setReg
          "grad_Lambda" .real [BLOCK_SIZE] v7).setReg
          "grad_s" .real [BLOCK_SIZE] v8).regs .bool [BLOCK_SIZE] "mask"
          = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
              active s0 batch_size dim BLOCK_SIZE idx.1 } := by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("mask" : RegName) ≠ "grad_s" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("mask" : RegName) ≠ "grad_Lambda" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("mask" : RegName) ≠ "grad_x" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("mask" : RegName) ≠ "grad_s" by decide)]
        exact hmaskP
      refine Stmt.TraceSafeListR.cons_intro ?_
        (fun _ _ => Stmt.TraceSafeListR.nil_intro)
      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
        MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
        memAccessActiveAddressSafeR, and_true, true_and, and_self]
      intro offsets hoffsets idx hactive
      rw [evalOpR_ref, hoffsF] at hoffsets
      obtain rfl := Option.some.inj hoffsets
      obtain ⟨masks, hm, hmi⟩ := hactive
      rw [evalOpR_ref, hmaskF] at hm
      obtain rfl := Option.some.inj hm
      have hact : active s0 batch_size dim BLOCK_SIZE idx.1 := by
        simpa using hmi
      simpa [Region.cast_id, timeOffset, reverseTime, colOffset]
        using hbgx idx.1 hact
    -- the branch itself
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun q4 hq4 => ?_)
    · refine dssmb_ifte_safeR
        (dssmb_guard_evalR R q3 (length - 1 - k) ht3')
        (by simp [Op.SafeAtR.eq_def]) ?_ ?_
      · intro hb
        have hkprev : k < length - 1 := by
          have := of_decide_eq_true hb
          omega
        refine Stmt.TraceSafeListR.cons_intro ?_
          (fun _ _ => Stmt.TraceSafeListR.nil_intro)
        simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
          MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
          and_true, true_and, and_self]
        intro offsets hoffsets idx hactive
        rw [dssmb_prevAddr_evalR R (batch_size * dim) BLOCK_SIZE q3
          (fun i => (length - 1 - k) * (batch_size * dim) +
            colOffset s0 BLOCK_SIZE i) hoffs3] at hoffsets
        obtain rfl := Option.some.inj hoffsets
        obtain ⟨masks, hm, hmi⟩ := hactive
        rw [evalOpR_ref, hmask3] at hm
        obtain rfl := Option.some.inj hm
        have hact : active s0 batch_size dim BLOCK_SIZE idx.1 := by
          simpa using hmi
        have hbound := hby hkprev idx.1 hact
        rw [← timeOffset_reverse_prev s0 batch_size dim BLOCK_SIZE length k
          idx.1 hkprev] at hbound
        simpa [Region.cast_id, timeOffset, reverseTime, colOffset]
          using hbound
      · intro hb
        have hklast : ¬ k < length - 1 := by
          have := of_decide_eq_false hb
          omega
        refine Stmt.TraceSafeListR.cons_intro ?_
          (fun _ _ => Stmt.TraceSafeListR.nil_intro)
        simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
          MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
          and_true, true_and, and_self]
        intro offsets hoffsets idx hactive
        rw [evalOpR_ref, hco3] at hoffsets
        obtain rfl := Option.some.inj hoffsets
        obtain ⟨masks, hm, hmi⟩ := hactive
        rw [evalOpR_ref, hmask3] at hm
        obtain rfl := Option.some.inj hm
        have hact : active s0 batch_size dim BLOCK_SIZE idx.1 := by
          simpa using hmi
        simpa [Region.cast_id, colOffset] using hbs hklast idx.1 hact
    · rcases dssmb_ifte_stepR_inv
          (dssmb_guard_evalR R q3 (length - 1 - k) ht3') hq4 with
        ⟨-, hrun⟩ | ⟨-, hrun⟩ <;>
      · obtain ⟨v4, rfl⟩ := dssmb_stepsR_singleton_assign_inv hrun
        exact htail v4

/-- `TraceSafeListR` of the two terminal stores from the loop-exit
register pins: both scatters' active lanes are the skin's
designated-step windows — in bounds by the `grad_s`/`grad_Λ` write-window
bounds. The first store's successor keeps every register (a memory
scatter), so the second leg reuses the same pins. -/
private theorem dssmb_postLoopSafeR (R : RoundingModel)
    (bounds : RegionBounds) (grad_s_ptr grad_lambda_ptr : RegionName)
    (batch_size dim BLOCK_SIZE : Nat) (s0 st : BlockState)
    (hCol : st.regs .nat [BLOCK_SIZE] "col_offsets"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          colOffset s0 BLOCK_SIZE idx.1 })
    (hMask : st.regs .bool [BLOCK_SIZE] "mask"
      = some { data := fun idx : TileIndex [BLOCK_SIZE] =>
          active s0 batch_size dim BLOCK_SIZE idx.1 })
    (hbgs : ∀ j : Fin BLOCK_SIZE, active s0 batch_size dim BLOCK_SIZE j →
      colOffset s0 BLOCK_SIZE j < bounds grad_s_ptr)
    (hbgl : ∀ j : Fin BLOCK_SIZE, active s0 batch_size dim BLOCK_SIZE j →
      colOffset s0 BLOCK_SIZE j < bounds grad_lambda_ptr) :
    Stmt.TraceSafeListR R bounds
      (diagSsmBackwardPostLoop grad_s_ptr grad_lambda_ptr BLOCK_SIZE)
      st := by
  unfold diagSsmBackwardPostLoop
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun q1 hq1 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
      MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR, and_true, true_and, and_self]
    intro offsets hoffsets idx hactive
    rw [evalOpR_ref, hCol] at hoffsets
    obtain rfl := Option.some.inj hoffsets
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hMask] at hm
    obtain rfl := Option.some.inj hm
    have hact : active s0 batch_size dim BLOCK_SIZE idx.1 := by
      simpa using hmi
    simpa [Region.cast_id] using hbgs idx.1 hact
  · -- the first scatter keeps every register
    have hcf : stepStmtR R (Stmt.store .real [BLOCK_SIZE]
        (.region grad_s_ptr (.ref .nat [BLOCK_SIZE] "col_offsets"))
        (.ref .real [BLOCK_SIZE] "grad_s")
        (.mask (.ref .bool [BLOCK_SIZE] "mask"))) st
        = stepStmt (Stmt.store .real [BLOCK_SIZE]
            (.region grad_s_ptr (.ref .nat [BLOCK_SIZE] "col_offsets"))
            (.ref .real [BLOCK_SIZE] "grad_s")
            (.mask (.ref .bool [BLOCK_SIZE] "mask"))) st := by
      simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def,
        BlockState.writeMemTypedR]
    rw [hcf] at hq1
    cases hgs : st.regs .real [BLOCK_SIZE] "grad_s" with
    | none =>
        simp [stepStmt, evalOp, evalOp.eq_def, hgs, Option.bind] at hq1
    | some gsv =>
        simp [stepStmt, evalOp, evalOp.eq_def, hgs, hCol, hMask,
          Option.bind] at hq1
        subst hq1
        refine Stmt.TraceSafeListR.cons_intro ?_
          (fun _ _ => Stmt.TraceSafeListR.nil_intro)
        simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
          MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
          memAccessActiveAddressSafeR, and_true, true_and, and_self]
        intro offsets hoffsets idx hactive
        rw [evalOpR_ref] at hoffsets
        rw [show (((TileShape.allIndices [BLOCK_SIZE]).foldl _ st).regs
              .nat [BLOCK_SIZE] "col_offsets")
            = st.regs .nat [BLOCK_SIZE] "col_offsets" from
          BlockState.foldl_writeMem_prop_masked_regs _ _ _ _ _ _ _ _ _,
          hCol] at hoffsets
        obtain rfl := Option.some.inj hoffsets
        obtain ⟨masks, hm, hmi⟩ := hactive
        rw [evalOpR_ref] at hm
        rw [show (((TileShape.allIndices [BLOCK_SIZE]).foldl _ st).regs
              .bool [BLOCK_SIZE] "mask")
            = st.regs .bool [BLOCK_SIZE] "mask" from
          BlockState.foldl_writeMem_prop_masked_regs _ _ _ _ _ _ _ _ _,
          hMask] at hm
        obtain rfl := Option.some.inj hm
        have hact : active s0 batch_size dim BLOCK_SIZE idx.1 := by
          simpa using hmi
        simpa [Region.cast_id] using hbgl idx.1 hact

/-- **The `TraceSafeR` walk for the whole backward kernel** — the
pre-loop by the manual cons walk, the reverse loop by the library's
`Stmt.forRangeTraceSafeR_inv` over the proven (launch-state-robust)
`diagSsmBackwardLoopContextInvariant`, whose register pins feed the
per-iteration body walk, and the terminal-store tail from the loop-exit
pins recovered through the cast-free collapse. The seven bound groups are
the skin's channel windows; `hT` instantiates the step index of every
step-independent window (a `Fin length` needs a step to exist). -/
private theorem dssmb_traceSafeR (R : RoundingModel)
    (bounds : RegionBounds)
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr
      grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hT : 0 < length) (hBS : BLOCK_SIZE ≤ batch_size * dim)
    (hGradXSNe : grad_x_ptr ≠ s_ptr) (hGradXYNe : grad_x_ptr ≠ y_ptr)
    (hGradXGradYNe : grad_x_ptr ≠ grad_y_ptr)
    (hGradXLambdaNe : grad_x_ptr ≠ lambda_ptr)
    (s : BlockState)
    (hbgy : ∀ (t : Fin length) (j : Fin BLOCK_SIZE),
      s.pids 0 * BLOCK_SIZE + j.val < batch_size * dim →
      t.val * (batch_size * dim) + (s.pids 0 * BLOCK_SIZE + j.val)
        < bounds grad_y_ptr)
    (hby : ∀ (t : Fin length) (j : Fin BLOCK_SIZE),
      0 < t.val ∧ s.pids 0 * BLOCK_SIZE + j.val < batch_size * dim →
      (t.val - 1) * (batch_size * dim) + (s.pids 0 * BLOCK_SIZE + j.val)
        < bounds y_ptr)
    (hbs : ∀ (t : Fin length) (j : Fin BLOCK_SIZE),
      t.val = 0 ∧ s.pids 0 * BLOCK_SIZE + j.val < batch_size * dim →
      s.pids 0 * BLOCK_SIZE + j.val < bounds s_ptr)
    (hbl : ∀ (t : Fin length) (j : Fin BLOCK_SIZE),
      s.pids 0 * BLOCK_SIZE + j.val < batch_size * dim →
      (s.pids 0 * BLOCK_SIZE + j.val) % dim < bounds lambda_ptr)
    (hbgx : ∀ (t : Fin length) (j : Fin BLOCK_SIZE),
      s.pids 0 * BLOCK_SIZE + j.val < batch_size * dim →
      t.val * (batch_size * dim) + (s.pids 0 * BLOCK_SIZE + j.val)
        < bounds grad_x_ptr)
    (hbgs : ∀ (t : Fin length) (j : Fin BLOCK_SIZE),
      t.val = 0 ∧ s.pids 0 * BLOCK_SIZE + j.val < batch_size * dim →
      s.pids 0 * BLOCK_SIZE + j.val < bounds grad_s_ptr)
    (hbgl : ∀ (t : Fin length) (j : Fin BLOCK_SIZE),
      t.val = 0 ∧ s.pids 0 * BLOCK_SIZE + j.val < batch_size * dim →
      s.pids 0 * BLOCK_SIZE + j.val < bounds grad_lambda_ptr) :
    ((diag_ssm_backward_kernel s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr
      grad_lambda_ptr grad_y_ptr length batch_size dim
      BLOCK_SIZE).toAlgKernel).TraceSafeR R bounds s := by
  unfold Kernel.TraceSafeR
  rw [diag_ssm_backward_kernel_toAlg_body]
  unfold diagSsmBackwardProjectedBody
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
    · exact dssmb_preLoopSafeR R bounds lambda_ptr batch_size dim
        BLOCK_SIZE s (fun j hj => hbl ⟨0, hT⟩ j hj)
    · intro s5 hs5
      rw [dssmb_preLoop_castFree R lambda_ptr batch_size dim BLOCK_SIZE
        s] at hs5
      have hCtx0 := diagSsmBackwardLoopContextInvariant_init_of_preloop s
        s5 s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr length batch_size
        dim BLOCK_SIZE hs5
      refine Stmt.TraceSafeListR.cons_intro ?_
        (fun _ _ => Stmt.TraceSafeListR.nil_intro)
      simp only [Stmt.TraceSafeR]
      refine Stmt.forRangeTraceSafeR_inv R bounds "i" length 1
        (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr length
          batch_size dim BLOCK_SIZE)
        (fun k st => if k < length then
          diagSsmBackwardLoopContextInvariant s s_ptr lambda_ptr y_ptr
            grad_x_ptr grad_y_ptr length batch_size dim BLOCK_SIZE k st
        else True) ?_ 0 s5 (by simpa [hT] using hCtx0)
      intro c stt hc hP
      have hCtxC : diagSsmBackwardLoopContextInvariant s s_ptr lambda_ptr
          y_ptr grad_x_ptr grad_y_ptr length batch_size dim BLOCK_SIZE c
          stt := by simpa [hc] using hP
      obtain ⟨hInv, hLambda, hCol, hMask, hSRead, hYRead, hGradYRead,
        hLambdaRead⟩ := hCtxC
      refine ⟨dssmb_bodySafeR R bounds s_ptr y_ptr grad_x_ptr grad_y_ptr
        batch_size dim BLOCK_SIZE s stt c hc hCol hMask
        (fun j hj => hbgy ⟨length - 1 - c, by omega⟩ j hj)
        (fun hkprev j hj => hby ⟨length - 1 - c, by omega⟩ j
          ⟨(show (0 : Nat) < length - 1 - c by omega), hj⟩)
        (fun _ j hj => hbs ⟨0, hT⟩ j ⟨rfl, hj⟩)
        (fun j hj => hbgx ⟨length - 1 - c, by omega⟩ j hj), ?_⟩
      obtain ⟨st', hstep, hCtx'⟩ :=
        diagSsmBackwardLoopContextInvariant_body_step_exists s stt s_ptr
          lambda_ptr y_ptr grad_x_ptr grad_y_ptr batch_size dim BLOCK_SIZE
          c hc
          ⟨hInv, hLambda, hCol, hMask, hSRead, hYRead, hGradYRead,
            hLambdaRead⟩
          (dssmb_gradXOffset_injective s batch_size dim BLOCK_SIZE length
            hBS)
          hGradXSNe hGradXYNe hGradXGradYNe hGradXLambdaNe
      refine ⟨st', ?_, ?_⟩
      · rw [dssmb_body_castFree R s_ptr y_ptr grad_x_ptr grad_y_ptr length
          batch_size dim BLOCK_SIZE]
        exact hstep
      · by_cases hnext : c + 1 < length
        · simpa [hnext] using hCtx'
        · simp [hnext]
  · intro s6 hs6
    rw [stepStmtsR_append R
        (diagSsmBackwardPreLoop lambda_ptr batch_size dim BLOCK_SIZE)
        [Stmt.forRange "i" 0 length 1
          (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr
            length batch_size dim BLOCK_SIZE)] s,
      dssmb_preLoop_castFree R lambda_ptr batch_size dim BLOCK_SIZE
        s] at hs6
    cases hPre : stepStmts (diagSsmBackwardPreLoop lambda_ptr batch_size
        dim BLOCK_SIZE) s with
    | none => rw [hPre] at hs6; simp at hs6
    | some s5 =>
        rw [hPre, Option.bind_some] at hs6
        cases hF : stepStmtR R (Stmt.forRange "i" 0 length 1
            (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr
              length batch_size dim BLOCK_SIZE)) s5 with
        | none =>
            simp only [stepStmtsR, hF] at hs6
            simp at hs6
        | some s6' =>
            rw [stepStmtsR_cons_some hF, stepStmtsR_nil] at hs6
            obtain rfl := Option.some.inj hs6
            have hFexact : stepStmt (.forRange "i" 0 length 1
                (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr
                  length batch_size dim BLOCK_SIZE)) s5 = some s6' := by
              rw [stepStmtR_forRange,
                stepForRangeAuxR_castFree R _
                  (dssmb_body_castFree R s_ptr y_ptr grad_x_ptr grad_y_ptr
                    length batch_size dim BLOCK_SIZE) "i"] at hF
              rwa [← stepForRangeAux.forRange_unfold] at hF
            have hCtxL := diagSsmBackwardForLoop_context_of_preloop s s5
              s6' s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr length
              batch_size dim BLOCK_SIZE
              (dssmb_gradXOffset_injective s batch_size dim BLOCK_SIZE
                length hBS)
              hGradXSNe hGradXYNe hGradXGradYNe hGradXLambdaNe hPre
              hFexact
            obtain ⟨-, -, hCol, hMask, -, -, -, -⟩ := hCtxL
            exact dssmb_postLoopSafeR R bounds grad_s_ptr grad_lambda_ptr
              batch_size dim BLOCK_SIZE s s6' hCol hMask
              (fun j hj => hbgs ⟨0, hT⟩ j ⟨rfl, hj⟩)
              (fun j hj => hbgl ⟨0, hT⟩ j ⟨rfl, hj⟩)

/-! ### The rounded Hoare triple (`hrun`) -/

/-- Termination, per-window values of all three output channels and the
per-cell frame of the whole reverse scan under `execR R`, from an
**arbitrary** launch state: the exact
`diagSsmBackwardLoopContextInvariant` stack runs verbatim (the surface is
cast-free, so `execR R` collapses onto the exact stepper), extended with
the per-segment memory frames and closed by the terminal-scatter
readback lemmas. -/
private theorem dssmb_runR (R : RoundingModel)
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr
      grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hBS : BLOCK_SIZE ≤ batch_size * dim)
    (hGradXSNe : grad_x_ptr ≠ s_ptr) (hGradXYNe : grad_x_ptr ≠ y_ptr)
    (hGradXGradYNe : grad_x_ptr ≠ grad_y_ptr)
    (hGradXLambdaNe : grad_x_ptr ≠ lambda_ptr)
    (hGradSGradXNe : grad_s_ptr ≠ grad_x_ptr)
    (hGradLambdaGradXNe : grad_lambda_ptr ≠ grad_x_ptr)
    (hGradLambdaGradSNe : grad_lambda_ptr ≠ grad_s_ptr)
    (s₀ : BlockState) :
    ∃ sfin,
      execR R (diag_ssm_backward_kernel s_ptr lambda_ptr y_ptr grad_s_ptr
          grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim
          BLOCK_SIZE).toAlgKernel s₀ = some sfin
      ∧ (∀ (t : Fin length) (j : Fin BLOCK_SIZE),
          active s₀ batch_size dim BLOCK_SIZE j →
          sfin.readMem grad_x_ptr
              (timeOffset s₀ batch_size dim BLOCK_SIZE t.val j)
            = diagSsmBackwardGradXSpec s₀ lambda_ptr grad_y_ptr batch_size
                dim BLOCK_SIZE length t.val j)
      ∧ (∀ j : Fin BLOCK_SIZE, active s₀ batch_size dim BLOCK_SIZE j →
          sfin.readMem grad_s_ptr (colOffset s₀ BLOCK_SIZE j)
            = diagSsmBackwardGradSAfter s₀ lambda_ptr grad_y_ptr
                batch_size dim BLOCK_SIZE length j length)
      ∧ (∀ j : Fin BLOCK_SIZE, active s₀ batch_size dim BLOCK_SIZE j →
          sfin.readMem grad_lambda_ptr (colOffset s₀ BLOCK_SIZE j)
            = diagSsmBackwardGradLambdaAfter s₀ s_ptr lambda_ptr y_ptr
                grad_y_ptr batch_size dim BLOCK_SIZE length j length)
      ∧ (∀ r oo,
          (r ≠ grad_x_ptr ∨ ∀ (tf : Fin length) (j : Fin BLOCK_SIZE),
            active s₀ batch_size dim BLOCK_SIZE j →
            oo ≠ timeOffset s₀ batch_size dim BLOCK_SIZE tf.val j) →
          (r ≠ grad_s_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            active s₀ batch_size dim BLOCK_SIZE j →
            oo ≠ colOffset s₀ BLOCK_SIZE j) →
          (r ≠ grad_lambda_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            active s₀ batch_size dim BLOCK_SIZE j →
            oo ≠ colOffset s₀ BLOCK_SIZE j) →
          sfin.mem r oo = s₀.mem r oo) := by
  have hInj := dssmb_gradXOffset_injective s₀ batch_size dim BLOCK_SIZE
    length hBS
  cases hPre : stepStmts (diagSsmBackwardPreLoop lambda_ptr batch_size dim
      BLOCK_SIZE) s₀ with
  | none =>
      unfold diagSsmBackwardPreLoop at hPre
      simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.cop,
        NumericDType.mul, NumericDType.add, IntegralDType.mod,
        ComparableDType.lt, Option.bind] at hPre
  | some stPre =>
      have hCtx0 := diagSsmBackwardLoopContextInvariant_init_of_preloop s₀
        stPre s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr length
        batch_size dim BLOCK_SIZE hPre
      have hPreMem : stPre.mem = s₀.mem :=
        dssm_stepStmts_assigns_mem
          (diagSsmBackwardPreLoop lambda_ptr batch_size dim BLOCK_SIZE)
          (by
            intro stmt hst
            simp only [diagSsmBackwardPreLoop, List.mem_cons,
              List.not_mem_nil, or_false] at hst
            rcases hst with rfl | rfl | rfl | rfl | rfl | rfl <;>
              exact ⟨_, _, _, _, rfl⟩)
          hPre
      obtain ⟨final, stLoop, hFor, hFinal, hP⟩ :=
        forRange_inv (idx := "i") (start := 0) (stop := length) (step := 1)
          (body := diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr
            grad_y_ptr length batch_size dim BLOCK_SIZE)
          (P := fun k st =>
            (if k < length then
              diagSsmBackwardLoopContextInvariant s₀ s_ptr lambda_ptr y_ptr
                grad_x_ptr grad_y_ptr length batch_size dim BLOCK_SIZE k st
            else
              diagSsmBackwardLoopContextInvariant s₀ s_ptr lambda_ptr y_ptr
                grad_x_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
                length st) ∧
            ∀ r oo,
              (r ≠ grad_x_ptr ∨ ∀ (tf : Fin length) (j : Fin BLOCK_SIZE),
                active s₀ batch_size dim BLOCK_SIZE j →
                oo ≠ timeOffset s₀ batch_size dim BLOCK_SIZE tf.val j) →
              st.mem r oo = s₀.mem r oo)
          (s_init := stPre)
          (by norm_num)
          (by
            refine ⟨?_, fun r oo _ => by rw [hPreMem]⟩
            by_cases hlen : 0 < length
            · simpa [hlen] using hCtx0
            · have hzero : length = 0 := by omega
              simpa [hlen, hzero] using hCtx0)
          (fun k st hk hP => by
            obtain ⟨hCtxIf, hFrame⟩ := hP
            have hCtxK : diagSsmBackwardLoopContextInvariant s₀ s_ptr
                lambda_ptr y_ptr grad_x_ptr grad_y_ptr length batch_size
                dim BLOCK_SIZE k st := by simpa [hk] using hCtxIf
            obtain ⟨st', hstep, hCtx'⟩ :=
              diagSsmBackwardLoopContextInvariant_body_step_exists s₀ st
                s_ptr lambda_ptr y_ptr grad_x_ptr grad_y_ptr batch_size
                dim BLOCK_SIZE k hk hCtxK hInj hGradXSNe hGradXYNe
                hGradXGradYNe hGradXLambdaNe
            obtain ⟨hInv, hLambda, hCol, hMask, hSRead, hYRead,
              hGradYRead, hLambdaRead⟩ := hCtxK
            refine ⟨st', hstep, ?_, ?_⟩
            · by_cases hnext : k + 1 < length
              · simpa [hnext] using hCtx'
              · have hlast : k + 1 = length := by omega
                simpa [hnext, hlast] using hCtx'
            · intro r oo hcond
              rw [dssmb_body_step_frame s₀ st st' s_ptr lambda_ptr y_ptr
                grad_x_ptr grad_y_ptr batch_size dim BLOCK_SIZE k hk
                hLambda hInv.1 hInv.2.1 hCol hMask hSRead hYRead
                hGradYRead hLambdaRead hstep r oo hcond]
              exact hFrame r oo hcond)
      obtain ⟨hCtxIfL, hFrameL⟩ := hP
      have hCtxL : diagSsmBackwardLoopContextInvariant s₀ s_ptr lambda_ptr
          y_ptr grad_x_ptr grad_y_ptr length batch_size dim BLOCK_SIZE
          length stLoop := by
        have hnot : ¬ final < length := by omega
        simpa [hnot] using hCtxIfL
      obtain ⟨hInvL, hLambdaL, hColL, hMaskL, -, -, -, -⟩ := hCtxL
      cases hPost : stepStmts (diagSsmBackwardPostLoop grad_s_ptr
          grad_lambda_ptr BLOCK_SIZE) stLoop with
      | none =>
          unfold diagSsmBackwardPostLoop at hPost
          simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hInvL.1,
            hInvL.2.1, hColL, hMaskL, Option.bind] at hPost
      | some sfin =>
          have hReadback := diagSsmBackwardPostLoop_readback s₀ stLoop
            sfin s_ptr lambda_ptr y_ptr grad_s_ptr grad_lambda_ptr
            grad_y_ptr batch_size dim BLOCK_SIZE length hInvL.1 hInvL.2.1
            hColL hMaskL hGradLambdaGradSNe hPost
          have hForR : stepStmtR R (Stmt.forRange "i" 0 length 1
              (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr grad_y_ptr
                length batch_size dim BLOCK_SIZE)) stPre
              = some stLoop := by
            rw [stepStmtR_forRange,
              stepForRangeAuxR_castFree R _
                (dssmb_body_castFree R s_ptr y_ptr grad_x_ptr grad_y_ptr
                  length batch_size dim BLOCK_SIZE) "i",
              ← stepForRangeAux.forRange_unfold]
            exact hFor
          refine ⟨sfin, ?_, ?_, ?_, ?_, ?_⟩
          · show execR R (diag_ssm_backward_kernel s_ptr lambda_ptr y_ptr
              grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length
              batch_size dim BLOCK_SIZE).toAlgKernel s₀ = some sfin
            unfold execR
            rw [diag_ssm_backward_kernel_toAlg_body]
            unfold diagSsmBackwardProjectedBody
            rw [stepStmtsR_append R
                (diagSsmBackwardPreLoop lambda_ptr batch_size dim
                    BLOCK_SIZE ++
                  [Stmt.forRange "i" 0 length 1
                    (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr
                      grad_y_ptr length batch_size dim BLOCK_SIZE)])
                (diagSsmBackwardPostLoop grad_s_ptr grad_lambda_ptr
                  BLOCK_SIZE) s₀,
              stepStmtsR_append R
                (diagSsmBackwardPreLoop lambda_ptr batch_size dim
                  BLOCK_SIZE)
                [Stmt.forRange "i" 0 length 1
                  (diagSsmBackwardLoopBody s_ptr y_ptr grad_x_ptr
                    grad_y_ptr length batch_size dim BLOCK_SIZE)] s₀,
              dssmb_preLoop_castFree R lambda_ptr batch_size dim
                BLOCK_SIZE s₀,
              hPre, Option.bind_some, stepStmtsR_cons_some hForR,
              stepStmtsR_nil, Option.bind_some,
              dssmb_postLoop_castFree R grad_s_ptr grad_lambda_ptr
                BLOCK_SIZE stLoop]
            exact hPost
          · intro t j hact
            have hk : length - 1 - t.val < length := by
              have := t.isLt
              omega
            have hrt : reverseTime length (length - 1 - t.val) = t.val := by
              have := t.isLt
              unfold reverseTime
              omega
            have hinv : stLoop.readMem grad_x_ptr
                (timeOffset s₀ batch_size dim BLOCK_SIZE
                  (reverseTime length (length - 1 - t.val)) j)
                = diagSsmBackwardGradXSpec s₀ lambda_ptr grad_y_ptr
                    batch_size dim BLOCK_SIZE length
                    (reverseTime length (length - 1 - t.val)) j :=
              hInvL.2.2 ((⟨length - 1 - t.val, hk⟩ : Fin length), j,
                PUnit.unit) hk hact
            rw [hrt] at hinv
            rw [diagSsmBackwardPostLoop_preserve_gradX s₀ stLoop sfin
              s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr
              grad_y_ptr batch_size dim BLOCK_SIZE length
              (timeOffset s₀ batch_size dim BLOCK_SIZE t.val j) hInvL.1
              hInvL.2.1 hColL hMaskL hGradSGradXNe hGradLambdaGradXNe
              hPost]
            exact hinv
          · exact hReadback.1
          · exact hReadback.2
          · intro r oo hcx hcs hcl
            rw [dssmb_postLoop_frame s₀ stLoop sfin s_ptr lambda_ptr y_ptr
              grad_s_ptr grad_lambda_ptr grad_y_ptr batch_size dim
              BLOCK_SIZE length hInvL.1 hInvL.2.1 hColL hMaskL hPost r oo
              hcs hcl]
            exact hFrameL r oo hcx

/-! ### The headline -/

/-- **The backward `⊨[R]` streaming headline (wave-5 S3 grouped emit
genre).** For every rounding model `R`, the faithful
`diag_ssm_backward_kernel` surface implements, on its grouped
`StreamGroupedEmitMasked3DKernelIO` signature, the **ideal ℝ reverse-time
gradient scan** over the streamed tiles, one closed form per output
channel (`diagSsmBackwardStreamSpec`, the shared constructor-head
matcher):

* `grad_x` at step `t` — the running `grad_s` after absorbing
  `grad_y[t]`: the upstream gradient at `t` plus the reverse **suffix**
  fold of `(gs ← (grad_y[T-1-u] + gs)·Λ)` over the `T-1-t` iterations
  strictly before `t` is processed;
* `grad_s` (terminal, designated step `t = 0`) — the full `T`-iteration
  reverse fold;
* `grad_Λ` (terminal) — the full accumulator
  `Σ (grad_y[u] + gs)·prev u`, where `prev` resolves the kernel's
  `if t > 0` branch on the streams (`y[t-1]` vs the initial state `s`).

The grid is 1-D, so every window ignores `pid₁`/`pid₂` and the headline
is ∀-both. The kernel has **zero rounding events** (all loads, the scan
arithmetic and all three raw stores are at `.real`; there is no
`castFloat`), so the skin's boundary quantization degenerates: the
readback's `R.round .real` is the identity by the model's defining
`round_real` field — the ∀-`R` face holds via the `RoundingModel` `.real`
identity fields, not as a `.triv` special case.

Layer map: the surface is cast-free, so under `execR R` it collapses
verbatim onto the exact stepper and the proven
`diagSsmBackwardLoopContextInvariant` / `forRange_inv` /
terminal-scatter stack above is reused unchanged; the `⊨[R]` face adds
the `TraceSafeR` walk (via the library's `forRangeTraceSafeR_inv`; the
runtime `if t > 0` branch is walked per iteration, both arms taken
depending on the counter), the per-cell memory frames
(`dssmb_body_step_frame` / `dssmb_postLoop_frame`), and the stream-lane
spec bridges (`dssmb_streamGradS_eq` / `dssmb_streamGradLambda_eq`).

All hypotheses are truth-forced, with provenance:

* `hT : 0 < length` — truth-forced twice. (1) The step-independent
  windows (`s`/`Λ` reads, the two terminal stores) are step-indexed
  (`Fin length`), so their bound groups are non-vacuous only when a step
  exists, yet the kernel issues the pre-loop `Λ` load and both terminal
  stores even for a 0-step scan (the wave-5 round-5 lesson). (2) At
  `length = 0` the two terminal stores still execute while the
  `Fin 0`-gated `writeMask` declares an **empty** write set, so the
  skin's frame condition would be outright false.
* `hBS : BLOCK_SIZE ≤ batch_size·dim` — the tiling-geometry form of the
  exact stack's `hOutInj` (`grad_x` full-grid window injectivity, which
  the loop invariant needs to carry previously-emitted windows through
  later scatters). Holds for every real launch (the grid is
  `⌈batch_size·dim / BLOCK_SIZE⌉` programs over a `batch_size·dim`-wide
  lane space).
* `hGradXSNe`/`hGradXYNe`/`hGradXGradYNe`/`hGradXLambdaNe` — inherited
  verbatim from the exact backward stack: the in-loop `grad_x` scatter
  must not clobber the four regions the scan keeps reading (`s`, `y`,
  `grad_y`, `λ`); e.g. if `grad_x_ptr = y_ptr`, iteration `k`'s store
  would corrupt the `y[t-1]` cells iteration `k+1` reads and the closed
  form would be false (`y_ptr` readback stability).
* `hGradSGradXNe`/`hGradLambdaGradXNe` — the terminal stores must not
  clobber the already-emitted `grad_x` windows
  (`diagSsmBackwardPostLoop_preserve_gradX`).
* `hGradLambdaGradSNe` — the second terminal scatter must not clobber
  the first one's cells (`diagSsmBackwardPostLoop_readback`).

Relation to the exact surface: the exact headline
`diag_ssm_backward_kernel_compute_correct` above is retained unchanged;
this `⊨[R]` face restates the same gradient content on the grouped
streaming emit skin, for every `R` at once (at the `.real` grid the two
faces carry the same exact cells). Both faces are kept per the
rounding-as-default doctrine. -/
specification diag_ssm_backward_io_correctness (R : RoundingModel)
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr
      grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat)
    (hT : 0 < length) (hBS : BLOCK_SIZE ≤ batch_size * dim)
    (hGradXSNe : grad_x_ptr ≠ s_ptr) (hGradXYNe : grad_x_ptr ≠ y_ptr)
    (hGradXGradYNe : grad_x_ptr ≠ grad_y_ptr)
    (hGradXLambdaNe : grad_x_ptr ≠ lambda_ptr)
    (hGradSGradXNe : grad_s_ptr ≠ grad_x_ptr)
    (hGradLambdaGradXNe : grad_lambda_ptr ≠ grad_x_ptr)
    (hGradLambdaGradSNe : grad_lambda_ptr ≠ grad_s_ptr) :
    diagSsmBackwardKernelIO s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr
        grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE ⊨[R]
      fun _ _ _ xs o t j => diagSsmBackwardStreamSpec xs o t j := by
  refine StreamGroupedEmitMasked3DKernelIO.ImplementsR.intro _ ?_ ?_ ?_ ?_
  · -- `hout`: every output channel's buffer is in the allocation list
    intro o
    rcases o with ⟨ov, ho⟩
    rcases ov with _ | _ | _ | ov
    · simp [diagSsmBackwardKernelIO]
    · simp [diagSsmBackwardKernelIO]
    · simp [diagSsmBackwardKernelIO]
    · exfalso
      have ho3 : ov + 1 + 1 + 1 < 3 := by
        simpa [diagSsmBackwardKernelIO] using ho
      omega
  · exact dssmb_backward_flattenOk s_ptr lambda_ptr y_ptr grad_s_ptr
      grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim
      BLOCK_SIZE
  · -- the safety walk
    intro bounds s xs _hx hbr hbw
    simp only [diagSsmBackwardKernelIO] at hbr hbw ⊢
    exact dssmb_traceSafeR R bounds s_ptr lambda_ptr y_ptr grad_s_ptr
      grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim
      BLOCK_SIZE hT hBS hGradXSNe hGradXYNe hGradXGradYNe hGradXLambdaNe s
      (fun t j hj => hbr (⟨0, by decide⟩ : Fin 4) t j hj)
      (fun t j hj => hbr (⟨1, by decide⟩ : Fin 4) t j hj)
      (fun t j hj => hbr (⟨2, by decide⟩ : Fin 4) t j hj)
      (fun t j hj => hbr (⟨3, by decide⟩ : Fin 4) t j hj)
      (fun t j hj => hbw (⟨0, by decide⟩ : Fin 3) t j hj)
      (fun t j hj => hbw (⟨1, by decide⟩ : Fin 3) t j hj)
      (fun t j hj => hbw (⟨2, by decide⟩ : Fin 3) t j hj)
  · -- the rounded Hoare triple
    intro s₀ xs _hundef hx
    simp only [diagSsmBackwardKernelIO] at hx ⊢
    obtain ⟨sfin, hexec, hvx, hvs, hvl, hframe⟩ :=
      dssmb_runR R s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr
        grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE hBS
        hGradXSNe hGradXYNe hGradXGradYNe hGradXLambdaNe hGradSGradXNe
        hGradLambdaGradXNe hGradLambdaGradSNe s₀
    refine ⟨sfin, hexec, ?_, ?_⟩
    · intro o t j hj
      rcases o with ⟨ov, ho⟩
      rcases ov with _ | _ | _ | ov
      · -- grad_x
        have hact : active s₀ batch_size dim BLOCK_SIZE j := hj
        have hlam : s₀.readMem lambda_ptr
            (colOffset s₀ BLOCK_SIZE j % dim)
              = xs (⟨3, by decide⟩ : Fin 4) t j :=
          hx (⟨3, by decide⟩ : Fin 4) t j hj
        have hgys : ∀ u : Fin length,
            s₀.readMem grad_y_ptr
              (timeOffset s₀ batch_size dim BLOCK_SIZE u.val j)
              = xs (⟨0, by decide⟩ : Fin 4) u j :=
          fun u => hx (⟨0, by decide⟩ : Fin 4) u j hj
        have hGS := dssmb_streamGradS_eq s₀ lambda_ptr grad_y_ptr
          (length := length) batch_size dim BLOCK_SIZE j
          (xs (⟨3, by decide⟩ : Fin 4) t j)
          (fun u => xs (⟨0, by decide⟩ : Fin 4) u j) hlam hgys
          (length - 1 - t.val) (by omega)
        have hbridge :
            xs (⟨0, by decide⟩ : Fin 4) t j +
                dssmbStreamGradS (T := length)
                  (fun u => xs (⟨0, by decide⟩ : Fin 4) u j)
                  (xs (⟨3, by decide⟩ : Fin 4) t j) (length - 1 - t.val)
              = diagSsmBackwardGradXSpec s₀ lambda_ptr grad_y_ptr
                  batch_size dim BLOCK_SIZE length t.val j := by
          rw [← hgys t, hGS]
          rfl
        have hv' : sfin.readMem grad_x_ptr
            (t.val * (batch_size * dim) +
              (s₀.pids 0 * BLOCK_SIZE + j.val))
            = diagSsmBackwardGradXSpec s₀ lambda_ptr grad_y_ptr batch_size
                dim BLOCK_SIZE length t.val j := hvx t j hact
        show sfin.readMemAs FloatDType.real grad_x_ptr
            (t.val * (batch_size * dim) +
              (s₀.pids 0 * BLOCK_SIZE + j.val))
          = FloatDType.real.ofReal (R.round FloatDType.real
              (xs (⟨0, by decide⟩ : Fin 4) t j +
                dssmbStreamGradS
                  (fun u => xs (⟨0, by decide⟩ : Fin 4) u j)
                  (xs (⟨3, by decide⟩ : Fin 4) t j) (length - 1 - t.val)))
        rw [BlockState.readMemAs_real, hv', ← hbridge]
        simp only [RoundingModel.round_real_apply, FloatDType.ofReal]
        rfl
      · -- grad_s (terminal, designated step)
        have hact : active s₀ batch_size dim BLOCK_SIZE j := hj.2
        have hlam : s₀.readMem lambda_ptr
            (colOffset s₀ BLOCK_SIZE j % dim)
              = xs (⟨3, by decide⟩ : Fin 4) t j :=
          hx (⟨3, by decide⟩ : Fin 4) t j hj.2
        have hgys : ∀ u : Fin length,
            s₀.readMem grad_y_ptr
              (timeOffset s₀ batch_size dim BLOCK_SIZE u.val j)
              = xs (⟨0, by decide⟩ : Fin 4) u j :=
          fun u => hx (⟨0, by decide⟩ : Fin 4) u j hj.2
        have hGS := dssmb_streamGradS_eq s₀ lambda_ptr grad_y_ptr
          (length := length) batch_size dim BLOCK_SIZE j
          (xs (⟨3, by decide⟩ : Fin 4) t j)
          (fun u => xs (⟨0, by decide⟩ : Fin 4) u j) hlam hgys length
          (Nat.le_refl length)
        have hv' : sfin.readMem grad_s_ptr
            (s₀.pids 0 * BLOCK_SIZE + j.val)
            = diagSsmBackwardGradSAfter s₀ lambda_ptr grad_y_ptr
                batch_size dim BLOCK_SIZE length j length := hvs j hact
        show sfin.readMemAs FloatDType.real grad_s_ptr
            (s₀.pids 0 * BLOCK_SIZE + j.val)
          = FloatDType.real.ofReal (R.round FloatDType.real
              (dssmbStreamGradS
                (fun u => xs (⟨0, by decide⟩ : Fin 4) u j)
                (xs (⟨3, by decide⟩ : Fin 4) t j) length))
        rw [BlockState.readMemAs_real, hv', ← hGS]
        simp only [RoundingModel.round_real_apply, FloatDType.ofReal]
        rfl
      · -- grad_Λ (terminal, designated step)
        have hact : active s₀ batch_size dim BLOCK_SIZE j := hj.2
        have hlam : s₀.readMem lambda_ptr
            (colOffset s₀ BLOCK_SIZE j % dim)
              = xs (⟨3, by decide⟩ : Fin 4) t j :=
          hx (⟨3, by decide⟩ : Fin 4) t j hj.2
        have hgys : ∀ u : Fin length,
            s₀.readMem grad_y_ptr
              (timeOffset s₀ batch_size dim BLOCK_SIZE u.val j)
              = xs (⟨0, by decide⟩ : Fin 4) u j :=
          fun u => hx (⟨0, by decide⟩ : Fin 4) u j hj.2
        have hys : ∀ u : Fin length, 0 < u.val →
            s₀.readMem y_ptr
                ((u.val - 1) * (batch_size * dim) +
                  colOffset s₀ BLOCK_SIZE j)
              = xs (⟨1, by decide⟩ : Fin 4) u j :=
          fun u hu => hx (⟨1, by decide⟩ : Fin 4) u j ⟨hu, hj.2⟩
        have hss : ∀ u : Fin length, u.val = 0 →
            s₀.readMem s_ptr (colOffset s₀ BLOCK_SIZE j)
              = xs (⟨2, by decide⟩ : Fin 4) u j :=
          fun u hu => hx (⟨2, by decide⟩ : Fin 4) u j ⟨hu, hj.2⟩
        have hGL := dssmb_streamGradLambda_eq s₀ s_ptr lambda_ptr y_ptr
          grad_y_ptr (length := length) batch_size dim BLOCK_SIZE j
          (xs (⟨3, by decide⟩ : Fin 4) t j)
          (fun u => xs (⟨0, by decide⟩ : Fin 4) u j)
          (fun u => xs (⟨1, by decide⟩ : Fin 4) u j)
          (fun u => xs (⟨2, by decide⟩ : Fin 4) u j)
          hlam hgys hys hss length (Nat.le_refl length)
        have hv' : sfin.readMem grad_lambda_ptr
            (s₀.pids 0 * BLOCK_SIZE + j.val)
            = diagSsmBackwardGradLambdaAfter s₀ s_ptr lambda_ptr y_ptr
                grad_y_ptr batch_size dim BLOCK_SIZE length j length :=
          hvl j hact
        show sfin.readMemAs FloatDType.real grad_lambda_ptr
            (s₀.pids 0 * BLOCK_SIZE + j.val)
          = FloatDType.real.ofReal (R.round FloatDType.real
              (dssmbStreamGradLambda
                (fun u => xs (⟨0, by decide⟩ : Fin 4) u j)
                (fun u => xs (⟨1, by decide⟩ : Fin 4) u j)
                (fun u => xs (⟨2, by decide⟩ : Fin 4) u j)
                (xs (⟨3, by decide⟩ : Fin 4) t j) length))
        rw [BlockState.readMemAs_real, hv', ← hGL]
        simp only [RoundingModel.round_real_apply, FloatDType.ofReal]
        rfl
      · exfalso
        have ho3 : ov + 1 + 1 + 1 < 3 := by
          simpa [diagSsmBackwardKernelIO] using ho
        omega
    · -- the frame
      intro r oo hcond
      refine hframe r oo ?_ ?_ ?_
      · by_cases hr : r = grad_x_ptr
        · exact Or.inr fun tf j hact =>
            hcond ⟨0, by omega⟩ tf j hact hr
        · exact Or.inl hr
      · by_cases hr : r = grad_s_ptr
        · exact Or.inr fun j hact =>
            hcond ⟨1, by omega⟩ ⟨0, hT⟩ j ⟨rfl, hact⟩ hr
        · exact Or.inl hr
      · by_cases hr : r = grad_lambda_ptr
        · exact Or.inr fun j hact =>
            hcond ⟨2, by omega⟩ ⟨0, hT⟩ j ⟨rfl, hact⟩ hr
        · exact Or.inl hr

end BackwardIOFace

end VeriTile.Bench.TritonBenchG.DiagSsmTriton
