import VeriTile.Triton

/-!
# `kldiv_triton` — strict per-kernel correctness

`kldiv_triton.py` provides the forward KL-divergence kernel
`_kldiv_kernel_forward` (per row `pid`, streams `n_cols` in `BLOCK_SIZE` tiles,
computing `y_true * (log(y_true) - y)` or, for `log_target`,
`exp(y_true) * (y_true - y)`; stores per element for `reduction=none` or sums
otherwise) and the backward kernel `_kldiv_kernel_backward` (`-target`, or
`-exp(target)` for `log_target`).

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies. The host launch (grid `(B,)`, the
`BLOCK_SIZE`/`num_warps` choices, the `reduction`-mode string mapping and the
host-side post-reduction `sum/mean/batchmean`, and how the runtime composes
per-program writes) is the *trusted boundary*, not a proof obligation here. The
row id `pid` is universally quantified, so each per-program statement covers
every row of the grid.

## Proof architecture

This is a multi-branch port (`log_target` × `reduction` × forward/backward);
there is no single top theorem. Each verified constexpr branch has its own
compute-facing correctness theorem:

```
kldiv_backward_default_compute_correct          (-target)        ← defaultSpec
  └─ kldiv_backward_default_correct
kldiv_backward_log_target_compute_correct       (-exp target)    ← logTargetSpec
  └─ kldiv_backward_log_target_correct
kldiv_forward_default_none_compute_correct      (reduction=none) ← forwardDefaultSpec
  └─ kldiv_forward_default_none_correct
kldiv_forward_log_target_none_compute_correct   (reduction=none) ← forwardLogTargetSpec
  └─ kldiv_forward_log_target_none_correct

kldiv_forward_surface_toAlgorithm_supported     full surface lowers (lowering only)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` /
`num_warps` are not modeled. **This port is partial.** The full reduction-mode
forward surface `kldiv_forward_surface` is only proved to *lower* to the
algorithm layer; value correctness targets the `reduction=0` (none) elementwise
branches plus both backward branches. The KL loss spec is the per-element
expression `y_true * (log(y_true) - y)` / `exp(y_true) * (y_true - y)` (this port
has **no `eps` clamp**, unlike the `kldiv_ops` port). Proofs target the
Python-tested **single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`**, where the
column loop runs once; masked lanes use `other=0.0` and out-of-bounds lanes
(`i ≥ n_cols`) are preserved. Output offset injectivity is an explicit
hypothesis. The specs reference `Real.log` / `Real.exp` directly, not
`VeriTile.Triton.Math.*`.
-/

namespace VeriTile.Bench.TritonBenchG.KldivTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `kldiv_triton.py`'s `_kldiv_kernel_forward`.

This keeps the row pointer advances, dynamic `n_cols` loop, `log_target`
selection, and reduction-mode store behavior. -/
def kldiv_forward_surface
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (log_target : Bool) (reduction : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  y_ptr += pid * $(y_stride)
  gt_ptr += pid * $(gt_stride)
  loss_ptr += pid * $(loss_stride)

  base_offsets = tl.arange(0, $(BLOCK_SIZE))

  for i in range($(0), $(n_cols), $(BLOCK_SIZE)) {
    offsets = i + base_offsets
    mask = offsets < $(n_cols)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)

    if not log_target {
      loss = y_true * (tl.log(y_true) - y)
    } else {
      loss = tl.exp(y_true) * (y_true - y)
    }

    if $(reduction) == $(0) {
      tl.store(loss_ptr + offsets, loss, mask=mask)
    } else {
      loss = tl.sum(loss, axis=0)
      tl.store(loss_ptr, loss)
      loss_ptr += $(1)
    }
  }
}

/-- The full KL-divergence forward surface lowers to the algorithm layer,
including the `log_target` branch, dynamic column loop, and reduction modes. -/
theorem kldiv_forward_surface_toAlgorithm_supported
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (log_target : Bool) (reduction : Nat) :
    ∃ alg,
      (kldiv_forward_surface y_ptr gt_ptr loss_ptr y_stride gt_stride
        loss_stride n_cols BLOCK_SIZE log_target reduction).toAlgorithm? =
        Except.ok alg := by
  simp [kldiv_forward_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Faithful transcription of `kldiv_triton.py`'s `_kldiv_kernel_backward`
for the `log_target = False` constexpr branch.

Includes the Python `for i in range(0, n_cols, BLOCK_SIZE)` loop. Proofs
target the Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`.

Allowed mechanical Lean-syntax-only changes:
- Python `log_target: tl.constexpr` → separate kernel defs per branch. -/
def kldiv_backward_default
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  input_ptr += pid * $(input_stride)
  target_ptr += pid * $(target_stride)
  base_offsets = tl.arange(0, $(BLOCK_SIZE))
  for i in range($(0), $(n_cols), $(BLOCK_SIZE)) {
    offsets = i + base_offsets
    mask = offsets < $(n_cols)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    res = target * -1
    tl.store(input_ptr + offsets, res, mask=mask)
  }
}

/-- Faithful transcription of `kldiv_triton.py`'s `_kldiv_kernel_forward`
for the `log_target = False`, `reduction = 0` (None) constexpr branch.

Includes the Python `for i in range(0, n_cols, BLOCK_SIZE)` loop; proofs
target the Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`.
Mirrors the Python `loss = y_true * (tl.log(y_true) - y)` body; no `eps`
clamp (unlike the kldiv_ops port). -/
def kldiv_forward_default_none
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  y_ptr += pid * $(y_stride)
  gt_ptr += pid * $(gt_stride)
  loss_ptr += pid * $(loss_stride)
  base_offsets = tl.arange(0, $(BLOCK_SIZE))
  for i in range($(0), $(n_cols), $(BLOCK_SIZE)) {
    offsets = i + base_offsets
    mask = offsets < $(n_cols)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
    loss = y_true * (tl.log(y_true) - y)
    tl.store(loss_ptr + offsets, loss, mask=mask)
  }
}

/-- Faithful transcription of `kldiv_triton.py`'s `_kldiv_kernel_forward`
for the `log_target = True`, `reduction = 0` (None) constexpr branch.

Includes the Python `for i in range(0, n_cols, BLOCK_SIZE)` loop with the
elementwise-store path; proofs target the Python-tested single-chunk regime
`0 < n_cols ≤ BLOCK_SIZE`. -/
def kldiv_forward_log_target_none
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  y_ptr += pid * $(y_stride)
  gt_ptr += pid * $(gt_stride)
  loss_ptr += pid * $(loss_stride)
  base_offsets = tl.arange(0, $(BLOCK_SIZE))
  for i in range($(0), $(n_cols), $(BLOCK_SIZE)) {
    offsets = i + base_offsets
    mask = offsets < $(n_cols)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
    loss = tl.exp(y_true) * (y_true - y)
    tl.store(loss_ptr + offsets, loss, mask=mask)
  }
}

/-- Faithful transcription of `_kldiv_kernel_backward` for the
`log_target = True` constexpr branch. Includes the Python
`for i in range(0, n_cols, BLOCK_SIZE)` loop; proofs target the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
def kldiv_backward_log_target
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  input_ptr += pid * $(input_stride)
  target_ptr += pid * $(target_stride)
  base_offsets = tl.arange(0, $(BLOCK_SIZE))
  for i in range($(0), $(n_cols), $(BLOCK_SIZE)) {
    offsets = i + base_offsets
    mask = offsets < $(n_cols)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    res = -tl.exp(target)
    tl.store(input_ptr + offsets, res, mask=mask)
  }
}


def inOffset (s : BlockState) (target_stride : Nat) (i : Fin BLOCK_SIZE) :
    Nat :=
  s.pid * target_stride + i.val

noncomputable def defaultSpec
    (s : BlockState) (target_ptr : RegionName) (target_stride : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem target_ptr (inOffset s target_stride i) * (0.0 - 1)

noncomputable def logTargetSpec
    (s : BlockState) (target_ptr : RegionName) (target_stride : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  0.0 - Real.exp (s.readMem target_ptr (inOffset s target_stride i))

/-- Forward KL-divergence per-element value (`log_target = False`)
for the `reduction = 0` (None) elementwise-store path. -/
noncomputable def forwardDefaultSpec
    (s : BlockState) (y_ptr gt_ptr : RegionName)
    (y_stride gt_stride : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  let y := s.readMem y_ptr (inOffset s y_stride i)
  let y_true := s.readMem gt_ptr (inOffset s gt_stride i)
  y_true * (Real.log y_true - y)

/-- Forward KL-divergence per-element value (`log_target = True`)
for the `reduction = 0` (None) elementwise-store path. -/
noncomputable def forwardLogTargetSpec
    (s : BlockState) (y_ptr gt_ptr : RegionName)
    (y_stride gt_stride : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  let y := s.readMem y_ptr (inOffset s y_stride i)
  let y_true := s.readMem gt_ptr (inOffset s gt_stride i)
  Real.exp y_true * (y_true - y)

/-- Algorithm-layer correctness for the default backward kernel under the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
theorem kldiv_backward_default_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s input_stride i))
    (hExec : exec (kldiv_backward_default input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem input_ptr (linearOffset s input_stride i) =
        if i.val < n_cols then
          defaultSpec s target_ptr target_stride i
        else s.readMem input_ptr (linearOffset s input_stride i) := by
  intro i
  have hBSne : BLOCK_SIZE ≠ 0 := Nat.pos_iff_ne_zero.mp hBS
  simp [exec, kldiv_backward_default, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.uop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt, hBS, hBSne, hLenPos, hLen,
        stepForRangeAux.forRangeDyn_unfold, stepForRangeAux.step_lt,
        stepForRangeAux.step_ge, Nat.zero_add] at hExec
  rw [← hExec]
  simp only [linearOffset]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  by_cases h : i.val < n_cols
  · simp [defaultSpec, inOffset, h]
  · simp [h]
/-- Algorithm-layer correctness for the log-target backward kernel under the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
theorem kldiv_backward_log_target_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s input_stride i))
    (hExec : exec (kldiv_backward_log_target input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem input_ptr (linearOffset s input_stride i) =
        if i.val < n_cols then
          logTargetSpec s target_ptr target_stride i
        else s.readMem input_ptr (linearOffset s input_stride i) := by
  intro i
  have hBSne : BLOCK_SIZE ≠ 0 := Nat.pos_iff_ne_zero.mp hBS
  simp [exec, kldiv_backward_log_target, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.uop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt, hBS, hBSne, hLenPos, hLen,
        stepForRangeAux.forRangeDyn_unfold, stepForRangeAux.step_lt,
        stepForRangeAux.step_ge, Nat.zero_add] at hExec
  rw [← hExec]
  simp only [linearOffset]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  by_cases h : i.val < n_cols
  · simp [logTargetSpec, inOffset, h]
  · simp [h]
/-- Compute-facing correctness for the default backward kernel under the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
specification kldiv_backward_default_compute_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s input_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_backward_default input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (input_ptr, linearOffset s input_stride i)))
      (expected := fun i => defaultSpec s target_ptr target_stride i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kldiv_backward_default]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := kldiv_backward_default_correct input_ptr target_ptr
    input_stride target_stride n_cols BLOCK_SIZE s s'
    hBS hLen hLenPos hOutInj hExec i
  simpa [hActive] using h
/-- Compute-facing correctness for the log-target backward kernel under the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
specification kldiv_backward_log_target_compute_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s input_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_backward_log_target input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (input_ptr, linearOffset s input_stride i)))
      (expected := fun i => logTargetSpec s target_ptr target_stride i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kldiv_backward_log_target]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := kldiv_backward_log_target_correct input_ptr target_ptr
    input_stride target_stride n_cols BLOCK_SIZE s s'
    hBS hLen hLenPos hOutInj hExec i
  simpa [hActive] using h
/-- Algorithm-layer correctness for the forward `log_target=False`,
`reduction=0` kernel under the Python-tested single-chunk regime
`0 < n_cols ≤ BLOCK_SIZE`. -/
theorem kldiv_forward_default_none_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i))
    (hExec : exec (kldiv_forward_default_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem loss_ptr (linearOffset s loss_stride i) =
        if i.val < n_cols then
          forwardDefaultSpec s y_ptr gt_ptr y_stride gt_stride i
        else s.readMem loss_ptr (linearOffset s loss_stride i) := by
  intro i
  have hBSne : BLOCK_SIZE ≠ 0 := Nat.pos_iff_ne_zero.mp hBS
  simp [exec, kldiv_forward_default_none, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.uop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt, hBS, hBSne, hLenPos, hLen,
        stepForRangeAux.forRangeDyn_unfold, stepForRangeAux.step_lt,
        stepForRangeAux.step_ge, Nat.zero_add] at hExec
  rw [← hExec]
  simp only [linearOffset]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  by_cases h : i.val < n_cols
  · rw [← Int.natCast_mul, Int.toNat_natCast]
    simp [forwardDefaultSpec, inOffset, h]
  · simp [h]

/-- Compute-facing correctness for the forward `log_target=False`,
`reduction=0` kernel under the Python-tested single-chunk regime
`0 < n_cols ≤ BLOCK_SIZE`. -/
specification kldiv_forward_default_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_forward_default_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (loss_ptr, linearOffset s loss_stride i)))
      (expected := fun i =>
        forwardDefaultSpec s y_ptr gt_ptr y_stride gt_stride i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kldiv_forward_default_none]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := kldiv_forward_default_none_correct y_ptr gt_ptr loss_ptr
    y_stride gt_stride loss_stride n_cols BLOCK_SIZE s s'
    hBS hLen hLenPos hOutInj hExec i
  simpa [hActive] using h

/-- Algorithm-layer correctness for the forward `log_target=True`,
`reduction=0` kernel under the Python-tested single-chunk regime
`0 < n_cols ≤ BLOCK_SIZE`. -/
theorem kldiv_forward_log_target_none_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i))
    (hExec : exec (kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem loss_ptr (linearOffset s loss_stride i) =
        if i.val < n_cols then
          forwardLogTargetSpec s y_ptr gt_ptr y_stride gt_stride i
        else s.readMem loss_ptr (linearOffset s loss_stride i) := by
  intro i
  have hBSne : BLOCK_SIZE ≠ 0 := Nat.pos_iff_ne_zero.mp hBS
  simp [exec, kldiv_forward_log_target_none, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.uop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt, hBS, hBSne, hLenPos, hLen,
        stepForRangeAux.forRangeDyn_unfold, stepForRangeAux.step_lt,
        stepForRangeAux.step_ge, Nat.zero_add] at hExec
  rw [← hExec]
  simp only [linearOffset]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  by_cases h : i.val < n_cols
  · rw [← Int.natCast_mul, Int.toNat_natCast]
    simp [forwardLogTargetSpec, inOffset, h]
  · simp [h]

/-- Compute-facing correctness for the forward log-target kernel under the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
specification kldiv_forward_log_target_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (loss_ptr, linearOffset s loss_stride i)))
      (expected := fun i =>
        forwardLogTargetSpec s y_ptr gt_ptr y_stride gt_stride i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kldiv_forward_log_target_none]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := kldiv_forward_log_target_none_correct y_ptr gt_ptr loss_ptr
    y_stride gt_stride loss_stride n_cols BLOCK_SIZE s s'
    hBS hLen hLenPos hOutInj hExec i
  simpa [hActive] using h

/-! ## The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre)

Everything below is purely additive; the exact surfaces above are untouched.
This is a consumer of the per-step emit skin `StreamEmitMasked2DKernelIO₂`
(streaming genre, style S3): the `tl.store(loss_ptr + offsets, loss,
mask=mask)` sits **inside** `for i in range(0, n_cols, BLOCK_SIZE)`, so the
output is a per-step `BLOCK_SIZE`-lane window family rather than one terminal
tile, and the kernel's spec `f t j` is the genre's *elementwise* shape (no
cross-iteration carried state).

Unlike the exact surfaces above — which target the Python-tested single-chunk
regime `0 < n_cols ≤ BLOCK_SIZE` — this face is the **general multi-chunk**
one: `T := ⌈n_cols / BLOCK_SIZE⌉`, driven by a genuine `forRange` invariant
(`klInv`) over the whole stream, with no bound relating `n_cols` and
`BLOCK_SIZE`. The price is the two truth-forced non-aliasing hypotheses
`loss_ptr ≠ y_ptr` / `loss_ptr ≠ gt_ptr` (later chunks load *after* earlier
chunks have stored), which the single-chunk surfaces do not need.

Structure of the `execR R` story: this kernel has **zero rounding events**.
Both loads and the in-loop store are at `.real`, and there is no `castFloat`
anywhere (the only casts are the `int64` program-id pointer arithmetic), so
the whole body collapses verbatim onto the exact stepper
(`stepForRangeAuxR_castFree`). The skin's readback contract at the default
`outDType := .real` grid carries `R.round .real`, the identity by
`round_real` — the ∀-`R` face is the exact streaming contract via the model's
`.real` identity fields, not a `.triv` special case. -/

section IOFace

open scoped VeriTile.Triton.StreamEmitMasked2DKernelIO₂

set_option linter.unusedVariables false

/-! ### Stream geometry -/

/-- Trip count of `for i in range(0, n_cols, BLOCK_SIZE)`:
`⌈n_cols / BLOCK_SIZE⌉`. -/
def klNumSteps (n_cols BLOCK_SIZE : Nat) : Nat := (n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE

/-- The stream covers the column axis: `n_cols ≤ ⌈n_cols/B⌉ · B`. -/
private theorem klNumSteps_mul_ge (n B : Nat) (hB : 0 < B) : n ≤ klNumSteps n B * B := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · exact Nat.zero_le _
  · unfold klNumSteps
    have heq : n + B - 1 = (n - 1) + B := by omega
    rw [heq, Nat.add_div_right _ hB]
    have h2 : (n - 1) % B + 1 ≤ B := Nat.mod_lt _ hB
    calc n = (n - 1) + 1 := by omega
      _ = (n - 1) / B * B + ((n - 1) % B + 1) := by
          rw [← Nat.add_assoc, Nat.div_add_mod']
      _ ≤ (n - 1) / B * B + B := Nat.add_le_add_left h2 _
      _ = ((n - 1) / B + 1) * B := (Nat.succ_mul _ _).symm

/-- Every in-range column index sits in a step of the stream. -/
private theorem klStep_lt_numSteps (n B i : Nat) (hB : 0 < B) (hi : i < n) :
    i / B < klNumSteps n B := by
  have h2 : i / B * B < klNumSteps n B * B :=
    Nat.lt_of_le_of_lt (Nat.div_mul_le_self i B)
      (Nat.lt_of_lt_of_le hi (klNumSteps_mul_ge n B hB))
  exact Nat.lt_of_mul_lt_mul_right h2

/-! ### The lowered program -/

/-- The lowered 5-statement prologue: the `int64` program id, the three
per-row pointer advances (`ptr += pid * stride`), and `base_offsets =
tl.arange(0, BLOCK_SIZE)`. Transcribed from the kernel's own lowering (see
`kl_body_decomp`, which is `rfl`). -/
def klPrefixStmts (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride BLOCK_SIZE : Nat) : List Stmt :=
  [ Stmt.assign TileDType.int [] "pid" (Op.programId 0).castNatToInt,
    Stmt.assign TileDType.ptr [] "y_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase y_ptr)
        (Op.mul NumericDType.int Broadcast.nil (Op.ref TileDType.int [] "pid")
            (Op.constNat y_stride).castNatToInt).castIntToNat),
    Stmt.assign TileDType.ptr [] "gt_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase gt_ptr)
        (Op.mul NumericDType.int Broadcast.nil (Op.ref TileDType.int [] "pid")
            (Op.constNat gt_stride).castNatToInt).castIntToNat),
    Stmt.assign TileDType.ptr [] "loss_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase loss_ptr)
        (Op.mul NumericDType.int Broadcast.nil (Op.ref TileDType.int [] "pid")
            (Op.constNat loss_stride).castNatToInt).castIntToNat),
    Stmt.assign TileDType.nat [BLOCK_SIZE] "base_offsets" (Op.arange BLOCK_SIZE) ]

/-- The lowered loop body: `offsets`, `mask`, the two masked `other=0.0`
loads, the elementwise `loss = y_true * (log(y_true) - y)`, and the **in-loop**
masked store — the S3 "emit" shape. Both loads and the store are
`TileDType.real`; the store carries no cast, hence the `.real` output grid. -/
def klLossBody (n_cols BLOCK_SIZE : Nat) : List Stmt :=
  [ Stmt.assign TileDType.nat [BLOCK_SIZE] "offsets"
      (Op.add NumericDType.nat Broadcast.scalarL (Op.ref TileDType.nat [] "i")
        (Op.ref TileDType.nat [BLOCK_SIZE] "base_offsets")),
    Stmt.assign TileDType.bool [BLOCK_SIZE] "mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BLOCK_SIZE] "offsets")
        (Op.constNat n_cols)),
    Stmt.assign TileDType.real [BLOCK_SIZE] "y"
      (Op.load TileDType.real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "y_ptr")
            (Op.ref TileDType.nat [BLOCK_SIZE] "offsets")))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BLOCK_SIZE] "mask")
          ((Op.const 0.0).broadcast [BLOCK_SIZE]))),
    Stmt.assign TileDType.real [BLOCK_SIZE] "y_true"
      (Op.load TileDType.real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "gt_ptr")
            (Op.ref TileDType.nat [BLOCK_SIZE] "offsets")))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BLOCK_SIZE] "mask")
          ((Op.const 0.0).broadcast [BLOCK_SIZE]))),
    Stmt.assign TileDType.real [BLOCK_SIZE] "loss"
      (Op.mul NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BLOCK_SIZE] "y_true")
        (Op.sub NumericDType.real Broadcast.nil.consSame
          (Op.ref TileDType.real [BLOCK_SIZE] "y_true").log
          (Op.ref TileDType.real [BLOCK_SIZE] "y"))),
    Stmt.store TileDType.real [BLOCK_SIZE]
      (MemAccess.ptr
        (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "loss_ptr")
          (Op.ref TileDType.nat [BLOCK_SIZE] "offsets")))
      (Op.ref TileDType.real [BLOCK_SIZE] "loss")
      (MaskOpt.mask (Op.ref TileDType.bool [BLOCK_SIZE] "mask")) ]

/-- The kernel's algorithm-layer body is the prologue followed by the single
static `forRange` over the column stream. Definitional (`rfl`). -/
theorem kl_body_decomp (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) :
    ((kldiv_forward_default_none y_ptr gt_ptr loss_ptr y_stride gt_stride
        loss_stride n_cols BLOCK_SIZE).toAlgKernel).body
      = klPrefixStmts y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride BLOCK_SIZE
        ++ [Stmt.forRange "i" 0 n_cols BLOCK_SIZE (klLossBody n_cols BLOCK_SIZE)] := rfl

/-! ### Per-statement recipes -/

/-- Per-lane value of `offsets = i + base_offsets` at loop counter `i`. -/
private theorem kl_offsets_eval (B i : Nat) (s : BlockState)
    (hbase : s.regs .nat [B] "base_offsets" = some (Tile.vec (fun j : Fin B => j.val))) :
    evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "i")
        (Op.ref .nat [B] "base_offsets")) (s.setReg "i" .nat [] (Tile.scalar i))
      = some (Tile.vec (fun j : Fin B => i + j.val)) := by
  rw [evalOp_add, evalOp_ref_setReg_same,
    evalOp_ref_setReg_ne_name _ _ _ _ _ _ _ _ (show ("base_offsets":RegName) ≠ "i" by decide),
    evalOp_ref, hbase]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
    Tile.scalar, Tile.vec, NumericDType.add]

/-- Per-lane value of `mask = offsets < n_cols`. -/
private theorem kl_mask_eval (n_cols B i : Nat) (s : BlockState)
    (hoffsets : s.regs .nat [B] "offsets" = some (Tile.vec (fun j : Fin B => i + j.val))) :
    evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [B] "offsets") (Op.constNat n_cols)) s
      = some (Tile.vec (fun j : Fin B => decide (i + j.val < n_cols))) := by
  rw [evalOp_lt, evalOp_ref, hoffsets, evalOp_constNat]
  apply congrArg some
  ext j
  simp only [Tile.cop_data, Tile.vec_data, Tile.scalar_data, Broadcast.leftIndex_scalarR,
    Broadcast.rightIndex_scalarR, ComparableDType.lt, decide_eq_decide]

/-- Per-lane value of a masked pointer load `tl.load(ptr + offsets,
mask=mask, other=0.0)`: active lanes read the pinned launch-state memory,
masked-off lanes take the `other=0.0` default (so no `undef` enters). -/
private theorem kl_load_eval (ptrReg : RegName) (r : RegionName) (base n_cols B i : Nat)
    (s0 s : BlockState)
    (hrm : s.readMem r = s0.readMem r)
    (hptr : s.regs .ptr [] ptrReg = some (Tile.scalar ((r : RegionName), base)))
    (hoffsets : s.regs .nat [B] "offsets" = some (Tile.vec (fun j : Fin B => i + j.val)))
    (hmask : s.regs .bool [B] "mask"
      = some (Tile.vec (fun j : Fin B => decide (i + j.val < n_cols)))) :
    evalOp (Op.load .real
        (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] ptrReg)
          (Op.ref .nat [B] "offsets")))
        (MaskOpt.maskOther (Op.ref .bool [B] "mask")
          (Op.broadcast (Op.const 0.0) [B]))) s
      = some ⟨fun j : TileIndex [B] =>
          some (if i + j.1.val < n_cols then s0.readMem r (base + (i + j.1.val)) else 0)⟩ := by
  simp only [evalOp, hptr, hoffsets, hmask]
  apply congrArg some
  ext j
  simp only [Tile.ptrAdd, Tile.scalar, Tile.vec, Broadcast.leftIndex_scalarL,
    Broadcast.rightIndex_scalarL, BlockState.readMemValue_real, hrm]
  by_cases h : i + j.1.val < n_cols
  · simp [h]
  · norm_num [h]

/-- Per-lane value of `loss = y_true * (tl.log(y_true) - y)`: the `WithBot ℝ`
carriers of the two loaded tiles combine to the real closed form. -/
private theorem kl_loss_eval (B : Nat) (s : BlockState) (g h : Fin B → ℝ)
    (hyt : s.regs .real [B] "y_true" = some ⟨fun j : TileIndex [B] => some (g j.1)⟩)
    (hy : s.regs .real [B] "y" = some ⟨fun j : TileIndex [B] => some (h j.1)⟩) :
    evalOp (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [B] "y_true")
        (Op.sub .real (Broadcast.consSame Broadcast.nil)
          (Op.log (Op.ref .real [B] "y_true"))
          (Op.ref .real [B] "y"))) s
      = some ⟨fun j : TileIndex [B] => some (g j.1 * (Real.log (g j.1) - h j.1))⟩ := by
  simp only [evalOp, hyt, hy, Option.bind_eq_bind, Option.bind_some]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Tile.uop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, NumericDType.sub, WithBot.realLog_some, WithBot.realMul,
    WithBot.realSub, Option.map₂, Option.bind, Option.map]

/-- The in-loop masked store `tl.store(loss_ptr + offsets, loss, mask=mask)`
steps to the masked scatter `foldl` over the block's lanes. The store is
`.real`, so `writeMemTyped` is the exact `writeMem`. -/
private theorem kl_store_step (o : RegionName) (B base : Nat) (s : BlockState)
    (offs : Fin B → Nat) (vals : Fin B → ℝ) (mask : Fin B → Bool)
    (hptr : s.regs .ptr [] "loss_ptr" = some (Tile.scalar ((o : RegionName), base)))
    (hoffsets : s.regs .nat [B] "offsets" = some (Tile.vec offs))
    (hloss : s.regs .real [B] "loss" = some ⟨fun j : TileIndex [B] => some (vals j.1)⟩)
    (hmask : s.regs .bool [B] "mask" = some (Tile.vec mask)) :
    stepStmt (Stmt.store .real [B]
        (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "loss_ptr")
          (Op.ref .nat [B] "offsets")))
        (Op.ref .real [B] "loss") (MaskOpt.mask (Op.ref .bool [B] "mask"))) s
      = some ((TileShape.allIndices [B]).foldl
          (fun acc (j : TileIndex [B]) =>
            if mask j.1 then acc.writeMem o (base + offs j.1) (vals j.1) else acc) s) := by
  simp only [stepStmt, evalOp_ref, evalOp_ptrAdd, hptr, hoffsets, hloss, hmask,
    Option.bind_eq_bind, Option.bind_some, Option.map, Tile.ptrAdd, Tile.vec, Tile.scalar,
    Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
    BlockState.writeMemTyped_real, FloatDType.real_storeValue, WithBot.unbotD_some]


/-! ### The stream-level spec and the loop invariant -/

/-- The forward KL value of output column `m` of row `pid₀`, as a closed form
over the **launch-state** input memory: `y_true · (log y_true − y)`. -/
noncomputable def klCell (y_ptr gt_ptr : RegionName) (y_stride gt_stride : Nat)
    (s₀ : BlockState) (m : Nat) : ℝ :=
  s₀.readMem gt_ptr (s₀.pids 0 * gt_stride + m) *
    (Real.log (s₀.readMem gt_ptr (s₀.pids 0 * gt_stride + m))
      - s₀.readMem y_ptr (s₀.pids 0 * y_stride + m))

/-- The column-loop invariant at counter `i`. The body is purely elementwise
— no cross-iteration carried state — so the invariant only has to pin the
prologue's registers, record which output cells have already been emitted, and
carry the per-cell memory frame:

* the counter is a multiple of `BLOCK_SIZE` (the loop's stride);
* the three advanced pointer registers and `base_offsets` still hold their
  prologue values;
* every output cell `m < n_cols` holds `klCell m` when `m < i` and its
  original launch-state value otherwise;
* every cell outside the `{loss_ptr} × {base + m : m < n_cols}` write window is
  bit-identical to the launch state.

The frame clause is what lets the later iterations' loads be re-pinned at the
launch state (`loss_ptr ≠ y_ptr` / `≠ gt_ptr`). -/
def klInv (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols B : Nat)
    (s₀ : BlockState) (i : Nat) (st : BlockState) : Prop :=
  B ∣ i ∧
  st.regs .ptr [] "y_ptr"
      = some (Tile.scalar ((y_ptr : RegionName), s₀.pids 0 * y_stride)) ∧
  st.regs .ptr [] "gt_ptr"
      = some (Tile.scalar ((gt_ptr : RegionName), s₀.pids 0 * gt_stride)) ∧
  st.regs .ptr [] "loss_ptr"
      = some (Tile.scalar ((loss_ptr : RegionName), s₀.pids 0 * loss_stride)) ∧
  st.regs .nat [B] "base_offsets" = some (Tile.vec (fun j : Fin B => j.val)) ∧
  (∀ m, m < n_cols → st.readMem loss_ptr (s₀.pids 0 * loss_stride + m)
      = if m < i then klCell y_ptr gt_ptr y_stride gt_stride s₀ m
        else s₀.readMem loss_ptr (s₀.pids 0 * loss_stride + m)) ∧
  (∀ r oo, (r ≠ loss_ptr ∨ ∀ m, m < n_cols → oo ≠ s₀.pids 0 * loss_stride + m) →
      st.mem r oo = s₀.mem r oo)

/-- Cell-level frame of a `Bool`-masked exact `writeMem` scatter `foldl`:
every cell not hit by an active lane is untouched. (The cell-level sibling of
the shared `foldl_store_preserve`, which is stated for `readMem`.) -/
private theorem kl_foldl_writeMem_preserve_cell {α : Type}
    {region : RegionName} (ofn : α → Nat) (vfn : α → ℝ) (mask : α → Bool)
    (r : RegionName) (oo : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, mask k → ¬(r = region ∧ oo = ofn k)) :
    (l.foldl (fun acc k =>
        if mask k then acc.writeMem region (ofn k) (vfn k) else acc) s).mem r oo
      = s.mem r oo := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      cases hm : mask hd
      · simp only [hm, Bool.false_eq_true, if_false]
        exact ih _ fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk
      · simp only [hm, if_true]
        rw [ih _ fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk,
          BlockState.writeMem_mem]
        exact if_neg (hnot hd List.mem_cons_self (by rw [hm]))

/-! ### The prologue -/

/-- Value of a prologue pointer advance `ptr += pid * stride`. The Python
`pid = tl.program_id(0).to(tl.int64)` makes the multiply an `int` one, so the
address is `Int.toNat (↑pid * ↑stride) = pid * stride`. -/
private theorem kl_ptrshift_eval (r : RegionName) (stride p : Nat) (s : BlockState)
    (hpid : s.regs .int [] "pid" = some (Tile.scalar (Int.ofNat p))) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase r)
        (Op.castIntToNat (Op.mul .int Broadcast.nil (Op.ref .int [] "pid")
          (Op.castNatToInt (Op.constNat stride))))) s
      = some (Tile.scalar ((r : RegionName), p * stride)) := by
  have hmul : Int.toNat (Int.ofNat p * Int.ofNat stride) = p * stride := by
    exact congrArg Int.toNat (rfl : Int.ofNat p * Int.ofNat stride = Int.ofNat (p * stride))
  simp only [evalOp, hpid, Option.bind_eq_bind, Option.bind_some]
  apply congrArg some
  apply Tile.ext
  intro j
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul, Region.cast_id, Nat.zero_add, hmul]

/-- The prologue runs from an arbitrary launch state and establishes the loop
invariant at counter `0` (it never touches memory). -/
private theorem kl_preLoop (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols B : Nat) (s₀ : BlockState) :
    ∃ sp, stepStmts (klPrefixStmts y_ptr gt_ptr loss_ptr y_stride gt_stride
        loss_stride B) s₀ = some sp
      ∧ klInv y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride n_cols B s₀ 0 sp := by
  unfold klPrefixStmts
  have hpid0 : evalOp (Op.castNatToInt (Op.programId 0)) s₀
      = some (Tile.scalar (Int.ofNat (s₀.pids 0))) := by
    simp only [evalOp]
    rfl
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hpid0)]
  set p1 := s₀.setReg "pid" .int [] (Tile.scalar (Int.ofNat (s₀.pids 0))) with hp1
  have hpidr1 : p1.regs .int [] "pid" = some (Tile.scalar (Int.ofNat (s₀.pids 0))) := by
    rw [hp1]; exact BlockState.setReg_same _ _ _ _ _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (kl_ptrshift_eval y_ptr y_stride (s₀.pids 0) p1 hpidr1))]
  set p2 := p1.setReg "y_ptr" .ptr [] (Tile.scalar ((y_ptr : RegionName), s₀.pids 0 * y_stride)) with hp2
  have hpidr2 : p2.regs .int [] "pid" = some (Tile.scalar (Int.ofNat (s₀.pids 0))) := by
    rw [hp2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("pid":RegName) ≠ "y_ptr" by decide)]
    exact hpidr1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (kl_ptrshift_eval gt_ptr gt_stride (s₀.pids 0) p2 hpidr2))]
  set p3 := p2.setReg "gt_ptr" .ptr [] (Tile.scalar ((gt_ptr : RegionName), s₀.pids 0 * gt_stride)) with hp3
  have hpidr3 : p3.regs .int [] "pid" = some (Tile.scalar (Int.ofNat (s₀.pids 0))) := by
    rw [hp3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("pid":RegName) ≠ "gt_ptr" by decide)]
    exact hpidr2
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (kl_ptrshift_eval loss_ptr loss_stride (s₀.pids 0) p3 hpidr3))]
  set p4 := p3.setReg "loss_ptr" .ptr [] (Tile.scalar ((loss_ptr : RegionName), s₀.pids 0 * loss_stride)) with hp4
  have harange : evalOp (Op.arange B) p4 = some (Tile.vec (fun j : Fin B => j.val)) := by
    simp only [evalOp_arange]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some harange), stepStmts.nil]
  refine ⟨_, rfl, ⟨0, rfl⟩, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("y_ptr":RegName) ≠ "base_offsets" by decide),
      hp4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("y_ptr":RegName) ≠ "loss_ptr" by decide),
      hp3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("y_ptr":RegName) ≠ "gt_ptr" by decide),
      hp2]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("gt_ptr":RegName) ≠ "base_offsets" by decide),
      hp4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("gt_ptr":RegName) ≠ "loss_ptr" by decide),
      hp3]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("loss_ptr":RegName) ≠ "base_offsets" by decide),
      hp4]
    exact BlockState.setReg_same _ _ _ _ _
  · exact BlockState.setReg_same _ _ _ _ _
  · intro m hm
    rw [if_neg (by omega)]
    simp only [hp4, hp3, hp2, hp1, BlockState.setReg_readMem]
  · intro r oo _
    simp only [hp4, hp3, hp2, hp1, BlockState.setReg_mem]


/-! ### One loop iteration -/

set_option maxHeartbeats 8000000 in
/-- **One column-loop iteration re-establishes the invariant.** The body's two
masked loads read the launch-state inputs (the invariant's frame plus
`loss_ptr ≠ y_ptr` / `≠ gt_ptr`), the elementwise arithmetic produces `klCell`
on every active lane, and the masked store extends the emitted window from
`[0, i)` to `[0, i + BLOCK_SIZE)` while leaving every other cell alone. -/
private theorem klInv_step (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols B : Nat)
    (hly : loss_ptr ≠ y_ptr) (hlg : loss_ptr ≠ gt_ptr)
    (s₀ : BlockState) (i : Nat) (st : BlockState) (hi : i < n_cols)
    (hP : klInv y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride n_cols B s₀ i st) :
    ∃ st', stepStmts (klLossBody n_cols B)
        (st.setReg "i" .nat [] (Tile.scalar i)) = some st'
      ∧ klInv y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride n_cols B s₀ (i + B) st' := by
  obtain ⟨hdvd, hyp, hgp, hlp, hbase, hread, hframe⟩ := hP
  -- the streamed values, pinned at `s₀`
  obtain ⟨X, hX⟩ : ∃ f : Fin B → ℝ, f = fun j =>
      if i + j.val < n_cols then s₀.readMem y_ptr (s₀.pids 0 * y_stride + (i + j.val))
      else 0 := ⟨_, rfl⟩
  obtain ⟨Y, hY⟩ : ∃ f : Fin B → ℝ, f = fun j =>
      if i + j.val < n_cols then s₀.readMem gt_ptr (s₀.pids 0 * gt_stride + (i + j.val))
      else 0 := ⟨_, rfl⟩
  have hrmY : st.readMem y_ptr = s₀.readMem y_ptr := by
    funext oo
    unfold BlockState.readMem
    rw [hframe y_ptr oo (Or.inl hly.symm)]
  have hrmG : st.readMem gt_ptr = s₀.readMem gt_ptr := by
    funext oo
    unfold BlockState.readMem
    rw [hframe gt_ptr oo (Or.inl hlg.symm)]
  unfold klLossBody
  -- offsets
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (kl_offsets_eval B i st hbase))]
  set u1 := (st.setReg "i" .nat [] (Tile.scalar i)).setReg "offsets" .nat [B]
    (Tile.vec (fun j : Fin B => i + j.val)) with hu1
  have ho1 : u1.regs .nat [B] "offsets" = some (Tile.vec (fun j : Fin B => i + j.val)) := by
    rw [hu1]; exact BlockState.setReg_same _ _ _ _ _
  -- mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (kl_mask_eval n_cols B i u1 ho1))]
  set u2 := u1.setReg "mask" .bool [B]
    (Tile.vec (fun j : Fin B => decide (i + j.val < n_cols))) with hu2
  have ho2 : u2.regs .nat [B] "offsets" = some (Tile.vec (fun j : Fin B => i + j.val)) := by
    rw [hu2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("offsets":RegName) ≠ "mask" by decide)]
    exact ho1
  have hm2 : u2.regs .bool [B] "mask"
      = some (Tile.vec (fun j : Fin B => decide (i + j.val < n_cols))) := by
    rw [hu2]; exact BlockState.setReg_same _ _ _ _ _
  have hyp2 : u2.regs .ptr [] "y_ptr"
      = some (Tile.scalar ((y_ptr : RegionName), s₀.pids 0 * y_stride)) := by
    rw [hu2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("y_ptr":RegName) ≠ "mask" by decide), hu1,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("y_ptr":RegName) ≠ "offsets" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("y_ptr":RegName) ≠ "i" by decide)]
    exact hyp
  have hgp2 : u2.regs .ptr [] "gt_ptr"
      = some (Tile.scalar ((gt_ptr : RegionName), s₀.pids 0 * gt_stride)) := by
    rw [hu2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("gt_ptr":RegName) ≠ "mask" by decide), hu1,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("gt_ptr":RegName) ≠ "offsets" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("gt_ptr":RegName) ≠ "i" by decide)]
    exact hgp
  have hrmY2 : u2.readMem y_ptr = s₀.readMem y_ptr := by
    rw [hu2, hu1]; exact hrmY
  have hrmG2 : u2.readMem gt_ptr = s₀.readMem gt_ptr := by
    rw [hu2, hu1]; exact hrmG
  -- y load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (kl_load_eval "y_ptr" y_ptr (s₀.pids 0 * y_stride) n_cols B i s₀ u2 hrmY2 hyp2 ho2 hm2)),
    show (⟨fun j : TileIndex [B] => some (if i + j.1.val < n_cols then
        s₀.readMem y_ptr (s₀.pids 0 * y_stride + (i + j.1.val)) else 0)⟩ : Tile .real [B])
      = ⟨fun j : TileIndex [B] => some (X j.1)⟩ from by rw [hX]]
  set u3 := u2.setReg "y" .real [B]
    (⟨fun j : TileIndex [B] => some (X j.1)⟩ : Tile .real [B]) with hu3
  have ho3 : u3.regs .nat [B] "offsets" = some (Tile.vec (fun j : Fin B => i + j.val)) := by
    rw [hu3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("offsets":RegName) ≠ "y" by decide)]
    exact ho2
  have hm3 : u3.regs .bool [B] "mask"
      = some (Tile.vec (fun j : Fin B => decide (i + j.val < n_cols))) := by
    rw [hu3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("mask":RegName) ≠ "y" by decide)]
    exact hm2
  have hgp3 : u3.regs .ptr [] "gt_ptr"
      = some (Tile.scalar ((gt_ptr : RegionName), s₀.pids 0 * gt_stride)) := by
    rw [hu3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("gt_ptr":RegName) ≠ "y" by decide)]
    exact hgp2
  have hrmG3 : u3.readMem gt_ptr = s₀.readMem gt_ptr := by
    rw [hu3]; exact hrmG2
  -- y_true load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (kl_load_eval "gt_ptr" gt_ptr (s₀.pids 0 * gt_stride) n_cols B i s₀ u3 hrmG3 hgp3 ho3 hm3)),
    show (⟨fun j : TileIndex [B] => some (if i + j.1.val < n_cols then
        s₀.readMem gt_ptr (s₀.pids 0 * gt_stride + (i + j.1.val)) else 0)⟩ : Tile .real [B])
      = ⟨fun j : TileIndex [B] => some (Y j.1)⟩ from by rw [hY]]
  set u4 := u3.setReg "y_true" .real [B]
    (⟨fun j : TileIndex [B] => some (Y j.1)⟩ : Tile .real [B]) with hu4
  have hyt4 : u4.regs .real [B] "y_true"
      = some (⟨fun j : TileIndex [B] => some (Y j.1)⟩ : Tile .real [B]) := by
    rw [hu4]; exact BlockState.setReg_same _ _ _ _ _
  have hy4 : u4.regs .real [B] "y"
      = some (⟨fun j : TileIndex [B] => some (X j.1)⟩ : Tile .real [B]) := by
    rw [hu4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("y":RegName) ≠ "y_true" by decide), hu3]
    exact BlockState.setReg_same _ _ _ _ _
  -- loss
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (kl_loss_eval B u4 Y X hyt4 hy4))]
  set u5 := u4.setReg "loss" .real [B]
    (⟨fun j : TileIndex [B] => some (Y j.1 * (Real.log (Y j.1) - X j.1))⟩ : Tile .real [B]) with hu5
  have hlp5 : u5.regs .ptr [] "loss_ptr"
      = some (Tile.scalar ((loss_ptr : RegionName), s₀.pids 0 * loss_stride)) := by
    rw [hu5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("loss_ptr":RegName) ≠ "loss" by decide), hu4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("loss_ptr":RegName) ≠ "y_true" by decide), hu3,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("loss_ptr":RegName) ≠ "y" by decide), hu2,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("loss_ptr":RegName) ≠ "mask" by decide), hu1,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("loss_ptr":RegName) ≠ "offsets" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("loss_ptr":RegName) ≠ "i" by decide)]
    exact hlp
  have ho5 : u5.regs .nat [B] "offsets" = some (Tile.vec (fun j : Fin B => i + j.val)) := by
    rw [hu5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offsets":RegName) ≠ "loss" by decide), hu4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offsets":RegName) ≠ "y_true" by decide)]
    exact ho3
  have hloss5 : u5.regs .real [B] "loss"
      = some (⟨fun j : TileIndex [B] => some (Y j.1 * (Real.log (Y j.1) - X j.1))⟩ : Tile .real [B]) := by
    rw [hu5]; exact BlockState.setReg_same _ _ _ _ _
  have hm5 : u5.regs .bool [B] "mask"
      = some (Tile.vec (fun j : Fin B => decide (i + j.val < n_cols))) := by
    rw [hu5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("mask":RegName) ≠ "loss" by decide), hu4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("mask":RegName) ≠ "y_true" by decide)]
    exact hm3
  -- the masked store
  rw [stepStmts.cons_some (kl_store_step loss_ptr B (s₀.pids 0 * loss_stride) u5
    (fun j : Fin B => i + j.val)
    (fun j : Fin B => Y j * (Real.log (Y j) - X j))
    (fun j : Fin B => decide (i + j.val < n_cols)) hlp5 ho5 hloss5 hm5),
    stepStmts.nil]
  have hyp5 : u5.regs .ptr [] "y_ptr"
      = some (Tile.scalar ((y_ptr : RegionName), s₀.pids 0 * y_stride)) := by
    rw [hu5, hu4, hu3, hu2, hu1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("y_ptr":RegName) ≠ "loss" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("y_ptr":RegName) ≠ "y_true" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("y_ptr":RegName) ≠ "y" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("y_ptr":RegName) ≠ "mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("y_ptr":RegName) ≠ "offsets" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("y_ptr":RegName) ≠ "i" by decide)]
    exact hyp
  have hgp5 : u5.regs .ptr [] "gt_ptr"
      = some (Tile.scalar ((gt_ptr : RegionName), s₀.pids 0 * gt_stride)) := by
    rw [hu5, hu4, hu3, hu2, hu1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("gt_ptr":RegName) ≠ "loss" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("gt_ptr":RegName) ≠ "y_true" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("gt_ptr":RegName) ≠ "y" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("gt_ptr":RegName) ≠ "mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("gt_ptr":RegName) ≠ "offsets" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("gt_ptr":RegName) ≠ "i" by decide)]
    exact hgp
  have hbase5 : u5.regs .nat [B] "base_offsets"
      = some (Tile.vec (fun j : Fin B => j.val)) := by
    rw [hu5, hu4, hu3, hu2, hu1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("base_offsets":RegName) ≠ "loss" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("base_offsets":RegName) ≠ "y_true" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("base_offsets":RegName) ≠ "y" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("base_offsets":RegName) ≠ "mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("base_offsets":RegName) ≠ "offsets" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("base_offsets":RegName) ≠ "i" by decide)]
    exact hbase
  have hmem5 : ∀ r oo, u5.mem r oo = st.mem r oo := by
    intro r oo; rw [hu5, hu4, hu3, hu2, hu1]; rfl
  have hrm5 : ∀ oo, u5.readMem loss_ptr oo = st.readMem loss_ptr oo := by
    intro oo; rw [hu5, hu4, hu3, hu2, hu1]; rfl
  refine ⟨_, rfl, dvd_add hdvd (dvd_refl B), ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [foldl_store_regs]; exact hyp5
  · rw [foldl_store_regs]; exact hgp5
  · rw [foldl_store_regs]; exact hlp5
  · rw [foldl_store_regs]; exact hbase5
  · -- the emitted window
    intro m hm
    by_cases hhit : i ≤ m ∧ m < i + B
    · obtain ⟨hlo, hhi⟩ := hhit
      have hjlt : m - i < B := by omega
      have hma : (fun k : TileIndex [B] => decide (i + k.1.val < n_cols))
          ((⟨m - i, hjlt⟩, PUnit.unit) : TileIndex [B]) = Bool.true := by
        show decide (i + (m - i) < n_cols) = Bool.true
        exact decide_eq_true (by omega)
      have hoa : (fun k : TileIndex [B] => s₀.pids 0 * loss_stride + (i + k.1.val))
          ((⟨m - i, hjlt⟩, PUnit.unit) : TileIndex [B]) = s₀.pids 0 * loss_stride + m := by
        show s₀.pids 0 * loss_stride + (i + (m - i)) = s₀.pids 0 * loss_stride + m
        omega
      have huniq : ∀ b ∈ TileShape.allIndices [B],
          (fun k : TileIndex [B] => decide (i + k.1.val < n_cols)) b = Bool.true →
          (fun k : TileIndex [B] => s₀.pids 0 * loss_stride + (i + k.1.val)) b
            = s₀.pids 0 * loss_stride + m →
          b = ((⟨m - i, hjlt⟩, PUnit.unit) : TileIndex [B]) := by
        intro b _ _ hb
        have hb' : s₀.pids 0 * loss_stride + (i + b.1.val)
            = s₀.pids 0 * loss_stride + m := hb
        refine Prod.ext (Fin.ext ?_) rfl
        show b.1.val = m - i
        omega
      rw [foldl_store_at (region := loss_ptr)
        (fun k : TileIndex [B] => s₀.pids 0 * loss_stride + (i + k.1.val))
        (fun k : TileIndex [B] => Y k.1 * (Real.log (Y k.1) - X k.1))
        (fun k : TileIndex [B] => decide (i + k.1.val < n_cols))
        (s₀.pids 0 * loss_stride + m) (TileShape.allIndices [B]) u5
        ((⟨m - i, hjlt⟩, PUnit.unit) : TileIndex [B])
        (TileShape.mem_allIndices _ _) hma hoa huniq (TileShape.allIndices_nodup _)]
      rw [if_pos (show m < i + B from hhi)]
      have hidx : i + ((⟨m - i, hjlt⟩, PUnit.unit) : TileIndex [B]).1.val = m := by
        show i + (m - i) = m
        omega
      show Y ((⟨m - i, hjlt⟩, PUnit.unit) : TileIndex [B]).1
          * (Real.log (Y ((⟨m - i, hjlt⟩, PUnit.unit) : TileIndex [B]).1)
            - X ((⟨m - i, hjlt⟩, PUnit.unit) : TileIndex [B]).1)
        = klCell y_ptr gt_ptr y_stride gt_stride s₀ m
      rw [hX, hY]
      simp only [hidx, if_pos hm, klCell]
    · have hnone : ∀ k ∈ TileShape.allIndices [B],
          (fun kk : TileIndex [B] => decide (i + kk.1.val < n_cols)) k = Bool.true →
          (fun kk : TileIndex [B] => s₀.pids 0 * loss_stride + (i + kk.1.val)) k
            ≠ s₀.pids 0 * loss_stride + m := by
        intro k _ _ heq
        have heq' : s₀.pids 0 * loss_stride + (i + k.1.val)
            = s₀.pids 0 * loss_stride + m := heq
        have hkb : k.1.val < B := k.1.isLt
        exact hhit ⟨by omega, by omega⟩
      rw [foldl_store_preserve (region := loss_ptr)
        (fun k : TileIndex [B] => s₀.pids 0 * loss_stride + (i + k.1.val))
        (fun k : TileIndex [B] => Y k.1 * (Real.log (Y k.1) - X k.1))
        (fun k : TileIndex [B] => decide (i + k.1.val < n_cols))
        (s₀.pids 0 * loss_stride + m) (TileShape.allIndices [B]) u5 hnone,
        hrm5, hread m hm]
      by_cases hlow : m < i
      · rw [if_pos hlow, if_pos (show m < i + B by omega)]
      · rw [if_neg hlow, if_neg (show ¬ m < i + B by omega)]
  · -- the per-cell frame
    intro r oo hcond
    rw [kl_foldl_writeMem_preserve_cell (region := loss_ptr)
      (fun k : TileIndex [B] => s₀.pids 0 * loss_stride + (i + k.1.val))
      (fun k : TileIndex [B] => Y k.1 * (Real.log (Y k.1) - X k.1))
      (fun k : TileIndex [B] => decide (i + k.1.val < n_cols))
      r oo (TileShape.allIndices [B]) u5 ?_, hmem5]
    · exact hframe r oo hcond
    · rintro k _ hmk ⟨hr, hoo⟩
      rcases hcond with hne | hno
      · exact hne hr
      · exact hno (i + (k.1 : Fin B).val) (by simpa using hmk) hoo


/-! ### Cast-free collapses and the covered fragment -/

/-- A `.real → .real` float cast is exact under every `R` (the model's
defining `round_real`). The kernel has no `castFloat` at all, so this only
appears as the residual normal form of the cast-free collapse below. -/
private theorem klRcast_real_real (R : RoundingModel) :
    R.cast .real .real = FloatDType.cast .real .real := by
  funext v
  simp [RoundingModel.cast, FloatDType.cast]

/-- The prologue is cast-free: it steps identically under `stepStmtsR R`. -/
private theorem klPrefix_castFree (R : RoundingModel)
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride B : Nat) (t : BlockState) :
    stepStmtsR R (klPrefixStmts y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride B) t
      = stepStmts (klPrefixStmts y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride B) t := by
  simp only [klPrefixStmts, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The loop body is cast-free **including its in-loop masked `.real` store**:
`stepStmtR` delegates a `.real`-typed store to the exact `writeMemTyped`, so
the whole storing loop steps identically under `stepStmtsR R`. -/
private theorem klLossBody_castFree (R : RoundingModel) (n_cols B : Nat) (t : BlockState) :
    stepStmtsR R (klLossBody n_cols B) t = stepStmts (klLossBody n_cols B) t := by
  simp only [klLossBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, klRcast_real_real, BlockState.writeMemTypedR]
  rfl

/-- The forward `reduction=0` surface sits inside the flat-memory bridge's
covered fragment (`FlattenOk`; the `forRange` clause recurses into the
cast-free body, and the pointer-register loads/stores are `ptrBase`/`ptrAdd`,
both covered). -/
theorem kldiv_forward_default_none_flattenOk (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) :
    ((kldiv_forward_default_none y_ptr gt_ptr loss_ptr y_stride gt_stride
      loss_stride n_cols BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [kl_body_decomp]
  simp [klPrefixStmts, klLossBody, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### The rounded Hoare triple -/

set_option maxHeartbeats 8000000 in
/-- **Termination, per-cell values and the per-cell frame under `execR R`,
from an arbitrary launch state.** The prologue and the body are cast-free, so
`execR R` collapses onto the exact stepper and the `klInv` `forRange` argument
runs verbatim; the loop's exit counter is `≥ n_cols`, so *every* column
`m < n_cols` has been emitted. -/
private theorem kl_runR (R : RoundingModel) (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols B : Nat) (hB : 0 < B)
    (hly : loss_ptr ≠ y_ptr) (hlg : loss_ptr ≠ gt_ptr) (s₀ : BlockState) :
    ∃ sfin,
      execR R (kldiv_forward_default_none y_ptr gt_ptr loss_ptr y_stride gt_stride
        loss_stride n_cols B).toAlgKernel s₀ = some sfin
      ∧ (∀ m, m < n_cols → sfin.readMem loss_ptr (s₀.pids 0 * loss_stride + m)
          = klCell y_ptr gt_ptr y_stride gt_stride s₀ m)
      ∧ (∀ r oo, (r ≠ loss_ptr ∨ ∀ m, m < n_cols → oo ≠ s₀.pids 0 * loss_stride + m) →
          sfin.mem r oo = s₀.mem r oo) := by
  obtain ⟨sp, hsp, hP0⟩ :=
    kl_preLoop y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride n_cols B s₀
  obtain ⟨final, sfin, hloop, hfinal, hPf⟩ :=
    forRange_inv (idx := "i") (start := 0) (stop := n_cols) (step := B)
      (body := klLossBody n_cols B)
      (P := klInv y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride n_cols B s₀)
      (Nat.pos_iff_ne_zero.mp hB) hP0
      (fun c stt hc hQ =>
        klInv_step y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride n_cols B
          hly hlg s₀ c stt hc hQ)
  obtain ⟨-, -, -, -, -, hread, hframe⟩ := hPf
  have hLoopR : stepStmtR R (Stmt.forRange "i" 0 n_cols B (klLossBody n_cols B)) sp
      = some sfin := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (klLossBody_castFree R n_cols B) "i",
      ← stepForRangeAux.forRange_unfold]
    exact hloop
  refine ⟨sfin, ?_, ?_, hframe⟩
  · show execR R (kldiv_forward_default_none y_ptr gt_ptr loss_ptr y_stride gt_stride
      loss_stride n_cols B).toAlgKernel s₀ = some sfin
    unfold execR
    rw [kl_body_decomp]
    rw [stepStmtsR_append R (klPrefixStmts y_ptr gt_ptr loss_ptr y_stride gt_stride
        loss_stride B) _ s₀,
      klPrefix_castFree R y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride B s₀,
      hsp, Option.bind_some, stepStmtsR_cons_some hLoopR, stepStmtsR_nil]
  · intro m hm
    rw [hread m hm, if_pos (lt_of_lt_of_le hm hfinal)]


/-! ### The `TraceSafeR` walk -/

/-- `evalOpR = evalOp` on the body's cast-free index ops (`R` never
enters). -/
private theorem kl_offsetsR_eq (R : RoundingModel) (B : Nat) (s : BlockState) :
    evalOpR R (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "i")
        (Op.ref .nat [B] "base_offsets")) s
      = evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "i")
        (Op.ref .nat [B] "base_offsets")) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem kl_maskR_eq (R : RoundingModel) (n_cols B : Nat) (s : BlockState) :
    evalOpR R (Op.lt .nat Broadcast.scalarR (Op.ref .nat [B] "offsets")
        (Op.constNat n_cols)) s
      = evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [B] "offsets")
        (Op.constNat n_cols)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem kl_ptraddR_eq (R : RoundingModel) (ptrReg : RegName) (B : Nat)
    (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] ptrReg)
        (Op.ref .nat [B] "offsets")) s
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] ptrReg)
        (Op.ref .nat [B] "offsets")) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- Per-lane value of a load/store address `ptr + offsets`. -/
private theorem kl_ptradd_eval (ptrReg : RegName) (r : RegionName) (base B i : Nat)
    (s : BlockState)
    (hptr : s.regs .ptr [] ptrReg = some (Tile.scalar ((r : RegionName), base)))
    (hoffsets : s.regs .nat [B] "offsets" = some (Tile.vec (fun j : Fin B => i + j.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] ptrReg)
        (Op.ref .nat [B] "offsets")) s
      = some (⟨fun j : TileIndex [B] => ((r : RegionName), base + (i + j.1.val))⟩
          : Tile .ptr [B]) := by
  rw [evalOp_ptrAdd, evalOp_ref, hptr, evalOp_ref, hoffsets]
  apply congrArg some
  apply Tile.ext
  intro j
  simp only [Tile.ptrAdd, Tile.scalar, Tile.vec, Broadcast.leftIndex_scalarL,
    Broadcast.rightIndex_scalarL]

set_option maxHeartbeats 8000000 in
/-- **Per-iteration `TraceSafeListR` for the loop body.** The index/mask/
arithmetic assigns are register-only; the two masked loads' and the masked
store's **active** lanes are exactly the skin's `mask1` / `mask2` /
`writeMask` windows at step `i / BLOCK_SIZE`, in bounds by the corresponding
window bounds (instantiated at the raw counter `i`). -/
private theorem kl_lossBodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols B : Nat)
    (s₀ st : BlockState) (i : Nat)
    (hyp : st.regs .ptr [] "y_ptr"
      = some (Tile.scalar ((y_ptr : RegionName), s₀.pids 0 * y_stride)))
    (hgp : st.regs .ptr [] "gt_ptr"
      = some (Tile.scalar ((gt_ptr : RegionName), s₀.pids 0 * gt_stride)))
    (hlp : st.regs .ptr [] "loss_ptr"
      = some (Tile.scalar ((loss_ptr : RegionName), s₀.pids 0 * loss_stride)))
    (hbase : st.regs .nat [B] "base_offsets" = some (Tile.vec (fun j : Fin B => j.val)))
    (hby : ∀ j : Fin B, i + j.val < n_cols →
      s₀.pids 0 * y_stride + (i + j.val) < bounds y_ptr)
    (hbg : ∀ j : Fin B, i + j.val < n_cols →
      s₀.pids 0 * gt_stride + (i + j.val) < bounds gt_ptr)
    (hbo : ∀ j : Fin B, i + j.val < n_cols →
      s₀.pids 0 * loss_stride + (i + j.val) < bounds loss_ptr) :
    Stmt.TraceSafeListR R bounds (klLossBody n_cols B)
      (st.setReg "i" .nat [] (Tile.scalar i)) := by
  unfold klLossBody
  -- offsets
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t1 ht1 => ?_)
  rw [stepStmtR_assign_eq_some
    ((kl_offsetsR_eq R B _).trans (kl_offsets_eval B i st hbase))] at ht1
  obtain rfl := Option.some.inj ht1
  set q1 := (st.setReg "i" .nat [] (Tile.scalar i)).setReg "offsets" .nat [B]
    (Tile.vec (fun j : Fin B => i + j.val)) with hq1
  have ho1 : q1.regs .nat [B] "offsets" = some (Tile.vec (fun j : Fin B => i + j.val)) := by
    rw [hq1]; exact BlockState.setReg_same _ _ _ _ _
  -- mask
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t2 ht2 => ?_)
  rw [stepStmtR_assign_eq_some
    ((kl_maskR_eq R n_cols B _).trans (kl_mask_eval n_cols B i q1 ho1))] at ht2
  obtain rfl := Option.some.inj ht2
  set q2 := q1.setReg "mask" .bool [B]
    (Tile.vec (fun j : Fin B => decide (i + j.val < n_cols))) with hq2
  have ho2 : q2.regs .nat [B] "offsets" = some (Tile.vec (fun j : Fin B => i + j.val)) := by
    rw [hq2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("offsets":RegName) ≠ "mask" by decide)]
    exact ho1
  have hm2 : q2.regs .bool [B] "mask"
      = some (Tile.vec (fun j : Fin B => decide (i + j.val < n_cols))) := by
    rw [hq2]; exact BlockState.setReg_same _ _ _ _ _
  have hyp2 : q2.regs .ptr [] "y_ptr"
      = some (Tile.scalar ((y_ptr : RegionName), s₀.pids 0 * y_stride)) := by
    rw [hq2, hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("y_ptr":RegName) ≠ "mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("y_ptr":RegName) ≠ "offsets" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("y_ptr":RegName) ≠ "i" by decide)]
    exact hyp
  have hgp2 : q2.regs .ptr [] "gt_ptr"
      = some (Tile.scalar ((gt_ptr : RegionName), s₀.pids 0 * gt_stride)) := by
    rw [hq2, hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("gt_ptr":RegName) ≠ "mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("gt_ptr":RegName) ≠ "offsets" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("gt_ptr":RegName) ≠ "i" by decide)]
    exact hgp
  have hlp2 : q2.regs .ptr [] "loss_ptr"
      = some (Tile.scalar ((loss_ptr : RegionName), s₀.pids 0 * loss_stride)) := by
    rw [hq2, hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("loss_ptr":RegName) ≠ "mask" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("loss_ptr":RegName) ≠ "offsets" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("loss_ptr":RegName) ≠ "i" by decide)]
    exact hlp
  -- the masked `y` load
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t3 ht3 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
      and_true, true_and, and_self]
    intro ptrs hptrs idx hactive
    rw [kl_ptraddR_eq, kl_ptradd_eval "y_ptr" y_ptr (s₀.pids 0 * y_stride) B i q2
      hyp2 ho2] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hmk, hmi⟩ := hactive
    rw [evalOpR_ref, hm2] at hmk
    obtain rfl := Option.some.inj hmk
    have hlt : i + idx.1.val < n_cols := by simpa [Tile.vec] using hmi
    simpa [Region.cast_id] using hby idx.1 hlt
  · obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv ht3
    set q3 := q2.setReg "y" .real [B] v3 with hq3
    have ho3 : q3.regs .nat [B] "offsets"
        = some (Tile.vec (fun j : Fin B => i + j.val)) := by
      rw [hq3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("offsets":RegName) ≠ "y" by decide)]
      exact ho2
    have hm3 : q3.regs .bool [B] "mask"
        = some (Tile.vec (fun j : Fin B => decide (i + j.val < n_cols))) := by
      rw [hq3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("mask":RegName) ≠ "y" by decide)]
      exact hm2
    have hgp3 : q3.regs .ptr [] "gt_ptr"
        = some (Tile.scalar ((gt_ptr : RegionName), s₀.pids 0 * gt_stride)) := by
      rw [hq3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("gt_ptr":RegName) ≠ "y" by decide)]
      exact hgp2
    have hlp3 : q3.regs .ptr [] "loss_ptr"
        = some (Tile.scalar ((loss_ptr : RegionName), s₀.pids 0 * loss_stride)) := by
      rw [hq3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("loss_ptr":RegName) ≠ "y" by decide)]
      exact hlp2
    -- the masked `y_true` load
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun t4 ht4 => ?_)
    · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
        and_true, true_and, and_self]
      intro ptrs hptrs idx hactive
      rw [kl_ptraddR_eq, kl_ptradd_eval "gt_ptr" gt_ptr (s₀.pids 0 * gt_stride) B i q3
        hgp3 ho3] at hptrs
      obtain rfl := Option.some.inj hptrs
      obtain ⟨masks, hmk, hmi⟩ := hactive
      rw [evalOpR_ref, hm3] at hmk
      obtain rfl := Option.some.inj hmk
      have hlt : i + idx.1.val < n_cols := by simpa [Tile.vec] using hmi
      simpa [Region.cast_id] using hbg idx.1 hlt
    · obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv ht4
      set q4 := q3.setReg "y_true" .real [B] v4 with hq4
      -- loss (register-only)
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t5 ht5 => ?_)
      obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv ht5
      set q5 := q4.setReg "loss" .real [B] v5 with hq5
      have ho5 : q5.regs .nat [B] "offsets"
          = some (Tile.vec (fun j : Fin B => i + j.val)) := by
        rw [hq5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("offsets":RegName) ≠ "loss" by decide), hq4,
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("offsets":RegName) ≠ "y_true" by decide)]
        exact ho3
      have hm5 : q5.regs .bool [B] "mask"
          = some (Tile.vec (fun j : Fin B => decide (i + j.val < n_cols))) := by
        rw [hq5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("mask":RegName) ≠ "loss" by decide), hq4,
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("mask":RegName) ≠ "y_true" by decide)]
        exact hm3
      have hlp5 : q5.regs .ptr [] "loss_ptr"
          = some (Tile.scalar ((loss_ptr : RegionName), s₀.pids 0 * loss_stride)) := by
        rw [hq5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("loss_ptr":RegName) ≠ "loss" by decide), hq4,
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("loss_ptr":RegName) ≠ "y_true" by decide)]
        exact hlp3
      -- the masked store
      refine Stmt.TraceSafeListR.cons_intro ?_
        (fun _ _ => Stmt.TraceSafeListR.nil_intro)
      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
        MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
        memAccessActiveAddressSafeR, and_true, true_and, and_self]
      intro ptrs hptrs idx hactive
      rw [kl_ptraddR_eq, kl_ptradd_eval "loss_ptr" loss_ptr (s₀.pids 0 * loss_stride) B i q5
        hlp5 ho5] at hptrs
      obtain rfl := Option.some.inj hptrs
      obtain ⟨masks, hmk, hmi⟩ := hactive
      rw [evalOpR_ref, hm5] at hmk
      obtain rfl := Option.some.inj hmk
      have hlt : i + idx.1.val < n_cols := by simpa [Tile.vec] using hmi
      simpa [Region.cast_id] using hbo idx.1 hlt


set_option maxHeartbeats 8000000 in
/-- **The `TraceSafeR` walk for the whole kernel** — the register-only
prologue, then `Stmt.forRangeTraceSafeR_inv` over `klInv` with the counter
advancing by the loop's stride `BLOCK_SIZE`. The three bound groups are the
skin's `read1` / `read2` / `write` windows; `hBS` converts the raw counter into
the step index `i / BLOCK_SIZE < ⌈n_cols/BLOCK_SIZE⌉` the windows are phrased
over. -/
private theorem kldiv_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols B : Nat) (hB : 0 < B)
    (hly : loss_ptr ≠ y_ptr) (hlg : loss_ptr ≠ gt_ptr) (s : BlockState)
    (hby : ∀ (t : Fin (klNumSteps n_cols B)) (j : Fin B), t.val * B + j.val < n_cols →
      s.pids 0 * y_stride + (t.val * B + j.val) < bounds y_ptr)
    (hbg : ∀ (t : Fin (klNumSteps n_cols B)) (j : Fin B), t.val * B + j.val < n_cols →
      s.pids 0 * gt_stride + (t.val * B + j.val) < bounds gt_ptr)
    (hbo : ∀ (t : Fin (klNumSteps n_cols B)) (j : Fin B), t.val * B + j.val < n_cols →
      s.pids 0 * loss_stride + (t.val * B + j.val) < bounds loss_ptr) :
    ((kldiv_forward_default_none y_ptr gt_ptr loss_ptr y_stride gt_stride
      loss_stride n_cols B).toAlgKernel).TraceSafeR R bounds s := by
  have hstepT : ∀ i, B ∣ i → ∀ j : Fin B, i + j.val < n_cols →
      ∃ t : Fin (klNumSteps n_cols B), t.val * B + j.val = i + j.val := by
    intro i hiB j hij
    have hiN : i < n_cols := Nat.lt_of_le_of_lt (Nat.le_add_right _ _) hij
    refine ⟨⟨i / B, klStep_lt_numSteps n_cols B i hB hiN⟩, ?_⟩
    simp [Nat.div_mul_cancel hiB]
  have hby' : ∀ i, B ∣ i → ∀ j : Fin B, i + j.val < n_cols →
      s.pids 0 * y_stride + (i + j.val) < bounds y_ptr := by
    intro i hiB j hij
    obtain ⟨t, ht⟩ := hstepT i hiB j hij
    have h := hby t j (by rw [ht]; exact hij)
    rwa [ht] at h
  have hbg' : ∀ i, B ∣ i → ∀ j : Fin B, i + j.val < n_cols →
      s.pids 0 * gt_stride + (i + j.val) < bounds gt_ptr := by
    intro i hiB j hij
    obtain ⟨t, ht⟩ := hstepT i hiB j hij
    have h := hbg t j (by rw [ht]; exact hij)
    rwa [ht] at h
  have hbo' : ∀ i, B ∣ i → ∀ j : Fin B, i + j.val < n_cols →
      s.pids 0 * loss_stride + (i + j.val) < bounds loss_ptr := by
    intro i hiB j hij
    obtain ⟨t, ht⟩ := hstepT i hiB j hij
    have h := hbo t j (by rw [ht]; exact hij)
    rwa [ht] at h
  unfold Kernel.TraceSafeR
  rw [kl_body_decomp]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro stmt hst s'
    simp only [klPrefixStmts, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s1 hs1
    obtain ⟨sp, hsp, hP0⟩ :=
      kl_preLoop y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride n_cols B s
    rw [klPrefix_castFree R y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride B s,
      hsp] at hs1
    obtain rfl := Option.some.inj hs1
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR]
    refine Stmt.forRangeTraceSafeR_inv R bounds "i" n_cols B (klLossBody n_cols B)
      (klInv y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride n_cols B s) ?_ 0 sp hP0
    intro c stt hc hP
    have hPd := hP
    obtain ⟨hdvd, hyp, hgp, hlp, hbase, -, -⟩ := hPd
    refine ⟨kl_lossBodySafeR R bounds y_ptr gt_ptr loss_ptr y_stride gt_stride
      loss_stride n_cols B s stt c hyp hgp hlp hbase
      (fun j hj => hby' c hdvd j hj) (fun j hj => hbg' c hdvd j hj)
      (fun j hj => hbo' c hdvd j hj), ?_⟩
    obtain ⟨st', hstep, hP'⟩ :=
      klInv_step y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride n_cols B
        hly hlg s c stt hc hP
    exact ⟨st', by rw [klLossBody_castFree]; exact hstep, hP'⟩

/-! ### IO signature and stream-level spec -/

/-- **Streaming IO signature** of `kldiv_forward_default_none` on the
two-stream per-step emit skin (S3: in-loop store). Step `t` (at `i =
t·BLOCK_SIZE`) reads the `BLOCK_SIZE`-lane `y` tile (`read1`) and the `gt`
tile (`read2`), and stores the `BLOCK_SIZE`-lane loss window (`write`) at the
**`.real`** grid (`outDType` default — the kernel's `tl.store(loss_ptr +
offsets, loss, mask=mask)` lowers to `Stmt.store TileDType.real` with no cast
on the value, so the per-step stores carry no quantization event). The windows
transcribe the kernel's pointer arithmetic verbatim (`offsets = i +
tl.arange(0, BLOCK_SIZE)` on top of `ptr += pid * stride`):

* `read1` step `t`, lane `j`: `pid₀·y_stride + (t·BLOCK_SIZE + j)`;
* `read2` step `t`, lane `j`: `pid₀·gt_stride + (t·BLOCK_SIZE + j)`;
* `write` step `t`, lane `j`: `pid₀·loss_stride + (t·BLOCK_SIZE + j)`.

All three masks are the kernel's single `mask`: `t·BLOCK_SIZE + j < n_cols`.
The second grid axis `pid₁` is unused (the launch grid is 1-D `(B,)`). -/
def kldivForwardDefaultNoneIO (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) :
    StreamEmitMasked2DKernelIO₂ where
  kernel := kldiv_forward_default_none y_ptr gt_ptr loss_ptr y_stride gt_stride
    loss_stride n_cols BLOCK_SIZE
  inp1 := y_ptr
  inp2 := gt_ptr
  out := loss_ptr
  T := klNumSteps n_cols BLOCK_SIZE
  B1 := BLOCK_SIZE
  B2 := BLOCK_SIZE
  C := BLOCK_SIZE
  read1 := fun p₀ _ t j => p₀ * y_stride + (t.val * BLOCK_SIZE + j.val)
  read2 := fun p₀ _ t j => p₀ * gt_stride + (t.val * BLOCK_SIZE + j.val)
  write := fun p₀ _ t j => p₀ * loss_stride + (t.val * BLOCK_SIZE + j.val)
  mask1 := fun _ _ t j => t.val * BLOCK_SIZE + j.val < n_cols
  mask2 := fun _ _ t j => t.val * BLOCK_SIZE + j.val < n_cols
  writeMask := fun _ _ t j => t.val * BLOCK_SIZE + j.val < n_cols

/-- The stream-level forward KL spec (the emit genre's *elementwise* shape:
output window `(t, j)` depends only on the step-`t` input tiles):
`ys t j · (log (ys t j) − xs t j)` — exact real arithmetic, the
stream-index spelling of the file's `forwardDefaultSpec`. -/
noncomputable def klStreamSpec (n_cols BLOCK_SIZE : Nat)
    (xs ys : Fin (klNumSteps n_cols BLOCK_SIZE) → Fin BLOCK_SIZE → ℝ)
    (t : Fin (klNumSteps n_cols BLOCK_SIZE)) (j : Fin BLOCK_SIZE) : ℝ :=
  ys t j * (Real.log (ys t j) - xs t j)

/-- Per-lane spec bridge: under the stream pin, the stream-level spec at a
masked window `(t, j)` **is** the memory-level `klCell` at global column
`t·BLOCK_SIZE + j`. -/
private theorem klStreamSpec_eq_klCell (y_ptr gt_ptr : RegionName)
    (y_stride gt_stride n_cols B : Nat) (s₀ : BlockState)
    (xs ys : Fin (klNumSteps n_cols B) → Fin B → ℝ)
    (hx : ∀ (t : Fin (klNumSteps n_cols B)) (j : Fin B), t.val * B + j.val < n_cols →
      s₀.readMem y_ptr (s₀.pids 0 * y_stride + (t.val * B + j.val)) = xs t j)
    (hy : ∀ (t : Fin (klNumSteps n_cols B)) (j : Fin B), t.val * B + j.val < n_cols →
      s₀.readMem gt_ptr (s₀.pids 0 * gt_stride + (t.val * B + j.val)) = ys t j)
    (t : Fin (klNumSteps n_cols B)) (j : Fin B) (hj : t.val * B + j.val < n_cols) :
    klStreamSpec n_cols B xs ys t j
      = klCell y_ptr gt_ptr y_stride gt_stride s₀ (t.val * B + j.val) := by
  unfold klStreamSpec klCell
  rw [← hx t j hj, ← hy t j hj]

set_option maxHeartbeats 4000000 in
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `kldiv_forward_default_none` surface
(the `log_target = False`, `reduction = 0` (None) forward branch) implements,
on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ elementwise
forward KL divergence** over the streamed tiles: emitted window `(t, j)` holds
`y_true[t,j] · (log y_true[t,j] − y[t,j])`, with `y` the log-space prediction
stream and `y_true` the ground-truth stream. The spec `f` is exact real
arithmetic — this port has **no `eps` clamp** (unlike the `kldiv_ops` port).

Rounding story: the kernel has **zero rounding events**. Both `tl.load`s and
the in-loop `tl.store` are at `.real` (the lowered store is
`Stmt.store TileDType.real … (MaskOpt.mask …)` — no cast on the stored value),
and the kernel contains no `castFloat` at all; the only casts are the
`int`-flavoured `pid = tl.program_id(0).to(tl.int64)` pointer arithmetic, which
is exact. So the skin's boundary quantization degenerates: at the declared grid
`outDType := .real` the readback contract's `R.round .real` is the identity by
the model's defining `round_real`, and the `.real` in-loop stores are exact
under `execR R`. The ∀-`R` face therefore holds via the `RoundingModel` `.real`
identity fields, **not** as a `.triv` special case.

Layer map: the prologue and the loop body are cast-free
(`klPrefix_castFree` / `klLossBody_castFree`), so under `execR R` they collapse
verbatim onto the exact stepper; the loop is driven by the `forRange`
invariant `klInv` (`kl_preLoop` + `klInv_step` + `kl_runR`), the safety walk by
`Stmt.forRangeTraceSafeR_inv` over the same invariant (`kl_lossBodySafeR` +
`kldiv_traceSafeR`), and the stream indices are matched to the memory-level
closed form by `klStreamSpec_eq_klCell`.

**Scope: this face is dimension-general — the full multi-chunk stream.** `T`
is `⌈n_cols / BLOCK_SIZE⌉`, so the loop is proved for an arbitrary number of
iterations; there is **no** `n_cols ≤ BLOCK_SIZE` regime hypothesis and no
`0 < n_cols` hypothesis. (The retained exact surfaces below/above are the
narrower single-chunk statements; see "Relation to the exact surfaces".)

Every hypothesis is truth-forced:

* `hBS : 0 < BLOCK_SIZE` — the loop steps by `BLOCK_SIZE`
  (`range(0, n_cols, BLOCK_SIZE)`); at `BLOCK_SIZE = 0` with `n_cols > 0` the
  `forRange` fold does not advance, `execR` cannot terminate the way the
  invariant requires, and the step index `i / BLOCK_SIZE` is meaningless. It
  holds for every real launch (the host picks
  `BLOCK_SIZE = min(16384, next_power_of_2(S))`).
* `hly : loss_ptr ≠ y_ptr`, `hlg : loss_ptr ≠ gt_ptr` — the store sits
  **inside** the loop, so iteration `t + 1` loads *after* iteration `t` has
  already stored. If the loss buffer aliased either input buffer, later chunks
  would re-read overwritten values and the closed form would be false. (In the
  single-chunk regime these are vacuous; they are exactly the price of the
  general multi-chunk face.) The host allocates `output_tensor` fresh, so both
  hold for every real launch.

Relation to the exact surfaces: the file's exact headline
`kldiv_forward_default_none_compute_correct`
(`Realizes_without_Rounding`, scoped to the Python-tested single-chunk regime
`0 < n_cols ≤ BLOCK_SIZE`, with the offset-injectivity side condition) is
retained unchanged; this `⊨[R]` face restates the same KL content on the
streaming emit skin, for every `R` at once and for an arbitrary number of
chunks (at the `.real` grid the two faces carry the same exact cell on the
single-chunk overlap). Both faces are kept per the rounding-as-default
doctrine. The `log_target = True` branch and the `reduction ≠ 0` reduction
modes keep their existing (exact, partial) treatment — the reduction-mode
surface `kldiv_forward_surface` is still only proved to lower. -/
specification kldiv_forward_default_none_io_correctness (R : RoundingModel)
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (hBS : 0 < BLOCK_SIZE)
    (hly : loss_ptr ≠ y_ptr) (hlg : loss_ptr ≠ gt_ptr) :
    kldivForwardDefaultNoneIO y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride
        n_cols BLOCK_SIZE ⊨[R]
      fun _ _ xs ys t j => klStreamSpec n_cols BLOCK_SIZE xs ys t j := by
  refine StreamEmitMasked2DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact kldiv_forward_default_none_flattenOk y_ptr gt_ptr loss_ptr y_stride
      gt_stride loss_stride n_cols BLOCK_SIZE
  · intro bounds s xs ys _hx _hy hbr1 hbr2 hbw
    simp only [kldivForwardDefaultNoneIO] at hbr1 hbr2 hbw ⊢
    exact kldiv_traceSafeR R bounds y_ptr gt_ptr loss_ptr y_stride gt_stride
      loss_stride n_cols BLOCK_SIZE hBS hly hlg s hbr1 hbr2 hbw
  · intro s₀ xs ys _hundef hx hy
    simp only [kldivForwardDefaultNoneIO] at hx hy ⊢
    obtain ⟨sfin, hexec, hval, hframe⟩ :=
      kl_runR R y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride n_cols
        BLOCK_SIZE hBS hly hlg s₀
    refine ⟨sfin, hexec, ?_, ?_⟩
    · intro t j hj
      have hk : t.val * BLOCK_SIZE + j.val < n_cols := hj
      rw [BlockState.readMemAs_real, hval _ hk,
        ← klStreamSpec_eq_klCell y_ptr gt_ptr y_stride gt_stride n_cols BLOCK_SIZE s₀
          xs ys hx hy t j hj]
      simp [FloatDType.ofReal]
    · intro r oo hcond
      refine hframe r oo ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · refine Or.inr fun m hm => ?_
        have hdm : m / BLOCK_SIZE * BLOCK_SIZE + m % BLOCK_SIZE = m := by
          rw [Nat.mul_comm]
          exact Nat.div_add_mod m BLOCK_SIZE
        have h := hno ⟨m / BLOCK_SIZE, klStep_lt_numSteps n_cols BLOCK_SIZE m hBS hm⟩
          ⟨m % BLOCK_SIZE, Nat.mod_lt _ hBS⟩ (by simp only [hdm]; exact hm)
        simpa [hdm] using h

end IOFace

end VeriTile.Bench.TritonBenchG.KldivTriton
