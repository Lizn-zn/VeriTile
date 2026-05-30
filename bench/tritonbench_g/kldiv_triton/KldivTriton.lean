import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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
theorem kldiv_backward_default_compute_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s input_stride i)) :
    ComputeCorrect.Realizes
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
theorem kldiv_backward_log_target_compute_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s input_stride i)) :
    ComputeCorrect.Realizes
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
theorem kldiv_forward_default_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes
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
theorem kldiv_forward_log_target_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes
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

end VeriTile.Bench.TritonBenchG.KldivTriton
