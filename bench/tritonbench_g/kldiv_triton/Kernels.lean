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



namespace TypedSource
open Lean

/-- Floating memory carriers. This is a frontend model, not a native-device support claim. -/
inductive Storage where
  | fp32 | fp64 | fp16 | bf16 | f8e4 | f8e5
  deriving DecidableEq, Repr

/-- Direct `tl.log` and `tl.exp` accept fp32/fp64 in pinned Triton 3.5.1. -/
inductive MathStorage where
  | fp32 | fp64
  deriving DecidableEq, Repr

syntax "klForward%" "[" tritonDType "," tritonDType "," tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(klForward%[$pred:tritonDType, $target:tritonDType, $output:tritonDType]) => do
    let wide := match pred, target with
      | `(tritonDType| tl.float64), _ | _, `(tritonDType| tl.float64) => Bool.true
      | _, _ => Bool.false
    let same := match output with
      | `(tritonDType| tl.float64) => wide
      | `(tritonDType| tl.float32) => !wide
      | _ => Bool.false
    let elementStore : TSyntax `tritonStmt ← if same then
      `(tritonStmt| tl.store(O + offsets, loss, mask=mask))
      else `(tritonStmt| tl.store(O + offsets, loss.to($output), mask=mask))
    let chunkStore : TSyntax `tritonStmt ← if same then
      `(tritonStmt| tl.store(O, loss))
      else `(tritonStmt| tl.store(O, loss.to($output)))
    `(fun (Y T O : RegionName) (sy st so N B : Nat) (logTarget : Bool) (reduction : Nat) =>
      ComputeKernel.mk [Y, T] [O] (triton {
        tl.region Y = $pred, T = $target, O = $output
        pid = tl.program_id(0).to(tl.int64)
        Y += pid * $(sy)
        T += pid * $(st)
        O += pid * $(so)
        base_offsets = tl.arange(0, $(B))
        for i in range(0, $((N : Int)), $(B)) {
          offsets = i + base_offsets
          mask = offsets < $((N : Int))
          y = tl.load(Y + offsets, mask=mask, other=0.0)
          y_true = tl.load(T + offsets, mask=mask, other=0.0)
          if not logTarget {
            loss = y_true * (tl.log(y_true) - y)
          } else {
            loss = tl.exp(y_true) * (y_true - y)
          }
          if $(reduction) == $(0) {
            $elementStore
          } else {
            loss = tl.sum(loss, axis=0)
            $chunkStore
            O += $(1)
          }
        }
      }).surfaceBody)

syntax "klBackward%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(klBackward%[$target:tritonDType, $output:tritonDType, $branch:ident]) => do
    let logarithmic := branch.getId == `logarithmic
    unless logarithmic || branch.getId == `defaultTarget do
      Macro.throwError "expected logarithmic or defaultTarget"
    let compute : TSyntax `tritonStmt ← if logarithmic then
      `(tritonStmt| res = -tl.exp(target))
      else `(tritonStmt| res = target * -1)
    let same := match target, output with
      | `(tritonDType| tl.float32), `(tritonDType| tl.float32)
      | `(tritonDType| tl.float64), `(tritonDType| tl.float64)
      | `(tritonDType| tl.float16), `(tritonDType| tl.float16)
      | `(tritonDType| tl.bfloat16), `(tritonDType| tl.bfloat16)
      | `(tritonDType| tl.float8e4nv), `(tritonDType| tl.float8e4nv)
      | `(tritonDType| tl.float8e5), `(tritonDType| tl.float8e5) => Bool.true
      | _, _ => Bool.false
    let store : TSyntax `tritonStmt ← if same then
      `(tritonStmt| tl.store(O + offsets, res, mask=mask))
      else `(tritonStmt| tl.store(O + offsets, res.to($output), mask=mask))
    `(fun (T O : RegionName) (st so N B : Nat) =>
      ComputeKernel.mk [T] [O] (triton {
        tl.region T = $target, O = $output
        pid = tl.program_id(0).to(tl.int64)
        O += pid * $(so)
        T += pid * $(st)
        offsets = tl.arange(0, $(B))
        mask = offsets < $((N : Int))
        for i in range(0, $((N : Int)), $(B)) {
          offsets = i + tl.arange(0, $(B))
          mask = offsets < $((N : Int))
          target = tl.load(T + offsets, mask=mask, other=0.0)
          $compute
          $store
        }
      }).surfaceBody)

abbrev Forward := RegionName → RegionName → RegionName → Nat → Nat → Nat → Nat → Nat →
  Bool → Nat → ComputeKernel
abbrev Backward := RegionName → RegionName → Nat → Nat → Nat → Nat → ComputeKernel

set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

/-- Every chunk is retained. Nonzero reduction writes one scalar per chunk and
advances the destination pointer; no whole-row reduction is substituted. -/
def forwardOriginal (pred : Storage) (target : MathStorage) (output : Storage) : Forward :=
  match pred, target, output with
  | .fp32, .fp32, .fp32 => klForward%[tl.float32, tl.float32, tl.float32]
  | .fp32, .fp32, .fp64 => klForward%[tl.float32, tl.float32, tl.float64]
  | .fp32, .fp32, .fp16 => klForward%[tl.float32, tl.float32, tl.float16]
  | .fp32, .fp32, .bf16 => klForward%[tl.float32, tl.float32, tl.bfloat16]
  | .fp32, .fp32, .f8e4 => klForward%[tl.float32, tl.float32, tl.float8e4nv]
  | .fp32, .fp32, .f8e5 => klForward%[tl.float32, tl.float32, tl.float8e5]
  | .fp32, .fp64, .fp32 => klForward%[tl.float32, tl.float64, tl.float32]
  | .fp32, .fp64, .fp64 => klForward%[tl.float32, tl.float64, tl.float64]
  | .fp32, .fp64, .fp16 => klForward%[tl.float32, tl.float64, tl.float16]
  | .fp32, .fp64, .bf16 => klForward%[tl.float32, tl.float64, tl.bfloat16]
  | .fp32, .fp64, .f8e4 => klForward%[tl.float32, tl.float64, tl.float8e4nv]
  | .fp32, .fp64, .f8e5 => klForward%[tl.float32, tl.float64, tl.float8e5]
  | .fp64, .fp32, .fp32 => klForward%[tl.float64, tl.float32, tl.float32]
  | .fp64, .fp32, .fp64 => klForward%[tl.float64, tl.float32, tl.float64]
  | .fp64, .fp32, .fp16 => klForward%[tl.float64, tl.float32, tl.float16]
  | .fp64, .fp32, .bf16 => klForward%[tl.float64, tl.float32, tl.bfloat16]
  | .fp64, .fp32, .f8e4 => klForward%[tl.float64, tl.float32, tl.float8e4nv]
  | .fp64, .fp32, .f8e5 => klForward%[tl.float64, tl.float32, tl.float8e5]
  | .fp64, .fp64, .fp32 => klForward%[tl.float64, tl.float64, tl.float32]
  | .fp64, .fp64, .fp64 => klForward%[tl.float64, tl.float64, tl.float64]
  | .fp64, .fp64, .fp16 => klForward%[tl.float64, tl.float64, tl.float16]
  | .fp64, .fp64, .bf16 => klForward%[tl.float64, tl.float64, tl.bfloat16]
  | .fp64, .fp64, .f8e4 => klForward%[tl.float64, tl.float64, tl.float8e4nv]
  | .fp64, .fp64, .f8e5 => klForward%[tl.float64, tl.float64, tl.float8e5]
  | .fp16, .fp32, .fp32 => klForward%[tl.float16, tl.float32, tl.float32]
  | .fp16, .fp32, .fp64 => klForward%[tl.float16, tl.float32, tl.float64]
  | .fp16, .fp32, .fp16 => klForward%[tl.float16, tl.float32, tl.float16]
  | .fp16, .fp32, .bf16 => klForward%[tl.float16, tl.float32, tl.bfloat16]
  | .fp16, .fp32, .f8e4 => klForward%[tl.float16, tl.float32, tl.float8e4nv]
  | .fp16, .fp32, .f8e5 => klForward%[tl.float16, tl.float32, tl.float8e5]
  | .fp16, .fp64, .fp32 => klForward%[tl.float16, tl.float64, tl.float32]
  | .fp16, .fp64, .fp64 => klForward%[tl.float16, tl.float64, tl.float64]
  | .fp16, .fp64, .fp16 => klForward%[tl.float16, tl.float64, tl.float16]
  | .fp16, .fp64, .bf16 => klForward%[tl.float16, tl.float64, tl.bfloat16]
  | .fp16, .fp64, .f8e4 => klForward%[tl.float16, tl.float64, tl.float8e4nv]
  | .fp16, .fp64, .f8e5 => klForward%[tl.float16, tl.float64, tl.float8e5]
  | .bf16, .fp32, .fp32 => klForward%[tl.bfloat16, tl.float32, tl.float32]
  | .bf16, .fp32, .fp64 => klForward%[tl.bfloat16, tl.float32, tl.float64]
  | .bf16, .fp32, .fp16 => klForward%[tl.bfloat16, tl.float32, tl.float16]
  | .bf16, .fp32, .bf16 => klForward%[tl.bfloat16, tl.float32, tl.bfloat16]
  | .bf16, .fp32, .f8e4 => klForward%[tl.bfloat16, tl.float32, tl.float8e4nv]
  | .bf16, .fp32, .f8e5 => klForward%[tl.bfloat16, tl.float32, tl.float8e5]
  | .bf16, .fp64, .fp32 => klForward%[tl.bfloat16, tl.float64, tl.float32]
  | .bf16, .fp64, .fp64 => klForward%[tl.bfloat16, tl.float64, tl.float64]
  | .bf16, .fp64, .fp16 => klForward%[tl.bfloat16, tl.float64, tl.float16]
  | .bf16, .fp64, .bf16 => klForward%[tl.bfloat16, tl.float64, tl.bfloat16]
  | .bf16, .fp64, .f8e4 => klForward%[tl.bfloat16, tl.float64, tl.float8e4nv]
  | .bf16, .fp64, .f8e5 => klForward%[tl.bfloat16, tl.float64, tl.float8e5]
  | .f8e4, .fp32, .fp32 => klForward%[tl.float8e4nv, tl.float32, tl.float32]
  | .f8e4, .fp32, .fp64 => klForward%[tl.float8e4nv, tl.float32, tl.float64]
  | .f8e4, .fp32, .fp16 => klForward%[tl.float8e4nv, tl.float32, tl.float16]
  | .f8e4, .fp32, .bf16 => klForward%[tl.float8e4nv, tl.float32, tl.bfloat16]
  | .f8e4, .fp32, .f8e4 => klForward%[tl.float8e4nv, tl.float32, tl.float8e4nv]
  | .f8e4, .fp32, .f8e5 => klForward%[tl.float8e4nv, tl.float32, tl.float8e5]
  | .f8e4, .fp64, .fp32 => klForward%[tl.float8e4nv, tl.float64, tl.float32]
  | .f8e4, .fp64, .fp64 => klForward%[tl.float8e4nv, tl.float64, tl.float64]
  | .f8e4, .fp64, .fp16 => klForward%[tl.float8e4nv, tl.float64, tl.float16]
  | .f8e4, .fp64, .bf16 => klForward%[tl.float8e4nv, tl.float64, tl.bfloat16]
  | .f8e4, .fp64, .f8e4 => klForward%[tl.float8e4nv, tl.float64, tl.float8e4nv]
  | .f8e4, .fp64, .f8e5 => klForward%[tl.float8e4nv, tl.float64, tl.float8e5]
  | .f8e5, .fp32, .fp32 => klForward%[tl.float8e5, tl.float32, tl.float32]
  | .f8e5, .fp32, .fp64 => klForward%[tl.float8e5, tl.float32, tl.float64]
  | .f8e5, .fp32, .fp16 => klForward%[tl.float8e5, tl.float32, tl.float16]
  | .f8e5, .fp32, .bf16 => klForward%[tl.float8e5, tl.float32, tl.bfloat16]
  | .f8e5, .fp32, .f8e4 => klForward%[tl.float8e5, tl.float32, tl.float8e4nv]
  | .f8e5, .fp32, .f8e5 => klForward%[tl.float8e5, tl.float32, tl.float8e5]
  | .f8e5, .fp64, .fp32 => klForward%[tl.float8e5, tl.float64, tl.float32]
  | .f8e5, .fp64, .fp64 => klForward%[tl.float8e5, tl.float64, tl.float64]
  | .f8e5, .fp64, .fp16 => klForward%[tl.float8e5, tl.float64, tl.float16]
  | .f8e5, .fp64, .bf16 => klForward%[tl.float8e5, tl.float64, tl.bfloat16]
  | .f8e5, .fp64, .f8e4 => klForward%[tl.float8e5, tl.float64, tl.float8e4nv]
  | .f8e5, .fp64, .f8e5 => klForward%[tl.float8e5, tl.float64, tl.float8e5]

/-- Default-target backward keeps the weak scalar multiplication by -1. -/
def backwardDefaultOriginal (target output : Storage) : Backward :=
  match target, output with
  | .fp32, .fp32 => klBackward%[tl.float32, tl.float32, defaultTarget]
  | .fp32, .fp64 => klBackward%[tl.float32, tl.float64, defaultTarget]
  | .fp32, .fp16 => klBackward%[tl.float32, tl.float16, defaultTarget]
  | .fp32, .bf16 => klBackward%[tl.float32, tl.bfloat16, defaultTarget]
  | .fp32, .f8e4 => klBackward%[tl.float32, tl.float8e4nv, defaultTarget]
  | .fp32, .f8e5 => klBackward%[tl.float32, tl.float8e5, defaultTarget]
  | .fp64, .fp32 => klBackward%[tl.float64, tl.float32, defaultTarget]
  | .fp64, .fp64 => klBackward%[tl.float64, tl.float64, defaultTarget]
  | .fp64, .fp16 => klBackward%[tl.float64, tl.float16, defaultTarget]
  | .fp64, .bf16 => klBackward%[tl.float64, tl.bfloat16, defaultTarget]
  | .fp64, .f8e4 => klBackward%[tl.float64, tl.float8e4nv, defaultTarget]
  | .fp64, .f8e5 => klBackward%[tl.float64, tl.float8e5, defaultTarget]
  | .fp16, .fp32 => klBackward%[tl.float16, tl.float32, defaultTarget]
  | .fp16, .fp64 => klBackward%[tl.float16, tl.float64, defaultTarget]
  | .fp16, .fp16 => klBackward%[tl.float16, tl.float16, defaultTarget]
  | .fp16, .bf16 => klBackward%[tl.float16, tl.bfloat16, defaultTarget]
  | .fp16, .f8e4 => klBackward%[tl.float16, tl.float8e4nv, defaultTarget]
  | .fp16, .f8e5 => klBackward%[tl.float16, tl.float8e5, defaultTarget]
  | .bf16, .fp32 => klBackward%[tl.bfloat16, tl.float32, defaultTarget]
  | .bf16, .fp64 => klBackward%[tl.bfloat16, tl.float64, defaultTarget]
  | .bf16, .fp16 => klBackward%[tl.bfloat16, tl.float16, defaultTarget]
  | .bf16, .bf16 => klBackward%[tl.bfloat16, tl.bfloat16, defaultTarget]
  | .bf16, .f8e4 => klBackward%[tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .bf16, .f8e5 => klBackward%[tl.bfloat16, tl.float8e5, defaultTarget]
  | .f8e4, .fp32 => klBackward%[tl.float8e4nv, tl.float32, defaultTarget]
  | .f8e4, .fp64 => klBackward%[tl.float8e4nv, tl.float64, defaultTarget]
  | .f8e4, .fp16 => klBackward%[tl.float8e4nv, tl.float16, defaultTarget]
  | .f8e4, .bf16 => klBackward%[tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4, .f8e4 => klBackward%[tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e4, .f8e5 => klBackward%[tl.float8e4nv, tl.float8e5, defaultTarget]
  | .f8e5, .fp32 => klBackward%[tl.float8e5, tl.float32, defaultTarget]
  | .f8e5, .fp64 => klBackward%[tl.float8e5, tl.float64, defaultTarget]
  | .f8e5, .fp16 => klBackward%[tl.float8e5, tl.float16, defaultTarget]
  | .f8e5, .bf16 => klBackward%[tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e5, .f8e4 => klBackward%[tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5, .f8e5 => klBackward%[tl.float8e5, tl.float8e5, defaultTarget]

/-- Log-target backward retains the raw exp input dtype and negation order. -/
def backwardLogOriginal (target : MathStorage) (output : Storage) : Backward :=
  match target, output with
  | .fp32, .fp32 => klBackward%[tl.float32, tl.float32, logarithmic]
  | .fp32, .fp64 => klBackward%[tl.float32, tl.float64, logarithmic]
  | .fp32, .fp16 => klBackward%[tl.float32, tl.float16, logarithmic]
  | .fp32, .bf16 => klBackward%[tl.float32, tl.bfloat16, logarithmic]
  | .fp32, .f8e4 => klBackward%[tl.float32, tl.float8e4nv, logarithmic]
  | .fp32, .f8e5 => klBackward%[tl.float32, tl.float8e5, logarithmic]
  | .fp64, .fp32 => klBackward%[tl.float64, tl.float32, logarithmic]
  | .fp64, .fp64 => klBackward%[tl.float64, tl.float64, logarithmic]
  | .fp64, .fp16 => klBackward%[tl.float64, tl.float16, logarithmic]
  | .fp64, .bf16 => klBackward%[tl.float64, tl.bfloat16, logarithmic]
  | .fp64, .f8e4 => klBackward%[tl.float64, tl.float8e4nv, logarithmic]
  | .fp64, .f8e5 => klBackward%[tl.float64, tl.float8e5, logarithmic]


/-- Candidate changes only the order of the exact signed row-address product. -/
def commuteOutput (source : ComputeKernel) (position : Nat) (O : RegionName)
    (stride : Nat) : ComputeKernel :=
  ComputeKernel.mk source.inputs source.outputs (source.surfaceBody.take position ++
    [.assign .ptr [] "O" (.alg (.ptrAdd .nil (.ptrBase O)
      (.castIntToNat (.mul .int .nil (.castNatToInt (.constNat stride))
        (.ref .int [] "pid")))))] ++ source.surfaceBody.drop (position + 1))

def forward (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (logTarget : Bool) (reduction : Nat) :
    ComputeKernel :=
  let source := forwardOriginal pred target output Y T O sy st so N B logTarget reduction
  if v then commuteOutput source 3 O so else source

def backwardDefault (target output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) : ComputeKernel :=
  let source := backwardDefaultOriginal target output T O st so N B
  if v then commuteOutput source 1 O so else source

def backwardLog (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) : ComputeKernel :=
  let source := backwardLogOriginal target output T O st so N B
  if v then commuteOutput source 1 O so else source

end TypedSource
end VeriTile.Bench.TritonBenchG.KldivTriton
