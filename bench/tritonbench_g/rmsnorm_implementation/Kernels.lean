import VeriTile.Triton

/-!
# `rmsnorm_implementation` — strict per-kernel correctness

`rmsnorm_triton` is a fused RMSNorm forward over a 3D tensor: each program
`(pid_batch, pid_m)` normalizes one row by its root-mean-square, scales by the
per-column RMS weights, and writes `out = (x / sqrt(mean(x²) + eps)) * w`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body, for one program. The host launch (`rmsnorm_triton[(batch, M,)](...)`, the
2D grid over `(batch, M)`, the host-side `BLOCK_N_SIZE = 4096` choice,
scheduling, and how the runtime composes per-program writes into one buffer) is
the *trusted boundary*, not a proof obligation here. Because the two program ids
`pid_batch = tl.program_id(0)` and `pid_m = tl.program_id(1)` are universally
quantified (via `s.pids 0` / `s.pids 1`), the per-program statement covers every
program of the 2D grid.

## Proof architecture

```
rmsnorm_implementation_compute_fullN_correct  ← TOP THEOREM (general N, multi-block)
  └─ rmsnorm_implementation_fullN_correct      ← algorithm-layer readback per column
       ├─ rmsnorm_implementation_staged_fullN_correct_from_preloop
       ├─ rmsVarForRange_context_of_preloop   ← var-loop forRange invariant
       └─ rmsOutForRange_fullN_of_init_stride_pos  ← output-loop forRange invariant
            └─ rmsOutLoopBody_step_output_invariant
rmsnorm_implementation_compute_correct        ← one-block slice (0 < N ≤ BLOCK_N_SIZE)
  └─ rmsnorm_implementation_correct
```

`rmsnorm_implementation_compute_fullN_correct` is the strongest result: it covers
arbitrary `N_SIZE` (var and output loops each tiled over `BLOCK_N_SIZE` and
closed by `forRange` loop invariants). The one-block `*_compute_correct` is the
single-iteration specialization. The RMSNorm row math is defined inline in this
file rather than reusing `VeriTile.Triton.Math.RMSNorm`.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `.to(tl.float32)` cast
on the loaded input reduces to identity at the algorithm layer (post-erasure all
dtypes unify to `ℝ`). The variance reduction `tl.sum(var) / N_SIZE` sums over the
*padded* `BLOCK_N_SIZE` block, but out-of-range lanes are masked to `0` (load
`other=0.0`), so the sum equals the logical row length `N_SIZE`. Unlike the
reciprocal-`rstd` variants, this kernel computes `std = sqrt(meanSq + eps)` and
divides (`x / std`); the spec models this as `x * rmsInvVarFullN` where
`rmsInvVarFullN = (rmsStdFullNSpec)⁻¹` and `rmsStdFullNSpec = sqrt(meanSq + eps)`.
The affine step multiplies by the per-column weight. The fullN theorem requires
`0 < BLOCK_N_SIZE`, `0 < stride_out_k` (column-store injectivity), and
output/input region disjointness (`x_ptr ≠ out_ptr`, `rms_w_ptr ≠ out_ptr`); the
one-block slice instead assumes `0 < N_SIZE ≤ BLOCK_N_SIZE` and output
injectivity. `@triton.autotune` is not modeled.
-/

namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `rmsnorm_implementation.py`'s `rmsnorm_triton`.

Allowed mechanical Lean-syntax-only changes:
- Python `N_SIZE: tl.constexpr` / `eps: tl.constexpr` / `BLOCK_N_SIZE: tl.constexpr`
  -> Lean `Nat` / `ℝ` parameters. -/
def rmsnorm_implementation
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k : Nat)
    (N_SIZE BLOCK_N_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(0)
  pid_m = tl.program_id(1)
  offset_m = pid_batch * $(stride_x_batch) + pid_m * $(stride_x_m)
  block_n_size = tl.arange(0, $(BLOCK_N_SIZE))
  var = tl.zeros([$(BLOCK_N_SIZE)], tl.float32)
  for block_n_strart_ptr in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offset_n = block_n_strart_ptr + block_n_size
    x_ptr_mask = offset_n < $(N_SIZE)
    x = tl.load(x_ptr + offset_m + offset_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    xf = (x).to(tl.float32)
    var += xf * xf
  }
  var = tl.sum(var, axis=0) / $(N_SIZE)
  std = tl.sqrt(var + $(eps))
  for block_n_strart_ptr in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offset_n = block_n_strart_ptr + block_n_size
    x_ptr_mask = offset_n < $(N_SIZE)
    rms_w_offset = tl.load(rms_w_ptr + offset_n * $(stride_rms_w), mask=x_ptr_mask)
    x = tl.load(x_ptr + offset_m + offset_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    x_new = x / std
    out = x_new * rms_w_offset
    out_offset = pid_batch * $(stride_out_batch) + pid_m * $(stride_out_m) +
      offset_n * $(stride_out_k)
    tl.store(out_ptr + out_offset, out, mask=x_ptr_mask)
  }
}


open Lean
syntax "rmsnormTyped%" "[" tritonDType "," tritonDType "," tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(rmsnormTyped%[$input:tritonDType, $weight:tritonDType, $output:tritonDType]) => do
    let wide := match input, weight with
      | `(tritonDType| tl.float64), _ => Bool.true
      | _, `(tritonDType| tl.float64) => Bool.true
      | _, _ => Bool.false
    let same := match output with
      | `(tritonDType| tl.float64) => wide
      | `(tritonDType| tl.float32) => !wide
      | _ => Bool.false
    let store : TSyntax `tritonStmt ← if same then
      `(tritonStmt| tl.store(out_ptr + out_offset, out, mask=x_ptr_mask))
      else `(tritonStmt| tl.store(out_ptr + out_offset, out.to($output), mask=x_ptr_mask))
    `(fun (x_ptr rms_w_ptr out_ptr : RegionName)
        (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) =>
      ComputeKernel.mk [x_ptr, rms_w_ptr] [out_ptr] (triton {
  tl.region x_ptr = $input, rms_w_ptr = $weight, out_ptr = $output
  pid_batch = tl.program_id(0)
  pid_m = tl.program_id(1)
  offset_m = pid_batch * $(sxb) + pid_m * $(sxm)
  block_n_size = tl.arange(0, $(B))
  var = tl.zeros([$(B)], tl.float32)
  for block_n_strart_ptr in range(0, $(N), $(B)) {
    offset_n = block_n_strart_ptr + block_n_size
    x_ptr_mask = offset_n < $(N)
    x = tl.load(x_ptr + offset_m + offset_n * $(sxk), mask=x_ptr_mask, other=0.0)
    xf = (x).to(tl.float32)
    var += xf * xf
  }
  var = tl.sum(var, axis=0) / $(N)
  std = tl.sqrt(var + $(eps))
  for block_n_strart_ptr in range(0, $(N), $(B)) {
    offset_n = block_n_strart_ptr + block_n_size
    x_ptr_mask = offset_n < $(N)
    rms_w_offset = tl.load(rms_w_ptr + offset_n * $(sw), mask=x_ptr_mask)
    x = tl.load(x_ptr + offset_m + offset_n * $(sxk), mask=x_ptr_mask, other=0.0)
    x_new = x / std
    out = x_new * rms_w_offset
    out_offset = pid_batch * $(sob) + pid_m * $(som) +
      offset_n * $(sok)
    $store
  }
}).surfaceBody)

/-- Independent input, weight and output storage; the wrapper uses output=input. -/
inductive Storage where
  | fp32 | fp64 | fp16 | bf16 | f8e4 | f8e5
  deriving DecidableEq, Repr

abbrev Source := RegionName → RegionName → RegionName →
  Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → ℝ → ComputeKernel

set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

/-- All source statements in both loops, including the uncast second input load
and weight load without `other`. The explicit store conversion models Triton's
implicit conversion to the output pointer element type. -/
def originalTyped (input weight output : Storage) : Source :=
  match input, weight, output with
  | .fp32, .fp32, .fp32 => rmsnormTyped%[tl.float32, tl.float32, tl.float32]
  | .fp32, .fp32, .fp64 => rmsnormTyped%[tl.float32, tl.float32, tl.float64]
  | .fp32, .fp32, .fp16 => rmsnormTyped%[tl.float32, tl.float32, tl.float16]
  | .fp32, .fp32, .bf16 => rmsnormTyped%[tl.float32, tl.float32, tl.bfloat16]
  | .fp32, .fp32, .f8e4 => rmsnormTyped%[tl.float32, tl.float32, tl.float8e4nv]
  | .fp32, .fp32, .f8e5 => rmsnormTyped%[tl.float32, tl.float32, tl.float8e5]
  | .fp32, .fp64, .fp32 => rmsnormTyped%[tl.float32, tl.float64, tl.float32]
  | .fp32, .fp64, .fp64 => rmsnormTyped%[tl.float32, tl.float64, tl.float64]
  | .fp32, .fp64, .fp16 => rmsnormTyped%[tl.float32, tl.float64, tl.float16]
  | .fp32, .fp64, .bf16 => rmsnormTyped%[tl.float32, tl.float64, tl.bfloat16]
  | .fp32, .fp64, .f8e4 => rmsnormTyped%[tl.float32, tl.float64, tl.float8e4nv]
  | .fp32, .fp64, .f8e5 => rmsnormTyped%[tl.float32, tl.float64, tl.float8e5]
  | .fp32, .fp16, .fp32 => rmsnormTyped%[tl.float32, tl.float16, tl.float32]
  | .fp32, .fp16, .fp64 => rmsnormTyped%[tl.float32, tl.float16, tl.float64]
  | .fp32, .fp16, .fp16 => rmsnormTyped%[tl.float32, tl.float16, tl.float16]
  | .fp32, .fp16, .bf16 => rmsnormTyped%[tl.float32, tl.float16, tl.bfloat16]
  | .fp32, .fp16, .f8e4 => rmsnormTyped%[tl.float32, tl.float16, tl.float8e4nv]
  | .fp32, .fp16, .f8e5 => rmsnormTyped%[tl.float32, tl.float16, tl.float8e5]
  | .fp32, .bf16, .fp32 => rmsnormTyped%[tl.float32, tl.bfloat16, tl.float32]
  | .fp32, .bf16, .fp64 => rmsnormTyped%[tl.float32, tl.bfloat16, tl.float64]
  | .fp32, .bf16, .fp16 => rmsnormTyped%[tl.float32, tl.bfloat16, tl.float16]
  | .fp32, .bf16, .bf16 => rmsnormTyped%[tl.float32, tl.bfloat16, tl.bfloat16]
  | .fp32, .bf16, .f8e4 => rmsnormTyped%[tl.float32, tl.bfloat16, tl.float8e4nv]
  | .fp32, .bf16, .f8e5 => rmsnormTyped%[tl.float32, tl.bfloat16, tl.float8e5]
  | .fp32, .f8e4, .fp32 => rmsnormTyped%[tl.float32, tl.float8e4nv, tl.float32]
  | .fp32, .f8e4, .fp64 => rmsnormTyped%[tl.float32, tl.float8e4nv, tl.float64]
  | .fp32, .f8e4, .fp16 => rmsnormTyped%[tl.float32, tl.float8e4nv, tl.float16]
  | .fp32, .f8e4, .bf16 => rmsnormTyped%[tl.float32, tl.float8e4nv, tl.bfloat16]
  | .fp32, .f8e4, .f8e4 => rmsnormTyped%[tl.float32, tl.float8e4nv, tl.float8e4nv]
  | .fp32, .f8e4, .f8e5 => rmsnormTyped%[tl.float32, tl.float8e4nv, tl.float8e5]
  | .fp32, .f8e5, .fp32 => rmsnormTyped%[tl.float32, tl.float8e5, tl.float32]
  | .fp32, .f8e5, .fp64 => rmsnormTyped%[tl.float32, tl.float8e5, tl.float64]
  | .fp32, .f8e5, .fp16 => rmsnormTyped%[tl.float32, tl.float8e5, tl.float16]
  | .fp32, .f8e5, .bf16 => rmsnormTyped%[tl.float32, tl.float8e5, tl.bfloat16]
  | .fp32, .f8e5, .f8e4 => rmsnormTyped%[tl.float32, tl.float8e5, tl.float8e4nv]
  | .fp32, .f8e5, .f8e5 => rmsnormTyped%[tl.float32, tl.float8e5, tl.float8e5]
  | .fp64, .fp32, .fp32 => rmsnormTyped%[tl.float64, tl.float32, tl.float32]
  | .fp64, .fp32, .fp64 => rmsnormTyped%[tl.float64, tl.float32, tl.float64]
  | .fp64, .fp32, .fp16 => rmsnormTyped%[tl.float64, tl.float32, tl.float16]
  | .fp64, .fp32, .bf16 => rmsnormTyped%[tl.float64, tl.float32, tl.bfloat16]
  | .fp64, .fp32, .f8e4 => rmsnormTyped%[tl.float64, tl.float32, tl.float8e4nv]
  | .fp64, .fp32, .f8e5 => rmsnormTyped%[tl.float64, tl.float32, tl.float8e5]
  | .fp64, .fp64, .fp32 => rmsnormTyped%[tl.float64, tl.float64, tl.float32]
  | .fp64, .fp64, .fp64 => rmsnormTyped%[tl.float64, tl.float64, tl.float64]
  | .fp64, .fp64, .fp16 => rmsnormTyped%[tl.float64, tl.float64, tl.float16]
  | .fp64, .fp64, .bf16 => rmsnormTyped%[tl.float64, tl.float64, tl.bfloat16]
  | .fp64, .fp64, .f8e4 => rmsnormTyped%[tl.float64, tl.float64, tl.float8e4nv]
  | .fp64, .fp64, .f8e5 => rmsnormTyped%[tl.float64, tl.float64, tl.float8e5]
  | .fp64, .fp16, .fp32 => rmsnormTyped%[tl.float64, tl.float16, tl.float32]
  | .fp64, .fp16, .fp64 => rmsnormTyped%[tl.float64, tl.float16, tl.float64]
  | .fp64, .fp16, .fp16 => rmsnormTyped%[tl.float64, tl.float16, tl.float16]
  | .fp64, .fp16, .bf16 => rmsnormTyped%[tl.float64, tl.float16, tl.bfloat16]
  | .fp64, .fp16, .f8e4 => rmsnormTyped%[tl.float64, tl.float16, tl.float8e4nv]
  | .fp64, .fp16, .f8e5 => rmsnormTyped%[tl.float64, tl.float16, tl.float8e5]
  | .fp64, .bf16, .fp32 => rmsnormTyped%[tl.float64, tl.bfloat16, tl.float32]
  | .fp64, .bf16, .fp64 => rmsnormTyped%[tl.float64, tl.bfloat16, tl.float64]
  | .fp64, .bf16, .fp16 => rmsnormTyped%[tl.float64, tl.bfloat16, tl.float16]
  | .fp64, .bf16, .bf16 => rmsnormTyped%[tl.float64, tl.bfloat16, tl.bfloat16]
  | .fp64, .bf16, .f8e4 => rmsnormTyped%[tl.float64, tl.bfloat16, tl.float8e4nv]
  | .fp64, .bf16, .f8e5 => rmsnormTyped%[tl.float64, tl.bfloat16, tl.float8e5]
  | .fp64, .f8e4, .fp32 => rmsnormTyped%[tl.float64, tl.float8e4nv, tl.float32]
  | .fp64, .f8e4, .fp64 => rmsnormTyped%[tl.float64, tl.float8e4nv, tl.float64]
  | .fp64, .f8e4, .fp16 => rmsnormTyped%[tl.float64, tl.float8e4nv, tl.float16]
  | .fp64, .f8e4, .bf16 => rmsnormTyped%[tl.float64, tl.float8e4nv, tl.bfloat16]
  | .fp64, .f8e4, .f8e4 => rmsnormTyped%[tl.float64, tl.float8e4nv, tl.float8e4nv]
  | .fp64, .f8e4, .f8e5 => rmsnormTyped%[tl.float64, tl.float8e4nv, tl.float8e5]
  | .fp64, .f8e5, .fp32 => rmsnormTyped%[tl.float64, tl.float8e5, tl.float32]
  | .fp64, .f8e5, .fp64 => rmsnormTyped%[tl.float64, tl.float8e5, tl.float64]
  | .fp64, .f8e5, .fp16 => rmsnormTyped%[tl.float64, tl.float8e5, tl.float16]
  | .fp64, .f8e5, .bf16 => rmsnormTyped%[tl.float64, tl.float8e5, tl.bfloat16]
  | .fp64, .f8e5, .f8e4 => rmsnormTyped%[tl.float64, tl.float8e5, tl.float8e4nv]
  | .fp64, .f8e5, .f8e5 => rmsnormTyped%[tl.float64, tl.float8e5, tl.float8e5]
  | .fp16, .fp32, .fp32 => rmsnormTyped%[tl.float16, tl.float32, tl.float32]
  | .fp16, .fp32, .fp64 => rmsnormTyped%[tl.float16, tl.float32, tl.float64]
  | .fp16, .fp32, .fp16 => rmsnormTyped%[tl.float16, tl.float32, tl.float16]
  | .fp16, .fp32, .bf16 => rmsnormTyped%[tl.float16, tl.float32, tl.bfloat16]
  | .fp16, .fp32, .f8e4 => rmsnormTyped%[tl.float16, tl.float32, tl.float8e4nv]
  | .fp16, .fp32, .f8e5 => rmsnormTyped%[tl.float16, tl.float32, tl.float8e5]
  | .fp16, .fp64, .fp32 => rmsnormTyped%[tl.float16, tl.float64, tl.float32]
  | .fp16, .fp64, .fp64 => rmsnormTyped%[tl.float16, tl.float64, tl.float64]
  | .fp16, .fp64, .fp16 => rmsnormTyped%[tl.float16, tl.float64, tl.float16]
  | .fp16, .fp64, .bf16 => rmsnormTyped%[tl.float16, tl.float64, tl.bfloat16]
  | .fp16, .fp64, .f8e4 => rmsnormTyped%[tl.float16, tl.float64, tl.float8e4nv]
  | .fp16, .fp64, .f8e5 => rmsnormTyped%[tl.float16, tl.float64, tl.float8e5]
  | .fp16, .fp16, .fp32 => rmsnormTyped%[tl.float16, tl.float16, tl.float32]
  | .fp16, .fp16, .fp64 => rmsnormTyped%[tl.float16, tl.float16, tl.float64]
  | .fp16, .fp16, .fp16 => rmsnormTyped%[tl.float16, tl.float16, tl.float16]
  | .fp16, .fp16, .bf16 => rmsnormTyped%[tl.float16, tl.float16, tl.bfloat16]
  | .fp16, .fp16, .f8e4 => rmsnormTyped%[tl.float16, tl.float16, tl.float8e4nv]
  | .fp16, .fp16, .f8e5 => rmsnormTyped%[tl.float16, tl.float16, tl.float8e5]
  | .fp16, .bf16, .fp32 => rmsnormTyped%[tl.float16, tl.bfloat16, tl.float32]
  | .fp16, .bf16, .fp64 => rmsnormTyped%[tl.float16, tl.bfloat16, tl.float64]
  | .fp16, .bf16, .fp16 => rmsnormTyped%[tl.float16, tl.bfloat16, tl.float16]
  | .fp16, .bf16, .bf16 => rmsnormTyped%[tl.float16, tl.bfloat16, tl.bfloat16]
  | .fp16, .bf16, .f8e4 => rmsnormTyped%[tl.float16, tl.bfloat16, tl.float8e4nv]
  | .fp16, .bf16, .f8e5 => rmsnormTyped%[tl.float16, tl.bfloat16, tl.float8e5]
  | .fp16, .f8e4, .fp32 => rmsnormTyped%[tl.float16, tl.float8e4nv, tl.float32]
  | .fp16, .f8e4, .fp64 => rmsnormTyped%[tl.float16, tl.float8e4nv, tl.float64]
  | .fp16, .f8e4, .fp16 => rmsnormTyped%[tl.float16, tl.float8e4nv, tl.float16]
  | .fp16, .f8e4, .bf16 => rmsnormTyped%[tl.float16, tl.float8e4nv, tl.bfloat16]
  | .fp16, .f8e4, .f8e4 => rmsnormTyped%[tl.float16, tl.float8e4nv, tl.float8e4nv]
  | .fp16, .f8e4, .f8e5 => rmsnormTyped%[tl.float16, tl.float8e4nv, tl.float8e5]
  | .fp16, .f8e5, .fp32 => rmsnormTyped%[tl.float16, tl.float8e5, tl.float32]
  | .fp16, .f8e5, .fp64 => rmsnormTyped%[tl.float16, tl.float8e5, tl.float64]
  | .fp16, .f8e5, .fp16 => rmsnormTyped%[tl.float16, tl.float8e5, tl.float16]
  | .fp16, .f8e5, .bf16 => rmsnormTyped%[tl.float16, tl.float8e5, tl.bfloat16]
  | .fp16, .f8e5, .f8e4 => rmsnormTyped%[tl.float16, tl.float8e5, tl.float8e4nv]
  | .fp16, .f8e5, .f8e5 => rmsnormTyped%[tl.float16, tl.float8e5, tl.float8e5]
  | .bf16, .fp32, .fp32 => rmsnormTyped%[tl.bfloat16, tl.float32, tl.float32]
  | .bf16, .fp32, .fp64 => rmsnormTyped%[tl.bfloat16, tl.float32, tl.float64]
  | .bf16, .fp32, .fp16 => rmsnormTyped%[tl.bfloat16, tl.float32, tl.float16]
  | .bf16, .fp32, .bf16 => rmsnormTyped%[tl.bfloat16, tl.float32, tl.bfloat16]
  | .bf16, .fp32, .f8e4 => rmsnormTyped%[tl.bfloat16, tl.float32, tl.float8e4nv]
  | .bf16, .fp32, .f8e5 => rmsnormTyped%[tl.bfloat16, tl.float32, tl.float8e5]
  | .bf16, .fp64, .fp32 => rmsnormTyped%[tl.bfloat16, tl.float64, tl.float32]
  | .bf16, .fp64, .fp64 => rmsnormTyped%[tl.bfloat16, tl.float64, tl.float64]
  | .bf16, .fp64, .fp16 => rmsnormTyped%[tl.bfloat16, tl.float64, tl.float16]
  | .bf16, .fp64, .bf16 => rmsnormTyped%[tl.bfloat16, tl.float64, tl.bfloat16]
  | .bf16, .fp64, .f8e4 => rmsnormTyped%[tl.bfloat16, tl.float64, tl.float8e4nv]
  | .bf16, .fp64, .f8e5 => rmsnormTyped%[tl.bfloat16, tl.float64, tl.float8e5]
  | .bf16, .fp16, .fp32 => rmsnormTyped%[tl.bfloat16, tl.float16, tl.float32]
  | .bf16, .fp16, .fp64 => rmsnormTyped%[tl.bfloat16, tl.float16, tl.float64]
  | .bf16, .fp16, .fp16 => rmsnormTyped%[tl.bfloat16, tl.float16, tl.float16]
  | .bf16, .fp16, .bf16 => rmsnormTyped%[tl.bfloat16, tl.float16, tl.bfloat16]
  | .bf16, .fp16, .f8e4 => rmsnormTyped%[tl.bfloat16, tl.float16, tl.float8e4nv]
  | .bf16, .fp16, .f8e5 => rmsnormTyped%[tl.bfloat16, tl.float16, tl.float8e5]
  | .bf16, .bf16, .fp32 => rmsnormTyped%[tl.bfloat16, tl.bfloat16, tl.float32]
  | .bf16, .bf16, .fp64 => rmsnormTyped%[tl.bfloat16, tl.bfloat16, tl.float64]
  | .bf16, .bf16, .fp16 => rmsnormTyped%[tl.bfloat16, tl.bfloat16, tl.float16]
  | .bf16, .bf16, .bf16 => rmsnormTyped%[tl.bfloat16, tl.bfloat16, tl.bfloat16]
  | .bf16, .bf16, .f8e4 => rmsnormTyped%[tl.bfloat16, tl.bfloat16, tl.float8e4nv]
  | .bf16, .bf16, .f8e5 => rmsnormTyped%[tl.bfloat16, tl.bfloat16, tl.float8e5]
  | .bf16, .f8e4, .fp32 => rmsnormTyped%[tl.bfloat16, tl.float8e4nv, tl.float32]
  | .bf16, .f8e4, .fp64 => rmsnormTyped%[tl.bfloat16, tl.float8e4nv, tl.float64]
  | .bf16, .f8e4, .fp16 => rmsnormTyped%[tl.bfloat16, tl.float8e4nv, tl.float16]
  | .bf16, .f8e4, .bf16 => rmsnormTyped%[tl.bfloat16, tl.float8e4nv, tl.bfloat16]
  | .bf16, .f8e4, .f8e4 => rmsnormTyped%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv]
  | .bf16, .f8e4, .f8e5 => rmsnormTyped%[tl.bfloat16, tl.float8e4nv, tl.float8e5]
  | .bf16, .f8e5, .fp32 => rmsnormTyped%[tl.bfloat16, tl.float8e5, tl.float32]
  | .bf16, .f8e5, .fp64 => rmsnormTyped%[tl.bfloat16, tl.float8e5, tl.float64]
  | .bf16, .f8e5, .fp16 => rmsnormTyped%[tl.bfloat16, tl.float8e5, tl.float16]
  | .bf16, .f8e5, .bf16 => rmsnormTyped%[tl.bfloat16, tl.float8e5, tl.bfloat16]
  | .bf16, .f8e5, .f8e4 => rmsnormTyped%[tl.bfloat16, tl.float8e5, tl.float8e4nv]
  | .bf16, .f8e5, .f8e5 => rmsnormTyped%[tl.bfloat16, tl.float8e5, tl.float8e5]
  | .f8e4, .fp32, .fp32 => rmsnormTyped%[tl.float8e4nv, tl.float32, tl.float32]
  | .f8e4, .fp32, .fp64 => rmsnormTyped%[tl.float8e4nv, tl.float32, tl.float64]
  | .f8e4, .fp32, .fp16 => rmsnormTyped%[tl.float8e4nv, tl.float32, tl.float16]
  | .f8e4, .fp32, .bf16 => rmsnormTyped%[tl.float8e4nv, tl.float32, tl.bfloat16]
  | .f8e4, .fp32, .f8e4 => rmsnormTyped%[tl.float8e4nv, tl.float32, tl.float8e4nv]
  | .f8e4, .fp32, .f8e5 => rmsnormTyped%[tl.float8e4nv, tl.float32, tl.float8e5]
  | .f8e4, .fp64, .fp32 => rmsnormTyped%[tl.float8e4nv, tl.float64, tl.float32]
  | .f8e4, .fp64, .fp64 => rmsnormTyped%[tl.float8e4nv, tl.float64, tl.float64]
  | .f8e4, .fp64, .fp16 => rmsnormTyped%[tl.float8e4nv, tl.float64, tl.float16]
  | .f8e4, .fp64, .bf16 => rmsnormTyped%[tl.float8e4nv, tl.float64, tl.bfloat16]
  | .f8e4, .fp64, .f8e4 => rmsnormTyped%[tl.float8e4nv, tl.float64, tl.float8e4nv]
  | .f8e4, .fp64, .f8e5 => rmsnormTyped%[tl.float8e4nv, tl.float64, tl.float8e5]
  | .f8e4, .fp16, .fp32 => rmsnormTyped%[tl.float8e4nv, tl.float16, tl.float32]
  | .f8e4, .fp16, .fp64 => rmsnormTyped%[tl.float8e4nv, tl.float16, tl.float64]
  | .f8e4, .fp16, .fp16 => rmsnormTyped%[tl.float8e4nv, tl.float16, tl.float16]
  | .f8e4, .fp16, .bf16 => rmsnormTyped%[tl.float8e4nv, tl.float16, tl.bfloat16]
  | .f8e4, .fp16, .f8e4 => rmsnormTyped%[tl.float8e4nv, tl.float16, tl.float8e4nv]
  | .f8e4, .fp16, .f8e5 => rmsnormTyped%[tl.float8e4nv, tl.float16, tl.float8e5]
  | .f8e4, .bf16, .fp32 => rmsnormTyped%[tl.float8e4nv, tl.bfloat16, tl.float32]
  | .f8e4, .bf16, .fp64 => rmsnormTyped%[tl.float8e4nv, tl.bfloat16, tl.float64]
  | .f8e4, .bf16, .fp16 => rmsnormTyped%[tl.float8e4nv, tl.bfloat16, tl.float16]
  | .f8e4, .bf16, .bf16 => rmsnormTyped%[tl.float8e4nv, tl.bfloat16, tl.bfloat16]
  | .f8e4, .bf16, .f8e4 => rmsnormTyped%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv]
  | .f8e4, .bf16, .f8e5 => rmsnormTyped%[tl.float8e4nv, tl.bfloat16, tl.float8e5]
  | .f8e4, .f8e4, .fp32 => rmsnormTyped%[tl.float8e4nv, tl.float8e4nv, tl.float32]
  | .f8e4, .f8e4, .fp64 => rmsnormTyped%[tl.float8e4nv, tl.float8e4nv, tl.float64]
  | .f8e4, .f8e4, .fp16 => rmsnormTyped%[tl.float8e4nv, tl.float8e4nv, tl.float16]
  | .f8e4, .f8e4, .bf16 => rmsnormTyped%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16]
  | .f8e4, .f8e4, .f8e4 => rmsnormTyped%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv]
  | .f8e4, .f8e4, .f8e5 => rmsnormTyped%[tl.float8e4nv, tl.float8e4nv, tl.float8e5]
  | .f8e4, .f8e5, .fp32 => rmsnormTyped%[tl.float8e4nv, tl.float8e5, tl.float32]
  | .f8e4, .f8e5, .fp64 => rmsnormTyped%[tl.float8e4nv, tl.float8e5, tl.float64]
  | .f8e4, .f8e5, .fp16 => rmsnormTyped%[tl.float8e4nv, tl.float8e5, tl.float16]
  | .f8e4, .f8e5, .bf16 => rmsnormTyped%[tl.float8e4nv, tl.float8e5, tl.bfloat16]
  | .f8e4, .f8e5, .f8e4 => rmsnormTyped%[tl.float8e4nv, tl.float8e5, tl.float8e4nv]
  | .f8e4, .f8e5, .f8e5 => rmsnormTyped%[tl.float8e4nv, tl.float8e5, tl.float8e5]
  | .f8e5, .fp32, .fp32 => rmsnormTyped%[tl.float8e5, tl.float32, tl.float32]
  | .f8e5, .fp32, .fp64 => rmsnormTyped%[tl.float8e5, tl.float32, tl.float64]
  | .f8e5, .fp32, .fp16 => rmsnormTyped%[tl.float8e5, tl.float32, tl.float16]
  | .f8e5, .fp32, .bf16 => rmsnormTyped%[tl.float8e5, tl.float32, tl.bfloat16]
  | .f8e5, .fp32, .f8e4 => rmsnormTyped%[tl.float8e5, tl.float32, tl.float8e4nv]
  | .f8e5, .fp32, .f8e5 => rmsnormTyped%[tl.float8e5, tl.float32, tl.float8e5]
  | .f8e5, .fp64, .fp32 => rmsnormTyped%[tl.float8e5, tl.float64, tl.float32]
  | .f8e5, .fp64, .fp64 => rmsnormTyped%[tl.float8e5, tl.float64, tl.float64]
  | .f8e5, .fp64, .fp16 => rmsnormTyped%[tl.float8e5, tl.float64, tl.float16]
  | .f8e5, .fp64, .bf16 => rmsnormTyped%[tl.float8e5, tl.float64, tl.bfloat16]
  | .f8e5, .fp64, .f8e4 => rmsnormTyped%[tl.float8e5, tl.float64, tl.float8e4nv]
  | .f8e5, .fp64, .f8e5 => rmsnormTyped%[tl.float8e5, tl.float64, tl.float8e5]
  | .f8e5, .fp16, .fp32 => rmsnormTyped%[tl.float8e5, tl.float16, tl.float32]
  | .f8e5, .fp16, .fp64 => rmsnormTyped%[tl.float8e5, tl.float16, tl.float64]
  | .f8e5, .fp16, .fp16 => rmsnormTyped%[tl.float8e5, tl.float16, tl.float16]
  | .f8e5, .fp16, .bf16 => rmsnormTyped%[tl.float8e5, tl.float16, tl.bfloat16]
  | .f8e5, .fp16, .f8e4 => rmsnormTyped%[tl.float8e5, tl.float16, tl.float8e4nv]
  | .f8e5, .fp16, .f8e5 => rmsnormTyped%[tl.float8e5, tl.float16, tl.float8e5]
  | .f8e5, .bf16, .fp32 => rmsnormTyped%[tl.float8e5, tl.bfloat16, tl.float32]
  | .f8e5, .bf16, .fp64 => rmsnormTyped%[tl.float8e5, tl.bfloat16, tl.float64]
  | .f8e5, .bf16, .fp16 => rmsnormTyped%[tl.float8e5, tl.bfloat16, tl.float16]
  | .f8e5, .bf16, .bf16 => rmsnormTyped%[tl.float8e5, tl.bfloat16, tl.bfloat16]
  | .f8e5, .bf16, .f8e4 => rmsnormTyped%[tl.float8e5, tl.bfloat16, tl.float8e4nv]
  | .f8e5, .bf16, .f8e5 => rmsnormTyped%[tl.float8e5, tl.bfloat16, tl.float8e5]
  | .f8e5, .f8e4, .fp32 => rmsnormTyped%[tl.float8e5, tl.float8e4nv, tl.float32]
  | .f8e5, .f8e4, .fp64 => rmsnormTyped%[tl.float8e5, tl.float8e4nv, tl.float64]
  | .f8e5, .f8e4, .fp16 => rmsnormTyped%[tl.float8e5, tl.float8e4nv, tl.float16]
  | .f8e5, .f8e4, .bf16 => rmsnormTyped%[tl.float8e5, tl.float8e4nv, tl.bfloat16]
  | .f8e5, .f8e4, .f8e4 => rmsnormTyped%[tl.float8e5, tl.float8e4nv, tl.float8e4nv]
  | .f8e5, .f8e4, .f8e5 => rmsnormTyped%[tl.float8e5, tl.float8e4nv, tl.float8e5]
  | .f8e5, .f8e5, .fp32 => rmsnormTyped%[tl.float8e5, tl.float8e5, tl.float32]
  | .f8e5, .f8e5, .fp64 => rmsnormTyped%[tl.float8e5, tl.float8e5, tl.float64]
  | .f8e5, .f8e5, .fp16 => rmsnormTyped%[tl.float8e5, tl.float8e5, tl.float16]
  | .f8e5, .f8e5, .bf16 => rmsnormTyped%[tl.float8e5, tl.float8e5, tl.bfloat16]
  | .f8e5, .f8e5, .f8e4 => rmsnormTyped%[tl.float8e5, tl.float8e5, tl.float8e4nv]
  | .f8e5, .f8e5, .f8e5 => rmsnormTyped%[tl.float8e5, tl.float8e5, tl.float8e5]

/-- Candidate changes only the first row-address product's operand order. -/
def typedKernel (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  let source := originalTyped input weight output X W O sxb sxm sxk sw sob som sok N B eps
  if v then ComputeKernel.mk source.inputs source.outputs
    (source.surfaceBody.take 2 ++
      [.assign .nat [] "offset_m" (.alg
        (.add .nat .nil
          (.mul .nat .nil (.constNat sxb) (.ref .nat [] "pid_batch"))
          (.mul .nat .nil (.ref .nat [] "pid_m") (.constNat sxm))))] ++
      source.surfaceBody.drop 3)
  else source

end VeriTile.Bench.TritonBenchG.RmsnormImplementation
