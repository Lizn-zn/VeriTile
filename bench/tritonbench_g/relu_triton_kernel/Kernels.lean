import VeriTile.Triton
import VeriTile.Examples.Common

namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel
open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁
/-- Faithful 1:1 transcription of `relu_triton_kernel.py`'s `relu_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `N: tl.constexpr` / `block_size: tl.constexpr` → Lean `Nat`.
- Python `if cond: body` → `if cond { body }`, the DSL-side gate (block syntax
  uses braces because Lean is whitespace-insensitive; the `if` keyword itself
  is shared). -/
def relu_kernel
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = pid * $(block_size)
  offsets = block_start + tl.arange(0, $(block_size))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  result = tl.where(x >= 0, x, 0.0)
  if pid == 0 {
    tl.store(out_ptr + offsets, result, mask=mask)
  }
}

/-- Inline ReLU into the guarded store, avoiding work for nonzero programs. -/
def relu_fused_kernel
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = pid * $(block_size)
  offsets = block_start + tl.arange(0, $(block_size))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  if pid == 0 {
    tl.store(out_ptr + offsets, tl.where(x >= 0, x, 0.0), mask=mask)
  }
}

end VeriTile.Bench.TritonBenchG.ReluTritonKernel
