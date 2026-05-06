import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.SinKernel

open VeriTile.Triton

/-- Faithful 1:1 transcription of `sin_kernel.py`'s `kernel_function`.

Allowed mechanical Lean-syntax-only changes:
- Python pointer args → Lean `RegionName` injected via `$(...)`.
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def kernel_function
    (x_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load($(x_ptr) + offsets, mask=mask)
  output = tl.math.sin(x)
  tl.store($(output_ptr) + offsets, output, mask=mask)
}

end VeriTile.Bench.TritonBenchG.SinKernel
