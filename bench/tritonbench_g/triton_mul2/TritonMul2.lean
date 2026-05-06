import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.TritonMul2

open VeriTile.Triton

/-- Faithful 1:1 transcription of `triton_mul2.py`'s `mul2_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python pointer args → Lean `RegionName` injected via `$(...)`.
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def mul2_kernel
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load($(in_ptr0) + offsets, mask=mask)
  output = 2 * x
  tl.store($(out_ptr) + offsets, output, mask=mask)
}

/-- Faithful 1:1 transcription of `triton_mul2.py`'s `mul2_inplace_kernel`.

Same allowed mechanical Lean-syntax-only changes as `mul2_kernel`. -/
def mul2_inplace_kernel
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load($(ptr) + offsets, mask=mask)
  output = 2 * x
  tl.store($(ptr) + offsets, output, mask=mask)
}

end VeriTile.Bench.TritonBenchG.TritonMul2
