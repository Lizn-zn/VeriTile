import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.CosineCompute

open VeriTile.Triton

/-- Faithful 1:1 transcription of `cosine_compute.py`'s `cos_func`.

Allowed mechanical Lean-syntax-only changes:
- Python pointer args → Lean `RegionName` injected via `$(...)`. -/
def cos_func
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  offset = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offset < $(n_elements)
  a_value = tl.load($(a) + offset, mask=mask)
  b_value = tl.cos((a_value).to(tl.float32))
  tl.store($(b) + offset, b_value, mask=mask)
}

end VeriTile.Bench.TritonBenchG.CosineCompute
