import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.VectorAdditionCustom

open VeriTile.Triton

/-- Faithful 1:1 transcription of `vector_addition_custom.py`'s `_add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python pointer args → Lean `RegionName` injected via `$(...)`.
- Python `BLOCK: tl.constexpr` → Lean `Nat` parameter. -/
def _add_kernel
    (A B C : RegionName)
    (size BLOCK : Nat) :
    ComputeKernel := triton {
  prog_id = tl.program_id(0)
  offs = prog_id * $(BLOCK) + tl.arange(0, $(BLOCK))
  a = tl.load($(A) + offs, mask=offs < $(size))
  b = tl.load($(B) + offs, mask=offs < $(size))
  tl.store($(C) + offs, a + b, mask=offs < $(size))
}

end VeriTile.Bench.TritonBenchG.VectorAdditionCustom
