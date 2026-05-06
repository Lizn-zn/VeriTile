import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.MatrixTranspose

open VeriTile.Triton

/-- Faithful 1:1 transcription of `matrix_transpose.py`'s `kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python pointer args → Lean `RegionName` injected via `$(...)`.
- Python `SIZE_M: tl.constexpr` / `D_HEAD: tl.constexpr` → Lean `Nat`
  parameters. -/
def kernel
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat) :
    ComputeKernel := triton {
  size_m_arange = tl.arange(0, $(SIZE_M))
  d_head_arange = tl.arange(0, $(D_HEAD))
  matrix_ptr = $(M) + d_head_arange[None, :] * $(matrix_stridey)
                + size_m_arange[:, None] * $(matrix_stridex)
  out_ptr = $(Out) + d_head_arange[None, :] * $(out_stridex)
             + size_m_arange[:, None] * $(out_stridey)
  matrix = tl.load(matrix_ptr)
  tl.store(out_ptr, matrix)
}

end VeriTile.Bench.TritonBenchG.MatrixTranspose
