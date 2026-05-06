import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.LogsumexpFwd

open VeriTile.Triton

/-- Faithful 1:1 transcription of `logsumexp_fwd.py`'s `logsumexp_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `D: tl.constexpr` / `B: tl.constexpr` / `HAS_SCALE: tl.constexpr` →
  Lean parameters; the `tl.constexpr` annotation is implicit on Lean params.
- Python `if cond: body` → `tl.if cond { body }`, the DSL-side gate equivalent.
- `scale` (Lean `ℝ` parameter) injected via `$ℝ(...)`. -/
def logsumexp_fwd_kernel
    (x z : RegionName)
    (D B : Nat) (HAS_SCALE : Bool) (scale : ℝ) :
    ComputeKernel := triton {
  i_n = tl.program_id(0).to(tl.int64)
  i_d = tl.program_id(1).to(tl.int64)
  o_d = i_d * B + tl.arange(0, B)
  m_d = o_d < $(D)
  b_x = tl.load(x + i_n * $(D) + o_d, mask=m_d, other=-inf)
  tl.if $(HAS_SCALE) {
    b_x = b_x * $ℝ(scale)
  }
  b_m = tl.max(b_x, 0)
  b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
  tl.store(z + i_n * tl.cdiv($(D), $(B)) + i_d, b_z)
}

end VeriTile.Bench.TritonBenchG.LogsumexpFwd
