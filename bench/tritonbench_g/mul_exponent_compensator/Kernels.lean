import VeriTile.Triton

/-! The complete unmasked scaling source and a scalar-left candidate. The FP
entry selects fp32 input and arithmetic, with the wrapper's fp32 output.
The Python wrapper itself does not constrain the input dtype. Integer addresses are Nat;
launch scheduling and fixed-width address overflow remain outside this model. -/

namespace VeriTile.Bench.TritonBenchG.MulExponentCompensator
open VeriTile.Triton

/-- The constexpr multiplier from `mul_exponent_compensator.py`. -/
noncomputable def exponentCompensator : ℝ :=
  (2 : ℝ) ^ (127 - 15)

/-- Faithful 1:1 transcription of `mul_exponent_compensator.py`'s
`mul_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python local constexpr literal `2.0 ** (127 - 15)` is represented by the
  Lean constant `exponentCompensator` and injected as a real scalar antiquote.
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
noncomputable def mul_kernel
    (src dst : RegionName)
    (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  exponent_compensator = $((exponentCompensator : ℝ))
  idxs = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(src + idxs)
  y = x * exponent_compensator
  tl.store(dst + idxs, y)
}

/-- Scalar-left scaling with the same load, constant and store. -/
noncomputable def mul_commuted_kernel (src dst : RegionName) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  exponent_compensator = $((exponentCompensator : ℝ))
  idxs = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(src + idxs)
  y = exponent_compensator * x
  tl.store(dst + idxs, y)
}

end VeriTile.Bench.TritonBenchG.MulExponentCompensator
