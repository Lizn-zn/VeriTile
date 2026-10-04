import VeriTile.Triton.DSL

/-!
FusedSiLU: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
-/

namespace VeriTile.Bench.Examples.FusedSiLU.Kernels
open VeriTile Triton

/-- Fused SiLU with a bf16-rounded output store. -/
def fusedSiLUKernel (xReg gateReg residualReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  x        := tl.load($(xReg) + offsets)
  gate     := tl.load($(gateReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  z        := x * gate
  silu     := z * tl.sigmoid(z)
  y        := residual + silu
  tl.store($(outReg) + offsets, (y).to(tl.bfloat16))
}

/-- Step A: materialize `z = x * gate` into the ℝ scratch tensor `z`. -/
def siluStepGate (xReg gateReg zReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  x       := tl.load($(xReg) + offsets)
  gate    := tl.load($(gateReg) + offsets)
  z       := x * gate
  tl.store($(zReg) + offsets, z)
}

/-- Step B: materialize `silu = z * sigmoid(z)` into the ℝ scratch tensor `silu`. -/
def siluStepSilu (zReg siluReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  z       := tl.load($(zReg) + offsets)
  silu    := z * tl.sigmoid(z)
  tl.store($(siluReg) + offsets, silu)
}

/-- Step C: `out = residual + silu`, with a bf16-rounded output store. -/
def siluStepResidual (siluReg residualReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  silu     := tl.load($(siluReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  y        := residual + silu
  tl.store($(outReg) + offsets, (y).to(tl.bfloat16))
}

/-- The unfused pipeline as one kernel: `ComputeKernel.seq` of the three step
kernels — the concatenation of their bodies (registers flow across the seams).
The staged execution is recovered by `unfused_exec_split` below. -/
def unfusedSiLUKernel
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel :=
  ComputeKernel.seq [xReg, gateReg, residualReg] [outReg]
    [siluStepGate xReg gateReg zReg blockSize,
     siluStepSilu zReg siluReg blockSize,
     siluStepResidual siluReg residualReg outReg blockSize]

end VeriTile.Bench.Examples.FusedSiLU.Kernels
