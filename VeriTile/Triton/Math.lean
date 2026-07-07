/-
VeriTile.Triton.Math

Barrel module for VeriTile's pure-math operator library.

Convention: `VeriTile.Triton.Math.*` collects mathematical operators and
identities that are (a) referenced by ≥ 2 kernel transcriptions or have
clear forthcoming reuse, (b) typed in `(Fin N → ℝ) → ℝ` style or similar
without `BlockState` / `RegionName` / memory-layout binding, and (c) carry
non-trivial mathematical content (named operator + at least one supporting
identity, not just a one-line argument-binding wrapper).

Kernel transcriptions in `bench/` and `Examples/` connect Triton's
memory/layout/grid behavior to these operators; they do not re-define them.
-/

import VeriTile.Triton.Math.Activation
import VeriTile.Triton.Math.Attention
import VeriTile.Triton.Math.Erf
import VeriTile.Triton.Math.L2Norm
import VeriTile.Triton.Math.LogSumExp
import VeriTile.Triton.Math.Loss
import VeriTile.Triton.Math.Optimizer
import VeriTile.Triton.Math.Reduction
import VeriTile.Triton.Math.RMSNorm
import VeriTile.Triton.Math.Softmax
