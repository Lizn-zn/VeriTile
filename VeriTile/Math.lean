/-
VeriTile.Math

Barrel module for VeriTile's pure-math operator library.

Convention: `VeriTile.Math.*` collects mathematical operators and
identities that are (a) referenced by ≥ 2 kernel transcriptions or have
clear forthcoming reuse, (b) typed in `(Fin N → ℝ) → ℝ` style or similar
without `BlockState` / `RegionName` / memory-layout binding, and (c) carry
non-trivial mathematical content (named operator + at least one supporting
identity, not just a one-line argument-binding wrapper).

Kernel transcriptions in `bench/` and `Examples/` connect Triton's
memory/layout/grid behavior to these operators; they do not re-define them.
-/

import VeriTile.Math.Activation
import VeriTile.Math.L2Norm
import VeriTile.Math.LogSumExp
import VeriTile.Math.Loss
import VeriTile.Math.Optimizer
import VeriTile.Math.Reduction
import VeriTile.Math.RMSNorm
import VeriTile.Math.Softmax
