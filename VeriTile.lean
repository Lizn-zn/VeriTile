-- VeriTile: top-level entry point.
-- Imports the public surface of the embedded Triton subset and worked examples.

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.LoopInvariant
import VeriTile.Triton.Examples
import VeriTile.Examples.SoftmaxEq
import VeriTile.Examples.LogSumExpEq
import VeriTile.Examples.SoftmaxReciprocal
import VeriTile.Examples.VectorAdd
import VeriTile.Examples.FusedSiLU
import VeriTile.Examples.ApproxGeLU
import VeriTile.Examples.RowWise
import VeriTile.Examples.WelfordMath
import VeriTile.Examples.WelfordKernels
import VeriTile.Examples.LayerNormKernels
import VeriTile.Examples.OnlineSoftmax
