-- VeriTile: default (lite) target — the neutral foundation + worked examples
-- without the analysis-heavy ApproxGeLU certificate.
--
-- ApproxGeLU and the GeLU error-certificate math (`VeriTile.Math.GeluTaylor20*`,
-- `RealErf`, `Tanh` — Taylor + integral bounds) live in the separate
-- `VeriTileMath` library and are pulled in by `VeriTileFull`. Routine kernel
-- work builds against this lite root, skipping the heavy Mathlib analysis chain.

import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Memory.Typing
import VeriTile.Memory
import VeriTile.Memory.Bounds
import VeriTile.Memory.Frame
import VeriTile.Memory.Footprint
import VeriTile.Concurrency
import VeriTile.Launch
import VeriTile.Kernel
import VeriTile.Examples.TritonSmoke
import VeriTile.Examples.MemorySafety
import VeriTile.Examples.MemoryFrame
import VeriTile.Examples.GridComposition
import VeriTile.Examples.LoopInvariant
import VeriTile.Examples.SoftmaxEq
import VeriTile.Examples.LogSumExpEq
import VeriTile.Examples.SoftmaxReciprocal
import VeriTile.Examples.VectorAdd
import VeriTile.Examples.FloatDType
import VeriTile.Examples.FusedSiLU
import VeriTile.Examples.RowWise
import VeriTile.Examples.WelfordKernels
import VeriTile.Examples.LayerNormKernels
import VeriTile.Examples.OnlineSoftmax
import VeriTile.Examples.HyperConnections.Manifold
import VeriTile.Examples.FlashAttention2
