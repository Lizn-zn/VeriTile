-- VeriTile: default (lite) target — Triton + worked examples without the
-- analysis-heavy ApproxGeLU certificate.
--
-- ApproxGeLU and the `VeriTile.Math.*` chain (Taylor + integral bounds for
-- the GeLU error certificate) live in the separate `VeriTileMath` library
-- and are pulled in by `VeriTileFull`. Routine kernel work builds against
-- this lite root, skipping the heavy Mathlib analysis chain.

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.MemoryTyping
import VeriTile.Triton.Memory
import VeriTile.Triton.MemoryBounds
import VeriTile.Triton.MemoryFrame
import VeriTile.Triton.Launch
import VeriTile.Triton.LoopInvariant
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
