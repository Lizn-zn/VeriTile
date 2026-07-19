/-
VeriTile.Triton.Semantics

Parent module for the typed operational semantics of the Triton subset.
-/

import VeriTile.Triton.Semantics.Scalar
import VeriTile.Triton.Semantics.State
import VeriTile.Triton.Semantics.Offset
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Triton.Semantics.EvalOp
import VeriTile.Triton.Semantics.Step
import VeriTile.Triton.Semantics.Eval
import VeriTile.Triton.Semantics.TiledIndexing
import VeriTile.Triton.Semantics.LaneBridge
import VeriTile.Triton.Semantics.MaskedReduction
import VeriTile.Triton.Semantics.BroadcastReshape
import VeriTile.Triton.Semantics.AtomicReduction
import VeriTile.Triton.Semantics.BlockPtrEval
import VeriTile.Triton.Semantics.StreamingAccumulator
