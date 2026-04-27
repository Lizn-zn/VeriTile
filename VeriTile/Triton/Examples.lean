/-
VeriTile.Triton.Examples

Hand-built example kernels using the P1 `Op` / `Stmt` / `Kernel` constructors.
These exist to (a) sanity-check the data types, (b) provide concrete inputs
for the operational semantics in `Semantics.lean`, and (c) serve as the first
test bed for proof attempts in P4.

Eventually these are produced by the Python lifter; for now we hand-write them.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL

namespace VeriTile.Triton.Examples

/-- Smoke test for scalar-pointer load/store syntax:
    `tl.load(ptr)` / `tl.store(ptr, value)` lower to offset `0`. -/
def scalarCopyKernel (xReg yReg : RegionName) : Kernel := triton {
  x := tl.load($(xReg))
  tl.store($(yReg), x)
}

/--
Naive softmax kernel. Triton source:

```python
@triton.jit
def naive_softmax(X, Y, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    x = tl.load(X + offs)         -- gather; modeled here as a scalar load over a forLoop
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + offs, y)
```

Notes on the embedding:
* Tile-valued offsets (gather) are not a single `Op.load` constructor. We model
  the per-element load via a `forLoop` that fills a register named `"x"` cell
  by cell. This is verbose for hand-writing but mechanical for the lifter.
* Address arithmetic uses `Op.constNat` so it stays in the Nat offset channel.
-/
def naiveSoftmax (N : Nat) : Kernel where
  inputs  := ["X"]
  outputs := ["Y"]
  body    := [
    -- pid = tl.program_id(0)
    .assign "pid" .programId,

    -- offs = pid * N + tl.arange(0, N)
    .assign "offs"
      (.add (.broadcast (.mul .programId (.constNat N)) N)
            (.arange N)),

    -- x = tl.load(X + offs)  -- gather modeled by per-element scalar loads.
    -- TODO(P1): a higher-level "tile load" constructor (`Op.gather`) once we
    -- choose how to type-check it. For now we initialize an empty tile by
    -- broadcasting 0 then overwrite per-element via a temporary scalar reg.
    .assign "x" (.broadcast (.const 0) N),
    .forLoop "i" N [
      .assign "x_i" (.load "X"
                      (.add (.mul .programId (.constNat N))
                            (.ref "i"))),
      -- TODO(P1): an `Op.tileSet` / `Stmt.tileSet` constructor to overwrite
      -- a single tile cell. Currently no such primitive; this `assign` will
      -- not compose into the tile correctly. Marked as a known gap.
      .assign "x_loaded" (.ref "x_i")
    ],

    -- m = tl.max(x, axis=0)
    .assign "m" (.reduceMax (.ref "x")),

    -- e = tl.exp(x - m)
    .assign "e" (.exp (.sub (.ref "x") (.broadcast (.ref "m") N))),

    -- s = tl.sum(e, axis=0)
    .assign "s" (.reduceSum (.ref "e")),

    -- y = e / s
    .assign "y" (.div (.ref "e") (.broadcast (.ref "s") N)),

    -- tl.store(Y + offs, y)
    .store "Y"
      (.add (.broadcast (.mul .programId (.constNat N)) N) (.arange N))
      (.ref "y")
  ]

/--
Trivial sanity-check example: `(0 + x) * 1 = x` should reduce evaluation-wise.
We don't yet provide a `simp`-able equality theorem; that lands in P1 lemma work.
-/
def trivialOp : Op := .mul (.add (.const 0) (.ref "x")) (.const 1)

end VeriTile.Triton.Examples
