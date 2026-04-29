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
                            (.ref "i"))
                      none none),
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
      none
  ]

/--
Trivial sanity-check example: `(0 + x) * 1 = x` should reduce evaluation-wise.
We don't yet provide a `simp`-able equality theorem; that lands in P1 lemma work.
-/
def trivialOp : Op := .mul (.add (.const 0) (.ref "x")) (.const 1)

/-! ## Mask extension sanity checks (Slice 2)

These exercise the new DSL surface for masked load/store and the 6
comparison operators. They are syntactic — the test is that the kernels
elaborate to a `Kernel` value with the expected AST shape. Operational
equivalence is covered by per-kernel correctness theorems (Phase B+). -/

/-- Vector-add kernel with explicit boundary mask (the canonical
    paste-in target from real Triton tutorials).

    Triton source:
    ```python
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)
    ```
-/
def addKernelMaskedSmoke (xReg yReg outReg : RegionName)
    (blockSize nElements : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask := offs < $(nElements)
  x    := tl.load($(xReg) + offs, mask=mask, other=0)
  y    := tl.load($(yReg) + offs, mask=mask, other=0)
  out  := x + y
  tl.store($(outReg) + offs, out, mask=mask)
}

/-- Smoke test: every comparison operator is reachable via the DSL. -/
def comparisonOpsSmoke : Op :=
  let lt := .lt (.ref "a") (.ref "b")
  let le := .le (.ref "a") (.ref "b")
  let eq := .eq (.ref "a") (.ref "b")
  let gt := .gt (.ref "a") (.ref "b")
  let ge := .ge (.ref "a") (.ref "b")
  let ne := .ne (.ref "a") (.ref "b")
  -- Tuple them all into a single Op so the def itself touches each constructor.
  .add lt (.add le (.add eq (.add gt (.add ge ne))))

/-- DSL surface form of the same: bare comparisons inside a `triton { ... }`
    block lower to the corresponding `Op.lt` / `Op.le` / etc. -/
example : Op := show Op from
  -- Inside an assign RHS, comparison ops are well-formed expressions.
  -- This kernel doesn't make semantic sense (we never use any of these
  -- registers); the test is purely that DSL lowering succeeds.
  let k : Kernel := triton {
    a   := 1
    b   := 2
    r1  := a < b
    r2  := a <= b
    r3  := a == b
    r4  := a > b
    r5  := a >= b
    r6  := a != b
  }
  -- Pull out the first assigned op as a witness that the kernel built.
  match k.body.head? with
  | some (.assign _ op) => op
  | _ => .const 0

/-! ## Slice 4 sanity: `tl.load(p, mask=m)` (no `other=`) uses `s.undef`

These verify the Triton-faithful semantic where masked-off lanes get
the per-state `undef` value, not a hardcoded 0. The same load on two
states with different `undef` produces different results — this is the
non-determinism captured in our model. -/

/-- Direct-AST smoke: `evalOp` on a masked load with no `other` returns
    `s.undef` for the masked-off lane. -/
example : evalOp (.load "X" (.constNat 0)
                    (some (.lt (.constNat 0) (.constNat 0)))  -- mask = 0 < 0 = false
                    none)
          { mem := fun _ _ => 100, regs := fun _ => none
          , pid := 0, undef := fun _ _ => 42 }
          = some (Value.scalar 42) := by
  show some (Value.scalar 42) = some (Value.scalar 42)
  rfl

/-- Non-determinism: two states with same `mem` but different `undef`
    produce different results from a masked-off load. -/
example :
    let s1 : BlockState :=
      { mem := fun _ _ => 0, regs := fun _ => none
      , pid := 0, undef := fun _ _ => 42 }
    let s2 : BlockState :=
      { mem := fun _ _ => 0, regs := fun _ => none
      , pid := 0, undef := fun _ _ => 99 }
    let mask_false : Op := .lt (.constNat 0) (.constNat 0)
    evalOp (.load "X" (.constNat 0) (some mask_false) none) s1
      ≠ evalOp (.load "X" (.constNat 0) (some mask_false) none) s2 := by
  show some (Value.scalar 42) ≠ some (Value.scalar 99)
  intro h; injection h with h; injection h with h; norm_num at h

end VeriTile.Triton.Examples
