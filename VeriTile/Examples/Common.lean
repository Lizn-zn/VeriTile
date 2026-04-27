/-
VeriTile.Examples.Common

Shared helpers for the parameterized-region kernel correctness pattern:
* `InputLoadedAt`  — region `R` holds the tile `xs` at offsets `[pid*N, pid*N + N)`.
* `observeAt`      — read region `R`'s cell at offset `pid*N + i.val` from
  the optional final state of `exec`.

Both are parameterized by the region name (a `RegionName = String`),
matching the DSL convention that kernels take their buffer regions as
explicit `RegionName` parameters and thread them via `tl.load($(xReg) + …)`
/ `tl.store($(outReg) + …, …)` pointer-like syntax.

These supersede the hardcoded `InputLoaded` ("X") / `observeY` ("Y") used
during early Phase A, which were dropped when we banned the bare-ident
form `tl.load(X + …)` from the DSL.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics

namespace VeriTile.Examples

open VeriTile.Triton

/-- Region `region` holds tile `xs` at offsets `[pid*N, pid*N + N - 1]`. -/
def InputLoadedAt (s : BlockState) (region : RegionName)
    (N : Nat) (xs : Fin N → ℝ) : Prop :=
  ∀ i : Fin N, s.mem region (s.pid * N + i.val) = xs i

/-- Read region `region` at the cell `pid*N + i.val` from the optional
    final `BlockState` of an `exec` call. -/
noncomputable def observeAt
    (sf : Option BlockState) (region : RegionName)
    (N : Nat) (basePid : Nat) (i : Fin N) : Option ℝ :=
  sf.map (·.readMem region (basePid * N + i.val))

end VeriTile.Examples
