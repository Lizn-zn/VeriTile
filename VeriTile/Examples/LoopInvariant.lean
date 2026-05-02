/-
VeriTile.Examples.LoopInvariant

Small sanity proofs for the Triton `forLoop_inv` API.
-/

import VeriTile.Triton.LoopInvariant

namespace VeriTile.Examples.LoopInvariant

open VeriTile.Triton

/-! ### Single-loop sanity check

A trivial 1-statement-body forLoop counter. We use `Nat`-channel arithmetic
because the loop index is a typed Nat scalar.
-/

private def counterBody : List Stmt :=
  [.assign .nat [] "cnt"
    (.add NumericDType.nat Broadcast.nil
      (.ref .nat [] "cnt") (.constNat 1))]

/-! ### Nested-loop sanity check

This toy kernel checks that `forLoop_inv` composes when an outer loop body
contains another `Stmt.forLoop`. The inner loop increments a Nat counter `N`
times; the outer loop runs that inner loop `M` times. The final counter is
therefore `M * N`.
-/

private def nestedCounterInnerBody : List Stmt :=
  [.assign .nat [] "cnt"
    (.add NumericDType.nat Broadcast.nil
      (.ref .nat [] "cnt") (.constNat 1))]

private def nestedCounterOuterBody (N : Nat) : List Stmt :=
  [.forLoop "j" N nestedCounterInnerBody]

private def nestedCounterKernel (M N : Nat) : Kernel :=
  { inputs := []
  , outputs := []
  , body :=
      [.assign .nat [] "cnt" (.constNat 0),
       .forLoop "i" M (nestedCounterOuterBody N)] }

end VeriTile.Examples.LoopInvariant
