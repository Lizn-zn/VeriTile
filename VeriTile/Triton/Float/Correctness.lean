/-
VeriTile.Triton.Float.Correctness

Float-facing correctness bridge for the current real-valued Triton semantics.
-/

import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics

namespace VeriTile.Triton

namespace Kernel

/-! ## Float-facing theorem bridge -/

/-- A generic postcondition-style correctness predicate for executed kernels. -/
def Correct (k : Kernel) (post : BlockState → BlockState → Prop) : Prop :=
  ∀ s s', exec k s = some s' → post s s'

/--
Correctness policy for kernels that carry explicit `.fp32` / `.fp16` /
`.bf16` annotations in the current real-valued semantics.

The theorem is stated on the typed-float kernel `k`, but execution is routed
through `k.eraseFloat`. This lets user-facing float kernels expose float-named
correctness theorems while reusing the mathematical `.real` proof.
-/
def CorrectViaFloatErasure (k : Kernel) (post : BlockState → BlockState → Prop) : Prop :=
  Correct k.eraseFloat post

/-- Transfer a real-kernel correctness theorem to a float-kernel theorem by
showing that floating-dtype erasure produces the real kernel. -/
theorem correctViaFloatErasure_of_erase_eq {k realK : Kernel}
    {post : BlockState → BlockState → Prop}
    (herase : k.eraseFloat = realK)
    (hreal : Correct realK post) :
    CorrectViaFloatErasure k post := by
  intro s s' h
  exact hreal s s' (by simpa [CorrectViaFloatErasure, Correct, herase] using h)

end Kernel

end VeriTile.Triton
