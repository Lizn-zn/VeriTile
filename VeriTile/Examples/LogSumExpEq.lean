/-
VeriTile.Examples.LogSumExpEq

Worked equivalence example: direct log-sum-exp kernel vs shift-trick
log-sum-exp kernel. The math identity `log_sum_exp_shift_invariant` is the
load-bearing fact; the kernel-level theorem composes operational walk-throughs
of both kernels with this identity.

Status: math identity and kernel-level refinement are closed over the typed
tile semantics.

Source Triton (`.py` reference, single-block-per-row flavour):

```python
@triton.jit
def direct_lse_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offs)
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    y = tl.log(s)

    tl.store(y_ptr + pid, y)

@triton.jit
def stable_lse_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = m + tl.log(s)

    tl.store(y_ptr + pid, y)
```
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.Common
import VeriTile.Examples.SoftmaxEq

namespace VeriTile.Examples

open VeriTile.Triton

/-- Direct log-sum-exp kernel: y = log(Σ exp(x)). -/
def directLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e, axis=0)
  y    := tl.log(s)
  tl.store($(yReg) + pid, y)
}

/-- Shift-trick LSE kernel: y = m + log(Σ exp(x - m)) where m = max(x). -/
def stableLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := m + tl.log(s)
  tl.store($(yReg) + pid, y)
}

/-! ## Kernel-level refinement -/

/-- Read region `region` at offset `basePid` (single scalar per program_id).
    Unlike softmax which writes a tile, LSE writes one cell per pid. -/
noncomputable def observeLSE
    (sf : Option BlockState) (region : RegionName) (basePid : Nat) : Option ℝ :=
  sf.map (·.readMem region basePid)

/-- What `directLSEKernel` writes at `Y[pid]`. -/
noncomputable def directLSESpec {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  Real.log (∑ j, Real.exp (xs j))

/-- What `stableLSEKernel` writes at `Y[pid]`. -/
noncomputable def stableLSESpec {N : Nat} (xs : Fin N → ℝ) (m : ℝ) : ℝ :=
  m + Real.log (∑ j, Real.exp (xs j - m))

end VeriTile.Examples
