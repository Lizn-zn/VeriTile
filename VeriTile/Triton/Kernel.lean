/-
VeriTile.Triton.Kernel

Barrel for the shared *kernel-proof* lemmas — reusable infrastructure extracted
from the `bench/` Triton kernel transcriptions (as opposed to the core
formalization in `Core` / `Semantics` / `DSL` / `Memory`). Bench files get all of
it with a single `import VeriTile.Triton.Kernel`.
-/
import VeriTile.Triton.Kernel.EvalHelpers
import VeriTile.Triton.Kernel.OffsetInjective
import VeriTile.Triton.Kernel.Matmul
import VeriTile.Triton.Kernel.ScatterStore
import VeriTile.Triton.Kernel.LoopInvariant
