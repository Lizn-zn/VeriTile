/-
VeriTile.Kernel

Barrel for the shared *kernel-proof* lemmas — reusable infrastructure extracted
from the `bench/` Triton kernel transcriptions (as opposed to the core
formalization in `Core` / `Semantics` / `DSL` / `Memory`). Bench files get all of
it with a single `import VeriTile.Kernel`.
-/
import VeriTile.Kernel.EvalHelpers
import VeriTile.Kernel.OffsetInjective
import VeriTile.Kernel.Matmul
import VeriTile.Kernel.ScatterStore
import VeriTile.Kernel.LoopInvariant
