/-
VeriTile.Triton.KernelLemmas

Barrel for the shared *kernel-proof* lemmas — reusable infrastructure extracted
from the `bench/` Triton kernel transcriptions (as opposed to the core
formalization in `Core` / `Semantics` / `DSL` / `Memory`). Bench files get all of
it with a single `import VeriTile.Triton.KernelLemmas`.
-/
import VeriTile.Triton.KernelLemmas.EvalHelpers
import VeriTile.Triton.KernelLemmas.OffsetInjective
import VeriTile.Triton.KernelLemmas.Matmul
import VeriTile.Triton.KernelLemmas.ScatterStore
import VeriTile.Triton.KernelLemmas.LoopInvariant
