/-
VeriTile.Triton.Memory.KernelSpec

The **KernelIO spec surface**: a kernel's headline correctness statement is
`io ⊨ f` — "the kernel described by the IO signature `io` implements the
mathematical function `f`" — a full Hoare triple packaged as one audited
definition. Every IO signature carries `projection`, a proof that its compute
kernel projects successfully. Unsupported effects cannot enter any exact,
rounding, or equivalence contract through the legacy empty-program fallback:

* **Precondition** (all universally quantified): any program id whose window
  is in bounds; any disjoint flat allocation of the declared buffers (∀ base
  pointers — the allocator contract); any launch state whose two input
  windows hold the input arrays, with **every other allocated cell and every
  register arbitrary** (junk).
* **Postcondition**: the translated pointer kernel terminates; the output
  window holds `f` of the inputs; **every flat cell outside the output
  window is unchanged** (frame).

The per-kernel statement surface is just the `KernelIO₂` value (which buffer
is which argument, buffer lengths, the per-program window) plus `f` — pure
mathematics. `Implements` is the audit-once combinator.

The module hosts **three relations** over these IO signatures:

* **Correctness** — `io ⊨ f` (`Implements`, exact-ℝ `exec`): the kernel
  computes the mathematical function `f` on its declared windows.
* **Rounding correctness** — `io ⊨[R] f` (`ImplementsR`, rounding-model
  `execR R`): the kernel computes `f` exactly and quantizes it once at the
  declared output dtype (`outDType`) — the boundary-rounding contract. The
  output windows hold `R.round outDType (f …)` as typed cells. At
  `R := .triv` every cast is exact, so the exact surface is this relation's
  degeneration.
* **Equivalence** — `io₁ ≡[R] io₂` (`Equiv`, rounding-model `execR R`): two
  kernels sharing one IO signature make the same writes — the `⊨`-grade
  form of the refinement surface. No `f` and **no input hypotheses** (equal
  inputs are "the same launch state"); each kernel may declare private
  `scratch` working buffers, which are allocated and writable but whose
  post-state is outside every contract. Instances share the interface by
  structure update (`{ referenceIO with kernel := …, scratch := … }`); the
  relation reads the interface from its left argument.

Arity/shape variants carry the same field vocabulary throughout:
`KernelIO₂`/`₁`/`₃`/`₃ₓ₂`/`₁ₓ₂` and the masked `MaskedKernelIO₂`/`₃ₓ₂`
(lane-wise contracts via a `mask`; `MaskedKernelIO₃ₓ₂` additionally
decouples the allocation list from the argument roles so in-place updates
are expressible). Not every struct carries every relation yet — relations
are added when a showcase needs them.

**Modeling boundary (read before trusting)**: the launch state is
`A.flattenState s₀` for an arbitrary region state `s₀`. Concretely this
means: every cell of every *allocated* buffer is arbitrary (in particular
the whole output buffer and the parts of the input buffers outside the
program's window), and the register file is arbitrary (translated). What is
*not* quantified: flat addresses outside every allocated buffer read as the
default cell (unallocated memory is unobservable — the kernel is
trace-safe, so it never touches it), and the `undef` bookkeeping channel is
launch-clean. Strengthening "unallocated cells" to full junk is the
execution-locality upgrade (`Memory/Locality.lean`) and can be layered on
per kernel.

The **launch grid** is quantified on the same terms: a skin pins the
`s₀.pids` axes its windows name and nothing else, so `s₀.numPids` — the
value of `tl.num_programs` — is arbitrary unless a skin says otherwise.
Exactly one does: `StreamGridStrideEmitMasked2DKernelIO₁` pins
`numPids 0` to the grid width its windows stride by, which is what a
grid-stride loop (`pid + j·num_ctas`) needs; see that skin's genre note
for why a pid-only signature provably cannot state such a contract.
-/

-- Compatibility entry point: focused clients may import individual families.
import VeriTile.Triton.Memory.KernelSpec.Base
import VeriTile.Triton.Memory.KernelSpec.Basic
import VeriTile.Triton.Memory.KernelSpec.Masked
import VeriTile.Triton.Memory.KernelSpec.MaskedND
import VeriTile.Triton.Memory.KernelSpec.BooleanMasked
import VeriTile.Triton.Memory.KernelSpec.Metadata
import VeriTile.Triton.Memory.KernelSpec.Scatter
import VeriTile.Triton.Memory.KernelSpec.Gather
import VeriTile.Triton.Memory.KernelSpec.Grouped
import VeriTile.Triton.Memory.KernelSpec.Stream
import VeriTile.Triton.Memory.KernelSpec.StreamEmit
import VeriTile.Triton.Memory.KernelSpec.StreamAttention
import VeriTile.Triton.Memory.KernelSpec.StreamAttentionMetadata
import VeriTile.Triton.Memory.KernelSpec.TileIndex
import VeriTile.Triton.Memory.KernelSpec.Tile
import VeriTile.Triton.Memory.KernelSpec.TileND
import VeriTile.Triton.Memory.KernelSpec.TileMetadata
import VeriTile.Triton.Memory.KernelSpec.TileGather
