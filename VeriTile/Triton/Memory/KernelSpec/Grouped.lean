/-
Kernel IO contracts: Grouped.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.KernelSpec.Base

namespace VeriTile.Triton



/-! ### The grouped genre: `Grouped*` skins

**Vector-channel** signatures: one program owns *many* same-length
windows, with the channel counts as **fields** (`nIn`/`nOut`), not
name subscripts. The named-field structs stop scaling exactly here —
the rotary pair writes 4–5 interleaved even/odd windows and the
spherical-harmonics forward writes 11 strided windows into one buffer;
an `₈ₓ₄`- or `₃ₓ₁₁`-style struct would need one field *and one intro
hypothesis and one value leg and one frame conjunct* per channel
(`write1..write11`, `writeMask1..writeMask11`, …), each new arity a new
struct with the whole quartet re-proved. A grouped skin instead indexes
channels by `Fin nIn`/`Fin nOut` and states each leg once, uniformly;
the consumer-facing spec `f` takes the output channel `o : Fin nOut` as
an argument. Everything else keeps the `Masked2D` family conventions:
uniform tile length `B`, two program ids, per-lane `Prop` masks, and the
`MaskedKernelIO₃ₓ₂`-style decoupled allocation list `bufs` so in-place
wiring (the rotary stores back into `Q`/`K`) stays expressible. -/

/-- IO signature of the **grouped masked 2D shape**: `nIn` float input
channels and `nOut` float output channels (channel counts are *fields*),
every channel a `B`-lane window with its own per-lane addresses and
active-lane gate, over the decoupled allocation list `bufs` (an in-place
kernel names the same buffer as an input and an output channel). Intended
consumers: the TritonBench-G rotary/spherical multi-store genre —
`rotary_emb` (`_rotary_kernel`: 8 read windows over `Q`/`K`/`Cos`/`Sin`,
4 interleaved even/odd stores in place on `Q`/`K`; also its Q-only and
K-only surfaces at `nIn = 6, nOut = 2`), `fused_rotary_embedding`'s
rotation faces and cache store slices (the paged-cache offsets enter as
`Nat` parameters of the port's proof-facing kernels, so no metadata slot
is needed here; the *full* fused surface with its in-kernel
`context_lengths`/`BLOCK_TABLES` loads would need a chained-metadata
grouped skin — a follow-up), and `fifth_order_sph_harmonics`
(`fifth_order_fwd`: 3 strided coordinate reads from one buffer, 11
strided `Y00..Y10` stores into one buffer, `nIn = 3, nOut = 11`; the
backward is `nIn = 14, nOut = 3`). -/
structure GroupedMasked2DKernelIO where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- Number of input channels (a field, never a name subscript). -/
  nIn : Nat
  /-- Number of output channels. -/
  nOut : Nat
  /-- The allocation list: every buffer the kernel touches, each exactly
  once. The channel buffers below point into this list; for an in-place
  kernel an output channel names the same buffer as an input channel. -/
  bufs : List RegionName
  /-- Input channel `i`'s buffer (channels may share a buffer — the
  spherical-harmonics `x`/`y`/`z` reads all come from one coordinate
  buffer). -/
  inp : Fin nIn → RegionName
  /-- Output channel `o`'s buffer (channels may share a buffer — the
  eleven `Y0k` stores all target one output buffer). -/
  out : Fin nOut → RegionName
  /-- Tile length: every channel owns `B`-lane windows. -/
  B : Nat
  /-- Input channel `i`'s lane-`j` read address for program
  `(pid₀, pid₁)`. -/
  read : Fin nIn → Nat → Nat → Fin B → Nat
  /-- Input channel `i`'s read-active lanes. -/
  readMask : Fin nIn → Nat → Nat → Fin B → Prop
  /-- Output channel `o`'s lane-`j` write address. -/
  write : Fin nOut → Nat → Nat → Fin B → Nat
  /-- Output channel `o`'s write-active lanes. -/
  writeMask : Fin nOut → Nat → Nat → Fin B → Prop

namespace GroupedMasked2DKernelIO

/-- `io.Implements f` — the grouped masked Hoare triple. The pinned
inputs are one function `xs : Fin nIn → Fin B → ℝ` (channel `i`, lane
`j`), pinned per channel on its read-active lanes; the spec `f` takes
the output channel as an argument and every write-active lane of every
output channel holds its value. For an in-place kernel `f` receives the
*old* contents of an updated buffer and the postcondition asserts its
*new* contents — the standard before/after reading. Frame: every cell
outside the union of all write-active output windows is untouched. -/
def Implements (io : GroupedMasked2DKernelIO)
    (f : Nat → Nat → (Fin io.nIn → Fin io.B → ℝ) → Fin io.nOut →
      Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = io.bufs →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
    (∀ (i : Fin io.nIn) (j : Fin io.B), io.readMask i pid₀ pid₁ j →
      io.read i pid₀ pid₁ j < A.extent (io.inp i)) →
    (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o pid₀ pid₁ j →
      io.write o pid₀ pid₁ j < A.extent (io.out o)) →
  ∀ (xs : Fin io.nIn → Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (i : Fin io.nIn) (j : Fin io.B), io.readMask i pid₀ pid₁ j →
      s₀.readMem (io.inp i) (io.read i pid₀ pid₁ j) = xs i j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o pid₀ pid₁ j →
          s'.readMem A.flat (A.addr (io.out o) (io.write o pid₀ pid₁ j))
            = f pid₀ pid₁ xs o j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o pid₀ pid₁ j →
              o' ≠ A.addr (io.out o) (io.write o pid₀ pid₁ j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => GroupedMasked2DKernelIO.Implements

/-- Embed into the unified core — proof plumbing for `Implements.intro`.
The embedding is (up to the membership side condition) the identity on
the channel structure: all channels are float, all arities `B`, windows
and masks ignore the pinned context. -/
private def toU (io : GroupedMasked2DKernelIO)
    (hout : ∀ o, io.out o ∈ io.bufs) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := io.nIn
  nOut := io.nOut
  nScr := 0
  bufs := io.bufs
  ity := fun _ => .float
  iarity := fun _ => io.B
  ibuf := io.inp
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := io.out
  obuf_mem := hout
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ _ j => io.read i p₀ p₁ j
  imask := fun i _ p₀ p₁ _ j => io.readMask i p₀ p₁ j
  owin := fun o _ p₀ p₁ _ j => io.write o p₀ p₁ j
  omask := fun o _ p₀ p₁ _ j => io.writeMask o p₀ p₁ j
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- Assembly lemma — grouped sibling of
`Masked2DKernelIO₃ₓ₃.Implements.intro`, with the channels indexed by
`Fin nIn`/`Fin nOut` instead of named fields, plus the membership side
condition `hout` tying every output channel's buffer into the declared
allocation list. `hrun`'s frame takes one exclusion condition
quantified over all output channels. -/
theorem Implements.intro (io : GroupedMasked2DKernelIO)
    {f : Nat → Nat → (Fin io.nIn → Fin io.B → ℝ) → Fin io.nOut →
      Fin io.B → ℝ}
    (hout : ∀ o, io.out o ∈ io.bufs)
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ (i : Fin io.nIn) (j : Fin io.B),
        io.readMask i (s.pids 0) (s.pids 1) j →
        io.read i (s.pids 0) (s.pids 1) j < bounds (io.inp i)) →
      (∀ (o : Fin io.nOut) (j : Fin io.B),
        io.writeMask o (s.pids 0) (s.pids 1) j →
        io.write o (s.pids 0) (s.pids 1) j < bounds (io.out o)) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.nIn → Fin io.B → ℝ),
      (∀ (i : Fin io.nIn) (j : Fin io.B),
        io.readMask i (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem (io.inp i) (io.read i (s₀.pids 0) (s₀.pids 1) j)
          = xs i j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ (o : Fin io.nOut) (j : Fin io.B),
            io.writeMask o (s₀.pids 0) (s₀.pids 1) j →
            s1.readMem (io.out o) (io.write o (s₀.pids 0) (s₀.pids 1) j)
              = f (s₀.pids 0) (s₀.pids 1) xs o j)
        ∧ (∀ r o',
            (∀ (oc : Fin io.nOut) (j : Fin io.B),
              io.writeMask oc (s₀.pids 0) (s₀.pids 1) j →
              r ≠ io.out oc ∨ o' ≠ io.write oc (s₀.pids 0) (s₀.pids 1) j) →
            s1.mem r o' = s₀.mem r o')) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`. The embedding being channel-identical
  -- makes both conversions eta-expansions — no `Fin`-literal plumbing.
  have hcore : (io.toU hout).Implements
      (fun p₀ p₁ vals o j => f p₀ p₁ (fun i j' => vals i j') o j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      exact hts bounds s (fun i j hj => hib i j hj)
        (fun o j hj => hob o j hj)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun i j => vals i j) (fun i j hj => hpins i j hj)
      refine ⟨s1, hexec, fun o j hj => hval o j hj, ?_⟩
      intro r o' hoc _hsc
      exact hframe r o' (fun oc j hj => hoc oc j hj)
  intro A hd hregs hcov pid₀ pid₁ hbr hbw xs s₀ hpid₀ hpid₁ hu hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2) (fun i j => xs i j) s₀ hpid₀ hpid₁ rfl hu
      (fun i j hj => hbr i j hj)
      (fun o j hj => hbw o j hj)
      (fun t => t.elim0)
      (fun i j hj => hx i j hj)
  refine ⟨s', hexec, fun o j hj => hval o j hj, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | hn
  · exact Or.inl hflat
  · exact Or.inr ⟨fun o j hj => hn o j hj, fun t => t.elim0⟩

/-- `io.ImplementsR R outDType f` — the **rounding-correctness** relation
for the grouped masked family, written `io ⊨[R, outDType] f`. Verbatim
`Implements`, with two changes: the kernel runs under the rounding model
(`execR R`), and each **write-active** lane of each output channel is read
back as an `outDType o`-typed cell holding the ideal real value quantized
**once**, `(outDType o).ofReal (R.round (outDType o) (f pid₀ pid₁ xs o j))`.
Inputs stay exact ℝ — the rounding model acts at the kernel's cast/store
sites, not at loads. Everything else (the allocation contract, the two
in-bounds obligations, the `pid`/`undef` pins, the frame) is unchanged; at
`outDType := fun _ => .real` the stores are exact and this degenerates to
the exact surface.

The grid is an **argument**, and — this skin being channel-indexed — a
**per-channel** one, `Fin nOut → FloatDType`: the rotary genre
(`rotary_emb`, `rope_transform`) stores back in place into half-precision
`Q`/`K`, while a grouped kernel may in the same launch write a wide
statistic channel, so one scalar grid for all `nOut` channels would be
the wrong shape, not merely a coarse one. The notation therefore keeps
four holes and names the vector: hiding it behind a three-hole
`io ⊨[R] f` would print two different per-channel quantization
assignments identically. -/
def ImplementsR (io : GroupedMasked2DKernelIO) (R : RoundingModel)
    (outDType : Fin io.nOut → FloatDType)
    (f : Nat → Nat → (Fin io.nIn → Fin io.B → ℝ) → Fin io.nOut →
      Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = io.bufs →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
    (∀ (i : Fin io.nIn) (j : Fin io.B), io.readMask i pid₀ pid₁ j →
      io.read i pid₀ pid₁ j < A.extent (io.inp i)) →
    (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o pid₀ pid₁ j →
      io.write o pid₀ pid₁ j < A.extent (io.out o)) →
  ∀ (xs : Fin io.nIn → Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (i : Fin io.nIn) (j : Fin io.B), io.readMask i pid₀ pid₁ j →
      s₀.readMem (io.inp i) (io.read i pid₀ pid₁ j) = xs i j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o pid₀ pid₁ j →
          s'.readMemAs (outDType o) A.flat
              (A.addr (io.out o) (io.write o pid₀ pid₁ j))
            = (outDType o).ofReal
                (R.round (outDType o) (f pid₀ pid₁ xs o j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o pid₀ pid₁ j →
              o' ≠ A.addr (io.out o) (io.write o pid₀ pid₁ j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R ", " outDType "] " f =>
  GroupedMasked2DKernelIO.ImplementsR io R outDType f

/-- Assembly lemma for `⊨[R, outDType]` — the rounding sibling of
`GroupedMasked2DKernelIO.Implements.intro`, riding the single-shot
family's rounding core `UKernelIO.ImplementsR.intro` through the same
`toU` embedding (at the per-channel output grid `outDType`). Obligations
as there — the membership side condition `hout`, `FlattenOk`, the safety
walk at `Kernel.TraceSafeR R`, and `hrun` returning a rounded region-model
triple: termination under `execR R`, the `readMemAs (outDType o)` per-lane
readback, and the frame. -/
theorem ImplementsR.intro (io : GroupedMasked2DKernelIO) {R : RoundingModel}
    {outDType : Fin io.nOut → FloatDType}
    {f : Nat → Nat → (Fin io.nIn → Fin io.B → ℝ) → Fin io.nOut →
      Fin io.B → ℝ}
    (hout : ∀ o, io.out o ∈ io.bufs)
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ (i : Fin io.nIn) (j : Fin io.B),
        io.readMask i (s.pids 0) (s.pids 1) j →
        io.read i (s.pids 0) (s.pids 1) j < bounds (io.inp i)) →
      (∀ (o : Fin io.nOut) (j : Fin io.B),
        io.writeMask o (s.pids 0) (s.pids 1) j →
        io.write o (s.pids 0) (s.pids 1) j < bounds (io.out o)) →
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.nIn → Fin io.B → ℝ),
      (∀ (i : Fin io.nIn) (j : Fin io.B),
        io.readMask i (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem (io.inp i) (io.read i (s₀.pids 0) (s₀.pids 1) j)
          = xs i j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀
          = some s1
        ∧ (∀ (o : Fin io.nOut) (j : Fin io.B),
            io.writeMask o (s₀.pids 0) (s₀.pids 1) j →
            s1.readMemAs (outDType o) (io.out o)
                (io.write o (s₀.pids 0) (s₀.pids 1) j)
              = (outDType o).ofReal
                  (R.round (outDType o)
                    (f (s₀.pids 0) (s₀.pids 1) xs o j)))
        ∧ (∀ r o',
            (∀ (oc : Fin io.nOut) (j : Fin io.B),
              io.writeMask oc (s₀.pids 0) (s₀.pids 1) j →
              r ≠ io.out oc ∨ o' ≠ io.write oc (s₀.pids 0) (s₀.pids 1) j) →
            s1.mem r o' = s₀.mem r o')) :
    io.ImplementsR R outDType f := by
  -- assemble the unified-core rounded triple once, then convert it back
  -- into the family statement; the flattening bridge lives in
  -- `UKernelIO.ImplementsR.intro`. The embedding being channel-identical
  -- makes both conversions eta-expansions — no `Fin`-literal plumbing.
  have hcore : (io.toU hout).ImplementsR R outDType
      (fun p₀ p₁ vals o j => f p₀ p₁ (fun i j' => vals i j') o j) := by
    refine UKernelIO.ImplementsR.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      exact hts bounds s (fun i j hj => hib i j hj)
        (fun o j hj => hob o j hj)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun i j => vals i j) (fun i j hj => hpins i j hj)
      refine ⟨s1, hexec, fun o j hj => hval o j hj, ?_⟩
      intro r o' hoc _hsc
      exact hframe r o' (fun oc j hj => hoc oc j hj)
  intro A hd hregs hcov pid₀ pid₁ hbr hbw xs s₀ hpid₀ hpid₁ hu hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2) (fun i j => xs i j) s₀ hpid₀ hpid₁ rfl hu
      (fun i j hj => hbr i j hj)
      (fun o j hj => hbw o j hj)
      (fun t => t.elim0)
      (fun i j hj => hx i j hj)
  refine ⟨s', hexec, fun o j hj => hval o j hj, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | hn
  · exact Or.inl hflat
  · exact Or.inr ⟨fun o j hj => hn o j hj, fun t => t.elim0⟩

end GroupedMasked2DKernelIO

/-- IO signature of the **metadata-grouped multi-store shape**: the vectorized
`GroupedMasked2DKernelIO` channel structure (`nIn` input / `nOut` output
channels are *fields*, every channel a `B`-lane window with its own per-lane
addresses and active-lane gate, over the decoupled allocation list `bufs`)
**plus two per-program `.nat` scalar slots** (`mbuf1`/`mwin1` and
`mbuf2`/`mwin2`), whose loaded values `s1`/`s2` are ordinary named arguments to
*every* channel window, *every* channel mask, and the spec `f`. This is the
grouped sibling of the `Meta*` genre: `GroupedMasked2DKernelIO` has vector
channels but no slot, the fixed-arity `Meta*` skins have slots but a fixed
channel count — this skin has both. Intended consumers: the TritonBench-G
varlen rotary pair (`rotary_transform` / `rotary_transform_ops` —
`rotary_kernel`'s `IS_VARLEN` branch loads `start_idx = tl.load(CU_SEQLENS +
pid_batch)` and `tl.load(CU_SEQLENS + pid_batch + 1)`, whose difference is the
per-program `seqlen`; `start_idx` shifts the `X`/`OUT` base offsets — it enters
the read/write *windows* — while `seqlen` gates every `rm < seqlen` load/store
*mask*). The two loaded cells are modelled as the two slots `s1`/`s2`, so the
consumer forms `seqlen := s2 - s1` inside its windows/masks/`f`. The slot
design is deliberately general (slots feed windows *and* masks *and* `f`) so
this skin can later host `fused_rotary`'s in-kernel `context_lengths` /
paged-KV metadata faces. Following the family precedent there is no `scratch`
field until a consumer needs one. -/
structure MetaGroupedMasked2DKernelIO where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- Number of input channels (a field, never a name subscript). -/
  nIn : Nat
  /-- Number of output channels. -/
  nOut : Nat
  /-- The allocation list: every buffer the kernel touches, each exactly
  once. The channel buffers below point into this list; for an in-place
  kernel an output channel names the same buffer as an input channel. -/
  bufs : List RegionName
  /-- First slot's buffer (one `.nat` cell per program). -/
  mbuf1 : RegionName
  /-- Second slot's buffer (one `.nat` cell per program). The rotary pair
  points both at the shared `CU_SEQLENS` buffer at adjacent cells. -/
  mbuf2 : RegionName
  /-- Input channel `i`'s buffer (channels may share a buffer). -/
  inp : Fin nIn → RegionName
  /-- Output channel `o`'s buffer (channels may share a buffer — an in-place
  channel names an input's buffer). -/
  out : Fin nOut → RegionName
  /-- Tile length: every channel owns `B`-lane windows. -/
  B : Nat
  /-- First slot's cell address for program `(pid₀, pid₁)`. -/
  mwin1 : Nat → Nat → Nat
  /-- Second slot's cell address. -/
  mwin2 : Nat → Nat → Nat
  /-- Input channel `i`'s lane-`j` read address at `(pid₀, pid₁, s1, s2, j)` —
  the two loaded slots are ordinary named arguments. -/
  read : Fin nIn → Nat → Nat → Nat → Nat → Fin B → Nat
  /-- Input channel `i`'s read-active lanes, given the loaded slots. -/
  readMask : Fin nIn → Nat → Nat → Nat → Nat → Fin B → Prop
  /-- Output channel `o`'s lane-`j` write address, given the loaded slots. -/
  write : Fin nOut → Nat → Nat → Nat → Nat → Fin B → Nat
  /-- Output channel `o`'s write-active lanes, given the loaded slots. -/
  writeMask : Fin nOut → Nat → Nat → Nat → Nat → Fin B → Prop

namespace MetaGroupedMasked2DKernelIO

/-- `io.Implements f` — the metadata-grouped masked Hoare triple. The two slot
values `s1`/`s2` are universally quantified and pinned to their slot cells; the
grouped inputs are one function `xs : Fin nIn → Fin B → ℝ`, pinned per channel
on its read-active lanes (windows/masks all eating the loaded slots). The spec
`f` takes the two slots and the output channel as arguments and every
write-active lane of every output channel holds its value. For an in-place
kernel `f` receives the *old* contents of an updated buffer and the
postcondition asserts its *new* contents. Frame: every cell outside the union
of all write-active output windows is untouched. -/
def Implements (io : MetaGroupedMasked2DKernelIO)
    (f : Nat → Nat → Nat → Nat → (Fin io.nIn → Fin io.B → ℝ) → Fin io.nOut →
      Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = io.bufs →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (s1 s2 : Nat) (xs : Fin io.nIn → Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    io.mwin1 pid₀ pid₁ < A.extent io.mbuf1 →
    io.mwin2 pid₀ pid₁ < A.extent io.mbuf2 →
    (∀ (i : Fin io.nIn) (j : Fin io.B), io.readMask i pid₀ pid₁ s1 s2 j →
      io.read i pid₀ pid₁ s1 s2 j < A.extent (io.inp i)) →
    (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o pid₀ pid₁ s1 s2 j →
      io.write o pid₀ pid₁ s1 s2 j < A.extent (io.out o)) →
    s₀.readMemValue .nat io.mbuf1 (io.mwin1 pid₀ pid₁) = s1 →
    s₀.readMemValue .nat io.mbuf2 (io.mwin2 pid₀ pid₁) = s2 →
    (∀ (i : Fin io.nIn) (j : Fin io.B), io.readMask i pid₀ pid₁ s1 s2 j →
      s₀.readMem (io.inp i) (io.read i pid₀ pid₁ s1 s2 j) = xs i j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o pid₀ pid₁ s1 s2 j →
          s'.readMem A.flat (A.addr (io.out o) (io.write o pid₀ pid₁ s1 s2 j))
            = f pid₀ pid₁ s1 s2 xs o j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o pid₀ pid₁ s1 s2 j →
              o' ≠ A.addr (io.out o) (io.write o pid₀ pid₁ s1 s2 j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => MetaGroupedMasked2DKernelIO.Implements

/-- Embed into the unified core: channels 0/1 are the two 1-lane `.nat` slots
(always read), channels `k+2` the `nIn` float grouped data channels whose
windows/masks read the two slots' pinned values; the membership side condition
`hout` ties every output channel's buffer into the declared allocation list. -/
private def toU (io : MetaGroupedMasked2DKernelIO)
    (hout : ∀ o, io.out o ∈ io.bufs) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := io.nIn + 2
  nOut := io.nOut
  nScr := 0
  bufs := io.bufs
  ity := fun i => match i with
    | ⟨0, _⟩ => .nat
    | ⟨1, _⟩ => .nat
    | ⟨_+2, _⟩ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => 1
    | ⟨_+2, _⟩ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.mbuf1
    | ⟨1, _⟩ => io.mbuf2
    | ⟨k+2, h⟩ => io.inp ⟨k, by omega⟩
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := io.out
  obuf_mem := hout
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => io.mwin1 p₀ p₁
    | ⟨1, _⟩ => fun _ => io.mwin2 p₀ p₁
    | ⟨k+2, h⟩ => fun j => io.read ⟨k, by omega⟩ p₀ p₁
        (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1)) j
  imask := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => True
    | ⟨k+2, h⟩ => fun j => io.readMask ⟨k, by omega⟩ p₀ p₁
        (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1)) j
  owin := fun o vals p₀ p₁ _ j => io.write o p₀ p₁
      (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
      (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1)) j
  omask := fun o vals p₀ p₁ _ j => io.writeMask o p₀ p₁
      (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
      (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1)) j
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- Assembly lemma — metadata-slot sibling of
`GroupedMasked2DKernelIO.Implements.intro`, with two pinned named `.nat`
scalars threaded through `hts`/`hrun` (the value-dependent windows need them to
state their bounds) and the same membership side condition `hout`. `hrun`'s
frame takes one exclusion condition quantified over all output channels. -/
theorem Implements.intro (io : MetaGroupedMasked2DKernelIO)
    {f : Nat → Nat → Nat → Nat → (Fin io.nIn → Fin io.B → ℝ) → Fin io.nOut →
      Fin io.B → ℝ}
    (hout : ∀ o, io.out o ∈ io.bufs)
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (s1 s2 : Nat),
      s.readMemValue .nat io.mbuf1 (io.mwin1 (s.pids 0) (s.pids 1)) = s1 →
      s.readMemValue .nat io.mbuf2 (io.mwin2 (s.pids 0) (s.pids 1)) = s2 →
      io.mwin1 (s.pids 0) (s.pids 1) < bounds io.mbuf1 →
      io.mwin2 (s.pids 0) (s.pids 1) < bounds io.mbuf2 →
      (∀ (i : Fin io.nIn) (j : Fin io.B),
        io.readMask i (s.pids 0) (s.pids 1) s1 s2 j →
        io.read i (s.pids 0) (s.pids 1) s1 s2 j < bounds (io.inp i)) →
      (∀ (o : Fin io.nOut) (j : Fin io.B),
        io.writeMask o (s.pids 0) (s.pids 1) s1 s2 j →
        io.write o (s.pids 0) (s.pids 1) s1 s2 j < bounds (io.out o)) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (s1 s2 : Nat) (xs : Fin io.nIn → Fin io.B → ℝ),
      s₀.readMemValue .nat io.mbuf1 (io.mwin1 (s₀.pids 0) (s₀.pids 1)) = s1 →
      s₀.readMemValue .nat io.mbuf2 (io.mwin2 (s₀.pids 0) (s₀.pids 1)) = s2 →
      (∀ (i : Fin io.nIn) (j : Fin io.B),
        io.readMask i (s₀.pids 0) (s₀.pids 1) s1 s2 j →
        s₀.readMem (io.inp i) (io.read i (s₀.pids 0) (s₀.pids 1) s1 s2 j)
          = xs i j) →
      ∃ s1', exec (io.kernel.toAlgKernel) s₀ = some s1'
        ∧ (∀ (o : Fin io.nOut) (j : Fin io.B),
            io.writeMask o (s₀.pids 0) (s₀.pids 1) s1 s2 j →
            s1'.readMem (io.out o) (io.write o (s₀.pids 0) (s₀.pids 1) s1 s2 j)
              = f (s₀.pids 0) (s₀.pids 1) s1 s2 xs o j)
        ∧ (∀ r o',
            (∀ (oc : Fin io.nOut) (j : Fin io.B),
              io.writeMask oc (s₀.pids 0) (s₀.pids 1) s1 s2 j →
              r ≠ io.out oc ∨ o' ≠ io.write oc (s₀.pids 0) (s₀.pids 1) s1 s2 j) →
            s1'.mem r o' = s₀.mem r o')) :
    io.Implements f := by
  have hcore : (io.toU hout).Implements
      (fun p₀ p₁ vals o j => f p₀ p₁
        (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (fun i j' => vals (⟨i.val + 2, by omega⟩ : Fin (io.nIn+2)) j') o j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib hob _hsb
      exact hts bounds s
        (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (hpins (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun i j hj => hib (⟨i.val + 2, by omega⟩ : Fin (io.nIn+2)) j hj)
        (fun o j hj => hob o j hj)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1', hexec, hval, hframe⟩ :=
        hrun s₀ (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
          (fun i j' => vals (⟨i.val + 2, by omega⟩ : Fin (io.nIn+2)) j')
          (hpins (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
          (hpins (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
          (fun i j hj => hpins (⟨i.val + 2, by omega⟩ : Fin (io.nIn+2)) j hj)
      refine ⟨s1', hexec, fun o j hj => hval o j hj, ?_⟩
      intro r o' hoc _hsc
      exact hframe r o' (fun oc j hj => hoc oc j hj)
  intro A hd hregs hcov pid₀ pid₁ s1 s2 xs s₀ hpid₀ hpid₁ hu hb1 hb2 hbr hbw
    hm1 hm2 hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ => s1
        | ⟨1, _⟩ => fun _ => s2
        | ⟨k+2, h⟩ => xs ⟨k, by have h2 : k + 2 < io.nIn + 2 := h; omega⟩)
      s₀ hpid₀ hpid₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hb1
        | ⟨1, _⟩ => fun _ _ => hb2
        | ⟨k+2, h⟩ => fun j hj =>
            hbr ⟨k, by have h2 : k + 2 < io.nIn + 2 := h; omega⟩ j hj)
      (fun o j hj => hbw o j hj)
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hm1
        | ⟨1, _⟩ => fun _ _ => hm2
        | ⟨k+2, h⟩ => fun j hj =>
            hx ⟨k, by have h2 : k + 2 < io.nIn + 2 := h; omega⟩ j hj)
  refine ⟨s', hexec, fun o j hj => hval o j hj, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | hn
  · exact Or.inl hflat
  · exact Or.inr ⟨fun o j hj => hn o j hj, fun t => t.elim0⟩

/-- `io.ImplementsR R outDType f` — the **rounding-correctness** relation
for the metadata-slot grouped masked family, written
`io ⊨[R, outDType] f`. Verbatim `Implements`, with two changes: the kernel
runs under the rounding model (`execR R`), and each **write-active** lane of
each output channel is read back as an `outDType o`-typed cell holding the
ideal real value quantized **once**,
`(outDType o).ofReal (R.round (outDType o) (f pid₀ pid₁ s1 s2 xs o j))`. The
two `.nat` slots stay exact — they are *input* channels, read through
`readMemValue .nat`, and no rounding grid applies to an integer load — and
so do the grouped float channels; the rounding model acts at the kernel's
cast/store sites, not at loads. Everything else (the allocation contract,
the slot and window in-bounds obligations, the slot pins, the `pid`/`undef`
pins, the frame) is unchanged; at `outDType := fun _ => .real` the stores
are exact and this degenerates to the exact surface.

The grid is an **argument**, and — this skin being channel-indexed — a
**per-channel** one, `Fin nOut → FloatDType`, exactly as in
`GroupedMasked2DKernelIO`: the varlen rotary genre (`rotary_transform`,
`rotary_transform_ops`, `rotary_emb_nopad`, `fused_rotary_embedding`) writes
back in place into half-precision `Q`/`K` channels, and a grouped kernel may
in the same launch write a wide statistic channel, so one scalar grid for
all `nOut` channels would be the wrong shape, not merely a coarse one. The
notation therefore keeps four holes and names the vector: hiding it behind a
three-hole `io ⊨[R] f` would print two different per-channel quantization
assignments identically. -/
def ImplementsR (io : MetaGroupedMasked2DKernelIO) (R : RoundingModel)
    (outDType : Fin io.nOut → FloatDType)
    (f : Nat → Nat → Nat → Nat → (Fin io.nIn → Fin io.B → ℝ) → Fin io.nOut →
      Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = io.bufs →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (s1 s2 : Nat) (xs : Fin io.nIn → Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    io.mwin1 pid₀ pid₁ < A.extent io.mbuf1 →
    io.mwin2 pid₀ pid₁ < A.extent io.mbuf2 →
    (∀ (i : Fin io.nIn) (j : Fin io.B), io.readMask i pid₀ pid₁ s1 s2 j →
      io.read i pid₀ pid₁ s1 s2 j < A.extent (io.inp i)) →
    (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o pid₀ pid₁ s1 s2 j →
      io.write o pid₀ pid₁ s1 s2 j < A.extent (io.out o)) →
    s₀.readMemValue .nat io.mbuf1 (io.mwin1 pid₀ pid₁) = s1 →
    s₀.readMemValue .nat io.mbuf2 (io.mwin2 pid₀ pid₁) = s2 →
    (∀ (i : Fin io.nIn) (j : Fin io.B), io.readMask i pid₀ pid₁ s1 s2 j →
      s₀.readMem (io.inp i) (io.read i pid₀ pid₁ s1 s2 j) = xs i j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o pid₀ pid₁ s1 s2 j →
          s'.readMemAs (outDType o) A.flat
              (A.addr (io.out o) (io.write o pid₀ pid₁ s1 s2 j))
            = (outDType o).ofReal
                (R.round (outDType o) (f pid₀ pid₁ s1 s2 xs o j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o pid₀ pid₁ s1 s2 j →
              o' ≠ A.addr (io.out o) (io.write o pid₀ pid₁ s1 s2 j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R ", " outDType "] " f =>
  MetaGroupedMasked2DKernelIO.ImplementsR io R outDType f

/-- Assembly lemma for `⊨[R, outDType]` — the rounding sibling of
`MetaGroupedMasked2DKernelIO.Implements.intro`, riding the single-shot
family's rounding core `UKernelIO.ImplementsR.intro` through the same `toU`
embedding (at the per-channel output grid `outDType`). Obligations as
there — the membership side condition `hout`, `FlattenOk`, the two pinned
named `.nat` scalars threaded through `hts`/`hrun` — with the safety walk at
`Kernel.TraceSafeR R` and `hrun` returning a rounded region-model triple:
termination under `execR R`, the `readMemAs (outDType o)` per-lane readback,
and the frame. -/
theorem ImplementsR.intro (io : MetaGroupedMasked2DKernelIO)
    {R : RoundingModel}
    {outDType : Fin io.nOut → FloatDType}
    {f : Nat → Nat → Nat → Nat → (Fin io.nIn → Fin io.B → ℝ) → Fin io.nOut →
      Fin io.B → ℝ}
    (hout : ∀ o, io.out o ∈ io.bufs)
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (s1 s2 : Nat),
      s.readMemValue .nat io.mbuf1 (io.mwin1 (s.pids 0) (s.pids 1)) = s1 →
      s.readMemValue .nat io.mbuf2 (io.mwin2 (s.pids 0) (s.pids 1)) = s2 →
      io.mwin1 (s.pids 0) (s.pids 1) < bounds io.mbuf1 →
      io.mwin2 (s.pids 0) (s.pids 1) < bounds io.mbuf2 →
      (∀ (i : Fin io.nIn) (j : Fin io.B),
        io.readMask i (s.pids 0) (s.pids 1) s1 s2 j →
        io.read i (s.pids 0) (s.pids 1) s1 s2 j < bounds (io.inp i)) →
      (∀ (o : Fin io.nOut) (j : Fin io.B),
        io.writeMask o (s.pids 0) (s.pids 1) s1 s2 j →
        io.write o (s.pids 0) (s.pids 1) s1 s2 j < bounds (io.out o)) →
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (s1 s2 : Nat) (xs : Fin io.nIn → Fin io.B → ℝ),
      s₀.readMemValue .nat io.mbuf1 (io.mwin1 (s₀.pids 0) (s₀.pids 1)) = s1 →
      s₀.readMemValue .nat io.mbuf2 (io.mwin2 (s₀.pids 0) (s₀.pids 1)) = s2 →
      (∀ (i : Fin io.nIn) (j : Fin io.B),
        io.readMask i (s₀.pids 0) (s₀.pids 1) s1 s2 j →
        s₀.readMem (io.inp i) (io.read i (s₀.pids 0) (s₀.pids 1) s1 s2 j)
          = xs i j) →
      ∃ s1', execR R (io.kernel.toAlgKernel) s₀ = some s1'
        ∧ (∀ (o : Fin io.nOut) (j : Fin io.B),
            io.writeMask o (s₀.pids 0) (s₀.pids 1) s1 s2 j →
            s1'.readMemAs (outDType o) (io.out o)
                (io.write o (s₀.pids 0) (s₀.pids 1) s1 s2 j)
              = (outDType o).ofReal
                  (R.round (outDType o)
                    (f (s₀.pids 0) (s₀.pids 1) s1 s2 xs o j)))
        ∧ (∀ r o',
            (∀ (oc : Fin io.nOut) (j : Fin io.B),
              io.writeMask oc (s₀.pids 0) (s₀.pids 1) s1 s2 j →
              r ≠ io.out oc ∨ o' ≠ io.write oc (s₀.pids 0) (s₀.pids 1) s1 s2 j) →
            s1'.mem r o' = s₀.mem r o')) :
    io.ImplementsR R outDType f := by
  -- assemble the unified-core rounded triple once, then convert it back
  -- into the family statement; the flattening bridge lives in
  -- `UKernelIO.ImplementsR.intro`
  have hcore : (io.toU hout).ImplementsR R outDType
      (fun p₀ p₁ vals o j => f p₀ p₁
        (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (fun i j' => vals (⟨i.val + 2, by omega⟩ : Fin (io.nIn+2)) j') o j) := by
    refine UKernelIO.ImplementsR.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib hob _hsb
      exact hts bounds s
        (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (hpins (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun i j hj => hib (⟨i.val + 2, by omega⟩ : Fin (io.nIn+2)) j hj)
        (fun o j hj => hob o j hj)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1', hexec, hval, hframe⟩ :=
        hrun s₀ (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
          (fun i j' => vals (⟨i.val + 2, by omega⟩ : Fin (io.nIn+2)) j')
          (hpins (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
          (hpins (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
          (fun i j hj => hpins (⟨i.val + 2, by omega⟩ : Fin (io.nIn+2)) j hj)
      refine ⟨s1', hexec, fun o j hj => hval o j hj, ?_⟩
      intro r o' hoc _hsc
      exact hframe r o' (fun oc j hj => hoc oc j hj)
  intro A hd hregs hcov pid₀ pid₁ s1 s2 xs s₀ hpid₀ hpid₁ hu hb1 hb2 hbr hbw
    hm1 hm2 hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ => s1
        | ⟨1, _⟩ => fun _ => s2
        | ⟨k+2, h⟩ => xs ⟨k, by have h2 : k + 2 < io.nIn + 2 := h; omega⟩)
      s₀ hpid₀ hpid₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hb1
        | ⟨1, _⟩ => fun _ _ => hb2
        | ⟨k+2, h⟩ => fun j hj =>
            hbr ⟨k, by have h2 : k + 2 < io.nIn + 2 := h; omega⟩ j hj)
      (fun o j hj => hbw o j hj)
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hm1
        | ⟨1, _⟩ => fun _ _ => hm2
        | ⟨k+2, h⟩ => fun j hj =>
            hx ⟨k, by have h2 : k + 2 < io.nIn + 2 := h; omega⟩ j hj)
  refine ⟨s', hexec, fun o j hj => hval o j hj, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | hn
  · exact Or.inl hflat
  · exact Or.inr ⟨fun o j hj => hn o j hj, fun t => t.elim0⟩

end MetaGroupedMasked2DKernelIO

/-- IO signature of the **chained-metadata grouped multi-store shape**: the
vectorized `MetaGroupedMasked2DKernelIO` channel structure (`nIn` input / `nOut`
output channels are *fields*, every channel a `B`-lane window with its own
per-lane addresses and active-lane gate, over the decoupled allocation list
`bufs`, plus two per-program `.nat` scalar slots feeding *every* window, *every*
mask, and the spec `f`) with **one change**: the two slots are **chained** —
the second slot's cell address `mwin2 : Nat → Nat → Nat → Nat` eats the first
slot's loaded value `m₁` (exactly `ChainMetaMasked2DKernelIO₂ₓ₂`'s chaining).
The grouped sibling of the `ChainMeta*` genre: `MetaGroupedMasked2DKernelIO` has
vector channels but *independent* slots; this skin has vector channels and the
paged-KV `block_table[f(context_length)]` chain.

Intended consumers: the TritonBench-G paged-KV rotary caches whose store cell is
gathered through a two-hop `context_lengths → BLOCK_TABLES` load. In
`rotary_emb_nopad`'s `fused_rotary_embedding_kernel_v2` cache face the chain is
`m₁ = context_lengths[token]` then `m₂ = BLOCK_TABLES[token·bts_stride +
((m₁−1)/block_size)·btb_stride]`; four float channels (`k0`/`k1`/`cos`/`sin`)
feed two chained KV-cache stores whose windows eat both `m₁` (the
`(m₁−1) % block_size` in-block offset) and `m₂` (the block id). It also redeems
`fused_rotary_embedding`'s store slices, which currently pin `block_id` as a host
`Nat` — this skin loads it in-kernel through the chained second slot. Following
the family precedent there is no `scratch` field until a consumer needs one.

**Injectivity design.** Because the block-id-dependent cache windows may alias
(different tokens sharing a block), each output channel's readback leg is guarded
by its *per-pinned-context* `WriteInj o` antecedent (the consumers' cache
`hOutInj` side conditions), exactly as in `ChainMetaMasked2DKernelIO₂ₓ₂` — never
an `Implements.intro` hypothesis, which would demand no-aliasing for arbitrary
block-table contents. In the core embedding `nOut` scratch channels shadow the
raw (ungated) write cells, carrying the unconditional frame exclusions and
trace-safety write bounds. -/
structure ChainMetaGroupedMasked2DKernelIO where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- Number of input channels (a field, never a name subscript). -/
  nIn : Nat
  /-- Number of output channels. -/
  nOut : Nat
  /-- The allocation list: every buffer the kernel touches, each exactly once.
  The channel buffers below point into this list; for an in-place kernel an
  output channel names the same buffer as an input channel. -/
  bufs : List RegionName
  /-- First slot's buffer (one `.nat` cell per program — the context-length
  array). -/
  mbuf1 : RegionName
  /-- Second slot's buffer (one `.nat` cell per program — the block table); its
  cell address depends on the first slot's loaded value. -/
  mbuf2 : RegionName
  /-- Input channel `i`'s buffer (channels may share a buffer). -/
  inp : Fin nIn → RegionName
  /-- Output channel `o`'s buffer (channels may share a buffer — an in-place
  channel names an input's buffer). -/
  out : Fin nOut → RegionName
  /-- Tile length: every channel owns `B`-lane windows. -/
  B : Nat
  /-- First slot's cell address for program `(pid₀, pid₁)`. -/
  mwin1 : Nat → Nat → Nat
  /-- Second slot's cell address at `(pid₀, pid₁, m₁)` — **chained**: it eats
  the first slot's loaded value. -/
  mwin2 : Nat → Nat → Nat → Nat
  /-- Input channel `i`'s lane-`j` read address at `(pid₀, pid₁, m₁, m₂, j)` —
  the two loaded slots are ordinary named arguments. -/
  read : Fin nIn → Nat → Nat → Nat → Nat → Fin B → Nat
  /-- Input channel `i`'s read-active lanes, given the loaded slots. -/
  readMask : Fin nIn → Nat → Nat → Nat → Nat → Fin B → Prop
  /-- Output channel `o`'s lane-`j` write address, given the loaded slots (the
  block-id-dependent cache cell). -/
  write : Fin nOut → Nat → Nat → Nat → Nat → Fin B → Nat
  /-- Output channel `o`'s write-active lanes, given the loaded slots. -/
  writeMask : Fin nOut → Nat → Nat → Nat → Nat → Fin B → Prop

namespace ChainMetaGroupedMasked2DKernelIO

/-- No two `out o`-write-active lanes share a destination cell — the
per-pinned-context injectivity that the slot-dependent cache readback needs
(the consumers' per-channel `hOutInj`). A static-window consumer discharges it
from its geometry. -/
def WriteInj (io : ChainMetaGroupedMasked2DKernelIO) (p₀ p₁ s1 s2 : Nat)
    (o : Fin io.nOut) : Prop :=
  ∀ j k : Fin io.B, io.writeMask o p₀ p₁ s1 s2 j → io.writeMask o p₀ p₁ s1 s2 k →
    io.write o p₀ p₁ s1 s2 j = io.write o p₀ p₁ s1 s2 k → j = k

/-- `io.Implements f` — the chained-metadata grouped masked Hoare triple. The
two slot values `s1`/`s2` are universally quantified and pinned to their slot
cells — `s2`'s pin reads `mbuf2` at the `s1`-dependent cell `mwin2 pid₀ pid₁ s1`,
so the chain is stated on the *loaded* first slot. The grouped inputs are one
function `xs : Fin nIn → Fin B → ℝ`, pinned per channel on its read-active lanes
(windows/masks all eating the loaded slots). The spec `f` takes the two slots
and the output channel; each output channel's readback leg is guarded by its
`WriteInj`; the frame excludes every ungated write window. -/
def Implements (io : ChainMetaGroupedMasked2DKernelIO)
    (f : Nat → Nat → Nat → Nat → (Fin io.nIn → Fin io.B → ℝ) → Fin io.nOut →
      Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = io.bufs →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (s1 s2 : Nat) (xs : Fin io.nIn → Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    io.mwin1 pid₀ pid₁ < A.extent io.mbuf1 →
    io.mwin2 pid₀ pid₁ s1 < A.extent io.mbuf2 →
    (∀ (i : Fin io.nIn) (j : Fin io.B), io.readMask i pid₀ pid₁ s1 s2 j →
      io.read i pid₀ pid₁ s1 s2 j < A.extent (io.inp i)) →
    (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o pid₀ pid₁ s1 s2 j →
      io.write o pid₀ pid₁ s1 s2 j < A.extent (io.out o)) →
    s₀.readMemValue .nat io.mbuf1 (io.mwin1 pid₀ pid₁) = s1 →
    s₀.readMemValue .nat io.mbuf2 (io.mwin2 pid₀ pid₁ s1) = s2 →
    (∀ (i : Fin io.nIn) (j : Fin io.B), io.readMask i pid₀ pid₁ s1 s2 j →
      s₀.readMem (io.inp i) (io.read i pid₀ pid₁ s1 s2 j) = xs i j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ o : Fin io.nOut, io.WriteInj pid₀ pid₁ s1 s2 o →
          ∀ j : Fin io.B, io.writeMask o pid₀ pid₁ s1 s2 j →
            s'.readMem A.flat (A.addr (io.out o) (io.write o pid₀ pid₁ s1 s2 j))
              = f pid₀ pid₁ s1 s2 xs o j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o pid₀ pid₁ s1 s2 j →
              o' ≠ A.addr (io.out o) (io.write o pid₀ pid₁ s1 s2 j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => ChainMetaGroupedMasked2DKernelIO.Implements

/-- Embed into the unified core: channels 0/1 are the two 1-lane `.nat` slots
(always read) — channel 1's window reads channel 0's pinned value, the chain
the core supports natively — and channels `k+2` the `nIn` float grouped data
channels whose windows/masks read both slots. Each output's mask is its write
gate conjoined with its `WriteInj`; `nOut` scratch channels shadow the same
cells with the ungated gates. `hout` ties every output channel's buffer into
the declared allocation list. -/
private def toU (io : ChainMetaGroupedMasked2DKernelIO)
    (hout : ∀ o, io.out o ∈ io.bufs) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := io.nIn + 2
  nOut := io.nOut
  nScr := io.nOut
  bufs := io.bufs
  ity := fun i => match i with
    | ⟨0, _⟩ => .nat
    | ⟨1, _⟩ => .nat
    | ⟨_+2, _⟩ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => 1
    | ⟨_+2, _⟩ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.mbuf1
    | ⟨1, _⟩ => io.mbuf2
    | ⟨k+2, h⟩ => io.inp ⟨k, by omega⟩
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := io.out
  obuf_mem := hout
  sarity := fun _ => io.B
  sbuf := io.out
  iwin := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => io.mwin1 p₀ p₁
    | ⟨1, _⟩ => fun _ => io.mwin2 p₀ p₁
        (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
    | ⟨k+2, h⟩ => fun j => io.read ⟨k, by omega⟩ p₀ p₁
        (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1)) j
  imask := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => True
    | ⟨k+2, h⟩ => fun j => io.readMask ⟨k, by omega⟩ p₀ p₁
        (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1)) j
  owin := fun o vals p₀ p₁ _ j => io.write o p₀ p₁
      (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
      (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1)) j
  omask := fun o vals p₀ p₁ _ j =>
    io.writeMask o p₀ p₁
      (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
      (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1)) j
    ∧ io.WriteInj p₀ p₁
        (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1)) o
  swin := fun t vals p₀ p₁ _ j => io.write t p₀ p₁
      (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
      (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1)) j
  smask := fun t vals p₀ p₁ _ j => io.writeMask t p₀ p₁
      (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
      (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1)) j

/-- Assembly lemma — chained-slot sibling of
`MetaGroupedMasked2DKernelIO.Implements.intro`: the two slots enter `hts`/`hrun`
as pinned named `.nat` scalars, `s2`'s pin at the `s1`-dependent cell; each
output channel's value leg receives its per-context `WriteInj o` antecedent (the
consumers thread their cache `hOutInj` there); `hrun`'s frame takes one ungated
exclusion condition quantified over all output channels; `hout` ties the output
buffers into `bufs`. -/
theorem Implements.intro (io : ChainMetaGroupedMasked2DKernelIO)
    {f : Nat → Nat → Nat → Nat → (Fin io.nIn → Fin io.B → ℝ) → Fin io.nOut →
      Fin io.B → ℝ}
    (hout : ∀ o, io.out o ∈ io.bufs)
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (s1 s2 : Nat),
      s.readMemValue .nat io.mbuf1 (io.mwin1 (s.pids 0) (s.pids 1)) = s1 →
      s.readMemValue .nat io.mbuf2 (io.mwin2 (s.pids 0) (s.pids 1) s1) = s2 →
      io.mwin1 (s.pids 0) (s.pids 1) < bounds io.mbuf1 →
      io.mwin2 (s.pids 0) (s.pids 1) s1 < bounds io.mbuf2 →
      (∀ (i : Fin io.nIn) (j : Fin io.B),
        io.readMask i (s.pids 0) (s.pids 1) s1 s2 j →
        io.read i (s.pids 0) (s.pids 1) s1 s2 j < bounds (io.inp i)) →
      (∀ (o : Fin io.nOut) (j : Fin io.B),
        io.writeMask o (s.pids 0) (s.pids 1) s1 s2 j →
        io.write o (s.pids 0) (s.pids 1) s1 s2 j < bounds (io.out o)) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (s1 s2 : Nat) (xs : Fin io.nIn → Fin io.B → ℝ),
      s₀.readMemValue .nat io.mbuf1 (io.mwin1 (s₀.pids 0) (s₀.pids 1)) = s1 →
      s₀.readMemValue .nat io.mbuf2 (io.mwin2 (s₀.pids 0) (s₀.pids 1) s1) = s2 →
      (∀ (i : Fin io.nIn) (j : Fin io.B),
        io.readMask i (s₀.pids 0) (s₀.pids 1) s1 s2 j →
        s₀.readMem (io.inp i) (io.read i (s₀.pids 0) (s₀.pids 1) s1 s2 j)
          = xs i j) →
      ∃ s1', exec (io.kernel.toAlgKernel) s₀ = some s1'
        ∧ (∀ o : Fin io.nOut, io.WriteInj (s₀.pids 0) (s₀.pids 1) s1 s2 o →
            ∀ j : Fin io.B, io.writeMask o (s₀.pids 0) (s₀.pids 1) s1 s2 j →
              s1'.readMem (io.out o) (io.write o (s₀.pids 0) (s₀.pids 1) s1 s2 j)
                = f (s₀.pids 0) (s₀.pids 1) s1 s2 xs o j)
        ∧ (∀ r o',
            (∀ (oc : Fin io.nOut) (j : Fin io.B),
              io.writeMask oc (s₀.pids 0) (s₀.pids 1) s1 s2 j →
              r ≠ io.out oc ∨ o' ≠ io.write oc (s₀.pids 0) (s₀.pids 1) s1 s2 j) →
            s1'.mem r o' = s₀.mem r o')) :
    io.Implements f := by
  have hcore : (io.toU hout).Implements
      (fun p₀ p₁ vals o j => f p₀ p₁
        (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (fun i j' => vals (⟨i.val + 2, by omega⟩ : Fin (io.nIn+2)) j') o j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib _hob hsb
      exact hts bounds s
        (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
        (hpins (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun i j hj => hib (⟨i.val + 2, by omega⟩ : Fin (io.nIn+2)) j hj)
        (fun o j hj => hsb o j hj)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1', hexec, hval, hframe⟩ :=
        hrun s₀ (vals (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1))
          (fun i j' => vals (⟨i.val + 2, by omega⟩ : Fin (io.nIn+2)) j')
          (hpins (⟨0, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
          (hpins (⟨1, by omega⟩ : Fin (io.nIn+2)) (⟨0, by decide⟩ : Fin 1) trivial)
          (fun i j hj => hpins (⟨i.val + 2, by omega⟩ : Fin (io.nIn+2)) j hj)
      refine ⟨s1', hexec, fun o j hj => hval o hj.2 j hj.1, ?_⟩
      intro r o' _hoc hsc
      exact hframe r o' (fun oc j hj => hsc oc j hj)
  intro A hd hregs hcov pid₀ pid₁ s1 s2 xs s₀ hpid₀ hpid₁ hu hb1 hb2 hbr hbw
    hm1 hm2 hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ => s1
        | ⟨1, _⟩ => fun _ => s2
        | ⟨k+2, h⟩ => xs ⟨k, by have h2 : k + 2 < io.nIn + 2 := h; omega⟩)
      s₀ hpid₀ hpid₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hb1
        | ⟨1, _⟩ => fun _ _ => hb2
        | ⟨k+2, h⟩ => fun j hj =>
            hbr ⟨k, by have h2 : k + 2 < io.nIn + 2 := h; omega⟩ j hj)
      (fun o j hj => hbw o j hj.1)
      (fun t j hj => hbw t j hj)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hm1
        | ⟨1, _⟩ => fun _ _ => hm2
        | ⟨k+2, h⟩ => fun j hj =>
            hx ⟨k, by have h2 : k + 2 < io.nIn + 2 := h; omega⟩ j hj)
  refine ⟨s', hexec, fun o hinj j hj => hval o j ⟨hj, hinj⟩, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | hn
  · exact Or.inl hflat
  · exact Or.inr ⟨fun o j hj => hn o j hj.1, fun t j hj => hn t j hj⟩

end ChainMetaGroupedMasked2DKernelIO

end VeriTile.Triton
