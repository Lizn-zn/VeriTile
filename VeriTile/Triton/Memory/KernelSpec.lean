/-
VeriTile.Triton.Memory.KernelSpec

The **KernelIO spec surface**: a kernel's headline correctness statement is
`io ⊨ f` — "the kernel described by the IO signature `io` implements the
mathematical function `f`" — a full Hoare triple packaged as one audited
definition:

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
-/

import VeriTile.Triton.Memory.Flatten
import VeriTile.Triton.Memory.FlattenR
import VeriTile.Triton.Memory.Denotation

namespace VeriTile.Triton

/-- IO signature of a two-input / one-output kernel with uniform tile length
`B`: which buffer is which argument, where each program instance **reads**
its two input tiles, and where it **writes** its output tile. The read
windows are the address half of the precondition, the write window the
address half of the postcondition. Buffer sizes are **not** part of the
signature — they belong to the allocation, and `Implements` quantifies over
every allocation large enough to contain the windows. This is the whole
kernel-specific audit surface of an `io ⊨ f` headline. -/
structure KernelIO₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- First input buffer. -/
  in1 : RegionName
  /-- Second input buffer. -/
  in2 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Tile length: each program instance reads/writes `B`-element windows. -/
  B : Nat
  /-- Where program `pid` reads its `in1` tile: `[read1 pid, read1 pid + B)`. -/
  read1 : Nat → Nat
  /-- Where program `pid` reads its `in2` tile. -/
  read2 : Nat → Nat
  /-- Where program `pid` writes its output tile. -/
  write : Nat → Nat
  /-- The output buffer's floating dtype — the quantization grid of the
  boundary store, used only by the rounding-correctness relation `⊨[R]`
  (its postcondition reads the output back as `outDType`-typed cells
  holding `R.round outDType (f …)`). Declared, not parsed: the headline
  proves the kernel's actual store cast matches. `.real` (the default)
  means an unrounded store — the exact relation `⊨` ignores this field. -/
  outDType : FloatDType := .real

namespace KernelIO₂

/-- `io.Implements f` — the kernel of `io` implements the mathematical
function `f` on its declared IO signature. Full Hoare triple; see the module
docstring for exactly what is quantified. -/
def Implements (io : KernelIO₂)
    (f : (Fin io.B → ℝ) → (Fin io.B → ℝ) → Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    -- ∀ base pointers: any disjoint allocation of exactly the three buffers
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    -- the program's window fits inside each allocated buffer
    io.read1 pid + io.B ≤ A.extent io.in1 →
    io.read2 pid + io.B ≤ A.extent io.in2 →
    io.write pid + io.B ≤ A.extent io.out →
  ∀ (xs ys : Fin io.B → ℝ) (s₀ : BlockState),
    -- the launch state: program id set, undef launch-clean, input windows
    -- loaded; everything else in s₀ (all buffer cells, registers) arbitrary
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, s₀.readMem io.in1 (io.read1 pid + j.val) = xs j) →
    (∀ j : Fin io.B, s₀.readMem io.in2 (io.read2 pid + j.val) = ys j) →
    ∃ s',
      -- termination of the translated pointer kernel …
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      -- … the output window holds f …
      ∧ (∀ j : Fin io.B,
          s'.readMem A.flat (A.addr io.out (io.write pid + j.val))
            = f xs ys j)
      -- … and every other cell is untouched (frame)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ j : Fin io.B, o' ≠ A.addr io.out (io.write pid + j.val)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => KernelIO₂.Implements

/-- Assembly lemma: `io ⊨ f` from three per-kernel obligations —
`FlattenOk` (bridge fragment membership), `TraceSafe` (the per-execution
safety walk, taking the window-in-bounds contract), and the region-model
Hoare triple `hrun` (termination + output values + frame, all against the
region model — the mathematical core). The flat-memory transport is done
here, once. -/
theorem Implements.intro (io : KernelIO₂)
    {f : (Fin io.B → ℝ) → (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      io.read1 s.pid + io.B ≤ bounds io.in1 →
      io.read2 s.pid + io.B ≤ bounds io.in2 →
      io.write s.pid + io.B ≤ bounds io.out →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs ys : Fin io.B → ℝ),
      (∀ j : Fin io.B, s₀.readMem io.in1 (io.read1 s₀.pid + j.val) = xs j) →
      (∀ j : Fin io.B, s₀.readMem io.in2 (io.read2 s₀.pid + j.val) = ys j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B,
            s1.readMem io.out (io.write s₀.pid + j.val) = f xs ys j)
        ∧ (∀ r o,
            (r ≠ io.out ∨ ∀ j : Fin io.B, o ≠ io.write s₀.pid + j.val) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  intro A hd hregs hcov pid h1 h2 h3 xs ys s₀ hpid hu hx hy
  subst hpid
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs ys hx hy
  have hts' : Kernel.TraceSafe A.extent (io.kernel.toAlgKernel) s₀ :=
    hts A.extent s₀ h1 h2 h3
  have hbridge := A.exec_flatten hd hcov _ s₀ hts' hok hu
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j
    have hmem : io.out ∈ A.regions := by rw [hregs]; simp
    have hlt : io.write s₀.pid + j.val < A.extent io.out := by
      have := j.isLt; omega
    rw [A.flattenState_readMem hd s1 hmem hlt]
    exact hval j
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ?_)
          by_cases hro : r = io.out
          · subst hro
            refine Or.inr fun j hoj => ?_
            rcases hcond with hflat | hnadr
            · exact hflat rfl
            · exact hnadr j (by rw [hoeq, hoj])
          · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

/-- `io.ImplementsR R f` — the **rounding-correctness** relation `io ⊨[R] f`:
the kernel of `io` computes the mathematical function `f` exactly and
quantizes it once at the declared output dtype (`io.outDType`) — the
boundary-rounding contract. Same full Hoare triple as `Implements` (same
precondition: ∀ disjoint allocation, ∀ in-bounds pid, ∀ launch state with
exact-ℝ inputs loaded), but the execution is `execR R` and the output
window holds **typed cells**: reading lane `j` back as `outDType` yields
`R.round outDType (f xs ys j)`. Inputs stay exact ℝ — the rounding model
acts at the kernel's cast/store sites, not at loads. At `R := .triv` the
store is exact and this degenerates to the exact surface. -/
def ImplementsR (io : KernelIO₂) (R : RoundingModel)
    (f : (Fin io.B → ℝ) → (Fin io.B → ℝ) → Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io.read1 pid + io.B ≤ A.extent io.in1 →
    io.read2 pid + io.B ≤ A.extent io.in2 →
    io.write pid + io.B ≤ A.extent io.out →
  ∀ (xs ys : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, s₀.readMem io.in1 (io.read1 pid + j.val) = xs j) →
    (∀ j : Fin io.B, s₀.readMem io.in2 (io.read2 pid + j.val) = ys j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B,
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid + j.val))
            = io.outDType.ofReal (R.round io.outDType (f xs ys j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ j : Fin io.B, o' ≠ A.addr io.out (io.write pid + j.val)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  KernelIO₂.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — the rounding sibling of `Implements.intro`:
`FlattenOk`, the `TraceSafeR R` safety walk, and the region-model rounded
Hoare triple `hrun` (termination under `execR R` + typed output readback +
frame). The flat transport is `execR_flatten` plus the typed readback
transport `flattenState_readMemAs`. -/
theorem ImplementsR.intro (io : KernelIO₂) {R : RoundingModel}
    {f : (Fin io.B → ℝ) → (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      io.read1 s.pid + io.B ≤ bounds io.in1 →
      io.read2 s.pid + io.B ≤ bounds io.in2 →
      io.write s.pid + io.B ≤ bounds io.out →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (xs ys : Fin io.B → ℝ),
      (∀ j : Fin io.B, s₀.readMem io.in1 (io.read1 s₀.pid + j.val) = xs j) →
      (∀ j : Fin io.B, s₀.readMem io.in2 (io.read2 s₀.pid + j.val) = ys j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B,
            s1.readMemAs io.outDType io.out (io.write s₀.pid + j.val)
              = io.outDType.ofReal (R.round io.outDType (f xs ys j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨ ∀ j : Fin io.B, o ≠ io.write s₀.pid + j.val) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid h1 h2 h3 xs ys s₀ hpid hu hx hy
  subst hpid
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs ys hx hy
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ h1 h2 h3
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j
    have hmem : io.out ∈ A.regions := by rw [hregs]; simp
    have hlt : io.write s₀.pid + j.val < A.extent io.out := by
      have := j.isLt; omega
    rw [A.flattenState_readMemAs hd s1 hmem hlt io.outDType]
    exact hval j
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ?_)
          by_cases hro : r = io.out
          · subst hro
            refine Or.inr fun j hoj => ?_
            rcases hcond with hflat | hnadr
            · exact hflat rfl
            · exact hnadr j (by rw [hoeq, hoj])
          · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

end KernelIO₂

/-- IO signature of a **masked** two-input / one-output kernel — the masked
sibling of `KernelIO₂`. Each program instance owns a `B`-lane window but only
its **active** lanes (`mask pid j`) touch memory: partial blocks at the end of
a buffer deactivate the overhanging lanes. Inactive lanes carry **no
obligations on either side** of the Hoare triple: the precondition constrains
input memory only at active lanes (in the flat world an inactive lane's
address may exceed the buffer or land in the next buffer, so requiring inputs
there would be nonsense), and the postcondition asserts output values only at
active lanes and frame everywhere else. -/
structure MaskedKernelIO₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- First input buffer. -/
  in1 : RegionName
  /-- Second input buffer. -/
  in2 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Tile length: each program instance owns `B`-element windows. -/
  B : Nat
  /-- Where program `pid` reads its `in1` tile: active lanes of
  `[read1 pid, read1 pid + B)`. -/
  read1 : Nat → Nat
  /-- Where program `pid` reads its `in2` tile. -/
  read2 : Nat → Nat
  /-- Where program `pid` writes its output tile. -/
  write : Nat → Nat
  /-- Program `pid`'s active lanes. Only these read, write, or carry spec
  content; the rest of the window is dead. -/
  mask : Nat → Fin B → Prop
  /-- This kernel's **private working buffers**, each with its per-program
  window start (lane-masked by `mask`, tile length `B`, like the output).
  They are allocated and the kernel may stage intermediates through them,
  but their post-state is not part of any contract: `Implements` and
  `Equiv` exclude them from the frame, and `Equiv` never compares them.
  Empty for kernels that stage nothing through memory. -/
  scratch : List (RegionName × (Nat → Nat)) := []

namespace MaskedKernelIO₂

/-- `io.Implements f` — masked sibling of `KernelIO₂.Implements`. Same full
Hoare triple, restricted to the active lanes: the window-in-bounds contract,
the loaded-inputs precondition, and the output-value postcondition are all
stated **lane-wise at active lanes only** (a partial block may overhang the
buffer on its inactive lanes), and the frame covers every cell outside the
active output lanes. -/
def Implements (io : MaskedKernelIO₂)
    (f : (Fin io.B → ℝ) → (Fin io.B → ℝ) → Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    -- ∀ base pointers: any disjoint allocation of exactly the declared
    -- buffers (the three interface buffers plus the private scratch)
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.out] ++ io.scratch.map Prod.fst →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    -- lane-wise bounds: every *active* lane lands inside its buffer
    (∀ j : Fin io.B, io.mask pid j → io.read1 pid + j.val < A.extent io.in1) →
    (∀ j : Fin io.B, io.mask pid j → io.read2 pid + j.val < A.extent io.in2) →
    (∀ j : Fin io.B, io.mask pid j → io.write pid + j.val < A.extent io.out) →
    (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.mask pid j →
      p.2 pid + j.val < A.extent p.1) →
  ∀ (xs ys : Fin io.B → ℝ) (s₀ : BlockState),
    -- the launch state: program id set, undef launch-clean, inputs loaded at
    -- the ACTIVE lanes only; everything else in s₀ arbitrary
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, io.mask pid j →
      s₀.readMem io.in1 (io.read1 pid + j.val) = xs j) →
    (∀ j : Fin io.B, io.mask pid j →
      s₀.readMem io.in2 (io.read2 pid + j.val) = ys j) →
    ∃ s',
      -- termination of the translated pointer kernel …
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      -- … every active output lane holds f …
      ∧ (∀ j : Fin io.B, io.mask pid j →
          s'.readMem A.flat (A.addr io.out (io.write pid + j.val))
            = f xs ys j)
      -- … and every cell outside the active output lanes and the active
      -- scratch lanes is untouched (frame)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.mask pid j →
                o' ≠ A.addr io.out (io.write pid + j.val)) ∧
             (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.mask pid j →
                o' ≠ A.addr p.1 (p.2 pid + j.val)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => MaskedKernelIO₂.Implements

/-- Assembly lemma — masked sibling of `KernelIO₂.Implements.intro`. The
three per-kernel obligations take the **lane-wise** contracts: `hts` gets the
active-lane bounds, `hrun` proves the region-model masked Hoare triple from
active-lane inputs only. The flat-memory transport is done here, once; the
per-lane extent bound feeding the flat read-back comes directly from the
write-lane hypothesis. -/
theorem Implements.intro (io : MaskedKernelIO₂)
    {f : (Fin io.B → ℝ) → (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ j : Fin io.B, io.mask s.pid j →
        io.read1 s.pid + j.val < bounds io.in1) →
      (∀ j : Fin io.B, io.mask s.pid j →
        io.read2 s.pid + j.val < bounds io.in2) →
      (∀ j : Fin io.B, io.mask s.pid j →
        io.write s.pid + j.val < bounds io.out) →
      (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.mask s.pid j →
        p.2 s.pid + j.val < bounds p.1) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs ys : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask s₀.pid j →
        s₀.readMem io.in1 (io.read1 s₀.pid + j.val) = xs j) →
      (∀ j : Fin io.B, io.mask s₀.pid j →
        s₀.readMem io.in2 (io.read2 s₀.pid + j.val) = ys j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B, io.mask s₀.pid j →
            s1.readMem io.out (io.write s₀.pid + j.val) = f xs ys j)
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.B, io.mask s₀.pid j →
                o ≠ io.write s₀.pid + j.val) →
            (∀ p ∈ io.scratch, r = p.1 →
              ∀ j : Fin io.B, io.mask s₀.pid j →
                o ≠ p.2 s₀.pid + j.val) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  intro A hd hregs hcov pid h1 h2 h3 hsc xs ys s₀ hpid hu hx hy
  subst hpid
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs ys hx hy
  have hts' : Kernel.TraceSafe A.extent (io.kernel.toAlgKernel) s₀ :=
    hts A.extent s₀ h1 h2 h3 hsc
  have hbridge := A.exec_flatten hd hcov _ s₀ hts' hok hu
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j hj
    have hmem : io.out ∈ A.regions := by rw [hregs]; simp
    have hlt : io.write s₀.pid + j.val < A.extent io.out := h3 j hj
    rw [A.flattenState_readMem hd s1 hmem hlt]
    exact hval j hj
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ?_ ?_)
          · by_cases hro : r = io.out
            · subst hro
              refine Or.inr fun j hj hoj => ?_
              rcases hcond with hflat | ⟨hnout, _⟩
              · exact hflat rfl
              · exact hnout j hj (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp j hj hoj
            rcases hcond with hflat | ⟨_, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp j hj (by rw [hoeq, hrp, hoj])
    · simp only [FlatAlloc.flattenState, if_neg hr]

/-- `io₁ ≡[R] io₂` — **kernel equivalence on a shared IO signature**, the
`⊨`-grade form of the refinement surface. The interface (buffers, windows,
mask) is read from `io₁` — instances share it by construction, e.g.
`{ referenceIO with kernel := rewritten, scratch := [] }`; `io₂` contributes
only its `kernel` and its private `scratch`. The claim: for every disjoint
flat allocation of the interface buffers plus **both** kernels' scratch,
every program id whose active lanes are in bounds, and **every** launch
state (no input hypotheses at all — equal inputs are "the same `s₀`"), both
kernels terminate under `execR R`, their active output lanes agree, and each
kernel leaves every cell outside the active output window and its own active
scratch windows untouched. Determinism makes this genuinely symmetric —
"refines" and "is equivalent to" coincide. -/
def Equiv (io₁ io₂ : MaskedKernelIO₂) (R : RoundingModel) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io₁.in1, io₁.in2, io₁.out]
      ++ (io₁.scratch.map Prod.fst ++ io₂.scratch.map Prod.fst) →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    -- lane-wise bounds: every *active* lane of every window is in bounds
    (∀ j : Fin io₁.B, io₁.mask pid j →
      io₁.read1 pid + j.val < A.extent io₁.in1) →
    (∀ j : Fin io₁.B, io₁.mask pid j →
      io₁.read2 pid + j.val < A.extent io₁.in2) →
    (∀ j : Fin io₁.B, io₁.mask pid j →
      io₁.write pid + j.val < A.extent io₁.out) →
    (∀ p ∈ io₁.scratch, ∀ j : Fin io₁.B, io₁.mask pid j →
      p.2 pid + j.val < A.extent p.1) →
    (∀ p ∈ io₂.scratch, ∀ j : Fin io₁.B, io₁.mask pid j →
      p.2 pid + j.val < A.extent p.1) →
  ∀ s₀ : BlockState,
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    ∃ s₁ s₂,
      -- both translated pointer kernels terminate under execR R …
      execR R (A.flattenKernel io₁.kernel.toAlgKernel) (A.flattenState s₀)
        = some s₁
      ∧ execR R (A.flattenKernel io₂.kernel.toAlgKernel) (A.flattenState s₀)
        = some s₂
      -- … their active output lanes agree …
      ∧ (∀ j : Fin io₁.B, io₁.mask pid j →
          s₁.readMem A.flat (A.addr io₁.out (io₁.write pid + j.val))
            = s₂.readMem A.flat (A.addr io₁.out (io₁.write pid + j.val)))
      -- … and each side frames outside the output window ∪ its own scratch
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io₁.B, io₁.mask pid j →
                o' ≠ A.addr io₁.out (io₁.write pid + j.val)) ∧
             (∀ p ∈ io₁.scratch, ∀ j : Fin io₁.B, io₁.mask pid j →
                o' ≠ A.addr p.1 (p.2 pid + j.val)))) →
          s₁.mem r' o' = (A.flattenState s₀).mem r' o')
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io₁.B, io₁.mask pid j →
                o' ≠ A.addr io₁.out (io₁.write pid + j.val)) ∧
             (∀ p ∈ io₂.scratch, ∀ j : Fin io₁.B, io₁.mask pid j →
                o' ≠ A.addr p.1 (p.2 pid + j.val)))) →
          s₂.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io₁ " ≡[" R "] " io₂ =>
  MaskedKernelIO₂.Equiv io₁ io₂ R

/-- Assembly lemma for `≡[R]` — the two-kernel sibling of
`Implements.intro`. Per-kernel obligations: `FlattenOk` and the rounding
trace-safety walk `TraceSafeR` (addresses don't round, but the walk runs
under `execR R`'s states). The mathematical core `hrun` is the region-model
equivalence: from **any** state, both kernels terminate, their active output
lanes agree, and each frames outside the output window ∪ its own scratch —
for existing refinement showcases this is a repackaging of the proven
`ComputeRefine.Refines` theorem plus per-kernel frame lemmas. -/
theorem Equiv.intro (io₁ io₂ : MaskedKernelIO₂) {R : RoundingModel}
    (hok₁ : (io₁.kernel.toAlgKernel).FlattenOk)
    (hok₂ : (io₂.kernel.toAlgKernel).FlattenOk)
    (hts₁ : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ j : Fin io₁.B, io₁.mask s.pid j →
        io₁.read1 s.pid + j.val < bounds io₁.in1) →
      (∀ j : Fin io₁.B, io₁.mask s.pid j →
        io₁.read2 s.pid + j.val < bounds io₁.in2) →
      (∀ j : Fin io₁.B, io₁.mask s.pid j →
        io₁.write s.pid + j.val < bounds io₁.out) →
      (∀ p ∈ io₁.scratch, ∀ j : Fin io₁.B, io₁.mask s.pid j →
        p.2 s.pid + j.val < bounds p.1) →
      (io₁.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hts₂ : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ j : Fin io₁.B, io₁.mask s.pid j →
        io₁.read1 s.pid + j.val < bounds io₁.in1) →
      (∀ j : Fin io₁.B, io₁.mask s.pid j →
        io₁.read2 s.pid + j.val < bounds io₁.in2) →
      (∀ j : Fin io₁.B, io₁.mask s.pid j →
        io₁.write s.pid + j.val < bounds io₁.out) →
      (∀ p ∈ io₂.scratch, ∀ j : Fin io₁.B, io₁.mask s.pid j →
        p.2 s.pid + j.val < bounds p.1) →
      (io₂.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ s₀ : BlockState,
      ∃ s1 s2,
        execR R (io₁.kernel.toAlgKernel) s₀ = some s1
        ∧ execR R (io₂.kernel.toAlgKernel) s₀ = some s2
        ∧ (∀ j : Fin io₁.B, io₁.mask s₀.pid j →
            s1.readMem io₁.out (io₁.write s₀.pid + j.val)
              = s2.readMem io₁.out (io₁.write s₀.pid + j.val))
        ∧ (∀ r o,
            (r ≠ io₁.out ∨
              ∀ j : Fin io₁.B, io₁.mask s₀.pid j →
                o ≠ io₁.write s₀.pid + j.val) →
            (∀ p ∈ io₁.scratch, r = p.1 →
              ∀ j : Fin io₁.B, io₁.mask s₀.pid j →
                o ≠ p.2 s₀.pid + j.val) →
            s1.mem r o = s₀.mem r o)
        ∧ (∀ r o,
            (r ≠ io₁.out ∨
              ∀ j : Fin io₁.B, io₁.mask s₀.pid j →
                o ≠ io₁.write s₀.pid + j.val) →
            (∀ p ∈ io₂.scratch, r = p.1 →
              ∀ j : Fin io₁.B, io₁.mask s₀.pid j →
                o ≠ p.2 s₀.pid + j.val) →
            s2.mem r o = s₀.mem r o)) :
    io₁.Equiv io₂ R := by
  intro A hd hregs hcov pid h1 h2 h3 hsc1 hsc2 s₀ hpid hu
  subst hpid
  obtain ⟨s1, s2, hexec1, hexec2, hval, hframe1, hframe2⟩ := hrun s₀
  have hts₁' : (io₁.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts₁ A.extent s₀ h1 h2 h3 hsc1
  have hts₂' : (io₂.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts₂ A.extent s₀ h1 h2 h3 hsc2
  have hbridge1 := A.execR_flatten hd hcov R _ s₀ hts₁' hok₁ hu
  have hbridge2 := A.execR_flatten hd hcov R _ s₀ hts₂' hok₂ hu
  have hmem : io₁.out ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, A.flattenState s2, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hbridge1, hexec1, Option.map_some]
  · rw [hbridge2, hexec2, Option.map_some]
  · intro j hj
    have hlt : io₁.write s₀.pid + j.val < A.extent io₁.out := h3 j hj
    rw [A.flattenState_readMem hd s1 hmem hlt,
        A.flattenState_readMem hd s2 hmem hlt]
    exact hval j hj
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe1 r o ?_ ?_)
          · by_cases hro : r = io₁.out
            · subst hro
              refine Or.inr fun j hj hoj => ?_
              rcases hcond with hflat | ⟨hnout, _⟩
              · exact hflat rfl
              · exact hnout j hj (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp j hj hoj
            rcases hcond with hflat | ⟨_, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp j hj (by rw [hoeq, hrp, hoj])
    · simp only [FlatAlloc.flattenState, if_neg hr]
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s2).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s2.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe2 r o ?_ ?_)
          · by_cases hro : r = io₁.out
            · subst hro
              refine Or.inr fun j hj hoj => ?_
              rcases hcond with hflat | ⟨hnout, _⟩
              · exact hflat rfl
              · exact hnout j hj (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp j hj hoj
            rcases hcond with hflat | ⟨_, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp j hj (by rw [hoeq, hrp, hoj])
    · simp only [FlatAlloc.flattenState, if_neg hr]

end MaskedKernelIO₂

/-- One **private working buffer** of an unmasked kernel: program `pid` may
stage intermediates in the window `[win pid, win pid + len)` of buffer
`buf`. Scratch buffers are allocated and writable, but their post-state is
not part of any contract: `Implements` and `Equiv` exclude them from the
frame, and `Equiv` never compares them. (The masked structs carry their own
scratch shape instead — there the active lane set comes from the kernel's
`mask`.) -/
structure ScratchSpec where
  /-- The buffer. -/
  buf : RegionName
  /-- Where program `pid`'s scratch window starts. -/
  win : Nat → Nat
  /-- The scratch window length. -/
  len : Nat

/-- IO signature of a **one-input / one-output** kernel. Unlike `KernelIO₂`,
the input and output tile lengths are independent (`Bin`/`Bout`) — this
covers whole-tile maps (softmax: `Bin = Bout = B`) as well as reductions
that write a single cell per program (LSE, row-wise sum/max:
`Bout = 1`). Same reading as `KernelIO₂`: `read` is the address half of
the precondition, `write` of the postcondition; buffer sizes are not
signature content. -/
structure KernelIO₁ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- Input buffer. -/
  inp : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Input tile length. -/
  Bin : Nat
  /-- Output tile length (`1` for scalar-per-program reductions). -/
  Bout : Nat
  /-- Where program `pid` reads its input tile: `[read pid, read pid + Bin)`. -/
  read : Nat → Nat
  /-- Where program `pid` writes its output tile. -/
  write : Nat → Nat
  /-- This kernel's private working buffers (see `ScratchSpec`). Empty for
  kernels that stage nothing through memory. -/
  scratch : List ScratchSpec := []

namespace KernelIO₁

/-- `io.Implements f` — one-input sibling of `KernelIO₂.Implements`; see
the module docstring for exactly what is quantified. -/
def Implements (io : KernelIO₁)
    (f : (Fin io.Bin → ℝ) → Fin io.Bout → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp, io.out] ++ io.scratch.map (·.buf) →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io.read pid + io.Bin ≤ A.extent io.inp →
    io.write pid + io.Bout ≤ A.extent io.out →
    (∀ p ∈ io.scratch, p.win pid + p.len ≤ A.extent p.buf) →
  ∀ (xs : Fin io.Bin → ℝ) (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.Bin, s₀.readMem io.inp (io.read pid + j.val) = xs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.Bout,
          s'.readMem A.flat (A.addr io.out (io.write pid + j.val))
            = f xs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.Bout,
                o' ≠ A.addr io.out (io.write pid + j.val)) ∧
             (∀ p ∈ io.scratch, ∀ k : Fin p.len,
                o' ≠ A.addr p.buf (p.win pid + k.val)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => KernelIO₁.Implements

/-- Assembly lemma — one-input sibling of `KernelIO₂.Implements.intro`. -/
theorem Implements.intro (io : KernelIO₁)
    {f : (Fin io.Bin → ℝ) → Fin io.Bout → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      io.read s.pid + io.Bin ≤ bounds io.inp →
      io.write s.pid + io.Bout ≤ bounds io.out →
      (∀ p ∈ io.scratch, p.win s.pid + p.len ≤ bounds p.buf) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.Bin → ℝ),
      (∀ j : Fin io.Bin, s₀.readMem io.inp (io.read s₀.pid + j.val) = xs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.Bout,
            s1.readMem io.out (io.write s₀.pid + j.val) = f xs j)
        ∧ (∀ r o,
            (r ≠ io.out ∨ ∀ j : Fin io.Bout, o ≠ io.write s₀.pid + j.val) →
            (∀ p ∈ io.scratch, r = p.buf →
              ∀ k : Fin p.len, o ≠ p.win s₀.pid + k.val) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  intro A hd hregs hcov pid h1 h2 hsc xs s₀ hpid hu hx
  subst hpid
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs hx
  have hts' : Kernel.TraceSafe A.extent (io.kernel.toAlgKernel) s₀ :=
    hts A.extent s₀ h1 h2 hsc
  have hbridge := A.exec_flatten hd hcov _ s₀ hts' hok hu
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j
    have hmem : io.out ∈ A.regions := by rw [hregs]; simp
    have hlt : io.write s₀.pid + j.val < A.extent io.out := by
      have := j.isLt; omega
    rw [A.flattenState_readMem hd s1 hmem hlt]
    exact hval j
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ?_ ?_)
          · by_cases hro : r = io.out
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨hnout, _⟩
              · exact hflat rfl
              · exact hnout j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp k hok'
            rcases hcond with hflat | ⟨_, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp k (by rw [hoeq, hrp, hok'])
    · simp only [FlatAlloc.flattenState, if_neg hr]

/-- `io₁ ≡[R] io₂` — kernel equivalence on a shared one-input IO signature;
the one-input sibling of `MaskedKernelIO₂.Equiv`. The interface is read
from `io₁` (instances share it by structure update); `io₂` contributes only
its `kernel` and its private `scratch`. For every disjoint allocation of the
interface buffers plus both scratches, every program id whose windows are in
bounds, and **every** launch state (no input hypotheses — equal inputs are
"the same `s₀`"), both kernels terminate under `execR R`, their output
windows agree, and each frames outside the output window ∪ its own
scratch. -/
def Equiv (io₁ io₂ : KernelIO₁) (R : RoundingModel) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io₁.inp, io₁.out]
      ++ (io₁.scratch.map (·.buf) ++ io₂.scratch.map (·.buf)) →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io₁.read pid + io₁.Bin ≤ A.extent io₁.inp →
    io₁.write pid + io₁.Bout ≤ A.extent io₁.out →
    (∀ p ∈ io₁.scratch, p.win pid + p.len ≤ A.extent p.buf) →
    (∀ p ∈ io₂.scratch, p.win pid + p.len ≤ A.extent p.buf) →
  ∀ s₀ : BlockState,
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    ∃ s₁ s₂,
      execR R (A.flattenKernel io₁.kernel.toAlgKernel) (A.flattenState s₀)
        = some s₁
      ∧ execR R (A.flattenKernel io₂.kernel.toAlgKernel) (A.flattenState s₀)
        = some s₂
      ∧ (∀ j : Fin io₁.Bout,
          s₁.readMem A.flat (A.addr io₁.out (io₁.write pid + j.val))
            = s₂.readMem A.flat (A.addr io₁.out (io₁.write pid + j.val)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io₁.Bout,
                o' ≠ A.addr io₁.out (io₁.write pid + j.val)) ∧
             (∀ p ∈ io₁.scratch, ∀ k : Fin p.len,
                o' ≠ A.addr p.buf (p.win pid + k.val)))) →
          s₁.mem r' o' = (A.flattenState s₀).mem r' o')
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io₁.Bout,
                o' ≠ A.addr io₁.out (io₁.write pid + j.val)) ∧
             (∀ p ∈ io₂.scratch, ∀ k : Fin p.len,
                o' ≠ A.addr p.buf (p.win pid + k.val)))) →
          s₂.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io₁ " ≡[" R "] " io₂ =>
  KernelIO₁.Equiv io₁ io₂ R

/-- Assembly lemma for the one-input `≡[R]` — sibling of
`MaskedKernelIO₂.Equiv.intro`; see there for the reading of the
obligations. -/
theorem Equiv.intro (io₁ io₂ : KernelIO₁) {R : RoundingModel}
    (hok₁ : (io₁.kernel.toAlgKernel).FlattenOk)
    (hok₂ : (io₂.kernel.toAlgKernel).FlattenOk)
    (hts₁ : ∀ (bounds : RegionBounds) (s : BlockState),
      io₁.read s.pid + io₁.Bin ≤ bounds io₁.inp →
      io₁.write s.pid + io₁.Bout ≤ bounds io₁.out →
      (∀ p ∈ io₁.scratch, p.win s.pid + p.len ≤ bounds p.buf) →
      (io₁.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hts₂ : ∀ (bounds : RegionBounds) (s : BlockState),
      io₁.read s.pid + io₁.Bin ≤ bounds io₁.inp →
      io₁.write s.pid + io₁.Bout ≤ bounds io₁.out →
      (∀ p ∈ io₂.scratch, p.win s.pid + p.len ≤ bounds p.buf) →
      (io₂.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ s₀ : BlockState,
      ∃ s1 s2,
        execR R (io₁.kernel.toAlgKernel) s₀ = some s1
        ∧ execR R (io₂.kernel.toAlgKernel) s₀ = some s2
        ∧ (∀ j : Fin io₁.Bout,
            s1.readMem io₁.out (io₁.write s₀.pid + j.val)
              = s2.readMem io₁.out (io₁.write s₀.pid + j.val))
        ∧ (∀ r o,
            (r ≠ io₁.out ∨
              ∀ j : Fin io₁.Bout, o ≠ io₁.write s₀.pid + j.val) →
            (∀ p ∈ io₁.scratch, r = p.buf →
              ∀ k : Fin p.len, o ≠ p.win s₀.pid + k.val) →
            s1.mem r o = s₀.mem r o)
        ∧ (∀ r o,
            (r ≠ io₁.out ∨
              ∀ j : Fin io₁.Bout, o ≠ io₁.write s₀.pid + j.val) →
            (∀ p ∈ io₂.scratch, r = p.buf →
              ∀ k : Fin p.len, o ≠ p.win s₀.pid + k.val) →
            s2.mem r o = s₀.mem r o)) :
    io₁.Equiv io₂ R := by
  intro A hd hregs hcov pid h1 h2 hsc1 hsc2 s₀ hpid hu
  subst hpid
  obtain ⟨s1, s2, hexec1, hexec2, hval, hframe1, hframe2⟩ := hrun s₀
  have hts₁' : (io₁.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts₁ A.extent s₀ h1 h2 hsc1
  have hts₂' : (io₂.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts₂ A.extent s₀ h1 h2 hsc2
  have hbridge1 := A.execR_flatten hd hcov R _ s₀ hts₁' hok₁ hu
  have hbridge2 := A.execR_flatten hd hcov R _ s₀ hts₂' hok₂ hu
  have hmem : io₁.out ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, A.flattenState s2, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hbridge1, hexec1, Option.map_some]
  · rw [hbridge2, hexec2, Option.map_some]
  · intro j
    have hlt : io₁.write s₀.pid + j.val < A.extent io₁.out := by
      have := j.isLt; omega
    rw [A.flattenState_readMem hd s1 hmem hlt,
        A.flattenState_readMem hd s2 hmem hlt]
    exact hval j
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe1 r o ?_ ?_)
          · by_cases hro : r = io₁.out
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨hnout, _⟩
              · exact hflat rfl
              · exact hnout j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp k hok'
            rcases hcond with hflat | ⟨_, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp k (by rw [hoeq, hrp, hok'])
    · simp only [FlatAlloc.flattenState, if_neg hr]
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s2).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s2.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe2 r o ?_ ?_)
          · by_cases hro : r = io₁.out
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨hnout, _⟩
              · exact hflat rfl
              · exact hnout j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp k hok'
            rcases hcond with hflat | ⟨_, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp k (by rw [hoeq, hrp, hok'])
    · simp only [FlatAlloc.flattenState, if_neg hr]

end KernelIO₁

/-- IO signature of a **three-input / one-output** kernel. All four tile
lengths are independent (`B1`/`B2`/`B3`/`Bout`), so this covers per-lane
ternary maps as well as kernels that mix per-program tiles with **shared**
buffers: a read window may be constant (e.g. `read2 := fun _ => 0` — every
program reads the same scalar cell). Same reading as `KernelIO₂`: the reads
are the address half of the precondition, the write of the postcondition;
buffer sizes are not signature content. -/
structure KernelIO₃ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- First input buffer. -/
  in1 : RegionName
  /-- Second input buffer. -/
  in2 : RegionName
  /-- Third input buffer. -/
  in3 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- First input tile length. -/
  B1 : Nat
  /-- Second input tile length. -/
  B2 : Nat
  /-- Third input tile length. -/
  B3 : Nat
  /-- Output tile length. -/
  Bout : Nat
  /-- Where program `pid` reads its `in1` tile: `[read1 pid, read1 pid + B1)`. -/
  read1 : Nat → Nat
  /-- Where program `pid` reads its `in2` tile. -/
  read2 : Nat → Nat
  /-- Where program `pid` reads its `in3` tile. -/
  read3 : Nat → Nat
  /-- Where program `pid` writes its output tile. -/
  write : Nat → Nat
  /-- This kernel's private working buffers (see `ScratchSpec`). Empty for
  kernels that stage nothing through memory. -/
  scratch : List ScratchSpec := []

namespace KernelIO₃

/-- `io.Implements f` — three-input sibling of `KernelIO₂.Implements`; see
the module docstring for exactly what is quantified. -/
def Implements (io : KernelIO₃)
    (f : (Fin io.B1 → ℝ) → (Fin io.B2 → ℝ) → (Fin io.B3 → ℝ) →
      Fin io.Bout → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.in3, io.out] ++ io.scratch.map (·.buf) →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io.read1 pid + io.B1 ≤ A.extent io.in1 →
    io.read2 pid + io.B2 ≤ A.extent io.in2 →
    io.read3 pid + io.B3 ≤ A.extent io.in3 →
    io.write pid + io.Bout ≤ A.extent io.out →
    (∀ p ∈ io.scratch, p.win pid + p.len ≤ A.extent p.buf) →
  ∀ (xs : Fin io.B1 → ℝ) (ys : Fin io.B2 → ℝ) (zs : Fin io.B3 → ℝ)
      (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B1, s₀.readMem io.in1 (io.read1 pid + j.val) = xs j) →
    (∀ j : Fin io.B2, s₀.readMem io.in2 (io.read2 pid + j.val) = ys j) →
    (∀ j : Fin io.B3, s₀.readMem io.in3 (io.read3 pid + j.val) = zs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.Bout,
          s'.readMem A.flat (A.addr io.out (io.write pid + j.val))
            = f xs ys zs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.Bout,
                o' ≠ A.addr io.out (io.write pid + j.val)) ∧
             (∀ p ∈ io.scratch, ∀ k : Fin p.len,
                o' ≠ A.addr p.buf (p.win pid + k.val)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => KernelIO₃.Implements

/-- Assembly lemma — three-input sibling of `KernelIO₂.Implements.intro`. -/
theorem Implements.intro (io : KernelIO₃)
    {f : (Fin io.B1 → ℝ) → (Fin io.B2 → ℝ) → (Fin io.B3 → ℝ) →
      Fin io.Bout → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      io.read1 s.pid + io.B1 ≤ bounds io.in1 →
      io.read2 s.pid + io.B2 ≤ bounds io.in2 →
      io.read3 s.pid + io.B3 ≤ bounds io.in3 →
      io.write s.pid + io.Bout ≤ bounds io.out →
      (∀ p ∈ io.scratch, p.win s.pid + p.len ≤ bounds p.buf) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.B1 → ℝ) (ys : Fin io.B2 → ℝ)
        (zs : Fin io.B3 → ℝ),
      (∀ j : Fin io.B1, s₀.readMem io.in1 (io.read1 s₀.pid + j.val) = xs j) →
      (∀ j : Fin io.B2, s₀.readMem io.in2 (io.read2 s₀.pid + j.val) = ys j) →
      (∀ j : Fin io.B3, s₀.readMem io.in3 (io.read3 s₀.pid + j.val) = zs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.Bout,
            s1.readMem io.out (io.write s₀.pid + j.val) = f xs ys zs j)
        ∧ (∀ r o,
            (r ≠ io.out ∨ ∀ j : Fin io.Bout, o ≠ io.write s₀.pid + j.val) →
            (∀ p ∈ io.scratch, r = p.buf →
              ∀ k : Fin p.len, o ≠ p.win s₀.pid + k.val) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  intro A hd hregs hcov pid h1 h2 h3 h4 hsc xs ys zs s₀ hpid hu hx hy hz
  subst hpid
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs ys zs hx hy hz
  have hts' : Kernel.TraceSafe A.extent (io.kernel.toAlgKernel) s₀ :=
    hts A.extent s₀ h1 h2 h3 h4 hsc
  have hbridge := A.exec_flatten hd hcov _ s₀ hts' hok hu
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j
    have hmem : io.out ∈ A.regions := by rw [hregs]; simp
    have hlt : io.write s₀.pid + j.val < A.extent io.out := by
      have := j.isLt; omega
    rw [A.flattenState_readMem hd s1 hmem hlt]
    exact hval j
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ?_ ?_)
          · by_cases hro : r = io.out
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨hnout, _⟩
              · exact hflat rfl
              · exact hnout j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp k hok'
            rcases hcond with hflat | ⟨_, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp k (by rw [hoeq, hrp, hok'])
    · simp only [FlatAlloc.flattenState, if_neg hr]

/-- `io₁ ≡[R] io₂` — kernel equivalence on a shared three-input IO
signature; the three-input sibling of `MaskedKernelIO₂.Equiv`. The interface
is read from `io₁` (instances share it by structure update); `io₂`
contributes only its `kernel` and its private `scratch`. No input
hypotheses: equal inputs are "the same `s₀`". -/
def Equiv (io₁ io₂ : KernelIO₃) (R : RoundingModel) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io₁.in1, io₁.in2, io₁.in3, io₁.out]
      ++ (io₁.scratch.map (·.buf) ++ io₂.scratch.map (·.buf)) →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io₁.read1 pid + io₁.B1 ≤ A.extent io₁.in1 →
    io₁.read2 pid + io₁.B2 ≤ A.extent io₁.in2 →
    io₁.read3 pid + io₁.B3 ≤ A.extent io₁.in3 →
    io₁.write pid + io₁.Bout ≤ A.extent io₁.out →
    (∀ p ∈ io₁.scratch, p.win pid + p.len ≤ A.extent p.buf) →
    (∀ p ∈ io₂.scratch, p.win pid + p.len ≤ A.extent p.buf) →
  ∀ s₀ : BlockState,
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    ∃ s₁ s₂,
      execR R (A.flattenKernel io₁.kernel.toAlgKernel) (A.flattenState s₀)
        = some s₁
      ∧ execR R (A.flattenKernel io₂.kernel.toAlgKernel) (A.flattenState s₀)
        = some s₂
      ∧ (∀ j : Fin io₁.Bout,
          s₁.readMem A.flat (A.addr io₁.out (io₁.write pid + j.val))
            = s₂.readMem A.flat (A.addr io₁.out (io₁.write pid + j.val)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io₁.Bout,
                o' ≠ A.addr io₁.out (io₁.write pid + j.val)) ∧
             (∀ p ∈ io₁.scratch, ∀ k : Fin p.len,
                o' ≠ A.addr p.buf (p.win pid + k.val)))) →
          s₁.mem r' o' = (A.flattenState s₀).mem r' o')
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io₁.Bout,
                o' ≠ A.addr io₁.out (io₁.write pid + j.val)) ∧
             (∀ p ∈ io₂.scratch, ∀ k : Fin p.len,
                o' ≠ A.addr p.buf (p.win pid + k.val)))) →
          s₂.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io₁ " ≡[" R "] " io₂ =>
  KernelIO₃.Equiv io₁ io₂ R

/-- Assembly lemma for the three-input `≡[R]` — sibling of
`MaskedKernelIO₂.Equiv.intro`; see there for the reading of the
obligations. -/
theorem Equiv.intro (io₁ io₂ : KernelIO₃) {R : RoundingModel}
    (hok₁ : (io₁.kernel.toAlgKernel).FlattenOk)
    (hok₂ : (io₂.kernel.toAlgKernel).FlattenOk)
    (hts₁ : ∀ (bounds : RegionBounds) (s : BlockState),
      io₁.read1 s.pid + io₁.B1 ≤ bounds io₁.in1 →
      io₁.read2 s.pid + io₁.B2 ≤ bounds io₁.in2 →
      io₁.read3 s.pid + io₁.B3 ≤ bounds io₁.in3 →
      io₁.write s.pid + io₁.Bout ≤ bounds io₁.out →
      (∀ p ∈ io₁.scratch, p.win s.pid + p.len ≤ bounds p.buf) →
      (io₁.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hts₂ : ∀ (bounds : RegionBounds) (s : BlockState),
      io₁.read1 s.pid + io₁.B1 ≤ bounds io₁.in1 →
      io₁.read2 s.pid + io₁.B2 ≤ bounds io₁.in2 →
      io₁.read3 s.pid + io₁.B3 ≤ bounds io₁.in3 →
      io₁.write s.pid + io₁.Bout ≤ bounds io₁.out →
      (∀ p ∈ io₂.scratch, p.win s.pid + p.len ≤ bounds p.buf) →
      (io₂.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ s₀ : BlockState,
      ∃ s1 s2,
        execR R (io₁.kernel.toAlgKernel) s₀ = some s1
        ∧ execR R (io₂.kernel.toAlgKernel) s₀ = some s2
        ∧ (∀ j : Fin io₁.Bout,
            s1.readMem io₁.out (io₁.write s₀.pid + j.val)
              = s2.readMem io₁.out (io₁.write s₀.pid + j.val))
        ∧ (∀ r o,
            (r ≠ io₁.out ∨
              ∀ j : Fin io₁.Bout, o ≠ io₁.write s₀.pid + j.val) →
            (∀ p ∈ io₁.scratch, r = p.buf →
              ∀ k : Fin p.len, o ≠ p.win s₀.pid + k.val) →
            s1.mem r o = s₀.mem r o)
        ∧ (∀ r o,
            (r ≠ io₁.out ∨
              ∀ j : Fin io₁.Bout, o ≠ io₁.write s₀.pid + j.val) →
            (∀ p ∈ io₂.scratch, r = p.buf →
              ∀ k : Fin p.len, o ≠ p.win s₀.pid + k.val) →
            s2.mem r o = s₀.mem r o)) :
    io₁.Equiv io₂ R := by
  intro A hd hregs hcov pid h1 h2 h3 h4 hsc1 hsc2 s₀ hpid hu
  subst hpid
  obtain ⟨s1, s2, hexec1, hexec2, hval, hframe1, hframe2⟩ := hrun s₀
  have hts₁' : (io₁.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts₁ A.extent s₀ h1 h2 h3 h4 hsc1
  have hts₂' : (io₂.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts₂ A.extent s₀ h1 h2 h3 h4 hsc2
  have hbridge1 := A.execR_flatten hd hcov R _ s₀ hts₁' hok₁ hu
  have hbridge2 := A.execR_flatten hd hcov R _ s₀ hts₂' hok₂ hu
  have hmem : io₁.out ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, A.flattenState s2, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hbridge1, hexec1, Option.map_some]
  · rw [hbridge2, hexec2, Option.map_some]
  · intro j
    have hlt : io₁.write s₀.pid + j.val < A.extent io₁.out := by
      have := j.isLt; omega
    rw [A.flattenState_readMem hd s1 hmem hlt,
        A.flattenState_readMem hd s2 hmem hlt]
    exact hval j
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe1 r o ?_ ?_)
          · by_cases hro : r = io₁.out
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨hnout, _⟩
              · exact hflat rfl
              · exact hnout j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp k hok'
            rcases hcond with hflat | ⟨_, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp k (by rw [hoeq, hrp, hok'])
    · simp only [FlatAlloc.flattenState, if_neg hr]
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s2).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s2.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe2 r o ?_ ?_)
          · by_cases hro : r = io₁.out
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨hnout, _⟩
              · exact hflat rfl
              · exact hnout j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp k hok'
            rcases hcond with hflat | ⟨_, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp k (by rw [hoeq, hrp, hok'])
    · simp only [FlatAlloc.flattenState, if_neg hr]

end KernelIO₃

/-- IO signature of a **three-input / two-output** kernel (`₃ₓ₂` = "3 × 2").
The five tile lengths are independent; `f` returns the two output tiles as a
pair, in field order (`out1`, `out2`). The frame guarantee covers every cell
outside the **union** of the two output windows — this is why a two-output
kernel needs its own combinator rather than two one-output statements, whose
frames would each (falsely) claim the other output untouched. -/
structure KernelIO₃ₓ₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- First input buffer. -/
  in1 : RegionName
  /-- Second input buffer. -/
  in2 : RegionName
  /-- Third input buffer. -/
  in3 : RegionName
  /-- First output buffer. -/
  out1 : RegionName
  /-- Second output buffer. -/
  out2 : RegionName
  /-- First input tile length. -/
  B1 : Nat
  /-- Second input tile length. -/
  B2 : Nat
  /-- Third input tile length. -/
  B3 : Nat
  /-- First output tile length. -/
  Bout1 : Nat
  /-- Second output tile length. -/
  Bout2 : Nat
  /-- Where program `pid` reads its `in1` tile: `[read1 pid, read1 pid + B1)`. -/
  read1 : Nat → Nat
  /-- Where program `pid` reads its `in2` tile. -/
  read2 : Nat → Nat
  /-- Where program `pid` reads its `in3` tile. -/
  read3 : Nat → Nat
  /-- Where program `pid` writes its `out1` tile. -/
  write1 : Nat → Nat
  /-- Where program `pid` writes its `out2` tile. -/
  write2 : Nat → Nat

namespace KernelIO₃ₓ₂

/-- `io.Implements f` — three-input / two-output sibling of
`KernelIO₂.Implements`. The postcondition asserts both output windows; the
frame covers every cell outside their union. -/
def Implements (io : KernelIO₃ₓ₂)
    (f : (Fin io.B1 → ℝ) → (Fin io.B2 → ℝ) → (Fin io.B3 → ℝ) →
      (Fin io.Bout1 → ℝ) × (Fin io.Bout2 → ℝ)) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.in3, io.out1, io.out2] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io.read1 pid + io.B1 ≤ A.extent io.in1 →
    io.read2 pid + io.B2 ≤ A.extent io.in2 →
    io.read3 pid + io.B3 ≤ A.extent io.in3 →
    io.write1 pid + io.Bout1 ≤ A.extent io.out1 →
    io.write2 pid + io.Bout2 ≤ A.extent io.out2 →
  ∀ (xs : Fin io.B1 → ℝ) (ys : Fin io.B2 → ℝ) (zs : Fin io.B3 → ℝ)
      (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B1, s₀.readMem io.in1 (io.read1 pid + j.val) = xs j) →
    (∀ j : Fin io.B2, s₀.readMem io.in2 (io.read2 pid + j.val) = ys j) →
    (∀ j : Fin io.B3, s₀.readMem io.in3 (io.read3 pid + j.val) = zs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.Bout1,
          s'.readMem A.flat (A.addr io.out1 (io.write1 pid + j.val))
            = (f xs ys zs).1 j)
      ∧ (∀ j : Fin io.Bout2,
          s'.readMem A.flat (A.addr io.out2 (io.write2 pid + j.val))
            = (f xs ys zs).2 j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.Bout1,
                o' ≠ A.addr io.out1 (io.write1 pid + j.val)) ∧
             (∀ j : Fin io.Bout2,
                o' ≠ A.addr io.out2 (io.write2 pid + j.val)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => KernelIO₃ₓ₂.Implements

/-- Assembly lemma — three-input / two-output sibling of
`KernelIO₂.Implements.intro`. The region-model triple `hrun` takes the frame
as two window conditions (one per output); a cell is untouched when it avoids
both windows. -/
theorem Implements.intro (io : KernelIO₃ₓ₂)
    {f : (Fin io.B1 → ℝ) → (Fin io.B2 → ℝ) → (Fin io.B3 → ℝ) →
      (Fin io.Bout1 → ℝ) × (Fin io.Bout2 → ℝ)}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      io.read1 s.pid + io.B1 ≤ bounds io.in1 →
      io.read2 s.pid + io.B2 ≤ bounds io.in2 →
      io.read3 s.pid + io.B3 ≤ bounds io.in3 →
      io.write1 s.pid + io.Bout1 ≤ bounds io.out1 →
      io.write2 s.pid + io.Bout2 ≤ bounds io.out2 →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.B1 → ℝ) (ys : Fin io.B2 → ℝ)
        (zs : Fin io.B3 → ℝ),
      (∀ j : Fin io.B1, s₀.readMem io.in1 (io.read1 s₀.pid + j.val) = xs j) →
      (∀ j : Fin io.B2, s₀.readMem io.in2 (io.read2 s₀.pid + j.val) = ys j) →
      (∀ j : Fin io.B3, s₀.readMem io.in3 (io.read3 s₀.pid + j.val) = zs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.Bout1,
            s1.readMem io.out1 (io.write1 s₀.pid + j.val) = (f xs ys zs).1 j)
        ∧ (∀ j : Fin io.Bout2,
            s1.readMem io.out2 (io.write2 s₀.pid + j.val) = (f xs ys zs).2 j)
        ∧ (∀ r o,
            (r ≠ io.out1 ∨
              ∀ j : Fin io.Bout1, o ≠ io.write1 s₀.pid + j.val) →
            (r ≠ io.out2 ∨
              ∀ j : Fin io.Bout2, o ≠ io.write2 s₀.pid + j.val) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  intro A hd hregs hcov pid h1 h2 h3 h4 h5 xs ys zs s₀ hpid hu hx hy hz
  subst hpid
  obtain ⟨s1, hexec, hval1, hval2, hframe⟩ := hrun s₀ xs ys zs hx hy hz
  have hts' : Kernel.TraceSafe A.extent (io.kernel.toAlgKernel) s₀ :=
    hts A.extent s₀ h1 h2 h3 h4 h5
  have hbridge := A.exec_flatten hd hcov _ s₀ hts' hok hu
  refine ⟨A.flattenState s1, ?_, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j
    have hmem : io.out1 ∈ A.regions := by rw [hregs]; simp
    have hlt : io.write1 s₀.pid + j.val < A.extent io.out1 := by
      have := j.isLt; omega
    rw [A.flattenState_readMem hd s1 hmem hlt]
    exact hval1 j
  · intro j
    have hmem : io.out2 ∈ A.regions := by rw [hregs]; simp
    have hlt : io.write2 s₀.pid + j.val < A.extent io.out2 := by
      have := j.isLt; omega
    rw [A.flattenState_readMem hd s1 hmem hlt]
    exact hval2 j
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ?_ ?_)
          · by_cases hro : r = io.out1
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨hn1, _⟩
              · exact hflat rfl
              · exact hn1 j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · by_cases hro : r = io.out2
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨_, hn2⟩
              · exact hflat rfl
              · exact hn2 j (by rw [hoeq, hoj])
            · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

end KernelIO₃ₓ₂

/-- IO signature of a **masked three-input / two-output** kernel, with the
allocation list decoupled from the argument roles so that **in-place
updates** are expressible: `bufs` lists every buffer exactly once, and the
role fields point into it — an update kernel declares the same buffer as
both an input and an output (e.g. an optimizer step reading and rewriting
its parameter buffer, `bufs = [p, grad, m]`, `out1 = in1 = p`). Uniform tile
length `B`, active lanes per `mask` as in `MaskedKernelIO₂`: inactive lanes
carry no obligations on either side of the triple. -/
structure MaskedKernelIO₃ₓ₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The allocation list: every buffer the kernel touches, each exactly
  once. The role fields below point into this list; for an in-place kernel
  an output names the same buffer as an input. -/
  bufs : List RegionName
  /-- First input buffer. -/
  in1 : RegionName
  /-- Second input buffer. -/
  in2 : RegionName
  /-- Third input buffer. -/
  in3 : RegionName
  /-- First output buffer (may coincide with an input buffer). -/
  out1 : RegionName
  /-- Second output buffer (may coincide with an input buffer). -/
  out2 : RegionName
  /-- Tile length: each program instance owns `B`-element windows. -/
  B : Nat
  /-- Where program `pid` reads its `in1` tile: active lanes of
  `[read1 pid, read1 pid + B)`. -/
  read1 : Nat → Nat
  /-- Where program `pid` reads its `in2` tile. -/
  read2 : Nat → Nat
  /-- Where program `pid` reads its `in3` tile. -/
  read3 : Nat → Nat
  /-- Where program `pid` writes its `out1` tile. -/
  write1 : Nat → Nat
  /-- Where program `pid` writes its `out2` tile. -/
  write2 : Nat → Nat
  /-- Program `pid`'s active lanes. Only these read, write, or carry spec
  content; the rest of the window is dead. -/
  mask : Nat → Fin B → Prop

namespace MaskedKernelIO₃ₓ₂

/-- `io.Implements f` — masked three-input / two-output combinator. The
precondition loads the three input windows at active lanes **of the launch
state**; for an in-place kernel `f` therefore receives the *old* contents of
an updated buffer, and the postcondition asserts its *new* contents — the
standard before/after reading of a Hoare triple. Frame: every cell outside
the union of the two active output windows is untouched. -/
def Implements (io : MaskedKernelIO₃ₓ₂)
    (f : (Fin io.B → ℝ) → (Fin io.B → ℝ) → (Fin io.B → ℝ) →
      (Fin io.B → ℝ) × (Fin io.B → ℝ)) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = io.bufs →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    (∀ j : Fin io.B, io.mask pid j → io.read1 pid + j.val < A.extent io.in1) →
    (∀ j : Fin io.B, io.mask pid j → io.read2 pid + j.val < A.extent io.in2) →
    (∀ j : Fin io.B, io.mask pid j → io.read3 pid + j.val < A.extent io.in3) →
    (∀ j : Fin io.B, io.mask pid j → io.write1 pid + j.val < A.extent io.out1) →
    (∀ j : Fin io.B, io.mask pid j → io.write2 pid + j.val < A.extent io.out2) →
  ∀ (xs ys zs : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, io.mask pid j →
      s₀.readMem io.in1 (io.read1 pid + j.val) = xs j) →
    (∀ j : Fin io.B, io.mask pid j →
      s₀.readMem io.in2 (io.read2 pid + j.val) = ys j) →
    (∀ j : Fin io.B, io.mask pid j →
      s₀.readMem io.in3 (io.read3 pid + j.val) = zs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.mask pid j →
          s'.readMem A.flat (A.addr io.out1 (io.write1 pid + j.val))
            = (f xs ys zs).1 j)
      ∧ (∀ j : Fin io.B, io.mask pid j →
          s'.readMem A.flat (A.addr io.out2 (io.write2 pid + j.val))
            = (f xs ys zs).2 j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.mask pid j →
                o' ≠ A.addr io.out1 (io.write1 pid + j.val)) ∧
             (∀ j : Fin io.B, io.mask pid j →
                o' ≠ A.addr io.out2 (io.write2 pid + j.val)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => MaskedKernelIO₃ₓ₂.Implements

/-- Assembly lemma — masked three-input / two-output sibling of
`MaskedKernelIO₂.Implements.intro`, plus the two membership side conditions
tying the output roles into the declared allocation list. -/
theorem Implements.intro (io : MaskedKernelIO₃ₓ₂)
    {f : (Fin io.B → ℝ) → (Fin io.B → ℝ) → (Fin io.B → ℝ) →
      (Fin io.B → ℝ) × (Fin io.B → ℝ)}
    (hout1 : io.out1 ∈ io.bufs) (hout2 : io.out2 ∈ io.bufs)
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ j : Fin io.B, io.mask s.pid j →
        io.read1 s.pid + j.val < bounds io.in1) →
      (∀ j : Fin io.B, io.mask s.pid j →
        io.read2 s.pid + j.val < bounds io.in2) →
      (∀ j : Fin io.B, io.mask s.pid j →
        io.read3 s.pid + j.val < bounds io.in3) →
      (∀ j : Fin io.B, io.mask s.pid j →
        io.write1 s.pid + j.val < bounds io.out1) →
      (∀ j : Fin io.B, io.mask s.pid j →
        io.write2 s.pid + j.val < bounds io.out2) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs ys zs : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask s₀.pid j →
        s₀.readMem io.in1 (io.read1 s₀.pid + j.val) = xs j) →
      (∀ j : Fin io.B, io.mask s₀.pid j →
        s₀.readMem io.in2 (io.read2 s₀.pid + j.val) = ys j) →
      (∀ j : Fin io.B, io.mask s₀.pid j →
        s₀.readMem io.in3 (io.read3 s₀.pid + j.val) = zs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B, io.mask s₀.pid j →
            s1.readMem io.out1 (io.write1 s₀.pid + j.val) = (f xs ys zs).1 j)
        ∧ (∀ j : Fin io.B, io.mask s₀.pid j →
            s1.readMem io.out2 (io.write2 s₀.pid + j.val) = (f xs ys zs).2 j)
        ∧ (∀ r o,
            (r ≠ io.out1 ∨
              ∀ j : Fin io.B, io.mask s₀.pid j →
                o ≠ io.write1 s₀.pid + j.val) →
            (r ≠ io.out2 ∨
              ∀ j : Fin io.B, io.mask s₀.pid j →
                o ≠ io.write2 s₀.pid + j.val) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  intro A hd hregs hcov pid h1 h2 h3 h4 h5 xs ys zs s₀ hpid hu hx hy hz
  subst hpid
  obtain ⟨s1, hexec, hval1, hval2, hframe⟩ := hrun s₀ xs ys zs hx hy hz
  have hts' : Kernel.TraceSafe A.extent (io.kernel.toAlgKernel) s₀ :=
    hts A.extent s₀ h1 h2 h3 h4 h5
  have hbridge := A.exec_flatten hd hcov _ s₀ hts' hok hu
  refine ⟨A.flattenState s1, ?_, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j hj
    have hmem : io.out1 ∈ A.regions := by rw [hregs]; exact hout1
    have hlt : io.write1 s₀.pid + j.val < A.extent io.out1 := h4 j hj
    rw [A.flattenState_readMem hd s1 hmem hlt]
    exact hval1 j hj
  · intro j hj
    have hmem : io.out2 ∈ A.regions := by rw [hregs]; exact hout2
    have hlt : io.write2 s₀.pid + j.val < A.extent io.out2 := h5 j hj
    rw [A.flattenState_readMem hd s1 hmem hlt]
    exact hval2 j hj
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ?_ ?_)
          · by_cases hro : r = io.out1
            · subst hro
              refine Or.inr fun j hj hoj => ?_
              rcases hcond with hflat | ⟨hn1, _⟩
              · exact hflat rfl
              · exact hn1 j hj (by rw [hoeq, hoj])
            · exact Or.inl hro
          · by_cases hro : r = io.out2
            · subst hro
              refine Or.inr fun j hj hoj => ?_
              rcases hcond with hflat | ⟨_, hn2⟩
              · exact hflat rfl
              · exact hn2 j hj (by rw [hoeq, hoj])
            · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

end MaskedKernelIO₃ₓ₂

/-- IO signature of a **one-input / two-output** kernel (`₁ₓ₂` = "1 × 2") —
e.g. a statistics kernel producing mean and variance from one tile. Carries
the same field vocabulary as the rest of the family. Its only relation so
far is `Equiv` (the kernel-equivalence surface); the correctness `⊨` form
can be added alongside when a showcase needs it. -/
structure KernelIO₁ₓ₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- Input buffer. -/
  inp : RegionName
  /-- First output buffer. -/
  out1 : RegionName
  /-- Second output buffer. -/
  out2 : RegionName
  /-- Input tile length. -/
  Bin : Nat
  /-- First output tile length. -/
  Bout1 : Nat
  /-- Second output tile length. -/
  Bout2 : Nat
  /-- Where program `pid` reads its input tile: `[read pid, read pid + Bin)`. -/
  read : Nat → Nat
  /-- Where program `pid` writes its `out1` tile. -/
  write1 : Nat → Nat
  /-- Where program `pid` writes its `out2` tile. -/
  write2 : Nat → Nat
  /-- This kernel's private working buffers (see `ScratchSpec`). Empty for
  kernels that stage nothing through memory. -/
  scratch : List ScratchSpec := []

namespace KernelIO₁ₓ₂

/-- `io₁ ≡[R] io₂` — kernel equivalence on a shared one-input / two-output
IO signature; sibling of `MaskedKernelIO₂.Equiv`. The interface is read from
`io₁` (instances share it by structure update); `io₂` contributes only its
`kernel` and its private `scratch`. Both output windows must agree; each
side frames outside the **union** of the two output windows ∪ its own
scratch. No input hypotheses: equal inputs are "the same `s₀`". -/
def Equiv (io₁ io₂ : KernelIO₁ₓ₂) (R : RoundingModel) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io₁.inp, io₁.out1, io₁.out2]
      ++ (io₁.scratch.map (·.buf) ++ io₂.scratch.map (·.buf)) →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io₁.read pid + io₁.Bin ≤ A.extent io₁.inp →
    io₁.write1 pid + io₁.Bout1 ≤ A.extent io₁.out1 →
    io₁.write2 pid + io₁.Bout2 ≤ A.extent io₁.out2 →
    (∀ p ∈ io₁.scratch, p.win pid + p.len ≤ A.extent p.buf) →
    (∀ p ∈ io₂.scratch, p.win pid + p.len ≤ A.extent p.buf) →
  ∀ s₀ : BlockState,
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    ∃ s₁ s₂,
      execR R (A.flattenKernel io₁.kernel.toAlgKernel) (A.flattenState s₀)
        = some s₁
      ∧ execR R (A.flattenKernel io₂.kernel.toAlgKernel) (A.flattenState s₀)
        = some s₂
      ∧ (∀ j : Fin io₁.Bout1,
          s₁.readMem A.flat (A.addr io₁.out1 (io₁.write1 pid + j.val))
            = s₂.readMem A.flat (A.addr io₁.out1 (io₁.write1 pid + j.val)))
      ∧ (∀ j : Fin io₁.Bout2,
          s₁.readMem A.flat (A.addr io₁.out2 (io₁.write2 pid + j.val))
            = s₂.readMem A.flat (A.addr io₁.out2 (io₁.write2 pid + j.val)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io₁.Bout1,
                o' ≠ A.addr io₁.out1 (io₁.write1 pid + j.val)) ∧
             (∀ j : Fin io₁.Bout2,
                o' ≠ A.addr io₁.out2 (io₁.write2 pid + j.val)) ∧
             (∀ p ∈ io₁.scratch, ∀ k : Fin p.len,
                o' ≠ A.addr p.buf (p.win pid + k.val)))) →
          s₁.mem r' o' = (A.flattenState s₀).mem r' o')
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io₁.Bout1,
                o' ≠ A.addr io₁.out1 (io₁.write1 pid + j.val)) ∧
             (∀ j : Fin io₁.Bout2,
                o' ≠ A.addr io₁.out2 (io₁.write2 pid + j.val)) ∧
             (∀ p ∈ io₂.scratch, ∀ k : Fin p.len,
                o' ≠ A.addr p.buf (p.win pid + k.val)))) →
          s₂.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io₁ " ≡[" R "] " io₂ =>
  KernelIO₁ₓ₂.Equiv io₁ io₂ R

/-- Assembly lemma for the one-input / two-output `≡[R]` — sibling of
`MaskedKernelIO₂.Equiv.intro`; see there for the reading of the
obligations. The region-model frames take THREE conditions: one per output
window, one for the respective kernel's scratch. -/
theorem Equiv.intro (io₁ io₂ : KernelIO₁ₓ₂) {R : RoundingModel}
    (hok₁ : (io₁.kernel.toAlgKernel).FlattenOk)
    (hok₂ : (io₂.kernel.toAlgKernel).FlattenOk)
    (hts₁ : ∀ (bounds : RegionBounds) (s : BlockState),
      io₁.read s.pid + io₁.Bin ≤ bounds io₁.inp →
      io₁.write1 s.pid + io₁.Bout1 ≤ bounds io₁.out1 →
      io₁.write2 s.pid + io₁.Bout2 ≤ bounds io₁.out2 →
      (∀ p ∈ io₁.scratch, p.win s.pid + p.len ≤ bounds p.buf) →
      (io₁.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hts₂ : ∀ (bounds : RegionBounds) (s : BlockState),
      io₁.read s.pid + io₁.Bin ≤ bounds io₁.inp →
      io₁.write1 s.pid + io₁.Bout1 ≤ bounds io₁.out1 →
      io₁.write2 s.pid + io₁.Bout2 ≤ bounds io₁.out2 →
      (∀ p ∈ io₂.scratch, p.win s.pid + p.len ≤ bounds p.buf) →
      (io₂.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ s₀ : BlockState,
      ∃ s1 s2,
        execR R (io₁.kernel.toAlgKernel) s₀ = some s1
        ∧ execR R (io₂.kernel.toAlgKernel) s₀ = some s2
        ∧ (∀ j : Fin io₁.Bout1,
            s1.readMem io₁.out1 (io₁.write1 s₀.pid + j.val)
              = s2.readMem io₁.out1 (io₁.write1 s₀.pid + j.val))
        ∧ (∀ j : Fin io₁.Bout2,
            s1.readMem io₁.out2 (io₁.write2 s₀.pid + j.val)
              = s2.readMem io₁.out2 (io₁.write2 s₀.pid + j.val))
        ∧ (∀ r o,
            (r ≠ io₁.out1 ∨
              ∀ j : Fin io₁.Bout1, o ≠ io₁.write1 s₀.pid + j.val) →
            (r ≠ io₁.out2 ∨
              ∀ j : Fin io₁.Bout2, o ≠ io₁.write2 s₀.pid + j.val) →
            (∀ p ∈ io₁.scratch, r = p.buf →
              ∀ k : Fin p.len, o ≠ p.win s₀.pid + k.val) →
            s1.mem r o = s₀.mem r o)
        ∧ (∀ r o,
            (r ≠ io₁.out1 ∨
              ∀ j : Fin io₁.Bout1, o ≠ io₁.write1 s₀.pid + j.val) →
            (r ≠ io₁.out2 ∨
              ∀ j : Fin io₁.Bout2, o ≠ io₁.write2 s₀.pid + j.val) →
            (∀ p ∈ io₂.scratch, r = p.buf →
              ∀ k : Fin p.len, o ≠ p.win s₀.pid + k.val) →
            s2.mem r o = s₀.mem r o)) :
    io₁.Equiv io₂ R := by
  intro A hd hregs hcov pid h1 h2 h3 hsc1 hsc2 s₀ hpid hu
  subst hpid
  obtain ⟨s1, s2, hexec1, hexec2, hval1, hval2, hframe1, hframe2⟩ := hrun s₀
  have hts₁' : (io₁.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts₁ A.extent s₀ h1 h2 h3 hsc1
  have hts₂' : (io₂.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts₂ A.extent s₀ h1 h2 h3 hsc2
  have hbridge1 := A.execR_flatten hd hcov R _ s₀ hts₁' hok₁ hu
  have hbridge2 := A.execR_flatten hd hcov R _ s₀ hts₂' hok₂ hu
  have hmem1 : io₁.out1 ∈ A.regions := by rw [hregs]; simp
  have hmem2 : io₁.out2 ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, A.flattenState s2, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hbridge1, hexec1, Option.map_some]
  · rw [hbridge2, hexec2, Option.map_some]
  · intro j
    have hlt : io₁.write1 s₀.pid + j.val < A.extent io₁.out1 := by
      have := j.isLt; omega
    rw [A.flattenState_readMem hd s1 hmem1 hlt,
        A.flattenState_readMem hd s2 hmem1 hlt]
    exact hval1 j
  · intro j
    have hlt : io₁.write2 s₀.pid + j.val < A.extent io₁.out2 := by
      have := j.isLt; omega
    rw [A.flattenState_readMem hd s1 hmem2 hlt,
        A.flattenState_readMem hd s2 hmem2 hlt]
    exact hval2 j
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe1 r o ?_ ?_ ?_)
          · by_cases hro : r = io₁.out1
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨hn1, _, _⟩
              · exact hflat rfl
              · exact hn1 j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · by_cases hro : r = io₁.out2
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨_, hn2, _⟩
              · exact hflat rfl
              · exact hn2 j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp k hok'
            rcases hcond with hflat | ⟨_, _, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp k (by rw [hoeq, hrp, hok'])
    · simp only [FlatAlloc.flattenState, if_neg hr]
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s2).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s2.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe2 r o ?_ ?_ ?_)
          · by_cases hro : r = io₁.out1
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨hn1, _, _⟩
              · exact hflat rfl
              · exact hn1 j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · by_cases hro : r = io₁.out2
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨_, hn2, _⟩
              · exact hflat rfl
              · exact hn2 j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp k hok'
            rcases hcond with hflat | ⟨_, _, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp k (by rw [hoeq, hrp, hok'])
    · simp only [FlatAlloc.flattenState, if_neg hr]

end KernelIO₁ₓ₂

end VeriTile.Triton
