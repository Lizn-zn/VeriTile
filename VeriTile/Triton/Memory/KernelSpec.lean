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
import VeriTile.Triton.Memory.KernelCore

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

/-- Embed into the unified core — proof plumbing for `Implements.intro`
(two float tile channels plus three 1-lane bound-witness channels — the
block bounds `w + B ≤ extent` carried as the masked per-lane bounds
`w + B - 1 < extent` gated on `0 < w + B` — one output, no scratch). -/
private def toU (io : KernelIO₂) : UKernelIO where
  kernel := io.kernel
  nIn := 5
  nOut := 1
  nScr := 0
  bufs := [io.in1, io.in2, io.out]
  ity := fun i => match i with
    | ⟨0, _⟩ => .float
    | ⟨1, _⟩ => .float
    | _ => .nat
  iarity := fun i => match i with
    | ⟨0, _⟩ => io.B
    | ⟨1, _⟩ => io.B
    | _ => 1
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | ⟨1, _⟩ => io.in2
    | ⟨2, _⟩ => io.in1
    | ⟨3, _⟩ => io.in2
    | _ => io.out
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ + j.val
    | ⟨1, _⟩ => fun j => io.read2 p₀ + j.val
    | ⟨2, _⟩ => fun _ => io.read1 p₀ + io.B - 1
    | ⟨3, _⟩ => fun _ => io.read2 p₀ + io.B - 1
    | _ => fun _ => io.write p₀ + io.B - 1
  imask := fun i _ p₀ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => True
    | ⟨2, _⟩ => fun _ => 0 < io.read1 p₀ + io.B
    | ⟨3, _⟩ => fun _ => 0 < io.read2 p₀ + io.B
    | _ => fun _ => 0 < io.write p₀ + io.B
  owin := fun _ _ p₀ _ j => io.write p₀ + j.val
  omask := fun _ _ _ _ _ => True
  swin := fun t => t.elim0
  smask := fun t => t.elim0

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
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun _p₀ _p₁ vals _o j =>
        f (fun j' => vals (⟨0, by decide⟩ : Fin 5) j')
          (fun j' => vals (⟨1, by decide⟩ : Fin 5) j') j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib _hob _hsb
      have hb1 : io.read1 (s.pids 0) + io.B ≤ bounds io.in1 := by
        by_cases hpos : 0 < io.read1 (s.pids 0) + io.B
        · have h : io.read1 (s.pids 0) + io.B - 1 < bounds io.in1 :=
            hib (⟨2, by decide⟩ : Fin 5) (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb2 : io.read2 (s.pids 0) + io.B ≤ bounds io.in2 := by
        by_cases hpos : 0 < io.read2 (s.pids 0) + io.B
        · have h : io.read2 (s.pids 0) + io.B - 1 < bounds io.in2 :=
            hib (⟨3, by decide⟩ : Fin 5) (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb3 : io.write (s.pids 0) + io.B ≤ bounds io.out := by
        by_cases hpos : 0 < io.write (s.pids 0) + io.B
        · have h : io.write (s.pids 0) + io.B - 1 < bounds io.out :=
            hib (⟨4, by decide⟩ : Fin 5) (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      exact hts bounds s hb1 hb2 hb3
    · intro s₀ vals hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨0, by decide⟩ : Fin 5) j)
          (fun j => vals (⟨1, by decide⟩ : Fin 5) j)
          (fun j => hpins (⟨0, by decide⟩ : Fin 5) j True.intro)
          (fun j => hpins (⟨1, by decide⟩ : Fin 5) j True.intro)
      refine ⟨s1, hexec, fun _o j _ => hval j, ?_⟩
      intro r o' hoc _hsc
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun j => ?_
        rcases hoc (⟨0, by decide⟩ : Fin 1) j True.intro with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid h1 h2 h3 xs ys s₀ hpid hu hx hy
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => ys
        | ⟨2, _⟩ => fun _ => ChanTy.read .nat s₀ io.in1 (io.read1 pid + io.B - 1)
        | ⟨3, _⟩ => fun _ => ChanTy.read .nat s₀ io.in2 (io.read2 pid + io.B - 1)
        | ⟨_+4, _⟩ => fun _ => ChanTy.read .nat s₀ io.out (io.write pid + io.B - 1))
      s₀ hpid rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j _ => by
            have hj : j.val < io.B := j.isLt
            have h : io.read1 pid + j.val < A.extent io.in1 := by omega
            exact h
        | ⟨1, _⟩ => fun j _ => by
            have hj : j.val < io.B := j.isLt
            have h : io.read2 pid + j.val < A.extent io.in2 := by omega
            exact h
        | ⟨2, _⟩ => fun _ hm => by
            have hm' : 0 < io.read1 pid + io.B := hm
            have h : io.read1 pid + io.B - 1 < A.extent io.in1 := by omega
            exact h
        | ⟨3, _⟩ => fun _ hm => by
            have hm' : 0 < io.read2 pid + io.B := hm
            have h : io.read2 pid + io.B - 1 < A.extent io.in2 := by omega
            exact h
        | ⟨_+4, _⟩ => fun _ hm => by
            have hm' : 0 < io.write pid + io.B := hm
            have h : io.write pid + io.B - 1 < A.extent io.out := by omega
            exact h)
      (fun _o j _ => by
        have hj : j.val < io.B := j.isLt
        have h : io.write pid + j.val < A.extent io.out := by omega
        exact h)
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j _ => hx j
        | ⟨1, _⟩ => fun j _ => hy j
        | ⟨2, _⟩ => fun _ _ => rfl
        | ⟨3, _⟩ => fun _ _ => rfl
        | ⟨_+4, _⟩ => fun _ _ => rfl)
  refine ⟨s', hexec, fun j => hval (⟨0, by decide⟩ : Fin 1) j True.intro, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | hout
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j _ => hout j, fun t => t.elim0⟩

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

/-- Embed into the unified core — proof plumbing for `Implements.intro`
(two float channels, one output, scratch as contract-free channels; every
window is lane-masked by `mask`, so no bound-witness channels are needed). -/
private def toU (io : MaskedKernelIO₂) : UKernelIO where
  kernel := io.kernel
  nIn := 2
  nOut := 1
  nScr := io.scratch.length
  bufs := [io.in1, io.in2, io.out] ++ io.scratch.map Prod.fst
  ity := fun _ => .float
  iarity := fun _ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | _ => io.in2
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun _ => io.B
  sbuf := fun t => (io.scratch.get t).1
  iwin := fun i _ p₀ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ + j.val
    | _ => fun j => io.read2 p₀ + j.val
  imask := fun _ _ p₀ _ j => io.mask p₀ j
  owin := fun _ _ p₀ _ j => io.write p₀ + j.val
  omask := fun _ _ p₀ _ j => io.mask p₀ j
  swin := fun t _ p₀ _ j => (io.scratch.get t).2 p₀ + j.val
  smask := fun _ _ p₀ _ j => io.mask p₀ j

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
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun _p₀ _p₁ vals _o j =>
        f (fun j' => vals (⟨0, by decide⟩ : Fin 2) j')
          (fun j' => vals (⟨1, by decide⟩ : Fin 2) j') j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob hsb
      refine hts bounds s (fun j hj => hib (⟨0, by decide⟩ : Fin 2) j hj)
        (fun j hj => hib (⟨1, by decide⟩ : Fin 2) j hj)
        (fun j hj => hob (⟨0, by decide⟩ : Fin 1) j hj) ?_
      intro q hq j hj
      obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
      have h : (io.scratch.get u).2 (s.pids 0) + j.val
          < bounds (io.scratch.get u).1 := hsb u j hj
      rw [hu] at h
      exact h
    · intro s₀ vals hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨0, by decide⟩ : Fin 2) j)
          (fun j => vals (⟨1, by decide⟩ : Fin 2) j)
          (fun j hj => hpins (⟨0, by decide⟩ : Fin 2) j hj)
          (fun j hj => hpins (⟨1, by decide⟩ : Fin 2) j hj)
      refine ⟨s1, hexec, fun _o j hj => hval j hj, ?_⟩
      intro r o' hoc hsc'
      refine hframe r o' ?_ ?_
      · by_cases hro : r = io.out
        · subst hro
          refine Or.inr fun j hj => ?_
          rcases hoc (⟨0, by decide⟩ : Fin 1) j hj with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · intro q hq hrq j hj
        obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
        have h : r ≠ (io.scratch.get u).1 ∨
            o' ≠ (io.scratch.get u).2 (s₀.pids 0) + j.val :=
          hsc' u j hj
        rw [hu] at h
        rcases h with hne | hno
        · exact absurd hrq hne
        · exact hno
  intro A hd hregs hcov pid h1 h2 h3 hsc xs ys s₀ hpid hu hx hy
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | _ => ys)
      s₀ hpid rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => h1 j hj
        | ⟨_+1, _⟩ => fun j hj => h2 j hj)
      (fun _o j hj => h3 j hj)
      (fun t j hj => hsc (io.scratch.get t) (io.scratch.get_mem t) j hj)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hx j hj
        | ⟨_+1, _⟩ => fun j hj => hy j hj)
  refine ⟨s', hexec, fun j hj => hval (⟨0, by decide⟩ : Fin 1) j hj, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hout, hscr⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j hj => hout j hj,
      fun t j hj => hscr (io.scratch.get t) (io.scratch.get_mem t) j hj⟩

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

/-- IO signature of a **masked** one-input / one-output kernel — the
one-input sibling of `MaskedKernelIO₂` (elementwise maps: relu, sin,
square, …). Each program instance owns a `B`-lane window but only its
**active** lanes touch memory; inactive lanes carry no obligations on
either side of the Hoare triple. The read side and the write side may have
**different** active sets (`mask` vs `writeMask`, e.g. a `pid == 0` store
gate over an ungated load); for the common symmetric case `writeMask`
defaults to `mask`. -/
structure MaskedKernelIO₁ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- Input buffer. -/
  inp : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Tile length: each program instance owns `B`-element windows. -/
  B : Nat
  /-- Where program `pid` reads its input tile: active lanes of
  `[read pid, read pid + B)`. -/
  read : Nat → Nat
  /-- Where program `pid` writes its output tile. -/
  write : Nat → Nat
  /-- Program `pid`'s **read-active** lanes: the lanes whose input cells the
  precondition constrains (and whose read addresses must be in bounds). -/
  mask : Nat → Fin B → Prop
  /-- Program `pid`'s **write-active** lanes: the lanes the postcondition
  asserts output values at (and whose write addresses must be in bounds);
  the frame holds everywhere else. Defaults to `mask` — override only for
  kernels whose store is gated more tightly than their load. -/
  writeMask : Nat → Fin B → Prop := mask
  /-- This kernel's **private working buffers**, each with its per-program
  window start (lane-masked by `writeMask`, tile length `B`, like the
  output); see `MaskedKernelIO₂.scratch`. -/
  scratch : List (RegionName × (Nat → Nat)) := []

namespace MaskedKernelIO₁

/-- `io.Implements f` — one-input sibling of `MaskedKernelIO₂.Implements`.
Full Hoare triple, restricted to the active lanes: window-in-bounds
contract, loaded-input precondition, and output-value postcondition all
**lane-wise at active lanes only**; frame everywhere outside the active
output and scratch lanes. -/
def Implements (io : MaskedKernelIO₁)
    (f : (Fin io.B → ℝ) → Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp, io.out] ++ io.scratch.map Prod.fst →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    (∀ j : Fin io.B, io.mask pid j → io.read pid + j.val < A.extent io.inp) →
    (∀ j : Fin io.B, io.writeMask pid j →
      io.write pid + j.val < A.extent io.out) →
    (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.writeMask pid j →
      p.2 pid + j.val < A.extent p.1) →
  ∀ (xs : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, io.mask pid j →
      s₀.readMem io.inp (io.read pid + j.val) = xs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask pid j →
          s'.readMem A.flat (A.addr io.out (io.write pid + j.val))
            = f xs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.writeMask pid j →
                o' ≠ A.addr io.out (io.write pid + j.val)) ∧
             (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.writeMask pid j →
                o' ≠ A.addr p.1 (p.2 pid + j.val)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => MaskedKernelIO₁.Implements

/-- Embed into the unified core — proof plumbing for `Implements.intro`
(one float channel, one output, scratch as contract-free channels; the
windows are lane-masked, reads by `mask` and writes by `writeMask`). -/
private def toU (io : MaskedKernelIO₁) : UKernelIO where
  kernel := io.kernel
  nIn := 1
  nOut := 1
  nScr := io.scratch.length
  bufs := [io.inp, io.out] ++ io.scratch.map Prod.fst
  ity := fun _ => .float
  iarity := fun _ => io.B
  ibuf := fun _ => io.inp
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun _ => io.B
  sbuf := fun t => (io.scratch.get t).1
  iwin := fun _ _ p₀ _ j => io.read p₀ + j.val
  imask := fun _ _ p₀ _ j => io.mask p₀ j
  owin := fun _ _ p₀ _ j => io.write p₀ + j.val
  omask := fun _ _ p₀ _ j => io.writeMask p₀ j
  swin := fun t _ p₀ _ j => (io.scratch.get t).2 p₀ + j.val
  smask := fun _ _ p₀ _ j => io.writeMask p₀ j

/-- Assembly lemma — one-input sibling of `MaskedKernelIO₂.Implements.intro`;
see there for the reading of the lane-wise obligations. -/
theorem Implements.intro (io : MaskedKernelIO₁)
    {f : (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ j : Fin io.B, io.mask s.pid j →
        io.read s.pid + j.val < bounds io.inp) →
      (∀ j : Fin io.B, io.writeMask s.pid j →
        io.write s.pid + j.val < bounds io.out) →
      (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.writeMask s.pid j →
        p.2 s.pid + j.val < bounds p.1) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask s₀.pid j →
        s₀.readMem io.inp (io.read s₀.pid + j.val) = xs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B, io.writeMask s₀.pid j →
            s1.readMem io.out (io.write s₀.pid + j.val) = f xs j)
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.B, io.writeMask s₀.pid j →
                o ≠ io.write s₀.pid + j.val) →
            (∀ p ∈ io.scratch, r = p.1 →
              ∀ j : Fin io.B, io.writeMask s₀.pid j →
                o ≠ p.2 s₀.pid + j.val) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun _p₀ _p₁ vals _o j =>
        f (fun j' => vals (⟨0, by decide⟩ : Fin 1) j') j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob hsb
      refine hts bounds s (fun j hj => hib (⟨0, by decide⟩ : Fin 1) j hj)
        (fun j hj => hob (⟨0, by decide⟩ : Fin 1) j hj) ?_
      intro q hq j hj
      obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
      have h : (io.scratch.get u).2 (s.pids 0) + j.val
          < bounds (io.scratch.get u).1 := hsb u j hj
      rw [hu] at h
      exact h
    · intro s₀ vals hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨0, by decide⟩ : Fin 1) j)
          (fun j hj => hpins (⟨0, by decide⟩ : Fin 1) j hj)
      refine ⟨s1, hexec, fun _o j hj => hval j hj, ?_⟩
      intro r o' hoc hsc'
      refine hframe r o' ?_ ?_
      · by_cases hro : r = io.out
        · subst hro
          refine Or.inr fun j hj => ?_
          rcases hoc (⟨0, by decide⟩ : Fin 1) j hj with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · intro q hq hrq j hj
        obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
        have h : r ≠ (io.scratch.get u).1 ∨
            o' ≠ (io.scratch.get u).2 (s₀.pids 0) + j.val :=
          hsc' u j hj
        rw [hu] at h
        rcases h with hne | hno
        · exact absurd hrq hne
        · exact hno
  intro A hd hregs hcov pid h1 h2 hsc xs s₀ hpid hu hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1) (fun _ => xs) s₀ hpid rfl hu
      (fun _i j hj => h1 j hj) (fun _o j hj => h2 j hj)
      (fun t j hj => hsc (io.scratch.get t) (io.scratch.get_mem t) j hj)
      (fun _i j hj => hx j hj)
  refine ⟨s', hexec, fun j hj => hval (⟨0, by decide⟩ : Fin 1) j hj, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hout, hscr⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j hj => hout j hj,
      fun t j hj => hscr (io.scratch.get t) (io.scratch.get_mem t) j hj⟩

end MaskedKernelIO₁

/-- IO signature of a **2D-grid, general-window** masked one-input /
one-output kernel — the two-axis sibling of `MaskedKernelIO₁` for kernels
that read `tl.program_id(1)` and/or address their lanes non-contiguously.
Two generalizations over the 1D family, both forced by real ports:

* **two program-id axes**: every field takes `(pid₀ pid₁ : Nat)`, and
  `Implements` pins **both** `s₀.pids 0` and `s₀.pids 1` (the 1D family
  leaves `pids 1` universally free, which falsifies any 2D kernel's ⊨);
* **per-lane windows**: `read`/`write` give lane `j`'s full address
  directly (`Nat → Nat → Fin B → Nat`), so strided rows
  (`base + j * stride_c`), block offsets (`i_d * B + j`), and scalar cells
  are all expressible — the 1D family's contiguous `base pid + j` is the
  special case `fun pid₀ _ j => base pid₀ + j.val`. -/
structure Masked2DKernelIO₁ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- Input buffer. -/
  inp : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Tile length: each program instance owns `B`-lane windows. -/
  B : Nat
  /-- Lane `j`'s read address for program `(pid₀, pid₁)`. -/
  read : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s write address for program `(pid₀, pid₁)`. -/
  write : Nat → Nat → Fin B → Nat
  /-- Program `(pid₀, pid₁)`'s **read-active** lanes. -/
  mask : Nat → Nat → Fin B → Prop
  /-- Program `(pid₀, pid₁)`'s **write-active** lanes; defaults to `mask`. -/
  writeMask : Nat → Nat → Fin B → Prop := mask
  /-- Private working buffers with per-lane windows (masked by
  `writeMask`); see `MaskedKernelIO₂.scratch`. -/
  scratch : List (RegionName × (Nat → Nat → Fin B → Nat)) := []

namespace Masked2DKernelIO₁

/-- `io.Implements f` — two-axis, general-window sibling of
`MaskedKernelIO₁.Implements`. Same lane-wise masked Hoare triple; the
launch state pins both program-id axes, and every address is the
signature's per-lane map evaluated at `(pid₀, pid₁, j)`. The spec `f`
takes both pids: on a tiled axis a per-block reduction's value is
irreducibly block-dependent (a full block computes a different function
than the tail block, and an all-masked block stores the finite fallback),
so a pid-independent spec would be falsifiable. Pid-independent kernels
simply ignore the two arguments. -/
def Implements (io : Masked2DKernelIO₁)
    (f : Nat → Nat → (Fin io.B → ℝ) → Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp, io.out] ++ io.scratch.map Prod.fst →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.read pid₀ pid₁ j < A.extent io.inp) →
    (∀ j : Fin io.B, io.writeMask pid₀ pid₁ j →
      io.write pid₀ pid₁ j < A.extent io.out) →
    (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.writeMask pid₀ pid₁ j →
      p.2 pid₀ pid₁ j < A.extent p.1) →
  ∀ (xs : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMem io.inp (io.read pid₀ pid₁ j) = xs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask pid₀ pid₁ j →
          s'.readMem A.flat (A.addr io.out (io.write pid₀ pid₁ j))
            = f pid₀ pid₁ xs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.writeMask pid₀ pid₁ j →
                o' ≠ A.addr io.out (io.write pid₀ pid₁ j)) ∧
             (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.writeMask pid₀ pid₁ j →
                o' ≠ A.addr p.1 (p.2 pid₀ pid₁ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => Masked2DKernelIO₁.Implements

/-- Embed into the unified core — proof plumbing for `Implements.intro`
(one float channel, one output, scratch as contract-free channels). -/
private def toU (io : Masked2DKernelIO₁) : UKernelIO where
  kernel := io.kernel
  nIn := 1
  nOut := 1
  nScr := io.scratch.length
  bufs := [io.inp, io.out] ++ io.scratch.map Prod.fst
  ity := fun _ => .float
  iarity := fun _ => io.B
  ibuf := fun _ => io.inp
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun _ => io.B
  sbuf := fun t => (io.scratch.get t).1
  iwin := fun _ _ p₀ p₁ j => io.read p₀ p₁ j
  imask := fun _ _ p₀ p₁ j => io.mask p₀ p₁ j
  owin := fun _ _ p₀ p₁ j => io.write p₀ p₁ j
  omask := fun _ _ p₀ p₁ j => io.writeMask p₀ p₁ j
  swin := fun t _ p₀ p₁ j => (io.scratch.get t).2 p₀ p₁ j
  smask := fun _ _ p₀ p₁ j => io.writeMask p₀ p₁ j

/-- Assembly lemma — two-axis sibling of `MaskedKernelIO₁.Implements.intro`;
the obligations' lane hypotheses are indexed by `(s.pids 0, s.pids 1)`. -/
theorem Implements.intro (io : Masked2DKernelIO₁)
    {f : Nat → Nat → (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.read (s.pids 0) (s.pids 1) j < bounds io.inp) →
      (∀ j : Fin io.B, io.writeMask (s.pids 0) (s.pids 1) j →
        io.write (s.pids 0) (s.pids 1) j < bounds io.out) →
      (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.writeMask (s.pids 0) (s.pids 1) j →
        p.2 (s.pids 0) (s.pids 1) j < bounds p.1) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) j) = xs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
            s1.readMem io.out (io.write (s₀.pids 0) (s₀.pids 1) j)
              = f (s₀.pids 0) (s₀.pids 1) xs j)
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) j) →
            (∀ p ∈ io.scratch, r = p.1 →
              ∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
                o ≠ p.2 (s₀.pids 0) (s₀.pids 1) j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j => f p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 1) j') j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob hsb
      refine hts bounds s (fun j hj => hib (⟨0, by decide⟩ : Fin 1) j hj)
        (fun j hj => hob (⟨0, by decide⟩ : Fin 1) j hj) ?_
      intro q hq j hj
      obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
      have h : (io.scratch.get u).2 (s.pids 0) (s.pids 1) j
          < bounds (io.scratch.get u).1 := hsb u j hj
      rw [hu] at h
      exact h
    · intro s₀ vals hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨0, by decide⟩ : Fin 1) j)
          (fun j hj => hpins (⟨0, by decide⟩ : Fin 1) j hj)
      refine ⟨s1, hexec, fun _o j hj => hval j hj, ?_⟩
      intro r o' hoc hsc'
      refine hframe r o' ?_ ?_
      · by_cases hro : r = io.out
        · subst hro
          refine Or.inr fun j hj => ?_
          rcases hoc (⟨0, by decide⟩ : Fin 1) j hj with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · intro q hq hrq j hj
        obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
        have h : r ≠ (io.scratch.get u).1 ∨
            o' ≠ (io.scratch.get u).2 (s₀.pids 0) (s₀.pids 1) j :=
          hsc' u j hj
        rw [hu] at h
        rcases h with hne | hno
        · exact absurd hrq hne
        · exact hno
  intro A hd hregs hcov pid₀ pid₁ h1 h2 hsc xs s₀ hpid₀ hpid₁ hu hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (fun _ => xs) s₀ hpid₀ hpid₁ hu
      (fun _i j hj => h1 j hj) (fun _o j hj => h2 j hj)
      (fun t j hj => hsc (io.scratch.get t) (io.scratch.get_mem t) j hj)
      (fun _i j hj => hx j hj)
  refine ⟨s', hexec, fun j hj => hval (⟨0, by decide⟩ : Fin 1) j hj, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hout, hscr⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j hj => hout j hj,
      fun t j hj => hscr (io.scratch.get t) (io.scratch.get_mem t) j hj⟩
end Masked2DKernelIO₁

/-- IO signature of a **2D-grid, general-window** masked two-input /
one-output kernel — the two-input sibling of `Masked2DKernelIO₁` (see there
for the two generalizations over the 1D family). -/
structure Masked2DKernelIO₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- First input buffer. -/
  in1 : RegionName
  /-- Second input buffer. -/
  in2 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Tile length: each program instance owns `B`-lane windows. -/
  B : Nat
  /-- Lane `j`'s `in1` read address for program `(pid₀, pid₁)`. -/
  read1 : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `in2` read address. -/
  read2 : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s write address. -/
  write : Nat → Nat → Fin B → Nat
  /-- Program `(pid₀, pid₁)`'s **read-active** lanes. -/
  mask : Nat → Nat → Fin B → Prop
  /-- Program `(pid₀, pid₁)`'s **`in2` read-active** lanes; defaults to
  `mask`. For kernels whose second input is read under a different gate
  than the first — e.g. an unmasked broadcast scalar read while `in1` is
  tail-masked; the default keeps symmetric consumers unchanged. -/
  read2Mask : Nat → Nat → Fin B → Prop := mask
  /-- Program `(pid₀, pid₁)`'s **write-active** lanes; defaults to `mask`. -/
  writeMask : Nat → Nat → Fin B → Prop := mask
  /-- Private working buffers with per-lane windows (masked by
  `writeMask`). -/
  scratch : List (RegionName × (Nat → Nat → Fin B → Nat)) := []

namespace Masked2DKernelIO₂

/-- `io.Implements f` — two-input sibling of
`Masked2DKernelIO₁.Implements`. -/
def Implements (io : Masked2DKernelIO₂)
    (f : Nat → Nat → (Fin io.B → ℝ) → (Fin io.B → ℝ) → Fin io.B → ℝ) :
    Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.out] ++ io.scratch.map Prod.fst →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.read1 pid₀ pid₁ j < A.extent io.in1) →
    (∀ j : Fin io.B, io.read2Mask pid₀ pid₁ j →
      io.read2 pid₀ pid₁ j < A.extent io.in2) →
    (∀ j : Fin io.B, io.writeMask pid₀ pid₁ j →
      io.write pid₀ pid₁ j < A.extent io.out) →
    (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.writeMask pid₀ pid₁ j →
      p.2 pid₀ pid₁ j < A.extent p.1) →
  ∀ (xs ys : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMem io.in1 (io.read1 pid₀ pid₁ j) = xs j) →
    (∀ j : Fin io.B, io.read2Mask pid₀ pid₁ j →
      s₀.readMem io.in2 (io.read2 pid₀ pid₁ j) = ys j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask pid₀ pid₁ j →
          s'.readMem A.flat (A.addr io.out (io.write pid₀ pid₁ j))
            = f pid₀ pid₁ xs ys j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.writeMask pid₀ pid₁ j →
                o' ≠ A.addr io.out (io.write pid₀ pid₁ j)) ∧
             (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.writeMask pid₀ pid₁ j →
                o' ≠ A.addr p.1 (p.2 pid₀ pid₁ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => Masked2DKernelIO₂.Implements

/-- Embed into the unified core — proof plumbing for `Implements.intro`
(two float channels with per-channel read gates, one output, scratch as
contract-free channels). -/
private def toU (io : Masked2DKernelIO₂) : UKernelIO where
  kernel := io.kernel
  nIn := 2
  nOut := 1
  nScr := io.scratch.length
  bufs := [io.in1, io.in2, io.out] ++ io.scratch.map Prod.fst
  ity := fun _ => .float
  iarity := fun _ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | _ => io.in2
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun _ => io.B
  sbuf := fun t => (io.scratch.get t).1
  iwin := fun i _ p₀ p₁ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ p₁ j
    | _ => fun j => io.read2 p₀ p₁ j
  imask := fun i _ p₀ p₁ => match i with
    | ⟨0, _⟩ => fun j => io.mask p₀ p₁ j
    | _ => fun j => io.read2Mask p₀ p₁ j
  owin := fun _ _ p₀ p₁ j => io.write p₀ p₁ j
  omask := fun _ _ p₀ p₁ j => io.writeMask p₀ p₁ j
  swin := fun t _ p₀ p₁ j => (io.scratch.get t).2 p₀ p₁ j
  smask := fun _ _ p₀ p₁ j => io.writeMask p₀ p₁ j

/-- Assembly lemma — two-input sibling of
`Masked2DKernelIO₁.Implements.intro`. -/
theorem Implements.intro (io : Masked2DKernelIO₂)
    {f : Nat → Nat → (Fin io.B → ℝ) → (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.read1 (s.pids 0) (s.pids 1) j < bounds io.in1) →
      (∀ j : Fin io.B, io.read2Mask (s.pids 0) (s.pids 1) j →
        io.read2 (s.pids 0) (s.pids 1) j < bounds io.in2) →
      (∀ j : Fin io.B, io.writeMask (s.pids 0) (s.pids 1) j →
        io.write (s.pids 0) (s.pids 1) j < bounds io.out) →
      (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.writeMask (s.pids 0) (s.pids 1) j →
        p.2 (s.pids 0) (s.pids 1) j < bounds p.1) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs ys : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) j) = xs j) →
      (∀ j : Fin io.B, io.read2Mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) j) = ys j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
            s1.readMem io.out (io.write (s₀.pids 0) (s₀.pids 1) j)
              = f (s₀.pids 0) (s₀.pids 1) xs ys j)
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) j) →
            (∀ p ∈ io.scratch, r = p.1 →
              ∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
                o ≠ p.2 (s₀.pids 0) (s₀.pids 1) j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 2) j')
          (fun j' => vals (⟨1, by decide⟩ : Fin 2) j') j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob hsb
      refine hts bounds s (fun j hj => hib (⟨0, by decide⟩ : Fin 2) j hj)
        (fun j hj => hib (⟨1, by decide⟩ : Fin 2) j hj)
        (fun j hj => hob (⟨0, by decide⟩ : Fin 1) j hj) ?_
      intro q hq j hj
      obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
      have h : (io.scratch.get u).2 (s.pids 0) (s.pids 1) j
          < bounds (io.scratch.get u).1 := hsb u j hj
      rw [hu] at h
      exact h
    · intro s₀ vals hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨0, by decide⟩ : Fin 2) j)
          (fun j => vals (⟨1, by decide⟩ : Fin 2) j)
          (fun j hj => hpins (⟨0, by decide⟩ : Fin 2) j hj)
          (fun j hj => hpins (⟨1, by decide⟩ : Fin 2) j hj)
      refine ⟨s1, hexec, fun _o j hj => hval j hj, ?_⟩
      intro r o' hoc hsc'
      refine hframe r o' ?_ ?_
      · by_cases hro : r = io.out
        · subst hro
          refine Or.inr fun j hj => ?_
          rcases hoc (⟨0, by decide⟩ : Fin 1) j hj with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · intro q hq hrq j hj
        obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
        have h : r ≠ (io.scratch.get u).1 ∨
            o' ≠ (io.scratch.get u).2 (s₀.pids 0) (s₀.pids 1) j :=
          hsc' u j hj
        rw [hu] at h
        rcases h with hne | hno
        · exact absurd hrq hne
        · exact hno
  intro A hd hregs hcov pid₀ pid₁ h1 h2 h3 hsc xs ys s₀ hpid₀ hpid₁ hu hx hy
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | _ => ys)
      s₀ hpid₀ hpid₁ hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => h1 j hj
        | ⟨_+1, _⟩ => fun j hj => h2 j hj)
      (fun _o j hj => h3 j hj)
      (fun t j hj => hsc (io.scratch.get t) (io.scratch.get_mem t) j hj)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hx j hj
        | ⟨_+1, _⟩ => fun j hj => hy j hj)
  refine ⟨s', hexec, fun j hj => hval (⟨0, by decide⟩ : Fin 1) j hj, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hout, hscr⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j hj => hout j hj,
      fun t j hj => hscr (io.scratch.get t) (io.scratch.get_mem t) j hj⟩
end Masked2DKernelIO₂

/-- IO signature of a **2D-grid, general-window** masked two-input /
two-output kernel — the two-output sibling of `Masked2DKernelIO₂` (see
`Masked2DKernelIO₁` for the two generalizations over the 1D family). Each
input carries its own read gate and each output its own write gate, all
defaulting to `mask`. No `scratch` field yet: it will be added when a
consumer appears. -/
structure Masked2DKernelIO₂ₓ₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- First input buffer. -/
  in1 : RegionName
  /-- Second input buffer. -/
  in2 : RegionName
  /-- First output buffer. -/
  out1 : RegionName
  /-- Second output buffer. -/
  out2 : RegionName
  /-- Tile length: each program instance owns `B`-lane windows. -/
  B : Nat
  /-- Lane `j`'s `in1` read address for program `(pid₀, pid₁)`. -/
  read1 : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `in2` read address. -/
  read2 : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `out1` write address. -/
  write1 : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `out2` write address. -/
  write2 : Nat → Nat → Fin B → Nat
  /-- Program `(pid₀, pid₁)`'s **read-active** lanes (`in1`). -/
  mask : Nat → Nat → Fin B → Prop
  /-- Program `(pid₀, pid₁)`'s **`in2` read-active** lanes; defaults to
  `mask` (see `Masked2DKernelIO₂.read2Mask`). -/
  read2Mask : Nat → Nat → Fin B → Prop := mask
  /-- Program `(pid₀, pid₁)`'s **`out1` write-active** lanes; defaults to
  `mask`. -/
  writeMask1 : Nat → Nat → Fin B → Prop := mask
  /-- Program `(pid₀, pid₁)`'s **`out2` write-active** lanes; defaults to
  `mask`. -/
  writeMask2 : Nat → Nat → Fin B → Prop := mask

namespace Masked2DKernelIO₂ₓ₂

/-- `io.Implements f` — two-output sibling of
`Masked2DKernelIO₂.Implements`; the spec `f` returns the pair of the two
outputs' value functions (`.1` for `out1`, `.2` for `out2`). Frame: every
cell outside the union of the two write-active output windows is
untouched. -/
def Implements (io : Masked2DKernelIO₂ₓ₂)
    (f : Nat → Nat → (Fin io.B → ℝ) → (Fin io.B → ℝ) →
      (Fin io.B → ℝ) × (Fin io.B → ℝ)) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.out1, io.out2] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.read1 pid₀ pid₁ j < A.extent io.in1) →
    (∀ j : Fin io.B, io.read2Mask pid₀ pid₁ j →
      io.read2 pid₀ pid₁ j < A.extent io.in2) →
    (∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ j →
      io.write1 pid₀ pid₁ j < A.extent io.out1) →
    (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ j →
      io.write2 pid₀ pid₁ j < A.extent io.out2) →
  ∀ (xs ys : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMem io.in1 (io.read1 pid₀ pid₁ j) = xs j) →
    (∀ j : Fin io.B, io.read2Mask pid₀ pid₁ j →
      s₀.readMem io.in2 (io.read2 pid₀ pid₁ j) = ys j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ j →
          s'.readMem A.flat (A.addr io.out1 (io.write1 pid₀ pid₁ j))
            = (f pid₀ pid₁ xs ys).1 j)
      ∧ (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ j →
          s'.readMem A.flat (A.addr io.out2 (io.write2 pid₀ pid₁ j))
            = (f pid₀ pid₁ xs ys).2 j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ j →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ j)) ∧
             (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ j →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => Masked2DKernelIO₂ₓ₂.Implements

/-- Embed into the unified core — proof plumbing for `Implements.intro`
(two float channels with per-channel read gates, two outputs with
per-output write gates, no scratch). -/
private def toU (io : Masked2DKernelIO₂ₓ₂) : UKernelIO where
  kernel := io.kernel
  nIn := 2
  nOut := 2
  nScr := 0
  bufs := [io.in1, io.in2, io.out1, io.out2]
  ity := fun _ => .float
  iarity := fun _ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | _ => io.in2
  oarity := fun _ => io.B
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.out1
    | _ => io.out2
  obuf_mem := fun o => by fin_cases o <;> simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ p₁ j
    | _ => fun j => io.read2 p₀ p₁ j
  imask := fun i _ p₀ p₁ => match i with
    | ⟨0, _⟩ => fun j => io.mask p₀ p₁ j
    | _ => fun j => io.read2Mask p₀ p₁ j
  owin := fun o _ p₀ p₁ => match o with
    | ⟨0, _⟩ => fun j => io.write1 p₀ p₁ j
    | _ => fun j => io.write2 p₀ p₁ j
  omask := fun o _ p₀ p₁ => match o with
    | ⟨0, _⟩ => fun j => io.writeMask1 p₀ p₁ j
    | _ => fun j => io.writeMask2 p₀ p₁ j
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- Assembly lemma — two-output sibling of
`Masked2DKernelIO₂.Implements.intro`; `hrun`'s frame takes one exclusion
condition per output region. -/
theorem Implements.intro (io : Masked2DKernelIO₂ₓ₂)
    {f : Nat → Nat → (Fin io.B → ℝ) → (Fin io.B → ℝ) →
      (Fin io.B → ℝ) × (Fin io.B → ℝ)}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.read1 (s.pids 0) (s.pids 1) j < bounds io.in1) →
      (∀ j : Fin io.B, io.read2Mask (s.pids 0) (s.pids 1) j →
        io.read2 (s.pids 0) (s.pids 1) j < bounds io.in2) →
      (∀ j : Fin io.B, io.writeMask1 (s.pids 0) (s.pids 1) j →
        io.write1 (s.pids 0) (s.pids 1) j < bounds io.out1) →
      (∀ j : Fin io.B, io.writeMask2 (s.pids 0) (s.pids 1) j →
        io.write2 (s.pids 0) (s.pids 1) j < bounds io.out2) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs ys : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) j) = xs j) →
      (∀ j : Fin io.B, io.read2Mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) j) = ys j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) j →
            s1.readMem io.out1 (io.write1 (s₀.pids 0) (s₀.pids 1) j)
              = (f (s₀.pids 0) (s₀.pids 1) xs ys).1 j)
        ∧ (∀ j : Fin io.B, io.writeMask2 (s₀.pids 0) (s₀.pids 1) j →
            s1.readMem io.out2 (io.write2 (s₀.pids 0) (s₀.pids 1) j)
              = (f (s₀.pids 0) (s₀.pids 1) xs ys).2 j)
        ∧ (∀ r o,
            (r ≠ io.out1 ∨
              ∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) j →
                o ≠ io.write1 (s₀.pids 0) (s₀.pids 1) j) →
            (r ≠ io.out2 ∨
              ∀ j : Fin io.B, io.writeMask2 (s₀.pids 0) (s₀.pids 1) j →
                o ≠ io.write2 (s₀.pids 0) (s₀.pids 1) j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals o => match o with
        | ⟨0, _⟩ => fun j =>
            (f p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 2) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 2) j')).1 j
        | ⟨_+1, _⟩ => fun j =>
            (f p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 2) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 2) j')).2 j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      exact hts bounds s (fun j hj => hib (⟨0, by decide⟩ : Fin 2) j hj)
        (fun j hj => hib (⟨1, by decide⟩ : Fin 2) j hj)
        (fun j hj => hob (⟨0, by decide⟩ : Fin 2) j hj)
        (fun j hj => hob (⟨1, by decide⟩ : Fin 2) j hj)
    · intro s₀ vals hpins
      obtain ⟨s1, hexec, hval1, hval2, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨0, by decide⟩ : Fin 2) j)
          (fun j => vals (⟨1, by decide⟩ : Fin 2) j)
          (fun j hj => hpins (⟨0, by decide⟩ : Fin 2) j hj)
          (fun j hj => hpins (⟨1, by decide⟩ : Fin 2) j hj)
      refine ⟨s1, hexec, fun o => match o with
        | ⟨0, _⟩ => fun j hj => hval1 j hj
        | ⟨_+1, _⟩ => fun j hj => hval2 j hj, ?_⟩
      intro r o' hoc _hsc
      refine hframe r o' ?_ ?_
      · by_cases hro : r = io.out1
        · subst hro
          refine Or.inr fun j hj => ?_
          rcases hoc (⟨0, by decide⟩ : Fin 2) j hj with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · by_cases hro : r = io.out2
        · subst hro
          refine Or.inr fun j hj => ?_
          rcases hoc (⟨1, by decide⟩ : Fin 2) j hj with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ h1 h2 h3 h4 xs ys s₀ hpid₀ hpid₁ hu hx hy
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | _ => ys)
      s₀ hpid₀ hpid₁ hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => h1 j hj
        | ⟨_+1, _⟩ => fun j hj => h2 j hj)
      (fun o => match o with
        | ⟨0, _⟩ => fun j hj => h3 j hj
        | ⟨_+1, _⟩ => fun j hj => h4 j hj)
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hx j hj
        | ⟨_+1, _⟩ => fun j hj => hy j hj)
  refine ⟨s', hexec, fun j hj => hval (⟨0, by decide⟩ : Fin 2) j hj,
    fun j hj => hval (⟨1, by decide⟩ : Fin 2) j hj, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hn1, hn2⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun oc => match oc with
      | ⟨0, _⟩ => fun j hj => hn1 j hj
      | ⟨_+1, _⟩ => fun j hj => hn2 j hj,
      fun t => t.elim0⟩
end Masked2DKernelIO₂ₓ₂

/-- IO signature of a **2D-grid, general-window** masked three-input /
three-output kernel — the widest member of the `Masked2DKernelIO` family
(see `Masked2DKernelIO₁` for the two generalizations over the 1D family).
Each input carries its own read gate and each output its own write gate,
all defaulting to `mask`. No `scratch` field yet: it will be added when a
consumer appears. -/
structure Masked2DKernelIO₃ₓ₃ where
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
  /-- Third output buffer. -/
  out3 : RegionName
  /-- Tile length: each program instance owns `B`-lane windows. -/
  B : Nat
  /-- Lane `j`'s `in1` read address for program `(pid₀, pid₁)`. -/
  read1 : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `in2` read address. -/
  read2 : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `in3` read address. -/
  read3 : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `out1` write address. -/
  write1 : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `out2` write address. -/
  write2 : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `out3` write address. -/
  write3 : Nat → Nat → Fin B → Nat
  /-- Program `(pid₀, pid₁)`'s **read-active** lanes (`in1`). -/
  mask : Nat → Nat → Fin B → Prop
  /-- Program `(pid₀, pid₁)`'s **`in2` read-active** lanes; defaults to
  `mask` (see `Masked2DKernelIO₂.read2Mask`). -/
  read2Mask : Nat → Nat → Fin B → Prop := mask
  /-- Program `(pid₀, pid₁)`'s **`in3` read-active** lanes; defaults to
  `mask`. -/
  read3Mask : Nat → Nat → Fin B → Prop := mask
  /-- Program `(pid₀, pid₁)`'s **`out1` write-active** lanes; defaults to
  `mask`. -/
  writeMask1 : Nat → Nat → Fin B → Prop := mask
  /-- Program `(pid₀, pid₁)`'s **`out2` write-active** lanes; defaults to
  `mask`. -/
  writeMask2 : Nat → Nat → Fin B → Prop := mask
  /-- Program `(pid₀, pid₁)`'s **`out3` write-active** lanes; defaults to
  `mask`. -/
  writeMask3 : Nat → Nat → Fin B → Prop := mask

namespace Masked2DKernelIO₃ₓ₃

/-- `io.Implements f` — three-input / three-output sibling of
`Masked2DKernelIO₂ₓ₂.Implements`. The spec `f` returns the triple of the
three outputs' value functions; `×` is right-associative, so the
components read `.1` (`out1`), `.2.1` (`out2`), `.2.2` (`out3`). Frame:
every cell outside the union of the three write-active output windows is
untouched. -/
def Implements (io : Masked2DKernelIO₃ₓ₃)
    (f : Nat → Nat → (Fin io.B → ℝ) → (Fin io.B → ℝ) → (Fin io.B → ℝ) →
      (Fin io.B → ℝ) × (Fin io.B → ℝ) × (Fin io.B → ℝ)) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.in3, io.out1, io.out2, io.out3] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.read1 pid₀ pid₁ j < A.extent io.in1) →
    (∀ j : Fin io.B, io.read2Mask pid₀ pid₁ j →
      io.read2 pid₀ pid₁ j < A.extent io.in2) →
    (∀ j : Fin io.B, io.read3Mask pid₀ pid₁ j →
      io.read3 pid₀ pid₁ j < A.extent io.in3) →
    (∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ j →
      io.write1 pid₀ pid₁ j < A.extent io.out1) →
    (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ j →
      io.write2 pid₀ pid₁ j < A.extent io.out2) →
    (∀ j : Fin io.B, io.writeMask3 pid₀ pid₁ j →
      io.write3 pid₀ pid₁ j < A.extent io.out3) →
  ∀ (xs ys zs : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMem io.in1 (io.read1 pid₀ pid₁ j) = xs j) →
    (∀ j : Fin io.B, io.read2Mask pid₀ pid₁ j →
      s₀.readMem io.in2 (io.read2 pid₀ pid₁ j) = ys j) →
    (∀ j : Fin io.B, io.read3Mask pid₀ pid₁ j →
      s₀.readMem io.in3 (io.read3 pid₀ pid₁ j) = zs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ j →
          s'.readMem A.flat (A.addr io.out1 (io.write1 pid₀ pid₁ j))
            = (f pid₀ pid₁ xs ys zs).1 j)
      ∧ (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ j →
          s'.readMem A.flat (A.addr io.out2 (io.write2 pid₀ pid₁ j))
            = (f pid₀ pid₁ xs ys zs).2.1 j)
      ∧ (∀ j : Fin io.B, io.writeMask3 pid₀ pid₁ j →
          s'.readMem A.flat (A.addr io.out3 (io.write3 pid₀ pid₁ j))
            = (f pid₀ pid₁ xs ys zs).2.2 j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ j →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ j)) ∧
             (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ j →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ j)) ∧
             (∀ j : Fin io.B, io.writeMask3 pid₀ pid₁ j →
                o' ≠ A.addr io.out3 (io.write3 pid₀ pid₁ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => Masked2DKernelIO₃ₓ₃.Implements

/-- Embed into the unified core — proof plumbing for `Implements.intro`
(three float channels with per-channel read gates, three outputs with
per-output write gates, no scratch). -/
private def toU (io : Masked2DKernelIO₃ₓ₃) : UKernelIO where
  kernel := io.kernel
  nIn := 3
  nOut := 3
  nScr := 0
  bufs := [io.in1, io.in2, io.in3, io.out1, io.out2, io.out3]
  ity := fun _ => .float
  iarity := fun _ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | ⟨1, _⟩ => io.in2
    | _ => io.in3
  oarity := fun _ => io.B
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.out1
    | ⟨1, _⟩ => io.out2
    | _ => io.out3
  obuf_mem := fun o => by fin_cases o <;> simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ p₁ j
    | ⟨1, _⟩ => fun j => io.read2 p₀ p₁ j
    | _ => fun j => io.read3 p₀ p₁ j
  imask := fun i _ p₀ p₁ => match i with
    | ⟨0, _⟩ => fun j => io.mask p₀ p₁ j
    | ⟨1, _⟩ => fun j => io.read2Mask p₀ p₁ j
    | _ => fun j => io.read3Mask p₀ p₁ j
  owin := fun o _ p₀ p₁ => match o with
    | ⟨0, _⟩ => fun j => io.write1 p₀ p₁ j
    | ⟨1, _⟩ => fun j => io.write2 p₀ p₁ j
    | _ => fun j => io.write3 p₀ p₁ j
  omask := fun o _ p₀ p₁ => match o with
    | ⟨0, _⟩ => fun j => io.writeMask1 p₀ p₁ j
    | ⟨1, _⟩ => fun j => io.writeMask2 p₀ p₁ j
    | _ => fun j => io.writeMask3 p₀ p₁ j
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- Assembly lemma — three-input / three-output sibling of
`Masked2DKernelIO₂ₓ₂.Implements.intro`; `hrun`'s frame takes one exclusion
condition per output region. -/
theorem Implements.intro (io : Masked2DKernelIO₃ₓ₃)
    {f : Nat → Nat → (Fin io.B → ℝ) → (Fin io.B → ℝ) → (Fin io.B → ℝ) →
      (Fin io.B → ℝ) × (Fin io.B → ℝ) × (Fin io.B → ℝ)}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.read1 (s.pids 0) (s.pids 1) j < bounds io.in1) →
      (∀ j : Fin io.B, io.read2Mask (s.pids 0) (s.pids 1) j →
        io.read2 (s.pids 0) (s.pids 1) j < bounds io.in2) →
      (∀ j : Fin io.B, io.read3Mask (s.pids 0) (s.pids 1) j →
        io.read3 (s.pids 0) (s.pids 1) j < bounds io.in3) →
      (∀ j : Fin io.B, io.writeMask1 (s.pids 0) (s.pids 1) j →
        io.write1 (s.pids 0) (s.pids 1) j < bounds io.out1) →
      (∀ j : Fin io.B, io.writeMask2 (s.pids 0) (s.pids 1) j →
        io.write2 (s.pids 0) (s.pids 1) j < bounds io.out2) →
      (∀ j : Fin io.B, io.writeMask3 (s.pids 0) (s.pids 1) j →
        io.write3 (s.pids 0) (s.pids 1) j < bounds io.out3) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs ys zs : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) j) = xs j) →
      (∀ j : Fin io.B, io.read2Mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) j) = ys j) →
      (∀ j : Fin io.B, io.read3Mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.in3 (io.read3 (s₀.pids 0) (s₀.pids 1) j) = zs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) j →
            s1.readMem io.out1 (io.write1 (s₀.pids 0) (s₀.pids 1) j)
              = (f (s₀.pids 0) (s₀.pids 1) xs ys zs).1 j)
        ∧ (∀ j : Fin io.B, io.writeMask2 (s₀.pids 0) (s₀.pids 1) j →
            s1.readMem io.out2 (io.write2 (s₀.pids 0) (s₀.pids 1) j)
              = (f (s₀.pids 0) (s₀.pids 1) xs ys zs).2.1 j)
        ∧ (∀ j : Fin io.B, io.writeMask3 (s₀.pids 0) (s₀.pids 1) j →
            s1.readMem io.out3 (io.write3 (s₀.pids 0) (s₀.pids 1) j)
              = (f (s₀.pids 0) (s₀.pids 1) xs ys zs).2.2 j)
        ∧ (∀ r o,
            (r ≠ io.out1 ∨
              ∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) j →
                o ≠ io.write1 (s₀.pids 0) (s₀.pids 1) j) →
            (r ≠ io.out2 ∨
              ∀ j : Fin io.B, io.writeMask2 (s₀.pids 0) (s₀.pids 1) j →
                o ≠ io.write2 (s₀.pids 0) (s₀.pids 1) j) →
            (r ≠ io.out3 ∨
              ∀ j : Fin io.B, io.writeMask3 (s₀.pids 0) (s₀.pids 1) j →
                o ≠ io.write3 (s₀.pids 0) (s₀.pids 1) j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals o => match o with
        | ⟨0, _⟩ => fun j =>
            (f p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨2, by decide⟩ : Fin 3) j')).1 j
        | ⟨1, _⟩ => fun j =>
            (f p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨2, by decide⟩ : Fin 3) j')).2.1 j
        | ⟨_+2, _⟩ => fun j =>
            (f p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨2, by decide⟩ : Fin 3) j')).2.2 j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      exact hts bounds s (fun j hj => hib (⟨0, by decide⟩ : Fin 3) j hj)
        (fun j hj => hib (⟨1, by decide⟩ : Fin 3) j hj)
        (fun j hj => hib (⟨2, by decide⟩ : Fin 3) j hj)
        (fun j hj => hob (⟨0, by decide⟩ : Fin 3) j hj)
        (fun j hj => hob (⟨1, by decide⟩ : Fin 3) j hj)
        (fun j hj => hob (⟨2, by decide⟩ : Fin 3) j hj)
    · intro s₀ vals hpins
      obtain ⟨s1, hexec, hval1, hval2, hval3, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨0, by decide⟩ : Fin 3) j)
          (fun j => vals (⟨1, by decide⟩ : Fin 3) j)
          (fun j => vals (⟨2, by decide⟩ : Fin 3) j)
          (fun j hj => hpins (⟨0, by decide⟩ : Fin 3) j hj)
          (fun j hj => hpins (⟨1, by decide⟩ : Fin 3) j hj)
          (fun j hj => hpins (⟨2, by decide⟩ : Fin 3) j hj)
      refine ⟨s1, hexec, fun o => match o with
        | ⟨0, _⟩ => fun j hj => hval1 j hj
        | ⟨1, _⟩ => fun j hj => hval2 j hj
        | ⟨_+2, _⟩ => fun j hj => hval3 j hj, ?_⟩
      intro r o' hoc _hsc
      refine hframe r o' ?_ ?_ ?_
      · by_cases hro : r = io.out1
        · subst hro
          refine Or.inr fun j hj => ?_
          rcases hoc (⟨0, by decide⟩ : Fin 3) j hj with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · by_cases hro : r = io.out2
        · subst hro
          refine Or.inr fun j hj => ?_
          rcases hoc (⟨1, by decide⟩ : Fin 3) j hj with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · by_cases hro : r = io.out3
        · subst hro
          refine Or.inr fun j hj => ?_
          rcases hoc (⟨2, by decide⟩ : Fin 3) j hj with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ h1 h2 h3 h4 h5 h6 xs ys zs s₀ hpid₀ hpid₁
    hu hx hy hz
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => ys
        | _ => zs)
      s₀ hpid₀ hpid₁ hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => h1 j hj
        | ⟨1, _⟩ => fun j hj => h2 j hj
        | ⟨_+2, _⟩ => fun j hj => h3 j hj)
      (fun o => match o with
        | ⟨0, _⟩ => fun j hj => h4 j hj
        | ⟨1, _⟩ => fun j hj => h5 j hj
        | ⟨_+2, _⟩ => fun j hj => h6 j hj)
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hx j hj
        | ⟨1, _⟩ => fun j hj => hy j hj
        | ⟨_+2, _⟩ => fun j hj => hz j hj)
  refine ⟨s', hexec, fun j hj => hval (⟨0, by decide⟩ : Fin 3) j hj,
    fun j hj => hval (⟨1, by decide⟩ : Fin 3) j hj,
    fun j hj => hval (⟨2, by decide⟩ : Fin 3) j hj, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hn1, hn2, hn3⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun oc => match oc with
      | ⟨0, _⟩ => fun j hj => hn1 j hj
      | ⟨1, _⟩ => fun j hj => hn2 j hj
      | ⟨_+2, _⟩ => fun j hj => hn3 j hj,
      fun t => t.elim0⟩
end Masked2DKernelIO₃ₓ₃

/-- IO signature of a **2D-grid, general-window** masked kernel with one ℝ
input, one **`Bool` input**, and one output — the bool-input-channel sibling
of `Masked2DKernelIO₁` (see there for the two generalizations over the 1D
family). This is the dropout shape: the kernel loads a boolean tile from
`mbuf` alongside the data tile, and the loaded bools may both enter the
computed value and gate which lanes are stored. Accordingly `writeMask`
takes the loaded bool tile — write-active lanes may depend on the data
(`tl.store(…, mask=keep)`) — while `mask` stays the **static** read-active
superset that bounds/trace-safety are stated against. No `scratch` field
yet: it will be added when a consumer appears. -/
structure Masked2DKernelIO₁ᵦ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- Input buffer (ℝ channel). -/
  inp : RegionName
  /-- Boolean input buffer (`.bool` channel). -/
  mbuf : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Tile length: each program instance owns `B`-lane windows. -/
  B : Nat
  /-- Lane `j`'s `inp` read address for program `(pid₀, pid₁)`. -/
  read : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `mbuf` read address (the bool tile's window). -/
  readm : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s write address. -/
  write : Nat → Nat → Fin B → Nat
  /-- Program `(pid₀, pid₁)`'s **static read-active** lanes — the
  bounds/trace-safety superset. -/
  mask : Nat → Nat → Fin B → Prop
  /-- Program `(pid₀, pid₁)`'s **write-active** lanes given the loaded bool
  tile; defaults to the static `mask` (data-independent stores). The data
  gate only ever *narrows* the static mask — `Implements.intro` takes that
  inclusion as its `hsub` hypothesis. -/
  writeMask : Nat → Nat → (Fin B → Bool) → Fin B → Prop :=
    fun p₀ p₁ _ j => mask p₀ p₁ j

namespace Masked2DKernelIO₁ᵦ

/-- `io.Implements f` — bool-input sibling of
`Masked2DKernelIO₁.Implements`. The bool tile `bs` is quantified alongside
the data tile: the launch state holds `bs` on the `.bool` channel of
`mbuf`'s window, and both the spec `f` and the write gate see it. All
address bounds are stated at the **static** `mask` (the superset
trace-safety needs); the data gate `writeMask … bs` only narrows which
output lanes carry the value contract and the frame exclusion. -/
def Implements (io : Masked2DKernelIO₁ᵦ)
    (f : Nat → Nat → (Fin io.B → Bool) → (Fin io.B → ℝ) → Fin io.B → ℝ) :
    Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp, io.mbuf, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.read pid₀ pid₁ j < A.extent io.inp) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.readm pid₀ pid₁ j < A.extent io.mbuf) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.write pid₀ pid₁ j < A.extent io.out) →
  ∀ (bs : Fin io.B → Bool) (xs : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMem io.inp (io.read pid₀ pid₁ j) = xs j) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMemValue .bool io.mbuf (io.readm pid₀ pid₁ j) = bs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask pid₀ pid₁ bs j →
          s'.readMem A.flat (A.addr io.out (io.write pid₀ pid₁ j))
            = f pid₀ pid₁ bs xs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.B, io.writeMask pid₀ pid₁ bs j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => Masked2DKernelIO₁ᵦ.Implements

/-- Embed into the unified core — proof plumbing for `Implements.intro`.
Channel 0 is the float tile, channel 1 the `.bool` tile (it enters both
the lifted spec and the data-dependent `omask`), and channel 2 is a
contract-free bound witness on the output window: the core states output
bounds only at `omask` (= data-gated) lanes, while the family's
trace-safety obligation needs them at the static `mask` — the witness
channel's `imask := mask` carries that wider bound through. -/
private def toU (io : Masked2DKernelIO₁ᵦ) : UKernelIO where
  kernel := io.kernel
  nIn := 3
  nOut := 1
  nScr := 0
  bufs := [io.inp, io.mbuf, io.out]
  ity := fun i => match i with
    | ⟨0, _⟩ => .float
    | ⟨1, _⟩ => .bool
    | _ => .nat
  iarity := fun _ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.inp
    | ⟨1, _⟩ => io.mbuf
    | _ => io.out
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ => match i with
    | ⟨0, _⟩ => fun j => io.read p₀ p₁ j
    | ⟨1, _⟩ => fun j => io.readm p₀ p₁ j
    | _ => fun j => io.write p₀ p₁ j
  imask := fun _ _ p₀ p₁ j => io.mask p₀ p₁ j
  owin := fun _ _ p₀ p₁ j => io.write p₀ p₁ j
  omask := fun _ vals p₀ p₁ j =>
    io.writeMask p₀ p₁ (fun j' => vals (⟨1, by decide⟩ : Fin 3) j') j
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- Assembly lemma — bool-input sibling of
`Masked2DKernelIO₁.Implements.intro`. `hsub` says the data gate only
narrows the static mask (`fun _ _ _ _ h => h` for the default `writeMask`);
it discharges the write-address bound at data-gated lanes from the
static-mask bound. The bool-tile input hypothesis lives on the region-model
state on both sides of the bridge, so no typed-read transport is needed. -/
theorem Implements.intro (io : Masked2DKernelIO₁ᵦ)
    {f : Nat → Nat → (Fin io.B → Bool) → (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hsub : ∀ p₀ p₁ (bs : Fin io.B → Bool) (j : Fin io.B),
      io.writeMask p₀ p₁ bs j → io.mask p₀ p₁ j)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.read (s.pids 0) (s.pids 1) j < bounds io.inp) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.readm (s.pids 0) (s.pids 1) j < bounds io.mbuf) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.write (s.pids 0) (s.pids 1) j < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (bs : Fin io.B → Bool) (xs : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) j) = xs j) →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMemValue .bool io.mbuf (io.readm (s₀.pids 0) (s₀.pids 1) j)
          = bs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) bs j →
            s1.readMem io.out (io.write (s₀.pids 0) (s₀.pids 1) j)
              = f (s₀.pids 0) (s₀.pids 1) bs xs j)
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) bs j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁ (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
          (fun j' => vals (⟨0, by decide⟩ : Fin 3) j') j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib _hob _hsb
      exact hts bounds s (fun j hj => hib (⟨0, by decide⟩ : Fin 3) j hj)
        (fun j hj => hib (⟨1, by decide⟩ : Fin 3) j hj)
        (fun j hj => hib (⟨2, by decide⟩ : Fin 3) j hj)
    · intro s₀ vals hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨1, by decide⟩ : Fin 3) j)
          (fun j => vals (⟨0, by decide⟩ : Fin 3) j)
          (fun j hj => hpins (⟨0, by decide⟩ : Fin 3) j hj)
          (fun j hj => hpins (⟨1, by decide⟩ : Fin 3) j hj)
      refine ⟨s1, hexec, fun _o j hj => hval j hj, ?_⟩
      intro r o' hoc _hsc
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun j hj => ?_
        rcases hoc (⟨0, by decide⟩ : Fin 1) j hj with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ h1 h2 h3 bs xs s₀ hpid₀ hpid₁ hu hx hb
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => bs
        | ⟨_+2, _⟩ => fun j =>
            ChanTy.read .nat s₀ io.out (io.write pid₀ pid₁ j))
      s₀ hpid₀ hpid₁ hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => h1 j hj
        | ⟨1, _⟩ => fun j hj => h2 j hj
        | ⟨_+2, _⟩ => fun j hj => h3 j hj)
      (fun _o j hj => h3 j (hsub _ _ _ j hj))
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hx j hj
        | ⟨1, _⟩ => fun j hj => hb j hj
        | ⟨_+2, _⟩ => fun _ _ => rfl)
  refine ⟨s', hexec, fun j hj => hval (⟨0, by decide⟩ : Fin 1) j hj, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | hn
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j hj => hn j hj, fun t => t.elim0⟩
end Masked2DKernelIO₁ᵦ

/-- IO signature of a **2D-grid, general-window** masked kernel with two ℝ
inputs, one **`Bool` input**, and one output — the two-input sibling of
`Masked2DKernelIO₁ᵦ` (see there for the bool-channel reading, and
`Masked2DKernelIO₁` for the two generalizations over the 1D family). This
is the masked-add shape; in-place updates (`out = in2`) are expressible by
the duplicate-region precedent of `MaskedKernelIO₃ₓ₂`. No `scratch` field
yet: it will be added when a consumer appears. -/
structure Masked2DKernelIO₂ᵦ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- First input buffer (ℝ channel). -/
  in1 : RegionName
  /-- Second input buffer (ℝ channel). -/
  in2 : RegionName
  /-- Boolean input buffer (`.bool` channel). -/
  mbuf : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Tile length: each program instance owns `B`-lane windows. -/
  B : Nat
  /-- Lane `j`'s `in1` read address for program `(pid₀, pid₁)`. -/
  read1 : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `in2` read address. -/
  read2 : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `mbuf` read address (the bool tile's window). -/
  readm : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s write address. -/
  write : Nat → Nat → Fin B → Nat
  /-- Program `(pid₀, pid₁)`'s **static read-active** lanes — the
  bounds/trace-safety superset. -/
  mask : Nat → Nat → Fin B → Prop
  /-- Program `(pid₀, pid₁)`'s **write-active** lanes given the loaded bool
  tile; defaults to the static `mask` (see `Masked2DKernelIO₁ᵦ.writeMask`). -/
  writeMask : Nat → Nat → (Fin B → Bool) → Fin B → Prop :=
    fun p₀ p₁ _ j => mask p₀ p₁ j

namespace Masked2DKernelIO₂ᵦ

/-- `io.Implements f` — two-input sibling of
`Masked2DKernelIO₁ᵦ.Implements`. -/
def Implements (io : Masked2DKernelIO₂ᵦ)
    (f : Nat → Nat → (Fin io.B → Bool) → (Fin io.B → ℝ) → (Fin io.B → ℝ) →
      Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.mbuf, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.read1 pid₀ pid₁ j < A.extent io.in1) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.read2 pid₀ pid₁ j < A.extent io.in2) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.readm pid₀ pid₁ j < A.extent io.mbuf) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.write pid₀ pid₁ j < A.extent io.out) →
  ∀ (bs : Fin io.B → Bool) (xs ys : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMem io.in1 (io.read1 pid₀ pid₁ j) = xs j) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMem io.in2 (io.read2 pid₀ pid₁ j) = ys j) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMemValue .bool io.mbuf (io.readm pid₀ pid₁ j) = bs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask pid₀ pid₁ bs j →
          s'.readMem A.flat (A.addr io.out (io.write pid₀ pid₁ j))
            = f pid₀ pid₁ bs xs ys j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.B, io.writeMask pid₀ pid₁ bs j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => Masked2DKernelIO₂ᵦ.Implements

/-- Embed into the unified core — proof plumbing for `Implements.intro`.
Channels 0/1 are the float tiles, channel 2 the `.bool` tile, and channel 3
a contract-free bound witness on the output window (see
`Masked2DKernelIO₁ᵦ.toU` for why the witness carries the static-`mask`
write bound). -/
private def toU (io : Masked2DKernelIO₂ᵦ) : UKernelIO where
  kernel := io.kernel
  nIn := 4
  nOut := 1
  nScr := 0
  bufs := [io.in1, io.in2, io.mbuf, io.out]
  ity := fun i => match i with
    | ⟨0, _⟩ => .float
    | ⟨1, _⟩ => .float
    | ⟨2, _⟩ => .bool
    | _ => .nat
  iarity := fun _ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | ⟨1, _⟩ => io.in2
    | ⟨2, _⟩ => io.mbuf
    | _ => io.out
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ p₁ j
    | ⟨1, _⟩ => fun j => io.read2 p₀ p₁ j
    | ⟨2, _⟩ => fun j => io.readm p₀ p₁ j
    | _ => fun j => io.write p₀ p₁ j
  imask := fun _ _ p₀ p₁ j => io.mask p₀ p₁ j
  owin := fun _ _ p₀ p₁ j => io.write p₀ p₁ j
  omask := fun _ vals p₀ p₁ j =>
    io.writeMask p₀ p₁ (fun j' => vals (⟨2, by decide⟩ : Fin 4) j') j
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- Assembly lemma — two-input sibling of
`Masked2DKernelIO₁ᵦ.Implements.intro` (see there for `hsub`). -/
theorem Implements.intro (io : Masked2DKernelIO₂ᵦ)
    {f : Nat → Nat → (Fin io.B → Bool) → (Fin io.B → ℝ) → (Fin io.B → ℝ) →
      Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hsub : ∀ p₀ p₁ (bs : Fin io.B → Bool) (j : Fin io.B),
      io.writeMask p₀ p₁ bs j → io.mask p₀ p₁ j)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.read1 (s.pids 0) (s.pids 1) j < bounds io.in1) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.read2 (s.pids 0) (s.pids 1) j < bounds io.in2) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.readm (s.pids 0) (s.pids 1) j < bounds io.mbuf) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.write (s.pids 0) (s.pids 1) j < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (bs : Fin io.B → Bool)
        (xs ys : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) j) = xs j) →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) j) = ys j) →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMemValue .bool io.mbuf (io.readm (s₀.pids 0) (s₀.pids 1) j)
          = bs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) bs j →
            s1.readMem io.out (io.write (s₀.pids 0) (s₀.pids 1) j)
              = f (s₀.pids 0) (s₀.pids 1) bs xs ys j)
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) bs j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁ (fun j' => vals (⟨2, by decide⟩ : Fin 4) j')
          (fun j' => vals (⟨0, by decide⟩ : Fin 4) j')
          (fun j' => vals (⟨1, by decide⟩ : Fin 4) j') j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib _hob _hsb
      exact hts bounds s (fun j hj => hib (⟨0, by decide⟩ : Fin 4) j hj)
        (fun j hj => hib (⟨1, by decide⟩ : Fin 4) j hj)
        (fun j hj => hib (⟨2, by decide⟩ : Fin 4) j hj)
        (fun j hj => hib (⟨3, by decide⟩ : Fin 4) j hj)
    · intro s₀ vals hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨2, by decide⟩ : Fin 4) j)
          (fun j => vals (⟨0, by decide⟩ : Fin 4) j)
          (fun j => vals (⟨1, by decide⟩ : Fin 4) j)
          (fun j hj => hpins (⟨0, by decide⟩ : Fin 4) j hj)
          (fun j hj => hpins (⟨1, by decide⟩ : Fin 4) j hj)
          (fun j hj => hpins (⟨2, by decide⟩ : Fin 4) j hj)
      refine ⟨s1, hexec, fun _o j hj => hval j hj, ?_⟩
      intro r o' hoc _hsc
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun j hj => ?_
        rcases hoc (⟨0, by decide⟩ : Fin 1) j hj with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ h1 h2 h3 h4 bs xs ys s₀ hpid₀ hpid₁ hu
    hx hy hb
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => ys
        | ⟨2, _⟩ => bs
        | ⟨_+3, _⟩ => fun j =>
            ChanTy.read .nat s₀ io.out (io.write pid₀ pid₁ j))
      s₀ hpid₀ hpid₁ hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => h1 j hj
        | ⟨1, _⟩ => fun j hj => h2 j hj
        | ⟨2, _⟩ => fun j hj => h3 j hj
        | ⟨_+3, _⟩ => fun j hj => h4 j hj)
      (fun _o j hj => h4 j (hsub _ _ _ j hj))
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hx j hj
        | ⟨1, _⟩ => fun j hj => hy j hj
        | ⟨2, _⟩ => fun j hj => hb j hj
        | ⟨_+3, _⟩ => fun _ _ => rfl)
  refine ⟨s', hexec, fun j hj => hval (⟨0, by decide⟩ : Fin 1) j hj, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | hn
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j hj => hn j hj, fun t => t.elim0⟩
end Masked2DKernelIO₂ᵦ

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

/-- Embed into the unified core — proof plumbing for `Implements.intro`
(one float tile channel plus 1-lane bound-witness channels for the input,
output, and every scratch buffer — the block bounds `w + L ≤ extent`
carried as the masked per-lane bounds `w + L - 1 < extent` gated on
`0 < w + L` — one output, scratch as contract-free channels). -/
private def toU (io : KernelIO₁) : UKernelIO where
  kernel := io.kernel
  nIn := 3 + io.scratch.length
  nOut := 1
  nScr := io.scratch.length
  bufs := [io.inp, io.out] ++ io.scratch.map (·.buf)
  ity := fun i => match i with
    | ⟨0, _⟩ => .float
    | _ => .nat
  iarity := fun i => match i with
    | ⟨0, _⟩ => io.Bin
    | _ => 1
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.inp
    | ⟨1, _⟩ => io.inp
    | ⟨2, _⟩ => io.out
    | ⟨k+3, h⟩ => (io.scratch.get ⟨k, by omega⟩).buf
  oarity := fun _ => io.Bout
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => (io.scratch.get t).len
  sbuf := fun t => (io.scratch.get t).buf
  iwin := fun i _ p₀ _ => match i with
    | ⟨0, _⟩ => fun j => io.read p₀ + j.val
    | ⟨1, _⟩ => fun _ => io.read p₀ + io.Bin - 1
    | ⟨2, _⟩ => fun _ => io.write p₀ + io.Bout - 1
    | ⟨k+3, h⟩ => fun _ =>
        (io.scratch.get ⟨k, by omega⟩).win p₀
          + (io.scratch.get ⟨k, by omega⟩).len - 1
  imask := fun i _ p₀ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => 0 < io.read p₀ + io.Bin
    | ⟨2, _⟩ => fun _ => 0 < io.write p₀ + io.Bout
    | ⟨k+3, h⟩ => fun _ =>
        0 < (io.scratch.get ⟨k, by omega⟩).win p₀
          + (io.scratch.get ⟨k, by omega⟩).len
  owin := fun _ _ p₀ _ j => io.write p₀ + j.val
  omask := fun _ _ _ _ _ => True
  swin := fun t _ p₀ _ k => (io.scratch.get t).win p₀ + k.val
  smask := fun _ _ _ _ _ => True

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
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun _p₀ _p₁ vals _o j =>
        f (fun j' => vals (⟨0, by omega⟩ : Fin (3 + io.scratch.length)) j') j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib _hob _hsb
      have hb1 : io.read (s.pids 0) + io.Bin ≤ bounds io.inp := by
        by_cases hpos : 0 < io.read (s.pids 0) + io.Bin
        · have h : io.read (s.pids 0) + io.Bin - 1 < bounds io.inp :=
            hib (⟨1, by omega⟩ : Fin (3 + io.scratch.length))
              (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb2 : io.write (s.pids 0) + io.Bout ≤ bounds io.out := by
        by_cases hpos : 0 < io.write (s.pids 0) + io.Bout
        · have h : io.write (s.pids 0) + io.Bout - 1 < bounds io.out :=
            hib (⟨2, by omega⟩ : Fin (3 + io.scratch.length))
              (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      refine hts bounds s hb1 hb2 ?_
      intro q hq
      obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
      subst hu
      show (io.scratch.get u).win (s.pids 0) + (io.scratch.get u).len
          ≤ bounds (io.scratch.get u).buf
      by_cases hpos : 0 < (io.scratch.get u).win (s.pids 0)
          + (io.scratch.get u).len
      · have h : (io.scratch.get u).win (s.pids 0)
            + (io.scratch.get u).len - 1 < bounds (io.scratch.get u).buf :=
          hib (⟨u.val + 3, by have := u.isLt; omega⟩ :
              Fin (3 + io.scratch.length))
            (⟨0, by decide⟩ : Fin 1) hpos
        omega
      · omega
    · intro s₀ vals hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨0, by omega⟩ : Fin (3 + io.scratch.length)) j)
          (fun j => hpins (⟨0, by omega⟩ : Fin (3 + io.scratch.length)) j
            True.intro)
      refine ⟨s1, hexec, fun _o j _ => hval j, ?_⟩
      intro r o' hoc hsc'
      refine hframe r o' ?_ ?_
      · by_cases hro : r = io.out
        · subst hro
          refine Or.inr fun j => ?_
          rcases hoc (⟨0, by decide⟩ : Fin 1) j True.intro with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · intro q hq hrq k
        obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
        subst hu
        rcases hsc' u k True.intro with hne | hno
        · exact absurd hrq hne
        · exact hno
  intro A hd hregs hcov pid h1 h2 hsc xs s₀ hpid hu hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => fun _ => ChanTy.read .nat s₀ io.inp (io.read pid + io.Bin - 1)
        | ⟨2, _⟩ => fun _ => ChanTy.read .nat s₀ io.out (io.write pid + io.Bout - 1)
        | ⟨k+3, h⟩ => fun _ =>
            have hk : k < io.scratch.length := by
              have h' : k + 3 < 3 + io.scratch.length := h
              omega
            ChanTy.read .nat s₀ (io.scratch.get ⟨k, hk⟩).buf
              ((io.scratch.get ⟨k, hk⟩).win pid
                + (io.scratch.get ⟨k, hk⟩).len - 1))
      s₀ hpid rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j _ => by
            have hj : j.val < io.Bin := j.isLt
            have h : io.read pid + j.val < A.extent io.inp := by omega
            exact h
        | ⟨1, _⟩ => fun _ hm => by
            have hm' : 0 < io.read pid + io.Bin := hm
            have h : io.read pid + io.Bin - 1 < A.extent io.inp := by omega
            exact h
        | ⟨2, _⟩ => fun _ hm => by
            have hm' : 0 < io.write pid + io.Bout := hm
            have h : io.write pid + io.Bout - 1 < A.extent io.out := by omega
            exact h
        | ⟨k+3, hk3⟩ => fun _ hm => by
            have hk : k < io.scratch.length := by
              have h' : k + 3 < 3 + io.scratch.length := hk3
              omega
            have hm' : 0 < (io.scratch.get ⟨k, hk⟩).win pid
                + (io.scratch.get ⟨k, hk⟩).len := hm
            have hb : (io.scratch.get ⟨k, hk⟩).win pid
                + (io.scratch.get ⟨k, hk⟩).len
                ≤ A.extent (io.scratch.get ⟨k, hk⟩).buf :=
              hsc (io.scratch.get ⟨k, hk⟩) (io.scratch.get_mem ⟨k, hk⟩)
            have h : (io.scratch.get ⟨k, hk⟩).win pid
                + (io.scratch.get ⟨k, hk⟩).len - 1
                < A.extent (io.scratch.get ⟨k, hk⟩).buf := by omega
            exact h)
      (fun _o j _ => by
        have hj : j.val < io.Bout := j.isLt
        have h : io.write pid + j.val < A.extent io.out := by omega
        exact h)
      (fun t k _ => by
        have hb : (io.scratch.get t).win pid + (io.scratch.get t).len
            ≤ A.extent (io.scratch.get t).buf :=
          hsc (io.scratch.get t) (io.scratch.get_mem t)
        have hk : k.val < (io.scratch.get t).len := k.isLt
        have h : (io.scratch.get t).win pid + k.val
            < A.extent (io.scratch.get t).buf := by omega
        exact h)
      (fun i => match i with
        | ⟨0, _⟩ => fun j _ => hx j
        | ⟨1, _⟩ => fun _ _ => rfl
        | ⟨2, _⟩ => fun _ _ => rfl
        | ⟨_+3, _⟩ => fun _ _ => rfl)
  refine ⟨s', hexec, fun j => hval (⟨0, by decide⟩ : Fin 1) j True.intro, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hout, hscr⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j _ => hout j,
      fun t k _ => hscr (io.scratch.get t) (io.scratch.get_mem t) k⟩

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

/-- Embed into the unified core — proof plumbing for `Implements.intro`
(three float tile channels plus 1-lane bound-witness channels for every
input, the output, and every scratch buffer — see `KernelIO₁.toU` — one
output, scratch as contract-free channels). -/
private def toU (io : KernelIO₃) : UKernelIO where
  kernel := io.kernel
  nIn := 7 + io.scratch.length
  nOut := 1
  nScr := io.scratch.length
  bufs := [io.in1, io.in2, io.in3, io.out] ++ io.scratch.map (·.buf)
  ity := fun i => match i with
    | ⟨0, _⟩ => .float
    | ⟨1, _⟩ => .float
    | ⟨2, _⟩ => .float
    | _ => .nat
  iarity := fun i => match i with
    | ⟨0, _⟩ => io.B1
    | ⟨1, _⟩ => io.B2
    | ⟨2, _⟩ => io.B3
    | _ => 1
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | ⟨1, _⟩ => io.in2
    | ⟨2, _⟩ => io.in3
    | ⟨3, _⟩ => io.in1
    | ⟨4, _⟩ => io.in2
    | ⟨5, _⟩ => io.in3
    | ⟨6, _⟩ => io.out
    | ⟨k+7, h⟩ => (io.scratch.get ⟨k, by omega⟩).buf
  oarity := fun _ => io.Bout
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => (io.scratch.get t).len
  sbuf := fun t => (io.scratch.get t).buf
  iwin := fun i _ p₀ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ + j.val
    | ⟨1, _⟩ => fun j => io.read2 p₀ + j.val
    | ⟨2, _⟩ => fun j => io.read3 p₀ + j.val
    | ⟨3, _⟩ => fun _ => io.read1 p₀ + io.B1 - 1
    | ⟨4, _⟩ => fun _ => io.read2 p₀ + io.B2 - 1
    | ⟨5, _⟩ => fun _ => io.read3 p₀ + io.B3 - 1
    | ⟨6, _⟩ => fun _ => io.write p₀ + io.Bout - 1
    | ⟨k+7, h⟩ => fun _ =>
        (io.scratch.get ⟨k, by omega⟩).win p₀
          + (io.scratch.get ⟨k, by omega⟩).len - 1
  imask := fun i _ p₀ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => True
    | ⟨2, _⟩ => fun _ => True
    | ⟨3, _⟩ => fun _ => 0 < io.read1 p₀ + io.B1
    | ⟨4, _⟩ => fun _ => 0 < io.read2 p₀ + io.B2
    | ⟨5, _⟩ => fun _ => 0 < io.read3 p₀ + io.B3
    | ⟨6, _⟩ => fun _ => 0 < io.write p₀ + io.Bout
    | ⟨k+7, h⟩ => fun _ =>
        0 < (io.scratch.get ⟨k, by omega⟩).win p₀
          + (io.scratch.get ⟨k, by omega⟩).len
  owin := fun _ _ p₀ _ j => io.write p₀ + j.val
  omask := fun _ _ _ _ _ => True
  swin := fun t _ p₀ _ k => (io.scratch.get t).win p₀ + k.val
  smask := fun _ _ _ _ _ => True

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
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun _p₀ _p₁ vals _o j =>
        f (fun j' => vals (⟨0, by omega⟩ : Fin (7 + io.scratch.length)) j')
          (fun j' => vals (⟨1, by omega⟩ : Fin (7 + io.scratch.length)) j')
          (fun j' => vals (⟨2, by omega⟩ : Fin (7 + io.scratch.length)) j')
          j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib _hob _hsb
      have hb1 : io.read1 (s.pids 0) + io.B1 ≤ bounds io.in1 := by
        by_cases hpos : 0 < io.read1 (s.pids 0) + io.B1
        · have h : io.read1 (s.pids 0) + io.B1 - 1 < bounds io.in1 :=
            hib (⟨3, by omega⟩ : Fin (7 + io.scratch.length))
              (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb2 : io.read2 (s.pids 0) + io.B2 ≤ bounds io.in2 := by
        by_cases hpos : 0 < io.read2 (s.pids 0) + io.B2
        · have h : io.read2 (s.pids 0) + io.B2 - 1 < bounds io.in2 :=
            hib (⟨4, by omega⟩ : Fin (7 + io.scratch.length))
              (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb3 : io.read3 (s.pids 0) + io.B3 ≤ bounds io.in3 := by
        by_cases hpos : 0 < io.read3 (s.pids 0) + io.B3
        · have h : io.read3 (s.pids 0) + io.B3 - 1 < bounds io.in3 :=
            hib (⟨5, by omega⟩ : Fin (7 + io.scratch.length))
              (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb4 : io.write (s.pids 0) + io.Bout ≤ bounds io.out := by
        by_cases hpos : 0 < io.write (s.pids 0) + io.Bout
        · have h : io.write (s.pids 0) + io.Bout - 1 < bounds io.out :=
            hib (⟨6, by omega⟩ : Fin (7 + io.scratch.length))
              (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      refine hts bounds s hb1 hb2 hb3 hb4 ?_
      intro q hq
      obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
      subst hu
      show (io.scratch.get u).win (s.pids 0) + (io.scratch.get u).len
          ≤ bounds (io.scratch.get u).buf
      by_cases hpos : 0 < (io.scratch.get u).win (s.pids 0)
          + (io.scratch.get u).len
      · have h : (io.scratch.get u).win (s.pids 0)
            + (io.scratch.get u).len - 1 < bounds (io.scratch.get u).buf :=
          hib (⟨u.val + 7, by have := u.isLt; omega⟩ :
              Fin (7 + io.scratch.length))
            (⟨0, by decide⟩ : Fin 1) hpos
        omega
      · omega
    · intro s₀ vals hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀
          (fun j => vals (⟨0, by omega⟩ : Fin (7 + io.scratch.length)) j)
          (fun j => vals (⟨1, by omega⟩ : Fin (7 + io.scratch.length)) j)
          (fun j => vals (⟨2, by omega⟩ : Fin (7 + io.scratch.length)) j)
          (fun j => hpins (⟨0, by omega⟩ : Fin (7 + io.scratch.length)) j
            True.intro)
          (fun j => hpins (⟨1, by omega⟩ : Fin (7 + io.scratch.length)) j
            True.intro)
          (fun j => hpins (⟨2, by omega⟩ : Fin (7 + io.scratch.length)) j
            True.intro)
      refine ⟨s1, hexec, fun _o j _ => hval j, ?_⟩
      intro r o' hoc hsc'
      refine hframe r o' ?_ ?_
      · by_cases hro : r = io.out
        · subst hro
          refine Or.inr fun j => ?_
          rcases hoc (⟨0, by decide⟩ : Fin 1) j True.intro with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · intro q hq hrq k
        obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
        subst hu
        rcases hsc' u k True.intro with hne | hno
        · exact absurd hrq hne
        · exact hno
  intro A hd hregs hcov pid h1 h2 h3 h4 hsc xs ys zs s₀ hpid hu hx hy hz
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => ys
        | ⟨2, _⟩ => zs
        | ⟨3, _⟩ => fun _ => ChanTy.read .nat s₀ io.in1 (io.read1 pid + io.B1 - 1)
        | ⟨4, _⟩ => fun _ => ChanTy.read .nat s₀ io.in2 (io.read2 pid + io.B2 - 1)
        | ⟨5, _⟩ => fun _ => ChanTy.read .nat s₀ io.in3 (io.read3 pid + io.B3 - 1)
        | ⟨6, _⟩ => fun _ => ChanTy.read .nat s₀ io.out (io.write pid + io.Bout - 1)
        | ⟨k+7, h⟩ => fun _ =>
            have hk : k < io.scratch.length := by
              have h' : k + 7 < 7 + io.scratch.length := h
              omega
            ChanTy.read .nat s₀ (io.scratch.get ⟨k, hk⟩).buf
              ((io.scratch.get ⟨k, hk⟩).win pid
                + (io.scratch.get ⟨k, hk⟩).len - 1))
      s₀ hpid rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j _ => by
            have hj : j.val < io.B1 := j.isLt
            have h : io.read1 pid + j.val < A.extent io.in1 := by omega
            exact h
        | ⟨1, _⟩ => fun j _ => by
            have hj : j.val < io.B2 := j.isLt
            have h : io.read2 pid + j.val < A.extent io.in2 := by omega
            exact h
        | ⟨2, _⟩ => fun j _ => by
            have hj : j.val < io.B3 := j.isLt
            have h : io.read3 pid + j.val < A.extent io.in3 := by omega
            exact h
        | ⟨3, _⟩ => fun _ hm => by
            have hm' : 0 < io.read1 pid + io.B1 := hm
            have h : io.read1 pid + io.B1 - 1 < A.extent io.in1 := by omega
            exact h
        | ⟨4, _⟩ => fun _ hm => by
            have hm' : 0 < io.read2 pid + io.B2 := hm
            have h : io.read2 pid + io.B2 - 1 < A.extent io.in2 := by omega
            exact h
        | ⟨5, _⟩ => fun _ hm => by
            have hm' : 0 < io.read3 pid + io.B3 := hm
            have h : io.read3 pid + io.B3 - 1 < A.extent io.in3 := by omega
            exact h
        | ⟨6, _⟩ => fun _ hm => by
            have hm' : 0 < io.write pid + io.Bout := hm
            have h : io.write pid + io.Bout - 1 < A.extent io.out := by omega
            exact h
        | ⟨k+7, hk7⟩ => fun _ hm => by
            have hk : k < io.scratch.length := by
              have h' : k + 7 < 7 + io.scratch.length := hk7
              omega
            have hm' : 0 < (io.scratch.get ⟨k, hk⟩).win pid
                + (io.scratch.get ⟨k, hk⟩).len := hm
            have hb : (io.scratch.get ⟨k, hk⟩).win pid
                + (io.scratch.get ⟨k, hk⟩).len
                ≤ A.extent (io.scratch.get ⟨k, hk⟩).buf :=
              hsc (io.scratch.get ⟨k, hk⟩) (io.scratch.get_mem ⟨k, hk⟩)
            have h : (io.scratch.get ⟨k, hk⟩).win pid
                + (io.scratch.get ⟨k, hk⟩).len - 1
                < A.extent (io.scratch.get ⟨k, hk⟩).buf := by omega
            exact h)
      (fun _o j _ => by
        have hj : j.val < io.Bout := j.isLt
        have h : io.write pid + j.val < A.extent io.out := by omega
        exact h)
      (fun t k _ => by
        have hb : (io.scratch.get t).win pid + (io.scratch.get t).len
            ≤ A.extent (io.scratch.get t).buf :=
          hsc (io.scratch.get t) (io.scratch.get_mem t)
        have hk : k.val < (io.scratch.get t).len := k.isLt
        have h : (io.scratch.get t).win pid + k.val
            < A.extent (io.scratch.get t).buf := by omega
        exact h)
      (fun i => match i with
        | ⟨0, _⟩ => fun j _ => hx j
        | ⟨1, _⟩ => fun j _ => hy j
        | ⟨2, _⟩ => fun j _ => hz j
        | ⟨3, _⟩ => fun _ _ => rfl
        | ⟨4, _⟩ => fun _ _ => rfl
        | ⟨5, _⟩ => fun _ _ => rfl
        | ⟨6, _⟩ => fun _ _ => rfl
        | ⟨_+7, _⟩ => fun _ _ => rfl)
  refine ⟨s', hexec, fun j => hval (⟨0, by decide⟩ : Fin 1) j True.intro, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hout, hscr⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j _ => hout j,
      fun t k _ => hscr (io.scratch.get t) (io.scratch.get_mem t) k⟩

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

/-- Embed into the unified core — proof plumbing for `Implements.intro`
(three float tile channels plus 1-lane bound-witness channels for every
input and output buffer — see `KernelIO₁.toU` — two outputs, no scratch). -/
private def toU (io : KernelIO₃ₓ₂) : UKernelIO where
  kernel := io.kernel
  nIn := 8
  nOut := 2
  nScr := 0
  bufs := [io.in1, io.in2, io.in3, io.out1, io.out2]
  ity := fun i => match i with
    | ⟨0, _⟩ => .float
    | ⟨1, _⟩ => .float
    | ⟨2, _⟩ => .float
    | _ => .nat
  iarity := fun i => match i with
    | ⟨0, _⟩ => io.B1
    | ⟨1, _⟩ => io.B2
    | ⟨2, _⟩ => io.B3
    | _ => 1
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | ⟨1, _⟩ => io.in2
    | ⟨2, _⟩ => io.in3
    | ⟨3, _⟩ => io.in1
    | ⟨4, _⟩ => io.in2
    | ⟨5, _⟩ => io.in3
    | ⟨6, _⟩ => io.out1
    | _ => io.out2
  oarity := fun o => match o with
    | ⟨0, _⟩ => io.Bout1
    | _ => io.Bout2
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.out1
    | _ => io.out2
  obuf_mem := fun o => by fin_cases o <;> simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ + j.val
    | ⟨1, _⟩ => fun j => io.read2 p₀ + j.val
    | ⟨2, _⟩ => fun j => io.read3 p₀ + j.val
    | ⟨3, _⟩ => fun _ => io.read1 p₀ + io.B1 - 1
    | ⟨4, _⟩ => fun _ => io.read2 p₀ + io.B2 - 1
    | ⟨5, _⟩ => fun _ => io.read3 p₀ + io.B3 - 1
    | ⟨6, _⟩ => fun _ => io.write1 p₀ + io.Bout1 - 1
    | _ => fun _ => io.write2 p₀ + io.Bout2 - 1
  imask := fun i _ p₀ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => True
    | ⟨2, _⟩ => fun _ => True
    | ⟨3, _⟩ => fun _ => 0 < io.read1 p₀ + io.B1
    | ⟨4, _⟩ => fun _ => 0 < io.read2 p₀ + io.B2
    | ⟨5, _⟩ => fun _ => 0 < io.read3 p₀ + io.B3
    | ⟨6, _⟩ => fun _ => 0 < io.write1 p₀ + io.Bout1
    | _ => fun _ => 0 < io.write2 p₀ + io.Bout2
  owin := fun o _ p₀ _ => match o with
    | ⟨0, _⟩ => fun j => io.write1 p₀ + j.val
    | _ => fun j => io.write2 p₀ + j.val
  omask := fun _ _ _ _ _ => True
  swin := fun t => t.elim0
  smask := fun t => t.elim0

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
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun _p₀ _p₁ vals o => match o with
        | ⟨0, _⟩ => fun j =>
            (f (fun j' => vals (⟨0, by decide⟩ : Fin 8) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 8) j')
              (fun j' => vals (⟨2, by decide⟩ : Fin 8) j')).1 j
        | ⟨_+1, _⟩ => fun j =>
            (f (fun j' => vals (⟨0, by decide⟩ : Fin 8) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 8) j')
              (fun j' => vals (⟨2, by decide⟩ : Fin 8) j')).2 j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib _hob _hsb
      have hb1 : io.read1 (s.pids 0) + io.B1 ≤ bounds io.in1 := by
        by_cases hpos : 0 < io.read1 (s.pids 0) + io.B1
        · have h : io.read1 (s.pids 0) + io.B1 - 1 < bounds io.in1 :=
            hib (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb2 : io.read2 (s.pids 0) + io.B2 ≤ bounds io.in2 := by
        by_cases hpos : 0 < io.read2 (s.pids 0) + io.B2
        · have h : io.read2 (s.pids 0) + io.B2 - 1 < bounds io.in2 :=
            hib (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb3 : io.read3 (s.pids 0) + io.B3 ≤ bounds io.in3 := by
        by_cases hpos : 0 < io.read3 (s.pids 0) + io.B3
        · have h : io.read3 (s.pids 0) + io.B3 - 1 < bounds io.in3 :=
            hib (⟨5, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb4 : io.write1 (s.pids 0) + io.Bout1 ≤ bounds io.out1 := by
        by_cases hpos : 0 < io.write1 (s.pids 0) + io.Bout1
        · have h : io.write1 (s.pids 0) + io.Bout1 - 1 < bounds io.out1 :=
            hib (⟨6, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb5 : io.write2 (s.pids 0) + io.Bout2 ≤ bounds io.out2 := by
        by_cases hpos : 0 < io.write2 (s.pids 0) + io.Bout2
        · have h : io.write2 (s.pids 0) + io.Bout2 - 1 < bounds io.out2 :=
            hib (⟨7, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      exact hts bounds s hb1 hb2 hb3 hb4 hb5
    · intro s₀ vals hpins
      obtain ⟨s1, hexec, hval1, hval2, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨0, by decide⟩ : Fin 8) j)
          (fun j => vals (⟨1, by decide⟩ : Fin 8) j)
          (fun j => vals (⟨2, by decide⟩ : Fin 8) j)
          (fun j => hpins (⟨0, by decide⟩ : Fin 8) j True.intro)
          (fun j => hpins (⟨1, by decide⟩ : Fin 8) j True.intro)
          (fun j => hpins (⟨2, by decide⟩ : Fin 8) j True.intro)
      refine ⟨s1, hexec, fun o => match o with
        | ⟨0, _⟩ => fun j _ => hval1 j
        | ⟨_+1, _⟩ => fun j _ => hval2 j, ?_⟩
      intro r o' hoc _hsc
      refine hframe r o' ?_ ?_
      · by_cases hro : r = io.out1
        · subst hro
          refine Or.inr fun j => ?_
          rcases hoc (⟨0, by decide⟩ : Fin 2) j True.intro with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · by_cases hro : r = io.out2
        · subst hro
          refine Or.inr fun j => ?_
          rcases hoc (⟨1, by decide⟩ : Fin 2) j True.intro with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
  intro A hd hregs hcov pid h1 h2 h3 h4 h5 xs ys zs s₀ hpid hu hx hy hz
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => ys
        | ⟨2, _⟩ => zs
        | ⟨3, _⟩ => fun _ => ChanTy.read .nat s₀ io.in1 (io.read1 pid + io.B1 - 1)
        | ⟨4, _⟩ => fun _ => ChanTy.read .nat s₀ io.in2 (io.read2 pid + io.B2 - 1)
        | ⟨5, _⟩ => fun _ => ChanTy.read .nat s₀ io.in3 (io.read3 pid + io.B3 - 1)
        | ⟨6, _⟩ => fun _ =>
            ChanTy.read .nat s₀ io.out1 (io.write1 pid + io.Bout1 - 1)
        | ⟨_+7, _⟩ => fun _ =>
            ChanTy.read .nat s₀ io.out2 (io.write2 pid + io.Bout2 - 1))
      s₀ hpid rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j _ => by
            have hj : j.val < io.B1 := j.isLt
            have h : io.read1 pid + j.val < A.extent io.in1 := by omega
            exact h
        | ⟨1, _⟩ => fun j _ => by
            have hj : j.val < io.B2 := j.isLt
            have h : io.read2 pid + j.val < A.extent io.in2 := by omega
            exact h
        | ⟨2, _⟩ => fun j _ => by
            have hj : j.val < io.B3 := j.isLt
            have h : io.read3 pid + j.val < A.extent io.in3 := by omega
            exact h
        | ⟨3, _⟩ => fun _ hm => by
            have hm' : 0 < io.read1 pid + io.B1 := hm
            have h : io.read1 pid + io.B1 - 1 < A.extent io.in1 := by omega
            exact h
        | ⟨4, _⟩ => fun _ hm => by
            have hm' : 0 < io.read2 pid + io.B2 := hm
            have h : io.read2 pid + io.B2 - 1 < A.extent io.in2 := by omega
            exact h
        | ⟨5, _⟩ => fun _ hm => by
            have hm' : 0 < io.read3 pid + io.B3 := hm
            have h : io.read3 pid + io.B3 - 1 < A.extent io.in3 := by omega
            exact h
        | ⟨6, _⟩ => fun _ hm => by
            have hm' : 0 < io.write1 pid + io.Bout1 := hm
            have h : io.write1 pid + io.Bout1 - 1 < A.extent io.out1 := by omega
            exact h
        | ⟨_+7, _⟩ => fun _ hm => by
            have hm' : 0 < io.write2 pid + io.Bout2 := hm
            have h : io.write2 pid + io.Bout2 - 1 < A.extent io.out2 := by omega
            exact h)
      (fun o => match o with
        | ⟨0, _⟩ => fun j _ => by
            have hj : j.val < io.Bout1 := j.isLt
            have h : io.write1 pid + j.val < A.extent io.out1 := by omega
            exact h
        | ⟨_+1, _⟩ => fun j _ => by
            have hj : j.val < io.Bout2 := j.isLt
            have h : io.write2 pid + j.val < A.extent io.out2 := by omega
            exact h)
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j _ => hx j
        | ⟨1, _⟩ => fun j _ => hy j
        | ⟨2, _⟩ => fun j _ => hz j
        | ⟨3, _⟩ => fun _ _ => rfl
        | ⟨4, _⟩ => fun _ _ => rfl
        | ⟨5, _⟩ => fun _ _ => rfl
        | ⟨6, _⟩ => fun _ _ => rfl
        | ⟨_+7, _⟩ => fun _ _ => rfl)
  refine ⟨s', hexec, fun j => hval (⟨0, by decide⟩ : Fin 2) j True.intro,
    fun j => hval (⟨1, by decide⟩ : Fin 2) j True.intro, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hn1, hn2⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun oc => match oc with
      | ⟨0, _⟩ => fun j _ => hn1 j
      | ⟨_+1, _⟩ => fun j _ => hn2 j,
      fun t => t.elim0⟩

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

/-- Embed into the unified core — proof plumbing for `Implements.intro`.
The allocation list is the struct's own `bufs` (decoupled from the channel
roles, so in-place duplicate-region wiring survives); the core's `obuf_mem`
field is exactly the intro's two membership side conditions, so they are
threaded through as arguments. -/
private def toU (io : MaskedKernelIO₃ₓ₂)
    (hout1 : io.out1 ∈ io.bufs) (hout2 : io.out2 ∈ io.bufs) : UKernelIO where
  kernel := io.kernel
  nIn := 3
  nOut := 2
  nScr := 0
  bufs := io.bufs
  ity := fun _ => .float
  iarity := fun _ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | ⟨1, _⟩ => io.in2
    | _ => io.in3
  oarity := fun _ => io.B
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.out1
    | _ => io.out2
  obuf_mem := fun o => match o with
    | ⟨0, _⟩ => hout1
    | ⟨_+1, _⟩ => hout2
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ + j.val
    | ⟨1, _⟩ => fun j => io.read2 p₀ + j.val
    | _ => fun j => io.read3 p₀ + j.val
  imask := fun _ _ p₀ _ j => io.mask p₀ j
  owin := fun o _ p₀ _ => match o with
    | ⟨0, _⟩ => fun j => io.write1 p₀ + j.val
    | _ => fun j => io.write2 p₀ + j.val
  omask := fun _ _ p₀ _ j => io.mask p₀ j
  swin := fun t => t.elim0
  smask := fun t => t.elim0

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
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : (io.toU hout1 hout2).Implements
      (fun _p₀ _p₁ vals o => match o with
        | ⟨0, _⟩ => fun j =>
            (f (fun j' => vals (⟨0, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨2, by decide⟩ : Fin 3) j')).1 j
        | ⟨_+1, _⟩ => fun j =>
            (f (fun j' => vals (⟨0, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨2, by decide⟩ : Fin 3) j')).2 j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      exact hts bounds s (fun j hj => hib (⟨0, by decide⟩ : Fin 3) j hj)
        (fun j hj => hib (⟨1, by decide⟩ : Fin 3) j hj)
        (fun j hj => hib (⟨2, by decide⟩ : Fin 3) j hj)
        (fun j hj => hob (⟨0, by decide⟩ : Fin 2) j hj)
        (fun j hj => hob (⟨1, by decide⟩ : Fin 2) j hj)
    · intro s₀ vals hpins
      obtain ⟨s1, hexec, hval1, hval2, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨0, by decide⟩ : Fin 3) j)
          (fun j => vals (⟨1, by decide⟩ : Fin 3) j)
          (fun j => vals (⟨2, by decide⟩ : Fin 3) j)
          (fun j hj => hpins (⟨0, by decide⟩ : Fin 3) j hj)
          (fun j hj => hpins (⟨1, by decide⟩ : Fin 3) j hj)
          (fun j hj => hpins (⟨2, by decide⟩ : Fin 3) j hj)
      refine ⟨s1, hexec, fun o => match o with
        | ⟨0, _⟩ => fun j hj => hval1 j hj
        | ⟨_+1, _⟩ => fun j hj => hval2 j hj, ?_⟩
      intro r o' hoc _hsc
      refine hframe r o' ?_ ?_
      · by_cases hro : r = io.out1
        · subst hro
          refine Or.inr fun j hj => ?_
          rcases hoc (⟨0, by decide⟩ : Fin 2) j hj with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · by_cases hro : r = io.out2
        · subst hro
          refine Or.inr fun j hj => ?_
          rcases hoc (⟨1, by decide⟩ : Fin 2) j hj with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
  intro A hd hregs hcov pid h1 h2 h3 h4 h5 xs ys zs s₀ hpid hu hx hy hz
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => ys
        | _ => zs)
      s₀ hpid rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => h1 j hj
        | ⟨1, _⟩ => fun j hj => h2 j hj
        | ⟨_+2, _⟩ => fun j hj => h3 j hj)
      (fun o => match o with
        | ⟨0, _⟩ => fun j hj => h4 j hj
        | ⟨_+1, _⟩ => fun j hj => h5 j hj)
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hx j hj
        | ⟨1, _⟩ => fun j hj => hy j hj
        | ⟨_+2, _⟩ => fun j hj => hz j hj)
  refine ⟨s', hexec, fun j hj => hval (⟨0, by decide⟩ : Fin 2) j hj,
    fun j hj => hval (⟨1, by decide⟩ : Fin 2) j hj, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hn1, hn2⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun oc => match oc with
      | ⟨0, _⟩ => fun j hj => hn1 j hj
      | ⟨_+1, _⟩ => fun j hj => hn2 j hj,
      fun t => t.elim0⟩
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
