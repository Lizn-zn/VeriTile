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
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ + j.val
    | ⟨1, _⟩ => fun j => io.read2 p₀ + j.val
    | ⟨2, _⟩ => fun _ => io.read1 p₀ + io.B - 1
    | ⟨3, _⟩ => fun _ => io.read2 p₀ + io.B - 1
    | _ => fun _ => io.write p₀ + io.B - 1
  imask := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => True
    | ⟨2, _⟩ => fun _ => 0 < io.read1 p₀ + io.B
    | ⟨3, _⟩ => fun _ => 0 < io.read2 p₀ + io.B
    | _ => fun _ => 0 < io.write p₀ + io.B
  owin := fun _ _ p₀ _ _ j => io.write p₀ + j.val
  omask := fun _ _ _ _ _ _ => True
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
    · intro s₀ vals _hundef hpins
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
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => ys
        | ⟨2, _⟩ => fun _ => ChanTy.read .nat s₀ io.in1 (io.read1 pid + io.B - 1)
        | ⟨3, _⟩ => fun _ => ChanTy.read .nat s₀ io.in2 (io.read2 pid + io.B - 1)
        | ⟨_+4, _⟩ => fun _ => ChanTy.read .nat s₀ io.out (io.write pid + io.B - 1))
      s₀ hpid rfl rfl hu
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
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun _ => io.B
  sbuf := fun t => (io.scratch.get t).1
  iwin := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ + j.val
    | _ => fun j => io.read2 p₀ + j.val
  imask := fun _ _ p₀ _ _ j => io.mask p₀ j
  owin := fun _ _ p₀ _ _ j => io.write p₀ + j.val
  omask := fun _ _ p₀ _ _ j => io.mask p₀ j
  swin := fun t _ p₀ _ _ j => (io.scratch.get t).2 p₀ + j.val
  smask := fun _ _ p₀ _ _ j => io.mask p₀ j

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
    · intro s₀ vals _hundef hpins
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
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | _ => ys)
      s₀ hpid rfl rfl hu
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
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun _ => io.B
  sbuf := fun t => (io.scratch.get t).1
  iwin := fun _ _ p₀ _ _ j => io.read p₀ + j.val
  imask := fun _ _ p₀ _ _ j => io.mask p₀ j
  owin := fun _ _ p₀ _ _ j => io.write p₀ + j.val
  omask := fun _ _ p₀ _ _ j => io.writeMask p₀ j
  swin := fun t _ p₀ _ _ j => (io.scratch.get t).2 p₀ + j.val
  smask := fun _ _ p₀ _ _ j => io.writeMask p₀ j

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
    · intro s₀ vals _hundef hpins
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
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2) (fun _ => xs) s₀ hpid rfl rfl hu
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
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun _ => io.B
  sbuf := fun t => (io.scratch.get t).1
  iwin := fun _ _ p₀ p₁ _ j => io.read p₀ p₁ j
  imask := fun _ _ p₀ p₁ _ j => io.mask p₀ p₁ j
  owin := fun _ _ p₀ p₁ _ j => io.write p₀ p₁ j
  omask := fun _ _ p₀ p₁ _ j => io.writeMask p₀ p₁ j
  swin := fun t _ p₀ p₁ _ j => (io.scratch.get t).2 p₀ p₁ j
  smask := fun _ _ p₀ p₁ _ j => io.writeMask p₀ p₁ j

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
    · intro s₀ vals _hundef hpins
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
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2) (fun _ => xs) s₀ hpid₀ hpid₁ rfl hu
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
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun _ => io.B
  sbuf := fun t => (io.scratch.get t).1
  iwin := fun i _ p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ p₁ j
    | _ => fun j => io.read2 p₀ p₁ j
  imask := fun i _ p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j => io.mask p₀ p₁ j
    | _ => fun j => io.read2Mask p₀ p₁ j
  owin := fun _ _ p₀ p₁ _ j => io.write p₀ p₁ j
  omask := fun _ _ p₀ p₁ _ j => io.writeMask p₀ p₁ j
  swin := fun t _ p₀ p₁ _ j => (io.scratch.get t).2 p₀ p₁ j
  smask := fun _ _ p₀ p₁ _ j => io.writeMask p₀ p₁ j

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
    · intro s₀ vals _hundef hpins
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
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | _ => ys)
      s₀ hpid₀ hpid₁ rfl hu
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

/-- Assembly lemma — **`undef`-pinned sibling** of `Implements.intro`. Proves
the identical `io.Implements f` conclusion, but its `hrun` obligation
additionally receives the pin `s₀.undef = (fun _ _ => 0)`. Kernels whose
masked loads lack an `other=` default read `s₀.undef` at masked-off lanes; if
the body then reduces across the tile (`tl.sum`, `tl.max`), the value stored
at an *active* lane depends on `undef` at *inactive* lanes, and no
`⊨`-shaped per-lane function of the loaded tiles exists that the plain
`intro` (whose `hrun` is quantified over *every* launch state) could be
handed. This sibling threads the `Implements`-hypothesised pin through, so
such a kernel can discharge `hrun` with its masked-off lanes fixed to `0`.
Intended consumer: `log_softmax`'s backward kernel. -/
theorem Implements.intro_undef (io : Masked2DKernelIO₂)
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
      s₀.undef = (fun _ _ => 0) →
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
  -- identical to `Implements.intro`, but the core's `undef` binder is
  -- threaded into `hrun` instead of discarded.
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
    · intro s₀ vals hundef hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨0, by decide⟩ : Fin 2) j)
          (fun j => vals (⟨1, by decide⟩ : Fin 2) j) hundef
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
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | _ => ys)
      s₀ hpid₀ hpid₁ rfl hu
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
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.out1
    | _ => io.out2
  obuf_mem := fun o => by fin_cases o <;> simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ p₁ j
    | _ => fun j => io.read2 p₀ p₁ j
  imask := fun i _ p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j => io.mask p₀ p₁ j
    | _ => fun j => io.read2Mask p₀ p₁ j
  owin := fun o _ p₀ p₁ _ => match o with
    | ⟨0, _⟩ => fun j => io.write1 p₀ p₁ j
    | _ => fun j => io.write2 p₀ p₁ j
  omask := fun o _ p₀ p₁ _ => match o with
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
    · intro s₀ vals _hundef hpins
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
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | _ => ys)
      s₀ hpid₀ hpid₁ rfl hu
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

/-- IO signature of a **3D-grid, general-window** masked two-input /
two-output kernel — the three-program-id sibling of `Masked2DKernelIO₂ₓ₂`.
Every window and mask takes **three** program ids `(pid₀, pid₁, pid₂)`, so a
kernel whose per-lane addressing genuinely reads `tl.program_id(2)` (a 3-D
launch grid, e.g. Mamba chunked-cumsum's `(batch, nchunk, head)` grid) can
state its true, pid₂-dependent windows. The `⊨` spec `f` still takes only the
two leading pids — the unified core supplies `f pid₀ pid₁` in its
postcondition — which is sufficient whenever the third axis enters solely
through the read/write windows and masks (the pid₂-selected head/row is pinned
into the input context, so the output value is pid₂-free). Each input carries
its own read gate and each output its own write gate, all defaulting to
`mask`. No `scratch` field yet: it will be added when a consumer appears. -/
structure Masked3DKernelIO₂ₓ₂ where
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
  /-- Lane `j`'s `in1` read address for program `(pid₀, pid₁, pid₂)`. -/
  read1 : Nat → Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `in2` read address. -/
  read2 : Nat → Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `out1` write address. -/
  write1 : Nat → Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `out2` write address. -/
  write2 : Nat → Nat → Nat → Fin B → Nat
  /-- Program `(pid₀, pid₁, pid₂)`'s **read-active** lanes (`in1`). -/
  mask : Nat → Nat → Nat → Fin B → Prop
  /-- Program `(pid₀, pid₁, pid₂)`'s **`in2` read-active** lanes; defaults to
  `mask` (see `Masked2DKernelIO₂.read2Mask`). -/
  read2Mask : Nat → Nat → Nat → Fin B → Prop := mask
  /-- Program `(pid₀, pid₁, pid₂)`'s **`out1` write-active** lanes; defaults to
  `mask`. -/
  writeMask1 : Nat → Nat → Nat → Fin B → Prop := mask
  /-- Program `(pid₀, pid₁, pid₂)`'s **`out2` write-active** lanes; defaults to
  `mask`. -/
  writeMask2 : Nat → Nat → Nat → Fin B → Prop := mask

namespace Masked3DKernelIO₂ₓ₂

/-- `io.Implements f` — three-program-id sibling of
`Masked2DKernelIO₂ₓ₂.Implements`; the spec `f` returns the pair of the two
outputs' value functions (`.1` for `out1`, `.2` for `out2`). The bounds,
readbacks and frame quantify over all three program ids; `f` sees only the two
leading ids (the third enters through the windows/masks). Frame: every cell
outside the union of the two write-active output windows is untouched. -/
def Implements (io : Masked3DKernelIO₂ₓ₂)
    (f : Nat → Nat → (Fin io.B → ℝ) → (Fin io.B → ℝ) →
      (Fin io.B → ℝ) × (Fin io.B → ℝ)) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.out1, io.out2] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
    (∀ j : Fin io.B, io.mask pid₀ pid₁ pid₂ j →
      io.read1 pid₀ pid₁ pid₂ j < A.extent io.in1) →
    (∀ j : Fin io.B, io.read2Mask pid₀ pid₁ pid₂ j →
      io.read2 pid₀ pid₁ pid₂ j < A.extent io.in2) →
    (∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ pid₂ j →
      io.write1 pid₀ pid₁ pid₂ j < A.extent io.out1) →
    (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ pid₂ j →
      io.write2 pid₀ pid₁ pid₂ j < A.extent io.out2) →
  ∀ (xs ys : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ pid₂ j →
      s₀.readMem io.in1 (io.read1 pid₀ pid₁ pid₂ j) = xs j) →
    (∀ j : Fin io.B, io.read2Mask pid₀ pid₁ pid₂ j →
      s₀.readMem io.in2 (io.read2 pid₀ pid₁ pid₂ j) = ys j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ pid₂ j →
          s'.readMem A.flat (A.addr io.out1 (io.write1 pid₀ pid₁ pid₂ j))
            = (f pid₀ pid₁ xs ys).1 j)
      ∧ (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ pid₂ j →
          s'.readMem A.flat (A.addr io.out2 (io.write2 pid₀ pid₁ pid₂ j))
            = (f pid₀ pid₁ xs ys).2 j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ pid₂ j →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ pid₂ j)) ∧
             (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ pid₂ j →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ pid₂ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => Masked3DKernelIO₂ₓ₂.Implements

/-- Embed into the unified core — proof plumbing for `Implements.intro`
(two float channels with per-channel read gates, two outputs with per-output
write gates, no scratch; every window/mask threads all three program ids). -/
private def toU (io : Masked3DKernelIO₂ₓ₂) : UKernelIO where
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
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.out1
    | _ => io.out2
  obuf_mem := fun o => by fin_cases o <;> simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ p₂ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ p₁ p₂ j
    | _ => fun j => io.read2 p₀ p₁ p₂ j
  imask := fun i _ p₀ p₁ p₂ => match i with
    | ⟨0, _⟩ => fun j => io.mask p₀ p₁ p₂ j
    | _ => fun j => io.read2Mask p₀ p₁ p₂ j
  owin := fun o _ p₀ p₁ p₂ => match o with
    | ⟨0, _⟩ => fun j => io.write1 p₀ p₁ p₂ j
    | _ => fun j => io.write2 p₀ p₁ p₂ j
  omask := fun o _ p₀ p₁ p₂ => match o with
    | ⟨0, _⟩ => fun j => io.writeMask1 p₀ p₁ p₂ j
    | _ => fun j => io.writeMask2 p₀ p₁ p₂ j
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- Assembly lemma — three-program-id sibling of
`Masked2DKernelIO₂ₓ₂.Implements.intro`; `hts`/`hrun` reference all three
`s.pids 0/1/2`, and `hrun`'s frame takes one exclusion condition per output
region. -/
theorem Implements.intro (io : Masked3DKernelIO₂ₓ₂)
    {f : Nat → Nat → (Fin io.B → ℝ) → (Fin io.B → ℝ) →
      (Fin io.B → ℝ) × (Fin io.B → ℝ)}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.read1 (s.pids 0) (s.pids 1) (s.pids 2) j < bounds io.in1) →
      (∀ j : Fin io.B, io.read2Mask (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.read2 (s.pids 0) (s.pids 1) (s.pids 2) j < bounds io.in2) →
      (∀ j : Fin io.B, io.writeMask1 (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.write1 (s.pids 0) (s.pids 1) (s.pids 2) j < bounds io.out1) →
      (∀ j : Fin io.B, io.writeMask2 (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.write2 (s.pids 0) (s.pids 1) (s.pids 2) j < bounds io.out2) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs ys : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) = xs j) →
      (∀ j : Fin io.B, io.read2Mask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) = ys j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMem io.out1 (io.write1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = (f (s₀.pids 0) (s₀.pids 1) xs ys).1 j)
        ∧ (∀ j : Fin io.B, io.writeMask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMem io.out2 (io.write2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = (f (s₀.pids 0) (s₀.pids 1) xs ys).2 j)
        ∧ (∀ r o,
            (r ≠ io.out1 ∨
              ∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
                o ≠ io.write1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) →
            (r ≠ io.out2 ∨
              ∀ j : Fin io.B, io.writeMask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
                o ≠ io.write2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
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
    · intro s₀ vals _hundef hpins
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
  intro A hd hregs hcov pid₀ pid₁ pid₂ h1 h2 h3 h4 xs ys s₀ hpid₀ hpid₁ hpid₂ hu hx hy
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ pid₂
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | _ => ys)
      s₀ hpid₀ hpid₁ hpid₂ hu
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
end Masked3DKernelIO₂ₓ₂

/-- IO signature of a **2D-grid, general-window** masked three-input /
three-output kernel — the widest member of the `Masked2DKernelIO` family
(see `Masked2DKernelIO₁` for the two generalizations over the 1D family).
Each input carries its own read gate and each output its own write gate,
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
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.out1
    | ⟨1, _⟩ => io.out2
    | _ => io.out3
  obuf_mem := fun o => by fin_cases o <;> simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ p₁ j
    | ⟨1, _⟩ => fun j => io.read2 p₀ p₁ j
    | _ => fun j => io.read3 p₀ p₁ j
  imask := fun i _ p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j => io.mask p₀ p₁ j
    | ⟨1, _⟩ => fun j => io.read2Mask p₀ p₁ j
    | _ => fun j => io.read3Mask p₀ p₁ j
  owin := fun o _ p₀ p₁ _ => match o with
    | ⟨0, _⟩ => fun j => io.write1 p₀ p₁ j
    | ⟨1, _⟩ => fun j => io.write2 p₀ p₁ j
    | _ => fun j => io.write3 p₀ p₁ j
  omask := fun o _ p₀ p₁ _ => match o with
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
    · intro s₀ vals _hundef hpins
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
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => ys
        | _ => zs)
      s₀ hpid₀ hpid₁ rfl hu
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
structure BoolMasked2DKernelIO₁ where
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

namespace BoolMasked2DKernelIO₁

/-- `io.Implements f` — bool-input sibling of
`Masked2DKernelIO₁.Implements`. The bool tile `bs` is quantified alongside
the data tile: the launch state holds `bs` on the `.bool` channel of
`mbuf`'s window, and both the spec `f` and the write gate see it. All
address bounds are stated at the **static** `mask` (the superset
trace-safety needs); the data gate `writeMask … bs` only narrows which
output lanes carry the value contract and the frame exclusion. -/
def Implements (io : BoolMasked2DKernelIO₁)
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

@[inherit_doc] scoped infix:25 " ⊨ " => BoolMasked2DKernelIO₁.Implements

/-- Embed into the unified core — proof plumbing for `Implements.intro`.
Channel 0 is the float tile, channel 1 the `.bool` tile (it enters both
the lifted spec and the data-dependent `omask`), and channel 2 is a
contract-free bound witness on the output window: the core states output
bounds only at `omask` (= data-gated) lanes, while the family's
trace-safety obligation needs them at the static `mask` — the witness
channel's `imask := mask` carries that wider bound through. -/
private def toU (io : BoolMasked2DKernelIO₁) : UKernelIO where
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
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j => io.read p₀ p₁ j
    | ⟨1, _⟩ => fun j => io.readm p₀ p₁ j
    | _ => fun j => io.write p₀ p₁ j
  imask := fun _ _ p₀ p₁ _ j => io.mask p₀ p₁ j
  owin := fun _ _ p₀ p₁ _ j => io.write p₀ p₁ j
  omask := fun _ vals p₀ p₁ _ j =>
    io.writeMask p₀ p₁ (fun j' => vals (⟨1, by decide⟩ : Fin 3) j') j
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- Assembly lemma — bool-input sibling of
`Masked2DKernelIO₁.Implements.intro`. `hsub` says the data gate only
narrows the static mask (`fun _ _ _ _ h => h` for the default `writeMask`);
it discharges the write-address bound at data-gated lanes from the
static-mask bound. The bool-tile input hypothesis lives on the region-model
state on both sides of the bridge, so no typed-read transport is needed. -/
theorem Implements.intro (io : BoolMasked2DKernelIO₁)
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
    · intro s₀ vals _hundef hpins
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
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => bs
        | ⟨_+2, _⟩ => fun j =>
            ChanTy.read .nat s₀ io.out (io.write pid₀ pid₁ j))
      s₀ hpid₀ hpid₁ rfl hu
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
end BoolMasked2DKernelIO₁

/-- IO signature of a **2D-grid, general-window** masked kernel with two ℝ
inputs, one **`Bool` input**, and one output — the two-input sibling of
`BoolMasked2DKernelIO₁` (see there for the bool-channel reading, and
`Masked2DKernelIO₁` for the two generalizations over the 1D family). This
is the masked-add shape; in-place updates (`out = in2`) are expressible by
the duplicate-region precedent of `MaskedKernelIO₃ₓ₂`. No `scratch` field
yet: it will be added when a consumer appears. -/
structure BoolMasked2DKernelIO₂ where
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
  tile; defaults to the static `mask` (see `BoolMasked2DKernelIO₁.writeMask`). -/
  writeMask : Nat → Nat → (Fin B → Bool) → Fin B → Prop :=
    fun p₀ p₁ _ j => mask p₀ p₁ j

namespace BoolMasked2DKernelIO₂

/-- `io.Implements f` — two-input sibling of
`BoolMasked2DKernelIO₁.Implements`. -/
def Implements (io : BoolMasked2DKernelIO₂)
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

@[inherit_doc] scoped infix:25 " ⊨ " => BoolMasked2DKernelIO₂.Implements

/-- Embed into the unified core — proof plumbing for `Implements.intro`.
Channels 0/1 are the float tiles, channel 2 the `.bool` tile, and channel 3
a contract-free bound witness on the output window (see
`BoolMasked2DKernelIO₁.toU` for why the witness carries the static-`mask`
write bound). -/
private def toU (io : BoolMasked2DKernelIO₂) : UKernelIO where
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
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ p₁ j
    | ⟨1, _⟩ => fun j => io.read2 p₀ p₁ j
    | ⟨2, _⟩ => fun j => io.readm p₀ p₁ j
    | _ => fun j => io.write p₀ p₁ j
  imask := fun _ _ p₀ p₁ _ j => io.mask p₀ p₁ j
  owin := fun _ _ p₀ p₁ _ j => io.write p₀ p₁ j
  omask := fun _ vals p₀ p₁ _ j =>
    io.writeMask p₀ p₁ (fun j' => vals (⟨2, by decide⟩ : Fin 4) j') j
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- Assembly lemma — two-input sibling of
`BoolMasked2DKernelIO₁.Implements.intro` (see there for `hsub`). -/
theorem Implements.intro (io : BoolMasked2DKernelIO₂)
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
    · intro s₀ vals _hundef hpins
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
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => ys
        | ⟨2, _⟩ => bs
        | ⟨_+3, _⟩ => fun j =>
            ChanTy.read .nat s₀ io.out (io.write pid₀ pid₁ j))
      s₀ hpid₀ hpid₁ rfl hu
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
end BoolMasked2DKernelIO₂

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
  oty := fun _ => .float
  oarity := fun _ => io.Bout
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => (io.scratch.get t).len
  sbuf := fun t => (io.scratch.get t).buf
  iwin := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun j => io.read p₀ + j.val
    | ⟨1, _⟩ => fun _ => io.read p₀ + io.Bin - 1
    | ⟨2, _⟩ => fun _ => io.write p₀ + io.Bout - 1
    | ⟨k+3, h⟩ => fun _ =>
        (io.scratch.get ⟨k, by omega⟩).win p₀
          + (io.scratch.get ⟨k, by omega⟩).len - 1
  imask := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => 0 < io.read p₀ + io.Bin
    | ⟨2, _⟩ => fun _ => 0 < io.write p₀ + io.Bout
    | ⟨k+3, h⟩ => fun _ =>
        0 < (io.scratch.get ⟨k, by omega⟩).win p₀
          + (io.scratch.get ⟨k, by omega⟩).len
  owin := fun _ _ p₀ _ _ j => io.write p₀ + j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun t _ p₀ _ _ k => (io.scratch.get t).win p₀ + k.val
  smask := fun _ _ _ _ _ _ => True

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
    · intro s₀ vals _hundef hpins
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
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
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
      s₀ hpid rfl rfl hu
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
  oty := fun _ => .float
  oarity := fun _ => io.Bout
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => (io.scratch.get t).len
  sbuf := fun t => (io.scratch.get t).buf
  iwin := fun i _ p₀ _ _ => match i with
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
  imask := fun i _ p₀ _ _ => match i with
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
  owin := fun _ _ p₀ _ _ j => io.write p₀ + j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun t _ p₀ _ _ k => (io.scratch.get t).win p₀ + k.val
  smask := fun _ _ _ _ _ _ => True

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
    · intro s₀ vals _hundef hpins
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
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
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
      s₀ hpid rfl rfl hu
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
  oty := fun _ => .float
  oarity := fun o => match o with
    | ⟨0, _⟩ => io.Bout1
    | _ => io.Bout2
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.out1
    | _ => io.out2
  obuf_mem := fun o => by fin_cases o <;> simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ + j.val
    | ⟨1, _⟩ => fun j => io.read2 p₀ + j.val
    | ⟨2, _⟩ => fun j => io.read3 p₀ + j.val
    | ⟨3, _⟩ => fun _ => io.read1 p₀ + io.B1 - 1
    | ⟨4, _⟩ => fun _ => io.read2 p₀ + io.B2 - 1
    | ⟨5, _⟩ => fun _ => io.read3 p₀ + io.B3 - 1
    | ⟨6, _⟩ => fun _ => io.write1 p₀ + io.Bout1 - 1
    | _ => fun _ => io.write2 p₀ + io.Bout2 - 1
  imask := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => True
    | ⟨2, _⟩ => fun _ => True
    | ⟨3, _⟩ => fun _ => 0 < io.read1 p₀ + io.B1
    | ⟨4, _⟩ => fun _ => 0 < io.read2 p₀ + io.B2
    | ⟨5, _⟩ => fun _ => 0 < io.read3 p₀ + io.B3
    | ⟨6, _⟩ => fun _ => 0 < io.write1 p₀ + io.Bout1
    | _ => fun _ => 0 < io.write2 p₀ + io.Bout2
  owin := fun o _ p₀ _ _ => match o with
    | ⟨0, _⟩ => fun j => io.write1 p₀ + j.val
    | _ => fun j => io.write2 p₀ + j.val
  omask := fun _ _ _ _ _ _ => True
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
    · intro s₀ vals _hundef hpins
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
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
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
      s₀ hpid rfl rfl hu
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
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.out1
    | _ => io.out2
  obuf_mem := fun o => match o with
    | ⟨0, _⟩ => hout1
    | ⟨_+1, _⟩ => hout2
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ + j.val
    | ⟨1, _⟩ => fun j => io.read2 p₀ + j.val
    | _ => fun j => io.read3 p₀ + j.val
  imask := fun _ _ p₀ _ _ j => io.mask p₀ j
  owin := fun o _ p₀ _ _ => match o with
    | ⟨0, _⟩ => fun j => io.write1 p₀ + j.val
    | _ => fun j => io.write2 p₀ + j.val
  omask := fun _ _ p₀ _ _ j => io.mask p₀ j
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
    · intro s₀ vals _hundef hpins
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
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => ys
        | _ => zs)
      s₀ hpid rfl rfl hu
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

/-! ### The metadata genre: `Meta*` skins

Windows that read memory. A `Meta*` struct declares scalar **slots** —
per-program single cells on the `.nat`/`.int` channel — whose loaded values
enter the data windows, the masks, and the spec `f` as honest named
binders, pinned to the slot cells by `readMemValue` preconditions (the
ghost-variable discipline). The prefix marks the capability; subscripts
stay pure input×output arity. First shape: the LightLLM
`B_Start_Loc`/`B_Seqlen` pattern (token_softmax family). -/

/-- IO signature of a **metadata-driven** 2D masked one-input / one-output
kernel: two per-program `nat` scalar slots (`mbuf1`/`mbuf2`) whose loaded
values (`m₁`/`m₂`) parametrize the data window, the masks, and the spec. -/
structure MetaMasked2DKernelIO₁ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- First scalar slot's buffer (one `.nat` cell per program). -/
  mbuf1 : RegionName
  /-- Second scalar slot's buffer. -/
  mbuf2 : RegionName
  /-- Input buffer. -/
  inp : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Tile length of the data channels. -/
  B : Nat
  /-- First slot's cell address for program `(pid₀, pid₁)`. -/
  mwin1 : Nat → Nat → Nat
  /-- Second slot's cell address. -/
  mwin2 : Nat → Nat → Nat
  /-- Data read window at `(pid₀, pid₁, m₁, m₂, j)` — the loaded scalars
  are ordinary named arguments. -/
  read : Nat → Nat → Nat → Nat → Fin B → Nat
  /-- Write window. -/
  write : Nat → Nat → Nat → Nat → Fin B → Nat
  /-- Read-active lanes, given the loaded scalars. -/
  mask : Nat → Nat → Nat → Nat → Fin B → Prop
  /-- Write-active lanes; defaults to `mask`. -/
  writeMask : Nat → Nat → Nat → Nat → Fin B → Prop := mask

namespace MetaMasked2DKernelIO₁

/-- `io.Implements f` — the metadata-genre masked Hoare triple. The slot
values `m₁`/`m₂` are universally quantified and pinned to the slot cells
inside the memory precondition, so the window/mask/value contracts all
speak about the *loaded* scalars. -/
def Implements (io : MetaMasked2DKernelIO₁)
    (f : Nat → Nat → Nat → Nat → (Fin io.B → ℝ) → Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.mbuf1, io.mbuf2, io.inp, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (m₁ m₂ : Nat) (xs : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    io.mwin1 pid₀ pid₁ < A.extent io.mbuf1 →
    io.mwin2 pid₀ pid₁ < A.extent io.mbuf2 →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      io.read pid₀ pid₁ m₁ m₂ j < A.extent io.inp) →
    (∀ j : Fin io.B, io.writeMask pid₀ pid₁ m₁ m₂ j →
      io.write pid₀ pid₁ m₁ m₂ j < A.extent io.out) →
    s₀.readMemValue .nat io.mbuf1 (io.mwin1 pid₀ pid₁) = m₁ →
    s₀.readMemValue .nat io.mbuf2 (io.mwin2 pid₀ pid₁) = m₂ →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      s₀.readMem io.inp (io.read pid₀ pid₁ m₁ m₂ j) = xs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask pid₀ pid₁ m₁ m₂ j →
          s'.readMem A.flat (A.addr io.out (io.write pid₀ pid₁ m₁ m₂ j))
            = f pid₀ pid₁ m₁ m₂ xs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.B, io.writeMask pid₀ pid₁ m₁ m₂ j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ m₁ m₂ j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => MetaMasked2DKernelIO₁.Implements

/-- Embed into the unified core: channels 0/1 are the 1-lane `.nat` slots
(always read), channel 2 the float data tile whose window/mask read the
slots' pinned values. -/
private def toU (io : MetaMasked2DKernelIO₁) : UKernelIO where
  kernel := io.kernel
  nIn := 3
  nOut := 1
  nScr := 0
  bufs := [io.mbuf1, io.mbuf2, io.inp, io.out]
  ity := fun i => match i with
    | ⟨0, _⟩ => .nat
    | ⟨1, _⟩ => .nat
    | ⟨2, _⟩ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => 1
    | ⟨2, _⟩ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.mbuf1
    | ⟨1, _⟩ => io.mbuf2
    | ⟨2, _⟩ => io.inp
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => io.mwin1 p₀ p₁
    | ⟨1, _⟩ => fun _ => io.mwin2 p₀ p₁
    | ⟨2, _⟩ => fun j => io.read p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) j
  imask := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => True
    | ⟨2, _⟩ => fun j => io.mask p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) j
  owin := fun _ vals p₀ p₁ _ j => io.write p₀ p₁
    (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
    (vals (⟨1, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) j
  omask := fun _ vals p₀ p₁ _ j => io.writeMask p₀ p₁
    (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
    (vals (⟨1, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) j
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- Assembly lemma: obligations in the skin's named vocabulary — the slots
enter `hts`/`hrun` as pinned named scalars, no lane-constancy plumbing
(slots are 1-lane channels). -/
theorem Implements.intro (io : MetaMasked2DKernelIO₁)
    {f : Nat → Nat → Nat → Nat → (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m₁ m₂ : Nat),
      s.readMemValue .nat io.mbuf1 (io.mwin1 (s.pids 0) (s.pids 1)) = m₁ →
      s.readMemValue .nat io.mbuf2 (io.mwin2 (s.pids 0) (s.pids 1)) = m₂ →
      io.mwin1 (s.pids 0) (s.pids 1) < bounds io.mbuf1 →
      io.mwin2 (s.pids 0) (s.pids 1) < bounds io.mbuf2 →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) m₁ m₂ j →
        io.read (s.pids 0) (s.pids 1) m₁ m₂ j < bounds io.inp) →
      (∀ j : Fin io.B, io.writeMask (s.pids 0) (s.pids 1) m₁ m₂ j →
        io.write (s.pids 0) (s.pids 1) m₁ m₂ j < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (m₁ m₂ : Nat) (xs : Fin io.B → ℝ),
      s₀.readMemValue .nat io.mbuf1 (io.mwin1 (s₀.pids 0) (s₀.pids 1)) = m₁ →
      s₀.readMemValue .nat io.mbuf2 (io.mwin2 (s₀.pids 0) (s₀.pids 1)) = m₂ →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
        s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) m₁ m₂ j) = xs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
            s1.readMem io.out (io.write (s₀.pids 0) (s₀.pids 1) m₁ m₂ j)
              = f (s₀.pids 0) (s₀.pids 1) m₁ m₂ xs j)
        ∧ (∀ r o,
            (r ≠ io.out ∨ ∀ j : Fin io.B,
              io.writeMask (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) m₁ m₂ j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j => f p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (fun j' => vals (⟨2, by decide⟩ : Fin 3) j') j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib hob _hsb
      exact hts bounds s
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (hpins (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨1, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨1, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hib (⟨2, by decide⟩ : Fin 3) j hj)
        (fun j hj => hob (⟨0, by decide⟩ : Fin 1) j hj)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (fun j => vals (⟨2, by decide⟩ : Fin 3) j)
        (hpins (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨1, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hpins (⟨2, by decide⟩ : Fin 3) j hj)
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
  intro A hd hregs hcov pid₀ pid₁ m₁ m₂ xs s₀ hpid₀ hpid₁ hu hb1 hb2 hbr
    hbw hm1 hm2 hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ => m₁
        | ⟨1, _⟩ => fun _ => m₂
        | ⟨2, _⟩ => xs)
      s₀ hpid₀ hpid₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hb1
        | ⟨1, _⟩ => fun _ _ => hb2
        | ⟨2, _⟩ => fun j hj => hbr j hj)
      (fun _o j hj => hbw j hj)
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hm1
        | ⟨1, _⟩ => fun _ _ => hm2
        | ⟨2, _⟩ => fun j hj => hx j hj)
  refine ⟨s', hexec, fun j hj => hval (⟨0, by decide⟩ : Fin 1) j hj, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | hout
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j hj => hout j hj, fun t => t.elim0⟩

end MetaMasked2DKernelIO₁

/-- IO signature of the **cross-entropy metadata shape**: one per-program
`.int` scalar slot (the label, `mbufL`, loaded at the pid-affine cell
`mwinL`), one masked float data row (`inp`, `B` lanes), one label-dependent
single-cell **gather** read (also from `inp`; window `gwin` and read gate
`gmask` both eat the loaded label), and two per-program single-cell float
outputs (`out1`/`out2`, each with its own cell address and write gate).
The subscript is pure data-input × output arity (row + gather cell ↦ two
scalar cells); the slot is capability (`Meta`), not arity. Intended
consumers: the TritonBench-G cross-entropy forward family
(`cross_entropy1` / `cross_entropy2` / `cross_entropy_ops` — per-
`(row_idx, col_block_idx)` loss/LSE cells with the guarded
`logits[label]` gather). Following the family precedent there is no
`scratch` field until a consumer needs one. -/
structure MetaGatherMasked2DKernelIO₂ₓ₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- Label slot's buffer (one `.int` cell per program). -/
  mbufL : RegionName
  /-- Data input buffer — both the masked row and the gather cell read it. -/
  inp : RegionName
  /-- First output buffer. -/
  out1 : RegionName
  /-- Second output buffer. -/
  out2 : RegionName
  /-- Tile length of the masked data row. -/
  B : Nat
  /-- Label slot's cell address for program `(pid₀, pid₁)`. -/
  mwinL : Nat → Nat → Nat
  /-- Data-row read window at `(pid₀, pid₁, lab, j)` — the loaded label is
  an ordinary named argument. -/
  read : Nat → Nat → Int → Fin B → Nat
  /-- Data-row read-active lanes, given the loaded label. -/
  mask : Nat → Nat → Int → Fin B → Prop
  /-- Gather cell's read address (label-dependent). -/
  gwin : Nat → Nat → Int → Nat
  /-- Gather read gate: the label-dependent condition under which the
  kernel actually loads the gather cell. -/
  gmask : Nat → Nat → Int → Prop
  /-- `out1`'s cell address for program `(pid₀, pid₁)`. -/
  write1 : Nat → Nat → Int → Nat
  /-- `out2`'s cell address. -/
  write2 : Nat → Nat → Int → Nat
  /-- `out1`'s write gate; defaults to always-on (the genre's loss/LSE
  stores are unconditional). -/
  writeMask1 : Nat → Nat → Int → Prop := fun _ _ _ => True
  /-- `out2`'s write gate; defaults to always-on. -/
  writeMask2 : Nat → Nat → Int → Prop := fun _ _ _ => True

namespace MetaGatherMasked2DKernelIO₂ₓ₂

/-- `io.Implements f` — the cross-entropy-genre masked Hoare triple. The
label `lab` is universally quantified and pinned to the slot cell inside
the memory precondition; the data row `xs` is pinned on the read-active
lanes and the gather cell `g` under its gate, so window/mask/value
contracts all speak about the *loaded* label. The spec `f` returns the
pair of the two output cells' values (`.1` for `out1`, `.2` for `out2`).
Frame: every cell outside the two gated output cells is untouched. -/
def Implements (io : MetaGatherMasked2DKernelIO₂ₓ₂)
    (f : Nat → Nat → Int → (Fin io.B → ℝ) → ℝ → ℝ × ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.mbufL, io.inp, io.out1, io.out2] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (lab : Int) (xs : Fin io.B → ℝ) (g : ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    io.mwinL pid₀ pid₁ < A.extent io.mbufL →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ lab j →
      io.read pid₀ pid₁ lab j < A.extent io.inp) →
    (io.gmask pid₀ pid₁ lab → io.gwin pid₀ pid₁ lab < A.extent io.inp) →
    (io.writeMask1 pid₀ pid₁ lab →
      io.write1 pid₀ pid₁ lab < A.extent io.out1) →
    (io.writeMask2 pid₀ pid₁ lab →
      io.write2 pid₀ pid₁ lab < A.extent io.out2) →
    s₀.readMemValue .int io.mbufL (io.mwinL pid₀ pid₁) = lab →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ lab j →
      s₀.readMem io.inp (io.read pid₀ pid₁ lab j) = xs j) →
    (io.gmask pid₀ pid₁ lab →
      s₀.readMem io.inp (io.gwin pid₀ pid₁ lab) = g) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (io.writeMask1 pid₀ pid₁ lab →
          s'.readMem A.flat (A.addr io.out1 (io.write1 pid₀ pid₁ lab))
            = (f pid₀ pid₁ lab xs g).1)
      ∧ (io.writeMask2 pid₀ pid₁ lab →
          s'.readMem A.flat (A.addr io.out2 (io.write2 pid₀ pid₁ lab))
            = (f pid₀ pid₁ lab xs g).2)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((io.writeMask1 pid₀ pid₁ lab →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ lab)) ∧
             (io.writeMask2 pid₀ pid₁ lab →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ lab)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => MetaGatherMasked2DKernelIO₂ₓ₂.Implements

/-- Embed into the unified core: channel 0 is the 1-lane `.int` label slot
(always read), channel 1 the `B`-lane float data row and channel 2 the
1-lane float gather cell, both with windows/masks reading the slot's
pinned value; the two outputs are 1-lane gated cells. -/
private def toU (io : MetaGatherMasked2DKernelIO₂ₓ₂) : UKernelIO where
  kernel := io.kernel
  nIn := 3
  nOut := 2
  nScr := 0
  bufs := [io.mbufL, io.inp, io.out1, io.out2]
  ity := fun i => match i with
    | ⟨0, _⟩ => .int
    | ⟨1, _⟩ => .float
    | ⟨2, _⟩ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => io.B
    | ⟨2, _⟩ => 1
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.mbufL
    | ⟨1, _⟩ => io.inp
    | ⟨2, _⟩ => io.inp
  oty := fun _ => .float
  oarity := fun _ => 1
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.out1
    | _ => io.out2
  obuf_mem := fun o => by fin_cases o <;> simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => io.mwinL p₀ p₁
    | ⟨1, _⟩ => fun j => io.read p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨2, _⟩ => fun _ => io.gwin p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
  imask := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun j => io.mask p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨2, _⟩ => fun _ => io.gmask p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
  owin := fun o vals p₀ p₁ _ => match o with
    | ⟨0, _⟩ => fun _ => io.write1 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
    | _ => fun _ => io.write2 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
  omask := fun o vals p₀ p₁ _ => match o with
    | ⟨0, _⟩ => fun _ => io.writeMask1 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
    | _ => fun _ => io.writeMask2 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- Assembly lemma: obligations in the skin's named vocabulary — the label
enters `hts`/`hrun` as one pinned named `Int` scalar (no lane-constancy
plumbing: the slot and the gather cell are 1-lane channels), the gather
pin and bound are gated by `gmask`, and `hrun`'s frame takes one exclusion
condition per output cell. -/
theorem Implements.intro (io : MetaGatherMasked2DKernelIO₂ₓ₂)
    {f : Nat → Nat → Int → (Fin io.B → ℝ) → ℝ → ℝ × ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (lab : Int),
      s.readMemValue .int io.mbufL (io.mwinL (s.pids 0) (s.pids 1)) = lab →
      io.mwinL (s.pids 0) (s.pids 1) < bounds io.mbufL →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) lab j →
        io.read (s.pids 0) (s.pids 1) lab j < bounds io.inp) →
      (io.gmask (s.pids 0) (s.pids 1) lab →
        io.gwin (s.pids 0) (s.pids 1) lab < bounds io.inp) →
      (io.writeMask1 (s.pids 0) (s.pids 1) lab →
        io.write1 (s.pids 0) (s.pids 1) lab < bounds io.out1) →
      (io.writeMask2 (s.pids 0) (s.pids 1) lab →
        io.write2 (s.pids 0) (s.pids 1) lab < bounds io.out2) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (lab : Int) (xs : Fin io.B → ℝ) (g : ℝ),
      s₀.readMemValue .int io.mbufL (io.mwinL (s₀.pids 0) (s₀.pids 1)) = lab →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) lab j →
        s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) lab j) = xs j) →
      (io.gmask (s₀.pids 0) (s₀.pids 1) lab →
        s₀.readMem io.inp (io.gwin (s₀.pids 0) (s₀.pids 1) lab) = g) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (io.writeMask1 (s₀.pids 0) (s₀.pids 1) lab →
            s1.readMem io.out1 (io.write1 (s₀.pids 0) (s₀.pids 1) lab)
              = (f (s₀.pids 0) (s₀.pids 1) lab xs g).1)
        ∧ (io.writeMask2 (s₀.pids 0) (s₀.pids 1) lab →
            s1.readMem io.out2 (io.write2 (s₀.pids 0) (s₀.pids 1) lab)
              = (f (s₀.pids 0) (s₀.pids 1) lab xs g).2)
        ∧ (∀ r o,
            (r ≠ io.out1 ∨ (io.writeMask1 (s₀.pids 0) (s₀.pids 1) lab →
              o ≠ io.write1 (s₀.pids 0) (s₀.pids 1) lab)) →
            (r ≠ io.out2 ∨ (io.writeMask2 (s₀.pids 0) (s₀.pids 1) lab →
              o ≠ io.write2 (s₀.pids 0) (s₀.pids 1) lab)) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals o => match o with
        | ⟨0, _⟩ => fun _ =>
            (f p₀ p₁ (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
              (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
              (vals (⟨2, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))).1
        | ⟨_+1, _⟩ => fun _ =>
            (f p₀ p₁ (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
              (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
              (vals (⟨2, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))).2) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib hob _hsb
      exact hts bounds s
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (hpins (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hib (⟨1, by decide⟩ : Fin 3) j hj)
        (fun hgm => hib (⟨2, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) hgm)
        (fun h1 => hob (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1) h1)
        (fun h2 => hob (⟨1, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1) h2)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval1, hval2, hframe⟩ := hrun s₀
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (fun j => vals (⟨1, by decide⟩ : Fin 3) j)
        (vals (⟨2, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (hpins (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hpins (⟨1, by decide⟩ : Fin 3) j hj)
        (fun hgm => hpins (⟨2, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) hgm)
      refine ⟨s1, hexec, fun o => match o with
        | ⟨0, _⟩ => fun _ h1 => hval1 h1
        | ⟨_+1, _⟩ => fun _ h2 => hval2 h2, ?_⟩
      intro r o' hoc _hsc
      refine hframe r o' ?_ ?_
      · by_cases hro : r = io.out1
        · subst hro
          refine Or.inr fun h1 => ?_
          rcases hoc (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1) h1
            with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · by_cases hro : r = io.out2
        · subst hro
          refine Or.inr fun h2 => ?_
          rcases hoc (⟨1, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1) h2
            with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ lab xs g s₀ hpid₀ hpid₁ hu hbL hbr hbg
    hbw1 hbw2 hmL hx hg
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ => lab
        | ⟨1, _⟩ => xs
        | ⟨2, _⟩ => fun _ => g)
      s₀ hpid₀ hpid₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hbL
        | ⟨1, _⟩ => fun j hj => hbr j hj
        | ⟨2, _⟩ => fun _ hgm => hbg hgm)
      (fun o => match o with
        | ⟨0, _⟩ => fun _ h1 => hbw1 h1
        | ⟨_+1, _⟩ => fun _ h2 => hbw2 h2)
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hmL
        | ⟨1, _⟩ => fun j hj => hx j hj
        | ⟨2, _⟩ => fun _ hgm => hg hgm)
  refine ⟨s', hexec,
    fun h1 => hval (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1) h1,
    fun h2 => hval (⟨1, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1) h2, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hn1, hn2⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun o => match o with
      | ⟨0, _⟩ => fun _ hj => hn1 hj
      | ⟨_+1, _⟩ => fun _ hj => hn2 hj,
      fun t => t.elim0⟩

end MetaGatherMasked2DKernelIO₂ₓ₂


/-! ### The scatter genre: `Scatter*` skins

Data-dependent **write** addresses. A `Scatter*` struct declares an
**index channel** — a `.nat` tile (a named field, never part of the
arity subscript) whose pinned per-lane values `ids` enter the write
window and the write gate. Scatter readback is only meaningful when no
two write-active lanes collide, so the readback leg inside `Implements`
is guarded by a per-context `WriteInj` antecedent, while the frame and
all bounds stay unconditional: the kernel writes the raw scatter cells
whether or not they collide, so in the core embedding those cells are
shadowed by a scratch channel (same buffer, same window, ungated mask),
which carries both the unconditional frame exclusion and the
unconditional trace-safety write bound. -/

/-- IO signature of a **compaction-scatter** masked one-input /
one-output kernel with a `.bool` gate tile: one float data tile (`inp`),
one bool select tile (`mbuf`), and one `.nat` **index tile** (`idxbuf`)
whose loaded values `ids` give each lane's scatter destination
`write … ids j`; the store is gated by `writeMask … bs ids` (typically
`mask ∧ bs j = true`). Intended consumer: the TritonBench-G compaction
scatter `masked_select` (data tile + select-mask tile + prefix-sum tile
whose values give the destinations `prefix_sum[j] − 1`).

**Injectivity design.** The readback leg of `Implements` is guarded by
the *per-pinned-context* antecedent `WriteInj pid₀ pid₁ bs ids` (no two
write-active lanes share a destination) — it is **not** a hypothesis of
`Implements.intro`: a `∀`-quantified intro hypothesis would demand
injectivity for arbitrary index-buffer contents (false for, e.g.,
duplicated indices), whereas the per-context antecedent is exactly the
host-side no-duplicate-destination guarantee (the consumers' `hOutInj` /
`hUniq` side conditions), which `hrun` receives as the antecedent of its
own value leg and threads to its scatter-readback lemma. The frame and
the trace-safety write bound are stated at the *ungated* `writeMask`
lanes, so they hold with or without injectivity. -/
structure BoolScatterMasked2DKernelIO₁ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- Input buffer (ℝ channel). -/
  inp : RegionName
  /-- Boolean input buffer (`.bool` channel — the select gate). -/
  mbuf : RegionName
  /-- Index buffer (`.nat` channel — the scatter destinations). -/
  idxbuf : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Tile length: each program instance owns `B`-lane windows. -/
  B : Nat
  /-- Lane `j`'s `inp` read address for program `(pid₀, pid₁)`. -/
  read : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `mbuf` read address (the bool tile's window). -/
  readm : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `idxbuf` read address (the index tile's window). -/
  readx : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s write address given the loaded index tile — the
  data-dependent scatter destination. -/
  write : Nat → Nat → (Fin B → Nat) → Fin B → Nat
  /-- Program `(pid₀, pid₁)`'s **read-active** lanes — the bounds /
  trace-safety superset. -/
  mask : Nat → Nat → Fin B → Prop
  /-- **Write-active** lanes given the loaded bool and index tiles;
  defaults to the static `mask`. -/
  writeMask : Nat → Nat → (Fin B → Bool) → (Fin B → Nat) → Fin B → Prop :=
    fun p₀ p₁ _ _ j => mask p₀ p₁ j

namespace BoolScatterMasked2DKernelIO₁

/-- No two write-active lanes share a scatter destination — the
per-pinned-context injectivity that scatter readback needs. For
`masked_select` this is the host prefix-sum's no-duplicate-destination
guarantee. -/
def WriteInj (io : BoolScatterMasked2DKernelIO₁) (p₀ p₁ : Nat)
    (bs : Fin io.B → Bool) (ids : Fin io.B → Nat) : Prop :=
  ∀ j k : Fin io.B, io.writeMask p₀ p₁ bs ids j →
    io.writeMask p₀ p₁ bs ids k →
    io.write p₀ p₁ ids j = io.write p₀ p₁ ids k → j = k

/-- `io.Implements f` — the compaction-scatter masked Hoare triple. The
bool tile `bs` and the index tile `ids` are quantified alongside the data
tile and pinned on the read-active lanes; the write addresses eat the
*loaded* `ids`. The readback leg is guarded by `WriteInj` (see the
struct docstring); the frame excludes the raw (ungated) scatter cells. -/
def Implements (io : BoolScatterMasked2DKernelIO₁)
    (f : Nat → Nat → (Fin io.B → Bool) → (Fin io.B → Nat) →
      (Fin io.B → ℝ) → Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp, io.mbuf, io.idxbuf, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (bs : Fin io.B → Bool) (ids : Fin io.B → Nat) (xs : Fin io.B → ℝ)
    (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.read pid₀ pid₁ j < A.extent io.inp) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.readm pid₀ pid₁ j < A.extent io.mbuf) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.readx pid₀ pid₁ j < A.extent io.idxbuf) →
    (∀ j : Fin io.B, io.writeMask pid₀ pid₁ bs ids j →
      io.write pid₀ pid₁ ids j < A.extent io.out) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMem io.inp (io.read pid₀ pid₁ j) = xs j) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMemValue .bool io.mbuf (io.readm pid₀ pid₁ j) = bs j) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMemValue .nat io.idxbuf (io.readx pid₀ pid₁ j) = ids j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (io.WriteInj pid₀ pid₁ bs ids →
          ∀ j : Fin io.B, io.writeMask pid₀ pid₁ bs ids j →
            s'.readMem A.flat (A.addr io.out (io.write pid₀ pid₁ ids j))
              = f pid₀ pid₁ bs ids xs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.B, io.writeMask pid₀ pid₁ bs ids j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ ids j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => BoolScatterMasked2DKernelIO₁.Implements

/-- Embed into the unified core: channel 0 is the float data tile,
channel 1 the `.bool` select tile, channel 2 the `.nat` index tile.
The single output's window eats the index channel's pinned values and
its mask is the write gate **conjoined with `WriteInj`**; a scratch
channel shadows the same cells with the *ungated* write gate, carrying
the unconditional frame exclusion and write bound. -/
private def toU (io : BoolScatterMasked2DKernelIO₁) : UKernelIO where
  kernel := io.kernel
  nIn := 3
  nOut := 1
  nScr := 1
  bufs := [io.inp, io.mbuf, io.idxbuf, io.out]
  ity := fun i => match i with
    | ⟨0, _⟩ => .float
    | ⟨1, _⟩ => .bool
    | ⟨2, _⟩ => .nat
  iarity := fun _ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.inp
    | ⟨1, _⟩ => io.mbuf
    | ⟨2, _⟩ => io.idxbuf
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun _ => io.B
  sbuf := fun _ => io.out
  iwin := fun i _ p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j => io.read p₀ p₁ j
    | ⟨1, _⟩ => fun j => io.readm p₀ p₁ j
    | ⟨2, _⟩ => fun j => io.readx p₀ p₁ j
  imask := fun _ _ p₀ p₁ _ j => io.mask p₀ p₁ j
  owin := fun _ vals p₀ p₁ _ j => io.write p₀ p₁
    (fun j' => vals (⟨2, by decide⟩ : Fin 3) j') j
  omask := fun _ vals p₀ p₁ _ j =>
    io.writeMask p₀ p₁ (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
      (fun j' => vals (⟨2, by decide⟩ : Fin 3) j') j
    ∧ io.WriteInj p₀ p₁ (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
        (fun j' => vals (⟨2, by decide⟩ : Fin 3) j')
  swin := fun _ vals p₀ p₁ _ j => io.write p₀ p₁
    (fun j' => vals (⟨2, by decide⟩ : Fin 3) j') j
  smask := fun _ vals p₀ p₁ _ j =>
    io.writeMask p₀ p₁ (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
      (fun j' => vals (⟨2, by decide⟩ : Fin 3) j') j

/-- Assembly lemma: obligations in the skin's named vocabulary — the
bool/index tiles enter `hts`/`hrun` as pinned tiles, and `hrun`'s value
leg receives the per-context `WriteInj` antecedent (the consumer threads
its `hOutInj`/`hUniq` side condition there); `hrun`'s frame and the
write bound are at the ungated `writeMask` lanes. -/
theorem Implements.intro (io : BoolScatterMasked2DKernelIO₁)
    {f : Nat → Nat → (Fin io.B → Bool) → (Fin io.B → Nat) →
      (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (bs : Fin io.B → Bool) (ids : Fin io.B → Nat),
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        s.readMemValue .bool io.mbuf (io.readm (s.pids 0) (s.pids 1) j)
          = bs j) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        s.readMemValue .nat io.idxbuf (io.readx (s.pids 0) (s.pids 1) j)
          = ids j) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.read (s.pids 0) (s.pids 1) j < bounds io.inp) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.readm (s.pids 0) (s.pids 1) j < bounds io.mbuf) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.readx (s.pids 0) (s.pids 1) j < bounds io.idxbuf) →
      (∀ j : Fin io.B, io.writeMask (s.pids 0) (s.pids 1) bs ids j →
        io.write (s.pids 0) (s.pids 1) ids j < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (bs : Fin io.B → Bool)
        (ids : Fin io.B → Nat) (xs : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) j) = xs j) →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMemValue .bool io.mbuf (io.readm (s₀.pids 0) (s₀.pids 1) j)
          = bs j) →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMemValue .nat io.idxbuf (io.readx (s₀.pids 0) (s₀.pids 1) j)
          = ids j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (io.WriteInj (s₀.pids 0) (s₀.pids 1) bs ids →
            ∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) bs ids j →
              s1.readMem io.out (io.write (s₀.pids 0) (s₀.pids 1) ids j)
                = f (s₀.pids 0) (s₀.pids 1) bs ids xs j)
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) bs ids j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) ids j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁ (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
          (fun j' => vals (⟨2, by decide⟩ : Fin 3) j')
          (fun j' => vals (⟨0, by decide⟩ : Fin 3) j') j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib _hob hsb
      exact hts bounds s
        (fun j => vals (⟨1, by decide⟩ : Fin 3) j)
        (fun j => vals (⟨2, by decide⟩ : Fin 3) j)
        (fun j hj => hpins (⟨1, by decide⟩ : Fin 3) j hj)
        (fun j hj => hpins (⟨2, by decide⟩ : Fin 3) j hj)
        (fun j hj => hib (⟨0, by decide⟩ : Fin 3) j hj)
        (fun j hj => hib (⟨1, by decide⟩ : Fin 3) j hj)
        (fun j hj => hib (⟨2, by decide⟩ : Fin 3) j hj)
        (fun j hj => hsb (⟨0, by decide⟩ : Fin 1) j hj)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨1, by decide⟩ : Fin 3) j)
          (fun j => vals (⟨2, by decide⟩ : Fin 3) j)
          (fun j => vals (⟨0, by decide⟩ : Fin 3) j)
          (fun j hj => hpins (⟨0, by decide⟩ : Fin 3) j hj)
          (fun j hj => hpins (⟨1, by decide⟩ : Fin 3) j hj)
          (fun j hj => hpins (⟨2, by decide⟩ : Fin 3) j hj)
      refine ⟨s1, hexec, fun _o j hj => hval hj.2 j hj.1, ?_⟩
      intro r o' _hoc hsc
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun j hj => ?_
        rcases hsc (⟨0, by decide⟩ : Fin 1) j hj with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ bs ids xs s₀ hpid₀ hpid₁ hu hbr hbm hbx
    hbw hx hb hi
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => bs
        | ⟨2, _⟩ => ids)
      s₀ hpid₀ hpid₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hbr j hj
        | ⟨1, _⟩ => fun j hj => hbm j hj
        | ⟨2, _⟩ => fun j hj => hbx j hj)
      (fun _o j hj => hbw j hj.1)
      (fun _t j hj => hbw j hj)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hx j hj
        | ⟨1, _⟩ => fun j hj => hb j hj
        | ⟨2, _⟩ => fun j hj => hi j hj)
  refine ⟨s', hexec,
    fun hinj j hj => hval (⟨0, by decide⟩ : Fin 1) j ⟨hj, hinj⟩, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | hn
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j hj => hn j hj.1, fun _t j hj => hn j hj⟩

end BoolScatterMasked2DKernelIO₁

/-- IO signature of the **penalty gather–scatter shape** (`Meta` slots +
`Scatter` writes): three per-program float scalar slots (`fbuf1`–`fbuf3`,
cells `fwin1`–`fwin3`), two per-program `.nat` scalar slots on **one**
metadata buffer (`mbuf`, cells `mwin1`/`mwin2` — the cumsum window
bounds), a `.nat` **index tile** (`idbuf`, window `readi`, values `ids` —
the gathered token ids), a `.nat` payload tile (`cntbuf`, window `readc`,
values `cnts` — the token counts), and one float data channel (`inp`)
**gather-read** at the index-dependent window `read … ids` and
**scatter-written** at `write … ids` (the in-place consumer instantiates
`out := inp` — duplicate-region wiring is supported by `bufs`). The
subscript is pure float-data arity (one gathered input ↦ one scattered
output); slots and `.nat` tiles are capability fields. Intended consumer:
the TritonBench-G LightLLM penalty kernel `apply_penalty`
(per-`cur_batch` presence/frequency/repetition penalties applied in place
to `Logits` at the gathered token positions).

**Injectivity design.** Same as `BoolScatterMasked2DKernelIO₁`: the
readback leg of `Implements` is guarded by the per-pinned-context
`WriteInj pid₀ pid₁ m₁ m₂ ids` antecedent (the consumer's `hUniq`
distinct-active-token-ids side condition) rather than by an
`Implements.intro` hypothesis, and the frame / trace-safety write bound
stay at the ungated `writeMask` lanes via the core scratch shadow. -/
structure MetaScatterMasked2DKernelIO₁ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- First float scalar slot's buffer (one ℝ cell per program). -/
  fbuf1 : RegionName
  /-- Second float scalar slot's buffer. -/
  fbuf2 : RegionName
  /-- Third float scalar slot's buffer. -/
  fbuf3 : RegionName
  /-- The `.nat` metadata buffer carrying both scalar slots (the
  consumer's cumulative-sequence-length array). -/
  mbuf : RegionName
  /-- Index buffer (`.nat` tile — the gather/scatter token ids). -/
  idbuf : RegionName
  /-- Payload buffer (`.nat` tile — the per-token counts). -/
  cntbuf : RegionName
  /-- Float data buffer, gather-read at the index-dependent window. -/
  inp : RegionName
  /-- Output buffer, scatter-written (the in-place consumer sets
  `out := inp`). -/
  out : RegionName
  /-- Tile length: each program instance owns `B`-lane windows. -/
  B : Nat
  /-- First float slot's cell address for program `(pid₀, pid₁)`. -/
  fwin1 : Nat → Nat → Nat
  /-- Second float slot's cell address. -/
  fwin2 : Nat → Nat → Nat
  /-- Third float slot's cell address. -/
  fwin3 : Nat → Nat → Nat
  /-- First `.nat` slot's cell address in `mbuf`. -/
  mwin1 : Nat → Nat → Nat
  /-- Second `.nat` slot's cell address in `mbuf`. -/
  mwin2 : Nat → Nat → Nat
  /-- Lane `j`'s `idbuf` read address at `(pid₀, pid₁, m₁, m₂)` — the
  loaded `.nat` scalars are ordinary named arguments. -/
  readi : Nat → Nat → Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `cntbuf` read address. -/
  readc : Nat → Nat → Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `inp` **gather** read address, given the loaded index
  tile. -/
  read : Nat → Nat → Nat → Nat → (Fin B → Nat) → Fin B → Nat
  /-- Read-active lanes, given the loaded `.nat` scalars (the consumer's
  `m₁ + j < m₂` batch window). -/
  mask : Nat → Nat → Nat → Nat → Fin B → Prop
  /-- Lane `j`'s **scatter** write address, given the loaded index tile. -/
  write : Nat → Nat → Nat → Nat → (Fin B → Nat) → Fin B → Nat
  /-- Write-active lanes given the loaded index tile; defaults to `mask`. -/
  writeMask : Nat → Nat → Nat → Nat → (Fin B → Nat) → Fin B → Prop :=
    fun p₀ p₁ m₁ m₂ _ j => mask p₀ p₁ m₁ m₂ j

namespace MetaScatterMasked2DKernelIO₁

/-- No two write-active lanes share a scatter destination — the
per-pinned-context injectivity that scatter readback needs. For
`apply_penalty` this is the distinct-active-token-ids side condition
(`hUniq`). -/
def WriteInj (io : MetaScatterMasked2DKernelIO₁) (p₀ p₁ m₁ m₂ : Nat)
    (ids : Fin io.B → Nat) : Prop :=
  ∀ j k : Fin io.B, io.writeMask p₀ p₁ m₁ m₂ ids j →
    io.writeMask p₀ p₁ m₁ m₂ ids k →
    io.write p₀ p₁ m₁ m₂ ids j = io.write p₀ p₁ m₁ m₂ ids k → j = k

/-- `io.Implements f` — the penalty gather–scatter masked Hoare triple.
The float slots `g₁ g₂ g₃`, the `.nat` slots `m₁ m₂`, the index tile
`ids`, the payload tile `cnts`, and the gathered data tile `xs` are all
universally quantified and pinned inside the memory precondition, so the
window/mask/value contracts speak about the *loaded* values throughout.
The readback leg is guarded by `WriteInj` (see the struct docstring);
the frame excludes the raw (ungated) scatter cells. -/
def Implements (io : MetaScatterMasked2DKernelIO₁)
    (f : Nat → Nat → ℝ → ℝ → ℝ → Nat → Nat → (Fin io.B → Nat) →
      (Fin io.B → Nat) → (Fin io.B → ℝ) → Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.fbuf1, io.fbuf2, io.fbuf3, io.mbuf, io.idbuf,
      io.cntbuf, io.inp, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (g₁ g₂ g₃ : ℝ) (m₁ m₂ : Nat) (ids cnts : Fin io.B → Nat)
    (xs : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    io.fwin1 pid₀ pid₁ < A.extent io.fbuf1 →
    io.fwin2 pid₀ pid₁ < A.extent io.fbuf2 →
    io.fwin3 pid₀ pid₁ < A.extent io.fbuf3 →
    io.mwin1 pid₀ pid₁ < A.extent io.mbuf →
    io.mwin2 pid₀ pid₁ < A.extent io.mbuf →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      io.readi pid₀ pid₁ m₁ m₂ j < A.extent io.idbuf) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      io.readc pid₀ pid₁ m₁ m₂ j < A.extent io.cntbuf) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      io.read pid₀ pid₁ m₁ m₂ ids j < A.extent io.inp) →
    (∀ j : Fin io.B, io.writeMask pid₀ pid₁ m₁ m₂ ids j →
      io.write pid₀ pid₁ m₁ m₂ ids j < A.extent io.out) →
    s₀.readMem io.fbuf1 (io.fwin1 pid₀ pid₁) = g₁ →
    s₀.readMem io.fbuf2 (io.fwin2 pid₀ pid₁) = g₂ →
    s₀.readMem io.fbuf3 (io.fwin3 pid₀ pid₁) = g₃ →
    s₀.readMemValue .nat io.mbuf (io.mwin1 pid₀ pid₁) = m₁ →
    s₀.readMemValue .nat io.mbuf (io.mwin2 pid₀ pid₁) = m₂ →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      s₀.readMemValue .nat io.idbuf (io.readi pid₀ pid₁ m₁ m₂ j) = ids j) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      s₀.readMemValue .nat io.cntbuf (io.readc pid₀ pid₁ m₁ m₂ j)
        = cnts j) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      s₀.readMem io.inp (io.read pid₀ pid₁ m₁ m₂ ids j) = xs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (io.WriteInj pid₀ pid₁ m₁ m₂ ids →
          ∀ j : Fin io.B, io.writeMask pid₀ pid₁ m₁ m₂ ids j →
            s'.readMem A.flat
                (A.addr io.out (io.write pid₀ pid₁ m₁ m₂ ids j))
              = f pid₀ pid₁ g₁ g₂ g₃ m₁ m₂ ids cnts xs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.B, io.writeMask pid₀ pid₁ m₁ m₂ ids j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ m₁ m₂ ids j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => MetaScatterMasked2DKernelIO₁.Implements

/-- Embed into the unified core: channels 0–2 are the 1-lane float
slots, channels 3–4 the 1-lane `.nat` slots (both on `mbuf`), channel 5
the `.nat` index tile and channel 6 the `.nat` payload tile (windows
reading the slots' pinned values), channel 7 the float gather tile
(window reading slots *and* the index channel). The output window eats
the index channel and its mask is the write gate conjoined with
`WriteInj`; a scratch channel shadows the same cells ungated. -/
private def toU (io : MetaScatterMasked2DKernelIO₁) : UKernelIO where
  kernel := io.kernel
  nIn := 8
  nOut := 1
  nScr := 1
  bufs := [io.fbuf1, io.fbuf2, io.fbuf3, io.mbuf, io.idbuf, io.cntbuf,
    io.inp, io.out]
  ity := fun i => match i with
    | ⟨0, _⟩ => .float
    | ⟨1, _⟩ => .float
    | ⟨2, _⟩ => .float
    | ⟨3, _⟩ => .nat
    | ⟨4, _⟩ => .nat
    | ⟨5, _⟩ => .nat
    | ⟨6, _⟩ => .nat
    | ⟨7, _⟩ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => 1
    | ⟨2, _⟩ => 1
    | ⟨3, _⟩ => 1
    | ⟨4, _⟩ => 1
    | ⟨5, _⟩ => io.B
    | ⟨6, _⟩ => io.B
    | ⟨7, _⟩ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.fbuf1
    | ⟨1, _⟩ => io.fbuf2
    | ⟨2, _⟩ => io.fbuf3
    | ⟨3, _⟩ => io.mbuf
    | ⟨4, _⟩ => io.mbuf
    | ⟨5, _⟩ => io.idbuf
    | ⟨6, _⟩ => io.cntbuf
    | ⟨7, _⟩ => io.inp
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun _ => io.B
  sbuf := fun _ => io.out
  iwin := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => io.fwin1 p₀ p₁
    | ⟨1, _⟩ => fun _ => io.fwin2 p₀ p₁
    | ⟨2, _⟩ => fun _ => io.fwin3 p₀ p₁
    | ⟨3, _⟩ => fun _ => io.mwin1 p₀ p₁
    | ⟨4, _⟩ => fun _ => io.mwin2 p₀ p₁
    | ⟨5, _⟩ => fun j => io.readi p₀ p₁
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨6, _⟩ => fun j => io.readc p₀ p₁
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨7, _⟩ => fun j => io.read p₀ p₁
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (fun j' => vals (⟨5, by decide⟩ : Fin 8) j') j
  imask := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => True
    | ⟨2, _⟩ => fun _ => True
    | ⟨3, _⟩ => fun _ => True
    | ⟨4, _⟩ => fun _ => True
    | ⟨5, _⟩ => fun j => io.mask p₀ p₁
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨6, _⟩ => fun j => io.mask p₀ p₁
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨7, _⟩ => fun j => io.mask p₀ p₁
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1)) j
  owin := fun _ vals p₀ p₁ _ j => io.write p₀ p₁
    (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
    (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
    (fun j' => vals (⟨5, by decide⟩ : Fin 8) j') j
  omask := fun _ vals p₀ p₁ _ j =>
    io.writeMask p₀ p₁
      (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
      (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
      (fun j' => vals (⟨5, by decide⟩ : Fin 8) j') j
    ∧ io.WriteInj p₀ p₁
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (fun j' => vals (⟨5, by decide⟩ : Fin 8) j')
  swin := fun _ vals p₀ p₁ _ j => io.write p₀ p₁
    (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
    (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
    (fun j' => vals (⟨5, by decide⟩ : Fin 8) j') j
  smask := fun _ vals p₀ p₁ _ j =>
    io.writeMask p₀ p₁
      (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
      (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
      (fun j' => vals (⟨5, by decide⟩ : Fin 8) j') j

/-- Assembly lemma: obligations in the skin's named vocabulary — all
scalar slots enter `hts`/`hrun` as pinned named values (no
lane-constancy plumbing; slots are 1-lane channels), and `hrun`'s value
leg receives the per-context `WriteInj` antecedent (the consumer threads
its `hUniq` side condition there); `hrun`'s frame and the write bound
are at the ungated `writeMask` lanes. -/
theorem Implements.intro (io : MetaScatterMasked2DKernelIO₁)
    {f : Nat → Nat → ℝ → ℝ → ℝ → Nat → Nat → (Fin io.B → Nat) →
      (Fin io.B → Nat) → (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m₁ m₂ : Nat)
        (ids : Fin io.B → Nat),
      s.readMemValue .nat io.mbuf (io.mwin1 (s.pids 0) (s.pids 1)) = m₁ →
      s.readMemValue .nat io.mbuf (io.mwin2 (s.pids 0) (s.pids 1)) = m₂ →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) m₁ m₂ j →
        s.readMemValue .nat io.idbuf
          (io.readi (s.pids 0) (s.pids 1) m₁ m₂ j) = ids j) →
      io.fwin1 (s.pids 0) (s.pids 1) < bounds io.fbuf1 →
      io.fwin2 (s.pids 0) (s.pids 1) < bounds io.fbuf2 →
      io.fwin3 (s.pids 0) (s.pids 1) < bounds io.fbuf3 →
      io.mwin1 (s.pids 0) (s.pids 1) < bounds io.mbuf →
      io.mwin2 (s.pids 0) (s.pids 1) < bounds io.mbuf →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) m₁ m₂ j →
        io.readi (s.pids 0) (s.pids 1) m₁ m₂ j < bounds io.idbuf) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) m₁ m₂ j →
        io.readc (s.pids 0) (s.pids 1) m₁ m₂ j < bounds io.cntbuf) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) m₁ m₂ j →
        io.read (s.pids 0) (s.pids 1) m₁ m₂ ids j < bounds io.inp) →
      (∀ j : Fin io.B, io.writeMask (s.pids 0) (s.pids 1) m₁ m₂ ids j →
        io.write (s.pids 0) (s.pids 1) m₁ m₂ ids j < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (g₁ g₂ g₃ : ℝ) (m₁ m₂ : Nat)
        (ids cnts : Fin io.B → Nat) (xs : Fin io.B → ℝ),
      s₀.readMem io.fbuf1 (io.fwin1 (s₀.pids 0) (s₀.pids 1)) = g₁ →
      s₀.readMem io.fbuf2 (io.fwin2 (s₀.pids 0) (s₀.pids 1)) = g₂ →
      s₀.readMem io.fbuf3 (io.fwin3 (s₀.pids 0) (s₀.pids 1)) = g₃ →
      s₀.readMemValue .nat io.mbuf (io.mwin1 (s₀.pids 0) (s₀.pids 1)) = m₁ →
      s₀.readMemValue .nat io.mbuf (io.mwin2 (s₀.pids 0) (s₀.pids 1)) = m₂ →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
        s₀.readMemValue .nat io.idbuf
          (io.readi (s₀.pids 0) (s₀.pids 1) m₁ m₂ j) = ids j) →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
        s₀.readMemValue .nat io.cntbuf
          (io.readc (s₀.pids 0) (s₀.pids 1) m₁ m₂ j) = cnts j) →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
        s₀.readMem io.inp
          (io.read (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids j) = xs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (io.WriteInj (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids →
            ∀ j : Fin io.B,
              io.writeMask (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids j →
              s1.readMem io.out
                  (io.write (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids j)
                = f (s₀.pids 0) (s₀.pids 1) g₁ g₂ g₃ m₁ m₂ ids cnts xs j)
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.B,
                io.writeMask (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁
          (vals (⟨0, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨1, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨2, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
          (fun j' => vals (⟨5, by decide⟩ : Fin 8) j')
          (fun j' => vals (⟨6, by decide⟩ : Fin 8) j')
          (fun j' => vals (⟨7, by decide⟩ : Fin 8) j') j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib _hob hsb
      exact hts bounds s
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (fun j' => vals (⟨5, by decide⟩ : Fin 8) j')
        (hpins (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hpins (⟨5, by decide⟩ : Fin 8) j hj)
        (hib (⟨0, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨1, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨2, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hib (⟨5, by decide⟩ : Fin 8) j hj)
        (fun j hj => hib (⟨6, by decide⟩ : Fin 8) j hj)
        (fun j hj => hib (⟨7, by decide⟩ : Fin 8) j hj)
        (fun j hj => hsb (⟨0, by decide⟩ : Fin 1) j hj)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀
        (vals (⟨0, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨2, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (fun j => vals (⟨5, by decide⟩ : Fin 8) j)
        (fun j => vals (⟨6, by decide⟩ : Fin 8) j)
        (fun j => vals (⟨7, by decide⟩ : Fin 8) j)
        (hpins (⟨0, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨1, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨2, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hpins (⟨5, by decide⟩ : Fin 8) j hj)
        (fun j hj => hpins (⟨6, by decide⟩ : Fin 8) j hj)
        (fun j hj => hpins (⟨7, by decide⟩ : Fin 8) j hj)
      refine ⟨s1, hexec, fun _o j hj => hval hj.2 j hj.1, ?_⟩
      intro r o' _hoc hsc
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun j hj => ?_
        rcases hsc (⟨0, by decide⟩ : Fin 1) j hj with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ g₁ g₂ g₃ m₁ m₂ ids cnts xs s₀ hpid₀ hpid₁
    hu hbf1 hbf2 hbf3 hbm1 hbm2 hbi hbc hbr hbw hg1 hg2 hg3 hm1 hm2 hid
    hcnt hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ => g₁
        | ⟨1, _⟩ => fun _ => g₂
        | ⟨2, _⟩ => fun _ => g₃
        | ⟨3, _⟩ => fun _ => m₁
        | ⟨4, _⟩ => fun _ => m₂
        | ⟨5, _⟩ => ids
        | ⟨6, _⟩ => cnts
        | ⟨7, _⟩ => xs)
      s₀ hpid₀ hpid₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hbf1
        | ⟨1, _⟩ => fun _ _ => hbf2
        | ⟨2, _⟩ => fun _ _ => hbf3
        | ⟨3, _⟩ => fun _ _ => hbm1
        | ⟨4, _⟩ => fun _ _ => hbm2
        | ⟨5, _⟩ => fun j hj => hbi j hj
        | ⟨6, _⟩ => fun j hj => hbc j hj
        | ⟨7, _⟩ => fun j hj => hbr j hj)
      (fun _o j hj => hbw j hj.1)
      (fun _t j hj => hbw j hj)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hg1
        | ⟨1, _⟩ => fun _ _ => hg2
        | ⟨2, _⟩ => fun _ _ => hg3
        | ⟨3, _⟩ => fun _ _ => hm1
        | ⟨4, _⟩ => fun _ _ => hm2
        | ⟨5, _⟩ => fun j hj => hid j hj
        | ⟨6, _⟩ => fun j hj => hcnt j hj
        | ⟨7, _⟩ => fun j hj => hx j hj)
  refine ⟨s', hexec,
    fun hinj j hj => hval (⟨0, by decide⟩ : Fin 1) j ⟨hj, hinj⟩, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | hn
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j hj => hn j hj.1, fun _t j hj => hn j hj⟩

end MetaScatterMasked2DKernelIO₁

/-! ### The chained-metadata genre: `ChainMeta*` skins

Slot windows that depend on **earlier slot values**: the second scalar
slot's cell address eats the first slot's loaded value (the paged-KV
`block_table[f(context_length)]` pattern). The core supports the chain
natively — every window eats the full pinned context — so the embedding
just wires slot 2's window to slot 1's pinned value. Slot binders carry
the **raw** loaded values; arithmetic like the decode path's `− 1`
(Nat-truncated, as in the DSL ports) belongs in the windows/spec, not in
the pin. -/

/-- IO signature of the **paged-KV-cache copy shape**: two *chained*
per-program `.nat` scalar slots — `mbuf1` (cell `mwin1`, value `m₁`, the
raw context/sequence length) and `mbuf2` (cell `mwin2 pid₀ pid₁ m₁`,
value `m₂`, the block id gathered from the block table at an
`m₁`-dependent cell) — and two masked float data channels copied to
slot-dependent cache windows: `in1 → out1` (K → KCache) and
`in2 → out2` (V → VCache), all windows/masks eating both loaded slots.
The subscript is pure data arity (2 inputs ↦ 2 outputs); the chained
slots are capability fields. Intended consumers: the TritonBench-G
paged-KV decode copies `kv_cache_copy` (`_copy_to_kvcache_seqlen1_kernel`,
both cache layouts) and its K-only sibling `kcache_copy_triton`
(`_copy_to_kcache_seqlen_n_kernel`, `n_tokens = 1` path) — the K-only
shape instantiates the second channel *off* (`mask2`/`writeMask2 :=
False`, `in2 := in1`, `out2 := out1`); note the sibling's
`split_x_idx = program_id(2)` axis is outside the family's two-pid
window convention and must be handled at the port.

**Injectivity design.** As in the `Scatter*` skins, each output's
readback leg is guarded by a *per-pinned-context* `WriteInj` antecedent
(the consumers' `hOutInj` no-aliasing side conditions on the
block-id-dependent cache windows) rather than by an `Implements.intro`
hypothesis — a `∀`-quantified hypothesis would demand no-aliasing for
arbitrary block-table contents. The frames and trace-safety write bounds
stay at the ungated `writeMask` lanes via core scratch shadows. -/
structure ChainMetaMasked2DKernelIO₂ₓ₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- First `.nat` scalar slot's buffer (the context-length array). -/
  mbuf1 : RegionName
  /-- Second `.nat` scalar slot's buffer (the block table); its cell
  address depends on the first slot's loaded value. -/
  mbuf2 : RegionName
  /-- First input buffer (K). -/
  in1 : RegionName
  /-- Second input buffer (V). -/
  in2 : RegionName
  /-- First output buffer (KCache). -/
  out1 : RegionName
  /-- Second output buffer (VCache). -/
  out2 : RegionName
  /-- Tile length: each program instance owns `B`-lane windows. -/
  B : Nat
  /-- First slot's cell address for program `(pid₀, pid₁)`. -/
  mwin1 : Nat → Nat → Nat
  /-- Second slot's cell address at `(pid₀, pid₁, m₁)` — **chained**: it
  eats the first slot's loaded value. -/
  mwin2 : Nat → Nat → Nat → Nat
  /-- Lane `j`'s `in1` read address at `(pid₀, pid₁, m₁, m₂)` — the
  loaded slots are ordinary named arguments. -/
  read1 : Nat → Nat → Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `in2` read address. -/
  read2 : Nat → Nat → Nat → Nat → Fin B → Nat
  /-- `in1`'s read-active lanes, given the loaded slots. -/
  mask1 : Nat → Nat → Nat → Nat → Fin B → Prop
  /-- `in2`'s read-active lanes. -/
  mask2 : Nat → Nat → Nat → Nat → Fin B → Prop
  /-- Lane `j`'s `out1` write address, given the loaded slots (the
  block-id-dependent cache cell). -/
  write1 : Nat → Nat → Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `out2` write address. -/
  write2 : Nat → Nat → Nat → Nat → Fin B → Nat
  /-- `out1`'s write-active lanes; defaults to `mask1`. -/
  writeMask1 : Nat → Nat → Nat → Nat → Fin B → Prop := mask1
  /-- `out2`'s write-active lanes; defaults to `mask2`. -/
  writeMask2 : Nat → Nat → Nat → Nat → Fin B → Prop := mask2

namespace ChainMetaMasked2DKernelIO₂ₓ₂

/-- No two `out1`-write-active lanes share a destination cell — the
per-pinned-context injectivity that the slot-dependent cache readback
needs (the consumers' K-cache `hOutInj`). -/
def WriteInj₁ (io : ChainMetaMasked2DKernelIO₂ₓ₂) (p₀ p₁ m₁ m₂ : Nat) :
    Prop :=
  ∀ j k : Fin io.B, io.writeMask1 p₀ p₁ m₁ m₂ j →
    io.writeMask1 p₀ p₁ m₁ m₂ k →
    io.write1 p₀ p₁ m₁ m₂ j = io.write1 p₀ p₁ m₁ m₂ k → j = k

/-- No two `out2`-write-active lanes share a destination cell (the
consumers' V-cache `hOutInj`). -/
def WriteInj₂ (io : ChainMetaMasked2DKernelIO₂ₓ₂) (p₀ p₁ m₁ m₂ : Nat) :
    Prop :=
  ∀ j k : Fin io.B, io.writeMask2 p₀ p₁ m₁ m₂ j →
    io.writeMask2 p₀ p₁ m₁ m₂ k →
    io.write2 p₀ p₁ m₁ m₂ j = io.write2 p₀ p₁ m₁ m₂ k → j = k

/-- `io.Implements f` — the chained-metadata masked Hoare triple. The
slot values `m₁`/`m₂` are universally quantified and pinned to the slot
cells inside the memory precondition — `m₂`'s pin reads `mbuf2` at the
`m₁`-dependent cell `mwin2 pid₀ pid₁ m₁`, so the chain is stated on the
*loaded* first slot. The spec `f` returns the pair of the two outputs'
value functions (`.1` for `out1`, `.2` for `out2`); each readback leg is
guarded by its `WriteInj` (see the struct docstring); the frame excludes
the two ungated write windows. -/
def Implements (io : ChainMetaMasked2DKernelIO₂ₓ₂)
    (f : Nat → Nat → Nat → Nat → (Fin io.B → ℝ) → (Fin io.B → ℝ) →
      (Fin io.B → ℝ) × (Fin io.B → ℝ)) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.mbuf1, io.mbuf2, io.in1, io.in2, io.out1, io.out2] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (m₁ m₂ : Nat) (xs ys : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    io.mwin1 pid₀ pid₁ < A.extent io.mbuf1 →
    io.mwin2 pid₀ pid₁ m₁ < A.extent io.mbuf2 →
    (∀ j : Fin io.B, io.mask1 pid₀ pid₁ m₁ m₂ j →
      io.read1 pid₀ pid₁ m₁ m₂ j < A.extent io.in1) →
    (∀ j : Fin io.B, io.mask2 pid₀ pid₁ m₁ m₂ j →
      io.read2 pid₀ pid₁ m₁ m₂ j < A.extent io.in2) →
    (∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ m₁ m₂ j →
      io.write1 pid₀ pid₁ m₁ m₂ j < A.extent io.out1) →
    (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ m₁ m₂ j →
      io.write2 pid₀ pid₁ m₁ m₂ j < A.extent io.out2) →
    s₀.readMemValue .nat io.mbuf1 (io.mwin1 pid₀ pid₁) = m₁ →
    s₀.readMemValue .nat io.mbuf2 (io.mwin2 pid₀ pid₁ m₁) = m₂ →
    (∀ j : Fin io.B, io.mask1 pid₀ pid₁ m₁ m₂ j →
      s₀.readMem io.in1 (io.read1 pid₀ pid₁ m₁ m₂ j) = xs j) →
    (∀ j : Fin io.B, io.mask2 pid₀ pid₁ m₁ m₂ j →
      s₀.readMem io.in2 (io.read2 pid₀ pid₁ m₁ m₂ j) = ys j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (io.WriteInj₁ pid₀ pid₁ m₁ m₂ →
          ∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ m₁ m₂ j →
            s'.readMem A.flat (A.addr io.out1 (io.write1 pid₀ pid₁ m₁ m₂ j))
              = (f pid₀ pid₁ m₁ m₂ xs ys).1 j)
      ∧ (io.WriteInj₂ pid₀ pid₁ m₁ m₂ →
          ∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ m₁ m₂ j →
            s'.readMem A.flat (A.addr io.out2 (io.write2 pid₀ pid₁ m₁ m₂ j))
              = (f pid₀ pid₁ m₁ m₂ xs ys).2 j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ m₁ m₂ j →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ m₁ m₂ j)) ∧
             (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ m₁ m₂ j →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ m₁ m₂ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => ChainMetaMasked2DKernelIO₂ₓ₂.Implements

/-- Embed into the unified core: channels 0/1 are the 1-lane `.nat`
slots (always read) — channel 1's window reads channel 0's pinned value,
the chain the core supports natively — and channels 2/3 the float data
tiles whose windows/masks read both slots. Each output's mask is its
write gate conjoined with its `WriteInj`; two scratch channels shadow
the same cells with the ungated gates. -/
private def toU (io : ChainMetaMasked2DKernelIO₂ₓ₂) : UKernelIO where
  kernel := io.kernel
  nIn := 4
  nOut := 2
  nScr := 2
  bufs := [io.mbuf1, io.mbuf2, io.in1, io.in2, io.out1, io.out2]
  ity := fun i => match i with
    | ⟨0, _⟩ => .nat
    | ⟨1, _⟩ => .nat
    | ⟨2, _⟩ => .float
    | ⟨3, _⟩ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => 1
    | ⟨2, _⟩ => io.B
    | ⟨3, _⟩ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.mbuf1
    | ⟨1, _⟩ => io.mbuf2
    | ⟨2, _⟩ => io.in1
    | ⟨3, _⟩ => io.in2
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.out1
    | ⟨_+1, _⟩ => io.out2
  obuf_mem := fun o => by fin_cases o <;> simp
  sarity := fun _ => io.B
  sbuf := fun t => match t with
    | ⟨0, _⟩ => io.out1
    | ⟨_+1, _⟩ => io.out2
  iwin := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => io.mwin1 p₀ p₁
    | ⟨1, _⟩ => fun _ => io.mwin2 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
    | ⟨2, _⟩ => fun j => io.read1 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨3, _⟩ => fun j => io.read2 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1)) j
  imask := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => True
    | ⟨2, _⟩ => fun j => io.mask1 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨3, _⟩ => fun j => io.mask2 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1)) j
  owin := fun o vals p₀ p₁ _ => match o with
    | ⟨0, _⟩ => fun j => io.write1 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨_+1, _⟩ => fun j => io.write2 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1)) j
  omask := fun o vals p₀ p₁ _ => match o with
    | ⟨0, _⟩ => fun j =>
        io.writeMask1 p₀ p₁
          (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1)) j
        ∧ io.WriteInj₁ p₀ p₁
            (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
            (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
    | ⟨_+1, _⟩ => fun j =>
        io.writeMask2 p₀ p₁
          (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1)) j
        ∧ io.WriteInj₂ p₀ p₁
            (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
            (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
  swin := fun t vals p₀ p₁ _ => match t with
    | ⟨0, _⟩ => fun j => io.write1 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨_+1, _⟩ => fun j => io.write2 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1)) j
  smask := fun t vals p₀ p₁ _ => match t with
    | ⟨0, _⟩ => fun j => io.writeMask1 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨_+1, _⟩ => fun j => io.writeMask2 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1)) j

/-- Assembly lemma: obligations in the skin's named vocabulary — the
chained slots enter `hts`/`hrun` as pinned named scalars, `m₂`'s pin at
the `m₁`-dependent cell (no lane-constancy plumbing; slots are 1-lane
channels); each value leg receives its per-context `WriteInj` antecedent
(the consumers thread their `hOutInj` side conditions there); `hrun`'s
frame takes one ungated exclusion condition per output region. -/
theorem Implements.intro (io : ChainMetaMasked2DKernelIO₂ₓ₂)
    {f : Nat → Nat → Nat → Nat → (Fin io.B → ℝ) → (Fin io.B → ℝ) →
      (Fin io.B → ℝ) × (Fin io.B → ℝ)}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m₁ m₂ : Nat),
      s.readMemValue .nat io.mbuf1 (io.mwin1 (s.pids 0) (s.pids 1)) = m₁ →
      s.readMemValue .nat io.mbuf2 (io.mwin2 (s.pids 0) (s.pids 1) m₁)
        = m₂ →
      io.mwin1 (s.pids 0) (s.pids 1) < bounds io.mbuf1 →
      io.mwin2 (s.pids 0) (s.pids 1) m₁ < bounds io.mbuf2 →
      (∀ j : Fin io.B, io.mask1 (s.pids 0) (s.pids 1) m₁ m₂ j →
        io.read1 (s.pids 0) (s.pids 1) m₁ m₂ j < bounds io.in1) →
      (∀ j : Fin io.B, io.mask2 (s.pids 0) (s.pids 1) m₁ m₂ j →
        io.read2 (s.pids 0) (s.pids 1) m₁ m₂ j < bounds io.in2) →
      (∀ j : Fin io.B, io.writeMask1 (s.pids 0) (s.pids 1) m₁ m₂ j →
        io.write1 (s.pids 0) (s.pids 1) m₁ m₂ j < bounds io.out1) →
      (∀ j : Fin io.B, io.writeMask2 (s.pids 0) (s.pids 1) m₁ m₂ j →
        io.write2 (s.pids 0) (s.pids 1) m₁ m₂ j < bounds io.out2) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (m₁ m₂ : Nat) (xs ys : Fin io.B → ℝ),
      s₀.readMemValue .nat io.mbuf1 (io.mwin1 (s₀.pids 0) (s₀.pids 1))
        = m₁ →
      s₀.readMemValue .nat io.mbuf2 (io.mwin2 (s₀.pids 0) (s₀.pids 1) m₁)
        = m₂ →
      (∀ j : Fin io.B, io.mask1 (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) m₁ m₂ j)
          = xs j) →
      (∀ j : Fin io.B, io.mask2 (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) m₁ m₂ j)
          = ys j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (io.WriteInj₁ (s₀.pids 0) (s₀.pids 1) m₁ m₂ →
            ∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
              s1.readMem io.out1 (io.write1 (s₀.pids 0) (s₀.pids 1) m₁ m₂ j)
                = (f (s₀.pids 0) (s₀.pids 1) m₁ m₂ xs ys).1 j)
        ∧ (io.WriteInj₂ (s₀.pids 0) (s₀.pids 1) m₁ m₂ →
            ∀ j : Fin io.B, io.writeMask2 (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
              s1.readMem io.out2 (io.write2 (s₀.pids 0) (s₀.pids 1) m₁ m₂ j)
                = (f (s₀.pids 0) (s₀.pids 1) m₁ m₂ xs ys).2 j)
        ∧ (∀ r o,
            (r ≠ io.out1 ∨
              ∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
                o ≠ io.write1 (s₀.pids 0) (s₀.pids 1) m₁ m₂ j) →
            (r ≠ io.out2 ∨
              ∀ j : Fin io.B, io.writeMask2 (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
                o ≠ io.write2 (s₀.pids 0) (s₀.pids 1) m₁ m₂ j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals o => match o with
        | ⟨0, _⟩ => fun j =>
            (f p₀ p₁
              (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
              (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
              (fun j' => vals (⟨2, by decide⟩ : Fin 4) j')
              (fun j' => vals (⟨3, by decide⟩ : Fin 4) j')).1 j
        | ⟨_+1, _⟩ => fun j =>
            (f p₀ p₁
              (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
              (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
              (fun j' => vals (⟨2, by decide⟩ : Fin 4) j')
              (fun j' => vals (⟨3, by decide⟩ : Fin 4) j')).2 j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib _hob hsb
      exact hts bounds s
        (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
        (hpins (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hib (⟨2, by decide⟩ : Fin 4) j hj)
        (fun j hj => hib (⟨3, by decide⟩ : Fin 4) j hj)
        (fun j hj => hsb (⟨0, by decide⟩ : Fin 2) j hj)
        (fun j hj => hsb (⟨1, by decide⟩ : Fin 2) j hj)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval1, hval2, hframe⟩ := hrun s₀
        (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
        (fun j => vals (⟨2, by decide⟩ : Fin 4) j)
        (fun j => vals (⟨3, by decide⟩ : Fin 4) j)
        (hpins (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hpins (⟨2, by decide⟩ : Fin 4) j hj)
        (fun j hj => hpins (⟨3, by decide⟩ : Fin 4) j hj)
      refine ⟨s1, hexec, fun o => match o with
        | ⟨0, _⟩ => fun j hj => hval1 hj.2 j hj.1
        | ⟨_+1, _⟩ => fun j hj => hval2 hj.2 j hj.1, ?_⟩
      intro r o' _hoc hsc
      refine hframe r o' ?_ ?_
      · by_cases hro : r = io.out1
        · subst hro
          refine Or.inr fun j hj => ?_
          rcases hsc (⟨0, by decide⟩ : Fin 2) j hj with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · by_cases hro : r = io.out2
        · subst hro
          refine Or.inr fun j hj => ?_
          rcases hsc (⟨1, by decide⟩ : Fin 2) j hj with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ m₁ m₂ xs ys s₀ hpid₀ hpid₁ hu hb1 hb2
    hbr1 hbr2 hbw1 hbw2 hm1 hm2 hx hy
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ => m₁
        | ⟨1, _⟩ => fun _ => m₂
        | ⟨2, _⟩ => xs
        | ⟨3, _⟩ => ys)
      s₀ hpid₀ hpid₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hb1
        | ⟨1, _⟩ => fun _ _ => hb2
        | ⟨2, _⟩ => fun j hj => hbr1 j hj
        | ⟨3, _⟩ => fun j hj => hbr2 j hj)
      (fun o => match o with
        | ⟨0, _⟩ => fun j hj => hbw1 j hj.1
        | ⟨_+1, _⟩ => fun j hj => hbw2 j hj.1)
      (fun t => match t with
        | ⟨0, _⟩ => fun j hj => hbw1 j hj
        | ⟨_+1, _⟩ => fun j hj => hbw2 j hj)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hm1
        | ⟨1, _⟩ => fun _ _ => hm2
        | ⟨2, _⟩ => fun j hj => hx j hj
        | ⟨3, _⟩ => fun j hj => hy j hj)
  refine ⟨s', hexec,
    fun hinj j hj => hval (⟨0, by decide⟩ : Fin 2) j ⟨hj, hinj⟩,
    fun hinj j hj => hval (⟨1, by decide⟩ : Fin 2) j ⟨hj, hinj⟩, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hn1, hn2⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun oc => match oc with
      | ⟨0, _⟩ => fun j hj => hn1 j hj.1
      | ⟨_+1, _⟩ => fun j hj => hn2 j hj.1,
      fun t => match t with
      | ⟨0, _⟩ => fun j hj => hn1 j hj
      | ⟨_+1, _⟩ => fun j hj => hn2 j hj⟩

end ChainMetaMasked2DKernelIO₂ₓ₂




/-! ### Three-output metadata skin: `MetaMasked2DKernelIO₂ₓ₃` -/

/-- IO signature of the **fused cross-entropy forward metadata shape** —
the `MetaGatherMasked2DKernelIO₂ₓ₂` cross-entropy genre with a **third**
per-program single-cell output: one per-program `.int` scalar slot (the
label, `mbufL`, loaded at the pid-affine cell `mwinL`), one masked float
data row (`inp`, `B` lanes), one label-dependent single-cell **gather**
read (also from `inp`; window `gwin` and read gate `gmask` both eat the
loaded label), and three per-program single-cell float outputs
(`out1`/`out2`/`out3`, each with its own cell address and write gate).
The subscript is pure data-input × output arity (row + gather cell ↦
three scalar cells); the slot is capability (`Meta`), not arity.
Intended consumers: the TritonBench-G fused cross-entropy forwards
(`cross_entropy2` / `cross_entropy_ops` — per-`(row_idx, col_block_idx)`
LSE + loss + z-loss cells with the guarded `logits[label]` gather). A
constexpr-gated store (the genre's `SPLIT`-guarded z-loss) puts the Lean
`Bool` parameter into its write gate as a conjunct — the
`swiglu_backward` precedent. Following the family precedent there is no
`scratch` field until a consumer needs one. -/
structure MetaMasked2DKernelIO₂ₓ₃ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- Label slot's buffer (one `.int` cell per program). -/
  mbufL : RegionName
  /-- Data input buffer — both the masked row and the gather cell read it. -/
  inp : RegionName
  /-- First output buffer. -/
  out1 : RegionName
  /-- Second output buffer. -/
  out2 : RegionName
  /-- Third output buffer. -/
  out3 : RegionName
  /-- Tile length of the masked data row. -/
  B : Nat
  /-- Label slot's cell address for program `(pid₀, pid₁)`. -/
  mwinL : Nat → Nat → Nat
  /-- Data-row read window at `(pid₀, pid₁, lab, j)` — the loaded label is
  an ordinary named argument. -/
  read : Nat → Nat → Int → Fin B → Nat
  /-- Data-row read-active lanes, given the loaded label. -/
  mask : Nat → Nat → Int → Fin B → Prop
  /-- Gather cell's read address (label-dependent). -/
  gwin : Nat → Nat → Int → Nat
  /-- Gather read gate: the label-dependent condition under which the
  kernel actually loads the gather cell. -/
  gmask : Nat → Nat → Int → Prop
  /-- `out1`'s cell address for program `(pid₀, pid₁)`. -/
  write1 : Nat → Nat → Int → Nat
  /-- `out2`'s cell address. -/
  write2 : Nat → Nat → Int → Nat
  /-- `out3`'s cell address. -/
  write3 : Nat → Nat → Int → Nat
  /-- `out1`'s write gate; defaults to always-on (the genre's LSE/loss
  stores are unconditional). -/
  writeMask1 : Nat → Nat → Int → Prop := fun _ _ _ => True
  /-- `out2`'s write gate; defaults to always-on. -/
  writeMask2 : Nat → Nat → Int → Prop := fun _ _ _ => True
  /-- `out3`'s write gate; defaults to always-on. A constexpr-gated store
  (the genre's `¬SPLIT` z-loss gate) goes here as a conjunct. -/
  writeMask3 : Nat → Nat → Int → Prop := fun _ _ _ => True

namespace MetaMasked2DKernelIO₂ₓ₃

/-- `io.Implements f` — three-output sibling of
`MetaGatherMasked2DKernelIO₂ₓ₂.Implements`. The label `lab` is universally
quantified and pinned to the slot cell inside the memory precondition;
the data row `xs` is pinned on the read-active lanes and the gather cell
`g` under its gate. The spec `f` returns the triple of the three output
cells' values; `×` is right-associative, so the components read `.1`
(`out1`), `.2.1` (`out2`), `.2.2` (`out3`). Frame: every cell outside
the three gated output cells is untouched. -/
def Implements (io : MetaMasked2DKernelIO₂ₓ₃)
    (f : Nat → Nat → Int → (Fin io.B → ℝ) → ℝ → ℝ × ℝ × ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.mbufL, io.inp, io.out1, io.out2, io.out3] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (lab : Int) (xs : Fin io.B → ℝ) (g : ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    io.mwinL pid₀ pid₁ < A.extent io.mbufL →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ lab j →
      io.read pid₀ pid₁ lab j < A.extent io.inp) →
    (io.gmask pid₀ pid₁ lab → io.gwin pid₀ pid₁ lab < A.extent io.inp) →
    (io.writeMask1 pid₀ pid₁ lab →
      io.write1 pid₀ pid₁ lab < A.extent io.out1) →
    (io.writeMask2 pid₀ pid₁ lab →
      io.write2 pid₀ pid₁ lab < A.extent io.out2) →
    (io.writeMask3 pid₀ pid₁ lab →
      io.write3 pid₀ pid₁ lab < A.extent io.out3) →
    s₀.readMemValue .int io.mbufL (io.mwinL pid₀ pid₁) = lab →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ lab j →
      s₀.readMem io.inp (io.read pid₀ pid₁ lab j) = xs j) →
    (io.gmask pid₀ pid₁ lab →
      s₀.readMem io.inp (io.gwin pid₀ pid₁ lab) = g) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (io.writeMask1 pid₀ pid₁ lab →
          s'.readMem A.flat (A.addr io.out1 (io.write1 pid₀ pid₁ lab))
            = (f pid₀ pid₁ lab xs g).1)
      ∧ (io.writeMask2 pid₀ pid₁ lab →
          s'.readMem A.flat (A.addr io.out2 (io.write2 pid₀ pid₁ lab))
            = (f pid₀ pid₁ lab xs g).2.1)
      ∧ (io.writeMask3 pid₀ pid₁ lab →
          s'.readMem A.flat (A.addr io.out3 (io.write3 pid₀ pid₁ lab))
            = (f pid₀ pid₁ lab xs g).2.2)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((io.writeMask1 pid₀ pid₁ lab →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ lab)) ∧
             (io.writeMask2 pid₀ pid₁ lab →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ lab)) ∧
             (io.writeMask3 pid₀ pid₁ lab →
                o' ≠ A.addr io.out3 (io.write3 pid₀ pid₁ lab)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => MetaMasked2DKernelIO₂ₓ₃.Implements

/-- Embed into the unified core: channel 0 is the 1-lane `.int` label slot
(always read), channel 1 the `B`-lane float data row and channel 2 the
1-lane float gather cell, both with windows/masks reading the slot's
pinned value; the three outputs are 1-lane gated cells. -/
private def toU (io : MetaMasked2DKernelIO₂ₓ₃) : UKernelIO where
  kernel := io.kernel
  nIn := 3
  nOut := 3
  nScr := 0
  bufs := [io.mbufL, io.inp, io.out1, io.out2, io.out3]
  ity := fun i => match i with
    | ⟨0, _⟩ => .int
    | ⟨1, _⟩ => .float
    | ⟨2, _⟩ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => io.B
    | ⟨2, _⟩ => 1
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.mbufL
    | ⟨1, _⟩ => io.inp
    | ⟨2, _⟩ => io.inp
  oty := fun _ => .float
  oarity := fun _ => 1
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.out1
    | ⟨1, _⟩ => io.out2
    | _ => io.out3
  obuf_mem := fun o => by fin_cases o <;> simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => io.mwinL p₀ p₁
    | ⟨1, _⟩ => fun j => io.read p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨2, _⟩ => fun _ => io.gwin p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
  imask := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun j => io.mask p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨2, _⟩ => fun _ => io.gmask p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
  owin := fun o vals p₀ p₁ _ => match o with
    | ⟨0, _⟩ => fun _ => io.write1 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
    | ⟨1, _⟩ => fun _ => io.write2 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
    | _ => fun _ => io.write3 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
  omask := fun o vals p₀ p₁ _ => match o with
    | ⟨0, _⟩ => fun _ => io.writeMask1 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
    | ⟨1, _⟩ => fun _ => io.writeMask2 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
    | _ => fun _ => io.writeMask3 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- Assembly lemma: obligations in the skin's named vocabulary — the label
enters `hts`/`hrun` as one pinned named `Int` scalar (no lane-constancy
plumbing: the slot and the gather cell are 1-lane channels), the gather
pin and bound are gated by `gmask`, and `hrun`'s frame takes one exclusion
condition per output cell. -/
theorem Implements.intro (io : MetaMasked2DKernelIO₂ₓ₃)
    {f : Nat → Nat → Int → (Fin io.B → ℝ) → ℝ → ℝ × ℝ × ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (lab : Int),
      s.readMemValue .int io.mbufL (io.mwinL (s.pids 0) (s.pids 1)) = lab →
      io.mwinL (s.pids 0) (s.pids 1) < bounds io.mbufL →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) lab j →
        io.read (s.pids 0) (s.pids 1) lab j < bounds io.inp) →
      (io.gmask (s.pids 0) (s.pids 1) lab →
        io.gwin (s.pids 0) (s.pids 1) lab < bounds io.inp) →
      (io.writeMask1 (s.pids 0) (s.pids 1) lab →
        io.write1 (s.pids 0) (s.pids 1) lab < bounds io.out1) →
      (io.writeMask2 (s.pids 0) (s.pids 1) lab →
        io.write2 (s.pids 0) (s.pids 1) lab < bounds io.out2) →
      (io.writeMask3 (s.pids 0) (s.pids 1) lab →
        io.write3 (s.pids 0) (s.pids 1) lab < bounds io.out3) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (lab : Int) (xs : Fin io.B → ℝ) (g : ℝ),
      s₀.readMemValue .int io.mbufL (io.mwinL (s₀.pids 0) (s₀.pids 1)) = lab →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) lab j →
        s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) lab j) = xs j) →
      (io.gmask (s₀.pids 0) (s₀.pids 1) lab →
        s₀.readMem io.inp (io.gwin (s₀.pids 0) (s₀.pids 1) lab) = g) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (io.writeMask1 (s₀.pids 0) (s₀.pids 1) lab →
            s1.readMem io.out1 (io.write1 (s₀.pids 0) (s₀.pids 1) lab)
              = (f (s₀.pids 0) (s₀.pids 1) lab xs g).1)
        ∧ (io.writeMask2 (s₀.pids 0) (s₀.pids 1) lab →
            s1.readMem io.out2 (io.write2 (s₀.pids 0) (s₀.pids 1) lab)
              = (f (s₀.pids 0) (s₀.pids 1) lab xs g).2.1)
        ∧ (io.writeMask3 (s₀.pids 0) (s₀.pids 1) lab →
            s1.readMem io.out3 (io.write3 (s₀.pids 0) (s₀.pids 1) lab)
              = (f (s₀.pids 0) (s₀.pids 1) lab xs g).2.2)
        ∧ (∀ r o,
            (r ≠ io.out1 ∨ (io.writeMask1 (s₀.pids 0) (s₀.pids 1) lab →
              o ≠ io.write1 (s₀.pids 0) (s₀.pids 1) lab)) →
            (r ≠ io.out2 ∨ (io.writeMask2 (s₀.pids 0) (s₀.pids 1) lab →
              o ≠ io.write2 (s₀.pids 0) (s₀.pids 1) lab)) →
            (r ≠ io.out3 ∨ (io.writeMask3 (s₀.pids 0) (s₀.pids 1) lab →
              o ≠ io.write3 (s₀.pids 0) (s₀.pids 1) lab)) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals o => match o with
        | ⟨0, _⟩ => fun _ =>
            (f p₀ p₁ (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
              (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
              (vals (⟨2, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))).1
        | ⟨1, _⟩ => fun _ =>
            (f p₀ p₁ (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
              (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
              (vals (⟨2, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))).2.1
        | ⟨_+2, _⟩ => fun _ =>
            (f p₀ p₁ (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
              (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
              (vals (⟨2, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))).2.2) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib hob _hsb
      exact hts bounds s
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (hpins (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hib (⟨1, by decide⟩ : Fin 3) j hj)
        (fun hgm => hib (⟨2, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) hgm)
        (fun h1 => hob (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) h1)
        (fun h2 => hob (⟨1, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) h2)
        (fun h3 => hob (⟨2, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) h3)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval1, hval2, hval3, hframe⟩ := hrun s₀
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (fun j => vals (⟨1, by decide⟩ : Fin 3) j)
        (vals (⟨2, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (hpins (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hpins (⟨1, by decide⟩ : Fin 3) j hj)
        (fun hgm => hpins (⟨2, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) hgm)
      refine ⟨s1, hexec, fun o => match o with
        | ⟨0, _⟩ => fun _ h1 => hval1 h1
        | ⟨1, _⟩ => fun _ h2 => hval2 h2
        | ⟨_+2, _⟩ => fun _ h3 => hval3 h3, ?_⟩
      intro r o' hoc _hsc
      refine hframe r o' ?_ ?_ ?_
      · by_cases hro : r = io.out1
        · subst hro
          refine Or.inr fun h1 => ?_
          rcases hoc (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) h1
            with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · by_cases hro : r = io.out2
        · subst hro
          refine Or.inr fun h2 => ?_
          rcases hoc (⟨1, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) h2
            with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · by_cases hro : r = io.out3
        · subst hro
          refine Or.inr fun h3 => ?_
          rcases hoc (⟨2, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) h3
            with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ lab xs g s₀ hpid₀ hpid₁ hu hbL hbr hbg
    hbw1 hbw2 hbw3 hmL hx hg
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ => lab
        | ⟨1, _⟩ => xs
        | ⟨2, _⟩ => fun _ => g)
      s₀ hpid₀ hpid₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hbL
        | ⟨1, _⟩ => fun j hj => hbr j hj
        | ⟨2, _⟩ => fun _ hgm => hbg hgm)
      (fun o => match o with
        | ⟨0, _⟩ => fun _ h1 => hbw1 h1
        | ⟨1, _⟩ => fun _ h2 => hbw2 h2
        | ⟨_+2, _⟩ => fun _ h3 => hbw3 h3)
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hmL
        | ⟨1, _⟩ => fun j hj => hx j hj
        | ⟨2, _⟩ => fun _ hgm => hg hgm)
  refine ⟨s', hexec,
    fun h1 => hval (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) h1,
    fun h2 => hval (⟨1, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) h2,
    fun h3 => hval (⟨2, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) h3, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hn1, hn2, hn3⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun o => match o with
      | ⟨0, _⟩ => fun _ hj => hn1 hj
      | ⟨1, _⟩ => fun _ hj => hn2 hj
      | ⟨_+2, _⟩ => fun _ hj => hn3 hj,
      fun t => t.elim0⟩

end MetaMasked2DKernelIO₂ₓ₃



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

end GroupedMasked2DKernelIO

/-- IO signature of the **quantized destination-index copy shape**: one
per-program `.nat` scalar slot (the destination row index, `mbuf1`, loaded
at the pid-affine cell `mwin1`), one masked float data tile (`inp`, `B`
lanes), and **two output channels of different tile lengths and independent
element types** — the transformed tile `out1` (`B` lanes, the data tile's
shape) and the per-row statistic column `out2` (`C` lanes, one cell per row
of that tile). The subscript is pure data-input × output arity (one tile ↦
two tiles); the slot is capability (`Meta`), not arity, and neither the two
lengths (`B`/`C`) nor the two output element types (`oty1`/`oty2`) are part
of a name — they are ordinary fields, so a `.to(tl.int8)`-typed tile beside
a `.float` scale column is *one* skin, not a dtype-suffixed family. The
outputs default to `.float` (`oty1 = oty2 = .float`, the ℝ `readMem`
readback), recovering the plain two-float shape for free; a consumer that
needs a typed output sets the field. Intended consumers: the TritonBench-G
LightLLM destination-index quantized KV copies (`quantize_kv_copy`,
`quantize_kv_transform` — `dest_index = tl.load(Dest_loc + cur_index)`, then
the `.to(tl.int8)` tile via `oty1 := .int` and its `max |x| / 127` scale
column, whose `.to(Out_scale.dtype.element_ty)` cast lowers to the ℝ
identity so `oty2 := .float`; `quantize_copy_kv`'s `.int` tile fits `oty1`
too, but its scale store is `.to(tl.float16)` — reduced-precision *float*,
i.e. the `FloatDType`/rounding axis, not a `ChanTy` — so that kernel's scale
conjunct is not typeable here, see the honest note below). Following the
family precedent there is no `scratch` field until a consumer needs one.

**Output typing vs. the rounding axis.** `oty1`/`oty2` range over `ChanTy`
(float / bool / nat / int) — the *value* readback view. A store whose
value is genuinely at reduced float precision (`.to(tl.float16)`) reads back
through `readMemValue .fp16` at a `TileCarrier .fp16`, which is **not** a
`ChanTy` carrier; that is the rounding axis and cannot be expressed by `oty`
alone. On the **exact** relation `⊨` (`Implements`), this skin therefore
types the `.int` quantized tile and any `.float`/`.nat`/`.bool`/`.int` scale
column, but not an fp16 scale column. The **rounding** relation `⊨[R]`
(`ImplementsR`) closes exactly that gap: `out1` keeps its `oty1` `ChanTy`
readback (e.g. the `.int` quantized tile), while `out2` becomes an
fp-typed rounding channel at the declared grid `out2DType : FloatDType`,
read back through `readMemAs out2DType` as `out2DType.ofReal (R.round
out2DType (f …))`. `quantize_copy_kv`'s fp16 scale store lives there. -/
structure MetaMasked2DKernelIO₁ₓ₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- Scalar slot's buffer (one `.nat` cell per program). -/
  mbuf1 : RegionName
  /-- Input buffer. -/
  inp : RegionName
  /-- First output buffer — the `B`-lane tile. -/
  out1 : RegionName
  /-- Second output buffer — the `C`-lane column. -/
  out2 : RegionName
  /-- Tile length of the data channel and of `out1`. -/
  B : Nat
  /-- Tile length of `out2` (the shorter, per-row column). -/
  C : Nat
  /-- `out1`'s channel element type. Defaults to `.float` (the ℝ `readMem`
  readback); set `oty1 := .int` for a `.to(tl.int8)`-typed quantized tile,
  whose value contract reads back through `readMemValue .int`. -/
  oty1 : ChanTy := .float
  /-- `out2`'s channel element type. Defaults to `.float`; the scale column
  of the `element_ty`-cast quantizers stays here (that cast is the ℝ
  identity). -/
  oty2 : ChanTy := .float
  /-- `out2`'s floating dtype — the quantization grid of the second output's
  boundary store, used only by the **rounding**-correctness relation `⊨[R]`
  (`ImplementsR`). Its postcondition reads `out2` back as `out2DType`-typed
  cells holding `out2DType.ofReal (R.round out2DType (f …))`, i.e. the
  reduced-precision-float scale column that `oty2 : ChanTy` cannot type (a
  `TileCarrier .fp16` is not a `ChanTy` carrier). `.real` (the default) is an
  unrounded store — the exact relation `⊨` ignores this field, so every
  existing consumer's `Implements` statement is unchanged. -/
  out2DType : FloatDType := .real
  /-- Slot's cell address for program `(pid₀, pid₁)`. -/
  mwin1 : Nat → Nat → Nat
  /-- Data read window at `(pid₀, pid₁, m1, j)` — the loaded scalar is an
  ordinary named argument. -/
  read : Nat → Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `out1` write address, given the loaded scalar. -/
  write1 : Nat → Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `out2` write address, given the loaded scalar. -/
  write2 : Nat → Nat → Nat → Fin C → Nat
  /-- Read-active lanes, given the loaded scalar. -/
  mask : Nat → Nat → Nat → Fin B → Prop
  /-- `out1`'s write-active lanes; defaults to `mask` (the genre stores the
  transformed tile exactly where it loaded it). -/
  writeMask1 : Nat → Nat → Nat → Fin B → Prop := mask
  /-- `out2`'s write-active lanes. No default: the column has its own lane
  count, so `mask` does not typecheck here. -/
  writeMask2 : Nat → Nat → Nat → Fin C → Prop

namespace MetaMasked2DKernelIO₁ₓ₂

/-- `io.Implements f` — the two-output, unequal-length sibling of
`MetaMasked2DKernelIO₁.Implements`. The slot value `m1` is universally
quantified and pinned to the slot cell inside the memory precondition, so
the windows, the masks and the spec all speak about the *loaded* scalar.
The spec `f` returns the pair of the two outputs' value functions (`.1`
for the `B`-lane `out1` at `oty1.carrier`, `.2` for the `C`-lane `out2` at
`oty2.carrier`); each output reads back through its channel's `read` view
(definitionally `readMem` when the type is `.float`). Frame: every cell
outside the union of the two write-active output windows is untouched. -/
def Implements (io : MetaMasked2DKernelIO₁ₓ₂)
    (f : Nat → Nat → Nat → (Fin io.B → ℝ) →
      (Fin io.B → io.oty1.carrier) × (Fin io.C → io.oty2.carrier)) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.mbuf1, io.inp, io.out1, io.out2] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (m1 : Nat) (xs : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    io.mwin1 pid₀ pid₁ < A.extent io.mbuf1 →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m1 j →
      io.read pid₀ pid₁ m1 j < A.extent io.inp) →
    (∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ m1 j →
      io.write1 pid₀ pid₁ m1 j < A.extent io.out1) →
    (∀ j : Fin io.C, io.writeMask2 pid₀ pid₁ m1 j →
      io.write2 pid₀ pid₁ m1 j < A.extent io.out2) →
    s₀.readMemValue .nat io.mbuf1 (io.mwin1 pid₀ pid₁) = m1 →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m1 j →
      s₀.readMem io.inp (io.read pid₀ pid₁ m1 j) = xs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ m1 j →
          io.oty1.read s' A.flat (A.addr io.out1 (io.write1 pid₀ pid₁ m1 j))
            = (f pid₀ pid₁ m1 xs).1 j)
      ∧ (∀ j : Fin io.C, io.writeMask2 pid₀ pid₁ m1 j →
          io.oty2.read s' A.flat (A.addr io.out2 (io.write2 pid₀ pid₁ m1 j))
            = (f pid₀ pid₁ m1 xs).2 j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ m1 j →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ m1 j)) ∧
             (∀ j : Fin io.C, io.writeMask2 pid₀ pid₁ m1 j →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ m1 j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => MetaMasked2DKernelIO₁ₓ₂.Implements

/-- Embed into the unified core: channel 0 is the 1-lane `.nat` slot
(always read), channel 1 the float data tile whose window/mask read the
slot's pinned value; the two outputs carry the two lane counts through the
core's per-output `oarity`. -/
private def toU (io : MetaMasked2DKernelIO₁ₓ₂) : UKernelIO where
  kernel := io.kernel
  nIn := 2
  nOut := 2
  nScr := 0
  bufs := [io.mbuf1, io.inp, io.out1, io.out2]
  ity := fun i => match i with
    | ⟨0, _⟩ => .nat
    | ⟨_+1, _⟩ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => 1
    | ⟨_+1, _⟩ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.mbuf1
    | ⟨_+1, _⟩ => io.inp
  oty := fun o => match o with
    | ⟨0, _⟩ => io.oty1
    | ⟨_+1, _⟩ => io.oty2
  oarity := fun o => match o with
    | ⟨0, _⟩ => io.B
    | ⟨_+1, _⟩ => io.C
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.out1
    | ⟨_+1, _⟩ => io.out2
  obuf_mem := fun o => by fin_cases o <;> simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => io.mwin1 p₀ p₁
    | ⟨_+1, _⟩ => fun j => io.read p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1)) j
  imask := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨_+1, _⟩ => fun j => io.mask p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1)) j
  owin := fun o vals p₀ p₁ _ => match o with
    | ⟨0, _⟩ => fun j => io.write1 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨_+1, _⟩ => fun j => io.write2 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1)) j
  omask := fun o vals p₀ p₁ _ => match o with
    | ⟨0, _⟩ => fun j => io.writeMask1 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨_+1, _⟩ => fun j => io.writeMask2 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1)) j
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- Assembly lemma: obligations in the skin's named vocabulary — the slot
enters `hts`/`hrun` as a pinned named scalar (no lane-constancy plumbing:
slots are 1-lane channels), and `hrun`'s frame takes one exclusion
condition per output region. -/
theorem Implements.intro (io : MetaMasked2DKernelIO₁ₓ₂)
    {f : Nat → Nat → Nat → (Fin io.B → ℝ) →
      (Fin io.B → io.oty1.carrier) × (Fin io.C → io.oty2.carrier)}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m1 : Nat),
      s.readMemValue .nat io.mbuf1 (io.mwin1 (s.pids 0) (s.pids 1)) = m1 →
      io.mwin1 (s.pids 0) (s.pids 1) < bounds io.mbuf1 →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) m1 j →
        io.read (s.pids 0) (s.pids 1) m1 j < bounds io.inp) →
      (∀ j : Fin io.B, io.writeMask1 (s.pids 0) (s.pids 1) m1 j →
        io.write1 (s.pids 0) (s.pids 1) m1 j < bounds io.out1) →
      (∀ j : Fin io.C, io.writeMask2 (s.pids 0) (s.pids 1) m1 j →
        io.write2 (s.pids 0) (s.pids 1) m1 j < bounds io.out2) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (m1 : Nat) (xs : Fin io.B → ℝ),
      s₀.readMemValue .nat io.mbuf1 (io.mwin1 (s₀.pids 0) (s₀.pids 1)) = m1 →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) m1 j →
        s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) m1 j) = xs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) m1 j →
            io.oty1.read s1 io.out1 (io.write1 (s₀.pids 0) (s₀.pids 1) m1 j)
              = (f (s₀.pids 0) (s₀.pids 1) m1 xs).1 j)
        ∧ (∀ j : Fin io.C, io.writeMask2 (s₀.pids 0) (s₀.pids 1) m1 j →
            io.oty2.read s1 io.out2 (io.write2 (s₀.pids 0) (s₀.pids 1) m1 j)
              = (f (s₀.pids 0) (s₀.pids 1) m1 xs).2 j)
        ∧ (∀ r o,
            (r ≠ io.out1 ∨
              ∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) m1 j →
                o ≠ io.write1 (s₀.pids 0) (s₀.pids 1) m1 j) →
            (r ≠ io.out2 ∨
              ∀ j : Fin io.C, io.writeMask2 (s₀.pids 0) (s₀.pids 1) m1 j →
                o ≠ io.write2 (s₀.pids 0) (s₀.pids 1) m1 j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals o => match o with
        | ⟨0, _⟩ => fun j =>
            (f p₀ p₁ (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1))
              (fun j' => vals (⟨1, by decide⟩ : Fin 2) j')).1 j
        | ⟨_+1, _⟩ => fun j =>
            (f p₀ p₁ (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1))
              (fun j' => vals (⟨1, by decide⟩ : Fin 2) j')).2 j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib hob _hsb
      exact hts bounds s
        (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1))
        (hpins (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hib (⟨1, by decide⟩ : Fin 2) j hj)
        (fun j hj => hob (⟨0, by decide⟩ : Fin 2) j hj)
        (fun j hj => hob (⟨1, by decide⟩ : Fin 2) j hj)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval1, hval2, hframe⟩ :=
        hrun s₀ (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1))
          (fun j => vals (⟨1, by decide⟩ : Fin 2) j)
          (hpins (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1) trivial)
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
  intro A hd hregs hcov pid₀ pid₁ m1 xs s₀ hpid₀ hpid₁ hu hb1 hbr hbw1 hbw2
    hm1 hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ => m1
        | ⟨_+1, _⟩ => xs)
      s₀ hpid₀ hpid₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hb1
        | ⟨_+1, _⟩ => fun j hj => hbr j hj)
      (fun o => match o with
        | ⟨0, _⟩ => fun j hj => hbw1 j hj
        | ⟨_+1, _⟩ => fun j hj => hbw2 j hj)
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hm1
        | ⟨_+1, _⟩ => fun j hj => hx j hj)
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

/-- Assembly lemma — **`undef`-pinned sibling** of `Implements.intro`. Same
`io.Implements f` conclusion, but `hrun` additionally receives the pin
`s₀.undef = (fun _ _ => 0)`. A quantization tail-block kernel whose masked
data load lacks an `other=` default and then reduces across the tile (the
`tl.max(abs_data, axis=1)` scale) reads `s₀.undef` at masked-off lanes; this
sibling threads the pin so those lanes are fixed to `0` inside `hrun`. -/
theorem Implements.intro_undef (io : MetaMasked2DKernelIO₁ₓ₂)
    {f : Nat → Nat → Nat → (Fin io.B → ℝ) →
      (Fin io.B → io.oty1.carrier) × (Fin io.C → io.oty2.carrier)}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m1 : Nat),
      s.readMemValue .nat io.mbuf1 (io.mwin1 (s.pids 0) (s.pids 1)) = m1 →
      io.mwin1 (s.pids 0) (s.pids 1) < bounds io.mbuf1 →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) m1 j →
        io.read (s.pids 0) (s.pids 1) m1 j < bounds io.inp) →
      (∀ j : Fin io.B, io.writeMask1 (s.pids 0) (s.pids 1) m1 j →
        io.write1 (s.pids 0) (s.pids 1) m1 j < bounds io.out1) →
      (∀ j : Fin io.C, io.writeMask2 (s.pids 0) (s.pids 1) m1 j →
        io.write2 (s.pids 0) (s.pids 1) m1 j < bounds io.out2) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (m1 : Nat) (xs : Fin io.B → ℝ),
      s₀.undef = (fun _ _ => 0) →
      s₀.readMemValue .nat io.mbuf1 (io.mwin1 (s₀.pids 0) (s₀.pids 1)) = m1 →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) m1 j →
        s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) m1 j) = xs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) m1 j →
            io.oty1.read s1 io.out1 (io.write1 (s₀.pids 0) (s₀.pids 1) m1 j)
              = (f (s₀.pids 0) (s₀.pids 1) m1 xs).1 j)
        ∧ (∀ j : Fin io.C, io.writeMask2 (s₀.pids 0) (s₀.pids 1) m1 j →
            io.oty2.read s1 io.out2 (io.write2 (s₀.pids 0) (s₀.pids 1) m1 j)
              = (f (s₀.pids 0) (s₀.pids 1) m1 xs).2 j)
        ∧ (∀ r o,
            (r ≠ io.out1 ∨
              ∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) m1 j →
                o ≠ io.write1 (s₀.pids 0) (s₀.pids 1) m1 j) →
            (r ≠ io.out2 ∨
              ∀ j : Fin io.C, io.writeMask2 (s₀.pids 0) (s₀.pids 1) m1 j →
                o ≠ io.write2 (s₀.pids 0) (s₀.pids 1) m1 j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- identical to `Implements.intro`, but the core's `undef` binder is
  -- threaded into `hrun` instead of discarded.
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals o => match o with
        | ⟨0, _⟩ => fun j =>
            (f p₀ p₁ (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1))
              (fun j' => vals (⟨1, by decide⟩ : Fin 2) j')).1 j
        | ⟨_+1, _⟩ => fun j =>
            (f p₀ p₁ (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1))
              (fun j' => vals (⟨1, by decide⟩ : Fin 2) j')).2 j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib hob _hsb
      exact hts bounds s
        (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1))
        (hpins (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hib (⟨1, by decide⟩ : Fin 2) j hj)
        (fun j hj => hob (⟨0, by decide⟩ : Fin 2) j hj)
        (fun j hj => hob (⟨1, by decide⟩ : Fin 2) j hj)
    · intro s₀ vals hundef hpins
      obtain ⟨s1, hexec, hval1, hval2, hframe⟩ :=
        hrun s₀ (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1))
          (fun j => vals (⟨1, by decide⟩ : Fin 2) j) hundef
          (hpins (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1) trivial)
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
  intro A hd hregs hcov pid₀ pid₁ m1 xs s₀ hpid₀ hpid₁ hu hb1 hbr hbw1 hbw2
    hm1 hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ => m1
        | ⟨_+1, _⟩ => xs)
      s₀ hpid₀ hpid₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hb1
        | ⟨_+1, _⟩ => fun j hj => hbr j hj)
      (fun o => match o with
        | ⟨0, _⟩ => fun j hj => hbw1 j hj
        | ⟨_+1, _⟩ => fun j hj => hbw2 j hj)
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hm1
        | ⟨_+1, _⟩ => fun j hj => hx j hj)
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

/-- `io.ImplementsR R f` — the **mixed-dtype rounding-correctness** relation
`io ⊨[R] f` for the two-output, unequal-length quantize skin. Same full Hoare
triple as `Implements` (∀ disjoint allocation, ∀ pids, ∀ launch state with the
slot pinned and the masked input tile loaded exact-ℝ), but the execution is
`execR R` and the two outputs split along the two axes:

* **`out1` — the `ChanTy` value axis.** Read back exactly through its channel
  view `io.oty1.read` (for the quantized tile, `oty1 := .int` ↦
  `readMemValue .int`). No rounding is applied to this readback: an `.int`
  store is exact even under `execR R`, so the honest contract is the *exact*
  carrier value `(f …).1 j`. Any `R`-dependence of that value (e.g. dividing
  by the fp16-rounded scale) lives inside the consumer's `f`, faithfully.
* **`out2` — the fp rounding axis.** A genuine reduced-precision-float store
  (`.to(tl.float16)` ↦ `writeMemAsR R .fp16`). Read back through
  `readMemAs io.out2DType` as the typed cell
  `io.out2DType.ofReal (R.round io.out2DType ((f …).2 j))` — the ideal real
  value `(f …).2 j : ℝ` quantized once at the declared grid. This is the
  channel `oty2 : ChanTy` cannot type; it is exactly the `KernelIO₂.ImplementsR`
  contract, per output lane.

At `R := .triv` and `out2DType := .real` the store is exact and this
degenerates to the exact `⊨` surface's second conjunct. -/
def ImplementsR (io : MetaMasked2DKernelIO₁ₓ₂) (R : RoundingModel)
    (f : Nat → Nat → Nat → (Fin io.B → ℝ) →
      (Fin io.B → io.oty1.carrier) × (Fin io.C → ℝ)) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.mbuf1, io.inp, io.out1, io.out2] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (m1 : Nat) (xs : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    io.mwin1 pid₀ pid₁ < A.extent io.mbuf1 →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m1 j →
      io.read pid₀ pid₁ m1 j < A.extent io.inp) →
    (∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ m1 j →
      io.write1 pid₀ pid₁ m1 j < A.extent io.out1) →
    (∀ j : Fin io.C, io.writeMask2 pid₀ pid₁ m1 j →
      io.write2 pid₀ pid₁ m1 j < A.extent io.out2) →
    s₀.readMemValue .nat io.mbuf1 (io.mwin1 pid₀ pid₁) = m1 →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m1 j →
      s₀.readMem io.inp (io.read pid₀ pid₁ m1 j) = xs j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ m1 j →
          io.oty1.read s' A.flat (A.addr io.out1 (io.write1 pid₀ pid₁ m1 j))
            = (f pid₀ pid₁ m1 xs).1 j)
      ∧ (∀ j : Fin io.C, io.writeMask2 pid₀ pid₁ m1 j →
          s'.readMemAs io.out2DType A.flat
              (A.addr io.out2 (io.write2 pid₀ pid₁ m1 j))
            = io.out2DType.ofReal
                (R.round io.out2DType ((f pid₀ pid₁ m1 xs).2 j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ m1 j →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ m1 j)) ∧
             (∀ j : Fin io.C, io.writeMask2 pid₀ pid₁ m1 j →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ m1 j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  MetaMasked2DKernelIO₁ₓ₂.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — the rounding sibling of `Implements.intro_undef`
(the fp16-scale quantizers reduce across the loaded tile, so the `undef` pin is
threaded into `hrun`). `FlattenOk`, the `TraceSafeR R` safety walk, and the
region-model rounded Hoare triple `hrun` (termination under `execR R`, the exact
`oty1` value readback of `out1`, the `readMemAs out2DType` rounded readback of
`out2`, and the two-output frame). The flat transport is `execR_flatten` plus
`ChanTy.read_flattenState` (for `out1`) and `flattenState_readMemAs` (for
`out2`) — no core embedding: the core's typed-output list is `ChanTy`-only and
`exec`-based, so a reduced-precision-float output on the rounding axis is
transported here directly. -/
theorem ImplementsR.intro (io : MetaMasked2DKernelIO₁ₓ₂) {R : RoundingModel}
    {f : Nat → Nat → Nat → (Fin io.B → ℝ) →
      (Fin io.B → io.oty1.carrier) × (Fin io.C → ℝ)}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m1 : Nat),
      s.readMemValue .nat io.mbuf1 (io.mwin1 (s.pids 0) (s.pids 1)) = m1 →
      io.mwin1 (s.pids 0) (s.pids 1) < bounds io.mbuf1 →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) m1 j →
        io.read (s.pids 0) (s.pids 1) m1 j < bounds io.inp) →
      (∀ j : Fin io.B, io.writeMask1 (s.pids 0) (s.pids 1) m1 j →
        io.write1 (s.pids 0) (s.pids 1) m1 j < bounds io.out1) →
      (∀ j : Fin io.C, io.writeMask2 (s.pids 0) (s.pids 1) m1 j →
        io.write2 (s.pids 0) (s.pids 1) m1 j < bounds io.out2) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (m1 : Nat) (xs : Fin io.B → ℝ),
      s₀.undef = (fun _ _ => 0) →
      s₀.readMemValue .nat io.mbuf1 (io.mwin1 (s₀.pids 0) (s₀.pids 1)) = m1 →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) m1 j →
        s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) m1 j) = xs j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) m1 j →
            io.oty1.read s1 io.out1 (io.write1 (s₀.pids 0) (s₀.pids 1) m1 j)
              = (f (s₀.pids 0) (s₀.pids 1) m1 xs).1 j)
        ∧ (∀ j : Fin io.C, io.writeMask2 (s₀.pids 0) (s₀.pids 1) m1 j →
            s1.readMemAs io.out2DType io.out2
                (io.write2 (s₀.pids 0) (s₀.pids 1) m1 j)
              = io.out2DType.ofReal
                  (R.round io.out2DType ((f (s₀.pids 0) (s₀.pids 1) m1 xs).2 j)))
        ∧ (∀ r o,
            (r ≠ io.out1 ∨
              ∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) m1 j →
                o ≠ io.write1 (s₀.pids 0) (s₀.pids 1) m1 j) →
            (r ≠ io.out2 ∨
              ∀ j : Fin io.C, io.writeMask2 (s₀.pids 0) (s₀.pids 1) m1 j →
                o ≠ io.write2 (s₀.pids 0) (s₀.pids 1) m1 j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ m1 xs s₀ hpid₀ hpid₁ hu hb1 hbr hbw1 hbw2 hm1 hx
  subst hpid₀
  subst hpid₁
  obtain ⟨s1, hexec, hval1, hval2, hframe⟩ := hrun s₀ m1 xs hu hm1 hx
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ m1 hm1 hb1 hbr hbw1 hbw2
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem1 : io.out1 ∈ A.regions := by rw [hregs]; simp
  have hmem2 : io.out2 ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, ?_, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j hj
    have hlt : io.write1 (s₀.pids 0) (s₀.pids 1) m1 j < A.extent io.out1 := hbw1 j hj
    rw [(io.oty1).read_flattenState A hd s1 hmem1 hlt]
    exact hval1 j hj
  · intro j hj
    have hlt : io.write2 (s₀.pids 0) (s₀.pids 1) m1 j < A.extent io.out2 := hbw2 j hj
    rw [A.flattenState_readMemAs hd s1 hmem2 hlt io.out2DType]
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
              refine Or.inr fun j hj => ?_
              rcases hcond with hflat | ⟨hn1, _⟩
              · exact absurd rfl hflat
              · intro hoj
                exact hn1 j hj (by rw [hoeq, hoj])
            · exact Or.inl hro
          · by_cases hro : r = io.out2
            · subst hro
              refine Or.inr fun j hj => ?_
              rcases hcond with hflat | ⟨_, hn2⟩
              · exact absurd rfl hflat
              · intro hoj
                exact hn2 j hj (by rw [hoeq, hoj])
            · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

end MetaMasked2DKernelIO₁ₓ₂

/-- IO signature of the **destination-index paired-tile copy shape**: one
per-program `.nat` scalar slot (the destination row index, `mbuf1`, loaded at
the pid-affine cell `mwin1`), **two unmasked float data tiles of independent
widths** (`in1`/`B1` and `in2`/`B2`), and **two float outputs of the matching
widths** (`out1`/`B1`, `out2`/`B2`) whose write windows are both driven by the
loaded slot. The subscript is pure data-input × output arity (two tiles ↦ two
tiles); the slot is capability (`Meta`), not arity, and the two tile widths
(`B1`/`B2`) are ordinary fields, so an unequal-width pair is *one* skin, not a
width-suffixed family. Intended consumer: the TritonBench-G LightLLM
destination-index KV copy (`destindex_copy` — `dest_index = tl.load(Dest_loc +
cur_index)`, then two verbatim `tl.store(O_* + dest_index·stride, tl.load(KV_*))`
of the nope/rope head tiles at `BLOCK_DMODEL_NOPE`/`BLOCK_DMODEL_ROPE`).
Following the family precedent there is no `scratch` field until a consumer
needs one.

**No output-typing field.** Unlike the two-output `MetaMasked2DKernelIO₁ₓ₂`
(whose quantizer consumers write a `.to(tl.int8)`-typed tile and therefore
carry `oty1`/`oty2 : ChanTy`), both stores here copy a float16 tile verbatim
with no `.to(...)` cast, so each output reads back through `readMem` at ℝ and
the spec `f` returns a plain `(Fin B1 → ℝ) × (Fin B2 → ℝ)` pair. The reduced
`fp16` storage precision is the orthogonal `ImplementsR`/rounding axis, not a
`ChanTy`, exactly as the `₁ₓ₂` note records — so a `ChanTy` output-type field
would be dead here and is omitted. -/
structure MetaMasked2DKernelIO₂ₓ₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- Scalar slot's buffer (one `.nat` cell per program). -/
  mbuf1 : RegionName
  /-- First input buffer — the `B1`-lane tile. -/
  in1 : RegionName
  /-- Second input buffer — the `B2`-lane tile. -/
  in2 : RegionName
  /-- First output buffer — the `B1`-lane tile. -/
  out1 : RegionName
  /-- Second output buffer — the `B2`-lane tile. -/
  out2 : RegionName
  /-- Tile length of the first input/output pair. -/
  B1 : Nat
  /-- Tile length of the second input/output pair. -/
  B2 : Nat
  /-- Slot's cell address for program `(pid₀, pid₁)`. -/
  mwin1 : Nat → Nat → Nat
  /-- First tile's read window at `(pid₀, pid₁, m1, j)` — the loaded slot is
  an ordinary named argument (the copy source ignores it). -/
  read1 : Nat → Nat → Nat → Fin B1 → Nat
  /-- Second tile's read window (loaded slot available as a named argument). -/
  read2 : Nat → Nat → Nat → Fin B2 → Nat
  /-- Lane `j`'s `out1` write address, given the loaded slot. -/
  write1 : Nat → Nat → Nat → Fin B1 → Nat
  /-- Lane `j`'s `out2` write address, given the loaded slot. -/
  write2 : Nat → Nat → Nat → Fin B2 → Nat
  /-- First tile's read-active lanes; defaults to always-on (the copy loads
  are unmasked). -/
  mask1 : Nat → Nat → Nat → Fin B1 → Prop := fun _ _ _ _ => True
  /-- Second tile's read-active lanes; defaults to always-on. -/
  mask2 : Nat → Nat → Nat → Fin B2 → Prop := fun _ _ _ _ => True
  /-- `out1`'s write-active lanes; defaults to `mask1` (the tile is stored
  exactly where it was loaded). -/
  writeMask1 : Nat → Nat → Nat → Fin B1 → Prop := mask1
  /-- `out2`'s write-active lanes; defaults to `mask2`. -/
  writeMask2 : Nat → Nat → Nat → Fin B2 → Prop := mask2

namespace MetaMasked2DKernelIO₂ₓ₂

/-- `io.Implements f` — the paired-tile sibling of
`MetaMasked2DKernelIO₁ₓ₂.Implements` with a *second independent input tile*.
The slot value `m1` is universally quantified and pinned to the slot cell
inside the memory precondition, so both read windows, both write windows,
their masks and the spec all speak about the *loaded* scalar. The spec `f`
returns the pair of the two outputs' value functions (`.1` for the `B1`-lane
`out1`, `.2` for the `B2`-lane `out2`), each pinned on its write-active lanes.
Frame: every cell outside the union of the two write-active output windows is
untouched. -/
def Implements (io : MetaMasked2DKernelIO₂ₓ₂)
    (f : Nat → Nat → Nat → (Fin io.B1 → ℝ) → (Fin io.B2 → ℝ) →
      (Fin io.B1 → ℝ) × (Fin io.B2 → ℝ)) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.mbuf1, io.in1, io.in2, io.out1, io.out2] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (m1 : Nat) (xs1 : Fin io.B1 → ℝ) (xs2 : Fin io.B2 → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    io.mwin1 pid₀ pid₁ < A.extent io.mbuf1 →
    (∀ j : Fin io.B1, io.mask1 pid₀ pid₁ m1 j →
      io.read1 pid₀ pid₁ m1 j < A.extent io.in1) →
    (∀ j : Fin io.B2, io.mask2 pid₀ pid₁ m1 j →
      io.read2 pid₀ pid₁ m1 j < A.extent io.in2) →
    (∀ j : Fin io.B1, io.writeMask1 pid₀ pid₁ m1 j →
      io.write1 pid₀ pid₁ m1 j < A.extent io.out1) →
    (∀ j : Fin io.B2, io.writeMask2 pid₀ pid₁ m1 j →
      io.write2 pid₀ pid₁ m1 j < A.extent io.out2) →
    s₀.readMemValue .nat io.mbuf1 (io.mwin1 pid₀ pid₁) = m1 →
    (∀ j : Fin io.B1, io.mask1 pid₀ pid₁ m1 j →
      s₀.readMem io.in1 (io.read1 pid₀ pid₁ m1 j) = xs1 j) →
    (∀ j : Fin io.B2, io.mask2 pid₀ pid₁ m1 j →
      s₀.readMem io.in2 (io.read2 pid₀ pid₁ m1 j) = xs2 j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B1, io.writeMask1 pid₀ pid₁ m1 j →
          s'.readMem A.flat (A.addr io.out1 (io.write1 pid₀ pid₁ m1 j))
            = (f pid₀ pid₁ m1 xs1 xs2).1 j)
      ∧ (∀ j : Fin io.B2, io.writeMask2 pid₀ pid₁ m1 j →
          s'.readMem A.flat (A.addr io.out2 (io.write2 pid₀ pid₁ m1 j))
            = (f pid₀ pid₁ m1 xs1 xs2).2 j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B1, io.writeMask1 pid₀ pid₁ m1 j →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ m1 j)) ∧
             (∀ j : Fin io.B2, io.writeMask2 pid₀ pid₁ m1 j →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ m1 j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => MetaMasked2DKernelIO₂ₓ₂.Implements

/-- Embed into the unified core: channel 0 is the 1-lane `.nat` slot (always
read), channels 1/2 the two float data tiles whose windows/masks read the
slot's pinned value; the two outputs carry the two lane counts (`B1`/`B2`)
through the core's per-output `oarity`. -/
private def toU (io : MetaMasked2DKernelIO₂ₓ₂) : UKernelIO where
  kernel := io.kernel
  nIn := 3
  nOut := 2
  nScr := 0
  bufs := [io.mbuf1, io.in1, io.in2, io.out1, io.out2]
  ity := fun i => match i with
    | ⟨0, _⟩ => .nat
    | ⟨1, _⟩ => .float
    | ⟨2, _⟩ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => io.B1
    | ⟨2, _⟩ => io.B2
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.mbuf1
    | ⟨1, _⟩ => io.in1
    | ⟨2, _⟩ => io.in2
  oty := fun _ => .float
  oarity := fun o => match o with
    | ⟨0, _⟩ => io.B1
    | ⟨_+1, _⟩ => io.B2
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.out1
    | ⟨_+1, _⟩ => io.out2
  obuf_mem := fun o => by fin_cases o <;> simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => io.mwin1 p₀ p₁
    | ⟨1, _⟩ => fun j => io.read1 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨2, _⟩ => fun j => io.read2 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) j
  imask := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun j => io.mask1 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨2, _⟩ => fun j => io.mask2 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) j
  owin := fun o vals p₀ p₁ _ => match o with
    | ⟨0, _⟩ => fun j => io.write1 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨_+1, _⟩ => fun j => io.write2 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) j
  omask := fun o vals p₀ p₁ _ => match o with
    | ⟨0, _⟩ => fun j => io.writeMask1 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨_+1, _⟩ => fun j => io.writeMask2 p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) j
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- Assembly lemma: obligations in the skin's named vocabulary — the slot
enters `hts`/`hrun` as a pinned named scalar (no lane-constancy plumbing:
slots are 1-lane channels), and `hrun`'s frame takes one exclusion condition
per output region. -/
theorem Implements.intro (io : MetaMasked2DKernelIO₂ₓ₂)
    {f : Nat → Nat → Nat → (Fin io.B1 → ℝ) → (Fin io.B2 → ℝ) →
      (Fin io.B1 → ℝ) × (Fin io.B2 → ℝ)}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m1 : Nat),
      s.readMemValue .nat io.mbuf1 (io.mwin1 (s.pids 0) (s.pids 1)) = m1 →
      io.mwin1 (s.pids 0) (s.pids 1) < bounds io.mbuf1 →
      (∀ j : Fin io.B1, io.mask1 (s.pids 0) (s.pids 1) m1 j →
        io.read1 (s.pids 0) (s.pids 1) m1 j < bounds io.in1) →
      (∀ j : Fin io.B2, io.mask2 (s.pids 0) (s.pids 1) m1 j →
        io.read2 (s.pids 0) (s.pids 1) m1 j < bounds io.in2) →
      (∀ j : Fin io.B1, io.writeMask1 (s.pids 0) (s.pids 1) m1 j →
        io.write1 (s.pids 0) (s.pids 1) m1 j < bounds io.out1) →
      (∀ j : Fin io.B2, io.writeMask2 (s.pids 0) (s.pids 1) m1 j →
        io.write2 (s.pids 0) (s.pids 1) m1 j < bounds io.out2) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (m1 : Nat) (xs1 : Fin io.B1 → ℝ)
        (xs2 : Fin io.B2 → ℝ),
      s₀.readMemValue .nat io.mbuf1 (io.mwin1 (s₀.pids 0) (s₀.pids 1)) = m1 →
      (∀ j : Fin io.B1, io.mask1 (s₀.pids 0) (s₀.pids 1) m1 j →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) m1 j) = xs1 j) →
      (∀ j : Fin io.B2, io.mask2 (s₀.pids 0) (s₀.pids 1) m1 j →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) m1 j) = xs2 j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B1, io.writeMask1 (s₀.pids 0) (s₀.pids 1) m1 j →
            s1.readMem io.out1 (io.write1 (s₀.pids 0) (s₀.pids 1) m1 j)
              = (f (s₀.pids 0) (s₀.pids 1) m1 xs1 xs2).1 j)
        ∧ (∀ j : Fin io.B2, io.writeMask2 (s₀.pids 0) (s₀.pids 1) m1 j →
            s1.readMem io.out2 (io.write2 (s₀.pids 0) (s₀.pids 1) m1 j)
              = (f (s₀.pids 0) (s₀.pids 1) m1 xs1 xs2).2 j)
        ∧ (∀ r o,
            (r ≠ io.out1 ∨
              ∀ j : Fin io.B1, io.writeMask1 (s₀.pids 0) (s₀.pids 1) m1 j →
                o ≠ io.write1 (s₀.pids 0) (s₀.pids 1) m1 j) →
            (r ≠ io.out2 ∨
              ∀ j : Fin io.B2, io.writeMask2 (s₀.pids 0) (s₀.pids 1) m1 j →
                o ≠ io.write2 (s₀.pids 0) (s₀.pids 1) m1 j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals o => match o with
        | ⟨0, _⟩ => fun j =>
            (f p₀ p₁ (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
              (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨2, by decide⟩ : Fin 3) j')).1 j
        | ⟨_+1, _⟩ => fun j =>
            (f p₀ p₁ (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
              (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨2, by decide⟩ : Fin 3) j')).2 j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib hob _hsb
      exact hts bounds s
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (hpins (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hib (⟨1, by decide⟩ : Fin 3) j hj)
        (fun j hj => hib (⟨2, by decide⟩ : Fin 3) j hj)
        (fun j hj => hob (⟨0, by decide⟩ : Fin 2) j hj)
        (fun j hj => hob (⟨1, by decide⟩ : Fin 2) j hj)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval1, hval2, hframe⟩ :=
        hrun s₀ (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
          (fun j => vals (⟨1, by decide⟩ : Fin 3) j)
          (fun j => vals (⟨2, by decide⟩ : Fin 3) j)
          (hpins (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial)
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
  intro A hd hregs hcov pid₀ pid₁ m1 xs1 xs2 s₀ hpid₀ hpid₁ hu hb1 hbr1 hbr2
    hbw1 hbw2 hm1 hx1 hx2
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ => m1
        | ⟨1, _⟩ => xs1
        | ⟨2, _⟩ => xs2)
      s₀ hpid₀ hpid₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hb1
        | ⟨1, _⟩ => fun j hj => hbr1 j hj
        | ⟨2, _⟩ => fun j hj => hbr2 j hj)
      (fun o => match o with
        | ⟨0, _⟩ => fun j hj => hbw1 j hj
        | ⟨_+1, _⟩ => fun j hj => hbw2 j hj)
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hm1
        | ⟨1, _⟩ => fun j hj => hx1 j hj
        | ⟨2, _⟩ => fun j hj => hx2 j hj)
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

end MetaMasked2DKernelIO₂ₓ₂

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

end MetaGroupedMasked2DKernelIO


/-! ### The index-tile genre: `Gather*` skins

`BoolScatterMasked2DKernelIO₁` models the scatter direction with a bool
select gate: an index tile drives the *write* window. The mirror
capability is a `.nat` index tile that drives the *read* window. Unlike a
`Meta*` scalar slot the loaded index is a **per-lane tile**, so it is a
`.nat` channel of the data channel's own arity whose pinned values feed
the data window — the core's `iwin`/`imask` take the full pinned-input
context, so this is expressible without touching the core. `Gather` names
the capability; the subscript stays pure data-input × output arity (the
index tile is capability, not arity). -/

/-- IO signature of an **index-tile gather/scatter** masked one-input /
one-output kernel: one `.nat` index tile (`idxbuf`, static window `readx`,
read gate `mask`) whose loaded values `ids` parametrize **both** the float
data tile's read window (`read … ids`, the gather direction) *and* the
output's write window (`write … ids`, the scatter direction). A pure
gather instantiates `write` to ignore `ids`; a pure scatter instantiates
`read` to ignore it. Intended consumers: the TritonBench-G
`index_select_cat` (`rows = tl.load(index_ptr + indices)`, then a source
read at the loaded rows and a static store) and `index_select_bwd` (the
same index tile, a static `grad_output` read and a store at the loaded
rows). Three regions — the genre has no bool gate, so it is *not*
`BoolScatterMasked2DKernelIO₁` with a constant-true gate: that skin's
allocation list carries a fourth, non-existent bool buffer. Following the
family precedent there is no `scratch` field until a consumer needs one.

**Injectivity design.** Because `write` may eat `ids`, two write-active
lanes can collide (`index_select_bwd` is routinely called with repeated
indices). The readback leg is therefore guarded by the *per-pinned-context*
antecedent `WriteInj pid₀ pid₁ ids`, exactly as in
`BoolScatterMasked2DKernelIO₁` — never a `∀`-quantified `intro`
hypothesis, which would demand injectivity for arbitrary index-buffer
contents. In the core embedding the raw (ungated) scatter cells are
shadowed by a scratch channel on the same buffer and window, which carries
the unconditional frame exclusion and the unconditional trace-safety write
bound. A pure-gather consumer discharges `WriteInj` from its static
`write` window. -/
structure GatherMasked2DKernelIO₁ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- Input buffer (ℝ channel). -/
  inp : RegionName
  /-- Index buffer (`.nat` channel — the per-lane gather/scatter rows). -/
  idxbuf : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Tile length: each program instance owns `B`-lane windows. -/
  B : Nat
  /-- Lane `j`'s `idxbuf` read address (the index tile's window — static,
  so the context stays causal). -/
  readx : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `inp` read address given the loaded index tile — the
  data-dependent gather source. -/
  read : Nat → Nat → (Fin B → Nat) → Fin B → Nat
  /-- Lane `j`'s write address given the loaded index tile — the
  data-dependent scatter destination. -/
  write : Nat → Nat → (Fin B → Nat) → Fin B → Nat
  /-- Program `(pid₀, pid₁)`'s **index-tile read-active** lanes (in the
  consumers, `indices < num_indices`). -/
  mask : Nat → Nat → Fin B → Prop
  /-- **Data read-active** lanes given the loaded index tile (in the
  consumers, the extra `cols < num_cols` conjunct); defaults to `mask`. -/
  readMask : Nat → Nat → (Fin B → Nat) → Fin B → Prop :=
    fun p₀ p₁ _ j => mask p₀ p₁ j
  /-- **Write-active** lanes given the loaded index tile; defaults to
  `mask`. -/
  writeMask : Nat → Nat → (Fin B → Nat) → Fin B → Prop :=
    fun p₀ p₁ _ j => mask p₀ p₁ j

namespace GatherMasked2DKernelIO₁

/-- No two write-active lanes share a destination — the per-pinned-context
injectivity that scatter readback needs. A pure gather (static `write`)
discharges it from its window; `index_select_bwd` gets it from the host's
no-duplicate-index guarantee. -/
def WriteInj (io : GatherMasked2DKernelIO₁) (p₀ p₁ : Nat)
    (ids : Fin io.B → Nat) : Prop :=
  ∀ j k : Fin io.B, io.writeMask p₀ p₁ ids j → io.writeMask p₀ p₁ ids k →
    io.write p₀ p₁ ids j = io.write p₀ p₁ ids k → j = k

/-- `io.Implements f` — the index-tile masked Hoare triple. The index tile
`ids` is quantified alongside the data tile and pinned on the index-read-
active lanes; both the data read window and the write window eat the
*loaded* `ids`. The readback leg is guarded by `WriteInj` (see the struct
docstring); the frame excludes the raw (ungated) write cells. -/
def Implements (io : GatherMasked2DKernelIO₁)
    (f : Nat → Nat → (Fin io.B → Nat) → (Fin io.B → ℝ) → Fin io.B → ℝ) :
    Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp, io.idxbuf, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (ids : Fin io.B → Nat) (xs : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.readx pid₀ pid₁ j < A.extent io.idxbuf) →
    (∀ j : Fin io.B, io.readMask pid₀ pid₁ ids j →
      io.read pid₀ pid₁ ids j < A.extent io.inp) →
    (∀ j : Fin io.B, io.writeMask pid₀ pid₁ ids j →
      io.write pid₀ pid₁ ids j < A.extent io.out) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMemValue .nat io.idxbuf (io.readx pid₀ pid₁ j) = ids j) →
    (∀ j : Fin io.B, io.readMask pid₀ pid₁ ids j →
      s₀.readMem io.inp (io.read pid₀ pid₁ ids j) = xs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (io.WriteInj pid₀ pid₁ ids →
          ∀ j : Fin io.B, io.writeMask pid₀ pid₁ ids j →
            s'.readMem A.flat (A.addr io.out (io.write pid₀ pid₁ ids j))
              = f pid₀ pid₁ ids xs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.B, io.writeMask pid₀ pid₁ ids j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ ids j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => GatherMasked2DKernelIO₁.Implements

/-- Embed into the unified core: channel 0 is the `.nat` index tile (static
window, so the context is causal), channel 1 the float data tile whose
window and read gate eat channel 0's pinned values. The output's window
eats the same values and its mask is the write gate **conjoined with
`WriteInj`**; a scratch channel shadows the same cells with the *ungated*
write gate, carrying the unconditional frame exclusion and write bound. -/
private def toU (io : GatherMasked2DKernelIO₁) : UKernelIO where
  kernel := io.kernel
  nIn := 2
  nOut := 1
  nScr := 1
  bufs := [io.inp, io.idxbuf, io.out]
  ity := fun i => match i with
    | ⟨0, _⟩ => .nat
    | ⟨_+1, _⟩ => .float
  iarity := fun _ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.idxbuf
    | ⟨_+1, _⟩ => io.inp
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun _ => io.B
  sbuf := fun _ => io.out
  iwin := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j => io.readx p₀ p₁ j
    | ⟨_+1, _⟩ => fun j => io.read p₀ p₁
        (fun j' => vals (⟨0, by decide⟩ : Fin 2) j') j
  imask := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j => io.mask p₀ p₁ j
    | ⟨_+1, _⟩ => fun j => io.readMask p₀ p₁
        (fun j' => vals (⟨0, by decide⟩ : Fin 2) j') j
  owin := fun _ vals p₀ p₁ _ j => io.write p₀ p₁
    (fun j' => vals (⟨0, by decide⟩ : Fin 2) j') j
  omask := fun _ vals p₀ p₁ _ j =>
    io.writeMask p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 2) j') j
    ∧ io.WriteInj p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 2) j')
  swin := fun _ vals p₀ p₁ _ j => io.write p₀ p₁
    (fun j' => vals (⟨0, by decide⟩ : Fin 2) j') j
  smask := fun _ vals p₀ p₁ _ j =>
    io.writeMask p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 2) j') j

/-- Assembly lemma: obligations in the skin's named vocabulary — the index
tile enters `hts`/`hrun` as a pinned tile, `hrun`'s value leg receives the
per-context `WriteInj` antecedent (a pure-gather consumer discharges it
from its static window), and `hrun`'s frame and the write bound are at the
ungated `writeMask` lanes. -/
theorem Implements.intro (io : GatherMasked2DKernelIO₁)
    {f : Nat → Nat → (Fin io.B → Nat) → (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (ids : Fin io.B → Nat),
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        s.readMemValue .nat io.idxbuf (io.readx (s.pids 0) (s.pids 1) j)
          = ids j) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.readx (s.pids 0) (s.pids 1) j < bounds io.idxbuf) →
      (∀ j : Fin io.B, io.readMask (s.pids 0) (s.pids 1) ids j →
        io.read (s.pids 0) (s.pids 1) ids j < bounds io.inp) →
      (∀ j : Fin io.B, io.writeMask (s.pids 0) (s.pids 1) ids j →
        io.write (s.pids 0) (s.pids 1) ids j < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (ids : Fin io.B → Nat)
        (xs : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMemValue .nat io.idxbuf (io.readx (s₀.pids 0) (s₀.pids 1) j)
          = ids j) →
      (∀ j : Fin io.B, io.readMask (s₀.pids 0) (s₀.pids 1) ids j →
        s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) ids j) = xs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (io.WriteInj (s₀.pids 0) (s₀.pids 1) ids →
            ∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) ids j →
              s1.readMem io.out (io.write (s₀.pids 0) (s₀.pids 1) ids j)
                = f (s₀.pids 0) (s₀.pids 1) ids xs j)
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) ids j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) ids j) →
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
    · intro bounds s vals hpins hib _hob hsb
      exact hts bounds s
        (fun j => vals (⟨0, by decide⟩ : Fin 2) j)
        (fun j hj => hpins (⟨0, by decide⟩ : Fin 2) j hj)
        (fun j hj => hib (⟨0, by decide⟩ : Fin 2) j hj)
        (fun j hj => hib (⟨1, by decide⟩ : Fin 2) j hj)
        (fun j hj => hsb (⟨0, by decide⟩ : Fin 1) j hj)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨0, by decide⟩ : Fin 2) j)
          (fun j => vals (⟨1, by decide⟩ : Fin 2) j)
          (fun j hj => hpins (⟨0, by decide⟩ : Fin 2) j hj)
          (fun j hj => hpins (⟨1, by decide⟩ : Fin 2) j hj)
      refine ⟨s1, hexec, fun _o j hj => hval hj.2 j hj.1, ?_⟩
      intro r o' _hoc hsc
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun j hj => ?_
        rcases hsc (⟨0, by decide⟩ : Fin 1) j hj with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ ids xs s₀ hpid₀ hpid₁ hu hbx hbr hbw hi hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => ids
        | ⟨_+1, _⟩ => xs)
      s₀ hpid₀ hpid₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hbx j hj
        | ⟨_+1, _⟩ => fun j hj => hbr j hj)
      (fun _o j hj => hbw j hj.1)
      (fun _t j hj => hbw j hj)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hi j hj
        | ⟨_+1, _⟩ => fun j hj => hx j hj)
  refine ⟨s', hexec,
    fun hinj j hj => hval (⟨0, by decide⟩ : Fin 1) j ⟨hj, hinj⟩, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | hn
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j hj => hn j hj.1, fun _t j hj => hn j hj⟩

end GatherMasked2DKernelIO₁

/-- IO signature of the **paired index-tile gather** shape: one `.nat`
metadata/index vector (`idxbuf`, static window `readx`, read gate `mask`)
whose loaded values `ids` drive one **shared** data read window `read`, two
per-output write windows (`write1`/`write2`) and the spec `f`, over **two
float input buffers** (`in1`/`in2`) read at that one window and **two float
outputs** (`out1`/`out2`). The index vector's cell count `N` is its **own
field**, decoupled from the data/output lane count `B`: the metadata vector
that a consumer folds over (the LightLLM cos/sin rotary cache transform's
`cumsum_lengths` of length `N_ELEMENTS`) is a different length from the copied
`HIDDEN_DIM` cache row, so a single-`B` gather cannot express it.

**Subscript.** Plain arity — **two data inputs × two outputs**, per the
house rule (subscripts carry data-input × output counts only). That both
input buffers happen to be read at the *one* shared `read` window driven by
the *one* index vector is a design property of this skin, not a naming axis;
the index vector is capability (the `Gather` prefix), not arity.
Intended consumers: the TritonBench-G `cache_transform_triton` pair —
`decoding_cache_kernel` (per-row `ori_seq_idx = tl.load(lengths + idx)` gathers
`cos_cache`/`sin_cache` rows into `cos_output`/`sin_output`) and
`prefill_cache_kernel` (the same dual gather, but the source row is
`idx − max(where(cumsum_lens ≤ idx, cumsum_lens, 0))`, a fold over the *entire*
`N_ELEMENTS`-cell `cumsum_lengths` vector — the fold sits inside the `read`
window, which eats the full pinned `ids : Fin N → Nat`, so the decoupled `N` is
all the skin needs). Both consumers' cos/sin rows are the same `HIDDEN_DIM`
width, so a single `B` with two write windows suffices. Following the family
precedent there is no `scratch` field until a consumer needs one.

**Injectivity design.** Because the write windows may eat `ids` (a scatter),
each output's readback leg is guarded by its *per-pinned-context* `WriteInj`
antecedent, exactly as in `GatherMasked2DKernelIO₁`/`ChainMetaMasked2DKernelIO₂ₓ₂`
— never an `Implements.intro` hypothesis, which would demand injectivity for
arbitrary index-buffer contents. In the core embedding two scratch channels
shadow the raw (ungated) write cells on the same buffers, carrying the
unconditional frame exclusions and trace-safety write bounds. The two
consumers' writes are static (`idx`-affine, not `ids`-dependent), so a
pure-gather consumer discharges both `WriteInj`s from its window geometry. -/
structure GatherMasked2DKernelIO₂ₓ₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- First input buffer (ℝ channel — e.g. `cos_cache`). -/
  in1 : RegionName
  /-- Second input buffer (ℝ channel — e.g. `sin_cache`). -/
  in2 : RegionName
  /-- Index buffer (`.nat` channel — the metadata vector). -/
  idxbuf : RegionName
  /-- First output buffer (e.g. `cos_output`). -/
  out1 : RegionName
  /-- Second output buffer (e.g. `sin_output`). -/
  out2 : RegionName
  /-- Data/output tile length: each program instance owns `B`-lane data read,
  `out1` write and `out2` write windows. -/
  B : Nat
  /-- **Index vector cell count** — its OWN field, decoupled from `B` (the
  fold's metadata vector `cumsum_lengths` has length `N_ELEMENTS ≠ HIDDEN_DIM`). -/
  N : Nat
  /-- Cell `j`'s `idxbuf` read address (the metadata vector's window — static,
  so the context stays causal). -/
  readx : Nat → Nat → Fin N → Nat
  /-- Lane `j`'s **shared** `in1`/`in2` read address given the loaded index
  vector — the data-dependent gather source (both caches read here). The whole
  `Fin N → Nat` context is available, so a fold over the entire vector fits. -/
  read : Nat → Nat → (Fin N → Nat) → Fin B → Nat
  /-- Lane `j`'s `out1` write address given the loaded index vector. -/
  write1 : Nat → Nat → (Fin N → Nat) → Fin B → Nat
  /-- Lane `j`'s `out2` write address given the loaded index vector. -/
  write2 : Nat → Nat → (Fin N → Nat) → Fin B → Nat
  /-- Program `(pid₀, pid₁)`'s **index-vector read-active** cells. -/
  mask : Nat → Nat → Fin N → Prop
  /-- **Data read-active** lanes given the loaded index vector. No default:
  its `Fin B` arity is decoupled from the index `mask`'s `Fin N` arity. -/
  readMask : Nat → Nat → (Fin N → Nat) → Fin B → Prop
  /-- `out1`'s write-active lanes; defaults to `readMask` (the gather stores
  each read lane exactly once). -/
  writeMask1 : Nat → Nat → (Fin N → Nat) → Fin B → Prop := readMask
  /-- `out2`'s write-active lanes; defaults to `readMask`. -/
  writeMask2 : Nat → Nat → (Fin N → Nat) → Fin B → Prop := readMask

namespace GatherMasked2DKernelIO₂ₓ₂

/-- No two `out1`-write-active lanes share a destination — the per-pinned-context
injectivity that scatter readback needs (the consumers' `hOutInj`). A pure
gather (static `write1`) discharges it from its window. -/
def WriteInj₁ (io : GatherMasked2DKernelIO₂ₓ₂) (p₀ p₁ : Nat)
    (ids : Fin io.N → Nat) : Prop :=
  ∀ j k : Fin io.B, io.writeMask1 p₀ p₁ ids j → io.writeMask1 p₀ p₁ ids k →
    io.write1 p₀ p₁ ids j = io.write1 p₀ p₁ ids k → j = k

/-- No two `out2`-write-active lanes share a destination. -/
def WriteInj₂ (io : GatherMasked2DKernelIO₂ₓ₂) (p₀ p₁ : Nat)
    (ids : Fin io.N → Nat) : Prop :=
  ∀ j k : Fin io.B, io.writeMask2 p₀ p₁ ids j → io.writeMask2 p₀ p₁ ids k →
    io.write2 p₀ p₁ ids j = io.write2 p₀ p₁ ids k → j = k

/-- `io.Implements f` — the paired index-tile masked Hoare triple. The index
vector `ids` is quantified alongside the two data tiles `xs`/`ys` and pinned on
the index-read-active cells; the shared data read window and both write windows
eat the *loaded* `ids`. The spec `f` returns the pair of the two outputs' value
functions (`.1` for `out1`, `.2` for `out2`); each readback leg is guarded by
its `WriteInj`; the frame excludes the two ungated write windows. -/
def Implements (io : GatherMasked2DKernelIO₂ₓ₂)
    (f : Nat → Nat → (Fin io.N → Nat) → (Fin io.B → ℝ) → (Fin io.B → ℝ) →
      (Fin io.B → ℝ) × (Fin io.B → ℝ)) :
    Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.idxbuf, io.out1, io.out2] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (ids : Fin io.N → Nat) (xs ys : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.N, io.mask pid₀ pid₁ j →
      io.readx pid₀ pid₁ j < A.extent io.idxbuf) →
    (∀ j : Fin io.B, io.readMask pid₀ pid₁ ids j →
      io.read pid₀ pid₁ ids j < A.extent io.in1) →
    (∀ j : Fin io.B, io.readMask pid₀ pid₁ ids j →
      io.read pid₀ pid₁ ids j < A.extent io.in2) →
    (∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ ids j →
      io.write1 pid₀ pid₁ ids j < A.extent io.out1) →
    (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ ids j →
      io.write2 pid₀ pid₁ ids j < A.extent io.out2) →
    (∀ j : Fin io.N, io.mask pid₀ pid₁ j →
      s₀.readMemValue .nat io.idxbuf (io.readx pid₀ pid₁ j) = ids j) →
    (∀ j : Fin io.B, io.readMask pid₀ pid₁ ids j →
      s₀.readMem io.in1 (io.read pid₀ pid₁ ids j) = xs j) →
    (∀ j : Fin io.B, io.readMask pid₀ pid₁ ids j →
      s₀.readMem io.in2 (io.read pid₀ pid₁ ids j) = ys j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (io.WriteInj₁ pid₀ pid₁ ids →
          ∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ ids j →
            s'.readMem A.flat (A.addr io.out1 (io.write1 pid₀ pid₁ ids j))
              = (f pid₀ pid₁ ids xs ys).1 j)
      ∧ (io.WriteInj₂ pid₀ pid₁ ids →
          ∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ ids j →
            s'.readMem A.flat (A.addr io.out2 (io.write2 pid₀ pid₁ ids j))
              = (f pid₀ pid₁ ids xs ys).2 j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ ids j →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ ids j)) ∧
             (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ ids j →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ ids j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => GatherMasked2DKernelIO₂ₓ₂.Implements

/-- Embed into the unified core: channel 0 is the `.nat` index vector of arity
`N` (static window, so the context is causal), channels 1/2 the two float data
tiles whose *shared* window and read gate eat channel 0's pinned values. Each
output's window eats the same values and its mask is the write gate **conjoined
with `WriteInj`**; two scratch channels shadow the same cells with the ungated
gates, carrying the unconditional frame exclusions and write bounds. -/
private def toU (io : GatherMasked2DKernelIO₂ₓ₂) : UKernelIO where
  kernel := io.kernel
  nIn := 3
  nOut := 2
  nScr := 2
  bufs := [io.in1, io.in2, io.idxbuf, io.out1, io.out2]
  ity := fun i => match i with
    | ⟨0, _⟩ => .nat
    | ⟨_+1, _⟩ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => io.N
    | ⟨_+1, _⟩ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.idxbuf
    | ⟨1, _⟩ => io.in1
    | ⟨_+2, _⟩ => io.in2
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.out1
    | ⟨_+1, _⟩ => io.out2
  obuf_mem := fun o => by fin_cases o <;> simp
  sarity := fun _ => io.B
  sbuf := fun t => match t with
    | ⟨0, _⟩ => io.out1
    | ⟨_+1, _⟩ => io.out2
  iwin := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j => io.readx p₀ p₁ j
    | ⟨1, _⟩ => fun j => io.read p₀ p₁
        (fun j' => vals (⟨0, by decide⟩ : Fin 3) j') j
    | ⟨_+2, _⟩ => fun j => io.read p₀ p₁
        (fun j' => vals (⟨0, by decide⟩ : Fin 3) j') j
  imask := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j => io.mask p₀ p₁ j
    | ⟨1, _⟩ => fun j => io.readMask p₀ p₁
        (fun j' => vals (⟨0, by decide⟩ : Fin 3) j') j
    | ⟨_+2, _⟩ => fun j => io.readMask p₀ p₁
        (fun j' => vals (⟨0, by decide⟩ : Fin 3) j') j
  owin := fun o vals p₀ p₁ _ => match o with
    | ⟨0, _⟩ => fun j => io.write1 p₀ p₁
        (fun j' => vals (⟨0, by decide⟩ : Fin 3) j') j
    | ⟨_+1, _⟩ => fun j => io.write2 p₀ p₁
        (fun j' => vals (⟨0, by decide⟩ : Fin 3) j') j
  omask := fun o vals p₀ p₁ _ => match o with
    | ⟨0, _⟩ => fun j =>
        io.writeMask1 p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 3) j') j
        ∧ io.WriteInj₁ p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 3) j')
    | ⟨_+1, _⟩ => fun j =>
        io.writeMask2 p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 3) j') j
        ∧ io.WriteInj₂ p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 3) j')
  swin := fun t vals p₀ p₁ _ => match t with
    | ⟨0, _⟩ => fun j => io.write1 p₀ p₁
        (fun j' => vals (⟨0, by decide⟩ : Fin 3) j') j
    | ⟨_+1, _⟩ => fun j => io.write2 p₀ p₁
        (fun j' => vals (⟨0, by decide⟩ : Fin 3) j') j
  smask := fun t vals p₀ p₁ _ => match t with
    | ⟨0, _⟩ => fun j => io.writeMask1 p₀ p₁
        (fun j' => vals (⟨0, by decide⟩ : Fin 3) j') j
    | ⟨_+1, _⟩ => fun j => io.writeMask2 p₀ p₁
        (fun j' => vals (⟨0, by decide⟩ : Fin 3) j') j

/-- Assembly lemma: obligations in the skin's named vocabulary — the index
vector enters `hts`/`hrun` as a pinned tile (gated by `mask`), the two shared
read pins feed `xs`/`ys`, each value leg receives its per-context `WriteInj`
antecedent (a pure-gather consumer discharges it from its static window), and
`hrun`'s frame and the write bounds are at the ungated `writeMask` lanes. -/
theorem Implements.intro (io : GatherMasked2DKernelIO₂ₓ₂)
    {f : Nat → Nat → (Fin io.N → Nat) → (Fin io.B → ℝ) → (Fin io.B → ℝ) →
      (Fin io.B → ℝ) × (Fin io.B → ℝ)}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (ids : Fin io.N → Nat),
      (∀ j : Fin io.N, io.mask (s.pids 0) (s.pids 1) j →
        s.readMemValue .nat io.idxbuf (io.readx (s.pids 0) (s.pids 1) j)
          = ids j) →
      (∀ j : Fin io.N, io.mask (s.pids 0) (s.pids 1) j →
        io.readx (s.pids 0) (s.pids 1) j < bounds io.idxbuf) →
      (∀ j : Fin io.B, io.readMask (s.pids 0) (s.pids 1) ids j →
        io.read (s.pids 0) (s.pids 1) ids j < bounds io.in1) →
      (∀ j : Fin io.B, io.readMask (s.pids 0) (s.pids 1) ids j →
        io.read (s.pids 0) (s.pids 1) ids j < bounds io.in2) →
      (∀ j : Fin io.B, io.writeMask1 (s.pids 0) (s.pids 1) ids j →
        io.write1 (s.pids 0) (s.pids 1) ids j < bounds io.out1) →
      (∀ j : Fin io.B, io.writeMask2 (s.pids 0) (s.pids 1) ids j →
        io.write2 (s.pids 0) (s.pids 1) ids j < bounds io.out2) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (ids : Fin io.N → Nat)
        (xs ys : Fin io.B → ℝ),
      (∀ j : Fin io.N, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMemValue .nat io.idxbuf (io.readx (s₀.pids 0) (s₀.pids 1) j)
          = ids j) →
      (∀ j : Fin io.B, io.readMask (s₀.pids 0) (s₀.pids 1) ids j →
        s₀.readMem io.in1 (io.read (s₀.pids 0) (s₀.pids 1) ids j) = xs j) →
      (∀ j : Fin io.B, io.readMask (s₀.pids 0) (s₀.pids 1) ids j →
        s₀.readMem io.in2 (io.read (s₀.pids 0) (s₀.pids 1) ids j) = ys j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (io.WriteInj₁ (s₀.pids 0) (s₀.pids 1) ids →
            ∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) ids j →
              s1.readMem io.out1 (io.write1 (s₀.pids 0) (s₀.pids 1) ids j)
                = (f (s₀.pids 0) (s₀.pids 1) ids xs ys).1 j)
        ∧ (io.WriteInj₂ (s₀.pids 0) (s₀.pids 1) ids →
            ∀ j : Fin io.B, io.writeMask2 (s₀.pids 0) (s₀.pids 1) ids j →
              s1.readMem io.out2 (io.write2 (s₀.pids 0) (s₀.pids 1) ids j)
                = (f (s₀.pids 0) (s₀.pids 1) ids xs ys).2 j)
        ∧ (∀ r o,
            (r ≠ io.out1 ∨
              ∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) ids j →
                o ≠ io.write1 (s₀.pids 0) (s₀.pids 1) ids j) →
            (r ≠ io.out2 ∨
              ∀ j : Fin io.B, io.writeMask2 (s₀.pids 0) (s₀.pids 1) ids j →
                o ≠ io.write2 (s₀.pids 0) (s₀.pids 1) ids j) →
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
        | ⟨_+1, _⟩ => fun j =>
            (f p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨2, by decide⟩ : Fin 3) j')).2 j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib _hob hsb
      exact hts bounds s
        (fun j' => vals (⟨0, by decide⟩ : Fin 3) j')
        (fun j hj => hpins (⟨0, by decide⟩ : Fin 3) j hj)
        (fun j hj => hib (⟨0, by decide⟩ : Fin 3) j hj)
        (fun j hj => hib (⟨1, by decide⟩ : Fin 3) j hj)
        (fun j hj => hib (⟨2, by decide⟩ : Fin 3) j hj)
        (fun j hj => hsb (⟨0, by decide⟩ : Fin 2) j hj)
        (fun j hj => hsb (⟨1, by decide⟩ : Fin 2) j hj)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval1, hval2, hframe⟩ :=
        hrun s₀ (fun j' => vals (⟨0, by decide⟩ : Fin 3) j')
          (fun j => vals (⟨1, by decide⟩ : Fin 3) j)
          (fun j => vals (⟨2, by decide⟩ : Fin 3) j)
          (fun j hj => hpins (⟨0, by decide⟩ : Fin 3) j hj)
          (fun j hj => hpins (⟨1, by decide⟩ : Fin 3) j hj)
          (fun j hj => hpins (⟨2, by decide⟩ : Fin 3) j hj)
      refine ⟨s1, hexec, fun o => match o with
        | ⟨0, _⟩ => fun j hj => hval1 hj.2 j hj.1
        | ⟨_+1, _⟩ => fun j hj => hval2 hj.2 j hj.1, ?_⟩
      intro r o' _hoc hsc
      refine hframe r o' ?_ ?_
      · by_cases hro : r = io.out1
        · subst hro
          refine Or.inr fun j hj => ?_
          rcases hsc (⟨0, by decide⟩ : Fin 2) j hj with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · by_cases hro : r = io.out2
        · subst hro
          refine Or.inr fun j hj => ?_
          rcases hsc (⟨1, by decide⟩ : Fin 2) j hj with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ ids xs ys s₀ hpid₀ hpid₁ hu hbx hbr1 hbr2
    hbw1 hbw2 hi hx hy
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => ids
        | ⟨1, _⟩ => xs
        | ⟨_+2, _⟩ => ys)
      s₀ hpid₀ hpid₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hbx j hj
        | ⟨1, _⟩ => fun j hj => hbr1 j hj
        | ⟨_+2, _⟩ => fun j hj => hbr2 j hj)
      (fun o => match o with
        | ⟨0, _⟩ => fun j hj => hbw1 j hj.1
        | ⟨_+1, _⟩ => fun j hj => hbw2 j hj.1)
      (fun t => match t with
        | ⟨0, _⟩ => fun j hj => hbw1 j hj
        | ⟨_+1, _⟩ => fun j hj => hbw2 j hj)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hi j hj
        | ⟨1, _⟩ => fun j hj => hx j hj
        | ⟨_+2, _⟩ => fun j hj => hy j hj)
  refine ⟨s', hexec,
    fun hinj j hj => hval (⟨0, by decide⟩ : Fin 2) j ⟨hj, hinj⟩,
    fun hinj j hj => hval (⟨1, by decide⟩ : Fin 2) j ⟨hj, hinj⟩, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hn1, hn2⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun oc => match oc with
      | ⟨0, _⟩ => fun j hj => hn1 j hj.1
      | ⟨_+1, _⟩ => fun j hj => hn2 j hj.1,
      fun t => match t with
      | ⟨0, _⟩ => fun j hj => hn1 j hj
      | ⟨_+1, _⟩ => fun j hj => hn2 j hj⟩

end GatherMasked2DKernelIO₂ₓ₂

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

/-! ### The streaming genre: `Stream*` skins

The `forRange`-loop kernels stream their inputs: for `t = 0 .. T-1` the
program loads one per-step tile from each input channel, folds it into a
register accumulator, and after the loop stores the `C`-lane result once.
The skins keep the per-step view **curried** — every input channel is
indexed `(t : Fin T, lane : Fin B)`, never flattened to `Fin (T*B)` — so a
consumer's windows, masks and spec all speak per step, exactly the way the
kernel's loop body does. `Stream` names the capability; the subscript stays
pure data-input × output arity (`T` and the lane counts are fields, never
name subscripts). -/

/-- IO signature of the **two-stream fold** shape (streaming genre, style
S1: fold + terminal store): a 2-D pid grid, **two streamed float input
channels** (`inp1`/`inp2`) read in `T` loop steps of `B1`/`B2` lanes each,
and **one output channel** (`out`, `C` lanes) written once after the loop.
The kernel folds the `T` per-step tile pairs into a register accumulator;
the spec `f` eats the *whole* curried streams `(Fin T → Fin B1 → ℝ)` /
`(Fin T → Fin B2 → ℝ)` and returns the terminal value directly — the fold
is expressed inside the consumer's `f` (a GEMM consumer writes
`∑ t, ∑ e, …`), and the skin does not prove the loop for the consumer: the
`ImplementsR.intro` obligation `hrun` is discharged with a
`forRange`-invariant argument on the consumer's side. Intended consumers:
the TritonBench-G matmul family and its streamed-reduction relatives.
Following the family precedent there is no `scratch` field until a consumer
needs one.

**Single-surface design.** The genre carries only the rounding-correctness
relation `⊨[R]` (`ImplementsR`, `execR R`) — no exact-`exec` `Implements`
and no `toU` core embedding. A streaming kernel's terminal store is
routinely a narrow-float typed store (the matmul family stores
`.to(tl.float16)`), and the exact surface's `readMem` readback strictly
mismatches a narrow-float cell; under `execR R` a `.real` store is exact
anyway (`stepStmtR` delegates `.real` writes to the exact semantics), so
the single `⊨[R]` surface with the `outDType := .real` default covers the
exact genre losslessly — the exact surface *is* this relation's
`R := .triv` degeneration, the direct continuation of the
rounding-as-default doctrine. -/
structure StreamMasked2DKernelIO₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- First streamed input buffer. -/
  inp1 : RegionName
  /-- Second streamed input buffer. -/
  inp2 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Number of streaming steps (the `forRange` trip count). -/
  T : Nat
  /-- Per-step tile length of the first input channel. -/
  B1 : Nat
  /-- Per-step tile length of the second input channel. -/
  B2 : Nat
  /-- Tile length of the terminal output store. -/
  C : Nat
  /-- The output buffer's floating dtype — the quantization grid of the
  terminal boundary store. `⊨[R]`'s postcondition reads the output back as
  `outDType`-typed cells holding `outDType.ofReal (R.round outDType (f …))`;
  an fp16-storing streaming kernel sets `.fp16`. `.real` (the default) is an
  unrounded store — exact under `execR R`, so the default recovers the
  exact genre. -/
  outDType : FloatDType := .real
  /-- Step `t`, lane `j`'s `inp1` read address for program `(pid₀, pid₁)`. -/
  read1 : Nat → Nat → Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s `inp2` read address. -/
  read2 : Nat → Nat → Fin T → Fin B2 → Nat
  /-- Lane `j`'s terminal write address. -/
  write : Nat → Nat → Fin C → Nat
  /-- `inp1`'s read-active lanes at step `t`. -/
  mask1 : Nat → Nat → Fin T → Fin B1 → Prop
  /-- `inp2`'s read-active lanes at step `t`. -/
  mask2 : Nat → Nat → Fin T → Fin B2 → Prop
  /-- The terminal store's write-active lanes. -/
  writeMask : Nat → Nat → Fin C → Prop

namespace StreamMasked2DKernelIO₂

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the two-stream fold skin, the genre's only surface (see the
structure's single-surface design note). Same full Hoare triple as the
family's `ImplementsR` relations (∀ disjoint allocation, ∀ pids, ∀ launch
state with both masked input streams loaded exact-ℝ step by step), the
execution is `execR R`, and the single output readback is the
`KernelIO₂.ImplementsR` contract per lane, generalized to the streamed
inputs: the terminal cell holds the *ideal* real value `f pid₀ pid₁ xs ys j`
of the whole `T`-step fold, quantized **once** at the declared grid
`outDType` — read back through `readMemAs io.outDType` as
`io.outDType.ofReal (R.round io.outDType (f …))`. Frame: every flat cell
outside the write-active output window is untouched. At `R := .triv` and
`outDType := .real` the store is exact and this is the exact streaming
contract. -/
def ImplementsR (io : StreamMasked2DKernelIO₂) (R : RoundingModel)
    (f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.inp2, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
    (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ t j →
      io.read1 pid₀ pid₁ t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ t j →
      io.read2 pid₀ pid₁ t j < A.extent io.inp2) →
    (∀ j : Fin io.C, io.writeMask pid₀ pid₁ j →
      io.write pid₀ pid₁ j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ t j) = ys t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C, io.writeMask pid₀ pid₁ j →
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ j))
            = io.outDType.ofReal
                (R.round io.outDType (f pid₀ pid₁ xs ys j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.C, io.writeMask pid₀ pid₁ j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamMasked2DKernelIO₂.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, mirroring
`MetaMasked2DKernelIO₁ₓ₂.ImplementsR.intro` (no core embedding: the core is
`exec`-based and its typed-output list is `ChanTy`-only, so the streamed
rounding surface is transported here directly via `execR_flatten` and
`flattenState_readMemAs`). Obligations in the skin's named vocabulary:
`FlattenOk`, the `TraceSafeR R` safety walk `hts` (fed the two pinned input
streams and the three window-bound groups), and the region-model rounded
Hoare triple `hrun` (termination under `execR R`, the `readMemAs outDType`
rounded readback of the terminal store, and the single-output frame; the
`undef` pin is threaded in for masked loads without an `other=` default).
`hrun` is where the consumer runs its `forRange` invariant argument — the
skin does not prove the loop. -/
theorem ImplementsR.intro (io : StreamMasked2DKernelIO₂) {R : RoundingModel}
    {f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → Fin io.C → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ),
      (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s.pids 0) (s.pids 1) t j →
        s.readMem io.inp1 (io.read1 (s.pids 0) (s.pids 1) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 (s.pids 0) (s.pids 1) t j →
        s.readMem io.inp2 (io.read2 (s.pids 0) (s.pids 1) t j) = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s.pids 0) (s.pids 1) t j →
        io.read1 (s.pids 0) (s.pids 1) t j < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 (s.pids 0) (s.pids 1) t j →
        io.read2 (s.pids 0) (s.pids 1) t j < bounds io.inp2) →
      (∀ j : Fin io.C, io.writeMask (s.pids 0) (s.pids 1) j →
        io.write (s.pids 0) (s.pids 1) j < bounds io.out) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.T → Fin io.B1 → ℝ)
        (ys : Fin io.T → Fin io.B2 → ℝ),
      s₀.undef = (fun _ _ => 0) →
      (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s₀.pids 0) (s₀.pids 1) t j →
        s₀.readMem io.inp1 (io.read1 (s₀.pids 0) (s₀.pids 1) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 (s₀.pids 0) (s₀.pids 1) t j →
        s₀.readMem io.inp2 (io.read2 (s₀.pids 0) (s₀.pids 1) t j) = ys t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.C, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
            s1.readMemAs io.outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) j)
              = io.outDType.ofReal
                  (R.round io.outDType (f (s₀.pids 0) (s₀.pids 1) xs ys j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.C, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ xs ys s₀ hpid₀ hpid₁ hu hbr1 hbr2 hbw hx hy
  subst hpid₀
  subst hpid₁
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs ys hu hx hy
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ xs ys hx hy hbr1 hbr2 hbw
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem : io.out ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j hj
    have hlt : io.write (s₀.pids 0) (s₀.pids 1) j < A.extent io.out :=
      hbw j hj
    rw [A.flattenState_readMemAs hd s1 hmem hlt io.outDType]
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
          refine congrArg A.trCell (hframe r o ?_)
          by_cases hro : r = io.out
          · subst hro
            refine Or.inr fun j hj => ?_
            rcases hcond with hflat | hn
            · exact absurd rfl hflat
            · intro hoj
              exact hn j hj (by rw [hoeq, hoj])
          · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

end StreamMasked2DKernelIO₂



end VeriTile.Triton
