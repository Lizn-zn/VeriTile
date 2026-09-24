/-
Kernel IO contracts: Masked.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.KernelSpec.Base

namespace VeriTile.Triton

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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  projection := io.projection
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

/-- `io.ImplementsR R outDType f` — the **rounding-correctness** relation
for the masked two-input family, written `io ⊨[R, outDType] f`. Verbatim
`Implements`, with two changes: the kernel runs under the rounding model
(`execR R`), and each **active** lane of the output window is read back as
an `outDType`-typed cell holding the ideal real value quantized **once**,
`outDType.ofReal (R.round outDType (f xs ys j))`. Inputs stay exact ℝ —
the rounding model acts at the kernel's cast/store sites, not at loads.
Everything else (the allocation contract, the four in-bounds obligations,
the scratch channels, the `pid`/`undef` pins, the frame) is unchanged; at
`outDType := .real` the store is exact and this degenerates to the exact
surface.

The grid is an **argument**, and the notation names it, exactly as in
`MaskedKernelIO₁`: this skin's consumers are the elementwise binary genre
(`vector_addition`, `swiglu_triton`, `geglu_tanh_triton`,
`softmax_triton3`), several of which load `.to(tl.float32)` out of a
half-precision tensor, compute wide, and store back into the *narrow*
buffer. Both a `.fp16` face and a `.real` face are therefore live for the
same signature, and a three-hole `io ⊨[R] f` would print them
identically. -/
def ImplementsR (io : MaskedKernelIO₂) (R : RoundingModel)
    (outDType : FloatDType)
    (f : (Fin io.B → ℝ) → (Fin io.B → ℝ) → Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.out] ++ io.scratch.map Prod.fst →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    (∀ j : Fin io.B, io.mask pid j → io.read1 pid + j.val < A.extent io.in1) →
    (∀ j : Fin io.B, io.mask pid j → io.read2 pid + j.val < A.extent io.in2) →
    (∀ j : Fin io.B, io.mask pid j → io.write pid + j.val < A.extent io.out) →
    (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.mask pid j →
      p.2 pid + j.val < A.extent p.1) →
  ∀ (xs ys : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, io.mask pid j →
      s₀.readMem io.in1 (io.read1 pid + j.val) = xs j) →
    (∀ j : Fin io.B, io.mask pid j →
      s₀.readMem io.in2 (io.read2 pid + j.val) = ys j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.mask pid j →
          s'.readMemAs outDType A.flat
              (A.addr io.out (io.write pid + j.val))
            = outDType.ofReal (R.round outDType (f xs ys j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.mask pid j →
                o' ≠ A.addr io.out (io.write pid + j.val)) ∧
             (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.mask pid j →
                o' ≠ A.addr p.1 (p.2 pid + j.val)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R ", " outDType "] " f =>
  MaskedKernelIO₂.ImplementsR io R outDType f

/-- Assembly lemma for `⊨[R, outDType]` — the rounding sibling of
`MaskedKernelIO₂.Implements.intro`, riding the single-shot family's
rounding core `UKernelIO.ImplementsR.intro` through the same `toU`
embedding (at the constant output grid `fun _ => outDType`). The
obligations are the family's usual three, with the safety walk at
`Kernel.TraceSafeR R` and `hrun` returning a rounded region-model triple:
termination under `execR R`, the `readMemAs outDType` per-lane readback,
and the frame. -/
theorem ImplementsR.intro (io : MaskedKernelIO₂) {R : RoundingModel}
    {outDType : FloatDType}
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
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs ys : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask s₀.pid j →
        s₀.readMem io.in1 (io.read1 s₀.pid + j.val) = xs j) →
      (∀ j : Fin io.B, io.mask s₀.pid j →
        s₀.readMem io.in2 (io.read2 s₀.pid + j.val) = ys j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀
          = some s1
        ∧ (∀ j : Fin io.B, io.mask s₀.pid j →
            s1.readMemAs outDType io.out (io.write s₀.pid + j.val)
              = outDType.ofReal (R.round outDType (f xs ys j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.B, io.mask s₀.pid j →
                o ≠ io.write s₀.pid + j.val) →
            (∀ p ∈ io.scratch, r = p.1 →
              ∀ j : Fin io.B, io.mask s₀.pid j →
                o ≠ p.2 s₀.pid + j.val) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R outDType f := by
  -- assemble the unified-core rounded triple once, then convert it back
  -- into the family statement; the flattening bridge lives in
  -- `UKernelIO.ImplementsR.intro`
  have hcore : io.toU.ImplementsR R (fun _ => outDType)
      (fun _p₀ _p₁ vals _o j =>
        f (fun j' => vals (⟨0, by decide⟩ : Fin 2) j')
          (fun j' => vals (⟨1, by decide⟩ : Fin 2) j') j) := by
    refine UKernelIO.ImplementsR.intro _ hok ?_ ?_
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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  projection := io.projection
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

/-- `io.ImplementsR R outDType f` — the **rounding-correctness** relation
for the masked one-input family, written `io ⊨[R, outDType] f`. Verbatim
`Implements`, with two changes: the kernel runs under the rounding model
(`execR R`), and each **write-active** lane of the output window is read
back as an `outDType`-typed cell holding the ideal real value quantized
**once**, `outDType.ofReal (R.round outDType (f xs j))`. Inputs stay exact
ℝ — the rounding model acts at the kernel's cast/store sites, not at
loads. Everything else (the allocation contract, the three in-bounds
obligations, the scratch channels, the `pid`/`undef` pins, the frame) is
unchanged; at `outDType := .real` the store is exact and this degenerates
to the exact surface.

Unlike the `Stream*` family — where `outDType` is a *field* of the
signature, so `io ⊨[R] f` can read the grid off `io` — `MaskedKernelIO₁`'s
grid is an **argument**, chosen per headline. The notation therefore has
four holes and names it: hiding it behind a three-hole `io ⊨[R] f` would
make two different quantization grids print identically. -/
def ImplementsR (io : MaskedKernelIO₁) (R : RoundingModel)
    (outDType : FloatDType) (f : (Fin io.B → ℝ) → Fin io.B → ℝ) : Prop :=
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
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask pid j →
          s'.readMemAs outDType A.flat
              (A.addr io.out (io.write pid + j.val))
            = outDType.ofReal (R.round outDType (f xs j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.writeMask pid j →
                o' ≠ A.addr io.out (io.write pid + j.val)) ∧
             (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.writeMask pid j →
                o' ≠ A.addr p.1 (p.2 pid + j.val)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R ", " outDType "] " f =>
  MaskedKernelIO₁.ImplementsR io R outDType f

/-- Assembly lemma for `⊨[R, outDType]` — the rounding sibling of
`MaskedKernelIO₁.Implements.intro`, riding the single-shot family's
rounding core `UKernelIO.ImplementsR.intro` through the same `toU`
embedding (at the constant output grid `fun _ => outDType`). The
obligations are the family's usual three, with the safety walk at
`Kernel.TraceSafeR R` and `hrun` returning a rounded region-model triple:
termination under `execR R`, the `readMemAs outDType` per-lane readback,
and the frame. -/
theorem ImplementsR.intro (io : MaskedKernelIO₁) {R : RoundingModel}
    {outDType : FloatDType} {f : (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ j : Fin io.B, io.mask s.pid j →
        io.read s.pid + j.val < bounds io.inp) →
      (∀ j : Fin io.B, io.writeMask s.pid j →
        io.write s.pid + j.val < bounds io.out) →
      (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.writeMask s.pid j →
        p.2 s.pid + j.val < bounds p.1) →
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask s₀.pid j →
        s₀.readMem io.inp (io.read s₀.pid + j.val) = xs j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀
          = some s1
        ∧ (∀ j : Fin io.B, io.writeMask s₀.pid j →
            s1.readMemAs outDType io.out (io.write s₀.pid + j.val)
              = outDType.ofReal (R.round outDType (f xs j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.B, io.writeMask s₀.pid j →
                o ≠ io.write s₀.pid + j.val) →
            (∀ p ∈ io.scratch, r = p.1 →
              ∀ j : Fin io.B, io.writeMask s₀.pid j →
                o ≠ p.2 s₀.pid + j.val) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R outDType f := by
  -- assemble the unified-core rounded triple once, then convert it back
  -- into the family statement; the flattening bridge lives in
  -- `UKernelIO.ImplementsR.intro`
  have hcore : io.toU.ImplementsR R (fun _ => outDType)
      (fun _p₀ _p₁ vals _o j =>
        f (fun j' => vals (⟨0, by decide⟩ : Fin 1) j') j) := by
    refine UKernelIO.ImplementsR.intro _ hok ?_ ?_
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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  projection := io.projection
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

/-- `io.ImplementsR R out1DType out2DType f` — the **rounding-correctness**
relation for the masked three-input / two-output family, written
`io ⊨[R, out1DType, out2DType] f`. Verbatim `Implements`, with two changes:
the kernel runs under the rounding model (`execR R`), and each **active**
lane of each output window is read back as a typed cell holding the ideal
real value quantized **once** — `out1DType` for `out1`, `out2DType` for
`out2`. Inputs stay exact ℝ — the rounding model acts at the kernel's
cast/store sites, not at loads. Everything else (the allocation contract,
the five in-bounds obligations, the `pid`/`undef` pins, the frame) is
unchanged; at both grids `.real` the stores are exact and this degenerates
to the exact surface.

The grids are **arguments** and there is **one per output channel**, both
named by the notation. This skin's genre is the in-place optimizer step
(`adam_update_triton`: `bufs = [p, grad, m]` with `out1 = in1 = p`), where
the parameter buffer and the moment buffer routinely live at *different*
precisions — half-precision weights alongside wide moments — so a single
scalar grid reused for both outputs would be the wrong shape, not merely a
coarse one. A three-hole `io ⊨[R] f` would in addition print two different
quantization assignments identically. -/
def ImplementsR (io : MaskedKernelIO₃ₓ₂) (R : RoundingModel)
    (out1DType out2DType : FloatDType)
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
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.mask pid j →
          s'.readMemAs out1DType A.flat
              (A.addr io.out1 (io.write1 pid + j.val))
            = out1DType.ofReal (R.round out1DType ((f xs ys zs).1 j)))
      ∧ (∀ j : Fin io.B, io.mask pid j →
          s'.readMemAs out2DType A.flat
              (A.addr io.out2 (io.write2 pid + j.val))
            = out2DType.ofReal (R.round out2DType ((f xs ys zs).2 j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.mask pid j →
                o' ≠ A.addr io.out1 (io.write1 pid + j.val)) ∧
             (∀ j : Fin io.B, io.mask pid j →
                o' ≠ A.addr io.out2 (io.write2 pid + j.val)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R ", " out1DType ", "
    out2DType "] " f =>
  MaskedKernelIO₃ₓ₂.ImplementsR io R out1DType out2DType f

/-- Assembly lemma for `⊨[R, out1DType, out2DType]` — the rounding sibling
of `MaskedKernelIO₃ₓ₂.Implements.intro`, riding the single-shot family's
rounding core `UKernelIO.ImplementsR.intro` through the same `toU`
embedding (at the per-channel output grid `out1DType`/`out2DType`).
Obligations as there — the two membership side conditions, `FlattenOk`, the
safety walk at `Kernel.TraceSafeR R`, and `hrun` returning a rounded
region-model triple: termination under `execR R`, the two `readMemAs`
per-lane readbacks, and the frame. -/
theorem ImplementsR.intro (io : MaskedKernelIO₃ₓ₂) {R : RoundingModel}
    {out1DType out2DType : FloatDType}
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
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs ys zs : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask s₀.pid j →
        s₀.readMem io.in1 (io.read1 s₀.pid + j.val) = xs j) →
      (∀ j : Fin io.B, io.mask s₀.pid j →
        s₀.readMem io.in2 (io.read2 s₀.pid + j.val) = ys j) →
      (∀ j : Fin io.B, io.mask s₀.pid j →
        s₀.readMem io.in3 (io.read3 s₀.pid + j.val) = zs j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀
          = some s1
        ∧ (∀ j : Fin io.B, io.mask s₀.pid j →
            s1.readMemAs out1DType io.out1 (io.write1 s₀.pid + j.val)
              = out1DType.ofReal (R.round out1DType ((f xs ys zs).1 j)))
        ∧ (∀ j : Fin io.B, io.mask s₀.pid j →
            s1.readMemAs out2DType io.out2 (io.write2 s₀.pid + j.val)
              = out2DType.ofReal (R.round out2DType ((f xs ys zs).2 j)))
        ∧ (∀ r o,
            (r ≠ io.out1 ∨
              ∀ j : Fin io.B, io.mask s₀.pid j →
                o ≠ io.write1 s₀.pid + j.val) →
            (r ≠ io.out2 ∨
              ∀ j : Fin io.B, io.mask s₀.pid j →
                o ≠ io.write2 s₀.pid + j.val) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R out1DType out2DType f := by
  -- assemble the unified-core rounded triple once, then convert it back
  -- into the family statement; the flattening bridge lives in
  -- `UKernelIO.ImplementsR.intro`
  have hcore : (io.toU hout1 hout2).ImplementsR R
      (fun o => match o with
        | ⟨0, _⟩ => out1DType
        | ⟨_+1, _⟩ => out2DType)
      (fun _p₀ _p₁ vals o => match o with
        | ⟨0, _⟩ => fun j =>
            (f (fun j' => vals (⟨0, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨2, by decide⟩ : Fin 3) j')).1 j
        | ⟨_+1, _⟩ => fun j =>
            (f (fun j' => vals (⟨0, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
              (fun j' => vals (⟨2, by decide⟩ : Fin 3) j')).2 j) := by
    refine UKernelIO.ImplementsR.intro _ hok ?_ ?_
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

end VeriTile.Triton
