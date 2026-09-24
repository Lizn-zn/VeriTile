/-
Kernel IO contracts: Metadata.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.KernelSpec.Base

namespace VeriTile.Triton

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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  projection := io.projection
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

/-- `io.ImplementsR R outDType f` — the **rounding-correctness** relation
for the metadata-slot masked one-input / one-output family, written
`io ⊨[R, outDType] f`. Verbatim `Implements`, with two changes: the kernel
runs under the rounding model (`execR R`), and each **write-active** lane of
the output window is read back as an `outDType`-typed cell holding the ideal
real value quantized **once**,
`outDType.ofReal (R.round outDType (f pid₀ pid₁ m₁ m₂ xs j))`. The two
`.nat` slots stay exact — they are *input* channels, read through
`readMemValue .nat`, and no rounding grid applies to an integer load — and
so does the float data tile; the rounding model acts at the kernel's
cast/store sites, not at loads. Everything else (the allocation contract,
the slot and window in-bounds obligations, the slot pins, the `pid`/`undef`
pins, the frame) is unchanged; at `outDType := .real` the store is exact and
this degenerates to the exact surface.

The grid is an **argument**, and the notation names it, exactly as in
`MaskedKernelIO₁`: this skin's consumers straddle two storage precisions —
the token-softmax pair (`token_softmax_llama`, `token_softmax_bloom`) writes
wide fp32 attention logits, while the KV-cache copy genre
(`destindex_copy_kv1`/`2`, the `quantize_*` transforms) stores into
half-precision cache buffers. Both a `.fp16` face and a `.real` face are
therefore live for the same signature, and a three-hole `io ⊨[R] f` would
print them identically. -/
def ImplementsR (io : MetaMasked2DKernelIO₁) (R : RoundingModel)
    (outDType : FloatDType)
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
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask pid₀ pid₁ m₁ m₂ j →
          s'.readMemAs outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ m₁ m₂ j))
            = outDType.ofReal (R.round outDType (f pid₀ pid₁ m₁ m₂ xs j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.B, io.writeMask pid₀ pid₁ m₁ m₂ j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ m₁ m₂ j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R ", " outDType "] " f =>
  MetaMasked2DKernelIO₁.ImplementsR io R outDType f

/-- Assembly lemma for `⊨[R, outDType]` — the rounding sibling of
`MetaMasked2DKernelIO₁.Implements.intro`, riding the single-shot family's
rounding core `UKernelIO.ImplementsR.intro` through the same `toU`
embedding (at the constant output grid `fun _ => outDType`). Obligations in
the skin's named vocabulary as there, with the safety walk at
`Kernel.TraceSafeR R` and `hrun` returning a rounded region-model triple:
termination under `execR R`, the `readMemAs outDType` per-lane readback, and
the frame. -/
theorem ImplementsR.intro (io : MetaMasked2DKernelIO₁) {R : RoundingModel}
    {outDType : FloatDType}
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
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (m₁ m₂ : Nat) (xs : Fin io.B → ℝ),
      s₀.readMemValue .nat io.mbuf1 (io.mwin1 (s₀.pids 0) (s₀.pids 1)) = m₁ →
      s₀.readMemValue .nat io.mbuf2 (io.mwin2 (s₀.pids 0) (s₀.pids 1)) = m₂ →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
        s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) m₁ m₂ j) = xs j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀
          = some s1
        ∧ (∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
            s1.readMemAs outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) m₁ m₂ j)
              = outDType.ofReal
                  (R.round outDType (f (s₀.pids 0) (s₀.pids 1) m₁ m₂ xs j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨ ∀ j : Fin io.B,
              io.writeMask (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) m₁ m₂ j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R outDType f := by
  -- assemble the unified-core rounded triple once, then convert it back
  -- into the family statement; the flattening bridge lives in
  -- `UKernelIO.ImplementsR.intro`
  have hcore : io.toU.ImplementsR R (fun _ => outDType)
      (fun p₀ p₁ vals _o j => f p₀ p₁
        (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
        (fun j' => vals (⟨2, by decide⟩ : Fin 3) j') j) := by
    refine UKernelIO.ImplementsR.intro _ hok ?_ ?_
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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  projection := io.projection
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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  projection := io.projection
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

/-- `io.ImplementsR R out1DType out2DType out3DType f` — the
**rounding-correctness** relation for the cross-entropy metadata family,
written `io ⊨[R, out1DType, out2DType, out3DType] f`. Verbatim
`Implements`, with two changes: the kernel runs under the rounding model
(`execR R`), and each **gated** output cell is read back as a typed cell
holding the ideal real value quantized **once** — `out1DType` for `out1`,
`out2DType` for `out2`, `out3DType` for `out3`. The `.int` label slot stays
exact — it is an *input* channel, read through `readMemValue .int`, and no
rounding grid applies to an integer load — and so do the data row and the
gather cell; the rounding model acts at the kernel's cast/store sites, not
at loads. Everything else (the allocation contract, the six in-bounds
obligations, the label pin, the gather gate, the `pid`/`undef` pins, the
frame) is unchanged; at all three grids `.real` the stores are exact and
this degenerates to the exact surface.

The grids are **arguments** and there is **one per output cell**, all three
named by the notation. The three cells are independent caller-allocated
buffers — the fused cross-entropy forwards (`cross_entropy2`,
`cross_entropy_ops`) keep the LSE wide because the backward pass reloads it,
while the loss (and the `SPLIT`-gated z-loss) tensor is commonly allocated
at the logits' own precision — so their storage grids are genuinely
independent, and a single scalar grid reused across them would be the wrong
shape rather than merely a coarse one. -/
def ImplementsR (io : MetaMasked2DKernelIO₂ₓ₃) (R : RoundingModel)
    (out1DType out2DType out3DType : FloatDType)
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
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (io.writeMask1 pid₀ pid₁ lab →
          s'.readMemAs out1DType A.flat
              (A.addr io.out1 (io.write1 pid₀ pid₁ lab))
            = out1DType.ofReal
                (R.round out1DType (f pid₀ pid₁ lab xs g).1))
      ∧ (io.writeMask2 pid₀ pid₁ lab →
          s'.readMemAs out2DType A.flat
              (A.addr io.out2 (io.write2 pid₀ pid₁ lab))
            = out2DType.ofReal
                (R.round out2DType (f pid₀ pid₁ lab xs g).2.1))
      ∧ (io.writeMask3 pid₀ pid₁ lab →
          s'.readMemAs out3DType A.flat
              (A.addr io.out3 (io.write3 pid₀ pid₁ lab))
            = out3DType.ofReal
                (R.round out3DType (f pid₀ pid₁ lab xs g).2.2))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((io.writeMask1 pid₀ pid₁ lab →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ lab)) ∧
             (io.writeMask2 pid₀ pid₁ lab →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ lab)) ∧
             (io.writeMask3 pid₀ pid₁ lab →
                o' ≠ A.addr io.out3 (io.write3 pid₀ pid₁ lab)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R ", " out1DType ", "
    out2DType ", " out3DType "] " f =>
  MetaMasked2DKernelIO₂ₓ₃.ImplementsR io R out1DType out2DType out3DType f

/-- Assembly lemma for `⊨[R, out1DType, out2DType, out3DType]` — the
rounding sibling of `MetaMasked2DKernelIO₂ₓ₃.Implements.intro`, riding the
single-shot family's rounding core `UKernelIO.ImplementsR.intro` through the
same `toU` embedding (at the per-cell output grid
`out1DType`/`out2DType`/`out3DType`). Obligations in the skin's named
vocabulary as there — the label enters `hts`/`hrun` as one pinned named
`Int` scalar, the gather pin and bound are gated by `gmask` — with the
safety walk at `Kernel.TraceSafeR R` and `hrun` returning a rounded
region-model triple: termination under `execR R`, the three `readMemAs`
cell readbacks, and the frame. -/
theorem ImplementsR.intro (io : MetaMasked2DKernelIO₂ₓ₃) {R : RoundingModel}
    {out1DType out2DType out3DType : FloatDType}
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
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (lab : Int) (xs : Fin io.B → ℝ) (g : ℝ),
      s₀.readMemValue .int io.mbufL (io.mwinL (s₀.pids 0) (s₀.pids 1)) = lab →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) lab j →
        s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) lab j) = xs j) →
      (io.gmask (s₀.pids 0) (s₀.pids 1) lab →
        s₀.readMem io.inp (io.gwin (s₀.pids 0) (s₀.pids 1) lab) = g) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀
          = some s1
        ∧ (io.writeMask1 (s₀.pids 0) (s₀.pids 1) lab →
            s1.readMemAs out1DType io.out1
                (io.write1 (s₀.pids 0) (s₀.pids 1) lab)
              = out1DType.ofReal
                  (R.round out1DType
                    (f (s₀.pids 0) (s₀.pids 1) lab xs g).1))
        ∧ (io.writeMask2 (s₀.pids 0) (s₀.pids 1) lab →
            s1.readMemAs out2DType io.out2
                (io.write2 (s₀.pids 0) (s₀.pids 1) lab)
              = out2DType.ofReal
                  (R.round out2DType
                    (f (s₀.pids 0) (s₀.pids 1) lab xs g).2.1))
        ∧ (io.writeMask3 (s₀.pids 0) (s₀.pids 1) lab →
            s1.readMemAs out3DType io.out3
                (io.write3 (s₀.pids 0) (s₀.pids 1) lab)
              = out3DType.ofReal
                  (R.round out3DType
                    (f (s₀.pids 0) (s₀.pids 1) lab xs g).2.2))
        ∧ (∀ r o,
            (r ≠ io.out1 ∨ (io.writeMask1 (s₀.pids 0) (s₀.pids 1) lab →
              o ≠ io.write1 (s₀.pids 0) (s₀.pids 1) lab)) →
            (r ≠ io.out2 ∨ (io.writeMask2 (s₀.pids 0) (s₀.pids 1) lab →
              o ≠ io.write2 (s₀.pids 0) (s₀.pids 1) lab)) →
            (r ≠ io.out3 ∨ (io.writeMask3 (s₀.pids 0) (s₀.pids 1) lab →
              o ≠ io.write3 (s₀.pids 0) (s₀.pids 1) lab)) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R out1DType out2DType out3DType f := by
  -- assemble the unified-core rounded triple once, then convert it back
  -- into the family statement; the flattening bridge lives in
  -- `UKernelIO.ImplementsR.intro`
  have hcore : io.toU.ImplementsR R
      (fun o => match o with
        | ⟨0, _⟩ => out1DType
        | ⟨1, _⟩ => out2DType
        | ⟨_+2, _⟩ => out3DType)
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
    refine UKernelIO.ImplementsR.intro _ hok ?_ ?_
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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  projection := io.projection
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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  projection := io.projection
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

end VeriTile.Triton
