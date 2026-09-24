/-
Kernel IO contracts: BooleanMasked.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.KernelSpec.Base

namespace VeriTile.Triton

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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  projection := io.projection
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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  projection := io.projection
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

end VeriTile.Triton
