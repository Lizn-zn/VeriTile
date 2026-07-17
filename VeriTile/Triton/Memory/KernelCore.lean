/-
VeriTile.Triton.Memory.KernelCore

The **unified KernelIO core**: one vectorized, channel-typed IO signature
(`UKernelIO`) whose single assembly lemma (`UKernelIO.Implements.intro`)
carries the flat-memory bridge for the *entire* concrete KernelIO family.

The concrete structs in `Memory/KernelSpec.lean` (`KernelIO₂`, the
`Masked*`/`Masked2D*` variants, …) remain the public, named-field statement
surface — their `Implements` definitions are untouched. What changes is the
*proof architecture*: each concrete `Implements.intro` is a corollary of the
core's, obtained by embedding the struct into `UKernelIO` (a `toU` value)
and converting the core triple back into the struct's vocabulary. The
flattening bridge (`exec_flatten` + `flattenState_readMem` + `decode_sound`)
is therefore proved **once**, here.

Design:

* **Channels.** Every input is a typed channel (`ChanTy`: float / bool /
  nat / int). Float channels pin values through the family's existing
  `readMem` convention; typed channels through the operational
  `readMemValue` view that `tl.load` itself uses.
* **Arities.** Each channel has its own lane count (`iarity`/`oarity`/
  `sarity`). A scalar metadata slot is a 1-lane channel; the concrete
  family's uniform-`B` tiles set every arity to `B`.
* **Value-dependent windows.** Every window/mask takes the full context
  `vals` of pinned input values, so a data window may depend on a loaded
  scalar — the data-dependent genre (LightLLM `start_loc`/`seq_len`
  metadata, cross_entropy labels, gather/scatter indices).
* **Simultaneous pinning.** `Implements` quantifies one `vals : io.Ctx` and
  pins all channels at once. Instances keep this non-vacuous by using
  *causal* windows (each channel's window depends only on earlier
  channels' values), so every launch state determines a unique context —
  the concrete skins' shapes (static data windows + scalar slots with
  pid-affine cells) are causal by construction.
* **Allocation list.** `bufs` (the region list handed to the allocator) is
  decoupled from the channel roles, so in-place kernels (`out = in2`) keep
  the `MaskedKernelIO₃ₓ₂` duplicate-region wiring.
* **Scratch.** Private working buffers are value-contract-free output-like
  channels: allocated, writable, excluded from the frame, absent from the
  postcondition.
-/
import VeriTile.Triton.Memory.Denotation

namespace VeriTile.Triton

/-! ## Channel types -/

/-- Element type of a `UKernelIO` input channel. -/
inductive ChanTy where
  | float
  | bool
  | nat
  | int
  deriving DecidableEq, Repr

namespace ChanTy

/-- Mathematical carrier of each channel type. -/
abbrev carrier : ChanTy → Type
  | .float => ℝ
  | .bool  => Bool
  | .nat   => Nat
  | .int   => Int

/-- Channel read: what a launch state holds at a region/address on this
channel. Float channels use the finite-fallback `readMem` (the family's
existing ℝ input convention); typed channels use the operational
`readMemValue` view that `tl.load` itself uses. -/
def read : (t : ChanTy) → BlockState → RegionName → Nat → t.carrier
  | .float => fun s r a => s.readMem r a
  | .bool  => fun s r a => s.readMemValue .bool r a
  | .nat   => fun s r a => s.readMemValue .nat r a
  | .int   => fun s r a => s.readMemValue .int r a

end ChanTy

/-! ## The core signature -/

/-- The unified, vectorized KernelIO signature. Arities and channel types
are *fields* (never part of a name); every window and mask takes the full
pinned-input context, so windows may depend on loaded values. This struct
is proof infrastructure — headline statements live on the concrete
named-field family in `Memory/KernelSpec.lean`. -/
structure UKernelIO where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- Number of input channels. -/
  nIn : Nat
  /-- Number of output channels. -/
  nOut : Nat
  /-- Number of scratch (private working) channels. -/
  nScr : Nat
  /-- The allocation list: exactly the regions handed to the allocator
  (`A.regions`). Decoupled from the channel roles so duplicate-region
  (in-place) wiring is expressible. -/
  bufs : List RegionName
  /-- Each input channel's element type. -/
  ity : Fin nIn → ChanTy
  /-- Each input channel's lane count (scalar slots are 1-lane). -/
  iarity : Fin nIn → Nat
  /-- Each input channel's buffer. -/
  ibuf : Fin nIn → RegionName
  /-- Each output channel's lane count. -/
  oarity : Fin nOut → Nat
  /-- Each output channel's buffer. -/
  obuf : Fin nOut → RegionName
  /-- Output buffers are allocated (needed to read the postcondition back
  through the flattening). -/
  obuf_mem : ∀ o, obuf o ∈ bufs
  /-- Each scratch channel's lane count. -/
  sarity : Fin nScr → Nat
  /-- Each scratch channel's buffer. -/
  sbuf : Fin nScr → RegionName
  /-- Input channel `i`'s per-lane read address — may depend on the pinned
  values of every channel (the data-dependent genre). -/
  iwin : (i : Fin nIn) → (∀ k, Fin (iarity k) → (ity k).carrier) →
    Nat → Nat → Fin (iarity i) → Nat
  /-- Input channel `i`'s read-active lanes. -/
  imask : (i : Fin nIn) → (∀ k, Fin (iarity k) → (ity k).carrier) →
    Nat → Nat → Fin (iarity i) → Prop
  /-- Output channel `o`'s per-lane write address. -/
  owin : (o : Fin nOut) → (∀ k, Fin (iarity k) → (ity k).carrier) →
    Nat → Nat → Fin (oarity o) → Nat
  /-- Output channel `o`'s write-active lanes. -/
  omask : (o : Fin nOut) → (∀ k, Fin (iarity k) → (ity k).carrier) →
    Nat → Nat → Fin (oarity o) → Prop
  /-- Scratch channel `t`'s per-lane write address. -/
  swin : (t : Fin nScr) → (∀ k, Fin (iarity k) → (ity k).carrier) →
    Nat → Nat → Fin (sarity t) → Nat
  /-- Scratch channel `t`'s write-active lanes. -/
  smask : (t : Fin nScr) → (∀ k, Fin (iarity k) → (ity k).carrier) →
    Nat → Nat → Fin (sarity t) → Prop

namespace UKernelIO

/-- The pinned-input context: one carrier value per channel per lane. -/
abbrev Ctx (io : UKernelIO) : Type :=
  ∀ k : Fin io.nIn, Fin (io.iarity k) → (io.ity k).carrier

/-- `io.Implements f` — the channel-typed masked Hoare triple. For any
context `vals` the launch state holds on every input channel (channel-typed,
at the `vals`-dependent windows), the translated kernel terminates, every
write-active lane of every output channel holds `f … vals o j`, and every
flat cell outside the declared output/scratch windows is unchanged. -/
def Implements (io : UKernelIO)
    (f : Nat → Nat → io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → ℝ) :
    Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = io.bufs →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (vals : io.Ctx) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (i : Fin io.nIn) (j : Fin (io.iarity i)), io.imask i vals pid₀ pid₁ j →
      io.iwin i vals pid₀ pid₁ j < A.extent (io.ibuf i)) →
    (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)), io.omask o vals pid₀ pid₁ j →
      io.owin o vals pid₀ pid₁ j < A.extent (io.obuf o)) →
    (∀ (t : Fin io.nScr) (j : Fin (io.sarity t)), io.smask t vals pid₀ pid₁ j →
      io.swin t vals pid₀ pid₁ j < A.extent (io.sbuf t)) →
    (∀ (i : Fin io.nIn) (j : Fin (io.iarity i)), io.imask i vals pid₀ pid₁ j →
      (io.ity i).read s₀ (io.ibuf i) (io.iwin i vals pid₀ pid₁ j) = vals i j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
          io.omask o vals pid₀ pid₁ j →
          s'.readMem A.flat (A.addr (io.obuf o) (io.owin o vals pid₀ pid₁ j))
            = f pid₀ pid₁ vals o j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
                io.omask o vals pid₀ pid₁ j →
                o' ≠ A.addr (io.obuf o) (io.owin o vals pid₀ pid₁ j)) ∧
             (∀ (t : Fin io.nScr) (j : Fin (io.sarity t)),
                io.smask t vals pid₀ pid₁ j →
                o' ≠ A.addr (io.sbuf t) (io.swin t vals pid₀ pid₁ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ᵤ " => UKernelIO.Implements

/-- Assembly lemma — **the** flattening-bridge proof of the whole family.
The obligations are the family's usual three, with the channel-typed pin
hypothesis threaded through both `hts` and `hrun` (value-dependent windows
need the pinned values to state their bounds). -/
theorem Implements.intro (io : UKernelIO)
    {f : Nat → Nat → io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (vals : io.Ctx),
      (∀ (i : Fin io.nIn) (j : Fin (io.iarity i)),
        io.imask i vals (s.pids 0) (s.pids 1) j →
        (io.ity i).read s (io.ibuf i) (io.iwin i vals (s.pids 0) (s.pids 1) j)
          = vals i j) →
      (∀ (i : Fin io.nIn) (j : Fin (io.iarity i)),
        io.imask i vals (s.pids 0) (s.pids 1) j →
        io.iwin i vals (s.pids 0) (s.pids 1) j < bounds (io.ibuf i)) →
      (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
        io.omask o vals (s.pids 0) (s.pids 1) j →
        io.owin o vals (s.pids 0) (s.pids 1) j < bounds (io.obuf o)) →
      (∀ (t : Fin io.nScr) (j : Fin (io.sarity t)),
        io.smask t vals (s.pids 0) (s.pids 1) j →
        io.swin t vals (s.pids 0) (s.pids 1) j < bounds (io.sbuf t)) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (vals : io.Ctx),
      (∀ (i : Fin io.nIn) (j : Fin (io.iarity i)),
        io.imask i vals (s₀.pids 0) (s₀.pids 1) j →
        (io.ity i).read s₀ (io.ibuf i) (io.iwin i vals (s₀.pids 0) (s₀.pids 1) j)
          = vals i j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
            io.omask o vals (s₀.pids 0) (s₀.pids 1) j →
            s1.readMem (io.obuf o) (io.owin o vals (s₀.pids 0) (s₀.pids 1) j)
              = f (s₀.pids 0) (s₀.pids 1) vals o j)
        ∧ (∀ r o',
            (∀ (oc : Fin io.nOut) (j : Fin (io.oarity oc)),
              io.omask oc vals (s₀.pids 0) (s₀.pids 1) j →
              r ≠ io.obuf oc ∨ o' ≠ io.owin oc vals (s₀.pids 0) (s₀.pids 1) j) →
            (∀ (t : Fin io.nScr) (j : Fin (io.sarity t)),
              io.smask t vals (s₀.pids 0) (s₀.pids 1) j →
              r ≠ io.sbuf t ∨ o' ≠ io.swin t vals (s₀.pids 0) (s₀.pids 1) j) →
            s1.mem r o' = s₀.mem r o')) :
    io.Implements f := by
  intro A hd hregs hcov pid₀ pid₁ vals s₀ hpid₀ hpid₁ hu hib hob hsb hpins
  subst hpid₀
  subst hpid₁
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ vals hpins
  have hts' : Kernel.TraceSafe A.extent (io.kernel.toAlgKernel) s₀ :=
    hts A.extent s₀ vals hpins hib hob hsb
  have hbridge := A.exec_flatten hd hcov _ s₀ hts' hok hu
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro o j hj
    have hmem : io.obuf o ∈ A.regions := by
      rw [hregs]; exact io.obuf_mem o
    have hlt : io.owin o vals (s₀.pids 0) (s₀.pids 1) j
        < A.extent (io.obuf o) := hob o j hj
    rw [A.flattenState_readMem hd s1 hmem hlt]
    exact hval o j hj
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
          refine congrArg A.trCell
            (hframe r o (fun oc j hj => ?_) (fun t j hj => ?_))
          · by_cases hro : r = io.obuf oc
            · subst hro
              refine Or.inr fun hoj => ?_
              rcases hcond with hflat | hn
              · exact hflat rfl
              · exact hn.1 oc j hj (by rw [hoeq, hoj])
            · exact Or.inl hro
          · by_cases hrs : r = io.sbuf t
            · subst hrs
              refine Or.inr fun hoj => ?_
              rcases hcond with hflat | hn
              · exact hflat rfl
              · exact hn.2 t j hj (by rw [hoeq, hoj])
            · exact Or.inl hrs
    · simp only [FlatAlloc.flattenState, if_neg hr]

end UKernelIO

end VeriTile.Triton
