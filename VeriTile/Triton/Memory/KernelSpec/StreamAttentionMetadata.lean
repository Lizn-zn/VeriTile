/-
Kernel IO contracts: StreamAttentionMetadata.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.KernelSpec.Base

namespace VeriTile.Triton

/-! ### The metadata attention skin: `StreamMetaMasked3DKernelIO₃`

`Stream` + `Meta` + `Masked` + `3D` at data arity 3×1: the
`StreamMetaMasked3DKernelIO₂` slot mechanism widened to three streamed
float inputs — the varlen-attention shape (LightLLM context_attn: Q/K/V
streams whose windows, masks and trip counts are parameterized by loaded
per-batch scalars `seq_len`/`start_loc`).

**Launch legality.** Unlike every earlier skin, this genre's live step
count grows with a pid × slot product that has **no pid-free upper
bound** (`⌈…·min((pid+1)·BM+…, seq+…)/BN⌉`): at illegitimate pids or
slot values the kernel would execute live unpinned loads, so an
unguarded `∀ pids, ∀ m` triple is unprovable — and would claim nonsense.
The skin therefore carries a **launch-legality precondition** `pre`
(default `True`): the `⊨[R]` triple is stated for launches satisfying
`pre pid₀ pid₁ pid₂ m`, and consumers instantiate it with the real
host-grid facts (`pid < cdiv(max_input_len, BM)`,
`seq_len ≤ max_input_len`) that each port already documents as its
trusted-launch boundary. `T` is then the pid-free upper bound and the
slot-eating step masks gate the live steps. -/

/-- IO signature of the **metadata-parametrized three-stream fold**
shape (streaming genre, style S1 on a **3-D pid grid**): `nMeta`
pre-loop scalar metadata slots (pid-causal windows, per-slot `ChanTy`),
three streamed float inputs whose windows/masks eat the loaded slot
vector, a `T`-step fold and one terminal store — plus the
launch-legality precondition `pre` (see the section note). Intended
consumers: the varlen context-attention family (context_attn_nopad /
mistral / fwd). Slot causality and the single-surface design are
inherited verbatim from `StreamMetaMasked3DKernelIO₂`. -/
structure StreamMetaMasked3DKernelIO₃ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- First streamed input buffer. -/
  inp1 : RegionName
  /-- Second streamed input buffer. -/
  inp2 : RegionName
  /-- Third streamed input buffer. -/
  inp3 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Number of scalar metadata slots (a field, never a name subscript). -/
  nMeta : Nat
  /-- Slot `k`'s element type (read back through `(sty k).read`). -/
  sty : Fin nMeta → ChanTy
  /-- Slot `k`'s region (one cell read per program). -/
  mbuf : Fin nMeta → RegionName
  /-- Slot `k`'s cell address — a function of the pids only (slot
  causality as in `StreamMetaMasked3DKernelIO₂`). -/
  mwin : Fin nMeta → Nat → Nat → Nat → Nat
  /-- Number of streaming steps — the pid-free upper bound of the walk;
  the slot-eating masks gate the live steps (see the section note). -/
  T : Nat
  /-- Per-step tile length of the first input channel. -/
  B1 : Nat
  /-- Per-step tile length of the second input channel. -/
  B2 : Nat
  /-- Per-step tile length of the third input channel. -/
  B3 : Nat
  /-- Tile length of the terminal output store. -/
  C : Nat
  /-- The output buffer's floating dtype (`.real`, the default, is exact
  under `execR R`). -/
  outDType : FloatDType := .real
  /-- The launch-legality precondition (see the section note): the
  triple is claimed only for `(pids, slots)` satisfying `pre` — the
  host-grid facts of the port's documented trusted-launch boundary.
  Defaults to fully unconstrained. -/
  pre : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) → Prop :=
    fun _ _ _ _ => True
  /-- Step `t`, lane `j`'s `inp1` read address, given the loaded slots. -/
  read1 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s `inp2` read address. -/
  read2 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B2 → Nat
  /-- Step `t`, lane `j`'s `inp3` read address. -/
  read3 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B3 → Nat
  /-- Lane `j`'s terminal write address, given the loaded slots. -/
  write : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin C → Nat
  /-- `inp1`'s read-active lanes at step `t` (slot-eating). -/
  mask1 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B1 → Prop
  /-- `inp2`'s read-active lanes at step `t` (slot-eating — the
  sequence-length slot bounds the live steps). -/
  mask2 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B2 → Prop
  /-- `inp3`'s read-active lanes at step `t`. -/
  mask3 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B3 → Prop
  /-- The terminal store's write-active lanes (slot-eating). -/
  writeMask : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin C → Prop

namespace StreamMetaMasked3DKernelIO₃

/-- The pinned slot-value context (see
`StreamMetaMasked3DKernelIO₂.Meta`). -/
abbrev Meta (io : StreamMetaMasked3DKernelIO₃) : Type :=
  ∀ k : Fin io.nMeta, (io.sty k).carrier

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the metadata three-stream fold skin, guarded by the
launch-legality precondition: the full Hoare triple of
`StreamMetaMasked3DKernelIO₂.ImplementsR` widened to three streams, with
`io.pre pid₀ pid₁ pid₂ m` hypothesised before the window bounds — the
contract says nothing about illegitimate launches (the host-grid
boundary each consumer documents). The terminal cell holds the *ideal*
real value `f pid₀ pid₁ pid₂ m xs ys zs j`, quantized **once** at
`outDType`; frame outside the write-active window. -/
def ImplementsR (io : StreamMetaMasked3DKernelIO₃) (R : RoundingModel)
    (f : Nat → Nat → Nat → io.Meta → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = List.ofFn io.mbuf ++ [io.inp1, io.inp2, io.inp3, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (m : io.Meta) (xs : Fin io.T → Fin io.B1 → ℝ)
    (ys : Fin io.T → Fin io.B2 → ℝ) (zs : Fin io.T → Fin io.B3 → ℝ)
    (s₀ : BlockState),
    io.pre pid₀ pid₁ pid₂ m →
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ k : Fin io.nMeta,
      io.mwin k pid₀ pid₁ pid₂ < A.extent (io.mbuf k)) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ m t j →
      io.read1 pid₀ pid₁ pid₂ m t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ m t j →
      io.read2 pid₀ pid₁ pid₂ m t j < A.extent io.inp2) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ m t j →
      io.read3 pid₀ pid₁ pid₂ m t j < A.extent io.inp3) →
    (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ m j →
      io.write pid₀ pid₁ pid₂ m j < A.extent io.out) →
    (∀ k : Fin io.nMeta,
      (io.sty k).read s₀ (io.mbuf k) (io.mwin k pid₀ pid₁ pid₂) = m k) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ m t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ m t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ m t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ pid₂ m t j) = ys t j) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ m t j →
      s₀.readMem io.inp3 (io.read3 pid₀ pid₁ pid₂ m t j) = zs t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ m j →
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ pid₂ m j))
            = io.outDType.ofReal
                (R.round io.outDType (f pid₀ pid₁ pid₂ m xs ys zs j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ m j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ m j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamMetaMasked3DKernelIO₃.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, the three-stream
widening of `StreamMetaMasked3DKernelIO₂.ImplementsR.intro` with the
launch-legality precondition threaded into both obligations (`hts` and
`hrun` receive `io.pre` alongside the slot pins, so the consumer's
safety walk and invariant argument may assume a legal launch). `hrun` is
where the consumer runs its `forRange` invariant argument — the skin
does not prove the loop. -/
theorem ImplementsR.intro (io : StreamMetaMasked3DKernelIO₃)
    {R : RoundingModel}
    {f : Nat → Nat → Nat → io.Meta → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      Fin io.C → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m : io.Meta)
        (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
        (zs : Fin io.T → Fin io.B3 → ℝ),
      io.pre (s.pids 0) (s.pids 1) (s.pids 2) m →
      (∀ k : Fin io.nMeta,
        (io.sty k).read s (io.mbuf k)
          (io.mwin k (s.pids 0) (s.pids 1) (s.pids 2)) = m k) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        s.readMem io.inp1
            (io.read1 (s.pids 0) (s.pids 1) (s.pids 2) m t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        s.readMem io.inp2
            (io.read2 (s.pids 0) (s.pids 1) (s.pids 2) m t j) = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        s.readMem io.inp3
            (io.read3 (s.pids 0) (s.pids 1) (s.pids 2) m t j) = zs t j) →
      (∀ k : Fin io.nMeta,
        io.mwin k (s.pids 0) (s.pids 1) (s.pids 2) < bounds (io.mbuf k)) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.read1 (s.pids 0) (s.pids 1) (s.pids 2) m t j
          < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.read2 (s.pids 0) (s.pids 1) (s.pids 2) m t j
          < bounds io.inp2) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.read3 (s.pids 0) (s.pids 1) (s.pids 2) m t j
          < bounds io.inp3) →
      (∀ j : Fin io.C, io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) m j →
        io.write (s.pids 0) (s.pids 1) (s.pids 2) m j < bounds io.out) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (m : io.Meta)
        (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
        (zs : Fin io.T → Fin io.B3 → ℝ),
      io.pre (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m →
      s₀.undef = (fun _ _ => 0) →
      (∀ k : Fin io.nMeta,
        (io.sty k).read s₀ (io.mbuf k)
          (io.mwin k (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)) = m k) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
        s₀.readMem io.inp1
            (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j)
          = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
        s₀.readMem io.inp2
            (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j)
          = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
        s₀.readMem io.inp3
            (io.read3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j)
          = zs t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.C,
            io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j →
            s1.readMemAs io.outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j)
              = io.outDType.ofReal
                  (R.round io.outDType
                    (f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m xs ys zs j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.C,
                io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ pid₂ m xs ys zs s₀ hpre hpid₀ hpid₁ hpid₂
    hu hbm hbr1 hbr2 hbr3 hbw hm hx hy hz
  subst hpid₀
  subst hpid₁
  subst hpid₂
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ m xs ys zs hpre hu hm hx hy hz
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ m xs ys zs hpre hm hx hy hz hbm hbr1 hbr2 hbr3 hbw
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem : io.out ∈ A.regions := by
    rw [hregs]
    exact List.mem_append_right _ (by simp)
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j hj
    have hlt : io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j
        < A.extent io.out := hbw j hj
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

end StreamMetaMasked3DKernelIO₃

/-! ### The gather attention skins: `StreamMetaGatherMasked3DKernelIO₂/₃`

`Stream` + `Meta` + `Gather` + `Masked` + `3D`: the
`StreamMetaMasked3DKernelIO₂/₃` slot mechanism widened by a **per-step
gathered index channel** — the paged-KV shape (LightLLM
`Req_to_tokens`/`B_Loc` page tables: an integer tile loaded each step
whose *values* address the K/V loads).

**The gather mechanism.** A fifth kind of buffer `gbuf` (a `ChanTy`-typed
index region) is read at a per-step window `gread` that eats the pids and
the slot vector only — **one chain level**: the gather never addresses
itself. The gathered tile `G : Fin T → Fin Bg → gty.carrier` is
universally quantified in the triple beside the slot vector `m`, pinned
with **two legs** matching the masked-load `other=` semantics: on
`gmask`-active lanes the memory readback equals `G t j`; on inactive
lanes `G t j = gother` (the kernel's substituted default, an io-record
field — a host argument, so consumers may hypothesise bounds on it).
Data windows that consume the gathered value (`read2`, and `read3` on
the ₃ skin) eat `G` as an extra argument.

**`f` does not eat `G`.** The gathered index affects *addresses* only;
the data pins already deliver the gathered values as the ordinary
streams, so the spec function keeps the `Meta`-family signature. Only
the consumer's spec-equals-genuine bridge sees `G` (rewriting the
kernel's gather closed form to `G t j` via the pin).

Launch legality (`pre`), slot causality, and the single-surface design
are inherited verbatim from `StreamMetaMasked3DKernelIO₃`. -/

/-- IO signature of the **gather-indexed two-stream fold** shape
(streaming genre, style S1 on a **3-D pid grid**): `nMeta` pre-loop
scalar slots, a per-step gathered index channel (`gbuf`/`Bg`/`gread`/
`gmask`/`gother` — see the section note), two streamed float inputs of
which the second's window eats the gathered tile, a `T`-step fold and
one terminal store, guarded by `pre`. Intended consumer:
softmax_reducev (Logics/V + `B_Loc` page table; its V load is unmasked,
so `mask2 ≡ True` and the sentinel-substituted address must be bounded —
dischargeable because `gother` is an io-record field). -/
structure StreamMetaGatherMasked3DKernelIO₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- First streamed input buffer. -/
  inp1 : RegionName
  /-- Second streamed input buffer (its window eats the gathered tile). -/
  inp2 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Number of scalar metadata slots (a field, never a name subscript). -/
  nMeta : Nat
  /-- Slot `k`'s element type (read back through `(sty k).read`). -/
  sty : Fin nMeta → ChanTy
  /-- Slot `k`'s region (one cell read per program). -/
  mbuf : Fin nMeta → RegionName
  /-- Slot `k`'s cell address — a function of the pids only. -/
  mwin : Fin nMeta → Nat → Nat → Nat → Nat
  /-- The index (gather) buffer, read once per step. -/
  gbuf : RegionName
  /-- The gathered channel's element type (`.nat` for unsigned page
  tables; `.int` for signed). -/
  gty : ChanTy := ChanTy.nat
  /-- Per-step tile length of the gather channel. -/
  Bg : Nat
  /-- The `other=` default the kernel substitutes on `gmask`-inactive
  lanes (see the section note's two-leg pin). -/
  gother : gty.carrier
  /-- Number of streaming steps — the pid-free upper bound of the walk. -/
  T : Nat
  /-- Per-step tile length of the first input channel. -/
  B1 : Nat
  /-- Per-step tile length of the second input channel. -/
  B2 : Nat
  /-- Tile length of the terminal output store. -/
  C : Nat
  /-- The output buffer's floating dtype (`.real`, the default, is exact
  under `execR R`). -/
  outDType : FloatDType := .real
  /-- The launch-legality precondition (see the
  `StreamMetaMasked3DKernelIO₃` section note). -/
  pre : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) → Prop :=
    fun _ _ _ _ => True
  /-- Step `t`, lane `j`'s `gbuf` read address — eats pids and slots
  only (one chain level; never eats the gathered tile itself). -/
  gread : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin Bg → Nat
  /-- The gather channel's read-active lanes (slot-eating). -/
  gmask : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin Bg → Prop
  /-- Step `t`, lane `j`'s `inp1` read address, given the loaded slots. -/
  read1 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s `inp2` read address, given the loaded slots
  **and the gathered tile**. -/
  read2 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    (Fin T → Fin Bg → gty.carrier) → Fin T → Fin B2 → Nat
  /-- Lane `j`'s terminal write address, given the loaded slots. -/
  write : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin C → Nat
  /-- `inp1`'s read-active lanes at step `t` (slot-eating). -/
  mask1 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B1 → Prop
  /-- `inp2`'s read-active lanes at step `t` (slot-eating; `≡ True` for
  an unmasked gather-addressed load). -/
  mask2 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B2 → Prop
  /-- The terminal store's write-active lanes (slot-eating). -/
  writeMask : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin C → Prop

namespace StreamMetaGatherMasked3DKernelIO₂

/-- The pinned slot-value context. -/
abbrev Meta (io : StreamMetaGatherMasked3DKernelIO₂) : Type :=
  ∀ k : Fin io.nMeta, (io.sty k).carrier

/-- The pinned gathered-index tile. -/
abbrev Gather (io : StreamMetaGatherMasked3DKernelIO₂) : Type :=
  Fin io.T → Fin io.Bg → io.gty.carrier

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the gather-indexed two-stream fold skin: the
`StreamMetaMasked3DKernelIO₂` triple with the gathered tile `G`
quantified beside the slot vector and pinned with the two-leg gather pin
(section note), and the second data channel's window/bounds/pins eating
`G`. The terminal cell holds the *ideal* real value
`f pid₀ pid₁ pid₂ m xs ys j`, quantized **once** at `outDType`; frame
outside the write-active window. -/
def ImplementsR (io : StreamMetaGatherMasked3DKernelIO₂)
    (R : RoundingModel)
    (f : Nat → Nat → Nat → io.Meta → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = List.ofFn io.mbuf ++ [io.gbuf, io.inp1, io.inp2, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (m : io.Meta) (G : io.Gather) (xs : Fin io.T → Fin io.B1 → ℝ)
    (ys : Fin io.T → Fin io.B2 → ℝ) (s₀ : BlockState),
    io.pre pid₀ pid₁ pid₂ m →
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ k : Fin io.nMeta,
      io.mwin k pid₀ pid₁ pid₂ < A.extent (io.mbuf k)) →
    (∀ (t : Fin io.T) (j : Fin io.Bg), io.gmask pid₀ pid₁ pid₂ m t j →
      io.gread pid₀ pid₁ pid₂ m t j < A.extent io.gbuf) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ m t j →
      io.read1 pid₀ pid₁ pid₂ m t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ m t j →
      io.read2 pid₀ pid₁ pid₂ m G t j < A.extent io.inp2) →
    (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ m j →
      io.write pid₀ pid₁ pid₂ m j < A.extent io.out) →
    (∀ k : Fin io.nMeta,
      (io.sty k).read s₀ (io.mbuf k) (io.mwin k pid₀ pid₁ pid₂) = m k) →
    (∀ (t : Fin io.T) (j : Fin io.Bg), io.gmask pid₀ pid₁ pid₂ m t j →
      io.gty.read s₀ io.gbuf (io.gread pid₀ pid₁ pid₂ m t j) = G t j) →
    (∀ (t : Fin io.T) (j : Fin io.Bg), ¬ io.gmask pid₀ pid₁ pid₂ m t j →
      G t j = io.gother) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ m t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ m t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ m t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ pid₂ m G t j) = ys t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ m j →
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ pid₂ m j))
            = io.outDType.ofReal
                (R.round io.outDType (f pid₀ pid₁ pid₂ m xs ys j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ m j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ m j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamMetaGatherMasked3DKernelIO₂.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, the gather
widening of `StreamMetaMasked3DKernelIO₂`'s assembly: `hts` and `hrun`
receive the gathered tile `G`, its two-leg pin, and its window bound
alongside the slot context. `hrun` is where the consumer runs its
`forRange` invariant argument — the skin does not prove the loop. -/
theorem ImplementsR.intro (io : StreamMetaGatherMasked3DKernelIO₂)
    {R : RoundingModel}
    {f : Nat → Nat → Nat → io.Meta → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → Fin io.C → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m : io.Meta)
        (G : io.Gather) (xs : Fin io.T → Fin io.B1 → ℝ)
        (ys : Fin io.T → Fin io.B2 → ℝ),
      io.pre (s.pids 0) (s.pids 1) (s.pids 2) m →
      (∀ k : Fin io.nMeta,
        (io.sty k).read s (io.mbuf k)
          (io.mwin k (s.pids 0) (s.pids 1) (s.pids 2)) = m k) →
      (∀ (t : Fin io.T) (j : Fin io.Bg),
        io.gmask (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.gty.read s io.gbuf
            (io.gread (s.pids 0) (s.pids 1) (s.pids 2) m t j) = G t j) →
      (∀ (t : Fin io.T) (j : Fin io.Bg),
        ¬ io.gmask (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        G t j = io.gother) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        s.readMem io.inp1
            (io.read1 (s.pids 0) (s.pids 1) (s.pids 2) m t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        s.readMem io.inp2
            (io.read2 (s.pids 0) (s.pids 1) (s.pids 2) m G t j)
          = ys t j) →
      (∀ k : Fin io.nMeta,
        io.mwin k (s.pids 0) (s.pids 1) (s.pids 2) < bounds (io.mbuf k)) →
      (∀ (t : Fin io.T) (j : Fin io.Bg),
        io.gmask (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.gread (s.pids 0) (s.pids 1) (s.pids 2) m t j
          < bounds io.gbuf) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.read1 (s.pids 0) (s.pids 1) (s.pids 2) m t j
          < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.read2 (s.pids 0) (s.pids 1) (s.pids 2) m G t j
          < bounds io.inp2) →
      (∀ j : Fin io.C, io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) m j →
        io.write (s.pids 0) (s.pids 1) (s.pids 2) m j < bounds io.out) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (m : io.Meta) (G : io.Gather)
        (xs : Fin io.T → Fin io.B1 → ℝ)
        (ys : Fin io.T → Fin io.B2 → ℝ),
      io.pre (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m →
      s₀.undef = (fun _ _ => 0) →
      (∀ k : Fin io.nMeta,
        (io.sty k).read s₀ (io.mbuf k)
          (io.mwin k (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)) = m k) →
      (∀ (t : Fin io.T) (j : Fin io.Bg),
        io.gmask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
        io.gty.read s₀ io.gbuf
            (io.gread (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j)
          = G t j) →
      (∀ (t : Fin io.T) (j : Fin io.Bg),
        ¬ io.gmask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
        G t j = io.gother) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
        s₀.readMem io.inp1
            (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j)
          = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
        s₀.readMem io.inp2
            (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m G t j)
          = ys t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.C,
            io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j →
            s1.readMemAs io.outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j)
              = io.outDType.ofReal
                  (R.round io.outDType
                    (f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m xs ys j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.C,
                io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ pid₂ m G xs ys s₀ hpre hpid₀ hpid₁
    hpid₂ hu hbm hbrG hbr1 hbr2 hbw hm hg hgo hx hy
  subst hpid₀
  subst hpid₁
  subst hpid₂
  obtain ⟨s1, hexec, hval, hframe⟩ :=
    hrun s₀ m G xs ys hpre hu hm hg hgo hx hy
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ m G xs ys hpre hm hg hgo hx hy hbm hbrG hbr1 hbr2 hbw
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem : io.out ∈ A.regions := by
    rw [hregs]
    exact List.mem_append_right _ (by simp)
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j hj
    have hlt : io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j
        < A.extent io.out := hbw j hj
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

end StreamMetaGatherMasked3DKernelIO₂

/-- IO signature of the **gather-indexed three-stream fold** shape
(streaming genre, style S1 on a **3-D pid grid**): the
`StreamMetaGatherMasked3DKernelIO₂` mechanism at data arity 3×1 — the
paged-KV context-attention shape (Q static, K and V both addressed
through the same gathered page-table tile). `read2` **and** `read3` eat
the gathered tile; everything else is the verbatim
`StreamMetaMasked3DKernelIO₃` field. Intended consumers:
context_attn_llama / context_attn_bloom (`Req_to_tokens` page table). -/
structure StreamMetaGatherMasked3DKernelIO₃ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- First streamed input buffer (window does not eat the gather). -/
  inp1 : RegionName
  /-- Second streamed input buffer (gather-addressed). -/
  inp2 : RegionName
  /-- Third streamed input buffer (gather-addressed). -/
  inp3 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Number of scalar metadata slots (a field, never a name subscript). -/
  nMeta : Nat
  /-- Slot `k`'s element type (read back through `(sty k).read`). -/
  sty : Fin nMeta → ChanTy
  /-- Slot `k`'s region (one cell read per program). -/
  mbuf : Fin nMeta → RegionName
  /-- Slot `k`'s cell address — a function of the pids only. -/
  mwin : Fin nMeta → Nat → Nat → Nat → Nat
  /-- The index (gather) buffer, read once per step. -/
  gbuf : RegionName
  /-- The gathered channel's element type. -/
  gty : ChanTy := ChanTy.nat
  /-- Per-step tile length of the gather channel. -/
  Bg : Nat
  /-- The `other=` default substituted on `gmask`-inactive lanes. -/
  gother : gty.carrier
  /-- Number of streaming steps — the pid-free upper bound of the walk. -/
  T : Nat
  /-- Per-step tile length of the first input channel. -/
  B1 : Nat
  /-- Per-step tile length of the second input channel. -/
  B2 : Nat
  /-- Per-step tile length of the third input channel. -/
  B3 : Nat
  /-- Tile length of the terminal output store. -/
  C : Nat
  /-- The output buffer's floating dtype (`.real`, the default, is exact
  under `execR R`). -/
  outDType : FloatDType := .real
  /-- The launch-legality precondition (see the
  `StreamMetaMasked3DKernelIO₃` section note). -/
  pre : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) → Prop :=
    fun _ _ _ _ => True
  /-- Step `t`, lane `j`'s `gbuf` read address — eats pids and slots
  only (one chain level; never eats the gathered tile itself). -/
  gread : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin Bg → Nat
  /-- The gather channel's read-active lanes (slot-eating). -/
  gmask : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin Bg → Prop
  /-- Step `t`, lane `j`'s `inp1` read address, given the loaded slots. -/
  read1 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s `inp2` read address, given the loaded slots
  **and the gathered tile**. -/
  read2 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    (Fin T → Fin Bg → gty.carrier) → Fin T → Fin B2 → Nat
  /-- Step `t`, lane `j`'s `inp3` read address, given the loaded slots
  **and the gathered tile**. -/
  read3 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    (Fin T → Fin Bg → gty.carrier) → Fin T → Fin B3 → Nat
  /-- Lane `j`'s terminal write address, given the loaded slots. -/
  write : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin C → Nat
  /-- `inp1`'s read-active lanes at step `t` (slot-eating). -/
  mask1 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B1 → Prop
  /-- `inp2`'s read-active lanes at step `t` (slot-eating). -/
  mask2 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B2 → Prop
  /-- `inp3`'s read-active lanes at step `t` (slot-eating). -/
  mask3 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B3 → Prop
  /-- The terminal store's write-active lanes (slot-eating). -/
  writeMask : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin C → Prop

namespace StreamMetaGatherMasked3DKernelIO₃

/-- The pinned slot-value context. -/
abbrev Meta (io : StreamMetaGatherMasked3DKernelIO₃) : Type :=
  ∀ k : Fin io.nMeta, (io.sty k).carrier

/-- The pinned gathered-index tile. -/
abbrev Gather (io : StreamMetaGatherMasked3DKernelIO₃) : Type :=
  Fin io.T → Fin io.Bg → io.gty.carrier

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the gather-indexed three-stream fold skin: the
`StreamMetaMasked3DKernelIO₃` triple with the gathered tile `G`
quantified beside the slot vector, pinned with the two-leg gather pin
(section note), and the second and third data channels' windows/bounds/
pins eating `G`. The terminal cell holds the *ideal* real value
`f pid₀ pid₁ pid₂ m xs ys zs j`, quantized **once** at `outDType`;
frame outside the write-active window. -/
def ImplementsR (io : StreamMetaGatherMasked3DKernelIO₃)
    (R : RoundingModel)
    (f : Nat → Nat → Nat → io.Meta → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions
      = List.ofFn io.mbuf ++ [io.gbuf, io.inp1, io.inp2, io.inp3, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (m : io.Meta) (G : io.Gather) (xs : Fin io.T → Fin io.B1 → ℝ)
    (ys : Fin io.T → Fin io.B2 → ℝ) (zs : Fin io.T → Fin io.B3 → ℝ)
    (s₀ : BlockState),
    io.pre pid₀ pid₁ pid₂ m →
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ k : Fin io.nMeta,
      io.mwin k pid₀ pid₁ pid₂ < A.extent (io.mbuf k)) →
    (∀ (t : Fin io.T) (j : Fin io.Bg), io.gmask pid₀ pid₁ pid₂ m t j →
      io.gread pid₀ pid₁ pid₂ m t j < A.extent io.gbuf) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ m t j →
      io.read1 pid₀ pid₁ pid₂ m t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ m t j →
      io.read2 pid₀ pid₁ pid₂ m G t j < A.extent io.inp2) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ m t j →
      io.read3 pid₀ pid₁ pid₂ m G t j < A.extent io.inp3) →
    (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ m j →
      io.write pid₀ pid₁ pid₂ m j < A.extent io.out) →
    (∀ k : Fin io.nMeta,
      (io.sty k).read s₀ (io.mbuf k) (io.mwin k pid₀ pid₁ pid₂) = m k) →
    (∀ (t : Fin io.T) (j : Fin io.Bg), io.gmask pid₀ pid₁ pid₂ m t j →
      io.gty.read s₀ io.gbuf (io.gread pid₀ pid₁ pid₂ m t j) = G t j) →
    (∀ (t : Fin io.T) (j : Fin io.Bg), ¬ io.gmask pid₀ pid₁ pid₂ m t j →
      G t j = io.gother) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ m t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ m t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ m t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ pid₂ m G t j) = ys t j) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ m t j →
      s₀.readMem io.inp3 (io.read3 pid₀ pid₁ pid₂ m G t j) = zs t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ m j →
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ pid₂ m j))
            = io.outDType.ofReal
                (R.round io.outDType (f pid₀ pid₁ pid₂ m xs ys zs j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ m j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ m j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamMetaGatherMasked3DKernelIO₃.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, the gather
widening of `StreamMetaMasked3DKernelIO₃.ImplementsR.intro`: `hts` and
`hrun` receive the gathered tile `G`, its two-leg pin, and its window
bound alongside the slot context. `hrun` is where the consumer runs its
`forRange` invariant argument — the skin does not prove the loop. -/
theorem ImplementsR.intro (io : StreamMetaGatherMasked3DKernelIO₃)
    {R : RoundingModel}
    {f : Nat → Nat → Nat → io.Meta → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      Fin io.C → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m : io.Meta)
        (G : io.Gather) (xs : Fin io.T → Fin io.B1 → ℝ)
        (ys : Fin io.T → Fin io.B2 → ℝ) (zs : Fin io.T → Fin io.B3 → ℝ),
      io.pre (s.pids 0) (s.pids 1) (s.pids 2) m →
      (∀ k : Fin io.nMeta,
        (io.sty k).read s (io.mbuf k)
          (io.mwin k (s.pids 0) (s.pids 1) (s.pids 2)) = m k) →
      (∀ (t : Fin io.T) (j : Fin io.Bg),
        io.gmask (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.gty.read s io.gbuf
            (io.gread (s.pids 0) (s.pids 1) (s.pids 2) m t j) = G t j) →
      (∀ (t : Fin io.T) (j : Fin io.Bg),
        ¬ io.gmask (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        G t j = io.gother) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        s.readMem io.inp1
            (io.read1 (s.pids 0) (s.pids 1) (s.pids 2) m t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        s.readMem io.inp2
            (io.read2 (s.pids 0) (s.pids 1) (s.pids 2) m G t j)
          = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        s.readMem io.inp3
            (io.read3 (s.pids 0) (s.pids 1) (s.pids 2) m G t j)
          = zs t j) →
      (∀ k : Fin io.nMeta,
        io.mwin k (s.pids 0) (s.pids 1) (s.pids 2) < bounds (io.mbuf k)) →
      (∀ (t : Fin io.T) (j : Fin io.Bg),
        io.gmask (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.gread (s.pids 0) (s.pids 1) (s.pids 2) m t j
          < bounds io.gbuf) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.read1 (s.pids 0) (s.pids 1) (s.pids 2) m t j
          < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.read2 (s.pids 0) (s.pids 1) (s.pids 2) m G t j
          < bounds io.inp2) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.read3 (s.pids 0) (s.pids 1) (s.pids 2) m G t j
          < bounds io.inp3) →
      (∀ j : Fin io.C, io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) m j →
        io.write (s.pids 0) (s.pids 1) (s.pids 2) m j < bounds io.out) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (m : io.Meta) (G : io.Gather)
        (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
        (zs : Fin io.T → Fin io.B3 → ℝ),
      io.pre (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m →
      s₀.undef = (fun _ _ => 0) →
      (∀ k : Fin io.nMeta,
        (io.sty k).read s₀ (io.mbuf k)
          (io.mwin k (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)) = m k) →
      (∀ (t : Fin io.T) (j : Fin io.Bg),
        io.gmask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
        io.gty.read s₀ io.gbuf
            (io.gread (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j)
          = G t j) →
      (∀ (t : Fin io.T) (j : Fin io.Bg),
        ¬ io.gmask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
        G t j = io.gother) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
        s₀.readMem io.inp1
            (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j)
          = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
        s₀.readMem io.inp2
            (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m G t j)
          = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
        s₀.readMem io.inp3
            (io.read3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m G t j)
          = zs t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.C,
            io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j →
            s1.readMemAs io.outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j)
              = io.outDType.ofReal
                  (R.round io.outDType
                    (f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m xs ys zs j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.C,
                io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ pid₂ m G xs ys zs s₀ hpre hpid₀ hpid₁
    hpid₂ hu hbm hbrG hbr1 hbr2 hbr3 hbw hm hg hgo hx hy hz
  subst hpid₀
  subst hpid₁
  subst hpid₂
  obtain ⟨s1, hexec, hval, hframe⟩ :=
    hrun s₀ m G xs ys zs hpre hu hm hg hgo hx hy hz
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ m G xs ys zs hpre hm hg hgo hx hy hz
      hbm hbrG hbr1 hbr2 hbr3 hbw
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem : io.out ∈ A.regions := by
    rw [hregs]
    exact List.mem_append_right _ (by simp)
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j hj
    have hlt : io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j
        < A.extent io.out := hbw j hj
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

end StreamMetaGatherMasked3DKernelIO₃

end VeriTile.Triton
