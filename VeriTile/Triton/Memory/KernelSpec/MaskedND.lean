/-
Kernel IO contracts: MaskedND.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.KernelSpec.Base

namespace VeriTile.Triton

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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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

/-- `io.ImplementsR R outDType f` — the **rounding-correctness** relation
for the two-axis one-input family, written `io ⊨[R, outDType] f`. Verbatim
`Implements`, with two changes: the kernel runs under the rounding model
(`execR R`), and each **write-active** lane of the output window is read
back as an `outDType`-typed cell holding the ideal real value quantized
**once**, `outDType.ofReal (R.round outDType (f pid₀ pid₁ xs j))`. Inputs
stay exact ℝ — the rounding model acts at the kernel's cast/store sites,
not at loads. Everything else (the allocation contract, the three
in-bounds obligations, the scratch channels, the two `pid` pins and the
`undef` pin, the frame) is unchanged; at `outDType := .real` the store is
exact and this degenerates to the exact surface.

The grid is an **argument**, and the notation names it. Narrowing is not
hypothetical here: the destination-index cache genre
(`destindex_copy_kv1`/`kv2`, `quantize_copy_kv`, `quantize_kv_transform`)
copies into a half-precision cache, while the reduction consumers
(`logsumexp_fwd`, `mean_reduction`, `ksoftmax_triton`) keep a wide
destination. Both faces are live for the same signature, so a three-hole
`io ⊨[R] f` would print two different quantization grids identically. -/
def ImplementsR (io : Masked2DKernelIO₁) (R : RoundingModel)
    (outDType : FloatDType)
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
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask pid₀ pid₁ j →
          s'.readMemAs outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ j))
            = outDType.ofReal (R.round outDType (f pid₀ pid₁ xs j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.writeMask pid₀ pid₁ j →
                o' ≠ A.addr io.out (io.write pid₀ pid₁ j)) ∧
             (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.writeMask pid₀ pid₁ j →
                o' ≠ A.addr p.1 (p.2 pid₀ pid₁ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R ", " outDType "] " f =>
  Masked2DKernelIO₁.ImplementsR io R outDType f

/-- Assembly lemma for `⊨[R, outDType]` — the rounding sibling of
`Masked2DKernelIO₁.Implements.intro`, riding the single-shot family's
rounding core `UKernelIO.ImplementsR.intro` through the same `toU`
embedding (at the constant output grid `fun _ => outDType`). The
obligations are the family's usual three, with the safety walk at
`Kernel.TraceSafeR R` and `hrun` returning a rounded region-model triple:
termination under `execR R`, the `readMemAs outDType` per-lane readback,
and the frame. -/
theorem ImplementsR.intro (io : Masked2DKernelIO₁) {R : RoundingModel}
    {outDType : FloatDType}
    {f : Nat → Nat → (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.read (s.pids 0) (s.pids 1) j < bounds io.inp) →
      (∀ j : Fin io.B, io.writeMask (s.pids 0) (s.pids 1) j →
        io.write (s.pids 0) (s.pids 1) j < bounds io.out) →
      (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.writeMask (s.pids 0) (s.pids 1) j →
        p.2 (s.pids 0) (s.pids 1) j < bounds p.1) →
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) j) = xs j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀
          = some s1
        ∧ (∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
            s1.readMemAs outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) j)
              = outDType.ofReal
                  (R.round outDType (f (s₀.pids 0) (s₀.pids 1) xs j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) j) →
            (∀ p ∈ io.scratch, r = p.1 →
              ∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
                o ≠ p.2 (s₀.pids 0) (s₀.pids 1) j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R outDType f := by
  -- assemble the unified-core rounded triple once, then convert it back
  -- into the family statement; the flattening bridge lives in
  -- `UKernelIO.ImplementsR.intro`
  have hcore : io.toU.ImplementsR R (fun _ => outDType)
      (fun p₀ p₁ vals _o j => f p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 1) j') j) := by
    refine UKernelIO.ImplementsR.intro _ hok ?_ ?_
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

/-- `io.ImplementsR R outDType f` — the **rounding-correctness** relation
for the two-axis two-input family, written `io ⊨[R, outDType] f`. Verbatim
`Implements`, with two changes: the kernel runs under the rounding model
(`execR R`), and each **write-active** lane of the output window is read
back as an `outDType`-typed cell holding the ideal real value quantized
**once**, `outDType.ofReal (R.round outDType (f pid₀ pid₁ xs ys j))`.
Inputs stay exact ℝ — the rounding model acts at the kernel's cast/store
sites, not at loads. Everything else (the allocation contract, the four
in-bounds obligations, the per-channel read gates, the scratch channels,
the two `pid` pins and the `undef` pin, the frame) is unchanged; at
`outDType := .real` the store is exact and this degenerates to the exact
surface.

The grid is an **argument**, and the notation names it. This skin carries
the file's densest population of *genuinely* narrowing stores — the
cache-copy and dequantize genre (`kv_cache_copy`, `kcache_copy_triton`,
`cache_transform_triton`, `dequantize_rowwise`, `dequantize_matmul`)
writes a half-precision destination while the row-statistic consumers
(`fast_rms_layernorm`, `cross_entropy*`) keep a wide one. Both faces are
live for the same signature, so a three-hole `io ⊨[R] f` would print two
different quantization grids identically. -/
def ImplementsR (io : Masked2DKernelIO₂) (R : RoundingModel)
    (outDType : FloatDType)
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
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask pid₀ pid₁ j →
          s'.readMemAs outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ j))
            = outDType.ofReal (R.round outDType (f pid₀ pid₁ xs ys j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.writeMask pid₀ pid₁ j →
                o' ≠ A.addr io.out (io.write pid₀ pid₁ j)) ∧
             (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.writeMask pid₀ pid₁ j →
                o' ≠ A.addr p.1 (p.2 pid₀ pid₁ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R ", " outDType "] " f =>
  Masked2DKernelIO₂.ImplementsR io R outDType f

/-- Assembly lemma for `⊨[R, outDType]` — the rounding sibling of
`Masked2DKernelIO₂.Implements.intro`, riding the single-shot family's
rounding core `UKernelIO.ImplementsR.intro` through the same `toU`
embedding (at the constant output grid `fun _ => outDType`). The
obligations are the family's usual three, with the safety walk at
`Kernel.TraceSafeR R` and `hrun` returning a rounded region-model triple:
termination under `execR R`, the `readMemAs outDType` per-lane readback,
and the frame. -/
theorem ImplementsR.intro (io : Masked2DKernelIO₂) {R : RoundingModel}
    {outDType : FloatDType}
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
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs ys : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) j) = xs j) →
      (∀ j : Fin io.B, io.read2Mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) j) = ys j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀
          = some s1
        ∧ (∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
            s1.readMemAs outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) j)
              = outDType.ofReal
                  (R.round outDType (f (s₀.pids 0) (s₀.pids 1) xs ys j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) j) →
            (∀ p ∈ io.scratch, r = p.1 →
              ∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
                o ≠ p.2 (s₀.pids 0) (s₀.pids 1) j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R outDType f := by
  -- assemble the unified-core rounded triple once, then convert it back
  -- into the family statement; the flattening bridge lives in
  -- `UKernelIO.ImplementsR.intro`
  have hcore : io.toU.ImplementsR R (fun _ => outDType)
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 2) j')
          (fun j' => vals (⟨1, by decide⟩ : Fin 2) j') j) := by
    refine UKernelIO.ImplementsR.intro _ hok ?_ ?_
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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  projection := io.projection
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

/-- `io.ImplementsR R out1DType out2DType f` — the **rounding-correctness**
relation for the 2D-grid masked two-input / two-output family, written
`io ⊨[R, out1DType, out2DType] f`. Verbatim `Implements`, with two changes:
the kernel runs under the rounding model (`execR R`), and each
**write-active** lane of each output window is read back as a typed cell
holding the ideal real value quantized **once** — `out1DType` for `out1`,
`out2DType` for `out2`. Inputs stay exact ℝ — the rounding model acts at the
kernel's cast/store sites, not at loads. Everything else (the allocation
contract, the four in-bounds obligations, the per-channel read/write gates,
the `pid`/`undef` pins, the frame) is unchanged; at both grids `.real` the
stores are exact and this degenerates to the exact surface.

The grids are **arguments** and there is **one per output channel**, both
named by the notation. The dominant consumer genre here is *quantization*
(`rowwise_quantization_triton`, `int8_quantization`), which writes a
narrow quantized data tile on `out1` and its wide per-row scale on `out2` —
two genuinely different storage precisions in one launch — so a single
scalar grid reused for both outputs would be the wrong shape, not merely a
coarse one. A three-hole `io ⊨[R] f` would in addition print two different
quantization assignments identically. -/
def ImplementsR (io : Masked2DKernelIO₂ₓ₂) (R : RoundingModel)
    (out1DType out2DType : FloatDType)
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
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ j →
          s'.readMemAs out1DType A.flat
              (A.addr io.out1 (io.write1 pid₀ pid₁ j))
            = out1DType.ofReal
                (R.round out1DType ((f pid₀ pid₁ xs ys).1 j)))
      ∧ (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ j →
          s'.readMemAs out2DType A.flat
              (A.addr io.out2 (io.write2 pid₀ pid₁ j))
            = out2DType.ofReal
                (R.round out2DType ((f pid₀ pid₁ xs ys).2 j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ j →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ j)) ∧
             (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ j →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R ", " out1DType ", "
    out2DType "] " f =>
  Masked2DKernelIO₂ₓ₂.ImplementsR io R out1DType out2DType f

/-- Assembly lemma for `⊨[R, out1DType, out2DType]` — the rounding sibling
of `Masked2DKernelIO₂ₓ₂.Implements.intro`, riding the single-shot family's
rounding core `UKernelIO.ImplementsR.intro` through the same `toU`
embedding (at the per-channel output grid `out1DType`/`out2DType`).
Obligations as there, with the safety walk at `Kernel.TraceSafeR R` and
`hrun` returning a rounded region-model triple: termination under `execR R`,
the two `readMemAs` per-lane readbacks, and the frame. -/
theorem ImplementsR.intro (io : Masked2DKernelIO₂ₓ₂) {R : RoundingModel}
    {out1DType out2DType : FloatDType}
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
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs ys : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) j) = xs j) →
      (∀ j : Fin io.B, io.read2Mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) j) = ys j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀
          = some s1
        ∧ (∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) j →
            s1.readMemAs out1DType io.out1
                (io.write1 (s₀.pids 0) (s₀.pids 1) j)
              = out1DType.ofReal
                  (R.round out1DType
                    ((f (s₀.pids 0) (s₀.pids 1) xs ys).1 j)))
        ∧ (∀ j : Fin io.B, io.writeMask2 (s₀.pids 0) (s₀.pids 1) j →
            s1.readMemAs out2DType io.out2
                (io.write2 (s₀.pids 0) (s₀.pids 1) j)
              = out2DType.ofReal
                  (R.round out2DType
                    ((f (s₀.pids 0) (s₀.pids 1) xs ys).2 j)))
        ∧ (∀ r o,
            (r ≠ io.out1 ∨
              ∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) j →
                o ≠ io.write1 (s₀.pids 0) (s₀.pids 1) j) →
            (r ≠ io.out2 ∨
              ∀ j : Fin io.B, io.writeMask2 (s₀.pids 0) (s₀.pids 1) j →
                o ≠ io.write2 (s₀.pids 0) (s₀.pids 1) j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R out1DType out2DType f := by
  -- assemble the unified-core rounded triple once, then convert it back
  -- into the family statement; the flattening bridge lives in
  -- `UKernelIO.ImplementsR.intro`
  have hcore : io.toU.ImplementsR R
      (fun o => match o with
        | ⟨0, _⟩ => out1DType
        | ⟨_+1, _⟩ => out2DType)
      (fun p₀ p₁ vals o => match o with
        | ⟨0, _⟩ => fun j =>
            (f p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 2) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 2) j')).1 j
        | ⟨_+1, _⟩ => fun j =>
            (f p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 2) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 2) j')).2 j) := by
    refine UKernelIO.ImplementsR.intro _ hok ?_ ?_
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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  projection := io.projection
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

/-- `io.ImplementsR R out1DType out2DType f` — the **rounding-correctness**
relation for the 3D-grid masked two-input / two-output family, written
`io ⊨[R, out1DType, out2DType] f`. Verbatim `Implements`, with two changes:
the kernel runs under the rounding model (`execR R`), and each
**write-active** lane of each output window is read back as a typed cell
holding the ideal real value quantized **once** — `out1DType` for `out1`,
`out2DType` for `out2`. Inputs stay exact ℝ — the rounding model acts at the
kernel's cast/store sites, not at loads. Everything else (the allocation
contract, the four in-bounds obligations, the three-program-id windows and
gates, the `pid`/`undef` pins, the frame) is unchanged; at both grids
`.real` the stores are exact and this degenerates to the exact surface.

The grids are **arguments** and there is **one per output channel**, both
named by the notation — the same decision as this skin's 2D sibling
`Masked2DKernelIO₂ₓ₂`, deliberately: the two skins differ only in how many
program ids their windows eat, and a consumer that grows a third launch axis
should be able to move its `⊨[R, …]` headline across without reshaping the
grid argument. A three-hole `io ⊨[R] f` would in addition print two
different quantization assignments identically. -/
def ImplementsR (io : Masked3DKernelIO₂ₓ₂) (R : RoundingModel)
    (out1DType out2DType : FloatDType)
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
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ pid₂ j →
          s'.readMemAs out1DType A.flat
              (A.addr io.out1 (io.write1 pid₀ pid₁ pid₂ j))
            = out1DType.ofReal
                (R.round out1DType ((f pid₀ pid₁ xs ys).1 j)))
      ∧ (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ pid₂ j →
          s'.readMemAs out2DType A.flat
              (A.addr io.out2 (io.write2 pid₀ pid₁ pid₂ j))
            = out2DType.ofReal
                (R.round out2DType ((f pid₀ pid₁ xs ys).2 j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ pid₂ j →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ pid₂ j)) ∧
             (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ pid₂ j →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ pid₂ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R ", " out1DType ", "
    out2DType "] " f =>
  Masked3DKernelIO₂ₓ₂.ImplementsR io R out1DType out2DType f

/-- Assembly lemma for `⊨[R, out1DType, out2DType]` — the rounding sibling
of `Masked3DKernelIO₂ₓ₂.Implements.intro`, riding the single-shot family's
rounding core `UKernelIO.ImplementsR.intro` through the same `toU`
embedding (at the per-channel output grid `out1DType`/`out2DType`).
Obligations as there — `hts`/`hrun` reference all three `s.pids 0/1/2` —
with the safety walk at `Kernel.TraceSafeR R` and `hrun` returning a rounded
region-model triple: termination under `execR R`, the two `readMemAs`
per-lane readbacks, and the frame. -/
theorem ImplementsR.intro (io : Masked3DKernelIO₂ₓ₂) {R : RoundingModel}
    {out1DType out2DType : FloatDType}
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
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs ys : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) = xs j) →
      (∀ j : Fin io.B, io.read2Mask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) = ys j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀
          = some s1
        ∧ (∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMemAs out1DType io.out1
                (io.write1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = out1DType.ofReal
                  (R.round out1DType
                    ((f (s₀.pids 0) (s₀.pids 1) xs ys).1 j)))
        ∧ (∀ j : Fin io.B, io.writeMask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMemAs out2DType io.out2
                (io.write2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = out2DType.ofReal
                  (R.round out2DType
                    ((f (s₀.pids 0) (s₀.pids 1) xs ys).2 j)))
        ∧ (∀ r o,
            (r ≠ io.out1 ∨
              ∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
                o ≠ io.write1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) →
            (r ≠ io.out2 ∨
              ∀ j : Fin io.B, io.writeMask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
                o ≠ io.write2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R out1DType out2DType f := by
  have hcore : io.toU.ImplementsR R
      (fun o => match o with
        | ⟨0, _⟩ => out1DType
        | ⟨_+1, _⟩ => out2DType)
      (fun p₀ p₁ vals o => match o with
        | ⟨0, _⟩ => fun j =>
            (f p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 2) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 2) j')).1 j
        | ⟨_+1, _⟩ => fun j =>
            (f p₀ p₁ (fun j' => vals (⟨0, by decide⟩ : Fin 2) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 2) j')).2 j) := by
    refine UKernelIO.ImplementsR.intro _ hok ?_ ?_
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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  projection := io.projection
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

/-- `io.ImplementsR R out1DType out2DType out3DType f` — the
**rounding-correctness** relation for the 2D-grid masked three-input /
three-output family, written `io ⊨[R, out1DType, out2DType, out3DType] f`.
Verbatim `Implements`, with two changes: the kernel runs under the rounding
model (`execR R`), and each **write-active** lane of each output window is
read back as a typed cell holding the ideal real value quantized **once** —
`out1DType` for `out1`, `out2DType` for `out2`, `out3DType` for `out3`.
Inputs stay exact ℝ — the rounding model acts at the kernel's cast/store
sites, not at loads. Everything else (the allocation contract, the six
in-bounds obligations, the per-channel read/write gates, the `pid`/`undef`
pins, the frame) is unchanged; at all three grids `.real` the stores are
exact and this degenerates to the exact surface.

The grids are **arguments** and there is **one per output channel**, all
three named by the notation. The genre demands it: `fast_layernorm` writes
the normalized row back at the *input's* storage precision while its two
companion channels are the wide fp32 per-row statistics, so the three
outputs of a single launch genuinely quantize at different grids — one
scalar grid reused across them would be the wrong shape, not merely a coarse
one, and a three-hole `io ⊨[R] f` would print different assignments
identically. -/
def ImplementsR (io : Masked2DKernelIO₃ₓ₃) (R : RoundingModel)
    (out1DType out2DType out3DType : FloatDType)
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
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ j →
          s'.readMemAs out1DType A.flat
              (A.addr io.out1 (io.write1 pid₀ pid₁ j))
            = out1DType.ofReal
                (R.round out1DType ((f pid₀ pid₁ xs ys zs).1 j)))
      ∧ (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ j →
          s'.readMemAs out2DType A.flat
              (A.addr io.out2 (io.write2 pid₀ pid₁ j))
            = out2DType.ofReal
                (R.round out2DType ((f pid₀ pid₁ xs ys zs).2.1 j)))
      ∧ (∀ j : Fin io.B, io.writeMask3 pid₀ pid₁ j →
          s'.readMemAs out3DType A.flat
              (A.addr io.out3 (io.write3 pid₀ pid₁ j))
            = out3DType.ofReal
                (R.round out3DType ((f pid₀ pid₁ xs ys zs).2.2 j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.B, io.writeMask1 pid₀ pid₁ j →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ j)) ∧
             (∀ j : Fin io.B, io.writeMask2 pid₀ pid₁ j →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ j)) ∧
             (∀ j : Fin io.B, io.writeMask3 pid₀ pid₁ j →
                o' ≠ A.addr io.out3 (io.write3 pid₀ pid₁ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R ", " out1DType ", "
    out2DType ", " out3DType "] " f =>
  Masked2DKernelIO₃ₓ₃.ImplementsR io R out1DType out2DType out3DType f

/-- Assembly lemma for `⊨[R, out1DType, out2DType, out3DType]` — the
rounding sibling of `Masked2DKernelIO₃ₓ₃.Implements.intro`, riding the
single-shot family's rounding core `UKernelIO.ImplementsR.intro` through the
same `toU` embedding (at the per-channel output grid
`out1DType`/`out2DType`/`out3DType`). Obligations as there, with the safety
walk at `Kernel.TraceSafeR R` and `hrun` returning a rounded region-model
triple: termination under `execR R`, the three `readMemAs` per-lane
readbacks, and the frame. -/
theorem ImplementsR.intro (io : Masked2DKernelIO₃ₓ₃) {R : RoundingModel}
    {out1DType out2DType out3DType : FloatDType}
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
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs ys zs : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) j) = xs j) →
      (∀ j : Fin io.B, io.read2Mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) j) = ys j) →
      (∀ j : Fin io.B, io.read3Mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.in3 (io.read3 (s₀.pids 0) (s₀.pids 1) j) = zs j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀
          = some s1
        ∧ (∀ j : Fin io.B, io.writeMask1 (s₀.pids 0) (s₀.pids 1) j →
            s1.readMemAs out1DType io.out1
                (io.write1 (s₀.pids 0) (s₀.pids 1) j)
              = out1DType.ofReal
                  (R.round out1DType
                    ((f (s₀.pids 0) (s₀.pids 1) xs ys zs).1 j)))
        ∧ (∀ j : Fin io.B, io.writeMask2 (s₀.pids 0) (s₀.pids 1) j →
            s1.readMemAs out2DType io.out2
                (io.write2 (s₀.pids 0) (s₀.pids 1) j)
              = out2DType.ofReal
                  (R.round out2DType
                    ((f (s₀.pids 0) (s₀.pids 1) xs ys zs).2.1 j)))
        ∧ (∀ j : Fin io.B, io.writeMask3 (s₀.pids 0) (s₀.pids 1) j →
            s1.readMemAs out3DType io.out3
                (io.write3 (s₀.pids 0) (s₀.pids 1) j)
              = out3DType.ofReal
                  (R.round out3DType
                    ((f (s₀.pids 0) (s₀.pids 1) xs ys zs).2.2 j)))
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
    refine UKernelIO.ImplementsR.intro _ hok ?_ ?_
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

end VeriTile.Triton
