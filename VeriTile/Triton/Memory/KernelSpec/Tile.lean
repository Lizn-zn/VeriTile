/-
Kernel IO contracts: Tile.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.KernelSpec.TileIndex

namespace VeriTile.Triton


/-! ## Tile-indexed masked IO — non-contiguous footprints

The masked families above address memory as **base + lane**
(`read pid + j.val`), which can only describe a *contiguous* window. Kernels
whose footprint is a genuine tile — `row * stride + col` and friends — cannot
be described that way at all.

`MaskedTileKernelIO₁` lifts that restriction the cheap way: the unified core
already takes a full per-lane address function (`UKernelIO.iwin` / `owin`), so
this is another **thin wrapper**, not a new core and not a new flattening
bridge. Lanes are `TileIndex shape`; `read` / `write` are full address
functions; `toU` enumerates the lanes through `TileShape.allIndices`.

Scratch channels are deliberately omitted — the tile-footprint ports do not use
them, and leaving them out keeps the `toU` plumbing short. Add them the same way
`MaskedKernelIO₁` does if a consumer ever needs them. -/

/-- One-input / one-output masked IO whose lanes are **tile indices** and whose
windows are full address functions. -/
structure MaskedTileKernelIO₁ where
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
  /-- The tile footprint each program instance owns. -/
  shape : TileShape
  /-- Lane `i`'s read address for program `pid` — a full address function, not
  a base offset. -/
  read : Nat → TileIndex shape → Nat
  /-- Lane `i`'s write address for program `pid`. -/
  write : Nat → TileIndex shape → Nat
  /-- Program `pid`'s **read-active** lanes. -/
  mask : Nat → TileIndex shape → Prop
  /-- Program `pid`'s **write-active** lanes; defaults to `mask`. -/
  writeMask : Nat → TileIndex shape → Prop := mask

namespace MaskedTileKernelIO₁

/-- `io.Implements f` — the tile-indexed sibling of
`MaskedKernelIO₁.Implements`: same Hoare triple, with lanes ranging over
`TileIndex io.shape` and addresses given by the window functions.

The spec `f` takes the **program id**, for the same reason
`Masked2DKernelIO₁`'s does: as soon as `io.mask` is pid-dependent — the usual
tail-masked tiling — a *reduction* over the active lanes is irreducibly
pid-dependent (a full block reduces a different index set than the tail
block), so a pid-independent spec would be falsifiable. Pointwise kernels
simply ignore the argument. -/
def Implements (io : MaskedTileKernelIO₁)
    (f : Nat → (TileIndex io.shape → ℝ) → TileIndex io.shape → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    (∀ i : TileIndex io.shape, io.mask pid i →
      io.read pid i < A.extent io.inp) →
    (∀ i : TileIndex io.shape, io.writeMask pid i →
      io.write pid i < A.extent io.out) →
  ∀ (xs : TileIndex io.shape → ℝ) (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ i : TileIndex io.shape, io.mask pid i →
      s₀.readMem io.inp (io.read pid i) = xs i) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ i : TileIndex io.shape, io.writeMask pid i →
          s'.readMem A.flat (A.addr io.out (io.write pid i)) = f pid xs i)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ i : TileIndex io.shape, io.writeMask pid i →
              o' ≠ A.addr io.out (io.write pid i)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => MaskedTileKernelIO₁.Implements

/-- Embed into the unified core. Lanes are enumerated by
`TileShape.allIndices`, so the core's `Fin`-indexed windows carry the tile
addresses verbatim. -/
private def toU (io : MaskedTileKernelIO₁) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 1
  nOut := 1
  nScr := 0
  bufs := [io.inp, io.out]
  ity := fun _ => .float
  iarity := fun _ => (TileShape.allIndices io.shape).length
  ibuf := fun _ => io.inp
  oty := fun _ => .float
  oarity := fun _ => (TileShape.allIndices io.shape).length
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun _ _ p₀ _ _ j =>
    io.read p₀ ((TileShape.allIndices io.shape).get j)
  imask := fun _ _ p₀ _ _ j =>
    io.mask p₀ ((TileShape.allIndices io.shape).get j)
  owin := fun _ _ p₀ _ _ j =>
    io.write p₀ ((TileShape.allIndices io.shape).get j)
  omask := fun _ _ p₀ _ _ j =>
    io.writeMask p₀ ((TileShape.allIndices io.shape).get j)
  swin := fun t _ _ _ _ _ => t.elim0
  smask := fun t _ _ _ _ _ => t.elim0

/-- Assembly lemma — the tile-indexed sibling of
`MaskedKernelIO₁.Implements.intro`. The three obligations are the family's
usual ones (`FlattenOk`, the safety walk, the region-model Hoare triple);
the only difference is that every lane-wise obligation ranges over
`TileIndex io.shape` and every address comes from a window *function*, so
nothing in the statement forces a contiguous footprint. -/
theorem Implements.intro (io : MaskedTileKernelIO₁)
    {f : Nat → (TileIndex io.shape → ℝ) → TileIndex io.shape → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ i : TileIndex io.shape, io.mask s.pid i →
        io.read s.pid i < bounds io.inp) →
      (∀ i : TileIndex io.shape, io.writeMask s.pid i →
        io.write s.pid i < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : TileIndex io.shape → ℝ),
      (∀ i : TileIndex io.shape, io.mask s₀.pid i →
        s₀.readMem io.inp (io.read s₀.pid i) = xs i) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ i : TileIndex io.shape, io.writeMask s₀.pid i →
            s1.readMem io.out (io.write s₀.pid i) = f s₀.pid xs i)
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ i : TileIndex io.shape, io.writeMask s₀.pid i →
                o ≠ io.write s₀.pid i) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the two `tilePos` round-trips are the whole content of
  -- the conversion
  have hcore : io.toU.Implements
      (fun p₀ _p₁ vals _o j =>
        f p₀ (fun i => vals (⟨0, by decide⟩ : Fin 1) (tilePos io.shape i))
          ((TileShape.allIndices io.shape).get j)) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      have hib' : ∀ i : TileIndex io.shape, io.mask s.pid i →
          io.read s.pid i < bounds io.inp := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        exact hib (⟨0, by decide⟩ : Fin 1) j
      have hob' : ∀ i : TileIndex io.shape, io.writeMask s.pid i →
          io.write s.pid i < bounds io.out := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        exact hob (⟨0, by decide⟩ : Fin 1) j
      exact hts bounds s hib' hob'
    · intro s₀ vals _hundef hpins
      have hpins' : ∀ i : TileIndex io.shape, io.mask s₀.pid i →
          s₀.readMem io.inp (io.read s₀.pid i)
            = vals (⟨0, by decide⟩ : Fin 1) (tilePos io.shape i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨0, by decide⟩ : Fin 1) j
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun i => vals (⟨0, by decide⟩ : Fin 1) (tilePos io.shape i))
          hpins'
      refine ⟨s1, hexec, fun _o j hj => hval _ hj, ?_⟩
      intro r o' hoc _hsc
      have hoc' : ∀ i : TileIndex io.shape, io.writeMask s₀.pid i →
          r ≠ io.out ∨ o' ≠ io.write s₀.pid i := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        exact hoc (⟨0, by decide⟩ : Fin 1) j
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun i hi => ?_
        rcases hoc' i hi with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid h1 h2 xs s₀ hpid hu hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
      (fun _ j => xs ((TileShape.allIndices io.shape).get j)) s₀ hpid rfl rfl hu
      (fun _i j hj => h1 _ hj) (fun _o j hj => h2 _ hj) (fun t => t.elim0)
      (fun _i j hj => hx _ hj)
  have hxs :
      (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        = xs :=
    funext fun i => by rw [get_tilePos]
  refine ⟨s', hexec, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 1) j hj).trans ?_
    show f pid
        (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        ((TileShape.allIndices io.shape).get j)
      = f pid xs ((TileShape.allIndices io.shape).get j)
    rw [hxs]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | hout
    · exact Or.inl hflat
    · exact Or.inr ⟨fun _o j hj => hout _ hj, fun t => t.elim0⟩

/-- `io.ImplementsR R outDType f` — the **rounding-correctness** relation
`io ⊨[R, outDType] f` for the tile-indexed masked one-input family. Verbatim
`Implements` with two changes: the kernel runs under `execR R`, and each active
output cell is read back at `outDType` and must hold
`R.round outDType (f pid xs i)`.

`outDType` is the dtype the kernel's terminal store writes at, and it is an
explicit argument rather than a field of `io` for the same reason as elsewhere in
this file: `Implements` and `ImplementsR` would otherwise print with the same
signature, and a three-hole `io ⊨[R] f` would render the `.fp16` and `.real`
faces identically. -/
def ImplementsR (io : MaskedTileKernelIO₁) (R : RoundingModel)
    (outDType : FloatDType)
    (f : Nat → (TileIndex io.shape → ℝ) → TileIndex io.shape → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    (∀ i : TileIndex io.shape, io.mask pid i →
      io.read pid i < A.extent io.inp) →
    (∀ i : TileIndex io.shape, io.writeMask pid i →
      io.write pid i < A.extent io.out) →
  ∀ (xs : TileIndex io.shape → ℝ) (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ i : TileIndex io.shape, io.mask pid i →
      s₀.readMem io.inp (io.read pid i) = xs i) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ i : TileIndex io.shape, io.writeMask pid i →
          s'.readMemAs outDType A.flat (A.addr io.out (io.write pid i))
            = outDType.ofReal (R.round outDType (f pid xs i)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ i : TileIndex io.shape, io.writeMask pid i →
              o' ≠ A.addr io.out (io.write pid i)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R ", " outDType "] " f =>
  MaskedTileKernelIO₁.ImplementsR io R outDType f

/-- Assembly lemma for `⊨[R, outDType]` — the rounding sibling of
`Implements.intro`, riding the same `toU` embedding through the family's rounding
core `UKernelIO.ImplementsR.intro` (at the constant output grid
`fun _ => outDType`). Obligations are the usual three, with the safety walk at
`Kernel.TraceSafeR R` and `hrun` returning a rounded region-model triple. -/
theorem ImplementsR.intro (io : MaskedTileKernelIO₁) {R : RoundingModel}
    {outDType : FloatDType}
    {f : Nat → (TileIndex io.shape → ℝ) → TileIndex io.shape → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ i : TileIndex io.shape, io.mask s.pid i →
        io.read s.pid i < bounds io.inp) →
      (∀ i : TileIndex io.shape, io.writeMask s.pid i →
        io.write s.pid i < bounds io.out) →
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : TileIndex io.shape → ℝ),
      (∀ i : TileIndex io.shape, io.mask s₀.pid i →
        s₀.readMem io.inp (io.read s₀.pid i) = xs i) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ i : TileIndex io.shape, io.writeMask s₀.pid i →
            s1.readMemAs outDType io.out (io.write s₀.pid i)
              = outDType.ofReal (R.round outDType (f s₀.pid xs i)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ i : TileIndex io.shape, io.writeMask s₀.pid i →
                o ≠ io.write s₀.pid i) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R outDType f := by
  have hcore : io.toU.ImplementsR R (fun _ => outDType)
      (fun p₀ _p₁ vals _o j =>
        f p₀ (fun i => vals (⟨0, by decide⟩ : Fin 1) (tilePos io.shape i))
          ((TileShape.allIndices io.shape).get j)) := by
    refine UKernelIO.ImplementsR.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      refine hts bounds s ?_ ?_
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨0, by decide⟩ : Fin 1) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hob (⟨0, by decide⟩ : Fin 1) j
    · intro s₀ vals _hundef hpins
      have hx : ∀ i : TileIndex io.shape, io.mask s₀.pid i →
          s₀.readMem io.inp (io.read s₀.pid i)
            = vals (⟨0, by decide⟩ : Fin 1) (tilePos io.shape i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨0, by decide⟩ : Fin 1) j
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun i => vals (⟨0, by decide⟩ : Fin 1) (tilePos io.shape i)) hx
      refine ⟨s1, hexec, fun _o j hj => hval _ hj, ?_⟩
      intro r o' hoc _hsc
      have hoc' : ∀ i : TileIndex io.shape, io.writeMask s₀.pid i →
          r ≠ io.out ∨ o' ≠ io.write s₀.pid i :=
        (forall_tileIndex_iff _).mp fun j => hoc (⟨0, by decide⟩ : Fin 1) j
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun i hi => ?_
        rcases hoc' i hi with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid h1 h2 xs s₀ hpid hu hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
      (fun _ j => xs ((TileShape.allIndices io.shape).get j)) s₀ hpid rfl rfl hu
      (fun _i j hj => h1 _ hj) (fun _o j hj => h2 _ hj) (fun t => t.elim0)
      (fun _i j hj => hx _ hj)
  have hxs :
      (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        = xs :=
    funext fun i => by rw [get_tilePos]
  refine ⟨s', hexec, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 1) j hj).trans ?_
    show outDType.ofReal (R.round outDType (f pid
        (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        ((TileShape.allIndices io.shape).get j)))
      = outDType.ofReal (R.round outDType
          (f pid xs ((TileShape.allIndices io.shape).get j)))
    rw [hxs]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | hout
    · exact Or.inl hflat
    · exact Or.inr ⟨fun _o j hj => hout _ hj, fun t => t.elim0⟩

end MaskedTileKernelIO₁


/-! ## Tile-indexed masked IO, two inputs and two program axes

`MaskedTile2DKernelIO₂` is to `Masked2DKernelIO₂` what `MaskedTileKernelIO₁` is
to `MaskedKernelIO₁`: same two-pid, two-input, lane-masked Hoare triple, with
lanes ranging over `TileIndex shape` instead of `Fin B`. `Masked2DKernelIO₂`
already carries **full address functions** (not `base + lane`), so the only
thing this adds is the lane type — which is exactly what a genuine 2-D tile
kernel needs: no `Fin (BLOCK_M * BLOCK_N)` ↔ `TileIndex [BLOCK_M, BLOCK_N]`
flattening anywhere in the statement or the proof.

The second input keeps `Masked2DKernelIO₂`'s independent `read2` / `read2Mask`
pair, which is what makes an **unmasked broadcast scalar** expressible: point
`read2` at the scalar's offset for every lane and take `read2Mask := fun _ _ _ =>
True`, and the pin hypothesis says every lane of `ys` holds that one cell.

Scratch channels are omitted, as in `MaskedTileKernelIO₁`. -/

/-- Two-input / one-output masked IO with two program axes, whose lanes are
**tile indices** and whose windows are full address functions. -/
structure MaskedTile2DKernelIO₂ where
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
  /-- The tile footprint each program instance owns. -/
  shape : TileShape
  /-- Lane `i`'s `in1` read address for program `(pid₀, pid₁)`. -/
  read1 : Nat → Nat → TileIndex shape → Nat
  /-- Lane `i`'s `in2` read address. -/
  read2 : Nat → Nat → TileIndex shape → Nat
  /-- Lane `i`'s write address. -/
  write : Nat → Nat → TileIndex shape → Nat
  /-- Program `(pid₀, pid₁)`'s **`in1` read-active** lanes. -/
  mask : Nat → Nat → TileIndex shape → Prop
  /-- Program `(pid₀, pid₁)`'s **`in2` read-active** lanes; defaults to `mask`.
  A broadcast scalar read is the `fun _ _ _ => True` instance. -/
  read2Mask : Nat → Nat → TileIndex shape → Prop := mask
  /-- Program `(pid₀, pid₁)`'s **write-active** lanes; defaults to `mask`. -/
  writeMask : Nat → Nat → TileIndex shape → Prop := mask

namespace MaskedTile2DKernelIO₂

/-- `io.Implements f` — the tile-indexed sibling of
`Masked2DKernelIO₂.Implements`. The spec `f` takes both pids for the same
reason that family's does: on a tiled axis a per-block value is irreducibly
block-dependent, so a pid-independent spec would be falsifiable. Pid-independent
kernels ignore the two arguments. -/
def Implements (io : MaskedTile2DKernelIO₂)
    (f : Nat → Nat → (TileIndex io.shape → ℝ) → (TileIndex io.shape → ℝ) →
      TileIndex io.shape → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ i →
      io.read1 pid₀ pid₁ i < A.extent io.in1) →
    (∀ i : TileIndex io.shape, io.read2Mask pid₀ pid₁ i →
      io.read2 pid₀ pid₁ i < A.extent io.in2) →
    (∀ i : TileIndex io.shape, io.writeMask pid₀ pid₁ i →
      io.write pid₀ pid₁ i < A.extent io.out) →
  ∀ (xs ys : TileIndex io.shape → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ i →
      s₀.readMem io.in1 (io.read1 pid₀ pid₁ i) = xs i) →
    (∀ i : TileIndex io.shape, io.read2Mask pid₀ pid₁ i →
      s₀.readMem io.in2 (io.read2 pid₀ pid₁ i) = ys i) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ i : TileIndex io.shape, io.writeMask pid₀ pid₁ i →
          s'.readMem A.flat (A.addr io.out (io.write pid₀ pid₁ i))
            = f pid₀ pid₁ xs ys i)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ i : TileIndex io.shape, io.writeMask pid₀ pid₁ i →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ i)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => MaskedTile2DKernelIO₂.Implements

/-- Embed into the unified core: two float tile channels with independent
windows and masks, one output, no scratch. Lanes are enumerated by
`TileShape.allIndices`. -/
private def toU (io : MaskedTile2DKernelIO₂) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 2
  nOut := 1
  nScr := 0
  bufs := [io.in1, io.in2, io.out]
  ity := fun _ => .float
  iarity := fun _ => (TileShape.allIndices io.shape).length
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | _ => io.in2
  oty := fun _ => .float
  oarity := fun _ => (TileShape.allIndices io.shape).length
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ p₁ ((TileShape.allIndices io.shape).get j)
    | _ => fun j => io.read2 p₀ p₁ ((TileShape.allIndices io.shape).get j)
  imask := fun i _ p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j => io.mask p₀ p₁ ((TileShape.allIndices io.shape).get j)
    | _ => fun j =>
        io.read2Mask p₀ p₁ ((TileShape.allIndices io.shape).get j)
  owin := fun _ _ p₀ p₁ _ j =>
    io.write p₀ p₁ ((TileShape.allIndices io.shape).get j)
  omask := fun _ _ p₀ p₁ _ j =>
    io.writeMask p₀ p₁ ((TileShape.allIndices io.shape).get j)
  swin := fun t _ _ _ _ _ => t.elim0
  smask := fun t _ _ _ _ _ => t.elim0

/-- Assembly lemma — the tile-indexed sibling of
`Masked2DKernelIO₂.Implements.intro`. Same three obligations; every lane-wise
obligation ranges over `TileIndex io.shape` and every address comes from a
window *function*, so nothing forces a contiguous footprint. -/
theorem Implements.intro (io : MaskedTile2DKernelIO₂)
    {f : Nat → Nat → (TileIndex io.shape → ℝ) → (TileIndex io.shape → ℝ) →
      TileIndex io.shape → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ i : TileIndex io.shape, io.mask (s.pids 0) (s.pids 1) i →
        io.read1 (s.pids 0) (s.pids 1) i < bounds io.in1) →
      (∀ i : TileIndex io.shape, io.read2Mask (s.pids 0) (s.pids 1) i →
        io.read2 (s.pids 0) (s.pids 1) i < bounds io.in2) →
      (∀ i : TileIndex io.shape, io.writeMask (s.pids 0) (s.pids 1) i →
        io.write (s.pids 0) (s.pids 1) i < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs ys : TileIndex io.shape → ℝ),
      (∀ i : TileIndex io.shape, io.mask (s₀.pids 0) (s₀.pids 1) i →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) i) = xs i) →
      (∀ i : TileIndex io.shape, io.read2Mask (s₀.pids 0) (s₀.pids 1) i →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) i) = ys i) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ i : TileIndex io.shape, io.writeMask (s₀.pids 0) (s₀.pids 1) i →
            s1.readMem io.out (io.write (s₀.pids 0) (s₀.pids 1) i)
              = f (s₀.pids 0) (s₀.pids 1) xs ys i)
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ i : TileIndex io.shape, io.writeMask (s₀.pids 0) (s₀.pids 1) i →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) i) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁ (fun i => vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shape i))
          (fun i => vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape i))
          ((TileShape.allIndices io.shape).get j)) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      have h1 : ∀ i : TileIndex io.shape, io.mask (s.pids 0) (s.pids 1) i →
          io.read1 (s.pids 0) (s.pids 1) i < bounds io.in1 := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        exact hib (⟨0, by decide⟩ : Fin 2) j
      have h2 : ∀ i : TileIndex io.shape,
          io.read2Mask (s.pids 0) (s.pids 1) i →
          io.read2 (s.pids 0) (s.pids 1) i < bounds io.in2 := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        exact hib (⟨1, by decide⟩ : Fin 2) j
      have h3 : ∀ i : TileIndex io.shape,
          io.writeMask (s.pids 0) (s.pids 1) i →
          io.write (s.pids 0) (s.pids 1) i < bounds io.out := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        exact hob (⟨0, by decide⟩ : Fin 1) j
      exact hts bounds s h1 h2 h3
    · intro s₀ vals _hundef hpins
      have hx : ∀ i : TileIndex io.shape,
          io.mask (s₀.pids 0) (s₀.pids 1) i →
          s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) i)
            = vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shape i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨0, by decide⟩ : Fin 2) j
      have hy : ∀ i : TileIndex io.shape,
          io.read2Mask (s₀.pids 0) (s₀.pids 1) i →
          s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) i)
            = vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨1, by decide⟩ : Fin 2) j
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun i => vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shape i))
          (fun i => vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape i)) hx hy
      refine ⟨s1, hexec, fun _o j hj => hval _ hj, ?_⟩
      intro r o' hoc _hsc
      have hoc' : ∀ i : TileIndex io.shape,
          io.writeMask (s₀.pids 0) (s₀.pids 1) i →
          r ≠ io.out ∨ o' ≠ io.write (s₀.pids 0) (s₀.pids 1) i := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        exact hoc (⟨0, by decide⟩ : Fin 1) j
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun i hi => ?_
        rcases hoc' i hi with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ h1 h2 h3 xs ys s₀ hp₀ hp₁ hu hx hy
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun j => xs ((TileShape.allIndices io.shape).get j)
        | _ => fun j => ys ((TileShape.allIndices io.shape).get j))
      s₀ hp₀ hp₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => h1 _ hj
        | ⟨_ + 1, _⟩ => fun j hj => h2 _ hj)
      (fun _o j hj => h3 _ hj) (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hx _ hj
        | ⟨_ + 1, _⟩ => fun j hj => hy _ hj)
  have hxs :
      (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        = xs :=
    funext fun i => by rw [get_tilePos]
  have hys :
      (fun i => ys ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        = ys :=
    funext fun i => by rw [get_tilePos]
  refine ⟨s', hexec, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 1) j hj).trans ?_
    show f pid₀ pid₁
        (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        (fun i => ys ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        ((TileShape.allIndices io.shape).get j)
      = f pid₀ pid₁ xs ys ((TileShape.allIndices io.shape).get j)
    rw [hxs, hys]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | hout
    · exact Or.inl hflat
    · exact Or.inr ⟨fun _o j hj => hout _ hj, fun t => t.elim0⟩

end MaskedTile2DKernelIO₂


/-! ## Tile-indexed masked IO with a **shape per channel**

`MaskedTile2DKernelIO₂` gives all three channels the same lane set, which is
right for elementwise and reduce-along-one-axis kernels. A contraction cannot
live there: a GEMM reads `[BLOCK_M, BLOCK_K]` and `[BLOCK_K, BLOCK_N]` and writes
`[BLOCK_M, BLOCK_N]` — three *different* tile shapes, related only through the
spec function.

`MaskedTileShapedKernelIO₂` gives each channel its own `TileShape`. The unified
core already has a per-channel arity (`iarity` / `oarity` are functions of the
channel), so this is again a thin wrapper: only the wrapper families had ever
tied the shapes together.

`Implements.intro` is the same three-obligation assembly as the sibling skins;
`tilePos` / `get_tilePos` / `tilePos_get` / `forall_tileIndex_iff` are already
shape-polymorphic, so they apply at each channel's own shape unchanged. -/

/-- Two-input / one-output masked IO with two program axes, where **each channel
carries its own tile shape**. -/
structure MaskedTileShapedKernelIO₂ where
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
  /-- First input's tile shape. -/
  shape1 : TileShape
  /-- Second input's tile shape. -/
  shape2 : TileShape
  /-- Output's tile shape. -/
  shapeOut : TileShape
  /-- Lane `i`'s `in1` read address for program `(pid₀, pid₁)`. -/
  read1 : Nat → Nat → TileIndex shape1 → Nat
  /-- Lane `i`'s `in2` read address. -/
  read2 : Nat → Nat → TileIndex shape2 → Nat
  /-- Lane `o`'s write address. -/
  write : Nat → Nat → TileIndex shapeOut → Nat
  /-- `in1`'s read-active lanes. -/
  mask1 : Nat → Nat → TileIndex shape1 → Prop
  /-- `in2`'s read-active lanes. -/
  mask2 : Nat → Nat → TileIndex shape2 → Prop
  /-- The output's write-active lanes. -/
  writeMask : Nat → Nat → TileIndex shapeOut → Prop

namespace MaskedTileShapedKernelIO₂

/-- `io.Implements f` — the per-channel-shape sibling of
`MaskedTile2DKernelIO₂.Implements`. `f` relates the two input lane functions,
each on its own shape, to a value at an **output** lane; a contraction is exactly
a spec that reads both inputs at lanes the output index does not name. -/
def Implements (io : MaskedTileShapedKernelIO₂)
    (f : Nat → Nat → (TileIndex io.shape1 → ℝ) → (TileIndex io.shape2 → ℝ) →
      TileIndex io.shapeOut → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
    (∀ i : TileIndex io.shape1, io.mask1 pid₀ pid₁ i →
      io.read1 pid₀ pid₁ i < A.extent io.in1) →
    (∀ i : TileIndex io.shape2, io.mask2 pid₀ pid₁ i →
      io.read2 pid₀ pid₁ i < A.extent io.in2) →
    (∀ o : TileIndex io.shapeOut, io.writeMask pid₀ pid₁ o →
      io.write pid₀ pid₁ o < A.extent io.out) →
  ∀ (xs : TileIndex io.shape1 → ℝ) (ys : TileIndex io.shape2 → ℝ)
    (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ i : TileIndex io.shape1, io.mask1 pid₀ pid₁ i →
      s₀.readMem io.in1 (io.read1 pid₀ pid₁ i) = xs i) →
    (∀ i : TileIndex io.shape2, io.mask2 pid₀ pid₁ i →
      s₀.readMem io.in2 (io.read2 pid₀ pid₁ i) = ys i) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ o : TileIndex io.shapeOut, io.writeMask pid₀ pid₁ o →
          s'.readMem A.flat (A.addr io.out (io.write pid₀ pid₁ o))
            = f pid₀ pid₁ xs ys o)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ o : TileIndex io.shapeOut, io.writeMask pid₀ pid₁ o →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ o)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => MaskedTileShapedKernelIO₂.Implements

/-- Embed into the unified core: two float tile channels with **independent
arities**, one output with its own arity, no scratch. -/
private def toU (io : MaskedTileShapedKernelIO₂) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 2
  nOut := 1
  nScr := 0
  bufs := [io.in1, io.in2, io.out]
  ity := fun _ => .float
  -- every per-channel match enumerates **all three** `Fin 2` patterns: a
  -- catch-all `| _ => …` leaves `i` unrefined, so `Fin (iarity i)` does not
  -- reduce in that branch and the two different arities cannot typecheck
  iarity := fun i => match i with
    | ⟨0, _⟩ => (TileShape.allIndices io.shape1).length
    | ⟨1, _⟩ => (TileShape.allIndices io.shape2).length
    | ⟨_ + 2, h⟩ => absurd h (by omega)
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | _ => io.in2
  oty := fun _ => .float
  oarity := fun _ => (TileShape.allIndices io.shapeOut).length
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j =>
        io.read1 p₀ p₁ ((TileShape.allIndices io.shape1).get j)
    | ⟨1, _⟩ => fun j =>
        io.read2 p₀ p₁ ((TileShape.allIndices io.shape2).get j)
    | ⟨_ + 2, h⟩ => absurd h (by omega)
  imask := fun i _ p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j =>
        io.mask1 p₀ p₁ ((TileShape.allIndices io.shape1).get j)
    | ⟨1, _⟩ => fun j =>
        io.mask2 p₀ p₁ ((TileShape.allIndices io.shape2).get j)
    | ⟨_ + 2, h⟩ => absurd h (by omega)
  owin := fun _ _ p₀ p₁ _ j =>
    io.write p₀ p₁ ((TileShape.allIndices io.shapeOut).get j)
  omask := fun _ _ p₀ p₁ _ j =>
    io.writeMask p₀ p₁ ((TileShape.allIndices io.shapeOut).get j)
  swin := fun t _ _ _ _ _ => t.elim0
  smask := fun t _ _ _ _ _ => t.elim0

/-- Assembly lemma — the per-channel-shape sibling of
`MaskedTile2DKernelIO₂.Implements.intro`. -/
theorem Implements.intro (io : MaskedTileShapedKernelIO₂)
    {f : Nat → Nat → (TileIndex io.shape1 → ℝ) → (TileIndex io.shape2 → ℝ) →
      TileIndex io.shapeOut → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ i : TileIndex io.shape1, io.mask1 (s.pids 0) (s.pids 1) i →
        io.read1 (s.pids 0) (s.pids 1) i < bounds io.in1) →
      (∀ i : TileIndex io.shape2, io.mask2 (s.pids 0) (s.pids 1) i →
        io.read2 (s.pids 0) (s.pids 1) i < bounds io.in2) →
      (∀ o : TileIndex io.shapeOut, io.writeMask (s.pids 0) (s.pids 1) o →
        io.write (s.pids 0) (s.pids 1) o < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : TileIndex io.shape1 → ℝ)
        (ys : TileIndex io.shape2 → ℝ),
      (∀ i : TileIndex io.shape1, io.mask1 (s₀.pids 0) (s₀.pids 1) i →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) i) = xs i) →
      (∀ i : TileIndex io.shape2, io.mask2 (s₀.pids 0) (s₀.pids 1) i →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) i) = ys i) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ o : TileIndex io.shapeOut,
            io.writeMask (s₀.pids 0) (s₀.pids 1) o →
            s1.readMem io.out (io.write (s₀.pids 0) (s₀.pids 1) o)
              = f (s₀.pids 0) (s₀.pids 1) xs ys o)
        ∧ (∀ r o',
            (r ≠ io.out ∨
              ∀ o : TileIndex io.shapeOut,
                io.writeMask (s₀.pids 0) (s₀.pids 1) o →
                o' ≠ io.write (s₀.pids 0) (s₀.pids 1) o) →
            s1.mem r o' = s₀.mem r o')) :
    io.Implements f := by
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁ (fun i => vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shape1 i))
          (fun i => vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape2 i))
          ((TileShape.allIndices io.shapeOut).get j)) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      have h1 : ∀ i : TileIndex io.shape1, io.mask1 (s.pids 0) (s.pids 1) i →
          io.read1 (s.pids 0) (s.pids 1) i < bounds io.in1 := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        exact hib (⟨0, by decide⟩ : Fin 2) j
      have h2 : ∀ i : TileIndex io.shape2, io.mask2 (s.pids 0) (s.pids 1) i →
          io.read2 (s.pids 0) (s.pids 1) i < bounds io.in2 := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        exact hib (⟨1, by decide⟩ : Fin 2) j
      have h3 : ∀ o : TileIndex io.shapeOut,
          io.writeMask (s.pids 0) (s.pids 1) o →
          io.write (s.pids 0) (s.pids 1) o < bounds io.out := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        exact hob (⟨0, by decide⟩ : Fin 1) j
      exact hts bounds s h1 h2 h3
    · intro s₀ vals _hundef hpins
      have hx : ∀ i : TileIndex io.shape1,
          io.mask1 (s₀.pids 0) (s₀.pids 1) i →
          s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) i)
            = vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shape1 i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨0, by decide⟩ : Fin 2) j
      have hy : ∀ i : TileIndex io.shape2,
          io.mask2 (s₀.pids 0) (s₀.pids 1) i →
          s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) i)
            = vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape2 i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨1, by decide⟩ : Fin 2) j
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun i => vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shape1 i))
          (fun i => vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape2 i)) hx hy
      refine ⟨s1, hexec, fun _o j hj => hval _ hj, ?_⟩
      intro r o' hoc _hsc
      have hoc' : ∀ o : TileIndex io.shapeOut,
          io.writeMask (s₀.pids 0) (s₀.pids 1) o →
          r ≠ io.out ∨ o' ≠ io.write (s₀.pids 0) (s₀.pids 1) o := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        exact hoc (⟨0, by decide⟩ : Fin 1) j
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun o hoact => ?_
        rcases hoc' o hoact with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ h1 h2 h3 xs ys s₀ hp₀ hp₁ hu hx hy
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun j => xs ((TileShape.allIndices io.shape1).get j)
        | ⟨1, _⟩ => fun j => ys ((TileShape.allIndices io.shape2).get j)
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
      s₀ hp₀ hp₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => h1 _ hj
        | ⟨1, _⟩ => fun j hj => h2 _ hj
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
      (fun _o j hj => h3 _ hj) (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hx _ hj
        | ⟨1, _⟩ => fun j hj => hy _ hj
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
  have hxs :
      (fun i => xs ((TileShape.allIndices io.shape1).get (tilePos io.shape1 i)))
        = xs :=
    funext fun i => by rw [get_tilePos]
  have hys :
      (fun i => ys ((TileShape.allIndices io.shape2).get (tilePos io.shape2 i)))
        = ys :=
    funext fun i => by rw [get_tilePos]
  refine ⟨s', hexec, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 1) j hj).trans ?_
    show f pid₀ pid₁
        (fun i =>
          xs ((TileShape.allIndices io.shape1).get (tilePos io.shape1 i)))
        (fun i =>
          ys ((TileShape.allIndices io.shape2).get (tilePos io.shape2 i)))
        ((TileShape.allIndices io.shapeOut).get j)
      = f pid₀ pid₁ xs ys ((TileShape.allIndices io.shapeOut).get j)
    rw [hxs, hys]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | hout
    · exact Or.inl hflat
    · exact Or.inr ⟨fun _o j hj => hout _ hj, fun t => t.elim0⟩

/-- `io.ImplementsR R outDType f` — the **rounding-correctness** relation
`io ⊨[R, outDType] f` for the per-channel-shape two-input family. Verbatim
`Implements` with two changes: the kernel runs under `execR R`, and each active
output cell is read back at `outDType` and must hold
`R.round outDType (f pid₀ pid₁ xs ys o)`.

This is the face a **narrowing** contraction wants: when the kernel's terminal
cast names a rounding grid (`.to(tl.float16)` lowers to
`Op.castFloat _ .fp16`, unlike `.to(tl.float32)` which erases to `.real`), the
`R.round outDType` on the right is not decoration — it is the quantization the
kernel actually performs, and `f` stays the exact ℝ contraction. -/
def ImplementsR (io : MaskedTileShapedKernelIO₂) (R : RoundingModel)
    (outDType : FloatDType)
    (f : Nat → Nat → (TileIndex io.shape1 → ℝ) → (TileIndex io.shape2 → ℝ) →
      TileIndex io.shapeOut → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
    (∀ i : TileIndex io.shape1, io.mask1 pid₀ pid₁ i →
      io.read1 pid₀ pid₁ i < A.extent io.in1) →
    (∀ i : TileIndex io.shape2, io.mask2 pid₀ pid₁ i →
      io.read2 pid₀ pid₁ i < A.extent io.in2) →
    (∀ o : TileIndex io.shapeOut, io.writeMask pid₀ pid₁ o →
      io.write pid₀ pid₁ o < A.extent io.out) →
  ∀ (xs : TileIndex io.shape1 → ℝ) (ys : TileIndex io.shape2 → ℝ)
    (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ i : TileIndex io.shape1, io.mask1 pid₀ pid₁ i →
      s₀.readMem io.in1 (io.read1 pid₀ pid₁ i) = xs i) →
    (∀ i : TileIndex io.shape2, io.mask2 pid₀ pid₁ i →
      s₀.readMem io.in2 (io.read2 pid₀ pid₁ i) = ys i) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ o : TileIndex io.shapeOut, io.writeMask pid₀ pid₁ o →
          s'.readMemAs outDType A.flat (A.addr io.out (io.write pid₀ pid₁ o))
            = outDType.ofReal (R.round outDType (f pid₀ pid₁ xs ys o)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ o : TileIndex io.shapeOut, io.writeMask pid₀ pid₁ o →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ o)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R ", " outDType "] " f =>
  MaskedTileShapedKernelIO₂.ImplementsR io R outDType f

/-- Assembly lemma for `⊨[R, outDType]` — the rounding sibling of
`Implements.intro`, riding the same `toU` embedding through the family's rounding
core `UKernelIO.ImplementsR.intro` (at the constant output grid
`fun _ => outDType`). -/
theorem ImplementsR.intro (io : MaskedTileShapedKernelIO₂) {R : RoundingModel}
    {outDType : FloatDType}
    {f : Nat → Nat → (TileIndex io.shape1 → ℝ) → (TileIndex io.shape2 → ℝ) →
      TileIndex io.shapeOut → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ i : TileIndex io.shape1, io.mask1 (s.pids 0) (s.pids 1) i →
        io.read1 (s.pids 0) (s.pids 1) i < bounds io.in1) →
      (∀ i : TileIndex io.shape2, io.mask2 (s.pids 0) (s.pids 1) i →
        io.read2 (s.pids 0) (s.pids 1) i < bounds io.in2) →
      (∀ o : TileIndex io.shapeOut, io.writeMask (s.pids 0) (s.pids 1) o →
        io.write (s.pids 0) (s.pids 1) o < bounds io.out) →
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : TileIndex io.shape1 → ℝ)
        (ys : TileIndex io.shape2 → ℝ),
      (∀ i : TileIndex io.shape1, io.mask1 (s₀.pids 0) (s₀.pids 1) i →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) i) = xs i) →
      (∀ i : TileIndex io.shape2, io.mask2 (s₀.pids 0) (s₀.pids 1) i →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) i) = ys i) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ o : TileIndex io.shapeOut,
            io.writeMask (s₀.pids 0) (s₀.pids 1) o →
            s1.readMemAs outDType io.out (io.write (s₀.pids 0) (s₀.pids 1) o)
              = outDType.ofReal
                  (R.round outDType (f (s₀.pids 0) (s₀.pids 1) xs ys o)))
        ∧ (∀ r o',
            (r ≠ io.out ∨
              ∀ o : TileIndex io.shapeOut,
                io.writeMask (s₀.pids 0) (s₀.pids 1) o →
                o' ≠ io.write (s₀.pids 0) (s₀.pids 1) o) →
            s1.mem r o' = s₀.mem r o')) :
    io.ImplementsR R outDType f := by
  have hcore : io.toU.ImplementsR R (fun _ => outDType)
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁ (fun i => vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shape1 i))
          (fun i => vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape2 i))
          ((TileShape.allIndices io.shapeOut).get j)) := by
    refine UKernelIO.ImplementsR.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      refine hts bounds s ?_ ?_ ?_
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨0, by decide⟩ : Fin 2) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨1, by decide⟩ : Fin 2) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hob (⟨0, by decide⟩ : Fin 1) j
    · intro s₀ vals _hundef hpins
      have hx : ∀ i : TileIndex io.shape1,
          io.mask1 (s₀.pids 0) (s₀.pids 1) i →
          s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) i)
            = vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shape1 i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨0, by decide⟩ : Fin 2) j
      have hy : ∀ i : TileIndex io.shape2,
          io.mask2 (s₀.pids 0) (s₀.pids 1) i →
          s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) i)
            = vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape2 i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨1, by decide⟩ : Fin 2) j
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun i => vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shape1 i))
          (fun i => vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape2 i)) hx hy
      refine ⟨s1, hexec, fun _o j hj => hval _ hj, ?_⟩
      intro r o' hoc _hsc
      have hoc' : ∀ o : TileIndex io.shapeOut,
          io.writeMask (s₀.pids 0) (s₀.pids 1) o →
          r ≠ io.out ∨ o' ≠ io.write (s₀.pids 0) (s₀.pids 1) o :=
        (forall_tileIndex_iff _).mp fun j => hoc (⟨0, by decide⟩ : Fin 1) j
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun o hoact => ?_
        rcases hoc' o hoact with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ h1 h2 h3 xs ys s₀ hp₀ hp₁ hu hx hy
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun j => xs ((TileShape.allIndices io.shape1).get j)
        | ⟨1, _⟩ => fun j => ys ((TileShape.allIndices io.shape2).get j)
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
      s₀ hp₀ hp₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => h1 _ hj
        | ⟨1, _⟩ => fun j hj => h2 _ hj
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
      (fun _o j hj => h3 _ hj) (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hx _ hj
        | ⟨1, _⟩ => fun j hj => hy _ hj
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
  have hxs :
      (fun i => xs ((TileShape.allIndices io.shape1).get (tilePos io.shape1 i)))
        = xs :=
    funext fun i => by rw [get_tilePos]
  have hys :
      (fun i => ys ((TileShape.allIndices io.shape2).get (tilePos io.shape2 i)))
        = ys :=
    funext fun i => by rw [get_tilePos]
  refine ⟨s', hexec, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 1) j hj).trans ?_
    show outDType.ofReal (R.round outDType (f pid₀ pid₁
        (fun i =>
          xs ((TileShape.allIndices io.shape1).get (tilePos io.shape1 i)))
        (fun i =>
          ys ((TileShape.allIndices io.shape2).get (tilePos io.shape2 i)))
        ((TileShape.allIndices io.shapeOut).get j)))
      = outDType.ofReal (R.round outDType
          (f pid₀ pid₁ xs ys ((TileShape.allIndices io.shapeOut).get j)))
    rw [hxs, hys]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | hout
    · exact Or.inl hflat
    · exact Or.inr ⟨fun _o j hj => hout _ hj, fun t => t.elim0⟩

end MaskedTileShapedKernelIO₂


/-! ## Tile-indexed masked IO for an **in-place** update with two aux inputs

Every skin above keeps the output buffer distinct from the inputs — its
allocation contract lists `[inputs…, out]`, so `out = in` is excluded outright.
A rotary-embedding update is the opposite shape: it reads its **main** buffer at
*two* windows, reads two read-only auxiliary buffers, and stores back into the
main buffer at one of those windows.

`InPlaceMaskedTileKernelIO` states exactly that. There are three buffers
(`main`, `aux1`, `aux2`) and **four** read channels — `main` twice plus one per
aux — with the main buffer also the output. Nothing in the core needed changing:
`UKernelIO` never required `obuf` to be disjoint from `ibuf`, and its pins are on
the *initial* state while the readback is on the final one, which is exactly the
right reading for an in-place update — `f`'s arguments are the pre-state values.

All four channels share one `mask` (the usual case: a single tail guard drives
every load), while the store keeps its own `writeMask`. -/

/-- Masked IO for an in-place tile update: `main` is read at two windows and
written at one, `aux1` / `aux2` are read-only. -/
structure InPlaceMaskedTileKernelIO where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- The buffer that is both read and written. -/
  main : RegionName
  /-- First read-only auxiliary buffer. -/
  aux1 : RegionName
  /-- Second read-only auxiliary buffer. -/
  aux2 : RegionName
  /-- The tile footprint each program instance owns. -/
  shape : TileShape
  /-- Lane `i`'s first `main` read address for program `(pid₀, pid₁)`. -/
  readMain1 : Nat → Nat → TileIndex shape → Nat
  /-- Lane `i`'s second `main` read address. -/
  readMain2 : Nat → Nat → TileIndex shape → Nat
  /-- Lane `i`'s `aux1` read address. -/
  readAux1 : Nat → Nat → TileIndex shape → Nat
  /-- Lane `i`'s `aux2` read address. -/
  readAux2 : Nat → Nat → TileIndex shape → Nat
  /-- Lane `i`'s `main` write address. -/
  write : Nat → Nat → TileIndex shape → Nat
  /-- The read-active lanes, shared by all four read channels. -/
  mask : Nat → Nat → TileIndex shape → Prop
  /-- The write-active lanes; defaults to `mask`. -/
  writeMask : Nat → Nat → TileIndex shape → Prop := mask

namespace InPlaceMaskedTileKernelIO

/-- `io.Implements f` — the in-place tile update's Hoare triple. `f` takes the
four channels' **pre-state** lane values in order (`main` at window 1, `main` at
window 2, `aux1`, `aux2`) and both program ids. -/
def Implements (io : InPlaceMaskedTileKernelIO)
    (f : Nat → Nat → (TileIndex io.shape → ℝ) → (TileIndex io.shape → ℝ) →
      (TileIndex io.shape → ℝ) → (TileIndex io.shape → ℝ) →
      TileIndex io.shape → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.main, io.aux1, io.aux2] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ i →
      io.readMain1 pid₀ pid₁ i < A.extent io.main) →
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ i →
      io.readMain2 pid₀ pid₁ i < A.extent io.main) →
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ i →
      io.readAux1 pid₀ pid₁ i < A.extent io.aux1) →
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ i →
      io.readAux2 pid₀ pid₁ i < A.extent io.aux2) →
    (∀ i : TileIndex io.shape, io.writeMask pid₀ pid₁ i →
      io.write pid₀ pid₁ i < A.extent io.main) →
  ∀ (q1 q2 c1 c2 : TileIndex io.shape → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ i →
      s₀.readMem io.main (io.readMain1 pid₀ pid₁ i) = q1 i) →
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ i →
      s₀.readMem io.main (io.readMain2 pid₀ pid₁ i) = q2 i) →
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ i →
      s₀.readMem io.aux1 (io.readAux1 pid₀ pid₁ i) = c1 i) →
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ i →
      s₀.readMem io.aux2 (io.readAux2 pid₀ pid₁ i) = c2 i) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ i : TileIndex io.shape, io.writeMask pid₀ pid₁ i →
          s'.readMem A.flat (A.addr io.main (io.write pid₀ pid₁ i))
            = f pid₀ pid₁ q1 q2 c1 c2 i)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ i : TileIndex io.shape, io.writeMask pid₀ pid₁ i →
              o' ≠ A.addr io.main (io.write pid₀ pid₁ i)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => InPlaceMaskedTileKernelIO.Implements

/-- Embed into the unified core: four float tile channels over three buffers,
one output whose buffer **is** the first two channels' buffer, no scratch. All
four arities coincide, so the per-channel matches need no `Fin` refinement. -/
private def toU (io : InPlaceMaskedTileKernelIO) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 4
  nOut := 1
  nScr := 0
  bufs := [io.main, io.aux1, io.aux2]
  ity := fun _ => .float
  iarity := fun _ => (TileShape.allIndices io.shape).length
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.main
    | ⟨1, _⟩ => io.main
    | ⟨2, _⟩ => io.aux1
    | _ => io.aux2
  oty := fun _ => .float
  oarity := fun _ => (TileShape.allIndices io.shape).length
  obuf := fun _ => io.main
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ _ j => match i with
    | ⟨0, _⟩ => io.readMain1 p₀ p₁ ((TileShape.allIndices io.shape).get j)
    | ⟨1, _⟩ => io.readMain2 p₀ p₁ ((TileShape.allIndices io.shape).get j)
    | ⟨2, _⟩ => io.readAux1 p₀ p₁ ((TileShape.allIndices io.shape).get j)
    | _ => io.readAux2 p₀ p₁ ((TileShape.allIndices io.shape).get j)
  imask := fun _ _ p₀ p₁ _ j =>
    io.mask p₀ p₁ ((TileShape.allIndices io.shape).get j)
  owin := fun _ _ p₀ p₁ _ j =>
    io.write p₀ p₁ ((TileShape.allIndices io.shape).get j)
  omask := fun _ _ p₀ p₁ _ j =>
    io.writeMask p₀ p₁ ((TileShape.allIndices io.shape).get j)
  swin := fun t _ _ _ _ _ => t.elim0
  smask := fun t _ _ _ _ _ => t.elim0

/-- Assembly lemma for the in-place family. -/
theorem Implements.intro (io : InPlaceMaskedTileKernelIO)
    {f : Nat → Nat → (TileIndex io.shape → ℝ) → (TileIndex io.shape → ℝ) →
      (TileIndex io.shape → ℝ) → (TileIndex io.shape → ℝ) →
      TileIndex io.shape → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ i : TileIndex io.shape, io.mask (s.pids 0) (s.pids 1) i →
        io.readMain1 (s.pids 0) (s.pids 1) i < bounds io.main) →
      (∀ i : TileIndex io.shape, io.mask (s.pids 0) (s.pids 1) i →
        io.readMain2 (s.pids 0) (s.pids 1) i < bounds io.main) →
      (∀ i : TileIndex io.shape, io.mask (s.pids 0) (s.pids 1) i →
        io.readAux1 (s.pids 0) (s.pids 1) i < bounds io.aux1) →
      (∀ i : TileIndex io.shape, io.mask (s.pids 0) (s.pids 1) i →
        io.readAux2 (s.pids 0) (s.pids 1) i < bounds io.aux2) →
      (∀ i : TileIndex io.shape, io.writeMask (s.pids 0) (s.pids 1) i →
        io.write (s.pids 0) (s.pids 1) i < bounds io.main) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (q1 q2 c1 c2 : TileIndex io.shape → ℝ),
      (∀ i : TileIndex io.shape, io.mask (s₀.pids 0) (s₀.pids 1) i →
        s₀.readMem io.main (io.readMain1 (s₀.pids 0) (s₀.pids 1) i) = q1 i) →
      (∀ i : TileIndex io.shape, io.mask (s₀.pids 0) (s₀.pids 1) i →
        s₀.readMem io.main (io.readMain2 (s₀.pids 0) (s₀.pids 1) i) = q2 i) →
      (∀ i : TileIndex io.shape, io.mask (s₀.pids 0) (s₀.pids 1) i →
        s₀.readMem io.aux1 (io.readAux1 (s₀.pids 0) (s₀.pids 1) i) = c1 i) →
      (∀ i : TileIndex io.shape, io.mask (s₀.pids 0) (s₀.pids 1) i →
        s₀.readMem io.aux2 (io.readAux2 (s₀.pids 0) (s₀.pids 1) i) = c2 i) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ i : TileIndex io.shape, io.writeMask (s₀.pids 0) (s₀.pids 1) i →
            s1.readMem io.main (io.write (s₀.pids 0) (s₀.pids 1) i)
              = f (s₀.pids 0) (s₀.pids 1) q1 q2 c1 c2 i)
        ∧ (∀ r o',
            (r ≠ io.main ∨
              ∀ i : TileIndex io.shape, io.writeMask (s₀.pids 0) (s₀.pids 1) i →
                o' ≠ io.write (s₀.pids 0) (s₀.pids 1) i) →
            s1.mem r o' = s₀.mem r o')) :
    io.Implements f := by
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁ (fun i => vals (⟨0, by decide⟩ : Fin 4) (tilePos io.shape i))
          (fun i => vals (⟨1, by decide⟩ : Fin 4) (tilePos io.shape i))
          (fun i => vals (⟨2, by decide⟩ : Fin 4) (tilePos io.shape i))
          (fun i => vals (⟨3, by decide⟩ : Fin 4) (tilePos io.shape i))
          ((TileShape.allIndices io.shape).get j)) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      refine hts bounds s ?_ ?_ ?_ ?_ ?_
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨0, by decide⟩ : Fin 4) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨1, by decide⟩ : Fin 4) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨2, by decide⟩ : Fin 4) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨3, by decide⟩ : Fin 4) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hob (⟨0, by decide⟩ : Fin 1) j
    · intro s₀ vals _hundef hpins
      have hp1 : ∀ i : TileIndex io.shape, io.mask (s₀.pids 0) (s₀.pids 1) i →
          s₀.readMem io.main (io.readMain1 (s₀.pids 0) (s₀.pids 1) i)
            = vals (⟨0, by decide⟩ : Fin 4) (tilePos io.shape i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨0, by decide⟩ : Fin 4) j
      have hp2 : ∀ i : TileIndex io.shape, io.mask (s₀.pids 0) (s₀.pids 1) i →
          s₀.readMem io.main (io.readMain2 (s₀.pids 0) (s₀.pids 1) i)
            = vals (⟨1, by decide⟩ : Fin 4) (tilePos io.shape i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨1, by decide⟩ : Fin 4) j
      have hp3 : ∀ i : TileIndex io.shape, io.mask (s₀.pids 0) (s₀.pids 1) i →
          s₀.readMem io.aux1 (io.readAux1 (s₀.pids 0) (s₀.pids 1) i)
            = vals (⟨2, by decide⟩ : Fin 4) (tilePos io.shape i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨2, by decide⟩ : Fin 4) j
      have hp4 : ∀ i : TileIndex io.shape, io.mask (s₀.pids 0) (s₀.pids 1) i →
          s₀.readMem io.aux2 (io.readAux2 (s₀.pids 0) (s₀.pids 1) i)
            = vals (⟨3, by decide⟩ : Fin 4) (tilePos io.shape i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨3, by decide⟩ : Fin 4) j
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun i => vals (⟨0, by decide⟩ : Fin 4) (tilePos io.shape i))
          (fun i => vals (⟨1, by decide⟩ : Fin 4) (tilePos io.shape i))
          (fun i => vals (⟨2, by decide⟩ : Fin 4) (tilePos io.shape i))
          (fun i => vals (⟨3, by decide⟩ : Fin 4) (tilePos io.shape i))
          hp1 hp2 hp3 hp4
      refine ⟨s1, hexec, fun _o j hj => hval _ hj, ?_⟩
      intro r o' hoc _hsc
      have hoc' : ∀ i : TileIndex io.shape,
          io.writeMask (s₀.pids 0) (s₀.pids 1) i →
          r ≠ io.main ∨ o' ≠ io.write (s₀.pids 0) (s₀.pids 1) i :=
        (forall_tileIndex_iff _).mp fun j => hoc (⟨0, by decide⟩ : Fin 1) j
      refine hframe r o' ?_
      by_cases hro : r = io.main
      · subst hro
        refine Or.inr fun i hi => ?_
        rcases hoc' i hi with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ h1 h2 h3 h4 h5 q1 q2 c1 c2 s₀ hp₀ hp₁ hu
    hx1 hx2 hx3 hx4
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun j => q1 ((TileShape.allIndices io.shape).get j)
        | ⟨1, _⟩ => fun j => q2 ((TileShape.allIndices io.shape).get j)
        | ⟨2, _⟩ => fun j => c1 ((TileShape.allIndices io.shape).get j)
        | _ => fun j => c2 ((TileShape.allIndices io.shape).get j))
      s₀ hp₀ hp₁ rfl hu
      -- `ibuf i` occurs in these hypotheses' types, so a catch-all pattern
      -- would leave the channel index unrefined; enumerate all four
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => h1 _ hj
        | ⟨1, _⟩ => fun j hj => h2 _ hj
        | ⟨2, _⟩ => fun j hj => h3 _ hj
        | ⟨3, _⟩ => fun j hj => h4 _ hj
        | ⟨_ + 4, h⟩ => absurd h (by simp only [toU]; omega))
      (fun _o j hj => h5 _ hj) (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hx1 _ hj
        | ⟨1, _⟩ => fun j hj => hx2 _ hj
        | ⟨2, _⟩ => fun j hj => hx3 _ hj
        | ⟨3, _⟩ => fun j hj => hx4 _ hj
        | ⟨_ + 4, h⟩ => absurd h (by simp only [toU]; omega))
  have hround : ∀ g : TileIndex io.shape → ℝ,
      (fun i => g ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        = g :=
    fun g => funext fun i => by rw [get_tilePos]
  refine ⟨s', hexec, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 1) j hj).trans ?_
    show f pid₀ pid₁
        (fun i => q1 ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        (fun i => q2 ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        (fun i => c1 ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        (fun i => c2 ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        ((TileShape.allIndices io.shape).get j)
      = f pid₀ pid₁ q1 q2 c1 c2 ((TileShape.allIndices io.shape).get j)
    rw [hround q1, hround q2, hround c1, hround c2]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | hout
    · exact Or.inl hflat
    · exact Or.inr ⟨fun _o j hj => hout _ hj, fun t => t.elim0⟩

end InPlaceMaskedTileKernelIO


/-! ## Tile-indexed masked IO with a **value + index** output pair

The reduce-with-indices genre (`tl.max(..., return_indices=True)`) writes two
buffers of *different channel types*: a real value and an integer position. Every
skin above has a single `.float` output, so those kernels had no io face at all.

`ValueIndexTileKernelIO` states the pair. The core is already channel-typed —
`UKernelIO.oty` is a function of the output channel and `ChanTy.read_flattenState`
covers every type — so this is another thin wrapper: output 0 is `.float`, output
1 is `.nat` (the store writes a nat-typed cell), and the index buffer is a `Region .int` (coerced into `bufs` by the
name-preserving `CoeOut`).

A bonus over the per-write-map summaries this replaces: those need an explicit
`mid_value ≠ mid_index` hypothesis so the integer store cannot clobber the real
one. Here that is *already* part of the allocation contract (`A.Disjoint` over a
three-region list), so it disappears from the headline. -/

/-- One-input / value-plus-index-output masked tile IO. -/
structure ValueIndexTileKernelIO where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- Input buffer. -/
  inp : RegionName
  /-- The real-valued output buffer. -/
  outVal : RegionName
  /-- The integer-valued output buffer. -/
  outIdx : Region .int
  /-- The tile footprint each program instance owns. -/
  shape : TileShape
  /-- Lane `i`'s read address for program `pid`. -/
  read : Nat → TileIndex shape → Nat
  /-- Lane `i`'s value-output address. -/
  writeVal : Nat → TileIndex shape → Nat
  /-- Lane `i`'s index-output address. -/
  writeIdx : Nat → TileIndex shape → Nat
  /-- Program `pid`'s read-active lanes. -/
  mask : Nat → TileIndex shape → Prop
  /-- Program `pid`'s write-active lanes (shared by both outputs); defaults to
  `mask`. -/
  writeMask : Nat → TileIndex shape → Prop := mask

namespace ValueIndexTileKernelIO

/-- `io.Implements fVal fIdx` — the value/index-pair Hoare triple. The value
output is read back as a real cell, the index output through the operational
`readMemValue .nat` view that `tl.load` / `tl.store` themselves use. -/
def Implements (io : ValueIndexTileKernelIO)
    (fVal : Nat → (TileIndex io.shape → ℝ) → TileIndex io.shape → ℝ)
    (fIdx : Nat → (TileIndex io.shape → ℝ) → TileIndex io.shape → Nat) :
    Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp, io.outVal, (io.outIdx : RegionName)] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    (∀ i : TileIndex io.shape, io.mask pid i →
      io.read pid i < A.extent io.inp) →
    (∀ i : TileIndex io.shape, io.writeMask pid i →
      io.writeVal pid i < A.extent io.outVal) →
    (∀ i : TileIndex io.shape, io.writeMask pid i →
      io.writeIdx pid i < A.extent (io.outIdx : RegionName)) →
  ∀ (xs : TileIndex io.shape → ℝ) (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ i : TileIndex io.shape, io.mask pid i →
      s₀.readMem io.inp (io.read pid i) = xs i) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ i : TileIndex io.shape, io.writeMask pid i →
          s'.readMem A.flat (A.addr io.outVal (io.writeVal pid i))
            = fVal pid xs i)
      ∧ (∀ i : TileIndex io.shape, io.writeMask pid i →
          s'.readMemValue .nat A.flat
              (A.addr (io.outIdx : RegionName) (io.writeIdx pid i))
            = fIdx pid xs i)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ i : TileIndex io.shape, io.writeMask pid i →
                o' ≠ A.addr io.outVal (io.writeVal pid i)) ∧
             (∀ i : TileIndex io.shape, io.writeMask pid i →
                o' ≠ A.addr (io.outIdx : RegionName) (io.writeIdx pid i)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

/-- Embed into the unified core: one float input channel, two output channels of
**different** types, no scratch. -/
private def toU (io : ValueIndexTileKernelIO) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 1
  nOut := 2
  nScr := 0
  bufs := [io.inp, io.outVal, (io.outIdx : RegionName)]
  ity := fun _ => .float
  iarity := fun _ => (TileShape.allIndices io.shape).length
  ibuf := fun _ => io.inp
  oty := fun o => match o with
    | ⟨0, _⟩ => .float
    | _ => .nat
  oarity := fun _ => (TileShape.allIndices io.shape).length
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.outVal
    | _ => (io.outIdx : RegionName)
  obuf_mem := fun o => by
    match o with
    | ⟨0, _⟩ => simp
    | ⟨1, _⟩ => simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun _ _ p₀ _ _ j =>
    io.read p₀ ((TileShape.allIndices io.shape).get j)
  imask := fun _ _ p₀ _ _ j =>
    io.mask p₀ ((TileShape.allIndices io.shape).get j)
  owin := fun o _ p₀ _ _ j => match o with
    | ⟨0, _⟩ => io.writeVal p₀ ((TileShape.allIndices io.shape).get j)
    | _ => io.writeIdx p₀ ((TileShape.allIndices io.shape).get j)
  omask := fun _ _ p₀ _ _ j =>
    io.writeMask p₀ ((TileShape.allIndices io.shape).get j)
  swin := fun t _ _ _ _ _ => t.elim0
  smask := fun t _ _ _ _ _ => t.elim0

/-- Assembly lemma for the value/index family. -/
theorem Implements.intro (io : ValueIndexTileKernelIO)
    {fVal : Nat → (TileIndex io.shape → ℝ) → TileIndex io.shape → ℝ}
    {fIdx : Nat → (TileIndex io.shape → ℝ) → TileIndex io.shape → Nat}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ i : TileIndex io.shape, io.mask s.pid i →
        io.read s.pid i < bounds io.inp) →
      (∀ i : TileIndex io.shape, io.writeMask s.pid i →
        io.writeVal s.pid i < bounds io.outVal) →
      (∀ i : TileIndex io.shape, io.writeMask s.pid i →
        io.writeIdx s.pid i < bounds (io.outIdx : RegionName)) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : TileIndex io.shape → ℝ),
      (∀ i : TileIndex io.shape, io.mask s₀.pid i →
        s₀.readMem io.inp (io.read s₀.pid i) = xs i) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ i : TileIndex io.shape, io.writeMask s₀.pid i →
            s1.readMem io.outVal (io.writeVal s₀.pid i) = fVal s₀.pid xs i)
        ∧ (∀ i : TileIndex io.shape, io.writeMask s₀.pid i →
            s1.readMemValue .nat (io.outIdx : RegionName)
                (io.writeIdx s₀.pid i)
              = fIdx s₀.pid xs i)
        ∧ (∀ r o',
            (∀ i : TileIndex io.shape, io.writeMask s₀.pid i →
              r ≠ io.outVal ∨ o' ≠ io.writeVal s₀.pid i) →
            (∀ i : TileIndex io.shape, io.writeMask s₀.pid i →
              r ≠ (io.outIdx : RegionName) ∨ o' ≠ io.writeIdx s₀.pid i) →
            s1.mem r o' = s₀.mem r o')) :
    io.Implements fVal fIdx := by
  have hcore : io.toU.Implements
      (fun p₀ _p₁ vals o j => match o with
        | ⟨0, _⟩ =>
            fVal p₀ (fun i => vals (⟨0, by decide⟩ : Fin 1) (tilePos io.shape i))
              ((TileShape.allIndices io.shape).get j)
        | ⟨1, _⟩ =>
            fIdx p₀ (fun i => vals (⟨0, by decide⟩ : Fin 1) (tilePos io.shape i))
              ((TileShape.allIndices io.shape).get j)) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      refine hts bounds s ?_ ?_ ?_
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨0, by decide⟩ : Fin 1) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hob (⟨0, by decide⟩ : Fin 2) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hob (⟨1, by decide⟩ : Fin 2) j
    · intro s₀ vals _hundef hpins
      have hx : ∀ i : TileIndex io.shape, io.mask s₀.pid i →
          s₀.readMem io.inp (io.read s₀.pid i)
            = vals (⟨0, by decide⟩ : Fin 1) (tilePos io.shape i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨0, by decide⟩ : Fin 1) j
      obtain ⟨s1, hexec, hvalV, hvalI, hframe⟩ :=
        hrun s₀ (fun i => vals (⟨0, by decide⟩ : Fin 1) (tilePos io.shape i)) hx
      refine ⟨s1, hexec, ?_, ?_⟩
      · intro o j hj
        match o with
        | ⟨0, _⟩ => exact hvalV _ hj
        | ⟨1, _⟩ => exact hvalI _ hj
      · intro r o' hoc _hsc
        refine hframe r o' ?_ ?_
        · exact (forall_tileIndex_iff _).mp fun j =>
            hoc (⟨0, by decide⟩ : Fin 2) j
        · exact (forall_tileIndex_iff _).mp fun j =>
            hoc (⟨1, by decide⟩ : Fin 2) j
  intro A hd hregs hcov pid h1 h2 h3 xs s₀ hpid hu hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
      (fun _ j => xs ((TileShape.allIndices io.shape).get j)) s₀ hpid rfl rfl hu
      (fun _i j hj => h1 _ hj)
      (fun o j hj => match o with
        | ⟨0, _⟩ => h2 _ hj
        | ⟨1, _⟩ => h3 _ hj)
      (fun t => t.elim0) (fun _i j hj => hx _ hj)
  have hxs :
      (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        = xs :=
    funext fun i => by rw [get_tilePos]
  refine ⟨s', hexec, ?_, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 2) j hj).trans ?_
    show fVal pid
        (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        ((TileShape.allIndices io.shape).get j)
      = fVal pid xs ((TileShape.allIndices io.shape).get j)
    rw [hxs]
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨1, by decide⟩ : Fin 2) j hj).trans ?_
    show fIdx pid
        (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        ((TileShape.allIndices io.shape).get j)
      = fIdx pid xs ((TileShape.allIndices io.shape).get j)
    rw [hxs]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | ⟨hv, hi⟩
    · exact Or.inl hflat
    · refine Or.inr ⟨fun o j hj => ?_, fun t => t.elim0⟩
      match o with
      | ⟨0, _⟩ => exact hv _ hj
      | ⟨1, _⟩ => exact hi _ hj

end ValueIndexTileKernelIO

end VeriTile.Triton
