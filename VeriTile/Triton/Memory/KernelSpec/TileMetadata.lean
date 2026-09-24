/-
Kernel IO contracts: TileMetadata.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.KernelSpec.TileIndex

namespace VeriTile.Triton


/-! ## Tile-indexed masked IO behind **three `.nat` metadata scalars**

The variable-length genre loads a handful of `.nat` scalars first — a segment
length, a source base, a destination base — and *then* uses them in the data
window's address **and** in its mask. None of the tile skins above can express
that: their windows are functions of the program ids alone.

`Meta3MaskedTileKernelIO₁` states it. The core has always allowed
value-dependent windows (`UKernelIO.iwin` takes the whole pinned context, not
just the pids), so this is yet another thin wrapper: three `.nat` channels of
arity 1, one `.float` tile channel whose window and mask are functions of the
three loaded scalars, and one `.float` tile output.

Following the older `Meta*` families, the three scalars are *universally
quantified* in the statement and pinned by the launch state, so the headline
reads as "for whatever the metadata says, the data window is that". -/

/-- One-input / one-output masked tile IO whose windows and mask are functions
of three `.nat` metadata scalars read at the program's own cell. -/
structure Meta3MaskedTileKernelIO₁ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- First `.nat`-read metadata buffer (a plain region; the `.nat` typing lives
  on the channel, not on the region). -/
  mbuf1 : RegionName
  /-- Second `.nat` metadata buffer. -/
  mbuf2 : RegionName
  /-- Third `.nat` metadata buffer. -/
  mbuf3 : RegionName
  /-- Float data input buffer. -/
  inp : RegionName
  /-- Float output buffer. -/
  out : RegionName
  /-- The tile footprint each program instance owns. -/
  shape : TileShape
  /-- First metadata cell's address for program `pid`. -/
  mwin1 : Nat → Nat
  /-- Second metadata cell's address. -/
  mwin2 : Nat → Nat
  /-- Third metadata cell's address. -/
  mwin3 : Nat → Nat
  /-- Lane `i`'s data read address, given the three loaded scalars. -/
  read : Nat → Nat → Nat → Nat → TileIndex shape → Nat
  /-- Lane `i`'s write address, given the three loaded scalars. -/
  write : Nat → Nat → Nat → Nat → TileIndex shape → Nat
  /-- Read-active lanes, given the three loaded scalars. -/
  mask : Nat → Nat → Nat → Nat → TileIndex shape → Prop
  /-- Write-active lanes; defaults to `mask`. -/
  writeMask : Nat → Nat → Nat → Nat → TileIndex shape → Prop := mask

namespace Meta3MaskedTileKernelIO₁

/-- `io.Implements f` — the metadata-driven masked tile triple. The three
scalars are universally quantified and pinned by the launch state; `f` takes
them alongside the program id and the loaded lane values. -/
def Implements (io : Meta3MaskedTileKernelIO₁)
    (f : Nat → Nat → Nat → Nat → (TileIndex io.shape → ℝ) →
      TileIndex io.shape → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.mbuf1, io.mbuf2,
      io.mbuf3, io.inp, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io.mwin1 pid < A.extent io.mbuf1 →
    io.mwin2 pid < A.extent io.mbuf2 →
    io.mwin3 pid < A.extent io.mbuf3 →
  ∀ m1 m2 m3 : Nat,
    (∀ i : TileIndex io.shape, io.mask pid m1 m2 m3 i →
      io.read pid m1 m2 m3 i < A.extent io.inp) →
    (∀ i : TileIndex io.shape, io.writeMask pid m1 m2 m3 i →
      io.write pid m1 m2 m3 i < A.extent io.out) →
  ∀ (xs : TileIndex io.shape → ℝ) (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    s₀.readMemValue .nat io.mbuf1 (io.mwin1 pid) = m1 →
    s₀.readMemValue .nat io.mbuf2 (io.mwin2 pid) = m2 →
    s₀.readMemValue .nat io.mbuf3 (io.mwin3 pid) = m3 →
    (∀ i : TileIndex io.shape, io.mask pid m1 m2 m3 i →
      s₀.readMem io.inp (io.read pid m1 m2 m3 i) = xs i) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ i : TileIndex io.shape, io.writeMask pid m1 m2 m3 i →
          s'.readMem A.flat (A.addr io.out (io.write pid m1 m2 m3 i))
            = f pid m1 m2 m3 xs i)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ i : TileIndex io.shape, io.writeMask pid m1 m2 m3 i →
              o' ≠ A.addr io.out (io.write pid m1 m2 m3 i)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => Meta3MaskedTileKernelIO₁.Implements

/-- Embed into the unified core: three arity-1 `.nat` channels, one `.float`
tile channel whose window and mask read those three, one `.float` output.

Every per-channel match enumerates all four `Fin 4` patterns: the channel type
*and* arity differ across channels, so a catch-all would leave both unreduced. -/
private def toU (io : Meta3MaskedTileKernelIO₁) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 4
  nOut := 1
  nScr := 0
  bufs := [io.mbuf1, io.mbuf2,
    io.mbuf3, io.inp, io.out]
  ity := fun i => match i with
    | ⟨0, _⟩ => .nat
    | ⟨1, _⟩ => .nat
    | ⟨2, _⟩ => .nat
    | _ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => 1
    | ⟨2, _⟩ => 1
    | _ => (TileShape.allIndices io.shape).length
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.mbuf1
    | ⟨1, _⟩ => io.mbuf2
    | ⟨2, _⟩ => io.mbuf3
    | _ => io.inp
  oty := fun _ => .float
  oarity := fun _ => (TileShape.allIndices io.shape).length
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i vals p₀ _ _ => match i with
    | ⟨0, _⟩ => fun _ => io.mwin1 p₀
    | ⟨1, _⟩ => fun _ => io.mwin2 p₀
    | ⟨2, _⟩ => fun _ => io.mwin3 p₀
    | ⟨3, _⟩ => fun j =>
        io.read p₀ (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨2, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
          ((TileShape.allIndices io.shape).get j)
    | ⟨_ + 4, h⟩ => absurd h (by omega)
  imask := fun i vals p₀ _ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => True
    | ⟨2, _⟩ => fun _ => True
    | ⟨3, _⟩ => fun j =>
        io.mask p₀ (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨2, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
          ((TileShape.allIndices io.shape).get j)
    | ⟨_ + 4, h⟩ => absurd h (by omega)
  owin := fun _ vals p₀ _ _ j =>
    io.write p₀ (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
      (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
      (vals (⟨2, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
      ((TileShape.allIndices io.shape).get j)
  omask := fun _ vals p₀ _ _ j =>
    io.writeMask p₀ (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
      (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
      (vals (⟨2, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
      ((TileShape.allIndices io.shape).get j)
  swin := fun t _ _ _ _ _ => t.elim0
  smask := fun t _ _ _ _ _ => t.elim0

/-- Assembly lemma for the three-metadata family. -/
theorem Implements.intro (io : Meta3MaskedTileKernelIO₁)
    {f : Nat → Nat → Nat → Nat → (TileIndex io.shape → ℝ) →
      TileIndex io.shape → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m1 m2 m3 : Nat),
      s.readMemValue .nat io.mbuf1 (io.mwin1 s.pid) = m1 →
      s.readMemValue .nat io.mbuf2 (io.mwin2 s.pid) = m2 →
      s.readMemValue .nat io.mbuf3 (io.mwin3 s.pid) = m3 →
      io.mwin1 s.pid < bounds io.mbuf1 →
      io.mwin2 s.pid < bounds io.mbuf2 →
      io.mwin3 s.pid < bounds io.mbuf3 →
      (∀ i : TileIndex io.shape, io.mask s.pid m1 m2 m3 i →
        io.read s.pid m1 m2 m3 i < bounds io.inp) →
      (∀ i : TileIndex io.shape, io.writeMask s.pid m1 m2 m3 i →
        io.write s.pid m1 m2 m3 i < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (m1 m2 m3 : Nat)
        (xs : TileIndex io.shape → ℝ),
      s₀.readMemValue .nat io.mbuf1 (io.mwin1 s₀.pid) = m1 →
      s₀.readMemValue .nat io.mbuf2 (io.mwin2 s₀.pid) = m2 →
      s₀.readMemValue .nat io.mbuf3 (io.mwin3 s₀.pid) = m3 →
      (∀ i : TileIndex io.shape, io.mask s₀.pid m1 m2 m3 i →
        s₀.readMem io.inp (io.read s₀.pid m1 m2 m3 i) = xs i) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ i : TileIndex io.shape, io.writeMask s₀.pid m1 m2 m3 i →
            s1.readMem io.out (io.write s₀.pid m1 m2 m3 i)
              = f s₀.pid m1 m2 m3 xs i)
        ∧ (∀ r o',
            (r ≠ io.out ∨
              ∀ i : TileIndex io.shape, io.writeMask s₀.pid m1 m2 m3 i →
                o' ≠ io.write s₀.pid m1 m2 m3 i) →
            s1.mem r o' = s₀.mem r o')) :
    io.Implements f := by
  have hcore : io.toU.Implements
      (fun p₀ _p₁ vals _o j =>
        f p₀ (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨2, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
          (fun i => vals (⟨3, by decide⟩ : Fin 4) (tilePos io.shape i))
          ((TileShape.allIndices io.shape).get j)) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib hob _hsb
      refine hts bounds s _ _ _
        (hpins (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨2, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨2, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1) trivial) ?_ ?_
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨3, by decide⟩ : Fin 4) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hob (⟨0, by decide⟩ : Fin 1) j
    · intro s₀ vals _hundef hpins
      have hx : ∀ i : TileIndex io.shape,
          io.mask s₀.pid (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
            (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
            (vals (⟨2, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1)) i →
          s₀.readMem io.inp
              (io.read s₀.pid
                (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
                (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
                (vals (⟨2, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1)) i)
            = vals (⟨3, by decide⟩ : Fin 4) (tilePos io.shape i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨3, by decide⟩ : Fin 4) j
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ _ _ _
          (fun i => vals (⟨3, by decide⟩ : Fin 4) (tilePos io.shape i))
          (hpins (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1) trivial)
          (hpins (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1) trivial)
          (hpins (⟨2, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1) trivial) hx
      refine ⟨s1, hexec, fun _o j hj => hval _ hj, ?_⟩
      intro r o' hoc _hsc
      have hoc' : ∀ i : TileIndex io.shape,
          io.writeMask s₀.pid
            (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
            (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
            (vals (⟨2, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1)) i →
          r ≠ io.out ∨ o' ≠ io.write s₀.pid
            (vals (⟨0, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
            (vals (⟨1, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1))
            (vals (⟨2, by decide⟩ : Fin 4) (⟨0, by decide⟩ : Fin 1)) i :=
        (forall_tileIndex_iff _).mp fun j => hoc (⟨0, by decide⟩ : Fin 1) j
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun i hi => ?_
        rcases hoc' i hi with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid hb1 hb2 hb3 m1 m2 m3 h1 h2 xs s₀ hpid hu hm1 hm2 hm3
    hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ => m1
        | ⟨1, _⟩ => fun _ => m2
        | ⟨2, _⟩ => fun _ => m3
        | ⟨3, _⟩ => fun j => xs ((TileShape.allIndices io.shape).get j)
        | ⟨_ + 4, h⟩ => absurd h (by simp only [toU]; omega))
      s₀ hpid rfl rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hb1
        | ⟨1, _⟩ => fun _ _ => hb2
        | ⟨2, _⟩ => fun _ _ => hb3
        | ⟨3, _⟩ => fun j hj => h1 _ hj
        | ⟨_ + 4, h⟩ => absurd h (by simp only [toU]; omega))
      (fun _o j hj => h2 _ hj) (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hm1
        | ⟨1, _⟩ => fun _ _ => hm2
        | ⟨2, _⟩ => fun _ _ => hm3
        | ⟨3, _⟩ => fun j hj => hx _ hj
        | ⟨_ + 4, h⟩ => absurd h (by simp only [toU]; omega))
  have hxs :
      (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        = xs :=
    funext fun i => by rw [get_tilePos]
  refine ⟨s', hexec, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 1) j hj).trans ?_
    show f pid m1 m2 m3
        (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        ((TileShape.allIndices io.shape).get j)
      = f pid m1 m2 m3 xs ((TileShape.allIndices io.shape).get j)
    rw [hxs]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | hout
    · exact Or.inl hflat
    · exact Or.inr ⟨fun _o j hj => hout _ hj, fun t => t.elim0⟩

end Meta3MaskedTileKernelIO₁


/-! ## Tile-indexed masked IO driven by **one** metadata scalar

`Meta3MaskedTileKernelIO₁` states the metadata-driven genre at three `.nat`
scalars. A paged-KV-cache writeback needs exactly one: a single `block_off` cell
selects the destination page, and every store address is built from it. Listing
two phantom metadata buffers to reuse the three-scalar skin would put regions in
the allocation contract that the kernel never touches, so the honest statement is
the one-scalar narrowing. -/

/-- One `.nat` metadata scalar, one float tile input, one float tile output. -/
structure Meta1MaskedTileKernelIO₁ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- The `.nat`-read metadata buffer (a plain region; the `.nat` typing lives on
  the channel, not on the region). -/
  mbuf : RegionName
  /-- Float data input buffer. -/
  inp : RegionName
  /-- Float output buffer. -/
  out : RegionName
  /-- The tile footprint each program instance owns. -/
  shape : TileShape
  /-- The metadata cell's address for program `pid`. -/
  mwin : Nat → Nat
  /-- Lane `i`'s data read address, given the loaded scalar. -/
  read : Nat → Nat → TileIndex shape → Nat
  /-- Lane `i`'s write address, given the loaded scalar. -/
  write : Nat → Nat → TileIndex shape → Nat
  /-- Read-active lanes, given the loaded scalar. -/
  mask : Nat → Nat → TileIndex shape → Prop
  /-- Write-active lanes; defaults to `mask`. -/
  writeMask : Nat → Nat → TileIndex shape → Prop := mask

namespace Meta1MaskedTileKernelIO₁

/-- `io.Implements f` — the one-metadata narrowing of
`Meta3MaskedTileKernelIO₁.Implements`. The scalar is universally quantified and
pinned by the launch state; `f` takes it alongside the program id and the loaded
lane values. -/
def Implements (io : Meta1MaskedTileKernelIO₁)
    (f : Nat → Nat → (TileIndex io.shape → ℝ) → TileIndex io.shape → ℝ) :
    Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.mbuf, io.inp, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io.mwin pid < A.extent io.mbuf →
  ∀ m : Nat,
    (∀ i : TileIndex io.shape, io.mask pid m i →
      io.read pid m i < A.extent io.inp) →
    (∀ i : TileIndex io.shape, io.writeMask pid m i →
      io.write pid m i < A.extent io.out) →
  ∀ (xs : TileIndex io.shape → ℝ) (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    s₀.readMemValue .nat io.mbuf (io.mwin pid) = m →
    (∀ i : TileIndex io.shape, io.mask pid m i →
      s₀.readMem io.inp (io.read pid m i) = xs i) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ i : TileIndex io.shape, io.writeMask pid m i →
          s'.readMem A.flat (A.addr io.out (io.write pid m i))
            = f pid m xs i)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ i : TileIndex io.shape, io.writeMask pid m i →
              o' ≠ A.addr io.out (io.write pid m i)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => Meta1MaskedTileKernelIO₁.Implements

/-- Embed into the unified core: one arity-1 `.nat` channel, one `.float` tile
channel whose window and mask read it, one `.float` output. Both per-channel
matches enumerate every `Fin 2` pattern — type *and* arity differ across the two
channels, so a catch-all would leave both unreduced. -/
private def toU (io : Meta1MaskedTileKernelIO₁) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 2
  nOut := 1
  nScr := 0
  bufs := [io.mbuf, io.inp, io.out]
  ity := fun i => match i with
    | ⟨0, _⟩ => .nat
    | _ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => 1
    | _ => (TileShape.allIndices io.shape).length
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.mbuf
    | _ => io.inp
  oty := fun _ => .float
  oarity := fun _ => (TileShape.allIndices io.shape).length
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i vals p₀ _ _ => match i with
    | ⟨0, _⟩ => fun _ => io.mwin p₀
    | ⟨1, _⟩ => fun j =>
        io.read p₀ (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1))
          ((TileShape.allIndices io.shape).get j)
    | ⟨_ + 2, h⟩ => absurd h (by omega)
  imask := fun i vals p₀ _ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun j =>
        io.mask p₀ (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1))
          ((TileShape.allIndices io.shape).get j)
    | ⟨_ + 2, h⟩ => absurd h (by omega)
  owin := fun _ vals p₀ _ _ j =>
    io.write p₀ (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1))
      ((TileShape.allIndices io.shape).get j)
  omask := fun _ vals p₀ _ _ j =>
    io.writeMask p₀ (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1))
      ((TileShape.allIndices io.shape).get j)
  swin := fun t _ _ _ _ _ => t.elim0
  smask := fun t _ _ _ _ _ => t.elim0

/-- Assembly lemma for the one-metadata family. -/
theorem Implements.intro (io : Meta1MaskedTileKernelIO₁)
    {f : Nat → Nat → (TileIndex io.shape → ℝ) → TileIndex io.shape → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m : Nat),
      s.readMemValue .nat io.mbuf (io.mwin s.pid) = m →
      io.mwin s.pid < bounds io.mbuf →
      (∀ i : TileIndex io.shape, io.mask s.pid m i →
        io.read s.pid m i < bounds io.inp) →
      (∀ i : TileIndex io.shape, io.writeMask s.pid m i →
        io.write s.pid m i < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (m : Nat) (xs : TileIndex io.shape → ℝ),
      s₀.readMemValue .nat io.mbuf (io.mwin s₀.pid) = m →
      (∀ i : TileIndex io.shape, io.mask s₀.pid m i →
        s₀.readMem io.inp (io.read s₀.pid m i) = xs i) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ i : TileIndex io.shape, io.writeMask s₀.pid m i →
            s1.readMem io.out (io.write s₀.pid m i) = f s₀.pid m xs i)
        ∧ (∀ r o',
            (r ≠ io.out ∨
              ∀ i : TileIndex io.shape, io.writeMask s₀.pid m i →
                o' ≠ io.write s₀.pid m i) →
            s1.mem r o' = s₀.mem r o')) :
    io.Implements f := by
  have hcore : io.toU.Implements
      (fun p₀ _p₁ vals _o j =>
        f p₀ (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1))
          (fun i => vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape i))
          ((TileShape.allIndices io.shape).get j)) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib hob _hsb
      refine hts bounds s _
        (hpins (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1) trivial) ?_ ?_
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨1, by decide⟩ : Fin 2) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hob (⟨0, by decide⟩ : Fin 1) j
    · intro s₀ vals _hundef hpins
      have hx : ∀ i : TileIndex io.shape,
          io.mask s₀.pid (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1))
            i →
          s₀.readMem io.inp
              (io.read s₀.pid
                (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1)) i)
            = vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨1, by decide⟩ : Fin 2) j
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ _ (fun i => vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape i))
          (hpins (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1) trivial) hx
      refine ⟨s1, hexec, fun _o j hj => hval _ hj, ?_⟩
      intro r o' hoc _hsc
      have hoc' : ∀ i : TileIndex io.shape,
          io.writeMask s₀.pid
            (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1)) i →
          r ≠ io.out ∨ o' ≠ io.write s₀.pid
            (vals (⟨0, by decide⟩ : Fin 2) (⟨0, by decide⟩ : Fin 1)) i :=
        (forall_tileIndex_iff _).mp fun j => hoc (⟨0, by decide⟩ : Fin 1) j
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun i hi => ?_
        rcases hoc' i hi with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid hbm m h1 h2 xs s₀ hpid hu hm hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ => m
        | ⟨1, _⟩ => fun j => xs ((TileShape.allIndices io.shape).get j)
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
      s₀ hpid rfl rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hbm
        | ⟨1, _⟩ => fun j hj => h1 _ hj
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
      (fun _o j hj => h2 _ hj) (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hm
        | ⟨1, _⟩ => fun j hj => hx _ hj
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
  have hxs :
      (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        = xs :=
    funext fun i => by rw [get_tilePos]
  refine ⟨s', hexec, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 1) j hj).trans ?_
    show f pid m
        (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        ((TileShape.allIndices io.shape).get j)
      = f pid m xs ((TileShape.allIndices io.shape).get j)
    rw [hxs]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | hout
    · exact Or.inl hflat
    · exact Or.inr ⟨fun _o j hj => hout _ hj, fun t => t.elim0⟩

end Meta1MaskedTileKernelIO₁


/-! ## One metadata scalar, two per-channel-shape inputs, two program axes

An attention epilogue divides an accumulator tile by a per-row denominator and
stores it under a mask read from a *loaded* sequence length: one `.nat` metadata
scalar, two float channels with **different** shapes (`[BM, BD]` and `[BM]`), one
output, two program axes. That is `Meta1MaskedTileKernelIO₁` ⊕
`Masked3DTileShapedKernelIO₂` — again a pure wrapper composition. -/

/-- One `.nat` metadata scalar, two float tile inputs with independent shapes,
one float tile output, over two program axes. -/
structure Meta1MaskedTileShapedKernelIO₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- The `.nat` metadata buffer. -/
  mbuf : RegionName
  /-- First float input buffer. -/
  in1 : RegionName
  /-- Second float input buffer. -/
  in2 : RegionName
  /-- Float output buffer. -/
  out : RegionName
  /-- First input's tile shape. -/
  shape1 : TileShape
  /-- Second input's tile shape. -/
  shape2 : TileShape
  /-- Output's tile shape. -/
  shapeOut : TileShape
  /-- The metadata cell's address for program `(pid₀, pid₁)`. -/
  mwin : Nat → Nat → Nat
  /-- Lane `i`'s `in1` read address, given the loaded scalar. -/
  read1 : Nat → Nat → Nat → TileIndex shape1 → Nat
  /-- Lane `i`'s `in2` read address, given the loaded scalar. -/
  read2 : Nat → Nat → Nat → TileIndex shape2 → Nat
  /-- Lane `o`'s write address, given the loaded scalar. -/
  write : Nat → Nat → Nat → TileIndex shapeOut → Nat
  /-- `in1`'s read-active lanes, given the loaded scalar. -/
  mask1 : Nat → Nat → Nat → TileIndex shape1 → Prop
  /-- `in2`'s read-active lanes, given the loaded scalar. -/
  mask2 : Nat → Nat → Nat → TileIndex shape2 → Prop
  /-- The output's write-active lanes, given the loaded scalar. -/
  writeMask : Nat → Nat → Nat → TileIndex shapeOut → Prop

namespace Meta1MaskedTileShapedKernelIO₂

/-- `io.Implements f` — the metadata-driven two-input Hoare triple. The scalar is
universally quantified and pinned by the launch state; every window and mask, and
`f` itself, may read it. -/
def Implements (io : Meta1MaskedTileShapedKernelIO₂)
    (f : Nat → Nat → Nat → (TileIndex io.shape1 → ℝ) →
      (TileIndex io.shape2 → ℝ) → TileIndex io.shapeOut → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.mbuf, io.in1, io.in2, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
    io.mwin pid₀ pid₁ < A.extent io.mbuf →
  ∀ m : Nat,
    (∀ i : TileIndex io.shape1, io.mask1 pid₀ pid₁ m i →
      io.read1 pid₀ pid₁ m i < A.extent io.in1) →
    (∀ i : TileIndex io.shape2, io.mask2 pid₀ pid₁ m i →
      io.read2 pid₀ pid₁ m i < A.extent io.in2) →
    (∀ o : TileIndex io.shapeOut, io.writeMask pid₀ pid₁ m o →
      io.write pid₀ pid₁ m o < A.extent io.out) →
  ∀ (xs : TileIndex io.shape1 → ℝ) (ys : TileIndex io.shape2 → ℝ)
    (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    s₀.readMemValue .nat io.mbuf (io.mwin pid₀ pid₁) = m →
    (∀ i : TileIndex io.shape1, io.mask1 pid₀ pid₁ m i →
      s₀.readMem io.in1 (io.read1 pid₀ pid₁ m i) = xs i) →
    (∀ i : TileIndex io.shape2, io.mask2 pid₀ pid₁ m i →
      s₀.readMem io.in2 (io.read2 pid₀ pid₁ m i) = ys i) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ o : TileIndex io.shapeOut, io.writeMask pid₀ pid₁ m o →
          s'.readMem A.flat (A.addr io.out (io.write pid₀ pid₁ m o))
            = f pid₀ pid₁ m xs ys o)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ o : TileIndex io.shapeOut, io.writeMask pid₀ pid₁ m o →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ m o)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " =>
  Meta1MaskedTileShapedKernelIO₂.Implements

/-- Embed into the unified core: one arity-1 `.nat` channel and two `.float` tile
channels with independent arities, one `.float` output. Every per-channel match
enumerates all four `Fin 3` patterns — type *and* arity vary. -/
private def toU (io : Meta1MaskedTileShapedKernelIO₂) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 3
  nOut := 1
  nScr := 0
  bufs := [io.mbuf, io.in1, io.in2, io.out]
  ity := fun i => match i with
    | ⟨0, _⟩ => .nat
    | _ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => (TileShape.allIndices io.shape1).length
    | ⟨2, _⟩ => (TileShape.allIndices io.shape2).length
    | ⟨_ + 3, h⟩ => absurd h (by omega)
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.mbuf
    | ⟨1, _⟩ => io.in1
    | _ => io.in2
  oty := fun _ => .float
  oarity := fun _ => (TileShape.allIndices io.shapeOut).length
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => io.mwin p₀ p₁
    | ⟨1, _⟩ => fun j =>
        io.read1 p₀ p₁ (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
          ((TileShape.allIndices io.shape1).get j)
    | ⟨2, _⟩ => fun j =>
        io.read2 p₀ p₁ (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
          ((TileShape.allIndices io.shape2).get j)
    | ⟨_ + 3, h⟩ => absurd h (by omega)
  imask := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun j =>
        io.mask1 p₀ p₁ (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
          ((TileShape.allIndices io.shape1).get j)
    | ⟨2, _⟩ => fun j =>
        io.mask2 p₀ p₁ (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
          ((TileShape.allIndices io.shape2).get j)
    | ⟨_ + 3, h⟩ => absurd h (by omega)
  owin := fun _ vals p₀ p₁ _ j =>
    io.write p₀ p₁ (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
      ((TileShape.allIndices io.shapeOut).get j)
  omask := fun _ vals p₀ p₁ _ j =>
    io.writeMask p₀ p₁ (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
      ((TileShape.allIndices io.shapeOut).get j)
  swin := fun t _ _ _ _ _ => t.elim0
  smask := fun t _ _ _ _ _ => t.elim0

/-- Assembly lemma for the metadata-driven two-input family. -/
theorem Implements.intro (io : Meta1MaskedTileShapedKernelIO₂)
    {f : Nat → Nat → Nat → (TileIndex io.shape1 → ℝ) →
      (TileIndex io.shape2 → ℝ) → TileIndex io.shapeOut → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m : Nat),
      s.readMemValue .nat io.mbuf (io.mwin (s.pids 0) (s.pids 1)) = m →
      io.mwin (s.pids 0) (s.pids 1) < bounds io.mbuf →
      (∀ i : TileIndex io.shape1, io.mask1 (s.pids 0) (s.pids 1) m i →
        io.read1 (s.pids 0) (s.pids 1) m i < bounds io.in1) →
      (∀ i : TileIndex io.shape2, io.mask2 (s.pids 0) (s.pids 1) m i →
        io.read2 (s.pids 0) (s.pids 1) m i < bounds io.in2) →
      (∀ o : TileIndex io.shapeOut, io.writeMask (s.pids 0) (s.pids 1) m o →
        io.write (s.pids 0) (s.pids 1) m o < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (m : Nat) (xs : TileIndex io.shape1 → ℝ)
        (ys : TileIndex io.shape2 → ℝ),
      s₀.readMemValue .nat io.mbuf (io.mwin (s₀.pids 0) (s₀.pids 1)) = m →
      (∀ i : TileIndex io.shape1, io.mask1 (s₀.pids 0) (s₀.pids 1) m i →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) m i) = xs i) →
      (∀ i : TileIndex io.shape2, io.mask2 (s₀.pids 0) (s₀.pids 1) m i →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) m i) = ys i) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ o : TileIndex io.shapeOut,
            io.writeMask (s₀.pids 0) (s₀.pids 1) m o →
            s1.readMem io.out (io.write (s₀.pids 0) (s₀.pids 1) m o)
              = f (s₀.pids 0) (s₀.pids 1) m xs ys o)
        ∧ (∀ r o',
            (r ≠ io.out ∨
              ∀ o : TileIndex io.shapeOut,
                io.writeMask (s₀.pids 0) (s₀.pids 1) m o →
                o' ≠ io.write (s₀.pids 0) (s₀.pids 1) m o) →
            s1.mem r o' = s₀.mem r o')) :
    io.Implements f := by
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁ (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1))
          (fun i => vals (⟨1, by decide⟩ : Fin 3) (tilePos io.shape1 i))
          (fun i => vals (⟨2, by decide⟩ : Fin 3) (tilePos io.shape2 i))
          ((TileShape.allIndices io.shapeOut).get j)) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib hob _hsb
      refine hts bounds s _
        (hpins (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial) ?_ ?_ ?_
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨1, by decide⟩ : Fin 3) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨2, by decide⟩ : Fin 3) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hob (⟨0, by decide⟩ : Fin 1) j
    · intro s₀ vals _hundef hpins
      have hx : ∀ i : TileIndex io.shape1,
          io.mask1 (s₀.pids 0) (s₀.pids 1)
            (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) i →
          s₀.readMem io.in1
              (io.read1 (s₀.pids 0) (s₀.pids 1)
                (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) i)
            = vals (⟨1, by decide⟩ : Fin 3) (tilePos io.shape1 i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨1, by decide⟩ : Fin 3) j
      have hy : ∀ i : TileIndex io.shape2,
          io.mask2 (s₀.pids 0) (s₀.pids 1)
            (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) i →
          s₀.readMem io.in2
              (io.read2 (s₀.pids 0) (s₀.pids 1)
                (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) i)
            = vals (⟨2, by decide⟩ : Fin 3) (tilePos io.shape2 i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨2, by decide⟩ : Fin 3) j
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ _ (fun i => vals (⟨1, by decide⟩ : Fin 3) (tilePos io.shape1 i))
          (fun i => vals (⟨2, by decide⟩ : Fin 3) (tilePos io.shape2 i))
          (hpins (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1) trivial) hx hy
      refine ⟨s1, hexec, fun _o j hj => hval _ hj, ?_⟩
      intro r o' hoc _hsc
      have hoc' : ∀ o : TileIndex io.shapeOut,
          io.writeMask (s₀.pids 0) (s₀.pids 1)
            (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) o →
          r ≠ io.out ∨ o' ≠ io.write (s₀.pids 0) (s₀.pids 1)
            (vals (⟨0, by decide⟩ : Fin 3) (⟨0, by decide⟩ : Fin 1)) o :=
        (forall_tileIndex_iff _).mp fun j => hoc (⟨0, by decide⟩ : Fin 1) j
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun o hoact => ?_
        rcases hoc' o hoact with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ hbm m hb1 hb2 hbo xs ys s₀ hp₀ hp₁ hu hm hx hy
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ => m
        | ⟨1, _⟩ => fun j => xs ((TileShape.allIndices io.shape1).get j)
        | ⟨2, _⟩ => fun j => ys ((TileShape.allIndices io.shape2).get j)
        | ⟨_ + 3, h⟩ => absurd h (by simp only [toU]; omega))
      s₀ hp₀ hp₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hbm
        | ⟨1, _⟩ => fun j hj => hb1 _ hj
        | ⟨2, _⟩ => fun j hj => hb2 _ hj
        | ⟨_ + 3, h⟩ => absurd h (by simp only [toU]; omega))
      (fun _o j hj => hbo _ hj) (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hm
        | ⟨1, _⟩ => fun j hj => hx _ hj
        | ⟨2, _⟩ => fun j hj => hy _ hj
        | ⟨_ + 3, h⟩ => absurd h (by simp only [toU]; omega))
  have hxs :
      (fun i => xs ((TileShape.allIndices io.shape1).get (tilePos io.shape1 i)))
        = xs := funext fun i => by rw [get_tilePos]
  have hys :
      (fun i => ys ((TileShape.allIndices io.shape2).get (tilePos io.shape2 i)))
        = ys := funext fun i => by rw [get_tilePos]
  refine ⟨s', hexec, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 1) j hj).trans ?_
    show f pid₀ pid₁ m
        (fun i =>
          xs ((TileShape.allIndices io.shape1).get (tilePos io.shape1 i)))
        (fun i =>
          ys ((TileShape.allIndices io.shape2).get (tilePos io.shape2 i)))
        ((TileShape.allIndices io.shapeOut).get j)
      = f pid₀ pid₁ m xs ys ((TileShape.allIndices io.shapeOut).get j)
    rw [hxs, hys]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | hout
    · exact Or.inl hflat
    · exact Or.inr ⟨fun _o j hj => hout _ hj, fun t => t.elim0⟩

end Meta1MaskedTileShapedKernelIO₂

end VeriTile.Triton
