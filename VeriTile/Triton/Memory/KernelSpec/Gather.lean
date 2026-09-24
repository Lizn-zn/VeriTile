/-
Kernel IO contracts: Gather.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.KernelSpec.Base

namespace VeriTile.Triton

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
  projection := io.projection
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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  projection := io.projection
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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  projection := io.projection
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

end VeriTile.Triton
