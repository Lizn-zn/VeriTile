import VeriTile.Triton.Memory.KernelSpec.MaskedND
import VeriTile.Meta.StatementAudit
namespace VeriTile.Triton.Masked2DKernelIO₂
/-! One integer input stream, one real scale stream and one real output.
Successful flat execution and the full tagged-cell frame remain explicit.
The integer input is read through its actual typed load channel. -/
/-- Integer weight inputs, real scale inputs, successful flat execution and full frame. -/
def ImplementsInt (io : Masked2DKernelIO₂)
    (f : Nat → Nat → (Fin io.B → Int) → (Fin io.B → ℝ) → Fin io.B → ℝ) :
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
  ∀ (xs : Fin io.B → Int) (ys : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMemValue .int io.in1 (io.read1 pid₀ pid₁ j) = xs j) →
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

@[inherit_doc] scoped infix:25 " ⊨ᵢ " => Masked2DKernelIO₂.ImplementsInt

/-- Integer weights and real scales, with the same finite IO and frame. -/
def toIntU (io : Masked2DKernelIO₂) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 2
  nOut := 1
  nScr := io.scratch.length
  bufs := [io.in1, io.in2, io.out] ++ io.scratch.map Prod.fst
  ity := fun i => match i with
    | ⟨0, _⟩ => .int
    | _ => .float
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

/-- Lift exact region execution to complete finite flat IO. -/
theorem ImplementsInt.intro (io : Masked2DKernelIO₂)
    {f : Nat → Nat → (Fin io.B → Int) → (Fin io.B → ℝ) → Fin io.B → ℝ}
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
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.B → Int) (ys : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMemValue .int io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) j) = xs j) →
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
    io.ImplementsInt f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toIntU.Implements
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
        | ⟨_+1, _⟩ => ys)
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


#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Triton.Masked2DKernelIO₂
