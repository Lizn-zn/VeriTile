import VeriTile.Triton.Memory.KernelSpec.TileND

/-! Assembly for a three-axis tile contract that uses its existing zero-oracle
premise in both execution and safety. This is a conditional model contract;
it does not assert that a Triton implementation supplies zero padding. -/
namespace VeriTile.Triton.Masked3DTileKernelIO₁

theorem Implements.intro_undef (io : Masked3DTileKernelIO₁)
    {f : Nat → Nat → (TileIndex io.shape → ℝ) → TileIndex io.shape → ℝ}
    (hok : io.kernel.toAlgKernel.FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      s.undef = (fun _ _ => 0) →
      (∀ i : TileIndex io.shape,
        io.mask (s.pids 0) (s.pids 1) (s.pids 2) i →
        io.read (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.inp) →
      (∀ i : TileIndex io.shape,
        io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) i →
        io.write (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.out) →
      io.kernel.toAlgKernel.TraceSafe bounds s)
    (hrun : ∀ (s : BlockState) (xs : TileIndex io.shape → ℝ),
      s.undef = (fun _ _ => 0) →
      (∀ i : TileIndex io.shape,
        io.mask (s.pids 0) (s.pids 1) (s.pids 2) i →
        s.readMem io.inp (io.read (s.pids 0) (s.pids 1) (s.pids 2) i) = xs i) →
      ∃ v, exec io.kernel.toAlgKernel s = some v ∧
        (∀ i : TileIndex io.shape,
          io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) i →
          v.readMem io.out (io.write (s.pids 0) (s.pids 1) (s.pids 2) i) =
            f (s.pids 0) (s.pids 1) xs i) ∧
        (∀ r off,
          (r ≠ io.out ∨ ∀ i : TileIndex io.shape,
            io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) i →
            off ≠ io.write (s.pids 0) (s.pids 1) (s.pids 2) i) →
          v.mem r off = s.mem r off)) : io.Implements f := by
  intro A hd hr hc p₀ p₁ p₂ hb hw xs s hp₀ hp₁ hp₂ hu hx
  subst p₀; subst p₁; subst p₂
  obtain ⟨v,he,hv,hf⟩ := hrun s xs hu hx
  have hs := hts A.extent s hu hb hw
  refine ⟨A.flattenState v, ?_, ?_, ?_⟩
  · rw [A.exec_flatten hd hc _ s hs hok hu, he, Option.map_some]
  · intro i hi
    exact (ChanTy.read_flattenState .float A hd v (by simp [hr]) (hw i hi)).trans (hv i hi)
  · intro r off hn
    by_cases hrflat : r = A.flat
    · subst r
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode off with
      | none => rfl
      | some cell =>
        obtain ⟨r',off'⟩ := cell
        obtain ⟨_,heq,_⟩ := A.decode_sound hdec
        change A.trCell (v.mem r' off') = A.trCell (s.mem r' off')
        apply congrArg A.trCell
        apply hf
        by_cases hro : r' = io.out
        · subst r'
          right
          intro i hi hoff
          rcases hn with hbad | hbad
          · exact hbad rfl
          · exact hbad i hi (by rw [heq, hoff])
        · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hrflat]

end VeriTile.Triton.Masked3DTileKernelIO₁
