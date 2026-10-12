import VeriTile.Triton.Float.ProfiledRewrite

/-! Interpret a profiled FP derivation in any model of its admitted atoms.
Signatures preserve the numerical domain through symmetry and composition,
so successful executions and failures can both be transported. -/
namespace VeriTile.Triton.FP.ProfiledRewrite

private theorem derivation_sound
    {R : Spec.Assumptions GuardedFragment} {lhs rhs : Program}
    (h : Spec.ProgramDerivation (fun _ _ : Program => False) R lhs rhs) :
    lhs.domain = rhs.domain ∧ Equivalent R lhs rhs := by
  induction h with
  | refl p => exact ⟨rfl, fun _ _ _ _ _ _ _ => rfl⟩
  | «syntax» _ context _ =>
      change _ = _ at context
      cases context
      exact ⟨rfl, fun _ _ _ _ _ _ _ => rfl⟩
  | structural _ proof => cases proof
  | numerical signature proof => exact ⟨congrArg Prod.snd signature, proof⟩
  | symm h ih =>
      refine ⟨ih.1.symm, ?_⟩
      intro α _ M D hM s hd
      exact (ih.2 α M D hM s (by simpa only [ih.1] using hd)).symm
  | trans first second ihFirst ihSecond =>
      refine ⟨ihFirst.1.trans ihSecond.1, ?_⟩
      intro α _ M D hM s hd
      exact (ihFirst.2 α M D hM s hd).trans
        (ihSecond.2 α M D hM s (by simpa only [ihFirst.1] using hd))

/-- A checked derivation gives complete-state execution equality in every
admitted model, under its displayed domain. This is not an IEEE claim. -/
theorem sound {R : Spec.Assumptions GuardedFragment} {lhs rhs : Program}
    (h : Spec.FloatingPoint R lhs rhs) : Equivalent R lhs rhs :=
  (derivation_sound h.derivation).2

end VeriTile.Triton.FP.ProfiledRewrite
