import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Triton.Correctness
import VeriTile.Meta.StatementAudit

open VeriTile.Triton

namespace KernelIOProjectionRegression

-- A real store must not disappear when an unsupported effect follows it.
def blocked : ComputeKernel := .mk [] []
  [.alg (Stmt.store' "victim" [] (Op.constNat 0) (Op.const 42)),
   .effectMarker "tl.async_copy"]

theorem projection_rejects :
    blocked.toAlgorithm? = .error (.requiresEffectProjection "tl.async_copy") := rfl

theorem exact_contract_rejects :
    ¬ ∃ io : KernelIO₁, io.kernel = blocked ∧ io.Implements (fun _ _ => 0) := by
  rintro ⟨io, h, _⟩
  have hp := io.projection
  rw [h, projection_rejects] at hp
  contradiction

theorem rounding_contract_rejects (R : RoundingModel) :
    ¬ ∃ io : KernelIO₂, io.kernel = blocked ∧ io.ImplementsR R (fun _ _ _ => 0) := by
  rintro ⟨io, h, _⟩
  have hp := io.projection
  rw [h, projection_rejects] at hp
  contradiction

theorem equivalence_rejects_left (R : RoundingModel) :
    ¬ ∃ io₁ io₂ : MaskedKernelIO₂, io₁.kernel = blocked ∧ io₁.Equiv io₂ R := by
  rintro ⟨io₁, _, h, _⟩
  have hp := io₁.projection
  rw [h, projection_rejects] at hp
  contradiction

theorem equivalence_rejects_right (R : RoundingModel) :
    ¬ ∃ io₁ io₂ : MaskedKernelIO₂, io₂.kernel = blocked ∧ io₁.Equiv io₂ R := by
  rintro ⟨_, io₂, h, _⟩
  have hp := io₂.projection
  rw [h, projection_rejects] at hp
  contradiction

theorem unified_core_rejects (io : UKernelIO) : io.kernel ≠ blocked := by
  intro h
  have hp := io.projection
  rw [h, projection_rejects] at hp
  contradiction

theorem denotation_rejects (flat out : RegionName) (regs : List (RegionName × Nat))
    (pid offset : Nat) (slots : List DenoteSlot) :
    denoteKernel blocked flat regs pid slots out offset = none := rfl

-- The automatic witness also works for an arbitrary algorithm body.
def supportedIO (body : List Stmt) : KernelIO₁ where
  kernel := ComputeKernel.fromKernelBody ["x"] ["y"] body
  inp := "x"
  out := "y"
  Bin := 1
  Bout := 1
  read := fun _ => 0
  write := fun _ => 0

#axiomsClean exact_contract_rejects
#axiomsClean rounding_contract_rejects
#axiomsClean equivalence_rejects_left
#axiomsClean equivalence_rejects_right
#axiomsClean unified_core_rejects
#axiomsClean denotation_rejects

end KernelIOProjectionRegression
