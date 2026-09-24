import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Triton.Correctness
import VeriTile.Triton.Float.Pipeline
import VeriTile.Triton.KernelLemmas.CarryFold
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

-- Public helper transformations must retain the rejection, too.
theorem erasure_rejects :
    blocked.eraseDType.toAlgorithm? = .error (.requiresEffectProjection "tl.async_copy") := by
  simp [ComputeKernel.eraseDType, projection_rejects]

def supported : ComputeKernel := .fromKernelBody [] [] []

theorem composition_rejects :
    (ComputeKernel.seq [] [] [supported, blocked, supported]).toAlgorithm? =
      .error (.requiresEffectProjection "tl.async_copy") := rfl

theorem transformed_contract_rejects :
    ¬ ∃ io : KernelIO₁, io.kernel = blocked.eraseDType := by
  rintro ⟨io, h⟩
  have hp := io.projection
  rw [h, erasure_rejects] at hp
  contradiction

theorem composed_contract_rejects :
    ¬ ∃ io : UKernelIO,
      io.kernel = ComputeKernel.seq [] [] [supported, blocked, supported] := by
  rintro ⟨io, h⟩
  have hp := io.projection
  rw [h, composition_rejects] at hp
  contradiction

theorem pipeline_rejects (R : RoundingModel) (s : BlockState) :
    execPipelineR R [supported, blocked, supported] s = none := by
  simp [execPipelineR, ComputeKernel.evalR, supported, projection_rejects,
    execR, stepStmtsR]

theorem chain_rejects (s : BlockState) :
    execChain [supported, blocked, supported] s = none := by
  simp [execChain, ComputeKernel.eval, supported, projection_rejects, exec]

-- Success paths must still execute the store, not merely return some state.
def store42 : ComputeKernel := .fromKernelBody [] ["victim"]
  [Stmt.store' "victim" [] (Op.constNat 0) (Op.const 42)]

theorem erased_store_projects : store42.eraseDType.toAlgorithm? = .ok store42.toAlgKernel := by
  simp [ComputeKernel.eraseDType, store42, Kernel.eraseDType,
    Stmt.store', Stmt.eraseDTypeList, Op.eraseDType, VeriTile.Triton.eraseDType]

theorem composed_store_projects :
    (ComputeKernel.seq [] ["victim"] [supported, store42, supported]).toAlgorithm? =
      .ok store42.toAlgKernel := rfl

theorem chain_executes_store (s : BlockState) :
    ∃ s', execChain [store42] s = some s' ∧ s'.readMem "victim" 0 = 42 := by
  simp [execChain, ComputeKernel.eval, store42, exec, stepStmts, stepStmt,
    Stmt.store', BlockState.readMem, BlockState.writeMem]

theorem pipeline_executes_store (s : BlockState) :
    ∃ s', execPipelineR .triv [store42] s = some s' ∧ s'.readMem "victim" 0 = 42 := by
  simpa [execPipelineR, execChain] using chain_executes_store s

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
#axiomsClean erasure_rejects
#axiomsClean composition_rejects
#axiomsClean transformed_contract_rejects
#axiomsClean composed_contract_rejects
#axiomsClean pipeline_rejects
#axiomsClean chain_rejects
#axiomsClean erased_store_projects
#axiomsClean composed_store_projects
#axiomsClean chain_executes_store
#axiomsClean pipeline_executes_store

end KernelIOProjectionRegression
