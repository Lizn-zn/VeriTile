import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit

open VeriTile.Triton

namespace ModuleSpecAuditRegression

-- Both the result type and a kernel parameter are on separate lines.
def multilineKernel (body : List Stmt)
    : ComputeKernel := ComputeKernel.fromKernelBody [] [] body

-- This takes a kernel but returns an IO wrapper; it must NOT be inventoried
-- as a kernel. Its deliberately dependent *Spec tests that distinction.
def wrapper (kernel : ComputeKernel)
    (projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel) : KernelIO₁ where
  kernel := kernel
  projection := projection
  inp := "x"
  out := "x"
  Bin := 1
  Bout := 1
  read := fun _ => 0
  write := fun _ => 0

def wrapperSpec := wrapper
abbrev valueSpec : Nat := 42
@[kernel_spec] def arbitrarilyNamedMath : Nat := 43
private def privateSpec : Nat := 44

-- A custom declaration form can explicitly register its execution role.
@[kernel_denotation] noncomputable denotation executionSpec (body : List Stmt) : Option ℝ :=
  denoteKernel (multilineKernel body) "flat" [("x", 1)] 0 [] "x" 0

/--
info: Spec audit: kernels=1, independentSpecs=4, denotations=1
kernels: [ModuleSpecAuditRegression.multilineKernel]
independent specs: [ModuleSpecAuditRegression.arbitrarilyNamedMath,
 ModuleSpecAuditRegression.privateSpec,
 ModuleSpecAuditRegression.valueSpec,
 ModuleSpecAuditRegression.wrapperSpec]
denotations: [ModuleSpecAuditRegression.executionSpec]
-/
#guard_msgs in
#auditModuleSpecs

-- A transitive dependency is rejected, including through a custom keyword.
def hiddenKernel := multilineKernel
denotation circularSpec : List Stmt → ComputeKernel := hiddenKernel

/-- error: ModuleSpecAuditRegression.circularSpec: SELF-REFERENTIAL — definition transitively uses kernel(s) [ModuleSpecAuditRegression.circularSpec,
 ModuleSpecAuditRegression.hiddenKernel,
 ModuleSpecAuditRegression.multilineKernel] -/
#guard_msgs in
#auditModuleSpecs

end ModuleSpecAuditRegression
