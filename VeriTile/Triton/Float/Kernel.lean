import VeriTile.Triton.Float.ConditionalIO

/-! Public FP kernel contracts. The aliases retain the existing execution,
domain, comparison and equivalence semantics without introducing new laws. -/

namespace VeriTile.Triton

/-- Replace the source while preserving its declared IO layout and scratch
regions. Lean checks that the new source supports the kernel projection. -/
abbrev KernelIO₁.withKernel (io : KernelIO₁) (kernel : ComputeKernel)
    (projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
      first | rfl | simp [ComputeKernel.toAlgKernel]) : KernelIO₁ :=
  { io with kernel := kernel, projection := projection }

/-- Replace the source of a three-input contract; retain its IO and scratch. -/
abbrev KernelIO₃.withKernel (io : KernelIO₃) (kernel : ComputeKernel)
    (projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
      first | rfl | simp [ComputeKernel.toAlgKernel]) : KernelIO₃ :=
  { io with kernel := kernel, projection := projection }

/-- Replace the source while retaining both output windows. -/
abbrev KernelIO₁ₓ₂.withKernel (io : KernelIO₁ₓ₂) (kernel : ComputeKernel)
    (projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
      first | rfl | simp [ComputeKernel.toAlgKernel]) : KernelIO₁ₓ₂ :=
  { io with kernel := kernel, projection := projection }

/-- Replace the source while retaining the lane mask, IO and scratch. -/
abbrev MaskedKernelIO₂.withKernel (io : MaskedKernelIO₂) (kernel : ComputeKernel)
    (projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
      first | rfl | simp [ComputeKernel.toAlgKernel]) : MaskedKernelIO₂ :=
  { io with kernel := kernel, projection := projection }

namespace FP

/-- One input region and one output region, with an explicit precision profile,
numerical domain and comparison requirements. Equivalence quantifies over a
shared reduction schedule and retains successful execution and memory frames. -/
abbrev Kernel₁ := Scheduled.ConditionalIO₁

/-- One input and two observable outputs, with a precision profile and domain. -/
abbrev Kernel₁ₓ₂ := Scheduled.IO₁ₓ₂

/-- Three inputs and one observable output, with a precision profile and domain. -/
abbrev Kernel₃ := Scheduled.IO₃

/-- One input and one output, checking operands after executing an explicit
prefix. Its optional default precision preserves explicitly typed operations;
reductions retain the source's opaque numerical operations. -/
abbrev PrefixKernel₁ := Guarded.IO

/-- Default fp32 computation and explicit fp32 sum trees. Source casts and
explicitly tagged operation precisions are preserved. -/
abbrev fp32 := Scheduled.fp32

/-- Reuse an execution proof that needs no comparison support. Examples with
no comparisons declare `comparisons := []`, retaining the original domain,
precision, shared reduction plans, successful execution and memory frames. -/
theorem Kernel₁.ofExecution {R : Spec.Assumptions GuardedFragment} {lhs rhs : Kernel₁}
    (h : Scheduled.Equivalent₁ R lhs.toIO₁ rhs.toIO₁) :
    Scheduled.ConditionalEquivalent₁ R lhs rhs :=
  ⟨h.1, h.2.1, fun α _ M D hM _ plans s hd => h.2.2 α M D hM plans s hd⟩

end FP
end VeriTile.Triton
