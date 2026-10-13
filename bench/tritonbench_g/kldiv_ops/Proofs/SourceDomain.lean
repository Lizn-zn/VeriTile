import bench.tritonbench_g.kldiv_ops.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivOps.TypedSource

/-- Frontend tensor types accepted by raw ordinary exp in Triton v3.5.1.
This does not certify device-specific lowering. -/
def ExpInputSupported (target : Storage) : Prop :=
  target = .fp32 ∨ target = .fp64

/-- A nonzero reduction carries the original fp32 `loss_sum = 0.0`.
An fp64 prediction or target would change its type in the dry-visited loop body,
which pinned Triton rejects. Runtime N=0 does not remove this type restriction.
Output storage does not determine the accumulator's type. -/
def CarrySupported (pred target : Storage) (reduction : Nat) : Prop :=
  reduction = 0 ∨ (pred ≠ .fp64 ∧ target ≠ .fp64)

/-- Frontend admission for a Python-float runtime fp32 epsilon argument.
The default clamp promotes narrower targets before log; the log-target branch
uses raw exp. This predicate states these frontend restrictions only: native
launch validity, integer overflow and hardware lowering remain separate. -/
def ForwardSourceSupported (pred target : Storage) (_output : Storage)
    (logTarget : Bool) (reduction : Nat) : Prop :=
  (logTarget = Bool.true → ExpInputSupported target) ∧
    CarrySupported pred target reduction

def ForwardDefaultSupported (pred target output : Storage) (reduction : Nat) : Prop :=
  ForwardSourceSupported pred target output Bool.false reduction

def ForwardLogSupported (pred : Storage) (target : MathStorage)
    (output : Storage) (reduction : Nat) : Prop :=
  ForwardSourceSupported pred target.storage output Bool.true reduction

/-- Pinned Triton exposes nonzero floating scalar builders for these four
storage types. Its fp8 type builders do not supply nonzero scalar values. -/
def NonzeroScalarSupported (target : Storage) : Prop :=
  target ≠ .f8e4 ∧ target ≠ .f8e5

/-- No accumulation is carried by the backward kernel. The default branch
requires the nonzero weak scalar -1; log-target exp has its own restriction. -/
def BackwardSourceSupported (target : Storage) (_output : Storage)
    (logTarget : Bool) : Prop :=
  if logTarget then ExpInputSupported target else NonzeroScalarSupported target

theorem forward_default_elementwise_supported (pred target output : Storage) :
    ForwardDefaultSupported pred target output 0 := by
  constructor
  · intro h
    cases h
  · exact Or.inl rfl

theorem forward_default_reduced_supported (pred target output : Storage)
    (reduction : Nat) (hp : pred ≠ .fp64) (ht : target ≠ .fp64) :
    ForwardDefaultSupported pred target output reduction := by
  constructor
  · intro h
    cases h
  · exact Or.inr ⟨hp, ht⟩

theorem math_storage_exp_supported (target : MathStorage) :
    ExpInputSupported target.storage := by
  cases target
  · exact Or.inl rfl
  · exact Or.inr rfl

theorem forward_log_elementwise_supported (pred : Storage) (target : MathStorage)
    (output : Storage) : ForwardLogSupported pred target output 0 :=
  ⟨fun _ ↦ math_storage_exp_supported target, Or.inl rfl⟩

theorem backward_default_supported (target output : Storage)
    (h : NonzeroScalarSupported target) :
    BackwardSourceSupported target output Bool.false := by
  exact h

theorem backward_log_supported (target : MathStorage) (output : Storage) :
    BackwardSourceSupported target.storage output Bool.true :=
  math_storage_exp_supported target

theorem fp8e4_backward_default_rejected (output : Storage) :
    ¬ BackwardSourceSupported .f8e4 output Bool.false := by
  intro h
  exact h.1 rfl

theorem fp8e5_backward_default_rejected (output : Storage) :
    ¬ BackwardSourceSupported .f8e5 output Bool.false := by
  intro h
  exact h.2 rfl

/-- The typed AST for fp64 reduction exists as a formal model, but it is not
admitted as a compilation of the unchanged Python accumulator. -/
theorem fp64_prediction_reduction_rejected (target output : Storage)
    (logTarget : Bool) (reduction : Nat) (hr : reduction ≠ 0) :
    ¬ ForwardSourceSupported .fp64 target output logTarget reduction := by
  rintro ⟨_, he | ⟨hp, _⟩⟩
  · exact hr he
  · exact hp rfl

theorem fp64_target_reduction_rejected (pred output : Storage)
    (logTarget : Bool) (reduction : Nat) (hr : reduction ≠ 0) :
    ¬ ForwardSourceSupported pred .fp64 output logTarget reduction := by
  rintro ⟨_, he | ⟨_, ht⟩⟩
  · exact hr he
  · exact ht rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivOps.TypedSource
