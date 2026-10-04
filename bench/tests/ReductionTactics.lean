import bench.examples.RowWiseSum.FPEquiv

/-!
Semantic decomposition and reduction search must use actual executions and a
permutation of all leaves. These kernels differ from the worked example in
register names, region parameters and an extra harmless assignment.
-/
namespace VeriTile.Tests.ReductionTactics
open VeriTile Triton FP.Structural FP.Equational
open scoped VeriTile.Spec
open VeriTile.Bench.Examples.RowWiseSumFPEquiv (Rules)
set_option maxHeartbeats 1600000
set_option linter.unusedVariables false
set_option linter.unusedSectionVars false

-- Importing the numerical adapter preserves the generic syntax path.
theorem syntax_fallback (R : Spec.Assumptions Nat) (h : Spec.Derivation R [1] [2]) :
    [0, 1, 9] ≡[R] [0, 2, 9] := by
  equiv_decompose
  all_goals fp_prove

def forward (input output : RegionName) (stride B : Nat) : ComputeKernel := triton {
  row_id := tl.program_id(0)
  indices := tl.arange(0, $(B))
  loaded := tl.load($(input) + row_id * $(stride) + indices, dtype=tl.float32)
  total := tl.sum(loaded, axis=0)
  tl.store($(output) + row_id, total)
}

def backward (input output : RegionName) (stride B : Nat) : ComputeKernel := triton {
  row_id := tl.program_id(0)
  unused := tl.program_id(0)
  indices := $(B - 1) - tl.arange(0, $(B))
  loaded := tl.load($(input) + row_id * $(stride) + indices, dtype=tl.float32)
  total := tl.sum(loaded, axis=0)
  tl.store($(output) + row_id, total)
}

def readSum {α : Type} (M : Algebra α) (B : Nat) (xs : Fin B → α) : α :=
  M.reduceSum (some .fp32) (shape := [B]) ⟨0, by simp⟩ Bool.false
    (fun i => M.fp32Load (xs i.1)) PUnit.unit

@[equiv_exec] theorem forward_run {α : Type} [Inhabited α] (M : Algebra α)
    (input output : RegionName) (stride B : Nat) (s : State α) (xs : Fin B → α)
    (hx : ∀ i : Fin B, (s.mem input (s.pids 0 * stride + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (forward input output stride B) s = some t ∧
      t.mem output (s.pids 0) = Cell.mk .real (readSum M B xs) ∧
      (∀ (r : RegionName) o, (r ≠ output ∨ o ≠ s.pids 0) → t.mem r o = s.mem r o) := by
  simp [forward, FP.Structural.exec, run, step, evalExpr, evalComputeOp, evalOp_unfold, numeric,
    FP.Structural.bop, store, TileShape.allIndices, TileShape.eraseAxis, Region.cast,
    ComputeDType.eraseDType, hx, readSum]
  intro r o hmiss
  exact (State.write_other _ output r (s.pids 0) o _ hmiss).trans rfl

@[equiv_exec] theorem backward_run {α : Type} [Inhabited α] (M : Algebra α)
    (input output : RegionName) (stride B : Nat) (s : State α) (xs : Fin B → α)
    (hx : ∀ i : Fin B, (s.mem input (s.pids 0 * stride + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (backward input output stride B) s = some t ∧
      t.mem output (s.pids 0) = Cell.mk .real (readSum M B (fun i => xs i.rev)) ∧
      (∀ (r : RegionName) o, (r ≠ output ∨ o ≠ s.pids 0) → t.mem r o = s.mem r o) := by
  have hrev (i : Fin B) : (s.mem input (s.pids 0 * stride + (B - 1 - i.val))).read .real = xs i.rev := by
    simpa only [Fin.val_rev, Nat.sub_sub, Nat.add_comm] using hx i.rev
  simp [backward, FP.Structural.exec, run, step, evalExpr, evalComputeOp, evalOp_unfold, numeric,
    FP.Structural.bop, store, TileShape.allIndices, TileShape.eraseAxis, Region.cast,
    ComputeDType.eraseDType, hrev, readSum]
  intro r o hmiss
  exact (State.write_other _ output r (s.pids 0) o _ hmiss).trans rfl

def io (input output : RegionName) (stride B : Nat) (reversed : Bool) : KernelIO₁ where
  kernel := if reversed then backward input output stride B else forward input output stride B
  inp := input
  out := output
  Bin := B
  Bout := 1
  read := fun pid => pid * stride
  write := id
  projection := by cases reversed <;> rfl

theorem renamed_equiv (input output : RegionName) (stride B : Nat) (R : Rules) :
    io input output stride B Bool.false ≡[R] io input output stride B Bool.true := by
  equiv_decompose
  all_goals fp_prove

theorem inplace_empty (stride : Nat) (R : Rules) :
    io "buffer" "buffer" stride 0 Bool.false ≡[R] io "buffer" "buffer" stride 0 Bool.true := by
  equiv_decompose
  all_goals fp_prove

-- Verify that structural decomposition exposes a numerical relation, and that
-- the difference is not changed into equality of the two address vectors.
theorem exposed_output (stride B : Nat) (R : Rules) :
    io "src" "dst" stride B Bool.false ≡[R] io "src" "dst" stride B Bool.true := by
  equiv_decompose
  change TermEq R.assumptions _ _
  fp_prove

section Values
variable {α : Type} (R : Spec.Assumptions ComputeStmt)
  (hc : Spec.Derivation R commuteLHS commuteRHS)
  (ha : Spec.Derivation R associateLHS associateRHS)
include hc ha

theorem arbitrary_permutation (plans : Schedules) (B : Nat) (xs : Fin B → Term α)
    (σ : Equiv.Perm (Fin B)) :
    TermEq R (fp32Sum plans B xs) (fp32Sum plans B (xs ∘ σ)) := by
  fp_prove

theorem inverse_permutation (plans : Schedules) (B : Nat) (xs : Fin B → Term α)
    (σ : Equiv.Perm (Fin B)) :
    TermEq R (fp32Sum plans B xs) (fp32Sum plans B (xs ∘ σ.symm)) := by
  fp_prove

theorem explicit_vector_equation (plans : Schedules) (B : Nat)
    (xs ys : Fin B → Term α) (σ : Equiv.Perm (Fin B)) (h : ys = xs ∘ σ) :
    TermEq R (fp32Sum plans B xs) (fp32Sum plans B ys) := by
  fp_prove

theorem common_cast (plans : Schedules) (B : Nat) (xs : Fin B → Term α) :
    TermEq R
      (.app (.cast (some .fp32) .real .bf16) [fp32Sum plans B xs])
      (.app (.cast (some .fp32) .real .bf16) [fp32Sum plans B (fun i => xs i.rev)]) := by
  fp_prove

theorem invalid_inputs_rejected (plans : Schedules) (xs : Fin 3 → Term α) : True := by
  fail_if_success
    have : TermEq R (fp32Sum plans 3 xs) (fp32Sum plans 3 (fun _ => xs 0)) := by fp_prove
  fail_if_success
    have : TermEq R (fp32Sum plans 3 xs)
        (fp32Sum plans 2 (fun i => xs ⟨i.val, by omega⟩)) := by fp_prove
  fail_if_success
    have : TermEq R (fp32Sum plans 3 xs)
        ((fp32Sum plans 3 xs).add (.app (.literal (some .fp32) .real 0) [])) := by fp_prove
  trivial

omit hc ha in
def sum64 (plans : Schedules) (xs : Fin 3 → Term α) : Term α :=
  (algebra plans).reduceSum (some .fp64) (shape := [3]) ⟨0, by decide⟩ Bool.false
    (fun i => xs i.1) PUnit.unit

theorem precision_and_cast_rejected (plans : Schedules) (xs : Fin 3 → Term α) : True := by
  fail_if_success
    have : TermEq R (fp32Sum plans 3 xs) (sum64 plans (fun i => xs i.rev)) := by fp_prove
  fail_if_success
    have : TermEq R
        (.app (.cast (some .fp32) .real .bf16) [fp32Sum plans 3 xs])
        (.app (.cast (some .fp32) .real .fp32) [fp32Sum plans 3 (fun i => xs i.rev)]) := by fp_prove
  trivial

theorem association_required (plans : Schedules) (B : Nat) (xs : Fin B → Term α) : True := by
  clear ha
  fail_if_success
    have : TermEq R (fp32Sum plans B xs) (fp32Sum plans B (fun i => xs i.rev)) := by fp_prove
  trivial

theorem commutation_required (plans : Schedules) (B : Nat) (xs : Fin B → Term α) : True := by
  clear hc
  fail_if_success
    have : TermEq R (fp32Sum plans B xs) (fp32Sum plans B (fun i => xs i.rev)) := by fp_prove
  trivial
end Values

-- The scalar adapter cannot silently omit an additional output cell or alter
-- the output address. Both remain obligations of the enclosing specification.
def twoOutputs (stride B : Nat) (rev : Bool) : KernelIO₁ :=
  { io "src" "dst" stride B rev with Bout := 2 }
def shiftedOutput (stride B : Nat) : KernelIO₁ :=
  { io "src" "dst" stride B Bool.true with write := fun pid => pid + 1 }

theorem output_contract_retained (stride B : Nat) (R : Rules) : True := by
  fail_if_success
    have : twoOutputs stride B Bool.false ≡[R] twoOutputs stride B Bool.true := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : io "src" "dst" stride B Bool.false ≡[R] shiftedOutput stride B := by
      equiv_decompose
      all_goals fp_prove
  trivial

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let deps := VeriTile.Meta.projValueClosure env
    #[``renamed_equiv, ``inplace_empty, ``exposed_output] {}
  if deps.contains ``VeriTile.Bench.Examples.RowWiseSumFPEquiv.rowwise_sum_equiv then
    throwError "The new kernels reused the completed example instead of composing generic lemmas"

/-- info: FP assumptions used by renamed_equiv:
---
info:   add_assoc
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions renamed_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.ReductionTactics
