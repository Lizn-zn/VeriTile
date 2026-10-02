/- Ordinary division versus one reciprocal shared by the output lanes.
The common max/exp/sum prefix is opaque. The only numerical rewrite is the
accepted scalar div_mul_rcp instance, with finite operands and a nonzero divisor.
This specializes the original kernel to fp32 inputs and computation. -/
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Reciprocal
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.SoftmaxReciprocalFPEquiv
open VeriTile Triton
open FP.Structural FP.Guarded

set_option maxHeartbeats 1600000

def commonPrefix (xReg : RegionName) (B : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(B) + tl.arange(0, $(B))
  x    := tl.load($(xReg) + offs, dtype=tl.float32)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
}

def originalKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(B) + tl.arange(0, $(B))
  x    := tl.load($(xReg) + offs, dtype=tl.float32)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

def reciprocalKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(B) + tl.arange(0, $(B))
  x    := tl.load($(xReg) + offs, dtype=tl.float32)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  inv_s := 1 / s
  y     := e * inv_s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

def loaded {α : Type} (M : Algebra α) (a : α) : α := M.fp32Load a

def exponentials {α : Type} (M : Algebra α) (B : Nat) (xs : Fin B → α) : Fin B → α :=
  fun i => M.unary (some .fp32) .exp
    (M.binary (some .fp32) .real .sub (loaded M (xs i))
      (M.reduceMax (some .fp32) (shape := [B]) ⟨0, by simp⟩ Bool.false
        (fun j => loaded M (xs j.1)) PUnit.unit))

def denominator {α : Type} (M : Algebra α) (B : Nat) (xs : Fin B → α) : α :=
  M.reduceSum (some .fp32) (shape := [B]) ⟨0, by simp⟩ Bool.false
    (fun i => exponentials M B xs i.1) PUnit.unit

def output {α : Type} (M : Algebra α) (v : α) : α := M.cast none .real .bf16 v

/-- Domain requirements for the two operands at each scalar rewrite site.
No distribution or experimental shape is imposed on the symbolic kernel. -/
def domain (B : Nat) : Precondition where
  code :=  (commonPrefix "x" B).surfaceBody
  guards := [⟨"e", [B], .finite⟩, ⟨"s", [], .finite⟩, ⟨"s", [], .nonzero⟩]

theorem domain_values {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (B : Nat) (hB : 0 < B) (s : State α) (xs : Fin B → α)
    (hx : ∀ i : Fin B, (s.mem "x" (s.pids 0 * B + i.val)).read .real = xs i)
    (h : (domain B).Holds M D s) :
    (∀ i, D .finite (exponentials M B xs i)) ∧
      D .finite (denominator M B xs) ∧ D .nonzero (denominator M B xs) := by
  simp [Precondition.Holds, domain, commonPrefix, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
    numeric, bop, TileShape.axisDim, TileShape.eraseAxis,
    hB, hx, Function.comp_def, TileIndex] at h
  exact ⟨h.1, h.2.1 PUnit.unit, h.2.2 PUnit.unit⟩

theorem original_run {α : Type} [Inhabited α] (M : Algebra α)
    (B : Nat) (hB : 0 < B) (s : State α) (xs : Fin B → α)
    (hx : ∀ i : Fin B, (s.mem "x" (s.pids 0 * B + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (originalKernel "x" "y" B) s = some t ∧
      (∀ i : Fin B, t.mem "y" (s.pids 0 * B + i.val) = Cell.mk .bf16
        (output M (FP.Reciprocal.quotient .fp32 M
          (exponentials M B xs i) (denominator M B xs)))) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [originalKernel, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, ComputeDType.eraseDType, numeric, bop, store,
    TileShape.axisDim, TileShape.eraseAxis, hB, toFloat, ofFloat]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    simp [hx, output, loaded, exponentials, denominator, Function.comp_def, bop,
      FP.Reciprocal.quotient, FP.Reciprocal.finish, FP.Reciprocal.precision]
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

theorem reciprocal_run {α : Type} [Inhabited α] (M : Algebra α)
    (B : Nat) (hB : 0 < B) (s : State α) (xs : Fin B → α)
    (hx : ∀ i : Fin B, (s.mem "x" (s.pids 0 * B + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (reciprocalKernel "x" "y" B) s = some t ∧
      (∀ i : Fin B, t.mem "y" (s.pids 0 * B + i.val) = Cell.mk .bf16
        (output M (FP.Reciprocal.reciprocal .fp32 M
          (exponentials M B xs i) (denominator M B xs)))) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [reciprocalKernel, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, ComputeDType.eraseDType, numeric, bop, store,
    TileShape.axisDim, TileShape.eraseAxis, hB, toFloat, ofFloat]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    simp [hx, output, loaded, exponentials, denominator, Function.comp_def, bop,
      FP.Reciprocal.reciprocal, FP.Reciprocal.finish, FP.Reciprocal.precision]
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

def originalIO (B : Nat) : FP.Guarded.IO where
  io := {
    kernel := originalKernel "x" "y" B
    inp := "x", out := "y", Bin := B, Bout := B
    read := fun pid => pid * B
    write := fun pid => pid * B }
  domain := domain B

def reciprocalIO (B : Nat) : FP.Guarded.IO :=
  { originalIO B with io := { (originalIO B).io with
      kernel := reciprocalKernel "x" "y" B, projection := by rfl } }

abbrev Rules := FP.Reciprocal.Rules .fp32

open scoped VeriTile.Spec

specification softmax_reciprocal_equiv (R : Rules) (B : Nat) (hB : 0 < B) :
    originalIO B ≡[R] reciprocalIO B := by
  apply Spec.FloatingPoint.ofNumerical (lhs := originalIO B) (rhs := reciprocalIO B)
    (structural := fun _ _ => False) rfl rfl
  refine ⟨?_, ?_, ?_⟩
  · simp [IO₁PrivateScratch, originalIO]
  · simp [IO₁PrivateScratch, originalIO, reciprocalIO]
  · intro α _ M D hM s hs
    let xs : Fin B → α := fun i => (s.mem "x" (s.pids 0 * B + i.val)).read .real
    obtain ⟨hf, hd, hn⟩ := domain_values M D B hB s xs (fun _ => rfl) hs
    obtain ⟨a, ha, hva, hfa⟩ := original_run M B hB s xs (fun _ => rfl)
    obtain ⟨b, hb, hvb, hfb⟩ := reciprocal_run M B hB s xs (fun _ => rfl)
    refine ⟨a, b, ha, hb, ?_, ?_, ?_⟩
    · intro i
      change a.mem "y" (s.pids 0 * B + i.val) = b.mem "y" (s.pids 0 * B + i.val)
      rw [hva i, hvb i, FP.Reciprocal.apply_rule R M D hM s _ _ (hf i) hd hn]
    · intro r o ho _
      exact hfa r o ho
    · intro r o ho _
      exact hfb r o ho

#print_fp_assumptions softmax_reciprocal_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.SoftmaxReciprocalFPEquiv
