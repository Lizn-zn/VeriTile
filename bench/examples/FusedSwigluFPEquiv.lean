/- FP equivalence of the original fused and materialized SwiGLU kernels.
Both retain the bf16 intermediate cast and the same numerical call tree.
The proof forwards typed scratch values only on active lanes, and frames every
inactive cell. No numerical identity, including cast idempotence, is assumed. -/
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.FusedSwigluFPEquiv
open VeriTile Triton
open FP.Structural

set_option maxHeartbeats 1200000

/-- Fused SwiGLU, with its original bf16 intermediate and output casts. -/
def swiglu_fused (X Y OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel := triton {
  start_col = tl.program_id(0) * $(BLOCK_N)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0)
  sil = (x * tl.sigmoid(x)).to(tl.bfloat16)
  out = sil * y
  tl.store(OUT + cols, (out).to(tl.bfloat16), mask=cols < $(ncols))
}

/-- Materialize the same bf16 intermediate at active lanes. -/
def silu_step (X S : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel := triton {
  start_col = tl.program_id(0) * $(BLOCK_N)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0)
  s = x * tl.sigmoid(x)
  tl.store(S + cols, (s).to(tl.bfloat16), mask=cols < $(ncols))
}

/-- The load annotation elaborates to a bf16-typed load. The multiplication
retains the same explicit conversion of that value as the fused kernel. -/
def mul_step (S Y OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel := triton {
  start_col = tl.program_id(0) * $(BLOCK_N)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  z = tl.load(S + cols, mask=cols < $(ncols)).to(tl.bfloat16)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0)
  out = z * y
  tl.store(OUT + cols, (out).to(tl.bfloat16), mask=cols < $(ncols))
}

/-- Preserve the original single-launch concatenation of the two stages. -/
def swiglu_unfused (X Y S OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel :=
  ComputeKernel.seq [X, Y] [OUT]
    [silu_step X S ncols BLOCK_N, mul_step S Y OUT ncols BLOCK_N]

def siluValue {α : Type} (M : Algebra α) (x : α) : α :=
  M.cast none .real .bf16 (M.binary none .real .mul x (M.unary none .sigmoid x))

def productValue {α : Type} (M : Algebra α) (sil y : α) : α :=
  M.cast none .real .bf16 (M.binary none .real .mul (M.cast none .bf16 .real sil) y)

theorem silu_run {α : Type} [Inhabited α] (M : Algebra α) (n B : Nat) (s : State α)
    (xs : Fin B → α)
    (hxs : ∀ i : Fin B, s.pids 0 * B + i.val < n →
      (s.mem "X" (s.pids 0 * B + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (silu_step "X" "S" n B) s = some t ∧
      (∀ i : Fin B, s.pids 0 * B + i.val < n →
        t.mem "S" (s.pids 0 * B + i.val) = Cell.mk .bf16 (siluValue M (xs i))) ∧
      t.pids = s.pids ∧
      (∀ (r : RegionName) o,
        (r ≠ "S" ∨ ∀ i : Fin B, s.pids 0 * B + i.val < n → o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [silu_step, FP.Structural.exec, run, step, evalExpr,
    evalOp_unfold, numeric, FP.Structural.bop, store, natLt, ofFloat, toFloat,
    Function.comp_def]
  refine ⟨fun i hi => ?_, fun r o hmiss => ?_⟩
  · rw [State.masked_scatter_readback _ _ _ _ _ (i, PUnit.unit) hi hinj]
    simp [hi, hxs, siluValue]
  · apply (State.masked_scatter_frame _ _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ hk => ho k.1 hk

theorem mul_run {α : Type} [Inhabited α] (M : Algebra α) (n B : Nat) (s : State α)
    (zs ys : Fin B → α)
    (hzs : ∀ i : Fin B, s.pids 0 * B + i.val < n →
      (s.mem "S" (s.pids 0 * B + i.val)).read .bf16 = zs i)
    (hys : ∀ i : Fin B, s.pids 0 * B + i.val < n →
      (s.mem "Y" (s.pids 0 * B + i.val)).read .real = ys i) :
    ∃ t, FP.Structural.exec M (mul_step "S" "Y" "OUT" n B) s = some t ∧
      (∀ i : Fin B, s.pids 0 * B + i.val < n →
        t.mem "OUT" (s.pids 0 * B + i.val) = Cell.mk .bf16 (productValue M (zs i) (ys i))) ∧
      t.pids = s.pids ∧
      (∀ (r : RegionName) o,
        (r ≠ "OUT" ∨ ∀ i : Fin B, s.pids 0 * B + i.val < n → o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [mul_step, FP.Structural.exec, run, step, evalExpr,
    evalOp_unfold, numeric, FP.Structural.bop, store, natLt, ofFloat, toFloat]
  refine ⟨fun i hi => ?_, fun r o hmiss => ?_⟩
  · rw [State.masked_scatter_readback _ _ _ _ _ (i, PUnit.unit) hi hinj]
    simp [hi, hzs, hys, productValue]
  · apply (State.masked_scatter_frame _ _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ hk => ho k.1 hk

theorem fused_run {α : Type} [Inhabited α] (M : Algebra α) (n B : Nat) (s : State α)
    (xs ys : Fin B → α)
    (hxs : ∀ i : Fin B, s.pids 0 * B + i.val < n →
      (s.mem "X" (s.pids 0 * B + i.val)).read .real = xs i)
    (hys : ∀ i : Fin B, s.pids 0 * B + i.val < n →
      (s.mem "Y" (s.pids 0 * B + i.val)).read .real = ys i) :
    ∃ t, FP.Structural.exec M (swiglu_fused "X" "Y" "OUT" n B) s = some t ∧
      (∀ i : Fin B, s.pids 0 * B + i.val < n →
        t.mem "OUT" (s.pids 0 * B + i.val) =
          Cell.mk .bf16 (productValue M (siluValue M (xs i)) (ys i))) ∧
      t.pids = s.pids ∧
      (∀ (r : RegionName) o,
        (r ≠ "OUT" ∨ ∀ i : Fin B, s.pids 0 * B + i.val < n → o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [swiglu_fused, FP.Structural.exec, run, step, evalExpr,
    evalOp_unfold, numeric, FP.Structural.bop, store, natLt, ofFloat, toFloat,
    Function.comp_def]
  refine ⟨fun i hi => ?_, fun r o hmiss => ?_⟩
  · rw [State.masked_scatter_readback _ _ _ _ _ (i, PUnit.unit) hi hinj]
    simp [hi, hxs, hys, productValue, siluValue]
  · apply (State.masked_scatter_frame _ _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ hk => ho k.1 hk

theorem unfused_run {α : Type} [Inhabited α] (M : Algebra α) (n B : Nat) (s : State α)
    (xs ys : Fin B → α)
    (hxs : ∀ i : Fin B, s.pids 0 * B + i.val < n →
      (s.mem "X" (s.pids 0 * B + i.val)).read .real = xs i)
    (hys : ∀ i : Fin B, s.pids 0 * B + i.val < n →
      (s.mem "Y" (s.pids 0 * B + i.val)).read .real = ys i) :
    ∃ t, FP.Structural.exec M (swiglu_unfused "X" "Y" "S" "OUT" n B) s = some t ∧
      (∀ i : Fin B, s.pids 0 * B + i.val < n →
        t.mem "OUT" (s.pids 0 * B + i.val) =
          Cell.mk .bf16 (productValue M (siluValue M (xs i)) (ys i))) ∧
      (∀ (r : RegionName) o,
        (r ≠ "OUT" ∨ ∀ i : Fin B, s.pids 0 * B + i.val < n → o ≠ s.pids 0 * B + i.val) →
        (r ≠ "S" ∨ ∀ i : Fin B, s.pids 0 * B + i.val < n → o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  obtain ⟨s1, he1, hz, hp1, hf1⟩ := silu_run M n B s xs hxs
  obtain ⟨s2, he2, hv, _, hf2⟩ := mul_run M n B s1 (fun i => siluValue M (xs i)) ys
    (by
      intro i hi
      rw [hp1] at hi ⊢
      rw [hz i hi]
      rfl)
    (by
      intro i hi
      rw [hp1] at hi ⊢
      rw [hf1 "Y" _ (Or.inl (by decide))]
      exact hys i hi)
  rw [hp1] at hv hf2
  refine ⟨s2, ?_, hv, ?_⟩
  · simp only [swiglu_unfused, exec_seq_cons, he1, Option.bind_some, he2, exec_seq_nil]
  · intro r o ho hs
    exact (hf2 r o ho).trans (hf1 r o hs)

def fusedIO (n B : Nat) : MaskedKernelIO₂ where
  kernel := swiglu_fused "X" "Y" "OUT" n B
  in1 := "X"
  in2 := "Y"
  out := "OUT"
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B
  mask := fun pid i => pid * B + i.val < n

def unfusedIO (n B : Nat) : MaskedKernelIO₂ :=
  { fusedIO n B with
    kernel := swiglu_unfused "X" "Y" "S" "OUT" n B
    projection := by rfl
    scratch := [("S", fun pid => pid * B)] }

def R : Spec.Assumptions ComputeStmt := []

open scoped VeriTile.Spec

/-- Both original kernels compute the same numerical expression on active
lanes. Dimensions are arbitrary, including zero and partial blocks. -/
specification swiglu_equiv (n B : Nat) : fusedIO n B ≡[R] unfusedIO n B := by
  apply Spec.FloatingPoint.ofStructural (lhs := fusedIO n B) (rhs := unfusedIO n B)
    (structural := MaskedIO₂Equiv) rfl rfl
  refine ⟨?_, ?_, ?_⟩
  · simp [MaskedPrivateScratch, fusedIO]
  · simp [MaskedPrivateScratch, unfusedIO, fusedIO]
  · intro α _ M s
    let xs : Fin B → α := fun i => (s.mem "X" (s.pids 0 * B + i.val)).read .real
    let ys : Fin B → α := fun i => (s.mem "Y" (s.pids 0 * B + i.val)).read .real
    obtain ⟨a, ha, hva, _, hfa⟩ := fused_run M n B s xs ys (fun _ _ => rfl) (fun _ _ => rfl)
    obtain ⟨b, hb, hvb, hfb⟩ := unfused_run M n B s xs ys (fun _ _ => rfl) (fun _ _ => rfl)
    refine ⟨a, b, ha, hb, fun i hi => (hva i hi).trans (hvb i hi).symm, ?_, ?_⟩
    · intro r o ho _; exact hfa r o ho
    · intro r o ho hs
      apply hfb r o ho
      by_cases hr : r = "S"
      · exact Or.inr (hs ("S", fun pid => pid * B) (by simp [unfusedIO]) hr)
      · exact Or.inl hr

#print_fp_assumptions swiglu_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FusedSwigluFPEquiv
