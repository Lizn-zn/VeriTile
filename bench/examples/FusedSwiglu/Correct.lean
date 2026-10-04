import bench.examples.FusedSwiglu.Kernels
/- Real correctness of the original fused and materialized SwiGLU kernels.
The public formula is (x * sigmoid(x)) * y on active lanes. The original bf16
intermediate and output casts are erased only for this mathematical proof.
The materialized pipeline retains the original single-launch concatenation. -/
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.FusedSwigluCorrect
open VeriTile.Bench.Examples.FusedSwiglu.Kernels
open VeriTile Triton
open scoped VeriTile.Triton.MaskedKernelIO₂

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl


/-- Executable real counterpart of the source fused kernel. -/
def swiglu_fused_math (X Y OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel := triton {
  start_col = tl.program_id(0) * $(BLOCK_N)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0)
  sil = x * tl.sigmoid(x)
  out = sil * y
  tl.store(OUT + cols, out, mask=cols < $(ncols))
}

def silu_step_math (X S : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel := triton {
  start_col = tl.program_id(0) * $(BLOCK_N)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0)
  s = x * tl.sigmoid(x)
  tl.store(S + cols, s, mask=cols < $(ncols))
}

def mul_step_math (S Y OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel := triton {
  start_col = tl.program_id(0) * $(BLOCK_N)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  z = tl.load(S + cols, mask=cols < $(ncols))
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0)
  out = z * y
  tl.store(OUT + cols, out, mask=cols < $(ncols))
}

def swiglu_unfused_math (X Y S OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel :=
  ComputeKernel.seq [X, Y] [OUT]
    [silu_step_math X S ncols BLOCK_N, mul_step_math S Y OUT ncols BLOCK_N]

theorem fused_projection (X Y OUT : RegionName) (n B : Nat) :
    (swiglu_fused X Y OUT n B).eraseDType.toAlgorithm? =
      Except.ok (swiglu_fused_math X Y OUT n B).toAlgKernel := by
  simp [swiglu_fused, swiglu_fused_math, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeExpr.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  repeat' apply And.intro
  all_goals
    rw [Op.eraseDType.eq_def]
    simp [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType]

theorem unfused_projection (X Y S OUT : RegionName) (n B : Nat) :
    (swiglu_unfused X Y S OUT n B).eraseDType.toAlgorithm? =
      Except.ok (swiglu_unfused_math X Y S OUT n B).toAlgKernel := by
  simp [swiglu_unfused, swiglu_unfused_math, silu_step, mul_step, silu_step_math, mul_step_math, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeExpr.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  repeat' apply And.intro
  all_goals
    rw [Op.eraseDType.eq_def]
    simp [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType]

private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

theorem silu_region_run (n B : Nat) (s : BlockState) (xs : Fin B → ℝ)
    (hxs : ∀ i : Fin B, s.pid * B + i.val < n → s.readMem "x" (s.pid * B + i.val) = xs i) :
    ∃ s1, exec (silu_step_math "x" "s" n B).toAlgKernel s = some s1 ∧
      (∀ i : Fin B, s.pid * B + i.val < n →
        s1.readMem "s" (s.pid * B + i.val) = xs i * Real.sigmoid (xs i)) ∧
      (∀ (r : RegionName) o,
        (r ≠ "s" ∨ ∀ i : Fin B, s.pid * B + i.val < n → o ≠ s.pid * B + i.val) →
        s1.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [BlockState.pid_eq] at hxs ⊢
  simp [silu_step_math, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, Tile.cop, NumericDType.add, NumericDType.mul, ComparableDType.lt]
  refine ⟨fun i hi => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hinj (i, PUnit.unit)]
    simp [hi, hxs i hi]
  · refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
    intro k _ hk hc
    rcases hmiss with hr | ho
    · exact hr hc.1.symm
    · exact ho k.1 hk hc.2.symm

theorem mul_region_run (n B : Nat) (s : BlockState) (ss ys : Fin B → ℝ)
    (hss : ∀ i : Fin B, s.pid * B + i.val < n → s.readMem "s" (s.pid * B + i.val) = ss i)
    (hys : ∀ i : Fin B, s.pid * B + i.val < n → s.readMem "y" (s.pid * B + i.val) = ys i) :
    ∃ s1, exec (mul_step_math "s" "y" "out" n B).toAlgKernel s = some s1 ∧
      (∀ i : Fin B, s.pid * B + i.val < n →
        s1.readMem "out" (s.pid * B + i.val) = ss i * ys i) ∧
      (∀ (r : RegionName) o,
        (r ≠ "out" ∨ ∀ i : Fin B, s.pid * B + i.val < n → o ≠ s.pid * B + i.val) →
        s1.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [BlockState.pid_eq] at hss hys ⊢
  simp [mul_step_math, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul, ComparableDType.lt]
  refine ⟨fun i hi => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hinj (i, PUnit.unit)]
    simp [hi, hss i hi, hys i hi]
  · refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
    intro k _ hk hc
    rcases hmiss with hr | ho
    · exact hr hc.1.symm
    · exact ho k.1 hk hc.2.symm

theorem fused_region_run (n B : Nat) (s : BlockState) (xs ys : Fin B → ℝ)
    (hxs : ∀ i : Fin B, s.pid * B + i.val < n → s.readMem "x" (s.pid * B + i.val) = xs i)
    (hys : ∀ i : Fin B, s.pid * B + i.val < n → s.readMem "y" (s.pid * B + i.val) = ys i) :
    ∃ s1, exec (swiglu_fused_math "x" "y" "out" n B).toAlgKernel s = some s1 ∧
      (∀ i : Fin B, s.pid * B + i.val < n →
        s1.readMem "out" (s.pid * B + i.val) = (xs i * Real.sigmoid (xs i)) * ys i) ∧
      (∀ (r : RegionName) o,
        (r ≠ "out" ∨ ∀ i : Fin B, s.pid * B + i.val < n → o ≠ s.pid * B + i.val) →
        s1.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [BlockState.pid_eq] at hxs hys ⊢
  simp [swiglu_fused_math, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, Tile.cop, NumericDType.add, NumericDType.mul, ComparableDType.lt]
  refine ⟨fun i hi => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hinj (i, PUnit.unit)]
    simp [hi, hxs i hi, hys i hi]
  · refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
    intro k _ hk hc
    rcases hmiss with hr | ho
    · exact hr hc.1.symm
    · exact ho k.1 hk hc.2.symm

/-- Real execution of the original two-stage concatenation. -/
theorem unfused_exec_split (n B : Nat) (s : BlockState) :
    exec (swiglu_unfused_math "x" "y" "s" "out" n B).toAlgKernel s =
      (exec (silu_step_math "x" "s" n B).toAlgKernel s).bind (fun s1 =>
        exec (mul_step_math "s" "y" "out" n B).toAlgKernel s1) := by
  have h := execR_toAlgKernel_seq .triv ["x", "y"] ["out"]
    [silu_step_math "x" "s" n B, mul_step_math "s" "y" "out" n B] s (by
      intro k hk
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
      rcases hk with rfl | rfl <;> rfl)
  simpa only [execR_triv, List.foldl_cons, List.foldl_nil, Option.bind_some] using h

theorem unfused_region_run (n B : Nat) (s : BlockState) (xs ys : Fin B → ℝ)
    (hx : ∀ i : Fin B, s.pid * B + i.val < n → s.readMem "x" (s.pid * B + i.val) = xs i)
    (hy : ∀ i : Fin B, s.pid * B + i.val < n → s.readMem "y" (s.pid * B + i.val) = ys i) :
    ∃ s2, exec (swiglu_unfused_math "x" "y" "s" "out" n B).toAlgKernel s = some s2 ∧
      (∀ i : Fin B, s.pid * B + i.val < n → s2.readMem "out" (s.pid * B + i.val) =
        (xs i * Real.sigmoid (xs i)) * ys i) ∧
      (∀ (r : RegionName) o,
        (r ≠ "out" ∨ ∀ i : Fin B, s.pid * B + i.val < n → o ≠ s.pid * B + i.val) →
        (r ≠ "s" ∨ ∀ i : Fin B, s.pid * B + i.val < n → o ≠ s.pid * B + i.val) →
        s2.mem r o = s.mem r o) := by
  obtain ⟨s1, he1, hs, hf1⟩ := silu_region_run n B s xs hx
  have hp1 := exec_pid he1
  have hs' : ∀ i : Fin B, s1.pid * B + i.val < n →
      s1.readMem "s" (s1.pid * B + i.val) = xs i * Real.sigmoid (xs i) := by
    simpa only [hp1] using hs
  have hy' : ∀ i : Fin B, s1.pid * B + i.val < n →
      s1.readMem "y" (s1.pid * B + i.val) = ys i := by
    intro i hi
    rw [hp1] at hi ⊢
    exact (BlockState.readMem_congr (hf1 "y" _ (Or.inl (by decide)))).trans (hy i hi)
  obtain ⟨s2, he2, hv, hf2⟩ := mul_region_run n B s1
    (fun i => xs i * Real.sigmoid (xs i)) ys hs' hy'
  refine ⟨s2, ?_, ?_, ?_⟩
  · rw [unfused_exec_split, he1, Option.bind_some, he2]
  · simpa only [hp1] using hv
  · intro r o ho hs
    exact (hf2 r o (by simpa only [hp1] using ho)).trans (hf1 r o hs)

theorem fused_flattenOk (X Y OUT : RegionName) (n B : Nat) :
    (swiglu_fused_math X Y OUT n B).toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk, swiglu_fused_math, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
theorem fused_traceSafe (X Y OUT : RegionName) (n B : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h0 : ∀ j : Fin B, s.pid * B + j.val < n → s.pid * B + j.val < bounds X)
    (h1 : ∀ j : Fin B, s.pid * B + j.val < n → s.pid * B + j.val < bounds Y)
    (h2 : ∀ j : Fin B, s.pid * B + j.val < n → s.pid * B + j.val < bounds OUT)
    : Kernel.TraceSafe bounds (swiglu_fused_math X Y OUT n B).toAlgKernel s := by
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at h0 h1 h2
  simp [swiglu_fused_math, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, BlockState.setReg, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt]
  exact ⟨fun a ha => h0 a ha, fun a ha => h1 a ha, fun a ha => h2 a ha⟩

theorem unfused_flattenOk (X Y S OUT : RegionName) (n B : Nat) :
    (swiglu_unfused_math X Y S OUT n B).toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk, swiglu_unfused_math, silu_step_math, mul_step_math, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
theorem unfused_traceSafe (X Y S OUT : RegionName) (n B : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h0 : ∀ j : Fin B, s.pid * B + j.val < n → s.pid * B + j.val < bounds X)
    (h1 : ∀ j : Fin B, s.pid * B + j.val < n → s.pid * B + j.val < bounds Y)
    (h2 : ∀ j : Fin B, s.pid * B + j.val < n → s.pid * B + j.val < bounds S)
    (h3 : ∀ j : Fin B, s.pid * B + j.val < n → s.pid * B + j.val < bounds OUT)
    : Kernel.TraceSafe bounds (swiglu_unfused_math X Y S OUT n B).toAlgKernel s := by
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at h0 h1 h2 h3
  simp [swiglu_unfused_math, silu_step_math, mul_step_math, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, BlockState.setReg, Tile.bop, Tile.cop,
    BlockState.foldl_writeMem_prop_masked_pids,
    NumericDType.add, NumericDType.mul, ComparableDType.lt]
  exact ⟨fun a ha => h0 a ha, fun a ha => h2 a ha, fun a ha => h1 a ha, fun a ha => h3 a ha⟩

def fusedIO (n B : Nat) : MaskedKernelIO₂ where
  kernel := (swiglu_fused "x" "y" "out" n B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, fused_projection]
  in1 := "x"
  in2 := "y"
  out := "out"
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B
  mask := fun pid i => pid * B + i.val < n

specification fused_swiglu_correct (n B : Nat) :
    Spec.Real (fusedIO n B ⊨ fun xs ys i => (xs i * Real.sigmoid (xs i)) * ys i) := by
  have halg : (fusedIO n B).kernel.toAlgKernel =
      (swiglu_fused_math "x" "y" "out" n B).toAlgKernel := by
    simp [fusedIO, ComputeKernel.toAlgKernel, fused_projection]
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact fused_flattenOk "x" "y" "out" n B
  · intro bounds s hx hy ho hsc
    rw [halg]
    exact fused_traceSafe "x" "y" "out" n B bounds s hx hy ho
  · intro s xs ys hx hy
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := fused_region_run n B s xs ys hx hy
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

def unfusedIO (n B : Nat) : MaskedKernelIO₂ where
  kernel := (swiglu_unfused "x" "y" "s" "out" n B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, unfused_projection]
  in1 := "x"
  in2 := "y"
  out := "out"
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B
  mask := fun pid i => pid * B + i.val < n
  scratch := [("s", fun pid => pid * B)]

specification unfused_swiglu_correct (n B : Nat) :
    Spec.Real (unfusedIO n B ⊨ fun xs ys i => (xs i * Real.sigmoid (xs i)) * ys i) := by
  have halg : (unfusedIO n B).kernel.toAlgKernel =
      (swiglu_unfused_math "x" "y" "s" "out" n B).toAlgKernel := by
    simp [unfusedIO, ComputeKernel.toAlgKernel, unfused_projection]
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact unfused_flattenOk "x" "y" "s" "out" n B
  · intro bounds s hx hy ho hsc
    rw [halg]
    exact unfused_traceSafe "x" "y" "s" "out" n B bounds s hx hy (hsc ("s", fun pid => pid * B) (by simp [unfusedIO])) ho
  · intro s xs ys hx hy
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := unfused_region_run n B s xs ys hx hy
    refine ⟨s1, he, hv, ?_⟩
    intro r o hmiss hsc
    apply hf r o hmiss
    by_cases hr : r = "s"
    · exact Or.inr (hsc ("s", fun pid => pid * B) (by simp [unfusedIO]) hr)
    · exact Or.inl hr

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FusedSwigluCorrect
