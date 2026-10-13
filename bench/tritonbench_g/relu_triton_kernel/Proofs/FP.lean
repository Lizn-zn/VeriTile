import bench.tritonbench_g.relu_triton_kernel.Kernels
import VeriTile.Triton.Float.Kernel

namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.FPProofs
open VeriTile Triton FP.Structural
set_option maxHeartbeats 1600000

/-- A masked scatter depends only on memory, even when temporary registers differ. -/
theorem scatter_mem_congr {α ι : Type} (l : List ι) (active : ι → Prop) [DecidablePred active]
    (r : RegionName) (off : ι → Nat) (v : ι → Cell α) (s t : State α)
    (h : s.mem = t.mem) :
    (l.foldl (fun a i => if active i then a.write r (off i) (v i) else a) s).mem =
      (l.foldl (fun a i => if active i then a.write r (off i) (v i) else a) t).mem := by
  induction l generalizing s t with
  | nil => exact h
  | cons i l ih =>
    simp only [List.foldl_cons]
    split <;> apply ih
    · simp only [State.write, h]
    · exact h

noncomputable def outputState {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (x y : RegionName) (N B : Nat) (s : State α) : State α :=
  if s.pids 0 = 0 then
    (TileShape.allIndices [B]).foldl (fun t i =>
      if s.pids 0 * B + i.1.val < N then
        t.write y (s.pids 0 * B + i.1.val) (.mk .real
          (let v := (s.mem x (s.pids 0 * B + i.1.val)).read .real
           if cmp (M.literal none .real 0) v then v else M.literal none .real 0))
      else t) s
  else s

theorem original_run {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (hc : M.compareLe none .real = some cmp)
    (x y : RegionName) (N B : Nat) (s : State α) :
    ∃ t, FP.Structural.exec M (relu_kernel x y N B) s = some t ∧
      t.mem = (outputState M cmp x y N B s).mem := by
  by_cases hp : s.pids 0 = 0
  all_goals simp [hp, relu_kernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, numericLe, numericLt, natLt, numericEq, FP.Structural.bop,
    store, hc, outputState]
  simp +contextual only [ite_true]
  norm_num only
  apply scatter_mem_congr _ (fun i : TileIndex [B] => i.1.val < N)
  rfl

theorem fused_run {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (hc : M.compareLe none .real = some cmp)
    (x y : RegionName) (N B : Nat) (s : State α) :
    ∃ t, FP.Structural.exec M (relu_fused_kernel x y N B) s = some t ∧
      t.mem = (outputState M cmp x y N B s).mem := by
  by_cases hp : s.pids 0 = 0
  all_goals simp [hp, relu_fused_kernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, numericLe, numericLt, natLt, numericEq, FP.Structural.bop,
    store, hc, outputState]
  simp +contextual only [ite_true]
  norm_num only
  apply scatter_mem_congr _ (fun i : TileIndex [B] => i.1.val < N)
  rfl

/-- Exact gated/tail frame, including inactive cells inside the output window. -/
theorem outputState_frame {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (x y : RegionName) (N B : Nat) (s : State α)
    (r : RegionName) (o : Nat)
    (hmiss : r ≠ y ∨ ∀ i : Fin B, s.pids 0 = 0 ∧ s.pids 0 * B + i.val < N →
      o ≠ s.pids 0 * B + i.val) :
    (outputState M cmp x y N B s).mem r o = s.mem r o := by
  unfold outputState
  split
  · apply State.masked_scatter_frame
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi => ho i.1 ⟨by assumption, hi⟩)
  · rfl

end VeriTile.Bench.TritonBenchG.ReluTritonKernel.FPProofs
