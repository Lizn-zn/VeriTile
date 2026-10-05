import bench.examples.OnlineSoftmax.Kernels
import VeriTile.Triton.Memory.TraceProofs

/-! Bounds for every scalar read in the recurrence and every lane in the
second-pass load/store. Both passes stay within the same row window. -/
namespace VeriTile.Bench.Examples.OnlineSoftmax.Memory
open VeriTile Triton TraceProofs Kernels
set_option maxHeartbeats 3200000
set_option linter.unusedSimpArgs false

private def loopBody (x : RegionName) (N : Nat) : List Stmt :=
  match ((onlineNormalizerKernel x "y" N).toAlgKernel.body[3]? : Option Stmt) with
  | some (.forLoop _ _ body) => body
  | _ => []

private theorem body_pid (x : RegionName) (N : Nat) (s t : BlockState)
    (h : stepStmts (loopBody x N) s = some t) :
    t.regs .nat [] "pid" = s.regs .nat [] "pid" := by
  obtain ⟨s1, h1, h⟩ := cons_inv h
  obtain ⟨v1, _, rfl⟩ := assign_inv h1
  obtain ⟨s2, h2, h⟩ := cons_inv h
  obtain ⟨v2, _, rfl⟩ := assign_inv h2
  obtain ⟨s3, h3, h⟩ := cons_inv h
  obtain ⟨v3, _, rfl⟩ := assign_inv h3
  obtain ⟨s4, h4, h⟩ := cons_inv h
  obtain ⟨v4, _, rfl⟩ := assign_inv h4
  have ht := Option.some.inj (by simpa using h)
  subst t
  simp [BlockState.setReg]

private theorem body_safe (x : RegionName) (N pid i : Nat) (hi : i < N)
    (bounds : RegionBounds) (hx : pid * N + N ≤ bounds x) (s : BlockState)
    (hp : s.regs .nat [] "pid" = some (Tile.scalar pid)) :
    Stmt.TraceSafeList bounds (loopBody x N) (s.setReg "i" .nat [] (Tile.scalar i)) := by
  apply assign_cons
  · simp only [Op.SafeAt.eq_def, and_true, true_and]
    apply region_safe (offsets := Tile.scalar (pid * N + i))
    · simp [evalOp.eq_def, BlockState.setReg, hp, Tile.bop, NumericDType.add, NumericDType.mul]
    · intro _
      exact lt_of_lt_of_le (Nat.add_lt_add_left hi _) hx
  intro xi
  apply assign_cons (by simp [Op.SafeAt.eq_def])
  intro mnew
  apply assign_cons (by simp [Op.SafeAt.eq_def])
  intro l
  apply assign_cons (by simp [Op.SafeAt.eq_def])
  intro m
  exact .nil_intro

private theorem loop_safe_all (x : RegionName) (N pid : Nat) (bounds : RegionBounds)
    (hx : pid * N + N ≤ bounds x) (s : BlockState)
    (hp : s.regs .nat [] "pid" = some (Tile.scalar pid)) :
    Stmt.TraceSafe bounds (.forLoop "i" N (loopBody x N)) s := by
  rw [Stmt.TraceSafe]
  apply loop_safe (P := fun t => t.regs .nat [] "pid" = some (Tile.scalar pid))
    (n := N) ?_ ?_ N 0 s (by omega) hp
  · intro i t hi ht
    exact body_safe x N pid i hi bounds hx t ht
  · intro i t u _ ht hu
    rw [body_pid x N _ _ hu]
    simpa [BlockState.setReg] using ht

theorem traceSafe (x y : RegionName) (N : Nat) (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * N + N ≤ bounds x) (hy : s.pid * N + N ≤ bounds y) :
    Kernel.TraceSafe bounds (onlineSoftmaxKernel x y N).toAlgKernel s := by
  change Stmt.TraceSafeList bounds _ s
  apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
  intro value hv
  simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
    NumericDType.add, NumericDType.mul] at hv
  subst value
  apply assign_cons (by simp [Op.SafeAt.eq_def])
  intro m
  apply assign_cons (by simp [Op.SafeAt.eq_def])
  intro l
  refine Stmt.TraceSafeList.cons_intro (loop_safe_all x N s.pid bounds hx _ (by simp [BlockState.setReg])) ?_
  intro q hq
  have hpid : q.pids 0 = s.pids 0 := by simpa using stepStmt_pid hq
  apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
  intro value hv
  simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
    NumericDType.add, NumericDType.mul, hpid] at hv
  subst value
  apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
  intro value hv
  simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
    NumericDType.add, NumericDType.mul, hpid] at hv
  subst value
  apply assign_cons
  · simp only [Op.SafeAt.eq_def, and_true, true_and]
    apply region_safe (offsets := ⟨fun i : TileIndex [N] => s.pid * N + i.1.val⟩)
    · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        NumericDType.add, NumericDType.mul, hpid]
    · intro i
      exact lt_of_lt_of_le (Nat.add_lt_add_left i.1.isLt _) hx
  intro xs
  apply assign_cons (by simp [Op.SafeAt.eq_def])
  intro e
  apply assign_cons (by simp [Op.SafeAt.eq_def])
  intro ys
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt, Op.SafeAt.eq_def, and_true, true_and]
  apply region_safe (offsets := ⟨fun i : TileIndex [N] => s.pid * N + i.1.val⟩)
  · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
      NumericDType.add, NumericDType.mul, hpid]
  · intro i
    exact lt_of_lt_of_le (Nat.add_lt_add_left i.1.isLt _) hy

theorem flattenOk (x y : RegionName) (N : Nat) :
    (onlineSoftmaxKernel x y N).toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk, onlineSoftmaxKernel, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

end VeriTile.Bench.Examples.OnlineSoftmax.Memory
