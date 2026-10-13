import bench.tritonbench_g.log_softmax.Proofs.Active
import bench.tritonbench_g.log_softmax.Proofs.Source

namespace VeriTile.Bench.TritonBenchG.LogSoftmax.SourceRegion
open VeriTile.Triton
set_option maxHeartbeats 12000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem forward_typed_region_exec (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat) (s : BlockState) :
    exec (typedForward input output v OUT X M N K R C).eraseDType.toAlgKernel s =
      exec (log_softmax_kernel OUT X M N K R C).toAlgKernel s := by
  rw [typedForward_projection]
  cases v <;> simp [mathForward, commutedMathForward, log_softmax_kernel,
    ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, FloatDType.toTileDType,
    exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, Option.bind, Option.map, NumericDType.mul, Nat.mul_comm,
    FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot]

/-- Successful full region execution and raw-memory frame; undefined padding is arbitrary. -/
theorem forward_run (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat) (hC : 0 < C) (hK : 0 < K) (s : BlockState) :
    ∃ t, exec (typedForward input output v OUT X M N K R C).eraseDType.toAlgKernel s = some t ∧
      (∀ i : TileIndex [R,C], active (s.pids 0) M N R i →
        t.readMem OUT (outOffset (s.pids 0) (s.pids 1) N K R i) =
          logSoftmaxSpec s X (s.pids 0) (s.pids 1) M N K R C i) ∧
      (∀ r o, (∀ i : TileIndex [R,C], active (s.pids 0) M N R i →
        ¬(OUT = r ∧ outOffset (s.pids 0) (s.pids 1) N K R i = o)) →
        t.mem r o = s.mem r o) := by
  obtain ⟨t, he⟩ := source_forward_terminates OUT X M N K R C hC s
  refine ⟨t, (forward_typed_region_exec input output v OUT X M N K R C s).trans he, ?_, ?_⟩
  · intro i hi
    exact SourceReadback.forward_active OUT X M N K R C hK s t he i hi
  · intro r o hm
    exact source_forward_frame OUT X M N K R C hC s t he r o hm

theorem backward_typed_region_exec (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) (s : BlockState) :
    exec (typedBackward input upstream output v Y G DX M N K R C).eraseDType.toAlgKernel s =
      exec (log_softmax_backward_kernel Y G DX M N K R C).toAlgKernel s := by
  rw [typedBackward_projection]
  cases v <;> simp [mathBackward, commutedMathBackward, log_softmax_backward_kernel,
    ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, FloatDType.toTileDType,
    exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, Option.bind, Option.map, NumericDType.mul, Nat.mul_comm,
    FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot]

/-- Successful full region execution and raw-memory frame; undefined padding is arbitrary. -/
theorem backward_run (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) (hK : 0 < K) (s : BlockState) :
    ∃ t, exec (typedBackward input upstream output v Y G DX M N K R C).eraseDType.toAlgKernel s = some t ∧
      (∀ i : TileIndex [R,C], active (s.pids 0) M N R i →
        t.readMem DX (outOffset (s.pids 0) (s.pids 1) N K R i) =
          logSoftmaxBackwardSpec s Y G M N K R C i) ∧
      (∀ r o, (∀ i : TileIndex [R,C], active (s.pids 0) M N R i →
        ¬(DX = r ∧ outOffset (s.pids 0) (s.pids 1) N K R i = o)) →
        t.mem r o = s.mem r o) := by
  obtain ⟨t, he⟩ := source_backward_terminates Y G DX M N K R C s
  refine ⟨t, (backward_typed_region_exec input upstream output v Y G DX M N K R C s).trans he, ?_, ?_⟩
  · intro i hi
    exact SourceReadback.backward_active Y G DX M N K R C hK s t he i hi
  · intro r o hm
    exact source_backward_frame Y G DX M N K R C s t he r o hm

end VeriTile.Bench.TritonBenchG.LogSoftmax.SourceRegion
