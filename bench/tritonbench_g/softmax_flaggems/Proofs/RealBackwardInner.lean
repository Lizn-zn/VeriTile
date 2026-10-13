import bench.tritonbench_g.softmax_flaggems.Kernels
import bench.tritonbench_g.softmax_flaggems.Proofs.RealMath
import bench.tritonbench_g.softmax_flaggems.Proofs.RealLoop
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.KernelLemmas.ScatterStore
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def source (P G Y : RegionName) (M N B C : Nat) (one : Bool) : Kernel :=
  (softmax_backward_kernel_inner .fp32 P G Y M N C B one).eraseDType.toAlgKernel

def branch (P G Y : RegionName) (M N B C : Nat) (one : Bool) : List Stmt :=
  match ((source P G Y M N B C one).body[2]? : Option Stmt) with
  | some (.ifThenElse _ yes no) => if one then yes else no
  | _ => []

def pass1Body (P G Y : RegionName) (M N B C : Nat) : List Stmt :=
  match ((branch P G Y M N B C Bool.false)[3]? : Option Stmt) with
  | some (.forRangeDyn _ _ _ _ body) | some (.forRange _ _ _ _ body) => body
  | _ => []

def offset (N r n : Nat) : Nat := r*N+n

def tileOffset {B C : Nat} (N rbase start : Nat) (i : TileIndex [C,B]) : Nat :=
  offset N (rbase+i.1.val) (start+i.2.1.val)

noncomputable def sample (s : BlockState) (R : RegionName) (M N r n : Nat) : ℝ :=
  if r<M ∧ n<N then s.readMem R (offset N r n) else s.undef R (offset N r n)

noncomputable def tileSample {B C : Nat} (s : BlockState) (R : RegionName)
    (M N rbase start : Nat) (i : TileIndex [C,B]) : ℝ :=
  sample s R M N (rbase+i.1.val) (start+i.2.1.val)

theorem pass1_step (P G Y : RegionName) (M N B C rbase start : Nat)
    (s : BlockState) (z : TileIndex [C,B] → ℝ)
    (hk : s.regs .nat [C] "m_offsets" = some ⟨fun i ↦ rbase+i.1.val⟩)
    (hn : s.regs .nat [B] "n_offsets" = some ⟨fun i ↦ start+i.1.val⟩)
    (ho : s.regs .nat [C,B] "offsets" = some ⟨tileOffset N rbase start⟩)
    (hz : s.regs .real [C,B] "scale" = some ⟨fun i ↦ some (z i)⟩) :
    ∃ next, stepStmts (pass1Body P G Y M N B C) s = some next ∧
      next.mem=s.mem ∧ next.pids=s.pids ∧ next.numPids=s.numPids ∧ next.undef=s.undef ∧
      next.regs .nat [C] "m_offsets" = some ⟨fun i ↦ rbase+i.1.val⟩ ∧
      next.regs .nat [B] "n_offsets" = some ⟨fun i ↦ start+B+i.1.val⟩ ∧
      next.regs .nat [C,B] "offsets" = some ⟨tileOffset N rbase (start+B)⟩ ∧
      next.regs .real [C,B] "scale" = some ⟨fun i ↦ some
        (z i + tileSample s P M N rbase start i * tileSample s G M N rbase start i)⟩ := by
  simp [pass1Body, branch, source, softmax_backward_kernel_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp only [stepStmts, stepStmt, evalOp.eq_def, hk, hn, ho, hz,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
    String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind, Option.pure_def]
  refine ⟨_, rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simpa only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] using hk
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq,
      String.reduceEq, not_false_eq_true]
    congr 1
    apply Tile.ext
    intro i
    simp [Tile.bop, NumericDType.add, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq,
      String.reduceEq, not_false_eq_true]
    congr 1
    apply Tile.ext
    intro i
    simp [Tile.bop, NumericDType.add, NumericDType.mul, tileOffset, offset, Nat.add_mul]
    ring
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq,
      String.reduceEq, not_false_eq_true]
    congr 1
    apply Tile.ext
    intro i
    simp [Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, tileSample, sample, BlockState.paddingValue,
      BlockState.readMemValue_real]
    split_ifs <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
