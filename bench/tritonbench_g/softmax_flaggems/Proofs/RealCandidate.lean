import bench.tritonbench_g.softmax_flaggems.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Memory.FlattenTolerant

/-! Whole-state real-interpreter transport for the discrete candidate change.
This lemma supplies no mathematical output theorem on its own. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealCandidate
open VeriTile Triton FullSource
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

inductive PrefixKind where
  | forwardNonInner | backwardNonInner | backwardInner

def indexPrefix (kind : PrefixKind) (swapped : Bool) (B : Nat) : List Stmt :=
  let (reg, dest) := match kind with
    | .forwardNonInner => ("pid_k", "k_offsets")
    | .backwardNonInner => ("pid_k", "offsets_k")
    | .backwardInner => ("pid_m", "m_offsets")
  let init := match kind with
    | .forwardNonInner => [.assign .nat [] "pid_k" (.programId 1),
        .assign .nat [] "pid_m" (.programId 0)]
    | .backwardNonInner => [.assign .nat [] "pid_m" (.programId 0),
        .assign .nat [] "pid_k" (.programId 1)]
    | .backwardInner => [.assign .nat [] "pid_m" (.programId 0)]
  init ++ [.assign .nat [B] dest (.add .nat .scalarL
    (if swapped then .mul .nat .nil (.constNat B) (.ref .nat [] reg)
     else .mul .nat .nil (.ref .nat [] reg) (.constNat B)) (.arange B))]

theorem prefix_flat (A : FlatAlloc) (kind : PrefixKind) (B : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts (indexPrefix kind Bool.false B)) s =
    stepStmts (A.flattenStmts (indexPrefix kind Bool.true B)) s := by
  cases kind <;>
    simp [indexPrefix, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp,
      stepStmts, stepStmt, evalOp.eq_def, Tile.bop, NumericDType.mul,
      NumericDType.add, Nat.mul_comm]

theorem forward_non_inner_body (input output : Storage) (Y X : RegionName)
    (M N K BN BK : Nat) (oneTile : Bool) :
    (softmax_kernel_non_inner_candidate input output Y X M N K BN BK oneTile).eraseDType.toAlgKernel.body =
      indexPrefix .forwardNonInner Bool.true BK ++
        (softmax_kernel_non_inner input output Y X M N K BN BK oneTile).eraseDType.toAlgKernel.body.drop 3 := by
  cases input <;> cases output
  all_goals simp [softmax_kernel_non_inner, softmax_kernel_non_inner_candidate,
    indexPrefix, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

theorem forward_non_inner_original_body (input output : Storage) (Y X : RegionName)
    (M N K BN BK : Nat) (oneTile : Bool) :
    (softmax_kernel_non_inner input output Y X M N K BN BK oneTile).eraseDType.toAlgKernel.body =
      indexPrefix .forwardNonInner Bool.false BK ++
        (softmax_kernel_non_inner input output Y X M N K BN BK oneTile).eraseDType.toAlgKernel.body.drop 3 := by
  cases input <;> cases output
  all_goals simp [softmax_kernel_non_inner, softmax_kernel_non_inner_candidate,
    indexPrefix, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

theorem forward_non_inner_flat_exec (A : FlatAlloc) (input output : Storage) (Y X : RegionName)
    (M N K BN BK : Nat) (oneTile : Bool) (s : BlockState) :
    exec (A.flattenKernel (softmax_kernel_non_inner input output Y X M N K BN BK oneTile).eraseDType.toAlgKernel) s =
    exec (A.flattenKernel (softmax_kernel_non_inner_candidate input output Y X M N K BN BK oneTile).eraseDType.toAlgKernel) s := by
  unfold exec FlatAlloc.flattenKernel
  conv_lhs => rw [forward_non_inner_original_body]
  conv_rhs => rw [forward_non_inner_body]
  simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, prefix_flat]

theorem backward_non_inner_body (storage : Storage) (P G Y : RegionName)
    (M N K BN BK : Nat) (oneTile : Bool) :
    (softmax_backward_kernel_non_inner_candidate storage P G Y M N K BN BK oneTile).eraseDType.toAlgKernel.body =
      indexPrefix .backwardNonInner Bool.true BK ++
        (softmax_backward_kernel_non_inner storage P G Y M N K BN BK oneTile).eraseDType.toAlgKernel.body.drop 3 := by
  cases storage
  all_goals simp [softmax_backward_kernel_non_inner, softmax_backward_kernel_non_inner_candidate,
    indexPrefix, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

theorem backward_non_inner_original_body (storage : Storage) (P G Y : RegionName)
    (M N K BN BK : Nat) (oneTile : Bool) :
    (softmax_backward_kernel_non_inner storage P G Y M N K BN BK oneTile).eraseDType.toAlgKernel.body =
      indexPrefix .backwardNonInner Bool.false BK ++
        (softmax_backward_kernel_non_inner storage P G Y M N K BN BK oneTile).eraseDType.toAlgKernel.body.drop 3 := by
  cases storage
  all_goals simp [softmax_backward_kernel_non_inner, softmax_backward_kernel_non_inner_candidate,
    indexPrefix, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

theorem backward_non_inner_flat_exec (A : FlatAlloc) (storage : Storage) (P G Y : RegionName)
    (M N K BN BK : Nat) (oneTile : Bool) (s : BlockState) :
    exec (A.flattenKernel (softmax_backward_kernel_non_inner storage P G Y M N K BN BK oneTile).eraseDType.toAlgKernel) s =
    exec (A.flattenKernel (softmax_backward_kernel_non_inner_candidate storage P G Y M N K BN BK oneTile).eraseDType.toAlgKernel) s := by
  unfold exec FlatAlloc.flattenKernel
  conv_lhs => rw [backward_non_inner_original_body]
  conv_rhs => rw [backward_non_inner_body]
  simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, prefix_flat]

theorem backward_inner_body (storage : Storage) (P G Y : RegionName)
    (M N BM BN : Nat) (oneTile : Bool) :
    (softmax_backward_kernel_inner_candidate storage P G Y M N BM BN oneTile).eraseDType.toAlgKernel.body =
      indexPrefix .backwardInner Bool.true BM ++
        (softmax_backward_kernel_inner storage P G Y M N BM BN oneTile).eraseDType.toAlgKernel.body.drop 2 := by
  cases storage
  all_goals simp [softmax_backward_kernel_inner, softmax_backward_kernel_inner_candidate,
    indexPrefix, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

theorem backward_inner_original_body (storage : Storage) (P G Y : RegionName)
    (M N BM BN : Nat) (oneTile : Bool) :
    (softmax_backward_kernel_inner storage P G Y M N BM BN oneTile).eraseDType.toAlgKernel.body =
      indexPrefix .backwardInner Bool.false BM ++
        (softmax_backward_kernel_inner storage P G Y M N BM BN oneTile).eraseDType.toAlgKernel.body.drop 2 := by
  cases storage
  all_goals simp [softmax_backward_kernel_inner, softmax_backward_kernel_inner_candidate,
    indexPrefix, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

theorem backward_inner_flat_exec (A : FlatAlloc) (storage : Storage) (P G Y : RegionName)
    (M N BM BN : Nat) (oneTile : Bool) (s : BlockState) :
    exec (A.flattenKernel (softmax_backward_kernel_inner storage P G Y M N BM BN oneTile).eraseDType.toAlgKernel) s =
    exec (A.flattenKernel (softmax_backward_kernel_inner_candidate storage P G Y M N BM BN oneTile).eraseDType.toAlgKernel) s := by
  unfold exec FlatAlloc.flattenKernel
  conv_lhs => rw [backward_inner_original_body]
  conv_rhs => rw [backward_inner_body]
  simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, prefix_flat]

def thenBody (k : Kernel) : List Stmt :=
  match (k.body[1]? : Option Stmt) with
  | some (Stmt.ifThenElse _ yes _) => yes
  | _ => []

def elseBody (k : Kernel) : List Stmt :=
  match (k.body[1]? : Option Stmt) with
  | some (Stmt.ifThenElse _ _ no) => no
  | _ => []

def innerPrefix (swapped : Bool) (Y X : RegionName) (N B : Nat) : List Stmt :=
  let base := if swapped then .mul .nat .nil (.constNat N) (.ref .nat [] "pid_m")
    else .mul .nat .nil (.ref .nat [] "pid_m") (.constNat N)
  [.assign .real [B] "m" (.full [B] .negInf),
   .assign .real [B] "z" (.full [B] (.const 0.0)),
   .assign .ptr [] "input_ptr" (.ptrAdd .nil (.ptrBase X) base),
   .assign .ptr [] "output_ptr" (.ptrAdd .nil (.ptrBase Y) base)]

theorem inner_prefix_flat (A : FlatAlloc) (Y X : RegionName) (N B : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts (innerPrefix Bool.false Y X N B)) s =
    stepStmts (A.flattenStmts (innerPrefix Bool.true Y X N B)) s := by
  simp [innerPrefix, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.ptrAdd, NumericDType.mul, Nat.mul_comm]

theorem forward_inner_body (input output : Storage) (Y X : RegionName)
    (M N B : Nat) (oneTile : Bool) :
    (softmax_kernel_inner_candidate input output Y X M N B oneTile).eraseDType.toAlgKernel.body =
      [.assign .nat [] "pid_m" (.programId 0),
       .ifThenElse (.constBool oneTile)
         (thenBody (softmax_kernel_inner input output Y X M N B oneTile).eraseDType.toAlgKernel)
         (innerPrefix Bool.true Y X N B ++
           (elseBody (softmax_kernel_inner input output Y X M N B oneTile).eraseDType.toAlgKernel).drop 4)] := by
  cases input <;> cases output
  all_goals simp [softmax_kernel_inner, softmax_kernel_inner_candidate, thenBody, elseBody,
    innerPrefix, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

theorem forward_inner_original_body (input output : Storage) (Y X : RegionName)
    (M N B : Nat) (oneTile : Bool) :
    (softmax_kernel_inner input output Y X M N B oneTile).eraseDType.toAlgKernel.body =
      [.assign .nat [] "pid_m" (.programId 0),
       .ifThenElse (.constBool oneTile)
         (thenBody (softmax_kernel_inner input output Y X M N B oneTile).eraseDType.toAlgKernel)
         (innerPrefix Bool.false Y X N B ++
           (elseBody (softmax_kernel_inner input output Y X M N B oneTile).eraseDType.toAlgKernel).drop 4)] := by
  cases input <;> cases output
  all_goals simp [softmax_kernel_inner, thenBody, elseBody,
    innerPrefix, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

theorem forward_inner_flat_exec (A : FlatAlloc) (input output : Storage) (Y X : RegionName)
    (M N B : Nat) (oneTile : Bool) (s : BlockState) :
    exec (A.flattenKernel (softmax_kernel_inner input output Y X M N B oneTile).eraseDType.toAlgKernel) s =
    exec (A.flattenKernel (softmax_kernel_inner_candidate input output Y X M N B oneTile).eraseDType.toAlgKernel) s := by
  unfold exec FlatAlloc.flattenKernel
  conv_lhs => rw [forward_inner_original_body]
  conv_rhs => rw [forward_inner_body]
  cases oneTile <;>
    simp [FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp,
      stepStmts, stepStmt, evalOp.eq_def, ProgramRewrite.append, inner_prefix_flat]
  all_goals simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, inner_prefix_flat]

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealCandidate
