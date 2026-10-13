import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.var_len_copy.Kernels
import VeriTile.Triton.KernelLemmas.ScatterStore

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal
open VeriTile Triton TypedSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def source (os ol ns : Region .int) (src dst : RegionName) (B : Nat) : Kernel :=
  (kernel .fp32 .fp32 Bool.false os ol ns src dst B).eraseDType.toAlgKernel

def body (os ol ns : Region .int) (src dst : RegionName) (B : Nat) : List Stmt :=
  match ((source os ol ns src dst B).body[6]? : Option Stmt) with
  | some (.forRangeInt _ _ _ _ b) => b
  | _ => []

def address {B : Nat} (start current : Int) (i : TileIndex [B]) : Nat :=
  (start + current + (i.1.val : Int)).toNat

def active {B : Nat} (length : Int) (i : TileIndex [B]) : Bool :=
  decide ((i.1.val : Int) < length)

noncomputable def value {B : Nat} (s : BlockState) (src : RegionName)
    (length start current : Int) (i : TileIndex [B]) : ℝ :=
  if active length i then s.readMem src (address start current i)
  else s.undef src (address start current i)

noncomputable def scatter {B : Nat} (s : BlockState) (dst : RegionName)
    (start current length : Int) (v : TileIndex [B] → ℝ) : BlockState :=
  (TileShape.allIndices [B]).foldl (fun st i ↦
    if active length i then st.writeMem dst (address start current i) (v i) else st) s

noncomputable def midState {B : Nat} (s : BlockState) (src : RegionName)
    (length start current : Int) : BlockState :=
  s.setReg "v" .real [B] ⟨fun i ↦ some (value s src length start current i)⟩

theorem step (os ol ns : Region .int) (src dst : RegionName) (B : Nat)
    (s : BlockState) (length oldStart newStart current : Int)
    (hL : s.regs .int [] "length" = some (Tile.scalar length))
    (hO : s.regs .int [] "old_start" = some (Tile.scalar oldStart))
    (hN : s.regs .int [] "new_start" = some (Tile.scalar newStart))
    (hI : s.regs .int [] "i" = some (Tile.scalar current))
    (hOld : s.regs .nat [B] "old_offset" = some (Tile.vec fun i ↦ i.val))
    (hNew : s.regs .nat [B] "new_offset" = some (Tile.vec fun i ↦ i.val)) :
    stepStmts (body os ol ns src dst B) s =
      some (scatter (B := B) (midState (B := B) s src length oldStart current) dst newStart current length
        (value s src length oldStart current)) := by
  simp [body, source, kernel, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp only [stepStmts, stepStmt, evalOp.eq_def, hL, hO, hN, hI, hOld, hNew,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
    String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind, Option.pure_def]
  simp [scatter, midState, value, address, active,
    Tile.bop, Tile.cop, Tile.uop, Tile.vec, NumericDType.add, ComparableDType.lt,
    BlockState.writeMemTyped_real, BlockState.readMemValue_real]
  simp only [apply_ite]
  rfl

end VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
