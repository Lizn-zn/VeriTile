import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.var_len_copy.Kernels
import VeriTile.Triton.KernelLemmas.ScatterStore

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteReal
open VeriTile Triton DiscreteSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def representative : Kind → Storage
  | .signed => .int32
  | .unsigned => .uint32
  | .boolean => .bool

def source (k : Kind) (os ol ns : Region .int) (src dst : RegionName) (B : Nat) : Kernel :=
  (kernel (representative k) Bool.false os ol ns src dst B).eraseDType.toAlgKernel

def body (k : Kind) (os ol ns : Region .int) (src dst : RegionName) (B : Nat) : List Stmt :=
  match ((source k os ol ns src dst B).body[6]? : Option Stmt) with
  | some (.forRangeInt _ _ _ _ b) => b
  | _ => []

def address {B : Nat} (start current : Int) (i : TileIndex [B]) : Nat :=
  (start + current + (i.1.val : Int)).toNat

def active {B : Nat} (length : Int) (i : TileIndex [B]) : Bool :=
  decide ((i.1.val : Int) < length)

noncomputable def value (k : Kind) {B : Nat} (s : BlockState) (src : RegionName)
    (length start current : Int) (i : TileIndex [B]) : TileCarrier k.dtype :=
  if active length i then s.readMemValue k.dtype src (address start current i)
  else BlockState.defaultCarrier k.dtype

noncomputable def scatter (k : Kind) {B : Nat} (s : BlockState) (dst : RegionName)
    (start current length : Int) (v : TileIndex [B] → TileCarrier k.dtype) : BlockState :=
  (TileShape.allIndices [B]).foldl (fun st i ↦
    if active length i then st.writeMemTyped k.dtype dst (address start current i) (v i) else st) s

noncomputable def midState (k : Kind) {B : Nat} (s : BlockState) (src : RegionName)
    (length start current : Int) : BlockState :=
  s.setReg "v" k.dtype [B] ⟨fun i ↦ value k s src length start current i⟩

theorem step (k : Kind) (os ol ns : Region .int) (src dst : RegionName) (B : Nat)
    (s : BlockState) (length oldStart newStart current : Int)
    (hL : s.regs .int [] "length" = some (Tile.scalar length))
    (hO : s.regs .int [] "old_start" = some (Tile.scalar oldStart))
    (hN : s.regs .int [] "new_start" = some (Tile.scalar newStart))
    (hI : s.regs .int [] "i" = some (Tile.scalar current))
    (hOld : s.regs .nat [B] "old_offset" = some (Tile.vec fun i ↦ i.val))
    (hNew : s.regs .nat [B] "new_offset" = some (Tile.vec fun i ↦ i.val)) :
    stepStmts (body k os ol ns src dst B) s =
      some (scatter k (B := B) (midState k (B := B) s src length oldStart current) dst newStart current length
        (value k s src length oldStart current)) := by
  cases k
  all_goals simp [body, source, representative, kernel, Kind.dtype, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  all_goals simp only [stepStmts, stepStmt, evalOp.eq_def, hL, hO, hN, hI, hOld, hNew,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
    String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind, Option.pure_def]
  all_goals simp [scatter, midState, value, address, active,
    Tile.bop, Tile.cop, Tile.uop, Tile.vec, NumericDType.add, ComparableDType.lt,
    Kind.dtype]
  all_goals simp only [apply_ite]
  all_goals simp only [Region.cast, value, active, address, Kind.dtype]
  all_goals rfl

end VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteReal

#guard_msgs (drop info) in
#auditModuleAxioms
