import VeriTile.Triton.DSL
import VeriTile.Triton.Semantics.BlockPadding
import VeriTile.Triton.Semantics.Step
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Source-level regression for block-load padding. The omitted and empty
options preserve unspecified values; only explicit zero selects zero. -/
namespace BlockPaddingDSLTests
open VeriTile.Triton
set_option linter.unusedSimpArgs false

private def xReg : RegionName := "x"

private def ptr : Op .blockPtr [2] :=
  .makeBlockPtrDyn "x" (.constNat 5) [3] [2] [2] [2]

private def prologue : List ComputeStmt :=
  [.assign .blockPtr [2] "p" (.alg ptr)]

def omitted : ComputeKernel := triton {
  p = tl.make_block_ptr($(xReg), base=$(5), shape=[3], strides=[2],
    offsets=[2], block_shape=[2])
  x = tl.load(p, boundary_check=([0] : List Nat), dtype=tl.float32)
}

def empty : ComputeKernel := triton {
  p = tl.make_block_ptr($(xReg), base=$(5), shape=[3], strides=[2],
    offsets=[2], block_shape=[2])
  x = tl.load(p, boundary_check=([0] : List Nat), padding_option="", dtype=tl.float32)
}

def zero : ComputeKernel := triton {
  p = tl.make_block_ptr($(xReg), base=$(5), shape=[3], strides=[2],
    offsets=[2], block_shape=[2])
  x = tl.load(p, boundary_check=([0] : List Nat), padding_option="zero", dtype=tl.float32)
}

def bf16 : ComputeKernel := triton {
  p = tl.make_block_ptr($(xReg), base=$(5), shape=[3], strides=[2],
    offsets=[2], block_shape=[2])
  x = tl.load(p, boundary_check=([0] : List Nat), dtype=tl.bfloat16)
}

def unchecked : ComputeKernel := triton {
  p = tl.make_block_ptr($(xReg), base=$(5), shape=[3], strides=[2],
    offsets=[2], block_shape=[2])
  x = tl.load(p, dtype=tl.float32)
}

theorem omitted_keeps_fp32 : omitted.surfaceBody = prologue ++
    [.assign .real [2] "x" (.compute (.load .fp32
      (.blockPtr (.ref .blockPtr [2] "p") [0]) .none .unspecified))] := rfl

theorem empty_is_omitted : empty = omitted := rfl

theorem zero_keeps_fp32 : zero.surfaceBody = prologue ++
    [.assign .real [2] "x" (.compute (.load .fp32
      (.blockPtr (.ref .blockPtr [2] "p") [0]) .none .zero))] := rfl

theorem bf16_keeps_storage : bf16.surfaceBody = prologue ++
    [.assign .bf16 [2] "x" (.alg (.load .bf16
      (.blockPtr (.ref .blockPtr [2] "p") [0]) .none .unspecified))] := rfl

theorem unchecked_keeps_no_checks : unchecked.surfaceBody = prologue ++
    [.assign .real [2] "x" (.compute (.load .fp32
      (.blockPtr (.ref .blockPtr [2] "p") []) .none .unspecified))] := rfl

theorem ordinary_pointer_empty_unchanged :
    (triton { x = tl.load($(xReg), padding_option="") } : ComputeKernel) =
      (triton { x = tl.load($(xReg)) } : ComputeKernel) := rfl

/-- The actual source execution sees the oracle at the checked-off lane. -/
theorem omitted_tail (s : BlockState) :
    (exec omitted s).bind (fun u => (u.regs .real [2] "x").map
      (fun v => v.data (1, PUnit.unit))) = some (some (s.undef "x" 11)) := by
  simp [exec, omitted, stepStmts, stepStmt, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType, evalOp, xReg, TileShape.indexToList,
    BlockState.paddingValue, BlockState.unspecifiedValue]
  rfl

theorem zero_tail (s : BlockState) :
    (exec zero s).bind (fun u => (u.regs .real [2] "x").map
      (fun v => v.data (1, PUnit.unit))) = some (some 0) := by
  simp [exec, zero, stepStmts, stepStmt, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType, evalOp, xReg, TileShape.indexToList,
    BlockState.defaultCarrier]
  rfl

/-- Without a boundary check, the source still reads memory at the tail. -/
theorem unchecked_tail (s : BlockState) :
    (exec unchecked s).bind (fun u => (u.regs .real [2] "x").map
      (fun v => v.data (1, PUnit.unit))) = some (some (s.readMem "x" 11)) := by
  simp [exec, unchecked, stepStmts, stepStmt, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, evalOp, xReg,
    TileShape.indexToList]
  rfl

/-- error: padding_option: unsupported value `nan`; only empty or zero padding is modeled -/
#guard_msgs in
example : ComputeKernel := triton {
  p = tl.make_block_ptr($(xReg), base=$(0), shape=[1], strides=[1],
    offsets=[0], block_shape=[2])
  x = tl.load(p, boundary_check=([0] : List Nat), padding_option="nan")
}

/-- error: tl.load: duplicate `padding_option=` kwarg -/
#guard_msgs in
example : ComputeKernel := triton {
  p = tl.make_block_ptr($(xReg), base=$(0), shape=[1], strides=[1],
    offsets=[0], block_shape=[2])
  x = tl.load(p, boundary_check=([0] : List Nat), padding_option="", padding_option="zero")
}

/-- error: tl.load: nonempty `padding_option` requires a block pointer -/
#guard_msgs in
example : ComputeKernel := triton {
  x = tl.load($(xReg), padding_option="zero")
}

/-- error: tl.store: `padding_option` is only valid on tl.load -/
#guard_msgs in
example : ComputeKernel := triton {
  tl.store($(xReg), 0.0, padding_option="")
}

#guard_msgs (drop info) in
#auditModuleAxioms
end BlockPaddingDSLTests
