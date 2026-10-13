import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerStore
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerMath

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def fullStoreBody (Y X : RegionName) (M N B : Nat) : List Stmt :=
  match ((fallback Y X M N B)[12]? : Option Stmt) with
  | some (.forRange _ _ _ _ body) => body
  | _ => []

noncomputable def fullStoreMid (s : BlockState) (X : RegionName) (row N base B : Nat)
    (mx : WithBot ℝ) (z : ℝ) : BlockState :=
  let ns : Tile .nat [B] := ⟨fun i ↦ base + i.1.val⟩
  let inp := score s X row N base B
  let out := fun i ↦ WithBot.realDiv (WithBot.realExp (WithBot.realSub (inp i) mx)) (some z)
  let s₁ := s.setReg "n_offsets" .nat [B] ns
  let s₃ := s₁.setReg "inp" .real [B] ⟨inp⟩
  s₃.setReg "o" .real [B] ⟨out⟩

theorem fullStore_step (Y X : RegionName) (M N B row previous start : Nat)
    (s : BlockState) (mx : WithBot ℝ) (z : ℝ)
    (hfull : previous - start + B ≤ N)
    (hp : s.regs .ptr [] "input_ptr" = some (Tile.scalar (X, row * N)))
    (ho : s.regs .ptr [] "output_ptr" = some (Tile.scalar (Y, row * N)))
    (hprev : s.regs .nat [] "previous_multiple" = some (Tile.scalar previous))
    (hstart : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hm : s.regs .real [] "m" = some (Tile.scalar mx))
    (hz : s.regs .real [] "z" = some (Tile.scalar (some z))) :
    stepStmts (fullStoreBody Y X M N B) s =
      some (scatter (B := B) (fullStoreMid s X row N (previous - start) B mx z) Y
        (tileOffset row N (previous - start))
        (storeValue s X row N (previous - start) mx z) (tileActive N (previous - start))) := by
  simp [fullStoreBody, fallback, source, softmax_kernel_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp only [stepStmts, stepStmt, evalOp.eq_def, hp, ho, hprev, hstart, hm, hz,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
    String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind, Option.pure_def]
  have hi : ∀ i : Fin B, previous - start + i.val < N := by
    intro i
    have := i.isLt
    omega
  simp [fullStoreMid, scatter, tileOffset, tileActive, storeValue, score,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, NumericDType.add, NumericDType.sub,
    NumericDType.mul, NumericDType.div, ComparableDType.lt,
    BlockState.writeMemTyped_real, BlockState.readMemValue_real, hi]
  congr 1
  congr 3
  funext i
  simp [score, hi]

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
