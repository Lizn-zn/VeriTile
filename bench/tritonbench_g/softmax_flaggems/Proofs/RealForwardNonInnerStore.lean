import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerDenominator
import VeriTile.Triton.Memory.Frame.Basic

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def pass2Body (Y X : RegionName) (M N K B C : Nat) : List Stmt :=
  match ((softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.false).eraseDType.toAlgKernel.body[3]? : Option Stmt) with
  | some (.ifThenElse _ _ no) =>
    match (no[7]? : Option Stmt) with
    | some (.forRange _ _ _ _ body) => body
    | some (.forRangeDyn _ _ _ _ body) => body
    | _ => []
  | _ => []

def tileOffset {B C : Nat} (row N K kbase base : Nat) (i : TileIndex [B, C]) : Nat :=
  row * N * K + (base + i.1.val) * K + (kbase + i.2.1.val)

def tileActive {B C : Nat} (N K kbase base : Nat) (i : TileIndex [B, C]) : Bool :=
  decide (base + i.1.val < N ∧ kbase + i.2.1.val < K)

noncomputable def storeValue {B C : Nat} (s : BlockState) (X : RegionName)
    (row N K kbase base : Nat) (mx : Fin C → WithBot ℝ) (dz : Fin C → ℝ)
    (i : TileIndex [B, C]) : ℝ :=
  (WithBot.realDiv
    (WithBot.realExp (WithBot.realSub (score s X row N K kbase base B C i) (mx i.2.1)))
    (some (dz i.2.1))).unbotD 0

noncomputable def scatter {B C : Nat} (s : BlockState) (Y : RegionName)
    (off : TileIndex [B, C] → Nat) (value : TileIndex [B, C] → ℝ)
    (active : TileIndex [B, C] → Bool) : BlockState :=
  (TileShape.allIndices [B, C]).foldl
    (fun st i ↦ if active i then st.writeMem Y (off i) (value i) else st) s

noncomputable def storeMid {B C : Nat} (s : BlockState) (X : RegionName)
    (row N K kbase base : Nat) (mx : Fin C → WithBot ℝ) (dz : Fin C → ℝ) : BlockState :=
  let s₁ := s.setReg "n_offsets" .nat [B] ⟨fun i ↦ base + i.1.val⟩
  let s₂ := s₁.setReg "offsets" .nat [B, C] ⟨tileOffset row N K kbase base⟩
  let s₃ := s₂.setReg "mask" .bool [B, C] ⟨tileActive N K kbase base⟩
  let s₄ := s₃.setReg "inp" .real [B, C] ⟨score s X row N K kbase base B C⟩
  s₄.setReg "o" .real [B, C] ⟨fun i ↦ WithBot.realDiv
    (WithBot.realExp (WithBot.realSub (score s X row N K kbase base B C i) (mx i.2.1)))
    (some (dz i.2.1))⟩

 theorem pass2_step (Y X : RegionName) (M N K B C row kbase previous start : Nat)
    (s : BlockState) (mx : Fin C → WithBot ℝ) (dz : Fin C → ℝ)
    (hrow : s.regs .nat [] "pid_m" = some (Tile.scalar row))
    (hk : s.regs .nat [C] "k_offsets" = some ⟨fun i ↦ kbase + i.1.val⟩)
    (hstart : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hprev : s.regs .nat [] "previous_multiple" = some (Tile.scalar previous))
    (hm : s.regs .real [C] "m" = some ⟨fun i ↦ mx i.1⟩)
    (hz : s.regs .real [C] "z" = some ⟨fun i ↦ some (dz i.1)⟩) :
    ∃ mid, stepStmts (pass2Body Y X M N K B C) s =
        some (scatter (B := B) (C := C) mid Y (tileOffset row N K kbase (previous - start))
          (storeValue s X row N K kbase (previous - start) mx dz)
          (tileActive N K kbase (previous - start))) ∧
      mid.mem = s.mem ∧ mid.pids = s.pids ∧ mid.numPids = s.numPids ∧ mid.undef = s.undef ∧
      mid.regs .nat [] "pid_m" = s.regs .nat [] "pid_m" ∧
      mid.regs .nat [C] "k_offsets" = s.regs .nat [C] "k_offsets" ∧
      mid.regs .nat [] "previous_multiple" = s.regs .nat [] "previous_multiple" ∧
      mid.regs .real [C] "m" = s.regs .real [C] "m" ∧
      mid.regs .real [C] "z" = s.regs .real [C] "z" := by
  simp [pass2Body, softmax_kernel_non_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp only [stepStmts, stepStmt, evalOp.eq_def, hrow, hk, hstart, hprev, hm, hz,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
    String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind, Option.pure_def]
  refine ⟨storeMid (B := B) (C := C) s X row N K kbase (previous - start) mx dz, ?_, rfl, rfl, rfl, rfl,
    ?_, ?_, ?_, ?_, ?_⟩
  · simp [storeMid, scatter, tileOffset, tileActive, storeValue, score,
      Tile.bop, Tile.cop, Tile.expandDim, Tile.uop, NumericDType.add, NumericDType.mul,
      NumericDType.sub, NumericDType.div, ComparableDType.lt,
      BlockState.writeMemTyped_real, BlockState.readMemValue_real]
    have hmask : (⟨fun i ↦ decide (previous - start + i.1.val < N) &&
        decide (kbase + i.2.1.val < K)⟩ : Tile .bool [B, C]) =
        ⟨tileActive N K kbase (previous - start)⟩ := by
      apply Tile.ext
      intro i
      by_cases hn : previous - start + i.1.val < N <;>
        by_cases hk' : kbase + i.2.1.val < K <;> simp [tileActive, hn, hk']
    rw [hmask]
    rfl
  all_goals simp only [storeMid, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
    not_false_eq_true, hrow, hk, hprev, hm, hz]

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
