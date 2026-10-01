/- Real correctness of the fused SiLU kernel and its original three-stage
materialized pipeline. Both implement residual + silu(x * gate). The source
kernels retain their output casts; correctness erases numerical precision.
The pipeline here retains the original single-launch concatenation scope. -/
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.FusedSiLUCorrect

open VeriTile Triton
open scoped VeriTile.Triton.KernelIO₃

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

/-- Fused SiLU with a bf16-rounded output store. -/
def fusedSiLUKernel (xReg gateReg residualReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  x        := tl.load($(xReg) + offsets)
  gate     := tl.load($(gateReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  z        := x * gate
  silu     := z * tl.sigmoid(z)
  y        := residual + silu
  tl.store($(outReg) + offsets, (y).to(tl.bfloat16))
}

/-- Step A: materialize `z = x * gate` into the ℝ scratch tensor `z`. -/
def siluStepGate (xReg gateReg zReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  x       := tl.load($(xReg) + offsets)
  gate    := tl.load($(gateReg) + offsets)
  z       := x * gate
  tl.store($(zReg) + offsets, z)
}

/-- Step B: materialize `silu = z * sigmoid(z)` into the ℝ scratch tensor `silu`. -/
def siluStepSilu (zReg siluReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  z       := tl.load($(zReg) + offsets)
  silu    := z * tl.sigmoid(z)
  tl.store($(siluReg) + offsets, silu)
}

/-- Step C: `out = residual + silu`, with a bf16-rounded output store. -/
def siluStepResidual (siluReg residualReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  silu     := tl.load($(siluReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  y        := residual + silu
  tl.store($(outReg) + offsets, (y).to(tl.bfloat16))
}

/-- The unfused pipeline as one kernel: `ComputeKernel.seq` of the three step
kernels — the concatenation of their bodies (registers flow across the seams).
The staged execution is recovered by `unfused_exec_split` below. -/
def unfusedSiLUKernel
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel :=
  ComputeKernel.seq [xReg, gateReg, residualReg] [outReg]
    [siluStepGate xReg gateReg zReg blockSize,
     siluStepSilu zReg siluReg blockSize,
     siluStepResidual siluReg residualReg outReg blockSize]


/-- Executable real counterparts, used by the mathematical interpretation. -/
def fusedMathKernel (xReg gateReg residualReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  x        := tl.load($(xReg) + offsets)
  gate     := tl.load($(gateReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  z        := x * gate
  silu     := z * tl.sigmoid(z)
  y        := residual + silu
  tl.store($(outReg) + offsets, y)
}

def residualMathKernel (siluReg residualReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  silu     := tl.load($(siluReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  y        := residual + silu
  tl.store($(outReg) + offsets, y)
}

def unfusedMathKernel (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (B : Nat) : ComputeKernel :=
  ComputeKernel.seq [xReg, gateReg, residualReg] [outReg]
    [siluStepGate xReg gateReg zReg B,
     siluStepSilu zReg siluReg B,
     residualMathKernel siluReg residualReg outReg B]

private theorem erase_output_cast (B : Nat) :
    (Op.castFloat .real .bf16 (Op.ref .real [B] "y")).eraseDType =
      Op.ref .real [B] "y" := by
  rw [Op.eraseDType_castFloat .real .bf16 (Op.ref .real [B] "y")]
  change (Op.ref .real [B] "y").eraseDType = Op.ref .real [B] "y"
  rw [Op.eraseDType_ref]
  rfl

theorem fused_projection (xReg gateReg residualReg outReg : RegionName) (B : Nat) :
    (fusedSiLUKernel xReg gateReg residualReg outReg B).eraseDType.toAlgorithm? =
      Except.ok (fusedMathKernel xReg gateReg residualReg outReg B).toAlgKernel := by
  simp [fusedSiLUKernel, fusedMathKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeExpr.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  exact erase_output_cast B

theorem unfused_projection (xReg gateReg residualReg zReg siluReg outReg : RegionName) (B : Nat) :
    (unfusedSiLUKernel xReg gateReg residualReg zReg siluReg outReg B).eraseDType.toAlgorithm? =
      Except.ok (unfusedMathKernel xReg gateReg residualReg zReg siluReg outReg B).toAlgKernel := by
  simp [unfusedSiLUKernel, unfusedMathKernel, siluStepGate, siluStepSilu, siluStepResidual, residualMathKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeExpr.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  exact erase_output_cast B

theorem gate_region_run (B : Nat) (s : BlockState) (xs gs : Fin B → ℝ)
    (hxs : ∀ i : Fin B, s.readMem "x" (s.pid * B + i.val) = xs i)
    (hgs : ∀ i : Fin B, s.readMem "gate" (s.pid * B + i.val) = gs i) :
    ∃ s1, exec (siluStepGate "x" "gate" "z" B).toAlgKernel s = some s1 ∧
      (∀ i : Fin B, s1.readMem "z" (s.pid * B + i.val) = xs i * gs i) ∧
      (∀ (r : RegionName) o, (r ≠ "z" ∨ ∀ i : Fin B, o ≠ s.pid * B + i.val) →
        s1.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [BlockState.pid_eq] at hxs hgs ⊢
  simp [siluStepGate, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, NumericDType.add, NumericDType.mul]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_nd _ _ _ hinj (i, PUnit.unit)]
    simp [hxs, hgs]
  · rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl
    · by_cases hr : r = "z"
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun k _ => Ne.symm (ho k.1)) _).trans rfl
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl

theorem silu_region_run (B : Nat) (s : BlockState) (zs : Fin B → ℝ)
    (hzs : ∀ i : Fin B, s.readMem "z" (s.pid * B + i.val) = zs i) :
    ∃ s1, exec (siluStepSilu "z" "silu" B).toAlgKernel s = some s1 ∧
      (∀ i : Fin B, s1.readMem "silu" (s.pid * B + i.val) = zs i * Real.sigmoid (zs i)) ∧
      (∀ (r : RegionName) o, (r ≠ "silu" ∨ ∀ i : Fin B, o ≠ s.pid * B + i.val) →
        s1.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [BlockState.pid_eq] at hzs ⊢
  simp [siluStepSilu, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_nd _ _ _ hinj (i, PUnit.unit)]
    simp [hzs]
  · rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl
    · by_cases hr : r = "silu"
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun k _ => Ne.symm (ho k.1)) _).trans rfl
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl

theorem residual_region_run (B : Nat) (s : BlockState) (ss rs : Fin B → ℝ)
    (hss : ∀ i : Fin B, s.readMem "silu" (s.pid * B + i.val) = ss i)
    (hrs : ∀ i : Fin B, s.readMem "residual" (s.pid * B + i.val) = rs i) :
    ∃ s1, exec (residualMathKernel "silu" "residual" "out" B).toAlgKernel s = some s1 ∧
      (∀ i : Fin B, s1.readMem "out" (s.pid * B + i.val) = rs i + ss i) ∧
      (∀ (r : RegionName) o, (r ≠ "out" ∨ ∀ i : Fin B, o ≠ s.pid * B + i.val) →
        s1.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [BlockState.pid_eq] at hss hrs ⊢
  simp [residualMathKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, NumericDType.add, NumericDType.mul]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_nd _ _ _ hinj (i, PUnit.unit)]
    simp [hss, hrs]
  · rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl
    · by_cases hr : r = "out"
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun k _ => Ne.symm (ho k.1)) _).trans rfl
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl

theorem fused_region_run (B : Nat) (s : BlockState) (xs gs rs : Fin B → ℝ)
    (hxs : ∀ i : Fin B, s.readMem "x" (s.pid * B + i.val) = xs i)
    (hgs : ∀ i : Fin B, s.readMem "gate" (s.pid * B + i.val) = gs i)
    (hrs : ∀ i : Fin B, s.readMem "residual" (s.pid * B + i.val) = rs i) :
    ∃ s1, exec (fusedMathKernel "x" "gate" "residual" "out" B).toAlgKernel s = some s1 ∧
      (∀ i : Fin B, s1.readMem "out" (s.pid * B + i.val) = rs i + (xs i * gs i) * Real.sigmoid (xs i * gs i)) ∧
      (∀ (r : RegionName) o, (r ≠ "out" ∨ ∀ i : Fin B, o ≠ s.pid * B + i.val) →
        s1.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [BlockState.pid_eq] at hxs hgs hrs ⊢
  simp [fusedMathKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_nd _ _ _ hinj (i, PUnit.unit)]
    simp [hxs, hgs, hrs]
  · rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl
    · by_cases hr : r = "out"
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun k _ => Ne.symm (ho k.1)) _).trans rfl
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl

/-- The original concatenated pipeline executes its three stages in order. -/
theorem unfused_exec_split (B : Nat) (s : BlockState) :
    exec (unfusedMathKernel "x" "gate" "residual" "z" "silu" "out" B).toAlgKernel s =
      (exec (siluStepGate "x" "gate" "z" B).toAlgKernel s).bind (fun s1 =>
        (exec (siluStepSilu "z" "silu" B).toAlgKernel s1).bind (fun s2 =>
          exec (residualMathKernel "silu" "residual" "out" B).toAlgKernel s2)) := by
  have h := execR_toAlgKernel_seq .triv ["x", "gate", "residual"] ["out"]
    [siluStepGate "x" "gate" "z" B, siluStepSilu "z" "silu" B,
      residualMathKernel "silu" "residual" "out" B] s (by
        intro k hk
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
        rcases hk with rfl | rfl | rfl <;> rfl)
  simpa only [execR_triv, List.foldl_cons, List.foldl_nil, Option.bind_some,
    Option.bind_assoc] using h

theorem unfused_region_run (B : Nat) (s : BlockState) (xs gs rs : Fin B → ℝ)
    (hx : ∀ i : Fin B, s.readMem "x" (s.pid * B + i.val) = xs i)
    (hg : ∀ i : Fin B, s.readMem "gate" (s.pid * B + i.val) = gs i)
    (hr : ∀ i : Fin B, s.readMem "residual" (s.pid * B + i.val) = rs i) :
    ∃ s3, exec (unfusedMathKernel "x" "gate" "residual" "z" "silu" "out" B).toAlgKernel s = some s3 ∧
      (∀ i : Fin B, s3.readMem "out" (s.pid * B + i.val) =
        rs i + (xs i * gs i) * Real.sigmoid (xs i * gs i)) ∧
      (∀ (r : RegionName) o,
        (r ≠ "out" ∨ ∀ i : Fin B, o ≠ s.pid * B + i.val) →
        (r ≠ "z" ∨ ∀ i : Fin B, o ≠ s.pid * B + i.val) →
        (r ≠ "silu" ∨ ∀ i : Fin B, o ≠ s.pid * B + i.val) →
        s3.mem r o = s.mem r o) := by
  obtain ⟨s1, he1, hz, hf1⟩ := gate_region_run B s xs gs hx hg
  have hp1 := exec_pid he1
  have hz' : ∀ i : Fin B, s1.readMem "z" (s1.pid * B + i.val) = xs i * gs i := by
    simpa only [hp1] using hz
  obtain ⟨s2, he2, hs, hf2⟩ := silu_region_run B s1 (fun i => xs i * gs i) hz'
  have hp2 : s2.pid = s.pid := (exec_pid he2).trans hp1
  have hs' : ∀ i : Fin B, s2.readMem "silu" (s2.pid * B + i.val) =
      (xs i * gs i) * Real.sigmoid (xs i * gs i) := by
    simpa only [hp1, hp2] using hs
  have hr' : ∀ i : Fin B, s2.readMem "residual" (s2.pid * B + i.val) = rs i := by
    intro i
    rw [hp2]
    exact (BlockState.readMem_congr (hf2 "residual" _ (Or.inl (by decide)))).trans
      ((BlockState.readMem_congr (hf1 "residual" _ (Or.inl (by decide)))).trans (hr i))
  obtain ⟨s3, he3, hv, hf3⟩ := residual_region_run B s2
    (fun i => (xs i * gs i) * Real.sigmoid (xs i * gs i)) rs hs' hr'
  refine ⟨s3, ?_, ?_, ?_⟩
  · rw [unfused_exec_split, he1, Option.bind_some, he2, Option.bind_some, he3]
  · simpa only [hp2] using hv
  · intro r o hout hz hs
    exact (hf3 r o (by simpa only [hp2] using hout)).trans
      ((hf2 r o (by simpa only [hp1] using hs)).trans (hf1 r o hz))

@[simp] private theorem foldl_writeMem_pids {α : Type}
    (reg : RegionName) (offset : α → Nat) (value : α → ℝ)
    (indices : List α) (s : BlockState) :
    (indices.foldl (fun t i => t.writeMem reg (offset i) (value i)) s).pids = s.pids := by
  induction indices generalizing s with
  | nil => rfl
  | cons i rest ih => simp only [List.foldl_cons, ih, BlockState.writeMem_pids]

theorem fused_flattenOk (xReg gateReg residualReg outReg : RegionName) (B : Nat) :
    (fusedMathKernel xReg gateReg residualReg outReg B).toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk, fusedMathKernel, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
theorem fused_traceSafe (xReg gateReg residualReg outReg : RegionName) (B : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h0 : s.pid * B + B ≤ bounds xReg)
    (h1 : s.pid * B + B ≤ bounds gateReg)
    (h2 : s.pid * B + B ≤ bounds residualReg)
    (h3 : s.pid * B + B ≤ bounds outReg) :
    Kernel.TraceSafe bounds (fusedMathKernel xReg gateReg residualReg outReg B).toAlgKernel s := by
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at h0 h1 h2 h3
  simp [fusedMathKernel, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, BlockState.setReg, Tile.bop, Tile.uop,
    NumericDType.add, NumericDType.mul]
  refine ⟨fun a => ?_, fun a => ?_, fun a => ?_, fun a => ?_⟩ <;>
    exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) (by assumption)

theorem unfused_flattenOk (xReg gateReg residualReg zReg siluReg outReg : RegionName) (B : Nat) :
    (unfusedMathKernel xReg gateReg residualReg zReg siluReg outReg B).toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk, unfusedMathKernel, siluStepGate, siluStepSilu, residualMathKernel, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
theorem unfused_traceSafe (xReg gateReg residualReg zReg siluReg outReg : RegionName) (B : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h0 : s.pid * B + B ≤ bounds xReg)
    (h1 : s.pid * B + B ≤ bounds gateReg)
    (h2 : s.pid * B + B ≤ bounds residualReg)
    (h3 : s.pid * B + B ≤ bounds zReg)
    (h4 : s.pid * B + B ≤ bounds siluReg)
    (h5 : s.pid * B + B ≤ bounds outReg) :
    Kernel.TraceSafe bounds (unfusedMathKernel xReg gateReg residualReg zReg siluReg outReg B).toAlgKernel s := by
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at h0 h1 h2 h3 h4 h5
  simp [unfusedMathKernel, siluStepGate, siluStepSilu, residualMathKernel, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, BlockState.setReg, Tile.bop, Tile.uop,
    NumericDType.add, NumericDType.mul]
  refine ⟨fun a => ?_, fun a => ?_, fun a => ?_, fun a => ?_, fun a => ?_, fun a => ?_⟩ <;>
    exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) (by assumption)

def fusedIO (B : Nat) : KernelIO₃ where
  kernel := (fusedSiLUKernel "x" "gate" "residual" "out" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, fused_projection]
  in1 := "x"
  in2 := "gate"
  in3 := "residual"
  out := "out"
  B1 := B
  B2 := B
  B3 := B
  Bout := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  read3 := fun pid => pid * B
  write := fun pid => pid * B

specification fused_silu_correct (B : Nat) :
    Spec.Real (fusedIO B ⊨ fun xs gs rs i =>
      rs i + (xs i * gs i) * Real.sigmoid (xs i * gs i)) := by
  have halg : (fusedIO B).kernel.toAlgKernel =
      (fusedMathKernel "x" "gate" "residual" "out" B).toAlgKernel := by
    simp [fusedIO, ComputeKernel.toAlgKernel, fused_projection]
  refine KernelIO₃.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact fused_flattenOk "x" "gate" "residual" "out" B
  · intro bounds s hx hg hr ho hsc
    rw [halg]
    exact fused_traceSafe "x" "gate" "residual" "out" B bounds s hx hg hr ho
  · intro s xs gs rs hx hg hr
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := fused_region_run B s xs gs rs hx hg hr
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

def unfusedIO (B : Nat) : KernelIO₃ where
  kernel := (unfusedSiLUKernel "x" "gate" "residual" "z" "silu" "out" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, unfused_projection]
  in1 := "x"
  in2 := "gate"
  in3 := "residual"
  out := "out"
  B1 := B
  B2 := B
  B3 := B
  Bout := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  read3 := fun pid => pid * B
  write := fun pid => pid * B
  scratch := [⟨"z", fun pid => pid * B, B⟩, ⟨"silu", fun pid => pid * B, B⟩]

specification unfused_silu_correct (B : Nat) :
    Spec.Real (unfusedIO B ⊨ fun xs gs rs i =>
      rs i + (xs i * gs i) * Real.sigmoid (xs i * gs i)) := by
  have halg : (unfusedIO B).kernel.toAlgKernel =
      (unfusedMathKernel "x" "gate" "residual" "z" "silu" "out" B).toAlgKernel := by
    simp [unfusedIO, ComputeKernel.toAlgKernel, unfused_projection]
  refine KernelIO₃.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact unfused_flattenOk "x" "gate" "residual" "z" "silu" "out" B
  · intro bounds s hx hg hr ho hsc
    rw [halg]
    exact unfused_traceSafe "x" "gate" "residual" "z" "silu" "out" B bounds s hx hg hr (hsc ⟨"z", fun pid => pid * B, B⟩ (by simp [unfusedIO]))
      (hsc ⟨"silu", fun pid => pid * B, B⟩ (by simp [unfusedIO])) ho
  · intro s xs gs rs hx hg hr
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := unfused_region_run B s xs gs rs hx hg hr
    refine ⟨s1, he, hv, ?_⟩
    intro r o hmiss hsc
    apply hf r o hmiss
    · by_cases hr : r = "z"
      · exact Or.inr (hsc ⟨"z", fun pid => pid * B, B⟩ (by simp [unfusedIO]) hr)
      · exact Or.inl hr
    · by_cases hr : r = "silu"
      · exact Or.inr (hsc ⟨"silu", fun pid => pid * B, B⟩ (by simp [unfusedIO]) hr)
      · exact Or.inl hr

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FusedSiLUCorrect
