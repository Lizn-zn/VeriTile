/- FP structural equivalence of the original fused and materialized SiLU
implementations. Every numerical primitive is interpreted by an arbitrary
function; intermediate loads/stores, not real algebra, connect the programs. -/
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.FusedSiLUFPEquiv
open VeriTile Triton
open FP.Structural

set_option maxHeartbeats 1200000

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
The staged execution is recovered by `unfused_run` below. -/
def unfusedSiLUKernel
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel :=
  ComputeKernel.seq [xReg, gateReg, residualReg] [outReg]
    [siluStepGate xReg gateReg zReg blockSize,
     siluStepSilu zReg siluReg blockSize,
     siluStepResidual siluReg residualReg outReg blockSize]


def siluValue {α : Type} (M : Algebra α) (z : α) : α :=
  M.binary none .real .mul z (M.unary none .sigmoid z)

def resultValue {α : Type} (M : Algebra α) (x g r : α) : α :=
  M.cast none .real .bf16 (M.binary none .real .add r
    (siluValue M (M.binary none .real .mul x g)))

theorem gate_run {α : Type} [Inhabited α] (M : Algebra α) (B : Nat) (s : State α)
    (xs gs : Fin B → α)
    (hxs : ∀ i : Fin B, (s.mem "x" (s.pids 0 * B + i.val)).read .real = xs i)
    (hgs : ∀ i : Fin B, (s.mem "gate" (s.pids 0 * B + i.val)).read .real = gs i)
    : ∃ t, FP.Structural.exec M (siluStepGate "x" "gate" "z" B) s = some t ∧
      (∀ i : Fin B, t.mem "z" (s.pids 0 * B + i.val) = Cell.mk .real (M.binary none .real .mul (xs i) (gs i))) ∧
      t.pids = s.pids ∧
      (∀ (r : RegionName) o, (r ≠ "z" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [siluStepGate, FP.Structural.exec, run, step, evalExpr,
    evalOp_unfold, numeric, bop, store]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    simp [hxs, hgs]
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

theorem silu_run {α : Type} [Inhabited α] (M : Algebra α) (B : Nat) (s : State α)
    (zs : Fin B → α)
    (hzs : ∀ i : Fin B, (s.mem "z" (s.pids 0 * B + i.val)).read .real = zs i)
    : ∃ t, FP.Structural.exec M (siluStepSilu "z" "silu" B) s = some t ∧
      (∀ i : Fin B, t.mem "silu" (s.pids 0 * B + i.val) = Cell.mk .real (siluValue M (zs i))) ∧
      t.pids = s.pids ∧
      (∀ (r : RegionName) o, (r ≠ "silu" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [siluStepSilu, FP.Structural.exec, run, step, evalExpr,
    evalOp_unfold, numeric, bop, store,
    Function.comp_def]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    simp [hzs, siluValue]
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

theorem residual_run {α : Type} [Inhabited α] (M : Algebra α) (B : Nat) (s : State α)
    (ys rs : Fin B → α)
    (hys : ∀ i : Fin B, (s.mem "silu" (s.pids 0 * B + i.val)).read .real = ys i)
    (hrs : ∀ i : Fin B, (s.mem "residual" (s.pids 0 * B + i.val)).read .real = rs i)
    : ∃ t, FP.Structural.exec M (siluStepResidual "silu" "residual" "out" B) s = some t ∧
      (∀ i : Fin B, t.mem "out" (s.pids 0 * B + i.val) = Cell.mk .bf16 (M.cast none .real .bf16 (M.binary none .real .add (rs i) (ys i)))) ∧
      t.pids = s.pids ∧
      (∀ (r : RegionName) o, (r ≠ "out" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [siluStepResidual, FP.Structural.exec, run, step, evalExpr,
    evalOp_unfold, numeric, bop, store, ofFloat, toFloat]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    simp [hys, hrs]
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

theorem fused_run {α : Type} [Inhabited α] (M : Algebra α) (B : Nat) (s : State α)
    (xs gs rs : Fin B → α)
    (hxs : ∀ i : Fin B, (s.mem "x" (s.pids 0 * B + i.val)).read .real = xs i)
    (hgs : ∀ i : Fin B, (s.mem "gate" (s.pids 0 * B + i.val)).read .real = gs i)
    (hrs : ∀ i : Fin B, (s.mem "residual" (s.pids 0 * B + i.val)).read .real = rs i)
    : ∃ t, FP.Structural.exec M (fusedSiLUKernel "x" "gate" "residual" "out" B) s = some t ∧
      (∀ i : Fin B, t.mem "out" (s.pids 0 * B + i.val) = Cell.mk .bf16 (resultValue M (xs i) (gs i) (rs i))) ∧
      t.pids = s.pids ∧
      (∀ (r : RegionName) o, (r ≠ "out" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [fusedSiLUKernel, FP.Structural.exec, run, step, evalExpr,
    evalOp_unfold, numeric, bop, store, ofFloat, toFloat,
    Function.comp_def]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    simp [hxs, hgs, hrs, siluValue, resultValue]
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

theorem unfused_run {α : Type} [Inhabited α] (M : Algebra α) (B : Nat) (s : State α)
    (xs gs rs : Fin B → α)
    (hxs : ∀ i : Fin B, (s.mem "x" (s.pids 0 * B + i.val)).read .real = xs i)
    (hgs : ∀ i : Fin B, (s.mem "gate" (s.pids 0 * B + i.val)).read .real = gs i)
    (hrs : ∀ i : Fin B, (s.mem "residual" (s.pids 0 * B + i.val)).read .real = rs i) :
    ∃ t, FP.Structural.exec M (unfusedSiLUKernel "x" "gate" "residual" "z" "silu" "out" B) s = some t ∧
      (∀ i : Fin B, t.mem "out" (s.pids 0 * B + i.val) = Cell.mk .bf16 (resultValue M (xs i) (gs i) (rs i))) ∧
      (∀ (r : RegionName) o,
        (r ≠ "out" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        (r ≠ "z" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        (r ≠ "silu" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  obtain ⟨s1, he1, hz, hp1, hf1⟩ := gate_run M B s xs gs hxs hgs
  let zs : Fin B → α := fun i => M.binary none .real .mul (xs i) (gs i)
  obtain ⟨s2, he2, hy, hp2, hf2⟩ := silu_run M B s1 zs (by
    intro i
    rw [hp1, hz i]
    rfl)
  rw [hp1] at hy hp2 hf2
  obtain ⟨s3, he3, hv, hp3, hf3⟩ := residual_run M B s2 (fun i => siluValue M (zs i)) rs
    (by intro i; rw [hp2, hy i]; rfl)
    (by
      intro i
      rw [hp2, hf2 "residual" _ (Or.inl (by decide)), hf1 "residual" _ (Or.inl (by decide))]
      exact hrs i)
  rw [hp2] at hv hp3 hf3
  refine ⟨s3, ?_, ?_, ?_⟩
  · simp only [unfusedSiLUKernel, exec_seq_cons, he1, Option.bind_some, he2, he3, exec_seq_nil]
  · simpa only [resultValue, zs] using hv
  · intro r o ho hz hs
    exact (hf3 r o ho).trans ((hf2 r o hs).trans (hf1 r o hz))

def fusedIO (B : Nat) : KernelIO₃ where
  kernel := fusedSiLUKernel "x" "gate" "residual" "out" B
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

def unfusedIO (B : Nat) : KernelIO₃ :=
  { fusedIO B with
    kernel := unfusedSiLUKernel "x" "gate" "residual" "z" "silu" "out" B
    projection := by rfl
    scratch := [{ buf := "z", win := fun pid => pid * B, len := B }, { buf := "silu", win := fun pid => pid * B, len := B }] }

/-- Fusion preserves the same opaque numerical call tree, so no numerical
identity is assumed in this example. -/
def R : Spec.Assumptions ComputeStmt := []

open scoped VeriTile.Spec

specification silu_equiv (B : Nat) : fusedIO B ≡[R] unfusedIO B := by
  apply Spec.FloatingPoint.ofStructural (lhs := fusedIO B) (rhs := unfusedIO B)
    (structural := IO₃Equiv) rfl rfl
  refine ⟨?_, ?_, ?_⟩
  · simp [PrivateScratch, fusedIO]
  · intro p hp
    simp only [unfusedIO, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl <;> simp only [unfusedIO, fusedIO] <;> decide
  · intro α _ M s
    let xs : Fin B → α := fun i => (s.mem "x" (s.pids 0 * B + i.val)).read .real
    let gs : Fin B → α := fun i => (s.mem "gate" (s.pids 0 * B + i.val)).read .real
    let rs : Fin B → α := fun i => (s.mem "residual" (s.pids 0 * B + i.val)).read .real
    obtain ⟨a, ha, hva, _, hfa⟩ := fused_run M B s xs gs rs (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
    obtain ⟨b, hb, hvb, hfb⟩ := unfused_run M B s xs gs rs (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
    refine ⟨a, b, ha, hb, fun i => (hva i).trans (hvb i).symm, ?_, ?_⟩
    · intro r o ho _; exact hfa r o ho
    · intro r o ho hs
      apply hfb r o ho
      · by_cases hr : r = "z"
        · exact Or.inr (hs { buf := "z", win := fun pid => pid * B, len := B } (by simp [unfusedIO]) hr)
        · exact Or.inl hr
      · by_cases hr : r = "silu"
        · exact Or.inr (hs { buf := "silu", win := fun pid => pid * B, len := B } (by simp [unfusedIO]) hr)
        · exact Or.inl hr

#print_fp_assumptions silu_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FusedSiLUFPEquiv
