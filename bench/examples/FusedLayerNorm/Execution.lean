import bench.examples.FusedLayerNorm.Kernels
/- Opaque FP execution of the original two-pass and Welford-based LayerNorm
sources. No numerical identity is used to evaluate the affine suffix. The
shared sources are imported from Kernels.lean, without the real correctness proofs. -/
import bench.examples.Welford.Execution

namespace VeriTile.Bench.Examples.LayerNormFPExecution
open VeriTile.Bench.Examples.FusedLayerNorm.Kernels
open VeriTile Triton FP.Structural
open WelfordFPExecution (recurrence twopassMean twopassVariance)

set_option maxHeartbeats 1600000


/-- The common affine suffix, with every operation and its order retained. -/
def outputValue {α : Type} (M : Algebra α) (ε : ℝ) (x gamma beta mean variance : α) : α :=
  let inverse := M.binary none .real .div (M.literal none .real 1)
    (M.unary none .sqrt (M.binary none .real .add variance (M.literal none .real ε)))
  M.cast none .real .bf16
    (M.binary none .real .add
      (M.binary none .real .mul
        (M.binary none .real .mul (M.binary none .real .sub x mean) inverse) gamma) beta)

def onlineVariance {α : Type} (M : Algebra α) (xs : Fin N → α) : α :=
  M.binary none .real .div (recurrence M xs N).2 (M.fromNat none N)

/-- This is the unchanged source after the Welford loop, not a replacement
kernel. The preceding four statements initialize and run the recurrence. -/
def suffix (x g b y : RegionName) (N stride : Nat) (ε : ℝ) : List ComputeStmt :=
  (fusedLayerNormKernel x g b y N stride ε).surfaceBody.drop 4

theorem fused_body (x g b y : RegionName) (N stride : Nat) (ε : ℝ) :
    (fusedLayerNormKernel x g b y N stride ε).surfaceBody =
      WelfordFPExecution.initialCode ++ [.forLoop "i" N (WelfordFPExecution.body x stride)] ++
        suffix x g b y N stride ε := rfl

/-- Both reductions stay opaque. The final store preserves bf16 and frames
all cells outside the original row; gamma and beta use feature offsets. -/
theorem twopass_run {α : Type} [Inhabited α] (M : Algebra α)
    (x g b y : RegionName) (stride : Nat) (ε : ℝ) (xs gs bs : Fin N → α) (s : State α)
    (hx : ∀ i : Fin N, (s.mem x (s.pids 0 * stride + i.val)).read .real = xs i)
    (hg : ∀ i : Fin N, (s.mem g i.val).read .real = gs i)
    (hb : ∀ i : Fin N, (s.mem b i.val).read .real = bs i) :
    ∃ t, FP.Structural.exec M (twoPassLayerNormKernel x g b y N stride ε) s = some t ∧
      (∀ i : Fin N, t.mem y (s.pids 0 * stride + i.val) =
        .mk .bf16 (outputValue M ε (xs i) (gs i) (bs i) (twopassMean M xs) (twopassVariance M xs))) ∧
      (∀ r o, (r ≠ y ∨ ∀ i : Fin N, o ≠ s.pids 0 * stride + i.val) → t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [N] => s.pids 0 * stride + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [twoPassLayerNormKernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, TileShape.eraseAxis, Region.cast, ofFloat, toFloat,
    hx, hg, hb, WelfordFPExecution.sumValue, twopassMean, twopassVariance, outputValue]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    rfl
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

/-- The loop leaves input memory and pid intact; the original suffix reads
x again and uses the unrounded statistics in its unchanged affine expression. -/
theorem fused_run {α : Type} [Inhabited α] (M : Algebra α)
    (x g b y : RegionName) (stride : Nat) (ε : ℝ) (xs gs bs : Fin N → α) (s : State α)
    (hx : ∀ i : Fin N, (s.mem x (s.pids 0 * stride + i.val)).read .real = xs i)
    (hg : ∀ i : Fin N, (s.mem g i.val).read .real = gs i)
    (hb : ∀ i : Fin N, (s.mem b i.val).read .real = bs i) :
    ∃ t, FP.Structural.exec M (fusedLayerNormKernel x g b y N stride ε) s = some t ∧
      (∀ i : Fin N, t.mem y (s.pids 0 * stride + i.val) =
        .mk .bf16 (outputValue M ε (xs i) (gs i) (bs i) (recurrence M xs N).1 (onlineVariance M xs))) ∧
      (∀ r o, (r ≠ y ∨ ∀ i : Fin N, o ≠ s.pids 0 * stride + i.val) → t.mem r o = s.mem r o) := by
  obtain ⟨u, hu, hm, hv, hpid, hmem, _⟩ :=
    WelfordFPExecution.loop_run_with_pid M x stride xs s hx
  have he : FP.Structural.exec M (fusedLayerNormKernel x g b y N stride ε) s =
      run M (WelfordFPExecution.initialCode ++ [.forLoop "i" N (WelfordFPExecution.body x stride)] ++
        suffix x g b y N stride ε) s :=
    congrArg (fun code => run M code s) (fused_body x g b y N stride ε)
  have hinj : Function.Injective (fun i : TileIndex [N] => s.pids 0 * stride + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  rw [he, run_append, hu]
  simp [suffix, fusedLayerNormKernel, ComputeKernel.surfaceBody, List.drop, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, Region.cast, ofFloat, toFloat,
    hm, hv, hpid, hmem, hx, hg, hb, outputValue, onlineVariance]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
  · trans u.mem r o
    · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
      rcases hmiss with hr | ho
      · exact Or.inl hr
      · exact Or.inr fun k _ => ho k.1
    · exact congrFun (congrFun hmem r) o

/-- Declare the three public input ports once each. The DSL's metadata lists
each antiquoted load occurrence, including the fused kernel's second x read;
the explicit IO declaration preserves the entire original statement body. -/
def twopassIO (x g b y : RegionName) (N stride : Nat) (ε : ℝ) : KernelIO₃ where
  kernel := .mk [x, g, b] [y] (twoPassLayerNormKernel x g b y N stride ε).surfaceBody
  projection := by rfl
  in1 := x
  in2 := g
  in3 := b
  out := y
  B1 := N
  B2 := N
  B3 := N
  Bout := N
  read1 := fun pid => pid * stride
  read2 := fun _ => 0
  read3 := fun _ => 0
  write := fun pid => pid * stride

def fusedIO (x g b y : RegionName) (N stride : Nat) (ε : ℝ) : KernelIO₃ :=
  { twopassIO x g b y N stride ε with
    kernel := .mk [x, g, b] [y] (fusedLayerNormKernel x g b y N stride ε).surfaceBody
    projection := by rfl }

theorem same_signature (x g b y : RegionName) (N stride : Nat) (ε : ℝ) :
    ioSignature (twopassIO x g b y N stride ε) =
      ioSignature (fusedIO x g b y N stride ε) := rfl

end VeriTile.Bench.Examples.LayerNormFPExecution
