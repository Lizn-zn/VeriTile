import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.Program
import VeriTile.Triton.Semantics.BroadcastReshape

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Tiles
open VeriTile.Triton Math Program
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

@[local simp] theorem some_lt (a b : ℝ) : @LT.lt (WithBot ℝ) WithBot.instLT (some a) (some b) ↔ a < b := WithBot.coe_lt_coe
@[local simp] theorem some_le (a b : ℝ) : @LE.le (WithBot ℝ) WithBot.instLE (some a) (some b) ↔ a ≤ b := WithBot.coe_le_coe

def matrix (p : Layout) (f : Fin p.BH → Fin p.BC → ℝ) : Tile .real [p.BH,p.BC] :=
  Tile.mat (fun i j => some (f i j))
def row (p : Layout) (f : Fin p.BH → ℝ) : Tile .real [p.BH] :=
  Tile.vec (fun i => some (f i))

theorem addBias_eval (p : Layout) (u : BlockState) (f : Fin p.BH → Fin p.BC → ℝ) (b : Fin p.BH → ℝ)
    (hf : u.regs .real [p.BH,p.BC] "dt" = some (matrix p f))
    (hb : u.regs .real [p.BH] "dt_bias" = some (row p b)) :
    evalOp (addBias p) u = some (matrix p (fun i j => f i j + b i)) := by
  simp only [addBias,dtOp,evalOp.eq_def,hf,hb,
    Option.bind_some,Option.bind_eq_bind]
  rfl

theorem softplus_eval (p : Layout) (u : BlockState) (f : Fin p.BH → Fin p.BC → ℝ)
    (hf : u.regs .real [p.BH,p.BC] "dt" = some (matrix p f)) :
    evalOp (softplusOp p) u = some (matrix p (fun i j => softplus (f i j))) := by
  simp only [softplusOp,dtOp,evalOp.eq_def,hf,Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [matrix,softplus,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
    WithBot.realLog,WithBot.realExp,apply_ite]
  split_ifs with h <;> simp_all

theorem lowerClamp_eval (p : Layout) (u : BlockState) (f : Fin p.BH → Fin p.BC → ℝ) (lower : ℝ)
    (hf : u.regs .real [p.BH,p.BC] "dt" = some (matrix p f)) :
    evalOp (lowerClamp p lower) u = some (matrix p (fun i j => max (f i j) lower)) := by
  simp only [lowerClamp,dtOp,evalOp.eq_def,hf,Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [matrix,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex]
  split_ifs with h
  · rw [max_eq_left (le_of_lt h)]
  · rw [max_eq_right (le_of_not_gt h)]

theorem clamp_eval (p : Layout) (u : BlockState) (f : Fin p.BH → Fin p.BC → ℝ) (lower : ℝ) (upper : UpperBound)
    (hf : u.regs .real [p.BH,p.BC] "dt" = some (matrix p f)) :
    evalOp (clamp p lower upper) u = some (matrix p (fun i j => upper.cap (max (f i j) lower))) := by
  cases upper with
  | finite upper =>
      simp only [UpperBound.cap,clamp,evalOp_where,evalOp_lt,lowerClamp_eval p u f lower hf,evalOp.eq_def,Option.bind_some,Option.bind_eq_bind]
      apply congrArg some
      apply Tile.ext
      rintro ⟨i,j,⟨⟩⟩
      simp [matrix,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex]
      split_ifs with h
      · rw [min_eq_left (le_of_lt (max_lt h.1 h.2))]
      · rw [min_eq_right (le_of_not_gt (fun hh => h (max_lt_iff.mp hh)))]
  | posInf =>
      have hmin (e : Op .real [p.BH,p.BC]) : evalOp (.minPosInf e) u = evalOp e u := by
        simp only [evalOp]
        cases evalOp e u <;> rfl
      rw [clamp,hmin,lowerClamp_eval p u f lower hf]
      rfl

theorem product_eval (p : Layout) (u : BlockState) (swapped : Bool)
    (f : Fin p.BH → Fin p.BC → ℝ) (a : Fin p.BH → ℝ)
    (hf : u.regs .real [p.BH,p.BC] "dt" = some (matrix p f))
    (ha : u.regs .real [p.BH] "A" = some (row p a)) :
    evalOp (product p swapped) u = some (matrix p (fun i j => f i j * a i)) := by
  cases swapped
  all_goals simp only [product,Bool.false_eq_true,↓reduceIte,dtOp,evalOp.eq_def,hf,ha,Option.bind_some,Option.bind_eq_bind]
  · rfl
  · apply congrArg some
    apply Tile.ext
    rintro ⟨i,j,⟨⟩⟩
    simp [matrix,row,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,mul_comm]

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Tiles
