/- Real correctness of the original two-pass and Welford-based LayerNorm
kernels. Both implement the population-variance normalization and affine
formula. Source bf16 stores are erased only for mathematical correctness. -/
import bench.examples.WelfordCorrect
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.FusedLayerNormCorrect

open VeriTile Triton
open VeriTile.Triton.TiledReduction.WelfordRec
open VeriTile.Examples (InputRowLoadedAt onlineWelfordLoopBody)
open WelfordCorrect (stepStmt_assign_inv welfordLoop_traceSafe welfordBody_pid_preserved)
open scoped VeriTile.Triton.KernelIO₃

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

/-- Two-pass LayerNorm kernel: `tl.sum` twice (mean and var), then affine. -/
def twoPassLayerNormKernel
    (xReg γReg βReg yReg : RegionName) (N rowStride : Nat) (ε : ℝ) :
    ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(rowStride) + tl.arange($(N))
  x      := tl.load($(xReg) + offs)
  s_x    := tl.sum(x)
  μ      := s_x / tl.toReal($(N))
  d      := x - μ
  s_d2   := tl.sum(d * d)
  v      := s_d2 / tl.toReal($(N))
  γ      := tl.load($(γReg) + tl.arange($(N)))
  β      := tl.load($(βReg) + tl.arange($(N)))
  σ_inv  := 1 / tl.sqrt(v + $(ε))
  y      := (x - μ) * σ_inv * γ + β
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

/-- Fused single-pass LayerNorm kernel: Welford `forLoop`, then affine. -/
def fusedLayerNormKernel
    (xReg γReg βReg yReg : RegionName) (N rowStride : Nat) (ε : ℝ) :
    ComputeKernel := triton {
  pid := tl.program_id(0)
  M   := 0
  S   := 0
  tl.for i in $(N) {
    xi      := tl.load($(xReg) + (pid * $(rowStride) + i))
    delta   := xi - M
    M       := M + delta / (tl.toReal(i) + 1)
    delta2  := xi - M
    S       := S + delta * delta2
  }
  μ       := M
  v       := S / tl.toReal($(N))
  σ_inv   := 1 / tl.sqrt(v + $(ε))
  -- Second pass to compute Y. The "fused" gain is that μ/var were
  -- computed in a single pass over `x`; the residual `(x − μ)` still
  -- needs the second read of x.
  offs    := pid * $(rowStride) + tl.arange($(N))
  x       := tl.load($(xReg) + offs)
  γ       := tl.load($(γReg) + tl.arange($(N)))
  β       := tl.load($(βReg) + tl.arange($(N)))
  y       := (x - μ) * σ_inv * γ + β
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

/-- Executable real counterpart of the two-pass implementation. -/

def twoPassMathKernel
    (xReg γReg βReg yReg : RegionName) (N rowStride : Nat) (ε : ℝ) :
    ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(rowStride) + tl.arange($(N))
  x      := tl.load($(xReg) + offs)
  s_x    := tl.sum(x)
  μ      := s_x / tl.toReal($(N))
  d      := x - μ
  s_d2   := tl.sum(d * d)
  v      := s_d2 / tl.toReal($(N))
  γ      := tl.load($(γReg) + tl.arange($(N)))
  β      := tl.load($(βReg) + tl.arange($(N)))
  σ_inv  := 1 / tl.sqrt(v + $(ε))
  y      := (x - μ) * σ_inv * γ + β
  tl.store($(yReg) + offs, y)
}


def fusedMathKernel
    (xReg γReg βReg yReg : RegionName) (N rowStride : Nat) (ε : ℝ) :
    ComputeKernel := triton {
  pid := tl.program_id(0)
  M   := 0
  S   := 0
  tl.for i in $(N) {
    xi      := tl.load($(xReg) + (pid * $(rowStride) + i))
    delta   := xi - M
    M       := M + delta / (tl.toReal(i) + 1)
    delta2  := xi - M
    S       := S + delta * delta2
  }
  μ       := M
  v       := S / tl.toReal($(N))
  σ_inv   := 1 / tl.sqrt(v + $(ε))
  -- Second pass to compute Y. The "fused" gain is that μ/var were
  -- computed in a single pass over `x`; the residual `(x − μ)` still
  -- needs the second read of x.
  offs    := pid * $(rowStride) + tl.arange($(N))
  x       := tl.load($(xReg) + offs)
  γ       := tl.load($(γReg) + tl.arange($(N)))
  β       := tl.load($(βReg) + tl.arange($(N)))
  y       := (x - μ) * σ_inv * γ + β
  tl.store($(yReg) + offs, y)
}


theorem twoPass_projection (xReg γReg βReg yReg : RegionName) (N stride : Nat) (ε : ℝ) :
    (twoPassLayerNormKernel xReg γReg βReg yReg N stride ε).eraseDType.toAlgorithm? =
      Except.ok (twoPassMathKernel xReg γReg βReg yReg N stride ε).toAlgKernel := by
  simp [twoPassLayerNormKernel, twoPassMathKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeExpr.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  repeat' apply And.intro
  all_goals
    rw [Op.eraseDType.eq_def]
    simp [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType]

theorem fused_projection (xReg γReg βReg yReg : RegionName) (N stride : Nat) (ε : ℝ) :
    (fusedLayerNormKernel xReg γReg βReg yReg N stride ε).eraseDType.toAlgorithm? =
      Except.ok (fusedMathKernel xReg γReg βReg yReg N stride ε).toAlgKernel := by
  simp [fusedLayerNormKernel, fusedMathKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeExpr.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  rw [Op.eraseDType.eq_def]
  simp [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType]

theorem twoPass_region_run (N stride : Nat) (ε : ℝ) (s : BlockState)
    (xs gs bs : Fin N → ℝ)
    (hx : InputRowLoadedAt s "x" stride N xs)
    (hg : ∀ i : Fin N, s.readMem "gamma" i.val = gs i)
    (hb : ∀ i : Fin N, s.readMem "beta" i.val = bs i) :
    ∃ t, exec (twoPassMathKernel "x" "gamma" "beta" "y" N stride ε).toAlgKernel s = some t ∧
      (∀ i : Fin N, t.readMem "y" (s.pid * stride + i.val) =
        (xs i - twoPassMean xs) / Real.sqrt (twoPassS xs / N + ε) * gs i + bs i) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ ∀ i : Fin N, o ≠ s.pid * stride + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [N] => s.pids 0 * stride + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  unfold InputRowLoadedAt at hx
  simp only [BlockState.pid_eq] at hx ⊢
  simp [twoPassMathKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, Tile.natToReal, NumericDType.add, NumericDType.mul, NumericDType.sub,
    NumericDType.div, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_nd _ _ _ hinj (i, PUnit.unit)]
    simp [hx, hg, hb, twoPassMean, twoPassS, pow_two, div_eq_mul_inv]
    exact Or.inl rfl
  · rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl
    · by_cases hr : r = "y"
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun k _ => Ne.symm (ho k.1)) _).trans rfl
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl

theorem fused_region_run (N stride : Nat) (ε : ℝ) (s : BlockState)
    (xs gs bs : Fin N → ℝ)
    (hx : InputRowLoadedAt s "x" stride N xs)
    (hg : ∀ i : Fin N, s.readMem "gamma" i.val = gs i)
    (hb : ∀ i : Fin N, s.readMem "beta" i.val = bs i) :
    ∃ t, exec (fusedMathKernel "x" "gamma" "beta" "y" N stride ε).toAlgKernel s = some t ∧
      (∀ i : Fin N, t.readMem "y" (s.pid * stride + i.val) =
        (xs i - twoPassMean xs) / Real.sqrt (twoPassS xs / N + ε) * gs i + bs i) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ ∀ i : Fin N, o ≠ s.pid * stride + i.val) →
        t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hM, hS, hp, _, hmem⟩ := WelfordCorrect.welford_loop_run N stride "x" s xs hx
  have hxt : ∀ i : Fin N, t.readMem "x" (s.pid * stride + i.val) = xs i := by
    intro i
    exact (BlockState.readMem_congr (hmem _ _)).trans (hx i)
  have hgt : ∀ i : Fin N, t.readMem "gamma" i.val = gs i := by
    intro i
    exact (BlockState.readMem_congr (hmem _ _)).trans (hg i)
  have hbt : ∀ i : Fin N, t.readMem "beta" i.val = bs i := by
    intro i
    exact (BlockState.readMem_congr (hmem _ _)).trans (hb i)
  have hinj : Function.Injective (fun i : TileIndex [N] => s.pids 0 * stride + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [BlockState.pid_eq] at hp hxt ⊢
  simp [fusedMathKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, Tile.natToReal]
  simp only [WelfordCorrect.loopInitial, onlineWelfordLoopBody] at he
  rw [he]
  simp [hM, hS, hp, NumericDType.add, NumericDType.mul,
    NumericDType.sub, NumericDType.div]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_nd _ _ _ hinj (i, PUnit.unit)]
    simp [hxt, hgt, hbt, div_eq_mul_inv]
  · rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans (hmem r o)
    · by_cases hr : r = "y"
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun k _ => Ne.symm (ho k.1)) _).trans (hmem _ o)
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans (hmem r o)

section Safety
variable (xReg γReg βReg yReg : RegionName) (N rowStride : Nat) (ε : ℝ) (s : BlockState)

private theorem offs_activeAddressSafe (bounds : RegionBounds) (pid₀ : Nat) (t : BlockState)
    (active : TileIndex [N] → Prop)
    (hread : t.regs .nat [N] "offs"
      = some ⟨fun i => pid₀ * rowStride + i.1.val⟩)
    (reg : RegionName) (hreg : pid₀ * rowStride + N ≤ bounds reg) :
    memAccessActiveAddressSafe  bounds
      (MemAccess.region reg (Op.ref .nat [N] "offs")) t active := by
  simp only [memAccessActiveAddressSafe]
  intro offsets hoffs i _
  rw [show evalOp  (Op.ref .nat [N] "offs") t
      = some ⟨fun i => pid₀ * rowStride + i.1.val⟩ from by
    simp [evalOp.eq_def, hread]] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact lt_of_lt_of_le (Nat.add_lt_add_left i.1.isLt _) hreg


private theorem arange_activeAddressSafe (bounds : RegionBounds) (t : BlockState)
    (active : TileIndex [N] → Prop)
    (reg : RegionName) (hreg : N ≤ bounds reg) :
    memAccessActiveAddressSafe  bounds
      (MemAccess.region reg (Op.arange N)) t active := by
  simp only [memAccessActiveAddressSafe]
  intro offsets hoffs i _
  rw [show evalOp  (Op.arange N) t = some ⟨fun i => i.1.val⟩ from by
    simp [evalOp.eq_def, Tile.vec]] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact lt_of_lt_of_le i.1.isLt hreg

set_option maxHeartbeats 1600000 in

theorem twoPass_traceSafe (bounds : RegionBounds)
    (hxb : s.pids 0 * rowStride + N ≤ bounds xReg)
    (hγb : N ≤ bounds γReg) (hβb : N ≤ bounds βReg)
    (hyb : s.pids 0 * rowStride + N ≤ bounds yReg) :
    Kernel.TraceSafe  bounds
      ((twoPassMathKernel xReg γReg βReg yReg N rowStride ε).toAlgKernel)
      s := by
  unfold Kernel.TraceSafe
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  rw [show evalOp  (Op.programId 0) s = some (Tile.scalar (s.pids 0)) from by
    simp [evalOp.eq_def]] at hv1
  obtain rfl := Option.some_inj.mp hv1
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  rw [show evalOp  (Op.add .nat .scalarL
      (Op.mul .nat .nil (Op.ref .nat [] "pid") (Op.constNat rowStride))
      (Op.arange N)) (s.setReg "pid" .nat [] (Tile.scalar (s.pids 0)))
      = some ⟨fun i => s.pids 0 * rowStride + i.1.val⟩ from by
    simp [evalOp.eq_def, Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
      Tile.vec, BlockState.setReg]] at hv2
  obtain rfl := Option.some_inj.mp hv2
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial,
      offs_activeAddressSafe N rowStride  bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) xReg hxb⟩
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmt_assign_inv hs3
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmt_assign_inv hs4
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmt_assign_inv hs5
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s6 hs6
  obtain ⟨v6, hv6, rfl⟩ := stepStmt_assign_inv hs6
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s7 hs7
  obtain ⟨v7, hv7, rfl⟩ := stepStmt_assign_inv hs7
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s8 hs8
  obtain ⟨v8, hv8, rfl⟩ := stepStmt_assign_inv hs8
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial, arange_activeAddressSafe N  bounds _ _ γReg hγb⟩
  intro s9 hs9
  obtain ⟨v9, hv9, rfl⟩ := stepStmt_assign_inv hs9
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial, arange_activeAddressSafe N  bounds _ _ βReg hβb⟩
  intro s10 hs10
  obtain ⟨v10, hv10, rfl⟩ := stepStmt_assign_inv hs10
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s11 hs11
  obtain ⟨v11, hv11, rfl⟩ := stepStmt_assign_inv hs11
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s12 hs12
  obtain ⟨v12, hv12, rfl⟩ := stepStmt_assign_inv hs12
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt, Op.SafeAt]
  refine ⟨trivial, by simp, trivial,
    offs_activeAddressSafe N rowStride  bounds (s.pids 0) _ _ ?_
      (Region.cast yReg) (by simpa [Region.cast_self] using hyb)⟩
  simp [BlockState.setReg]


theorem twoPass_flattenOk :
    ((twoPassMathKernel xReg γReg βReg yReg N rowStride ε
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [twoPassMathKernel, ComputeKernel.toAlgKernel, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk.eq_def]

private theorem welfordLoop_pid_preserved (xReg : RegionName) (rowStride : Nat)
    :
    ∀ (fuel k n : Nat) (t t' : BlockState), n - k ≤ fuel →
      stepForLoopAux  "i" k n (onlineWelfordLoopBody xReg rowStride) t
        = some t' →
      t'.regs .nat [] "pid" = t.regs .nat [] "pid"
  | fuel, k, n, t, t', hf, h => by
      rw [stepForLoopAux] at h
      split at h
      next hlt =>
        cases fuel with
        | zero => omega
        | succ fuel =>
            cases hb : stepStmts  (onlineWelfordLoopBody xReg rowStride)
                (t.setReg "i" .nat [] (Tile.scalar k)) with
            | none =>
                rw [hb] at h
                exact absurd h (by simp)
            | some s1 =>
                rw [hb] at h
                replace h : stepForLoopAux  "i" (k + 1) n
                    (onlineWelfordLoopBody xReg rowStride) s1 = some t' := h
                rw [welfordLoop_pid_preserved xReg rowStride  fuel (k + 1) n
                    s1 t' (by omega) h,
                  welfordBody_pid_preserved xReg rowStride hb]
                simp [BlockState.setReg]
      next =>
        obtain rfl := Option.some_inj.mp h
        rfl

set_option maxHeartbeats 1600000 in

theorem fused_traceSafe (bounds : RegionBounds)
    (hxb : s.pids 0 * rowStride + N ≤ bounds xReg)
    (hγb : N ≤ bounds γReg) (hβb : N ≤ bounds βReg)
    (hyb : s.pids 0 * rowStride + N ≤ bounds yReg) :
    Kernel.TraceSafe  bounds
      ((fusedMathKernel xReg γReg βReg yReg N rowStride ε).toAlgKernel)
      s := by
  unfold Kernel.TraceSafe
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  rw [show evalOp  (Op.programId 0) s = some (Tile.scalar (s.pids 0)) from by
    simp [evalOp.eq_def]] at hv1
  obtain rfl := Option.some_inj.mp hv1
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmt_assign_inv hs3
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe]
    exact welfordLoop_traceSafe xReg N rowStride  bounds (s.pids 0) hxb
      N 0 _ (by omega) (by simp [BlockState.setReg])
  intro s4 hs4
  have hpid4 : s4.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)) := by
    simp only [stepStmt] at hs4
    rw [welfordLoop_pid_preserved xReg rowStride  N 0 N _ s4 (by omega) hs4]
    simp [BlockState.setReg]
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmt_assign_inv hs5
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s6 hs6
  obtain ⟨v6, hv6, rfl⟩ := stepStmt_assign_inv hs6
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s7 hs7
  obtain ⟨v7, hv7, rfl⟩ := stepStmt_assign_inv hs7
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s8 hs8
  obtain ⟨v8, hv8, rfl⟩ := stepStmt_assign_inv hs8
  rw [show evalOp  (Op.add .nat .scalarL
      (Op.mul .nat .nil (Op.ref .nat [] "pid") (Op.constNat rowStride))
      (Op.arange N))
      (((s4.setReg "μ" .real [] v5).setReg "v" .real [] v6).setReg
        "σ_inv" .real [] v7)
      = some ⟨fun i => s.pids 0 * rowStride + i.1.val⟩ from by
    simp [evalOp.eq_def, Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
      Tile.vec, BlockState.setReg, hpid4]] at hv8
  obtain rfl := Option.some_inj.mp hv8
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial,
      offs_activeAddressSafe N rowStride  bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) xReg hxb⟩
  intro s9 hs9
  obtain ⟨v9, hv9, rfl⟩ := stepStmt_assign_inv hs9
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial, arange_activeAddressSafe N  bounds _ _ γReg hγb⟩
  intro s10 hs10
  obtain ⟨v10, hv10, rfl⟩ := stepStmt_assign_inv hs10
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial, arange_activeAddressSafe N  bounds _ _ βReg hβb⟩
  intro s11 hs11
  obtain ⟨v11, hv11, rfl⟩ := stepStmt_assign_inv hs11
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s12 hs12
  obtain ⟨v12, hv12, rfl⟩ := stepStmt_assign_inv hs12
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt, Op.SafeAt]
  refine ⟨trivial, by simp, trivial,
    offs_activeAddressSafe N rowStride  bounds (s.pids 0) _ _ ?_
      (Region.cast yReg) (by simpa [Region.cast_self] using hyb)⟩
  simp [BlockState.setReg]


theorem fused_flattenOk :
    ((fusedMathKernel xReg γReg βReg yReg N rowStride ε
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [fusedMathKernel, ComputeKernel.toAlgKernel, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk.eq_def]

end Safety

/-- The real interpretation retains the original row and feature-array layout. -/
def twoPassIO (N stride : Nat) (ε : ℝ) : KernelIO₃ where
  kernel := (twoPassLayerNormKernel "x" "gamma" "beta" "y" N stride ε).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, twoPass_projection]
  in1 := "x"
  in2 := "gamma"
  in3 := "beta"
  out := "y"
  B1 := N
  B2 := N
  B3 := N
  Bout := N
  read1 := fun pid => pid * stride
  read2 := fun _ => 0
  read3 := fun _ => 0
  write := fun pid => pid * stride

specification two_pass_layernorm_correct (N stride : Nat) (ε : ℝ) :
    Spec.Real (twoPassIO N stride ε ⊨ fun xs gs bs i =>
      let μ := (∑ j, xs j) / N
      let variance := (∑ j, (xs j - μ) ^ 2) / N
      (xs i - μ) / Real.sqrt (variance + ε) * gs i + bs i) := by
  have halg : (twoPassIO N stride ε).kernel.toAlgKernel =
      (twoPassMathKernel "x" "gamma" "beta" "y" N stride ε).toAlgKernel := by
    simp [twoPassIO, ComputeKernel.toAlgKernel, twoPass_projection]
  refine KernelIO₃.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact twoPass_flattenOk "x" "gamma" "beta" "y" N stride ε
  · intro bounds s hx hg hb hy _
    rw [halg]
    simp only [twoPassIO, Nat.zero_add] at hg hb
    exact twoPass_traceSafe "x" "gamma" "beta" "y" N stride ε s bounds hx hg hb hy
  · intro s xs gs bs hx hg hb
    rw [halg]
    simp only [twoPassIO, Nat.zero_add] at hg hb
    obtain ⟨t, he, hv, hf⟩ := twoPass_region_run N stride ε s xs gs bs hx hg hb
    refine ⟨t, he, ?_, ?_⟩
    · simpa [twoPassMean, twoPassS] using hv
    · intro r o hmiss _; exact hf r o hmiss

def fusedIO (N stride : Nat) (ε : ℝ) : KernelIO₃ where
  kernel := (fusedLayerNormKernel "x" "gamma" "beta" "y" N stride ε).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, fused_projection]
  in1 := "x"
  in2 := "gamma"
  in3 := "beta"
  out := "y"
  B1 := N
  B2 := N
  B3 := N
  Bout := N
  read1 := fun pid => pid * stride
  read2 := fun _ => 0
  read3 := fun _ => 0
  write := fun pid => pid * stride

specification fused_layernorm_correct (N stride : Nat) (ε : ℝ) :
    Spec.Real (fusedIO N stride ε ⊨ fun xs gs bs i =>
      let μ := (∑ j, xs j) / N
      let variance := (∑ j, (xs j - μ) ^ 2) / N
      (xs i - μ) / Real.sqrt (variance + ε) * gs i + bs i) := by
  have halg : (fusedIO N stride ε).kernel.toAlgKernel =
      (fusedMathKernel "x" "gamma" "beta" "y" N stride ε).toAlgKernel := by
    simp [fusedIO, ComputeKernel.toAlgKernel, fused_projection]
  refine KernelIO₃.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact fused_flattenOk "x" "gamma" "beta" "y" N stride ε
  · intro bounds s hx hg hb hy _
    rw [halg]
    simp only [fusedIO, Nat.zero_add] at hg hb
    exact fused_traceSafe "x" "gamma" "beta" "y" N stride ε s bounds hx hg hb hy
  · intro s xs gs bs hx hg hb
    rw [halg]
    simp only [fusedIO, Nat.zero_add] at hg hb
    obtain ⟨t, he, hv, hf⟩ := fused_region_run N stride ε s xs gs bs hx hg hb
    refine ⟨t, he, ?_, ?_⟩
    · simpa [twoPassMean, twoPassS] using hv
    · intro r o hmiss _; exact hf r o hmiss

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FusedLayerNormCorrect
