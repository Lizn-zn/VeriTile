/-
VeriTile.Examples.LayerNormKernels

Tier 2 kernel-pair: fused single-pass LayerNorm ≡ two-pass LayerNorm.
Composes the Welford recurrence (Phase B #4) with the affine
`(x − μ)/√(var+ε) · γ + β` transform.

The two-pass kernel proof is closed over the typed tile semantics. The fused
kernel proof still depends on the Welford loop invariant.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.Common
import VeriTile.Examples.WelfordKernels

namespace VeriTile.Examples

open VeriTile.Triton

/-- Two-pass LayerNorm kernel: `tl.sum` twice (mean and var), then affine. -/
def twoPassLayerNormKernel
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) : Kernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(N) + tl.arange($(N))
  x      := tl.load(tl.ptr($(xReg)) + offs)
  s_x    := tl.sum(x)
  μ      := s_x / tl.toReal($(N))
  d      := x - μ
  s_d2   := tl.sum(d * d)
  v      := s_d2 / tl.toReal($(N))
  γ      := tl.load(tl.ptr($(γReg)) + tl.arange($(N)))
  β      := tl.load(tl.ptr($(βReg)) + tl.arange($(N)))
  σ_inv  := 1 / tl.sqrt(v + $ℝ(ε))
  y      := (x - μ) * σ_inv * γ + β
  tl.store(tl.ptr($(yReg)) + offs, y)
}

/-- Fused single-pass LayerNorm kernel: Welford `forLoop`, then affine. -/
def fusedLayerNormKernel
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) : Kernel := triton {
  pid := tl.program_id(0)
  M   := 0
  S   := 0
  tl.for i in $(N) {
    xi      := tl.load(tl.ptr($(xReg)) + (pid * $(N) + i))
    delta   := xi - M
    M       := M + delta / (tl.toReal(i) + 1)
    delta2  := xi - M
    S       := S + delta * delta2
  }
  μ       := M
  v       := S / tl.toReal($(N))
  σ_inv   := 1 / tl.sqrt(v + $ℝ(ε))
  -- Second pass to compute Y. The "fused" gain is that μ/var were
  -- computed in a single pass over `x`; the residual `(x − μ)` still
  -- needs the second read of x.
  offs    := pid * $(N) + tl.arange($(N))
  x       := tl.load(tl.ptr($(xReg)) + offs)
  γ       := tl.load(tl.ptr($(γReg)) + tl.arange($(N)))
  β       := tl.load(tl.ptr($(βReg)) + tl.arange($(N)))
  y       := (x - μ) * σ_inv * γ + β
  tl.store(tl.ptr($(yReg)) + offs, y)
}

def layerNormAffineTailKernel
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) : Kernel :=
  { inputs := [xReg, γReg, βReg]
    outputs := [yReg]
    body :=
      [ .assign .real [] "μ" (.ref .real [] "M")
      , .assign .real [] "v"
          (.div .real .nil (.ref .real [] "S") (Op.natToReal (.constNat N)))
      , .assign .real [] "σ_inv"
          (.div .real .nil (.const 1)
            (.sqrt (.add .real .nil (.ref .real [] "v") (.const ε))))
      , .assign .nat [N] "offs"
          (.add .nat .scalarL
            (.mul .nat .nil (.ref .nat [] "pid") (.constNat N))
            (.arange N))
      , .assign .real [N] "x" (.load xReg (.ref .nat [N] "offs"))
      , .assign .real [N] "γ" (.load γReg (.arange N))
      , .assign .real [N] "β" (.load βReg (.arange N))
      , .assign .real [N] "y"
          (.add .real (.consSame .nil)
            (.mul .real (.consSame .nil)
              (.mul .real .scalarR
                (.sub .real .scalarR (.ref .real [N] "x") (.ref .real [] "μ"))
                (.ref .real [] "σ_inv"))
              (.ref .real [N] "γ"))
            (.ref .real [N] "β"))
      , .store yReg [N] (.ref .nat [N] "offs") (.ref .real [N] "y")
      ] }

/-- LayerNorm spec: `y_i = (x_i − μ) / √(var + ε) · γ_i + β_i`. -/
noncomputable def layerNormSpec {N : Nat}
    (xs γs βs : Fin N → ℝ) (ε : ℝ) (i : Fin N) : ℝ :=
  let μ : ℝ := twoPassMean xs
  let v : ℝ := twoPassS xs / N
  (xs i - μ) / Real.sqrt (v + ε) * γs i + βs i

private def P_layernorm {N : Nat}
    (xs γs βs : Fin N → ℝ) (xReg γReg βReg : RegionName)
    (origPid : Nat) (k : Nat) (s : BlockState) : Prop :=
  s.regs .real [] "M" = some (Tile.scalar (welfordMean xs k))
  ∧ s.regs .real [] "S" = some (Tile.scalar (welfordS xs k))
  ∧ s.regs .nat [] "pid" = some (Tile.scalar origPid)
  ∧ s.pid = origPid
  ∧ InputLoadedAt s xReg N xs
  ∧ InputFeatureLoadedAt s γReg N γs
  ∧ InputFeatureLoadedAt s βReg N βs

private theorem layernorm_welford_step
    {N : Nat} (xs γs βs : Fin N → ℝ)
    (xReg γReg βReg : RegionName) (origPid i : Nat)
    (s : BlockState) (hi : i < N)
    (hP : P_layernorm xs γs βs xReg γReg βReg origPid i s) :
    ∃ s',
      stepStmts (onlineWelfordLoopBody xReg N)
        (s.setReg "i" .nat [] (Tile.scalar i)) = some s' ∧
      P_layernorm xs γs βs xReg γReg βReg origPid (i + 1) s' := by
  rcases hP with ⟨hM, hS, hpidReg, hpid, hX, hγ, hβ⟩
  let xi : ℝ := s.mem xReg (origPid * N + i)
  have hxi : xi = xs ⟨i, hi⟩ := by
    have hx := hX ⟨i, hi⟩
    rw [hpid] at hx
    exact hx
  let m : ℝ := welfordMean xs i
  let ssum : ℝ := welfordS xs i
  let delta : ℝ := xi - m
  let m' : ℝ := m + delta / ((i : ℝ) + 1)
  let delta2 : ℝ := xi - m'
  let ssum' : ℝ := ssum + delta * delta2
  let s' :=
    (((((s.setReg "i" .nat [] (Tile.scalar i)).setReg
      "xi" .real [] (Tile.scalar xi)).setReg
      "delta" .real [] (Tile.scalar delta)).setReg
      "M" .real [] (Tile.scalar m')).setReg
      "delta2" .real [] (Tile.scalar delta2)).setReg
      "S" .real [] (Tile.scalar ssum')
  refine ⟨s', ?_, ?_⟩
  · simp [onlineWelfordLoopBody, stepStmts, stepStmt, evalOp, Tile.bop,
      Tile.natToReal, NumericDType.add, NumericDType.mul, NumericDType.sub,
      NumericDType.div, BlockState.readMem, hM, hS, hpidReg,
      xi, m, ssum, delta, m', delta2, ssum', s',
      WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv,
      BlockState.setReg]
    rfl
  · simp [P_layernorm, s', InputLoadedAt, InputFeatureLoadedAt, welfordMean,
      welfordS, hi, xi, m, ssum, delta, m', delta2, ssum', hpidReg, hpid, hxi]
    constructor
    · intro j
      have hx := hX j
      rw [hpid] at hx
      exact hx
    · exact ⟨hγ, hβ⟩

private theorem layernorm_welford_loop
    (xReg γReg βReg : RegionName) (N : Nat)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (h_x : InputLoadedAt s xReg N xs)
    (h_γ : InputFeatureLoadedAt s γReg N γs)
    (h_β : InputFeatureLoadedAt s βReg N βs) :
    let s0 :=
      ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
        "M" .real [] (Tile.scalar 0)).setReg
        "S" .real [] (Tile.scalar 0)
    ∃ sLoop,
      stepStmt (.forLoop "i" N (onlineWelfordLoopBody xReg N)) s0 = some sLoop
      ∧ P_layernorm xs γs βs xReg γReg βReg s.pid N sLoop := by
  intro s0
  have h_init : P_layernorm xs γs βs xReg γReg βReg s.pid 0 s0 := by
    simp [P_layernorm, s0, welfordMean, welfordS]
    exact ⟨h_x, h_γ, h_β⟩
  exact forLoop_inv
    (idx := "i") (n := N)
    (body := onlineWelfordLoopBody xReg N)
    (P := P_layernorm xs γs βs xReg γReg βReg s.pid)
    (s_init := s0)
    h_init
    (fun k st hk hP =>
      layernorm_welford_step xs γs βs xReg γReg βReg s.pid k st hk hP)

private theorem layernorm_affine_tail_correct
    (xReg γReg βReg yReg : RegionName) (N : Nat) (hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (hP : P_layernorm xs γs βs xReg γReg βReg s.pid N s) :
    ∀ i : Fin N,
      observeAt (exec (layerNormAffineTailKernel xReg γReg βReg yReg N ε) s)
                yReg N s.pid i
        = some (layerNormSpec xs γs βs ε i) := by
  intro i
  rcases hP with ⟨hM, hS, hpidReg, _hpid, hX, hγ, hβ⟩
  have hMean := (welford_eq_two_pass hN xs).1
  have hSEq := (welford_eq_two_pass hN xs).2
  have h_inj : Function.Injective
      (fun idx : TileIndex [N] => s.pid * N + idx.1.val) :=
    injective_offset_singleton (s.pid * N)
  simp [observeAt, exec, layerNormAffineTailKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.natToReal,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        BlockState.setReg, BlockState.readMem, layerNormSpec,
        hM, hS, hpidReg, hMean, hSEq]
  unfold InputLoadedAt at hX
  unfold InputFeatureLoadedAt at hγ hβ
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [hX, hγ, hβ, div_eq_mul_inv]

set_option maxHeartbeats 800000 in
theorem twopass_layernorm_correct
    (xReg γReg βReg yReg : RegionName) (N : Nat) (_hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs)
    (_h_γ : InputFeatureLoadedAt s γReg N γs)
    (_h_β : InputFeatureLoadedAt s βReg N βs)
    (_h_yx : yReg ≠ xReg) (_h_yγ : yReg ≠ γReg) (_h_yβ : yReg ≠ βReg) :
    ∀ i : Fin N,
      observeAt (exec (twoPassLayerNormKernel xReg γReg βReg yReg N ε) s)
                yReg N s.pid i
        = some (layerNormSpec xs γs βs ε i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [N] => s.pid * N + idx.1.val) :=
    injective_offset_singleton (s.pid * N)
  simp [observeAt, exec, twoPassLayerNormKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        Tile.natToReal,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        BlockState.setReg, BlockState.readMem, layerNormSpec, twoPassMean,
        twoPassS]
  unfold InputLoadedAt at _h_x
  unfold InputFeatureLoadedAt at _h_γ _h_β
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x, _h_γ, _h_β, pow_two, div_eq_mul_inv]
  exact Or.inl rfl

theorem fused_layernorm_correct
    (xReg γReg βReg yReg : RegionName) (N : Nat) (_hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs)
    (_h_γ : InputFeatureLoadedAt s γReg N γs)
    (_h_β : InputFeatureLoadedAt s βReg N βs)
    (_h_yx : yReg ≠ xReg) (_h_yγ : yReg ≠ γReg) (_h_yβ : yReg ≠ βReg) :
    ∀ i : Fin N,
      observeAt (exec (fusedLayerNormKernel xReg γReg βReg yReg N ε) s)
                yReg N s.pid i
        = some (layerNormSpec xs γs βs ε i) := by
  intro i
  let s0 :=
    ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
      "M" .real [] (Tile.scalar 0)).setReg
      "S" .real [] (Tile.scalar 0)
  obtain ⟨sLoop, hLoop, hPloop⟩ :=
    layernorm_welford_loop xReg γReg βReg N s xs γs βs _h_x _h_γ _h_β
  have hLoopAux :
      stepForLoopAux "i" 0 N (onlineWelfordLoopBody xReg N) s0 =
        some sLoop := by
    simpa [s0, stepForLoopAux.forLoop_unfold] using hLoop
  have hLoopAuxExpanded :
      stepForLoopAux "i" 0 N
        [Stmt.assign .real [] "xi"
            (Op.load xReg
              (Op.add .nat .nil
                (Op.mul .nat .nil (Op.ref .nat [] "pid")
                  (Op.constNat N))
                (Op.ref .nat [] "i"))),
          Stmt.assign .real [] "delta"
            (Op.sub .real .nil (Op.ref .real [] "xi")
              (Op.ref .real [] "M")),
          Stmt.assign .real [] "M"
            (Op.add .real .nil (Op.ref .real [] "M")
              (Op.div .real .nil (Op.ref .real [] "delta")
                (Op.add .real .nil (Op.ref .nat [] "i").natToReal
                  (Op.const 1)))),
          Stmt.assign .real [] "delta2"
            (Op.sub .real .nil (Op.ref .real [] "xi")
              (Op.ref .real [] "M")),
          Stmt.assign .real [] "S"
            (Op.add .real .nil (Op.ref .real [] "S")
              (Op.mul .real .nil (Op.ref .real [] "delta")
                (Op.ref .real [] "delta2")))]
        s0 = some sLoop := by
    simpa [onlineWelfordLoopBody] using hLoopAux
  have h_exec_tail :
      exec (fusedLayerNormKernel xReg γReg βReg yReg N ε) s =
        exec (layerNormAffineTailKernel xReg γReg βReg yReg N ε) sLoop := by
    -- The fused kernel's body is `[pid; M:=0; S:=0; forLoop body; tail...]`.
    -- The affine tail kernel's body is the same `tail...`. Therefore:
    -- exec fused s = stepStmts (tail) sLoop = exec affineTail sLoop.
    have hpid : stepStmt (.assign .nat [] "pid" (.programId 0)) s
                  = some (s.setReg "pid" .nat [] (Tile.scalar s.pid)) := by
      simp [stepStmt, evalOp]
    have hM0 : stepStmt (.assign .real [] "M" (.const 0))
                  (s.setReg "pid" .nat [] (Tile.scalar s.pid))
                = some ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
                    "M" .real [] (Tile.scalar 0)) := by
      simp [stepStmt, evalOp]
      rfl
    have hS0 : stepStmt (.assign .real [] "S" (.const 0))
                  ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
                    "M" .real [] (Tile.scalar 0))
                = some s0 := by
      simp [stepStmt, evalOp, s0]
      rfl
    show stepStmts (fusedLayerNormKernel xReg γReg βReg yReg N ε).body s
        = stepStmts (layerNormAffineTailKernel xReg γReg βReg yReg N ε).body sLoop
    -- Unfold kernel body literals so `stepStmts.cons_some` can match `::`.
    simp only [show (fusedLayerNormKernel xReg γReg βReg yReg N ε).body =
        ([ .assign .nat [] "pid" (.programId 0)
         , .assign .real [] "M" (.const 0)
         , .assign .real [] "S" (.const 0)
         , .forLoop "i" N (onlineWelfordLoopBody xReg N)
         ] ++ (layerNormAffineTailKernel xReg γReg βReg yReg N ε).body) from rfl]
    show stepStmts ([_, _, _, _] ++ _) s = _
    -- Reduce stepStmts on append: stepStmts (l ++ rest) s steps through l first
    rw [show ∀ (a b c d : Stmt) (rest : List Stmt) (s : BlockState),
        stepStmts ([a, b, c, d] ++ rest) s
          = stepStmts (a :: b :: c :: d :: rest) s from fun _ _ _ _ _ _ => rfl]
    rw [stepStmts.cons_some hpid]
    rw [stepStmts.cons_some hM0]
    rw [stepStmts.cons_some hS0]
    rw [stepStmts.cons_some hLoop]
  rw [h_exec_tail]
  have hpidLoop : sLoop.pid = s.pid := by
    exact hPloop.2.2.2.1
  have hPtail : P_layernorm xs γs βs xReg γReg βReg sLoop.pid N sLoop := by
    rcases hPloop with ⟨hM, hS, hpidReg, hpid, hX, hγ, hβ⟩
    rw [hpid]
    exact ⟨hM, hS, hpidReg, hpid, hX, hγ, hβ⟩
  rw [← hpidLoop]
  exact layernorm_affine_tail_correct xReg γReg βReg yReg N _hN ε sLoop
    xs γs βs hPtail i

theorem layernorm_kernels_refinement
    (xReg γReg βReg yReg : RegionName) (N : Nat) (_hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs)
    (_h_γ : InputFeatureLoadedAt s γReg N γs)
    (_h_β : InputFeatureLoadedAt s βReg N βs)
    (_h_yx : yReg ≠ xReg) (_h_yγ : yReg ≠ γReg) (_h_yβ : yReg ≠ βReg) :
    ∀ i : Fin N,
      observeAt (exec (twoPassLayerNormKernel xReg γReg βReg yReg N ε) s)
                yReg N s.pid i
        = observeAt (exec (fusedLayerNormKernel xReg γReg βReg yReg N ε) s)
                yReg N s.pid i := by
  intro i
  rw [twopass_layernorm_correct
        xReg γReg βReg yReg N _hN ε s xs γs βs
        _h_x _h_γ _h_β _h_yx _h_yγ _h_yβ i,
      fused_layernorm_correct
        xReg γReg βReg yReg N _hN ε s xs γs βs
        _h_x _h_γ _h_β _h_yx _h_yγ _h_yβ i]

/-- View-level surface for `layernorm_kernels_refinement`. -/
theorem layernorm_kernels_refinement_view
    (xReg γReg βReg yReg : RegionName) (N : Nat) (hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1))
    (h_γ : TensorView.loaded s (featureView γReg N)
      (fun idx : TileIndex [N] => γs idx.1))
    (h_β : TensorView.loaded s (featureView βReg N)
      (fun idx : TileIndex [N] => βs idx.1))
    (h_yx : yReg ≠ xReg) (h_yγ : yReg ≠ γReg) (h_yβ : yReg ≠ βReg) :
    ∀ idx : TileIndex [N],
      TensorView.observe (exec (twoPassLayerNormKernel xReg γReg βReg yReg N ε) s)
          (programTileView s yReg N) idx
        = TensorView.observe (exec (fusedLayerNormKernel xReg γReg βReg yReg N ε) s)
          (programTileView s yReg N) idx := by
  intro idx
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := N) (xs := xs) h_x
  have hγ := inputFeatureLoadedAt_of_featureView_loaded (s := s) (region := γReg)
    (N := N) (xs := γs) h_γ
  have hβ := inputFeatureLoadedAt_of_featureView_loaded (s := s) (region := βReg)
    (N := N) (xs := βs) h_β
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt]
    using layernorm_kernels_refinement xReg γReg βReg yReg N hN ε s xs γs βs
      hx hγ hβ h_yx h_yγ h_yβ idx.1

end VeriTile.Examples
