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
import VeriTile.Examples.WelfordMath

namespace VeriTile.Examples

open VeriTile.Triton

/-- Two-pass LayerNorm kernel: `tl.sum` twice (mean and var), then affine. -/
def twoPassLayerNormKernel
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) : Kernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(N) + tl.arange($(N))
  x      := tl.load($(xReg) + offs)
  s_x    := tl.sum(x)
  μ      := s_x / tl.toReal($(N))
  d      := x - μ
  s_d2   := tl.sum(d * d)
  v      := s_d2 / tl.toReal($(N))
  γ      := tl.load($(γReg) + tl.arange($(N)))
  β      := tl.load($(βReg) + tl.arange($(N)))
  σ_inv  := 1 / tl.sqrt(v + $ℝ(ε))
  y      := (x - μ) * σ_inv * γ + β
  tl.store($(yReg) + offs, y)
}

/-- Fused single-pass LayerNorm kernel: Welford `forLoop`, then affine. -/
def fusedLayerNormKernel
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) : Kernel := triton {
  pid := tl.program_id(0)
  M   := 0
  S   := 0
  tl.for i in $(N) {
    xi      := tl.load($(xReg) + (pid * $(N) + i))
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
  x       := tl.load($(xReg) + offs)
  γ       := tl.load($(γReg) + tl.arange($(N)))
  β       := tl.load($(βReg) + tl.arange($(N)))
  y       := (x - μ) * σ_inv * γ + β
  tl.store($(yReg) + offs, y)
}

def layerNormAffineTailKernel
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) : Kernel :=
  { inputs := [xReg, γReg, βReg]
    outputs := [yReg]
    body :=
      [ .assign .real .scalar "μ" (.ref .real .scalar "M")
      , .assign .real .scalar "v"
          (.div .real .same (.ref .real .scalar "S") (Op.natToReal (.constNat N)))
      , .assign .real .scalar "σ_inv"
          (.div .real .same (.const 1)
            (.sqrt (.add .real .same (.ref .real .scalar "v") (.const ε))))
      , .assign .nat (.vec N) "offs"
          (.add .nat .left
            (.mul .nat .same (.ref .nat .scalar "pid") (.constNat N))
            (.arange N))
      , .assign .real (.vec N) "x" (.load xReg (.ref .nat (.vec N) "offs"))
      , .assign .real (.vec N) "γ" (.load γReg (.arange N))
      , .assign .real (.vec N) "β" (.load βReg (.arange N))
      , .assign .real (.vec N) "y"
          (.add .real .same
            (.mul .real .same
              (.mul .real .right
                (.sub .real .right (.ref .real (.vec N) "x") (.ref .real .scalar "μ"))
                (.ref .real .scalar "σ_inv"))
              (.ref .real (.vec N) "γ"))
            (.ref .real (.vec N) "β"))
      , .store yReg (.vec N) (.ref .nat (.vec N) "offs") (.ref .real (.vec N) "y")
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
  s.regs .real .scalar "M" = some (Tile.scalar (welfordMean xs k))
  ∧ s.regs .real .scalar "S" = some (Tile.scalar (welfordS xs k))
  ∧ s.regs .nat .scalar "pid" = some (Tile.scalar origPid)
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
        (s.setReg "i" .nat .scalar (Tile.scalar i)) = some s' ∧
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
    (((((s.setReg "i" .nat .scalar (Tile.scalar i)).setReg
      "xi" .real .scalar (Tile.scalar xi)).setReg
      "delta" .real .scalar (Tile.scalar delta)).setReg
      "M" .real .scalar (Tile.scalar m')).setReg
      "delta2" .real .scalar (Tile.scalar delta2)).setReg
      "S" .real .scalar (Tile.scalar ssum')
  refine ⟨s', ?_, ?_⟩
  · simp [onlineWelfordLoopBody, stepStmts, stepStmt, evalOp, Tile.bop,
      Tile.natToReal, NumericDType.add, NumericDType.mul, NumericDType.sub,
      NumericDType.div, BlockState.readMem, hM, hS, hpidReg,
      xi, m, ssum, delta, m', delta2, ssum', s',
      WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv,
      ← WithBot.coe_add, ← WithBot.coe_mul, BlockState.setReg]
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
      ((s.setReg "pid" .nat .scalar (Tile.scalar s.pid)).setReg
        "M" .real .scalar (Tile.scalar 0)).setReg
        "S" .real .scalar (Tile.scalar 0)
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
  -- TODO(W11.M3.4): WithBot refactor — full LayerNorm tail proof needs the
  -- coe-bridge simp set tuned for `√(var + ε)` and `(x - μ) * γ + β` patterns.
  sorry

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
  have h_inj : Function.Injective (fun k : Fin N => s.pid * N + k.val) := by
    intro a b hab
    exact Fin.ext (Nat.add_left_cancel hab)
  simp [observeAt, exec, twoPassLayerNormKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.natToReal,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        BlockState.setReg, BlockState.readMem, layerNormSpec, twoPassMean,
        twoPassS]
  simp [Broadcast.leftIndex, Broadcast.rightIndex]
  unfold InputLoadedAt at _h_x
  unfold InputFeatureLoadedAt at _h_γ _h_β
  rw [BlockState.scatter_readback _ _ _ h_inj i]
  simp [_h_x, _h_γ, _h_β, pow_two, div_eq_mul_inv]

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
    ((s.setReg "pid" .nat .scalar (Tile.scalar s.pid)).setReg
      "M" .real .scalar (Tile.scalar 0)).setReg
      "S" .real .scalar (Tile.scalar 0)
  obtain ⟨sLoop, hLoop, hPloop⟩ :=
    layernorm_welford_loop xReg γReg βReg N s xs γs βs _h_x _h_γ _h_β
  have hLoopAux :
      stepForLoopAux "i" 0 N (onlineWelfordLoopBody xReg N) s0 =
        some sLoop := by
    simpa [s0, stepForLoopAux.forLoop_unfold] using hLoop
  have hLoopAuxExpanded :
      stepForLoopAux "i" 0 N
        [Stmt.assign .real .scalar "xi"
            (Op.load xReg
              (Op.add .nat .same
                (Op.mul .nat .same (Op.ref .nat .scalar "pid")
                  (Op.constNat N))
                (Op.ref .nat .scalar "i"))),
          Stmt.assign .real .scalar "delta"
            (Op.sub .real .same (Op.ref .real .scalar "xi")
              (Op.ref .real .scalar "M")),
          Stmt.assign .real .scalar "M"
            (Op.add .real .same (Op.ref .real .scalar "M")
              (Op.div .real .same (Op.ref .real .scalar "delta")
                (Op.add .real .same (Op.ref .nat .scalar "i").natToReal
                  (Op.const 1)))),
          Stmt.assign .real .scalar "delta2"
            (Op.sub .real .same (Op.ref .real .scalar "xi")
              (Op.ref .real .scalar "M")),
          Stmt.assign .real .scalar "S"
            (Op.add .real .same (Op.ref .real .scalar "S")
              (Op.mul .real .same (Op.ref .real .scalar "delta")
                (Op.ref .real .scalar "delta2")))]
        s0 = some sLoop := by
    simpa [onlineWelfordLoopBody] using hLoopAux
  have h_exec_tail :
      exec (fusedLayerNormKernel xReg γReg βReg yReg N ε) s =
        exec (layerNormAffineTailKernel xReg γReg βReg yReg N ε) sLoop := by
    -- TODO(W11.M3.4): same WithBot bridge issue as Welford's final stitch.
    sorry
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

end VeriTile.Examples
