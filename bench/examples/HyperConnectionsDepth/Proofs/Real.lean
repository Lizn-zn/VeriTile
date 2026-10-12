import bench.examples.HyperConnectionsDepth.Kernels
import VeriTile.Triton.Math.Sinkhorn
import VeriTile.Triton.Math.MatrixRewrite
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Math.SinkhornMemory

/-! Real execution, arithmetic and memory lemmas for HyperConnectionsDepth.
The public IO contracts and mathematical specifications are in ../Correct.lean. -/

namespace VeriTile.Bench.Examples.HyperConnectionsDepth.Memory
open VeriTile Triton TraceProofs Kernels

set_option maxHeartbeats 6400000

set_option linter.unusedSimpArgs false

theorem traceSafe (optimized : Bool) (S T D iters : Nat) (tau : ℝ)
    (bounds : RegionBounds) (s : BlockState)
    (hres : s.pid * (S * D) + S * D ≤ bounds "res_mix")
    (hbranch : s.pid * (T * D) + T * D ≤ bounds "branch_out")
    (hpost : T * S ≤ bounds "h_post")
    (hout : s.pid * (S * D) + S * D ≤ bounds "out") :
    Kernel.TraceSafe bounds
      (if optimized then (matrixOptimized "res_mix" "branch_out" "h_post" "out" S T D iters tau).toAlgKernel
       else (matrixOriginal S T D iters tau).toAlgKernel) s := by
  cases optimized <;> simp only [Bool.false_eq_true, ↓reduceIte]
  all_goals
    change Stmt.TraceSafeList bounds _ s
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_cons
    · simp only [Op.SafeAt.eq_def, and_true, true_and]
      apply region_safe (offsets := ⟨fun i : TileIndex [S, D] => (s.pid * (S * D)) + i.1.val * D + i.2.1.val⟩)
      · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
          Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
          NumericDType.add, NumericDType.mul]
      · intro i
        exact lt_of_lt_of_le (matrix_offset_lt (s.pid * (S * D)) S D i.1 i.2.1) hres
    intro res
    apply assign_cons
    · simp only [Op.SafeAt.eq_def, and_true, true_and]
      apply region_safe (offsets := ⟨fun i : TileIndex [T, D] => (s.pid * (T * D)) + i.1.val * D + i.2.1.val⟩)
      · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
          Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
          NumericDType.add, NumericDType.mul]
      · intro i
        exact lt_of_lt_of_le (matrix_offset_lt (s.pid * (T * D)) T D i.1 i.2.1) hbranch
    intro branch
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_cons
    · simp only [Op.SafeAt.eq_def, and_true, true_and]
      apply region_safe (offsets := ⟨fun i : TileIndex [T, S] => 0 + i.1.val * S + i.2.1.val⟩)
      · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
          Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
          NumericDType.add, NumericDType.mul]
      · intro i
        exact lt_of_lt_of_le (matrix_offset_lt 0 T S i.1 i.2.1) (by simpa using hpost)
    intro post
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro z
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro u
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro v
    refine Stmt.TraceSafeList.cons_intro (LogSinkhorn.loop_traceSafe T S iters bounds _) ?_
    intro q hq
    have hb := LogSinkhorn.loop_register T S iters _ q hq .nat [] "b" (by decide) (by decide) (by decide)
    have hs := LogSinkhorn.loop_register T S iters _ q hq .nat [S] "offs_s" (by decide) (by decide) (by decide)
    have hd := LogSinkhorn.loop_register T S iters _ q hq .nat [D] "offs_d" (by decide) (by decide) (by decide)
    simp only [BlockState.setReg] at hb hs hd
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro weights
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro mix
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro out
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul, hb, hs, hd] at hval
    subst val
    refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
    simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt, Op.SafeAt.eq_def, and_true, true_and]
    apply region_safe (offsets := ⟨fun i : TileIndex [S, D] => s.pid * (S * D) + i.1.val * D + i.2.1.val⟩)
    · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec, Tile.scalar,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul, hb, hs, hd]
    · intro i
      exact lt_of_lt_of_le (matrix_offset_lt (s.pid * (S * D)) S D i.1 i.2.1) hout

/-- Both sources remain in the fragment supported by address relocation. -/
theorem flattenOk (optimized : Bool) (S T D iters : Nat) (tau : ℝ) :
    (if optimized then (matrixOptimized "res_mix" "branch_out" "h_post" "out" S T D iters tau).toAlgKernel
     else (matrixOriginal S T D iters tau).toAlgKernel).FlattenOk := by
  cases optimized <;> simp [Kernel.FlattenOk, matrixOriginal, matrixKernel, matrixOptimized, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

end VeriTile.Bench.Examples.HyperConnectionsDepth.Memory

namespace VeriTile.Bench.Examples.HyperConnectionsDepth
open VeriTile.Bench.Examples.HyperConnectionsDepth.Kernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?
  VeriTile.Triton.ComputeOp.toAlgorithm? VeriTile.Triton.ComputeDType.eraseDType

open VeriTile.Triton
open VeriTile.Triton.KernelIO₃ (Implements)
open scoped VeriTile.Triton.KernelIO₃

/-- Value half: the run terminates and `out[pid]` holds
`resMix + exp(hPost/τ)·branchOut` (in terms of the launch state's own memory
cells). -/
private theorem mhcDepth_exec_view (tau : ℝ) (s : BlockState) :
    (exec ((mhcDepthConnectionKernel ⟨"res_mix"⟩ ⟨"branch_out"⟩ ⟨"h_post"⟩
        ⟨"out"⟩ 1 1 1 0 tau).toAlgKernel) s).map
      (fun s' => s'.readMem ⟨"out"⟩ s.pid)
      = some
        (s.readMem ⟨"res_mix"⟩ s.pid
          + Real.exp (s.readMem ⟨"h_post"⟩ 0 / tau)
            * s.readMem ⟨"branch_out"⟩ s.pid) := by
  simp [mhcDepthConnectionKernel, exec, stepStmts,
    stepStmt, NumericDType.add, NumericDType.mul, NumericDType.div,
    WithBot.realAdd, WithBot.realMul, WithBot.realDiv,
    BlockState.writeMem, BlockState.readMem]

/-- Frame half: every memory cell other than the output cell `out[pid]` is
preserved by the run. -/
private theorem mhcDepth_frame (tau : ℝ) (s s1 : BlockState)
    (hExec : exec ((mhcDepthConnectionKernel ⟨"res_mix"⟩ ⟨"branch_out"⟩
        ⟨"h_post"⟩ ⟨"out"⟩ 1 1 1 0 tau).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ¬(r = ⟨"out"⟩ ∧ o = s.pid)) :
    s1.mem r o = s.mem r o := by
  simp [exec, mhcDepthConnectionKernel,
    ComputeKernel.toAlgKernel, stepStmts, stepStmt,
    NumericDType.add, NumericDType.mul, NumericDType.div] at hExec
  subst hExec
  rw [BlockState.writeMem_mem, if_neg hmiss]
  rfl

/-- **The region-model Hoare triple** — termination, the output-cell value,
and frame, from any launch state whose input windows are loaded. This is
what the `⊨` headline transports to flat memory. -/
theorem mhcDepth_region_run (tau : ℝ) (s₀ : BlockState)
    (resMix branchOut hPost : Fin 1 → ℝ)
    (hmix : ∀ j : Fin 1, s₀.readMem ⟨"res_mix"⟩ (s₀.pid + j.val) = resMix j)
    (hbout : ∀ j : Fin 1, s₀.readMem ⟨"branch_out"⟩ (s₀.pid + j.val) = branchOut j)
    (hhpost : ∀ j : Fin 1, s₀.readMem ⟨"h_post"⟩ (0 + j.val) = hPost j) :
    ∃ s1,
      exec ((mhcDepthConnectionKernel ⟨"res_mix"⟩ ⟨"branch_out"⟩ ⟨"h_post"⟩
          ⟨"out"⟩ 1 1 1 0 tau).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin 1, s1.readMem ⟨"out"⟩ (s₀.pid + j.val)
          = resMix 0 + Real.exp (hPost 0 / tau) * branchOut 0)
      ∧ (∀ r o,
          (r ≠ ⟨"out"⟩ ∨ ∀ j : Fin 1, o ≠ s₀.pid + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have h0mix : s₀.readMem ⟨"res_mix"⟩ s₀.pid = resMix 0 := by simpa using hmix 0
  have h0bout : s₀.readMem ⟨"branch_out"⟩ s₀.pid = branchOut 0 := by
    simpa using hbout 0
  have h0hpost : s₀.readMem ⟨"h_post"⟩ 0 = hPost 0 := by simpa using hhpost 0
  have hview := mhcDepth_exec_view tau s₀
  cases hsrc : exec ((mhcDepthConnectionKernel ⟨"res_mix"⟩ ⟨"branch_out"⟩
      ⟨"h_post"⟩ ⟨"out"⟩ 1 1 1 0 tau).toAlgKernel) s₀ with
  | none => rw [hsrc] at hview; simp at hview
  | some s1 =>
      rw [hsrc] at hview
      simp only [Option.map_some, Option.some_inj] at hview
      refine ⟨s1, rfl, fun j => ?_, fun r o hcond => ?_⟩
      · have hj : (j : Nat) = 0 := Nat.lt_one_iff.mp j.isLt
        rw [hj, Nat.add_zero, hview, h0mix, h0hpost, h0bout]
      · refine mhcDepth_frame tau s₀ s1 hsrc r o ?_
        rintro ⟨hr, ho⟩
        rcases hcond with hne | hno
        · exact hne hr
        · exact hno 0 (by simpa using ho)

/-- Inversion for a successful `assign` step. -/
private theorem stepStmt_assign_inv {d : TileDType} {sh : TileShape}
    {nm : RegName} {e : Op d sh} {s s' : BlockState}
    (h : stepStmt (.assign d sh nm e) s = some s') :
    ∃ v, evalOp e s = some v ∧ s' = s.setReg nm d sh v := by
  simp only [stepStmt] at h
  cases hv : evalOp e s with
  | none => rw [hv] at h; exact absurd h (by simp)
  | some v =>
      rw [hv] at h
      replace h : some (s.setReg nm d sh v) = some s' := h
      exact ⟨v, rfl, (Option.some_inj.mp h).symm⟩

/-- Bounds discharge for a single-cell access through the scalar `b`
register: the one addressed cell is `pid₀`, in bounds when
`pid₀ + 1 ≤ bounds reg`. -/
private theorem scalarB_activeAddressSafe (bounds : RegionBounds)
    (pid₀ : Nat) (t : BlockState) (active : TileIndex [] → Prop)
    (hb : t.regs .nat [] "b" = some (Tile.scalar (dtype := .nat) pid₀))
    (reg : RegionName) (hreg : pid₀ + 1 ≤ bounds reg) :
    MemAccess.ActiveAddressSafe bounds
      (MemAccess.region reg (Op.ref .nat [] "b")) t active := by
  simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
  intro offsets hoffs i _
  rw [show evalOp (Op.ref .nat [] "b") t
      = some (Tile.scalar (dtype := .nat) pid₀) from by simp [hb]] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact hreg

/-- Bounds discharge for the shared scalar-logit load: the one addressed
cell is the constant `0`, in bounds when the buffer is nonempty. -/
private theorem sharedScalar_activeAddressSafe (bounds : RegionBounds)
    (t : BlockState) (active : TileIndex [] → Prop)
    (reg : RegionName) (hreg : 0 + 1 ≤ bounds reg) :
    MemAccess.ActiveAddressSafe bounds
      (MemAccess.region reg (Op.constNat 0)) t active := by
  simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
  intro offsets hoffs i _
  rw [show evalOp (Op.constNat 0) t
      = some (Tile.scalar (dtype := .nat) 0) from by simp] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact hreg

set_option maxHeartbeats 1600000 in
/-- The fixed-rank depth kernel is trace-safe: of its seven statements, four
touch memory — three single-cell loads (`res_mix[pid]`, `branch_out[pid]`,
the shared `h_post[0]`) and one single-cell store (`out[pid]`); the
`program_id`, `exp`/`div`/`mul`, and `add` assigns are memory-silent. -/
theorem mhcDepth_traceSafe (tau : ℝ) (bounds : RegionBounds) (s : BlockState)
    (hmix : s.pid + 1 ≤ bounds ⟨"res_mix"⟩)
    (hbout : s.pid + 1 ≤ bounds ⟨"branch_out"⟩)
    (hhpost : 0 + 1 ≤ bounds ⟨"h_post"⟩)
    (hout : s.pid + 1 ≤ bounds ⟨"out"⟩) :
    Kernel.TraceSafe bounds
      ((mhcDepthConnectionKernel ⟨"res_mix"⟩ ⟨"branch_out"⟩ ⟨"h_post"⟩
        ⟨"out"⟩ 1 1 1 0 tau).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- statement 1: b := program_id(0)
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  rw [show evalOp (Op.programId 0) s
      = some (Tile.scalar (dtype := .nat) (s.pids 0)) from by simp] at hv1
  obtain rfl := Option.some_inj.mp hv1
  -- statement 2: res_mix := load(res_mix + b)   (single cell, unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [ComputeDType.eraseDType, Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial,
      scalarB_activeAddressSafe bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) _ hmix⟩
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  -- statement 3: branch_out := load(branch_out + b)   (single cell, unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [ComputeDType.eraseDType, Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial,
      scalarB_activeAddressSafe bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) _ hbout⟩
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmt_assign_inv hs3
  -- statement 4: h_post := load(h_post)   (shared scalar cell 0)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [ComputeDType.eraseDType, Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial, sharedScalar_activeAddressSafe bounds _ _ _ hhpost⟩
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmt_assign_inv hs4
  -- statement 5: branch_mix := exp(h_post / tau) * branch_out   (register op)
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmt_assign_inv hs5
  -- statement 6: out := res_mix + branch_mix   (register op)
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s6 hs6
  obtain ⟨v6, hv6, rfl⟩ := stepStmt_assign_inv hs6
  -- statement 7: store(out + b, out)   (single cell, unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt]
  refine ⟨by simp [Op.SafeAt], by simp [Op.SafeAt], trivial,
    scalarB_activeAddressSafe bounds (s.pids 0) _ _ ?_ _ hout⟩
  simp [BlockState.setReg]

/-- The fixed-rank depth kernel sits inside the bridge's covered fragment. -/
theorem mhcDepth_flattenOk (tau : ℝ) :
    ((mhcDepthConnectionKernel ⟨"res_mix"⟩ ⟨"branch_out"⟩ ⟨"h_post"⟩
      ⟨"out"⟩ 1 1 1 0 tau).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [mhcDepthConnectionKernel,
    ComputeKernel.toAlgKernel, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

theorem optimized_exec (tau : ℝ) (s : BlockState) :
    exec (optimizedKernel tau).toAlgKernel s =
      exec (mhcDepthConnectionKernel "res_mix" "branch_out" "h_post" "out" 1 1 1 0 tau).toAlgKernel s := by
  simp [optimizedKernel, mhcDepthConnectionKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Region.cast, Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
      NumericDType.div, add_comm]

end Optimized

end VeriTile.Bench.Examples.HyperConnectionsDepth

namespace VeriTile.Bench.Examples.HyperConnectionsDepth.MatrixCorrect
open VeriTile Triton LogSinkhorn
open VeriTile.Bench.Examples.HyperConnectionsDepth.Kernels

set_option maxHeartbeats 3200000

theorem prefix_run (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    ∃ a, stepStmts ((matrixOriginal S T D iters tau).body.take 14) s = some a ∧
      a.regs .real [T, S] "z" = some (scale (readMatrix s "h_post" 0 T S) tau) ∧
      a.regs .real [T] "u" = some (iterates (scale (readMatrix s "h_post" 0 T S) tau) 0).1 ∧
      a.regs .real [S] "v" = some (iterates (scale (readMatrix s "h_post" 0 T S) tau) 0).2 ∧
      a.regs .real [S, D] "res_mix" = some (readMatrix s "res_mix" (s.pid * (S * D)) S D) ∧
      a.regs .real [T, D] "branch_out" = some (readMatrix s "branch_out" (s.pid * (T * D)) T D) ∧
      a.regs .nat [] "b" = some (Tile.scalar s.pid) ∧
      a.regs .nat [S] "offs_s" = some (Tile.vec (fun i => i.val)) ∧
      a.regs .nat [D] "offs_d" = some (Tile.vec (fun i => i.val)) ∧
      a.mem = s.mem ∧ a.pids = s.pids := by
  simp [matrixOriginal, matrixKernel, ComputeKernel.body, ComputeKernel.toAlgKernel,
    List.take, stepStmts, stepStmt, evalOp.eq_def, BlockState.setReg,
    NumericDType.add, NumericDType.mul, NumericDType.div, readMatrix, scale, iterates,
    BlockState.readMem, Region.cast, Option.bind, Tile.bop, Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis]
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

open scoped VeriTile.Triton.KernelIO₃

theorem matrix_original_run (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    Spec.Real (∃ t, exec (matrixOriginal S T D iters tau) s = some t ∧
      (∀ i : TileIndex [S, D], t.readMem "out" (s.pid * (S * D) + i.1.val * D + i.2.1.val) =
        (((fun {S T D : Nat} (res : Tile .real [S, D]) (branch : Tile .real [T, D])
    (logits : Tile .real [T, S]) (tau : ℝ) (iters : Nat) =>
Tile.bop WithBot.realAdd (.consSame (.consSame .nil)) res
    (Tile.dot [] (Tile.transpose [] (weights (scale logits tau) iters)) branch)) (readMatrix s "res_mix" (s.pid * (S * D)) S D)
          (readMatrix s "branch_out" (s.pid * (T * D)) T D)
          (readMatrix s "h_post" 0 T S) tau iters).data i).unbotD 0) ∧
      (∀ (r : RegionName) o, (r ≠ "out" ∨ ∀ i : TileIndex [S, D],
        o ≠ s.pid * (S * D) + i.1.val * D + i.2.1.val) → t.mem r o = s.mem r o)) := by
  obtain ⟨a, ha, hz, hu, hv, hres, hbranch, hb, hs, hd, hm, hp⟩ := prefix_run S T D iters tau s
  obtain ⟨t, ht, hz', hu', hv', hmem, hpids, hr⟩ :=
    loop_run (scale (readMatrix s "h_post" 0 T S) tau) iters a hz hu hv
  have hres' := (hr .real [S, D] "res_mix" (by decide) (by decide) (by decide)).trans hres
  have hbranch' := (hr .real [T, D] "branch_out" (by decide) (by decide) (by decide)).trans hbranch
  have hb' := (hr .nat [] "b" (by decide) (by decide) (by decide)).trans hb
  have hs' := (hr .nat [S] "offs_s" (by decide) (by decide) (by decide)).trans hs
  have hd' := (hr .nat [D] "offs_d" (by decide) (by decide) (by decide)).trans hd
  have hbody : (matrixOriginal S T D iters tau).toAlgKernel.body =
      (matrixOriginal S T D iters tau).body.take 14 ++
        .forLoop "iter" iters (LogSinkhorn.body T S) :: (matrixOriginal S T D iters tau).body.drop 15 := rfl
  change ∃ u, stepStmts _ s = some u ∧ _
  rw [hbody, stepStmts.append_some ha, stepStmts.cons_some ht]
  simp [matrixOriginal, matrixKernel, ComputeKernel.body, ComputeKernel.toAlgKernel,
    List.drop, stepStmts, stepStmt, evalOp.eq_def, BlockState.setReg,
    hz', hu', hv', hres', hbranch', hb', hs', hd', Option.bind,
    NumericDType.add, NumericDType.mul, weights]
  refine ⟨?_, ?_⟩
  · intro i j
    rw [BlockState.scatter_readback_nd _ _ _ (address_injective _ S D) (i, j, PUnit.unit)]
    rfl
  · intro r o hmiss
    rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans
        (congrFun (congrFun (hmem.trans hm) r) o)
    · by_cases hro : r = "out"
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun i _ he => ho i.1 i.2.1 he.symm) _).trans (congrFun (congrFun (hmem.trans hm) _) o)
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hro o _).trans
          (congrFun (congrFun (hmem.trans hm) r) o)

theorem matrix_optimized_run (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    Spec.Real (∃ t, exec (matrixOptimized "res_mix" "branch_out" "h_post" "out" S T D iters tau) s = some t ∧
      (∀ i : TileIndex [S, D], t.readMem "out" (s.pid * (S * D) + i.1.val * D + i.2.1.val) =
        (((fun {S T D : Nat} (res : Tile .real [S, D]) (branch : Tile .real [T, D])
    (logits : Tile .real [T, S]) (tau : ℝ) (iters : Nat) =>
Tile.bop WithBot.realAdd (.consSame (.consSame .nil)) res
    (Tile.dot [] (Tile.transpose [] (weights (scale logits tau) iters)) branch)) (readMatrix s "res_mix" (s.pid * (S * D)) S D)
          (readMatrix s "branch_out" (s.pid * (T * D)) T D)
          (readMatrix s "h_post" 0 T S) tau iters).data i).unbotD 0) ∧
      (∀ (r : RegionName) o, (r ≠ "out" ∨ ∀ i : TileIndex [S, D],
        o ≠ s.pid * (S * D) + i.1.val * D + i.2.1.val) → t.mem r o = s.mem r o)) := by
  obtain ⟨t, ht, post⟩ := matrix_original_run S T D iters tau s
  refine ⟨t, ?_, post⟩
  exact MatrixRewrite.add_commute
    ((matrixOriginal S T D iters tau).body.take 17)
    ((matrixOriginal S T D iters tau).body.drop 18) "out" _ _ s t ht

/-- Transport the matrix output and memory frame to the flat windows used by
the public correctness specification. -/
theorem matrix_region_run (optimized : Bool) (S T D iters : Nat) (tau : ℝ)
    (s : BlockState) (xs : Fin (S * D) → ℝ) (ys : Fin (T * D) → ℝ)
    (zs : Fin (T * S) → ℝ)
    (hx : ∀ i, s.readMem "res_mix" (s.pid * (S * D) + i.val) = xs i)
    (hy : ∀ i, s.readMem "branch_out" (s.pid * (T * D) + i.val) = ys i)
    (hz : ∀ i, s.readMem "h_post" (0 + i.val) = zs i) :
    ∃ t, exec (if optimized then
        matrixOptimized "res_mix" "branch_out" "h_post" "out" S T D iters tau
      else matrixOriginal S T D iters tau) s = some t ∧
      (∀ i : Fin (S * D), t.readMem "out" (s.pid * (S * D) + i.val) =
        ((Tile.bop WithBot.realAdd (.consSame (.consSame .nil)) (matrixTile xs)
          (Tile.dot [] (Tile.transpose [] (weights (scale (matrixTile zs) tau) iters))
            (matrixTile ys))).data (matrixIndex i)).unbotD 0) ∧
      (∀ (r : RegionName) o,
        (r ≠ "out" ∨ ∀ i : Fin (S * D), o ≠ s.pid * (S * D) + i.val) →
        t.mem r o = s.mem r o) := by
  cases optimized
  ·
    obtain ⟨t, ht, hv, hf⟩ := matrix_original_run S T D iters tau s
    have hxs := readMatrix_eq_tile s "res_mix" (s.pid * (S * D)) S D xs hx
    have hys := readMatrix_eq_tile s "branch_out" (s.pid * (T * D)) T D ys hy
    have hzs := readMatrix_eq_tile s "h_post" 0 T S zs hz
    refine ⟨t, ht, ?_, ?_⟩
    · intro i
      simpa only [hxs, hys, hzs, Nat.add_assoc, matrixIndex_address] using hv (matrixIndex i)
    · intro r o h
      exact matrix_frame hf r o h
  ·
    obtain ⟨t, ht, hv, hf⟩ := matrix_optimized_run S T D iters tau s
    have hxs := readMatrix_eq_tile s "res_mix" (s.pid * (S * D)) S D xs hx
    have hys := readMatrix_eq_tile s "branch_out" (s.pid * (T * D)) T D ys hy
    have hzs := readMatrix_eq_tile s "h_post" 0 T S zs hz
    refine ⟨t, ht, ?_, ?_⟩
    · intro i
      simpa only [hxs, hys, hzs, Nat.add_assoc, matrixIndex_address] using hv (matrixIndex i)
    · intro r o h
      exact matrix_frame hf r o h

end VeriTile.Bench.Examples.HyperConnectionsDepth.MatrixCorrect
