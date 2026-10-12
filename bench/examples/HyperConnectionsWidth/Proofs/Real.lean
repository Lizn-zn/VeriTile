import bench.examples.HyperConnectionsWidth.Kernels
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

/-! Real execution, arithmetic and memory lemmas for HyperConnectionsWidth.
The public IO contracts and mathematical specifications are in ../Correct.lean. -/

namespace VeriTile.Bench.Examples.HyperConnectionsWidth.Memory
open VeriTile Triton TraceProofs Kernels

set_option maxHeartbeats 6400000

set_option linter.unusedSimpArgs false

theorem traceSafe (optimized : Bool) (S T D iters : Nat) (tau : ℝ)
    (bounds : RegionBounds) (s : BlockState)
    (hres : s.pid * (S * D) + S * D ≤ bounds "res")
    (hreslog : S * S ≤ bounds "h_res")
    (hprelog : S * T ≤ bounds "h_pre")
    (hmix : s.pid * (S * D) + S * D ≤ bounds "res_mix")
    (hbranch : s.pid * (T * D) + T * D ≤ bounds "branch_in") :
    Kernel.TraceSafe bounds
      (if optimized then (matrixOptimized "res" "h_res" "h_pre" "res_mix" "branch_in" S T D iters tau).toAlgKernel
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
    apply assign_cons
    · simp only [Op.SafeAt.eq_def, and_true, true_and]
      apply region_safe (offsets := ⟨fun i : TileIndex [S, D] => (s.pid * (S * D)) + i.1.val * D + i.2.1.val⟩)
      · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
          Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
          NumericDType.add, NumericDType.mul]
      · intro i
        exact lt_of_lt_of_le (matrix_offset_lt (s.pid * (S * D)) S D i.1 i.2.1) hres
    intro res
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_cons
    · simp only [Op.SafeAt.eq_def, and_true, true_and]
      apply region_safe (offsets := ⟨fun i : TileIndex [S, S] => 0 + i.1.val * S + i.2.1.val⟩)
      · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
          Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
          NumericDType.add, NumericDType.mul]
      · intro i
        exact lt_of_lt_of_le (matrix_offset_lt 0 S S i.1 i.2.1) (by simpa using hreslog)
    intro reslog
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro z
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro u
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro v
    refine Stmt.TraceSafeList.cons_intro (LogSinkhorn.loop_traceSafe S S iters bounds _) ?_
    intro q hq
    have hb := LogSinkhorn.loop_register S S iters _ q hq .nat [] "b" (by decide) (by decide) (by decide)
    have hs := LogSinkhorn.loop_register S S iters _ q hq .nat [S] "offs_s" (by decide) (by decide) (by decide)
    have ht := LogSinkhorn.loop_register S S iters _ q hq .nat [T] "offs_t" (by decide) (by decide) (by decide)
    have hd := LogSinkhorn.loop_register S S iters _ q hq .nat [D] "offs_d" (by decide) (by decide) (by decide)
    simp only [BlockState.setReg] at hb hs ht hd
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro weights
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro mix
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul, hb, hs, ht, hd] at hval
    subst val
    apply assign_cons
    · simp only [Op.SafeAt.eq_def, and_true, true_and]
      apply region_safe (offsets := ⟨fun i : TileIndex [S, T] => 0 + i.1.val * T + i.2.1.val⟩)
      · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
          Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
          NumericDType.add, NumericDType.mul, hb, hs, ht, hd]
      · intro i
        exact lt_of_lt_of_le (matrix_offset_lt 0 S T i.1 i.2.1) (by simpa using hprelog)
    intro prelog
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro z2
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro u2
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro v2
    refine Stmt.TraceSafeList.cons_intro (LogSinkhorn.loop_traceSafe S T iters bounds _) ?_
    intro r hr
    have hb2 := LogSinkhorn.loop_register S T iters _ r hr .nat [] "b" (by decide) (by decide) (by decide)
    have hs2 := LogSinkhorn.loop_register S T iters _ r hr .nat [S] "offs_s" (by decide) (by decide) (by decide)
    have ht2 := LogSinkhorn.loop_register S T iters _ r hr .nat [T] "offs_t" (by decide) (by decide) (by decide)
    have hd2 := LogSinkhorn.loop_register S T iters _ r hr .nat [D] "offs_d" (by decide) (by decide) (by decide)
    simp only [BlockState.setReg] at hb2 hs2 ht2 hd2
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro weights2
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro branch
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul, hb, hs, ht, hd, hb2, hs2, ht2, hd2] at hval
    subst val
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul, hb, hs, ht, hd, hb2, hs2, ht2, hd2] at hval
    subst val
    refine Stmt.TraceSafeList.cons_intro ?_ ?_
    · simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt, Op.SafeAt.eq_def, and_true, true_and]
      apply region_safe (offsets := ⟨fun i : TileIndex [S, D] => (s.pid * (S * D)) + i.1.val * D + i.2.1.val⟩)
      · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
          Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
          NumericDType.add, NumericDType.mul, hb, hs, ht, hd, hb2, hs2, ht2, hd2]
      · intro i
        exact lt_of_lt_of_le (matrix_offset_lt (s.pid * (S * D)) S D i.1 i.2.1) hmix
    intro t hstore
    have hregs := store_regs _ _ _ _ _ hstore
    refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
    simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt, Op.SafeAt.eq_def, and_true, true_and]
    apply region_safe (offsets := ⟨fun i : TileIndex [T, D] => (s.pid * (T * D)) + i.1.val * D + i.2.1.val⟩)
    · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul, hb, hs, ht, hd, hb2, hs2, ht2, hd2, hregs]
    · intro i
      exact lt_of_lt_of_le (matrix_offset_lt (s.pid * (T * D)) T D i.1 i.2.1) hbranch

/-- Both sources remain in the fragment supported by address relocation. -/
theorem flattenOk (optimized : Bool) (S T D iters : Nat) (tau : ℝ) :
    (if optimized then (matrixOptimized "res" "h_res" "h_pre" "res_mix" "branch_in" S T D iters tau).toAlgKernel
     else (matrixOriginal S T D iters tau).toAlgKernel).FlattenOk := by
  cases optimized <;> simp [Kernel.FlattenOk, matrixOriginal, matrixKernel, matrixOptimized, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

end VeriTile.Bench.Examples.HyperConnectionsWidth.Memory

namespace VeriTile.Bench.Examples.HyperConnectionsWidth
open VeriTile.Bench.Examples.HyperConnectionsWidth.Kernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?
  VeriTile.Triton.ComputeOp.toAlgorithm? VeriTile.Triton.ComputeDType.eraseDType

open VeriTile.Triton
open VeriTile.Triton.KernelIO₃ₓ₂ (Implements)
open scoped VeriTile.Triton.KernelIO₃ₓ₂

/-- Value half: the run terminates, `res_mix[pid]` holds `exp(hRes/τ)·res`
and `branch_in[pid]` holds `exp(hPre/τ)·res` (in terms of the launch state's
own memory cells). -/
private theorem mhcWidth_exec_view (tau : ℝ) (s : BlockState) :
    (exec ((mhcWidthConnectionKernel ⟨"res"⟩ ⟨"h_res"⟩ ⟨"h_pre"⟩ ⟨"res_mix"⟩
        ⟨"branch_in"⟩ 1 1 1 0 tau).toAlgKernel) s).map
      (fun s' => (s'.readMem ⟨"res_mix"⟩ s.pid, s'.readMem ⟨"branch_in"⟩ s.pid))
      = some
        (Real.exp (s.readMem ⟨"h_res"⟩ 0 / tau) * s.readMem ⟨"res"⟩ s.pid,
         Real.exp (s.readMem ⟨"h_pre"⟩ 0 / tau) * s.readMem ⟨"res"⟩ s.pid) := by
  simp [mhcWidthConnectionKernel, exec, stepStmts,
    stepStmt, NumericDType.mul, NumericDType.div,
    WithBot.realMul, WithBot.realDiv, BlockState.writeMem, BlockState.readMem]

/-- Frame half: every memory cell other than the two output cells
`res_mix[pid]` / `branch_in[pid]` is preserved by the run. -/
private theorem mhcWidth_frame (tau : ℝ) (s s1 : BlockState)
    (hExec : exec ((mhcWidthConnectionKernel ⟨"res"⟩ ⟨"h_res"⟩ ⟨"h_pre"⟩
        ⟨"res_mix"⟩ ⟨"branch_in"⟩ 1 1 1 0 tau).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (h1 : ¬(r = ⟨"res_mix"⟩ ∧ o = s.pid))
    (h2 : ¬(r = ⟨"branch_in"⟩ ∧ o = s.pid)) :
    s1.mem r o = s.mem r o := by
  simp [exec, mhcWidthConnectionKernel,
    ComputeKernel.toAlgKernel, stepStmts, stepStmt,
    NumericDType.mul, NumericDType.div] at hExec
  subst hExec
  rw [BlockState.writeMem_mem, if_neg h2, BlockState.writeMem_mem, if_neg h1]
  rfl

/-- **The region-model Hoare triple** — termination, both output-cell values,
and the two-window frame, from any launch state whose input windows are
loaded. This is what the `⊨` headline transports to flat memory. -/
theorem mhcWidth_region_run (tau : ℝ) (s₀ : BlockState)
    (res hRes hPre : Fin 1 → ℝ)
    (hres : ∀ j : Fin 1, s₀.readMem ⟨"res"⟩ (s₀.pid + j.val) = res j)
    (hhres : ∀ j : Fin 1, s₀.readMem ⟨"h_res"⟩ (0 + j.val) = hRes j)
    (hhpre : ∀ j : Fin 1, s₀.readMem ⟨"h_pre"⟩ (0 + j.val) = hPre j) :
    ∃ s1,
      exec ((mhcWidthConnectionKernel ⟨"res"⟩ ⟨"h_res"⟩ ⟨"h_pre"⟩ ⟨"res_mix"⟩
          ⟨"branch_in"⟩ 1 1 1 0 tau).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin 1, s1.readMem ⟨"res_mix"⟩ (s₀.pid + j.val)
          = Real.exp (hRes 0 / tau) * res 0)
      ∧ (∀ j : Fin 1, s1.readMem ⟨"branch_in"⟩ (s₀.pid + j.val)
          = Real.exp (hPre 0 / tau) * res 0)
      ∧ (∀ r o,
          (r ≠ ⟨"res_mix"⟩ ∨ ∀ j : Fin 1, o ≠ s₀.pid + j.val) →
          (r ≠ ⟨"branch_in"⟩ ∨ ∀ j : Fin 1, o ≠ s₀.pid + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have h0res : s₀.readMem ⟨"res"⟩ s₀.pid = res 0 := by simpa using hres 0
  have h0hres : s₀.readMem ⟨"h_res"⟩ 0 = hRes 0 := by simpa using hhres 0
  have h0hpre : s₀.readMem ⟨"h_pre"⟩ 0 = hPre 0 := by simpa using hhpre 0
  have hview := mhcWidth_exec_view tau s₀
  cases hsrc : exec ((mhcWidthConnectionKernel ⟨"res"⟩ ⟨"h_res"⟩ ⟨"h_pre"⟩
      ⟨"res_mix"⟩ ⟨"branch_in"⟩ 1 1 1 0 tau).toAlgKernel) s₀ with
  | none => rw [hsrc] at hview; simp at hview
  | some s1 =>
      rw [hsrc] at hview
      simp only [Option.map_some, Option.some_inj, Prod.mk.injEq] at hview
      obtain ⟨hv1, hv2⟩ := hview
      refine ⟨s1, rfl, fun j => ?_, fun j => ?_, fun r o hc1 hc2 => ?_⟩
      · have hj : (j : Nat) = 0 := Nat.lt_one_iff.mp j.isLt
        rw [hj, Nat.add_zero, hv1, h0hres, h0res]
      · have hj : (j : Nat) = 0 := Nat.lt_one_iff.mp j.isLt
        rw [hj, Nat.add_zero, hv2, h0hpre, h0res]
      · refine mhcWidth_frame tau s₀ s1 hsrc r o ?_ ?_
        · rintro ⟨hr, ho⟩
          rcases hc1 with hne | hno
          · exact hne hr
          · exact hno 0 (by simpa using ho)
        · rintro ⟨hr, ho⟩
          rcases hc2 with hne | hno
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

/-- Bounds discharge for the shared scalar-logit loads: the one addressed
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
/-- The fixed-rank width kernel is trace-safe: of its eight statements, five
touch memory — three single-cell loads (`res[pid]`, the shared `h_res[0]` /
`h_pre[0]`) and two single-cell stores (`res_mix[pid]`, `branch_in[pid]`);
the `program_id` assign and the two `exp`/`div`/`mul` assigns are
memory-silent. -/
theorem mhcWidth_traceSafe (tau : ℝ) (bounds : RegionBounds) (s : BlockState)
    (hres : s.pid + 1 ≤ bounds ⟨"res"⟩)
    (hhres : 0 + 1 ≤ bounds ⟨"h_res"⟩)
    (hhpre : 0 + 1 ≤ bounds ⟨"h_pre"⟩)
    (hmix : s.pid + 1 ≤ bounds ⟨"res_mix"⟩)
    (hbin : s.pid + 1 ≤ bounds ⟨"branch_in"⟩) :
    Kernel.TraceSafe bounds
      ((mhcWidthConnectionKernel ⟨"res"⟩ ⟨"h_res"⟩ ⟨"h_pre"⟩ ⟨"res_mix"⟩
        ⟨"branch_in"⟩ 1 1 1 0 tau).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- statement 1: b := program_id(0)
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  rw [show evalOp (Op.programId 0) s
      = some (Tile.scalar (dtype := .nat) (s.pids 0)) from by simp] at hv1
  obtain rfl := Option.some_inj.mp hv1
  -- statement 2: residual := load(res + b)   (single cell, unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [ComputeDType.eraseDType, Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial,
      scalarB_activeAddressSafe bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) _ hres⟩
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  -- statement 3: h_res := load(h_res)   (shared scalar cell 0)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [ComputeDType.eraseDType, Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial, sharedScalar_activeAddressSafe bounds _ _ _ hhres⟩
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmt_assign_inv hs3
  -- statement 4: res_mix := exp(h_res / tau) * residual   (register op)
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmt_assign_inv hs4
  -- statement 5: h_pre := load(h_pre)   (shared scalar cell 0)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [ComputeDType.eraseDType, Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial, sharedScalar_activeAddressSafe bounds _ _ _ hhpre⟩
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmt_assign_inv hs5
  -- statement 6: branch_in := exp(h_pre / tau) * residual   (register op)
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s6 hs6
  obtain ⟨v6, hv6, rfl⟩ := stepStmt_assign_inv hs6
  -- statement 7: store(res_mix + b, res_mix)   (single cell, unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt]
    refine ⟨by simp [Op.SafeAt], by simp [Op.SafeAt], trivial,
      scalarB_activeAddressSafe bounds (s.pids 0) _ _ ?_ _ hmix⟩
    simp [BlockState.setReg]
  intro s7 hs7
  simp [stepStmt, BlockState.setReg, TileShape.allIndices] at hs7
  subst hs7
  -- statement 8: store(branch_in + b, branch_in)   (single cell, unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt]
  refine ⟨by simp [Op.SafeAt], by simp [Op.SafeAt], trivial,
    scalarB_activeAddressSafe bounds (s.pids 0) _ _ ?_ _ hbin⟩
  simp

/-- The fixed-rank width kernel sits inside the bridge's covered fragment. -/
theorem mhcWidth_flattenOk (tau : ℝ) :
    ((mhcWidthConnectionKernel ⟨"res"⟩ ⟨"h_res"⟩ ⟨"h_pre"⟩ ⟨"res_mix"⟩
      ⟨"branch_in"⟩ 1 1 1 0 tau).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [mhcWidthConnectionKernel,
    ComputeKernel.toAlgKernel, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

theorem optimized_exec (tau : ℝ) (s : BlockState) :
    exec (optimizedKernel tau).toAlgKernel s =
      exec (mhcWidthConnectionKernel "res" "h_res" "h_pre" "res_mix" "branch_in" 1 1 1 0 tau).toAlgKernel s := by
  simp [optimizedKernel, mhcWidthConnectionKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Region.cast, Tile.bop, Tile.uop, NumericDType.mul, NumericDType.div, mul_comm]

end Optimized

end VeriTile.Bench.Examples.HyperConnectionsWidth

namespace VeriTile.Bench.Examples.HyperConnectionsWidth.MatrixCorrect
open VeriTile Triton LogSinkhorn
open VeriTile.Bench.Examples.HyperConnectionsWidth.Kernels

set_option maxHeartbeats 6400000

set_option linter.unusedSimpArgs false

theorem prefix_run (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    ∃ a, stepStmts ((matrixOriginal S T D iters tau).body.take 12) s = some a ∧
      a.regs .real [S, S] "z" = some (scale (readMatrix s "h_res" 0 S S) tau) ∧
      a.regs .real [S] "u" = some (iterates (scale (readMatrix s "h_res" 0 S S) tau) 0).1 ∧
      a.regs .real [S] "v" = some (iterates (scale (readMatrix s "h_res" 0 S S) tau) 0).2 ∧
      a.regs .real [S, D] "residuals" = some (readMatrix s "res" (s.pid * (S * D)) S D) ∧
      a.regs .nat [] "b" = some (Tile.scalar s.pid) ∧
      a.regs .nat [S] "offs_s" = some (Tile.vec (fun i => i.val)) ∧
      a.regs .nat [T] "offs_t" = some (Tile.vec (fun i => i.val)) ∧
      a.regs .nat [D] "offs_d" = some (Tile.vec (fun i => i.val)) ∧
      a.mem = s.mem ∧ a.pids = s.pids := by
  simp [matrixOriginal, matrixKernel, ComputeKernel.body, ComputeKernel.toAlgKernel,
    List.take, stepStmts, stepStmt, evalOp.eq_def, BlockState.setReg,
    NumericDType.add, NumericDType.mul, NumericDType.div, readMatrix, scale, iterates,
    BlockState.readMem, Region.cast, Option.bind, Tile.bop, Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis]
  repeat' first | exact rfl | apply And.intro

open scoped VeriTile.Triton.KernelIO₃ₓ₂

theorem matrix_original_run (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    Spec.Real (∃ t, exec (matrixOriginal S T D iters tau) s = some t ∧
      (∀ i : TileIndex [S, D], t.readMem "res_mix" (s.pid * (S * D) + i.1.val * D + i.2.1.val) =
        (((fun {S D : Nat} (res : Tile .real [S, D]) (logits : Tile .real [S, S])
    (tau : ℝ) (iters : Nat) =>
Tile.dot [] (weights (scale logits tau) iters) res) (readMatrix s "res" (s.pid * (S * D)) S D)
          (readMatrix s "h_res" 0 S S) tau iters).data i).unbotD 0) ∧
      (∀ i : TileIndex [T, D], t.readMem "branch_in" (s.pid * (T * D) + i.1.val * D + i.2.1.val) =
        (((fun {S T D : Nat} (res : Tile .real [S, D]) (logits : Tile .real [S, T])
    (tau : ℝ) (iters : Nat) =>
Tile.dot [] (Tile.transpose [] (weights (scale logits tau) iters)) res) (readMatrix s "res" (s.pid * (S * D)) S D)
          (readMatrix s "h_pre" 0 S T) tau iters).data i).unbotD 0) ∧
      (∀ (r : RegionName) o,
        (r ≠ "res_mix" ∨ ∀ i : TileIndex [S, D], o ≠ s.pid * (S * D) + i.1.val * D + i.2.1.val) →
        (r ≠ "branch_in" ∨ ∀ i : TileIndex [T, D], o ≠ s.pid * (T * D) + i.1.val * D + i.2.1.val) →
        t.mem r o = s.mem r o)) := by
  obtain ⟨a, ha, hz, hu, hv, hres, hb, hs, ht, hd, hm, hp⟩ := prefix_run S T D iters tau s
  obtain ⟨t, hloop, hz', hu', hv', hmem, hpids, hr⟩ :=
    loop_run (scale (readMatrix s "h_res" 0 S S) tau) iters a hz hu hv
  have hres' := (hr .real [S, D] "residuals" (by decide) (by decide) (by decide)).trans hres
  have hb' := (hr .nat [] "b" (by decide) (by decide) (by decide)).trans hb
  have hs' := (hr .nat [S] "offs_s" (by decide) (by decide) (by decide)).trans hs
  have ht' := (hr .nat [T] "offs_t" (by decide) (by decide) (by decide)).trans ht
  have hd' := (hr .nat [D] "offs_d" (by decide) (by decide) (by decide)).trans hd
  have hmem' : t.mem = s.mem := hmem.trans hm
  have hmid : ∃ q, stepStmts (((matrixOriginal S T D iters tau).body.drop 13).take 7) t = some q ∧
      q.regs .real [S, T] "z" = some (scale (readMatrix s "h_pre" 0 S T) tau) ∧
      q.regs .real [S] "u" = some (iterates (scale (readMatrix s "h_pre" 0 S T) tau) 0).1 ∧
      q.regs .real [T] "v" = some (iterates (scale (readMatrix s "h_pre" 0 S T) tau) 0).2 ∧
      q.regs .real [S, D] "res_mix" = some ((fun {S D : Nat} (res : Tile .real [S, D]) (logits : Tile .real [S, S])
    (tau : ℝ) (iters : Nat) =>
Tile.dot [] (weights (scale logits tau) iters) res)
        (readMatrix s "res" (s.pid * (S * D)) S D) (readMatrix s "h_res" 0 S S) tau iters) ∧
      q.regs .real [S, D] "residuals" = some (readMatrix s "res" (s.pid * (S * D)) S D) ∧
      q.regs .nat [] "b" = some (Tile.scalar s.pid) ∧
      q.regs .nat [S] "offs_s" = some (Tile.vec (fun i => i.val)) ∧
      q.regs .nat [T] "offs_t" = some (Tile.vec (fun i => i.val)) ∧
      q.regs .nat [D] "offs_d" = some (Tile.vec (fun i => i.val)) ∧ q.mem = s.mem := by
    simp [matrixOriginal, matrixKernel, ComputeKernel.body, ComputeKernel.toAlgKernel,
      List.drop, List.take, stepStmts, stepStmt, evalOp.eq_def, BlockState.setReg,
      hz', hu', hv', hres', hb', hs', ht', hd', hmem', Option.bind,
      NumericDType.add, NumericDType.mul, NumericDType.div, weights, scale, readMatrix, iterates,
      BlockState.readMem, Region.cast, Tile.bop, Tile.uop, Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis]
    repeat' first | exact rfl | apply And.intro
  obtain ⟨q, hq, hqz, hqu, hqv, hmix, hqr, hqb, hqs, hqt, hqd, hqm⟩ := hmid
  obtain ⟨u, hloop2, huz, huu, huv, hum, _, hur⟩ :=
    loop_run (scale (readMatrix s "h_pre" 0 S T) tau) iters q hqz hqu hqv
  have hmix' := (hur .real [S, D] "res_mix" (by decide) (by decide) (by decide)).trans hmix
  have hur' := (hur .real [S, D] "residuals" (by decide) (by decide) (by decide)).trans hqr
  have hub := (hur .nat [] "b" (by decide) (by decide) (by decide)).trans hqb
  have hus := (hur .nat [S] "offs_s" (by decide) (by decide) (by decide)).trans hqs
  have hut := (hur .nat [T] "offs_t" (by decide) (by decide) (by decide)).trans hqt
  have hud := (hur .nat [D] "offs_d" (by decide) (by decide) (by decide)).trans hqd
  have hbody : (matrixOriginal S T D iters tau).toAlgKernel.body =
      (matrixOriginal S T D iters tau).body.take 12 ++
        [.forLoop "iter" iters (LogSinkhorn.body S S)] ++
        ((matrixOriginal S T D iters tau).body.drop 13).take 7 ++
        .forLoop "iter" iters (LogSinkhorn.body S T) :: (matrixOriginal S T D iters tau).body.drop 21 := rfl
  change ∃ v, stepStmts _ s = some v ∧ _
  rw [hbody, List.append_assoc, List.append_assoc, stepStmts.append_some ha]
  change ∃ v, stepStmts (.forLoop "iter" iters (LogSinkhorn.body S S) :: _) a = some v ∧ _
  rw [stepStmts.cons_some hloop]
  change ∃ v, stepStmts (((matrixOriginal S T D iters tau).body.drop 13).take 7 ++
    .forLoop "iter" iters (LogSinkhorn.body S T) :: (matrixOriginal S T D iters tau).body.drop 21) t = some v ∧ _
  rw [stepStmts.append_some hq, stepStmts.cons_some hloop2]
  simp [matrixOriginal, matrixKernel, ComputeKernel.body, ComputeKernel.toAlgKernel,
    List.drop, stepStmts, stepStmt, evalOp.eq_def, BlockState.setReg,
    huz, huu, huv, hmix', hur', hub, hus, hut, hud, Option.bind,
    NumericDType.add, NumericDType.mul, weights]
  refine ⟨?_, ?_, ?_⟩
  · intro i j
    rw [scatter_read_other _ _ _ _ _ _ _ (by decide)]
    rw [BlockState.scatter_readback_nd _ _ _ (address_injective _ S D) (i, j, PUnit.unit)]
    rintro ⟨⟩
    rfl
  · intro i j
    rw [BlockState.scatter_readback_nd _ _ _ (address_injective _ T D) (i, j, PUnit.unit)]
    rintro ⟨⟩
    rfl
  · intro r o hR hB
    rw [scatter_frame _ _ _ _ _ r o (by simpa using hB), scatter_frame _ _ _ _ _ r o (by simpa using hR)]
    exact congrFun (congrFun (hum.trans hqm) r) o

theorem matrix_optimized_run (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    Spec.Real (∃ t, exec (matrixOptimized "res" "h_res" "h_pre" "res_mix" "branch_in" S T D iters tau) s = some t ∧
      (∀ i : TileIndex [S, D], t.readMem "res_mix" (s.pid * (S * D) + i.1.val * D + i.2.1.val) =
        (((fun {S D : Nat} (res : Tile .real [S, D]) (logits : Tile .real [S, S])
    (tau : ℝ) (iters : Nat) =>
Tile.dot [] (weights (scale logits tau) iters) res) (readMatrix s "res" (s.pid * (S * D)) S D)
          (readMatrix s "h_res" 0 S S) tau iters).data i).unbotD 0) ∧
      (∀ i : TileIndex [T, D], t.readMem "branch_in" (s.pid * (T * D) + i.1.val * D + i.2.1.val) =
        (((fun {S T D : Nat} (res : Tile .real [S, D]) (logits : Tile .real [S, T])
    (tau : ℝ) (iters : Nat) =>
Tile.dot [] (Tile.transpose [] (weights (scale logits tau) iters)) res) (readMatrix s "res" (s.pid * (S * D)) S D)
          (readMatrix s "h_pre" 0 S T) tau iters).data i).unbotD 0) ∧
      (∀ (r : RegionName) o,
        (r ≠ "res_mix" ∨ ∀ i : TileIndex [S, D], o ≠ s.pid * (S * D) + i.1.val * D + i.2.1.val) →
        (r ≠ "branch_in" ∨ ∀ i : TileIndex [T, D], o ≠ s.pid * (T * D) + i.1.val * D + i.2.1.val) →
        t.mem r o = s.mem r o)) := by
  obtain ⟨t, ht, post⟩ := matrix_original_run S T D iters tau s
  refine ⟨t, ?_, post⟩
  have hmid : exec (matrixMiddle "res" "h_res" "h_pre" "res_mix" "branch_in" S T D iters tau) s = some t :=
    MatrixRewrite.div_mul_rcp ((matrixOriginal S T D iters tau).body.take 9)
      ((matrixOriginal S T D iters tau).body.drop 10) "z" _ tau s t ht
  exact MatrixRewrite.div_mul_rcp
    ((matrixMiddle "res" "h_res" "h_pre" "res_mix" "branch_in" S T D iters tau).body.take 17)
    ((matrixMiddle "res" "h_res" "h_pre" "res_mix" "branch_in" S T D iters tau).body.drop 18)
    "z" _ tau s t hmid

/-- Transport both matrix outputs and their shared memory frame to the flat
input/output windows used by the public correctness specification. -/
theorem matrix_region_run (optimized : Bool) (S T D iters : Nat) (tau : ℝ)
    (s : BlockState) (xs : Fin (S * D) → ℝ) (ys : Fin (S * S) → ℝ)
    (zs : Fin (S * T) → ℝ)
    (hx : ∀ i, s.readMem "res" (s.pid * (S * D) + i.val) = xs i)
    (hy : ∀ i, s.readMem "h_res" (0 + i.val) = ys i)
    (hz : ∀ i, s.readMem "h_pre" (0 + i.val) = zs i) :
    ∃ t, exec (if optimized then
        matrixOptimized "res" "h_res" "h_pre" "res_mix" "branch_in" S T D iters tau
      else matrixOriginal S T D iters tau) s = some t ∧
      (∀ i : Fin (S * D), t.readMem "res_mix" (s.pid * (S * D) + i.val) =
        ((Tile.dot [] (weights (scale (matrixTile ys) tau) iters)
          (matrixTile xs)).data (matrixIndex i)).unbotD 0) ∧
      (∀ i : Fin (T * D), t.readMem "branch_in" (s.pid * (T * D) + i.val) =
        ((Tile.dot [] (Tile.transpose [] (weights (scale (matrixTile zs) tau) iters))
          (matrixTile xs)).data (matrixIndex i)).unbotD 0) ∧
      (∀ (r : RegionName) o,
        (r ≠ "res_mix" ∨ ∀ i : Fin (S * D), o ≠ s.pid * (S * D) + i.val) →
        (r ≠ "branch_in" ∨ ∀ i : Fin (T * D), o ≠ s.pid * (T * D) + i.val) →
        t.mem r o = s.mem r o) := by
  cases optimized
  ·
    obtain ⟨t, ht, hv1, hv2, hf⟩ := matrix_original_run S T D iters tau s
    have hxs := readMatrix_eq_tile s "res" (s.pid * (S * D)) S D xs hx
    have hys := readMatrix_eq_tile s "h_res" 0 S S ys hy
    have hzs := readMatrix_eq_tile s "h_pre" 0 S T zs hz
    refine ⟨t, ht, ?_, ?_, ?_⟩
    · intro i
      simpa only [hxs, hys, Nat.add_assoc, matrixIndex_address] using hv1 (matrixIndex i)
    · intro i
      simpa only [hxs, hzs, Nat.add_assoc, matrixIndex_address] using hv2 (matrixIndex i)
    · intro r o h1 h2
      apply hf r o
      · rcases h1 with h | h
        · exact Or.inl h
        · exact Or.inr (fun i => by simpa [finProdFinEquiv, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm, Nat.mul_comm] using (h (finProdFinEquiv (i.1, i.2.1))))
      · rcases h2 with h | h
        · exact Or.inl h
        · exact Or.inr (fun i => by simpa [finProdFinEquiv, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm, Nat.mul_comm] using (h (finProdFinEquiv (i.1, i.2.1))))
  ·
    obtain ⟨t, ht, hv1, hv2, hf⟩ := matrix_optimized_run S T D iters tau s
    have hxs := readMatrix_eq_tile s "res" (s.pid * (S * D)) S D xs hx
    have hys := readMatrix_eq_tile s "h_res" 0 S S ys hy
    have hzs := readMatrix_eq_tile s "h_pre" 0 S T zs hz
    refine ⟨t, ht, ?_, ?_, ?_⟩
    · intro i
      simpa only [hxs, hys, Nat.add_assoc, matrixIndex_address] using hv1 (matrixIndex i)
    · intro i
      simpa only [hxs, hzs, Nat.add_assoc, matrixIndex_address] using hv2 (matrixIndex i)
    · intro r o h1 h2
      apply hf r o
      · rcases h1 with h | h
        · exact Or.inl h
        · exact Or.inr (fun i => by simpa [finProdFinEquiv, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm, Nat.mul_comm] using (h (finProdFinEquiv (i.1, i.2.1))))
      · rcases h2 with h | h
        · exact Or.inl h
        · exact Or.inr (fun i => by simpa [finProdFinEquiv, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm, Nat.mul_comm] using (h (finProdFinEquiv (i.1, i.2.1))))

end VeriTile.Bench.Examples.HyperConnectionsWidth.MatrixCorrect
