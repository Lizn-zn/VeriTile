import bench.tritonbench_g.cosine_compute.Kernels
import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.CosineCompute.RealProof
open VeriTile.Triton VeriTile.Examples
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

/-- Executable real projection: precision erasure removes the conversions. -/
def realKernel (swapped : Bool) (a b : RegionName) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  if swapped then ComputeKernel.mk [a] [b] (triton {
    offset = $(BLOCK_SIZE) * tl.program_id(0) + tl.arange(0, $(BLOCK_SIZE))
    mask = offset < $(n_elements)
    a_value = tl.load(a + offset, mask=mask)
    b_value = tl.cos(a_value)
    tl.store(b + offset, b_value, mask=mask)
  }).surfaceBody else ComputeKernel.mk [a] [b] (triton {
    offset = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
    mask = offset < $(n_elements)
    a_value = tl.load(a + offset, mask=mask)
    b_value = tl.cos(a_value)
    tl.store(b + offset, b_value, mask=mask)
  }).surfaceBody

theorem source_projects (storage : Storage) (swapped : Bool) (a b : RegionName) (n B : Nat) :
    (kernel storage swapped a b n B).toAlgorithm? = .ok (kernel storage swapped a b n B).toAlgKernel := by
  cases storage <;> cases swapped <;> rfl

theorem projection (storage : Storage) (swapped : Bool) (a b : RegionName) (n B : Nat) :
    (kernel storage swapped a b n B).eraseDType.toAlgorithm? =
      .ok (kernel storage swapped a b n B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType,source_projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

/-- Every storage branch erases to this same source projection. -/
theorem projects (storage : Storage) (swapped : Bool) (a b : RegionName) (n B : Nat) :
    (kernel storage swapped a b n B).eraseDType.toAlgKernel =
      (realKernel swapped a b n B).toAlgKernel := by
  cases storage <;> cases swapped
  all_goals simp only [kernel,original,indexCommuted,realKernel,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind,eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType]
  all_goals rfl

theorem cos_func_correct
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s a BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (realKernel Bool.false a b n_elements BLOCK_SIZE) s)
                b BLOCK_SIZE s.pid i
        = some (if addr < n_elements then Real.cos (xs i)
                else s.readMem b addr) := by
  intro i
  have h_inj := injective_offset_singleton (n := BLOCK_SIZE) (s.pid * BLOCK_SIZE)
  simp [observeAt, exec, realKernel, ComputeKernel.surfaceBody, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.uop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, WithBot.realCos,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x]
  · simp [hi]

set_option maxHeartbeats 1600000 in

theorem cos_func_frame
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat) (s s1 : BlockState)
    (hExec : exec ((realKernel Bool.false a b n_elements BLOCK_SIZE).toAlgKernel) s
      = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + i.val < n_elements →
      ¬(b = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, realKernel, ComputeKernel.surfaceBody, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.uop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, WithBot.realCos,
    FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

theorem cos_func_region_run
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s₀ : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem a (s₀.pid * BLOCK_SIZE + j.val) = xs j) :
    ∃ s1, exec ((realKernel Bool.false a b n_elements BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
          s1.readMem b (s₀.pid * BLOCK_SIZE + j.val) = Real.cos (xs j))
      ∧ (∀ r o,
          (r ≠ b ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < n_elements →
              o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := cos_func_correct a b n_elements BLOCK_SIZE
    s₀ (fun j => s₀.readMem a (s₀.pid * BLOCK_SIZE + j.val)) (fun _ => rfl)
  rw [show exec (realKernel Bool.false a b n_elements BLOCK_SIZE) s₀
      = exec ((realKernel Bool.false a b n_elements BLOCK_SIZE).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((realKernel Bool.false a b n_elements BLOCK_SIZE).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, realKernel, ComputeKernel.surfaceBody, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, Tile.bop, Tile.uop, Tile.cop, NumericDType.add,
          NumericDType.mul, ComparableDType.lt, WithBot.realCos,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [observeAt, Option.map_some, Option.some_inj, if_pos hj]
          at hje
        rw [hje, hx j hj]
      · refine cos_func_frame a b n_elements BLOCK_SIZE
          s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in

theorem cos_func_traceSafe
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds a)
    (hout : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds b) :
    Kernel.TraceSafe bounds
      ((realKernel Bool.false a b n_elements BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe

  simp [realKernel, ComputeKernel.surfaceBody, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt, WithBot.realCos,
    FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, fun a ha => hout a ha⟩

theorem cos_func_flattenOk
    (a b : RegionName) (n_elements BLOCK_SIZE : Nat) :
    ((realKernel Bool.false a b n_elements BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [realKernel, ComputeKernel.surfaceBody, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]


/-- Exact integer arithmetic makes the two real executions coincide. -/
theorem commuted_exec (a b : RegionName) (n B : Nat) (s : BlockState) :
    exec (realKernel Bool.true a b n B).toAlgKernel s =
      exec (realKernel Bool.false a b n B).toAlgKernel s := by
  simp [exec, realKernel, ComputeKernel.surfaceBody, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeKernel.toAlgKernel, stepStmts, stepStmt, evalOp.eq_def, Tile.bop,
    NumericDType.mul, Nat.mul_comm]

theorem commuted_traceSafe (a b : RegionName) (n B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin B, s.pid*B+j.val < n → s.pid*B+j.val < bounds a)
    (ho : ∀ j : Fin B, s.pid*B+j.val < n → s.pid*B+j.val < bounds b) :
    Kernel.TraceSafe bounds (realKernel Bool.true a b n B).toAlgKernel s := by
  have h := cos_func_traceSafe a b n B bounds s hx ho
  simpa [Kernel.TraceSafe,realKernel,ComputeKernel.surfaceBody,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeKernel.toAlgKernel,Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,
    MaskOpt.SafeAt,stepStmt,evalOp.eq_def,Tile.bop,NumericDType.mul,Nat.mul_comm] using h

theorem commuted_flattenOk (a b : RegionName) (n B : Nat) :
    (realKernel Bool.true a b n B).toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk,realKernel,ComputeKernel.surfaceBody,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeKernel.toAlgKernel,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CosineCompute.RealProof
