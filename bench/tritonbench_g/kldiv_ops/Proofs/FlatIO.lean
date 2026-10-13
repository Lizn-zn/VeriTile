import bench.tritonbench_g.kldiv_ops.Proofs.ForwardSafety
import bench.tritonbench_g.kldiv_ops.Proofs.BackwardSafety

/-! Complete mathematical flat-memory executions of the source algorithms.
Allocation coverage and disjointness express the existing flattening bridge. Distinct region
names have separate allocations; equal names remain aliases. Only active accesses are bounded.
The bridge currently needs zero undefined cells even though the source uses explicit load padding.
These results use ideal real arithmetic, Nat strides and unbounded signed loop arithmetic;
typed rounding, native machine-width overflow and host/grid obligations are separate surfaces. -/

namespace VeriTile.Bench.TritonBenchG.KldivOps.FlatIO
open VeriTile.Triton FullAlgorithm
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Every cell outside the independently specified output window retains its raw contents. -/
theorem flatten_frame (A : FlatAlloc) (s t : BlockState) (O : RegionName) (base count : Nat)
    (hf : BlockState.WriteWithin
      (fun address ↦ address.1 = O ∧ ∃ k, k < count ∧ address.2 = base + k) s t)
    (r : RegionName) (off : Nat)
    (hno : r ≠ A.flat ∨ ∀ k : Fin count, off ≠ A.addr O (base + k.val)) :
    (A.flattenState t).mem r off = (A.flattenState s).mem r off := by
  change (if r = A.flat then A.readFlat t off else MemCell.real 0) =
    (if r = A.flat then A.readFlat s off else MemCell.real 0)
  by_cases hr : r = A.flat
  · subst r
    simp only [if_pos rfl]
    unfold FlatAlloc.readFlat
    cases hc : A.decode off with
    | none => rfl
    | some p =>
      apply congrArg A.trCell
      apply Eq.symm
      apply hf p.1 p.2
      rintro ⟨hp, k, hk, he⟩
      change p.1 = O at hp
      change p.2 = base + k at he
      rcases hno with hne | hmiss
      · exact hne rfl
      · apply hmiss ⟨k, hk⟩
        have haddr := (A.decode_sound hc).2.1
        simpa only [hp, he] using haddr
  · simp [hr]

theorem forward_flattenOk (Y T O : RegionName) (sy st so N B : Nat)
    (eps : ℝ) (flag : Bool) (reduction : Nat) :
    (forward Y T O sy st so N B eps flag reduction).FlattenOk := by
  simp [forward, forwardSurface, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.FlattenOk, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def,
    Except.bind, bind]

theorem backward_flattenOk (T O : RegionName) (st so N B : Nat) (flag : Bool) :
    (backward T O st so N B flag).FlattenOk := by
  cases flag <;>
    simp [backward, backwardSurface, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Kernel.FlattenOk, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def,
      Except.bind, bind]

/-- Complete elementwise execution with the independent loss and full raw-cell frame. -/
theorem element_run (A : FlatAlloc) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ) (flag : Bool)
    (hd : A.Disjoint) (hO : O ∈ A.regions)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (hu : s.undef = fun _ _ ↦ 0)
    (hB : 0 < B) (hYO : Y ≠ O) (hTO : T ≠ O)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < A.extent Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T)
    (hbo : ∀ k : Fin N,
      s.pids 0 * so + k.val < A.extent O) :
    ∃ t, exec (A.flattenKernel (forward Y T O sy st so N B eps flag 0))
        (A.flattenState s) = some t ∧
      (∀ k : Fin N,
        t.readMem A.flat (A.addr O (s.pids 0 * so + k.val)) =
          FullMath.loss flag eps (s.readMem Y (s.pids 0 * sy + k.val))
            (s.readMem T (s.pids 0 * st + k.val))) ∧
      (∀ r off, (r ≠ A.flat ∨ ∀ k : Fin N,
        off ≠ A.addr O (s.pids 0 * so + k.val)) →
          t.mem r off = (A.flattenState s).mem r off) := by
  obtain ⟨t, he, hv, hf⟩ :=
    FullForward.element_source_run s Y T O sy st so N B eps flag hB hYO hTO
  have hs := FullForward.element_source_safe A.extent s Y T O sy st so N B eps flag
    hYO hTO hby hbt hbo
  have hflat := A.execR_flatten hd hc .triv
    (forward Y T O sy st so N B eps flag 0) s hs
    (forward_flattenOk Y T O sy st so N B eps flag 0) hu
  simp only [execR_triv, he, Option.map_some] at hflat
  refine ⟨A.flattenState t, hflat, ?_, flatten_frame A s t O (s.pids 0 * so)
    N hf⟩
  intro k
  have hh := A.flattenState_readMemAs hd t hO (hbo k) .real
  simp only [BlockState.readMemAs_real] at hh
  exact (Option.some.inj hh).trans (hv k.val k.isLt)

/-- Full flat backward execution with independent gradients and a raw-cell frame. -/
theorem backward_run (A : FlatAlloc) (s : BlockState) (T O : RegionName)
    (st so N B : Nat) (flag : Bool)
    (hd : A.Disjoint) (hO : O ∈ A.regions)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (hu : s.undef = fun _ _ ↦ 0)
    (hB : 0 < B) (hTO : T ≠ O)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T)
    (hbo : ∀ k : Fin N, s.pids 0 * so + k.val < A.extent O) :
    ∃ t, exec (A.flattenKernel (backward T O st so N B flag)) (A.flattenState s) = some t ∧
      (∀ k : Fin N, t.readMem A.flat (A.addr O (s.pids 0 * so + k.val)) =
        FullBackward.gradient flag (s.readMem T (s.pids 0 * st + k.val))) ∧
      (∀ r off, (r ≠ A.flat ∨ ∀ k : Fin N, off ≠ A.addr O (s.pids 0 * so + k.val)) →
        t.mem r off = (A.flattenState s).mem r off) := by
  obtain ⟨t, he, hv, hf⟩ := FullBackward.source_run s T O st so N B flag hB hTO
  have hs := FullBackward.source_safe A.extent s T O st so N B flag hTO hbt hbo
  have hflat := A.execR_flatten hd hc .triv (backward T O st so N B flag) s hs
    (backward_flattenOk T O st so N B flag) hu
  simp only [execR_triv, he, Option.map_some] at hflat
  refine ⟨A.flattenState t, hflat, ?_, flatten_frame A s t O (s.pids 0 * so) N hf⟩
  intro k
  have hh := A.flattenState_readMemAs hd t hO (hbo k) .real
  simp only [BlockState.readMemAs_real] at hh
  exact (Option.some.inj hh).trans (hv k.val k.isLt)

/-- Every nonzero reduction reads all inputs before its one scalar store. Region aliases
are permitted. N=0 still requires the scalar allocation and writes the empty row sum.
The zero-undef convention is required by the current flattening bridge, not the region proof.
These mathematical contracts do not assert fixed-width overflow or native grid race safety. -/
theorem reduction_run (A : FlatAlloc) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ) (flag : Bool) (reduction : Nat)
    (hd : A.Disjoint) (hO : O ∈ A.regions)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (hu : s.undef = fun _ _ ↦ 0)
    (hB : 0 < B) (hR : reduction ≠ 0)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < A.extent Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T)
    (hbo : s.pids 0 * so < A.extent O) :
    ∃ t, exec (A.flattenKernel (forward Y T O sy st so N B eps flag reduction))
        (A.flattenState s) = some t ∧
      t.readMem A.flat (A.addr O (s.pids 0 * so)) =
        FullMath.rowLoss flag eps (fun k ↦ s.readMem Y (s.pids 0 * sy + k))
          (fun k ↦ s.readMem T (s.pids 0 * st + k)) N ∧
      (∀ r off, (r ≠ A.flat ∨ off ≠ A.addr O (s.pids 0 * so)) →
        t.mem r off = (A.flattenState s).mem r off) := by
  obtain ⟨t, he, hv, hf⟩ := FullForward.reduction_source_run s Y T O sy st so N B
    eps flag reduction hB hR
  have hs := FullForward.reduction_source_safe A.extent s Y T O sy st so N B
    eps flag reduction hB hR hby hbt hbo
  have hflat := A.execR_flatten hd hc .triv
    (forward Y T O sy st so N B eps flag reduction) s hs
    (forward_flattenOk Y T O sy st so N B eps flag reduction) hu
  simp only [execR_triv, he, Option.map_some] at hflat
  refine ⟨A.flattenState t, hflat, ?_, ?_⟩
  · have hh := A.flattenState_readMemAs hd t hO hbo .real
    simp only [BlockState.readMemAs_real] at hh
    exact (Option.some.inj hh).trans hv
  · have hw : BlockState.WriteWithin
        (fun address ↦ address.1 = O ∧ ∃ k, k < 1 ∧ address.2 = s.pids 0 * so + k) s t := by
      intro r off hno
      apply hf r off
      intro heq
      apply hno
      cases heq
      exact ⟨rfl, 0, by omega, by simp⟩
    intro r off hno
    apply flatten_frame A s t O (s.pids 0 * so) 1 hw r off
    rcases hno with hne | hne
    · exact Or.inl hne
    · right
      intro k
      have hk : k.val = 0 := by omega
      simpa [hk] using hne

end VeriTile.Bench.TritonBenchG.KldivOps.FlatIO
#axiomsClean VeriTile.Bench.TritonBenchG.KldivOps.FlatIO.flatten_frame
#axiomsClean VeriTile.Bench.TritonBenchG.KldivOps.FlatIO.forward_flattenOk
#axiomsClean VeriTile.Bench.TritonBenchG.KldivOps.FlatIO.backward_flattenOk
#axiomsClean VeriTile.Bench.TritonBenchG.KldivOps.FlatIO.element_run
#axiomsClean VeriTile.Bench.TritonBenchG.KldivOps.FlatIO.backward_run
#axiomsClean VeriTile.Bench.TritonBenchG.KldivOps.FlatIO.reduction_run
#guard_msgs (drop info) in
#auditModuleAxioms
