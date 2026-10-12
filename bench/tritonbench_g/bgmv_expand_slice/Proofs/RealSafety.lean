import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealLoopSafety
import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealSetupSafety

/-! Access safety and flattening support for the entire actual source. Bounds
are stated on input/weight/output addresses, not on intermediate results.
There is no global N mask in the source: its computed split length determines
all output and weight rows, including a padded split window. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealSafety
open VeriTile.Triton RealSetupSource RealSetup RealLoop
noncomputable section
set_option maxHeartbeats 2600000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- The vector is loaded independently of output-loop length. Weight bounds
need only rank-active lanes; output bounds cover both read-modify-write and
plain stores. There is no hidden formula or final-output premise here. -/
structure ActiveBounds (bounds : RegionBounds) (p : Parameters) (s : BlockState)
    (slot : Nat) (even : Bool) : Prop where
  input : ∀ k : Fin p.BK, even = Bool.true ∨ k.val < p.K →
    s.pids 1 * p.inputBatchStride + k.val * p.inputStride < bounds p.input
  weights : ∀ g, g < p.length → ∀ k : Fin p.BK, k.val < p.K →
    (p.geometry s slot).weightBase + g*p.rowStride + k.val*p.colStride < bounds p.weights
  output : ∀ g, g < p.length → (p.geometry s slot).outputAddress g < bounds p.output

theorem active_safe (bounds : RegionBounds) (p : Parameters) (s : BlockState) (slot : Nat)
    (cast swapped even add : Bool)
    (hm : s.readMemValue .int (Region.cast p.indices) (s.pids 1) = Int.ofNat slot)
    (hmeta : s.pids 1 < bounds (Region.cast p.indices))
    (hb : ActiveBounds bounds p s slot even) :
    Stmt.TraceSafeListR .triv bounds (p.body cast swapped even add) s := by
  unfold Parameters.body
  refine Stmt.TraceSafeListR.append_intro _ _
    (RealSetupSafety.prelude_safe bounds p s hmeta) (fun t ht => ?_)
  rw [stepStmtsR_triv, prelude_exec p s (Int.ofNat slot) hm] at ht
  obtain rfl := Option.some.inj ht
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  have hg : evalOp Parameters.guard (afterPrelude s (Int.ofNat slot)) = some (Tile.scalar Bool.true) := by
    have hn : Int.ofNat slot ≠ -1 := by change (slot : ℤ) ≠ -1; omega
    simpa only [decide_eq_true hn] using guard_eval (afterPrelude s (Int.ofNat slot))
      (Int.ofNat slot) (by simp [afterPrelude])
  simp only [Stmt.TraceSafeR, Parameters.guard, Op.SafeAtR.eq_def, true_and, evalOpR_triv]
  change (match evalOp Parameters.guard (afterPrelude s (Int.ofNat slot)) with
    | some v => if v.data PUnit.unit then Stmt.TraceSafeListR .triv bounds
        (p.setup cast even ++ [p.loop add swapped]) (afterPrelude s (Int.ofNat slot)) else True
    | none => True)
  rw [hg]
  simp only [Tile.scalar_data, if_true]
  refine Stmt.TraceSafeListR.append_intro _ _
    (RealSetupSafety.setup_safe bounds p s slot cast even hb.input) (fun u hu => ?_)
  rw [stepStmtsR_triv, setup_exec p s slot cast even] at hu
  obtain rfl := Option.some.inj hu
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  exact RealLoopSafety.loop_safe bounds (p.geometry s slot) (inputValues p s even)
    (afterSetup p s slot cast even) add swapped (setup_registers p s slot cast even) hb.weights hb.output

theorem sentinel_safe (bounds : RegionBounds) (p : Parameters) (s : BlockState)
    (cast swapped even add : Bool)
    (hm : s.readMemValue .int (Region.cast p.indices) (s.pids 1) = -1)
    (hmeta : s.pids 1 < bounds (Region.cast p.indices)) :
    Stmt.TraceSafeListR .triv bounds (p.body cast swapped even add) s := by
  unfold Parameters.body
  refine Stmt.TraceSafeListR.append_intro _ _
    (RealSetupSafety.prelude_safe bounds p s hmeta) (fun t ht => ?_)
  rw [stepStmtsR_triv, prelude_exec p s (-1) hm] at ht
  obtain rfl := Option.some.inj ht
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  have hg : evalOp Parameters.guard (afterPrelude s (-1)) = some (Tile.scalar Bool.false) :=
    guard_eval _ (-1) (by simp [afterPrelude])
  simp only [Stmt.TraceSafeR, Parameters.guard, Op.SafeAtR.eq_def, true_and, evalOpR_triv]
  change (match evalOp Parameters.guard (afterPrelude s (-1)) with
    | some v => if v.data PUnit.unit then Stmt.TraceSafeListR .triv bounds
        (p.setup cast even ++ [p.loop add swapped]) (afterPrelude s (-1)) else True
    | none => True)
  rw [hg]
  trivial

theorem source_flatten_ok (p : Parameters) (input : TypedSource.InputStorage)
    (weight : TypedSource.WeightStorage) (out_dtype : FloatDType)
    (cast swapped even add : Bool) :
    (p.source input weight out_dtype cast swapped even add).eraseDType.toAlgKernel.FlattenOk := by
  simp only [Kernel.FlattenOk, Parameters.source_body]
  cases cast <;> cases swapped <;> cases even
  all_goals simp [Parameters.body, Parameters.prelude, Parameters.setup, Parameters.guard,
    Parameters.inputLoad, Parameters.inputAddress, Parameters.split, Parameters.weightPointer,
    Parameters.outputPointer, Parameters.loop, RealLoopSource.body, RealLoopSource.loadPrefix,
    RealBlock.tail, RealBlock.update, RealBlock.accumulated, RealBlock.reduction,
    RealBlock.oldOutputLoad, RealBlock.outputAddress, RealBlock.outputStore,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def, MemAccess.FlattenOkC, MaskOpt.FlattenOkC]

theorem source_active_safe (bounds : RegionBounds) (p : Parameters) (input : TypedSource.InputStorage)
    (weight : TypedSource.WeightStorage) (out_dtype : FloatDType)
    (cast swapped even add : Bool) (s : BlockState) (slot : Nat)
    (hm : s.readMemValue .int (Region.cast p.indices) (s.pids 1) = Int.ofNat slot)
    (hmeta : s.pids 1 < bounds (Region.cast p.indices))
    (hb : ActiveBounds bounds p s slot even) :
    (p.source input weight out_dtype cast swapped even add).eraseDType.toAlgKernel.TraceSafeR .triv bounds s := by
  simpa only [Kernel.TraceSafeR, Parameters.source_body] using active_safe bounds p s slot cast swapped even add hm hmeta hb

theorem source_sentinel_safe (bounds : RegionBounds) (p : Parameters) (input : TypedSource.InputStorage)
    (weight : TypedSource.WeightStorage) (out_dtype : FloatDType)
    (cast swapped even add : Bool) (s : BlockState)
    (hm : s.readMemValue .int (Region.cast p.indices) (s.pids 1) = -1)
    (hmeta : s.pids 1 < bounds (Region.cast p.indices)) :
    (p.source input weight out_dtype cast swapped even add).eraseDType.toAlgKernel.TraceSafeR .triv bounds s := by
  simpa only [Kernel.TraceSafeR, Parameters.source_body] using sentinel_safe bounds p s cast swapped even add hm hmeta

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealSafety
