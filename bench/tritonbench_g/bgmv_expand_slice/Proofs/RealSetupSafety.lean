import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealAccessSafety

/-! Bounds for the signed metadata cell and the input vector read before the
loop. Input bounds remain required on an active path even when N=0, whereas
the -1 path needs only its metadata bound. Remaining setup is register-only. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealSetupSafety
open VeriTile.Triton RealSetupSource RealSetup
noncomputable section
set_option maxHeartbeats 2200000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem prelude_safe (bounds : RegionBounds) (p : Parameters) (initial : BlockState)
    (hb : initial.pids 1 < bounds (Region.cast p.indices)) :
    Stmt.TraceSafeListR .triv bounds p.prelude initial := by
  unfold Parameters.prelude
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s1 hs1 => ?_)
  rw [stepStmtR_triv, stepStmt_assign_eq_some (by simp :
    evalOp (Op.programId 0) initial = some (Tile.scalar (initial.pids 0)))] at hs1
  obtain rfl := Option.some.inj hs1
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s2 hs2 => ?_)
  rw [stepStmtR_triv, stepStmt_assign_eq_some (by simp :
    evalOp (Op.programId 1) _ = some (Tile.scalar (initial.pids 1)))] at hs2
  obtain rfl := Option.some.inj hs2
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, true_and,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR, MaskOpt.ActiveR, evalOpR_triv, evalOp_ref]
  intro offsets hoff i _
  simp only [BlockState.setReg_same] at hoff
  obtain rfl := Option.some.inj hoff
  exact hb

private theorem input_addresses (p : Parameters) (initial s : BlockState)
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar (initial.pids 1)))
    (hk : s.regs .nat [p.BK] "offset_k" = some (Tile.vec (fun k => k.val))) :
    evalOp (.add .nat .scalarL
      (.mul .nat .nil (.ref .nat [] "cur_batch") (.constNat p.inputBatchStride))
      (.mul .nat .scalarR (.ref .nat [p.BK] "offset_k") (.constNat p.inputStride))) s =
      some (Tile.vec (fun k => initial.pids 1 * p.inputBatchStride + k.val * p.inputStride)) := by
  simp only [evalOp, hcb, hk, Tile.bop, Tile.scalar_data, Tile.vec,
    Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
    Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR, NumericDType.add, NumericDType.mul]
  rfl

theorem input_safe (bounds : RegionBounds) (p : Parameters) (initial s : BlockState) (even : Bool)
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar (initial.pids 1)))
    (hk : s.regs .nat [p.BK] "offset_k" = some (Tile.vec (fun k => k.val)))
    (hb : ∀ k : Fin p.BK, even = Bool.true ∨ k.val < p.K →
      initial.pids 1 * p.inputBatchStride + k.val * p.inputStride < bounds p.input) :
    (p.inputLoad even).SafeAtR .triv bounds s := by
  have ha := input_addresses p initial s hcb hk
  cases even <;> simp only [Parameters.inputLoad, Parameters.inputAddress,
    Bool.false_eq_true, if_false, if_true, Op.SafeAtR.eq_def, true_and, and_true,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR, MaskOpt.ActiveR, evalOpR_triv]
  all_goals
    intro offsets hoff i hactive
    rw [ha] at hoff
    obtain rfl := Option.some.inj hoff
  · obtain ⟨mask, hmask, hi⟩ := hactive
    have he : evalOp (.lt .nat .scalarR (.ref .nat [p.BK] "offset_k") (.constNat p.K)) s =
        some (Tile.vec (fun k => decide (k.val < p.K))) := by
      simp only [evalOp, hk, Tile.cop, Tile.scalar_data, Tile.vec,
        Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR, ComparableDType.lt]
      rfl
    rw [he] at hmask
    obtain rfl := Option.some.inj hmask
    exact hb i.1 (Or.inr (by simpa [Tile.vec] using hi))
  · exact hb i.1 (Or.inl rfl)

private theorem register_suffix_safe (bounds : RegionBounds) (p : Parameters) (cast : Bool) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds
      ([.assign .nat [] "split_n_length" p.split] ++
      (if cast then [.assign .real [p.BK] "tiled_a" (.ref .real [p.BK] "tiled_a")] else []) ++
      [.assign .ptr [] "b_ptr" p.weightPointer, .assign .ptr [] "c_ptr" p.outputPointer]) s := by
  apply Stmt.TraceSafeListR.of_forall
  intro st hst t
  cases cast <;> simp only [Bool.false_eq_true, if_false, if_true, List.cons_append,
    List.nil_append, List.append_nil, List.mem_cons, List.not_mem_nil, or_false] at hst
  all_goals rcases hst with rfl | rfl | rfl | rfl
  all_goals simp [Stmt.TraceSafeR, Parameters.split, Parameters.weightPointer,
    Parameters.outputPointer, Op.SafeAtR.eq_def]

theorem setup_safe (bounds : RegionBounds) (p : Parameters) (initial : BlockState)
    (slot : Nat) (cast even : Bool)
    (hb : ∀ k : Fin p.BK, even = Bool.true ∨ k.val < p.K →
      initial.pids 1 * p.inputBatchStride + k.val * p.inputStride < bounds p.input) :
    Stmt.TraceSafeListR .triv bounds (p.setup cast even) (afterPrelude initial (Int.ofNat slot)) := by
  unfold Parameters.setup
  simp only [List.append_assoc, List.cons_append, List.nil_append]
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s1 hs1 => ?_)
  rw [stepStmtR_triv, stepStmt_assign_eq_some (by simp :
    evalOp (Op.arange p.BK) (afterPrelude initial (Int.ofNat slot)) =
      some (Tile.vec (fun k => k.val)))] at hs1
  obtain rfl := Option.some.inj hs1
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s2 hs2 => ?_)
  rw [stepStmtR_triv, stepStmt_assign_eq_some (by simp :
    evalOp (Op.arange p.BN) _ = some (Tile.vec (fun i => i.val)))] at hs2
  obtain rfl := Option.some.inj hs2
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t _ => ?_)
  · have hi := input_safe bounds p initial
      (((afterPrelude initial (Int.ofNat slot)).setReg "offset_k" .nat [p.BK]
        (Tile.vec (fun k => k.val))).setReg "offset_n" .nat [p.BN] (Tile.vec (fun i => i.val))) even
      (by simp [afterPrelude]) (by simp) hb
    cases even <;> simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, evalOpR_triv,
      evalOp, Tile.scalar_data, Bool.false_eq_true, if_false, if_true, true_and]
    all_goals
      refine Stmt.TraceSafeListR.cons_intro (by simpa only [Stmt.TraceSafeR] using hi)
        (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  · simpa only [List.cons_append, List.nil_append, List.append_assoc] using
      register_suffix_safe bounds p cast t

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealSetupSafety
