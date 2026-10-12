import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHAccessSafety

/-! Safety of both complete gate branches. Only the selected branch requires
its two-dimensional gate tile and one-dimensional full-chunk endpoint bounds.
Numerical values remain abstract; safety does not require zero padding. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHGateSafety
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHAccessSafety
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (r : Regions) (c : Nat) (gateK : Bool)
    (bounds : RegionBounds) : Prop where
  key : gateK = Bool.true → KeyBounds p l r.gate c bounds
  keyEnd : gateK = Bool.true → EndpointBounds p l r.gate c p.keyHead p.keyFeature p.K p.BK (l.key*p.BK) bounds
  value : gateK = Bool.false → ValueBounds p l r.gate c bounds
  valueEnd : gateK = Bool.false → EndpointBounds p l r.gate c p.valueHead p.valueFeature p.V p.BV (l.value*p.BV) bounds

theorem key_safe (p : Layout) (l : Launch) (r : Regions) (c : Nat)
    (bounds : RegionBounds) (u : BlockState) (hb : Bounds p l r c Bool.true bounds)
    (hi : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    Stmt.TraceSafeList bounds (RealHProgram.gateKey p r) u := by
  unfold RealHProgram.gateKey
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (key_eval p l r.gate c u hh hi ht))]
  refine ⟨by simp [Stmt.TraceSafe,keyPtr,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (endpoint_eval p l r.gate c
    p.keyHead p.keyFeature p.K p.BK l.key "i_k" _
    (by simp [hh]) (by simp [ht]) (by simp [hi])))]
  refine ⟨by simp [Stmt.TraceSafe,endpointPtr,endpointOp,times,Op.SafeAt],?_⟩
  apply TraceProofs.assign_cons (load1_safe r.gate (l.head*p.keyHead) (p.T*p.K) p.BK
    p.keyFeature ((c*p.BT+p.BT-1)*p.K+l.key*p.BK) "p_gn" bounds _ (hb.keyEnd rfl) (by simp [endpointPointer]))
  intro endpoint
  apply TraceProofs.assign_cons (by simp [carry,Op.SafeAt,Op.SafeAt.eq_def])
  intro scaled
  apply TraceProofs.assign_cons (load2_safe r.gate (l.head*p.keyHead) p.K p.T p.BK p.BT p.keyFeature p.keyTime (l.key*p.BK) (c*p.BT)
    "p_g" bounds _ (hb.key rfl) (by simp [keyPointer]))
  intro gate
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]

theorem value_safe (p : Layout) (l : Launch) (r : Regions) (c : Nat)
    (bounds : RegionBounds) (u : BlockState) (hb : Bounds p l r c Bool.false bounds)
    (hi : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    Stmt.TraceSafeList bounds (RealHProgram.gateValue p r) u := by
  unfold RealHProgram.gateValue
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (value_eval p l r.gate c u hh hi ht))]
  refine ⟨by simp [Stmt.TraceSafe,valuePtr,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (endpoint_eval p l r.gate c
    p.valueHead p.valueFeature p.V p.BV l.value "i_v" _
    (by simp [hh]) (by simp [ht]) (by simp [hi])))]
  refine ⟨by simp [Stmt.TraceSafe,endpointPtr,endpointOp,times,Op.SafeAt],?_⟩
  apply TraceProofs.assign_cons (load1_safe r.gate (l.head*p.valueHead) (p.T*p.V) p.BV
    p.valueFeature ((c*p.BT+p.BT-1)*p.V+l.value*p.BV) "p_gn" bounds _ (hb.valueEnd rfl) (by simp [endpointPointer]))
  intro endpoint
  apply TraceProofs.assign_cons (by simp [carry,Op.SafeAt,Op.SafeAt.eq_def])
  intro scaled
  apply TraceProofs.assign_cons (load2_safe r.gate (l.head*p.valueHead) p.T p.V p.BT p.BV p.valueTime p.valueFeature (c*p.BT) (l.value*p.BV)
    "p_g" bounds _ (hb.value rfl) (by simp [valuePointer]))
  intro gate
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]

theorem branch_safe (p : Layout) (l : Launch) (r : Regions) (c : Nat) (gateK : Bool)
    (bounds : RegionBounds) (u : BlockState) (hb : Bounds p l r c gateK bounds)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    Stmt.TraceSafe bounds (.ifThenElse (.constBool gateK) (gateKey p r) (RealHProgram.gateValue p r)) u := by
  cases gateK
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,Bool.false_eq_true,↓reduceIte,true_and]
    exact value_safe p l r c bounds u hb hiv hih hit
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,↓reduceIte,true_and]
    exact key_safe p l r c bounds u hb hik hih hit

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHGateSafety
