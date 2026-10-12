import bench.tritonbench_g.attn_fwd_causal.Proofs.RealLoads
import bench.tritonbench_g.attn_fwd_causal.Proofs.RealNumeric

/-! Execution of the actual helper loads and pointer advances. The helper
selector changes only the source-local pointer names; both helpers read the
same input-memory views, with their own current tile origin. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMemory
open VeriTile.Triton RealLoads
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 800000
set_option linter.unusedSimpArgs false

def keyName (second : Bool) : RegName := if second then "K_ptrs_second" else "K_ptrs_first"
def scaleName (second : Bool) : RegName := if second then "K_scale_ptr_second" else "K_scale_ptr_first"
def valueName (second : Bool) : RegName := if second then "V_ptrs_second" else "V_ptrs_first"

theorem loads_execute (s : BlockState) (c : Config) (BM BN BD start : Nat) (second : Bool)
    (hp : s.regs .ptr [BD,BN] (keyName second) = some (keyPointer c BD BN start))
    (hsp : s.regs .ptr [] (scaleName second) = some (keyScalePointer c BN start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hu : ∀ r a, s.undef r a = 0) :
    ∃ t, stepStmts (RealBody.loadPrefix (keyName second) (scaleName second) BD BN c.context c.activeHead) s = some t ∧
      t.regs .real [BD,BN] "k" = some (RealTiles.key (inputs s c BM BN BD) start c.context c.activeHead) ∧
      t.regs .real [] "k_scale" = some (Tile.scalar (some ((inputs s c BM BN BD).keyScale (start/BN)))) ∧
      t.regs .nat [] "start_n" = some (Tile.scalar start) ∧
      Frame ["start_n","k_mask","k","k_scale"] s t := by
  let sn := s.setReg "start_n" .nat [] (Tile.scalar start)
  let sm := sn.setReg "k_mask" .bool [BD,BN] (keyMask BD BN c.context c.activeHead start)
  have hm := eval_key_mask sn BD BN c.context c.activeHead start (by simpa [sn] using hn) (by simp [sn])
  have hk := key_load sm c BM BN BD start (keyName second)
    (by cases second <;> simpa [sm, sn, keyName] using hp)
    (by simp [sm]) (by simpa [sm, sn] using hu)
  rw [inputs_same_memory sm s c BM BN BD rfl] at hk
  let sk := sm.setReg "k" .real [BD,BN] (RealTiles.key (inputs s c BM BN BD) start c.context c.activeHead)
  have hscale := key_scale_load sk c BM BN BD start (scaleName second)
    (by cases second <;> simpa [sk, sm, sn, scaleName] using hsp)
  rw [inputs_same_memory sk s c BM BN BD rfl] at hscale
  simp only [RealBody.loadPrefix]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.ref .nat [] "start_n") s =
    some (Tile.scalar start) from by simpa using hs)),
    stepStmts.cons_some (stepStmt_assign_eq_some hm),
    stepStmts.cons_some (stepStmt_assign_eq_some hk),
    stepStmts.cons_some (stepStmt_assign_eq_some hscale), stepStmts.nil]
  refine ⟨_, rfl, by simp [sk], by simp, by simp [sk, sm, sn], rfl, rfl, rfl, ?_⟩
  intro dt sh name hname
  simp only [List.mem_cons, not_or] at hname
  simp [sk, sm, sn, hname.1, hname.2.1, hname.2.2.1, hname.2.2.2.1]

theorem value_executes (s : BlockState) (c : Config) (BM BN BD start : Nat) (diagonal second : Bool)
    (hp : s.regs .ptr [BN,BD] (valueName second) = some (valuePointer c BN BD start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hu : ∀ r a, s.undef r a = 0) :
    ∃ t, stepStmt (RealBody.loadValue (valueName second) BN BD c.context c.activeHead) s = some t ∧
      t.regs .real [BN,BD] "v" = some (RealTiles.value (inputs s c BM BN BD) ⟨start,diagonal⟩ c.context c.activeHead) ∧
      Frame ["v"] s t := by
  refine ⟨_, stepStmt_assign_eq_some (value_load s c BM BN BD start diagonal (valueName second) hp hn hs hu), by simp, ?_⟩
  exact Frame.setReg _ s "v" _ _ _ (by simp)

theorem advances_execute (s : BlockState) (c : Config) (BN BD start : Nat) (hN : 0 < BN) (second : Bool)
    (hk : s.regs .ptr [BD,BN] (keyName second) = some (keyPointer c BD BN start))
    (hs : s.regs .ptr [] (scaleName second) = some (keyScalePointer c BN start))
    (hv : s.regs .ptr [BN,BD] (valueName second) = some (valuePointer c BN BD start)) :
    ∃ t, stepStmts (RealBody.advance (keyName second) (scaleName second) (valueName second) BN BD c.headDim) s = some t ∧
      t.regs .ptr [BD,BN] (keyName second) = some (keyPointer c BD BN (start+BN)) ∧
      t.regs .ptr [] (scaleName second) = some (keyScalePointer c BN (start+BN)) ∧
      t.regs .ptr [BN,BD] (valueName second) = some (valuePointer c BN BD (start+BN)) ∧
      Frame [keyName second,scaleName second,valueName second] s t := by
  have hsk : scaleName second ≠ keyName second := by cases second <;> decide
  have hvk : valueName second ≠ keyName second := by cases second <;> decide
  have hvs : valueName second ≠ scaleName second := by cases second <;> decide
  let sk := s.setReg (keyName second) .ptr [BD,BN] (keyPointer c BD BN (start+BN))
  let ss := sk.setReg (scaleName second) .ptr [] (keyScalePointer c BN (start+BN))
  have h1 := key_advance s c BN BD start (keyName second) hk
  have h2 := scale_advance sk c BN start hN (scaleName second) (by simpa [sk, hsk] using hs)
  have h3 := value_advance ss c BN BD start (valueName second) (by simpa [ss, sk, hvk, hvs] using hv)
  simp only [RealBody.advance]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h1),
    stepStmts.cons_some (stepStmt_assign_eq_some h2),
    stepStmts.cons_some (stepStmt_assign_eq_some h3), stepStmts.nil]
  refine ⟨_, rfl, by simp [ss, sk, Ne.symm hvk, Ne.symm hsk],
    by simp [ss, Ne.symm hvs], by simp, rfl, rfl, rfl, ?_⟩
  intro dt sh name hn
  simp only [List.mem_cons, not_or] at hn
  simp [ss, sk, hn.1, hn.2.1, hn.2.2.1]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMemory
