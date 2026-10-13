import bench.tritonbench_g.rotary_emb.Proofs.Source
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FPProof
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FullSource
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

def sourcePrefix (v : Bool) (BH : Nat) : List ComputeStmt := [
  .assign .nat [] "cur_head_index" (.alg (.programId 0)),
  .assign .nat [] "cur_seq_index" (.alg (.programId 1)),
  .assign .nat [BH] "cur_head_range" (.alg (.add .nat .scalarL
    (if v then .mul .nat .nil (.constNat BH) (.ref .nat [] "cur_head_index")
      else .mul .nat .nil (.ref .nat [] "cur_head_index") (.constNat BH))
    (.arange BH)))]

theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (BH : Nat) (after : List ComputeStmt) :
    run (engine M) (sourcePrefix Bool.false BH ++ after) s =
      run (engine M) (sourcePrefix Bool.true BH ++ after) s := by
  change Contextual M s
    [.assign .nat [] "cur_head_index" (.alg (.programId 0)),
      .assign .nat [] "cur_seq_index" (.alg (.programId 1))] after
    (.assign .nat [BH] "cur_head_range" (.alg (.add .nat .scalarL
      (.mul .nat .nil (.ref .nat [] "cur_head_index") (.constNat BH)) (.arange BH))))
    (.assign .nat [BH] "cur_head_range" (.alg (.add .nat .scalarL
      (.mul .nat .nil (.constNat BH) (.ref .nat [] "cur_head_index")) (.arange BH))))
  fp_prove

theorem template_prefix (data cos sin : Storage) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    (template data cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd
      L HQ HK BH BS D).surfaceBody.take 3 = sourcePrefix Bool.false BH := by
  cases data <;> cases cos <;> cases sin <;> rfl

theorem original_prefix (q k cos sin : Storage) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    (original q k cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd
      L HQ HK BH BS D).surfaceBody.take 3 = sourcePrefix Bool.false BH := by
  simp [original, ComputeKernel.surfaceBody, List.take_append, List.take_take,
    template_shape, template_prefix]

theorem decomposition (q k cos sin : Storage) (v : Bool) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    (kernel q k cos sin v Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd
      L HQ HK BH BS D).surfaceBody = sourcePrefix v BH ++
      (original q k cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd
        L HQ HK BH BS D).surfaceBody.drop 3 := by
  have hp := original_prefix q k cos sin Q K Cos Sin sqb sqh sqd skb skh skd
    scb scd ssb ssd L HQ HK BH BS D
  cases v
  · simpa only [kernel, Bool.false_eq_true, ↓reduceIte, hp] using
      (List.take_append_drop 3 (original q k cos sin Q K Cos Sin sqb sqh sqd
        skb skh skd scb scd ssb ssd L HQ HK BH BS D).surfaceBody).symm
  · have hp2 := congrArg (List.take 2) hp
    simp only [List.take_take, sourcePrefix, List.take_succ_cons, List.take_zero] at hp2
    simpa only [kernel, ↓reduceIte, ComputeKernel.surfaceBody, hp2, sourcePrefix,
      List.cons_append, List.nil_append]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FPProof
