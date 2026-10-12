import bench.tritonbench_g.attn_fwd_causal.Proofs.RealExecution

/-! The output formula depends only on inputs actually read by the source.
Q/K/V pins apply only to active masked lanes; key/value/scale pins range over
the finite list of visited tiles. Query scale and every visited key scale are
unmasked loads, even if all lanes of a visited tile are inactive. No computed
output, intermediate register or execution result is a pinning premise. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealInputPins
open VeriTile.Triton RealFormula RealLoads
noncomputable section

structure Agree {BM BN BD : Nat} (xs ys : Inputs BM BN BD)
    (startM N HA stage : Nat) : Prop where
  query : ∀ row d, startM*BM+row.val < N → d.val < HA → xs.query row d = ys.query row d
  key : ∀ v ∈ allVisits startM BM BN stage, ∀ j d,
    v.start+j.val < N → d.val < HA → xs.key v.start j d = ys.key v.start j d
  value : ∀ v ∈ allVisits startM BM BN stage, ∀ j d,
    v.start+j.val < N → d.val < HA → xs.value v.start j d = ys.value v.start j d
  queryScale : xs.queryScale = ys.queryScale
  keyScale : ∀ v ∈ allVisits startM BM BN stage,
    xs.keyScale (v.start/BN) = ys.keyScale (v.start/BN)

theorem score_congr {BM BN BD : Nat} {xs ys : Inputs BM BN BD}
    {startM N HA stage : Nat} (h : Agree xs ys startM N HA stage)
    (v : Visit) (hv : v ∈ allVisits startM BM BN stage) (row : Fin BM) (j : Fin BN) :
    score xs startM N HA v row j = score ys startM N HA v row j := by
  unfold score
  rw [h.queryScale, h.keyScale v hv]
  congr 2
  apply Finset.sum_congr rfl
  intro d _
  congr 1
  · split_ifs with hd
    · exact h.query row d hd.1 hd.2
    · rfl
  · split_ifs with hd
    · exact h.key v hv j d hd.1 hd.2
    · rfl

theorem loaded_value_congr {BM BN BD : Nat} {xs ys : Inputs BM BN BD}
    {startM N HA stage : Nat} (h : Agree xs ys startM N HA stage)
    (v : Visit) (hv : v ∈ allVisits startM BM BN stage) (j : Fin BN) (d : Fin BD) :
    loadedValue xs N HA v j d = loadedValue ys N HA v j d := by
  unfold loadedValue
  split_ifs with hd
  · exact h.value v hv j d hd.1 hd.2
  · rfl

theorem blocks_congr {BM BN BD : Nat} {xs ys : Inputs BM BN BD}
    {startM N HA stage : Nat} (h : Agree xs ys startM N HA stage)
    (row : Fin BM) (d : Fin BD) :
    blocks xs startM N HA stage row d = blocks ys startM N HA stage row d := by
  unfold blocks
  apply List.map_congr_left
  intro v hv
  unfold blockPairs
  apply List.map_congr_left
  intro j _
  simp only [logit, score_congr h v hv, loaded_value_congr h v hv]

theorem expected_congr {BM BN BD : Nat} {xs ys : Inputs BM BN BD}
    {startM N HA stage : Nat} (h : Agree xs ys startM N HA stage)
    (row : Fin BM) (d : Fin BD) :
    expected xs startM N HA stage row d = expected ys startM N HA stage row d := by
  simp only [expected, blocks_congr h row d]

structure Pins {BM BN BD : Nat} (s : BlockState) (c : Config) (stage : Nat)
    (xs : Inputs BM BN BD) : Prop where
  query : ∀ row d, c.startM*BM+row.val < c.context → d.val < c.activeHead →
    s.readMem c.Q (c.base+(c.startM*BM+row.val)*c.strideM+d.val*c.strideD) = xs.query row d
  key : ∀ v ∈ allVisits c.startM BM BN stage, ∀ j d,
    v.start+j.val < c.context → d.val < c.activeHead →
    s.readMem c.K (c.base+d.val+j.val*c.strideKN+c.headDim*v.start) = xs.key v.start j d
  value : ∀ v ∈ allVisits c.startM BM BN stage, ∀ j d,
    v.start+j.val < c.context → d.val < c.activeHead →
    s.readMem c.V (c.base+j.val*c.strideM+d.val*c.strideD+c.headDim*v.start) = xs.value v.start j d
  queryScale : s.readMem c.QS (c.queryScaleBase+c.startM) = xs.queryScale
  keyScale : ∀ v ∈ allVisits c.startM BM BN stage,
    s.readMem c.KS (c.keyScaleBase+v.start/BN) = xs.keyScale (v.start/BN)

/-- Every initial memory supplies inputs satisfying these pins. -/
theorem pins_from_memory (BM BN BD : Nat) (s : BlockState) (c : Config) (stage : Nat) :
    Pins s c stage (inputs s c BM BN BD) :=
  ⟨fun _ _ _ _ => rfl, fun _ _ _ _ _ _ => rfl, fun _ _ _ _ _ _ => rfl,
    rfl, fun _ _ => rfl⟩

theorem agrees {BM BN BD : Nat} (s : BlockState) (c : Config) (stage : Nat)
    (xs : Inputs BM BN BD) (h : Pins s c stage xs) :
    Agree (inputs s c BM BN BD) xs c.startM c.context c.activeHead stage :=
  ⟨h.query, h.key, h.value, h.queryScale, h.keyScale⟩

theorem formula_of_pins {BM BN BD : Nat} (s : BlockState) (c : Config) (stage : Nat)
    (xs : Inputs BM BN BD) (h : Pins s c stage xs) (row : Fin BM) (d : Fin BD) :
    expected (inputs s c BM BN BD) c.startM c.context c.activeHead stage row d =
      expected xs c.startM c.context c.activeHead stage row d :=
  expected_congr (agrees s c stage xs h) row d

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealInputPins
