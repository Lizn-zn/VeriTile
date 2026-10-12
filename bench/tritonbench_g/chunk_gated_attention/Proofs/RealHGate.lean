import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHAlgebra

/-! Execution of each complete six-statement gate branch, including both
pointer constructions and both loads. Arbitrary tail values remain explicit;
these proofs impose no zero-padding premise. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHGate
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles RealHLoads RealHAlgebra
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def keyState (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (f : StateValue p) (u : BlockState) : BlockState :=
  let x := loaded p l r a padding c
  let s1 := u.setReg "p_g" .blockPtr [p.BK,p.BT] (keyPointer p l r.gate c)
  let s2 := s1.setReg "p_gn" .blockPtr [p.BK]
    (endpointPointer p l r.gate c p.keyHead p.keyFeature p.K p.BK (l.key*p.BK))
  let s3 := s2.setReg "b_gn" .real [p.BK] (Tile.vec (fun i => some (x.keyEnd i)))
  let s4 := s3.setReg "b_h" .real [p.BK,p.BV] (stateTile p (scaled x Bool.true f))
  let s5 := s4.setReg "b_g" .real [p.BK,p.BT] (Tile.mat (fun i j => some (x.keyGate i j)))
  s5.setReg "b_k" .real [p.BK,p.BT] (Tile.mat (fun i j => some (gatedKey x Bool.true i j)))

theorem key_executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (f : StateValue p) (u : BlockState)
    (hi : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hcarry : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f))
    (hdata : u.regs .real [p.BK,p.BT] "b_k" =
      some (Tile.mat (fun i j => some ((loaded p l r a padding c).key i j))))
    (hm : ∀ off, u.readMem r.gate off = a.gate off) (hu : u.undef = padding) :
    stepStmts (gateKey p r) u = some (keyState p l r a padding c f u) := by
  let x := loaded p l r a padding c
  unfold gateKey
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_eval p l r.gate c u hh hi ht))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (endpoint_eval p l r.gate c
    p.keyHead p.keyFeature p.K p.BK l.key "i_k" _
    (by simp [hh]) (by simp [ht]) (by simp [hi])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (endpoint_load p l r.gate a.gate
    (padding r.gate) c p.keyHead p.keyFeature p.K p.BK (l.key*p.BK) "p_gn" _
    (by simpa using hm) (by simpa using congrFun hu r.gate) (by simp)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (scale_key p x f _
    (by simp [hcarry]) (by simp [x,loaded]; rfl)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_load p l r.gate a.gate
    (padding r.gate) c "p_g" _ (by simpa using hm)
    (by simpa using congrFun hu r.gate) (by simp)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (gate_key p x _
    (by simpa using hdata) (by simp [x,loaded]) (by simp [x,loaded]; rfl))),stepStmts.nil]
  rfl

def valueState (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (f : StateValue p) (u : BlockState) : BlockState :=
  let x := loaded p l r a padding c
  let s1 := u.setReg "p_g" .blockPtr [p.BT,p.BV] (valuePointer p l r.gate c)
  let s2 := s1.setReg "p_gn" .blockPtr [p.BV]
    (endpointPointer p l r.gate c p.valueHead p.valueFeature p.V p.BV (l.value*p.BV))
  let s3 := s2.setReg "b_gn" .real [p.BV] (Tile.vec (fun i => some (x.valueEnd i)))
  let s4 := s3.setReg "b_h" .real [p.BK,p.BV] (stateTile p (scaled x Bool.false f))
  let s5 := s4.setReg "b_g" .real [p.BT,p.BV] (Tile.mat (fun i j => some (x.valueGate i j)))
  s5.setReg "b_v" .real [p.BT,p.BV] (Tile.mat (fun i j => some (gatedValue x Bool.false i j)))

theorem value_executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (f : StateValue p) (u : BlockState)
    (hi : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hcarry : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f))
    (hdata : u.regs .real [p.BT,p.BV] "b_v" =
      some (Tile.mat (fun i j => some ((loaded p l r a padding c).value i j))))
    (hm : ∀ off, u.readMem r.gate off = a.gate off) (hu : u.undef = padding) :
    stepStmts (RealHProgram.gateValue p r) u = some (valueState p l r a padding c f u) := by
  let x := loaded p l r a padding c
  unfold RealHProgram.gateValue
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_eval p l r.gate c u hh hi ht))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (endpoint_eval p l r.gate c
    p.valueHead p.valueFeature p.V p.BV l.value "i_v" _
    (by simp [hh]) (by simp [ht]) (by simp [hi])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (endpoint_load p l r.gate a.gate
    (padding r.gate) c p.valueHead p.valueFeature p.V p.BV (l.value*p.BV) "p_gn" _
    (by simpa using hm) (by simpa using congrFun hu r.gate) (by simp)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (scale_value p x f _
    (by simp [hcarry]) (by simp [x,loaded]; rfl)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_load p l r.gate a.gate
    (padding r.gate) c "p_g" _ (by simpa using hm)
    (by simpa using congrFun hu r.gate) (by simp)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (gate_value p x _
    (by simpa using hdata) (by simp [x,loaded]) (by simp [x,loaded]; rfl))),stepStmts.nil]
  rfl

def output (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (gateK : Bool)
    (f : StateValue p) (u : BlockState) : BlockState :=
  if gateK then keyState p l r a padding c f u else valueState p l r a padding c f u

structure Result (p : Layout) (x : Loaded p) (gateK : Bool) (f : StateValue p)
    (u v : BlockState) : Prop where
  carry : v.regs .real [p.BK,p.BV] "b_h" = some (stateTile p (scaled x gateK f))
  key : v.regs .real [p.BK,p.BT] "b_k" = some (Tile.mat (fun e t => some (gatedKey x gateK e t)))
  value : v.regs .real [p.BT,p.BV] "b_v" = some (Tile.mat (fun t j => some (gatedValue x gateK t j)))
  keyIndex : v.regs .nat [] "i_k" = u.regs .nat [] "i_k"
  valueIndex : v.regs .nat [] "i_v" = u.regs .nat [] "i_v"
  headIndex : v.regs .nat [] "i_bh" = u.regs .nat [] "i_bh"
  timeIndex : v.regs .nat [] "i_t" = u.regs .nat [] "i_t"
  mem : v.mem = u.mem
  pids : v.pids = u.pids
  undef : v.undef = u.undef

theorem output_result (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (gateK : Bool)
    (f : StateValue p) (u : BlockState)
    (hk : u.regs .real [p.BK,p.BT] "b_k" =
      some (Tile.mat (fun e t => some ((loaded p l r a padding c).key e t))))
    (hv : u.regs .real [p.BT,p.BV] "b_v" =
      some (Tile.mat (fun t j => some ((loaded p l r a padding c).value t j)))) :
    Result p (loaded p l r a padding c) gateK f u (output p l r a padding c gateK f u) := by
  cases gateK <;> constructor
  all_goals first | rfl | simp [output,keyState,valueState,gatedKey,gatedValue,hk,hv]

theorem executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (gateK : Bool)
    (f : StateValue p) (u : BlockState)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hcarry : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f))
    (hk : u.regs .real [p.BK,p.BT] "b_k" =
      some (Tile.mat (fun e t => some ((loaded p l r a padding c).key e t))))
    (hv : u.regs .real [p.BT,p.BV] "b_v" =
      some (Tile.mat (fun t j => some ((loaded p l r a padding c).value t j))))
    (hm : ∀ off, u.readMem r.gate off = a.gate off) (hu : u.undef = padding) :
    stepStmt (.ifThenElse (.constBool gateK) (gateKey p r) (RealHProgram.gateValue p r)) u =
      some (output p l r a padding c gateK f u) := by
  cases gateK
  · simpa only [stepStmt,evalOp,Option.bind_eq_bind,Option.bind_some,
      Tile.scalar,Bool.false_eq_true,↓reduceIte,output] using
      value_executes p l r a padding c f u hiv hh ht hcarry hv hm hu
  · simpa only [stepStmt,evalOp,Option.bind_eq_bind,Option.bind_some,
      Tile.scalar,↓reduceIte,output] using
      key_executes p l r a padding c f u hik hh ht hcarry hk hm hu

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHGate
