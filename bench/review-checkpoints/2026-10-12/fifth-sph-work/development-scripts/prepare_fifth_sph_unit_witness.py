from pathlib import Path
p=Path('/tmp/FifthSphDraft/TypedExecution.lean');s=p.read_text().replace('import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel\n','').replace('open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)\n','')
marker='set_option linter.unusedSimpArgs false\n'
model='''
/-- An abstract singleton-valued backend. Storage tags, integer addresses and
masks remain observable; numerical operations and casts return the sole value.
There are no FP atoms to satisfy. This only witnesses nonempty typed execution,
not physical rounding, numerical accuracy or the real polynomial result. -/
def model : Algebra PUnit where
  literal := fun _ _ _ => PUnit.unit
  negInf := PUnit.unit
  binary := fun _ _ _ _ _ => PUnit.unit
  atomicAdd := fun _ => some (fun _ _ => PUnit.unit)
  unary := fun _ _ _ => PUnit.unit
  compareLe := fun _ _ => some (fun _ _ => Bool.true)
  compareLt := fun _ _ => some (fun _ _ => Bool.false)
  compareEq := fun _ _ => some (fun _ _ => Bool.true)
  cast := fun _ _ _ _ => PUnit.unit
  fromNat := fun _ _ => PUnit.unit
  fromInt := fun _ _ => PUnit.unit
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => PUnit.unit
  reduceSum := fun _ {_} _ _ _ _ => PUnit.unit
  reduceSumTyped := fun _ d {_} _ _ _ => some (fun _ => ofFloat d PUnit.unit)
  scan := fun _ {_} _ _ _ v => some v
  dot := fun _ {_} {_} {_} {_} _ _ => some (fun _ => PUnit.unit)
  dotAcc := fun _ _ _ out _ {_} {_} {_} {_} _ _ _ => some (fun _ => ofFloat out PUnit.unit)
  dotTyped := fun _ _ _ out _ {_} {_} {_} {_} _ _ => some (fun _ => ofFloat out PUnit.unit)

'''
s=s.replace(marker,marker+model).replace('State ℝ','State PUnit').replace('(storageType input) 2','(storageType input) PUnit.unit').replace('(storageType upstream) 1','(storageType upstream) PUnit.unit').replace('(storageType destination) 5','(storageType destination) PUnit.unit').replace('    ℝ model','    PUnit model')
p.write_text(s)
p=Path('/tmp/FifthSphDraft/README.md');s=p.read_text();s+='\nTyped nonempty-execution witnesses use an explicitly singleton-valued abstract\nalgebra: integer addresses, masks and storage tags remain visible; numerical\noperations are collapsed to one value. With no FP atoms this establishes\ninhabited typed execution only. Numerical polynomial outputs are checked by the\nseparate real finite-memory witnesses; no physical rounding claim is made.\n';p.write_text(s)
