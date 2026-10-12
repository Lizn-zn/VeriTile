import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealProgram

/-! A complete-source arithmetic witness, starting with empty registers and
signed metadata. It exercises nonzero program IDs and slot, the N/splits
calculation, nonunit strides, both input-load choices, both cast choices,
two output blocks, accumulation and the masked output tail. It is a real
block-state witness; the flat-allocation contract is still separate. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealExecution
open VeriTile.Triton RealLoop RealSetupSource
noncomputable section
set_option maxHeartbeats 1800000
set_option linter.unusedSimpArgs false

def parameters : Parameters where
  input := "X"
  weights := "W"
  output := "Y"
  indices := "I"
  N := 5
  K := 3
  inputBatchStride := 5
  inputStride := 2
  slotStride := 5
  rowStride := 7
  colStride := 2
  outputBatchStride := 2
  outputStride := 3
  slice := 2
  BN := 2
  BK := 4
  splits := 2

private def vector (o : Nat) : ℝ :=
  if o = 5 then 2 else if o = 7 then -1 else if o = 9 then 3 else 99

def initial (slot : Int) : BlockState where
  mem := fun r o =>
    if r = "I" then ⟨.int, slot⟩ else .real
      (if r = "X" then vector o else if r = "W" then (o : ℝ)-21 else 100+o)
  regs := fun _ _ _ => none
  pids := fun _ => 1

private theorem read_input (slot : Int) (o : Nat) :
    (initial slot).readMem "X" o = vector o := by rfl
private theorem read_weight (slot : Int) (o : Nat) :
    (initial slot).readMem "W" o = (o : ℝ)-21 := by rfl
private theorem read_output (slot : Int) (o : Nat) :
    (initial slot).readMem "Y" o = 100+(o : ℝ) := by rfl

private theorem formula_values (add : Bool) (g : Fin 3) :
    RealProgram.formula parameters (initial 1) 1 add g.val =
      if add then 141+31*(g.val : ℝ) else 30+28*(g.val : ℝ) := by
  simp only [RealProgram.formula, parameters, Parameters.length, read_input, read_weight, read_output]
  change (let v : ℝ := ∑ k : Fin 3, vector (5+k.val*2) *
      (((5+1*((5+2-1)/2)*7+g.val*7+k.val*2 : Nat) : ℝ)-21)
    if add then v+(100+((2+1*((5+2-1)/2)+2*3+g.val*3 : Nat) : ℝ)) else v) = _
  cases add <;> fin_cases g <;> norm_num [Fin.sum_univ_succ, vector]

/-- The complete actual source executes from empty registers. For both
versions, all precision/cast choices and both input-load and accumulation
flags, output is [141,172,203] when accumulating and [30,58,86] otherwise.
Every cell except Y[11], Y[14], Y[17] is preserved, including masked Y[20]. -/
theorem executes (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (out_dtype : FloatDType) (cast swapped even add : Bool) :
    ∃ final, exec (parameters.source input weight out_dtype cast swapped even add).eraseDType.toAlgKernel
        (initial 1) = some final ∧
      (∀ g : Fin 3, final.readMem "Y" (11+g.val*3) =
        if add then 141+31*(g.val : ℝ) else 30+28*(g.val : ℝ)) ∧
      (∀ (r : RegionName) o, (r ≠ "Y" ∨ (o ≠ 11 ∧ o ≠ 14 ∧ o ≠ 17)) →
        final.mem r o = (initial 1).mem r o) := by
  obtain ⟨final, he, hv, hf⟩ := RealProgram.source_executes parameters input weight out_dtype
    cast swapped even add (initial 1) 1 (by rfl) (by decide) (by decide)
    (by decide) (by decide) (by decide)
  refine ⟨final, he, ?_, ?_⟩
  · intro g
    exact (hv g.val g.isLt).trans (formula_values add g)
  · intro r o hm
    apply hf
    intro g hg hbad
    rcases hm with hr | ho
    · exact hr hbad.1.symm
    · have heq : 11+g*3 = o := hbad.2
      have hg' : g < 3 := hg
      omega

theorem sentinel_skips (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (out_dtype : FloatDType) (cast swapped even add : Bool) :
    ∃ final, exec (parameters.source input weight out_dtype cast swapped even add).eraseDType.toAlgKernel
        (initial (-1)) = some final ∧ final.mem = (initial (-1)).mem := by
  exact RealProgram.sentinel_skips parameters input weight out_dtype cast swapped even add
    (initial (-1)) (by rfl)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealExecution
