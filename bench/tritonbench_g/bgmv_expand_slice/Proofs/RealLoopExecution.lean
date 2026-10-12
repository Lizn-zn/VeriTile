import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealLoop

/-! A nonempty arithmetic witness for the whole output loop, with two blocks,
a masked output tail, nonunit strides, negative input, and a nonzero padded K
lane. The accumulating output is 141,172,203; the non-accumulating output is
30,58,86. This checks the real loop theorem, not the pending complete flat IO. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealLoopExecution
open VeriTile.Triton RealLoop
noncomputable section
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false

private def geometry : Geometry where
  weights := "W"
  output := "Y"
  weightBase := 5
  rowStride := 7
  colStride := 2
  outputBase := 11
  outputStride := 3
  K := 3
  length := 3
  BN := 2
  BK := 4

private def input (k : Fin 4) : ℝ :=
  match k.val with | 0 => 2 | 1 => -1 | 2 => 3 | _ => 99

private def initial : BlockState :=
  let s : BlockState := {
    mem := fun r o => MemCell.real (if r = "W" then (o : ℝ) else 100+o)
    regs := fun _ _ _ => none
    pids := fun _ => 1 }
  let s := s.setReg "split_n_length" .nat [] (Tile.scalar 3)
  let s := s.setReg "offset_n" .nat [2] (Tile.vec (fun i => i.val))
  let s := s.setReg "offset_k" .nat [4] (Tile.vec (fun k => k.val))
  let s := s.setReg "b_ptr" .ptr [] (Tile.scalar ("W",5))
  let s := s.setReg "c_ptr" .ptr [] (Tile.scalar ("Y",11))
  s.setReg "tiled_a" .real [4] ⟨fun i => some (input i.1)⟩

private theorem registers : geometry.Registers input initial := by
  constructor <;> simp [geometry, initial]

private theorem read_weight (o : Nat) : initial.readMem "W" o = (o : ℝ) := by rfl
private theorem read_output (o : Nat) : initial.readMem "Y" o = 100+(o : ℝ) := by rfl

private theorem expected_values (add : Bool) (g : Fin 3) :
    geometry.expected input initial add g.val =
      if add then 141+31*(g.val : ℝ) else 30+28*(g.val : ℝ) := by
  simp only [Geometry.expected, Geometry.outputAddress, geometry, RealBlock.dot,
    read_weight, read_output]
  change (let v : ℝ := ∑ k : Fin 4, input k *
      (if k.val < 3 then ((5+g.val*7+k.val*2 : Nat) : ℝ) else 0)
    if add then v + (100+((11+g.val*3 : Nat) : ℝ)) else v) = _
  cases add <;> fin_cases g <;> norm_num [Fin.sum_univ_succ, input]

/-- Both complete loop versions execute with the stated arithmetic values and
preserve every cell except Y[11], Y[14], Y[17], including the masked Y[20]. -/
theorem executes (add swapped : Bool) :
    ∃ final, stepStmt (geometry.statement add swapped) initial = some final ∧
      (∀ g : Fin 3, final.readMem "Y" (11+g.val*3) =
        if add then 141+31*(g.val : ℝ) else 30+28*(g.val : ℝ)) ∧
      (∀ (r : RegionName) o, (r ≠ "Y" ∨ (o ≠ 11 ∧ o ≠ 14 ∧ o ≠ 17)) → final.mem r o = initial.mem r o) := by
  obtain ⟨final, he, hv, hf⟩ := geometry.executes input initial add swapped
    (by decide) (by decide) (by decide) registers
  refine ⟨final, he, ?_, ?_⟩
  · intro g
    exact (hv g.val g.isLt).trans (expected_values add g)
  · intro r o hm
    apply hf
    intro g hg hbad
    rcases hm with hr | ho
    · exact hr hbad.1.symm
    · have heq : 11+g*3 = o := hbad.2
      have hg' : g < 3 := hg
      omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealLoopExecution
