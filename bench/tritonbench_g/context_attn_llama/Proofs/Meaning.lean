import bench.tritonbench_g.context_attn_llama.Proofs.Frames
import bench.tritonbench_g.context_attn_llama.Proofs.LoopState
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.Meaning
open VeriTile.Triton Formula Program CellMath
noncomputable section
set_option maxHeartbeats 1100000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

theorem read_real (u v : BlockState) (h : u.mem = v.mem) (r : RegionName) (off : Nat) :
    u.readMem r off = v.readMem r off := by unfold BlockState.readMem; rw [h]

theorem read_int (u v : BlockState) (h : u.mem = v.mem) (r : RegionName) (off : Nat) :
    u.readMemValue .int r off = v.readMemValue .int r off := by
  simp only [BlockState.readMemValue,BlockState.readMemTyped,h]

theorem pages_match (p : Layout) (r : Regions) (u initial : BlockState) (c : Nat)
    (hm : u.mem = initial.mem) (j : Fin p.BN) :
    KeysExecution.pages p r ((inputs initial r).request (Formula.batch p (launch initial)))
      (c*p.BN) (Formula.limit p (launch initial) (inputs initial r)) u j.val =
      LoopState.pages p r initial c j := by
  simp only [KeysExecution.pages,LoopState.pages,Formula.page,Nat.cast_add,Nat.cast_mul,
    inputs,read_int u initial hm]

theorem key_matches (p : Layout) (r : Regions) (u initial : BlockState) (c : Nat)
    (hm : u.mem = initial.mem) :
    KeysExecution.loaded p r (Formula.head p (launch initial)/p.kvGroup)
      ((inputs initial r).request (Formula.batch p (launch initial)))
      (c*p.BN) (Formula.limit p (launch initial) (inputs initial r)) u = LoopState.key p r initial c := by
  apply Tile.ext
  rintro ⟨d,j,⟨⟩⟩
  simp only [KeysExecution.loaded,LoopState.key,Tile.mat_data,Formula.key,KeysExecution.address]
  rw [pages_match p r u initial c hm j]
  simp only [LoopState.pages,read_real u initial hm,inputs,Nat.cast_add,Nat.cast_mul,Int.natCast_div]

theorem value_matches (p : Layout) (r : Regions) (u initial : BlockState) (c : Nat)
    (hm : u.mem = initial.mem) :
    ValuesExecution.loaded p r (Formula.head p (launch initial)/p.kvGroup)
      (LoopState.pages p r initial c) (c*p.BN)
      (Formula.limit p (launch initial) (inputs initial r)) u = LoopState.value p r initial c := by
  apply Tile.ext
  rintro ⟨j,d,⟨⟩⟩
  simp only [ValuesExecution.loaded,LoopState.value,Tile.mat_data,Formula.value,
    ValuesExecution.address,LoopState.pages,read_real u initial hm,inputs,Nat.cast_add,Nat.cast_mul,Int.natCast_div]

theorem scores_match (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (initial : BlockState) (c : Nat) :
    ScoresExecution.result p swapped scale (LoopState.rows p initial)
      ((inputs initial r).prompt (Formula.batch p (launch initial))) (c*p.BN)
      (LoopState.query p r initial) (LoopState.key p r initial c) = LoopState.score p r initial scale c := by
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  have hdot := ctxg_dot_row (LoopState.query p r initial) (LoopState.key p r initial c) i j
    (fun d => Formula.query p (launch initial) (inputs initial r) i.val d.val)
    (fun d => Formula.key p (launch initial) (inputs initial r) (c*p.BN+j.val) d.val)
    (by intros; rfl) (by intros; rfl)
  cases swapped
  all_goals simp only [ScoresExecution.result,ScoresExecution.scaled,Bool.false_eq_true,↓reduceIte,
    Tile.select_data,ScoresExecution.mask,Tile.mat_data,Tile.bop_data,Tile.scalar_data,
    Broadcast.leftIndex,Broadcast.rightIndex,LoopState.rows,LoopState.score,
    Formula.score,Nat.cast_add,Nat.cast_mul,hdot,NumericDType.mul,WithBot.realMul,
    Option.map₂,Option.bind,Option.map,decide_eq_true_eq]
  all_goals split_ifs <;> simp_all [mul_comm]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.Meaning
