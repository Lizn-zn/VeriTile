import bench.tritonbench_g.context_attn_nopad.Proofs.Frames
import bench.tritonbench_g.context_attn_nopad.Proofs.LoopState
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.Meaning
open VeriTile.Triton Formula Program
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.NumericOps
noncomputable section
set_option maxHeartbeats 1100000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

theorem read_real (u v : BlockState) (h : u.mem = v.mem) (r : RegionName) (off : Nat) :
    u.readMem r off = v.readMem r off := by unfold BlockState.readMem; rw [h]

theorem pointer_address (row headStride dimStride head d j : Nat) (start sn : Int)
    (h : 0 ≤ start+sn) :
    j*row+head*headStride+d*dimStride+((start+sn)*row).toNat =
      ((start+sn+j)*row+head*headStride+d*dimStride).toNat := by
  have hn : 0 ≤ (start+sn)*(row : Int) := mul_nonneg h (Int.natCast_nonneg _)
  have hc : (((start+sn)*row).toNat : Int) = (start+sn)*row := by omega
  have he : ((j*row+head*headStride+d*dimStride+((start+sn)*row).toNat : Nat) : Int) =
      (start+sn+j)*row+head*headStride+d*dimStride := by
    simp only [Nat.cast_add,Nat.cast_mul,hc]
    ring
  simpa only [Int.toNat_natCast] using congrArg Int.toNat he

theorem key_matches (p : Layout) (r : Regions) (u initial : BlockState) (c : Nat)
    (hm : u.mem = initial.mem)
    (hs : 0 ≤ (inputs initial r).start (launch initial).batch+(c*p.BN : Nat)) :
    KeysExecution.loaded p r (launch initial).head
      ((inputs initial r).start (launch initial).batch) (c*p.BN)
      (Formula.sequence p (launch initial) (inputs initial r)) u = LoopState.key p r initial c := by
  apply Tile.ext
  rintro ⟨d,j,⟨⟩⟩
  have ha := pointer_address p.kRow p.kHead p.kDim (launch initial).head
    d.val j.val ((inputs initial r).start (launch initial).batch) (c*p.BN) hs
  simp only [KeysExecution.loaded,KeysExecution.address,LoopState.key,Tile.mat_data,Formula.key]
  rw [ha]
  simp only [read_real u initial hm,inputs,Nat.cast_add,Nat.cast_mul,Int.natCast_div]
  congr 3 <;> ring

theorem value_matches (p : Layout) (r : Regions) (u initial : BlockState) (c : Nat)
    (hm : u.mem = initial.mem)
    (hs : 0 ≤ (inputs initial r).start (launch initial).batch+(c*p.BN : Nat)) :
    ValuesExecution.loaded p r (launch initial).head
      ((inputs initial r).start (launch initial).batch) (c*p.BN)
      (Formula.sequence p (launch initial) (inputs initial r)) u = LoopState.value p r initial c := by
  apply Tile.ext
  rintro ⟨j,d,⟨⟩⟩
  have ha := pointer_address p.vRow p.vHead p.vDim (launch initial).head
    d.val j.val ((inputs initial r).start (launch initial).batch) (c*p.BN) hs
  simp only [ValuesExecution.loaded,ValuesExecution.address,LoopState.value,Tile.mat_data,Formula.value]
  rw [ha]
  simp only [read_real u initial hm,inputs,Nat.cast_add,Nat.cast_mul,Int.natCast_div]
  congr 3 <;> ring

theorem scores_match (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (initial : BlockState) (c : Nat) :
    ScoresExecution.masked p swapped scale (LoopState.rows p initial)
      (c*p.BN)
      (LoopState.query p r initial) (LoopState.key p r initial c) = LoopState.score p r initial scale c := by
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  have hdot := ctxg_dot_row (LoopState.query p r initial) (LoopState.key p r initial c) i j
    (fun d => Formula.query p (launch initial) (inputs initial r) i.val d.val)
    (fun d => Formula.key p (launch initial) (inputs initial r) (c*p.BN+j.val) d.val)
    (by intros; rfl) (by intros; rfl)
  have hcmp : ((c : Int)*(p.BN : Int)+(j.val : Int) ≤ (queryRow p (launch initial) i.val : Int)) ↔
      c*p.BN+j.val ≤ queryRow p (launch initial) i.val := by
    constructor <;> intro h <;> exact_mod_cast h
  cases swapped
  all_goals simp only [ScoresExecution.masked,ScoresExecution.scaled,ScoresExecution.added,ScoresExecution.zero,Bool.false_eq_true,↓reduceIte,
    Tile.select_data,ScoresExecution.mask,Tile.mat_data,Tile.bop_data,Tile.scalar_data,
    Broadcast.leftIndex,Broadcast.rightIndex,LoopState.rows,LoopState.score,
    FormulaStep.maskedScore,Formula.score,Nat.cast_add,Nat.cast_mul,hdot,NumericDType.add,WithBot.realAdd,NumericDType.mul,WithBot.realMul,
    Option.map₂,Option.bind,Option.map,decide_eq_true_eq]
  all_goals split_ifs <;> simp_all [mul_comm]
  all_goals omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.Meaning
