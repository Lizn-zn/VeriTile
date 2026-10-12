import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardFlat
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardBounds

/-! Complete flat runs of both sources. Nonzero allocation bases expose the
final dead-pointer mismatch at head zero. DL aliases S; a nonzero initial
padding oracle, final unwritten slices, and tagged neighboring cells are kept.
The one-chunk fixture allocates no DI or DG storage. -/
namespace VeriTile.Tests.ChunkGateBackwardFlat
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkGateRecurrence RealProgram RealBackwardPointers
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
open scoped BigOperators

private abbrev p : Layout := ⟨3,4,4,2,2⟩
private def u : BlockState where
  mem := fun r off =>
    if r = "S" then MemCell.real ((off / 16 + 1 + off % 16 / 4 + off % 4 : Nat) : ℝ)
    else if r = "D" then MemCell.real 2
    else if r = "DS" then MemCell.real ((off / 16 + 1 : Nat) : ℝ)
    else MemCell.of .nat 77
  regs := fun _ _ _ => none
  pids := fun _ => 0
  undef := fun _ _ => 7

private def a : Inputs p where
  source := fun c e j => ((c+1+e.val+j.val : Nat) : ℝ)
  gate := fun _ => 2
  seed := fun _ _ => 0
  incoming := fun c _ _ => ((c+1 : Nat) : ℝ)

private theorem source_values : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N-1 →
    u.readMem "S" (address p (launch u) t e.val j.val) = a.source t e j := by
  intro t e j ht
  change t < 2 at ht
  interval_cases t <;> fin_cases e <;> fin_cases j <;>
    simp [BlockState.readMem,u,address,launch,p,a,String.reduceEq]

private theorem incoming_values : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
    u.readMem "DS" (address p (launch u) t e.val j.val) = a.incoming t e j := by
  intro t e j ht
  change t < 3 at ht
  interval_cases t <;> fin_cases e <;> fin_cases j <;>
    simp [BlockState.readMem,u,address,launch,p,a,String.reduceEq]

private theorem gate_values : ∀ t, t < p.N → u.readMem "D" (gateAddress p (launch u) t) = a.gate t := by
  intro t _
  simp [BlockState.readMem,u,a,String.reduceEq]

private theorem suffix_two (e : Fin p.BK) (j : Fin p.BV) : RealMath.reverse a 3 2 e j = 3 := by
  rw [RealMath.reverse_step a 3 2 (by decide),RealMath.reverse_end]
  norm_num [a]

private theorem suffix_one (e : Fin p.BK) (j : Fin p.BV) : RealMath.reverse a 3 1 e j = 8 := by
  rw [RealMath.reverse_step a 3 1 (by decide),suffix_two]
  norm_num [a]

private theorem suffix_zero (e : Fin p.BK) (j : Fin p.BV) : RealMath.reverse a 3 0 e j = 17 := by
  rw [RealMath.reverse_step a 3 0 (by decide),suffix_one]
  norm_num [a]

private theorem gradient_zero : RealMath.gateGradient a 3 0 = 64 := by
  simp only [RealMath.gateGradient,show 0+1=1 from rfl,suffix_one]
  norm_num [p,a,Fin.sum_univ_succ]

private theorem gradient_one : RealMath.gateGradient a 3 1 = 36 := by
  simp only [RealMath.gateGradient,show 1+1=2 from rfl,suffix_two]
  norm_num [p,a,Fin.sum_univ_succ]


private def A : FlatAlloc where
  flat := "flat"
  regions := ["S","D","DI","DG","DS"]
  base := fun r => if r = "S" then 100 else if r = "D" then 200 else if r = "DI" then 300
    else if r = "DG" then 400 else if r = "DS" then 500 else 0
  extent := fun r => if r ∈ (["S","D","DI","DG","DS"] : List RegionName) then 48 else 0

private theorem disjoint : A.Disjoint := by
  intro r hr s hs hne
  simp [A] at hr hs
  rcases hr with rfl | rfl | rfl | rfl | rfl <;>
    rcases hs with rfl | rfl | rfl | rfl | rfl <;>
    (first | exact (hne rfl).elim | decide)

private theorem covered : ∀ r, r ∉ A.regions → A.extent r = 0 := by
  intro r h
  exact if_neg h

private theorem bounds : RealBackwardSafety.Bounds p (launch u) "S" "D" "DI" "DG" "S" "DS" A.extent := by
  apply RealBackwardBounds.sufficient
  all_goals norm_num [p,launch,u,numK,numV,A,String.reduceEq]

private theorem untouched (off : Nat) (h : off = 332 ∨ off = 408 ∨ off = 302) :
    ¬ RealBackwardFlat.footprint A p (launch u) "DI" "DG" "S" (A.flat,off) := by
  rintro ⟨_,r,o,hw,heq⟩
  rcases hw with (⟨rfl,t,e,j,ht,rfl⟩ | ⟨rfl,t,ht,rfl⟩) | ⟨rfl,e,j,rfl⟩
  · have he := e.isLt
    have hj := j.isLt
    norm_num [FlatAlloc.addr,A,address,p,launch,u,String.reduceEq] at heq ht he hj
    simp only [Region.mk.injEq,String.reduceEq,↓reduceIte] at heq
    omega
  · norm_num [FlatAlloc.addr,A,gradientAddress,numK,numV,p,launch,u,String.reduceEq] at heq ht
    simp only [Region.mk.injEq,String.reduceEq,↓reduceIte] at heq
    omega
  · have he := e.isLt
    have hj := j.isLt
    norm_num [FlatAlloc.addr,A,seedAddress,p,launch,u,String.reduceEq] at heq he hj
    simp only [Region.mk.injEq,String.reduceEq,↓reduceIte] at heq
    omega

theorem complete_flat (swapped : Bool) : ∃ v : BlockState,
    exec (A.flattenKernel
      (RealBackwardFlat.source swapped p "S" "D" "DI" "DG" "S" "DS" .fp16 .bf16 .fp32 1).eraseDType.toAlgKernel)
      (A.flattenState u) = some v ∧
    v.readMem A.flat 300 = 8 ∧ v.readMem A.flat 316 = 3 ∧
    v.readMem A.flat 400 = 64 ∧ v.readMem A.flat 404 = 36 ∧
    v.readMem A.flat 100 = 17 ∧
    v.mem A.flat 332 = MemCell.of .nat 77 ∧
    v.mem A.flat 408 = MemCell.of .nat 77 ∧
    v.mem A.flat 302 = MemCell.of .nat 77 := by
  obtain ⟨v,he,hi,hg,hl,hf⟩ := RealBackwardFlat.source_executes A disjoint covered p
    "S" "D" "DI" "DG" "S" "DS" a swapped .fp16 .bf16 .fp32 1 u
    (by decide) (by decide) (by decide) (by decide) (by decide)
    (by constructor <;> decide) (by decide) (by decide) source_values incoming_values gate_values bounds
  refine ⟨v,he,(hi 0 0 0 (by decide)).trans (suffix_one 0 0),
    (hi 1 0 0 (by decide)).trans (suffix_two 0 0),
    (hg 0 (by decide)).trans gradient_zero,(hg 1 (by decide)).trans gradient_one,
    (hl 0 0).trans (suffix_zero 0 0),?_,?_,?_⟩
  · rw [← hf A.flat 332 (untouched 332 (by simp))]
    change (A.flattenState u).mem A.flat (A.addr "DI" 32) = _
    rw [A.flattenState_mem_addr disjoint u (by simp [A]) (by decide)]
    rfl
  · rw [← hf A.flat 408 (untouched 408 (by simp))]
    change (A.flattenState u).mem A.flat (A.addr "DG" 8) = _
    rw [A.flattenState_mem_addr disjoint u (by simp [A]) (by decide)]
    rfl
  · rw [← hf A.flat 302 (untouched 302 (by simp))]
    change (A.flattenState u).mem A.flat (A.addr "DI" 2) = _
    rw [A.flattenState_mem_addr disjoint u (by simp [A]) (by decide)]
    rfl

/-- Final dead-pointer arithmetic really differs after relocation; the complete
flat theorem above therefore cannot rely on equality of every final register. -/
theorem final_dead_pointer_differs :
    Tile.ptrSub .scalarR (A.trTile (backwardPointer p (launch u) "S" 1 1)) (Tile.scalar 16) ≠
      A.trTile (backwardPointer p (launch u) "S" 1 2) := by
  intro h
  have hv := congrArg (fun t : Tile .ptr [2,2] => t.data (0,0,PUnit.unit)) h
  norm_num [Tile.ptrSub,Broadcast.leftIndex,Broadcast.rightIndex,FlatAlloc.trTile,
    FlatAlloc.trCarrier,FlatAlloc.addr,backwardPointer,address,p,launch,u,A,String.reduceEq] at hv


private def B : FlatAlloc := { A with
  regions := ["S","D","DS"]
  extent := fun r => if r ∈ (["S","D","DS"] : List RegionName) then
    if r = "D" then 1 else 16 else 0 }

private theorem single_disjoint : B.Disjoint := by
  intro r hr s hs hne
  simp [B] at hr hs
  rcases hr with rfl | rfl | rfl <;> rcases hs with rfl | rfl | rfl <;>
    (first | exact (hne rfl).elim | decide)

private theorem single_covered : ∀ r, r ∉ B.regions → B.extent r = 0 := by
  intro r h
  exact if_neg h

/-- The actual empty-loop source still emits DL, with no DI/DG allocation and
with every other flat cell unchanged, including its dtype tag. -/
theorem single_chunk (swapped : Bool) : ∃ v : BlockState,
    exec (B.flattenKernel
      (RealBackwardFlat.source swapped ⟨1,4,4,2,2⟩ "S" "D" "DI" "DG" "S" "DS" .fp16 .bf16 .fp32 1).eraseDType.toAlgKernel)
      (B.flattenState u) = some v ∧
    v.readMem B.flat 100 = 1 ∧
    (∀ off, off ≠ 100 → off ≠ 101 → off ≠ 104 → off ≠ 105 →
      v.mem B.flat off = (B.flattenState u).mem B.flat off) := by
  let q : Layout := ⟨1,4,4,2,2⟩
  have hb : RealBackwardSafety.Bounds q (launch u) "S" "D" "DI" "DG" "S" "DS" B.extent := by
    apply RealBackwardBounds.sufficient
    all_goals norm_num [q,launch,u,numK,numV,B,A,String.reduceEq]
    all_goals simp [String.reduceEq]
  obtain ⟨v,he,_,_,hl,hf⟩ := RealBackwardFlat.source_executes B single_disjoint single_covered q
    "S" "D" "DI" "DG" "S" "DS" a swapped .fp16 .bf16 .fp32 1 u
    (by decide) (by decide) (by decide) (by decide) (by decide)
    (by constructor <;> decide) (by decide) (by decide)
    (by intro t e j ht; exact False.elim (Nat.not_lt_zero t ht))
    (by
      intro t e j ht
      have ht0 : t = 0 := by change t < 1 at ht; omega
      subst t
      fin_cases e <;> fin_cases j <;>
        simp [BlockState.readMem,u,address,launch,q,a,String.reduceEq])
    (by intro t ht; simp [BlockState.readMem,u,a,String.reduceEq]) hb
  refine ⟨v,he,?_,?_⟩
  · simpa [FlatAlloc.addr,B,A,q,seedAddress,launch,u,RealMath.reverse_single,a,String.reduceEq] using hl 0 0
  · intro off h0 h1 h4 h5
    apply Eq.symm
    apply hf
    rintro ⟨_,r,o,hw,heq⟩
    rcases hw with (⟨rfl,t,e,j,ht,rfl⟩ | ⟨rfl,t,ht,rfl⟩) | ⟨rfl,e,j,rfl⟩
    · exact Nat.not_lt_zero t ht
    · exact Nat.not_lt_zero t ht
    · fin_cases e <;> fin_cases j <;>
        norm_num [FlatAlloc.addr,B,A,seedAddress,q,launch,u,String.reduceEq] at heq <;> omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.ChunkGateBackwardFlat
