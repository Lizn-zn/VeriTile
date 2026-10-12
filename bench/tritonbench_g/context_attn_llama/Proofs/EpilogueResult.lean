import bench.tritonbench_g.context_attn_llama.Proofs.WholeExecution
import VeriTile.Triton.KernelLemmas.ScatterStore
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.EpilogueResult
open VeriTile.Triton Formula Program EpilogueExecution
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 3000

theorem readback (p : Layout) (r : Regions) (head : Nat) (start seq : Int)
    (rows : Fin p.BM → Nat) (v : Fin p.BM → Fin p.D → ℝ) (den : Fin p.BM → ℝ) (u : BlockState)
    (i : TileIndex [p.BM,p.D]) (hi : active p seq rows i)
    (hinj : ∀ j, active p seq rows j → address p head start rows j = address p head start rows i → j = i) :
    (prepared p r head start seq rows v den u).readMem r.out (address p head start rows i) = v i.1 i.2.1 / den i.1 := by
  exact BlockState.scatter_readback_prop_masked_nd_of_true
    (addresses p r head start rows (normalized p v den u)) (address p head start rows)
    (fun j => v j.1 j.2.1 / den j.1) (fun j => active p seq rows j = Bool.true) i hi hinj

theorem frame (p : Layout) (r : Regions) (head : Nat) (start seq : Int)
    (rows : Fin p.BM → Nat) (v : Fin p.BM → Fin p.D → ℝ) (den : Fin p.BM → ℝ) (u : BlockState)
    (R : RegionName) (off : Nat)
    (hnot : ∀ i, active p seq rows i → ¬(r.out = R ∧ address p head start rows i = off)) :
    (prepared p r head start seq rows v den u).mem R off = u.mem R off := by
  unfold prepared stored
  rw [foldl_store_preserve_cell (address p head start rows) (fun j => v j.1 j.2.1 / den j.1)
    (fun j => active p seq rows j = Bool.true) R off _ _ (fun i _ => hnot i)]
  simp only [addresses,normalized,BlockState.setReg_mem]

def activeLane (p : Layout) (r : Regions) (u : BlockState) (i : TileIndex [p.BM,p.D]) : Prop :=
  (queryRow p (launch u) i.1.val : Int) < Formula.sequence p (launch u) (inputs u r)

def offset (p : Layout) (r : Regions) (u : BlockState) (i : TileIndex [p.BM,p.D]) : Nat :=
  address p (u.pids 1%p.heads) ((inputs u r).start (u.pids 1/p.heads)) (fun i => queryRow p (launch u) i.val) i

def Writes (p : Layout) (r : Regions) (u : BlockState) (R : RegionName) (off : Nat) : Prop :=
  ∃ i, activeLane p r u i ∧ r.out = R ∧ offset p r u i = off

/-- Complete successful execution, active formula readback, and preservation of
all other tagged memory cells. Injectivity is only required for active writes. -/
theorem correct (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) (scale : ℝ)
    (initial : BlockState) (hBN : 0 < p.BN)
    (hinj : ∀ i j, activeLane p r initial i → activeLane p r initial j →
      offset p r initial i = offset p r initial j → i = j) :
    ∃ final,
      stepStmts (kernel p r storage swapped scale).eraseDType.toAlgKernel.body initial = some final ∧
      (∀ i, activeLane p r initial i → final.readMem r.out (offset p r initial i) =
        WholeExecution.output p r initial scale i.1 i.2.1) ∧
      (∀ R off, ¬ Writes p r initial R off → final.mem R off = initial.mem R off) := by
  obtain ⟨mid,hs,he⟩ := WholeExecution.complete p r storage swapped scale initial hBN
  refine ⟨_,he,?_,?_⟩
  · intro i hi
    rw [← WholeExecution.output_agrees]
    apply readback
    · simpa only [active,decide_eq_true_eq] using hi
    · intro j hj heq
      exact hinj j i (by simpa only [active,decide_eq_true_eq] using hj) hi heq
  · intro R off hnot
    rw [frame]
    · exact congrFun (congrFun hs.memory R) off
    · intro i hi heq
      exact hnot ⟨i,by simpa only [active,decide_eq_true_eq] using hi,heq⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.EpilogueResult
