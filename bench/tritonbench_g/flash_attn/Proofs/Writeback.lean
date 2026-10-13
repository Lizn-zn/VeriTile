import bench.tritonbench_g.flash_attn.Proofs.Formula

/-! Independent two-output memory specification. Writes occur in L-then-O
order, with row-major lane enumeration. This retains the source's last-write
behavior even when arbitrary strides alias output addresses. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.Formula
open VeriTile.Triton
noncomputable section

def writeState (p : Layout) (pid₀ pid₁ : Nat) (L O : RegionName)
    (lvalues : Fin p.BLOCK_M → ℝ) (ovalues : TileIndex [p.BLOCK_M,p.DIM] → ℝ)
    (s : BlockState) : BlockState :=
  (TileShape.allIndices [p.BLOCK_M,p.DIM]).foldl
    (fun t i => t.writeMem O (oOffset p pid₀ pid₁ i.1.val i.2.1.val) (ovalues i))
    ((TileShape.allIndices [p.BLOCK_M]).foldl
      (fun t i => t.writeMem L (lOffset p pid₀ pid₁ i.1.val) (lvalues i.1)) s)

def referenceState (p : Layout) (s : BlockState) (Q K V L O : RegionName) : BlockState :=
  writeState p (s.pids 0) (s.pids 1) L O
    (fun i => lse p s.readMem Q K (s.pids 0) (s.pids 1) i.val)
    (fun i => output p s.readMem Q K V (s.pids 0) (s.pids 1) i.1.val i.2.1.val) s

theorem writeState_mem_congr (p : Layout) (pid₀ pid₁ : Nat) (L O : RegionName)
    (ls : Fin p.BLOCK_M → ℝ) (os : TileIndex [p.BLOCK_M,p.DIM] → ℝ)
    (s t : BlockState) (hm : t.mem = s.mem) :
    (writeState p pid₀ pid₁ L O ls os t).mem = (writeState p pid₀ pid₁ L O ls os s).mem := by
  funext r o
  apply BlockState.foldl_writeMem_mem_congr _ _ _ _ (fun _ _ => rfl)
  apply BlockState.foldl_writeMem_mem_congr _ _ _ _ (fun _ _ => rfl)
  exact congrFun (congrFun hm r) o

private theorem scatter_frame {α : Type} (is : List α) (region : RegionName)
    (off : α → Nat) (value : α → ℝ) (s : BlockState) (r : RegionName) (o : Nat)
    (h : ∀ i, ¬(region = r ∧ off i = o)) :
    (is.foldl (fun t i => t.writeMem region (off i) (value i)) s).mem r o = s.mem r o := by
  by_cases hr : r = region
  · subst r
    exact BlockState.foldl_writeMem_mem_preserve_unhit off value is o
      (fun i _ hi => h i ⟨rfl,hi⟩) s
  · exact BlockState.foldl_writeMem_mem_preserve_other_region off value is r hr o s

theorem writeState_frame (p : Layout) (pid₀ pid₁ : Nat) (L O : RegionName)
    (ls : Fin p.BLOCK_M → ℝ) (os : TileIndex [p.BLOCK_M,p.DIM] → ℝ)
    (s : BlockState) (r : RegionName) (o : Nat)
    (h : ¬writeWindow p L O pid₀ pid₁ r o) :
    (writeState p pid₀ pid₁ L O ls os s).mem r o = s.mem r o := by
  unfold writeState
  rw [scatter_frame _ O _ _ _ r o (fun i hi =>
    h (Or.inr ⟨hi.1.symm,i.1,i.2.1,hi.2.symm⟩))]
  exact scatter_frame _ L _ _ s r o (fun i hi =>
    h (Or.inl ⟨hi.1.symm,i.1,hi.2.symm⟩))

theorem referenceState_frame (p : Layout) (s : BlockState) (Q K V L O : RegionName)
    (r : RegionName) (o : Nat) (h : ¬writeWindow p L O (s.pids 0) (s.pids 1) r o) :
    (referenceState p s Q K V L O).mem r o = s.mem r o :=
  writeState_frame p (s.pids 0) (s.pids 1) L O _ _ s r o h

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.Formula
