import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardPost
import VeriTile.Triton.Memory.TraceProofs

/-! The final pointer tail may change only the correspondence of S/DI/DG.
This relation keeps every memory tag and every other register value. The
actual DL postlude preserves it because it never reads those three names. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardFlatState
open VeriTile.Triton RealProgram RealBackwardProgram
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def Live (name : RegName) : Prop := name ≠ "S" ∧ name ≠ "DI" ∧ name ≠ "DG"

structure Agree (s t : BlockState) : Prop where
  mem : s.mem = t.mem
  regs : ∀ d sh name, Live name → s.regs d sh name = t.regs d sh name

theorem Agree.refl (s : BlockState) : Agree s s := ⟨rfl,fun _ _ _ _ => rfl⟩

theorem Agree.setReg {s t : BlockState} (h : Agree s t) (name : RegName)
    (d : TileDType) (sh : TileShape) (v : Tile d sh) :
    Agree (s.setReg name d sh v) (t.setReg name d sh v) := by
  refine ⟨h.mem,?_⟩
  intro d' sh' n hn
  simp only [BlockState.setReg,h.regs d' sh' n hn]

theorem Agree.setDead {s t : BlockState} (h : Agree s t) (name : RegName) (hn : ¬ Live name)
    (d : TileDType) (sh : TileShape) (v w : Tile d sh) :
    Agree (s.setReg name d sh v) (t.setReg name d sh w) := by
  refine ⟨h.mem,?_⟩
  intro d' sh' n hl
  have hne : n ≠ name := by rintro rfl; exact hn hl
  simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne,h.regs d' sh' n hl]

theorem Agree.writeMemTyped {s t : BlockState} (h : Agree s t) (d : TileDType)
    (r : RegionName) (off : Nat) (v : TileCarrier d) :
    Agree (s.writeMemTyped d r off v) (t.writeMemTyped d r off v) := by
  constructor
  · cases d <;> simp [BlockState.writeMemTyped,BlockState.writeMemAs,BlockState.writeMem,h.mem]
  · intro d' sh name hn
    simpa only [BlockState.writeMemTyped_regs] using h.regs d' sh name hn

theorem Agree.scatter {s t : BlockState} (h : Agree s t) {ι : Type} (xs : List ι)
    (r : ι → RegionName) (off : ι → Nat) (v : ι → TileCarrier .real) :
    Agree (xs.foldl (fun u i => u.writeMemTyped .real (r i) (off i) (v i)) s)
      (xs.foldl (fun u i => u.writeMemTyped .real (r i) (off i) (v i)) t) := by
  induction xs generalizing s t with
  | nil => exact h
  | cons i xs ih => exact ih (h.writeMemTyped .real (r i) (off i) (v i))

theorem assign_congr {s t t' : BlockState} (h : Agree s t) {d sh name} (op : Op d sh)
    (he : evalOp op s = evalOp op t) (ht : stepStmt (.assign d sh name op) t = some t') :
    ∃ s', stepStmt (.assign d sh name op) s = some s' ∧ Agree s' t' := by
  obtain ⟨v,hv,rfl⟩ := TraceProofs.assign_inv ht
  exact ⟨s.setReg name d sh v,stepStmt_assign_eq_some (he.trans hv),h.setReg name d sh v⟩

theorem store_congr {s t t' : BlockState} (h : Agree s t) {sh}
    (ptr : Op .ptr sh) (val : Op .real sh)
    (hp : evalOp ptr s = evalOp ptr t) (hv : evalOp val s = evalOp val t)
    (ht : stepStmt (.store .real sh (.ptr ptr) val .none) t = some t') :
    ∃ s', stepStmt (.store .real sh (.ptr ptr) val .none) s = some s' ∧ Agree s' t' := by
  simp only [stepStmt] at ht ⊢
  rw [hp,hv]
  cases hv' : evalOp val t with
  | none => simp [hv'] at ht
  | some values =>
    cases hp' : evalOp ptr t with
    | none => simp [hv',hp'] at ht
    | some ptrs =>
      simp only [hv',hp',Option.bind_some,Option.bind_eq_bind,ite_true] at ht ⊢
      cases ht
      exact ⟨_,rfl,h.scatter (TileShape.allIndices sh) (fun i => (ptrs.data i).1)
        (fun i => (ptrs.data i).2) values.data⟩

theorem flatten_expandDim (A : FlatAlloc) {d : TileDType} {sh : TileShape} (ax : Fin (sh.length+1)) (op : Op d sh) :
    A.flattenOp (.expandDim ax op) = .expandDim ax (A.flattenOp op) := by
  rw [FlatAlloc.flattenOp.eq_def]

theorem seedPointer_equal (A : FlatAlloc) (p : Layout) (DL : RegionName)
    {s t : BlockState} (h : Agree s t) :
    evalOp (A.flattenOp (RealBackwardInit.seedPointerOp p DL)) s =
      evalOp (A.flattenOp (RealBackwardInit.seedPointerOp p DL)) t := by
  have hh := h.regs .nat [] "offset_bh" (by simp [Live])
  have hk := h.regs .nat [] "offset_d" (by simp [Live])
  have hv := h.regs .nat [] "offset_s" (by simp [Live])
  have hek := h.regs .nat [p.BK] "offs_k" (by simp [Live])
  have hev := h.regs .nat [p.BV] "offs_v" (by simp [Live])
  have hfk := flatten_expandDim A (sh := [p.BK]) ⟨1,by simp⟩ (.ref .nat [p.BK] "offs_k")
  have hfv := flatten_expandDim A (sh := [p.BV]) ⟨0,by simp⟩ (.ref .nat [p.BV] "offs_v")
  simp only [RealBackwardInit.seedPointerOp,RealBackwardInit.offsetOp,RealBackwardInit.headOp,
    ↓reduceIte,FlatAlloc.flattenOp]
  erw [hfk,hfv]
  simp only [FlatAlloc.flattenOp]
  simp only [evalOp.eq_def,hh,hk,hv,hek,hev]

theorem load_equal {sh} (name : RegName) (hn : Live name) {s t : BlockState} (h : Agree s t) :
    evalOp (.load .real (.ptr (.ref .ptr sh name)) .none) s =
      evalOp (.load .real (.ptr (.ref .ptr sh name)) .none) t := by
  simp only [evalOp.eq_def,h.regs .ptr sh name hn]
  have hm : ∀ r o, s.readMemValue .real r o = t.readMemValue .real r o := by
    intro r o
    simp only [BlockState.readMemValue_real,BlockState.readMem,h.mem]
  simp only [hm,Option.bind_some,Option.bind_eq_bind,ite_true]

theorem update_equal (p : Layout) {s t : BlockState} (h : Agree s t) :
    evalOp (update p) s = evalOp (update p) t := by
  simp only [update,evalOp.eq_def,h.regs .real [p.BK,p.BV] "Dacc" (by simp [Live]),
    h.regs .real [p.BK,p.BV] "DS_i" (by simp [Live]),h.regs .real [] "d_i" (by simp [Live])]

theorem post_congr (A : FlatAlloc) (p : Layout) (DL : RegionName)
    {s t t' : BlockState} (h : Agree s t)
    (he : stepStmts (A.flattenStmts (RealBackwardPost.code p DL)) t = some t') :
    ∃ s', stepStmts (A.flattenStmts (RealBackwardPost.code p DL)) s = some s' ∧ Agree s' t' := by
  simp only [RealBackwardPost.code,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,
    FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.flattenOp_load] at he ⊢
  simp only [update,FlatAlloc.flattenOp] at he ⊢
  change stepStmts [
    .assign .ptr [p.BK,p.BV] "DL" (A.flattenOp (RealBackwardInit.seedPointerOp p DL)),
    .assign .real [p.BK,p.BV] "DS_i" (.load .real (.ptr (.ref .ptr [p.BK,p.BV] "DS")) .none),
    .assign .real [] "d_i" (.load .real (.ptr (.ref .ptr [] "D")) .none),
    .assign .real [p.BK,p.BV] "Dacc" (update p),
    .store .real [p.BK,p.BV] (.ptr (.ref .ptr [p.BK,p.BV] "DL")) (.ref .real [p.BK,p.BV] "Dacc") .none] t = some t' at he
  obtain ⟨t1,ht1,he⟩ := TraceProofs.cons_inv he
  obtain ⟨s1,hs1,h1⟩ := assign_congr h _ (seedPointer_equal A p DL h) ht1
  obtain ⟨t2,ht2,he⟩ := TraceProofs.cons_inv he
  obtain ⟨s2,hs2,h2⟩ := assign_congr h1 _ (load_equal "DS" (by simp [Live]) h1) ht2
  obtain ⟨t3,ht3,he⟩ := TraceProofs.cons_inv he
  obtain ⟨s3,hs3,h3⟩ := assign_congr h2 _ (load_equal "D" (by simp [Live]) h2) ht3
  obtain ⟨t4,ht4,he⟩ := TraceProofs.cons_inv he
  obtain ⟨s4,hs4,h4⟩ := assign_congr h3 _ (update_equal p h3) ht4
  obtain ⟨t5,ht5,he⟩ := TraceProofs.cons_inv he
  obtain ⟨s5,hs5,h5⟩ := store_congr h4 (.ref .ptr [p.BK,p.BV] "DL") (.ref .real [p.BK,p.BV] "Dacc")
    (by simpa only [evalOp_ref] using h4.regs .ptr [p.BK,p.BV] "DL" (by simp [Live]))
    (by simpa only [evalOp_ref] using h4.regs .real [p.BK,p.BV] "Dacc" (by simp [Live])) ht5
  simp only [stepStmts.nil,Option.some.injEq] at he
  subst t'
  refine ⟨s5,?_,h5⟩
  change stepStmts (_ :: _ :: _ :: _ :: _ :: []) s = some s5
  simp only [update] at hs4
  rw [stepStmts.cons_some hs1,stepStmts.cons_some hs2,stepStmts.cons_some hs3,
    stepStmts.cons_some hs4,stepStmts.cons_some hs5,stepStmts.nil]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardFlatState
