import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealPost
import VeriTile.Triton.Memory.Footprint
import VeriTile.Triton.Memory.TraceProofs

/-! Exact cell-level frame for the full real program. The post-loop proof tracks
actual output pointers while leaving numerical values abstract. Its footprint
contains only active lanes of the three stores, so masked output cells and
all other cells retain their complete tagged values. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealFrame
open VeriTile.Triton RealMath
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def assignsAway (name : RegName) : Stmt → Prop
  | .assign _ _ dst _ => name ≠ dst
  | _ => False

theorem assignsAway_regs (code : List Stmt) (s t : BlockState)
    (d : TileDType) (sh : TileShape) (name : RegName)
    (hcode : ∀ st ∈ code, assignsAway name st)
    (hrun : stepStmts code s = some t) :
    t.regs d sh name = s.regs d sh name := by
  induction code generalizing s with
  | nil => simpa using congrArg (fun x => x.map (fun s => s.regs d sh name)) hrun.symm
  | cons st rest ih =>
    obtain ⟨u, hu, ht⟩ := TraceProofs.cons_inv hrun
    have hs := hcode st (by simp)
    have he := ih u (fun st hm => hcode st (List.mem_cons_of_mem _ hm)) ht
    rw [he]
    cases st <;> simp only [assignsAway] at hs
    case assign d shape dst op =>
      obtain ⟨v, _, rfl⟩ := TraceProofs.assign_inv hu
      simp [BlockState.setReg, hs]

theorem block_store_regs {d sh} {ptr : Op .blockPtr sh} {bc : List Nat}
    {value : Op d sh} {s t : BlockState}
    (h : stepStmt (.store d sh (.blockPtr ptr bc) value .none) s = some t) :
    t.regs = s.regs := by
  cases hv : evalOp value s with
  | none => simp [stepStmt, hv] at h
  | some v =>
    cases hp : evalOp ptr s with
    | none => simp [stepStmt, hv, hp] at h
    | some p =>
      simp only [stepStmt, hv, hp, Option.bind_some] at h
      cases h
      funext dtype shape name
      exact BlockState.foldl_writeMemTypedAt_masked_regs _ _ _ _ _ _ _ _ _ _

def qkPointer (p : Layout) (l : Launch) (r : RegionName) : Tile .blockPtr [p.BT, p.BK] :=
  ⟨fun _ =>
    { region := r, baseOffset := l.head * p.s_k_h,
      parentShape := [p.T, p.K], blockShape := [p.BT, p.BK], strides := [p.s_k_t, 1],
      offsets := [l.time * p.BT, l.key * p.BK] }⟩

def dgPointer (p : Layout) (l : Launch) (r : RegionName) : Tile .blockPtr [p.BT] :=
  ⟨fun _ =>
    { region := r, baseOffset := (l.key * l.heads + l.head) * p.T,
      parentShape := [p.T], blockShape := [p.BT], strides := [1],
      offsets := [l.time * p.BT] }⟩

def qkAddress (p : Layout) (l : Launch) (i : TileIndex [p.BT, p.BK]) : Nat :=
  l.head * p.s_k_h + (l.time * p.BT + i.1.val) * p.s_k_t + l.key * p.BK + i.2.1.val

def dgAddress (p : Layout) (l : Launch) (i : TileIndex [p.BT]) : Nat :=
  (l.key * l.heads + l.head) * p.T + l.time * p.BT + i.1.val

def qkActive (p : Layout) (l : Launch) (i : TileIndex [p.BT, p.BK]) : Prop :=
  l.time * p.BT + i.1.val < p.T ∧ l.key * p.BK + i.2.1.val < p.K

def dgActive (p : Layout) (l : Launch) (i : TileIndex [p.BT]) : Prop :=
  l.time * p.BT + i.1.val < p.T

def footprint (p : Layout) (l : Launch) (dq dk dg : RegionName) : WriteFootprint :=
  fun addr =>
    WriteFootprint.activeTileImage dq (qkAddress p l) (qkActive p l) addr ∨
    WriteFootprint.activeTileImage dk (qkAddress p l) (qkActive p l) addr ∨
    WriteFootprint.activeTileImage dg (dgAddress p l) (dgActive p l) addr

structure Grid (s : BlockState) (l : Launch) : Prop where
  key : s.regs .nat [] "i_k" = some (Tile.scalar l.key)
  time : s.regs .nat [] "i_t" = some (Tile.scalar l.time)
  head : s.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  heads : s.regs .nat [] "n_bh" = some (Tile.scalar l.heads)

theorem qk_pointer (s : BlockState) (p : Layout) (l : Launch) (r : RegionName)
    (hg : Grid s l) :
    evalOp (Op.makeBlockPtrDynOffsets r
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat p.s_k_h))
      [p.T, p.K] [p.BT, p.BK] [p.s_k_t, 1]
      [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat p.BT),
       Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat p.BK)]) s =
      some (qkPointer p l r) := by
  exact RealStep.make_pointer _ _ _ _ _ _ _ _ _ _ _ _ _ _
    (RealStep.mul_ref _ _ _ _ hg.head)
    (RealStep.mul_ref _ _ _ _ hg.time)
    (RealStep.mul_ref _ _ _ _ hg.key)

theorem dg_pointer (s : BlockState) (p : Layout) (l : Launch) (r : RegionName)
    (hg : Grid s l) :
    evalOp (Op.makeBlockPtrDynOffsets r
      (Op.mul .nat Broadcast.nil
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.ref .nat [] "n_bh"))
          (Op.ref .nat [] "i_bh")) (Op.constNat p.T))
      [p.T] [p.BT] [1]
      [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat p.BT)]) s =
      some (dgPointer p l r) := by
  simp only [evalOp, hg.key, hg.time, hg.head, hg.heads, List.mapM_cons,
    List.mapM_nil, Option.bind_some, Option.pure_def, Option.bind_eq_bind]
  rfl

theorem grid_set (s : BlockState) (l : Launch) (d sh name v)
    (hg : Grid s l)
    (hn : name ≠ "i_k" ∧ name ≠ "i_t" ∧ name ≠ "i_bh" ∧ name ≠ "n_bh") : Grid (s.setReg name d sh v) l := by
  rcases hn with ⟨hk, ht, hb, hn⟩
  constructor <;> simp [BlockState.setReg, Ne.symm hk, Ne.symm ht,
    Ne.symm hb, Ne.symm hn, hg.key, hg.time, hg.head, hg.heads]


private theorem qk_store (s t : BlockState) (p : Layout) (l : Launch)
    (r : RegionName) (name : RegName) (value : Op .real [p.BT, p.BK])
    (hp : s.regs .blockPtr [p.BT, p.BK] name = some (qkPointer p l r))
    (he : stepStmt (.store .real [p.BT, p.BK]
      (.blockPtr (.ref .blockPtr [p.BT, p.BK] name) [0, 1]) value .none) s = some t) :
    BlockState.WriteWithin
      (WriteFootprint.activeTileImage r (qkAddress p l) (qkActive p l)) s t := by
  apply stepStmt_store_writeWithin he
  intro ptrs hptrs i hi
  simp only [evalOp_ref, hp, Option.some.injEq] at hptrs
  subst ptrs
  refine ⟨rfl, i, ?_, ?_⟩
  · simpa only [qkPointer, TileShape.indexToList, BlockPtr.inBounds_2d_offsets,
      decide_eq_true_eq, qkActive] using hi
  · simp [qkPointer, TileShape.indexToList, BlockPtr.address_2d_offsets,
      qkAddress, Nat.add_assoc]

private theorem dg_store (s t : BlockState) (p : Layout) (l : Launch)
    (r : RegionName) (name : RegName) (value : Op .real [p.BT])
    (hp : s.regs .blockPtr [p.BT] name = some (dgPointer p l r))
    (he : stepStmt (.store .real [p.BT]
      (.blockPtr (.ref .blockPtr [p.BT] name) [0]) value .none) s = some t) :
    BlockState.WriteWithin
      (WriteFootprint.activeTileImage r (dgAddress p l) (dgActive p l)) s t := by
  apply stepStmt_store_writeWithin he
  intro ptrs hptrs i hi
  simp only [evalOp_ref, hp, Option.some.injEq] at hptrs
  subst ptrs
  refine ⟨rfl, i, ?_, ?_⟩
  · simpa only [dgPointer, TileShape.indexToList, BlockPtr.inBounds_1d,
      decide_eq_true_eq, dgActive] using hi
  · simp [dgPointer, TileShape.indexToList, BlockPtr.address_1d,
      dgAddress, Nat.add_assoc]

/-- The tail writes only its three active output tiles. Every other tagged
cell is preserved, including masked lanes within each output allocation. -/
theorem post_frame (s t : BlockState) (p : Layout) (l : Launch)
    (q k dq dk dg : RegionName) (hg : Grid s l)
    (hrun : stepStmts
      (cbdPostLoop q k dq dk dg p.s_k_h p.s_k_t p.scale p.T p.K p.BT p.BK) s = some t) :
    BlockState.WriteWithin (footprint p l dq dk dg) s t := by
  let code := cbdPostLoop q k dq dk dg p.s_k_h p.s_k_t p.scale p.T p.K p.BT p.BK
  have hs : code = code.take 14 ++ code.drop 14 := (List.take_append_drop 14 code).symm
  change stepStmts code s = some t at hrun
  rw [hs, stepStmts.append_some_iff] at hrun
  obtain ⟨u, hu, hr⟩ := hrun
  have hm : u.mem = s.mem := storeFree_stepStmts_mem _ _ _
    (by simp [code, cbdPostLoop, storeFree]) hu
  have hg' : Grid u l := by
    constructor
    · exact (assignsAway_regs _ _ _ .nat [] "i_k"
        (by simp [code, cbdPostLoop, assignsAway]) hu).trans hg.key
    · exact (assignsAway_regs _ _ _ .nat [] "i_t"
        (by simp [code, cbdPostLoop, assignsAway]) hu).trans hg.time
    · exact (assignsAway_regs _ _ _ .nat [] "i_bh"
        (by simp [code, cbdPostLoop, assignsAway]) hu).trans hg.head
    · exact (assignsAway_regs _ _ _ .nat [] "n_bh"
        (by simp [code, cbdPostLoop, assignsAway]) hu).trans hg.heads
  simp only [code, cbdPostLoop, List.drop] at hr
  obtain ⟨u1, hq, hr⟩ := TraceProofs.cons_inv hr
  obtain ⟨pq, hpq, rfl⟩ := TraceProofs.assign_inv hq
  have hpeq : pq = qkPointer p l dq := Option.some.inj (hpq.symm.trans (qk_pointer u p l dq hg'))
  subst pq
  have g1 := grid_set u l .blockPtr [p.BT, p.BK] "p_dq" (qkPointer p l dq) hg' (by decide)
  obtain ⟨u2, hk, hr⟩ := TraceProofs.cons_inv hr
  obtain ⟨pk, hpk, rfl⟩ := TraceProofs.assign_inv hk
  have hpeq : pk = qkPointer p l dk := Option.some.inj (hpk.symm.trans (qk_pointer _ p l dk g1))
  subst pk
  have g2 := grid_set _ l .blockPtr [p.BT, p.BK] "p_dk" (qkPointer p l dk) g1 (by decide)
  obtain ⟨u3, hd, hr⟩ := TraceProofs.cons_inv hr
  obtain ⟨pg, hpg, rfl⟩ := TraceProofs.assign_inv hd
  have hpeq : pg = dgPointer p l dg := Option.some.inj (hpg.symm.trans (dg_pointer _ p l dg g2))
  subst pg
  obtain ⟨v1, hqStore, hr⟩ := TraceProofs.cons_inv hr
  obtain ⟨v2, hkStore, hr⟩ := TraceProofs.cons_inv hr
  obtain ⟨v3, hgStore, hr⟩ := TraceProofs.cons_inv hr
  simp only [stepStmts.nil, Option.some.injEq] at hr
  subst v3
  have hqr := block_store_regs hqStore
  have hkr := block_store_regs hkStore
  have fq := qk_store _ _ p l dq "p_dq" _ (by simp) hqStore
  have fk := qk_store _ _ p l dk "p_dk" _ (by rw [hqr]; simp) hkStore
  have fg := dg_store _ _ p l dg "p_dg" _ (by rw [hkr, hqr]; simp) hgStore
  intro r off hn
  have hnq : ¬ WriteFootprint.activeTileImage dq (qkAddress p l) (qkActive p l) (r,off) :=
    fun h => hn (Or.inl h)
  have hnk : ¬ WriteFootprint.activeTileImage dk (qkAddress p l) (qkActive p l) (r,off) :=
    fun h => hn (Or.inr (Or.inl h))
  have hng : ¬ WriteFootprint.activeTileImage dg (dgAddress p l) (dgActive p l) (r,off) :=
    fun h => hn (Or.inr (Or.inr h))
  exact (congrFun (congrFun hm.symm r) off).trans
    ((fq r off hnq).trans ((fk r off hnk).trans (fg r off hng)))


/-- Full-source frame, including every value-loop iteration and all three
stores. It follows from the actual launch registers and executed pointers;
no output-state or write-footprint assumption is supplied by the caller. -/
theorem original_frame (s sf : BlockState) (p : Layout)
    (qk valueStorage hist dhist : TypedSource.Storage)
    (q k v h g dout dh dqOut dkOut dgOut : RegionName)
    (hu : s.undef = fun _ _ => 0)
    (hrun : exec
      (RealSource.source qk valueStorage hist dhist Bool.false q k v h g dout dh dqOut dkOut dgOut
        p.s_k_h p.s_k_t p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.scale
        p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType.toAlgKernel s = some sf) :
    BlockState.WriteWithin (footprint p (RealLoop.launch s) dqOut dkOut dgOut) s sf := by
  let regions : RealStep.Regions := ⟨v, h, dout, dh⟩
  let a := RealPost.inputs s q k v h g dout dh
  obtain ⟨setup, t, hsetup, hprefix, hmem, hi⟩ :=
    RealLoop.prologue_and_loop s p regions g a hu
      (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
  obtain ⟨setup', hsetup', _, _, _, hik, hit, hib, hnb, _⟩ :=
    cbdPreLoop_run g p.T p.BT p.BK s (congrFun hu g)
  have heq : setup' = setup := Option.some.inj (hsetup'.symm.trans hsetup)
  subst setup'
  have grid : Grid t (RealLoop.launch s) := ⟨
    ((hi.fixed .nat [] "i_k" (by decide)).trans hik),
    ((hi.fixed .nat [] "i_t" (by decide)).trans hit),
    ((hi.fixed .nat [] "i_bh" (by decide)).trans hib),
    ((hi.fixed .nat [] "n_bh" (by decide)).trans hnb)⟩
  rw [exec, RealSource.real_body_matches, cbd_body_eq] at hrun
  change stepStmts
    ((cbdPreLoop g p.T p.BT p.BK ++ [RealLoop.statement p regions]) ++
      cbdPostLoop q k dqOut dkOut dgOut p.s_k_h p.s_k_t p.scale p.T p.K p.BT p.BK) s = _ at hrun
  rw [stepStmts.append_some hprefix] at hrun
  have hf := post_frame t sf p (RealLoop.launch s) q k dqOut dkOut dgOut grid hrun
  intro r off hn
  exact (congrFun (congrFun (hi.mem.trans hmem).symm r) off).trans (hf r off hn)

/-- Standard row stride suffices even when BK exceeds K: only active columns
participate in the collision check. Masked lanes can alias without writing. -/
theorem qk_injective_on (p : Layout) (l : Launch) (hstride : p.K ≤ p.s_k_t)
    (i j : TileIndex [p.BT,p.BK]) (hi : qkActive p l i) (hj : qkActive p l j)
    (he : qkAddress p l i = qkAddress p l j) : i = j := by
  have hx : l.key * p.BK + i.2.1.val < p.s_k_t := lt_of_lt_of_le hi.2 hstride
  have hy : l.key * p.BK + j.2.1.val < p.s_k_t := lt_of_lt_of_le hj.2 hstride
  have hp : 0 < p.s_k_t := lt_of_le_of_lt (Nat.zero_le _) hx
  have he' : (l.time * p.BT + i.1.val) * p.s_k_t + (l.key * p.BK + i.2.1.val) =
      (l.time * p.BT + j.1.val) * p.s_k_t + (l.key * p.BK + j.2.1.val) := by
    simp only [qkAddress, Nat.add_assoc] at he
    exact Nat.add_left_cancel he
  have hdiv := congrArg (fun n => n / p.s_k_t) he'
  have ha : ((l.time * p.BT + i.1.val) * p.s_k_t + (l.key * p.BK + i.2.1.val)) / p.s_k_t =
      l.time * p.BT + i.1.val := by
    rw [Nat.mul_comm, Nat.mul_add_div hp, Nat.div_eq_of_lt hx, Nat.add_zero]
  have hb : ((l.time * p.BT + j.1.val) * p.s_k_t + (l.key * p.BK + j.2.1.val)) / p.s_k_t =
      l.time * p.BT + j.1.val := by
    rw [Nat.mul_comm, Nat.mul_add_div hp, Nat.div_eq_of_lt hy, Nat.add_zero]
  dsimp only at hdiv
  rw [ha,hb] at hdiv
  have hr : i.1 = j.1 := Fin.ext (Nat.add_left_cancel hdiv)
  have hc : i.2.1 = j.2.1 := by
    have hrv : i.1.val = j.1.val := congrArg Fin.val hr
    rw [hrv] at he'
    exact Fin.ext (Nat.add_left_cancel (Nat.add_left_cancel he'))
  rcases i with ⟨a,b,⟨⟩⟩
  rcases j with ⟨c,d,⟨⟩⟩
  simpa only [Prod.mk.injEq] using And.intro hr (And.intro hc trivial)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealFrame
