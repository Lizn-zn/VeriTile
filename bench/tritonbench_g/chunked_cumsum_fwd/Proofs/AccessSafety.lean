import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.Execution
import VeriTile.Triton.Memory.TraceProofs

/-! Bounds apply only to lanes actually read or written. Bias bounds are
required only when that branch is enabled. No bound is imposed on pointer
arithmetic for masked-out lanes. -/
namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Safety
open VeriTile.Triton Math Program Initialization Tiles Loads Preparation Execution
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (r : Regions) (hasBias : Bool) (bounds : RegionBounds) : Prop where
  dt : ∀ (i : Fin p.BH) (j : Fin p.BC), activeInput p l i.val j.val → dtAddress p l i.val j.val < bounds r.dt
  bias : hasBias = Bool.true → ∀ i : Fin p.BH, head p l i.val < p.heads → head p l i.val*p.biasHead < bounds r.bias
  rate : ∀ i : Fin p.BH, head p l i.val < p.heads → head p l i.val*p.rateHead < bounds r.rate
  out : ∀ (i : Fin p.BH) (j : Fin p.BC), activeOutput p l i.val j.val → outAddress p l i.val j.val < bounds r.out
  sum : ∀ (i : Fin p.BH) (j : Fin p.BC), activeOutput p l i.val j.val → sumAddress p l i.val j.val < bounds r.sum

theorem initialization_safe (p : Layout) (r : Regions) (bounds : RegionBounds) (u : BlockState) :
    Stmt.TraceSafeList bounds (initialization p r) u := by
  apply TraceProofs.all_safe
  simp [initialization,dtOffset,offset,pointer,headOp,columnOp,limit,remaining,Stmt.TraceSafe,Op.SafeAt.eq_def]

theorem dt_safe (p : Layout) (l : Launch) (r : Regions) (u : BlockState) (bounds : RegionBounds)
    (h : Ready p l r u)
    (hb : ∀ (i : Fin p.BH) (j : Fin p.BC), activeInput p l i.val j.val → dtAddress p l i.val j.val < bounds r.dt) :
    (loadDt p).SafeAt bounds u := by
  rw [loadDt,Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt.eq_def],by simp [inputMask,mask,headOp,columnOp,zero,Op.SafeAt.eq_def],?_⟩
  unfold MemAccess.ActiveAddressSafe memAccessActiveAddressSafe Op.PointerAddressesSafeOn
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ptr he idx hi
  rw [evalOp_ref,h.dt] at he
  cases he
  obtain ⟨ms,he,hi⟩ := hi
  rw [inputMask_eval p l u h.head h.column h.limit] at he
  cases he
  exact hb idx.1 idx.2.1 (by simpa only [Tile.mat,decide_eq_true_eq] using hi)

theorem rate_safe (p : Layout) (l : Launch) (r : Regions) (u : BlockState) (bounds : RegionBounds)
    (h : Ready p l r u)
    (hb : ∀ i : Fin p.BH, head p l i.val < p.heads → head p l i.val*p.rateHead < bounds r.rate) :
    (loadRate p).SafeAt bounds u := by
  rw [loadRate,Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt.eq_def],by simp [headMask,headOp,Op.SafeAt.eq_def],?_⟩
  unfold MemAccess.ActiveAddressSafe memAccessActiveAddressSafe Op.PointerAddressesSafeOn
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ptr he idx hi
  rw [evalOp_ref,h.rate] at he
  cases he
  obtain ⟨ms,he,hi⟩ := hi
  rw [headMask_eval p l u h.head] at he
  cases he
  exact hb idx.1 (by simpa only [Tile.vec,decide_eq_true_eq] using hi)

theorem bias_safe (p : Layout) (l : Launch) (r : Regions) (u : BlockState) (bounds : RegionBounds)
    (hh : u.regs .nat [p.BH] "offs_h" = some (heads p l))
    (hb : ∀ i : Fin p.BH, head p l i.val < p.heads → head p l i.val*p.biasHead < bounds r.bias) :
    (loadBias p r).SafeAt bounds u := by
  have hoff : evalOp (.mul .nat .scalarR (headOp p) (.constNat p.biasHead)) u =
      some (Tile.vec (fun i => head p l i.val*p.biasHead)) := by
    simp only [headOp,evalOp.eq_def,hh,Option.bind,Option.bind_eq_bind]
    rfl
  rw [loadBias,Op.SafeAt.eq_def]
  refine ⟨by simp [headOp,Op.SafeAt.eq_def],by simp [headMask,headOp,Op.SafeAt.eq_def],?_⟩
  unfold MemAccess.ActiveAddressSafe memAccessActiveAddressSafe
  intro offsets he idx hi
  rw [hoff] at he
  cases he
  obtain ⟨ms,he,hi⟩ := hi
  rw [headMask_eval p l u hh] at he
  cases he
  exact hb idx.1 (by simpa only [Tile.vec,decide_eq_true_eq] using hi)

theorem store_safe (p : Layout) (l : Launch) (out : RegionName) (address : Stores.Address p)
    (ptrName valName : RegName) (u : BlockState) (bounds : RegionBounds)
    (hp : u.regs .ptr [p.BH,p.BC] ptrName = some (Tile.mat (fun i j => (out,address i j))))
    (hh : u.regs .nat [p.BH] "offs_h" = some (heads p l))
    (hc : u.regs .nat [p.BC] "offs_c" = some (columns p))
    (hb : ∀ (i : Fin p.BH) (j : Fin p.BC), activeOutput p l i.val j.val → address i j < bounds out) :
    Stmt.TraceSafe bounds (.store .real [p.BH,p.BC] (.ptr (.ref .ptr [p.BH,p.BC] ptrName))
      (.ref .real [p.BH,p.BC] valName) (.mask (outputMask p))) u := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt,Op.SafeAt.eq_def],by simp [Op.SafeAt.eq_def],
    by simp [MaskOpt.SafeAt,outputMask,mask,headOp,columnOp,Op.SafeAt.eq_def],?_⟩
  unfold MemAccess.ActiveAddressSafe memAccessActiveAddressSafe Op.PointerAddressesSafeOn
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ptr he idx hi
  rw [evalOp_ref,hp] at he
  cases he
  obtain ⟨ms,he,hi⟩ := hi
  rw [outputMask_eval p l u hh hc] at he
  cases he
  exact hb idx.1 idx.2.1 (by simpa only [Tile.mat,decide_eq_true_eq] using hi)

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Safety
