import VeriTile.Triton.Core.Ast
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.ProjectionLists
open VeriTile.Triton

theorem surface (k : ComputeKernel)
    (hk : k.toAlgorithm? = .ok k.toAlgKernel) :
    ComputeStmt.listToAlgorithm? k.surfaceBody = .ok k.toAlgKernel.body := by
  cases k with
  | mk inputs outputs body =>
      cases hb : ComputeStmt.listToAlgorithm? body <;>
        simp_all [ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel]

theorem erase_map (xs : List Stmt) : Stmt.eraseDTypeList xs = xs.map Stmt.eraseDType := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simpa only [Stmt.eraseDTypeList, List.map_cons] using congrArg (x.eraseDType :: ·) ih

/-- Successful source projection commutes with a statement prefix. -/
theorem take (n : Nat) (xs : List ComputeStmt) (ys : List Stmt)
    (h : ComputeStmt.listToAlgorithm? xs = .ok ys) :
    ComputeStmt.listToAlgorithm? (xs.take n) = .ok (ys.take n) := by
  induction n generalizing xs ys with
  | zero => rfl
  | succ n ih =>
      cases xs with
      | nil =>
          have hy : ys=[] := by simpa [ComputeStmt.listToAlgorithm?] using h.symm
          subst ys
          rfl
      | cons x xs =>
          cases hx : ComputeStmt.toAlgorithm? x with
          | error e => simp [ComputeStmt.listToAlgorithm?, hx] at h
          | ok y =>
              cases ht : ComputeStmt.listToAlgorithm? xs with
              | error e => simp [ComputeStmt.listToAlgorithm?, hx, ht] at h
              | ok tail =>
                  have hy : y::tail=ys := by
                    simpa [ComputeStmt.listToAlgorithm?, hx, ht] using h
                  subst ys
                  simp only [List.take_succ_cons, ComputeStmt.listToAlgorithm?, hx,
                    ih xs tail ht, Except.bind]

/-- Successful source projection commutes with a statement suffix. -/
theorem drop (n : Nat) (xs : List ComputeStmt) (ys : List Stmt)
    (h : ComputeStmt.listToAlgorithm? xs = .ok ys) :
    ComputeStmt.listToAlgorithm? (xs.drop n) = .ok (ys.drop n) := by
  induction n generalizing xs ys with
  | zero => exact h
  | succ n ih =>
      cases xs with
      | nil =>
          have hy : ys=[] := by simpa [ComputeStmt.listToAlgorithm?] using h.symm
          subst ys
          rfl
      | cons x xs =>
          cases hx : ComputeStmt.toAlgorithm? x with
          | error e => simp [ComputeStmt.listToAlgorithm?, hx] at h
          | ok y =>
              cases ht : ComputeStmt.listToAlgorithm? xs with
              | error e => simp [ComputeStmt.listToAlgorithm?, hx, ht] at h
              | ok tail =>
                  have hy : y::tail=ys := by
                    simpa [ComputeStmt.listToAlgorithm?, hx, ht] using h
                  subst ys
                  exact ih xs tail ht

theorem joined_supported (a b : ComputeKernel) (inputs outputs : List RegionName) (n : Nat)
    (ha : a.toAlgorithm? = .ok a.toAlgKernel)
    (hb : b.toAlgorithm? = .ok b.toAlgKernel) :
    (ComputeKernel.mk inputs outputs (a.surfaceBody.take n ++ b.surfaceBody.drop n)).toAlgorithm? =
      .ok (ComputeKernel.mk inputs outputs (a.surfaceBody.take n ++ b.surfaceBody.drop n)).toAlgKernel := by
  have hp : ComputeStmt.listToAlgorithm? (a.surfaceBody.take n ++ b.surfaceBody.drop n) =
      .ok (a.toAlgKernel.body.take n ++ b.toAlgKernel.body.drop n) := by
    rw [ComputeStmt.listToAlgorithm?_append,
      take n _ _ (surface a ha), drop n _ _ (surface b hb)]
  simp [ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, hp]

/-- Composing two complete source templates at a fixed statement boundary
commutes with checked projection and dtype erasure. No source statement is
discarded because of projection failure. -/
theorem joined_erased_body (a b : ComputeKernel) (inputs outputs : List RegionName) (n : Nat)
    (ha : a.toAlgorithm? = .ok a.toAlgKernel)
    (hb : b.toAlgorithm? = .ok b.toAlgKernel) :
    (ComputeKernel.mk inputs outputs (a.surfaceBody.take n ++ b.surfaceBody.drop n)).eraseDType.toAlgKernel.body =
      a.eraseDType.toAlgKernel.body.take n ++ b.eraseDType.toAlgKernel.body.drop n := by
  have hp : ComputeStmt.listToAlgorithm? (a.surfaceBody.take n ++ b.surfaceBody.drop n) =
      .ok (a.toAlgKernel.body.take n ++ b.toAlgKernel.body.drop n) := by
    rw [ComputeStmt.listToAlgorithm?_append,
      take n _ _ (surface a ha), drop n _ _ (surface b hb)]
  simp only [ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?_mk, hp, ha, hb,
    ComputeKernel.toAlgKernel_fromKernelBody, Kernel.eraseDType, erase_map,
    List.map_append, List.map_take, List.map_drop]

theorem replace_supported (a : ComputeKernel) (inputs outputs : List RegionName) (n : Nat)
    (replacement : ComputeStmt) (st : Stmt)
    (ha : a.toAlgorithm? = .ok a.toAlgKernel)
    (hr : replacement.toAlgorithm? = .ok st) :
    (ComputeKernel.mk inputs outputs
      (a.surfaceBody.take n ++ [replacement] ++ a.surfaceBody.drop (n+1))).toAlgorithm? =
      .ok (ComputeKernel.mk inputs outputs
        (a.surfaceBody.take n ++ [replacement] ++ a.surfaceBody.drop (n+1))).toAlgKernel := by
  have hp : ComputeStmt.listToAlgorithm?
      (a.surfaceBody.take n ++ [replacement] ++ a.surfaceBody.drop (n+1)) =
      .ok (a.toAlgKernel.body.take n ++ [st] ++ a.toAlgKernel.body.drop (n+1)) := by
    simp only [ComputeStmt.listToAlgorithm?_append, take n _ _ (surface a ha),
      drop (n+1) _ _ (surface a ha), ComputeStmt.listToAlgorithm?, hr, Except.bind]
  simp [ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, hp]

theorem replace_erased_body (a : ComputeKernel) (inputs outputs : List RegionName) (n : Nat)
    (replacement : ComputeStmt) (st : Stmt)
    (ha : a.toAlgorithm? = .ok a.toAlgKernel)
    (hr : replacement.toAlgorithm? = .ok st) :
    (ComputeKernel.mk inputs outputs
      (a.surfaceBody.take n ++ [replacement] ++ a.surfaceBody.drop (n+1))).eraseDType.toAlgKernel.body =
      a.eraseDType.toAlgKernel.body.take n ++ [st.eraseDType] ++
        a.eraseDType.toAlgKernel.body.drop (n+1) := by
  have hp : ComputeStmt.listToAlgorithm?
      (a.surfaceBody.take n ++ [replacement] ++ a.surfaceBody.drop (n+1)) =
      .ok (a.toAlgKernel.body.take n ++ [st] ++ a.toAlgKernel.body.drop (n+1)) := by
    simp only [ComputeStmt.listToAlgorithm?_append, take n _ _ (surface a ha),
      drop (n+1) _ _ (surface a ha), ComputeStmt.listToAlgorithm?, hr, Except.bind]
  simp only [ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?_mk, hp, ha,
    ComputeKernel.toAlgKernel_fromKernelBody, Kernel.eraseDType, erase_map,
    List.map_append, List.map_take, List.map_drop, List.map_cons, List.map_nil]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.ProjectionLists
