import VeriTile.Triton.Semantics
import VeriTile.Triton.KernelLemmas

/-! Finite log-domain Sinkhorn normalization. The mathematical recurrence is
uᵢ = -log Σⱼ exp(zᵢⱼ + vⱼ), then vⱼ = -log Σᵢ exp(zᵢⱼ + uᵢ).
The result is exp(zᵢⱼ + uᵢ + vⱼ); there is no convergence claim. -/
namespace VeriTile.Triton.LogSinkhorn

noncomputable def rowUpdate (z : Tile .real [A, B]) (v : Tile .real [B]) : Tile .real [A] :=
  Tile.bop WithBot.realSub .scalarL (Tile.scalar (0 : WithBot ℝ))
    (Tile.uop WithBot.realLog (Tile.reduceSum ⟨1, by simp⟩ false
      (Tile.uop WithBot.realExp (Tile.bop WithBot.realAdd (.consR (.consSame .nil)) z
        (Tile.expandDim ⟨0, by simp⟩ v)))))

noncomputable def colUpdate (z : Tile .real [A, B]) (u : Tile .real [A]) : Tile .real [B] :=
  Tile.bop WithBot.realSub .scalarL (Tile.scalar (0 : WithBot ℝ))
    (Tile.uop WithBot.realLog (Tile.reduceSum ⟨0, by simp⟩ false
      (Tile.uop WithBot.realExp (Tile.bop WithBot.realAdd (.consSame (.consR .nil)) z
        (Tile.expandDim ⟨1, by simp⟩ u)))))

noncomputable def iterates (z : Tile .real [A, B]) : Nat → Tile .real [A] × Tile .real [B]
  | 0 => (⟨fun _ => (0 : WithBot ℝ)⟩, ⟨fun _ => (0 : WithBot ℝ)⟩)
  | i + 1 => let u := rowUpdate z (iterates z i).2; (u, colUpdate z u)

noncomputable def weights (z : Tile .real [A, B]) (iters : Nat) : Tile .real [A, B] :=
  Tile.uop WithBot.realExp (Tile.bop WithBot.realAdd (.consR (.consSame .nil))
    (Tile.bop WithBot.realAdd (.consSame (.consR .nil)) z
      (Tile.expandDim ⟨1, by simp⟩ (iterates z iters).1))
    (Tile.expandDim ⟨0, by simp⟩ (iterates z iters).2))

/-- Shared loop body in both matrix examples; the enclosing kernels contain
its source spelling. This is only the operational bridge to the recurrence. -/
def body (A B : Nat) : List Stmt :=
  [.assign .real [A] "u" (.sub .real .scalarL (.const 0)
    (.log (.reduceSum ⟨1, by simp⟩ false
      (.exp (.add .real (.consR (.consSame .nil)) (.ref .real [A, B] "z")
        (.expandDim ⟨0, by simp⟩ (.ref .real [B] "v"))))))),
   .assign .real [B] "v" (.sub .real .scalarL (.const 0)
    (.log (.reduceSum ⟨0, by simp⟩ false
      (.exp (.add .real (.consSame (.consR .nil)) (.ref .real [A, B] "z")
        (.expandDim ⟨1, by simp⟩ (.ref .real [A] "u")))))))]

/-- Registers other than loop temporaries, memory and launch coordinates are
preserved. In particular later matrix products still use the loaded input. -/
def Invariant (z : Tile .real [A, B]) (origin : BlockState) (i : Nat) (s : BlockState) : Prop :=
  s.regs .real [A, B] "z" = some z ∧
  s.regs .real [A] "u" = some (iterates z i).1 ∧
  s.regs .real [B] "v" = some (iterates z i).2 ∧
  s.mem = origin.mem ∧ s.pids = origin.pids ∧
  (∀ d sh n, n ≠ "u" → n ≠ "v" → n ≠ "iter" → s.regs d sh n = origin.regs d sh n)

set_option maxHeartbeats 1600000 in
theorem loop_run (z : Tile .real [A, B]) (N : Nat) (s : BlockState)
    (hz : s.regs .real [A, B] "z" = some z)
    (hu : s.regs .real [A] "u" = some (iterates z 0).1)
    (hv : s.regs .real [B] "v" = some (iterates z 0).2) :
    ∃ t, stepStmt (.forLoop "iter" N (body A B)) s = some t ∧ Invariant z s N t := by
  apply forLoop_inv (P := Invariant z s)
  · exact ⟨hz, hu, hv, rfl, rfl, fun _ _ _ _ _ _ => rfl⟩
  · intro i t _ ht
    rcases ht with ⟨hz, hu, hv, hm, hp, hr⟩
    let u := rowUpdate z (iterates z i).2
    let v := colUpdate z u
    let out := ((t.setReg "iter" .nat [] (Tile.scalar i)).setReg "u" .real [A] u).setReg "v" .real [B] v
    refine ⟨out, ?_, ?_⟩
    · simp [body, stepStmts, stepStmt, evalOp.eq_def, hz, hv, BlockState.setReg,
        out, u, v, rowUpdate, colUpdate]
      simp [Option.bind, hz]
      rfl
    · simp [out, Invariant, iterates, BlockState.setReg, u, v, hz, hm, hp]
      intro d sh n hnu hnv hni
      simpa [hnu, hnv, hni] using hr d sh n hnu hnv hni

/-- A row-major mathematical matrix read from the initial input memory. -/
def readMatrix (s : BlockState) (r : RegionName) (base : Nat) (A B : Nat) : Tile .real [A, B] :=
  ⟨fun (i, j, _) => (s.readMem r (base + i.val * B + j.val) : WithBot ℝ)⟩

noncomputable def scale (z : Tile .real [A, B]) (tau : ℝ) : Tile .real [A, B] :=
  Tile.bop WithBot.realDiv .scalarR z (Tile.scalar (tau : WithBot ℝ))

theorem address_injective (base S D : Nat) :
    Function.Injective (fun i : TileIndex [S, D] => base + i.1.val * D + i.2.1.val) := by
  rintro ⟨a, b, _⟩ ⟨c, d, _⟩ h
  dsimp only [Prod.fst, Prod.snd] at h
  have h' : a.val * D + b.val = c.val * D + d.val :=
    Nat.add_left_cancel (by simpa only [Nat.add_assoc] using h)
  have hcol := congrArg (fun n => n % D) h'
  rw [Nat.add_comm (a.val * D), Nat.add_comm (c.val * D)] at hcol
  simp only [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt b.isLt, Nat.mod_eq_of_lt d.isLt] at hcol
  obtain rfl : b = d := Fin.ext hcol
  have hrow := Nat.eq_of_mul_eq_mul_right (Nat.zero_lt_of_lt b.isLt) (Nat.add_right_cancel h')
  obtain rfl : a = c := Fin.ext hrow
  rfl


/-- Cell frame for one row-major matrix store. -/
theorem scatter_frame {ι : Type} (r : RegionName) (off : ι → Nat) (v : ι → ℝ)
    (l : List ι) (s : BlockState) (r' : RegionName) (o : Nat)
    (h : r' ≠ r ∨ ∀ i, o ≠ off i) :
    (l.foldl (fun st i => st.writeMem r (off i) (v i)) s).mem r' o = s.mem r' o := by
  by_cases hr : r' = r
  · subst r'
    exact BlockState.foldl_writeMem_mem_preserve_unhit off v l o
      (fun i _ he => (h.resolve_left (by simp)) i he.symm) s
  · exact BlockState.foldl_writeMem_mem_preserve_other_region off v l r' hr o s

/-- Reading a different output buffer survives a subsequent scatter store. -/
theorem scatter_read_other {ι : Type} (r : RegionName) (off : ι → Nat) (v : ι → ℝ)
    (l : List ι) (s : BlockState) (r' : RegionName) (o : Nat) (h : r' ≠ r) :
    (l.foldl (fun st i => st.writeMem r (off i) (v i)) s).readMem r' o = s.readMem r' o := by
  simp only [BlockState.readMem, BlockState.foldl_writeMem_mem_preserve_other_region off v l r' h o s]

end VeriTile.Triton.LogSinkhorn
