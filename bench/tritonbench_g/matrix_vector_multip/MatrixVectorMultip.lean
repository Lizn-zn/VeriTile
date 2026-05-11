import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `matrix_vector_multip.py`'s `mv_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `A_ptrs += BLOCK_M * stride_am` / `B_ptrs += BLOCK_M * stride_bm`
  pointer mutation → address recomputed from loop variable `m`.
- Python `.to(tl.float32)` casts are omitted at the algorithm layer.
- Python `BLOCK_N` / `BLOCK_M: tl.constexpr` → Lean `Nat` parameters. -/
def mv_kernel
    (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  offset_n = pid * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))[:, None]
  offset_m = tl.arange(0, $(BLOCK_M))[None, :]
  n_mask = offset_n < $(N)
  acc = tl.zeros([$(BLOCK_N), $(BLOCK_M)])
  for m in range(0, $(M), $(BLOCK_M)) {
    m_mask = (m + offset_m) < $(M)
    a = tl.load(A + offset_n * $(stride_an) + (m + offset_m) * $(stride_am),
      mask=n_mask and m_mask, other=0.0)
    b = tl.load(B + (m + offset_m) * $(stride_bm),
      mask=m_mask, other=0.0)
    acc = acc + a * b
  }
  acc = tl.sum(acc, axis=1)
  c_ptrs = C + offset_n * $(stride_cn)
  tl.store(c_ptrs, acc[:, None], mask=n_mask)
}

def nIndex (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * BLOCK_N + i.val

def cOffset (s : BlockState) (stride_cn BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  nIndex s BLOCK_N i * stride_cn

def aOffset
    (s : BlockState) (stride_an stride_am BLOCK_N : Nat)
    (i : Fin BLOCK_N) (j : Fin BLOCK_M) : Nat :=
  nIndex s BLOCK_N i * stride_an + j.val * stride_am

def bOffset (stride_bm : Nat) (j : Fin BLOCK_M) : Nat :=
  j.val * stride_bm

noncomputable def mvProdTile
    (s : BlockState) (A B : RegionName)
    (N M stride_an stride_am stride_bm BLOCK_N BLOCK_M : Nat) :
    Tile .real [BLOCK_N, BLOCK_M] :=
  { data := fun idx =>
      let ni := (TileShape.dropInsertedIndex [BLOCK_N] 1 1 (idx.1, 0, PUnit.unit)).1
      let mj := (TileShape.dropInsertedIndex [BLOCK_M] 0 1 (0, idx.2.1, PUnit.unit)).1
      Option.map₂ (fun a b => a * b)
        (if s.pids 0 * BLOCK_N + ni.val < N ∧ mj.val < M then
          some (s.readMem A ((s.pids 0 * BLOCK_N + ni.val) * stride_an + mj.val * stride_am))
        else some (0.0 : ℝ))
        (if mj.val < M then
          some (s.readMem B (mj.val * stride_bm))
        else some (0.0 : ℝ)) }

noncomputable def mvSpec
    (s : BlockState) (A B : RegionName)
    (N M stride_an stride_am stride_bm BLOCK_N BLOCK_M : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_M]) ⟨1, by simp⟩ Bool.false
      (mvProdTile s A B N M stride_an stride_am stride_bm BLOCK_N BLOCK_M)).data
        (i, PUnit.unit))

/-- Algorithm-layer correctness for the one-block matrix-vector slice. -/
theorem mv_kernel_correct
    (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => cOffset s stride_cn BLOCK_N i))
    (hExec : exec (mv_kernel A B C N M stride_an stride_am stride_bm
        stride_cn BLOCK_N BLOCK_M) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem C (cOffset s stride_cn BLOCK_N i) =
        if nIndex s BLOCK_N i < N then
          mvSpec s A B N M stride_an stride_am stride_bm BLOCK_N BLOCK_M i
        else s.readMem C (cOffset s stride_cn BLOCK_N i) := by
  sorry
/-- Compute-facing correctness for the one-block matrix-vector slice. -/
theorem mv_kernel_compute_correct
    (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => cOffset s stride_cn BLOCK_N i)) :
    ComputeCorrect.Realizes
      (kernel := mv_kernel A B C N M stride_an stride_am stride_bm
        stride_cn BLOCK_N BLOCK_M)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N)
        (fun i => (C, cOffset s stride_cn BLOCK_N i)))
      (expected := fun i =>
        mvSpec s A B N M stride_an stride_am stride_bm BLOCK_N BLOCK_M i) := by
  sorry
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip
