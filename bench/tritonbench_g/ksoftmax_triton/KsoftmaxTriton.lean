import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton

open VeriTile.Triton

/-- Surface transcription of `ksoftmax_triton.py`'s `_softmax` for the
`MASK_TYPE='qk'` branch.

This covers the tested qk-mask forward paths, including optional causal masking,
optional fp16 accumulator cast, and the `LOG`/softmax output branch. -/
def ksoftmax_forward_qk_surface
    (Y X Mask : RegionName)
    (stride_ym stride_yn stride_xm stride_xn stride_m K DEPTH : Nat)
    (LOG CAUSAL IS_FP16 : Bool) :
    ComputeKernel := triton {
  m = tl.program_id(axis=0)
  n = tl.program_id(axis=1)
  k = tl.arange(0, $(DEPTH))
  x_ptrs = X + m * $(stride_xm) + n * $(stride_xn) + k
  io_mask = k < $(K)
  if CAUSAL {
    io_mask = io_mask and (k <= n)
  }
  x = tl.load(x_ptrs, mask=io_mask, other=-inf)
  if CAUSAL {
    off = -inf
    off = (off).to(x.dtype)
    x = tl.where(k > n, off, x)
  }
  mask_ptrs = Mask + n * $(stride_m) + k
  add_mask = tl.load(mask_ptrs, mask=io_mask, other=-inf)
  x += add_mask
  z = x - tl.max(x, axis=0)
  if IS_FP16 {
    z = (z).to(tl.float32)
  }
  num = tl.exp(z)
  denom = tl.sum(num, axis=0)
  if LOG {
    y = z - tl.log(denom)
  } else {
    y = num / denom
  }
  y_ptrs = Y + m * $(stride_ym) + n * $(stride_yn) + k
  tl.store(y_ptrs, y, mask=k < $(K))
}

/-- Surface transcription of `ksoftmax_triton.py`'s `_softmax` for the
`MASK_TYPE='bk'` branch.

This covers the tested bk-mask forward paths, including the `LOG` branch. The
same definition keeps the optional causal branch because the Python kernel also
allows that combination. -/
def ksoftmax_forward_bk_surface
    (Y X Mask : RegionName)
    (stride_ym stride_yn stride_xm stride_xn stride_m K DEPTH : Nat)
    (LOG CAUSAL IS_FP16 : Bool) :
    ComputeKernel := triton {
  m = tl.program_id(axis=0)
  n = tl.program_id(axis=1)
  k = tl.arange(0, $(DEPTH))
  x_ptrs = X + m * $(stride_xm) + n * $(stride_xn) + k
  io_mask = k < $(K)
  if CAUSAL {
    io_mask = io_mask and (k <= n)
  }
  x = tl.load(x_ptrs, mask=io_mask, other=-inf)
  if CAUSAL {
    off = -inf
    off = (off).to(x.dtype)
    x = tl.where(k > n, off, x)
  }
  mask_ptrs = Mask + m * $(stride_m) + k
  add_mask = tl.load(mask_ptrs, mask=io_mask, other=-inf)
  x += add_mask
  z = x - tl.max(x, axis=0)
  if IS_FP16 {
    z = (z).to(tl.float32)
  }
  num = tl.exp(z)
  denom = tl.sum(num, axis=0)
  if LOG {
    y = z - tl.log(denom)
  } else {
    y = num / denom
  }
  y_ptrs = Y + m * $(stride_ym) + n * $(stride_yn) + k
  tl.store(y_ptrs, y, mask=k < $(K))
}

/-- Surface transcription of `ksoftmax_triton.py`'s `_softmax_backward`.

This covers the tested backward paths for `LOG=true/false` and
`CAUSAL=true/false`, preserving the masked gradient/out loads and the two
gradient formulas. -/
def ksoftmax_backward_surface
    (GradIn GradOut Out : RegionName)
    (stride_bm stride_bn stride_gm stride_gn stride_om stride_on K DEPTH : Nat)
    (LOG CAUSAL IS_FP16 : Bool) :
    ComputeKernel := triton {
  m = tl.program_id(axis=0)
  n = tl.program_id(axis=1)
  k = tl.arange(0, $(DEPTH))
  grad_out_ptrs = GradOut + m * $(stride_gm) + n * $(stride_gn) + k
  out_ptrs = Out + m * $(stride_om) + n * $(stride_on) + k
  io_mask = k < $(K)
  if CAUSAL {
    io_mask = io_mask and (k <= n)
  }
  g = tl.load(grad_out_ptrs, mask=io_mask, other=0.0)
  o = tl.load(out_ptrs, mask=io_mask, other=0.0)
  if CAUSAL {
    zero = 0.0
    zero = (zero).to(g.dtype)
    g = tl.where(k > n, zero, g)
    o = tl.where(k > n, zero, o)
  }
  if LOG {
    s = tl.sum(g, axis=0)
    if IS_FP16 {
      o = (o).to(tl.float32)
    }
    grad_in = g - tl.exp(o) * s
  } else {
    s = tl.sum(g * o, axis=0)
    grad_in = o * (g - s)
  }
  grad_in_ptrs = GradIn + m * $(stride_bm) + n * $(stride_bn) + k
  tl.store(grad_in_ptrs, grad_in, mask=k < $(K))
}

/-- Proof-oriented forward softmax slice of `ksoftmax_triton.py`'s `_softmax`.

This specializes the constexpr branches to:
- `LOG = false`
- `MASK_TYPE = None`
- `CAUSAL = false`
- `IS_FP16 = false`

It preserves the 2D `(m, n)` program ids, strided 3D row addressing, masked
load over the last dimension, stable softmax normalization, and masked store. -/
def ksoftmax_forward_plain
    (Y X : RegionName)
    (stride_ym stride_yn stride_xm stride_xn K DEPTH : Nat) :
    ComputeKernel := triton {
  m = tl.program_id(0)
  n = tl.program_id(1)
  k = tl.arange(0, $(DEPTH))
  x_ptrs = X + m * $(stride_xm) + n * $(stride_xn) + k
  io_mask = k < $(K)
  x = tl.load(x_ptrs, mask=io_mask, other=-inf)
  z = x - tl.max(x, axis=0)
  num = tl.exp(z)
  denom = tl.sum(num, axis=0)
  y = num / denom
  y_ptrs = Y + m * $(stride_ym) + n * $(stride_yn) + k
  tl.store(y_ptrs, y, mask=k < $(K))
}

def xOffset
    (s : BlockState) (stride_xm stride_xn : Nat) (i : Fin DEPTH) : Nat :=
  s.pids 0 * stride_xm + s.pids 1 * stride_xn + i.val

def yOffset
    (s : BlockState) (stride_ym stride_yn : Nat) (i : Fin DEPTH) : Nat :=
  s.pids 0 * stride_ym + s.pids 1 * stride_yn + i.val

noncomputable def ksoftmaxInputTile
    (s : BlockState) (X : RegionName)
    (stride_xm stride_xn K DEPTH : Nat) :
    Tile .real [DEPTH] :=
  { data := fun idx =>
      if idx.1.val < K then
        some (s.readMem X (xOffset s stride_xm stride_xn idx.1))
      else none }

noncomputable def ksoftmaxSpec
    (s : BlockState) (X : RegionName)
    (stride_xm stride_xn K DEPTH : Nat) (idx : Fin DEPTH) : ℝ :=
  let row := ksoftmaxInputTile s X stride_xm stride_xn K DEPTH
  match Tile.reduceMax (shape := [DEPTH]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let num := Tile.uop WithBot.realExp shifted
      let denom := Tile.reduceSum (shape := [DEPTH]) ⟨0, by simp⟩ Bool.false num
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR num denom).data
          (idx, PUnit.unit))
  | none => 0

/-- Algorithm-layer correctness for the plain forward softmax slice. -/
theorem ksoftmax_forward_plain_correct
    (Y X : RegionName)
    (stride_ym stride_yn stride_xm stride_xn K DEPTH : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin DEPTH => yOffset s stride_ym stride_yn i))
    (hExec : exec (ksoftmax_forward_plain Y X
        stride_ym stride_yn stride_xm stride_xn K DEPTH) s = some s') :
    ∀ i : Fin DEPTH,
      s'.readMem Y (yOffset s stride_ym stride_yn i) =
        if i.val < K then
          ksoftmaxSpec s X stride_xm stride_xn K DEPTH i
        else s.readMem Y (yOffset s stride_ym stride_yn i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [DEPTH] =>
        s.pids 0 * stride_ym + s.pids 1 * stride_yn + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [yOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hD : 0 < DEPTH
  · simp [exec, ksoftmax_forward_plain, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
          ComparableDType.lt, hD] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < K
    · simp [hi, ksoftmaxSpec, ksoftmaxInputTile, xOffset,
            Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum,
            Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
            TileShape.insertAxisIndex, hD]
      congr
    · simp [hi]
  · exact False.elim (hD (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the plain forward softmax slice. -/
theorem ksoftmax_forward_plain_compute_correct
    (Y X : RegionName)
    (stride_ym stride_yn stride_xm stride_xn K DEPTH : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin DEPTH => yOffset s stride_ym stride_yn i)) :
    ComputeCorrect.Realizes
      (kernel := ksoftmax_forward_plain Y X
        stride_ym stride_yn stride_xm stride_xn K DEPTH)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin DEPTH => i.val < K)
        (fun i => (Y, yOffset s stride_ym stride_yn i)))
      (expected := fun i => ksoftmaxSpec s X stride_xm stride_xn K DEPTH i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [ksoftmax_forward_plain]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := ksoftmax_forward_plain_correct Y X
    stride_ym stride_yn stride_xm stride_xn K DEPTH s s' hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.KsoftmaxTriton
