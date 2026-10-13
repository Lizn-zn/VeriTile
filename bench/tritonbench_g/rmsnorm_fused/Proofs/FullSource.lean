import bench.tritonbench_g.rmsnorm_fused.Proofs.FullMath

namespace VeriTile.Bench.TritonBenchG.RmsnormFused.General
open VeriTile.Triton

def pre (X Y : RegionName) (stride B : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "row" (Op.programId 0),
    Stmt.assign .ptr [] "Y"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase Y)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "row") (Op.constNat stride))),
    Stmt.assign .ptr [] "X"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase X)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "row") (Op.constNat stride))),
    Stmt.assign .real [B] "_var" (Op.full [B] (Op.const 0)) ]

/-- Pass 1's body: masked `x` load, the `tl.where` re-zeroing, and the
`_var += x * x` fold. Register-only apart from the masked load. -/
def varBody (N B : Nat) : List Stmt :=
  [ Stmt.assign .nat [B] "cols"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "off") (Op.arange B)),
    Stmt.assign .real [B] "x"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "X") (Op.ref .nat [B] "cols")))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR (Op.ref .nat [B] "cols") (Op.constNat N))
          ((Op.const 0.0).broadcast [B]))),
    Stmt.assign .real [B] "x"
      ((Op.lt .nat Broadcast.scalarR (Op.ref .nat [B] "cols") (Op.constNat N)).where
        (Op.ref .real [B] "x") ((Op.const 0.0).broadcast [B])),
    Stmt.assign .real [B] "_var"
      (Op.add .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [B] "_var")
        (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [B] "x")
          (Op.ref .real [B] "x"))) ]

/-- The reduce / reciprocal-sqrt tail between the two passes. -/
def post (N B : Nat) (eps : ℝ) : List Stmt :=
  [ Stmt.assign .real [] "var"
      (Op.div .real Broadcast.nil
        (Op.reduceSum (shape := [B]) ⟨0, by simp⟩ Bool.false (Op.ref .real [B] "_var"))
        (Op.const (N : ℝ))),
    Stmt.assign .real [] "rstd"
      (Op.div .real Broadcast.nil (Op.const 1)
        (Op.sqrt (Op.add .real Broadcast.nil (Op.ref .real [] "var") (Op.const eps)))) ]

/-- Pass 2's body — the **emit** body: two masked loads, the affine scaling,
and the in-loop masked `.real` store into `Y`. -/
def writeBody (W : RegionName) (N B : Nat) : List Stmt :=
  [ Stmt.assign .nat [B] "cols"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "off") (Op.arange B)),
    Stmt.assign .bool [B] "mask"
      (Op.lt .nat Broadcast.scalarR (Op.ref .nat [B] "cols") (Op.constNat N)),
    Stmt.assign .real [B] "w"
      (Op.load .real (MemAccess.region W (Op.ref .nat [B] "cols"))
        (MaskOpt.mask (Op.ref .bool [B] "mask"))),
    Stmt.assign .real [B] "x"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "X") (Op.ref .nat [B] "cols")))
        (MaskOpt.maskOther (Op.ref .bool [B] "mask")
          ((Op.const 0.0).broadcast [B]))),
    Stmt.assign .real [B] "x_hat"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [B] "x") (Op.ref .real [] "rstd")),
    Stmt.assign .real [B] "y"
      (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [B] "x_hat")
        (Op.ref .real [B] "w")),
    Stmt.store .real [B]
      (MemAccess.ptr
        (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "Y") (Op.ref .nat [B] "cols")))
      (Op.ref .real [B] "y") (MaskOpt.mask (Op.ref .bool [B] "mask")) ]

/-- The lowered body is `prologue ++ pass 1 ++ tail ++ pass 2`, definitionally. -/
theorem body_decomp (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (rms_norm_fwd_fused X Y W stride N B eps).toAlgKernel.body
      = pre X Y stride B
        ++ [Stmt.forRange "off" 0 N B (varBody N B)]
        ++ post N B eps
        ++ [Stmt.forRange "off" 0 N B (writeBody W N B)] := rfl

end VeriTile.Bench.TritonBenchG.RmsnormFused.General
