import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.MatrixTranspose
open VeriTile.Triton
open Lean

/-- Faithful 1:1 transcription of `matrix_transpose.py`'s `kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `SIZE_M: tl.constexpr` / `D_HEAD: tl.constexpr` → Lean `Nat`
  parameters. -/
def kernel
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat) :
    ComputeKernel := triton {
  size_m_arange = tl.arange(0, $(SIZE_M))
  d_head_arange = tl.arange(0, $(D_HEAD))
  matrix_ptr = M + size_m_arange[:, None] * $(matrix_stridex)
                + d_head_arange[None, :] * $(matrix_stridey)
  out_ptr = Out + d_head_arange[:, None] * $(out_stridex)
             + size_m_arange[None, :] * $(out_stridey)
  matrix = tl.load(matrix_ptr)
  tl.store(out_ptr, tl.trans(matrix))
}


/-! The wrapper allocates fp16 input and output. Both pointer tiles retain
all four supplied strides. The candidate changes only exact index multiplication. -/
syntax "transposeSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(transposeSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let inputPtr : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| matrix_ptr = M + size_m_arange[:, None] * $(matrix_stridex)
          + d_head_arange[None, :] * $(matrix_stridey))
      else `(tritonStmt| matrix_ptr = M + $(matrix_stridex) * size_m_arange[:, None]
          + d_head_arange[None, :] * $(matrix_stridey))
    `(fun (M Out : RegionName)
        (matrix_stridex matrix_stridey out_stridex out_stridey SIZE_M D_HEAD : Nat) =>
      ComputeKernel.mk [M] [Out] (triton {
  tl.region M = $storage, Out = $storage
  size_m_arange = tl.arange(0, $(SIZE_M))
  d_head_arange = tl.arange(0, $(D_HEAD))
  $inputPtr
  out_ptr = Out + d_head_arange[:, None] * $(out_stridex)
    + size_m_arange[None, :] * $(out_stridey)
  matrix = tl.load(matrix_ptr)
  tl.store(out_ptr, tl.trans(matrix))
}).surfaceBody)

abbrev Source := RegionName → RegionName → Nat → Nat → Nat → Nat → Nat → Nat → ComputeKernel

/-- Complete float16 source used by the Python wrapper. -/
def typedKernel (commuted : Bool) : Source :=
  if commuted then transposeSource%[tl.float16, indexCommuted]
  else transposeSource%[tl.float16, original]

end VeriTile.Bench.TritonBenchG.MatrixTranspose
