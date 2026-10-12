import VeriTile.Triton

/-! Custom masked vector addition. The addition stays inside tl.store in both
sources; the candidate changes a + b to b + a with identical addresses and
masks. Both dimensions remain symbolic. The FP entry supplies the fp32 profile;
the real entry uses mathematical arithmetic. No performance claim is made. -/

namespace VeriTile.Bench.TritonBenchG.VectorAdditionCustom
open VeriTile.Triton

/-- Faithful 1:1 transcription of `vector_addition_custom.py`'s `_add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK: tl.constexpr` → Lean `Nat` parameter. -/
def _add_kernel
    (A B C : RegionName)
    (size BLOCK : Nat) :
    ComputeKernel := triton {
  prog_id = tl.program_id(0)
  offs = prog_id * $(BLOCK) + tl.arange(0, $(BLOCK))
  a = tl.load(A + offs, mask=offs < $(size))
  b = tl.load(B + offs, mask=offs < $(size))
  tl.store(C + offs, a + b, mask=offs < $(size))
}

/-- Commute only the value expression of the original masked store. -/
def _add_commuted_kernel
    (A B C : RegionName)
    (size BLOCK : Nat) :
    ComputeKernel := triton {
  prog_id = tl.program_id(0)
  offs = prog_id * $(BLOCK) + tl.arange(0, $(BLOCK))
  a = tl.load(A + offs, mask=offs < $(size))
  b = tl.load(B + offs, mask=offs < $(size))
  tl.store(C + offs, b + a, mask=offs < $(size))
}

end VeriTile.Bench.TritonBenchG.VectorAdditionCustom
