/- Syntax view for deriving kernel equivalence from admitted atomic rewrites.
Experimental configuration records how each atomic assumption was selected;
the derivation works with symbolic kernel dimensions under those assumptions.
the existing algorithm projection remains the real correctness model. -/
import VeriTile.Triton.Core.Ast
import VeriTile.Spec

namespace VeriTile.Triton

instance computeKernelProgramSyntax : Spec.ProgramSyntax ComputeKernel where
  Statement := ComputeStmt
  Signature := List RegionName × List RegionName
  signature := fun (.mk inputs outputs _) => (inputs, outputs)
  body := fun (.mk _ _ body) => body

end VeriTile.Triton
