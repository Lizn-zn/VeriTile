import bench.tritonbench_g.rmsnorm_fused_llama.Correct
import bench.tritonbench_g.rmsnorm_fused_llama.FPEquiv
import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.Precision
import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.Execution

#print axioms VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.rms_norm_fwd_fused_llama_output_summary
#print axioms VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.rms_norm_fwd_fused_llama_io_correctness
#print axioms VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.original_real_correct
#print axioms VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.candidate_real_correct
#print axioms VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.FPEquiv.source_equivalence
#print axioms VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.General.typed_full_trace_safe
#print axioms VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.NumericExecution.numeric_cells
#print axioms VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.originalTyped
#print axioms VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.typedKernel
#print axioms VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.Precision.memory_profiles
#print axioms VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.Precision.computation_profiles
#print axioms VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.Precision.variance_precision
#print axioms VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.Precision.io_roles

#print_fp_assumptions VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.FPEquiv.source_equivalence

#guard_msgs (drop info) in
#auditModuleAxioms

/- Audit every theorem in the port, including private historical helpers. -/
run_cmd do
  let env ← Lean.getEnv
  let mut count := 0
  for (name, info) in env.constants.toList do
    let userName := ((Lean.privateToUserName? name).getD name).eraseMacroScopes
    if (`VeriTile.Bench.TritonBenchG.RmsnormFusedLlama).isPrefixOf userName then
      match info with
      | .thmInfo _ =>
        VeriTile.Meta.auditAxioms name
        count := count + 1
      | _ => pure ()
  Lean.logInfo m!"Complete port theorem audit: {count} declarations"
