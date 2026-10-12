"""Check proof-producing equivalence tactics and their failure boundaries."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (
    "VectorAdd", "FlatVectorAdd", "FloatDTypeAdd", "TritonBenchVectorAddition",
    "AdamUpdateGridLaunch", "HyperConnectionsDepth", "HyperConnectionsWidth",
    "RowWiseSum",
)


class EquivalenceTacticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ["lake", "build", "VeriTile.Meta.FPProve",
             "VeriTile.Triton.Float.ExpressionRewrite",
             "VeriTile.Triton.Float.BroadcastRewrite",
             "bench.tritonbench_g.apply_penalty.FPEquiv",
             "bench.tritonbench_g.attention_fwd_triton2.FPEquiv",
             "bench.tritonbench_g.attention_fwd_triton3.FPEquiv",
             "bench.tritonbench_g.attention_kernel.FPEquiv",
             "bench.tritonbench_g.bgmv_expand_slice.Proofs.FP",
             "bench.tritonbench_g.bgmv_shrink_kernel.Proofs.FP",
             "bench.tritonbench_g.block_sparse_attn.Proofs.FP",
             "VeriTile.Triton.Float.Integer"]
            + [f"bench.examples.{family}.FPEquiv" for family in FAMILIES],
            cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def lean(self, source):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "EquivTacticCheck.lean"
            path.write_text(source)
            return subprocess.run(
                ["lake", "env", "lean", str(path)], cwd=ROOT,
                capture_output=True, text=True, timeout=180,
            )

    def test_composition_metadata_precision_and_admission_boundaries(self):
        result = self.lean((ROOT / "bench/tests/EquivTactics.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_semantic_decomposition_and_reduction_boundaries(self):
        result = self.lean((ROOT / "bench/tests/ReductionTactics.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_contextual_rewrites_keep_domains_and_precision(self):
        result = self.lean((ROOT / "bench/tests/ProfiledRewriteTactics.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_nested_expression_rewrites_preserve_branches_and_execution(self):
        result = self.lean((ROOT / "bench/tests/FPExpressionRewrite.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)
        self.assertIn("div_mul_rcp", result.stdout)

    def test_signed_minmax_hints_and_views_preserve_negative_values(self):
        result = self.lean((ROOT / "bench/tests/SignedMinMaxViews.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_positive_infinite_clamp_preserves_fp_operation_and_precision(self):
        result = self.lean((ROOT / "bench/tests/PositiveInfinityClamp.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_integral_address_arithmetic_preserves_reference_conventions(self):
        result = self.lean((ROOT / "bench/tests/FPIntegralExecution.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_runtime_signed_division_matches_truncation_and_preserves_failure(self):
        result = self.lean((ROOT / "bench/tests/RuntimeIntegerDivision.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_block_pointers_preserve_boundaries_types_and_failure(self):
        result = self.lean((ROOT / "bench/tests/FPBlockPointer.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_index_remapping_preserves_dtype_precision_and_failure(self):
        result = self.lean((ROOT / "bench/tests/FPRemap.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_dynamic_block_advance_preserves_signed_coordinates(self):
        result = self.lean((ROOT / "bench/tests/DynamicBlockAdvance.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_signed_ranges_preserve_indices_bounds_and_failure(self):
        result = self.lean((ROOT / "bench/tests/SignedRange.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_range_rewrites_check_reachable_intermediate_operands(self):
        result = self.lean((ROOT / "bench/tests/FPRangeRewrite.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_branch_rewrites_check_only_the_selected_arm(self):
        result = self.lean((ROOT / "bench/tests/FPBranchRewrite.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_signed_range_rewrites_keep_negative_visits_and_reachable_guards(self):
        result = self.lean((ROOT / "bench/tests/FPIntRangeRewrite.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_casted_csr_loads_keep_signed_storage_and_loop_bounds(self):
        result = self.lean((ROOT / "bench/tests/PointerSignedCast.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_static_pointer_sums_keep_negative_addends_until_address_conversion(self):
        result = self.lean((ROOT / "bench/tests/PointerSignedOffset.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_compile_time_print_has_no_runtime_failure(self):
        result = self.lean((ROOT / "bench/tests/StaticPrintExecution.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_block_sparse_source_retains_signed_csr_casts_and_intrinsics(self):
        result = self.lean((ROOT / "bench/tests/FPBlockSparseSource.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_offsetted_pointer_loads_keep_current_storage_precision(self):
        result = self.lean((ROOT / "bench/tests/PointerOffsetPrecision.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_atomic_execution_preserves_storage_masks_and_read_modify_write(self):
        result = self.lean((ROOT / "bench/tests/FPAtomicExecution.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_bgmv_shrink_keeps_typed_store_and_atomic_tails(self):
        result = self.lean((ROOT / "bench/tests/FPBgmvShrinkSource.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_bgmv_storage_cast_and_output_loop_specializations(self):
        result = self.lean((ROOT / "bench/tests/FPBgmvExpandSliceSource.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_dot_precision_survives_lowering_and_backend_dispatch(self):
        result = self.lean((ROOT / "bench/tests/FPDotOptions.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_broadcast_division_keeps_actual_domain_and_compute_precision(self):
        result = self.lean((ROOT / "bench/tests/FPBroadcastRewrite.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_broadcast_multiplication_keeps_load_cast_scan_and_store(self):
        result = self.lean((ROOT / "bench/tests/FPBroadcastMultiplication.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)
        self.assertIn("mul_commute", result.stdout)

    def test_attention_preserves_output_cast_store_and_mask(self):
        result = self.lean((ROOT / "bench/tests/FPAttentionSource.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_attention_bf16_output_keeps_explicit_value_cast(self):
        result = self.lean((ROOT / "bench/tests/FPAttentionFwdTriton2Source.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_attention_signed_window_and_conditional_normalization(self):
        result = self.lean((ROOT / "bench/tests/FPAttentionFwdTriton3Source.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_attention_bias_keeps_storage_precision_and_default_dot_output(self):
        result = self.lean((ROOT / "bench/tests/FPAttentionKernelSource.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_pointer_erasure_preserves_addresses_and_block_descriptors(self):
        result = self.lean((ROOT / "bench/tests/PointerDTypeErasure.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_dot_does_not_silently_discard_invalid_options(self):
        for options, diagnostic in (
            ("unknown_mode=True", "unsupported option `unknown_mode`"),
            ("allow_tf32=True, allow_tf32=False", "repeated option `allow_tf32`"),
            ('allow_tf32=False, input_precision="ieee"',
             "specify only one of allow_tf32 and input_precision"),
        ):
            with self.subTest(options=options):
                result = self.lean('''
import VeriTile.Triton
open VeriTile.Triton
def invalidDot : ComputeKernel := triton {
  a = tl.full([16, 16], 0, dtype=tl.float32)
  b = tl.full([16, 16], 0, dtype=tl.float32)
  c = tl.dot(a, b, ''' + options + ''')
}
''')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stdout)

    def test_explicit_precision_rewrites_locate_repeated_assignments(self):
        result = self.lean((ROOT / "bench/tests/GuardedRewriteTactics.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_decomposition_is_independent_of_the_fp_model(self):
        result = self.lean("""
import VeriTile.Meta.EquivDecompose
open VeriTile
open scoped VeriTile.Spec
theorem example_decomposition (R : Spec.Assumptions Nat)
    (h : Spec.Derivation R [1] [2]) : [0, 1, 9] ≡[R] [0, 2, 9] := by
  equiv_decompose
  exact h
open Lean Elab Command in
run_cmd do
  if (← getEnv).contains `VeriTile.Triton.ComputeKernel then
    throwError "Generic decomposition imported the Triton numerical model"
""")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_tactic_option_does_not_reserve_structure_field_names(self):
        result = self.lean("""
import VeriTile.Triton.Float.RewriteTactics
structure UserOptions where
  maxSteps : Nat
example : ({ maxSteps := 5 } : UserOptions).maxSteps = 5 := by
  simp (config := { maxSteps := 100 })
""")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_failed_search_explains_the_remaining_obligation(self):
        result = self.lean("""
import VeriTile.Meta.FPProve
example : VeriTile.Spec.Derivation ([] : VeriTile.Spec.Assumptions Nat) [1] [2] := by
  fp_prove
""")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fp_prove could not close this goal", result.stdout)
        self.assertIn("domain premise or structural lemma", result.stdout)
        self.assertNotIn("maximum number of heartbeats", result.stdout)


if __name__ == "__main__":
    unittest.main()
