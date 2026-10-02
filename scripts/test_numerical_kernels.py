"""CPU-only wiring/oracle regressions; never create gate records.

Set TRITON_INTERPRET=1 to exercise the kernels through Triton's interpreter.
These checks cover masks, indexing and formulas, not GPU instruction rounding.
"""
from copy import deepcopy
import importlib.util
import os
from types import SimpleNamespace
import unittest

from scripts import check_numerics as experiment

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRITON = importlib.util.find_spec("triton") is not None


@unittest.skipUnless(HAS_TORCH, "optional CPU oracle checks require torch")
class OracleTests(unittest.TestCase):
    def test_ulp_including_binade_boundary_and_subnormals(self):
        import torch
        values = torch.tensor([0, 1, 2, 0.5, 2.0**-140], dtype=torch.float64)
        actual = experiment.ulp(torch, values, "fp32")
        expected = torch.tensor([2.0**-149, 2.0**-23, 2.0**-22, 2.0**-24, 2.0**-149], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual = experiment.ulp(torch, values, "bf16")
        expected = torch.tensor([2.0**-133, 2.0**-7, 2.0**-6, 2.0**-8, 2.0**-133], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_domain_is_not_silently_conditioned(self):
        import torch
        inputs = [torch.tensor([-1.0, 1]), torch.tensor([0.0, 1]), torch.ones(2)]
        for rule in ("SQRT-RSQRT", "DIV-RCP"):
            with self.assertRaises(experiment.NumericEvent) as error:
                experiment.oracle(torch, rule, inputs)
            self.assertEqual(error.exception.status, "INCONCLUSIVE")

    def test_nonfinite_candidate_reference_and_oracle_fail(self):
        import torch
        from scripts import numerical_gates as gates
        x = torch.ones((2, 3))
        nan = x * float("nan")
        for reference, candidate, oracle in ((x, nan, x), (nan, x, x), (x, x, nan)):
            obs = experiment.observe(torch, reference, candidate, oracle, "fp32")
            k = gates.amplification([obs["reference_error"]], [obs["candidate_error"]])
            self.assertEqual(k[0], float("inf"))

    def test_vector_buckets_do_not_cancel_and_scale_uses_golden(self):
        import torch
        ref = torch.tensor([1., 1.])
        cand = torch.tensor([1.25, .75])
        obs = experiment.observe(torch, ref, cand, torch.ones(2).double(), "fp32")
        self.assertEqual(obs["delta"].tolist(), [2.**21, -2.**21])
        ref = torch.ones((2, 3))
        cand = torch.full((2, 3), 2.)
        obs = experiment.observe(torch, ref, cand, ref.double(), "fp32")
        self.assertEqual(obs["candidate_error"], 2.**23)
        self.assertEqual(obs["delta"].tolist(), [2.**23] * 3)

    def test_large_exact_output_does_not_hide_small_output_error(self):
        import torch
        from scripts import numerical_gates as gates
        # A large, error-free row used to inflate the allowance for both rows.
        # A two-ULP error near 1 must still count as two local ULPs.
        for fmt, dtype, step in (("bf16", torch.bfloat16, 2.**-7),
                                 ("fp32", torch.float32, 2.**-23)):
            with self.subTest(fmt=fmt):
                ref = torch.tensor([[1.], [1024.]], dtype=dtype)
                cand = torch.tensor([[1. + 2 * step], [1024.]], dtype=dtype)
                obs = experiment.observe(torch, ref, cand, ref.double(), fmt)
                self.assertEqual(obs["delta"].tolist(), [1.])
                self.assertEqual(obs["candidate_error"], 2.)
                self.assertEqual(gates.amplification([obs["reference_error"]],
                                                   [obs["candidate_error"]])[0], float("inf"))

    def test_normalization_precedes_mean_and_is_power_of_two_invariant(self):
        import torch
        for fmt, dtype, step in (("bf16", torch.bfloat16, 2.**-7),
                                 ("fp32", torch.float32, 2.**-23)):
            # +1 ULP at 1 and -1 ULP at 1024 have a zero normalized mean.
            ref = torch.tensor([[1.], [1024.]], dtype=dtype)
            cand = torch.tensor([[1. + step], [1024. - 1024. * step]], dtype=dtype)
            for multiplier in (1., 8.):
                with self.subTest(fmt=fmt, multiplier=multiplier):
                    obs = experiment.observe(torch, ref * multiplier, cand * multiplier,
                                             ref.double() * multiplier, fmt)
                    self.assertEqual(obs["delta"].tolist(), [0.])
                    self.assertEqual(obs["reference_error"], 0.)
                    self.assertEqual(obs["candidate_error"], 1.)

    def test_zero_and_subnormal_golden_use_minimum_spacing(self):
        import torch
        for fmt, dtype, step in (("bf16", torch.bfloat16, 2.**-133),
                                 ("fp32", torch.float32, 2.**-149)):
            ref = torch.tensor([0., step], dtype=dtype)
            cand = torch.tensor([step, 2 * step], dtype=dtype)
            obs = experiment.observe(torch, ref, cand, ref.double(), fmt)
            self.assertEqual(obs["delta"].tolist(), [1., 1.])
            self.assertEqual(obs["candidate_error"], 1.)

    def test_overflowed_golden_scale_cannot_hide_finite_output_errors(self):
        import torch
        from scripts import numerical_gates as gates
        from scripts.test_numerical_gates import profile
        for fmt, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
            golden = torch.full((2, 1), 2. * torch.finfo(dtype).max, dtype=torch.float64)
            reference, candidate = torch.ones((2, 1), dtype=dtype), torch.zeros((2, 1), dtype=dtype)
            obs = experiment.observe(torch, reference, candidate, golden, fmt)
            self.assertEqual(gates.bias_gate([obs["delta"], obs["delta"]],
                                            profile()["gates"]["bias"])["status"], "FAIL")
            self.assertEqual(gates.amplification([obs["reference_error"]],
                                               [obs["candidate_error"]])[0], float("inf"))

    def test_ulp_rounds_scale_and_handles_max_finite(self):
        import torch
        for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
            maximum = torch.tensor(torch.finfo(dtype).max, dtype=dtype)
            inward = maximum.double() - torch.nextafter(maximum, torch.zeros_like(maximum)).double()
            self.assertEqual(experiment.ulp(torch, maximum, name).item(), inward.item())
        # This rounds to bf16 2.0: spacing must be above that rounded value.
        self.assertEqual(experiment.ulp(torch, torch.tensor(1.999, dtype=torch.float64), "bf16").item(), 2.**-6)


@unittest.skipUnless(HAS_TORCH and HAS_TRITON and os.environ.get("TRITON_INTERPRET") == "1",
                     "optional wiring checks require torch, triton and TRITON_INTERPRET=1")
class TemplateWiringTests(unittest.TestCase):
    def test_all_templates_masked_rectangular_fp32(self):
        import torch
        import triton
        module = experiment.load_module(experiment.KERNELS)

        class InterpretedLaunch:
            def __init__(self, kernel):
                self.kernel = kernel

            def __getitem__(self, grid):
                def invoke(*args, **kwargs):
                    self.kernel[grid](*args, **kwargs)
                    # Production expects compiler output. This test stub is never
                    # serialized and never passed to the gate runner or importer.
                    return SimpleNamespace(asm={"ptx": "interpreter-wiring-test-only"})
                return invoke

        kernels = SimpleNamespace(SUPPORTED=module.SUPPORTED, elementwise=InterpretedLaunch(module.elementwise))
        self.assertEqual(module.SUPPORTED, set(experiment.registry.load_catalog()))
        profile = experiment.validate_profile(deepcopy(experiment.load_module(experiment.DEFAULT_PROFILE).PROFILE))
        profile["shape"] = [3, 17]  # Non-square batch with a masked final tile.
        fmt = profile["formats"][-1]
        generator = torch.Generator().manual_seed(823)
        for rule in profile["rules"]:
            if rule == "BF16-WIDEN-RETURN":
                continue  # Production marks this fp32 instance unsupported.
            with self.subTest(rule=rule):
                inputs = [torch.rand(experiment.shapes_for(profile, rule)[k], generator=generator) + 0.5
                          for k in ("a", "b", "c")]
                exact = experiment.oracle(torch, rule, inputs)
                outputs, _ = experiment.launch_pair(torch, triton, kernels, rule, inputs, profile, fmt)
                for output in outputs:
                    # Cast candidates explicitly quantize to bf16 even with fp32
                    # storage; all other cases exercise fp32 indexing/formulas.
                    tolerance = 0.02 if rule in ("CAST-MOVE", "CAST-REMOVE") else 2e-5
                    torch.testing.assert_close(output.double(), exact, rtol=tolerance, atol=tolerance)


if __name__ == "__main__":
    unittest.main()
