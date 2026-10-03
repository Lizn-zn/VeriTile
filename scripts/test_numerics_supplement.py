"""Supplement regressions. Synthetic/interpreter fixtures are NOT GPU evidence."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import check_numerics_supplement as runner, supplement_numerics as supplement
from scripts.test_numerical_gates import observations

HAS_TORCH = importlib.util.find_spec("torch") is not None
INTERPRET = (HAS_TORCH and importlib.util.find_spec("triton") is not None
             and os.environ.get("TRITON_INTERPRET") == "1")


def profile():
    p = deepcopy(runner.load_module(runner.DEFAULT_PROFILE).PROFILE)
    return runner.validate_profile(p)


def fixture_bundle(root, smoke=False, fp64=False):
    """Write temporary synthetic replay inputs; no claim of GPU execution."""
    p = profile()
    p.update(shape=[2, 3], replicates=4, replicates_max=4, batch=4,
             rules=["DIV-MUL-RCP" if fp64 else "ADD-ZERO"])
    p["formats"] = [p["formats"][3 if fp64 else 2]]
    fmt, rule = p["formats"][0], p["rules"][0]
    sources = runner.source_hashes()
    for source in runner.SOURCES:
        target = root / "sources" / source.relative_to(runner.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    backend = {"kind": "synthetic-test", "implementation_version": "fixture", "target": "not-a-gpu",
               "compiler": None, "compile_options": {}, "launch": p["launch"]}
    name = f"{fmt['name']}__{rule}"
    entry = root / name
    entry.mkdir()
    lowerings = {}
    for side in (("reference", "candidate", "oracle") if fp64 else ("reference", "candidate")):
        data = f"synthetic {side}, not executable PTX".encode()
        (entry / f"{side}.0.ptx").write_bytes(data)
        lowerings[side] = [runner.sha(data)]
    runner.write_json(root / "manifest.json", {
        "bundle_version": runner.BUNDLE_VERSION, "profile": p, "sources": sources,
        "smoke": smoke, "backend": backend, "entries": [name]})
    config = runner.contract_for(p, fmt, rule, backend, sources, lowerings)
    arrays = observations()
    np.savez_compressed(entry / "observations.npz", **arrays)
    result = runner.gates.evaluate(arrays, p["gates"], runner.seed_for(p, fmt, rule), 4, smoke)
    result.update(stopping_reason="empirical_fallback", completed_replicates=4)
    runner.write_json(entry / "record.json", {
        "rule_id": rule, "format": fmt["name"], "state": "COMPLETE", "config": config,
        "instance_key": supplement.instance_key(config), "lowerings": lowerings,
        "observations_sha256": runner.sha((entry / "observations.npz").read_bytes()),
        "result": result, "decision": result["decision"]})
    return entry


class ContractTests(unittest.TestCase):
    def test_shared_sampling_gates_and_current_report_sources(self):
        old = runner.original.load_module(runner.original.DEFAULT_PROFILE).PROFILE
        p = profile()
        for key in ("shape", "distribution", "seed", "replicates", "replicates_max", "batch", "launch", "gates"):
            self.assertEqual(p[key], old[key])
        self.assertEqual(p["formats"][:3], old["formats"])
        self.assertIs(runner.gates, runner.original.gates)
        frozen = runner.read_json(runner.ROOT / "experiments/floating_point/report/experiment.json")
        self.assertEqual(runner.original.source_hashes(), frozen["sources"])

    def test_supported_matrix_and_invalid_precision(self):
        p = profile()
        self.assertEqual(len(p["rules"]), 14)
        self.assertEqual(sum(supplement.unsupported(r, f) is None for r in p["rules"] for f in p["formats"]), 43)
        for field in ("input", "output", "accumulator"):
            bad = deepcopy(p)
            bad["formats"][-1][field] = "bf16"
            with self.assertRaises(ValueError):
                runner.validate_profile(bad)

    def test_exp_sub_admission_identifies_libdevice(self):
        p = profile()
        fmt = p['formats'][2]
        lowerings = {k: ['0' * 64] for k in ('reference', 'candidate')}
        for rule, intrinsic in [('EXP-SUB', 'libdevice.exp'), ('LOG-EXP', 'tl.exp')]:
            config = supplement.contract_for(p, fmt, rule, {}, runner.source_hashes(), lowerings)
            self.assertEqual(config['numerics']['intrinsics']['exp'], intrinsic)

    def test_composite_rejected_before_creating_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "absent"
            process = subprocess.run([sys.executable, str(Path(runner.__file__)), "run",
                                      "--rules", "SOFTMAX-ONLINE", "--output", str(output)],
                                     text=True, capture_output=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("atomic rule", process.stderr)
            self.assertFalse(output.exists())

    def test_contract_binds_domain_division_and_final_cast(self):
        p = profile()
        sources = runner.source_hashes()
        lowerings = {k: ["0" * 64] for k in ("reference", "candidate", "oracle")}
        config = runner.contract_for(p, p["formats"][-1], "DIV-MUL-RCP", {}, sources, lowerings)
        self.assertEqual(config["relation"]["final_cast"], "fp32")
        self.assertEqual(config["numerics"]["intrinsics"]["div"], "ordinary Triton /")
        self.assertIn("b != 0", config["relation"]["domain"])
        changed = deepcopy(config)
        changed["relation"]["domain"] = "unconditional"
        self.assertNotEqual(supplement.instance_key(config), supplement.instance_key(changed))

    def test_replay_and_report_preserve_smoke_and_domain_outcomes(self):
        for smoke in (False, True):
            with self.subTest(smoke=smoke), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                entry = fixture_bundle(root, smoke)
                result = runner.replay(root)
                self.assertEqual(len(result["accepted"]), 0 if smoke else 1)
                runner.publish_report(root, result, root / "report")
                table = runner.read_json(root / "report/summary.json")
                self.assertEqual(table["rows"][0]["decision"], "SMOKE_ONLY" if smoke else "ACCEPT")
                record = runner.read_json(entry / "record.json")
                record.update(state="NUMERIC_EVENT", decision="INCONCLUSIVE", reason="sampled domain violation")
                runner.write_json(entry / "record.json", record)
                self.assertEqual(runner.replay(root)["accepted"], [])

    def test_replay_detects_source_ptx_observation_contract_and_decision_tampering(self):
        for kind in ("source", "ptx", "observations", "domain", "decision", "count"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                entry = fixture_bundle(root)
                if kind == "source":
                    (root / "sources" / supplement.KERNELS.relative_to(runner.ROOT)).write_text("changed")
                elif kind == "ptx":
                    (entry / "reference.0.ptx").write_text("changed")
                elif kind == "observations":
                    (entry / "observations.npz").write_bytes(b"changed")
                else:
                    record = runner.read_json(entry / "record.json")
                    if kind == "domain":
                        record["config"]["relation"]["domain"] = "unconditional"
                    elif kind == "decision":
                        record["decision"] = "REJECT"
                    else:
                        record["result"]["completed_replicates"] = 3
                    runner.write_json(entry / "record.json", record)
                with self.assertRaises(ValueError):
                    runner.replay(root)

    def test_fp64_replay_requires_the_recorded_oracle(self):
        for change in ("ptx", "missing"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                entry = fixture_bundle(root, fp64=True)
                row = runner.replay(root)["accepted"][0]
                self.assertEqual(row["config"]["numerics"]["input_formats"]["a"], "fp64")
                self.assertIn("oracle_lowering_sha256", row["config"]["numerics"]["intrinsics"])
                if change == "ptx":
                    (entry / "oracle.0.ptx").write_text("changed oracle")
                else:
                    record = runner.read_json(entry / "record.json")
                    del record["lowerings"]["oracle"]
                    runner.write_json(entry / "record.json", record)
                with self.assertRaises(ValueError):
                    runner.replay(root)

    def test_replay_rejects_column_buckets_and_previous_schema(self):
        for change in ('shape', 'schema'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                entry = fixture_bundle(root)
                if change == 'schema':
                    manifest = runner.read_json(root / 'manifest.json')
                    manifest['bundle_version'] = 'scalar-supplement-4'
                    runner.write_json(root / 'manifest.json', manifest)
                else:
                    np.savez_compressed(entry / 'observations.npz', **observations(buckets=3))
                    record = runner.read_json(entry / 'record.json')
                    record['observations_sha256'] = runner.sha((entry / 'observations.npz').read_bytes())
                    runner.write_json(entry / 'record.json', record)
                with self.assertRaisesRegex(ValueError, 'bundle schema|shape/dtype'):
                    runner.replay(root)


@unittest.skipUnless(HAS_TORCH, "optional CPU numerical wiring checks require torch")
class OracleTests(unittest.TestCase):
    def test_supplement_combines_ulp_bias_with_absolute_peak_errors(self):
        import torch
        reference = torch.tensor([[1., 2.], [4., 8.]])
        candidate = reference + reference * torch.tensor([[1., -1.], [2., -2.]]) * 2.**-23
        obs = supplement.observe(torch, reference, candidate, reference.double(), profile()['formats'][2])
        self.assertEqual(obs['delta'].tolist(), [0.])
        self.assertEqual(obs['reference_error'], 0.)
        self.assertEqual(obs['candidate_error'], 16 * 2.**-23)

    def test_runner_lifecycle_and_failure_records(self):
        import torch

        class CPUFixtureTorch:
            # Only the test redirects device allocation. The production CLI
            # refuses CPU execution and never accepts this as GPU evidence.
            def __getattr__(self, key):
                return getattr(torch, key)

            def Generator(self, device):
                return torch.Generator(device="cpu")

            def randn(self, shape, **kwargs):
                return torch.randn(shape, **{**kwargs, "device": "cpu"})

        def synthetic_pair(_torch, _triton, _kernels, _rule, inputs, _profile, _fmt):
            return [inputs[0].clone(), inputs[0].clone()], {
                k: [f"synthetic {k}, not executable PTX"] for k in ("reference", "candidate")}, None

        for outcome in ("complete", "compile_error", "domain_event"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                entry = fixture_bundle(root, smoke=True)
                manifest = runner.read_json(root / "manifest.json")
                p = manifest["profile"]
                pair = synthetic_pair if outcome != "compile_error" else RuntimeError("synthetic compiler error")
                oracle = (runner.oracle if outcome != "domain_event" else
                          supplement.NumericEvent("INCONCLUSIVE", "synthetic domain event"))
                with patch.object(runner, "launch_pair", side_effect=pair), \
                        patch.object(runner, "oracle", side_effect=oracle), contextlib.redirect_stdout(io.StringIO()):
                    decision = runner.run_instance(CPUFixtureTorch(), None, None, p, p["formats"][0],
                                                   "ADD-ZERO", entry, manifest["backend"],
                                                   manifest["sources"], True)
                record = runner.read_json(entry / "record.json")
                self.assertEqual(record["state"], {"complete": "COMPLETE", "compile_error": "ERROR",
                                                  "domain_event": "NUMERIC_EVENT"}[outcome])
                self.assertEqual(decision, "NOT_EVALUATED" if outcome == "compile_error" else "SMOKE_ONLY")
                self.assertEqual(runner.replay(root)["accepted"], [])

    def test_domain_events_leave_samples_unchanged(self):
        import torch
        for rule, a, b in (("DIV-MUL-RCP", 1., 0.), ("MUL-RCP-CANCEL", 0., 1.),
                           ("LOG-MUL", -1., 1.), ("LOG-MUL", 1., 0.)):
            inputs = [torch.full((2, 3), x) for x in (a, b, 1.)]
            before = [x.clone() for x in inputs]
            with self.subTest(rule=rule), self.assertRaises(supplement.NumericEvent):
                supplement.oracle(torch, rule, inputs)
            for x, original in zip(inputs, before):
                torch.testing.assert_close(x, original, rtol=0, atol=0)

    def test_fp64_work_error_uses_quotient_residual(self):
        import torch
        a = torch.tensor([[1., 2., 0., 2.**-140, 2.**120]], dtype=torch.float64)
        b = torch.tensor([[3., 7. + 2.**-49, 3., 7., 3.]], dtype=torch.float64)
        q = (a / b).float()
        worse = torch.nextafter(q, torch.full_like(q, float("inf")))

        def cpu_fused_errors(output):
            # Exact rationals model one rounded FMA then one rounded division.
            # This is a small CPU oracle fixture, not the production GPU oracle.
            residuals = [abs(float(Fraction(float(x)) - Fraction(float(y)) * Fraction(float(z))) / float(z))
                         for y, x, z in zip(output.flatten(), a.flatten(), b.flatten())]
            return torch.tensor(residuals, dtype=torch.float64).reshape(a.shape)

        errors = [cpu_fused_errors(output) for output in (q, worse)]
        obs = supplement.observe(torch, q, worse, a / b, profile()["formats"][-1], errors)
        for field, output in (("reference_error", q), ("candidate_error", worse)):
            exact_errors = [abs(Fraction(float(y)) - Fraction(float(x)) / Fraction(float(z)))
                            for y, x, z in zip(output.flatten(), a.flatten(), b.flatten())]
            self.assertAlmostEqual(obs[field] / float(max(exact_errors)), 1., places=14)
        # The rounded fp64 quotient equals the output, but the exact error is
        # nonzero. Subtracting that rounded oracle would wrongly report zero.
        y = 1. + 2.**-23
        b = torch.tensor([[1. + 2.**-52]], dtype=torch.float64)
        a = torch.tensor([[y * b.item()]], dtype=torch.float64)
        q = torch.tensor([[y]], dtype=torch.float32)
        self.assertEqual((a / b).item(), y)
        errors = [cpu_fused_errors(q), cpu_fused_errors(q)]
        obs = supplement.observe(torch, q, q, a / b, profile()["formats"][-1], errors)
        self.assertEqual(obs["reference_error"], 2.**-75 / b.item())
        with self.assertRaisesRegex(ValueError, "residual oracle"):
            supplement.observe(torch, q, q, a / b, profile()["formats"][-1])


@unittest.skipUnless(INTERPRET, "set TRITON_INTERPRET=1 with torch/triton for CPU wiring checks")
class InterpreterTests(unittest.TestCase):
    def test_all_43_pairs_and_masked_rectangular_layout(self):
        import torch
        import triton
        module = runner.load_module(runner.KERNELS)

        class InterpretedLaunch:
            def __init__(self, fn):
                self.fn = fn

            def __getitem__(self, grid):
                def invoke(*args, **kwargs):
                    self.fn[grid](*args, **kwargs)
                    return SimpleNamespace(asm={"ptx": "synthetic interpreter fixture, not GPU evidence"})
                return invoke

        kernels = SimpleNamespace(elementwise=InterpretedLaunch(module.elementwise),
                                  quotient_errors=InterpretedLaunch(module.quotient_errors))
        self.assertEqual(module.SUPPORTED, set(supplement.load_catalog()))
        p = profile()
        p["shape"] = [3, 17]
        generator = torch.Generator().manual_seed(314159)
        for fmt in p["formats"]:
            for rule in p["rules"]:
                if supplement.unsupported(rule, fmt):
                    continue
                with self.subTest(rule=rule, fmt=fmt["name"]):
                    # Positive *test fixture* for LOG-MUL; production still samples
                    # the configured unconditioned normal distribution.
                    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}[fmt["input"]]
                    inputs = [(torch.rand(p["shape"], generator=generator, dtype=torch.float64) + .5).to(dtype) for _ in range(3)]
                    exact = supplement.oracle(torch, rule, inputs)
                    outputs, _, errors = supplement.launch_pair(torch, triton, kernels, rule, inputs, p, fmt)
                    if errors is not None:
                        self.assertTrue(all(bool(torch.isfinite(e).all()) for e in errors))
                    for output in outputs:
                        self.assertEqual(output.dtype, torch.bfloat16 if fmt["output"] == "bf16" else torch.float32)
                        tolerance = .04 if fmt["compute"] == "bf16" else .008 if fmt["output"] == "bf16" else 2e-6
                        torch.testing.assert_close(output.double(), exact, atol=tolerance, rtol=tolerance)


if __name__ == "__main__":
    unittest.main()
