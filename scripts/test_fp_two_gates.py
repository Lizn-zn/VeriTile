"""Statistical edge cases and replay integrity. Fixtures are NOT GPU evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import fp_experiment as experiment, fp_two_gates as gates


def profile():
    result = deepcopy(experiment.load_module(experiment.DEFAULT_PROFILE).PROFILE)
    result.update(shape=[2, 3], replicates=4, rules=["ADD-COMMUTE"])
    result["formats"] = [result["formats"][-1]]
    return experiment.validate_profile(result)


def observations(count=4, buckets=3):
    return {"delta": np.zeros((count, buckets)), "ulp": np.ones((count, buckets)),
            "reference_error": np.zeros(count), "candidate_error": np.zeros(count), "epsilon": np.ones(count)}


class GateTests(unittest.TestCase):
    def setUp(self):
        self.config = profile()["gates"]

    def test_zero_variance_bias_branches(self):
        for mean, expected in ((0, "PASS"), (0.5, "WARN"), (2, "FAIL")):
            result = gates.bias_gate(np.full((8, 2), mean), np.ones((8, 2)), self.config["bias"])
            self.assertEqual(result["status"], expected)
            json.dumps(result, allow_nan=False)

    def test_replicates_are_rows_not_elements(self):
        result = gates.bias_gate(np.ones((1, 10000)), np.ones((1, 10000)), self.config["bias"])
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_nonfinite_statistics_and_invalid_scales_never_pass(self):
        for x in (np.nan, np.inf, -np.inf):
            obs = observations()
            obs["delta"][0, 0] = x
            self.assertEqual(gates.evaluate(obs, self.config, 1, 4)["decision"], "INCONCLUSIVE")
        for x in (0, -1, np.inf, np.nan):
            self.assertEqual(gates.bias_gate(np.zeros((2, 1)), np.full((2, 1), x), self.config["bias"])["status"], "INCONCLUSIVE")
        result = gates.bias_gate(np.array([[1e308], [-1e308]]), np.ones((2, 1)), self.config["bias"])
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_additive_floor_and_zero_denominator(self):
        actual = gates.amplification([0, 0, 2, 2], [0, 2, 5, 1], [1, 1, 1, 1])
        np.testing.assert_array_equal(actual, [0, np.inf, 2, 0])
        result = gates.vars_gate([0, 1], [2, 1], [1, 1], self.config["vars"], 1)
        self.assertEqual(result["status"], "FAIL")

    def test_constant_positive_tail_is_not_a_fabricated_fit(self):
        result = gates.vars_gate(np.ones(4096), np.full(4096, 2), np.ones(4096), self.config["vars"], 1)
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertNotIn("upper", result)

    def test_pwm_and_return_level_against_exponential_quantiles(self):
        samples = -np.log1p(-(np.arange(10000) + 0.5) / 10000)
        xi, scale = gates.pwm_fit(samples)
        self.assertAlmostEqual(xi, 0, delta=0.002)
        self.assertAlmostEqual(scale, 1, delta=0.002)
        self.assertAlmostEqual(gates.return_level(1, 0, 2, 100, 0.1), 1 + 2 * np.log(10))
        # A positive shape remains positive; it is never capped at zero.
        heavy = ((1 - (np.arange(10000) + 0.5) / 10000) ** -0.2 - 1) / 0.2
        self.assertGreater(gates.pwm_fit(heavy)[0], 0.15)

    def test_bootstrap_is_replayable(self):
        config = {**self.config["vars"], "bootstrap": 16, "min_exceedances": 16}
        ratios = np.random.default_rng(12).exponential(0.1, 4096)
        args = (np.ones(4096), ratios + 1, np.ones(4096), config, 123)
        first = gates.vars_gate(*args)
        self.assertEqual(first, gates.vars_gate(*args))
        self.assertEqual(first["branch"], "pot_pwm")

    def test_warning_policy_and_smoke(self):
        obs = observations()
        obs["delta"][:] = 0.5
        self.assertEqual(gates.evaluate(obs, self.config, 1, 4)["decision"], "WARN_NOT_ACCEPTED")
        self.config["warning_policy"] = "allow_warn"
        self.assertEqual(gates.evaluate(obs, self.config, 1, 4)["decision"], "ACCEPT_WITH_WARNING")
        self.assertEqual(gates.evaluate(obs, self.config, 1, 4, smoke=True)["decision"], "SMOKE_ONLY")

    def test_incomplete_budget_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            gates.evaluate(observations(), self.config, 1, 4096)


def fixture_bundle(directory, smoke=False):
    """Synthetic replay fixture. Never retained as an experiment artifact."""
    p = profile()
    fmt, rule = p["formats"][0], p["rules"][0]
    sources = experiment.source_hashes()
    (directory / "sources").mkdir()
    for path in experiment.SOURCES:
        (directory / "sources" / path.name).write_bytes(path.read_bytes())
    backend = {"kind": "synthetic-test", "implementation_version": "fixture", "target": "not-a-gpu",
               "compiler": None, "compile_options": {}, "launch": p["launch"]}
    name = f"{fmt['name']}__{rule}"
    entry = directory / name
    entry.mkdir()
    lowerings = {}
    for side in ("reference", "candidate"):
        data = f"synthetic {side}, not executable PTX".encode()
        (entry / f"{side}.0.ptx").write_bytes(data)
        lowerings[side] = [experiment.sha(data)]
    manifest = {"bundle_version": 1, "profile": p, "sources": sources, "smoke": smoke,
                "backend": backend, "entries": [name]}
    experiment.write_json(directory / "manifest.json", manifest)
    config = experiment.contract_for(p, fmt, rule, backend, sources, lowerings)
    obs = observations()
    np.savez_compressed(entry / "observations.npz", **obs)
    result = gates.evaluate(obs, p["gates"], experiment.seed_for(p, fmt, rule), p["replicates"], smoke)
    record = {"rule_id": rule, "format": fmt["name"], "state": "COMPLETE", "config": config,
              "instance_key": experiment.registry.instance_key(config), "lowerings": lowerings,
              "observations_sha256": experiment.sha((entry / "observations.npz").read_bytes()),
              "result": result, "decision": result["decision"]}
    experiment.write_json(entry / "record.json", record)
    return entry


class ReplayTests(unittest.TestCase):
    def test_validated_configuration_and_identity(self):
        p = profile()
        for key, value in (("shape", [1, "N"]), ("replicates", True), ("rules", ["bogus"]),
                           ("distribution", {"family": "normal", "mean": 1, "std": 0})):
            bad = deepcopy(p)
            bad[key] = value
            with self.assertRaises(ValueError):
                experiment.validate_profile(bad)
        hashes = experiment.source_hashes()
        before = experiment.graph_hash("ADD-COMMUTE", "reference", p, p["formats"][0], hashes)
        hashes["scripts/fp_experiment.py"] = "0" * 64
        self.assertNotEqual(before, experiment.graph_hash("ADD-COMMUTE", "reference", p, p["formats"][0], hashes))

    def test_smoke_cannot_populate_accepted_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_bundle(Path(tmp), smoke=True)
            report = experiment.replay(Path(tmp))
            self.assertEqual(report["accepted"], [])
            self.assertEqual(report["rows"][0]["decision"], "SMOKE_ONLY")

    def test_replay_checks_decisions_data_and_lowering(self):
        for target in ("record.json", "observations.npz", "reference.0.ptx"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                entry = fixture_bundle(root)
                self.assertEqual(len(experiment.replay(root)["accepted"]), 1)
                if target == "record.json":
                    data = experiment.read_json(entry / target)
                    data["result"]["bias"]["status"] = "FAIL"
                    experiment.write_json(entry / target, data)
                else:
                    with (entry / target).open("ab") as output:
                        output.write(b"corruption")
                with self.assertRaisesRegex(ValueError, "mismatch|disagree"):
                    experiment.replay(root)

    def test_wrong_shape_rejected_even_with_updated_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = fixture_bundle(root)
            np.savez_compressed(entry / "observations.npz", **observations(buckets=2))
            data = experiment.read_json(entry / "record.json")
            data["observations_sha256"] = experiment.sha((entry / "observations.npz").read_bytes())
            experiment.write_json(entry / "record.json", data)
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                experiment.replay(root)

    def test_missing_and_error_rows_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = fixture_bundle(root)
            record = experiment.read_json(entry / "record.json")
            record.update(state="ERROR", decision="ACCEPT")
            experiment.write_json(entry / "record.json", record)
            self.assertEqual(experiment.replay(root)["accepted"], [])
            (entry / "record.json").unlink()
            self.assertEqual(experiment.replay(root)["accepted"], [])

    def test_changed_configuration_or_source_cannot_reuse_acceptance(self):
        for field in ("profile", "sources"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture_bundle(root)
                manifest = experiment.read_json(root / "manifest.json")
                if field == "profile":
                    manifest["profile"]["distribution"]["std"] = 2.0
                else:
                    manifest["sources"]["scripts/fp_two_gates.py"] = "0" * 64
                experiment.write_json(root / "manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, "identity mismatch|source hashes differ"):
                    experiment.replay(root)


if __name__ == "__main__":
    unittest.main()
