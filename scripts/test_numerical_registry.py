"""Identity/incomplete-evidence regressions for floating-point rule registration."""
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from scripts import numerical_registry as registry


def fixture():
    # Synthetic identity fixture, not a measured experiment or runnable graph.
    return {
        "schema_version": 1, "rule_id": "ADD-ASSOC",
        "reference": {"graph_sha256": "a" * 64, "lowering_sha256": "b" * 64},
        "candidate": {"graph_sha256": "c" * 64, "lowering_sha256": "d" * 64},
        "numerics": {"semantics_version": "test", "input_formats": {"a": "bf16"},
                     "node_formats": {"add0": "fp32"}, "accumulator_formats": {},
                     "output_formats": {"out": "bf16"}, "rounding": "rne",
                     "nan": "canonical", "subnormal": {"inputs": "preserve", "results": "preserve"},
                     "intrinsics": {}},
        "layout": {"shapes": {"a": [32]}, "strides": {"a": [1]},
                   "reduction": None, "scan": None, "dot": None},
        "backend": {"kind": "software", "implementation_version": "test", "target": "cpu",
                    "compiler": None, "compile_options": {}, "launch": {}},
        "probe": {"family": "gaussian", "roles": {"a": {"mean": 0, "std": 1}},
                  "joint_distribution": "independent", "weights": None, "seed": 410,
                  "quantization": "round-inputs", "special_values": "reject-conflicts"},
        "protocol": {"name": "two-gates", "version": "test-only", "checker_version": "test-only",
                     "bias": {"replicates": 1024}, "vars": {"replicates": 4096},
                     "decision_policy": "test-only"},
    }


class RegistryTests(unittest.TestCase):
    def test_catalog_matches_documented_rules(self):
        rules = registry.load_catalog()
        documented = re.findall(r"^\| ([A-Z][A-Z0-9-]+) \|", (
            registry.ROOT / "documents/FloatingPointRewriteRules.md").read_text(), re.M)
        self.assertEqual(set(rules), set(documented) - {"ID"})
        self.assertEqual(len(rules), 14)
        self.assertEqual(sum(r["category"] == "exact_candidate" for r in rules.values()), 4)

    def test_composite_and_structural_transforms_cannot_be_registered(self):
        from scripts import check_numerics as experiment
        removed = ("SOFTMAX-SHIFT", "SOFTMAX-ONLINE", "LAYERNORM-WELFORD", "SWIGLU-FUSE",
                   "REDUCE-REORDER", "REDUCE-SPLIT", "SCAN-REORDER", "DOT-LOWER", "DOT-ACC-FUSE",
                   "GEMM-SPLIT-K", "LAYOUT-INVERSE", "STORE-LOAD-FORWARD")
        for rule_id in removed:
            with self.subTest(rule=rule_id):
                config = fixture()
                config["rule_id"] = rule_id
                with self.assertRaisesRegex(ValueError, "atomic rule"):
                    registry.pending_record(config)
                profile = deepcopy(experiment.load_module(experiment.DEFAULT_PROFILE).PROFILE)
                profile["rules"] = [rule_id]
                with self.assertRaisesRegex(ValueError, "atomic rule"):
                    experiment.validate_profile(profile)

    def test_all_identity_dimensions_and_direction_are_bound(self):
        config = fixture()
        baseline = registry.instance_key(config)
        reversed_order = dict(reversed(list(config.items())))
        self.assertEqual(baseline, registry.instance_key(reversed_order))
        edits = [("reference", "graph_sha256", "e" * 64), ("candidate", "lowering_sha256", "f" * 64),
                 ("numerics", "rounding", "rtz"), ("numerics", "accumulator_formats", {"sum": "fp32"}),
                 ("layout", "shapes", {"a": [64]}), ("layout", "strides", {"a": [2]}),
                 ("layout", "reduction", "another-tree"), ("backend", "target", "another-device"),
                 ("backend", "compile_options", {"fast_math": True}), ("backend", "launch", {"warps": 4}),
                 ("probe", "seed", 411), ("probe", "roles", {"a": {"mean": 2, "std": 1}}),
                 ("probe", "joint_distribution", "correlated"), ("protocol", "checker_version", "different"),
                 ("protocol", "bias", {"replicates": 5000})]
        for group, key, value in edits:
            with self.subTest(group=group, key=key):
                changed = deepcopy(config)
                changed[group][key] = value
                self.assertNotEqual(baseline, registry.instance_key(changed))
        config["reference"], config["candidate"] = config["candidate"], config["reference"]
        self.assertNotEqual(baseline, registry.instance_key(config))

    def test_missing_dimensions_and_invalid_json_are_rejected(self):
        for name, fields in registry.DIMENSIONS.items():
            for field in fields:
                with self.subTest(name=name, field=field):
                    config = fixture()
                    del config[name][field]
                    with self.assertRaises(ValueError):
                        registry.pending_record(config)
        bad_values = [("reference", "graph_sha256", "not-a-hash"),
                      ("protocol", "checker_version", None), ("numerics", "input_formats", {}),
                      ("probe", "seed", float("nan")), ("probe", "seed", float("inf")),
                      ("probe", "roles", {1: "ambiguous-key"}), ("probe", "roles", ("a", "b")),
                      ("layout", "shapes", {"a": ["N"]}), ("layout", "strides", {"a": []})]
        for group, key, value in bad_values:
            with self.subTest(group=group, key=key, value=value):
                config = fixture()
                config[group][key] = value
                with self.assertRaises(ValueError):
                    registry.pending_record(config)

    def test_pending_is_never_accepted_even_for_exact_candidates(self):
        for rule_id in registry.load_catalog():
            config = fixture()
            config["rule_id"] = rule_id
            record = registry.pending_record(config)
            self.assertEqual(record["decision"], "NOT_EVALUATED")
            self.assertEqual(record["strict_fp"], {"status": "UNPROVEN", "evidence": None})
            for gate in ("bias", "vars"):
                self.assertEqual(record[gate], {"status": "NOT_RUN", "evidence": None})
            config["probe"]["seed"] += 1
            self.assertNotEqual(record["instance_key"], registry.instance_key(config))
            self.assertEqual(record["instance_key"], registry.instance_key(record["config"]))

    def test_cli_refuses_to_overwrite_existing_record(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            output = Path(directory) / "record.json"
            config.write_text(json.dumps(fixture()))
            command = ["python3", str(registry.ROOT / "scripts/numerical_registry.py"),
                       "init", str(config), "--output", str(output)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            saved = output.read_bytes()
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(saved, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
