"""Regression checks for the core evaluator's asymmetric R² CI schema."""

import copy
import unittest
from cam.data.representation_spec import DOMAINS
from cam.data.representation_spec import core_evaluations
from cam.data.representation_spec import result
from cam.data.representation_spec import validate_evaluations


class CoreReuseTests(unittest.TestCase):
    def setUp(self):
        self.core = {
            "status": "complete",
            "smoke": False,
            "domains": DOMAINS,
            "n_train_contexts": 100,
            "point": {},
        }
        for family in ("linear", "mlp", "flow"):
            metrics = {"sample_mean_r2": -0.125}
            if family != "flow":
                metrics.update(sample_mean_r2_ci_low=-0.2, sample_mean_r2_ci_high=-0.05)
            self.core["point"][family] = {domain: dict(metrics) for domain in DOMAINS}

    def test_actual_schema_preserves_points_and_recorded_intervals(self):
        for family, representation in [
            ("linear", "last_token"),
            ("mlp", "32_bins"),
            ("flow", "32_bins"),
        ]:
            metrics = core_evaluations(self.core, family, 100)
            value = result(family, representation, 100, metrics, reused=True)
            for domain in DOMAINS:
                self.assertEqual(value["evaluations"][domain]["r2"], -0.125)
                self.assertEqual(metrics[domain]["r2_ci_low"], None if family == "flow" else -0.2)
                self.assertEqual(
                    metrics[domain]["r2_ci_status"],
                    "not_recorded_in_source" if family == "flow" else "recorded",
                )

    def test_missing_interval_is_only_allowed_for_reused_flow(self):
        metrics = core_evaluations(self.core, "flow", 100)
        for family, representation, reused in [
            ("flow", "32_bins", False),
            ("flow", "last_token", True),
            ("mlp", "32_bins", True),
        ]:
            with self.assertRaises(AssertionError):
                result(family, representation, 100, metrics, reused=reused)

    def test_partial_or_missing_nonflow_intervals_are_rejected(self):
        for family in ("linear", "mlp", "flow"):
            core = copy.deepcopy(self.core)
            core["point"][family]["LMSYS"].pop("sample_mean_r2_ci_low", None)
            core["point"][family]["LMSYS"]["sample_mean_r2_ci_high"] = 0.2
            with self.assertRaises(AssertionError):
                core_evaluations(core, family, 100)

    def test_nonfinite_points_and_wrong_counts_are_rejected(self):
        metrics = core_evaluations(self.core, "flow", 100)
        metrics["LMSYS"]["r2"] = float("nan")
        with self.assertRaises(AssertionError):
            validate_evaluations(metrics, True)
        with self.assertRaises(AssertionError):
            core_evaluations(self.core, "flow", 500000)
        self.core["domains"] = {**DOMAINS, "WeirdChat": 532}
        with self.assertRaises(AssertionError):
            core_evaluations(self.core, "flow", 100)


if __name__ == "__main__":
    unittest.main()
