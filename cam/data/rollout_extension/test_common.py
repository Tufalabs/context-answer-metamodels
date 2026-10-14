"""Regression checks for censored-answer selection and total-token budgets."""

import unittest

from .common import continuation_budget, continuation_seed


class ContinuationTests(unittest.TestCase):
    def budget(self, *, reason="length", count=1024, prompt_length=20):
        return continuation_budget(rollout={"finish_reason": reason, "generated_token_ids": [1] * count,
                                            "generated_token_count": count},
                                   prompt_tokens=[2] * prompt_length, source_cap=1024,
                                   target_cap=4096, context_limit=8192)

    def test_total_cap_includes_saved_prefix(self):
        self.assertEqual(self.budget(), 3072)

    def test_natural_stop_at_cap_is_reused(self):
        self.assertEqual(self.budget(reason="stop"), 0)
        self.assertEqual(self.budget(reason="stop", count=40), 0)

    def test_context_truncation_is_not_silently_extended(self):
        with self.assertRaises(ValueError):
            self.budget(count=1000)
        with self.assertRaises(ValueError):
            self.budget(prompt_length=4097)

    def test_seed_is_stable_and_identity_specific(self):
        self.assertEqual(continuation_seed(identity="source:4"), continuation_seed(identity="source:4"))
        self.assertNotEqual(continuation_seed(identity="source:4"), continuation_seed(identity="source:5"))


if __name__ == "__main__":
    unittest.main()
