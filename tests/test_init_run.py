"""Check input links and collection metadata for experiment initialization."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class InputLinksTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        project = self.root / "project"
        (project / "scripts").mkdir(parents=True)
        (project / "config").mkdir()
        self.script = project / "scripts/init_run.py"
        shutil.copyfile(
            src=Path(__file__).resolve().parents[1] / "scripts/init_run.py", dst=self.script
        )
        self.main = self.root / "assets/main"
        self.evaluation = self.root / "assets/evaluation"
        behavior = self.root / "assets/behavior"
        for root, names in [
            (
                self.main,
                ["data/cross", "prepared", "collection", "references", "layers/activations"],
            ),
            (self.evaluation, ["data/main_100000", "data/main_500000", "appendix_a/data"]),
            (behavior, ["data", "labels", "prepared"]),
        ]:
            for name in names:
                (root / name).mkdir(parents=True)
        (self.main / "CORE_COMPLETE.json").write_text("{}\n")
        self.manifest = self.evaluation / "manifest.json"
        self.manifest.write_text(
            json.dumps({"status": "verified_complete", "generation": {"max_new_tokens": 4096}})
        )
        (project / "config/paths.json").write_text(
            json.dumps(
                {
                    "data_project": str(self.root / "assets"),
                    "main_campaign": str(self.main),
                    "evaluation_campaign": str(self.evaluation),
                    "behavior_campaign": str(behavior),
                }
            )
        )
        self.output = self.root / "run"

    def initialize(self):
        return subprocess.run(
            args=[sys.executable, str(self.script), "--run-root", str(self.output)],
            capture_output=True,
            text=True,
        )

    def test_evaluation_and_cross_model_links_are_both_available(self):
        for _ in range(2):
            result = self.initialize()
            self.assertEqual(first=result.returncode, second=0, msg=result.stderr)
        for name in ["data/main_100000", "data/main_500000", "appendix_a/data"]:
            self.assertEqual(
                first=(self.output / "main" / name).resolve(), second=self.evaluation / name
            )
        self.assertEqual(
            first=(self.output / "main/data/cross").resolve(), second=self.main / "data/cross"
        )
        self.assertTrue(expr=(self.output / "main/CORE_COMPLETE.json").is_file())
        self.assertFalse(expr=(self.output / "main/data").is_symlink())

    def test_mismatched_cap_is_rejected_before_input_links_are_created(self):
        self.manifest.write_text(
            json.dumps({"status": "verified_complete", "generation": {"max_new_tokens": 1024}})
        )
        result = self.initialize()
        self.assertNotEqual(first=result.returncode, second=0)
        self.assertIn(member="4,096-token", container=result.stderr)
        self.assertFalse(expr=self.output.exists())

    def test_data_root_link_requires_a_fresh_run(self):
        (self.output / "main").mkdir(parents=True)
        (self.output / "main/data").symlink_to(target=self.evaluation / "data")
        result = self.initialize()
        self.assertNotEqual(first=result.returncode, second=0)
        self.assertIn(member="fresh run root", container=result.stderr)
        self.assertFalse(expr=(self.evaluation / "data/cross").exists())


if __name__ == "__main__":
    unittest.main()
