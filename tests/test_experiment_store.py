from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_store import ExperimentStore, _ref_from_record, build_index_records, load_index_records, write_index_records


class ExperimentStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "data"
        self.run_dir = self.root / "model-a" / "exp-a" / "seed0"
        (self.run_dir / "renders").mkdir(parents=True)
        (self.run_dir / "summary.csv").write_text(
            "round,agent_id,obs_text\n"
            '1,alpha,"Total team score: 10"\n',
            encoding="utf-8",
        )
        (self.run_dir / "args.json").write_text(
            json.dumps({"model": "model-a", "provider": "local", "seed": 0, "exp_name": "exp-a"}),
            encoding="utf-8",
        )
        (self.run_dir / "results.json").write_text(
            json.dumps({"score": 20, "rounds": 2, "prompt_tokens": 123}),
            encoding="utf-8",
        )
        (self.run_dir / "record.jsonl").write_text(
            json.dumps({"round": 1, "agent_id": "alpha"}) + "\n",
            encoding="utf-8",
        )
        (self.run_dir / "renders" / "round_1.svg").write_text("<svg></svg>", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_build_index_records(self) -> None:
        records = build_index_records(self.root)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["label"], "model-a/exp-a/seed0")
        self.assertEqual(record["model"], "model-a")
        self.assertEqual(record["score"], 20)
        self.assertEqual(record["render_count"], 1)
        self.assertIn("summary_csv", record["artifacts"])

    def test_write_and_load_index_records(self) -> None:
        records = build_index_records(self.root)
        out_dir = Path(self.tmp.name) / "metadata"
        jsonl_path, csv_path = write_index_records(records, out_dir)
        self.assertTrue(jsonl_path.exists())
        self.assertTrue(csv_path.exists())
        self.assertEqual(load_index_records(jsonl_path)[0]["label"], "model-a/exp-a/seed0")

    def test_local_store_reads_artifacts(self) -> None:
        record = build_index_records(self.root)[0]
        ref = _ref_from_record(record, "local")
        store = ExperimentStore({ref.label: ref}, "local")
        self.assertIn("Total team score", store.read_text(ref.label, "summary.csv"))
        self.assertEqual(store.list_render_files(ref.label), ["round_1.svg"])


if __name__ == "__main__":
    unittest.main()
