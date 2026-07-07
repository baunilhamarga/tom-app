import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def write_run(
    root: Path,
    label: str,
    started_at: str,
    scores: list[int | None],
    mission: dict | None = None,
) -> None:
    run_dir = root / label
    renders_dir = run_dir / "renders"
    renders_dir.mkdir(parents=True)
    model, experiment, seed = label.split("/")
    (run_dir / "args.json").write_text(
        json.dumps({
            "model": model,
            "exp_name": experiment,
            "seed": seed,
            "started_at": started_at,
        }),
        encoding="utf-8",
    )

    records = []
    csv_rows = ["round,agent_id,obs_text,action,comm,new_belief"]
    for round_id, score in enumerate(scores, 1):
        obs_text = "No score recorded" if score is None else f"Total team score: {score}"
        record = {
            "round": round_id,
            "agent_id": "alpha",
            "obs_text": obs_text,
            "action": "wait",
            "comm": "",
            "new_belief": "belief",
        }
        records.append(record)
        csv_rows.append(f'{round_id},alpha,"{obs_text}",wait,,belief')
        (renders_dir / f"round_{round_id}.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>',
            encoding="utf-8",
        )

    (run_dir / "record.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (run_dir / "summary.csv").write_text("\n".join(csv_rows), encoding="utf-8")
    if mission is not None:
        (run_dir / "results.json").write_text(
            json.dumps({
                "score": scores[-1],
                "rounds": len(scores),
                "started_at": started_at,
                "completed_at": started_at,
                "mission": mission,
            }),
            encoding="utf-8",
        )


class DashboardBrowsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        temp_root = Path(self.tmp.name)
        self.full_root = temp_root / "full"
        self.sample_root = temp_root / "sample"

        write_run(
            self.full_root,
            "model-a/old-exp/seed0",
            "2026-01-01T10:00:00+00:00",
            [0],
            {
                "success": True,
                "status": "accomplished",
                "reason_code": "all_objectives_completed",
                "reason": "All bombs were defused.",
                "max_score": 10,
            },
        )
        write_run(
            self.full_root,
            "model-a/new-exp/seed0",
            "2026-01-03T10:00:00+00:00",
            [10, 20],
            {
                "success": False,
                "status": "failed",
                "reason_code": "round_limit_reached",
                "reason": "Not all bombs were defused before the round limit.",
                "max_score": 60,
            },
        )
        write_run(
            self.sample_root,
            "sample-model/no-outcome/seed0",
            "2025-11-01T10:00:00+00:00",
            [None],
        )
        write_run(
            self.sample_root,
            "sample-model/sample-exp/seed0",
            "2025-12-01T10:00:00+00:00",
            [90],
        )

        self.env = {
            "TOM_APP_DATA_SOURCE": "local",
            "TOM_APP_DATA_ROOT": str(self.full_root),
            "TOM_APP_SAMPLE_ROOT": str(self.sample_root),
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_app(self) -> AppTest:
        import utils

        utils.get_store.cache_clear()
        utils._clear_artifact_caches()
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    def test_latest_run_is_selected_and_date_filter_is_inclusive(self):
        with patch.dict(os.environ, self.env):
            app = self.run_app()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox[1].value, "new-exp")
            self.assertEqual(
                app.date_input[0].value,
                (date(2026, 1, 1), date(2026, 1, 3)),
            )

            app.date_input[0].set_value((date(2026, 1, 1),)).run()
            self.assertEqual(app.date_input[0].value, (date(2026, 1, 1),))
            self.assertEqual(app.selectbox[1].value, "old-exp")

            app.date_input[0].set_value(
                (date(2026, 1, 1), date(2026, 1, 1))
            ).run()
            self.assertEqual(app.selectbox[1].value, "old-exp")
            self.assertIn("Mission accomplished", [item.value for item in app.success])

            app.date_input[0].set_value(
                (date(2026, 1, 2), date(2026, 1, 2))
            ).run()
            self.assertTrue(
                any("No experiments match" in item.value for item in app.warning)
            )
            self.assertFalse(app.exception)

    def test_source_switch_is_session_local_and_selects_latest_sample(self):
        with patch.dict(os.environ, self.env):
            first_app = self.run_app()
            second_app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

            first_app.segmented_control[0].select("Sample data").run()
            self.assertEqual(first_app.selectbox[0].value, "sample-model")
            self.assertEqual(second_app.segmented_control[0].value, "Full data")
            self.assertEqual(second_app.selectbox[1].value, "new-exp")

    def test_outcome_appears_only_on_last_round_and_legacy_is_warned(self):
        with patch.dict(os.environ, self.env):
            app = self.run_app()
            self.assertFalse(app.error)

            app.slider[0].set_value(2).run()
            self.assertIn(
                "Mission failed: Not all bombs were defused before the round limit.",
                [item.value for item in app.error],
            )
            self.assertFalse(
                any("Legacy outcome inferred" in item.value for item in app.warning)
            )

            app.segmented_control[0].select("Sample data").run()
            self.assertIn("Mission accomplished", [item.value for item in app.success])
            self.assertTrue(
                any("Legacy outcome inferred" in item.value for item in app.warning)
            )

            app.date_input[0].set_value(
                (date(2025, 11, 1), date(2025, 11, 1))
            ).run()
            self.assertIn("Mission outcome unavailable", [item.value for item in app.info])


if __name__ == "__main__":
    unittest.main()
