import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class RoundControlTests(unittest.TestCase):
    def test_slider_and_arrows_keep_round_controls_in_sync(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "test-model" / "test-experiment" / "seed0"
            renders_dir = run_dir / "renders"
            renders_dir.mkdir(parents=True)

            rows = ["round,agent_id,obs_text,action,comm,new_belief"]
            for round_id in range(1, 6):
                rows.append(
                    f'{round_id},alpha,"Total team score: 0",wait,,belief'
                )
                (renders_dir / f"round_{round_id}.svg").write_text(
                    '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>',
                    encoding="utf-8",
                )
            (run_dir / "summary.csv").write_text("\n".join(rows), encoding="utf-8")

            env = {
                "TOM_APP_DATA_SOURCE": "local",
                "TOM_APP_DATA_ROOT": temp_dir,
            }
            app_path = Path(__file__).parents[1] / "app.py"
            with patch.dict(os.environ, env):
                import utils

                utils.get_store.cache_clear()
                utils._clear_artifact_caches()
                app = AppTest.from_file(str(app_path), default_timeout=30).run()

                self.assertFalse(app.exception)
                self.assertEqual((app.slider[0].value, app.number_input[0].value), (1, 1))

                next(
                    button for button in app.button if button.label == "\u25b6"
                ).click().run()
                self.assertEqual((app.slider[0].value, app.number_input[0].value), (2, 2))
                self.assertIn("### Round 2", [item.value for item in app.markdown])

                app.slider[0].set_value(5).run()
                self.assertEqual((app.slider[0].value, app.number_input[0].value), (5, 5))
                self.assertIn("### Round 5", [item.value for item in app.markdown])

                next(
                    button for button in app.button if button.label == "\u25c0"
                ).click().run()
                self.assertEqual((app.slider[0].value, app.number_input[0].value), (4, 4))
                self.assertIn("### Round 4", [item.value for item in app.markdown])


if __name__ == "__main__":
    unittest.main()
