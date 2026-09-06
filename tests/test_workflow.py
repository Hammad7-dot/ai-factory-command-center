import json
import sqlite3
from contextlib import closing
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from factory.workflow import analyze_note, generate_explanation, retrieve, run_agents, save_decision, simulate


class WorkflowTests(unittest.TestCase):
    def test_negation(self):
        result = analyze_note("No vibration or noise. Bearing is overheating.")
        self.assertNotIn("vibration", result["symptoms"])
        self.assertNotIn("noise", result["symptoms"])
        self.assertIn("overheating", result["symptoms"])

    def test_retrieval_abstains(self):
        self.assertEqual(retrieve("zzzzqxyq"), [])
        matches = retrieve("specialbearing", [{"source": "uploaded", "text": "Inspect specialbearing weekly."}])
        self.assertEqual(matches[0]["source"], "uploaded")

    def test_simulation_monotone(self):
        low, high = simulate(.1, .1), simulate(.9, .1)
        self.assertEqual(len(low), 3)
        self.assertGreater(high[0]["expected_cost"], low[0]["expected_cost"])
        self.assertLess(high[0]["expected_units"], low[0]["expected_units"])
        self.assertTrue(all(0 <= s["downtime_hours"] <= 1 for s in simulate(.8, .2, hours=1)))
        with self.assertRaises(ValueError):
            simulate(float("nan"), .1)

    def test_audit_and_offline(self):
        incident = run_agents({"failure_probability": .8}, {"defect_probability": .4}, "vibration")
        self.assertEqual(len(incident["messages"]), 4)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "audit.sqlite"
            with self.assertRaises(ValueError):
                save_decision(incident, "modify", "", "maintenance", path)
            result = save_decision(incident, "modify", "Supervisor checked bearing", "maintenance", path)
            with closing(sqlite3.connect(path)) as db:
                snapshot = json.loads(db.execute("SELECT incident_json FROM decisions").fetchone()[0])
            self.assertEqual(snapshot["incident_id"], result["incident_id"])
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            self.assertEqual(generate_explanation(incident)["mode"], "offline_extractive")

    def test_horizon_conversion(self):
        self.assertAlmostEqual(simulate(.4, .1, hours=6)[0]["failure_probability"], .4)
        self.assertAlmostEqual(simulate(.4, .1, hours=12)[0]["failure_probability"], .64)

    def test_image_probability_not_batch_yield(self):
        self.assertEqual(simulate(.4, .01), simulate(.4, .99))
        self.assertLess(simulate(.4, .01, observed_reject_rate=.5)[0]["expected_units"],
                        simulate(.4, .01, observed_reject_rate=.01)[0]["expected_units"])
        incident = run_agents({"probability": .4, "features": [{"feature": "vibration_rms"}],
                               "observed_reject_rate": .1}, {"defect_probability": .9}, "")
        self.assertTrue(incident["recommendation"]["quality_review_required"])
        self.assertIn("vibration rms", incident["messages"][2]["payload"]["query"])
        self.assertIn("surface defect", incident["messages"][2]["payload"]["query"])


if __name__ == "__main__":
    unittest.main()
