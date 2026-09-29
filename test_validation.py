import unittest

from llm_safety import screen_responses
from model_utils import compute_sofa_risk_metrics


class SofaRiskMetricsTests(unittest.TestCase):
    def test_risk_bands_and_alert_counts_use_expected_thresholds(self):
        actual = [0, 4.9, 5, 9.9, 10, 12, 9, 4]
        predicted = [4, 5, 6, 8, 11, 7.9, 10, 9]

        metrics = compute_sofa_risk_metrics(actual, predicted)
        alert = metrics["high_risk_alert"]

        self.assertEqual(metrics["risk_bands"]["confusion_matrix"], [
            [1, 2, 0],
            [0, 2, 1],
            [0, 1, 1],
        ])
        self.assertEqual(alert["confusion_matrix"], [[3, 3], [1, 1]])
        self.assertEqual(alert["false_negative"], 1)
        self.assertEqual(alert["false_positive"], 3)
        self.assertAlmostEqual(alert["auroc"], 0.75)
        self.assertIsNotNone(alert["pr_auc"])

    def test_auc_is_unavailable_when_holdout_has_one_class(self):
        metrics = compute_sofa_risk_metrics([2, 3], [4, 5])

        self.assertIsNone(metrics["high_risk_alert"]["auroc"])
        self.assertIsNone(metrics["high_risk_alert"]["pr_auc"])

    def test_rejects_mismatched_or_empty_inputs(self):
        for actual, predicted in (([], []), ([1], [1, 2])):
            with self.subTest(actual=actual, predicted=predicted):
                with self.assertRaises(ValueError):
                    compute_sofa_risk_metrics(actual, predicted)


class LlmSafetyTests(unittest.TestCase):
    def test_all_responses_pass_when_no_unsafe_claims_are_found(self):
        result = screen_responses(
            ["Current observations are stable. Reassess with the responsible clinician."],
            "Heart rate: 88 bpm",
        )

        self.assertTrue(result["safe"])
        self.assertEqual(result["findings"], [])

    def test_flags_diagnoses_medication_doses_and_unsupported_measurements(self):
        responses = [
            "This is sepsis; administer norepinephrine 0.1 mcg/kg/min.",
            "The heart rate is 120 bpm.",
        ]

        result = screen_responses(responses, "Heart rate: 88 bpm")

        self.assertFalse(result["safe"])
        self.assertEqual([item["response"] for item in result["findings"]], [1, 2])
        self.assertIn("diagnosis-like term", result["findings"][0]["reasons"])
        self.assertIn("numeric measurement not found in supplied inputs: 120", result["findings"][1]["reasons"])


if __name__ == "__main__":
    unittest.main()