import unittest

import numpy as np
import pandas as pd

from src.generator import OK, FaultyDataGenerator
from src.pipeline import decide, run_pipeline
from src.validator import SensorValidator


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.v = SensorValidator({"t": {"min": 0, "max": 150}}, frozen_window=10)

    def test_frozen_run_flagged_only_at_window_length(self):
        noise = list(np.random.RandomState(0).normal(0, 1, 5))
        s = pd.Series(noise + [70.0] * 10 + noise + [70.0] * 9 + noise)
        mask = self.v.frozen_mask(s)
        self.assertEqual(mask.sum(), 10)
        self.assertTrue(mask[5:15].all())

    def test_robust_z_ignores_outlier_pull(self):
        s = pd.Series([70.0, 70.5, 69.5, 70.2, 69.8] * 20 + [180.0] * 10)
        z = SensorValidator.robust_z_scores(s)
        self.assertTrue((z[s == 180] > 3.5).all())
        self.assertTrue((z[s != 180].abs() <= 3.5).all())

    def test_robust_z_mad_zero_fallback(self):
        s = pd.Series([1.0] * 9 + [100.0])
        self.assertGreater(SensorValidator.robust_z_scores(s).iloc[-1], 3.5)

    def test_decision_rule(self):
        flags = pd.DataFrame({"a": [True] + [False] * 99})
        self.assertEqual(decide(flags, 0.01).label, "SAFE")    # 1 % is not > 1 %
        flags.iloc[1, 0] = True
        self.assertEqual(decide(flags, 0.01).label, "UNSAFE")  # 2 %


class PipelineTests(unittest.TestCase):
    def test_generator_labels(self):
        df, labels = FaultyDataGenerator().generate_data()
        self.assertEqual((labels == "spike").sum(), 10)
        self.assertEqual((labels == "frozen").sum(), 20)
        self.assertTrue((df.temperature[labels == "frozen"] == 70).all())

    def test_all_faults_caught(self):
        r = run_pipeline()
        self.assertEqual(r.evaluation.loc["frozen", "frozen (/20)"], 20)
        self.assertEqual(r.evaluation.loc["ANY", "missed"], 0)
        self.assertEqual(r.verdict.label, "UNSAFE")


if __name__ == "__main__":
    unittest.main()
