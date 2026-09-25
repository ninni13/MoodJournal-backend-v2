"""Guard against silent population/probability corruption in report inputs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('summarize_results', ROOT / 'src/analysis/summarize_results.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class PredictionIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'predictions.csv'
        self.expected = pd.DataFrame({'id': [f'u{i}' for i in range(7)], 'label': summary.LABELS})
        self.frame = self.expected.rename(columns={'label': 'true_label'}).copy()
        self.frame['predicted_label'] = summary.LABELS
        for i, col in enumerate(summary.PROBS):
            self.frame[col] = np.eye(7)[:, i]

    def check(self, frame):
        frame.to_csv(self.path, index=False)
        return summary.checked_predictions(self.path, self.expected)

    def test_reorders_by_id_not_row_position(self):
        result = self.check(self.frame.iloc[::-1])
        self.assertEqual(result.id.tolist(), self.expected.id.tolist())

    def test_rejects_population_or_target_corruption(self):
        cases = {}
        duplicate = self.frame.copy(); duplicate.loc[1, 'id'] = duplicate.loc[0, 'id']
        cases['duplicate'] = duplicate
        cases['missing'] = self.frame.iloc[:-1]
        wrong = self.frame.copy(); wrong.loc[0, 'true_label'] = 'Fear'
        cases['label'] = wrong
        for name, frame in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.check(frame)

    def test_rejects_invalid_probabilities_and_argmax(self):
        for name, change in [('nan', ('prob_Anger', np.nan)), ('negative', ('prob_Anger', -1)), ('sum', ('prob_Anger', 0.5)), ('argmax', ('predicted_label', 'Fear'))]:
            frame = self.frame.copy(); frame.loc[0, change[0]] = change[1]
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.check(frame)


if __name__ == '__main__':
    unittest.main()
