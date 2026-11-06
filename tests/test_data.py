"""Unit tests for dataset parsing and preprocessing routines."""

import unittest
import pandas as pd
from src.data.preprocessing import clean_dataframe, separate_features_and_labels


class TestDataProcessing(unittest.TestCase):
    """Test data processing and separation functions."""

    def test_feature_label_separation(self):
        sample_data = {
            "IPV4_SRC_ADDR": ["192.168.1.1"],
            "IPV4_DST_ADDR": ["10.0.0.1"],
            "L4_SRC_PORT": [80],
            "L4_DST_PORT": [443],
            "IN_BYTES": [1024],
            "OUT_BYTES": [2048],
            "Label": [0],
            "Attack": ["Benign"]
        }
        df = pd.DataFrame(sample_data)
        features, labels = separate_features_and_labels(df, drop_identifiers=True)

        self.assertNotIn("IPV4_SRC_ADDR", features.columns)
        self.assertNotIn("Label", features.columns)
        self.assertIn("IN_BYTES", features.columns)
        self.assertIn("Label", labels.columns)


if __name__ == "__main__":
    unittest.main()

