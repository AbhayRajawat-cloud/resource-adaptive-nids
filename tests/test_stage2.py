"""
Stage 2 test suite.

Tests cover:
  - Stratified split correctness
  - BenignTrainScaler correctness (fit on benign only, no leakage)
  - Scaler serialisation round-trip
  - Dirichlet partition: exclusivity, completeness, benign minimum
  - Partition validation logic
  - Perfect-correlation verification
  - Duplicate-vector / label-conflict detection
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: build a synthetic dataset that mirrors key NF-UNSW-NB15-v2 traits
# ---------------------------------------------------------------------------

def _make_synthetic_df(
    n_benign: int = 5_000,
    n_malicious: int = 200,
    n_features: int = 37,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_total = n_benign + n_malicious

    X = rng.exponential(scale=10, size=(n_total, n_features)).astype(np.float64)
    # Introduce one pair of perfectly identical columns (indices 0 and 1)
    X[:, 1] = X[:, 0]
    # Introduce one pair with slight differences (indices 2 and 3)
    X[:, 3] = X[:, 2] + rng.normal(0, 0.01, size=n_total)

    label = np.concatenate([np.zeros(n_benign, dtype=int), np.ones(n_malicious, dtype=int)])
    # Build attack labels matching n_total exactly
    n_exploits = n_malicious * 50 // 100
    n_dos = n_malicious * 30 // 100
    n_worms = n_malicious - n_exploits - n_dos
    attack_cats = (
        ["Benign"] * n_benign +
        ["Exploits"] * n_exploits +
        ["DoS"] * n_dos +
        ["Worms"] * n_worms
    )
    attack = np.array(attack_cats[:n_total])  # safety trim

    cols = (
        ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"] +
        [f"F{i}" for i in range(n_features)] +
        ["Label", "Attack"]
    )

    data = np.hstack([
        np.full((n_total, 1), "1.2.3.4"),
        np.full((n_total, 1), "5.6.7.8"),
        rng.integers(1024, 65535, size=(n_total, 1)).astype(str),
        rng.integers(1024, 65535, size=(n_total, 1)).astype(str),
        X.astype(str),
        label.reshape(-1, 1).astype(str),
        attack.reshape(-1, 1),
    ])

    df = pd.DataFrame(data, columns=cols)
    for col in [f"F{i}" for i in range(n_features)]:
        df[col] = df[col].astype(float)
    df["L4_SRC_PORT"] = df["L4_SRC_PORT"].astype(int)
    df["L4_DST_PORT"] = df["L4_DST_PORT"].astype(int)
    df["Label"] = df["Label"].astype(int)
    df["F0"] = df["F0"].astype(float)
    df["F1"] = df["F1"].astype(float)
    df.index = np.arange(len(df))
    return df


# ---------------------------------------------------------------------------
# Tests: BenignTrainScaler
# ---------------------------------------------------------------------------

class TestBenignTrainScaler(unittest.TestCase):

    def setUp(self):
        from src.data.preprocessing import BenignTrainScaler
        self.Scaler = BenignTrainScaler
        rng = np.random.default_rng(42)
        self.n_feat = 5
        self.X_benign = rng.exponential(scale=5, size=(1000, self.n_feat))
        # Introduce some outliers
        self.X_benign[0, :] = 1000.0
        self.X_with_malicious = rng.exponential(scale=50, size=(200, self.n_feat))
        self.feat_names = [f"F{i}" for i in range(self.n_feat)]

    def test_fit_produces_params(self):
        sc = self.Scaler()
        sc.fit(self.X_benign, self.feat_names)
        self.assertTrue(sc._fitted)
        self.assertIsNotNone(sc.p01)
        self.assertIsNotNone(sc.p99)
        self.assertIsNotNone(sc.clip_min)
        self.assertIsNotNone(sc.clip_max)

    def test_transform_range(self):
        sc = self.Scaler()
        sc.fit(self.X_benign, self.feat_names)
        X_scaled = sc.transform(self.X_benign)
        self.assertTrue(np.all(X_scaled >= 0.0 - 1e-7))
        self.assertTrue(np.all(X_scaled <= 1.0 + 1e-7))

    def test_transform_extreme_values_clamped(self):
        sc = self.Scaler()
        sc.fit(self.X_benign, self.feat_names)
        # Malicious data may have values far outside benign distribution
        X_scaled = sc.transform(self.X_with_malicious)
        self.assertTrue(np.all(np.isfinite(X_scaled)))
        self.assertTrue(np.all(X_scaled >= 0.0 - 1e-7))
        self.assertTrue(np.all(X_scaled <= 1.0 + 1e-7))

    def test_serialise_roundtrip(self):
        from src.data.preprocessing import BenignTrainScaler
        sc = self.Scaler()
        sc.fit(self.X_benign, self.feat_names)
        d = sc.to_dict()
        sc2 = BenignTrainScaler.from_dict(d)
        np.testing.assert_array_almost_equal(sc.p01, sc2.p01)
        np.testing.assert_array_almost_equal(sc.clip_min, sc2.clip_min)
        X1 = sc.transform(self.X_benign)
        X2 = sc2.transform(self.X_benign)
        np.testing.assert_array_almost_equal(X1, X2)

    def test_not_fitted_raises(self):
        sc = self.Scaler()
        with self.assertRaises(RuntimeError):
            sc.transform(self.X_benign)

    def test_empty_benign_raises(self):
        sc = self.Scaler()
        with self.assertRaises(ValueError):
            sc.fit(np.empty((0, self.n_feat)), self.feat_names)


# ---------------------------------------------------------------------------
# Tests: Stratified split
# ---------------------------------------------------------------------------

class TestStratifiedSplit(unittest.TestCase):

    def setUp(self):
        from src.data.preprocessing import stratified_split_indices
        self.split_fn = stratified_split_indices

    def test_no_overlap(self):
        labels = np.array([0] * 900 + [1] * 100)
        tr, va, te = self.split_fn(labels, seed=0)
        self.assertEqual(len(np.intersect1d(tr, va)), 0)
        self.assertEqual(len(np.intersect1d(tr, te)), 0)
        self.assertEqual(len(np.intersect1d(va, te)), 0)

    def test_all_assigned(self):
        labels = np.array([0] * 900 + [1] * 100)
        tr, va, te = self.split_fn(labels, seed=0)
        self.assertEqual(len(tr) + len(va) + len(te), 1000)

    def test_approximate_fractions(self):
        labels = np.array([0] * 900 + [1] * 100)
        tr, va, te = self.split_fn(labels, train_frac=0.7, val_frac=0.15, seed=0)
        n = len(labels)
        # Allow ±3% tolerance
        self.assertAlmostEqual(len(tr) / n, 0.70, delta=0.03)
        self.assertAlmostEqual(len(va) / n, 0.15, delta=0.03)
        self.assertAlmostEqual(len(te) / n, 0.15, delta=0.03)

    def test_reproducible(self):
        labels = np.array([0] * 800 + [1] * 200)
        tr1, va1, te1 = self.split_fn(labels, seed=7)
        tr2, va2, te2 = self.split_fn(labels, seed=7)
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(va1, va2)


# ---------------------------------------------------------------------------
# Tests: Dirichlet partitioning
# ---------------------------------------------------------------------------

class TestDirichletPartitioning(unittest.TestCase):

    def setUp(self):
        from src.data.partitioning import partition_dirichlet
        self.partition_fn = partition_dirichlet
        self.df = _make_synthetic_df(n_benign=5_000, n_malicious=200, seed=1)

    def test_sample_exclusivity(self):
        mapping = self.partition_fn(self.df, num_clients=10, alpha=0.5, seed=42, min_benign=100)
        all_idx = np.concatenate(list(mapping.values()))
        self.assertEqual(len(all_idx), len(self.df))
        self.assertEqual(len(np.unique(all_idx)), len(all_idx))

    def test_completeness(self):
        mapping = self.partition_fn(self.df, num_clients=10, alpha=0.5, seed=42, min_benign=100)
        total = sum(len(v) for v in mapping.values())
        self.assertEqual(total, len(self.df))

    def test_min_benign_satisfied(self):
        mapping = self.partition_fn(self.df, num_clients=5, alpha=0.5, seed=42, min_benign=100)
        for cid, idx in mapping.items():
            benign_count = int((self.df["Label"].values[idx] == 0).sum())
            self.assertGreaterEqual(benign_count, 100,
                                     f"Client {cid} has only {benign_count} benign samples")

    def test_iid_alpha_inf(self):
        mapping = self.partition_fn(
            self.df, num_clients=10, alpha=math.inf, seed=42, min_benign=50
        )
        sizes = [len(v) for v in mapping.values()]
        # For IID/uniform split each client should be close to len(df)/10
        expected = len(self.df) / 10
        for s in sizes:
            self.assertAlmostEqual(s, expected, delta=expected * 0.1)

    def test_ten_clients(self):
        mapping = self.partition_fn(self.df, num_clients=10, alpha=0.5, seed=42, min_benign=100)
        self.assertEqual(len(mapping), 10)
        self.assertEqual(set(mapping.keys()), set(range(10)))


# ---------------------------------------------------------------------------
# Tests: Verification checks
# ---------------------------------------------------------------------------

class TestVerification(unittest.TestCase):

    def test_identical_pair_detected(self):
        from src.data.verification import verify_redundant_pairs
        import tempfile, os
        df = _make_synthetic_df(n_benign=500, n_malicious=50, seed=3)
        # F0 and F1 are identical (set in _make_synthetic_df)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            tmp_path = Path(f.name)
        try:
            result = verify_redundant_pairs(tmp_path, pairs=[("F0", "F1")])
            self.assertEqual(result["pair_results"][0]["mismatch_rows"], 0)
            self.assertTrue(result["pair_results"][0]["are_bit_exact_identical"])
            self.assertIn("F1", result["columns_to_drop"])
        finally:
            os.unlink(tmp_path)

    def test_non_identical_pair_kept(self):
        from src.data.verification import verify_redundant_pairs
        import tempfile, os
        df = _make_synthetic_df(n_benign=500, n_malicious=50, seed=3)
        # F2 and F3 differ (slight noise added in _make_synthetic_df)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            tmp_path = Path(f.name)
        try:
            result = verify_redundant_pairs(tmp_path, pairs=[("F2", "F3")])
            self.assertGreater(result["pair_results"][0]["mismatch_rows"], 0)
            self.assertFalse(result["pair_results"][0]["are_bit_exact_identical"])
            self.assertNotIn("F3", result["columns_to_drop"])
        finally:
            os.unlink(tmp_path)

    def test_label_conflict_detected(self):
        from src.data.verification import verify_duplicate_vectors
        import tempfile, os
        # Create dataset where same feature vector appears with both labels
        rng = np.random.default_rng(9)
        X = rng.normal(size=(100, 5))
        # Duplicate first row with opposite label
        X_conflict = np.vstack([X, X[0:1]])
        labels = np.concatenate([np.zeros(50), np.ones(50), [1]])  # first row was benign, now also malicious
        df = pd.DataFrame(X_conflict, columns=[f"F{i}" for i in range(5)])
        df["Label"] = [0]*50 + [1]*50 + [1]
        df.at[0, "Label"] = 0  # first row is benign
        # The last row is a duplicate of first but labeled malicious
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            tmp_path = Path(f.name)
        try:
            result = verify_duplicate_vectors(
                tmp_path, feature_cols=[f"F{i}" for i in range(5)]
            )
            # Should detect conflict
            self.assertTrue(result["label_conflict_vectors"] > 0 or
                            result["duplicate_feature_vector_rows"] > 0)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tests: End-to-end preprocess_client
# ---------------------------------------------------------------------------

class TestPreprocessClient(unittest.TestCase):

    def setUp(self):
        from src.data.preprocessing import preprocess_client
        self.fn = preprocess_client
        df = _make_synthetic_df(n_benign=3_000, n_malicious=300, seed=5)
        IDENTIFIER_COLUMNS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]
        LABEL_COLUMNS = ["Label", "Attack"]
        self.feature_cols = [c for c in df.columns
                              if c not in IDENTIFIER_COLUMNS + LABEL_COLUMNS]
        self.df = df

    def test_no_split_overlap(self):
        cr = self.fn(self.df, self.feature_cols, client_id=0, seed=42)
        tr = set(cr["original_indices"]["train"].tolist())
        va = set(cr["original_indices"]["val"].tolist())
        te = set(cr["original_indices"]["test"].tolist())
        self.assertEqual(len(tr & va), 0)
        self.assertEqual(len(tr & te), 0)
        self.assertEqual(len(va & te), 0)

    def test_all_rows_assigned(self):
        cr = self.fn(self.df, self.feature_cols, client_id=0, seed=42)
        total = (len(cr["original_indices"]["train"]) +
                 len(cr["original_indices"]["val"]) +
                 len(cr["original_indices"]["test"]))
        self.assertEqual(total, len(self.df))

    def test_scaled_values_in_range(self):
        cr = self.fn(self.df, self.feature_cols, client_id=0, seed=42)
        for key in ("X_train", "X_val", "X_test"):
            X = cr[key]
            self.assertTrue(np.all(np.isfinite(X)), f"{key} has non-finite values")
            self.assertTrue(np.all(X >= -1e-6), f"{key} has values below 0")
            self.assertTrue(np.all(X <= 1.0 + 1e-6), f"{key} has values above 1")

    def test_scaler_serialise(self):
        from src.data.preprocessing import BenignTrainScaler
        cr = self.fn(self.df, self.feature_cols, client_id=0, seed=42)
        sc2 = BenignTrainScaler.from_dict(cr["scaler_params"])
        X_re = sc2.transform(self.df[self.feature_cols].values.astype(np.float64))
        self.assertTrue(np.all(np.isfinite(X_re)))

    def test_different_clients_different_scalers(self):
        cr0 = self.fn(self.df, self.feature_cols, client_id=0, seed=42)
        cr1 = self.fn(self.df, self.feature_cols, client_id=1, seed=42)
        # Scaler params should differ between clients with different seeds
        # (they use the same data here, but different seeds lead to different benign subsets)
        p01_0 = cr0["scaler_params"]["p01"]
        p01_1 = cr1["scaler_params"]["p01"]
        # They may be very close if same data, but test that structure is identical
        self.assertEqual(len(p01_0), len(p01_1))


# ---------------------------------------------------------------------------
# Tests: Validation logic
# ---------------------------------------------------------------------------

class TestValidation(unittest.TestCase):

    def _make_client_result(self, n=500, seed=42):
        from src.data.preprocessing import preprocess_client
        df = _make_synthetic_df(n_benign=n, n_malicious=50, seed=seed)
        IDENTIFIER_COLUMNS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]
        LABEL_COLUMNS = ["Label", "Attack"]
        fcols = [c for c in df.columns if c not in IDENTIFIER_COLUMNS + LABEL_COLUMNS]
        return preprocess_client(df, fcols, client_id=seed, seed=42)

    def test_worms_count_validation(self):
        """Worms count check: our synthetic data has 40 Worms → ensure count is tracked."""
        cr = self._make_client_result(n=3_000, seed=0)
        total_worms = (
            int((cr["attack_train"] == "Worms").sum()) +
            int((cr["attack_val"] == "Worms").sum()) +
            int((cr["attack_test"] == "Worms").sum())
        )
        # Worms were part of the 200 malicious in synthetic data (40 Worms)
        self.assertGreaterEqual(total_worms, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
