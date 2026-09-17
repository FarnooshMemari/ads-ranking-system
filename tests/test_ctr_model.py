"""Tests for GlobalCTRBaseline and CTRModel, using tiny synthetic fixtures."""

import numpy as np
import pandas as pd
import pytest

from src.models.ctr_model import CTRModel, GlobalCTRBaseline


class TestGlobalCTRBaseline:
    def test_predicts_training_mean_for_every_row(self):
        y = [1, 1, 0, 0, 0]  # mean = 0.4
        X = pd.DataFrame({"anything": range(5)})
        model = GlobalCTRBaseline().fit(X, y)

        preds = model.predict(pd.DataFrame({"anything": range(3)}))
        assert np.allclose(preds, 0.4)
        assert len(preds) == 3

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            GlobalCTRBaseline().predict(pd.DataFrame({"x": [1]}))

    def test_save_load_roundtrip(self, tmp_path):
        model = GlobalCTRBaseline().fit(pd.DataFrame({"x": range(4)}), [1, 0, 0, 0])
        path = tmp_path / "baseline.txt"
        model.save(str(path))

        loaded = GlobalCTRBaseline().load(str(path))
        assert loaded.global_ctr_ == pytest.approx(0.25)


def _synthetic_training_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    campaign = rng.integers(0, 5, size=n)
    # campaign 0 has a much higher click rate than the others -- a learnable signal
    click_prob = np.where(campaign == 0, 0.9, 0.1)
    click = rng.binomial(1, click_prob)
    X = pd.DataFrame(
        {
            "campaign": pd.Series(campaign, dtype="category"),
            "cat1": pd.Series(rng.integers(0, 3, size=n), dtype="category"),
            "user_clicks_before": rng.integers(0, 5, size=n).astype(float),
        }
    )
    return X, click


class TestCTRModel:
    def test_fit_predict_learns_something(self):
        X, y = _synthetic_training_frame()
        params = {"objective": "binary", "verbose": -1, "seed": 42, "num_leaves": 7}
        model = CTRModel(params, categorical_features=["campaign", "cat1"], num_boost_round=30,
                          early_stopping_rounds=None).fit(X, y)

        preds = model.predict(X)
        assert len(preds) == len(X)
        assert ((preds >= 0) & (preds <= 1)).all()

        # campaign 0 rows should get systematically higher predicted CTR
        campaign0_mean = preds[np.asarray(X["campaign"]) == 0].mean()
        other_mean = preds[np.asarray(X["campaign"]) != 0].mean()
        assert campaign0_mean > other_mean

    def test_early_stopping_uses_validation_set(self):
        X, y = _synthetic_training_frame(seed=1)
        X_val, y_val = _synthetic_training_frame(seed=2, n=100)
        params = {"objective": "binary", "verbose": -1, "seed": 42, "num_leaves": 7}
        model = CTRModel(params, categorical_features=["campaign", "cat1"], num_boost_round=200,
                          early_stopping_rounds=5)
        model.fit(X, y, eval_set=(X_val, y_val))

        assert model.booster.best_iteration > 0
        assert model.booster.best_iteration <= 200

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            CTRModel({"objective": "binary"}).predict(pd.DataFrame({"x": [1]}))

    def test_feature_importance_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            CTRModel({"objective": "binary"}).feature_importance()

    def test_feature_importance_returns_all_features(self):
        X, y = _synthetic_training_frame()
        params = {"objective": "binary", "verbose": -1, "seed": 42, "num_leaves": 7}
        model = CTRModel(params, categorical_features=["campaign", "cat1"], num_boost_round=20,
                          early_stopping_rounds=None).fit(X, y)

        importance = model.feature_importance()
        assert set(importance.index) == set(X.columns)
        assert (importance.values >= 0).all()

    def test_save_load_roundtrip_predictions_match(self, tmp_path):
        X, y = _synthetic_training_frame()
        params = {"objective": "binary", "verbose": -1, "seed": 42, "num_leaves": 7}
        model = CTRModel(params, categorical_features=["campaign", "cat1"], num_boost_round=20,
                          early_stopping_rounds=None).fit(X, y)
        preds_before = model.predict(X)

        path = tmp_path / "model.txt"
        model.save(str(path))
        loaded = CTRModel(params).load(str(path))
        preds_after = loaded.predict(X)

        assert np.allclose(preds_before, preds_after, atol=1e-6)

    def test_deterministic_with_fixed_seed(self):
        X, y = _synthetic_training_frame()
        params = {"objective": "binary", "verbose": -1, "seed": 42, "num_leaves": 7, "deterministic": True}
        model_a = CTRModel(params, categorical_features=["campaign", "cat1"], num_boost_round=20,
                            early_stopping_rounds=None).fit(X, y)
        model_b = CTRModel(params, categorical_features=["campaign", "cat1"], num_boost_round=20,
                            early_stopping_rounds=None).fit(X, y)

        assert np.allclose(model_a.predict(X), model_b.predict(X))
