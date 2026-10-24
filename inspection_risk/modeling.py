"""Models and calibrators: LightGBM, a logistic-regression baseline, Platt and isotonic calibration."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import BORO_LEVELS, CATEGORICAL

GRID = [{"num_leaves": leaves, "min_child_samples": child} for leaves in (7, 15, 31) for child in (20, 80)]
BASE_PARAMS = {
    "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.03, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1,
    "deterministic": True, "force_row_wise": True, "num_threads": 4,
}
MAX_ROUNDS = 2000
EARLY_STOP = 100


def model_frame(features: pd.DataFrame, numeric: list[str], cuisine_levels: list[str]) -> pd.DataFrame:
    """The model input: numeric columns as float, categorical columns with fixed category lists (so a
    category the training window never saw becomes missing instead of shifting the integer codes)."""
    frame = features[numeric].astype(float).copy()
    frame["boro"] = pd.Categorical(features["boro"], categories=BORO_LEVELS)
    levels = list(cuisine_levels) + ["Other", "Unknown"]
    frame["cuisine_group"] = pd.Categorical(features["cuisine_group"], categories=levels)
    return frame


def fit_lgbm(x_train: pd.DataFrame, y_train, x_valid: pd.DataFrame, y_valid, *, seed: int,
             params: dict | None = None) -> tuple[lgb.Booster, int, float]:
    """Train with early stopping on the validation log loss. Returns (booster, best iteration, best loss)."""
    chosen = {**BASE_PARAMS, **(params or {}), "seed": seed}
    train_set = lgb.Dataset(x_train, label=np.asarray(y_train))
    valid_set = lgb.Dataset(x_valid, label=np.asarray(y_valid), reference=train_set)
    booster = lgb.train(chosen, train_set, num_boost_round=MAX_ROUNDS, valid_sets=[valid_set],
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    loss = booster.best_score["valid_0"]["binary_logloss"]
    return booster, int(booster.best_iteration), float(loss)


def predict_lgbm(booster: lgb.Booster, frame: pd.DataFrame) -> np.ndarray:
    return booster.predict(frame, num_iteration=booster.best_iteration)


def fit_logistic(frame: pd.DataFrame, y) -> object:
    numeric = [c for c in frame.columns if c not in CATEGORICAL]
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = make_pipeline(prep, LogisticRegression(C=0.5, max_iter=2000))
    model.fit(_plain(frame), np.asarray(y))
    return model


def predict_logistic(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(_plain(frame))[:, 1]


def _plain(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in CATEGORICAL:
        out[column] = out[column].astype(object)
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Platt:
    """Sigmoid calibration: p' = sigmoid(a * logit(p) + b)."""
    name = "platt"

    def fit(self, p, y):
        model = LogisticRegression(C=1e6, max_iter=1000)
        model.fit(_logit(p).reshape(-1, 1), np.asarray(y))
        self.a = float(model.coef_[0, 0])
        self.b = float(model.intercept_[0])
        return self

    def predict(self, p):
        return 1.0 / (1.0 + np.exp(-(self.a * _logit(p) + self.b)))

    def to_dict(self):
        return {"kind": "platt", "a": self.a, "b": self.b}


class Isotonic:
    """Isotonic calibration, applied with numpy interpolation between the fitted thresholds."""
    name = "isotonic"

    def fit(self, p, y):
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(np.asarray(p, float), np.asarray(y, float))
        self.x = [float(v) for v in model.X_thresholds_]
        self.y = [float(v) for v in model.y_thresholds_]
        return self

    def predict(self, p):
        return np.interp(np.asarray(p, float), self.x, self.y)

    def to_dict(self):
        return {"kind": "isotonic", "x": self.x, "y": self.y}


def calibrator_from_dict(spec: dict):
    if spec["kind"] == "platt":
        obj = Platt()
        obj.a, obj.b = spec["a"], spec["b"]
        return obj
    if spec["kind"] == "isotonic":
        obj = Isotonic()
        obj.x, obj.y = list(spec["x"]), list(spec["y"])
        return obj
    raise ValueError(f"unknown calibrator kind {spec['kind']!r}")
