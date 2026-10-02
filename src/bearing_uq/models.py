"""Classifiers compared in the study. All expose predict_proba."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


def make_models(seed: int = 0) -> dict:
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1),
        "svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                class_weight="balanced", random_state=seed)),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, eval_metric="mlogloss")
    return models


def fit(model, X, y):
    """Fit with class balancing (XGBoost has no class_weight, so use sample weights)."""
    if XGBClassifier is not None and isinstance(model, XGBClassifier):
        return model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return model.fit(X, y)
