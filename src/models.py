import joblib
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def train_lightgbm(X, y):
    model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X, y)
    return model


def train_xgboost(X, y):
    model = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def train_rf(X, y):
    model = RandomForestClassifier(
        n_estimators=600,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def save_model(model, path):
    joblib.dump(model, path)
