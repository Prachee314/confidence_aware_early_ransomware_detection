import numpy as np
from src.data_utils import PROC_COL, LABEL_COL, ID_COLS

def get_feature_columns(df):
    return [c for c in df.columns if c not in ID_COLS + [LABEL_COL]]


def execution_features(df):
    rows, labels = [], []
    FEATURE_COLS = get_feature_columns(df)

    for pid, g in df.groupby(PROC_COL):
        X = g[FEATURE_COLS].values

        feat = np.concatenate([
            X.mean(axis=0),
            X.std(axis=0),
            X.max(axis=0),
            np.mean(np.abs(np.diff(X, axis=0)), axis=0)
        ])

        rows.append(feat)
        labels.append(g[LABEL_COL].iloc[0])

    return np.array(rows), np.array(labels)
