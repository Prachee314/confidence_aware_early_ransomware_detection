import pandas as pd
from pathlib import Path

PROC_COL = "ProcessGuid"
LABEL_COL = "class"

ID_COLS = [
    "ProcessGuid", "ProcessId",
    "ParentProcessGuid", "ParentProcessId",
    "TargetProcessGUID", "TargetProcessId"
]

EARLY_K = 50




BASE_DIR = Path(__file__).resolve().parent.parent

def load_data():
    train_path = BASE_DIR / "dataset" / "JamilIsp-SILRAD-dataset-d4a3625" / "SILRAD-dataset" / "fasttext-trainmodel.csv"

    test_path = BASE_DIR / "dataset" / "JamilIsp-SILRAD-dataset-d4a3625" / "SILRAD-dataset" / "fasttext-testmodel.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_test

def early_window(df, K=50):
    """
    Select first K events per execution (ProcessGuid).
    Uses EventIndex if available; otherwise relies on row order.
    """
    if "EventIndex" in df.columns:
        df = df.sort_values([PROC_COL, "EventIndex"])
    else:
        df = df.sort_index()

    return (
        df.groupby(PROC_COL)
          .head(K)
          .reset_index(drop=True)
    )

