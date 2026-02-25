import pandas as pd

PROC_COL = "ProcessGuid"
LABEL_COL = "class"

ID_COLS = [
    "ProcessGuid", "ProcessId",
    "ParentProcessGuid", "ParentProcessId",
    "TargetProcessGUID", "TargetProcessId"
]

EARLY_K = 50


def load_data():
    df_train = pd.read_csv("C:/Users/PRACHEE DEWANGAN/OneDrive/Desktop/mp/ranosmware_pjt/dataset/JamilIsp-SILRAD-dataset-d4a3625/SILRAD-dataset/fasttext-trainmodel.csv")
    df_test  = pd.read_csv("C:/Users/PRACHEE DEWANGAN/OneDrive/Desktop/mp/ranosmware_pjt/dataset/JamilIsp-SILRAD-dataset-d4a3625/SILRAD-dataset/fasttext-testmodel.csv")
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

