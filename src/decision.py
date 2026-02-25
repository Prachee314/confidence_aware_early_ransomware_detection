import numpy as np

RISK_TH = 0.5

def explanation_strength(shap_values):
    return np.mean(np.abs(shap_values), axis=1)


def decision(risk, expl, expl_th):
    if risk >= RISK_TH and expl >= expl_th:
        return "ALERT"
    elif risk >= RISK_TH:
        return "DEFER"
    else:
        return "BENIGN"
