"""
This script tests H3B:

Respiration has functional relevance reflected in modulation of response time across different phases of the respiratory cycle.

"""
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_data, LMEM_analysis
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# PARAMETERS
N_NULL_LMEM = 500 # 10000!!
OUTLIER_THRESHOLD = 3
SIMPLE_MODEL = True
RT_COL = "log_rt" # could also be "rt"

# Suppress warnings from statsmodels
SUPPRESS_WARNINGS = True

if SUPPRESS_WARNINGS:
    import warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="statsmodels")


def LMEM(data):
    # model specification in R would be
    # log_rt ~ sin_phase + cos_phase + (sin_phase + cos_phase + intensity | participant)

    model = smf.mixedlm(
        f"{RT_COL} ~ sin_phase + cos_phase",  # Fixed effects
        data=data,
        groups=data["participant"],  # Random effects
        re_formula="~ sin_phase + cos_phase + intensity"  # Controlling for intensity by including it as random slope
    )
    
    # SIMPLE IF IT DOES NOT CONVERGE
    # log_rt ~ sin_phase + cos_phase + (intensity | participant)
    if SIMPLE_MODEL:
        model = smf.mixedlm(
            f"{RT_COL} ~ sin_phase + cos_phase",  # Fixed effects
            data=data,
            groups=data["participant"],  # Random effects
            re_formula="~intensity"  # Controlling for intensity by including it as random slope
        )

    return model.fit()



if __name__ == "__main__":

    figpath = Path(__file__).parent / "results"

    data = load_data(["rt", "circ", "intensity"])

    LMEM_data = pd.DataFrame(columns=["participant", "rt", "cos_phase", "sin_phase", "intensity"])
    
    for participant, values in data.items():
        # remove nans from rt
        rt = np.array(values["rt"])
        # find the indices where rt is not none
        rt_indices = np.where(~np.isnan(rt))[0]
        rt = rt[rt_indices]

        circ = values["circ"]
        targets_with_responses = circ.data[rt_indices]
        intensity = values["intensity"][rt_indices]

        # detect outliers
        z_scores = np.abs(stats.zscore(rt))
        outliers = np.where(z_scores > OUTLIER_THRESHOLD)[0]

        # remove outliers
        rt = np.delete(rt, outliers)
        targets_with_responses = np.delete(targets_with_responses, outliers)
        intensity = np.delete(intensity, outliers)

        cos_phase = np.cos(targets_with_responses)
        sin_phase = np.sin(targets_with_responses)

        new_data = pd.DataFrame({
            "participant": participant,
            "rt": rt,
            "log_rt": np.log(rt),
            "cos_phase": cos_phase,
            "sin_phase": sin_phase,
            "intensity": intensity
        })

        LMEM_data = pd.concat([LMEM_data, new_data], ignore_index=True)
    

    print(f"\nNumber of NaNs in LMEM_data:\n {LMEM_data.isnull().sum()}\n")

    # drop rows with nans
    LMEM_data = LMEM_data.dropna()

    LMEM_analysis(
        LMEM=LMEM, 
        data=LMEM_data,
        dependent_variable=RT_COL,
        n_null=N_NULL_LMEM, 
        figpath=figpath / "h3b_LMEM_phase_modulates_RT.png", 
        txtpath=figpath / "h3b_LMEM_phase_modulates_RT.txt"
    )