"""
This script tests H3:

Respiration has functional relevance reflected in modulation of response time across different phases of the respiratory cycle.

"""
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_data, LMEM_analysis
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
from pyriodic import Circular, CircPlot
from utils import binned_stats



# PARAMETERS
N_NULL_LMEM = 10_000
OUTLIER_THRESHOLD = 3
SIMPLE_MODEL = True
RT_COL = "log_rt" # could also be "rt"

# Suppress warnings from statsmodels
SUPPRESS_WARNINGS = True

if SUPPRESS_WARNINGS:
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
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


def plot_subject_RT_by_phase(data, figpath, filter_outliers=True, num_bins=40, stat="mean"):
    """
    Plot response time by phase for each subject in a polar plot.

    Parameters
    ----------
    data : dict
        Dictionary containing data for each subject.
    sfreq : int
        Sampling frequency.
    figpath : Path
        Path to save the figure.
    filter_outliers : bool, optional
        Whether to filter out outliers, by default True.
    num_bins : int, optional
        Number of bins for phase, by default 12.
    stat : str, optional
        Statistic to compute for each bin, by default "mean". Other options: "median", "std".

    Returns
    -------
    None
        Saves the figure to the specified path.
    """
    n_rows = 3
    n_cols = len(data) // n_rows + (len(data) % n_rows > 0)

    figsize = (n_cols * 4, n_rows * 4)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, subplot_kw={"projection": "polar"}, sharex=True, sharey=False)

    for i, (subj_id, dat) in enumerate(data.items()):

        #preproc = dat["preproc"]
        circ = dat["circ"]
        rt = dat["rt"] 

        # find the indices where rt is not none
        rt_indices = np.where(~np.isnan(rt))[0]
        rt = rt[rt_indices]

        targets_with_responses = circ.data[rt_indices-1] # get the phase angle at the target stimuli

        # check that target is in label
        assert ["target" in label for label in circ.labels[rt_indices-1]], "something went wrong"

        if filter_outliers:
            # detect outliers
            z_scores = np.abs(stats.zscore(rt))
            outliers = np.where(z_scores > OUTLIER_THRESHOLD)[0]

            # remove outliers
            rt = np.delete(rt, outliers)
            targets_with_responses = np.delete(targets_with_responses, outliers)

        # binned stats
        bin_centers, avg_response_times, std_response_times = binned_stats(targets_with_responses, rt, n_bins=num_bins, stat=stat)

        circ_for_plot = Circular(bin_centers)
        plot = CircPlot(circ_for_plot, group_by_labels=False, ax=axes[i], title=subj_id)
        plot.add_connected_points(y=avg_response_times, color='forestgreen', alpha=0.5, marker ='o')
        plot.add_hline(y=np.nanmean(avg_response_times), color='gray', linestyle='--', label=f'{stat.capitalize()} RT')
        axes[i].set_ylim(0.4, np.nanmax(avg_response_times) + 0.1 * np.nanmax(avg_response_times))



    plt.tight_layout()

    if figpath:
        plt.savefig(figpath, dpi=300)






if __name__ == "__main__":

    dataset = "pilots"
    variables = ["rt", "circ", "intensity"]
    data = load_data(variables, dataset)

    figpath = Path(__file__).parent / "results" / "h3"
    figpath.mkdir(exist_ok=True, parents=True)


    LMEM_data = pd.DataFrame(columns=["participant", "rt", "cos_phase", "sin_phase", "intensity"])


    plot_subject_RT_by_phase(
        data=data,
        figpath=figpath / "h3_RT_by_phase_polar.png",
        filter_outliers=True,
        num_bins=20,
        stat="mean"
    )

    #plot_grand_average_modulation()
        
    for participant, values in data.items():
        # remove nans from rt
        rt = np.array(values["rt"])
        # find the indices where rt is not none
        rt_indices = np.where(~np.isnan(rt))[0]
        rt = rt[rt_indices]

        circ = values["circ"]
        targets_with_responses = circ.data[rt_indices-1] # get the phase angle at the target stimuli
        intensity = values["intensity"][rt_indices-1] # get the intensity of the target stimuli

        # check that target is in label
        assert ["target" in label for label in circ.labels[rt_indices-1]], "something went wrong"

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

        LMEM_data = pd.concat([
            LMEM_data if not LMEM_data.empty else None,
            new_data
        ], ignore_index=True)

    print(LMEM_data.head())
    print(f"\nNumber of NaNs in LMEM_data:\n {LMEM_data.isnull().sum()}\n")

    # drop rows with nans
    LMEM_data = LMEM_data.dropna()

    print(LMEM_data.head())

    LMEM_analysis(
        LMEM=LMEM, 
        data=LMEM_data,
        dependent_variable=RT_COL,
        n_null=N_NULL_LMEM, 
        figpath=figpath / "h3_LMEM_phase_modulates_RT.png", 
        txtpath=figpath / "h3_LMEM_phase_modulates_RT.txt",
        n_jobs=-1
    )
