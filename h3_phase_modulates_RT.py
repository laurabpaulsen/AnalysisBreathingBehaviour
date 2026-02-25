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
RT_COL = "log_rt"




def LMEM(data):
    # SIMPLE IF IT DOES NOT CONVERGE
    # log_rt ~ sin_phase + cos_phase + (1 + intensity | participant)
    if SIMPLE_MODEL:
        model = smf.mixedlm(
            f"{RT_COL} ~ sin_phase + cos_phase",  # Fixed effects
            data=data,
            groups=data["participant"],  # Random effects
            re_formula="~intensity"  # Controlling for intensity by including it as random slope
        )
    else:
        # model specification in R would be
        # log_rt ~ sin_phase + cos_phase + (1 + sin_phase + cos_phase + intensity | participant)

        model = smf.mixedlm(
            f"{RT_COL} ~ sin_phase + cos_phase",  # Fixed effects
            data=data,
            groups=data["participant"],  # Random effects
            re_formula="~ sin_phase + cos_phase + intensity"  # Controlling for intensity by including it as random slopes
        )

    return model.fit()


def plot_subject_RT_by_phase(data, figpath, filter_outliers=True, num_bins=20, stat="mean"):
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
    #n_rows = 3
    #n_cols = len(data) // n_rows + (len(data) % n_rows > 0)

    #figsize = (n_cols * 4, n_rows * 4)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "polar"})

    colours = plt.get_cmap('tab20', len(data))
    colours = [colours(i) for i in range(len(data))]

    for (subj_id, dat), colour in zip(data.items(), colours):

        #preproc = dat["preproc"]
        circ = dat["circ"]["target"]
        rt = np.array(circ.metadata["rt"]) 

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
        plot = CircPlot(circ_for_plot, group_by_labels=False, ax=ax)
        plot.add_connected_points(y=avg_response_times, color=colour, alpha=0.4, marker ='.', linewidth=2)
        #plot.add_hline(y=np.nanmean(avg_response_times), color='gray', linestyle='--', label=f'{stat.capitalize()} RT')
        #axes[i].set_ylim(0.4, np.nanmax(avg_response_times) + 0.1 * np.nanmax(avg_response_times))


    # add ticks and labels
    #ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    #ax.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4'])
    ax.set_ylabel("Response Time (s)")
    
    # add y ticks
    # y lim 
    ylim = ax.get_ylim()
    ax.set_ylim(0, ylim[1])
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 3))
    ax.set_yticklabels([np.round(tick, 2) for tick in np.linspace(ylim[0], ylim[1], 3)])
    ax.set_rlabel_position(30)  # Move radial labels away from plotted line

    # add grid  
    ax.grid(True)


    plt.tight_layout()

    if figpath:
        plt.savefig(figpath, dpi=300)


if __name__ == "__main__":

    dataset = "raw"
    variables = ["circ"]
    data = load_data(variables, dataset)

    figpath = Path(__file__).parent / "results" / "h3"
    figpath.mkdir(exist_ok=True, parents=True)


    LMEM_data = pd.DataFrame(columns=["participant", "rt", "log_rt", "cos_phase", "sin_phase", "intensity"])


    
    plot_subject_RT_by_phase(
        data=data,
        figpath=figpath / "h3_RT_by_phase_polar.svg",
        filter_outliers=True,
        num_bins=15,
        stat="mean"
    )

    #plot_grand_average_modulation()
        
    for participant, values in data.items():
        circ = values["circ"]["target"]

        # with responses
        circ_correct, circ_incorrect = circ["/correct"], circ["/incorrect"]
        circ_with_responses = circ_correct + circ_incorrect

        # extract relevant data for the LMEM
        intensity = circ_with_responses.metadata["intensity"]
        rt = circ_with_responses.metadata["rt"]
        targets_with_responses = circ_with_responses.data


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

    LMEM_analysis(
        LMEM=LMEM, 
        data=LMEM_data,
        dependent_variable=RT_COL,
        n_null=N_NULL_LMEM, 
        figpath=figpath / "h3_LMEM_phase_modulates_RT.svg", 
        txtpath=figpath / "h3_LMEM_phase_modulates_RT.txt",
        n_jobs=-1
    )
