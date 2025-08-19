"""
This script tests H3A:

Respiration has functional relevance reflected in systematic variations in sensory sensitivity across different phases of the respiratory cycle.


Things to be decided
* Should the intensity be normalised? Between 0 and the salient intensity?
* In the ps refitting, should the parameters from the fit on the whole data be passed as priors or should the be locked (except for threshold ofc)

"""

import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, LMEM_analysis
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
import psignifit as ps
import psignifit.psigniplot as psp
from tqdm import tqdm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# PARAMETERS
# for bins refitting of psychometric function
DELTA_CENTER = np.pi/5 # SHOULD BE 30
WIDTH = np.pi/5
N_NULL_LMEM = 500#10_000

SUPPRESS_WARNINGS = True

if SUPPRESS_WARNINGS:
    import warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="statsmodels")

rng = np.random.default_rng(47)

psignifit_kwargs = {
    'experiment_type': '2AFC', 
    'stimulus_range': [0, 3],

}

def LMEM(data):

    model = smf.mixedlm(
        "threshold ~ sin_phase + cos_phase",  # Fixed effects
        data=data,
        groups=data["participant"],  # Random effects
        re_formula="~ sin_phase + cos_phase"  # Random slopes  
    )

    # model specification in R would be
    # threshold ~ sin_phase + cos_phase + (sin_phase + cos_phase | participant)

    return model.fit()

def get_participant_salient_intensity(participant):
    path = Path(__file__).parent / "simulation" / "data" / "raw" / f"participant_{participant}_events.csv"
    participant_data = pd.read_csv(path)

    # intensity in first row
    return participant_data["intensity"].iloc[0]


def phase_bin_mask(phase_angles, center, width):
    # signed shortest distance in radians, in range [-π, π]
    diff = (phase_angles - center + np.pi) % (2*np.pi) - np.pi
    return np.abs(diff) <= width / 2


def format_data_for_psignifit(intensities, hit_or_miss):
    """
    Formats the data for psignifit by creating a 2D array where each row corresponds to a unique intensity
    and contains the number of hits and total trials for that intensity.

    Args:
        intensities (np.ndarray): Array of stimulus intensities.
        hit_or_miss (np.ndarray): Array of binary responses (1 for hit, 0 for miss).

    Returns:
        np.ndarray: Formatted data array for psignifit.
    """
    unique_intensities = np.unique(intensities)
    formatted_data = np.zeros((len(unique_intensities), 3))

    for i, intensity in enumerate(unique_intensities):
        hits = np.sum((intensities == intensity) & (hit_or_miss == 1))
        misses = np.sum((intensities == intensity) & (hit_or_miss == 0))
        formatted_data[i] = [intensity, hits, hits + misses]

    return formatted_data

def plot_subject_level(refitted_results, full_fitted_result, figpath = None):   

    fig, ax = plt.subplots(figsize=(10, 6))

    for res in refitted_results:
        psp.plot_psychometric_function(res, ax=ax, line_color='gray', line_width=0.3, plot_parameter=False, plot_data=False)

    psp.plot_psychometric_function(full_fitted_result, ax=ax, line_color="k", line_width=3, plot_parameter=False, plot_data=True, x_label="Stimulus intensity", y_label="Proportion of hits")

    ax.set_ylim((0, 1))

    if figpath:
        plt.savefig(figpath)
    plt.close()


    


if __name__ == "__main__":
    figpath = Path(__file__).parent / "results" / "h3"
    figpath.mkdir(parents=True, exist_ok=True)

    data = load_data(["intensity", "circ"])

    # empty dataframe to store threshold estimates
    threshold_estimates = pd.DataFrame(columns=["participant", "sin_phase", "cos_phase", "threshold"])

    for participant, values in tqdm(data.items(), desc="Fitting psychometric functions"):

        circ = values["circ"]
        intensities = values["intensity"]

        idx_hit = [idx for idx, label in enumerate(circ.labels) if "hit" in label]
        idx_miss = [idx for idx, label in enumerate(circ.labels) if "miss" in label]

        intensity_hit = intensities[idx_hit]
        intensity_miss = intensities[idx_miss]

        circ_hit = circ["hit"]
        circ_miss = circ["miss"]

        PA_hit = circ_hit.data
        PA_miss = circ_miss.data

        assert len(intensity_hit) == len(PA_hit)
        assert len(intensity_miss) == len(PA_miss)

        hit_or_miss = np.concatenate([
                np.ones(len(PA_hit), dtype=int),
                np.zeros(len(PA_miss), dtype=int)
        ])
        intensities = np.concatenate([intensity_hit, intensity_miss])
        PA = np.concatenate([PA_hit, PA_miss])


        result_all_data = ps.psignifit(
            format_data_for_psignifit(intensities, hit_or_miss), **psignifit_kwargs
        )

        # REFITTING ACROSS THE BINS
        # using the identified threshold and width (i.e., slope) from the overall fit as priors for an iterative refitting of the PsychF on subsets of trials obtained from the moving window
        center_of_bins = np.arange(0, 2 * np.pi, DELTA_CENTER)


        refitted_results = []
        for c in center_of_bins:
            mask = phase_bin_mask(PA, c, WIDTH)
            tmp_int = intensities[mask]
            tmp_hit_or_miss = hit_or_miss[mask]

            # All parameters except the threshold were then fixed and used as priors for fitting the psychometric function iteratively to an angle-specific subset of trials (gray functions).
            tmp_result = ps.psignifit(
                format_data_for_psignifit(tmp_int, tmp_hit_or_miss),
                
                # fixing parameters from fit on full data set
                fixed_parameters = {
                    'lambda': result_all_data.parameter_estimate['lambda'],
                    'width': result_all_data.parameter_estimate['width']
                },
                **psignifit_kwargs
            )

            refitted_results.append(tmp_result)

        new_data = pd.DataFrame({
            "participant": participant,
            "sin_phase": np.sin(center_of_bins),
            "cos_phase": np.cos(center_of_bins),
            "threshold": [res.parameter_estimate['threshold'] for res in refitted_results]
        })

        threshold_estimates = pd.concat([threshold_estimates, new_data], ignore_index=True)

        plot_subject_level(refitted_results, result_all_data, figpath / f"{participant}_psychometric_function.png")

    LMEM_analysis(
        LMEM=LMEM,
        data = threshold_estimates,
        dependent_variable="threshold",
        n_null=N_NULL_LMEM,
        figpath=figpath / "h3_LMEM_phase_modulates_sensitivity.png",
        txtpath=figpath / "h3_LMEM_results.txt"
    )
    # Save threshold estimates to CSV
    threshold_estimates.to_csv(figpath / "threshold_estimates.csv", index=False)
