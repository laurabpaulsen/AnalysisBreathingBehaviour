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

# PARAMETERS
# for bins refitting of psychometric function
DELTA_CENTER = np.pi/20
WIDTH = np.pi/5
N_NULL_LMEM = 10_000


psignifit_kwargs = {
    'experiment_type': '2AFC', 
    'stimulus_range': [0, 3], # CHANGE THIS!!

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


def plot_priors(res, figpath=None):

    psp.plot_prior(res)

    if figpath:
        plt.savefig(figpath)
    plt.close()

def plot_grand_average_modulation(threshold_estimates, figpath=None):

    average_zscored_thresholds = []
    center = []
    
    for c in threshold_estimates["center"].unique():
        average_zscored_thresholds.append(threshold_estimates[threshold_estimates["center"] == c]["zscored_threshold"].mean())
        center.append(c)

    fig, ax = plt.subplots(1, 1, figsize = (4, 4), dpi = 300, subplot_kw={'projection': 'polar'})


    circ = Circular(center) 
    plot = CircPlot(circ=circ, group_by_labels=False, ax=ax)
    plot.add_hline(np.mean(average_zscored_thresholds), c = "lightgray", linestyle='--', linewidth=1)

    plot.add_connected_points(
        y = average_zscored_thresholds,
        color = "forestgreen",
        linewidth = 1,
        marker = "o",
        markersize = 2
    )

    if figpath:
        plt.savefig(figpath)
    plt.close()


if __name__ == "__main__":


    variables = ["intensity", "circ"]
    dataset = "before_pilots"
    data = load_data(variables, dataset)

    figpath = Path(__file__).parent / "results" / "h3"
    figpath.mkdir(parents=True, exist_ok=True)

    # empty dataframe to store threshold estimates
    threshold_estimates = pd.DataFrame(columns=["participant", "center", "sin_phase", "cos_phase", "threshold", "zscored_threshold"])

    for participant, values in tqdm(data.items(), desc="Fitting psychometric functions"):

        circ, intensities = values["circ"], values["intensity"]

        idx_hit = [idx for idx, label in enumerate(circ.labels) if "/correct" in label]
        idx_miss = [idx for idx, label in enumerate(circ.labels) if "/incorrect" in label]

        intensity_hit, intensity_miss = intensities[idx_hit], intensities[idx_miss]

        PA_hit, PA_miss = circ["/correct"].data, circ["/incorrect"].data

        assert len(intensity_hit) == len(PA_hit)
        assert len(intensity_miss) == len(PA_miss)

        hit_or_miss = np.concatenate([
                np.ones(len(PA_hit), dtype=int),
                np.zeros(len(PA_miss), dtype=int)
        ])
        intensities = np.concatenate([intensity_hit, intensity_miss])
        PA = np.concatenate([PA_hit, PA_miss])


        result_all_data = ps.psignifit(
            format_data_for_psignifit(intensities, hit_or_miss), debug=True, **psignifit_kwargs
        )

        # REFITTING ACROSS THE BINS
        # using the identified threshold and width (i.e., slope) from the overall fit as priors for an iterative refitting of the PsychF on subsets of trials obtained from the moving window
        center_of_bins = np.arange(0, 2 * np.pi, DELTA_CENTER)


        refitted_results = []
        for i, c in enumerate(center_of_bins):
            mask = phase_bin_mask(PA, c, WIDTH)
            tmp_int = intensities[mask]
            tmp_hit_or_miss = hit_or_miss[mask]
            print(tmp_hit_or_miss)
            
            
            # sanity check of mask only for one participant
            from pyriodic import Circular, CircPlot

            if participant == "04":
                fig, ax = plt.subplots(figsize=(6, 6))
                circ_masked = Circular(PA[mask])
                plot_sanity = CircPlot(circ_masked)
                plot_sanity.add_points()

                plt.savefig(figpath / f"{participant}_sanity_check_masked_phase_{c:.2f}.png")
                plt.close()

                plot_priors(result_all_data, figpath / f"{participant}_priors_all_data.png")

            # All parameters except the threshold were then fixed and used as priors for fitting the psychometric function iteratively to an angle-specific subset of trials (gray functions).
            tmp_result = ps.psignifit(
                format_data_for_psignifit(tmp_int, tmp_hit_or_miss),
                
                # fixing parameters from fit on full data set
                fixed_parameters = {
                    'lambda': result_all_data.parameter_estimate['lambda'],
                    'width': result_all_data.parameter_estimate['width']
                },
                debug=True,
                **psignifit_kwargs
            )
            #if i == 0:
            #    plot_priors(tmp_result, figpath / f"{participant}_priors_phase_{c:.2f}.png")

            refitted_results.append(tmp_result)


        thresholds_refitted = [res.parameter_estimate['threshold'] for res in refitted_results]
        zscored_thresholds = (thresholds_refitted - np.mean(thresholds_refitted)) / np.std(thresholds_refitted)
        
        new_data = pd.DataFrame({
            "participant": participant,
            "center": center_of_bins,
            "sin_phase": np.sin(center_of_bins),
            "cos_phase": np.cos(center_of_bins),
            "threshold": thresholds_refitted,
            "zscored_threshold": zscored_thresholds
        })


        threshold_estimates = pd.concat([
            threshold_estimates if not threshold_estimates.empty else None, 
            new_data
            ], ignore_index=True)

        plot_subject_level(refitted_results, result_all_data, figpath / f"{participant}_psychometric_function.png")

        # Save threshold estimates to CSV
    threshold_estimates.to_csv(figpath / "threshold_estimates.csv", index=False)

    plot_grand_average_modulation(threshold_estimates, figpath / "grand_average_modulation.png")

    LMEM_analysis(
        LMEM=LMEM,
        data = threshold_estimates,
        dependent_variable="threshold",
        n_null=N_NULL_LMEM,
        figpath=figpath / "h3_LMEM_phase_modulates_sensitivity.png",
        txtpath=figpath / "h3_LMEM_results.txt",
        n_jobs=-1
    )
