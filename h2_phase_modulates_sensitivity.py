"""
This script tests h2:

Respiration has functional relevance reflected in systematic variations in sensory sensitivity across different phases of the respiratory cycle.
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
from pyriodic import Circular, CircPlot
# PARAMETERS
# for bins refitting of psychometric function
DELTA_CENTER = np.pi/20
WIDTH = np.pi/4
N_NULL_LMEM = 10_000

MIN_TRIALS_PER_BIN = 30


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


def plot_subject_level(refitted_results, full_fitted_result, center_of_bins, participant=None, figpath=None):   
    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.cm.twilight
    norm = plt.Normalize(vmin=0, vmax=2*np.pi)

    # Plot psychometric functions color-coded by phase
    for res, c in zip(refitted_results, center_of_bins):
        try:
            if np.isnan(res.parameter_estimate["threshold"]):
                print(f"⚠️ Skipping bin at phase {c:.2f} (invalid fit)")
                continue

            color = cmap(norm(c))
            psp.plot_psychometric_function(
                res,
                ax=ax,
                line_color=color,
                line_width=0.8,
                plot_parameter=False,
                plot_data=False
            )
        except Exception as e:
            print(f"⚠️ Could not plot bin at phase {c:.2f}: {e}")
            continue

    # Plot the global fit on top
    psp.plot_psychometric_function(
        full_fitted_result,
        ax=ax,
        line_color="k",
        line_width=3,
        plot_parameter=False,
        plot_data=True,
        x_label="Intensity (mA)",
        y_label="Proportion correct"
    )

    ax.set_ylim((0, 1))

    # ---- ADD POLAR COLOR LEGEND ----

    inset_size = 0.25  # relative size of inset
    inset_margin = 0.15  # margin from edges
    inset_ax = fig.add_axes([1 - inset_margin - inset_size, inset_margin, inset_size, inset_size], projection='polar')
    theta = np.linspace(0, 2*np.pi, 200)
    radii = np.ones_like(theta)


    circ = Circular(center_of_bins) 
    plot = CircPlot(circ=circ, group_by_labels=False, ax=inset_ax)
    

    thresholds = [res.parameter_estimate['threshold'] for res in refitted_results]
    thresholds = np.array(thresholds)


     # Add a color bar legend

    plot.add_points(
        y = thresholds,
        color = [cmap(norm(c)) for c in center_of_bins],
        s=10
    )

    # add hline at the threshold of the full data fit
    plot.add_hline(full_fitted_result.parameter_estimate['threshold'], c = "k", linestyle='--', linewidth=1,)

    inset_ax.set_ylim(np.min(thresholds) * 0.7, np.max(thresholds) * 1.3)
    
    # change theta direction to match remaining plots
    #inset_ax.set_theta_direction(-1)

    # Make a colormap around the circle
    #inset_ax.scatter(theta, radii, c=theta, cmap=cmap, norm=norm, s=15)
    #inset_ax.set_yticklabels([])  # remove radius labels
    #inset_ax.set_xticks([0, np.pi])
    #inset_ax.set_xticklabels(["0", "π"], fontsize=6)

    # also plot the threshold estimates as points
    

    #inset_ax.scatter(center_of_bins, thresholds, c=center_of_bins, cmap=cmap, norm=norm, s=10, edgecolor='k', linewidth=0.5)

    if figpath:
        plt.savefig(figpath / f"{participant}_psychometric_function.png", dpi=300)
        plt.savefig(figpath / f"{participant}_psychometric_function.svg")
    plt.close()



def plot_priors(res, figpath=None):

    psp.plot_prior(res)

    if figpath:
        plt.savefig(figpath)
    plt.close()


if __name__ == "__main__":


    variables = ["circ"]
    dataset = "raw"
    data = load_data(variables, dataset)

    figpath = Path(__file__).parent / "results" / "h2"
    figpath_participant = figpath / "participant_level"
    figpath_participant.mkdir(parents=True, exist_ok=True)

    # empty dataframe to store threshold estimates
    threshold_estimates = pd.DataFrame(columns=["participant", "center", "sin_phase", "cos_phase", "threshold", "zscored_threshold"])
    

    for participant, values in tqdm(data.items(), desc="Fitting psychometric functions"):
        
        print(f"Processing participant {participant}...")

        # extract circular data
        circ = values["circ"]
        circ_target = circ["target"]

        # split up into correct and incorrect trials 
        circ_hits = circ_target["/correct"]
        circ_misses = circ_target["/incorrect"]

        salient_intensity = circ.metadata["intensity"].max()

        psignifit_kwargs = {
            'experiment_type': '2AFC', 
            'stimulus_range': [1, salient_intensity],
        }
        print(f"Using stimulus range: {psignifit_kwargs['stimulus_range']}")

        intensity_hit, intensity_miss = circ_hits.metadata["intensity"], circ_misses.metadata["intensity"]


        PA_hit, PA_miss = circ_hits.data, circ_misses.data

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
        center_of_bins_loop = np.arange(0, 2 * np.pi, DELTA_CENTER)


        refitted_results = []
        center_of_bins = []
        for i, c in enumerate(center_of_bins_loop):
            mask = phase_bin_mask(PA, c, WIDTH)
            tmp_int = intensities[mask]
            tmp_hit_or_miss = hit_or_miss[mask]

            if len(tmp_hit_or_miss) < MIN_TRIALS_PER_BIN:
                print(f"⚠️ Skipping bin at phase {c:.2f} (only {len(tmp_hit_or_miss)} trials)")
                continue
            else:
                center_of_bins.append(c)

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
        
        plot_subject_level(refitted_results, result_all_data, center_of_bins, participant, figpath_participant)

    # Save threshold estimates to CSV
    threshold_estimates.to_csv(figpath / "threshold_estimates.csv", index=False)

    LMEM_analysis(
        LMEM=LMEM,
        data = threshold_estimates,
        dependent_variable="threshold",
        n_null=N_NULL_LMEM,
        figpath=figpath / "h2_LMEM_phase_modulates_sensitivity.svg",
        txtpath=figpath / "h2_LMEM_results.txt",
        n_jobs=-1
    )
