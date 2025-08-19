"""
This script tests H2:
* Across different ISIs, the breathing patterns of the participants will align to target stimuli such that it occurs preferentially at specific phases of the respiratory cycle.



CONSIDER 
* Which method to use for null-sampling?
* U2? 
* Which method for comparing the two groups (mann-whitney, t-test, something different?) 
"""
import numpy as np
from utils import load_data
from pathlib import Path
import matplotlib.pyplot as plt
from pyriodic.permutation import permutation_test_against_null, permutation_test_within_units
from pyriodic import Circular, CircPlot

# Parameters
N_NULL = 2000 
N_PERMUTATIONS = 10000
N_BINS = 20


def plot_participant_lvl(results, figpath=None):
    n_participants = len(results)

    n_rows = int(np.ceil(n_participants / 5))
    n_cols = int(np.ceil(n_participants / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), subplot_kw={'projection': 'polar'})
    axes = axes.flatten()

    for i, (participant, values) in enumerate(results.items()):
        circ_null = values["circ_null"]
        circ_target = values["circ_target"]
        plot_tmp = CircPlot(circ_null, ax=axes[i], title=f"{participant}")
        plot_tmp.add_circular_mean(color = "lightblue")
        plot_tmp.add_arrows(
            np.array([circ_target.mean()]),
            np.array([circ_target.r()]),
            color = "darkblue"
        )


    plt.tight_layout()

    if figpath is not None:
        plt.savefig(figpath)
    plt.close()


def plot_permutation_result(null_distribution, observed_stat, pval, figpath=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(null_distribution, bins=100, facecolor='lightgray', edgecolor='black', alpha=0.5)
    ax.axvline(observed_stat, color='forestgreen', linewidth=3)

    # add label
    ax.text(observed_stat + ax.get_xlim()[1]/30, ax.get_ylim()[1]*0.9, 'Observed Statistic', color='forestgreen', ha='center')

    ax.set_xlabel('Statistic Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'p={pval:.3f}')

    plt.tight_layout()

    if figpath is not None:
        plt.savefig(figpath)
    plt.close()


if __name__ == "__main__":
    figpath = Path(__file__).parent / "results" / "h2"
    figpath.mkdir(parents=True, exist_ok=True)

    data = load_data(["circ", "phase_ts"])

    results = {}

    for participant, values in data.items():
        circ = values["circ"]
        circ_target = circ["target"]

        phase_ts = values["phase_ts"]
        phase_ts = phase_ts[~np.isnan(phase_ts)] # remove nans

        obs_stat, pval, null_samples, obs_vs_null, null_vs_null = permutation_test_against_null(
            circ_target.data,
            phase_ts,
            n_null=N_NULL,
            n_permutations=N_PERMUTATIONS,
            n_bins=N_BINS,
            return_null_samples=True,
            return_obs_and_null_stats=True
        )

        circ_null = Circular.from_multiple(
            [Circular(samp, labels=[i]*len(samp)) for i, samp in enumerate(null_samples)]
        )

        results[participant] = {
            "circ_target": circ_target,
            "circ_null": circ_null,
            "pval": pval,
            "obs_vs_null": obs_vs_null,
            "null_vs_null": null_vs_null
        }

    # sort the results dictionary by the participants so they are ordered
    results = dict(sorted(results.items()))

    plot_participant_lvl(results, figpath=figpath / "h2_participant_lvl.png")


    # group level inference
    null = [results[subj_id]["null_vs_null"] for subj_id in results]
    obs = [results[subj_id]["obs_vs_null"] for subj_id in results]

    group_obs_stat, group_pval, group_null = permutation_test_within_units(
        data1=obs,
        data2=null,
        n_permutations=N_PERMUTATIONS,
        alternative="greater",
        verbose=True,
        return_null_distribution=True
    )

    plot_permutation_result(group_null, group_obs_stat, figpath=figpath / "h2_group_level.png", pval=group_pval)
    print(f"\n Group-level p-value: {group_pval}")