"""
This script tests h1:
* Across different ISIs, the breathing patterns of the participants will align to target stimuli such that it occurs preferentially at specific phases of the respiratory cycle.

"""
import numpy as np
from utils import load_data
from pathlib import Path
import matplotlib.pyplot as plt

from typing import Union
from tqdm import tqdm
from scipy import stats

import pandas as pd

from pyriodic import Circular, CircPlot
from pyriodic.density import vonmises_kde
from pyriodic.surrogate import surrogate_shuffle_breath_cycles
from pyriodic.stats_surrogate import test_against_surrogate

# Parameters
N_SURR = 5000
N_BINS = 200 # for density estimation
KAPPA = 20  # for density estimation

# set random seed for reproducibility
np.random.seed(42)

def plot_participant_lvl(results, figpath=None, stat_fun_name="Maximum density", ylim=None, colours=None, circ_key="circ_target"):
    """Plot participant-level circular mean + permutation null distribution."""
    for participant, values in results.items():
        fig = plt.figure(figsize=(12, 6))
        ax_circ = fig.add_subplot(1, 2, 1, projection="polar")
        ax_hist = fig.add_subplot(1, 2, 2)

        circ = values[circ_key]
        pval = values["pval"]
        null_stats = values["null_stats"]
        obs_stat = values["obs_stat"]
        null_samples = values["null_samples"]

        # ---- Left: Circular Plot ----
        plot_tmp = CircPlot(circ, ax=ax_circ, title=f"{participant} (p={pval:.3f})", group_by_labels=False)
        plot_tmp.add_density(color=colours[participant-1] if colours is not None else "forestgreen", kappa=KAPPA, n_bins=N_BINS, label="Observed", linewidth=2)


        # loop over the null samples and plot their maximum density and angle as a point on the circular plot
        for i, null_sample in enumerate(null_samples):
            if i < 1000:
                max_dens, max_angle = max_density(null_sample, return_angle=True)
                ax_circ.scatter(
                    max_angle, max_dens,
                    color="lightgray", alpha=0.1, s=10,
                    label=f"Surrogate {stat_fun_name} (1000 out of {len(null_samples)})" if i == 0 else ""
                )

        # highlight the point with the maximum density
        max_dens, max_angle = max_density(circ.data, return_angle=True)
        ax_circ.scatter(
            max_angle, max_dens,
            color="darkred", s=25, label="Observed max density", zorder=2.5
        )
        ax_circ.legend()

        # set ylim
        if ylim is not None:
            ax_circ.set_ylim(ylim)
  
        ax_hist.hist(null_stats, bins=50, facecolor='lightgray', edgecolor='black', alpha=0.6, label=f"Surrogate distribution ({stat_fun_name})")

        ax_hist.axvline(obs_stat, color='forestgreen', linewidth=3, label=f"Observed statistic {stat_fun_name}")

        ax_hist.set_title(f"p={pval:.3f}")
        ax_hist.set_xlabel(f"{stat_fun_name} value")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend()

    


        plt.tight_layout()
        if figpath is not None:
            plt.savefig(figpath / f"{participant}_perm_result.svg")
        plt.close()

def plot_group_zscores(results_group, colours = None, participant_ids=None, savepath=None):
    """
    Plot group-level standardized effects (z-scores).
    This is Panel D.
    """
    z = np.asarray(results_group["z_scores"])
    n = len(z)

    if participant_ids is None:
        participant_ids = [f"{i+1}" for i in range(n)]

    # colours for grouped
    if colours is None:
        colours = plt.get_cmap('tab20', len(results_group["z_scores"]))
        colours = [colours(i) for i in range(len(results_group["z_scores"]))]

    # Mean and 95% CI
    mean_z = z.mean()
    sem_z = z.std(ddof=1) / np.sqrt(n)
    ci_low, ci_high = stats.t.interval(
        0.95, df=n-1, loc=mean_z, scale=sem_z
    )

    # Sort by z-score for readability
    order = np.argsort(z)
    z_sorted = z[order]
    ids_sorted = np.array(participant_ids)[order]
    y = np.arange(n)

    colours_sorted = np.array(colours)[order]

    fig, ax = plt.subplots(figsize=(4, 6), dpi=300)

    # Individual participants
    ax.scatter(
        z_sorted, y,
        color=colours_sorted,
        s=35,
        zorder=3
    )

    # Zero line
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)

    # Group mean and CI
    ax.axvline(mean_z, color="darkgray", linewidth=2, label="Group mean")
    ax.fill_betweenx(
        [-1, n],
        ci_low, ci_high,
        color="darkgray",
        alpha=0.2,
        label="95% CI"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(ids_sorted)
    ax.set_xlabel("Z-score (subject-wise standardized effect)")
    ax.set_ylabel("Participant")

    ax.set_xlim(
        min(-1, z.min() - 0.5),
        max(1, z.max() + 0.5)
    )

    ax.set_title(
        f"Mean z = {mean_z:.2f}, "
        f"p = {results_group['p_w']:.3f}"
    )

    ax.legend(frameon=False)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath / "group_zscores.svg")

    plt.close()

def max_density(phases, return_density_at_phase: Union[bool, float] = False, return_angle: bool = False) -> Union[float, tuple[float, float]]:
    """
    Compute the maximum density of a circular sample using kernel density estimation.

    Parameters
    ----------
    phases : np.ndarray
        Sample of circular data (values should be in [0, 2π]).

    Returns
    -------
    float
        The maximum density value from the kernel density estimate.
    """
    x, densities = vonmises_kde(phases, kappa=KAPPA, n_bins=N_BINS)
    
    if return_angle:
        max_density = np.max(densities)
        max_angle = x[np.argmax(densities)]
        return max_density, max_angle
    
    if return_density_at_phase:
        idx = np.where(np.isclose(x, return_density_at_phase, atol=1e-2))[0]

        dens_at_phase = densities[idx]
        return dens_at_phase
    
    return np.max(densities)


def group_inference_z(obs_stats, surr_stats, one_sided=True, subj_labels = None, zscore_path=None):
    """
    Perform group-level inference by converting observed stats into per-subject z-scores
    relative to their own null distributions, then testing across participants.

    Parameters
    ----------
    obs_stats : array-like, shape (n_subjects,)
        Observed statistic per subject.
    null_stats : array-like, shape (n_subjects, surr_stats)
        Null distribution statistics per subject.
    one_sided : bool
        If True, compute one-sided tests (testing whether observed > null).
        If False, compute two-sided tests.

    Returns
    -------
    results : dict
        Dictionary with per-subject z-scores and group-level test results.
    """

    obs_stats = np.asarray(obs_stats)
    surr_stats = np.asarray(surr_stats)
    n_subj, n_surr = surr_stats.shape

    # Per-subject null mean & std
    null_mean = surr_stats.mean(axis=1)
    null_std = surr_stats.std(axis=1, ddof=1)
    # avoid divide-by-zero
    null_std[null_std == 0] = 1e-10

    # Compute z-scores
    z_scores = (obs_stats - null_mean) / null_std

    # Group tests
    # One-sample t-test against 0
    t_stat, p_two = stats.ttest_1samp(z_scores, popmean=0)
    if one_sided:
        p_t = p_two/2 if t_stat > 0 else 1 - p_two/2
    else:
        p_t = p_two

    # Wilcoxon signed-rank test
    # (requires n_subj > ~10; "greater" tests if median(z) > 0)
    try:
        w_stat, p_w = stats.wilcoxon(z_scores, alternative='greater' if one_sided else 'two-sided',
                                     zero_method='wilcox', mode='approx')
    except Exception as e:
        w_stat, p_w = np.nan, np.nan

    # Effect size (Cohen's d)
    mean_z = z_scores.mean()
    sd_z = z_scores.std(ddof=1)
    cohen_d = mean_z / sd_z if sd_z > 0 else np.nan

    # dataframe with results
    results_df = pd.DataFrame({
        "participant": subj_labels if subj_labels is not None else [f"Subj {i+1}" for i in range(n_subj)],
        "z_score": z_scores
    })

    # save z_scores for each participant
    if zscore_path is not None:
        results_df.to_csv(zscore_path / "group_zscores.csv", index=False)
    

    return {
        "z_scores": z_scores,
        "mean_z": mean_z,
        "cohens_d": cohen_d,
        "t_stat": t_stat,
        "p_t": p_t,
        "wilcoxon_W": w_stat,
        "p_w": p_w,
        "n_subjects": n_subj
    }



def circplot_group_level(results, savepath=None, return_ylim: bool = False, colours=None, circ_key="circ_target"):
    """Plot group-level circular density across participants."""
    fig = plt.figure(figsize=(12, 6), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection="polar")

    all_phases = np.concatenate([values[circ_key].data for values in results.values()])
    labels = np.concatenate([[i+1]*len(values[circ_key]) for i, (subj_id, values) in enumerate(results.items())])
    
    # colours for grouped
    if colours is None:
        colours = plt.get_cmap('tab20', len(results))
        colours = [colours(i) for i in range(len(results))]

    circ_all = Circular(all_phases, labels=labels)
    plot_tmp = CircPlot(circ_all, ax=ax, group_by_labels=False, colours=colours)

    plot_tmp.add_density(color="k", kappa=KAPPA, n_bins=N_BINS, label="Density (all participants)", linewidth=2)
    plot_tmp.add_density(kappa=KAPPA, n_bins=N_BINS, linewidth=1.5, grouped=True, alpha=0.7)
    plot_tmp.add_legend(loc="upper left", bbox_to_anchor=(1.1, 1.1), fontsize=8)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath / "group_level_density.svg")

    if return_ylim:
        ylim = ax.get_ylim()
        plt.close()
        return ylim
    
    plt.close()


if __name__ == "__main__":
    figpath = Path(__file__).parent / "results" / "h1"
    figpath_participant = figpath / "participant_level"
    figpath_participant.mkdir(parents=True, exist_ok=True)

    variables = ["circ", "phase_ts"]
    dataset = "raw"
    data = load_data(variables, dataset)

    stat_fun = max_density
    stat_fun_name = "max density"


    results = {}

    for participant, values in tqdm(data.items()):
        circ = values["circ"]
        circ_target = circ["target"]
        # 
        events = circ_target.metadata["event_samples"]
        phase_ts = values["phase_ts"]

        surr_samples = surrogate_shuffle_breath_cycles(
            phase_pool=phase_ts,
            events=events,
            n_surrogate=N_SURR,
            rng=None,
        )

        pval, obs_stat, null_stats = test_against_surrogate(
            stat_fun=stat_fun,
            observed=circ_target.data,
            surrogate_samples=np.array(surr_samples),
            alternative="greater",
            verbose=False,
        )
 
        circ_null = Circular.from_multiple(
            [Circular(samp, labels=[i]*len(samp)) for i, samp in enumerate(surr_samples)]
        )

        results[participant] = {
            "circ_target": circ_target,
            "circ_null": circ_null,
            "pval": pval,
            "null_stats": null_stats,
            "obs_stat": obs_stat,
            "null_samples": surr_samples,
        }
    
    # group level inference
    surr_group = [results[subj_id]["null_stats"] for subj_id in results]
    obs_group = [results[subj_id]["obs_stat"] for subj_id in results]

    results_group = group_inference_z(obs_group, surr_group, one_sided=True, zscore_path=figpath, subj_labels=list(results.keys()))
    colours = plt.get_cmap('tab20', len(results))
    colours = [colours(i) for i in range(len(results))]
    plot_group_zscores(
        results_group,
        participant_ids=list(results.keys()),
        savepath=figpath,
        colours=colours
    )
    
    print("\nGroup-level inference results:")
    for k, v in results_group.items():
        print(f"{k}: {v}")

    ylim = circplot_group_level(results, savepath=figpath, return_ylim=True)

    plot_participant_lvl(
        results, figpath=figpath_participant, 
        stat_fun_name=stat_fun_name,
        ylim=ylim, colours=colours
        )