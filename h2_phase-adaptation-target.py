"""
This script tests H2:
* Across different ISIs, the breathing patterns of the participants will align to target stimuli such that it occurs preferentially at specific phases of the respiratory cycle.



CONSIDER 
* Which method to use for null-sampling?
* U2? 
* Which method for comparing the two groups (mann-whitney, t-test, something different?) 
"""
import numpy as np
from utils import load_data, create_trigger_mapping
from pathlib import Path
import matplotlib.pyplot as plt
from pyriodic.utils import calculate_p_value
from pyriodic import Circular, CircPlot
from pyriodic.density import vonmises_kde
from tqdm import tqdm
from scipy import stats
from pyriodic.permutation import get_breathing_cycles, precompute_cycle_array, make_scrambled_PA

from typing import Union, Optional, Literal, Callable

# Parameters
N_NULL = 5000
N_PERMUTATIONS = 10000
N_BINS = 100 # for density estimation
KAPPA = 20  # for density estimation



def test_against_null(
    stat_fun: Callable,
    observed: np.ndarray,
    phase_pool: Union[np.ndarray, Literal["uniform"]] = "uniform",
    time_shift: bool = False,
    scramble_breath_cycles: bool = False,
    events = None,
    n_null: int = 1000,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
    return_null_samples: bool = False):
    

    if rng is None:
        rng = np.random.default_rng()

    n_events = len(observed)

    # Generate null samples
    if isinstance(phase_pool, str) and phase_pool == "uniform":
        phase_pool = np.linspace(0, 2 * np.pi, len(observed), endpoint=False)
    elif not isinstance(phase_pool, np.ndarray):
        raise ValueError("phase_pool must be a numpy array or 'uniform'.")

    
    if time_shift or scramble_breath_cycles:
        if events is None:
            raise ValueError("events must be provided when time_shift is True.")
        if len(events) != len(observed):
            raise ValueError("events and observed must have the same length.")
    
    if time_shift:
        print("Generating null samples with time shifts...")
        null_samples = []

        # get an integer shift for each null sample )(between 0 and the number of samples in PA)
        shifts = rng.integers(0, len(phase_pool), size=n_null)
        for shift in shifts:
            # shift PA by the random amount
            shifted = np.roll(phase_pool, shift)
            # sample the same number of phases as in the observed data
            null_sample = shifted[events]
            null_samples.append(null_sample)
    elif scramble_breath_cycles:
        print("Generating null samples by scrambling breathing cycles...")
        breath_cycles = get_breathing_cycles(phase_pool)
        cycle_array, cycle_boundaries = precompute_cycle_array(breath_cycles)

        null_samples = []
        for _ in range(n_null):
            scrambled_PA = make_scrambled_PA(cycle_array, cycle_boundaries, rng, len(phase_pool))
            null_samples.append(scrambled_PA[events])
    else:
        print("Generating null samples by random sampling from phase pool...")

        null_samples = [
            rng.choice(phase_pool, size=n_events, replace=False) for _ in range(n_null)
        ]



    obs_stat = stat_fun(observed)

    # check if obs_stat is a tuple (some stat functions might return multiple values)
    if isinstance(obs_stat, tuple):
        obs_stat, max_angle = obs_stat
        # Compute obs-vs-null test statistics
        null_stats = np.apply_along_axis(
            lambda row: stat_fun(row, return_density_at_phase=max_angle, return_angle=False), 1, null_samples
        )

    else:
        # Compute obs-vs-null test statistics
        null_stats = np.apply_along_axis(stat_fun, 1, null_samples)

    p_val = calculate_p_value(obs_stat, null_stats, alternative)

    if verbose:
        print(f"p val: {p_val}, observed stat: {obs_stat:.3f}, mean null stat: {np.mean(null_stats):.3f}")

    results = (p_val, obs_stat, null_stats)
    if return_null_samples:
        return results + (null_samples,)
    else:
        return results


def plot_participant_lvl(results, figpath=None, stat_fun_name="Maximum density"):
    """Plot participant-level circular mean + permutation null distribution."""
    for participant, values in results.items():
        fig = plt.figure(figsize=(15, 4))
        ax_circ = fig.add_subplot(1, 2, 1, projection="polar")
        ax_hist = fig.add_subplot(1, 2, 2)

        circ_target = values["circ_target"]
        pval = values["pval"]
        null_stats = values["null_stats"]
        obs_stat = values["obs_stat"]
        null_samples = values["null_samples"]

        # ---- Left: Circular Plot ----
        plot_tmp = CircPlot(circ_target, ax=ax_circ, title=f"{participant} (p={pval:.3f})", group_by_labels=False)
        plot_tmp.add_density(color="forestgreen", kappa=KAPPA, n_bins=N_BINS, label="Observed", linewidth=2)


        # loop over the null samples and plot their maximum density and angle as a point on the circular plot
        for null_sample in null_samples:
            max_dens, max_angle = max_density(null_sample, return_angle=True)
            ax_circ.scatter(
                max_angle, max_dens,
                color="lightgray", alpha=0.1, s=10
            )

        # highlight the point with the maximum density
        max_dens, max_angle = max_density(circ_target.data, return_angle=True)
        ax_circ.scatter(
            max_angle, max_dens,
            color="darkred", s=20, label="Observed max density"
        )
  


        # ---- middle: Histogram of permutation statistics ----
        ax_hist.hist(null_stats, bins=50, facecolor='lightgray', edgecolor='black', alpha=0.6, label=f"Null distribution ({stat_fun_name})")

        ax_hist.axvline(obs_stat, color='forestgreen', linewidth=3, label=f"Observed statistic {stat_fun_name}")

        ax_hist.set_title(f"p={pval:.3f}")
        ax_hist.set_xlabel(f"{stat_fun_name} value")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend()

        # ---- Right: Permutation test result ----
        #ax_perm.hist(perm_stats, bins=50, facecolor='lightgray', edgecolor='black', alpha=0.6)
        #ax_perm.axvline(obs_stat, color='forestgreen', linewidth=3, label=f"Observed statistic {perm_stat_name}")
        #ax_perm.set_title(f"Permutation test p={pval:.3f}")
        #ax_perm.set_xlabel("Statistic Value")
        #ax_perm.set_ylabel("Frequency")
        #ax_perm.legend()


        plt.tight_layout()
        if figpath is not None:
            plt.savefig(figpath / f"{participant}_perm_result.png")
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


def group_inference_z(obs_stats, null_stats, one_sided=True):
    """
    Perform group-level inference by converting observed stats into per-subject z-scores
    relative to their own null distributions, then testing across participants.

    Parameters
    ----------
    obs_stats : array-like, shape (n_subjects,)
        Observed statistic per subject.
    null_stats : array-like, shape (n_subjects, n_nulls)
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
    null_stats = np.asarray(null_stats)
    n_subj, n_nulls = null_stats.shape

    # Per-subject null mean & std
    null_mean = null_stats.mean(axis=1)
    null_std = null_stats.std(axis=1, ddof=1)
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

    variables = ["circ", "phase_ts", "event_samples", "event_ids"]
    dataset = "pilots"
    data = load_data(variables, dataset)

    trigger_mapping = create_trigger_mapping()
    # opposite of what we had before, because we want to align to target events
    targets_id  = [v for k, v in trigger_mapping.items() if "target" in k]
    print("Trigger mapping:", targets_id)

    stat_fun = max_density
    stat_fun_name = "max density"


    results = {}

    for participant, values in tqdm(data.items(), desc="Participant:"):
        circ = values["circ"]
        circ_target = circ["target"]
        events = values["event_samples"]
        events_ids = values["event_ids"]
        #print(events_ids)

        # only keep events that are target events
        idx_target = np.array([i for i, eid in enumerate(events_ids) if eid in targets_id])
        events = np.array(events)[idx_target]

        # check that the lengts of events and circ_target are the same
        if len(events) != len(circ_target):
            raise ValueError(f"Length of events ({len(events)}) and circ_target ({len(circ_target)}) do not match for participant {participant}.")

        phase_ts = values["phase_ts"]
        #phase_ts = phase_ts[~np.isnan(phase_ts)] # remove nans #REMEMBER TO TURN BACK ON IF NOT USING BREATH CYCLE SCRAMBLING

        pval, obs_stat, null_stats, null_samples = test_against_null(
                    observed=circ_target.data,
                    phase_pool=phase_ts,
                    stat_fun=stat_fun, 
                    scramble_breath_cycles=True,
                    events=events,
                    n_null=N_NULL, 
                    verbose=False, 
                    alternative="greater",
                    return_null_samples=True
                )
 

        circ_null = Circular.from_multiple(
            [Circular(samp, labels=[i]*len(samp)) for i, samp in enumerate(null_samples)]
        )
        print(null_stats)
        print(null_stats.shape)

        results[participant] = {
            "circ_target": circ_target,
            "circ_null": circ_null,
            "pval": pval,
            "null_stats": null_stats,
            "obs_stat": obs_stat,
            "null_samples": null_samples
        }

    # sort the results dictionary by the participants so they are ordered
    results = dict(sorted(results.items()))

    plot_participant_lvl(
        results, figpath=figpath, 
        stat_fun_name=stat_fun_name
        )
    print("Participant-level results saved.")

    # group level inference
    null_group = [results[subj_id]["null_stats"] for subj_id in results]
    obs_group = [results[subj_id]["obs_stat"] for subj_id in results]

    results_group = group_inference_z(obs_group, null_group, one_sided=True)
    print("\nGroup-level inference results:")
    for k, v in results_group.items():
        print(f"{k}: {v}")


    #plot_permutation_result(group_null, group_obs_stat, figpath=figpath / "h2_group_level.png", pval=group_pval)
    #print(f"\n Group-level p-value: {group_pval}")