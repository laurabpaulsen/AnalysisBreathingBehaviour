import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Callable
from joblib import Parallel, delayed

def load_data(variables:list[str], dataset:str = "simulated", suffix = "preproc.pkl"):
    if dataset == "simulated":
        # error: FileNotFoundError: No files found with suffix 'preproc.pkl' in /Users/au661930/Library/CloudStorage/OneDrive-Aarhusuniversitet/Dokumenter/projects/_BehaviouralBreathing/code/AnalysisBreathingBehaviour/simulation/data/intermediate
        # path to one of the files: /Users/au661930/Library/CloudStorage/OneDrive-Aarhusuniversitet/Dokumenter/projects/_BehaviouralBreathing/code/AnalysisBreathingBehaviour/simulation/data/intermediate/participant_00_preproc.pkl
        datapath = Path(__file__).parent / "simulation" / "data" / "intermediate"
    elif dataset == "before_pilots":
        datapath = Path(__file__).parent / "data" / "before_pilots" / "intermediate"
    elif dataset == "pilots":
        datapath = Path(__file__).parent / "data" / "pilots" / "intermediate"
    elif dataset == "raw":
        datapath = Path(__file__).parent / "data" / "raw" / "intermediate"
    
    else:
        raise ValueError(f"Dataset '{dataset}' not recognised. Choose from 'simulated', 'before_pilots', 'pilots', or 'raw'.")


    # find all files in the path
    files = list(datapath.glob(f"**/*{suffix}"))
    if not files:
        raise FileNotFoundError(f"No files found with suffix '{suffix}' in {datapath}")

    data = {}

    # sort the files to ensure consistent order
    files = sorted(files)

    for i, file in enumerate(files):
        with open(file, "rb") as f:
            # get the participant labelª
            file_data = pkl.load(f)
            data_tmp = {var: file_data[var] for var in variables if var in file_data}
            data[i+1] = data_tmp
    
    return data


def create_trigger_mapping(
        simulated = False,
        stim = 1,
        target = 2,
        middle = 4,
        index = 8,
        response = 16,
        correct = 32,
        incorrect = 64):
    if simulated:
        trigger_mapping = {
            "stim/salient": 1,
            "target/right/hit": 2,
            "target/right/miss": 3,
            "target/left/hit": 4,
            "target/left/miss": 5,
        }

    else:   
        trigger_mapping = {
            "stim/salient": stim,
            "target/middle": target + middle,
            "target/index": target + index,
            "response/index/correct": response + index + correct,
            "response/middle/incorrect": response + middle + incorrect,
            "response/middle/correct": response + middle + correct,
            "response/index/incorrect": response + index + incorrect, 
            "break/start": 128,
            "break/end": 129,
            "experiment/start": 254,
            "experiment/end": 255
            }


    return trigger_mapping


def phase_vector_norm(sine, cos):

    return np.sqrt(sine**2 + cos**2)


def plot_LMEM_result(empirical_norm, null_norms, pval, figpath=None, n_bins=100, empirical_color = "forestgreen"):
    fig, ax = plt.subplots(figsize=(10, 6), dpi = 300)

    # Plot empirical norm and null norms
    ax.axvline(empirical_norm, color=empirical_color, label='Empirical', linewidth=3)
    ax.hist(null_norms, bins=n_bins, label='Null', facecolor='lightgray', edgecolor='black', linewidth=1)


    # add a label to the empirical weight
    ax.text(empirical_norm + ax.get_xlim()[1]/40, 0.9 * ax.get_ylim()[1], "Empirical weight", color=empirical_color)

    ax.set_title(f"pval: {pval:.3f}")
    ax.set_xlabel("LMEM weight")
    ax.set_ylabel("# of iterations")

    if figpath:
        plt.savefig(figpath)
    plt.close()


def LMEM_analysis(
    LMEM:Callable, 
    data, 
    n_null, 
    participant_col="participant", 
    dependent_variable="threshold", 
    figpath=None, 
    txtpath=None, 
    n_jobs=1
):
    # --- empirical fit ---
    result = LMEM(data)

    if txtpath:
        with open(txtpath, "w") as f:
            f.write(result.summary().as_text())
            f.write("\nRandom Effects:\n")
            for participant, re in result.random_effects.items():
                f.write(f"Participant {participant}:\n{re}\n\n")

    empirical_resp_phase_vector_norm = phase_vector_norm(
        result.params["sin_phase"], result.params["cos_phase"]
    )

    # --- define shuffle function ---
    def one_shuffle(seed=None):
        shuffled = data.copy()
        rng = np.random.default_rng(seed)
        for participant in shuffled[participant_col].unique():
            mask = shuffled[participant_col] == participant
            shuffled.loc[mask, dependent_variable] = rng.permutation(
                shuffled.loc[mask, dependent_variable].values
            )
        res = LMEM(shuffled)
        return phase_vector_norm(res.params["sin_phase"], res.params["cos_phase"])

    # --- run in parallel ---
    null_resp_phase_vector_norms = Parallel(n_jobs=n_jobs)(
        delayed(one_shuffle)(seed) for seed in tqdm(range(n_null), desc="LMEM fitting on shuffled dependent variable")
    )

    # --- compute p-value ---
    pval = np.mean(null_resp_phase_vector_norms >= empirical_resp_phase_vector_norm)
    print(f"p-value: {pval}")

    # --- plot ---
    plot_LMEM_result(empirical_resp_phase_vector_norm, null_resp_phase_vector_norms, pval, figpath=figpath)

"""
def LMEM_analysis(LMEM:Callable, data, n_null, participant_col = "participant", dependent_variable = "threshold", figpath=None, txtpath=None):
    result = LMEM(data)

    if txtpath:
        with open(txtpath, "w") as f:
            f.write(result.summary().as_text())

            # also write random effects to text file
            f.write("\nRandom Effects:\n")
            for participant, re in result.random_effects.items():
                f.write(f"Participant {participant}:\n")
                f.write(f"{re}\n\n")


    empirical_resp_phase_vector_norm = phase_vector_norm(result.params["sin_phase"], result.params["cos_phase"])

    # refitting with shuffled thresholds to estimate significance
    null_resp_phase_vector_norms = []
    
    for _ in tqdm(range(n_null), desc="LMEM fitting on shuffled dependent variable"):
        shuffled = data.copy()
        # permute the thresholds (within each subject)
        for participant in shuffled[participant_col].unique():
            mask = shuffled[participant_col] == participant
            shuffled.loc[mask, dependent_variable] = np.random.permutation(shuffled.loc[mask, dependent_variable].values)

        # fit the mixed effects model
        result = LMEM(shuffled)

        null_phase_vector_norm = phase_vector_norm(result.params["sin_phase"], result.params["cos_phase"])
        null_resp_phase_vector_norms.append(null_phase_vector_norm)


    # compute pval (comparing emperical to null)
    pval = np.mean(null_resp_phase_vector_norms >= empirical_resp_phase_vector_norm)
    print(f"p-value: {pval}")

    # plot
    plot_LMEM_result(empirical_resp_phase_vector_norm, null_resp_phase_vector_norms, pval, figpath=figpath)
"""

# defining some helper functions
def binned_stats(phase_angles, var, n_bins=10, stat="mean"):
    """
    Calculate binned statistics for response times.
    
    Parameters:
    rt (array-like): Response times.
    n_bins (int): Number of bins to use.
    stat (str): Statistic to calculate ('mean' or 'median').
    
    Returns:
    bin_centers (array): Centers of the bins.
    avg_response_times (array): Average response times in each bin.
    std_response_times (array): Standard deviation of response times in each bin.
    """
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    avg_response_times = np.zeros(n_bins+1)
    std_response_times = np.zeros(n_bins+1)
    
    for i in range(n_bins):
        bin_mask = (phase_angles >= bin_edges[i]) & (phase_angles < bin_edges[i + 1])
        if stat == "mean":
            avg_response_times[i] = np.mean(var[bin_mask]) if np.any(bin_mask) else np.nan
        elif stat == "median":
            avg_response_times[i] = np.median(var[bin_mask]) if np.any(bin_mask) else np.nan
        else:
            raise ValueError("stat must be 'mean' or 'median'")
        # Ensure the last two dots are connected
        std_response_times[i] = np.std(var[bin_mask]) if np.any(bin_mask) else np.nan
            
    # Ensure the last two dots are connected
    avg_response_times[-1] = avg_response_times[0]
    std_response_times[-1] = std_response_times[0]
    bin_centers = np.concatenate((bin_centers, [bin_centers[0]]))

    return bin_centers, avg_response_times, std_response_times


