import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Callable

def load_data(variables:list[str], dataset:str = "simulated", suffix = "preproc.pkl"):
    if dataset == "simulated":
        # error: FileNotFoundError: No files found with suffix 'preproc.pkl' in /Users/au661930/Library/CloudStorage/OneDrive-Aarhusuniversitet/Dokumenter/projects/_BehaviouralBreathing/code/AnalysisBreathingBehaviour/simulation/data/intermediate
        # path to one of the files: /Users/au661930/Library/CloudStorage/OneDrive-Aarhusuniversitet/Dokumenter/projects/_BehaviouralBreathing/code/AnalysisBreathingBehaviour/simulation/data/intermediate/participant_00_preproc.pkl
        datapath = Path(__file__).parent / "simulation" / "data" / "intermediate"

    elif dataset == "real":
        raise NotImplementedError("Real dataset loading is not implemented yet.")

    # find all files in the path
    files = list(datapath.glob(f"**/*{suffix}"))
    if not files:
        raise FileNotFoundError(f"No files found with suffix '{suffix}' in {datapath}")

    data = {}

    for file in files:
        with open(file, "rb") as f:

            # get the participant label
            participant = file.stem.split("_")[1]
            file_data = pkl.load(f)
            data_tmp = {var: file_data[var] for var in variables if var in file_data}
            data[participant] = data_tmp
    
    return data




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
