"""
This script tests H1:
* Respiration rate adapts to the stimulus presentation rate (i.e., shorter interstimulus intervals will result in faster breathing, and longer interstimulus intervals will result in slower breathing).
"""

from utils import load_data
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def respfreq(n_peaks, duration):
    return n_peaks / duration

def mixed_effect_model(df):

    # ISI is numeric, respiratory_frequency is continuous
    # Fit linear mixed-effects model with random intercept and slope per participant
    model = smf.mixedlm("respiratory_frequency ~ ISI_centered", data=df, groups=df["participant"],
                        re_formula="~ISI_centered")
    result = model.fit()


    # model specification in R would be
    # respiratory_frequency ~ ISI_centered + (ISI_centered | participant)

    return result


def plot(df, figpath):
    fig, ax = plt.subplots(figsize=(10, 6))

    for participant in df["participant"].unique():
        subset = df[df["participant"] == participant]

        isis = []
        resp_freqs = []
        
        for isi in subset["ISI"].unique():
            tmp_subset = subset[subset["ISI"] == isi]
            isis.append(isi)
            resp_freqs.append(tmp_subset["respiratory_frequency"].mean())

        # find the idx that sorts isis
        sort_idx = np.argsort(isis)
        isis = np.array(isis)[sort_idx]
        resp_freqs = np.array(resp_freqs)[sort_idx]

        ax.plot(isis, resp_freqs, alpha=0.3, linewidth=1)

    # plot the average across participants for each isi
    avg_resp_freqs = df.groupby("ISI")["respiratory_frequency"].mean()
    ax.plot(avg_resp_freqs.index, avg_resp_freqs.values, color="black", linewidth=2, label="Average Across Participants", marker="o")

    # standard deviation across participants
    std = df.groupby("ISI")["respiratory_frequency"].std()
    ax.fill_between(avg_resp_freqs.index, avg_resp_freqs.values - std, avg_resp_freqs.values + std, color="black", alpha=0.1)

    ax.set_xlabel("ISI (s)")
    ax.set_ylabel("Breaths per minute")
    ax.set_title("Respiratory Frequency vs. ISI")
    ax.legend()
    ax.grid()

    # set ax ticks to the actual measured isis
    ax.set_xticks(df["ISI"].unique())
    ax.set_xticklabels(df["ISI"].unique())

    if figpath:
        plt.savefig(figpath)


if __name__ == "__main__":
    sampling_rate = 100
    n_stim_per_sequence = 4
    n_sequence_per_block = 10

    variables = ["peaks", "event_samples", "event_ids"]
    data = load_data(variables)

    figpath = Path(__file__).parent / "results" / "h1"/ "h1_respiratory_frequency_vs_isi.png"
    figpath.parent.mkdir(exist_ok=True, parents=True)

    # create an empty DataFrame
    df = pd.DataFrame(columns=["participant", "respiratory_frequency", "ISI", "block_number"])

    for participant, values in data.items():

        peaks = values["peaks"]
        event_samples = values["event_samples"]
        event_ids = values["event_ids"]

        # first step is to determine when each block begins and ends
        # we do this by looking at the time between the event samples
        diffs = np.diff(event_samples)
        for dif in diffs:
            print(dif)
        block_changes = np.where(np.abs(np.diff(diffs)) > 2)[0]
        block_start_indices = [0] + (block_changes + 1).tolist()
        block_end_indices = (block_changes + 1).tolist() + [len(event_samples) - 1]
        block_indices = list(zip(block_start_indices, block_end_indices))

        for i, (start_idx, end_idx) in enumerate(block_indices):
            start_samp = event_samples[start_idx]
            end_samp = event_samples[end_idx]

            duration = (end_samp - start_samp) / sampling_rate

            # Calculate ISI from event samples inside the block
            block_event_samples = event_samples[start_idx:end_idx + 1]
            block_diffs = np.diff(block_event_samples) / sampling_rate
            isi = np.round(np.mean(block_diffs), 2)
            #print(f"Participant {participant}, Block {i}, ISI: {isi}")

            peak_in_block = peaks[(peaks >= start_samp) & (peaks <= end_samp)]

            # first peak samp
            first_peak_samp = peak_in_block[0]
            last_peak_samp = peak_in_block[-1]

            duration_peak_to_peak = (last_peak_samp - first_peak_samp) / sampling_rate

            resp_freq = respfreq(len(peak_in_block), duration_peak_to_peak/60)
            df = pd.concat([df, pd.DataFrame({"participant": [participant], "respiratory_frequency": [resp_freq], "ISI": [isi], "block_number": [i]})], ignore_index=True)

    print(df["ISI"].unique())

    df["ISI_centered"] = df["ISI"] - df["ISI"].min()

    result = mixed_effect_model(df)

    txtpath = figpath.parent / "h1_LMEM_results.txt"
    with open(txtpath, "w") as f:
        f.write(result.summary().as_text())

        # also write random effects to text file
        f.write("\nRandom Effects:\n")
        for participant, re in result.random_effects.items():
            f.write(f"Participant {participant}:\n")
            f.write(f"{re}\n\n")


    # sort the df by participant
    df = df.sort_values("participant")
    plot(df, figpath)