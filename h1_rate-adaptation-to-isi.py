"""
This script tests H1:
* Respiration rate adapts to the stimulus presentation rate (i.e., shorter interstimulus intervals will result in faster breathing, and longer interstimulus intervals will result in slower breathing).
"""
from scipy import stats

from utils import load_data
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def respfreq(n_peaks, duration):
    return n_peaks / duration

def LMEM(df):

    # ISI is numeric, respiratory_frequency is continuous
    # Fit linear mixed-effects model with random intercept and slope per participant
    model = smf.mixedlm("respiratory_frequency ~ ISI_centered", data=df, groups=df["participant"],
                        re_formula="~ISI_centered")

    # model specification in R would be
    # respiratory_frequency ~ ISI_centered + (ISI_centered | participant)

    return model.fit()


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



def extract_blocks(data_behav, event_samples, sfreq):
    """Return block-level timing and sample indices."""
    blocks = []
    print(data_behav["block"].unique())

    data_behav["block"] = pd.to_numeric(data_behav["block"], errors="coerce")

    unique_blocks = data_behav["block"].unique()

    # only keep the ones that are integers (remove practice blocks)
    unique_blocks = unique_blocks[~np.isnan(unique_blocks)]
    unique_blocks = unique_blocks[unique_blocks.astype(int) == unique_blocks]

    for block in unique_blocks:
        idx_start = data_behav.index[data_behav["block"] == block][0]
        idx_end = data_behav.index[data_behav["block"] == block][-1]

        time_start = data_behav.iloc[idx_start]["time"]
        time_end = data_behav.iloc[idx_end]["time"]
        isi = data_behav.iloc[idx_start]["ISI"]
        sample_start = event_samples[idx_start]
        sample_end = event_samples[idx_end]

        duration = time_end - time_start
        sample_duration = (sample_end - sample_start) / sfreq

        if not np.isclose(sample_duration, duration, atol=0.1):
            print(
                f"⚠️ Block {block}: sample vs time duration mismatch "
                f"({sample_duration:.3f} vs {duration:.3f} s)"
            )

        blocks.append(
            dict(
                block=block,
                time_start=time_start,
                time_end=time_end,
                ISI=isi,
                sample_start=sample_start,
                sample_end=sample_end,
                duration=duration,
            )
        )

    return pd.DataFrame(blocks)


def compute_resp_freq(blocks, peaks, sfreq, participant):
    """Compute respiratory frequency per block."""
    results = []
    for _, row in blocks.iterrows():
        peak_in_block = peaks[
            (peaks >= row["sample_start"]) & (peaks <= row["sample_end"])
        ]
        if len(peak_in_block) < 2:
            continue

        duration_peak_to_peak = (peak_in_block[-1] - peak_in_block[0]) / sfreq
        resp_freq = respfreq(len(peak_in_block), duration_peak_to_peak / 60)

        results.append(
            dict(
                participant=participant,
                respiratory_frequency=resp_freq,
                ISI=row["ISI"],
                block_number=row["block"],
            )
        )
    return pd.DataFrame(results)

if __name__ == "__main__":
    variables = ["peaks", "event_samples_all", "event_ids_all", "sfreq"]
    dataset = "pilots"
    data = load_data(variables, dataset)


    # prepare output paths
    outdir = Path(__file__).parent / "results" / "h1"
    outdir.mkdir(exist_ok=True, parents=True)
    
    figpath = outdir / "h1_respiratory_frequency_vs_isi.png"
    txtpath = outdir/ "h1_LMEM_results.txt"


    # create an empty DataFrame to store the results
    df = pd.DataFrame(columns=["participant", "respiratory_frequency", "ISI", "block_number"])

    for participant, values in data.items():
        peaks = values["peaks"]
        event_samples = values["event_samples_all"]
        event_ids = values["event_ids_all"]
        sfreq = values["sfreq"]

        # load in the behavioural data
        behav_path = Path(__file__).parent / "data" / dataset / "raw" / f"{participant}_behavioural_data.csv"
        data_behav = pd.read_csv(behav_path)

        blocks = extract_blocks(data_behav, event_samples, sfreq)
        df_participant = compute_resp_freq(blocks, peaks, sfreq, participant)
        
        df = pd.concat([df, df_participant], ignore_index=True)


    df["ISI_centered"] = df["ISI"] - df["ISI"].min()

    result = LMEM(df)
    beta = result.params["ISI_centered"]
    ci = result.conf_int().loc["ISI_centered"]
    p = result.pvalues["ISI_centered"]

    ci_lower, ci_upper = ci[0], ci[1]
    print(f"\nbeta (ISI_centered) = {beta:.3f}, 95% CI = [{ci_lower:.3f}, {ci_upper:.3f}], p = {p:.3f}\n")

    with open(txtpath, "w") as f:
        f.write(result.summary().as_text())

        # also write random effects to text file
        f.write("\nRandom Effects:\n")
        for participant, re in result.random_effects.items():
            f.write(f"Participant {participant}:\n")
            f.write(f"{re}\n\n")


    # sort the df by participant
    df = df.sort_values("participant")
    plot(df, figpath=figpath)



    model_null = smf.mixedlm("respiratory_frequency ~ 1", data=df, groups=df["participant"])
    result_null = model_null.fit()


    # log-likelihoods
    llf_full = result.llf
    llf_null = result_null.llf

    # test statistic: 2 * difference in log-likelihoods
    lr_stat = 2 * (llf_full - llf_null)
    df_diff = result.df_modelwc - result_null.df_modelwc  # difference in number of estimated parameters

    # p-value
    pval = stats.chi2.sf(lr_stat, df_diff)

    print("Likelihood Ratio Test:")
    print(f"LR statistic = {lr_stat:.3f}, df = {df_diff}, p = {pval:.3g}")

