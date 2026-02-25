import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils import load_data, create_trigger_mapping
import pandas as pd

trigger_mapping = create_trigger_mapping()
trigger_mapping = {v: k for k, v in trigger_mapping.items()}


def plot_surrogate_procedure(data, figpath):
    fig, axes = plt.subplots(
        5, 1, figsize=(8, 9), dpi=300, sharex=True,
        gridspec_kw=dict(hspace=0.6, height_ratios=[1.2, 1, 1, 1, 1])
    )

    samp_min, samp_max = 504060, 529108 + 20000
    ts = data["preprocessed"]
    phase_ts = data["phase_ts"]
    peaks = data["peaks"] - samp_min
    peaks = peaks[(peaks >= 0) & (peaks < (samp_max - samp_min))]

    events = data["event_samples_all"] - samp_min
    events_ids = data["event_ids_all"]
    events = np.array([ev for ev, ev_id in zip(events, events_ids)
                       if "target" in trigger_mapping[ev_id]])
    events = events[(events >= 0) & (events < (samp_max - samp_min))]

    colors = plt.colormaps["tab20"].colors
    
    observed_colour = "forestgreen"

    
    segments = [(int(peaks[i]), int(peaks[i + 1])) for i in range(len(peaks) - 1)]
    segment_colors = colors[:len(segments)]

    
    for (start, end), color in zip(segments, segment_colors):
        axes[0].plot(np.arange(start, end),
                     ts[samp_min + start:samp_min + end],
                     color=color, lw=1)
    axes[0].set_ylabel("Respiration signal", rotation=0, labelpad=40)
    axes[0].set_title("Observed data", fontsize=11, loc="left", pad=10)
    #axes[0].spines[['bottom']].set_visible(False)

    # --- (2) Phase signal ---
    for (start, end), color in zip(segments, segment_colors):
        axes[1].plot(np.arange(start, end),
                     phase_ts[samp_min + start:samp_min + end],
                     color=color, lw=1)
    axes[1].set_ylabel("Phase angle", rotation=0, labelpad=40)
    #axes[1].spines[['top', 'bottom']].set_visible(False)

    # Add event markers to observed phase
    for idx_ev, ev in enumerate(events):
        axes[1].axvline(ev, color="grey", linestyle="--", alpha=0.4, zorder=3)
        axes[1].scatter(ev, phase_ts[samp_min + ev], color=observed_colour, s=10, zorder=5, label ="Observed events" if idx_ev == 1 else "")
        axes[1].legend(loc="upper right", fontsize=8)



    # --- (3–5) Surrogates ---
    for i_sur, ax in enumerate(axes[2:], start=1):
        np.random.seed(40 + i_sur)
        shuffled_indices = np.random.permutation(len(segments))

        surrogate = np.concatenate([
            phase_ts[samp_min + segments[idx][0]: samp_min + segments[idx][1]]
            for idx in shuffled_indices
        ])
        segment_boundaries = np.cumsum([0] + [segments[idx][1] - segments[idx][0]
                                              for idx in shuffled_indices])

        for i, idx in enumerate(shuffled_indices):
            start, end = segment_boundaries[i], segment_boundaries[i + 1]
            ax.plot(np.arange(start, end),
                    surrogate[start:end],
                    color=segment_colors[idx], lw=1)

        ax.set_ylabel(f"Surrogate\n{i_sur}", rotation=0, labelpad=40, color="dimgray")

        for ev_idx, ev in enumerate(events):
            if ev < len(surrogate):
                ax.axvline(ev, color="lightgrey", linestyle="--", alpha=0.5, zorder=3)

                ax.scatter(ev, surrogate[ev], color="grey", s=8, zorder=5, label="Surrogate events" if ev_idx == 1 else "")
                if i_sur == 1:
                    ax.legend(loc="upper right", fontsize=8)

        

    axes[-1].set_xlabel("Samples")

    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)

    for i, ax in enumerate(axes[1:]):
        ax.set_yticks([0, np.pi, 2 * np.pi])
        ax.set_yticklabels(["0", r"$\pi$", r"2$\pi$"])
        ax.set_ylim([0, 2 * np.pi])
        # turn spines and ticks gray for surrogate plots
        if i > 0:
            ax.spines[['left', 'bottom']].set_color("dimgray")
            ax.tick_params(axis='y', colors="dimgray")


    plt.tight_layout()
    plt.subplots_adjust(hspace=0.7)
    plt.savefig(figpath, bbox_inches="tight")
    plt.close(fig)


def plot_QUEST_moving_window(circ, figpath=None):


    # parameters from analysis
    # for bins refitting of psychometric function
    DELTA_CENTER = np.pi/20
    WIDTH = np.pi/4


    circ = circ["target"]
    metadata = circ.metadata

    # convert series (metadata["n_in_block"]) to integer
    n_target = metadata["n_in_block"].to_numpy()/4 + metadata["block"].astype(int).to_numpy() * 8
    intensity = metadata["intensity"].to_numpy()
    correct = metadata["correct"].astype("float").to_numpy()
    phase_angle = circ.data


    
    # plotting!!!
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # QUEST procedure

    # line between all points
    axes[0].plot(n_target, intensity, color="lightgray", lw=1)

    masks = {
        "Correct":    (correct == 1),
        "Incorrect":  (correct == 0),
        "No response": np.isnan(correct),
    }

    colors = {
        "Correct": "forestgreen",
        "Incorrect": "indianred",
        "No response": "gray",
    }

    for label, mask in masks.items():
        axes[0].scatter(
            n_target[mask],
            intensity[mask],
            color=colors[label],
            label=label,
            alpha=1,
            s=15,
            zorder=2
        )

        if label != "No response":

            # moving window
            axes[1].scatter(
                    phase_angle[mask],
                    intensity[mask],
                    color=colors[label],
                    label=label,
                    alpha=1,
                    s=15
                )


    axes[0].set_xlabel("Target number")
    axes[0].set_xlim([0, 200])
    axes[0].legend()

    # set ticks for phase angle plot
    axes[1].set_xticks([0, np.pi, 2 * np.pi])
    axes[1].set_xticklabels(["0", r"$\pi$", r"2$\pi$"])
    axes[1].set_xlabel("Phase angle")
    axes[1].set_xlim([0, 2 * np.pi])

    for ax in axes:
        ax.set_ylabel("Intensity (mA)")
        # remove top and right spines
        ax.spines[['top', 'right']].set_visible(False)


    # draw one example of moving window
    center = np.pi - DELTA_CENTER * 12
    window_start = center - WIDTH / 2
    window_end = center + WIDTH / 2

    # red lines to indicate the window
    axes[1].axvspan(
        window_start, window_end, 
        edgecolor = "red", fill=False, facecolor=None,
        linewidth=2)

    
        
    plt.tight_layout()
    # adjust hspace
    plt.subplots_adjust(hspace=0.5)

    plt.savefig(figpath)
    plt.close(fig)
    

def plot_trials_per_ISI(data, figpath=None):
    df_all = pd.DataFrame()

    for ID, dat in data.items():
        behav_data = dat["behav_data"]
        behav_data["ID"] = ID
        df_all = pd.concat([df_all, behav_data], ignore_index=True)
    
    
    df_counts = (
        df_all
        .loc[df_all["event_type"] == "response"]            # keep trials with responses
        .groupby(["ID", "ISI"])
        .size()
        .unstack(fill_value=0)
        .astype(int)
        .sort_index(axis=1)                                 # ISIs in ascending order
    )

    # also make a group level mean 
    df_counts.loc["group_mean"] = df_counts.mean(axis=0)
    fig, ax = plt.subplots(figsize=(16, 6), dpi = 300)

    # reorder: subjects first, group mean last
    df_plot = df_counts.copy()
    subject_idx = [i for i in df_plot.index if i != "group_mean"]
    df_plot = df_plot.loc[subject_idx + ["group_mean"]]

    isi_levels = sorted(df_plot.columns)

    x = np.arange(len(df_plot))
    width = 0.2

    cmap  = plt.colormaps["Greens"]
    color_vals = np.linspace(0.35, 0.9, len(isi_levels))

    for i, (isi, cval) in enumerate(zip(isi_levels, color_vals)):
        bars = ax.bar(
            x + i * width,
            df_plot[isi].values,
            width,
            color=cmap(cval),
            label=f'{isi}'
        )

        # hatch the group mean bars
        bars[-1].set_hatch("//")
        bars[-1].set_edgecolor("black")

    # x-axis formatting
    ax.set_xticks(x + width * (len(isi_levels) - 1) / 2)
    ax.set_xticklabels(df_plot.index, rotation=45)

    ax.set_xlabel('Participant')
    ax.set_ylabel('Number of responses')
   

    ax.axhline(
        y=104,
        color='grey',
        linestyle='--',
        label='Targets presented per ISI'
    )
    
    ax.legend(ncols=1, bbox_to_anchor=(1, 1), loc='upper left')
    # remove top and right spines
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()

    
    if figpath:
        plt.savefig(figpath, bbox_inches="tight")

if __name__ == "__main__":
    outdir = Path(__file__).parent / "results" / "methods_fig"
    outdir.mkdir(exist_ok=True, parents=True)

    #variables = ["peaks", "troughs", "event_samples_all", "event_ids_all",
    #             "preprocessed", "phase_ts"]
    
    #data = load_data(variables, "pilots")
    #plot_surrogate_procedure(data["JES"], outdir / "surrogate_procedure.svg")


    data = load_data(["circ"], "raw")

    participant_id = 5
    plot_QUEST_moving_window(data[participant_id]["circ"], outdir / "QUEST_moving_window.svg")

    variables = ["behav_data"] 
    data = load_data(variables, "raw")

    plot_trials_per_ISI(data, outdir / "trials_per_ISI.svg")