import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt


from pyriodic.preproc import RawSignal
from pyriodic.viz import plot_phase_diagnostics, CircPlot
from pyriodic.phase_events import create_phase_events

LOW_FREQ = 0.1
HIGH_FREQ = 1
WINDOW_SIZE_SMOOTHING = 50

DOWNSAMPLING_FREQ = 100

def preprocess(raw):
    # DOWNSAMPLE
    try:  
        raw.resample(sfreq=DOWNSAMPLING_FREQ)
    except NotImplementedError:
        print("REMEMBER TO IMPLEMENT RESAMPLING")
        pass
    raw.filter_bandpass(low = LOW_FREQ, high = HIGH_FREQ)
    raw.smoothing(window_size = WINDOW_SIZE_SMOOTHING)
    raw.zscore()

def sanity_check_phase_angle(resp_timeseries = None, normalised_ts = None, peaks = None, troughs = None, phase_angle = None, savepath = None):
    fig, axes = plt.subplots(2, 1, figsize = (40, 4), dpi = 300)

    for var, label, color in zip([resp_timeseries, normalised_ts], ["original timeseries", "normalised interpolated timeseries"], ["darkblue", "forestgreen", "k"]):
        if var is not None:
            axes[0].plot(var, label = label, linewidth=1, color = color, alpha = 0.6)
    
    for var, label in zip([peaks, troughs], ["peaks", "troughs"]):
        tmp_y = [normalised_ts[i] for i in var]
        axes[0].scatter(var, tmp_y, zorder=1, alpha=0.5, s=2, label = label)

    if phase_angle is not None: 
        axes[1].plot(phase_angle, color = "grey", linewidth = 1)
        axes[1].set_ylabel("phase angle")

    axes[0].legend()

    for ax in axes:
        ax.set_xlim((0, len(normalised_ts))) # will give problems if normalised timeseries is not provided...

    plt.tight_layout()

    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath)

    else:
        return fig, axes




def extract_PA_events(phase, event_samples, event_ids, fig_path):
    trigger_mapping = {
        "stim/salient": 1,
        "target/right/hit": 2,
        "target/right/miss": 3,
        "target/left/hit": 4,
        "target/left/miss": 5,
    }

    # flip so the trigger is the key
    condition_mapping = {v: k for k, v in trigger_mapping.items()}

    event_labels = [condition_mapping.get(event_id, "unknown") for event_id in event_ids]
   # print(event_labels)

    circ, rejected_indices = create_phase_events(
        phase_ts=phase,
        events=event_samples,
        event_labels=np.array(event_labels),
        rejection_method="segment_duration_sd",
        rejection_criterion=3,
        return_rejected=True
    )

    if fig_path:
        circ.plot(group_by_labels=True)
        plt.savefig(fig_path)
        plt.close()

    return circ, rejected_indices


if __name__ == "__main__":

    # Paths
    data_dir = Path(__file__).parent / "simulation" / "data" / "raw"

    output_dir = data_dir.parent / "intermediate"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = data_dir.parent / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    participant_files = sorted(data_dir.glob("participant_*_respiration.csv"))

    # Parameters
    sampling_rate = 100
    plot_phase_diag = False

    # Loop through each participant
    for resp_path in participant_files:

        participant_id = resp_path.stem.split("_")[1]
        event_path = data_dir / f"participant_{participant_id}_events.csv"

        print(f"Processing participant {participant_id}...")

        # --- Load respiration data ---
        resp_data = np.loadtxt(resp_path, delimiter=",", skiprows=1)
        t, resp = resp_data[:, 0], resp_data[:, 1]

        # --- Load events ---
        event_log = np.loadtxt(event_path, delimiter=",", skiprows=1)
        event_samples, event_ids, rt, intensity = event_log[:, 0], event_log[:, 1], event_log[:, 2], event_log[:, 3]

        # change event samples and event ids to int
        event_samples = event_samples.astype(int)
        event_ids = event_ids.astype(int)


        # --- Preprocess ---
        raw = RawSignal(resp, fs=sampling_rate)
        preprocess(raw)


        # --- Extract phase angles ---
        phase, peaks, troughs = raw.phase_twopoint(prominence=0.1, distance=0.5)

        # check how preprocessing went by plotting
        if plot_phase_diag:
            plot_phase_diagnostics(
                {
                    "Raw data": resp,
                    "Two-point": phase
                },
                start = 20,
                window_duration = 20,
                fs = raw.fs,
            data = raw.ts, 
            peaks=peaks,
            troughs=troughs
            )

        # --- Extract phase events ---
        circ, rejected_indices = extract_PA_events(
            phase=phase,
            event_samples=event_samples,
            event_ids=event_ids,
            fig_path = fig_dir / f"participant_{participant_id}_circ_plot.png",
            )
        
        # find the indices of where event_samples is equal to the rejected_indices
        idx_rejected = [i for i, sample in enumerate(event_samples) if sample in rejected_indices]

        rt_no_rejected = [rt[i] for i in range(len(rt)) if i not in idx_rejected]
        intensity_no_rejected = [intensity[i] for i in range(len(intensity)) if i not in idx_rejected]

        # --- Save to pickle ---
        output_file = output_dir / f"participant_{participant_id}_preproc.pkl"
        with open(output_file, "wb") as f:
            pickle.dump({
                "t": t,
                "raw": resp,
                "event_samples": event_samples,
                "event_ids": event_ids,
                "preprocessed": raw.ts,
                "phase_ts": phase,
                "peaks": peaks,
                "circ": circ,
                "rt": np.array(rt_no_rejected),
                "intensity": np.array(intensity_no_rejected)

            }, f)

        print(f"Saved to: {output_file.name}")