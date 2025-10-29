import numpy as np
from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import mne

from utils import create_trigger_mapping
from pyriodic.preproc import RawSignal
from pyriodic.viz import plot_phase_diagnostics, CircPlot
from pyriodic.phase_events import create_phase_events
from bioread import read_file

LOW_FREQ = 0.1
HIGH_FREQ = 1
WINDOW_SIZE_SMOOTHING = 50 #ms


def csv_to_events_array(df: pd.DataFrame, sfreq: float) -> np.ndarray:
    """
    Convert a CSV logfile to an events array in the format [sample_idx, 0, trigger_value].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'event_time' (s) and 'trigger' columns.
    sfreq : float
        Sampling frequency of the trigger channel (Hz).

    Returns
    -------
    events_array : np.ndarray
        Array of shape (n_events, 3) with [sample_idx, 0, trigger_value].
    """
    event_times = df['time'].values
    triggers = df['trigger'].values

    # convert times to sample indices
    sample_indices = np.round(event_times * sfreq).astype(int)

    # create 3-column array
    events_array = np.column_stack((sample_indices, np.zeros_like(sample_indices, dtype=int), triggers))
    return events_array


def trigger_extraction(data, sfreq):
    trigger_channels = [ch for ch in data.channels if "stim" in ch.name.lower()]

    trigger_data = np.array([ch.data for ch in trigger_channels])

    # dividing by 5, as trigger value is 5 when it is on. I want just 1's to correspond to the bits defined in trigger_mapping
    trigger_data = trigger_data/5


    # combining the values of the trigger channels, taking into account the bits they represent
    combined = np.zeros_like(trigger_data[0])
    for i, ch_data in enumerate(trigger_data):
        combined += ch_data.astype(int) << i  # each channel as a different bit

    raw_combined = mne.io.RawArray(
        combined[np.newaxis, :],
        mne.create_info(ch_names=['TRIG'], sfreq=sfreq, ch_types=['stim'])
    )
    
    return mne.find_events(raw_combined, shortest_event=1)

"""
def insert_missing_triggers(events_from_raw, events_from_csv):
    updated_events = []
    updated_events_description = []
    i_raw, i_csv = 0, 0
    mismatch_count = 0

    while i_raw < len(events_from_raw) and i_csv < len(events_from_csv):
        sample_raw, _, trig_raw = events_from_raw[i_raw]
        sample_csv, _, trig_csv = events_from_csv[i_csv]

        if trig_raw == trig_csv and abs(sample_raw - sample_csv) <= 100:
            # perfect match
            updated_events.append([sample_raw, 0, trig_raw])
            i_raw += 1
            i_csv += 1
            updated_events_description.append("Perfect match")
        elif abs(sample_raw - sample_csv) > 100:
            # check which one has the lowest sample index
            if sample_raw > sample_csv:
                # insert CSV event
                updated_events.append([sample_csv, 0, trig_csv])
                i_csv += 1
                mismatch_count += 1
                updated_events_description.append("Inserted CSV event")
            else:
                # assume that a 
                # insert raw event
                #updated_events.append([sample_raw, 0, trig_raw])
                i_raw += 1
                mismatch_count += 1
                updated_events_description.append("Inserted raw event")
                #print("inserted_raw event")

        else:
            print(f"could not fix match with csv trig {trig_csv}, sample {sample_csv} and raw trig {trig_raw}, sample {sample_raw}\
                  Moving on to the next csv index")
            i_csv += 1

    print(f"Mismatches resolved: {mismatch_count}")
    updated_events = np.array(updated_events)
       
    return updated_events

"""

def align_csv_with_raw(events_from_raw, events_from_csv, max_delta=100):
    """
    Align CSV events with raw triggers:
    - Keep all CSV events in order (CSV is ground truth)
    - Replace sample index with raw timing when a close match is found
    - Otherwise, keep CSV timing

    Parameters
    ----------
    events_from_raw : list of tuples
        Each tuple: (sample_raw, _, trig_raw)
    events_from_csv : list of tuples
        Each tuple: (sample_csv, _, trig_csv)
    max_delta : int
        Maximum difference in samples to consider a match.

    Returns
    -------
    aligned_events : np.ndarray
        Array of [sample_aligned, 0, trigger] (same length as CSV).
    match_flags : np.ndarray (bool)
        True where raw timing was used, False where CSV timing was kept.
    """
    raw_samples = np.array([s for s, _, _ in events_from_raw])
    raw_triggers = np.array([t for _, _, t in events_from_raw])

    aligned_events = []
    match_flags = []

    for sample_csv, _, trig_csv in events_from_csv:
        # Find candidate matches
        mask = raw_triggers == trig_csv
        if not np.any(mask):
            # No raw events with this trigger → use CSV timing
            aligned_events.append([sample_csv, 0, trig_csv])
            match_flags.append(False)
            continue

        # Compute distance from CSV event to all matching raw events
        candidates = raw_samples[mask]
        deltas = np.abs(candidates - sample_csv)
        idx_min = np.argmin(deltas)

        if deltas[idx_min] <= max_delta:
            # Use raw timing
            aligned_events.append([candidates[idx_min], 0, trig_csv])
            match_flags.append(True)
        else:
            # No close match → use CSV timing
            aligned_events.append([sample_csv, 0, trig_csv])
            match_flags.append(False)

    return np.array(aligned_events), np.array(match_flags)


def preprocess(raw):

    print("Applying bandpass filter")
    raw.filter_bandpass(low = LOW_FREQ, high = HIGH_FREQ)
    
    print("Smoothing")
    raw.smoothing(window_size = WINDOW_SIZE_SMOOTHING)
    
    print("Z-scoring")
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




def extract_PA_events(trigger_mapping, phase, event_samples, event_ids, fig_path):

    # flip so the trigger is the key
    condition_mapping = {v: k for k, v in trigger_mapping.items()}

    event_labels = [condition_mapping.get(trig, "unknown") for trig in event_ids]
    n_responses = 0
    n_targets = 0

    # for target events, check if there is a response after in event labels. If it is update the label by adding /hit or /miss accordingly. If there is no response, add /noresponse
    for i, label in enumerate(event_labels):
        if "response" in label:
            n_responses += 1
            
        if "target" in label:
            n_targets += 1
            if (i+1 < len(event_labels)):
                if "incorrect" in event_labels[i+1]:
                    event_labels[i] += "/incorrect"
                elif "correct" in event_labels[i+1]:
                    event_labels[i] += "/correct"
                else:
                    event_labels[i] += "/noresponse"
            else:
                event_labels[i] += "/noresponse"

    print(f"Number of targets: {n_targets}, number of responses: {n_responses}")
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


def filter_by_rejection(event_samples, rejected_indices, *arrays):
    mask_keep = [s not in rejected_indices for s in event_samples]
    return ( [s for s, keep in zip(event_samples, mask_keep) if keep],
             *([arr for arr, keep in zip(a, mask_keep) if keep] for a in arrays) )


if __name__ == "__main__":

    dataset = "pilots" # can also be "before_pilots" "pilots" and "raw", "simulation"

    # Paths
    data_dir = Path(__file__).parent / "data" / dataset / "raw"

    output_dir = data_dir.parent / "intermediate"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = data_dir.parent / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    participant_files = sorted(data_dir.glob("*.acq"))

    trigger_mapping = create_trigger_mapping(simulated=True if dataset=="simulation" else False)


    # Parameters
    plot_phase_diag = True

    # Loop through each participant
    for resp_path in participant_files:
        print(resp_path)
        

        participant_id = resp_path.stem.split("_")[0]
        behav_path = data_dir / f"{participant_id}_behavioural_data.csv"
        

        print(f"Processing participant {participant_id}...")

        # --- Load respiration data ---
        data = read_file(resp_path)
        sampling_rate = data.samples_per_second
        print(f"Sampling rate: {sampling_rate} Hz")
        t, resp = data.time_index, data.channels[0].data  # assuming respiration is the first channel
        #resp_data = np.loadtxt(resp_path, delimiter=",", skiprows=1)
        #t, resp = resp_data[:, 0], resp_data[:, 1]


        behav_data = pd.read_csv(behav_path)
        rt = behav_data['rt'].values
        print(len(rt), "responses found")
        intensity = behav_data['intensity'].values

        # --- create events ---
        print(f"Creating events for {participant_id}")
        events_from_raw = trigger_extraction(data, sfreq=sampling_rate)

        # updating events by inserting missing ones from the behavioural data
        print(f"Updating events by inserting missing ones from the behavioural data for {participant_id}")
        events_from_csv = csv_to_events_array(behav_data, sfreq=sampling_rate)
        events_from_csv[:, 0] = events_from_csv[:, 0] - events_from_csv[0, 0] + events_from_raw[0, 0]
        #events = insert_missing_triggers(events_from_csv=events_from_csv, events_from_raw=events_from_raw)
        events, match_flags = align_csv_with_raw(events_from_raw, events_from_csv)


        event_samples = events[:, 0]
        event_ids = events[:, -1]
        print(len(event_samples), "events found")

        print(f"Matched {match_flags.sum()} / {len(match_flags)} events ({100*match_flags.mean():.1f}%).")

        # --- Preprocess ---
        print(f"Preprocessing {participant_id}")
        raw = RawSignal(resp, fs=sampling_rate)
        preprocess(raw)

        # --- Extract phase angles ---
        phase, peaks, troughs = raw.phase_twopoint(prominence=0.3, distance=0.5)

        # set phase angles to NAN in breaks
        break_starts = event_samples[np.where(event_ids == trigger_mapping["break/start"])]
        break_ends = event_samples[np.where(event_ids == trigger_mapping["break/end"])]

        for start, end in zip(break_starts, break_ends):
            phase[start:end], raw.ts[start:end] = np.nan, np.nan

        # set phase angles to nan before experiement start and experiment end
        if participant_id == "SIL":
            exp_start, exp_end = 623186, 3307725

        elif participant_id == "JES":
            exp_start, exp_end = 162331, 2913923 
        
        else:
            exp_start = int(event_samples[np.where(event_ids == trigger_mapping["experiment/start"])])
            exp_end = int(event_samples[np.where(event_ids == trigger_mapping["experiment/end"])])
        
        phase[:exp_start], phase[exp_end:] = np.nan, np.nan
        raw.ts[:exp_start], raw.ts[exp_end:] = np.nan, np.nan

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
            troughs=troughs,
            events=event_samples,
            event_labels=event_ids
            )

        # --- Extract phase events ---
        circ, rejected_indices = extract_PA_events(
            trigger_mapping=trigger_mapping,
            phase=phase,
            event_samples=event_samples,
            event_ids=event_ids,
            fig_path=fig_dir / f"{participant_id}_circ_plot.png",
        )
        
        # find the indices of where event_samples is equal to the rejected_indices
        #idx_rejected = np.array([i for i, sample in enumerate(event_samples) if sample in rejected_indices])

        print(f"\n{[len(arr) for arr in [rt, intensity, event_ids]]} lle of arrays \n")
        event_samples_no_rejected, rt_no_rejected, intensity_no_rejected, event_ids_no_rejected = \
            filter_by_rejection(event_samples, rejected_indices, rt, intensity, event_ids)


        # --- Save to pickle ---
        output_file = output_dir / f"{participant_id}_preproc.pkl"
        with open(output_file, "wb") as f:
            pickle.dump({
                "t": t,
                "raw": resp,
                # "rejected_idx": np.array(idx_rejected),
                "event_samples_all": event_samples,
                "event_ids_all": event_ids,
                "event_ids": event_ids_no_rejected,
                "event_samples": event_samples_no_rejected,
                "preprocessed": raw.ts,
                "phase_ts": phase,
                "peaks": peaks,
                "circ": circ,
                "rt": np.array(rt_no_rejected),
                "intensity": np.array(intensity_no_rejected),
                "sfreq": sampling_rate,

            }, f)

        print(circ)

        print(f"Saved to: {output_file.name}")
  