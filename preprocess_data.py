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
WINDOW_SIZE_SMOOTHING = 50
DOWNSAMPLING_FREQ = 100


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
            }


    return trigger_mapping

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
        else:
            if abs(sample_raw - sample_csv) > 100:

                # check which one has the lowest sample index
                if sample_raw > sample_csv:
                    # insert CSV event
                    updated_events.append([sample_csv, 0, trig_csv])
                    i_csv += 1
                    mismatch_count += 1
                    updated_events_description.append("Inserted CSV event")
                else:
                    # insert raw event
                    updated_events.append([sample_raw, 0, trig_raw])
                    i_raw += 1
                    mismatch_count += 1
                    updated_events_description.append("Inserted raw event")

    print(f"Mismatches resolved: {mismatch_count}")
    updated_events = np.array(updated_events)
    updated_events.shape


       
    return updated_events


def preprocess(raw):
    # DOWNSAMPLE
    try:  
        raw.resample(sfreq=DOWNSAMPLING_FREQ)
    except NotImplementedError:
        print("REMEMBER TO IMPLEMENT RESAMPLING???")
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

    print(event_labels , "\n")
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


if __name__ == "__main__":

    dataset = "before_pilots" # can also be "before_pilots" "pilots" and "raw", "simulation"

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
        

        participant_id = resp_path.stem.split("_")[0]
        behav_path = data_dir / f"{participant_id}_behavioural_data.csv"
        

        print(f"Processing participant {participant_id}...")

        # --- Load respiration data ---
        data = read_file(resp_path)
        sampling_rate = data.samples_per_second
        t, resp = data.time_index, data.channels[0].data  # assuming respiration is the first channel
        #resp_data = np.loadtxt(resp_path, delimiter=",", skiprows=1)
        #t, resp = resp_data[:, 0], resp_data[:, 1]


        behav_data = pd.read_csv(behav_path)
        rt = behav_data['rt'].values
        print(len(rt), "responses found")
        intensity = behav_data['intensity'].values

        # --- create events ---
        events_from_raw = trigger_extraction(data, sfreq=sampling_rate)
        print(events_from_raw)


        # updating events by inserting missing ones from the behavioural data
        if len(behav_data) != len(events_from_raw):
            events_from_csv = csv_to_events_array(behav_data, sfreq=sampling_rate)
            events_from_csv[:, 0] = events_from_csv[:, 0] - events_from_csv[0, 0] + events_from_raw[0, 0]
            print(events_from_csv)
            events = insert_missing_triggers(
                events_from_csv=events_from_csv,
                events_from_raw=events_from_raw
            )
        else: 
            events = events_from_raw


        event_samples = events[:, 0]
        event_ids = events[:, -1]
        print(len(event_samples), "events found")

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
            trigger_mapping=trigger_mapping,
            phase=phase,
            event_samples=event_samples,
            event_ids=event_ids,
            fig_path=fig_dir / f"{participant_id}_circ_plot.png",
        )
        
        # find the indices of where event_samples is equal to the rejected_indices
        idx_rejected = [i for i, sample in enumerate(event_samples) if sample in rejected_indices]

        rt_no_rejected = [rt[i] for i in range(len(rt)) if i not in idx_rejected]
        intensity_no_rejected = [intensity[i] for i in range(len(intensity)) if i not in idx_rejected]
        event_ids_no_rejected = [event_ids[i] for i in range(len(event_ids)) if i not in idx_rejected]
        event_samples_no_rejected = [event_samples[i] for i in range(len(event_samples)) if i not in idx_rejected]

        # --- Save to pickle ---
        output_file = output_dir / f"{participant_id}_preproc.pkl"
        with open(output_file, "wb") as f:
            pickle.dump({
                "t": t,
                "raw": resp,
                "rejected_idx": np.array(idx_rejected),
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
  