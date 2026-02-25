import numpy as np
from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import mne

from utils import create_trigger_mapping
from pyriodic.preproc import RawSignal
from pyriodic.viz import plot_phase_diagnostics
from pyriodic.phase_events import create_phase_events
from bioread import read_file

WINDOW_SIZE_SMOOTHING = 300 #ms
IDS = [
    "VUS", "DGO", "QVG", "OJN", "LIM", "FUS", "HAS", 
    "SXC", "NAL", "ULO", "KBY", "ZLP", "JAK", "LAM", 
    "HUC", "ZEL", "KIL", "MAJ", "JQA", "PAX", "INH", 
    "IJQ", "XIH", "LJB", "WRS", "PLO"
    ]

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

    events_from_csv[:, 0] = events_from_csv[:, 0] - events_from_csv[0, 0] + events_from_raw[0, 0]

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
 

    aligned_events = np.array(aligned_events)
    match_flags = np.array(match_flags)    
    

    return aligned_events, match_flags

def preprocess(raw):
    
    print("Smoothing")
    raw.smoothing(window_size = WINDOW_SIZE_SMOOTHING)
    
    print("Z-scoring")
    raw.zscore()

    return raw


def update_event_labels(event_ids, trigger_mapping, behav_data):

    # check that behav_data has same length as event_ids
    if len(behav_data) != len(event_ids):
        print("Warning: behavioural data length does not match event IDs length.")
        print(f"Behavioural data length: {len(behav_data)}, Event IDs length: {len(event_ids)}")
    # flip so the trigger is the key
    condition_mapping = {v: k for k, v in trigger_mapping.items()}

    event_labels = [condition_mapping[eid] if eid in condition_mapping else f"unknown_{eid}" for eid in event_ids]

    updated_labels = event_labels.copy()

    n_responses = 0
    n_targets = 0

    # for target events, check if there is a response after in event labels. 
    # update the label by adding /correct, /incorrect or /noresponse
    # also add ISI before target label
    for i, label in enumerate(event_labels):
        if "response" in label:
            n_responses += 1
            
        if "target" in label:
            isi = behav_data.loc[i, "ISI"]
            n_targets += 1
            new_label = f"{isi}/{label}"
            if (i+1 < len(event_labels)):
                if "incorrect" in event_labels[i+1]:
                    updated_labels[i]=new_label + "/incorrect"
                elif "correct" in event_labels[i+1]:
                    updated_labels[i]=new_label + "/correct"
                else:
                    updated_labels[i]=new_label + "/noresponse"
            else:
                updated_labels[i]=new_label + "/noresponse"

        if "salient" in label:
            isi = behav_data.loc[i, "ISI"]
            new_label = f"{isi}/{label}"
            updated_labels[i]=new_label

    print(f"Number of targets: {n_targets}, number of responses: {n_responses}")
    print(np.unique(updated_labels, return_counts=True))
    return updated_labels

def extract_PA_events(phase, event_samples, event_labels, metadata_df, fig_path=None):

    circ = create_phase_events(
        phase_ts=phase,
        events=np.array(event_samples),
        event_labels=np.array(event_labels),
        rejection_method="segment_duration_sd",
        rejection_criterion=3,
        metadata=metadata_df
    )

    if fig_path:
        circ.plot(group_by_labels=True)
        plt.savefig(fig_path)
        plt.close()

    return circ


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data" / "raw" 

    output_dir = data_dir / "intermediate"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = data_dir / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    trigger_mapping = create_trigger_mapping()

    # Parameters
    plot_phase_diag = False
    
    for i_subj, participant_id in enumerate(IDS):
        #participant_id = resp_path.stem.split("_")[0]
        behav_path = data_dir / "behavioural" / f"{participant_id[:3]}_behavioural_data.csv"
        resp_path = data_dir / f"{participant_id}0000.acq"
        

        print(f"Processing participant {participant_id}...")

        # --- Load respiration data ---
        data = read_file(resp_path)
        sampling_rate = data.samples_per_second
        t, resp = data.time_index, data.channels[0].data  # assuming respiration is the first channel

        behav_data = pd.read_csv(behav_path)
        
        # add the response time and correct from the response to the previous target
        for i, row in behav_data.iterrows():
            if row["event_type"] == "response":
                rt = row["rt"]
                correct = row["correct"]
                
                # check that the row before is a target
                if "target" in behav_data.loc[i-1, "event_type"]:
                    behav_data.loc[i-1, "rt"] = rt
                    behav_data.loc[i-1, "correct"] = correct

        # --- create events ---
        print(f"Creating events for {participant_id}")
        events_from_raw = trigger_extraction(data, sfreq=sampling_rate)


        # remove the first two events if they are break start and break end
        if events_from_raw[0, -1] == trigger_mapping["break/start"] and events_from_raw[1, -1] == trigger_mapping["break/end"]:

            # save the sample indices of the removed events
            print("Removed initial break start and break end from raw events")
            removed_events = events_from_raw[1]
            events_from_raw = events_from_raw[2:]


        # updating events by inserting missing ones from the behavioural data
        print(f"Updating events by inserting missing ones from the behavioural data for {participant_id}")
        events_from_csv = csv_to_events_array(behav_data, sfreq=sampling_rate)
        # remove experiment start from csv if it is there
        if events_from_csv[0, -1] == trigger_mapping["experiment/start"]:
            events_from_csv = events_from_csv[1:]
            print("Removed experiment start from CSV events")
        
        events, match_flags = align_csv_with_raw(events_from_raw, events_from_csv)

        event_samples = events[:, 0]
        event_ids = events[:, -1]


        # add the sample for the experiment start 
        if 'removed_events' in locals():
            event_samples = np.insert(event_samples, 0, removed_events[0])
            event_ids = np.insert(event_ids, 0, trigger_mapping["experiment/start"])
            print("Added back removed initial break events to the aligned events")

        print(f"Matched {match_flags.sum()} / {len(match_flags)} events ({100*match_flags.mean():.1f}%).")

        # --- Preprocess ---
        print(f"Preprocessing {participant_id}")
        raw = RawSignal(resp, fs=sampling_rate)
        raw = preprocess(raw)


        # --- Extract phase angles ---
        if participant_id == "NAL" or participant_id == "PAX":
            prominence, distance = 0.2, 0.3
        elif participant_id == "LAM":
            prominence, distance = 0.01, 0.3
        elif participant_id == "JQA":
            prominence, distance = 0.1, 0.3
        elif participant_id == "WRS":
            prominence, distance = 0.05, 0.3
        elif participant_id == "SXC" or participant_id == "PLO":
            prominence, distance = 0.2, 0.4
        else:
            prominence, distance = 0.3, 0.5
        
    
        phase, peaks, troughs = raw.phase_twopoint(prominence=prominence, distance=distance)

        # set phase angles to NAN in breaks
        break_starts = event_samples[np.where(event_ids == trigger_mapping["break/start"])]
        break_ends = event_samples[np.where(event_ids == trigger_mapping["break/end"])]

        for start, end in zip(break_starts, break_ends):
            phase[start:end], raw.ts[start:end] = np.nan, np.nan

        # set phase angles to nan before experiment start and experiment end
        try: 
            exp_start = int(event_samples[np.where(event_ids == trigger_mapping["experiment/start"])][0])
            exp_end = int(event_samples[np.where(event_ids == trigger_mapping["experiment/end"])][0])
        
            phase[:exp_start], phase[exp_end:] = np.nan, np.nan
            raw.ts[:exp_start], raw.ts[exp_end:] = np.nan, np.nan
        except:
            print("No experiment start or end trigger found")

        event_labels = update_event_labels(event_ids, trigger_mapping, behav_data)
        
        # add intensity information to the response events in the behav_data
        intensity = []
        for i in range(len(behav_data)):
            if "response" in behav_data.iloc[i]["event_type"]:
                intensity.append(behav_data.iloc[i-1]["intensity"])
            else:
                intensity.append(behav_data.iloc[i]["intensity"])
        behav_data["intensity"] = intensity
        
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
            event_labels=[lab.split('/')[0] for lab in event_labels],  # only plot the main label
            )

        behav_data['event_samples'] = event_samples
        behav_data['event_ids'] = event_ids

    
        # --- Extract phase events ---
        circ = extract_PA_events(
            phase=phase,
            event_samples=event_samples,
            event_labels=event_labels,
            fig_path=fig_dir / f"{i_subj+1}_circ_plot.png",
            metadata_df=behav_data.copy()
        )
        
        # --- Save to pickle ---
        output_file = output_dir / f"{i_subj+1}_preproc.pkl"
        with open(output_file, "wb") as f:
            pickle.dump({
                "t": t,
                "raw": resp,
                "preprocessed": raw.ts,
                "phase_ts": phase,
                "peaks": peaks,
                "troughs": np.array(troughs),
                "circ": circ,
                "sfreq": sampling_rate,
                "behav_data": behav_data

            }, f)

        print(f"Saved to: {output_file.name}")
  