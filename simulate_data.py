import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv


def simulate_event_stream_with_samples(
    sampling_rate=100,
    order=[0, 1, 2, 3, 4]* 12,
    ISIs=[1.23, 1.47, 1.38, 1.52, 1.75],
    n_sequences=5,
    prop_left_right=[0.5, 0.5],
    trigger_mapping=None,
    seed=42
):
    if trigger_mapping is None:
        trigger_mapping = {
        "stim/salient": 1,
        "target/right/hit": 2,
        "target/right/miss": 3,
        "target/left/hit": 4,
        "target/left/miss": 5,
        "buttonpress": 6
    }

    np.random.seed(seed)
    t = 0.0
    event_log = []
    
        
    for isi_idx in order:
        ISI = ISIs[isi_idx]
        for _ in range(n_sequences):
            for _ in range(3):  # 3 salient stimuli
                t += ISI
                sample = int(t * sampling_rate)
                event_log.append((sample, trigger_mapping["stim/salient"],np.nan, 3) )

            intensity = np.round(np.random.uniform(0.2, 2.0), 1)

            t += ISI
            side = "left" if np.random.rand() < prop_left_right[0] else "right"
            prob_correct = 0.5 + intensity * 0.3

            correct = 1 if np.random.rand() < prob_correct else 0

            event_type = f"target/{side}/hit" if correct else f"target/{side}/miss"
            rt = np.random.uniform(0.1, 0.5)
            sample = int(t * sampling_rate)
            event_log.append((sample, trigger_mapping[event_type], rt, intensity))
            #event_log.append((sample + rt * sampling_rate, trigger_mapping["buttonpress"], np.nan, np.nan))

    total_samples = int(t * sampling_rate)
    return event_log, total_samples


def simulate_respiration_belt_data(
    duration_s,
    sampling_rate,
    base_rate_hz,
    variability=0.0001,
    noise_std=0.02
):
    t = np.linspace(0, duration_s, int(duration_s * sampling_rate))
    freq_mod = np.sin(0.1 * 2 * np.pi * t) * variability
    amp_mod = 1 + np.sin(0.05 * 2 * np.pi * t) * variability
    phase = 2 * np.pi * (base_rate_hz + freq_mod) * t
    signal = amp_mod * np.sin(phase)
    signal += np.random.normal(0, noise_std, size=t.shape)
    return t, signal


if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path(__file__).parent / "data" / "simulation"
    data_dir.mkdir(exist_ok=True, parents=True)

    n_participants = 30


    for i in range(n_participants):
        event_log, total_samples = simulate_event_stream_with_samples(
            sampling_rate=100,  
            seed=42 + i  # vary seeds
        )

        baserate = np.random.uniform(0.2, 0.4)

        duration_s = total_samples / 100 + 50
        t, signal = simulate_respiration_belt_data(
            base_rate_hz=baserate,
            duration_s=duration_s,
            sampling_rate=100
        )
        if i == 0:
            print(f"Total samples: {total_samples}, Duration: {duration_s/60:.2f} minutes")

        # Save respiration
        resp_file = data_dir / f"{i:02d}.csv"
        np.savetxt(resp_file, np.column_stack((t, signal)), delimiter=",", header="time,signal", comments="")

        # Save event log
        event_file = data_dir / f"{i:02d}_behavioural_data.csv"
        with open(event_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample", "event_id", "intensity", "rt"])
            writer.writerows(event_log)

        print(f"Saved participant {i:02d}")
