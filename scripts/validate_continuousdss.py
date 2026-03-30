"""Validate ContinuousDSS biomarker extraction on real lexical decision EEG.

Streams sub-02 DSI-24 data through ContinuousDSS and compares condition-related
effects (HF words, LF words, pseudowords) in DSS sources vs raw channels.

Usage:
    python scripts/validate_continuousdss.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import mne
from scipy.signal import welch
from scipy.stats import ttest_ind, ttest_rel
from mne_denoise.dss.streaming import ContinuousDSS


# ---- Configuration ----
VHDR = r"D:\PSY3008\eeg\bids\sub-02\ses-001\eeg\sub-02_ses-001_task-lexical_run-01_eeg.vhdr"
EEG_CHANNELS = [
    "P3", "C3", "F3", "Fz", "F4", "C4", "P4", "Cz",
    "Fp1", "Fp2", "T7", "P7", "O1", "O2", "F7", "F8", "P8", "T8",
]
BLOCK_SIZE = 30       # samples per streaming block (100 ms at 300 Hz)
EPOCH_TMIN = -0.2     # seconds before stimulus
EPOCH_TMAX = 0.8      # seconds after stimulus
N400_TMIN = 0.3       # N400 window start
N400_TMAX = 0.5       # N400 window end

# DSS configurations to compare
DSS_CONFIGS = {
    "N400-band (2-7 Hz)": {"freq_band": (2, 7), "lambda_0": 0.999, "lambda_1": 0.99, "warmup": 100},
    "Theta (4-8 Hz)":     {"freq_band": (4, 8), "lambda_0": 0.995, "lambda_1": 0.99, "warmup": 50},
    "Alpha (8-12 Hz)":    {"freq_band": (8, 12), "lambda_0": 0.995, "lambda_1": 0.99, "warmup": 50},
}


def load_data():
    """Load and preprocess the lexical decision EEG data."""
    raw = mne.io.read_raw_brainvision(VHDR, preload=True, verbose=False)
    raw_eeg = raw.copy().pick(EEG_CHANNELS)
    raw_eeg.filter(0.5, 30, verbose=False)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    stim_events = {
        "HF":     events[events[:, 2] == event_id["Stimulus/S131"]],
        "LF":     events[events[:, 2] == event_id["Stimulus/S132"]],
        "Pseudo": events[events[:, 2] == event_id["Stimulus/S140"]],
    }
    return raw_eeg, stim_events


def stream_dss(data, sfreq, config):
    """Run ContinuousDSS in streaming mode and return source timeseries."""
    n_channels = data.shape[0]
    cdss = ContinuousDSS(
        n_channels=n_channels,
        sfreq=sfreq,
        bias="bandpass",
        bias_params={"freq_band": config["freq_band"]},
        n_components=3,
        lambda_0=config["lambda_0"],
        lambda_1=config["lambda_1"],
        solve_interval=10,
        warmup_blocks=config["warmup"],
    )

    n_blocks = data.shape[1] // BLOCK_SIZE
    all_sources = np.zeros((3, n_blocks * BLOCK_SIZE))

    for i in range(n_blocks):
        block = data[:, i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
        cdss.process_block(block)
        if cdss.filters is not None:
            mean = cdss._cov_baseline.mean
            sources = cdss.filters @ (block - mean[:, np.newaxis])
            all_sources[:, i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE] = sources

    return all_sources, cdss


def epoch_and_erp(signal, stim_events, sfreq, is_multichannel=False):
    """Epoch a signal around stimulus events and compute ERPs.

    Parameters
    ----------
    signal : ndarray
        (n_comp, n_samples) for DSS sources or (n_samples,) for single channel.
    stim_events : dict
        Condition name -> events array.
    sfreq : float
    is_multichannel : bool
        If True, signal is (n_comp, n_samples).

    Returns
    -------
    erps : dict of condition -> (n_comp, epoch_len) or (epoch_len,)
    epochs_by_cond : dict of condition -> list of epoch arrays
    """
    n_pre = int(abs(EPOCH_TMIN) * sfreq)
    n_post = int(EPOCH_TMAX * sfreq)

    erps = {}
    epochs_by_cond = {}

    for cond, evts in stim_events.items():
        epochs = []
        for evt in evts:
            onset = evt[0]
            start, end = onset - n_pre, onset + n_post
            if is_multichannel:
                if start >= 0 and end < signal.shape[1]:
                    ep = signal[:, start:end].copy()
                    ep -= ep[:, :n_pre].mean(axis=1, keepdims=True)
                    epochs.append(ep)
            else:
                if start >= 0 and end < signal.shape[0]:
                    ep = signal[start:end].copy()
                    ep -= ep[:n_pre].mean()
                    epochs.append(ep)

        epochs = np.array(epochs)
        erps[cond] = epochs.mean(axis=0)
        epochs_by_cond[cond] = epochs

    return erps, epochs_by_cond


def compute_n400_stats(epochs_by_cond, sfreq, comp_idx=0, is_multichannel=True):
    """Compute N400 amplitude statistics per condition."""
    n_pre = int(abs(EPOCH_TMIN) * sfreq)
    t = np.arange(-n_pre, int(EPOCH_TMAX * sfreq)) / sfreq
    n400_mask = (t >= N400_TMIN) & (t <= N400_TMAX)

    stats = {}
    for cond, epochs in epochs_by_cond.items():
        if is_multichannel:
            amps = epochs[:, comp_idx, :][:, n400_mask].mean(axis=1)
        else:
            amps = epochs[:, n400_mask].mean(axis=1)
        stats[cond] = amps

    return stats


def main():
    print("=" * 60)
    print("ContinuousDSS Validation: Lexical Decision EEG (sub-02)")
    print("=" * 60)

    raw_eeg, stim_events = load_data()
    data = raw_eeg.get_data()
    sfreq = raw_eeg.info["sfreq"]

    print(f"\nData: {data.shape[0]} channels, {data.shape[1]/sfreq:.0f}s, {sfreq:.0f} Hz")
    print(f"Trials: HF={len(stim_events['HF'])}, LF={len(stim_events['LF'])}, "
          f"Pseudo={len(stim_events['Pseudo'])}")

    # --- Raw channel analysis (Cz as reference) ---
    print(f"\n{'='*60}")
    print("RAW CHANNEL ANALYSIS (Cz - classic N400 site)")
    print("=" * 60)

    cz_idx = EEG_CHANNELS.index("Cz")
    cz_data = data[cz_idx]
    _, cz_epochs = epoch_and_erp(cz_data, stim_events, sfreq, is_multichannel=False)
    cz_stats = compute_n400_stats(cz_epochs, sfreq, is_multichannel=False)

    for cond in ["HF", "LF", "Pseudo"]:
        amp = cz_stats[cond]
        print(f"  {cond:6s}: N400 mean = {amp.mean()*1e6:+.2f} uV (SE={amp.std()/np.sqrt(len(amp))*1e6:.2f})")

    for c1, c2 in [("LF", "HF"), ("Pseudo", "HF")]:
        t_val, p_val = ttest_ind(cz_stats[c1], cz_stats[c2])
        d = (cz_stats[c1].mean() - cz_stats[c2].mean()) / np.sqrt(
            (cz_stats[c1].std()**2 + cz_stats[c2].std()**2) / 2
        )
        print(f"  {c1} vs {c2}: d={d:.3f}, t={t_val:.2f}, p={p_val:.4f}")

    # --- DSS analysis ---
    for config_name, config in DSS_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"DSS: {config_name}")
        print("=" * 60)

        sources, cdss = stream_dss(data, sfreq, config)
        eigenvalues = cdss.eigenvalues

        print(f"  Eigenvalues: {np.round(eigenvalues, 4)}")

        _, dss_epochs = epoch_and_erp(sources, stim_events, sfreq, is_multichannel=True)

        for comp in range(3):
            dss_stats = compute_n400_stats(dss_epochs, sfreq, comp_idx=comp)

            print(f"\n  DSS{comp+1} (eigenvalue={eigenvalues[comp]:.4f}):")
            for cond in ["HF", "LF", "Pseudo"]:
                amp = dss_stats[cond]
                print(f"    {cond:6s}: N400 mean = {amp.mean():+.6f} (SE={amp.std()/np.sqrt(len(amp)):.6f})")

            for c1, c2 in [("LF", "HF"), ("Pseudo", "HF")]:
                pooled_std = np.sqrt(
                    (dss_stats[c1].std()**2 + dss_stats[c2].std()**2) / 2
                )
                d = (dss_stats[c1].mean() - dss_stats[c2].mean()) / pooled_std if pooled_std > 0 else 0
                t_val, p_val = ttest_ind(dss_stats[c1], dss_stats[c2])
                sig = "*" if p_val < 0.05 else ""
                print(f"    {c1} vs {c2}: d={d:+.3f}, t={t_val:+.2f}, p={p_val:.4f} {sig}")

    print(f"\n{'='*60}")
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
