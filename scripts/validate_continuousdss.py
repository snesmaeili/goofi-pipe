"""Validate ContinuousDSS biomarker extraction on real lexical decision EEG.

Streams sub-02 DSI-24 data through ContinuousDSS and evaluates:
1. Condition-related ERP effects (HF words, LF words, pseudowords)
2. Single-trial classification accuracy (3-class and HF-vs-LF)
3. Multi-band DSS vs raw channel features

Usage:
    python scripts/validate_continuousdss.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import mne
from scipy.signal import welch, resample
from scipy.stats import ttest_ind
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

DSS_BANDS = {
    "Alpha (8-12 Hz)": (8, 12),
    "Theta (4-8 Hz)":  (4, 8),
}


def load_data():
    """Load and preprocess the lexical decision EEG data."""
    raw = mne.io.read_raw_brainvision(VHDR, preload=True, verbose=False)
    raw_eeg = raw.copy().pick(EEG_CHANNELS)
    raw_eeg.filter(0.5, 30, verbose=False)

    # Extract events from the Trigger channel
    trigger_ch = raw.copy().pick(["Trigger"]).get_data().flatten()
    stim_events = []
    for i in range(1, len(trigger_ch)):
        val = int(round(trigger_ch[i]))
        prev = int(round(trigger_ch[i - 1]))
        if val in (131, 132, 140) and val != prev:
            stim_events.append((i, val))

    return raw_eeg, stim_events


def stream_dss(data, sfreq, freq_band, n_components=3):
    """Run ContinuousDSS in streaming mode and return source timeseries."""
    n_channels = data.shape[0]
    cdss = ContinuousDSS(
        n_channels=n_channels,
        sfreq=sfreq,
        bias="bandpass",
        bias_params={"freq_band": freq_band},
        n_components=n_components,
        lambda_0=0.995,
        lambda_1=0.99,
        solve_interval=10,
        warmup_blocks=50,
    )

    n_blocks = data.shape[1] // BLOCK_SIZE
    all_sources = np.zeros((n_components, n_blocks * BLOCK_SIZE))

    for i in range(n_blocks):
        block = data[:, i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
        cdss.process_block(block)
        if cdss.filters is not None:
            mean = cdss._cov_baseline.mean
            sources = cdss.filters @ (block - mean[:, np.newaxis])
            all_sources[:, i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE] = sources

    return all_sources, cdss


def extract_epochs(signal, stim_events, sfreq):
    """Epoch a multichannel signal around stimulus events.

    Returns dict of {condition_label: (n_trials, n_channels, epoch_len)}.
    """
    n_pre = int(abs(EPOCH_TMIN) * sfreq)
    n_post = int(EPOCH_TMAX * sfreq)
    label_map = {131: "HF", 132: "LF", 140: "Pseudo"}

    epochs = {"HF": [], "LF": [], "Pseudo": []}
    for onset, code in stim_events:
        start, end = onset - n_pre, onset + n_post
        if start < 0 or end >= signal.shape[-1]:
            continue
        if signal.ndim == 1:
            ep = signal[start:end].copy()
            ep -= ep[:n_pre].mean()
        else:
            ep = signal[:, start:end].copy()
            ep -= ep[:, :n_pre].mean(axis=-1, keepdims=True)
        epochs[label_map[code]].append(ep)

    return {k: np.array(v) for k, v in epochs.items()}


def erp_analysis(epochs_by_cond, sfreq, label, comp_idx=None):
    """Print N400 amplitude statistics and condition contrasts."""
    n_pre = int(abs(EPOCH_TMIN) * sfreq)
    t = np.arange(epochs_by_cond["HF"].shape[-1]) / sfreq + EPOCH_TMIN
    n400_mask = (t >= N400_TMIN) & (t <= N400_TMAX)

    print(f"\n  {label}:")
    stats = {}
    for cond in ["HF", "LF", "Pseudo"]:
        ep = epochs_by_cond[cond]
        if comp_idx is not None:
            amps = ep[:, comp_idx, :][:, n400_mask].mean(axis=1)
        else:
            amps = ep[:, n400_mask].mean(axis=1)
        stats[cond] = amps
        se = amps.std() / np.sqrt(len(amps))
        print(f"    {cond:6s}: N400 mean = {amps.mean():+.6f} (SE={se:.6f}, n={len(amps)})")

    for c1, c2 in [("LF", "HF"), ("Pseudo", "HF")]:
        pooled_std = np.sqrt((stats[c1].std()**2 + stats[c2].std()**2) / 2)
        d = (stats[c1].mean() - stats[c2].mean()) / pooled_std if pooled_std > 0 else 0
        t_val, p_val = ttest_ind(stats[c1], stats[c2])
        sig = " *" if p_val < 0.05 else ""
        print(f"    {c1} vs {c2}: d={d:+.3f}, t={t_val:+.2f}, p={p_val:.4f}{sig}")

    return stats


def classification_benchmark(data, dss_sources_dict, stim_events, sfreq):
    """Run cross-validated classification on raw, DSS, and combined features."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA

    n_pre = int(abs(EPOCH_TMIN) * sfreq)
    n_post = int(EPOCH_TMAX * sfreq)
    label_map = {131: 0, 132: 1, 140: 2}
    ds_points = 30  # downsample to 30 time points

    X_raw_list, X_dss_list, X_all_list, y_list = [], [], [], []

    for onset, code in stim_events:
        start, end = onset - n_pre, onset + n_post
        if start < 0 or end >= data.shape[1]:
            continue

        # Raw waveform features
        raw_ep = data[:, start:end].copy()
        raw_ep -= raw_ep[:, :n_pre].mean(axis=1, keepdims=True)
        raw_ds = resample(raw_ep, ds_points, axis=1)

        # DSS waveform features (all bands concatenated)
        dss_parts = []
        valid = True
        for src in dss_sources_dict.values():
            if end >= src.shape[1]:
                valid = False
                break
            dss_ep = src[:, start:end].copy()
            dss_ep -= dss_ep[:, :n_pre].mean(axis=1, keepdims=True)
            dss_parts.append(resample(dss_ep, ds_points, axis=1))
        if not valid:
            continue

        dss_ds = np.vstack(dss_parts)

        X_raw_list.append(raw_ds.flatten())
        X_dss_list.append(dss_ds.flatten())
        X_all_list.append(np.concatenate([raw_ds.flatten(), dss_ds.flatten()]))
        y_list.append(label_map[code])

    X_raw = np.array(X_raw_list)
    X_dss = np.array(X_dss_list)
    X_all = np.array(X_all_list)
    y = np.array(y_list)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def run_cv(X, y, label):
        pipe_svm = Pipeline([
            ("s", StandardScaler()),
            ("pca", PCA(n_components=20)),
            ("c", SVC(kernel="rbf", C=1.0)),
        ])
        pipe_lr = Pipeline([
            ("s", StandardScaler()),
            ("pca", PCA(n_components=20)),
            ("c", LogisticRegression(C=1.0, max_iter=1000)),
        ])

        svm_3 = cross_val_score(pipe_svm, X, y, cv=cv, scoring="accuracy")
        lr_3 = cross_val_score(pipe_lr, X, y, cv=cv, scoring="accuracy")

        mask_hl = y < 2
        svm_hl = cross_val_score(pipe_svm, X[mask_hl], y[mask_hl], cv=cv, scoring="accuracy")
        lr_hl = cross_val_score(pipe_lr, X[mask_hl], y[mask_hl], cv=cv, scoring="accuracy")

        print(f"  {label}:")
        print(f"    3-class  SVM: {svm_3.mean():.1%} +/- {svm_3.std():.1%}  |  LR: {lr_3.mean():.1%} +/- {lr_3.std():.1%}")
        print(f"    HF-vs-LF SVM: {svm_hl.mean():.1%} +/- {svm_hl.std():.1%}  |  LR: {lr_hl.mean():.1%} +/- {lr_hl.std():.1%}")

    print(f"\n  Trials: {len(y)}, Features: raw={X_raw.shape[1]}, dss={X_dss.shape[1]}, combined={X_all.shape[1]}")
    run_cv(X_raw, y, f"Raw channels ({X_raw.shape[1]}d)")
    run_cv(X_dss, y, f"Multi-band DSS ({X_dss.shape[1]}d)")
    run_cv(X_all, y, f"Raw + DSS ({X_all.shape[1]}d)")
    print(f"  Chance: 3-class=33.3%, HF-vs-LF=50.0%")


def main():
    print("=" * 60)
    print("ContinuousDSS Validation: Lexical Decision EEG (sub-02)")
    print("=" * 60)

    raw_eeg, stim_events = load_data()
    data = raw_eeg.get_data()
    sfreq = raw_eeg.info["sfreq"]
    n_trials = {131: 0, 132: 0, 140: 0}
    for _, code in stim_events:
        n_trials[code] += 1

    print(f"\nData: {data.shape[0]} channels, {data.shape[1]/sfreq:.0f}s, {sfreq:.0f} Hz")
    print(f"Trials: HF={n_trials[131]}, LF={n_trials[132]}, Pseudo={n_trials[140]}")

    # ---- Part 1: ERP Analysis ----
    print(f"\n{'='*60}")
    print("PART 1: ERP CONDITION EFFECTS")
    print("=" * 60)

    # Raw Cz
    cz_idx = EEG_CHANNELS.index("Cz")
    cz_epochs = extract_epochs(data[cz_idx], stim_events, sfreq)
    erp_analysis(cz_epochs, sfreq, "Raw Cz (classic N400 site)")

    # DSS per band
    dss_sources_dict = {}
    for band_name, freq_band in DSS_BANDS.items():
        print(f"\n  --- DSS: {band_name} ---")
        sources, cdss = stream_dss(data, sfreq, freq_band)
        dss_sources_dict[band_name] = sources
        print(f"  Eigenvalues: {np.round(cdss.eigenvalues, 4)}")

        dss_epochs = extract_epochs(sources, stim_events, sfreq)
        for comp in range(3):
            erp_analysis(dss_epochs, sfreq, f"DSS{comp+1} (ev={cdss.eigenvalues[comp]:.4f})", comp_idx=comp)

    # ---- Part 2: Classification ----
    print(f"\n{'='*60}")
    print("PART 2: SINGLE-TRIAL CLASSIFICATION (5-fold CV)")
    print("=" * 60)

    classification_benchmark(data, dss_sources_dict, stim_events, sfreq)

    print(f"\n{'='*60}")
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
