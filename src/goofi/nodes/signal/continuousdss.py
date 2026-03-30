import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam


class ContinuousDSS(Node):
    """
    Real-time Denoising Source Separation (DSS) for streaming EEG/MEG data.

    Uses mne-denoise's ContinuousDSS with a bandpass bias to extract spatial
    components that maximize power in a target frequency band. The extracted
    sources serve as real-time neural biomarkers (e.g., alpha, theta rhythms).

    Inputs:
    - data: Multichannel EEG array with metadata including "sfreq".

    Outputs:
    - sources: DSS source components maximizing target-band power.
    - eigenvalues: Current DSS eigenvalues (bias/baseline power ratio per component).
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "sources": DataType.ARRAY,
            "eigenvalues": DataType.ARRAY,
        }

    def config_params():
        return {
            "dss": {
                "n_components": IntParam(3, 1, 20, doc="Number of DSS components to extract"),
                "freq_low": FloatParam(8.0, 0.1, 200.0, doc="Bandpass low frequency (Hz)"),
                "freq_high": FloatParam(12.0, 0.1, 200.0, doc="Bandpass high frequency (Hz)"),
                "lambda_baseline": FloatParam(0.995, 0.9, 0.9999, doc="Baseline covariance forgetting factor"),
                "lambda_biased": FloatParam(0.99, 0.9, 0.9999, doc="Biased covariance forgetting factor"),
                "solve_interval": IntParam(10, 1, 100, doc="Eigensolve every N blocks"),
                "warmup_blocks": IntParam(50, 5, 500, doc="Blocks before first eigensolve"),
            },
        }

    def setup(self):
        self._dss = None
        self._last_params = None

    def _get_param_snapshot(self):
        p = self.params["dss"]
        return (
            p["n_components"].value,
            p["freq_low"].value,
            p["freq_high"].value,
            p["lambda_baseline"].value,
            p["lambda_biased"].value,
            p["solve_interval"].value,
            p["warmup_blocks"].value,
        )

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        block = data.data
        if block.ndim == 1:
            block = block[np.newaxis, :]

        sfreq = data.meta.get("sfreq", 300.0)
        n_channels = block.shape[0]

        # Rebuild DSS if parameters changed or first run
        current_params = self._get_param_snapshot()
        if self._dss is None or current_params != self._last_params:
            from mne_denoise.dss.streaming import ContinuousDSS as CDSS

            p = self.params["dss"]
            self._dss = CDSS(
                n_channels=n_channels,
                sfreq=sfreq,
                bias="bandpass",
                bias_params={
                    "freq_band": (p["freq_low"].value, p["freq_high"].value),
                },
                n_components=p["n_components"].value,
                lambda_0=p["lambda_baseline"].value,
                lambda_1=p["lambda_biased"].value,
                solve_interval=p["solve_interval"].value,
                warmup_blocks=p["warmup_blocks"].value,
            )
            self._last_params = current_params

        # Update covariances and trigger periodic eigensolve
        self._dss.process_block(block)

        n_comp = self.params["dss"]["n_components"].value
        filters = self._dss.filters  # (n_components, n_channels) or None

        if filters is not None:
            mean = self._dss._cov_baseline.mean
            centered = block - mean[:, np.newaxis]
            sources = filters @ centered  # (n_comp, n_samples)
            eigenvalues = self._dss.eigenvalues
        else:
            # Still in warmup
            sources = np.zeros((n_comp, block.shape[1]))
            eigenvalues = np.zeros(n_comp)

        sources_meta = {
            "sfreq": sfreq,
            "channels": {"dim0": [f"DSS{i+1}" for i in range(n_comp)]},
        }
        eigenvalues_meta = {
            "channels": {"dim0": [f"DSS{i+1}" for i in range(n_comp)]},
        }

        return {
            "sources": (sources, sources_meta),
            "eigenvalues": (eigenvalues, eigenvalues_meta),
        }
