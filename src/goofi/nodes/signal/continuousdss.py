import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class ContinuousDSS(Node):
    """
    Real-time Denoising Source Separation (DSS) for streaming EEG/MEG data.

    Inputs:
    - data: Multichannel EEG array with metadata including sampling frequency ("sfreq").

    Outputs:
    - denoised: The denoised EEG signal (same channels as input).
    - sources: The extracted DSS source components.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "denoised": DataType.ARRAY,
            "sources": DataType.ARRAY,
        }

    def config_params():
        return {
            "dss": {
                "n_components": IntParam(5, 1, 20, doc="Number of DSS components"),
                "bias_type": StringParam(
                    "line_noise",
                    options=["bandpass", "line_noise"],
                    doc="Bias operator type",
                ),
                "freq_low": FloatParam(8.0, 0.1, 200.0, doc="Bandpass low freq"),
                "freq_high": FloatParam(12.0, 0.1, 200.0, doc="Bandpass high freq"),
                "line_freq": FloatParam(60.0, 40.0, 80.0, doc="Line noise frequency"),
                "lambda_baseline": FloatParam(0.995, 0.9, 0.9999, doc="Data cov forgetting"),
                "lambda_biased": FloatParam(0.99, 0.9, 0.9999, doc="Bias cov forgetting"),
                "solve_interval": IntParam(10, 1, 100, doc="Eigensolve every N blocks"),
                "warmup_blocks": IntParam(30, 5, 500, doc="Warmup before denoising"),
            },
        }

    def setup(self):
        self._dss = None

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        block = data.data
        if block.ndim == 1:
            block = block[np.newaxis, :]

        sfreq = data.meta.get("sfreq", 300.0)
        n_channels = block.shape[0]

        if self._dss is None:
            from mne_denoise.dss.streaming import ContinuousDSS as CDSS

            p = self.params["dss"]
            bias_type = p["bias_type"].value

            if bias_type == "bandpass":
                bias_params = {"freq_band": (p["freq_low"].value, p["freq_high"].value)}
            else:
                bias_params = {"freq": p["line_freq"].value}

            self._dss = CDSS(
                n_channels=n_channels,
                sfreq=sfreq,
                bias=bias_type,
                bias_params=bias_params,
                n_components=p["n_components"].value,
                lambda_0=p["lambda_baseline"].value,
                lambda_1=p["lambda_biased"].value,
                solve_interval=p["solve_interval"].value,
                warmup_blocks=p["warmup_blocks"].value,
            )

        denoised = self._dss.process_block(block)

        # Build metadata for denoised (same channels as input)
        denoised_meta = dict(data.meta)

        # Build metadata for sources (different number of channels)
        sources_meta = {"sfreq": sfreq}
        if self._dss.filters is not None:
            mean = self._dss._cov_baseline.mean
            centered = block - mean[:, np.newaxis]
            sources = self._dss.filters @ centered
            n_comp = sources.shape[0]
            sources_meta["channels"] = {"dim0": [f"DSS{i+1}" for i in range(n_comp)]}
        else:
            n_comp = self.params["dss"]["n_components"].value
            sources = np.zeros((n_comp, block.shape[1]))
            sources_meta["channels"] = {"dim0": [f"DSS{i+1}" for i in range(n_comp)]}

        return {
            "denoised": (denoised, denoised_meta),
            "sources": (sources, sources_meta),
        }
