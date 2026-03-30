import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


class ConditionERP(Node):
    """
    Condition-aware event-related potential (ERP) node for real-time BCI.

    Monitors a trigger channel for stimulus codes and epochs an input signal
    (e.g., DSS sources) around each event. Maintains separate running averages
    per condition, enabling real-time comparison of neural responses across
    experimental conditions.

    Designed for lexical decision paradigms but works with any trigger-coded task.

    Inputs:
    - signal: Continuous multichannel data (e.g., DSS sources) with "sfreq" metadata.
    - trigger: Single-channel trigger/event signal carrying integer stimulus codes.

    Outputs:
    - erp_a: Running average ERP for condition A.
    - erp_b: Running average ERP for condition B.
    - erp_c: Running average ERP for condition C.
    """

    def config_input_slots():
        return {"signal": DataType.ARRAY, "trigger": DataType.ARRAY}

    def config_output_slots():
        return {
            "erp_a": DataType.ARRAY,
            "erp_b": DataType.ARRAY,
            "erp_c": DataType.ARRAY,
        }

    def config_params():
        return {
            "conditions": {
                "code_a": IntParam(131, 0, 999, doc="Trigger code for condition A"),
                "code_b": IntParam(132, 0, 999, doc="Trigger code for condition B"),
                "code_c": IntParam(140, 0, 999, doc="Trigger code for condition C"),
                "label_a": StringParam("HF", doc="Label for condition A"),
                "label_b": StringParam("LF", doc="Label for condition B"),
                "label_c": StringParam("Pseudo", doc="Label for condition C"),
            },
            "epoch": {
                "duration": FloatParam(0.8, 0.1, 3.0, doc="Epoch duration post-trigger (s)"),
                "baseline": FloatParam(0.0, 0.0, 1.0, doc="Baseline duration pre-trigger (s)"),
                "reset": BoolParam(False, trigger=True, doc="Reset all averages"),
            },
        }

    def setup(self):
        self._buffer = []
        self._trigger_buf = []
        self._collecting = {}  # code -> list of signal chunks
        self._erps = {}        # code -> (sum_array, count)
        self._sfreq = None

    def process(self, signal, trigger):
        if signal is None:
            return None

        if self.params["epoch"]["reset"].value:
            self._erps.clear()
            self._collecting.clear()
            self._buffer.clear()
            self._trigger_buf.clear()

        sfreq = signal.meta.get("sfreq", 300.0)
        self._sfreq = sfreq
        sig = signal.data
        if sig.ndim == 1:
            sig = sig[np.newaxis, :]

        codes = [
            self.params["conditions"]["code_a"].value,
            self.params["conditions"]["code_b"].value,
            self.params["conditions"]["code_c"].value,
        ]

        # Extract trigger values from the trigger channel
        trig = None
        if trigger is not None and trigger.data is not None:
            trig = trigger.data
            if trig.ndim > 1:
                trig = trig.flatten()

        # Detect new trigger onsets in this chunk
        if trig is not None:
            for i in range(len(trig)):
                val = int(round(trig[i]))
                if val in codes:
                    # Check it's an onset (not continuation of same trigger)
                    prev_val = 0
                    if i > 0:
                        prev_val = int(round(trig[i - 1]))
                    elif len(self._trigger_buf) > 0:
                        prev_val = self._trigger_buf[-1]
                    if val != prev_val:
                        # New trigger onset — start collecting
                        n_baseline = int(self.params["epoch"]["baseline"].value * sfreq)
                        # Grab baseline from buffer if available
                        if n_baseline > 0 and len(self._buffer) > 0:
                            past = np.concatenate(self._buffer, axis=1)
                            if past.shape[1] >= n_baseline:
                                baseline_chunk = past[:, -n_baseline:]
                            else:
                                baseline_chunk = np.zeros((sig.shape[0], n_baseline))
                            self._collecting[len(self._collecting)] = {
                                "code": val,
                                "chunks": [baseline_chunk, sig[:, i:i+1]],
                            }
                        else:
                            self._collecting[len(self._collecting)] = {
                                "code": val,
                                "chunks": [sig[:, i:i+1]],
                            }

            # Store last trigger value for onset detection
            self._trigger_buf.append(int(round(trig[-1])) if len(trig) > 0 else 0)
            if len(self._trigger_buf) > 10:
                self._trigger_buf = self._trigger_buf[-5:]

        # Append current signal to all active collections
        finished = []
        n_epoch = int(self.params["epoch"]["duration"].value * sfreq)
        n_baseline = int(self.params["epoch"]["baseline"].value * sfreq)
        target_len = n_epoch + n_baseline

        for key, col in self._collecting.items():
            col["chunks"].append(sig)
            total = sum(c.shape[1] for c in col["chunks"])
            if total >= target_len:
                # Epoch complete
                epoch = np.concatenate(col["chunks"], axis=1)[:, :target_len]

                # Baseline correction
                if n_baseline > 0:
                    bl = epoch[:, :n_baseline].mean(axis=1, keepdims=True)
                    epoch = epoch - bl

                code = col["code"]
                if code not in self._erps:
                    self._erps[code] = (epoch, 1)
                else:
                    prev_sum, prev_n = self._erps[code]
                    if prev_sum.shape == epoch.shape:
                        self._erps[code] = (prev_sum + epoch, prev_n + 1)
                    else:
                        self._erps[code] = (epoch, 1)

                finished.append(key)

        for key in finished:
            del self._collecting[key]

        # Keep signal buffer for baseline (max 1s)
        self._buffer.append(sig)
        max_buf_chunks = max(1, int(sfreq / max(sig.shape[1], 1)))
        if len(self._buffer) > max_buf_chunks:
            self._buffer = self._buffer[-max_buf_chunks:]

        # Build outputs
        results = {}
        for slot, code_key, label_key in [
            ("erp_a", "code_a", "label_a"),
            ("erp_b", "code_b", "label_b"),
            ("erp_c", "code_c", "label_c"),
        ]:
            code = self.params["conditions"][code_key].value
            label = self.params["conditions"][label_key].value
            if code in self._erps:
                erp_sum, erp_n = self._erps[code]
                erp = erp_sum / erp_n
                meta = dict(signal.meta)
                meta["condition"] = label
                meta["n_trials"] = erp_n
                meta["trigger_code"] = code
                results[slot] = (erp, meta)
            else:
                results[slot] = None

        return results
