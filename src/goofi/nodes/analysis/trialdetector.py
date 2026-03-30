import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam, StringParam


class TrialDetector(Node):
    """
    Detects stimulus events from a trigger channel and outputs the current
    trial condition as a string. Tracks trial counts per condition.

    Inputs:
    - trigger: Single-channel trigger signal with integer stimulus codes.

    Outputs:
    - condition: Current trial condition label (e.g., "LF word #42").
    """

    def config_input_slots():
        return {"trigger": DataType.ARRAY}

    def config_output_slots():
        return {"condition": DataType.STRING}

    def config_params():
        return {
            "codes": {
                "code_a": IntParam(131, 0, 999, doc="Trigger code A"),
                "label_a": StringParam("HF word", doc="Label A"),
                "code_b": IntParam(132, 0, 999, doc="Trigger code B"),
                "label_b": StringParam("LF word", doc="Label B"),
                "code_c": IntParam(140, 0, 999, doc="Trigger code C"),
                "label_c": StringParam("Pseudoword", doc="Label C"),
            },
        }

    def setup(self):
        self._prev_val = 0
        self._counts = {}
        self._last_label = ""

    def process(self, trigger):
        if trigger is None or trigger.data is None:
            return None

        trig = trigger.data.flatten()
        code_map = {
            self.params["codes"]["code_a"].value: self.params["codes"]["label_a"].value,
            self.params["codes"]["code_b"].value: self.params["codes"]["label_b"].value,
            self.params["codes"]["code_c"].value: self.params["codes"]["label_c"].value,
        }

        for val in trig:
            v = int(round(val))
            if v in code_map and v != self._prev_val:
                label = code_map[v]
                self._counts[label] = self._counts.get(label, 0) + 1
                self._last_label = f"{label} #{self._counts[label]}"
            self._prev_val = v

        if self._last_label:
            total = sum(self._counts.values())
            summary = " | ".join(f"{k}: {v}" for k, v in self._counts.items())
            output = f"{self._last_label}  [{summary} | total: {total}]"
            return {"condition": (output, {})}

        return None
