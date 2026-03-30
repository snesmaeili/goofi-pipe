import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class BrainStateNarrator(Node):
    """
    Real-time brain state interpreter for DSS + PhiID pipelines.

    Takes PhiID information dynamics, DSS eigenvalues, and power band ratios
    to generate a human-readable narrative of the current neural state.
    Optionally formats a prompt for an LLM to produce richer narration.

    Inputs:
    - inf_dyn: Information dynamics from PhiID (storage, copy, transfer, erasure).
    - eigenvalues: DSS eigenvalues (component quality).
    - theta: Theta-band relative power from PowerBandEEG.
    - alpha: Alpha-band relative power from PowerBandEEG.

    Outputs:
    - narrative: Human-readable brain state description.
    - llm_prompt: Formatted prompt for LLM narration (connect to TextGeneration).
    """

    def config_input_slots():
        return {
            "inf_dyn": DataType.ARRAY,
            "eigenvalues": DataType.ARRAY,
            "theta": DataType.ARRAY,
            "alpha": DataType.ARRAY,
        }

    def config_output_slots():
        return {
            "narrative": DataType.STRING,
            "llm_prompt": DataType.STRING,
        }

    def config_params():
        return {
            "narrator": {
                "update_interval": IntParam(5, 1, 30, doc="Update narrative every N calls"),
                "system_context": StringParam(
                    "lexical_decision",
                    options=["lexical_decision", "resting_state", "generic"],
                    doc="Task context for interpretation",
                ),
            },
        }

    def setup(self):
        self._call_count = 0
        self._prev_state = None

    def process(self, inf_dyn, eigenvalues, theta, alpha):
        self._call_count += 1
        interval = self.params["narrator"]["update_interval"].value
        if self._call_count % interval != 0:
            return None

        context = self.params["narrator"]["system_context"].value

        # Extract values safely
        id_vals = inf_dyn.data.flatten() if inf_dyn is not None else None
        ev_vals = eigenvalues.data.flatten() if eigenvalues is not None else None
        th_val = float(theta.data.mean()) if theta is not None else None
        al_val = float(alpha.data.mean()) if alpha is not None else None

        # Build narrative from available data
        parts = []
        state = {}

        # Information dynamics interpretation
        # PhiID inf_dyn output shape: (6, n_channels) in one-vs-others mode
        # Metrics: Storage, Copy, Transfer, Erasure, Downward causation, Upward causation
        if id_vals is not None and id_vals.size >= 6:
            if id_vals.ndim > 1:
                # Average across channels to get one value per metric
                metrics = id_vals.mean(axis=-1) if id_vals.shape[0] == 6 else id_vals.flatten()
            else:
                metrics = id_vals

            labels = ["Storage", "Copy", "Transfer", "Erasure", "Down-causation", "Up-causation"]
            n_metrics = min(len(labels), len(metrics))
            vals = np.abs(metrics[:n_metrics])
            dominant_idx = np.argmax(vals)
            dominant = labels[dominant_idx]
            state["dominant_dynamic"] = dominant
            state["dynamics_vals"] = metrics[:n_metrics]

            descriptions = {
                "Storage": "Maintaining neural patterns across time",
                "Copy": "Broadcasting information across spatial components",
                "Transfer": "Directional information flow between components",
                "Erasure": "Actively clearing or resetting neural patterns",
                "Down-causation": "Whole-to-part causal influence (global constraining local)",
                "Up-causation": "Part-to-whole causal influence (local driving global)",
            }
            parts.append(f"{dominant}: {descriptions.get(dominant, '')}")

            total = vals.sum()
            if total > 0:
                pcts = vals / total * 100
                top3_idx = np.argsort(vals)[::-1][:3]
                summary = " | ".join(f"{labels[i]} {pcts[i]:.0f}%" for i in top3_idx)
                parts.append(f"[{summary}]")

        # DSS eigenvalue interpretation
        if ev_vals is not None:
            top_ev = ev_vals[0]
            state["top_eigenvalue"] = top_ev
            if top_ev > 0.5:
                parts.append(f"Strong spatial pattern detected (eigenvalue={top_ev:.3f})")
            elif top_ev > 0.1:
                parts.append(f"Moderate spatial organization (eigenvalue={top_ev:.3f})")
            else:
                parts.append(f"Weak spatial pattern (eigenvalue={top_ev:.3f})")

        # Theta/Alpha ratio (engagement marker)
        if th_val is not None and al_val is not None and al_val > 0:
            ta_ratio = th_val / al_val
            state["theta_alpha_ratio"] = ta_ratio
            if ta_ratio > 1.5:
                parts.append(f"High cognitive load (theta/alpha={ta_ratio:.2f})")
            elif ta_ratio > 0.8:
                parts.append(f"Active processing (theta/alpha={ta_ratio:.2f})")
            else:
                parts.append(f"Relaxed/idle state (theta/alpha={ta_ratio:.2f})")

            if context == "lexical_decision":
                if ta_ratio > 1.2:
                    parts.append("Likely processing unfamiliar or complex word")
                else:
                    parts.append("Likely processing familiar word or resting between trials")

        # Detect state changes
        if self._prev_state is not None:
            prev_dom = self._prev_state.get("dominant_dynamic")
            curr_dom = state.get("dominant_dynamic")
            if prev_dom and curr_dom and prev_dom != curr_dom:
                parts.append(f"State shift: {prev_dom} -> {curr_dom}")

            prev_ta = self._prev_state.get("theta_alpha_ratio", 0)
            curr_ta = state.get("theta_alpha_ratio", 0)
            if prev_ta > 0 and curr_ta > 0:
                change = (curr_ta - prev_ta) / prev_ta * 100
                if abs(change) > 20:
                    direction = "increased" if change > 0 else "decreased"
                    parts.append(f"Cognitive engagement {direction} by {abs(change):.0f}%")

        self._prev_state = state

        narrative = " | ".join(parts) if parts else "Waiting for data..."

        # Build LLM prompt
        llm_prompt = self._build_llm_prompt(state, context)

        return {
            "narrative": (narrative, {}),
            "llm_prompt": (llm_prompt, {}),
        }

    def _build_llm_prompt(self, state, context):
        lines = [
            "You are a real-time neural state interpreter. Based on the following "
            "brain measurements, provide a brief (1-2 sentences) interpretation. "
            "Be specific and scientifically grounded.",
            "",
        ]

        if context == "lexical_decision":
            lines.append(
                "Context: Subject is performing a French lexical decision task "
                "(classifying high-frequency words, low-frequency words, and pseudowords)."
            )

        lines.append("")
        lines.append("Current measurements:")

        dom = state.get("dominant_dynamic")
        dyn_vals = state.get("dynamics_vals")
        if dom:
            lines.append(f"- Dominant information dynamic: {dom}")
        if dyn_vals is not None:
            dyn_labels = ["Storage", "Copy", "Transfer", "Erasure", "Down-causation", "Up-causation"]
            for i, v in enumerate(dyn_vals):
                if i < len(dyn_labels):
                    lines.append(f"  {dyn_labels[i]}: {v:.4f}")

        ev = state.get("top_eigenvalue")
        if ev is not None:
            lines.append(f"- Top DSS eigenvalue: {ev:.3f} (higher = stronger spatial pattern in theta-alpha band)")

        ta = state.get("theta_alpha_ratio")
        if ta is not None:
            lines.append(f"- Theta/alpha power ratio: {ta:.2f} (>1.0 = high cognitive load, <0.5 = relaxed)")

        lines.append("")
        lines.append("Interpretation:")

        return "\n".join(lines)
