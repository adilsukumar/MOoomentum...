"""Extend the temporal-window sweep to 4s and 5s without overwriting prior results."""
import json
import time
from pathlib import Path

import features as F
from experiment_window_lengths import load_data, evaluate

ROOT = Path(__file__).resolve().parent.parent
RESULT_PATH = ROOT / "models" / "window_length_results.json"


def main():
    df = load_data()
    channels = F.channel_list(use_back=False)
    results = json.loads(RESULT_PATH.read_text()) if RESULT_PATH.exists() else {}
    for seconds in (4.0, 5.0):
        F.WINDOW = int(seconds * F.SAMPLE_RATE_HZ)
        F.STRIDE = F.SAMPLE_RATE_HZ
        t0 = time.time()
        print(f"[window] {seconds:.1f}s / stride 1.0s", flush=True)
        data = F.build_windows(df, channels)
        X = F.extract_window_features(data["windows"])
        result = evaluate(X, data["coarse"], data["dog_id"])
        result.update({
            "window_seconds": seconds,
            "stride_seconds": 1.0,
            "n_windows": int(len(X)),
            "elapsed_seconds": time.time() - t0,
        })
        results[f"window_{seconds:g}s"] = result
        print(json.dumps(result, indent=2), flush=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved {RESULT_PATH}")


if __name__ == "__main__":
    main()
