import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "swing_graphs"


def save_keyframes_plot(landmarks_per_frame, key_frames, timestamp, output_dir=OUTPUT_DIR, swing_start_idx=None, peaks=None, troughs=None):
    wrist_y = [frame[15].y for frame in landmarks_per_frame]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(wrist_y, color="steelblue", linewidth=1, label="wrist-y (left wrist, landmark 15)")

    if swing_start_idx is not None:
        ax.axvline(swing_start_idx, color="gray", linestyle=":", label="swing_start_idx (raw)")

    # All candidate peaks/troughs find_peaks considered, not just the ones
    # find_key_frames picked -- lets you see whether a spurious extra peak
    # was close to changing the outcome.
    if peaks is not None and len(peaks) > 0:
        ax.scatter(peaks, [wrist_y[i] for i in peaks], marker="^", color="black",
                   zorder=5, label="candidate peaks (find_peaks)")
    if troughs is not None and len(troughs) > 0:
        ax.scatter(troughs, [wrist_y[i] for i in troughs], marker="v", color="dimgray",
                   zorder=5, label="candidate troughs (find_peaks)")

    markers = [
        ("address", key_frames["address"], "green"),
        ("top", key_frames["top"], "orange"),
        ("impact", key_frames["impact"], "red"),
        ("finish", key_frames["finish"], "purple"),
    ]
    for label, idx, color in markers:
        ax.axvline(idx, color=color, linestyle="--", linewidth=1.5)
        ax.annotate(label, (idx, wrist_y[idx]), textcoords="offset points",
                    xytext=(4, 8), color=color, fontweight="bold")

    ax.invert_yaxis()  # lower y = higher hand position
    ax.set_xlabel("frame index")
    ax.set_ylabel("wrist y (normalized, inverted)")
    ax.set_title(f"find_key_frames: swing_{timestamp}")
    ax.legend(loc="upper right", fontsize=8)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"swing_{timestamp}_keyframes.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
