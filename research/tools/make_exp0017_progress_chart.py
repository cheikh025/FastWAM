"""Generate a static progress chart for exp0017 (RoboTwin full-backbone +
LIBERO-Long-protected candidate), matching the format of the sibling LIBERO-only
project's own research/tools/make_summary_charts.py.

Data pulled directly from research/progress/PROGRESS_0017_backbone_low_lr_robotwin_heavy_longrun.md.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/workspace/FastWAM/research/progress/summary_report"

labels = [
    "step 5,240\ncurated panel, n=5/task",
    "step ~10,740\ncurated panel, n=5/task",
    "step ~14,740\ncurated panel, n=5/task",
    "step ~19,740\ncurated panel, n=5/task",
    "step ~19,740\nFULL 50-TASK, n=5/task",
]
x = list(range(len(labels)))

# Curated 4-task panel mean (click_alarmclock, turn_switch, press_stapler, open_laptop)
curated = [20.0, 30.0, 35.0, 40.0, None]
# Full 50-task Clean mean (only measured once so far, at the last checkpoint)
full50 = [None, None, None, None, 23.2]
# LIBERO retention sentinels at the same checkpoints
spatial = [94.0, 96.0, 96.0, 96.0, 96.0]
long_ = [90.0, 94.0, 96.0, 96.0, 96.0]

annotations = [
    "cont1, phase-1 checkpoint\nturn_switch still 0%",
    "cont2, after disk-full\nrecovery -- 3/4 tasks\nrecovering",
    "cont3, seamless\ndirectory-resume\nturn_switch finally >0%",
    "cont3 completed its full\n9000-step ceiling cleanly",
    "decisive: 23/50 tasks nonzero\n(was ~10-11), 3 bimanual\nbreakthroughs (was 0)",
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), height_ratios=[1.3, 1])

# ================= Panel 1: RoboTwin (primary target) =================
ax1.axhline(90, color="#c0392b", linestyle="--", linewidth=1.3, zorder=1)
ax1.text(0.99, 91.5, "90% target", color="#c0392b", fontsize=10, ha="right", transform=ax1.get_yaxis_transform())
ax1.axhline(12, color="#888888", linestyle=":", linewidth=1.2, zorder=1)
ax1.text(0.01, 13.5, "exp0014 frozen-backbone ceiling (~12%)", color="#888888", fontsize=9,
         transform=ax1.get_yaxis_transform())

xc = [xi for xi, v in zip(x, curated) if v is not None]
yc = [v for v in curated if v is not None]
ax1.plot(xc, yc, color="#2a78d6", linewidth=2.2, marker="o", markersize=9,
         markerfacecolor="#2a78d6", markeredgecolor="white", markeredgewidth=1.5, zorder=3,
         label="Curated 4-task panel (mean)")

xf = [xi for xi, v in zip(x, full50) if v is not None]
yf = [v for v in full50 if v is not None]
ax1.scatter(xf, yf, s=170, color="#eb6834", edgecolor="white", linewidth=1.8, zorder=4,
            label="Full 50-task scan (mean)")

for xi, yi in zip(xc, yc):
    ax1.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points", xytext=(0, 14),
                 ha="center", fontsize=10, fontweight="bold", color="#1a5490")
for xi, yi in zip(xf, yf):
    ax1.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points", xytext=(0, 16),
                 ha="center", fontsize=10, fontweight="bold", color="#b8471f")

for xi, note in zip(x, annotations):
    ax1.annotate(note, (xi, 0), textcoords="offset points", xytext=(0, -58),
                 ha="center", fontsize=7.8, color="#444444", linespacing=1.3)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8.5)
ax1.set_xlim(-0.5, len(x) - 0.3)
ax1.set_ylim(0, 105)
ax1.set_ylabel("RoboTwin Clean success rate (%)", fontsize=10.5)
ax1.set_title("exp0017 -- RoboTwin progress toward the >=90% full-50-task target", fontsize=13, fontweight="bold", pad=14)
ax1.legend(loc="upper left", fontsize=9.5, frameon=False)
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", color="#eeeeee", linewidth=0.8, zorder=0)

# ================= Panel 2: LIBERO retention =================
ax2.axhline(90, color="#c0392b", linestyle="--", linewidth=1.3, zorder=1)
ax2.text(0.01, 90.6, "90% floor", color="#c0392b", fontsize=10, transform=ax2.get_yaxis_transform())

ax2.plot(x, spatial, color="#1baf7a", linewidth=2.2, marker="o", markersize=8,
         markerfacecolor="#1baf7a", markeredgecolor="white", markeredgewidth=1.3, zorder=3,
         label="LIBERO-Spatial")
ax2.plot(x, long_, color="#4a3aa7", linewidth=2.2, marker="o", markersize=8,
         markerfacecolor="#4a3aa7", markeredgecolor="white", markeredgewidth=1.3, zorder=3,
         label="LIBERO-Long")

for xi, yi in zip(x, spatial):
    ax2.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points", xytext=(0, 11),
                 ha="center", fontsize=9, fontweight="bold", color="#0f7a4f")
for xi, yi in zip(x, long_):
    ax2.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points", xytext=(0, -16),
                 ha="center", fontsize=9, fontweight="bold", color="#332570")

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8.5)
ax2.set_xlim(-0.5, len(x) - 0.3)
ax2.set_ylim(85, 100)
ax2.set_ylabel("LIBERO success rate (%)", fontsize=10.5)
ax2.set_title("LIBERO retention -- never at risk across the entire candidate", fontsize=12, fontweight="bold", pad=12)
ax2.legend(loc="lower right", fontsize=9.5, frameon=False)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="y", color="#eeeeee", linewidth=0.8, zorder=0)

fig.suptitle("FastWAM -- autoresearch/robotwin-multiembodiment-v1 -- exp0017", fontsize=10, color="#888888", y=0.995)
fig.tight_layout(rect=[0, 0.02, 1, 0.98])
fig.savefig(f"{OUT}/1_robotwin_progress.png", dpi=180, facecolor="white")
print(f"Saved {OUT}/1_robotwin_progress.png")
