"""Full-project RoboTwin progress chart -- every candidate in this project's history
that reached a full-50-task Clean scan (the actual canonical/promotion metric), from
exp0014 through exp0017. exp0013 never reached this measurement (it collapsed on the
curated panel before a full-50-task scan was run) and is shown as a separate labeled
prologue rather than a fabricated point on the main quantitative series.

Data pulled directly from research/STATE.md and research/progress/PROGRESS_0013-0017*.md.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/workspace/FastWAM/research/progress/summary_report"

labels = [
    "exp0013\n(full backbone, unprotected)",
    "exp0014\nstep 2600 (frozen backbone)",
    "exp0014\nstep 4600 (frozen backbone)",
    "exp0015\n(frozen, RoboTwin-heavy 1:3)",
    "exp0016\nphase 1, step 1000 (low-LR)",
    "exp0017\nstep ~19,740 (full, protected)",
]
x = list(range(len(labels)))

# Full-50-task Clean mean -- None where no full-50-task scan was ever run
full50 = [None, 12.6, 11.6, 11.6, 9.6, 23.2]
n_trials = [None, 10, 5, 5, 5, 5]

annotations = [
    "curated 2-task panel only\n(click_alarmclock 80%,\nturn_switch 60% -> 0%)\ncollapsed before a full-50-\ntask scan could run",
    "first full-50-task scan\nof the project",
    "+2000 steps, same recipe\n-- essentially no movement",
    "3x RoboTwin gradient share\n-- still no movement\n(REJECT)",
    "backbone given LR=3e-6\nplasticity -- still flat\n(undertrained, <1% of\nan epoch)",
    "full backbone + Long-\nprotected mixing, ~19,740\nsteps -- nearly DOUBLE\nany prior candidate",
]

fig, ax = plt.subplots(figsize=(14, 7.5))

ax.axhline(90, color="#c0392b", linestyle="--", linewidth=1.3, zorder=1)
ax.text(0.99, 91.3, "90% target", color="#c0392b", fontsize=10, ha="right", transform=ax.get_yaxis_transform())

# exp0013 prologue -- shown as a faded marker with no y-value on the main metric,
# annotated separately, so it is never implied to be numerically comparable.
ax.scatter([0], [3], marker="x", s=90, color="#999999", zorder=3)
ax.annotate("no full-50-task\nmeasurement", (0, 3), textcoords="offset points", xytext=(0, 14),
            ha="center", fontsize=9, color="#777777", style="italic")

xr = [xi for xi, v in zip(x, full50) if v is not None]
yr = [v for v in full50 if v is not None]
ax.plot(xr, yr, color="#2a78d6", linewidth=2.3, marker="o", markersize=11,
        markerfacecolor="#2a78d6", markeredgecolor="white", markeredgewidth=1.8, zorder=3)

# highlight the exp0017 point distinctly (the decisive result)
ax.scatter([xr[-1]], [yr[-1]], s=280, facecolor="none", edgecolor="#eb6834", linewidth=2.2, zorder=4)

for xi, yi, n in zip(xr, yr, [v for v in n_trials if v is not None]):
    ax.annotate(f"{yi:.1f}%\n(n={n})", (xi, yi), textcoords="offset points", xytext=(0, 16),
                ha="center", fontsize=10.5, fontweight="bold", color="#1a5490")

for xi, note in zip(x, annotations):
    ax.annotate(note, (xi, 0), textcoords="offset points", xytext=(0, -95),
                ha="center", fontsize=7.6, color="#444444", linespacing=1.3)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.tick_params(axis="x", pad=14)
ax.set_xlim(-0.5, len(x) - 0.4)
ax.set_ylim(0, 100)
ax.set_ylabel("RoboTwin full-50-task Clean success rate (%)", fontsize=11)
ax.set_title("Full project history -- RoboTwin canonical metric, exp0013 -> exp0017", fontsize=14, fontweight="bold", pad=16)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#eeeeee", linewidth=0.8, zorder=0)

fig.suptitle("FastWAM -- autoresearch/robotwin-multiembodiment-v1", fontsize=10, color="#888888", y=0.99)
fig.tight_layout(rect=[0, 0.09, 1, 0.97])
fig.savefig(f"{OUT}/2_full_project_robotwin_progress.png", dpi=180, facecolor="white")
print(f"Saved {OUT}/2_full_project_robotwin_progress.png")
