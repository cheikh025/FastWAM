import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.family"] = "DejaVu Sans"

OUT = "/workspace/fastwam_libero90/research/progress/summary_report"

# ---- Verified data, pulled directly from research/EXPERIMENTS.jsonl and progress reports ----
labels = [
    "baseline\n(released ckpt)\ncanonical, 50 trials/task",
    "exp0009\n8k steps\npanel, 5 trials/task",
    "exp0010\n12k steps\npanel, 5 trials/task",
    "exp0013\n+8k steps (resume-fixed)\npanel, 5 trials/task",
    "exp0013\nconfirmation\n130 tasks, 15 trials/task",
    "exp0013\nCANONICAL\n130 tasks, 50 trials/task",
    "exp0014\n+6k steps (dir-resume)\nconfirmation, 15 trials/task",
    "exp0014\nCANONICAL\n130 tasks, 50 trials/task",
    "exp0017\n(via 2 rejected detours)\ntargeted data eng., confirm.",
    "exp0017\nCANONICAL\n130 tasks, 50 trials/task",
    "exp0018\nbroadened targeting\n(7→11 tasks), confirm.",
    "exp0018\nCANONICAL\n130 tasks, 50 trials/task",
    "exp0019\nSpatial-targeted\noversampling, confirm.",
    "exp0019\nCANONICAL\n130 tasks, 50 trials/task",
]
x = list(range(len(labels)))

libero90 = [15.40, 72.00, 76.00, 80.00, 88.59, 88.49, 91.19, 91.09, 91.63, 92.60, 94.74, 95.49, 95.19, 95.13]
spatial  = [96.60, 70.00, 85.00, 95.00, 96.00, 94.40, 97.33, 97.20, 94.00, 96.80, 94.67, 94.80, 96.00, 97.00]
object_  = [99.40, 100.00, 100.00, 90.00, 96.67, 98.00, 100.00, 100.00, 100.00, 99.60, 99.33, 99.60, 99.33, 99.60]
goal     = [96.80, 80.00, 85.00, 95.00, 98.67, 97.40, 97.33, 96.60, 100.00, 97.20, 100.00, 97.40, 97.33, 97.20]
long_    = [94.60, 50.00, 100.00, 100.00, 96.67, 93.00, 93.33, 93.40, 96.67, 96.00, 95.33, 95.20, 97.33, 98.00]

annotations = [
    "starting point:\nreleased FastWAM\nLIBERO checkpoint",
    "+ LIBERO-90 data,\ngoal5x/long5x mix,\nfresh 8k-step run",
    "extend to 12k steps\n(fresh run)",
    "extend +8k steps\nfrom exp0010 weights\n(after fixing 2 resume\ncorruption bugs)",
    "full 90-task LIBERO-90\ncoverage confirms\nthe panel signal",
    "promotion-grade evidence\n→ PROMOTED as new\nmain-line checkpoint",
    "extend +6k steps via\ndirectory-resume\n(optimizer state preserved)",
    "crosses 90% on ALL FIVE\nsuites for the first time\n→ PROMOTED, replaces exp0013",
    "targeted oversampling of\nLIBERO-90's weakest tasks\n(after 2 generic-training\nvariants plateaued)",
    "ALL FOUR original suites\nclear 95% simultaneously\n→ PROMOTED, replaces exp0014",
    "broadened subset 7→11 tasks\n(added newly-emerged\nweak tasks from exp0017)",
    "LIBERO-90 clears 95% for the\nfirst time — 4/5 suites met\n→ PROMOTED, replaces exp0017",
    "targeted oversampling of\nSpatial's 2 weakest tasks\n(task4/task5)",
    "ALL FIVE SUITES clear 95%\nsimultaneously for the first time\n→ PROMOTED, replaces exp0018",
]

# ================= Figure 1: LIBERO-90 (primary target) =================
fig, ax = plt.subplots(figsize=(24, 7.5))

ax.axhline(90, color="#c0392b", linestyle="--", linewidth=1.3, zorder=1)
ax.text(0.02, 90.9, "90% target (met)", color="#c0392b", fontsize=10, transform=ax.get_yaxis_transform())
ax.axhline(95, color="#b8860b", linestyle="--", linewidth=1.3, zorder=1)
ax.text(0.02, 95.9, "95% goal (current)", color="#b8860b", fontsize=10, transform=ax.get_yaxis_transform())

ax.plot(x, libero90, color="#2e8b57", linewidth=2.2, zorder=2)
ax.scatter(x, libero90, s=110, color="#2e8b57", edgecolor="white", linewidth=1.5, zorder=3)

for xi, yi, val in zip(x, libero90, libero90):
    ax.annotate(f"{val:.2f}%", (xi, yi), textcoords="offset points", xytext=(0, 14),
                ha="center", fontsize=10, fontweight="bold", color="#1b5e3a")

for xi, note in zip(x, annotations):
    yoff = -70 if xi == len(x) - 1 else -55
    ax.annotate(note, (xi, libero90[xi]), textcoords="offset points", xytext=(0, yoff),
                ha="center", fontsize=8.2, color="#444444", linespacing=1.3)

# next step marker (placed above-right of the last point, clear of its own label)
ax.annotate("current best checkpoint. ALL FIVE suites now\nclear the 95% goal simultaneously —\nfirst time in the project's history",
            xy=(x[-1], libero90[-1]), xytext=(x[-1] - 1.3, 100),
            fontsize=9, color="#555555", style="italic", ha="center",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=1))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.3)
ax.set_xlim(-0.5, len(x) - 0.15)
ax.set_ylabel("LIBERO-90 success rate (%)", fontsize=12)
ax.set_ylim(0, 108)
ax.set_title("FastWAM on LIBERO-90: primary target progress", fontsize=15, fontweight="bold", pad=14)
ax.grid(axis="y", alpha=0.3)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(f"{OUT}/1_libero90_progress.png", dpi=160)
plt.close(fig)

# ================= Figure 2: retention suites =================
fig, ax = plt.subplots(figsize=(24, 6.5))

ax.axhline(90, color="#c0392b", linestyle="--", linewidth=1.3, zorder=1)
ax.text(0.02, 90.9, "90% retention floor", color="#c0392b", fontsize=10, transform=ax.get_yaxis_transform())
ax.axhline(95, color="#b8860b", linestyle="--", linewidth=1.3, zorder=1)
ax.text(0.02, 95.9, "95% goal (current)", color="#b8860b", fontsize=10, transform=ax.get_yaxis_transform())

series = [
    ("LIBERO-Spatial", spatial, "#3b6fd6"),
    ("LIBERO-Object", object_, "#e08c1e"),
    ("LIBERO-Goal", goal, "#8e44ad"),
    ("LIBERO-Long/10", long_, "#17a398"),
]
for name, vals, color in series:
    ax.plot(x, vals, marker="o", linewidth=2, markersize=6, color=color, label=name)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.3)
ax.set_xlim(-0.5, len(x) - 0.5)
ax.set_ylabel("Success rate (%)", fontsize=12)
ax.set_ylim(40, 108)
ax.set_title("Retention check: original suites dipped mid-training,\nthen recovered — all four now clear 95% simultaneously", fontsize=14, fontweight="bold", pad=14)
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(f"{OUT}/2_retention_suites.png", dpi=160)
plt.close(fig)

print("done")
