from pathlib import Path
import numpy as np

def plot_run(out, tasks, matrices, comparisons, pairs, point_differences, lower, upper, ranks, accuracy):
    """Plots show actual saved metrics; missing plotting dependency is recorded by caller."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, a, label in zip(axes, (matrices[0], matrices[-1]), ("Before training", "After final task")):
        im = ax.imshow(a, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(tasks)), tasks, rotation=35, ha="right")
        ax.set_yticks(range(len(tasks)), tasks)
        ax.set_title(label)
        for i in range(len(tasks)):
            for j in range(len(tasks)):
                ax.text(j, i, f"{a[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if a[i,j] < .55 else "black")
    fig.colorbar(im, ax=axes, label="Mean squared cosine of principal angles", shrink=.8)
    fig.savefig(out / "affinity_heatmaps.png", dpi=180)
    fig.savefig(out / "affinity_heatmaps.svg")
    plt.close(fig)
    strength = np.minimum(np.abs(point_differences[0]), np.abs(point_differences[-1]))
    raw = point_differences[0] * point_differences[-1] < 0
    selected = sorted(range(len(comparisons)), key=lambda k: (raw[k], strength[k]), reverse=True)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, k in zip(axes.flat, selected):
        p, q = comparisons[k]
        name = lambda pair: tasks[pair[0]] + "/" + tasks[pair[1]]
        values = point_differences[:, k]
        ax.axhline(0, color="grey", linewidth=1)
        ax.plot(range(len(values)), values, marker="o")
        for endpoint, ci_index in ((0, 0), (len(values)-1, 1)):
            ax.vlines(endpoint, lower[ci_index, k], upper[ci_index, k], color="tab:red", linewidth=2)
        ax.set_xticks(range(len(values)))
        ax.set_xlabel("Completed tasks (0 = baseline)")
        ax.set_ylabel("Affinity difference")
        ax.set_title(name(pairs[p]) + " minus\n" + name(pairs[q]), fontsize=9)
    fig.suptitle("Selected trajectories; endpoint intervals adjusted within this run")
    fig.savefig(out / "affinity_differences.png", dpi=180)
    fig.savefig(out / "affinity_differences.svg")
    plt.close(fig)
    if accuracy:
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for task in tasks:
            values = [row[task] for row in accuracy]
            ax.plot(range(len(values)), values, marker="o", label=task)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Completed tasks (0 = baseline)")
        ax.set_ylabel("Accuracy on recorded evaluation split")
        ax.legend()
        fig.savefig(out / "accuracy_trajectory.png", dpi=180)
        fig.savefig(out / "accuracy_trajectory.svg")
        plt.close(fig)

