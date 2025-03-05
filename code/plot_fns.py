import numpy as np
import matplotlib.pyplot as plt
from metrics import MACER, BPCER


def plot_det_curve(y_true: np.ndarray, y_pred: np.ndarray):
    ypred_scores = np.sort(np.unique(y_pred))
    bpcer_vals = []
    macer_vals = []

    for threshold in ypred_scores:
        bpcer = BPCER(y_true, y_pred, threshold)
        macer = MACER(y_true, y_pred, threshold)
        bpcer_vals.append(bpcer)
        macer_vals.append(macer)

    plt.figure(figsize=(8, 6))
    plt.plot(
        bpcer_vals, macer_vals, marker="o", linestyle="-", color="b", label="DET Curve"
    )
    plt.xlabel("BPCER")
    plt.ylabel("MACER")
    plt.title("Detection Error Tradeoff (DET) Curve")
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()


def plot_macer_timeseries(macer_history, epochs):
    """
    Plot a time-series of MACER@BPCER=1% over training epochs.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(
        epochs,
        macer_history,
        marker="s",
        linestyle="-",
        color="g",
        label="MACER@BPCER=1%",
    )
    plt.xlabel("Epoch")
    plt.ylabel("MACER@BPCER=1%")
    plt.title("MACER@BPCER=1% Over Training Epochs")
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()


def plot_det_3d_evolution(y_true_list: list, y_pred_list: list, epoch_numbers: list):
    """
    Plot a 3D plot of the DET curve evolution over epochs.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, (y_true, y_pred, epoch) in enumerate(
        zip(y_true_list, y_pred_list, epoch_numbers)
    ):
        thresholds = np.sort(np.unique(y_pred))
        bpcer_vals = []
        macer_vals = []

        for threshold in thresholds:
            bpcer = BPCER(y_true, y_pred, threshold)
            macer = MACER(y_true, y_pred, threshold)
            bpcer_vals.append(bpcer)
            macer_vals.append(macer)

        bpcer_vals = np.array(bpcer_vals)
        macer_vals = np.array(macer_vals)

        ax.plot(
            bpcer_vals,
            macer_vals,
            zs=epoch,
            zdir="z",
            label=f"Epoch {epoch}",
            alpha=0.7,
        )

    ax.set_xlabel("BPCER")
    ax.set_ylabel("MACER")
    ax.set_zlabel("Epoch")
    ax.set_title("3D DET Curve Evolution Over Training")
    ax.legend(loc="upper right")
    plt.show()
