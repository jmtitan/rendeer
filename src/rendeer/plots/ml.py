"""ML evaluation plots with explicit tie, binning and objective conventions."""

from typing import Any

import numpy as np

from ..data import vector, xy_frame
from ..figure import PLOTS, Panel, plot
from .comparison import _axis_label


def _binary(y_true: Any, scores: Any, *, both: bool = True) -> tuple[np.ndarray, np.ndarray]:
    truth = vector(None, y_true, "y_true", numeric=True)
    values = vector(None, scores, "scores", numeric=True)
    if len(truth) != len(values) or not np.isfinite(truth).all() or not np.isfinite(values).all():
        raise ValueError("Labels and scores must be paired finite observations")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("Binary labels must be 0 or 1")
    if both and len(np.unique(truth)) != 2:
        raise ValueError("Both binary classes must be observed")
    return truth, values


def _threshold_counts(truth: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    thresholds = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(scores) - 1]
    tp = np.cumsum(truth[order])[thresholds]
    fp = 1 + thresholds - tp
    return np.r_[0, tp], np.r_[0, fp]


@plot
def roc(
    panel: Panel,
    *,
    y_true: Any,
    y_score: Any,
    label: str = "Ours",
    reference: bool = True,
    legend: bool = True,
) -> None:
    """Binary ROC with one step per distinct score; label 1 is the positive class."""
    truth, scores = _binary(y_true, y_score)
    tp, fp = _threshold_counts(truth, scores)
    fpr, tpr = fp / (len(truth) - truth.sum()), tp / truth.sum()
    style = panel.style(label)
    style.pop("marker")
    panel.ax.plot(fpr, tpr, label=label, **style)
    if reference:
        panel.ax.plot([0, 1], [0, 1], color="#888888", linestyle=":", lw=0.8)
    panel.ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    panel.finish(
        "roc",
        xlabel="False positive rate",
        ylabel="True positive rate",
        legend=legend,
        positive_label=1,
        n=len(truth),
        ties="single threshold",
        auc=float(integrate(tpr, fpr)),
    )


@plot
def precision_recall(
    panel: Panel,
    *,
    y_true: Any,
    y_score: Any,
    label: str = "Ours",
    reference: bool = True,
    legend: bool = True,
) -> None:
    """Binary precision-recall staircase; no-positive-prediction endpoint is (0, 1)."""
    truth, scores = _binary(y_true, y_score)
    tp, fp = _threshold_counts(truth, scores)
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    recall = tp / truth.sum()
    style = panel.style(label)
    style.pop("marker")
    panel.ax.step(recall, precision, where="pre", label=label, **style)
    if reference:
        panel.ax.axhline(truth.mean(), color="#888888", linestyle=":", lw=0.8)
    panel.ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    panel.finish(
        "precision_recall",
        xlabel="Recall",
        ylabel="Precision",
        legend=legend,
        positive_label=1,
        n=len(truth),
        prevalence=float(truth.mean()),
        average_precision=float(np.sum(np.diff(recall) * precision[1:])),
    )


@plot
def calibration(
    panel: Panel,
    *,
    y_true: Any,
    y_prob: Any,
    bins: int = 10,
    label: str = "Ours",
    legend: bool = True,
) -> None:
    """Equal-width reliability bins at mean predicted probability; empty bins are gaps."""
    truth, probabilities = _binary(y_true, y_prob, both=False)
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must lie within [0, 1]")
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    edges = np.linspace(0, 1, bins + 1)
    assignment = np.minimum(np.searchsorted(edges, probabilities, side="right") - 1, bins - 1)
    mean, fraction = np.full(bins, np.nan), np.full(bins, np.nan)
    counts = np.bincount(assignment, minlength=bins)
    for index in range(bins):
        members = assignment == index
        if members.any():
            mean[index], fraction[index] = probabilities[members].mean(), truth[members].mean()
    panel.ax.plot(mean, fraction, label=label, **panel.style(label))
    panel.ax.plot([0, 1], [0, 1], color="#888888", linestyle=":", lw=0.8)
    panel.ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    panel.finish(
        "calibration",
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        legend=legend,
        bin_edges=edges.tolist(),
        counts=counts.tolist(),
        bin_closure="left closed, right open, except final bin includes 1",
        n=len(truth),
    )


@plot
def pareto(
    panel: Panel,
    data: Any = None,
    *,
    x: Any,
    y: Any,
    minimize_x: bool = True,
    maximize_y: bool = True,
    hue: Any = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: Any = None,
    legend: bool = True,
) -> None:
    """All observed points and the 2D nondominated frontier; no extrapolation."""
    frame = xy_frame(data, x, y, hue)
    if frame.y.isna().any():
        raise ValueError("Pareto objectives cannot be missing")
    costs = frame.x.to_numpy(dtype=float) * (1 if minimize_x else -1)
    benefits = frame.y.to_numpy() * (1 if maximize_y else -1)
    order = np.lexsort((-benefits, costs))
    chosen, best = [], -np.inf
    for index in order:
        if benefits[index] > best:
            chosen.append(int(index))
            best = benefits[index]
    PLOTS["scatter"](panel, data, x=x, y=y, hue=hue, xlabel=xlabel, ylabel=ylabel, legend=legend)
    frontier = frame.iloc[chosen].sort_values("x", kind="stable")
    panel.ax.plot(
        frontier.x,
        frontier.y,
        color="#252525",
        linestyle="--",
        lw=0.9,
        label="Nondominated frontier",
    )
    if labels is not None:
        panel.annotate(vector(data, labels, "labels").tolist(), x=frame.x, y=frame.y)
    panel.finish(
        "pareto",
        xlabel=_axis_label(x, xlabel),
        ylabel=_axis_label(y, ylabel),
        legend=legend,
        minimize_x=minimize_x,
        maximize_y=maximize_y,
        frontier_indices=chosen,
        interpolation="guide only; no unmeasured result implied",
    )
