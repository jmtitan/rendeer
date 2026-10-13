"""Executable synthetic research gallery; values are demonstrations, not research results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import rendeer as rd

SEED = 20261013


def synthetic_data(seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for method, ceiling in [("Ours", 0.88), ("Baseline", 0.79), ("Ablation", 0.83)]:
        for replicate in range(5):
            offset = rng.normal(0, 0.008)
            for step in range(0, 121, 10):
                value = 0.47 + (ceiling - 0.47) * (1 - np.exp(-step / 35))
                rows.append((step, value + offset + rng.normal(0, 0.006), method, replicate))
    curves = pd.DataFrame(rows, columns=["step", "accuracy", "method", "seed"])
    benchmark = pd.DataFrame(
        [
            (task, score, method)
            for task, scores in [
                ("Task A", [0.85, 0.76, 0.80]),
                ("Task B", [0.78, 0.70, 0.73]),
                ("Task C", [0.90, 0.83, 0.86]),
            ]
            for method, score in zip(["Ours", "Baseline", "Ablation"], scores, strict=True)
        ],
        columns=["task", "f1", "method"],
    )
    distribution = pd.DataFrame(
        {
            "latency": np.r_[
                rng.lognormal(2.3, 0.23, 80),
                rng.lognormal(2.7, 0.26, 80),
                rng.lognormal(2.5, 0.2, 80),
            ],
            "method": np.repeat(["Ours", "Baseline", "Ablation"], 80),
        }
    )
    truth = rng.integers(0, 2, 120)
    scores = rng.beta(2 + 3 * truth, 5 - 3 * truth)
    return {
        "curves": curves,
        "benchmark": benchmark,
        "distribution": distribution,
        "truth": truth,
        "scores": scores,
        "tradeoff": pd.DataFrame(
            {"compute": [4, 8, 16, 32, 64], "accuracy": [0.64, 0.75, 0.82, 0.80, 0.87]}
        ),
        "ablation": pd.DataFrame(
            {"component": ["Full", "No memory", "No search"], "f1": [0.85, 0.79, 0.74]}
        ),
        "features": pd.DataFrame(rng.normal(size=(40, 3)), columns=["Size", "Cost", "Score"]),
    }


def composite(data: dict | None = None) -> rd.Figure:
    data = synthetic_data() if data is None else data
    fig = rd.figure(layout=[["a", "b"], ["c", "d"]], width="double", height=5.4, panel_labels=True)
    fig["a"].line(
        data["curves"],
        x="step",
        y="accuracy",
        hue="method",
        unit="seed",
        error="sd",
        xlabel="Training step",
        ylabel="Accuracy",
    )
    fig["b"].bar(
        data["benchmark"],
        x="task",
        y="f1",
        hue="method",
        highlight="Ours",
        xlabel="Benchmark",
        ylabel="F1",
    )
    fig["c"].bar(
        data["ablation"],
        x="component",
        y="f1",
        highlight="Full",
        orient="horizontal",
        xlabel="F1",
        ylabel="Configuration",
    )
    fig["d"].pareto(
        data["tradeoff"],
        x="compute",
        y="accuracy",
        xlabel="Compute (GPU h)",
        ylabel="Accuracy",
        legend=False,
    )
    fig.legend(ncols=4)
    return fig


def make_gallery() -> dict[str, rd.Figure]:
    """Return fresh native figures. Nothing is exported until the caller saves."""
    data = synthetic_data()
    curves, bench, samples = data["curves"], data["benchmark"], data["distribution"]
    dist = dict(data=samples, x="latency", hue="method", xlabel="Latency (ms)")
    figures = {
        "line": rd.line(
            curves,
            x="step",
            y="accuracy",
            hue="method",
            unit="seed",
            error="sd",
            xlabel="Training step",
            ylabel="Accuracy",
        ),
        "scatter": rd.scatter(
            x=[4, 8, 16, 32, 64],
            y=[0.64, 0.75, 0.82, 0.80, 0.87],
            xlabel="Compute (GPU h)",
            ylabel="Accuracy",
        ),
        "bar": rd.bar(bench, x="task", y="f1", hue="method", xlabel="Task", ylabel="F1"),
        "interval": rd.interval(
            x=["A", "B", "C"],
            y=[0.72, 0.8, 0.75],
            lower=[0.68, 0.75, 0.71],
            upper=[0.77, 0.83, 0.8],
            uncertainty="illustrative supplied range",
            xlabel="Task",
            ylabel="F1",
        ),
        "area": rd.area(
            x=[0, 1, 2, 3, 4], y=[0, 2, 3, 2.5, 4], xlabel="Elapsed time (s)", ylabel="Memory (GB)"
        ),
        "hist": rd.hist(**dist, bins=12),
        "kde": rd.kde(**dist),
        "box": rd.box(**dist),
        "violin": rd.violin(**dist),
        "ecdf": rd.ecdf(**dist),
        "heatmap": rd.heatmap(
            [[0.73, 0.82, 0.84], [0.69, np.nan, 0.80], [0.77, 0.86, 0.88]],
            xticklabels=["Small", "Medium", "Large"],
            yticklabels=["A", "B", "C"],
            xlabel="Model size",
            ylabel="Task",
            cbar_label="F1",
            annot=True,
            vmin=0.65,
            vmax=0.9,
        ),
        "correlation": rd.correlation(data["features"]),
        "confusion": rd.confusion(
            [[32, 3, 1], [2, 27, 4], [0, 5, 26]], labels=["A", "B", "C"], normalize="true"
        ),
        "pareto": rd.pareto(
            data["tradeoff"], x="compute", y="accuracy", xlabel="Compute (GPU h)", ylabel="Accuracy"
        ),
        "roc": rd.roc(y_true=data["truth"], y_score=data["scores"]),
        "precision_recall": rd.precision_recall(y_true=data["truth"], y_score=data["scores"]),
        "calibration": rd.calibration(y_true=data["truth"], y_prob=data["scores"], bins=6),
        "facet": rd.facet(
            curves,
            col="method",
            x="step",
            y="accuracy",
            unit="seed",
            error="sd",
            xlabel="Step",
            ylabel="Accuracy",
            wrap=3,
        ),
        "pairplot": rd.pairplot(data["features"][["Size", "Score"]]),
        "composite": composite(data),
        "ablation": rd.bar(
            data["ablation"],
            x="component",
            y="f1",
            orient="horizontal",
            highlight="Full",
            xlabel="F1",
            ylabel="Configuration",
        ),
        "stacked": rd.bar(
            pd.DataFrame(
                {
                    "model": ["Small", "Large"] * 3,
                    "memory": [2, 4, 1, 2, 0.5, 1],
                    "component": [
                        "Weights",
                        "Weights",
                        "Activations",
                        "Activations",
                        "Cache",
                        "Cache",
                    ],
                }
            ),
            x="model",
            y="memory",
            hue="component",
            stacked=True,
            xlabel="Model",
            ylabel="Memory (GB)",
        ),
        "missing": rd.line(
            x=[0, 1, 2, 3, 4], y=[0.62, 0.70, np.nan, 0.79, 0.81], xlabel="Step", ylabel="Accuracy"
        ),
        "long_labels": rd.bar(
            x=["Transformer baseline", "Retrieval augmented", "Proposed method"],
            y=[0.71, 0.78, 0.84],
            xlabel="Method",
            ylabel="F1",
        ),
        "log_scale": rd.line(
            x=[1, 10, 100, 1000, 10000],
            y=[0.8, 0.55, 0.37, 0.25, 0.17],
            xscale="log",
            yscale="log",
            xlabel="Parameters (millions)",
            ylabel="Loss",
        ),
        "symlog": rd.line(
            x=[-100, -10, -1, 0, 1, 10, 100],
            y=[-8, -4, -1, 0, 1, 4, 8],
            xscale="symlog",
            xlabel="Signed perturbation",
            ylabel="Response",
        ),
        "time_series": rd.line(
            x=pd.date_range("2026-01-01", periods=6, freq="MS"),
            y=[12, 14, 13, 18, 17, 19],
            xlabel="Month",
            ylabel="Throughput (requests/s)",
        ),
    }
    with rd.context(theme="grayscale"):
        figures["grayscale"] = rd.line(
            curves,
            x="step",
            y="accuracy",
            hue="method",
            unit="seed",
            error="se",
            xlabel="Training step",
            ylabel="Accuracy",
        )
    many = rd.bar(
        x=[f"Configuration {i:02}" for i in range(12)],
        y=np.linspace(0.55, 0.88, 12),
        orient="horizontal",
        width="double",
        height=4.5,
        xlabel="F1",
        ylabel="Configuration",
    )
    figures["many_categories"] = many
    shared = rd.figure(layout=[["a", "b"]], width="double", panel_labels=True)
    for name, values in [("a", [[0.5, 0.7], [0.6, 0.8]]), ("b", [[0.7, 0.8], [0.8, 0.9]])]:
        shared[name].heatmap(
            values, vmin=0.5, vmax=0.9, colorbar=False, annot=True, xlabel="Task", ylabel="Method"
        )
    shared.colorbar(panels=["a", "b"], label="F1")
    figures["shared_colorbar"] = shared
    for fig in figures.values():
        fig.records.append(
            {"panel": "*", "kind": "source", "dataset": "synthetic demonstration", "seed": SEED}
        )
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", choices=["line"], help="One-item end-to-end smoke test")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.only:
        data = synthetic_data()
        figures = {
            "line": rd.line(
                data["curves"],
                x="step",
                y="accuracy",
                hue="method",
                error="sd",
                unit="seed",
                xlabel="Step",
                ylabel="Accuracy",
            )
        }
    else:
        figures = make_gallery()
    reports = {}
    for name, fig in figures.items():
        report = fig.validate()
        reports[name] = report.to_dict()
        if not report.passed:
            (output / "quality.json").write_text(json.dumps(reports, indent=2))
            raise rd.QualityError(report)
        fig.save(output / f"{name}.png", overwrite=True)
        print(f"Rendered {name}: {len(report.warnings)} advisory warnings", flush=True)
    (output / "quality.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
