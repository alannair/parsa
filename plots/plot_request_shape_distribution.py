# Copyright (C) 2026 M.R. Siavash Katebzadeh <mr@katebzadeh.xyz>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Plot Figure 11: actual request prompt/generated-token density."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy import interpolate
from sklearn.cluster import KMeans

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plots.plot_token_histogram import load_scheduled_result_tokens
from plots.style import set_size


TOKEN_COLUMNS = ("ContextTokens", "GeneratedTokens")


def request_shape_cmap():
    palette = LinearSegmentedColormap.from_list(
        "request_shape_density", ["#bdd8c5", "#193624"],
    )
    palette.set_under("white")
    return palette


def positive_token_pairs(frame):
    missing = set(TOKEN_COLUMNS) - set(frame.columns)
    if missing:
        raise KeyError("token data missing columns: %s" % ", ".join(sorted(missing)))
    pairs = frame.loc[:, TOKEN_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna().copy()
    pairs = pairs[np.isfinite(pairs["ContextTokens"]) & np.isfinite(pairs["GeneratedTokens"])]
    pairs = pairs[(pairs["ContextTokens"] > 0) & (pairs["GeneratedTokens"] > 0)]
    excluded = len(frame) - len(pairs)
    if pairs.empty:
        raise ValueError("no positive prompt/generated token pairs remain for logarithmic axes")
    return pairs, excluded


def _log_edges(values, bins):
    low, high = float(values.min()), float(values.max())
    if low == high:
        low /= np.sqrt(10)
        high *= np.sqrt(10)
    return np.geomspace(low, high, bins + 1)


def load_centroids(path):
    centroids = pd.read_csv(path)
    pairs, _ = positive_token_pairs(centroids)
    if len(pairs) != len(centroids):
        raise ValueError("centroid CSV requires finite positive ContextTokens and GeneratedTokens")
    if "label" in centroids:
        pairs["label"] = centroids.loc[pairs.index, "label"].astype(str)
    return pairs


def cluster_outline(points):
    points = sorted(set((float(x), float(y)) for x, y in points))
    if len(points) < 2:
        return None

    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) == 2:
        return hull
    return hull + [hull[0]]


def smooth_cluster_outline(outline):
    if not outline or len(outline) < 4:
        return outline
    vertices = np.asarray(outline[:-1], dtype=float)
    degree = min(3, len(vertices) - 1)
    tck, _ = interpolate.splprep(vertices.T, s=0, per=True, k=degree)
    values = interpolate.splev(np.linspace(0, 1, 128), tck)
    return list(zip(values[0], values[1]))


def fit_log_clusters(frame):
    pairs, _ = positive_token_pairs(frame)
    points = np.log10(pairs[["GeneratedTokens", "ContextTokens"]].to_numpy(dtype=float))
    if len(points) < 2 or len(np.unique(points, axis=0)) < 2:
        raise ValueError("--centroid-contour requires at least two distinct positive token pairs")
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(points)
    centers = np.array([points[labels == index].mean(axis=0) for index in range(2)])
    centroids = pd.DataFrame({
        "GeneratedTokens": 10 ** centers[:, 0],
        "ContextTokens": 10 ** centers[:, 1],
    })
    return points, labels, centroids


def fit_log_centroids(frame):
    return fit_log_clusters(frame)[2]


def plot_request_shape_distribution(frame, *, bins=50, centroids=None, centroid_contour=False):
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    pairs, excluded = positive_token_pairs(frame)
    prompt = pairs["ContextTokens"].to_numpy(dtype=float)
    generated = pairs["GeneratedTokens"].to_numpy(dtype=float)
    generated_edges = _log_edges(generated, bins)
    prompt_edges = _log_edges(prompt, bins)
    counts, _, _ = np.histogram2d(generated, prompt, bins=(generated_edges, prompt_edges))
    figure, axis = plt.subplots(figsize=set_size(350))
    image = axis.pcolormesh(
        generated_edges,
        prompt_edges,
        counts.T,
        norm=Normalize(vmin=1, vmax=max(2, float(counts.max()))),
        cmap=request_shape_cmap(),
        shading="auto",
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Count")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.xaxis.set_minor_locator(mticker.NullLocator())
    axis.yaxis.set_minor_locator(mticker.NullLocator())
    axis.set_xlabel("Generated Tokens")
    axis.set_ylabel("Prompt Tokens")
    if centroids is not None:
        centroid_pairs, _ = positive_token_pairs(centroids)
        axis.scatter(centroid_pairs["GeneratedTokens"], centroid_pairs["ContextTokens"],
                     marker="x", color="white", linewidths=1.5, zorder=3)
        if "label" in centroids:
            for index, row in centroid_pairs.iterrows():
                axis.annotate(str(centroids.loc[index, "label"]),
                              (row["GeneratedTokens"], row["ContextTokens"]),
                              xytext=(4, 4), textcoords="offset points", color="white")
    if centroid_contour:
        log_points, labels, contour_centroids = fit_log_clusters(pairs)
        axis.scatter(contour_centroids["GeneratedTokens"], contour_centroids["ContextTokens"],
                     marker="x", color="white", linewidths=1.5, zorder=3)
        for index in range(2):
            outline = cluster_outline(log_points[labels == index])
            if outline:
                smooth_outline = smooth_cluster_outline(outline)
                x_values, y_values = zip(*[(10 ** x, 10 ** y) for x, y in smooth_outline])
                axis.plot(x_values, y_values, color="white", linewidth=3.0, zorder=3)
                axis.plot(x_values, y_values, color="black", linewidth=1.0, zorder=4)
    figure.tight_layout()
    return figure, axis, excluded


def _png_path(value):
    path = Path(value)
    if path.suffix.lower() != ".png":
        raise argparse.ArgumentTypeError("--output must end in .png")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="plots.plot_request_shape_distribution", description=__doc__)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--result-csv", required=True)
    parser.add_argument("--output", required=True, type=_png_path)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--centroids-csv", help="Optional CSV with ContextTokens, GeneratedTokens, and optional label")
    parser.add_argument("--centroid-contour", action="store_true",
                        help="Fit two log-space k-means clusters, mark centroids, and outline each cluster")
    args = parser.parse_args(argv)
    frame = load_scheduled_result_tokens(args.schedule, args.result_csv)
    centroids = load_centroids(args.centroids_csv) if args.centroids_csv else None
    figure, _, excluded = plot_request_shape_distribution(
        frame, bins=args.bins, centroids=centroids, centroid_contour=args.centroid_contour,
    )
    if excluded:
        print("excluded %d nonpositive or invalid token pairs" % excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
