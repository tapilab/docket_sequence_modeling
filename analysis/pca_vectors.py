import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.pyplot as plt
import seaborn as sns

from training.teacher_forcing import build_teacher_forced_sequences

warnings.filterwarnings("ignore")


"""
Step 1: Extraction of the pre-classifier hidden vectors
"""

def extract_pre_classifier_vectors(model, seq):
    device = next(model.parameters()).device
    seq = seq[-model.max_len:]
    T = len(seq)
    if T == 0:
        return None

    hidden_features = model.fuse[0].in_features

    reps = [
        model.encode_proceeding(p, i, hidden_features)
        for i, p in enumerate(seq)
    ]

    # Stack -> (1, T, hidden_dim)
    x = torch.stack(reps, dim=1)

    # Add positional encoding
    x = x + model.pos[:, :T, :]

    # Causal mask
    causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

    # Transformer encoder
    out = model.encoder(x, is_causal=True, mask=causal_mask)

    # Return last-timestep hidden state -> (hidden_dim,)
    return out[0, -1, :].detach().cpu()


def extract_all_pre_classifier_vectors(
    model,
    data,
    label_processor,
    device,
    max_samples=None,
    use_teacher_forcing=True,
    verbose=True,
):
    """
    Returns
    -------
    vectors : np.ndarray  (n_cases, hidden_dim)
        Last-timestep pre-classifier embedding, one row per case.

    final_labels : List[np.ndarray]
        Binary label vector at the final timestep.

    all_case_labels_matrix : np.ndarray  (n_cases, n_labels)
        Binary matrix where cell [i, j] == 1 if label j appeared at ANY
        timestep in case i.

    metadata_list : List[dict]
    """
    model.eval()
    model.to(device)

    all_vectors = []
    final_labels = []
    all_case_labels = []
    metadata_list = []

    samples = data[:max_samples] if max_samples is not None else data
    desc = f"Extracting pre-classifier vectors for {len(samples)} samples" if verbose else None

    for case_idx, (seq, labels_seq) in enumerate(tqdm(samples, desc=desc, disable=not verbose)):
        if len(seq) == 0:
            continue

        # Convert labels to binary if needed
        if isinstance(labels_seq[0], list) and (
            len(labels_seq[0]) == 0 or isinstance(labels_seq[0][0], str)
        ):
            labels_binary = label_processor.transform_sequence(labels_seq)
        else:
            labels_binary = labels_seq

        labels_binary_np = []
        for lbl in labels_binary:
            if isinstance(lbl, torch.Tensor):
                labels_binary_np.append(lbl.cpu().numpy())
            else:
                labels_binary_np.append(np.array(lbl))

        if use_teacher_forcing:
            teacher_seq, _ = build_teacher_forced_sequences(
                seq, labels_binary_np, label_processor, include_initial_state=True
            )
            input_seq = teacher_seq
        else:
            input_seq = seq

        if len(input_seq) == 0:
            continue

        with torch.no_grad():
            vector = extract_pre_classifier_vectors(model, input_seq)

        if vector is None:
            continue

        all_vectors.append(vector.numpy())

        # Final timestep labels
        final_label_idx = len(input_seq) - 1
        final_labels.append(labels_binary_np[final_label_idx])

        # Any label seen anywhere in case
        stacked = np.stack(labels_binary_np, axis=0)
        ever_seen = (stacked.max(axis=0) > 0).astype(int)
        all_case_labels.append(ever_seen)

        first_proceeding = input_seq[0]
        metadata_list.append({
            "case_idx": case_idx,
            "defendant_id": first_proceeding.get("defendant_id", "unknown"),
            "case_id": first_proceeding.get(
                "case_id",
                first_proceeding.get("docket_id", f"case_{case_idx}")
            ),
            "seq_len": len(seq),
            "court_type": first_proceeding.get("type", "unknown"),
        })

    vectors = np.vstack(all_vectors) if all_vectors else np.empty((0,))
    all_case_labels_matrix = np.stack(all_case_labels, axis=0) if all_case_labels else np.empty((0,))

    if verbose and vectors.size > 0:
        print(f"Extracted {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")

    return vectors, final_labels, all_case_labels_matrix, metadata_list


"""
Step 2: PCA Reduction
"""

def reduce_vectors_with_pca(
    vectors: np.ndarray,
    n_components: int = 2,
    standardize: bool = True,
    random_state=42,
) -> Tuple[np.ndarray, Optional[StandardScaler], PCA]:
    scaler = None
    X = vectors.copy()

    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)

    print(
        f"PCA: Reduced from {vectors.shape[1]} to {n_components} dimensions, "
        f"explained variance ratio: {pca.explained_variance_ratio_}"
    )
    return X_reduced, scaler, pca


"""
Step 3: Clustering
"""

def run_kmeans(X, nclusters, random_state):
    kmeans = KMeans(n_clusters=nclusters, n_init=20, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X)
    return cluster_labels, kmeans


def run_dbscan_func(X, eps=1.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X)
    return cluster_labels, dbscan


"""
Step 4: Evaluation of clustering
"""

def evaluate_clustering(X, cluster_labels, true_labels_flat):
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Number of clusters found: {n_clusters}")

    if n_clusters <= 1:
        print("Not enough clusters to evaluate.")
        return {}

    metrics = {"number_of_clusters": n_clusters}

    if n_clusters >= 2 and len(X) >= n_clusters:
        silhouette = silhouette_score(X, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X, cluster_labels)
        metrics["silhouette_score"] = silhouette
        metrics["calinski_harabasz_score"] = calinski_harabasz

    if true_labels_flat is not None:
        adjusted_rand = adjusted_rand_score(true_labels_flat, cluster_labels)
        normalized_mutual_info = normalized_mutual_info_score(true_labels_flat, cluster_labels)
        metrics["adjusted_rand_index"] = adjusted_rand
        metrics["normalized_mutual_info"] = normalized_mutual_info

    return metrics


"""
Step 5: Visualization
"""

def _truncate_label(label: str, max_len: int = 35) -> str:
    return label if len(label) <= max_len else label[:max_len - 1] + "…"


def _ensure_label_array(label_matrix_like):
    if isinstance(label_matrix_like, np.ndarray):
        return label_matrix_like
    return np.array([lbl for lbl in label_matrix_like])


def plot_pca_label_grid(
    projected: np.ndarray,
    label_matrix,
    label_names: List[str],
    pca,
    output_path=None,
    title=None,
    min_prevalence_pct: float = 0.0,
    max_cols: int = 5,
):
    """
    One subplot per label. Shows all labels by default.
    """
    labels_arr = _ensure_label_array(label_matrix)
    N, L = labels_arr.shape

    label_counts = labels_arr.sum(axis=0)
    prevalence = (label_counts / N) * 100
    active_indices = [i for i in range(L) if prevalence[i] >= min_prevalence_pct]

    if not active_indices:
        print("No labels exceed the minimum prevalence threshold.")
        return

    n_plots = len(active_indices)
    n_cols = min(max_cols, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    evr = pca.explained_variance_ratio_

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        sharex=True,
        sharey=True,
    )

    if n_plots == 1:
        axes = np.array([axes])

    axes_flat = axes.flatten()
    cmap = plt.get_cmap("tab20" if n_plots <= 20 else "nipy_spectral", n_plots)

    for plot_idx, label_idx in enumerate(active_indices):
        ax = axes_flat[plot_idx]
        mask = labels_arr[:, label_idx] == 1
        n_active = mask.sum()
        pct = prevalence[label_idx]

        ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c="lightgray",
            s=6,
            alpha=0.3,
            edgecolors="none",
            rasterized=True,
        )

        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            c=[cmap(plot_idx)],
            s=12,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.2,
            rasterized=True,
        )

        short_name = _truncate_label(label_names[label_idx], 30)
        ax.set_title(f"{short_name}\n(n={n_active:,}, {pct:.1f}%)", fontsize=7, pad=3)
        ax.tick_params(labelsize=5)

    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.supxlabel(f"PC-1 ({evr[0] * 100:.1f}% var)", fontsize=10)
    fig.supylabel(f"PC-2 ({evr[1] * 100:.1f}% var)", fontsize=10)
    fig.suptitle(
        title or "PCA Embeddings — Label Presence",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved label grid to {output_path}")

    plt.show()
    plt.close(fig)


def plot_pca_by_clusters(
    projected,
    cluster_labels,
    pca,
    centroids_2d=None,
    output_path=None,
    title=None,
):
    fig, ax = plt.subplots(figsize=(12, 9))
    unique_clusters = sorted(set(cluster_labels))
    n_clusters = len(unique_clusters)
    cmap = plt.get_cmap("Set2" if n_clusters <= 8 else "tab20", n_clusters)

    for idx, cluster in enumerate(unique_clusters):
        mask = cluster_labels == cluster
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            c=[cmap(idx)],
            label=f"Cluster {cluster}" if cluster != -1 else "Noise",
            s=25,
            alpha=0.65,
            edgecolors="w",
            linewidth=0.3,
        )

    if centroids_2d is not None:
        ax.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            c="black",
            marker="X",
            s=200,
            zorder=10,
            label="Centroids",
            edgecolors="white",
            linewidth=1.5,
        )

    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC-1 ({evr[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC-2 ({evr[1] * 100:.1f}% var)")
    ax.set_title(
        title or "PCA of Pre-Classifier Vectors — Cluster Assignment",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved cluster PCA to {output_path}")

    plt.show()
    plt.close(fig)


def _build_multilabel_matrix(cluster_labels, label_matrix, label_names):
    """
    Build (n_clusters x n_labels) matrix.
    Cell = count of samples in that cluster with label active.
    """
    labels_arr = _ensure_label_array(label_matrix)
    unique_clusters = sorted(set(cluster_labels) - {-1})
    n_labels = len(label_names)

    raw = np.zeros((len(unique_clusters), n_labels), dtype=int)
    sizes = []

    for cl_idx, cluster in enumerate(unique_clusters):
        mask = cluster_labels == cluster
        n = mask.sum()
        sizes.append(n)
        raw[cl_idx] = labels_arr[mask].sum(axis=0).astype(int)

    pct = np.zeros_like(raw, dtype=float)
    for i, s in enumerate(sizes):
        if s > 0:
            pct[i] = (raw[i] / s) * 100.0

    return raw, pct, sizes, unique_clusters


import textwrap

def plot_multilabel_cluster_heatmap(
    cluster_labels,
    label_matrix,
    label_names,
    output_path=None,
    title=None,
    min_prevalence_pct: float = 0.0,
    label_name_map: dict | None = None,
):
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size':          7,
        'axes.titlesize':     9,
        'axes.labelsize':     8,
        'xtick.labelsize':    7,
        'ytick.labelsize':    7,
        'axes.linewidth':     0.4,
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
    })

    raw, pct, sizes, clusters = _build_multilabel_matrix(
        cluster_labels, label_matrix, label_names
    )

    col_mask = pct.max(axis=0) >= min_prevalence_pct
    pct_f    = pct[:, col_mask]
    raw_f    = raw[:, col_mask]
    names_f  = [label_names[i]
                for i in range(len(label_names)) if col_mask[i]]

    if pct_f.shape[1] == 0:
        print("No labels exceed the minimum prevalence threshold.")
        return

    # ── TRANSPOSE: rows = labels, cols = clusters ─────────────────────────────
    pct_T = pct_f.T
    raw_T = raw_f.T

    # ── annotation: percentage only, no counts ────────────────────────────────
    annot = np.empty_like(pct_T, dtype=object)
    for i in range(pct_T.shape[0]):
        for j in range(pct_T.shape[1]):
            p = pct_T[i, j]
            if p >= 1.0:
                annot[i, j] = f"{p:.0f}%"
            elif p > 0:
                annot[i, j] = "<1%"
            else:
                annot[i, j] = ""

    # ── axis labels ───────────────────────────────────────────────────────────
    def _clean_label(name: str) -> str:
        cleaned = name.replace('_', ' ').title()
        cleaned = cleaned.replace('Orleans ', '').replace('Orleans', '')
        return cleaned.strip()

    y_labels = []
    any_wrapped = False
    for n in names_f:
        if label_name_map and n in label_name_map:
            y_labels.append(label_name_map[n])
        else:
            wrapped = textwrap.fill(_clean_label(n), width=18)
            if "\n" in wrapped:
                any_wrapped = True
            y_labels.append(wrapped)

    x_labels = [f"Cluster {c}" for c in clusters]

    # ── figure sizing ─────────────────────────────────────────────────────────
    n_rows         = pct_T.shape[0]
    per_row_height = 0.42 if any_wrapped else 0.24
    min_fig_h      = 5.0  if any_wrapped else 3.0
    fig_w          = 3.15
    fig_h          = max(min_fig_h, n_rows * per_row_height + 1.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        pct_T,
        annot=annot, fmt="",
        cmap="YlOrRd",
        xticklabels=x_labels,
        yticklabels=y_labels,
        ax=ax,
        linewidths=0.3, linecolor="white",
        vmin=0, vmax=100,
        annot_kws={"size": 4.5},
        cbar_kws={
            "label":   "% of cases in cluster\nwith label",
            "shrink":  0.4,
            "pad":     0.02,
            "aspect":  15,
        },
    )

    # colorbar font
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(
        "% of cases in cluster\nwith label",
        fontsize=6, labelpad=3,
    )

    # Only add a title if one is explicitly provided
    if title:
        ax.set_title(
            title,
            fontsize=9,
            fontweight='bold',
            pad=6,
        )

    # Remove axis titles
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        ha='center',
        fontsize=7,
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=6.5,
        va='center',
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=600, bbox_inches='tight', format='png')
        print(f"Saved multi-label heatmap to {output_path}")

    plt.close(fig)

    # ── console summary ───────────────────────────────────────────────────────
    print("\nMULTI-LABEL CLUSTER SUMMARY")
    print("-" * 70)
    for i, (cl, sz) in enumerate(zip(clusters, sizes)):
        sorted_idx = np.argsort(pct[i])[::-1]
        label_parts = [
            f"{label_names[j]} ({pct[i, j]:.0f}%, n={raw[i, j]})"
            for j in sorted_idx if raw[i, j] > 0
        ]
        label_str = ", ".join(label_parts) if label_parts else "No active labels"
        print(f"  Cluster {cl} ({sz:,} samples): {label_str}")

def plot_cluster_label_clustermap(
    cluster_labels,
    label_matrix,
    label_names,
    output_path=None,
    title=None,
    min_prevalence_pct: float = 0.0,
):
    raw, pct, sizes, clusters = _build_multilabel_matrix(
        cluster_labels, label_matrix, label_names
    )

    col_mask = pct.max(axis=0) >= min_prevalence_pct
    pct_f = pct[:, col_mask]
    names_f = [
        _truncate_label(label_names[i], 30)
        for i in range(len(label_names))
        if col_mask[i]
    ]

    if pct_f.shape[1] == 0:
        print("No labels exceed the minimum prevalence threshold.")
        return

    df = pd.DataFrame(
        pct_f,
        index=[f"Cluster {c} (n={s:,})" for c, s in zip(clusters, sizes)],
        columns=names_f,
    )

    g = sns.clustermap(
        df,
        row_cluster=False,
        col_cluster=True,
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        linecolor="white",
        figsize=(max(12, len(names_f) * 0.7), max(5, len(clusters) * 0.85)),
        dendrogram_ratio=(0.0, 0.12),
        cbar_pos=(0.02, 0.82, 0.03, 0.15),
        cbar_kws={"label": "% prevalence"},
    )

    g.ax_heatmap.set_xlabel("Label (hierarchically clustered)", fontsize=10)
    g.ax_heatmap.set_ylabel("Cluster", fontsize=10)
    g.ax_heatmap.set_title(
        title or "Cluster × Label — Hierarchically Clustered Labels",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=55, ha="right", fontsize=7)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8)

    if output_path:
        g.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved clustermap to {output_path}")

    plt.show()
    plt.close()


def plot_cluster_label_stacked_bars(
    cluster_labels,
    label_matrix,
    label_names,
    output_path=None,
    title=None,
    top_n_labels: Optional[int] = None,
):
    raw, pct, sizes, clusters = _build_multilabel_matrix(
        cluster_labels, label_matrix, label_names
    )

    total_per_label = raw.sum(axis=0)

    if top_n_labels is None:
        top_n_labels = len(label_names)

    top_idx = np.argsort(total_per_label)[::-1][:top_n_labels]
    pct_top = pct[:, top_idx]
    top_names = [_truncate_label(label_names[i], 30) for i in top_idx]
    row_labels = [f"Cluster {c}\n(n={s:,})" for c, s in zip(clusters, sizes)]

    fig, ax = plt.subplots(figsize=(14, max(4, len(clusters) * 0.8)))
    cmap = plt.get_cmap("tab20" if len(top_idx) <= 20 else "nipy_spectral", len(top_idx))

    y_pos = np.arange(len(clusters))
    left = np.zeros(len(clusters))

    for j in range(len(top_idx)):
        bars = ax.barh(
            y_pos,
            pct_top[:, j],
            left=left,
            color=cmap(j),
            edgecolor="white",
            linewidth=0.3,
            label=top_names[j],
        )

        for i, bar in enumerate(bars):
            w = bar.get_width()
            if w > 5:
                ax.text(
                    left[i] + w / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold",
                    color="white",
                )

        left += pct_top[:, j]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Label prevalence (%, not mutually exclusive)", fontsize=10)
    ax.set_title(
        title or f"Cluster Composition — {len(top_idx)} Labels",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        frameon=True,
        framealpha=0.9,
    )
    ax.set_xlim(0, None)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved stacked bars to {output_path}")

    plt.show()
    plt.close(fig)


def plot_elbow(vectors_2d, range_of_k, output_path=None):
    inertias, silhouettes = [], []

    for k in range_of_k:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(vectors_2d)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(vectors_2d, labels) if k >= 2 else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(list(range_of_k), inertias, "o-", color="steelblue")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Curve")
    ax1.grid(True, alpha=0.3)

    ax2.plot(list(range_of_k), silhouettes, "s-", color="coral")
    ax2.set_xlabel("K")
    ax2.set_ylabel("Silhouette")
    ax2.set_title("Silhouette vs. K")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Choosing the Number of Clusters", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.show()
    plt.close(fig)


def _plot_pca_by_fold(
    projected: np.ndarray,
    fold_ids: np.ndarray,
    pca: PCA,
    n_folds: int,
    output_path: Optional[Path] = None,
):
    fig, ax = plt.subplots(figsize=(12, 9))
    cmap = plt.get_cmap("Set1", n_folds)

    for fold in sorted(set(fold_ids)):
        mask = fold_ids == fold
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            c=[cmap(int(fold) - 1)],
            label=f"Fold {int(fold)}",
            s=20,
            alpha=0.55,
            edgecolors="w",
            linewidths=0.2,
        )

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC-1 ({ev[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC-2 ({ev[1] * 100:.1f}% var)")
    ax.set_title("Combined Embeddings — Coloured by Fold", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.show()
    plt.close(fig)


def _resolve_label_matrix(label_mode, final_labels, all_case_labels_matrix):
    if label_mode == "final":
        return np.array([lbl for lbl in final_labels])
    elif label_mode == "ever":
        return all_case_labels_matrix
    else:
        raise ValueError("label_mode must be either 'final' or 'ever'")


"""
Step 7: Full Pipeline Methods
"""

def run_embedding_clusters_kfolds_pipeline(
    cv_runner,
    label_processor,
    device,
    output_dir,
    max_samples_per_fold=None,
    n_clusters=6,
    use_teacher_forcing=True,
    run_dbscan=False,
    dbscan_eps=1.5,
    dbscan_min_samples=5,
    show_elbow=True,
    verbose=True,
    random_state=42,
    label_mode="ever",
    min_prevalence_pct=0.0,
    stacked_bar_top_n=None,
):
    """
    K-fold embedding analysis.

    label_mode:
        "final" -> only labels active at final timestep
        "ever"  -> labels seen anywhere in the case
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_folds = cv_runner.n_splits

    all_vectors = []
    all_final_labels = []
    all_case_labels = []
    all_metadata = []

    for fold_idx in range(n_folds):
        print(f"\n{'─' * 70}")
        print(f"  Extracting embeddings from Fold {fold_idx + 1} / {n_folds}")
        print(f"{'─' * 70}")

        fold_model = cv_runner.load_fold_model(fold_idx)
        fold_model.eval()
        fold_model.to(device)

        val_data, _ = cv_runner.get_fold_val_data(fold_idx)

        vectors, final_labels, all_case_labels_matrix, metadata = extract_all_pre_classifier_vectors(
            fold_model,
            val_data,
            label_processor,
            device,
            max_samples=max_samples_per_fold,
            use_teacher_forcing=use_teacher_forcing,
            verbose=verbose,
        )

        for m in metadata:
            m["fold_idx"] = fold_idx + 1

        vectors = StandardScaler().fit_transform(vectors)
        vectors = normalize(vectors, norm="l2")

        all_vectors.append(vectors)
        all_final_labels.extend(final_labels)
        all_case_labels.append(all_case_labels_matrix)
        all_metadata.extend(metadata)

        print(
            f"  Fold {fold_idx + 1}: extracted {vectors.shape[0]} vectors "
            f"(dim={vectors.shape[1]})"
        )

        del fold_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    vectors_combined = np.vstack(all_vectors)
    all_case_labels_combined = np.vstack(all_case_labels)
    N = vectors_combined.shape[0]
    fold_ids = np.array([m["fold_idx"] for m in all_metadata])

    print(f"\n{'=' * 70}")
    print(f"COMBINED: {N:,} vectors × {vectors_combined.shape[1]} dims across {n_folds} folds")
    print(f"{'=' * 70}")

    for f in sorted(set(fold_ids)):
        n = (fold_ids == f).sum()
        print(f"  Fold {int(f)}: {n:,} samples ({n / N * 100:.1f}%)")

    label_matrix_for_analysis = _resolve_label_matrix(
        label_mode=label_mode,
        final_labels=all_final_labels,
        all_case_labels_matrix=all_case_labels_combined,
    )

    projected, scaler, pca = reduce_vectors_with_pca(
        vectors_combined,
        n_components=2,
        standardize=True,
        random_state=random_state,
    )

    if show_elbow:
        print("\nElbow / Silhouette analysis …")
        plot_elbow(
            projected,
            range_of_k=range(2, 13),
            output_path=output_dir / "elbow_silhouette.png",
        )

    print(f"\nKMeans with K={n_clusters} …")
    km_labels, km_model = run_kmeans(
        projected,
        nclusters=n_clusters,
        random_state=random_state,
    )
    km_metrics = evaluate_clustering(projected, km_labels, true_labels_flat=None)
    print(f"   KMeans metrics: {km_metrics}")
    centroids_2d = km_model.cluster_centers_

    plot_pca_label_grid(
        projected,
        label_matrix_for_analysis,
        label_processor.label_names,
        pca,
        output_path=output_dir / "pca_label_grid.png",
        title=f"K-Fold Combined Embeddings — Label Presence ({label_mode})",
        min_prevalence_pct=min_prevalence_pct,
    )

    plot_pca_by_clusters(
        projected,
        km_labels,
        pca,
        centroids_2d=centroids_2d,
        output_path=output_dir / "pca_by_kmeans_cluster.png",
        title="K-Fold Combined Embeddings — KMeans Clusters",
    )

    _plot_pca_by_fold(
        projected,
        fold_ids,
        pca,
        n_folds,
        output_path=output_dir / "pca_by_fold.png",
    )

    plot_multilabel_cluster_heatmap(
        km_labels,
        label_matrix_for_analysis,
        label_processor.label_names,
        output_path=output_dir / "multilabel_cluster_heatmap.png",
        title=f"K-Fold Combined: Multi-Label Cluster Composition ({label_mode})",
        min_prevalence_pct=min_prevalence_pct,
    )

    plot_cluster_label_clustermap(
        km_labels,
        label_matrix_for_analysis,
        label_processor.label_names,
        output_path=output_dir / "cluster_label_clustermap.png",
        title=f"K-Fold Combined: Cluster × Label ({label_mode})",
        min_prevalence_pct=min_prevalence_pct,
    )

    plot_cluster_label_stacked_bars(
        km_labels,
        label_matrix_for_analysis,
        label_processor.label_names,
        output_path=output_dir / "cluster_label_stacked_bars.png",
        title=f"K-Fold Combined: Cluster Composition ({label_mode})",
        top_n_labels=stacked_bar_top_n,
    )

    db_results = {}
    if run_dbscan:
        print(f"\nDBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples}) …")
        db_labels, db_model = run_dbscan_func(
            projected,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
        )
        db_metrics = evaluate_clustering(projected, db_labels, true_labels_flat=None)
        print(f"   DBSCAN metrics: {db_metrics}")

        plot_pca_by_clusters(
            projected,
            db_labels,
            pca,
            output_path=output_dir / "pca_by_dbscan_cluster.png",
            title="K-Fold Combined — DBSCAN Clusters",
        )

        plot_multilabel_cluster_heatmap(
            db_labels,
            label_matrix_for_analysis,
            label_processor.label_names,
            output_path=output_dir / "dbscan_multilabel_heatmap.png",
            title=f"K-Fold DBSCAN: Multi-Label Cluster Composition ({label_mode})",
            min_prevalence_pct=min_prevalence_pct,
        )

        db_results = {
            "dbscan_labels": db_labels,
            "dbscan_model": db_model,
            "dbscan_metrics": db_metrics,
        }

    df = pd.DataFrame(all_metadata)
    df["pc1"] = projected[:, 0]
    df["pc2"] = projected[:, 1]
    df["kmeans_cluster"] = km_labels

    if run_dbscan:
        df["dbscan_cluster"] = db_labels

    csv_path = output_dir / "embedding_clustering_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults CSV: {csv_path}")

    labels_arr = label_matrix_for_analysis

    print("\n" + "=" * 70)
    print(f"CLUSTER COMPOSITION SUMMARY (KMeans, label_mode={label_mode})")
    print("=" * 70)

    for cl in sorted(set(km_labels)):
        mask = km_labels == cl
        n = mask.sum()
        label_sums = labels_arr[mask].sum(axis=0)

        all_indices = np.argsort(label_sums)[::-1]
        label_parts = [
            f"{label_processor.label_names[j]} ({int(label_sums[j])})"
            for j in all_indices
            if label_sums[j] > 0
        ]
        top_str = ", ".join(label_parts) if label_parts else "No active labels"

        fold_dist = Counter(fold_ids[mask])
        fold_str = ", ".join(f"F{int(f)}:{c}" for f, c in sorted(fold_dist.items()))

        print(f"  Cluster {cl} — {n} samples — folds: [{fold_str}]")
        print(f"      labels: {top_str}")

    print(f"\nAnalysis complete. Outputs: {output_dir}")

    results = {
        "vectors": vectors_combined,
        "projected_2d": projected,
        "pca": pca,
        "scaler": scaler,
        "kmeans_labels": km_labels,
        "kmeans_model": km_model,
        "kmeans_metrics": km_metrics,
        "metadata": all_metadata,
        "final_labels": all_final_labels,
        "all_case_labels_matrix": all_case_labels_combined,
        "label_matrix_used": label_matrix_for_analysis,
        "fold_ids": fold_ids,
        "dataframe": df,
        **db_results,
    }
    return results


def run_embedding_clusters_pipeline(
    model,
    data,
    label_processor,
    device,
    output_dir,
    max_samples=None,
    n_clusters=6,
    use_teacher_forcing=True,
    run_dbscan=False,
    dbscan_eps=1.5,
    dbscan_min_samples=5,
    show_elbow=True,
    verbose=True,
    random_state=42,
    label_mode="ever",
    min_prevalence_pct=0.0,
    stacked_bar_top_n=None,
):
    """
    Single-model embedding cluster analysis.

    label_mode:
        "final" -> only labels active at final timestep
        "ever"  -> labels seen anywhere in the case
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EMBEDDING CLUSTERING ANALYSIS")
    print("=" * 70)

    vectors, final_labels, all_case_labels_matrix, metadata = extract_all_pre_classifier_vectors(
        model,
        data,
        label_processor,
        device,
        max_samples=max_samples,
        use_teacher_forcing=use_teacher_forcing,
        verbose=verbose,
    )

    N = vectors.shape[0]
    print(f"\nExtracted {N:,} vectors (dim={vectors.shape[1]})")

    label_matrix_for_analysis = _resolve_label_matrix(
        label_mode=label_mode,
        final_labels=final_labels,
        all_case_labels_matrix=all_case_labels_matrix,
    )

    projected, scaler, pca = reduce_vectors_with_pca(
        vectors,
        n_components=2,
        standardize=True,
        random_state=random_state,
    )

    if show_elbow:
        print("\nElbow / Silhouette analysis …")
        plot_elbow(
            projected,
            range_of_k=range(2, 13),
            output_path=output_dir / "elbow_silhouette.png",
        )

    print(f"\nKMeans with K={n_clusters} …")
    km_labels, km_model = run_kmeans(
        projected,
        nclusters=n_clusters,
        random_state=random_state,
    )
    km_metrics = evaluate_clustering(projected, km_labels, true_labels_flat=None)
    print(f"   KMeans metrics: {km_metrics}")
    centroids_2d = km_model.cluster_centers_

    plot_pca_label_grid(
        projected,
        label_matrix_for_analysis,
        label_processor.label_names,
        pca,
        output_path=output_dir / "pca_label_grid.png",
        title=f"PCA Embeddings — Label Presence ({label_mode})",
        min_prevalence_pct=min_prevalence_pct,
    )

    plot_pca_by_clusters(
        projected,
        km_labels,
        pca,
        centroids_2d=centroids_2d,
        output_path=output_dir / "pca_by_kmeans_cluster.png",
    )

    plot_multilabel_cluster_heatmap(
        km_labels,
        label_matrix_for_analysis,
        label_processor.label_names,
        output_path=output_dir / "multilabel_cluster_heatmap.png",
        title=f"Multi-Label Cluster Composition ({label_mode})",
        min_prevalence_pct=min_prevalence_pct,
    )

    plot_cluster_label_clustermap(
        km_labels,
        label_matrix_for_analysis,
        label_processor.label_names,
        output_path=output_dir / "cluster_label_clustermap.png",
        title=f"Cluster × Label ({label_mode})",
        min_prevalence_pct=min_prevalence_pct,
    )

    plot_cluster_label_stacked_bars(
        km_labels,
        label_matrix_for_analysis,
        label_processor.label_names,
        output_path=output_dir / "cluster_label_stacked_bars.png",
        title=f"Cluster Composition ({label_mode})",
        top_n_labels=stacked_bar_top_n,
    )

    db_results = {}
    if run_dbscan:
        print(f"\nDBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples}) …")
        db_labels, db_model = run_dbscan_func(
            projected,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
        )
        db_metrics = evaluate_clustering(projected, db_labels, true_labels_flat=None)
        print(f"   DBSCAN metrics: {db_metrics}")

        plot_pca_by_clusters(
            projected,
            db_labels,
            pca,
            output_path=output_dir / "pca_by_dbscan_cluster.png",
            title="Pre-Classifier Embeddings (PCA) — DBSCAN Clusters",
        )

        plot_multilabel_cluster_heatmap(
            db_labels,
            label_matrix_for_analysis,
            label_processor.label_names,
            output_path=output_dir / "dbscan_multilabel_heatmap.png",
            title=f"DBSCAN: Multi-Label Cluster Composition ({label_mode})",
            min_prevalence_pct=min_prevalence_pct,
        )

        db_results = {
            "dbscan_labels": db_labels,
            "dbscan_model": db_model,
            "dbscan_metrics": db_metrics,
        }

    df = pd.DataFrame(metadata)
    df["pc1"] = projected[:, 0]
    df["pc2"] = projected[:, 1]
    df["kmeans_cluster"] = km_labels

    if run_dbscan:
        df["dbscan_cluster"] = db_labels

    csv_path = output_dir / "embedding_clustering_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults CSV: {csv_path}")

    labels_arr = label_matrix_for_analysis

    print("\n" + "=" * 70)
    print(f"CLUSTER COMPOSITION SUMMARY (KMeans, label_mode={label_mode})")
    print("=" * 70)

    for cl in sorted(set(km_labels)):
        mask = km_labels == cl
        n = mask.sum()
        label_sums = labels_arr[mask].sum(axis=0)

        all_indices = np.argsort(label_sums)[::-1]
        label_parts = [
            f"{label_processor.label_names[j]} ({int(label_sums[j])})"
            for j in all_indices
            if label_sums[j] > 0
        ]
        label_str = ", ".join(label_parts) if label_parts else "No active labels"

        print(f"  Cluster {cl} — {n} samples — labels: {label_str}")

    print(f"\nAnalysis complete. Outputs: {output_dir}")

    results = {
        "vectors": vectors,
        "projected_2d": projected,
        "pca": pca,
        "scaler": scaler,
        "kmeans_labels": km_labels,
        "kmeans_model": km_model,
        "kmeans_metrics": km_metrics,
        "metadata": metadata,
        "final_labels": final_labels,
        "all_case_labels_matrix": all_case_labels_matrix,
        "label_matrix_used": label_matrix_for_analysis,
        "dataframe": df,
        **db_results,
    }
    return results