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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.preprocessing import normalize

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

    # Stack → (1, T, hidden_dim)
    x = torch.stack(reps, dim=1)

    # Add positional encoding
    x = x + model.pos[:, :T, :]

    # Causal mask
    causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

    # Transformer encoder
    out = model.encoder(x, is_causal=True, mask=causal_mask)

    # Return last-timestep hidden state  →  (hidden_dim,)
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
        Binary label vector at the final timestep — used for scatter-plot
        dominant-label colouring only.
    all_case_labels_matrix : np.ndarray  (n_cases, n_labels)
        Binary matrix where cell [i, j] == 1 if label j appeared at ANY
        timestep in case i.  Used for the heatmap.
    metadata_list : List[dict]
    """
    model.eval()
    model.to(device)
    all_vectors = []
    final_labels = []           # last-timestep binary vec  → scatter colouring
    all_case_labels = []        # OR across all timesteps   → heatmap
    metadata_list = []

    samples = data[:max_samples] if max_samples is not None else data
    desc = f"Extracting pre-classifier vectors for {len(samples)} samples" if verbose else None

    for case_idx, (seq, labels_seq) in enumerate(tqdm(samples, desc=desc, disable=not verbose)):
        if len(seq) == 0:
            continue

        # --- Convert labels to binary if they are raw strings ---
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

        # Last-timestep labels (for scatter-plot dominant-label colouring)
        final_label_idx = len(input_seq) - 1
        final_labels.append(labels_binary_np[final_label_idx])

        # Union of all labels that appeared anywhere in this case
        stacked = np.stack(labels_binary_np, axis=0)        # (T, n_labels)
        ever_seen = (stacked.max(axis=0) > 0).astype(int)   # (n_labels,)
        all_case_labels.append(ever_seen)

        first_proceeding = input_seq[0]
        metadata_list.append({
            "case_idx": case_idx,
            "defendant_id": first_proceeding.get("defendant_id", "unknown"),
            "case_id": first_proceeding.get("case_id",
                       first_proceeding.get("docket_id", f"case_{case_idx}")),
            "seq_len": len(seq),
            "court_type": first_proceeding.get("type", "unknown"),
        })

    vectors = np.vstack(all_vectors) if all_vectors else np.empty((0,))
    all_case_labels_matrix = np.stack(all_case_labels, axis=0) if all_case_labels else np.empty((0,))

    if verbose:
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
Step 5: Visualization of clusters and label distributions
"""

def _truncate_label(label: str, max_len: int = 35) -> str:
    return label if len(label) <= max_len else label[:max_len - 1] + "…"

def plot_pca_label_grid(
    projected: np.ndarray,
    final_labels: List[np.ndarray],
    label_names: List[str],
    pca,
    output_path=None,
    title=None,
    min_prevalence_pct: float = 1.0,
    max_cols: int = 5,
):
    """
    Small-multiples PCA scatter: one subplot per label.

    Each subplot shows ALL points in light gray, then overlays the points
    that have that specific label active in a bold color.  A sample with
    3 labels appears highlighted in 3 different subplots — no information
    is lost.

    This is the standard approach for multi-label embeddings
    (see scikit-learn's multilabel classification visualization example).

    Args:
        projected:          (N, 2) PCA-projected coordinates
        final_labels:       list of N binary label vectors, each shape (L,)
        label_names:        list of L label name strings
        pca:                fitted PCA object (for explained variance labels)
        min_prevalence_pct: skip labels present in < this % of samples
        max_cols:           max columns in the subplot grid
    """
    labels_arr = np.array([lbl for lbl in final_labels])  # (N, L)
    N, L = labels_arr.shape

    # Filter to labels with enough prevalence to be visible
    label_counts = labels_arr.sum(axis=0)
    prevalence = (label_counts / N) * 100
    active_indices = [i for i in range(L) if prevalence[i] >= min_prevalence_pct]

    if not active_indices:
        print("⚠️  No labels exceed the minimum prevalence threshold.")
        return

    n_plots = len(active_indices)
    n_cols = min(max_cols, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    evr = pca.explained_variance_ratio_

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        sharex=True, sharey=True,
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

        # Background: all points in light gray
        ax.scatter(
            projected[:, 0], projected[:, 1],
            c="lightgray", s=6, alpha=0.3, edgecolors="none", rasterized=True,
        )
        # Foreground: highlighted points that have this label
        ax.scatter(
            projected[mask, 0], projected[mask, 1],
            c=[cmap(plot_idx)], s=12, alpha=0.7,
            edgecolors="white", linewidths=0.2, rasterized=True,
        )

        short_name = _truncate_label(label_names[label_idx], 30)
        ax.set_title(f"{short_name}\n(n={n_active:,}, {pct:.1f}%)", fontsize=7, pad=3)
        ax.tick_params(labelsize=5)

    # Hide unused subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.supxlabel(f"PC-1  ({evr[0]*100:.1f}% var)", fontsize=10)
    fig.supylabel(f"PC-2  ({evr[1]*100:.1f}% var)", fontsize=10)
    fig.suptitle(
        title or "PCA Embeddings — Label Presence (one subplot per label)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"💾 Saved label grid to {output_path}")
    plt.show()
    plt.close(fig)

def plot_pca_by_clusters(
    projected, cluster_labels, pca,
    centroids_2d=None, output_path=None, title=None,
):
    """Scatter plot colored by cluster assignment.  Unchanged from original."""
    fig, ax = plt.subplots(figsize=(12, 9))
    unique_clusters = sorted(set(cluster_labels))
    n_clusters = len(unique_clusters)
    cmap = plt.get_cmap("Set2" if n_clusters <= 8 else "tab20", n_clusters)

    for idx, cluster in enumerate(unique_clusters):
        mask = cluster_labels == cluster
        ax.scatter(
            projected[mask, 0], projected[mask, 1],
            c=[cmap(idx)],
            label=f"Cluster {cluster}" if cluster != -1 else "Noise",
            s=25, alpha=0.65, edgecolors="w", linewidth=0.3,
        )
    if centroids_2d is not None:
        ax.scatter(
            centroids_2d[:, 0], centroids_2d[:, 1],
            c="black", marker="X", s=200, zorder=10,
            label="Centroids", edgecolors="white", linewidth=1.5,
        )
    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC-1  ({evr[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC-2  ({evr[1]*100:.1f}% var)")
    ax.set_title(
        title or "PCA of Pre-Classifier Vectors — Cluster Assignment",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"💾 Saved cluster PCA to {output_path}")
    plt.show()
    plt.close(fig)

def _build_multilabel_matrix(cluster_labels, final_labels, label_names):
    """
    Build (n_clusters × n_labels) matrix.  Each cell = count of samples
    in that cluster with the label active.  One sample can count in many
    columns — that's the whole point.
    """
    labels_arr = np.array([lbl for lbl in final_labels])  # (N, L)
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

def plot_multilabel_cluster_heatmap(
    cluster_labels,
    final_labels,
    label_names,
    output_path=None,
    title=None,
    min_prevalence_pct: float = 1.0,
):
    """
    Heatmap where cell = '% of cluster X samples that have label Y active'.

    Every active label in a sample contributes — no dominant-label lossy step.
    Cells show both the percentage and raw count for transparency.

    Args:
        min_prevalence_pct: drop columns where no cluster exceeds this %.
    """
    raw, pct, sizes, clusters = _build_multilabel_matrix(
        cluster_labels, final_labels, label_names
    )

    col_mask = pct.max(axis=0) >= min_prevalence_pct
    pct_f = pct[:, col_mask]
    raw_f = raw[:, col_mask]
    names_f = [_truncate_label(label_names[i], 30)
               for i in range(len(label_names)) if col_mask[i]]

    if pct_f.shape[1] == 0:
        print("⚠️  No labels exceed the minimum prevalence threshold.")
        return

    # Annotation: "pct%\n(count)"
    annot = np.empty_like(pct_f, dtype=object)
    for i in range(pct_f.shape[0]):
        for j in range(pct_f.shape[1]):
            p, n = pct_f[i, j], raw_f[i, j]
            if p >= 1.0:
                annot[i, j] = f"{p:.0f}%\n({n})"
            elif n > 0:
                annot[i, j] = f"<1%\n({n})"
            else:
                annot[i, j] = ""

    row_labels = [f"Cluster {c}\n(n={s:,})" for c, s in zip(clusters, sizes)]

    fig, ax = plt.subplots(
        figsize=(max(12, len(names_f) * 0.75), max(5, len(clusters) * 0.9))
    )
    sns.heatmap(
        pct_f,
        annot=annot, fmt="",
        cmap="YlOrRd",
        xticklabels=names_f,
        yticklabels=row_labels,
        ax=ax,
        linewidths=0.5, linecolor="white",
        vmin=0, vmax=100,
        cbar_kws={"label": "% of cluster samples with label active"},
    )
    ax.set_title(
        title or "Multi-Label Cluster Composition",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Label", fontsize=10)
    ax.set_ylabel("Cluster", fontsize=10)
    plt.xticks(rotation=55, ha="right", fontsize=7)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"💾 Saved multi-label heatmap to {output_path}")
    plt.show()
    plt.close(fig)

    # Console summary
    print("\n📊 MULTI-LABEL CLUSTER SUMMARY")
    print("-" * 70)
    for i, (cl, sz) in enumerate(zip(clusters, sizes)):
        top_idx = np.argsort(pct[i])[::-1][:5]
        top_str = ", ".join(
            f"{_truncate_label(label_names[j], 25)} ({pct[i, j]:.0f}%)"
            for j in top_idx if pct[i, j] > 0
        )
        print(f"  Cluster {cl} ({sz:,} samples): {top_str}")


# ── 6d. Hierarchically-clustered label columns ──────────────────────────────

def plot_cluster_label_clustermap(
    cluster_labels,
    final_labels,
    label_names,
    output_path=None,
    title=None,
    min_prevalence_pct: float = 1.0,
):
    """
    seaborn.clustermap with hierarchically clustered label COLUMNS.
    Groups co-occurring labels automatically. Rows stay in cluster order.
    """
    raw, pct, sizes, clusters = _build_multilabel_matrix(
        cluster_labels, final_labels, label_names
    )

    col_mask = pct.max(axis=0) >= min_prevalence_pct
    pct_f = pct[:, col_mask]
    names_f = [_truncate_label(label_names[i], 30)
               for i in range(len(label_names)) if col_mask[i]]

    if pct_f.shape[1] == 0:
        print("⚠️  No labels exceed the minimum prevalence threshold.")
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
        vmin=0, vmax=100,
        annot=True, fmt=".0f",
        linewidths=0.5, linecolor="white",
        figsize=(max(12, len(names_f) * 0.7), max(5, len(clusters) * 0.85)),
        dendrogram_ratio=(0.0, 0.12),
        cbar_pos=(0.02, 0.82, 0.03, 0.15),
        cbar_kws={"label": "% prevalence"},
    )
    g.ax_heatmap.set_xlabel("Label (hierarchically clustered)", fontsize=10)
    g.ax_heatmap.set_ylabel("Cluster", fontsize=10)
    g.ax_heatmap.set_title(
        title or "Cluster × Label — Hierarchically Clustered Labels",
        fontsize=13, fontweight="bold", pad=15,
    )
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=55, ha="right", fontsize=7)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8)

    if output_path:
        g.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"💾 Saved clustermap to {output_path}")
    plt.show()
    plt.close()

def plot_cluster_label_stacked_bars(
    cluster_labels,
    final_labels,
    label_names,
    output_path=None,
    title=None,
    top_n_labels: int = 10,
):
    """
    Horizontal stacked bar chart: top-N labels per cluster.
    Quick visual when the full heatmap is too dense to read at a glance.
    """
    raw, pct, sizes, clusters = _build_multilabel_matrix(
        cluster_labels, final_labels, label_names
    )

    total_per_label = raw.sum(axis=0)
    top_idx = np.argsort(total_per_label)[::-1][:top_n_labels]
    pct_top = pct[:, top_idx]
    top_names = [_truncate_label(label_names[i], 30) for i in top_idx]
    row_labels = [f"Cluster {c}\n(n={s:,})" for c, s in zip(clusters, sizes)]

    fig, ax = plt.subplots(figsize=(14, max(4, len(clusters) * 0.8)))
    cmap = plt.get_cmap("tab20" if top_n_labels <= 20 else "nipy_spectral", top_n_labels)

    y_pos = np.arange(len(clusters))
    left = np.zeros(len(clusters))

    for j in range(len(top_idx)):
        bars = ax.barh(
            y_pos, pct_top[:, j], left=left,
            color=cmap(j), edgecolor="white", linewidth=0.3,
            label=top_names[j],
        )
        for i, bar in enumerate(bars):
            w = bar.get_width()
            if w > 5:
                ax.text(
                    left[i] + w / 2, bar.get_y() + bar.get_height() / 2,
                    f"{w:.0f}%", ha="center", va="center",
                    fontsize=6, fontweight="bold", color="white",
                )
        left += pct_top[:, j]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Label prevalence (%, not mutually exclusive)", fontsize=10)
    ax.set_title(
        title or f"Cluster Composition — Top {top_n_labels} Labels",
        fontsize=13, fontweight="bold",
    )
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
        frameon=True, framealpha=0.9,
    )
    ax.set_xlim(0, None)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"💾 Saved stacked bars to {output_path}")
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
    ax1.set_xlabel("K"); ax1.set_ylabel("Inertia"); ax1.set_title("Elbow Curve")
    ax1.grid(True, alpha=0.3)
    ax2.plot(list(range_of_k), silhouettes, "s-", color="coral")
    ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette"); ax2.set_title("Silhouette vs. K")
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Choosing the Number of Clusters", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"💾 Saved: {output_path}")
    plt.show()
    plt.close(fig)


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
):
    """
    K-fold embedding analysis: extract per fold, analyze combined.

    Flow:
        1. For each fold: load model → extract pre-classifier vectors + labels
        2. Combine all vectors, labels, metadata across folds
        3. PCA on combined vectors (one shared projection space)
        4. Cluster on combined projected data
        5. All plots on combined data only
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_folds = cv_runner.n_splits

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Extract vectors from each fold
    # ══════════════════════════════════════════════════════════════════════
    all_vectors = []
    all_final_labels = []
    all_metadata = []

    for fold_idx in range(n_folds):
        print(f"\n{'─' * 70}")
        print(f"  Extracting embeddings from Fold {fold_idx + 1} / {n_folds}")
        print(f"{'─' * 70}")

        fold_model = cv_runner.load_fold_model(fold_idx)
        fold_model.eval()
        fold_model.to(device)

        val_data, _ = cv_runner.get_fold_val_data(fold_idx)

        vectors, final_labels, _, metadata = extract_all_pre_classifier_vectors(
            fold_model, val_data, label_processor, device,
            max_samples=max_samples_per_fold,
            use_teacher_forcing=use_teacher_forcing,
            verbose=verbose,
        )

        # Tag each sample with its fold index
        for m in metadata:
            m["fold_idx"] = fold_idx + 1  # 1-indexed for display
        
        vectors = StandardScaler().fit_transform(vectors)
        vectors = normalize(vectors, norm="l2")

        all_vectors.append(vectors)
        all_final_labels.extend(final_labels)
        all_metadata.extend(metadata)

        print(f"  Fold {fold_idx + 1}: extracted {vectors.shape[0]} vectors "
              f"(dim={vectors.shape[1]})")

        # Free GPU memory
        del fold_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Combine
    # ══════════════════════════════════════════════════════════════════════
    vectors_combined = np.vstack(all_vectors)
    N = vectors_combined.shape[0]
    fold_ids = np.array([m["fold_idx"] for m in all_metadata])

    print(f"\n{'=' * 70}")
    print(f"COMBINED: {N:,} vectors × {vectors_combined.shape[1]} dims "
          f"across {n_folds} folds")
    print(f"{'=' * 70}")

    # Quick sanity check: fold distribution
    for f in sorted(set(fold_ids)):
        n = (fold_ids == f).sum()
        print(f"  Fold {int(f)}: {n:,} samples ({n/N*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: PCA (one shared space)
    # ══════════════════════════════════════════════════════════════════════
    projected, scaler, pca = reduce_vectors_with_pca(
        vectors_combined, n_components=2, standardize=True,
        random_state=random_state,
    )

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 4: Clustering (one pass on combined data)
    # ══════════════════════════════════════════════════════════════════════
    if show_elbow:
        print("\n📊 Elbow / Silhouette analysis …")
        plot_elbow(projected, range_of_k=range(2, 13),
                   output_path=output_dir / "elbow_silhouette.png")

    print(f"\n🔶 KMeans with K={n_clusters} …")
    km_labels, km_model = run_kmeans(
        projected, nclusters=n_clusters, random_state=random_state,
    )
    km_metrics = evaluate_clustering(projected, km_labels, true_labels_flat=None)
    print(f"   KMeans metrics: {km_metrics}")
    centroids_2d = km_model.cluster_centers_

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 5: All plots (combined data only)
    # ══════════════════════════════════════════════════════════════════════

    # 5a. Small-multiples PCA — one subplot per label
    plot_pca_label_grid(
        projected, all_final_labels, label_processor.label_names, pca,
        output_path=output_dir / "pca_label_grid.png",
        title="K-Fold Combined Embeddings — Label Presence",
    )

    # 5b. PCA colored by cluster
    plot_pca_by_clusters(
        projected, km_labels, pca,
        centroids_2d=centroids_2d,
        output_path=output_dir / "pca_by_kmeans_cluster.png",
        title="K-Fold Combined Embeddings — KMeans Clusters",
    )

    # 5c. PCA colored by fold (sanity check — should be well-mixed)
    _plot_pca_by_fold(
        projected, fold_ids, pca, n_folds,
        output_path=output_dir / "pca_by_fold.png",
    )

    # 5d. Multi-label cluster heatmap
    plot_multilabel_cluster_heatmap(
        km_labels, all_final_labels, label_processor.label_names,
        output_path=output_dir / "multilabel_cluster_heatmap.png",
        title="K-Fold Combined: Multi-Label Cluster Composition",
    )

    # 5e. Hierarchically-clustered clustermap
    plot_cluster_label_clustermap(
        km_labels, all_final_labels, label_processor.label_names,
        output_path=output_dir / "cluster_label_clustermap.png",
        title="K-Fold Combined: Cluster × Label (Hierarchically Clustered)",
    )

    # 5f. Stacked bar chart
    plot_cluster_label_stacked_bars(
        km_labels, all_final_labels, label_processor.label_names,
        output_path=output_dir / "cluster_label_stacked_bars.png",
        title="K-Fold Combined: Cluster Composition",
    )

    # Optional DBSCAN
    db_results = {}
    if run_dbscan:
        print(f"\n🔷 DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples}) …")
        db_labels, db_model = run_dbscan_func(
            projected, eps=dbscan_eps, min_samples=dbscan_min_samples,
        )
        db_metrics = evaluate_clustering(projected, db_labels, true_labels_flat=None)
        print(f"   DBSCAN metrics: {db_metrics}")

        plot_pca_by_clusters(
            projected, db_labels, pca,
            output_path=output_dir / "pca_by_dbscan_cluster.png",
            title="K-Fold Combined — DBSCAN Clusters",
        )
        plot_multilabel_cluster_heatmap(
            db_labels, all_final_labels, label_processor.label_names,
            output_path=output_dir / "dbscan_multilabel_heatmap.png",
            title="K-Fold DBSCAN: Multi-Label Cluster Composition",
        )
        db_results = {
            "dbscan_labels": db_labels,
            "dbscan_model": db_model,
            "dbscan_metrics": db_metrics,
        }

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 6: Save results
    # ══════════════════════════════════════════════════════════════════════
    df = pd.DataFrame(all_metadata)
    df["pc1"] = projected[:, 0]
    df["pc2"] = projected[:, 1]
    df["kmeans_cluster"] = km_labels
    if run_dbscan:
        df["dbscan_cluster"] = db_labels

    csv_path = output_dir / "embedding_clustering_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results CSV: {csv_path}")

    # ── Console summary ──
    labels_arr = np.array([lbl for lbl in all_final_labels])
    print("\n" + "=" * 70)
    print("CLUSTER COMPOSITION SUMMARY (KMeans)")
    print("=" * 70)
    for cl in sorted(set(km_labels)):
        mask = km_labels == cl
        n = mask.sum()
        label_sums = labels_arr[mask].sum(axis=0)
        top_indices = np.argsort(label_sums)[::-1][:5]
        top_str = ", ".join(
            f"{label_processor.label_names[j][:25]} ({int(label_sums[j])})"
            for j in top_indices if label_sums[j] > 0
        )
        fold_dist = Counter(fold_ids[mask])
        fold_str = ", ".join(f"F{int(f)}:{c}" for f, c in sorted(fold_dist.items()))
        print(f"  Cluster {cl} — {n} samples — folds: [{fold_str}]")
        print(f"      top labels: {top_str}")

    print(f"\n✅ K-Fold Embedding Analysis Complete. Outputs: {output_dir}")

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
        "fold_ids": fold_ids,
        "dataframe": df,
        **db_results,
    }
    return results


def _plot_pca_by_fold(
    projected: np.ndarray,
    fold_ids: np.ndarray,
    pca: PCA,
    n_folds: int,
    output_path: Optional[Path] = None,
):
    """PCA scatter colored by fold.  Sanity check that folds are well-mixed."""
    fig, ax = plt.subplots(figsize=(12, 9))
    cmap = plt.get_cmap("Set1", n_folds)

    for fold in sorted(set(fold_ids)):
        mask = fold_ids == fold
        ax.scatter(
            projected[mask, 0], projected[mask, 1],
            c=[cmap(int(fold) - 1)],
            label=f"Fold {int(fold)}",
            s=20, alpha=0.55,
            edgecolors="w", linewidths=0.2,
        )

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC-1  ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC-2  ({ev[1]*100:.1f}% var)")
    ax.set_title("Combined Embeddings — Coloured by Fold", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"💾 Saved: {output_path}")
    plt.show()
    plt.close(fig)

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
):
    """
    Embedding cluster analysis for train/val split (single model).

    Flow:
        1. Extract pre-classifier vectors from model on the given data
        2. PCA → 2D
        3. Cluster (KMeans, optionally DBSCAN)
        4. All plots: label grid, cluster PCA, heatmap, clustermap, stacked bars
        5. Save CSV
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Extract pre-classifier vectors
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("EMBEDDING CLUSTERING ANALYSIS")
    print("=" * 70)

    vectors, final_labels, metadata = extract_all_pre_classifier_vectors(
        model, data, label_processor, device,
        max_samples=max_samples,
        use_teacher_forcing=use_teacher_forcing,
        verbose=verbose,
    )

    N = vectors.shape[0]
    print(f"\nExtracted {N:,} vectors (dim={vectors.shape[1]})")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: PCA → 2D
    # ══════════════════════════════════════════════════════════════════════
    projected, scaler, pca = reduce_vectors_with_pca(
        vectors, n_components=2, standardize=True, random_state=random_state,
    )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Clustering
    # ══════════════════════════════════════════════════════════════════════
    if show_elbow:
        print("\n📊 Elbow / Silhouette analysis …")
        plot_elbow(
            projected, range_of_k=range(2, 13),
            output_path=output_dir / "elbow_silhouette.png",
        )

    print(f"\n🔶 KMeans with K={n_clusters} …")
    km_labels, km_model = run_kmeans(
        projected, nclusters=n_clusters, random_state=random_state,
    )
    km_metrics = evaluate_clustering(projected, km_labels, true_labels_flat=None)
    print(f"   KMeans metrics: {km_metrics}")
    centroids_2d = km_model.cluster_centers_

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Plots
    # ══════════════════════════════════════════════════════════════════════

    # 4a. Small-multiples PCA — one subplot per label
    plot_pca_label_grid(
        projected, final_labels, label_processor.label_names, pca,
        output_path=output_dir / "pca_label_grid.png",
    )

    # 4b. PCA colored by cluster
    plot_pca_by_clusters(
        projected, km_labels, pca,
        centroids_2d=centroids_2d,
        output_path=output_dir / "pca_by_kmeans_cluster.png",
    )

    # 4c. Multi-label cluster heatmap
    plot_multilabel_cluster_heatmap(
        km_labels, final_labels, label_processor.label_names,
        output_path=output_dir / "multilabel_cluster_heatmap.png",
    )

    # 4d. Hierarchically-clustered clustermap
    plot_cluster_label_clustermap(
        km_labels, final_labels, label_processor.label_names,
        output_path=output_dir / "cluster_label_clustermap.png",
    )

    # 4e. Stacked bar chart
    plot_cluster_label_stacked_bars(
        km_labels, final_labels, label_processor.label_names,
        output_path=output_dir / "cluster_label_stacked_bars.png",
    )

    # Optional DBSCAN
    db_results = {}
    if run_dbscan:
        print(f"\n🔷 DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples}) …")
        db_labels, db_model = run_dbscan_func(
            projected, eps=dbscan_eps, min_samples=dbscan_min_samples,
        )
        db_metrics = evaluate_clustering(projected, db_labels, true_labels_flat=None)
        print(f"   DBSCAN metrics: {db_metrics}")

        plot_pca_by_clusters(
            projected, db_labels, pca,
            output_path=output_dir / "pca_by_dbscan_cluster.png",
            title="Pre-Classifier Embeddings (PCA) — DBSCAN Clusters",
        )
        plot_multilabel_cluster_heatmap(
            db_labels, final_labels, label_processor.label_names,
            output_path=output_dir / "dbscan_multilabel_heatmap.png",
            title="DBSCAN: Multi-Label Cluster Composition",
        )
        db_results = {
            "dbscan_labels": db_labels,
            "dbscan_model": db_model,
            "dbscan_metrics": db_metrics,
        }

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: Save results
    # ══════════════════════════════════════════════════════════════════════
    df = pd.DataFrame(metadata)
    df["pc1"] = projected[:, 0]
    df["pc2"] = projected[:, 1]
    df["kmeans_cluster"] = km_labels
    if run_dbscan:
        df["dbscan_cluster"] = db_labels

    csv_path = output_dir / "embedding_clustering_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results CSV: {csv_path}")

    # ── Console summary ──
    labels_arr = np.array([lbl for lbl in final_labels])
    print("\n" + "=" * 70)
    print("CLUSTER COMPOSITION SUMMARY (KMeans)")
    print("=" * 70)
    for cl in sorted(set(km_labels)):
        mask = km_labels == cl
        n = mask.sum()
        label_sums = labels_arr[mask].sum(axis=0)
        top_indices = np.argsort(label_sums)[::-1][:5]
        top_str = ", ".join(
            f"{label_processor.label_names[j][:25]} ({int(label_sums[j])})"
            for j in top_indices if label_sums[j] > 0
        )
        print(f"  Cluster {cl} — {n} samples — top labels: {top_str}")

    print(f"\n✅ Analysis complete. Outputs: {output_dir}")

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
        "dataframe": df,
        **db_results,
    }
    return results


