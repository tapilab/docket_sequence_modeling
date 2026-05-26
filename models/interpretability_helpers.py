from pathlib import Path
from typing import Dict, List, Optional
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from models.transformer_helpers import discretize_bond_amount, bond_bin_to_index
from training.teacher_forcing import build_teacher_forced_sequences

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from scipy import stats
import seaborn as sns


RACE_GRID = {'B': 0, 'W': 1, 'H': 2, 'A': 3, 'I': 4}
GENDER_GRID = {'M': 0, 'F': 1}

BOND_BINS_ORDERED = [
    (0, "set_at_0",           0),
    (1, "set_0_to_1000",      500),
    (2, "set_1000_to_5000",   2500),
    (3, "set_5000_to_10000",  7500),
    (4, "set_10000_to_25000", 15000),
    (5, "set_25000_to_50000", 35000),
    (6, "set_50000_to_100000",75000),
    (7, "set_over_100000",    150000),
]

# Map numerical value to bin label
def numerical_value_to_bin_label(value, bins) -> str:
    try:
        value = float(value)
    except (ValueError, TypeError):
        return str(value)  # Return original if not a number
    for label, low, high in bins:
        if low <= value < high:
            return label
    return bins[-1][0]  # Return last bin label if above all bins


# Bond Swap

def bond_cyclic_swap(bond_amount):
    bin_id = discretize_bond_amount(bond_amount)    
    if bin_id == -1:
        return {} 
    cur_idx = bin_id
    nxt_idx = (cur_idx + 1) % len(BOND_BINS_ORDERED)

    _, nxt_label, nxt_val = BOND_BINS_ORDERED[nxt_idx]

    return {nxt_label: nxt_val}

# Age Swap

def build_age_quartile_bins(sequences) -> list:
    ages = []
    for seq in sequences:
        for pdict in seq:
            age = pdict.get("age")
            if age is not None:
                try:
                    ages.append(float(age))
                except (ValueError, TypeError):
                    continue
    if not ages:
        return [
            ("young (0-25)", 0, 24.9),
            ("young_adult (25-40)", 25, 39.9),
            ("middle_aged (40-54)", 40, 54.9),
            ("older (55+)", 55, 100),
        ]
    ages = np.array(ages)
    q25,q50,q75 = np.percentile(ages, [25, 50, 75])
    bins = [
        (f"Q1 (<{q25:.0f})",            0,    q25),
        (f"Q2 ({q25:.0f}-{q50:.0f})",   q25,  q50),
        (f"Q3 ({q50:.0f}-{q75:.0f})",   q50,  q75),
        (f"Q4 ({q75:.0f}+)",            q75,  999),
    ]
    print(f"  Age quartile bins: {[b[0] for b in bins]}")
    return bins

def age_grid_swap(age, age_bins) -> dict:
    current_label = numerical_value_to_bin_label(age, age_bins)
    grid = {}
    for label, low, high in age_bins:
        if label == current_label:
            continue
        mid = (low + high) / 2 if high < 999 else low + 10
        grid[label] = mid
    return grid


# Prior case count swap
def build_prior_case_quartile_bins(sequences) -> list:
    all_counts = []
    non_zero_counts = []
    for seq in sequences:
        for pdict in seq:
            pcc = pdict.get("prior_case_count")
            if pcc is not None:
                try:
                    count = int(pcc)
                    all_counts.append(count)
                    if count > 0:
                        non_zero_counts.append(count)
                except (ValueError, TypeError):
                    continue
    if all_counts:
        n_total = len(all_counts)
        n_zero = sum(1 for c in all_counts if c == 0)
        percent_zero = (n_zero / n_total) * 100
        print(f"  Prior case count distribution: {percent_zero:.1f}% zero cases")
    
    if not all_counts:
        bins = [
            ("none (0)", 0, 0),
            ("low (1-2)", 1, 2),
            ("medium (3-5)", 3, 5),
            ("high (6+)", 6, 999),
        ]
        print(f"  Prior case count bins (default): {[b[0] for b in bins]}")
        return bins
    
    nonzero_counts = np.array(non_zero_counts)
    q50, q75 = np.percentile(nonzero_counts, [50, 75])
    q50, q75 = max(1, int(q50)), max(2, int(q75))
    bins = [
        ("none (0)", 0, 0),
        (f"low (1-{q50})", 1, q50),
        (f"medium ({q50+1}-{q75})", q50 + 1, q75),
        (f"high ({q75+1}+)", q75 + 1, 999),
    ]
    print(f"  Prior case count bins: {[b[0] for b in bins]}")
    return bins

def prior_case_count_grid_swap(prior_case_count, prior_case_count_bins) -> dict:
    current_label = numerical_value_to_bin_label(prior_case_count, prior_case_count_bins)
    grid = {}
    for label, low, high in prior_case_count_bins:
        if label == current_label:
            continue
        mid = (low + high) / 2 if high < 999 else low + 3
        grid[label] = mid
    return grid





class CounterfactualMagnitudeStudy: 
    def __init__(self, model, label_names: List[str], device: torch.device,
                 label_processor, sequences: list = None, age_bins: list = None, prior_case_bins: list = None):
        self.model            = model
        self.label_names      = label_names
        self.device           = device
        self.label_processor  = label_processor
        self.model.eval()

        seqs = sequences or []
        self.age_bins = age_bins or build_age_quartile_bins(seqs)
        self.prior_case_bins = prior_case_bins or build_prior_case_quartile_bins(seqs)

    def _build_teacher_seq(self, seq: list, labels_binary: list) -> list:
        teacher_seq, _ = build_teacher_forced_sequences(
            seq, labels_binary, self.label_processor, include_initial_state=True
        )
        return teacher_seq
 
    def _predict_all_steps(self, teacher_seq: list) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(teacher_seq)          # (1, T, num_labels)
        return torch.sigmoid(logits[0]).cpu().numpy() # (T, num_labels)
 
    def _swap_and_predict_all(self, teacher_seq: list, field: str,
                              new_value) -> np.ndarray:
        
        seq_copy = [{**pdict, field: new_value} for pdict in teacher_seq]
        return self._predict_all_steps(seq_copy)
 
    def analyse_sequence(self, seq: list, labels_raw,
                            label_processor) -> Dict[str, pd.DataFrame]:
            def _is_already_binary(lbl):
                if isinstance(lbl, np.ndarray):
                    return lbl.dtype != object and lbl.ndim == 1
                if isinstance(lbl, (list, tuple)) and len(lbl) > 0:
                    return isinstance(lbl[0], (int, float, np.integer, np.floating))
                return False

            if len(labels_raw) > 0 and _is_already_binary(labels_raw[0]):
                labels_binary = [
                    lbl.cpu().numpy() if isinstance(lbl, torch.Tensor) 
                    else np.array(lbl, dtype=float)
                    for lbl in labels_raw
                ]
            else:
                # Normalize: wrap bare strings in a list
                normalized = [
                    [step] if isinstance(step, str) 
                    else list(step)
                    for step in labels_raw
                ]
                labels_binary = list(label_processor.transform_sequence(normalized))
                labels_binary = [
                    lbl.cpu().numpy() if isinstance(lbl, torch.Tensor) 
                    else np.array(lbl, dtype=float)
                    for lbl in labels_binary
                ]

            teacher_seq  = self._build_teacher_seq(seq, labels_binary)
            T            = len(teacher_seq)
            baseline_all = self._predict_all_steps(teacher_seq)  # (T, num_labels)
            pdict_0      = teacher_seq[0]                         # defendant info
            
            results: Dict[str, list] = {
                feat: [] for feat in
                ["race", "gender", "age", "original_bond", "prior_case_count"]
            }

            # Reverse lookup maps: value → label string
            RACE_ID_TO_LABEL   = {v: k for k, v in RACE_GRID.items()}
            RACE_ID_TO_LABEL[5] = "Unknown"
            GENDER_ID_TO_LABEL  = {v: k for k, v in GENDER_GRID.items()}
            GENDER_ID_TO_LABEL[2] = "Unknown"

            # ── Defendant-level features: swap once, measure across all steps ──────
            def record_case_swap(feat_name, feature_key, value_grid, original_val,
                                id_to_label=None):
                orig_label = id_to_label.get(original_val, str(original_val)) \
                            if id_to_label else str(original_val)
                for swap_label, swap_val in value_grid.items():
                    if swap_val == original_val:
                        continue
                    swapped_all = self._swap_and_predict_all(
                        teacher_seq, feature_key, swap_val)  # (T, num_labels)
                    for proc_idx in range(T):
                        for i, label in enumerate(self.label_names):
                            delta = float(swapped_all[proc_idx, i] - baseline_all[proc_idx, i])
                            results[feat_name].append({
                                "proceeding_idx": proc_idx,
                                "original_value": orig_label,
                                "swapped_value":  swap_label,
                                "label":          label,
                                "delta_prob":     delta,
                                "abs_delta":      abs(delta),
                            })

            record_case_swap("race",   "race_id",   RACE_GRID,
                            pdict_0.get("race_id", 5),   id_to_label=RACE_ID_TO_LABEL)
            record_case_swap("gender", "gender_id", GENDER_GRID,
                            pdict_0.get("gender_id", 2), id_to_label=GENDER_ID_TO_LABEL)

            actual_age = pdict_0.get("age", 25) or 25
            age_original_label  = numerical_value_to_bin_label(actual_age, self.age_bins)
            record_case_swap("age", "age", 
                            age_grid_swap(actual_age, self.age_bins),
                            age_original_label)

            # ── Proceeding-level features: swap per step ──────────────────────────
            def record_step_swap(feat_name, feature_key, proc_idx, value_grid, original_label):
                for swap_label, swap_val in value_grid.items():
                    if swap_label == original_label:
                        continue
                    seq_copy = list(teacher_seq)
                    seq_copy[proc_idx] = dict(teacher_seq[proc_idx])
                    seq_copy[proc_idx][feature_key] = swap_val
                    swapped_all = self._predict_all_steps(seq_copy)
                    for i, label in enumerate(self.label_names):
                        delta = float(swapped_all[proc_idx, i] - baseline_all[proc_idx, i])
                        results[feat_name].append({
                            "proceeding_idx": proc_idx,
                            "original_value": original_label,
                            "swapped_value":  swap_label,
                            "label":          label,
                            "delta_prob":     delta,
                            "abs_delta":      abs(delta),
                        })

            for proc_idx in range(T):
                pdict = teacher_seq[proc_idx]

                # ── original_bond ─────────────────────────────────────────────────
                raw_bond  = pdict.get("original_bond")
                bond_swap = bond_cyclic_swap(raw_bond)
                if bond_swap:
                    bin_id    = discretize_bond_amount(raw_bond)
                    cur_label = BOND_BINS_ORDERED[bin_id][1] if bin_id >= 0 else "not_set"
                    record_step_swap("original_bond", "original_bond",
                                    proc_idx, bond_swap, cur_label)

                # ── prior_case_count ──────────────────────────────────────────────
                prior_case_count_value   = pdict.get("prior_case_count", 0)
                prior_case_count_label   = numerical_value_to_bin_label(prior_case_count_value, self.prior_case_bins)
                pcc_swap_grid            = prior_case_count_grid_swap(prior_case_count_value, self.prior_case_bins)
                record_step_swap("prior_case_count", "prior_case_count",
                                proc_idx, pcc_swap_grid, prior_case_count_label)
            return {feat: pd.DataFrame(rows) for feat, rows in results.items()}
 
    # ──────────────────────────────────────────────────────────────────────────
    def run(self, sequences: list, label_sequences: list, max_samples: int = 200, output_dir: Optional[Path] = None, n_workers: int = 4, parallel: bool = True, tqdm_position: int = 0) -> Dict[str, pd.DataFrame]:
        all_records: Dict[str, list] = {f: [] for f in
            ["race", "gender", "age", "original_bond", "prior_case_count"]}

        n = min(max_samples, len(sequences))

        def _process(idx):
            seq = sequences[idx]
            if isinstance(seq, tuple):
                seq = seq[0]
            try:
                result = self.analyse_sequence(seq, label_sequences[idx], self.label_processor)
                return idx, result
            except Exception as e:
                return idx, e

        def _collect(idx, result):
            if isinstance(result, Exception):
                print(f"  ⚠️  Skipping sample {idx}: {result}")
                return
            for feat, df in result.items():
                if not df.empty:
                    df["sample_idx"] = idx
                    all_records[feat].append(df)

        if parallel:
            lock = threading.Lock()
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_process, idx): idx for idx in range(n)}
                for future in tqdm(as_completed(futures), total=n,
                                desc=f"Counterfactual study (parallel) [GPU {tqdm_position}]",
                                position=tqdm_position, leave=True):
                    idx, result = future.result()
                    with lock:
                        _collect(idx, result)
        else:
            for idx in tqdm(range(n),
                            desc=f"Counterfactual study [GPU {tqdm_position}]",
                            position=tqdm_position, leave=True):
                _, result = _process(idx)
                _collect(idx, result)

        # Aggregate per (swapped_value, label)
        aggregated: Dict[str, pd.DataFrame] = {}
        raw:        Dict[str, pd.DataFrame] = {}

        for feat, record_list in all_records.items():
            if not record_list:
                aggregated[feat] = pd.DataFrame()
                raw[feat]        = pd.DataFrame()
                continue

            combined  = pd.concat(record_list, ignore_index=True)
            raw[feat] = combined   # ← preserve original_value here

            agg = (combined
                .groupby(["original_value", "swapped_value", "label"])
                .agg(
                    mean_abs_delta=("abs_delta",  "mean"),
                    std_abs_delta =("abs_delta",  "std"),
                    mean_delta    =("delta_prob", "mean"),
                    std_delta     =("delta_prob", "std"),
                    n             =("abs_delta",  "count"),
                )
                .reset_index())
            agg["se_abs_delta"] = agg["std_abs_delta"] / np.sqrt(agg["n"])
            agg['se_delta']     = agg["std_delta"]    / np.sqrt(agg["n"])
            agg["ci95_half_width_abs_delta"]   = agg.apply(
                lambda r: stats.t.ppf(0.975, df=max(r["n"] - 1, 1)) * r["se_abs_delta"]
                          if r["n"] > 1 else np.nan, axis=1)
            agg["ci95_half_width_delta"] = agg.apply(
                lambda r: stats.t.ppf(0.975, df=max(r["n"] - 1, 1)) * r["se_delta"]
                          if r["n"] > 1 else np.nan, axis=1)
            agg["ci95_abs_delta"] = agg.apply(
                lambda r: f"{r['mean_abs_delta']:.4f} +/- {r['ci95_half_width_abs_delta']:.4f}"
                          if r["n"] > 1 else f"{r['mean_abs_delta']:.4f} (n=1)", axis=1)
            agg["ci95_delta"] = agg.apply(
                lambda r: f"{r['mean_delta']:.4f} +/- {r['ci95_half_width_delta']:.4f}"
                          if r["n"] > 1 else f"{r['mean_delta']:.4f} (n=1)", axis=1)
            aggregated[feat] = agg

        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            for feat, df in aggregated.items():
                if not df.empty:
                    df.to_excel(Path(output_dir) / f"counterfactual_{feat}.xlsx",
                                index=False)
            for feat, df in raw.items():
                if not df.empty:
                    df.to_pickle(Path(output_dir) / f"counterfactual_{feat}_raw.pkl")
            print(f"✅ Tables saved to {output_dir}")

        return aggregated, raw



def plot_counterfactual_results(
    aggregated: Dict[str, pd.DataFrame],
    raw: Optional[Dict[str, pd.DataFrame]] = None,
    output_dir: Optional[Path] = None,
    show: bool = True,
    top_n_labels: int = 8,
):
    out = Path(output_dir) if output_dir else Path(".")

    # ── Plot 1: Per-feature swapped-to bar chart (existing) ───────────────────
    for feat, df in aggregated.items():
        if df.empty:
            continue

        top_labels = (df.groupby("label")["mean_abs_delta"]
                        .mean()
                        .nlargest(top_n_labels)
                        .index.tolist())
        df_plot = df[df["label"].isin(top_labels)]

        pivot = df_plot.pivot_table(
            index="swapped_value", columns="label",
            values="mean_abs_delta", aggfunc="mean"
        ).fillna(0)

        fig, ax = plt.subplots(figsize=(
            max(10, len(pivot.columns) * 0.9),
            max(4,  len(pivot)         * 0.55),
        ))
        pivot.plot(kind="barh", ax=ax, alpha=0.85)

        ax.set_title(
            f"'{feat}' — Mean |ΔProbability| per swap\n"
            "(higher = prediction changes more when this value is substituted)",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Mean |ΔProbability|")
        ax.set_ylabel("Swapped-to value")
        ax.legend(title="Outcome Label", bbox_to_anchor=(1.01, 1), fontsize=8)
        plt.tight_layout()

        if output_dir:
            fig.savefig(out / f"counterfactual_{feat}.png",
                        dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ── Plot 2: CI error bar plot (swap-level) ─────────────────────────────────────
    for feat, df in aggregated.items():
        if df.empty or "ci95_half_width_abs_delta" not in df.columns:
            continue
        top_labels = (df.groupby("label")["mean_abs_delta"]
                        .mean()
                        .nlargest(top_n_labels)
                        .index.tolist())
        df_plot = df[df["label"].isin(top_labels)].copy()
        swap_vals = sorted(df_plot["swapped_value"].unique())
        y_pos = np.arange(len(swap_vals))
        n_labels = len(top_labels)
        fig, axes = plt.subplots(1, n_labels, figsize=(max(4 * n_labels, 12), max(4, len(swap_vals) * 0.55)), sharey=True)
        if n_labels == 1:
            axes = [axes]
        for ax, label in zip(axes, top_labels):
            df_label = (df_plot[df_plot["label"] == label]
                    .groupby("swapped_value")
                    .agg(mean_abs_delta=("mean_abs_delta", "mean"), 
                    ci95_half_width_abs_delta= ("ci95_half_width_abs_delta", "mean"))
                    .reindex(swap_vals))
            means = df_label["mean_abs_delta"].values
            half_widths = df_label["ci95_half_width_abs_delta"].fillna(0).values
            ax.barh(y_pos, means, xerr=half_widths, align="center", alpha=0.85, capsize=4, color="#54a0e0", ecolor="#2c3e50")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(swap_vals)
            ax.set_xlabel("Mean |ΔProbability|")
            ax.set_title(f"Label: {label}", fontsize=11, fontweight="bold")
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        fig.suptitle(
            f"'{feat}' — Mean |ΔProbability| with 95% CI\n"
            "by swapped-to value and outcome label",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        if output_dir:
            fig.savefig(out / f"counterfactual_{feat}_ci.png",
                        dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    # ── Plot 3 and 4: Before → After transition and transition CI ────────────
    if raw is not None:
        for feat, df_raw in raw.items():
            if df_raw.empty or "original_value" not in df_raw.columns:
                continue
            df_agg = aggregated.get(feat, pd.DataFrame())
            has_ci = not df_agg.empty and "ci95_half_width_abs_delta" in df_agg.columns
            
            top_labels= (df_agg.groupby("label")["mean_abs_delta"]
                            .mean()
                            .nlargest(top_n_labels)
                            .index.tolist())
            df_plot = df_raw[df_raw["label"].isin(top_labels)].copy()
            
            def _make_mean_and_ci_annotation_for_heatmap(pivot_mean, label_filter=None):
                if not has_ci:
                    return pivot_mean.applymap(lambda v: f"{v:.3f}")
                df_ci = df_agg.copy()
                if label_filter is not None:
                    df_ci = df_ci[df_ci["label"] == label_filter]
                ci_pivot = (df_ci.pivot_table(
                    index="original_value", columns="swapped_value",
                    values="ci95_half_width_abs_delta", aggfunc="mean")
                    .reindex(index=pivot_mean.index, columns=pivot_mean.columns)
                    .fillna(0))
                annotated_df = pd.DataFrame(index=pivot_mean.index, columns=pivot_mean.columns)
                for row in pivot_mean.index:
                    for col in pivot_mean.columns:
                        mean_val = pivot_mean.loc[row, col]
                        ci_half_width = ci_pivot.loc[row, col] if (row in ci_pivot.index and col in ci_pivot.columns) else 0
                        annotated_df.loc[row, col] = f"{mean_val:.3f} ± {ci_half_width:.3f}"
                return annotated_df
            
            for label in top_labels:
                df_label = df_plot[df_plot["label"] == label]
                if df_label.empty:
                    continue
                pivot = df_label.pivot_table(
                    index="original_value", columns="swapped_value",
                    values="abs_delta", aggfunc="mean").fillna(0)
                if pivot.empty:
                    continue
                annotation = _make_mean_and_ci_annotation_for_heatmap(pivot, label_filter=label)
                fig, ax = plt.subplots(figsize=(
                    max(6,  len(pivot.columns) * 0.9),
                    max(4,  len(pivot)         * 0.7),
                ))
                sns.heatmap(
                    pivot, annot=annotation, fmt="",  # Use preformatted annotations
                    cmap="YlOrRd", ax=ax,
                    annot_kws={"size":8},
                    cbar_kws={"label": "Mean |ΔProbability|"},
                )
                ax.set_title(f'{feat} — Before → After |ΔProbability|\nLabel: {label}', fontsize=13, fontweight="bold")
                ax.set_xlabel("Swapped-to value")
                ax.set_ylabel("Original value")
                plt.tight_layout()
                if output_dir:
                    safe_label = label.replace("/", "_").replace(" ", "_")
                    fig.savefig(out / f"counterfactual_{feat}_{safe_label}_transition_ci.png",
                                dpi=300, bbox_inches="tight")
                if show:
                    plt.show()
                else:
                    plt.close(fig)
            if has_ci and "ci95_half_width_delta" in df_agg.columns:
                df_ci_feat = df_agg[df_agg["label"].isin(top_labels)].copy()
                swap_vals     = sorted(df_ci_feat["swapped_value"].unique())
                original_vals = sorted(df_ci_feat["original_value"].unique())
                n_labels  = len(top_labels)
                cmap_ci   = plt.cm.get_cmap("tab10", len(swap_vals))
 
                # ── Single-column layout for a two-column research paper ──────────
                # IEEE / ACM single column = 3.5 in  |  Elsevier ≈ 90 mm ≈ 3.54 in
                # Arrange subplots in a 2-wide grid so the figure stays narrow but
                # tall rather than an unreadable 32-inch-wide strip.
                N_COLS   = 2
                n_rows   = int(np.ceil(n_labels / N_COLS))
                # Height per row: enough for the original-value y-ticks + swap bars
                ax_h     = max(1.0, len(original_vals) * 0.28 + len(swap_vals) * 0.18 + 0.35)
                fig, axes = plt.subplots(
                    n_rows, N_COLS,
                    figsize=(3.5, n_rows * ax_h + 0.45),
                    sharey=False,
                    constrained_layout=True,   # handles tight spacing at small size
                )
                axes_flat = np.array(axes).flatten()
 
                for idx, label in enumerate(top_labels):
                    ax = axes_flat[idx]
                    df_lbl = df_ci_feat[df_ci_feat["label"] == label]
                    y_base          = np.arange(len(original_vals))
                    number_of_swaps = len(swap_vals)
                    width           = 0.8 / max(number_of_swaps, 1)
 
                    for swap_index, swap_val in enumerate(swap_vals):
                        df_sv = (
                            df_lbl[df_lbl["swapped_value"] == swap_val]
                            .set_index("original_value")
                            .reindex(original_vals)
                        )
                        # ── signed ΔProbability (positive = prediction rises) ──
                        means       = df_sv["mean_delta"].fillna(0).values
                        half_widths = df_sv["ci95_half_width_delta"].fillna(0).values
                        offset = (swap_index - number_of_swaps / 2 + 0.5) * width
 
                        ax.barh(
                            y_base + offset, means,
                            xerr=half_widths,
                            height=width * 0.85,
                            align="center",
                            alpha=0.85,
                            capsize=1.5,
                            color=cmap_ci(swap_index),
                            ecolor="black",
                            linewidth=0,
                            label=swap_val,
                        )
 
                    ax.set_yticks(y_base)
                    ax.set_yticklabels(original_vals, fontsize=5)
                    ax.tick_params(axis="x", labelsize=5)
                    ax.set_xlabel("ΔP", fontsize=5)
                    # Zero-line so direction of change is immediately readable
                    ax.axvline(0, color="black", linestyle="-", linewidth=0.6, zorder=3)
                    # Truncate long label names so they fit inside a narrow column
                    short_lbl = (label[:22] + "…") if len(label) > 22 else label
                    ax.set_title(short_lbl, fontsize=5.5, fontweight="bold", pad=2)
 
                # One shared legend on the first panel only
                axes_flat[0].legend(
                    title="Swap→",
                    fontsize=4.5,
                    title_fontsize=5,
                    loc="lower right",
                    framealpha=0.7,
                    handlelength=1.0,
                    borderpad=0.4,
                )
 
                # Hide any unused grid cells (when n_labels is odd)
                for idx in range(n_labels, len(axes_flat)):
                    axes_flat[idx].set_visible(False)
 
                fig.suptitle(
                    f"{feat}  —  ΔP (signed) with 95% CI",
                    fontsize=6, fontweight="bold",
                )
                if output_dir:
                    fig.savefig(
                        out / f"counterfactual_{feat}_transition_ci.png",
                        dpi=300, bbox_inches="tight",
                    )
                if show:
                    plt.show()
                else:
                    plt.close(fig)
    # ── Plot 5: Overall feature sensitivity summary ──────────────────────────────────────
    summary_rows = [{"feature": feat, "mean_abs_delta": df["mean_abs_delta"].mean()} for feat, df in aggregated.items() if not df.empty]
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("mean_abs_delta", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        colors     = ["#e07b54", "#e0a354", "#54a0e0", "#e05454", "#54e09a", "#a054e0"] 
        ax.barh(summary_df["feature"], summary_df["mean_abs_delta"], color=colors[:len(summary_df)], alpha=0.85)
        ax.set_title(
            "Overall Feature Sensitivity\n"
            "Mean |ΔProbability| averaged over all swaps and labels",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Mean |ΔProbability|")
        plt.tight_layout()
        if output_dir:
            fig.savefig(out / "counterfactual_summary.png", dpi=300, bbox_inches="tight")
            summary_df.to_excel(out / "counterfactual_summary.xlsx", index=False)
        if show:
            plt.show()
        else:
            plt.close(fig)
        print("\n📊 Feature Sensitivity Summary:")
        print(summary_df.to_string(index=False))