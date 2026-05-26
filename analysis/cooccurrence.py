# Co-occurence 

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from config import get_model_config

## For All Proceedings
def export_cooccurrence_from_valid_completed_cases(
    valid_completed_defendant_cases,
    court_data,
    cached_outcome_labels_v2,
    label_names,
    output_path,
):
    model_configuration = get_model_config()
    mag_df = court_data.get("mag")
    cdc_df = court_data.get("cdc")

    def valid_value(x):
        return x not in ("", None) and not pd.isna(x)

    mag_nums_set = set(case[0] for case in valid_completed_defendant_cases if valid_value(case[0]))
    case_ids_set = set(case[1] for case in valid_completed_defendant_cases if valid_value(case[1]))

    mag_ids = set(
        mag_df.loc[mag_df["mag_num"].astype(str).isin(map(str, mag_nums_set)), "proceeding_id"]
        .astype(str).tolist()
    ) if mag_df is not None else set()

    cdc_ids = set(
        cdc_df.loc[cdc_df["case_id"].astype(str).isin(map(str, case_ids_set)), "proceeding_id"]
        .astype(str).tolist()
    ) if cdc_df is not None else set()

    proceeding_ids = mag_ids | cdc_ids
    print(f"Including {len(proceeding_ids)} proceedings from valid+completed cases.")

    n = len(label_names)
    label2idx = {name: i for i, name in enumerate(label_names)}

    cooccur = np.zeros((n, n), dtype=int)
    alone = np.zeros(n, dtype=int)
    totals = np.zeros(n, dtype=int)

    for pid in proceeding_ids:
        labels = cached_outcome_labels_v2.get(str(pid), [])
        active = [label2idx[l] for l in labels if l in label2idx and l not in model_configuration.EXCLUDED_LABELS]
        if not active:
            continue
        if len(active) == 1:
            i = active[0]
            alone[i] += 1
            totals[i] += 1
        else:
            for i in active:
                totals[i] += 1
                for j in active:
                    if i != j:
                        cooccur[i, j] += 1

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for i, lname in enumerate(label_names):
            # exclude self column
            other_labels = [lab for j, lab in enumerate(label_names) if j != i]

            # --- Counts row ---
            row_counts = [[lname, int(alone[i])] + [int(cooccur[i, j]) for j in range(n) if j != i]]
            df_counts = pd.DataFrame(row_counts, columns=["Label", "ALONE"] + other_labels)

            # --- Percentages row ---
            t = int(totals[i])
            if t > 0:
                row_pct = [[lname, f"{(alone[i]/t)*100:.1f}%"] +
                           [f"{(cooccur[i, j]/t)*100:.1f}%" for j in range(n) if j != i]]
            else:
                row_pct = [[lname] + ["0.0%"] * (len(other_labels) + 1)]
            df_percentages = pd.DataFrame(row_pct, columns=["Label", "ALONE"] + other_labels)

            # --- Write two sheets ---
            sheet_counts = f"{lname[:23]}_Counts"
            sheet_pct = f"{lname[:23]}_Pct"

            df_counts.to_excel(writer, sheet_name=sheet_counts, index=False)
            df_percentages.to_excel(writer, sheet_name=sheet_pct, index=False)

            # format sheets
            for sheet in [sheet_counts, sheet_pct]:
                ws = writer.sheets[sheet]
                ws.freeze_panes(1, 1)
                for c in range(len(other_labels) + 2):  # Label + ALONE + others
                    ws.set_column(c, c, 15 if c == 0 else 12)

    print(f"Saved per-label counts & percentages (excluding self) to {output_path}")
## For Chosen Sequence Proceedings
def create_cooccurrence_matrix_excel(sequences, labels_list, label_names, output_path):
    """
    Create Excel file where each label gets two sheets:
      - <LABEL>_Counts : row of co-occurrence counts (excluding self column)
      - <LABEL>_Pct    : row of co-occurrence percentages (excluding self column)
    """

    print(f"Creating per-label co-occurrence sheets for {len(sequences)} sequences...")

    n_labels = len(label_names)
    cooccur_matrix = np.zeros((n_labels, n_labels), dtype=int)
    alone_counts = np.zeros(n_labels, dtype=int)
    total_counts = np.zeros(n_labels, dtype=int)

    # --- Build matrices ---
    for sequence, sequence_labels in zip(sequences, labels_list):
        for label_vec in sequence_labels:
            if isinstance(label_vec, torch.Tensor):
                label_vec = label_vec.cpu().numpy()
            elif isinstance(label_vec, list):
                label_vec = np.array(label_vec)

            active_indices = [i for i, v in enumerate(label_vec) if v == 1]

            if len(active_indices) == 1:
                i = active_indices[0]
                alone_counts[i] += 1
                total_counts[i] += 1
            elif len(active_indices) > 1:
                for i in active_indices:
                    total_counts[i] += 1
                    for j in active_indices:
                        if i != j:
                            cooccur_matrix[i, j] += 1

    # --- Write Excel ---
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for i, lname in enumerate(label_names):
            # other labels (exclude self)
            other_labels = [lab for j, lab in enumerate(label_names) if j != i]

            # Counts
            row_counts = [[lname, int(alone_counts[i])] +
                          [int(cooccur_matrix[i, j]) for j in range(n_labels) if j != i]]
            df_counts = pd.DataFrame(row_counts, columns=["Label", "ALONE"] + other_labels)

            # Percentages
            t = int(total_counts[i])
            if t > 0:
                row_pct = [[lname, f"{(alone_counts[i]/t)*100:.1f}%"] +
                           [f"{(cooccur_matrix[i, j]/t)*100:.1f}%" for j in range(n_labels) if j != i]]
            else:
                row_pct = [[lname] + ["0.0%"] * (len(other_labels) + 1)]
            df_pct = pd.DataFrame(row_pct, columns=["Label", "ALONE"] + other_labels)

            # Safe sheet names (<= 31 chars)
            sheet_counts = f"{lname[:23]}_Counts"
            sheet_pct = f"{lname[:23]}_Pct"

            df_counts.to_excel(writer, sheet_name=sheet_counts, index=False)
            df_pct.to_excel(writer, sheet_name=sheet_pct, index=False)

            # Formatting
            for sheet in [sheet_counts, sheet_pct]:
                ws = writer.sheets[sheet]
                ws.freeze_panes(1, 1)
                for c in range(len(other_labels) + 2):  # Label + ALONE + others
                    ws.set_column(c, c, 15 if c == 0 else 12)

    print(f"Saved per-label co-occurrence sheets to {output_path}")
    return